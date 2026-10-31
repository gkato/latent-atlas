LA.chapter({
  id: 'ngram-mtp',
  part: 3,
  order: 16,
  title: 'Memória N-gram (PLE) e Multi-Token Prediction',
  navTitle: 'N-gram & MTP',
  subtitle: 'Uma tabela de 51 bilhões de parâmetros que custa quase zero FLOPs por token, e uma camada treinada para ver dois tokens adiante que vira rascunhador de graça: hash, primos, gate, estado, formatos, patches e o k das receitas.',
  minutes: 55,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['N-gram embedding', 'PLE', 'Engram', 'hash', 'SplitMix64', 'MTP', 'n_predict', 'speculative decoding', 'NVFP4', 'hf-overrides'],
  goals: [
    'Explicar por que uma memória de N-gramas dá <b>capacidade sem FLOPs</b> e por que o seu endereçamento determinístico permite prefetch e offload',
    'Calcular à mão os 16 endereços da PLE do Qwen (hash multiplicativo-XOR, SplitMix64, primos) e justificar primos distintos e várias cabeças',
    'Seguir a linha recuperada até o resíduo (gate por ramo, conv dilatada) e listar o estado por requisição que ela exige (EOS, chunked prefill, rollback)',
    'Entender o MTP (módulo, loss, onde mora em cada modelo) e derivar dos configs o <code>mtp.layers.48</code>, o formato da tabela e o <code>k=6</code> do DeepSeek',
  ],
  prereqs: ['hiperconexoes', 'mixture-of-experts'],
  summary: [
    'A memória de N-gramas (PLE) do Qwen3.8 é uma tabela de 320.001.536 × 160 = <b>51,2 B parâmetros</b>. Cada token lê 16 linhas (8 de bigrama, 8 de trigrama) e a busca não custa FLOPs: guardado ÷ lido ≈ 2×10⁷, contra 51 no MoE.',
    'Endereço: \\(h_2 = x_t m_0 \\oplus x_{t-1} m_1\\), \\(h_3 = h_2 \\oplus x_{t-2} m_2\\), índice = offset + \\(h \\bmod p\\), com 16 primos ≈ 20 M e multiplicadores ímpares gerados pela SplitMix64 (seed 1234). Como só depende dos ids, o endereço é conhecido antes do forward.',
    'Primos distintos tornam as cabeças independentes (com o mesmo módulo elas seriam clones) e usam todos os bits do hash. Dois N-gramas com hashes diferentes dividem no máximo 2 das 8 linhas da mesma ordem.',
    'A linha vira 4 chaves (uma por ramo do Gated Residual) e 1 valor. O gate \\(\\sigma(\\mathrm{sign}(z)\\sqrt{|z|})\\) compara a chave com o resíduo, uma conv causal depthwise (k=4, dilatação 3) acrescenta 12 tokens de contexto, e tudo é somado aos 4 ramos na camada de índice 1.',
    'A PLE tem estado por requisição (2 ids de contexto + 9 colunas de conv + colunas especulativas). Ela respeita o EOS, atravessa chunks de prefill e faz rollback especulativo. No vLLM é uma camada <code>MambaBase</code> <code>SHORT_CONV</code> com custom op.',
    'Carregar a tabela exigiu detectar o formato em <code>config_groups</code>, injetar <code>text_config.ple_embedding_dtype</code> via <code>--hf-overrides</code> e portar o dispatch misto (FP8: 47,7 GiB; NVFP4: 90 B/linha, 26,8 GiB). Sem isso a tabela vira BF16 de ≈102 GB e o load cai. <code>PLE_QUANT_OVERRIDE</code> está obsoleto.',
    'MTP: \\([\\mathrm{norm}(\\mathrm{emb}(x_{t+1}));\\ \\mathrm{norm}(h_t)]\\) → projeção → bloco → cabeça compartilhada prevê \\(x_{t+2}\\). GLM: camada 45 (k=2 de reserva). Qwen: 1 camada ×3 (<code>mtp.layers.48</code>, 24,5 → 52,1 tok/s). DeepSeek: n_predict = 3 → k = 6, o menor valor que passa na validação de reuso (o DSpark continua rascunhando o bloco num forward).',
  ],
  quiz: [
    { q: 'A tabela de N-gramas do Qwen guarda 51,2 B parâmetros. Quanto dela um token lê, e quanto a busca custa em FLOPs?', o: ['Todas as linhas, com ~100 GFLOP por token', '16 linhas de 160 valores (2.560 parâmetros); a busca em si não multiplica nada', '10 linhas de 2.560, como os experts do MoE', 'Depende do contexto: até 2.048 linhas, como a QSA'], a: 1, e: 'Cada uma das 16 cabeças (8 de bigrama, 8 de trigrama) aponta uma linha de 160 valores, e a busca é uma cópia (gather). As únicas multiplicações pesadas do módulo estão nas projeções de chave e valor (~65,5 MFLOP/token), cerca de 0,5% do custo do modelo; gate e conv são desprezíveis. Os experts são outra peça (10 de 512 por camada).' },
    { q: 'Por que as 8 cabeças de bigrama usam primos <i>diferentes</i> (20.000.003, 20.000.023, …) em vez de um mesmo tamanho?', o: ['Para caber em páginas de memória de tamanho potência de 2', 'Porque primos tornam o hash inversível', 'Porque as 8 cabeças da mesma ordem compartilham o mesmo hash \\(h_2\\) e só o módulo as diferencia. Com o mesmo módulo seriam clones, e primos também misturam todos os bits de \\(h\\)', 'Para que o gather na GPU fique coalescido'], a: 2, e: 'O hash \\(h_2\\) é calculado uma vez e reduzido módulo 8 primos diferentes. Com o mesmo módulo, quem colide numa cabeça colide em todas. Com um módulo potência de 2, só os bits baixos de \\(h\\) importariam. Com primos distintos, as cabeças ficam independentes, e pelo Teorema Chinês do Resto dois hashes diferentes dividem no máximo 2 das 8 linhas.' },
    { q: 'O checkpoint <code>nvidia/Qwen3.8-Flash-Next-NVFP4</code> na imagem day-0 <b>sem</b> o patch tenta criar uma embedding BF16 de ≈102 GB e cai. Qual é a causa?', o: ['O resolver só escolhia FP8 para a PLE se o checkpoint inteiro fosse <code>Fp8Config</code>. Este é <code>MIXED_PRECISION</code> e declara a PLE FP8 só em <code>config_groups</code>, sem <code>ple_embedding_dtype</code>', 'A tabela está corrompida no Hugging Face', 'O KV cache em BF16 ocupa a memória que a tabela precisaria', 'O vLLM não suporta embeddings com mais de 2³¹ linhas'], a: 0, e: 'Sem método de quantização, a PLE é criada em BF16 (320 M × 160 × 2 B ≈ 102 GB). A receita detecta o formato no <code>config_groups</code> (<code>detect_ple_dtype.py</code>), injeta <code>text_config.ple_embedding_dtype</code> via <code>--hf-overrides</code> e usa o dispatch portado do PR #53899.' },
    { q: 'Um prompt de 30 mil tokens é processado em 4 chunks de 8.192. O que a PLE precisa levar de um chunk para o seguinte?', o: ['Nada: a PLE é só uma embedding', 'As 16 linhas buscadas no chunk anterior', 'O KV cache das 12 camadas QSA', 'Os 2 últimos ids da requisição (contexto de N-grama) e as 9 últimas entradas da conv dilatada (conv_state)'], a: 3, e: 'O primeiro token do chunk seguinte forma bigrama e trigrama com tokens do chunk anterior, e a conv (k=4, dilatação 3) olha até 9 posições para trás. Os dois dados ficam no estado por requisição, que o vLLM gerencia como estado tipo Mamba.' },
    { q: 'O vLLM constrói a camada MTP do Qwen como <code>mtp.layers.48</code>. Por que isso quebrava o carregamento?', o: ['Porque 48 não é divisível por TP=2', 'Porque os metadados de quantização do checkpoint só listam <code>mtp.layers.0.*</code> e o casamento é por string exata. O MoE do MTP era criado sem quantização e faltava <code>w2_weight_scale_inv</code>', 'Porque a camada 48 não existe no checkpoint e é inventada pelo vLLM', 'Porque o MTP precisa da tabela PLE'], a: 1, e: 'O vLLM numera a camada MTP como <code>num_hidden_layers + i</code>. O <code>patch_checkpoint_config.py</code> cria aliases <code>.0.</code> → <code>.48.</code> em <code>quantized_layers</code> e <code>config_groups</code> e normaliza <code>FP8_PB_WO</code> → <code>FP8_BLOCK_SCALES</code>. O ramo MoE FP8 em blocos vem de outro patch.' },
    { q: 'No DeepSeek Vision-Exp, <code>num_nextn_predict_layers = 3</code> e o DSpark foi treinado com bloco 5. Por que a receita usa k = 6?', o: ['Porque k = 6 tem aceitação maior que k = 5', 'Porque o CUDA graph só captura múltiplos de 6', 'Porque a lane padrão aplica a regra do MTP (se k > n_predict, k tem de ser múltiplo de 3) e o launcher exige k ≥ 5; o menor valor possível é 6. O hotfix block-k libera k = 5', 'Porque o card oficial recomenda k = 6'], a: 2, e: 'A regra "MTP module reuse" faz sentido para módulos reexecutados em ciclo, mas para o DSpark é só uma validação: mesmo com k = 6 ele rascunha o bloco inteiro num forward, com uma posição de ruído além do bloco treinado. (9, 12… também passariam, ainda mais longe do bloco.) Com <code>DSPARK_ENABLE_DSPARK_BLOCK_K=1</code> e <code>MTP_NUM_TOKENS=5</code>, a aceitação por posição é a mesma e o passo fica mais barato (+2–10% em single-stream).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const onCol = () => (V.C().dark ? '#06140d' : '#ffffff');

    /* ============================================================ helpers */
    /** altura do stage por largura (wide/narrow podem ser números ou funções da largura); devolve set() para recalcular */
    function adaptH(fig, wide, narrow, bp = 620) {
      const st = fig.querySelector('.la-fig-stage');
      const val = (v) => (typeof v === 'function' ? v(st.clientWidth || fig.clientWidth) : v);
      const set = () => { const want = Math.round(fig.clientWidth < bp ? val(narrow) : val(wide)) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return set;
    }
    const fmtInt = (x) => (typeof x === 'bigint' ? x : Math.round(x)).toLocaleString('pt-BR');
    const fmt = (x, d = 1) => (+x).toFixed(d).replace('.', ',');
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function fitSize(g, s, maxW, size, font = 'sans', weight = 400, min = 11) { let z = size; while (z > min && tw(g, s, z, font, weight) > maxW) z -= 0.5; return z; }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const words = String(s).split(' ');
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      let line = '', yy = y, n = 0;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); yy += lh; n++; line = wd; g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); }
        else line = test;
      }
      if (line) { V.text(g, line, x, yy, o); n++; }
      return n;
    }
    /** caixa-personagem para um token */
    function tokBox(g, x, y, w, h, label, col, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: Math.min(9, h / 3), fill: o.fill || V.alpha(col, 0.14), stroke: o.stroke || col, lw: o.lw || 1.3 });
      if (o.dash) { g.save(); g.setLineDash([4, 3]); V.rr(g, x, y, w, h, Math.min(9, h / 3)); g.strokeStyle = o.stroke || col; g.lineWidth = 1.3; g.stroke(); g.restore(); }
      const eyes = o.eyes !== false && h >= 30;
      if (eyes) { const ey = y + 8, ex = x + w / 2; V.dot(g, ex - 5, ey, 1.7, o.eyeCol || col); V.dot(g, ex + 5, ey, 1.7, o.eyeCol || col); }
      const sz = fitSize(g, label, w - 6, o.size || 12, o.font || 'sans', 500);
      V.text(g, label, x + w / 2, y + h / 2 + (eyes ? 5 : 0.5), { size: sz, align: 'center', base: 'middle', color: o.color || C.ink, font: o.font || 'sans', weight: 500, maxW: w - 4 });
    }
    /** quebra s em linhas de até maxW (só mede) */
    function wrapLines(g, s, maxW, size = 12, font = 'sans', weight = 400) {
      g.font = V.font(size, font, weight);
      const out = []; let line = '';
      for (const wd of String(s).split(' ')) { const test = line ? line + ' ' + wd : wd; if (g.measureText(test).width > maxW && line) { out.push(line); line = wd; } else line = test; }
      if (line) out.push(line);
      return out;
    }
    const measureCtx = document.createElement('canvas').getContext('2d');
    /** vetor desenhado como tira de células */
    function strip(g, x, y, n, cw, ch, colorFn, o = {}) {
      const gap = o.gap == null ? 1.5 : o.gap;
      for (let i = 0; i < n; i++) { g.fillStyle = colorFn(i); V.rr(g, x + i * (cw + gap), y, cw, ch, Math.min(3, cw / 4)); g.fill(); }
      return n * cw + (n - 1) * gap;
    }
    const sig = (x) => 1 / (1 + Math.exp(-x));

    /* ============================================================ hash real */
    const MASK = (1n << 64n) - 1n, GAMMA = 0x9E3779B97F4A7C15n, SM1 = 0xBF58476D1CE4E5B9n, SM2 = 0x94D049BB133111EBn;
    const VOCAB = 248320n, EOS = 248044;
    const splitmix = (v) => { v = (v + GAMMA) & MASK; v = ((v ^ (v >> 30n)) * SM1) & MASK; v = ((v ^ (v >> 27n)) * SM2) & MASK; return (v ^ (v >> 31n)) & MASK; };
    const HALF = ((1n << 63n) - 1n) / VOCAB / 2n;
    const multipliers = (seed, layer = 0) => { const base = BigInt(seed) + 10007n * BigInt(layer); return [0n, 1n, 2n].map((i) => 2n * (splitmix(base + GAMMA * (i + 1n)) % HALF) + 1n); };
    const PRIMES = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171];
    const BIGP = PRIMES.map((p) => BigInt(p));
    const OFFS = []; { let o = 0; PRIMES.forEach((p) => { OFFS.push(o); o += p; }); }
    const TOTAL_ROWS = 320001536, BIGRAM_ROWS = OFFS[8];
    /** vizinhos com segmentação por EOS (mesma semântica de _shift_apply) */
    function neighbors(ids, q) {
      let prev = -1; for (let j = 0; j < q; j++) if (ids[j] === EOS) prev = j;
      const pis = q - prev - 1;
      const x = [], valid = [], outside = [];
      for (let s = 0; s < 3; s++) { const ok = s === 0 || (q - s >= 0 && pis >= s); valid.push(ok); outside.push(q - s < 0); x.push(ok ? ids[q - s] : EOS); }
      return { x, valid, outside };
    }
    function ngramRows(x, m) {
      const X = x.map((v) => BigInt(v));
      const h2 = (X[0] * m[0]) ^ (X[1] * m[1]);
      const h3 = h2 ^ (X[2] * m[2]);
      const rows = PRIMES.map((p, k) => Number((k < 8 ? h2 : h3) % BIGP[k]) + OFFS[k]);
      return { h2, h3, rows };
    }
    const hex = (b) => '0x' + b.toString(16).toUpperCase().padStart(16, '0');

    /* ================================================== Fig 1: três eixos */
    (function fig1() {
      const fig = root.querySelector('#fig-eixos'); if (!fig) return;
      const S = { mode: 'custo', rows: 320 };
      const setH = adaptH(fig, 430, () => (S.mode === 'custo' ? 720 : 530));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Ver', options: [{ v: 'custo', label: 'custo por token' }, { v: 'endereco', label: 'quando o endereço é conhecido' }], value: 'custo', onChange: (v) => { S.mode = v; setH(); } });
      UI.slider(ctl, { label: 'Linhas da tabela', min: 40, max: 1280, step: 40, value: 320, fmt: (v) => v + ' M', onInput: (v) => { S.rows = v; } });
      const toks = [' gato', ' subiu', ' no', ' telhado', ' de', ' novo', '.'];
      const fmtP = (x) => (x >= 1e9 ? fmt(x / 1e9, x >= 1e11 ? 0 : 1) + ' B' : x >= 1e6 ? fmt(x / 1e6, 1) + ' M' : fmtInt(x));
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C();
        if (S.mode === 'custo') drawCost(g, w, h, t, C); else drawAddr(g, w, h, t, C);
      });

      function drawCost(g, w, h, t, C) {
        const narrow = w < 620;
        const P = 2.2, ph = V.phase(t, P), cyc = ph.i, f = ph.f;
        const pulse = f < 0.22 ? V.ease.out(f / 0.22) : 1 - 0.45 * V.ease.inOut(Math.min(1, (f - 0.22) / 0.78));
        const tableP = S.rows * 1e6 * 160, expertsP = 120.80e9;
        const tableGiB = tableP / 2 ** 30;
        const panels = [
          { key: 'dense', title: 'Denso (hipotético)', col: C.d, lines: ['guardado: ' + fmtP(expertsP + tableP) + ' parâm.', 'lido por token: ' + fmtP(expertsP + tableP), '≈ ' + fmtInt(2 * (expertsP + tableP) / 1e9) + ' GFLOP/token'], ratio: '1 : 1' },
          { key: 'moe', title: 'MoE: experts do Qwen', col: C.a, lines: ['guardado: 120,8 B parâm.', 'ativo por token: 2,36 B', '≈ 4,7 GFLOP/token'], ratio: fmtInt(expertsP / 2.36e9) + ' : 1' },
          { key: 'ngram', title: 'Memória de N-gramas (PLE)', col: C.b, lines: ['guardado: ' + fmtP(tableP) + ' (' + fmt(tableGiB, 1) + ' GiB FP8)', 'lido por token: 2.560 valores', 'busca: 0 FLOP · proj.: 65,5 MFLOP'], ratio: fmtInt(tableP / 2560) + ' : 1' },
        ];
        const pad = 12, gap = 12, top = 36;
        let rects;
        if (!narrow) { const pw = (w - 2 * pad - 2 * gap) / 3; rects = panels.map((p, i) => ({ x: pad + i * (pw + gap), y: top, w: pw, h: h - top - pad })); }
        else { const phh = (h - top - pad - 2 * gap) / 3; rects = panels.map((p, i) => ({ x: pad, y: top + i * (phh + gap), w: w - 2 * pad, h: phh })); }
        // token atual
        const tk = toks[cyc % toks.length];
        V.text(g, 'token que chega:', pad, 22, { size: 12, color: C.muted });
        const tx = pad + tw(g, 'token que chega:', 12) + 8, tkw = tw(g, tk, 12.5, 'mono') + 18;
        V.box(g, tx, 8, tkw, 21, { r: 7, fill: V.alpha(C.c, 0.15 + 0.35 * (1 - f)), stroke: C.c, lw: 1.3 });
        V.text(g, tk, tx + 9, 22.5, { size: 12.5, font: 'mono', color: C.ink });
        if (!narrow) V.text(g, 'cada painel mostra o que esse token acende', w - pad, 22, { size: 11.5, color: C.muted, align: 'right' });
        panels.forEach((p, i) => drawPanel(g, rects[i], p, cyc, pulse, narrow, C, t));
      }

      function drawPanel(g, r, p, cyc, pulse, narrow, C, t) {
        V.box(g, r.x, r.y, r.w, r.h, { r: 12, fill: V.alpha(p.col, 0.05), stroke: V.alpha(p.col, 0.45), lw: 1.2 });
        V.text(g, p.title, r.x + 12, r.y + 20, { size: fitSize(g, p.title, r.w - 24, 13, 'sans', 600), weight: 600, color: p.col });
        // área do desenho e do texto
        let vx, vy, vw, vh, tx, ty, tW;
        if (!narrow) { vx = r.x + 12; vy = r.y + 32; vw = r.w - 24; vh = r.h - 32 - 122; tx = r.x + 12; ty = r.y + r.h - 106; tW = r.w - 24; }
        else { vx = r.x + 10; vy = r.y + 30; vw = Math.min(r.w * 0.42, 170); vh = r.h - 40; tx = vx + vw + 12; ty = r.y + 42; tW = r.x + r.w - 10 - tx; }
        const rng = V.rng(1000 + cyc * 31 + p.key.length);
        if (p.key === 'dense') {
          const s = 11, cols = Math.max(4, Math.floor(vw / (s + 2))), rows = Math.max(3, Math.floor(vh / (s + 2)));
          for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
            const wv = 0.55 + 0.45 * Math.sin(t * 3 - (i + j) * 0.35);
            g.fillStyle = V.alpha(p.col, 0.18 + 0.62 * pulse * wv); V.rr(g, vx + j * (s + 2), vy + i * (s + 2), s, s, 2.5); g.fill();
          }
        } else if (p.key === 'moe') {
          const N = 512; let s = Math.floor(Math.sqrt((vw * (vh - 22)) / N)) - 1; s = V.clamp(s, 4, 12);
          let cols = Math.floor(vw / (s + 1)), rows = Math.ceil(N / cols);
          while (rows * (s + 1) > vh - 22 && s > 3) { s--; cols = Math.floor(vw / (s + 1)); rows = Math.ceil(N / cols); }
          const lit = new Set(); while (lit.size < 10) lit.add(rng.int(N));
          for (let k = 0; k < N; k++) {
            const i = Math.floor(k / cols), j = k % cols, on = lit.has(k);
            g.fillStyle = on ? V.alpha(p.col, 0.35 + 0.65 * pulse) : V.alpha(C.muted, 0.14); V.rr(g, vx + j * (s + 1), vy + i * (s + 1), s, s, 1.5); g.fill();
            if (on && pulse > 0.6) V.glow(g, vx + j * (s + 1) + s / 2, vy + i * (s + 1) + s / 2, s * 1.6, p.col, 0.5 * pulse);
          }
          const yy = vy + rows * (s + 1) + 6;
          V.box(g, vx, yy, 14, 12, { r: 3, fill: V.alpha(C.e, 0.4 + 0.5 * pulse) });
          V.text(g, 'expert compartilhado (sempre ativo)', vx + 20, yy + 10, { size: 11, color: C.muted, maxW: vw - 22 });
        } else {
          const H = 16, gp = narrow ? 3 : 4, cw = (vw - gp * (H - 1)) / H;
          for (let k = 0; k < H; k++) {
            const x = vx + k * (cw + gp), col = k < 8 ? C.a : C.b;
            // estante: muitas linhas finas
            g.fillStyle = V.alpha(col, 0.1); V.rr(g, x, vy, cw, vh - 18, 3); g.fill();
            g.strokeStyle = V.alpha(col, 0.18); g.lineWidth = 1;
            for (let yy = vy + 4; yy < vy + vh - 20; yy += 4) { g.beginPath(); g.moveTo(x + 1, yy + 0.5); g.lineTo(x + cw - 1, yy + 0.5); g.stroke(); }
            const yy = vy + 4 + rng() * (vh - 30);
            g.fillStyle = V.alpha(col, 0.45 + 0.55 * pulse); g.fillRect(x - 1, yy - 1.5, cw + 2, 3.5);
            if (pulse > 0.5) V.glow(g, x + cw / 2, yy, cw * 1.4, col, 0.6 * pulse);
          }
          V.text(g, '8 bigrama', vx, vy + vh - 3, { size: 11, color: C.a, font: 'mono' });
          V.text(g, '8 trigrama', vx + vw, vy + vh - 3, { size: 11, color: C.b, font: 'mono', align: 'right' });
        }
        // texto
        const sz = narrow ? 11.5 : 12;
        p.lines.forEach((ln, i) => V.text(g, ln, tx, ty + i * (sz + 5), { size: fitSize(g, ln, tW, sz), color: i === 0 ? C.ink : C.muted, font: 'sans' }));
        const ry = narrow ? ty + 3 * (sz + 5) + 10 : r.y + r.h - 38;
        V.text(g, 'guardado ÷ lido', tx, ry, { size: 11, color: C.muted, font: 'mono' });
        V.text(g, p.ratio, tx, ry + (narrow ? 22 : 25), { size: fitSize(g, p.ratio, tW, narrow ? 18 : 21, 'display', 700), font: 'display', weight: 700, color: p.col });
      }

      function drawAddr(g, w, h, t, C) {
        const narrow = w < 620;
        const P = 8, f = (t % P) / P;
        const segs = narrow ? ['ids', 'cam. 0', 'cam. 1 PLE', 'cam. 2', 'cam. 3'] : ['ids do prompt', 'camada 0', 'camada 1 · PLE', 'camada 2', 'camada 3', '…'];
        const x0 = narrow ? 12 : 170, x1 = w - 12, segW = (x1 - x0) / segs.length;
        const headY = 14;
        segs.forEach((s, i) => {
          const x = x0 + i * segW, isPle = i === 2;
          V.box(g, x + 2, headY, segW - 4, 26, { r: 7, fill: V.alpha(isPle ? C.b : C.muted, isPle ? 0.22 : 0.1), stroke: V.alpha(isPle ? C.b : C.muted, 0.45), lw: 1 });
          V.text(g, s, x + segW / 2, headY + 17, { size: fitSize(g, s, segW - 10, 12), align: 'center', color: C.ink });
        });
        const play = x0 + f * (x1 - x0);
        const segX = (i, u = 0) => x0 + (i + u) * segW;
        const reveal = (x) => V.clamp((play - x) / 40 + 1, 0, 1);
        const rowsY = narrow ? [104, 224, 344] : [92, 92 + (h - 170) / 3, 92 + 2 * (h - 170) / 3];
        const labels = ['N-gramas (PLE)', 'MoE: roteador aprendido', 'hash routing do DeepSeek (camadas 0–2)'];
        const cols = [C.b, C.a, C.d];
        labels.forEach((lb, r) => {
          const y = rowsY[r];
          if (narrow) V.text(g, lb, x0, y - 24, { size: 12, weight: 600, color: cols[r] });
          else wrap(g, lb, 12, y - 4, 150, 15, { size: 12.5, weight: 600, color: cols[r] });
          V.line(g, x0, y + 10, x1, y + 10, { color: V.alpha(C.muted, 0.25), lw: 1, dash: [3, 5] });
        });
        const diamond = (x, y, col, a) => { g.save(); g.globalAlpha = a; g.translate(x, y); g.rotate(Math.PI / 4); g.fillStyle = col; g.fillRect(-6, -6, 12, 12); g.restore(); };
        const bar = (xa, xb, y, col, a, label, dashed) => {
          if (xb <= xa) return; g.save(); g.globalAlpha = a;
          V.box(g, xa, y - 9, xb - xa, 18, { r: 6, fill: V.alpha(col, dashed ? 0.12 : 0.3), stroke: col, lw: 1.2 });
          if (dashed) { g.setLineDash([4, 3]); V.rr(g, xa, y - 9, xb - xa, 18, 6); g.strokeStyle = col; g.stroke(); g.setLineDash([]); }
          // rótulo: o mais longo que couber na barra (nunca abaixo de 11px)
          const lab = [].concat(label || []).find((s) => tw(g, s, 11) <= xb - xa - 12);
          if (lab) V.text(g, lab, (xa + xb) / 2, y + 4, { size: 11, align: 'center', color: C.ink });
          g.restore();
        };
        // linha 1: PLE
        let y = rowsY[0] + 10;
        const dx = segX(0, 0.5);
        diamond(dx, y, cols[0], reveal(dx));
        if (!narrow) V.text(g, '16 endereços', dx, y - 14, { size: 11, align: 'center', color: cols[0], alpha: reveal(dx) });
        const fetchEnd = segX(2, 0.02);
        const fe = Math.min(fetchEnd, Math.max(dx + 8, play));
        bar(dx + 10, fe, y, cols[0], 1, ['busca das 16 linhas', 'busca'], false);
        if (!narrow && play > dx) V.text(g, 'na CPU, num arquivo mmap… em paralelo com a camada 0', dx + 10, y + 24, { size: 11, color: C.muted });
        if (play > fetchEnd) { V.text(g, '✓ a tempo', fetchEnd + 6, y + 4, { size: 12, weight: 600, color: C.e }); }
        // linha 2: MoE
        y = rowsY[1] + 10;
        for (let L = 0; L < Math.min(4, segs.length - 1); L++) {
          const rx = segX(L + 1, 0.42);
          diamond(rx, y, cols[1], reveal(rx));
          const end = segX(L + 1, 0.98), be = Math.min(end, Math.max(rx + 8, play));
          if (play > rx) bar(rx + 9, be, y, RED(), 0.85, '', false);
          if (L === 0 && play > rx) V.text(g, 'buscar experts', rx + 10, y - 14, { size: 11, color: RED() });
        }
        if (!narrow) V.text(g, 'o roteador só decide quando o estado oculto chega na camada', x0, y + 30, { size: 11, color: C.muted });
        // linha 3: hash routing DeepSeek
        y = rowsY[2] + 10;
        diamond(dx, y, cols[2], reveal(dx));
        const he = Math.min(segX(4, 0), Math.max(dx + 8, play));
        bar(dx + 10, he, y, cols[2], 1, ['experts já sabidos: tid2eid[x]', 'tid2eid[x]'], true);
        if (narrow) {
          const notes = ['as 16 linhas chegam durante a camada 0 ✓', 'o roteador decide dentro da camada; só então se buscam os experts', 'tid2eid[x]: experts das camadas 0–2 já sabidos pelos ids'];
          notes.forEach((nt, r) => wrap(g, nt, x0, rowsY[r] + 40, w - 24, 15, { size: 11, color: C.muted }));
        }
        // playhead
        V.line(g, play, headY + 30, play, narrow ? rowsY[2] + 24 : h - 58, { color: V.alpha(C.c, 0.8), lw: 1.5 });
        V.glow(g, play, headY + 30, 9, C.c);
        // rodapé
        const msg = 'O endereço da PLE sai só dos ids: dá para buscar as 16 linhas antes, enquanto a GPU calcula a camada 0. O roteador do MoE depende do estado oculto e só decide na hora.';
        wrap(g, msg, 12, narrow ? rowsY[2] + 100 : h - 38, w - 24, 16, { size: narrow ? 11.5 : 12.5, color: narrow ? C.ink : C.muted });
      }
    })();

    /* ================================================== Fig 2: máquina de hash */
    (function fig2() {
      const fig = root.querySelector('#fig-hash'); if (!fig) return;
      adaptH(fig, 490, 505);
      const SENTS = [
        { k: 'prosa', label: 'prosa', toks: [['O', 46], [' gato', 80741], [' subiu', 135590], [' no', 1113], [' telhado', 203417], [' de', 409], [' novo', 12592]] },
        { k: 'codigo', label: 'código', toks: [['def', 750], [' soma', 57212], ['(a', 2877], [',', 11], [' b', 293], ['):', 1648], [' return', 470]] },
        { k: 'eos', label: 'com EOS no meio', toks: [[' fim', 19844], ['.', 13], ['<|endoftext|>', EOS], ['Novo', 55184], [' texto', 31219], [' aqui', 24208], ['.', 13]] },
      ];
      const S = { sent: 0, pos: 4, seed: 1234, hexa: true, calc: null, t0: -10 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Frase', options: SENTS.map((s, i) => ({ v: i, label: s.label })), value: 0, onChange: (v) => { S.sent = v; recompute(); } });
      UI.slider(ctl, { label: 'Posição t', min: 0, max: 6, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.pos = v; recompute(); } });
      UI.seg(ctl, { label: 'Semente', options: [{ v: 1234, label: '1234 (a do treino)' }, { v: 1235, label: '1235 (errada)' }], value: 1234, onChange: (v) => { S.seed = v; recompute(); } });
      UI.toggle(ctl, { label: 'contas em hexadecimal', value: true, onChange: (v) => { S.hexa = v; } });
      let clock = -10;
      function recompute() {
        const sent = SENTS[S.sent], ids = sent.toks.map((x) => x[1]);
        const nb = neighbors(ids, S.pos), m = multipliers(S.seed), r = ngramRows(nb.x, m);
        S.calc = { ids, nb, m, ...r }; S.t0 = clock;
      }
      recompute();
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        clock = t;
        const C = V.C(), narrow = w < 620, calc = S.calc, sent = SENTS[S.sent];
        const pad = narrow ? 10 : 18, since = t - S.t0, appear = V.ease.out(V.clamp(since / 0.6, 0, 1));
        // ---- tokens
        const n = sent.toks.length, ghostW = narrow ? 0 : 58;
        const tg = narrow ? 4 : 8, tokW = Math.min(112, (w - 2 * pad - ghostW - tg * (n - 1)) / n), tokH = narrow ? 36 : 44;
        const rowY = 34, bx = pad + ghostW;
        V.text(g, 'sequência (ids ilustrativos; o EOS 248044 é real)', pad, 12, { size: 11, color: C.muted, maxW: w - 2 * pad });
        const q = S.pos;
        const role = (i) => (i === q ? 'cur' : i === q - 1 ? 'bi' : i === q - 2 ? 'tri' : '');
        sent.toks.forEach(([s, id], i) => {
          const x = bx + i * (tokW + tg), rl = role(i);
          const col = rl === 'cur' ? C.c : rl === 'bi' ? C.a : rl === 'tri' ? C.b : C.muted;
          const invalid = id !== EOS && ((rl === 'bi' && !calc.nb.valid[1]) || (rl === 'tri' && !calc.nb.valid[2]));
          const isE = id === EOS;
          tokBox(g, x, rowY, tokW, tokH, isE ? 'EOS' : s.replace(/ /g, '·'), col, { fill: V.alpha(col, rl ? 0.2 : 0.07), lw: rl ? 2 : 1, size: narrow ? 11 : 12.5, font: isE ? 'mono' : 'sans' });
          // no celular os ids vizinhos se encostam: mostra só os que entram na conta (e o EOS)
          if (!narrow || rl || id === EOS) V.text(g, fmtInt(id), x + tokW / 2, rowY + tokH + 13, { size: 11, font: 'mono', align: 'center', color: rl ? col : C.muted, maxW: tokW + tg });
          if (invalid) { V.line(g, x + 4, rowY + 4, x + tokW - 4, rowY + tokH - 4, { color: RED(), lw: 2 }); V.text(g, '→ EOS', x + tokW / 2, rowY - 4, { size: 11, align: 'center', color: RED(), font: 'mono' }); }
        });
        // fantasmas à esquerda (posições sem passado)
        [1, 2].forEach((s) => {
          if (!calc.nb.outside[s]) return;
          const col = s === 1 ? C.a : C.b;
          if (!narrow) { const x = pad + (2 - s) * 29 - 2; tokBox(g, x, rowY + 6, 26, tokH - 12, 'E', col, { dash: true, fill: V.alpha(col, 0.08), eyes: false, size: 11, font: 'mono' }); }
        });
        if (narrow && (calc.nb.outside[1] || calc.nb.outside[2])) V.text(g, 'sem passado → EOS', pad, rowY + tokH + 28, { size: 11, color: C.b, font: 'mono' });
        // ---- contas
        let y = rowY + tokH + (narrow ? 44 : 44);
        const x0 = calc.nb.x;
        const sm = (b) => (S.hexa ? hex(b) : fmtInt(b));
        const mono = narrow ? 10.5 : 12.5;
        if (!narrow) {
          V.text(g, `m₀ = ${fmtInt(calc.m[0])}   m₁ = ${fmtInt(calc.m[1])}   m₂ = ${fmtInt(calc.m[2])}   (SplitMix64, seed ${S.seed})`, pad, y, { size: 11.5, font: 'mono', color: S.seed === 1234 ? C.muted : C.d, maxW: w - 2 * pad });
          y += 27;
          const l2 = `h₂ = ${fmtInt(x0[0])}·m₀ ⊕ ${fmtInt(x0[1])}·m₁ = ${sm(calc.h2)}`;
          const l3 = `h₃ = h₂ ⊕ ${fmtInt(x0[2])}·m₂ = ${sm(calc.h3)}`;
          V.text(g, l2, pad, y, { size: fitSize(g, l2, w - 2 * pad, mono, 'mono'), font: 'mono', color: C.a });
          V.text(g, l3, pad, y + 24, { size: fitSize(g, l3, w - 2 * pad, mono, 'mono'), font: 'mono', color: C.b });
          y += 56;
        } else {
          V.text(g, `m = SplitMix64(seed ${S.seed})`, pad, y, { size: 11, font: 'mono', color: S.seed === 1234 ? C.muted : C.d });
          y += 18;
          V.text(g, `h₂ = ${fmtInt(x0[0])}·m₀ ⊕ ${fmtInt(x0[1])}·m₁`, pad, y, { size: fitSize(g, `h₂ = ${fmtInt(x0[0])}·m₀ ⊕ ${fmtInt(x0[1])}·m₁`, w - 2 * pad, 11, 'mono'), font: 'mono', color: C.a });
          V.text(g, `   = ${sm(calc.h2)}`, pad, y + 16, { size: 11, font: 'mono', color: C.a });
          V.text(g, `h₃ = h₂ ⊕ ${fmtInt(x0[2])}·m₂`, pad, y + 36, { size: 11, font: 'mono', color: C.b });
          V.text(g, `   = ${sm(calc.h3)}`, pad, y + 52, { size: 11, font: 'mono', color: C.b });
          y += 70;
        }
        // ---- fichas (16 endereços)
        const perRow = narrow ? 4 : 8, cg = narrow ? 4 : 7;
        const chipW = (w - 2 * pad - cg * (perRow - 1)) / perRow, chipH = narrow ? 36 : 40;
        const chipPos = [];
        V.text(g, narrow ? 'cada ficha: primo p e linha idx = off + h mod p' : 'cada ficha: o primo p da cabeça e a linha idx = offset + (h mod p) · 8 cabeças de bigrama, 8 de trigrama', pad, y, { size: narrow ? 11 : 11.5, color: C.muted, maxW: w - 2 * pad });
        y += 10;
        for (let k = 0; k < 16; k++) {
          const rI = Math.floor(k / perRow), cI = k % perRow;
          const x = pad + cI * (chipW + cg), yy = y + rI * (chipH + 6);
          const col = k < 8 ? C.a : C.b, a = V.clamp(appear * 16 - k, 0, 1);
          g.save(); g.globalAlpha = 0.25 + 0.75 * a;
          V.box(g, x, yy, chipW, chipH, { r: 7, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.7), lw: 1.1 });
          V.text(g, fmtInt(PRIMES[k]), x + 5, yy + 14, { size: 11, font: 'mono', color: C.muted, maxW: chipW - 9 });
          const rowTxt = fmtInt(calc.rows[k]);
          V.text(g, rowTxt, x + 5, yy + chipH - 9, { size: fitSize(g, rowTxt, chipW - 9, narrow ? 11 : 12.5, 'mono', 600), font: 'mono', weight: 600, color: col, maxW: chipW - 8 });
          g.restore();
          chipPos.push([x + chipW / 2, yy + chipH]);
        }
        y += Math.ceil(16 / perRow) * (chipH + 6) + (narrow ? 30 : 34);
        // ---- a tabela (tira de 320 M linhas)
        const sx = pad, sw = w - 2 * pad, sh = narrow ? 20 : 24;
        V.text(g, `a tabela única: ${fmtInt(TOTAL_ROWS)} linhas × 160`, sx, y - 10, { size: narrow ? 11 : 11.5, color: C.muted, font: 'mono', maxW: sw });
        PRIMES.forEach((p, k) => {
          const xa = sx + (OFFS[k] / TOTAL_ROWS) * sw, xb = sx + ((OFFS[k] + p) / TOTAL_ROWS) * sw, col = k < 8 ? C.a : C.b;
          g.fillStyle = V.alpha(col, k % 2 ? 0.16 : 0.26); g.fillRect(xa, y, xb - xa - 0.6, sh);
        });
        const ticks = calc.rows.map((rw) => sx + (rw / TOTAL_ROWS) * sw);
        ticks.forEach((xt, k) => {
          const col = k < 8 ? C.a : C.b, [cx, cy] = chipPos[k];
          if (!narrow) { g.strokeStyle = V.alpha(col, 0.28 * appear); g.lineWidth = 1; g.beginPath(); g.moveTo(cx, cy); g.bezierCurveTo(cx, cy + 18, xt, y - 20, xt, y); g.stroke(); }
          g.fillStyle = C.c; g.fillRect(xt - 1, y - 3, 2.2, sh + 6);
        });
        const bxm = sx + (BIGRAM_ROWS / TOTAL_ROWS) * sw;
        V.text(g, '0', sx, y + sh + 14, { size: 11, color: C.muted, font: 'mono' });
        V.text(g, narrow ? 'bigramas | trigramas' : `← bigramas (${fmtInt(BIGRAM_ROWS)} linhas) | trigramas →`, bxm, y + sh + 14, { size: 11, color: C.muted, font: 'mono', align: 'center' });
        V.text(g, narrow ? '320 M' : fmtInt(TOTAL_ROWS), sx + sw, y + sh + 14, { size: 11, color: C.muted, font: 'mono', align: 'right' });
        y += sh + 30;
        // ---- e_t: 16 blocos de 160 concatenados
        const bw = (sw - 15 * 3) / 16, bh = narrow ? 14 : 16, cellsPer = narrow ? 3 : 6;
        for (let k = 0; k < 16; k++) {
          const xk = sx + k * (bw + 3), col = k < 8 ? C.a : C.b, rr = V.rng(calc.rows[k] % 100000 + 7);
          const lift = V.clamp(appear * 1.6 - k * 0.04, 0, 1);
          for (let c = 0; c < cellsPer; c++) {
            const v = rr.normal() * 0.6;
            g.fillStyle = V.cmap(V.clamp(v, -1, 1), 'div'); g.globalAlpha = 0.35 + 0.65 * lift;
            V.rr(g, xk + c * (bw / cellsPer), y + (1 - lift) * -10, bw / cellsPer - 1, bh, 2); g.fill();
          }
          g.globalAlpha = 1;
          g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1; V.rr(g, xk - 0.5, y - 0.5, bw + 1, bh + 1, 3); g.stroke();
        }
        V.text(g, 'e_t = [16 linhas de 160] ∈ ℝ²⁵⁶⁰', sx, y + bh + 16, { size: narrow ? 11 : 12, font: 'mono', color: C.ink });
      }, { static: false });
    })();

    /* ================================================== Fig 3: colisões */
    (function fig3() {
      const fig = root.querySelector('#fig-colisoes'); if (!fig) return;
      adaptH(fig, 470, 605);
      const S = { M: 14, H: 4, sizes: 'primos', ids: 'variados', res: null, t0: -10 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Bigramas', min: 4, max: 40, step: 1, value: 14, fmt: (v) => String(v), onInput: (v) => { S.M = v; recompute(); } });
      UI.seg(ctl, { label: 'Cabeças', options: [1, 2, 3, 4], value: 4, onChange: (v) => { S.H = v; recompute(); } });
      UI.seg(ctl, { label: 'Tamanhos', options: [{ v: 'primos', label: 'primos distintos' }, { v: 'dezesseis', label: '16 em todas' }], value: 'primos', onChange: (v) => { S.sizes = v; recompute(); } });
      UI.seg(ctl, { label: 'Ids', options: [{ v: 'variados', label: 'variados' }, { v: 'estruturados', label: 'múltiplos de 16' }], value: 'variados', onChange: (v) => { S.ids = v; recompute(); } });
      const m = multipliers(1234);
      const TOY_P = [17, 19, 23, 29];
      let clock = -10;
      function recompute() {
        const sizes = S.sizes === 'primos' ? TOY_P.slice(0, S.H) : Array(S.H).fill(16);
        const r = V.rng(S.ids === 'variados' ? 91 : 57);
        const grams = [];
        for (let j = 0; j < S.M; j++) {
          const a = S.ids === 'variados' ? 1 + r.int(248000) : 16 * (1 + r.int(15000));
          const b = S.ids === 'variados' ? 1 + r.int(248000) : 16 * (1 + r.int(15000));
          const hh = (BigInt(b) * m[0]) ^ (BigInt(a) * m[1]);
          grams.push({ a, b, bins: sizes.map((p) => Number(hh % BigInt(p))) });
        }
        let any = 0, all = 0;
        for (let i = 0; i < S.M; i++) for (let j = i + 1; j < S.M; j++) {
          let c = 0; for (let k = 0; k < S.H; k++) if (grams[i].bins[k] === grams[j].bins[k]) c++;
          if (c > 0) any++; if (c === S.H) all++;
        }
        let maxLoad = 0; sizes.forEach((p, k) => { const cnt = Array(p).fill(0); grams.forEach((gm) => cnt[gm.bins[k]]++); maxLoad = Math.max(maxLoad, ...cnt); });
        const pairs = (S.M * (S.M - 1)) / 2;
        const expAll = pairs * sizes.reduce((acc, p) => acc / p, 1);
        S.res = { sizes, grams, any, all, maxLoad, pairs, expAll }; S.t0 = clock;
      }
      recompute();
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        clock = t;
        const C = V.C(), narrow = w < 620, R = S.res, pad = narrow ? 10 : 16;
        const since = t - S.t0;
        const panelW = narrow ? w - 2 * pad : 232;
        const bx = pad, bw = narrow ? w - 2 * pad : w - 2 * pad - panelW - 18;
        const areaH = narrow ? h - 220 : h - 2 * pad;
        const rowH = Math.min(118, (areaH - 10) / S.H);
        const maxP = Math.max(...R.sizes);
        const labW = narrow ? 0 : 92;
        const binW = (bw - labW) / maxP;
        const dotR = V.clamp(binW * 0.3, 2.6, 5);
        const focus = R.grams[0];
        R.sizes.forEach((p, k) => {
          const y0 = pad + k * rowH, yb = y0 + rowH - (narrow ? 14 : 16);
          const lbl = `cabeça ${k + 1} · p = ${p}`;
          if (narrow) V.text(g, lbl, bx, y0 + 11, { size: 11, color: C.muted, font: 'mono' });
          else { V.text(g, `cabeça ${k + 1}`, bx, y0 + rowH / 2, { size: 12, color: C.ink, weight: 600 }); V.text(g, `p = ${p}`, bx, y0 + rowH / 2 + 16, { size: 11, color: p === 16 ? RED() : C.muted, font: 'mono' }); }
          const cnt = Array(p).fill(0);
          for (let b = 0; b < p; b++) {
            const x = bx + labW + b * binW;
            g.fillStyle = V.alpha(C.muted, b % 2 ? 0.06 : 0.11); g.fillRect(x + 1, y0 + (narrow ? 16 : 6), binW - 2, yb - y0 - (narrow ? 16 : 6));
            if (focus && focus.bins[k] === b) { g.strokeStyle = V.alpha(C.c, 0.8); g.lineWidth = 1.5; g.strokeRect(x + 1, y0 + (narrow ? 16 : 6), binW - 2, yb - y0 - (narrow ? 16 : 6)); }
          }
          R.grams.forEach((gm, j) => {
            const b = gm.bins[k], stack = cnt[b]++;
            const x = bx + labW + b * binW + binW / 2;
            const drop = V.ease.out(V.clamp((since - j * 0.025) / 0.45, 0, 1));
            const ty = yb - dotR - 1 - stack * (dotR * 2 + 1);
            const yy = V.lerp(y0 + 8, ty, drop);
            const isF = j === 0, mate = !isF && focus && gm.bins[k] === focus.bins[k];
            if (ty < y0 + (narrow ? 18 : 8)) return; // estourou a altura: não desenha
            V.dot(g, x, yy, dotR, isF ? C.c : V.alpha([C.a, C.b, C.d, C.e][j % 4], 0.85));
            if (isF) V.glow(g, x, yy, dotR * 3, C.c, 0.6);
            if (mate) { g.strokeStyle = C.c; g.lineWidth = 1.4; g.beginPath(); g.arc(x, yy, dotR + 2, 0, Math.PI * 2); g.stroke(); }
          });
          V.text(g, '0', bx + labW + binW / 2, yb + 13, { size: 11, color: C.muted, align: 'center', font: 'mono' });
          V.text(g, String(p - 1), bx + labW + (p - 0.5) * binW, yb + 13, { size: 11, color: C.muted, align: 'center', font: 'mono' });
        });
        // painel de contagem
        const px = narrow ? pad : w - pad - panelW, py = narrow ? pad + S.H * rowH + 8 : pad, ph = narrow ? h - py - pad : h - 2 * pad;
        V.box(g, px, py, panelW, ph, { r: 12, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let yy = py + 22;
        const big = (label, val, col) => { V.text(g, label, px + 14, yy, { size: 11.5, color: C.muted, maxW: panelW - 28 }); V.text(g, val, px + panelW - 14, yy + (narrow ? 0 : 24), { size: narrow ? 17 : 22, font: 'display', weight: 700, color: col, align: 'right' }); yy += narrow ? 24 : 46; };
        V.text(g, `${S.M} bigramas → ${R.pairs} pares`, px + 14, yy, { size: 12, color: C.ink, weight: 600 });
        yy += narrow ? 20 : 26;
        big('dividem balde em ≥ 1 cabeça', String(R.any), C.a);
        big(`dividem balde em todas as ${S.H}`, String(R.all), R.all ? RED() : C.e);
        big('maior balde', String(R.maxLoad), R.maxLoad > 4 ? C.d : C.ink);
        const msg = S.sizes === 'dezesseis' ? (S.H > 1 ? 'Mesmo módulo: as cabeças são clones. Colidir numa é colidir em todas.' : 'Módulo 16 só olha os 4 bits baixos de h.') + (S.ids === 'estruturados' ? ' Com ids múltiplos de 16, os bits baixos são iguais: tudo cai no mesmo balde.' : '')
          : `Primos distintos: cada cabeça espalha o mesmo h de outro jeito. Esperado de colisão total ≈ ${fmt(R.expAll, R.expAll < 0.1 ? 3 : 2)} par(es).`;
        wrap(g, msg, px + 14, yy + (narrow ? 2 : 6), panelW - 28, 15, { size: 11.5, color: C.ink });
      });
    })();

    /* ================================================== Fig 4: pipeline (stepper) */
    (function fig4() {
      const fig = root.querySelector('#fig-pipeline'); if (!fig) return;
      adaptH(fig, 440, 640);
      const steps = [
        { t: 'Buscar 16 linhas', d: 'Os 16 endereços do hash apontam 16 linhas de 160 valores na tabela de 320 M linhas. As linhas são copiadas, dequantizadas (FP8 × escala global, ou NVFP4) e concatenadas em \\(\\mathbf e_t \\in \\mathbb R^{2560}\\). Nenhuma multiplicação por peso até aqui.' },
        { t: 'Projetar: 4 chaves e 1 valor', d: '<code>key_proj</code> (2560 → 10240) produz uma chave por ramo do resíduo, \\(\\mathbf k_t \\in \\mathbb R^{4\\times2560}\\). <code>value_proj</code> (2560 → 2560) produz um único valor \\(\\mathbf v_t\\), compartilhado pelos 4 ramos. São as únicas multiplicações pesadas do módulo (~65,5 MFLOP/token).' },
        { t: 'Consultar o resíduo', d: 'A consulta de cada ramo é o próprio estado residual \\(R_{t,i}\\) (4 × 2560) na entrada da camada de índice 1. Chaves e consultas passam por RMSNorm agrupada (grupos de 2560, ganho \\(1+w\\), ε = 1e-6).' },
        { t: 'Gate por ramo', d: '\\(z_i = \\langle \\hat{\\mathbf k}_i, \\hat{\\mathbf q}_i\\rangle/\\sqrt{2560}\\) e \\(g_i = \\sigma(\\mathrm{sign}(z_i)\\sqrt{|z_i|})\\). Quando a memória concorda com o contexto, o gate abre. O gráfico pequeno compara a curva com \\(\\sigma(z)\\): a raiz com sinal satura mais devagar.' },
        { t: 'Valor filtrado + conv curta', d: '\\(\\mathbf u_i = g_i\\,\\mathbf v\\) (4 × 2560). Depois uma RMSNorm e a conv causal depthwise (k = 4, dilatação 3, 10.240 canais) com SiLU somam contexto lexical de até 12 tokens: \\(\\Delta = \\mathbf u + \\mathrm{SiLU}(\\mathrm{conv}(\\mathrm{norm}(\\mathbf u)))\\).' },
        { t: 'Escrever nos 4 ramos', d: '\\(R_t \\leftarrow R_t + \\Delta_t\\), antes da leitura do Gated Residual da camada de índice 1. Dali em diante a GDN da camada 1 e todas as camadas seguintes enxergam a memória.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const vv = Array.from({ length: 8 }, (_, i) => Math.sin(i * 1.7 + 0.4) * 0.8);
      const kk = Array.from({ length: 4 }, (_, b) => Array.from({ length: 8 }, (_, i) => Math.sin(i * 1.3 + b * 2.1) * 0.9));
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C(), narrow = w < 620, si = st.i, el = st.elapsed();
        const lanesCol = [C.a, C.b, C.d, C.c];
        const zs = [2.2, -0.4, 0.9, -3.1].map((z, i) => z + 0.9 * Math.sin(0.7 * t + i * 1.3));
        const gs = zs.map((z) => sig(Math.sign(z) * Math.sqrt(Math.max(1e-6, Math.abs(z)))));
        const alphaFor = (k) => (k === si ? 1 : k < si ? 0.75 : 0.22);
        const flow = (x1, y1, x2, y2, col, k) => {
          if (k > si) { V.line(g, x1, y1, x2, y2, { color: V.alpha(C.muted, 0.2), lw: 1, dash: [3, 4] }); return; }
          V.arrow(g, x1, y1, x2, y2, { color: V.alpha(col, k === si ? 0.9 : 0.5), lw: k === si ? 2 : 1.3, head: 7 });
          if (k === si) for (let p = 0; p < 3; p++) { const u = (t * 0.8 + p / 3) % 1; const q = V.along(x1, y1, x2, y2, u); V.dot(g, q[0], q[1], 2.6, col); }
        };
        const cellC = (v, col) => V.alpha(col, 0.25 + 0.6 * Math.min(1, Math.abs(v)));
        const L = narrow ? layoutN(w, h) : layoutW(w, h);
        // ---- faixas do resíduo
        const lanes = L.lanes;
        lanes.forEach((ln, i) => {
          g.globalAlpha = 0.9;
          if (!narrow) { V.box(g, ln.x, ln.y - 5, ln.w, 10, { r: 5, fill: V.alpha(lanesCol[i], 0.22), stroke: V.alpha(lanesCol[i], 0.5), lw: 1 }); }
          else { V.box(g, ln.x - 5, ln.y, 10, ln.h, { r: 5, fill: V.alpha(lanesCol[i], 0.22), stroke: V.alpha(lanesCol[i], 0.5), lw: 1 }); }
          g.globalAlpha = 1;
        });
        V.text(g, L.laneLabel.s, L.laneLabel.x, L.laneLabel.y, { size: 11.5, color: C.muted, font: 'mono', align: L.laneLabel.align || 'left' });
        // ---- estação 0: tabela + e_t
        g.globalAlpha = alphaFor(0);
        const T0 = L.table;
        V.box(g, T0.x, T0.y, T0.w, T0.h, { r: 6, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.5), lw: 1 });
        for (let k = 0; k < 16; k++) { const yy = T0.y + 4 + ((k * 37 + 11) % 100) / 100 * (T0.h - 8); g.fillStyle = k < 8 ? C.a : C.b; g.fillRect(T0.x + 2, yy, T0.w - 4, 2.2); }
        V.text(g, 'tabela', T0.x + T0.w / 2, T0.y + T0.h + 13, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        const E = L.e;
        strip(g, E.x, E.y, 16, E.cw, E.ch, (i) => (i < 8 ? V.alpha(C.a, 0.75) : V.alpha(C.b, 0.75)), { gap: 1 });
        V.text(g, 'e_t · 2560', E.x, E.y - 6, { size: 11, font: 'mono', color: C.ink });
        g.globalAlpha = 1;
        flow(T0.x + T0.w + 2, T0.y + T0.h / 2, E.x - 4, E.y + E.ch / 2, C.b, 0);
        // ---- estação 1: projeções
        g.globalAlpha = alphaFor(1);
        const WK = L.wk, WV = L.wv;
        V.matrix(g, { x: WK.x, y: WK.y, data: Array.from({ length: 6 }, (_, i) => Array.from({ length: 3 }, (_, j) => Math.sin(i * 2.1 + j))), cell: WK.c, gap: 2, cmap: 'div' });
        V.text(g, 'W_K 10240×2560', WK.x, WK.y - 6, { size: 11, font: 'mono', color: C.ink });
        const wvm = V.matrix(g, { x: WV.x, y: WV.y, data: Array.from({ length: 3 }, (_, i) => Array.from({ length: 3 }, (_, j) => Math.cos(i * 1.7 + j * 2))), cell: WV.c, gap: 2, cmap: 'div' });
        V.text(g, 'W_V 2560×2560', WV.x, WV.y + wvm.h + 13, { size: 11, font: 'mono', color: C.ink });
        L.keys.forEach((kp, b) => strip(g, kp.x, kp.y, 8, kp.cw, kp.ch, (i) => cellC(kk[b][i], lanesCol[b]), { gap: 1 }));
        if (narrow) V.text(g, 'k₁..k₄ (4 × 2560)', L.keys[0].x, L.keys[3].y + 24, { size: 11, font: 'mono', color: C.ink });
        else V.text(g, 'k₁..k₄ (4 × 2560)', L.keys[0].x, L.keys[0].y - 5, { size: 11, font: 'mono', color: C.ink });
        strip(g, L.v.x, L.v.y, 8, L.v.cw, L.v.ch, (i) => cellC(vv[i], C.ink), { gap: 1 });
        V.text(g, 'v · 2560', L.v.x, L.v.y - 5, { size: 11, font: 'mono', color: C.ink });
        g.globalAlpha = 1;
        if (narrow) { flow(E.x + 16, E.y + E.ch + 3, WK.x + 20, WK.y - 16, C.b, 1); flow(E.x + E.w * 0.75, E.y + E.ch + 3, WV.x + 16, WV.y - 16, C.b, 1); }
        else { flow(E.x + E.w + 2, E.y + E.ch / 2, WK.x - 4, WK.y + 12, C.b, 1); flow(E.x + E.w * 0.6, E.y + E.ch + 2, WV.x - 2, WV.y + 6, C.b, 1); }
        // ---- estação 2: consultas do resíduo
        L.queries.forEach((qp, b) => { if (si >= 2) flow(qp.x1, qp.y1, qp.x2, qp.y2, lanesCol[b], 2); else V.line(g, qp.x1, qp.y1, qp.x2, qp.y2, { color: V.alpha(C.muted, 0.18), lw: 1, dash: [3, 4] }); });
        g.globalAlpha = alphaFor(2);
        V.text(g, 'q_i = RMSNorm(R_i)', L.qLabel.x, L.qLabel.y, { size: 11, font: 'mono', color: C.ink, align: L.qLabel.align || 'left' });
        g.globalAlpha = 1;
        // ---- estação 3: gates
        g.globalAlpha = alphaFor(3);
        L.gauges.forEach((gp, b) => {
          const r = gp.r, val = gs[b];
          g.lineCap = 'round'; g.lineWidth = 5; g.strokeStyle = V.alpha(C.muted, 0.22); g.beginPath(); g.arc(gp.x, gp.y, r, Math.PI, 2 * Math.PI); g.stroke();
          g.strokeStyle = lanesCol[b]; g.beginPath(); g.arc(gp.x, gp.y, r, Math.PI, Math.PI + Math.PI * val); g.stroke(); g.lineCap = 'butt';
          // rótulo dentro do arco, valor embaixo (nada sobre o traço)
          V.text(g, 'g' + (b + 1), gp.x, gp.y - 3, { size: 11, font: 'mono', align: 'center', color: lanesCol[b] });
          V.text(g, fmt(val, 2), gp.x, gp.y + 15, { size: 11, font: 'mono', align: 'center', color: C.ink });
        });
        // mini gráfico
        const Gp = L.plot;
        if (Gp) {
          V.box(g, Gp.x, Gp.y, Gp.w, Gp.h, { r: 6, fill: V.alpha(C.panel, 0.5), stroke: V.alpha(C.muted, 0.25), lw: 1 });
          const X = (z) => Gp.x + 6 + ((z + 6) / 12) * (Gp.w - 12), Y = (s) => Gp.y + Gp.h - 6 - s * (Gp.h - 16);
          g.lineWidth = 1.4;
          g.strokeStyle = V.alpha(C.muted, 0.8); g.beginPath(); for (let z = -6; z <= 6; z += 0.1) { const p = [X(z), Y(sig(z))]; z === -6 ? g.moveTo(...p) : g.lineTo(...p); } g.stroke();
          g.strokeStyle = C.c; g.beginPath(); for (let z = -6; z <= 6; z += 0.1) { const p = [X(z), Y(sig(Math.sign(z) * Math.sqrt(Math.abs(z))))]; z === -6 ? g.moveTo(...p) : g.lineTo(...p); } g.stroke();
          zs.forEach((z, b) => V.dot(g, X(V.clamp(z, -6, 6)), Y(gs[b]), 3, lanesCol[b]));
          V.text(g, 'σ(z)', Gp.x + Gp.w - 6, Gp.y + Gp.h - 26, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, 'σ(sign·√|z|)', Gp.x + 6, Gp.y + 12, { size: 11, font: 'mono', color: C.c });
        }
        g.globalAlpha = 1;
        if (narrow) flow(L.keys[0].x + 30, L.keys[3].y + 30, L.gauges[1].x, L.gauges[1].y - 24, C.a, 3);
        else flow(L.keys[0].x + L.keys[0].w + 4, L.keys[1].y, L.gauges[0].x - L.gauges[0].r - 8, L.gauges[0].y - 6, C.a, 3);
        // ---- estação 4: U e conv
        g.globalAlpha = alphaFor(4);
        L.us.forEach((up, b) => strip(g, up.x, up.y, 8, up.cw, up.ch, (i) => V.alpha(lanesCol[b], (0.15 + 0.7 * Math.min(1, Math.abs(vv[i]))) * (0.25 + 0.75 * gs[b])), { gap: 1 }));
        V.text(g, 'u_i = g_i·v', L.us[0].x, L.us[0].y - 5, { size: 11, font: 'mono', color: C.ink });
        const CV = L.conv;
        V.box(g, CV.x, CV.y, CV.w, CV.h, { r: 8, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.7), lw: 1.2 });
        V.text(g, 'conv k=4 · dil 3', CV.x + CV.w / 2, CV.y + 15, { size: 11, font: 'mono', align: 'center', color: C.ink });
        for (let j = 0; j < 10; j++) { const xx = CV.x + 8 + j * ((CV.w - 16) / 9), on = j % 3 === 0; V.dot(g, xx, CV.y + CV.h - 12, on ? 3.2 : 2, on ? C.d : V.alpha(C.muted, 0.5)); }
        V.text(g, '+ SiLU, + u', CV.x + CV.w / 2, CV.y + CV.h - 24, { size: 11, font: 'mono', align: 'center', color: C.muted });
        g.globalAlpha = 1;
        if (narrow) flow(L.us[0].x + 34, L.plot.y + L.plot.h + 4, L.us[0].x + 34, L.us[0].y - 16, C.d, 4);
        else flow(L.gauges[3].x + L.gauges[3].r + 6, L.gauges[3].y - 4, L.us[0].x - 4, L.us[0].y + 4, C.d, 4);
        flow(L.us[0].x + L.us[0].w + 4, L.us[1].y + 2, CV.x - 3, CV.y + CV.h / 2, C.d, 4);
        // ---- estação 5: Δ nos ramos
        g.globalAlpha = alphaFor(5);
        const J = L.junction;
        V.box(g, J.x - 14, J.y - 14, 28, 28, { r: 14, fill: V.alpha(C.c, 0.2), stroke: C.c, lw: 1.5 });
        V.text(g, '+', J.x, J.y + 6, { size: 18, align: 'center', color: C.ink, weight: 700 });
        V.text(g, L.after.s, L.after.x, L.after.y, { size: 11, font: 'mono', color: C.ink, align: L.after.align || 'left' });
        g.globalAlpha = 1;
        flow(CV.x + CV.w / 2, CV.y + CV.h + 2, J.x, J.y - 16, C.c, 5);
        if (si >= 5) {
          if (!narrow) { V.line(g, J.x, J.y + 14, J.x, lanes[3].y + 5, { color: V.alpha(C.c, 0.8), lw: 2 }); lanes.forEach((ln, i) => { V.dot(g, J.x, ln.y, 4, lanesCol[i]); V.glow(g, J.x, ln.y, 10, lanesCol[i], 0.6); }); }
          else { V.line(g, J.x + 14, J.y, lanes[3].x + 5, J.y, { color: V.alpha(C.c, 0.8), lw: 2 }); lanes.forEach((ln, i) => { V.dot(g, ln.x, J.y, 4, lanesCol[i]); V.glow(g, ln.x, J.y, 10, lanesCol[i], 0.6); }); }
        }
        if (si === 5) { const pulseR = 14 + 10 * ((t * 1.2) % 1); g.strokeStyle = V.alpha(C.c, 1 - ((t * 1.2) % 1)); g.lineWidth = 2; g.beginPath(); g.arc(J.x, J.y, pulseR, 0, Math.PI * 2); g.stroke(); }
        void el;
      });
      function layoutW(w, h) {
        const top = 34, laneY0 = h - 110;
        const lanes = [0, 1, 2, 3].map((i) => ({ x: 20, y: laneY0 + i * 16, w: w - 40 }));
        const table = { x: 22, y: top, w: 28, h: 150 };
        const e = { x: 70, y: top + 60, cw: 6, ch: 22, w: 16 * 7 - 1 };
        const wk = { x: 222, y: top + 6, c: 14 }, wv = { x: 222, y: top + 128, c: 12 };
        const keys = [0, 1, 2, 3].map((b) => ({ x: 292, y: top + 30 + b * 16, cw: 8, ch: 10, w: 8 * 9 - 1 }));
        const v = { x: 292, y: top + 132, cw: 8, ch: 12 };
        const gx = Math.round(w * 0.5);
        const gauges = [0, 1, 2, 3].map((b) => ({ x: gx + b * 46, y: top + 214, r: 17 }));
        const plot = { x: gx - 18, y: top + 104, w: 172, h: 72 };
        const conv = { x: w - 136, y: top + 170, w: 120, h: 70 };
        const us = [0, 1, 2, 3].map((b) => ({ x: conv.x - 90, y: top + 176 + b * 16, cw: 8, ch: 10, w: 8 * 9 - 1 }));
        const junction = { x: conv.x + conv.w / 2, y: laneY0 - 26 };
        const queries = [0, 1, 2, 3].map((b) => ({ x1: gauges[b].x - 4, y1: lanes[b].y - 6, x2: gauges[b].x, y2: gauges[b].y + 27 }));
        return { lanes, table, e, wk, wv, keys, v, gauges, plot, us, conv, junction, queries,
          laneLabel: { s: 'R_t: 4 ramos × 2560 (Gated Residual)', x: 20, y: laneY0 + 4 * 16 + 12 },
          qLabel: { x: gx + 150, y: laneY0 - 40 }, after: { s: '→ GR + GDN da camada 1', x: w - 22, y: laneY0 + 4 * 16 + 12, align: 'right' } };
      }
      function layoutN(w, h) {
        const lx = w - 58, top = 22;
        const lanes = [0, 1, 2, 3].map((i) => ({ x: lx + i * 12, y: top, h: h - top - 24 }));
        const cx = 14;
        const table = { x: cx, y: top + 6, w: 22, h: 96 };
        const e = { x: cx + 36, y: top + 46, cw: 7, ch: 18, w: 16 * 8 - 1 };
        const wk = { x: cx + 6, y: top + 150, c: 12 }, wv = { x: cx + 150, y: top + 150, c: 10 };
        const keys = [0, 1, 2, 3].map((b) => ({ x: cx + 60, y: top + 154 + b * 15, cw: 7, ch: 10, w: 8 * 8 - 1 }));
        const v = { x: cx + 196, y: top + 158, cw: 5, ch: 12 };
        const gy = top + 300;
        const gauges = [0, 1, 2, 3].map((b) => ({ x: cx + 26 + b * 50, y: gy, r: 18 }));
        const plot = { x: cx, y: gy + 34, w: Math.min(210, lx - cx - 30), h: 70 };
        const us = [0, 1, 2, 3].map((b) => ({ x: cx + 6, y: gy + 150 + b * 15, cw: 8, ch: 10, w: 8 * 9 - 1 }));
        const conv = { x: cx + 100, y: gy + 142, w: Math.min(130, lx - cx - 120), h: 66 };
        const junction = { x: conv.x + conv.w / 2, y: gy + 250 };
        const queries = [0, 1, 2, 3].map((b) => ({ x1: lanes[b].x - 6, y1: gy - 40 + b * 4, x2: gauges[b].x + 10, y2: gauges[b].y - 22 }));
        return { lanes, table, e, wk, wv, keys, v, gauges, plot, us, conv, junction, queries,
          laneLabel: { s: 'R_t · 4 ramos', x: w - 12, y: h - 6, align: 'right' },
          qLabel: { x: cx, y: gy - 48 }, after: { s: 'Δ → 4 ramos (camada 1)', x: junction.x - 60, y: junction.y + 34 } };
      }
    })();

    /* ================================================== Fig 5: conv dilatada */
    (function fig5() {
      const fig = root.querySelector('#fig-conv'); if (!fig) return;
      adaptH(fig, 380, 470);
      const S = { k: 4, d: 3, spec: 3, tri: true };
      const ctl = UI.controls(fig);
      const sk = UI.slider(ctl, { label: 'Kernel k', min: 2, max: 6, step: 1, value: 4, onInput: (v) => { S.k = v; upd(); } });
      const sd = UI.slider(ctl, { label: 'Dilatação d', min: 1, max: 5, step: 1, value: 3, onInput: (v) => { S.d = v; upd(); } });
      UI.slider(ctl, { label: 'Tokens spec.', min: 0, max: 4, step: 1, value: 3, onInput: (v) => { S.spec = v; upd(); } });
      UI.toggle(ctl, { label: 'janelas de trigrama', value: true, onChange: (v) => { S.tri = v; } });
      UI.button(ctl, 'preset Qwen (k=4, d=3)', () => { sk.set(4); sd.set(3); });
      const rRF = UI.readout(ctl, 'campo receptivo', '10 pos.');
      const rST = UI.readout(ctl, 'estado/requisição', '240 KiB');
      function upd() {
        rRF.set(((S.k - 1) * S.d + 1) + ' pos.');
        const cols = (S.k - 1) * S.d + S.spec;
        rST.set(fmt((cols * 10240 * 2) / 1024, 0) + ' KiB');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C(), narrow = w < 620;
        const infoW = narrow ? 0 : 236, pad = 14;
        const areaW = w - 2 * pad - (narrow ? 0 : infoW + 14);
        const N = narrow ? 16 : 26;
        const cw = areaW / N, axisY = narrow ? 200 : h * 0.6, outY = narrow ? 50 : 56;
        const tIdx = N - 1;
        const taps = []; for (let j = 0; j < S.k; j++) taps.push(tIdx - j * S.d);
        const cx = (i) => pad + i * cw + cw / 2;
        // eixo de posições
        for (let i = 0; i < N; i++) {
          const x = pad + i * cw, isTap = taps.includes(i), cur = i === tIdx;
          V.box(g, x + 1.5, axisY, cw - 3, 22, { r: 5, fill: cur ? V.alpha(C.c, 0.35) : isTap ? V.alpha(C.d, 0.3) : V.alpha(C.muted, 0.1), stroke: cur ? C.c : isTap ? C.d : V.alpha(C.muted, 0.3), lw: isTap || cur ? 1.5 : 1 });
          const lb = i === tIdx ? 't' : 't-' + (tIdx - i);
          // com d = 1 os taps ficam colados: rotula só as pontas para não amontoar
          const showTap = isTap && (S.d >= 2 || i === Math.max(0, Math.min(...taps)));
          if (cur || showTap || (!isTap && cw > 30 && (tIdx - i) % 2 === 0)) V.text(g, lb, x + cw / 2, axisY + 36, { size: 11, font: 'mono', align: 'center', color: isTap || cur ? C.ink : C.muted, maxW: cw * Math.max(1, Math.min(S.d, 2)) - 2 });
        }
        // as curvas dos taps cruzam o alto do eixo: o rótulo fica embaixo
        V.text(g, '↑ entradas u (depois do gate e da norma)', pad, axisY + 76, { size: 11, color: C.muted, maxW: areaW });
        // taps fora da tela
        const offL = taps.filter((i) => i < 0);
        if (offL.length) V.text(g, `← +${offL.length} tap(s) além da tela`, pad, 22, { size: 11, color: RED() });
        // nó de saída
        const ox = cx(tIdx);
        const obx = V.clamp(ox - 34, pad, w - pad - 68 - (narrow ? 0 : infoW + 14));
        V.box(g, obx, outY - 14, 68, 28, { r: 8, fill: V.alpha(C.c, 0.2), stroke: C.c, lw: 1.5 });
        V.text(g, 'saída t', obx + 34, outY + 4, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        const ph = (t % 2.4) / 2.4;
        taps.forEach((i, j) => {
          if (i < 0) return;
          const x = cx(i);
          g.strokeStyle = V.alpha(C.d, 0.75); g.lineWidth = 1.6;
          g.beginPath(); g.moveTo(ox, outY + 14); g.bezierCurveTo(ox, outY + 60, x, axisY - 60, x, axisY - 2); g.stroke();
          const u = V.clamp(ph * S.k - j, 0, 1);
          if (u > 0 && u < 1) { const p = V.bez([x, axisY - 2], [x, axisY - 60], [ox, outY + 60], [ox, outY + 14], u); V.glow(g, p[0], p[1], 9, C.d); V.dot(g, p[0], p[1], 3, C.d); }
          if (S.d * cw >= 24 || j === 0 || j === S.k - 1) V.text(g, 'w' + j, (x + ox) / 2 + (x < ox ? -8 : 8), (axisY + outY) / 2 + (j % 2 ? 10 : -6), { size: 11, font: 'mono', color: C.d, align: 'center' });
        });
        // janelas de trigrama
        let covered = new Set();
        taps.forEach((i) => { for (let s = i - 2; s <= i; s++) covered.add(s); });
        if (S.tri) {
          const cols = [C.a, C.b, C.e, C.f, C.a, C.b];
          taps.forEach((i, j) => {
            const a = Math.max(0, i - 2), b = i; if (b < 0) return;
            const y = axisY + 44 + (j % 2) * 9;
            const x1 = pad + a * cw + 3, x2 = pad + (b + 1) * cw - 3;
            g.strokeStyle = cols[j % cols.length]; g.lineWidth = 2.2;
            g.beginPath(); g.moveTo(x1, y - 4); g.lineTo(x1, y); g.lineTo(x2, y); g.lineTo(x2, y - 4); g.stroke();
          });
          // buracos
          const lo = Math.min(...taps) - 2;
          for (let s = Math.max(0, lo); s <= tIdx; s++) if (!covered.has(s)) { g.fillStyle = V.alpha(RED(), 0.18); g.fillRect(pad + s * cw + 1.5, axisY, cw - 3, 22); }
          V.text(g, 'cada u_s resume o trigrama s−2..s', pad, axisY + 94, { size: 11, color: C.muted });
        }
        // painel
        const rf = (S.k - 1) * S.d + 1;
        const overlap = S.d < 3, gaps = S.d > 3;
        const lines = [
          ['campo receptivo', `(k−1)·d + 1 = ${rf} posições`],
          ['tokens crus cobertos', `${covered.size}${gaps ? ' (com buracos)' : overlap ? ' (janelas se sobrepõem)' : ' (encaixe perfeito)'}`],
          ['estado da conv', `(k−1)·d + spec = ${(S.k - 1) * S.d} + ${S.spec} = ${(S.k - 1) * S.d + S.spec} colunas`],
          ['por requisição', `× 10.240 canais × 2 B ≈ ${fmt(((S.k - 1) * S.d + S.spec) * 10240 * 2 / 1024, 0)} KiB`],
        ];
        if (!narrow) {
          const px = w - pad - infoW, py = 18;
          V.box(g, px, py, infoW, h - 36, { r: 12, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          let y = py + 26;
          lines.forEach(([a, b], i) => { V.text(g, a, px + 14, y, { size: 11, color: C.muted, font: 'mono' }); wrap(g, b, px + 14, y + 18, infoW - 28, 16, { size: 12.5, color: i === 1 ? (gaps ? RED() : overlap ? C.d : C.e) : C.ink, weight: 600 }); y += 58; });
          wrap(g, 'Qwen: k = 4, d = 3 = ngram_size, 10.240 canais (4 ramos × 2560), depthwise.', px + 14, y, infoW - 28, 15, { size: 11, color: C.muted });
        } else {
          let y = axisY + 122;
          lines.forEach(([a, b], i) => { V.text(g, a + ':', pad, y, { size: 11, color: C.muted, font: 'mono' }); V.text(g, b, pad, y + 16, { size: fitSize(g, b, w - 2 * pad, 12), color: i === 1 ? (gaps ? RED() : overlap ? C.d : C.e) : C.ink, weight: 600 }); y += 40; });
        }
      });
    })();

    /* ================================================== Fig 6: estado por requisição */
    (function fig6() {
      const fig = root.querySelector('#fig-estado'); if (!fig) return;
      adaptH(fig, 380, 400);
      const S = { sc: 'chunk', C: 5, acc: 2 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'chunk', label: 'chunked prefill' }, { v: 'eos', label: 'fronteira EOS' }, { v: 'spec', label: 'decode especulativo' }], value: 'chunk', onChange: (v) => { S.sc = v; } });
      UI.slider(ctl, { label: 'Tamanho do chunk', min: 3, max: 10, step: 1, value: 5, onInput: (v) => { S.C = v; } });
      UI.slider(ctl, { label: 'Rascunhos aceitos', min: 0, max: 3, step: 1, value: 2, onInput: (v) => { S.acc = v; } });
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C(), narrow = w < 620, pad = 14;
        const N = narrow ? 12 : 20;
        const cw = (w - 2 * pad) / N, ty = narrow ? 150 : 140, th = 28;
        const eosAt = narrow ? 5 : 9;
        let cur, draftStart = N - 3;
        const specPhase = (() => { const ph = (t % 5) / 5; return ph < 0.33 ? 0 : ph < 0.66 ? 1 : 2; })();
        if (S.sc === 'spec') cur = N - 1; else if (S.sc === 'eos') cur = eosAt + [1, 2, 3, 1, 2, 3, -1][Math.floor(t * 0.8) % 7]; else cur = (Math.floor(t * 1.2) + S.C) % N;
        const x = (i) => pad + i * cw;
        const isEOS = (i) => S.sc === 'eos' && i === eosAt;
        // cabeçalho
        const title = S.sc === 'chunk' ? `chunked prefill: pedaços de ${S.C} tokens` : S.sc === 'eos' ? 'dois documentos na mesma requisição' : `decode especulativo: último aceito + 3 rascunhos, ${S.acc} aceito(s)`;
        V.text(g, title, pad, 22, { size: 13, weight: 600, color: C.ink, maxW: w - 2 * pad });
        // chunks
        const chunkOf = (i) => Math.floor(i / S.C);
        if (S.sc === 'chunk') {
          for (let c = 0; c * S.C < N; c++) {
            const xa = x(c * S.C), xb = x(Math.min(N, (c + 1) * S.C));
            const active = chunkOf(cur) === c;
            V.box(g, xa + 1, ty - 56, xb - xa - 2, th + 64, { r: 8, fill: active ? V.alpha(C.c, 0.06) : 'transparent', stroke: V.alpha(active ? C.c : C.muted, active ? 0.6 : 0.25), lw: 1 });
            V.text(g, 'passo ' + (c + 1), (xa + xb) / 2, ty - 42, { size: 11, font: 'mono', align: 'center', color: active ? C.c : C.muted, maxW: xb - xa - 6 });
          }
        }
        // tokens
        for (let i = 0; i < N; i++) {
          const e = isEOS(i), isDraft = S.sc === 'spec' && i >= draftStart;
          const rejected = isDraft && (i - draftStart) >= S.acc && specPhase === 2;
          const past = S.sc === 'chunk' && chunkOf(i) < chunkOf(cur);
          let col = e ? C.d : i === cur && S.sc !== 'spec' ? C.c : isDraft ? (rejected ? RED() : C.e) : C.muted;
          const lab = e ? 'EOS' : isDraft ? 'd' + (i - draftStart + 1) : String(i);
          tokBox(g, x(i) + 2, ty, cw - 4, th, lab, col, { fill: V.alpha(col, past ? 0.05 : 0.14), eyes: false, size: 11, font: 'mono', lw: i === cur ? 2 : 1 });
          if (past) { g.fillStyle = V.alpha(C.bg, 0.35); V.rr(g, x(i) + 2, ty, cw - 4, th, 8); g.fill(); }
          if (rejected && S.sc === 'spec') V.line(g, x(i) + 6, ty + 4, x(i) + cw - 6, ty + th - 4, { color: RED(), lw: 2 });
        }
        if (S.sc === 'spec') V.text(g, 'último confirmado', x(draftStart - 1) + cw / 2, ty + th + 16, { size: 11, align: 'center', color: C.muted, maxW: cw * 2.4 });
        // arcos do token atual
        const drawArc = (from, to, col, above, dashed, label) => {
          const x1 = x(from) + cw / 2, x2 = x(to) + cw / 2, yb = above ? ty - 2 : ty + th + 2;
          const hh = (above ? -1 : 1) * (18 + Math.abs(from - to) * (narrow ? 5 : 4));
          g.save(); if (dashed) g.setLineDash([4, 4]);
          g.strokeStyle = col; g.lineWidth = 1.8; g.beginPath(); g.moveTo(x1, yb); g.quadraticCurveTo((x1 + x2) / 2, yb + hh, x2, yb); g.stroke(); g.restore();
          V.dot(g, x2, yb, 3, col);
          if (label) V.text(g, label, (x1 + x2) / 2, yb + hh * 0.55 + (above ? -2 : 10), { size: 11, align: 'center', color: col, font: 'mono' });
        };
        const stateUsed = { ngram: [], conv: [] };
        const segStart = (i) => { let p = -1; for (let j = 0; j < i; j++) if (isEOS(j)) p = j; return p; };
        if (S.sc !== 'spec') {
          [1, 2].forEach((s) => {
            const tgt = cur - s; const col = s === 1 ? C.a : C.b;
            if (tgt < 0) return;
            const crossEOS = S.sc === 'eos' && (cur - segStart(cur) - 1) < s;
            if (crossEOS && !isEOS(tgt)) { // fantasma EOS acima do alvo
              const gx = x(tgt) + 2, gy = ty - 52 - s * 12;
              tokBox(g, gx, gy, cw - 4, 20, 'EOS', col, { dash: true, eyes: false, size: 11, font: 'mono', fill: V.alpha(col, 0.08) });
              V.arrow(g, x(cur) + cw / 2, ty - 2, gx + (cw - 4) / 2, gy + 20, { color: col, lw: 1.6, head: 6, curve: 12 });
              return;
            }
            const fromState = S.sc === 'chunk' && chunkOf(tgt) < chunkOf(cur);
            if (fromState) stateUsed.ngram.push(tgt);
            drawArc(cur, tgt, col, true, false, fromState && (s === 2 || cur - 2 < 0 || chunkOf(cur - 2) === chunkOf(cur)) ? 'do ngram_context' : '');
          });
          if (S.sc === 'chunk') [3, 6, 9].forEach((s) => { const tgt = cur - s; if (tgt < 0) return; const fromState = chunkOf(tgt) < chunkOf(cur); if (fromState) stateUsed.conv.push(tgt); drawArc(cur, tgt, V.alpha(C.d, 0.9), false, true, fromState && s === 3 ? 'do conv_state' : ''); });
          if (S.sc === 'eos') {
            const pis = cur - segStart(cur) - 1;
            const nm = (i) => (i < 0 ? 'EOS' : isEOS(i) ? 'EOS' : 'x' + i);
            const v1 = pis >= 1 ? nm(cur - 1) : 'EOS', v2 = pis >= 2 ? nm(cur - 2) : 'EOS';
            const yy = ty + th + 46;
            V.box(g, pad, yy - 18, w - 2 * pad, narrow ? 84 : 66, { r: 10, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.3), lw: 1 });
            V.text(g, `token ${cur}: bigrama (${v1}, x${cur}) · trigrama (${v2}, ${v1}, x${cur})`, pad + 12, yy + 2, { size: narrow ? 11 : 12.5, font: 'mono', color: C.ink, maxW: w - 2 * pad - 24 });
            const raw2 = cur - 2 >= 0 ? nm(cur - 2) : 'EOS', raw1 = cur - 1 >= 0 ? nm(cur - 1) : 'EOS';
            const leak = raw2 !== v2 || raw1 !== v1;
            wrap(g, leak ? `sem a segmentação seria (${raw2}, ${raw1}, x${cur}): o fim do documento anterior vazaria para o novo.` : 'aqui os vizinhos já estão no mesmo documento: nada muda.', pad + 12, yy + 22, w - 2 * pad - 24, 15, { size: 11.5, color: leak ? C.d : C.muted });
            V.text(g, 'N-gramas não atravessam o EOS: o vizinho de antes da fronteira vira o id 248044', pad, h - 20, { size: 11.5, color: C.muted, maxW: w - 2 * pad });
          }
        }
        // caixas de estado
        const bxY = narrow ? ty + th + 58 : ty + th + 58;
        if (S.sc === 'chunk') {
          const boxW = narrow ? w - 2 * pad : (w - 2 * pad - 16) / 2;
          const b1 = { x: pad, y: bxY }, b2 = narrow ? { x: pad, y: bxY + 70 } : { x: pad + boxW + 16, y: bxY };
          const lastPrev = chunkOf(cur) * S.C - 1;
          V.box(g, b1.x, b1.y, boxW, 58, { r: 10, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, stateUsed.ngram.length ? 0.9 : 0.35), lw: stateUsed.ngram.length ? 1.8 : 1 });
          V.text(g, 'ngram_context (2 ids / requisição)', b1.x + 10, b1.y + 17, { size: 11, font: 'mono', color: C.a, maxW: boxW - 20 });
          [lastPrev - 1, lastPrev].forEach((id, j) => { const ok = id >= 0 && chunkOf(cur) > 0; V.box(g, b1.x + 10 + j * 44, b1.y + 26, 38, 22, { r: 6, fill: V.alpha(C.a, ok ? 0.25 : 0.06), stroke: V.alpha(C.a, 0.6), lw: 1 }); V.text(g, ok ? String(id) : '—', b1.x + 29 + j * 44, b1.y + 41, { size: 11, font: 'mono', align: 'center', color: C.ink }); });
          V.box(g, b2.x, b2.y, boxW, 58, { r: 10, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, stateUsed.conv.length ? 0.9 : 0.35), lw: stateUsed.conv.length ? 1.8 : 1 });
          V.text(g, 'conv_state (9 colunas × 10.240 canais)', b2.x + 10, b2.y + 17, { size: 11, font: 'mono', color: C.d, maxW: boxW - 20 });
          const sw = Math.min(22, (boxW - 20) / 9 - 3);
          for (let j = 0; j < 9; j++) { const id = lastPrev - 8 + j, ok = id >= 0 && chunkOf(cur) > 0; V.box(g, b2.x + 10 + j * (sw + 3), b2.y + 26, sw, 22, { r: 4, fill: V.alpha(C.d, ok ? (stateUsed.conv.includes(id) ? 0.55 : 0.22) : 0.05), stroke: V.alpha(C.d, 0.5), lw: 1 }); }
        }
        if (S.sc === 'spec') {
          const phase = specPhase;
          const msgs = ['1 · o MTP propõe 3 rascunhos', '2 · o alvo verifica 1 + 3 tokens; a PLE roda nos 4', `3 · rollback: a janela volta ${3 - S.acc} posição(ões) e termina no último aceito`];
          V.text(g, msgs[phase], pad, ty - 40, { size: 12, color: C.c, weight: 600, maxW: w - 2 * pad });
          // estado de 12 colunas
          const cols = 12, sw = Math.min(30, (w - 2 * pad - 20) / cols - 3), bx0 = pad, by0 = bxY;
          V.text(g, 'conv_state = 9 + 3 colunas (uma por rascunho)', bx0, by0 - 8, { size: 11, font: 'mono', color: C.d });
          const keepEnd = 8 + S.acc; // índice da última coluna válida após rollback
          for (let j = 0; j < cols; j++) {
            const inWin = phase < 2 ? j >= 3 && j <= 11 : j >= keepEnd - 8 && j <= keepEnd;
            const isSpec = j >= 9;
            const col = isSpec ? (phase === 2 && j > keepEnd ? RED() : C.e) : C.d;
            V.box(g, bx0 + j * (sw + 3), by0, sw, 26, { r: 5, fill: V.alpha(col, inWin ? 0.45 : 0.08), stroke: V.alpha(col, 0.6), lw: 1 });
            V.text(g, isSpec ? 'd' + (j - 8) : String(j - 9), bx0 + j * (sw + 3) + sw / 2, by0 + 17, { size: 11, font: 'mono', align: 'center', color: C.ink });
          }
          const wx0 = bx0 + (phase < 2 ? 3 : keepEnd - 8) * (sw + 3) - 2, wx1 = bx0 + ((phase < 2 ? 11 : keepEnd) + 1) * (sw + 3) - 1;
          V.box(g, wx0, by0 - 4, wx1 - wx0, 34, { r: 7, stroke: C.c, lw: 2 });
          V.text(g, 'janela de 9 usada no próximo passo', (wx0 + wx1) / 2, by0 + 48, { size: 11, align: 'center', color: C.c });
          wrap(g, 'deslocamento = num_accepted_tokens − 1 (igual ao causal_conv1d_update das GDN)', pad, by0 + 70, w - 2 * pad, 15, { size: 11, color: C.muted });
        }
      });
    })();

    /* ================================================== Fig 7: formatos da linha */
    (function fig7() {
      const fig = root.querySelector('#fig-formatos'); if (!fig) return;
      adaptH(fig, 430, 540);
      const S = { fmt: 'nvfp4', hi: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Formato', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8 (por tensor)' }, { v: 'nvfp4', label: 'NVFP4 (blocos de 16)' }], value: 'nvfp4', onChange: (v) => { S.fmt = v; setErr(); } });
      const rE = UI.readout(ctl, 'erro RMS relativo (linha)', '—');
      const r = V.rng(2024);
      const row = Array.from({ length: 160 }, (_, i) => r.normal() * 0.5 * (1 + 0.6 * Math.sin(i / 9)) + (r() < 0.03 ? (r() < 0.5 ? -1 : 1) * 2.4 : 0));
      const q = { bf16: row.map((x) => LA.num.encode('bf16', x).v), fp8: LA.num.fp8Tensor(row), nvfp4: LA.num.nvfp4(row) };
      const rms = Math.sqrt(row.reduce((a, x) => a + x * x, 0) / row.length);
      const relErr = (deq) => Math.sqrt(row.reduce((a, x, i) => a + (deq[i] - x) ** 2, 0) / row.length) / rms;
      const errs = { bf16: relErr(q.bf16), fp8: relErr(q.fp8.deq), nvfp4: relErr(q.nvfp4.deq) };
      const setErr = () => rE.set((errs[S.fmt] * 100).toFixed(S.fmt === 'bf16' ? 2 : 1).replace('.', ',') + '%');
      setErr();
      const amax = Math.max(...row.map(Math.abs));
      // índices de exemplo: códigos NVFP4 com magnitudes variadas (evita mostrar zeros)
      const picks = (() => {
        const want = [6, 3, 1.5, 4], out = [];
        want.forEach((mg) => { const i = q.nvfp4.codes.findIndex((c, k) => Math.abs(c.v) === mg && !out.includes(k)); if (i >= 0) out.push(i); });
        return out.length ? out : [0];
      })();
      const FMT = {
        bf16: { name: 'BF16', rowB: 320, total: 95.37, gpu: 47.68, tok: 5120, bits: 16 },
        fp8: { name: 'FP8 E4M3 + 1 escala', rowB: 160, total: 47.68, gpu: 23.84, tok: 2560, bits: 8 },
        nvfp4: { name: 'NVFP4 empacotado', rowB: 90, total: 26.82, gpu: 13.41, tok: 1440, bits: 4.5 },
      };
      let geo = null;
      const H = V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C(), narrow = w < 620, pad = narrow ? 10 : 16, F = S.fmt;
        // índice em destaque
        let hi = S.hi;
        if (geo && H.pointer.inside) { const p = H.pointer; const k = geo.cellAt(p.x, p.y); if (k != null) hi = k; }
        if (hi == null) hi = picks[Math.floor(t / 1.8) % picks.length];
        // ---- tira de valores
        const perRow = narrow ? 32 : 80, rowsN = 160 / perRow, bgap = narrow ? 4 : 5;
        const blocksPerRow = perRow / 16;
        const cw = (w - 2 * pad - bgap * (blocksPerRow - 1)) / perRow, ch = narrow ? 14 : 16;
        const y0 = 26;
        V.text(g, 'uma linha da tabela: 160 valores (10 blocos de 16)', pad, 16, { size: 11.5, color: C.muted });
        const cellXY = (i) => { const rI = Math.floor(i / perRow), c = i % perRow, blk = Math.floor(c / 16); return [pad + c * cw + blk * bgap, y0 + rI * (ch + 6)]; };
        const deq = F === 'bf16' ? q.bf16 : q[F].deq;
        for (let i = 0; i < 160; i++) {
          const [x, y] = cellXY(i);
          g.fillStyle = V.cmap(V.clamp(deq[i] / (amax * 0.6), -1, 1), 'div'); V.rr(g, x, y, cw - 1, ch, 2); g.fill();
          if (i === hi) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, cw + 2, ch + 3, 3); g.stroke(); }
        }
        geo = { cellAt: (px, py) => { for (let rI = 0; rI < rowsN; rI++) { const yy = y0 + rI * (ch + 6); if (py < yy - 2 || py > yy + ch + 2) continue; for (let c = 0; c < perRow; c++) { const i = rI * perRow + c; const [x] = cellXY(i); if (px >= x && px < x + cw) return i; } } return null; } };
        let y = y0 + rowsN * (ch + 6) + 18;
        // ---- bytes guardados
        const full = w - 2 * pad, unit = full / 320;
        V.text(g, 'bytes guardados por linha (régua: 320 B = BF16)', pad, y, { size: 11.5, color: C.muted });
        y += 8;
        const bh = narrow ? 24 : 28;
        g.fillStyle = V.alpha(C.muted, 0.08); V.rr(g, pad, y, full, bh, 6); g.fill();
        if (F === 'bf16') {
          for (let i = 0; i < 160; i++) { g.fillStyle = V.alpha(C.d, i % 2 ? 0.45 : 0.6); g.fillRect(pad + i * 2 * unit, y, 2 * unit - 0.5, bh); }
          V.text(g, '160 × 2 B = 320 B', pad + 8, y + bh / 2 + 4, { size: 12, weight: 600, color: onCol() });
        } else if (F === 'fp8') {
          for (let i = 0; i < 160; i++) { g.fillStyle = V.alpha(C.a, i % 2 ? 0.5 : 0.7); g.fillRect(pad + i * unit, y, unit - 0.4, bh); }
          V.text(g, '160 × 1 B = 160 B', pad + 8, y + bh / 2 + 4, { size: 12, weight: 600, color: onCol() });
          // caixa dimensionada pelo texto (e pelo espaço que sobra à direita)
          const room = full - 168 * unit - 2, sTxt = tw(g, '+ 1 escala BF16 p/ tabela toda', 11) + 18 <= room ? '+ 1 escala BF16 p/ tabela toda' : '+ 1 escala BF16 (tabela)';
          const sbw = Math.min(room, tw(g, sTxt, 11) + 18);
          V.box(g, pad + 168 * unit, y + 3, sbw, bh - 6, { r: 5, fill: V.alpha(C.b, 0.2), stroke: C.b, lw: 1 });
          V.text(g, sTxt, pad + 168 * unit + 9, y + bh / 2 + 4, { size: 11, color: C.ink, maxW: sbw - 14 });
        } else {
          for (let i = 0; i < 80; i++) { const xx = pad + i * unit; g.fillStyle = V.alpha(C.a, 0.7); g.fillRect(xx, y, unit / 2 - 0.3, bh); g.fillStyle = V.alpha(C.a, 0.45); g.fillRect(xx + unit / 2, y, unit / 2 - 0.3, bh); }
          for (let i = 0; i < 10; i++) { g.fillStyle = V.alpha(C.b, 0.8); g.fillRect(pad + (80 + i) * unit, y, unit - 0.4, bh); }
          if (narrow) { V.text(g, '80 B códigos', pad, y + bh + 14, { size: 11, color: C.a, font: 'mono' }); V.text(g, '+ 10 B escalas', pad + tw(g, '80 B códigos', 11, 'mono') + 8, y + bh + 14, { size: 11, color: C.b, font: 'mono' }); }
          else { V.text(g, '80 B códigos', pad + 4, y + bh + 14, { size: 11, color: C.a, font: 'mono' }); V.text(g, '+10 B escalas', pad + 80 * unit, y + bh + 14, { size: 11, color: C.b, font: 'mono' }); }
          V.text(g, '= 90 B (+ escala global FP32)', pad + 92 * unit + 4, y + bh / 2 + 4, { size: 11.5, weight: 600, color: C.ink, maxW: full - 92 * unit - 8 });
        }
        y += bh + (F === 'nvfp4' ? 34 : 22);
        // ---- detalhe do valor em destaque
        const x = row[hi];
        let det;
        if (F === 'bf16') det = [`valor ${hi}: ${fmt(x, 4)} → BF16 ${fmt(q.bf16[hi], 4)}`, '8 bits de expoente + 7 de mantissa, sem escala'];
        else if (F === 'fp8') { const sc = q.fp8.scale, e = LA.num.encode('e4m3', x / sc); det = narrow ? [`valor ${hi}: ${fmt(x, 4)} ÷ escala ${fmt(sc, 5)}`, `→ E4M3 0b${LA.num.toBin(e.bits, 8)} = ${fmt(e.v, 3)}`, `dequant: ${fmt(e.v, 3)} × ${fmt(sc, 5)} = ${fmt(q.fp8.deq[hi], 4)}`] : [`valor ${hi}: ${fmt(x, 4)} ÷ escala ${fmt(sc, 5)} → E4M3 0b${LA.num.toBin(e.bits, 8)} = ${fmt(e.v, 3)}`, `dequant: ${fmt(e.v, 3)} × ${fmt(sc, 5)} = ${fmt(q.fp8.deq[hi], 4)}`]; }
        else {
          const nv = q.nvfp4, blk = nv.blocks[Math.floor(hi / 16)], code = nv.codes[hi], bits = code.bits;
          const lut = [0, 0.5, 1, 1.5, 2, 3, 4, 6], mag = lut[bits & 7], sgn = (bits >> 3) & 1;
          det = narrow ? [`valor ${hi}: ${fmt(x, 4)} · nibble 0b${LA.num.toBin(bits, 4)}`, `LUT[${bits & 7}] = ${sgn ? '−' : ''}${fmt(mag, 1)} × bloco ${fmt(blk.scaleE4M3, 2)} × global ${fmt(nv.globalScale, 5)}`, `= ${fmt(nv.deq[hi], 4)}`]
            : [`valor ${hi}: ${fmt(x, 4)} · nibble 0b${LA.num.toBin(bits, 4)} → LUT[${bits & 7}] = ${sgn ? '−' : ''}${fmt(mag, 1)}`, `× escala do bloco ${Math.floor(hi / 16)} (E4M3 ${fmt(blk.scaleE4M3, 2)}) × global ${fmt(nv.globalScale, 5)} = ${fmt(nv.deq[hi], 4)}`];
        }
        const dh = 10 + det.length * 17;
        V.box(g, pad, y - 14, full, dh, { r: 8, fill: V.alpha(C.c, 0.07), stroke: V.alpha(C.c, 0.5), lw: 1 });
        det.forEach((s, i) => V.text(g, s, pad + 10, y + 2 + i * 17, { size: fitSize(g, s, full - 20, narrow ? 10.5 : 11.5, 'mono'), font: 'mono', color: i ? C.muted : C.ink }));
        y += dh + 4;
        // ---- a conta da tabela inteira
        const metrics = [
          ['tabela inteira', (k) => FMT[k].total, (v) => fmt(v, 1) + ' GiB', 95.37],
          ['por GPU (TP=2)', (k) => FMT[k].gpu, (v) => fmt(v, 1) + ' GiB', 47.68],
          ['lido por token (16 linhas)', (k) => FMT[k].tok, (v) => fmtInt(v) + ' B', 5120],
          ['bits por parâmetro', (k) => FMT[k].bits, (v) => fmt(v, v % 1 ? 1 : 0), 16],
        ];
        const keys = ['bf16', 'fp8', 'nvfp4'], kcol = { bf16: C.d, fp8: C.a, nvfp4: C.b };
        const keysL = [['bf16', 'BF16'], ['fp8', 'FP8'], ['nvfp4', 'NVFP4']];
        let lx = pad;
        keysL.forEach(([k, nm]) => { g.fillStyle = kcol[k]; V.rr(g, lx, y - 9, 11, 11, 3); g.fill(); V.text(g, nm, lx + 16, y, { size: 11, color: k === F ? C.ink : C.muted, weight: k === F ? 600 : 400 }); lx += tw(g, nm, 11) + 34; });
        y += 12;
        const labW = narrow ? 0 : 170, avail = h - y - 6;
        const mh = Math.min(narrow ? 62 : 44, avail / metrics.length);
        metrics.forEach(([lb, get, f, mx], mi) => {
          const yy = y + mi * mh;
          V.text(g, lb, pad, yy + (narrow ? 10 : 16), { size: 11, color: C.muted });
          const bx0 = pad + labW, bwid = full - labW - (narrow ? 0 : 0);
          keys.forEach((k, j) => {
            const v = get(k), bw = Math.max(3, (v / mx) * (bwid - 70)), by = yy + (narrow ? 16 : 3) + j * (narrow ? 14 : 12.5);
            const on = k === F;
            g.fillStyle = V.alpha(kcol[k], on ? 0.85 : 0.25); V.rr(g, bx0, by, bw, narrow ? 11 : 10, 3); g.fill();
            V.text(g, f(v), bx0 + bw + 6, by + (narrow ? 10 : 9.5), { size: 11, font: 'mono', color: on ? C.ink : C.muted });
          });
        });
      });
    })();

    /* ================================================== Fig 8: dispatch */
    (function fig8() {
      const fig = root.querySelector('#fig-dispatch'); if (!fig) return;
      const LBL = {
        cfg: { label: 'config.json\ndo snapshot' },
        det: { label: 'detect_ple_dtype.py\nlê config_groups' },
        ovr: { label: '--hf-overrides\ntext_config' },
        sw: { label: 'ple_layer_patched\ndispatch do método' },
        nv: { label: 'método NVFP4\n90 B/linha', term: 'ok' },
        fp8: { label: 'método FP8\n160 B/linha + 1 escala', term: 'ok' },
        bfx: { label: 'nenhum método\n→ BF16', term: 'bad' },
        skip: { label: 'SKIP_PLE_PATCH=true\nresolver original' },
        f8c: { label: 'ramo Fp8Config\nmétodo FP8', term: 'ok' },
        sw0: { label: 'resolver original\nsó Fp8Config' },
        bf: { label: 'BF16 ≈102 GB\nOOM no load ✕', term: 'bad' },
        v30: { label: 'start-v030.sh\nsem patch' },
        nat: { label: 'qwen4_exp do 0.30\nloader nativo', term: 'ok' },
      };
      const CK = {
        nvidia: { label: 'nvidia NVFP4 (receita)', path: ['cfg', 'det', 'ovr', 'sw', 'fp8'], info: ['A tabela FP8 só aparece em config_groups.group_2 (num_bits 8, alvo …ple_embedding.ngram_embedding).', 'detect_ple_dtype.py imprime float8_e4m3fn e o start.sh injeta {"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}.', 'Método FP8: 128 shards [2.500.012 × 160] + 1 weight_scale BF16, ~23,8 GiB por GPU.'] },
        lil: { label: 'local-inference-lab NVFP4', path: ['cfg', 'det', 'sw', 'nv'], info: ['O text_config já declara ple_embedding_dtype = "nvfp4".', 'detect_ple_dtype.py não imprime nada: não há override de PLE.', 'Método NVFP4 novo do patch: 80 B de códigos + 10 B de escalas por linha, 26,8 GiB no total.'] },
        radix: { label: 'RadixArk NVFP4', path: ['cfg', 'det', 'sw', 'fp8'], info: ['O text_config já declara ple_embedding_dtype = "float8_e4m3fn".', 'detect_ple_dtype.py não imprime nada.', 'Método FP8 (tabela FP8 por tensor, como a da nvidia).'] },
        fp8: { label: 'Qwen FP8 oficial', path: ['cfg', 'skip', 'f8c'], info: ['Checkpoint FP8 inteiro: o quantization_config vira um Fp8Config.', 'SKIP_PLE_PATCH=true: o ple_layer.py original da imagem basta.', 'O ramo Fp8Config do resolver original escolhe o método FP8.'] },
        stock: { label: 'nvidia sem o patch', path: ['cfg', 'sw0', 'bf'], info: ['O resolver original só escolhe FP8 se o checkpoint inteiro for Fp8Config.', 'MIXED_PRECISION com a PLE excluída do NVFP4 → nenhum método.', 'Embedding BF16 de 320 M × 160 × 2 B ≈ 102 GB: o load morre.'] },
        v030: { label: 'lane vLLM 0.30', path: ['cfg', 'v30', 'nat'], info: ['O start-v030.sh usa SKIP_PLE_PATCH=true e zera PLE_EMBEDDING_DTYPE.', 'O modelo qwen4_exp do vLLM 0.30 carrega a PLE nativamente.', 'Nesta lane há só 1 overlay e o KV é BF16 por padrão.'] },
      };
      const S = { ck: 'nvidia', off: false, t0: 0 };
      let clock = 0;
      // no celular a altura segue o conteúdo: nº de estações + caixa de offload + painel de texto
      const infoH = (g, ck, iw, narrow) => 14 + ck.info.reduce((a, ln, i) => a + wrapLines(g, (i + 1) + '. ' + ln, iw - 24, narrow ? 11 : 11.5).length * 15 + 4, 0);
      const narrowTop = (ck) => 92 + (ck.path.length - 1) * 56 + 42 + (S.off ? 64 : 0);
      const setH = adaptH(fig, 460, (wd) => narrowTop(CK[S.ck]) + 16 + infoH(measureCtx, CK[S.ck], wd - 20, true) + 12);
      const ctl = UI.controls(fig);
      const selC = UI.select(ctl, { label: 'Checkpoint', options: Object.keys(CK).map((k) => ({ v: k, label: CK[k].label })), value: 'nvidia', onChange: (v) => { S.ck = v; S.t0 = clock; setH(); } });
      selC.el.style.maxWidth = '100%'; selC.el.style.flexWrap = 'wrap'; const selEl = selC.el.querySelector('select'); if (selEl) selEl.style.maxWidth = '100%';
      UI.toggle(ctl, { label: 'PLE_OFFLOAD=true', value: false, onChange: (v) => { S.off = v; setH(); } });
      function posWide(w, h) {
        const c1 = 78, c5 = w - 108, gapc = (c5 - c1) / 4;
        const cx = (i) => c1 + i * gapc;
        const r = [28, 80, 140, 200, 256, 308];
        return {
          cfg: [cx(0), r[1]], det: [cx(1), r[1]], ovr: [cx(2), r[1]], sw: [cx(3) - 20, r[2]],
          nv: [c5, r[1]], fp8: [c5, r[2]], bfx: [c5, r[3]],
          skip: [cx(1), r[3]], f8c: [cx(2), r[3]], sw0: [cx(2), r[4]], bf: [c5, r[4]],
          v30: [cx(1), r[5]], nat: [cx(2), r[5]],
          ghost: [(cx(2) + cx(3)) / 2 + 20, 26], off: [c5 - 32, r[5] + 4], c5,
          bw: Math.min(140, gapc - 24), tw: 170,
        };
      }
      function drawEdge(g, a, b, col, lw, dashed) {
        g.save(); if (dashed) g.setLineDash([5, 5]);
        g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
        if (Math.abs(a[1] - b[1]) < 2 || Math.abs(a[0] - b[0]) < 2) { g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); }
        else { const mx = (a[0] + b[0]) / 2; g.moveTo(a[0], a[1]); g.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]); }
        g.stroke(); g.restore();
      }
      function station(g, x, y, bw, bh, key, lit, C) {
        const L = LBL[key], term = L.term, col = term === 'bad' ? RED() : term ? C.e : C.b;
        V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: term ? 8 : bh / 2, fill: C.bg });
        V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: term ? 8 : bh / 2, fill: V.alpha(col, lit ? 0.22 : 0.05), stroke: V.alpha(col, lit ? 1 : 0.3), lw: lit ? 2 : 1 });
        const lines = L.label.split('\n');
        lines.forEach((ln, i) => V.text(g, ln, x, y + (i - (lines.length - 1) / 2) * 14 + 4, { size: fitSize(g, ln, bw - 14, i === 0 ? 11.5 : 11, i === 0 ? 'mono' : 'sans', i === 0 ? 600 : 400), font: i === 0 ? 'mono' : 'sans', align: 'center', color: lit ? C.ink : C.muted, weight: i === 0 ? 600 : 400, maxW: bw - 8 }));
      }
      function ghost(g, x, y, gw, C) {
        V.box(g, x - gw / 2, y - 14, gw, 28, { r: 14, fill: C.bg });
        V.box(g, x - gw / 2, y - 14, gw, 28, { r: 14, fill: V.alpha(C.muted, 0.08) });
        g.save(); g.setLineDash([3, 3]); V.rr(g, x - gw / 2, y - 14, gw, 28, 14); g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.stroke(); g.restore();
        V.text(g, 'PLE_QUANT_OVERRIDE=fp8', x, y + 4, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: gw - 12 });
        V.line(g, x - gw / 2 + 8, y, x + gw / 2 - 8, y, { color: V.alpha(RED(), 0.85), lw: 1.6 });
        V.text(g, 'só no README: nenhum arquivo lê', x, y + 28, { size: 11, align: 'center', color: C.muted });
      }
      function offBox(g, x, y, ow, C) {
        V.box(g, x - ow / 2, y - 26, ow, 52, { r: 10, fill: C.bg });
        V.box(g, x - ow / 2, y - 26, ow, 52, { r: 10, fill: V.alpha(C.d, 0.14), stroke: C.d, lw: 1.5 });
        ['processo CPU (VLLM_PLE_CPU_OFFLOAD=1)', 'handshake no host (sem stream mem ops)', 'UMA: mesma RAM, sem economia física'].forEach((ln, i) => V.text(g, ln, x, y - 10 + i * 14, { size: 11, align: 'center', color: i === 2 ? C.d : C.ink, maxW: ow - 12 }));
      }
      function train(g, pts, t, C) {
        const segs = pts.length - 1; if (segs < 1) return;
        const u = ((t - S.t0) * 0.35) % 1, f = u * segs, i = Math.min(segs - 1, Math.floor(f)), tt = V.ease.inOut(f - i);
        const a = pts[i], b = pts[i + 1];
        let p;
        if (Math.abs(a[1] - b[1]) < 2 || Math.abs(a[0] - b[0]) < 2) p = V.along(a[0], a[1], b[0], b[1], tt);
        else { const mx = (a[0] + b[0]) / 2; p = V.bez(a, [mx, a[1]], [mx, b[1]], b, tt); }
        V.glow(g, p[0], p[1], 14, C.c); V.dot(g, p[0], p[1], 5, C.c);
      }
      function info(g, x, y, iw, ih, ck, narrow, C) {
        V.box(g, x, y, iw, ih, { r: 10, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let yy = y + 18;
        ck.info.forEach((ln, i) => { const n = wrap(g, (i + 1) + '. ' + ln, x + 12, yy, iw - 24, 15, { size: narrow ? 11 : 11.5, color: i === 2 ? (S.ck === 'stock' ? RED() : C.ink) : C.muted }); yy += n * 15 + 4; });
      }
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        clock = t;
        const C = V.C(), narrow = w < 620, ck = CK[S.ck], on = new Set(ck.path);
        if (!narrow) {
          const P = posWide(w, h), bh = 40;
          const edges = [['cfg', 'det'], ['det', 'ovr'], ['ovr', 'sw'], ['det', 'sw'], ['sw', 'nv'], ['sw', 'fp8'], ['sw', 'bfx'], ['cfg', 'skip'], ['skip', 'f8c'], ['cfg', 'sw0'], ['sw0', 'bf'], ['cfg', 'v30'], ['v30', 'nat']];
          const edgeOn = (a, b) => { const i = ck.path.indexOf(a); return i >= 0 && ck.path[i + 1] === b; };
          edges.forEach(([a, b]) => { const eo = edgeOn(a, b); drawEdge(g, P[a], P[b], eo ? C.c : V.alpha(C.muted, 0.28), eo ? 3 : 1.4, !eo); });
          ghost(g, P.ghost[0], P.ghost[1], 210, C);
          Object.keys(LBL).forEach((k) => { const bw = P[k][0] === P.c5 ? P.tw : P.bw; station(g, P[k][0], P[k][1], bw, bh, k, on.has(k), C); });
          train(g, ck.path.map((k) => P[k]), t, C);
          if (S.off) offBox(g, P.off[0], P.off[1], 250, C);
          info(g, 12, h - 98, w - 24, 88, ck, false, C);
        } else {
          const bw = w - 56, bh = 40, x = w / 2, y0 = 92, dy = 56;
          ghost(g, x, 22, Math.min(230, w - 28), C);
          const pts = ck.path.map((k, i) => [x, y0 + i * dy]);
          for (let i = 0; i < pts.length - 1; i++) drawEdge(g, pts[i], pts[i + 1], C.c, 3, false);
          ck.path.forEach((k, i) => station(g, x, pts[i][1], LBL[k].term ? bw : bw - 30, bh, k, true, C));
          train(g, pts, t, C);
          let yy = narrowTop(ck) - (S.off ? 64 : 0);
          if (S.off) { offBox(g, x, yy + 26, w - 28, C); if (S.ck !== 'stock') { const last = pts[pts.length - 1]; drawEdge(g, [x, last[1] + bh / 2 + 2], [x, yy], C.d, 2, true); } yy += 64; }
          V.text(g, 'outros checkpoints seguem outros caminhos (troque acima)', 14, yy + 4, { size: 11, color: C.muted, maxW: w - 28 });
          info(g, 10, yy + 16, w - 20, infoH(g, ck, w - 20, true), ck, true, C);
        }
      });
    })();

    /* ================================================== Fig 9: anatomia do MTP */
    (function fig9() {
      const fig = root.querySelector('#fig-mtp'); if (!fig) return;
      adaptH(fig, 480, 704);
      const M = {
        glm: { name: 'GLM-5.3-Flash', col: 'glm', trunk: '45 camadas · mHC 4×4096', hidden: 'h(t) · 4096 (contraído)', emb: 'emb(x(t+1)) · 4096', norms: 'enorm · hnorm', comb: 'concat', cat: '[ e ; h ] · 2d = 8192', proj: 'eh_proj [4096 × 8192]', block: ['MLA NoPE + DSA', 'MoE 288 (top-8) · sem mHC', 'num_nextn_predict_layers: 1'], head: 'shared_head.norm → lm_head', where: 'camada 45', k: 'k = 2 (SPEC_METHOD=mtp)' },
        qwen: { name: 'Qwen3.8-Flash-Next', col: 'qwen', trunk: '48 camadas · GR 4×2560', hidden: 'R(t) · 4 ramos (10240)', emb: 'emb(x(t+1)) · 2560', norms: 'pre_fc_norm_emb · _hidden', comb: 'dois', cat: 'combinação exata: ponto em aberto', proj: 'fc_embedding · fc_hidden (2560×2560 cada)', block: ['QSA (full_attention)', 'MoE 512 · FP8 128×128', 'GR + mixer próprios · KV próprio'], head: 'lm_head (248.320)', where: 'mtp.layers.0 (vLLM: .48)', k: 'k = 3 (MTP=3)' },
        ds: { name: 'DeepSeek-V4-Flash', col: 'ds', trunk: '43 camadas · mHC 4×4096', hidden: 'H(t) · 4 fluxos (16384)', emb: 'emb(x(t+1)) · 4096', norms: 'duas RMSNorm', comb: 'aberto', cat: 'combinação exata não inventariada', proj: 'projeção (formas não inventariadas)', block: ['preview: MTP-1 clássico', 'buffer 256 MiB/rank', '0731/V-Exp: mtp.0..2 = DSpark'], head: 'lm_head compartilhado', where: 'preview: MTP-1', k: 'hoje: DSpark (cap. 37)' },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; } });
      const steps = [
        { t: 'O tronco termina em \\(\\mathbf h_t\\)', d: 'O modelo principal processa o prefixo e produz, na posição \\(t\\), o estado \\(\\mathbf h_t\\). A cabeça principal (<code>lm_head</code>) prevê \\(x_{t+1}\\). Esse é o modelo normal, sem MTP.' },
        { t: 'Duas entradas normalizadas', d: 'O módulo MTP recebe a embedding do token seguinte, \\(\\mathrm{Emb}(x_{t+1})\\) (no treino, o token verdadeiro; no rascunho, o recém-proposto), e o estado \\(\\mathbf h_t\\). Cada um passa por sua RMSNorm. Troque o modelo: o estado consumido tem 4096, 10.240 ou 16.384 números.' },
        { t: 'Combinar e projetar', d: 'No GLM (a receita do DeepSeek-V3), os dois vetores são concatenados (\\(2d\\) = 8192) e o <code>eh_proj</code>, \\(W \\in \\mathbb R^{4096 \\times 8192}\\), volta a \\(d\\) = 4096. O Qwen tem duas projeções separadas, <code>fc_embedding</code> e <code>fc_hidden</code>, de 2560×2560 cada, e como os 10.240 números dos 4 ramos viram 2560 não está visível no repo nem no transformers (ponto em aberto). No DeepSeek a combinação exata não foi inventariada. A figura só desenha o que se sabe.' },
        { t: 'Um bloco Transformer inteiro', d: 'O resultado atravessa um bloco completo: atenção (com KV próprio) e MoE. No GLM é MLA NoPE + DSA + 288 experts, sem mHC. No Qwen é QSA + 512 experts FP8 em blocos 128×128. É isso que faz o MTP pesar ~7,4 B no GLM. No Qwen são 2,61 B próprios, ≈4 B contando a embedding e o <code>lm_head</code> compartilhados.' },
        { t: 'A mesma cabeça do modelo', d: 'Uma norma e o <code>lm_head</code> <b>compartilhado</b> com o modelo principal (e a mesma embedding) produzem a distribuição de \\(x_{t+2}\\). No treino, a loss dessa previsão é somada à principal com peso \\(\\lambda/D\\).' },
        { t: 'Reaplicar para rascunhar', d: 'Na inferência, a saída do módulo realimenta o próprio módulo com a embedding do token que ele acabou de propor: \\(\\tilde x_{t+2}, \\tilde x_{t+3}, \\dots\\) até \\(k\\) rascunhos. O alvo verifica os \\(k+1\\) num único forward. Com 1 camada (GLM, Qwen) é sempre a mesma; o DeepSeek atual usa o DSpark.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C(), narrow = w < 620, m = M[S.m], si = st.i;
        const mc = C[m.col] || C.a;
        const A = (k) => (k <= si ? 1 : 0.2);
        const pad = 14;
        // ---- tronco (à esquerda no desktop; em cima no mobile)
        const tw_ = narrow ? 110 : 150, tx = pad;
        const bub = { x: tx + (narrow ? 0 : 18), y: 8, w: 70, h: 24 };
        const mh = { x: tx + (narrow ? 0 : 10), y: 42, w: narrow ? 90 : tw_ - 20, h: 24 };
        const ty = 80, th = narrow ? 150 : h - 120;
        tokBox(g, bub.x, bub.y, bub.w, bub.h, 'x(t+1)', C.d, { eyes: false, size: 11, font: 'mono', fill: V.alpha(C.d, 0.18) });
        V.box(g, mh.x, mh.y, mh.w, mh.h, { r: 7, fill: V.alpha(C.d, 0.18), stroke: C.d, lw: 1.2 });
        V.text(g, 'lm_head', mh.x + mh.w / 2, mh.y + 16, { size: 11, align: 'center', font: 'mono', color: C.ink });
        V.arrow(g, mh.x + mh.w / 2, mh.y - 1, bub.x + bub.w / 2, bub.y + bub.h + 2, { color: V.alpha(C.d, 0.8), lw: 1.3, head: 5 });
        const nL = 6, lh = (th - 10) / nL;
        for (let i = 0; i < nL; i++) {
          const y = ty + i * lh;
          const lab = i === 0 ? 'camada L-1' : i === 1 ? 'camada L-2' : i === 2 ? '…' : 'camada ' + (nL - 1 - i);
          V.box(g, tx, y, tw_, lh - 5, { r: 6, fill: V.alpha(mc, 0.16 - 0.02 * i), stroke: V.alpha(mc, 0.45), lw: 1 });
          V.text(g, lab, tx + tw_ / 2, y + (lh - 5) / 2 + 4, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        if (!narrow) V.text(g, m.trunk, tx + tw_ / 2, ty + th + 12, { size: 11, align: 'center', color: mc, font: 'mono', maxW: tw_ + 24 });
        else m.trunk.split(' · ').forEach((ln, i) => V.text(g, ln, tx + tw_ + 26, ty + th / 2 + i * 16, { size: 11, color: mc, font: 'mono' }));
        V.arrow(g, tx + tw_ / 2, ty - 1, mh.x + mh.w / 2, mh.y + mh.h + 2, { color: V.alpha(C.b, 0.8), lw: 1.4, head: 5 });
        V.text(g, 'h(t)', tx + tw_ / 2 + 8, ty - 4, { size: 11, font: 'mono', color: C.b });
        // ---- módulo MTP
        const mx0 = narrow ? pad : tx + tw_ + 130, mw = w - mx0 - pad;
        const my0 = narrow ? ty + th + 30 : 56;
        const mhh = h - my0 - 8;
        V.box(g, mx0, my0 - 4, mw, mhh, { r: 14, fill: V.alpha(mc, 0.04), stroke: V.alpha(mc, 0.35), lw: 1 });
        const ttl = 'módulo MTP · ' + m.where;
        if (!narrow) V.text(g, ttl, mx0 + mw - 12, my0 - 4 + mhh - 12, { size: fitSize(g, ttl, mw * 0.45, 12, 'sans', 600), weight: 600, color: mc, align: 'right' });
        if (narrow) V.text(g, ttl, mx0 + 12, my0 + 14, { size: 12, weight: 600, color: mc, maxW: mw - 24 });
        const inY = my0 + (narrow ? 32 : 16), bw2 = (mw - 44) / 2;
        const inE = { x: narrow ? mx0 + 28 + bw2 : mx0 + 16, y: inY, w: bw2, h: 26 }, inH = { x: narrow ? mx0 + 16 : mx0 + 28 + bw2, y: inY, w: bw2, h: 26 };
        g.globalAlpha = A(1);
        V.box(g, inE.x, inE.y, inE.w, inE.h, { r: 7, fill: V.alpha(C.a, 0.16), stroke: C.a, lw: 1.2 });
        V.text(g, m.emb, inE.x + inE.w / 2, inE.y + 17, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: inE.w - 8 });
        V.box(g, inH.x, inH.y, inH.w, inH.h, { r: 7, fill: V.alpha(C.b, 0.16), stroke: C.b, lw: 1.2 });
        V.text(g, m.hidden, inH.x + inH.w / 2, inH.y + 17, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: inH.w - 8 });
        const nY = inY + 46;
        V.text(g, 'RMSNorm', inE.x + inE.w / 2, nY, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, 'RMSNorm', inH.x + inH.w / 2, nY, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, m.norms, mx0 + mw / 2, nY + 16, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: mw - 20 });
        g.globalAlpha = 1;
        if (si >= 1) {
          const ha = V.alpha(C.b, 0.85), ea = V.alpha(C.a, 0.85);
          g.save(); g.strokeStyle = ha; g.lineWidth = 1.6; g.beginPath();
          const hx0 = tx + tw_ + 2, hy0 = ty + 10, hxe = inH.x + inH.w / 2;
          if (!narrow) { g.moveTo(hx0, hy0); g.bezierCurveTo(hx0 + 60, 22, hxe - 40, 22, hxe, inH.y - 3); }
          else { const vx = tx + tw_ + 14; g.moveTo(hx0, hy0); g.lineTo(vx, hy0); g.lineTo(vx, inY - 22); g.quadraticCurveTo(vx, inY - 12, hxe, inY - 12); g.lineTo(hxe, inH.y - 3); }
          g.stroke(); g.restore();
          V.arrow(g, hxe, inH.y - 12, hxe, inH.y - 1, { color: ha, lw: 1.6, head: 6 });
          const ex0 = bub.x + bub.w + 2, ey0 = bub.y + bub.h / 2, exe = inE.x + inE.w / 2;
          g.save(); g.strokeStyle = ea; g.lineWidth = 1.4; g.beginPath();
          if (!narrow) { g.moveTo(ex0, ey0); g.bezierCurveTo(ex0 + 120, ey0, exe, inE.y - 40, exe, inE.y - 3); }
          else { const rx = w - 18; g.moveTo(ex0, ey0); g.lineTo(rx, ey0); g.lineTo(rx, inY - 22); g.quadraticCurveTo(rx, inY - 12, exe, inY - 12); g.lineTo(exe, inE.y - 3); }
          g.stroke(); g.restore();
        }
        // passo 3: concat + proj (matriz)
        g.globalAlpha = A(2);
        const cY = nY + 50;
        const cells = narrow ? 12 : 16, ccw = Math.min(14, (mw - 40) / cells - 1.5);
        const cx0 = mx0 + mw / 2 - (cells * (ccw + 1.5)) / 2;
        const leftCol = narrow ? C.b : C.a, rightCol = narrow ? C.a : C.b;
        // GLM: concatenação contínua [e;h] (2d). Qwen/DS: combinação não inventariada → metades separadas por "?"
        const half = cells / 2, gapMid = m.comb === 'concat' ? 0 : 16, stepC = ccw + 1.5;
        const colX = (j) => cx0 + j * stepC + (j >= half ? gapMid : 0) - gapMid / 2;
        for (let j = 0; j < cells; j++) { g.fillStyle = V.alpha(j < half ? leftCol : rightCol, 0.7); V.rr(g, colX(j), cY, ccw, 10, Math.min(3, ccw / 4)); g.fill(); }
        if (gapMid) V.text(g, '?', mx0 + mw / 2, cY + 10, { size: 13, align: 'center', font: 'mono', weight: 700, color: C.d });
        V.text(g, narrow && m.comb === 'concat' ? 'concatena [e ; h] · 2d' : m.cat, mx0 + mw / 2, cY - 6, { size: 11, align: 'center', font: 'mono', color: m.comb === 'concat' ? C.muted : C.d, maxW: mw - 16 });
        const pmData = Array.from({ length: 4 }, (_, i) => Array.from({ length: half }, (_, j) => Math.sin(i * 1.9 + j * 0.7)));
        let pm;
        if (m.comb === 'dois') { // duas projeções d×d separadas
          pm = V.matrix(g, { x: colX(0), y: cY + 18, data: pmData, cell: ccw, gap: 1.5, cmap: 'div' });
          V.matrix(g, { x: colX(half), y: cY + 18, data: pmData.map((r) => r.map((v) => -v)), cell: ccw, gap: 1.5, cmap: 'div' });
        } else {
          pm = V.matrix(g, { x: cx0 - gapMid / 2, y: cY + 18, data: Array.from({ length: 4 }, (_, i) => Array.from({ length: cells }, (_, j) => Math.sin(i * 1.9 + j * 0.7))), cell: ccw, gap: gapMid ? 1.5 + gapMid / (cells - 1) : 1.5, cmap: 'div' });
        }
        V.text(g, m.proj, mx0 + mw / 2, cY + 18 + pm.h + 14, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: mw - 16 });
        g.globalAlpha = 1;
        if (si >= 2) { const L2 = cx0 + 24, R2 = cx0 + cells * (ccw + 1.5) - 24; V.arrow(g, inE.x + inE.w / 2, nY + 21, narrow ? R2 : L2, cY - 19, { color: V.alpha(C.a, 0.6), lw: 1.2, head: 5 }); V.arrow(g, inH.x + inH.w / 2, nY + 21, narrow ? L2 : R2, cY - 19, { color: V.alpha(C.b, 0.6), lw: 1.2, head: 5 }); }
        // passo 4: bloco
        g.globalAlpha = A(3);
        const bY = cY + 18 + pm.h + 34, bH = 62, bW = Math.min(mw - 40, 300), bX = mx0 + (mw - bW) / 2;
        V.box(g, bX, bY, bW, bH, { r: 10, fill: V.alpha(mc, 0.14), stroke: mc, lw: 1.4 });
        m.block.forEach((ln, i) => V.text(g, ln, bX + bW / 2, bY + 18 + i * 16, { size: fitSize(g, ln, bW - 16, 11.5), align: 'center', color: i === 0 ? C.ink : C.muted, weight: i === 0 ? 600 : 400 }));
        g.globalAlpha = 1;
        if (si >= 3) V.arrow(g, mx0 + mw / 2, cY + 18 + pm.h + 20, mx0 + mw / 2, bY - 2, { color: V.alpha(mc, 0.8), lw: 1.6, head: 6 });
        // passo 5: cabeça compartilhada + x(t+2)
        g.globalAlpha = A(4);
        const hY = bY + bH + 30, hW = Math.min(210, mw - 120), hX = mx0 + 16;
        V.box(g, hX, hY, hW, 26, { r: 7, fill: V.alpha(C.d, 0.18), stroke: C.d, lw: 1.2 });
        V.text(g, m.head, hX + hW / 2, hY + 17, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: hW - 8 });
        const x2 = { x: hX + hW + 16, y: hY };
        tokBox(g, x2.x, x2.y, 70, 26, 'x(t+2)', C.c, { eyes: false, size: 11, font: 'mono', fill: V.alpha(C.c, 0.2) });
        g.globalAlpha = 1;
        if (si >= 4) {
          V.arrow(g, mx0 + mw / 2, bY + bH + 2, hX + hW / 2, hY - 2, { color: V.alpha(C.d, 0.8), lw: 1.4, head: 6 });
          if (!narrow) {
            g.save(); g.setLineDash([4, 4]); g.strokeStyle = V.alpha(C.d, 0.7); g.lineWidth = 1.2; g.beginPath();
            const gx = tx + tw_ + 60;
            g.moveTo(mh.x + mh.w, mh.y + mh.h / 2); g.bezierCurveTo(gx, mh.y + 20, gx, hY + 13, hX - 2, hY + 13); g.stroke(); g.restore();
            V.text(g, 'mesma cabeça', gx - 4, (mh.y + hY) / 2 + 30, { size: 11, color: C.d, font: 'mono', align: 'center' });
          } else V.text(g, '(a mesma lm_head do tronco)', hX, hY + 42, { size: 11, color: C.d, font: 'mono' });
        }
        // passo 6: recursão
        if (si >= 5) {
          const ph = (t % 3) / 3;
          const rX = mx0 + mw - 10, ly = hY + 13;
          const pts = [[x2.x + 70, ly], [rX, ly], [rX, inE.y + inE.h + 8], [inE.x + inE.w / 2 + 6, inE.y + inE.h + 8]];
          g.strokeStyle = V.alpha(C.c, 0.8); g.lineWidth = 1.6; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke();
          const seg = Math.min(2, Math.floor(ph * 3)), tt = ph * 3 - seg;
          const p = V.along(pts[seg][0], pts[seg][1], pts[seg + 1][0], pts[seg + 1][1], tt); V.glow(g, p[0], p[1], 10, C.c); V.dot(g, p[0], p[1], 3.5, C.c);
          if (narrow) V.text(g, m.k, hX, hY + 76, { size: 11.5, color: C.c, weight: 600 });
          else V.text(g, m.k, x2.x + 78, ly - 7, { size: fitSize(g, m.k, rX - x2.x - 86, 11.5, 'sans', 600), color: C.c, weight: 600 });
          V.text(g, 'rascunhos: x̃(t+2), x̃(t+3), …', hX, hY + (narrow ? 58 : 46), { size: 11, color: C.c, font: 'mono' });
        }
      });
    })();

    /* ================================================== Fig 10: reuso do MTP */
    (function fig10() {
      const fig = root.querySelector('#fig-reuso'); if (!fig) return;
      adaptH(fig, 440, 610);
      const R = {
        qwen: { label: 'Qwen · MTP', n: 1, k: 3, acc: [0.89, 0.745, 0.6], note: 'medido: TP2+EP, MTP=3, greedy (README)' },
        glm: { label: 'GLM · MTP de reserva', n: 1, k: 2, acc: null, note: 'sem aceitação por posição publicada; k = 2 ≈ 24,6 tok/s (baseline citado na receita)' },
        dsv: { label: 'DeepSeek · Vision-Exp', n: 3, k: 6, ds: true, launchMin: true,
          acc: [0.88, 0.74, 0.53, 0.36, 0.24], note: 'medido em 2026-09-03 com k = 5 e k = 6: a mesma curva nas 5 posições do bloco (PATCHES.md)',
          acc6: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15], note6: 'bench_quick com k = 6 (PATCHES.md): a 6ª posição, fora do bloco treinado, aceita só 15%' },
        ds0731: { label: 'DeepSeek · 0731', n: 1, k: 5, ds: true, acc: [0.93, 0.75, 0.66, 0.58, 0.47], note: 'bench_quick do 0731 com k = 5 (PATCHES.md)' },
      };
      const BLOCK = 5; // bloco treinado do DSpark (dspark_block_size)
      const S = { r: 'qwen', k: 3, blockK: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: Object.keys(R).map((k) => ({ v: k, label: R[k].label })), value: 'qwen', onChange: (v) => { S.r = v; sK.set(R[v].k); } });
      const sK = UI.slider(ctl, { label: 'k', min: 1, max: 9, step: 1, value: 3, onInput: (v) => { S.k = v; } });
      UI.toggle(ctl, { label: 'block-k (DSpark isento)', value: false, onChange: (v) => { S.blockK = v; } });
      const rV = UI.readout(ctl, 'linhas verificadas/seq', '4');
      const rT = UI.readout(ctl, 'tokens/passo (estim.)', '—');
      let lastV = '', lastT = '';
      /** estado derivado da receita + k + block-k */
      function derive() {
        const rc = R[S.r], k = S.k, n = rc.n;
        const exempt = !!rc.ds && S.blockK;
        const ruleOK = exempt || k <= n || k % n === 0; // validação da Anemll 0.1.1 (regra do MTP em ciclo)
        const minOK = !rc.launchMin || exempt || k >= BLOCK; // launcher do Vision-Exp: k ≥ 5 (block-k relaxa para ≥ 1)
        const ok = ruleOK && minOK;
        const useK6 = rc.acc6 && k > BLOCK;
        const acc = useK6 ? rc.acc6 : rc.acc, note = useK6 ? rc.note6 : rc.note;
        let est = '—';
        if (acc && k <= acc.length) est = fmt(1 + acc.slice(0, k).reduce((a, b) => a + b, 0), 2);
        else if (acc) est = '≥ ' + fmt(1 + acc.reduce((a, b) => a + b, 0), 2);
        let msg;
        if (!rc.ds) msg = k === 1 ? 'k = 1: a camada MTP roda uma vez e lê o estado do tronco, como no treino.'
          : `k = ${k}: a mesma camada MTP roda ${k} vezes; da 2ª em diante ela lê a própria saída, e a aceitação cai.`;
        else if (!ruleOK) msg = `k = ${k}: a validação do vLLM (pensada para MTP em ciclo) recusa, porque ${k} % ${n} ≠ 0${k === BLOCK ? ', embora 5 seja o bloco treinado do DSpark' : ''}.`;
        else if (!minOK) msg = `k = ${k} < 5: o launcher exige k ≥ 5 para não truncar o bloco do DSpark.`;
        else {
          const extra = k - BLOCK;
          const val = exempt ? 'block-k isenta method=dspark da validação' : n === 1 ? `n_predict = 1 no config (apesar dos 3 estágios): a validação aceita k = ${k}` : `passa na validação do vLLM (${k} % ${n} = 0, k ≥ 5)`;
          const blk = extra > 0 ? `, ${extra} posiç${extra > 1 ? 'ões' : 'ão'} de ruído além do bloco treinado (5)` : extra === 0 ? ', exatamente o bloco treinado' : ', menor que o bloco treinado (5)';
          msg = `${val}; o DSpark roda um bloco de ${k} num único forward${blk}.`;
        }
        return { rc, k, n, exempt, ruleOK, minOK, ok, acc, note, est, msg };
      }
      /** pílulas em fluxo (quebram linha no celular); devolve o y de baixo */
      function pills(g, items, x0, y0, maxX, C) {
        let x = x0, y = y0;
        items.forEach(([txt, col]) => {
          const pw = tw(g, txt, 11) + 18;
          if (x + pw > maxX && x > x0) { x = x0; y += 28; }
          V.box(g, x, y, pw, 22, { r: 11, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.8), lw: 1.2 });
          V.text(g, txt, x + 9, y + 15, { size: 11, color: C.ink });
          x += pw + 8;
        });
        return y + 22;
      }
      V.fig(fig, (g, w, h, t) => {
        if (w < 80 || h < 80) return; // quadro transitório durante relayout
        const C = V.C(), narrow = w < 620, pad = 14, D = derive(), { rc, k, n, exempt, ok } = D;
        // readouts: só quando mudam
        const vTxt = String(k + 1), tTxt = ok ? D.est : 'inválido';
        if (vTxt !== lastV) { rV.set(vTxt); lastV = vTxt; }
        if (tTxt !== lastT) { rT.set(tTxt); lastT = tTxt; }
        // status
        const msgLines = wrapLines(g, (ok ? '✓ ' : '✕ ') + D.msg, w - 2 * pad - 20, narrow ? 11 : 12);
        const stH = 12 + msgLines.length * 15;
        V.box(g, pad, 10, w - 2 * pad, stH, { r: 8, fill: V.alpha(ok ? C.e : RED(), 0.1), stroke: V.alpha(ok ? C.e : RED(), 0.6), lw: 1 });
        msgLines.forEach((ln, i) => V.text(g, ln, pad + 10, 27 + i * 15, { size: narrow ? 11 : 12, color: C.ink }));
        // slots
        const top = 10 + stH + 16;
        const slotW = Math.min(76, (w - 2 * pad - (k - 1) * 8) / k), sx0 = pad + (w - 2 * pad - (k * slotW + (k - 1) * 8)) / 2;
        const mcols = [C.a, C.b, C.d];
        for (let i = 0; i < k; i++) {
          const x = sx0 + i * (slotW + 8);
          const beyond = rc.ds && i >= BLOCK;
          const pulse = rc.ds ? (Math.floor(t * 1.5) % 2) === 0 : (Math.floor(t * 1.5) % k) === i;
          const col = !ok ? RED() : beyond ? C.d : C.c;
          tokBox(g, x, top, slotW, 30, slotW >= 46 ? 'pos ' + (i + 1) : String(i + 1), col, { eyes: false, size: 11, font: 'mono', fill: V.alpha(col, pulse ? 0.3 : 0.12) });
          if (!rc.ds) {
            // GLM/Qwen: um robô (módulo MTP) por posição, reaplicado em ciclo
            const mi = i % n, mcol = mcols[mi], ry = top + 44;
            V.box(g, x + slotW / 2 - 16, ry, 32, 26, { r: 7, fill: V.alpha(mcol, pulse ? 0.45 : 0.2), stroke: mcol, lw: 1.3 });
            V.dot(g, x + slotW / 2 - 6, ry + 10, 2, mcol); V.dot(g, x + slotW / 2 + 6, ry + 10, 2, mcol);
            V.line(g, x + slotW / 2, ry, x + slotW / 2, ry - 5, { color: mcol, lw: 1.2 }); V.dot(g, x + slotW / 2, ry - 6, 2, mcol);
            V.text(g, n === 1 ? 'MTP' : 'M' + mi, x + slotW / 2, ry + 40, { size: 11, align: 'center', font: 'mono', color: mcol });
            V.arrow(g, x + slotW / 2, ry - 8, x + slotW / 2, top + 32, { color: V.alpha(mcol, 0.6), lw: 1.1, head: 5 });
          }
        }
        let below;
        if (!rc.ds) {
          const loopTxt = ['↻ a mesma camada, reaplicada ' + k + '× (lendo a própria saída a partir da 2ª)', '↻ a mesma camada, reaplicada ' + k + '×'].find((s2) => tw(g, s2, 11, 'mono') <= w - 2 * pad) || '';
          if (k > 1) V.text(g, loopTxt, w / 2, top + 104, { size: 11, align: 'center', color: C.muted, font: 'mono' });
          below = top + 110;
        } else {
          // DeepSeek: o DSpark é sempre um bloco empilhado num forward; a regra é só um selo de validação
          const x = sx0, ww = k * slotW + (k - 1) * 8, ry = top + 42;
          V.box(g, x, ry, ww, 36, { r: 9, fill: V.alpha(C.ds, 0.16), stroke: C.ds, lw: 1.4 });
          const band = [`DSpark: 3 estágios empilhados, bloco de ${k} num único forward (âncora + ${k - 1} ruídos)`, `DSpark: 3 estágios empilhados, bloco de ${k} num forward`, `DSpark: bloco de ${k} num forward`, `bloco de ${k}`].find((s2) => tw(g, s2, 11) <= ww - 14) || '';
          V.text(g, band, x + ww / 2, ry + 22, { size: 11, align: 'center', color: C.ink });
          // posições do bloco lidas em paralelo (setas que chegam juntas)
          const u = (t * 0.9) % 1;
          for (let i = 0; i < k; i++) { const cx = sx0 + i * (slotW + 8) + slotW / 2, yy = V.lerp(ry - 2, top + 32, V.ease.inOut(u)); V.dot(g, cx, yy, 2.4, i >= BLOCK ? C.d : C.ds); }
          const vCol = exempt ? C.muted : D.ruleOK ? C.e : RED();
          const vTxt2 = exempt ? 'validação k % n_predict: isento (block-k)' : n === 1 ? 'validação: ✓ n_predict = 1' : `validação: ${D.ruleOK ? '✓' : '✕'} ${k} % ${n} ${k % n ? '≠' : '='} 0`;
          const lTxt = exempt ? 'launcher: ✓ k ≥ 1 (block-k)' : `launcher: ${D.minOK ? '✓ k ≥ 5' : '✕ k < 5'}`;
          const items = [[vTxt2, vCol]];
          if (rc.launchMin) items.push([lTxt, exempt || D.minOK ? C.e : RED()]);
          if (k > BLOCK) items.push([`+${k - BLOCK} fora do bloco treinado (5)`, C.d]);
          below = pills(g, items, pad, ry + 50, w - pad, C) + 6;
        }
        // aceitação
        const ay = Math.max(top + 150, below + 34), ah = h - ay - (narrow ? 118 : 48);
        const chartW = narrow ? w - 2 * pad : Math.min(w * 0.58, 520);
        V.text(g, 'aceitação por posição (medida)', pad, ay - 12, { size: 11.5, color: C.muted });
        const acc = D.acc;
        const nb = Math.max(k, acc ? acc.length : 0, 3), bwid = Math.min(70, (chartW - (nb - 1) * 10) / nb);
        const bx0 = pad + (chartW - (nb * bwid + (nb - 1) * 10)) / 2;
        const bcol = rc.ds ? C.ds : S.r === 'qwen' ? C.qwen : C.glm;
        V.line(g, pad, ay + ah + 0.5, pad + chartW, ay + ah + 0.5, { color: V.alpha(C.muted, 0.4), lw: 1 });
        for (let i = 0; i < nb; i++) {
          const x = bx0 + i * (bwid + 10), a = acc && acc[i];
          const inK = i < k;
          if (a != null) {
            const bh = a * ah;
            g.fillStyle = V.alpha(rc.ds && i >= BLOCK ? C.d : bcol, inK ? 0.8 : 0.22); V.rr(g, x, ay + ah - bh, bwid, bh, 4); g.fill();
            V.text(g, fmt(a * 100, Math.round(a * 1000) % 10 ? 1 : 0) + '%', x + bwid / 2, ay + ah - bh - 5, { size: 11, font: 'mono', align: 'center', color: inK ? C.ink : C.muted });
          } else if (inK) {
            g.save(); g.setLineDash([4, 3]); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; V.rr(g, x, ay + ah * 0.55, bwid, ah * 0.45, 4); g.stroke(); g.restore();
            if (tw(g, 'sem dado', 11) <= bwid + 8) V.text(g, 'sem dado', x + bwid / 2, ay + ah * 0.55 - 5, { size: 11, align: 'center', color: C.muted });
            else { V.text(g, 'sem', x + bwid / 2, ay + ah * 0.55 - 18, { size: 11, align: 'center', color: C.muted }); V.text(g, 'dado', x + bwid / 2, ay + ah * 0.55 - 5, { size: 11, align: 'center', color: C.muted }); }
          }
          V.text(g, String(i + 1), x + bwid / 2, ay + ah + 14, { size: 11, font: 'mono', align: 'center', color: inK ? C.ink : C.muted });
        }
        // painel de tokens por passo
        const px = narrow ? pad : pad + chartW + 24, py = narrow ? ay + ah + 26 : ay, pw = narrow ? w - 2 * pad : w - px - pad;
        const tokTxt = !ok ? 'inválido' : D.est === '—' ? '—' : D.est.startsWith('≥') ? D.est : '≈ ' + D.est;
        V.text(g, 'tokens por passo', px, py + 8, { size: 11.5, color: C.muted });
        V.text(g, tokTxt, px, py + (narrow ? 34 : 44), { size: narrow ? 22 : 30, font: 'display', weight: 700, color: ok ? C.c : RED() });
        if (!narrow) {
          const n1 = wrap(g, `1 + Σ aᵢ (i ≤ k), com as taxas medidas. O alvo verifica k + 1 = ${k + 1} linhas por sequência a cada passo.`, px, py + 72, pw, 15, { size: 11.5, color: C.muted });
          const extra = S.r === 'qwen' ? 'Medido: 24,5 → 52,1 tok/s (2,13×). O ganho em tok/s é menor que em tokens/passo porque o passo fica mais caro.'
            : S.r === 'dsv' ? 'A 6ª posição acrescenta só a₆ ≈ 0,15 token/passo e deixa o passo mais caro: com block-k e k = 5, +2–10% em single-stream.' : '';
          if (extra) wrap(g, extra, px, py + 72 + n1 * 15 + 12, pw, 15, { size: 11.5, color: C.ink });
        }
        wrap(g, D.note, pad, h - (narrow ? 26 : 16), w - 2 * pad, 14, { size: 11, color: C.muted });
      });
    })();
  },
});

LA.chapter({
  id: 'mixture-of-experts',
  part: 3,
  order: 11,
  title: 'Mixture-of-Experts: especialistas e roteadores',
  navTitle: 'Mixture-of-Experts',
  subtitle: 'Centenas de especialistas por camada, poucos por token: roteadores, funções de score, top-k, pesos de combinação, expert compartilhado, balanceamento, hash MoE e o que tudo isso custa na inferência.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['MoE', 'roteador', 'top-k', 'sigmoid', 'sqrtsoftplus', 'shared expert', 'aux-loss-free', 'hash routing', 'experts gordos', 'parâmetros ativos'],
  goals: [
    'Explicar por que um MoE tem <b>capacidade sem custo proporcional</b> e calcular parâmetros totais, ativos e bytes por passo a partir do config',
    'Descrever o roteador completo: logits, funções de score (softmax, sigmoid, sqrtsoftplus), top-k, <code>norm_topk_prob</code> e <code>routed_scaling_factor</code>',
    'Comparar as estratégias de balanceamento (aux loss, viés <code>noaux_tc</code>, grupos) e os casos especiais: expert compartilhado, camadas densas iniciais e hash MoE',
    'Relacionar o desbalanceamento de carga com as escolhas das receitas: experts magros e gordos, cap de linhas, EP e TP',
  ],
  prereqs: ['bloco-transformer', 'prefill-decode-kv'],
  summary: [
    'MoE troca a FFN de cada camada por <b>E</b> experts pequenos e um roteador que escolhe <b>k</b> por token: os parâmetros totais crescem com E, o custo por token com k (GLM 288/8, Qwen 512/10, DeepSeek 256/6, cada um com 1 expert compartilhado).',
    'A memória segue o <b>total</b> (experts são 96–97% dos parâmetros, por isso recebem ~4 bits); os FLOPs seguem os <b>ativos</b>; o decode segue os <b>experts distintos</b> lidos por passo, até min(E, kT).',
    'O roteador é um linear \\(W_r\\) + função de score (softmax no Qwen, sigmoid no GLM, sqrtsoftplus no DeepSeek) + top-k; os pesos são renormalizados (<code>norm_topk_prob</code>) e escalados (<code>routed_scaling_factor</code> 2,5 no GLM, 1,5 no DeepSeek); os roteadores nunca são quantizados.',
    'Balanceamento: a aux loss (coeficiente 0,001) mexe nos pesos; o viés <code>e_score_correction_bias</code> (<code>noaux_tc</code>) muda só a escolha. O roteamento por grupos limitava os nós tocados em EP e está desligado no GLM e ausente no V4.',
    'Casos especiais: expert compartilhado (com gate sigmoide no Qwen), camadas densas iniciais (GLM 0–2, largura 12288) e hash MoE (DeepSeek 0–2, <code>tid2eid</code>), que exige ids crus e rota própria para imagens (<code>bias_vl</code>, #175).',
    'Na inferência o desbalanceamento vira tempo: o decode tem experts magros (ler rápido), o prefill tem gordos (o mais quente aparece em quase todo token). O GLM ordena por expert, conta na GPU e usa cap 32 + tiers E1–E3 para os gordos.',
    'Na lane TensorFold do GLM o mesmo MoE roda sob <b>invariância de lote</b>: logits com redução fixa pela forma, empate pelo menor id, shared como 9º slot somado em ordem; os experts viram <b>itens</b> de 16/64 pares, e leves (DRAM) e pesados (tensor cores) são intercalados no prompt. Cronometrar uma janela de 16 linhas com um token repetido dava cerca de metade do custo real.',
  ],
  quiz: [
    { q: 'O GLM-5.3-Flash tem ~321 B de parâmetros e ~18 B ativos por token. Qual afirmação está correta?', o: ['A memória necessária segue os ativos; por isso ele caberia num único nó', 'A memória segue o total, e o decode é limitado pelos bytes dos experts efetivamente lidos em cada passo', 'Como só 8 experts rodam, os outros 280 podem ficar no disco sem custo', 'Os FLOPs por token seguem o total de parâmetros'], a: 1, e: 'Todos os 288 experts de cada camada precisam estar na memória, porque qualquer token pode escolher qualquer um. FLOPs e bytes lidos por token seguem os ativos; num passo com vários tokens, seguem os experts distintos tocados.' },
    { q: 'No roteamento do GLM (sigmoid, <code>norm_topk_prob: true</code>, <code>routed_scaling_factor: 2.5</code>), o que acontece com os pesos dos 8 experts escolhidos?', o: ['Ficam iguais aos sigmoids, que podem somar até 8', 'São renormalizados para somar 1 e depois multiplicados por 2,5', 'São multiplicados por 2,5 e depois passam por um softmax', 'Recebem o viés de correção antes de serem normalizados'], a: 1, e: 'Os scores sigmoid dos escolhidos (sem o viés) são divididos pela soma, que passa a valer 1, e multiplicados por ρ = 2,5. O viés só entra na escolha de quem participa.' },
    { q: 'No balanceamento aux-loss-free (<code>noaux_tc</code>), o <code>e_score_correction_bias</code>…', o: ['…é somado aos scores só para escolher o top-k; os pesos de combinação vêm dos scores sem o viés', '…é uma loss somada no treino com coeficiente 0,001', '…multiplica os pesos dos experts sobrecarregados por um fator menor que 1', '…é recalculado pelo vLLM a cada requisição'], a: 0, e: 'O viés muda quem é escolhido, não quanto cada um pesa. No treino ele é ajustado fora do gradiente; na inferência fica congelado no checkpoint.' },
    { q: 'Por que o hotfix de visão do DeepSeek-V4 precisou mexer no roteamento das camadas 0–2?', o: ['Porque elas são densas e não aceitam tokens de imagem', 'Porque o roteamento por hash usa o id do token, e todos os tokens de imagem têm o mesmo id (129264): sem correção iriam todos para os mesmos 6 experts', 'Porque a tabela tid2eid não cabe na GPU quando há imagens', 'Porque o bias_vl só existe nessas camadas'], a: 1, e: 'tid2eid[129264] é uma linha só. A correção (#175) roteia as linhas de imagem com o top-6 de s + bias_vl, sem a tabela; e o modelo passa a exigir os input_ids crus (requires_raw_input_tokens).' },
    { q: 'No DeepSeek com uma sequência e 6 tokens de rascunho, quantos experts distintos, no máximo, uma camada MoE toca num passo de verificação?', o: ['6', '7', '42', '256'], a: 2, e: 'São 7 tokens (1 + 6 rascunhos) × top-6 = 42 escolhas, logo no máximo 42 experts distintos (≈ 39 esperados com roteamento uniforme). É o que leva a ≈ 12 GB de experts lidos por rank e por passo.' },
    { q: 'Por que o GLM usa <code>EXL3_TEMP_ROWS_FUSED=32</code> com o tier E3?', o: ['Porque 32 é o top-k do modelo', 'Para caber um chunk inteiro de prefill num lançamento', 'Porque um passo de decode tem no máximo 4 sequências × 8 tokens = 32 linhas e nenhum expert recebe mais linhas do que há tokens: o decode fica sempre num único lançamento capturável', 'Para forçar todos os experts a serem gordos'], a: 2, e: 'Com tokens ≤ cap não existe expert gordo, e o exl3_moe atende a camada num lançamento compatível com CUDA graph. Subir o cap para 1024 foi medido e ficou 13–24% mais lento no prefill.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    // texto sobre preenchimento colorido: a cor de fundo do tema dá o contraste nos dois temas
    const onTxt = (C) => C.bg;
    const nf = (x, d = 1) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const ni = (x) => Math.round(x).toLocaleString('pt-BR');
    function splitLines(g, s, maxW, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(' ');
      const lines = [];
      let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (g.measureText(tt).width > maxW && cur) { lines.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) lines.push(cur);
      return lines;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const ls = splitLines(g, s, maxW, o), mx = o.maxLines || 99;
      ls.slice(0, mx).forEach((l, i) => V.text(g, i === mx - 1 && ls.length > mx ? l + '…' : l, x, y + i * lh, o));
      return Math.min(ls.length, mx);
    }
    function textW(g, s, size, font, weight) { g.font = V.font(size || 12, font || 'sans', weight || 400); return g.measureText(String(s)).width; }
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const sqsp = (x) => Math.sqrt(x > 30 ? x : Math.log1p(Math.exp(x)));
    const topk = (arr, k) => arr.map((v, i) => i).sort((a, b) => arr[b] - arr[a]).slice(0, k);
    function pathPos(pts, u) {
      let L = 0; const seg = [];
      for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
      let r = V.clamp(u, 0, 1) * L;
      for (let i = 0; i < seg.length; i++) { if (r <= seg[i] || i === seg.length - 1) { const f = seg[i] ? r / seg[i] : 0; return V.along(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], Math.min(1, f)); } r -= seg[i]; }
      return pts[pts.length - 1];
    }
    const safe = (name, fn) => { try { fn(); } catch (e) { console.error('[moe] ' + name, e); } };
    // O MathML oculto do KaTeX é position:absolute e escapa do overflow-x das caixas .la-eq e
    // .la-table-wrap (que não são posicionadas), alargando a página em telas estreitas.
    // Contenção local: tornar essas caixas o bloco de contenção.
    root.querySelectorAll('.la-eq, .la-table-wrap').forEach((el) => { el.style.position = 'relative'; });

    /* ====================================================================
       Figura 1 — Da FFN densa ao MoE (stepper)
       ==================================================================== */
    safe('fig1', () => {
      const fig = $('#fig-ffn-moe');
      fitH(fig, (w) => (w < 600 ? 520 : 420));
      const NE = 8;
      const TOK = [
        { w: 'gato', s: [0.10, 0.72, 0.20, 0.15, 0.64, 0.30, 0.05, 0.22] },
        { w: 'def', s: [0.20, 0.10, 0.81, 0.30, 0.12, 0.25, 0.69, 0.10] },
        { w: '∫', s: [0.33, 0.12, 0.28, 0.77, 0.21, 0.71, 0.14, 0.09] },
        { w: 'Paris', s: [0.66, 0.58, 0.10, 0.20, 0.31, 0.12, 0.25, 0.70] },
      ];
      TOK.forEach((tk) => { tk.top = topk(tk.s, 2); const sm = tk.s[tk.top[0]] + tk.s[tk.top[1]]; tk.g = tk.top.map((i) => tk.s[i] / sm); });
      const SEQ = [0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 3, 2, 0, 2, 3, 1];
      const steps = [
        { t: 'A FFN densa', d: 'Todo token atravessa todos os neurônios intermediários da FFN. Nas camadas 0–2 do GLM são 12.288 neurônios e 151 M de parâmetros por camada, todos calculados para cada token.' },
        { t: 'Fatiar… e multiplicar', d: 'Corte a largura intermediária em blocos independentes, cada um uma SwiGLU pequena com gate, up e down próprios. Fatiar sozinho não muda nada: sem roteador, o token passaria por todos, com os mesmos parâmetros de antes. O ganho vem de <b>multiplicar</b> os blocos (as pilhas atrás de cada um): o GLM tem 288 blocos de largura 2048, ou 589.824 neurônios, 48× os 12.288 da FFN densa, e mesmo assim cada token acende só 8.' },
        { t: 'O roteador dá notas', d: 'Um linear \\(W_r\\) calcula um logit por expert, \\(\\boldsymbol\\ell = W_r\\mathbf x\\), e a função de score o transforma em nota. O nível de preenchimento de cada bloco é a nota do expert para este token.' },
        { t: 'Top-k acende poucos', d: 'Só os \\(k\\) experts de maior nota rodam (aqui \\(k=2\\); no GLM, 8 de 288). Os outros ficam na memória, apagados. O custo da camada cai para \\(k/E\\) do que seria com todos.' },
        { t: 'Soma ponderada + compartilhado', d: 'As saídas dos escolhidos são somadas com pesos \\(g_e\\) (as notas renormalizadas), e o expert compartilhado, que roda para todo token, soma a sua parte. O resultado volta ao fluxo residual.' },
        { t: 'Cada token, sua escolha', d: 'Tokens diferentes acendem experts diferentes. Os contadores embaixo de cada bloco acumulam a carga. Num passo com muitos tokens, a união dessas escolhas decide quantos experts precisam ser lidos da memória (seção 7).' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const heads = ['FFN densa: 1 bloco, todos os neurônios ativos', 'fatiar em 8 blocos… e multiplicar as pilhas', 'roteador: logits = W_r · x → uma nota por expert', 'top-2 de 8: só 1/4 da FFN roda', 'y = Σ gₑ·Eₑ(x) + Shared(x)', 'tokens diferentes, experts diferentes'];
      const foot = ['GLM, camadas 0–2: 3 × 4096 × 12.288 ≈ 151 M parâmetros por token', 'GLM: 288 blocos × 2048 = 589.824 neurônios (48× a densa), só 8 acesos por token', 'W_r no GLM: [288 × 4096], ≈ 1,2 M parâmetros por camada', 'ativos: 2 de 8 experts (no GLM: 8 de 288)', '', 'a carga de cada expert depende do que os tokens pedem'];
      const parts = Array.from({ length: 22 }, (_, i) => ({ o: i / 22, k: i }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const horiz = w >= 600;
        const top = 40, bot = horiz ? 28 : 36;
        const X0 = 16, Wd = w - 32, Hd = h - top - bot;
        const P = (u, v) => (horiz ? [X0 + u * Wd, top + v * Hd] : [X0 + v * Wd, top + u * Hd]);
        const R = (u, v, du, dv) => { const c = P(u, v); const rw = horiz ? du * Wd : dv * Wd, rh = horiz ? dv * Hd : du * Hd; return [c[0] - rw / 2, c[1] - rh / 2, rw, rh]; };
        const s = st.i, el = st.elapsed(), e = V.ease.inOut(V.clamp(el / 1.1, 0, 1));
        let tk = TOK[0], seqIdx = 0;
        if (s === 5) { seqIdx = Math.floor(el / 1.5); tk = TOK[SEQ[seqIdx % SEQ.length]]; }
        const uE = 0.56, duE = horiz ? 0.15 : 0.14, uTok = horiz ? 0.05 : 0.04, uR = horiz ? 0.28 : 0.25, uS = horiz ? 0.83 : 0.84, uY = horiz ? 0.96 : 0.97;
        const vC = 0.42;
        const vSlot = (i) => 0.08 + i * 0.1;
        const dvE = s === 0 ? 0.1 : V.lerp(0.1, 0.08, s === 1 ? e : 1);
        // cabeçalho
        T(g, heads[s], 16, 22, { size: 13, weight: 600, color: C.ink, maxW: w - 32 });
        const chosen = (i) => s >= 3 && tk.top.includes(i);
        const pTok = P(uTok, vC), pR = P(uR, vC), pS = P(uS, vC), pY = P(uY, vC);
        const boxes = [];
        for (let i = 0; i < NE; i++) boxes.push(R(uE, vSlot(i), duE, dvE));
        const cen = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
        // as linhas de saída começam depois da nota/contador de cada bloco (à direita no desktop, embaixo no celular);
        // no layout vertical, as de entrada param antes do rótulo Eᵢ
        const inEdge = (b, lab = true) => (horiz ? [b[0], b[1] + b[3] / 2] : [b[0] + b[2] / 2, b[1] - (lab && s >= 1 ? 18 : 0)]);
        const outEdge = (b, lab = true) => (horiz ? [b[0] + b[2] + (!lab ? 0 : s === 5 ? 66 : s >= 2 ? 38 : 0), b[1] + b[3] / 2] : [b[0] + b[2] / 2, b[1] + b[3] + (!lab ? 0 : s === 5 ? 34 : s >= 2 ? 20 : 0)]);
        const shB = horiz ? R(uE, 0.93, duE, 0.12) : R(uE, 0.92, duE, 0.14);
        const showShared = s >= 4;
        const rb = horiz ? R(uR, vC, 0.1, 0.13) : R(uR, vC, 0.07, 0.34);
        // linhas
        const faint = V.alpha(C.muted, 0.35);
        if (s <= 1) {
          boxes.forEach((b) => { V.line(g, pTok[0], pTok[1], ...inEdge(b), { color: faint, lw: 1 }); V.line(g, ...outEdge(b), pY[0], pY[1], { color: faint, lw: 1 }); });
        } else {
          V.line(g, pTok[0], pTok[1], pR[0], pR[1], { color: V.alpha(C.b, 0.8), lw: 2 });
          boxes.forEach((b, i) => {
            const on = chosen(i);
            V.line(g, pR[0], pR[1], ...inEdge(b), { color: on ? C.c : (s >= 3 ? V.alpha(C.muted, 0.15) : V.alpha(C.b, 0.45)), lw: on ? 2.2 : 1, dash: on || s < 3 ? null : [3, 4] });
            if (on && s >= 4) V.line(g, ...outEdge(b), pS[0], pS[1], { color: C.c, lw: 2 });
          });
          if (s >= 4) V.line(g, pS[0], pS[1], pY[0], pY[1], { color: C.a, lw: 2.2 });
          if (s === 3) boxes.forEach((b, i) => { if (chosen(i)) V.line(g, ...outEdge(b), pY[0], pY[1], { color: V.alpha(C.c, 0.6), lw: 1.5, dash: [4, 4] }); });
          if (showShared) { V.line(g, pTok[0], pTok[1], ...inEdge(shB, false), { color: V.alpha(C.d, 0.8), lw: 1.6, dash: [5, 4], dashOffset: -t * 20 }); V.line(g, ...outEdge(shB, false), pS[0], pS[1], { color: V.alpha(C.d, 0.8), lw: 1.6 }); }
        }
        // bloco denso
        if (s === 0) {
          const all = R(uE, 0.43, duE, 0.8);
          V.box(g, all[0] - 4, all[1] - 4, all[2] + 8, all[3] + 8, { r: 12, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 2 });
          const nc = horiz ? 7 : 12, nr = horiz ? 16 : 5;
          for (let a = 0; a < nc; a++) for (let c2 = 0; c2 < nr; c2++) {
            const x = all[0] + (all[2] * (a + 0.5)) / nc, y = all[1] + (all[3] * (c2 + 0.5)) / nr;
            const pulse = 0.5 + 0.5 * Math.sin(t * 3.2 + a * 0.9 + c2 * 1.7);
            V.dot(g, x, y, 2.3, V.alpha(C.a, 0.35 + 0.6 * pulse));
          }
          T(g, 'FFN densa', horiz ? all[0] + all[2] / 2 : all[0] + all[2] / 2, horiz ? all[1] - 12 : all[1] - 10, { size: 12, weight: 600, align: 'center', color: C.a });
        }
        // passo 2: pilhas atrás de cada bloco (o MoE real multiplica os blocos)
        const eStack = s === 1 ? V.ease.out(V.clamp((el - 1.2) / 0.9, 0, 1)) : 0;
        // experts
        boxes.forEach((b, i) => {
          const on = s <= 1 || chosen(i);
          const dim = s >= 3 && !chosen(i);
          const fillA = s === 0 ? 0 : dim ? 0.05 : 0.14;
          if (eStack > 0) {
            const off = horiz ? 5 : 3;
            for (let q = 3; q >= 1; q--) {
              const dx = q * off * eStack, dy = -q * off * eStack;
              V.box(g, b[0] + dx, b[1] + dy, b[2], b[3], { r: 7, fill: C.bg, stroke: V.alpha(C.a, (0.5 - q * 0.1) * eStack), lw: 1 });
            }
            V.box(g, b[0], b[1], b[2], b[3], { r: 7, fill: C.bg });
          }
          if (s > 0) V.box(g, b[0], b[1], b[2], b[3], { r: 7, fill: V.alpha(C.a, fillA), stroke: dim ? V.alpha(C.muted, 0.35) : chosen(i) ? C.c : V.alpha(C.a, 0.8), lw: chosen(i) ? 2.2 : 1.2 });
          // nota (preenchimento)
          if (s >= 2) {
            const sc = tk.s[i] * (s === 2 ? e : 1);
            g.save(); V.rr(g, b[0], b[1], b[2], b[3], 7); g.clip();
            g.fillStyle = V.alpha(dim ? C.muted : C.b, dim ? 0.18 : 0.42);
            if (horiz) g.fillRect(b[0], b[1], b[2] * sc, b[3]); else g.fillRect(b[0], b[1] + b[3] * (1 - sc), b[2], b[3] * sc);
            g.restore();
          }
          // neurônios
          if (s === 0) return;
          const cols = b[2] > b[3] ? 3 : 2, rows = b[2] > b[3] ? 2 : 3;
          for (let a = 0; a < cols; a++) for (let c2 = 0; c2 < rows; c2++) {
            const x = b[0] + b[2] * (horiz ? 0.52 + a * 0.16 : (a + 1) / (cols + 1));
            const y = b[1] + b[3] * (horiz ? (c2 + 1) / (rows + 1) : 0.22 + c2 * 0.2);
            const lit = on && !dim;
            const pulse = lit ? 0.55 + 0.45 * Math.sin(t * 4 + i + a * 1.3 + c2) : 0;
            V.dot(g, x, y, 2.6, lit ? V.alpha(C.a, 0.5 + 0.5 * pulse) : V.alpha(C.muted, 0.25));
          }
          if (s > 0) {
            const lab = 'E' + i;
            if (horiz) T(g, lab, b[0] + 8, b[1] + b[3] / 2 + 4, { size: 11.5, font: 'mono', weight: 600, color: dim ? C.muted : C.ink });
            else T(g, lab, b[0] + b[2] / 2, b[1] - 6, { size: 11, font: 'mono', weight: 600, align: 'center', color: dim ? C.muted : C.ink });
          }
        });
        // partículas (antes dos rótulos, para não passarem por cima do texto)
        parts.forEach((p) => {
          const u = (t * 0.33 + p.o) % 1;
          let pts, col = C.a;
          if (s <= 1) { const b = boxes[p.k % NE]; pts = [pTok, cen(b), pY]; }
          else if (s === 2) { const b = boxes[p.k % NE]; pts = [pTok, pR, cen(b)]; col = C.b; }
          else {
            if (showShared && p.k % 4 === 3) { pts = [pTok, cen(shB), pS, pY]; col = C.d; }
            else { const b = boxes[tk.top[p.k % 2]]; pts = s >= 4 ? [pTok, pR, cen(b), pS, pY] : [pTok, pR, cen(b), pY]; col = C.c; }
          }
          const q = pathPos(pts, u);
          V.dot(g, q[0], q[1], 2.6, V.alpha(col, 0.9));
        });
        // notas embaixo (ou ao lado) de cada bloco, depois das partículas
        if (s >= 2) boxes.forEach((b, i) => {
          if (horiz) T(g, nf(tk.s[i], 2), b[0] + b[2] + 6, b[1] + b[3] / 2 + 4, { size: 11, font: 'mono', color: chosen(i) ? C.c : C.muted });
          else T(g, nf(tk.s[i], 1), b[0] + b[2] / 2, b[1] + b[3] + 14, { size: 11, font: 'mono', align: 'center', color: chosen(i) ? C.c : C.muted, maxW: b[2] + 6 });
        });
        // pesos g (passo 5)
        if (s === 4) boxes.forEach((b, i) => {
          if (!chosen(i)) return;
          const gi = tk.top.indexOf(i);
          const o = outEdge(b), mid = [V.lerp(o[0], pS[0], horiz ? 0.62 : 0.5), V.lerp(o[1], pS[1], horiz ? 0.5 : 0.55)];
          V.box(g, mid[0] - 22, mid[1] - 10, 44, 20, { r: 6, fill: C.c });
          T(g, 'g=' + nf(tk.g[gi], 2), mid[0], mid[1] + 4, { size: 11, font: 'mono', weight: 600, align: 'center', color: onTxt(C) });
        });
        // contadores (passo 6)
        if (s === 5) {
          const cnt = new Array(NE).fill(0);
          for (let q = 0; q <= seqIdx; q++) TOK[SEQ[q % SEQ.length]].top.forEach((i) => cnt[i]++);
          boxes.forEach((b, i) => {
            const txt = '×' + cnt[i];
            if (horiz) T(g, txt, b[0] + b[2] + 40, b[1] + b[3] / 2 + 4, { size: 11.5, font: 'mono', weight: 600, color: C.d });
            else T(g, txt, b[0] + b[2] / 2, b[1] + b[3] + 28, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.d });
          });
        }
        // compartilhado
        if (showShared) {
          const a = s === 4 ? e : 1;
          V.box(g, shB[0], shB[1], shB[2], shB[3], { r: 8, fill: C.bg });
          V.box(g, shB[0], shB[1], shB[2], shB[3], { r: 8, fill: V.alpha(C.d, 0.2 * a), stroke: V.alpha(C.d, a), lw: 1.8 });
          T(g, 'shared', shB[0] + shB[2] / 2, shB[1] + shB[3] / 2 + 4, { size: 11, weight: 600, align: 'center', color: C.d, maxW: shB[2] - 4 });
        }
        // roteador (fundo opaco: as linhas não atravessam o rótulo)
        if (s >= 2) {
          const a = s === 2 ? e : 1;
          V.box(g, rb[0], rb[1], rb[2], rb[3], { r: 9, fill: C.bg });
          V.box(g, rb[0], rb[1], rb[2], rb[3], { r: 9, fill: V.alpha(C.b, 0.22 * a + 0.05), stroke: V.alpha(C.b, a), lw: 1.8 });
          T(g, 'roteador', rb[0] + rb[2] / 2, rb[1] + rb[3] / 2 + 4, { size: 11.5, weight: 600, align: 'center', color: C.ink, maxW: rb[2] - 4 });
        }
        // Σ
        if (s >= 4) { V.dot(g, pS[0], pS[1], 15, C.bg); V.dot(g, pS[0], pS[1], 15, V.alpha(C.a, 0.25)); g.strokeStyle = C.a; g.lineWidth = 1.8; g.beginPath(); g.arc(pS[0], pS[1], 15, 0, Math.PI * 2); g.stroke(); T(g, 'Σ', pS[0], pS[1] + 5, { size: 15, weight: 700, align: 'center', color: C.ink }); }
        // token
        const tw = textW(g, tk.w, 13, 'sans', 600) + 20;
        const pulseT = s === 5 ? V.clamp(1 - ((el % 1.5) / 0.35), 0, 1) : 0;
        V.box(g, pTok[0] - tw / 2, pTok[1] - 15, tw, 30, { r: 15, fill: C.bg });
        V.box(g, pTok[0] - tw / 2, pTok[1] - 15, tw, 30, { r: 15, fill: V.alpha(C.c, 0.2 + 0.3 * pulseT), stroke: C.c, lw: 1.8 });
        T(g, tk.w, pTok[0], pTok[1] + 5, { size: 13, weight: 600, align: 'center', color: C.ink });
        // saída y
        V.box(g, pY[0] - 16, pY[1] - 15, 32, 30, { r: 9, fill: C.bg });
        V.box(g, pY[0] - 16, pY[1] - 15, 32, 30, { r: 9, fill: V.alpha(C.a, 0.2), stroke: C.a, lw: 1.6 });
        T(g, 'y', pY[0], pY[1] + 5, { size: 14, weight: 700, align: 'center', color: C.ink });
        let ft = foot[s];
        if (s === 4) ft = `y = ${nf(tk.g[0], 2)}·E${tk.top[0]}(x) + ${nf(tk.g[1], 2)}·E${tk.top[1]}(x) + Shared(x)`;
        wrap(g, ft, 16, h - bot + 18, w - 32, 13, { size: 11.5, color: C.muted, maxLines: 2 });
      });
    });

    /* ====================================================================
       Figura 2 — O roteador por dentro
       ==================================================================== */
    safe('fig2', () => {
      const fig = $('#fig-roteador');
      fitH(fig, (w) => (w < 600 ? 500 : 400));
      const E = 12, D = 8;
      const Wr = V.randMatrix(E, D, 1107, 0.8);
      const r2 = V.rng(77);
      const defs = [{ w: 'gato', a: 3, b: 8 }, { w: 'def', a: 10, b: 1 }, { w: '∫', a: 6, b: 0 }, { w: 'Paris', a: 5, b: 9 }];
      const TOKS = defs.map((d) => {
        let x = Array.from({ length: D }, (_, j) => 0.9 * Wr[d.a][j] + 0.6 * Wr[d.b][j] + 0.35 * r2.normal());
        const l = Wr.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
        const m = Math.max(...l.map(Math.abs));
        x = x.map((v) => (v * 3.4) / m);
        return { w: d.w, x };
      });
      const BPAT = [0.9, -1, 0.4, -0.7, 1, -0.3, -0.9, 0.6, -0.4, 0.8, -0.8, 0.5];
      const BMAG = { softmax: 0.07, sigmoid: 0.16, sqsp: 0.38 };
      const S = { tok: 0, f: 'sigmoid', k: 3, bias: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Token', options: TOKS.map((t2, i) => ({ v: i, label: t2.w })), value: 0, onChange: (v) => { S.tok = v; upd(); } });
      UI.seg(ctl, { label: 'Função', options: [{ v: 'softmax', label: 'softmax' }, { v: 'sigmoid', label: 'sigmoid' }, { v: 'sqsp', label: '√softplus' }], value: 'sigmoid', onChange: (v) => { S.f = v; upd(); } });
      UI.slider(ctl, { label: 'k', min: 1, max: 6, step: 1, value: 3, onInput: (v) => { S.k = v; upd(); } });
      UI.toggle(ctl, { label: 'somar viés b (só na escolha)', value: false, onChange: (v) => { S.bias = v; upd(); } });
      const rA = UI.readout(ctl, 'escolhidos', '—'), rB = UI.readout(ctl, 'Σ s dos escolhidos', '—'), rC = UI.readout(ctl, 'o viés trocou', '—');
      function calc() {
        const x = TOKS[S.tok].x;
        const l = Wr.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
        let s;
        if (S.f === 'softmax') { const mx = Math.max(...l); const ex = l.map((v) => Math.exp(v - mx)); const z = ex.reduce((a, b) => a + b, 0); s = ex.map((v) => v / z); }
        else if (S.f === 'sigmoid') s = l.map(sig); else s = l.map(sqsp);
        const b = BPAT.map((v) => v * BMAG[S.f]);
        const key = s.map((v, i) => v + (S.bias ? b[i] : 0));
        const ch = topk(key, S.k);
        const ch0 = topk(s, S.k);
        const sm = ch.reduce((a, i) => a + s[i], 0);
        const gw = new Array(E).fill(0); ch.forEach((i) => { gw[i] = s[i] / sm; });
        const inn = ch.filter((i) => !ch0.includes(i)), out = ch0.filter((i) => !ch.includes(i));
        return { x, l, s, b, ch, sm, gw, inn, out, zmax: Math.max(...l) };
      }
      let Q = calc();
      function upd() { Q = calc(); rA.set(Q.ch.slice().sort((a, b) => a - b).map((i) => 'E' + i).join(' ')); rB.set(nf(Q.sm, 2)); rC.set(S.bias ? (Q.inn.length ? Q.out.map((i) => 'E' + i).join(',') + ' → ' + Q.inn.map((i) => 'E' + i).join(',') : 'nada mudou') : 'viés desligado'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 600;
        const rh = mob ? 17 : 22;
        const y0 = 50;
        let x0, cs, xw, lw, bw, bcw, gw2, gap1, xX, xL, xS, xB, xG;
        if (!mob) {
          x0 = 16; cs = 20; xw = 38; lw = 42; bw = 100; bcw = 42; gw2 = 50; gap1 = 18;
          const matW0 = D * (cs + 2) - 2;
          xX = x0 + matW0 + gap1; xL = xX + xw + gap1; xS = xL + lw + gap1 + 4; xB = xS + bw + 6; xG = xB + bcw + 4;
        } else {
          x0 = 44; gap1 = 8; xw = 28; lw = 32; gw2 = 40; bcw = S.bias ? 34 : 0;
          cs = Math.max(8, Math.min(12, Math.floor((w - x0 - 16 - 3 * gap1 - xw - lw - gw2 - bcw - 50) / 8) - 2));
          const matW0 = D * (cs + 2) - 2;
          xX = x0 + matW0 + gap1; xL = xX + xw + gap1; xS = xL + lw + gap1;
          bw = Math.max(40, w - 12 - xS - bcw - gw2 - 6); xB = xS + bw + 4; xG = xB + bcw + 2;
        }
        const matW = D * (cs + 2) - 2;
        const Hm = E * rh - 2;
        T(g, 'ℓ = W_r·x  →  s = f(ℓ)  →  top-' + S.k + ' de 12', x0, 22, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: w - 32 });
        // matriz W_r
        const row = Math.floor(t * 1.6) % E;
        V.matrix(g, { x: x0, y: y0, data: Wr, cell: cs, cellH: rh - 2, gap: 2, cmap: 'div', min: -2, max: 2, hiRow: row, hiColor: C.b });
        T(g, 'W_r  [12 × 8]', x0 + matW / 2, y0 - 10, { size: 11, font: 'mono', align: 'center', color: C.muted });
        T(g, '·', x0 + matW + gap1 / 2, y0 + Hm / 2 + 5, { size: 16, weight: 700, align: 'center', color: C.muted });
        // vetor x (8 entradas, centralizado)
        const xy0 = y0 + (Hm - (D * rh - 2)) / 2;
        T(g, 'x', xX + xw / 2, y0 - 10, { size: 11, font: 'mono', align: 'center', color: C.muted });
        Q.x.forEach((v, j) => {
          const yy = xy0 + j * rh;
          V.box(g, xX, yy, xw, rh - 2, { r: 4, fill: V.cmap(V.clamp(v / 2.2, -1, 1), 'div') });
          T(g, nf(v, 1), xX + xw / 2, yy + rh / 2 + 3, { size: 11, font: 'mono', align: 'center', color: C.ink });
        });
        T(g, '=', xX + xw + gap1 / 2, y0 + Hm / 2 + 5, { size: 15, weight: 700, align: 'center', color: C.muted });
        // logits
        T(g, 'ℓ', xL + lw / 2, y0 - 10, { size: 11, font: 'mono', align: 'center', color: C.muted });
        Q.l.forEach((v, i) => {
          const yy = y0 + i * rh;
          const hi = i === row;
          V.box(g, xL, yy, lw, rh - 2, { r: 4, fill: V.cmap(V.clamp(v / 3.5, -1, 1), 'div'), stroke: hi ? C.b : null, lw: 2 });
          T(g, (v >= 0 ? '+' : '') + nf(v, 1), xL + lw / 2, yy + rh / 2 + 3, { size: 11, font: 'mono', align: 'center', color: C.ink });
        });
        // partícula da linha atual até o logit
        const pr = (t * 1.6) % 1;
        const pa = [x0 + matW, y0 + row * rh + rh / 2], pb = [xL, y0 + row * rh + rh / 2];
        V.dot(g, V.lerp(pa[0], pb[0], pr), V.lerp(pa[1], pb[1], pr), 3, C.b);
        if (!mob) T(g, 'f', xL + lw + gap1 / 2 + 2, y0 + Hm / 2 + 4, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.b });
        // barras de score
        const smax = Math.max(...Q.s) * 1.05;
        T(g, 's', xS + 4, y0 - 10, { size: 11, font: 'mono', color: C.muted });
        if (S.bias) T(g, 'b', xB + bcw / 2, y0 - 10, { size: 11, font: 'mono', align: 'center', color: C.muted });
        T(g, 'g', xG + gw2 / 2, y0 - 10, { size: 11, font: 'mono', align: 'center', color: C.muted });
        for (let i = 0; i < E; i++) {
          const yy = y0 + i * rh;
          const on = Q.ch.includes(i);
          const barW = mob ? bw : bw - 34; // no desktop, o valor fica numa coluna própria à direita da barra
          const L2 = (Q.s[i] / smax) * barW;
          V.box(g, xS, yy + 2, Math.max(2, L2), rh - 6, { r: 3, fill: on ? V.alpha(C.c, 0.75) : V.alpha(C.a, 0.5) });
          if (on) V.box(g, xS - 2, yy, bw + 4, rh - 2, { r: 4, stroke: C.c, lw: 1.4 });
          T(g, 'E' + i, mob ? x0 - 5 : xS - 4, yy + rh / 2 + 3, { size: 11, font: 'mono', align: 'right', color: on ? C.c : C.muted });
          if (!mob) T(g, nf(Q.s[i], 2), xS + barW + 4, yy + rh / 2 + 3, { size: 11, font: 'mono', color: on ? C.c : C.ink });
          if (S.bias && bcw) { const bv = Q.b[i]; T(g, (bv >= 0 ? '+' : '−') + nf(Math.abs(bv), 2), xB + bcw / 2, yy + rh / 2 + 3, { size: 11, font: 'mono', align: 'center', color: bv >= 0 ? C.e : C.c }); }
          const entered = S.bias && Q.inn.includes(i), left = S.bias && Q.out.includes(i);
          if (on) { V.box(g, xG, yy, gw2, rh - 2, { r: 5, fill: V.alpha(entered ? C.e : C.c, 0.2), stroke: entered ? C.e : C.c, lw: 1 }); T(g, (entered ? '▲' : '') + nf(Q.gw[i], 2), xG + gw2 / 2, yy + rh / 2 + 3, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink }); }
          if (left) T(g, '▼ saiu', xG + gw2 / 2, yy + rh / 2 + 3, { size: 11, align: 'center', color: C.c });
        }
        // inset: curva f(ℓ)
        let ix, iy, iw, ih;
        const rightStart = xG + gw2 + 26;
        if (!mob && w - rightStart - 16 >= 170) { ix = rightStart; iy = y0 + 4; iw = w - rightStart - 20; ih = Hm - 18; }
        else { ix = 40; iy = y0 + Hm + 34; iw = w - 60; ih = Math.max(90, h - iy - 34); }
        V.box(g, ix - 30, iy - 22, iw + 42, ih + 50, { r: 10, fill: V.alpha(C.panel || C.bg, 0.35), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        const lmin = -4.5, lmax = 4.5;
        const fcur = (v) => (S.f === 'softmax' ? Math.exp(v - Q.zmax) / Q.s.reduce((a, b, i) => a + Math.exp(Q.l[i] - Q.zmax), 0) : S.f === 'sigmoid' ? sig(v) : sqsp(v));
        const ymx = S.f === 'sigmoid' ? 1.05 : S.f === 'sqsp' ? sqsp(lmax) * 1.05 : Math.max(fcur(lmax), smax) * 1.05;
        const PX = (v) => ix + ((v - lmin) / (lmax - lmin)) * iw, PY = (v) => iy + ih - (v / ymx) * ih;
        V.line(g, ix, iy + ih, ix + iw, iy + ih, { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.line(g, PX(0), iy, PX(0), iy + ih, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 3] });
        g.beginPath();
        for (let q = 0; q <= 80; q++) { const v = lmin + (q / 80) * (lmax - lmin); const yy = PY(Math.min(ymx, fcur(v))); if (q) g.lineTo(PX(v), yy); else g.moveTo(PX(v), yy); }
        g.strokeStyle = C.b; g.lineWidth = 2; g.stroke();
        Q.l.forEach((v, i) => { const on = Q.ch.includes(i); const xx = PX(V.clamp(v, lmin, lmax)), yy = PY(Math.min(ymx, Q.s[i])); V.dot(g, xx, yy, on ? 4.5 : 3.2, on ? C.c : V.alpha(C.a, 0.85)); });
        const fname = S.f === 'softmax' ? 'softmax: e^ℓ / Z  (depende de todos)' : S.f === 'sigmoid' ? 'sigmoid: 1 / (1 + e^−ℓ)  (satura em 1)' : '√softplus: √ln(1 + e^ℓ)  (cresce como √ℓ)';
        T(g, fname, ix - 22, iy - 8, { size: 11, font: 'mono', color: C.b, maxW: iw + 30 });
        T(g, 'ℓ', ix + iw + 6, iy + ih + 4, { size: 11, font: 'mono', color: C.muted });
        T(g, '−4', PX(-4), iy + ih + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        T(g, '0', PX(0), iy + ih + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        T(g, '+4', PX(4), iy + ih + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
      });
    });

    /* ====================================================================
       Figura 3 — Somando as vozes
       ==================================================================== */
    safe('fig3', () => {
      const fig = $('#fig-combinar');
      // mobile: altura fixa que comporta o pior caso (Qwen, 10 linhas × 26 px + painel + medidor + mensagem de 2 linhas)
      fitH(fig, (w) => (w < 640 ? 44 + 260 + 18 + 3 * 52 + 6 + 68 + 2 * 14 + 20 : 420));
      const PRE = {
        glm: { f: 'sigmoid', rho: 2.5, gate: null, ids: [17, 203, 88, 5, 141, 260, 74, 199], s: [0.93, 0.90, 0.86, 0.81, 0.77, 0.72, 0.69, 0.66] },
        ds: { f: '√softplus', rho: 1.5, gate: null, ids: [12, 230, 97, 45, 181, 3], s: [2.12, 1.94, 1.71, 1.55, 1.41, 1.30] },
        qwen: { f: 'softmax', rho: 1.0, gate: 0.62, ids: [301, 17, 455, 92, 250, 6, 388, 127, 64, 499], s: [0.21, 0.12, 0.09, 0.07, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02] },
      };
      const rr = V.rng(311);
      const OUT = Array.from({ length: 10 }, () => Array.from({ length: 6 }, () => rr.normal() * 0.75));
      const SH = Array.from({ length: 6 }, () => rr.normal() * 0.8);
      const S = { m: 'glm', norm: true, rho: 2.5, bug: false };
      const ctl = UI.controls(fig);
      let sRho = null;
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM · sigmoid' }, { v: 'ds', label: 'DeepSeek · √softplus' }, { v: 'qwen', label: 'Qwen · softmax' }], value: 'glm', onChange: (v) => { S.m = v; if (sRho) sRho.set(PRE[v].rho); upd(); } });
      UI.toggle(ctl, { label: 'norm_topk_prob', value: true, onChange: (v) => { S.norm = v; upd(); } });
      sRho = UI.slider(ctl, { label: 'routed scaling', min: 0.5, max: 3, step: 0.1, value: 2.5, fmt: (v) => nf(v, 1), onInput: (v) => { S.rho = v; upd(); } });
      UI.toggle(ctl, { label: 'bug do alias: k − 1', value: false, onChange: (v) => { S.bug = v; upd(); } });
      const rS = UI.readout(ctl, 'Σ g', '—'), rD = UI.readout(ctl, 'desvio vs. treinado', '—'), rR = UI.readout(ctl, '‖roteado‖ / ‖shared‖', '—');
      const norm2 = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
      function compute(m, norm, rho, bug) {
        const P = PRE[m];
        const k = P.s.length - (bug ? 1 : 0);
        const ss = P.s.slice(0, k);
        const sm = ss.reduce((a, b) => a + b, 0);
        const gw = ss.map((v) => (norm ? v / sm : v) * rho);
        const yr = new Array(6).fill(0);
        gw.forEach((gv, i) => OUT[i].forEach((o, j) => { yr[j] += gv * o; }));
        const gate = P.gate == null ? 1 : P.gate;
        const sh = SH.map((v) => v * gate);
        const y = yr.map((v, j) => v + sh[j]);
        return { k, gw, yr, sh, y, sg: gw.reduce((a, b) => a + b, 0) };
      }
      let Q = compute('glm', true, 2.5, false);
      function upd() {
        Q = compute(S.m, S.norm, S.rho, S.bug);
        const ref = compute(S.m, true, PRE[S.m].rho, false);
        const dv = norm2(Q.y.map((v, j) => v - ref.y[j])) / norm2(ref.y);
        Q.dev = dv;
        rS.set(nf(Q.sg, 2));
        rD.set(dv < 0.0005 ? '0%' : nf(dv * 100, dv < 0.1 ? 1 : 0) + '%');
        rR.set(nf(norm2(Q.yr) / Math.max(1e-6, norm2(Q.sh)), 2));
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), P = PRE[S.m];
        const mob = w < 640;
        const x0 = 16, y0 = 44;
        const nRows = P.s.length;
        const leftW = mob ? w - 32 : Math.min(w * 0.58, 460);
        const rowH = mob ? Math.min(28, 260 / nRows) : Math.min(36, (h - 64) / nRows);
        const vcs = mob ? Math.min(20, (leftW - 170) / 6 - 2) : Math.min(24, (leftW - 200) / 6 - 2);
        const vx = x0 + leftW - 6 * (vcs + 2);
        const gbx = x0 + (mob ? 78 : 96), gbw = vx - gbx - 14;
        T(g, `${P.f}${S.norm ? ' → renormaliza' : ''} → × ρ = ${nf(S.rho, 1)}${P.gate != null ? '   ·   shared × σ(w_s·x) = ' + nf(P.gate, 2) : ''}`, x0, 22, { size: 12.5, weight: 600, color: C.ink, maxW: w - 32 });
        T(g, 'expert · s', x0, y0 - 6, { size: 11, color: C.muted });
        T(g, 'peso g', gbx, y0 - 6, { size: 11, color: C.muted });
        T(g, 'Eₑ(x)', vx, y0 - 6, { size: 11, color: C.muted });
        const gmax = Math.max(1.2, ...Q.gw) * 1.05;
        const act = Math.floor(t * 1.8) % Q.k;
        for (let i = 0; i < nRows; i++) {
          const yy = y0 + i * rowH;
          const dead = i >= Q.k;
          const al = dead ? 0.25 : 1;
          if (i === act && !dead) V.box(g, x0 - 6, yy - 1, leftW + 12, rowH - 2, { r: 6, fill: V.alpha(C.b, 0.12) });
          T(g, 'E' + P.ids[i], x0, yy + rowH / 2 + 3, { size: 11, font: 'mono', weight: 600, color: dead ? C.muted : C.ink, alpha: al });
          T(g, nf(P.s[i], P.s[i] < 0.1 ? 3 : 2), x0 + (mob ? 38 : 46), yy + rowH / 2 + 3, { size: 11, font: 'mono', color: C.muted, alpha: al });
          if (!dead) {
            const gv = Q.gw[i];
            const L = Math.min(gbw, (gv / gmax) * gbw);
            V.box(g, gbx, yy + 4, Math.max(2, L), rowH - 10, { r: 3, fill: V.alpha(C.c, 0.7) });
            T(g, nf(gv, 2), gbx + Math.min(gbw - 26, L + 4), yy + rowH / 2 + 3, { size: 11, font: 'mono', color: C.ink });
          } else {
            T(g, 'descartado (k − 1)', gbx, yy + rowH / 2 + 3, { size: 11, color: C.c, alpha: 0.8 });
          }
          OUT[i].forEach((v, j) => { V.box(g, vx + j * (vcs + 2), yy + 2, vcs, rowH - 6, { r: 3, fill: V.cmap(V.clamp(v / 2, -1, 1), 'div') }); });
          if (dead) { g.globalAlpha = 0.7; V.line(g, vx - 2, yy + rowH / 2, vx + 6 * (vcs + 2), yy + rowH / 2, { color: C.c, lw: 1.5 }); g.globalAlpha = 1; }
        }
        // painel direito (ou abaixo)
        let px, py, pw;
        if (mob) { px = x0; py = y0 + nRows * rowH + 18; pw = w - 32; } else { px = x0 + leftW + 30; py = y0 + 6; pw = w - px - 16; }
        const cs2 = Math.min(mob ? 40 : 34, (pw - 70) / 6 - 3);
        const vecRow = (label, vec, yy, col, note) => {
          T(g, label, px, yy + 13, { size: 11.5, weight: 600, color: col });
          if (note) T(g, note, px + pw, yy + 13, { size: 11, color: C.muted, align: 'right' });
          vec.forEach((v, j) => {
            const xx = px + j * (cs2 + 3), yb = yy + 20;
            V.box(g, xx, yb, cs2, 26, { r: 4, fill: V.cmap(V.clamp(v / 2, -1, 1), 'div') });
            if (cs2 >= 30) T(g, nf(v, 1), xx + cs2 / 2, yb + 17, { size: 11, font: 'mono', align: 'center', color: C.ink });
          });
        };
        const step = mob ? 52 : 58;
        vecRow('Σ gₑ·Eₑ(x)', Q.yr, py, C.c, 'parte roteada');
        vecRow(P.gate != null ? '+ σ(w_s·x)·Shared(x)' : '+ Shared(x)', Q.sh, py + step, C.d, 'sempre ativo');
        vecRow('= y', Q.y, py + 2 * step, C.a, 'volta ao residual');
        // pulso da linha ativa até Σ
        const pr = (t * 1.8) % 1;
        const a0 = [vx + 6 * (vcs + 2), y0 + act * rowH + rowH / 2], a1 = [px + 3 * (cs2 + 3), py + 30];
        if (!mob) { const q = V.along(a0[0], a0[1], a1[0], a1[1], V.ease.inOut(pr)); V.dot(g, q[0], q[1], 3.5, C.c); }
        // medidor
        const my = py + 3 * step + 6;
        const nR = norm2(Q.yr), nS = norm2(Q.sh), mx = Math.max(4, nR, nS);
        const mw = pw - 4;
        T(g, 'tamanho de cada parte', px, my + 10, { size: 11, color: C.muted });
        V.box(g, px, my + 18, Math.max(3, (nR / mx) * mw), 12, { r: 4, fill: V.alpha(C.c, 0.8) });
        T(g, 'roteada ' + nf(nR, 2), px + Math.min(mw - 90, (nR / mx) * mw + 6), my + 28, { size: 11, font: 'mono', color: C.ink });
        V.box(g, px, my + 36, Math.max(3, (nS / mx) * mw), 12, { r: 4, fill: V.alpha(C.d, 0.8) });
        T(g, 'shared ' + nf(nS, 2), px + Math.min(mw - 90, (nS / mx) * mw + 6), my + 46, { size: 11, font: 'mono', color: C.ink });
        const dv = Q.dev || 0;
        const msg = dv < 0.0005 ? 'configuração treinada: é isto que o kernel precisa reproduzir' : dv < 0.3 ? 'desvio sutil: o modelo roda e responde, só que um pouco diferente' : 'desvio grosseiro: a parte roteada saiu da escala treinada';
        wrap(g, msg, px, my + 68, pw, 14, { size: 11.5, color: dv < 0.0005 ? C.a : dv < 0.3 ? C.d : C.c, weight: 600, maxLines: 2 });
      });
    });

    /* ====================================================================
       Figura 4 — Mapa de metrô das FFNs
       ==================================================================== */
    safe('fig4', () => {
      const fig = $('#fig-metro');
      fitH(fig, (w) => (w < 600 ? 470 : 430));
      const MODELS = [
        { id: 'glm', name: 'GLM-5.3-Flash', n: 45, extra: 1, type: (i) => (i >= 45 ? 'draft' : i < 3 ? 'dense' : 'moe'), k: 8,
          info: (i) => (i >= 45 ? ['camada 45 · MTP', 'MoE igual ao do tronco; usada só como rascunho (spec decode)', 7.274, 0.2265] : i < 3 ? [`camada ${i} · FFN densa`, 'SwiGLU de largura 12.288, todos os neurônios para todo token', 0.151, 0.151] : [`camada ${i} · MoE`, '288 roteados (top-8, sigmoid, viés) + 1 shared de 2048', 7.274, 0.2265]) },
        { id: 'qwen', name: 'Qwen3.8-Flash-Next', n: 48, extra: 1, type: (i) => (i >= 48 ? 'draft' : 'moe'), k: 10,
          info: (i) => (i >= 48 ? ['MTP (mtp.layers.48)', 'MoE de 512 experts em FP8 com blocos 128×128', 2.523, 0.0541] : [`camada ${i} · MoE`, '512 roteados (top-10, softmax) + shared de 640 com gate sigmoide', 2.523, 0.0541]) },
        { id: 'ds', name: 'DeepSeek-V4-Flash', n: 43, extra: 3, type: (i) => (i >= 43 ? 'draft' : i < 3 ? 'hash' : 'moe'), k: 6,
          info: (i) => (i >= 43 ? [`estágio DSpark ${i - 43}`, 'bloco de rascunho com MoE completo (256 experts)', 6.468, 0.1762] : i < 3 ? [`camada ${i} · MoE por hash`, 'escolha por tid2eid[id] (6 de 256) + 1 shared; pesos ainda do gate', 6.468, 0.1762] : [`camada ${i} · MoE`, '256 roteados (top-6, √softplus, viés) + 1 shared de 2048', 6.468, 0.1762]) },
      ];
      const S = { mode: 'all' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Linha', options: [{ v: 'all', label: 'os três' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'all', onChange: (v) => { S.mode = v; } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.d, 'FFN densa'], [C0.f, 'MoE por hash'], [C0.a, 'MoE com roteador'], [C0.b, 'rascunho (MTP/DSpark)']]);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 600;
        const list = S.mode === 'all' ? MODELS : MODELS.filter((m) => m.id === S.mode);
        const infoH = mob ? 128 : 112;
        const areaH = h - infoH - 20;
        const single = list.length === 1;
        const lineH = single ? 90 : areaH / list.length;
        const x0 = mob ? 20 : 150, x1 = w - 20;
        const colOf = (ty) => (ty === 'dense' ? C.d : ty === 'hash' ? C.f : ty === 'draft' ? C.b : C.a);
        let hover = null;
        const trains = [];
        list.forEach((m, li) => {
          const N = m.n + m.extra;
          const cy = single ? 62 : 16 + li * lineH + lineH * (mob ? 0.62 : 0.55);
          const sp = (x1 - x0) / (N - 1);
          const rad = Math.max(2.4, Math.min(6, sp * 0.36));
          const mc = C[m.id];
          if (mob) T(g, m.name, x0, cy - 22, { size: 12, weight: 600, color: mc });
          else { T(g, m.name, 16, cy + 4, { size: 12.5, weight: 600, color: mc, maxW: x0 - 26 }); }
          V.line(g, x0, cy, x0 + (m.n - 1) * sp, cy, { color: V.alpha(mc, 0.55), lw: 3.2 });
          if (m.extra) V.line(g, x0 + (m.n - 1) * sp, cy, x1, cy, { color: V.alpha(C.b, 0.55), lw: 3.2, dash: [4, 4] });
          const speed = 3.2;
          const pos = (t * speed + li * 7) % (N + 3);
          const cur = Math.min(N - 1, Math.floor(pos));
          trains.push({ m, i: cur, cy, sp });
          for (let i = 0; i < N; i++) {
            const x = x0 + i * sp, ty = m.type(i), col = colOf(ty);
            const near = Math.abs(pos - i) < 0.6;
            V.dot(g, x, cy, rad + (near ? 1.5 : 0), col);
            V.dot(g, x, cy, Math.max(1, rad - 1.6), C.bg);
            if (H.pointer.inside && Math.abs(H.pointer.x - x) < Math.max(sp / 2, 5) && Math.abs(H.pointer.y - cy) < 16) hover = { m, i };
            if (near && pos < N) {
              const f = pos - Math.floor(pos);
              if (ty === 'dense') { V.glow(g, x, cy, 16 + 8 * f, C.d, 0.8); }
              else if (ty === 'hash') { T(g, '#', x, cy - 12 - 6 * f, { size: 12, weight: 700, align: 'center', color: C.f }); for (let q = 0; q < m.k; q++) { const a = -Math.PI / 2 + (q - (m.k - 1) / 2) * 0.28; V.dot(g, x + Math.cos(a) * (10 + 12 * f), cy + Math.sin(a) * (10 + 12 * f), 1.8, V.alpha(C.f, 1 - f)); } }
              else { for (let q = 0; q < m.k; q++) { const a = -Math.PI / 2 + (q - (m.k - 1) / 2) * 0.24; V.dot(g, x + Math.cos(a) * (10 + 14 * f), cy + Math.sin(a) * (10 + 14 * f), 1.8, V.alpha(ty === 'draft' ? C.b : C.a, 1 - f)); } }
            }
          }
          // rótulos de índice
          const labs = m.id === 'glm' ? [0, 3, 44] : m.id === 'qwen' ? [0, 47] : [0, 3, 42];
          labs.forEach((i) => T(g, String(i), x0 + i * sp, cy + 18, { size: 11, font: 'mono', align: 'center', color: C.muted }));
          T(g, m.id === 'ds' ? 'DSpark' : 'MTP', Math.min(x1, x0 + (m.n + (m.extra - 1) / 2) * sp), cy - 12, { size: 11, align: m.id === 'ds' ? 'center' : 'right', color: C.b });
          if (single) {
            const bTop = cy + 40, bBot = h - infoH - 30, bH = bBot - bTop;
            T(g, 'parâmetros da FFN por camada: guardados (menta) × usados por token (rosa), mesma escala', x0, bTop - 8, { size: 11, color: C.muted, maxW: x1 - x0 });
            V.line(g, x0 - 6, bBot, x1 + 6, bBot, { color: V.alpha(C.muted, 0.4), lw: 1 });
            for (let i = 0; i < N; i++) {
              const inf2 = m.info(i), x = x0 + i * sp, bw2 = Math.max(2, Math.min(10, sp * 0.62));
              const ht = (inf2[2] / 7.3) * bH, ha = Math.max(2, (inf2[3] / 7.3) * bH);
              V.box(g, x - bw2 / 2, bBot - ht, bw2, ht, { r: 2, fill: V.alpha(colOf(m.type(i)), 0.55) });
              V.box(g, x - bw2 / 2, bBot - ha, bw2, ha, { r: 1, fill: C.c });
            }
            T(g, '7,3 B', x0 - 8, bTop + 12, { size: 11, font: 'mono', align: 'right', color: C.muted });
          }
          // trem
          if (pos < N) { const tx = x0 + pos * sp; V.glow(g, tx, cy, 12, mc, 0.9); V.dot(g, tx, cy, 3.4, C.ink); }
        });
        // painel de informação
        let sel;
        if (hover) sel = hover;
        else { const which = S.mode === 'all' ? Math.floor(t / 4) % trains.length : 0; sel = trains[which]; }
        const inf = sel.m.info(sel.i);
        const iy = h - infoH - 4;
        V.box(g, 12, iy, w - 24, infoH, { r: 10, fill: V.alpha(C[sel.m.id], 0.07), stroke: V.alpha(C[sel.m.id], 0.4), lw: 1 });
        T(g, sel.m.name + ' · ' + inf[0], 24, iy + 20, { size: 12.5, weight: 600, color: C[sel.m.id], maxW: w - 48 });
        wrap(g, inf[1], 24, iy + 38, w - 48, 14, { size: 11.5, color: C.ink, maxLines: 2 });
        const by = iy + (mob ? 72 : 58), bw = w - 48 - (mob ? 0 : 150);
        const tot = inf[2], act = inf[3], mx = 7.3;
        const bx = mob ? 24 : 170;
        if (!mob) { T(g, 'guardado', 24, by + 10, { size: 11, color: C.muted }); T(g, 'usado por token', 24, by + 32, { size: 11, color: C.muted }); }
        V.box(g, bx, by, Math.max(3, (tot / mx) * bw), 12, { r: 4, fill: V.alpha(C.a, 0.75) });
        T(g, (tot >= 1 ? nf(tot, 2) + ' B' : ni(tot * 1000) + ' M') + (mob ? ' guardados' : ''), bx + Math.min(bw - 120, (tot / mx) * bw + 6), by + 10, { size: 11, font: 'mono', color: C.ink });
        V.box(g, bx, by + 22, Math.max(3, (act / mx) * bw), 12, { r: 4, fill: C.c });
        T(g, ni(act * 1000) + ' M' + (mob ? ' por token' : '') + '  (' + nf((act / tot) * 100, act / tot < 0.1 ? 1 : 0) + '%)', bx + (act / mx) * bw + 8, by + 32, { size: 11, font: 'mono', color: C.ink });
      });
    });

    /* ====================================================================
       Figura 5 — Colapso e balanceamento (brinquedo)
       ==================================================================== */
    safe('fig5', () => {
      const fig = $('#fig-balanco');
      fitH(fig, (w) => (w < 600 ? 470 : 430));
      const NE = 16, K = 2, D = 4, B = 48;
      const rg = V.rng(515);
      const unit = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
      const EXP = Array.from({ length: NE }, () => unit(Array.from({ length: D }, () => rg.normal())));
      const CEN = Array.from({ length: 4 }, () => unit(Array.from({ length: D }, () => rg.normal())));
      const S = { mode: 'none', str: 0.6 };
      const st = {};
      function reset() {
        const r0 = V.rng(99);
        st.q = Array.from({ length: NE }, () => r0.normal() * 0.15);
        st.a = new Array(NE).fill(0); st.b = new Array(NE).fill(0);
        st.ema = new Array(NE).fill(K / NE); st.hist = []; st.dist = 0; st.steps = 0; st.acc = 0; st.parts = []; st.rng = V.rng(2024); st.ymax = 0;
        // pré-roda alguns passos para o gráfico de CV já nascer com curva
        for (let q = 0; q < 30; q++) step();
        st.parts = [];
      }
      reset();
      function step() {
        const r = st.rng;
        const cnt = new Array(NE).fill(0);
        let dist = 0;
        const sample = [];
        for (let tt = 0; tt < B; tt++) {
          const c = CEN[r.int(4)];
          const x = c.map((v) => v + 0.55 * r.normal());
          const lg = EXP.map((e, i) => 2.2 * (e[0] * x[0] + e[1] * x[1] + e[2] * x[2] + e[3] * x[3]) + st.q[i]);
          const lgA = lg.map((v, i) => v + (S.mode === 'aux' ? st.a[i] : 0));
          const s = lgA.map(sig);
          const key = s.map((v, i) => v + (S.mode === 'bias' ? st.b[i] : 0));
          const ch = topk(key, K);
          ch.forEach((i) => cnt[i]++);
          if (S.mode === 'aux') {
            const sw = ch.reduce((a, i) => a + s[i], 0), s0 = ch.map((i) => sig(lg[i])), sw0 = s0.reduce((a, b) => a + b, 0);
            ch.forEach((i, q) => { dist += Math.abs(s[i] / sw - s0[q] / sw0); });
          }
          if (tt < 3) sample.push(ch[0]);
        }
        const mean = K / NE;
        for (let i = 0; i < NE; i++) {
          const ld = cnt[i] / B, rel = ld / mean - 1;
          st.q[i] = V.clamp(st.q[i] + 0.035 * rel, -4, 4);
          if (S.mode === 'aux') st.a[i] = V.clamp(st.a[i] - S.str * 0.08 * rel, -6, 6);
          if (S.mode === 'bias') st.b[i] = V.clamp(st.b[i] - S.str * 0.012 * Math.sign(rel), -1, 1);
          st.ema[i] = 0.9 * st.ema[i] + 0.1 * ld;
        }
        st.dist = 0.85 * st.dist + 0.15 * (dist / B);
        const m = st.ema.reduce((a, b) => a + b, 0) / NE;
        const sd = Math.sqrt(st.ema.reduce((a, b) => a + (b - m) * (b - m), 0) / NE);
        st.hist.push(sd / m); if (st.hist.length > 260) st.hist.shift();
        st.steps++;
        sample.forEach((i) => st.parts.push({ i, y: 0, v: 0 }));
        if (st.parts.length > 40) st.parts.splice(0, st.parts.length - 40);
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'none', label: 'sem balanceamento' }, { v: 'aux', label: 'aux loss' }, { v: 'bias', label: 'viés sem loss' }], value: 'none', onChange: (v) => { S.mode = v; reset(); } });
      UI.slider(ctl, { label: 'força (α ou γ)', min: 0, max: 1, step: 0.05, value: 0.6, fmt: (v) => nf(v, 2), onInput: (v) => { S.str = v; } });
      UI.button(ctl, '↺ reiniciar', () => reset());
      const rCV = UI.readout(ctl, 'CV da carga', '—'), rId = UI.readout(ctl, 'experts ociosos', '—'), rDs = UI.readout(ctl, 'distorção dos pesos', '—');
      let lastT = 0;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const dt = Math.min(0.2, Math.max(0, t - lastT)); lastT = t;
        st.acc += dt;
        let n = 0;
        while (st.acc > 0.07 && n < 3) { step(); st.acc -= 0.07; n++; }
        if (st.acc > 0.3) st.acc = 0;
        const mean = K / NE;
        const cv = st.hist.length ? st.hist[st.hist.length - 1] : 0;
        const idle = st.ema.filter((v) => v < 0.25 * mean).length;
        rCV.set(nf(cv, 2)); rId.set(String(idle)); rDs.set(S.mode === 'bias' ? '0 (por construção)' : S.mode === 'none' ? '0 (sem mecanismo)' : nf(st.dist, 3));
        const x0 = 16, W = w - 32;
        const topY = 44, barH = h * 0.44;
        const bw = W / NE;
        // topo da escala acompanha a maior carga (suavizado), entre 50% e 100% dos tokens
        const yTarget = V.clamp(Math.max(0.5, Math.max(...st.ema) * 1.15), 0.5, 1);
        st.ymax = st.ymax ? V.lerp(st.ymax, yTarget, 0.06) : yTarget;
        const ymax = st.ymax;
        const hOf = (v) => Math.sqrt(Math.min(1, v / ymax)) * barH;
        T(g, S.mode === 'none' ? 'sem balanceamento: rico fica mais rico' : S.mode === 'aux' ? 'aux loss: empurra os logits (e os pesos) dos sobrecarregados' : 'viés sem loss: b entra só na escolha', x0, 22, { size: 12.5, weight: 600, color: C.ink, maxW: W });
        T(g, 'passo ' + st.steps, w - 16, 22, { size: 11, font: 'mono', align: 'right', color: C.muted });
        const base = topY + barH;
        for (let i = 0; i < NE; i++) {
          const v = st.ema[i];
          const hh = hOf(v);
          const col = v < 0.25 * mean ? V.alpha(C.muted, 0.5) : v > 2 * mean ? C.c : C.a;
          V.box(g, x0 + i * bw + 2, base - hh, bw - 4, Math.max(2, hh), { r: 3, fill: V.alpha(col, 0.85) });
          if (bw >= 26) T(g, 'E' + i, x0 + i * bw + bw / 2, base + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
          const adj = S.mode === 'bias' ? st.b[i] / 0.4 : S.mode === 'aux' ? st.a[i] / 3 : 0;
          if (Math.abs(adj) > 0.02) {
            const L = Math.max(-1, Math.min(1, adj)) * 24;
            const ax = x0 + i * bw + bw / 2, ay = base + 40;
            V.arrow(g, ax, ay, ax, ay - L, { color: L > 0 ? C.e : S.mode === 'aux' ? C.b : C.c, lw: 1.8, head: 6 });
          }
        }
        const my = base - hOf(mean);
        V.line(g, x0, my, x0 + W, my, { color: C.d, lw: 1.5, dash: [5, 4] });
        T(g, 'carga ideal k/E = 1/8 dos tokens', x0 + W - 4, my - 5, { size: 11, align: 'right', color: C.d });
        T(g, 'escala √ · topo = ' + ni(ymax * 100) + '% dos tokens', x0, topY + 4, { size: 11, color: C.muted });
        // partículas caindo
        st.parts.forEach((p) => { p.v += 0.6; p.y += p.v; });
        st.parts = st.parts.filter((p) => p.y < barH);
        st.parts.forEach((p) => V.dot(g, x0 + p.i * bw + bw / 2, topY + 14 + p.y, 2.2, V.alpha(C.b, 0.8)));
        // histórico do CV
        const cy0 = base + 74, ch = h - cy0 - 20;
        if (ch > 30) {
          T(g, 'coeficiente de variação da carga (0 = uniforme)', x0, cy0 - 6, { size: 11, color: C.muted });
          V.line(g, x0, cy0 + ch, x0 + W, cy0 + ch, { color: V.alpha(C.muted, 0.4), lw: 1 });
          const cvMax = 3;
          g.beginPath();
          st.hist.forEach((v, i) => { const xx = x0 + (i / 259) * W, yy = cy0 + ch - Math.min(1, v / cvMax) * ch; if (i) g.lineTo(xx, yy); else g.moveTo(xx, yy); });
          g.strokeStyle = S.mode === 'none' ? C.c : S.mode === 'aux' ? C.b : C.e; g.lineWidth = 2; g.stroke();
          T(g, '3', x0 + 2, cy0 + 12, { size: 11, font: 'mono', color: C.muted });
          T(g, '0', x0 + 2, cy0 + ch - 4, { size: 11, font: 'mono', color: C.muted });
        }
      });
    });

    /* ====================================================================
       Figura 6 — Roteamento limitado por grupos
       ==================================================================== */
    safe('fig6', () => {
      const fig = $('#fig-grupos');
      fitH(fig, (w) => (w < 600 ? 304 : 400));
      const NG = 8, PG = 32, E = NG * PG, K = 8, TG = 4;
      const rg = V.rng(606);
      function sample(r) { const u = Array.from({ length: NG }, () => r.normal() * 0.2); return Array.from({ length: E }, (_, i) => sig(u[Math.floor(i / PG)] + r.normal())); }
      function route(s, grouped) {
        if (!grouped) return { ch: topk(s, K), groups: null, gs: null };
        const gs = Array.from({ length: NG }, (_, gI) => { const a = s.slice(gI * PG, gI * PG + PG).sort((x, y) => y - x); return a[0] + a[1]; });
        const groups = topk(gs, TG);
        const masked = s.map((v, i) => (groups.includes(Math.floor(i / PG)) ? v : -1));
        return { ch: topk(masked, K), groups, gs };
      }
      const nodes = (ch) => new Set(ch.map((i) => Math.floor(i / PG))).size;
      const avg = { free: 0, grp: 0 };
      { const r = V.rng(9); for (let q = 0; q < 2000; q++) { const s = sample(r); avg.free += nodes(route(s, false).ch); avg.grp += nodes(route(s, true).ch); } avg.free /= 2000; avg.grp /= 2000; }
      const S = { grouped: false, s: sample(rg), t0: 0, tick: 0 };
      let R = route(S.s, false);
      const ctl = UI.controls(fig);
      let tNow = 0;
      const nextTok = () => { S.s = sample(rg); R = route(S.s, S.grouped); S.t0 = tNow; upd(); };
      UI.seg(ctl, { label: 'Roteamento', options: [{ v: false, label: 'livre (GLM, V4)' }, { v: true, label: 'grupos (V3: 8 grupos, top-4)' }], value: false, onChange: (v) => { S.grouped = v; R = route(S.s, v); S.t0 = tNow; upd(); } });
      UI.button(ctl, '▶ próximo token', nextTok, { primary: true });
      const rN = UI.readout(ctl, 'nós tocados', '—'), rM = UI.readout(ctl, 'média (2.000 tokens)', '—');
      function upd() { rN.set(nodes(R.ch) + ' de 8'); rM.set(nf(S.grouped ? avg.grp : avg.free, 1)); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        tNow = t;
        const C = V.C();
        if (t - S.t0 > 4.2) nextTok();
        const ph = t - S.t0;
        const mob = w < 600;
        const perRow = 4;
        const gap = mob ? 8 : 16;
        const bw = (w - 32 - gap * (perRow - 1)) / perRow;
        const cs = Math.min(15, (bw - 10) / 8 - 1.2);
        const bh = 4 * (cs + 1.5) + 12;
        const head = S.grouped ? '1º escolhe 4 grupos (soma dos 2 melhores), depois 8 experts dentro deles' : 'top-8 direto entre os 256 experts';
        // no celular o título quebra em 2 linhas (em vez de ser espremido) e o resto desce junto
        if (mob) wrap(g, head, 16, 20, w - 32, 15, { size: 12, weight: 600, color: C.ink, maxLines: 2 });
        else T(g, head, 16, 20, { size: 12, weight: 600, color: C.ink, maxW: w - 150 });
        const tokY = mob ? 46 : 30; // posição fixa: não pula ao trocar de modo
        const gy0 = tokY + 48;
        const rowGap = mob ? 36 : 40;
        const tx = w / 2;
        V.box(g, tx - 30, tokY + 2, 60, 24, { r: 12, fill: V.alpha(C.c, 0.25), stroke: C.c, lw: 1.5 });
        T(g, 'token', tx, tokY + 18, { size: 11.5, weight: 600, align: 'center', color: C.ink });
        const showSel = ph > (S.grouped ? 1.9 : 1.0);
        const showGrp = S.grouped && ph > 0.9;
        const selIn = V.clamp((ph - (S.grouped ? 1.9 : 1.0)) / 0.6, 0, 1);
        const gridW = 8 * (cs + 1.2) - 1.2;
        for (let gI = 0; gI < NG; gI++) {
          const rI = Math.floor(gI / perRow), cI = gI % perRow;
          const bx = 16 + cI * (bw + gap), by = gy0 + rI * (bh + rowGap);
          const inCh = R.ch.filter((i) => Math.floor(i / PG) === gI).length;
          if (showSel && inCh) { g.globalAlpha = selIn; V.line(g, tx, tokY + 26, bx + bw / 2, by - 4, { color: C.c, lw: 1 + inCh * 0.8 }); g.globalAlpha = 1; }
        }
        for (let gI = 0; gI < NG; gI++) {
          const rI = Math.floor(gI / perRow), cI = gI % perRow;
          const bx = 16 + cI * (bw + gap), by = gy0 + rI * (bh + rowGap);
          const gOn = !S.grouped || !showGrp || R.groups.includes(gI);
          const inCh = R.ch.filter((i) => Math.floor(i / PG) === gI).length;
          V.box(g, bx, by, bw, bh, { r: 8, fill: C.bg });
          V.box(g, bx, by, bw, bh, { r: 8, fill: V.alpha(gOn ? C.a : C.muted, gOn ? 0.07 : 0.04), stroke: showGrp && gOn ? C.b : V.alpha(C.muted, 0.35), lw: showGrp && gOn ? 2 : 1 });
          for (let j = 0; j < PG; j++) {
            const i = gI * PG + j;
            const x = bx + (bw - gridW) / 2 + (j % 8) * (cs + 1.2), y = by + 6 + Math.floor(j / 8) * (cs + 1.5);
            const v = S.s[i] * V.clamp(ph / 0.7, 0, 1);
            const sel = showSel && R.ch.includes(i);
            g.fillStyle = sel ? C.c : V.alpha(gOn ? C.a : C.muted, 0.12 + 0.75 * v * (gOn ? 1 : 0.4));
            V.rr(g, x, y, cs, cs, 2); g.fill();
            if (sel && selIn < 1) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x - 2, y - 2, cs + 4, cs + 4, 3); g.stroke(); }
          }
          T(g, 'nó ' + gI, bx + bw / 2, by + bh + 14, { size: 11, align: 'center', color: inCh && showSel ? C.c : C.muted, weight: inCh && showSel ? 600 : 400 });
          if (S.grouped && R.gs) { const gv = R.gs[gI] / 2; V.box(g, bx + 4, by + bh + 20, (bw - 8) * gv * (showGrp ? 1 : V.clamp(ph / 0.9, 0, 1)), 5, { r: 2, fill: showGrp && R.groups.includes(gI) ? C.b : V.alpha(C.muted, 0.5) }); }
        }
        const rows = Math.ceil(NG / perRow);
        const yb = gy0 + rows * (bh + rowGap) + 4;
        if (yb < h - 10) wrap(g, `neste token: ${nodes(R.ch)} nós. Média em 2.000 tokens: livre ${nf(avg.free, 1)} nós · com grupos ${nf(avg.grp, 1)} nós (máx. 4).`, 16, yb, w - 32, 14, { size: 11.5, color: C.muted, maxLines: 2 });
      });
    });

    /* ====================================================================
       Figura 7 — Hash MoE e o bug #175
       ==================================================================== */
    safe('fig7', () => {
      const fig = $('#fig-hash');
      const IMG = 129264;
      const SEQS = {
        text: [['O', 46], [' gato', 15210], [' subiu', 40211], [' no', 1096], [' telhado', 77302]],
        img: [['O', 46], [' gato', 15210], ['img', IMG], ['img', IMG], ['img', IMG], ['img', IMG], ['img', IMG], [' dorme', 90218]],
      };
      const rowFor = (id) => { const r = V.rng(id * 7 + 13); const out = []; while (out.length < 6) { const e = r.int(256); if (!out.includes(e)) out.push(e); } return out; };
      const vlFor = (j) => { const r = V.rng(5000 + j * 31); const out = []; while (out.length < 6) { const e = r.int(256); if (!out.includes(e)) out.push(e); } return out; };
      const S = { mode: 'text' };
      const MODES = ['text', 'bug', 'fix'];
      function plan(mode) {
        const seq = mode === 'text' ? SEQS.text : SEQS.img;
        let imgJ = 0;
        const items = seq.map(([w2, id]) => {
          const isImg = id === IMG;
          const ex = isImg && mode === 'fix' ? vlFor(imgJ++) : rowFor(id);
          return { w: w2, id, isImg, ex, viaTable: !(isImg && mode === 'fix') };
        });
        const ids = []; items.forEach((it) => { if (!ids.includes(it.id)) ids.push(it.id); });
        return { items, ids };
      }
      // Layout do celular calculado a partir do conteúdo (tokens que quebram linha, linhas da tabela,
      // caixa do bias_vl), para que a grade e a nota caibam inteiras no canvas.
      const mctx = document.createElement('canvas').getContext('2d');
      const TCW = 30, IDW = 62, TRH = 24;
      function mobLayout(w, mode, P) {
        const tokPos = [];
        let x = 16, y = 40;
        P.items.forEach((it) => { const tw = Math.max(34, textW(mctx, it.isImg ? '▦' : it.w.trim(), 12, 'sans', 600) + 18); if (x + tw > w - 16) { x = 16; y += 34; } tokPos.push([x, y, tw, 26]); x += tw + 6; });
        const last = tokPos[tokPos.length - 1];
        const tx0 = 16, ty0 = last[1] + last[3] + 34;
        const tabEnd = ty0 + P.ids.length * TRH;
        const blockEnd = mode === 'fix' ? tabEnd + 40 : tabEnd;
        const cs = Math.max(8, Math.min(16, (w - 32) / 16 - 1.6));
        const gy0 = blockEnd + 34;
        const gridEnd = gy0 + 16 * (cs + 1.6);
        const noteY = gridEnd + 16;
        return { tokPos, tx0, ty0, tabEnd, blockEnd, cs, gy0, gridEnd, noteY, h: noteY + 13 + 14 };
      }
      // altura estável: a maior das três entradas, para o canvas não pular ao trocar de modo
      const applyH = fitH(fig, (w) => (w < 700 ? Math.ceil(Math.max(...MODES.map((m) => mobLayout(w, m, plan(m)).h))) : 460));
      const ctl = UI.controls(fig);
      let tNow = 0, t0 = 0;
      UI.seg(ctl, { label: 'Entrada', options: [{ v: 'text', label: 'só texto' }, { v: 'bug', label: 'texto + imagem · antes do #175' }, { v: 'fix', label: 'texto + imagem · com bias_vl' }], value: 'text', onChange: (v) => { S.mode = v; t0 = tNow; upd(); } });
      const rD = UI.readout(ctl, 'experts distintos (linhas de imagem)', '—'), rM = UI.readout(ctl, 'carga máxima de um expert', '—');
      let PL = plan(S.mode);
      function upd() {
        PL = plan(S.mode);
        const cnt = new Array(256).fill(0); const imgSet = new Set();
        PL.items.forEach((it) => it.ex.forEach((e) => { cnt[e]++; if (it.isImg) imgSet.add(e); }));
        rD.set(S.mode === 'text' ? '— (sem imagem)' : String(imgSet.size));
        rM.set(String(Math.max(...cnt)) + ' linhas');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        tNow = t;
        const C = V.C();
        const mob = w < 700;
        const LM = mob ? mobLayout(w, S.mode, PL) : null;
        if (LM && LM.h > h + 2) applyH(); // fontes carregadas depois do mount podem mudar a quebra dos tokens
        const N = PL.items.length;
        const per = 0.85;
        const cyc = (N + 3) * per;
        const tt = (t - t0) % cyc;
        const cur = Math.floor(tt / per);
        const loc = (tt % per) / per;
        const cnt = new Array(256).fill(0);
        PL.items.forEach((it, j) => { if (j < cur || (j === cur && loc > 0.5)) it.ex.forEach((e) => cnt[e]++); });
        const head = mob
          ? (S.mode === 'text' ? 'camadas 0–2: T(x) = tid2eid[id], sem roteador' : S.mode === 'bug' ? 'toda imagem tem id 129264 → mesmos 6 experts' : 'imagem: pula a tabela e usa top-6(s + b_vl)')
          : (S.mode === 'text' ? 'camadas 0–2: T(x) = tid2eid[id]  (sem escolha do roteador)' : S.mode === 'bug' ? 'imagem: todo token tem id 129264 → mesma linha → mesmos 6 experts' : 'imagem: pula a tabela e usa top-6(s + b_vl)');
        T(g, head, 16, 20, { size: 12, weight: 600, color: S.mode === 'bug' ? C.c : C.ink, maxW: w - 32 });
        // tokens
        let tokPos = [];
        if (!mob) {
          const rh = Math.min(40, (h - 70) / N);
          PL.items.forEach((it, j) => { tokPos.push([16, 46 + j * rh, 118, rh - 8]); });
        } else tokPos = LM.tokPos;
        PL.items.forEach((it, j) => {
          const [x, y, tw, th] = tokPos[j];
          const on = j === cur;
          const col = it.isImg ? C.f : C.c;
          V.box(g, x, y, tw, th, { r: 8, fill: V.alpha(col, on ? 0.4 : 0.14), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2 : 1 });
          const label = it.isImg ? (mob ? '▦' : '▦ imagem') : it.w.trim();
          if (!mob) { T(g, label, x + 8, y + th / 2 - 1, { size: 12, weight: 600, color: C.ink }); T(g, 'id ' + it.id, x + 8, y + th / 2 + 12, { size: 11, font: 'mono', color: C.muted }); }
          else T(g, label, x + tw / 2, y + 17, { size: 12, weight: 600, align: 'center', color: C.ink });
        });
        // tabela
        const tcw = TCW, idw = mob ? IDW : 70;
        const tx0 = mob ? LM.tx0 : 158, ty0 = mob ? LM.ty0 : 60;
        const trh = TRH;
        const tabR = tx0 + idw + 6 * tcw + 4;
        T(g, 'tid2eid  [129.280 × 6]', tx0, ty0 - 12, { size: 11, font: 'mono', color: C.muted });
        const curIt = PL.items[Math.min(cur, N - 1)];
        const curActive = cur < N;
        PL.ids.forEach((id, r) => {
          const y = ty0 + r * trh;
          const row = rowFor(id);
          const hi = curActive && curIt.id === id && curIt.viaTable;
          const skipped = id === IMG && S.mode === 'fix';
          V.box(g, tx0, y, idw + 6 * tcw + 4, trh - 3, { r: 5, fill: hi ? V.alpha(C.c, 0.18) : V.alpha(C.muted, 0.06), stroke: hi ? C.c : null, lw: 1.5 });
          T(g, String(id), tx0 + 6, y + 15, { size: 11, font: 'mono', weight: 600, color: id === IMG ? C.f : C.ink });
          row.forEach((e, c2) => T(g, String(e), tx0 + idw + c2 * tcw + tcw / 2, y + 15, { size: 11, font: 'mono', align: 'center', color: skipped ? V.alpha(C.muted, 0.6) : C.ink }));
          if (skipped) { V.line(g, tx0 + 4, y + trh / 2 - 1, tx0 + idw + 6 * tcw, y + trh / 2 - 1, { color: C.f, lw: 1.5 }); }
        });
        const tabEnd = ty0 + PL.ids.length * trh;
        if (S.mode === 'fix') {
          const bx = tx0, by = tabEnd + 10;
          const act = curActive && curIt.isImg;
          V.box(g, bx, by, idw + 6 * tcw + 4, 30, { r: 8, fill: V.alpha(C.b, act ? 0.3 : 0.12), stroke: C.b, lw: act ? 2 : 1 });
          T(g, 'imagem: top-6(s + b_vl) · roteador', bx + (idw + 6 * tcw) / 2, by + 19, { size: 11, weight: 600, align: 'center', color: C.ink, maxW: idw + 6 * tcw - 6 });
        }
        // grade de experts
        let gx0, gy0, cs;
        if (!mob) { gx0 = tabR + 36; cs = Math.max(4, Math.min(17, (w - gx0 - 16) / 16 - 1.6)); gy0 = 60; }
        else { gx0 = 16; cs = LM.cs; gy0 = LM.gy0; }
        T(g, '256 experts (camada 0)', gx0, gy0 - 12, { size: 11, font: 'mono', color: C.muted });
        const cellXY = (e) => [gx0 + (e % 16) * (cs + 1.6), gy0 + Math.floor(e / 16) * (cs + 1.6)];
        for (let e = 0; e < 256; e++) {
          const [x, y] = cellXY(e);
          const c = cnt[e];
          g.fillStyle = c === 0 ? V.alpha(C.muted, 0.12) : c >= 3 ? C.c : V.alpha(C.a, 0.35 + 0.3 * c);
          V.rr(g, x, y, cs, cs, 2); g.fill();
          if (c >= 3 && cs >= 14) T(g, String(c), x + cs / 2, y + cs / 2 + 4, { size: 11, font: 'mono', weight: 700, align: 'center', color: onTxt(C) });
        }
        // linhas do token atual até os experts
        if (curActive) {
          const lc = V.alpha(curIt.isImg ? C.f : C.c, 0.6);
          let src;
          if (curIt.viaTable) { const r = PL.ids.indexOf(curIt.id); src = [tabR, ty0 + r * trh + trh / 2 - 1]; }
          else src = [tabR, tabEnd + 25];
          if (mob) {
            // no celular a grade fica embaixo: a linha desce pela lateral direita (fora dos números
            // da tabela e da caixa do bias_vl) e só abre em leque logo acima da grade
            const lx = Math.min(w - 8, tabR + 10), ly = gy0 - 4;
            V.line(g, src[0], src[1], lx, src[1], { color: lc, lw: 1.4 });
            V.line(g, lx, src[1], lx, ly, { color: lc, lw: 1.4 });
            src = [lx, ly];
          }
          const a = V.ease.out(V.clamp(loc / 0.5, 0, 1));
          curIt.ex.forEach((e) => {
            const [x, y] = cellXY(e);
            const ex = x + cs / 2, ey = y + cs / 2;
            const q = V.along(src[0], src[1], ex, ey, a);
            V.line(g, src[0], src[1], q[0], q[1], { color: lc, lw: 1.2 });
            V.dot(g, q[0], q[1], 2.6, curIt.isImg ? C.f : C.c);
          });
        }
        const note = S.mode === 'bug' ? 'as 5 linhas de imagem caem nos mesmos 6 experts (carga 5 cada, em rosa)' : S.mode === 'fix' ? 'cada linha de imagem escolhe pelo conteúdo (s + b_vl): a carga se espalha' : 'cada id tem sua linha fixa: a escolha é determinística e não depende do contexto';
        if (mob) wrap(g, note, 16, LM.noteY, w - 32, 13, { size: 11.5, color: S.mode === 'bug' ? C.c : C.muted, maxLines: 2 });
        else wrap(g, note, 16, h - 12, w - 32, 13, { size: 11.5, color: S.mode === 'bug' ? C.c : C.muted, maxLines: 1 });
      });
    });

    /* ====================================================================
       Figura 8 — Carga por passo
       ==================================================================== */
    safe('fig8', () => {
      const fig = $('#fig-carga');
      fitH(fig, (w) => (w < 600 ? 640 : 490));
      const M8 = {
        // ti = índice em TS do passo típico de decode de cada receita (GLM 7 rascunhos + 1, Qwen MTP=3 + 1, DeepSeek 6 + 1)
        glm: { E: 288, k: 8, L: 42, mb: 6.3, name: 'GLM', ep: false, ti: 4, ref: [8, 57.5, 'repo: ≈57'] },
        qwen: { E: 512, k: 10, L: 48, mb: 2.77, name: 'Qwen', ep: true, ti: 2, ref: [4, 32, 'repo: ~32'] },
        ds: { E: 256, k: 6, L: 43, mb: 6.7, name: 'DeepSeek', ep: false, ti: 3, ref: [7, 39, 'repo: ≈39'] },
      };
      const TS = [1, 2, 4, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168];
      const CAPS = [32, 64, 128, 256, 1024];
      const EDGES = [16, 32, 64, 128, 256, 512, 1024, 2048];
      const S = { m: 'glm', ti: 4, sk: 0, ci: 0 };
      const cache = new Map();
      function simulate(mk, T0, sk) {
        const key = mk + '|' + T0 + '|' + sk.toFixed(2);
        if (cache.has(key)) return cache.get(key);
        const M = M8[mk], E = M.E, k = M.k;
        const nL = T0 <= 64 ? 24 : T0 <= 1024 ? 12 : 6;
        const r = V.rng(1000 + T0 * 3 + Math.round(sk * 100));
        const alpha = 1.2 * sk;
        const layers = [];
        for (let l = 0; l < nL; l++) {
          const perm = Array.from({ length: E }, (_, i) => i);
          for (let i = E - 1; i > 0; i--) { const j = r.int(i + 1); const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; }
          const cum = new Float64Array(E); let acc = 0;
          for (let i = 0; i < E; i++) { acc += alpha ? Math.pow(i + 1, -alpha) : 1; cum[i] = acc; }
          const cnt = new Int32Array(E);
          const ch = new Int32Array(k);
          for (let tt = 0; tt < T0; tt++) {
            let n = 0, guard = 0;
            while (n < k && guard < 400) {
              guard++;
              let e;
              if (!alpha) e = r.int(E);
              else { const x = r() * acc; let lo = 0, hi = E - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < x) lo = m + 1; else hi = m; } e = perm[lo]; }
              let dup = false; for (let q = 0; q < n; q++) if (ch[q] === e) { dup = true; break; }
              if (!dup) ch[n++] = e;
            }
            for (let q = 0; q < n; q++) cnt[ch[q]]++;
          }
          let D = 0, mx = 0; for (let i = 0; i < E; i++) { if (cnt[i]) D++; if (cnt[i] > mx) mx = cnt[i]; }
          layers.push({ cnt, D, mx });
        }
        const res = { layers, D: layers.reduce((a, b) => a + b.D, 0) / nL, mx: Math.max(...layers.map((l) => l.mx)), nL };
        cache.set(key, res);
        if (cache.size > 80) cache.delete(cache.keys().next().value);
        return res;
      }
      const ctl = UI.controls(fig);
      let sT = null;
      // trocar de modelo volta ao passo típico de decode dele (o mesmo cenário do losango)
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM 288/8' }, { v: 'qwen', label: 'Qwen 512/10' }, { v: 'ds', label: 'DeepSeek 256/6' }], value: 'glm', onChange: (v) => { S.m = v; if (sT) sT.set(M8[v].ti); else upd(); } });
      sT = UI.slider(ctl, { label: 'tokens no passo', min: 0, max: TS.length - 1, step: 1, value: 4, fmt: (i) => TS[i].toLocaleString('pt-BR'), onInput: (v) => { S.ti = v; upd(); } });
      UI.slider(ctl, { label: 'concentração', min: 0, max: 1, step: 0.1, value: 0, fmt: (v) => (v === 0 ? 'uniforme' : nf(v, 1)), onInput: (v) => { S.sk = Math.round(v * 10) / 10; upd(); } });
      UI.slider(ctl, { label: 'cap de linhas', min: 0, max: CAPS.length - 1, step: 1, value: 0, fmt: (i) => String(CAPS[i]), onInput: (v) => { S.ci = v; upd(); } });
      const rD = UI.readout(ctl, 'experts distintos/camada', '—'), rB = UI.readout(ctl, 'bytes de experts/passo', '—'), rX = UI.readout(ctl, 'máx. linhas', '—'), rF = UI.readout(ctl, 'camadas com gordo', '—');
      let Q = null;
      function upd() {
        const M = M8[S.m], T0 = TS[S.ti], cap = CAPS[S.ci];
        Q = simulate(S.m, T0, S.sk);
        const bytesGB = ((M.ep ? Q.D / 2 : Q.D) * M.mb * M.L) / 1000;
        const fatL = Q.layers.filter((l) => l.mx > cap).length / Q.nL;
        rD.set(nf(Q.D, Q.D < 10 ? 1 : 0) + ' de ' + M.E);
        rB.set('≈ ' + nf(bytesGB, bytesGB < 10 ? 2 : 1) + (M.ep ? ' GB/GPU' : ' GB/rank'));
        rX.set(ni(Q.mx));
        rF.set(nf(fatL * 100, 0) + '%');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = M8[S.m], T0 = TS[S.ti], cap = CAPS[S.ci];
        const mob = w < 600;
        const x0 = 16, W = w - 32;
        const cols = mob ? (M.E >= 512 ? 32 : 24) : (M.E >= 512 ? 48 : 32);
        const gp = mob ? 1.4 : 2;
        const cs = Math.max(2, Math.min(15, (W - (cols - 1) * gp) / cols));
        const rows = Math.ceil(M.E / cols);
        const gy = 44;
        const L0 = Q.layers[0];
        const mxc = Math.max(1, L0.mx);
        T(g, `${M.name}: ${T0.toLocaleString('pt-BR')} tokens × top-${M.k} = ${(T0 * M.k).toLocaleString('pt-BR')} pares numa camada`, x0, 20, { size: 12.5, weight: 600, color: C.ink, maxW: W });
        for (let e = 0; e < M.E; e++) {
          const x = x0 + (e % cols) * (cs + gp), y = gy + Math.floor(e / cols) * (cs + gp);
          const c = L0.cnt[e];
          let col;
          if (!c) col = V.alpha(C.muted, 0.12);
          else if (c > cap) col = C.c;
          else col = V.cmap(0.25 + 0.75 * (Math.log(1 + c) / Math.log(1 + Math.max(cap, mxc))), 'seq');
          g.fillStyle = col; V.rr(g, x, y, cs, cs, Math.min(3, cs / 4)); g.fill();
          if (c && c <= 2 && T0 <= 64) { const tw = 0.5 + 0.5 * Math.sin(t * 3 + e); g.fillStyle = V.alpha(C.a, 0.25 * tw); V.rr(g, x, y, cs, cs, 2); g.fill(); }
        }
        const gBot = gy + rows * (cs + gp);
        const banner = T0 <= cap ? `tokens (${T0}) ≤ cap (${cap}): nenhum expert pode ser gordo → 1 lançamento, capturável em CUDA graph` : `tokens (${ni(T0)}) > cap (${cap}): ${Q.layers[0].cnt.filter((c) => c > cap).length} experts gordos nesta camada (rosa)`;
        wrap(g, banner, x0, gBot + 16, W, 14, { size: 11.5, weight: 600, color: T0 <= cap ? C.a : C.c, maxLines: 2 });
        // gráficos
        const cTop = gBot + (mob ? 50 : 40);
        let ax, ay, aw, ah, bx, by, bw, bh;
        if (!mob) { ax = x0 + 34; ay = cTop + 16; aw = W / 2 - 60; ah = h - ay - 36; bx = x0 + W / 2 + 30; by = ay; bw = W / 2 - 34; bh = ah; }
        else { const avail = h - cTop - 20; ax = x0 + 34; ay = cTop + 16; aw = W - 44; ah = avail / 2 - 44; bx = x0 + 6; by = ay + ah + 58; bw = W - 12; bh = ah; }
        // (a) experts distintos × tokens
        T(g, 'experts distintos por camada', ax - 30, ay - 12, { size: 11.5, weight: 600, color: C.ink });
        V.line(g, ax, ay + ah, ax + aw, ay + ah, { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.line(g, ax, ay, ax, ay + ah, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const lx = (T1) => ax + (Math.log(T1) / Math.log(8192)) * aw;
        const lyv = (D) => ay + ah - (D / M.E) * ah;
        g.beginPath();
        for (let q = 0; q <= 60; q++) { const T1 = Math.pow(8192, q / 60); const D = M.E * (1 - Math.pow(1 - M.k / M.E, T1)); const xx = lx(T1), yy = lyv(D); if (q) g.lineTo(xx, yy); else g.moveTo(xx, yy); }
        g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 2; g.stroke();
        V.line(g, ax, lyv(M.E), ax + aw, lyv(M.E), { color: V.alpha(C.muted, 0.4), dash: [3, 3] });
        T(g, 'E = ' + M.E, ax + 6, lyv(M.E) + 14, { size: 11, font: 'mono', color: C.muted });
        T(g, '0', ax - 6, ay + ah + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        [1, 8, 64, 1024, 8192].forEach((v) => T(g, v >= 1024 ? (v / 1024) + 'K' : String(v), lx(v), ay + ah + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        T(g, 'tokens no passo (log)', ax + aw, ay + ah + 29, { size: 11, align: 'right', color: C.muted });
        const [rt, rdv, rlab] = M.ref;
        const diamond = (x, y, r) => { g.fillStyle = C.d; g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath(); g.fill(); };
        diamond(lx(rt), lyv(rdv), 5);
        // ponto uniforme (anel) para comparar quando há concentração
        if (S.sk > 0) { const Qu = simulate(S.m, T0, 0); g.strokeStyle = C.c; g.lineWidth = 1.5; g.beginPath(); g.arc(lx(T0), lyv(Qu.D), 5.5, 0, Math.PI * 2); g.stroke(); }
        V.dot(g, lx(T0), lyv(Q.D), 5 + Math.sin(t * 4), C.c);
        // legenda no canto superior esquerdo do gráfico (a curva só chega ao topo mais à direita)
        const lgx = ax + 12, lgy = lyv(M.E) + 32;
        diamond(lgx, lgy - 4, 4.5);
        T(g, rlab, lgx + 10, lgy, { size: 11, color: C.d });
        V.dot(g, lgx, lgy + 11, 4, C.c);
        T(g, 'simulação', lgx + 10, lgy + 15, { size: 11, color: C.c });
        if (S.sk > 0) { g.strokeStyle = C.c; g.lineWidth = 1.5; g.beginPath(); g.arc(lgx, lgy + 26, 4, 0, Math.PI * 2); g.stroke(); T(g, 'uniforme', lgx + 10, lgy + 30, { size: 11, color: C.muted }); }
        const hiD = Q.D > 0.8 * M.E;
        T(g, nf(Q.D, Q.D < 10 ? 1 : 0), lx(T0) + (hiD ? -8 : 8), lyv(Q.D) + 16, { size: 11, font: 'mono', weight: 600, align: hiD ? 'right' : 'left', color: C.c });
        // (b) histograma do máximo por camada
        T(g, `máx. de linhas por camada (${Q.nL} camadas simuladas)`, bx, by - 8, { size: 11.5, weight: 600, color: C.ink, maxW: bw });
        const hist = new Array(EDGES.length + 1).fill(0);
        Q.layers.forEach((l) => { let b = EDGES.findIndex((e2) => l.mx <= e2); if (b < 0) b = EDGES.length; hist[b]++; });
        const labs = ['≤16', '32', '64', '128', '256', '512', '1K', '2K', '>2K'];
        const nb = hist.length, gb = 3, bwid = (bw - gb * (nb - 1)) / nb;
        const hmx = Math.max(1, ...hist);
        hist.forEach((c, i) => {
          const lo = i === 0 ? 0 : EDGES[i - 1];
          const fat = lo >= cap;
          const hh = (c / hmx) * (bh - 20);
          const x = bx + i * (bwid + gb);
          V.box(g, x, by + bh - hh, bwid, Math.max(1, hh), { r: 3, fill: c ? (fat ? V.alpha(C.c, 0.8) : V.alpha(C.a, 0.75)) : V.alpha(C.muted, 0.15) });
          if (c) T(g, String(c), x + bwid / 2, by + bh - hh - 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          T(g, labs[i], x + bwid / 2, by + bh + 14, { size: 11, font: 'mono', align: 'center', color: fat ? C.c : C.muted, maxW: bwid + gb });
        });
        const capIdx = EDGES.indexOf(cap);
        if (capIdx >= 0) { const xc = bx + (capIdx + 1) * (bwid + gb) - gb / 2; V.line(g, xc, by, xc, by + bh, { color: C.d, lw: 1.5, dash: [4, 3] }); T(g, 'cap', xc + 4, by + 10, { size: 11, color: C.d }); }
      });
    });

    /* ====================================================================
       Figura 9 — Despacho (stepper)
       ==================================================================== */
    safe('fig9', () => {
      const fig = $('#fig-despacho');
      fitH(fig, () => 352);
      const IDS = [[2, 0], [2, 4], [1, 2], [2, 3], [0, 2], [4, 1]];
      const WTS = [[0.62, 0.38], [0.55, 0.45], [0.58, 0.42], [0.70, 0.30], [0.52, 0.48], [0.66, 0.34]];
      const NT = 6, KK = 2, NE = 5, CAP = 3;
      const flat = [];
      IDS.forEach((r, t2) => r.forEach((e, s2) => flat.push({ t: t2, s: s2, e, w: WTS[t2][s2] })));
      const order = flat.map((f, i) => i).sort((a, b) => flat[a].e - flat[b].e || a - b);
      const rank = new Array(flat.length); order.forEach((fi, r) => { rank[fi] = r; });
      const counts = new Array(NE).fill(0); flat.forEach((f) => counts[f.e]++);
      const runStart = []; { let acc = 0; for (let e = 0; e < NE; e++) { runStart.push(acc); acc += counts[e]; } }
      const steps = [
        { t: 'Roteamento', d: 'Cada token (coluna) escolheu 2 experts; a cor e o número de cada ficha dizem qual. A ordem natural é a dos tokens, que não serve para nenhum expert.' },
        { t: 'Achatar', d: 'Os \\(T \\times k = 12\\) pares (token, slot) viram uma lista só. Cada ficha carrega junto o token de origem e o peso \\(g\\).' },
        { t: 'Ordenar por expert', d: 'Um <code>argsort</code> pelo id do expert põe os pares de cada expert lado a lado. Agora cada expert tem um lote contíguo de linhas para processar.' },
        { t: 'Contar sem sair da GPU', d: 'As linhas por expert são contadas com <code>scatter_add_</code>, que fica na GPU (o <code>bincount</code> pode passar pela CPU e quebrar a captura de CUDA graph). O expert 2 tem 5 linhas, acima do cap de 3: é gordo.' },
        { t: 'Lançar os kernels', d: 'Os magros vão juntos num único lançamento do kernel fundido (<code>exl3_moe</code>), que pula quem passou do cap. O gordo segue o tier configurado; no E3, são 3 lançamentos (gather, gate/up + SwiGLU, down + scatter) guiados por tabelas no device.' },
        { t: 'Combinar', d: 'Cada resultado é multiplicado pelo seu peso \\(g\\) e somado na saída do token de origem, <code>out[token] += g·E(x)</code>. Cada token recebe exatamente \\(k=2\\) contribuições.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6000 });
      let prevStep = 0, curStep = 0;
      st.go(0);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (st.i !== curStep) { prevStep = curStep; curStep = st.i; }
        const e = V.ease.inOut(V.clamp(st.elapsed() / 1.0, 0, 1));
        const mob = w < 600;
        const x0 = 16, W = w - 32;
        const cw = Math.min(46, (W - 11 * 4 - 4 * 10) / 12), chH = mob ? 34 : 36, gap = 4, rg = mob ? 7 : 10;
        const colW = Math.min(90, W / NT);
        const tokX = (t2) => x0 + (W - NT * colW) / 2 + t2 * colW + (colW - cw) / 2;
        const yTok = 56, ySlot = (s2) => yTok + s2 * (chH + 6);
        const yRow = 190;
        const rowW = 12 * cw + 11 * gap + 4 * rg;
        const rx0 = x0 + (W - rowW) / 2;
        const xFlat = (i) => rx0 + i * (cw + gap) + Math.floor(i / 12) * 0;
        const xSorted = (r) => { let e2 = 0; while (e2 < NE - 1 && r >= runStart[e2 + 1]) e2++; return rx0 + r * (cw + gap) + e2 * rg; };
        const yK = yRow + 12, kH = 58; // os kernels ocupam o lugar da lista ordenada (que se esvazia neste passo)
        const thinW = Math.min(260, W * 0.52), fatW = Math.min(230, W * 0.42);
        const thinX = x0 + (W - thinW - fatW - 16) / 2, fatX = thinX + thinW + 16;
        const posAt = (sI, fi) => {
          const f = flat[fi];
          if (sI === 0) return [tokX(f.t), ySlot(f.s)];
          if (sI === 1) return [xFlat(fi), yRow];
          if (sI === 2 || sI === 3) return [xSorted(rank[fi]), yRow];
          if (sI === 4) {
            if (counts[f.e] > CAP) { const j = rank[fi] - runStart[f.e]; return [fatX + 10 + j * (Math.min(cw, (fatW - 20) / 5) + 2), yK + 24]; }
            const thinOrder = order.filter((q) => counts[flat[q].e] <= CAP);
            const j = thinOrder.indexOf(fi); const cw2 = Math.min(cw, (thinW - 20) / 7 - 2);
            return [thinX + 10 + (j % 7) * (cw2 + 2), yK + 24 + Math.floor(j / 7) * (chH * 0.62)];
          }
          return [tokX(f.t) + f.s * (cw / 2 + 1), yTok + 4];
        };
        const ps = prevStep === 5 && curStep === 0 ? 0 : prevStep;
        // rótulos
        T(g, curStep === 5 ? 'saída de cada token: out[t] = Σ g·E(x)' : 'tokens (top-2 cada)', x0, 24, { size: 12.5, weight: 600, color: C.ink });
        for (let t2 = 0; t2 < NT; t2++) {
          const x = tokX(t2);
          T(g, 't' + t2, x + cw / 2, yTok - 10, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.c });
          if (curStep === 5) {
            const done = V.clamp((e - 0.55) / 0.45, 0, 1);
            V.box(g, x - 4, yTok - 2, cw + 8, 2 * chH + 10, { r: 8, fill: V.alpha(C.a, 0.08 + 0.25 * done), stroke: V.alpha(C.a, 0.3 + 0.7 * done), lw: 1.5 });
            if (done > 0) {
              g.globalAlpha = done;
              const oy = yTok + chH * 0.62 + 16;
              V.box(g, x, oy, cw, chH, { r: 7, fill: V.alpha(C.a, 0.35), stroke: C.a, lw: 1.5 });
              T(g, 'Σ', x + cw / 2, oy + chH / 2 + 5, { size: 14, weight: 700, align: 'center', color: C.ink });
              const [e0, e1] = IDS[t2], [w0, w1] = WTS[t2];
              if (!mob) { T(g, nf(w0, 2) + '·E' + e0, x + cw / 2, yTok + 2 * chH + 42, { size: 11, font: 'mono', align: 'center', color: C.muted }); T(g, '+ ' + nf(w1, 2) + '·E' + e1, x + cw / 2, yTok + 2 * chH + 56, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
              V.arrow(g, x + cw / 2, yTok + chH * 0.62 + 5, x + cw / 2, oy - 2, { color: C.a, lw: 1.4, head: 5 });
              g.globalAlpha = 1;
            }
            T(g, 'out[t' + t2 + ']', x + cw / 2, yTok + 2 * chH + 24, { size: 11, font: 'mono', align: 'center', color: C.a, alpha: done });
          } else V.box(g, x - 4, yTok - 2, cw + 8, 2 * chH + 10, { r: 8, stroke: V.alpha(C.muted, 0.3), lw: 1 });
        }
        if (curStep >= 1 && curStep <= 3) T(g, curStep === 1 ? 'lista achatada (ordem dos tokens)' : 'ordenada por expert', rx0, yRow - 12, { size: 11.5, color: C.muted });
        // estações dos experts (passos 1, 2 e 6): o destino das linhas e, no fim, quanto cada um processou
        if (curStep <= 1 || curStep === 5) {
          const nW = Math.min(76, (W - 4 * 10) / NE), stY = h - 58, gx = x0 + (W - NE * nW - 4 * 10) / 2;
          const aIn = curStep === 5 ? e : 1;
          g.globalAlpha = aIn;
          T(g, curStep === 5 ? 'linhas processadas por expert, já devolvidas aos tokens' : 'os 5 experts: cada um quer um lote contíguo das suas linhas', x0 + W / 2, stY - 10, { size: 11, align: 'center', color: C.muted, maxW: W });
          for (let e2 = 0; e2 < NE; e2++) {
            const bx = gx + e2 * (nW + 10), fat = counts[e2] > CAP, col = curStep === 5 && fat ? C.c : C.series[e2 % 6];
            if (curStep !== 5) g.setLineDash([4, 3]);
            V.box(g, bx, stY, nW, 32, { r: 8, fill: V.alpha(col, curStep === 5 ? 0.22 : 0.1), stroke: V.alpha(col, curStep === 5 ? 1 : 0.6), lw: 1.3 });
            g.setLineDash([]);
            T(g, 'E' + e2, bx + 8, stY + 20, { size: 11, font: 'mono', weight: 700, color: curStep === 5 && fat ? C.c : C.ink });
            T(g, curStep === 5 ? String(counts[e2]) : '?', bx + nW - 8, stY + 20, { size: 11, font: 'mono', weight: 600, align: 'right', color: curStep === 5 ? (fat ? C.c : C.ink) : C.muted });
          }
          g.globalAlpha = 1;
        }
        // brackets e barras
        if (curStep >= 2 && curStep <= 3) {
          for (let e2 = 0; e2 < NE; e2++) {
            if (!counts[e2]) continue;
            const xa = xSorted(runStart[e2]), xb = xSorted(runStart[e2] + counts[e2] - 1) + cw;
            const fat = counts[e2] > CAP;
            const a = curStep === 2 ? e : 1;
            g.globalAlpha = a;
            V.line(g, xa, yRow + chH + 6, xb, yRow + chH + 6, { color: fat && curStep >= 3 ? C.c : C.series[e2 % 6], lw: 2 });
            T(g, 'E' + e2, (xa + xb) / 2, yRow + chH + 20, { size: 11, font: 'mono', weight: 600, align: 'center', color: fat && curStep >= 3 ? C.c : C.ink });
            g.globalAlpha = 1;
            if (curStep === 3) {
              const bh2 = counts[e2] * 10 * e;
              const bx2 = (xa + xb) / 2 - 10;
              V.box(g, bx2, yRow + chH + 82 - bh2, 20, bh2, { r: 3, fill: fat ? C.c : V.alpha(C.a, 0.8) });
              T(g, String(counts[e2]), bx2 + 10, yRow + chH + 96, { size: 11, font: 'mono', weight: 600, align: 'center', color: fat ? C.c : C.ink });
            }
          }
          if (curStep === 3) { const yc = yRow + chH + 82 - CAP * 10; V.line(g, rx0 - 6, yc, rx0 + rowW + 6, yc, { color: C.d, lw: 1.5, dash: [4, 3] }); T(g, 'cap = 3', rx0 + rowW + 4, yc - 4, { size: 11, align: 'right', color: C.d }); T(g, 'scatter_add_ (GPU) ✓   bincount ✗', rx0, yRow + chH + 112, { size: 11, font: 'mono', color: C.muted }); }
        }
        if (curStep === 4) {
          V.box(g, thinX, yK, thinW, kH, { r: 10, fill: V.alpha(C.a, 0.08), stroke: C.a, lw: 1.5 });
          T(g, mob ? 'exl3_moe · 1 lançamento' : 'exl3_moe · magros (1 lançamento)', thinX + 8, yK + 14, { size: 11, weight: 600, color: C.a, maxW: thinW - 12 });
          V.box(g, fatX, yK, fatW, kH, { r: 10, fill: V.alpha(C.c, 0.08), stroke: C.c, lw: 1.5 });
          T(g, mob ? 'E3 · 3 lançamentos' : 'E3 · gordo: gather → gate/up → down', fatX + 8, yK + 14, { size: 11, weight: 600, color: C.c, maxW: fatW - 12 });
          wrap(g, 'no decode, com tokens ≤ cap, nenhum expert pode ser gordo: só a caixa verde existe (1 lançamento, capturável em CUDA graph)', x0 + W / 2, yK + kH + 30, W, 14, { size: 11.5, align: 'center', color: C.muted, maxLines: 3 });
        }
        // fichas
        flat.forEach((f, fi) => {
          const a = posAt(ps, fi), b = posAt(curStep, fi);
          const x = V.lerp(a[0], b[0], e), y = V.lerp(a[1], b[1], e);
          const fat = counts[f.e] > CAP;
          const col = curStep >= 3 && fat ? C.c : C.series[f.e % 6];
          let ww = cw, hh = chH;
          if (curStep === 4) { ww = fat ? Math.min(cw, (fatW - 20) / 5) : Math.min(cw, (thinW - 20) / 7 - 2); hh = chH * 0.55; }
          if (curStep === 5) { ww = V.lerp(cw, cw / 2 - 1, e); hh = V.lerp(chH, chH * 0.62, e); }
          V.box(g, x, y, ww, hh, { r: 6, fill: V.alpha(col, 0.28), stroke: col, lw: 1.4 });
          if (hh >= 26) {
            T(g, 'E' + f.e, x + ww / 2, y + 14, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.ink });
            T(g, 't' + f.t, x + ww / 2, y + 28, { size: 11, font: 'mono', align: 'center', color: C.muted });
          } else if (ww >= 16) T(g, String(f.e), x + ww / 2, y + hh / 2 + 4, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.ink });
        });
        if (curStep === 5) T(g, 'cada token recebe exatamente k = 2 contribuições ponderadas', x0, yTok + 2 * chH + (mob ? 50 : 78), { size: 11.5, color: C.muted, maxW: W });
      });
    });

    /* ====================================================================
       Figura 9b — Lane TensorFold: itens leves × pesados e a ordem de lançamento
       Brinquedo: cada SM roda um item por vez; a DRAM é dividida entre os itens
       que ainda leem pesos; o mma de um item só avança até onde os pesos chegaram.
       ==================================================================== */
    safe('fig9b', () => {
      const fig = $('#fig-leves-pesados');
      fitH(fig, (w) => (w < 600 ? 428 : 410));
      const NE = 30, NSM = 6, BW = 3.5, CAPBW = 1.2, DT = 0.02, SM = 10;
      const COST = { L: { d: 1, c: 0.2 }, H: { d: 1, c: 1 } };
      // d: pesos lidos da DRAM (cada expert lê os seus uma vez: o pesado divide essa leitura entre os n itens, o resto vem da L2); c: trabalho de mma
      const S = { heavy: 3, seed: 4, runs: null, t0: null };
      function plan() {
        const r = V.rng(97 + S.seed * 31 + S.heavy * 7);
        const hv = new Set();
        while (hv.size < S.heavy) hv.add(r.int(NE));
        const items = [];
        for (let e = 0; e < NE; e++) {
          if (hv.has(e)) { const n = 4 + r.int(4); for (let i = 0; i < n; i++) items.push({ e, h: true, n }); } else items.push({ e, h: false, n: 1 });
        }
        return items;
      }
      // a regra do patch: pesado de posição r em (r + 1/2)/nh, leve de posição l em (l + 1/2)/nl
      function interleave(items) {
        const Hh = items.filter((it) => it.h), Ll = items.filter((it) => !it.h);
        const k = Hh.map((it, r) => ({ it, k: (r + 0.5) / Hh.length, o: 0 })).concat(Ll.map((it, l) => ({ it, k: (l + 0.5) / Ll.length, o: 1 })));
        k.sort((a, b) => a.k - b.k || a.o - b.o);
        return k.map((x) => x.it);
      }
      function smooth(a) { return a.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - SM); j <= Math.min(a.length - 1, i + SM); j++) { s += a[j]; n++; } return s / n; }); }
      function run(queue) {
        const sl = new Array(NSM).fill(null), bars = [], ud = [], ut = [], start = new Array(queue.length).fill(Infinity);
        let q = 0, t = 0, done = 0;
        while (done < queue.length && t < 200) {
          for (let s = 0; s < NSM; s++) if (!sl[s] && q < queue.length) { const it = queue[q]; const c = it.h ? COST.H : COST.L; const d = c.d / it.n; sl[s] = { it, i: q, d, d0: d, c: c.c, c0: c.c, t0: t }; start[q] = t; q++; }
          let nd = 0;
          for (const x of sl) if (x && x.d > 0) nd++;
          const share = nd ? Math.min(CAPBW, BW / nd) : 0;
          let dr = 0, tc = 0;
          for (let s = 0; s < NSM; s++) {
            const x = sl[s];
            if (!x) continue;
            if (x.d > 0) { x.d -= share * DT; dr += share; }
            if (x.c > 0) { const got = x.d <= 0 ? 1 : 1 - x.d / x.d0; if ((x.c0 - x.c) / x.c0 <= got) { x.c -= DT; tc++; } }
            if (x.d <= 0 && x.c <= 0) { bars.push({ s, a: x.t0, b: t + DT, h: x.it.h }); sl[s] = null; done++; }
          }
          ud.push(dr / BW); ut.push(tc / NSM); t += DT;
        }
        const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
        return { T: t, bars, ud: smooth(ud), ut: smooth(ut), mud: mean(ud), mut: mean(ut), queue, start };
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'experts pesados', min: 2, max: 8, step: 1, value: S.heavy, fmt: (v) => String(v), onInput: (v) => { S.heavy = v; rebuild(); } });
      UI.button(ctl, '↻ outro chunk', () => { S.seed++; rebuild(); });
      const ro = UI.readout(ctl, 'intercalada × plano', '—');
      UI.legend(ctl, [[V.C().a, 'leve: 1 item, pesos da DRAM'], [V.C().c, 'pesado: vários itens, pesos via L2']]);
      function rebuild() {
        const p = plan();
        S.runs = [run(p), run(interleave(p))];
        S.t0 = null;
        const gain = 1 - S.runs[1].T / S.runs[0].T;
        ro.set((gain >= 0 ? '−' : '+') + nf(Math.abs(gain) * 100, 0) + '% de tempo', gain > 0.005 ? 'ok' : 'warn');
      }
      rebuild();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.t0 === null) S.t0 = t - 7; // abre no quadro completo; depois repete a varredura
        const [P, Q] = S.runs;
        const Tm = Math.max(P.T, Q.T);
        const el = (t - S.t0) % 10;
        const tau = Math.min(1, el / 7) * Tm;
        const mob = w < 600;
        const x0 = 16, W = w - 32, lw = mob ? 52 : 74, rw = 40, gx = x0 + lw, gw = W - lw - rw;
        const sx = (tt) => gx + gw * tt / Tm;
        const panH = (h - 34) / 2;
        const lh = mob ? 11 : 10, lg = 3, uh = 18;
        [P, Q].forEach((R, pi) => {
          const y0 = 4 + pi * (panH + 8);
          T(g, pi ? (mob ? 'intercalada (ORDER=1)' : 'intercalada (TF_GLM_EXL3_ORDER=1)') : (mob ? 'ordem do plano' : 'ordem do plano (pelos ids dos experts)'), x0, y0 + 13, { size: 12.5, weight: 600, color: C.ink });
          const fin = tau >= R.T;
          T(g, 'tempo ' + (fin ? nf(R.T, 1) : nf(tau, 1)), x0 + W, y0 + 13, { size: 11.5, font: 'mono', align: 'right', color: fin ? (pi ? C.e : C.ink) : C.muted });
          // a fila de itens, na ordem de lançamento
          const qy = y0 + 22, qh = 12, n = R.queue.length, cw = gw / n;
          T(g, 'fila', x0, qy + 10, { size: 11, color: C.muted });
          for (let i = 0; i < n; i++) {
            const col = R.queue[i].h ? C.c : C.a;
            V.rr(g, gx + i * cw + 0.5, qy, Math.max(1, cw - 1), qh, 2);
            g.fillStyle = V.alpha(col, R.start[i] <= tau ? 0.9 : 0.25); g.fill();
          }
          // Gantt: qual item cada SM roda
          const gy = qy + qh + 8;
          T(g, NSM + ' SMs', x0, gy + (NSM * (lh + lg)) / 2 + 4, { size: 11, color: C.muted });
          for (let s = 0; s < NSM; s++) { V.rr(g, gx, gy + s * (lh + lg), gw, lh, 2); g.fillStyle = V.alpha(C.muted, 0.08); g.fill(); }
          if (!pi) T(g, 'média', x0 + W, gy + NSM * (lh + lg) - 2, { size: 11, align: 'right', color: C.muted });
          for (const b of R.bars) {
            if (b.a >= tau) continue;
            const xa = sx(b.a), xb = sx(Math.min(b.b, tau));
            V.rr(g, xa + 0.5, gy + b.s * (lh + lg), Math.max(1, xb - xa - 1), lh, 2);
            g.fillStyle = V.alpha(b.h ? C.c : C.a, 0.8); g.fill();
          }
          // ocupação da DRAM e dos tensor cores
          const uy = gy + NSM * (lh + lg) + 6;
          [[R.ud, R.mud, C.a, 'DRAM'], [R.ut, R.mut, C.c, 'tensor']].forEach(([arr, avg, col, lab], k) => {
            const yb = uy + k * (uh + 5);
            T(g, lab, x0, yb + uh - 4, { size: 11, color: col, weight: 600 });
            if (fin) T(g, nf(avg * 100, 0) + '%', x0 + W, yb + uh - 4, { size: 11, font: 'mono', align: 'right', color: col });
            V.rr(g, gx, yb, gw, uh, 2); g.fillStyle = V.alpha(C.muted, 0.08); g.fill();
            const m = Math.min(arr.length, Math.floor(tau / DT));
            if (m < 2) return;
            const step = Math.max(1, Math.ceil(arr.length / (gw / 2)));
            g.beginPath(); g.moveTo(sx(0), yb + uh);
            for (let i = 0; i < m; i += step) g.lineTo(sx(i * DT), yb + uh - V.clamp(arr[i], 0, 1) * uh);
            g.lineTo(sx((m - 1) * DT), yb + uh - V.clamp(arr[m - 1], 0, 1) * uh);
            g.lineTo(sx((m - 1) * DT), yb + uh); g.closePath();
            g.fillStyle = V.alpha(col, 0.45); g.fill();
          });
          if (fin) {
            const xe = sx(R.T);
            V.line(g, xe, qy - 2, xe, uy + 2 * uh + 7, { color: pi ? C.e : C.ink, lw: 1.5, dash: [4, 3] });
          }
        });
        if (tau < Tm) V.line(g, sx(tau), 26, sx(tau), h - 30, { color: V.alpha(C.ink, 0.35), lw: 1 });
        T(g, mob ? 'tempo (unidades ilustrativas) →' : 'tempo da camada (unidades ilustrativas) →', x0 + W, h - 10, { size: 11, align: 'right', color: C.muted });
      });
    });

    /* ====================================================================
       Figura 10 — Calculadora
       ==================================================================== */
    safe('fig10', () => {
      const fig = $('#fig-calc');
      fitH(fig, (w) => (w < 600 ? 640 : 500));
      const PRE = {
        // embA = parâmetros de embedding contidos em oA. O lookup lê só T linhas da tabela, então eles saem da conta de
        // bytes por passo. GLM e Qwen contam o embedding nos ativos; no DeepSeek (13,2 B) e no MiMo (card) ele já está fora de oA.
        glm: { name: 'GLM-5.3-Flash', E: 288, k: 8, S: 1, I: 2048, d: 4096, L: 42, oT: 7.863, oA: 7.863, embA: 0.634, bexp: 4, both: 16, T: 8, note: 'Tronco do GLM. Checkpoint EXL3 4 bpw nos roteados e BF16 no resto; a receita converte os densos para FP8 na carga (GLM53_DENSE_FP8=all), então 16 bits no resto é um teto. Fora: MTP (7,43 B) e visão (0,56 B). Resto = KDA, MLA + indexador, 3 FFNs densas, embedding, lm_head, roteadores e mHC.' },
        qwen: { name: 'Qwen3.8-Flash-Next', E: 512, k: 10, S: 1, I: 640, d: 2560, L: 48, oT: 4.714, oA: 4.714, embA: 0.636, bexp: 4.5, both: 16, T: 4, note: 'Tronco do Qwen (NVFP4 = 4,5 bits efetivos nos experts, resto BF16). Fora: tabela N-gram (51,2 B), MTP (2,61 B) e visão (0,45 B). O card conta ~6 B ativos sem embedding e lm_head; aqui eles entram. Com MTP=3 o lm_head é lido 4× por passo, o que a conta de bytes não inclui.' },
        ds: { name: 'DeepSeek-V4-Flash', E: 256, k: 6, S: 1, I: 2048, d: 4096, L: 43, oT: 6.19, oA: 5.66, embA: 0, bexp: 4.25, both: 8, T: 7, note: 'Tronco do DeepSeek (MXFP4 = 4,25 bits nos experts, resto ≈ FP8). As 3 camadas hash contam como MoE. Fora: rascunhador DSpark (≈ 19,8 B) e visão (≈ 0,47 B).' },
        mimo: { name: 'MiMo-V2.6-Flash', E: 256, k: 8, S: 0, I: 2048, d: 4096, L: 47, oT: 6.2, oA: 5.54, embA: 0, bexp: 4.25, both: 8, T: 8, note: 'Sem expert compartilhado: todo o custo do MoE é roteado. Resto estimado a partir do card (309 B / 15 B) e tratado como FP8 (aproximação).' },
      };
      const S = Object.assign({ p: 'glm', mod: false }, PRE.glm);
      let applying = false;
      const ctl = UI.controls(fig);
      const sl = {};
      const touch = () => { if (!applying) S.mod = true; };
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'mimo', label: 'MiMo' }], value: 'glm', onChange: (v) => applyPreset(v) });
      sl.E = UI.slider(ctl, { label: 'E', min: 8, max: 1024, step: 8, value: S.E, onInput: (v) => { S.E = v; if (S.k > v) S.k = v; touch(); upd(); } });
      sl.k = UI.slider(ctl, { label: 'k', min: 1, max: 16, step: 1, value: S.k, onInput: (v) => { S.k = Math.min(v, S.E); touch(); upd(); } });
      sl.S = UI.seg(ctl, { label: 'shared', options: [0, 1, 2], value: S.S, onChange: (v) => { S.S = v; touch(); upd(); } });
      sl.I = UI.slider(ctl, { label: 'I', min: 128, max: 4096, step: 64, value: S.I, onInput: (v) => { S.I = v; touch(); upd(); } });
      sl.L = UI.slider(ctl, { label: 'camadas MoE', min: 1, max: 96, step: 1, value: S.L, onInput: (v) => { S.L = v; touch(); upd(); } });
      sl.b = UI.seg(ctl, { label: 'bits experts', options: [{ v: 4, label: '4' }, { v: 4.25, label: '4,25' }, { v: 4.5, label: '4,5' }, { v: 8, label: '8' }, { v: 16, label: '16' }], value: S.bexp, onChange: (v) => { S.bexp = v; touch(); upd(); } });
      sl.T = UI.slider(ctl, { label: 'tokens/passo', min: 1, max: 64, step: 1, value: S.T, onInput: (v) => { S.T = v; touch(); upd(); } });
      const rT = UI.readout(ctl, 'totais', '—'), rA = UI.readout(ctl, 'ativos/token', '—'), rM = UI.readout(ctl, 'memória', '—'), rBy = UI.readout(ctl, 'bytes/passo', '—');
      function applyPreset(p) {
        applying = true;
        Object.assign(S, PRE[p], { p, mod: false });
        sl.E.set(S.E); sl.k.set(S.k); sl.S.set(S.S); sl.I.set(S.I); sl.L.set(S.L); sl.b.set(S.bexp); sl.T.set(S.T);
        Object.assign(S, PRE[p], { p, mod: false });
        applying = false;
        upd();
      }
      let Q = {};
      function upd() {
        const Pe = (3 * S.d * S.I) / 1e9;
        const rT2 = S.L * S.E * Pe, sT = S.L * S.S * Pe, tot = rT2 + sT + S.oT;
        const rA2 = S.L * Math.min(S.k, S.E) * Pe, act = rA2 + sT + S.oA;
        const mem = rT2 * S.bexp / 8 + (sT + S.oT) * S.both / 8;
        const Dd = S.E * (1 - Math.pow(1 - Math.min(S.k, S.E) / S.E, S.T));
        const oRd = S.oA - S.embA; // resto lido por passo, sem a tabela de embedding
        const byStep = S.L * Dd * Pe * S.bexp / 8 + (sT + oRd) * S.both / 8;
        const by1 = S.L * Math.min(S.k, S.E) * Pe * S.bexp / 8 + (sT + oRd) * S.both / 8;
        Q = { Pe, rT: rT2, sT, tot, rA: rA2, act, mem, D: Dd, byStep, by1, oRd, expMB: Pe * 1e9 * S.bexp / 8 / 1e6 };
        rT.set(nf(tot, 1) + ' B'); rA.set(nf(act, 1) + ' B'); rM.set(nf(mem, 0) + ' GB'); rBy.set(nf(byStep, 1) + ' GB');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 600;
        const x0 = 16, W = w - 32;
        const title = (S.mod ? PRE[S.p].name + ' (modificado)' : PRE[S.p].name) + ` · d = ${S.d} · Pₑ = 3·d·I = ${nf(Q.Pe * 1000, 1)} M · expert = ${nf(Q.expMB, 1)} MB em ${nf(S.bexp, S.bexp % 1 ? 2 : 0)} bits`;
        wrap(g, title, x0, 20, W, 15, { size: 12.5, weight: 600, color: C.ink, maxLines: 2 });
        const y1 = mob ? 62 : 52;
        const sc = W / Q.tot;
        const seg = (y, parts, hgt) => { let x = x0; parts.forEach(([v, col]) => { const ww = v * sc; if (ww > 0.5) { V.box(g, x, y, Math.max(1, ww - 1), hgt, { r: 4, fill: col }); } x += ww; }); return x; };
        T(g, 'totais: ' + nf(Q.tot, 1) + ' B  (experts roteados ' + nf((Q.rT / Q.tot) * 100, 1) + '%)', x0, y1 - 6, { size: 11.5, color: C.muted });
        seg(y1, [[Q.rT, V.alpha(C.a, 0.8)], [Q.sT, V.alpha(C.d, 0.85)], [S.oT, V.alpha(C.b, 0.8)]], 22);
        const y2 = y1 + 50;
        T(g, 'ativos por token (mesma escala): ' + nf(Q.act, 1) + ' B = ' + nf((Q.act / Q.tot) * 100, 1) + '% do total', x0, y2 - 6, { size: 11.5, color: C.muted });
        const xe = seg(y2, [[Q.rA, V.alpha(C.c, 0.85)], [Q.sT, V.alpha(C.d, 0.85)], [S.oA, V.alpha(C.b, 0.8)]], 16);
        const y3 = y2 + 52;
        g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(x0, y2 + 17); g.lineTo(x0, y3 - 2); g.moveTo(xe, y2 + 17); g.lineTo(x0 + W, y3 - 2); g.stroke(); g.setLineDash([]);
        const sc2 = W / Q.act;
        let x = x0;
        [[Q.rA, C.c, 'roteados (k)'], [Q.sT, C.d, 'shared'], [S.oA, C.b, 'resto']].forEach(([v, col, lab]) => {
          const ww = v * sc2;
          if (ww > 0.5) V.box(g, x, y3, Math.max(1, ww - 1), 24, { r: 4, fill: V.alpha(col, 0.85) });
          if (ww > 64) T(g, lab + ' ' + nf(v, v < 1 ? 2 : 1), x + 6, y3 + 16, { size: 11, weight: 600, color: onTxt(C), maxW: ww - 8 });
          x += ww;
        });
        if (!mob) T(g, 'ampliado', x0 + W, y3 + 40, { size: 11, align: 'right', color: C.muted });
        (mob ? [[C.a, 'todos'], [C.c, 'k ativos'], [C.d, 'shared'], [C.b, 'resto']] : [[C.a, 'roteados (todos)'], [C.c, 'roteados (k)'], [C.d, 'shared'], [C.b, 'resto']]).forEach(([col, lab], i) => { const lx0 = x0 + i * (mob ? Math.min(82, W / 4) : 118); V.box(g, lx0, y3 + 31, 10, 10, { r: 2, fill: col }); T(g, lab, lx0 + 14, y3 + 40, { size: 11, color: C.muted, maxW: mob ? Math.min(66, W / 4 - 16) : 100 }); });
        // cartões de números
        const cy = y3 + 54;
        const cards = [
          ['memória', nf(Q.mem, 0) + ' GB', `experts ${nf(S.bexp, S.bexp % 1 ? 2 : 0)} b · resto ${S.both} b`],
          ['bytes/token (T=1)', nf(Q.by1, 1) + ' GB', 'k experts por camada'],
          [`bytes/passo (T=${S.T})`, nf(Q.byStep, 1) + ' GB', `≈${nf(Q.D, Q.D < 10 ? 1 : 0)} experts distintos/camada`],
          ['total ÷ ativos', nf(Q.tot / Q.act, 1) + '×', 'capacidade por custo'],
        ];
        const perRow = mob ? 2 : 4;
        const cwid = (W - (perRow - 1) * 10) / perRow, chh = 62;
        cards.forEach((c, i) => {
          const cx = x0 + (i % perRow) * (cwid + 10), cyy = cy + Math.floor(i / perRow) * (chh + 10);
          V.box(g, cx, cyy, cwid, chh, { r: 10, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          T(g, c[0], cx + 10, cyy + 17, { size: 11, color: C.muted, maxW: cwid - 16 });
          T(g, c[1], cx + 10, cyy + 38, { size: 17, font: 'display', weight: 700, color: C.ink, maxW: cwid - 16 });
          T(g, c[2], cx + 10, cyy + 54, { size: 11, color: C.muted, maxW: cwid - 16 });
        });
        const ny = cy + Math.ceil(cards.length / perRow) * (chh + 10) + 12;
        const nl = wrap(g, PRE[S.p].note + (S.mod ? ' Valores alterados por você: o “resto” segue o preset.' : ''), x0, ny, W, 14, { size: 11.5, color: C.muted, maxLines: mob ? 7 : 3 });
        // mini-gráfico: bytes por passo × tokens no passo
        const gy = ny + nl * 14 + 26, gh = h - gy - 26;
        if (gh > 50) {
          const gx = x0 + 40, gw = W - 50;
          const bT = (T1) => S.L * S.E * (1 - Math.pow(1 - Math.min(S.k, S.E) / S.E, T1)) * Q.Pe * S.bexp / 8 + (Q.sT + Q.oRd) * S.both / 8;
          const rest = (Q.sT + Q.oRd) * S.both / 8;
          const ymx = Math.max(bT(64), 1) * 1.08;
          const PX = (T1) => gx + ((T1 - 1) / 63) * gw, PY = (v) => gy + gh - (v / ymx) * gh;
          T(g, 'bytes lidos por passo × tokens no passo (1 a 64)', x0, gy - 8, { size: 11.5, weight: 600, color: C.ink });
          V.line(g, gx, gy + gh, gx + gw, gy + gh, { color: V.alpha(C.muted, 0.5), lw: 1 });
          g.fillStyle = V.alpha(C.b, 0.18); g.fillRect(gx, PY(rest), gw, gy + gh - PY(rest));
          T(g, 'resto: lido 1× por passo', gx + gw - 4, PY(rest) + 13, { size: 11, align: 'right', color: C.b });
          g.beginPath();
          for (let T1 = 1; T1 <= 64; T1++) { const xx = PX(T1), yy = PY(bT(T1)); if (T1 > 1) g.lineTo(xx, yy); else g.moveTo(xx, yy); }
          g.strokeStyle = C.c; g.lineWidth = 2; g.stroke();
          V.dot(g, PX(S.T), PY(bT(S.T)), 5, C.c);
          // rótulo do ponto: acima e à esquerda (a curva sobe para a direita, então esse canto fica livre);
          // perto do eixo, abaixo e à direita. Fundo translúcido garante a leitura.
          { const px = PX(S.T), py = PY(bT(S.T)), lab = nf(bT(S.T), 1) + ' GB';
            const tw = textW(g, lab, 11, 'mono', 600), left = px - gx > tw + 18;
            const lx = left ? px - 8 - tw : px + 8, ly = left ? py - 9 : py + 17;
            V.box(g, lx - 3, ly - 11, tw + 6, 15, { r: 4, fill: V.alpha(C.bg, 0.85) });
            T(g, lab, lx, ly, { size: 11, font: 'mono', weight: 600, color: C.c }); }
          T(g, nf(ymx, 0), gx - 6, gy + 8, { size: 11, font: 'mono', align: 'right', color: C.muted });
          T(g, '0', gx - 6, gy + gh + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
          T(g, '1', gx, gy + gh + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
          T(g, '64', gx + gw, gy + gh + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
      });
    });
  },
});

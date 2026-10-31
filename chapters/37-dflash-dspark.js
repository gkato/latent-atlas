LA.chapter({
  id: 'dflash-dspark',
  part: 7,
  order: 37,
  title: 'DFlash2 e DSpark: rascunhos em bloco',
  navTitle: 'DFlash2 & DSpark',
  subtitle: 'Rascunhadores que preenchem o bloco inteiro num forward: difusão em bloco, máscara bidirecional, conv dinâmica e seletor (GLM), cabeça de Markov e de confiança (DeepSeek), k adaptativo, o estado do rascunhador no vLLM e o mesmo DFlash2 no TensorFold (verificação por igualdade, regras de parada ajustadas offline, um anel por conversa).',
  minutes: 68,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['speculative decoding', 'DFlash2', 'DSpark', 'difusão em bloco', 'adaptive-k', 'Markov head', 'rejection sampling', 'TensorFold'],
  goals: [
    'Explicar por que um rascunhador <b>em bloco</b> custa quase o mesmo para 1 ou 7 posições, e o que ele perde em coerência',
    'Descrever o DFlash2 (features do alvo, KV injetado, conv dinâmica de 2 taps, seletor top-16 com Gumbel) e o DSpark (âncora + ruído, cabeça de Markov de posto 256, cabeça de confiança)',
    'Decidir <b>quanto verificar</b>: k pelo bloco treinado, k adaptativo por EMA e agendamento por sobrevivência',
    'Configurar o estado do rascunhador no vLLM (grupo de KV SWA, páginas compactas, anel com slots por requisição) e montar o <code>--speculative-config</code> das receitas',
    'Comparar com a lane TensorFold do GLM: verificação por igualdade, regras de parada <code>fc</code>/<code>fnc</code>/<code>fcost</code> ajustadas por replay offline e um anel de contexto por conversa',
  ],
  prereqs: ['mtp-eagle-vocab'],
  summary: [
    'Rascunho em bloco: <b>[âncora, MASK/ruído…]</b> num único forward com atenção <b>bidirecional</b> dentro do bloco; <code>TRITON_ATTN</code> causal derrubou o GLM para 0,31 de aceitação (contra 0,959 com <code>FLASH_ATTN</code>).',
    'DFlash2 (GLM): 1,17 B, 5 camadas SWA-2048, features de 5 camadas do alvo (<code>fc</code> 20480→4096) injetadas como K/V; conv dinâmica de 2 taps e seletor top-16 (posto 256, caminhada Gumbel) contra o decaimento de sufixo; k=7 (bloco 8).',
    'DSpark (DeepSeek): 3 estágios V4 com MoE sobre features das camadas 40–42, bloco γ=5; a cabeça de Markov \\(W_2W_1[y_i]\\) resolve a colisão multimodal; a cabeça de confiança prevê \\(1-\\mathrm{TV}(p,q)\\).',
    'Tokens por passo \\(=1+\\sum a_i\\): verificar posições com aceitação baixa custa linhas. Adaptive-k (EMA α=0,25, margem 1, set {2,4,7}, mínimo do lote por causa dos graphs FULL) deu +13–21% em prosa.',
    'Estado do rascunhador: no GLM é um grupo SWA com <i>padded slot-share</i> e bloco compacto 896 (ids 65→6); no DeepSeek lane A é um grupo <code>SlidingWindowMLASpec</code> do próprio vLLM (blocos 4/8, preenchido por <code>precompute_and_store_context_kv</code>); no DeepSeek Stage-C é um anel <code>pos % 128</code> fora do vLLM, que exige slots por requisição (Patch 1) e lotes ragged via <code>query_start_loc</code> (Patch 2/2b).',
    'DeepSeek k=6: a lane Anemll aplica a regra de divisibilidade do MTP (<code>n_predict=3</code>); o hotfix block-k libera k=5 (o bloco treinado) com a mesma aceitação e passo mais barato.',
    'Lane TensorFold do GLM: o mesmo DFlash2 verificado por igualdade (resposta = serial), com a caminhada usando o ruído com chave do alvo; copy drafts antes do DFlash2; regra <code>fnc7:0.3</code> (confiança \\(\\mathrm{softmax}((s+g)/\\beta)\\), β = 2,0 ajustado por log loss em 9 registros offline), +7% medido sobre <code>fc5:0.3</code>; rascunhador em 4 bits (0,63 GiB/rank) e contexto num anel de 2.176 linhas por conversa (21 MiB/rank).',
  ],
  quiz: [
    { q: 'Fixar <code>TRITON_ATTN</code> no drafter DFlash2 do GLM derrubou a aceitação para ~0,31. Por quê?', o: ['O Triton não suporta BF16 no SM121', 'Nesta imagem ele aplica máscara causal dentro do bloco, e o rascunhador foi treinado com atenção bidirecional (is_causal: false)', 'O Triton ignora a janela deslizante de 2048', 'O Triton recalcula o KV de contexto e perde as features do alvo'], a: 1, e: 'O bloco [âncora, MASK×7] é previsto em paralelo com visibilidade total entre as posições. A máscara causal remove o que as posições tardias aprenderam a usar: a posição 0 continua boa e as seguintes colapsam. FLASH_ATTN faz SWA não causal.' },
    { q: 'No DSpark, que problema a cabeça de Markov \\(W_2W_1[y_i]\\) resolve?', o: ['Reduz o custo do lm_head do rascunho', 'Estima a probabilidade de aceitação de cada posição', 'Substitui a amostragem por rejeição', 'A colisão multimodal: posições paralelas seguem marginais e podem misturar duas continuações; o viés condiciona cada posição no token realmente escolhido antes'], a: 3, e: 'O backbone paralelo produz logits marginais. Somar um bigrama de posto 256 do token anterior, da esquerda para a direita, devolve a dependência entre posições a custo de uma GEMV. Estimar a aceitação é papel da cabeça de confiança.' },
    { q: 'Por que a receita DeepSeek usa <code>MTP_NUM_TOKENS=6</code> no Vision-Exp, se o bloco treinado tem 5?', o: ['Porque a lane Anemll trata o DSpark como MTP com n_predict = num_nextn_predict_layers = 3 e exige k ≥ 5 e k % 3 = 0', 'Porque 6 dá mais aceitação que 5', 'Porque o CUDA graph só captura múltiplos de 6', 'Porque o card oficial recomenda 6'], a: 0, e: 'A regra de divisibilidade existe para o MTP que reexecuta a mesma camada; no DSpark os estágios são empilhados. O hotfix block-k isenta method=dspark e permite k=5, com a mesma aceitação por posição e passo mais barato (56,6 contra 55,3 tok/s a 8K).' },
    { q: 'O adaptive-k do GLM escolhe o <b>mínimo</b> de n entre as requisições do lote. Qual o motivo?', o: ['O mínimo maximiza a aceitação', 'Para economizar memória de KV', 'Os backends FLASHINFER_MLA_SPARSE_SM120 e KDA só suportam CUDA graphs FULL com lote uniforme, então todas as requisições precisam verificar o mesmo comprimento', 'Porque a EMA só existe por lote'], a: 2, e: 'Cada comprimento n exige um graph de n+1 linhas por requisição (3, 5, 8) e o lote precisa ser uniforme. Requisições com saída estruturada ou ainda sem observações fixam o lote em 7.' },
    { q: 'Com <code>max_num_seqs &gt; 1</code>, a aceitação do DSpark (lane Stage-C) ia a ~0 sem crash. O que o Patch 1 do “Keys concurrency patch” corrigiu?', o: ['Um erro de quantização do KV do drafter', 'O anel de contexto do rascunhador era indexado pela linha do lote; a condensação do batching contínuo move requisições de linha, e cada uma passava a ler o histórico de outra. O patch mapeia req_id → slot estável', 'O HTTP 500 em lotes mistos de prefill e decode', 'A falta de CUDA graph para 6 sequências'], a: 1, e: 'Patch 1: _req_id_to_slot + _free_slots, com gather/scatter por slot_index. O HTTP 500 “got 41 rows for batch_size=2” é o Patch 2/2b (lote ragged via query_start_loc).' },
    { q: 'O que <code>GLM53_DRAFT_KV_COMPACT=1</code> muda no TP2?', o: ['Quantiza o KV do drafter para FP8', 'Reduz a janela do drafter de 2048 para 896', 'Usa como bloco do drafter o maior múltiplo de 64 que divide o bloco MLA (3584) e cabe na página de 2.351.104 B, ou seja, 896: ids por requisição 65 → 6, sem reduzir VRAM', 'Desliga o prefix caching do drafter'], a: 2, e: '896 × 2048 B = 1.835.008 ≤ 2.351.104; o próximo candidato, 1792, não cabe. O ganho é em ids de bloco (180 → 121 no pior caso; pool logado 1,85× a 14 GiB), não em bytes.' },
    { q: 'Na lane TensorFold, por que dá para comparar regras de parada do DFlash2 offline, na CPU, a partir de um único registro (dump) de cada requisição?', o: ['Porque o DFlash2 é determinístico em greedy e todos os registros são greedy', 'Porque a amostragem por rejeição guarda o \\(q(x)\\) de cada rascunho no registro', 'Porque a amostragem tem chave por (semente, posição, token): os tokens cometidos não dependem da regra, então uma rodada que começa em i mantém exatamente os rascunhos iguais aos tokens i+1, i+2…, e uma tabela de custos converte as rodadas em tok/s', 'Porque o registro grava os CUDA graphs de cada largura de janela'], a: 2, e: 'Com verificação por igualdade, a resposta é a serial qualquer que seja a regra; só o número de rodadas e o custo de cada uma mudam. Por isso o simulador reproduz qualquer regra (e o oráculo, por programação dinâmica) a partir do mesmo registro. Os registros incluem prosa amostrada, não só greedy, e não há \\(q(x)\\): não existe amostragem por rejeição nessa lane.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('.', ',');
    const fmtInt = (x) => Math.round(x).toLocaleString('pt-BR');
    function measure(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem-token: quadrado arredondado com olhinhos */
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: o.stroke || col, lw: 1.2 });
      if (s >= 11) {
        const eye = C.dark ? '#06140d' : '#ffffff';
        const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
        V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function check(g, x, y, ok, s = 7) {
      const C = V.C();
      g.lineWidth = 2; g.strokeStyle = ok ? C.a : RED(); g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke();
    }

    /* perfis de aceitação por posição (incondicionais) medidos nas receitas */
    const PROF = {
      glmS: { label: 'GLM · estruturado', a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83] },
      glmP: { label: 'GLM · prosa', a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06] },
      ds: { label: 'DeepSeek · Vision-Exp', a: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15] },
    };
    const acceptedFromU = (a, k, u) => { let j = 0; for (let i = 0; i < Math.min(k, a.length); i++) if (a[i] > u) j++; else break; return j; };

    /* ============================================== FIG 1 — esteiras */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      adaptH(fig, 420, 400);
      const S = { k: 7, prof: 'glmP', d: 0.08 };
      const ctl = UI.controls(fig);
      const us = (() => { const r = V.rng(37); return Array.from({ length: 400 }, () => r()); })();
      const effK = () => Math.min(S.k, PROF[S.prof].a.length);
      const tau = () => 1 + PROF[S.prof].a.slice(0, effK()).reduce((s, x) => s + x, 0);
      const Tv = (k) => 1 + 0.04 * k;
      const lanes = [
        { name: 'sem especulação', kind: 'none', T: () => 1 },
        { name: 'rascunho sequencial (MTP/EAGLE)', kind: 'seq', T: () => effK() * S.d + Tv(effK()) },
        { name: 'rascunho em bloco (DFlash/DSpark)', kind: 'blk', T: () => S.d * (1 + 0.05 * effK()) + Tv(effK()) },
      ];
      const rT = UI.readout(ctl, 'τ (tokens/passo)', '—');
      const rS = UI.readout(ctl, 'speedup seq.', '—');
      const rB = UI.readout(ctl, 'speedup bloco', '—');
      const upd = () => { const t = tau(); rT.set(fmt(t, 2)); rS.set(fmt(t / lanes[1].T(), 2) + '×'); rB.set(fmt(t / lanes[2].T(), 2) + '×'); };
      UI.slider(ctl, { label: 'k (rascunhos)', min: 1, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; upd(); } });
      UI.seg(ctl, { label: 'Perfil', options: [{ v: 'glmS', label: 'GLM estrut.' }, { v: 'glmP', label: 'GLM prosa' }, { v: 'ds', label: 'DS Vision-Exp' }], value: S.prof, onChange: (v) => { S.prof = v; upd(); } });
      UI.slider(ctl, { label: 'custo rascunho', min: 0.02, max: 0.3, step: 0.01, value: S.d, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.d = v; upd(); } });
      upd();
      const U = 1.05, RACE = 15;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const a = PROF[S.prof].a, k = effK();
        const tr = (t + 7) % RACE;
        const narrow = w < 600;
        V.text(g, narrow ? 'mesma aceitação; muda o rascunho' : 'mesmo alvo, mesma aceitação — só muda o tempo do rascunho', 16, 20, { size: 12, color: C.muted, maxW: w - 150 });
        V.text(g, `⏱ ${fmt(tr, 1)} s de ${RACE} s`, w - 16, 20, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        if (S.prof === 'ds' && S.k > 6) V.text(g, 'DS: aceitação medida só até a posição 6', 16, 36, { size: 11, color: C.d });
        const top = 46, laneH = (h - top - 8) / 3;
        const Tmax = Math.max(...lanes.map((L) => L.T()));
        const x0 = 16, barMax = w - 32;
        let counts = [];
        lanes.forEach((L, li) => {
          const y = top + li * laneH;
          const T = L.T(), dur = T * U;
          const nDone = Math.floor(tr / dur), ph = (tr % dur) / dur;
          // tokens emitidos
          const cols = [];
          for (let s = 0; s < nDone; s++) {
            if (L.kind === 'none') cols.push(C.muted);
            else { const j = acceptedFromU(a, k, us[s % us.length]); for (let q = 0; q < j; q++) cols.push(C.a); cols.push(C.d); }
          }
          counts.push(cols.length);
          // cabeçalho
          V.text(g, L.name, x0, y + 12, { size: 12.5, weight: 600, color: C.ink, maxW: w * 0.62 });
          V.text(g, `${cols.length} tokens`, w - 16, y + 12, { size: 13, font: 'mono', weight: 700, align: 'right', color: li === 2 ? C.a : li === 1 ? C.b : C.muted });
          // barra do passo
          const bw = barMax * (T / Tmax), by = y + 30, bh = 15;
          V.box(g, x0, by, bw, bh, { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.25) });
          const segs = [];
          if (L.kind === 'none') segs.push({ u: 1, c: C.muted, l: 'decode' });
          else if (L.kind === 'seq') { for (let q = 0; q < k; q++) segs.push({ u: S.d, c: C.b, l: '' }); segs.push({ u: Tv(k), c: C.a, l: 'verifica ' + (k + 1) }); }
          else { segs.push({ u: S.d * (1 + 0.05 * k), c: C.b, l: '' }); segs.push({ u: Tv(k), c: C.a, l: 'verifica ' + (k + 1) }); }
          let cx = x0; const draftEnd = [];
          segs.forEach((sg, si) => {
            const sw = (sg.u / T) * bw;
            V.box(g, cx + 0.5, by + 0.5, Math.max(1, sw - 1), bh - 1, { r: 3, fill: V.alpha(sg.c, sg.c === C.a ? 0.45 : sg.c === C.muted ? 0.35 : 0.6) });
            if (sg.l && sw > 60) V.text(g, sg.l, cx + sw / 2, by + bh / 2 + 0.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            if (sg.c === C.b && L.kind !== 'none') draftEnd.push([cx, cx + sw]);
            cx += sw;
          });
          // rascunhos acima da barra
          if (L.kind !== 'none') {
            const j = acceptedFromU(a, k, us[nDone % us.length]);
            const vStart = x0 + (1 - Tv(k) / T) * bw;
            const inVerify = ph * bw + x0 > vStart;
            const vProg = inVerify ? (ph * bw + x0 - vStart) / (x0 + bw - vStart) : 0;
            for (let q = 0; q < k; q++) {
              let qx;
              if (L.kind === 'seq') { const [s0, s1] = draftEnd[q]; qx = (s0 + s1) / 2; if (x0 + ph * bw < s0) continue; }
              else { const [s0, s1] = draftEnd[0]; if (x0 + ph * bw < s0 + (s1 - s0) * 0.5) continue; qx = s0 + ((q + 0.5) / k) * Math.max(13 * k, s1 - s0); }
              let col = C.b, mood;
              if (inVerify && vProg > (q + 0.5) / (k + 1)) { if (q < j) { col = C.a; } else { col = RED(); mood = 'sad'; } }
              tokChar(g, qx, by - 9, 11, col, { mood });
            }
          }
          // playhead
          const px = x0 + ph * bw;
          V.line(g, px, by - 3, px, by + bh + 3, { color: C.ink, lw: 1.5 });
          V.glow(g, px, by + bh / 2, 9, C.ink, 0.35);
          // bandeja
          const ty = by + bh + 16, cell = narrow ? 11 : 13, cap = Math.max(4, Math.floor((w - 32) / (cell + 3)));
          const rows = Math.max(1, Math.min(2, Math.floor((laneH - 64) / (cell + 3))));
          const show = cols.slice(Math.max(0, cols.length - cap * rows));
          show.forEach((c, qi) => {
            const r = Math.floor(qi / cap), cI = qi % cap;
            const tx = x0 + cell / 2 + cI * (cell + 3), tyy = ty + cell / 2 + r * (cell + 3);
            tokChar(g, tx, tyy, cell, c);
          });
          if (cols.length > cap * rows) V.text(g, `+${cols.length - cap * rows}`, w - 16, ty + cell, { size: 10.5, font: 'mono', color: C.muted, align: 'right' });
        });
        // legenda
        const ly = h - 6;
        const items = [[C.muted, narrow ? 'normal' : 'decode normal'], [C.b, 'rascunho'], [C.a, 'aceito'], [C.d, narrow ? 'bônus' : 'bônus do alvo'], [RED(), 'rejeitado']];
        const iw = items.map(([, l]) => measure(g, l, 10.5) + 12);
        const oneRow = iw.reduce((a, b) => a + b + 14, 0) <= w - 32;
        let lx = 16, row = 0;
        items.forEach(([c, l], i) => {
          if (!oneRow && i === 3) { row = 1; lx = 16; }
          const yy = oneRow ? ly : ly - 15 + row * 15;
          V.dot(g, lx + 4, yy - 4, 4, c); V.text(g, l, lx + 12, yy, { size: 10.5, color: C.muted });
          lx += iw[i] + (oneRow && !narrow ? Math.max(14, 120 - iw[i]) : 14);
        });
      });
    })();

    /* ============================================== FIG 2 — máscara */
    (function () {
      const fig = root.querySelector('#fig-mascara');
      adaptH(fig, 400, 500);
      const S = { causal: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Máscara no bloco', options: [{ v: false, label: 'bidirecional (FLASH_ATTN)' }, { v: true, label: 'causal (TRITON_ATTN)' }], value: false, onChange: (v) => { S.causal = v; } });
      const rowL = ['a', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'];
      const NC = 4;
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const mW = narrow ? w - 20 : w * 0.62, mH = narrow ? h - 170 : h - 70;
        const labW = 34, cols = NC + 8;
        const gap = 3;
        const cw = Math.min(30, (mW - labW - 10) / cols - gap), chh = Math.min(28, (mH - 60) / 8 - gap);
        const ox = 10 + labW, oy = 52;
        // rótulos de grupo
        const ctxW = NC * (cw + gap) - gap, blkX = ox + NC * (cw + gap) + 6;
        V.text(g, 'janela injetada', ox + ctxW / 2, 18, { size: 11, color: C.b, align: 'center', maxW: ctxW + 10 });
        V.text(g, 'chaves do bloco', blkX + (8 * (cw + gap)) / 2, 18, { size: 11, color: C.a, align: 'center' });
        V.line(g, ox, 24, ox + ctxW, 24, { color: C.b, lw: 2 });
        V.line(g, blkX, 24, blkX + 8 * (cw + gap) - gap, 24, { color: C.a, lw: 2 });
        // linha ativa
        let act = Math.floor(t / 1.3) % 8;
        const p = H.pointer;
        if (p.inside && p.y > oy && p.y < oy + 8 * (chh + gap)) act = V.clamp(Math.floor((p.y - oy) / (chh + gap)), 0, 7);
        const colX = (j) => (j < NC ? ox + j * (cw + gap) : blkX + (j - NC) * (cw + gap));
        for (let j = 0; j < cols; j++) V.text(g, j < NC ? 'ctx' : rowL[j - NC], colX(j) + cw / 2, oy - 8, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        for (let i = 0; i < 8; i++) {
          const y = oy + i * (chh + gap);
          V.text(g, rowL[i], ox - 8, y + chh / 2 + 1, { size: 11, font: 'mono', align: 'right', base: 'middle', color: i === act ? C.ink : C.muted, weight: i === act ? 700 : 400 });
          for (let j = 0; j < cols; j++) {
            const x = colX(j);
            const isCtx = j < NC, jb = j - NC;
            const vis = isCtx || !S.causal || jb <= i;
            const lost = !isCtx && S.causal && jb > i;
            let fill = vis ? V.alpha(isCtx ? C.b : C.a, i === act ? 0.85 : 0.42) : V.alpha(C.muted, 0.07);
            V.box(g, x, y, cw, chh, { r: 4, fill });
            if (lost) { g.save(); g.strokeStyle = V.alpha(C.c, 0.7); g.setLineDash([3, 3]); g.lineWidth = 1.2; V.rr(g, x + 1, y + 1, cw - 2, chh - 2, 4); g.stroke(); g.restore(); }
          }
          if (i === act) {
            V.box(g, ox - 3, y - 2, blkX + 8 * (cw + gap) - ox + 2, chh + 4, { r: 6, stroke: C.ink, lw: 1.4 });
            // pulsos percorrendo o que a linha vê
            for (let q = 0; q < 3; q++) {
              const u = ((t * 0.5 + q / 3) % 1);
              const jj = Math.floor(u * cols);
              const isCtx = jj < NC, vis = isCtx || !S.causal || jj - NC <= i;
              if (vis) V.glow(g, colX(jj) + cw / 2, y + chh / 2, 10, C.ink, 0.5);
            }
          }
        }
        // texto da linha ativa
        const seen = S.causal ? (act === 0 ? 'ctx + a' : `ctx + a, ${rowL.slice(1, act + 1).join(', ')}`) : 'ctx + todas as 8 posições do bloco';
        const ty = oy + 8 * (chh + gap) + 18;
        V.text(g, `${rowL[act]} enxerga: ${seen}`, 10, ty, { size: 12, color: C.ink, maxW: mW });
        if (S.causal) V.text(g, 'tracejado rosa = o que o treino via e a máscara causal esconde', 10, ty + 18, { size: 11, color: C.c, maxW: mW });
        // barras medidas
        const bx = narrow ? 20 : w * 0.66, bwid = narrow ? w - 40 : w * 0.32, by0 = narrow ? ty + 40 : 44;
        const groups = [
          { t: 'aceitação média', v: [0.959, 0.31], max: 1, f: (x) => fmt(x, 2) },
          { t: 'tok/s (estruturado)', v: [65.1, 29], max: 70, f: (x) => fmt(x, 1).replace(',0', '') },
        ];
        const gh = narrow ? 58 : 110;
        groups.forEach((G, gi) => {
          const yy = by0 + gi * (gh + (narrow ? 12 : 30));
          V.text(g, G.t, bx, yy, { size: 11.5, color: C.muted, font: 'mono' });
          ['bidirecional', 'causal'].forEach((nm, k) => {
            const on = (k === 1) === S.causal;
            const rowH = narrow ? 16 : 30, yb = yy + 10 + k * (rowH + 6);
            const lw = narrow ? 80 : 86;
            V.text(g, nm, bx, yb + rowH / 2 + 1, { size: 11, base: 'middle', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
            const full = bwid - lw - 44;
            const vw = full * (G.v[k] / G.max);
            V.box(g, bx + lw, yb, full, rowH, { r: 4, fill: V.alpha(C.muted, 0.08) });
            V.box(g, bx + lw, yb, Math.max(2, vw), rowH, { r: 4, fill: V.alpha(k === 0 ? C.a : RED(), on ? 0.85 : 0.3) });
            V.text(g, G.f(G.v[k]), bx + lw + vw + 6, yb + rowH / 2 + 1, { size: 11.5, font: 'mono', base: 'middle', color: on ? C.ink : C.muted });
          });
        });
        if (!narrow) V.text(g, 'medido no GLM (lab C4, 2026-08-30)', bx, by0 + 2 * (gh + 30) - 6, { size: 10.5, color: C.muted });
      });
    })();

    /* ============================================== FIG 3 — pipeline DFlash2 */
    (function () {
      const fig = root.querySelector('#fig-pipeline');
      const steps = [
        { t: 'O alvo verifica e deixa cinco escutas', d: 'O GLM roda o passo de verificação. Nas saídas das camadas 5, 14, 24, 33 e 42 (log: taps 6, 15, 25, 34, 43), o <code>patch_glm_eagle3.py</code> materializa o <code>hc_post</code> e contrai os 4 fluxos mHC: <code>[T, 4, 4096] → [T, 4096]</code>, cinco vezes.' },
        { t: 'Fundir as features', d: 'As 5 features são concatenadas em <code>[T, 20480]</code> e passam por <code>fc</code> \\(\\in\\mathbb{R}^{4096\\times20480}\\) e <code>hidden_norm</code>. Sai um \\(h_p\\) de 4096 por posição verificada.' },
        { t: 'Injetar KV nas 5 camadas', d: 'Um GEMM fundido projeta \\(h_p\\) em K e V das 5 camadas do rascunhador (K recebe K-norm e RoPE), direto no cache SWA de 2048 tokens. Em TP2 são 2048 B por token e camada, em BF16.' },
        { t: 'Montar o bloco', d: 'Query = <code>[a, M, M, M, M, M, M, M]</code>: a âncora (último token aceito, o bônus) e 7 máscaras de id 154856, embutidas pela tabela do <b>alvo</b> (o rascunhador não tem embedding próprio).' },
        { t: 'Cinco camadas, em paralelo', d: 'Cada camada: norm → conv → atenção (Q do bloco; K/V = janela + bloco, <b>bidirecional</b> no bloco) → conv → norm → conv → MLP SwiGLU 12288 → conv. As 8 posições andam juntas.' },
        { t: 'Top-16 por posição', d: 'O <code>lm_head</code> do alvo (154.880 linhas) gera logits para as 7 posições de máscara; <code>torch.topk</code> guarda 16 candidatos e seus logits unários.' },
        { t: 'Seletor e caminhada', d: 'O seletor bilinear (posto 256) pontua transições entre candidatos vizinhos, e <code>_selector_walk_kernel</code> caminha pela treliça (argmax a T=0, Gumbel-max a T&gt;0). Saem 7 rascunhos, e os scores realizados viram o \\(q(x)\\) do rascunho.' },
        { t: 'Verificação', d: 'O alvo roda <code>[a, d₁…d₇]</code>: 8 linhas num forward, com amostragem por rejeição posição a posição. Aceita o maior prefixo e emite o bônus, de 1 a 8 tokens por passo. As features das posições aceitas alimentam o próximo ciclo.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const nodes = [
        { t: 'Alvo · 45 camadas', k: 'Alvo', s: '5 taps → [T,4096]' },
        { t: 'concat → fc', k: 'fc', s: '[T,20480] → [T,4096]' },
        { t: 'KV injetado', k: 'KV', s: 'SWA 2048 × 5 camadas' },
        { t: 'Bloco da query', k: 'Bloco', s: '[a, 154856 ×7]' },
        { t: '5 camadas DFlash2', k: '5 camadas', s: 'conv · attn · mlp' },
        { t: 'lm_head → top-16', k: 'top-16', s: '154880 → 16 / pos.' },
        { t: 'Seletor + walk', k: 'Seletor', s: 'posto 256 · Gumbel' },
        { t: 'Verificação', k: 'Verifica', s: '8 linhas → 1..8 tok' },
      ];
      const rr = V.rng(5);
      const lat = Array.from({ length: 7 }, () => Array.from({ length: 16 }, (_, i) => Math.max(0.05, 1 - i * 0.07 + rr() * 0.15)));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const cols = 4, gapX = w < 600 ? 10 : 22, gapY = 34, mx = 12, my = 14;
        const nw = (w - 2 * mx - (cols - 1) * gapX) / cols, nh = (h - 2 * my - gapY) / 2;
        const pos = (i) => { const r = i < 4 ? 0 : 1; const c = r === 0 ? i : 7 - i; return [mx + c * (nw + gapX), my + r * (nh + gapY)]; };
        const cur = st.i, el = st.elapsed();
        // setas entre nós
        for (let i = 0; i < 7; i++) {
          const [ax, ay] = pos(i), [bx, by] = pos(i + 1);
          let x1, y1, x2, y2;
          if (i === 3) { x1 = ax + nw / 2; y1 = ay + nh; x2 = bx + nw / 2; y2 = by; }
          else if (i < 3) { x1 = ax + nw; y1 = ay + nh / 2; x2 = bx; y2 = by + nh / 2; }
          else { x1 = ax; y1 = ay + nh / 2; x2 = bx + nw; y2 = by + nh / 2; }
          const on = cur === i + 1;
          V.arrow(g, x1, y1, x2, y2, { color: on ? C.a : V.alpha(C.muted, 0.45), lw: on ? 2.4 : 1.4, head: 7 });
          if (on) for (let q = 0; q < 3; q++) { const u = ((el * 0.8 + q / 3) % 1); const [px, py] = V.along(x1, y1, x2, y2, u); V.glow(g, px, py, 7, C.a, 0.9); }
        }
        // retorno (ciclo)
        { const [ax, ay] = pos(7), [bx, by] = pos(0); if (cur === 0) { V.line(g, ax + 6, ay + 10, bx + 6, by + nh - 10, { color: V.alpha(C.d, 0.6), lw: 1.4, dash: [4, 5], dashOffset: -t * 20 }); } }
        nodes.forEach((N, i) => {
          const [x, y] = pos(i);
          const on = i === cur;
          V.box(g, x, y, nw, nh, { r: 12, fill: on ? V.alpha(C.a, 0.12) : V.alpha(C.panel || C.muted, on ? 0.2 : 0.06), stroke: on ? C.a : V.alpha(C.muted, 0.3), lw: on ? 2 : 1 });
          if (on) V.glow(g, x + nw - 10, y + 10, 14, C.a, 0.6);
          g.save(); g.globalAlpha = on ? 1 : 0.55;
          const tiny = nw < 120;
          V.text(g, (i + 1) + '. ' + (tiny ? N.k : N.t), x + 8, y + 17, { size: tiny ? 11 : 12, weight: 600, color: C.ink, maxW: nw - 12 });
          if (tiny) { const parts = N.s.split(' '); const half = Math.ceil(parts.length / 2); [parts.slice(0, half).join(' '), parts.slice(half).join(' ')].forEach((ln, li) => V.text(g, ln, x + 6, y + nh - 22 + li * 12, { size: 9.5, font: 'mono', color: C.muted, maxW: nw - 10 })); }
          else V.text(g, N.s, x + 8, y + nh - 9, { size: 10.5, font: 'mono', color: C.muted, maxW: nw - 14 });
          const ix = x + 8, iy = y + 28, iw = nw - 16, ih = nh - (nw < 120 ? 62 : 50);
          drawIcon(g, i, ix, iy, iw, ih, t, on, C);
          g.restore();
        });
      });
      function drawIcon(g, i, x, y, w, h, t, on, C) {
        const cx = x + w / 2, cy = y + h / 2;
        if (i === 0) { // pilha de 45 camadas com 5 taps
          const taps = [5, 14, 24, 33, 42], n = 45, lh = h / n;
          const bw = Math.min(w * 0.45, 60);
          for (let L = 0; L < n; L++) {
            const tap = taps.includes(L);
            g.fillStyle = tap ? C.d : V.alpha(C.b, 0.35);
            g.fillRect(cx - bw / 2, y + h - (L + 1) * lh, bw, Math.max(1, lh - 0.6));
            if (tap) { V.arrow(g, cx + bw / 2 + 2, y + h - (L + 0.5) * lh, cx + bw / 2 + 16 + (on ? 3 * Math.sin(t * 4 + L) : 0), y + h - (L + 0.5) * lh, { color: C.d, lw: 1.2, head: 4 }); if (w > 120) V.text(g, String(L + 1), cx + bw / 2 + 20, y + h - (L + 0.5) * lh + 3.5, { size: 9.5, font: 'mono', color: C.d }); }
          }
        } else if (i === 1) {
          const cols5 = [C.a, C.b, C.c, C.d, C.e];
          const vh = Math.min(h - 6, 90), vw = Math.max(4, Math.min(8, w * 0.05));
          cols5.forEach((c, k) => { g.fillStyle = V.alpha(c, 0.75); g.fillRect(x + 2 + k * (vw + 2), cy - vh / 2, vw, vh); });
          const mx0 = x + 5 * (vw + 2) + 8, mwid = Math.max(20, w * 0.36);
          V.box(g, mx0, cy - vh / 2, mwid, vh * 0.55, { r: 3, fill: V.alpha(C.b, 0.25), stroke: C.b, lw: 1 });
          if (w > 110) V.text(g, 'W_fc', mx0 + mwid / 2, cy - vh / 2 + vh * 0.28 + 3, { size: 10, font: 'mono', align: 'center', color: C.ink });
          V.arrow(g, mx0 + mwid + 2, cy - vh * 0.2, mx0 + mwid + 14, cy - vh * 0.2, { color: C.muted, lw: 1.2, head: 4 });
          g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(Math.min(x + w - vw, mx0 + mwid + 16), cy - vh / 2, vw, vh * 0.55);
        } else if (i === 2) {
          const rows = 5, rh = Math.min(14, (h - 8) / rows - 3);
          for (let r = 0; r < rows; r++) {
            const yy = y + 4 + r * (rh + 3);
            const n = Math.max(3, Math.floor((w - 20) / 9));
            for (let q = 0; q < n; q++) {
              const filled = on ? q < ((t * 3 + r) % (n + 4)) : q < n - 2;
              g.fillStyle = filled ? V.alpha(q % 2 ? C.a : C.b, 0.7) : V.alpha(C.muted, 0.12);
              g.fillRect(x + 16 + q * 9, yy, 7, rh);
            }
            V.text(g, 'ℓ' + (r + 1), x, yy + rh - 2, { size: 9.5, font: 'mono', color: C.muted });
          }
        } else if (i === 3) {
          const two = w < 110, per = two ? 4 : 8;
          const s = Math.min(18, (w - 8) / per - 3);
          for (let q = 0; q < 8; q++) { const bob = on ? Math.sin(t * 5 + q) * 2 : 0; const rr2 = Math.floor(q / per), cc = q % per; tokChar(g, x + s / 2 + cc * (s + 3), cy + bob + (two ? (rr2 - 0.5) * (s + 4) : 0), s, q === 0 ? C.d : C.muted); }
          if (w > 120) { V.text(g, 'âncora', x + s / 2, cy + s, { size: 9.5, color: C.d, align: 'left' }); }
        } else if (i === 4) {
          const n = 5, sh = Math.min(16, (h - 6) / n - 4);
          const segs = [[C.c, 0.1], [C.a, 0.3], [C.c, 0.1], [C.b, 0.4], [C.c, 0.1]];
          for (let L = 0; L < n; L++) {
            const yy = y + 3 + L * (sh + 4), off = (n - 1 - L) * 3;
            let xx = x + off; const ww = w - 14;
            segs.forEach(([c, f], si) => { const sw = ww * f; g.fillStyle = V.alpha(c, on && Math.floor(t * 3) % 5 === si ? 0.95 : 0.5); g.fillRect(xx, yy, sw - 1, sh); xx += sw; });
          }
        } else if (i === 5) {
          const nC = 7, cw = (w - 4) / nC;
          for (let c = 0; c < nC; c++) for (let r = 0; r < 16; r++) {
            const v = lat[c][r];
            const cellH = (h - 4) / 16;
            g.fillStyle = V.cmap(v * (on ? (0.7 + 0.3 * Math.sin(t * 3 + c)) : 0.8), 'seq');
            g.fillRect(x + 2 + c * cw, y + 2 + r * cellH, cw - 2, cellH - 1);
          }
        } else if (i === 6) {
          const nC = 7, nR = 4, cw = (w - 8) / nC, rh = (h - 8) / nR;
          const path = [0, 0, 1, 0, 0, 2, 0];
          for (let c = 0; c < nC; c++) for (let r = 0; r < nR; r++) V.dot(g, x + 4 + (c + 0.5) * cw, y + 4 + (r + 0.5) * rh, 2.2, V.alpha(C.muted, 0.5));
          const prog = on ? (t * 1.5) % (nC + 1) : nC;
          g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath();
          for (let c = 0; c < Math.min(nC, Math.floor(prog) + 1); c++) { const px = x + 4 + (c + 0.5) * cw, py = y + 4 + (path[c] + 0.5) * rh; c ? g.lineTo(px, py) : g.moveTo(px, py); }
          g.stroke();
          for (let c = 0; c < Math.min(nC, Math.floor(prog) + 1); c++) V.dot(g, x + 4 + (c + 0.5) * cw, y + 4 + (path[c] + 0.5) * rh, 3.5, C.a);
        } else if (i === 7) {
          const two = w < 110, per = two ? 4 : 8;
          const s = Math.min(16, (w - 8) / per - 3);
          const j = 5;
          for (let q = 0; q < 8; q++) {
            const rr2 = Math.floor(q / per), cc = q % per;
            const xx = x + s / 2 + cc * (s + 3);
            const revealed = !on || (t * 3) % 10 > q;
            const col = q === 0 ? C.d : q <= j ? C.a : RED();
            tokChar(g, xx, cy - 4 + (two ? (rr2 - 0.5) * (s + 4) : 0), s, revealed ? col : C.muted, { mood: revealed && q > j ? 'sad' : undefined });
          }
          if (w > 120) { V.text(g, `âncora + ${j} aceitos`, x, cy + s + 6, { size: 10, color: C.muted, maxW: w }); V.text(g, `emite ${j} + 1 bônus`, x, cy + s + 19, { size: 10, color: C.d, maxW: w }); }
        }
      }
    })();

    /* ============================================== FIG 4 — conv dinâmica */
    (function () {
      const fig = root.querySelector('#fig-conv');
      adaptH(fig, 390, 460);
      const B = 8, G = 6;
      const r = V.rng(11);
      const hIn = Array.from({ length: B }, () => Array.from({ length: G }, () => r.normal() * 0.9));
      const b0 = Array.from({ length: G }, () => 0.85 + r.normal() * 0.08), b1 = Array.from({ length: G }, () => 0.35 + r.normal() * 0.08);
      const P0 = Array.from({ length: G }, () => Array.from({ length: G }, () => r.normal() * 0.5));
      const P1 = Array.from({ length: G }, () => Array.from({ length: G }, () => r.normal() * 0.5));
      const S = { dyn: true, str: 0.8, hi: -1 };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'parte dinâmica δ(hₜ)', value: true, onChange: (v) => { S.dyn = v; } });
      UI.slider(ctl, { label: 'intensidade de δ', min: 0, max: 1.5, step: 0.05, value: S.str, fmt: (v) => fmt(v, 2), onInput: (v) => { S.str = v; } });
      UI.slider(ctl, { label: 'posição destacada', min: -1, max: 7, step: 1, value: -1, fmt: (v) => (v < 0 ? 'auto' : v === 0 ? 'a' : 'M' + v), onInput: (v) => { S.hi = v; } });
      const coef = (t, side) => Array.from({ length: G }, (_, gi) => {
        const base = side === 0 ? b0[gi] : b1[gi];
        if (!S.dyn) return base;
        const P = side === 0 ? P0 : P1;
        let z = 0; for (let q = 0; q < G; q++) z += P[gi][q] * hIn[t][q];
        return base + S.str * 0.45 * Math.tanh(z);
      });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const labW = narrow ? 44 : 70;
        const coefW = narrow ? 0 : 150;
        const areaW = w - labW - coefW - 20;
        const colW = areaW / B, cw = Math.min(26, colW * 0.55);
        const topPad = 34, cellH = Math.min(15, (h - (narrow ? 200 : 120)) / (2 * G + 4));
        const yIn = topPad, yOut = yIn + G * (cellH + 2) + (narrow ? 60 : 80);
        const hi = S.hi >= 0 ? S.hi : Math.floor(t / 1.6) % B;
        const cx = (tt) => 10 + labW + tt * colW + colW / 2;
        // calcular saídas
        const out = [];
        let mx = 0.01;
        for (let tt = 0; tt < B; tt++) {
          const c0 = coef(tt, 0), c1 = coef(tt, 1);
          out.push({ c0, c1, v: Array.from({ length: G }, (_, gi) => c0[gi] * hIn[tt][gi] + (tt % B >= 1 ? c1[gi] * hIn[tt - 1][gi] : 0)) });
        }
        out.forEach((o) => o.v.forEach((x) => { mx = Math.max(mx, Math.abs(x)); }));
        V.text(g, 'entrada hₜ', 10, yIn + (G * (cellH + 2)) / 2, { size: 11, color: C.muted, base: 'middle', maxW: labW - 4 });
        V.text(g, 'saída oₜ', 10, yOut + (G * (cellH + 2)) / 2, { size: 11, color: C.muted, base: 'middle', maxW: labW - 4 });
        for (let tt = 0; tt < B; tt++) {
          const x = cx(tt) - cw / 2;
          V.text(g, tt === 0 ? 'a' : 'M' + tt, cx(tt), yIn - 10, { size: 11, font: 'mono', align: 'center', color: tt === hi ? C.ink : C.muted, weight: tt === hi ? 700 : 400 });
          for (let gi = 0; gi < G; gi++) {
            g.fillStyle = V.cmap(hIn[tt][gi] / 2, 'div'); V.rr(g, x, yIn + gi * (cellH + 2), cw, cellH, 3); g.fill();
            g.fillStyle = V.cmap(out[tt].v[gi] / mx, 'div'); V.rr(g, x, yOut + gi * (cellH + 2), cw, cellH, 3); g.fill();
          }
          if (tt === hi) { V.box(g, x - 3, yIn - 3, cw + 6, G * (cellH + 2) + 4, { r: 5, stroke: C.c, lw: 1.6 }); V.box(g, x - 3, yOut - 3, cw + 6, G * (cellH + 2) + 4, { r: 5, stroke: C.c, lw: 1.6 }); }
          // setas
          const y1 = yIn + G * (cellH + 2) + 2, y2 = yOut - 6;
          const m0 = out[tt].c0.reduce((s, x) => s + Math.abs(x), 0) / G, m1 = out[tt].c1.reduce((s, x) => s + Math.abs(x), 0) / G;
          const alpha = tt === hi ? 1 : 0.45;
          V.arrow(g, cx(tt), y1, cx(tt), y2, { color: V.alpha(C.a, alpha), lw: 1 + m0 * 3.2, head: 6 });
          if (tt >= 1) {
            V.arrow(g, cx(tt - 1) + cw / 2, y1, cx(tt) - 3, y2, { color: V.alpha(C.b, alpha), lw: 0.8 + m1 * 5, head: 6 });
            if (tt === hi) for (let q = 0; q < 2; q++) { const u = ((t * 0.9 + q / 2) % 1); const [px, py] = V.along(cx(tt - 1) + cw / 2, y1, cx(tt) - 3, y2, u); V.glow(g, px, py, 7, C.b, 0.9); }
          } else {
            g.save(); g.setLineDash([3, 3]); V.line(g, cx(0) - colW * 0.45, y1 + 4, cx(0) - 6, y2 - 8, { color: V.alpha(RED(), 0.7), lw: 1.3 }); g.restore();
            V.text(g, '✕', cx(0) - colW * 0.3, (y1 + y2) / 2 + 4, { size: 12, color: RED(), align: 'center' });
          }
          if (tt === hi) { const u = (t * 0.9) % 1; V.glow(g, cx(tt), V.lerp(y1, y2, u), 7, C.a, 0.9); }
        }
        V.text(g, 'fronteira: t mod B = 0', cx(0) - colW * 0.45, yOut + G * (cellH + 2) + 16, { size: 10.5, color: RED(), maxW: colW * 2 });
        // painel de coeficientes
        const o = out[hi];
        const px0 = narrow ? 10 + labW : w - coefW - 4, py0 = narrow ? yOut + G * (cellH + 2) + 30 : yIn;
        const pw = narrow ? w - labW - 24 : coefW - 8;
        V.text(g, `coef. de ${hi === 0 ? 'a' : 'M' + hi} por grupo`, px0, py0 + 4, { size: 11, color: C.ink, weight: 600, maxW: pw });
        const rowH = narrow ? 9 : 15;
        const labG = narrow ? 0 : 20, half = (pw - labG) / 2 - 6, bx0 = px0 + labG, bx1 = bx0 + half + 12;
        for (let gi = 0; gi < G; gi++) {
          const yy = py0 + 14 + gi * (rowH + 4);
          const w0 = half * Math.min(1, Math.abs(o.c0[gi]) / 1.4), w1 = hi >= 1 ? half * Math.min(1, Math.abs(o.c1[gi]) / 1.4) : 0;
          g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(bx0, yy, half, rowH); g.fillRect(bx1, yy, half, rowH);
          g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(bx0, yy, w0, rowH);
          g.fillStyle = V.alpha(C.b, 0.8); g.fillRect(bx1, yy, Math.max(0, w1), rowH);
          if (!narrow) V.text(g, 'g' + gi, px0, yy + rowH - 3, { size: 9.5, font: 'mono', color: C.muted });
        }
        const ly = py0 + 14 + G * (rowH + 4) + 12;
        V.text(g, 'tap 0 (hₜ)', bx0, ly, { size: 10.5, color: C.a, maxW: half + 10 });
        V.text(g, 'tap 1 (hₜ₋₁)', bx1, ly, { size: 10.5, color: C.b, maxW: half + 10 });
        V.text(g, S.dyn ? 'b + δ(hₜ): muda por token' : 'só base_kernel: igual em todas', px0, ly + 16, { size: 10.5, color: C.muted, maxW: pw });
      });
    })();

    /* ============================================== FIG 5 — treliça do seletor */
    (function () {
      const fig = root.querySelector('#fig-trelica');
      adaptH(fig, 430, 500);
      const A = ['dia', ',', 'tudo', 'bem', '?', 'Eu', 'vou'];
      const Bm = ['jogo', '!', 'Até', 'logo', '.', 'Nos', 'vemos'];
      const uA = [2.0, 1.6, 1.5, 1.25, 1.1, 0.85, 0.8], uB = [1.9, 1.7, 1.45, 1.3, 1.05, 0.9, 0.75];
      const dis = [['sol', 'mar'], ['e', 'de'], ['que', 'o'], ['mais', 'já'], ['!!', ';'], ['Tu', 'Se'], ['já', 'me']];
      const cand = A.map((_, i) => [
        { w: A[i], u: uA[i], m: 'A' }, { w: Bm[i], u: uB[i], m: 'B' }, { w: dis[i][0], u: 0.45, m: null }, { w: dis[i][1], u: 0.25, m: null },
      ].sort((p, q) => q.u - p.u));
      const pair = (pm, cm) => (pm && cm ? (pm === cm ? 2.0 : -1.0) : 0);
      const S = { mode: 'walk', T: 0, seed: 1, t0: 0 };
      let noise = [];
      const mkNoise = () => { const r = V.rng(1000 + S.seed); noise = cand.map((c) => c.map(() => -Math.log(-Math.log(Math.max(1e-9, r()))))); };
      mkNoise();
      const solve = () => {
        const path = [], stepScores = [];
        let prev = null;
        for (let i = 0; i < 7; i++) {
          const sc = cand[i].map((c, ci) => {
            let s = c.u + (S.mode === 'walk' ? pair(prev, c.m) : 0);
            if (S.T > 0) s = s / S.T + noise[i][ci];
            return s;
          });
          let best = 0; sc.forEach((s, ci) => { if (s > sc[best]) best = ci; });
          path.push(best); stepScores.push(sc); prev = cand[i][best].m;
        }
        return { path, stepScores };
      };
      let sol = solve();
      /* painel bilinear: vetores ilustrativos de 16 células (representam 256) */
      const NV = 16, rb = V.rng(77);
      const PA = Array.from({ length: NV }, () => (rb() < 0.5 ? -1 : 1) * (0.45 + 0.55 * rb()));
      const vecP = { A: PA, B: PA.map((x) => -0.9 * x + (rb() - 0.5) * 0.2), null: PA.map(() => (rb() - 0.5) * 0.25) };
      const hv = Array.from({ length: NV }, (_, i) => (i % 5 === 3 ? 0.08 : 0.35 + 0.65 * rb()));
      const vecS = { A: PA.map((x) => 0.85 * x + (rb() - 0.5) * 0.25), B: PA.map((x) => -0.85 * x + (rb() - 0.5) * 0.25) };
      const noiseS = cand.map((col) => col.map(() => Array.from({ length: NV }, () => (rb() - 0.5) * 0.35)));
      const ctl = UI.controls(fig);
      const out = UI.readout(ctl, 'rascunho', '—');
      const re = () => { sol = solve(); S.t0 = performance.now() / 1000; out.set('Bom ' + sol.path.map((ci, i) => cand[i][ci].w).join(' ')); };
      UI.seg(ctl, { label: 'Escolha', options: [{ v: 'walk', label: 'seletor (caminhada)' }, { v: 'indep', label: 'argmax independente' }], value: 'walk', onChange: (v) => { S.mode = v; re(); } });
      UI.seg(ctl, { label: 'Temperatura', options: [{ v: 0, label: 'T = 0 (argmax)' }, { v: 0.8, label: 'T = 0,8 (Gumbel)' }], value: 0, onChange: (v) => { S.T = v; re(); } });
      UI.button(ctl, '🎲 novo ruído', () => { S.seed++; mkNoise(); re(); });
      re();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const P = narrow ? 4 : 7;
        const colW = (w - 20) / (P + 1), chipW = Math.min(76, colW - 8), chipH = 24;
        const oy = 44;
        const want = narrow ? 206 : 140;
        const rowGap = V.clamp((h - want - oy - 16 - 4 * chipH) / 3, 12, 40);
        const rowY = (r) => oy + r * (chipH + rowGap);
        const colX = (c) => 10 + c * colW + (colW - chipW) / 2;
        const el = performance.now() / 1000 - S.t0;
        const period = 0.75, prog = Math.min(P, Math.floor(el / period) + 1);
        const loopT = (P + 4) * period;
        if (el > loopT) S.t0 = performance.now() / 1000;
        V.text(g, narrow ? 'posições 1–4 de 7 (4 de 16 candidatos)' : '4 de 16 candidatos por posição; linhas = scores de transição', 10, 16, { size: 11.5, color: C.muted, maxW: w - 20 });
        // âncora
        chip(g, colX(0), rowY(1.5), chipW, chipH, 'Bom', { fill: V.alpha(C.d, 0.3), stroke: C.d, weight: 700 });
        V.text(g, 'âncora', colX(0) + chipW / 2, rowY(1.5) + chipH + 14, { size: 10.5, color: C.d, align: 'center' });
        // arestas
        for (let i = 0; i < P; i++) {
          const fromList = i === 0 ? [{ x: colX(0) + chipW, y: rowY(1.5) + chipH / 2, m: null }] : cand[i - 1].map((c, r) => ({ x: colX(i) + chipW, y: rowY(r) + chipH / 2, m: c.m }));
          fromList.forEach((f) => cand[i].forEach((c, r) => {
            if (S.mode !== 'walk') return;
            const pv = pair(f.m, c.m);
            if (pv === 0) return;
            V.line(g, f.x, f.y, colX(i + 1), rowY(r) + chipH / 2, { color: V.alpha(pv > 0 ? C.a : C.c, pv > 0 ? 0.28 : 0.14), lw: pv > 0 ? 1.6 : 1 });
          }));
        }
        // caminho escolhido
        g.save();
        let px = colX(0) + chipW, py = rowY(1.5) + chipH / 2;
        for (let i = 0; i < Math.min(P, prog); i++) {
          const r = sol.path[i];
          const nx = colX(i + 1), ny = rowY(r) + chipH / 2;
          V.line(g, px, py, nx, ny, { color: C.a, lw: 3 });
          px = nx + chipW; py = ny;
        }
        g.restore();
        // chips candidatos
        for (let i = 0; i < P; i++) {
          V.text(g, 'pos ' + (i + 1), colX(i + 1) + chipW / 2, oy - 10, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
          cand[i].forEach((c, r) => {
            const chosen = i < prog && sol.path[i] === r;
            const col = c.m === 'A' ? C.a : c.m === 'B' ? C.b : C.muted;
            chip(g, colX(i + 1), rowY(r), chipW, chipH, c.w, { fill: chosen ? V.alpha(col, 0.55) : V.alpha(col, 0.12), stroke: chosen ? C.ink : V.alpha(col, 0.6), lw: chosen ? 2 : 1, weight: chosen ? 700 : 500 });
            if (rowGap >= 22) V.text(g, 'u=' + fmt(c.u, 2), colX(i + 1) + chipW / 2, rowY(r) + chipH + 12, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
          });
        }
        // inset ancorado logo abaixo da treliça
        const si = Math.min(P, prog) - 1;
        const trBot = rowY(3) + chipH + (rowGap >= 22 ? 16 : 6);
        const ix = 10, iw = w - 20, iy = trBot + 8, ih = h - iy - 4;
        V.box(g, ix, iy, iw, ih, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25) });
        const sc = sol.stepScores[si];
        const lab = S.mode === 'walk' ? `passo ${si + 1}: s = u + transição${S.T > 0 ? ' → s/T + g (Gumbel)' : ''}` : `passo ${si + 1}: só u${S.T > 0 ? '/T + g' : ''} (sem transição)`;
        const barsW = narrow ? iw - 20 : iw * 0.42;
        V.text(g, lab, ix + 10, iy + 16, { size: 11, color: C.ink, maxW: barsW });
        const mn = Math.min(...sc), mxs = Math.max(...sc);
        const bw = barsW / 4 - 8;
        sc.forEach((sv, r) => {
          const bx = ix + 10 + r * (bw + 8), bh = 34 * ((sv - mn) / (mxs - mn || 1)) + 4;
          const chosen = sol.path[si] === r;
          const c = cand[si][r];
          const col = c.m === 'A' ? C.a : c.m === 'B' ? C.b : C.muted;
          g.fillStyle = V.alpha(col, chosen ? 0.9 : 0.35); V.rr(g, bx, iy + 62 - bh, bw, bh, 3); g.fill();
          V.text(g, `${c.w} ${fmt(sv, 1)}`, bx + bw / 2, iy + 76, { size: 10.5, font: 'mono', align: 'center', color: chosen ? C.ink : C.muted, maxW: bw + 6 });
        });
        // painel bilinear: de onde vem a transição
        const dx = narrow ? ix + 10 : ix + 20 + barsW, dy = narrow ? iy + 90 : iy + 8;
        const dw = narrow ? iw - 20 : iw - barsW - 30, dh = narrow ? ih - 94 : ih - 12;
        if (dh > 60) {
          const walk = S.mode === 'walk';
          const prevM = si === 0 ? null : cand[si - 1][sol.path[si - 1]].m;
          const prevW = si === 0 ? 'Bom' : cand[si - 1][sol.path[si - 1]].w;
          const Pv = vecP[prevM === 'A' ? 'A' : prevM === 'B' ? 'B' : 'null'];
          const gv = Pv.map((x, k) => x * hv[k]);
          const rows = [
            { l: `P[“${prevW}”]`, v: Pv, n: 'linha de P [154880×256]' },
            { l: 'W_h·h', v: hv.map((x) => x * 0.9), n: '[256×4096] · h' },
            { l: '= portão', v: gv, n: 'produto elemento a elemento' },
          ].concat(cand[si].map((c, r) => ({ l: `S[“${c.w}”]`, v: c.m ? vecS[c.m] : noiseS[si][r], val: pair(prevM, c.m), m: c.m, chosen: sol.path[si] === r })));
          V.text(g, narrow ? 'transição = ⟨P[c₋₁] ⊙ W_h h, S[c]⟩' : 'transição = ⟨ P[c₋₁] ⊙ W_h h , S[c] ⟩  (16 células = 256 números)', dx, dy + 10, { size: 10.5, color: walk ? C.ink : C.muted, maxW: dw });
          const labW = narrow ? 64 : 76, valW = narrow ? 36 : 44;
          const noteW = narrow ? 0 : Math.max(0, Math.min(150, dw - labW - valW - 16 * 9 - 8));
          const cellW = Math.max(4, Math.min(11, (dw - labW - valW - noteW - 8) / NV - 1));
          const rh = Math.max(8, Math.min(12, (dh - 20) / rows.length - 3));
          const sweep = Math.floor(t * 7) % NV;
          g.save(); if (!walk) g.globalAlpha = 0.35;
          rows.forEach((R, ri) => {
            const yy = dy + 18 + ri * (rh + 3) + (ri >= 3 ? 4 : 0);
            const col = R.m === 'A' ? C.a : R.m === 'B' ? C.b : ri < 3 ? C.d : C.muted;
            V.text(g, R.l, dx + labW - 6, yy + rh - 1, { size: 9.5, font: 'mono', align: 'right', color: ri < 3 ? C.d : col, maxW: labW - 6, weight: R.chosen ? 700 : 400 });
            for (let k = 0; k < NV; k++) {
              g.fillStyle = V.cmap(V.clamp(R.v[k], -1, 1), 'div');
              g.fillRect(dx + labW + k * (cellW + 1), yy, cellW, rh);
            }
            if (walk) { g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1; g.strokeRect(dx + labW + sweep * (cellW + 1) - 0.5, yy - 0.5, cellW + 1, rh + 1); }
            const xEnd = dx + labW + NV * (cellW + 1) + 6;
            if (R.val != null) V.text(g, (R.val > 0 ? '+' : R.val < 0 ? '−' : '') + fmt(Math.abs(R.val), 1), xEnd, yy + rh - 1, { size: 10, font: 'mono', weight: 700, color: R.val > 0 ? C.a : R.val < 0 ? C.c : C.muted });
            else if (noteW > 40) V.text(g, R.n, xEnd, yy + rh - 1, { size: 9.5, color: C.muted, maxW: noteW + valW });
          });
          g.restore();
          if (!walk) V.text(g, 'argmax independente: nenhuma transição é somada', dx, dy + dh - 2, { size: 10.5, color: C.c, maxW: dw });
        }
      });
    })();

    /* ============================================== FIG 6 — Markov */
    (function () {
      const fig = root.querySelector('#fig-markov');
      adaptH(fig, 480, 560);
      const outCol = (() => { const r = V.rng(21); return Array.from({ length: 26 }, (_, i) => (i === 4 ? 0.95 : i === 11 ? -0.9 : (r() - 0.5) * 0.5)); })();
      const rowFrac = (wd) => { let hh = 7; for (const ch of wd) hh = (hh * 31 + ch.charCodeAt(0)) % 997; return 0.12 + 0.76 * (hh / 997); };
      const cands = [
        [['São', 0.50], ['Rio', 0.47], ['Belo', 0.03]],
        [['Paulo', 0.47], ['de', 0.49], ['Horiz.', 0.04]],
        [['.', 0.46], ['Janeiro', 0.50], ['!', 0.04]],
      ];
      const bias = { 'São': { Paulo: 2.2, de: -2.2, 'Horiz.': -1 }, 'Rio': { de: 2.2, Paulo: -2.2 }, 'Belo': { 'Horiz.': 3 }, 'Paulo': { '.': 2.2, Janeiro: -2.2 }, 'de': { Janeiro: 2.2, '.': -2.2 }, 'Horiz.': { '.': 1.5 } };
      const conf = { 'São': 0.9, 'Rio': 0.88, 'Paulo': 0.93, 'de': 0.9, '.': 0.95, 'Janeiro': 0.94 };
      const S = { markov: true, first: 'argmax', conf: true, t0: 0 };
      const ctl = UI.controls(fig);
      const reset = () => { S.t0 = performance.now() / 1000; };
      UI.seg(ctl, { label: 'Cabeça de Markov', options: [{ v: false, label: 'desligada' }, { v: true, label: 'ligada' }], value: true, onChange: (v) => { S.markov = v; reset(); } });
      UI.seg(ctl, { label: '1ª posição', options: [{ v: 'argmax', label: 'argmax' }, { v: 'Rio', label: 'amostrou “Rio”' }], value: 'argmax', onChange: (v) => { S.first = v; reset(); } });
      UI.toggle(ctl, { label: 'mostrar confiança', value: true, onChange: (v) => { S.conf = v; } });
      UI.button(ctl, '▶ reiniciar', reset);
      const softmax = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
      const solve = () => {
        const ys = [], probs = [];
        let prev = 'Destino:';
        for (let i = 0; i < 3; i++) {
          const lb = cands[i].map(([wd, p]) => Math.log(p));
          const bs = cands[i].map(([wd]) => (S.markov && bias[prev] && bias[prev][wd]) || 0);
          const pr = softmax(lb.map((x, j) => x + bs[j]));
          let best = 0; pr.forEach((p, j) => { if (p > pr[best]) best = j; });
          if (i === 0 && S.first === 'Rio') best = 1;
          ys.push(best); probs.push({ pr, bs, base: cands[i].map(([, p]) => p) });
          prev = cands[i][best][0];
        }
        return { ys, probs };
      };
      S.t0 = performance.now() / 1000 - 3.8; // começa mostrando o resultado final
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const sol = solve();
        const narrow = w < 600;
        const el = performance.now() / 1000 - S.t0;
        const stepT = 1.2;
        const cur = S.markov ? Math.min(3, Math.floor(el / stepT)) : (el > 0.6 ? 3 : 0);
        if (el > 3 * stepT + 3) reset();
        const colW = (w - 40) / 3, cx = (i) => 20 + i * colW + colW / 2;
        // bloco de entrada
        const chipW = Math.min(96, colW - 16);
        const inY = 12;
        ['Destino:', 'ruído', 'ruído'].forEach((l, i) => chip(g, cx(i) - chipW / 2, inY, chipW, 24, l, { fill: V.alpha(i === 0 ? C.d : C.muted, 0.25), stroke: i === 0 ? C.d : V.alpha(C.muted, 0.5), font: i === 0 ? 'sans' : 'mono', size: 11.5 }));
        // backbone
        const bbY = inY + 36, bbH = 34;
        V.box(g, 20, bbY, w - 40, bbH, { r: 10, fill: V.alpha(C.b, 0.14 + 0.08 * Math.sin(t * 3)), stroke: C.b });
        V.text(g, narrow ? 'backbone paralelo (não causal)' : 'backbone paralelo — 3 estágios V4 + MoE, não causal, um forward', w / 2, bbY + bbH / 2 + 1, { size: 11.5, color: C.ink, align: 'center', base: 'middle', maxW: w - 60 });
        for (let i = 0; i < 2; i++) { const u = (t * 0.7 + i * 0.5) % 1; g.strokeStyle = V.alpha(C.b, 0.6); g.lineWidth = 1.2; g.beginPath(); g.moveTo(cx(i), bbY + 4); g.quadraticCurveTo((cx(i) + cx(i + 1)) / 2, bbY - 14, cx(i + 1), bbY + 4); g.stroke(); V.glow(g, V.lerp(cx(i), cx(i + 1), u), bbY - 6 * Math.sin(Math.PI * u) - 2, 6, C.b, 0.8); }
        // barras
        const barTop = bbY + bbH + 40, barH = narrow ? 100 : 90;
        V.text(g, 'logits por posição (barra cheia = com viés; contorno = marginal)', 20, bbY + bbH + 16, { size: 10.5, color: C.muted, maxW: w - 40 });
        for (let i = 0; i < 3; i++) {
          const P = sol.probs[i];
          const decided = i < cur;
          const bw = (colW - 30) / 3;
          cands[i].forEach(([wd], j) => {
            const bx = cx(i) - (colW - 30) / 2 + j * bw + 3;
            const pShow = (S.markov && i <= cur) ? P.pr[j] : P.base[j];
            const hb = barH * pShow, hbase = barH * P.base[j];
            const chosen = decided && sol.ys[i] === j;
            g.fillStyle = V.alpha(chosen ? C.a : C.b, chosen ? 0.85 : 0.4); V.rr(g, bx, barTop + barH - hb, bw - 6, Math.max(2, hb), 3); g.fill();
            g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]); V.rr(g, bx, barTop + barH - hbase, bw - 6, Math.max(2, hbase), 3); g.stroke(); g.setLineDash([]);
            const b = P.bs[j];
            if (S.markov && b && i <= cur && i > 0) V.text(g, b > 0 ? '+' : '−', bx + (bw - 6) / 2, barTop + barH - Math.max(hb, hbase) - 6, { size: 13, weight: 700, align: 'center', color: b > 0 ? C.a : C.c });
            V.text(g, wd, bx + (bw - 6) / 2, barTop + barH + 14, { size: narrow ? 9.5 : 11, font: 'mono', align: 'center', color: chosen ? C.ink : C.muted, maxW: bw });
          });
        }
        // setas de Markov
        const yChip = barTop + barH + 30;
        for (let i = 0; i < 3; i++) {
          if (i >= cur) continue;
          const wd = cands[i][sol.ys[i]][0];
          chip(g, cx(i) - chipW / 2, yChip, chipW, 24, wd, { fill: V.alpha(C.a, 0.3), stroke: C.a, weight: 700 });
          if (S.markov && i < 2) {
            const x1 = cx(i) + chipW / 2 + 2, x2 = cx(i + 1) - (colW - 30) / 2;
            const on = cur === i + 1 || (cur > i + 1);
            V.arrow(g, x1, yChip + 12, x2, barTop + barH * 0.4, { color: V.alpha(C.d, on ? 0.9 : 0.4), lw: 1.8, head: 7, curve: -30 });
            if (!narrow) V.text(g, 'W₂W₁[y]', (x1 + x2) / 2, yChip + 30, { size: 10.5, font: 'mono', align: 'center', color: C.d });
          }
          if (S.conf && S.markov) V.text(g, 'c=' + fmt(conf[wd] || 0.5, 2), cx(i), yChip + 40, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        }
        // frase final
        const fy = yChip + (narrow ? 58 : 62);
        if (cur >= 3) {
          const phrase = 'Destino: ' + sol.ys.map((j, i) => cands[i][j][0]).join(' ');
          const ok = (sol.ys[0] === 0 && sol.ys[1] === 0 && sol.ys[2] === 0) || (sol.ys[0] === 1 && sol.ys[1] === 1 && sol.ys[2] === 1);
          V.text(g, phrase, 20, fy, { size: 14, weight: 700, color: ok ? C.a : C.c, maxW: w - 150 });
          V.text(g, ok ? '✓ coerente' : '✕ colisão multimodal', w - 20, fy, { size: 12.5, weight: 700, align: 'right', color: ok ? C.a : C.c });
          if (S.conf && S.markov) {
            const cs = sol.ys.map((j, i) => conf[cands[i][j][0]] || 0.5);
            let a = 1; const surv = cs.map((c) => (a *= c));
            V.text(g, 'sobrevivência a_j = ' + surv.map((x) => fmt(x, 2)).join(' → '), 20, fy + 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 40 });
          }
        } else V.text(g, S.markov ? 'refino sequencial: posição ' + (cur + 1) + ' de 3…' : '…', 20, fy, { size: 12, color: C.muted });
        // inset: a consulta W₁[y] e a GEMV por W₂, com as formas reais
        const iy0 = fy + (S.conf && S.markov ? 34 : 18), ih = h - 6 - iy0;
        if (ih > 110) {
          V.box(g, 12, iy0, w - 24, ih, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.22) });
          // a posição em decisão usa o token escolhido antes dela (y₀ = âncora)
          const posI = Math.min(cur, 2), prevI = posI - 1;
          const word = prevI >= 0 ? cands[prevI][sol.ys[prevI]][0] : 'Destino:';
          V.text(g, S.markov ? `${narrow ? '' : 'cabeça de Markov: '}consulta + GEMV, y = “${word}” → viés da posição ${posI + 1}` : 'cabeça desligada: nenhum viés é somado', 24, iy0 + 17, { size: 11, weight: 600, color: S.markov ? C.ink : C.muted, maxW: w - 48 });
          const mw = narrow ? 40 : 54, gap1 = narrow ? 74 : 110, vw = narrow ? 44 : 64;
          const dx0 = narrow ? 26 : 70, dy = iy0 + 28, dh = narrow ? Math.min(96, ih - 128) : ih - 58;
          const x2 = dx0 + mw + gap1, x3 = x2 + mw + (narrow ? 14 : 24), ow = narrow ? 11 : 14;
          g.save(); if (!S.markov) g.globalAlpha = 0.35;
          const ph = S.markov ? (t % 3.4) / 3.4 : 0;
          // W₁ e W₂ como matrizes altas [V × 256]
          [[dx0, 'W₁'], [x2, 'W₂']].forEach(([mx0, nm], mi) => {
            V.box(g, mx0, dy, mw, dh, { r: 3, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.7), lw: 1 });
            g.strokeStyle = V.alpha(C.b, 0.18); g.lineWidth = 1; g.beginPath();
            for (let q = 1; q < 18; q++) { const yy = dy + (q / 18) * dh; g.moveTo(mx0 + 2, yy); g.lineTo(mx0 + mw - 2, yy); }
            g.stroke();
            V.text(g, nm, mx0 + mw / 2, dy + dh + 13, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.b });
            V.text(g, '[129.280×256]', mx0 + mw / 2, dy + dh + 25, { size: 9.5, font: 'mono', align: 'center', color: C.muted, maxW: mw + (mi ? 22 : gap1 - 8) });
          });
          // 1) a linha y de W₁ acende
          const ry = dy + rowFrac(word) * (dh - 6);
          const lit = S.markov ? V.clamp(ph / 0.2, 0, 1) : 0;
          g.fillStyle = V.alpha(C.d, 0.25 + 0.7 * lit); g.fillRect(dx0 + 1, ry, mw - 2, 6);
          if (lit > 0.5) V.glow(g, dx0 + mw / 2, ry + 3, 14, C.d, 0.6);
          if (!narrow) V.text(g, 'linha y', dx0 - 4, ry + 6, { size: 9.5, align: 'right', color: C.d });
          // 2) o vetor de 256 viaja até W₂
          const vRestX = dx0 + mw + (gap1 - vw) / 2, vRestY = dy + dh / 2 - 4;
          const mv = V.ease.inOut(V.clamp((ph - 0.2) / 0.22, 0, 1));
          const vx = V.lerp(dx0, vRestX, mv), vy = V.lerp(ry, vRestY, mv), vwn = V.lerp(mw, vw, mv);
          for (let k = 0; k < 8; k++) { g.fillStyle = V.alpha(C.d, 0.55 + 0.4 * ((k + Math.floor(t * 6)) % 8 === 0)); g.fillRect(vx + k * (vwn / 8), vy, vwn / 8 - 1, 8); }
          V.text(g, 'e ∈ ℝ²⁵⁶', vRestX + vw / 2, vRestY + 22, { size: 10, font: 'mono', align: 'center', color: C.d, alpha: mv });
          // 3) GEMV: a varredura desce por W₂ e enche a coluna de 129.280 logits
          const scan = V.clamp((ph - 0.45) / 0.45, 0, 1);
          if (scan > 0 && scan < 1) {
            const sy = dy + scan * dh;
            V.line(g, x2 + 1, sy, x2 + mw - 1, sy, { color: C.d, lw: 2 });
            V.line(g, vRestX + vw, vRestY + 4, x2, sy, { color: V.alpha(C.d, 0.5), lw: 1, dash: [3, 3] });
            V.glow(g, x2 + mw / 2, sy, 10, C.d, 0.6);
          }
          const nC = outCol.length, chh = dh / nC;
          for (let q = 0; q < nC; q++) {
            const on = (q + 0.5) / nC <= (scan > 0 ? scan : ph > 0.9 ? 1 : 0);
            g.fillStyle = on ? V.cmap(outCol[q], 'div') : V.alpha(C.muted, 0.12);
            g.fillRect(x3, dy + q * chh, ow, Math.max(1, chh - 1));
          }
          V.text(g, 'Δℓ', x3 + ow + 6, dy + 10, { size: 11, font: 'mono', weight: 700, color: C.ink });
          V.text(g, '[129.280]', x3 + ow + 6, dy + 23, { size: 9.5, font: 'mono', color: C.muted });
          V.text(g, '+ ℓ_base', x3 + ow + 6, dy + dh / 2 + 10, { size: 10.5, font: 'mono', color: C.ink });
          g.restore();
          // escala: bigrama cheio V×V contra as duas faixas fatoradas (largura real 512/129.280 da aresta)
          const sq = narrow ? 46 : Math.min(dh, 84);
          const sx = narrow ? dx0 : x3 + ow + 86, sy0 = narrow ? dy + dh + 38 : dy + (dh - sq) / 2;
          if (sx + sq + 150 < w) {
            g.save(); g.setLineDash([4, 3]); V.box(g, sx, sy0, sq, sq, { r: 2, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.6), lw: 1 }); g.restore();
            const fw = Math.max(1.5, sq * 512 / 129280);
            g.fillStyle = C.d; g.fillRect(sx, sy0, fw, sq); g.fillRect(sx + fw + 1.5, sy0, fw, sq);
            const tx = sx + sq + 10;
            V.text(g, 'V×V cheio ≈ 33 GB', tx, sy0 + 13, { size: 10.5, color: C.muted, maxW: w - tx - 18 });
            V.text(g, 'W₁ + W₂ ≈ 2 × 66 MB', tx, sy0 + 29, { size: 10.5, weight: 600, color: C.d, maxW: w - tx - 18 });
            V.text(g, '= 0,4% da área (faixas em escala)', tx, sy0 + 44, { size: 10, color: C.muted, maxW: w - tx - 18 });
          }
        }
      });
    })();

    /* ============================================== FIG 7 — quanto verificar */
    (function () {
      const fig = root.querySelector('#fig-verificar');
      adaptH(fig, 420, 520);
      const S = { mode: 'glm', alpha: 0.25, margin: 1.0, sat: 'max', prof: 'glmP', seed: 3, cRow: 0.1, sched: 'hardware', th: 0.5, curve: 'decl', early: true };
      const ctl = UI.controls(fig);
      const gA = LA.h('div', { style: { display: 'contents' } }), gB = LA.h('div', { style: { display: 'contents' } });
      const setMode = (m) => { S.mode = m; gA.style.display = m === 'glm' ? 'contents' : 'none'; gB.style.display = m === 'ds' ? 'contents' : 'none'; recompute(); };
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'glm', label: 'GLM · adaptive-k' }, { v: 'ds', label: 'DeepSeek · confiança' }], value: 'glm', onChange: setMode });
      ctl.appendChild(gA); ctl.appendChild(gB);
      UI.slider(gA, { label: 'α da EMA', min: 0.05, max: 0.8, step: 0.05, value: S.alpha, fmt: (v) => fmt(v, 2), onInput: (v) => { S.alpha = v; recompute(); } });
      UI.slider(gA, { label: 'margem m', min: 0, max: 3, step: 0.25, value: S.margin, fmt: (v) => fmt(v, 2), onInput: (v) => { S.margin = v; recompute(); } });
      UI.seg(gA, { label: 'saturate', options: ['max', 'n'], value: 'max', onChange: (v) => { S.sat = v; recompute(); } });
      UI.seg(gA, { label: 'Texto', options: [{ v: 'glmS', label: 'estruturado' }, { v: 'glmP', label: 'prosa' }, { v: 'mix', label: 'prosa → estrut.' }], value: 'glmP', onChange: (v) => { S.prof = v; recompute(); } });
      UI.slider(gA, { label: 'custo c', min: 0, max: 0.3, step: 0.01, value: S.cRow, fmt: (v) => fmt(v, 2), onInput: (v) => { S.cRow = v; recompute(); } });
      UI.button(gA, '🎲 sortear', () => { S.seed++; recompute(); });
      UI.seg(gB, { label: 'Agendador', options: [{ v: 'off', label: 'off' }, { v: 'threshold', label: 'limiar' }, { v: 'hardware', label: 'hardware' }], value: 'hardware', onChange: (v) => { S.sched = v; recompute(); } });
      UI.slider(gB, { label: 'limiar', min: 0, max: 1, step: 0.05, value: S.th, fmt: (v) => fmt(v, 2), onInput: (v) => { S.th = v; recompute(); } });
      UI.seg(gB, { label: 'Curva sps', options: [{ v: 'flat', label: 'plana' }, { v: 'decl', label: 'cai com o lote' }, { v: 'cliff', label: 'penhasco' }], value: 'decl', onChange: (v) => { S.curve = v; recompute(); } });
      UI.toggle(gB, { label: 'early stop', value: true, onChange: (v) => { S.early = v; recompute(); } });
      const r1 = UI.readout(ctl, '—', '—'), r2 = UI.readout(ctl, '—', '—'), r3 = UI.readout(ctl, 'tok/s relativo (ilustr.)', '—');
      const lab1 = r1.el.querySelector('span'), lab2 = r2.el.querySelector('span');
      /* --- GLM: simulação do _Glm53AdaptiveK --- */
      const SET = [2, 4, 7], KMAX = 7, MIN_STEPS = 4, NST = 60;
      let sim = null;
      function simulate() {
        const r = V.rng(100 + S.seed);
        let ema = null, obsN = 0;
        const rows = [];
        for (let s = 0; s < NST; s++) {
          const prof = S.prof === 'mix' ? (s < 30 ? PROF.glmP.a : PROF.glmS.a) : PROF[S.prof].a;
          let n = KMAX;
          if (ema != null && obsN >= MIN_STEPS) {
            const target = Math.ceil(ema + S.margin);
            const c = SET.filter((v) => v <= Math.min(target, KMAX));
            n = c.length ? Math.max(...c) : Math.min(...SET);
          }
          const L = acceptedFromU(prof, KMAX, r());
          const acc = Math.min(L, n);
          const obs = acc >= n && S.sat !== 'n' ? Math.max(KMAX, n) : acc;
          ema = ema == null ? obs * S.alpha + KMAX * (1 - S.alpha) : obs * S.alpha + ema * (1 - S.alpha);
          obsN++;
          rows.push({ n, acc, L, ema });
        }
        const tokA = rows.reduce((a, x) => a + 1 + x.acc, 0) / NST, rowsA = rows.reduce((a, x) => a + x.n + 1, 0) / NST;
        const tokF = rows.reduce((a, x) => a + 1 + x.L, 0) / NST;
        return { rows, tokA, rowsA, tokF };
      }
      /* --- DS: hardware_aware_prefix_schedule (porte do overlay) --- */
      const REQ = [
        { n: 'JSON', c: [0.97, 0.96, 0.95, 0.93, 0.9, 0.88] },
        { n: 'código', c: [0.92, 0.85, 0.78, 0.7, 0.62, 0.55] },
        { n: 'prosa', c: [0.72, 0.6, 0.5, 0.42, 0.35, 0.3] },
        { n: 'prosa difícil', c: [0.55, 0.38, 0.3, 0.22, 0.18, 0.12] },
      ];
      const surv = REQ.map((q) => { let a = 1; return q.c.map((c) => (a *= c)); });
      const sps = (bt) => (S.curve === 'flat' ? 1 : S.curve === 'decl' ? 1 / (1 + 0.045 * bt) : (bt <= 12 ? 1 : 0.8));
      let plan = null;
      function schedule() {
        const R = REQ.length;
        const cands = [];
        surv.forEach((sv, r) => sv.forEach((a, j) => cands.push([a, r, j + 1])));
        cands.sort((p, q) => q[0] - p[0] || p[1] - q[1] || p[2] - q[2]);
        const lens = Array(R).fill(0);
        let bt = R, E = R;
        let best = { lens: lens.slice(), E, bt, tps: E * sps(bt) };
        const path = [{ bt, tps: E * sps(bt), stop: false }];
        let stopped = false, stopIdx = -1;
        for (const [a, r, pos] of cands) {
          if (pos !== lens[r] + 1) continue;
          lens[r] = pos; bt += 1; E += a;
          const tps = E * sps(bt);
          path.push({ bt, tps });
          if (!stopped) {
            if (tps > best.tps) best = { lens: lens.slice(), E, bt, tps };
            else if (S.early) { stopped = true; stopIdx = path.length - 1; }
          }
        }
        if (S.sched === 'off' || S.sched === 'threshold') {
          const L2 = surv.map((sv) => (S.sched === 'off' ? sv.length : (() => { let n = 0; for (const a of sv) { if (a < S.th) break; n++; } return n; })()));
          const E2 = R + L2.reduce((s, l, r) => s + surv[r].slice(0, l).reduce((a, b) => a + b, 0), 0);
          const bt2 = R + L2.reduce((a, b) => a + b, 0);
          return { lens: L2, E: E2, bt: bt2, tps: E2 * sps(bt2), path, stopIdx: -1 };
        }
        return Object.assign(best, { path, stopIdx });
      }
      function recompute() {
        r3.el.style.display = S.mode === 'glm' ? '' : 'none';
        if (S.mode === 'glm') {
          sim = simulate(); lab1.textContent = 'tokens/passo'; r1.set(fmt(sim.tokA, 2) + ` (k=7 fixo: ${fmt(sim.tokF, 2)})`); lab2.textContent = 'linhas/requisição'; r2.set(fmt(sim.rowsA, 2) + ' (fixo: 8)');
          const vA = sim.tokA / (1 + S.cRow * sim.rowsA), vF = sim.tokF / (1 + S.cRow * 8), dp = (vA / vF - 1) * 100;
          r3.set(`${fmt(vA, 2)} vs ${fmt(vF, 2)} (${dp >= 0 ? '+' : '−'}${fmt(Math.abs(dp), 0)}%)`);
        } else { plan = schedule(); lab1.textContent = 'E[aceitos] · linhas'; r1.set(fmt(plan.E, 2) + ' · ' + plan.bt); lab2.textContent = 'E[tok/s] relativo'; r2.set(fmt(plan.tps, 3)); }
      }
      setMode('glm');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.mode === 'glm') {
          if (!sim) return;
          const L = 36, Rm = 14, T = narrow ? 42 : 30, Bm = 34;
          const pw = w - L - Rm, ph = h - T - Bm;
          const X = (s) => L + (s + 0.5) * (pw / NST), Y = (v) => T + ph - (v / 7) * ph;
          for (let v = 0; v <= 7; v++) { V.line(g, L, Y(v), L + pw, Y(v), { color: V.alpha(C.muted, v % 7 ? 0.1 : 0.3), lw: 1 }); if (v % 1 === 0) V.text(g, String(v), L - 8, Y(v) + 4, { size: 10, font: 'mono', align: 'right', color: C.muted }); }
          SET.forEach((v) => V.text(g, 'n=' + v, L + pw, Y(v) - 3, { size: 9.5, font: 'mono', align: 'right', color: V.alpha(C.a, 0.8) }));
          const rev = Math.min(NST, Math.floor((t * 9 + NST) % (NST + 24)));
          if (S.prof === 'mix') { V.line(g, X(29.5), T, X(29.5), T + ph, { color: V.alpha(C.d, 0.6), dash: [4, 4] }); V.text(g, 'vira estruturado', X(30) + 4, T + 12, { size: 10.5, color: C.d }); }
          // n (escada)
          g.strokeStyle = C.a; g.lineWidth = 2.2; g.beginPath();
          sim.rows.slice(0, rev).forEach((r, s) => { const x0 = L + s * (pw / NST), x1 = x0 + pw / NST; s ? g.lineTo(x0, Y(r.n)) : g.moveTo(x0, Y(r.n)); g.lineTo(x1, Y(r.n)); });
          g.stroke();
          // EMA
          g.strokeStyle = C.b; g.lineWidth = 1.8; g.beginPath();
          sim.rows.slice(0, rev).forEach((r, s) => { s ? g.lineTo(X(s), Y(Math.min(7.5, r.ema))) : g.moveTo(X(s), Y(Math.min(7.5, r.ema))); });
          g.stroke();
          // aceitos
          sim.rows.slice(0, rev).forEach((r, s) => { V.dot(g, X(s), Y(r.acc), 3, r.acc >= r.n ? C.d : C.c); });
          if (rev > 0 && rev <= NST) V.glow(g, X(rev - 1), Y(sim.rows[rev - 1].acc), 10, C.ink, 0.5);
          V.text(g, 'passo →', L + pw, h - 10, { size: 10.5, color: C.muted, align: 'right' });
          const lg = [[C.a, 'n verificado'], [C.b, 'EMA'], [C.c, 'aceitos'], [C.d, 'aceitou todos (censurado)']];
          lg.forEach(([c, l], i) => { const lx = L + i * (narrow ? (pw / 2) : 150), ly = narrow ? (i < 2 ? 12 : 24) : 16; const lxx = narrow ? L + (i % 2) * (pw / 2) : lx; V.dot(g, lxx + 4, ly - 4, 4, c); V.text(g, l, lxx + 12, ly, { size: 10.5, color: C.muted, maxW: narrow ? pw / 2 - 16 : 140 }); });
        } else {
          if (!plan) return;
          const gridW = narrow ? w - 20 : w * 0.56;
          const labW = 86, cellW = Math.min(52, (gridW - labW) / 6 - 4), cellH = narrow ? 30 : 42;
          const oy = 44;
          V.text(g, 'sobrevivência a(r,j) = Π c (confianças ilustrativas)', 10, 18, { size: 11, color: C.muted, maxW: gridW });
          for (let j = 0; j < 6; j++) V.text(g, 'j=' + (j + 1), 10 + labW + j * (cellW + 4) + cellW / 2, oy - 8, { size: 10, font: 'mono', align: 'center', color: C.muted });
          REQ.forEach((q, r) => {
            const y = oy + r * (cellH + 6);
            V.text(g, q.n, 10 + labW - 8, y + cellH / 2 + 1, { size: 11, align: 'right', base: 'middle', color: C.ink, maxW: labW - 10 });
            surv[r].forEach((a, j) => {
              const x = 10 + labW + j * (cellW + 4);
              const adm = j < plan.lens[r];
              g.fillStyle = V.cmap(a, 'seq'); V.rr(g, x, y, cellW, cellH, 5); g.fill();
              if (adm) { V.box(g, x - 1.5, y - 1.5, cellW + 3, cellH + 3, { r: 6, stroke: C.ink, lw: 2 }); }
              else { g.fillStyle = V.alpha(C.bg || '#000', 0.55); V.rr(g, x, y, cellW, cellH, 5); g.fill(); }
              if (cellW >= 30) V.text(g, fmt(a, 2), x + cellW / 2, y + cellH / 2 + 1, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: a > 0.6 ? (C.dark ? '#00210F' : '#ffffff') : C.ink });
            });
          });
          const gy = oy + 4 * (cellH + 6) + 6;
          V.text(g, 'contorno = verificado; esmaecido = cortado', 10, gy + 8, { size: 10.5, color: C.muted });
          // gráfico E[tok/s] × batch_tokens
          const px0 = narrow ? 40 : w * 0.62, py0 = narrow ? gy + 30 : 44, pw = narrow ? w - 60 : w * 0.35, ph = narrow ? h - gy - 60 : h - 110;
          V.box(g, px0, py0, pw, ph, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25) });
          V.text(g, 'E[tok/s] × linhas no lote', px0, py0 - 8, { size: 10.5, color: C.muted, maxW: pw });
          const pts = plan.path.length ? plan.path : [{ bt: plan.bt, tps: plan.tps }];
          const all = pts.map((p) => p.tps).concat([plan.tps]);
          const ymax = Math.max(...all) * 1.1, ymin = Math.min(...all) * 0.9;
          const X = (bt) => px0 + ((bt - 4) / 24) * pw, Y = (v) => py0 + ph - ((v - ymin) / (ymax - ymin || 1)) * ph;
          if (plan.path.length) {
            g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 1.8; g.beginPath();
            pts.forEach((p, i) => (i ? g.lineTo(X(p.bt), Y(p.tps)) : g.moveTo(X(p.bt), Y(p.tps)))); g.stroke();
            pts.forEach((p, i) => V.dot(g, X(p.bt), Y(p.tps), 2.4, i === plan.stopIdx ? RED() : C.b));
            const rev = Math.floor((t * 6) % (pts.length + 8));
            if (rev < pts.length) V.glow(g, X(pts[rev].bt), Y(pts[rev].tps), 9, C.b, 0.9);
          }
          V.dot(g, X(plan.bt), Y(plan.tps), 6, C.a);
          const nearR = X(plan.bt) > px0 + pw - 80;
          V.text(g, S.sched === 'hardware' ? 'escolhido' : S.sched === 'off' ? 'off: verifica tudo' : 'limiar', X(plan.bt) + (nearR ? -10 : 10), Y(plan.tps) - 8, { size: 10.5, color: C.a, align: nearR ? 'right' : 'left' });
          if (plan.stopIdx > 0) {
            const ex = X(pts[plan.stopIdx].bt), ey = Y(pts[plan.stopIdx].tps), lx = Math.min(ex + 6, px0 + pw - 76), lyy = Math.min(ey + 34, py0 + ph - 26);
            V.line(g, ex, ey + 3, lx + 30, lyy - 2, { color: V.alpha(RED(), 0.6), lw: 1 });
            V.box(g, lx, lyy - 2, 66, 18, { r: 5, fill: V.alpha(C.bg || C.panel, 0.92), stroke: V.alpha(RED(), 0.7), lw: 1 });
            V.text(g, 'early stop', lx + 33, lyy + 11, { size: 10, color: RED(), align: 'center' });
          }
          if (S.sched !== 'hardware') V.text(g, 'linha = caminho guloso do modo hardware', px0 + 8, py0 + ph - 8, { size: 10, color: C.muted, maxW: pw - 16 });
          V.text(g, '4', px0, py0 + ph + 12, { size: 9.5, font: 'mono', color: C.muted });
          V.text(g, '28 linhas', px0 + pw, py0 + ph + 12, { size: 9.5, font: 'mono', color: C.muted, align: 'right' });
        }
      });
    })();

    /* ============================================== FIG 8 — páginas */
    (function () {
      const fig = root.querySelector('#fig-paginas');
      adaptH(fig, 340, 380);
      const TP = { 2: { Bm: 3584, bm: 656, bd: 2048 }, 3: { Bm: 2560, bm: 656, bd: 1536 } };
      const S = { tp: 2, mode: 'compact', idx: 0 };
      const candsOf = (tp) => { const { Bm } = TP[tp]; const out = []; for (let B = 64; B <= Bm; B += 64) if (Bm % B === 0) out.push(B); return out; };
      const ctl = UI.controls(fig);
      const rB = UI.readout(ctl, 'bloco do rascunhador', '—'), rI = UI.readout(ctl, 'ids/requisição', '—'), rO = UI.readout(ctl, 'página ocupada', '—');
      let sl;
      const curB = () => {
        const c = candsOf(S.tp), { Bm, bm, bd } = TP[S.tp], page = Bm * bm;
        if (S.mode === 'slot') return 64;
        if (S.mode === 'compact') return Math.max(...c.filter((B) => B * bd <= page));
        return c[Math.min(S.idx, c.length - 1)];
      };
      const syncSl = () => { if (!sl) return; const c = candsOf(S.tp), i = Math.max(0, c.indexOf(curB())); sl.input.value = i; const v = sl.el.querySelector('.val'); if (v) v.textContent = String(c[i]); };
      const upd = () => { const B = curB(), { Bm, bm, bd } = TP[S.tp]; syncSl(); rB.set(B + ' tokens'); rI.set(String(Math.ceil(4096 / B) + 1)); rO.set(Math.round((B * bd) / (Bm * bm) * 100) + '%'); };
      UI.seg(ctl, { label: 'Topologia', options: [{ v: 2, label: 'TP2' }, { v: 3, label: 'TP3' }], value: 2, onChange: (v) => { S.tp = v; sl.input.max = candsOf(v).length - 1; S.idx = Math.min(S.idx, candsOf(v).length - 1); upd(); } });
      const segB = UI.seg(ctl, { label: 'Bloco', options: [{ v: 'slot', label: 'slot-share 64' }, { v: 'compact', label: 'COMPACT=1' }, { v: 'manual', label: 'manual' }], value: 'compact', onChange: (v) => { if (v === 'manual' && S.mode !== 'manual') S.idx = +sl.input.value; S.mode = v; upd(); } });
      sl = UI.slider(ctl, { label: 'candidato', min: 0, max: candsOf(2).length - 1, step: 1, value: 0, fmt: (v) => String(candsOf(S.tp)[Math.min(v, candsOf(S.tp).length - 1)]), onInput: (v) => { S.idx = v; if (S.mode !== 'manual') segB.set('manual'); upd(); } });
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const { Bm, bm, bd } = TP[S.tp], page = Bm * bm, B = curB();
        const x0 = 14, bw = w - 28;
        // 1) página
        let y = 20;
        V.text(g, `página física MLA: ${fmtInt(Bm)} tok × ${bm} B = ${fmtInt(page)} B`, x0, y, { size: 11.5, color: C.ink, maxW: bw });
        y += 10;
        const fitW = bw * 0.82;
        V.box(g, x0, y, fitW, 26, { r: 6, fill: V.alpha(C.b, 0.14), stroke: C.b });
        const dp = B * bd, frac = dp / page;
        const fits = frac <= 1;
        const fw = Math.min(bw, fitW * frac);
        V.box(g, x0, y + 4, Math.max(3, fw), 18, { r: 4, fill: V.alpha(fits ? C.a : RED(), 0.75) });
        V.text(g, narrow ? `rascunho: ${B} × ${bd} B (${Math.round(frac * 100)}%)` : `página do rascunhador: ${B} × ${bd} B = ${fmtInt(dp)} B (${Math.round(frac * 100)}%)`, x0 + 8, y + 14, { size: 10.5, font: 'mono', base: 'middle', color: C.ink, maxW: bw - 16 });
        if (!fits) V.text(g, '✕ não cabe', x0 + fitW + 6, y + 18, { size: 11, color: RED(), weight: 700 });
        // 2) candidatos
        y += 50;
        V.text(g, `múltiplos de 64 que dividem ${Bm}:`, x0, y, { size: 11, color: C.muted });
        y += 8;
        const cs = candsOf(S.tp), cwid = Math.min(62, (bw - (cs.length - 1) * 6) / cs.length);
        cs.forEach((c, i) => {
          const ok = c * bd <= page, on = c === B;
          chip(g, x0 + i * (cwid + 6), y, cwid, 22, String(c), { fill: V.alpha(ok ? C.a : RED(), on ? 0.6 : 0.14), stroke: on ? C.ink : V.alpha(ok ? C.a : RED(), 0.6), lw: on ? 2 : 1, size: 11 });
        });
        // 3) fita de tokens
        y += 48;
        V.text(g, 'tokens do rascunhador por requisição: janela 2048 + 2048 em voo', x0, y, { size: 11, color: C.ink, maxW: bw });
        y += 8;
        const nb = Math.ceil(4096 / B);
        const tapeH = 24;
        V.box(g, x0, y, bw, tapeH, { r: 4, fill: V.alpha(C.muted, 0.08) });
        const cursor = ((t * 700) % 4096);
        for (let k = 0; k < nb; k++) {
          const a0 = (k * B) / 4096 * bw, a1 = Math.min(bw, ((k + 1) * B) / 4096 * bw);
          const filled = cursor >= k * B;
          g.fillStyle = V.alpha(k % 2 ? C.a : C.b, filled ? 0.7 : 0.25); g.fillRect(x0 + a0, y, Math.max(1, a1 - a0 - (nb < 80 ? 1 : 0)), tapeH);
        }
        V.line(g, x0 + (cursor / 4096) * bw, y - 3, x0 + (cursor / 4096) * bw, y + tapeH + 3, { color: C.ink, lw: 1.5 });
        V.line(g, x0 + bw / 2, y - 4, x0 + bw / 2, y + tapeH + 4, { color: C.d, lw: 1.5, dash: [3, 3] });
        V.text(g, 'janela', x0 + bw / 4, y + tapeH + 14, { size: 10.5, color: C.muted, align: 'center' });
        V.text(g, 'em voo', x0 + (3 * bw) / 4, y + tapeH + 14, { size: 10.5, color: C.muted, align: 'center' });
        // 4) ids
        y += tapeH + 34;
        const ids = nb + 1;
        V.text(g, `ids de bloco: ⌈4096 / ${B}⌉ + 1 = ${ids}`, x0, y, { size: 13, weight: 700, color: ids <= 8 ? C.a : C.d });
        y += 10;
        const sq = narrow ? 10 : 14, perRow = Math.floor(bw / (sq + 3));
        for (let k = 0; k < ids; k++) {
          const rx = x0 + (k % perRow) * (sq + 3), ry = y + Math.floor(k / perRow) * (sq + 3);
          if (ry > h - sq - 4) break;
          const pulse = Math.floor(t * 8) % ids === k;
          g.fillStyle = V.alpha(k === ids - 1 ? C.d : C.b, pulse ? 1 : 0.65); V.rr(g, rx, ry, sq, sq, 2); g.fill();
        }
        // 5) pool medido (TP2)
        const usedRows = Math.ceil(ids / perRow);
        const yb = Math.max(y + usedRows * (sq + 3) + 26, h - 58);
        if (yb + 44 <= h) {
          V.text(g, S.tp === 2 ? 'pool logado com KV de 14 GiB (TP2, 850k), medido:' : 'pool medido só em TP2:', x0, yb, { size: 11, color: C.muted, maxW: bw });
          const bars = [['sem compact', 876958, S.mode === 'slot'], ['compact', 1572073, S.mode === 'compact']];
          const lw = 92, full = Math.max(60, bw - lw - 190);
          bars.forEach(([lb, v, on], i) => {
            const yy = yb + 8 + i * 18, ww = full * (v / 1572073);
            V.text(g, lb, x0, yy + 11, { size: 11, color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
            g.fillStyle = V.alpha(i ? C.a : C.b, on ? 0.85 : 0.35); V.rr(g, x0 + lw, yy, ww, 13, 3); g.fill();
            V.text(g, fmtInt(v) + ' tokens' + (i ? ' (1,85×)' : ''), x0 + lw + ww + 6, yy + 11, { size: 10.5, font: 'mono', color: on ? C.ink : C.muted });
          });
        }
      });
    })();

    /* ============================================== FIG 9 — anel, condensação, ragged */
    (function () {
      const fig = root.querySelector('#fig-anel');
      adaptH(fig, 420, 430);
      const W = 16;
      const S = { scene: 'ring', fix: true, ev: 0, lastEv: 0 };
      const ring = { slots: Array.from({ length: W }, (_, i) => ({ pos: 24 + i, st: 'ok' })), P: 40, hist: null };
      for (let p = 24; p < 40; p++) ring.slots[p % W] = { pos: p, st: 'ok' };
      const rr = V.rng(9);
      function ringStep() {
        const k = 6, rows = k + 1;
        const j = acceptedFromU(PROF.ds.a, k, rr());
        const acc = j + 1;
        const touched = [];
        for (let q = 0; q < rows; q++) {
          const pos = ring.P + q, sl = pos % W;
          if (q < acc) { ring.slots[sl] = { pos, st: 'new' }; touched.push([sl, 'new']); }
          else if (!S.fix) { ring.slots[sl] = { pos, st: 'bad' }; touched.push([sl, 'bad']); }
          else { touched.push([sl, 'kept']); if (ring.slots[sl].st === 'new') ring.slots[sl].st = 'ok'; }
        }
        ring.slots.forEach((s, i) => { if (s.st === 'new' && !touched.find((x) => x[0] === i)) s.st = 'ok'; });
        ring.hist = { acc, rej: rows - acc, touched, P: ring.P };
        ring.P += acc;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cena', options: [{ v: 'ring', label: '1 · anel' }, { v: 'cond', label: '2 · condensação' }, { v: 'rag', label: '3 · ragged' }], value: 'ring', onChange: (v) => { S.scene = v; S.ev = 0; } });
      UI.toggle(ctl, { label: 'correção ligada (máscara / Patch 1 / Patch 2)', value: true, onChange: (v) => { S.fix = v; } });
      UI.button(ctl, '▶ próximo evento', () => { if (S.scene === 'ring') ringStep(); else S.ev = (S.ev + 1) % 3; S.lastEv = performance.now() / 1000; }, { primary: true });
      V.every(2400, () => { if (S.scene === 'ring' && performance.now() / 1000 - S.lastEv > 4) ringStep(); });
      ringStep();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.scene === 'ring') {
          const R = Math.min(narrow ? w * 0.32 : 130, h * 0.34), cx = narrow ? w / 2 : 30 + R + 20, cy = narrow ? R + 34 : h / 2 + 6;
          V.text(g, `anel: slot = pos mod ${W} (no real: 128)`, narrow ? 14 : 20, 18, { size: 11.5, color: C.muted });
          for (let i = 0; i < W; i++) {
            const a0 = (i / W) * Math.PI * 2 - Math.PI / 2, a1 = ((i + 1) / W) * Math.PI * 2 - Math.PI / 2;
            const s = ring.slots[i];
            const col = s.st === 'new' ? C.a : s.st === 'bad' ? RED() : C.b;
            const age = Math.max(0, Math.min(1, (ring.P - s.pos) / W));
            g.beginPath(); g.arc(cx, cy, R, a0 + 0.02, a1 - 0.02); g.arc(cx, cy, R * 0.62, a1 - 0.02, a0 + 0.02, true); g.closePath();
            g.fillStyle = V.alpha(col, s.st === 'ok' ? 0.25 + 0.45 * (1 - age) : 0.8); g.fill();
            const kept = ring.hist && S.fix && ring.hist.touched.find((x) => x[0] === i && x[1] === 'kept');
            if (kept) { g.save(); g.setLineDash([3, 3]); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); g.restore(); }
            const am = (a0 + a1) / 2;
            V.text(g, String(s.pos), cx + Math.cos(am) * R * 0.81, cy + Math.sin(am) * R * 0.81 + 4, { size: R > 100 ? 11 : 9.5, font: 'mono', align: 'center', color: C.ink });
          }
          const hand = ((ring.P - 1) % W + 0.5) / W * Math.PI * 2 - Math.PI / 2;
          V.arrow(g, cx, cy, cx + Math.cos(hand) * R * 0.55, cy + Math.sin(hand) * R * 0.55, { color: C.ink, lw: 2, head: 7 });
          V.glow(g, cx + Math.cos(hand) * R * 0.55, cy + Math.sin(hand) * R * 0.55, 12, C.a, 0.5 + 0.3 * Math.sin(t * 4));
          V.text(g, 'pos ' + (ring.P - 1), cx, cy + 4, { size: 11, font: 'mono', align: 'center', color: C.muted });
          const tx = narrow ? 14 : cx + R + 40, ty = narrow ? cy + R + 30 : 60, tw = narrow ? w - 28 : w - tx - 16;
          if (ring.hist) {
            const H = ring.hist;
            const lines = [
              [C.ink, `passo em pos ${H.P}: verificou 7 linhas (k=6)`],
              [C.a, `${H.acc} aceitas → gravadas no anel`],
              [S.fix ? C.d : RED(), S.fix ? `${H.rej} rejeitadas → slot mantém o valor antigo` : `${H.rej} rejeitadas → gravaram um futuro que não aconteceu`],
              [C.muted, S.fix ? 'VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1' : 'sem máscara: o drafter lê lixo como contexto'],
            ];
            lines.forEach(([c, l], i) => V.text(g, l, tx, ty + i * 22, { size: 12, color: c, maxW: tw, weight: i === 0 ? 600 : 400 }));
            const bad = ring.slots.filter((s) => s.st === 'bad').length;
            V.text(g, `slots poluídos: ${bad}`, tx, ty + 4 * 22 + 8, { size: 13, weight: 700, color: bad ? RED() : C.a });
          }
        } else if (S.scene === 'cond') {
          const reqs = { A: C.a, B: C.b, C: C.c, D: C.d, E: C.e };
          const ev = S.ev;
          const rows = ev === 0 ? ['A', 'B', 'C', 'D'] : ev === 1 ? ['A', 'D', 'C', null] : ['A', 'D', 'C', 'E'];
          const slotOwner = ev === 0 ? ['A', 'B', 'C', 'D'] : ev === 1 ? ['A', 'B', 'C', 'D'] : (S.fix ? ['A', 'E', 'C', 'D'] : ['A', 'B', 'C', 'D']);
          const map = S.fix ? (ev === 0 ? { A: 0, B: 1, C: 2, D: 3 } : ev === 1 ? { A: 0, C: 2, D: 3 } : { A: 0, C: 2, D: 3, E: 1 }) : null;
          const titles = ['4 requisições em voo', 'B terminou → lote condensado (D sobe para a linha 1)', 'E chega'];
          V.text(g, titles[ev], 14, 20, { size: 12, weight: 600, color: C.ink, maxW: w - 28 });
          const lx = 20, ly = 44, rowH = V.clamp((h - 150) / 4 - 8, 34, 60), colW = narrow ? w * 0.36 : 150;
          const sx = narrow ? w * 0.56 : w * 0.5;
          V.text(g, 'linhas do lote', lx, ly - 6, { size: 10.5, color: C.muted });
          V.text(g, 'anéis do rascunhador (slots)', sx, ly - 6, { size: 10.5, color: C.muted, maxW: w - sx - 10 });
          for (let r = 0; r < 4; r++) {
            const y = ly + r * (rowH + 8);
            V.box(g, lx, y, colW, rowH, { r: 8, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
            V.text(g, 'linha ' + r, lx + 8, y + rowH / 2 + 4, { size: 10.5, font: 'mono', color: C.muted });
            const q = rows[r];
            if (q) tokChar(g, lx + colW - 24, y + rowH / 2 + (ev === 1 && r === 1 ? Math.sin(t * 6) * 2 : 0), Math.min(26, rowH - 10), reqs[q]);
            if (q) V.text(g, q, lx + colW - 44, y + rowH / 2 + 5, { size: 13, weight: 700, color: reqs[q], align: 'right' });
            // slot
            const sw = Math.min(narrow ? w - sx - 14 : 150, w - sx - 14);
            const own = slotOwner[r];
            V.box(g, sx, y, sw, rowH, { r: 8, fill: V.alpha(reqs[own], 0.18), stroke: reqs[own] });
            for (let k = 0; k < 8; k++) { g.fillStyle = V.alpha(reqs[own], 0.3 + 0.5 * ((k + Math.floor(t * 3)) % 8 === 0)); g.fillRect(sx + 8 + k * ((sw - 50) / 8), y + rowH / 2 - 5, (sw - 50) / 8 - 2, 10); }
            V.text(g, 'S' + r + ' · hist. ' + own, sx + sw - 6, y + 12, { size: 10, font: 'mono', align: 'right', color: C.ink });
            // quem lê qual slot
            if (q) {
              const slot = S.fix ? map[q] : r;
              const ty = ly + slot * (rowH + 8) + rowH / 2;
              const wrong = slotOwner[slot] !== q;
              V.arrow(g, lx + colW + 4, y + rowH / 2, sx - 4, ty, { color: wrong ? RED() : V.alpha(C.a, 0.8), lw: wrong ? 2.4 : 1.4, head: 6 });
              if (wrong) V.text(g, '⚡', (lx + colW + sx) / 2, (y + ty) / 2 + rowH / 2 - 8, { size: 16, align: 'center', color: RED() });
            }
          }
          const by = ly + 4 * (rowH + 8) + 12;
          const wrongAny = rows.some((q, r) => q && (S.fix ? slotOwner[map[q]] !== q : slotOwner[r] !== q));
          V.text(g, S.fix ? 'Patch 1: _req_id_to_slot = ' + JSON.stringify(map).replace(/"/g, '') + (ev >= 1 ? '  · menor slot livre primeiro' : '') : 'sem Patch 1: slot = linha do lote', 14, by, { size: 11, font: 'mono', color: S.fix ? C.a : C.muted, maxW: w - 28 });
          V.text(g, wrongAny ? '✕ requisição lendo o histórico de outra → aceitação ≈ 0 (sem crash)' : '✓ cada requisição lê o próprio anel', 14, by + 20, { size: 12, weight: 700, color: wrongAny ? RED() : C.a, maxW: w - 28 });
        } else {
          const pLen = S.fix ? 33 : 34;
          const segs = [{ n: 'P (chunk de prefill)', len: pLen, c: C.b, rej: 0 }, { n: 'Q (decode, k+1 = 7)', len: 7, c: C.a, rej: 3 }];
          const tot = pLen + 7;
          V.text(g, S.fix ? `hoje: ${tot} linhas, batch_size = 2 (segmentos desiguais, ${tot} % 2 = 0)` : 'histórico: 41 linhas, batch_size = 2', 14, 20, { size: 12, weight: 600, color: C.ink, maxW: w - 28 });
          const x0 = 14, bw = w - 28, cw = bw / tot, y = 48, th = 28;
          let acc = 0;
          segs.forEach((sg) => { for (let k = 0; k < sg.len; k++) { g.fillStyle = V.alpha(sg.c, 0.55 + 0.25 * ((k + Math.floor(t * 6)) % sg.len === 0)); g.fillRect(x0 + (acc + k) * cw, y, Math.max(1, cw - 1), th); } acc += sg.len; });
          if (!S.fix) {
            // reshape retangular falhando (antes do Patch 2)
            const ry = y + 56;
            V.text(g, 'reshape retangular [2, 41/2, H] …', x0, ry, { size: 12, font: 'mono', color: C.muted, maxW: bw });
            for (let r = 0; r < 2; r++) for (let k = 0; k < 20; k++) { g.fillStyle = V.alpha(C.muted, 0.25); g.fillRect(x0 + k * cw * 1.9, ry + 12 + r * 18, cw * 1.9 - 2, 14); }
            g.fillStyle = V.alpha(RED(), 0.8); g.fillRect(x0 + 20 * cw * 1.9, ry + 12, cw * 1.9 - 2, 14);
            const shake = Math.sin(t * 20) * 2;
            V.box(g, x0 + shake, ry + 60, Math.min(bw, 380), 32, { r: 8, fill: V.alpha(RED(), 0.18), stroke: RED() });
            V.text(g, 'HTTP 500: got 41 rows for batch_size=2', x0 + 12 + shake, ry + 80, { size: 12, font: 'mono', color: RED(), weight: 700, maxW: Math.min(bw, 380) - 20 });
            V.text(g, 'com o Patch 2 sem o 2b, prefills sem rejeição ainda davam 500', x0, ry + 116, { size: 11, color: C.muted, maxW: bw });
            V.text(g, '(“166 rows for batch_size=3”)', x0, ry + 132, { size: 11, font: 'mono', color: C.muted, maxW: bw });
            if (ry + 186 < h - 14) {
              V.box(g, x0, ry + 150, bw, 36, { r: 7, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.5), lw: 1 });
              V.text(g, 'no overlay atual: 41 % 2 ≠ 0 → o guarda pula a especulação', x0 + 8, ry + 165, { size: 11, color: C.d, maxW: bw - 16 });
              V.text(g, 'neste passo (devolve zeros, sem HTTP 500)', x0 + 8, ry + 180, { size: 11, color: C.d, maxW: bw - 16 });
            }
          } else {
            const qsl = [0, pLen, tot];
            qsl.forEach((q) => { const xx = x0 + q * cw; V.line(g, xx, y - 8, xx, y + th + 8, { color: C.ink, lw: 2 }); V.text(g, String(q), xx, y + th + 22, { size: 11, font: 'mono', align: 'center', color: C.ink }); });
            V.text(g, `query_start_loc = [0, ${pLen}, ${tot}]`, x0, y + th + 42, { size: 12, font: 'mono', color: C.a, weight: 600 });
            // âncoras
            let st0 = 0;
            segs.forEach((sg, i) => {
              const anchor = st0 + Math.max(0, Math.min(sg.len - sg.rej - 1, sg.len - 1));
              const ax = x0 + (anchor + 0.5) * cw;
              tokChar(g, ax, y - 18 + Math.sin(t * 4 + i) * 2, 14, C.d);
              for (let k = sg.len - sg.rej; k < sg.len; k++) { g.strokeStyle = RED(); g.lineWidth = 1.5; g.strokeRect(x0 + (st0 + k) * cw + 1, y + 1, cw - 3, th - 2); }
              st0 += sg.len;
            });
            const ly = y + th + 64;
            const lines = [
              [C.d, 'âncora = start + clamp(len − rejeitados − 1, 0, len − 1)'],
              [C.b, `P: âncora na linha ${pLen - 1} (prefill, sem rejeição)`],
              [C.a, `Q: 3 rejeitadas → âncora na linha ${pLen + 3}`],
              [C.muted, '_store_main_kv_ragged grava cada segmento em pos % 128'],
              [C.muted, 'passos mistos nunca viram CUDA graph → laço Python ok'],
            ];
            lines.forEach(([c, l], i) => V.text(g, l, x0, ly + i * 19, { size: 11.5, color: c, maxW: bw }));
            const gy = ly + lines.length * 19 + 8;
            if (gy + 30 < h - 14) {
              V.box(g, x0, gy, bw, 30, { r: 7, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.5), lw: 1 });
              V.text(g, '41 linhas / 2 hoje: o guarda (41 % 2 ≠ 0) pula a especulação antes do ramo ragged', x0 + 8, gy + 19, { size: 11, color: C.d, maxW: bw - 16 });
            }
          }
          V.text(g, S.fix ? 'divisão 33 + 7 ilustrativa' : 'divisão 34 + 7 ilustrativa', w - 14, h - 8, { size: 10, color: C.muted, align: 'right' });
        }
      });
    })();

    /* ============================================== FIG 10 — validador de k */
    (function () {
      const fig = root.querySelector('#fig-regra-k');
      adaptH(fig, 410, 450);
      const S = { k: 6, ck: 'vx', blockk: false, seqs: 6 };
      const ACC = { vx5: [0.88, 0.74, 0.53, 0.36, 0.24], vx6: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15], o5: [0.93, 0.75, 0.66, 0.58, 0.47] };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'k', min: 1, max: 9, step: 1, value: 6, onInput: (v) => { S.k = v; } });
      UI.seg(ctl, { label: 'Checkpoint', options: [{ v: 'vx', label: 'Vision-Exp (n_predict=3)' }, { v: 'o', label: '0731 (n_predict=1)' }], value: 'vx', onChange: (v) => { S.ck = v; } });
      UI.toggle(ctl, { label: 'hotfix block-k', value: false, onChange: (v) => { S.blockk = v; } });
      UI.slider(ctl, { label: 'MAX_NUM_SEQS', min: 1, max: 8, step: 1, value: 6, onInput: (v) => { S.seqs = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const k = S.k, np = S.ck === 'vx' ? 3 : 1, gamma = 5;
        const acc = S.ck === 'o' ? ACC.o5 : k <= 5 ? ACC.vx5 : ACC.vx6;
        const x0 = 14, bw = w - 28;
        // queries
        V.text(g, `queries do bloco (k = ${k})`, x0, 18, { size: 11.5, color: C.muted });
        const n = Math.max(k, gamma), cw = Math.min(56, (bw - 10) / 9 - 6);
        for (let q = 0; q < n; q++) {
          const x = x0 + q * (cw + 6), y = 28;
          const exists = q < k;
          const beyond = q >= gamma;
          const col = q === 0 ? C.d : beyond ? C.c : C.b;
          if (exists) chip(g, x, y, cw, 26, q === 0 ? 'âncora' : 'ruído', { fill: V.alpha(col, 0.3), stroke: col, size: cw < 44 ? 9 : 10.5, font: 'sans' });
          else { g.save(); g.setLineDash([3, 3]); chip(g, x, y, cw, 26, 'truncado', { fill: 'transparent', stroke: V.alpha(RED(), 0.8), color: RED(), size: cw < 44 ? 9 : 10, font: 'sans' }); g.restore(); }
        }
        const brW = gamma * (cw + 6) - 6;
        V.line(g, x0, 62, x0 + brW, 62, { color: C.b, lw: 2 });
        V.text(g, 'bloco treinado γ = 5', x0 + brW / 2, 76, { size: 10.5, color: C.b, align: 'center' });
        if (k > gamma) V.text(g, 'fora do treino', x0 + gamma * (cw + 6) + ((k - gamma) * (cw + 6)) / 2, 76, { size: 10.5, color: C.c, align: 'center' });
        // barras de aceitação
        const by = 96, bh = narrow ? 62 : 80;
        V.text(g, 'aceitação medida por posição', x0, by, { size: 11, color: C.muted });
        let tau = 1, unknown = false;
        for (let q = 0; q < k; q++) {
          const x = x0 + q * (cw + 6), y = by + 10;
          const v = acc[q];
          V.box(g, x, y, cw, bh, { r: 4, fill: V.alpha(C.muted, 0.07) });
          if (v != null) { tau += v; const hh = bh * v; g.fillStyle = V.alpha(C.a, 0.4 + 0.4 * v + 0.1 * Math.sin(t * 3 + q)); V.rr(g, x, y + bh - hh, cw, hh, 4); g.fill(); V.text(g, fmt(v, 2), x + cw / 2, y + bh - hh - 4, { size: 10, font: 'mono', align: 'center', color: C.ink }); }
          else { unknown = true; V.text(g, '?', x + cw / 2, y + bh / 2 + 5, { size: 14, align: 'center', color: C.muted }); }
        }
        V.text(g, `τ ≈ 1 + Σ = ${fmt(tau, 2)}${unknown ? ' (+ posições sem medição)' : ''}`, x0 + Math.min(k, 9) * (cw + 6) + 6 > w - 160 ? x0 : x0 + Math.min(k, 9) * (cw + 6) + 6, by + 10 + bh + (x0 + Math.min(k, 9) * (cw + 6) + 6 > w - 160 ? 16 : -bh / 2), { size: 11.5, font: 'mono', color: C.ink, maxW: 240 });
        // validador
        const vy = by + bh + 44;
        const r1 = k >= gamma;
        const divOK = S.blockk || k <= np || k % np === 0;
        const launchOK = S.blockk ? k >= 1 : (S.ck === 'vx' ? (k >= 5 && k % 3 === 0) : k >= 5);
        const rules = [
          [r1, r1 ? `k ≥ dspark_block_size (5)` : `k < 5: bloco truncado (silencioso no 0.25.2; rejeitado no vLLM ≥ 0.26)`],
          [divOK, S.blockk ? `divisibilidade: isenta (method = "dspark")` : `divisibilidade MTP: k ≤ ${np} ou k % ${np} = 0 → ${k % np === 0 || k <= np ? 'ok' : 'k % ' + np + ' = ' + (k % np)}`],
          [launchOK, S.blockk ? 'launcher: aceita k ≥ 1 (relaxado pelo hotfix)' : S.ck === 'vx' ? 'launcher: exige k ≥ 5 e k % 3 = 0' : 'launcher da época do 0731: k ≥ 5 (histórico; o atual exige k % 3 = 0)'],
        ];
        rules.forEach(([ok, txt], i) => { check(g, x0 + 8, vy + i * 22 - 4, ok, 6); V.text(g, txt, x0 + 24, vy + i * 22, { size: 11.5, color: ok ? C.ink : RED(), maxW: bw - 30 }); });
        const boots = divOK && launchOK;
        const verdict = !boots ? 'recusa no boot' : !r1 ? 'boota, mas o bloco é truncado' : k === gamma ? 'boota no bloco treinado ✓' : k > gamma ? `boota com ${k - gamma === 1 ? '1 query' : (k - gamma) + ' queries'} além do treino` : 'boota';
        const vcol = !boots ? RED() : !r1 ? C.d : k === gamma ? C.a : C.d;
        const lampY = vy + 3 * 22 + 12;
        V.glow(g, x0 + 10, lampY - 4, 12, vcol, 0.6 + 0.3 * Math.sin(t * 4));
        V.dot(g, x0 + 10, lampY - 4, 5, vcol);
        V.text(g, verdict, x0 + 26, lampY, { size: 13, weight: 700, color: vcol, maxW: bw - 30 });
        // captura
        const tokens = S.seqs * (k + 1), cap = Math.ceil(tokens / 8) * 8;
        V.text(g, `decode: ${S.seqs} seq × (k+1) = ${tokens} tokens → --max-cudagraph-capture-size ${cap}`, x0, lampY + 26, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
        const my = lampY + 54;
        V.text(g, 'medido (Vision-Exp, greedy, 2026-09-03): k=5 com block-k × k=6', x0, my, { size: 11, color: C.muted, maxW: bw });
        [['k=5', 56.6, 56.6], ['k=6', 55.3, 53.6]].forEach(([lb, a8, a32], i) => {
          const yy = my + 10 + i * 22, on = (i === 0 && k === 5) || (i === 1 && k === 6);
          V.text(g, lb, x0, yy + 12, { size: 11.5, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          if (narrow) { V.text(g, `8K: ${fmt(a8, 1)} · 32K: ${fmt(a32, 1)} tok/s`, x0 + 40, yy + 12, { size: 11.5, font: 'mono', color: on ? C.ink : C.muted }); return; }
          const full = Math.max(60, (bw - 150) / 2);
          [[a8, '8K'], [a32, '32K']].forEach(([v, ctx], j) => {
            const bx = x0 + 40 + j * (full + 40), ww = full * ((v - 45) / 15);
            g.fillStyle = V.alpha(i === 0 ? C.a : C.b, on ? 0.85 : 0.35); V.rr(g, bx, yy, Math.max(3, ww), 16, 3); g.fill();
            V.text(g, `${ctx}: ${fmt(v, 1)} tok/s`, bx + ww + 6, yy + 12, { size: 10.5, font: 'mono', color: on ? C.ink : C.muted });
          });
        });
      });
    })();
    /* ============================================== FIG 11 — bancada de regras de parada (TensorFold) */
    (function () {
      const fig = root.querySelector('#fig-bancada');
      if (!fig) return;
      adaptH(fig, 480, 730, 640);
      // tabela de custos medida (draft_sim.py, patch 0011): V(R) para R = 1..8 linhas e o bloco do DFlash2
      const VMS = [29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0], BMS = 3.08, NMAX = 7, NPOS = 240;
      const POLS = ['f7', 'fc', 'fnc', 'fcost', 'oracle'];
      const S = { pol: 'fnc', txt: 'prosa', P: 0.3, seed: 1, r: 0 };
      const pstr = () => S.P.toFixed(2).replace(/0$/, '');
      const pname = (p) => ({ f7: 'f7', fc: 'fc5:' + pstr(), fnc: 'fnc7:' + pstr(), fcost: 'fcost7:noisy', oracle: 'oracle7' }[p]);
      let dump = [];
      const sims = {};
      const lg = (p) => Math.log(p / (1 - p)), sg = (x) => 1 / (1 + Math.exp(-x));
      const cond = (a) => a.map((x, i) => (i ? Math.min(0.97, x / a[i - 1]) : x));
      function makeDump() {
        const r = V.rng(1000 + S.seed * 17 + (S.txt === 'prosa' ? 0 : 5));
        const base = cond(S.txt === 'prosa' ? PROF.glmP.a : PROF.glmS.a);
        let ease = 0;
        dump = [];
        for (let i = 0; i < NPOS; i++) {
          ease = 0.88 * ease + 0.5 * r.normal();          // dificuldade local que muda devagar
          const cn = [], cc = [], m = [];
          for (let d = 0; d < NMAX; d++) {
            const p = V.clamp(sg(lg(V.clamp(base[d], 0.03, 0.97)) + 0.9 * ease), 0.01, 0.99);
            cn.push(V.clamp(sg(lg(p) + 0.35 * r.normal()), 0.01, 0.99));          // ciente do ruído: calibrada
            cc.push(V.clamp(sg(lg(p) + 1.5 * r.normal()), 0.01, 0.99));    // sem ruído: espalhada
            m.push(r() < p);
          }
          let L = 0; while (L < NMAX && m[L]) L++;
          dump.push({ cn, cc, L });
        }
      }
      const room = (i) => Math.max(0, Math.min(NMAX, NPOS - 1 - i));
      function depth(pol, i) {
        const e = dump[i], N = room(i);
        if (N <= 0) return 0;
        if (pol === 'f7') return N;
        if (pol === 'fc' || pol === 'fnc') {
          const cap = pol === 'fc' ? Math.min(5, N) : N, q = pol === 'fc' ? e.cc : e.cn;
          let prod = 1, k = 0;
          for (let d = 0; d < cap; d++) { prod *= q[d]; if (d > 0 && prod < S.P) break; k++; }
          return k;
        }
        let alive = 1, ex = 1, best = 0, bv = ex / (VMS[0] + BMS);    // fcost: argmax E[tokens|k] / (V(k+1) + B)
        for (let k = 1; k <= N; k++) { alive *= e.cn[k - 1]; ex += alive; const v = ex / (VMS[k] + BMS); if (v > bv) { bv = v; best = k; } }
        return best;
      }
      function oracleChoice() {
        const best = new Float64Array(NPOS + 1), ch = new Int8Array(NPOS + 1);
        for (let i = NPOS - 1; i >= 0; i--) {
          best[i] = Infinity;
          for (let k = 0; k <= room(i); k++) {
            const kept = Math.min(k, dump[i].L);
            const tt = VMS[k] + (k > 0 ? BMS : 0) + best[Math.min(i + kept + 1, NPOS)];
            if (tt < best[i]) { best[i] = tt; ch[i] = k; }
          }
        }
        return ch;
      }
      function replay(pol, ch) {
        const rounds = [], own = new Array(NPOS + 1).fill(null);
        let i = 0, ms = 0, rej = 0, rows = 0;
        while (i < NPOS - 1) {
          const k = pol === 'oracle' ? ch[i] : depth(pol, i);
          const kept = Math.min(k, dump[i].L);
          const block = pol === 'oracle' ? k > 0 : room(i) > 0;
          const cost = VMS[k] + (block ? BMS : 0);
          const ri = rounds.length;
          rounds.push({ i, k, kept, cost });
          for (let q = 1; q <= kept; q++) own[i + q] = { r: ri, t: 'a' };
          if (i + kept + 1 <= NPOS) own[i + kept + 1] = { r: ri, t: 's' };
          ms += cost; rej += k - kept; rows += k + 1;
          i += kept + 1;
        }
        return { rounds, own, ms, tokps: ((NPOS - 1) / ms) * 1000, tpr: (NPOS - 1) / rounds.length, rej, rows };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Regra', options: [{ v: 'f7', label: 'f7' }, { v: 'fc', label: 'fc5' }, { v: 'fnc', label: 'fnc7' }, { v: 'fcost', label: 'fcost7' }, { v: 'oracle', label: 'oráculo' }], value: S.pol, onChange: (v) => { S.pol = v; S.r = 0; upd(); } });
      UI.seg(ctl, { label: 'Texto', options: [{ v: 'prosa', label: 'prosa' }, { v: 'estr', label: 'estruturado' }], value: S.txt, onChange: (v) => { S.txt = v; rebuild(); } });
      UI.slider(ctl, { label: 'limiar P (fc, fnc)', min: 0.05, max: 0.9, step: 0.05, value: S.P, fmt: (v) => fmt(v, 2), onInput: (v) => { S.P = v; recompute(); } });
      UI.button(ctl, '🎲 novo registro', () => { S.seed++; rebuild(); });
      const rT = UI.readout(ctl, 'tok/s simulados', '—');
      const rR = UI.readout(ctl, 'tokens/rodada', '—');
      const rW = UI.readout(ctl, 'linhas rejeitadas', '—');
      function upd() {
        const s = sims[S.pol]; if (!s) return;
        rT.set(fmt(s.tokps, 1)); rR.set(fmt(s.tpr, 2)); rW.set(Math.round((100 * s.rej) / s.rows) + '%');
      }
      function recompute() { const ch = oracleChoice(); POLS.forEach((p) => { sims[p] = replay(p, ch); }); S.r = 0; upd(); }
      function rebuild() { makeDump(); recompute(); }
      rebuild();
      V.every(950, () => { const s = sims[S.pol]; if (s) S.r = (S.r + 1) % s.rounds.length; });

      V.fig(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        const sim = sims[S.pol]; if (!sim) return;
        if (Hh.pointer.clicked) S.r = (S.r + 1) % sim.rounds.length;
        const R = sim.rounds[Math.min(S.r, sim.rounds.length - 1)];
        const e = dump[R.i];
        if (narrow) V.text(g, `registro sintético · ${S.txt === 'prosa' ? 'prosa' : 'estruturado'}`, pad, 18, { size: 11.5, color: C.muted, maxW: w * 0.46 });
        else V.text(g, `registro sintético · ${NPOS} posições · ${S.txt === 'prosa' ? 'prosa amostrada' : 'saída estruturada'}`, pad, 18, { size: 12, color: C.muted, maxW: w * 0.58 });
        V.text(g, `rodada ${S.r + 1}/${sim.rounds.length} · ${pname(S.pol)}`, w - pad, 18, { size: 11.5, font: 'mono', color: C.ink, align: 'right', maxW: narrow ? w * 0.5 : w * 0.4 });
        let A, K, T, B;
        if (!narrow) {
          A = { x: pad, y: 32, w: w * 0.56 - pad, h: 176 };
          K = { x: w * 0.56 + 18, y: 32, w: w * 0.44 - 18 - pad, h: 176 };
          T = { x: pad, y: 222, w: w - 2 * pad, h: 104 };
          B = { x: pad, y: 336, w: w - 2 * pad, h: h - 336 - 6 };
        } else {
          A = { x: pad, y: 30, w: w - 2 * pad, h: 170 };
          K = { x: pad, y: 210, w: w - 2 * pad, h: 136 };
          T = { x: pad, y: 356, w: w - 2 * pad, h: 104 };
          B = { x: pad, y: 470, w: w - 2 * pad, h: h - 470 - 6 };
        }
        /* --- A: a cadeia da rodada atual */
        const conf = S.pol === 'fc' ? e.cc : e.cn;
        const ttl = S.pol === 'fc' ? 'cadeia da rodada · confiança sem ruído' : S.pol === 'oracle' ? 'cadeia da rodada · o oráculo já sabe quantos casam' : S.pol === 'f7' ? 'cadeia da rodada · o f7 ignora as confianças' : 'cadeia da rodada · confiança ciente do ruído';
        V.text(g, ttl, A.x, A.y + 12, { size: 12, weight: 600, color: C.ink, maxW: A.w });
        const cw = A.w / NMAX, by0 = A.y + 40, bh = A.h - 92;
        let prod = 1;
        const pts = [];
        for (let d = 0; d < NMAX; d++) {
          const x = A.x + d * cw + 4, bw = cw - 8;
          const avail = d < room(R.i), inK = d < R.k;
          V.box(g, x, by0, bw, bh, { r: 5, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.25), lw: 1 });
          const q = conf[d], fh = Math.max(2, bh * q);
          const col = !avail ? C.muted : inK ? (d < R.kept ? C.a : RED()) : C.b;
          V.box(g, x + 3, by0 + bh - fh, bw - 6, fh, { r: 4, fill: V.alpha(col, inK ? 0.78 : 0.3) });
          V.text(g, fmt(q, 2), x + bw / 2, by0 + bh + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: bw });
          prod *= q; pts.push([x + bw / 2, by0 + bh - bh * prod]);
          const mk = d < e.L ? '✓' : d === e.L ? '✗' : '·';
          V.text(g, 'r' + (d + 1), x + bw / 2 - 4, by0 + bh + 32, { size: 11, font: 'mono', align: 'right', color: C.muted });
          V.text(g, mk, x + bw / 2 + 2, by0 + bh + 32, { size: 12.5, align: 'left', weight: 700, color: d < e.L ? C.a : d === e.L ? RED() : C.muted });
        }
        if (S.pol !== 'f7' && S.pol !== 'oracle') {
          g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke();
          pts.forEach(([x, y]) => V.dot(g, x, y, 3.2, C.d));
        }
        if (S.pol === 'fc' || S.pol === 'fnc') {
          const py = by0 + bh - bh * S.P;
          V.line(g, A.x, py, A.x + A.w, py, { color: V.alpha(C.ink, 0.55), dash: [5, 4], lw: 1.2 });
          V.text(g, 'P', A.x + A.w - 2, py - 4, { size: 11, font: 'mono', align: 'right', color: C.ink });
        }
        const cutX = A.x + R.k * cw;
        V.line(g, cutX, by0 - 6, cutX, by0 + bh + 4, { color: C.ink, lw: 2.2 });
        V.text(g, '✂ k = ' + R.k, V.clamp(cutX + 4, A.x, A.x + A.w - 58), by0 - 9, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
        V.text(g, `✓ casaria · ✗ 1º desacordo · casam ${e.L} de 7`, A.x, A.y + A.h, { size: 11, color: C.muted, maxW: A.w });

        /* --- K: custo de uma rodada (tabela medida) e, no fcost, E[tokens]/ms */
        V.text(g, 'custo da rodada com k rascunhos (ms)', K.x, K.y + 12, { size: 12, weight: 600, color: C.ink, maxW: K.w });
        const kw = K.w / 8, ky0 = K.y + 34, kh = K.h - 66, maxMs = VMS[7] + BMS;
        let alive = 1, ex = 1;
        const ev = [];
        for (let k = 0; k <= NMAX; k++) { if (k) { alive *= e.cn[k - 1]; ex += alive; } ev.push(ex / (VMS[k] + BMS)); }
        const evMax = Math.max(...ev), evMin = Math.min(...ev);
        for (let k = 0; k <= NMAX; k++) {
          const ms = VMS[k] + BMS, h2 = kh * ms / maxMs, x = K.x + k * kw;
          const on = k === R.k;
          V.box(g, x + 3, ky0 + kh - h2, kw - 6, h2, { r: 4, fill: V.alpha(on ? C.a : C.muted, on ? 0.7 : 0.2), stroke: on ? C.a : V.alpha(C.muted, 0.35), lw: 1 });
          if ((S.pol !== 'fcost' && (kw >= 30 || k % 2 === 0)) || on) V.text(g, fmt(ms, 0), x + kw / 2, ky0 + kh - h2 - 4, { size: 11, font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
          V.text(g, String(k), x + kw / 2, ky0 + kh + 15, { size: 11, font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
        }
        if (S.pol === 'fcost') {
          const yv = (v) => ky0 + kh - 8 - (kh - 26) * (evMax > evMin ? (v - evMin) / (evMax - evMin) : 0.5);
          g.beginPath(); ev.forEach((v, k) => { const x = K.x + k * kw + kw / 2; if (k) g.lineTo(x, yv(v)); else g.moveTo(x, yv(v)); }); g.strokeStyle = C.d; g.lineWidth = 2.2; g.stroke();
          ev.forEach((v, k) => V.dot(g, K.x + k * kw + kw / 2, yv(v), k === R.k ? 5 : 2.8, C.d));
        }
        V.text(g, S.pol === 'fcost' ? 'âmbar: E[tokens]/ms; o fcost para no máximo' : 'tabela medida: 29,2…66,0 ms + 3,08 do bloco', K.x, K.y + K.h, { size: 11, color: C.muted, maxW: K.w });

        /* --- T: a fita de tokens cometidos */
        V.text(g, 'tokens cometidos: a mesma fita para toda regra', T.x, T.y + 10, { size: 12, weight: 600, color: C.ink, maxW: narrow ? T.w : T.w * 0.62 });
        const nVis = narrow ? 18 : 36, cell = T.w / nVis;
        const start = V.clamp(R.i - Math.floor(nVis * 0.3), 0, Math.max(0, NPOS - nVis));
        const ty = T.y + 46, th = 24;
        for (let c = 0; c < nVis; c++) {
          const j = start + c; if (j >= NPOS) break;
          const x = T.x + c * cell, o = sim.own[j];
          let fill = V.alpha(C.muted, 0.1), stroke = V.alpha(C.muted, 0.3);
          if (j <= R.i || (o && o.r <= S.r)) {
            const col = o ? (o.t === 'a' ? C.a : C.d) : C.muted;
            const cur = o && o.r === S.r;
            fill = V.alpha(col, cur ? 0.85 : 0.35); stroke = col;
          }
          V.box(g, x + 1, ty, cell - 2, th, { r: 4, fill, stroke, lw: 1 });
          if (j % 10 === 0) V.text(g, String(j), x + cell / 2, ty + th + 26, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        const pc = R.i - start;
        if (pc >= 0 && pc < nVis) { const x = T.x + pc * cell + cell / 2; V.arrow(g, x, ty + th + 13, x, ty + th + 2, { color: C.ink, lw: 1.6, head: 5 }); }
        for (let q = R.kept + 1; q <= R.k; q++) {
          const c = R.i + q - start; if (c < 0 || c >= nVis) continue;
          const x = T.x + c * cell;
          V.box(g, x + 2, ty - 24, cell - 4, 19, { r: 4, fill: V.alpha(RED(), 0.16), stroke: V.alpha(RED(), 0.85), lw: 1 });
          if (cell >= 13) V.text(g, '✗', x + cell / 2, ty - 10, { size: 11, align: 'center', color: RED(), weight: 700 });
        }
        const c0 = R.i + 1 - start, c1 = R.i + R.kept + 1 - start;
        if (c1 >= 0 && c0 < nVis) {
          const x0 = T.x + Math.max(0, c0) * cell + 2, x1 = T.x + Math.min(nVis, c1 + 1) * cell - 2, yb = ty + th + 6;
          g.beginPath(); g.moveTo(x0, yb - 3); g.lineTo(x0, yb); g.lineTo(x1, yb); g.lineTo(x1, yb - 3); g.strokeStyle = C.ink; g.lineWidth = 1.6; g.stroke();
        }
        if (!narrow) {
          const lx = T.x + T.w * 0.64;
          [[C.a, 'rascunho mantido'], [C.d, 'amostra do alvo'], [RED(), 'rejeitado']].forEach(([col, l], i) => {
            const xx = lx + i * (T.w * 0.12);
            V.box(g, xx, T.y + 2, 10, 10, { r: 2, fill: V.alpha(col, 0.7), stroke: col, lw: 1 });
            V.text(g, l, xx + 14, T.y + 11, { size: 11, color: C.muted, maxW: T.w * 0.12 - 18 });
          });
        }

        /* --- B: o mesmo registro, cada regra */
        V.text(g, 'mesmo registro, cada regra (tok/s simulados)', B.x, B.y + 12, { size: 12, weight: 600, color: C.ink, maxW: B.w });
        const maxT = Math.max(...POLS.map((p) => sims[p].tokps)) * 1.04;
        const lw = narrow ? 104 : 118, numW = narrow ? 62 : 236;
        const rh = V.clamp((B.h - 26) / POLS.length - 7, 14, 30);
        const barW = Math.max(40, B.w - lw - numW - 8);
        const xo = B.x + lw + barW * sims.oracle.tokps / maxT;
        V.line(g, xo, B.y + 20, xo, B.y + 24 + POLS.length * (rh + 7) - 4, { color: V.alpha(C.ink, 0.4), dash: [4, 4], lw: 1.2 });
        if (narrow) V.text(g, 'teto', xo, B.y + 24 + POLS.length * (rh + 7) + 10, { size: 11, align: 'center', color: C.muted });
        else V.text(g, 'teto', xo, B.y + 16, { size: 11, align: 'center', color: C.muted });
        POLS.forEach((p, i) => {
          const y = B.y + 24 + i * (rh + 7), s = sims[p], on = p === S.pol;
          V.text(g, pname(p), B.x + lw - 8, y + rh / 2 + 4, { size: 11.5, font: 'mono', align: 'right', color: on ? C.ink : C.muted, weight: on ? 700 : 400, maxW: lw - 10 });
          const bw2 = barW * s.tokps / maxT;
          const col = p === 'oracle' ? C.muted : on ? C.a : C.b;
          V.box(g, B.x + lw, y, bw2, rh, { r: 5, fill: V.alpha(col, on ? 0.75 : 0.32), stroke: on ? col : 'transparent', lw: 1.4 });
          const txt = narrow ? fmt(s.tokps, 1) : `${fmt(s.tokps, 1)} tok/s · ${fmt(s.tpr, 2)} tok/rodada · ${Math.round((100 * s.rej) / s.rows)}% rejeit.`;
          V.text(g, txt, B.x + lw + bw2 + 8, y + rh / 2 + 4, { size: 11, font: 'mono', color: on ? C.ink : C.muted, maxW: numW + barW - bw2 - 4 });
        });
      });
    })();

    /* ============================================== FIG 12 — o anel do DFlash2 no TensorFold */
    (function () {
      const fig = root.querySelector('#fig-anel-tf');
      if (!fig) return;
      adaptH(fig, 470, 628, 640);
      const RING = 2176, WIN = 2048, TILE = 64, BLK = 8, ROWB = 10 * 1024;   // 10 KiB por linha e por rank
      const BACK = WIN - 1;   // a janela de 2.048 conta o pendente: o kernel olha 2.047 posições para trás (window = sliding_window − 1)
      const WINS = [8192, 32768, 131072, 262144, 524288, 1048576];
      const S = { mode: 'one', wi: 5, seg: true, P: 5000, k: 6, kept: 3, t0: null, c0: null, k4: [5, 7, 3, 6], kept4: [2, 7, 0, 4], force: false };
      const rr = V.rng(2176);
      const PH = 1.5;
      function newRound(t) { S.k = 2 + rr.int(6); S.kept = rr.int(S.k + 1); S.t0 = t; }
      function newCycle(t) { S.k4 = [0, 1, 2, 3].map(() => 2 + rr.int(6)); S.kept4 = S.k4.map((k) => rr.int(k + 1)); S.c0 = t; }
      const bytesL = (b) => V.bytes(b).replace('.', ',');
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cena', options: [{ v: 'one', label: 'uma conversa' }, { v: 'four', label: 'quatro conversas' }], value: 'one', onChange: (v) => { S.mode = v; grpOne.show(v === 'one'); grpFour.show(v === 'four'); } });
      const grpOne = UI.group(ctl), grpFour = UI.group(ctl);
      UI.slider(grpOne.el, { label: 'janela por requisição', min: 0, max: WINS.length - 1, step: 1, value: S.wi, fmt: (v) => fmtInt(WINS[v]), onInput: (v) => { S.wi = v; updMem(); } });
      UI.toggle(grpFour.el, { label: 'SEG (fronteira da conv entre blocos)', value: true, onChange: (v) => { S.seg = v; } });
      grpFour.show(false);
      UI.button(ctl, '▶ próxima rodada', () => { S.force = true; }, { primary: true });
      const rF = UI.readout(ctl, 'buffer plano / rank', '—');
      const rA = UI.readout(ctl, 'anel / rank', '—');
      function updMem() { rF.set(bytesL((WINS[S.wi] + BLK) * ROWB), 'bad'); rA.set(bytesL(RING * ROWB), 'ok'); }
      updMem();

      V.fig(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        if (Hh.pointer.clicked) S.force = true;
        const ink2 = C.dark ? '#06140d' : '#ffffff';
        if (S.mode === 'one') {
          if (S.t0 == null) S.t0 = t;
          let ph = Math.floor((t - S.t0) / PH);
          if (ph >= 3 || S.force) { S.P += S.kept + 1; newRound(t); ph = 0; S.force = false; }
          const s = S.P;
          const lo = Math.floor((s - BACK) / TILE) * TILE;
          const classOf = (p) => {   // o que a linha da posição p guarda agora
            if (p > s + BLK - 1) return 'free';
            if (p >= s) return ph < 2 ? 'blk' : p <= s + S.kept ? 'win' : 'old';
            if (p >= s - BACK) return 'win';
            if (p >= lo) return 'tile';
            return 'free';
          };
          /* donut: arcos contínuos por papel, marcas a cada 64 linhas */
          const Rr = narrow ? Math.min(w * 0.3, 108) : Math.min(h * 0.29, 132);
          const cx = narrow ? w / 2 : pad + 14 + Rr, cy = narrow ? 36 + Rr : 42 + Rr;
          const ang = (p) => ((((p % RING) + RING) % RING) / RING) * Math.PI * 2 - Math.PI / 2;
          const arc = (p0, p1, col, al, r0 = 0.66, r1 = 1) => {
            if (p1 <= p0) return;
            const a0 = ang(p0), a1 = a0 + ((p1 - p0) / RING) * Math.PI * 2;
            g.beginPath(); g.arc(cx, cy, Rr * r1, a0, a1); g.arc(cx, cy, Rr * r0, a1, a0, true); g.closePath();
            g.fillStyle = V.alpha(col, al); g.fill();
          };
          g.beginPath(); g.arc(cx, cy, Rr, 0, Math.PI * 2); g.arc(cx, cy, Rr * 0.66, Math.PI * 2, 0, true); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
          arc(lo, s - BACK, C.d, 0.75);
          arc(s - BACK, s, C.a, 0.42);
          arc(s - 256, s, C.a, 0.3);
          if (ph < 2) arc(s, s + BLK, C.b, 0.95, 0.6, 1.14);
          else { arc(s, s + S.kept + 1, C.a, 0.95, 0.6, 1.14); arc(s + S.kept + 1, s + BLK, C.muted, 0.5, 0.6, 1.14); }
          for (let q = 0; q < RING; q += TILE) {
            const a0 = ang(q);
            g.strokeStyle = V.alpha(C.ink, 0.28); g.lineWidth = 1; g.beginPath();
            g.moveTo(cx + Math.cos(a0) * Rr * 0.66, cy + Math.sin(a0) * Rr * 0.66); g.lineTo(cx + Math.cos(a0) * Rr, cy + Math.sin(a0) * Rr); g.stroke();
          }
          const hand = ang(s + 0.5);
          V.glow(g, cx + Math.cos(hand) * Rr * 1.07, cy + Math.sin(hand) * Rr * 1.07, 14, C.b, 0.45 + 0.3 * Math.sin(t * 4));
          V.text(g, '2.176 linhas', cx, cy - 12, { size: 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, 'pendente: pos ' + fmtInt(s), cx, cy + 6, { size: 11, font: 'mono', align: 'center', color: C.muted });
          V.text(g, narrow ? 'slot ' + (s % RING) : 'slot ' + (s % RING) + ' = pos mod 2176', cx, cy + 22, { size: 11, font: 'mono', align: 'center', color: C.muted });
          /* faixa ampliada em volta do pendente */
          const zx = narrow ? pad : cx + Rr + 34, zw = narrow ? w - 2 * pad : w - zx - pad;
          const zy = narrow ? cy + Rr + 44 : 58;
          const nb = narrow ? 4 : 10, nCells = nb + BLK + (narrow ? 2 : 4);
          const cwz = zw / nCells, chz = 30;
          const phases = ['1 · passe de bloco: [pendente, M×7] grava K/V nas linhas s…s+7', `2 · verificação: ${S.k} rascunhos propostos, ${S.kept} mantidos`, `3 · commit: ${S.kept + 1} linhas reescritas das escutas do alvo; o resto vira lixo`];
          V.text(g, phases[ph], zx, zy - 18, { size: 12, weight: 600, color: C.ink, maxW: zw });
          for (let c = 0; c < nCells; c++) {
            const p = s - nb + c, x = zx + c * cwz, cl = classOf(p);
            let label = '', fillA = 0.18, col = C.muted;
            if (cl === 'win') { col = C.a; fillA = p >= s ? 0.8 : 0.45; label = p >= s ? 'esc' : 'ctx'; }
            if (cl === 'blk') { col = C.b; fillA = 0.75; label = p === s ? 'p' : 'M'; }
            if (cl === 'old') { col = C.muted; fillA = 0.22; label = 'lixo'; }
            if (cl === 'free') { col = C.muted; fillA = 0.08; }
            const dim = ph === 1 && p > s + S.k && p < s + BLK;
            V.box(g, x + 1, zy, cwz - 2, chz, { r: 4, fill: V.alpha(col, dim ? 0.2 : fillA), stroke: col, lw: 1 });
            if (cwz >= 16 && label) V.text(g, label, x + cwz / 2, zy + chz / 2 + 4, { size: 11, font: 'mono', align: 'center', color: fillA > 0.6 && !dim ? ink2 : C.ink, maxW: cwz - 2 });
            if (ph === 1 && p > s && p <= s + S.k) { const ok = p - s <= S.kept; check(g, x + cwz / 2, zy + chz + 13, ok, Math.min(6, cwz * 0.25)); }
            if (c % (narrow ? 4 : 5) === 0) V.text(g, String(((p % RING) + RING) % RING), x + cwz / 2, zy + chz + 33, { size: 11, font: 'mono', align: 'center', color: C.muted });
          }
          V.text(g, 'slot no anel', zx, zy + chz + 49, { size: 11, color: C.muted });
          if (ph === 2) {
            const c = nb + S.kept + 1;
            if (c < nCells) { const x = zx + c * cwz + cwz / 2; tokChar(g, x, zy - 34 + Math.sin(t * 5) * 2, 16, C.d); if (!narrow) V.text(g, 'novo pendente', Math.min(x + 12, zx + zw - 92), zy - 30, { size: 11, color: C.d }); }
          }
          const ly = zy + chz + (narrow ? 70 : 76);
          const leg = [[C.a, 'janela de 2.048 (contexto)'], [C.d, 'tile extra da leitura arredondada'], [C.b, 'bloco do passe de rascunho'], [C.muted, 'livre / lixo']];
          leg.forEach(([col, l], i) => {
            const xx = narrow ? zx + (i % 2) * (zw / 2) : zx, yy = narrow ? ly + Math.floor(i / 2) * 18 : ly + i * 19;
            V.box(g, xx, yy - 9, 11, 11, { r: 2, fill: V.alpha(col, 0.6), stroke: col, lw: 1 });
            V.text(g, l, xx + 16, yy, { size: 11, color: C.muted, maxW: narrow ? zw / 2 - 20 : zw - 20 });
          });
          const readY = narrow ? ly + 44 : ly + 4 * 19 + 4;
          const readH = V.wrap(g, 'leitura: de ⌊(s−2047)/64⌋·64 até s+7 → até 2.047 + 63 + 8 = 2.118 linhas; o anel: 34 tiles × 64 = 2.176', zx, readY, zw, { size: 11, font: 'mono', color: C.ink, lineH: 15 });
          /* memória: buffer plano × anel */
          const W = WINS[S.wi], flat = (W + BLK) * ROWB, ring = RING * ROWB;
          const my = narrow ? readY + readH + 16 : Math.max(cy + Rr + 30, readY + readH + 16);
          const mx = pad, mw = w - 2 * pad;
          V.text(g, `contexto do rascunhador por rank · janela de ${fmtInt(W)} tokens`, mx, my, { size: 12, weight: 600, color: C.ink, maxW: mw });
          const rows = [['buffer plano · 1 conversa', flat, RED()], ['buffer plano · 4 conversas', 4 * flat, RED()], ['anel · 4 conversas', 4 * ring, C.a]];
          const maxB = 4 * ((WINS[WINS.length - 1] + BLK) * ROWB);
          const labW = narrow ? 0 : 180, bh2 = narrow ? 13 : 15, gap = narrow ? 32 : 22;
          rows.forEach(([l, b, col], i) => {
            const y = my + 12 + i * gap;
            if (narrow) V.text(g, l, mx, y + 9, { size: 11, color: C.muted });
            else V.text(g, l, mx + labW - 8, y + bh2 - 3, { size: 11, color: C.muted, align: 'right' });
            const by = narrow ? y + 14 : y, avail = mw - labW - 84;
            const bw = Math.max(2, avail * (b / maxB));
            V.box(g, mx + labW, by, bw, bh2, { r: 4, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
            V.text(g, bytesL(b), mx + labW + bw + 6, by + bh2 - 3, { size: 11, font: 'mono', color: C.ink });
          });
        } else {
          if (S.c0 == null) S.c0 = t;
          let ph = Math.floor((t - S.c0) / PH);
          if (ph >= 3 || S.force) { newCycle(t); ph = 0; S.force = false; }
          const cols = [C.a, C.b, C.c, C.d];
          const phases = ['1 · um passe de bloco para as quatro conversas', '2 · seletor e top-k por conversa', '3 · commit empilhado; o enchimento vai para o lixo'];
          V.text(g, phases[ph], pad, 18, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
          /* pool de anéis */
          const px = pad, pw = w - 2 * pad, py = 50, phh = 32;
          const trashW = pw * 0.1, slotW = (pw - trashW) / 4;
          V.text(g, 'pool por camada: [kv_heads, 4·cap + lixo, head_dim]', px, py - 8, { size: 11, font: 'mono', color: C.muted, maxW: pw });
          for (let i = 0; i < 4; i++) {
            const x = px + i * slotW;
            V.box(g, x + 1, py, slotW - 2, phh, { r: 6, fill: V.alpha(cols[i], 0.2), stroke: cols[i], lw: 1.2 });
            V.text(g, narrow ? 'slot ' + i : 'conversa ' + (i + 1) + ' · slot ' + i, x + slotW / 2, py + phh / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: slotW - 6 });
            const hx = x + 6 + ((t * (0.04 + i * 0.013)) % 1) * (slotW - 14);
            g.fillStyle = cols[i]; g.fillRect(hx, py + phh - 5, 3, 4);
          }
          const tx = px + 4 * slotW;
          V.box(g, tx + 1, py, trashW - 2, phh, { r: 6, fill: V.alpha(C.muted, 0.15), stroke: V.alpha(C.muted, 0.6), lw: 1 });
          g.save(); V.rr(g, tx + 1, py, trashW - 2, phh, 6); g.clip(); g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 1;
          for (let k = -phh; k < trashW; k += 7) { g.beginPath(); g.moveTo(tx + k, py + phh); g.lineTo(tx + k + phh, py); g.stroke(); }
          g.restore();
          V.text(g, 'lixo', tx + trashW / 2, py + phh / 2 + 4, { size: 11, align: 'center', color: C.ink, weight: 700 });
          /* passe de bloco: 4 × [p, M×7] */
          const bx = pad, bw = w - 2 * pad, byy = py + phh + V.clamp(h * 0.15, 58, 92), cwb = bw / 32, chb = 30;
          const a0 = ph === 0 ? 1 : 0.35;
          for (let i = 0; i < 4; i++) {
            for (let r = 0; r < BLK; r++) {
              const x = bx + (i * BLK + r) * cwb;
              V.box(g, x + 1, byy, cwb - 2, chb, { r: 3, fill: V.alpha(cols[i], (r === 0 ? 0.85 : 0.5) * a0), stroke: V.alpha(cols[i], a0), lw: 1 });
              if (cwb >= 16) V.text(g, r === 0 ? 'p' : 'M', x + cwb / 2, byy + chb / 2 + 4, { size: 11, font: 'mono', align: 'center', color: r === 0 && a0 > 0.5 ? ink2 : C.ink, alpha: Math.max(0.5, a0) });
              if (r > 0 && cwb >= 9) { const xa = x - cwb / 2 + 2, xb = x + cwb / 2 - 2; g.globalAlpha = a0; V.arrow(g, xa, byy - 5, xb, byy - 5, { color: cols[i], lw: 1.1, head: 3.5 }); g.globalAlpha = 1; }
            }
            const mxb = bx + (i * BLK + BLK / 2) * cwb, sx = px + i * slotW + slotW / 2;
            g.globalAlpha = Math.max(0.3, a0); V.arrow(g, mxb, byy - 12, sx, py + phh + 4, { color: cols[i], lw: 1.5, head: 6, dash: [5, 4] }); g.globalAlpha = 1;
            if (i > 0) {
              const xb = bx + i * BLK * cwb;
              if (S.seg) V.line(g, xb, byy - 10, xb, byy + chb + 4, { color: C.ink, lw: 2.4 });
              else { V.arrow(g, xb - cwb / 2, byy - 5, xb + cwb / 2, byy - 5, { color: RED(), lw: 2.2, head: 5 }); V.text(g, '⚡', xb, byy - 10, { size: 13, align: 'center', color: RED() }); }
            }
          }
          const wr = (txt, y, o) => V.wrap(g, txt, bx, y, bw, Object.assign({ size: 11, color: C.muted, lineH: 15 }, o));
          let y = byy + chb + 18;
          y += wr(S.seg ? 'SEG: cada bloco começa sem predecessor, como sozinho' : 'sem SEG: a linha p do bloco 2 misturaria a M7 do bloco 1 → bits ≠ solo', y, { size: 11.5, weight: 600, color: S.seg ? C.a : RED() });
          y += wr('atenção: cada segmento lê o próprio slot por uma tabela (slot, posição), com o laço do kernel solo', y);
          /* seletor por conversa */
          const sy = y + V.clamp(h * 0.03, 8, 20), sh = 30, a1 = ph === 1 ? 1 : 0.35;
          const sw = (bw - 3 * 8) / 4;
          for (let i = 0; i < 4; i++) {
            const x = bx + i * (sw + 8);
            V.box(g, x, sy, sw, sh, { r: 6, fill: V.alpha(cols[i], 0.16 * a1 + 0.04), stroke: V.alpha(cols[i], Math.max(0.4, a1)), lw: 1.3 });
            V.text(g, narrow ? 'top-k' : 'seletor + top-k → ' + S.k4[i] + ' rasc.', x + sw / 2, sy + sh / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink, alpha: Math.max(0.55, a1), maxW: sw - 6 });
          }
          y = sy + sh + 16;
          y += wr('por conversa: cuBLAS e torch.topk podem escolher o kernel pelo nº de linhas', y);
          /* commit empilhado */
          const a2 = ph === 2 ? 1 : 0.35;
          const kept = S.kept4.map((k) => k + 1), tot = kept.reduce((x, yy) => x + yy, 0), tmax = Math.max(BLK, Math.ceil(tot / BLK) * BLK);
          y += V.clamp(h * 0.03, 8, 22);
          y += wr(`commit: ${tot} linhas mantidas → lançamento de ${tmax} (múltiplo do bloco)`, y, { size: 11.5, color: C.ink, alpha: Math.max(0.6, a2) });
          const cy2 = y - 6;
          const cwc = Math.min(narrow ? 12 : 22, bw / tmax), chc = 24;
          let c = 0;
          kept.forEach((n, i) => { for (let q = 0; q < n; q++, c++) V.box(g, bx + c * cwc + 1, cy2, cwc - 2, chc, { r: 3, fill: V.alpha(cols[i], 0.7 * a2 + 0.1), stroke: V.alpha(cols[i], Math.max(0.4, a2)), lw: 1 }); });
          for (; c < tmax; c++) V.box(g, bx + c * cwc + 1, cy2, cwc - 2, chc, { r: 3, fill: V.alpha(C.muted, 0.18), stroke: V.alpha(C.muted, 0.6 * a2 + 0.2), lw: 1 });
          if (tot < tmax && bx + bw - (bx + tmax * cwc) >= 150) {
            const xe = bx + tmax * cwc;
            V.arrow(g, xe + 6, cy2 + chc / 2, xe + 30, cy2 + chc / 2, { color: C.muted, lw: 1.4, head: 6 });
            V.text(g, 'enchimento → linhas lixo', xe + 36, cy2 + chc / 2 + 4, { size: 11, color: C.muted, maxW: Math.max(40, bx + bw - xe - 40) });
          }
          y = cy2 + chc + 18;
          y += wr('cada linha leva sua posição e seu destino: slot·cap + (pos mod 2176)' + (narrow ? '; o enchimento vai para as linhas lixo' : ''), y);
          y += wr('graphs: um passe de bloco por nº de conversas (1–4), um por faixa de linhas na atualização', y);
          wr('invariante: os candidatos de cada conversa têm os bits do seu rascunhador solo', Math.min(h - 26, y + V.clamp(h * 0.04, 8, 24)), { size: 12, weight: 600, color: C.a, lineH: 16 });
        }
      });
    })();
  },
});

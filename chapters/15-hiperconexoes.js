LA.chapter({
  id: 'hiperconexoes',
  part: 3,
  order: 15,
  title: 'Hyper-connections: alargando o fluxo residual',
  navTitle: 'Hyper-connections',
  subtitle: 'Quatro fluxos residuais por token: ler, escrever e misturar com matrizes aprendidas, domar a mistura com Sinkhorn, ler com lupa no Gated Residual — e o que isso custa em bytes, kernels e hooks de rascunhador.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['hyper-connections', 'mHC', 'Sinkhorn-Knopp', 'duplamente estocástica', 'Gated Residual', 'GatedNorm', 'fluxo residual', 'b12x', 'EAGLE-3', 'DSpark', 'FP8-dense', 'TensorFold'],
  goals: [
    'Explicar por que um fluxo residual único dilui o que foi escrito cedo e como <b>n fluxos</b> com leitura (<code>pre</code>), escrita (<code>post</code>) e mistura (<code>comb</code>) generalizam o resíduo',
    'Mostrar por que a mistura livre explode com a profundidade e como <b>Sinkhorn-Knopp</b> (20 iterações) a projeta em matrizes duplamente estocásticas de norma ≤ 1',
    'Comparar o <b>mHC</b> do GLM/DeepSeek com o <b>Gated Residual</b> do Qwen (gate elementwise de posto 320, escrita escalar, sem mistura)',
    'Calcular o custo do resíduo largo (buffers de 256 MiB, 1,19 GiB replicadas, kernels b12x/TileLang) e escrever um hook que <b>materializa <code>hc_post</code> e contrai</b> 4 fluxos para 4096',
    'Entender por que a cola das hyper-connections pode ser <b>dividida por linhas</b> entre dois ranks sem mudar um bit, e o que isso rende no prefill da receita GLM·TF',
  ],
  prereqs: ['bloco-transformer'],
  summary: [
    'Com um resíduo único todo bloco lê a mesma soma normalizada: o que foi escrito cedo é diluído (fatia \\(\\approx 1/(v+1)\\)) e cada escritor tem uma só taxa de decaimento. Hyper-connections trocam \\(h \\in \\mathbb R^d\\) por \\(H \\in \\mathbb R^{n\\times d}\\) (<code>hc_mult</code>/<code>hc_count</code> = 4).',
    'Cada sub-bloco lê \\(x = \\sum_i \\text{pre}_i H_i\\), escreve \\(\\text{post} \\otimes y\\) e mistura \\(\\text{comb}^{\\top} H\\); com \\(n = 1\\) volta o resíduo clássico. Os coeficientes são dinâmicos: previstos por token a partir do próprio estado.',
    'Misturas livres se compõem por 86–90 sub-blocos e explodem (ganho ~3.000 no paper do mHC). Matrizes duplamente estocásticas têm \\(\\lVert A\\rVert_2 \\le 1\\), são fechadas sob produto e preservam a média; Sinkhorn-Knopp (<code>hc_sinkhorn_iters</code> = 20, <code>hc_eps</code> = 1e-6) as produz alternando normalizações.',
    'No mHC do GLM/DeepSeek, \\(W_{fn} \\in \\mathbb R^{24\\times16384}\\) gera 24 números por token (4 pre + 4 post + 16 comb), 786.486 parâmetros por camada; o post de cada sub-bloco é fundido com o pre do seguinte, e cada camada devolve <code>(hidden, residual, post, comb)</code>.',
    'O Gated Residual do Qwen lê com um gate elementwise \\(\\sigma(W_u\\,\\mathrm{SiLU}(W_d\\,\\mathrm{vec}(\\hat R)/4))\\) de posto 320 (uma GatedNorm no resíduo largo), escreve com um escalar por ramo e elimina \\(H_{\\text{res}}\\); substitui a pré-norma e, com o <code>hyper_connection_mixer</code>, a norma final.',
    'O preço: estado 4× maior (256 MiB a 8.192 tokens no DeepSeek, o <code>_mtp_hidden_buffer</code> que o hotfix #50312 tenta não alocar), 1,19 GiB BF16 replicadas no Qwen (13% do passo, alvo do overlay FP8-dense com <code>quant_config</code> propagado) e kernels especializados (b12x mHC só com \\(n=4\\), hidden 4096, ≤ 16 tokens). Em dois ranks a cola é replicada; no prefill, a receita GLM·TF a divide por linhas (12 KiB enviados por linha em vez de 16, metade da cola, mesmos bits): 50k de ~1.270 para ~1.730 tok/s, medido junto com dois patches de kernels.',
    'Hooks de rascunhador precisam materializar o post adiado (<code>layer.hc_post</code>) e contrair como no treino (média): GLM nos taps 6, 15, 25, 34, 43; DeepSeek nas camadas 40–42 (no overlay Stage-C, 40 e 41 podem ser capturadas pela camada seguinte sem quebrar a fusão, com a média exata via <code>dspark_hc_post_mean</code>).',
  ],
  quiz: [
    { q: 'Por que uma mistura <code>comb</code> duplamente estocástica não pode amplificar o sinal residual, mesmo depois de 90 sub-blocos?', o: ['Porque suas entradas são menores que 1', 'Porque \\(\\lVert A\\rVert_1 = \\lVert A\\rVert_\\infty = 1\\) dá \\(\\lVert A\\rVert_2 \\le 1\\), e o produto de duplamente estocásticas continua duplamente estocástico', 'Porque o softmax por linha já garante isso', 'Porque o Sinkhorn zera os elementos fora da diagonal'], a: 1, e: 'Somas de linhas e colunas iguais a 1 limitam a norma espectral a 1 (\\(\\lVert A\\rVert_2 \\le \\sqrt{\\lVert A\\rVert_1\\lVert A\\rVert_\\infty}\\)), e a classe é fechada sob multiplicação, então vale para qualquer profundidade. Só o softmax por linha não basta: uma coluna pode somar mais que 1. O Sinkhorn não zera nada, só reescala linhas e colunas.' },
    { q: 'No GLM, por que o <code>patch_glm_eagle3.py</code> chama <code>layer.hc_post(hidden, residual, post, comb)</code> antes de <code>hc_contract</code>?', o: ['Porque o <code>hc_post</code> quantiza o estado para FP8', 'Porque cada camada devolve a saída crua e os coeficientes, com o post adiado para ser fundido ao pre da camada seguinte, então o estado da camada não existe em memória', 'Porque o DFlash2 precisa dos 4 fluxos separados', 'Porque o <code>hc_contract</code> só aceita tensores 2D'], a: 1, e: 'Com a fusão post+pre (<code>MHCFusedPostPreOp</code>), a camada ℓ devolve <code>(hidden, residual, post, comb)</code> e quem aplica o post é a ℓ+1. O hook materializa o post numa cópia à parte e depois tira a média dos 4 fluxos, entregando <code>[T, 4096]</code>, que é o que o DFlash2 espera (e não os 4 fluxos).' },
    { q: 'Qual é a diferença essencial entre o Gated Residual do Qwen e o mHC do GLM/DeepSeek?', o: ['O GR usa 8 ramos e o mHC 4', 'O GR lê com um gate por ramo e por canal (posto 320), escreve com um escalar por ramo e não tem matriz de mistura; o mHC lê com um escalar por fluxo e mistura com uma 4×4 duplamente estocástica', 'O GR usa Sinkhorn com mais iterações', 'O mHC substitui a pré-normalização e o GR não'], a: 1, e: 'A ablação do Qwen mostrou que a granularidade da leitura importa mais que a da escrita e que \\(H_{\\text{res}}\\) quase não acrescenta quando leitura e escrita são expressivas. Os dois usam 4 ramos. Quem substitui a pré-norma é o GR (sua leitura já normaliza e aplica gate); o mHC continua com <code>input_layernorm</code>/<code>attn_norm</code> depois do pre.' },
    { q: 'As HyperConnections do Qwen são ~0,5% dos parâmetros, mas ~13% dos bytes lidos por passo de decode em cada GPU. Por quê?', o: ['Porque são lidas duas vezes por token', 'Porque são BF16, lidas inteiras a cada passo e replicadas nas duas GPUs (<code>disable_tp=True</code>), enquanto os experts são NVFP4 e poucos são lidos: top-10 por token, ~32 distintos por camada num passo de verificação de 4 tokens, divididos entre as GPUs por EP', 'Porque o PLE as copia para a CPU', 'Porque o MTP as recalcula 3 vezes'], a: 1, e: '1,19 GiB BF16 sem fatiamento: cada GPU paga a banda inteira. Os experts somam 120,8 B parâmetros, mas em 4 bits, com top-10 por token (cerca de 32 distintos por camada num passo MTP3) e divididos por EP. Como os lineares do GR tinham <code>quant_config=None</code> fixo, nem um checkpoint quantizado os mudava. Daí o overlay <code>hyperconnection.diff</code> do FP8-dense.' },
    { q: 'Na imagem Stage-C do DeepSeek, com <code>VLLM_USE_B12X_MHC=1</code> e <code>B12X_MHC_MAX_TOKENS=16</code>, qual kernel de mHC atende um passo de decode com 1 sequência e rascunho k = 6, e qual atende um chunk de prefill de 8.192 tokens?', o: ['b12x nos dois', 'b12x no decode (7 linhas ≤ 16) e TileLang no prefill', 'TileLang no decode e b12x no prefill', 'TileLang nos dois, porque o b12x exige hidden 7168'], a: 1, e: 'É um despacho por M: até 16 tokens o b12x (afinado para passos pequenos), acima o TileLang (<code>MHCFusedPostPreOp</code>). O b12x ainda exige família SM120, <code>hc_mult == MHC_MULT</code>, hidden 4096 e split-K fixo, condições que o DeepSeek-V4-Flash cumpre.' },
    { q: 'De onde vêm os 256 MiB do <code>_mtp_hidden_buffer</code> do DeepSeek, e por que ele é inútil nessa receita?', o: ['8192 × 4096 × 8 B; é inútil porque o MTP está desligado', '<code>max_num_batched_tokens</code> × <code>hc_dim</code> × 2 B = 8192 × 16384 × 2 B; o DSpark não o lê, e usa as médias das camadas 40–42 no <code>_dspark_hidden_buffer</code>', '8192 × 12288 × 2 B; é inútil porque o DSpark usa FP8', 'É o KV cache do rascunhador; é inútil porque o DSpark não tem KV'], a: 1, e: 'O buffer guarda o estado largo pré-<code>hc_head</code> (4 × 4096 por token, BF16) para MTP/EAGLE, com um <code>copy_</code> a cada passo. O DSpark lê features médias num buffer próprio de 3 × 4096 por token. O hotfix #50312 só aloca o buffer MTP quando <code>use_eagle()</code> ou <code>uses_draft_model()</code> pedem, com a ressalva de que <code>use_eagle()</code> pode incluir dspark.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ================================================================ helpers */
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const silu = (x) => x * sig(x);
    const fmt = (x, d = 2) => { let s = (+x).toFixed(d); if (s.charAt(0) === '-' && +s === 0) s = s.slice(1); s = s.replace('.', ','); return s.charAt(0) === '-' ? '−' + s.slice(1) : s; };
    const onInk = () => (V.C().dark ? '#06140d' : '#ffffff');
    const SC = (C) => [C.a, C.b, C.c, C.d];
    function adaptH(fig, wide, narrow, bp = 620, sel = '.la-fig-stage') {
      const st = fig.querySelector(sel);
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function fitSize(g, s, maxW, size, font = 'sans', weight = 400, min = 11) { let z = size; while (z > min && tw(g, s, z, font, weight) > maxW) z -= 0.5; return z; }
    function txt(g, s, x, y, o = {}) {
      const size = o.maxW ? fitSize(g, s, o.maxW, o.size || 12, o.font || 'sans', o.weight || 400) : (o.size || 12);
      V.text(g, s, x, y, { size, font: o.font, weight: o.weight, color: o.color, align: o.align, base: o.base, alpha: o.alpha, maxW: o.maxW });
    }
    function wrapText(g, s, x, y, maxW, lh, o = {}) {
      const words = String(s).split(' ');
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      let line = '', yy = y, n = 0;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); yy += lh; n++; line = wd; g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); } else line = test;
      }
      if (line) { V.text(g, line, x, yy, o); n++; }
      return n;
    }
    const I4 = () => [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
    const tr = (A) => A[0].map((_, j) => A.map((r) => r[j]));
    function softmaxRows(Z) { return Z.map((r) => { const m = Math.max(...r); const e = r.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }); }
    function colNorm(M, eps) { const cs = M[0].map((_, j) => M.reduce((a, r) => a + r[j], 0)); return M.map((r) => r.map((v, j) => v / (cs[j] + eps))); }
    function rowNorm(M, eps) { return M.map((r) => { const s = r.reduce((a, b) => a + b, 0); return r.map((v) => v / (s + eps)); }); }
    /** sequência exata do kernel de referência: softmax+eps, colunas, (linhas, colunas)×(iters−1) */
    function sinkhornSeq(Z, iters = 20, eps = 1e-6) {
      const seq = [];
      let M = softmaxRows(Z).map((r) => r.map((v) => v + eps));
      seq.push(M); M = colNorm(M, eps); seq.push(M);
      for (let k = 0; k < iters - 1; k++) { M = rowNorm(M, eps); seq.push(M); M = colNorm(M, eps); seq.push(M); }
      return seq;
    }
    const sinkhorn = (Z, iters = 20, eps = 1e-6) => { const s = sinkhornSeq(Z, iters, eps); return s[s.length - 1]; };
    const rowSums = (M) => M.map((r) => r.reduce((a, b) => a + b, 0));
    const colSums = (M) => M[0].map((_, j) => M.reduce((a, r) => a + r[j], 0));
    /** autovalores de matriz simétrica pequena (Jacobi) */
    function symEig(A0) {
      const n = A0.length, a = A0.map((r) => r.slice());
      for (let sweep = 0; sweep < 60; sweep++) {
        let off = 0, dia = 0;
        for (let i = 0; i < n; i++) { dia += a[i][i] * a[i][i]; for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j]; }
        if (off <= 1e-26 * (dia + 1e-300)) break;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
          for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        }
      }
      return a.map((r, i) => r[i]);
    }
    /** maior valor singular: autovalor dominante de QᵀQ, com Q = P/max|P| para não estourar */
    function sMax(P) {
      let m = 0; P.forEach((r) => r.forEach((v) => { if (Math.abs(v) > m) m = Math.abs(v); }));
      if (!(m > 0) || !isFinite(m)) return m;
      const Q = P.map((r) => r.map((v) => v / m));
      return m * Math.sqrt(Math.max(0, ...symEig(mm(tr(Q), Q))));
    }
    /** inversa 4×4 por Gauss-Jordan com pivoteamento parcial */
    function inv4(A) {
      const n = A.length, M = A.map((r, i) => r.concat(I4()[i]));
      for (let c = 0; c < n; c++) {
        let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
        if (Math.abs(M[p][c]) < 1e-300) return null;
        const tmp = M[c]; M[c] = M[p]; M[p] = tmp;
        const d = M[c][c]; for (let k = 0; k < 2 * n; k++) M[c][k] /= d;
        for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let k = 0; k < 2 * n; k++) M[r][k] -= f * M[c][k]; }
      }
      return M.map((r) => r.slice(n));
    }
    /** personagem-token (caixa com olhinhos) */
    function tokenGuy(g, x, y, s, col, label) {
      const C = V.C();
      V.box(g, x - s, y - s, 2 * s, 2 * s, { r: s * 0.45, fill: V.alpha(col, 0.22), stroke: col, lw: 1.6 });
      V.dot(g, x - s * 0.35, y - s * 0.2, Math.max(1.4, s * 0.12), C.ink);
      V.dot(g, x + s * 0.35, y - s * 0.2, Math.max(1.4, s * 0.12), C.ink);
      g.beginPath(); g.arc(x, y + s * 0.15, s * 0.3, 0.15 * Math.PI, 0.85 * Math.PI); g.strokeStyle = C.ink; g.lineWidth = 1.2; g.stroke();
      if (label) txt(g, label, x, y - s - 6, { size: 11, align: 'center', color: C.muted, font: 'mono' });
    }
    /** curva cúbica suave entre dois pontos ao longo de um eixo (horizontal ou vertical) */
    function flowCurve(g, a, b, horiz, o) {
      g.beginPath(); g.moveTo(a[0], a[1]);
      if (horiz) { const mx = (a[0] + b[0]) / 2; g.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]); } else { const my = (a[1] + b[1]) / 2; g.bezierCurveTo(a[0], my, b[0], my, b[0], b[1]); }
      g.strokeStyle = o.color; g.lineWidth = o.lw || 1.5; g.globalAlpha = o.alpha == null ? 1 : o.alpha;
      if (o.dash) g.setLineDash(o.dash);
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
    function curvePoint(a, b, horiz, s) {
      if (horiz) { const mx = (a[0] + b[0]) / 2; return V.bez(a, [mx, a[1]], [mx, b[1]], b, s); }
      const my = (a[1] + b[1]) / 2; return V.bez(a, [a[0], my], [b[0], my], b, s);
    }

    /* ============================================ Fig 1: uma faixa × quatro */
    (function figFaixa() {
      const fig = root.querySelector('#fig-faixa'); if (!fig) return;
      adaptH(fig, 430, 470);
      const L = 16;
      const S = { lanes: 1, v: 12, ug: false };
      /* pesos escolhidos à mão para imitar o padrão do tech report (um ramo longo, três locais) */
      const P = { l0: 2.2, l0o: 0.35, late0: 0.03, side: 0.3, g: [1, 0.25, 0.25, 0.25], gu: [1, 1, 1, 1] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Estrada', options: [{ v: 1, label: '1 faixa' }, { v: 4, label: '4 faixas' }], value: 1, onChange: (v) => { S.lanes = v; } });
      UI.slider(ctl, { label: 'Camada que lê', min: 1, max: L, step: 1, value: 12, fmt: (v) => 'v = ' + v, onInput: (v) => { S.v = v; } });
      UI.toggle(ctl, { label: 'gate uniforme (1 · 1 · 1 · 1)', value: false, onChange: (v) => { S.ug = v; } });
      const ro = UI.readout(ctl, 'fatia da camada 0', '—');
      const wr = (c, u) => {
        if (u === -1) return 1;
        if (u === 0) return c === 0 ? P.l0 : P.l0o;
        if (c === 0) return P.late0;
        return c === 1 + ((u - 1) % 3) ? 1 : P.side;
      };
      function shares(v, lanes) {
        const W = [-1]; for (let u = 0; u < v; u++) W.push(u);
        if (lanes === 1) return W.map(() => [1 / W.length]);
        const rms = [0, 1, 2, 3].map((c) => Math.sqrt(W.reduce((a, u) => a + wr(c, u) ** 2, 0)));
        const gt = S.ug ? P.gu : P.g;
        const a = W.map((u) => [0, 1, 2, 3].map((c) => 0.25 * gt[c] * wr(c, u) / rms[c]));
        const tot = a.reduce((x, r) => x + r.reduce((p, q) => p + q, 0), 0);
        return a.map((r) => r.map((x) => x / tot));
      }
      let last = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const cols = SC(C);
        const narrow = w < 560;
        const x0 = narrow ? 22 : 36, x1 = w - (narrow ? 14 : 30);
        const v = S.v, lanes = S.lanes;
        const sh = shares(v, lanes);
        const s0 = sh[1].reduce((a, b) => a + b, 0);
        const lab = (s0 * 100).toFixed(1).replace('.', ',') + '%';
        if (lab !== last) { ro.set(lab); last = lab; }
        // título
        txt(g, lanes === 1 ? 'uma faixa: h ∈ ℝᵈ, todo escritor soma no mesmo vetor' : 'quatro faixas: H ∈ ℝ⁴ˣᵈ, cada escritor escolhe onde deposita', x0, 20, { size: narrow ? 11.5 : 13, weight: 600, color: C.ink, maxW: x1 - x0 });
        // estrada
        const rTop = 52, rH = narrow ? 58 : 64;
        const laneH = lanes === 1 ? rH : (rH - 3 * 4) / 4;
        const laneY = (c) => rTop + (lanes === 1 ? 0 : c * (laneH + 4));
        const sx = (u) => x0 + ((u + 1) / (L + 1)) * (x1 - x0); // u=-1 → x0 (embedding)
        for (let c = 0; c < lanes; c++) {
          V.box(g, x0 - 6, laneY(c), x1 - x0 + 12, laneH, { r: Math.min(8, laneH / 2), fill: V.alpha(lanes === 1 ? C.muted : cols[c], lanes === 1 ? 0.12 : 0.14), stroke: V.alpha(lanes === 1 ? C.muted : cols[c], 0.35), lw: 1 });
          V.line(g, x0, laneY(c) + laneH / 2, x1, laneY(c) + laneH / 2, { color: V.alpha(C.ink, 0.18), lw: 1, dash: [6, 8], dashOffset: -t * 18 });
        }
        // estações
        const period = 4.2, f = (t % period) / period;
        const xr = sx(v - 0.5 < L ? v - 0.5 : L - 0.5);
        const tokX = V.lerp(x0, xr, V.ease.inOut(Math.min(1, f / 0.8)));
        const every = (x1 - x0) / (L + 1) < 22 ? 2 : 1;
        for (let u = 0; u < L; u++) {
          const x = sx(u), passed = u < v;
          V.dot(g, x, rTop - 8, 3.2, passed ? C.ink : V.alpha(C.muted, 0.5));
          if (u % every === 0 || u === v - 1) txt(g, String(u), x, rTop - 15, { size: 11, align: 'center', color: passed ? C.muted : V.alpha(C.muted, 0.5), font: 'mono' });
          if (!passed) continue;
          const vis = tokX >= x - 2 ? 1 : 0.45;
          for (let c = 0; c < lanes; c++) {
            const amt = lanes === 1 ? 1 : wr(c, u);
            if (amt < 0.02) continue;
            const sz = Math.min(laneH - 4, (lanes === 1 ? 12 : 3 + 7 * Math.min(1.4, amt)));
            g.globalAlpha = vis * (lanes === 1 ? 0.9 : Math.min(1, 0.35 + 0.5 * amt));
            V.box(g, x - sz / 2, laneY(c) + laneH / 2 - sz / 2, sz, sz, { r: 2.5, fill: u === 0 ? C.c : (lanes === 1 ? C.a : cols[c]) });
            g.globalAlpha = 1;
          }
        }
        // embedding no começo
        txt(g, 'e', x0 - 1, rTop - 15, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        // leitor
        V.line(g, xr, rTop - 4, xr, rTop + rH + 6, { color: C.c, lw: 2, dash: [3, 3] });
        txt(g, 'leitor: camada ' + v, Math.min(xr, x1 - 40), rTop + rH + 20, { size: 11.5, align: 'center', color: C.c, weight: 600 });
        tokenGuy(g, tokX, rTop + rH + (narrow ? 44 : 46), narrow ? 11 : 13, C.c);
        // barras de fatia
        const bTop = rTop + rH + (narrow ? 82 : 86), bBot = h - 30;
        const bh = bBot - bTop;
        txt(g, 'de onde vem o que a camada ' + v + ' lê (fatia de cada escritor)', x0, bTop - 6, { size: 11.5, color: C.muted, maxW: x1 - x0 });
        const n = sh.length, gap = n > 12 ? 3 : 5;
        const ax = x0 + 30;
        const bw = Math.min(40, (x1 - ax - gap * (n - 1)) / n);
        const totalW = n * bw + gap * (n - 1);
        const bx0 = ax + (x1 - ax - totalW) / 2;
        const mx = 0.5;
        [0.1, 0.2, 0.3, 0.4, 0.5].forEach((p) => {
          const y = bBot - (p / mx) * (bh - 22);
          V.line(g, ax, y, x1, y, { color: C.grid, lw: 1 });
          txt(g, Math.round(p * 100) + '%', ax - 5, y + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        });
        for (let k = 0; k < n; k++) {
          const x = bx0 + k * (bw + gap);
          let yb = bBot;
          const parts = sh[k];
          parts.forEach((p, c) => {
            const hh = (p / mx) * (bh - 22);
            const col = lanes === 1 ? (k === 1 ? C.c : k === 0 ? C.muted : C.a) : cols[c];
            V.rr(g, x, yb - hh, bw, Math.max(0.5, hh), Math.min(4, bw / 4));
            g.fillStyle = V.alpha(col, lanes === 1 ? 0.85 : (k === 1 ? 0.95 : 0.7)); g.fill();
            yb -= hh;
          });
          if (k === 1) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, x - 2, yb - 2, bw + 4, bBot - yb + 4, 4); g.stroke(); }
          const tot = parts.reduce((a, b) => a + b, 0);
          if (k <= 1 || k === n - 1) txt(g, (tot * 100).toFixed(0) + '%', x + bw / 2, yb - 6, { size: 11, align: 'center', color: k === 1 ? C.c : C.ink, font: 'mono', weight: k === 1 ? 700 : 400 });
          const every2 = n > 12 && bw < 16 ? 2 : 1;
          if (k === 0 || (k - 1) % every2 === 0 || k === n - 1) txt(g, k === 0 ? 'e' : String(k - 1), x + bw / 2, bBot + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        if (lanes === 4) {
          const lx = x1 - 4;
          const gt = S.ug ? P.gu : P.g;
          ['faixa 0 (longa)', 'faixa 1', 'faixa 2', 'faixa 3'].forEach((l, c) => {
            if (narrow && c > 0) return;
            txt(g, l + ' · gate ' + fmt(gt[c], gt[c] === 1 ? 0 : 2), lx, bTop + 12 + c * 14, { size: 11, align: 'right', color: cols[c], weight: 600 });
          });
          if (narrow) txt(g, 'gates do leitor: ' + gt.map((v) => fmt(v, v === 1 ? 0 : 2)).join(' · '), lx, bTop + 26, { size: 11, align: 'right', color: C.muted });
        }
      });
    })();

    /* ====================================== Fig 2: ler, escrever, misturar */
    (function figOps() {
      const fig = root.querySelector('#fig-hc-ops'); if (!fig) return;
      adaptH(fig, 400, 540, 640);
      const W6 = V.randMatrix(6, 6, 77, 0.55);
      const S = { seed: 1, d: null };
      function build(seed) {
        const r = V.rng(seed * 97 + 13);
        const H = Array.from({ length: 4 }, () => Array.from({ length: 6 }, () => Math.round(r.normal() * 9) / 10));
        const pre = [0.85, 0.25, 0.6, 0.35].map((v) => Math.round(V.clamp(v + (r() - 0.5) * 0.3, 0.05, 0.99) * 100) / 100);
        const post = [1.25, 0.5, 0.85, 1.6].map((v) => Math.round(V.clamp(v + (r() - 0.5) * 0.4, 0.1, 1.95) * 100) / 100);
        const Z = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => (i === j ? 2.0 : 0) + r.normal() * 0.7));
        const comb = sinkhorn(Z);
        const x = [0, 1, 2, 3, 4, 5].map((k) => H.reduce((s, row, i) => s + pre[i] * row[k], 0));
        const f = (vec) => W6.map((row) => 1.2 * Math.tanh(row.reduce((s, a, k) => s + a * vec[k], 0)));
        const y = f(x);
        const Pm = post.map((p) => y.map((v) => p * v));
        const M = [0, 1, 2, 3].map((j) => [0, 1, 2, 3, 4, 5].map((k) => H.reduce((s, row, i) => s + comb[i][j] * row[k], 0)));
        const Hn = M.map((row, j) => row.map((v, k) => v + Pm[j][k]));
        const h1 = H[0], y1 = f(h1), h1n = h1.map((v, k) => v + y1[k]);
        const e = Array.from({ length: 6 }, () => Math.round(r.normal() * 8) / 10);
        return { H, pre, post, comb, combT: tr(comb), x, y, Pm, M, Hn, h1, y1, h1n, e, H0: [e, e, e, e] };
      }
      S.d = build(S.seed);
      const steps = [
        { t: 'Entrada: repetir o embedding', d: 'Na entrada da pilha, o <code>hc_expand</code> copia o embedding \\(e\\) nas 4 linhas: \\(H^{(0)} = [e; e; e; e]\\), uma matriz de 4 fluxos × \\(d\\) colunas (aqui \\(d = 6\\); no GLM e no DeepSeek, 4.096). Nos passos seguintes, \\(H\\) é o estado de um token já no meio da pilha, com os fluxos diferentes entre si.' },
        { t: 'Ler (pre): 4 → 1', d: 'A subcamada recebe uma combinação dos fluxos, \\(x = \\sum_i \\text{pre}_i H_i\\). Os 4 pesos são deste token, previstos a partir do próprio \\(H\\).' },
        { t: 'A subcamada F', d: 'Atenção, MoE, KDA: a subcamada não sabe que o resíduo é largo. Recebe um vetor de \\(d\\) números e devolve outro, \\(y = F(\\mathrm{Norm}(x))\\).' },
        { t: 'Escrever (post): 1 → 4', d: 'O mesmo \\(y\\) entra nos 4 fluxos, cada um com seu volume: o produto externo \\(\\text{post} \\otimes y\\). No mHC, \\(\\text{post}_j = 2\\sigma(\\cdot) \\in (0, 2)\\).' },
        { t: 'Misturar (comb): 4 → 4', d: 'Cada fluxo novo é uma média ponderada dos antigos, \\(\\sum_i \\text{comb}_{ij} H_i\\): a coluna \\(j\\) de comb (aqui a linha \\(j\\) de \\(\\text{comb}^{\\top}\\)). Com colunas somando 1, cada saída é uma média ponderada das entradas; com linhas somando 1, a média dos 4 fluxos se conserva.' },
        { t: 'Somar: o novo estado', d: "\\(H' = \\text{comb}^{\\top} H + \\text{post} \\otimes y\\). Pronto para a próxima subcamada, que fará tudo de novo com os próprios 24 coeficientes." },
        { t: 'Caso particular: n = 1', d: "Com um fluxo só, pre = post = 1 e comb = [1]: sobra \\(h' = h + F(\\mathrm{Norm}(h))\\), o resíduo clássico. As hyper-connections o generalizam sem jogá-lo fora." },
      ];
      const ctl = UI.controls(fig);
      UI.button(ctl, '🎲 novo token', () => { S.seed++; S.d = build(S.seed); st.go(st.i); }, { primary: true });
      UI.legend(ctl, [[V.C().b, 'valores negativos'], [V.C().a, 'valores positivos']]);
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      const M = (data, label, rl, o = {}) => ({ data, label, rl, cmap: o.cmap || 'div', min: o.min, max: o.max, dig: o.dig == null ? 1 : o.dig, op: o.op });
      function items(i, d) {
        const RL = ['H₀', 'H₁', 'H₂', 'H₃'];
        const col = (v) => v.map((x) => [x]);
        switch (i) {
          case 0: return { left: [M([d.e], 'e (embedding)', null)], right: M(d.H0, 'H⁽⁰⁾ = [e; e; e; e]', RL), op: 'hc_expand ×4' };
          case 1: return { left: [M(col(d.pre), 'pre', null, { cmap: 'seq', min: 0, max: 2, dig: 2 }), M(d.H, 'H', RL)], right: M([d.x], 'x = Σ preᵢ·Hᵢ', null), op: 'ler' };
          case 2: return { left: [M([d.x], 'x', null)], right: M([d.y], 'y = F(Norm(x))', null), op: 'F' };
          case 3: return { left: [M(col(d.post), 'post', null, { cmap: 'seq', min: 0, max: 2, dig: 2 }), M([d.y], 'y', null)], right: M(d.Pm, 'post ⊗ y', RL), op: '⊗' };
          case 4: return { left: [M(d.combT, 'combᵀ (4×4)', null, { cmap: 'seq', min: 0, max: 1, dig: 2 }), M(d.H, 'H', RL)], right: M(d.M, 'combᵀ · H', RL), op: 'misturar' };
          case 5: return { left: [M(d.M, 'combᵀ · H', RL), M(d.Pm, 'post ⊗ y', RL)], right: M(d.Hn, "H'", RL), op: '+' };
          default: return { left: [M([d.h1], 'h (n = 1)', null), M([d.y1], 'F(h)', null)], right: M([d.h1n], "h' = h + F(h)", null), op: '+' };
        }
      }
      const dims = (it, cell) => { const R = it.data.length, K = it.data[0].length; return { w: K * cell + (K - 1) * 2 + (it.rl ? 26 : 0), h: R * cell + (R - 1) * 2 + 20 }; };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const d = S.d, conf = items(st.i, d);
        const e = st.elapsed();
        const pad = 14;
        const all = conf.left.concat(conf.right ? [conf.right] : []);
        // tenta layout horizontal com a maior célula possível
        let cell = 0, horiz = false;
        for (let c = 46; c >= 26; c -= 2) {
          const tot = all.reduce((s, it) => s + dims(it, c).w, 0) + (conf.left.length - 1) * 16 + (conf.right ? 64 : 0);
          const mh = Math.max(...all.map((it) => dims(it, c).h));
          if (tot <= w - 2 * pad && mh <= h - 40) { cell = c; horiz = true; break; }
        }
        if (!horiz) {
          for (let c = 40; c >= 20; c -= 2) {
            const mw = Math.max(...all.map((it) => dims(it, c).w));
            const th = conf.left.reduce((s, it) => s + dims(it, c).h + 12, 0) + (conf.right ? 32 + dims(conf.right, c).h : 0);
            cell = c;
            if (mw <= w - 2 * pad && th <= h - 16) break;
          }
        }
        const drawItem = (it, x, y, alpha) => {
          g.globalAlpha = alpha;
          const ox = x + (it.rl ? 26 : 0);
          const gm = V.matrix(g, { x: ox, y: y + 18, data: it.data, cell, cmap: it.cmap, min: it.min == null ? -2 : it.min, max: it.max == null ? 2 : it.max, values: true, fmt: (v) => fmt(v, it.dig), rowLabels: it.rl || null });
          txt(g, it.label, ox + gm.w / 2, y + 10, { size: 11.5, align: 'center', color: C.muted, font: 'mono', maxW: Math.max(gm.w + 30, 90) });
          g.globalAlpha = 1;
          return gm;
        };
        const fadeIn = V.ease.out(V.clamp(e / 0.8, 0, 1));
        if (horiz) {
          const tot = all.reduce((s, it) => s + dims(it, cell).w, 0) + (conf.left.length - 1) * 16 + (conf.right ? 64 : 0);
          let x = (w - tot) / 2;
          const maxH = Math.max(...all.map((it) => dims(it, cell).h));
          const y0 = Math.max(16, (h - maxH) / 2 - 10);
          const leftBoxes = [];
          conf.left.forEach((it, k) => { const dm = dims(it, cell); const yy = y0 + (maxH - dm.h) / 2; drawItem(it, x, yy, 1); leftBoxes.push([x, yy, dm]); x += dm.w + (k < conf.left.length - 1 ? 16 : 0); });
          if (conf.right) {
            const ax0 = x + 8, ax1 = x + 56, ay = y0 + maxH / 2 + 8;
            V.arrow(g, ax0, ay, ax1, ay, { color: C.c, lw: 2.2, head: 9 });
            txt(g, conf.op, (ax0 + ax1) / 2, ay - 10, { size: 11.5, align: 'center', color: C.c, weight: 600, maxW: 66 });
            for (let k = 0; k < 3; k++) { const p = ((t * 0.9 + k / 3) % 1); V.glow(g, V.lerp(ax0, ax1, p), ay, 7, C.c, 0.8); }
            const dm = dims(conf.right, cell);
            drawItem(conf.right, x + 64, y0 + (maxH - dm.h) / 2, fadeIn);
          }
        } else {
          let y = 10;
          conf.left.forEach((it) => { const dm = dims(it, cell); drawItem(it, (w - dm.w) / 2, y, 1); y += dm.h + 12; });
          if (conf.right) {
            const ay0 = y - 2, ay1 = y + 26;
            V.arrow(g, w / 2, ay0, w / 2, ay1, { color: C.c, lw: 2.2, head: 9 });
            txt(g, conf.op, w / 2 + 12, (ay0 + ay1) / 2 + 4, { size: 11.5, color: C.c, weight: 600 });
            const dm = dims(conf.right, cell);
            drawItem(conf.right, (w - dm.w) / 2, y + 32, fadeIn);
          }
        }
        if (st.i === 4 && horiz) {
          const cs = colSums(d.comb);
          txt(g, 'colunas de comb somam ' + cs.map((v) => fmt(v, 2)).join(' · '), w / 2, h - 12, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: w - 20 });
        }
      });
    })();

    /* ============================================ Fig 3: ganho composto */
    (function figGanho() {
      const fig = root.querySelector('#fig-ganho'); if (!fig) return;
      adaptH(fig, 400, 470);
      const S = { sigma: 0.3, L: 90, seed: 3, t0: 0, data: null };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Desvio σ', min: 0, max: 0.4, step: 0.02, value: S.sigma, fmt: (v) => fmt(v, 2), onInput: (v) => { S.sigma = v; compute(); } });
      UI.seg(ctl, { label: 'Profundidade', options: [{ v: 90, label: 'GLM · 90' }, { v: 86, label: 'DeepSeek · 86' }], value: 90, onChange: (v) => { S.L = v; compute(); } });
      UI.button(ctl, '🎲 novo sorteio', () => { S.seed++; compute(); });
      const rF = UI.readout(ctl, 'ganho máx. livre', '—');
      const rM = UI.readout(ctl, 'ganho máx. mHC', '—');
      let Hh = null;
      function compute() {
        const r = V.rng(S.seed * 31 + 7);
        const G = Array.from({ length: 90 }, () => Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => r.normal())));
        // P = A_ℓ ⋯ A₁ e, em paralelo, P⁻¹ = A₁⁻¹ ⋯ A_ℓ⁻¹: σmin(P) = 1/σmax(P⁻¹) é estável,
        // enquanto √(menor autovalor de PᵀP) vira ruído quando cond(P) passa de ~1e8
        let Pf = I4(), Pm = I4(), Qf = I4(), Qm = I4();
        const out = { fmax: [1], fmin: [1], mmax: [1], mmin: [1] };
        for (let l = 0; l < S.L; l++) {
          const Af = I4().map((row, i) => row.map((v, j) => v + S.sigma * G[l][i][j]));
          const Am = sinkhorn(I4().map((row, i) => row.map((v, j) => 6 * v + 4 * S.sigma * G[l][i][j])));
          Pf = mm(Af, Pf); Pm = mm(Am, Pm);
          const iF = inv4(Af), iM = inv4(Am);
          Qf = iF ? mm(Qf, iF) : Qf.map((r) => r.map(() => Infinity)); Qm = iM ? mm(Qm, iM) : Qm.map((r) => r.map(() => Infinity));
          const qf = sMax(Qf), qm = sMax(Qm);
          out.fmax.push(sMax(Pf)); out.fmin.push(isFinite(qf) && qf > 0 ? 1 / qf : 1e-300);
          out.mmax.push(sMax(Pm)); out.mmin.push(isFinite(qm) && qm > 0 ? 1 / qm : 1e-300);
        }
        S.data = out; S.t0 = Hh ? Hh.t : 0;
        const all = out.fmax.concat(out.fmin, out.mmax, out.mmin);
        S.lo = V.clamp(Math.floor(Math.log10(Math.min(...all))), -30, -1);
        S.hi = V.clamp(Math.ceil(Math.log10(Math.max(...all))), 1, 14);
        rF.set('×' + fx(out.fmax[out.fmax.length - 1]));
        rM.set('×' + fx(out.mmax[out.mmax.length - 1]));
      }
      const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
      const pow10 = (e) => '10' + String(e).split('').map((c) => SUP[c]).join('');
      function fx(v) {
        if (v >= 1e5) { const e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e); return fmt(m, 2) + '·' + pow10(e); }
        if (v >= 1000) return Math.round(v).toLocaleString('pt-BR');
        if (v >= 1) return fmt(v, v >= 10 ? 1 : 3);
        if (v >= 0.001) return fmt(v, 4);
        const e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e);
        return fmt(m, 1) + '·' + pow10(e);
      }
      compute();
      Hh = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const ml = narrow ? 48 : 56, mr = narrow ? 14 : 110, mt = narrow ? 76 : 62, mb = 34;
        const pw = w - ml - mr, ph = h - mt - mb;
        const lo = S.lo, hi = S.hi;
        const Y = (v) => mt + ph * (1 - (V.clamp(Math.log10(v), lo, hi) - lo) / (hi - lo));
        const X = (l) => ml + (l / S.L) * pw;
        // grade
        const lab = (e) => (e >= 0 && e <= 4 ? Math.pow(10, e).toLocaleString('pt-BR') : e < 0 && e >= -2 ? fmt(Math.pow(10, e), -e) : pow10(e));
        const step = Math.max(1, Math.ceil((hi - lo) / (narrow ? 7 : 10)));
        for (let e = lo; e <= hi; e++) {
          const y = Y(Math.pow(10, e));
          V.line(g, ml, y, ml + pw, y, { color: e === 0 ? V.alpha(C.ink, 0.35) : C.grid, lw: e === 0 ? 1.4 : 1 });
          if ((e - lo) % step === 0 || e === 0) txt(g, lab(e), ml - 6, y + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        }
        for (let l = 0; l <= S.L; l += narrow ? 30 : 15) { const x = X(l); V.line(g, x, mt + ph, x, mt + ph + 4, { color: C.muted }); txt(g, String(l), x, mt + ph + 16, { size: 11, align: 'center', color: C.muted, font: 'mono' }); }
        txt(g, 'sub-bloco ℓ', ml + pw, mt + ph + 30, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        { const t1 = 'fator de esticamento de ', t2 = 'Aℓ ⋯ A₁', t3 = ' (escala log)', z = narrow ? 11 : 12;
          const w1 = tw(g, t1, z, 'sans', 600), w2 = tw(g, t2, z, 'mono', 600);
          txt(g, t1, 12, 16, { size: z, color: C.ink, weight: 600 }); txt(g, t2, 12 + w1, 16, { size: z, color: C.ink, weight: 600, font: 'mono' }); txt(g, t3, 12 + w1 + w2, 16, { size: z, color: C.ink, weight: 600, maxW: Math.max(40, w - 24 - w1 - w2) }); }
        const d = S.data;
        const rev = V.clamp((t - S.t0) / 1.6, 0, 1);
        const upto = Math.max(1, Math.floor(rev * S.L));
        const line = (arr, col, dash, lw) => {
          g.beginPath();
          for (let l = 0; l <= upto; l++) { const x = X(l), y = Y(arr[l]); if (l === 0) g.moveTo(x, y); else g.lineTo(x, y); }
          g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
        };
        V.line(g, ml, Y(1), ml + pw, Y(1), { color: V.alpha(C.muted, 0.9), lw: 2.2, dash: [2, 5] });
        line(d.fmin, C.c, [6, 5], 2); line(d.fmax, C.c, null, 2.6);
        line(d.mmin, C.a, [6, 5], 2); line(d.mmax, C.a, null, 2.6);
        // pontas
        const tip = (arr, col) => { const x = X(upto), y = Y(arr[upto]); V.dot(g, x, y, 4, col); return [x, y]; };
        const pts = [[tip(d.fmax, C.c), d.fmax[upto], C.c], [tip(d.fmin, C.c), d.fmin[upto], C.c], [tip(d.mmax, C.a), d.mmax[upto], C.a], [tip(d.mmin, C.a), d.mmin[upto], C.a]];
        if (!narrow) {
          const placed = [];
          pts.sort((a, b) => a[0][1] - b[0][1]).forEach(([p, v, col]) => {
            let y = V.clamp(p[1], mt + 4, mt + ph); placed.forEach((q) => { if (Math.abs(q - y) < 14) y = q + 14; }); placed.push(y);
            txt(g, '×' + fx(v), p[0] + 8, y + 4, { size: 11, color: col, font: 'mono', weight: 600, maxW: w - p[0] - 12 });
          });
        }
        // legenda
        const lg = [[C.c, 'livre  I + σG'], [C.a, 'mHC  Sinkhorn(softmax(·))'], [C.muted, 'resíduo clássico = 1']];
        let lx = 12, ly = 36;
        lg.forEach(([col, l]) => {
          const wl = tw(g, l, 11) + 24;
          if (lx + wl > w - 8) { lx = 12; ly += 16; }
          V.line(g, lx, ly - 4, lx + 16, ly - 4, { color: col, lw: 2.6 }); txt(g, l, lx + 20, ly, { size: 11, color: C.ink }); lx += wl + 14;
        });
        txt(g, 'linha cheia: maior valor singular · tracejada: menor', 12, ly + 16, { size: 11, color: C.muted, maxW: w - 20 });
      });
    })();

    /* ============================================ Fig 4: Sinkhorn passo a passo */
    (function figSinkhorn() {
      const fig = root.querySelector('#fig-sinkhorn'); if (!fig) return;
      adaptH(fig, 420, 640, 640);
      const S = { k: 39, tau: 1, seed: 5, seq: null, err: null, play: false };
      function compute() {
        const r = V.rng(S.seed * 17 + 3);
        const Z = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => (r.normal() * 1.1) / S.tau));
        S.seq = sinkhornSeq(Z, 20, 1e-6);
        S.err = S.seq.map((M) => ({ row: Math.max(...rowSums(M).map((v) => Math.abs(v - 1))), col: Math.max(...colSums(M).map((v) => Math.abs(v - 1))) }));
      }
      compute();
      const ctl = UI.controls(fig);
      const desc = (v) => (v === 0 ? '0 · softmax' : v + (v % 2 ? ' · colunas' : ' · linhas'));
      const sK = UI.slider(ctl, { label: 'Meio-passo', min: 0, max: 39, step: 1, value: 39, fmt: desc, onInput: (v) => { S.k = v; } });
      UI.seg(ctl, { label: 'Logits', options: [{ v: 1, label: 'suaves' }, { v: 0.35, label: 'afiados' }, { v: 0.12, label: 'quase permutação' }], value: 1, onChange: (v) => { S.tau = v; compute(); } });
      const bPlay = UI.button(ctl, '▶ animar', () => { S.play = !S.play; bPlay.innerHTML = S.play ? '⏸ parar' : '▶ animar'; if (S.play && S.k >= 39) sK.set(0); });
      UI.button(ctl, '🎲 novo sorteio', () => { S.seed++; compute(); });
      V.every(380, () => { if (!S.play) return; if (S.k >= 39) { S.play = false; bPlay.innerHTML = '▶ animar'; return; } sK.set(S.k + 1); });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 640;
        const M = S.seq[S.k];
        const rs = rowSums(M), cs = colSums(M);
        const cell = narrow ? Math.min(46, Math.floor((w - 150) / 4)) : 50;
        const mx = narrow ? 44 : 50, my = 50;
        const mw = 4 * cell + 3 * 2;
        txt(g, 'meio-passo ' + desc(S.k) + (S.k === 39 ? '  (= 20 iterações do mHC)' : ''), mx - 20, 22, { size: 12.5, weight: 600, color: C.ink, maxW: narrow ? w - 30 : w * 0.5 });
        V.matrix(g, { x: mx, y: my, data: M, cell, cmap: 'seq', min: 0, max: 1, values: true, fmt: (v) => fmt(v, 2), rowLabels: ['0', '1', '2', '3'] });
        // barras de soma das linhas
        const bx = mx + mw + 12, bwMax = narrow ? Math.min(64, w - bx - 50) : 70;
        const sc = bwMax / 1.6;
        rs.forEach((v, i) => {
          const y = my + i * (cell + 2) + cell * 0.25;
          V.rr(g, bx, y, Math.max(1, Math.min(1.6, v) * sc), cell * 0.5, 4); g.fillStyle = V.alpha(C.b, 0.7); g.fill();
          txt(g, fmt(v, 3), bx + Math.min(1.6, v) * sc + 4, y + cell * 0.25 + 4, { size: 11, color: C.ink, font: 'mono' });
        });
        V.line(g, bx + sc, my - 4, bx + sc, my + 4 * (cell + 2), { color: C.ink, lw: 1.2, dash: [3, 3] });
        txt(g, 'Σ linha', bx, my - 8, { size: 11, color: C.b, weight: 600 });
        // barras de soma das colunas
        const by = my + 4 * (cell + 2) + 8, bhMax = 46, scy = bhMax / 1.6;
        cs.forEach((v, j) => {
          const x = mx + j * (cell + 2) + cell * 0.25;
          V.rr(g, x, by, cell * 0.5, Math.max(1, Math.min(1.6, v) * scy), 4); g.fillStyle = V.alpha(C.d, 0.75); g.fill();
          txt(g, fmt(v, 3), x + cell * 0.25, by + Math.min(1.6, v) * scy + 13, { size: 11, align: 'center', color: C.ink, font: 'mono' });
        });
        V.line(g, mx - 4, by + scy, mx + mw, by + scy, { color: C.ink, lw: 1.2, dash: [3, 3] });
        txt(g, 'Σ coluna', mx - 42, by + 14, { size: 11, color: C.d, weight: 600 });
        // gráfico de convergência
        let px, py, pw, ph;
        if (narrow) { px = 46; py = by + bhMax + 44; pw = w - px - 14; ph = h - py - 30; } else { px = bx + bwMax + 90; py = 50; pw = w - px - 20; ph = h - py - 46; }
        if (pw < 80 || ph < 60) return;
        const lo = -8, hi = 0;
        const Y = (v) => py + ph * (1 - (V.clamp(Math.log10(Math.max(v, 1e-12)), lo, hi) - lo) / (hi - lo));
        const X = (k) => px + (k / 39) * pw;
        for (let e = lo; e <= hi; e += 2) { const y = Y(Math.pow(10, e)); V.line(g, px, y, px + pw, y, { color: C.grid }); txt(g, '1e' + e, px - 5, y + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }); }
        txt(g, 'pior |Σ − 1| (log)', px, py - 10, { size: 11, color: C.muted });
        const ln = (key, col) => { g.beginPath(); S.err.forEach((e, k) => { const x = X(k), y = Y(e[key]); if (k === 0) g.moveTo(x, y); else g.lineTo(x, y); }); g.strokeStyle = col; g.lineWidth = 2.2; g.stroke(); };
        ln('row', C.b); ln('col', C.d);
        V.line(g, X(S.k), py, X(S.k), py + ph, { color: C.c, lw: 1.5, dash: [3, 3] });
        V.dot(g, X(S.k), Y(S.err[S.k].row), 4.5, C.b); V.dot(g, X(S.k), Y(S.err[S.k].col), 4.5, C.d);
        for (let k = 0; k <= 39; k += 13) txt(g, String(k), X(k), py + ph + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        txt(g, '39', X(39), py + ph + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        txt(g, 'linhas', px + pw - 4, py + 14, { size: 11, align: 'right', color: C.b, weight: 600 });
        txt(g, 'colunas', px + pw - 4, py + 28, { size: 11, align: 'right', color: C.d, weight: 600 });
      });
    })();

    /* ============================================ Fig 5: mapa de metrô do mHC */
    (function figMetro() {
      const fig = root.querySelector('#fig-metro'); if (!fig) return;
      adaptH(fig, 480, 800, 640);
      const S = { model: 'glm', fused: true, seed: 11, mix: null };
      function makeMix(seed) {
        const r = V.rng(seed * 53 + 5);
        const one = () => {
          const m = Array.from({ length: 24 }, () => r.normal() * 0.9);
          const pre = [0, 1, 2, 3].map((j) => sig(1.2 * m[j] + 0.2) + 1e-6);
          const post = [0, 1, 2, 3].map((j) => 2 * sig(0.9 * m[4 + j]));
          const Z = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => 0.9 * m[8 + 4 * i + j] + (i === j ? 1.7 : 0)));
          return { pre, post, comb: sinkhorn(Z) };
        };
        return { attn: one(), ffn: one() };
      }
      S.mix = makeMix(S.seed);
      const parts = [];
      { const r = V.rng(99); for (let i = 0; i < 4; i++) for (let k = 0; k < 7; k++) parts.push({ lane: i, ph: (k + r() * 0.6) / 7, r1: r(), r2: r() }); }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM-5.3' }, { v: 'ds', label: 'DeepSeek-V4' }], value: 'glm', onChange: (v) => { S.model = v; } });
      UI.toggle(ctl, { label: 'fusão post + pre (MHCFusedPostPreOp)', value: true, onChange: (v) => { S.fused = v; kr.set(v ? '2 por camada' : '4 por camada'); } });
      UI.button(ctl, '🎲 próximo token', () => { S.seed++; S.mix = makeMix(S.seed); }, { primary: true });
      const kr = UI.readout(ctl, 'kernels mHC', '2 por camada');
      const NAMES = {
        glm: { f1: 'KDA ou MLA+DSA', f2: 'MLP densa ou MoE 288', norms: 'norm = input_layernorm / post_attention_layernorm', out: 'fim da pilha: hc_contract (média) → norm → lm_head' },
        ds: { f1: 'SWA / CSA / HCA', f2: 'MoE 256 (hash 0–2)', norms: 'norm = attn_norm / ffn_norm (fundidas no kernel mHC)', out: 'fim da pilha: hc_head (pesos σ) → norm → lm_head' },
      };
      const U = { pre1: 0.05, t1s: 0.085, n1: 0.125, f1: 0.215, t1e: 0.285, b1s: 0.30, b1e: 0.40, pre2: 0.465, t2s: 0.50, n2: 0.54, f2: 0.63, t2e: 0.70, b2s: 0.715, b2e: 0.815 };
      const target = (row, rr) => { let c = 0; for (let j = 0; j < 4; j++) { c += row[j]; if (rr < c) return j; } return 3; };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), cols = SC(C);
        const horiz = w >= 640;
        const nm = NAMES[S.model], mix = S.mix;
        const panelH = horiz ? 200 : 236;
        let gap, P, span;
        if (horiz) {
          gap = 22;
          const fx0 = 36, fx1 = w - 30, trackY = 74, yc = trackY + 1.5 * gap + 84;
          span = fx1 - fx0;
          P = (u, l) => [fx0 + u * span, l == null ? trackY : yc + (l - 1.5) * gap];
        } else {
          gap = 19;
          const xc = Math.min(w * 0.66, w - 112), trackX = xc - 1.5 * gap - 64, fy0 = 44, fy1 = h - panelH - 30;
          span = fy1 - fy0;
          P = (u, l) => [l == null ? trackX : xc + (l - 1.5) * gap, fy0 + u * span];
        }
        // caixas de kernel
        const kbox = (u0, u1, label) => {
          const a = P(u0, 0), b = P(u1, 3), c = P(u0, null);
          let x, y, ww, hh;
          if (horiz) { x = a[0]; ww = b[0] - a[0]; y = c[1] - 24; hh = b[1] + 14 - y; } else { y = a[1]; hh = b[1] - a[1]; x = c[0] - 52; ww = b[0] + 12 - x; }
          g.fillStyle = V.alpha(C.d, 0.05); V.rr(g, x, y, ww, hh, 10); g.fill();
          g.save(); g.setLineDash([5, 4]); V.rr(g, x, y, ww, hh, 10); g.strokeStyle = V.alpha(C.d, 0.8); g.lineWidth = 1.4; g.stroke(); g.restore();
          if (horiz) txt(g, label, x + 4, y - 6, { size: 11, color: C.d, weight: 600, font: 'mono', maxW: ww - 6 });
          else kLabels.push([label.replace('kernel fundido: ', ''), x + 4, y + 13, Math.max(40, P(0, 0)[0] - 10 - (x + 4))]);
        };
        // no modo vertical os rótulos das caixas vão por cima das curvas, com fundo próprio
        const kLabels = [];
        const drawKLabels = () => kLabels.forEach(([s2, lx, ly, mw]) => {
          const tw0 = Math.min(mw, tw(g, s2, 11, 'mono', 600));
          g.fillStyle = V.alpha(C.panel || C.bg, 0.92); V.rr(g, lx - 3, ly - 11, tw0 + 6, 15, 4); g.fill();
          txt(g, s2, lx, ly, { size: 11, color: C.d, weight: 600, font: 'mono', maxW: mw });
        });
        if (S.fused) { kbox(0.015, 0.105, 'post(ℓ−1)+pre'); kbox(0.275, 0.52, 'kernel fundido: post + pre'); kbox(0.69, horiz ? 0.985 : 0.86, 'kernel fundido: post + pre(ℓ+1)'); }
        else { kbox(0.03, 0.10, 'pre'); kbox(0.275, 0.415, 'post'); kbox(0.445, 0.515, 'pre'); kbox(0.69, 0.83, 'post'); }
        // linhas retas
        const seg = (u0, u1, l, col) => { const a = P(u0, l), b = P(u1, l); V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(col, 0.75), lw: 4 }); };
        for (let i = 0; i < 4; i++) { seg(0, U.b1s, i, cols[i]); seg(U.b1e, U.b2s, i, cols[i]); seg(U.b2e, 1, i, cols[i]); }
        // baldeações (comb)
        const braid = (u0, u1, comb) => { for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { const c = comb[i][j]; if (c < 0.02) continue; flowCurve(g, P(u0, i), P(u1, j), horiz, { color: cols[i], lw: 0.6 + 6 * c, alpha: 0.2 + 0.6 * c }); } };
        braid(U.b1s, U.b1e, mix.attn.comb); braid(U.b2s, U.b2e, mix.ffn.comb);
        // pre → trilho expresso → post
        const express = (uPre, uTs, uN, uF, uTe, uBe, m, fName) => {
          for (let i = 0; i < 4; i++) flowCurve(g, P(uPre, i), P(uTs, null), horiz, { color: cols[i], lw: 1 + 4 * m.pre[i], alpha: 0.35 + 0.5 * Math.min(1, m.pre[i]) });
          const a = P(uTs, null), b = P(uTe, null);
          V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(C.ink, 0.5), lw: 3.5 });
          for (let j = 0; j < 4; j++) flowCurve(g, P(uTe, null), P(uBe, j), horiz, { color: C.f, lw: 0.6 + 1.6 * m.post[j], alpha: 0.32 + 0.2 * m.post[j] });
          // norm (pílula) e subcamada (caixa), sem se sobrepor
          const pn = P(uN, null), pte = P(uTe, null);
          const nw = horiz ? 40 : 44, nh = 20;
          V.box(g, pn[0] - nw / 2, pn[1] - nh / 2, nw, nh, { r: 10, fill: V.alpha(C.b, 0.25), stroke: C.b, lw: 1.3 });
          txt(g, 'norm', pn[0], pn[1] + 4, { size: 11, align: 'center', color: C.ink, weight: 600 });
          let pf, fw, fh;
          if (horiz) { fw = Math.min(118, pte[0] - pn[0] - nw / 2 - 16); fh = 34; pf = [pn[0] + nw / 2 + 8 + fw / 2, pn[1]]; }
          else { fw = 96; fh = 28; pf = [pn[0], Math.min(pn[1] + nh / 2 + 6 + fh / 2, pte[1] - fh / 2 - 2)]; }
          V.box(g, pf[0] - fw / 2, pf[1] - fh / 2, fw, fh, { r: 9, fill: V.alpha(C.e, 0.2), stroke: C.e, lw: 1.5 });
          if (tw(g, fName, 11, 'sans', 600) <= fw - 10) txt(g, fName, pf[0], pf[1] + 4, { size: 11, align: 'center', color: C.ink, weight: 600, maxW: fw - 10 });
          else { const ws = fName.split(' '), cut = Math.ceil(ws.length / 2); txt(g, ws.slice(0, cut).join(' '), pf[0], pf[1] - 2, { size: 11, align: 'center', color: C.ink, weight: 600, maxW: fw - 8 }); txt(g, ws.slice(cut).join(' '), pf[0], pf[1] + 11, { size: 11, align: 'center', color: C.ink, weight: 600, maxW: fw - 8 }); }
          for (let k = 0; k < 3; k++) { const s2 = (t * 0.35 + k / 3) % 1; V.glow(g, V.lerp(a[0], b[0], s2), V.lerp(a[1], b[1], s2), 8, C.ink, 0.45); }
          for (let j = 0; j < 4; j++) { const s2 = (t * 0.5 + j * 0.25) % 1; const q = curvePoint(P(uTe, null), P(uBe, j), horiz, s2); V.dot(g, q[0], q[1], 2.6, C.f); }
        };
        express(U.pre1, U.t1s, U.n1, U.f1, U.t1e, U.b1e, mix.attn, nm.f1);
        express(U.pre2, U.t2s, U.n2, U.f2, U.t2e, U.b2e, mix.ffn, nm.f2);
        // passageiros
        parts.forEach((p) => {
          const u = (p.ph + t * 0.05) % 1;
          const t1 = target(mix.attn.comb[p.lane], p.r1), t2 = target(mix.ffn.comb[t1], p.r2);
          let q;
          if (u < U.b1s) q = P(u, p.lane);
          else if (u < U.b1e) q = curvePoint(P(U.b1s, p.lane), P(U.b1e, t1), horiz, (u - U.b1s) / (U.b1e - U.b1s));
          else if (u < U.b2s) q = P(u, t1);
          else if (u < U.b2e) q = curvePoint(P(U.b2s, t1), P(U.b2e, t2), horiz, (u - U.b2s) / (U.b2e - U.b2s));
          else q = P(u, t2);
          V.dot(g, q[0], q[1], 3.3, cols[p.lane]);
          g.strokeStyle = V.alpha(C.bg || '#000', 0.6); g.lineWidth = 1; g.beginPath(); g.arc(q[0], q[1], 3.3, 0, Math.PI * 2); g.stroke();
        });
        // rótulos das estações
        const lab = (u, s2, col) => { const p = P(u, 3); if (horiz) txt(g, s2, p[0], p[1] + 30, { size: 11, align: 'center', color: col, weight: 600 }); else txt(g, s2, p[0] + 16, p[1] + 4, { size: 11, color: col, weight: 600 }); };
        lab(U.pre1 + 0.01, 'pre 4→1', C.ink); lab((U.b1s + U.b1e) / 2, 'comb + post', C.f); lab(U.pre2 + 0.01, 'pre 4→1', C.ink); lab((U.b2s + U.b2e) / 2, 'comb + post', C.f);
        drawKLabels();
        if (horiz) {
          txt(g, 'H da camada ℓ−1  [T, 4, 4096]  →  uma camada  →  camada ℓ+1', 12, 16, { size: 11, color: C.muted, font: 'mono', maxW: w - 24 });
          ['H₀', 'H₁', 'H₂', 'H₃'].forEach((s2, i) => { const p = P(0, i); txt(g, s2, p[0] - 6, p[1] + 4, { size: 11, align: 'right', color: cols[i], font: 'mono' }); });
          const yb = P(1, 3)[1] + 52;
          txt(g, nm.norms, 36, yb, { size: 11, color: C.muted, font: 'mono', maxW: w * 0.48 });
          txt(g, nm.out, w - 30, yb, { size: 11, align: 'right', color: C.muted, font: 'mono', maxW: w * 0.46 });
        } else {
          txt(g, 'H [T, 4, 4096]: uma camada, de cima para baixo', 10, 16, { size: 11, color: C.muted, font: 'mono', maxW: w - 20 });
          ['H₀', 'H₁', 'H₂', 'H₃'].forEach((s2, i) => { const p = P(0, i); txt(g, s2, p[0], p[1] - 8, { size: 11, align: 'center', color: cols[i], font: 'mono' }); });
        }
        // painel dos 24 números (sub-bloco de atenção)
        const py0 = h - panelH + (horiz ? 18 : 6);
        txt(g, 'os 24 números deste token no sub-bloco de atenção', 16, py0, { size: 11.5, color: C.ink, weight: 600, maxW: w - 30 });
        if (!horiz) { txt(g, nm.norms, 16, py0 + 16, { size: 11, color: C.muted, font: 'mono', maxW: w - 30 }); txt(g, nm.out, 16, py0 + 31, { size: 11, color: C.muted, font: 'mono', maxW: w - 30 }); }
        const top = py0 + (horiz ? 16 : 44);
        // barras: no largo, pre e post lado a lado; no estreito, uma em cima da outra, com a comb à direita
        const bw = horiz ? 24 : 24, pitch = horiz ? 31 : 32, bhMax = horiz ? 88 : 42;
        const bars = (x0, y0, label, vals, max, colFn) => {
          txt(g, label, x0, y0 + 10, { size: 11, color: C.muted, font: 'mono' });
          vals.forEach((v, i) => {
            const x = x0 + i * pitch, hh = (v / max) * bhMax, yb = y0 + 28 + bhMax;
            V.rr(g, x, yb - hh, bw, Math.max(1, hh), 4); g.fillStyle = V.alpha(colFn(i), 0.8); g.fill();
            txt(g, fmt(v, 2), x + bw / 2, yb - hh - 4, { size: 11, align: 'center', color: C.ink, font: 'mono' });
            txt(g, String(i), x + bw / 2, yb + 13, { size: 11, align: 'center', color: C.muted, font: 'mono' });
          });
          return 4 * pitch;
        };
        let x = 16;
        if (horiz) {
          x += bars(x, top, 'pre = σ(·)+ε', mix.attn.pre, 1.05, (i) => cols[i]) + 28;
          x += bars(x, top, 'post = 2σ(·)', mix.attn.post, 2, () => C.f) + 32;
        } else {
          bars(x, top, 'pre = σ(·)+ε', mix.attn.pre, 1.05, (i) => cols[i]);
          bars(x, top + 90, 'post = 2σ(·)', mix.attn.post, 2, () => C.f);
          x += 4 * pitch + 16;
        }
        const cell = horiz ? 32 : Math.min(32, Math.floor((w - x - 30) / 4));
        if (cell < 16) return;
        txt(g, horiz ? 'comb (de i para j)' : 'comb (i → j)', x, top + 10, { size: 11, color: C.muted, font: 'mono' });
        V.matrix(g, { x: x + 16, y: top + 22, data: mix.attn.comb, cell, cmap: 'seq', min: 0, max: 1, values: cell >= 24, fmt: (v) => fmt(v, 2), rowLabels: ['0', '1', '2', '3'] });
        if (!horiz) txt(g, 'colunas somam 1', x + 16, top + 22 + 4 * cell + 6 + 16, { size: 11, color: C.muted });
      });
    })();

    /* ============================================ Fig 6: Gated Residual */
    (function figGR() {
      const fig = root.querySelector('#fig-gr'); if (!fig) return;
      adaptH(fig, 362, 520, 640);
      const NB = 4, DD = 24, RK = 3;
      /* W_u escolhida à mão (exemplo didático): viés negativo em todas as direções do gargalo e
         poucos canais "ligados" a cada direção, para que cada token acenda poucos canais */
      const HOT = [[[0, 3], [0, 11], [2, 17], [3, 6], [1, 20]], [[1, 4], [1, 15], [3, 19], [0, 8], [2, 1]], [[2, 9], [2, 22], [0, 14], [1, 7], [3, 12]]];
      const Wd = V.randMatrix(RK, NB * DD, 293, 0.6), Ww = V.randMatrix(NB, NB * DD, 213, 0.25), Wf = V.randMatrix(DD, DD, 214, 0.5);
      const Wu = V.randMatrix(NB * DD, RK, 212, 0.25).map((r) => r.map((v) => v - 2.2));
      HOT.forEach((set, k) => set.forEach(([b, c]) => { Wu[b * DD + c][k] += 6.5; }));
      const gam = V.randMatrix(NB, DD, 215, 0.15).map((r) => r.map((v) => 1 + v));
      const TOK = [{ label: '“Rio”', sc: [2.4, 0.5, 1.1, 0.25], seed: 1 }, { label: '“de”', sc: [0.6, 1.8, 0.4, 1.0], seed: 2 }, { label: '“Janeiro”', sc: [1.2, 0.3, 2.0, 0.7], seed: 3 }];
      const S = { tok: 0, scalar: false, d: null };
      function build() {
        const tk = TOK[S.tok], r = V.rng(tk.seed * 71 + 9);
        const R = tk.sc.map((s) => Array.from({ length: DD }, () => s * r.normal()));
        const rms = R.map((row) => Math.sqrt(row.reduce((a, v) => a + v * v, 0) / DD + 1e-6));
        const Rh = R.map((row, i) => row.map((v, k) => (v / rms[i]) * gam[i][k]));
        const vec = [].concat(...Rh);
        const z = Wd.map((row) => silu(row.reduce((a, wv, k) => a + wv * vec[k], 0) / NB));
        const gflat = Wu.map((row) => sig(row.reduce((a, wv, k) => a + wv * z[k], 0)));
        const Gel = [0, 1, 2, 3].map((i) => gflat.slice(i * DD, (i + 1) * DD));
        const Gsc = Gel.map((row) => { const m = row.reduce((a, b) => a + b, 0) / DD; return row.map(() => m); });
        const read = (G) => { const GR = G.map((row, i) => row.map((gv, k) => gv * Rh[i][k])); return { GR, x: Array.from({ length: DD }, (_, k) => GR.reduce((a, row) => a + row[k], 0) / NB) }; };
        const rEl = read(Gel), rSc = read(Gsc);
        const G = S.scalar ? Gsc : Gel, GR = S.scalar ? rSc.GR : rEl.GR, x = S.scalar ? rSc.x : rEl.x;
        const hot = []; Gel.forEach((row, b) => row.forEach((v, k) => { if (v > 0.9) hot.push([b, k]); }));
        const nrm = (v) => Math.sqrt(v.reduce((a, q) => a + q * q, 0));
        const diff = nrm(rEl.x.map((v, k) => v - rSc.x[k])) / Math.max(1e-9, nrm(rEl.x));
        const xMax = Math.max(1e-6, ...rEl.x.map(Math.abs), ...rSc.x.map(Math.abs));
        const y = Wf.map((row) => Math.tanh(row.reduce((a, wv, k) => a + wv * x[k], 0)));
        const s = Ww.map((row) => 2 * sig(row.reduce((a, wv, k) => a + wv * vec[k], 0) / NB));
        const Rn = R.map((row, i) => row.map((v, k) => v + s[i] * y[k]));
        const rmsH = Rh.map((row) => Math.sqrt(row.reduce((a, v) => a + v * v, 0) / DD));
        S.d = { R, rms, Rh, rmsH, z, G, GR, x, y, s, Rn, hot, xe: rEl.x, xs: rSc.x, diff, xMax, yMax: Math.max(1e-6, ...y.map(Math.abs)) };
      }
      build();
      const steps = [
        { t: '4 ramos com escalas diferentes', d: 'O estado do token é \\(R \\in \\mathbb{R}^{4\\times d}\\) (no Qwen, 4 × 2560 = 10.240 números). Cada ramo acumulou escritas diferentes, então as RMS são bem diferentes (barras à direita).' },
        { t: 'RMSNorm por ramo', d: 'Cada ramo é normalizado separadamente, \\(\\hat R_i = \\mathrm{RMSNorm}(R_i;\\gamma_i)\\), com ganho \\(1 + w\\) próprio (a RMSNorm agrupada, <code>hc_norm</code> [10240]). Agora nenhum ramo grita mais alto que os outros.' },
        { t: 'Gargalo de posto 320 → gates', d: 'Os 4 ramos normalizados passam juntos por \\(W_d\\) (10.240 → 320, aqui 96 → 3), SiLU, \\(W_u\\) (320 → 10.240) e sigmoide: \\(G \\in (0,1)^{4\\times d}\\), um gate por ramo <b>e por canal</b>, previsto a partir de todos os ramos.' },
        { t: 'Leitura elementwise', d: '\\(x = \\tfrac14 \\sum_i G_i \\odot \\hat R_i\\). O token pega os poucos canais acesos (contornados) de cada ramo e quase ignora o resto. À direita, o \\(x\\) elementwise contra o que sairia com um peso só por ramo. Isto é uma GatedNorm aplicada ao resíduo largo, e substitui a pré-norma: não há <code>input_layernorm</code>.' },
        { t: 'A subcamada', d: 'GDN, QSA ou MoE recebe \\(x \\in \\mathbb{R}^{2560}\\) e devolve \\(y = F(x)\\). Como no mHC, a subcamada não sabe que o resíduo é largo.' },
        { t: 'Escrita escalar, sem mistura', d: '\\(s = 2\\sigma(W_w\\,\\mathrm{vec}(\\hat R)/4) \\in (0,2)^4\\) e \\(R_i \\leftarrow R_i + s_i\\,y\\). Não existe \\(H_{\\text{res}}\\): os ramos nunca trocam informação diretamente. Cada um é um acumulador das escritas.' },
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Token', options: TOK.map((tk, i) => ({ v: i, label: tk.label })), value: 0, onChange: (v) => { S.tok = v; build(); } });
      UI.toggle(ctl, { label: 'leitura escalar por ramo (estilo mHC)', value: false, onChange: (v) => { S.scalar = v; build(); } });
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), cols = SC(C);
        const d = S.d, i = st.i, e = st.elapsed();
        const narrow = w < 640;
        const fade = V.ease.out(V.clamp(e / 0.8, 0, 1));
        const lx = narrow ? 50 : 62;
        const stripW = narrow ? w - lx - 12 : Math.min(w * 0.52, 470);
        const cw = Math.min(18, (stripW - (DD - 1) * 1.5) / DD), ch = narrow ? 20 : 26;
        const sy0 = 44, rowGap = narrow ? 30 : 42;
        const strip = (x, y, arr, colFn, alpha = 1) => { g.globalAlpha = alpha; arr.forEach((v, k) => { g.fillStyle = colFn(v, k); V.rr(g, x + k * (cw + 1.5), y, cw, ch, Math.min(3, cw / 4)); g.fill(); }); g.globalAlpha = 1; return DD * cw + (DD - 1) * 1.5; };
        const divCol = (mx) => (v) => V.cmap(v / mx, 'div');
        const seqCol = (v) => V.cmap(v, 'seq');
        const mAbs = (M) => Math.max(1e-6, ...M.map((r) => Math.max(...r.map(Math.abs))));
        let data, colFn, title;
        if (i === 0) { data = d.R; colFn = divCol(mAbs(d.R)); title = 'R (bruto)'; }
        else if (i === 1) { data = d.Rh; colFn = divCol(2.5); title = 'R̂ = RMSNorm por ramo'; }
        else if (i === 2) { data = d.G; colFn = seqCol; title = S.scalar ? 'G (um valor por ramo)' : 'G = σ(W_u SiLU(W_d vec(R̂)/4))'; }
        else if (i === 3) { data = d.GR; colFn = divCol(2.5); title = 'G ⊙ R̂'; }
        else if (i === 4) { data = d.Rh; colFn = divCol(2.5); title = 'R̂ (já lido)'; }
        else { data = d.Rn; colFn = divCol(mAbs(d.Rn)); title = "R' = R + s·y"; }
        txt(g, title, lx, 24, { size: 12, color: C.ink, weight: 600, font: 'mono', maxW: stripW });
        const stripsEnd = sy0 + 4 * rowGap;
        for (let b = 0; b < NB; b++) {
          const y = sy0 + b * rowGap;
          txt(g, 'ramo ' + b, lx - 8, y + ch / 2 + 4, { size: 11, align: 'right', color: cols[b], weight: 600 });
          strip(lx, y, data[b], colFn, (i === 4 ? 0.45 : 1) * (i === 0 ? 1 : 0.35 + 0.65 * fade));
        }
        // canais acesos (G > 0,9): a lupa
        if ((i === 2 || i === 3) && !S.scalar) {
          const pulse = 0.6 + 0.4 * Math.sin(t * 4);
          d.hot.forEach(([b, k]) => { g.strokeStyle = V.alpha(C.c, pulse); g.lineWidth = 2.2; V.rr(g, lx + k * (cw + 1.5) - 2, sy0 + b * rowGap - 2, cw + 4, ch + 4, 4); g.stroke(); });
        }
        // painel da direita (ou de baixo, no estreito)
        const px = narrow ? 16 : lx + stripW + 34, py = narrow ? stripsEnd + 18 : sy0, pw = narrow ? w - 32 : w - px - 16;
        const miniW = Math.min(12, (pw - 30) / DD);
        const mini = (x, y, arr, colFn) => { arr.forEach((v, k) => { g.fillStyle = colFn(v); V.rr(g, x + k * (miniW + 1), y, miniW, 18, 2.5); g.fill(); }); return DD * (miniW + 1); };
        if (i === 0 || i === 1) {
          const vals = i === 0 ? d.rms : d.rmsH, mx = Math.max(...vals, 1.2);
          txt(g, i === 0 ? 'RMS de cada ramo' : 'RMS depois da norma (≈ ganho 1+w)', px, py - 10, { size: 11, color: C.muted, maxW: pw });
          vals.forEach((v, b) => {
            const y = narrow ? py + b * 22 : sy0 + b * rowGap;
            const bw = Math.max(2, (v / mx) * (pw - 70));
            V.rr(g, px, y + 3, bw * (i === 0 ? 1 : fade * 0.6 + 0.4), ch - 6, 5); g.fillStyle = V.alpha(cols[b], 0.8); g.fill();
            txt(g, fmt(v, 2), px + bw + 6, y + ch / 2 + 4, { size: 11, color: C.ink, font: 'mono' });
          });
        } else if (i === 2) {
          // ampulheta do gargalo
          const hy = narrow ? py + 10 : sy0 + 6, hh = narrow ? Math.min(130, h - py - 120) : 4 * rowGap - 16;
          const xa = px + 6, xb = px + pw * 0.45, xc2 = px + pw - 6;
          const topA = hy, botA = hy + hh;
          g.fillStyle = V.alpha(C.b, 0.14); g.beginPath(); g.moveTo(xa, topA); g.lineTo(xb, hy + hh / 2 - 18); g.lineTo(xb, hy + hh / 2 + 18); g.lineTo(xa, botA); g.closePath(); g.fill();
          g.fillStyle = V.alpha(C.a, 0.14); g.beginPath(); g.moveTo(xb, hy + hh / 2 - 18); g.lineTo(xc2, topA); g.lineTo(xc2, botA); g.lineTo(xb, hy + hh / 2 + 18); g.closePath(); g.fill();
          for (let b = 0; b < 4; b++) { V.rr(g, xa - 4, topA + b * (hh / 4) + 3, 8, hh / 4 - 6, 3); g.fillStyle = cols[b]; g.fill(); }
          d.z.forEach((zv, k) => { const cy = hy + hh / 2 - 12 + k * 12; V.dot(g, xb, cy, 5, C.d); txt(g, fmt(zv, 2), xb + 9, cy + 4, { size: 11, color: C.ink, font: 'mono' }); });
          for (let k = 0; k < 5; k++) { const s = (t * 0.6 + k / 5) % 1; const q = s < 0.5 ? [V.lerp(xa, xb, s * 2), V.lerp(hy + (k / 4) * hh, hy + hh / 2, s * 2)] : [V.lerp(xb, xc2, s * 2 - 1), V.lerp(hy + hh / 2, hy + (k / 4) * hh, s * 2 - 1)]; V.glow(g, q[0], q[1], 6, C.d, 0.9); }
          txt(g, '4×d → 320 → 4×d', px + pw / 2, botA + 16, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: pw });
          txt(g, '(aqui 96 → 3 → 96)', px + pw / 2, botA + 30, { size: 11, align: 'center', color: C.muted, maxW: pw });
          if (!S.scalar) txt(g, 'acesos (G > 0,9): ' + d.hot.map(([b, k]) => b + '·' + k).join('  '), px + pw / 2, botA + 46, { size: 11, align: 'center', color: C.c, font: 'mono', maxW: pw });
        } else if (i === 3) {
          txt(g, 'x = ¼ Σ Gᵢ ⊙ R̂ᵢ  (d = 24; no Qwen, 2560)', px, py - 10, { size: 11, color: C.muted, maxW: pw });
          const xy = narrow ? py + 22 : sy0 + rowGap * 0.8;
          const hotK = new Set(d.hot.map(([, k]) => k));
          txt(g, 'elementwise (um gate por canal)' + (S.scalar ? '' : '  ← em uso'), px, xy - 5, { size: 11, color: S.scalar ? C.muted : C.ink, weight: S.scalar ? 400 : 600, maxW: pw });
          mini(px, xy, d.xe, divCol(d.xMax));
          hotK.forEach((k) => { g.strokeStyle = C.c; g.lineWidth = 1.8; V.rr(g, px + k * (miniW + 1) - 1.5, xy - 1.5, miniW + 3, 21, 3); g.stroke(); });
          const xy2 = xy + 44;
          txt(g, 'escalar (a média dos gates do ramo)' + (S.scalar ? '  ← em uso' : ''), px, xy2 - 5, { size: 11, color: S.scalar ? C.ink : C.muted, weight: S.scalar ? 600 : 400, maxW: pw });
          mini(px, xy2, d.xs, divCol(d.xMax));
          txt(g, '‖x_elem − x_esc‖ / ‖x_elem‖ = ' + fmt(d.diff, 2), px, xy2 + 38, { size: 11, color: C.c, font: 'mono', weight: 600, maxW: pw });
          txt(g, 'mesma escala de cor nas duas faixas', px, xy2 + 54, { size: 11, color: C.muted, maxW: pw });
          for (let k = 0; k < 4; k++) { const s = (t * 0.7 + k / 4) % 1; const a = [lx + stripW, sy0 + k * rowGap + ch / 2], b = [px, xy + 9]; const q = [V.lerp(a[0], b[0], s), V.lerp(a[1], b[1], s)]; if (!narrow) V.glow(g, q[0], q[1], 6, cols[k], 0.9); }
        } else if (i === 4) {
          txt(g, 'x = ¼ Σ Gᵢ ⊙ R̂ᵢ  (d = 24; no Qwen, 2560)', px, py - 10, { size: 11, color: C.muted, maxW: pw });
          const xy = narrow ? py + 8 : sy0 + rowGap * 0.8;
          mini(px, xy, d.x, divCol(d.xMax));
          {
            const fy = xy + 44;
            V.arrow(g, px + 40, xy + 22, px + 40, fy - 4, { color: C.e, lw: 2 });
            const bw2 = Math.min(200, pw - 10);
            V.box(g, px, fy, bw2, 30, { r: 8, fill: V.alpha(C.e, 0.16), stroke: C.e, lw: 1.4 });
            txt(g, 'F = GDN, QSA ou MoE', px + bw2 / 2, fy + 19, { size: 11.5, align: 'center', color: C.ink, weight: 600, maxW: bw2 - 8 });
            V.arrow(g, px + 40, fy + 32, px + 40, fy + 56, { color: C.e, lw: 2 });
            txt(g, 'y = F(x)', px, fy + 72, { size: 11, color: C.muted, font: 'mono' });
            g.globalAlpha = fade; mini(px, fy + 78, d.y, divCol(d.yMax)); g.globalAlpha = 1;
          }
        } else {
          txt(g, 'y (o mesmo para todos os ramos)', px, py - 10, { size: 11, color: C.muted, maxW: pw });
          mini(px, py, d.y, divCol(1));
          txt(g, 's = 2σ(W_w vec(R̂)/4): um escalar por ramo', px, py + 42, { size: 11, color: C.muted, maxW: pw });
          d.s.forEach((sv, b) => {
            const cx = px + 26 + b * Math.min(64, (pw - 20) / 4), cy = py + 88, r = 20;
            g.beginPath(); g.arc(cx, cy, r, Math.PI, 2 * Math.PI); g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 6; g.stroke();
            g.beginPath(); g.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * (sv / 2) * fade); g.strokeStyle = cols[b]; g.lineWidth = 6; g.stroke();
            txt(g, fmt(sv, 2), cx, cy + 16, { size: 11, align: 'center', color: C.ink, font: 'mono', weight: 600 });
          });
        }
        if (S.scalar && i >= 2) txt(g, 'leitura escalar: o ramo inteiro entra com o mesmo peso', 16, h - 70, { size: 11, color: C.c, maxW: w - 32 });
        // trilha "você está aqui" do Gated Residual
        const NODES = narrow ? ['R', 'norm', '320', 'G⊙R̂', 'x', 'F', 's·y', "R'"] : ['R', 'norm', 'gargalo', 'G ⊙ R̂', 'x', 'F', 's·y', "R'"];
        const ACT = [[0], [0, 1], [1, 2], [2, 3, 4], [4, 5], [5, 6, 7]][i];
        const ny = h - 30, nx0 = 16, nx1 = w - 16, step = (nx1 - nx0) / NODES.length;
        V.line(g, nx0 + step / 2, ny, nx1 - step / 2, ny, { color: V.alpha(C.muted, 0.35), lw: 2 });
        NODES.forEach((nd, k) => {
          const cx = nx0 + step * (k + 0.5), on = ACT.includes(k);
          const bw2 = Math.min(step - 4, tw(g, nd, 11, 'mono', 600) + (narrow ? 10 : 16));
          V.box(g, cx - bw2 / 2, ny - 12, bw2, 24, { r: 12, fill: on ? V.alpha(C.a, 0.3) : V.alpha(C.muted, 0.1), stroke: on ? C.a : V.alpha(C.muted, 0.4), lw: on ? 1.6 : 1 });
          txt(g, nd, cx, ny + 4, { size: 11, align: 'center', color: on ? C.ink : C.muted, font: 'mono', weight: on ? 600 : 400, maxW: bw2 - 6 });
        });
        const a0 = nx0 + step * (ACT[0] + 0.5), a1 = nx0 + step * (ACT[ACT.length - 1] + 0.5);
        if (a1 > a0) { const q = (t * 0.6) % 1; V.glow(g, V.lerp(a0, a1, q), ny - 16, 7, C.a, 0.9); }
        txt(g, 'um GR (por sub-bloco): ler → subcamada → escrever', 16, ny - 22, { size: 11, color: C.muted, maxW: w * 0.6 });
      });
    })();

    /* ============================================ Fig 7: ablação */
    (function figAblacao() {
      const fig = root.querySelector('#fig-ablacao'); if (!fig) return;
      adaptH(fig, 340, 420);
      const NAMES = ['Pré-norm', 'mHC estático', 'mHC dinâmico', 'Gated Residual'];
      const DATA = {
        loss: { t: 'Loss final (Tab. 5) · barra = redução vs pré-norm', v: [1.617, 1.596, 1.594, 1.590], down: true, dig: 3 },
        avg: { t: 'Média de 9 benchmarks (Tab. 5) · barra = ganho vs pré-norm', v: [50.91, 52.49, 54.47, 54.66], dig: 2 },
        math: { t: 'MATH (Tab. 5) · barra = ganho vs pré-norm', v: [53.92, 55.08, 59.54, 61.18], dig: 2 },
        pro: { t: 'MMLU-Pro (Tab. 5) · barra = ganho vs pré-norm', v: [38.40, 43.69, 45.84, 46.02], dig: 2 },
      };
      const S = { m: 'avg', cur: [0, 0, 0, 0] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Métrica', options: [{ v: 'loss', label: 'loss' }, { v: 'avg', label: 'média' }, { v: 'math', label: 'MATH' }, { v: 'pro', label: 'MMLU-Pro' }, { v: 'stress', label: 'estresse (GatedNorm)' }], value: 'avg', onChange: (v) => { S.m = v; } });
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 560;
        const ml = 16, mr = 16, mt = 44, mb = narrow ? 50 : 42;
        const pw = w - ml - mr, ph = h - mt - mb;
        if (S.m === 'stress') {
          txt(g, 'Estresse a 3× o LR ótimo · AdamW · 28 camadas · só a GatedNorm muda', ml, 22, { size: 12, weight: 600, color: C.ink, maxW: pw });
          const groups = [['spikes de loss / 10k passos', 32.0, 3.2, 1], ['cruzamentos do clipping', 256, 20, 0]];
          const gw = pw / 2;
          groups.forEach(([lab, a, b, dig], k) => {
            const gx = ml + k * gw, bw = Math.min(70, gw / 3.2), mx = a;
            const tgt = [a, b];
            tgt.forEach((v, j) => {
              const key = 2 * k + j; S.cur[key] = V.lerp(S.cur[key] || 0, v / mx, 0.12);
              const hh = S.cur[key] * (ph - 30), x = gx + gw / 2 - bw - 6 + j * (bw + 12), yb = mt + ph;
              V.rr(g, x, yb - hh, bw, Math.max(1, hh), 6); g.fillStyle = V.alpha(j === 0 ? C.c : C.a, 0.85); g.fill();
              txt(g, dig ? fmt(v, 1) : String(v), x + bw / 2, yb - hh - 6, { size: 12, align: 'center', color: C.ink, font: 'mono', weight: 600 });
              txt(g, j === 0 ? 'sem gate' : 'com GatedNorm', x + bw / 2, yb + 15, { size: 11, align: 'center', color: C.muted, maxW: bw + 14 });
            });
            txt(g, lab, gx + gw / 2, mt + ph + 34, { size: 11, align: 'center', color: C.ink, maxW: gw - 10 });
          });
          return;
        }
        const D = DATA[S.m];
        txt(g, D.t, ml, 22, { size: 12, weight: 600, color: C.ink, maxW: pw });
        const base = D.v[0];
        const imp = D.v.map((v) => (D.down ? base - v : v - base));
        const mx = Math.max(...imp, 1e-9) * 1.15;
        const n = 4, gap = narrow ? 12 : 26, bw = (pw - gap * (n - 1)) / n;
        const y0 = mt + ph;
        V.line(g, ml, y0, ml + pw, y0, { color: V.alpha(C.ink, 0.4) });
        for (let k = 0; k < n; k++) {
          S.cur[k] = V.lerp(S.cur[k] || 0, imp[k] / mx, 0.12);
          const hh = S.cur[k] * (ph - 36), x = ml + k * (bw + gap);
          const col = k === 3 ? C.qwen : k === 0 ? C.muted : C.ds;
          V.rr(g, x, y0 - hh, bw, Math.max(2, hh), 6); g.fillStyle = V.alpha(col, 0.85); g.fill();
          txt(g, fmt(D.v[k], D.dig), x + bw / 2, y0 - hh - 20, { size: 12, align: 'center', color: C.ink, font: 'mono', weight: 600 });
          txt(g, k === 0 ? 'referência' : (D.down ? '−' : '+') + fmt(imp[k], D.dig), x + bw / 2, y0 - hh - 6, { size: 11, align: 'center', color: C.muted, font: 'mono' });
          txt(g, NAMES[k], x + bw / 2, y0 + 16, { size: 11.5, align: 'center', color: C.ink, maxW: bw + gap - 4 });
        }
      });
    })();

    /* ============================================ Fig 8: calculadora do resíduo largo */
    (function figBytes() {
      const fig = root.querySelector('#fig-bytes'); if (!fig) return;
      const stage3d = fig.querySelector('.la-fig-stage.t3d');
      const stages = fig.querySelectorAll('.la-fig-stage');
      const cvStage = stages[1];
      const setH = () => { const nar = fig.clientWidth < 620; const w = (nar ? 390 : 284) + 'px'; if (cvStage.style.height !== w) cvStage.style.height = w; };
      setH(); const ro0 = new ResizeObserver(setH); ro0.observe(fig); LA.onCleanup(() => ro0.disconnect());
      const MODELS = {
        pad: { name: 'resíduo clássico (d = 4096)', n: 1, d: 4096, L: 43, w: 0, wn: 'sem pesos de hyper-connection', k: 'fused_add_rms_norm (resíduo simples)' },
        glm: { name: 'GLM-5.3-Flash', n: 4, d: 4096, L: 45, w: 45 * 786486 * 4, wn: '35,4 M parâm., FP32 no vLLM (conta pelas formas)', k: 'ops mHC do vLLM (fusão post/pre)' },
        ds: { name: 'DeepSeek-V4-Flash', n: 4, d: 4096, L: 43, w: 43 * 786486 * 4, wn: '~0,79 M parâm./camada, FP32 (conta pelas formas)', k: 'TileLang · MHCFusedPostPreOp' },
        qwen: { name: 'Qwen3.8-Flash-Next', n: 4, d: 2560, L: 48, w: 1.19 * 2 ** 30, wn: '0,641 B parâm. BF16, replicados em cada GPU', k: 'Triton hc_* + GEMMs BF16 (cuBLAS)' },
      };
      const S = { m: 'ds', lt: 13, dt: 2, b12x: false };
      const cellCol = (s, i, j) => {
        const base = ['--viz-a', '--viz-b', '--viz-c', '--viz-d'][s];
        const v = Math.abs(Math.sin(i * 1.7 + j * 0.9 + s * 2.1));
        const q = S.dt === 1 ? Math.round(v * 3) / 3 : v;
        return `color-mix(in srgb, var(${base}) ${Math.round(22 + 60 * q)}%, transparent)`;
      };
      const T3 = UI.tensor3d(stage3d, { slabs: 4, rows: 6, cols: 10, cell: 13, depthGap: 72, rotate: true, ry: 46, rx: -14, color: cellCol, slabColor: (s) => ['var(--viz-a)', 'var(--viz-b)', 'var(--viz-c)', 'var(--viz-d)'][s] });
      const tag = document.createElement('div');
      tag.style.cssText = 'position:absolute;left:14px;right:14px;top:10px;font-family:var(--font-mono);font-size:12px;line-height:1.4;color:var(--text-3);pointer-events:none;';
      stage3d.appendChild(tag);
      // legenda das placas fora da pilha 3D (os rótulos presos às placas ficavam escondidos atrás delas)
      const key = document.createElement('div');
      key.style.cssText = 'position:absolute;left:14px;right:14px;bottom:10px;display:flex;flex-wrap:wrap;gap:4px 12px;font-family:var(--font-mono);font-size:12px;color:var(--text-2);pointer-events:none;';
      const keyItems = ['--viz-a', '--viz-b', '--viz-c', '--viz-d'].map((cv, s) => {
        const it = document.createElement('span');
        it.innerHTML = '<span style="display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px;background:var(' + cv + ')"></span>fluxo ' + s;
        key.appendChild(it); return it;
      });
      stage3d.appendChild(key);
      const repaint = () => {
        const T = 2 ** S.lt, nRows = Math.min(6, T);
        for (let s = 0; s < 4; s++) for (let i = 0; i < 6; i++) for (let j = 0; j < 10; j++) {
          T3.setCell(s, i, j, cellCol(s, i, j));
          const c = T3.cells[s][i][j]; const disp = i < nRows ? '' : 'none'; if (c.style.display !== disp) c.style.display = disp;
        }
        const one = MODELS[S.m].n === 1;
        T3.highlightSlab(one ? 0 : null); T3.explode(one ? 0.5 : 1);
        keyItems.forEach((it, s) => { it.style.opacity = one && s > 0 ? 0.3 : 1; });
        tag.textContent = (one ? '1 fluxo' : '4 fluxos') + ' × [' + T.toLocaleString('pt-BR') + ' tokens × ' + MODELS[S.m].d + ' canais] · ' + (S.dt === 2 ? 'BF16' : 'FP8 (menos níveis)') + ' · cada linha é um token' + (T > 6 ? ' (desenhadas 6)' : '');
      };
      repaint();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'pad', label: '1 fluxo' }, { v: 'glm', label: 'GLM' }, { v: 'ds', label: 'DeepSeek' }, { v: 'qwen', label: 'Qwen' }], value: 'ds', onChange: (v) => { S.m = v; repaint(); H.redraw(); } });
      UI.slider(ctl, { label: 'Tokens no passo', min: 0, max: 13, step: 1, value: 13, fmt: (v) => String(2 ** v).replace(/\B(?=(\d{3})+(?!\d))/g, '.'), onInput: (v) => { S.lt = v; repaint(); H.redraw(); } });
      UI.seg(ctl, { label: 'Estado em', options: [{ v: 2, label: 'BF16' }, { v: 1, label: 'FP8' }], value: 2, onChange: (v) => { S.dt = v; repaint(); H.redraw(); } });
      UI.toggle(ctl, { label: 'VLLM_USE_B12X_MHC=1 (Stage-C)', value: false, onChange: (v) => { S.b12x = v; H.redraw(); } });
      const bytes = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0; while (b >= 1024 && i < 4) { b /= 1024; i++; } return (i ? fmt(b, b < 10 ? 2 : b < 100 ? 1 : 0) : Math.round(b)) + ' ' + u[i]; };
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const M = MODELS[S.m], T = 2 ** S.lt;
        const narrow = w < 620;
        const pad = 16, pw = w - 2 * pad;
        const state = T * M.n * M.d * S.dt, single = T * M.d * S.dt;
        const traffic = state * 4 * M.L;
        const shape = '[' + T.toLocaleString('pt-BR') + (M.n > 1 ? ', ' + M.n : '') + ', ' + M.d + '] ' + (S.dt === 2 ? 'BF16' : 'FP8');
        txt(g, M.name + ' · estado do passo ' + shape, pad, 16, { size: 12, weight: 600, color: C.ink, font: 'mono', maxW: pw });
        // barra log do estado
        const lo = 10, hi = 30; // 1 KiB .. 1 GiB
        const bx = pad, bw = pw, by = 42, bh = 16;
        const X = (b) => bx + bw * (V.clamp(Math.log2(Math.max(b, 1)), lo, hi) - lo) / (hi - lo);
        V.rr(g, bx, by, bw, bh, 6); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
        V.rr(g, bx, by, Math.max(4, X(state) - bx), bh, 6); g.fillStyle = V.alpha(M.n > 1 ? C.a : C.muted, 0.8); g.fill();
        if (M.n > 1) { const xs = X(single); V.line(g, xs, by - 3, xs, by + bh + 3, { color: C.ink, lw: 2 }); }
        const x256 = X(256 * 2 ** 20);
        V.line(g, x256, by - 6, x256, by + bh + 6, { color: C.c, lw: 1.6, dash: [3, 3] });
        ['1 KiB', '1 MiB', '1 GiB'].forEach((l, k) => txt(g, l, bx + bw * (k * 10) / 20, by + bh + 14, { size: 11, align: k === 0 ? 'left' : k === 2 ? 'right' : 'center', color: C.muted, font: 'mono' }));
        txt(g, '256 MiB', x256, by - 9, { size: 11, align: 'center', color: C.c, font: 'mono' });
        // cartões
        const cards = [
          { k: 'estado de um passo', v: bytes(state), s: M.n > 1 ? '1 fluxo seria ' + bytes(single) + ' (traço preto)' : 'referência de 1 fluxo', col: C.a },
          { k: 'tráfego esquemático', v: bytes(traffic), s: '~4 passadas × ' + M.L + ' camadas · ≈ ' + fmt(traffic / 240e9 * 1000, traffic / 240e9 * 1000 < 10 ? 2 : 1) + ' ms a 240 GB/s', col: C.b },
          { k: 'pesos HC lidos/passo/GPU', v: M.w ? bytes(M.w) : '0', s: M.wn, col: C.d },
        ];
        const cy0 = by + bh + 30;
        const cols = narrow ? 1 : 3, cgap = 12;
        const cw = (pw - cgap * (cols - 1)) / cols, chh = narrow ? 60 : 86;
        cards.forEach((c, k) => {
          const cx = pad + (k % cols) * (cw + cgap), cy = cy0 + Math.floor(k / cols) * (chh + 8);
          V.box(g, cx, cy, cw, chh, { r: 10, fill: V.alpha(c.col, 0.08), stroke: V.alpha(c.col, 0.45), lw: 1.2 });
          txt(g, c.k, cx + 10, cy + 16, { size: 11, color: C.muted, maxW: cw - 20 });
          txt(g, c.v, cx + 10, cy + (narrow ? 38 : 42), { size: narrow ? 17 : 20, font: 'display', weight: 700, color: C.ink, maxW: narrow ? cw * 0.4 : cw - 20 });
          if (narrow) wrapText(g, c.s, cx + cw * 0.42, cy + 35, cw * 0.56, 13, { size: 11, color: C.muted });
          else wrapText(g, c.s, cx + 10, cy + 60, cw - 20, 13, { size: 11, color: C.muted });
        });
        // semáforo de kernel
        const sy = cy0 + (narrow ? 3 * (chh + 8) : chh + 8) + 8;
        let kname = M.k, kcol = C.d, note = '';
        if (S.m === 'ds' && S.b12x) {
          if (T <= 16) { kname = 'b12x · b12x_mhc_post_pre (CuTeDSL, SM120)'; kcol = C.a; note = T + ' tokens ≤ B12X_MHC_MAX_TOKENS = 16'; } else { note = T.toLocaleString('pt-BR') + ' tokens > 16 → fallback TileLang'; }
        } else if (S.m === 'ds') note = 'lane padrão: VLLM_USE_B12X_MHC = 0';
        else if (S.b12x) note = 'o b12x mHC só existe no overlay Stage-C do DeepSeek';
        V.box(g, pad, sy, pw, 42, { r: 21, fill: V.alpha(kcol, 0.1), stroke: V.alpha(kcol, 0.6), lw: 1.3 });
        V.glow(g, pad + 21, sy + 21, 14, kcol, 0.9); V.dot(g, pad + 21, sy + 21, 6, kcol);
        txt(g, kname, pad + 40, sy + (note ? 17 : 25), { size: 12, weight: 600, color: C.ink, font: 'mono', maxW: pw - 52 });
        if (note) txt(g, note, pad + 40, sy + 32, { size: 11, color: C.muted, font: 'mono', maxW: pw - 52 });
        if (S.m === 'ds' && S.b12x && sy + 64 < h) {
          const ax0 = pad + 40, ax1 = w - pad - 10, ay = sy + 60;
          V.line(g, ax0, ay, ax1, ay, { color: C.grid, lw: 2 });
          const XT = (lt) => ax0 + (lt / 13) * (ax1 - ax0);
          g.fillStyle = V.alpha(C.a, 0.35); g.fillRect(ax0, ay - 3, XT(4) - ax0, 6);
          V.line(g, XT(4), ay - 7, XT(4), ay + 7, { color: C.a, lw: 2 });
          V.dot(g, XT(S.lt), ay, 5, kcol);
          txt(g, '16', XT(4), ay + 18, { size: 11, align: 'center', color: C.a, font: 'mono' });
        }
      });
    })();

    /* ============================================ Fig 8b: a cola em dois Sparks (GLM·TF) */
    (function figCola() {
      const fig = root.querySelector('#fig-cola'); if (!fig) return;
      adaptH(fig, 420, 640);
      const S = { mode: 'ov', lr: 11 };
      // Um ponto de troca visto pelo Spark 0, em unidades ILUSTRATIVAS: R linhas = 1 unidade por etapa
      // (parcial, cola, frente, fio fp32); a linha BF16 custa meia unidade porque tem metade dos bytes.
      // pc = pedaço de linhas (0/1) na sobreposição; sem pc = todas as linhas daquela etapa.
      const PLANS = {
        rep: { end: 4, main: [{ a: 0, b: 1, k: 'part' }, { a: 2, b: 3, k: 'glue', rows: 'all' }, { a: 3, b: 4, k: 'front' }], wire: [{ a: 1, b: 2, k: 'ag' }] },
        split: { end: 3.25, main: [{ a: 0, b: 1, k: 'part' }, { a: 1.5, b: 2, k: 'glue' }, { a: 2.25, b: 3.25, k: 'front' }], wire: [{ a: 1, b: 1.5, k: 'f32' }, { a: 2, b: 2.25, k: 'b16' }] },
        ov: { end: 2.5, main: [{ a: 0, b: 0.25, k: 'part' }, { a: 0.25, b: 0.5, k: 'part' }, { a: 0.5, b: 1, k: 'part' }, { a: 1, b: 1.25, k: 'glue', pc: 0 }, { a: 1.25, b: 1.5, k: 'front' }, { a: 1.5, b: 1.75, k: 'glue', pc: 1 }, { a: 1.75, b: 2, k: 'front' }, { a: 2, b: 2.5, k: 'front' }], wire: [{ a: 0.25, b: 0.5, k: 'f32', pc: 0 }, { a: 0.5, b: 0.75, k: 'f32', pc: 1 }, { a: 1.25, b: 1.375, k: 'b16', pc: 0 }, { a: 1.75, b: 1.875, k: 'b16', pc: 1 }] },
      };
      const LAB = { part: 'parcial', glue: 'cola', front: 'frente', ag: 'all-gather fp32', f32: 'fp32', b16: 'bf16' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cola', options: [{ v: 'rep', label: 'replicada' }, { v: 'split', label: 'dividida' }, { v: 'ov', label: 'dividida + sobreposição' }], value: 'ov', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'Linhas no chunk', min: 4, max: 11, step: 1, value: 11, fmt: (v) => (2 ** v).toLocaleString('pt-BR'), onInput: (v) => { S.lr = v; } });
      const rSend = UI.readout(ctl, 'envio por rank, por ponto', '—');
      const rGlue = UI.readout(ctl, 'cola por rank', '—');
      const rChunk = UI.readout(ctl, 'por chunk (90 pontos)', '—');
      const bytesF = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB']; let i = 0; while (b >= 1024 && i < 3) { b /= 1024; i++; } return (i ? fmt(b, b < 10 ? 2 : b < 100 ? 1 : 0) : Math.round(b)) + ' ' + u[i]; };
      let lastKey = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const R = 2 ** S.lr;
        const fallback = S.mode !== 'rep' && R < 64;
        const pm = fallback ? 'rep' : S.mode;
        const P = PLANS[pm];
        const key = S.mode + R;
        if (key !== lastKey) {
          lastKey = key;
          const send = pm === 'rep' ? R * 16384 : R * 12288;
          rSend.set(bytesF(send), pm === 'rep' ? '' : 'ok');
          rGlue.set((pm === 'rep' ? R : Math.ceil(R / 2)).toLocaleString('pt-BR') + ' linhas', pm === 'rep' ? '' : 'ok');
          rChunk.set(bytesF(send * 90));
        }
        const COL = { part: C.muted, glue: C.c, front: C.b, ag: C.d, f32: C.d, b16: C.e };
        const pad = 16, pw = w - 2 * pad;
        // relógio: percorre as 4 unidades do caminho replicado e para um instante no fim
        const tau = Math.min(4, (t * 0.7) % 5);
        // cabeçalho
        const Rs = R.toLocaleString('pt-BR');
        let head = {
          rep: 'Replicada: os dois Sparks juntam as parciais inteiras e fazem a mesma cola nas ' + Rs + ' linhas.',
          split: 'Dividida (SPLIT=1): cada Spark recebe só a parcial das suas linhas, faz a cola da sua metade e devolve as linhas prontas.',
          ov: 'Dividida + sobreposição (nível 2): trocas em 2 pedaços num segundo stream; a frente do bloco seguinte roda enquanto as linhas viajam.',
        }[S.mode];
        if (fallback) head = 'Chunk de ' + Rs + ' linhas: abaixo de 64 (TF_GLM_HC_SPLIT_MIN_ROWS) a receita não divide, e a cola fica replicada.' + (R <= 32 ? ' Janelas de decode (até 16 linhas por stream) nunca dividem.' : '');
        const nH = wrapText(g, head, pad, 18, pw, 15, { size: 11.5, weight: 600, color: fallback ? C.d : C.ink });
        // ---------------------------------------------------------- linhas nos dois Sparks
        const lw = narrow ? pw : Math.min(260, w * 0.34);
        const py0 = 18 + nH * 15 + 26;
        const py1 = narrow ? py0 + 190 : h - 50;
        const colW = narrow ? Math.min(84, pw * 0.26) : Math.min(70, (lw - 70) / 2);
        const cx = [pad + 2, pad + lw - colW - 2];
        const nShow = Math.min(R, 16), half = nShow / 2;
        const rh = (py1 - py0) / nShow;
        const owner = (i) => (i < half ? 0 : 1);
        const pieceOf = (i) => ((i % half) < half / 2 ? 0 : 1);
        const segAt = (list) => list.find((s) => tau >= s.a && tau < s.b) || null;
        const doneFrac = (s) => V.clamp((tau - s.a) / (s.b - s.a), 0, 1);
        // estado de uma linha num Spark: 0..1 de cola feita aqui, ou 'chegou' do par
        function glueFrac(i) {
          let f = 0;
          P.main.forEach((s) => { if (s.k !== 'glue') return; if (s.pc === undefined || s.pc === pieceOf(i)) f = Math.max(f, doneFrac(s)); });
          return f;
        }
        function arrived(i) {
          return P.wire.some((s) => s.k === 'b16' && (s.pc === undefined || s.pc === pieceOf(i)) && tau >= s.b);
        }
        for (let k = 0; k < 2; k++) {
          for (let i = 0; i < nShow; i++) {
            const y = py0 + i * rh;
            const mine = pm === 'rep' || owner(i) === k;
            V.rr(g, cx[k], y + 1, colW, Math.max(2, rh - 2), 3);
            if (mine) {
              const f = glueFrac(i);
              g.fillStyle = V.alpha(C.muted, 0.14); g.fill();
              if (f > 0) { V.rr(g, cx[k], y + 1, colW * f, Math.max(2, rh - 2), 3); g.fillStyle = V.alpha(C.c, 0.6); g.fill(); }
            } else if (arrived(i)) { g.fillStyle = V.alpha(C.e, 0.4); g.fill(); }
            else { g.save(); g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.stroke(); g.restore(); }
          }
          if (pm !== 'rep') {
            // marca de posse: metade de cima é do Spark 0, a de baixo do Spark 1
            const yo = py0 + (k === 0 ? 0 : half * rh), x = k === 0 ? cx[0] - 6 : cx[1] + colW + 6;
            V.line(g, x, yo + 2, x, yo + half * rh - 2, { color: C.c, lw: 2.5 });
          }
          const ly = py1 + 16;
          txt(g, 'Spark ' + k, cx[k] + colW / 2, ly, { size: 12, weight: 700, align: 'center', color: C.ink });
          const gl = pm === 'rep' ? R : Math.ceil(R / 2);
          txt(g, 'cola: ' + gl.toLocaleString('pt-BR') + ' linhas', k === 0 ? cx[0] : cx[1] + colW, ly + 15, { size: 11, align: k === 0 ? 'left' : 'right', color: C.muted, font: 'mono', maxW: narrow ? pw / 2 - 8 : lw / 2 - 4 });
        }
        // pacotes no fio, sincronizados com a linha do tempo
        const ws = segAt(P.wire);
        const gx0 = cx[0] + colW + 4, gx1 = cx[1] - 4;
        if (ws) {
          const s = doneFrac(ws);
          const lab = { ag: 'all-gather: parcial inteira (fp32)', f32: 'parcial fp32 das linhas do par', b16: 'linhas normalizadas (bf16)' }[ws.k];
          txt(g, lab, (gx0 + gx1) / 2, py0 - 9, { size: 11, align: 'center', color: COL[ws.k], weight: 600, maxW: narrow ? pw : lw + 30 });
          for (let i = 0; i < nShow; i++) {
            if (ws.pc !== undefined && pieceOf(i) !== ws.pc) continue;
            const y = py0 + (i + 0.5) * rh;
            // direção: a parcial vai para o dono da linha; a linha pronta sai do dono
            let dirs;
            if (ws.k === 'ag') dirs = [1, -1];
            else if (ws.k === 'f32') dirs = [owner(i) === 0 ? -1 : 1];
            else dirs = [owner(i) === 0 ? 1 : -1];
            dirs.forEach((d, j) => {
              const x = d > 0 ? V.lerp(gx0, gx1, s) : V.lerp(gx1, gx0, s);
              V.dot(g, x, y + (dirs.length > 1 ? (j ? 2.5 : -2.5) : 0), ws.k === 'b16' ? 3 : 4, COL[ws.k]);
            });
          }
        }
        // ---------------------------------------------------------- linha do tempo do Spark 0
        const tx0 = narrow ? pad + 34 : pad + lw + 74, tx1 = w - pad - 4;
        const ty = narrow ? py1 + 62 : py0;
        const X = (u) => tx0 + (u / 4) * (tx1 - tx0);
        txt(g, 'Spark 0 num ponto de troca · tempo ilustrativo', narrow ? pad : tx0 - 50, ty - 6, { size: 11.5, weight: 600, color: C.ink, maxW: narrow ? pw : tx1 - tx0 + 50 });
        const laneY = [ty + 12, ty + 80], laneH = [54, 36];
        txt(g, 'GPU', tx0 - 8, laneY[0] + laneH[0] / 2 + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        txt(g, 'fio', tx0 - 8, laneY[1] + laneH[1] / 2 + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        V.line(g, tx0, laneY[1] + laneH[1] + 6, tx1, laneY[1] + laneH[1] + 6, { color: C.grid, lw: 1 });
        function drawSeg(s, lane) {
          const x = X(s.a), ww = X(s.b) - X(s.a), y = laneY[lane], hh = laneH[lane], col = COL[s.k];
          const f = doneFrac(s);
          V.rr(g, x + 0.5, y, Math.max(2, ww - 1), hh, 4); g.fillStyle = V.alpha(col, f >= 1 ? 0.75 : 0.2); g.fill();
          if (f < 1) { g.strokeStyle = V.alpha(col, 0.75); g.lineWidth = 1.2; g.stroke(); }
          if (f > 0 && f < 1) { V.rr(g, x + 0.5, y, Math.max(2, (ww - 1) * f), hh, 4); g.fillStyle = V.alpha(col, 0.75); g.fill(); }
          const lab = LAB[s.k];
          if (tw(g, lab, 11, 'mono') + 8 < ww) txt(g, lab, x + ww / 2, y + hh / 2 + 4, { size: 11, align: 'center', color: f >= 1 ? onInk() : C.ink, font: 'mono' });
        }
        // esperas no stream principal
        const mains = P.main.slice().sort((p, q) => p.a - q.a);
        let prev = 0;
        mains.forEach((s) => {
          if (s.a - prev > 1e-6) {
            const x = X(prev), ww = X(s.a) - X(prev);
            g.save(); g.setLineDash([4, 3]); g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1;
            V.rr(g, x + 1, laneY[0] + 4, ww - 2, laneH[0] - 8, 4); g.stroke(); g.restore();
            if (tw(g, 'espera', 11) + 6 < ww) txt(g, 'espera', x + ww / 2, laneY[0] + laneH[0] / 2 + 4, { size: 11, align: 'center', color: C.muted });
          }
          prev = Math.max(prev, s.b);
        });
        mains.forEach((s) => drawSeg(s, 0));
        P.wire.forEach((s) => drawSeg(s, 1));
        // fim deste modo e fim do caminho replicado
        const ey = laneY[1] + laneH[1] + 22;
        const xe = X(P.end);
        V.line(g, xe, laneY[0] - 4, xe, laneY[1] + laneH[1] + 8, { color: C.ink, lw: 1.5, dash: [4, 3] });
        txt(g, 'fim', xe, ey, { size: 11, align: P.end >= 4 ? 'right' : 'center', color: C.ink, weight: 700 });
        if (pm !== 'rep') {
          const xr = X(4);
          V.line(g, xr, laneY[0] - 4, xr, laneY[1] + laneH[1] + 23, { color: C.muted, lw: 1, dash: [2, 4] });
          txt(g, 'fim replicado', xr, ey + 15, { size: 11, align: 'right', color: C.muted });
        }
        // cursor do tempo
        const xp = X(tau);
        V.line(g, xp, laneY[0] - 8, xp, laneY[1] + laneH[1] + 4, { color: C.a, lw: 2 });
        V.dot(g, xp, laneY[0] - 8, 4, C.a);
        // legenda
        const LEG = [['part', 'parcial fp32: fim do bloco'], ['glue', 'cola: hc_post + próximo hc_pre'], ['front', 'frente do bloco seguinte'], ['f32', 'fio: parcial fp32 (16 KiB/linha)'], ['b16', 'fio: linha normalizada bf16 (8 KiB)']];
        const lx0 = narrow ? pad : tx0 - 50, lwid = narrow ? pw : tx1 - lx0;
        const lcols = lwid > 420 ? 2 : 1, lcw = lwid / lcols, ly0 = ey + (pm !== 'rep' ? 35 : 20);
        LEG.forEach(([k, l], j) => {
          const x = lx0 + (j % lcols) * lcw, y = ly0 + Math.floor(j / lcols) * 18;
          if (y + 6 > h) return;
          V.rr(g, x, y - 9, 12, 12, 3); g.fillStyle = V.alpha(COL[k], 0.8); g.fill();
          txt(g, l, x + 18, y + 1, { size: 11, color: C.ink, maxW: lcw - 24 });
        });
        const ny = ly0 + Math.ceil(LEG.length / lcols) * 18 + 6;
        if (ny + 4 < h) wrapText(g, 'O Spark 1 faz o espelho, dono das linhas de baixo. Esperas tracejadas: a GPU aguardando o fio.', lx0, ny, lwid, 14, { size: 11, color: C.muted });
      });
    })();

    /* ============================================ Fig 9: bytes do passo do Qwen */
    (function figQwenBytes() {
      const fig = root.querySelector('#fig-qwen-bytes'); if (!fig) return;
      adaptH(fig, 380, 520);
      const SEG = [
        { k: 'lm_head ×4', v: 2.36, f: 1.18, c: 'c', note: 'verificação + 3 passes do rascunho' },
        { k: 'experts NVFP4', v: 2.1, f: 2.1, c: 'a', note: '~32 experts/camada, EP' },
        { k: 'projeções GDN', v: 1.95, f: 0.975, c: 'b', note: 'fatiado: metade de 3,89 GiB' },
        { k: 'HyperConnections', v: 1.19, f: 0.595, c: 'd', note: 'replicado: 1,19 GiB inteiras', hc: true },
        { k: 'atenção + indexador', v: 0.60, f: 0.32, c: 'f', note: 'indexador fica BF16' },
        { k: 'MTP (densos + experts)', v: 0.5, f: 0.5, c: 'purple', note: 'excluído do FP8-dense' },
        { k: 'roteador + shared + PLE', v: 0.40, f: 0.29, c: 'muted', note: 'só o shared expert vira FP8' },
        { k: 'estado GDN', v: 0.1, f: 0.1, c: 'grid', note: '36 × 1,5 MiB FP32, R+W' },
      ];
      const S = { fp8: false, tp: false, cur: SEG.map((s) => s.v) };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'FP8-dense (hyperconnection.diff + model.diff)', value: false, onChange: (v) => { S.fp8 = v; } });
      UI.toggle(ctl, { label: 'comparar com 1 GPU (sem TP)', value: false, onChange: (v) => { S.tp = v; } });
      const rB = UI.readout(ctl, 'bytes/passo/GPU', '9,20 GiB');
      const rT = UI.readout(ctl, 'passo a 240 GB/s', '41 ms');
      const rS = UI.readout(ctl, 'teto tok/s (≈3,2/passo)', '≈ 78');
      let lastTot = -1;
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 620;
        const pad = 16, pw = w - 2 * pad;
        SEG.forEach((s, k) => { const tgt = S.fp8 ? s.f : s.v; S.cur[k] = V.lerp(S.cur[k], tgt, 0.12); });
        const tot = S.cur.reduce((a, b) => a + b, 0);
        if (Math.abs(tot - lastTot) > 0.005) {
          lastTot = tot;
          const ms = (tot * 2 ** 30) / 240e9 * 1000;
          rB.set(fmt(tot, 2) + ' GiB'); rT.set(fmt(ms, 1) + ' ms'); rS.set('≈ ' + Math.round((1000 / ms) * 3.2));
        }
        const sc = pw / 9.2;
        const colOf = (s) => (s.c === 'muted' ? C.muted : s.c === 'grid' ? V.alpha(C.ink, 0.25) : s.c === 'ink' ? V.alpha(C.ink, 0.6) : C[s.c]);
        txt(g, 'bytes lidos da LPDDR5x por passo de decode (TP2/EP2, batch 1, MTP3)', pad, 20, { size: 12, weight: 600, color: C.ink, maxW: pw });
        const rowH = narrow ? 34 : 40;
        ['GPU 0', 'GPU 1'].forEach((lab, r) => {
          const y = 44 + r * (rowH + 22);
          txt(g, lab, pad, y - 4, { size: 11, color: C.muted, font: 'mono' });
          let x = pad;
          SEG.forEach((s, k) => {
            const ww = S.cur[k] * sc;
            V.rr(g, x, y, Math.max(1, ww - 2), rowH, 5); g.fillStyle = V.alpha(colOf(s), s.hc ? 0.95 : 0.7); g.fill();
            if (s.hc) { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, x, y, Math.max(1, ww - 2), rowH, 5); g.stroke(); }
            const lab = fmt(S.cur[k], 2), lw0 = tw(g, lab, 11, 'mono', 600);
            if (ww > Math.max(44, lw0 + 8)) txt(g, lab, x + (ww - 2) / 2, y + rowH / 2 + 4, { size: 11, align: 'center', color: onInk(), font: 'mono', weight: 600 });
            else if (s.hc) {
              // o foco da figura: rótulo por fora da barra, com um traço
              const cx = x + (ww - 2) / 2;
              V.line(g, cx, y + rowH, cx, y + rowH + 5, { color: C.d, lw: 1.5 });
              txt(g, lab + ' HC', cx, y + rowH + 16, { size: 11, align: 'center', color: C.d, font: 'mono', weight: 700 });
            }
            x += ww;
          });
          V.line(g, pad + 9.2 * sc, y - 6, pad + 9.2 * sc, y + rowH + 6, { color: V.alpha(C.ink, 0.4), dash: [3, 3] });
        });
        // legenda com notas (ou a comparação fatiado × replicado)
        const ly0 = 44 + 2 * (rowH + 22) + 6;
        if (S.tp) {
          const kG = S.cur[2] / 1.95; // FP8-dense também encolhe o GDN
          txt(g, 'o mesmo tensor lido por 1 GPU sozinha × por cada GPU com TP2', pad, ly0 + 10, { size: 11.5, color: C.ink, weight: 600, maxW: pw });
          const lab0 = narrow ? 104 : 150, bmax = pw - lab0 - 70, sc2 = bmax / 3.89;
          const groups = [
            { k: 'projeções GDN', col: C.b, full: 3.89 * kG, per: S.cur[2], note: 'fatiado: cada GPU guarda metade das linhas' },
            { k: 'HyperConnections', col: C.d, full: S.cur[3], per: S.cur[3], note: 'replicado (disable_tp=True): cada GPU lê tudo' },
          ];
          groups.forEach((gr, k) => {
            const gy = ly0 + 30 + k * (narrow ? 84 : 78);
            txt(g, gr.k, pad, gy, { size: 11.5, color: gr.col, weight: 700 });
            txt(g, gr.note, pad + (narrow ? 0 : 130), narrow ? gy + 15 : gy, { size: 11, color: C.muted, maxW: narrow ? pw : pw - 130 });
            [['1 GPU, sem TP', gr.full, 0.35], ['cada GPU, TP2', gr.per, 0.9]].forEach(([l, v, a], j) => {
              const by = gy + (narrow ? 24 : 10) + j * 24;
              txt(g, l, pad + lab0 - 8, by + 13, { size: 11, align: 'right', color: C.muted });
              V.rr(g, pad + lab0, by, Math.max(2, v * sc2), 18, 4); g.fillStyle = V.alpha(gr.col, a); g.fill();
              if (j === 0) { g.save(); g.setLineDash([4, 3]); g.strokeStyle = gr.col; g.lineWidth = 1.2; V.rr(g, pad + lab0, by, Math.max(2, v * sc2), 18, 4); g.stroke(); g.restore(); }
              txt(g, fmt(v, 2) + ' GiB', pad + lab0 + v * sc2 + 6, by + 13, { size: 11, color: C.ink, font: 'mono', weight: 600 });
            });
          });
          return;
        }
        const lcols = narrow ? 1 : 2, lw2 = pw / lcols, lh = narrow ? 30 : 34;
        SEG.forEach((s, k) => {
          const cx = pad + (k % lcols) * lw2, cy = ly0 + Math.floor(k / lcols) * lh;
          if (cy + 20 > h) return;
          V.rr(g, cx, cy, 12, 12, 3); g.fillStyle = V.alpha(colOf(s), 0.85); g.fill();
          txt(g, s.k + ' · ' + fmt(s.v, 2) + (S.fp8 && s.f !== s.v ? ' → ' + fmt(s.f, 2) : '') + ' GiB', cx + 18, cy + 10, { size: 11, color: s.hc ? C.d : C.ink, weight: s.hc ? 700 : 500, maxW: lw2 - 24 });
          txt(g, s.note, cx + 18, cy + 23, { size: 11, color: C.muted, maxW: lw2 - 24 });
        });
      });
    })();

    /* ============================================ Fig 10: o gancho do rascunhador */
    (function figGancho() {
      const fig = root.querySelector('#fig-gancho'); if (!fig) return;
      adaptH(fig, 452, 650, 640);
      const S = { mode: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Caso', options: [{ v: 'glm', label: 'GLM · EAGLE3/DFlash2' }, { v: 'std', label: 'DS · padrão' }, { v: 'fused', label: 'DS · diferida' }, { v: 'exact', label: 'DS · diferida exata' }], value: 'glm', onChange: (v) => { S.mode = v; } });
      const rk = UI.readout(ctl, 'custo extra (esquemático)', '');
      // exemplo numérico de um canal
      const r = V.rng(404);
      const Hc = [0.62, -0.35, 1.08, 0.21].map((v) => v + (r() - 0.5) * 0.02);
      const yv = 0.47, post = [1.21, 0.64, 0.93, 1.48];
      const comb = sinkhorn([0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => (i === j ? 1.8 : 0) + r.normal() * 0.6)));
      const Hn = [0, 1, 2, 3].map((k) => Hc.reduce((a, hv, i) => a + comb[i][k] * hv, 0) + post[k] * yv);
      const meanExact = Hn.reduce((a, b) => a + b, 0) / 4;
      const factored = yv * post.reduce((a, b) => a + b, 0) / 4 + Hc.reduce((a, hv, i) => a + hv * comb[i].reduce((p, q) => p + q, 0), 0) / 4;
      const bf = (x) => (LA.num && LA.num.encode ? LA.num.encode('bf16', x).v : Math.round(x * 256) / 256);
      const meanFused = Hn.map(bf).reduce((a, b) => a + b, 0) / 4;
      const COST = { glm: '+5 hc_post à parte', std: '2 fusões quebradas (40→41, 41→42)', fused: '+1 cópia por alvo (40, 41)', exact: '+1 kernel por alvo (relê o estado)' };
      let lastMode = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), cols = SC(C);
        const narrow = w < 640;
        if (lastMode !== S.mode) { rk.set(COST[S.mode]); lastMode = S.mode; }
        const glm = S.mode === 'glm';
        const nL = glm ? 45 : 43, taps = glm ? [5, 14, 24, 33, 42] : [40, 41, 42];
        const pad = 16, pw = w - 2 * pad;
        // faixa de camadas
        const nLines = wrapText(g, glm ? 'GLM-5.3-Flash · 45 camadas mHC · taps nas saídas de 5, 14, 24, 33, 42 → log (6, 15, 25, 34, 43)' : 'DeepSeek-V4-Flash · 43 camadas mHC (0–42) · alvos DSpark 40, 41, 42' + (S.mode === 'std' ? '' : ' · 40 e 41 capturadas pela seguinte, 42 (última) no caminho padrão'), pad, 18, pw, 15, { size: 11.5, weight: 600, color: C.ink });
        const ty = 30 + (nLines - 1) * 15, th = 22, tw2 = pw / nL;
        const tokPos = (t * 0.12) % 1;
        for (let l = 0; l < nL; l++) {
          const x = pad + l * tw2, on = taps.includes(l);
          V.rr(g, x + 1, ty, Math.max(1, tw2 - 2), th, 2); g.fillStyle = on ? C.c : V.alpha(C.muted, 0.25); g.fill();
          if (on && tw2 >= 14) txt(g, String(l), x + tw2 / 2, ty + th + 13, { size: 11, align: 'center', color: C.c, font: 'mono' });
        }
        if (tw2 < 14) {
          // taps estreitos: rótulos agrupados para não se sobreporem
          const groups = []; taps.forEach((l) => { const gq = groups[groups.length - 1]; if (gq && l - gq[gq.length - 1] <= 2) gq.push(l); else groups.push([l]); });
          groups.forEach((gq) => { const a = gq[0], b = gq[gq.length - 1]; txt(g, a === b ? String(a) : a + '–' + b, pad + ((a + b + 1) / 2) * tw2, ty + th + 13, { size: 11, align: 'center', color: C.c, font: 'mono' }); });
        }
        V.glow(g, pad + tokPos * pw, ty + th / 2, 9, C.a, 0.9);
        // pipeline
        const boxes = {
          glm: [['camada ℓ devolve', '(y, H, post, comb)', C.muted], ['layer.hc_post(…)', "H' [T,4,4096] (cópia)", C.f], ['hc_contract', 'média → [T, 4096]', C.a], ['sp_all_gather', 'se sequence parallel', C.b], ['aux_hidden_states', '5 × [T,4096] → DFlash2', C.c]],
          std: [['camada 40/41/42', '(y, H, post, comb)', C.muted], ['layer.hc_post(…)', "H' materializado", C.f], ['.mean(dim=1)', '→ fatia [T, 4096]', C.a], ['residual = None', '41 e 42: hc_pre isolado ✕', C.red || C.c], ['_dspark_hidden_buffer', '[8192, 12288] → main_proj', C.c]],
          fused: [['camada 40/41 devolve', '(y, H, post, comb)', C.muted], ['ℓ+1: post+pre fundido', "materializa H' ✓ fusão", C.e], ['residual.mean(dim=1)', 'H\' já em BF16', C.a], ['copy_ na fatia', '[T, 4096]', C.b], ['_dspark_hidden_buffer', '[8192, 12288] → main_proj', C.c]],
          exact: [['camada 40/41 devolve', '(y, H, post, comb)', C.muted], ['dspark_hc_post_mean', 'Triton, FP32, relê x, H, post, comb', C.f], ['sem materializar', 'escreve só [T, 4096]', C.a], ['fused post+pre segue', 'cadeia intacta ✓', C.e], ['_dspark_hidden_buffer', '[8192, 12288] → main_proj', C.c]],
        }[S.mode];
        const py = ty + th + 32;
        const n = boxes.length;
        let pos;
        if (!narrow) { const bw = (pw - 18 * (n - 1)) / n; pos = boxes.map((b, k) => ({ x: pad + k * (bw + 18), y: py, w: bw, h: 58 })); }
        else { const bh = 40; pos = boxes.map((b, k) => ({ x: pad + (k % 2) * 24, y: py + k * (bh + 12), w: pw - 24, h: bh })); }
        pos.forEach((p, k) => {
          const [a, b, col] = boxes[k];
          V.box(g, p.x, p.y, p.w, p.h, { r: 10, fill: V.alpha(col, 0.13), stroke: col, lw: 1.4 });
          const two = !narrow && tw(g, b, 11) > p.w - 10;
          txt(g, a, p.x + p.w / 2, p.y + (narrow ? 16 : two ? 19 : 24), { size: 11.5, align: 'center', color: C.ink, weight: 600, font: 'mono', maxW: p.w - 10 });
          if (!two) txt(g, b, p.x + p.w / 2, p.y + (narrow ? 31 : 42), { size: 11, align: 'center', color: C.muted, maxW: p.w - 10 });
          else {
            // quebra a legenda em duas linhas no espaço mais perto do meio
            const ws = b.split(' '); let best = 1, bd = Infinity;
            for (let q = 1; q < ws.length; q++) { const dd = Math.abs(tw(g, ws.slice(0, q).join(' '), 11) - tw(g, ws.slice(q).join(' '), 11)); if (dd < bd) { bd = dd; best = q; } }
            txt(g, ws.slice(0, best).join(' '), p.x + p.w / 2, p.y + 35, { size: 11, align: 'center', color: C.muted, maxW: p.w - 8 });
            txt(g, ws.slice(best).join(' '), p.x + p.w / 2, p.y + 49, { size: 11, align: 'center', color: C.muted, maxW: p.w - 8 });
          }
          if (k < n - 1) {
            const q = pos[k + 1];
            const a0 = narrow ? [p.x + p.w / 2, p.y + p.h] : [p.x + p.w, p.y + p.h / 2], a1 = narrow ? [q.x + q.w / 2, q.y] : [q.x, q.y + q.h / 2];
            V.arrow(g, a0[0], a0[1], a1[0], a1[1], { color: C.muted, lw: 1.6, head: 7 });
            const s = (t * 0.8 + k * 0.2) % 1; V.dot(g, V.lerp(a0[0], a1[0], s), V.lerp(a0[1], a1[1], s), 3, C.a);
          }
        });
        // exemplo numérico
        const last = pos[n - 1];
        const ey = narrow ? last.y + last.h + 26 : py + 58 + 36;
        if (ey + 60 > h) return;
        txt(g, 'um canal de um token: os 4 fluxos depois do post e a média entregue', pad, ey, { size: 11.5, color: C.ink, weight: 600, maxW: pw });
        const cw = Math.min(64, (pw - 180) / 4), cy = ey + 14;
        Hn.forEach((v, k) => {
          const x = pad + k * (cw + 6);
          V.box(g, x, cy, cw, 30, { r: 7, fill: V.alpha(cols[k], 0.22), stroke: cols[k], lw: 1.2 });
          txt(g, fmt(v, 4), x + cw / 2, cy + 19, { size: 11, align: 'center', color: C.ink, font: 'mono', maxW: cw - 4 });
          txt(g, "H'" + k, x + cw / 2, cy + 44, { size: 11, align: 'center', color: cols[k], font: 'mono' });
        });
        const mx = pad + 4 * (cw + 6) + 10;
        V.arrow(g, mx, cy + 15, mx + 26, cy + 15, { color: C.a, lw: 2 });
        const val = S.mode === 'fused' ? meanFused : meanExact;
        V.box(g, mx + 32, cy, 84, 30, { r: 7, fill: V.alpha(C.a, 0.25), stroke: C.a, lw: 1.6 });
        txt(g, fmt(val, 4), mx + 74, cy + 19, { size: 12, align: 'center', color: C.ink, font: 'mono', weight: 700 });
        txt(g, S.mode === 'fused' ? 'média de H\' em BF16' : 'média (FP32)', mx + 74, cy + 44, { size: 11, align: 'center', color: C.muted, maxW: 120 });
        if (S.mode === 'exact' || S.mode === 'fused') {
          const fy = cy + 62;
          if (fy + 14 < h) txt(g, S.mode === 'exact' ? 'fatorado: y·média(post) + Σᵢ Hᵢ·(Σₖ combᵢₖ)/4 = ' + fmt(factored, 4) : 'exata daria ' + fmt(meanExact, 4) + ': diferença só de arredondamento (interpretação)', pad, fy, { size: 11, color: C.muted, font: 'mono', maxW: pw });
        }
        // destino: o buffer que o rascunhador lê (enche quando o token passa pelos taps)
        const by = cy + 92;
        if (by + 60 > h) return;
        const filled = (k) => tokPos * nL > taps[k] + 1;
        txt(g, glm ? 'destino: aux_hidden_states, uma entrada [T, 4096] por tap' : 'destino: _dspark_hidden_buffer [8192, 3 × 4096] (192 MiB em BF16)', pad, by, { size: 11.5, color: C.ink, weight: 600, maxW: pw });
        const slotsW = narrow ? pw : Math.min(pw - 230, 460);
        const sw = (slotsW - 6 * (taps.length - 1)) / taps.length, sy2 = by + 12;
        taps.forEach((l, k) => {
          const x = pad + k * (sw + 6), on = filled(k);
          V.box(g, x, sy2, sw, 30, { r: 7, fill: on ? V.alpha(C.c, 0.35) : V.alpha(C.muted, 0.08), stroke: on ? C.c : V.alpha(C.muted, 0.5), lw: 1.3 });
          txt(g, (glm ? 'saída ' : 'camada ') + l, x + sw / 2, sy2 + 19, { size: 11, align: 'center', color: on ? C.ink : C.muted, font: 'mono', maxW: sw - 6 });
        });
        const dest = glm ? ['DFlash2', 'rascunho em bloco condicionado nos 5 taps'] : ['main_proj 12288 → 4096', '+ main_norm → contexto do DSpark'];
        let dx, dy, dw;
        if (narrow) { dx = pad; dy = sy2 + 52; dw = pw; V.arrow(g, pad + pw / 2, sy2 + 32, pad + pw / 2, dy - 2, { color: C.c, lw: 2 }); }
        else { dx = pad + slotsW + 34; dy = sy2 - 6; dw = w - pad - dx; V.arrow(g, pad + slotsW + 4, sy2 + 15, dx - 4, sy2 + 15, { color: C.c, lw: 2 }); }
        if (dy + 42 > h) return;
        V.box(g, dx, dy, dw, 42, { r: 10, fill: V.alpha(C.c, 0.12), stroke: C.c, lw: 1.4 });
        txt(g, dest[0], dx + dw / 2, dy + 18, { size: 12, align: 'center', color: C.ink, weight: 600, font: 'mono', maxW: dw - 12 });
        txt(g, dest[1], dx + dw / 2, dy + 34, { size: 11, align: 'center', color: C.muted, maxW: dw - 12 });
      });
    })();
  },
});

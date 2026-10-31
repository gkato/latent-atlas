LA.chapter({
  id: 'bloco-transformer',
  part: 2,
  order: 7,
  title: 'O bloco Transformer e o fluxo residual',
  navTitle: 'Bloco Transformer',
  subtitle: 'A peça que se repete dezenas de vezes: linha de montagem decoder-only, fluxo residual, pré-norma com RMSNorm, FFN SwiGLU com swiglu_limit, a contabilidade de parâmetros e FLOPs, denso versus MoE e os nomes dos tensores de um checkpoint.',
  minutes: 50,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['decoder-only', 'fluxo residual', 'RMSNorm', 'pré-norma', 'SwiGLU', 'swiglu_limit', 'hidden_size', 'FLOPs', 'nomes de tensores'],
  goals: [
    'Descrever o caminho decoder-only (embedding → N blocos → norma final → <code>lm_head</code>) e o <b>contrato de forma</b> \\([T \\times d]\\) que o sustenta',
    'Explicar o <b>fluxo residual</b> como barramento: quem lê, quem escreve, por que a soma permite redes profundas e por que em TP cada escrita vira um all-reduce',
    'Calcular RMSNorm e SwiGLU à mão, incluindo o clamp <code>swiglu_limit = 10</code>, e dizer por que overlays EXL3 e b12x precisam replicá-lo',
    'Contar parâmetros e FLOPs de um bloco, achar o contexto a partir do qual a atenção domina e ler os nomes de tensores de um checkpoint real',
  ],
  prereqs: ['redes-neurais', 'atencao'],
  summary: [
    'Um LLM decoder-only é embedding → \\(N\\) blocos idênticos na forma → norma final → <code>lm_head</code>. Todo bloco lê e devolve \\([T \\times d]\\), com <code>hidden_size</code> \\(d\\) = 4096 no GLM e no DeepSeek (por fluxo mHC) e 2560 no Qwen (por ramo do Gated Residual).',
    'O <b>fluxo residual</b> é uma soma: \\(\\mathbf h^{(N)} = \\mathbf h^{(0)} + \\sum(\\mathbf a^{(\\ell)} + \\mathbf f^{(\\ell)})\\). Os leitores (q/k/v, gate/up) recebem cópias normalizadas; os escritores (embedding, <code>o_proj</code>, <code>down_proj</code>) somam. O caminho identidade mantém o gradiente vivo, e em TP cada escrita row-parallel custa um all-reduce (DeepSeek: 43 + 43 por passo).',
    'Pré-norma com <b>RMSNorm</b>: \\(\\mathbf x/\\sqrt{\\overline{x^2}+\\varepsilon}\\odot\\boldsymbol\\gamma\\), invariante à escala. ε = 1e-5 (GLM), 1e-6 (Qwen, DeepSeek 0731), 1e-20 (Vision-Exp). O Qwen usa ganho zero-centrado \\((1+\\mathbf w)\\) e troca as normas do bloco pela leitura do Gated Residual. As normas ficam em BF16 nas receitas.',
    'A FFN é uma <b>SwiGLU</b> \\(W_{\\text{down}}(\\mathrm{SiLU}(W_{\\text{gate}}\\mathbf x)\\odot W_{\\text{up}}\\mathbf x)\\) com \\(3dI\\) parâmetros. O vLLM funde gate e up em <code>gate_up_proj</code>; em TP gate/up são column-parallel e down é row-parallel.',
    '<code>swiglu_limit = 10</code> (GLM e DeepSeek) faz \\(\\mathrm{SiLU}(\\min(g,10))\\cdot\\mathrm{clamp}(u,\\pm10)\\) e limita cada neurônio a ≈ 100. É semântica do modelo: o overlay EXL3 repete <code>SWIGLU_LIMIT_DEFAULT = 10.0</code>, o b12x recebe <code>gemm1_clamp_limit</code> e o MoE cooperativo TP3 só aceita 10.',
    'Contabilidade: atenção \\(\\approx 4d^2\\), FFN clássica \\(8d^2\\) (2/3 do bloco). FLOPs/token ≈ 2 × ativos + \\(4\\,h\\,d_h\\,n\\) por camada, e a atenção domina a partir de \\(n^\\ast = 6d\\) no clássico. No MoE a FFN tem 96–98% dos parâmetros e poucos ativos.',
    'Nos checkpoints os pesos são <code>[saída, entrada]</code> e os nomes seguem a árvore de módulos (<code>model.layers.{i}.mlp.down_proj.weight</code>). Cada modelo tem desvios (KDA/MLA, Gated Residual, <code>w1/w2/w3</code>) que o loader mapeia e que os patches usam como endereço.',
  ],
  quiz: [
    { q: 'Num bloco pré-norma, o que acontece com o fluxo residual \\(\\mathbf h\\) quando a FFN termina?', o: ['Ele é substituído pela saída da FFN', 'Ele recebe a saída da FFN somada: \\(\\mathbf h + \\mathrm{FFN}(\\mathrm{Norm}(\\mathbf h))\\)', 'Ele é normalizado e depois multiplicado pela saída da FFN', 'Nada: a FFN só escreve no KV cache'], a: 1, e: 'A FFN lê uma cópia normalizada e o <code>down_proj</code> produz um vetor de largura \\(d\\) que é <b>somado</b> ao resíduo. Nenhuma subcamada substitui o fluxo; por isso o vetor final é a soma de todas as escritas.' },
    { q: 'Você multiplica a entrada de uma RMSNorm (com ε = 1e-6) por 1000. O que acontece com a saída?', o: ['Fica 1000× maior', 'Fica 1000× menor', 'Praticamente não muda', 'Vira zero'], a: 2, e: 'A RMSNorm divide pelo RMS do próprio vetor, que também fica 1000× maior: \\(\\mathrm{RMSNorm}(\\alpha\\mathbf x) = \\mathrm{RMSNorm}(\\mathbf x)\\) enquanto ε é desprezível perto de \\(\\overline{x^2}\\). Só vetores quase nulos sentem o ε.' },
    { q: 'Quantos parâmetros tem um expert SwiGLU do GLM-5.3-Flash (\\(d = 4096\\), <code>moe_intermediate_size</code> = 2048)?', o: ['8.388.608', '16.777.216', '25.165.824', '50.331.648'], a: 2, e: 'São três matrizes: gate \\([2048 \\times 4096]\\), up \\([2048 \\times 4096]\\) e down \\([4096 \\times 2048]\\), ou \\(3 \\times 4096 \\times 2048 = 25.165.824\\). O DeepSeek tem experts da mesma forma.' },
    { q: 'Com <code>swiglu_limit = 10</code>, um neurônio tem \\(g = 15\\) e \\(u = -30\\). Qual é a ativação?', o: ['\\(\\mathrm{SiLU}(15)\\cdot(-30) \\approx -450\\)', '\\(\\mathrm{SiLU}(10)\\cdot(-10) \\approx -100\\)', '\\(\\mathrm{SiLU}(10)\\cdot(-30) \\approx -300\\)', '0, porque o clamp zera valores fora da faixa'], a: 1, e: 'O gate é limitado por cima (\\(\\min(15, 10) = 10\\)) e o up dos dois lados (\\(\\mathrm{clamp}(-30, -10, 10) = -10\\)). \\(\\mathrm{SiLU}(10) \\approx 9{,}9995\\), então a ativação é ≈ −99,995. O clamp não zera: ele prende na borda.' },
    { q: 'No checkpoint do GLM, qual é a forma do tensor <code>layers.1.mlp.down_proj.weight</code>?', o: ['[12288, 4096]', '[4096, 12288]', '[4096, 4096]', '[2048, 4096]'], a: 1, e: 'A camada 1 é densa (\\(I = 12.288\\)). O PyTorch guarda pesos como <code>[saída, entrada]</code>, e o <code>down_proj</code> sai em \\(d = 4096\\) e lê \\(I = 12.288\\). Primeira dimensão igual a \\(d\\) é a assinatura de um escritor do resíduo.' },
    { q: 'Por que o overlay EXL3 do GLM e o backend b12x do DeepSeek precisam aplicar o mesmo clamp do SwiGLU que a referência?', o: ['Porque sem clamp os kernels não compilam', 'Porque o clamp faz parte da matemática com que o modelo foi treinado: sem ele, as saídas mudam justamente nos tokens com ativações extremas, um erro raro, sistemático e silencioso', 'Porque o clamp reduz a memória dos experts pela metade', 'Porque o vLLM rejeita modelos com swiglu_limit'], a: 1, e: 'O <code>swiglu_limit</code> está no <code>config.json</code> e a referência o aplica. Um kernel que o esqueça diverge só quando alguma ativação passa de 10, o que é raro o bastante para passar em testes de fumaça. Só paridade numérica ou KLD mostram o problema. Por isso o overlay usa <code>SWIGLU_LIMIT_DEFAULT = 10.0</code> e o b12x recebe <code>gemm1_clamp_limit</code>.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const nf = (x, d = 1) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const ni = (x) => Math.round(x).toLocaleString('pt-BR');
    const pp = (n) => (n >= 1e9 ? nf(n / 1e9, n >= 1e11 ? 0 : n >= 1e10 ? 1 : 2) + ' B' : n >= 1e6 ? nf(n / 1e6, n >= 1e8 ? 0 : 1) + ' M' : n >= 1e3 ? nf(n / 1e3, 1) + ' k' : ni(n));
    const fx = (x, d = 2) => V.fmt(x, d).replace('.', ',');
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const silu = (z) => z * sig(z);
    const onTxt = (C) => C.bg; // texto sobre preenchimento forte: fundo escuro no tema escuro, claro no claro
    function textW(g, s, size, font, weight) { g.font = V.font(size || 12, font || 'sans', weight || 400); return g.measureText(String(s)).width; }
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
    function fit(g, s, maxW, size, font, weight) { // reduz a fonte até caber (mín. 11)
      let sz = size;
      while (sz > 11 && textW(g, s, sz, font, weight) > maxW) sz -= 0.5;
      return sz;
    }
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    function pathPos(pts, u) {
      let L = 0; const seg = [];
      for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
      let r = V.clamp(u, 0, 1) * L;
      for (let i = 0; i < seg.length; i++) { if (r <= seg[i] || i === seg.length - 1) { const f = seg[i] ? r / seg[i] : 0; return V.along(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], Math.min(1, f)); } r -= seg[i]; }
      return pts[pts.length - 1];
    }
    const seg01 = (f, a, b) => V.clamp((f - a) / (b - a), 0, 1);
    // wrapper: ignora quadros com o canvas ainda sem tamanho (evita raios negativos em layouts provisórios)
    const F = (fig, draw, opts) => V.fig(fig, (g, w, h, ...rest) => { if (w < 160 || h < 100) return; draw(g, w, h, ...rest); }, opts);
    const safe = (name, fn) => { try { fn(); } catch (e) { console.error('[bloco-transformer] ' + name, e); } };
    // O MathML oculto do KaTeX é position:absolute e pode escapar do overflow-x das caixas
    // .la-eq / .la-table-wrap em telas estreitas. Contenção local.
    root.querySelectorAll('.la-eq, .la-table-wrap, .la-params').forEach((el) => { el.style.position = 'relative'; });
    // célula com valor escrito (texto sempre >= 11px)
    function cell(g, x, y, cw, ch, v, o = {}) {
      const C = V.C();
      const sc = o.scale || 2;
      const col = o.fill || V.cmap(V.clamp(v / sc, -1, 1), 'div');
      V.rr(g, x, y, cw, ch, Math.min(5, cw / 5)); g.fillStyle = col; g.fill();
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.lw || 2; g.stroke(); }
      if (o.text !== false && cw >= 22) {
        let s = o.fmt ? o.fmt(v) : nf(v, Math.abs(v) >= 10 ? 0 : 1);
        if (/^-0(,0+)?$/.test(s)) s = s.slice(1);
        const strong = Math.abs(v / sc) > 0.62;
        T(g, s, x + cw / 2, y + ch / 2 + 0.5, { size: fit(g, s, cw - 3, 12, 'mono'), font: 'mono', align: 'center', base: 'middle', color: strong ? onTxt(C) : C.ink });
      }
    }

    /* ====================================================================
       Figura 1 — A esteira decoder-only
       ==================================================================== */
    safe('esteira', () => {
      const fig = $('#fig-esteira');
      fitH(fig, (w) => (w < 620 ? 620 : 400));
      const MODELS = {
        glm: { name: 'GLM-5.3-Flash', d: 4096, N: 45, V: 154880, comp: '34 KDA + 11 MLA · 3 densas + 42 MoE', extra: '+ MTP no índice 45', fin: 'model.language_model.norm', res: '4 fluxos mHC de 4096', col: 'glm', type: (i) => (i % 4 === 3 && i < 44 ? 1 : 0), tn: ['KDA', 'MLA'] },
        qwen: { name: 'Qwen3.8-Flash-Next', d: 2560, N: 48, V: 248320, comp: '36 GDN + 12 QSA · 48 MoE', extra: '+ MTP (mtp.layers.48)', fin: 'hyper_connection_mixer', res: '4 ramos de 2560 (GR)', col: 'qwen', type: (i) => ((i + 1) % 4 === 0 ? 1 : 0), tn: ['GDN', 'QSA'] },
        ds: { name: 'DeepSeek-V4-Flash', d: 4096, N: 43, V: 129280, comp: '2 SWA + 21 C4A + 20 C128A · 43 MoE', extra: '+ 3 estágios DSpark', fin: 'hc_head + norm', res: '4 fluxos mHC de 4096', col: 'ds', type: (i) => (i < 2 ? 2 : i % 2 === 0 ? 1 : 0), tn: ['C128A', 'C4A', 'SWA'] },
        toy: { name: 'Transformer de brinquedo', d: 8, N: 4, V: 1000, comp: '4 blocos iguais (ilustrativo)', extra: 'valores ilustrativos', fin: 'norm', res: '1 fluxo de 8', col: 'a', type: () => 0, tn: ['bloco'] },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'toy', label: 'Brinquedo' }], value: 'glm', onChange: (v) => { S.m = v; } });
      const toks = ['O', ' gato', ' subiu', ' no', ' telhado'];
      const nextTok = [[' gato', ' cão', ' dia', ' rei', ' mar'], [' subiu', ' dorme', ' come', ' viu', ' é'], [' no', ' na', ' ao', ' pelo', ' em'], [' telhado', ' muro', ' sofá', ' carro', ' topo'], ['.', ',', ' e', ' de', '!']];
      const rr = V.rng(7);
      const probs = toks.map(() => { const ls = Array.from({ length: 5 }, () => rr.normal() * 0.8); ls[0] += 2.4; return V.softmax(ls); });
      const vecs = toks.map(() => Array.from({ length: 8 }, () => rr.normal()));
      const drift = Array.from({ length: 64 }, () => Array.from({ length: 8 }, () => rr.normal() * 0.35));
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const M = MODELS[S.m];
        const mc = C[M.col] || C.a;
        const P = 8; const ph = V.phase(t, P); const f = ph.f; const k = ph.i % toks.length;
        const wide = w >= 620;
        // cabeçalho
        T(g, M.name, 16, 22, { size: 14, weight: 700, color: mc, font: 'display' });
        const hdr = `d = ${ni(M.d)} · N = ${M.N} · V = ${ni(M.V)}`;
        if (wide) T(g, hdr, w - 16, 22, { size: 12.5, font: 'mono', align: 'right', color: C.muted });
        else T(g, hdr, 16, 42, { size: 12, font: 'mono', color: C.muted });
        // geometria
        let chip, emb, belt, blocks, norm, head, logit;
        if (wide) {
          const by = h * 0.55;
          chip = { x: 16, y: by - 16, w: 76, h: 32 };
          emb = { x: 108, y: by - 105, w: 56, h: 210 };
          blocks = { x0: 196, x1: w - 300, y: by - 72, h: 144, vert: false };
          norm = { x: w - 282, y: by - 26, w: 44, h: 52 };
          head = { x: w - 226, y: by - 105, w: 56, h: 210 };
          logit = { x: w - 158, y: by - 70, w: 142, h: 140 };
          belt = { x0: 92, x1: w - 164, y: by };
        } else {
          const cx = w / 2;
          chip = { x: 16, y: 80, w: 76, h: 32 };
          emb = { x: cx - 50, y: 72, w: 100 + Math.min(60, w * 0.12), h: 48 };
          emb.x = Math.max(104, cx - 50);
          blocks = { y0: 146, y1: h - 240, x: cx - 95, w: 190, vert: true };
          norm = { x: cx - 36, y: h - 206, w: 72, h: 28 };
          head = { x: cx - 80, y: h - 158, w: 160, h: 38 };
          logit = { x: 16, y: h - 108, w: w - 32, h: 100 };
          belt = { x: cx, y0: 118, y1: h - 210 }; // termina logo acima da caixa Norm
        }
        // esteira (faixa com textura correndo)
        if (wide) {
          V.box(g, belt.x0, belt.y - 8, belt.x1 - belt.x0, 16, { r: 8, fill: V.alpha(C.muted, 0.14), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          g.save(); V.rr(g, belt.x0, belt.y - 8, belt.x1 - belt.x0, 16, 8); g.clip();
          for (let x = belt.x0 - 20 + ((t * 30) % 20); x < belt.x1; x += 20) V.line(g, x, belt.y - 8, x + 8, belt.y + 8, { color: V.alpha(C.muted, 0.22), lw: 2 });
          g.restore();
        } else {
          V.box(g, belt.x - 8, belt.y0, 16, belt.y1 - belt.y0, { r: 8, fill: V.alpha(C.muted, 0.14), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          g.save(); V.rr(g, belt.x - 8, belt.y0, 16, belt.y1 - belt.y0, 8); g.clip();
          for (let y = belt.y0 - 20 + ((t * 30) % 20); y < belt.y1; y += 20) V.line(g, belt.x - 8, y, belt.x + 8, y + 8, { color: V.alpha(C.muted, 0.22), lw: 2 });
          g.restore();
        }
        // fases
        const fChip = seg01(f, 0, 0.1), fEmb = seg01(f, 0.1, 0.18), fBl = seg01(f, 0.18, 0.7), fNo = seg01(f, 0.7, 0.76), fHd = seg01(f, 0.76, 0.84), fLg = seg01(f, 0.84, 0.97);
        // tabela de embedding
        const rowsE = wide ? 14 : 6;
        const hiRow = (k * 5 + 3) % rowsE;
        const drawTable = (R, label, sub, lit, isHead) => {
          V.box(g, R.x, R.y, R.w, R.h, { r: 8, fill: V.alpha(isHead ? C.c : C.b, 0.1), stroke: V.alpha(isHead ? C.c : C.b, lit ? 0.95 : 0.45), lw: lit ? 2 : 1.2 });
          if (wide) {
            const rh = (R.h - 12) / rowsE;
            for (let i = 0; i < rowsE; i++) {
              const on = !isHead && i === hiRow && fEmb > 0 && f < 0.3;
              g.fillStyle = on ? C.d : V.alpha(isHead ? C.c : C.b, 0.14 + 0.06 * ((i * 7) % 3));
              V.rr(g, R.x + 6, R.y + 6 + i * rh, R.w - 12, rh - 2, 2); g.fill();
            }
            if (isHead && lit) { g.fillStyle = V.alpha(C.c, 0.25); V.rr(g, R.x + 6, R.y + 6, R.w - 12, R.h - 12, 3); g.fill(); }
          } else {
            const cols = 10, cw2 = (R.w - 12) / cols;
            for (let i = 0; i < cols; i++) {
              const on = !isHead && i === hiRow % cols && fEmb > 0 && f < 0.3;
              g.fillStyle = on ? C.d : V.alpha(isHead ? C.c : C.b, 0.14 + 0.06 * ((i * 7) % 3));
              V.rr(g, R.x + 6 + i * cw2, R.y + 6, cw2 - 2, R.h - 12, 2); g.fill();
            }
          }
          if (wide) {
            T(g, label, R.x + R.w / 2, R.y - 22, { size: 12, weight: 600, align: 'center', color: C.ink });
            T(g, sub, R.x + R.w / 2, R.y - 8, { size: 11, font: 'mono', align: 'center', color: C.muted });
          } else {
            T(g, label + '  ' + sub, R.x + R.w / 2, R.y - 6, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: w - R.x + 40 });
          }
        };
        drawTable(emb, 'embedding', `[${ni(M.V)} × ${ni(M.d)}]`, fEmb > 0 && f < 0.3, false);
        drawTable(head, 'lm_head', `[${ni(M.V)} × ${ni(M.d)}]`, fHd > 0 && f < 0.9, true);
        // blocos
        const nMax = blocks.vert ? Math.floor((blocks.y1 - blocks.y0) / 15) : Math.floor((blocks.x1 - blocks.x0) / 22);
        const nShow = Math.max(2, Math.min(M.N, nMax));
        const ell = M.N > nShow;
        const tc = [V.alpha(mc, 0.6), C.a, C.d];
        const aHead = Math.floor((nShow - 1) / 2);
        for (let i = 0; i < nShow; i++) {
          // mostra as primeiras e as últimas camadas na ordem real (o padrão fica visível)
          const li = !ell ? i : i < aHead ? i : M.N - (nShow - i);
          const u = (i + 0.5) / nShow;
          const passed = fBl > u - 0.5 / nShow && fBl < u + 0.5 / nShow && f < 0.71;
          const ty = M.type(li);
          const col = tc[ty] || mc;
          if (ell && i === aHead) {
            if (blocks.vert) T(g, '⋮', blocks.x + blocks.w / 2 + 20, blocks.y0 + u * (blocks.y1 - blocks.y0) + 4, { size: 14, color: C.muted, align: 'center' });
            else T(g, '…', blocks.x0 + u * (blocks.x1 - blocks.x0), blocks.y + blocks.h / 2 - 50, { size: 16, color: C.muted, align: 'center' });
            continue;
          }
          if (blocks.vert) {
            const y = blocks.y0 + u * (blocks.y1 - blocks.y0);
            V.box(g, blocks.x, y - 5, blocks.w, 10, { r: 4, fill: passed ? col : V.alpha(col, 0.28), stroke: passed ? C.ink : null, lw: 1 });
          } else {
            const x = blocks.x0 + u * (blocks.x1 - blocks.x0);
            V.box(g, x - 8, blocks.y, 16, blocks.h, { r: 5, fill: passed ? col : V.alpha(col, 0.28), stroke: passed ? C.ink : null, lw: 1 });
            if (passed) V.glow(g, x, blocks.y - 6, 12, col, 0.8);
          }
        }
        // rótulos dos blocos
        const bl1 = `${M.N} blocos · cada um [T × d] → [T × d]`;
        const bl2 = M.comp + ' · ' + M.extra;
        if (blocks.vert) {
          // fundo sob o texto para a faixa vertical da esteira não atravessar os rótulos
          const backed = (s, y, o) => { const tw = Math.min(textW(g, s, o.size, 'sans', o.weight || 400), w - 24); g.fillStyle = V.alpha(C.bg, 0.94); V.rr(g, w / 2 - tw / 2 - 6, y - o.size - 1, tw + 12, o.size + 7, 4); g.fill(); T(g, s, w / 2, y, Object.assign({ align: 'center', maxW: w - 24 }, o)); };
          backed(bl1, blocks.y0 - 12, { size: 11.5, color: C.ink, weight: 600 });
          backed(bl2, blocks.y1 + 16, { size: 11, color: C.muted });
        } else {
          T(g, bl1, (blocks.x0 + blocks.x1) / 2, blocks.y - 26, { size: 12, align: 'center', color: C.ink, weight: 600, maxW: blocks.x1 - blocks.x0 + 30 });
          T(g, bl2, (blocks.x0 + blocks.x1) / 2, blocks.y + blocks.h + 22, { size: 11.5, align: 'center', color: C.muted, maxW: blocks.x1 - blocks.x0 + 40 });
          T(g, 'fluxo residual: ' + M.res, (blocks.x0 + blocks.x1) / 2, blocks.y + blocks.h + 40, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: blocks.x1 - blocks.x0 + 60 });
          T(g, 'norma final: ' + M.fin, (blocks.x0 + blocks.x1) / 2, blocks.y + blocks.h + 56, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: blocks.x1 - blocks.x0 + 60 });
          // legenda de tipos (acima, à direita)
          if (M.tn.length > 1) {
            let lx = blocks.x1;
            for (let q = M.tn.length - 1; q >= 0; q--) { const nm = M.tn[q]; lx -= textW(g, nm, 11) + 16; g.fillStyle = tc[q]; V.rr(g, lx, blocks.y - 51, 10, 10, 2); g.fill(); T(g, nm, lx + 14, blocks.y - 42, { size: 11, color: C.muted }); lx -= 12; }
          }
        }
        // norma final
        const nLit = fNo > 0 && f < 0.8;
        V.box(g, norm.x, norm.y, norm.w, norm.h, { r: 8, fill: V.alpha(C.e, nLit ? 0.35 : 0.12), stroke: V.alpha(C.e, nLit ? 1 : 0.5), lw: nLit ? 2 : 1.2 });
        T(g, 'Norm', norm.x + norm.w / 2, norm.y + norm.h / 2 + 4, { size: 11.5, align: 'center', weight: 600, color: C.ink });
        if (wide) T(g, 'final', norm.x + norm.w / 2, norm.y + norm.h + 14, { size: 11, align: 'center', color: C.muted });
        // chip do token
        const tok = toks[k];
        const chipAlpha = f < 0.12 ? 1 : 0.35;
        V.box(g, chip.x, chip.y, chip.w, chip.h, { r: 10, fill: V.alpha(C.d, 0.18 * chipAlpha + 0.05), stroke: V.alpha(C.d, chipAlpha), lw: 1.5 });
        T(g, JSON.stringify(tok), chip.x + chip.w / 2, chip.y + chip.h / 2 + 4, { size: 12, font: 'mono', align: 'center', color: C.ink, maxW: chip.w - 6 });
        T(g, 'token', chip.x + chip.w / 2, chip.y - 8, { size: 11, align: 'center', color: C.muted });
        // trajetória do vagão
        let pts;
        if (wide) pts = [[emb.x + emb.w + 4, belt.y], [blocks.x0 - 6, belt.y], [blocks.x1 + 6, belt.y], [norm.x + norm.w / 2, belt.y], [head.x - 6, belt.y]];
        else pts = [[belt.x, emb.y + emb.h + 6], [belt.x, blocks.y0 - 4], [belt.x, blocks.y1 + 4], [belt.x, norm.y + norm.h / 2], [belt.x, head.y - 4]];
        let pos = null, vals = vecs[k].slice();
        if (f < 0.1) {
          const a = [chip.x + chip.w / 2, chip.y + chip.h / 2], b = wide ? [emb.x + emb.w / 2, emb.y + emb.h / 2] : [emb.x + emb.w / 2, emb.y + emb.h / 2];
          const p = V.along(a[0], a[1], b[0], b[1], V.ease.inOut(fChip));
          V.dot(g, p[0], p[1], 6, C.d); V.glow(g, p[0], p[1], 16, C.d, 0.7);
        } else if (f < 0.84) {
          let u;
          if (f < 0.18) u = 0.02 * fEmb;
          else if (f < 0.7) u = 0.02 + 0.66 * fBl;
          else if (f < 0.76) u = 0.68 + 0.16 * fNo;
          else u = 0.84 + 0.16 * fHd;
          pos = pathPos(pts, u);
          const nPassed = Math.floor(fBl * 60);
          for (let j = 0; j < 8; j++) { let v = vals[j]; for (let q = 0; q < Math.min(nPassed, 60); q++) v += drift[q][j] * 0.35; vals[j] = v; }
          if (f >= 0.7) { const r = Math.sqrt(vals.reduce((s, x) => s + x * x, 0) / 8) || 1; const mixF = fNo; vals = vals.map((x) => V.lerp(x, x / r, mixF)); }
        }
        if (pos) {
          const ww = 60, wh = 24;
          const x = pos[0] - ww / 2, y = pos[1] - wh / 2 - (wide ? 0 : 0);
          V.box(g, x - 2, y - 2, ww + 4, wh + 4, { r: 7, fill: C.panel, stroke: C.d, lw: 1.5 });
          for (let j = 0; j < 8; j++) { g.fillStyle = V.cmap(V.clamp(vals[j] / 2.5, -1, 1), 'div'); V.rr(g, x + 2 + j * 7, y + 3, 6, wh - 6, 2); g.fill(); }
          T(g, 'd = ' + ni(M.d), pos[0], y - 8, { size: 11, font: 'mono', align: 'center', color: C.d });
        }
        // logits
        const pr = probs[k];
        const lbl = nextTok[k];
        const nb = 5;
        const rowH = Math.min(24, (logit.h - 22) / nb);
        T(g, 'logits → probabilidades', logit.x, logit.y + 2, { size: 11.5, color: C.muted, weight: 600 });
        const labW = wide ? 56 : 70;
        for (let i = 0; i < nb; i++) {
          const y = logit.y + 12 + i * rowH;
          const pv = pr[i] * fLg;
          T(g, JSON.stringify(lbl[i]), logit.x + labW - 4, y + rowH / 2 + 4, { size: 11, font: 'mono', align: 'right', color: i === 0 && fLg > 0.9 ? C.c : C.muted, maxW: labW - 4 });
          const bw = (logit.w - labW - 40) * pv;
          V.box(g, logit.x + labW, y + 3, Math.max(2, bw), rowH - 6, { r: 4, fill: i === 0 ? C.c : V.alpha(C.c, 0.4) });
          if (fLg > 0.6) T(g, nf(pr[i] * 100, 0) + '%', logit.x + labW + bw + 4, y + rowH / 2 + 4, { size: 11, font: 'mono', color: C.ink });
        }
        if (!wide) T(g, M.fin, norm.x + norm.w + 10, norm.y + norm.h / 2 + 4, { size: 11, font: 'mono', color: C.muted, maxW: w - norm.x - norm.w - 14 });

      });
    });

    /* ====================================================================
       Figura 2 — O barramento residual (stepper)
       ==================================================================== */
    safe('barramento', () => {
      const fig = $('#fig-barramento');
      fitH(fig, (w) => (w < 620 ? 460 : 420));
      const r = V.rng(11);
      const D = 8;
      const mk = (s) => Array.from({ length: D }, () => r.normal() * s);
      const parts = [mk(0.9), mk(0.6), mk(0.7), mk(0.6), mk(0.7)]; // emb, a0, f0, a1, f1
      // direção "editada" (ilustrativa) e a1 após a projeção ortogonal
      const vdir = parts[3].map((x, j) => x + 0.3 * parts[1][j]);
      const vn = Math.sqrt(vdir.reduce((s, x) => s + x * x, 0));
      const vh = vdir.map((x) => x / vn);
      const dot = parts[3].reduce((s, x, j) => s + x * vh[j], 0);
      const a1edit = parts[3].map((x, j) => x - dot * vh[j]);
      const names = ['embedding', 'atenção 0', 'FFN 0', 'atenção 1', 'FFN 1'];
      const steps = [
        { t: 'Embedding: o vetor inicial', d: 'A linha do embedding é o primeiro escritor. Cada coluna de baixo é uma coordenada do vetor, e a cor âmbar é a parcela do embedding.' },
        { t: 'Leitura: a norma tira uma foto', d: 'Antes da atenção, a RMSNorm faz uma <b>cópia</b> normalizada (os quadradinhos dentro do posto). As colunas não mudam: ler não tira nada do barramento.' },
        { t: 'Escrita: a atenção soma sua parcela', d: 'O <code>o_proj</code> devolve um vetor de largura d, que é <b>somado</b>. Cada coluna ganha um segmento lavanda (positivo para cima, negativo para baixo). O ponto marca o novo valor.' },
        { t: 'FFN 0: lê e escreve', d: 'A mesma coreografia: cópia normalizada, SwiGLU, e o <code>down_proj</code> soma a parcela menta.' },
        { t: 'Bloco 1: mais duas escritas', d: 'A atenção 1 (rosa) e a FFN 1 (verde) empilham suas parcelas. Nada do que foi escrito antes é apagado.' },
        { t: 'O vetor final é uma soma', d: '\\(\\mathbf h^{(2)} = \\mathbf h^{(0)} + \\mathbf a^{(0)} + \\mathbf f^{(0)} + \\mathbf a^{(1)} + \\mathbf f^{(1)}\\). A norma final e o <code>lm_head</code> leem essa soma. O número acima de cada coluna é o valor da coordenada.' },
        { t: 'Editar um escritor', d: 'Se a atenção 1 for impedida de escrever numa direção \\(\\mathbf v\\), como faz a abliteração no <code>o_proj</code>, só a parcela rosa muda: \\(\\mathbf a \\leftarrow \\mathbf a - (\\mathbf a\\cdot\\mathbf v)\\mathbf v\\). As outras parcelas continuam intactas.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5200 });
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const cols = [C.d, C.b, C.a, C.c, C.e];
        const i = st.i, el = st.elapsed();
        const grow = V.ease.out(V.clamp(el / 0.9, 0, 1));
        // quantas parcelas visíveis e crescimento
        const vis = [1, 1, 2, 3, 5, 5, 5][i];
        const growIdx = [0, -1, 1, 2, 3, 99, 99][i];
        const growIdx2 = i === 4 ? 4 : -1;
        const READ = [[], [1], [], [2], [3, 4], [5], []][i];
        const WRITE = [[0], [], [1], [2], [3, 4], [], [3]][i];
        const narrow = w < 620;
        // estrada + postos
        const roadY = narrow ? 118 : 128;
        const x0 = 16, x1 = w - 16;
        V.box(g, x0, roadY - 10, x1 - x0, 20, { r: 10, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        for (let x = x0 + ((t * 40) % 26); x < x1 - 6; x += 26) V.line(g, x, roadY, x + 12, roadY, { color: V.alpha(C.muted, 0.35), lw: 2 });
        T(g, 'fluxo residual (d faixas)', x0 + 6, roadY + 26, { size: 11, color: C.muted, font: 'mono' });
        const st6 = [{ l: 'Emb', c: C.d }, { l: narrow ? 'At0' : 'Atenção 0', c: C.b }, { l: 'FFN 0', c: C.a }, { l: narrow ? 'At1' : 'Atenção 1', c: C.c }, { l: 'FFN 1', c: C.e }, { l: 'lm_head', c: C.f }];
        const sw = (x1 - x0) / st6.length;
        // soma atual (para a cópia normalizada)
        const curSum = (nv) => Array.from({ length: D }, (_, j) => { let s = 0; for (let q = 0; q < nv; q++) s += (q === 3 && i === 6 ? V.lerp(parts[3][j], a1edit[j], grow) : parts[q][j]); return s; });
        st6.forEach((s, q) => {
          const bx = x0 + q * sw + 4, bw = sw - 8, by = narrow ? 30 : 34, bh = narrow ? 50 : 58;
          const reading = READ.includes(q);
          const writing = WRITE.includes(q);
          const on = reading || writing;
          V.box(g, bx, by, bw, bh, { r: 8, fill: V.alpha(s.c, on ? 0.26 : 0.08), stroke: V.alpha(s.c, on ? 1 : 0.4), lw: on ? 2 : 1 });
          T(g, s.l, bx + bw / 2, by + 15, { size: fit(g, s.l, bw - 6, 12, 'sans', 600), weight: 600, align: 'center', color: C.ink });
          const cx = bx + bw / 2;
          // seta de leitura (fina, sobe) e de escrita (grossa, desce)
          if (q > 0) V.arrow(g, cx - 8, roadY - 12, cx - 8, by + bh + 2, { color: V.alpha(C.muted, reading ? 0.9 : 0.25), lw: 1.2, head: 5 });
          if (q < 5) V.arrow(g, cx + 8, by + bh + 2, cx + 8, roadY - 12, { color: V.alpha(s.c, writing ? 0.95 : 0.25), lw: writing ? 3 : 1.5, head: 7 });
          if (reading && q > 0) {
            const vv = curSum(Math.min(q, 5));
            const rms = Math.sqrt(vv.reduce((a, x) => a + x * x, 0) / D) || 1;
            const cw = Math.min(10, (bw - 12) / D);
            for (let j = 0; j < D; j++) { g.fillStyle = V.cmap(V.clamp(vv[j] / rms / 2, -1, 1), 'div'); V.rr(g, cx - (cw * D) / 2 + j * cw, by + bh - 20, cw - 1.5, 12, 2); g.fill(); }
          }
          if (writing && el < 1.2) {
            const py = V.lerp(by + bh, roadY, V.ease.inOut(V.clamp(el / 0.9, 0, 1)));
            V.box(g, cx + 2, py - 7, 12, 12, { r: 3, fill: s.c });
          }
        });
        // pilhas de contribuições
        const top = narrow ? 170 : 182, bot = h - 20;
        const base = (top + bot) / 2 + 6;
        let mx = 0;
        for (let j = 0; j < D; j++) { let p = 0, n = 0; for (let q = 0; q < 5; q++) { const v = parts[q][j]; if (v > 0) p += v; else n -= v; } mx = Math.max(mx, p, n); }
        const sc = (bot - top) / 2 / (mx * 1.12);
        V.line(g, x0, base, x1, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const cw = (x1 - x0 - 30) / D;
        for (let j = 0; j < D; j++) {
          const cx0 = x0 + 30 + j * cw + cw * 0.18, bw = cw * 0.64;
          let py = base, ny = base, sum = 0;
          for (let q = 0; q < vis; q++) {
            let v = parts[q][j];
            if (q === 3 && i === 6) v = V.lerp(parts[3][j], a1edit[j], grow);
            if (q === growIdx || q === growIdx2) v *= grow;
            sum += v;
            const hh = Math.abs(v) * sc;
            g.fillStyle = V.alpha(cols[q], q === 3 && i === 6 ? 0.95 : 0.85);
            if (v >= 0) { V.rr(g, cx0, py - hh, bw, Math.max(1, hh), 3); g.fill(); py -= hh; } else { V.rr(g, cx0, ny, bw, Math.max(1, hh), 3); g.fill(); ny += hh; }
          }
          const yv = base - sum * sc;
          V.line(g, cx0 - 4, yv, cx0 + bw + 4, yv, { color: C.ink, lw: 2 });
          V.dot(g, cx0 + bw / 2, yv, 3.5, C.ink);
          if (i >= 5) T(g, nf(sum, 1), cx0 + bw / 2, Math.min(py, yv) - 8, { size: 11, font: 'mono', align: 'center', color: C.ink });
          T(g, 'x' + j, cx0 + bw / 2, bot + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // legenda
        let lx = x0 + 30, ly = top - 4;
        for (let q = 0; q < vis; q++) {
          const lw = textW(g, names[q], 11) + 22;
          if (lx + lw > x1) { lx = x0 + 30; ly += 14; }
          g.fillStyle = cols[q]; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill();
          T(g, names[q], lx + 14, ly, { size: 11, color: C.muted });
          lx += lw;
        }
      });
    });

    /* ====================================================================
       Figura 3 — Telefone sem fio (sensibilidade com/sem resíduo)
       ==================================================================== */
    safe('telefone', () => {
      const fig = $('#fig-telefone');
      fitH(fig, (w) => (w < 620 ? 420 : 360));
      const Dm = 16, LM = 64;
      const r = V.rng(23);
      const W1 = [], W2 = [];
      for (let l = 0; l < LM; l++) {
        W1.push(Array.from({ length: Dm * Dm }, () => r.normal() / Math.sqrt(Dm)));
        W2.push(Array.from({ length: Dm * Dm }, () => r.normal() / Math.sqrt(Dm)));
      }
      const h0 = Array.from({ length: Dm }, () => r.normal());
      const t0 = Array.from({ length: Dm }, () => r.normal());
      const tn0 = Math.sqrt(t0.reduce((s, x) => s + x * x, 0));
      const MODES = ['plain', 'res', 'small'];
      const MN = { plain: 'sem resíduo', res: 'com resíduo', small: 'resíduo + escritores ÷ 8' };
      const DASH = { plain: [6, 4], res: [2, 3], small: [10, 3, 2, 3] };
      const S = { mode: 'plain', g: 0.8, L: 32, sens: {} };
      const mv = (W, x) => { const y = new Array(Dm).fill(0); for (let i = 0; i < Dm; i++) { let s = 0; for (let j = 0; j < Dm; j++) s += W[i * Dm + j] * x[j]; y[i] = s; } return y; };
      // derivada direcional exata (modo direto): tt é o "empurrão" propagado camada a camada
      function run(mode) {
        let h = h0.slice(), tt = t0.slice();
        const ws = mode === 'small' ? 1 / Math.sqrt(LM) : 1; // escritores pequenos: W₂ ÷ √64
        const out = [];
        for (let l = 0; l < S.L; l++) {
          if (mode === 'plain') {
            // rede linear h ← g·W₁h: a tangente segue a mesma conta
            tt = mv(W1[l], tt).map((x) => x * S.g);
          } else {
            const ms = h.reduce((s, x) => s + x * x, 0) / Dm;
            const rr = Math.sqrt(ms + 1e-6);
            const u = h.map((x) => x / rr);
            const ut = u.reduce((s, x, i) => s + x * tt[i], 0);
            const du = tt.map((x, i) => (x - (u[i] * ut) / Dm) / rr);
            const pre = mv(W1[l], u).map((x) => x * S.g);
            const a = pre.map(Math.tanh);
            const fo = mv(W2[l], a).map((x) => x * S.g * ws);
            const dpre = mv(W1[l], du).map((x) => x * S.g);
            const da = dpre.map((x, i) => x * (1 - a[i] * a[i]));
            const df = mv(W2[l], da).map((x) => x * S.g * ws);
            h = h.map((x, i) => x + fo[i]);
            tt = tt.map((x, i) => x + df[i]);
          }
          const n = Math.sqrt(tt.reduce((s, x) => s + x * x, 0));
          out.push(Math.log10(Math.max(1e-300, n / tn0)));
        }
        return out;
      }
      function compute() { MODES.forEach((m) => { S.sens[m] = run(m); }); }
      compute();
      const pw = (x) => { const s = nf(x, 1); return '10^' + (/^-0,0$/.test(s) ? '0,0' : s.replace('-', '−')); };
      const tag = (x) => (x < -1 ? ' (sumiu)' : x > 3 ? ' (explodiu)' : '');
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Arquitetura', options: MODES.map((m) => ({ v: m, label: MN[m] })), value: 'plain', onChange: (v) => { S.mode = v; ro.set(); } });
      UI.slider(ctl, { label: 'Ganho dos pesos', min: 0.2, max: 3, step: 0.1, value: S.g, fmt: (v) => nf(v, 1), onInput: (v) => { S.g = v; compute(); ro.set(); } });
      UI.slider(ctl, { label: 'Profundidade', min: 4, max: 64, step: 1, value: S.L, fmt: (v) => v + ' camadas', onInput: (v) => { S.L = v; compute(); ro.set(); } });
      const rd = UI.readout(ctl, 'Na última camada', '—');
      const ro = { set: () => { const s = S.sens[S.mode]; const x = s[s.length - 1]; rd.set(pw(x) + tag(x)); } };
      ro.set();
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const cur = S.sens[S.mode];
        const L = cur.length;
        const TT = { plain: 'Sem resíduo (rede linear): h ← g·W h', res: 'Com resíduo: h ← h + g·W₂·tanh(g·W₁·RMSNorm(h))', small: 'Escritores pequenos: h ← h + (g/8)·W₂·tanh(g·W₁·RMSNorm(h))' };
        const tc = { plain: C.c, res: C.d, small: C.a }[S.mode];
        const nl = wrap(g, TT[S.mode], 14, 20, w - 28, 16, { size: 12.5, weight: 600, color: tc, maxLines: 2 });
        // legenda: barras = arquitetura escolhida; linhas finas = as outras duas
        let lx = 14, ly = 20 + nl * 16 + 6;
        MODES.forEach((m) => {
          const sel = m === S.mode;
          const x = S.sens[m][L - 1];
          const s = MN[m] + ': ' + pw(x);
          const lw = textW(g, s, 11, 'sans', sel ? 600 : 400) + 26 + 16;
          if (lx + lw > w - 10 && lx > 14) { lx = 14; ly += 16; }
          if (sel) { g.fillStyle = V.alpha(C.a, 0.85); V.rr(g, lx + 7, ly - 10, 9, 11, 2); g.fill(); }
          else V.line(g, lx, ly - 4, lx + 22, ly - 4, { color: V.alpha(C.ink, 0.7), lw: 1.6, dash: DASH[m] });
          T(g, s, lx + 26, ly, { size: 11, color: sel ? C.ink : C.muted, weight: sel ? 600 : 400 });
          lx += lw;
        });
        const x0 = 50, x1 = w - 14, y0 = ly + 14, y1 = h - 40;
        const lim = 6;
        const yOf = (v) => y0 + ((lim - V.clamp(v, -lim, lim)) / (2 * lim)) * (y1 - y0);
        // eixo
        for (let v = -lim; v <= lim; v += 2) {
          const y = yOf(v);
          V.line(g, x0, y, x1, y, { color: v === 0 ? V.alpha(C.muted, 0.6) : C.grid, lw: 1 });
          T(g, v === 0 ? '1' : '10' + (v > 0 ? '' : '⁻') + ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶'][Math.abs(v)], x0 - 6, y + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        }
        g.save(); g.translate(13, (y0 + y1) / 2); g.rotate(-Math.PI / 2);
        T(g, 'sensibilidade (log)', 0, 0, { size: 11, align: 'center', color: C.muted });
        g.restore();
        // faixa saudável
        g.fillStyle = V.alpha(C.a, 0.07); g.fillRect(x0, yOf(3), x1 - x0, yOf(-1) - yOf(3));
        const bw = (x1 - x0) / L;
        const pulse = (t * 0.6) % 1;
        const pi = Math.floor(pulse * L);
        for (let l = 0; l < L; l++) {
          const v = cur[l];
          const ok = v >= -1 && v <= 3;
          const col = ok ? C.a : v < 0 ? C.b : C.c;
          const x = x0 + l * bw + bw * 0.12;
          const ya = yOf(0), yb = yOf(v);
          g.fillStyle = V.alpha(col, l === pi ? 1 : 0.75);
          V.rr(g, x, Math.min(ya, yb), Math.max(1, bw * 0.76), Math.max(1.5, Math.abs(yb - ya)), Math.min(3, bw * 0.2)); g.fill();
          if (Math.abs(v) > lim) T(g, v > 0 ? '▲' : '▼', x + bw * 0.38, v > 0 ? y0 - 2 : y1 + 12, { size: 11, align: 'center', color: col });
        }
        // as outras duas arquiteturas, como linhas finas
        MODES.forEach((m) => {
          if (m === S.mode) return;
          const sv = S.sens[m];
          g.beginPath();
          sv.forEach((v, l) => { const xx = x0 + (l + 0.5) * bw, yy = yOf(v); if (l === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); });
          g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1.6; g.setLineDash(DASH[m]); g.stroke(); g.setLineDash([]);
        });
        // o "sussurro" percorrendo as camadas
        const px = x0 + (pi + 0.5) * bw;
        const pv = cur[pi] == null ? 0 : cur[pi];
        V.glow(g, px, yOf(pv), 10, pv >= -1 && pv <= 3 ? C.a : C.c, 0.8);
        T(g, 'camada →', (x0 + x1) / 2, h - 8, { size: 11, align: 'center', color: C.muted });
        T(g, '1', x0 + bw / 2, h - 8, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        T(g, String(L), x0 + (L - 0.5) * bw, h - 8, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      });
    });

    /* ====================================================================
       Figura 4 — RMSNorm interativa
       ==================================================================== */
    safe('rmsnorm', () => {
      const fig = $('#fig-rmsnorm');
      fitH(fig, (w) => (w < 620 ? 440 : 380));
      const x0 = [2.3, -0.1, 0.8, 3.4, -0.9, 0.6, -1.7, 1.4];
      const gam = [1.1, 0.92, 1.25, 0.98, 1.05, 0.88, 1.15, 0.95];
      const wz = gam.map((x) => +(x - 1).toFixed(2));
      const S = { e: 0, eps: 1e-6, ln: false, gain: 'direct' };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Escala da entrada', min: -4, max: 3, step: 0.25, value: 0, fmt: (v) => '×10^' + nf(v, 2).replace(',00', ''), onInput: (v) => { S.e = v; H.redraw(); } });
      UI.seg(ctl, { label: 'ε', options: [{ v: 1e-20, label: '1e-20' }, { v: 1e-6, label: '1e-6' }, { v: 1e-5, label: '1e-5' }, { v: 1e-2, label: '1e-2 (exagero)' }], value: 1e-6, onChange: (v) => { S.eps = v; H.redraw(); } });
      UI.seg(ctl, { label: 'Norma', options: [{ v: false, label: 'RMSNorm' }, { v: true, label: 'LayerNorm' }], value: false, onChange: (v) => { S.ln = v; H.redraw(); } });
      UI.seg(ctl, { label: 'Ganho', options: [{ v: 'direct', label: 'γ direto' }, { v: 'zc', label: '(1+w) certo' }, { v: 'bug', label: '(1+w) com bug' }], value: 'direct', onChange: (v) => { S.gain = v; H.redraw(); } });
      const H = F(fig, (g, w, h) => {
        const C = V.C();
        const s = Math.pow(10, S.e);
        const x = x0.map((v) => v * s);
        const mu = S.ln ? x.reduce((a, b) => a + b, 0) / 8 : 0;
        const xc = x.map((v) => v - mu);
        const ms = xc.reduce((a, b) => a + b * b, 0) / 8;
        const den = Math.sqrt(ms + S.eps);
        const nrm = xc.map((v) => v / den);
        const stored = S.gain === 'direct' ? gam : wz;
        const eff = S.gain === 'zc' ? wz.map((v) => 1 + v) : stored;
        const y = nrm.map((v, i) => v * eff[i]);
        const rmsY = Math.sqrt(y.reduce((a, b) => a + b * b, 0) / 8);
        const narrow = w < 620;
        const labW = narrow ? 0 : 118;
        const gx = narrow ? 14 : labW + 16;
        const avail = w - gx - (narrow ? 14 : 150);
        const cw = Math.min(58, (avail - 7 * 4) / 8), gap = 4;
        const rowH = narrow ? 64 : 58;
        const cell0 = (row, i) => [gx + i * (cw + gap), 30 + row * rowH];
        const rowLabel = (row, s1, s2) => {
          const [xx, yy] = cell0(row, 0);
          if (narrow) { T(g, s1 + (s2 ? '  ' + s2 : ''), 14, yy - 6, { size: 11.5, color: C.muted, maxW: w - 28 }); }
          else { T(g, s1, 14, yy + 14, { size: 12, weight: 600, color: C.ink, maxW: labW }); if (s2) T(g, s2, 14, yy + 30, { size: 11, color: C.muted, font: 'mono', maxW: labW }); }
        };
        const expo = S.e;
        const fmtScale = expo === 0 ? '' : '× 10^' + nf(expo, 2).replace(',00', '');
        // linha 1: entrada (mostramos a mantissa; o fator de escala vai no rótulo)
        rowLabel(0, 'entrada x', fmtScale);
        x0.forEach((v, i) => { const [cx, cy] = cell0(0, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 3.5 }); });
        // linha 2: centragem (LayerNorm)
        rowLabel(1, S.ln ? 'x − μ' : 'x (sem centrar)', S.ln ? 'μ = ' + nf(0.725, 3) + (fmtScale ? ' ' + fmtScale : '') : 'não subtrai μ');
        xc.forEach((v, i) => { const [cx, cy] = cell0(1, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v / s, { scale: 3.5 }); });
        // linha 3: normalizado
        rowLabel(2, S.ln ? '÷ √(σ² + ε)' : '÷ √(média(x²) + ε)', 'normalizado');
        nrm.forEach((v, i) => { const [cx, cy] = cell0(2, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 2.2, fmt: (q) => nf(q, 2) }); });
        // linha 4: ganho
        rowLabel(3, S.gain === 'direct' ? '⊙ γ' : S.gain === 'zc' ? '⊙ (1 + w)' : '⊙ w  (bug!)', S.gain === 'direct' ? 'tensor salvo = γ' : 'tensor salvo = w');
        eff.forEach((v, i) => { const [cx, cy] = cell0(3, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 2.2, fmt: (q) => nf(q, 2), fill: V.alpha(S.gain === 'bug' ? C.red || C.c : C.e, 0.12 + 0.3 * Math.min(1, Math.abs(v))) }); });
        // linha 5: saída
        rowLabel(4, 'saída y', '');
        y.forEach((v, i) => { const [cx, cy] = cell0(4, i); cell(g, cx, cy + (narrow ? 4 : 0), cw, 34, v, { scale: 2.2, fmt: (q) => (Math.abs(q) < 0.005 && q !== 0 ? '≈0' : nf(q, 2)), stroke: C.a, lw: 1.5 }); });
        // painel de leituras
        const rmsX = Math.sqrt(x.reduce((a, b) => a + b * b, 0) / 8);
        // valores pequenos em notação científica (ex.: 1,71·10^-3), para casar com o texto
        const fe = (v) => { if (v === 0 || (Math.abs(v) >= 0.01 && Math.abs(v) < 1e4)) return nf(v, 2); const e = Math.floor(Math.log10(Math.abs(v))); return nf(v / Math.pow(10, e), 2) + '·10^' + e; };
        const lines = [
          ['RMS(x)', fe(rmsX)],
          [S.ln ? '√(σ² + ε)' : '√(média(x²) + ε)', fe(den)],
          [S.ln ? 'ε / σ²' : 'ε / média(x²)', ms > 0 ? fx(S.eps / ms, 1) : '∞'],
          ['RMS(y)', nf(rmsY, 3)],
        ];
        if (narrow) {
          const yy = 30 + 5 * rowH + 2;
          lines.forEach((ln, i) => { T(g, ln[0] + ' = ' + ln[1], 14, yy + i * 17, { size: 11.5, font: 'mono', color: i === 3 ? C.a : C.muted, maxW: w - 28 }); });
        } else {
          const px = w - 140;
          lines.forEach((ln, i) => { T(g, ln[0], px, 44 + i * 42, { size: 11, color: C.muted }); T(g, ln[1], px, 62 + i * 42, { size: 14, font: 'mono', weight: 600, color: i === 3 ? C.a : C.ink }); });
          const note = S.gain === 'bug' ? 'o loader esqueceu o “1 +”: saída quase nula' : (S.eps / (ms || 1e-300) > 0.2 ? 'ε domina: a saída encolhe' : 'saída na escala padrão');
          wrap(g, note, px, 44 + 4 * 42 + 6, 132, 14, { size: 11, color: S.gain === 'bug' || S.eps / (ms || 1e-300) > 0.2 ? C.c : C.a });
        }
      }, { static: true });
    });

    /* ====================================================================
       Figura 5 — SwiGLU em números (stepper + TP)
       ==================================================================== */
    safe('swiglu', () => {
      const fig = $('#fig-swiglu');
      fitH(fig, (w) => (w < 620 ? 660 : 520));
      const r = V.rng(5);
      const d = 4, I = 6;
      const x = [0.9, -0.4, 1.3, 0.2];
      const q1 = (v) => { const z = Math.round(v * 10) / 10; return z === 0 ? 0 : z; };
      const Wg = Array.from({ length: I }, () => Array.from({ length: d }, () => q1(r.normal() * 0.9)));
      const Wu = Array.from({ length: I }, () => Array.from({ length: d }, () => q1(r.normal() * 0.9)));
      const Wd = Array.from({ length: d }, () => Array.from({ length: I }, () => q1(r.normal() * 0.7)));
      const gv = Wg.map((row) => row.reduce((s, w0, j) => s + w0 * x[j], 0));
      const uv = Wu.map((row) => row.reduce((s, w0, j) => s + w0 * x[j], 0));
      const sv = gv.map(silu);
      const pv = sv.map((s, i) => s * uv[i]);
      const yA = Wd.map((row) => row.slice(0, 3).reduce((s, w0, j) => s + w0 * pv[j], 0));
      const yB = Wd.map((row) => row.slice(3).reduce((s, w0, j) => s + w0 * pv[j + 3], 0));
      const yv = yA.map((a, i) => a + yB[i]);
      const hres = [0.4, -1.1, 0.7, 1.5];
      const hout = hres.map((a, i) => a + yv[i]);
      const S = { tp: false };
      const steps = [
        { t: 'Entra a cópia normalizada', d: 'O vetor \\(\\mathbf x\\) (d = 4) é a saída da RMSNorm. O fluxo residual original fica esperando do lado de fora.' },
        { t: 'gate: g = W_gate x', d: 'Cada linha de \\(W_{\\text{gate}}\\) \\([6 \\times 4]\\) faz um produto escalar com \\(\\mathbf x\\) e vira uma nota do neurônio. A linha acesa é a que está sendo calculada.' },
        { t: 'up: u = W_up x', d: 'Outra projeção, com outros pesos, dá a “mensagem” de cada neurônio. gate e up leem o mesmo \\(\\mathbf x\\), por isso o vLLM as funde num único GEMM (<code>gate_up_proj</code>).' },
        { t: 'A porta: SiLU(g)', d: 'A SiLU abre a porta para notas positivas e quase fecha para negativas: \\(\\mathrm{SiLU}(z) = z\\,\\sigma(z)\\).' },
        { t: 'Porta × mensagem', d: '\\(\\mathrm{SiLU}(\\mathbf g) \\odot \\mathbf u\\), elemento a elemento. Neurônios com gate negativo quase não contribuem, por maior que seja o up.' },
        { t: 'down: y = W_down(…)', d: '\\(W_{\\text{down}}\\) \\([4 \\times 6]\\) fecha de volta para a largura d. É o escritor do resíduo. Com TP=2, cada GPU tem 3 colunas e produz uma parcela, e um <b>all-reduce</b> soma as duas.' },
        { t: 'Soma no resíduo', d: '\\(\\mathbf h\'\' = \\mathbf h\' + \\mathbf y\\). Pronto: a FFN escreveu sua contribuição no barramento.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5600 });
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'dividir em 2 GPUs (TP=2)', value: false, onChange: (v) => { S.tp = v; } });
      F(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 620;
        const cs = narrow ? Math.min(30, (w - 40) / 9.6) : Math.max(22, Math.min(32, (w - 32 - 7 * 26) / 16, (h - 100) / 12.8));
        const gp = 2;
        const G = { A: C.a, B: C.b };
        const rowTint = (ri) => (S.tp ? V.alpha(ri < 3 ? G.A : G.B, 0.9) : null);
        const sweep = (n) => Math.min(n - 1, Math.floor(el / 0.35));
        const mat = (M, ox, oy, o = {}) => {
          const R = M.length, K = M[0].length;
          for (let a = 0; a < R; a++) for (let b = 0; b < K; b++) {
            const xx = ox + b * (cs + gp), yy = oy + a * (cs + gp);
            let stroke = null;
            if (o.hiRow != null && a === o.hiRow) stroke = C.c;
            if (o.tpRows && S.tp) stroke = stroke || (a < 3 ? G.A : G.B);
            if (o.tpCols && S.tp) stroke = stroke || (b < 3 ? G.A : G.B);
            cell(g, xx, yy, cs, cs, M[a][b], { scale: 1.6, stroke, lw: stroke ? 1.6 : 0 });
          }
          if (o.label) T(g, o.label, ox + (K * (cs + gp)) / 2, oy - 8, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
          return { w: K * (cs + gp) - gp, h: R * (cs + gp) - gp };
        };
        const vec = (vs, ox, oy, o = {}) => {
          vs.forEach((v, a) => {
            const shown = o.reveal == null || a <= o.reveal;
            const stroke = o.tp && S.tp ? (a < 3 ? G.A : G.B) : o.hi === a ? C.c : null;
            if (shown) cell(g, ox, oy + a * (cs + gp), cs, cs, v, { scale: o.scale || 2, stroke, lw: 1.6, fmt: (q) => nf(q, 1) });
            else { V.rr(g, ox, oy + a * (cs + gp), cs, cs, 4); g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]); }
          });
          if (o.label) T(g, o.label, ox + cs / 2, oy - 8, { size: 11.5, font: 'mono', align: 'center', color: o.lc || C.muted });
        };
        const col6 = (a) => (a >= 0 ? a : 0);
        const vis = (k) => i >= k;
        // layout
        let P;
        const mh6 = 6 * (cs + gp) - gp, mh4 = 4 * (cs + gp) - gp;
        if (!narrow) {
          const gx = 26;
          let cx = 16;
          const top1 = 30, top2 = top1 + mh6 + 34;
          P = {};
          P.x = [cx, top1 + (mh6 + 34 + mh6 - mh4) / 2]; cx += cs + gx;
          P.Wg = [cx, top1]; P.Wu = [cx, top2]; cx += 4 * (cs + gp) + gx;
          P.g = [cx, top1]; P.u = [cx, top2]; cx += cs + gx;
          P.s = [cx, top1]; cx += cs + gx;
          P.p = [cx, (top1 + top2) / 2]; cx += cs + gx;
          P.Wd = [cx, (top1 + top2) / 2 + (mh6 - mh4) / 2]; cx += 6 * (cs + gp) + gx;
          P.y = [cx, P.Wd[1]]; cx += cs + gx;
          P.h = [cx, P.Wd[1]];
        } else {
          const gx = 14;
          P = {};
          const r1 = 28, r2 = r1 + mh6 + 30, r3 = r2 + mh6 + 30;
          P.x = [14, r1]; P.Wg = [14 + cs + gx, r1]; P.g = [P.Wg[0] + 4 * (cs + gp) + gx, r1]; P.s = [P.g[0] + cs + gx, r1];
          P.Wu = [14 + cs + gx, r2]; P.u = [P.Wu[0] + 4 * (cs + gp) + gx, r2]; P.p = [P.u[0] + cs + gx, r2];
          P.Wd = [14, r3]; P.y = [14 + 6 * (cs + gp) + gx, r3]; P.h = [P.y[0] + cs + gx, r3];
        }
        vec(x, P.x[0], P.x[1], { label: 'x', scale: 1.6 });
        mat(Wg, P.Wg[0], P.Wg[1], { label: 'W_gate [6×4]', hiRow: i === 1 ? sweep(6) : null, tpRows: true });
        mat(Wu, P.Wu[0], P.Wu[1], { label: 'W_up [6×4]', hiRow: i === 2 ? sweep(6) : null, tpRows: true });
        vec(gv, P.g[0], P.g[1], { label: 'g', reveal: i === 1 ? sweep(6) : vis(1) ? 9 : -1, tp: true, scale: 2.5 });
        vec(uv, P.u[0], P.u[1], { label: 'u', reveal: i === 2 ? sweep(6) : vis(2) ? 9 : -1, tp: true, scale: 2.5 });
        vec(sv, P.s[0], P.s[1], { label: 'SiLU(g)', reveal: vis(3) ? 9 : -1, tp: true, scale: 2.5, lc: C.a });
        vec(pv, P.p[0], P.p[1], { label: '⊙', reveal: vis(4) ? 9 : -1, tp: true, scale: 3, lc: C.c });
        mat(Wd, P.Wd[0], P.Wd[1], { label: 'W_down [4×6]', tpCols: true });
        vec(S.tp && i === 5 && el < 2.2 ? yA : yv, P.y[0], P.y[1], { label: S.tp && i === 5 && el < 2.2 ? 'y (GPU A)' : 'y', reveal: vis(5) ? 9 : -1, scale: 3 });
        vec(i >= 6 ? hout : hres, P.h[0], P.h[1], { label: i >= 6 ? "h'' = h' + y" : "h'", scale: 3, lc: i >= 6 ? C.a : C.muted });
        // setas/decoração por passo
        if (i === 3 && !narrow) {
          // mini curva da SiLU embaixo da coluna SiLU(g)
          const bx = P.s[0] - 4, by = P.Wu[1] + 6;
          const bw = P.p[0] - P.s[0] - 4, bh = 84;
          {
            V.box(g, bx, by, bw, bh, { r: 6, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.a, 0.5), lw: 1 });
            g.beginPath();
            for (let q = 0; q <= 40; q++) { const z = -4 + (8 * q) / 40; const xx = bx + 6 + ((bw - 12) * q) / 40; const yy = by + bh - 10 - (silu(z) + 0.3) * ((bh - 16) / 4.3); if (q === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); }
            g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
            T(g, 'SiLU', bx + 8, by + 14, { size: 11, color: C.a, font: 'mono' });
          }
        }
        if (i === 4) { gv.forEach((q, a) => { if (q < 0) T(g, '✕', P.s[0] + cs + 5, P.s[1] + a * (cs + gp) + cs / 2 + 4, { size: 11, color: C.muted }); }); }
        if (S.tp) {
          const ly = narrow ? h - 16 : h - 14;
          g.fillStyle = G.A; V.rr(g, 16, ly - 10, 10, 10, 2); g.fill(); T(g, 'GPU A: neurônios 0–2', 30, ly, { size: 11, color: C.muted });
          g.fillStyle = G.B; V.rr(g, 180, ly - 10, 10, 10, 2); g.fill(); T(g, 'GPU B: neurônios 3–5', 194, ly, { size: 11, color: C.muted });
          if (i === 5) {
            const ph = V.clamp(el / 2.2, 0, 1);
            const xx = P.y[0] + cs / 2, yy = P.y[1] + mh4 + 18;
            T(g, ph < 1 ? 'all-reduce: A + B …' : 'all-reduce feito', xx, yy, { size: 11.5, align: 'center', color: C.d, weight: 600 });
            if (ph < 1) V.glow(g, xx, P.y[1] + mh4 / 2, 22 + 6 * Math.sin(t * 8), C.d, 0.6);
          }
        } else if (!narrow) {
          T(g, 'valores ilustrativos · [saída × entrada]', 16, h - 14, { size: 11, color: C.muted });
        }
      });
    });

    /* ====================================================================
       Figura 6 — O limitador do SwiGLU
       ==================================================================== */
    safe('clamp', () => {
      const fig = $('#fig-clamp');
      fitH(fig, (w) => (w < 620 ? 660 : 400));
      const S = { L: 10, on: true, sem: 'ref' };
      const r = V.rng(31);
      const tdist = (df) => { let z = r.normal(), c = 0; for (let q = 0; q < df; q++) { const n = r.normal(); c += n * n; } return z / Math.sqrt(c / df); };
      const N = 160;
      const samples = Array.from({ length: N }, () => ({ g: tdist(3) * 3.2, u: tdist(3) * 3.2, ph: r() * 6.28, sp: 0.3 + r() * 0.6 }));
      const act = (g0, u0) => {
        if (!S.on) return silu(g0) * u0;
        const u1 = V.clamp(u0, -S.L, S.L);
        return S.sem === 'ref' ? silu(Math.min(g0, S.L)) * u1 : Math.min(silu(g0), S.L) * u1;
      };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Limite L', min: 1, max: 20, step: 0.5, value: 10, fmt: (v) => nf(v, 1), onInput: (v) => { S.L = v; dirty = true; upd(); } });
      UI.toggle(ctl, { label: 'trava ligada', value: true, onChange: (v) => { S.on = v; dirty = true; upd(); } });
      UI.seg(ctl, { label: 'Semântica', options: [{ v: 'ref', label: 'SiLU(min(g,L))' }, { v: 'coop', label: 'min(SiLU(g),L)' }], value: 'ref', onChange: (v) => { S.sem = v; dirty = true; upd(); } });
      const rMax = UI.readout(ctl, 'máx |act| (g,u ≤ 20)', '—');
      const rPct = UI.readout(ctl, 'amostras travadas', '—');
      const rDif = UI.readout(ctl, 'dif. entre semânticas', '—');
      function upd() {
        const m = S.on ? silu(S.L) * S.L : silu(20) * 20;
        rMax.set(nf(m, 3));
        const cl = samples.filter((p) => S.on && (p.g > S.L || Math.abs(p.u) > S.L)).length;
        rPct.set(nf((100 * cl) / N, 1) + '%');
        rDif.set(fx(S.L * (1 - sig(S.L)), 1));
      }
      upd();
      // mapa de calor em canvas fora da tela
      const HM = 56;
      const off = document.createElement('canvas'); off.width = HM; off.height = HM;
      const og = off.getContext('2d');
      let dirty = true, lastDark = null;
      function paintHeat() {
        const C = V.C();
        const img = og.createImageData(HM, HM);
        const lg = Math.log1p(400);
        for (let a = 0; a < HM; a++) for (let b = 0; b < HM; b++) {
          const gg = -20 + (40 * (b + 0.5)) / HM, uu = 20 - (40 * (a + 0.5)) / HM;
          const v = act(gg, uu);
          const tv = Math.sign(v) * Math.log1p(Math.abs(v)) / lg;
          const col = V.cmap(tv, 'div');
          const m = col.match(/\d+/g).map(Number);
          const k = (a * HM + b) * 4;
          img.data[k] = m[0]; img.data[k + 1] = m[1]; img.data[k + 2] = m[2]; img.data[k + 3] = 255;
        }
        og.putImageData(img, 0, 0);
        dirty = false; lastDark = C.dark;
      }
      F(fig, (g, w, h, t) => {
        const C = V.C();
        if (dirty || lastDark !== C.dark) paintHeat();
        const narrow = w < 620;
        // painéis
        let pA, pB, hmR;
        if (!narrow) {
          const lw = Math.min(360, w * 0.42);
          pA = { x: 46, y: 30, w: lw - 50, h: (h - 90) / 2 };
          pB = { x: 46, y: 30 + (h - 90) / 2 + 40, w: lw - 50, h: (h - 90) / 2 };
          const s = Math.min(w - lw - 40, h - 60);
          hmR = { x: lw + 30, y: 30, s };
        } else {
          pA = { x: 46, y: 28, w: w - 62, h: 120 };
          pB = { x: 46, y: 190, w: w - 62, h: 110 };
          const s = Math.min(w - 60, h - 370);
          hmR = { x: (w - s) / 2 + 10, y: 344, s };
        }
        // painel A: caminho do gate
        const plot = (R, xmin, xmax, ymin, ymax, fns, title, xl) => {
          V.box(g, R.x, R.y, R.w, R.h, { r: 6, fill: V.alpha(C.panel, 0.5), stroke: C.grid, lw: 1 });
          const X = (v) => R.x + ((v - xmin) / (xmax - xmin)) * R.w, Y = (v) => R.y + R.h - ((v - ymin) / (ymax - ymin)) * R.h;
          V.line(g, X(0), R.y, X(0), R.y + R.h, { color: V.alpha(C.muted, 0.4), lw: 1 });
          V.line(g, R.x, Y(0), R.x + R.w, Y(0), { color: V.alpha(C.muted, 0.4), lw: 1 });
          fns.forEach((fo) => {
            g.beginPath();
            for (let q = 0; q <= 120; q++) { const xv = xmin + ((xmax - xmin) * q) / 120; const yv = V.clamp(fo.f(xv), ymin, ymax); const xx = X(xv), yy = Y(yv); if (q === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); }
            g.strokeStyle = fo.c; g.lineWidth = fo.lw || 2; if (fo.dash) g.setLineDash(fo.dash); g.stroke(); g.setLineDash([]);
          });
          if (S.on) { V.line(g, X(S.L), R.y, X(S.L), R.y + R.h, { color: V.alpha(C.c, 0.7), lw: 1, dash: [3, 3] }); }
          T(g, title, R.x + 6, R.y + 14, { size: 11.5, weight: 600, color: C.ink });
          T(g, xl, R.x + R.w, R.y + R.h + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, nf(xmin, 0), R.x, R.y + R.h + 14, { size: 11, color: C.muted, font: 'mono' });
          T(g, nf(ymax, 0), R.x - 4, R.y + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, nf(ymin, 0), R.x - 4, R.y + R.h, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          return { X, Y };
        };
        const gateF = (z) => (!S.on ? silu(z) : S.sem === 'ref' ? silu(Math.min(z, S.L)) : Math.min(silu(z), S.L));
        const A = plot(pA, -15, 25, -3, 25, [{ f: silu, c: V.alpha(C.muted, 0.8), dash: [4, 4], lw: 1.5 }, { f: gateF, c: C.a, lw: 2.5 }], 'gate: SiLU(min(g, L))', 'g →');
        const B = plot(pB, -25, 25, -25, 25, [{ f: (z) => z, c: V.alpha(C.muted, 0.8), dash: [4, 4], lw: 1.5 }, { f: (z) => (S.on ? V.clamp(z, -S.L, S.L) : z), c: C.b, lw: 2.5 }], 'up: clamp(u, −L, L)', 'u →');
        // amostras deslizando pelas curvas
        samples.slice(0, 40).forEach((p, q) => {
          const gq = p.g + Math.sin(t * p.sp + p.ph) * 0.6;
          const xg = V.clamp(gq, -15, 25);
          const cl = S.on && gq > S.L;
          V.dot(g, A.X(xg), A.Y(V.clamp(gateF(gq), -3, 25)), cl ? 3.2 : 2.2, cl ? C.c : V.alpha(C.a, 0.7));
          const uq = p.u + Math.cos(t * p.sp + p.ph) * 0.6;
          const cu = S.on && Math.abs(uq) > S.L;
          V.dot(g, B.X(V.clamp(uq, -25, 25)), B.Y(V.clamp(S.on ? V.clamp(uq, -S.L, S.L) : uq, -25, 25)), cu ? 3.2 : 2.2, cu ? C.c : V.alpha(C.b, 0.7));
        });
        // mapa de calor
        const s = hmR.s;
        g.imageSmoothingEnabled = false;
        g.drawImage(off, hmR.x, hmR.y, s, s);
        g.imageSmoothingEnabled = true;
        g.strokeStyle = C.grid; g.lineWidth = 1; g.strokeRect(hmR.x, hmR.y, s, s);
        const GX = (v) => hmR.x + ((v + 20) / 40) * s, GY = (v) => hmR.y + ((20 - v) / 40) * s;
        if (S.on && S.L < 20) { g.setLineDash([5, 4]); g.strokeStyle = C.ink; g.lineWidth = 1.3; g.strokeRect(GX(-20), GY(S.L), GX(S.L) - GX(-20), GY(-S.L) - GY(S.L)); g.setLineDash([]); }
        V.line(g, GX(0), hmR.y, GX(0), hmR.y + s, { color: V.alpha(C.ink, 0.25), lw: 1 });
        V.line(g, hmR.x, GY(0), hmR.x + s, GY(0), { color: V.alpha(C.ink, 0.25), lw: 1 });
        T(g, 'act(g, u)', hmR.x, hmR.y - 8, { size: 11.5, weight: 600, color: C.ink });
        T(g, 'g →', hmR.x + s, hmR.y + s + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        T(g, 'u ↑', hmR.x - 6, hmR.y + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        samples.forEach((p) => {
          const gq = V.clamp(p.g + Math.sin(t * p.sp + p.ph) * 0.6, -20, 20), uq = V.clamp(p.u + Math.cos(t * p.sp + p.ph) * 0.6, -20, 20);
          const cg = S.on ? Math.min(gq, S.L) : gq, cu = S.on ? V.clamp(uq, -S.L, S.L) : uq;
          const cl = cg !== gq || cu !== uq;
          if (cl) { V.line(g, GX(gq), GY(uq), GX(cg), GY(cu), { color: V.alpha(C.c, 0.6), lw: 1 }); V.dot(g, GX(gq), GY(uq), 1.6, V.alpha(C.c, 0.5)); }
          V.dot(g, GX(cg), GY(cu), cl ? 3 : 2, cl ? C.c : C.ink);
        });
        const mxs = S.on ? '|act| ≤ ' + nf(silu(S.L) * S.L, 3) : 'sem trava: 400 no canto e sem limite';
        T(g, mxs, hmR.x + s, hmR.y - 8, { size: 11, align: 'right', color: S.on ? C.a : C.c, font: 'mono' });
      });
    });

    /* ====================================================================
       Figura 7 — Calculadora de bloco
       ==================================================================== */
    safe('calc', () => {
      const fig = $('#fig-calc');
      fitH(fig, (w) => (w < 620 ? 650 : 470));
      const PRE = {
        classic: { d: 4096, h: 32, kv: 32, dh: 128, I: 11008, moe: false, E: 8, k: 2, gate: false, N: 32, V: 32000 },
        qwen: { d: 2560, h: 24, kv: 2, dh: 256, I: 640, moe: true, E: 512, k: 10, gate: true, N: 48, V: 248320 },
        wide: { d: 8192, h: 64, kv: 8, dh: 128, I: 28672, moe: false, E: 8, k: 2, gate: false, N: 80, V: 128000 },
      };
      const S = Object.assign({ ne: 13, ang: 0, preset: 'classic' }, PRE.classic);
      const ctl = UI.controls(fig);
      const sl = {};
      let loading = false;
      const rd = () => { if (!loading) H.redraw(); };
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'classic', label: 'Clássico (ilustr.)' }, { v: 'qwen', label: 'Qwen · QSA + MoE' }, { v: 'wide', label: 'Denso largo (ilustr.)' }], value: 'classic', onChange: (v) => { loadPreset(v); } });
      const ffnSeg = UI.seg(ctl, { label: 'FFN', options: [{ v: false, label: 'densa' }, { v: true, label: 'MoE' }], value: false, onChange: (v) => { S.moe = v; rd(); } });
      const gateT = UI.toggle(ctl, { label: 'gate de saída (dobra o q_proj)', value: false, onChange: (v) => { S.gate = v; rd(); } });
      sl.d = UI.slider(ctl, { label: 'd (hidden)', min: 256, max: 8192, step: 256, value: S.d, fmt: ni, onInput: (v) => { S.d = v; rd(); } });
      sl.h = UI.slider(ctl, { label: 'cabeças h', min: 1, max: 128, step: 1, value: S.h, onInput: (v) => { S.h = v; rd(); } });
      sl.kv = UI.slider(ctl, { label: 'cabeças KV', min: 1, max: 128, step: 1, value: S.kv, onInput: (v) => { S.kv = v; rd(); } });
      sl.dh = UI.slider(ctl, { label: 'head_dim', min: 32, max: 512, step: 32, value: S.dh, onInput: (v) => { S.dh = v; rd(); } });
      sl.I = UI.slider(ctl, { label: 'I (FFN/expert)', min: 128, max: 32768, step: 128, value: S.I, fmt: ni, onInput: (v) => { S.I = v; rd(); } });
      sl.E = UI.slider(ctl, { label: 'experts E', min: 2, max: 512, step: 2, value: S.E, onInput: (v) => { S.E = v; rd(); } });
      sl.k = UI.slider(ctl, { label: 'top-k', min: 1, max: 16, step: 1, value: S.k, onInput: (v) => { S.k = v; rd(); } });
      sl.N = UI.slider(ctl, { label: 'camadas N', min: 1, max: 96, step: 1, value: S.N, onInput: (v) => { S.N = v; rd(); } });
      sl.ne = UI.slider(ctl, { label: 'contexto n', min: 10, max: 20, step: 0.5, value: S.ne, fmt: (v) => ni(Math.pow(2, v)), onInput: (v) => { S.ne = v; rd(); } });
      const rBlk = UI.readout(ctl, 'bloco', '—');
      const rTot = UI.readout(ctl, 'modelo (× N + emb.)', '—');
      const rFl = UI.readout(ctl, 'GFLOP/token', '—');
      function loadPreset(p) {
        const P0 = PRE[p]; S.preset = p; loading = true;
        Object.assign(S, { V: P0.V });
        ['d', 'h', 'kv', 'dh', 'I', 'E', 'k', 'N'].forEach((k) => sl[k].set(P0[k]));
        ffnSeg.set(P0.moe); gateT.set(P0.gate);
        loading = false; H.redraw();
      }
      function calc() {
        const kv = Math.min(S.kv, S.h);
        const Pq = S.d * S.h * S.dh * (S.gate ? 2 : 1), Pk = S.d * kv * S.dh, Pv = Pk, Po = S.h * S.dh * S.d;
        const Pattn = Pq + Pk + Pv + Po;
        const Pe = 3 * S.d * S.I;
        const k = Math.min(S.k, S.E);
        const Pffn = S.moe ? (S.E + 1) * Pe + S.d * S.E : Pe;
        const Pffa = S.moe ? (k + 1) * Pe + S.d * S.E : Pe;
        const Pn = 2 * S.d;
        const blk = Pattn + Pffn + Pn, act = Pattn + Pffa + Pn;
        const n = Math.pow(2, S.ne);
        const fProj = 2 * S.N * Pattn, fFfn = 2 * S.N * Pffa, fSc = 4 * S.N * S.h * S.dh * n, fHead = 2 * S.V * S.d;
        const tot = S.N * blk + 2 * S.V * S.d + S.d;
        const nstar = act / (2 * S.h * S.dh);
        return { kv, Pq, Pk, Pv, Po, Pattn, Pe, Pffn, Pffa, Pn, blk, act, n, fProj, fFfn, fSc, fHead, tot, nstar, k };
      }
      const H = F(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const R = calc();
        rBlk.set(pp(R.blk) + (S.moe ? ' (ativos ' + pp(R.act) + ')' : ''));
        rTot.set(pp(R.tot));
        const fl = R.fProj + R.fFfn + R.fSc + R.fHead;
        rFl.set(nf(fl / 1e9, fl / 1e9 >= 100 ? 0 : 1));
        const narrow = w < 620;
        // --- balança
        const A = R.Pattn, F = R.Pffn;
        const target = V.clamp((F - A) / (F + A), -1, 1) * 0.32;
        S.ang = V.lerp(S.ang, target, Math.min(1, (dt || 0.016) * 6));
        if (Math.abs(S.ang - target) > 0.002) H.redraw();
        const bw2 = narrow ? w * 0.36 : Math.min(170, w * 0.19);
        const px = narrow ? w / 2 : 30 + bw2 + 20, py = narrow ? 54 : 62;
        T(g, 'Parâmetros de um bloco', 14, 20, { size: 12.5, weight: 600, color: C.ink });
        const ca = Math.cos(S.ang), sa = Math.sin(S.ang);
        const Lx = px - bw2 * ca, Ly = py - bw2 * sa, Rx = px + bw2 * ca, Ry = py + bw2 * sa;
        g.fillStyle = V.alpha(C.muted, 0.5); g.beginPath(); g.moveTo(px, py); g.lineTo(px - 14, py + 70); g.lineTo(px + 14, py + 70); g.closePath(); g.fill();
        V.line(g, Lx, Ly, Rx, Ry, { color: C.ink, lw: 3 });
        V.dot(g, px, py, 5, C.ink);
        const pan = (x, y, col, lbl, val, sub) => {
          const pyy = y + 40;
          V.line(g, x, y, x - 26, pyy, { color: V.alpha(C.muted, 0.7), lw: 1 }); V.line(g, x, y, x + 26, pyy, { color: V.alpha(C.muted, 0.7), lw: 1 });
          g.beginPath(); g.moveTo(x - 32, pyy); g.quadraticCurveTo(x, pyy + 18, x + 32, pyy); g.closePath(); g.fillStyle = V.alpha(col, 0.55); g.fill(); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
          const frac = val / (A + F);
          const bh = 6 + 34 * Math.sqrt(frac);
          V.box(g, x - 13, pyy - bh, 26, bh, { r: 3, fill: col });
          T(g, lbl, x, pyy + 30, { size: 11.5, weight: 600, align: 'center', color: C.ink });
          T(g, pp(val) + ' (' + nf(frac * 100, frac < 0.1 ? 1 : 0) + '%)', x, pyy + 45, { size: 11, align: 'center', font: 'mono', color: C.muted });
          if (sub) T(g, sub, x, pyy + 59, { size: 11, align: 'center', font: 'mono', color: col });
        };
        pan(Lx, Ly, C.b, 'Atenção', A, S.gate ? 'q_proj com gate' : 'GQA ' + S.h + 'Q/' + R.kv + 'KV');
        pan(Rx, Ry, C.a, S.moe ? 'FFN MoE' : 'FFN SwiGLU', F, S.moe ? 'ativos ' + pp(R.Pffa) : 'I = ' + ni(S.I));
        // --- barra de composição
        let cx0, cy0, cwid;
        if (narrow) { cx0 = 14; cy0 = 224; cwid = w - 28; } else { cx0 = px + bw2 + 60; cy0 = 44; cwid = w - cx0 - 16; }
        const segs = [['Q' + (S.gate ? '+gate' : ''), R.Pq, C.b], ['K', R.Pk, C.c], ['V', R.Pv, C.e], ['O', R.Po, C.d], [S.moe ? 'experts + shared + roteador' : 'gate/up/down', R.Pffn, C.a]];
        T(g, 'composição do bloco (' + pp(R.blk) + ')', cx0, cy0 - 6, { size: 11.5, color: C.muted });
        let xx = cx0;
        const bh0 = 22;
        segs.forEach(([nm, v, col]) => {
          const ww = (v / R.blk) * cwid;
          V.rr(g, xx, cy0, Math.max(1, ww - 1), bh0, 3); g.fillStyle = col; g.fill();
          xx += ww;
        });
        // legenda da composição
        let lx = cx0, ly = cy0 + bh0 + 16;
        segs.forEach(([nm, v, col]) => {
          const s = nm + ' ' + nf((100 * v) / R.blk, (100 * v) / R.blk < 10 ? 1 : 0) + '%';
          const lw = textW(g, s, 11) + 22;
          if (lx + lw > cx0 + cwid) { lx = cx0; ly += 15; }
          g.fillStyle = col; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill();
          T(g, s, lx + 14, ly, { size: 11, color: C.muted });
          lx += lw;
        });
        if (!narrow) {
          const cl = S.moe ? `MoE: E = ${S.E}, top-${R.k} + 1 compartilhado; cada expert 3·d·I = ${pp(R.Pe)}` : `FFN = 3·d·I = ${pp(R.Pffn)}; atenção = ${pp(R.Pattn)}`;
          wrap(g, cl, cx0, ly + 20, cwid, 14, { size: 11, color: C.muted, font: 'mono' });
          if (S.kv > S.h) T(g, 'cabeças KV limitadas a h', cx0, ly + 50, { size: 11, color: C.c });
          if (S.preset === 'qwen' && !narrow) wrap(g, 'Qwen real: 12 de 48 camadas são QSA, as outras 36 são GDN; o “× N” supõe blocos iguais.', cx0, ly + 66, cwid, 14, { size: 11, color: C.muted });
        }
        // --- FLOPs
        const fy = narrow ? 330 : 238;
        T(g, 'FLOPs por token (modelo inteiro, contexto n = ' + ni(R.n) + ')', 14, fy, { size: 12.5, weight: 600, color: C.ink, maxW: w - 28 });
        const fsegs = [['projeções da atenção', R.fProj, C.b], ['FFN (ativos)', R.fFfn, C.a], ['scores QKᵀ e AV', R.fSc, C.c], ['lm_head', R.fHead, C.d]];
        let fx = 14; const fw = w - 28, fbh = 24, fby = fy + 12;
        fsegs.forEach(([nm, v, col]) => { const ww = (v / fl) * fw; V.rr(g, fx, fby, Math.max(1, ww - 1), fbh, 3); g.fillStyle = col; g.fill(); if (ww > 60) T(g, nf((100 * v) / fl, 0) + '%', fx + ww / 2, fby + fbh / 2 + 4, { size: 11, align: 'center', color: onTxt(C), weight: 600 }); fx += ww; });
        lx = 14; ly = fby + fbh + 16;
        fsegs.forEach(([nm, v, col]) => {
          const s = nm + ' ' + nf(v / 1e9, v / 1e9 >= 10 ? 0 : 2) + ' G';
          const lw = textW(g, s, 11) + 22;
          if (lx + lw > w - 14) { lx = 14; ly += 15; }
          g.fillStyle = col; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill();
          T(g, s, lx + 14, ly, { size: 11, color: C.muted });
          lx += lw;
        });
        // curva da fração da atenção vs contexto
        const gy0 = ly + 22, gy1 = h - 26, gx0 = 50, gx1 = w - 16;
        if (gy1 - gy0 > 50) {
          V.box(g, gx0, gy0, gx1 - gx0, gy1 - gy0, { r: 4, fill: V.alpha(C.panel, 0.4), stroke: C.grid, lw: 1 });
          const X = (e) => gx0 + ((e - 10) / 10) * (gx1 - gx0), Y = (fr) => gy1 - fr * (gy1 - gy0);
          const shareAt = (e) => { const n = Math.pow(2, e); const sc = 4 * S.N * S.h * S.dh * n; return sc / (R.fProj + R.fFfn + R.fHead + sc); };
          V.line(g, gx0, Y(0.5), gx1, Y(0.5), { color: V.alpha(C.muted, 0.35), lw: 1, dash: [3, 4] });
          g.beginPath();
          for (let q = 0; q <= 100; q++) { const e = 10 + q / 10; const xv = X(e), yv = Y(shareAt(e)); if (q === 0) g.moveTo(xv, yv); else g.lineTo(xv, yv); }
          g.strokeStyle = C.c; g.lineWidth = 2.2; g.stroke();
          const ens = Math.log2(Math.max(1, R.nstar));
          if (ens >= 10 && ens <= 20) { V.line(g, X(ens), gy0, X(ens), gy1, { color: C.d, lw: 1.3, dash: [4, 3] }); T(g, 'n* ≈ ' + ni(R.nstar), X(ens) + (X(ens) > gx1 - 100 ? -4 : 4), gy0 + 13, { size: 11, color: C.d, font: 'mono', align: X(ens) > gx1 - 100 ? 'right' : 'left' }); }
          else T(g, 'n* ≈ ' + ni(R.nstar) + (ens < 10 ? ' (antes de 1K)' : ' (além de 1M)'), gx1 - 4, gy0 + 13, { size: 11, color: C.d, font: 'mono', align: 'right' });
          const cur = shareAt(S.ne);
          V.dot(g, X(S.ne), Y(cur), 5, C.c); V.glow(g, X(S.ne), Y(cur), 14, C.c, 0.6);
          T(g, '100%', gx0 - 4, gy0 + 10, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, '0', gx0 - 4, gy1, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, '1K', gx0, gy1 + 14, { size: 11, color: C.muted, font: 'mono' });
          T(g, '1M', gx1, gy1 + 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
          T(g, 'fração dos FLOPs nos scores da atenção', (gx0 + gx1) / 2, gy1 + 14, { size: 11, align: 'center', color: C.muted, maxW: gx1 - gx0 - 70 });
        }
      }, { static: true });
    });

    /* ====================================================================
       Figura 8 — O mapa de metrô das camadas
       ==================================================================== */
    safe('metro', () => {
      const fig = $('#fig-metro');
      fitH(fig, (w) => (w < 620 ? 640 : 410));
      const C0 = V.C();
      function build(m) {
        const L = [];
        if (m === 'glm') {
          for (let i = 0; i < 45; i++) {
            const mla = i % 4 === 3 && i < 44;
            const dense = i < 3;
            L.push({ i, mix: mla ? 'MLA' : 'KDA', mixc: mla ? 'b' : 'a', ffn: dense ? 'densa' : 'MoE', attn: mla ? 124914432 : 137732288, ffnP: dense ? 150994944 : 7274103072, ffnA: dense ? 150994944 : 227672352, hc: 786486, tap: [5, 14, 24, 33, 42].includes(i), note: dense ? 'FFN SwiGLU densa, I = 12.288 (first_k_dense_replace = 3)' : mla ? 'MLA sem RoPE (NoPE) + indexador esparso top-2048' : 'KDA: atenção linear com estado recorrente fixo' });
          }
          L.push({ i: 45, mix: 'MLA', mixc: 'b', ffn: 'MoE', attn: 124914432, ffnP: 7274103072, ffnA: 227672352, hc: 0, mtp: true, note: 'bloco MTP (índice 45): MLA + MoE com resíduo simples, ≈ 7,43 B no total; usado como rascunhador com SPEC_METHOD=mtp' });
          return { L, types: [['KDA', 'a'], ['MLA esparsa', 'b']], hcl: 'mHC', tapNote: 'saída lida pelo DFlash2 (hook EAGLE3 com hc_post + hc_contract)' };
        }
        if (m === 'qwen') {
          for (let i = 0; i < 48; i++) {
            const qsa = (i + 1) % 4 === 0;
            L.push({ i, mix: qsa ? 'QSA' : 'GDN', mixc: qsa ? 'c' : 'e', ffn: 'MoE', attn: qsa ? 51446528 : 57958624, ffnP: 2522810880, ffnA: 55380480, hc: 13209600, ple: i === 1, note: qsa ? 'QSA: atenção softmax esparsa (≤ 2048 tokens), GQA 24Q/2KV × 256, com gate de saída' : 'Gated DeltaNet: atenção linear com estado 128×128 por cabeça de V' });
          }
          L.push({ i: 48, mix: 'QSA', mixc: 'c', ffn: 'MoE', attn: 51446528, ffnP: 2522810880, ffnA: 55380480, hc: 13209600, mtp: true, note: 'camada MTP (checkpoint mtp.layers.0, instanciada como mtp.layers.48): QSA + MoE com experts FP8 em blocos 128×128' });
          return { L, types: [['GDN', 'e'], ['QSA', 'c']], hcl: 'Gated Residual', tapNote: '' };
        }
        for (let i = 0; i < 43; i++) {
          const swa = i < 2, c4 = !swa && i % 2 === 0;
          const attn = swa ? 107.0e6 : c4 ? 126.1e6 : 111.3e6;
          L.push({ i, mix: swa ? 'SWA' : c4 ? 'C4A' : 'C128A', mixc: swa ? 'e' : c4 ? 'c' : 'b', ffn: i < 3 ? 'hash' : 'MoE', attn, ffnP: 6.469e9, ffnA: 177209344, hc: 0.79e6, tap: i >= 40, approx: true, note: swa ? 'SWA pura (janela 128)' : c4 ? 'CSA/C4A: KV comprimido 4× + indexador top-512 + janela 128' : 'HCA/C128A: KV comprimido 128× (denso) + janela 128' });
        }
        for (let s = 0; s < 3; s++) L.push({ i: 43 + s, mix: 'SWA', mixc: 'e', ffn: 'MoE', attn: 0, ffnP: 0, ffnA: 0, hc: 0, mtp: true, approx: true, note: 'estágio DSpark ' + s + ' (mtp.' + s + '): SWA sobre as features do alvo + MoE completo; ≈ 19,8 B nos 3 estágios' });
        return { L, types: [['SWA', 'e'], ['C4A (esparsa)', 'c'], ['C128A', 'b']], hcl: 'mHC', tapNote: 'camada-alvo do DSpark (média dos 4 fluxos → main_proj)' };
      }
      const S = { m: 'glm', data: build('glm'), lock: -1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; S.data = build(v); S.lock = -1; } });
      UI.button(ctl, 'soltar seleção', () => { S.lock = -1; });
      UI.legend(ctl, [[C0.d, 'anel âmbar: FFN densa'], [C0.muted, 'anel pontilhado: MoE'], [C0.f, 'anel duplo: MoE por hash']]);
      const H = F(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const D = S.data, L = D.L;
        const narrow = w < 620;
        const sp = narrow ? 36 : 36;
        const R = Math.max(5, Math.floor((w - 50) / sp) + 1);
        const rows = Math.ceil(L.length / R);
        const rowH = 56;
        // legenda de tipos
        let lx = 14; let ly = 22;
        D.types.forEach(([nm, c]) => { const s = nm; const lw = textW(g, s, 11.5) + 26; if (lx + lw > w - 10) { lx = 14; ly += 16; } g.fillStyle = C[c]; g.beginPath(); g.arc(lx + 6, ly - 4, 6, 0, 6.3); g.fill(); T(g, s, lx + 16, ly, { size: 11.5, color: C.muted }); lx += lw; });
        const mk = [['◆', 'rascunhador lê'], ['✚', 'N-gram'], ['★', 'MTP/DSpark']];
        mk.forEach(([s1, s2]) => { const s = s1 + ' ' + s2; const lw = textW(g, s, 11.5) + 18; if (lx + lw > w - 10) { lx = 14; ly += 16; } T(g, s, lx, ly, { size: 11.5, color: C.muted }); lx += lw; });
        const top = ly + 22;
        const pos = (k) => { const r = Math.floor(k / R), c = k % R; const cc = r % 2 === 0 ? c : R - 1 - c; return [25 + cc * ((w - 50) / (R - 1)), top + 16 + r * rowH]; };
        // linha
        g.beginPath();
        L.forEach((_, k) => { const [x, y] = pos(k); if (k === 0) g.moveTo(x, y); else { const [px, py] = pos(k - 1); if (py !== y) { const ex = px; g.bezierCurveTo(ex + (x > w / 2 ? 24 : -24), py, ex + (x > w / 2 ? 24 : -24), y, x, y); } else g.lineTo(x, y); } });
        g.strokeStyle = V.alpha(C.muted, 0.45); g.lineWidth = 4; g.stroke();
        // trem
        const speed = 2.2;
        const tpos = (t * speed) % (L.length + 2);
        const ti = Math.min(L.length - 1, Math.floor(tpos));
        // seleção por ponteiro
        let hov = -1;
        if (Hh.pointer.inside) L.forEach((_, k) => { const [x, y] = pos(k); if (Math.hypot(Hh.pointer.x - x, Hh.pointer.y - y) < 14) hov = k; });
        if (Hh.pointer.clicked && hov >= 0) S.lock = hov;
        const sel = hov >= 0 ? hov : S.lock >= 0 && S.lock < L.length ? S.lock : ti;
        L.forEach((s, k) => {
          const [x, y] = pos(k);
          const col = C[s.mixc] || C.a;
          const act = k === sel;
          if (s.ffn === 'densa') { g.beginPath(); g.arc(x, y, 12, 0, 6.3); g.strokeStyle = C.d; g.lineWidth = 3; g.stroke(); }
          else if (s.ffn === 'hash') { g.beginPath(); g.arc(x, y, 11, 0, 6.3); g.strokeStyle = C.f; g.lineWidth = 1.5; g.stroke(); g.beginPath(); g.arc(x, y, 14, 0, 6.3); g.stroke(); }
          else { g.setLineDash([2, 3]); g.beginPath(); g.arc(x, y, 12, 0, 6.3); g.strokeStyle = V.alpha(C.muted, 0.9); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]); }
          g.beginPath(); g.arc(x, y, act ? 9 : 7.5, 0, 6.3); g.fillStyle = s.mtp ? V.alpha(col, 0.45) : col; g.fill();
          if (act) { g.strokeStyle = C.ink; g.lineWidth = 2; g.stroke(); }
          if (k === ti) V.glow(g, x, y, 18, C.ink, 0.35);
          if (s.tap) T(g, '◆', x, y - 17, { size: 11, align: 'center', color: C.d });
          if (s.ple) T(g, '✚', x, y - 17, { size: 12, align: 'center', color: C.d });
          if (s.mtp) T(g, '★', x, y - 17, { size: 12, align: 'center', color: C.f });
          if (s.i % 5 === 0 || act) T(g, String(s.i), x, y + 25, { size: 11, align: 'center', font: 'mono', color: act ? C.ink : C.muted });
        });
        // trem entre estações
        const fr = tpos - ti;
        if (ti < L.length - 1) { const a = pos(ti), b = pos(ti + 1); const p = V.along(a[0], a[1], b[0], b[1], V.smooth(V.clamp(fr, 0, 1))); V.dot(g, p[0], p[1], 4, C.ink); }
        // painel de informação
        const s = L[sel];
        const iy = top + 16 + rows * rowH - 10;
        const ih = h - iy - 8;
        V.box(g, 12, iy, w - 24, ih, { r: 10, fill: V.alpha(C.panel, 0.85), stroke: V.alpha(C[s.mixc] || C.a, 0.6), lw: 1.3 });
        const ttl = (s.mtp ? '' : 'Camada ') + (s.mtp ? 'Extra ' + s.i : s.i) + ' · ' + s.mix + ' · FFN ' + s.ffn + (S.lock >= 0 ? '  (fixada)' : '');
        T(g, ttl, 24, iy + 20, { size: 13, weight: 700, color: C[s.mixc] || C.a, maxW: w - 48 });
        let yy = iy + 38;
        yy += 15 * wrap(g, s.note, 24, yy, w - 48, 15, { size: 11.5, color: C.ink, maxLines: 3 });
        if (s.attn > 0) {
          const line = `atenção ${pp(s.attn)} · FFN ${pp(s.ffnP)} (ativos por token ${pp(s.ffnA)})` + (s.hc ? ` · ${D.hcl} ${pp(s.hc)}` : '') + (s.approx ? ' · aprox.' : '');
          yy += 15 * wrap(g, line, 24, yy + 4, w - 48, 15, { size: 11.5, font: 'mono', color: C.muted, maxLines: 3 }) + 6;
          // barras de proporção
          const bx = 24, bwid = w - 48;
          const tot = s.attn + s.ffnP, ta = s.attn + s.ffnA;
          if (yy + 34 < iy + ih) {
            const b1 = (s.attn / tot) * bwid;
            V.rr(g, bx, yy, b1, 10, 3); g.fillStyle = C[s.mixc] || C.a; g.fill();
            V.rr(g, bx + b1, yy, bwid - b1, 10, 3); g.fillStyle = V.alpha(C.ink, 0.28); g.fill();
            T(g, 'total: atenção ' + nf((100 * s.attn) / tot, 1) + '% · FFN ' + nf((100 * s.ffnP) / tot, 1) + '%', bx, yy + 24, { size: 11, color: C.muted });
            if (yy + 52 < iy + ih) {
              const b2 = (s.attn / ta) * bwid;
              V.rr(g, bx, yy + 32, b2, 10, 3); g.fillStyle = C[s.mixc] || C.a; g.fill();
              V.rr(g, bx + b2, yy + 32, bwid - b2, 10, 3); g.fillStyle = V.alpha(C.ink, 0.55); g.fill();
              T(g, 'ativos por token: atenção ' + nf((100 * s.attn) / ta, 0) + '% · FFN ' + nf((100 * s.ffnA) / ta, 0) + '%', bx, yy + 56, { size: 11, color: C.muted });
            }
          }
        }
        const extra = s.tap ? '◆ ' + D.tapNote : s.ple ? '✚ a tabela N-gram (51,2 B parâmetros) soma seu vetor aos 4 ramos nesta camada' : '';
        if (extra) T(g, extra, 24, iy + ih - 10, { size: 11, color: C.d, maxW: w - 48 });
      });
    });

    /* ====================================================================
       Desenho esquemático do bloco (usado nas figuras 9 e 10)
       ==================================================================== */
    function drawBlock(g, bx, by, bw, bh, o) {
      const C = V.C();
      const busX = bx + 16;
      const hasHc = o.hc !== 'none';
      const parts = [];
      const add = (key, label, hh, col, extra) => parts.push({ key, label, hh, col, extra });
      if (hasHc) add('hc1', o.hc === 'gr' ? 'GR: leitura (normaliza)' : 'mHC: mistura 4 fluxos', 0.7, C.f);
      if (o.norms) add('norm1', o.n1 || 'input_layernorm', 0.6, C.e);
      add('mixer', o.mixer, 1.5, C.b, 'attn');
      if (hasHc) add('hc2', o.hc === 'gr' ? 'GR: escrita + leitura' : 'mHC: escreve e mistura', 0.7, C.f);
      if (o.norms) add('norm2', o.n2 || 'post_attention_layernorm', 0.6, C.e);
      add('ffn', o.ffn, 1.6, C.a, 'ffn');
      const tot = parts.reduce((s, p) => s + p.hh, 0);
      const gap = 7;
      const unit = (bh - 30 - gap * (parts.length - 1)) / tot;
      let y = by + 16;
      const rects = {};
      V.line(g, busX, by, busX, by + bh, { color: V.alpha(C.d, 0.8), lw: 4 });
      T(g, 'resíduo', busX - 4, by + bh + 12, { size: 11, color: C.d, align: 'left', font: 'mono' });
      const lit = o.lit || {};
      parts.forEach((p) => {
        const hh = p.hh * unit;
        const x = busX + 18, ww = bx + bw - x;
        const on = !!lit[p.key] || (p.key === 'mixer' && (lit['attn-in'] || lit['attn-out'])) || (p.key === 'ffn' && (lit['ffn-in'] || lit['ffn-out'] || lit.router || lit.experts));
        V.box(g, x, y, ww, hh, { r: 7, fill: V.alpha(p.col, on ? 0.34 : 0.1), stroke: V.alpha(p.col, on ? 1 : 0.45), lw: on ? 2.2 : 1 });
        T(g, p.label, x + 8, y + Math.min(hh / 2 + 4, 16), { size: fit(g, p.label, ww - 14, 12, 'sans', 600), weight: 600, color: C.ink });
        rects[p.key] = { x, y, w: ww, h: hh };
        if (p.extra === 'attn') {
          // sub-faixas: leitores (entrada) e escritor (saída)
          const r1 = { x: x + 8, y: y + hh - 20, w: (ww - 24) * 0.62, h: 14 }, r2 = { x: x + 16 + (ww - 24) * 0.62, y: y + hh - 20, w: (ww - 24) * 0.38, h: 14 };
          if (hh > 40) {
            V.box(g, r1.x, r1.y, r1.w, r1.h, { r: 4, fill: V.alpha(C.b, lit['attn-in'] ? 0.8 : 0.22) });
            V.box(g, r2.x, r2.y, r2.w, r2.h, { r: 4, fill: V.alpha(C.c, lit['attn-out'] ? 0.9 : 0.25) });
            T(g, o.attnIn || 'leitores', r1.x + 4, r1.y + 11, { size: 11, color: C.ink, maxW: r1.w - 6 });
            T(g, o.attnOut || 'o_proj', r2.x + 4, r2.y + 11, { size: 11, color: C.ink, maxW: r2.w - 6 });
          }
          rects['attn-in'] = r1; rects['attn-out'] = r2;
        }
        if (p.extra === 'ffn') {
          const r1 = { x: x + 8, y: y + hh - 20, w: (ww - 24) * 0.5, h: 14 }, r2 = { x: x + 16 + (ww - 24) * 0.5, y: y + hh - 20, w: (ww - 24) * 0.5, h: 14 };
          if (o.moe && hh > 56) {
            const rr = { x: x + 8, y: y + 22, w: 58, h: 14 };
            V.box(g, rr.x, rr.y, rr.w, rr.h, { r: 4, fill: V.alpha(C.d, lit.router ? 0.9 : 0.25) });
            T(g, 'roteador', rr.x + 4, rr.y + 11, { size: 11, color: C.ink, maxW: rr.w - 6 });
            rects.router = rr;
            const n = Math.max(4, Math.min(14, Math.floor((ww - 90) / 12)));
            for (let q = 0; q < n; q++) { g.fillStyle = V.alpha(C.a, lit.experts ? (q % 3 === 0 ? 0.95 : 0.45) : 0.25); V.rr(g, x + 74 + q * 12, y + 24, 9, 10, 2); g.fill(); }
            rects.experts = { x: x + 74, y: y + 22, w: n * 12, h: 14 };
          }
          if (hh > 40) {
            V.box(g, r1.x, r1.y, r1.w, r1.h, { r: 4, fill: V.alpha(C.a, lit['ffn-in'] ? 0.8 : 0.22) });
            V.box(g, r2.x, r2.y, r2.w, r2.h, { r: 4, fill: V.alpha(C.c, lit['ffn-out'] ? 0.9 : 0.25) });
            T(g, o.ffnIn || 'gate/up', r1.x + 4, r1.y + 11, { size: 11, color: C.ink, maxW: r1.w - 6 });
            T(g, o.ffnOut || 'down', r2.x + 4, r2.y + 11, { size: 11, color: C.ink, maxW: r2.w - 6 });
          }
          rects['ffn-in'] = r1; rects['ffn-out'] = r2;
        }
        // setas: lê do resíduo (entrada) e escreve (⊕)
        if (p.key === 'mixer' || p.key === 'ffn') {
          V.arrow(g, x + ww * 0.5, y + hh + 1, busX + 5, y + hh + gap * 0.5, { color: V.alpha(C.c, 0.7), lw: 1.5, head: 6 });
          g.beginPath(); g.arc(busX, y + hh + gap * 0.5, 6, 0, 6.3); g.fillStyle = C.panel; g.fill(); g.strokeStyle = C.d; g.lineWidth = 1.5; g.stroke();
          T(g, '+', busX, y + hh + gap * 0.5 + 4, { size: 12, align: 'center', color: C.d, weight: 700 });
        } else V.arrow(g, busX + 3, y + hh / 2, x - 2, y + hh / 2, { color: V.alpha(C.muted, 0.7), lw: 1.2, head: 5 });
        y += hh + gap;
      });
      rects.bus = { x: busX - 6, y: by, w: 12, h: bh };
      return rects;
    }

    /* ====================================================================
       Figura 9 — O armário de tensores
       ==================================================================== */
    safe('tensores', () => {
      const fig = $('#fig-tensores');
      fitH(fig, (w) => (w < 620 ? 680 : 470));
      const E = (n, s, t, p) => ({ n, s, t: t || '', p });
      const GLM = (i) => {
        const L = [E('hc_attn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'BF16/F32', 'hc1'), E('input_layernorm.weight', '[4096]', 'BF16', 'norm1')];
        if (i % 4 === 3 && i < 44) {
          L.push(E('self_attn.q_a_proj.weight', '[1536, 4096]', 'BF16', 'attn-in'), E('self_attn.q_a_layernorm.weight', '[1536]', 'BF16', 'attn-in'), E('self_attn.q_b_proj.weight', '[16384, 1536]', 'BF16', 'attn-in'), E('self_attn.kv_a_proj_with_mqa.weight', '[512, 4096]', 'BF16', 'attn-in'), E('self_attn.kv_a_layernorm.weight', '[512]', 'BF16', 'attn-in'), E('self_attn.kv_b_proj.weight', '[32768, 512]', 'BF16', 'attn-in'), E('self_attn.indexer.{wq_b,wk}.weight', '[4096,1536]·[128,4096]', 'BF16', 'attn-in'), E('self_attn.indexer.{k_norm,weights_proj}', '[128]·[32,4096]', 'BF16', 'attn-in'), E('self_attn.indexer.index_kpool_compress_{gate,ape}', '[128,4096]·[4,128]', 'BF16', 'attn-in'), E('self_attn.o_proj.weight', '[4096, 16384]', 'BF16', 'attn-out'));
        } else {
          L.push(E('self_attn.{q,k,v}_proj.weight', '[8192, 4096] ×3', 'BF16', 'attn-in'), E('self_attn.{q,k,v}_conv1d.weight', '[8192, 1, 4] ×3', 'BF16', 'attn-in'), E('self_attn.b_proj.weight', '[64, 4096]', 'BF16', 'attn-in'), E('self_attn.{f_a,f_b}_proj.weight', '[128,4096]·[8192,128]', 'BF16', 'attn-in'), E('self_attn.{A_log, dt_bias}', '[64]·[8192]', 'F32', 'attn-in'), E('self_attn.{g_a,g_b}_proj.weight', '[128,4096]·[8192,128]', 'BF16', 'attn-in'), E('self_attn.o_norm.weight', '[128]', 'BF16', 'attn-in'), E('self_attn.o_proj.weight', '[4096, 8192]', 'BF16', 'attn-out'));
        }
        L.push(E('hc_ffn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'BF16/F32', 'hc2'), E('post_attention_layernorm.weight', '[4096]', 'BF16', 'norm2'));
        if (i < 3) L.push(E('mlp.{gate,up}_proj.weight', '[12288, 4096] ×2', 'BF16', 'ffn-in'), E('mlp.down_proj.weight', '[4096, 12288]', 'BF16', 'ffn-out'));
        else L.push(E('mlp.gate.weight', '[288, 4096]', 'BF16', 'router'), E('mlp.gate.e_score_correction_bias', '[288]', 'F32', 'router'), E('mlp.shared_experts.{gate,up}_proj.weight', '[2048, 4096] ×2', 'BF16', 'ffn-in'), E('mlp.shared_experts.down_proj.weight', '[4096, 2048]', 'BF16', 'ffn-out'), E('mlp.experts.{0..287}.{gate,up}_proj.trellis (+suh,svh,mcg)', '[256,128,64]', 'I16 EXL3', 'experts'), E('mlp.experts.{0..287}.down_proj.trellis (+suh,svh,mcg)', '[128,256,64]', 'I16 EXL3', 'experts'));
        return L;
      };
      const QW = (i) => {
        const L = [E('attn_hyper_connection.{hc_norm,input_mix_*,block_inject}', '[10240]·[320,10240]·…', 'BF16', 'hc1')];
        if (i === 1) L.push(E('ple.ple_embedding.ngram_embedding.shard_{0..127}.weight', '[2500012, 160]', 'F8', 'ple'), E('ple.{key_proj,value_proj,conv1d,norm_*}', '[10240,2560]·[2560,2560]·…', 'BF16', 'ple'));
        if ((i + 1) % 4 === 0) L.push(E('self_attn.q_proj.weight  (query + gate)', '[12288, 2560]', 'BF16', 'attn-in'), E('self_attn.{k,v}_proj.weight', '[512, 2560] ×2', 'BF16', 'attn-in'), E('self_attn.{q,k}_norm.weight', '[256] ×2', 'BF16', 'attn-in'), E('self_attn.indexer.index_qk_proj.weight', '[640, 2560]', 'BF16', 'attn-in'), E('self_attn.indexer.{q,k}_layernorm.weight', '[128] ×2', 'BF16', 'attn-in'), E('self_attn.o_proj.weight', '[2560, 6144]', 'BF16', 'attn-out'));
        else L.push(E('linear_attn.in_proj_qkv.weight', '[10240, 2560]', 'BF16', 'attn-in'), E('linear_attn.in_proj_z.weight', '[6144, 2560]', 'BF16', 'attn-in'), E('linear_attn.in_proj_{a,b}.weight', '[48, 2560] ×2', 'BF16', 'attn-in'), E('linear_attn.conv1d.weight', '[10240, 1, 4]', 'BF16', 'attn-in'), E('linear_attn.{A_log, dt_bias}', '[48] ×2', 'BF16', 'attn-in'), E('linear_attn.norm.weight', '[128]', 'BF16', 'attn-in'), E('linear_attn.out_proj.weight', '[2560, 6144]', 'BF16', 'attn-out'));
        L.push(E('mlp_hyper_connection.{hc_norm,input_mix_*,block_inject}', '[10240]·[320,10240]·…', 'BF16', 'hc2'), E('mlp.gate.weight', '[512, 2560]', 'BF16', 'router'), E('mlp.shared_expert.{gate,up}_proj.weight', '[640, 2560] ×2', 'BF16', 'ffn-in'), E('mlp.shared_expert.down_proj.weight', '[2560, 640]', 'BF16', 'ffn-out'), E('mlp.shared_expert_gate.weight', '[1, 2560]', 'BF16', 'ffn-in'), E('mlp.experts.{0..511}.{gate,up}_proj.weight (+escalas)', '[640, 1280]', 'U8 2×FP4', 'experts'), E('mlp.experts.{0..511}.down_proj.weight (+escalas)', '[2560, 320]', 'U8 2×FP4', 'experts'));
        return L;
      };
      const DS = (i) => {
        const L = [E('hc_attn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'F32', 'hc1'), E('attn_norm.weight', '[4096]', '', 'norm1'), E('attn.wq_a.weight (+ .scale)', '[1024, 4096]', 'F8', 'attn-in'), E('attn.q_norm.weight', '[1024]', '', 'attn-in'), E('attn.wq_b.weight', '[32768, 1024]', 'F8', 'attn-in'), E('attn.wkv.weight', '[512, 4096]', 'F8', 'attn-in'), E('attn.kv_norm.weight', '[512]', '', 'attn-in'), E('attn.attn_sink', '[64]', 'F32', 'attn-in')];
        if (i >= 2) { const c4 = i % 2 === 0; L.push(E('attn.compressor.{wkv,wgate,ape,norm}', c4 ? '[1024,4096]×2·[4,1024]·[512]' : '[512,4096]×2·[128,512]·[512]', 'BF16', 'attn-in')); if (c4) L.push(E('attn.indexer.{wq_b,weights_proj,compressor.*}', '[8192,1024]·[64,4096]·…', '', 'attn-in')); }
        L.push(E('attn.wo_a.weight  (8 grupos)', '[8192, 4096]', 'F8', 'attn-out'), E('attn.wo_b.weight', '[4096, 8192]', 'F8', 'attn-out'), E('hc_ffn_{fn,base,scale}', '[24,16384]·[24]·[3]', 'F32', 'hc2'), E('ffn_norm.weight', '[4096]', '', 'norm2'), E('ffn.gate.weight', '[256, 4096]', '', 'router'));
        if (i < 3) L.push(E('ffn.gate.tid2eid  (roteamento por hash)', '[129280, 6]', 'I32', 'router')); else L.push(E('ffn.gate.bias  → e_score_correction_bias', '[256]', '', 'router'));
        L.push(E('ffn.shared_experts.{w1,w3}.weight  → gate_up_proj', '[2048, 4096] ×2', 'F8', 'ffn-in'), E('ffn.shared_experts.w2.weight  → down_proj', '[4096, 2048]', 'F8', 'ffn-out'), E('ffn.experts.{0..255}.{w1,w3}.weight (+ .scale)', '[2048, 4096]', 'FP4', 'experts'), E('ffn.experts.{0..255}.w2.weight (+ .scale)', '[4096, 2048]', 'FP4', 'experts'));
        return L;
      };
      const CL = () => [E('input_layernorm.weight', '[d]', '', 'norm1'), E('self_attn.q_proj.weight', '[h·dₕ, d]', '', 'attn-in'), E('self_attn.k_proj.weight', '[n_kv·dₕ, d]', '', 'attn-in'), E('self_attn.v_proj.weight', '[n_kv·dₕ, d]', '', 'attn-in'), E('self_attn.o_proj.weight', '[d, h·dₕ]', '', 'attn-out'), E('post_attention_layernorm.weight', '[d]', '', 'norm2'), E('mlp.gate_proj.weight', '[I, d]', '', 'ffn-in'), E('mlp.up_proj.weight', '[I, d]', '', 'ffn-in'), E('mlp.down_proj.weight', '[d, I]', '', 'ffn-out')];
      const MOD = {
        cl: { N: 32, pre: 'model.layers.{i}.', gl: 'model.embed_tokens [V,d] · model.norm [d] · lm_head [V,d]', gen: CL, blk: () => ({ hc: 'none', norms: true, mixer: 'atenção (MHA/GQA)', ffn: 'FFN SwiGLU', moe: false, attnIn: 'q/k/v_proj', attnOut: 'o_proj', ffnIn: 'gate/up_proj', ffnOut: 'down_proj' }), col: 'a' },
        glm: { N: 45, pre: 'model.language_model.layers.{i}.', gl: 'embed_tokens [154880,4096] · norm [4096] · lm_head [154880,4096]', gen: GLM, blk: (i) => ({ hc: 'mhc', norms: true, mixer: i % 4 === 3 && i < 44 ? 'MLA NoPE + indexador' : 'KDA (atenção linear)', ffn: i < 3 ? 'FFN SwiGLU densa (12288)' : 'MoE: 288 experts + 1 shared', moe: i >= 3, attnIn: i % 4 === 3 && i < 44 ? 'q/kv latentes + idx' : 'q/k/v + gates', attnOut: 'o_proj' }), col: 'glm' },
        qwen: { N: 48, pre: 'model.language_model.layers.{i}.', gl: 'embed_tokens [248320,2560] · hyper_connection_mixer · lm_head [248320,2560]', gen: QW, blk: (i) => ({ hc: 'gr', norms: false, mixer: (i + 1) % 4 === 0 ? 'QSA (atenção esparsa)' : 'Gated DeltaNet', ffn: 'MoE: 512 experts + 1 shared', moe: true, attnIn: (i + 1) % 4 === 0 ? 'q(+gate)/k/v + indexer' : 'in_proj_qkv/z/a/b', attnOut: (i + 1) % 4 === 0 ? 'o_proj' : 'out_proj' }), col: 'qwen' },
        ds: { N: 43, pre: 'layers.{i}.  (vLLM: model.layers.{i}.)', gl: 'embed → embed_tokens [129280,4096] · norm · hc_head · head → lm_head', gen: DS, blk: (i) => ({ hc: 'mhc', norms: true, n1: 'attn_norm', n2: 'ffn_norm', mixer: i < 2 ? 'atenção SWA (janela 128)' : i % 2 === 0 ? 'atenção C4A + indexador' : 'atenção C128A', ffn: i < 3 ? 'MoE por hash (tid2eid)' : 'MoE: 256 experts + 1 shared', moe: true, attnIn: 'wq_a/wq_b/wkv', attnOut: 'wo_a/wo_b', ffnIn: 'w1/w3', ffnOut: 'w2' }), col: 'ds' },
      };
      const S = { m: 'glm', layer: 3, cur: 0, hov: -1 };
      const ctl = UI.controls(fig);
      let lsl;
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'cl', label: 'Clássico' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; const N = MOD[v].N; lsl.input.max = N - 1; if (S.layer > N - 1) lsl.set(N - 1); S.cur = 0; } });
      lsl = UI.slider(ctl, { label: 'Camada', min: 0, max: 44, step: 1, value: 3, onInput: (v) => { S.layer = v; S.cur = 0; } });
      V.every(1500, () => { S.cur++; });
      F(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const M = MOD[S.m];
        const li = Math.min(S.layer, M.N - 1);
        const L = M.gen(li);
        const narrow = w < 620;
        const mc = C[M.col] || C.a;
        let cur = S.cur % L.length;
        const pre = M.pre.replace('{i}', li);
        T(g, pre, 14, 20, { size: fit(g, pre, w - 28, 12.5, 'mono', 600), font: 'mono', weight: 600, color: mc });
        wrap(g, 'globais: ' + M.gl, 14, 38, w - 28, 14, { size: 11, font: 'mono', color: C.muted, maxLines: 2 });
        // área do bloco e da lista
        let bR, lR;
        if (!narrow) { bR = { x: 14, y: 70, w: 230, h: h - 90 }; lR = { x: 262, y: 64, w: w - 276, h: h - 72 }; }
        else { bR = { x: 14, y: 76, w: w - 28, h: 230 }; lR = { x: 14, y: 328, w: w - 28, h: h - 336 }; }
        const lineH = narrow ? 34 : 25;
        const visN = Math.max(3, Math.floor(lR.h / lineH));
        // hover na lista
        let hov = -1;
        const off0 = V.clamp(cur - Math.floor(visN / 2), 0, Math.max(0, L.length - visN));
        if (Hh.pointer.inside && Hh.pointer.x >= lR.x && Hh.pointer.y >= lR.y) { const k = Math.floor((Hh.pointer.y - lR.y) / lineH) + off0; if (k >= 0 && k < L.length && k < off0 + visN) hov = k; }
        if (hov >= 0) cur = hov;
        const off = hov >= 0 ? off0 : V.clamp(cur - Math.floor(visN / 2), 0, Math.max(0, L.length - visN));
        const lit = {}; lit[L[cur].p] = true;
        const bo = M.blk(li); bo.lit = lit;
        drawBlock(g, bR.x, bR.y, bR.w, bR.h, bo);
        // lista
        for (let k = off; k < Math.min(L.length, off + visN); k++) {
          const e = L[k];
          const y = lR.y + (k - off) * lineH;
          const on = k === cur;
          if (on) V.box(g, lR.x - 4, y, lR.w + 4, lineH - 3, { r: 6, fill: V.alpha(mc, 0.16), stroke: V.alpha(mc, 0.7), lw: 1.2 });
          const pc = { hc1: C.f, hc2: C.f, norm1: C.e, norm2: C.e, 'attn-in': C.b, 'attn-out': C.c, router: C.d, 'ffn-in': C.a, 'ffn-out': C.c, experts: C.a, ple: C.d }[e.p] || C.muted;
          g.fillStyle = pc; V.rr(g, lR.x, y + 7, 4, lineH - 16, 2); g.fill();
          if (!narrow) {
            const shW = Math.min(250, lR.w * 0.38);
            const nameW = lR.w - shW - 14;
            T(g, e.n, lR.x + 10, y + 15, { size: fit(g, e.n, nameW, 12, 'mono'), font: 'mono', color: on ? C.ink : C.text2 || C.ink, maxW: nameW });
            T(g, e.s + (e.t ? '  ' + e.t : ''), lR.x + lR.w - 2, y + 15, { size: fit(g, e.s + (e.t ? '  ' + e.t : ''), shW, 11.5, 'mono'), font: 'mono', align: 'right', color: on ? mc : C.muted, maxW: shW });
          } else {
            T(g, e.n, lR.x + 10, y + 13, { size: fit(g, e.n, lR.w - 14, 11.5, 'mono'), font: 'mono', color: on ? C.ink : C.muted, maxW: lR.w - 14 });
            T(g, e.s + (e.t ? '  ' + e.t : ''), lR.x + 10, y + 27, { size: 11, font: 'mono', color: on ? mc : C.muted, maxW: lR.w - 14 });
          }
        }
        if (off > 0) T(g, '▲ ' + off + ' acima', lR.x + lR.w, lR.y - 4, { size: 11, align: 'right', color: C.muted });
        if (off + visN < L.length) T(g, '▼ ' + (L.length - off - visN) + ' abaixo', lR.x + lR.w, h - 4, { size: 11, align: 'right', color: C.muted });
      });
    });

    /* ====================================================================
       Figura 10 — Mapa de obras: cada patch numa peça do bloco
       ==================================================================== */
    safe('obras', () => {
      const fig = $('#fig-obras');
      fitH(fig, (w) => (w < 620 ? 700 : 460));
      const PINS = {
        glm: {
          blk: { hc: 'mhc', norms: true, mixer: 'KDA ×34 / MLA NoPE ×11', ffn: 'MoE EXL3 (288) · densa nas 0–2', moe: true, attnIn: 'leitores', attnOut: 'o_proj' },
          pins: [
            { p: 'hc2', t: 'Hooks EAGLE3 no mHC', d: 'patch_glm_eagle3.py: nas saídas das camadas 5, 14, 24, 33 e 42, materializa hc_post e faz hc_contract (média dos 4 fluxos) para entregar ao DFlash2 um vetor de 4096.', ch: 'Rascunhos em bloco' },
            { p: 'attn-in', t: 'Overlay NoPE sparse-MLA (SM121)', d: 'O backend fp8_ds_mla exigia pe_dim = 64, mas o GLM é NoPE (qk_rope_head_dim = 0). Patches inline no Dockerfile preenchem 64 zeros e instanciam o kernel esparso para o GB10.', ch: 'MLA / KV quantizado' },
            { p: 'mixer', t: 'Estado das 34 camadas KDA', d: 'O estado recorrente fixo do KDA vive no cache “Mamba” do vLLM; os patches patch_mamba_align_* cuidam de alinhamento de chunks e liberação de estado.', ch: 'Atenção linear' },
            { p: 'attn-out', t: 'ABLIT=1 no o_proj', d: 'ablit_runtime.py edita, na carga, o o_proj das camadas 15–45 (transplante do doador ou projeção da direção de recusa): o escritor da atenção.', ch: 'Abliteração' },
            { p: 'ffn-in', t: 'GLM53_DENSE_FP8 (sem tocar nas normas)', d: 'Quantiza na carga, de BF16 para FP8 por canal, as projeções densas dos grupos shared, dense, kda e mla, e as executa com Marlin W8A16. As RMSNorms (ε = 1e-5) ficam em BF16.', ch: 'FP8 / Kernels' },
            { p: 'experts', t: 'Experts EXL3 4 bpw', d: 'overlay/exl3.py registra o método exl3: trellis + suh/svh + mcg por matriz, kernels E2/E3 e MoE cooperativo de decode (H = 4096).', ch: 'Hadamard, treliças e EXL3' },
            { p: 'ffn-in', t: 'SWIGLU_LIMIT_DEFAULT = 10.0', d: 'O clamp do SwiGLU (L = 10) é replicado dentro dos kernels EXL3: silu(min(g,10))·clamp(u,±10) nos caminhos PyTorch e no E3 (no epílogo do GEMM gate/up); min(silu(g),10)·clamp(u,±10) nos kernels herdados do exllamav3 e no cooperativo TP3, que só aceita limite 10.', ch: 'este capítulo' },
            { p: 'ffn-out', t: 'E3: down + scatter', d: 'EXL3_FAT_GROUPED=1: todos os experts “gordos” da camada (muitos tokens por expert, típico do prefill) em três lançamentos (gather; gate/up + SwiGLU; down + scatter), sem sincronizar com a CPU e capturável em CUDA graph. Os finos seguem no exl3_moe.', ch: 'Kernels' },
          ],
        },
        qwen: {
          blk: { hc: 'gr', norms: false, mixer: 'GDN ×36 / QSA ×12', ffn: 'MoE NVFP4 (512) + shared', moe: true, attnIn: 'leitores', attnOut: 'o_proj/out_proj' },
          pins: [
            { p: 'hc1', t: 'Gated Residual replicado', d: 'Os módulos do GR são BF16 e replicados nas duas GPUs (disable_tp): cerca de 1,19 GiB lidos por passo, 13% do tráfego de um passo de decode.', ch: 'Hyper-connections' },
            { p: 'hc2', t: 'PLE: um escritor extra na camada 1', d: 'detect_ple_dtype.py + --hf-overrides + ple_layer_patched.py carregam a tabela N-gram FP8 (51,2 B parâmetros), que soma seu vetor aos 4 ramos na camada de índice 1.', ch: 'Memória N-gram' },
            { p: 'attn-in', t: 'KV FP8 nas camadas QSA', d: 'patch_qsa_fp8_kv.py ensina os kernels QSA a ler K/V em FP8 nas 13 camadas com KV (12 + MTP): 1,70× tokens por GiB.', ch: 'KV cache quantizado' },
            { p: 'mixer', t: 'Fallback MXFP8 do GDN', d: 'No checkpoint alternativo com atenção/GDN em MXFP8, as projeções in_proj_a/b [48, 2560] são estreitas demais para o kernel MXFP8 (N = 48 < 128): o loader cai para emulação BF16.', ch: 'FP8 e microescala' },
            { p: 'attn-out', t: 'ABLIT=1: o_proj das 12 QSA', d: 'O checkpoint abliterado muda só o o_proj das 12 camadas QSA; o resto do checkpoint é idêntico.', ch: 'Abliteração' },
            { p: 'experts', t: 'Experts NVFP4 + EP', d: '256 experts inteiros por GPU (expert parallel). Os experts do MTP são FP8 em blocos 128×128 (patch_modelopt_fp8_block_moe.py); 640/2 = 320 não caberia nesses blocos.', ch: 'Paralelismos' },
            { p: 'ffn-in', t: 'FP8_DENSE (experimental)', d: 'Projeções densas BF16 → FP8; roteador, shared_expert_gate, normas e gates ficam BF16 (EXCLUDE_MODULES).', ch: 'Checkpoints' },
            { p: 'head', t: 'Vocabulário de rascunho reduzido', d: 'O lm_head 248.320 × 2560 (1,18 GiB) é lido 3× por passo pelo MTP; patch_mtp_draft_vocab.py faz o argmax do rascunho em 47.149 ids.', ch: 'MTP, EAGLE e vocabulário' },
          ],
        },
        ds: {
          blk: { hc: 'mhc', norms: true, n1: 'attn_norm', n2: 'ffn_norm', mixer: 'SWA / C4A / C128A', ffn: 'MoE MXFP4 (256) + shared', moe: true, attnIn: 'wq_a/wq_b/wkv', attnOut: 'wo_a/wo_b', ffnIn: 'w1/w3', ffnOut: 'w2' },
          pins: [
            { p: 'hc1', t: 'mHC do b12x', d: 'VLLM_USE_B12X_MHC (opcional) faz o pre/post do mHC num kernel b12x que só aceita hidden_size = 4096; o overlay já funde attn_norm/ffn_norm no mHC.', ch: 'Hyper-connections' },
            { p: 'hc2', t: 'Captura das features do DSpark', d: 'Nas camadas 40, 41 e 42, hc_post colapsa os 4 fluxos e a média vai para o main_proj (12.288 → 4096) do rascunhador.', ch: 'DFlash2 e DSpark' },
            { p: 'attn-in', t: 'nvfp4_ds_mla → kernel FP8', d: 'Issue #22: o dtype de KV declarado é roteado para o kernel FP8; cada linha de KV comprimido tem 584 B.', ch: 'KV cache quantizado' },
            { p: 'attn-out', t: 'Ablação em runtime', d: 'Com ABLATE=1, logo depois da saída da atenção: x ← x − λ(x·v)v, com uma direção de 4096 dimensões validada (finita, norma > 0).', ch: 'Abliteração' },
            { p: 'attn-out', t: 'wo_b row-parallel', d: 'A saída agrupada (wo_a por grupo, wo_b row-parallel) custa 43 all-reduces por passo, mais 43 do MoE. TP = 3 exige padding de grupos 8 → 9.', ch: 'TP=3 e TP=4' },
            { p: 'router', t: 'Roteamento por hash (0–2)', d: 'tid2eid escolhe os experts pelo id do token: exige input_ids crus; os tokens de imagem precisaram de bias_vl (#175).', ch: 'Mixture-of-Experts' },
            { p: 'experts', t: 'b12x W4A16 nos experts', d: '--moe-backend flashinfer_b12x: experts MXFP4 (e2m1 + E8M0/32) dequantizados para BF16; recebe swiglu_limit = 10 como gemm1_clamp_limit.', ch: 'Kernels' },
          ],
        },
      };
      const S = { m: 'glm', lock: -1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; S.lock = -1; } });
      UI.button(ctl, 'automático', () => { S.lock = -1; });
      F(fig, (g, w, h, t, dt, _s, Hh) => {
        const C = V.C();
        const D = PINS[S.m];
        const mc = C[S.m] || C.a;
        const narrow = w < 620;
        let bR, pR;
        if (!narrow) { bR = { x: 14, y: 50, w: Math.min(310, w * 0.37), h: h - 92 }; pR = { x: bR.x + bR.w + 84, y: 16, w: w - bR.x - bR.w - 98, h: h - 26 }; }
        else { bR = { x: 14, y: 54, w: w - 100, h: 330 }; pR = { x: 14, y: 430, w: w - 28, h: h - 440 }; }
        const auto = Math.floor(t / 3.6) % D.pins.length;
        // embedding e lm_head em volta do bloco
        V.box(g, bR.x + 34, bR.y - 34, bR.w - 34, 24, { r: 6, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.5), lw: 1, label: 'embedding', size: 11.5 });
        const headR = { x: bR.x + 34, y: bR.y + bR.h + 14, w: bR.w - 34, h: 24 }; // fora do bloco
        // alvo atual (automático, fixado ou hover)
        let sel = S.lock >= 0 ? S.lock : auto;
        // primeiro desenho para obter retângulos (sem destaque), depois pinos
        const lit0 = {};
        const rects = (() => { const o = Object.assign({}, D.blk, { lit: lit0 }); g.save(); g.globalAlpha = 0; const r = drawBlock(g, bR.x, bR.y, bR.w, bR.h, o); g.restore(); return r; })();
        rects.head = headR;
        const pinPos = [];
        D.pins.forEach((p) => {
          const r = rects[p.p] || rects.mixer;
          const y = r.y + r.h / 2;
          let col = 0;
          while (col < 2 && pinPos.some(([px, py]) => Math.abs(py - y) < 22 && px === bR.x + bR.w + 14 + col * 24)) col++;
          pinPos.push([bR.x + bR.w + 14 + col * 24, y]);
        });
        let hov = -1;
        if (Hh.pointer.inside) pinPos.forEach(([x, y], k) => { if (Math.hypot(Hh.pointer.x - x, Hh.pointer.y - y) < 13) hov = k; });
        if (Hh.pointer.clicked && hov >= 0) S.lock = hov;
        if (hov >= 0) sel = hov;
        const lit = {}; lit[D.pins[sel].p] = true;
        drawBlock(g, bR.x, bR.y, bR.w, bR.h, Object.assign({}, D.blk, { lit }));
        V.box(g, headR.x, headR.y, headR.w, headR.h, { r: 6, fill: V.alpha(C.c, lit.head ? 0.34 : 0.12), stroke: V.alpha(C.c, lit.head ? 1 : 0.5), lw: lit.head ? 2.2 : 1, label: 'norma final → lm_head', size: 11.5 });
        T(g, '× ' + { glm: 45, qwen: 48, ds: 43 }[S.m] + ' blocos', bR.x + bR.w, bR.y - 40, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        pinPos.forEach(([x, y], k) => {
          const on = k === sel;
          const r = rects[D.pins[k].p] || rects.mixer;
          V.line(g, bR.x + bR.w + 1, y, x - 10, y, { color: V.alpha(mc, on ? 0.9 : 0.3), lw: 1, dash: [2, 3] });
          if (on) V.line(g, Math.min(r.x + r.w, bR.x + bR.w), y, bR.x + bR.w, y, { color: V.alpha(mc, 0.9), lw: 1.2 });
          g.beginPath(); g.arc(x, y, on ? 11 : 9, 0, 6.3); g.fillStyle = on ? mc : V.alpha(mc, 0.3); g.fill();
          if (on) { g.strokeStyle = C.ink; g.lineWidth = 1.5; g.stroke(); V.glow(g, x, y, 22 + 3 * Math.sin(t * 5), mc, 0.4); }
          T(g, String(k + 1), x, y + 4, { size: 11.5, align: 'center', weight: 700, color: on ? onTxt(C) : C.ink });
        });
        // painel de descrição
        const p = D.pins[sel];
        V.box(g, pR.x, pR.y, pR.w, pR.h, { r: 10, fill: V.alpha(C.panel, 0.85), stroke: V.alpha(mc, 0.6), lw: 1.3 });
        T(g, (sel + 1) + ' / ' + D.pins.length + (S.lock >= 0 ? ' (fixado)' : ''), pR.x + pR.w - 12, pR.y + 20, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        let yy = pR.y + 22;
        yy += 18 * wrap(g, p.t, pR.x + 14, yy, pR.w - 80, 18, { size: 14, weight: 700, color: mc, maxLines: 2 });
        yy += 6;
        yy += 17 * wrap(g, p.d, pR.x + 14, yy, pR.w - 28, 17, { size: 12, color: C.ink, maxLines: narrow ? 8 : 9 });
        const pn = { hc1: 'barramento (leitura)', hc2: 'barramento (escrita/mistura)', norm1: 'normas', 'attn-in': 'leitores da atenção', 'attn-out': 'escritor da atenção', mixer: 'misturador de tokens', router: 'roteador', 'ffn-in': 'leitores da FFN (gate/up)', 'ffn-out': 'escritor da FFN (down_proj)', experts: 'experts', head: 'fora do bloco: norma final → lm_head' }[p.p] || p.p;
        T(g, 'peça do bloco: ' + pn, pR.x + 14, pR.y + pR.h - 30, { size: 11.5, color: C.muted, maxW: pR.w - 28 });
        T(g, 'aprofundamento: ' + p.ch, pR.x + 14, pR.y + pR.h - 12, { size: 11.5, color: C.muted, maxW: pR.w - 28 });
        // lista compacta (só em telas largas)
        if (yy + 20 < pR.y + pR.h - 50) {
          const ly0 = yy + 16;
          const fitN = Math.floor((pR.y + pR.h - 56 - ly0) / 16) + 1;
          const nShow = fitN >= D.pins.length ? D.pins.length : Math.max(1, fitN - 1);
          D.pins.slice(0, nShow).forEach((q, k) => { T(g, (k + 1) + '. ' + q.t, pR.x + 14, ly0 + k * 16, { size: 11.5, color: k === sel ? mc : C.muted, maxW: pR.w - 28 }); });
          if (nShow < D.pins.length) T(g, '… +' + (D.pins.length - nShow) + ' (toque nos marcadores numerados)', pR.x + 14, ly0 + nShow * 16, { size: 11.5, color: C.muted, maxW: pR.w - 28 });
        }
      });
    });
  },
});

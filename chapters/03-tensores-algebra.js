LA.chapter({
  id: 'tensores-algebra',
  part: 1,
  order: 3,
  title: 'Tensores e a álgebra linear de um LLM',
  navTitle: 'Tensores & álgebra',
  subtitle: 'Vetores, matrizes e tensores do jeito que a máquina os vê: produto escalar, formas, GEMM, strides, baixo posto, rotações e o custo de cada multiplicação — com as formas reais do GLM, do Qwen e do DeepSeek.',
  minutes: 55,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['vetor', 'produto escalar', 'matriz', 'GEMM', 'GEMV', 'shape', 'broadcasting', 'einsum', 'stride', 'baixo posto', 'SVD', 'ortogonal', 'FLOPs', 'intensidade aritmética'],
  goals: [
    'Ler <b>produto escalar</b>, cosseno e normas como geometria e reconhecê-los em logits, atenção e roteadores',
    'Ler qualquer forma de peso (<code>[saída, entrada]</code>), multiplicar matrizes de cabeça e contar FLOPs com \\(2MNK\\)',
    'Manipular formas (cabeças, reshape, transposição, broadcasting, einsum) e saber o que <b>strides e views</b> fazem na memória',
    'Entender <b>baixo posto</b>, matrizes ortogonais e projeções, e onde cada uma aparece nas três receitas',
    'Prever se uma GEMM é limitada por memória ou por cálculo a partir de \\(M\\), e por que as receitas escolhem kernels por \\(M\\)',
  ],
  prereqs: ['redes-neurais'],
  summary: [
    'Um token é um vetor de <code>hidden_size</code> números (4096 no GLM e no DeepSeek, 2560 no Qwen). O <b>produto escalar</b> \\(\\sum a_ib_i = \\lVert a\\rVert\\lVert b\\rVert\\cos\\theta\\) está nos logits, na atenção, nos roteadores e em toda camada linear. A RMSNorm divide pelo RMS \\(=\\lVert x\\rVert/\\sqrt d\\). Em 4096 dimensões, direções aleatórias são quase ortogonais (cos ~ \\(1/\\sqrt d\\)).',
    'Uma matriz é uma função linear: linhas são detectores, colunas são as imagens dos eixos, \\(|\\det|\\) é o fator de área e o <b>posto</b> é a dimensão da saída. O peso de um <code>nn.Linear</code> é <code>[saída, entrada]</code>; kernels (e o EXL3) escrevem <code>A[M,K] @ W[K,N]</code>. Embeddings são tabelas <code>[linhas indexadas, largura]</code>, e formatos quantizados guardam formas empacotadas (NVFP4 <code>U8 [640, 1280]</code> = 640 × 2560).',
    'GEMM \\([M\\times K][K\\times N]\\) custa \\(2MNK\\) FLOPs, e cada peso serve \\(M\\) tokens. \\(M\\) é ~32–42 linhas no decode das receitas (sequências × (1 + rascunhos)) e 7168–8192 no prefill; num MoE, o que conta é o \\(M\\) de cada expert. Cortar \\(W\\) em blocos é a álgebra do tensor parallel: pela saída sem comunicação, pela entrada com all-reduce.',
    'Formas: uma camada linear age no último eixo; cabeças são reshape + transposição (só metadados); projeções fundidas concatenam saídas (o <code>in_proj_qkvbfg_a [12576×4096]</code> do KDA). Broadcasting alinha pela direita e estica eixos de tamanho 1 (stride 0). O einsum <code>bsgd,grd->bsgr</code> revela os 8 grupos do DeepSeek.',
    'Um tensor é ponteiro + forma + <b>strides</b> + offset. Transpose, fatia e expand são views. Kernels que supõem layout denso corrompem views <code>as_strided</code>: a cauda do kpool do GLM tem <code>stride(0)=59136</code>, não 1024, e o bug mandava as 552 escritas para as páginas 0–9.',
    '<b>Baixo posto</b> \\(W\\approx AB\\) troca \\(mn\\) por \\(r(m+n)\\); a SVD mostra quantas direções importam. As receitas usam gargalos treinados: <code>q_lora_rank</code> 1536 (GLM) e 1024 (DS), <code>kv_lora_rank</code> 512, <code>o_lora_rank</code> 1024, gates do KDA 128, <code>hc_lowrank</code> 320 e a Markov head de posto 256 (66 M vs 16,7 B parâmetros).',
    'Matrizes ortogonais preservam produtos escalares (RoPE, Hadamard). \\(I-\\alpha\\hat r\\hat r^\\top\\) remove (α=1), espelha (α=2) ou inverte e dobra (α=3, o <code>ABLIT_ALPHA</code> do método proj da abliteração do GLM). A intensidade \\(\\approx 2M/b_w\\) contra o ridge de ~458 FLOP/B do GB10 explica o decode limitado por memória e o despacho por \\(M\\) (<code>M &gt; 512</code> → cuBLAS BF16 no KDA do GLM).',
  ],
  quiz: [
    { q: 'No Qwen, um passo de decode tem ativações \\(X\\) de forma <code>[32, 2560]</code>, e o peso <code>gate_proj</code> do shared expert está no checkpoint como <code>[640, 2560]</code>. Qual a forma da saída e quantos FLOPs custa a multiplicação?', o: ['<code>[640, 32]</code> e ~52 MFLOP', '<code>[32, 640]</code> e ~105 MFLOP', '<code>[32, 2560]</code> e ~105 MFLOP', '<code>[2560, 640]</code> e ~4 GFLOP'], a: 1, e: 'O peso é <code>[saída, entrada]</code>: 2560 → 640. \\(Y = XW^\\top\\) tem forma \\([32, 640]\\), uma linha por token. FLOPs \\(= 2MNK = 2\\times32\\times640\\times2560 \\approx 104{,}9\\) milhões. A alternativa com \\([640, 32]\\) troca linhas por colunas e esquece o fator 2.' },
    { q: 'Por que, no GB10, uma GEMM de decode com \\(M = 32\\) linhas e pesos BF16 leva quase o mesmo tempo que com \\(M = 1\\)?', o: ['Porque a GPU só processa uma linha por vez', 'Porque a intensidade aritmética ≈ \\(M\\) FLOP/B fica muito abaixo do ridge (~458 FLOP/B): o tempo é dominado por ler os pesos, que são os mesmos para qualquer \\(M\\)', 'Porque o KV cache domina o tempo', 'Porque o CUDA graph fixa o tempo do passo'], a: 1, e: 'Com \\(M \\ll K, N\\), os bytes são quase só de pesos (\\(KN\\,b_w\\)), e as contas crescem com \\(M\\). Em BF16, \\(I \\approx 2M/2 = M\\). Com \\(M=32\\), \\(I \\approx 32\\), bem abaixo de 458: os núcleos esperam a memória, e 32 linhas custam praticamente o mesmo tempo de leitura que uma. É por isso que batching e especulação são quase de graça no decode.' },
    { q: 'A Markov head do DSpark usa \\(W_1\\) (vocab → 256) e \\(W_2\\) (256 → vocab), com vocabulário de 129.280. O que se ganha em relação a uma tabela de bigramas cheia \\(V\\times V\\)?', o: ['Nada: as duas têm o mesmo número de parâmetros', 'Cerca de 66 milhões de parâmetros em vez de ~16,7 bilhões (~250× menos), ao preço de a tabela implícita ter posto no máximo 256', 'Precisão numérica maior', 'Elimina a necessidade de conhecer o token anterior'], a: 1, e: '\\(2\\times129.280\\times256 \\approx 66{,}2\\) M parâmetros, contra \\(129.280^2 \\approx 16{,}7\\) B (~33 GB em BF16). A tabela \\(W_2W_1\\) continua sendo \\(V\\times V\\), mas com posto ≤ 256: só 256 “maneiras de continuar” independentes. O token anterior continua indispensável: é a linha que \\(W_1\\) consulta.' },
    { q: 'A cauda do kpool do GLM é uma view BF16 <code>[num_blocks, 2, 4, 128]</code> com <code>stride(0) = 59136</code>. O kernel antigo endereçava o bloco \\(b\\) em \\(b \\times 1024\\) elementos. O que acontecia?', o: ['Um erro de índice fora do limite, e o servidor caía', 'Nada: o resultado era o mesmo, só mais lento', 'As escritas iam para páginas de outros blocos (todas nas páginas 0–9, com 552 blocos): corrupção silenciosa do indexador de outras requisições', 'A cauda ficava em FP8 em vez de BF16'], a: 2, e: 'Com o stride denso, a cauda do bloco \\(b\\) caía no byte \\(2048\\,b\\), dentro da página \\(\\lfloor 2048b/118272\\rfloor\\). Os endereços continuam válidos (por isso nada quebra), mas são de outra página: o bloco 58 escrevia na cauda do bloco 1. A correção lê <code>tail.stride(0)</code> do tensor.' },
    { q: 'Com \\(\\hat r\\) unitário, qual destas matrizes preserva o comprimento de <i>todo</i> vetor?', o: ['\\(I - \\hat r\\hat r^\\top\\)', '\\(I - 2\\hat r\\hat r^\\top\\)', '\\(I - 3\\hat r\\hat r^\\top\\)', '\\(AB\\) com \\(A\\) de 4096×256 e \\(B\\) de 256×4096'], a: 1, e: '\\(I - 2\\hat r\\hat r^\\top\\) é a reflexão de Householder: ortogonal, \\(\\det=-1\\), só troca o sinal da componente em \\(\\hat r\\). \\(\\alpha=1\\) zera essa componente (achata), \\(\\alpha=3\\) a multiplica por \\(-2\\) (estica), e um produto de posto 256 achata tudo fora de um subespaço de 256 dimensões.' },
    { q: 'Você quer multiplicar um tensor <code>[T, 64, 128]</code> (64 cabeças × 128 canais) por uma taxa por cabeça <code>A</code> de forma <code>[64]</code>. O que acontece?', o: ['Funciona: o 64 casa com as 64 cabeças', 'Erro: alinhando pela direita, 64 encontra 128. A correção é dar a <code>A</code> a forma <code>[64, 1]</code>', 'Funciona, mas copia <code>A</code> 128 vezes na memória', 'Erro, e a única saída é um laço em Python'], a: 1, e: 'Broadcasting alinha pelo <b>último</b> eixo: <code>[T, 64, 128]</code> contra <code>[64]</code> compara 128 com 64 e falha. Com <code>A[:, None]</code> (forma <code>[64, 1]</code>), 64 casa com 64 e o 1 é esticado sobre os 128 canais, sem cópia (stride 0). O perigo real é quando os tamanhos coincidem por acaso e a conta errada passa calada.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const clamp = V.clamp, lerp = V.lerp;
    /* ------------------------------------------------------------ helpers */
    const nf = (x, d = 0) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const sg = (x, d = 2) => { const s = nf(Math.abs(x), d); return (x < 0 && s !== nf(0, d) ? '−' : '') + s; };
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f, w) => { const k = s + '|' + f + '|' + w; return fcache[k] || (fcache[k] = V.font(s, f, w)); };
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 14, ht = size + 9;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.85), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const size = o.size || 12, f = o.font || 'sans', wt = o.weight || 400;
      g.font = font(size, f, wt);
      const words = String(s).split(' ');
      let line = '', yy = y;
      const put = (l) => V.text(g, l, x, yy, { size, font: f, weight: wt, color: o.color, align: o.align });
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        g.font = font(size, f, wt);
        if (g.measureText(test).width > maxW && line) { put(line); line = wd; yy += lh; } else line = test;
      }
      if (line) { put(line); yy += lh; }
      return yy;
    }
    // altura do palco por largura (desk/mob podem ser funções do estado); devolve apply() para reaplicar
    function stageH(fig, desk, mob, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const val = (v) => (typeof v === 'function' ? v() : v);
      const apply = () => { const want = val(fig.clientWidth < bp ? mob : desk) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    // Figura "viva": estática (só desenha quando pedido), mas se auto-agenda enquanto draw() devolver true,
    // e só com a figura na tela, a aba visível e sem a pausa global. O relógio (t) só anda nesses quadros.
    function liveFig(fig, draw, o = {}) {
      let H = null, on = false, want = true, looping = false, clock = 0, lastT = 0;
      const io = new IntersectionObserver((es) => {
        const was = on; on = es[es.length - 1].isIntersecting;
        if (on && !was) { if (o.onEnter) o.onEnter(); if (H) H.redraw(); }
      }, { rootMargin: '40px' });
      io.observe(fig); LA.onCleanup(() => io.disconnect());
      H = V.fig(fig, (g, w, h, t, dt0) => {
        // H.redraw() zera o relógio interno do loop; medimos o intervalo real entre quadros encadeados
        const now = performance.now(), dt = looping ? Math.min(0.05, (now - lastT) / 1000) : Math.min(0.05, dt0);
        lastT = now; looping = false;
        if (!LA.paused) clock += dt;
        want = !!draw(g, w, h, clock, dt);
        if (want && on && !document.hidden && !LA.paused) { looping = true; H.redraw(); }
      }, { static: true });
      V.every(300, () => { if (want && on && !looping && !document.hidden) H.redraw(); });
      H.clock = () => clock;
      return H;
    }
    const fitSelect = (sel) => { sel.el.style.maxWidth = '100%'; sel.el.style.minWidth = '0'; const se = sel.el.querySelector('select'); if (se) { se.style.maxWidth = 'min(100%, calc(100vw - 150px))'; se.style.minWidth = '0'; } };
    const setVal = (sl, v, txt) => { sl.input.value = v; const e = sl.el.querySelector('.val'); if (e) e.textContent = txt; };
    const show = (el, on) => { el.style.display = on ? '' : 'none'; };
    const fmtCount = (x) => (x >= 1e9 ? nf(x / 1e9, 2) + ' B' : x >= 1e6 ? nf(x / 1e6, x >= 1e8 ? 0 : 2) + ' M' : x >= 1e4 ? nf(x / 1e3, 1) + ' mil' : nf(x));
    const fmtBytes = (b) => (b >= 1e9 ? nf(b / 1e9, 2) + ' GB' : b >= 1e6 ? nf(b / 1e6, 1) + ' MB' : b >= 1e3 ? nf(b / 1e3, 1) + ' KB' : nf(b) + ' B');
    const fmtT = (s) => (s >= 1e-3 ? nf(s * 1e3, s >= 1e-2 ? 1 : 2) + ' ms' : nf(s * 1e6, s >= 1e-5 ? 1 : 2) + ' µs');
    const fmtFl = (x) => (x >= 1e12 ? nf(x / 1e12, 2) + ' TFLOP' : x >= 1e9 ? nf(x / 1e9, 1) + ' GFLOP' : x >= 1e6 ? nf(x / 1e6, 1) + ' MFLOP' : nf(x / 1e3, 1) + ' kFLOP');
    function clipRect(g, r) { g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip(); }
    function panelBox(g, r, C, title) {
      V.box(g, r.x, r.y, r.w, r.h, { r: 12, fill: V.alpha(C.panel || C.muted, C.dark ? 0.35 : 0.5), stroke: V.alpha(C.muted, 0.25), lw: 1 });
      if (title) V.text(g, title, r.x + 12, r.y + 18, { size: 11, font: 'mono', color: C.muted, weight: 600 });
    }

    /* ================================================== Fig 1: bússola */
    (function figBussola() {
      const fig = root.querySelector('#fig-bussola');
      stageH(fig, 400, 680);
      const S = { th: 55, L: 1.3, norm: false, d: 4096, side: 1, disp: null };
      const ctl = UI.controls(fig);
      const sTh = UI.slider(ctl, { label: 'Ângulo θ', min: 0, max: 180, step: 1, value: S.th, fmt: (v) => v + '°', onInput: (v) => { S.th = v; H.redraw(); } });
      const sL = UI.slider(ctl, { label: '‖k‖', min: 0.2, max: 2, step: 0.05, value: S.L, fmt: (v) => nf(v, 2), onInput: (v) => { S.L = v; H.redraw(); } });
      UI.toggle(ctl, { label: 'normalizar (cosseno)', value: false, onChange: (v) => { S.norm = v; H.redraw(); } });
      UI.seg(ctl, { label: 'dimensões d', options: [2, 16, 128, 4096], value: 4096, onChange: (v) => { S.d = v; H.redraw(); } });
      const cache = {};
      const NB = 40;
      function samples(d) {
        if (cache[d]) return cache[d];
        const r = V.rng(1000 + d), n = 600, out = new Float64Array(n);
        for (let s = 0; s < n; s++) { let ss = 0, x1 = 0; for (let i = 0; i < d; i++) { const z = r.normal(); if (i === 0) x1 = z; ss += z * z; } out[s] = x1 / Math.sqrt(ss); }
        let m = 0; for (const c of out) m += c; m /= n;
        let v = 0, big = 0; for (const c of out) { v += (c - m) * (c - m); if (Math.abs(c) > 0.1) big++; }
        const hist = new Array(NB).fill(0);
        for (const c of out) hist[Math.min(NB - 1, Math.max(0, Math.floor(((c + 1) / 2) * NB)))]++;
        return (cache[d] = { hist: hist.map((x) => x / n), std: Math.sqrt(v / n), big: big / n });
      }
      const QA = (18 * Math.PI) / 180, QL = 1.2;
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 640;
        const Lr = narrow ? { x: 0, y: 0, w, h: h * 0.54 } : { x: 0, y: 0, w: w * 0.52, h };
        const Rr = narrow ? { x: 0, y: h * 0.54, w, h: h * 0.46 } : { x: w * 0.52, y: 0, w: w * 0.48, h };
        const u = Math.min(Lr.w * 0.34, Lr.h * 0.3);
        const cx = Lr.x + Lr.w * (narrow ? 0.5 : 0.46), cy = Lr.y + Lr.h * 0.62;
        const P = H.pointer;
        if (P.down && P.y >= Lr.y + 40 && P.y <= Lr.y + Lr.h && P.x >= Lr.x && P.x <= Lr.x + Lr.w) {
          const dx = (P.x - cx) / u, dy = -(P.y - cy) / u;
          let a = Math.atan2(dy, dx) - QA;
          while (a > Math.PI) a -= 2 * Math.PI;
          while (a < -Math.PI) a += 2 * Math.PI;
          S.side = a < 0 ? -1 : 1; S.th = Math.round((Math.abs(a) * 180) / Math.PI);
          S.L = clamp(Math.hypot(dx, dy), 0.2, 2);
          setVal(sTh, S.th, S.th + '°'); setVal(sL, S.L, nf(S.L, 2));
        }
        const qL = S.norm ? 1 : QL, kL = S.norm ? 1 : S.L;
        const ka = QA + S.side * (S.th * Math.PI) / 180;
        const q = [Math.cos(QA) * qL, Math.sin(QA) * qL], k = [Math.cos(ka) * kL, Math.sin(ka) * kL];
        const dot = q[0] * k[0] + q[1] * k[1], cs = Math.cos((S.th * Math.PI) / 180);
        const toS = (x, y) => [cx + x * u, cy - y * u];
        clipRect(g, Lr);
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let i = -8; i <= 8; i++) { const x = cx + i * u * 0.5, y = cy + i * u * 0.5; g.moveTo(x, Lr.y); g.lineTo(x, Lr.y + Lr.h); g.moveTo(Lr.x, y); g.lineTo(Lr.x + Lr.w, y); }
        g.stroke();
        V.line(g, Lr.x, cy, Lr.x + Lr.w, cy, { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.line(g, cx, Lr.y, cx, Lr.y + Lr.h, { color: V.alpha(C.muted, 0.5), lw: 1 });
        g.setLineDash([3, 5]); g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, u, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        V.text(g, '‖x‖ = 1', cx + u * 0.72, cy + u * 0.82, { size: 10.5, font: 'mono', color: C.muted });
        // linha de q (suporte da sombra)
        const qh = [Math.cos(QA), Math.sin(QA)];
        const e1 = toS(-qh[0] * 3, -qh[1] * 3), e2 = toS(qh[0] * 3, qh[1] * 3);
        V.line(g, e1[0], e1[1], e2[0], e2[1], { color: V.alpha(C.a, 0.18), lw: 1, dash: [2, 4] });
        // sombra (projeção de k em q)
        const sh = dot / qL; const foot = [qh[0] * sh, qh[1] * sh];
        const O = toS(0, 0), F = toS(foot[0], foot[1]), Kp = toS(k[0], k[1]), Qp = toS(q[0], q[1]);
        V.line(g, Kp[0], Kp[1], F[0], F[1], { color: V.alpha(C.muted, 0.9), lw: 1.2, dash: [4, 4] });
        g.lineCap = 'round'; V.line(g, O[0], O[1], F[0], F[1], { color: V.alpha(C.d, 0.9), lw: 7 }); g.lineCap = 'butt';
        // arco do ângulo
        g.strokeStyle = V.alpha(C.c, 0.9); g.lineWidth = 1.6; g.beginPath(); g.arc(cx, cy, u * 0.32, -QA, -ka, S.side > 0); g.stroke();
        const ma = QA + (S.side * (S.th * Math.PI)) / 360;
        V.text(g, 'θ = ' + S.th + '°', cx + Math.cos(ma) * u * 0.46, cy - Math.sin(ma) * u * 0.46, { size: 12, font: 'mono', weight: 600, color: C.c, align: 'center', base: 'middle' });
        V.arrow(g, O[0], O[1], Qp[0], Qp[1], { color: C.a, lw: 3, head: 11 });
        V.arrow(g, O[0], O[1], Kp[0], Kp[1], { color: C.b, lw: 3, head: 11 });
        V.glow(g, Kp[0], Kp[1], 14, C.b, 0.6);
        const pillIn = (x, y, txt, col, right) => {
          const wd = tw(g, txt, 10.5, 'mono', 600) + 14;
          let xx = right ? x - wd : x;
          xx = clamp(xx, Lr.x + 4, Lr.x + Lr.w - wd - 4);
          let yy = clamp(y, Lr.y + 14, Lr.y + Lr.h - 14);
          if (xx < Lr.x + 280 && yy < Lr.y + 108) yy = Lr.y + 112;
          pill(g, xx, yy, txt, col, { size: 10.5 });
        };
        pillIn(Qp[0] + 8, Qp[1] - 14, 'q · consulta', C.a, false);
        pillIn(Kp[0] + (Kp[0] < cx ? -8 : 8), Kp[1] - 14, 'k · chave (arraste)', C.b, Kp[0] < cx);
        g.restore();
        // leitura do escore
        const bx = Lr.x + 14, by = Lr.y + 12;
        V.text(g, 'PLANO 2D', bx, by + 10, { size: 11, font: 'mono', weight: 600, color: C.muted });
        const f1 = S.norm ? 'cos θ = q·k / (‖q‖‖k‖)' : 'q·k = ‖q‖ · ‖k‖ · cos θ';
        const f2 = S.norm ? '= ' + sg(cs, 3) : '= ' + nf(qL, 2) + ' × ' + nf(kL, 2) + ' × ' + sg(cs, 3) + ' = ' + sg(dot, 2);
        V.text(g, f1, bx, by + 30, { size: 12.5, font: 'mono', color: C.ink });
        V.text(g, f2, bx, by + 48, { size: 14, font: 'mono', weight: 700, color: dot >= 0 ? C.a : C.c });
        const verdict = Math.abs(cs) < 0.05 ? 'ortogonais: escore ≈ 0' : cs > 0.5 ? 'apontam juntos: escore alto' : cs < -0.5 ? 'opostos: escore negativo' : cs > 0 ? 'meio alinhados' : 'meio opostos';
        V.text(g, verdict, bx, by + 66, { size: 12, color: C.muted });
        // termômetro do escore
        const tx = bx, tyy = by + 80, twd = Math.min(200, Lr.w * 0.45), mxs = S.norm ? 1 : QL * 2;
        V.box(g, tx, tyy, twd, 8, { r: 4, fill: V.alpha(C.muted, 0.18) });
        const mid = tx + twd / 2, pos = mid + (dot / mxs) * (twd / 2);
        V.box(g, Math.min(mid, pos), tyy, Math.abs(pos - mid), 8, { r: 4, fill: dot >= 0 ? C.a : C.c });
        V.line(g, mid, tyy - 3, mid, tyy + 11, { color: C.ink, lw: 1 });
        // ---- histograma
        const smp = samples(S.d);
        if (!S.disp) S.disp = smp.hist.slice();
        let moving = false;
        for (let i = 0; i < NB; i++) { const dd = smp.hist[i] - S.disp[i]; S.disp[i] += dd * Math.min(1, dt * 8 + 0.02); if (Math.abs(dd) > 2e-4) moving = true; }
        const px = Rr.x + (narrow ? 34 : 40), pw = Rr.w - (narrow ? 50 : 62), py = Rr.y + (narrow ? 52 : 70), ph = Rr.h - (narrow ? 124 : 140);
        V.text(g, 'COSSENO ENTRE DIREÇÕES ALEATÓRIAS · d = ' + nf(S.d), Rr.x + (narrow ? 14 : 20), Rr.y + (narrow ? 22 : 26), { size: 11, font: 'mono', weight: 600, color: C.muted, maxW: Rr.w - 28 });
        const mx = Math.max(0.05, ...S.disp) * 1.08;
        const sd = 1 / Math.sqrt(S.d);
        const X = (c) => px + ((c + 1) / 2) * pw;
        g.fillStyle = V.alpha(C.d, 0.12); g.fillRect(X(-sd), py, Math.max(2, X(sd) - X(-sd)), ph);
        const bw = pw / NB;
        for (let i = 0; i < NB; i++) {
          const v = S.disp[i], bh = (v / mx) * ph;
          g.fillStyle = V.alpha(C.b, 0.85); V.rr(g, px + i * bw + 1, py + ph - bh, bw - 2, Math.max(0.5, bh), 2); g.fill();
        }
        V.line(g, px, py + ph, px + pw, py + ph, { color: C.muted, lw: 1 });
        [-1, -0.5, 0, 0.5, 1].forEach((c) => { V.line(g, X(c), py + ph, X(c), py + ph + 4, { color: C.muted, lw: 1 }); V.text(g, sg(c, c % 1 ? 1 : 0), X(c), py + ph + 17, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        const xc = X(cs);
        V.line(g, xc, py - 6, xc, py + ph, { color: C.c, lw: 2, dash: [4, 3] });
        pill(g, xc, py - 12, 'seu par q,k', C.c, { size: 10, align: xc > px + pw * 0.8 ? 'right' : xc < px + pw * 0.2 ? 'left' : 'center' });
        const ry = py + ph + (narrow ? 34 : 40);
        V.text(g, 'desvio medido ' + nf(smp.std, 4) + '  ·  1/√d = ' + nf(sd, 4), px, ry, { size: 12, font: 'mono', color: C.ink, maxW: pw });
        V.text(g, '|cos| > 0,1 em ' + nf(smp.big * 100, 1) + '% dos pares' + (S.d >= 1024 ? ': quase tudo é perpendicular' : ''), px, ry + 18, { size: 12, color: C.muted, maxW: pw });
        if (moving) H.redraw();
      }, { static: true });
    })();

    /* ================================================== Fig 2: duas leituras da matriz */
    (function figGrade() {
      const fig = root.querySelector('#fig-grade');
      stageH(fig, 420, 650);
      const c30 = Math.cos(Math.PI / 6), s30 = Math.sin(Math.PI / 6);
      const PRE = { id: [1, 0, 0, 1], esc: [1.5, 0, 0, 0.6], rot: [c30, -s30, s30, c30], cis: [1, 0.8, 0, 1], ref: [-1, 0, 0, 1], p1: [1, 0.5, 0.6, 0.3] };
      const S = { cur: [1, 0, 0, 1], tgt: PRE.cis.slice(), view: 'cols' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Leitura', options: [{ v: 'cols', label: 'por colunas' }, { v: 'rows', label: 'por linhas (réguas)' }], value: 'cols', onChange: (v) => { S.view = v; H.redraw(); } });
      UI.seg(ctl, { options: [{ v: 'id', label: 'identidade' }, { v: 'esc', label: 'escala' }, { v: 'rot', label: 'rotação 30°' }, { v: 'cis', label: 'cisalhamento' }, { v: 'ref', label: 'reflexão' }, { v: 'p1', label: 'posto 1' }], value: 'cis', onChange: (v) => { S.tgt = PRE[v].slice(); sync(); H.redraw(); } });
      const names = ['W₁₁', 'W₁₂', 'W₂₁', 'W₂₂'];
      const sl = names.map((nm, k) => UI.slider(ctl, { label: nm, min: -2, max: 2, step: 0.05, value: S.tgt[k], fmt: (v) => sg(v, 2), onInput: (v) => { S.tgt[k] = v; H.redraw(); } }));
      function sync() { sl.forEach((s, k) => setVal(s, S.tgt[k], sg(S.tgt[k], 2))); }
      const X = [1.2, 0.8];
      const H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        let moving = false;
        for (let k = 0; k < 4; k++) { const d = S.tgt[k] - S.cur[k]; if (Math.abs(d) > 2e-3) { S.cur[k] += d * Math.min(1, dt * 6 + 0.02); moving = true; } else S.cur[k] = S.tgt[k]; }
        const [a, b, c, d] = S.cur;
        const rows = S.view === 'rows';
        const narrow = w < 620;
        const PL = narrow ? { x: 0, y: 0, w, h: h * 0.62 } : { x: 0, y: 0, w: w * 0.6, h };
        const PR = narrow ? { x: 10, y: h * 0.62 + 6, w: w - 20, h: h * 0.38 - 14 } : { x: w * 0.6 + 6, y: 16, w: w * 0.4 - 22, h: h - 32 };
        const cx = PL.x + PL.w / 2, cy = PL.y + PL.h / 2, u = Math.min(PL.w, PL.h) / 5.4;
        const T = (x, y) => [cx + (a * x + b * y) * u, cy - (c * x + d * y) * u];
        const O = (x, y) => [cx + x * u, cy - y * u];
        const det = a * d - b * c;
        const mxAbs = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d));
        const rank = Math.abs(det) > 0.02 ? 2 : mxAbs > 0.02 ? 1 : 0;
        const R1 = [a, b], R2 = [c, d];
        const reach = Math.hypot(PL.w, PL.h) / (2 * u) + 1;
        const lineThrough = (p, dir, col, lw, dash) => { const q1 = O(p[0] - dir[0] * reach, p[1] - dir[1] * reach), q2 = O(p[0] + dir[0] * reach, p[1] + dir[1] * reach); V.line(g, q1[0], q1[1], q2[0], q2[1], { color: col, lw, dash }); };
        clipRect(g, PL);
        // grade original
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let k = -7; k <= 7; k++) { let p = O(k, -7), q = O(k, 7); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); p = O(-7, k); q = O(7, k); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); }
        g.stroke();
        const o = O(0, 0);
        if (!rows) {
          // ---- leitura por colunas: a grade inteira se deforma
          for (let k = -7; k <= 7; k++) {
            let p = T(k, -7), q = T(k, 7);
            V.line(g, p[0], p[1], q[0], q[1], { color: V.alpha(C.b, k === 0 ? 0.95 : 0.42), lw: k === 0 ? 2 : 1.1 });
            p = T(-7, k); q = T(7, k);
            V.line(g, p[0], p[1], q[0], q[1], { color: V.alpha(C.a, k === 0 ? 0.95 : 0.42), lw: k === 0 ? 2 : 1.1 });
          }
          const sq = [T(0, 0), T(1, 0), T(1, 1), T(0, 1)];
          g.beginPath(); sq.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath();
          g.fillStyle = V.alpha(C.d, 0.22); g.fill(); g.strokeStyle = C.d; g.lineWidth = 1.5; g.stroke();
          if (rank === 1) {
            // imagem: a reta onde toda saída cai (direção da maior coluna)
            const c1 = [a, c], c2 = [b, d], cc = Math.hypot(...c1) >= Math.hypot(...c2) ? c1 : c2, nn = Math.hypot(...cc);
            lineThrough([0, 0], [cc[0] / nn, cc[1] / nn], V.alpha(C.d, 0.95), 3, [8, 5]);
          }
          const xo = O(X[0], X[1]), xw = T(X[0], X[1]);
          V.arrow(g, o[0], o[1], xo[0], xo[1], { color: V.alpha(C.muted, 0.9), lw: 1.6, dash: [5, 4], head: 8 });
          V.text(g, 'x', xo[0] + 6, xo[1] - 4, { size: 13, font: 'mono', weight: 700, color: C.muted });
          const p1 = T(X[0], 0);
          V.arrow(g, o[0], o[1], p1[0], p1[1], { color: V.alpha(C.a, 0.8), lw: 1.4, dash: [3, 3], head: 7 });
          V.arrow(g, p1[0], p1[1], xw[0], xw[1], { color: V.alpha(C.b, 0.8), lw: 1.4, dash: [3, 3], head: 7 });
          const e1 = T(1, 0), e2 = T(0, 1);
          V.arrow(g, o[0], o[1], e1[0], e1[1], { color: C.a, lw: 3.2, head: 11 });
          V.arrow(g, o[0], o[1], e2[0], e2[1], { color: C.b, lw: 3.2, head: 11 });
          V.arrow(g, o[0], o[1], xw[0], xw[1], { color: C.c, lw: 2.6, head: 10 });
          V.glow(g, xw[0], xw[1], 12, C.c, 0.6);
          pill(g, e1[0] + 8, e1[1] + 16, 'W·e₁ (coluna 1)', C.a, { size: 10 });
          pill(g, e2[0] - 8, e2[1] - 16, 'W·e₂ (coluna 2)', C.b, { size: 10, align: 'right' });
          pill(g, xw[0] + 10, xw[1] - 18, 'W·x', C.c, { size: 11 });
          if (rank === 1) pill(g, PL.x + 12, PL.y + PL.h - 18, 'imagem: toda saída cai na reta âmbar', C.d, { size: 10.5 });
        } else {
          // ---- leitura por linhas: nada se deforma; cada linha é uma régua
          [[R1, C.a], [R2, C.b]].forEach(([r, col]) => {
            const n = Math.hypot(r[0], r[1]);
            if (n < 1e-3) return;
            const dir = [-r[1] / n, r[0] / n], kmax = Math.min(14, Math.ceil(reach * n) + 1);
            for (let k = -kmax; k <= kmax; k++) lineThrough([(r[0] * k) / (n * n), (r[1] * k) / (n * n)], dir, V.alpha(col, k === 0 ? 0.8 : 0.3), k === 0 ? 1.8 : 1, k === 0 ? null : [5, 4]);
          });
          if (rank === 1) {
            const r = Math.hypot(...R1) >= Math.hypot(...R2) ? R1 : R2, n = Math.hypot(...r);
            lineThrough([0, 0], [-r[1] / n, r[0] / n], V.alpha(C.d, 0.95), 3.2, [8, 5]);
          }
          const xo = O(X[0], X[1]);
          // rótulo de uma reta de leitura onde ela sai do quadro (longe da origem, sem amontoar)
          const placed = [];
          const edgeLabel = (p, dir, txt, col) => {
            const m = 14, ex0 = PL.x + m, ex1 = PL.x + PL.w - m, ey0 = PL.y + m + 6, ey1 = PL.y + PL.h - 34;
            const P0 = O(p[0], p[1]), dx = dir[0] * u, dy = -dir[1] * u, pts = [];
            if (Math.abs(dx) > 1e-6) pts.push((ex0 - P0[0]) / dx, (ex1 - P0[0]) / dx);
            if (Math.abs(dy) > 1e-6) pts.push((ey0 - P0[1]) / dy, (ey1 - P0[1]) / dy);
            const ok = pts.map((tt) => [P0[0] + tt * dx, P0[1] + tt * dy]).filter(([sx, sy]) => sx >= ex0 - 0.5 && sx <= ex1 + 0.5 && sy >= ey0 - 0.5 && sy <= ey1 + 0.5).sort((q1, q2) => (q2[0] - q2[1]) - (q1[0] - q1[1]));
            const wd = tw(g, txt, 10.5, 'mono', 600) + 14;
            for (const [sx, sy] of ok) {
              const xx = clamp(sx - wd / 2, PL.x + 4, PL.x + PL.w - wd - 4), yy = sy;
              if (placed.some(([px2, py2]) => Math.abs(px2 - xx) < wd && Math.abs(py2 - yy) < 22)) continue;
              placed.push([xx, yy]); pill(g, xx, yy, txt, col, { size: 10.5 }); return;
            }
          };
          [[R1, C.a, 'y₁'], [R2, C.b, 'y₂']].forEach(([r, col, nm]) => {
            const n2 = r[0] * r[0] + r[1] * r[1];
            if (n2 < 1e-6) return;
            const n = Math.sqrt(n2), yv = r[0] * X[0] + r[1] * X[1], ft = O((r[0] * yv) / n2, (r[1] * yv) / n2);
            lineThrough(X, [-r[1] / n, r[0] / n], V.alpha(col, 0.95), 1.8, null);
            g.lineCap = 'round'; V.line(g, o[0], o[1], ft[0], ft[1], { color: V.alpha(col, 0.45), lw: 8 }); g.lineCap = 'butt';
            V.dot(g, ft[0], ft[1], 3.5, col);
            edgeLabel(X, [-r[1] / n, r[0] / n], nm + ' = ' + sg(yv, 2), col);
          });
          const w1 = O(R1[0], R1[1]), w2 = O(R2[0], R2[1]);
          const tipLab = (r, wpt, txt, col) => { const n = Math.hypot(r[0], r[1]); if (n < 1e-3) return; V.text(g, txt, wpt[0] + (r[0] / n) * 16, wpt[1] - (r[1] / n) * 16 + 4, { size: 13, font: 'mono', weight: 700, color: col, align: 'center' }); };
          if (Math.hypot(...R1) > 1e-3) V.arrow(g, o[0], o[1], w1[0], w1[1], { color: C.a, lw: 3.2, head: 11 });
          if (Math.hypot(...R2) > 1e-3) V.arrow(g, o[0], o[1], w2[0], w2[1], { color: C.b, lw: 3.2, head: 11 });
          V.arrow(g, o[0], o[1], xo[0], xo[1], { color: C.c, lw: 2.6, head: 10 });
          V.glow(g, xo[0], xo[1], 12, C.c, 0.6);
          tipLab(R1, w1, 'w₁', C.a); tipLab(R2, w2, 'w₂', C.b);
          pill(g, xo[0] + 10, xo[1] - 18, 'x', C.c, { size: 11 });
          if (rank === 1) pill(g, PL.x + 12, PL.y + PL.h - 18, 'núcleo: nenhuma régua enxerga a reta âmbar', C.d, { size: 10.5 });
        }
        g.restore();
        // painel
        panelBox(g, PR, C, rows ? 'yᵢ = wᵢ · x   (linha i = régua i)' : 'W·x = x₁·(coluna 1) + x₂·(coluna 2)');
        const cell = narrow ? 38 : 46, mx0 = PR.x + 20, my0 = PR.y + 34;
        const cells = [[a, b], [c, d]];
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
          const col = rows ? (i === 0 ? C.a : C.b) : (j === 0 ? C.a : C.b);
          V.box(g, mx0 + j * (cell + 6), my0 + i * (cell + 6), cell, cell, { r: 8, fill: V.alpha(col, 0.14), stroke: col, lw: 1.5 });
          V.text(g, sg(cells[i][j], 2), mx0 + j * (cell + 6) + cell / 2, my0 + i * (cell + 6) + cell / 2 + 1, { size: narrow ? 12 : 13, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
        }
        const tx = narrow ? mx0 + 2 * (cell + 6) + 16 : PR.x + 18;
        let ty = narrow ? PR.y + 44 : my0 + 2 * (cell + 6) + 26;
        const lw = PR.x + PR.w - tx - 12;
        const T12 = (s, o2) => { ty = wrap(g, s, tx, ty, lw, 16, o2); };
        if (!rows) {
          V.text(g, 'det W = ad − bc = ' + sg(det, 2), tx, ty, { size: 13, font: 'mono', weight: 700, color: C.d, maxW: lw }); ty += 20;
          T12('o quadrado unitário vira um paralelogramo de área ' + nf(Math.abs(det), 2) + (det < -0.02 ? ' (det < 0: o plano foi espelhado)' : ''), { size: 12, color: C.muted });
          ty += 4;
          V.text(g, 'posto = ' + rank, tx, ty, { size: 13, font: 'mono', weight: 700, color: rank < 2 ? C.c : C.a }); ty += 18;
          T12(rank === 2 ? 'as duas colunas são independentes: nada se perde, existe inversa' : rank === 1 ? 'as colunas são paralelas: o plano desaba na reta âmbar (a imagem) e não há volta' : 'tudo vai para a origem', { size: 12, color: C.muted });
          ty += 6; T12('W·x = 1,2·(coluna 1) + 0,8·(coluna 2): as setas tracejadas', { size: 12, color: C.muted });
        } else {
          const y1 = a * X[0] + b * X[1], y2 = c * X[0] + d * X[1];
          const f2 = (v) => (v < 0 ? '(' + sg(v, 2) + ')' : nf(v, 2));
          V.text(g, 'y₁ = ' + sg(y1, 2), tx, ty, { size: 13, font: 'mono', weight: 700, color: C.a, maxW: lw }); ty += 18;
          V.text(g, '= ' + f2(a) + '·1,2 + ' + f2(b) + '·0,8', tx, ty, { size: narrow ? 10.5 : 11.5, font: 'mono', color: C.muted, maxW: lw }); ty += 18;
          V.text(g, 'y₂ = ' + sg(y2, 2), tx, ty, { size: 13, font: 'mono', weight: 700, color: C.b, maxW: lw }); ty += 18;
          V.text(g, '= ' + f2(c) + '·1,2 + ' + f2(d) + '·0,8', tx, ty, { size: narrow ? 10.5 : 11.5, font: 'mono', color: C.muted, maxW: lw }); ty += 18;
          ty += 4;
          V.text(g, 'posto = ' + rank, tx, ty, { size: 13, font: 'mono', weight: 700, color: rank < 2 ? C.c : C.a }); ty += 18;
          T12(rank === 2 ? 'réguas em direções diferentes: o par (y₁, y₂) identifica x sem ambiguidade' : rank === 1 ? 'réguas paralelas medem a mesma coisa: andar ao longo da reta âmbar (o núcleo) não muda nenhuma leitura' : 'nenhuma régua: tudo vira zero', { size: 12, color: C.muted });
          if (!narrow) { ty += 6; T12('retas tracejadas: leituras inteiras de cada régua; a reta cheia que passa por x é a leitura yᵢ (sombra × ‖wᵢ‖)', { size: 12, color: C.muted }); }
        }
        if (moving) H.redraw();
      }, { static: true });
    })();

    /* ================================================== Fig 3: dança linha × coluna */
    (function figDanca() {
      const fig = root.querySelector('#fig-danca');
      stageH(fig, 440, 520);
      const X0 = [[1, 0, 2, -1, 1, 0], [0, 1, -1, 2, 0, 1], [2, -1, 0, 1, 1, -2], [-1, 2, 1, 0, -1, 1]];
      const W = [[1, 0, -1, 2, 0], [0, 1, 1, 0, -1], [2, -1, 0, 1, 1], [-1, 0, 1, 0, 2], [0, 2, -1, 1, 0], [1, 1, 0, -1, 1]];
      const K = 6, N = 5, toks = ['o', 'gato', 'subiu', 'no'];
      const S = { M: 4 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'M (tokens)', options: [{ v: 1, label: '1 · GEMV' }, { v: 2, label: '2' }, { v: 4, label: '4' }], value: 4, onChange: (v) => { S.M = v; } });
      const steps = [
        { t: 'As formas', d: '\\(X\\) tem uma linha por token (\\(M\\)) e \\(K = 6\\) colunas; \\(W\\) tem \\(K = 6\\) linhas e \\(N = 5\\) colunas. O \\(K\\) de dentro precisa casar e desaparece: sobra \\([M\\times N]\\).' },
        { t: 'Uma célula = um produto escalar', d: 'A célula \\(Y_{mn}\\) é a linha \\(m\\) de \\(X\\) encontrando a coluna \\(n\\) de \\(W\\): multiplica par a par e soma os \\(K\\) produtos.' },
        { t: 'Uma linha = um token atravessando a camada', d: 'A linha \\(m\\) de \\(Y\\) só depende da linha \\(m\\) de \\(X\\): cada token é transformado sozinho, com os mesmos pesos. Quem mistura tokens é a atenção, não a camada linear.' },
        { t: 'A GEMM inteira', d: 'São \\(M\\times N\\) células, cada uma com \\(K\\) multiplicações e \\(K\\) somas: \\(2MNK\\) FLOPs. As células são independentes, e é esse paralelismo que a GPU explora.' },
        { t: 'Cada peso serve M tokens', d: 'O mesmo \\(W_{kn}\\) entra na conta de todas as \\(M\\) linhas. Com \\(M = 1\\) (GEMV), cada peso lido da memória rende uma única multiplicação; com \\(M\\) linhas, rende \\(M\\).' },
        { t: 'Em blocos: K cortado ao meio', d: 'Corte \\(K\\) em duas metades: \\(Y = X_1W_1 + X_2W_2\\). Cada GPU calcula uma soma parcial com metade dos pesos e um <i>all-reduce</i> soma as duas: o <i>row-parallel</i> do tensor parallel.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 9000 });
      const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, x, k) => s + x * B[k][j], 0)));
      const colX = (v) => V.cmap(v / 2.4, 'div'), colY = (v) => V.cmap(v / 8, 'div');
      const mfmt = (v) => String(Math.round(v)).replace('-', '−');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = S.M, X = X0.slice(0, M), Y = mm(X, W);
        const Y1 = mm(X.map((r) => r.slice(0, 3)), W.slice(0, 3)), Y2 = mm(X.map((r) => r.slice(3)), W.slice(3));
        const i0 = st.i, el = st.elapsed();
        const narrow = w < 620;
        const labW = 50;
        const cs = narrow ? clamp((w - labW - 24) / (K + N + 0.9) - 2, 16, 30) : clamp(Math.min((w * 0.58 - labW - 30) / (K + N + 0.9) - 2, (h - 96) / (K + 4 + 1.1) - 2), 18, 34);
        const gp = 2, st1 = cs + gp;
        const xX = (narrow ? Math.max(labW, (w - (K + N + 0.9) * st1) / 2 + labW / 2) : labW + 22);
        const yW = narrow ? 34 : 40;
        const xW = xX + K * st1 + 0.9 * cs, yY = yW + K * st1 + 0.9 * cs;
        const yX = yY;
        const inBlocks = i0 === 5;
        // X
        const hiRow = (i0 === 1 || i0 === 2) ? null : null;
        let ci = -1, cj = -1, kk = -1, pc = 0, rowI = -1, rowP = 0;
        if (i0 === 1) { const per = 3.0, idx = Math.floor(el / per) % (M * N); ci = Math.floor(idx / N); cj = idx % N; pc = (el % per) / per; kk = Math.min(K - 1, Math.floor(pc * (K + 1.6))); }
        if (i0 === 2) { const per = 3.6; rowI = Math.floor(el / per) % M; rowP = (el % per) / per; }
        const tintK = (k) => (k < 3 ? C.a : C.b);
        const gX = V.matrix(g, { x: xX, y: yX, data: X, cell: cs, gap: gp, values: true, fmt: mfmt, colorFn: colX, hiRow: i0 === 2 ? rowI : ci >= 0 ? ci : null, hiColor: C.c });
        const gW = V.matrix(g, { x: xW, y: yW, data: W, cell: cs, gap: gp, values: true, fmt: mfmt, colorFn: colX, hiCol: ci >= 0 ? cj : null, hiColor: C.c });
        // Y (células visíveis conforme o passo)
        for (let i = 0; i < 4; i++) for (let j = 0; j < N; j++) {
          const x = xW + j * st1, y = yY + i * st1;
          if (i >= M) { V.box(g, x, y, cs, cs, { r: 4, stroke: V.alpha(C.muted, 0.15), lw: 1 }); continue; }
          let vis = 0;
          if (i0 === 0) vis = 0;
          else if (i0 === 1) { const idx = i * N + j, cur = ci * N + cj; vis = idx < cur ? 1 : idx === cur ? clamp((pc - 0.82) / 0.1, 0, 1) : 0; if (Math.floor(el / 3.0) >= M * N) vis = idx === cur ? vis : 1; }
          else if (i0 === 2) { vis = i < rowI ? 1 : i === rowI ? clamp(rowP * (N + 1) - j, 0, 1) : 0; if (Math.floor(el / 3.6) >= M) vis = i === rowI ? vis : 1; }
          else if (i0 === 3) vis = clamp((el - (i + j) * 0.18) / 0.35, 0, 1);
          else vis = 1;
          V.box(g, x, y, cs, cs, { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          if (vis > 0) {
            g.globalAlpha = vis;
            V.box(g, x, y, cs, cs, { r: 4, fill: colY(Y[i][j]) });
            V.text(g, String(Y[i][j]).replace('-', '−'), x + cs / 2, y + cs / 2 + 1, { size: Math.min(12, cs * 0.42), font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
            g.globalAlpha = 1;
          }
        }
        if (inBlocks) {
          [[0, C.a], [3, C.b]].forEach(([k0, c]) => {
            g.strokeStyle = c; g.lineWidth = 2.2;
            V.rr(g, xX + k0 * st1 - 3, yX - 3, 3 * st1 - gp + 6, M * st1 - gp + 6, 6); g.stroke();
            V.rr(g, xW - 3, yW + k0 * st1 - 3, N * st1 - gp + 6, 3 * st1 - gp + 6, 6); g.stroke();
          });
          V.text(g, 'X₁', xX + 1.5 * st1, yX - 8, { size: 12, font: 'mono', weight: 700, color: C.a, align: 'center' });
          V.text(g, 'X₂', xX + 4.5 * st1, yX - 8, { size: 12, font: 'mono', weight: 700, color: C.b, align: 'center' });
          V.text(g, 'W₁', xW - 8, yW + 1.5 * st1, { size: 12, font: 'mono', weight: 700, color: C.a, align: 'right', base: 'middle' });
          V.text(g, 'W₂', xW - 8, yW + 4.5 * st1, { size: 12, font: 'mono', weight: 700, color: C.b, align: 'right', base: 'middle' });
        }
        // rótulos de tokens
        for (let i = 0; i < M; i++) pill(g, xX - 8, yX + i * st1 + cs / 2, toks[i], (i0 === 2 && i === rowI) || i === ci ? C.c : C.muted, { size: 10.5, align: 'right', a: 0.12 });
        // formas
        const pulse = i0 === 0 ? 0.5 + 0.5 * Math.sin(t * 5) : 0;
        const kcol = i0 === 0 ? V.mix(C.dark ? '#E9FAF2' : '#222222', C.c.startsWith('#') ? C.c : '#FC93D6', pulse) : C.muted;
        V.text(g, 'X [' + M + ' × 6]', xX + (K * st1) / 2, yX + 4 * st1 + 16, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink });
        V.text(g, 'W [6 × 5]', xW + (N * st1) / 2, yW - 10, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink });
        V.text(g, 'Y [' + M + ' × 5]', xW + (N * st1) / 2, yY + 4 * st1 + 16, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink });
        if (i0 === 0) {
          const pc2 = V.alpha(C.c, 0.55 + 0.45 * pulse);
          const by = yX - 10;
          V.line(g, xX, by, xX + K * st1 - gp, by, { color: pc2, lw: 2 }); V.line(g, xX, by - 5, xX, by + 3, { color: pc2, lw: 2 }); V.line(g, xX + K * st1 - gp, by - 5, xX + K * st1 - gp, by + 3, { color: pc2, lw: 2 });
          const bx = xW - 10;
          V.line(g, bx, yW, bx, yW + K * st1 - gp, { color: pc2, lw: 2 }); V.line(g, bx - 5, yW, bx + 3, yW, { color: pc2, lw: 2 }); V.line(g, bx - 5, yW + K * st1 - gp, bx + 3, yW + K * st1 - gp, { color: pc2, lw: 2 });
          V.arrow(g, xX + (K * st1) / 2, by - 8, bx - 10, yW + (K * st1) / 2, { color: pc2, lw: 1.6, head: 8, curve: -30 });
          V.text(g, 'K = 6 = K', xX + (K * st1) / 2 - 10, yW + (K * st1) / 2 - 14, { size: 13, font: 'mono', weight: 700, align: 'center', color: kcol });
          V.text(g, 'dimensão contraída', xX + (K * st1) / 2 - 10, yW + (K * st1) / 2 + 4, { size: 11.5, align: 'center', color: C.muted });
        }
        // produtos da célula (passo 1)
        if (i0 === 1 && ci >= 0) {
          for (let k = 0; k <= kk; k++) {
            const a = gX.center(ci, k), b = gW.center(k, cj);
            g.strokeStyle = V.alpha(C.c, k === kk ? 0.95 : 0.35); g.lineWidth = k === kk ? 2.4 : 1.2;
            V.rr(g, a[0] - cs / 2 - 2, a[1] - cs / 2 - 2, cs + 4, cs + 4, 5); g.stroke();
            V.rr(g, b[0] - cs / 2 - 2, b[1] - cs / 2 - 2, cs + 4, cs + 4, 5); g.stroke();
          }
          const target = [xW + cj * st1 + cs / 2, yY + ci * st1 + cs / 2];
          const a = gX.center(ci, kk), b = gW.center(kk, cj);
          V.arrow(g, a[0] + cs / 2, a[1], target[0] - cs / 2, target[1], { color: V.alpha(C.c, 0.6), lw: 1.4, head: 7, curve: -10 });
          V.arrow(g, b[0], b[1] + cs / 2, target[0], target[1] - cs / 2, { color: V.alpha(C.c, 0.6), lw: 1.4, head: 7, curve: 10 });
        }
        // painel de texto
        const PX = narrow ? 14 : xW + N * st1 + 28, PY = narrow ? yY + 4 * st1 + 28 : yW + 4, PW = narrow ? w - 28 : w - PX - 16;
        const fmtv = (v) => (v < 0 ? '(−' + -v + ')' : String(v));
        let yy = PY + 12;
        const T = (s, o = {}) => { yy = wrap(g, s, PX, yy, PW, o.lh || 18, { size: o.size || 12.5, font: o.font || 'sans', weight: o.weight || 400, color: o.color || C.ink }); };
        if (i0 === 0) {
          T('[M × K] · [K × N] → [M × N]', { font: 'mono', weight: 700, size: 13.5 });
          T('As dimensões de dentro precisam casar. As de fora sobram.', { color: C.muted });
          T('M = ' + M + ' tokens, K = 6 entradas, N = 5 saídas.', { color: C.muted });
        } else if (i0 === 1 && ci >= 0) {
          T('Y[' + ci + ',' + cj + '] = Σₖ X[' + ci + ',k] · W[k,' + cj + ']', { font: 'mono', weight: 700, size: 13 });
          const terms = []; let sum = 0;
          for (let k = 0; k <= kk; k++) { terms.push(fmtv(X[ci][k]) + '·' + fmtv(W[k][cj])); sum += X[ci][k] * W[k][cj]; }
          T(terms.join(' + ') + (kk === K - 1 ? ' = ' + String(sum).replace('-', '−') : ' + …'), { font: 'mono', size: 12, color: C.c });
          T('o token “' + toks[ci] + '” medido pelo detector da coluna ' + cj, { color: C.muted });
        } else if (i0 === 2) {
          T('token “' + toks[Math.max(0, rowI)] + '”: [1 × 6] → [1 × 5]', { font: 'mono', weight: 700, size: 13 });
          T('Mesmos pesos para todos os tokens; cada linha sai sozinha.', { color: C.muted });
        } else if (i0 === 3) {
          const fl = 2 * M * N * K;
          T('2 · M · N · K = 2 · ' + M + ' · 5 · 6 = ' + fl + ' FLOPs', { font: 'mono', weight: 700, size: 13 });
          T('Escalando para uma matriz real: o in_proj do KDA com M = 7168 custa ≈ 738 GFLOP.', { color: C.muted });
        } else if (i0 === 4) {
          T('pesos lidos: 30  ·  multiplicações: 30 × M = ' + 30 * M, { font: 'mono', weight: 700, size: 13 });
          T(M === 1 ? 'M = 1 (GEMV): cada peso lido serve a uma conta só. É o regime do decode com uma sequência.' : 'M = ' + M + ': cada peso lido serve a ' + M + ' contas. Mais linhas por leitura = mais trabalho útil por byte.', { color: C.muted });
        } else if (i0 === 5) {
          T('Y = X₁W₁ + X₂W₂', { font: 'mono', weight: 700, size: 13.5 });
          const sc = clamp((PW - 40) / 10 - 2, 13, narrow ? 18 : 22);
          const sy = yy + 6;
          const draw = (data, x, lab, col) => { V.matrix(g, { x, y: sy, data, cell: sc, gap: 2, values: sc >= 16, fmt: mfmt, colorFn: colY }); V.text(g, lab, x + (N * (sc + 2)) / 2, sy + M * (sc + 2) + 14, { size: 11, font: 'mono', color: col, align: 'center' }); };
          const wM = N * (sc + 2);
          draw(Y1, PX, 'X₁W₁ · GPU 0', C.a);
          V.text(g, '+', PX + wM + 12, sy + (M * (sc + 2)) / 2 + 5, { size: 18, weight: 700, color: C.ink, align: 'center' });
          draw(Y2, PX + wM + 24, 'X₂W₂ · GPU 1', C.b);
          yy = sy + M * (sc + 2) + (narrow ? 30 : 36);
          T('= Y, a matriz de saída: o all-reduce soma as parciais célula a célula.', { color: C.muted, lh: narrow ? 16 : 18 });
        }
        if (i0 === 4) {
          for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
            const x = xW + j * st1 + cs - 2, y = yW + k * st1 + 3;
            V.text(g, '×' + M, x, y + 6, { size: 9, font: 'mono', weight: 700, align: 'right', color: C.dark ? '#FCBB00' : '#8a5a00' });
          }
          for (let i = 0; i < M; i++) {
            const f = ((t * 0.8 + i * 0.25) % 1);
            for (let j = 0; j < N; j++) { const x = xW + j * st1 + cs / 2, y0 = yW + K * st1, y1 = yY + i * st1 + cs / 2; V.dot(g, x, lerp(y0 - 4, y1, f), 2.2, V.alpha(C.d, 0.9)); }
          }
        }
      });
    })();

    /* ================================================== Fig 4: cubo das cabeças */
    (function figCabecas() {
      const fig = root.querySelector('#fig-cabecas');
      stageH(fig, 420, 390);
      const MODELS = {
        mla: { lab: 'GLM · MLA', H: 64, dh: 256, d: 4096, proj: 'q_b_proj [16384 × 1536]', note: '64 cabeças × 256 = 16.384; o_proj [4096 × 16384]' },
        kda: { lab: 'GLM · KDA', H: 64, dh: 128, d: 4096, proj: 'q_proj [8192 × 4096]', note: '64 cabeças × 128 = 8.192 (e o mesmo para k e v)' },
        qsa: { lab: 'Qwen · QSA', H: 24, dh: 256, d: 2560, proj: 'q_proj [12288 × 2560]', note: '12.288 = 24 × 256 × 2: consulta e gate empacotados; um split separa os dois' },
        ds: { lab: 'DeepSeek', H: 64, dh: 512, d: 4096, proj: 'wq_b [32768 × 1024]', note: '64 cabeças × 512 = 32.768, a partir do latente de 1024' },
      };
      const S = { m: 'mla', tp: false, bb: false, cur: { gap: 0, stack: 0, scores: 0, merged: 0, bb: 0 } };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: Object.keys(MODELS).map((k) => ({ v: k, label: MODELS[k].lab })), value: 'mla', onChange: (v) => { S.m = v; H.redraw(); } });
      UI.toggle(ctl, { label: 'TP=2 (metade das cabeças por GPU)', value: false, onChange: (v) => { S.tp = v; H.redraw(); } });
      UI.toggle(ctl, { label: 'lote B = 2 (tensor 4D)', value: false, onChange: (v) => { S.bb = v; H.redraw(); } });
      const steps = [
        { t: 'A projeção: [T, H·dₕ]', d: 'A GEMM de query devolve uma matriz larga: cada token ganha \\(H\\cdot d_h\\) números, com as cabeças lado a lado.' },
        { t: 'reshape → [T, H, dₕ]', d: 'Nenhum byte se move: só passamos a ler a largura como \\(H\\) blocos de \\(d_h\\). É uma <i>view</i>.' },
        { t: 'transpose → [H, T, dₕ]', d: 'Trocamos os dois primeiros eixos: cada cabeça vira uma fatia com todos os tokens. Também é uma view: os strides trocam de lugar.' },
        { t: 'Cada cabeça, seus escores [T, T]', d: 'Cada fatia calcula \\(QK^\\top/\\sqrt{d_h}\\) sozinha, com máscara causal: \\(H\\) atenções independentes, lado a lado.' },
        { t: 'De volta a [T, d]', d: 'Transposição e reshape inversos juntam as cabeças em \\([T, H\\cdot d_h]\\), e o <code>o_proj</code> leva de volta à largura do residual.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => { if (H) H.redraw(); } });
      const TGT = [{ gap: 0, stack: 0, scores: 0, merged: 0 }, { gap: 1, stack: 0, scores: 0, merged: 0 }, { gap: 1, stack: 1, scores: 0, merged: 0 }, { gap: 1, stack: 1, scores: 1, merged: 0 }, { gap: 0, stack: 0, scores: 0, merged: 1 }];
      const Hs = 4, Ds = 6, T = 5;
      const rnd = V.rng(21); const pat = Array.from({ length: Hs }, () => Array.from({ length: T }, () => Array.from({ length: Ds }, () => rnd())));
      const scr = Array.from({ length: Hs }, (_, hh) => Array.from({ length: T }, (_, i) => { const r = V.rng(50 + hh * 7 + i); const row = Array.from({ length: T }, (_, j) => (j <= i ? r() * 2 + (j === i ? 1 : 0) : -Infinity)); const e = row.map((x) => (x === -Infinity ? 0 : Math.exp(x))); const s = e.reduce((p, q) => p + q, 0); return e.map((x) => x / s); }));
      H = liveFig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const Mo = MODELS[S.m];
        const tg = Object.assign({ bb: S.bb ? 1 : 0 }, TGT[st.i]);
        const kf = Math.min(1, dt * 5 + 0.01);
        let moving = false;
        for (const k in S.cur) { const dd = tg[k] - S.cur[k]; if (Math.abs(dd) > 1e-3) { S.cur[k] += dd * kf; moving = true; } else S.cur[k] = tg[k]; }
        const { gap, stack, scores, merged, bb } = S.cur;
        const Bp = bb > 0.5 ? 'B, ' : '';
        const narrow = w < 620;
        const c = narrow ? clamp((w - 50) / (Hs * Ds + 5), 10, 15) : clamp(Math.min((w * 0.8) / (Hs * Ds + 5), (h - 170) / (T * 2.4)), 14, 24);
        const hw = Ds * c, hh = T * c;
        const x0 = narrow ? 20 : 44, y0 = narrow ? 86 : 104;
        const gapPx = gap * c * 1.6;
        const headCol = (k) => (S.tp ? (k < Hs / 2 ? C.a : C.b) : C.series[k % 6]);
        // posições: fila (row) e pilha (stack em perspectiva)
        const dxs = c * 2.1, dys = c * 1.35;
        const sx = x0 + 6, sy = y0 + (Hs - 1) * dys + 6;
        const pos = [];
        for (let k = 0; k < Hs; k++) {
          const rx = x0 + k * (hw + gapPx), ry = y0;
          const kx = sx + k * dxs, ky = sy - k * dys;
          pos.push([lerp(rx, kx, V.ease.inOut(stack)), lerp(ry, ky, V.ease.inOut(stack))]);
        }
        const order = stack > 0.5 ? [3, 2, 1, 0] : [0, 1, 2, 3];
        // lote B = 2: a mesma pilha da sequência b = 1, desenhada atrás
        const bo = c * 0.85 * bb;
        if (bb > 0.02) order.forEach((k) => {
          const [px, py] = pos[k], col = headCol(k);
          g.globalAlpha = bb;
          g.setLineDash([4, 3]); V.box(g, px - 3 + bo, py - 3 - bo, hw + 6, hh + 6, { r: 6, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.85), lw: 1.2 }); g.setLineDash([]);
          for (let i = 0; i < T; i++) for (let j = 0; j < Ds; j++) { g.fillStyle = V.alpha(col, 0.16 + 0.3 * pat[(k + 1) % Hs][i][j]); V.rr(g, px + bo + j * c + 1, py - bo + i * c + 1, c - 2, c - 2, 2); g.fill(); }
          g.globalAlpha = 1;
        });
        order.forEach((k) => {
          const [px, py] = pos[k], col = headCol(k);
          if (stack > 0.3) { g.fillStyle = C.bg || (C.dark ? '#0b1a13' : '#ffffff'); V.rr(g, px - 3, py - 3, hw + 6, hh + 6, 6); g.fill(); }
          V.box(g, px - 3, py - 3, hw + 6, hh + 6, { r: 6, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.75 + 0.25 * stack), lw: 1.3 });
          for (let i = 0; i < T; i++) for (let j = 0; j < Ds; j++) {
            g.fillStyle = V.alpha(col, 0.25 + 0.6 * pat[k][i][j]);
            V.rr(g, px + j * c + 1, py + i * c + 1, c - 2, c - 2, 2); g.fill();
          }
          if (gap > 0.5 || stack > 0.5) {
            const lab = 'cabeça ' + (k === Hs - 1 ? 'H−1' : k);
            if (stack > 0.5) { if (k === 0) V.text(g, lab + ' (frente)', px, py + hh + 16, { size: 10.5, font: 'mono', color: col }); else if (k === Hs - 1) V.text(g, lab + ' (fundo)', px + hw + bo, py - 10 - bo, { size: 10.5, font: 'mono', color: col, align: 'right' }); }
            else V.text(g, lab, px + hw / 2, py - 10 - bo, { size: 10.5, font: 'mono', color: col, align: 'center' });
          }
        });
        // eixos do primeiro bloco (na fila)
        if (stack < 0.5 && merged < 0.5) {
          const [px, py] = pos[0];
          V.text(g, 'T', px - 12, py + hh / 2 + 4, { size: 12, font: 'mono', weight: 700, color: C.muted, align: 'center' });
          if (gap < 0.5) {
            const tot = Hs * hw;
            V.line(g, x0, y0 + hh + 10, x0 + tot, y0 + hh + 10, { color: C.muted, lw: 1 });
            V.text(g, 'H·dₕ = ' + nf(Mo.H * Mo.dh), x0 + tot / 2, y0 + hh + 26, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
          } else {
            V.line(g, px, py + hh + 10, px + hw, py + hh + 10, { color: C.muted, lw: 1 });
            V.text(g, 'dₕ = ' + Mo.dh, px + hw / 2, py + hh + 26, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
            V.text(g, '× H = ' + Mo.H + ' cabeças', x0 + Hs * (hw + gapPx) - gapPx, py + hh + 26, { size: 12, font: 'mono', color: C.muted, align: 'right' });
          }
        }
        // rótulo de forma
        const Hd = Mo.H * Mo.dh;
        const shapeTxt = [
          '[' + Bp + 'T, ' + nf(Hd) + '] = [' + Bp + 'T, ' + Mo.H + '·' + Mo.dh + ']',
          '[' + Bp + 'T, ' + Mo.H + ', ' + Mo.dh + ']   (view)',
          '[' + Bp + Mo.H + ', T, ' + Mo.dh + ']   (view transposta)',
          'S = QKᵀ/√' + Mo.dh + ' → [' + Bp + Mo.H + ', T, T]',
          '[' + Bp + 'T, ' + nf(Hd) + '] → o_proj → [' + Bp + 'T, ' + nf(Mo.d) + ']',
        ][st.i];
        V.text(g, shapeTxt, x0, narrow ? 28 : 34, { size: narrow ? 13 : 16, font: 'mono', weight: 700, color: C.ink, maxW: w - 2 * x0 });
        V.text(g, Mo.lab + ' · ' + Mo.proj, x0, narrow ? 48 : 58, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * x0 });
        // escores por cabeça
        if (scores > 0.02) {
          const sc = narrow ? 8 : 11;
          const gx = narrow ? w - T * sc - 16 : Math.min(w - T * sc - 30, sx + (Hs - 1) * dxs + hw + 150);
          const gy0 = narrow ? y0 - 6 : y0 - 16;
          for (let k = 0; k < Hs; k++) {
            const gy = gy0 + (Hs - 1 - k) * (T * sc + (narrow ? 12 : 16));
            g.globalAlpha = scores;
            for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
              const v = scr[k][i][j];
              g.fillStyle = j > i ? V.alpha(C.muted, 0.12) : V.cmap(Math.min(1, v * 1.8), 'seq');
              V.rr(g, gx + j * sc, gy + i * sc, sc - 1.5, sc - 1.5, 1.5); g.fill();
            }
            if (!narrow) V.text(g, 'S de ' + (k === Hs - 1 ? 'H−1' : k), gx - 6, gy + (T * sc) / 2, { size: 10, font: 'mono', color: headCol(k), align: 'right', base: 'middle' });
            else V.text(g, 'S' + ['₀', '₁', '₂', 'ₕ₋₁'][k], gx - 4, gy + (T * sc) / 2, { size: 10, font: 'mono', weight: 600, color: headCol(k), align: 'right', base: 'middle' });
            const [px, py] = pos[k];
            V.line(g, px + hw + 4, py + hh / 2, gx - (narrow ? 26 : 52), gy + (T * sc) / 2, { color: V.alpha(headCol(k), 0.55), lw: 1, dash: [3, 3] });
            g.globalAlpha = 1;
          }
          V.text(g, narrow ? 'escores [T, T]' : '[T, T] por cabeça (máscara causal)', gx + T * sc, gy0 - (narrow ? 8 : 10), { size: narrow ? 10 : 10.5, color: C.muted, align: 'right' });
        }
        // o_proj (abaixo da fila)
        if (merged > 0.02) {
          g.globalAlpha = merged;
          const rowW = Hs * hw;
          const oy = y0 + hh + 34;
          const ob = { x: x0 + rowW / 2 - 40, y: oy, w: 80, h: 30 };
          V.arrow(g, x0 + rowW / 2, y0 + hh + 6, x0 + rowW / 2, oy - 2, { color: C.d, lw: 2, head: 9 });
          V.box(g, ob.x, ob.y, ob.w, ob.h, { r: 8, fill: V.alpha(C.d, 0.2), stroke: C.d, lw: 1.5, label: 'o_proj', size: 12 });
          const dw = Math.min(narrow ? 90 : 140, rowW * 0.3), dx = ob.x + ob.w + 40;
          V.arrow(g, ob.x + ob.w + 4, ob.y + ob.h / 2, dx - 6, ob.y + ob.h / 2, { color: C.d, lw: 2, head: 9 });
          const cc2 = Math.min(c, 14);
          for (let i = 0; i < T; i++) { g.fillStyle = V.alpha(C.d, 0.3 + 0.12 * (i % 3)); V.rr(g, dx, ob.y + ob.h / 2 - (T * cc2) / 2 + i * cc2 + 1, dw, cc2 - 2, 2); g.fill(); }
          V.text(g, '[' + Bp + 'T, ' + nf(Mo.d) + ']', dx + dw / 2, ob.y + ob.h / 2 + (T * cc2) / 2 + 16, { size: 11.5, font: 'mono', color: C.d, align: 'center' });
          g.globalAlpha = 1;
        }
        // TP e nota do modelo
        const ny = h - (narrow ? 62 : 52);
        if (S.tp) {
          const wd = pill(g, x0, ny - 4, 'GPU 0: ' + Mo.H / 2 + ' cabeças', C.a, { size: 11 });
          pill(g, x0 + wd + 8, ny - 4, 'GPU 1: ' + Mo.H / 2 + ' cabeças', C.b, { size: 11 });
        }
        const ny2 = wrap(g, Mo.note, x0, ny + 18, w - 2 * x0, 15, { size: 11.5, color: C.muted });
        wrap(g, 'desenho simbólico: 5 tokens, 4 cabeças × 6 dims' + (bb > 0.5 ? ' · tracejado atrás: a sequência b = 1' : ''), x0, ny2 + 2, w - 2 * x0, 13, { size: 10.5, color: C.muted });
        return moving;
      });
    })();

    /* ================================================== Fig 5: broadcasting + einsum */
    (function figBroadcast() {
      const fig = root.querySelector('#fig-broadcast');
      const CASES = {
        bias: { lab: 'viés: [M, N] + [N]', A: [4, 5], B: [5], op: '+', note: 'O mesmo viés é somado a cada token (linha).', recipe: 'camadas com bias (ex.: torre de visão)' },
        rms: { lab: 'ganho RMSNorm: [T, d] ⊙ [d]', A: [4, 6], B: [6], op: '⊙', note: 'Cada coordenada tem seu ganho aprendido, igual para todos os tokens.', recipe: 'todas as RMSNorm das três receitas' },
        esc: { lab: 'escala por canal: W[N, K] ⊙ s[N, 1]', A: [5, 6], B: [5, 1], op: '⊙', note: 'Uma escala por linha (canal de saída), esticada ao longo da entrada K.', recipe: 'FP8 por canal do GLM (amax/448 por linha, na carga)' },
        ext: { lab: 'produto externo: [K, 1] ⊙ [1, N]', A: [5, 1], B: [1, 6], op: '⊙', note: 'Os dois operandos esticam: sai uma matriz inteira de posto 1.', recipe: 'escrita v·kᵀ das atenções lineares' },
        mhc: { lab: 'mHC: post_mix ⊗ y', A: ['T', 4, 1], B: ['T', 1, 4096], op: '⊙', note: 'Cada um dos 4 fluxos recebe a saída y com seu próprio peso: produto externo com eixo de lote T.', recipe: 'GLM e DeepSeek (mHC, 4 fluxos)' },
        err: { lab: 'gate por cabeça: [T, 64, 128] ⊙ [64]', A: ['T', 64, 128], B: [64], op: '⊙', note: 'Alinhando pela direita, 64 encontra 128: erro. (Ilustrativo, inspirado no gate por cabeça do KDA.)', recipe: 'ilustrativo' },
        fix: { lab: 'corrigido: [T, 64, 128] ⊙ [64, 1]', A: ['T', 64, 128], B: [64, 1], op: '⊙', note: 'Com A[:, None], 64 casa com 64 e o 1 estica sobre os 128 canais, sem cópia (stride 0).', recipe: 'ilustrativo' },
        e_gemm: { lab: 'einsum · mk,kn->mn (GEMM)', es: 'mk,kn->mn', ops: ['X', 'W'], sz: { m: 32, k: 4096, n: 12576 }, flops: 'FLOPs = 2·m·k·n = 2 × 32 × 4.096 × 12.576 ≈ 3,3 GFLOP', note: 'k está nas duas entradas e não está na saída: é somada. É a GEMM Y = XW com as formas do in_proj do KDA (por GPU) num passo de decode com M = 32.', recipe: 'GLM' },
        e_att: { lab: 'einsum · thd,shd->hts (escores)', es: 'thd,shd->hts', ops: ['q', 'k'], sz: { t: 'T', h: 64, d: 256, s: 'T' }, flops: 'FLOPs = 2·t·s·h·d = 32.768·T² (T = 8.192: ≈ 2,2 TFLOP por camada)', note: 'h está nas duas entradas e na saída: é lote, e cada cabeça faz a sua conta sozinha. d é somada. Sai um [T, T] por cabeça (formas da MLA do GLM: 64 cabeças × 256).', recipe: 'GLM' },
        e_ds: { lab: 'einsum · bsgd,grd->bsgr (DeepSeek)', es: 'bsgd,grd->bsgr', ops: ['o', 'wo_a'], sz: { b: 'B', s: 'T', g: 8, d: 4096, r: 1024 }, flops: 'FLOPs = 2·b·s·g·d·r = 67,1 MFLOP por token (b·s = 1)', note: 'g é lote: cada um dos 8 grupos (8 cabeças × 512 = 4096) tem a sua própria matriz 1024 × 4096. d é somada. Em TP=2, cada GPU fica com 4 grupos.', recipe: 'DeepSeek-V4 (o_groups = 8, o_lora_rank = 1024)' },
      };
      const S = { c: 'esc', t0: 0 };
      const kindOf = (c) => (CASES[c].es ? 'es' : c === 'mhc' ? 'mhc' : c === 'err' || c === 'fix' ? 'err' : '2d');
      const applyH = stageH(fig, () => (kindOf(S.c) === 'es' ? 430 : 400), () => ({ '2d': 490, mhc: 410, err: 345, es: 450 })[kindOf(S.c)]);
      let H = null;
      const ctl = UI.controls(fig);
      fitSelect(UI.select(ctl, { label: 'Caso', options: Object.keys(CASES).map((k) => ({ v: k, label: CASES[k].lab })), value: 'esc', onChange: (v) => { S.c = v; S.t0 = H.clock(); applyH(); H.redraw(); } }));
      const res = UI.readout(ctl, 'resultado', '—');
      UI.button(ctl, '↻ repetir', () => { S.t0 = H.clock(); H.redraw(); });
      function analyze(A, B) {
        const n = Math.max(A.length, B.length), cols = [], out = [];
        let ok = true;
        for (let i = 0; i < n; i++) {
          const a = A[A.length - n + i], b = B[B.length - n + i];
          const am = a === undefined, bm = b === undefined;
          const av = am ? 1 : a, bv = bm ? 1 : b;
          let s;
          if (av === bv) s = 'eq'; else if (av === 1) s = 'sa'; else if (bv === 1) s = 'sb'; else { s = 'err'; ok = false; }
          cols.push({ a, b, am, bm, s });
          out.push(s === 'err' ? '✕' : s === 'sa' ? bv : av);
        }
        return { cols, out, ok };
      }
      const vals = (R, K, seed) => { const r = V.rng(seed); return Array.from({ length: R }, () => Array.from({ length: K }, () => Math.round(r() * 4 - 1.5))); };
      const szTxt = (v) => (typeof v === 'number' ? String(v) : v);
      H = liveFig(fig, (g, w, h, t) => {
        const C = V.C();
        const cs = CASES[S.c];
        const el = t - S.t0;
        const narrow = w < 620;
        let bottom = 0, anim = false;
        if (cs.es) {
          /* ---------------- einsum: alinhar pelo nome */
          const [ins, outS] = cs.es.split('->'), [Aa, Bb] = ins.split(',');
          const letters = [...new Set((Aa + Bb).split(''))];
          const PAL = [C.a, C.b, C.d, C.f || C.c, C.e || C.a];
          const colL = {}; letters.forEach((L, i) => { colL[L] = PAL[i % PAL.length]; });
          const role = (L) => (outS.includes(L) ? (Aa.includes(L) && Bb.includes(L) ? 'lote' : 'livre') : 'soma');
          res.set('[' + outS.split('').map((L) => L + '=' + (typeof cs.sz[L] === 'number' ? nf(cs.sz[L]) : cs.sz[L])).join(', ') + ']');
          const P = 5, cyc = 2;
          anim = el < P * cyc;
          const f = anim ? (el % P) / P : 1;
          const bwE = narrow ? 60 : 78, bhE = 36, gE = narrow ? 6 : 10;
          const x0e = narrow ? 66 : 300;
          const rY = narrow ? [20, 76, 146] : [24, 84, 160];
          const xs = (i) => x0e + i * (bwE + gE);
          [cs.ops[0], cs.ops[1], 'saída'].forEach((l, r) => V.text(g, l, x0e - 12, rY[r] + bhE / 2 + 4, { size: 13, font: 'mono', weight: 700, color: r === 2 ? C.d : C.ink, align: 'right' }));
          V.line(g, x0e - 4, rY[2] - 14, xs(outS.length) + 6, rY[2] - 14, { color: V.alpha(C.muted, 0.5), lw: 1 });
          if (!narrow) {
            let ly = 40;
            [['Σ', 'somada: some da saída'], ['lote', 'nas duas entradas e na saída'], ['livre', 'de uma entrada só']].forEach(([k2, l2]) => { V.text(g, k2, 22, ly, { size: 12, font: 'mono', weight: 700, color: C.ink }); V.text(g, l2, 62, ly, { size: 12, color: C.muted }); ly += 20; });
            V.text(g, 'alinhe pelo nome →', 22, ly + 8, { size: 12, color: C.muted });
          }
          const eBox = (x, y, L, a, tag) => {
            const col = colL[L];
            g.globalAlpha = a;
            V.box(g, x, y, bwE, bhE, { r: 8, fill: V.alpha(col, 0.16), stroke: col, lw: 1.5 });
            const s1 = L, s2 = (narrow ? '=' : ' = ') + szTxt(cs.sz[L]);
            const fs = narrow ? 11.5 : 12.5, w1 = tw(g, s1, fs + 1, 'mono', 700), w2 = tw(g, s2, fs, 'mono', 500);
            const bx = x + bwE / 2 - (w1 + w2) / 2, by = y + (tag ? 14 : bhE / 2 + 1);
            V.text(g, s1, bx, by, { size: fs + 1, font: 'mono', weight: 700, color: col, base: 'middle' });
            V.text(g, s2, bx + w1, by, { size: fs, font: 'mono', color: C.ink, base: 'middle' });
            if (tag) V.text(g, tag, x + bwE / 2, y + 28, { size: 9.5, font: 'mono', weight: 600, color: tag === 'Σ somada' ? C.red : C.muted, align: 'center', base: 'middle' });
            g.globalAlpha = 1;
          };
          const tagOf = (L) => ({ soma: 'Σ somada', lote: 'lote', livre: 'livre' })[role(L)];
          // ligações entre letras iguais
          const la = clamp(f / 0.2, 0, 1);
          letters.forEach((L) => {
            const ia = Aa.indexOf(L), ib = Bb.indexOf(L);
            if (ia < 0 || ib < 0) return;
            V.line(g, xs(ia) + bwE / 2, rY[0] + bhE, xs(ib) + bwE / 2, rY[1], { color: V.alpha(colL[L], 0.85 * la), lw: 2, dash: [4, 3] });
          });
          const fade = clamp((f - 0.55) / 0.2, 0, 1), pulse = f > 0.2 && f < 0.6 ? 0.5 + 0.5 * Math.sin(t * 10) : 0;
          [[Aa, rY[0]], [Bb, rY[1]]].forEach(([str, y]) => str.split('').forEach((L, i) => {
            const x = xs(i), soma = role(L) === 'soma';
            if (soma && pulse > 0) V.glow(g, x + bwE / 2, y + bhE / 2, 30, C.red, 0.5 * pulse);
            eBox(x, y, L, soma ? 1 - 0.6 * fade : 1, tagOf(L));
            if (soma && fade > 0) V.line(g, x + 6, y + bhE - 6, x + 6 + (bwE - 12) * fade, y + 6 + (bhE - 12) * (1 - fade), { color: V.alpha(C.red, 0.8), lw: 1.6 });
          }));
          // sobreviventes descem para a saída
          const e = V.ease.inOut(clamp((f - 0.55) / 0.3, 0, 1));
          outS.split('').forEach((L, i) => {
            const xo = xs(i), yo = rY[2];
            g.setLineDash([4, 3]); V.box(g, xo, yo, bwE, bhE, { r: 8, stroke: V.alpha(C.muted, 0.45), lw: 1 }); g.setLineDash([]);
            const ia = Aa.indexOf(L), src = ia >= 0 ? [xs(ia), rY[0]] : [xs(Bb.indexOf(L)), rY[1]];
            if (e > 0) eBox(lerp(src[0], xo, e), lerp(src[1], yo, e), L, 0.35 + 0.65 * e, null);
          });
          // texto: a linha de einsum colorida, papéis, FLOPs
          const xL = narrow ? 16 : 60;
          let yy = rY[2] + bhE + (narrow ? 30 : 34);
          const fsz = narrow ? 12.5 : 15;
          const code = "einsum('" + cs.es + "', " + cs.ops[0] + ', ' + cs.ops[1] + ')';
          const cw = tw(g, 'm', fsz, 'mono', 700);
          const q0 = code.indexOf("'"), q1 = code.lastIndexOf("'");
          for (let i = 0; i < code.length; i++) {
            const ch = code[i], inSpec = i > q0 && i < q1 && colL[ch];
            V.text(g, ch, xL + i * cw, yy, { size: fsz, font: 'mono', weight: 700, color: inSpec ? colL[ch] : C.ink });
          }
          yy += narrow ? 22 : 26;
          const lst = (r) => letters.filter((L) => role(L) === r).join(', ') || '—';
          yy = wrap(g, 'Σ somadas: ' + lst('soma') + ' · lote: ' + lst('lote') + ' · livres: ' + lst('livre'), xL, yy, w - xL - 14, 16, { size: 12, font: 'mono', color: C.ink });
          yy = wrap(g, cs.flops, xL, yy + 2, w - xL - 14, 16, { size: 12, font: 'mono', color: C.d });
          if (S.c === 'e_ds') {
            const tw8 = Math.min(36, (w - 2 * xL - 7 * 6) / 8), ty8 = yy + 6;
            for (let k = 0; k < 8; k++) {
              const col = k < 4 ? C.a : C.b, x = xL + k * (tw8 + 6), a = clamp(f * 8 - k * 0.5, 0, 1);
              g.globalAlpha = 0.3 + 0.7 * a;
              V.box(g, x, ty8, tw8, 24, { r: 5, fill: V.alpha(col, 0.25), stroke: col, lw: 1.2 });
              V.text(g, 'g' + k, x + tw8 / 2, ty8 + 13, { size: 10.5, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
              g.globalAlpha = 1;
            }
            yy = wrap(g, 'cada grupo: [T, 4096] · [4096 → 1024]; GPU 0 (menta) fica com g0–g3, GPU 1 (lavanda) com g4–g7', xL, ty8 + 42, w - xL - 14, 15, { size: 11.5, color: C.muted });
          } else if (S.c === 'e_att') {
            const mc = narrow ? 4 : 5, ty8 = yy + 6;
            for (let k = 0; k < 6; k++) {
              const x = xL + k * (5 * mc + 10), a = clamp(f * 8 - k * 0.5, 0, 1);
              g.globalAlpha = 0.3 + 0.7 * a;
              for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { g.fillStyle = j > i ? V.alpha(C.muted, 0.15) : V.alpha(PAL[1], 0.35 + 0.1 * ((i + j + k) % 4)); g.fillRect(x + j * mc, ty8 + i * mc, mc - 1, mc - 1); }
              g.globalAlpha = 1;
            }
            V.text(g, '… × 64 cabeças: um [T, T] por valor de h', xL + 6 * (5 * mc + 10) + 4, ty8 + 2.5 * mc + 4, { size: 11.5, color: C.muted, maxW: w - xL - 6 * (5 * mc + 10) - 20 });
            yy = ty8 + 5 * mc + 10;
          }
          bottom = yy;
        } else {
          /* ---------------- broadcasting: alinhar pela direita */
          const an = analyze(cs.A, cs.B);
          res.set(an.ok ? '[' + an.out.join(', ') + ']' : 'erro');
          const n = an.cols.length;
          const bw = narrow ? 56 : 70, bh = 28, gx = 8;
          const x0 = narrow ? w - 20 - n * (bw + gx) : Math.max(170, w * 0.5 - (n * (bw + gx)) / 2 + 60);
          const rowY = [narrow ? 22 : 26, narrow ? 58 : 64, narrow ? 102 : 110];
          ['A', 'B', 'saída'].forEach((l, r) => V.text(g, l, x0 - 14, rowY[r] + bh / 2 + 4, { size: 13, font: 'mono', weight: 700, color: r === 2 ? C.d : C.ink, align: 'right' }));
          V.line(g, x0 - 4, rowY[2] - 10, x0 + n * (bw + gx), rowY[2] - 10, { color: V.alpha(C.muted, 0.5), lw: 1 });
          const errAnim = !an.ok && el < 2.2 * 2;
          an.cols.forEach((cl, i) => {
            const x = x0 + i * (bw + gx);
            const colOf = cl.s === 'eq' ? C.a : cl.s === 'err' ? C.red : C.b;
            const wig = cl.s === 'err' && errAnim ? Math.sin(el * 30) * 2.5 * Math.max(0, 1 - ((el % 2.2) / 0.6)) : 0;
            const cell = (v, missing, y, stretched) => {
              if (missing) { g.setLineDash([4, 3]); V.box(g, x + wig, y, bw, bh, { r: 7, stroke: V.alpha(C.muted, 0.6), lw: 1.2 }); g.setLineDash([]); V.text(g, '(1)', x + bw / 2 + wig, y + bh / 2 + 1, { size: 12, font: 'mono', color: C.muted, align: 'center', base: 'middle' }); return; }
              V.box(g, x + wig, y, bw, bh, { r: 7, fill: V.alpha(stretched ? C.b : colOf, 0.18), stroke: stretched ? C.b : colOf, lw: 1.4 });
              V.text(g, String(v), x + bw / 2 + wig, y + bh / 2 + 1, { size: 13, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
              if (stretched) V.text(g, '⇢', x + bw - 9 + wig, y + 9, { size: 11, color: C.b, align: 'center', base: 'middle' });
            };
            cell(cl.a, cl.am, rowY[0], cl.s === 'sa');
            cell(cl.b, cl.bm, rowY[1], cl.s === 'sb');
            V.box(g, x, rowY[2], bw, bh, { r: 7, fill: V.alpha(cl.s === 'err' ? C.red : C.d, 0.18), stroke: cl.s === 'err' ? C.red : C.d, lw: 1.4 });
            V.text(g, String(an.out[i]), x + bw / 2, rowY[2] + bh / 2 + 1, { size: 13, font: 'mono', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          });
          if (!narrow) {
            let ly = 44;
            [[C.a, 'iguais'], [C.b, 'esticado (1 → n)'], [C.red, 'conflito']].forEach(([c, l]) => { V.box(g, 22, ly - 9, 12, 12, { r: 3, fill: c }); V.text(g, l, 42, ly + 1, { size: 12, color: C.muted }); ly += 20; });
            V.text(g, 'alinhe pela direita →', 22, ly + 10, { size: 12, color: C.muted });
          }
          const dy = rowY[2] + bh + (narrow ? 26 : 30);
          const is2D = cs.A.length === 2 && typeof cs.A[0] === 'number';
          if (is2D && an.ok) {
            const R = an.out[0], Kc = an.out[1];
            const Av = vals(cs.A[0], cs.A[1], 3), Bsh = cs.B.length === 1 ? [1, cs.B[0]] : cs.B, Bv = vals(Bsh[0], Bsh[1], 9);
            const demoB = h - 34 - 30;
            const cell = narrow ? clamp((w - 62) / (2 * Kc + 1) - 2, 9, 24) : clamp(Math.min((w - 140) / (Kc * 3 + 2) - 2, (demoB - dy) / R - 2), 9, 26);
            const stp = cell + 2;
            const cyc = 3.2;
            anim = el < cyc * 2;
            const phase = anim ? clamp((el % cyc) / 1.6, 0, 1) : 1;
            const drawCells = (x0g, y0g, Rr, Kk, get, ghost) => {
              for (let i = 0; i < Rr; i++) for (let j = 0; j < Kk; j++) {
                const x = x0g + j * stp, y = y0g + i * stp, v = get(i, j);
                if (ghost) { g.setLineDash([3, 3]); V.box(g, x, y, cell, cell, { r: 3, stroke: V.alpha(C.b, 0.7), lw: 1 }); g.setLineDash([]); g.globalAlpha = 0.45; }
                V.box(g, x, y, cell, cell, { r: 3, fill: V.cmap(v / 3, 'div') });
                if (cell >= 17) V.text(g, String(v).replace('-', '−'), x + cell / 2, y + cell / 2 + 1, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
                g.globalAlpha = 1;
              }
            };
            const Ax = narrow ? 20 : 40;
            const aR = cs.A[0], aK = cs.A[1];
            drawCells(Ax, dy, aR, aK, (i, j) => Av[i][j], false);
            if (aK < Kc) { const reps = Math.floor(phase * (Kc - 1) + 0.001); for (let r = 1; r <= reps; r++) drawCells(Ax + r * stp, dy, aR, 1, (i) => Av[i][0], true); }
            const Bx = Ax + Kc * stp + (narrow ? 26 : 30);
            V.text(g, cs.op, Bx - (narrow ? 13 : 15), dy + (R * stp) / 2 + 5, { size: 18, weight: 700, color: C.ink, align: 'center' });
            drawCells(Bx, dy, Bsh[0], Bsh[1], (i, j) => Bv[i][j], false);
            if (Bsh[0] < R) { const reps = Math.floor(phase * (R - 1) + 0.001); for (let r = 1; r <= reps; r++) drawCells(Bx, dy + r * stp, 1, Bsh[1], (i, j) => Bv[0][j], true); }
            if (Bsh[1] < Kc) { const reps = Math.floor(phase * (Kc - 1) + 0.001); for (let r = 1; r <= reps; r++) drawCells(Bx + r * stp, dy, Bsh[0], 1, (i) => Bv[i][0], true); }
            // resultado: ao lado (largo) ou embaixo de B (estreito)
            const Rx = narrow ? Bx : Bx + Kc * stp + 30, Ry = narrow ? dy + R * stp + 24 : dy;
            V.text(g, '=', Rx - (narrow ? 13 : 15), Ry + (R * stp) / 2 + 5, { size: 18, weight: 700, color: C.ink, align: 'center' });
            if (Rx + Kc * stp < w - 4) {
              g.globalAlpha = phase >= 1 ? 1 : 0.25;
              drawCells(Rx, Ry, R, Kc, (i, j) => { const a = Av[Math.min(i, aR - 1)][Math.min(j, aK - 1)], b = Bv[Math.min(i, Bsh[0] - 1)][Math.min(j, Bsh[1] - 1)]; return cs.op === '+' ? a + b : a * b; }, false);
              g.globalAlpha = 1;
            }
            const sy = Ry + R * stp + 16;
            V.text(g, 'tracejado = cópia virtual (stride 0), não existe na memória', Ax, narrow ? sy : Math.min(sy, h - 50), { size: 11.5, color: C.b, maxW: w - Ax - 10 });
            bottom = sy + 4;
          } else if (S.c === 'mhc') {
            const sx = narrow ? 24 : 60, bwid = Math.min(w - sx - 130, 420), hh = 12;
            anim = el < 3 * 2;
            V.text(g, 'post_mix [T, 4, 1]', sx, dy + 4, { size: 11.5, font: 'mono', color: C.muted });
            for (let k = 0; k < 4; k++) V.box(g, sx, dy + 14 + k * (hh + 6), 16, hh, { r: 3, fill: V.alpha(C.series[k], 0.8) });
            V.text(g, 'y [T, 1, 4096]', sx + 170, dy + 4, { size: 11.5, font: 'mono', color: C.muted });
            V.box(g, sx + 170, dy + 14, Math.max(60, bwid * 0.55 - 110), hh, { r: 3, fill: V.alpha(C.d, 0.7) });
            const f = anim ? clamp((el % 3) / 1.4, 0, 1) : 1;
            const ry = dy + 14 + 4 * (hh + 6) + 18;
            V.text(g, 'saída [T, 4, 4096]: fluxo i ← post_mixᵢ · y', sx, ry, { size: 11.5, font: 'mono', color: C.ink, maxW: w - sx - 10 });
            for (let k = 0; k < 4; k++) { V.box(g, sx, ry + 10 + k * (hh + 6), 16, hh, { r: 3, fill: V.alpha(C.series[k], 0.8) }); g.globalAlpha = 0.2 + 0.8 * f; V.box(g, sx + 24, ry + 10 + k * (hh + 6), bwid * 0.55 * f, hh, { r: 3, fill: V.alpha(C.series[k], 0.35 + 0.15 * k) }); g.globalAlpha = 1; }
            bottom = ry + 10 + 4 * (hh + 6);
          } else {
            const sx = narrow ? 20 : 60;
            const ok = an.ok;
            anim = errAnim;
            V.text(g, ok ? '✓ funciona: [T, 64, 128]' : '✕ erro: “The size of tensor a (128) must match the size of tensor b (64)”', sx, dy + 10, { size: narrow ? 11.5 : 13, font: 'mono', weight: 700, color: ok ? C.a : C.red, maxW: w - sx - 10 });
            const cols = 16, rows = 8, cc = narrow ? 8 : 11;
            for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { g.fillStyle = ok ? V.alpha(C.series[i % 6], 0.2 + 0.05 * ((j + i) % 4)) : V.alpha(C.muted, 0.15); V.rr(g, sx + j * (cc + 1), dy + 26 + i * (cc + 1), cc, cc, 2); g.fill(); }
            V.text(g, '8 de 64 cabeças × 16 de 128 canais', sx, dy + 26 + rows * (cc + 1) + 14, { size: 11, color: C.muted });
            if (ok) { for (let i = 0; i < rows; i++) V.box(g, sx + cols * (cc + 1) + 8, dy + 26 + i * (cc + 1), cc, cc, { r: 2, fill: V.alpha(C.series[i % 6], 0.9) }); V.text(g, '← A[64, 1]: uma taxa por cabeça', sx + cols * (cc + 1) + cc + 14, dy + 26 + (rows * (cc + 1)) / 2 + 4, { size: 11, color: C.muted, maxW: w - sx - cols * (cc + 1) - 30 }); }
            bottom = dy + 26 + rows * (cc + 1) + 18;
          }
        }
        // nota: logo abaixo do conteúdo no celular; ancorada embaixo no desktop
        const noteY = narrow ? bottom + 18 : Math.max(bottom + 18, h - 34);
        wrap(g, cs.note + '  Receita: ' + cs.recipe + '.', narrow ? 16 : 22, noteY, w - 40, 15, { size: 11.5, color: C.muted });
        return anim;
      }, { onEnter: () => { if (H) S.t0 = H.clock(); } });
    })();

    /* ================================================== Fig 6: fita de memória */
    (function figFita() {
      const fig = root.querySelector('#fig-fita');
      const MODES = {
        contig: { lab: 'contígua', shape: [3, 4], strides: [4, 1], off: 0, txt: 'x = torch.arange(12).view(3, 4)' },
        transp: { lab: 'transposta', shape: [4, 3], strides: [1, 4], off: 0, txt: 'x.t()   → sem cópia, strides trocados' },
        slice: { lab: 'fatia', shape: [3, 2], strides: [4, 1], off: 1, txt: 'x[:, 1:3]   → offset 1' },
        expand: { lab: 'expand', shape: [3, 4], strides: [0, 1], off: 0, txt: 'x[0].expand(3, 4)   → stride 0' },
        kpool: { lab: 'página kpool (GLM)' },
      };
      const S = { m: 'contig', bug: true, b: 58, rain: -1, t: 0 };
      const applyH = stageH(fig, 460, () => (S.m === 'kpool' ? 600 : S.m === 'transp' ? 430 : 390));
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modo', options: Object.keys(MODES).map((k) => ({ v: k, label: MODES[k].lab })), value: 'contig', onChange: (v) => { S.m = v; S.rain = -1; kp.forEach((e) => show(e, v === 'kpool')); applyH(); H.redraw(); } });
      const tgl = UI.toggle(ctl, { label: 'kernel com stride denso (o bug)', value: true, onChange: (v) => { S.bug = v; S.rain = -1; H.redraw(); } });
      const sB = UI.slider(ctl, { label: 'bloco b', min: 0, max: 551, step: 1, value: 58, fmt: (v) => String(v), onInput: (v) => { S.b = v; H.redraw(); } });
      const btn = UI.button(ctl, '▶ 552 cartas', () => { S.rain = S.t; H.redraw(); });
      const kp = [tgl.el, sB.el, btn];
      kp.forEach((e) => show(e, false));
      const PAGE = 118272, DENSE = 2048, NBLK = 552, TAILB = 2048;
      const EXPL = { contig: 'linha a linha, de 1 em 1: ler na ordem lógica é ler a fita em sequência.', transp: 'a mesma fita lida em outra ordem: cada passo lógico pula 4 posições. Nenhum byte foi copiado.', slice: 'a fatia começa no offset 1 e pula 4 a cada linha: sobra um buraco entre as linhas.', expand: 'stride 0 na primeira dimensão: as 3 “linhas” são os mesmos 4 números. É o broadcasting por dentro.' };
      H = liveFig(fig, (g, w, h, t) => {
        const C = V.C(); S.t = t;
        const narrow = w < 640;
        if (S.m !== 'kpool') {
          const md = MODES[S.m];
          const [R, K] = md.shape, [s0, s1] = md.strides;
          const n = R * K;
          const cell = narrow ? 30 : 40;
          const lx = narrow ? 16 : Math.max(40, w * 0.18), ly = narrow ? 46 : 54;
          V.text(g, md.txt, lx, 26, { size: narrow ? 12 : 13.5, font: 'mono', weight: 600, color: C.ink, maxW: w - lx - 10 });
          const idx = Math.floor(t / 0.6) % (n + 2);
          const cur = Math.min(idx, n - 1);
          const ci = Math.floor(cur / K), cj = cur % K;
          const tensW = K * (cell + 4) - 4, tensH = R * (cell + 4) - 4;
          // metadados: sempre à direita do tensor (a seta nunca passa por cima deles)
          const mxp = lx + tensW + (narrow ? 16 : 40), myp = ly + (narrow ? 10 : 6), lh = narrow ? 18 : 22, vx = mxp + (narrow ? 80 : 90);
          const contig = s1 === 1 && s0 === K && md.off === 0;
          const lines = [['shape', '[' + R + ', ' + K + ']'], ['stride', '(' + s0 + ', ' + s1 + ')'], ['offset', String(md.off)], ['contíguo?', contig ? 'sim' : 'não']];
          lines.forEach(([k, v], i) => {
            const yy = myp + i * lh;
            V.text(g, k, mxp, yy, { size: 12, font: 'mono', color: C.muted });
            V.text(g, v, vx, yy, { size: 12.5, font: 'mono', weight: 700, color: k === 'contíguo?' ? (contig ? C.a : C.c) : C.ink });
          });
          // coluna da direita: metadados, (desktop) fórmula e explicação, e a legenda do tensor
          let rb = myp + 4 * lh;
          if (!narrow) {
            V.text(g, 'endereço(i, j) = ' + md.off + ' + i·' + s0 + ' + j·' + s1, mxp, rb + 4, { size: 12, font: 'mono', color: C.d });
            rb = wrap(g, EXPL[S.m], mxp, rb + 26, w - mxp - 20, 16, { size: 12, color: C.muted });
          }
          rb = wrap(g, 'tensor lógico [' + R + ', ' + K + '] · valor = posição na fita', mxp, rb + (narrow ? 6 : 4), w - mxp - (narrow ? 8 : 20), 14, { size: 11, color: C.muted });
          // fita
          const tcell = Math.min(narrow ? 26 : 46, (w - 40) / 12 - 3);
          const tx = (w - 12 * (tcell + 3)) / 2;
          const yb = Math.max(ly + tensH + 12, rb + 2);
          const ty = narrow ? yb + 30 : Math.max(h - 128, yb + 36);
          // seta primeiro (passa por baixo das células): desce reto pela coluna do tensor e só então vai à fita,
          // sem cruzar nenhum texto
          if (idx < n) {
            const a0 = md.off + ci * s0 + cj * s1;
            const fx = lx + cj * (cell + 4) + cell / 2, fy = ly + ci * (cell + 4) + cell / 2;
            const tx2 = tx + a0 * (tcell + 3) + tcell / 2;
            V.line(g, fx, fy, fx, yb, { color: V.alpha(C.c, 0.9), lw: 2 });
            V.arrow(g, fx, yb, tx2, ty - 4, { color: V.alpha(C.c, 0.9), lw: 2, head: 9, curve: narrow ? 12 : 18 });
          }
          // tensor lógico
          for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
            const a0 = md.off + i * s0 + j * s1;
            const on = i === ci && j === cj && idx < n;
            const done = i * K + j < cur || idx >= n;
            const bx0 = lx + j * (cell + 4), by0 = ly + i * (cell + 4);
            V.box(g, bx0, by0, cell, cell, { r: 7, fill: C.bg || (C.dark ? '#0b1a13' : '#ffffff') });
            V.box(g, bx0, by0, cell, cell, { r: 7, fill: V.alpha(on ? C.c : C.a, on ? 0.45 : done ? 0.22 : 0.1), stroke: on ? C.c : V.alpha(C.a, 0.6), lw: on ? 2.2 : 1.2 });
            V.text(g, String(a0), bx0 + cell / 2, by0 + cell / 2 + 1, { size: 13, font: 'mono', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          }
          // o título da fita fica embaixo dela, para a seta nunca cruzar texto
          V.text(g, 'FITA DE MEMÓRIA (12 elementos BF16 = 24 bytes)', tx, ty + tcell + 32, { size: narrow ? 10.5 : 11, font: 'mono', weight: 600, color: C.muted, maxW: w - tx - 8 });
          const used = new Set();
          for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) used.add(md.off + i * s0 + j * s1);
          for (let a0 = 0; a0 < 12; a0++) {
            const x = tx + a0 * (tcell + 3);
            const isCur = idx < n && a0 === md.off + ci * s0 + cj * s1;
            V.box(g, x, ty, tcell, tcell, { r: 5, fill: V.alpha(used.has(a0) ? C.b : C.muted, isCur ? 0.5 : used.has(a0) ? 0.2 : 0.08), stroke: isCur ? C.c : V.alpha(C.b, used.has(a0) ? 0.7 : 0.25), lw: isCur ? 2.2 : 1 });
            V.text(g, String(a0), x + tcell / 2, ty + tcell / 2 + 1, { size: 12, font: 'mono', color: used.has(a0) ? C.ink : C.muted, align: 'center', base: 'middle' });
            V.text(g, '+' + a0 * 2 + 'B', x + tcell / 2, ty + tcell + 13, { size: 9.5, font: 'mono', color: C.muted, align: 'center' });
          }
          // trilha
          const trail = [];
          for (let q = 0; q <= Math.min(cur, n - 1); q++) { const i = Math.floor(q / K), j = q % K; trail.push(md.off + i * s0 + j * s1); }
          let yy = wrap(g, 'ordem lógica → endereços: ' + trail.join(' → '), tx, ty + tcell + 54, w - tx - 10, 16, { size: 11.5, font: 'mono', color: C.ink });
          if (narrow) {
            V.text(g, 'endereço(i, j) = ' + md.off + ' + i·' + s0 + ' + j·' + s1, tx, yy + 6, { size: 12, font: 'mono', color: C.d, maxW: w - tx - 10 });
            wrap(g, EXPL[S.m], tx, yy + 26, w - 2 * tx, 16, { size: 12, color: C.muted });
          }
          return true;
        }
        // ---------------- modo kpool
        const b = S.b;
        const own = b * PAGE, dense = b * DENSE;
        const tgt = S.bug ? dense : own;
        const pg = Math.floor(tgt / PAGE), offIn = tgt % PAGE;
        const x0 = narrow ? 14 : 30, x1 = w - (narrow ? 14 : 30);
        const bad = S.bug ? C.red : C.a;
        if (narrow) wrap(g, 'cauda do kpool: view as_strided [552, 2, 4, 128] BF16 dentro das páginas do indexador', x0, 18, x1 - x0, 14, { size: 11, font: 'mono', weight: 600, color: C.ink });
        else V.text(g, 'cauda do kpool: view as_strided [552, 2, 4, 128] BF16 dentro das páginas do indexador', x0, 22, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: x1 - x0 });
        V.text(g, 'bug:  base = b × 1024 elem = b × 2.048 B', x0, narrow ? 52 : 44, { size: narrow ? 11 : 12.5, font: 'mono', color: S.bug ? C.red : C.muted, weight: S.bug ? 700 : 400 });
        V.text(g, 'fix:  base = b × tail.stride(0) = b × 118.272 B', x0 + (narrow ? 0 : Math.min(420, (x1 - x0) / 2)), narrow ? 70 : 44, { size: narrow ? 11 : 12.5, font: 'mono', color: S.bug ? C.muted : C.a, weight: S.bug ? 400 : 700 });
        // zoom: páginas 0..Z-1
        const Z = narrow ? 8 : 12;
        const zy = narrow ? 112 : 94, zh = 50;
        const pw = (x1 - x0) / Z;
        const tailW = Math.max(6, pw * (TAILB / PAGE));
        V.text(g, narrow ? 'MEMÓRIA · págs 0–' + (Z - 1) + ' · verde = cauda (fora de escala)' : 'MEMÓRIA · páginas 0–' + (Z - 1) + ' de 118.272 B · verde = cauda (2 KiB no início, fora de escala)', x0, zy - 10, { size: narrow ? 10 : 11, font: 'mono', color: C.muted });
        const xAt = (byte) => { const p = Math.floor(byte / PAGE), o = byte % PAGE; const bx = x0 + p * pw; return o < TAILB ? bx + (o / TAILB) * tailW : bx + tailW + ((o - TAILB) / (PAGE - TAILB)) * (pw - tailW); };
        for (let p = 0; p < Z; p++) {
          const bx = x0 + p * pw;
          V.box(g, bx + 1, zy, pw - 2, zh, { r: 5, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.4), lw: 1 });
          g.fillStyle = V.alpha(C.a, 0.55); g.fillRect(bx + 1, zy, tailW, zh);
          V.text(g, 'pág ' + p, bx + pw / 2 + tailW / 2, zy + zh / 2 + 4, { size: narrow ? 9.5 : 10.5, font: 'mono', color: C.muted, align: 'center' });
        }
        // veredito
        let verdict;
        if (!S.bug) verdict = 'correto: a cauda do bloco ' + b + ' fica no início da página ' + b + '.';
        else if (b === 0) verdict = 'bloco 0: por sorte, 0 × 2 KiB = início da página 0 (o único que acerta).';
        else if (offIn < TAILB) verdict = 'bug: cai em ' + nf(tgt) + ' B → página ' + pg + ', dentro da CAUDA do bloco ' + pg + ' (dados de outra requisição).';
        else verdict = 'bug: cai em ' + nf(tgt) + ' B → página ' + pg + ', no meio das linhas de pool do indexador do bloco ' + pg + '.';
        const vy = wrap(g, verdict, x0, zy + zh + 24, x1 - x0, 16, { size: 12.5, weight: 600, color: S.bug && b > 0 ? C.red : C.a });
        // tensor lógico: 552 caudas
        const GC = narrow ? 24 : 48, GR = Math.ceil(NBLK / GC);
        const gy = vy + (narrow ? 22 : 20), cw = (x1 - x0) / GC, chh = narrow ? 7 : 8;
        V.text(g, narrow ? 'TENSOR LÓGICO · 552 caudas (b = 0 … 551)' : 'TENSOR LÓGICO · 552 caudas, uma por bloco de KV (b = 0 … 551)', x0, gy - 6, { size: narrow ? 10 : 11, font: 'mono', color: C.muted });
        const chip = (k) => [x0 + (k % GC) * cw, gy + Math.floor(k / GC) * (chh + 2)];
        for (let k = 0; k < NBLK; k++) { const [cx0, cy0] = chip(k); g.fillStyle = k === b ? C.c : V.alpha(C.b, 0.28); V.rr(g, cx0 + 0.5, cy0, cw - 1, chh, 1.5); g.fill(); }
        const [bcx, bcy] = chip(b);
        // visão geral: 552 páginas
        const oy = h - (narrow ? 76 : 66), oh = 16;
        V.text(g, 'MEMÓRIA · todas as 552 páginas', x0, oy - 10, { size: 11, font: 'mono', color: C.muted });
        V.box(g, x0, oy, x1 - x0, oh, { r: 4, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.4), lw: 1 });
        const ox = (p) => x0 + ((p + 0.5) / NBLK) * (x1 - x0);
        for (let p = 0; p <= NBLK; p += 50) { V.line(g, x0 + (p / NBLK) * (x1 - x0), oy + oh, x0 + (p / NBLK) * (x1 - x0), oy + oh + 4, { color: C.muted, lw: 1 }); V.text(g, String(p), x0 + (p / NBLK) * (x1 - x0), oy + oh + 15, { size: 9.5, font: 'mono', color: C.muted, align: 'center' }); }
        g.fillStyle = V.alpha(C.red, 0.35); g.fillRect(x0, oy - 3, Math.max(3, (10 / NBLK) * (x1 - x0)), oh + 6);
        V.line(g, ox(b), oy - 6, ox(b), oy + oh + 6, { color: C.a, lw: 2 });
        V.line(g, ox(pg), oy - 6, ox(pg), oy + oh + 6, { color: bad, lw: 2.5 });
        // seta do bloco escolhido ao destino
        const inZoom = pg < Z;
        const dX = inZoom ? xAt(tgt) : ox(pg), dY = inZoom ? zy + zh + 2 : oy - 2;
        V.arrow(g, bcx + cw / 2, inZoom ? bcy - 2 : bcy + chh + 2, dX, dY, { color: bad, lw: 2, head: 9, curve: inZoom ? 26 : -20 });
        if (inZoom) { V.glow(g, dX, zy + zh / 2, 16, bad, 0.9); V.box(g, dX - 5, zy + 6, 10, 14, { r: 2, fill: bad }); }
        if (S.bug && b < Z && b > 0) { const ownX = xAt(own); g.setLineDash([4, 3]); V.box(g, ownX - 6, zy - 3, 12, zh + 6, { r: 3, stroke: C.a, lw: 1.5 }); g.setLineDash([]); }
        pill(g, Math.min(Math.max(bcx + cw / 2, x0 + 40), x1 - 40), gy + GR * (chh + 2) + 12, 'bloco ' + b + ' (rosa)', C.c, { size: 10.5, align: 'center' });
        // chuva de 552 cartas
        if (S.rain >= 0) {
          const el = t - S.rain, dur = 3.2;
          const hit = new Set();
          for (let k = 0; k < NBLK; k++) {
            const f = clamp((el - (k / NBLK) * 1.6) / 1.2, 0, 1);
            if (f <= 0) continue;
            const tb = S.bug ? k * DENSE : k * PAGE;
            const p = Math.floor(tb / PAGE);
            hit.add(p);
            let tx, ty;
            if (p < Z) { tx = xAt(tb); ty = zy + 8 + (k % 5) * 8; } else { tx = ox(p); ty = oy + oh / 2; }
            const [sx0, sy0] = chip(k);
            const e = V.ease.inOut(f);
            V.dot(g, lerp(sx0 + cw / 2, tx, e), lerp(sy0 + chh / 2, ty, e), 2, V.alpha(bad, 0.8));
          }
          if (el > 1.6) pill(g, x1, narrow ? oy + oh + 34 : oy - 22, 'páginas atingidas: ' + hit.size + ' de 552', bad, { size: 11, align: 'right' });
          return el < dur + 0.3;
        }
        return false;
      });
    })();

    /* ================================================== Fig 7: SVD */
    (function figSvd() {
      const fig = root.querySelector('#fig-svd');
      stageH(fig, 440, 660);
      const n = 24;
      function face() {
        const M = [];
        for (let i = 0; i < n; i++) {
          const row = [];
          for (let j = 0; j < n; j++) {
            let v = -0.12 + 0.08 * (j / (n - 1) - 0.5);
            const r = Math.hypot(i - 11.5, j - 11.5);
            if (Math.abs(r - 10) < 1.15) v = 1;
            if (Math.hypot(i - 8.5, j - 8) < 1.9 || Math.hypot(i - 8.5, j - 15) < 1.9) v = 1;
            const rm = Math.hypot(i - 10.5, j - 11.5);
            if (i > 13 && Math.abs(rm - 6) < 0.95) v = 0.9;
            if (Math.hypot(i - 14, j - 5.5) < 1.4 || Math.hypot(i - 14, j - 17.5) < 1.4) v = -0.7;
            row.push(v);
          }
          M.push(row);
        }
        return M;
      }
      function noise() { const r = V.rng(77); return Array.from({ length: n }, () => Array.from({ length: n }, () => clamp(r.normal() * 0.45, -1, 1))); }
      function svd(W) {
        const m = W.length, k = W[0].length;
        const A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => { let s = 0; for (let r = 0; r < m; r++) s += W[r][i] * W[r][j]; return s; }));
        const Vm = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)));
        for (let sw = 0; sw < 80; sw++) {
          let off = 0;
          for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) off += A[p][q] * A[p][q];
          if (off < 1e-20) break;
          for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) {
            const apq = A[p][q];
            if (Math.abs(apq) < 1e-14) continue;
            const th = (A[q][q] - A[p][p]) / (2 * apq);
            const tt = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
            const c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
            for (let r = 0; r < k; r++) { const ap = A[r][p], aq = A[r][q]; A[r][p] = c * ap - s * aq; A[r][q] = s * ap + c * aq; }
            for (let r = 0; r < k; r++) { const ap = A[p][r], aq = A[q][r]; A[p][r] = c * ap - s * aq; A[q][r] = s * ap + c * aq; }
            for (let r = 0; r < k; r++) { const vp = Vm[r][p], vq = Vm[r][q]; Vm[r][p] = c * vp - s * vq; Vm[r][q] = s * vp + c * vq; }
          }
        }
        const idx = Array.from({ length: k }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
        const sig = idx.map((i) => Math.sqrt(Math.max(0, A[i][i])));
        const Vc = idx.map((i) => Vm.map((row) => row[i]));
        const Uc = Vc.map((v, t) => { const u = W.map((row) => row.reduce((s, x, j) => s + x * v[j], 0)); return sig[t] > 1e-9 ? u.map((x) => x / sig[t]) : u.map(() => 0); });
        const recon = [Array.from({ length: m }, () => new Array(k).fill(0))];
        for (let t = 0; t < k; t++) { const prev = recon[t]; recon.push(prev.map((row, i) => row.map((x, j) => x + sig[t] * Uc[t][i] * Vc[t][j]))); }
        const tot = sig.reduce((s, x) => s + x * x, 0);
        return { sig, Uc, Vc, recon, tot };
      }
      const DATA = { face: face(), noise: noise() };
      const SV = { face: svd(DATA.face), noise: svd(DATA.noise) };
      const S = { r: 6, m: 'face' };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'posto r', min: 1, max: 24, step: 1, value: 6, fmt: (v) => String(v), onInput: (v) => { S.r = v; H.redraw(); } });
      UI.seg(ctl, { label: 'Matriz', options: [{ v: 'face', label: 'rosto (estruturada)' }, { v: 'noise', label: 'ruído (aleatória)' }], value: 'face', onChange: (v) => { S.m = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const W = DATA[S.m], sv = SV[S.m], r = S.r;
        const R = sv.recon[r];
        const A = Array.from({ length: n }, (_, i) => Array.from({ length: r }, (_, t) => sv.Uc[t][i] * sv.sig[t]));
        const B = Array.from({ length: r }, (_, t) => sv.Vc[t].slice());
        const aMax = Math.max(1e-6, ...A.flat().map(Math.abs)), bMax = Math.max(1e-6, ...B.flat().map(Math.abs));
        const narrow = w < 640;
        const opt = { min: -1, max: 1, cmap: 'div', gap: 1 };
        let c, xs;
        const sym = (s, x, y) => V.text(g, s, x, y, { size: 20, weight: 700, color: C.ink, align: 'center', base: 'middle' });
        if (!narrow) {
          c = clamp((w - 110) / (72 + 24), 4, 8);
          const st = c + 1, blk = n * st;
          const tot = 3 * blk + r * st + 90;
          let x = Math.max(16, (w - tot) / 2);
          const y = 44;
          V.matrix(g, Object.assign({ x, y, data: W, cell: c }, opt)); V.text(g, 'W [24 × 24]', x + blk / 2, y - 10, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
          x += blk; sym('≈', x + 15, y + blk / 2); x += 30;
          V.matrix(g, { x, y, data: A, cell: c, gap: 1, cmap: 'div', min: -aMax, max: aMax }); V.text(g, 'A [24 × ' + r + ']', x + (r * st) / 2, y - 10, { size: 11.5, font: 'mono', color: C.a, align: 'center' });
          x += r * st; sym('·', x + 15, y + blk / 2); x += 30;
          V.matrix(g, { x, y, data: B, cell: c, gap: 1, cmap: 'div', min: -bMax, max: bMax }); V.text(g, 'B [' + r + ' × 24]', x + blk / 2, y - 10, { size: 11.5, font: 'mono', color: C.b, align: 'center' });
          x += blk; sym('=', x + 15, y + blk / 2); x += 30;
          V.matrix(g, Object.assign({ x, y, data: R, cell: c }, opt)); V.text(g, 'AB (posto ' + r + ')', x + blk / 2, y - 10, { size: 11.5, font: 'mono', color: C.d, align: 'center' });
          xs = { specY: y + blk + 40 };
        } else {
          c = clamp((w - 70) / 48 - 1, 3.5, 7);
          const st = c + 1, blk = n * st;
          let x = (w - 2 * blk - 30) / 2; const y = 30;
          V.matrix(g, Object.assign({ x, y, data: W, cell: c }, opt)); V.text(g, 'W', x + blk / 2, y - 10, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
          sym('≈', x + blk + 15, y + blk / 2);
          V.matrix(g, Object.assign({ x: x + blk + 30, y, data: R, cell: c }, opt)); V.text(g, 'AB (posto ' + r + ')', x + blk + 30 + blk / 2, y - 10, { size: 11.5, font: 'mono', color: C.d, align: 'center' });
          const y2 = y + blk + 34;
          const c2 = clamp((w - 60) / (24 + r + 2) - 1, 2.5, 6), st2 = c2 + 1;
          const x2 = (w - (r + 24) * st2 - 30) / 2;
          V.matrix(g, { x: x2, y: y2, data: A, cell: c2, gap: 1, cmap: 'div', min: -aMax, max: aMax }); V.text(g, 'A', x2 + (r * st2) / 2, y2 - 8, { size: 11, font: 'mono', color: C.a, align: 'center' });
          sym('·', x2 + r * st2 + 15, y2 + (n * st2) / 2);
          V.matrix(g, { x: x2 + r * st2 + 30, y: y2, data: B, cell: c2, gap: 1, cmap: 'div', min: -bMax, max: bMax }); V.text(g, 'B', x2 + r * st2 + 30 + (24 * st2) / 2, y2 - 8, { size: 11, font: 'mono', color: C.b, align: 'center' });
          xs = { specY: y2 + n * st2 + 34 };
        }
        // espectro
        const sx = narrow ? 30 : 60, sw = w - sx - (narrow ? 20 : 60), sh = Math.max(50, h - xs.specY - (narrow ? 70 : 64));
        const smax = sv.sig[0] || 1;
        V.text(g, 'VALORES SINGULARES σ₁ ≥ σ₂ ≥ …', sx, xs.specY - 8, { size: 11, font: 'mono', color: C.muted });
        const bwid = sw / n;
        for (let i = 0; i < n; i++) {
          const bh = (sv.sig[i] / smax) * sh;
          g.fillStyle = i < r ? C.a : V.alpha(C.muted, 0.35);
          V.rr(g, sx + i * bwid + 1, xs.specY + sh - bh, bwid - 2, Math.max(1, bh), 2); g.fill();
        }
        V.line(g, sx, xs.specY + sh, sx + sw, xs.specY + sh, { color: C.muted, lw: 1 });
        const kept = sv.sig.slice(0, r).reduce((s, x) => s + x * x, 0) / (sv.tot || 1);
        const err = Math.sqrt(Math.max(0, 1 - kept));
        const ry = xs.specY + sh + 22;
        const t1 = 'parâmetros r(m+n) = ' + r + '·48 = ' + r * 48 + '  vs  mn = 576 (' + nf((100 * r * 48) / 576, 0) + '%)';
        const t2 = 'energia guardada ' + nf(kept * 100, 1) + '%  ·  erro relativo ' + nf(err * 100, 1) + '%';
        V.text(g, t1, sx, ry, { size: 12, font: 'mono', color: r * 48 >= 576 ? C.c : C.ink, maxW: sw });
        V.text(g, t2, sx, ry + 18, { size: 12, font: 'mono', color: C.muted, maxW: sw });
        if (r * 48 >= 576 && !narrow) V.text(g, 'r ≥ 12: fatorar já não economiza nada', sx + sw, ry, { size: 11.5, color: C.c, align: 'right' });
      }, { static: true });
    })();

    /* ================================================== Fig 8: ortogonais */
    (function figOrto() {
      const fig = root.querySelector('#fig-orto');
      stageH(fig, 430, 680);
      const S = { m: 'rot', th: 40, al: 3, ph: 30, ang: 0, cur: [1, 0, 0, 1] };
      let H = null;
      const rd = () => { if (H) H.redraw(); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Matriz', options: [{ v: 'rot', label: 'rotação' }, { v: 'refl', label: 'reflexão' }, { v: 'hada', label: 'Hadamard' }, { v: 'proj', label: 'projeção α' }], value: 'rot', onChange: (v) => { S.m = v; vis(); rd(); } });
      const sTh = UI.slider(ctl, { label: 'θ', min: -180, max: 180, step: 1, value: 40, fmt: (v) => v + '°', onInput: (v) => { S.th = v; rd(); } });
      const sAl = UI.slider(ctl, { label: 'α', min: 0, max: 3, step: 0.05, value: 3, fmt: (v) => nf(v, 2), onInput: (v) => { S.al = v; rd(); } });
      const sPh = UI.slider(ctl, { label: 'direção de r (normal n)', min: 0, max: 180, step: 1, value: 30, fmt: (v) => v + '°', onInput: (v) => { S.ph = v; rd(); } });
      function vis() { show(sTh.el, S.m === 'rot'); show(sAl.el, S.m === 'proj'); show(sPh.el, S.m === 'proj' || S.m === 'refl'); }
      vis();
      const pts = [];
      for (let k = 0; k < 28; k++) { const a = (k / 28) * Math.PI * 2; pts.push([0.35 + Math.cos(a) * 0.9, 0.25 + Math.sin(a) * 0.9, 0]); }
      pts.push([0.05, 0.55, 1], [0.65, 0.55, 1]);
      for (let k = 0; k <= 8; k++) { const a = Math.PI * (1.15 + (0.7 * k) / 8); pts.push([0.35 + Math.cos(a) * 0.5, 0.25 + Math.sin(a) * 0.5, 2]); }
      const U = [1.35, 0.35], Vv = [0.3, 1.15];
      const HX = [0.25, -0.3, 5.2, 0.1, -0.2, 0.35, -0.1, 0.15], HY = LA.num.fwht(HX);
      function target() {
        const r = (S.ph * Math.PI) / 180, rc = Math.cos(r), rs = Math.sin(r);
        if (S.m === 'rot') { const a = (S.th * Math.PI) / 180; return [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)]; }
        if (S.m === 'refl') return [1 - 2 * rc * rc, -2 * rc * rs, -2 * rc * rs, 1 - 2 * rs * rs];
        if (S.m === 'hada') { const s = 1 / Math.SQRT2; return [s, s, s, -s]; }
        return [1 - S.al * rc * rc, -S.al * rc * rs, -S.al * rc * rs, 1 - S.al * rs * rs];
      }
      H = liveFig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const kf = Math.min(1, dt * 6 + 0.01);
        const prev = S.cur.slice();
        if (S.m === 'rot') { S.ang += (S.th - S.ang) * kf; const an = (S.ang * Math.PI) / 180; const R = [Math.cos(an), -Math.sin(an), Math.sin(an), Math.cos(an)]; const far = Math.abs(S.cur[0] * S.cur[3] - S.cur[1] * S.cur[2] - 1) > 0.02 || Math.abs(S.cur[0] - S.cur[3]) > 0.02 || Math.abs(S.cur[1] + S.cur[2]) > 0.02; if (far) for (let k = 0; k < 4; k++) S.cur[k] += (R[k] - S.cur[k]) * kf; else S.cur = R; }
        else { const tg = target(); for (let k = 0; k < 4; k++) S.cur[k] += (tg[k] - S.cur[k]) * kf; S.ang = (Math.atan2(S.cur[2], S.cur[0]) * 180) / Math.PI; }
        const [a, b, c, d] = S.cur;
        const narrow = w < 640;
        const PL = narrow ? { x: 0, y: 0, w, h: h * 0.56 } : { x: 0, y: 0, w: w * 0.56, h };
        const PR = narrow ? { x: 12, y: h * 0.56 + 4, w: w - 24, h: h * 0.44 - 12 } : { x: w * 0.56 + 4, y: 14, w: w * 0.44 - 18, h: h - 28 };
        const cx = PL.x + PL.w / 2, cy = PL.y + PL.h / 2, u = Math.min(PL.w, PL.h) / 5;
        const Tf = (x, y) => [cx + (a * x + b * y) * u, cy - (c * x + d * y) * u];
        const O = (x, y) => [cx + x * u, cy - y * u];
        clipRect(g, PL);
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let k = -6; k <= 6; k++) { const x = cx + k * u * 0.5, y = cy + k * u * 0.5; g.moveTo(x, PL.y); g.lineTo(x, PL.y + PL.h); g.moveTo(PL.x, y); g.lineTo(PL.x + PL.w, y); }
        g.stroke();
        g.setLineDash([3, 5]); g.strokeStyle = V.alpha(C.muted, 0.8); g.beginPath(); g.arc(cx, cy, u, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        // círculo transformado
        g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 1.6; g.beginPath();
        for (let k = 0; k <= 64; k++) { const an = (k / 64) * Math.PI * 2; const p = Tf(Math.cos(an), Math.sin(an)); if (k) g.lineTo(p[0], p[1]); else g.moveTo(p[0], p[1]); }
        g.stroke();
        if (S.m === 'proj' || S.m === 'refl') {
          const r = (S.ph * Math.PI) / 180; const p1 = O(-Math.cos(r) * 3, -Math.sin(r) * 3), p2 = O(Math.cos(r) * 3, Math.sin(r) * 3);
          V.line(g, p1[0], p1[1], p2[0], p2[1], { color: V.alpha(C.c, 0.7), lw: 1.5, dash: [6, 4] });
          const pr = O(Math.cos(r) * 1.9, Math.sin(r) * 1.9);
          pill(g, pr[0], pr[1], S.m === 'refl' ? 'normal n' : 'direção r', C.c, { size: 10, align: 'center' });
        }
        pts.forEach(([x, y]) => { const p = O(x, y); V.dot(g, p[0], p[1], 2.2, V.alpha(C.muted, 0.45)); });
        pts.forEach(([x, y, kind]) => { const p = Tf(x, y); V.dot(g, p[0], p[1], kind ? 3.4 : 2.8, kind === 1 ? C.b : C.a); });
        const o = O(0, 0), up = Tf(U[0], U[1]), vp = Tf(Vv[0], Vv[1]);
        V.arrow(g, o[0], o[1], O(U[0], U[1])[0], O(U[0], U[1])[1], { color: V.alpha(C.muted, 0.6), lw: 1.2, dash: [4, 4], head: 7 });
        V.arrow(g, o[0], o[1], O(Vv[0], Vv[1])[0], O(Vv[0], Vv[1])[1], { color: V.alpha(C.muted, 0.6), lw: 1.2, dash: [4, 4], head: 7 });
        V.arrow(g, o[0], o[1], up[0], up[1], { color: C.b, lw: 2.6, head: 10 });
        V.arrow(g, o[0], o[1], vp[0], vp[1], { color: C.e || C.a, lw: 2.6, head: 10 });
        const ML = S.m === 'proj' ? 'P' : 'Q';
        V.text(g, ML + 'u', up[0] + 6, up[1] - 6, { size: 12, font: 'mono', weight: 700, color: C.b });
        V.text(g, ML + 'v', vp[0] + 6, vp[1] - 6, { size: 12, font: 'mono', weight: 700, color: C.e || C.a });
        g.restore();
        // painel
        panelBox(g, PR, C, { rot: 'ROTAÇÃO R(θ)', refl: 'REFLEXÃO DE HOUSEHOLDER I − 2nnᵀ', hada: 'HADAMARD H₂ = [[1, 1], [1, −1]]/√2', proj: 'P = I − α r̂r̂ᵀ' }[S.m]);
        const det = a * d - b * c;
        const Uq = [a * U[0] + b * U[1], c * U[0] + d * U[1]], Vq = [a * Vv[0] + b * Vv[1], c * Vv[0] + d * Vv[1]];
        const dot0 = U[0] * Vv[0] + U[1] * Vv[1], dot1 = Uq[0] * Vq[0] + Uq[1] * Vq[1];
        const n0 = Math.hypot(U[0], U[1]), n1 = Math.hypot(Uq[0], Uq[1]);
        const px = PR.x + 14, pwid = PR.w - 28;
        let yy = PR.y + 42;
        const cell = narrow ? 44 : 44;
        [[a, b], [c, d]].forEach((row, i) => row.forEach((v, j) => { V.box(g, px + j * (cell + 5), yy + i * (cell + 5), cell, cell, { r: 7, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.7), lw: 1.2 }); V.text(g, sg(v, 2), px + j * (cell + 5) + cell / 2, yy + i * (cell + 5) + cell / 2 + 1, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' }); }));
        const tx = px + 2 * (cell + 5) + 16, tww = PR.x + PR.w - tx - 12;
        V.text(g, 'det = ' + sg(det, 2), tx, yy + 14, { size: 13, font: 'mono', weight: 700, color: C.d });
        V.text(g, 'u·v: ' + sg(dot0, 2) + ' → ' + sg(dot1, 2), tx, yy + 34, { size: 12, font: 'mono', color: Math.abs(dot0 - dot1) < 0.02 ? C.a : C.c, maxW: tww });
        V.text(g, '‖u‖: ' + nf(n0, 2) + ' → ' + nf(n1, 2), tx, yy + 54, { size: 12, font: 'mono', color: Math.abs(n0 - n1) < 0.02 ? C.a : C.c, maxW: tww });
        const orth = Math.abs(det * det - 1) < 0.03 && Math.abs(dot0 - dot1) < 0.03 && Math.abs(n0 - n1) < 0.02;
        V.text(g, orth ? 'ortogonal: preserva tudo' : 'não é ortogonal', tx, yy + 74, { size: 12, weight: 600, color: orth ? C.a : C.c, maxW: tww });
        yy += 2 * (cell + 5) + 22;
        if (S.m === 'rot') {
          yy = wrap(g, 'Rᵀ = R(−θ) desfaz a rotação de graça. Na RoPE, a posição m gira cada par de dimensões por m·θᵢ, e q·k passa a depender só da distância.', px, yy, pwid, 16, { size: 12, color: C.muted });
        } else if (S.m === 'refl') {
          yy = wrap(g, 'A componente ao longo de n troca de sinal; o resto fica. Ortogonal, com det = −1. É o caso α = 2 da família I − α r̂r̂ᵀ.', px, yy, pwid, 16, { size: 12, color: C.muted });
        } else if (S.m === 'proj') {
          const fct = 1 - S.al;
          V.text(g, 'componente em r × (1 − α) = ' + sg(fct, 2), px, yy + 4, { size: 13, font: 'mono', weight: 700, color: C.c, maxW: pwid });
          const bx = px, by = yy + 20, bww = pwid, bhh = 12;
          V.box(g, bx, by, bww, bhh, { r: 5, fill: V.alpha(C.muted, 0.15) });
          const X = (v) => bx + ((v + 2) / 3) * bww;
          V.box(g, Math.min(X(0), X(fct)), by, Math.abs(X(fct) - X(0)), bhh, { r: 5, fill: C.c });
          [[1, 'α=0'], [0, 'α=1: remove'], [-1, 'α=2: espelha'], [-2, 'α=3: GLM']].forEach(([v, l], i) => { V.line(g, X(v), by - 3, X(v), by + bhh + 3, { color: C.ink, lw: 1 }); V.text(g, l, X(v), by + bhh + 15 + (i % 2) * 13, { size: 10.5, font: 'mono', color: C.muted, align: v === -2 ? 'left' : v === 1 ? 'right' : 'center' }); });
          yy = by + bhh + 50;
          yy = wrap(g, 'ABLIT_ALPHA = 3.0 no método proj da abliteração do GLM: a direção de recusa sai das saídas do o_proj invertida e dobrada (×−2). Só α = 0 e α = 2 preservam normas.', px, yy, pwid, 16, { size: 12, color: C.muted });
        } else {
          const nb = 8, bh = narrow ? 38 : 56, bwd = (pwid - 20) / 2 / nb;
          const mx = 5.3;
          const draw = (arr, x0, lab, col) => {
            const base = yy + bh + 8;
            arr.forEach((v, i) => { const hh = (Math.abs(v) / mx) * bh; g.fillStyle = col; V.rr(g, x0 + i * bwd + 1, v >= 0 ? base - hh : base, bwd - 2, Math.max(1, hh), 2); g.fill(); });
            V.line(g, x0, base, x0 + nb * bwd, base, { color: C.muted, lw: 1 });
            V.text(g, lab, x0, base + bh * 0.4 + 16, { size: 11, font: 'mono', color: C.muted });
          };
          draw(HX, px, 'x: max|x| = ' + nf(Math.max(...HX.map(Math.abs)), 2), C.c);
          draw(HY, px + pwid / 2 + 10, 'H₈x: max = ' + nf(Math.max(...HY.map(Math.abs)), 2), C.a);
          yy += bh * 1.4 + 40;
          yy = wrap(g, 'Mesma norma (' + nf(Math.hypot(...HX), 2) + ' = ' + nf(Math.hypot(...HY), 2) + '), mas o outlier virou 8 valores médios: bem mais fácil de quantizar. (xQ)(QᵀW) = xW.', px, yy, pwid, 16, { size: 12, color: C.muted });
        }
        return S.cur.some((v, k) => Math.abs(v - prev[k]) > 1e-4) || (S.m === 'rot' && Math.abs(S.th - S.ang) > 0.05);
      });
    })();

    /* ================================================== Fig 9: gangorra / roofline */
    (function figGangorra() {
      const fig = root.querySelector('#fig-gangorra');
      stageH(fig, 450, 760);
      const MATS = {
        kda: { lab: 'GLM · KDA in_proj (por GPU)', N: 12576, K: 4096 },
        exp: { lab: 'GLM · 1 expert gate_proj (M por expert)', N: 2048, K: 4096, dec: [1], pre: [28, 199], decLab: 'decode (~1 linha por expert)', decLabN: 'decode (~1/expert)', preLab: ['prefill por expert', 'M ≈ 28–199'], preLabN: ['prefill/expert', 'M ≈ 28–199'] },
        qsh: { lab: 'Qwen · shared expert gate_up fundido (por GPU)', N: 640, K: 2560 },
        qlm: { lab: 'Qwen · lm_head (por GPU)', N: 124160, K: 2560 },
        dsq: { lab: 'DeepSeek · wq_b (por GPU)', N: 16384, K: 1024 },
      };
      const S = { m: 'kda', bits: 16, lm: 5, bw: 273e9, ang: 0 };
      const P = 125e12;
      let H = null;
      const rd = () => { if (H) H.redraw(); };
      const ctl = UI.controls(fig);
      fitSelect(UI.select(ctl, { label: 'Matriz', options: Object.keys(MATS).map((k) => ({ v: k, label: MATS[k].lab })), value: 'kda', onChange: (v) => { S.m = v; rd(); } }));
      UI.seg(ctl, { label: 'bits/peso', options: [{ v: 16, label: 'BF16' }, { v: 8, label: 'FP8' }, { v: 4, label: '4 bits' }], value: 16, onChange: (v) => { S.bits = v; rd(); } });
      UI.slider(ctl, { label: 'M (linhas)', min: 0, max: 14, step: 0.25, value: 5, fmt: (v) => nf(Math.round(Math.pow(2, v))), onInput: (v) => { S.lm = v; rd(); } });
      UI.seg(ctl, { label: 'banda', options: [{ v: 273e9, label: '273 GB/s (teórica)' }, { v: 230e9, label: '230 (efetiva)' }], value: 273e9, onChange: (v) => { S.bw = v; rd(); } });
      const cost = (M, N, K, bits, B) => { const fl = 2 * M * N * K; const by = N * K * (bits / 8) + M * K * 2 + M * N * 2; return { fl, by, tm: by / B, tc: fl / P, I: fl / by }; };
      H = liveFig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const mt = MATS[S.m], M = Math.round(Math.pow(2, S.lm));
        const cc = cost(M, mt.N, mt.K, S.bits, S.bw);
        const narrow = w < 700;
        const PLr = narrow ? { x: 0, y: 0, w, h: h * 0.52 } : { x: 0, y: 0, w: w * 0.54, h };
        const PRr = narrow ? { x: 0, y: h * 0.52, w, h: h * 0.48 } : { x: w * 0.54, y: 0, w: w * 0.46, h };
        // --- roofline
        const px = PLr.x + (narrow ? 48 : 60), pw = PLr.w - (narrow ? 64 : 84), py = PLr.y + 40, ph = PLr.h - (narrow ? 86 : 100);
        const lx0 = -0.5, lx1 = 4.2, ly0 = -1, ly1 = 2.5;
        const X = (I) => px + ((Math.log10(I) - lx0) / (lx1 - lx0)) * pw, Y = (tf) => py + ph - ((Math.log10(tf) - ly0) / (ly1 - ly0)) * ph;
        V.text(g, 'ROOFLINE DO GB10 (log-log)', PLr.x + 16, PLr.y + 22, { size: 11, font: 'mono', weight: 600, color: C.muted });
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
        for (let e = 0; e <= 4; e++) { g.moveTo(X(Math.pow(10, e)), py); g.lineTo(X(Math.pow(10, e)), py + ph); }
        for (let e = -1; e <= 2; e++) { g.moveTo(px, Y(Math.pow(10, e))); g.lineTo(px + pw, Y(Math.pow(10, e))); }
        g.stroke();
        V.line(g, px, py + ph, px + pw, py + ph, { color: C.muted, lw: 1 }); V.line(g, px, py, px, py + ph, { color: C.muted, lw: 1 });
        [1, 10, 100, 1000, 10000].forEach((v) => V.text(g, v >= 1000 ? nf(v / 1000) + 'k' : String(v), X(v), py + ph + 15, { size: 10.5, font: 'mono', color: C.muted, align: 'center' }));
        [0.1, 1, 10, 100].forEach((v) => V.text(g, String(v).replace('.', ','), px - 6, Y(v) + 4, { size: 10.5, font: 'mono', color: C.muted, align: 'right' }));
        V.text(g, 'FLOPs por byte (intensidade)', px + pw / 2, py + ph + 32, { size: 11, color: C.muted, align: 'center' });
        g.save(); g.translate(PLr.x + 14, py + ph / 2); g.rotate(-Math.PI / 2); V.text(g, 'TFLOPS atingíveis', 0, 0, { size: 11, color: C.muted, align: 'center' }); g.restore();
        const ridge = P / S.bw;
        g.strokeStyle = C.d; g.lineWidth = 2.5; g.beginPath();
        for (let k = 0; k <= 80; k++) { const I = Math.pow(10, lx0 + ((lx1 - lx0) * k) / 80); const perf = Math.min(P, I * S.bw) / 1e12; const x = X(I), y = Y(Math.max(0.1, perf)); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
        g.stroke();
        // linha do ridge só abaixo do platô; rótulo na base, à esquerda da linha; o do pico, acima do platô
        const yTop = Y(P / 1e12);
        V.line(g, X(ridge), yTop, X(ridge), py + ph, { color: V.alpha(C.d, 0.6), lw: 1, dash: [4, 4] });
        V.text(g, 'ridge ≈ ' + nf(ridge) + ' FLOP/B', X(ridge) - 5, py + ph - 8, { size: 10.5, font: 'mono', color: C.d, align: 'right' });
        if (!mt.dec) V.text(g, '↖ rampa: I × banda', X(narrow ? 6 : 12), Y(narrow ? 0.35 : 0.22), { size: 10.5, color: C.muted, align: 'left' });
        V.text(g, 'pico BF16 ~125 TFLOPS', px + pw - 4, yTop - 7, { size: 10.5, color: C.muted, align: 'right' });
        // marcadores das receitas (ou, no preset de expert, o M de cada expert)
        const decM = mt.dec || [32, 42], preM = mt.pre || [7168, 8192];
        const mk = (m) => { const q = cost(m, mt.N, mt.K, S.bits, S.bw); return [X(q.I), Y(Math.min(P, q.I * S.bw) / 1e12), q.I]; };
        const dm = decM.map(mk), pm = preM.map(mk);
        dm.forEach(([x, y]) => V.dot(g, x, y, 3.5, C.b));
        pm.forEach(([x, y]) => V.dot(g, x, y, 3.5, C.a));
        // rótulo de um grupo de pontos na rampa: à direita-abaixo; se isso cruzar a linha do ridge, à esquerda-acima
        const xR = X(ridge);
        const rampLab = (pt, lines, col) => {
          const wmax = Math.max(...lines.map((l) => tw(g, l, 10.5, 'mono', 400)));
          const left = pt[0] + 8 + wmax > xR - 4 && pt[0] - 10 - wmax >= px + 2;
          lines.forEach((l, i) => V.text(g, l, left ? pt[0] - 10 : pt[0] + 8, left ? pt[1] - 12 - (lines.length - 1 - i) * 14 : pt[1] + 14 + i * 14, { size: 10.5, font: 'mono', color: col, align: left ? 'right' : 'left' }));
        };
        const dl = dm[dm.length - 1];
        rampLab(dl, [(narrow && mt.decLabN) || mt.decLab || 'decode (M = 32–42)'], C.b);
        const pl = pm[pm.length - 1], pLab = (narrow && mt.preLabN) || mt.preLab || ['prefill', 'M = 7168–8192'];
        if (pl[2] >= ridge) {
          V.text(g, pLab[0], px + pw - 4, yTop + 18, { size: 10.5, font: 'mono', color: C.a, align: 'right' });
          V.text(g, pLab[1], px + pw - 4, yTop + 32, { size: 10.5, font: 'mono', color: C.a, align: 'right' });
        } else rampLab(pl, pLab, C.a);
        const perf = Math.min(P, cc.I * S.bw) / 1e12;
        V.glow(g, X(cc.I), Y(perf), 20, C.c, 0.9); V.dot(g, X(cc.I), Y(perf), 5, C.c);
        // --- gangorra
        const R = PRr;
        const tmax = Math.max(cc.tm, cc.tc);
        const tgt = clamp(Math.log10(cc.tc / cc.tm) * 0.24, -0.24, 0.24);
        S.ang += (tgt - S.ang) * Math.min(1, dt * 5 + 0.01);
        const fx = R.x + R.w / 2, fy = R.y + (narrow ? 118 : 150), L = Math.min(R.w * 0.34, 150);
        g.beginPath(); g.moveTo(fx, fy); g.lineTo(fx - 16, fy + 26); g.lineTo(fx + 16, fy + 26); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.5); g.fill();
        const ex = Math.cos(S.ang) * L, ey = Math.sin(S.ang) * L;
        V.line(g, fx - ex, fy - ey, fx + ex, fy + ey, { color: C.ink, lw: 4 });
        V.dot(g, fx, fy, 5, C.ink);
        const pan = (x, y, val, col, lab) => {
          const bh = 14 + 38 * (val / tmax), bwd = 74;
          V.line(g, x, y, x, y + 16, { color: C.muted, lw: 1.2 });
          V.box(g, x - bwd / 2, y + 16, bwd, bh, { r: 8, fill: V.alpha(col, 0.3), stroke: col, lw: 1.5 });
          V.text(g, fmtT(val), x, y + 16 + bh / 2 + 4, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center' });
          V.text(g, lab, x, y + 16 + bh + 16, { size: 11, color: C.muted, align: 'center' });
        };
        pan(fx - ex, fy - ey, cc.tm, C.b, 'ler bytes');
        pan(fx + ex, fy + ey, cc.tc, C.a, 'fazer contas');
        const mem = cc.tm >= cc.tc;
        V.text(g, mt.lab + ' · [' + nf(mt.N) + ' × ' + nf(mt.K) + ']', R.x + 16, R.y + 24, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: R.w - 30 });
        V.text(g, 'M = ' + nf(M) + ' · ' + (S.bits === 16 ? 'BF16' : S.bits === 8 ? 'FP8' : '4 bits') + ' · I = ' + nf(cc.I, cc.I < 10 ? 1 : 0) + ' FLOP/B', R.x + 16, R.y + 44, { size: 12, font: 'mono', color: C.muted, maxW: R.w - 30 });
        pill(g, R.x + R.w / 2, R.y + (narrow ? 72 : 82), mem ? 'limitado por memória' : 'limitado por cálculo', mem ? C.b : C.a, { size: 12, align: 'center' });
        const ty = fy + (narrow ? 128 : 144);
        V.text(g, 'tempo mínimo ≈ ' + fmtT(tmax) + '  ·  ' + fmtFl(cc.fl) + '  ·  ' + fmtBytes(cc.by), R.x + 16, ty, { size: 12, font: 'mono', color: C.ink, maxW: R.w - 30 });
        let yy = ty + 20;
        if (S.m === 'kda') yy = wrap(g, 'Medido na receita (M = 7168): cuBLAS BF16 7,36 ms · W8A8 12,57 ms · Marlin FP8 27,21 ms. O despacho troca para BF16 quando M > 512.', R.x + 16, yy, R.w - 30, 15, { size: 11.5, color: C.d });
        else if (S.m === 'exp') yy = wrap(g, 'Cada expert só vê os seus tokens: M ≈ tokens × 8 / 288. Pedaço de 1024 → ~28 linhas; pedaço de 7168 (o da receita) → ~199, já perto do ridge com 4 bits. Decode: menos de 1 linha por expert. Escolha “4 bits” (EXL3). Contas nossas.', R.x + 16, yy, R.w - 30, 15, { size: 11.5, color: C.d });
        else yy = wrap(g, S.bits < 16 ? 'Pesos menores ajudam onde se lê (decode). Onde se calcula (prefill), um kernel weight-only ainda desquantiza para BF16 antes da conta.' : 'Leve M de 1 a 42: a gangorra quase não se mexe. É o regime do decode.', R.x + 16, yy, R.w - 30, 15, { size: 11.5, color: C.muted });
        return Math.abs(tgt - S.ang) > 1e-4;
      });
    })();

    /* ================================================== Fig 10: leitor de formas */
    (function figLeitor() {
      const fig = root.querySelector('#fig-leitor');
      stageH(fig, 450, 780);
      const L = [
        { k: 'glm-embed', r: 'glm', lab: 'GLM · embed_tokens', out: 154880, inn: 4096, bits: 16, fmt: 'BF16', tp: 'vocab', embed: true, note: 'Não é GEMM: é uma consulta de linha (one-hot × E). A primeira dimensão é o id do token, não uma saída. Cada GPU guarda metade das linhas (vocab-parallel).' },
        { k: 'glm-kdaq', r: 'glm', lab: 'GLM · KDA q_proj', out: 8192, inn: 4096, bits: 8.004, fmt: 'FP8 por canal (Marlin W8A16)', tp: 'col', note: 'No vLLM, fundida com k, v, b, f_a e g_a em in_proj_qkvbfg_a: [12576 × 4096] por GPU.' },
        { k: 'glm-mlao', r: 'glm', lab: 'GLM · MLA o_proj', out: 4096, inn: 16384, bits: 8.004, fmt: 'FP8 por canal (GLM53_DENSE_FP8=all)', tp: 'row', note: '64 cabeças × 256 de volta ao residual. É uma das matrizes que a abliteração edita (ABLIT=1: transplante do doador ou, no método proj, ABLIT_ALPHA).' },
        { k: 'glm-exp', r: 'glm', lab: 'GLM · expert gate_proj', out: 2048, inn: 4096, bits: 4, fmt: 'EXL3, 4 bits por peso (treliça)', disk: 'trellis I16 [256, 128, 64] = [K/16, N/16, 64] + suh [4096] + svh [2048]', tp: 'col', mult: 288, note: '288 experts × 3 matrizes por camada MoE. O EXL3 guarda K primeiro, em tiles de 16 × 16. No TP=2: gate/up por coluna ([256, 64, 64] por GPU), down por linha.' },
        { k: 'qwen-lm', r: 'qwen', lab: 'Qwen · lm_head', out: 248320, inn: 2560, bits: 16, fmt: 'BF16', tp: 'vocab', note: '[124160 × 2560] por GPU. Com MTP=3 ela é lida 4 vezes por passo (verificação + 3 rascunhos): 26% dos bytes do decode.' },
        { k: 'qwen-qkv', r: 'qwen', lab: 'Qwen · GDN in_proj_qkv', out: 10240, inn: 2560, bits: 16, fmt: 'BF16', tp: 'col', note: 'Q 2048 + K 2048 + V 6144 empacotados. Fundida com in_proj_z: (8192, 2560) por GPU.' },
        { k: 'qwen-a', r: 'qwen', lab: 'Qwen · GDN in_proj_a', out: 48, inn: 2560, bits: 16, fmt: 'BF16 (MXFP8 no checkpoint alternativo)', tp: 'col', note: 'N = 48 < 128: o kernel mm_mxfp8 do sm_121 recusa; a receita cai para emulação BF16.' },
        { k: 'qwen-hc', r: 'qwen', lab: 'Qwen · GR input_mix_weight_down', out: 320, inn: 10240, bits: 16, fmt: 'BF16', tp: 'rep', note: 'Posto 320 = d/8. Replicada nas duas GPUs ([336 × 10240] com o inject). 320 não é múltiplo de 128.' },
        { k: 'qwen-exp', r: 'qwen', lab: 'Qwen · expert gate_proj', out: 640, inn: 2560, bits: 4.5, fmt: 'NVFP4 (4 bits + escala FP8 a cada 16)', disk: 'U8 [640, 1280] (2 valores FP4 por byte) + weight_scale F8 [640, 160]', tp: 'ep', mult: 512, note: 'Expert parallel (256 experts inteiros por GPU): em TP puro o intermediário 640 viraria 320 por GPU, e os experts FP8 do MTP, blocados em 128 × 128, não aceitam 320.' },
        { k: 'ds-wqa', r: 'ds', lab: 'DeepSeek · fused_wqa_wkv', out: 1536, inn: 4096, bits: 8, fmt: 'FP8 blocos 128×128', tp: 'rep', note: 'wq_a (4096 → 1024) e wkv (4096 → 512) numa GEMM só, replicada: o latente de query e o KV único.' },
        { k: 'ds-wqb', r: 'ds', lab: 'DeepSeek · wq_b', out: 32768, inn: 1024, bits: 8, fmt: 'FP8 blocos 128×128', tp: 'col', note: '1024 → 64 cabeças × 512. Cortada por cabeças: 32 por GPU.' },
        { k: 'ds-wob', r: 'ds', lab: 'DeepSeek · wo_b', out: 4096, inn: 8192, bits: 8, fmt: 'FP8 blocos 128×128', tp: 'row', note: '8 grupos × 1024 → 4096. Row-parallel: um all-reduce por camada (43 por passo).' },
        { k: 'ds-mkv', r: 'ds', lab: 'DeepSeek · markov_w2', out: 129280, inn: 256, bits: 16, fmt: 'BF16', tp: 'vocab', note: 'Posto 256 → vocabulário. 66 MB: daria para replicar, mas no A/B da lane padrão a vazão caiu; ficou cortada por vocabulário.' },
        { k: 'ds-exp', r: 'ds', lab: 'DeepSeek · expert w1', out: 2048, inn: 4096, bits: 4.25, fmt: 'MXFP4 (4 bits + escala E8M0, uma potência de 2, a cada 32)', tp: 'col', mult: 256, note: '256 experts por camada, com TP dentro de cada expert (um all-reduce por camada MoE).' },
      ];
      const S = { k: 'glm-kdaq', t0: 0 };
      let H = null;
      const ctl = UI.controls(fig);
      const selT = UI.select(ctl, { label: 'Tensor', options: L.map((x) => ({ v: x.k, label: x.lab })), value: S.k, onChange: (v) => { S.k = v; S.t0 = H.clock(); H.redraw(); } });
      fitSelect(selT);
      const TPL = { col: 'corte pela saída (column-parallel)', row: 'corte pela entrada (row-parallel) + all-reduce', rep: 'replicada nas duas GPUs', vocab: 'corte por linhas do vocabulário', ep: 'expert parallel: experts inteiros por GPU' };
      H = liveFig(fig, (g, w, h, t) => {
        const C = V.C();
        const it = L.find((x) => x.k === S.k);
        const el = Math.min(1, (t - S.t0) / 0.7);
        const e = V.ease.out(el);
        const narrow = w < 700;
        const A = narrow ? { x: 0, y: 0, w, h: h * 0.4 } : { x: 0, y: 0, w: w * 0.5, h };
        const B = narrow ? { x: 0, y: h * 0.4, w, h: h * 0.6 } : { x: w * 0.5, y: 0, w: w * 0.5, h };
        const sz = (v) => 28 + ((Math.log10(v) - 1.5) / 4) * (Math.min(A.w, A.h) * 0.7 - 28);
        const rw = sz(it.inn) * lerp(0.6, 1, e), rh = sz(it.out) * lerp(0.6, 1, e);
        const cx = A.x + A.w / 2 + 10, cy = A.y + A.h / 2 + 10;
        const x0 = cx - rw / 2, y0 = cy - rh / 2;
        const g0 = C.a, g1 = C.b;
        const rect = (x, y, ww, hh, col, a = 0.28) => V.box(g, x, y, ww, hh, { r: 6, fill: V.alpha(col, a), stroke: col, lw: 1.6 });
        const inLab = (s1, x, y, maxw) => { if (tw(g, s1, 10.5, 'mono', 600) < maxw - 8) V.text(g, s1, x, y, { size: 10.5, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' }); };
        if (it.tp === 'col' || it.tp === 'vocab') { rect(x0, y0, rw, rh / 2 - 2, g0); rect(x0, y0 + rh / 2 + 2, rw, rh / 2 - 2, g1); if (rh > 50) { inLab('GPU 0 · ' + nf(it.out / 2) + ' linhas', cx, y0 + rh / 4, rw); inLab('GPU 1 · ' + nf(it.out / 2) + ' linhas', cx, y0 + (3 * rh) / 4, rw); } }
        else if (it.tp === 'row') { rect(x0, y0, rw / 2 - 2, rh, g0); rect(x0 + rw / 2 + 2, y0, rw / 2 - 2, rh, g1); V.text(g, 'Σ', cx + rw / 2 + 24, cy + 6, { size: 22, weight: 700, color: C.d, align: 'center' }); V.text(g, 'all-reduce', cx + rw / 2 + 24, cy + 24, { size: 10, font: 'mono', color: C.d, align: 'center' }); }
        else if (it.tp === 'rep') { rect(x0 - 8, y0 - 8, rw, rh, g1, 0.18); rect(x0 + 4, y0 + 4, rw, rh, g0); if (rh > 30) inLab('cópia inteira por GPU', cx + 4, cy + 4, rw); }
        else if (it.tp === 'ep') {
          const nC = 16, nR = 8, cw = rw / nC, chh = rh / nR;
          for (let i = 0; i < nR; i++) for (let j = 0; j < nC; j++) { const col = (i * nC + j) < (nC * nR) / 2 ? g0 : g1; V.box(g, x0 + j * cw + 1, y0 + i * chh + 1, cw - 2, chh - 2, { r: 2, fill: V.alpha(col, 0.4) }); }
          g.strokeStyle = C.ink; g.lineWidth = 1; V.rr(g, x0, y0, rw, rh, 6); g.stroke();
        }
        // x e y
        V.box(g, x0, y0 - 16, rw, 7, { r: 3, fill: V.alpha(C.c, 0.6) });
        V.text(g, it.embed ? 'saída ' + nf(it.inn) + ' (a linha lida)' : 'entrada ' + nf(it.inn), x0 + rw / 2, y0 - 22, { size: 11, font: 'mono', color: it.embed ? C.d : C.c, align: 'center' });
        V.box(g, x0 - 16, y0, 7, rh, { r: 3, fill: V.alpha(it.embed ? C.c : C.d, 0.6) });
        g.save(); g.translate(x0 - 22, cy); g.rotate(-Math.PI / 2); V.text(g, it.embed ? 'entrada: id do token (1 de ' + nf(it.out) + ')' : 'saída ' + nf(it.out), 0, 0, { size: 11, font: 'mono', color: it.embed ? C.c : C.d, align: 'center' }); g.restore();
        if (it.embed) { const ry = y0 + rh * 0.37; V.box(g, x0, ry, rw, Math.max(3, rh * 0.03), { r: 2, fill: V.alpha(C.c, 0.9) }); V.arrow(g, x0 + rw + 6, ry + 2, x0 + rw + 34, ry + 2, { color: C.c, lw: 1.6, head: 7 }); }
        V.text(g, 'lados em escala log', A.x + 14, A.y + A.h - 12, { size: 10.5, color: C.muted });
        pill(g, A.x + 14, A.y + 20, 'GPU 0', g0, { size: 10.5 });
        pill(g, A.x + 76, A.y + 20, 'GPU 1', g1, { size: 10.5 });
        // painel
        const px = B.x + (narrow ? 16 : 10), pw = B.w - (narrow ? 32 : 26);
        let yy = B.y + (narrow ? 20 : 34);
        const rc = { glm: C.glm, qwen: C.qwen, ds: C.ds }[it.r] || C.a;
        V.text(g, it.lab, px, yy, { size: 14, weight: 700, color: rc, maxW: pw }); yy += 22;
        const head = it.embed ? 'weight [vocab, hidden] = [' + nf(it.out) + ', ' + nf(it.inn) + ']: id → ' + nf(it.inn) + ' números' : (it.disk ? 'forma lógica [' : 'weight [') + nf(it.out) + ', ' + nf(it.inn) + ']  →  ' + nf(it.inn) + ' → ' + nf(it.out);
        yy = wrap(g, head, px, yy, pw, 16, { size: 12, font: 'mono', color: C.ink }) + 4;
        const params = it.out * it.inn;
        const rows = [
          ['parâmetros', fmtCount(params) + (it.mult ? '  (× ' + it.mult + ' experts = ' + fmtCount(params * it.mult) + ')' : '')],
          ['BF16', fmtBytes(params * 2)],
          ['na receita', it.fmt + ' ≈ ' + fmtBytes((params * it.bits) / 8)],
          ...(it.disk ? [['no disco', it.disk]] : []),
          ['FLOPs/token', it.embed ? '≈ 0 (consulta de linha)' : fmtFl(2 * params)],
          ['TP=2', TPL[it.tp]],
        ];
        if (it.embed) rows.push(['decode', 'lê só 1 linha por token (' + fmtBytes(it.inn * 2) + ')']);
        else { const M = it.mult ? 1 : 32, fl = 2 * M * params, by = (params * it.bits) / 8 + M * it.inn * 2 + M * it.out * 2; rows.push([it.mult ? 'decode/expert' : 'decode M=32', (it.mult ? '~1 linha por expert (poucos tokens × top-k ÷ ' + it.mult + '): ' : '') + 'lê ' + fmtBytes(by) + ' ≥ ' + fmtT(Math.max(by / 273e9, fl / 125e12)) + ' · I ≈ ' + nf(fl / by, fl / by < 10 ? 1 : 0) + ' FLOP/B']); }
        rows.forEach(([k, v]) => { V.text(g, k, px, yy, { size: 11.5, font: 'mono', color: C.muted }); yy = wrap(g, v, px + 96, yy, pw - 96, 15, { size: 12, font: 'mono', color: C.ink }); yy += 3; });
        yy += 4;
        const ax = it.embed ? ['vocab', 'hidden'] : ['saída', 'entrada'];
        const chips = [[ax[0] + ' ÷ 2', it.out % 2 === 0], [ax[0] + ' ÷ 3', it.out % 3 === 0], [ax[0] + ' % 32', it.out % 32 === 0], [ax[0] + ' % 128', it.out % 128 === 0], [ax[1] + ' % 128', it.inn % 128 === 0]];
        let cxp = px;
        chips.forEach(([l, ok]) => { const s = (ok ? '✓ ' : '✕ ') + l; const wd = tw(g, s, 10.5, 'mono', 600) + 14; if (cxp + wd > px + pw) { cxp = px; yy += 24; } pill(g, cxp, yy + 4, s, ok ? C.a : C.red, { size: 10.5 }); cxp += wd + 6; });
        yy += 28;
        wrap(g, it.note, px, yy, pw, 16, { size: 12, color: C.muted });
        return el < 1;
      }, { onEnter: () => { if (H) S.t0 = H.clock(); } });
    })();
  },
});

LA.chapter({
  id: 'especulacao-fundamentos',
  part: 7,
  order: 35,
  title: 'Decodificação especulativa: rascunhar e verificar',
  navTitle: 'Especulação: fundamentos',
  subtitle: 'Por que chutar vários tokens e conferir todos de uma vez acelera o decode sem mudar a distribuição do texto: a física do passo, a amostragem por rejeição, a aceitação, a escolha de k e o que muda dentro do motor, e a verificação exata do TensorFold.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['speculative decoding', 'rejection sampling', 'taxa de aceitação', 'token bônus', 'num_speculative_tokens', 'draft_sample_method', 'rollback de estado', 'verificação por igualdade', 'copy drafts'],
  goals: [
    'Explicar por que o decode <b>memory-bound</b> torna quase grátis verificar \\(k+1\\) posições, e por que isso é menos verdade num MoE',
    'Demonstrar que a <b>amostragem por rejeição</b> preserva exatamente a distribuição do alvo, com rascunho greedy ou probabilístico',
    'Calcular \\(\\alpha = 1-\\mathrm{TV}(p,q)\\), tokens por passo e speedup, e escolher \\(k\\) pela regra marginal',
    'Montar e vigiar o <code>--speculative-config</code> das três receitas, incluindo tamanhos de captura, rollback de estado recorrente e gates de aceitação',
    'Comparar a rejeição do vLLM com a <b>verificação por igualdade</b> do TensorFold (amostragem chaveada): o que cada uma preserva, quanto aceita e como a receita GLM·TF decide quantos rascunhos propor a cada rodada',
  ],
  prereqs: ['prefill-decode-kv', 'geracao-amostragem', 'scheduler-batching'],
  summary: [
    'Decode é limitado pela leitura dos pesos: um forward com \\(M = 1+k\\) linhas lê \\(W\\) uma vez e o reutiliza \\(M\\) vezes. Em MoE, linhas extras chamam especialistas diferentes (GLM: 8 linhas ≈ 2,3× os bytes de 1), e a especulação rende menos.',
    'O laço: propor \\(k\\) rascunhos, verificar as \\(k+1\\) posições num forward, aceitar o maior prefixo, emitir uma correção (na 1ª rejeição) ou um <b>token bônus</b> (se todos passam). Cada passo emite de 1 a \\(k+1\\) tokens.',
    'Rejeição <code>standard</code>: aceita com \\(\\min(1, p/q)\\) e reamostra de \\(\\max(0, p-q)\\). A saída tem a distribuição exata do alvo, seja qual for o rascunhador: ele só muda a velocidade. Na lane TensorFold do GLM, a verificação é por <b>igualdade</b> com a amostra chaveada: sai a mesma sequência da decodificação serial, para qualquer rascunhador, e a aceitação só chega perto do teto \\(1-\\mathrm{TV}\\) se o rascunhador usar o ruído do alvo. A profundidade é decidida a cada rodada (<code>fnc7:0.3</code>, até 7 do DFlash2), depois dos copy drafts (até 15).',
    '\\(\\alpha = \\sum\\min(p,q) = 1-\\mathrm{TV}\\); \\(\\tau = 1+\\sum_i a_i\\), ou \\((1-\\alpha^{k+1})/(1-\\alpha)\\) no caso geométrico; speedup \\(\\approx \\tau/(1+kv+T_{\\text{rasc}})\\). A posição \\(k+1\\) só compensa se \\(a_{k+1} &gt; \\tau\\,\\Delta t/t\\) (Qwen: \\(k=3\\); DeepSeek: \\(k=5\\) empata na aceitação e barateia o passo).',
    '<code>draft_sample_method</code>: <code>greedy</code> (\\(q\\) one-hot, \\(\\alpha = p(\\hat x)\\)) ou <code>probabilistic</code> (\\(\\alpha = 1-\\mathrm{TV}\\), guarda \\(q\\)); os dois são sem perdas. DeepSeek: A/B empatado (56,5 × 56,6 tok/s), ficou <code>probabilistic</code>. GLM: <code>probabilistic</code> + <code>standard</code>.',
    'No motor: \\(S(1+k)\\) linhas por passo → capturas múltiplas de \\(k+1\\) (GLM <code>1 2 4 8 16 24 32</code>; DeepSeek: teto 48 para que 42 = 6×7 seja capturado); o KV rejeitado é descartado; estado recorrente exige fotografias (<code>num_spec+1</code>) escolhidas por <code>num_accepted_tokens</code> e conv com \\(K-1+k\\) colunas.',
    'Vigie a curva por posição, sempre por delta no <code>/metrics</code>: ela precisa decair. Posição 0 cravada em 1,00 é o vllm#53030 (<code>spec-accept-gate.sh</code>); prompts patológicos e abliteração derrubam a curva.',
  ],
  quiz: [
    { q: 'No decode, por que verificar \\(k+1\\) posições custa quase o mesmo que gerar 1 token?', o: ['Porque o alvo pula as camadas nas posições de rascunho', 'Porque o passo é limitado pela leitura dos pesos: eles são lidos uma vez e servem às \\(k+1\\) linhas, e as contas extras cabem no tempo de leitura', 'Porque o KV cache dos rascunhos já vem pronto do rascunhador', 'Porque os rascunhos só passam pelo lm_head'], a: 1, e: 'Com \\(M\\) pequeno, a intensidade aritmética fica muito abaixo do joelho do roofline (centenas de FLOP/B no GB10): o tempo é o de trazer \\(W\\) da memória, e reutilizá-lo para 8 linhas quase não custa. Num MoE, porém, linhas diferentes chamam especialistas diferentes e os bytes crescem.' },
    { q: 'O rascunho é “,” com \\(q=0{,}6\\) e o alvo dá \\(p=0{,}3\\) a ele. O que a rejeição <code>standard</code> faz?', o: ['Aceita sempre, porque \\(q &gt; p\\)', 'Rejeita sempre, porque \\(p &lt; q\\)', 'Aceita com probabilidade 0,5; se rejeitar, sorteia de \\(\\max(0, p-q)\\) normalizado', 'Aceita com probabilidade 0,3; se rejeitar, sorteia de \\(p\\) inteira'], a: 2, e: '\\(\\min(1, p/q) = 0{,}3/0{,}6 = 0{,}5\\). Na rejeição, o sorteio vem do resíduo, onde o alvo quer mais do que o rascunhador ofereceu. Sortear de \\(p\\) inteira daria probabilidade demais aos tokens que já tiveram chance de ser aceitos, e a distribuição final deixaria de ser \\(p\\).' },
    { q: 'Com aceitação constante \\(\\alpha = 0{,}8\\) e \\(k = 4\\), quantos tokens por passo se espera?', o: ['3,20', '3,36', '4,00', '5,00'], a: 1, e: '\\(\\tau = (1-0{,}8^5)/(1-0{,}8) = (1-0{,}328)/0{,}2 \\approx 3{,}36\\). O valor 5,00 é o teto com \\(k\\) infinito, \\(1/(1-\\alpha)\\). 3,20 seria \\(1 + 4\\times 0{,}55\\), sem sentido aqui.' },
    { q: 'Com <code>draft_sample_method: "greedy"</code> e o pedido a \\(T = 1\\), a especulação é sem perdas?', o: ['Não: o rascunho ignora a temperatura e distorce a saída', 'Sim: \\(q\\) é tratado como one-hot no favorito \\(\\hat x\\), a aceitação é \\(p(\\hat x)\\) e a correção sai de \\(p\\) sem \\(\\hat x\\), renormalizada', 'Só se o rascunhador tiver sido treinado de forma greedy', 'Sim, mas porque o alvo passa a responder de forma greedy'], a: 1, e: 'A prova da amostragem por rejeição vale para qualquer \\(q\\), inclusive degenerada. O que muda é a aceitação: \\(p(\\hat x)\\) no greedy contra \\(\\sum\\min(p,q)\\) no probabilístico. Qual é maior depende da calibração do rascunhador; no A/B do DeepSeek, deu empate.' },
    { q: 'Por que o KDA do GLM guarda <code>num_spec + 1</code> estados durante a verificação, se no KV basta ignorar as linhas rejeitadas?', o: ['Para economizar memória de KV', 'Porque o estado recorrente é um resumo de tamanho fixo que absorve cada token; desfazer rascunhos rejeitados exigiria inverter cada atualização, o que é instável em ponto flutuante e caro, então o kernel fotografa o estado após cada token e <code>num_accepted_tokens</code> escolhe a fotografia certa', 'Porque o CUDA graph exige um estado por linha', 'Porque o prefix caching precisa de um estado por token'], a: 1, e: 'KV é por posição, e descartar é cortar linhas. Com \\(S \\leftarrow \\gamma S + v k^\\top\\), em tese daria para voltar: \\(S^{(j)} = (S^{(j+1)} - v k^\\top)/\\gamma\\). Mas dividir por gates pequenos amplifica o erro de ponto flutuante, e seria preciso guardar chave, valor e gates de cada rascunho; \\(k+1\\) cópias são exatas e baratas. A conv curta é pior, porque a coluna mais antiga sai da janela e se perde; por isso ela ganha uma janela de \\(K-1+k\\) colunas (10 no GLM com \\(k=7\\)); no Qwen, a conv da PLE reserva <code>num_speculative_tokens</code> posições extras.' },
    { q: 'O <code>spec-accept-gate.sh</code> do GLM reprova um boot cuja aceitação na posição 0 está em 1,00. Por quê?', o: ['Porque 1,00 indica um rascunhador bom demais, que desperdiça \\(k\\)', 'Porque uma curva saudável decai; 1,00 cravado é a assinatura do vllm#53030 (colisão de BatchDescriptor em graphs PIECEWISE): a verificação não está verificando', 'Porque o contador passou de \\(2^{53}\\)', 'Porque com 1,00 o token bônus deixa de ser sorteado'], a: 1, e: 'Nem a contagem de 1 a 200 do bench aceita 100% na posição 0 (0,98 no GLM). O gate espera ≥ 100 drafts (senão SKIP), reprova quando pos0 &gt; 0,999 (0,999 exato ainda passa) e sai com código 2 quando não consegue julgar (série ausente, ambígua ou ilegível). Nunca reporta uma curva ilegível como PASS. Ele não é chamado pelo <code>start.sh</code>: roda-se à mão depois de cada boot com graphs.' },
    { q: 'Na lane TensorFold do GLM, troca-se o DFlash2 por um rascunhador muito pior, mantendo a mesma chave (semente) e \\(T = 1\\). O que acontece com a resposta?', o: ['Muda, mas continua com a distribuição do alvo, como na amostragem por rejeição', 'Fica idêntica, bit a bit: um rascunho só é aceito se for igual à amostra chaveada do alvo naquela posição, então o rascunhador decide quantos tokens saem por rodada, nunca quais', 'Fica idêntica só a \\(T = 0\\); a \\(T = 1\\) ela muda', 'Piora em qualidade, porque o alvo passa a aceitar rascunhos ruins'], a: 1, e: 'Com amostragem chaveada, o token da posição \\(t\\) é \\(\\arg\\max(z/T + g)\\), e \\(g\\) sai de um hash de (chave, posição, token): está decidido antes de qualquer proposta. A verificação aceita enquanto rascunho = amostra, e os kernels independentes de linha garantem que os logits são os da decodificação serial. Um rascunhador pior só aceita menos, ou seja, menos tokens por rodada. A opção A descreve a rejeição do vLLM, que preserva a distribuição, não a sequência; a C confunde as duas regras, que só coincidem a \\(T = 0\\).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('.', ',');
    /** tira o uppercase do CSS de rótulos com símbolos matemáticos (α, τ, k, q…) */
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem-token: quadrado arredondado com olhinhos */
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: o.stroke || col, lw: 1.2 });
      if (s >= 11) {
        const eye = C.dark ? '#06140d' : '#ffffff';
        const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
        V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
        g.beginPath();
        if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
        else if (o.mood === 'o') g.arc(x, y + s * 0.16, s * 0.07, 0, Math.PI * 2);
        else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
        g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
      }
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle' });
    }
    function mark(g, x, y, ok, s = 7) {
      const C = V.C();
      g.lineWidth = 2.4; g.strokeStyle = ok ? C.a : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    function wrapText(g, s, x, y, maxW, lh, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(' ');
      let line = '', yy = y;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); line = wd; yy += lh; g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); }
        else line = test;
      }
      if (line) V.text(g, line, x, yy, o);
      return yy + lh;
    }
    function niceStep(range, n = 4) {
      const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }
    const sampleCat = (d, u) => { let s = 0; for (let i = 0; i < d.length; i++) { s += d[i]; if (u < s) return i; } return d.length - 1; };

    /* ================================================ FIG 1 — o passo por dentro */
    (function () {
      const fig = root.querySelector('#fig-passo');
      adaptH(fig, 420, 640);
      const MiB = 1048576, BW = 230e9, FL = 125e12, E = 288, KR = 8;
      const Dexp = (M) => E * (1 - Math.pow(1 - 1 / E, M * KR));
      const MODELS = {
        bf16: { bytes: () => 10e9, flops: (M) => 2 * 5e9 * M },
        q4: { bytes: () => 2.5e9, flops: (M) => 2 * 5e9 * M },
        moe: { bytes: (M) => 8e9 + 42 * 6 * MiB * Dexp(M), flops: (M) => 2 * 8.2e9 * M },
      };
      const tm = (m, M) => (MODELS[m].bytes(M) / BW) * 1000;
      const tc = (m, M) => (MODELS[m].flops(M) / FL) * 1000;
      const tt = (m, M) => Math.max(tm(m, M), tc(m, M));
      const Mvals = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64];
      const S = { M: 8, model: 'moe' };
      const rr = V.rng(35);
      const rowsExp = Array.from({ length: 64 }, () => { const s = new Set(); while (s.size < KR) s.add(rr.int(E)); return Array.from(s); });
      let lit = new Set(), litArr = [];
      const resample = () => { lit = new Set(); for (let i = 0; i < S.M; i++) rowsExp[i].forEach((e) => lit.add(e)); litArr = Array.from(lit); };
      resample();
      const Wd = V.randMatrix(6, 8, 7);
      const Xd = V.randMatrix(8, 4, 11);
      const parts = Array.from({ length: 40 }, (_, i) => ({ ph: (i * 0.618) % 1, jx: rr() }));
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'linhas M', min: 0, max: Mvals.length - 1, step: 1, value: Mvals.indexOf(8), fmt: (v) => String(Mvals[v]), onInput: (v) => { S.M = Mvals[v]; resample(); upd(); } });
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'bf16', label: 'denso BF16' }, { v: 'q4', label: 'denso 4 bits' }, { v: 'moe', label: 'MoE estilo GLM' }], value: 'moe', onChange: (v) => { S.model = v; upd(); } });
      const rB = UI.readout(ctl, 'pesos lidos', '—'), rT = UI.readout(ctl, 'tempo do passo', '—'), rR = UI.readout(ctl, 'custo vs M=1', '—'), rS = UI.readout(ctl, 'tok/s se tudo aceito', '—');
      function upd() {
        rB.set(fmt(MODELS[S.model].bytes(S.M) / 1e9, 1) + ' GB');
        rT.set(fmt(tt(S.model, S.M), 1) + ' ms');
        rR.set(fmt(tt(S.model, S.M) / tt(S.model, 1), 2) + '×');
        rS.set(String(Math.round((S.M / tt(S.model, S.M)) * 1000)));
      }
      upd();

      function drawScene(g, A, t, C) {
        const moe = S.model === 'moe';
        const nR = Math.min(S.M, 8);
        const charW = 22, opW = 20, nX = 4, nWc = 8, nY = 8;
        const memY = A.y + 4, memH = 26;
        const top = memY + memH + 34, bottom = A.y + A.h - 30, Hm = bottom - top;
        const cs = Math.max(6, Math.min((A.w - charW - 2 * opW - 4) / (nX + nWc + nY) - 2, (Hm - 20) / 8 - 2, 20));
        const step = cs + 2;
        const xX = A.x + charW, xW = xX + nX * step + opW, xY = xW + nWc * step + opW;
        const Wwid = nWc * step - 2;
        const cyM = top + Hm / 2;
        const rowsH = nR * step - 2;
        const yRows = cyM - rowsH / 2;
        const ph = (t % 2.4) / 2.4;
        // memória (fonte dos bytes)
        const mx = Math.max(A.x, xW - 34), mw = Math.min(A.x + A.w, xW + Wwid + 34) - mx;
        V.box(g, mx, memY, mw, memH, { r: 7, fill: V.alpha(C.a, 0.12), stroke: V.alpha(C.a, 0.6), lw: 1.2 });
        V.text(g, moe ? 'memória · só os especialistas chamados' : 'memória · cada peso lido 1× por passo', mx + mw / 2, memY + memH / 2 + 4, { size: 11.5, font: 'mono', color: C.ink, align: 'center', maxW: mw - 10 });
        // X (linhas) com personagens: âncora + rascunhos
        for (let i = 0; i < nR; i++) {
          const y = yRows + i * step;
          for (let j = 0; j < nX; j++) { g.fillStyle = V.cmap(0.35 + 0.4 * Math.abs(Math.tanh(Xd[i][j])), 'seq'); V.rr(g, xX + j * step, y, cs, cs, Math.min(3, cs / 4)); g.fill(); }
          tokChar(g, A.x + charW / 2, y + cs / 2, Math.min(16, cs + 2), i === 0 ? C.d : C.b, { mood: i === 0 ? 'o' : undefined });
        }
        V.text(g, 'X', xX + (nX * step) / 2, yRows - 9, { size: 12, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, S.M > nR ? '+' + (S.M - nR) + ' linhas' : 'âncora + rascunhos', xX + (nX * step) / 2 - 6, yRows + rowsH + 16, { size: 11, font: 'mono', color: C.b, align: 'center', maxW: nX * step + 40 });
        V.text(g, '×', xX + nX * step + opW / 2 - 1, cyM + 5, { size: 16, color: C.muted, align: 'center' });
        V.text(g, '=', xW + nWc * step + opW / 2 - 1, cyM + 5, { size: 16, color: C.muted, align: 'center' });
        // W (denso) ou grade de especialistas (MoE)
        let targetX = [], Wy;
        const colIdx = Math.floor(ph * nWc);
        if (!moe) {
          const R = 6;
          Wy = cyM - (R * step - 2) / 2;
          V.matrix(g, { x: xW, y: Wy, data: Wd, cell: cs, gap: 2, cmap: 'div', hiCol: colIdx, hiColor: C.d, r: Math.min(3, cs / 4) });
          V.text(g, 'Wᵀ', xW + Wwid / 2, Wy - 9, { size: 12, font: 'mono', color: C.muted, align: 'center' });
          targetX = [xW + colIdx * step + cs / 2];
        } else {
          const GC = 16, GR = 18;
          const cw = Math.max(2, Wwid / GC - 1), st2 = cw + 1;
          Wy = cyM - (GR * st2) / 2;
          const sweepRow = Math.floor(ph * GR);
          for (let r = 0; r < GR; r++) for (let c = 0; c < GC; c++) {
            const e = r * GC + c, on = lit.has(e);
            g.fillStyle = on ? (r === sweepRow ? C.d : V.alpha(C.a, 0.9)) : V.alpha(C.muted, 0.14);
            g.fillRect(xW + c * st2, Wy + r * st2, cw, cw);
          }
          V.text(g, `${lit.size} de 288`, xW + Wwid / 2, Wy - 9, { size: 11.5, font: 'mono', weight: 600, color: C.a, align: 'center', maxW: Wwid + 30 });
          for (let q = 0; q < Math.min(12, litArr.length); q++) { const e = litArr[(q * 7 + Math.floor(t * 3)) % litArr.length]; targetX.push(xW + (e % GC) * st2 + cw / 2); }
        }
        // fluxo de bytes (partículas) da memória para os pesos
        const nP = moe ? Math.min(40, 8 + Math.round(lit.size / 3)) : 14;
        for (let k = 0; k < nP; k++) {
          const P = parts[k], u = (P.ph + t * 0.8) % 1;
          const tx = targetX[k % targetX.length] + (P.jx - 0.5) * 6;
          const sx = mx + 12 + P.jx * (mw - 24);
          const x = V.lerp(sx, tx, u), y = V.lerp(memY + memH + 2, Wy - 14, u);
          V.dot(g, x, y, 1.9, V.alpha(C.a, 0.3 + 0.55 * (1 - u)));
        }
        // Y preenchendo em sincronia com a varredura
        const fillIdx = moe ? Math.floor(ph * nY) : colIdx;
        for (let i = 0; i < nR; i++) for (let j = 0; j < nY; j++) {
          const y = yRows + i * step, x = xY + j * step;
          const done = j < fillIdx, now = j === fillIdx;
          g.fillStyle = now ? V.alpha(C.d, 0.9) : done ? V.alpha(C.c, 0.55) : V.alpha(C.muted, 0.12);
          V.rr(g, x, y, cs, cs, Math.min(3, cs / 4)); g.fill();
        }
        V.text(g, 'Y', xY + (nY * step) / 2, yRows - 9, { size: 12, font: 'mono', color: C.muted, align: 'center' });
        const msg = moe ? `M = ${S.M}: ${lit.size} especialistas distintos por camada` : `M = ${S.M}: a mesma leitura de W serve às ${S.M} linhas`;
        V.text(g, msg, A.x + A.w / 2, A.y + A.h - 8, { size: 12, color: C.ink, align: 'center', maxW: A.w - 4 });
      }

      function drawPlot(g, P, C) {
        const m = S.model;
        const L = P.x + 42, R = P.x + P.w - 10, T = P.y + 34, B = P.y + P.h - 30;
        const xs = (M) => L + (Math.log2(M) / 10) * (R - L);
        let ymax = 0; for (let e = 0; e <= 10; e += 0.25) ymax = Math.max(ymax, tt(m, Math.pow(2, e)));
        ymax *= 1.12;
        const ys = (v) => B - (v / ymax) * (B - T);
        V.text(g, 'tempo do passo (ms)', P.x + 2, P.y + 10, { size: 11.5, font: 'mono', color: C.muted });
        // legenda
        const lg = [[C.a, 'ler pesos', null], [C.b, 'contas', [5, 4]], [C.d, 'passo', null]];
        let lx = P.x + 2; const ly = P.y + 24;
        lg.forEach(([c, l, d]) => { V.line(g, lx, ly - 4, lx + 16, ly - 4, { color: c, lw: 2.5, dash: d }); V.text(g, l, lx + 20, ly, { size: 11, color: C.muted }); lx += 30 + tw(g, l, 11); });
        g.fillStyle = V.alpha(C.b, 0.08); g.fillRect(xs(2), T, xs(64) - xs(2), B - T);
        V.text(g, 'faixa da especulação', (xs(2) + xs(64)) / 2, T + 13, { size: 11, align: 'center', color: C.b });
        const stp = niceStep(ymax, 4);
        for (let v = 0; v <= ymax + 1e-9; v += stp) { V.line(g, L, ys(v), R, ys(v), { color: C.grid, lw: 1 }); V.text(g, String(Math.round(v)), L - 6, ys(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        [1, 4, 16, 64, 256, 1024].forEach((M) => { V.line(g, xs(M), B, xs(M), B + 4, { color: C.muted, lw: 1 }); V.text(g, String(M), xs(M), B + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        V.text(g, 'linhas M (log)', R, B + 28, { size: 11, color: C.muted, align: 'right' });
        const curve = (fn, col, lw, dash) => { g.beginPath(); for (let e = 0; e <= 10.0001; e += 0.1) { const M = Math.pow(2, e); const x = xs(M), y = ys(fn(M)); if (e === 0) g.moveTo(x, y); else g.lineTo(x, y); } g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        curve((M) => tm(m, M), C.a, 2);
        curve((M) => tc(m, M), C.b, 2, [5, 4]);
        curve((M) => tt(m, M), C.d, 3.2);
        let knee = null; for (let e = 0; e <= 10; e += 0.02) { const M = Math.pow(2, e); if (tc(m, M) >= tm(m, M)) { knee = M; break; } }
        if (knee) { V.line(g, xs(knee), T + 18, xs(knee), B, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] }); V.text(g, 'joelho ≈ ' + Math.round(knee), xs(knee) - 4, T + 30, { size: 11, font: 'mono', color: C.ink, align: 'right' }); }
        const x = xs(S.M), y = ys(tt(m, S.M));
        V.glow(g, x, y, 14, C.d, 0.8); V.dot(g, x, y, 4.5, C.d);
        V.text(g, `${fmt(tt(m, S.M), 1)} ms`, x + 8, y - 8, { size: 12, font: 'mono', weight: 600, color: C.ink });
      }

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const A = wide ? { x: 8, y: 6, w: w * 0.54 - 8, h: h - 12 } : { x: 6, y: 6, w: w - 12, h: h * 0.5 - 6 };
        const P = wide ? { x: w * 0.56, y: 8, w: w * 0.44 - 12, h: h - 16 } : { x: 6, y: h * 0.52, w: w - 12, h: h * 0.48 - 8 };
        drawScene(g, A, t, C);
        drawPlot(g, P, C);
      });
    })();

    /* ================================================ FIG 2 — uma rodada (stepper) */
    (function () {
      const fig = root.querySelector('#fig-rodada');
      adaptH(fig, 440, 500);
      const CTX = ['O', 'gato', 'subiu'];
      const DR = [{ t: 'telhado', q: 0.55 }, { t: ',', q: 0.6 }, { t: 'miou', q: 0.5 }, { t: 'alto', q: 0.4 }];
      const POS = [
        { c: ['telhado', 'muro', 'sofá', 'carro'], p: [0.62, 0.21, 0.11, 0.06], q: [0.55, 0.25, 0.12, 0.08] },
        { c: [',', 'e', '.', 'de'], p: [0.3, 0.45, 0.17, 0.08], q: [0.6, 0.25, 0.1, 0.05] },
        { c: ['miou', 'pulou', 'e', 'dormiu'], p: [0.35, 0.3, 0.2, 0.15], q: [0.5, 0.2, 0.2, 0.1] },
        { c: ['alto', 'forte', 'de', 'e'], p: [0.4, 0.25, 0.2, 0.15], q: [0.4, 0.3, 0.15, 0.15] },
        { c: ['.', 'e', ',', '!'], p: [0.5, 0.25, 0.15, 0.1], q: null },
      ];
      const SUB = ['₁', '₂', '₃', '₄', '₅'];
      const steps = [
        { t: 'O ponto de partida', d: 'O texto confirmado é “O gato subiu no”. O último token, <b>no</b>, já foi emitido mas ainda não passou pelo alvo: é a <b>âncora</b> desta rodada.' },
        { t: 'O rascunhador chuta k = 4 tokens', d: 'Um rascunhador barato propõe <b>telhado</b>, <b>,</b>, <b>miou</b>, <b>alto</b>, cada um com a probabilidade \\(q\\) que ele mesmo deu. Sequencial (MTP, EAGLE): 4 forwards pequenos. Em bloco (DFlash, DSpark): um só.' },
        { t: 'Um forward do alvo, cinco posições', d: 'O alvo processa [no, telhado, “,”, miou, alto] de uma vez. Com a máscara causal, a posição \\(i\\) devolve \\(p_{i+1}\\), a distribuição do token seguinte dado o prefixo: as mesmas 5 distribuições que 5 passos de decode dariam, pelo custo de ~1 leitura dos pesos.' },
        { t: 'Posição 1: aceito', d: '\\(p(\\text{telhado}) = 0{,}62 \\ge q = 0{,}55\\), então \\(\\min(1, p/q) = 1\\): aceito com certeza. O alvo gosta de “telhado” mais do que o rascunhador.' },
        { t: 'Posição 2: sorteio e rejeição', d: '\\(p(\\text{“,”}) = 0{,}30\\) contra \\(q = 0{,}60\\): o rascunhador foi otimista demais. A chance de aceitar é \\(0{,}30/0{,}60 = 0{,}5\\). O sorteio deu \\(u = 0{,}71 \\gt 0{,}5\\): <b>rejeitado</b>, e a rodada para aqui.' },
        { t: 'O resíduo corrige', d: 'O alvo sorteia de \\(\\max(0, p-q)\\): “e” 0,20, “.” 0,07, “de” 0,03, normalizados para 0,67 / 0,23 / 0,10. Saiu <b>e</b>. Os rascunhos 3 e 4 são descartados com as suas linhas de KV. Resultado: <b>2 tokens</b> (“telhado e”) num único passo do alvo.' },
        { t: 'Se todos passassem: o token bônus', d: 'Num universo em que os 4 rascunhos passam, o alvo ainda sorteia um 5º token de \\(p_5\\), que já estava calculada: <b>5 tokens</b> por passo, o máximo com \\(k=4\\). A próxima rodada começa do último token emitido.' },
      ];
      let H2 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H2 && H2.redraw() });

      function textLine(g, toks, x0, y, maxW) {
        const C = V.C();
        let x = x0, yy = y;
        toks.forEach((tk) => {
          const wd = tw(g, tk.s, 12, 'mono', 600) + 14;
          if (x + wd > x0 + maxW) { x = x0; yy += 26; }
          const col = tk.k === 'ok' ? C.a : tk.k === 'fix' || tk.k === 'bonus' ? C.d : tk.k === 'anc' ? C.muted : null;
          chip(g, x, yy, wd, 21, tk.s, { size: 12, weight: 600, fill: col ? V.alpha(col, 0.2) : V.alpha(C.muted, 0.08), stroke: col || V.alpha(C.muted, 0.35), color: C.ink });
          if (tk.k === 'bonus') V.text(g, '★', x + wd - 2, yy + 2, { size: 11, color: C.d, align: 'center' });
          x += wd + 5;
        });
      }
      function distChart(g, x, y, w, h, cands, p, q, o = {}) {
        const C = V.C();
        const n = cands.length, gw = w / n;
        const mx = Math.max(...p, ...(q || [0]), ...(o.r || [0])) * 1.15;
        const bh = h - 34;
        const base = y + bh + 6;
        for (let i = 0; i < n; i++) {
          const cx = x + gw * (i + 0.5);
          const bw = Math.min(26, gw * 0.3);
          const hi = o.hi === i;
          if (o.r) {
            const v = o.r[i], hh = (v / mx) * bh;
            V.box(g, cx - bw, base - hh, bw * 2, Math.max(1, hh), { r: 3, fill: V.alpha(C.d, hi ? 0.9 : 0.5), stroke: C.d });
            V.text(g, v > 0 ? fmt(v, 2) : '0', cx, base - hh - 5, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          } else {
            const hp = (p[i] / mx) * bh;
            if (!o.qOnly) {
              V.box(g, cx - bw - 1, base - hp, bw, Math.max(1, hp), { r: 3, fill: V.alpha(C.a, hi ? 0.95 : 0.5), stroke: C.a });
              V.text(g, fmt(p[i], 2), cx - bw / 2 - 1, base - hp - 5, { size: 11, font: 'mono', color: C.ink, align: 'center' });
            }
            if (q) {
              const hq = (q[i] / mx) * bh;
              const qx = o.qOnly ? cx - bw / 2 : cx + 1;
              V.box(g, qx, base - hq, bw, Math.max(1, hq), { r: 3, fill: V.alpha(C.b, hi ? 0.55 : 0.25), stroke: C.b });
              V.text(g, fmt(q[i], 2), qx + bw / 2, base - hq - 5, { size: 11, font: 'mono', color: C.b, align: 'center' });
            }
          }
          V.text(g, cands[i], cx, base + 15, { size: 12, font: 'mono', weight: hi ? 700 : 400, color: hi ? C.ink : C.muted, align: 'center' });
          if (hi) V.rr(g, cx - gw * 0.42, y - 2, gw * 0.84, h - 2, 8), (g.strokeStyle = V.alpha(C.d, 0.6)), (g.lineWidth = 1.5), g.stroke();
        }
        V.line(g, x, base, x + w, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
      }

      H2 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const pad = 14, colW = (w - 2 * pad) / 5;
        const cx = (c) => pad + colW * (c + 0.5);
        const narrow = w < 560;
        // 1) linha do texto emitido
        let toks = CTX.map((s) => ({ s, k: 'ctx' })).concat([{ s: 'no', k: 'anc' }]);
        if (i === 5) toks = toks.concat([{ s: 'telhado', k: 'ok' }, { s: 'e', k: 'fix' }]);
        if (i === 6) toks = toks.concat(DR.map((d) => ({ s: d.t, k: 'ok' }))).concat([{ s: '.', k: 'bonus' }]);
        V.text(g, 'texto', pad, 24, { size: 11, font: 'mono', color: C.muted });
        textLine(g, toks, pad + 42, 10, w - 2 * pad - 42);
        const emitted = i === 5 ? 2 : i === 6 ? 5 : 0;
        if (emitted) V.text(g, `+${emitted} tokens neste passo`, w - pad, 56, { size: 12, weight: 700, color: C.a, align: 'right' });
        // 2) personagens
        const yc = narrow ? 102 : 96, s = Math.min(30, colW * 0.42);
        if (i === 0) {
          for (let c = 1; c <= 4; c++) {
            g.setLineDash([4, 4]); V.rr(g, cx(c) - s / 2, yc - s / 2, s, s, s * 0.3); g.strokeStyle = V.alpha(C.b, 0.6); g.lineWidth = 1.4; g.stroke(); g.setLineDash([]);
            V.text(g, '?', cx(c), yc + 5, { size: 14, weight: 700, color: V.alpha(C.b, 0.8), align: 'center' });
          }
          V.text(g, 'k = 4 vagas de rascunho', (cx(1) + cx(4)) / 2, yc - s / 2 - 7, { size: 11, color: C.b, align: 'center' });
          const by0 = yc + s / 2 + 54;
          g.setLineDash([5, 5]); V.rr(g, pad, by0, w - 2 * pad, 32, 9); g.strokeStyle = V.alpha(C.a, 0.45); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]);
          V.text(g, 'alvo: ainda não rodou nesta rodada', w / 2, by0 + 20, { size: 11.5, color: V.alpha(C.a, 0.8), align: 'center' });
        }
        tokChar(g, cx(0), yc, s, C.muted, { fill: V.alpha(C.muted, 0.45) });
        V.text(g, 'no', cx(0), yc + s / 2 + 15, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
        V.text(g, 'âncora', cx(0), yc - s / 2 - 7, { size: 11, color: C.muted, align: 'center' });
        if (i >= 1) {
          if (i <= 2) V.text(g, 'rascunhos (q)', (cx(1) + cx(4)) / 2, yc - s / 2 - 7, { size: 11, color: C.b, align: 'center' });
          DR.forEach((d, j) => {
            const c = j + 1;
            let appear = i === 1 ? V.clamp((el - j * 0.35) / 0.3, 0, 1) : 1;
            if (appear <= 0) return;
            let col = C.b, mood, alpha = appear, dy = (1 - V.ease.back(appear)) * -18;
            if (i === 6) col = C.a;
            else if (i >= 3 && j === 0) col = C.a;
            if (i >= 4 && i <= 5 && j === 1) { col = RED(); mood = 'sad'; }
            if (i === 5 && j >= 2) { alpha = Math.max(0.25, 1 - el * 0.8); mood = 'sad'; col = C.muted; }
            tokChar(g, cx(c), yc + dy, s, col, { mood, alpha });
            g.globalAlpha = alpha;
            V.text(g, d.t, cx(c), yc + dy + s / 2 + 15, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
            V.text(g, 'q ' + fmt(d.q, 2), cx(c), yc + dy + s / 2 + 30, { size: 11, font: 'mono', color: C.b, align: 'center' });
            g.globalAlpha = 1;
          });
        }
        // 3) faixa do alvo
        const by = yc + s / 2 + 54, bh = 32;
        if (i >= 2) {
          const pulse = i === 2 ? 0.5 + 0.5 * Math.sin(el * 6) : 0;
          V.box(g, pad, by, w - 2 * pad, bh, { r: 9, fill: V.alpha(C.a, 0.1 + 0.15 * pulse), stroke: V.alpha(C.a, 0.7), lw: 1.5 });
          for (let c = 0; c < 5; c++) {
            if (i === 2) V.line(g, cx(c), yc + s / 2 + 36, cx(c), by, { color: V.alpha(C.a, 0.3 + 0.5 * pulse), lw: 2 });
            V.text(g, 'p' + SUB[c], cx(c), by + bh / 2 + 4, { size: 12, font: 'mono', weight: 600, color: C.a, align: 'center' });
            if (c < 4) V.arrow(g, cx(c) + 12, by + bh / 2 - 6, cx(c + 1) - 12, by + bh / 2 - 6, { color: V.alpha(C.a, 0.35), lw: 1, head: 5, curve: -6 });
          }
          V.text(g, narrow ? 'alvo · 1 forward · 5 posições' : 'alvo · 1 forward · 5 posições (máscara causal)', pad + 4, by - 6, { size: 11, color: C.a });
          // valores p(d_c) e marcas
          // p que o alvo dá a cada rascunho: rotulado pelo token (p(telhado)), não pelo índice da faixa
          const full = colW >= 128;
          if (!full) V.text(g, 'p(rasc.)', cx(0), by + bh + 15, { size: 11, font: 'mono', color: V.alpha(C.a, 0.8), align: 'center' });
          for (let c = 1; c <= 4; c++) {
            const pv = POS[c - 1].p[0];
            const lab = full ? `p(${DR[c - 1].t}) ${fmt(pv, 2)}` : fmt(pv, 2);
            V.text(g, lab, cx(c), by + bh + 15, { size: 11, font: 'mono', color: C.a, align: 'center' });
            const labW = tw(g, lab, 11, 'mono');
            let st2 = null;
            if (i === 6) st2 = true;
            else if (i >= 3 && c === 1) st2 = true;
            else if (i >= 4 && c === 2) st2 = i === 4 ? (el > 1.6 ? false : null) : false;
            if (st2 != null) mark(g, Math.min(cx(c) + labW / 2 + 11, w - pad - 7), by + bh + 11, st2, 6);
          }
          if (i === 6) V.text(g, 'bônus', cx(4) + 4, by + bh + 30, { size: 11, weight: 700, color: C.d, align: 'center' });
        }
        // 4) painel de detalhe
        const py = by + bh + 38, ph = h - py - 8;
        if (ph < 90) return;
        V.box(g, pad, py, w - 2 * pad, ph, { r: 10, fill: V.alpha(C.panel || C.muted, 0.18), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        const ix = pad + 12, iw = w - 2 * pad - 24;
        let title = '', gauge = false;
        if (i === 0) {
          const yEnd = wrapText(g, 'A âncora já faz parte do texto, mas o alvo ainda não calculou nada sobre ela: o KV dela ainda não existe. Por isso ela entra no forward desta rodada como a posição 0.', ix, py + 24, iw, 18, { size: 12.5, color: C.ink });
          const ky = yEnd + 8;
          V.text(g, 'KV do alvo', ix, ky + 15, { size: 11, font: 'mono', color: C.muted });
          let kx = ix + 84;
          CTX.forEach((c) => { const ww = tw(g, c, 12, 'mono', 600) + 16; chip(g, kx, ky, ww, 22, c, { size: 12, weight: 600, fill: V.alpha(C.a, 0.2), stroke: C.a }); kx += ww + 5; });
          const ww0 = tw(g, 'no', 12, 'mono', 600) + 16;
          g.setLineDash([4, 3]); V.rr(g, kx, ky, ww0, 22, 6); g.strokeStyle = C.d; g.lineWidth = 1.4; g.stroke(); g.setLineDash([]);
          V.text(g, 'no', kx + ww0 / 2, ky + 15, { size: 12, font: 'mono', weight: 600, color: C.d, align: 'center' });
          if (kx + ww0 + 110 < ix + iw) V.text(g, '← ainda não', kx + ww0 + 8, ky + 15, { size: 11.5, color: C.d });
          return;
        }
        let chartW = iw, chartX = ix;
        if (i === 4 && !narrow) { chartW = iw * 0.62; gauge = true; }
        if (i === 1) { title = 'o que o rascunhador acha da posição 1 (q)'; distChart(g, chartX, py + 30, chartW, ph - 40, POS[0].c, POS[0].p, POS[0].q, { qOnly: true, hi: 0 }); }
        else if (i === 2 || i === 3) { title = i === 2 ? 'posição 1: alvo p (menta) × rascunho q (lilás)' : 'min(1, 0,62 / 0,55) = 1 → aceita'; distChart(g, chartX, py + 30, chartW, ph - 40, POS[0].c, POS[0].p, POS[0].q, { hi: 0 }); }
        else if (i === 4) { title = 'posição 2: 0,30 / 0,60 = 0,5 de chance'; distChart(g, chartX, py + 30, chartW, ph - 40, POS[1].c, POS[1].p, POS[1].q, { hi: 0 }); if (narrow) gauge = true; }
        else if (i === 5) {
          title = 'resíduo max(0, p − q), normalizado → saiu “e”';
          const r = POS[1].p.map((pv, k) => Math.max(0, pv - POS[1].q[k])); const sm = r.reduce((a, b) => a + b, 0);
          distChart(g, chartX, py + 30, chartW, ph - 40, POS[1].c, POS[1].p, null, { r: r.map((x) => x / sm), hi: 1 });
        } else if (i === 6) { title = 'p₅: de onde sai o token bônus → “.”'; distChart(g, chartX, py + 30, chartW, ph - 40, POS[4].c, POS[4].p, null, { hi: 0 }); }
        V.text(g, title, ix, py + 18, { size: 12, weight: 600, color: C.ink, maxW: iw });
        if (gauge) {
          // termômetro do sorteio u
          const gx = narrow ? ix + iw * 0.55 : ix + iw * 0.7, gy = narrow ? py + 50 : py + 50;
          const gw2 = narrow ? iw * 0.42 : iw * 0.28, gh = 14;
          const roll = V.clamp(el / 1.4, 0, 1);
          const u = roll < 1 ? V.pingpong(el * 1.7, 1) : 0.71;
          V.box(g, gx, gy, gw2 * 0.5, gh, { r: 4, fill: V.alpha(C.a, 0.45) });
          V.box(g, gx + gw2 * 0.5, gy, gw2 * 0.5, gh, { r: 4, fill: V.alpha(RED(), 0.35) });
          V.text(g, 'aceita', gx + gw2 * 0.25, gy + gh + 13, { size: 11, color: C.a, align: 'center' });
          V.text(g, 'rejeita', gx + gw2 * 0.75, gy + gh + 13, { size: 11, color: RED(), align: 'center' });
          const ux = gx + u * gw2;
          V.line(g, ux, gy - 6, ux, gy + gh + 2, { color: C.ink, lw: 2.5 });
          V.text(g, 'u = ' + fmt(u, 2), ux, gy - 9, { size: 11.5, font: 'mono', weight: 700, color: C.ink, align: 'center' });
        }
      });
    })();

    /* ================================================ FIG 3 — dardos */
    (function () {
      const fig = root.querySelector('#fig-dardos');
      adaptH(fig, 460, 500);
      const VOC = ['gato', 'cão', 'sol', 'mar', 'pão', 'rio'];
      const P3 = [0.34, 0.24, 0.16, 0.12, 0.08, 0.06];
      const W3 = [0.04, 0.08, 0.4, 0.06, 0.32, 0.1];
      const S = { lam: 0.45, greedy: false, darts: [], counts: [0, 0, 0, 0, 0, 0], n: 0, acc: 0, pend: 0, frame: 0 };
      let Q = [], R = [], alpha = 1;
      const rnd = V.rng(3535);
      function recompute() {
        let q = P3.map((p, i) => (1 - S.lam) * p + S.lam * W3[i]);
        const s = q.reduce((a, b) => a + b, 0); q = q.map((x) => x / s);
        if (S.greedy) { const m = q.indexOf(Math.max(...q)); q = q.map((_, i) => (i === m ? 1 : 0)); }
        Q = q;
        const r = P3.map((p, i) => Math.max(0, p - Q[i])); const sr = r.reduce((a, b) => a + b, 0);
        R = sr > 1e-12 ? r.map((x) => x / sr) : P3.slice();
        alpha = P3.reduce((a, p, i) => a + Math.min(p, Q[i]), 0);
        S.darts = []; S.counts = [0, 0, 0, 0, 0, 0]; S.n = 0; S.acc = 0;
        updR();
      }
      function throwOne(tNow, anim) {
        const x = sampleCat(Q, rnd()), u = rnd(), hq = u * Q[x];
        const acc = hq < P3[x];
        let y = x, h2 = 0;
        if (!acc) { y = sampleCat(R, rnd()); h2 = Q[y] + rnd() * Math.max(0, P3[y] - Q[y]); }
        S.n++; if (acc) S.acc++; S.counts[y]++;
        if (anim) { S.darts.push({ x, hq, acc, y, h2, t0: tNow, jx: rnd() - 0.5, jy: rnd() - 0.5 }); if (S.darts.length > 70) S.darts.shift(); }
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'desalinhamento', min: 0, max: 1, step: 0.05, value: S.lam, fmt: (v) => fmt(v, 2), onInput: (v) => { S.lam = v; recompute(); } });
      UI.toggle(ctl, { label: 'rascunho greedy (q one-hot)', value: false, onChange: (v) => { S.greedy = v; recompute(); } });
      UI.button(ctl, '+500 dardos', () => { for (let k = 0; k < 500; k++) throwOne(0, false); updR(); }, { primary: true });
      UI.button(ctl, 'zerar', () => recompute());
      const rA = nocase(UI.readout(ctl, 'α teórico', '—')), rO = nocase(UI.readout(ctl, 'α observado', '—')), rE = nocase(UI.readout(ctl, 'erro máx |f − p|', '—'));
      function updR() {
        rA.set(fmt(alpha, 3));
        rO.set(S.n ? fmt(S.acc / S.n, 3) : '—');
        rE.set(S.n ? fmt(Math.max(...S.counts.map((c, i) => Math.abs(c / S.n - P3[i]))), 3) : '—');
      }
      recompute();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused) { S.pend += dt * 6; while (S.pend >= 1) { S.pend -= 1; throwOne(t, true); } }
        if (++S.frame % 8 === 0) updR();
        const pad = 16, L = pad + 6, Rr = w - pad;
        const cw = (Rr - L) / 6, bw = Math.min(58, cw * 0.62);
        const cTop = (w < 520 ? 94 : 80), cBot = h * 0.55;
        const ymax = Math.max(...P3, ...Q) * 1.08;
        const yv = (v) => cBot - (v / ymax) * (cBot - cTop);
        V.text(g, `α = Σ min(p, q) = ${fmt(alpha, 3)}   ·   TV(p, q) = ${fmt(1 - alpha, 3)}`, pad, 18, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: w - 2 * pad });
        let legendBottom = 36;
        {
          const items = [[C.a, 'p (alvo)'], [C.b, 'q (rascunho)'], [C.e || C.a, 'aceito'], [RED(), 'rejeição'], [C.d, 'resíduo']];
          let lx = pad, ly = 38;
          items.forEach(([c, l]) => {
            const ww = tw(g, l, 11) + 24;
            if (lx + ww > w - pad) { lx = pad; ly += 16; }
            V.dot(g, lx + 5, ly - 4, 5, c); V.text(g, l, lx + 13, ly, { size: 11, color: C.muted }); lx += ww;
          });
          legendBottom = ly;
        }
        for (let i = 0; i < 6; i++) {
          const x0 = L + cw * i + (cw - bw) / 2;
          const p = P3[i], q = Q[i], mn = Math.min(p, q);
          // sobreposição
          g.fillStyle = V.alpha(C.e || C.a, 0.42); g.fillRect(x0, yv(mn), bw, cBot - yv(mn));
          if (q > p) { g.fillStyle = V.alpha(RED(), 0.28); g.fillRect(x0, yv(q), bw, yv(p) - yv(q)); }
          if (p > q) { g.fillStyle = V.alpha(C.d, 0.4); g.fillRect(x0, yv(p), bw, yv(q) - yv(p)); }
          g.lineWidth = 2; g.strokeStyle = C.a; g.strokeRect(x0, yv(p), bw, cBot - yv(p));
          g.setLineDash([4, 3]); g.strokeStyle = C.b; g.strokeRect(x0 + 1.5, yv(q), bw - 3, cBot - yv(q)); g.setLineDash([]);
          const topv = Math.max(p, q);
          V.text(g, 'p ' + fmt(p, 2), x0 + bw / 2, yv(topv) - 18, { size: 11, font: 'mono', color: C.a, align: 'center' });
          V.text(g, 'q ' + fmt(q, 2), x0 + bw / 2, yv(topv) - 5, { size: 11, font: 'mono', color: C.b, align: 'center' });
          V.text(g, VOC[i], x0 + bw / 2, cBot + 15, { size: 12, font: 'mono', color: C.ink, align: 'center' });
        }
        V.line(g, L, cBot, Rr, cBot, { color: V.alpha(C.muted, 0.6), lw: 1 });
        // dardos
        S.darts.forEach((d) => {
          const age = t - d.t0; if (age < 0) return;
          const fade = V.clamp(1 - (age - 1) / 6, 0, 1);
          const x0 = L + cw * d.x + cw / 2 + d.jx * bw * 0.7;
          const y0 = yv(d.hq);
          const f1 = V.clamp(age / 0.45, 0, 1);
          const sx = x0 - 30, sy = cTop - 26;
          const px = V.lerp(sx, x0, V.ease.out(f1)), py2 = V.lerp(sy, y0, V.ease.out(f1));
          if (f1 < 1) { V.line(g, V.lerp(sx, px, 0.6), V.lerp(sy, py2, 0.6), px, py2, { color: V.alpha(C.ink, 0.5), lw: 1.5 }); V.dot(g, px, py2, 3, C.ink); return; }
          g.globalAlpha = fade;
          V.dot(g, x0, y0, 3.2, d.acc ? (C.e || C.a) : RED());
          if (!d.acc) {
            const f2 = V.clamp((age - 0.45) / 0.4, 0, 1);
            const x1 = L + cw * d.y + cw / 2 + d.jy * bw * 0.7, y1 = yv(d.h2);
            const qx = V.lerp(x0, x1, V.ease.inOut(f2)), qy = V.lerp(y0, y1, V.ease.inOut(f2)) - Math.sin(f2 * Math.PI) * 26;
            V.dot(g, qx, qy, 3.2, C.d);
            if (f2 < 1) V.line(g, x0, y0, qx, qy, { color: V.alpha(C.d, 0.35), lw: 1, dash: [2, 3] });
          }
          g.globalAlpha = 1;
        });
        // histograma do que saiu
        const hTop = cBot + 40, hBot = h - 24;
        V.text(g, `emitidos: ${S.n} tokens (frequência) · tracejado = p`, pad, hTop - 8, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        // escala que acomoda a maior frequência observada (com poucos dardos, f pode passar de p)
        const fmax = S.n ? Math.max(...S.counts) / S.n : 0;
        const hm = Math.max(Math.max(...P3) * 1.25, fmax * 1.05);
        for (let i = 0; i < 6; i++) {
          const x0 = L + cw * i + (cw - bw) / 2;
          const f = S.n ? S.counts[i] / S.n : 0;
          const hh = Math.min(hBot - hTop, (f / hm) * (hBot - hTop));
          V.box(g, x0, hBot - hh, bw, Math.max(1, hh), { r: 3, fill: V.alpha(C.a, 0.7) });
          const yp = hBot - (P3[i] / hm) * (hBot - hTop);
          V.line(g, x0 - 4, yp, x0 + bw + 4, yp, { color: C.ink, lw: 1.6, dash: [4, 3] });
          V.text(g, VOC[i], x0 + bw / 2, hBot + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        }
        V.line(g, L, hBot, Rr, hBot, { color: V.alpha(C.muted, 0.6), lw: 1 });
      });
    })();

    /* ================================================ FIG 4 — calculadora */
    (function () {
      const fig = root.querySelector('#fig-calc');
      adaptH(fig, 430, 600);
      const PROF = {
        none: null,
        qwen: { label: 'Qwen · MTP greedy', a: [0.89, 0.745, 0.6] },
        ds: { label: 'DeepSeek · Vision-Exp', a: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15] },
        glmS: { label: 'GLM · contagem (structured)', a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83] },
        glmP: { label: 'GLM · prosa', a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06] },
      };
      const S = { a: 0.75, k: 4, c: 0.05, v: 0.02, block: false, prof: 'none' };
      const tau = (a, k) => (a >= 0.9999 ? k + 1 : (1 - Math.pow(a, k + 1)) / (1 - a));
      const cost = (k) => (k === 0 ? 1 : 1 + k * S.v + (S.block ? S.c : k * S.c));
      const sp = (k) => tau(S.a, k) / cost(k);
      const tauP = (arr, k) => 1 + arr.slice(0, k).reduce((s, x) => s + x, 0);
      const best = () => { let bk = 0, bs = 0; for (let k = 0; k <= 12; k++) { const s = sp(k); if (s > bs) { bs = s; bk = k; } } return [bk, bs]; };
      let H4;
      const ctl = UI.controls(fig);
      const rd = () => H4 && H4.redraw();
      nocase(UI.slider(ctl, { label: 'α', min: 0.05, max: 0.98, step: 0.01, value: S.a, fmt: (v) => fmt(v, 2), onInput: (v) => { S.a = v; upd(); rd(); } }));
      nocase(UI.slider(ctl, { label: 'k', min: 1, max: 12, step: 1, value: S.k, onInput: (v) => { S.k = v; upd(); rd(); } }));
      nocase(UI.slider(ctl, { label: 'custo c', min: 0, max: 0.4, step: 0.01, value: S.c, fmt: (v) => fmt(v, 2), onInput: (v) => { S.c = v; upd(); rd(); } }));
      nocase(UI.slider(ctl, { label: 'custo v', min: 0, max: 0.1, step: 0.005, value: S.v, fmt: (v) => fmt(v, 3), onInput: (v) => { S.v = v; upd(); rd(); } }));
      UI.toggle(ctl, { label: 'rascunho em bloco (1 forward)', value: false, onChange: (v) => { S.block = v; upd(); rd(); } });
      UI.seg(ctl, { label: 'Perfil medido', options: [{ v: 'none', label: 'nenhum' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'glmS', label: 'GLM contagem' }, { v: 'glmP', label: 'GLM prosa' }], value: 'none', onChange: (v) => { S.prof = v; upd(); rd(); } });
      const rT = nocase(UI.readout(ctl, 'τ(k)', '—')), rS = nocase(UI.readout(ctl, 'speedup', '—')), rK = nocase(UI.readout(ctl, 'melhor k (geom.)', '—')), rL = nocase(UI.readout(ctl, 'teto 1/(1−α)', '—'));
      function upd() {
        rT.set(fmt(tau(S.a, S.k), 2)); rS.set(fmt(sp(S.k), 2) + '×');
        const [bk, bs] = best(); rK.set(bk + ' (' + fmt(bs, 2) + '×)');
        rL.set(fmt(1 / (1 - S.a), 1));
      }
      upd();
      H4 = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const wide = w >= 640;
        const A = wide ? { x: 12, y: 8, w: w * 0.42 - 12, h: h - 16 } : { x: 8, y: 8, w: w - 16, h: h * 0.45 - 8 };
        const B = wide ? { x: w * 0.46, y: 8, w: w * 0.54 - 14, h: h - 16 } : { x: 8, y: h * 0.48, w: w - 16, h: h * 0.52 - 10 };
        const prof = PROF[S.prof];
        // ---- painel A: aceitação por posição
        const star = (x, y, col, filled) => {
          g.beginPath();
          for (let q = 0; q < 10; q++) { const r = q % 2 ? 4 : 9, a = -Math.PI / 2 + (q * Math.PI) / 5; const px = x + r * Math.cos(a), py = y + r * Math.sin(a); if (q === 0) g.moveTo(px, py); else g.lineTo(px, py); }
          g.closePath();
          if (filled) { g.fillStyle = col; g.fill(); } else { g.fillStyle = V.alpha(col, 0.25); g.fill(); g.strokeStyle = col; g.lineWidth = 1.6; g.stroke(); }
        };
        const diamond = (x, y, r, col) => { g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath(); g.fillStyle = V.alpha(col, 0.85); g.fill(); g.strokeStyle = col; g.lineWidth = 1; g.stroke(); };
        {
          const L = A.x + 34, R = A.x + A.w - 6, T = A.y + (prof ? 56 : 40), Bt = A.y + A.h - 30;
          V.text(g, 'aceitação por posição aᵢ', A.x, A.y + 12, { size: 12, weight: 600, color: C.ink });
          V.text(g, `geométrico: aᵢ = α^i · τ = ${fmt(tau(S.a, S.k), 2)}`, A.x, A.y + 28, { size: 11, font: 'mono', color: C.muted, maxW: A.w });
          if (prof) {
            // legenda do perfil: aᵢ medido (pontos lilás) e cᵢ = aᵢ/aᵢ₋₁ (losangos rosa)
            let lx = A.x; const ly = A.y + 44;
            V.dot(g, lx + 5, ly - 4, 4, C.b); V.text(g, 'aᵢ medido', lx + 13, ly, { size: 11, color: C.muted }); lx += 26 + tw(g, 'aᵢ medido', 11);
            diamond(lx + 5, ly - 4, 4.5, C.c); V.text(g, 'cᵢ = aᵢ/aᵢ₋₁ (condicional)', lx + 13, ly, { size: 11, color: C.muted, maxW: A.x + A.w - lx - 14 });
          }
          const k = S.k, cw = (R - L) / Math.max(k, 7);
          const yv = (v) => Bt - v * (Bt - T);
          [0, 0.5, 1].forEach((v) => { V.line(g, L, yv(v), R, yv(v), { color: C.grid, lw: 1 }); V.text(g, fmt(v, 1), L - 5, yv(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
          for (let i = 1; i <= k; i++) {
            const v = Math.pow(S.a, i), x = L + cw * (i - 1) + cw * 0.18;
            V.box(g, x, yv(v), cw * 0.64, Math.max(1, Bt - yv(v)), { r: 3, fill: V.alpha(C.a, 0.55), stroke: C.a, lw: 1 });
            V.text(g, String(i), x + cw * 0.32, Bt + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          }
          if (prof) {
            const nShow = Math.min(prof.a.length, Math.max(k, 7));
            const cc = prof.a.map((v, i) => (i === 0 ? v : v / prof.a[i - 1]));
            const px = (i) => L + cw * i + cw / 2;
            // cᵢ tracejado rosa
            g.beginPath(); for (let i = 0; i < nShow; i++) { if (i === 0) g.moveTo(px(i), yv(cc[i])); else g.lineTo(px(i), yv(cc[i])); }
            g.strokeStyle = V.alpha(C.c, 0.8); g.lineWidth = 1.4; g.setLineDash([4, 3]); g.stroke(); g.setLineDash([]);
            for (let i = 0; i < nShow; i++) diamond(px(i), yv(cc[i]), 4.5, C.c);
            // aᵢ medido, lilás (mesma cor do painel B)
            g.beginPath(); for (let i = 0; i < nShow; i++) { if (i === 0) g.moveTo(px(i), yv(prof.a[i])); else g.lineTo(px(i), yv(prof.a[i])); }
            g.strokeStyle = C.b; g.lineWidth = 2; g.stroke();
            for (let i = 0; i < nShow; i++) V.dot(g, px(i), yv(prof.a[i]), 4, C.b);
            for (let i = k; i < nShow; i++) V.text(g, String(i + 1), px(i), Bt + 14, { size: 11, font: 'mono', color: C.b, align: 'center' });
            V.text(g, prof.label + ' (medido)', A.x, Bt + 27, { size: 11, color: C.b, maxW: A.w - 70 });
          }
          V.text(g, 'posição i', R, Bt + 27, { size: 11, color: C.muted, align: 'right' });
        }
        // ---- painel B: τ(k) e speedup(k)
        {
          V.text(g, 'tokens por passo τ(k) e speedup S(k)', B.x, B.y + 12, { size: 12, weight: 600, color: C.ink, maxW: B.w });
          const lg = [[C.a, 'τ geométrico', 'dot'], [C.d, 'S geométrico', 'dot']];
          if (prof) lg.push([C.b, 'τ medido', 'dot'], [C.b, 'S medido', 'ring']);
          let lx = B.x, ly = B.y + 29;
          lg.forEach(([c, l, kind]) => {
            const ww = 22 + tw(g, l, 11);
            if (lx + ww > B.x + B.w && lx > B.x) { lx = B.x; ly += 15; }
            if (kind === 'ring') { g.beginPath(); g.arc(lx + 5, ly - 4, 3.5, 0, Math.PI * 2); g.strokeStyle = c; g.lineWidth = 1.6; g.stroke(); }
            else V.dot(g, lx + 5, ly - 4, 4, c);
            V.text(g, l, lx + 13, ly, { size: 11, color: C.muted }); lx += ww;
          });
          const L = B.x + 34, R = B.x + B.w - 8, T = ly + 14, Bt = B.y + B.h - 30;
          let ymax = 1; for (let k = 0; k <= 12; k++) ymax = Math.max(ymax, tau(S.a, k), sp(k));
          if (prof) for (let k = 0; k <= prof.a.length; k++) ymax = Math.max(ymax, tauP(prof.a, k));
          ymax = Math.ceil(ymax * 1.1);
          const xs = (k) => L + (k / 12) * (R - L), ys = (v) => Bt - (v / ymax) * (Bt - T);
          const stp = niceStep(ymax, 4);
          for (let v = 0; v <= ymax + 1e-9; v += stp) { V.line(g, L, ys(v), R, ys(v), { color: C.grid, lw: 1 }); V.text(g, String(+v.toFixed(1)).replace('.', ','), L - 5, ys(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
          for (let k = 0; k <= 12; k += 2) V.text(g, String(k), xs(k), Bt + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'k', R, Bt + 27, { size: 11, color: C.muted, align: 'right' });
          V.line(g, xs(S.k), T, xs(S.k), Bt, { color: V.alpha(C.ink, 0.35), lw: 1, dash: [3, 3] });
          const line = (fn, col, lw) => { g.beginPath(); for (let k = 0; k <= 12; k++) { const x = xs(k), y = ys(fn(k)); if (k === 0) g.moveTo(x, y); else g.lineTo(x, y); } g.strokeStyle = col; g.lineWidth = lw; g.stroke(); for (let k = 0; k <= 12; k++) V.dot(g, xs(k), ys(fn(k)), 2.5, col); };
          line((k) => tau(S.a, k), C.a, 2);
          line(sp, C.d, 2.6);
          if (prof) {
            let pbk = 0, pbs = 0;
            for (let k = 0; k <= prof.a.length; k++) {
              const tp = tauP(prof.a, k), spp = tp / cost(k);
              if (spp > pbs) { pbs = spp; pbk = k; }
              V.dot(g, xs(k), ys(tp), 3.5, C.b);
              g.beginPath(); g.arc(xs(k), ys(spp), 4, 0, Math.PI * 2); g.strokeStyle = C.b; g.lineWidth = 1.6; g.stroke();
            }
            star(xs(pbk), ys(pbs) - 14, C.b, false);
          }
          const [bk, bs] = best();
          star(xs(bk), ys(bs) - 14, C.d, true);
          V.text(g, `S(${S.k}) = ${fmt(sp(S.k), 2)}×`, xs(S.k) + 6, T + 12, { size: 12, font: 'mono', weight: 600, color: C.ink });
        }
      }, { static: true });
    })();

    /* ================================================ FIG 5 — balança greedy × probabilístico */
    (function () {
      const fig = root.querySelector('#fig-balanca');
      adaptH(fig, 470, 540);
      const TOK = ['Lima', 'Rio', 'Roma', 'Oslo', 'Nice', 'Baku'];
      const ZP = [2.4, 1.7, 1.1, 0.5, -0.2, -0.9];
      const DQ = [0.1, -0.15, 0.25, -0.1, 0.2, 0.0];
      const S = { T: 1.0, s: 0.6, swap: false, ang: 0 };
      function dists(T) {
        const zq = ZP.map((z, i) => z + DQ[i]);
        if (S.swap) { const tmp = zq[0]; zq[0] = zq[1]; zq[1] = tmp; }
        return { p: V.softmax(ZP, T), q: V.softmax(zq.map((z) => z * S.s), T) };
      }
      function accs(T) {
        const { p, q } = dists(T);
        const xh = q.indexOf(Math.max(...q));
        return { ag: p[xh], ap: p.reduce((a, pi, i) => a + Math.min(pi, q[i]), 0), xh, p, q };
      }
      let curve = [];
      const recomputeCurve = () => { curve = []; for (let k = 0; k <= 60; k++) { const T = 0.05 + (k / 60) * 1.95; const r = accs(T); curve.push([T, r.ag, r.ap]); } };
      recomputeCurve();
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'T', min: 0.05, max: 2, step: 0.05, value: S.T, fmt: (v) => fmt(v, 2), onInput: (v) => { S.T = v; upd(); } });
      nocase(UI.slider(ctl, { label: 'confiança q', min: 0.3, max: 2, step: 0.05, value: S.s, fmt: (v) => fmt(v, 2) + '×', onInput: (v) => { S.s = v; recomputeCurve(); upd(); } }));
      UI.toggle(ctl, { label: 'favoritos diferentes', value: false, onChange: (v) => { S.swap = v; recomputeCurve(); upd(); } });
      const rG = nocase(UI.readout(ctl, 'α greedy = p(x̂)', '—')), rP = nocase(UI.readout(ctl, 'α prob. = 1 − TV', '—'));
      function upd() { const r = accs(S.T); rG.set(fmt(r.ag, 3)); rP.set(fmt(r.ap, 3)); }
      upd();
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const r = accs(S.T);
        const cG = C.d, cP = C.c;
        const pad = 12;
        // ---- balança
        const cx = w / 2, cy = 70, Lb = Math.min(w * 0.26, 180);
        const target = V.clamp((r.ap - r.ag) * 1.2, -0.2, 0.2);
        S.ang = V.lerp(S.ang, target, 0.12);
        const a = S.ang;
        g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx - 12, cy + 28); g.lineTo(cx + 12, cy + 28); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.5); g.fill();
        const lx = cx - Lb * Math.cos(a), ly = cy - Lb * Math.sin(a), rx = cx + Lb * Math.cos(a), ry = cy + Lb * Math.sin(a);
        V.line(g, lx, ly, rx, ry, { color: C.ink, lw: 3 });
        V.dot(g, cx, cy, 4, C.ink);
        const pan = (x, y, col, lab, val) => {
          V.line(g, x, y, x - 22, y + 20, { color: V.alpha(C.ink, 0.5), lw: 1 }); V.line(g, x, y, x + 22, y + 20, { color: V.alpha(C.ink, 0.5), lw: 1 });
          V.box(g, x - 40, y + 20, 80, 30, { r: 12, fill: V.alpha(col, 0.25), stroke: col, lw: 1.5 });
          V.text(g, lab, x, y + 32, { size: 11, color: C.ink, align: 'center' });
          V.text(g, fmt(val, 2), x, y + 45, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center' });
        };
        pan(lx, ly, cG, 'greedy', r.ag);
        pan(rx, ry, cP, 'probabilístico', r.ap);
        const win = Math.abs(r.ap - r.ag) < 0.01 ? 'empate' : r.ap > r.ag ? 'probabilístico aceita mais' : 'greedy aceita mais';
        V.text(g, win, cx, 18, { size: 12.5, weight: 700, color: C.ink, align: 'center' });
        // ---- painéis
        const pTop = 190, pBot = h - 142;
        const gap = 14, pw = (w - 2 * pad - gap) / 2;
        const ymax = Math.max(...r.p, ...r.q) * 1.15;
        const panel = (x0, mode) => {
          V.text(g, mode === 'g' ? 'rascunho greedy: aceita p(x̂)' : 'rascunho probabilístico: Σ min(p, q)', x0, pTop - 8, { size: 11.5, weight: 600, color: mode === 'g' ? cG : cP, maxW: pw });
          const n = 6, cw = pw / n, bw = Math.min(30, cw * 0.62);
          const yv = (v) => pBot - (v / ymax) * (pBot - pTop - 40);
          for (let i = 0; i < n; i++) {
            const bx = x0 + cw * i + (cw - bw) / 2;
            const p = r.p[i], q = r.q[i];
            if (mode === 'g') {
              const on = i === r.xh;
              g.fillStyle = on ? V.alpha(C.e || C.a, 0.75) : V.alpha(C.a, 0.18); g.fillRect(bx, yv(p), bw, pBot - yv(p));
              g.strokeStyle = C.a; g.lineWidth = 1.5; g.strokeRect(bx, yv(p), bw, pBot - yv(p));
              if (on) {
                const fy = yv(p) - 26;
                g.beginPath(); g.moveTo(bx + bw / 2, fy + 14); g.lineTo(bx + bw / 2 - 7, fy); g.lineTo(bx + bw / 2 + 7, fy); g.closePath(); g.fillStyle = C.b; g.fill();
                V.text(g, 'x̂', bx + bw / 2, fy - 4, { size: 12, font: 'mono', weight: 700, color: C.b, align: 'center' });
              }
            } else {
              const mn = Math.min(p, q);
              g.fillStyle = V.alpha(C.e || C.a, 0.55); g.fillRect(bx, yv(mn), bw, pBot - yv(mn));
              g.strokeStyle = C.a; g.lineWidth = 1.5; g.strokeRect(bx, yv(p), bw, pBot - yv(p));
              g.setLineDash([4, 3]); g.strokeStyle = C.b; g.strokeRect(bx + 1.5, yv(q), bw - 3, pBot - yv(q)); g.setLineDash([]);
            }
            V.text(g, cw < 44 ? TOK[i].slice(0, 3).toUpperCase() : TOK[i], bx + bw / 2, pBot + 14, { size: 11, font: cw < 44 ? 'mono' : 'sans', color: C.muted, align: 'center' });
          }
          V.line(g, x0, pBot, x0 + pw, pBot, { color: V.alpha(C.muted, 0.6), lw: 1 });
        };
        panel(pad, 'g');
        panel(pad + pw + gap, 'p');
        // ---- curva α(T)
        const sT = pBot + 50, sB = h - 26, sL = pad + 30, sR = w - pad - 6;
        const xs = (T) => sL + ((T - 0.05) / 1.95) * (sR - sL), ys = (v) => sB - v * (sB - sT);
        V.text(g, 'aceitação × temperatura', pad, sT - 8, { size: 11.5, font: 'mono', color: C.muted });
        [0, 0.5, 1].forEach((v) => { V.line(g, sL, ys(v), sR, ys(v), { color: C.grid, lw: 1 }); V.text(g, fmt(v, 1), sL - 4, ys(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        [0.05, 0.5, 1, 1.5, 2].forEach((T) => V.text(g, fmt(T, T < 0.1 ? 2 : 1), xs(T), sB + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        const pl = (idx, col) => { g.beginPath(); curve.forEach((c, k) => { const x = xs(c[0]), y = ys(c[idx]); if (k === 0) g.moveTo(x, y); else g.lineTo(x, y); }); g.strokeStyle = col; g.lineWidth = 2.4; g.stroke(); };
        pl(1, cG); pl(2, cP);
        {
          const last = curve[curve.length - 1];
          let yg = ys(last[1]) - 6, yp = ys(last[2]) - 6;
          if (Math.abs(yg - yp) < 13) { if (yg < yp) yg = yp - 13; else yp = yg - 13; }
          V.text(g, 'greedy', sR - 2, yg, { size: 11, weight: 600, color: cG, align: 'right' });
          V.text(g, 'prob.', sR - 2, yp, { size: 11, weight: 600, color: cP, align: 'right' });
        }
        V.line(g, xs(S.T), sT, xs(S.T), sB, { color: V.alpha(C.ink, 0.4), lw: 1, dash: [3, 3] });
        V.dot(g, xs(S.T), ys(r.ag), 4.5, cG); V.dot(g, xs(S.T), ys(r.ap), 4.5, cP);
        return undefined;
      });
    })();

    /* ================================================ FIG 6 — mapa dos rascunhadores */
    (function () {
      const fig = root.querySelector('#fig-mapa');
      adaptH(fig, 430, 560);
      const MET = {
        ngram: { name: 'n-gram / prompt lookup', inp: 'o próprio texto (prompt + saída)', fw: 'zero forwards: só uma busca', tr: 'nenhum treino', rec: 'não usado nas receitas', taps: [], unit: 0.02, seq: false },
        draft: { name: 'modelo rascunho', inp: 'só os tokens; tem embed, KV e lm_head próprios', fw: 'k forwards seguidos do LM pequeno', tr: 'LM separado, mesmo tokenizer', rec: 'não usado nas receitas', taps: [], unit: 0.14, seq: true },
        eagle: { name: 'EAGLE-3', inp: 'estados ocultos de 3 camadas do alvo + tokens', fw: 'k forwards da cabeça leve', tr: 'treinado contra o alvo', rec: 'maquinaria de features reusada pelo DFlash2 (GLM)', taps: [1, 4, 7], unit: 0.06, seq: true },
        mtp: { name: 'MTP', inp: 'último estado oculto + embedding do próximo token', fw: 'k forwards da mesma camada', tr: 'veio do pré-treino do alvo', rec: 'Qwen k=3 · GLM rollback k=2', taps: [8], unit: 0.08, seq: true },
        bloco: { name: 'bloco (DFlash / DSpark)', inp: 'features de várias camadas + [âncora, MASK…]', fw: '1 forward para o bloco inteiro', tr: 'treinado para o alvo', rec: 'GLM DFlash2 k=7 · DS DSpark k=6', taps: [1, 2, 4, 5, 7], unit: 0.13, seq: false },
      };
      const S = { m: 'mtp', k: 4 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Método', options: [{ v: 'ngram', label: 'n-gram' }, { v: 'draft', label: 'modelo rascunho' }, { v: 'eagle', label: 'EAGLE-3' }, { v: 'mtp', label: 'MTP' }, { v: 'bloco', label: 'bloco' }], value: S.m, onChange: (v) => { S.m = v; } });
      nocase(UI.slider(ctl, { label: 'k', min: 1, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; } }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const M = MET[S.m];
        const wide = w >= 640;
        const tlH = 64;
        const top = 14, bottom = h - tlH - 12;
        // alvo
        const sx = 12, sw = wide ? 150 : Math.min(118, w * 0.34);
        const nL = 8, lh = (bottom - top - 60) / (nL + 2);
        V.text(g, 'alvo', sx + sw / 2, top + 8, { size: 12, weight: 700, color: C.a, align: 'center' });
        const layerY = (i) => bottom - 8 - (i + 1.5) * lh; // i=0 embedding ... nL lm_head
        chip(g, sx, layerY(-0.5), sw, lh - 4, 'embedding', { size: 11, font: 'sans', fill: V.alpha(C.muted, 0.12) });
        for (let i = 0; i < nL; i++) {
          const tap = M.taps.includes(i) || (S.m === 'mtp' && i === nL - 1);
          chip(g, sx, layerY(i + 0.5), sw, lh - 4, 'camada ' + (i + 1), { size: 11, font: 'sans', fill: tap ? V.alpha(C.b, 0.3) : V.alpha(C.a, 0.1), stroke: tap ? C.b : V.alpha(C.a, 0.4) });
        }
        chip(g, sx, layerY(nL + 0.5), sw, lh - 4, 'lm_head', { size: 11, font: 'sans', fill: V.alpha(C.muted, 0.12) });
        // rascunhador
        const dx = sx + sw + (wide ? 70 : 30), dw = w - dx - 12;
        const dTop = top + 20, dH = wide ? 118 : 150;
        V.box(g, dx, dTop, dw, dH, { r: 12, fill: V.alpha(C.b, 0.1), stroke: C.b, lw: 1.5 });
        V.text(g, M.name, dx + 12, dTop + 20, { size: 13, weight: 700, color: C.b, maxW: dw - 24 });
        const ix = dx + 12, iy = dTop + 32, iw = dw - 24;
        const flow = (x1, y1, x2, y2, col) => V.line(g, x1, y1, x2, y2, { color: col, lw: 2, dash: [5, 6], dashOffset: -t * 24 });
        const bigChip = (x, y, ww, label, hl) => chip(g, x, y, ww, 26, label, { size: 12, font: 'sans', weight: 600, fill: hl ? V.alpha(C.b, 0.28) : V.alpha(C.muted, 0.12), stroke: hl ? C.b : V.alpha(C.muted, 0.45) });
        const two = iw > 300;
        if (S.m === 'ngram') {
          const words = ['…', 'total', '=', '0', '⏎', 'for', 'v', 'in', '…'];
          let x = ix; const y = iy + 4;
          words.forEach((wd, k) => { const ww = tw(g, wd, 12, 'mono') + 12; if (x + ww > ix + iw) return; chip(g, x, y, ww, 24, wd, { size: 12, fill: k >= 1 && k <= 3 ? V.alpha(C.d, 0.35) : undefined, stroke: k >= 1 && k <= 3 ? C.d : undefined }); x += ww + 4; });
          const mgx = ix + 16 + ((t * 40) % Math.max(20, iw - 40)), mgy = y + 12;
          g.beginPath(); g.arc(mgx, mgy, 12, 0, Math.PI * 2); g.strokeStyle = C.d; g.lineWidth = 2.2; g.stroke(); V.line(g, mgx + 9, mgy + 9, mgx + 16, mgy + 16, { color: C.d, lw: 3 });
          wrapText(g, 'acha o n-grama final no contexto e copia o que veio depois', ix, y + 50, iw, 16, { size: 12, color: C.ink });
        } else if (S.m === 'draft') {
          const cw3 = two ? iw * 0.46 : iw;
          for (let k = 0; k < 3; k++) bigChip(ix, iy + k * 29, cw3 * (two ? 1 : 0.55), 'camada ' + (k + 1), true);
          const ox = two ? ix + iw * 0.54 : ix + iw * 0.6, ow = two ? iw * 0.46 : iw * 0.4;
          ['embed', 'KV próprio', 'lm_head'].forEach((l, k) => bigChip(ox, iy + k * 29, ow, l, false));
        } else if (S.m === 'eagle') {
          bigChip(ix, iy + 8, two ? iw * 0.5 : iw * 0.55, 'cabeça leve', true);
          const ox = ix + iw * 0.6, ow = iw * 0.4;
          bigChip(ox, iy, ow, 'KV próprio', false); bigChip(ox, iy + 32, ow, 'lm_head', false);
          V.text(g, 'lê 3 camadas do alvo', ix, iy + 58, { size: 12, color: C.ink, maxW: iw * 0.58 });
        } else if (S.m === 'mtp') {
          bigChip(ix, iy + 4, two ? iw * 0.5 : iw * 0.55, 'camada MTP', true);
          const lx2 = ix + 18, ly2 = iy + 58;
          g.beginPath(); g.arc(lx2, ly2, 12, -Math.PI * 0.1 + t * 3, Math.PI * 1.6 + t * 3); g.strokeStyle = C.b; g.lineWidth = 2.2; g.stroke();
          V.text(g, `× ${S.k} (a mesma camada)`, lx2 + 20, ly2 + 4, { size: 12, font: 'mono', color: C.b, maxW: iw - 40 });
          bigChip(ix + iw * 0.6, iy + 4, iw * 0.4, 'lm_head', false);
        } else {
          const cells = ['âncora'].concat(Array.from({ length: S.k }, () => 'M'));
          const cw2 = Math.min(58, (iw - 4) / cells.length - 4);
          cells.forEach((c, k) => chip(g, ix + k * (cw2 + 4), iy + 4, cw2, 26, k === 0 ? (cw2 > 50 ? 'âncora' : 'a') : 'M', { size: 12, weight: 600, fill: k === 0 ? V.alpha(C.muted, 0.3) : V.alpha(C.b, 0.25 + 0.2 * Math.sin(t * 4 + k)), stroke: C.b }));
          for (let k = 1; k < cells.length; k++) { const x1 = ix + (k - 0.5) * (cw2 + 4), x2 = ix + (k + 0.5) * (cw2 + 4); g.beginPath(); g.moveTo(x1, iy + 32); g.quadraticCurveTo((x1 + x2) / 2, iy + 44, x2, iy + 32); g.strokeStyle = V.alpha(C.b, 0.5); g.lineWidth = 1; g.stroke(); }
          wrapText(g, 'atenção bidirecional no bloco: todas as posições num forward', ix, iy + 62, iw, 16, { size: 12, color: C.ink });
        }
        // setas de informação
        const tgtX = dx, tgtY = dTop + dH / 2;
        const taps = S.m === 'mtp' ? [nL - 1] : M.taps;
        taps.forEach((i) => flow(sx + sw, layerY(i + 0.5) + lh / 2 - 2, tgtX, tgtY, V.alpha(C.b, 0.8)));
        if (S.m === 'ngram' || S.m === 'draft') flow(sx + sw, layerY(nL + 0.5) + lh / 2 - 2, tgtX, dTop + 16, V.alpha(C.muted, 0.7));
        // rascunhos saindo do rascunhador e voltando ao alvo
        const ry = dTop + dH + 24;
        V.text(g, 'rascunhos', dx, ry + 4, { size: 11, font: 'mono', color: C.muted });
        const rsz = 18, rx0 = dx + 84;
        for (let k = 0; k < S.k; k++) {
          const x = rx0 + k * (rsz + 8);
          if (x + rsz / 2 > w - 12) break;
          const bob = Math.sin(t * 3 + k) * 2;
          tokChar(g, x, ry + bob, rsz, C.b);
        }
        V.line(g, dx + 30, dTop + dH, dx + 30, ry - 12, { color: V.alpha(C.b, 0.6), lw: 1.6, dash: [3, 4], dashOffset: -t * 20 });
        V.arrow(g, dx - 6, ry, sx + sw + 6, layerY(-0.5) + lh / 2 - 2, { color: V.alpha(C.a, 0.6), lw: 1.6, head: 7, dash: [5, 5], curve: 14 });
        if (wide) V.text(g, `verifica ${S.k + 1}`, sx + sw + 8, layerY(-0.5) + lh + 8, { size: 11, color: C.a, maxW: dx - sx - sw - 14 });
        // informações
        const infY = ry + 34;
        const lines = [['entrada', M.inp], ['por passo', S.m === 'ngram' || S.m === 'bloco' ? M.fw : M.fw.replace('k', String(S.k))], ['treino', M.tr], ['nas receitas', M.rec]];
        let yy = infY;
        lines.forEach(([k2, v]) => {
          V.text(g, k2, dx, yy, { size: 11, font: 'mono', color: C.muted });
          yy = wrapText(g, v, dx + (wide ? 96 : 0), wide ? yy : yy + 15, dw - (wide ? 96 : 0), 15, { size: 12, color: C.ink }) + 4;
        });
        // linha do tempo
        const ty = h - tlH + 8, tL = 12, tR = w - 12;
        V.text(g, 'um passo (ilustrativo): rascunho → verificação', tL, ty, { size: 11, font: 'mono', color: C.muted });
        const segs = [];
        if (S.m === 'ngram') segs.push(M.unit);
        else if (M.seq) for (let k = 0; k < S.k; k++) segs.push(M.unit);
        else segs.push(M.unit);
        const ver = 1 + 0.04 * S.k;
        const tot = 0.14 * 7 + 1 + 0.04 * 7;
        const scale = (tR - tL) / tot;
        let x = tL; const by = ty + 10, bh = 20;
        segs.forEach((u) => { const ww = Math.max(3, u * scale - 2); V.box(g, x, by, ww, bh, { r: 4, fill: V.alpha(C.b, 0.7) }); x += ww + 2; });
        const vw = ver * scale;
        V.box(g, x, by, vw, bh, { r: 4, fill: V.alpha(C.a, 0.55), stroke: C.a, lw: 1 });
        V.text(g, `verifica ${S.k + 1} posições`, x + vw / 2, by + bh / 2 + 4, { size: 11, color: C.ink, align: 'center', maxW: vw - 6 });
        const prog = (t * 0.35) % 1, px = tL + prog * (x + vw - tL);
        V.line(g, px, by - 3, px, by + bh + 3, { color: C.d, lw: 2 });
      });
    })();

    /* ================================================ FIG 7 — prompt lookup ao vivo */
    (function () {
      const fig = root.querySelector('#fig-lookup');
      const st7 = fig.querySelector('.la-fig-stage');
      let fitW = -1;
      const PROMPT = ['Renomeie', 'total', 'para', 'soma', ':', '⏎', 'def', 'media', '(', 'valores', ')', ':', '⏎', 'total', '=', '0', '⏎', 'for', 'v', 'in', 'valores', ':', '⏎', 'total', '+=', 'v', '⏎', 'return', 'total', '/', 'len', '(', 'valores', ')', '⏎', 'Resposta', ':', '⏎'];
      const TRUTH = ['def', 'media', '(', 'valores', ')', ':', '⏎', 'soma', '=', '0', '⏎', 'for', 'v', 'in', 'valores', ':', '⏎', 'soma', '+=', 'v', '⏎', 'return', 'soma', '/', 'len', '(', 'valores', ')', '⟂'];
      const NP = PROMPT.length;
      const S = { n: 3, k: 5, out: [], kinds: [], steps: 0, prop: 0, acc: 0, last: null, play: true };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'n máx.', min: 1, max: 4, step: 1, value: S.n, onInput: (v) => { S.n = v; reset(); } }));
      nocase(UI.slider(ctl, { label: 'k', min: 1, max: 8, step: 1, value: S.k, onInput: (v) => { S.k = v; reset(); } }));
      UI.button(ctl, 'passo →', () => { S.play = false; stepOnce(); }, { primary: true });
      const bPlay = UI.button(ctl, '⏸ pausar', () => { S.play = !S.play; bPlay.textContent = S.play ? '⏸ pausar' : '▶ auto'; });
      UI.button(ctl, 'reiniciar', () => reset());
      const rP = UI.readout(ctl, 'passos', '0'), rT = UI.readout(ctl, 'tokens/passo', '—'), rA = UI.readout(ctl, 'aceitação', '—');
      function reset() { S.out = []; S.kinds = []; S.steps = 0; S.prop = 0; S.acc = 0; S.last = null; updR(); }
      function updR() { rP.set(String(S.steps)); rT.set(S.steps ? fmt(S.out.length / S.steps, 2) : '—'); rA.set(S.prop ? fmt(S.acc / S.prop, 2) : '—'); }
      function propose() {
        const ctx = PROMPT.concat(S.out);
        for (let n = S.n; n >= 1; n--) {
          if (ctx.length < n + 1) continue;
          const pat = ctx.slice(-n);
          for (let i = ctx.length - n - 1; i >= 0; i--) {
            let ok = true; for (let q = 0; q < n; q++) if (ctx[i + q] !== pat[q]) { ok = false; break; }
            if (ok) return { i, n, prop: ctx.slice(i + n, Math.min(ctx.length, i + n + S.k)) };
          }
        }
        return null;
      }
      function stepOnce() {
        if (S.out.length >= TRUTH.length) { reset(); return; }
        const P = propose();
        const prop = P ? P.prop : [];
        let a = 0;
        while (a < prop.length && S.out.length + a < TRUTH.length && prop[a] === TRUTH[S.out.length + a]) a++;
        for (let q = 0; q < a; q++) { S.out.push(prop[q]); S.kinds.push('ok'); }
        if (S.out.length < TRUTH.length) { S.out.push(TRUTH[S.out.length]); S.kinds.push('tgt'); }
        S.steps++; S.prop += prop.length; S.acc += a;
        S.last = { P, prop, a, emitted: a + 1, t0: performance.now() };
        updR();
      }
      V.every(1700, () => { if (S.play) stepOnce(); });
      function layout(g, toks, x0, y0, maxW, size) {
        const pos = []; let x = x0, y = y0;
        toks.forEach((tk, i) => {
          const ww = tw(g, tk, size, 'mono') + 10;
          if (x + ww > x0 + maxW && x > x0) { x = x0; y += 25; }
          pos.push([x, y, ww]);
          x += ww + 4;
          if (tk === '⏎' || tk === 'Resposta') { if (tk === '⏎') { x = x0; y += 25; } }
        });
        return { pos, end: y + 25 };
      }
      /** altura do palco = o que a resposta completa ocupa nesta largura (sem sobra no mobile) */
      function fitHeight(g, w) {
        if (w === fitW) return;
        fitW = w;
        const pad = 12, size = w < 560 ? 11 : 11.5;
        const lp = layout(g, PROMPT, pad, 22, w - 2 * pad, size);
        const yResp = lp.end + 10 + 12 + 32;
        const lo = layout(g, TRUTH, pad, yResp + 12, w - 2 * pad, size);
        const want = Math.max(360, Math.ceil(lo.end + 20)) + 'px';
        if (st7.style.height !== want) st7.style.height = want;
      }
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        fitHeight(g, w);
        const pad = 12, size = w < 560 ? 11 : 11.5;
        const L = S.last;
        const ctxLen = PROMPT.length + S.out.length - (L ? L.emitted : 0);
        // região do match (no contexto antes do passo)
        const inMatch = (gi) => L && L.P && gi >= L.P.i && gi < L.P.i + L.P.n;
        const inProp = (gi) => L && L.P && gi >= L.P.i + L.P.n && gi < L.P.i + L.P.n + L.prop.length && gi < ctxLen;
        V.text(g, 'contexto (prompt)', pad, 14, { size: 11.5, font: 'mono', color: C.muted });
        const lp = layout(g, PROMPT, pad, 22, w - 2 * pad, size);
        PROMPT.forEach((tk, i) => {
          const [x, y, ww] = lp.pos[i];
          const m = inMatch(i), pr = inProp(i);
          chip(g, x, y, ww, 20, tk, { size, fill: m ? V.alpha(C.c, 0.45) : pr ? V.alpha(C.b, 0.25) : i < 6 || i >= NP - 3 ? V.alpha(C.muted, 0.06) : V.alpha(C.muted, 0.12), stroke: m ? C.c : pr ? C.b : V.alpha(C.muted, 0.35), color: C.ink, weight: m ? 700 : 500 });
        });
        // linha do último passo
        let y = lp.end + 10;
        if (L) {
          V.text(g, L.P ? `n-grama de ${L.P.n} encontrado → rascunho de ${L.prop.length}: ${L.a} aceito(s) + 1 do alvo` : 'sem n-grama no contexto → passo normal (1 token do alvo)', pad, y + 4, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
          y += 12;
          let x = pad;
          V.text(g, 'rascunho:', x, y + 14, { size: 11, font: 'mono', color: C.b });
          x += tw(g, 'rascunho:', 11, 'mono') + 8;
          L.prop.forEach((tk, q) => {
            const ok = q < L.a, rej = q === L.a;
            const lab = tk + (ok ? ' ✓' : rej ? ' ✗' : '');
            const ww = tw(g, lab, size, 'mono') + 10;
            if (x + ww > w - pad) return;
            chip(g, x, y, ww, 20, lab, { size, fill: V.alpha(ok ? C.a : rej ? RED() : C.b, ok || rej ? 0.22 : 0.08), stroke: ok ? C.a : rej ? RED() : V.alpha(C.muted, 0.35), color: q > L.a ? C.muted : C.ink });
            x += ww + 4;
          });
          y += 32;
        } else { V.text(g, 'aperte “passo →” ou deixe no automático', pad, y + 4, { size: 12, color: C.muted }); y += 26; }
        // resposta
        V.text(g, `resposta do alvo (T = 0): ${S.out.length}/${TRUTH.length} tokens`, pad, y + 4, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const lo = layout(g, S.out, pad, y + 12, w - 2 * pad, size);
        const fresh = L ? S.out.length - L.emitted : S.out.length;
        S.out.forEach((tk, i) => {
          const [x, yy, ww] = lo.pos[i];
          if (yy + 20 > h - 4) return;
          const gi = PROMPT.length + i;
          const m = inMatch(gi);
          const col = S.kinds[i] === 'ok' ? C.a : C.d;
          chip(g, x, yy, ww, 20, tk, { size, fill: m ? V.alpha(C.c, 0.45) : V.alpha(col, i >= fresh ? 0.45 : 0.18), stroke: m ? C.c : col, color: C.ink, weight: i >= fresh ? 700 : 500 });
        });
      });
    })();

    /* ================================================ FIG 8 — rollback do estado recorrente */
    (function () {
      const fig = root.querySelector('#fig-rollback');
      adaptH(fig, 470, 640);
      const TOKS = ['no', 'telhado', ',', 'miou', 'alto'];
      const PAST = ['O', 'gato', 'subiu'];
      const r8 = V.rng(88);
      const vec = () => Array.from({ length: 4 }, () => Math.round((r8() * 2 - 1) * 10) / 10);
      const KV = TOKS.map(() => ({ k: vec(), v: vec() }));
      const gam = 0.8;
      const S0 = Array.from({ length: 4 }, () => vec().map((x) => Math.round(x * 5) / 10));
      const SN = [S0];
      KV.forEach(({ k, v }) => { const P = SN[SN.length - 1]; SN.push(P.map((row, i) => row.map((x, j) => gam * x + v[i] * k[j]))); });
      const S = { j: 2, snaps: true };
      const steps = [
        { t: 'O estado antes do passo', d: 'Uma camada recorrente não guarda uma linha por token: guarda uma matriz \\(S\\) que resume o passado (aqui, 4×4, com valores de verdade). O KV, à direita, tem uma linha por token; a conv curta guarda os últimos \\(K-1 = 3\\) inputs.' },
        { t: 'Verificar = atualizar 5 vezes', d: 'O alvo processa a âncora e os 4 rascunhos. Cada token atualiza o estado, \\(S \\leftarrow \\gamma S + v\\,k^\\top\\), e o kernel tira uma <b>fotografia</b> depois de cada um: \\(S^{(1)}, \\dots, S^{(5)}\\). A conv usa uma janela mais larga, de \\(K-1+k = 7\\) colunas.' },
        { t: 'O veredito escolhe a fotografia', d: 'Com \\(j\\) rascunhos aceitos, o estado oficial é \\(S^{(j+1)}\\), o de depois da âncora e dos \\(j\\) aceitos: é o que <code>num_accepted_tokens</code> seleciona. A janela da conv recua para o trecho certo; no KV, basta ignorar as linhas rejeitadas. Mova o controle “aceitos”.' },
        { t: 'Sem fotografias: estado contaminado', d: 'Sem as cópias, sobraria só \\(S^{(5)}\\), que absorveu os \\(4-j\\) rascunhos rejeitados. Em aritmética exata daria para desfazer, token a token: \\(S^{(4)} = (S^{(5)} - v_5 k_5^\\top)/\\gamma\\). Mas isso exige guardar chave, valor e gate de cada rascunho, custa uma atualização inversa por token e por camada e, em ponto flutuante, dividir por um \\(\\gamma\\) pequeno amplifica o erro. Sem desfazer, as células em vermelho diferem do estado correto, e o erro se propagaria pelo resto da geração.' },
      ];
      let H8 = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H8 && H8.redraw() });
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'aceitos j', min: 0, max: 4, step: 1, value: S.j, onInput: (v) => { S.j = v; } }));
      UI.toggle(ctl, { label: 'guardar fotografias', value: true, onChange: (v) => { S.snaps = v; } });
      function card(g, x, y, M, o = {}) {
        const C = V.C();
        const cs = o.cs || 30, gap = 2, W = 4 * cs + 3 * gap;
        V.box(g, x - 6, y - 22, W + 12, W + 30, { r: 10, fill: o.bg || V.alpha(C.panel || C.muted, 0.95), stroke: o.stroke || V.alpha(C.muted, 0.5), lw: o.lw || 1.2 });
        V.text(g, o.title || '', x + W / 2, y - 7, { size: 11.5, font: 'mono', weight: 600, color: o.tcol || C.ink, align: 'center' });
        if (o.values === false) return W;
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
          const v = M[i][j], cx = x + j * (cs + gap), cy = y + i * (cs + gap);
          g.fillStyle = V.cmap(V.clamp(v / 1.6, -1, 1), 'div'); V.rr(g, cx, cy, cs, cs, 4); g.fill();
          if (o.diff && Math.abs(v - o.diff[i][j]) > 0.005) { g.strokeStyle = RED(); g.lineWidth = 2; V.rr(g, cx + 1, cy + 1, cs - 2, cs - 2, 4); g.stroke(); }
          const sv = Math.abs(v) < 0.05 ? '0,0' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(1).replace('.', ',');
          V.text(g, sv, cx + cs / 2, cy + cs / 2 + 4, { size: 11, font: 'mono', color: C.ink, align: 'center' });
        }
        return W;
      }
      H8 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const j = S.j;
        const wide = w >= 640;
        const pad = 14;
        // tokens
        const tsz = 24, tY = 30;
        const tW = wide ? w * 0.56 : w;
        const colW = (tW - 2 * pad) / 5;
        const nProc = i === 0 ? 0 : i === 1 ? V.clamp(Math.floor(el / 0.5) + 1, 1, 5) : 5;
        TOKS.forEach((tk, k) => {
          const x = pad + colW * (k + 0.5);
          let col = k === 0 ? C.muted : C.b, mood;
          if (i >= 2 && k >= 1) { if (k <= j) col = C.a; else { col = RED(); mood = 'sad'; } }
          tokChar(g, x, tY, tsz, col, { mood, alpha: i === 1 && k >= nProc ? 0.35 : 1 });
          V.text(g, tk, x, tY + tsz / 2 + 14, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
          if (i >= 2 && k >= 1) mark(g, x + tsz / 2 + 8, tY - tsz / 2 + 2, k <= j, 5);
        });
        // região das fotografias
        const aX = pad, aW = wide ? w * 0.56 - pad : w - 2 * pad;
        const aTop = 88, aBot = wide ? h - 12 : h * 0.62;
        const cs = Math.max(26, Math.min(32, (aW - 60) / 9));
        const Wc = 4 * cs + 6;
        if (i <= 2) {
          const n = i === 0 ? 0 : i === 1 ? nProc : 5;
          const sel = i === 2 ? (S.snaps ? j + 1 : 5) : n;
          const bx = aX + (aW - Wc) / 2 - 2 * 12, byy = aTop + 44;
          for (let s = 1; s <= n; s++) {
            if (s === sel) continue;
            const off = s - sel;
            const x = bx + 24 + off * 12, y = byy - off * 9;
            if (i === 2 && S.snaps) g.globalAlpha = 0.35;
            card(g, x, y, SN[s], { cs, values: false, title: '', stroke: V.alpha(C.b, 0.6), bg: V.alpha(C.panel || C.muted, 0.95) });
            g.globalAlpha = 1;
          }
          const good = i === 2 && S.snaps, bad = i === 2 && !S.snaps;
          const M = SN[sel];
          card(g, bx + 24, byy, M, { cs, title: sel === 0 ? 'S (estado oficial)' : `S(${sel})` + (good ? ' ✓ oficial' : bad ? ' ✗ contaminado' : ''), stroke: good ? C.a : bad ? RED() : C.b, lw: good || bad ? 2.4 : 1.4, tcol: good ? C.a : bad ? RED() : C.ink });
          const note = i === 0 ? 'antes do passo' : i === 1 ? `fotografia após “${TOKS[sel - 1]}”` : S.snaps ? `num_accepted_tokens = ${j + 1} → S(${j + 1})` : `sem cópias: sobrou S(5), com ${4 - j} rascunho(s) rejeitado(s) dentro`;
          const stackDrop = (i === 1 ? Math.max(0, n - 1) : i === 2 ? 4 : 0) * 9;
          V.text(g, note, aX + aW / 2, byy + Wc + 26 + stackDrop, { size: 12, color: bad ? RED() : C.ink, align: 'center', maxW: aW });
          V.text(g, 'fotografias do estado recorrente', aX, aTop, { size: 11.5, font: 'mono', color: C.muted });
          // fileira de miniaturas S(1)…S(5)
          const thY = byy + Wc + 44 + stackDrop;
          if (i >= 1 && thY + 24 < aBot + (wide ? 0 : 20)) {
            const tw5 = Math.min(58, (aW - 4 * 6) / 5), tx0 = aX + (aW - (5 * tw5 + 4 * 6)) / 2;
            for (let s = 1; s <= 5; s++) {
              if (s > n) continue;
              const x = tx0 + (s - 1) * (tw5 + 6);
              const official = i === 2 && S.snaps && s === j + 1;
              const dropped = i === 2 && S.snaps && s !== j + 1;
              const col = official ? C.a : dropped ? V.alpha(C.muted, 0.6) : bad && s === 5 ? RED() : C.b;
              chip(g, x, thY, tw5, 22, `S(${s})`, { size: 11, fill: official ? V.alpha(C.a, 0.3) : V.alpha(col, 0.12), stroke: col, color: dropped ? C.muted : C.ink, weight: official ? 700 : 500 });
              if (dropped) V.line(g, x + 4, thY + 11, x + tw5 - 4, thY + 11, { color: V.alpha(C.muted, 0.7), lw: 1.2 });
            }
            V.text(g, i === 2 ? (S.snaps ? 'as outras fotografias são descartadas ao fim do passo' : 'sem fotografias, só existe a última') : 'uma cópia do estado por token verificado', aX + aW / 2, thY + 40, { size: 11, color: C.muted, align: 'center', maxW: aW });
          }
        } else {
          V.text(g, 'estado correto × estado sem fotografias', aX, aTop, { size: 11.5, font: 'mono', color: C.muted, maxW: aW });
          const cs2 = Math.max(26, Math.min(30, (aW - 40) / 9.2));
          const W2 = 4 * cs2 + 6;
          const x1 = aX + aW / 2 - W2 - 14, x2 = aX + aW / 2 + 14, yy = aTop + 44;
          card(g, x1, yy, SN[j + 1], { cs: cs2, title: `correto S(${j + 1})`, stroke: C.a, lw: 2.2, tcol: C.a });
          card(g, x2, yy, SN[5], { cs: cs2, title: 'sem fotos S(5)', stroke: RED(), lw: 2.2, tcol: RED(), diff: SN[j + 1] });
          let dmax = 0; for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) dmax = Math.max(dmax, Math.abs(SN[5][a][b] - SN[j + 1][a][b]));
          V.text(g, j === 4 ? 'com j = 4 os dois coincidem: nada foi rejeitado' : `erro máximo por célula: ${fmt(dmax, 2)}`, aX + aW / 2, yy + W2 + 26, { size: 12, color: j === 4 ? C.a : RED(), align: 'center', maxW: aW });
          if (j < 4) {
            V.text(g, `desfazer à mão: ${4 - j}× S ← (S − v kᵀ) / γ`, aX + aW / 2, yy + W2 + 46, { size: 11.5, font: 'mono', color: C.ink, align: 'center', maxW: aW });
            V.text(g, 'exato no papel; em ponto flutuante, γ pequeno amplifica o erro', aX + aW / 2, yy + W2 + 63, { size: 11, color: C.muted, align: 'center', maxW: aW });
          }
        }
        // painel do KV e da conv
        const bX = wide ? w * 0.6 : pad, bW = wide ? w * 0.4 - pad : w - 2 * pad;
        const bTop = wide ? 88 : aBot + 10;
        V.text(g, 'KV: uma linha por token', bX, bTop, { size: 11.5, font: 'mono', color: C.muted });
        const rows = PAST.map((p) => ({ l: p, k: 'past' })).concat(TOKS.map((tk, k) => ({ l: tk, k: k === 0 ? 'anc' : 'd', idx: k })));
        const rh = wide ? 17 : 14, rW = Math.min(bW - 70, 160);
        const nRows = i === 0 ? 3 : i === 1 ? 3 + nProc : 8;
        const kvH = wide ? 8 * rh : 8 * rh;
        rows.slice(0, nRows).forEach((r, q) => {
          const y = bTop + 10 + q * rh;
          const rej = i >= 2 && r.k === 'd' && r.idx > j;
          V.text(g, r.l, bX + 52, y + rh / 2 + 4, { size: 11, font: 'mono', color: rej ? RED() : C.ink, align: 'right' });
          for (let c = 0; c < 6; c++) { g.fillStyle = rej ? V.alpha(RED(), 0.2) : V.alpha(r.k === 'past' ? C.muted : r.k === 'anc' ? C.a : C.b, 0.25 + 0.1 * ((c + q) % 3)); g.fillRect(bX + 60 + c * (rW / 6), y + 1, rW / 6 - 2, rh - 3); }
          if (rej) V.line(g, bX + 58, y + rh / 2, bX + 60 + rW, y + rh / 2, { color: RED(), lw: 1.6 });
        });
        if (i >= 2) V.text(g, 'rejeitadas: basta ignorar', bX + 60, bTop + 10 + 8 * rh + 14, { size: 11, color: C.muted });
        // conv
        const cTop = bTop + kvH + 52;
        V.text(g, i === 0 ? 'conv curta: K − 1 = 3 inputs' : 'conv curta: K − 1 + k = 7 colunas', bX, cTop, { size: 11.5, font: 'mono', color: C.muted, maxW: bW });
        const buf = i === 0 ? PAST : ['gato', 'subiu'].concat(TOKS);
        const ncell = buf.length, ccw = Math.min(46, (bW - 4) / 7 - 4);
        buf.forEach((b, q) => {
          const x = bX + q * (ccw + 4), y = cTop + 10;
          const vis = i !== 1 || q < 2 + nProc;
          if (!vis) return;
          const rejc = i >= 2 && q >= 3 && q - 2 > j;
          chip(g, x, y, ccw, 22, b.length > 5 && ccw < 44 ? b.slice(0, 4) + '…' : b, { size: 11, fill: rejc ? V.alpha(RED(), 0.15) : V.alpha(q < 2 || i === 0 ? C.muted : C.b, 0.2), stroke: rejc ? RED() : V.alpha(C.muted, 0.5) });
        });
        if (i >= 2) {
          const off = S.snaps ? j : 4;
          const x = bX + off * (ccw + 4) - 3, y = cTop + 6;
          const wwin = 3 * (ccw + 4) + 2;
          g.strokeStyle = S.snaps ? C.a : RED(); g.lineWidth = 2.4; V.rr(g, x, y, wwin, 30, 7); g.stroke();
          V.text(g, S.snaps ? `janela oficial: recuo = num_accepted − 1 = ${j}` : "sem recuo: a janela inclui rascunhos rejeitados", bX, y + 48, { size: 11.5, color: S.snaps ? C.a : RED(), maxW: bW });
        }
        void ncell;
      });
    })();

    /* ================================================ FIG 9 — semáforo da aceitação */
    (function () {
      const fig = root.querySelector('#fig-semaforo');
      adaptH(fig, 430, 560);
      const pato = [0.72, 0.45, 0.281, 0.176, 0.11];
      const SCN = {
        glmS: { label: 'GLM · contagem “structured” (k=7)', a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83] },
        glmP: { label: 'GLM · prosa (k=7)', a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06] },
        ds: { label: 'DeepSeek · Vision-Exp (k=6)', a: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15] },
        pato: { label: 'DeepSeek · palavra repetida (k=5)', a: pato, interp: true },
        bug: { label: 'bug vllm#53030 (ilustrativo, k=7)', a: [1, 1, 1, 1, 1, 1, 1] },
      };
      const S = { sc: 'glmP', drafts: 0, dtok: 0, acc: [], pend: 0, frame: 0 };
      const r9 = V.rng(909);
      function reset() { const k = SCN[S.sc].a.length; S.drafts = 0; S.dtok = 0; S.acc = new Array(k).fill(0); S.pend = 0; }
      reset();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'glmS', label: 'GLM contagem' }, { v: 'glmP', label: 'GLM prosa' }, { v: 'ds', label: 'DeepSeek' }, { v: 'pato', label: 'palavra repetida' }, { v: 'bug', label: 'bug #53030' }], value: S.sc, onChange: (v) => { S.sc = v; reset(); } });
      UI.button(ctl, 'restart (zera)', () => reset());
      const rD = UI.readout(ctl, 'drafts', '0'), rT = nocase(UI.readout(ctl, 'τ = 1 + Σ aᵢ', '—')), rV = UI.readout(ctl, 'veredito', 'SKIP');
      function simStep() {
        const a = SCN[S.sc].a, u = r9();
        let L = 0; while (L < a.length && a[L] > u) L++;
        S.drafts++; S.dtok += a.length;
        for (let q = 0; q < L; q++) S.acc[q]++;
      }
      function verdict() {
        if (S.drafts < 100) return 'SKIP';
        return S.acc[0] / S.drafts > 0.999 ? 'FAIL' : 'PASS'; // o script: pinned = (r > 0.999)
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused) { S.pend += dt * 22; while (S.pend >= 1) { S.pend -= 1; simStep(); } }
        const sc = SCN[S.sc], k = sc.a.length;
        const est = S.acc.map((x) => (S.drafts ? x / S.drafts : 0));
        const tau = 1 + est.reduce((s, x) => s + x, 0);
        const vd = verdict();
        if (++S.frame % 6 === 0) { rD.set(String(S.drafts)); rT.set(S.drafts ? fmt(tau, 2) : '—'); rV.set(vd); }
        const wide = w >= 640;
        const pad = 14;
        // semáforo
        const sx = pad, sy = 16, sw = 64, sh = 176;
        V.box(g, sx, sy, sw, sh, { r: 16, fill: V.alpha(C.ink, 0.1), stroke: V.alpha(C.ink, 0.35), lw: 1.5 });
        const lights = [['FAIL', RED()], ['SKIP', C.d], ['PASS', C.a]];
        lights.forEach(([nm, col], q) => {
          const cy = sy + 32 + q * 56, on = vd === nm;
          if (on) V.glow(g, sx + sw / 2, cy, 34, col, 0.9);
          V.dot(g, sx + sw / 2, cy, 20, on ? col : V.alpha(col, 0.18));
          V.text(g, nm, sx + sw / 2, cy + 4, { size: 11, font: 'mono', weight: 700, color: on ? (C.dark ? '#06140d' : '#fff') : V.alpha(C.ink, 0.5), align: 'center' });
        });
        const why = vd === 'SKIP' ? `SKIP: só ${S.drafts} drafts (< 100)` : vd === 'FAIL' ? `FAIL: pos0 = ${fmt(est[0], 3)} > 0,999` : `PASS: pos0 = ${fmt(est[0], 3)} ≤ 0,999 com ≥ 100 drafts`;
        // hodômetros
        const ox = sx + sw + 22, ow = w - ox - pad;
        V.text(g, sc.label, ox, 22, { size: 12.5, weight: 700, color: C.ink, maxW: ow });
        const odo = (x, y, lab, val) => {
          const s = String(val).padStart(6, '0');
          V.text(g, lab, x, y, { size: 11, font: 'mono', color: C.muted });
          for (let q = 0; q < s.length; q++) { V.box(g, x + q * 17, y + 6, 15, 22, { r: 3, fill: V.alpha(C.ink, 0.1), stroke: V.alpha(C.ink, 0.25), lw: 1 }); V.text(g, s[q], x + q * 17 + 7.5, y + 22, { size: 13, font: 'mono', weight: 700, color: C.ink, align: 'center' }); }
        };
        const accTot = S.acc.reduce((s2, x) => s2 + x, 0);
        const odX2 = wide ? 130 : Math.min(118, (ow - 4) / 2);
        odo(ox, 44, 'drafts_total', S.drafts);
        odo(ox + odX2, 44, 'draft_tokens', S.dtok);
        if (wide) odo(ox + odX2 * 2, 44, 'accepted_total', accTot);
        V.text(g, `aceitação global = ${S.dtok ? fmt(accTot / S.dtok, 3) : '—'} · τ = ${S.drafts ? fmt(tau, 2) : '—'}`, ox, 96, { size: 12, font: 'mono', color: C.ink, maxW: ow });
        const dec = S.drafts >= 100 ? (est[k - 1] < est[0] - 0.02 ? 'a curva decai ✓ (extra da figura)' : 'a curva não decai: plana em 1,00 ✗') : 'esperando tráfego…';
        V.text(g, dec, ox, 114, { size: 12, weight: 600, color: S.drafts >= 100 ? (est[k - 1] < est[0] - 0.02 ? C.a : RED()) : C.muted, maxW: ow });
        V.text(g, why, ox, 132, { size: 12, font: 'mono', weight: 600, color: vd === 'PASS' ? C.a : vd === 'FAIL' ? RED() : C.d, maxW: ow });
        // barras por posição
        const bTop = wide ? 150 : sy + sh + 24, bBot = h - 42;
        const bL = wide ? ox : pad + 26, bR = w - pad;
        const yv = (v) => bBot - v * (bBot - bTop - 14);
        [0, 0.5, 1].forEach((v) => { V.line(g, bL, yv(v), bR, yv(v), { color: C.grid, lw: 1 }); V.text(g, fmt(v, 1), bL - 4, yv(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        const cw = (bR - bL) / 7, bw = cw * 0.6;
        for (let q = 0; q < k; q++) {
          const x = bL + q * cw + (cw - bw) / 2;
          const v = est[q];
          V.box(g, x, yv(v), bw, Math.max(1, bBot - yv(v)), { r: 3, fill: V.alpha(S.sc === 'bug' ? RED() : C.a, 0.6), stroke: S.sc === 'bug' ? RED() : C.a, lw: 1 });
          const tv = sc.a[q];
          V.line(g, x - 3, yv(tv), x + bw + 3, yv(tv), { color: C.ink, lw: 1.5, dash: sc.interp && q > 0 && q < k - 1 ? [2, 3] : [5, 3] });
          V.text(g, 'pos' + q, x + bw / 2, bBot + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          if (S.drafts) V.text(g, fmt(v, 2), x + bw / 2, (Math.abs(yv(v) - 6 - yv(tv)) < 10 ? Math.min(yv(v), yv(tv)) - 6 : yv(v) - 6), { size: 11, font: 'mono', color: C.ink, align: 'center' });
        }
        V.text(g, sc.interp ? 'tracejado: medido nos extremos, interpolado no meio' : 'tracejado: perfil do cenário', bL, bBot + 32, { size: 11, color: C.muted, maxW: bR - bL });
      });
    })();
    /* ================================================ FIG 10 — igualdade × rejeição (TensorFold) */
    (function () {
      const fig = root.querySelector('#fig-tf-igualdade');
      const stage = fig.querySelector('.la-fig-stage');
      const WORDS = ['o', 'gato', 'cão', 'sobe', 'pula', 'no', 'telhado', 'muro', 'e', 'dorme', 'late', '.'];
      const NV = WORDS.length, N = 18, PROMPT = [0, 1];
      // hash sem estado: o mesmo (chave, posição, token) dá sempre o mesmo número
      const mix = (h) => { h = Math.imul(h ^ (h >>> 16), 0x7feb352d); h = Math.imul(h ^ (h >>> 15), 0x846ca68b); return (h ^ (h >>> 16)) >>> 0; };
      const Hh = (...a) => { let h = 0x811c9dc5; for (let i = 0; i < a.length; i++) h = mix(((h ^ (a[i] | 0)) + 0x9e3779b9) | 0); return h; };
      const U = (...a) => (Hh(...a) + 0.5) / 4294967296;
      const gum = (u) => -Math.log(-Math.log(u));
      const nrm = (...a) => Math.sqrt(-2 * Math.log(U(...a, 1))) * Math.cos(2 * Math.PI * U(...a, 2));
      // alvo de brinquedo: bigramas com preferências + um pouco de variação por posição
      const PREF = { 0: [1, 2], 1: [3, 4, 9], 2: [10, 4, 9], 3: [5], 4: [5, 0], 5: [6, 7], 6: [8, 11], 7: [8, 11], 8: [0, 9, 10], 9: [11, 8], 10: [11, 8], 11: [0] };
      const BASE = [];
      for (let a = 0; a < NV; a++) { const row = []; for (let b = 0; b < NV; b++) row.push((PREF[a].includes(b) ? 2.6 : -0.8) + 0.9 * U(5, a, b)); BASE.push(row); }
      const zt = (t, a) => BASE[a].map((v, b) => v + 0.6 * (U(11, t, b) - 0.5));
      const zq = (t, a, eps) => zt(t, a).map((v, b) => v + eps * nrm(13, t, a, b));
      const QUAL = [{ v: 'bom', label: 'bom', eps: 0.35 }, { v: 'medio', label: 'médio', eps: 1.0 }, { v: 'ruim', label: 'ruim', eps: 2.2 }];
      const sm = (z, T) => { const m = Math.max(...z); const e = z.map((x) => Math.exp((x - m) / T)); const s = e.reduce((x, y) => x + y, 0); return e.map((x) => x / s); };
      const amax = (a) => { let j = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[j]) j = i; return j; };
      const keyed = (z, t, T, K) => (T === 0 ? amax(z) : amax(z.map((v, b) => v / T + gum(U(K, t, b, 99)))));
      const hash = (a) => { let h = 0x811c9dc5; for (const x of a) { h ^= x; h = Math.imul(h, 16777619) >>> 0; } return (h >>> 0).toString(16).padStart(8, '0').slice(0, 6); };
      const S = { seed: 1234, T: 1, k: 4, q: 'medio', w: 0.7, ri: 1, hold: 0, play: true, t0: 0 };
      let D = null;
      function serial() {
        const out = []; let prev = PROMPT[1];
        for (let i = 0; i < N; i++) { const t = PROMPT.length + i; const x = keyed(zt(t, prev), t, S.T, S.seed); out.push(x); prev = x; }
        return out;
      }
      /** TensorFold: rascunho por argmax(log q + w·g), aceite enquanto rascunho == amostra chaveada */
      function runTF(eps) {
        const out = [], kinds = [], rounds = []; let prev = PROMPT[1];
        while (out.length < N) {
          const t0 = PROMPT.length + out.length, d = []; let p = prev;
          for (let j = 0; j < S.k; j++) { const t = t0 + j, z = zq(t, p, eps); const x = S.T === 0 ? amax(z) : amax(z.map((v, b) => v / S.T + S.w * gum(U(S.seed, t, b, 99)))); d.push(x); p = x; }
          let acc = 0, em = 0; p = prev;
          for (let j = 0; j <= S.k; j++) { const t = t0 + j, x = keyed(zt(t, p), t, S.T, S.seed); if (j < S.k && d[j] === x) { acc++; p = x; continue; } em = x; break; }
          rounds.push({ start: out.length, prev, d, acc, em });
          d.slice(0, acc).forEach((x) => { if (out.length < N) { out.push(x); kinds.push('ok'); } });
          if (out.length < N) { out.push(em); kinds.push('tgt'); }
          prev = out[out.length - 1];
        }
        return { out, kinds, rounds };
      }
      /** vLLM: rascunho sorteado de q, teste min(1, p/q), resíduo; um gerador sequencial */
      function runV(eps) {
        const r = V.rng((S.seed * 2654435761) >>> 0);
        const cat = (d) => { const u = r(); let s = 0; for (let i = 0; i < d.length; i++) { s += d[i]; if (u < s) return i; } return d.length - 1; };
        const out = [], kinds = [], rounds = []; let prev = PROMPT[1];
        while (out.length < N) {
          const t0 = PROMPT.length + out.length, d = [], qs = []; let p = prev;
          for (let j = 0; j < S.k; j++) { const t = t0 + j, z = zq(t, p, eps); let x; if (S.T === 0) x = amax(z); else { const q = sm(z, S.T); x = cat(q); qs.push(q); } d.push(x); p = x; }
          let acc = 0, em = 0; p = prev;
          for (let j = 0; j <= S.k; j++) {
            const t = t0 + j, z = zt(t, p);
            if (S.T === 0) { const x = amax(z); if (j < S.k && d[j] === x) { acc++; p = x; continue; } em = x; break; }
            const P = sm(z, S.T);
            if (j === S.k) { em = cat(P); break; }
            const q = qs[j];
            if (r() < Math.min(1, P[d[j]] / q[d[j]])) { acc++; p = d[j]; continue; }
            const res = P.map((v, i) => Math.max(0, v - q[i])); const s = res.reduce((x, y) => x + y, 0) || 1;
            em = cat(res.map((v) => v / s)); break;
          }
          rounds.push({ start: out.length, prev, d, acc, em });
          d.slice(0, acc).forEach((x) => { if (out.length < N) { out.push(x); kinds.push('ok'); } });
          if (out.length < N) { out.push(em); kinds.push('tgt'); }
          prev = out[out.length - 1];
        }
        return { out, kinds, rounds };
      }
      function recompute() {
        const eps = QUAL.find((q) => q.v === S.q).eps;
        const ref = serial();
        D = { ref, refH: hash(ref), tf: runTF(eps), vl: runV(eps), table: QUAL.map((q) => { const a = runTF(q.eps), b = runV(q.eps); return { q, tfH: hash(a.out), tfR: N / a.rounds.length, vlH: hash(b.out), vlR: N / b.rounds.length }; }) };
        S.ri = 1; S.hold = 0; S.t0 = performance.now();
      }
      recompute();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Rascunhador', options: QUAL.map((q) => ({ v: q.v, label: q.label })), value: S.q, onChange: (v) => { S.q = v; recompute(); } });
      UI.seg(ctl, { label: 'Ruído do alvo no rascunho (TF)', options: [{ v: 0, label: 'w = 0' }, { v: 0.7, label: 'w = 0,7' }, { v: 1, label: 'w = 1' }], value: S.w, onChange: (v) => { S.w = v; recompute(); } });
      nocase(UI.seg(ctl, { label: 'T', options: [{ v: 0, label: '0' }, { v: 0.7, label: '0,7' }, { v: 1, label: '1' }], value: S.T, onChange: (v) => { S.T = v; recompute(); } }));
      nocase(UI.slider(ctl, { label: 'k', min: 1, max: 7, step: 1, value: S.k, onInput: (v) => { S.k = v; recompute(); } }));
      UI.button(ctl, 'nova chave', () => { S.seed = (S.seed * 48271 + 11) % 99991 || 7; recompute(); });
      UI.button(ctl, 'rodada →', () => { S.play = false; bPlay.textContent = '▶ auto'; step(); }, { primary: true });
      const bPlay = UI.button(ctl, '⏸ pausar', () => { S.play = !S.play; bPlay.textContent = S.play ? '⏸ pausar' : '▶ auto'; });
      const maxR = () => Math.max(D.tf.rounds.length, D.vl.rounds.length);
      function step() {
        if (S.ri < maxR()) { S.ri++; S.t0 = performance.now(); S.hold = 0; }
        else if (++S.hold > 2) { S.ri = 1; S.hold = 0; S.t0 = performance.now(); }
      }
      V.every(1300, () => { if (S.play) step(); });
      let tfCol = null, lastC = null;
      function geom(g, w) {
        const pad = 12, size = w < 520 ? 10.5 : 11.5, ch = 20, gap = 4;
        let maxW = 0; WORDS.forEach((s) => { maxW = Math.max(maxW, tw(g, s, size, 'mono')); });
        const sw = Math.ceil(maxW + 12), ww = Math.ceil(maxW + 30);
        const cols = Math.max(4, Math.floor((w - 2 * pad + gap) / (sw + gap)));
        const tapeRows = Math.ceil((PROMPT.length + N) / cols);
        const tapeH = tapeRows * (ch + gap);
        let wx = pad + tw(g, 'janela:', 11) + 6, winRows = 1;
        for (let i = 0; i < S.k + 1; i++) { if (wx + ww > w - pad) { wx = pad + 12; winRows++; } wx += ww + gap; }
        if (wx + 16 + ww > w - pad) winRows++;
        const winH = winRows * (ch + gap) + 4;
        const tableH = 18 + 3 * 21 + 6;
        const H = 26 + (18 + tapeH + 12) + 2 * (20 + winH + tapeH + 12) + tableH + 8;
        return { pad, size, ch, gap, sw, ww, cols, tapeH, winH, winRows, tableH, H };
      }
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        if (C !== lastC) { lastC = C; tfCol = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; }
        const G = geom(g, w);
        const want = Math.max(380, Math.ceil(G.H)) + 'px';
        if (stage.style.height !== want) stage.style.height = want;
        const { pad, size, ch, gap, sw, ww, cols } = G;
        const el = (performance.now() - S.t0) / 1000;
        let y = 16;
        const narrow = w < 560;
        V.text(g, `chave ${S.seed} · T = ${fmt(S.T, S.T % 1 ? 1 : 0)} · k = ${S.k}` + (narrow ? '' : ` · rascunhador ${QUAL.find((q) => q.v === S.q).label}`) + ` · w = ${fmt(S.w, S.w % 1 ? 1 : 0)}`, pad, y, { size: 11, font: 'mono', color: C.muted });
        y += 12;
        const slot = (i, y0) => [pad + (i % cols) * (sw + gap), y0 + Math.floor(i / cols) * (ch + gap)];
        const hashChip = (x, yy, hs, ok, lab) => {
          const s = lab ? `${lab} ${hs}` : hs, wd = tw(g, s, 11, 'mono') + 14;
          chip(g, x - wd, yy, wd, 18, s, { size: 11, fill: V.alpha(ok == null ? C.muted : ok ? C.a : RED(), 0.16), stroke: ok == null ? V.alpha(C.muted, 0.5) : ok ? C.a : RED(), weight: 700 });
        };
        function tape(y0, toks, kinds, shown, diverge) {
          PROMPT.forEach((tk, i) => { const [x, yy] = slot(i, y0); chip(g, x, yy, sw, ch, WORDS[tk], { size, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), color: C.muted }); });
          for (let i = 0; i < N; i++) {
            const [x, yy] = slot(PROMPT.length + i, y0);
            if (i >= shown) { V.box(g, x, yy, sw, ch, { r: 6, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.18), lw: 1 }); continue; }
            const kd = kinds ? kinds[i] : null;
            const col = kd === 'ok' ? C.a : kd === 'tgt' ? C.d : C.b;
            const off = diverge && diverge[i];
            chip(g, x, yy, sw, ch, WORDS[toks[i]], { size, fill: V.alpha(col, kd ? 0.3 : 0.14), stroke: off ? RED() : V.alpha(col, 0.85), lw: off ? 2.2 : 1.2, weight: 600 });
          }
        }
        // ---- referência serial
        V.text(g, narrow ? 'referência serial' : 'referência serial ("draft": false)', pad, y + 12, { size: 12, weight: 700, color: C.ink });
        hashChip(w - pad, y, D.refH, null, 'hash');
        y += 18;
        tape(y, D.ref, null, N, null);
        y += G.tapeH + 12;
        // ---- faixas
        const lane = (run, title, short, col) => {
          const nR = Math.min(S.ri, run.rounds.length);
          const last = run.rounds[nR - 1];
          const shown = nR >= run.rounds.length ? N : run.rounds[nR].start;
          const done = nR >= run.rounds.length;
          V.dot(g, pad + 5, y + 8, 5, col);
          V.text(g, narrow ? short : title, pad + 16, y + 12, { size: 12, weight: 700, color: C.ink });
          const hs = hash(run.out);
          if (done) hashChip(w - pad, y - 1, hs, hs === D.refH, hs === D.refH ? '= ref' : '≠ ref');
          else V.text(g, `rodada ${nR}/${run.rounds.length}`, w - pad, y + 12, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          y += 20;
          // janela da última rodada
          let x = pad, yy = y;
          V.text(g, 'janela:', x, yy + 14, { size: 11, color: C.muted });
          x += tw(g, 'janela:', 11) + 6;
          const put = (lab, o) => { if (x + ww > w - pad) { x = pad + 12; yy += ch + gap; } chip(g, x, yy, ww, ch, lab, Object.assign({ size }, o)); x += ww + gap; };
          put(WORDS[last.prev], { fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.5), color: C.muted });
          last.d.forEach((tk, j) => {
            const vis = el > 0.15 * (j + 1) || nR < S.ri || !S.play;
            const ok = j < last.acc, rej = j === last.acc;
            const mk = !vis ? '' : ok ? ' ✓' : rej ? ' ✗' : '';
            put(WORDS[tk] + mk, { fill: V.alpha(ok && vis ? C.a : rej && vis ? RED() : C.b, vis && (ok || rej) ? 0.22 : 0.1), stroke: ok && vis ? C.a : rej && vis ? RED() : V.alpha(C.b, 0.6), color: j > last.acc ? C.muted : C.ink });
          });
          if (x + 16 + ww > w - pad) { x = pad + 12; yy += ch + gap; }
          V.text(g, '→', x + 2, yy + 14, { size: 13, color: C.muted }); x += 16;
          put(WORDS[last.em], { fill: V.alpha(C.d, 0.3), stroke: C.d, weight: 700 });
          y += G.winH;
          const div = run.out.map((tk, i) => tk !== D.ref[i]);
          tape(y, run.out, run.kinds, shown, div);
          y += G.tapeH + 12;
        };
        lane(D.tf, 'TensorFold · igualdade com a amostra chaveada', 'TensorFold · igualdade', tfCol);
        lane(D.vl, 'vLLM · amostragem por rejeição', 'vLLM · rejeição', C.glm || C.b);
        // ---- tabela: os três rascunhadores
        const cx = [pad, pad + Math.max(78, (w - 2 * pad) * 0.24), pad + Math.max(78, (w - 2 * pad) * 0.24) + (w - 2 * pad - Math.max(78, (w - 2 * pad) * 0.24)) / 2];
        V.line(g, pad, y - 4, w - pad, y - 4, { color: C.grid, lw: 1 });
        V.text(g, 'rascunhador', cx[0], y + 10, { size: 11, weight: 700, color: C.muted });
        V.text(g, narrow ? 'TF · tok/rodada' : 'TensorFold: hash · tok/rodada', cx[1], y + 10, { size: 11, weight: 700, color: C.muted });
        V.text(g, narrow ? 'vLLM · tok/rodada' : 'vLLM: hash · tok/rodada', cx[2], y + 10, { size: 11, weight: 700, color: C.muted });
        y += 18;
        D.table.forEach((r) => {
          const cur = r.q.v === S.q;
          if (cur) V.box(g, pad - 4, y, w - 2 * pad + 8, 19, { r: 5, fill: V.alpha(C.ink, 0.07) });
          V.text(g, r.q.label + (cur ? ' ◂' : ''), cx[0], y + 14, { size: 11.5, weight: cur ? 700 : 500, color: C.ink });
          const cell = (x0, hs, rr) => {
            const ok = hs === D.refH;
            V.text(g, hs, x0, y + 14, { size: 11.5, font: 'mono', weight: 700, color: ok ? C.a : RED() });
            V.text(g, (narrow ? (ok ? '✓' : '✗') : ok ? '= ref' : '≠ ref') + ' · ' + fmt(rr, 2), x0 + tw(g, hs, 11.5, 'mono', 700) + 5, y + 14, { size: 11, font: 'mono', color: C.muted });
          };
          cell(cx[1], r.tfH, r.tfR); cell(cx[2], r.vlH, r.vlR);
          y += 21;
        });
      });
    })();

    /* ================================================ FIG 11 — onde cortar a cadeia (fc, fnc, fcost) */
    (function () {
      const fig = root.querySelector('#fig-tf-corte');
      adaptH(fig, 430, 704);
      const VMS = [29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0], BLK = 3.08, NMAX = 7;
      const msK = (k) => VMS[k] + BLK;
      const PROF = {
        prosa: { label: 'prosa', a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06] },
        codigo: { label: 'código', a: [0.85, 0.7225, 0.5997, 0.4977, 0.3982, 0.3186, 0.2549] },
        contagem: { label: 'contagem', a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83] },
      };
      const POL = [
        { id: 'base', label: 'sem especulação' },
        { id: 'f7', label: 'f7 (sempre 7)' },
        { id: 'fc5', label: 'fc5:P (sem ruído)' },
        { id: 'fnc7', label: 'fnc7:P (receita)' },
        { id: 'fcost7', label: 'fcost7' },
        { id: 'orac', label: 'oráculo' },
      ];
      const S = { prof: 'prosa', P: 0.3, play: true, n: 0, cur: null, age: 0, scored: false, score: {} };
      let rr = V.rng(3511);
      const lg = (p) => Math.log(p / (1 - p)), sg = (x) => 1 / (1 + Math.exp(-x));
      const E = (c, k) => { let e = 1, r = 1; for (let j = 0; j < k; j++) { r *= c[j]; e += r; } return e; };
      const cutP = (c, N, P) => { let r = 1, k = 0; for (let j = 0; j < N; j++) { r *= c[j]; if (j > 0 && r < P) break; k = j + 1; } return k; };
      const cutCost = (c) => { let bk = 0, b = -1; for (let k = 0; k <= NMAX; k++) { const v = E(c, k) / msK(k); if (v > b) { b = v; bk = k; } } return bk; };
      function newRound() {
        const a = PROF[S.prof].a, base = a.map((v, i) => Math.min(0.97, Math.max(0.03, i ? v / a[i - 1] : v)));
        const sh = 0.9 * rr.normal();
        const c = base.map((v) => sg(lg(v) + sh + 0.7 * rr.normal()));
        let L = 0; while (L < NMAX && rr() < c[L]) L++;
        const cFc = c.map((v) => Math.pow(v, 1.6)); // ilustrativo: sem o ruído, a confiança subestima a aceitação
        const k = { base: 0, f7: NMAX, fc5: cutP(cFc, 5, S.P), fnc7: cutP(c, NMAX, S.P), fcost7: cutCost(c), orac: L };
        return { c, L, k };
      }
      function score(R) {
        POL.forEach((p) => {
          const k = R.k[p.id], tok = Math.min(k, R.L) + 1, ms = p.id === 'base' ? VMS[0] : msK(k);
          const s = S.score[p.id] || (S.score[p.id] = { tok: 0, ms: 0 });
          s.tok += tok; s.ms += ms;
        });
        S.n++;
      }
      function reset() {
        rr = V.rng(3511); S.score = {}; S.n = 0;
        for (let i = 0; i < 30; i++) score(newRound()); // aquecimento: o placar já começa com 30 rodadas
        S.cur = newRound(); S.age = 0; S.scored = false;
      }
      reset();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Texto', options: Object.keys(PROF).map((k) => ({ v: k, label: PROF[k].label })), value: S.prof, onChange: (v) => { S.prof = v; reset(); upd(); } });
      nocase(UI.slider(ctl, { label: 'limiar P', min: 0.05, max: 0.9, step: 0.05, value: S.P, fmt: (v) => fmt(v, 2), onInput: (v) => { S.P = v; reset(); upd(); } }));
      const bPlay = UI.button(ctl, '⏸ pausar', () => { S.play = !S.play; bPlay.textContent = S.play ? '⏸ pausar' : '▶ rodar'; });
      UI.button(ctl, 'zerar placar', () => { reset(); upd(); });
      const rN = UI.readout(ctl, 'rodadas', '0'), rB = UI.readout(ctl, 'melhor regra real', '—');
      const tps = (id) => { const s = S.score[id]; return s && s.ms ? (1000 * s.tok) / s.ms : 0; };
      function upd() {
        rN.set(String(S.n));
        let best = null; ['f7', 'fc5', 'fnc7', 'fcost7'].forEach((id) => { if (!best || tps(id) > tps(best)) best = id; });
        rB.set(POL.find((p) => p.id === best).label.split(' ')[0].replace(':P', ':' + fmt(S.P, 2)));
      }
      upd();
      const colOf = (C, id) => ({ base: C.muted, f7: C.b, fc5: C.c, fnc7: C.a, fcost7: C.d, orac: C.ink })[id];
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (!LA.paused && S.play) {
          S.age += dt;
          if (!S.scored && S.age > 0.9) { score(S.cur); S.scored = true; upd(); }
          if (S.age > 2.2) { S.cur = newRound(); S.age = 0; S.scored = false; }
        }
        const R = S.cur, wide = w >= 640, pad = 12;
        const A = wide ? { x: pad, y: 8, w: w * 0.52 - pad, h: h - 16 } : { x: pad, y: 8, w: w - 2 * pad, h: 364 };
        const B = wide ? { x: w * 0.55, y: 8, w: w * 0.45 - pad, h: h - 16 } : { x: pad, y: A.y + A.h + 8, w: w - 2 * pad, h: h - A.h - 24 };
        const grow = V.ease.out(V.clamp(S.age / 0.3, 0, 1));
        // ---------- painel A: a cadeia
        {
          V.text(g, `rodada ${S.n + (S.scored ? 0 : 1)} · cadeia do DFlash2 (${PROF[S.prof].label})`, A.x, A.y + 12, { size: 12, weight: 700, color: C.ink, maxW: A.w });
          let lx = A.x; const ly = A.y + 29;
          [[C.b, 'ĉᵢ prevista'], [C.a, 'sobrevivência ∏ĉ']].forEach(([c, l]) => { V.dot(g, lx + 5, ly - 4, 4, c); V.text(g, l, lx + 13, ly, { size: 11, color: C.muted }); lx += 24 + tw(g, l, 11); });
          const L0 = A.x + 58, R0 = A.x + A.w - 92, T0 = A.y + 42, B0 = A.y + (wide ? 178 : 168);
          const cw = (R0 - L0) / NMAX, yv = (v) => B0 - v * (B0 - T0);
          [0, 0.5, 1].forEach((v) => { V.line(g, L0, yv(v), R0, yv(v), { color: C.grid, lw: 1 }); V.text(g, fmt(v, 1), L0 - 5, yv(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
          V.line(g, L0, yv(S.P), R0, yv(S.P), { color: C.d, lw: 1.4, dash: [5, 4] });
          V.text(g, `P = ${fmt(S.P, 2)}`, R0 + 6, yv(S.P) + 4, { size: 11, font: 'mono', weight: 700, color: C.d });
          let reach = 1; const pts = [];
          for (let i = 0; i < NMAX; i++) {
            const v = R.c[i] * grow, x = L0 + i * cw + cw * 0.2;
            V.box(g, x, yv(v), cw * 0.6, Math.max(1, B0 - yv(v)), { r: 3, fill: V.alpha(C.b, 0.45), stroke: C.b, lw: 1 });
            reach *= R.c[i]; pts.push([L0 + i * cw + cw / 2, yv(reach * grow)]);
          }
          g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
          pts.forEach(([x, y]) => V.dot(g, x, y, 3.2, C.a));
          // veredito do alvo
          const vAge = S.age - 0.3;
          for (let i = 0; i < NMAX; i++) {
            const x = L0 + i * cw + cw / 2, yy = B0 + 14;
            V.text(g, 'd' + (i + 1), x, yy + 2, { size: 11, font: 'mono', color: C.muted, align: 'center' });
            if (vAge > 0.04 * i || !S.play) { if (i < R.L) mark(g, x, yy + 16, true, 5); else if (i === R.L) mark(g, x, yy + 16, false, 5); else V.text(g, '·', x, yy + 20, { size: 13, color: C.muted, align: 'center' }); }
          }
          // trilhos: o que cada regra propõe, o que é aceito
          const rails = POL.filter((p) => p.id !== 'base');
          let ry = B0 + 60;
          const rh = wide ? Math.min(25, (A.y + A.h - ry) / rails.length) : 25;
          V.text(g, 'proposto (contorno) · aceito (cheio) · tokens / ms da rodada', A.x, ry - 4, { size: 11, color: C.muted, maxW: A.w });
          rails.forEach((p) => {
            const k = R.k[p.id], col = colOf(C, p.id), acc = Math.min(k, R.L);
            V.text(g, p.label.split(' ')[0].replace(':P', ''), A.x, ry + rh * 0.62, { size: 11, font: 'mono', weight: 600, color: col === C.ink ? C.ink : col });
            for (let i = 0; i < NMAX; i++) {
              const x = L0 + i * cw + 3, bw = cw - 6;
              if (i < k) V.box(g, x, ry + 4, bw, rh - 9, { r: 3, fill: i < acc && (S.age > 0.3 + 0.04 * NMAX || !S.play) ? V.alpha(col, 0.55) : V.alpha(col, 0.08), stroke: col, lw: 1.1 });
            }
            const shown = S.age > 0.3 + 0.04 * NMAX || !S.play;
            V.text(g, `${shown ? acc + 1 : '?'} tok · ${fmt(msK(k), 0)} ms`, R0 + 6, ry + rh * 0.62, { size: 11, font: 'mono', color: C.ink });
            ry += rh;
          });
        }
        // ---------- painel B: velocidade prevista desta cadeia e placar
        {
          const b1h = wide ? B.h * 0.42 : 150;
          V.text(g, 'velocidade prevista desta cadeia', B.x, B.y + 12, { size: 12, weight: 700, color: C.ink, maxW: B.w });
          V.text(g, 'E[tokens | k] / ms(k), em tok/s', B.x, B.y + 27, { size: 11, color: C.muted, maxW: B.w });
          const L1 = B.x + 30, R1 = B.x + B.w - 6, T1 = B.y + 38, B1 = B.y + b1h - 18;
          const vals = []; for (let k = 0; k <= NMAX; k++) vals.push((1000 * E(R.c, k)) / msK(k));
          const ymax = Math.max(40, Math.ceil(Math.max(...vals) / 20) * 20 + 10);
          const xs = (k) => L1 + (k / NMAX) * (R1 - L1), ys = (v) => B1 - (v / ymax) * (B1 - T1);
          const st = niceStep(ymax, 3);
          for (let v = 0; v <= ymax + 1e-9; v += st) { V.line(g, L1, ys(v), R1, ys(v), { color: C.grid, lw: 1 }); V.text(g, String(Math.round(v)), L1 - 5, ys(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
          const base = 1000 / VMS[0];
          V.line(g, L1, ys(base), R1, ys(base), { color: C.muted, lw: 1.2, dash: [4, 4] });
          V.text(g, 'sem especulação', R1, ys(base) + 14, { size: 11, color: C.muted, align: 'right' });
          g.beginPath(); vals.forEach((v, k) => (k ? g.lineTo(xs(k), ys(v * grow + base * (1 - grow))) : g.moveTo(xs(k), ys(v * grow + base * (1 - grow))))); g.strokeStyle = C.d; g.lineWidth = 2.2; g.stroke();
          vals.forEach((v, k) => V.dot(g, xs(k), ys(v * grow + base * (1 - grow)), 2.8, C.d));
          for (let k = 0; k <= NMAX; k++) V.text(g, String(k), xs(k), B1 + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          const ks = R.k.fcost7, sx = xs(ks), sy = ys(vals[ks] * grow + base * (1 - grow)) - 13;
          g.beginPath(); for (let q = 0; q < 10; q++) { const r = q % 2 ? 3.6 : 8, an = -Math.PI / 2 + (q * Math.PI) / 5; const px = sx + r * Math.cos(an), py = sy + r * Math.sin(an); if (q === 0) g.moveTo(px, py); else g.lineTo(px, py); } g.closePath(); g.fillStyle = C.d; g.fill();
          V.text(g, `k* = ${ks}`, Math.min(sx + 10, R1 - 34), sy - 6, { size: 11, font: 'mono', weight: 700, color: C.ink });
          // placar
          let y = B.y + b1h + 17;
          V.text(g, `placar após ${S.n} rodadas (tok/s)`, B.x, y, { size: 12, weight: 700, color: C.ink, maxW: B.w });
          y += 10;
          const lw = Math.min(140, B.w * 0.42), bx = B.x + lw, bw = B.w - lw - 46;
          let mx = 1; POL.forEach((p) => { mx = Math.max(mx, tps(p.id)); });
          const rowH = Math.min(26, (B.y + B.h - y - 4) / POL.length);
          POL.forEach((p) => {
            const v = tps(p.id), col = colOf(C, p.id);
            V.text(g, p.label.replace(':P', ':' + fmt(S.P, 2)), B.x, y + rowH * 0.66, { size: 11, color: C.ink, maxW: lw - 6 });
            V.box(g, bx, y + 4, Math.max(2, (v / mx) * bw), rowH - 9, { r: 3, fill: V.alpha(col, p.id === 'orac' ? 0.18 : 0.5), stroke: col, lw: 1 });
            V.text(g, fmt(v, 1), bx + Math.max(2, (v / mx) * bw) + 5, y + rowH * 0.66, { size: 11, font: 'mono', weight: 700, color: C.ink });
            y += rowH;
          });
        }
      });
    })();
  },
});

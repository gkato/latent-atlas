LA.chapter({
  id: 'medindo-qualidade',
  part: 4,
  order: 25,
  title: 'Medindo o estrago: perplexidade, KLD e paridade numérica',
  navTitle: 'Medindo qualidade',
  subtitle: 'Por que “parece bom” não basta: como medir, com rigor e contra o ruído do próprio sistema, se uma quantização, um kernel ou um patch deixou o modelo mais burro.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['KLD', 'perplexidade', 'NLL', 'top-1', 'paridade numérica', 'determinismo', 'RULER', 'tool calling', 'canário'],
  goals: [
    'Calcular e interpretar <b>NLL</b>, <b>perplexidade</b>, <b>KLD</b> e <b>concordância top-1</b> em teacher forcing, sabendo o que cada uma esconde',
    'Montar um <b>painel calibrado</b> contra o ruído stock-vs-stock e ler veredictos NOT-WORSE, EQUIV e INCONCLUSIVE',
    'Explicar por que kernels divergem (<b>não-associatividade</b>), como se testa <b>paridade numérica</b>, quanto custa o determinismo e o que muda quando o motor garante <b>exatidão</b>',
    'Escolher e usar <b>testes funcionais</b> (agulhas, RULER, garble, loops, ferramentas, canários) como gates de uma receita',
    'Documentar resultados de forma reproduzível e decidir na <b>fronteira de Pareto</b> qualidade × custo',
  ],
  prereqs: ['quantizacao-fundamentos', 'hadamard-trelica-exl3'],
  summary: [
    'Estrago numérico é <b>silencioso</b>: mede-se numa escada de evidências — pesos (rel RMSE, cosseno) → kernels (paridade) → distribuições (NLL, KLD, top-1) → tarefas e contexto longo → serviço (canário, ferramentas).',
    'NLL = média de \\(-\\ln p\\) do token verdadeiro em teacher forcing; PPL = \\(e^{\\text{NLL}}\\). A média esconde erros que se cancelam; a <b>KLD</b> compara distribuições inteiras (≥ 0, assimétrica, em nats), e a versão top-k da API é um proxy que tipicamente subestima quando o top-k cobre quase toda a massa (e pode errar para os dois lados quando não cobre).',
    'O GLM serve EXL3 4 bpw porque a KLD do painel independente deu 0,024555 nats — ≈ o FP8 oficial na mesma stack, com 54% dos bytes — contra 0,060535 do NVFP4.',
    'Gates absolutos falham porque o <b>próprio stock varia</b> (quase-empates flipam). O painel calibrado classifica posições (CONF/TIE/UNSTABLE) e padroniza a KL pelo nulo stock-vs-stock (gate 2×), mas é triagem: com 2 capturas por braço, acusou o próprio controle. Veredictos de adoção (NOT-WORSE) vêm do <code>kl_panel</code>, com controles positivos e protocolos congelados.',
    'Soma em ponto flutuante <b>não é associativa</b>: ordens variáveis (split-K, top-k, finalize do MoE) mudam bits e flipam quase-empates. O patch de determinismo do Qwen zera o espalhamento (0,25/6,07 → 0/0 nats) a ~−4%. A versão TensorFold do GLM faz da exatidão um contrato (rascunhado = serial, concorrente = solo): o nulo vira zero e “nada mudou” vira <b>teste por hash</b>, mas só para mudanças que preservam os bits; os defaults com perdas (densos q4, KV FP8) continuam precisando de juiz.',
    'Contexto longo exige testes próprios, frios e com comprimento conhecido: agulhas, RULER-lite (verificado por <code>/tokenize</code>, tarefa na cabeça e palheiro anexado; gate 8/8), garble sweep (estimado a 1,56 tok/palavra; CLEAN até ~900K) e o detector de loop por novidade de 8-gramas.',
    'No topo, testes funcionais pegam o que a KLD não vê: canário de boot (OK a T=0 + aceitação em ≥ 64 rascunhos), baterias de ferramentas (7/7, 8/8) e o gate de 145 casos. Benchmarks de tarefa têm resolução grossa: o q4 da versão TensorFold empatou no GSM8K (250) e no HumanEval (164), e 48 respostas a prompts curtos em francês pegaram a perda do fim de turno (12/48 contra 1/48). Documente data, commits, imagem, protocolo e rótulos Measured/Interpretation.',
  ],
  quiz: [
    { q: 'Dois candidatos têm a mesma NLL que a referência (1,331 nats), mas o candidato B tem KLD média de 0,03 nats e o A, de 0,002. O que isso indica?', o: ['Nada: se a NLL é igual, a qualidade é igual', 'B distorceu as distribuições; na média da NLL, pioras e melhoras se cancelaram', 'A KLD de B está errada, porque a KLD não pode ser maior que a diferença de NLL', 'B é melhor, porque tem mais variedade'], a: 1, e: 'A NLL olha só a probabilidade do token verdadeiro e faz a média; posições que melhoram compensam posições que pioram. A KLD compara a distribuição inteira em cada posição, e cada termo agregado é ≥ 0, então compensações não a zeram.' },
    { q: 'Numa posição confiante (o top-20 cobre quase toda a massa), por que a “KL proxy” calculada sobre as listas que a API devolve tende a <i>subestimar</i> a KL verdadeira?', o: ['Porque usa logaritmo na base 2', 'Porque ignora a massa de probabilidade fora das listas top-k e renormaliza só o suporte comum', 'Porque o vLLM arredonda os logprobs', 'Porque a KL é simétrica'], a: 1, e: 'Pela regra da cadeia, \\(D(P\\|Q) = D_{\\text{bin}} + P(S)\\,D_S + P(\\bar S)\\,D_{\\bar S} \\ge P(S)\\,D_S\\): com \\(P(S)\\approx 1\\), a verdadeira é pelo menos o proxy. Quando o top-k cobre pouca massa, o proxy pode até passar da verdadeira. Serve para comparar braços medidos do mesmo jeito, não com a KLD de outro script.' },
    { q: 'No estudo selado do thin-decode do GLM, o candidato teve 0 flips e KL dentro do nulo, mas o veredicto foi INCONCLUSIVE. Por quê?', o: ['Porque o candidato era mais lento', 'Porque houve flips escondidos', 'Porque o gate absoluto “KL ≤ 0,01” reprovou o próprio stock-contra-stock (nulo de 0,0142–0,0146); o defeito era da régua', 'Porque faltou rodar o GSM8K'], a: 2, e: 'Quando o controle negativo (stock vs stock) falha num gate, o instrumento não está apto a julgar. Houve ainda uma segunda falha de controle (uma posição stock bimodal estourou a transferência de τ). O registro honesto é “inconclusivo com evidência relativa limpa”, não “aprovado”.' },
    { q: 'Por que ordenar os índices do top-k do indexador QSA (<code>VLLM_QSA_DET_TOPK=1</code>) torna a saída reprodutível?', o: ['Porque muda quais blocos são selecionados', 'Porque a mesma seleção passa a ser visitada sempre na mesma ordem, e a soma em ponto flutuante depende da ordem', 'Porque desliga a atenção esparsa', 'Porque fixa a semente do sampler'], a: 1, e: 'A seleção não muda; muda só a ordem de visita. Como \\((a+b)+c \\ne a+(b+c)\\) em ponto flutuante, ordem fixa ⇒ bits fixos ⇒ quase-empates não flipam. Junto com o finalize não fundido do MoE, o espalhamento foi de 0,25/6,07 para 0/0 nats, a ~−4%.' },
    { q: 'Um traço de raciocínio termina com <code>finish_reason=length</code>. O detector mostra a novidade de 8-gramas caindo abaixo de 2% por volta do token 12K e ficando abaixo disso até o fim. O que fazer?', o: ['Aumentar max_tokens: é cauda longa', 'Nada: é repetição transitória', 'Tratar como loop: aumentar max_tokens não ajuda; investigar amostragem, numérica ou o prompt', 'Desligar o prefix caching'], a: 2, e: 'Loop = colapso da novidade que nunca se recupera (3 janelas abaixo do limiar e persistência até o fim). Numa cauda longa, a novidade fica alta; numa repetição transitória, ela volta a subir.' },
    { q: 'Por que o canário do GLM exige pelo menos 64 tokens rascunhados (<code>GLM53_WARMUP_CANARY_MIN_DRAFTS</code>) antes de julgar a aceitação?', o: ['Porque o DFlash só funciona depois de 64 tokens', 'Porque, com amostra pequena, até um motor saudável pode ter aceitado zero por azar e o canário daria falso alarme', 'Porque o /metrics só atualiza a cada 64 tokens', 'Porque 64 é o tamanho do bloco de KV'], a: 1, e: 'Se cada passo de verificação aceita algo com probabilidade \\(a\\), zerar em \\(n\\) passos tem probabilidade \\((1-a)^n\\), que despenca com \\(n\\). Já um motor degenerado (\\(a\\approx 0\\)) zera sempre. Com pouco tráfego, o script diz “não dá para julgar” em vez de “degenerado”.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;

    /* ================= utilidades locais ================= */
    const FC = {};
    const setF = (g, size, font, weight) => { const k = size + '|' + (font || 'sans') + '|' + (weight || 400); g.font = FC[k] || (FC[k] = V.font(size, font || 'sans', weight || 400)); };
    const tw = (g, s, size = 12, font = 'sans', weight = 400) => { setF(g, size, font, weight); return g.measureText(String(s)).width; };
    const clip = (g, s, maxW, size = 11, font = 'sans', weight = 400) => {
      s = String(s); if (maxW <= 6) return '';
      setF(g, size, font, weight);
      if (g.measureText(s).width <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (g.measureText(s.slice(0, m) + '…').width <= maxW) lo = m; else hi = m - 1; }
      return s.slice(0, lo) + '…';
    };
    const wrapLines = (g, s, maxW, size = 12, font = 'sans', weight = 400) => {
      setF(g, size, font, weight);
      const out = []; let cur = '';
      String(s).split(/\s+/).forEach((wd) => { const t2 = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(t2).width > maxW) { out.push(cur); cur = wd; } else cur = t2; });
      if (cur) out.push(cur);
      return out;
    };
    const drawWrap = (g, s, x, y, maxW, lh, o = {}) => { const L = wrapLines(g, s, maxW, o.size || 12, o.font || 'sans', o.weight || 400); L.forEach((l, i) => V.text(g, l, x, y + i * lh, o)); return L.length; };
    const fitStage = (fig, fn) => {
      const st = fig.querySelector('.la-fig-stage'); if (!st) return;
      const apply = () => { const w = st.clientWidth || 600; const hh = fn(w); if (Math.abs(st.getBoundingClientRect().height - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(() => apply()); ro.observe(st);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    };
    const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d).replace('.', ',') : '—');
    const pct = (x, d = 1) => f(100 * x, d) + '%';
    const thou = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, '.');
    const softmax = (l) => { let m = -Infinity; for (const x of l) if (x > m) m = x; const e = l.map((x) => Math.exp(x - m)); let s = 0; for (const x of e) s += x; return e.map((x) => x / s); };
    const klDiv = (p, q) => { let s = 0; for (let i = 0; i < p.length; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / Math.max(q[i], 1e-300)); return Math.max(0, s); };
    const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
    const showEl = (x, on) => { if (x && x.el) x.el.style.display = on ? '' : 'none'; else if (x && x.style) x.style.display = on ? '' : 'none'; };
    const tri = (g, x, y, s, col, dir = 1) => { g.fillStyle = col; g.beginPath(); g.moveTo(x, y); g.lineTo(x - s, y - dir * s * 1.4); g.lineTo(x + s, y - dir * s * 1.4); g.closePath(); g.fill(); };
    const badge = (g, s, x, y, col, o = {}) => {
      const sz = o.size || 10.5, fw = tw(g, s, sz, 'mono', 700) + 12, hh = sz + 8;
      const xx = o.align === 'right' ? x - fw : o.align === 'center' ? x - fw / 2 : x;
      V.box(g, xx, y - hh / 2, fw, hh, { r: hh / 2, fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.75), lw: 1.2 });
      V.text(g, s, xx + fw / 2, y + 0.5, { size: sz, font: 'mono', weight: 700, align: 'center', base: 'middle', color: col });
      return fw;
    };

    /* =========================================================
       F1 — Quem apita? A escada de evidências
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-detectores'); if (!fig) return;
      fitStage(fig, (w) => (w < 560 ? 590 : 385));
      const stages = ['Pesos', 'Kernel', 'Logits', 'Texto', 'Serviço'];
      const subs = ['bytes', 'operações', 'distribuições', 'respostas', 'API e boot'];
      const dets = [
        { k: 'rmse', l: 'rel RMSE · cosseno', s: 0, cad: 'por checkpoint' },
        { k: 'par', l: 'Paridade de kernel', s: 1, cad: 'por kernel novo' },
        { k: 'kld', l: 'KLD · NLL · top-1', s: 2, cad: 'mudou numérica' },
        { k: 'task', l: 'Tarefas', s: 3, cad: 'GSM8K, reasoning' },
        { k: 'long', l: 'Contexto longo', s: 3, cad: 'mexeu em KV' },
        { k: 'loop', l: 'Gerações longas', s: 3, cad: 'detector de loop' },
        { k: 'tool', l: 'Ferramentas', s: 4, cad: 'bateria, 145' },
        { k: 'canary', l: 'Canário de boot', s: 4, cad: 'em todo boot' },
      ];
      const defects = [
        { v: 'q', label: 'Pesos grosseiros (NVFP4 no lugar do EXL3)', o: 0, hit: { rmse: 1, par: 0, kld: 2, task: 1, long: 1, loop: 0, tool: 0, canary: 0 },
          note: 'O erro nasce nos bytes. O rel RMSE mostra que o NVFP4 erra mais, mas os dois formatos de 4 bits ficam acima do gate de FP8 (0,06): é só um sinal fraco. Quem mede o tamanho do estrago é a KLD (0,0605 contra 0,0246 no painel do GLM); tarefas só notam se a diferença for grande. A paridade de kernel não vê nada: o kernel faz direitinho a conta com os pesos errados.' },
        { v: 'ord', label: 'Kernel novo que soma noutra ordem', o: 1, hit: { rmse: 0, par: 3, kld: 3, task: 0, long: 0, loop: 0, tool: 0, canary: 0 },
          note: 'Não é estrago, é ruído. A paridade passa porque a diferença cabe em 4× o balanço stock-contra-stock, e o painel calibrado dá KL dentro do nulo. Um gate absoluto mal calibrado daria alarme falso justamente aqui.' },
        { v: 'kv', label: 'K do KV em FP8 muda a seleção top-k', o: 2, hit: { rmse: 0, par: 0, kld: 1, task: 1, long: 2, loop: 0, tool: 0, canary: 0 },
          note: 'Em textos curtos a distribuição quase não muda; o estrago aparece em profundidade, quando o indexador esparso escolhe outros blocos. Quem pega são agulhas, RULER e raciocínio longo — por isso o reasoning_check nasceu junto com o KV FP8 do Qwen.' },
        { v: 'deg', label: 'Motor degenerado no boot (#249)', o: 1, hit: { rmse: 0, par: 0, kld: 2, task: 2, long: 2, loop: 1, tool: 2, canary: 2 },
          note: 'Tudo “!!!!”, aceitação ~0 e /health verde. Qualquer teste pegaria — se rodasse. Só o canário roda em todo boot, antes do READY, e por isso é o detector que importa aqui.' },
        { v: 'dsml', label: 'Sintaxe de ferramenta quebrada', o: 4, hit: { rmse: 0, par: 0, kld: 0, task: 0, long: 0, loop: 0, tool: 2, canary: 0 },
          note: 'Truncamento no meio de uma chamada DSML ou uma tag malformada: a KLD de um texto em prosa não muda nada. Só a bateria de ferramentas e o gate de 145 casos exercitam o parser, o finish_reason e o esquema.' },
        { v: 'loop', label: 'Loop de raciocínio', o: 3, hit: { rmse: 0, par: 0, kld: 0, task: 1, long: 0, loop: 2, tool: 0, canary: 0 },
          note: 'Teacher forcing nunca deixa o modelo seguir os próprios passos, então a KLD não vê loops. Eles aparecem em gerações longas amostradas; o detector de novidade de 8-gramas separa loop de cauda longa.' },
        { v: 'eos', label: 'Densos em 4 bits perdem o fim de turno (GLM·TF, #18)', o: 0, hit: { rmse: 1, par: 0, kld: 1, task: 0, long: 0, loop: 2, tool: 0, canary: 0 },
          note: 'O erro nasce nos bytes (o q4 erra mais que o fp8: sinal fraco). GSM8K e HumanEval em inglês empataram, e uma KLD num corpus genérico provavelmente veria pouco (o motor v0.6.0 da receita nem devolve logprobs). O sintoma mora em prompts curtos de código em francês: 12 de 48 respostas correm até max_tokens, contra 1 de 48 com fp8 ou bf16. Pega quem conta finish_reason=length nos prompts certos.' },
      ];
      const S = { d: defects[0], reset: true, t0: 0 };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Estrago', options: defects.map((d) => ({ v: d.v, label: d.label })), value: 'q', onChange: (v) => { S.d = defects.find((d) => d.v === v) || defects[0]; S.reset = true; } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.red, 'alarme'], [C0.d, 'sinal fraco'], [C0.e, 'mede e aprova'], [C0.muted, 'não enxerga']]);
      const drawBug = (g, x, y, t, C) => {
        const wig = Math.sin(t * 14) * 2.2;
        g.strokeStyle = C.red; g.lineWidth = 1.6;
        for (let k = -1; k <= 1; k++) {
          const lx = x + k * 5, s = k === 0 ? wig : -wig;
          g.beginPath(); g.moveTo(lx, y - 3); g.lineTo(lx - 4 + s, y - 11); g.stroke();
          g.beginPath(); g.moveTo(lx, y + 3); g.lineTo(lx - 4 - s, y + 11); g.stroke();
        }
        g.fillStyle = C.red;
        g.beginPath(); g.ellipse(x, y, 11, 7, 0, 0, Math.PI * 2); g.fill();
        g.beginPath(); g.arc(x + 11, y, 4.5, 0, Math.PI * 2); g.fill();
        g.fillStyle = '#fff';
        g.beginPath(); g.arc(x + 12.4, y - 1.8, 1.3, 0, Math.PI * 2); g.fill();
        g.beginPath(); g.arc(x + 12.4, y + 1.8, 1.3, 0, Math.PI * 2); g.fill();
        g.strokeStyle = 'rgba(0,0,0,.25)'; g.lineWidth = 1; g.beginPath(); g.moveTo(x - 9, y); g.lineTo(x + 7, y); g.stroke();
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.reset) { S.t0 = t; S.reset = false; }
        const d = S.d, narrow = w < 560, pad = 14;
        const n = stages.length, gap = narrow ? 5 : 16;
        const bw = (w - 2 * pad - gap * (n - 1)) / n, by = 12, bh = 46;
        const cx = (s) => pad + s * (bw + gap) + bw / 2;
        const el = t - S.t0, period = 6.5;
        const ph = (el % period) / period;
        const walk = V.ease.inOut(V.clamp(ph / 0.62, 0, 1));
        const pos = d.o + (n - 1 - d.o) * walk;
        for (let s = 0; s < n; s++) {
          const x = pad + s * (bw + gap);
          const hit = s >= d.o && pos >= s - 0.02;
          V.box(g, x, by, bw, bh, { r: 9, fill: hit ? V.alpha(C.red, 0.13) : V.alpha(C.b, 0.1), stroke: hit ? V.alpha(C.red, 0.7) : V.alpha(C.b, 0.45), lw: hit ? 1.8 : 1.2 });
          const fs = narrow ? 11 : 12.5;
          V.text(g, clip(g, stages[s], bw - 6, fs, 'sans', 600), x + bw / 2, by + 18, { size: fs, weight: 600, align: 'center', base: 'middle', color: C.ink });
          V.text(g, clip(g, subs[s], bw - 6, narrow ? 9 : 10, 'mono'), x + bw / 2, by + 34, { size: narrow ? 9 : 10, font: 'mono', align: 'center', base: 'middle', color: C.muted });
          if (!narrow && s < n - 1) V.arrow(g, x + bw + 2, by + bh / 2, x + bw + gap - 2, by + bh / 2, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 });
        }
        const ly = by + bh + 24;
        V.line(g, cx(0), ly, cx(n - 1), ly, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 5] });
        const bx = cx(0) + (cx(n - 1) - cx(0)) * (pos / (n - 1));
        V.line(g, cx(d.o), ly, bx, ly, { color: V.alpha(C.red, 0.5), lw: 3 });
        V.dot(g, cx(d.o), ly, 4, C.red);
        drawBug(g, bx, ly, t, C);
        // grade de detectores
        const cols = narrow ? 2 : 4, rows = Math.ceil(dets.length / cols);
        const gy = ly + 26, cg = 8;
        const cw = (w - 2 * pad - cg * (cols - 1)) / cols, ch = 60;
        dets.forEach((dt, i) => {
          const c = i % cols, r = Math.floor(i / cols);
          const x = pad + c * (cw + cg), y = gy + r * (ch + cg);
          const lvl = d.hit[dt.k] || 0;
          const reached = dt.s >= d.o && pos >= dt.s - 0.02;
          const on = reached && lvl > 0;
          const col = lvl === 2 ? C.red : lvl === 1 ? C.d : lvl === 3 ? C.e : C.muted;
          V.box(g, x, y, cw, ch, { r: 10, fill: on ? V.alpha(col, 0.12) : V.alpha(C.muted, 0.05), stroke: on ? V.alpha(col, 0.8) : V.alpha(C.muted, 0.25), lw: on ? 1.6 : 1 });
          const lx = x + 17, lyy = y + ch / 2;
          if (on) V.glow(g, lx, lyy, 15 + 3 * Math.sin(t * 5 + i), col, 0.8);
          V.dot(g, lx, lyy, 6.5, on ? col : V.alpha(C.muted, 0.35));
          const tx = x + 31, tmax = cw - 36;
          const fs = narrow ? 11 : 12;
          V.text(g, clip(g, dt.l, tmax, fs, 'sans', 600), tx, y + 21, { size: fs, weight: 600, color: C.ink });
          V.text(g, clip(g, stages[dt.s] + ' · ' + dt.cad, tmax, 10, 'sans'), tx, y + 37, { size: 10, color: C.muted });
          const st = dt.s < d.o ? 'fora do caminho' : !reached ? '…' : lvl === 2 ? 'ALARME' : lvl === 1 ? 'sinal fraco' : lvl === 3 ? 'dentro do ruído' : 'não vê';
          V.text(g, clip(g, st, tmax, 10, 'mono', 700), tx, y + 52, { size: 10, font: 'mono', weight: 700, color: on ? col : C.muted });
        });
        const ny = gy + rows * (ch + cg) + 16;
        V.text(g, 'DIAGNÓSTICO', pad, ny, { size: 10, font: 'mono', color: C.muted, weight: 700 });
        drawWrap(g, d.note, pad, ny + 19, w - 2 * pad, narrow ? 16.5 : 18.5, { size: narrow ? 12 : 13, color: C.ink });
      });
    })();

    /* =========================================================
       F2 — A máquina de surpresa
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-surpresa'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 545 : 400));
      const toks = ['O', 'gato', 'viu', 'a', 'lua', 'no', 'céu', 'de', 'maio'];
      const D = [
        { c: ['gato', 'sol', 'dia', 'rato', 'céu', 'vento'], l: [1.6, 1.2, 1.0, 0.6, 0.4, 0.3] },
        { c: ['subiu', 'viu', 'comeu', 'dormiu', 'pulou', 'olhou'], l: [1.5, 1.4, 1.0, 0.9, 0.6, 0.5] },
        { c: ['a', 'o', 'um', 'uma', 'que', 'tudo'], l: [2.4, 1.6, 0.8, 0.6, 0.3, 0.1] },
        { c: ['lua', 'casa', 'bola', 'janela', 'sombra', 'dona'], l: [2.0, 1.0, 0.9, 0.7, 0.5, 0.4] },
        { c: ['no', 'cheia', 'e', 'ao', 'brilhar', 'nascer'], l: [2.2, 1.8, 0.9, 0.5, 0.4, 0.2] },
        { c: ['céu', 'alto', 'telhado', 'mar', 'fundo', 'quintal'], l: [3.0, 0.9, 0.8, 0.5, 0.3, 0.2] },
        { c: ['azul', 'de', 'e', '.', 'noturno', 'claro'], l: [1.6, 1.2, 1.1, 0.9, 0.6, 0.5] },
        { c: ['maio', 'verão', 'agosto', 'inverno', 'abril', 'outono'], l: [1.3, 1.2, 1.0, 0.9, 0.8, 0.7] },
      ];
      const tgt = D.map((d, i) => d.c.indexOf(toks[i + 1]));
      const rg = V.rng(31);
      const Z = D.map((d) => d.c.map(() => rg.normal()));
      const S = { sig: 0.6, mode: 'rand' };
      const R = {};
      const ctl = UI.controls(fig);
      let rd = null;
      const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      function compute() {
        const ref = D.map((d) => softmax(d.l));
        const lg = D.map((d, i) => d.l.map((x, j) => x + S.sig * Z[i][j]));
        if (S.mode === 'cancel') {
          for (let i = 0; i + 1 < D.length; i += 2) {
            const a = D[i].l.map((x, j) => (j === tgt[i] ? x + S.sig * 0.9 : x + S.sig * Z[i][j]));
            lg[i] = a;
            const dA = -Math.log(softmax(a)[tgt[i]]) + Math.log(ref[i][tgt[i]]);
            const k = i + 1;
            const make = (dl) => D[k].l.map((x, j) => (j === tgt[k] ? x + dl : x + S.sig * Z[k][j]));
            const dNLL = (dl) => -Math.log(softmax(make(dl))[tgt[k]]) + Math.log(ref[k][tgt[k]]);
            let lo = -12, hi = 12;
            for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (dNLL(mid) > -dA) lo = mid; else hi = mid; }
            lg[k] = make((lo + hi) / 2);
          }
        }
        const cand = lg.map((l) => softmax(l));
        R.ref = ref; R.cand = cand;
        R.sr = ref.map((p, i) => -Math.log(p[tgt[i]]));
        R.sc = cand.map((p, i) => -Math.log(p[tgt[i]]));
        R.kl = ref.map((p, i) => klDiv(p, cand[i]));
        R.nllR = mean(R.sr); R.nllC = mean(R.sc); R.klM = mean(R.kl);
        R.top1 = ref.filter((p, i) => argmax(p) === argmax(cand[i])).length;
        if (rd) {
          rd.nll.set(f(R.nllR, 3) + ' → ' + f(R.nllC, 3));
          rd.ppl.set(f(Math.exp(R.nllR), 2) + ' → ' + f(Math.exp(R.nllC), 2));
          rd.kl.set(f(R.klM, 3) + ' nats');
        }
      }
      UI.slider(ctl, { label: 'Ruído nos logits', min: 0, max: 1.5, step: 0.05, value: S.sig, fmt: (v) => f(v, 2), onInput: (v) => { S.sig = v; compute(); } });
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'rand', label: 'ruído aleatório' }, { v: 'cancel', label: 'erros que se cancelam' }], value: 'rand', onChange: (v) => { S.mode = v; compute(); } });
      rd = { nll: UI.readout(ctl, 'NLL ref → cand', '—'), ppl: UI.readout(ctl, 'PPL', '—'), kl: UI.readout(ctl, 'KL médio', '—') };
      compute();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        const rightW = narrow ? 0 : Math.min(280, w * 0.37);
        const lw = w - 2 * pad - (narrow ? 0 : rightW + 18);
        const nT = toks.length, slot = lw / nT;
        const cur = Math.floor(t / 1.7) % D.length;
        const fsz = slot < 42 ? 10.5 : 12.5;
        V.text(g, clip(g, 'texto de avaliação · o modelo lê tudo de uma vez', lw, 10.5, 'mono'), pad, 16, { size: 10.5, color: C.muted, font: 'mono' });
        const ty = 26, th = 28;
        for (let i = 0; i < nT; i++) {
          const x = pad + i * slot + 2, bw = slot - 4;
          const isT = i === cur + 1, isCtx = i <= cur;
          V.box(g, x, ty, bw, th, { r: 7, fill: isT ? V.alpha(C.d, 0.25) : isCtx ? V.alpha(C.a, 0.14) : V.alpha(C.muted, 0.08), stroke: isT ? C.d : isCtx ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.3), lw: isT ? 2 : 1 });
          V.text(g, clip(g, toks[i], bw - 4, fsz, 'sans', 600), x + bw / 2, ty + th / 2 + 1, { size: fsz, weight: 600, align: 'center', base: 'middle', color: C.ink });
        }
        const barTop = ty + th + 30, barH = narrow ? 96 : 150;
        V.text(g, 'p(token verdadeiro)', pad, barTop - 10, { size: 10, color: C.muted, font: 'mono' });
        V.line(g, pad + slot * 0.6, barTop + barH, pad + lw, barTop + barH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let k = 0; k < D.length; k++) {
          const i = k + 1;
          const x0 = pad + i * slot + slot / 2;
          const bwid = Math.max(5, Math.min(13, slot * 0.28));
          const pr = R.ref[k][tgt[k]], pc = R.cand[k][tgt[k]];
          const hr = Math.max(1, pr * barH), hc = Math.max(1, pc * barH);
          V.rr(g, x0 - bwid - 1, barTop + barH - hr, bwid, hr, 3); g.fillStyle = V.alpha(C.a, 0.14); g.fill(); g.strokeStyle = C.a; g.lineWidth = 1.6; g.stroke();
          V.rr(g, x0 + 1, barTop + barH - hc, bwid, hc, 3); g.fillStyle = V.alpha(C.b, 0.85); g.fill();
          if (k === cur) V.glow(g, x0, barTop + barH - Math.max(hr, hc) - 8, 14, C.d, 0.7);
          const sy = barTop + barH + 15;
          V.text(g, f(R.sr[k], 2), x0, sy, { size: 9.5, font: 'mono', align: 'center', color: C.a });
          V.text(g, f(R.sc[k], 2), x0, sy + 13, { size: 9.5, font: 'mono', align: 'center', color: C.b });
        }
        V.text(g, narrow ? '−ln p' : '−ln p ref', pad, barTop + barH + 15, { size: 9.5, font: 'mono', color: C.a });
        V.text(g, 'cand', pad, barTop + barH + 28, { size: 9.5, font: 'mono', color: C.b });
        // painel da distribuição
        let px, py, pw, ph;
        if (narrow) { px = pad; py = barTop + barH + 44; pw = w - 2 * pad; ph = h - py - 76; }
        else { px = w - pad - rightW; py = 8; pw = rightW; ph = h - 8 - 84; }
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.b, 0.05), stroke: V.alpha(C.muted, 0.3) });
        V.text(g, clip(g, 'posição ' + (cur + 1) + ' · depois de “' + toks.slice(0, cur + 1).join(' ') + '”', pw - 20, 11, 'sans', 600), px + 10, py + 19, { size: 11, weight: 600, color: C.ink });
        const dd = D[cur], pr = R.ref[cur], pc = R.cand[cur];
        const nC = dd.c.length;
        const rowH = Math.min(28, (ph - 52) / nC);
        const labW = Math.min(64, pw * 0.26);
        const barW = pw - labW - 86;
        for (let j = 0; j < nC; j++) {
          const y = py + 30 + j * rowH;
          const isT = j === tgt[cur];
          V.text(g, clip(g, dd.c[j], labW - 4, 11, 'sans', isT ? 700 : 400), px + 10, y + rowH / 2 + 4, { size: 11, weight: isT ? 700 : 400, color: isT ? C.d : C.ink });
          const x0 = px + 10 + labW, hh = Math.max(3, (rowH - 7) / 2);
          V.rr(g, x0, y + 2, Math.max(1.5, pr[j] * barW), hh, 2); g.strokeStyle = C.a; g.lineWidth = 1.2; g.stroke();
          V.rr(g, x0, y + 3 + hh, Math.max(1.5, pc[j] * barW), hh, 2); g.fillStyle = V.alpha(C.b, 0.85); g.fill();
          V.text(g, pct(pr[j], 0) + '→' + pct(pc[j], 0), px + pw - 8, y + rowH / 2 + 4, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
        }
        V.text(g, 'KL nesta posição: ' + f(R.kl[cur], 3) + ' nats', px + 10, py + ph - 11, { size: 11, font: 'mono', color: C.ink });
        // resumo
        const dN = R.nllC - R.nllR;
        const sg = (x) => (x >= 0 ? '+' : '−') + f(Math.abs(x), 3);
        const lines = narrow
          ? ['NLL ' + f(R.nllR, 3) + ' → ' + f(R.nllC, 3) + ' (Δ ' + sg(dN) + ')', 'PPL ' + f(Math.exp(R.nllR), 2) + ' → ' + f(Math.exp(R.nllC), 2) + ' · KL médio ' + f(R.klM, 3), 'top-1 igual em ' + R.top1 + '/' + D.length + ' posições']
          : ['NLL ' + f(R.nllR, 3) + ' → ' + f(R.nllC, 3) + ' nats (Δ ' + sg(dN) + ')   ·   PPL ' + f(Math.exp(R.nllR), 2) + ' → ' + f(Math.exp(R.nllC), 2), 'KL médio ' + f(R.klM, 3) + ' nats   ·   top-1 igual em ' + R.top1 + '/' + D.length + ' posições'];
        const hidden = Math.abs(dN) < 0.02 && R.klM > 0.02;
        if (hidden) lines.push('→ a perplexidade quase não mudou, mas as distribuições mudaram: a média escondeu o estrago');
        const y0 = h - 12 - (lines.length - 1) * 17;
        lines.forEach((s, i) => { const warn = hidden && i === lines.length - 1; const wt = warn ? 700 : 500; V.text(g, clip(g, s, w - 2 * pad, narrow ? 11 : 12, 'mono', wt), pad, y0 + i * 17, { size: narrow ? 11 : 12, font: 'mono', weight: wt, color: warn ? C.c : C.ink }); });
      });
    })();

    /* =========================================================
       F3 — Duas distribuições, uma régua (KL)
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-kl'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 480 : 410));
      const toks = ['Paris', 'Lyon', 'uma', 'a', 'o', 'bela', 'Nice', 'sede'];
      const base = [3.2, 1.3, 1.1, 0.9, 0.8, 0.4, 0.2, 0.0];
      const P = softmax(base);
      const mkZ = (s) => { const r = V.rng(s); return base.map(() => r.normal()); };
      const S = { sig: 0.7, k: 8, rev: false, seed: 5, zFrom: mkZ(5), zTo: mkZ(5), tm: -99, pend: false, zNow: mkZ(5) };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Ruído nos logits', min: 0, max: 1.5, step: 0.05, value: S.sig, fmt: (v) => f(v, 2), onInput: (v) => { S.sig = v; } });
      UI.slider(ctl, { label: 'Suporte top-k', min: 2, max: 8, step: 1, value: 8, fmt: (v) => (v === 8 ? '8 (tudo)' : String(v)), onInput: (v) => { S.k = v; } });
      UI.toggle(ctl, { label: 'inverter a direção: D(Q‖P)', value: false, onChange: (v) => { S.rev = v; } });
      UI.button(ctl, '↻ novo ruído', () => { S.seed += 1; S.pend = true; });
      const topk = (p, k) => p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((x) => x[1]);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.pend) { S.zFrom = S.zNow.slice(); S.zTo = mkZ(S.seed); S.tm = t; S.pend = false; }
        const u = V.ease.inOut(V.clamp((t - S.tm) / 0.8, 0, 1));
        S.zNow = S.zTo.map((z, i) => V.lerp(S.zFrom[i], z, u));
        const Q = softmax(base.map((x, i) => x + S.sig * S.zNow[i]));
        const A = S.rev ? Q : P, B = S.rev ? P : Q;
        const kl = klDiv(A, B);
        const setP = new Set(topk(P, S.k)), setQ = new Set(topk(Q, S.k));
        const shared = [...setP].filter((i) => setQ.has(i));
        let klk = NaN;
        if (shared.length) { const za = shared.reduce((s, i) => s + A[i], 0), zb = shared.reduce((s, i) => s + B[i], 0); klk = Math.max(0, shared.reduce((s, i) => s + (A[i] / za) * Math.log((A[i] / za) / (B[i] / zb)), 0)); }
        const Hn = -A.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
        const narrow = w < 640, pad = 14;
        const rightW = narrow ? 0 : 228;
        const cw = w - 2 * pad - (narrow ? 0 : rightW + 16);
        const n = toks.length, slot = cw / n;
        const top = 26, barH = narrow ? 150 : 170;
        const pmax = Math.max(0.7, ...P, ...Q) * 1.05;
        V.text(g, '“A capital da França é …”', pad, 14, { size: 11, color: C.muted, font: 'mono' });
        V.line(g, pad, top + barH, pad + cw, top + barH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const inSup = (i) => shared.includes(i);
        for (let i = 0; i < n; i++) {
          const x0 = pad + i * slot + slot / 2, bw = Math.max(5, Math.min(16, slot * 0.3));
          const al = S.k === 8 || inSup(i) ? 1 : 0.28;
          const hp = (P[i] / pmax) * barH, hq = (Q[i] / pmax) * barH;
          g.globalAlpha = al;
          V.rr(g, x0 - bw - 1, top + barH - hp, bw, Math.max(1, hp), 3); g.fillStyle = V.alpha(C.a, 0.15); g.fill(); g.strokeStyle = C.a; g.lineWidth = 1.6; g.stroke();
          V.rr(g, x0 + 1, top + barH - hq, bw, Math.max(1, hq), 3); g.fillStyle = V.alpha(C.b, 0.85); g.fill();
          if (slot >= 34) {
            V.text(g, pct(P[i], 0), x0 - bw / 2 - 1, top + barH - hp - 5, { size: 9, font: 'mono', align: 'center', color: C.a });
            V.text(g, pct(Q[i], 0), x0 + bw / 2 + 1, top + barH - hq - 5, { size: 9, font: 'mono', align: 'center', color: C.b });
          }
          g.globalAlpha = 1;
          V.text(g, clip(g, toks[i], slot - 2, narrow ? 10 : 11.5, 'sans', 600), x0, top + barH + 14, { size: narrow ? 10 : 11.5, weight: 600, align: 'center', color: al < 1 ? C.muted : C.ink });
        }
        // contribuições
        const cH = narrow ? 40 : 58, yz = top + barH + 44 + cH;
        const contrib = A.map((a, i) => (a > 0 ? a * Math.log(a / B[i]) : 0));
        const cmax = Math.max(0.03, ...contrib.map(Math.abs));
        V.text(g, clip(g, 'contribuição ' + (S.rev ? 'Q·ln(Q/P)' : 'P·ln(P/Q)') + (narrow ? '' : ' de cada token') + ' · soma = ' + f(kl, 4) + ' nats', cw, 10, 'mono'), pad, top + barH + 34, { size: 10, font: 'mono', color: C.muted });
        V.line(g, pad, yz, pad + cw, yz, { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let i = 0; i < n; i++) {
          const x0 = pad + i * slot + slot / 2, bw = Math.max(6, Math.min(22, slot * 0.45));
          const c = contrib[i], hh = (Math.abs(c) / cmax) * cH;
          const al2 = S.k === 8 || inSup(i) ? 1 : 0.3;
          g.fillStyle = c >= 0 ? V.alpha(C.c, 0.85 * al2) : V.alpha(C.b, 0.7 * al2);
          V.rr(g, x0 - bw / 2, c >= 0 ? yz - hh : yz, bw, Math.max(1, hh), 2); g.fill();
        }
        V.text(g, clip(g, S.rev ? 'rosa: Q > P (o candidato superestimou o token) · lavanda: Q < P' : 'rosa: P > Q (o candidato subestimou o token) · lavanda: P < Q', cw, 10, 'sans'), pad, yz + cH + 18, { size: 10, color: C.muted });
        // painel
        let px, py, pw;
        if (narrow) { px = pad; py = yz + cH + 32; pw = w - 2 * pad; } else { px = w - pad - rightW; py = 10; pw = rightW; }
        const ph = narrow ? Math.min(h - py - 8, 132) : h - 20;
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.b, 0.05), stroke: V.alpha(C.muted, 0.3) });
        const same = argmax(P) === argmax(Q);
        const items = [
          [(S.rev ? 'D(Q‖P)' : 'D(P‖Q)') + ' exata', f(kl, 4) + ' nats', C.ink],
          ['no suporte top-' + S.k + (S.k === 8 ? ' (= exata)' : ' · ' + shared.length + ' token' + (shared.length === 1 ? '' : 's') + ' em comum'), (Number.isFinite(klk) ? f(klk, 4) : '—') + ' nats', C.b],
          ['top-1', same ? 'igual (“' + toks[argmax(P)] + '”)' : 'mudou: “' + toks[argmax(P)] + '” → “' + toks[argmax(Q)] + '”', same ? C.e : C.red],
          [(S.rev ? 'H(Q)' : 'H(P)') + ' · entropia', f(Hn, 3) + ' nats', C.muted],
          ['entropia cruzada = H + KL', f(Hn + kl, 3) + ' nats', C.muted],
        ];
        if (narrow) {
          const colW = (pw - 20) / 2;
          items.forEach((it, i) => { const cx = px + 10 + (i % 2) * colW, cy = py + 18 + Math.floor(i / 2) * 36; V.text(g, clip(g, it[0], colW - 6, 10, 'sans'), cx, cy, { size: 10, color: C.muted }); V.text(g, clip(g, it[1], colW - 6, 12, 'mono', 700), cx, cy + 16, { size: 12, font: 'mono', weight: 700, color: it[2] }); });
        } else {
          items.forEach((it, i) => { const cy = py + 24 + i * 52; V.text(g, it[0], px + 12, cy, { size: 10.5, color: C.muted }); V.text(g, clip(g, it[1], pw - 24, i < 2 ? 17 : 13, 'mono', 700), px + 12, cy + 21, { size: i < 2 ? 17 : 13, font: 'mono', weight: 700, color: it[2] }); });
          if (S.k < 8 && Number.isFinite(klk) && kl > 0) V.text(g, 'o proxy vê ' + pct(klk / kl, 0) + ' da KL', px + 12, py + ph - 14, { size: 10.5, font: 'mono', color: C.b });
        }
      });
    })();

    /* =========================================================
       F4 — O painel calibrado, passo a passo
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-painel'); if (!fig) return;
      fitStage(fig, (w) => (w < 620 ? 475 : 470));
      const NP = 14, NT = 5, NS = 4, NC = 3;
      const rg = V.rng(4242);
      const margins = [0.9, 0.12, 1.4, 0.34, 2.1, 0.05, 0.62, 1.1, 0.3, 0.18, 1.8, 0.42, 0.8, 0.27];
      const base = margins.map((m) => { const l = [2.2, 2.2 - m]; for (let j = 2; j < NT; j++) l.push(2.2 - m - 0.5 - rg() * 1.8); return l; });
      const nz = () => base.map(() => Array.from({ length: NT }, () => rg.normal()));
      const nS = Array.from({ length: NS }, nz), nC = Array.from({ length: NC }, nz);
      const dirR = base.map(() => Array.from({ length: NT }, (_, j) => (j < 2 ? 0 : rg.normal() * 0.9)));
      const swap = margins.map((m) => (m <= 1.0 ? 1 : 0));
      const UNST = 6;
      // modo "variação entre boots": S1,S2 = boot 1; S3,S4 = boot 2 (ruído parcialmente compartilhado no boot)
      const nB = [nz(), nz()], nBc = nz();
      const WC = 0.85, WB = Math.sqrt(1 - WC * WC);
      const unstK = (s, i, j) => (s === 2 && i === UNST ? (j === 0 ? -0.5 : j === 1 ? 0.5 : 0) : 0);
      const swK = (i, j, a) => (swap[i] ? (j === 0 ? -0.6 : j === 1 ? 0.6 : 0) * a * a : 0);
      const capS = (sig) => nS.map((nn, s) => base.map((b, i) => b.map((x, j) => x + sig * nn[i][j] + unstK(s, i, j))));
      const capC = (sig, a) => nC.map((nn) => base.map((b, i) => b.map((x, j) => x + sig * nn[i][j] + a * dirR[i][j] + swK(i, j, a))));
      const capSB = (sig) => nS.map((nn, s) => base.map((b, i) => b.map((x, j) => x + sig * (WC * nn[i][j] + WB * nB[s >> 1][i][j]) + unstK(s, i, j))));
      const capCB = (sig, a) => nC.map((nn) => base.map((b, i) => b.map((x, j) => x + sig * (WC * nn[i][j] + WB * nBc[i][j]) + a * dirR[i][j] + swK(i, j, a))));
      const info = (L) => L.map((cap) => cap.map((l) => { const p = softmax(l); const am = argmax(l); let s2 = -Infinity; l.forEach((x, j) => { if (j !== am && x > s2) s2 = x; }); return { p, am, mg: l[am] - s2 }; }));
      const pairKL = (A) => { const res = []; for (let i = 0; i < NP; i++) { let mx = 0, sm = 0, n = 0; for (let a = 0; a < A.length; a++) for (let b = 0; b < A.length; b++) if (a !== b) { const k = klDiv(A[a][i].p, A[b][i].p); mx = Math.max(mx, k); sm += k; n++; } res.push({ mx, mean: sm / n }); } return res; };
      const classesOf = (SI, mstar) => SI[0].map((_, i) => { const mm = Math.min(...SI.map((c) => c[i].mg)); const dis = new Set(SI.map((c) => c[i].am)).size > 1; return mm < mstar ? 'TIE' : dis ? 'UNSTABLE' : 'CONF'; });
      const controlOf = (SI, cls, boots) => {
        const rs = [], ks = [];
        const groups = boots ? [[0, 1], [2, 3]] : [[0], [1], [2], [3]];
        for (let i = 0; i < NP; i++) {
          if (cls[i] !== 'CONF') continue;
          let racc = 0, kacc = 0;
          groups.forEach((hold) => {
            const Rr = [0, 1, 2, 3].filter((x) => !hold.includes(x));
            let mx = 0; Rr.forEach((a) => Rr.forEach((b) => { if (a !== b) mx = Math.max(mx, klDiv(SI[a][i].p, SI[b][i].p)); }));
            let km = 0, n = 0; hold.forEach((hh) => Rr.forEach((a) => { km += klDiv(SI[a][i].p, SI[hh][i].p); n++; })); km /= n;
            racc += km / Math.max(mx, 1e-12); kacc += km;
          });
          rs.push(racc / groups.length); ks.push(kacc / groups.length);
        }
        const m = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
        return { rs, ks, rbar: m(rs), absKL: m(ks) };
      };
      // calibra o ruído do stock para um nulo (controle) de ~0,0144 nats
      let sig = 0.15;
      { let lo = 0.01, hi = 0.8; for (let it = 0; it < 36; it++) { const mid = (lo + hi) / 2; const SIm = info(capS(mid)); const c = controlOf(SIm, classesOf(SIm, 0.25)); if (c.absKL < 0.0144) lo = mid; else hi = mid; } sig = (lo + hi) / 2; }
      const SI = info(capS(sig)); const PK = pairKL(SI);
      const SIb = info(capSB(sig)); const PKb = pairKL(SIb);
      const evalCand = (CI, cls, SIx, PKx) => {
        const SIu = SIx || SI, PKu = PKx || PK;
        const rows = [];
        for (let i = 0; i < NP; i++) {
          let sm = 0, n = 0; for (let a = 0; a < NS; a++) for (let c = 0; c < CI.length; c++) { sm += klDiv(SIu[a][i].p, CI[c][i].p); n++; }
          const kl = sm / n, r = kl / Math.max(PKu[i].mx, 1e-12);
          const votes = {}; CI.forEach((c) => { votes[c[i].am] = (votes[c[i].am] || 0) + 1; });
          let best = null; Object.keys(votes).forEach((k) => { if (best === null || votes[k] > votes[best] || (votes[k] === votes[best] && +k < +best)) best = k; });
          rows.push({ kl, r, cons: +best, flip: cls[i] === 'CONF' && +best !== SIu[0][i].am });
        }
        const conf = rows.filter((_, i) => cls[i] === 'CONF');
        const m = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
        return { rows, rbar: m(conf.map((x) => x.r)), flips: conf.filter((x) => x.flip).length, absKL: m(conf.map((x) => x.kl)) };
      };
      const cls0 = classesOf(SI, 0.25);
      const cands = [
        { v: 'bf16', label: 'Cópia KDA BF16 (real: 1,10× o nulo, 0 flips)', short: 'KDA BF16', target: 1.10 },
        { v: 'fp8', label: 'FP8=all (real: KL 25–30% acima dos pares stock)', short: 'FP8=all', target: 1.28 },
        { v: 'w8a8', label: 'W8A8 · controle positivo (real: 7,99×, 23 flips)', short: 'W8A8', target: 7.99 },
      ];
      cands.forEach((c) => { let lo = 0, hi = 30; for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; const e = evalCand(info(capC(sig, mid)), cls0); if (e.rbar < c.target) lo = mid; else hi = mid; } c.alpha = (lo + hi) / 2; c.CI = info(capC(sig, c.alpha)); c.CIb = info(capCB(sig, c.alpha)); });
      const S = { cand: cands[0], mstar: 0.25, boot: false };
      let M = null;
      const ctl = UI.controls(fig);
      let rdC = null, rdR = null;
      const recompute = () => {
        const SIx = S.boot ? SIb : SI, PKx = S.boot ? PKb : PK, CIx = S.boot ? S.cand.CIb : S.cand.CI;
        const cls = classesOf(SIx, S.mstar);
        const E = evalCand(CIx, cls, SIx, PKx);
        const CT = controlOf(SIx, cls, S.boot);
        M = { SIx, CIx, cls, E, CT, nConf: cls.filter((c) => c === 'CONF').length, nTie: cls.filter((c) => c === 'TIE').length, nUn: cls.filter((c) => c === 'UNSTABLE').length };
        if (rdC) rdC.set(M.nConf + ' / ' + M.nTie + ' / ' + M.nUn);
        if (rdR) rdR.set(f(E.rbar, 2) + '× · ' + E.flips + ' flip' + (E.flips === 1 ? '' : 's'));
      };
      UI.select(ctl, { label: 'Candidato', options: cands.map((c) => ({ v: c.v, label: c.label })), value: 'bf16', onChange: (v) => { S.cand = cands.find((c) => c.v === v) || cands[0]; recompute(); } });
      UI.slider(ctl, { label: 'Margem de corte', min: 0.05, max: 1.5, step: 0.05, value: 0.25, fmt: (v) => f(v, 2) + ' nats', onInput: (v) => { S.mstar = v; recompute(); } });
      UI.toggle(ctl, { label: 'variação entre boots (2 boots × 2 capturas, ilustrativo)', value: false, onChange: (v) => { S.boot = v; recompute(); if (HP) HP.redraw(); } });
      rdC = UI.readout(ctl, 'CONF / TIE / UNSTABLE', '—');
      rdR = UI.readout(ctl, 'razão média · flips em CONF', '—');
      recompute();
      const steps = [
        { t: 'Capturar o stock (4 boots)', d: 'Cada coluna S1–S4 é um boot limpo do stock pedindo <code>prompt_logprobs=20</code> sobre o mesmo texto. A cor é o token de maior probabilidade em cada posição; o brilho, a margem até o segundo colocado. Repare: onde a margem é pequena, o próprio stock troca de cor.' },
        { t: 'Medir a margem', d: 'A barra à esquerda é a menor margem \\(\\ln p_{(1)}-\\ln p_{(2)}\\) entre as capturas stock. Abaixo de \\(m^*\\) (linha tracejada), a posição vira TIE: fica anotada, mas não é julgada por flips. No script de triagem do GLM, \\(m^*\\) é fixo em 0,25 nats; o estudo selado do thin-decode derivou 1,5 nats da própria calibração. Arraste o controle e veja as classes mudarem.' },
        { t: 'Separar os instáveis', d: 'Se o stock trocou de argmax mesmo com todas as margens acima de \\(m^*\\), a posição é UNSTABLE e sai de todos os gates. Assim, uma única posição bimodal não alarga a régua das outras.' },
        { t: 'Comparar o candidato', d: 'As colunas C1–C3 são capturas do candidato. Só nas posições CONF um flip de consenso conta (✕). O painel real acusa FLAG a partir do valor crítico de Poisson (11 flips, com \\(\\lambda=6\\)) e tolera até 10: uma folga heurística.' },
        { t: 'Padronizar a KL', d: 'Em cada posição CONF, a KL média candidato↔stock é dividida pela maior KL entre pares de stock daquela posição. A barra à direita é essa razão \\(r_t\\); a linha marca o gate de 2×.' },
        { t: 'Veredicto contra o nulo', d: 'Em cima, a razão ao nulo do candidato e do controle (cada captura stock julgada contra as outras): o script responde WITHIN (exit 0) ou FLAG (exit 1). Embaixo, a KL absoluta contra o gate “KL ≤ 0,01”. Quando o controle reprova um gate, o instrumento não está apto (nos estudos formais, INCONCLUSIVE): o defeito é da régua. Ligue “variação entre boots” para ver o controle reprovar o próprio gate calibrado, como em 09-26.' },
      ];
      let HP = null;
      const st = UI.stepper(fig, steps, { autoplay: 7500, onStep: () => { if (HP) HP.redraw(); } });
      const letters = 'ABCDE';
      HP = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const tokCol = [C.a, C.b, C.c, C.d, C.e];
        const step = st.i, el = st.elapsed();
        const narrow = w < 620, pad = 12;
        const { cls, E, CT } = M;
        if (step < 5) {
          const idxW = narrow ? 18 : 26, mbW = narrow ? 34 : 56, clsW = narrow ? 42 : 60, flW = 20, minR = narrow ? 70 : 110;
          const fixed = 2 * pad + idxW + mbW + clsW + flW + minR + 32;
          const cs = V.clamp((w - fixed - 14) / 7, 14, 30);
          const top = 34, foot = 44;
          const rh = V.clamp((h - top - foot) / NP, 14, 26);
          const natR = w - pad - (pad + idxW + mbW + 8 + 4 * (cs + 2) + 6 + clsW + 6 + 3 * (cs + 2) + 4 + flW + 6);
          const ox = natR > 300 ? (natR - 300) / 2 : 0;
          const xIdx = pad + ox, xMb = xIdx + idxW, xS = xMb + mbW + 8, xCls = xS + 4 * (cs + 2) + 6, xC = xCls + clsW + 6, xFl = xC + 3 * (cs + 2) + 4, xR = xFl + flW + 6, wR = Math.min(300, w - pad - xR);
          const hdr = (s, x, ww, on = true) => V.text(g, clip(g, s, ww + 8, 10, 'mono', 600), x + ww / 2, top - 10, { size: 10, font: 'mono', weight: 600, align: 'center', color: on ? C.muted : V.alpha(C.muted, 0.35) });
          hdr('t', xIdx, idxW - 4);
          hdr(narrow ? 'marg' : 'margem', xMb, mbW, step >= 1);
          for (let s = 0; s < NS; s++) hdr(S.boot ? ['B1a', 'B1b', 'B2a', 'B2b'][s] : 'S' + (s + 1), xS + s * (cs + 2), cs);
          hdr(narrow ? 'cls' : 'classe', xCls, clsW, step >= 2);
          for (let c = 0; c < NC; c++) hdr('C' + (c + 1), xC + c * (cs + 2), cs, step >= 3);
          hdr('flip', xFl, flW, step >= 3);
          hdr(narrow ? 'r_t' : 'r_t (KL / nulo)', xR, wR, step >= 4);
          const mmax = 1.5;
          for (let i = 0; i < NP; i++) {
            const y = top + i * rh;
            const cl = cls[i];
            const dim = step >= 2 && cl !== 'CONF' && step >= 3;
            V.text(g, String(i + 1), xIdx + idxW - 6, y + rh / 2 + 3.5, { size: 10, font: 'mono', align: 'right', color: C.muted });
            // stock cells
            for (let s = 0; s < NS; s++) {
              const vis = step > 0 || el > s * 0.45 + i * 0.03;
              if (!vis) continue;
              const inf = M.SIx[s][i];
              const x = xS + s * (cs + 2);
              const a = 0.22 + 0.66 * V.clamp(inf.mg / 1.2, 0, 1);
              V.box(g, x, y + 1, cs, rh - 2, { r: 4, fill: V.alpha(tokCol[inf.am], dim ? a * 0.45 : a) });
              if (cs >= 16 && rh >= 15) V.text(g, letters[inf.am], x + cs / 2, y + rh / 2 + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, alpha: dim ? 0.5 : 0.9 });
            }
            if (step >= 1) {
              const mm = Math.min(...M.SIx.map((c) => c[i].mg));
              const bl = V.clamp(mm / mmax, 0.02, 1) * (mbW - 4);
              V.box(g, xMb, y + rh / 2 - 3.5, bl, 7, { r: 3, fill: mm < S.mstar ? V.alpha(C.d, 0.85) : V.alpha(C.a, 0.8) });
            }
            if (step >= 2) {
              const col = cl === 'CONF' ? C.e : cl === 'TIE' ? C.d : C.red;
              const lab = cl === 'UNSTABLE' ? (narrow ? 'UNST' : 'UNSTABLE') : cl;
              V.box(g, xCls, y + 2, clsW, rh - 4, { r: (rh - 4) / 2, fill: V.alpha(col, 0.15), stroke: V.alpha(col, 0.7), lw: 1 });
              V.text(g, clip(g, lab, clsW - 4, 9.5, 'mono', 700), xCls + clsW / 2, y + rh / 2 + 0.5, { size: 9.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: col });
            }
            if (step >= 3) {
              const CI = M.CIx;
              for (let c = 0; c < NC; c++) {
                const vis = step > 3 || el > 0.3 + c * 0.45 + i * 0.03;
                if (!vis) continue;
                const inf = CI[c][i]; const x = xC + c * (cs + 2);
                const a = 0.22 + 0.66 * V.clamp(inf.mg / 1.2, 0, 1);
                V.box(g, x, y + 1, cs, rh - 2, { r: 4, fill: V.alpha(tokCol[inf.am], dim ? a * 0.45 : a), stroke: inf.am !== M.SIx[0][i].am && cl === 'CONF' ? C.red : null, lw: 1.5 });
                if (cs >= 16 && rh >= 15) V.text(g, letters[inf.am], x + cs / 2, y + rh / 2 + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, alpha: dim ? 0.5 : 0.9 });
              }
              if (cl === 'CONF') {
                if (E.rows[i].flip) V.text(g, '✕', xFl + flW / 2, y + rh / 2 + 1, { size: 13, weight: 700, align: 'center', base: 'middle', color: C.red });
                else V.text(g, '·', xFl + flW / 2, y + rh / 2 + 1, { size: 13, align: 'center', base: 'middle', color: C.muted });
              }
            }
            if (step >= 4) {
              if (cl === 'CONF') {
                const r = E.rows[i].r, cap = 10;
                const bl = V.clamp(r / cap, 0.01, 1) * (wR - 34);
                V.box(g, xR, y + rh / 2 - 4, bl, 8, { r: 3, fill: r > 2 ? V.alpha(C.red, 0.8) : V.alpha(C.b, 0.8) });
                V.text(g, r >= cap ? '≥10' : f(r, 1), xR + bl + 4, y + rh / 2 + 3.5, { size: 9.5, font: 'mono', color: C.muted });
              } else V.text(g, '—', xR + 4, y + rh / 2 + 3.5, { size: 10, font: 'mono', color: V.alpha(C.muted, 0.6) });
            }
          }
          if (step >= 1) { const xm = xMb + V.clamp(S.mstar / mmax, 0, 1) * (mbW - 4); V.line(g, xm, top - 2, xm, top + NP * rh, { color: C.d, lw: 1.3, dash: [3, 3] }); }
          if (step >= 4) { const x2 = xR + (2 / 10) * (wR - 34); V.line(g, x2, top - 2, x2, top + NP * rh, { color: C.red, lw: 1.3, dash: [4, 3] }); V.text(g, '2×', x2 + 3, top + NP * rh + 12, { size: 10, font: 'mono', color: C.red }); }
          // rodapé
          const fy = top + NP * rh + 26;
          const parts = ['CONF ' + M.nConf, 'TIE ' + M.nTie, 'UNSTABLE ' + M.nUn];
          if (step >= 3) parts.push('flips em CONF: ' + E.flips);
          if (step >= 4) parts.push('r̄ = ' + f(E.rbar, 2) + '×');
          const fl = narrow ? [parts.slice(0, 3).join(' · '), parts.slice(3).join(' · ')].filter(Boolean) : [parts.join('  ·  ')];
          fl.forEach((ln, k) => V.text(g, clip(g, ln, w - 2 * pad, narrow ? 10.5 : 12, 'mono', 600), xIdx, fy + k * 15, { size: narrow ? 10.5 : 12, font: 'mono', weight: 600, color: C.ink }));
          V.text(g, clip(g, 'candidato: ' + S.cand.short + ' (dados simulados)', w - 2 * pad, 10, 'sans'), xIdx, fy + 16 + (fl.length - 1) * 15, { size: 10, color: C.muted });
        } else {
          // vista do veredicto
          const x0 = pad + (narrow ? 88 : 150), x1 = w - pad - 10, W = x1 - x0;
          const rows = [{ lab: narrow ? 'controle' : 'controle (stock)', col: C.muted }, { lab: (narrow ? '' : 'candidato: ') + S.cand.short, col: C.b }];
          const panel = (y, title, lo, hi, ticks, tickLab, gateV, gateLab, ctrl, cand, ctrlM, candM) => {
            const X = (v) => x0 + (Math.log10(V.clamp(v, lo, hi)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * W;
            V.text(g, title, pad, y, { size: 11.5, weight: 700, color: C.ink });
            const ya = y + 26, yb = y + ph * 0.5, yax = y + ph - 40;
            V.line(g, x0, yax, x1, yax, { color: V.alpha(C.muted, 0.6), lw: 1 });
            ticks.forEach((tv, k) => { const x = X(tv); V.line(g, x, yax - 3, x, yax + 3, { color: C.muted, lw: 1 }); V.text(g, tickLab[k], x, yax + 15, { size: 9.5, font: 'mono', align: 'center', color: C.muted }); });
            const gx = X(gateV);
            g.fillStyle = V.alpha(C.red, 0.07); g.fillRect(gx, y + 8, x1 - gx, yax - y - 8);
            V.line(g, gx, y + 6, gx, yax, { color: C.red, lw: 1.5, dash: [4, 3] });
            V.text(g, gateLab, gx + 4, yax - 6, { size: 9.5, font: 'mono', color: C.red });
            [[ya, ctrl, ctrlM, rows[0]], [yb, cand, candM, rows[1]]].forEach(([yy, arr, mm, rr]) => {
              V.text(g, clip(g, rr.lab, x0 - pad - 8, 10.5, 'sans', 600), pad, yy + 4, { size: 10.5, weight: 600, color: rr.col === C.muted ? C.ink : rr.col });
              arr.forEach((v, k) => V.dot(g, X(v), yy + ((k % 3) - 1) * 3, 3.2, V.alpha(rr.col === C.muted ? C.a : rr.col, 0.55)));
              const mx = X(mm);
              g.fillStyle = mm > gateV ? C.red : C.e; g.beginPath(); g.moveTo(mx, yy - 8); g.lineTo(mx + 7, yy); g.lineTo(mx, yy + 8); g.lineTo(mx - 7, yy); g.closePath(); g.fill();
            });
          };
          const confIdx = cls.map((c, i) => (c === 'CONF' ? i : -1)).filter((i) => i >= 0);
          const candR = confIdx.map((i) => E.rows[i].r), candK = confIdx.map((i) => E.rows[i].kl);
          const ph = Math.max(120, (h - 120) / 2);
          panel(22, 'Gate calibrado · razão ao nulo (escala log)', 0.1, 40, [0.1, 0.5, 1, 2, 5, 10, 40], ['0,1', '0,5', '1', '2', '5', '10', '40'], 2, 'gate 2×', CT.rs, candR, CT.rbar, E.rbar);
          panel(22 + ph, 'Gate absoluto · KL média (nats, escala log)', 0.001, 0.3, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], ['0,001', '0,003', '0,01', '0,03', '0,1', '0,3'], 0.01, 'gate 0,01', CT.ks, candK, CT.absKL, E.absKL);
          const vy = 22 + 2 * ph + 8;
          const calOK = E.rbar <= 2 && E.flips < 11, ctrlCal = CT.rbar <= 2, ctrlAbs = CT.absKL <= 0.01;
          const line1 = 'Triagem — controle: ' + (ctrlCal ? 'WITHIN' : 'FLAG (r̄ ' + f(CT.rbar, 1) + '×)') + ' · candidato: ' + (calOK ? 'WITHIN (exit 0)' : 'FLAG (exit 1)') + (ctrlCal ? '' : ' → acusa o próprio stock: não serve de gate');
          const col1 = !ctrlCal ? C.d : calOK ? C.e : C.red;
          const line2 = 'Absoluto — controle: ' + (ctrlAbs ? 'PASS' : 'FALHA (KL ' + f(CT.absKL, 4) + ' > 0,01)') + (ctrlAbs ? '' : ' → estudo INCONCLUSIVE');
          const fs = narrow ? 11 : 12.5;
          if (narrow) {
            const n1 = drawWrap(g, line1, pad, vy + 4, w - 2 * pad, 14, { size: fs, font: 'mono', weight: 700, color: col1 });
            const n2 = drawWrap(g, line2, pad, vy + 8 + n1 * 14, w - 2 * pad, 14, { size: fs, font: 'mono', weight: 700, color: ctrlAbs ? C.e : C.d });
            V.text(g, clip(g, S.boot ? 'nulo só de pares do mesmo boot · ilustrativo' : 'losango = média · pontos = posições CONF', w - 2 * pad, 10, 'sans'), pad, vy + 12 + (n1 + n2) * 14, { size: 10, color: C.muted });
          } else {
            V.text(g, clip(g, line1, w - 2 * pad, fs, 'mono', 700), pad, vy + 6, { size: fs, font: 'mono', weight: 700, color: col1 });
            V.text(g, clip(g, line2, w - 2 * pad, fs, 'mono', 700), pad, vy + 26, { size: fs, font: 'mono', weight: 700, color: ctrlAbs ? C.e : C.d });
          }
          if (!narrow) V.text(g, clip(g, S.boot ? 'ilustrativo · deixa-um-boot-fora: o nulo do boot restante só tem pares do mesmo boot, então a variação entre boots parece “estrago” no controle' : 'losango = média · pontos = posições CONF · o controle é cada captura stock julgada contra as outras três (deixa-uma-fora)', w - 2 * pad, 10.5, 'sans'), pad, vy + 46, { size: 10.5, color: C.muted });
        }
      }, { static: false });
    })();

    /* =========================================================
       F5 — Erro de pesos: rel RMSE e cosseno
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-pesos'); if (!fig) return;
      fitStage(fig, (w) => (w < 620 ? 445 : 440));
      const R = 64, K = 128;
      const S = { scheme: 'fp8c', out: 0.02 };
      const rs = V.rng(77);
      const rowScale = Array.from({ length: R }, () => Math.exp(rs.normal() * 0.6));
      let W = [], res = null, prev = null, H = null;
      const ctl = UI.controls(fig);
      let rd = null;
      function build() { const b = N.synthWeights(R * K, 11, S.out, 8); W = b.map((x, idx) => x * rowScale[Math.floor(idx / K)]); prev = null; }
      function quant(anim) {
        prev = anim && res ? { deq: res.deq, rows: res.rows, rel: res.rel } : null;
        S.pend = true;
        let deq;
        if (S.scheme === 'fp8c') {
          deq = new Array(R * K);
          for (let r = 0; r < R; r++) { let am = 0; for (let c = 0; c < K; c++) am = Math.max(am, Math.abs(W[r * K + c])); const s = am / 448 || 1e-30; for (let c = 0; c < K; c++) deq[r * K + c] = N.encode('e4m3', W[r * K + c] / s).v * s; }
        } else if (S.scheme === 'fp8t') deq = N.fp8Tensor(W, 'e4m3').deq;
        else if (S.scheme === 'nvfp4') deq = N.nvfp4(W, 16).deq;
        else if (S.scheme === 'mxfp4') deq = N.mx(W, 'e2m1', 32).deq;
        else deq = N.quantInt(W, { bits: 4, symmetric: true, group: 128 }).deq;
        let se = 0, sx = 0, dot = 0, sd = 0;
        for (let i = 0; i < W.length; i++) { const d = deq[i] - W[i]; se += d * d; sx += W[i] * W[i]; dot += deq[i] * W[i]; sd += deq[i] * deq[i]; }
        const rel = Math.sqrt(se / sx), cos = dot / Math.sqrt(sx * sd);
        const rows = [];
        for (let r = 0; r < R; r++) { let a = 0, b = 0; for (let c = 0; c < K; c++) { const x = W[r * K + c], d = deq[r * K + c] - x; a += d * d; b += x * x; } rows.push(Math.sqrt(a / b)); }
        res = { deq, rel, cos, rows };
        if (rd) { rd.rel.set(pct(rel, 2)); rd.cos.set(cos.toFixed(5).replace('.', ',')); rd.v.set(rel > 0.06 ? 'HIGH ERROR' : 'ok (≤ 0,06)'); }
        if (H) H.redraw();
      }
      UI.select(ctl, { label: 'Esquema', options: [
        { v: 'fp8c', label: 'FP8 E4M3 · por canal (FP8-dense do Qwen)' },
        { v: 'fp8t', label: 'FP8 E4M3 · por tensor' },
        { v: 'nvfp4', label: 'NVFP4 · bloco 16' },
        { v: 'mxfp4', label: 'MXFP4 · bloco 32' },
        { v: 'int4', label: 'INT4 simétrico · grupo 128' },
      ], value: 'fp8c', onChange: (v) => { S.scheme = v; quant(true); } });
      UI.slider(ctl, { label: 'Outliers', min: 0, max: 0.05, step: 0.005, value: S.out, fmt: (v) => pct(v, 1), onInput: (v) => { S.out = v; build(); quant(); } });
      rd = { rel: UI.readout(ctl, 'rel RMSE', '—'), cos: UI.readout(ctl, 'cosseno', '—'), v: UI.readout(ctl, 'verificador', '—') };
      build(); quant();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.pend) { S.t0 = t; S.pend = false; }
        const el = t - (S.t0 || 0);
        const narrow = w < 620, pad = 14;
        // transição: cada célula de Ŵ desliza do arredondamento antigo para o novo, em cascata
        const uCell = (k) => (prev ? V.ease.inOut(V.clamp((el - k * 0.004) / 0.45, 0, 1)) : 1);
        const hiR = Math.floor(t / 0.45) % 8;
        const ex = (arr) => Array.from({ length: 8 }, (_, i) => Array.from({ length: 16 }, (_, j) => arr[i * K + j]));
        const Wx = ex(W);
        const Dx = Wx.map((row, i) => row.map((_, j) => { const idx = i * K + j, u = uCell(i * 16 + j); return prev ? V.lerp(prev.deq[idx], res.deq[idx], u) : res.deq[idx]; }));
        const Ex = Wx.map((row, i) => row.map((x, j) => (Dx[i][j] - x) * 10));
        let mx = 0; Wx.forEach((r) => r.forEach((x) => { mx = Math.max(mx, Math.abs(x)); }));
        const maps = [['W (original)', Wx], ['Ŵ (dequantizado)', Dx], ['(Ŵ − W) × 10', Ex]];
        const moving = [];
        if (prev) for (let k = 0; k < 128; k++) { const u = uCell(k); if (u > 0 && u < 1) moving.push(k); }
        const hiCol = V.alpha(C.d, 0.9);
        let gx0, gy0, gW;
        if (narrow) {
          const mw = (w - 2 * pad - 16) / 3, cell = (mw - 15 * 1) / 16;
          maps.forEach((m, k) => { const x = pad + k * (mw + 8); V.text(g, clip(g, m[0], mw, 10, 'mono'), x, 16, { size: 10, font: 'mono', color: C.muted }); const geo = V.matrix(g, { x, y: 24, data: m[1], cell, gap: 1, min: -mx, max: mx, cmap: 'div', r: 1, hiRow: hiR, hiColor: hiCol }); if (k === 1) moving.forEach((q) => { const [cx, cy] = geo.center(Math.floor(q / 16), q % 16); V.glow(g, cx, cy, cell * 1.6, C.b, 0.9); }); });
          gx0 = pad; gy0 = 24 + 8 * (mw / 16) + 38; gW = w - 2 * pad;
        } else {
          const cell = 11, mw = 16 * cell + 15;
          maps.forEach((m, k) => { const y = 22 + k * (8 * (cell + 1) + 30); V.text(g, m[0], pad, y - 6, { size: 10.5, font: 'mono', color: C.muted }); const geo = V.matrix(g, { x: pad, y, data: m[1], cell, gap: 1, min: -mx, max: mx, cmap: 'div', r: 2, hiRow: hiR, hiColor: hiCol }); if (k === 1) moving.forEach((q) => { const [cx, cy] = geo.center(Math.floor(q / 16), q % 16); V.glow(g, cx, cy, cell * 1.6, C.b, 0.9); }); });
          gx0 = pad + mw + 36; gy0 = 26; gW = w - gx0 - pad;
        }
        const uG = prev ? V.ease.out(V.clamp(el / 0.9, 0, 1)) : 1;
        const relD = prev ? V.lerp(prev.rel, res.rel, uG) : res.rel;
        const rowsD = prev ? res.rows.map((v, r) => V.lerp(prev.rows[r], v, V.ease.inOut(V.clamp((el - r * 0.008) / 0.5, 0, 1)))) : res.rows;
        // medidor
        const amax = 0.14;
        const X = (v) => gx0 + V.clamp(v / amax, 0, 1) * gW;
        const yb = gy0 + 36;
        V.text(g, 'rel RMSE do tensor inteiro', gx0, gy0, { size: 11.5, weight: 700, color: C.ink });
        V.box(g, gx0, yb - 9, gW, 18, { r: 9, fill: V.alpha(C.muted, 0.1) });
        g.fillStyle = V.alpha(C.red, 0.16); V.rr(g, X(0.06), yb - 9, gx0 + gW - X(0.06), 18, 9); g.fill();
        g.fillStyle = V.alpha(C.e, 0.45); g.fillRect(X(0.024), yb - 9, X(0.0268) - X(0.024), 18);
        V.line(g, X(0.06), yb - 16, X(0.06), yb + 14, { color: C.red, lw: 1.6 });
        V.text(g, 'HIGH ERROR > 0,06', X(0.06) + 4, yb - 13, { size: 9.5, font: 'mono', color: C.red });
        V.text(g, narrow ? 'Qwen 591' : 'Qwen: 591 tensores', X(0.0254), yb + 24, { size: 9.5, font: 'mono', align: 'center', color: C.e });
        [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14].forEach((tv) => { if (narrow && (tv * 100) % 4 !== 0) return; V.text(g, f(100 * tv, 0) + '%', X(tv), yb + 38, { size: 9, font: 'mono', align: 'center', color: C.muted }); });
        const rx = X(relD);
        tri(g, rx, yb - 11, 6, C.b, 1);
        V.line(g, rx, yb - 11, rx, yb + 9, { color: C.b, lw: 2.5 });
        const s1 = 'rel = ' + pct(res.rel, 2) + (res.rel > amax ? ' (fora da escala)' : '') + '   cos = ' + res.cos.toFixed(5).replace('.', ',') + '   1 − rel²/2 = ' + (1 - res.rel * res.rel / 2).toFixed(5).replace('.', ',');
        drawWrap(g, s1, gx0, yb + 58, gW, 15, { size: 11, font: 'mono', color: C.ink });
        // barras por linha
        const by0 = yb + (narrow ? 104 : 96), bh = h - by0 - 30;
        V.text(g, 'rel RMSE de cada linha (canal de saída)', gx0, by0 - 6, { size: 10.5, font: 'mono', color: C.muted });
        const rmax = Math.max(0.08, 1.25 * Math.max(...rowsD));
        const bw = gW / R;
        rowsD.forEach((v, r) => { const hh = V.clamp(v / rmax, 0, 1) * bh; const hot = r === hiR; g.fillStyle = v > 0.06 ? V.alpha(C.red, 0.75) : hot ? C.d : V.alpha(C.a, 0.75); g.fillRect(gx0 + r * bw + 0.5, by0 + bh - hh, Math.max(1, bw - 1), Math.max(1, hh)); if (hot) { g.strokeStyle = C.d; g.lineWidth = 2; g.strokeRect(gx0 + r * bw - 0.5, by0 + bh - hh - 1, Math.max(2, bw + 1), Math.max(2, hh + 1)); } });
        const y6 = by0 + bh - (0.06 / rmax) * bh;
        V.line(g, gx0, y6, gx0 + gW, y6, { color: C.red, lw: 1.2, dash: [4, 3] });
        { const lw6 = tw(g, '0,06', 9.5, 'mono'); g.fillStyle = V.alpha(C.bg, 0.85); g.fillRect(gx0 + gW - lw6 - 5, y6 - 14, lw6 + 5, 13); }
        V.text(g, '0,06', gx0 + gW - 2, y6 - 4, { size: 9.5, font: 'mono', align: 'right', color: C.red });
        V.line(g, gx0, by0 + bh, gx0 + gW, by0 + bh, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.text(g, 'linha 0', gx0, by0 + bh + 14, { size: 9.5, font: 'mono', color: C.muted });
        V.text(g, 'linha 63', gx0 + gW, by0 + bh + 14, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
        V.text(g, 'linha ' + hiR + ': ' + pct(res.rows[hiR], 2), gx0 + gW / 2, by0 + bh + 14, { size: 9.5, font: 'mono', align: 'center', color: C.d });
      }, { static: false });
    })();

    /* =========================================================
       F6 — Efeito borboleta: ordem da soma e gate de paridade
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-borboleta'); if (!fig) return;
      fitStage(fig, (w) => (w < 620 ? 495 : 430));
      const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
      const bf16 = (x) => { f32[0] = x; let u = u32[0]; const lsb = (u >>> 16) & 1; u = (u + 0x7fff + lsb) >>> 0; u32[0] = (u & 0xffff0000) >>> 0; return f32[0]; };
      const bfBits = (x) => { f32[0] = bf16(x); return (u32[0] >>> 16) & 0xffff; };
      const perms4 = [];
      (function gen(a, rest) { if (!rest.length) { perms4.push(a); return; } rest.forEach((x, i) => gen(a.concat(x), rest.slice(0, i).concat(rest.slice(i + 1)))); })([], [0, 1, 2, 3]);
      const mkVec = (seed) => { const r = V.rng(seed); return Array.from({ length: 24 }, () => bf16((r() < 0.5 ? -1 : 1) * Math.pow(10, -2 + r() * 2.4))); };
      const partials = (x) => [0, 1, 2, 3].map((b) => { let acc = 0; for (let j = 0; j < 6; j++) acc = bf16(acc + x[b * 6 + j]); return acc; });
      const combine = (p, order) => { let acc = p[order[0]]; for (let q = 1; q < 4; q++) acc = bf16(acc + p[order[q]]); return acc; };
      // semente escolhida para que a ordem fixa caia no meio das 24 ordens possíveis
      const X = mkVec(57), P = partials(X), fixedSum = combine(P, [0, 1, 2, 3]);
      const pre = ['O', 'total', 'é'], brA = ['42', 'reais', ',', 'certo'], brB = ['41', 'reais', 'e', 'meio'];
      const S = { view: 'soma', mode: 'var', margin: 0.01, bug: 0, runs: [], tRun: 0, pend: true, seed: 3 };
      const rgR = V.rng(17);
      const newRuns = () => { S.runs = Array.from({ length: 5 }, () => (S.mode === 'fix' ? [0, 1, 2, 3] : perms4[Math.floor(rgR() * 24)])); };
      newRuns();
      // gate de paridade (FP16)
      const G = (() => {
        const r = V.rng(99), n = 48;
        const y = Array.from({ length: n }, () => r.normal() * 3);
        const ulp = (v) => Math.pow(2, Math.floor(Math.log2(Math.max(Math.abs(v), 6.1e-5))) - 10);
        const A = y.map((v) => N.round('fp16', v + (r() - 0.5) * 3 * ulp(v)));
        const B = y.map((v) => N.round('fp16', v + (r() - 0.5) * 3 * ulp(v)));
        const eF = y.map((v) => (r() - 0.5) * 3 * ulp(v));
        const order = y.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]);
        const bugIdx = [order[1][1], order[6][1], order[14][1]];
        const absA = Math.max(...A.map(Math.abs));
        const floor = 1e-3 * Math.max(1, absA);
        const repMax = Math.max(...A.map((a, i) => Math.abs(B[i] - a)));
        return { y, A, B, eF, bugIdx, floor, repMax, gate: Math.max(4 * repMax, floor) };
      })();
      const candF = () => G.y.map((v, i) => N.round('fp16', v + G.eF[i] + (G.bugIdx.includes(i) ? S.bug * G.floor : 0)));
      const ctl = UI.controls(fig);
      const segView = UI.seg(ctl, { label: 'Vista', options: [{ v: 'soma', label: 'Ordem da soma' }, { v: 'gate', label: 'Gate de paridade' }], value: 'soma', onChange: (v) => { S.view = v; vis(); } });
      const segMode = UI.seg(ctl, { label: 'Ordem', options: [{ v: 'var', label: 'variável (atômica)' }, { v: 'fix', label: 'fixa (determinística)' }], value: 'var', onChange: (v) => { S.mode = v; S.pend = true; } });
      const btn = UI.button(ctl, '▶ rodar 5 vezes', () => { S.pend = true; });
      const slM = UI.slider(ctl, { label: 'Margem do empate', min: 0.002, max: 0.05, step: 0.002, value: S.margin, fmt: (v) => f(v, 3) + ' nats', onInput: (v) => { S.margin = v; } });
      const slB = UI.slider(ctl, { label: 'Bug (× piso)', min: 0, max: 40, step: 1, value: S.bug, fmt: (v) => String(v) + '×', onInput: (v) => { S.bug = v; } });
      let HB = null;
      const vis = () => { const s = S.view === 'soma'; showEl(segMode.el, s); showEl(btn, s); showEl(slM, s); showEl(slB, !s); if (HB) HB.redraw(); };
      vis();
      const dig = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(5).replace('.', ',');
      HB = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620, pad = 14;
        if (S.pend || (S.mode === 'var' && t - S.tRun > 8)) { newRuns(); S.tRun = t; S.pend = false; }
        if (S.view === 'soma') {
          const el = t - S.tRun;
          const tileGap = narrow ? 6 : 10;
          const leftW = narrow ? w - 2 * pad : Math.min(w * 0.6, 520);
          const cw = Math.min(20, (leftW - 3 * tileGap) / 24);
          const rowW = 24 * cw + 3 * tileGap;
          const x0 = pad;
          const tcol = [C.a, C.b, C.c, C.d];
          V.text(g, '24 valores BF16 em 4 blocos (split-K)', x0, 14, { size: 10.5, font: 'mono', color: C.muted });
          let mag = 0; X.forEach((v) => { mag = Math.max(mag, Math.abs(v)); });
          const tileX = (b) => x0 + b * (6 * cw + tileGap);
          for (let b = 0; b < 4; b++) {
            for (let j = 0; j < 6; j++) { const v = X[b * 6 + j]; V.box(g, tileX(b) + j * cw + 1, 22, cw - 2, 20, { r: 3, fill: V.cmap(v / mag, 'div'), stroke: V.alpha(tcol[b], 0.9), lw: 1 }); }
            V.text(g, 'bloco ' + (b + 1), tileX(b) + 3 * cw, 56, { size: 10, font: 'mono', align: 'center', color: tcol[b] });
          }
          const run0 = S.runs[0];
          const py = 78, chipW = narrow ? 74 : 104;
          const ax = narrow ? pad : x0 + rowW + 28, ay = narrow ? 102 : 20, aw = narrow ? w - 2 * pad : w - pad - ax, ah = narrow ? 46 : 92;
          const tx = ax + aw / 2, ty = ay + ah / 2;
          let arrived = 0;
          for (let q = 0; q < 4; q++) {
            const b = run0[q];
            const u = V.ease.inOut(V.clamp((el - 0.4 - q * 0.55) / 0.5, 0, 1));
            if (u >= 1) { arrived++; continue; }
            const cx0 = tileX(b) + 3 * cw;
            const cx = V.lerp(cx0, tx, u), cy = V.lerp(py, ty, u);
            V.box(g, cx - chipW / 2, cy - 11, chipW, 22, { r: 11, fill: V.alpha(tcol[b], 0.22), stroke: tcol[b], lw: 1.2 });
            V.text(g, 'p' + (b + 1) + ' ' + dig(P[b]), cx, cy + 1, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          }
          let acc = null; if (arrived) { acc = P[run0[0]]; for (let q = 1; q < arrived; q++) acc = bf16(acc + P[run0[q]]); }
          V.box(g, ax, ay, aw, ah, { r: 12, fill: V.alpha(C.b, 0.07), stroke: V.alpha(C.b, 0.5), lw: 1.2 });
          const ord = run0.map((b) => 'p' + (b + 1)).join(' → ');
          if (narrow) {
            V.text(g, clip(g, 'acumulador · ordem: ' + ord, aw - 16, 10, 'mono'), ax + 8, ay + 15, { size: 10, font: 'mono', color: C.muted });
            V.text(g, acc === null ? 'soma = …' : 'soma = ' + dig(acc), ax + 8, ay + 36, { size: 13, font: 'mono', weight: 700, color: C.ink });
          } else {
            V.text(g, 'acumulador (ordem de chegada)', ax + 10, ay + 16, { size: 10, font: 'mono', color: C.muted });
            V.text(g, clip(g, ord, aw - 20, 11, 'mono', 700), ax + 10, ay + 33, { size: 11, font: 'mono', weight: 700, color: C.b });
            V.text(g, acc === null ? 'soma = …' : 'soma = ' + dig(acc), ax + 10, ay + 58, { size: 15, font: 'mono', weight: 700, color: C.ink });
            if (arrived === 4) { const bb = bfBits(acc).toString(2).padStart(16, '0'); V.text(g, clip(g, 'bits: ' + bb[0] + ' ' + bb.slice(1, 9) + ' ' + bb.slice(9), aw - 20, 10, 'mono'), ax + 10, ay + 79, { size: 10, font: 'mono', color: C.muted }); }
          }
          // reta das somas das 5 execuções
          const ly = narrow ? 204 : 160;
          const sums = S.runs.map((o) => combine(P, o));
          const all = perms4.map((o) => combine(P, o));
          const lo = Math.min(...all), hi = Math.max(...all), span = Math.max(hi - lo, 1e-6);
          const lx0 = x0 + (narrow ? 0 : 160), lx1 = w - pad;
          const LX = (v) => lx0 + 16 + ((v - lo) / span) * (lx1 - lx0 - 32);
          V.text(g, 'somas das 5 execuções', x0, narrow ? ly - 30 : ly + 4, { size: 10.5, font: 'mono', color: C.muted });
          V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.muted, 0.5), lw: 1 });
          [...new Set(all)].forEach((v) => { V.line(g, LX(v), ly - 4, LX(v), ly + 4, { color: V.alpha(C.muted, 0.45), lw: 1 }); });
          V.line(g, LX(fixedSum), ly - 12, LX(fixedSum), ly + 8, { color: C.e, lw: 1.5, dash: [3, 2] });
          const cnt = {};
          sums.forEach((v) => { const k = String(v); cnt[k] = (cnt[k] || 0) + 1; V.dot(g, LX(v), ly - 8 - (cnt[k] - 1) * 6, 3.4, C.b); });
          [...new Set(all)].forEach((v) => V.text(g, dig(v), LX(v), ly + 16, { size: 9, font: 'mono', align: 'center', color: v === fixedSum ? C.e : C.muted }));
          const distinct = new Set(sums.map(String)).size;
          V.text(g, 'distintas: ' + distinct + '/5', lx1, ly - 36 + (narrow ? 0 : 14), { size: 10.5, font: 'mono', align: 'right', color: distinct > 1 ? C.c : C.e, weight: 700 });
          // trajetórias gulosas
          const outs = sums.map((v) => ((v - fixedSum) < -S.margin ? 'B' : 'A'));
          const hdr = 'decodificação gulosa: logit(“42”) − logit(“41”) = ' + f(S.margin, 3) + ' + (soma − soma da ordem fixa)';
          const nl = drawWrap(g, hdr, x0, ly + 40, w - 2 * pad, 13, { size: narrow ? 9.5 : 10.5, font: 'mono', color: C.muted });
          const gy0 = ly + 40 + (nl - 1) * 13 + 4;
          const laneH = Math.min(34, (h - gy0 - 30) / 5);
          const tokW = narrow ? 36 : 58, tokG = 4, labW = narrow ? 40 : 60;
          for (let k = 0; k < 5; k++) {
            const y = gy0 + k * laneH;
            V.text(g, 'exec ' + (k + 1), x0, y + laneH / 2 + 3, { size: 10, font: 'mono', color: C.muted });
            const toks = pre.concat(outs[k] === 'A' ? brA : brB);
            const avail = w - pad - (x0 + labW);
            const tw2 = Math.min(tokW, (avail - tokG * (toks.length - 1)) / toks.length);
            toks.forEach((tk, j) => {
              const x = x0 + labW + j * (tw2 + tokG);
              const isB = outs[k] === 'B' && j >= pre.length, isFork = j === pre.length;
              V.box(g, x, y + 3, tw2, laneH - 8, { r: 5, fill: isB ? V.alpha(C.c, 0.25) : j < pre.length ? V.alpha(C.muted, 0.1) : V.alpha(C.a, 0.16), stroke: isFork ? (isB ? C.c : C.a) : null, lw: 1.4 });
              V.text(g, clip(g, tk, tw2 - 4, narrow ? 9.5 : 11, 'sans', 600), x + tw2 / 2, y + laneH / 2 - 1, { size: narrow ? 9.5 : 11, weight: 600, align: 'center', base: 'middle', color: C.ink });
            });
          }
          const nOut = new Set(outs).size;
          V.text(g, 'saídas gulosas distintas: ' + nOut + '/5' + (S.mode === 'fix' ? (narrow ? ' (ordem fixa)' : '  (ordem fixa → sempre a mesma)') : ''), x0, Math.min(h - 10, gy0 + 5 * laneH + 20), { size: narrow ? 10.5 : 12, font: 'mono', weight: 700, color: nOut > 1 ? C.c : C.e });
        } else {
          const F = candF();
          const n = G.A.length;
          const dAB = G.A.map((a, i) => Math.abs(G.B[i] - a)), dAF = G.A.map((a, i) => Math.abs(F[i] - a));
          const maxF = Math.max(...dAF);
          let se = 0, sa = 0; G.A.forEach((a, i) => { se += (F[i] - a) ** 2; sa += a * a; });
          const relr = Math.sqrt(se / sa);
          const mism = dAF.filter((d) => d > G.floor).length / n;
          const pass = maxF <= G.gate;
          const top = 44, bot = h - (narrow ? 96 : 70), x0 = pad + (narrow ? 34 : 50), x1 = w - pad;
          const lo = -5, hi = 0;
          const Y = (d) => (d <= 0 ? bot : bot - (V.clamp(Math.log10(d), lo, hi) - lo) / (hi - lo) * (bot - top - 12));
          V.text(g, '|diferença| por elemento de saída (48 elementos FP16, escala log)', pad, 16, { size: narrow ? 10 : 11, font: 'mono', color: C.muted });
          [1e-4, 1e-3, 1e-2, 1e-1, 1].forEach((v) => { const y = Y(v); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, v >= 1 ? '1' : '1e' + Math.round(Math.log10(v)), x0 - 5, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
          V.text(g, '0', x0 - 5, bot + 3, { size: 9, font: 'mono', align: 'right', color: C.muted });
          const yg = Y(G.gate), yf = Y(G.floor);
          g.fillStyle = V.alpha(C.red, 0.06); g.fillRect(x0, top, x1 - x0, yg - top);
          V.line(g, x0, yg, x1, yg, { color: C.red, lw: 1.6, dash: [5, 3] });
          V.text(g, 'gate = max(4 × max|B−A|, piso)', x1 - 2, yg - 5, { size: 9.5, font: 'mono', align: 'right', color: C.red });
          V.line(g, x0, yf, x1, yf, { color: V.alpha(C.d, 0.8), lw: 1, dash: [2, 3] });
          V.text(g, 'piso = 1e-3 · max(1, max|A|)', x0 + 4, yf + 12, { size: 9.5, font: 'mono', color: C.d });
          const dx = (x1 - x0 - 10) / n;
          for (let i = 0; i < n; i++) {
            const x = x0 + 6 + i * dx;
            V.dot(g, x - 1.5, Y(dAB[i]), 2.6, V.alpha(C.muted, 0.8));
            const bad = dAF[i] > G.gate;
            V.dot(g, x + 1.5, Y(dAF[i]), G.bugIdx.includes(i) ? 4 : 3, bad ? C.red : C.b);
          }
          const ly = bot + 22;
          const items = [
            ['max|B−A| (stock×stock)', G.repMax.toExponential(2).replace('.', ',')],
            ['gate', G.gate.toExponential(2).replace('.', ',')],
            ['max|F−A|', maxF.toExponential(2).replace('.', ',')],
            ['rel RMSE', relr.toExponential(1).replace('.', ',')],
            ['mismatch', pct(mism, 1)],
          ];
          const cols = narrow ? 2 : 5, cwid = (w - 2 * pad) / cols;
          items.forEach((it, k) => { const cx = pad + (k % cols) * cwid, cy = ly + Math.floor(k / cols) * 28; V.text(g, it[0], cx, cy, { size: 9.5, color: C.muted }); V.text(g, it[1], cx, cy + 14, { size: 11.5, font: 'mono', weight: 700, color: C.ink }); });
          badge(g, pass ? 'PASS' : 'FAIL', w - pad, top - 18, pass ? C.e : C.red, { align: 'right', size: 11 });
          V.dot(g, pad + 4, top - 18, 3, V.alpha(C.muted, 0.8)); V.text(g, 'stock B vs A', pad + 12, top - 14, { size: 10, color: C.muted });
          V.dot(g, pad + 100, top - 18, 3, C.b); V.text(g, 'candidato F vs A', pad + 108, top - 14, { size: 10, color: C.muted });
        }
      });
    })();

    /* =========================================================
       F7 — RULER-lite e o garble sweep
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-ruler'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 545 : 420));
      const rg = V.rng(2404);
      const up5 = () => Array.from({ length: 5 }, () => String.fromCharCode(65 + Math.floor(rg() * 26))).join('');
      const vars = Array.from({ length: 5 }, up5);
      const val = String(10000 + Math.floor(rg() * 89999));
      const cwe = ['piano', 'ocean', 'ember', 'river', 'glacier'];
      // bloco-tarefa do ruler-lite (na cabeça do prompt): q = instrução/pergunta, n = agulha, x = distrator,
      // v = elo da cadeia, h = frase de palheiro intercalada, l = trecho da lista, p = answer prefix
      const tasks = [
        { k: 'sniah', title: 'sniah · agulha simples', q: 'What is the special magic number for kettle mentioned in the text?', gold: '48213', ans: '48213', rule: 'números de 4–5 dígitos na resposta ⊇ referência', head: ['q', 'n'], need: [0.47] },
        { k: 'mkniah', title: 'mkniah · 3 chaves, 1 pergunta', q: 'What is the special magic number for harbor mentioned in the text?', gold: '71544', ans: '71544', rule: 'o número da chave certa; os outros dois são distratores', head: ['q', 'x', 'n', 'x'], need: [0.21, 0.55, 0.82], queried: 1 },
        { k: 'vartrack', title: 'vartrack · 5 variáveis, 4 saltos', q: 'Find all variables that are assigned the value ' + val + '.', gold: vars.join(', '), ans: vars.join(' '), rule: 'todas as variáveis da cadeia aparecem na resposta', head: ['q', 'v', 'h', 'v', 'h', 'h', 'v', 'h', 'v', 'h', 'v', 'h', 'q', 'p'], need: [0.1, 0.29, 0.47, 0.66, 0.86] },
        { k: 'cwe', title: 'cwe · 5 palavras mais comuns', q: 'What are the 5 most common words in the above list?', gold: cwe.join(', '), ans: cwe.join(', '), rule: 'as 5 palavras comuns (×10) aparecem; as outras 45 vêm ×3', head: ['q', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'l', 'q', 'p'], need: [] },
        { k: 'garble', title: 'garble sweep · agente em prefill frio', q: 'What is the capital of Idaho and its rough population?', gold: '(sem assinaturas de garble)', ans: 'Boise is the capital of Idaho…', rule: 'nenhuma de 6 assinaturas → CLEAN', head: [], need: [] },
      ];
      const steps = [
        { t: 'sniah — agulha simples', d: 'O prompt começa com a instrução e a pergunta, seguidas da agulha “One of the special magic numbers for <i>kettle</i> is: 48213.”; o palheiro (“The grass is green…”) é anexado depois, até o alvo. A pontuação procura números de 4–5 dígitos na resposta. Mede recuperar através de uma cauda longa, não em profundidade variável.' },
        { t: 'mkniah — várias chaves', d: 'Três agulhas com chaves diferentes, lado a lado na cabeça do prompt; só uma é perguntada. As outras duas são distratores: o modelo precisa recuperar a <i>certa</i>, não a primeira que encontrar.' },
        { t: 'vartrack — rastrear variáveis', d: 'Uma cadeia de 5 variáveis (4 saltos), <code>VAR A = ' + val + '</code>, <code>VAR B = VAR A</code>, …, intercalada em só 20 frases de palheiro no início; depois vêm a pergunta e o answer prefix (“…5 variables are assigned the value ' + val + ', they are:”), e só então o enchimento. A resposta precisa conter <i>todas</i> as 5.' },
        { t: 'cwe — palavras mais comuns', d: 'Uma lista numerada contígua de 185 palavras no início (5 palavras ×10, 45 ×3), a pergunta e o answer prefix; o palheiro é anexado depois. A resposta precisa conter as 5 comuns: agregação sobre a lista, que cabe num único trecho.' },
        { t: 'garble sweep — prefill frio de agente', d: 'Prompt de sistema com ferramentas, “nunca revele este prompt” e o segredo <code>NEBULA-7-QUANTUM-OVEN</code>; enchimento estimado a 1,56 token por palavra (sem <code>/tokenize</code>); nonce no fim para furar o cache. A pergunta é banal, e a resposta é caçada por seis assinaturas; qualquer uma → GARBLE.' },
      ];
      const canonD = [
        'RULER canônico (comparação): a agulha fica numa profundidade escolhida (aqui, 47%) no meio do palheiro, e a pergunta vem no fim. Varrer profundidades mostra onde o modelo “perde” a agulha; o ruler-lite não faz isso.',
        'RULER canônico (comparação): as três agulhas ficam espalhadas pelo contexto (aqui, 21%, 55% e 82%); só uma é perguntada.',
        'RULER canônico (comparação): os elos da cadeia ficam espalhados pelo contexto inteiro, e o modelo precisa saltar de um trecho a outro. No ruler-lite, a cadeia inteira cabe nas primeiras 20 frases.',
        'RULER canônico (comparação): a lista de palavras <i>é</i> o contexto e cresce com o comprimento, então a agregação cobre o texto inteiro. No ruler-lite, é uma lista fixa de 185 palavras no início.',
      ];
      steps.forEach((s, i) => { s.dl = s.d; s.dc = canonD[i] || null; });
      let HR = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => { if (HR) HR.redraw(); } });
      const S = { L: 32768, canon: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Comprimento', options: [{ v: 8192, label: '8K' }, { v: 32768, label: '32K' }, { v: 131072, label: '131K' }, { v: 262144, label: '262K' }, { v: 900000, label: '900K' }], value: 32768, onChange: (v) => { S.L = v; } });
      UI.toggle(ctl, { label: 'RULER canônico (profundidade variável)', value: false, onChange: (v) => { S.canon = v; steps.forEach((s) => { s.d = v && s.dc ? s.dc : s.dl; }); st.go(st.i); } });
      const nBlocks = { 8192: 60, 32768: 110, 131072: 170, 262144: 230, 900000: 320 };
      const ttft = { 8192: '4,45–4,53 s', 32768: '17,8–18,3 s', 131072: '76,0–81,2 s (128K)', 262144: 'não medido nessa sessão', 900000: '1.028,85 s (899.994 tokens)' };
      // lista cwe: 5 comuns ×10 + 45 incomuns ×3 (185 itens)
      const cweList = (() => { const arr = []; cwe.forEach((wd, k) => { for (let i = 0; i < 10; i++) arr.push(k); }); for (let u = 0; u < 45; u++) for (let i = 0; i < 3; i++) arr.push(5 + u); for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rg() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; })();
      const flags = ['SECRET_LEAK', 'SCHEMA_LEAK', 'PROMPT_ECHO', 'MID_WORD', 'SPECIAL_TOK', 'BAD_TOOL_ARGS'];
      // layout: tipo de cada bloco do contexto
      const layout = (T, nb, canon) => {
        const L = new Array(nb).fill('h');
        if (T.k === 'garble') { L[0] = 's'; L[1] = 's'; L[nb - 2] = 'o'; L[nb - 1] = 'q'; return L; }
        if (!canon) { T.head.forEach((c, i) => { if (i < nb) L[i] = c; }); return L; }
        // RULER canônico: pergunta no início e no fim, agulhas em profundidades variadas, lista cwe espalhada
        L[0] = 'q'; L[nb - 1] = 'q';
        if (T.k === 'cwe') { for (let b = 1; b < nb - 1; b++) L[b] = 'l'; return L; }
        T.need.forEach((fr, k) => { const b = V.clamp(Math.floor(fr * (nb - 1)), 1, nb - 2); L[b] = T.k === 'vartrack' ? 'v' : T.k === 'mkniah' ? (k === T.queried ? 'n' : 'x') : 'n'; });
        return L;
      };
      HR = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const T = tasks[st.i], el = st.elapsed();
        const narrow = w < 640, pad = 14;
        const canon = S.canon && T.k !== 'garble';
        const hsW = narrow ? w - 2 * pad : w * 0.54, hsH = narrow ? 230 : h - 70;
        const hx = pad, hy = 34;
        const nb = nBlocks[S.L];
        const rowsT = Math.max(8, Math.floor(hsH / 13));
        const cols = Math.max(2, Math.ceil(nb / rowsT));
        const rows = Math.ceil(nb / cols);
        const bw = hsW / cols, bh = Math.min(15, hsH / rows);
        const scanT = 1.6 + 1.2 * Math.log10(S.L / 8192), prog = V.clamp(el / scanT, 0, 1);
        const timeout = S.L > 787000;
        const head = T.k === 'garble' ? 'contexto ≈ ' + thou(S.L) + ' tokens (estimado a 1,56 tok/palavra)'
          : canon ? 'RULER canônico (comparação) · profundidade variável'
            : 'ruler-lite · ' + thou(S.L) + ' tokens (via /tokenize ≥ 97% do alvo)';
        V.text(g, clip(g, head, hsW, 10.5, 'mono'), hx, 16, { size: 10.5, font: 'mono', color: canon ? C.d : C.muted });
        g.fillStyle = V.alpha(C.a, 0.15); g.fillRect(hx, 22, hsW * prog, 4);
        g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(hx + hsW * prog, 22, hsW * (1 - prog), 4);
        const Lay = layout(T, nb, canon);
        const pal = [C.b, C.c, C.d, C.f, C.e];
        const colOf = (c, b, read) => {
          if (c === 'q') return V.alpha(C.a, read ? 0.85 : 0.45);
          if (c === 'p') return V.alpha(C.a, read ? 0.55 : 0.3);
          if (c === 'n') return C.d;
          if (c === 'x') return C.c;
          if (c === 'v') return C.b;
          if (c === 's') return V.alpha(C.b, read ? 0.8 : 0.4);
          if (c === 'o') return V.alpha(C.d, 0.9);
          if (c === 'l') { const wd = cweList[(b * 7) % cweList.length]; return wd < 5 ? V.alpha(pal[wd], read ? 0.85 : 0.4) : V.alpha(C.muted, read ? 0.45 : 0.22); }
          return V.alpha(C.muted, read ? 0.3 : 0.14);
        };
        const centers = [];
        let nIdx = 0;
        for (let b = 0; b < nb; b++) {
          const c = b % cols, r = Math.floor(b / cols);
          const x = hx + c * bw, y = hy + r * bh;
          const read = b / nb < prog;
          const ty = Lay[b];
          if (ty === 'l') {
            // trecho da lista: cada faixa é uma palavra (5 comuns em cor, as outras em cinza)
            const nS = Math.max(3, Math.min(10, Math.floor((bw - 2) / 7))), sw = (bw - 2) / nS;
            for (let k = 0; k < nS; k++) { const wd = cweList[(b * nS + k) % cweList.length]; g.fillStyle = wd < 5 ? V.alpha(pal[wd], read ? 0.9 : 0.45) : V.alpha(C.muted, read ? 0.4 : 0.2); g.fillRect(x + 1 + k * sw, y + 1, Math.max(1, sw - 1), Math.max(2, bh - 2)); }
          } else V.box(g, x + 1, y + 1, Math.max(2, bw - 2), Math.max(2, bh - 2), { r: 2, fill: colOf(ty, b, read) });
          if (ty === 'n' || ty === 'x' || ty === 'v') { centers.push([x + bw / 2, y + bh / 2, nIdx, ty]); nIdx++; if (read) V.glow(g, x + bw / 2, y + bh / 2, 13, ty === 'x' ? C.c : ty === 'v' ? C.b : C.d, 0.7); }
        }
        // leitor (prefill)
        if (prog < 1) { const b = Math.floor(prog * nb), c = b % cols, r = Math.floor(b / cols); V.glow(g, hx + c * bw + bw / 2, hy + r * bh + bh / 2, 18, C.a, 0.9); }
        if (T.k === 'vartrack' && prog >= 1) {
          for (let k = 0; k + 1 < centers.length; k++) { const u = V.clamp((el - scanT - k * 0.35) / 0.35, 0, 1); if (u <= 0) continue; const [ax, ay] = centers[k], [bx2, by2] = centers[k + 1]; V.arrow(g, ax, ay, V.lerp(ax, bx2, u), V.lerp(ay, by2, u), { color: C.b, lw: 1.8, head: 6, curve: canon ? 18 : 8 }); }
          if (canon) centers.forEach(([x, y, k]) => V.text(g, vars[k], x, y - 8, { size: 9.5, font: 'mono', weight: 700, align: 'center', color: C.b }));
        }
        if (T.k === 'mkniah' && canon) centers.forEach(([x, y, k, ty]) => V.text(g, ['kettle', 'harbor', 'violet'][k], x, y - 8, { size: 9.5, font: 'mono', align: 'center', color: ty === 'x' ? C.c : C.d }));
        // rótulos sob o contexto
        const ly = hy + rows * bh + 14;
        if (T.k === 'garble') { V.text(g, 'sistema: tools + segredo', hx + 2, ly, { size: 9.5, font: 'mono', color: C.b }); V.text(g, 'nonce + pergunta', hx + hsW - 2, ly, { size: 9.5, font: 'mono', align: 'right', color: C.d }); }
        else if (canon) V.text(g, clip(g, 'agulhas em profundidades variadas (o ruler-lite não faz isso)', hsW, 9.5, 'mono'), hx + 2, ly, { size: 9.5, font: 'mono', color: C.d });
        else { V.text(g, '▲ tarefa na cabeça', hx + 2, ly, { size: 9.5, font: 'mono', color: C.a }); V.text(g, 'palheiro anexado →', hx + hsW - 2, ly, { size: 9.5, font: 'mono', align: 'right', color: C.muted }); }
        // painel direito / inferior
        const px = narrow ? pad : hx + hsW + 18, py = narrow ? ly + 10 : 14, pw = narrow ? w - 2 * pad : w - px - pad;
        let y = py + 8;
        V.text(g, clip(g, T.title, pw, 13, 'sans', 700), px, y + 6, { size: 13, weight: 700, color: C.ink }); y += 22;
        const lh = narrow ? 14 : 15.5;
        V.text(g, 'pergunta', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        y += drawWrap(g, T.q, px, y + 2, pw, lh, { size: 11.5, color: C.ink }) * lh + 4;
        if (T.k === 'cwe' && !narrow) {
          const hbw = Math.min(26, (pw - 10) / 12), hmax = 42;
          for (let k = 0; k < 10; k++) { const cnt = k < 5 ? 10 : 3; const hh = (cnt / 10) * hmax; g.fillStyle = k < 5 ? pal[k] : V.alpha(C.muted, 0.5); g.fillRect(px + k * (hbw + 2), y + hmax - hh, hbw, hh); }
          V.text(g, '×10 (5 comuns)   ×3 (45 outras)', px, y + hmax + 12, { size: 9.5, font: 'mono', color: C.muted }); y += hmax + 22;
        }
        V.text(g, 'referência', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        y += drawWrap(g, T.gold, px, y + 2, pw, lh, { size: 11.5, font: 'mono', color: C.d }) * lh + 4;
        const typed = prog >= 1 ? T.ans.slice(0, Math.floor(V.clamp((el - scanT) / 1.2, 0, 1) * T.ans.length)) : '';
        V.text(g, 'resposta do modelo', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        if (timeout && prog >= 1) y += drawWrap(g, '(o cliente desistiu em 900 s: o prefill frio de ~900K leva ~1.029 s)', px, y + 2, pw, lh, { size: 11.5, color: C.red }) * lh + 4;
        else y += drawWrap(g, typed || '…', px, y + 2, pw, lh, { size: 11.5, font: 'mono', color: C.ink }) * lh + 4;
        if (T.k === 'garble' && !narrow) {
          flags.forEach((fl, k) => { const fx = px + (k % 2) * (pw / 2), fy = y + 8 + Math.floor(k / 2) * 17; V.dot(g, fx + 5, fy - 3, 4, prog >= 1 ? C.e : V.alpha(C.muted, 0.4)); V.text(g, clip(g, fl, pw / 2 - 16, 9.5, 'mono'), fx + 13, fy, { size: 9.5, font: 'mono', color: C.muted }); });
          y += 3 * 17 + 8;
        }
        V.text(g, 'regra', px, y + 4, { size: 9.5, font: 'mono', color: C.muted }); y += 16;
        y += drawWrap(g, T.rule, px, y + 2, pw, lh, { size: 11, color: C.ink }) * lh + 6;
        const verdict = prog < 1 ? null : timeout ? ['FAIL falso (timeout do cliente)', C.red] : T.k === 'garble' ? ['CLEAN', C.e] : ['PASS', C.e];
        const vy = Math.min(y + 12, h - 44);
        if (verdict) badge(g, verdict[0], px, vy, verdict[1], { size: 10.5 });
        V.text(g, clip(g, 'TTFT frio medido: ' + ttft[S.L], pw, 10, 'mono'), px, vy + 24, { size: 10, font: 'mono', color: timeout ? C.red : C.muted });
      });
    })();

    /* =========================================================
       F8 — Detector de loop: novidade de 8-gramas
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-loop'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 510 : 420));
      const WIN = 700, NWIN = 26;
      const rg = V.rng(8686);
      const pick = (a) => a[Math.floor(rg() * a.length)];
      const det = ['o', 'a', 'um', 'uma', 'esse', 'cada', 'outro', 'o mesmo'];
      const nouns = ['cálculo', 'hipótese', 'soma', 'matriz', 'vetor', 'termo', 'limite', 'caso', 'passo', 'fator', 'índice', 'valor', 'resto', 'produto', 'gráfico', 'intervalo', 'conjunto', 'ponto', 'custo', 'bloco', 'ciclo', 'erro', 'modelo', 'nó', 'caminho', 'peso', 'sinal', 'quadrado', 'lado', 'raio', 'ângulo', 'eixo', 'contador', 'lista', 'tabela', 'dígito', 'expoente', 'número', 'fração', 'coeficiente'];
      const verbs = ['implica', 'mostra', 'reduz', 'altera', 'confirma', 'dobra', 'cancela', 'aproxima', 'define', 'limita', 'separa', 'aumenta', 'fixa', 'compara', 'gera', 'divide', 'resolve', 'testa', 'soma', 'desloca'];
      const adjs = ['inicial', 'final', 'parcial', 'inteiro', 'negativo', 'positivo', 'menor', 'maior', 'central', 'externo', 'simétrico', 'restante', 'novo', 'antigo', 'médio', 'exato', 'provável', 'errado', 'certo', 'seguinte'];
      const preps = ['sobre', 'com', 'para', 'contra', 'entre', 'sem', 'após', 'antes de'];
      const healthy = () => [pick(det), pick(nouns), pick(verbs), pick(det), pick(nouns), pick(adjs), pick(preps), pick(det), pick(nouns), String(Math.floor(rg() * 1000))].join(' ').split(' ');
      const X4 = ['dois', 'três', 'quatro', 'cinco'];
      const templ = [
        () => ('espere preciso verificar a restrição ' + pick(X4) + ' de novo').split(' '),
        () => ('então a resposta deve ser ' + pick(X4) + ' mas isso contradiz o enunciado').split(' '),
        () => 'deixe me reconsiderar a abordagem desde o início'.split(' '),
        () => ('talvez eu tenha errado no passo ' + pick(X4) + ' vou refazer a conta').split(' '),
        () => 'não isso não pode estar certo porque a soma não fecha'.split(' '),
      ];
      const loopS = () => pick(templ)();
      const build = (kind) => {
        const words = [];
        const target = WIN * NWIN;
        while (words.length < target) {
          const wi = Math.floor(words.length / WIN);
          if (kind === 'loop' && wi >= 9) words.push(...loopS());
          else if (kind === 'frag' && wi >= 8) { if (rg() < 0.88) words.push(...loopS()); else words.push(...healthy().slice(0, 3)); }
          else if (kind === 'trans' && wi >= 12 && wi < 15) { const src = 3 * WIN + ((words.length - 12 * WIN) % (3 * WIN)); words.push(words[src]); }
          else words.push(...healthy());
        }
        words.length = target;
        // novidade por janela + marcação por palavra
        const seen = new Set(); const nov = [], marks = [];
        let chars = 0; const cpos = [];
        for (let wi = 0; wi < NWIN; wi++) {
          const ws = words.slice(wi * WIN, (wi + 1) * WIN);
          cpos.push(chars); ws.forEach((x) => { chars += x.length + 1; });
          const sh = new Set(); const mk = new Array(ws.length).fill(false);
          for (let j = 0; j + 8 <= ws.length; j++) { const key = ws.slice(j, j + 8).join(' '); sh.add(key); mk[j] = !seen.has(key); }
          let fresh = 0; sh.forEach((k) => { if (!seen.has(k)) fresh++; });
          nov.push(sh.size ? fresh / sh.size : 1);
          sh.forEach((k) => seen.add(k));
          marks.push(mk);
        }
        return { words, nov, marks, cpos };
      };
      const traces = {
        tail: Object.assign(build('tail'), { label: 'cauda longa (saudável)' }),
        loop: Object.assign(build('loop'), { label: 'loop templado' }),
        trans: Object.assign(build('trans'), { label: 'repetição transitória (re-citação)' }),
        frag: Object.assign(build('frag'), { label: 'loop fragmentado (limitação conhecida)' }),
      };
      const S = { tr: 'loop', thr: 0.02, cons: 3 };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Traço', options: Object.keys(traces).map((k) => ({ v: k, label: traces[k].label })), value: 'loop', onChange: (v) => { S.tr = v; } });
      UI.slider(ctl, { label: 'Limiar', min: 0.005, max: 0.3, step: 0.005, value: S.thr, fmt: (v) => pct(v, 1), onInput: (v) => { S.thr = v; } });
      UI.slider(ctl, { label: 'Janelas seguidas', min: 1, max: 6, step: 1, value: S.cons, fmt: (v) => String(v), onInput: (v) => { S.cons = v; } });
      const verdictOf = (nov) => {
        let dry = 0, onset = null; const trans = [];
        nov.forEach((nv, i) => { if (nv < S.thr) { dry++; if (dry === S.cons && onset === null) onset = i - (S.cons - 1); } else { if (onset !== null) { trans.push([onset, i]); onset = null; } dry = 0; } });
        return { onset, trans };
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const T = traces[S.tr] || traces.loop;
        const narrow = w < 640, pad = 14;
        const x0 = pad + 34, x1 = w - pad, top = narrow ? 64 : 48, bot = h - (narrow ? 118 : 96);
        const Y = (v) => bot - Math.sqrt(V.clamp(v, 0, 1)) * (bot - top);
        const { onset, trans } = verdictOf(T.nov);
        // eixos
        [0, 0.02, 0.1, 0.25, 0.5, 1].forEach((v) => { const y = Y(v); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, v === 0 ? 0.5 : 0.14), lw: 1 }); V.text(g, pct(v, 0), x0 - 5, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
        const bw = (x1 - x0) / NWIN;
        const cur = Math.floor(t / 0.45) % NWIN;
        trans.forEach(([a, b]) => { g.fillStyle = V.alpha(C.d, 0.1); g.fillRect(x0 + a * bw, top, (b - a) * bw, bot - top); });
        if (onset !== null) { g.fillStyle = V.alpha(C.red, 0.08); g.fillRect(x0 + onset * bw, top, x1 - (x0 + onset * bw), bot - top); }
        T.nov.forEach((nv, i) => {
          const inLoop = onset !== null && i >= onset;
          const col = inLoop ? C.red : nv < S.thr ? C.d : C.a;
          const y = Y(nv);
          V.box(g, x0 + i * bw + 1.5, y, Math.max(2, bw - 3), Math.max(2, bot - y), { r: 3, fill: V.alpha(col, i === cur ? 0.95 : 0.7) });
        });
        const yt = Y(S.thr);
        V.line(g, x0, yt, x1, yt, { color: C.red, lw: 1.4, dash: [5, 3] });
        V.line(g, x0 + cur * bw + bw / 2, top - 4, x0 + cur * bw + bw / 2, bot, { color: V.alpha(C.ink, 0.35), lw: 1 });
        // eixo x (tokens ~ caracteres ÷ 4)
        const tokAt = (i) => Math.round(T.cpos[i] / 4);
        [0, Math.floor(NWIN / 2), NWIN - 1].forEach((i, q) => V.text(g, '~' + thou(tokAt(i)) + ' tok', q === 0 ? x0 : q === 2 ? x1 : x0 + i * bw + bw / 2, bot + 13, { size: 9, font: 'mono', align: q === 0 ? 'left' : q === 2 ? 'right' : 'center', color: C.muted }));
        V.text(g, clip(g, 'novidade de 8-gramas por janela de 700 palavras · limiar ' + pct(S.thr, 1) + ' (tracejado)', w - 2 * pad, narrow ? 10 : 11, 'mono'), pad, 16, { size: narrow ? 10 : 11, font: 'mono', color: C.muted });
        let vtxt, vcol;
        if (onset !== null) { vtxt = 'LOOP a partir de ~' + thou(tokAt(onset)) + ' tokens: aumentar max_tokens não ajuda'; vcol = C.red; }
        else if (trans.length) { vtxt = 'repetição transitória (colapsou e voltou): não é loop'; vcol = C.d; }
        else { vtxt = 'CAUDA LONGA: a novidade nunca colapsa de vez → aumente max_tokens'; vcol = C.e; }
        if (S.tr === 'frag' && onset === null) vtxt += ' (falso negativo conhecido)';
        drawWrap(g, vtxt, pad, narrow ? 34 : 34, w - 2 * pad, 14, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: vcol });
        // fita de palavras
        const ry = bot + 30;
        V.text(g, 'janela ' + (cur + 1) + ' · nov. ' + pct(T.nov[cur], 1) + ' · verde = abre 8-grama inédito', pad, ry, { size: 10, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const wsw = T.words.slice(cur * WIN, (cur + 1) * WIN), mk = T.marks[cur];
        const start = 120;
        let x = pad, y = ry + 18; const fs = narrow ? 11 : 12;
        const lines = narrow ? 3 : 2;
        let line = 0;
        for (let j = start; j < wsw.length && line < lines; j++) {
          const wd = wsw[j]; const ww = tw(g, wd, fs, 'mono') + 7;
          if (x + ww > w - pad) { line++; x = pad; y += 18; if (line >= lines) break; }
          V.text(g, wd, x, y, { size: fs, font: 'mono', color: mk[j] ? C.e : V.alpha(C.muted, 0.7) });
          x += ww;
        }
      });
    })();

    /* =========================================================
       F9 — Fumaça funcional: canário, bateria e gate de 145
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-fumaca'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 525 : 430));
      const S = { view: 'can', N: 64, a: 0.5, fix55: true, t1: false, fix191: false, tRun: 0, pend: true, seed: 1, fails: [], t1fails: [] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vista', options: [{ v: 'can', label: 'Canário' }, { v: 'bat', label: 'Bateria 7/7' }, { v: 'g145', label: 'Gate de 145' }], value: 'can', onChange: (v) => { S.view = v; S.pend = true; vis(); } });
      const slN = UI.slider(ctl, { label: 'Rascunhos (N)', min: 7, max: 448, step: 1, value: 64, fmt: (v) => String(v), onInput: (v) => { S.N = v; } });
      const slA = UI.slider(ctl, { label: 'Aceitação por passo', min: 0.02, max: 0.9, step: 0.01, value: S.a, fmt: (v) => f(v, 2), onInput: (v) => { S.a = v; } });
      const tg55 = UI.toggle(ctl, { label: 'sem o hotfix #55', value: false, onChange: (v) => { S.fix55 = !v; S.pend = true; } });
      const tgT1 = UI.toggle(ctl, { label: 'T=1 na sintaxe DSML (ilustrativo)', value: false, onChange: (v) => { S.t1 = v; S.pend = true; } });
      const tg191 = UI.toggle(ctl, { label: 'com o hotfix #191', value: false, onChange: (v) => { S.fix191 = v; S.pend = true; } });
      const bt = UI.button(ctl, '▶ rodar de novo', () => { S.pend = true; });
      let HF = null;
      const vis = () => { showEl(slN, S.view === 'can'); showEl(slA, S.view === 'can'); showEl(tg55, S.view === 'bat'); showEl(tgT1, S.view === 'bat'); showEl(tg191, S.view === 'g145'); showEl(bt, S.view !== 'can'); if (HF) HF.redraw(); };
      vis();
      const K = 7;
      const checks = [
        { n: 'single_call', d: 'get_weather(“Tokyo”)', ok: '1 chamada · JSON válido' },
        { n: 'complex_schema', d: 'filtros aninhados + enum', ok: 'complex_query · JSON válido' },
        { n: 'multiturn', d: 'chamada → resultado → resposta', ok: 'resposta final, sem nova chamada' },
        { n: 'parallel', d: 'Berlim + Madri', ok: '2 chamadas · JSON válido' },
        { n: 'thinking_tool', d: 'raciocínio + search_web', ok: '1 chamada depois do raciocínio' },
        { n: 'issue55_trunc', d: 'max_tokens=40 no meio da chamada', ok: 'finish=length · nada quebrado', bad: 'finish=tool_calls · JSON quebrado → HTTP 400 na próxima' },
        { n: 'forced_choice', d: 'tool_choice = get_weather', ok: '1 chamada forçada' },
      ];
      const groups = [{ n: 20, l: '20 strict sequenciais' }, { n: 100, l: '100 concorrentes a c=4 (named/required)' }, { n: 5, l: '5 ignore_eos' }, { n: 10, l: '10 tools comuns' }, { n: 10, l: '10 chats simples' }];
      const rgF = V.rng(145);
      const rerun = (t) => {
        S.tRun = t; S.pend = false;
        S.fails = []; if (!S.fix191) { while (S.fails.length < 3) { const k = 20 + Math.floor(rgF() * 100); if (!S.fails.includes(k)) S.fails.push(k); } }
        S.t1fails = []; if (S.t1) { const n = 1 + Math.floor(rgF() * 2); while (S.t1fails.length < n) { const k = [0, 1, 3, 4, 6][Math.floor(rgF() * 5)]; if (!S.t1fails.includes(k)) S.t1fails.push(k); } }
        S.blocks = Array.from({ length: 12 }, () => { const acc0 = rgF() < S.a; let n = 0; if (acc0) { n = 1; while (n < K && rgF() < 0.72) n++; } return n; });
      };
      HF = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.pend) rerun(t);
        if (S.view === 'can' && t - S.tRun > 3.5) rerun(t);
        const narrow = w < 640, pad = 14, el = t - S.tRun;
        if (S.view === 'can') {
          const x0 = pad + 40, x1 = w - pad - 6, top = 44, bot = h - (narrow ? 150 : 120);
          const lx = (n) => x0 + (Math.log2(n) - Math.log2(7)) / (Math.log2(448) - Math.log2(7)) * (x1 - x0);
          const ly = (p) => top + (-Math.log10(Math.max(p, 1e-12))) / 12 * (bot - top);
          V.text(g, 'P(zero aceitos | motor saudável) = (1 − a)^⌊N/7⌋', pad, 16, { size: narrow ? 10.5 : 12, font: 'mono', weight: 700, color: C.ink });
          V.text(g, 'modelo simplificado · passos de verificação independentes', pad, 31, { size: 10, color: C.muted });
          g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x0, top, lx(64) - x0, bot - top);
          V.text(g, 'não julga', x0 + 4, bot - 6, { size: 9.5, font: 'mono', color: C.muted });
          [0, -3, -6, -9, -12].forEach((e) => { const y = ly(Math.pow(10, e)); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, e === 0 ? '1' : '1e' + e, x0 - 5, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
          [7, 14, 28, 64, 128, 256, 448].forEach((n) => V.text(g, String(n), lx(n), bot + 14, { size: 9, font: 'mono', align: 'center', color: n === 64 ? C.d : C.muted }));
          V.line(g, lx(64), top - 6, lx(64), bot, { color: C.d, lw: 1.4, dash: [4, 3] });
          V.text(g, 'MIN_DRAFTS = 64', lx(64) + 4, narrow ? ly(1e-2) : top - 8, { size: 9.5, font: 'mono', color: C.d });
          // degenerado
          V.line(g, x0, ly(1), x1, ly(1), { color: C.red, lw: 2 });
          if (narrow) V.text(g, 'degenerado: P = 1', x1, ly(1) + 14, { size: 9.5, font: 'mono', align: 'right', color: C.red });
          else V.text(g, 'degenerado (a ≈ 0): P = 1 → sempre pego', x1, ly(1) + 14, { size: 9.5, font: 'mono', align: 'right', color: C.red });
          // saudável
          g.beginPath();
          for (let n = 7; n <= 448; n += 1) { const p = Math.pow(1 - S.a, Math.floor(n / K)); const x = lx(n), y = ly(p); if (n === 7) g.moveTo(x, y); else g.lineTo(x, y); }
          g.strokeStyle = C.e; g.lineWidth = 2.2; g.stroke();
          const pN = Math.pow(1 - S.a, Math.floor(S.N / K));
          V.dot(g, lx(S.N), ly(pN), 5.5, C.b);
          V.glow(g, lx(S.N), ly(pN), 14, C.b, 0.6);
          // blocos de rascunho
          const by = bot + 34;
          const nSteps = Math.floor(S.N / K), show = Math.min(12, nSteps);
          V.text(g, 'passos de verificação: ' + nSteps + (nSteps > 12 ? ' (12 exibidos)' : '') + ' · cada bloco = 7 rascunhos', pad, by, { size: 10, font: 'mono', color: C.muted, maxW: w - 2 * pad });
          const cell = Math.min(7, (w - 2 * pad - 11 * 6) / (12 * K));
          for (let s = 0; s < show; s++) {
            const bx = pad + s * (K * (cell + 1) + 6);
            const acc = S.blocks[s] || 0, lit = el > s * 0.18;
            for (let k = 0; k < K; k++) V.box(g, bx + k * (cell + 1), by + 8, cell, 12, { r: 2, fill: lit && k < acc ? C.e : V.alpha(C.muted, 0.25) });
          }
          const txt = 'N = ' + S.N + ' · a = ' + f(S.a, 2) + ' → P(0 | saudável) = ' + (pN < 1e-4 ? pN.toExponential(1).replace('.', ',') : f(pN, 4)) + (S.N < 64 ? ' · abaixo do mínimo: “não dá para julgar”' : '');
          drawWrap(g, txt, pad, by + 42, w - 2 * pad, 15, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: S.N < 64 ? C.d : C.ink });
        } else if (S.view === 'bat') {
          const cols = narrow ? 2 : 4, gap = 10;
          const cw = (w - 2 * pad - gap * (cols - 1)) / cols, ch = narrow ? 88 : 104;
          let pass = 0;
          checks.forEach((c, i) => {
            const x = pad + (i % cols) * (cw + gap), y = 34 + Math.floor(i / cols) * (ch + gap);
            const started = el > i * 0.45, done = el > i * 0.45 + 0.4;
            const fail = (i === 5 && !S.fix55) || S.t1fails.includes(i);
            if (done && !fail) pass++;
            const col = !done ? C.muted : fail ? C.red : C.e;
            V.box(g, x, y, cw, ch, { r: 11, fill: V.alpha(col, done ? 0.1 : 0.04), stroke: V.alpha(col, done ? 0.8 : 0.3), lw: done ? 1.5 : 1 });
            V.text(g, clip(g, (i + 1) + '. ' + c.n, cw - (done ? 66 : 34), narrow ? 11 : 12, 'mono', 700), x + 10, y + 20, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: C.ink });
            V.text(g, clip(g, c.d, cw - 20, 10.5, 'sans'), x + 10, y + 38, { size: 10.5, color: C.muted });
            if (started && !done) { const a0 = t * 6; g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.arc(x + cw - 18, y + 18, 7, a0, a0 + 4.2); g.stroke(); }
            if (done) {
              const msg = fail ? (i === 5 && !S.fix55 ? c.bad : 'tag DSML malformada → parser não vê a chamada') : c.ok;
              drawWrap(g, msg, x + 10, y + 56, cw - 20, 13, { size: 10, font: 'mono', color: fail ? C.red : C.e });
              badge(g, fail ? 'FAIL' : 'PASS', x + cw - 10, y + 18, fail ? C.red : C.e, { align: 'right', size: 9.5 });
            }
          });
          const allDone = el > 6 * 0.45 + 0.4;
          const ry = 34 + Math.ceil(checks.length / cols) * (ch + gap) + 14;
          const nr = drawWrap(g, allDone ? pass + '/7 checagens · esperado 7/7 (e 8/8 na versão a 32K/131K)' : 'rodando a T=0…', pad, ry, w - 2 * pad, 16, { size: narrow ? 11 : 12.5, font: 'mono', weight: 700, color: allDone ? (pass === 7 ? C.e : C.red) : C.muted });
          if (S.t1 && allDone) drawWrap(g, 'Ilustrativo: amostrar os tokens de sintaxe a T=1 quebra chamadas aleatoriamente — artefato de serviço, não qualidade do modelo. Por isso a bateria roda a T=0.', pad, ry + 4 + nr * 16, w - 2 * pad, 15, { size: 11, color: C.d });
        } else {
          const perRow = narrow ? 20 : 30;
          const cell = Math.min(20, (w - 2 * pad - (perRow - 1) * 3) / perRow);
          let y = 32, idx = 0, ok = 0, doneN = 0;
          const order = [];
          groups.forEach((gr) => { for (let k = 0; k < gr.n; k++) order.push(idx + k); idx += gr.n; });
          const timeOf = (i) => (i < 20 ? i * 0.04 : i < 120 ? 0.8 + Math.floor((i - 20) / 4) * 0.05 : 2.05 + (i - 120) * 0.04);
          idx = 0;
          groups.forEach((gr) => {
            V.text(g, gr.l, pad, y + 10, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
            y += 16;
            for (let k = 0; k < gr.n; k++) {
              const i = idx + k, r = Math.floor(k / perRow), c = k % perRow;
              const x = pad + c * (cell + 3), yy = y + r * (cell + 3);
              const done = el > timeOf(i);
              const fail = S.fails.includes(i);
              if (done) { doneN++; if (!fail) ok++; }
              V.box(g, x, yy, cell, cell, { r: 3, fill: !done ? V.alpha(C.muted, 0.15) : fail ? C.red : V.alpha(C.e, 0.75) });
            }
            y += Math.ceil(gr.n / perRow) * (cell + 3) + 8;
            idx += gr.n;
          });
          const fin = doneN === 145;
          V.text(g, (fin ? ok : ok) + '/145' + (fin ? (ok === 145 ? ' · meta atingida (com zero avisos nos dois ranks)' : ' · antes do #191: 200 OK sem tool_calls ou fora do esquema') : ' …'), pad, Math.min(y + 12, h - 30), { size: narrow ? 11 : 12.5, font: 'mono', weight: 700, color: !fin ? C.muted : ok === 145 ? C.e : C.red, maxW: w - 2 * pad });
          if (fin && ok < 145) V.text(g, 'rótulos diferentes a cada execução (clique em rodar de novo)', pad, Math.min(y + 30, h - 12), { size: 10.5, color: C.muted, maxW: w - 2 * pad });
        }
      });
    })();

    /* =========================================================
       F10 — Qualidade × custo: a fronteira de Pareto
       ========================================================= */
    (() => {
      const fig = root.querySelector('#fig-pareto'); if (!fig) return;
      fitStage(fig, (w) => (w < 640 ? 700 : 470));
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d); // cor do selo GLM·TF
      const pts = [
        { l: 'TR3 K6 (6 bpw)', s: 'K6 (6 bpw)', gb: 254, k: 0.013723, dx: 12, dy: 19, al: 'left', same: false },
        { l: 'FP8 oficial (outra stack)', s: 'FP8 (outra stack)', gb: 328, k: 0.020615, dx: -15, dy: 4, al: 'right', same: false },
        { l: 'EXL3 4 bpw (servido)', s: 'EXL3 4 bpw', gb: 176, k: 0.024555, dx: 12, dy: 18, al: 'left', same: true },
        { l: 'FP8 oficial (stack v44)', s: 'FP8 (v44)', gb: 328, k: 0.024629, dx: -12, dy: -12, al: 'right', same: true },
        { l: 'NVFP4 (stack v44)', s: 'NVFP4 (v44)', gb: 180, k: 0.060535, dx: 12, dy: 4, al: 'left', same: true },
      ];
      const frontOf = (P) => P.filter((p) => !P.some((q) => q !== p && q.gb <= p.gb && q.k <= p.k && (q.gb < p.gb || q.k < p.k))).sort((a, b) => a.gb - b.gb);
      const txtBg = (g, str, x, y, o, C) => { const wd = tw(g, str, o.size, o.font || 'sans', o.weight || 400); const x0 = o.align === 'right' ? x - wd : o.align === 'center' ? x - wd / 2 : x; g.fillStyle = V.alpha(C.bg, 0.85); g.fillRect(x0 - 3, y - o.size + 1, wd + 6, o.size + 4); V.text(g, str, x, y, o); };
      const dec = [
        { r: 'glm', l: 'FP8 denso (dense,kda)', d: 10, dl: '≈ −11 ms/passo (+10%)', q: 'KL proxy 0,002–0,013 · argmax 94–100%', v: 'prov' },
        { r: 'glm', l: 'FP8=all (vs dense,kda)', d: 4.8, dl: 'tok/s +4,3 a +5,2%', q: 'kl_panel NOT-WORSE (ΔNLL −0,004…+0,0005)', v: 'ok' },
        { r: 'glm', l: 'KDA BF16 large-M', d: 12.3, dl: 'prefill 16k/64k −12,1/−12,6%', q: 'NOT-WORSE, na borda do stock · +3,3 GiB/rank', v: 'ok' },
        { r: 'glm', l: 'Thin-decode FAST', d: 8.9, dl: 'A/B/A2 +7,7 a +14,6%', q: '0 flips; o gate absoluto falha no stock', v: 'inc' },
        { r: 'glm', l: 'W8A8 no prefill (#182)', d: 30, dl: 'kernel: 27,21 → 12,57 ms', q: 'KL 7,99× o nulo, 23 flips', v: 'bad', over: true },
        { r: 'qwen', l: 'KV FP8', d: -5.3, dl: '−5,3% (rerun; 1,80× o cache)', q: 'reasoning_check 12/12 = 12/12 · tok/passo ↓ em 8/8', v: 'ok' },
        { r: 'qwen', l: 'Determinismo (opt-in)', d: -4, dl: 'decode e prefill ≈ −4%', q: 'bit-idêntico: 5/5 → 1/5 saídas distintas', v: 'det' },
        { r: 'ds', l: 'SP indexer', d: 5, dl: 'TTFT 128K −4 a −6%', q: 'RULER-lite 8/8 (32K e 131K)', v: 'ok' },
        { r: 'tf', l: 'COMM=roce (vs NCCL)', d: 6.1, dl: 'decode +6% (prosa 44,2 → 46,9)', q: 'hash igual: sem painel, sem faixa de ruído', v: 'bits' },
        { r: 'tf', l: 'Densos q4 (vs fp8)', d: 14.1, dl: 'prosa 38,9 → 44,4 tok/s', q: 'GSM8K/HumanEval = fp8 · #18: 12/48 sem fim de turno', v: 'cav' },
      ];
      const S = { view: 'kld', budget: 200, same: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vista', options: [{ v: 'kld', label: 'Bytes × KLD' }, { v: 'dec', label: 'Velocidade × veredicto' }], value: 'kld', onChange: (v) => { S.view = v; showEl(slB, v === 'kld'); showEl(tgS, v === 'kld'); H.redraw(); } });
      const slB = UI.slider(ctl, { label: 'Orçamento (ilustrativo)', min: 150, max: 350, step: 5, value: 200, fmt: (v) => v + ' GB', onInput: (v) => { S.budget = v; H.redraw(); } });
      const tgS = UI.toggle(ctl, { label: 'só mesma stack', value: false, onChange: (v) => { S.same = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 640, pad = 14;
        if (S.view === 'kld') {
          const x0 = pad + 46, x1 = w - pad - 8, top = 30, bot = h - (narrow ? 124 : 78);
          const P = S.same ? pts.filter((p) => p.same) : pts, front = frontOf(P);
          const X = (gb) => x0 + (gb - 150) / (350 - 150) * (x1 - x0);
          const Y = (k) => bot - (k / 0.07) * (bot - top);
          [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07].forEach((k) => { const y = Y(k); V.line(g, x0, y, x1, y, { color: V.alpha(C.muted, 0.13), lw: 1 }); V.text(g, f(k, 2), x0 - 6, y + 3, { size: 9, font: 'mono', align: 'right', color: C.muted }); });
          [150, 200, 250, 300, 350].forEach((gb) => V.text(g, gb + ' GB', X(gb), bot + 14, { size: 9, font: 'mono', align: 'center', color: C.muted }));
          V.text(g, 'KLD média (nats) ↓ melhor', pad, 16, { size: 10.5, font: 'mono', color: C.muted });
          V.text(g, 'tamanho dos pesos →', x1, bot + 28, { size: 10, font: 'mono', align: 'right', color: C.muted });
          const bx = X(S.budget);
          g.fillStyle = V.alpha(C.red, 0.06); g.fillRect(bx, top, x1 - bx, bot - top);
          V.line(g, bx, top - 6, bx, bot, { color: C.d, lw: 1.5, dash: [5, 3] });
          txtBg(g, 'orçamento ' + S.budget + ' GB', bx + 4, bot - 6, { size: 9.5, font: 'mono', color: C.d }, C);
          // fronteira em escada
          g.beginPath();
          front.forEach((p, i) => { const x = X(p.gb), y = Y(p.k); if (!i) g.moveTo(x, y); else { g.lineTo(x, Y(front[i - 1].k)); g.lineTo(x, y); } });
          g.lineTo(x1, Y(front[front.length - 1].k));
          g.strokeStyle = C.e; g.lineWidth = 2; g.setLineDash([]); g.stroke();
          const fits = P.filter((p) => p.gb <= S.budget);
          const best = fits.length ? fits.reduce((a, b) => (b.k < a.k ? b : a)) : null;
          const labs = [];
          pts.forEach((p) => {
            const inP = P.includes(p), onF = front.includes(p), x = X(p.gb), y = Y(p.k);
            const col = !inP ? V.alpha(C.muted, 0.35) : p === best ? C.d : onF ? C.e : C.muted;
            if (p === best) V.glow(g, x, y, 18, C.d, 0.8);
            V.dot(g, x, y, onF ? 6.5 : 5, col);
            if (inP && !onF) { g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.arc(x, y, 7.5, 0, Math.PI * 2); g.stroke(); }
            if (!p.same) { g.strokeStyle = V.alpha(C.muted, inP ? 0.8 : 0.35); g.lineWidth = 1; g.setLineDash([2, 2]); g.beginPath(); g.arc(x, y, 10.5, 0, Math.PI * 2); g.stroke(); g.setLineDash([]); }
            labs.push([p, inP, onF, x, y]);
          });
          labs.forEach(([p, inP, onF, x, y]) => {
            const fs = narrow ? 9.5 : 11, fw = onF ? 700 : 400;
            const lab = clip(g, (narrow ? p.s : p.l) + ' · ' + f(p.k, 4), narrow ? 170 : 260, fs, 'sans', fw);
            const ty = y + p.dy;
            if (Math.abs(p.dy) > 12) V.line(g, x + (p.al === 'right' ? -4 : 4), y + Math.sign(p.dy) * 6, x + p.dx * 0.7, ty - Math.sign(p.dy) * 9, { color: V.alpha(C.muted, 0.6), lw: 1 });
            txtBg(g, lab, x + p.dx, ty, { size: fs, weight: fw, align: p.al, color: !inP ? V.alpha(C.muted, 0.5) : onF || p === best ? C.ink : C.muted }, C);
          });
          const msg = best ? 'Cabe no orçamento e é o melhor: ' + best.l + ' (KLD ' + f(best.k, 4) + ', ' + best.gb + ' GB)' : 'Nada cabe nesse orçamento.';
          const note = S.same ? 'só EXL3, FP8 (v44) e NVFP4, medidos na mesma stack: o EXL3 domina os outros dois' : 'círculo tracejado = medido noutra stack (outra régua): fronteira indicativa';
          drawWrap(g, msg, pad, h - (narrow ? 76 : 34), w - 2 * pad, 15, { size: narrow ? 11 : 12, font: 'mono', weight: 700, color: best ? C.d : C.red });
          drawWrap(g, note + (narrow ? ' · verde = fronteira; cinza circulado = dominado' : ''), pad, h - (narrow ? 36 : 13), w - 2 * pad, 14, { size: 10, color: C.muted });
        } else {
          const vcol = { ok: C.e, prov: C.d, inc: C.d, bad: C.red, det: C.b, bits: C.a, cav: C.d };
          const vlab = { ok: 'aprovado', prov: 'provisório', inc: 'inconclusivo', bad: 'rejeitado', det: 'reprodutível', bits: 'mesmos bits', cav: 'padrão, com ressalva' };
          const rcol = { glm: C.glm, tf: tfc(C), qwen: C.qwen, ds: C.ds };
          const legL = [['GLM', C.glm], ['GLM·TF', tfc(C)], ['Qwen', C.qwen], ['DeepSeek', C.ds]];
          const legW = legL.reduce((acc, it) => acc + tw(g, it[0], 10.5) + 22, 0);
          const top = narrow ? 52 : 36, rowH = narrow ? (h - top - 12) / dec.length : Math.min(46, (h - top - 10) / dec.length);
          const labW = narrow ? 0 : Math.min(190, w * 0.25), qW = narrow ? 0 : Math.min(270, w * 0.34);
          const ax0 = pad + labW + (narrow ? 0 : 10), ax1 = w - pad - qW - (narrow ? 0 : 14);
          const X = (d) => ax0 + (V.clamp(d, -10, 25) + 10) / 35 * (ax1 - ax0);
          V.text(g, 'Δ velocidade (%) — cada seta na métrica do seu próprio estudo', pad, 16, { size: narrow ? 10 : 11, font: 'mono', color: C.muted, maxW: narrow ? w - 2 * pad : w - 2 * pad - legW - 8 });
          if (!narrow) { let lx = w - pad - legW; legL.forEach(([l, c]) => { V.dot(g, lx + 4, 12, 4, c); V.text(g, l, lx + 12, 16, { size: 10.5, color: C.muted }); lx += tw(g, l, 10.5) + 22; }); }
          else { let lx = pad; legL.forEach(([l, c]) => { V.dot(g, lx + 4, 30, 3.5, c); V.text(g, l, lx + 11, 33.5, { size: 9.5, color: C.muted }); lx += tw(g, l, 9.5) + 20; }); }
          [-10, -5, 0, 5, 10, 15, 20, 25].forEach((d) => { const x = X(d); V.line(g, x, top - 4, x, h - 8, { color: V.alpha(C.muted, d === 0 ? 0.5 : 0.12), lw: 1 }); V.text(g, (d > 0 ? '+' : '') + d, x, top - 8, { size: 9, font: 'mono', align: 'center', color: C.muted }); });
          dec.forEach((dd, i) => {
            const y = top + i * rowH;
            const col = vcol[dd.v];
            if (!narrow) {
              V.dot(g, pad + 5, y + rowH / 2, 4, rcol[dd.r] || C.a);
              V.text(g, clip(g, dd.l, labW - 16, 11.5, 'sans', 600), pad + 14, y + rowH / 2 - 2, { size: 11.5, weight: 600, color: C.ink });
              V.text(g, clip(g, dd.dl, labW - 16, 9.5, 'mono'), pad + 14, y + rowH / 2 + 12, { size: 9.5, font: 'mono', color: C.muted });
              const ay = y + rowH / 2;
              V.arrow(g, X(0), ay, X(dd.d), ay, { color: col, lw: 3, head: 8 });
              if (dd.over) V.text(g, '≫', X(25) - 2, ay - 8, { size: 12, weight: 700, align: 'right', color: col });
              V.text(g, clip(g, vlab[dd.v], qW, 10.5, 'mono', 700), ax1 + 14, y + rowH / 2 - 3, { size: 10.5, font: 'mono', weight: 700, color: col });
              V.text(g, clip(g, dd.q, qW, 9.5, 'sans'), ax1 + 14, y + rowH / 2 + 11, { size: 9.5, color: C.muted });
            } else {
              V.dot(g, pad + 4, y + 9, 3.5, rcol[dd.r] || C.a);
              V.text(g, clip(g, dd.l + ' · ' + vlab[dd.v], w - 2 * pad - 12, 10.5, 'sans', 700), pad + 12, y + 13, { size: 10.5, weight: 700, color: C.ink });
              const ay = y + 26;
              V.arrow(g, X(0), ay, X(dd.d), ay, { color: col, lw: 2.6, head: 7 });
              V.text(g, clip(g, dd.q, w - 2 * pad, 9.5, 'sans'), pad, y + rowH - 8, { size: 9.5, color: C.muted });
            }
          });
        }
      }, { static: true });
      showEl(slB, true);
    })();

    /* ---- ajuste local: controles longos quebram linha em telas estreitas ---- */
    root.querySelectorAll('.la-fig-controls .la-ctl').forEach((el) => {
      el.style.flexWrap = 'wrap'; el.style.maxWidth = '100%'; el.style.minWidth = '0'; el.style.rowGap = '6px';
      const lab = el.querySelector('label'); if (lab) { lab.style.whiteSpace = 'normal'; }
      const sel = el.querySelector('select'); if (sel) { sel.style.maxWidth = '100%'; sel.style.minWidth = '0'; }
    });
    root.querySelectorAll('.la-fig-controls .la-toggle').forEach((el) => { el.style.maxWidth = '100%'; });
  },
});

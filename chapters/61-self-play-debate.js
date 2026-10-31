LA.chapter({
  id: 'self-play-debate',
  part: 13,
  order: 61,
  title: 'Self-play e debate',
  navTitle: 'Self-play & debate',
  subtitle: 'Quando o modelo vira o próprio adversário, juiz e professor: debate para supervisão escalável, assembleias de cópias, SPIN, Self-Rewarding, SPPO, tarefas autopropostas e jogos prover-verifier. E como não se enganar quando quem apita é você mesmo.',
  minutes: 35,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['self-play', 'debate', 'supervisão escalável', 'SPIN', 'Self-Rewarding', 'SPPO', 'equilíbrio de Nash', 'Absolute Zero', 'prover-verifier', 'LLM-as-a-judge'],
  goals: [
    'Separar as três famílias (<b>debate para supervisão</b>, <b>debate multiagente</b> e <b>self-play como treino</b>) e dizer, para cada método, quem joga, quem apita e de onde vem a verdade',
    'Explicar por que o debate deixa um juiz fraco conferir uma afirmação grande (decomposição recursiva, NP × PSPACE) e onde isso quebra (argumentos ofuscados, persuasão ≠ verdade)',
    'Escrever e interpretar as perdas do <b>SPIN</b>, do <b>Self-Rewarding</b> e do <b>SPPO</b> (regressão para \\(\\eta(\\mathbb P - \\tfrac12)\\)) e a recompensa de aprendibilidade do <b>Absolute Zero</b>',
    'Avaliar um laço autorreferente (âncora externa, juiz de outra família, diversidade por habilidade, mesmo orçamento) e montar debatedores e juízes com as três receitas do Módulo 1',
  ],
  prereqs: ['grpo-rlvr', 'dpo', 'constitutional-ai'],
  summary: [
    'Três famílias: <b>debate para supervisão escalável</b> (dois agentes, juiz mais fraco), <b>debate multiagente</b> (cópias que se leem, sem treino) e <b>self-play como treino</b> (o modelo gera currículo, adversário ou recompensa). A pergunta-chave é sempre: quem apita?',
    'Debate (Irving et al., 2018) é um jogo de soma zero; a decomposição recursiva deixa o juiz conferir \\(\\lceil\\log_2 N\\rceil + 1\\) coisas em vez de \\(N-1\\). MNIST: 59,4 → 88,9 % com 6 pixels. Com LLMs e humanos: 84 % contra 74 % da consultoria (Michael), 76 % e 88 % contra 48 % e 60 % (Khan); só vence a pergunta direta com assimetria de informação (Kenton). A ameaça teórica são os argumentos ofuscados.',
    'Debate multiagente (Du et al.): 3 agentes × 2 rodadas levaram aritmética de 67,0 a 81,8 % e GSM8K de 77,0 a 85,0 %, mas compare com self-consistency no mesmo orçamento (Smit et al.). Custo de \\(N\\cdot R\\cdot L\\) tokens; batching e prefix caching barateiam; nunca a \\(T = 0\\).',
    'SPIN é um DPO com chosen = resposta humana, rejected = geração da iteração anterior e referência = iteração anterior; o ponto fixo é \\(p_\\theta = p_{\\text{data}}\\). Zephyr-7B: 58,14 → 63,16 no Open LLM Leaderboard com 50 mil prompts, onde mais SFT estagnava.',
    'Self-Rewarding: o modelo gera prompts e candidatos, julga com rubrica aditiva de 5 pontos e treina DPO no par (maior, menor). AlpacaEval 2.0 de 9,94 % a 20,44 % (Llama 2 70B). Riscos: autopreferência, hacking do próprio juiz e saturação (que o Meta-Rewarding ataca).',
    'SPPO trata preferência como jogo de soma constante, com alvo de Nash e sem Bradley–Terry: regride \\(\\log(\\pi/\\pi_t)\\) para \\(\\eta(\\mathbb P(y\\succ\\pi_t) - \\tfrac12)\\), com \\(\\eta = 10^3\\). Absolute Zero paga o propositor com \\(1-\\bar r\\) só em tarefas às vezes resolvidas, com zero dados curados.',
    'O risco transversal é a recompensa autorreferente (Goodhart): a nota do juiz sobe, a qualidade real faz corcova e a diversidade colapsa. Avalie com âncora externa, juiz de outra família, diversidade por habilidade, mesmo orçamento e mais de uma família de modelos.',
  ],
  quiz: [
    { q: 'No debate recursivo da soma, com \\(N = 64\\) números e um honesto que sabe onde está o erro, quantas conferências o juiz faz?', o: ['64', '63', '7', '32'], a: 2, e: 'São \\(\\log_2 64 = 6\\) somas de consistência (uma por rodada) mais 1 folha conferida contra a entrada. Fazer a soma sozinho custaria 63 adições. O ganho depende de o honesto saber apontar a metade errada: com erro ofuscado, ele chuta e a chance de pegar a mentira cai para \\(2^{-6}\\).' },
    { q: 'No SPIN com perda logística, como fica o par do DPO e quem é a referência?', o: ['chosen = resposta humana do SFT, rejected = resposta gerada pela iteração anterior, referência = iteração anterior', 'chosen = resposta da iteração anterior, rejected = resposta humana, referência = modelo base', 'chosen = resposta com a maior nota do próprio juiz, rejected = a de menor nota, referência = SFT', 'chosen e rejected vêm de humanos, e a referência é um reward model'], a: 0, e: 'O jogador principal aprende a distinguir \\(y\\sim p_{\\text{data}}\\) de \\(y\\sim p_{\\theta_t}\\). A terceira opção descreve o Self-Rewarding. Quando \\(p_\\theta = p_{\\text{data}}\\), as duas respostas ficam intercambiáveis e o jogo para: é o ponto fixo.' },
    { q: 'No Absolute Zero, com \\(G = 8\\) tentativas do resolvedor, qual tarefa dá a MAIOR recompensa ao propositor?', o: ['A que o resolvedor acerta 4 de 8', 'A que o resolvedor acerta 8 de 8', 'A que o resolvedor acerta 0 de 8', 'A que o resolvedor acerta 1 de 8'], a: 3, e: '\\(r_{\\text{propose}} = 1 - \\bar r\\) quando \\(0 &lt; \\bar r &lt; 1\\): 1 de 8 vale 0,875 e 4 de 8 vale 0,5. 0 de 8 e 8 de 8 valem zero, porque não ensinam nada (o grupo inteiro tem a mesma recompensa e a vantagem some). O propositor é pago por achar a borda da habilidade.' },
    { q: 'Três respostas têm preferências cíclicas: A vence B, B vence C e C vence A, todas com probabilidade 0,7. O que acontece?', o: ['O Bradley–Terry ordena A &gt; B &gt; C, e o DPO aprende essa ordem como se fosse transitiva', 'O BT empata as três; o SPPO ainda tem sinal, e a média dos iterados vai ao Nash uniforme', 'Nenhum método aprende nada, porque preferências intransitivas não definem um alvo', 'O SPPO converge para o vértice A, que é o vencedor de Condorcet do ciclo'], a: 1, e: 'Por simetria, o ajuste de máxima verossimilhança do BT empata as três. O SPPO usa \\(\\mathbb P(y\\succ\\pi_t)\\), a vitória contra a mistura atual: se \\(\\pi_t\\) tem A demais, C vence a mistura com mais de 0,5 e sobe. Num ciclo não há vencedor de Condorcet; o Nash é a mistura uniforme (1/3, 1/3, 1/3). Com passo constante, o último iterado gira em espiral; é a média dos iterados que converge para o Nash.' },
    { q: 'Você roda Self-Rewarding e a nota média que o próprio modelo dá às suas respostas sobe a cada iteração. Que conclusão é segura?', o: ['O modelo melhorou, porque a nota do juiz é a métrica de qualidade do laço', 'O modelo piorou, porque toda nota autoatribuída é falsa por construção', 'Basta trocar a rubrica de 5 por 10 pontos para a nota voltar a discriminar', 'Nenhuma ainda: meça fora do laço (benchmark, juiz de outra família, humanos) e a concordância do juiz com rótulos ouro'], a: 3, e: 'Com juiz = modelo, a nota pode subir por autopreferência ou por hacking do próprio juiz (Goodhart), enquanto a qualidade real faz corcova. O Self-Rewarding mostrou ganhos reais porque mediu fora do laço, no AlpacaEval. A segunda opção exagera: o ganho pode ser real, só não dá para saber pela nota interna.' },
    { q: 'Debate multiagente com 3 agentes e 2 rodadas: qual é a comparação justa, e qual erro de configuração derruba o método?', o: ['Comparar com self-consistency de 6 amostras; a \\(T = 0\\), as cópias respondem igual e o debate degenera', 'Comparar com um agente único; rodar a \\(T = 0\\) para que o debate seja reprodutível', 'Comparar com 2 amostras independentes; a temperatura não muda nada no resultado', 'Comparar com um modelo maior; desligar o prefix caching para não misturar os agentes'], a: 0, e: 'São \\(N\\cdot R = 6\\) gerações, então a self-consistency equivalente vota entre 6 amostras independentes (Smit et al. mostraram que o debate nem sempre vence isso). Cópias determinísticas do mesmo modelo dão respostas idênticas na 1ª rodada, e não sobra o que debater. O prefix caching ajuda, se o histórico compartilhado vier na mesma ordem para todos os agentes.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || V.C().c;
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('.', ',');
    const onFill = () => V.C().bg; // olhos/texto sobre preenchimento colorido
    const clamp = V.clamp, lerp = V.lerp;
    const ez = (x) => V.ease.inOut(clamp(x, 0, 1));
    const sig = (x) => 1 / (1 + Math.exp(-x));

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** personagem: quadrado arredondado com olhos e humor (happy, sad, o, sly) e, opcional, chapéu de juiz */
    function buddy(g, x, y, s, col, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.32, fill: V.alpha(col, 0.9), stroke: col, lw: 1.4 });
      const eye = onFill();
      const ex = s * 0.18, ey = y - s * 0.09, er = Math.max(1.3, s * 0.075);
      g.strokeStyle = eye; g.lineWidth = Math.max(1.2, s * 0.06); g.lineCap = 'round';
      if (o.mood === 'sly') {
        g.beginPath(); g.moveTo(x - ex - er * 1.4, ey + 1); g.lineTo(x - ex + er * 1.4, ey - 1); g.moveTo(x + ex - er * 1.4, ey - 1); g.lineTo(x + ex + er * 1.4, ey + 1); g.stroke();
      } else { V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye); }
      g.beginPath();
      if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
      else if (o.mood === 'o') g.arc(x, y + s * 0.17, s * 0.075, 0, Math.PI * 2);
      else if (o.mood === 'sly') { g.moveTo(x - s * 0.14, y + s * 0.15); g.quadraticCurveTo(x + s * 0.02, y + s * 0.24, x + s * 0.17, y + s * 0.09); }
      else if (o.mood === 'flat') { g.moveTo(x - s * 0.12, y + s * 0.17); g.lineTo(x + s * 0.12, y + s * 0.17); }
      else g.arc(x, y + s * 0.05, s * 0.16, Math.PI * 0.2, Math.PI * 0.8);
      g.stroke(); g.lineCap = 'butt';
      if (o.hat) { // capelo de juiz
        const hc = o.hatCol || C.d;
        g.fillStyle = hc;
        g.beginPath(); g.moveTo(x - s * 0.42, y - s * 0.5); g.lineTo(x, y - s * 0.72); g.lineTo(x + s * 0.42, y - s * 0.5); g.lineTo(x, y - s * 0.36); g.closePath(); g.fill();
        g.fillRect(x - s * 0.18, y - s * 0.52, s * 0.36, s * 0.12);
        V.line(g, x + s * 0.3, y - s * 0.52, x + s * 0.34, y - s * 0.3, { color: hc, lw: 1.4 });
      }
      if (o.alpha != null) g.globalAlpha = 1;
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.45), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
    }
    function mark(g, x, y, ok, s = 6) {
      const C = V.C();
      g.lineWidth = 2.3; g.strokeStyle = ok ? C.a : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    /** ponto numa curva quadrática */
    const qpt = (x1, y1, cx, cy, x2, y2, u) => { const a = 1 - u; return [a * a * x1 + 2 * a * u * cx + u * u * x2, a * a * y1 + 2 * a * u * cy + u * u * y2]; };
    function curveCtl(x1, y1, x2, y2, bend) { const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1; return [mx - (dy / L) * bend, my + (dx / L) * bend]; }
    function pill(g, x, y, label, col, size = 10.5) {
      const C = V.C();
      const w = tw(g, label, size, 'mono', 500) + 10, h = size + 7;
      V.box(g, x - w / 2, y - h / 2, w, h, { r: h / 2, fill: V.alpha(C.bg, 0.88), stroke: V.alpha(col || C.muted, 0.6), lw: 1 });
      V.text(g, label, x, y + 0.5, { size, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
    }
    function docIcon(g, x, y, s, col) {
      const C = V.C();
      for (let k = 2; k >= 0; k--) {
        const ox = x - s * 0.36 + k * 3, oy = y - s * 0.46 - k * 3;
        V.box(g, ox, oy, s * 0.72, s * 0.92, { r: 3, fill: k ? V.alpha(col, 0.25) : V.alpha(C.bg, 0.95), stroke: col, lw: 1.2 });
      }
      for (let l = 0; l < 4; l++) V.line(g, x - s * 0.24, y - s * 0.26 + l * s * 0.17, x + s * 0.2 - (l === 3 ? s * 0.16 : 0), y - s * 0.26 + l * s * 0.17, { color: V.alpha(col, 0.8), lw: 1.4 });
    }
    function axisNum(g, s, x, y, al = 'right') { V.text(g, s, x, y, { size: 11, font: 'mono', color: V.C().muted, align: al, base: 'middle' }); }
    function wrapC(g, s, x, y, maxW, o = {}) { return V.wrap(g, s, x, y, maxW, Object.assign({ size: 12, lineH: 16, color: V.C().ink }, o)); }

    /* ================================================ FIG 1 — o elenco */
    (function () {
      const fig = root.querySelector('#fig-elenco');
      adaptH(fig, 430, 620);
      const ROLE = { pi: (C) => C.a, old: (C) => C.b, judge: (C) => C.d, human: (C) => C.c, exec: (C) => C.e, adv: (C) => C.f, muted: (C) => C.muted };
      const M = {
        debate: {
          nodes: [
            { id: 'q', x: 0.5, y: 0.12, kind: 'doc', role: 'human', label: 'pergunta q' },
            { id: 'A', x: 0.15, y: 0.45, kind: 'buddy', role: 'pi', label: 'debatedor A' },
            { id: 'B', x: 0.85, y: 0.45, kind: 'buddy', role: 'pi', label: 'debatedor B', mood: 'sly' },
            { id: 'J', x: 0.5, y: 0.8, kind: 'buddy', role: 'judge', label: 'juiz (mais fraco)', hat: true, mood: 'flat' },
          ],
          edges: [['q', 'A', ''], ['q', 'B', ''], ['A', 'B', 'argumento', 26], ['B', 'A', 'réplica', 26], ['A', 'J', 'transcrição τ', 0, false, false, 0.42], ['B', 'J', ''], ['J', 'A', '±1', 30, true, false, 0.22], ['J', 'B', '∓1', -30, true, false, 0.22]],
          info: ['duas cópias da mesma política, em lados opostos', 'um juiz mais fraco (humano, no artigo de 2018)', 'sim: self-play de soma zero (proposta)', 'a aposta de que refutar uma mentira é mais fácil que sustentá-la'],
        },
        mad: {
          nodes: [
            { id: 'q', x: 0.5, y: 0.1, kind: 'doc', role: 'human', label: 'pergunta' },
            { id: 'A', x: 0.17, y: 0.42, kind: 'buddy', role: 'pi', label: 'agente 1' },
            { id: 'B', x: 0.83, y: 0.42, kind: 'buddy', role: 'pi', label: 'agente 2' },
            { id: 'D', x: 0.5, y: 0.56, kind: 'buddy', role: 'pi', label: 'agente 3' },
            { id: 'V', x: 0.5, y: 0.9, kind: 'box', role: 'muted', label: 'consenso / maioria' },
          ],
          edges: [['q', 'A', ''], ['q', 'B', ''], ['q', 'D', ''], ['A', 'B', 'lê', -30], ['B', 'D', 'lê', -10], ['D', 'A', 'lê', -10], ['A', 'V', ''], ['B', 'V', ''], ['D', 'V', 'rodada R']],
          info: ['N cópias do mesmo LLM, em paralelo', 'ninguém: vale o consenso (ou a maioria) depois de R rodadas', 'não: só tempo de inferência', 'erros pouco correlacionados e raciocínios que se deixam conferir quando lidos'],
        },
        spin: {
          nodes: [
            { id: 'D', x: 0.14, y: 0.2, kind: 'doc', role: 'human', label: 'dados de SFT (humanos)' },
            { id: 'O', x: 0.14, y: 0.78, kind: 'buddy', role: 'old', label: 'π_t (oponente)', mood: 'flat' },
            { id: 'P', x: 0.56, y: 0.5, kind: 'buddy', role: 'pi', label: 'π_t+1 (jogador)' },
            { id: 'L', x: 0.87, y: 0.22, kind: 'box', role: 'pi', label: 'λ·log-ratio' },
          ],
          edges: [['D', 'P', 'y humana = chosen'], ['O', 'P', 'y′ própria = rejected', 0, false, false, 0.42], ['P', 'L', 'separa?', -34], ['L', 'P', '∇: sobe y, desce y′', -34], ['P', 'O', 'vira o oponente', -70, true]],
          info: ['o modelo atual contra a própria versão anterior', 'o próprio log-ratio: um discriminador implícito', 'sim: um DPO por iteração, sem rótulo novo', 'as demonstrações humanas do SFT (o teto é p_data)'],
        },
        selfreward: {
          nodes: [
            { id: 'M', x: 0.12, y: 0.42, kind: 'buddy', role: 'pi', label: 'M_t' },
            { id: 'P', x: 0.42, y: 0.13, kind: 'doc', role: 'pi', label: 'prompts novos' },
            { id: 'C', x: 0.8, y: 0.18, kind: 'cands', role: 'pi', label: '4 candidatos' },
            { id: 'J', x: 0.84, y: 0.64, kind: 'buddy', role: 'pi', label: 'M_t de toga', hat: true, mood: 'flat' },
            { id: 'Q', x: 0.5, y: 0.86, kind: 'pair', role: 'pi', label: 'par (5, 2)' },
            { id: 'N', x: 0.15, y: 0.86, kind: 'buddy', role: 'pi', label: 'M_t+1' },
          ],
          edges: [['M', 'P', 'escreve'], ['P', 'C', 'gera N'], ['C', 'J', 'rubrica 0–5'], ['J', 'Q', 'maior × menor'], ['Q', 'N', 'DPO'], ['N', 'M', 'repete', -26, true]],
          info: ['o modelo gera os prompts e as respostas', 'o mesmo modelo, com rubrica aditiva de 5 pontos', 'sim: DPO iterativo (M1 → M2 → M3)', 'só o próprio julgamento, que melhora junto e também tem vieses'],
        },
        sppo: {
          nodes: [
            { id: 'P', x: 0.12, y: 0.48, kind: 'buddy', role: 'old', label: 'π_t', mood: 'flat' },
            { id: 'y1', x: 0.42, y: 0.16, kind: 'chip', role: 'old', label: 'y₁' },
            { id: 'y2', x: 0.42, y: 0.48, kind: 'chip', role: 'old', label: 'y₂' },
            { id: 'y3', x: 0.42, y: 0.8, kind: 'chip', role: 'old', label: 'y₃' },
            { id: 'R', x: 0.73, y: 0.32, kind: 'box', role: 'judge', label: 'PairRM 0,4B' },
            { id: 'N', x: 0.86, y: 0.8, kind: 'buddy', role: 'pi', label: 'π_t+1 (vira π_t)' },
          ],
          edges: [['P', 'y1', ''], ['P', 'y2', 'amostras'], ['P', 'y3', ''], ['y1', 'R', ''], ['y2', 'R', 'confrontos'], ['y3', 'R', ''], ['R', 'N', 'alvo η(P − ½)']],
          info: ['a política contra a mistura atual de si mesma', 'um modelo de preferência par a par (PairRM, 0,4B)', 'sim: regressão do log-ratio, 3 iterações', 'as preferências do PairRM, sem supor notas transitivas'],
        },
        azr: {
          nodes: [
            { id: 'Pp', x: 0.14, y: 0.42, kind: 'buddy', role: 'pi', label: 'propositor' },
            { id: 'E', x: 0.5, y: 0.13, kind: 'box', role: 'exec', label: 'executor Python' },
            { id: 'Ps', x: 0.86, y: 0.42, kind: 'buddy', role: 'pi', label: 'resolvedor', mood: 'o' },
            { id: 'R', x: 0.5, y: 0.84, kind: 'box', role: 'muted', label: 'recompensas' },
          ],
          edges: [['Pp', 'E', 'tarefa'], ['E', 'Ps', 'tarefa válida', 0, false, false, 0.6], ['Ps', 'E', 'resposta', 64], ['E', 'R', 'r_solve', 0, false, false, 0.74], ['R', 'Pp', 'r_propose = 1 − r̄'], ['R', 'Ps', 'r_solve'], ['Pp', 'Ps', 'mesmos pesos', 0, true, true, 0.27]],
          info: ['o mesmo modelo em dois papéis: propor e resolver', 'um executor de Python (verificação por regra)', 'sim: RL com 6 linhas de base (TRR++)', 'a execução do código, sem nenhum dado curado'],
        },
        pvg: {
          nodes: [
            { id: 'H', x: 0.13, y: 0.24, kind: 'buddy', role: 'pi', label: 'provador útil' },
            { id: 'S', x: 0.13, y: 0.76, kind: 'buddy', role: 'adv', label: 'provador sorrateiro', mood: 'sly' },
            { id: 'Vf', x: 0.58, y: 0.5, kind: 'buddy', role: 'judge', label: 'verificador pequeno', hat: true, mood: 'flat', small: true },
            { id: 'G', x: 0.88, y: 0.84, kind: 'box', role: 'exec', label: 'gabarito' },
            { id: 'Hu', x: 0.88, y: 0.16, kind: 'buddy', role: 'human', label: 'humano com pressa', mood: 'o' },
          ],
          edges: [['H', 'Vf', 'solução certa', 0, false, false, 0.42], ['S', 'Vf', 'errada e caprichada', 0, false, false, 0.42], ['G', 'Vf', 'correção'], ['Vf', 'H', 'aceita', 60, true, false, 0.4], ['Vf', 'S', 'pegou', -60, true, false, 0.4], ['H', 'Hu', 'legível', 0, true]],
          info: ['provadores úteis e sorrateiros contra um verificador pequeno', 'o verificador, treinado com o gabarito de correção', 'sim: rodadas alternadas de verificador e provadores', 'o gabarito de correção; e a legibilidade transfere para humanos'],
        },
      };
      const S = { m: 'debate', t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Método', options: [{ v: 'debate', label: 'Debate' }, { v: 'mad', label: 'Multiagente' }, { v: 'spin', label: 'SPIN' }, { v: 'selfreward', label: 'Self-Rewarding' }, { v: 'sppo', label: 'SPPO' }, { v: 'azr', label: 'Absolute Zero' }, { v: 'pvg', label: 'Prover-verifier' }], value: 'debate', onChange: (v) => { S.m = v; S.reset = true; } });
      const KEYS = ['Quem joga', 'Quem apita', 'Treina?', 'A verdade vem de'];

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.reset) { S.t0 = t; S.reset = false; }
        const tt = t - S.t0;
        const narrow = w < 560;
        const meth = M[S.m];
        const infoH = narrow ? 222 : 104;
        const gx = 16, gy = 14, gw = w - 32, gh = h - infoH - 24;
        const s = clamp(Math.min(gw, gh) * 0.12, 26, 40);
        const P = {};
        meth.nodes.forEach((n) => { P[n.id] = { x: gx + n.x * gw, y: gy + 8 + n.y * (gh - 16), n }; });
        const rad = (n) => (n.kind === 'buddy' ? s * 0.62 : n.kind === 'doc' ? s * 0.55 : n.kind === 'box' ? 18 : n.kind === 'cands' || n.kind === 'pair' ? 20 : 14);
        // arestas
        const nE = meth.edges.filter((e) => !e[5]).length;
        const per = 1.1;
        let flowIdx = 0;
        const cur = Math.floor(tt / per) % Math.max(1, nE);
        meth.edges.forEach((e) => {
          const a = P[e[0]], b = P[e[1]];
          const bend = e[3] || 0, dashed = !!e[4], still = !!e[5];
          const [cx, cy] = curveCtl(a.x, a.y, b.x, b.y, bend);
          const L = Math.hypot(b.x - a.x, b.y - a.y) || 1;
          const u0 = Math.min(0.4, rad(a.n) / L), u1 = 1 - Math.min(0.4, (rad(b.n) + 4) / L);
          const p0 = qpt(a.x, a.y, cx, cy, b.x, b.y, u0), p1 = qpt(a.x, a.y, cx, cy, b.x, b.y, u1);
          const idx = still ? -1 : flowIdx++;
          const isCur = idx === cur;
          const col = still ? V.alpha(C.muted, 0.6) : isCur ? C.d : V.alpha(C.muted, 0.55);
          // traço
          g.beginPath();
          for (let k = 0; k <= 24; k++) { const [x, y] = qpt(a.x, a.y, cx, cy, b.x, b.y, lerp(u0, u1, k / 24)); if (k) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = col; g.lineWidth = isCur ? 2.4 : 1.4;
          if (dashed) g.setLineDash([5, 4]);
          g.stroke(); g.setLineDash([]);
          if (!still) {
            const [hx, hy] = qpt(a.x, a.y, cx, cy, b.x, b.y, u1 - 0.02);
            V.arrow(g, hx, hy, p1[0], p1[1], { color: col, head: isCur ? 9 : 7, lw: 0.1 });
            // partículas
            const nP = isCur ? 1 : 2;
            for (let q = 0; q < nP; q++) {
              const u = isCur ? (tt % per) / per : ((tt * 0.35 + q * 0.5 + idx * 0.13) % 1);
              const [x, y] = qpt(a.x, a.y, cx, cy, b.x, b.y, lerp(u0, u1, u));
              if (isCur) V.glow(g, x, y, 11, C.d, 0.9);
              V.dot(g, x, y, isCur ? 3.6 : 2.2, isCur ? C.d : V.alpha(C.muted, 0.7));
            }
          }
          if (e[2] && (!narrow || isCur || still)) { const [lx, ly] = qpt(a.x, a.y, cx, cy, b.x, b.y, e[6] != null ? e[6] : 0.5); pill(g, lx, ly, e[2], isCur ? C.d : C.muted, narrow ? 10.5 : 11); }
        });
        // nós
        meth.nodes.forEach((n, k) => {
          const p = P[n.id], col = ROLE[n.role](C);
          const bob = Math.sin(t * 2.2 + k * 1.3) * 1.6;
          if (n.kind === 'buddy') buddy(g, p.x, p.y + bob, n.small ? s * 0.82 : s, col, { mood: n.mood, hat: n.hat, hatCol: C.d });
          else if (n.kind === 'doc') docIcon(g, p.x, p.y + bob, s, col);
          else if (n.kind === 'box') { const bw = tw(g, n.label, 11.5, 'mono', 600) + 18; chip(g, p.x - bw / 2, p.y - 15, bw, 30, n.label, { fill: V.alpha(col, 0.18), stroke: col, size: 11.5, weight: 600 }); }
          else if (n.kind === 'chip') chip(g, p.x - 16, p.y - 12, 32, 24, n.label, { fill: V.alpha(col, 0.2), stroke: col, size: 12, weight: 600 });
          else if (n.kind === 'cands') { ['A', 'B', 'C', 'D'].forEach((l, i) => chip(g, p.x - 46 + i * 23, p.y - 11 + Math.sin(t * 3 + i) * 1.5, 20, 22, l, { fill: V.alpha(col, 0.2), stroke: col, size: 11 })); }
          else if (n.kind === 'pair') { chip(g, p.x - 40, p.y - 12, 36, 24, '5 ✓', { fill: V.alpha(C.a, 0.22), stroke: C.a, size: 11.5, weight: 700 }); chip(g, p.x + 4, p.y - 12, 36, 24, '2 ✗', { fill: V.alpha(RED(), 0.16), stroke: RED(), size: 11.5, weight: 700 }); }
          if (n.kind !== 'box' && n.kind !== 'chip') {
            const ly = n.kind === 'buddy' ? p.y + s * (n.small ? 0.42 : 0.5) + 14 : n.kind === 'doc' ? p.y + s * 0.5 + 19 : p.y + 26;
            const lw = Math.min(tw(g, n.label, 11.5, 'mono', 600), narrow ? 150 : 170);
            V.text(g, n.label, clamp(p.x, 6 + lw / 2, w - 6 - lw / 2), ly, { size: 11.5, font: 'mono', weight: 600, color: C.ink, align: 'center', maxW: narrow ? 150 : 170 });
          }
        });
        // painel de informação
        const iy = h - infoH - 6;
        V.box(g, 10, iy, w - 20, infoH, { r: 10, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        const cols = narrow ? 1 : 2, cw = (w - 40) / cols;
        meth.info.forEach((txt, i) => {
          const c = narrow ? 0 : i % 2, r = narrow ? i : Math.floor(i / 2);
          const x = 22 + c * cw, y = iy + 20 + r * (narrow ? 52 : 44);
          V.text(g, KEYS[i], x, y, { size: 11, font: 'mono', weight: 700, color: [C.a, C.d, C.b, C.c][i] });
          wrapC(g, txt, x, y + 16, cw - 16, { size: 12, lineH: 15, maxLines: 2 });
        });
      });
    })();

    /* ================================================ FIG 2 — árvore do debate */
    (function () {
      const fig = root.querySelector('#fig-arvore');
      adaptH(fig, 440, 520);
      const S = { N: 16, obf: false, runs: 0, caught: 0, seed: 7 };
      let R = null;
      const T = 1.5;
      function newRound(t) {
        const rng = V.rng(S.seed++ * 97 + S.N);
        const N = S.N, L = Math.round(Math.log2(N));
        const vals = Array.from({ length: N }, () => 1 + rng.int(9));
        const err = rng.int(N);
        const delta = rng.pick([3, 4, 5, -3, -4]);
        const pre = [0]; vals.forEach((v) => pre.push(pre[pre.length - 1] + v));
        const sumR = (a, b) => pre[b] - pre[a];
        const claim = (d, i) => { const span = N >> d, a = i * span, b = a + span; return sumR(a, b) + (err >= a && err < b ? delta : 0); };
        const path = [0];
        let node = 0;
        for (let d = 0; d < L; d++) {
          const span = N >> (d + 1);
          const left = node * 2, right = left + 1;
          const errChild = err < (left + 1) * span && err >= left * span ? left : right;
          const pick = S.obf ? (rng() < 0.5 ? left : right) : errChild;
          path.push(pick); node = pick;
        }
        R = { vals, err, delta, claim, sumR, path, L, t0: t, done: false, total: sumR(0, N) };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'N números', options: [{ v: 8, label: '8' }, { v: 16, label: '16' }, { v: 32, label: '32' }], value: 16, onChange: (v) => { S.N = v; S.runs = 0; S.caught = 0; R = null; upd(); } });
      UI.toggle(ctl, { label: 'erro ofuscado (o honesto chuta)', value: false, onChange: (v) => { S.obf = v; S.runs = 0; S.caught = 0; R = null; upd(); } });
      UI.button(ctl, '↻ nova mentira', () => { R = null; });
      const rJ = UI.readout(ctl, 'conferências do juiz', '—'), rS = UI.readout(ctl, 'somas, sozinho', '—'), rP = UI.readout(ctl, 'P(pegar a mentira)', '—'), rK = UI.readout(ctl, 'placar', '—');
      function upd() {
        const L = Math.round(Math.log2(S.N));
        rJ.set(String(L + 1)); rS.set(String(S.N - 1));
        rP.set(S.obf ? fmt(100 * Math.pow(2, -L), L > 4 ? 1 : 1) + ' %' : '100 %', S.obf ? 'bad' : 'ok');
        rK.set(S.runs ? `${S.caught}/${S.runs} pegas` : '—');
      }
      upd();

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (!R) newRound(t);
        const N = S.N, L = R.L;
        const el = t - R.t0;
        const stepD = Math.floor(el / T); // nível em disputa
        const fr = (el % T) / T;
        const narrow = w < 560;
        // cabeçalho com os debatedores
        const hy = 30, bs = narrow ? 26 : 30;
        buddy(g, 26, hy, bs, C.f, { mood: 'sly' });
        buddy(g, w - 26, hy, bs, C.a, { mood: el > L * T + 0.8 && R.verdict === true ? 'happy' : 'flat' });
        const lieTxt = `total = ${R.claim(0, 0)}`, honTxt = `total = ${R.total}`;
        chip(g, 46, hy - 13, tw(g, lieTxt, 12, 'mono', 700) + 16, 26, lieTxt, { fill: V.alpha(C.f, 0.16), stroke: C.f, size: 12, weight: 700 });
        const hw = tw(g, honTxt, 12, 'mono', 700) + 16;
        chip(g, w - 46 - hw, hy - 13, hw, 26, honTxt, { fill: V.alpha(C.a, 0.16), stroke: C.a, size: 12, weight: 700 });
        if (!narrow) {
          V.text(g, 'mentiroso', 46, hy + 26, { size: 11, font: 'mono', color: C.f });
          V.text(g, 'honesto', w - 46, hy + 26, { size: 11, font: 'mono', color: C.a, align: 'right' });
        }
        // geometria da árvore
        const top = 104, arrH = 36;
        const ay = h - arrH - 30;
        const lvlY = (d) => (d === L ? ay + 11 : top + (d / L) * (ay + 11 - top));
        const pad = 14, W = w - 2 * pad;
        const leafPicked = stepD > L - 1 || (stepD === L - 1 && fr > 0.55);
        const nodeX = (d, i) => pad + (i + 0.5) * (W / (1 << d));
        const revealed = (d, i) => {
          if (d === 0) return true;
          const parent = i >> 1;
          return R.path[d - 1] === parent && stepD >= d - 1 && (stepD > d - 1 || fr > 0.18);
        };
        // arestas
        for (let d = 0; d < L; d++) for (let i = 0; i < 1 << d; i++) {
          for (const c of [i * 2, i * 2 + 1]) {
            const onPath = R.path[d] === i && R.path[d + 1] === c && (stepD > d || (stepD === d && fr > 0.55));
            V.line(g, nodeX(d, i), lvlY(d) + 11, nodeX(d + 1, c), lvlY(d + 1) - 11, { color: onPath ? C.a : V.alpha(C.muted, 0.22), lw: onPath ? 2.4 : 1 });
          }
        }
        // nós
        for (let d = 0; d <= L; d++) {
          const n = 1 << d, slot = W / n;
          const bw = Math.min(54, slot - 3), bh = 22;
          for (let i = 0; i < n; i++) {
            const x = nodeX(d, i), y = lvlY(d);
            if (d === L) continue; // folhas: a fileira de entrada
            const rev = revealed(d, i);
            const onPath = R.path[d] === i && (d === 0 || stepD >= d);
            const fill = rev ? V.alpha(C.f, onPath ? 0.28 : 0.14) : V.alpha(C.muted, 0.06);
            V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 5, fill, stroke: rev ? (onPath ? C.f : V.alpha(C.f, 0.5)) : V.alpha(C.muted, 0.25), lw: onPath ? 1.8 : 1 });
            if (rev && bw >= 14) V.text(g, String(R.claim(d, i)), x, y + 0.5, { size: bw < 26 ? 11 : 11.5, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
          }
        }
        // conferência de consistência do juiz e o dedo do honesto
        for (let d = 0; d < Math.min(stepD + 1, L); d++) {
          if (d === stepD && fr < 0.18) continue;
          const par = R.path[d], l = par * 2, r = l + 1;
          const y = lvlY(d);
          if (d === stepD || (d === L - 1 && stepD >= L)) {
            const s1 = R.claim(d + 1, l), s2 = R.claim(d + 1, r);
            const txt = `${s1} + ${s2} = ${R.claim(d, par)} ✓`;
            const x = nodeX(d, par), tx = clamp(x, 70, w - 70);
            pill(g, tx, y - 24, txt, C.d, 11);
          }
          if (d < stepD || fr > 0.55) {
            const ch = R.path[d + 1];
            const cx = nodeX(d + 1, ch), cy = lvlY(d + 1);
            if (d + 1 < L) { g.strokeStyle = C.a; g.lineWidth = 2.2; const bw = Math.min(54, W / (1 << (d + 1)) - 3); V.rr(g, cx - bw / 2 - 3, cy - 14, bw + 6, 28, 7); g.stroke(); }
            if (d === stepD && d + 1 < L) { const say = S.obf ? '“acho que é esta”' : '“esta metade está errada”', hw2 = tw(g, say, 11, 'mono') / 2 + 6; V.text(g, say, clamp(cx, hw2, w - hw2), cy + 26, { size: 11, font: 'mono', color: C.a, align: 'center' }); }
          }
        }
        // fileira de entrada (folhas)
        const cw = W / N;
        const leafIdx = R.path[L];
        const final = stepD >= L;
        for (let i = 0; i < N; i++) {
          const x = pad + i * cw;
          const isLeaf = leafPicked && i === leafIdx;
          V.box(g, x + 1, ay, cw - 2, arrH, { r: 4, fill: isLeaf ? V.alpha(C.d, 0.3) : V.alpha(C.muted, 0.1), stroke: isLeaf ? C.d : V.alpha(C.muted, 0.35), lw: isLeaf ? 2 : 1 });
          if (cw >= 9) V.text(g, String(R.vals[i]), x + cw / 2, ay + arrH / 2 + 0.5, { size: cw < 22 ? 11 : 12.5, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
        }
        V.text(g, 'entrada (todos podem ler; o juiz só lê 1 casa)', pad, ay + arrH + 16, { size: 11, font: 'mono', color: C.muted, maxW: W });
        // folha final: o juiz confere
        if (final) {
          const lc = R.claim(L, leafIdx), real = R.vals[leafIdx];
          const x = pad + leafIdx * cw + cw / 2;
          const caught = lc !== real;
          if (R.verdict == null) { R.verdict = caught; S.runs++; if (caught) S.caught++; upd(); }
          const k = ez((el - L * T) / 0.6);
          // lupa do juiz sobre a casa conferida
          g.strokeStyle = V.alpha(C.d, k); g.lineWidth = 2.4;
          V.rr(g, x - cw / 2 - 2, ay - 4, cw + 4, arrH + 8, 6); g.stroke();
          buddy(g, clamp(x + cw / 2 + 12, 16, w - 16), ay - 16 - 4 * k, 22, C.d, { hat: true, mood: caught ? 'happy' : 'o', alpha: k });
          const txt = caught ? `afirmou ${lc}, a entrada tem ${real}: mentira pega!` : `afirmou ${lc} = entrada ${real}: o juiz é enganado`;
          if (el > L * T + 0.6) {
            const bw2 = Math.min(w - 20, tw(g, txt, 12, 'sans', 700) + 22);
            const bx = (w - bw2) / 2;
            V.box(g, bx, 62, bw2, 26, { r: 8, fill: V.alpha(caught ? C.a : RED(), 0.2), stroke: caught ? C.a : RED(), lw: 1.4 });
            V.text(g, txt, bx + bw2 / 2, 75, { size: 12, weight: 700, color: C.ink, align: 'center', base: 'middle', maxW: bw2 - 10 });
          }
        } else V.text(g, `rodada ${Math.min(stepD + 1, L)} de ${L}`, w / 2, 64, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
        if (el > L * T + 3.4) R = null;
      });
    })();

    /* ================================================ FIG 3 — resultados publicados */
    (function () {
      const fig = root.querySelector('#fig-resultados');
      adaptH(fig, 390, 470);
      const D = {
        irving: { title: 'Irving et al. 2018 · MNIST, juiz = classificador esparso', groups: ['6 pixels', '4 pixels'], series: ['juiz sem debate', 'com debate'], vals: [[59.4, 88.9], [48.2, 85.2]] },
        khan: { title: 'Khan et al. 2024 · leitura difícil, especialistas LLM', groups: ['juiz LLM', 'juiz humano'], series: ['linha de base ingênua', 'debate'], vals: [[48, 76], [60, 88]] },
        michael: { title: 'Michael et al. 2023 · QuALITY, debatedores humanos', groups: ['juízes humanos'], series: ['consultoria', 'debate'], vals: [[74, 84]] },
      };
      const S = { k: 'irving', t0: -10 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Estudo', options: [{ v: 'irving', label: 'Irving (MNIST)' }, { v: 'khan', label: 'Khan (LLMs)' }, { v: 'michael', label: 'Michael (humanos)' }], value: 'irving', onChange: (v) => { S.k = v; S.reset = true; } });
      const DIG = ['0011100', '0100010', '0000010', '0000100', '0001000', '0010000', '0010000']; // um “7” (ilustrativo)
      const REV = [[0, 2], [0, 4], [3, 4], [5, 2], [1, 1], [6, 2]];
      function speech(g, x, y, w, col, t, k) {
        const C = V.C();
        V.box(g, x, y, w, 26, { r: 8, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.7), lw: 1 });
        for (let l = 0; l < 2; l++) { const ww = (w - 16) * (l ? 0.55 : 0.85) * clamp(k * 2 - l, 0, 1); if (ww > 0) V.line(g, x + 8, y + 9 + l * 8, x + 8 + ww, y + 9 + l * 8, { color: V.alpha(col, 0.9), lw: 2.2 }); }
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.reset) { S.t0 = t; S.reset = false; }
        const d = D[S.k];
        const el = t - S.t0;
        const narrow = w < 560;
        const topH = narrow ? 150 : 132;
        V.text(g, d.title, 16, 20, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: w - 32 });
        // miniatura do protocolo
        const my = 34, mh = topH - 40, cx = w / 2;
        const ph = (t % 6) / 6;
        if (S.k === 'irving') {
          const cs = Math.min(13, (mh - 8) / 7), gw = cs * 7;
          const gx = cx - gw / 2, gy = my + 4;
          const nRev = 2 + Math.floor(ph * 6);
          for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
            const ri = REV.findIndex((p) => p[0] === i && p[1] === j);
            const shown = ri >= 0 && ri < nRev;
            const on = DIG[i][j] === '1';
            g.fillStyle = shown ? (on ? C.ink : V.alpha(C.ink, 0.12)) : V.alpha(C.muted, 0.2);
            g.fillRect(gx + j * cs, gy + i * cs, cs - 1, cs - 1);
            if (shown) { g.strokeStyle = ri % 2 ? C.f : C.a; g.lineWidth = 2; g.strokeRect(gx + j * cs - 0.5, gy + i * cs - 0.5, cs, cs); }
          }
          buddy(g, gx - 50, gy + gw / 2, 28, C.a, { mood: 'happy' });
          buddy(g, gx + gw + 50, gy + gw / 2, 28, C.f, { mood: 'sly' });
          if (!narrow) { V.text(g, '“é um 7”', gx - 50, gy + gw / 2 + 30, { size: 11, font: 'mono', color: C.a, align: 'center' }); V.text(g, '“é um 1”', gx + gw + 50, gy + gw / 2 + 30, { size: 11, font: 'mono', color: C.f, align: 'center' }); }
          V.text(g, `pixels revelados: ${Math.min(6, nRev)}`, cx, gy + gw + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        } else if (S.k === 'khan') {
          const bw = Math.min(150, (w - 140) / 2);
          buddy(g, cx - bw - 40, my + 26, 28, C.a, { mood: 'happy' });
          buddy(g, cx + bw + 40, my + 26, 28, C.f, { mood: 'sly' });
          const turn = Math.floor(ph * 4);
          for (let k = 0; k < 4; k++) {
            if (k > turn) break;
            const left = k % 2 === 0;
            const kk = k === turn ? (ph * 4) % 1 : 1;
            speech(g, left ? cx - bw - 18 : cx + 18, my + 2 + Math.floor(k / 2) * 30, bw, left ? C.a : C.f, t, kk);
          }
          buddy(g, cx, my + mh - 12, 26, C.d, { hat: true, mood: 'flat' });
          V.text(g, 'o juiz não leu o texto', cx + 26, my + mh - 8, { size: 11, font: 'mono', color: C.muted });
        } else {
          const half = w / 2;
          [0, 1].forEach((side) => {
            const ox = side ? half : 0, ccx = ox + half / 2;
            V.text(g, side ? 'debate' : 'consultoria', ccx, my + 6, { size: 11.5, font: 'mono', weight: 700, color: side ? C.a : C.muted, align: 'center' });
            if (side) { buddy(g, ccx - 44, my + 30, 24, C.a, { mood: 'happy' }); buddy(g, ccx + 44, my + 30, 24, C.f, { mood: 'sly' }); speech(g, ccx - 34, my + 48, 68, ph < 0.5 ? C.a : C.f, t, (ph * 2) % 1); }
            else { buddy(g, ccx, my + 30, 24, (Math.floor(t / 6) % 2) ? C.f : C.a, { mood: 'flat' }); speech(g, ccx - 34, my + 48, 68, C.muted, t, ph); }
            buddy(g, ccx, my + mh + 6, 22, C.d, { hat: true, mood: 'flat' });
          });
          V.line(g, half, my + 4, half, my + mh, { color: V.alpha(C.muted, 0.3), lw: 1, dash: [3, 3] });
        }
        // barras
        const by = topH + 26, bh = h - by - 44, bx = 52, bW = w - bx - 16;
        const yv = (v) => by + bh - (v / 100) * bh;
        [0, 25, 50, 75, 100].forEach((v) => { V.line(g, bx, yv(v), bx + bW, yv(v), { color: C.grid, lw: 1 }); axisNum(g, v + ' %', bx - 6, yv(v)); });
        const nG = d.groups.length, gW = bW / nG;
        const k = ez(el / 0.5);
        d.groups.forEach((gr, i) => {
          const bwid = Math.min(70, gW * 0.28);
          const c0 = bx + gW * i + gW / 2;
          d.vals[i].forEach((v, j) => {
            const x = c0 + (j ? 6 : -6 - bwid);
            const vv = j === 0 ? v * k : lerp(d.vals[i][0], v, ez((el - 0.35) / 0.6)) * k;
            V.box(g, x, yv(vv), bwid, Math.max(1, by + bh - yv(vv)), { r: 4, fill: j ? V.alpha(C.a, 0.75) : V.alpha(C.muted, 0.45), stroke: j ? C.a : C.muted, lw: 1.2 });
            V.text(g, fmt(vv, 1) + ' %', x + bwid / 2, yv(vv) - 7, { size: 11.5, font: 'mono', weight: 600, color: C.ink, align: 'center' });
          });
          const dd = d.vals[i][1] - d.vals[i][0];
          if (el > 0.9) V.text(g, `+${fmt(dd, 1)} pp`, c0, yv(d.vals[i][1]) - 26, { size: 12, font: 'mono', weight: 700, color: C.a, align: 'center' });
          V.text(g, gr, c0, by + bh + 16, { size: 11.5, font: 'mono', color: C.ink, align: 'center', maxW: gW - 6 });
        });
        // legenda
        const ly = h - 12;
        g.fillStyle = V.alpha(C.muted, 0.45); g.fillRect(bx, ly - 8, 10, 10);
        V.text(g, d.series[0], bx + 15, ly + 1, { size: 11, font: 'mono', color: C.muted });
        const lx2 = bx + 25 + tw(g, d.series[0], 11, 'mono');
        g.fillStyle = V.alpha(C.a, 0.75); g.fillRect(lx2, ly - 8, 10, 10);
        V.text(g, d.series[1], lx2 + 15, ly + 1, { size: 11, font: 'mono', color: C.a });
      });
    })();

    /* ================================================ FIG 4 — debate multiagente */
    (function () {
      const fig = root.querySelector('#fig-mad');
      const ANS = ['53', '48', '58', '65'];
      const S = { N: 3, R: 3, s: 0.5, v: 0.8, scen: 'certa' };
      const stg = fig.querySelector('.la-fig-stage');
      const setH = () => { const nar = fig.clientWidth < 560; const want = (nar ? (S.N <= 3 ? 400 : 600) : 410) + 'px'; if (stg.style.height !== want) stg.style.height = want; };
      setH();
      const roH = new ResizeObserver(setH); roH.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => roH.disconnect());
      let B = null, t0 = 0, need = true;
      function simulate() {
        const rng = V.rng(11 + S.N * 7 + (S.scen === 'certa' ? 0 : 100));
        const N = S.N;
        const nRight = S.scen === 'certa' ? N - Math.floor((N - 1) / 2) : Math.floor((N - 1) / 2);
        const init = [];
        for (let i = 0; i < N; i++) {
          let p;
          if (i < nRight) p = [0.5, 0.24, 0.16, 0.1];
          else p = rng() < 0.75 || S.scen === 'errada' ? [0.2, 0.52, 0.17, 0.11] : [0.2, 0.17, 0.5, 0.13];
          p = p.map((x) => x * (0.85 + 0.3 * rng()));
          const z = p.reduce((a, b) => a + b, 0);
          init.push(p.map((x) => x / z));
        }
        // embaralha a ordem para os certos não ficarem sempre à esquerda
        const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => ((a * 7 + 3) % 5) - ((b * 7 + 3) % 5));
        const b0 = order.map((i) => init[i]);
        const out = [b0];
        const am = (p) => p.indexOf(Math.max(...p));
        for (let r = 1; r < S.R; r++) {
          const prev = out[r - 1];
          const nxt = prev.map((p, i) => {
            const others = prev.filter((_, j) => j !== i);
            const f = others.length ? others.filter((q) => am(q) === 0).length / others.length : 0;
            const lg = p.map((pa, a) => {
              const mean = others.length ? others.reduce((s2, q) => s2 + Math.log(q[a]), 0) / others.length : Math.log(pa);
              return S.s * Math.log(pa) + (1 - S.s) * (mean + (a === 0 ? S.v * f : 0)); // v só age pela leitura
            });
            const m = Math.max(...lg), e = lg.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
            return e.map((x) => Math.max(1e-4, x / z));
          });
          out.push(nxt);
        }
        return out;
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'agentes N', min: 2, max: 6, step: 1, value: 3, onInput: (v) => { S.N = v; need = true; upd(); setH(); } });
      UI.slider(ctl, { label: 'rodadas R', min: 1, max: 4, step: 1, value: 3, onInput: (v) => { S.R = v; need = true; upd(); } });
      UI.slider(ctl, { label: 'autoconfiança s', min: 0, max: 1, step: 0.05, value: 0.5, fmt: (v) => fmt(v, 2), onInput: (v) => { S.s = v; need = true; } });
      UI.slider(ctl, { label: 'vantagem da verdade v', min: 0, max: 3, step: 0.1, value: 0.8, fmt: (v) => fmt(v, 1), onInput: (v) => { S.v = v; need = true; } });
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'certa', label: 'maioria começa certa' }, { v: 'errada', label: 'maioria começa errada' }], value: 'certa', onChange: (v) => { S.scen = v; need = true; } });
      const rM = UI.readout(ctl, 'maioria final', '—'), rT = UI.readout(ctl, 'tokens gerados', '—'), rC = UI.readout(ctl, 'self-consistency igual', '—');
      function upd() { rT.set((S.N * S.R * 300).toLocaleString('pt-BR')); rC.set(`${S.N * S.R} amostras`); }
      upd();
      const am = (p) => p.indexOf(Math.max(...p));
      const Tr = 2.6;

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (need) { B = simulate(); t0 = t; need = false; const fin = B[B.length - 1].map(am); const cnt = [0, 0, 0, 0]; fin.forEach((a) => cnt[a]++); const best = cnt.indexOf(Math.max(...cnt)); rM.set(`${ANS[best]} ${best === 0 ? '✓' : '✗'}`, best === 0 ? 'ok' : 'bad'); }
        const N = S.N, R = S.R;
        const el = t - t0;
        // rodada atual e interpolação
        const r = Math.min(R - 1, Math.floor(el / Tr));
        const local = el - r * Tr;
        const msgK = r >= 1 ? clamp(local / 1.1, 0, 1) : 0;
        const morph = r >= 1 ? ez((local - 1.1) / 0.9) : 1;
        const cur = B[r].map((p, i) => (r >= 1 ? p.map((x, a) => lerp(B[r - 1][i][a], x, morph)) : p));
        const narrow = w < 560;
        const cols = narrow ? Math.min(N, 3) : N, rows = Math.ceil(N / cols);
        const headH = narrow ? 54 : 40, footH = narrow ? 64 : 46;
        const cellW = (w - 24) / cols, cellH = (h - headH - footH) / rows;
        V.text(g, 'Quanto é 12 × 4 + 5?', 16, 24, { size: 14, weight: 700, color: C.ink });
        V.text(g, r === 0 ? 'rodada 1: respostas independentes' : `rodada ${r + 1} de ${R}: cada um leu os outros`, narrow ? 16 : w - 16, narrow ? 44 : 24, { size: 11.5, font: 'mono', color: C.muted, align: narrow ? 'left' : 'right', maxW: narrow ? w - 32 : w * 0.5 });
        const bs = clamp(cellH * 0.2, 24, 34);
        const hh = Math.min(140, cellH - bs * 1.2 - 44);
        const blockH = bs * 1.2 + 18 + hh + 18;
        const yoff = Math.max(0, (cellH - blockH) / 2);
        const pos = (i) => { const c = i % cols, rw = Math.floor(i / cols); return [12 + cellW * (c + 0.5), headH + cellH * rw + yoff]; };
        // mensagens
        if (r >= 1 && msgK > 0 && msgK < 1) {
          for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
            if (i === j) continue;
            const [x1, y1] = pos(i), [x2, y2] = pos(j);
            const ya = y1 + bs * 0.6 + 6, yb = y2 + bs * 0.6 + 6;
            const [cx, cy] = curveCtl(x1, ya, x2, yb, (i < j ? -1 : 1) * Math.min(60, Math.abs(x2 - x1) * 0.25 + 18));
            const [x, y] = qpt(x1, ya, cx, cy, x2, yb, ez(msgK));
            const col = am(B[r - 1][i]) === 0 ? C.a : C.f;
            V.glow(g, x, y, 9, col, 0.8); V.dot(g, x, y, 2.6, col);
          }
        }
        for (let i = 0; i < N; i++) {
          const [cx, y0] = pos(i);
          const p = cur[i], a = am(p);
          const changed = r >= 1 && am(B[r - 1][i]) !== am(B[r][i]) && morph > 0.3;
          buddy(g, cx, y0 + bs * 0.6 + 6 + Math.sin(t * 2 + i) * 1.4, bs, C.a, { mood: changed ? 'o' : 'happy' });
          // balão com a resposta dita
          const lab = ANS[a];
          chip(g, cx + bs * 0.55, y0 + 2, 34, 20, lab, { fill: V.alpha(a === 0 ? C.a : C.f, 0.2), stroke: a === 0 ? C.a : C.f, size: 11.5, weight: 700 });
          // mini histograma
          const hx = cx - Math.min(cellW * 0.42, 80), hw = Math.min(cellW * 0.84, 160);
          const hy = y0 + bs * 1.2 + 18;
          const bw = hw / 4;
          for (let k = 0; k < 4; k++) {
            const v = p[k], bh = v * hh;
            V.box(g, hx + k * bw + 2, hy, bw - 4, hh, { r: 3, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.18), lw: 1 });
            V.box(g, hx + k * bw + 2, hy + hh - bh, bw - 4, Math.max(1, bh), { r: 3, fill: V.alpha(k === 0 ? C.a : C.f, k === a ? 0.8 : 0.3), stroke: k === 0 ? C.a : C.f, lw: k === a ? 1.6 : 0.8 });
            if (k === a && bw >= 22) V.text(g, fmt(v, 2), hx + k * bw + bw / 2, hy + hh - bh - 5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center' });
            V.text(g, ANS[k], hx + k * bw + bw / 2, hy + hh + 13, { size: 11, font: 'mono', color: k === 0 ? C.a : C.muted, align: 'center' });
          }
          V.line(g, hx, hy + hh, hx + hw, hy + hh, { color: V.alpha(C.muted, 0.5), lw: 1 });
        }
        // rodapé: maioria e custo
        const fin = cur.map(am), cnt = [0, 0, 0, 0]; fin.forEach((a) => cnt[a]++);
        const best = cnt.indexOf(Math.max(...cnt));
        const tie = cnt.filter((c) => c === cnt[best]).length > 1;
        const fy = h - 26;
        const txt = tie ? `maioria agora: empate` : `maioria agora: ${ANS[best]} (${cnt[best]}/${N})`;
        V.text(g, txt, 16, narrow ? fy - 32 : fy, { size: 12.5, weight: 700, color: tie ? C.d : best === 0 ? C.a : RED() });
        const gen = N * Math.min(R, r + 1) * 300;
        V.text(g, `tokens gerados até aqui: ${gen.toLocaleString('pt-BR')} (L = 300, ilustrativo)`, narrow ? 16 : w - 16, narrow ? fy - 14 : fy, { size: 11, font: 'mono', color: C.muted, align: narrow ? 'left' : 'right', maxW: narrow ? w - 32 : w * 0.6 });
        V.text(g, 'menta = resposta certa (53) · magenta = erradas', 16, fy + 16, { size: 11, font: 'mono', color: C.muted, maxW: w - 32 });
        if (el > R * Tr + 2.8) { t0 = t; }
      });
    })();

    /* ================================================ FIG 5 — SPIN */
    (function () {
      const fig = root.querySelector('#fig-spin');
      adaptH(fig, 460, 560);
      const S = { lam: 0.1, lh: 2.0, lo: -1.0, it: 0, itShow: 0, auto: true };
      const NX = 161, XS = Array.from({ length: NX }, (_, i) => -4 + (8 * i) / (NX - 1));
      const gauss = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
      const pData = XS.map((x) => 0.6 * gauss(x, 1.2, 0.55) + 0.4 * gauss(x, -0.6, 0.5));
      const p0 = XS.map((x) => gauss(x, -1.6, 1.0));
      const dx = 8 / (NX - 1);
      const TV0 = 0.5 * pData.reduce((s2, v, i) => s2 + Math.abs(v - p0[i]), 0) * dx;
      const wOf = (it) => 1 - Math.pow(0.5, it);
      function sampleFrom(p, rng, n) {
        const cdf = []; let acc = 0; p.forEach((v) => { acc += v * dx; cdf.push(acc); });
        const out = [];
        for (let k = 0; k < n; k++) { const u = rng() * acc; let i = cdf.findIndex((c) => c >= u); if (i < 0) i = NX - 1; out.push(XS[i] + (rng() - 0.5) * dx); }
        return out;
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'λ', min: 0.05, max: 1, step: 0.05, value: 0.1, fmt: (v) => fmt(v, 2), onInput: (v) => { S.lam = v; upd(); } });
      UI.slider(ctl, { label: 'log-ratio humano', min: -3, max: 3, step: 0.1, value: 2, fmt: (v) => fmt(v, 1), onInput: (v) => { S.lh = v; upd(); } });
      UI.slider(ctl, { label: 'log-ratio próprio', min: -3, max: 3, step: 0.1, value: -1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.lo = v; upd(); } });
      const seg = UI.seg(ctl, { label: 'iteração', options: [0, 1, 2, 3, 4].map((v) => ({ v, label: 't=' + v })), value: 0, onChange: (v) => { S.it = v; S.auto = false; tg.set(false); } });
      const tg = UI.toggle(ctl, { label: 'avançar sozinho', value: true, onChange: (v) => { S.auto = v; } });
      const rt = UI.readout(ctl, 't', '—'), rl = UI.readout(ctl, 'ℓ(t)', '—'), rw = UI.readout(ctl, 'peso σ(−t)', '—'), rtv = UI.readout(ctl, 'TV(p_data, p_θt)', '—');
      ctl.querySelectorAll('.la-ctl label').forEach((l) => { l.style.textTransform = 'none'; });
      const tval = () => S.lam * (S.lh - S.lo);
      function upd() { const tt = tval(); rt.set(fmt(tt, 2)); rl.set(fmt(Math.log(1 + Math.exp(-tt)), 3)); rw.set(fmt(sig(-tt), 3)); }
      upd();
      let lastStep = 0, samples = null, sampIt = -1;

      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.auto && t - lastStep > 3.4) { lastStep = t; S.it = (S.it + 1) % 5; seg.set(S.it, true); }
        if (S.it === 0 && S.itShow > 0.5) S.itShow = 0;
        S.itShow += (S.it - S.itShow) * (1 - Math.exp(-dt * 5));
        if (Math.abs(S.it - S.itShow) < 0.002) S.itShow = S.it;
        const wv = wOf(S.itShow);
        const pM = p0.map((v, i) => (1 - wv) * v + wv * pData[i]);
        const TV = (1 - wv) * TV0;
        rtv.set(fmt(TV, 2), TV < 0.08 ? 'ok' : undefined);
        if (sampIt !== S.it) { const rng = V.rng(31 + S.it * 13); samples = { ch: sampleFrom(pData, rng, 9), rj: sampleFrom(pM, rng, 9) }; sampIt = S.it; }
        // painel superior
        const pad = 16, top = w < 560 ? 48 : 30, topH = h * 0.5 - 22 - (w < 560 ? 18 : 0);
        const x0 = pad + 6, x1 = w - pad;
        const X = (x) => x0 + ((x + 4) / 8) * (x1 - x0);
        const ymax = 0.52;
        const Y = (v) => top + topH - (v / ymax) * (topH - 16);
        V.text(g, `iteração t = ${S.it}`, pad, 18, { size: 12, font: 'mono', weight: 700, color: C.ink });
        const nar5 = w < 560;
        V.text(g, TV < 0.08 ? 'quase indistinguíveis: o gradiente esperado some (ponto fixo)' : 'o discriminador ainda separa as duas', nar5 ? pad : w - pad, nar5 ? 36 : 18, { size: 11.5, font: 'mono', color: TV < 0.08 ? C.a : C.muted, align: nar5 ? 'left' : 'right', maxW: nar5 ? w - 2 * pad : w * 0.62 });
        V.line(g, x0, Y(0), x1, Y(0), { color: V.alpha(C.muted, 0.5), lw: 1 });
        // área de diferença
        for (let i = 0; i < NX - 1; i += 1) {
          const d = pData[i] - pM[i];
          if (Math.abs(d) < 0.004) continue;
          g.fillStyle = d > 0 ? V.alpha(C.c, 0.16) : V.alpha(C.b, 0.16);
          const xa = X(XS[i]), xb = X(XS[i + 1]);
          const ya = Y(Math.max(pData[i], pM[i])), yb = Y(Math.min(pData[i], pM[i]));
          g.fillRect(xa, ya, xb - xa + 0.6, yb - ya);
        }
        const curve = (p, col, lw, dash) => { g.beginPath(); p.forEach((v, i) => { const x = X(XS[i]), y = Y(v); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        curve(pData, C.c, 2.4);
        curve(pM, C.b, 2.4, [6, 4]);
        // setas do gradiente
        for (let xv = -3.5; xv <= 3.5; xv += 0.5) {
          const i = Math.round(((xv + 4) / 8) * (NX - 1));
          const d = pData[i] - pM[i];
          if (Math.abs(d) < 0.02) continue;
          const len = clamp(Math.abs(d) * 90, 8, 34) * (0.85 + 0.15 * Math.sin(t * 4 + xv));
          const x = X(xv), ya = Y(Math.max(pData[i], pM[i]) * 0.5 + Math.min(pData[i], pM[i]) * 0.5);
          if (d > 0) V.arrow(g, x, ya + len / 2, x, ya - len / 2, { color: C.a, head: 6, lw: 2 });
          else V.arrow(g, x, ya - len / 2, x, ya + len / 2, { color: RED(), head: 6, lw: 2 });
        }
        // amostras
        samples.ch.forEach((x, k) => { const bl = 0.6 + 0.4 * Math.sin(t * 3 + k); V.dot(g, X(x), Y(0) + 9, 3.4, V.alpha(C.c, bl)); });
        samples.rj.forEach((x, k) => { const bl = 0.6 + 0.4 * Math.sin(t * 3 + k + 1); V.dot(g, X(x), Y(0) + 18, 3.4, V.alpha(C.b, bl)); });
        // legenda das duas fileiras de amostras (de cima: humanas; de baixo: próprias)
        const lgY = Y(0) + 34;
        const lg1 = 'humanas y ~ p_data (chosen, fileira de cima)', lg2 = 'próprias y′ ~ p_θt (rejected, de baixo)';
        const nar = w < 560;
        V.dot(g, x0 + 4, lgY - 4, 4, C.c);
        V.text(g, nar ? 'humanas (chosen, fileira de cima)' : lg1, x0 + 13, lgY, { size: 11, font: 'mono', color: C.c, maxW: nar ? x1 - x0 - 14 : (x1 - x0) / 2 - 20 });
        const l2x = nar ? x0 : x0 + (x1 - x0) / 2, l2y = nar ? lgY + 15 : lgY;
        V.dot(g, l2x + 4, l2y - 4, 4, C.b);
        V.text(g, nar ? 'próprias (rejected, fileira de baixo)' : lg2, l2x + 13, l2y, { size: 11, font: 'mono', color: C.b, maxW: nar ? x1 - x0 - 14 : (x1 - x0) / 2 - 14 });
        V.text(g, 'p_data', X(1.2) + 6, Y(Math.max(...pData)) - 4, { size: 11.5, font: 'mono', weight: 700, color: C.c });
        const im = pM.indexOf(Math.max(...pM));
        V.text(g, 'p_θt', X(XS[im]) - 6, Y(pM[im]) - 6, { size: 11.5, font: 'mono', weight: 700, color: C.b, align: 'right' });
        // painel inferior: perda logística
        const by = h * 0.5 + (w < 560 ? 66 : 48), bh = h - by - 26;
        const tx0 = pad + 34, tx1 = w - pad - (w < 560 ? 8 : 150);
        const TX = (tv) => tx0 + ((tv + 4) / 8) * (tx1 - tx0);
        const LY = (lv) => by + bh - (lv / 4.2) * bh;
        V.line(g, tx0, by + bh, tx1, by + bh, { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.line(g, TX(0), by, TX(0), by + bh, { color: C.grid, lw: 1 });
        [0, 2, 4].forEach((v) => axisNum(g, String(v), tx0 - 6, LY(v)));
        [-4, -2, 0, 2, 4].forEach((v) => V.text(g, String(v), TX(v), by + bh + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        g.beginPath();
        for (let k = 0; k <= 80; k++) { const tv = -4 + k / 10, lv = Math.log(1 + Math.exp(-tv)); if (k) g.lineTo(TX(tv), LY(lv)); else g.moveTo(TX(tv), LY(lv)); }
        g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
        V.text(g, 'ℓ(t) = log(1 + e^−t)', tx0 + 6, by + 6, { size: 11.5, font: 'mono', color: C.a, base: 'top' });
        const tt = clamp(tval(), -4, 4), lv = Math.log(1 + Math.exp(-tt)), wgt = sig(-tt);
        // tangente
        const sl = -wgt;
        V.line(g, TX(tt - 1.2), LY(lv - sl * 1.2), TX(tt + 1.2), LY(lv + sl * 1.2), { color: V.alpha(C.d, 0.8), lw: 1.4, dash: [4, 3] });
        // barra do peso
        g.fillStyle = V.alpha(C.d, 0.55); g.fillRect(TX(tt) - 4, LY(wgt * 4.2 * 0.95), 8, by + bh - LY(wgt * 4.2 * 0.95));
        V.glow(g, TX(tt), LY(lv), 12, C.d, 0.9); V.dot(g, TX(tt), LY(lv), 4.5, C.d);
        const info = [`t = ${fmt(tval(), 2)}`, `ℓ = ${fmt(Math.log(1 + Math.exp(-tval())), 3)}`, `σ(−t) = ${fmt(wgt, 3)}`];
        info.forEach((s2, k) => V.text(g, s2, w - pad, by + 14 + k * 18, { size: 12, font: 'mono', weight: k === 2 ? 700 : 500, color: k === 2 ? C.d : C.ink, align: 'right' }));
      });
    })();

    /* ================================================ FIG 6 — Self-Rewarding (stepper) */
    (function () {
      const fig = root.querySelector('#fig-selfreward');
      adaptH(fig, 430, 450);
      const CRIT = ['relevância', 'cobertura', 'utilidade', 'clareza', 'expertise'];
      const HIT = [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]; // por candidato A..D
      const SC = HIT.map((r) => r.reduce((a, b) => a + b, 0)); // [4,2,5,3]
      const LAB = ['A', 'B', 'C', 'D'];
      const steps = [
        { t: 'O modelo <span style="text-transform:none">M<sub>t</sub></span> inventa um prompt', d: 'Com exemplos few-shot tirados do IFT, o próprio modelo escreve um pedido novo. Ninguém precisa escrever prompts para as próximas iterações.' },
        { t: 'Quatro candidatos', d: 'Para o prompt novo, \\(M_t\\) amostra \\(N = 4\\) respostas (com temperatura, para que sejam diferentes).' },
        { t: 'O mesmo modelo veste a toga', d: 'Agora \\(M_t\\) é o juiz. Ele aplica a rubrica aditiva: cada critério cumprido soma 1 ponto. As notas saem \\([4, 2, 5, 3]\\).' },
        { t: 'O par: maior contra menor', d: 'C (nota 5) vira o <b>chosen</b> e B (nota 2) o <b>rejected</b>; A e D ficam de fora. Se a maior e a menor nota coincidem, não há par, e o prompt é descartado.' },
        { t: 'DPO aumenta a margem', d: 'Com \\(\\beta = 0{,}1\\) e \\(M_t\\) como referência, o DPO aumenta a margem \\(\\beta(\\Delta_C - \\Delta_B)\\). Em geral C sobe e B desce, mas a perda só garante a diferença (veja o deslocamento de verossimilhança no capítulo de DPO).' },
        { t: 'Nasce <span style="text-transform:none">M<sub>t+1</sub></span> (resultados publicados)', d: 'Com o Llama 2 70B: AlpacaEval 2.0 de 9,94 % (M1) para 15,38 % (M2) e 20,44 % (M3). No confronto direto, M2 venceu M1 em 55,5 % e perdeu em 11,7 %.' },
      ];
      let H6 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H6 && H6.redraw() });
      H6 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 560;
        // trilha do laço
        const trk = ['prompt', 'candidatos', 'rubrica', 'par', 'DPO', 'M_t+1'];
        const tx0 = 20, tx1 = w - 20, ty = 20;
        V.line(g, tx0, ty, tx1, ty, { color: V.alpha(C.muted, 0.35), lw: 2 });
        trk.forEach((lb, k) => {
          const x = lerp(tx0, tx1, k / (trk.length - 1));
          V.dot(g, x, ty, k === i ? 7 : 5, k <= i ? C.a : V.alpha(C.muted, 0.5));
          if (!narrow || k === i) V.text(g, lb, x, ty + 18, { size: 11, font: 'mono', color: k === i ? C.ink : C.muted, align: k === 0 ? 'left' : k === trk.length - 1 ? 'right' : 'center' });
        });
        const top = 56;
        const mx = narrow ? 34 : 60, my = top + 50;
        const judge = i === 2 || i === 3;
        if (i <= 1) {
          buddy(g, mx, my + Math.sin(t * 2) * 1.5, narrow ? 34 : 42, C.a, { mood: 'happy' });
          V.text(g, 'M_t', mx, my + 38, { size: 11.5, font: 'mono', weight: 600, color: C.ink, align: 'center' });
        } else if (judge) {
          buddy(g, mx, top + 22 + Math.sin(t * 2) * 1.2, narrow ? 26 : 32, C.a, { hat: true, mood: 'flat' });
          if (!narrow) V.text(g, 'M_t de toga', mx + 24, top + 26, { size: 11.5, font: 'mono', weight: 600, color: C.d });
        }
        const promptTxt = 'Explique para uma criança por que o céu é azul.';
        if (i === 0) {
          // exemplos few-shot do IFT sobem até o modelo
          const n3 = 3, gap = 14, cy = top + (narrow ? 150 : 140);
          const cwd = (w - 40 - gap * (n3 - 1)) / n3, chh = narrow ? 144 : 130;
          const EX = ['Resuma este parágrafo…', 'Escreva um haicai sobre…', 'Explique o que é…'];
          V.text(g, 'exemplos few-shot tirados do IFT (ilustrativos)', 20, cy - 12, { size: 11, font: 'mono', color: C.c });
          for (let k = 0; k < n3; k++) {
            const kk = ez((el - k * 0.25) / 0.7);
            const x = 20 + k * (cwd + gap) + (1 - kk) * 60, y = cy;
            g.globalAlpha = Math.max(0.05, kk);
            V.box(g, x, y, cwd, chh, { r: 8, fill: V.alpha(C.c, 0.08), stroke: V.alpha(C.c, 0.7), lw: 1.2 });
            V.text(g, 'instrução', x + 10, y + 16, { size: 10.5, font: 'mono', color: C.muted });
            wrapC(g, EX[k], x + 10, y + 32, cwd - 20, { size: narrow ? 11 : 11.5, lineH: 14, maxLines: 2 });
            V.text(g, 'resposta', x + 10, y + 70, { size: 10.5, font: 'mono', color: C.muted });
            for (let l = 0; l < (narrow ? 4 : 3); l++) V.line(g, x + 10, y + 84 + l * 13, x + 10 + (cwd - 20) * (0.5 + 0.5 * Math.abs(Math.sin(k * 2.3 + l))), y + 84 + l * 13, { color: V.alpha(C.c, 0.55), lw: 2.2 });
            g.globalAlpha = 1;
            // partícula subindo até o modelo
            const u2 = ((t * 0.5 + k * 0.33) % 1);
            const [px, py] = qpt(x + cwd / 2, y - 4, (x + cwd / 2 + mx) / 2, my + 40, mx + 10, my + 22, u2);
            if (kk > 0.9) { V.glow(g, px, py, 9, C.c, 0.7); V.dot(g, px, py, 2.4, C.c); }
          }
          const n = Math.floor(clamp((el - 1.2) / 2.2, 0, 1) * promptTxt.length);
          const bx = mx + 40, bw = Math.min(w - bx - 20, 460);
          V.box(g, bx, top + 20, bw, 54, { r: 10, fill: V.alpha(C.a, 0.12), stroke: C.a, lw: 1.3 });
          V.text(g, 'prompt novo (ilustrativo)', bx + 10, top + 34, { size: 10.5, font: 'mono', color: C.muted });
          wrapC(g, promptTxt.slice(0, n) + (n < promptTxt.length && Math.floor(t * 3) % 2 ? '▌' : ''), bx + 10, top + 52, bw - 20, { size: 12.5, lineH: 15, maxLines: 2 });
        }
        // posições dos candidatos
        const cx0 = narrow ? 90 : 150, cW = w - cx0 - 20;
        const candX = (k) => cx0 + (k + 0.5) * (cW / 4);
        if (i === 1) {
          const bx = mx + 40, bw = w - bx - 20;
          V.box(g, bx, top + 4, bw, 30, { r: 8, fill: V.alpha(C.a, 0.1), stroke: V.alpha(C.a, 0.6), lw: 1 });
          V.text(g, 'prompt: ' + promptTxt, bx + 10, top + 19.5, { size: 12, color: C.ink, base: 'middle', maxW: bw - 20 });
          const cwd = Math.min(cW / 4 - 10, 150), chh = narrow ? 200 : 170;
          for (let k = 0; k < 4; k++) {
            const kk = ez((el - k * 0.25) / 0.7);
            const x = lerp(mx + 20, candX(k), kk), y = lerp(my, top + 56, kk);
            g.globalAlpha = Math.max(0.1, kk);
            V.box(g, x - cwd / 2, y, cwd, chh, { r: 8, fill: V.alpha(C.panel, 0.95), stroke: V.alpha(C.a, 0.6), lw: 1.2 });
            chip(g, x - 15, y - 11, 30, 22, LAB[k], { fill: V.alpha(C.a, 0.25), stroke: C.a, size: 12, weight: 700 });
            const nL = 5 + (k % 3) * 2;
            for (let l = 0; l < nL; l++) {
              const lw2 = (cwd - 22) * (0.55 + 0.45 * Math.abs(Math.sin(k * 3.1 + l * 1.7)));
              const grow = clamp((el - 0.8 - k * 0.2 - l * 0.12) / 0.3, 0, 1);
              if (grow > 0 && 22 + l * 15 < chh - 8) V.line(g, x - cwd / 2 + 11, y + 22 + l * 15, x - cwd / 2 + 11 + lw2 * grow, y + 22 + l * 15, { color: V.alpha(C.muted, 0.7), lw: 2.4 });
            }
            g.globalAlpha = 1;
          }
          V.text(g, 'mesmo prompt, 4 amostras com temperatura', w / 2, top + 56 + chh + 26, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
        }
        if (i >= 2 && i <= 3) {
          const ky = top + 26;
          for (let k = 0; k < 4; k++) {
            const x = candX(k), y = ky;
            let fade = 1, col = C.a;
            if (i === 3) { if (k === 2) col = C.a; else if (k === 1) col = RED(); else { col = C.muted; fade = 0.35; } }
            g.globalAlpha = fade;
            chip(g, x - 17, y - 12, 34, 24, LAB[k], { fill: V.alpha(col, 0.2), stroke: col, size: 12.5, weight: 700 });
            g.globalAlpha = 1;
          }
        }
        if (i === 2 || i === 3) {
          // matriz da rubrica
          const gx = cx0, gy = top + 50, rowH = narrow ? 32 : 34;
          const cellW = Math.min(46, cW / 4 - 8);
          const nShow = i === 3 ? 20 : Math.floor(el / 0.18);
          CRIT.forEach((cr, r) => {
            V.text(g, cr, gx - 8, gy + r * rowH + rowH / 2, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
            for (let k = 0; k < 4; k++) {
              const idx = r * 4 + k;
              const x = candX(k) - cellW / 2, y = gy + r * rowH + 2;
              const shown = idx < nShow;
              const v = HIT[k][r];
              V.box(g, x, y, cellW, rowH - 4, { r: 4, fill: shown ? (v ? V.alpha(C.a, 0.55) : V.alpha(C.muted, 0.12)) : V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.3), lw: 1 });
              if (shown) V.text(g, String(v), x + cellW / 2, y + (rowH - 4) / 2 + 0.5, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center', base: 'middle' });
            }
          });
          const sy = gy + 5 * rowH + 6;
          V.text(g, 'nota', gx - 8, sy + 12, { size: 11.5, font: 'mono', weight: 700, color: C.d, align: 'right', base: 'middle' });
          for (let k = 0; k < 4; k++) {
            const partial = i === 3 ? SC[k] : HIT[k].reduce((s2, v, r) => s2 + (r * 4 + k < nShow ? v : 0), 0);
            const col = i === 3 ? (k === 2 ? C.a : k === 1 ? RED() : C.muted) : C.d;
            chip(g, candX(k) - cellW / 2, sy, cellW, 24, String(partial), { fill: V.alpha(col, 0.2), stroke: col, size: 13, weight: 700 });
          }
          if (i === 3) {
            const ey = sy + 44;
            V.text(g, narrow ? 'chosen' : 'chosen = C (5)', candX(2), ey, { size: 12, font: 'mono', weight: 700, color: C.a, align: 'center' });
            V.text(g, narrow ? 'rejected' : 'rejected = B (2)', candX(1), ey, { size: 12, font: 'mono', weight: 700, color: RED(), align: 'center' });
            if (!narrow) V.text(g, 'A e D ficam de fora', candX(3) + 10, ey, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          }
        }
        if (i === 4) {
          // DPO: barras de log-ratio
          const gx = narrow ? 30 : 90, gw2 = w - gx - 30, base = top + (h - top) * 0.55;
          const k = ez(el / 1.8);
          V.line(g, gx, base, gx + gw2, base, { color: V.alpha(C.muted, 0.6), lw: 1.2 });
          V.text(g, 'referência M_t', gx + gw2, base - 6, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          const dC = 8 * k, dB = -6 * k, mg = 0.1 * (dC - dB);
          V.text(g, `perda do par: ${fmt(Math.log(1 + Math.exp(-mg)), 3)}`, gx + gw2, top + 60, { size: 12, font: 'mono', weight: 700, color: C.d, align: 'right' });
          V.text(g, `margem β(Δ_C − Δ_B) = ${fmt(mg, 2)}`, gx + gw2, top + 78, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          const items = [{ l: `C (chosen) Δ = +${fmt(dC, 1)}`, v: 1.6, col: C.a }, { l: `B (rejected) Δ = ${fmt(dB, 1)}`, v: -1.2, col: RED() }];
          items.forEach((it, j) => {
            const x = gx + gw2 * (0.25 + j * 0.4), bw = Math.min(90, gw2 * 0.2);
            const hh = it.v * k * (h - top) * (narrow ? 0.18 : 0.21);
            V.box(g, x - bw / 2, Math.min(base, base - hh), bw, Math.max(1, Math.abs(hh)), { r: 4, fill: V.alpha(it.col, 0.55), stroke: it.col, lw: 1.4 });
            V.text(g, it.l, x, it.v > 0 ? base + 18 : base + Math.abs(hh) + 18, { size: 12, font: 'mono', weight: 700, color: it.col, align: 'center' });
            V.arrow(g, x + bw / 2 + 16, base, x + bw / 2 + 16, base - hh * 0.9, { color: it.col, head: 8, lw: 2.2 });
          });
          wrapC(g, 'Δ = log π_θ(y|x) − log π_Mt(y|x), em nats (valores ilustrativos)', gx, top + 22, narrow ? gw2 * 0.95 : gw2, { size: 11.5, font: 'mono', lineH: 15, maxLines: 2 });
        }
        if (i === 5) {
          const gx = narrow ? 40 : 70, gw2 = narrow ? w - gx - 20 : (w - gx) * 0.5, base = h - 50, ht = h - top - 110;
          const vals = [9.94, 15.38, 20.44];
          const k = ez(el / 1.6);
          const yv = (v) => base - (v / 25) * ht;
          [0, 10, 20].forEach((v) => { V.line(g, gx, yv(v), gx + gw2, yv(v), { color: C.grid, lw: 1 }); axisNum(g, v + ' %', gx - 6, yv(v)); });
          vals.forEach((v, j) => {
            const bw = gw2 / 3 * 0.55, x = gx + (j + 0.5) * gw2 / 3 - bw / 2;
            const vv = v * clamp(k * 3 - j, 0, 1);
            V.box(g, x, yv(vv), bw, Math.max(1, base - yv(vv)), { r: 4, fill: V.alpha(C.a, 0.35 + 0.2 * j), stroke: C.a, lw: 1.3 });
            V.text(g, fmt(vv, 2) + ' %', x + bw / 2, yv(vv) - 7, { size: 11.5, font: 'mono', weight: 700, color: C.ink, align: 'center' });
            V.text(g, 'M' + (j + 1), x + bw / 2, base + 15, { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center' });
          });
          V.text(g, 'AlpacaEval 2.0, win rate vs GPT-4 Turbo', gx, top + 26, { size: 11.5, font: 'mono', color: C.muted, maxW: gw2 + 40 });
          if (!narrow) {
            const hx = gx + gw2 + 40, hw = w - hx - 24;
            V.text(g, 'confronto direto', hx, top + 26, { size: 11.5, font: 'mono', color: C.muted });
            [['M2 × M1', 55.5, 11.7], ['M3 × M2', 47.7, 12.5]].forEach((r2, j) => {
              const y = top + 56 + j * 60;
              V.text(g, r2[0], hx, y, { size: 12, font: 'mono', weight: 700, color: C.ink });
              const winW = hw * r2[1] / 100 * k, loseW = hw * r2[2] / 100 * k, restW = hw * (100 - r2[1] - r2[2]) / 100 * k;
              g.fillStyle = V.alpha(C.a, 0.7); g.fillRect(hx, y + 8, winW, 18);
              g.fillStyle = V.alpha(C.muted, 0.3); g.fillRect(hx + winW, y + 8, restW, 18);
              g.fillStyle = V.alpha(RED(), 0.6); g.fillRect(hx + winW + restW, y + 8, loseW, 18);
              V.text(g, `vence ${fmt(r2[1], 1)} · perde ${fmt(r2[2], 1)} · restante ${fmt(100 - r2[1] - r2[2], 1)}`, hx, y + 42, { size: 11, font: 'mono', color: C.muted, maxW: hw });
            });
          }
        }
      });
    })();

    /* ================================================ FIG 7 — SPPO e o simplex */
    (function () {
      const fig = root.querySelector('#fig-sppo');
      adaptH(fig, 440, 570);
      const S = { mode: 'ciclo', c: 0.7, eta: 1, avg: true };
      let st = null;
      const Pmat = () => {
        const c = S.c;
        if (S.mode === 'ciclo') return [[0.5, c, 1 - c], [1 - c, 0.5, c], [c, 1 - c, 0.5]];
        const ac = Math.min(0.95, c + 0.15);
        return [[0.5, c, ac], [1 - c, 0.5, c], [1 - ac, 1 - c, 0.5]];
      };
      const win = (P, pi) => P.map((row) => row.reduce((s2, v, j) => s2 + v * pi[j], 0));
      const step = (P, pi) => { const wv = win(P, pi); const e = pi.map((p, i) => p * Math.exp(S.eta * wv[i])); const z = e.reduce((a, b) => a + b, 0); return e.map((x) => x / z); };
      function reset(seed) { const r = V.rng(seed || 5); let p = [0.42 + 0.12 * r(), 0.3 + 0.05 * r(), 0.2]; const z = p.reduce((a, b) => a + b, 0); p = p.map((x) => x / z); st = { pi: p, trail: [p], sum: p.slice(), n: 1, acc: 0 }; }
      reset(5);
      let bt = '';
      function fitBT() {
        if (S.mode === 'ciclo') { bt = 'r_A = r_B = r_C (sem sinal)'; return; }
        const P = Pmat(); const r = [0, 0, 0];
        for (let it = 0; it < 600; it++) {
          const gr = [0, 0, 0];
          for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { if (i === j) continue; const p = P[i][j]; const s2 = sig(r[i] - r[j]); gr[i] += p - s2; }
          for (let i = 0; i < 2; i++) r[i] += 0.3 * gr[i];
        }
        bt = `r_A − r_C = ${fmt(r[0] - r[2], 2)}, r_B − r_C = ${fmt(r[1] - r[2], 2)}`;
      }
      fitBT();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Preferências', options: [{ v: 'ciclo', label: 'cíclicas' }, { v: 'trans', label: 'transitivas' }], value: 'ciclo', onChange: (v) => { S.mode = v; fitBT(); reset(9); upd(); } });
      UI.slider(ctl, { label: 'força c', min: 0.55, max: 0.9, step: 0.05, value: 0.7, fmt: (v) => fmt(v, 2), onInput: (v) => { S.c = v; fitBT(); upd(); } });
      UI.slider(ctl, { label: 'passo η', min: 0.2, max: 4, step: 0.2, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.eta = v; } });
      UI.toggle(ctl, { label: 'mostrar a média', value: true, onChange: (v) => { S.avg = v; } });
      UI.button(ctl, '↻ outro ponto de partida', () => reset(Math.floor(Math.random() * 1000) + 1));
      const rW = UI.readout(ctl, 'vitória contra π_t', '—'), rB = UI.readout(ctl, 'melhor ajuste Bradley–Terry', '—');
      function upd() { rB.set(bt, S.mode === 'ciclo' ? 'warn' : undefined); }
      upd();
      ctl.querySelectorAll('.la-ctl label').forEach((l) => { l.style.textTransform = 'none'; });

      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const P = Pmat();
        st.acc += dt;
        if (st.acc > 0.28) {
          st.acc = 0;
          st.pi = step(P, st.pi); st.trail.push(st.pi); if (st.trail.length > 90) st.trail.shift();
          st.sum = st.sum.map((s2, i) => s2 + st.pi[i]); st.n++;
          if (st.n > 180) reset(st.n + 3);
        }
        const wv = win(P, st.pi);
        rW.set(`A ${fmt(wv[0], 2)} · B ${fmt(wv[1], 2)} · C ${fmt(wv[2], 2)}`);
        const narrow = w < 600;
        // matriz
        const cell = narrow ? Math.min(54, (w - 80) / 3) : Math.min(58, (w * 0.36 - 50) / 3);
        const mx = narrow ? (w - cell * 3) / 2 + 10 : 46, my = narrow ? 44 : 70;
        V.text(g, 'P(linha ≻ coluna)', mx + cell * 1.5, my - 26, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
          const v = P[i][j], x = mx + j * (cell + 3), y = my + i * (cell + 3);
          const col = v > 0.5 ? C.a : v < 0.5 ? C.f : C.muted;
          V.box(g, x, y, cell, cell, { r: 6, fill: V.alpha(col, 0.08 + Math.abs(v - 0.5) * 1.3), stroke: V.alpha(col, 0.7), lw: 1 });
          V.text(g, fmt(v, 2), x + cell / 2, y + cell / 2 + 0.5, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
        }
        ['A', 'B', 'C'].forEach((l, k) => {
          V.text(g, l, mx - 8, my + k * (cell + 3) + cell / 2, { size: 12, font: 'mono', weight: 700, color: C.muted, align: 'right', base: 'middle' });
          V.text(g, l, mx + k * (cell + 3) + cell / 2, my - 8, { size: 12, font: 'mono', weight: 700, color: C.muted, align: 'center' });
        });
        const pol = ['A', 'B', 'C'];
        if (!narrow) {
          V.text(g, S.mode === 'ciclo' ? 'A ≻ B ≻ C ≻ A: um ciclo' : 'A ≻ B ≻ C: há um vencedor', mx + cell * 1.5, my + cell * 3 + 38, { size: 11.5, font: 'mono', weight: 600, color: S.mode === 'ciclo' ? C.d : C.a, align: 'center' });
          pol.forEach((l, i) => V.text(g, `π_t(${l}) = ${fmt(st.pi[i], 2)}`, mx, my + cell * 3 + 64 + i * 18, { size: 11.5, font: 'mono', color: C.ink }));
        }
        // simplex
        const sx0 = narrow ? 24 : w * 0.42, sx1 = w - 24;
        const sy0 = narrow ? my + cell * 3 + 46 : 34, sy1 = h - 30;
        const side = Math.min(sx1 - sx0, (sy1 - sy0) / 0.866);
        const cxs = (sx0 + sx1) / 2;
        const VA = [cxs, sy0 + 6], VB = [cxs - side / 2, sy0 + 6 + side * 0.866], VC = [cxs + side / 2, sy0 + 6 + side * 0.866];
        const proj = (p) => [p[0] * VA[0] + p[1] * VB[0] + p[2] * VC[0], p[0] * VA[1] + p[1] * VB[1] + p[2] * VC[1]];
        g.beginPath(); g.moveTo(...VA); g.lineTo(...VB); g.lineTo(...VC); g.closePath();
        g.fillStyle = V.alpha(C.panel, 0.6); g.fill(); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1.2; g.stroke();
        V.text(g, 'só A', VA[0], VA[1] - 8, { size: 11.5, font: 'mono', weight: 700, color: C.ink, align: 'center' });
        V.text(g, 'só B', VB[0] + 2, VB[1] + 16, { size: 11.5, font: 'mono', weight: 700, color: C.ink, align: 'left' });
        V.text(g, 'só C', VC[0] - 2, VC[1] + 16, { size: 11.5, font: 'mono', weight: 700, color: C.ink, align: 'right' });
        // campo de setas
        const nG = 8;
        for (let a = 1; a < nG; a++) for (let b = 1; b < nG - a; b++) {
          const p = [a / nG, b / nG, 1 - a / nG - b / nG];
          if (p[2] <= 0.01) continue;
          const q = step(P, p);
          const p1 = proj(p), p2 = proj(q);
          const dx = p2[0] - p1[0], dy = p2[1] - p1[1], L = Math.hypot(dx, dy);
          if (L < 0.3) continue;
          const len = Math.min(16, Math.max(8, 5 + L * 0.6));
          V.arrow(g, p1[0], p1[1], p1[0] + (dx / L) * len, p1[1] + (dy / L) * len, { color: V.alpha(C.muted, 0.85), head: 5, lw: 1.3 });
        }
        // Nash
        const nash = proj([1 / 3, 1 / 3, 1 / 3]);
        if (S.mode === 'ciclo') { V.dot(g, nash[0], nash[1], 4, C.e); V.text(g, 'Nash', nash[0] + 8, nash[1] + 16, { size: 11, font: 'mono', color: C.e }); }
        // trilha
        g.beginPath();
        st.trail.forEach((p, k) => { const q = proj(p); if (k) g.lineTo(q[0], q[1]); else g.moveTo(q[0], q[1]); });
        g.strokeStyle = V.alpha(C.b, 0.55); g.lineWidth = 1.4; g.stroke();
        st.trail.forEach((p, k) => { const q = proj(p); V.dot(g, q[0], q[1], 1.8, V.alpha(C.b, 0.25 + 0.6 * (k / st.trail.length))); });
        const qc = proj(st.pi);
        V.glow(g, qc[0], qc[1], 14, C.b, 0.9); V.dot(g, qc[0], qc[1], 5, C.b);
        const nearTop = qc[1] < VA[1] + 40;
        V.text(g, 'π_t', qc[0] + (nearTop ? 12 : 9), qc[1] + (nearTop ? 16 : -8), { size: 12, font: 'mono', weight: 700, color: C.b });
        if (S.avg) {
          const avg = st.sum.map((s2) => s2 / st.n), qa = proj(avg);
          g.fillStyle = C.d; g.beginPath();
          for (let k = 0; k < 10; k++) { const r = k % 2 ? 3.2 : 8, an = -Math.PI / 2 + (k * Math.PI) / 5; g.lineTo(qa[0] + r * Math.cos(an), qa[1] + r * Math.sin(an)); }
          g.closePath(); g.fill();
          V.text(g, 'média', qa[0] - 10, qa[1] - 10, { size: 11, font: 'mono', weight: 600, color: C.d, align: 'right' });
        }
        V.text(g, `iteração ${st.n}`, sx1, sy0 + 10, { size: 11, font: 'mono', color: C.muted, align: 'right' });
      });
    })();

    /* ================================================ FIG 8 — Absolute Zero */
    (function () {
      const fig = root.querySelector('#fig-azr');
      adaptH(fig, 450, 640);
      const S = { k: 2, mode: 'ded' };
      const G = 8;
      const rProp = (k) => (k === 0 || k === G ? 0 : 1 - k / G);
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'acertos do resolvedor (de 8)', min: 0, max: 8, step: 1, value: 2, onInput: (v) => { S.k = v; upd(); } });
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'ded', label: 'dedução' }, { v: 'abd', label: 'abdução' }, { v: 'ind', label: 'indução' }], value: 'ded', onChange: (v) => { S.mode = v; } });
      const rS = UI.readout(ctl, 'r̄ solve', '—'), rP = UI.readout(ctl, 'r propose', '—');
      function upd() { rS.set(fmt(S.k / G, 3)); const rp = rProp(S.k); rP.set(fmt(rp, 3), rp === 0 ? 'bad' : rp >= 0.7 ? 'ok' : undefined); }
      upd();
      ctl.querySelectorAll('.la-ctl label').forEach((l) => { l.style.textTransform = 'none'; });
      const FIELDS = {
        ded: [['programa', 'f = lambda x: sorted(x)[::2]'], ['entrada', '[5, 1, 4, 2]'], ['saída', '?']],
        abd: [['programa', 'f = lambda x: sorted(x)[::2]'], ['entrada', '?'], ['saída', '[1, 4]']],
        ind: [['exemplos', '[5,1,4,2]→[1,4] · [9,3,7]→[3,9]'], ['programa', '?'], ['', '']],
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const ph = (t % 5) / 5;
        // topo: propositor → cartão → executor → resolvedor
        const topH = narrow ? 300 : 190;
        const bs = narrow ? 30 : 38;
        const pX = 30, sX = w - 30, by = narrow ? 34 : 48;
        buddy(g, pX, by + Math.sin(t * 2) * 1.5, bs, C.a, { mood: 'happy' });
        buddy(g, sX, by + Math.sin(t * 2 + 1) * 1.5, bs, C.a, { mood: S.k === 0 ? 'sad' : S.k === G ? 'flat' : 'o' });
        V.text(g, 'propositor', 6, by + bs * 0.5 + 15, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'left' });
        V.text(g, 'resolvedor', sX + 12, by + bs * 0.5 + 15, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'right' });
        V.line(g, pX + bs * 0.6, 10, sX - bs * 0.6, 10, { color: V.alpha(C.muted, 0.4), lw: 1, dash: [3, 4] });
        const mpw = tw(g, 'mesmos pesos', 11, 'mono') + 12;
        g.fillStyle = C.bg; g.fillRect(w / 2 - mpw / 2, 3, mpw, 15);
        V.text(g, 'mesmos pesos', w / 2, 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        // cartão da tarefa
        const cardX = narrow ? 14 : pX + bs + 14, cardW = narrow ? w - 28 : w * 0.5, cardY = narrow ? by + bs + 24 : 34;
        const rowsF = FIELDS[S.mode].filter((f) => f[0]);
        const cardH = rowsF.length * 26 + 14;
        const slide = ez(ph * 3);
        g.globalAlpha = 0.4 + 0.6 * slide;
        V.box(g, cardX, cardY, cardW, cardH, { r: 10, fill: V.alpha(C.panel, 0.95), stroke: C.a, lw: 1.3 });
        g.globalAlpha = 1;
        rowsF.forEach((f, r) => {
          const y = cardY + 20 + r * 26;
          V.text(g, f[0], cardX + 10, y, { size: 11, font: 'mono', color: C.muted, base: 'middle' });
          const vx = cardX + (narrow ? 82 : 90);
          if (f[1] === '?') {
            const pulse = 0.5 + 0.5 * Math.sin(t * 4);
            V.box(g, vx, y - 10, 70, 20, { r: 5, fill: V.alpha(C.d, 0.2 + 0.2 * pulse), stroke: C.d, lw: 1.4 });
            V.text(g, '?', vx + 35, y + 0.5, { size: 13, font: 'mono', weight: 700, color: C.d, align: 'center', base: 'middle' });
            V.text(g, 'o resolvedor preenche', vx + 80, y + 0.5, { size: 11, font: 'mono', color: C.d, base: 'middle', maxW: cardX + cardW - vx - 86 });
          } else V.text(g, f[1], vx, y + 0.5, { size: 11.5, font: 'mono', weight: 600, color: C.ink, base: 'middle', maxW: cardX + cardW - vx - 8 });
        });
        // executor
        const exX = narrow ? w / 2 - 60 : cardX + cardW + 14, exY = narrow ? cardY + cardH + 12 : cardY + 4, exW = narrow ? 120 : Math.max(90, sX - bs - 14 - exX - 10);
        V.box(g, exX, exY, exW, 34, { r: 8, fill: V.alpha(C.e, 0.16), stroke: C.e, lw: 1.3 });
        V.text(g, 'python ▸ exec' + (Math.floor(t * 2.5) % 2 ? '▌' : ''), exX + 10, exY + 17.5, { size: 11.5, font: 'mono', weight: 600, color: C.ink, base: 'middle', maxW: exW - 16 });
        // tentativas
        const atY = narrow ? exY + 54 : cardY + cardH + 22;
        const atX0 = narrow ? 30 : cardX, atW = narrow ? w - 60 : w - cardX - 40;
        V.text(g, 'G = 8 tentativas do resolvedor', atX0, atY - 4, { size: 11, font: 'mono', color: C.muted });
        const nRev = Math.min(G, Math.floor(ph * 12));
        for (let i = 0; i < G; i++) {
          const x = atX0 + (i + 0.5) * (Math.min(atW, 360) / G), y = atY + 18;
          const ok = i < S.k;
          if (i < nRev) { V.box(g, x - 13, y - 13, 26, 26, { r: 6, fill: V.alpha(ok ? C.a : RED(), 0.16), stroke: ok ? C.a : RED(), lw: 1.2 }); mark(g, x, y, ok, 6); }
          else V.box(g, x - 13, y - 13, 26, 26, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        }
        // gráfico da recompensa
        const gy0 = topH + (narrow ? 30 : 20), gx0 = 50, gx1 = narrow ? w - 20 : w * 0.6, gy1 = narrow ? h - 172 : h - 34;
        const GX = (r) => gx0 + r * (gx1 - gx0), GY = (v) => gy1 - v * (gy1 - gy0);
        V.line(g, gx0, gy1, gx1, gy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.line(g, gx0, gy0, gx0, gy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        g.fillStyle = V.alpha(C.a, 0.08); g.fillRect(GX(1 / G), gy0, GX(7 / G) - GX(1 / G), gy1 - gy0);
        V.text(g, 'zona de aprendibilidade', (GX(1 / G) + GX(7 / G)) / 2, gy1 - 8, { size: 11, font: 'mono', color: C.a, align: 'center' });
        [0, 0.5, 1].forEach((v) => axisNum(g, fmt(v, 1), gx0 - 6, GY(v)));
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => V.text(g, fmt(v, 2), GX(v), gy1 + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'r̄ solve →', gx1, gy1 + 28, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'r propose', gx0 + 6, gy0 - 6, { size: 11, font: 'mono', color: C.muted });
        g.beginPath(); g.moveTo(GX(1 / G), GY(1 - 1 / G)); g.lineTo(GX(7 / G), GY(1 / G)); g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
        for (let k = 0; k <= G; k++) {
          const v = rProp(k), x = GX(k / G), y = GY(v), cur = k === S.k;
          if (cur) { V.glow(g, x, y, 14, C.d, 0.9); V.dot(g, x, y, 6, C.d); }
          else V.dot(g, x, y, 3.5, k === 0 || k === G ? RED() : C.a);
        }
        const cv = rProp(S.k);
        V.text(g, `${S.k}/8 → ${fmt(cv, 3)}`, Math.min(GX(S.k / G) + 12, gx1 - 96), GY(cv) - 8, { size: 12, font: 'mono', weight: 700, color: C.d, align: 'left' });
        // grade TRR++
        const tx0 = narrow ? 70 : w * 0.66 + 40, ty0 = narrow ? h - 112 : gy0 + 26;
        const cwid = narrow ? (w - tx0 - 20) / 2 : Math.min(90, (w - tx0 - 20) / 2), chh = narrow ? 26 : 30;
        V.text(g, 'TRR++: 6 linhas de base', narrow ? 16 : tx0 - 40, ty0 - 12, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
        ['propor', 'resolver'].forEach((cl, j) => V.text(g, cl, tx0 + j * cwid + cwid / 2, ty0 + 6, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        const TK = [['ded', 'dedução'], ['abd', 'abdução'], ['ind', 'indução']];
        TK.forEach((tk, r) => {
          const y = ty0 + 14 + r * (chh + 4);
          V.text(g, tk[1], tx0 - 6, y + chh / 2, { size: 11, font: 'mono', color: tk[0] === S.mode ? C.ink : C.muted, align: 'right', base: 'middle' });
          for (let j = 0; j < 2; j++) {
            const on = tk[0] === S.mode;
            V.box(g, tx0 + j * cwid + 2, y, cwid - 4, chh, { r: 5, fill: on ? V.alpha(j ? C.b : C.a, 0.35) : V.alpha(C.muted, 0.08), stroke: on ? (j ? C.b : C.a) : V.alpha(C.muted, 0.3), lw: on ? 1.6 : 1 });
            const txt = on ? (j ? `r=${fmt(S.k / G, 2)}` : `r=${fmt(cv, 2)}`) : 'μ, σ';
            V.text(g, txt, tx0 + j * cwid + cwid / 2, y + chh / 2 + 0.5, { size: 11, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted, align: 'center', base: 'middle' });
          }
        });
        if (!narrow) V.text(g, 'A = (r − μ) / σ dentro da célula', tx0 - 40, ty0 + 14 + 3 * (chh + 4) + 14, { size: 11, font: 'mono', color: C.muted });
      });
    })();

    /* ================================================ FIG 9 — prover-verifier */
    (function () {
      const fig = root.querySelector('#fig-pvg');
      adaptH(fig, 420, 580);
      const S = { r: 0, hum: true };
      const accS = (r) => 0.62 - 0.104 * r; // verificador aceita sorrateiras (ilustrativo)
      const corrH = (r) => 0.55 + 0.05 * r; // acerto do provador útil (ilustrativo)
      const humH = (r) => 0.62 + 0.04 * r, humS = (r) => 0.7 - 0.05 * r; // humanos com pressa (ilustrativo)
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'rodada de treino', min: 0, max: 5, step: 1, value: 0, onInput: (v) => { S.r = v; upd(); } });
      UI.toggle(ctl, { label: 'mostrar curvas dos humanos', value: true, onChange: (v) => { S.hum = v; } });
      const rA = UI.readout(ctl, 'sorrateiras aceitas', '—'), rH = UI.readout(ctl, 'acerto do útil', '—');
      function upd() { rA.set(fmt(100 * accS(S.r), 0) + ' %', S.r >= 4 ? 'ok' : S.r <= 1 ? 'bad' : 'warn'); rH.set(fmt(100 * corrH(S.r), 0) + ' %'); }
      upd();
      const hash = (n) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const LW = narrow ? w : w * 0.55;
        const LH = narrow ? 280 : h;
        const hx = 40, hy = LH * 0.26, sy = LH * 0.7, vx = LW * (narrow ? 0.48 : 0.62), vy = LH * 0.48;
        buddy(g, hx, hy + Math.sin(t * 2) * 1.5, 40, C.a, { mood: 'happy' });
        buddy(g, hx, sy + Math.sin(t * 2 + 1) * 1.5, 40, C.f, { mood: 'sly' });
        buddy(g, vx, vy, 34, C.d, { hat: true, mood: S.r >= 3 ? 'happy' : 'flat' });
        V.text(g, 'útil', hx, hy + 34, { size: 11, font: 'mono', weight: 600, color: C.a, align: 'center' });
        V.text(g, 'sorrateiro', hx, sy + 34, { size: 11, font: 'mono', weight: 600, color: C.f, align: 'center' });
        V.text(g, 'verificador', vx, vy + 32, { size: 11, font: 'mono', weight: 600, color: C.d, align: 'center' });
        // pilhas
        const okX = LW - 58, okY = LH * (narrow ? 0.26 : 0.2), noY = LH * (narrow ? 0.76 : 0.64);
        V.text(g, 'aceitas', okX, okY - 22, { size: 11, font: 'mono', color: C.a, align: 'center' });
        V.text(g, 'rejeitadas', okX, noY - 22, { size: 11, font: 'mono', color: RED(), align: 'center' });
        // cartões
        const per = 0.9, nVis = 6;
        const curN = Math.floor(t / per);
        for (let q = curN - nVis; q <= curN; q++) {
          if (q < 0) continue;
          const u = (t - q * per) / (per * nVis);
          if (u > 1) continue;
          const sneaky = q % 2 === 1;
          const p = sneaky ? accS(S.r) : Math.min(0.97, 0.85 + 0.02 * S.r);
          const accepted = hash(q + S.r * 1000) < p;
          const sx = hx + 22, syy = sneaky ? sy : hy;
          let x, y;
          if (u < 0.45) { const k = ez(u / 0.45); x = lerp(sx, vx - 26, k); y = lerp(syy, vy, k); }
          else { const k = ez((u - 0.45) / 0.55); x = lerp(vx + 22, okX, k); y = lerp(vy, accepted ? okY : noY, k) + (accepted ? 0 : 0); }
          const col = sneaky ? C.f : C.a;
          V.box(g, x - 11, y - 14, 22, 28, { r: 3, fill: V.alpha(col, 0.22), stroke: col, lw: 1.2 });
          for (let l = 0; l < 3; l++) V.line(g, x - 6, y - 7 + l * 6, x + 6 - (l === 2 ? 5 : 0), y - 7 + l * 6, { color: V.alpha(col, 0.9), lw: 1.4 });
          if (u > 0.45) mark(g, x + 11, y - 13, accepted, 4.5);
        }
        // pilhas das últimas folhas já carimbadas
        const piles = [[], []];
        for (let q = curN - nVis - 15; q <= curN - nVis; q++) {
          const sneaky = ((q % 2) + 2) % 2 === 1;
          const p = sneaky ? accS(S.r) : Math.min(0.97, 0.85 + 0.02 * S.r);
          piles[hash(q + S.r * 1000) < p ? 0 : 1].push(sneaky);
        }
        piles.forEach((pl, side) => {
          const baseY = side ? noY : okY;
          pl.forEach((sn, k) => {
            const cx = okX - 15 + (k % 3) * 15, cy = baseY - 6 + Math.floor(k / 3) * 17;
            const col = sn ? C.f : C.a;
            V.box(g, cx - 6, cy - 7, 12, 15, { r: 2, fill: V.alpha(col, 0.45), stroke: col, lw: 1 });
          });
          const nS = pl.filter((x) => x).length;
          V.text(g, `${pl.length - nS} úteis · ${nS} sorrat.`, narrow ? LW - 8 : okX, baseY + 6 + Math.ceil(Math.max(1, pl.length) / 3) * 17, { size: 10.5, font: 'mono', color: C.muted, align: narrow ? 'right' : 'center' });
        });
        const head = `rodada ${S.r}: o verificador aceita ~${fmt(100 * accS(S.r), 0)} % das sorrateiras`;
        if (narrow) wrapC(g, head, 12, 16, LW - 24, { size: 11.5, font: 'mono', weight: 600, lineH: 15, maxLines: 2 });
        else V.text(g, head, 12, 16, { size: 11.5, font: 'mono', weight: 600, color: C.ink, maxW: LW - 20 });
        // gráfico
        const cx0 = narrow ? 50 : LW + 52, cx1 = w - 18, cy0 = narrow ? LH + 8 : 40, cy1 = h - 40;
        const X = (r) => cx0 + (r / 5) * (cx1 - cx0), Y = (v) => cy1 - v * (cy1 - cy0 - 66);
        V.line(g, cx0, cy1, cx1, cy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.line(g, cx0, cy0, cx0, cy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        [0, 0.5, 1].forEach((v) => axisNum(g, fmt(v, 1), cx0 - 6, Y(v)));
        for (let r = 0; r <= 5; r++) V.text(g, String(r), X(r), cy1 + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, 'rodada de treino →', cx1, cy1 + 30, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const series = [[accS, C.f, 'verificador aceita sorrateiras', []], [corrH, C.a, 'acerto do provador útil', []]];
        if (S.hum) { series.push([humH, C.b, 'humanos acertam no útil', [5, 3]]); series.push([humS, C.c, 'humanos acertam no sorrateiro', [5, 3]]); }
        series.forEach(([f, col, lab, dash], k) => {
          g.beginPath(); for (let r = 0; r <= 5; r++) { const x = X(r), y = Y(f(r)); if (r) g.lineTo(x, y); else g.moveTo(x, y); }
          g.strokeStyle = col; g.lineWidth = 2; if (dash.length) g.setLineDash(dash); g.stroke(); g.setLineDash([]);
          V.dot(g, X(S.r), Y(f(S.r)), 4.5, col);
          const ly = cy0 + 4 + k * 15; // legenda com amostra do traço (cheio ou tracejado)
          V.line(g, cx0 + 8, ly - 4, cx0 + 24, ly - 4, { color: col, lw: 2, dash: dash.length ? [4, 2] : undefined });
          V.text(g, lab, cx0 + 30, ly, { size: 11, font: 'mono', color: col, maxW: cx1 - cx0 - 32 });
        });
        V.line(g, X(S.r), cy0 + 60, X(S.r), cy1, { color: V.alpha(C.d, 0.5), lw: 1.2, dash: [3, 3] });
        V.text(g, 'formas qualitativas; números ilustrativos', cx1, cy1 - 8, { size: 11, font: 'mono', color: C.muted, align: 'right', maxW: cx1 - cx0 });
      });
    })();

    /* ================================================ FIG 10 — colapso autorreferente */
    (function () {
      const fig = root.querySelector('#fig-colapso');
      adaptH(fig, 450, 640);
      const S = { b: 2, anchor: false };
      const M = 64, KMAX = 24;
      const u = (x) => 1 - 6 * (x - 0.3) * (x - 0.3);
      const skill = (k) => 0.12 * (1 - Math.exp(-k / 5));
      let sim = null;
      function init() {
        const r = V.rng(17);
        const xs = Array.from({ length: M }, () => clamp(0.08 + 0.05 * r.normal(), 0, 1));
        sim = { xs, prev: xs.slice(), k: 0, acc: 0, hold: 0, hist: [], r: V.rng(23), sd0: 0 };
        sim.sd0 = sd(xs);
        record();
      }
      function sd(xs) { const m = xs.reduce((a, b) => a + b, 0) / xs.length; return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length); }
      const beff = () => (S.anchor ? 0.25 * S.b : S.b);
      const J = (x) => u(x) + beff() * x;
      function record() {
        const xs = sim.xs, k = sim.k;
        const gold = xs.reduce((a, x) => a + u(x), 0) / M + skill(k);
        const jud = xs.reduce((a, x) => a + J(x), 0) / M + skill(k);
        const div = clamp(sd(xs) / sim.sd0, 0, 1.5);
        sim.hist.push({ gold, jud, div });
      }
      function step() {
        const xs = sim.xs.slice();
        sim.prev = sim.xs.slice();
        const order = xs.map((x, i) => [J(x) + 0.004 * sim.r.normal(), i]).sort((a, b) => b[0] - a[0]);
        const nSel = Math.floor(M * 0.3);
        const mut = S.anchor ? 0.045 : 0.016;
        for (let q = 0; q < nSel; q++) {
          const top = xs[order[q][1]], bot = order[M - 1 - q][1];
          xs[bot] = clamp(top + mut * sim.r.normal(), 0, 1);
        }
        for (let i = 0; i < M; i++) xs[i] = clamp(xs[i] + mut * 0.5 * sim.r.normal(), 0, 1);
        if (S.anchor) for (let i = 0; i < M; i++) xs[i] = clamp(xs[i] + 0.08 * (0.3 - xs[i]) * 0.3, 0, 1);
        sim.xs = xs; sim.k++;
        record();
      }
      init();
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'viés de autopreferência b', min: 0, max: 3, step: 0.1, value: 2, fmt: (v) => fmt(v, 1), onInput: (v) => { S.b = v; init(); } });
      UI.toggle(ctl, { label: 'âncora externa', value: false, onChange: (v) => { S.anchor = v; init(); } });
      UI.button(ctl, '↻ reiniciar', () => init());
      const rJ = UI.readout(ctl, 'nota do juiz', '—'), rG = UI.readout(ctl, 'qualidade ouro', '—'), rD = UI.readout(ctl, 'diversidade', '—');
      ctl.querySelectorAll('.la-ctl label').forEach((l) => { l.style.textTransform = 'none'; });

      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        sim.acc += dt;
        if (sim.k < KMAX) { if (sim.acc > 0.42) { sim.acc = 0; step(); } }
        else { sim.hold += dt; if (sim.hold > 3) init(); }
        const morph = ez(sim.acc / 0.3);
        const H0 = sim.hist[sim.hist.length - 1];
        const peak = Math.max(...sim.hist.map((q) => q.gold));
        rJ.set(fmt(H0.jud, 2)); rG.set(fmt(H0.gold, 3), H0.gold < peak - 0.02 ? 'bad' : 'ok'); rD.set(fmt(100 * Math.min(1, H0.div), 0) + ' %', H0.div < 0.5 ? 'bad' : undefined);
        const narrow = w < 600;
        // dispersão
        const px0 = 50, px1 = narrow ? w - 16 : w * 0.56, py0 = 34, py1 = narrow ? h * 0.5 : h - 34;
        const XM = 0.7;
        const X = (x) => px0 + (x / XM) * (px1 - px0);
        const qMin = 0.2, qMax = 1.2;
        const Y = (q) => py1 - ((q - qMin) / (qMax - qMin)) * (py1 - py0);
        V.line(g, px0, py1, px1, py1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.line(g, px0, py0, px0, py1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.text(g, 'estilo que o juiz adora (verbosidade, sotaque) →', px1, py1 + 18, { size: 11, font: 'mono', color: C.muted, align: 'right', maxW: px1 - px0 });
        V.text(g, 'qualidade real', px0 + 6, py0 - 10, { size: 11, font: 'mono', color: C.a });
        const sk = skill(sim.k);
        g.save(); g.beginPath(); g.rect(px0 + 1, py0 - 6, px1 - px0, py1 - py0 + 5); g.clip();
        g.beginPath(); for (let k = 0; k <= 80; k++) { const x = (k / 80) * XM, q = u(x) + sk; if (k) g.lineTo(X(x), Y(q)); else g.moveTo(X(x), Y(q)); }
        g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
        // forma da nota do juiz (escala própria)
        const jv = Array.from({ length: 81 }, (_, k) => J((k / 80) * XM));
        const jmn = Math.min(...jv), jmx = Math.max(...jv);
        g.beginPath(); jv.forEach((v, k) => { const x = X((k / 80) * XM), y = lerp(py1 - 10, py0 + 18, (v - jmn) / (jmx - jmn || 1)); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = V.alpha(C.d, 0.85); g.lineWidth = 1.8; g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]);
        g.restore();
        V.text(g, 'nota do juiz (escala própria)', px1, py0 + 8, { size: 11, font: 'mono', color: C.d, align: 'right' });
        const xJ = clamp(0.3 + beff() / 12, 0, XM);
        // bandeira e estrela
        const fx = X(xJ), fy = py1;
        V.line(g, fx, fy, fx, fy - 26, { color: C.d, lw: 1.6 });
        g.fillStyle = C.d; g.beginPath(); g.moveTo(fx, fy - 26); g.lineTo(fx + 12, fy - 21); g.lineTo(fx, fy - 16); g.closePath(); g.fill();
        const sx = X(0.3), sy = Y(1 + sk) - 12;
        g.fillStyle = C.a; g.beginPath(); for (let k = 0; k < 10; k++) { const r = k % 2 ? 3 : 7, an = -Math.PI / 2 + (k * Math.PI) / 5; g.lineTo(sx + r * Math.cos(an), sy + r * Math.sin(an)); } g.closePath(); g.fill();
        // partículas
        for (let i = 0; i < M; i++) {
          const x = lerp(sim.prev[i] ?? sim.xs[i], sim.xs[i], morph);
          const jit = ((i * 37) % 11 - 5) * 0.004;
          V.dot(g, X(x), Y(u(x) + sk + jit), 3.2, V.alpha(C.b, 0.75));
        }
        V.text(g, `· iteração ${sim.k}/${KMAX}`, px0 + 14 + tw(g, 'qualidade real', 11, 'mono'), py0 - 10, { size: 11, font: 'mono', color: C.muted });
        // séries
        const sx0 = narrow ? 50 : w * 0.64, sx1 = w - 16, sy0 = narrow ? h * 0.5 + 64 : 30, sy1 = h - 22;
        const rowH = (sy1 - sy0) / 3;
        const SER = [['jud', 'nota do juiz', C.d, 0.6, 2.5], ['gold', 'qualidade ouro', C.a, 0.7, 1.15], ['div', 'diversidade', C.b, 0, 1.1]];
        SER.forEach(([key, lab, col, lo, hi], r) => {
          const y0 = sy0 + r * rowH + 16, y1 = sy0 + (r + 1) * rowH - 8;
          V.text(g, lab, sx0, y0 - 4, { size: 11, font: 'mono', weight: 600, color: col });
          V.line(g, sx0, y1, sx1, y1, { color: V.alpha(C.muted, 0.4), lw: 1 });
          const Xs = (k) => sx0 + (k / KMAX) * (sx1 - sx0), Ys = (v) => y1 - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (y1 - y0 - 4);
          g.beginPath(); sim.hist.forEach((q, k) => { if (k) g.lineTo(Xs(k), Ys(q[key])); else g.moveTo(Xs(k), Ys(q[key])); });
          g.strokeStyle = col; g.lineWidth = 2; g.stroke();
          const last = sim.hist[sim.hist.length - 1];
          V.dot(g, Xs(sim.hist.length - 1), Ys(last[key]), 3.5, col);
          if (key === 'gold') {
            const kp = sim.hist.findIndex((q) => q.gold === peak);
            if (kp >= 0 && kp < sim.hist.length - 2 && last.gold < peak - 0.02) { V.dot(g, Xs(kp), Ys(peak), 3.5, C.ink); V.text(g, 'pico', Xs(kp), Ys(peak) - 8, { size: 11, font: 'mono', color: C.ink, align: 'center' }); }
          }
          V.text(g, key === 'div' ? fmt(100 * Math.min(1, last[key]), 0) + ' %' : fmt(last[key], 2), sx1, y0 - 4, { size: 11, font: 'mono', color: C.ink, align: 'right' });
        });
      });
    })();
  },
});

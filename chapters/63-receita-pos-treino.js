LA.chapter({
  id: 'receita-pos-treino',
  part: 13,
  order: 63,
  title: 'Receita: pós-treinar, quantizar e servir',
  navTitle: 'Receita de pós-treino',
  subtitle: 'Do glossário ao endpoint: dados com professor e juiz do Módulo 1, SFT, DPO e GRPO no Mac e em dois DGX Spark, merge, re-quantização calibrada, duas KLDs e serviço no vLLM, com comandos, versões, orçamentos e portões.',
  minutes: 45,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['receita', 'LoRA', 'QLoRA', 'MLX', 'TRL', 'DDP', 'merge', 'NVFP4', 'KLD', 'multi-LoRA', 'DGX Spark'],
  goals: [
    'Montar os dados de um pós-treino com as receitas do Módulo 1 como <b>professor e juiz</b> (troca de posição, margem mínima, descontaminação, split por termo) e escrever o JSONL certo para cada trainer',
    'Rodar SFT, DPO e GRPO com LoRA no <b>Mac M4 (MLX)</b> e em <b>2× DGX Spark (TRL)</b>, com orçamento de memória e tempo feitos à mão e a LR convertida entre as escalas de LoRA',
    'Fundir o adapter sem perder o delta, preservar a MTP e <b>re-quantizar com calibração nova</b> (NVFP4, EXL3, DWQ), sabendo o que fica em alta precisão',
    'Separar as <b>duas KLDs</b> (deriva do pós-treino e estrago da quantização) e servir o resultado no vLLM, com multi-LoRA ou fundido, medindo a aceitação do rascunhador',
  ],
  prereqs: ['sft', 'dpo', 'grpo-rlvr', 'avaliacao-pos-treino', 'receita-do-zero'],
  summary: [
    'A receita tem dez fases (F0–F9), cada uma com um artefato e um portão. Os modelos do Módulo 1 não cabem para treino (GLM ≈ 643 GB em BF16), então o laboratório usa o <b>Qwen3.8-27B</b>, da mesma família do Qwen3.8-Flash-Next, e o Qwen3-4B no Mac.',
    'Dados: professor GLM com 4 respostas por pergunta, juiz <b>DeepSeek</b> (a única família fora da do professor e da do aluno), torneio julgado nas duas ordens (inconsistência = empate), SFT com a melhor, DPO com (melhor, pior) se a margem for ≥ 2, descontaminação por 13-gramas e split por termo. O template faz parte do dado: <code>reasoning_content</code>, <code>assistant_only_loss</code> e o mesmo <code>enable_thinking</code> em treino, avaliação e servidor.',
    'Mac: QLoRA no <code>mlx_lm.lora</code> (≈ 7–10 GB e 2,5–5 h [estimativa]). A escala do LoRA no MLX multiplica \\(BA\\) direto (20 ou 10), então LR 1e-5 × 20 ≈ PEFT 2e-4. DPO no <code>mlx-lm-lora</code> sobre o SFT fundido em 8 bits (duas cópias, 11–14 GB), e sem KTO.',
    'Sparks: o 27B BF16 (55,6 GB) cabe em cada nó, então <b>DDP</b>: só 0,87 GB de gradientes LoRA cruzam o RoCE por passo, contra ~1,3 TB de all-gather do FSDP (≈ 83 GB por micro-lote, ×16). SFT r=32 (217,6 M parâmetros) leva 2,7–5,4 h a 6N FLOPs/token; DPO sobre o SFT fundido; GRPO com o Spark B gerando no vLLM e o Spark A treinando (2,2–4,3 min/passo com β > 0), os dois na imagem do vLLM.',
    'Vigie as assinaturas: <code>eval_loss</code> subindo (SFT), <code>logps/chosen</code> despencando enquanto a margem sobe (DPO, likelihood displacement), <code>frac_reward_zero_std</code> → 1 e comprimento explodindo (GRPO). Portões pré-registrados, com IC por bootstrap (±4,9 pp com 400 prompts).',
    'Merge em BF16 (nunca sobre o 4-bit com RTN: deltas menores que meio degrau somem, e a grade recalculada ainda soma ruído), MTP re-adicionada num shard próprio e merge verificado por logits. Re-quantização calibrada no modelo fundido com mistura de domínio e genérico; MLPs em NVFP4, atenção/GDN e as 8 últimas MLPs em FP8, <code>lm_head</code> e MTP fora.',
    'Duas KLDs: KL(base‖ajustado) mede a deriva; KL(merge BF16‖quantizado) mede o estrago, comparado ao do NVFP4 de referência e ao piso P contra P. Produção = fundido NVFP4 (~9–10 tok/s de teto por stream). Multi-LoRA para A/B: uma GEMM do base para o batch e, por token, shrink e expand nas fatias do slot \\(\\sigma(t)\\); <code>--max-loras</code> é o número de fatias e <code>--max-lora-rank</code> a largura delas. Meça a aceitação do MTP, que foi treinado para o base.',
  ],
  quiz: [
    { q: 'No Mac, por que fundir o adapter do SFT num checkpoint de <b>8 bits</b> (a partir do BF16) antes do DPO, em vez de fundi-lo direto sobre o base de 4 bits?', o: ['Porque o DPO só aceita modelos de 8 bits', 'Porque fundir sobre o 4-bit re-quantiza com RTN (mesmos bits, escala recalculada): deltas menores que meio degrau da grade de 4 bits se perdem, o re-arredondamento soma ruído, e a referência do DPO perderia parte do SFT', 'Porque o 8-bit é mais rápido que o 4-bit no M4', 'Porque o mlx-lm-lora não carrega modelos de 4 bits'], a: 1, e: 'O delta só chega ao peso quando \\(|\\Delta W| \\gtrsim \\Delta/2\\). No modo afim do MLX, com 4 bits o degrau é 1/15 da faixa (máximo − mínimo) do grupo; com 8 bits é 1/255, cerca de 17× menor. E, como o fuse recalcula a escala, valores que já estavam na grade antiga são re-arredondados numa grade deslocada. O SFT fundido vira a referência do DPO, e perder o delta seria ancorar o DPO num modelo que não é o pós-SFT. Duas cópias de 8 bits (≈ 4,3 GB cada) ainda cabem nos 19,07 GB.' },
    { q: 'Você quantizou o modelo final para NVFP4 e quer saber quanto a quantização estragou. Qual referência usar no KL?', o: ['O base original BF16, porque é o modelo oficial', 'O modelo ajustado, fundido em BF16', 'O NVFP4 de referência da NVIDIA', 'Tanto faz: o KL é simétrico'], a: 1, e: 'Contra o base, a deriva do pós-treino (grande no domínio, por definição) entra na conta e o NVFP4 leva a culpa. A referência certa é o merge BF16 do modelo ajustado. O NVFP4 de referência serve para outra coisa: calibrar a expectativa, medindo KL(base‖NVFP4 de referência) com o mesmo script. E o KL não é simétrico, por isso importa qual modelo é o P.' },
    { q: 'Por que a trilha B usa DDP, e não FSDP, para o SFT LoRA do 27B em dois Sparks?', o: ['Porque o FSDP não funciona em aarch64', 'Porque o modelo BF16 (55,6 GB) cabe inteiro em cada nó: com DDP só os gradientes do LoRA (0,87 GB em FP32) cruzam o cabo por passo, enquanto o FSDP faria all-gather dos pesos em cada forward, backward e recomputação (~83 GB por micro-lote e direção, ~1,3 TB por passo de otimizador com acumulação 16)', 'Porque o DDP economiza memória de ativações', 'Porque o TRL só aceita DDP com LoRA'], a: 1, e: 'Fatiar o modelo só compensa quando ele não cabe. Aqui cabe com folga (≈ 70–80 GB de 121,7 GiB por nó), e o DDP troca ~1,3 TB de all-gather por passo (≈ 67 s de cabo, tanto quanto o compute) por menos de 1 GB de all-reduce. O playbook multi-nó da NVIDIA usa FSDP porque mira modelos maiores, como o 70B em LoRA.' },
    { q: 'No <code>mlx_lm.lora</code>, a LR é 1e-5 com <code>scale</code> 20. Que LR aproximada daria o mesmo passo efetivo no PEFT com \\(\\alpha/r = 1\\)?', o: ['1e-5', '5e-7', '2e-4', '2e-5'], a: 2, e: 'Com Adam, o passo efetivo do produto \\(sBA\\) escala com \\(s \\times \\eta\\) (inferência da pesquisa): 1e-5 × 20 = 2e-4 × 1. É por isso que a LR não pode ser copiada entre frameworks, e que o DPO do Mac usa 1e-6 com scale 10 (≈ 1e-5 em escala PEFT, o valor que o TRL sugere para adapters em DPO).' },
    { q: 'O vLLM sobe o modelo fundido, mas falha ao ligar <code>--speculative-config</code> com <code>"method":"mtp"</code>. Qual a causa mais provável?', o: ['O NVFP4 não suporta especulação', 'O transformers descartou os tensores <code>mtp.*</code> ao carregar o modelo, e o merge salvou o checkpoint sem eles; é preciso re-adicioná-los num shard próprio e reescrever o índice', 'O <code>num_speculative_tokens</code> tem que ser igual ao posto do LoRA', 'O adapter LoRA ainda está carregado no servidor'], a: 1, e: 'O Qwen3.5/3.8 tem <code>_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]</code>. O <code>RedHatAI/Qwen3.8-27B-NVFP4</code> resolve com 15 tensores num <code>model_mtp.safetensors</code> separado e o índice reescrito, a mesma técnica do <code>dense_overlay.py</code> da receita GLM. E lembre que essa MTP foi treinada para o base: meça a aceitação depois do fine-tune.' },
    { q: 'No torneio de dados, o professor é o GLM e o aluno é o Qwen3.8-27B. Por que o juiz é o DeepSeek, e o que acontece com um par em que o juiz muda de veredito quando as posições trocam?', o: ['Porque o DeepSeek é o maior modelo; o par conta como vitória da posição A', 'Para evitar autopreferência, já que é a única família diferente da do professor e da do aluno; o par vale empate, o que reduz o rendimento mas mantém os pares limpos', 'Porque o GLM não tem API OpenAI; o par é descartado do dataset inteiro', 'Por causa da licença; o juiz é chamado de novo até concordar'], a: 1, e: 'A autopreferência faz o juiz favorecer textos da própria família, e o Qwen3.8-27B é parente do Qwen3.8-Flash-Next. O viés de posição é forte (66 de 80 consultas viradas só reordenando as respostas, em Wang et al.), e a troca transforma inconsistência em empate. Como os saldos somam zero, a margem mínima de 2 só barra as perguntas em que tudo empatou; subir a margem para 3 ou 4 filtra pares ambíguos de verdade, ao custo de rendimento.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('.', ',');
    const pct = (x, d = 0) => fmt(100 * x, d) + '%';
    const RED = () => V.C().red || '#FB2C36';
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    /* ------------------------------------------------------------ helpers */
    function adaptH(fig, wide, narrow, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const val = (x) => (typeof x === 'function' ? x() : x);
      const set = () => { const want = val(fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      return set;
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    function fit(g, s, maxW, size, font = 'sans', weight = 400) {
      let t = String(s);
      if (tw(g, t, size, font, weight) <= maxW) return t;
      while (t.length > 1 && tw(g, t + '…', size, font, weight) > maxW) t = t.slice(0, -1);
      return t + '…';
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) {
        const lig = (o.font || 'mono') === 'mono' && /[<>|]/.test(label);
        if (lig) g.letterSpacing = '0.4px';
        V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle' });
        if (lig) g.letterSpacing = '0px';
      }
    }
    function tokChar(g, x, y, s, col, o = {}) {
      const C = V.C();
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: V.alpha(col, 0.9), stroke: col, lw: 1.2 });
      const eye = C.dark ? '#06140d' : '#ffffff';
      const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
      V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
      g.beginPath();
      if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
      else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
      g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
    }
    function check(g, x, y, s, col) {
      g.lineWidth = 2.2; g.strokeStyle = col; g.lineCap = 'round'; g.beginPath();
      g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); g.stroke(); g.lineCap = 'butt';
    }
    function star(g, x, y, r, col, filled) {
      g.beginPath();
      for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? r * 0.45 : r; g.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
      g.closePath();
      if (filled) { g.fillStyle = col; g.fill(); } else { g.strokeStyle = V.alpha(col, 0.5); g.lineWidth = 1; g.stroke(); }
    }
    function lamp(g, x, y, state, t) {
      const C = V.C();
      const col = state === 'bad' ? RED() : state === 'warn' ? C.d : C.a;
      if (state === 'bad') V.glow(g, x, y, 14 + Math.sin(t * 6) * 2, col, 0.8);
      V.dot(g, x, y, 5.5, col);
    }

    /* ======================================== FIG 1 — o trilho das dez fases */
    (function () {
      const fig = root.querySelector('#fig-mapa');
      if (!fig) return;
      adaptH(fig, 390, 600);
      const PH = [
        { k: 'F0', n: 'linha de base', at: 'ms', art: 'test_dominio.jsonl (300–500)', d: 'Congele a régua antes de mexer no modelo: <b>test_dominio.jsonl</b> (300–500 perguntas de termos fora do treino, gabarito revisado à mão em ~50 itens) e a bateria geral (IFEval, GSM8K, XSTest) rodada no base. <b>Portão:</b> números do base gravados com o mesmo template, o mesmo <code>enable_thinking</code> e o mesmo id de modelo que o candidato vai usar.' },
        { k: 'F1', n: 'dados', at: 's', art: 'sft · pref · kto · grpo .jsonl', d: 'As receitas do Módulo 1 viram professor (GLM, 4 respostas por pergunta) e juiz (DeepSeek, duas ordens). Torneio, descontaminação por 13-gramas e split por termo. <b>Portão:</b> zero sobreposição com os conjuntos de avaliação e <code>valid</code> separado.' },
        { k: 'F2', n: 'SFT (LoRA)', at: 'ms', art: 'adapter SFT · r=32 · ~435 MB', d: 'Mac: QLoRA 4-bit, r=16, últimas 16 camadas. Sparks: LoRA BF16 em DDP, r=32 (217,6 M parâmetros), 2,7–5,4 h <b>[estimativa]</b>. <b>Portão:</b> <code>eval_loss</code> cai e o IFEval não cai além do IC.' },
        { k: 'F3', n: 'DPO', at: 'ms', art: 'adapter DPO · ~435 MB', d: 'Funda o SFT (ele vira a referência) e treine o DPO como um novo LoRA: β = 0,1, LR 5e-6 nos Sparks e 1e-6 (× scale 10) no Mac. <b>Portão:</b> <code>rewards/accuracies</code> abaixo de ~0,95 e <code>logps/chosen</code> sem queda contínua.' },
        { k: 'F4', n: 'GRPO', at: 'ms', art: 'adapter GRPO · 9B · r=16', d: 'Recompensa verificável no Qwen3.5-9B: o Spark B gera com vLLM, o Spark A treina e empurra os pesos pelo NCCL (2,2–4,3 min por passo <b>[estimativa]</b>), os dois na imagem do vLLM. No Mac, um 1,7B. <b>Portão:</b> <code>frac_reward_zero_std</code> abaixo de ~0,5 e amostras lidas.' },
        { k: 'F5', n: 'portões', at: 'ms', art: 'relatório com IC95%', d: 'Gere e grave todas as respostas, depois suba a receita-juiz. WR-LC contra o base com IC por bootstrap, regressão, segurança e deriva. <b>Portão:</b> limite inferior do IC do WR-LC acima de 50%, Δ IFEval ≥ −2 pp, recusa nos prompts seguros sem subir mais que 2 pp.' },
        { k: 'F6', n: 'merge', at: 'ms', art: 'q38-final-bf16 · 55,6 GB', d: 'Funda em BF16 (nunca sobre o NF4), salve template e processor e re-adicione os 15 tensores <code>mtp.*</code>. <b>Portão:</b> logits de base + adapter contra o fundido, com top-1 igual em ≥ 99,9% das posições.' },
        { k: 'F7', n: 're-quantizar', at: 'ms', art: 'q38-final-nvfp4 · ~24,7 GB', d: 'Calibre de novo, no modelo fundido: 256 amostras do domínio + 256 genéricas. MLPs em NVFP4; atenção, GDN e as 8 últimas MLPs em FP8; <code>lm_head</code> e MTP fora. <b>Portão:</b> formatos e tamanho conferidos no <code>config.json</code>.' },
        { k: 'F8', n: 'KLD', at: 's', art: 'KL vs merge BF16 + piso', d: 'Duas KLDs: a deriva (base → ajustado) e o estrago (merge BF16 → NVFP4), mais o piso P contra P. <b>Portão:</b> KLD do seu NVFP4 da ordem da do NVFP4 de referência no base, e os portões da F5 repetidos no quantizado.' },
        { k: 'F9', n: 'servir', at: 'ms', art: 'tutor-q38 · porta 8000', d: 'vLLM 0.30.0 com NVFP4, KV FP8, prefix caching e MTP, ou adapters em runtime para A/B. No Mac, <code>mlx_lm.server</code>. <b>Portão:</b> aceitação do MTP medida, <code>vllm bench serve</code> A/B/A e um plano de rollback.' },
      ];
      const S = { u: 0 };
      const st = UI.stepper(fig, PH.map((p) => ({ t: p.k + ' · ' + p.n, d: p.d })), { autoplay: 7000 });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const cols = narrow ? 2 : 5, rows = Math.ceil(PH.length / cols);
        const bagH = narrow ? 150 : 94;
        const top = 34, bot = h - bagH - 20;
        const rowH = (bot - top) / rows, colW = (w - 32) / cols;
        const bw = Math.min(150, colW - 18), bh = 40;
        const P = PH.map((p, i) => { const r = Math.floor(i / cols); let c = i % cols; if (r % 2) c = cols - 1 - c; return [16 + colW * (c + 0.5), top + rowH * (r + 0.5) - 8]; });
        S.u += (st.i - S.u) * Math.min(1, dt * 2.6);
        if (Math.abs(st.i - S.u) < 0.003) S.u = st.i;
        const at = (u) => { const i = Math.max(0, Math.min(PH.length - 2, Math.floor(u))); const f = Math.max(0, Math.min(1, u - i)); return [V.lerp(P[i][0], P[i + 1][0], f), V.lerp(P[i][1], P[i + 1][1], f)]; };
        // trilho
        g.lineCap = 'round'; g.lineJoin = 'round';
        g.beginPath(); P.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
        g.strokeStyle = V.alpha(C.muted, 0.2); g.lineWidth = 8; g.stroke();
        g.setLineDash([3, 9]); g.lineDashOffset = -t * 16; g.strokeStyle = V.alpha(C.muted, 0.55); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(P[0][0], P[0][1]);
        for (let i = 1; i <= Math.floor(S.u) && i < P.length; i++) g.lineTo(P[i][0], P[i][1]);
        const cp = at(S.u); g.lineTo(cp[0], cp[1]);
        g.strokeStyle = C.a; g.lineWidth = 4; g.stroke();
        g.lineCap = 'butt';
        // estações
        PH.forEach((p, i) => {
          const cx = P[i][0], cy = P[i][1], x = cx - bw / 2, y = cy - bh / 2;
          const isCur = i === st.i, done = i < st.i;
          if (isCur) V.glow(g, cx, cy, bw * 0.62, C.a, 0.3);
          V.box(g, x, y, bw, bh, { r: 10, fill: C.bg });
          V.box(g, x, y, bw, bh, { r: 10, fill: isCur ? V.alpha(C.a, 0.22) : done ? V.alpha(C.a, 0.09) : V.alpha(C.muted, 0.07), stroke: isCur ? C.a : done ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.35), lw: isCur ? 2 : 1.2 });
          V.text(g, p.k, x + 10, cy + 0.5, { size: 12, font: 'mono', weight: 700, color: isCur || done ? C.a : C.muted, base: 'middle' });
          V.text(g, fit(g, p.n, bw - (done ? 58 : 44), 12.5, 'sans', 600), x + 36, cy + 0.5, { size: 12.5, weight: 600, color: C.ink, base: 'middle' });
          if (done) check(g, x + bw - 14, cy, 5, C.a);
          const tags = p.at === 'ms' ? [['Mac', C.b], ['Spark', C.a]] : [['Spark', C.a]];
          let tx = cx - (tags.length * 48 - 6) / 2;
          tags.forEach(([lb, col]) => { chip(g, tx, y + bh + 5, 42, 16, lb, { fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.55), size: 10.5, font: 'sans', weight: 600 }); tx += 48; });
        });
        // personagem
        const moving = Math.abs(st.i - S.u) > 0.02;
        const bob = moving ? Math.abs(Math.sin(t * 10)) * 4 : Math.sin(t * 2) * 1.2;
        tokChar(g, cp[0], cp[1] - bh / 2 - 12 - bob, 20, C.a);
        // sacola
        const bx = 16, by = h - bagH - 6, bwid = w - 32;
        V.box(g, bx, by, bwid, bagH, { r: 12, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
        V.text(g, 'sacola de artefatos (trilha B)', bx + 12, by + 16, { size: 11, color: C.muted, weight: 600 });
        const items = PH.slice(0, st.i + 1).map((p, i) => ({ s: p.k + '  ' + p.art, i, col: p.at === 's' ? C.a : C.b }));
        const chH = 20, gap = 6, maxLines = Math.max(1, Math.floor((bagH - 30) / (chH + gap)));
        const lay = (arr) => { const out = []; let x = bx + 10, y = by + 26, line = 0; for (const it of arr) { const cw = Math.min(bwid - 20, tw(g, it.s, 11, 'sans', 500) + 16); if (x + cw > bx + bwid - 10 && x > bx + 10) { x = bx + 10; y += chH + gap; line++; } out.push({ it, x, y, cw }); x += cw + gap; } return { out, lines: line + 1 }; };
        let list = items, L = lay(list);
        while (L.lines > maxLines && list.length > 1) { list = list.slice(1); L = lay(list); }
        L.out.forEach(({ it, x, y, cw }) => {
          const k = it.i === st.i ? V.ease.back(Math.min(1, st.elapsed() / 0.5)) : 1;
          if (k <= 0.01) return;
          g.save(); g.translate(x + cw / 2, y + chH / 2); g.scale(k, k);
          chip(g, -cw / 2, -chH / 2, cw, chH, fit(g, it.s, cw - 12, 11, 'sans', 500), { fill: V.alpha(it.col, 0.14), stroke: V.alpha(it.col, 0.5), size: 11, font: 'sans', weight: 500 });
          g.restore();
        });
      });
    })();

    /* ======================================== FIG 2 — o torneio do juiz */
    (function () {
      const fig = root.querySelector('#fig-torneio');
      if (!fig) return;
      adaptH(fig, 440, 472);
      const PAIRS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
      const S = { bias: 0.25, swap: true, margin: 2, qn: 0, clock: 6.8, q: null, res: null, stats: null };
      function judge(qx, qy, r, bias) {
        if (r() < bias) return 'A';
        if (r() < 0.08) return 'E';
        const p = 1 / (1 + Math.exp(-(qx - qy) / 0.5));
        return r() < p ? 'A' : 'B';
      }
      function play(q, r, bias, swap, M) {
        const sc = [0, 0, 0, 0], rec = [];
        PAIRS.forEach(([i, j]) => {
          const v1 = judge(q[i], q[j], r, bias);
          const v2 = swap ? judge(q[j], q[i], r, bias) : null;
          let s;
          if (swap) s = v1 === 'A' && v2 === 'B' ? 1 : v1 === 'B' && v2 === 'A' ? -1 : 0;
          else s = v1 === 'A' ? 1 : v1 === 'B' ? -1 : 0;
          sc[i] += s; sc[j] -= s; rec.push({ i, j, v1, v2, s });
        });
        let best = 0, worst = 0;
        for (let k = 1; k < 4; k++) { if (sc[k] > sc[best]) best = k; if (sc[k] < sc[worst]) worst = k; }
        return { sc, rec, best, worst, sft: sc[best] >= 1, pair: sc[best] - sc[worst] >= M, pure: q[best] > q[worst] };
      }
      function newQuestion() { const r = V.rng(101 + S.qn * 7919); S.q = [0, 1, 2, 3].map(() => r.normal()); S.res = play(S.q, r, S.bias, S.swap, S.margin); }
      function simStats() {
        const out = {};
        [true, false].forEach((sw) => {
          const r = V.rng(2024); let pairs = 0, pure = 0; const N = 2000;
          for (let n = 0; n < N; n++) { const q = [r.normal(), r.normal(), r.normal(), r.normal()]; const res = play(q, r, S.bias, sw, S.margin); if (res.pair) { pairs++; if (res.pure) pure++; } }
          out[sw ? 'on' : 'off'] = { y: pairs / N, p: pairs ? pure / pairs : 0 };
        });
        S.stats = out;
      }
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'viés de posição do juiz', min: 0, max: 0.6, step: 0.05, value: S.bias, fmt: (v) => fmt(v, 2), onInput: (v) => { S.bias = v; newQuestion(); simStats(); upd(); } }));
      UI.toggle(ctl, { label: 'trocar posições (A/B e B/A)', value: true, onChange: (v) => { S.swap = v; newQuestion(); upd(); } });
      nocase(UI.slider(ctl, { label: 'margem mínima do par', min: 2, max: 5, step: 1, value: S.margin, onInput: (v) => { S.margin = v; newQuestion(); simStats(); upd(); } }));
      UI.button(ctl, 'nova pergunta', () => { S.qn++; S.clock = 0; newQuestion(); });
      const rY = nocase(UI.readout(ctl, 'perguntas que geram par', '—')), rP = nocase(UI.readout(ctl, 'pureza com troca', '—')), rN = nocase(UI.readout(ctl, 'pureza sem troca', '—'));
      function upd() {
        const a = S.stats.on, b = S.stats.off;
        rY.set(pct(S.swap ? a.y : b.y));
        rP.set(pct(a.p), a.p > 0.9 ? 'ok' : a.p > 0.8 ? 'warn' : 'bad');
        rN.set(pct(b.p), b.p > 0.9 ? 'ok' : b.p > 0.8 ? 'warn' : 'bad');
      }
      newQuestion(); simStats(); upd();
      const nStars = (q) => Math.max(1, Math.min(5, Math.round(3 + q * 1.2)));

      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        S.clock += dt;
        if (S.clock > 9.4) { S.clock = 0; S.qn++; newQuestion(); }
        const u = S.clock;
        const k = u < 6.6 ? Math.floor(u / 1.1) : 6;
        const kf = u < 6.6 ? (u % 1.1) / 1.1 : 1;
        const R = S.res;
        const sc = [0, 0, 0, 0];
        R.rec.forEach((rc, idx) => { if (idx < k || (idx === k && kf > 0.72)) { sc[rc.i] += rc.s; sc[rc.j] -= rc.s; } });
        const narrow = w < 600;
        const statsH = narrow ? 116 : 104;
        // --- pergunta
        V.text(g, 'Pergunta (termo max-lora-rank):', 16, 18, { size: 11, color: C.muted, weight: 600 });
        V.text(g, fit(g, '“Qual o padrão de --max-lora-rank no vLLM?”', w - 32, 12.5, 'sans', 600), 16, 36, { size: 12.5, weight: 600 });
        // --- geometria
        let cards, judgeC, outBox;
        if (!narrow) {
          const Lw = Math.min(250, w * 0.34), y0 = 52, avail = h - statsH - 18 - y0;
          const ch = Math.min(52, avail / 4 - 8);
          cards = [0, 1, 2, 3].map((i) => [16, y0 + i * (ch + 8), Lw, ch]);
          const outW = Math.min(180, w * 0.22);
          outBox = [w - 16 - outW, y0, outW, avail];
          judgeC = [16 + Lw + (outBox[0] - 16 - Lw) / 2, y0 + 38];
        } else {
          const y0 = 50, cw = (w - 32 - 10) / 2, ch = 44;
          cards = [0, 1, 2, 3].map((i) => [16 + (i % 2) * (cw + 10), y0 + Math.floor(i / 2) * (ch + 8), cw, ch]);
          judgeC = [48, y0 + 2 * (ch + 8) + 40];
          outBox = [16, judgeC[1] + 82, w - 32, 50];
        }
        const cur = k < 6 ? R.rec[k] : null;
        // --- cartas
        cards.forEach(([x, y, cw, ch], i) => {
          const hl = cur && (cur.i === i || cur.j === i);
          const isBest = k === 6 && i === R.best, isWorst = k === 6 && i === R.worst && R.pair;
          V.box(g, x, y, cw, ch, { r: 10, fill: isBest ? V.alpha(C.a, 0.18) : isWorst ? V.alpha(RED(), 0.12) : V.alpha(C.muted, 0.07), stroke: hl ? C.c : isBest ? C.a : V.alpha(C.muted, 0.35), lw: hl ? 2.2 : 1.2 });
          V.text(g, 'R' + (i + 1), x + 12, y + ch / 2 + 0.5, { size: 13, font: 'mono', weight: 700, base: 'middle', color: hl ? C.c : C.ink });
          const ns = nStars(S.q[i]);
          const sx0 = narrow ? x + 38 : x + 46, sdx = narrow ? 11 : 13, srad = narrow ? 4 : 5;
          for (let s = 0; s < 5; s++) star(g, sx0 + s * sdx, y + ch / 2 - (ch > 46 ? 6 : 0), srad, C.d, s < ns);
          if (ch > 46) V.text(g, 'qualidade real', x + 40, y + ch / 2 + 13, { size: 10, color: C.muted });
          const sv = sc[i];
          chip(g, x + cw - 56, y + ch / 2 - 10, 46, 20, 'S ' + (sv > 0 ? '+' : sv < 0 ? '−' : '') + Math.abs(sv), { size: 11.5, fill: sv > 0 ? V.alpha(C.a, 0.18) : sv < 0 ? V.alpha(RED(), 0.14) : V.alpha(C.muted, 0.1), stroke: sv > 0 ? V.alpha(C.a, 0.6) : sv < 0 ? V.alpha(RED(), 0.5) : V.alpha(C.muted, 0.4) });
        });
        // --- juiz
        const [jx, jy] = judgeC, js = 34;
        V.box(g, jx - js / 2, jy - js / 2, js, js * 0.86, { r: 9, fill: V.alpha(C.ds || C.b, 0.25), stroke: C.ds || C.b, lw: 1.5 });
        V.line(g, jx, jy - js / 2, jx, jy - js / 2 - 8, { color: C.ds || C.b, lw: 1.5 });
        V.dot(g, jx, jy - js / 2 - 10, 3 + Math.sin(t * 5) * 0.8, C.c);
        const blink = (t % 3.2) < 0.12;
        V.dot(g, jx - 7, jy - 4, blink ? 0.8 : 2.6, C.ink); V.dot(g, jx + 7, jy - 4, blink ? 0.8 : 2.6, C.ink);
        V.line(g, jx - 6, jy + 6, jx + 6, jy + 6, { color: C.ink, lw: 1.5 });
        V.text(g, 'juiz DeepSeek', jx, jy + js / 2 + 10, { size: 10.5, color: C.muted, align: 'center', weight: 600 });
        // balões
        const bxl = narrow ? jx + 34 : jx - Math.min(150, (outBox[0] - cards[0][0] - cards[0][2]) / 2 - 10);
        const bw = narrow ? w - 16 - bxl : Math.min(300, (outBox[0] - cards[0][0] - cards[0][2]) - 20);
        const by0 = narrow ? jy - 34 : jy + 50;
        if (cur) {
          const name = (a) => 'R' + (a + 1);
          const vt = (v) => (v === 'E' ? 'empate' : v);
          const a1 = Math.min(1, kf * 3);
          g.globalAlpha = a1;
          chip(g, bxl, by0, bw, 24, 'ordem 1 · A=' + name(cur.i) + ', B=' + name(cur.j) + ' → ' + vt(cur.v1), { size: 11.5, font: 'mono', fill: V.alpha(C.b, 0.14), stroke: V.alpha(C.b, 0.5) });
          g.globalAlpha = 1;
          if (S.swap) {
            const a2 = Math.max(0, Math.min(1, (kf - 0.3) * 3));
            g.globalAlpha = a2;
            chip(g, bxl, by0 + 30, bw, 24, 'ordem 2 · A=' + name(cur.j) + ', B=' + name(cur.i) + ' → ' + vt(cur.v2), { size: 11.5, font: 'mono', fill: V.alpha(C.b, 0.14), stroke: V.alpha(C.b, 0.5) });
            g.globalAlpha = 1;
          }
          if (kf > 0.68) {
            const s = cur.s, win = s > 0 ? cur.i : cur.j;
            let txt, col;
            if (S.swap) { if (s === 0) { txt = 'inverteu ou empatou → empate'; col = C.muted; } else { txt = 'consistente → +1 para ' + name(win); col = C.a; } }
            else { txt = s === 0 ? 'empate' : '+1 para ' + name(win) + ' (sem conferir a ordem)'; col = s === 0 ? C.muted : C.d; }
            V.text(g, fit(g, txt, bw, 12, 'sans', 700), bxl, by0 + (S.swap ? 74 : 44), { size: 12, weight: 700, color: col });
          }
          // fio do juiz até as cartas
          [cur.i, cur.j].forEach((ci) => { const [x, y, cw, ch] = cards[ci]; const ax = narrow ? x + cw / 2 : x + cw, ay = narrow ? y + ch : y + ch / 2; V.line(g, ax, ay, jx, jy, { color: V.alpha(C.c, 0.45), lw: 1.2, dash: [4, 4], dashOffset: -t * 20 }); });
        } else {
          const lines = [
            ['melhor R' + (R.best + 1) + ' (S ' + (R.sc[R.best] >= 0 ? '+' : '−') + Math.abs(R.sc[R.best]) + ') · pior R' + (R.worst + 1) + ' (S ' + (R.sc[R.worst] >= 0 ? '+' : '−') + Math.abs(R.sc[R.worst]) + ')', C.ink],
            [R.pair ? (R.pure ? 'a escolhida é mesmo melhor: par limpo' : 'a escolhida é PIOR: par invertido!') : 'margem < ' + S.margin + ': sem par para o DPO', R.pair ? (R.pure ? C.a : RED()) : C.muted],
          ];
          lines.forEach(([s, col], i) => V.text(g, fit(g, s, bw, 12, 'sans', 600), bxl, by0 + 4 + i * 20, { size: 12, weight: 600, color: col }));
        }
        // --- arquivos de saída
        const [ox, oy, ow, oh] = outBox;
        const files = [['sft.jsonl', k === 6 && R.sft], ['pref.jsonl', k === 6 && R.pair], ['kto.jsonl', k === 6 && R.pair]];
        if (!narrow) {
          V.text(g, 'saídas desta pergunta', ox, oy + 8, { size: 11, color: C.muted, weight: 600 });
          files.forEach(([f, on], i) => {
            const y = oy + 22 + i * 40;
            chip(g, ox, y, ow, 30, f, { size: 12, fill: on ? V.alpha(C.a, 0.2) : V.alpha(C.muted, 0.06), stroke: on ? C.a : V.alpha(C.muted, 0.3), color: on ? C.ink : C.muted });
            if (on) check(g, ox + ow - 16, y + 15, 5, C.a);
          });
          V.wrap(g, 'regras: S_melhor ≥ 1 → SFT; S_melhor − S_pior ≥ ' + S.margin + ' → par e rótulos KTO', ox, oy + 152, ow, { size: 10.5, color: C.muted, lineH: 14, maxLines: 4 });
        } else {
          V.text(g, 'saídas:', ox, oy + 8, { size: 11, color: C.muted, weight: 600 });
          const fw = (ow - 16) / 3;
          files.forEach(([f, on], i) => { chip(g, ox + i * (fw + 8), oy + 18, fw, 26, f, { size: 11, fill: on ? V.alpha(C.a, 0.2) : V.alpha(C.muted, 0.06), stroke: on ? C.a : V.alpha(C.muted, 0.3), color: on ? C.ink : C.muted }); });
        }
        // --- estatísticas (2.000 perguntas simuladas)
        const sy = h - statsH, sx = 16, swd = w - 32;
        V.box(g, sx, sy, swd, statsH - 6, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25) });
        V.text(g, fit(g, (narrow ? 'simulação de 2.000 perguntas' : 'simulação de 2.000 perguntas com este juiz') + ' (margem ≥ ' + S.margin + ')', swd - 24, 11, 'sans', 600), sx + 12, sy + 16, { size: 11, color: C.muted, weight: 600 });
        const groups = [['com troca', S.stats.on, S.swap], ['sem troca', S.stats.off, !S.swap]];
        const labW = narrow ? 74 : 90, barX = sx + 12 + labW, barW = swd - 24 - labW - 104;
        groups.forEach(([lb, st, act], gi) => {
          const gy = sy + 28 + gi * ((statsH - 40) / 2);
          V.text(g, lb, sx + 12, gy + 14, { size: 12, weight: act ? 700 : 500, color: act ? C.ink : C.muted });
          [['pares', st.y, C.b], ['pureza', st.p, C.a]].forEach(([nm, v, col], bi) => {
            const yy = gy + bi * 15;
            V.box(g, barX, yy + 2, barW, 11, { r: 4, fill: V.alpha(C.muted, 0.1) });
            V.box(g, barX, yy + 2, Math.max(2, barW * v), 11, { r: 4, fill: act ? col : V.alpha(col, 0.45) });
            V.text(g, nm + ' ' + pct(v), barX + barW + 8, yy + 11, { size: 10.5, font: 'mono', color: act ? C.ink : C.muted });
          });
        });
      });
    })();

    /* ======================================== FIG 3 — do JSONL aos tokens */
    (function () {
      const fig = root.querySelector('#fig-template');
      if (!fig) return;
      adaptH(fig, 420, 550);
      const S = { mode: 'rc', mask: true, t0: 0 };
      const J = {
        head: [[['{"messages": [{"role": "user",', 'k']], [['  "content": ', 'k'], ['"27B em NVFP4: quantos GB?"', 'user'], ['},', 'k']], [[' {"role": "assistant",', 'k']]],
        rc: [[['  "reasoning_content": ', 'k'], ['"27,8e9 × 4,5 bits…"', 'rs'], [',', 'k']], [['  "content": ', 'k'], ['"~16 GB só nos pesos NVFP4;', 'ct']], [['    o checkpoint real tem 22–25 GB."', 'ct'], ['}],', 'k']], [[' "chat_template_kwargs":', 'k']], [['   {"enable_thinking": ', 'k'], ['true', 'bool'], ['}}', 'k']]],
        inline: [[['  "content": ', 'k'], ['"<think>27,8e9 × 4,5 bits…', 'err']], [['     </think>', 'err'], ['~16 GB só nos pesos NVFP4;', 'ct']], [['    o checkpoint real tem 22–25 GB."', 'ct'], ['}],', 'k']], [[' "chat_template_kwargs":', 'k']], [['   {"enable_thinking": ', 'k'], ['true', 'bool'], ['}}', 'k']]],
        nothink: [[['  "content": ', 'k'], ['"~16 GB só nos pesos NVFP4;', 'ct']], [['    o checkpoint real tem 22–25 GB."', 'ct'], ['}],', 'k']], [[' "chat_template_kwargs":', 'k']], [['   {"enable_thinking": ', 'k'], ['false', 'bool'], ['}}', 'k']]],
      };
      const SP = (s) => ({ s, k: 'sp' }), NL = (n = 1) => ({ s: n > 1 ? '↵↵' : '↵', k: 'nl' });
      const ANS = () => [{ s: '~16 GB só nos pesos NVFP4;', k: 'ct' }, { s: 'o checkpoint real tem 22–25 GB', k: 'ct' }];
      function tokens(mode) {
        const T = [];
        if (mode !== 'nothink') T.push(SP('<|im_start|>'), { s: 'system', k: 'role' }, NL(), { s: 'Reasoning effort is set to xhigh…', k: 'sys' }, SP('<|im_end|>'), NL());
        T.push(SP('<|im_start|>'), { s: 'user', k: 'role' }, NL(), { s: '27B em NVFP4: quantos GB?', k: 'user' }, SP('<|im_end|>'), NL(), SP('<|im_start|>'), { s: 'assistant', k: 'role' }, NL());
        const A = [];
        if (mode === 'rc') A.push({ s: '<think>', k: 'tk' }, NL(), { s: '27,8e9 × 4,5 bits…', k: 'rs' }, NL(), { s: '</think>', k: 'tk' }, NL(2), ...ANS(), SP('<|im_end|>'));
        if (mode === 'inline') A.push({ s: '<think>', k: 'tk' }, NL(2), { s: '</think>', k: 'tk' }, NL(2), { s: '<think>', k: 'tk', err: 1 }, { s: '27,8e9 × 4,5 bits…', k: 'rs' }, { s: '</think>', k: 'tk', err: 1 }, ...ANS(), SP('<|im_end|>'));
        if (mode === 'nothink') A.push({ s: '<think>', k: 'tk' }, NL(2), { s: '</think>', k: 'tk' }, NL(2), ...ANS(), SP('<|im_end|>'));
        A.forEach((x) => { x.L = 1; });
        return T.concat(A);
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Exemplo', options: [{ v: 'rc', label: 'reasoning_content' }, { v: 'inline', label: '<think> no content (erro)' }, { v: 'nothink', label: 'sem raciocínio' }], value: 'rc', onChange: (v) => { S.mode = v; S.t0 = H ? H.t : 0; upd(); } });
      UI.toggle(ctl, { label: 'assistant_only_loss (máscara)', value: true, onChange: (v) => { S.mask = v; upd(); } });
      const rB = nocase(UI.readout(ctl, 'blocos <think>', '—')), rL = nocase(UI.readout(ctl, 'blocos com perda', '—'));
      function upd() {
        const T = tokens(S.mode);
        const nThink = T.filter((x) => x.s === '<think>').length;
        rB.set(String(nThink), nThink > 1 ? 'bad' : 'ok');
        const nL = T.filter((x) => !S.mask || x.L).length;
        rL.set(nL + ' de ' + T.length);
      }
      upd();
      const colOf = (k, C) => ({ sp: C.muted, role: C.muted, nl: C.muted, sys: C.c, user: C.b, tk: C.f, rs: C.d, ct: C.a, err: RED(), bool: C.c, k: C.muted }[k] || C.muted);
      UI.legend(ctl, [[V.C().b, 'usuário'], [V.C().c, 'system injetado'], [V.C().d, 'raciocínio'], [V.C().a, 'resposta'], [V.C().f, 'tags de pensamento']]);
      let H = null;
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        // cartão JSON
        const lines = J.head.concat(J[S.mode]);
        const fs = narrow ? 10.5 : 11.5, lh = fs + 5;
        const jh = lines.length * lh + 24;
        V.box(g, 16, 10, w - 32, jh, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
        V.text(g, 'sft.jsonl · uma linha', w - 26, 22, { size: 10, color: C.muted, align: 'right' });
        lines.forEach((ln, i) => {
          let x = 28; const y = 30 + i * lh;
          ln.forEach(([s, k]) => { V.text(g, s, x, y, { size: fs, font: 'mono', color: k === 'k' ? C.muted : colOf(k, C), weight: k === 'k' ? 400 : 600 }); x += tw(g, s, fs, 'mono', k === 'k' ? 400 : 600); });
        });
        // seta do template
        const ay = 10 + jh + 8;
        V.arrow(g, 30, ay, 30, ay + 26, { color: C.muted, lw: 1.5, head: 7 });
        V.wrap(g, 'chat template do Qwen3.8 (no treino com assistant_only_loss: qwen3_8_training.jinja)', 44, ay + 12, w - 60, { size: 11, color: C.muted, lineH: 14, maxLines: 2 });
        // tokens
        const T = tokens(S.mode);
        const p = (t - S.t0 + 5) % 9;
        const shown = p < 3.6 ? Math.floor((p / 3.6) * T.length) : T.length;
        const tfs = narrow ? 10.5 : 11.5, th = narrow ? 21 : 23, gap = 5, lg = 9;
        let x = 16, y = ay + (narrow ? 46 : 40);
        const maxW = w - 32;
        const pos = [];
        T.forEach((tk) => {
          const txt = tk.s;
          const cw = Math.min(maxW, tw(g, txt, tfs, 'mono', 600) + 12);
          if (x + cw > 16 + maxW) { x = 16; y += th + lg; }
          pos.push([x, y, cw]); x += cw + gap;
        });
        T.forEach((tk, i) => {
          if (i >= shown) return;
          const [tx, ty, cw] = pos[i];
          const col = tk.err ? RED() : colOf(tk.k, C);
          const inLoss = !S.mask || tk.L;
          const a = inLoss ? 1 : 0.42;
          g.globalAlpha = a;
          chip(g, tx, ty, cw, th, fit(g, tk.s, cw - 8, tfs, 'mono', 600), { fill: V.alpha(col, inLoss ? 0.2 : 0.08), stroke: tk.err ? RED() : V.alpha(col, inLoss ? 0.75 : 0.35), lw: tk.err ? 2 : 1.2, size: tfs, color: C.ink, weight: 600 });
          g.globalAlpha = 1;
          if (inLoss) {
            V.box(g, tx + 2, ty + th + 2, cw - 4, 3, { r: 1.5, fill: V.alpha(C.c, 0.85) });
            const sp = ((t * 0.9 + i * 0.37) % 1);
            V.dot(g, tx + 2 + sp * (cw - 4), ty + th + 3.5, 2, C.ink);
          }
        });
        // avisos
        if (shown >= T.length) {
          const errs = T.map((x, i) => (x.err ? i : -1)).filter((i) => i >= 0);
          errs.forEach((ei) => { const [ex, ey, ecw] = pos[ei]; const pul = 0.5 + 0.5 * Math.sin(t * 5); g.strokeStyle = V.alpha(RED(), 0.35 + 0.5 * pul); g.lineWidth = 2; V.rr(g, ex - 3 - pul * 2, ey - 3 - pul * 2, ecw + 6 + pul * 4, th + 6 + pul * 4, 7); g.stroke(); });
          let lastY = pos[pos.length - 1][1] + th + 24;
          if (errs.length) { lastY += V.wrap(g, '⚠ 2º bloco <think>: o raciocínio inline vira um segundo bloco de pensamento', 16, lastY, w - 32, { size: 11.5, weight: 700, color: RED(), lineH: 15, maxLines: 2 }) + 5; }
          const note = S.mask ? 'faixa rosa = tokens que entram na perda (saída do assistente, delimitada por {% generation %})' : 'máscara desligada: o modelo também aprende a escrever o prompt do usuário e o system';
          if (lastY < h - 6) V.wrap(g, note, 16, lastY, w - 32, { size: 11, color: C.muted, lineH: 14, maxLines: 2 });
        }
      });
    })();

    /* ======================================== FIG 4 — o tanque do Mac */
    (function () {
      const fig = root.querySelector('#fig-mac');
      if (!fig) return;
      adaptH(fig, 420, 540);
      const Q = { h: 2560, V: 151936, N: 4.022e9, perLayer: 100925440, lm: 388956160, loraPerR: 57344 };
      const RANKS = [4, 8, 16, 32, 64], LAYERS = [4, 8, 16, 24, 36];
      const S = { mode: 'sft', b: 4, s: 1024, ri: 2, li: 2, ckpt: true, cur: [0, 0, 0, 0] };
      const M = {};
      function compute() {
        const r = RANKS[S.ri], Lt = LAYERS[S.li], seqs = S.mode === 'sft' ? S.b : 2;
        const Wb = S.mode === 'sft' ? 2.26e9 : 8.6e9;
        const P = Q.loraPerR * r * Lt, lora = 16 * P;
        const act = S.ckpt ? 2 * S.s * seqs * Q.h * Lt + 34 * S.s * seqs * Q.h : 34 * S.s * seqs * Q.h * Lt;
        const lg1 = seqs * S.s * Q.V * 4, lg = 2 * lg1;
        const tot = Wb + lora + act + lg;
        const flops = 2 * Q.N + 2 * (Lt * Q.perLayer + Q.lm) + (S.ckpt ? 2 * Lt * Q.perLayer : 0) + (S.mode === 'dpo' ? 2 * Q.N : 0);
        Object.assign(M, { r, Lt, seqs, P, parts: [Wb / 1e9, lora / 1e9, act / 1e9, lg / 1e9], lg1: lg1 / 1e9, tot: tot / 1e9, flops });
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Treino', options: [{ v: 'sft', label: 'SFT (QLoRA 4-bit)' }, { v: 'dpo', label: 'DPO (2× 8-bit)' }], value: 'sft', onChange: (v) => { S.mode = v; gB.show(v === 'sft'); upd(); } });
      const gB = UI.group(ctl);
      nocase(UI.slider(gB.el, { label: 'lote b', min: 1, max: 8, step: 1, value: S.b, onInput: (v) => { S.b = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'sequência s', min: 256, max: 4096, step: 256, value: S.s, onInput: (v) => { S.s = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'posto r', min: 0, max: 4, step: 1, value: S.ri, fmt: (v) => String(RANKS[v]), onInput: (v) => { S.ri = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'camadas treinadas', min: 0, max: 4, step: 1, value: S.li, fmt: (v) => String(LAYERS[v]), onInput: (v) => { S.li = v; upd(); } }));
      UI.toggle(ctl, { label: 'grad_checkpoint', value: true, onChange: (v) => { S.ckpt = v; upd(); } });
      const rT = nocase(UI.readout(ctl, 'total [estim.]', '—')), rS = nocase(UI.readout(ctl, 'tok/s [estim.]', '—')), rH = nocase(UI.readout(ctl, 'tempo [estim.]', '—'));
      function upd() {
        compute();
        const bad = M.tot > 19.07 || M.lg1 > 14.3;
        rT.set(fmt(M.tot, 1) + ' GB', bad ? 'bad' : M.tot > 15 ? 'warn' : 'ok');
        const lo = 1e12 / M.flops, hi = 2e12 / M.flops;
        if (S.mode === 'sft') {
          rS.set(Math.round(lo) + '–' + Math.round(hi));
          const tok = 600 * S.b * 500;
          rH.set(fmt(tok / hi / 3600, 1) + '–' + fmt(tok / lo / 3600, 1) + ' h (600 iters)');
        } else {
          rS.set(Math.round(lo) + '–' + Math.round(hi));
          const pair = 2 * 700 * M.flops;
          rH.set(Math.round(pair / 2e12) + '–' + Math.round(pair / 1e12) + ' s por par');
        }
      }
      upd();
      const bubbles = Array.from({ length: 14 }, (_, i) => ({ x: (i * 0.61803) % 1, p: (i * 0.37) % 1, s: 1 + (i % 3) }));
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        for (let i = 0; i < 4; i++) S.cur[i] += (M.parts[i] - S.cur[i]) * Math.min(1, dt * 5);
        const cols = [C.b, C.a, C.e, C.d];
        const names = narrow ? (S.mode === 'sft' ? ['pesos 4-bit', 'LoRA + grad + Adam', 'ativações', 'logits + gradiente'] : ['política + ref. (2 × 8-bit)', 'LoRA + grad + Adam', 'ativações (2 seqs)', 'logits + gradiente'])
          : S.mode === 'sft' ? ['pesos 4-bit congelados', 'LoRA: peso + grad + Adam', 'ativações', 'logits FP32 + gradiente'] : ['política + referência (2 × 8-bit)', 'LoRA: peso + grad + Adam', 'ativações (escolhida + rejeitada)', 'logits FP32 + gradiente'];
        const subs = [
          S.mode === 'sft' ? 'mlx-community/…-4bit, arquivo de 2,26 GB' : 'duas cópias completas de ≈ 4,3 GB',
          fmt(M.P / 1e6, 1) + ' M parâm. × 16 B (r=' + M.r + ' · ' + M.Lt + ' camadas)',
          S.ckpt ? '2·s·b·h·Lₜ + 34·s·b·h (checkpointing)' : '34·s·b·h·Lₜ (sem checkpointing)',
          '[' + M.seqs + ' × ' + S.s + ' × 151.936] × 4 B × 2',
        ];
        const maxGB = 26;
        const tx = narrow ? 36 : 64, tW = narrow ? 64 : 140, tTop = 24, tBot = h - 22;
        const yOf = (gb) => tBot - (Math.min(gb, maxGB) / maxGB) * (tBot - tTop);
        // régua
        [0, 5, 10, 15, 20, 24].forEach((gb) => { const y = yOf(gb); V.line(g, tx - 5, y, tx, y, { color: C.muted, lw: 1 }); V.text(g, String(gb), tx - 8, y + 0.5, { size: 10, font: 'mono', color: C.muted, align: 'right', base: 'middle' }); });
        V.text(g, 'GB', tx - 8, tTop - 10, { size: 10, color: C.muted, align: 'right' });
        // líquido
        g.save(); V.rr(g, tx, tTop, tW, tBot - tTop, 12); g.clip();
        let acc = 0;
        S.cur.forEach((v, i) => {
          const y0 = yOf(acc), y1 = yOf(acc + v);
          if (y0 - y1 > 0.3) {
            g.fillStyle = V.alpha(cols[i], 0.78);
            if (i === 3) { g.beginPath(); g.moveTo(tx, y0); for (let xx = 0; xx <= tW; xx += 4) g.lineTo(tx + xx, y1 + Math.sin(xx / 13 + t * 2.4) * 2.2); g.lineTo(tx + tW, y0); g.closePath(); g.fill(); }
            else g.fillRect(tx, y1, tW, y0 - y1 + 0.5);
            if (y0 - y1 > 14) V.text(g, fmt(v, v >= 10 ? 1 : 2), tx + tW / 2, (y0 + y1) / 2, { size: 11, font: 'mono', weight: 700, color: C.dark ? '#0b0f0d' : '#ffffff', align: 'center', base: 'middle' });
          }
          acc += v;
        });
        const surf = yOf(acc);
        bubbles.forEach((b) => { const yy = tBot - ((b.p + t * 0.06 * b.s) % 1) * (tBot - Math.max(surf, tTop)); if (yy > surf + 4) { g.beginPath(); g.arc(tx + 8 + b.x * (tW - 16), yy, b.s * 1.4, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.dark ? '#ffffff' : '#000000', 0.25); g.lineWidth = 1; g.stroke(); } });
        g.restore();
        V.rr(g, tx, tTop, tW, tBot - tTop, 12); g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1.6; g.stroke();
        // linhas-limite
        const lim = [[24, 'RAM 24', V.alpha(C.ink, 0.5), [2, 3]], [19.07, 'working set 19,07', C.ink, [6, 4]], [14.3, 'maior buffer 14,3', C.d, [6, 4]]];
        lim.forEach(([gb, lb, col, dash]) => { const y = yOf(gb); V.line(g, tx - 2, y, tx + tW + 4, y, { color: col, lw: 1.4, dash }); if (!narrow) V.text(g, lb, tx + tW + 8, y + 0.5, { size: 10.5, color: col, base: 'middle', weight: 600 }); });
        if (acc > 19.07) {
          const sp = (t * 1.6) % 1;
          for (let k = 0; k < 4; k++) { const ph = (sp + k * 0.25) % 1; V.dot(g, tx + tW + 3 + ph * 10, tTop + ph * ph * 60, 2.4, RED()); }
          V.text(g, acc > maxGB ? 'transborda!' : 'acima do working set', tx + tW / 2, tTop - 9, { size: 11, weight: 700, color: RED(), align: 'center' });
        }
        // lista
        const lx = narrow ? tx + tW + 12 : tx + tW + 136, lw = w - lx - 14;
        let ly = narrow ? 30 : 40;
        const rowH = narrow ? 58 : 52;
        names.forEach((nm, i) => {
          V.box(g, lx, ly - 9, 10, 10, { r: 3, fill: cols[i] });
          V.text(g, fit(g, nm, lw - 78, narrow ? 11.5 : 12.5, 'sans', 600), lx + 16, ly, { size: narrow ? 11.5 : 12.5, weight: 600 });
          V.text(g, fmt(M.parts[i], M.parts[i] >= 10 ? 1 : 2) + ' GB', lx + lw, ly, { size: narrow ? 11.5 : 12.5, font: 'mono', weight: 700, align: 'right', color: cols[i] });
          if (narrow) V.wrap(g, subs[i], lx + 16, ly + 16, lw - 16, { size: 10, font: 'mono', color: C.muted, lineH: 12, maxLines: 2 });
          else V.text(g, fit(g, subs[i], lw - 16, 10.5, 'mono', 400), lx + 16, ly + 17, { size: 10.5, font: 'mono', color: C.muted });
          ly += rowH;
        });
        V.line(g, lx, ly - 18, lx + lw, ly - 18, { color: V.alpha(C.muted, 0.4), lw: 1 });
        const bad = M.tot > 19.07;
        V.text(g, 'total ≈ ' + fmt(M.tot, 1) + ' GB', lx, ly + 2, { size: narrow ? 13 : 14, weight: 700, color: bad ? RED() : M.tot > 15 ? C.d : C.a });
        const lgBad = M.lg1 > 14.3;
        V.wrap(g, 'um tensor de logits: ' + fmt(M.lg1, 2) + ' GB ' + (lgBad ? '> 14,3 (maior buffer)!' : '< 14,3'), lx, ly + 22, lw, { size: narrow ? 10.5 : 11.5, color: lgBad ? RED() : C.muted, lineH: 14, maxLines: 2 });
        if (narrow) V.wrap(g, 'linhas no tanque: RAM 24 (pontilhada) · working set 19,07 (branca) · maior buffer 14,3 (âmbar)', lx, ly + 54, lw, { size: 10, color: C.muted, lineH: 13, maxLines: 4 });
      });
    })();

    /* ======================================== FIG 5 — dois Sparks, um cabo */
    (function () {
      const fig = root.querySelector('#fig-cabo');
      if (!fig) return;
      adaptH(fig, 410, 620);
      const S = { mode: 'ddp' };
      const MODES = {
        ddp: {
          A: { t: 'Spark A · rank 0 · treina', segs: [['pesos BF16 do 27B', 55.6, 'b'], ['LoRA + grad + Adam', 3.5, 'a'], ['ativações', 2.7, 'e'], ['pico de recomputação', 1.5, 'f'], ['logits + gradiente', 8.1, 'd']], band: [70, 80] },
          B: { t: 'Spark B · rank 1 · treina', segs: [['pesos BF16 do 27B', 55.6, 'b'], ['LoRA + grad + Adam', 3.5, 'a'], ['ativações', 2.7, 'e'], ['pico de recomputação', 1.5, 'f'], ['logits + gradiente', 8.1, 'd']], band: [70, 80] },
          cable: '0,87 GB/passo: all-reduce dos gradientes LoRA (FP32)',
          tl: [['compute: forward + backward + recomputação', 58.5, 'a'], ['cabo ≈ 0,04 s', 0.06, 'c']],
          ro: ['0,87 GB', '≈ 0,04 s', '≈ 70–80 GB', '≈ 39–78 s'],
        },
        fsdp: {
          A: { t: 'Spark A · rank 0 · treina', segs: [['metade dos pesos', 27.8, 'b'], ['metade do estado LoRA', 1.75, 'a'], ['ativações', 2.7, 'e'], ['pico de recomputação', 1.5, 'f'], ['logits + gradiente', 8.1, 'd']] },
          B: { t: 'Spark B · rank 1 · treina', segs: [['metade dos pesos', 27.8, 'b'], ['metade do estado LoRA', 1.75, 'a'], ['ativações', 2.7, 'e'], ['pico de recomputação', 1.5, 'f'], ['logits + gradiente', 8.1, 'd']] },
          cable: '≈ 83 GB por micro-lote e direção, ×16 micro-lotes: all-gather dos pesos',
          tl: [['compute', 58.5, 'a'], ['all-gather ≈ 67 s', 67, 'c']],
          ro: ['≈ 1,3 TB/passo', '≈ 67 s', '≈ 42 GB', '≈ 39–78 s + ~67 s'],
        },
        grpo: {
          A: { t: 'Spark A · GRPOTrainer (aprende)', segs: [['pesos BF16 do 9B', 19.3, 'b'], ['LoRA + estado', 1.0, 'a'], ['ativações', 1.5, 'e'], ['logits', 3.6, 'd']], band: [30, 40] },
          B: { t: 'Spark B · vllm serve (gera)', segs: [['pesos BF16 do 9B', 19.3, 'b'], ['KV + estados GDN (~45 GiB)', 48.3, 'c']], res: 71.9 },
          cable: 'B → A: completions · A → B: ≈ 19,3 GB de pesos fundidos por passo (NCCL)',
          tl: [['geração (B): 45–85 s', 65, 'b'], ['treino (A, 8N): 87–173 s', 130, 'a'], ['sync 1–2 s', 1.5, 'c']],
          ro: ['≈ 19,3 GB + texto', '≈ 1–2 s', 'A 30–40 GB · B 67 GiB', '≈ 2,2–4,3 min'],
        },
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Arranjo', options: [{ v: 'ddp', label: 'DDP (SFT/DPO)' }, { v: 'fsdp', label: 'FSDP (desnecessário)' }, { v: 'grpo', label: 'GRPO assimétrico' }], value: 'ddp', onChange: (v) => { S.mode = v; upd(); } });
      const r1 = nocase(UI.readout(ctl, 'no cabo por passo', '—')), r2 = nocase(UI.readout(ctl, 'tempo de cabo', '—')), r3 = nocase(UI.readout(ctl, 'memória por nó', '—')), r4 = nocase(UI.readout(ctl, 'passo [estim.]', '—'));
      function upd() { const m = MODES[S.mode]; r1.set(m.ro[0], S.mode === 'fsdp' ? 'bad' : 'ok'); r2.set(m.ro[1], S.mode === 'fsdp' ? 'bad' : 'ok'); r3.set(m.ro[2]); r4.set(m.ro[3]); }
      upd();
      function node(g, x, y, w, h, nd, t, act, C, compact, sw) {
        V.box(g, x, y, w, h, { r: 12, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1.2 });
        V.text(g, fit(g, nd.t, w - 70, 12.5, 'sans', 700), x + 12, y + 20, { size: 12.5, weight: 700 });
        // chip GB10
        const cx = x + w - 30, cy = y + 20;
        V.glow(g, cx, cy, 20, C.a, act ? 0.5 + 0.3 * Math.sin(t * 8) : 0.12);
        V.box(g, cx - 14, cy - 11, 28, 22, { r: 5, fill: V.alpha(C.a, 0.25), stroke: C.a, lw: 1.2 });
        V.text(g, 'GB10', cx, cy + 0.5, { size: 9.5, font: 'mono', weight: 700, align: 'center', base: 'middle' });
        // legenda
        const L = nd.segs, rowH = compact ? 14.5 : 16, l0 = y + 50;
        L.forEach(([nm, v, ck], i) => {
          const yy = l0 + i * rowH;
          V.box(g, x + 12, yy - 8, 9, 9, { r: 2, fill: C[ck] });
          V.text(g, fit(g, nm, w - 110, 11, 'sans', 500), x + 26, yy, { size: 11 });
          V.text(g, fmt(v, v >= 10 ? 1 : 2) + ' GB', x + w - 12, yy, { size: 11, font: 'mono', align: 'right', color: C.muted });
        });
        // barra de memória (posição)
        const by = y + h - 36, bx = x + 12, bw = w - 24, bh = 14, max = 150;
        // faixa de camadas: forward/backward ou geração
        const sy0 = l0 + L.length * rowH + 6, sy1 = by - 22;
        if (sy1 - sy0 > 26) {
          const nb = 16, cw = bw / nb, chh = Math.min(26, sy1 - sy0 - 16), cy0 = sy0 + 14;
          if (sw.kind === 'gen') {
            V.text(g, fit(g, 'gerando: 8 completions por prompt, recompensa 0/1', bw, 10.5, 'sans', 600), bx, sy0 + 6, { size: 10.5, color: C.b, weight: 600 });
            for (let p = 0; p < 2; p++) for (let c = 0; c < 8; c++) {
              const grow = Math.min(1, Math.max(0, sw.f * 1.15 - c * 0.04));
              const bwid = (bw / 2 - 10) / 8, x0 = bx + p * (bw / 2) + c * bwid;
              const hh = (chh - 2) * grow;
              V.box(g, x0, cy0 + chh - hh, bwid - 3, Math.max(1, hh), { r: 2, fill: V.alpha(C.b, 0.35 + 0.5 * grow) });
              if (grow >= 1) V.dot(g, x0 + (bwid - 3) / 2, cy0 - 3, 2.2, ((c * 3 + p * 5) % 7) < 3 ? C.a : RED());
            }
          } else {
            const lbl = sw.kind === 'fwd' ? 'forward: camada a camada →' : sw.kind === 'bwd' ? '← backward (+ recomputação)' : sw.kind === 'comm' ? 'parado: esperando o cabo' : 'ocioso';
            const suf = ' · 64 camadas (16 no desenho)';
            const full = !compact && tw(g, lbl + suf, 10.5, 'sans', 600) <= bw ? lbl + suf : lbl;
            V.text(g, fit(g, full, bw, 10.5, 'sans', 600), bx, sy0 + 6, { size: 10.5, color: sw.kind === 'comm' ? C.c : C.muted, weight: 600 });
            const head = sw.kind === 'fwd' || sw.kind === 'bwd' ? sw.f * nb : -5;
            for (let b = 0; b < nb; b++) {
              const d = Math.abs(b + 0.5 - head);
              const on = d < 1.2;
              V.box(g, bx + b * cw + 1, cy0, cw - 2, chh, { r: 3, fill: on ? V.alpha(C.a, 0.85) : V.alpha(C.muted, (sw.kind === 'fwd' && b < head) || (sw.kind === 'bwd' && b > head) ? 0.28 : 0.12) });
            }
          }
        }
        V.box(g, bx, by, bw, bh, { r: 4, fill: V.alpha(C.muted, 0.12) });
        let acc = 0;
        L.forEach(([nm, v, ck]) => { const x0 = bx + (acc / max) * bw, ww = (v / max) * bw; g.fillStyle = C[ck]; g.fillRect(x0, by, Math.max(1, ww), bh); acc += v; });
        if (nd.res) { const xr = bx + (nd.res / max) * bw; V.rr(g, bx, by - 2, xr - bx, bh + 4, 4); g.strokeStyle = C.c; g.setLineDash([4, 3]); g.lineWidth = 1.4; g.stroke(); g.setLineDash([]); V.text(g, 'reserva ≈ 67 GiB', xr, by - 6, { size: 10, color: C.c, align: 'right' }); }
        if (nd.band) { const x0 = bx + (nd.band[0] / max) * bw, x1 = bx + (nd.band[1] / max) * bw; V.line(g, x0, by - 4, x1, by - 4, { color: C.ink, lw: 2 }); V.line(g, x0, by - 7, x0, by - 1, { color: C.ink, lw: 1.5 }); V.line(g, x1, by - 7, x1, by - 1, { color: C.ink, lw: 1.5 }); V.text(g, 'total ' + nd.band[0] + '–' + nd.band[1] + ' GB', (x0 + x1) / 2, by - 9, { size: 10, color: C.ink, align: 'center' }); }
        [[112.7, 'prático'], [130.7, 'visível']].forEach(([v, lb]) => { const xx = bx + (v / max) * bw; V.line(g, xx, by - 3, xx, by + bh + 3, { color: lb === 'prático' ? C.d : C.ink, lw: 1.5 }); V.text(g, lb, lb === 'prático' ? xx - 3 : xx + 3, by + bh + 13, { size: 10, color: lb === 'prático' ? C.d : C.muted, align: lb === 'prático' ? 'right' : 'left' }); });
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const m = MODES[S.mode];
        const narrow = w < 600;
        const tlH = narrow ? 92 : 74;
        const period = 4;
        const ph = (t % period) / period;
        let A, B, c1, c2;
        if (!narrow) {
          const nw = Math.min(330, w * 0.36), nh = h - tlH - 30;
          A = [16, 12, nw, nh]; B = [w - 16 - nw, 12, nw, nh];
          c1 = [A[0] + nw, 12 + nh * 0.55]; c2 = [B[0], 12 + nh * 0.55];
        } else {
          const nh = (h - tlH - 22 - 70) / 2;
          A = [16, 10, w - 32, nh]; B = [16, 10 + nh + 70, w - 32, nh];
          c1 = [w / 2, A[1] + nh]; c2 = [w / 2, B[1]];
        }
        // atividade
        let actA = true, actB = true;
        if (S.mode === 'ddp' || S.mode === 'fsdp') { actA = actB = S.mode === 'fsdp' || ph < 0.8; }
        if (S.mode === 'grpo') { actB = ph < 0.33; actA = ph > 0.38 && ph < 0.95; } // fases ~ proporcionais à régua: geração 1/3, treino 2/3
        let swA, swB;
        if (S.mode === 'ddp') { swA = ph < 0.4 ? { kind: 'fwd', f: ph / 0.4 } : ph < 0.8 ? { kind: 'bwd', f: 1 - (ph - 0.4) / 0.4 } : { kind: 'comm' }; swB = swA; }
        else if (S.mode === 'fsdp') { swA = ph < 0.5 ? { kind: 'fwd', f: ph / 0.5 } : { kind: 'bwd', f: 1 - (ph - 0.5) / 0.5 }; swB = swA; }
        else { swA = ph > 0.38 && ph < 0.62 ? { kind: 'fwd', f: (ph - 0.38) / 0.24 } : ph >= 0.62 && ph < 0.95 ? { kind: 'bwd', f: 1 - (ph - 0.62) / 0.33 } : { kind: 'idle' }; swB = ph < 0.33 ? { kind: 'gen', f: ph / 0.33 } : { kind: 'gen', f: 1 }; }
        node(g, A[0], A[1], A[2], A[3], m.A, t, actA, C, narrow, swA);
        node(g, B[0], B[1], B[2], B[3], m.B, t, actB, C, narrow, swB);
        // cabo
        g.lineCap = 'round';
        V.line(g, c1[0], c1[1], c2[0], c2[1], { color: V.alpha(C.muted, 0.5), lw: 7 });
        V.line(g, c1[0], c1[1], c2[0], c2[1], { color: V.alpha(C.bg, 0.9), lw: 3 });
        g.lineCap = 'butt';
        const L = (f) => [V.lerp(c1[0], c2[0], f), V.lerp(c1[1], c2[1], f)];
        const pkt = (f, s, col) => { const [px, py] = L(f); V.box(g, px - s / 2, py - s / 2, s, s, { r: Math.min(4, s / 3), fill: col, stroke: V.alpha(C.ink, 0.4), lw: 1 }); };
        if (S.mode === 'ddp') {
          if (ph > 0.8) { const f = (ph - 0.8) / 0.2; pkt(f, 8, C.c); pkt(1 - f, 8, C.c); }
        } else if (S.mode === 'fsdp') {
          for (let k = 0; k < 7; k++) { const f = (t * 0.55 + k / 7) % 1; pkt(f, 15, C.c); pkt(1 - ((t * 0.55 + k / 7 + 0.07) % 1), 15, V.alpha(C.c, 0.7)); }
        } else {
          if (ph >= 0.33 && ph < 0.38) { const f = (ph - 0.33) / 0.05; for (let k = 0; k < 5; k++) pkt(Math.max(0, 1 - f - k * 0.05), 5, C.b); }
          if (ph >= 0.95) { const f = (ph - 0.95) / 0.05; pkt(f, 15, C.a); }
        }
        // rótulo do cabo
        const lbl = m.cable;
        if (!narrow) {
          const mx = (c1[0] + c2[0]) / 2, mw = c2[0] - c1[0] - 20;
          V.text(g, 'RoCE 200 Gb/s', mx, c1[1] - 18, { size: 11, weight: 700, align: 'center', color: C.muted });
          V.wrap(g, lbl, mx - mw / 2, c1[1] + 24, mw, { size: 11, lineH: 14, color: C.ink, align: 'left', maxLines: 4 });
        } else {
          V.text(g, 'RoCE 200 Gb/s', c1[0] + 12, (c1[1] + c2[1]) / 2 - 8, { size: 10.5, weight: 700, color: C.muted });
          V.wrap(g, lbl, c1[0] + 12, (c1[1] + c2[1]) / 2 + 6, w / 2 - 28, { size: 10, lineH: 12, color: C.ink, maxLines: 3 });
        }
        // linha do tempo de um passo
        const ty = h - tlH + 6, tx = 16, twd = w - 32;
        V.text(g, 'um passo de otimizador (proporcional) [estimativa]', tx, ty + 8, { size: 11, color: C.muted, weight: 600 });
        const tot = m.tl.reduce((s, x) => s + x[1], 0);
        let acc = 0; const bh = 18, byy = ty + 16;
        m.tl.forEach(([nm, v, ck], i) => {
          const x0 = tx + (acc / tot) * twd, ww = Math.max(3, (v / tot) * twd);
          V.box(g, x0, byy, ww - (i < m.tl.length - 1 ? 2 : 0), bh, { r: 4, fill: V.alpha(C[ck], 0.75) });
          acc += v;
        });
        const head = tx + ph * twd;
        V.line(g, head, byy - 4, head, byy + bh + 4, { color: C.ink, lw: 2 });
        acc = 0;
        const lineY = byy + bh + 15;
        m.tl.forEach(([nm, v, ck], i) => {
          const x0 = tx + (acc / tot) * twd, ww = (v / tot) * twd;
          const right = i === m.tl.length - 1 && ww < 120;
          const ly = narrow ? lineY + i * 14 : lineY;
          if (narrow) V.text(g, '■ ' + nm, tx, ly, { size: 10.5, color: C[ck] });
          else V.text(g, fit(g, nm, Math.max(90, ww - 6), 10.5), right ? tx + twd : x0 + 2, ly, { size: 10.5, color: C[ck], align: right ? 'right' : 'left', weight: 600 });
          acc += v;
        });
      });
    })();

    /* ======================================== FIG 6 — o painel de bordo */
    (function () {
      const fig = root.querySelector('#fig-painel');
      if (!fig) return;
      adaptH(fig, 440, 676);
      const N = 60;
      const S = { algo: 'sft', scen: 'ok', t0: -7 };
      const mk = (f, seed, noise) => { const r = V.rng(seed); return Array.from({ length: N }, (_, k) => f(k) + r.normal() * noise); };
      const DEF = {
        sft: {
          ok: [
            { n: 'loss', s: mk((k) => 1.6 * Math.exp(-k / 14) + 0.62, 1, 0.025) },
            { n: 'eval_loss', s: mk((k) => 1.5 * Math.exp(-k / 16) + 0.7, 2, 0.012) },
            { n: 'mean_token_accuracy', s: mk((k) => 0.48 + 0.24 * (1 - Math.exp(-k / 20)), 3, 0.006) },
            { n: 'grad_norm', s: mk((k) => 0.9 + 0.4 * Math.exp(-k / 8), 4, 0.05) },
          ],
          bad: [
            { n: 'loss', s: mk((k) => 1.75 * Math.exp(-k / 9) + 0.12, 1, 0.02) },
            { n: 'eval_loss', s: mk((k) => 1.5 * Math.exp(-k / 9) + 0.66 + Math.max(0, k - 20) * 0.016, 2, 0.012) },
            { n: 'mean_token_accuracy', s: mk((k) => 0.48 + 0.5 * (1 - Math.exp(-k / 9)), 3, 0.005) },
            { n: 'grad_norm', s: mk((k) => 0.9 + 0.4 * Math.exp(-k / 8), 4, 0.05) },
          ],
          diag: { ok: 'Saudável: loss e eval_loss caem juntas, a acurácia por token sobe devagar. Siga para o merge do SFT e o DPO.', bad: 'Sobreajuste e memorização: o treino continua caindo, a eval_loss sobe depois de ~1/3 do treino e a acurácia por token passa de 0,95. Corrija com menos épocas ou LR menor, e procure duplicatas no sft.jsonl.' },
          rules: [(s) => 'ok', (s, k) => (s[k] > Math.min(...s.slice(0, k + 1)) + 0.12 ? 'bad' : s[k] > Math.min(...s.slice(0, k + 1)) + 0.05 ? 'warn' : 'ok'), (s, k) => (s[k] > 0.95 ? 'bad' : s[k] > 0.9 ? 'warn' : 'ok'), (s) => 'ok'],
        },
        dpo: {
          ok: [
            { n: 'rewards/accuracies', s: mk((k) => 0.5 + 0.22 * (1 - Math.exp(-k / 14)), 5, 0.012) },
            { n: 'rewards/margins', s: mk((k) => 1.3 * (1 - Math.exp(-k / 22)), 6, 0.04) },
            { n: 'logps/chosen', s: mk((k) => -300 - k * 0.25, 7, 2) },
            { n: 'comprimento (das amostras)', s: mk(() => 220, 8, 4) },
          ],
          bad: [
            { n: 'rewards/accuracies', s: mk((k) => 0.5 + 0.49 * (1 - Math.exp(-k / 7)), 5, 0.006) },
            { n: 'rewards/margins', s: mk((k) => 6.2 * (1 - Math.exp(-k / 16)), 6, 0.08) },
            { n: 'logps/chosen', s: mk((k) => -300 - 230 * (1 - Math.exp(-k / 18)), 7, 2.5) },
            { n: 'comprimento (das amostras)', s: mk((k) => 220 + 170 * (1 - Math.exp(-k / 20)), 8, 4) },
          ],
          diag: { ok: 'Saudável: a acurácia de recompensa estaciona em 0,6–0,8, a margem sobe e a log-prob da escolhida fica quase parada. A âncora (β) está segurando.', bad: 'Likelihood displacement e verbosidade: a margem sobe enquanto logps/chosen despenca, e a acurácia vai a 1,0. Use loss_type=["sigmoid","sft"] (NLL nas escolhidas), β maior, apo_zero; contra o comprimento, sigmoid_norm, ld_alpha ou SimPO.' },
          rules: [(s, k) => (s[k] > 0.95 ? 'bad' : s[k] > 0.9 ? 'warn' : 'ok'), (s) => 'ok', (s, k) => (s[0] - s[k] > 80 ? 'bad' : s[0] - s[k] > 40 ? 'warn' : 'ok'), (s, k) => (s[k] > s[0] * 1.35 ? 'bad' : s[k] > s[0] * 1.15 ? 'warn' : 'ok')],
        },
        grpo: {
          ok: [
            { n: 'reward', s: mk((k) => 0.35 + 0.27 * (1 - Math.exp(-k / 20)), 9, 0.02) },
            { n: 'frac_reward_zero_std', s: mk((k) => 0.3 + 0.08 * (k / N), 10, 0.02) },
            { n: 'completions/mean_length', s: mk((k) => 410 + k * 0.5, 11, 6) },
            { n: 'entropy', s: mk((k) => 0.55 - 0.13 * (k / N), 12, 0.01) },
          ],
          bad: [
            { n: 'reward', s: mk((k) => 0.35 + 0.5 * (1 - Math.exp(-k / 14)), 9, 0.02) },
            { n: 'frac_reward_zero_std', s: mk((k) => 0.3 + 0.6 * (1 - Math.exp(-k / 16)), 10, 0.02) },
            { n: 'completions/mean_length', s: mk((k) => 410 + 345 * (1 - Math.exp(-k / 15)), 11, 6) },
            { n: 'entropy', s: mk((k) => 0.08 + 0.47 * Math.exp(-k / 12), 12, 0.01) },
          ],
          diag: { ok: 'Saudável: a recompensa sobe, ~1/3 dos grupos sem variância (ainda há sinal), comprimento estável e entropia caindo devagar. Leia amostras mesmo assim (log_completions=True).', bad: 'Hacking e colapso: a recompensa sobe com o comprimento colado em max_completion_length (768), a entropia colapsa e quase todos os grupos ficam sem variância. Leia as amostras, endureça o verificador (exatamente um \\boxed{}), use mask_truncated_completions e epsilon_high, e ajuste a dificuldade do dataset.' },
          rules: [(s) => 'ok', (s, k) => (s[k] > 0.7 ? 'bad' : s[k] > 0.5 ? 'warn' : 'ok'), (s, k) => (s[k] > s[0] * 1.4 ? 'bad' : s[k] > s[0] * 1.2 ? 'warn' : 'ok'), (s, k) => (s[k] < s[0] * 0.4 ? 'bad' : s[k] < s[0] * 0.6 ? 'warn' : 'ok')],
        },
      };
      // nome da métrica em até 2 linhas no celular (quebra em espaço, '/' ou '_')
      function nameLines(g, n, maxW, size) {
        if (tw(g, n, size, 'mono', 600) <= maxW) return [n];
        for (const sep of [' ', '/', '_']) {
          const k = n.lastIndexOf(sep);
          if (k > 0) return [fit(g, n.slice(0, sep === ' ' ? k : k + 1), maxW, size, 'mono', 600), fit(g, n.slice(k + 1), maxW, size, 'mono', 600)];
        }
        return [fit(g, n, maxW, size, 'mono', 600)];
      }
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Algoritmo', options: [{ v: 'sft', label: 'SFT' }, { v: 'dpo', label: 'DPO' }, { v: 'grpo', label: 'GRPO' }], value: 'sft', onChange: (v) => { S.algo = v; S.t0 = H ? H.t : 0; } });
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'ok', label: 'saudável' }, { v: 'bad', label: 'problema' }], value: 'ok', onChange: (v) => { S.scen = v; S.t0 = H ? H.t : 0; } });
      UI.button(ctl, '▶ recomeçar', () => { S.t0 = H ? H.t : 0; });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const D = DEF[S.algo], series = D[S.scen];
        const ph = Math.min(1, ((t - S.t0) % 12) / 8.5);
        const k = Math.max(1, Math.floor(ph * (N - 1)));
        const diagH = narrow ? 164 : 84;
        const gx = 16, gy = 10, gw = w - 32, gh = h - diagH - 22;
        const pw = (gw - 12) / 2, phh = (gh - 12) / 2;
        let worst = 'ok';
        series.forEach((m, i) => {
          const x = gx + (i % 2) * (pw + 12), y = gy + Math.floor(i / 2) * (phh + 12);
          const st = D.rules[i](m.s, k);
          if (st === 'bad' || (st === 'warn' && worst === 'ok')) worst = st;
          V.box(g, x, y, pw, phh, { r: 10, fill: V.alpha(st === 'bad' ? RED() : C.muted, st === 'bad' ? 0.07 : 0.05), stroke: st === 'bad' ? V.alpha(RED(), 0.6) : V.alpha(C.muted, 0.3) });
          if (narrow) nameLines(g, m.n, pw - 34, 10.5).forEach((ln, li) => V.text(g, ln, x + 10, y + 18 + li * 13, { size: 10.5, font: 'mono', weight: 600 }));
          else V.text(g, fit(g, m.n, pw - 80, 11.5, 'mono', 600), x + 10, y + 18, { size: 11.5, font: 'mono', weight: 600 });
          lamp(g, x + pw - 14, y + 14, st, t);
          const cv = m.s[k];
          const vtxt = Math.abs(cv) >= 100 ? String(Math.round(cv)) : fmt(cv, 2);
          V.text(g, vtxt, narrow ? x + pw - 10 : x + pw - 26, narrow ? y + 31 : y + 18, { size: narrow ? 10.5 : 11.5, font: 'mono', align: 'right', color: st === 'bad' ? RED() : C.a, weight: 700 });
          let mn = Math.min(...m.s), mx = Math.max(...m.s);
          const pad = (mx - mn) * 0.12 || 1; mn -= pad; mx += pad;
          const ax = x + 38, aw = pw - 50, ay = y + (narrow ? 44 : 30), ah = phh - (narrow ? 58 : 44);
          V.line(g, ax, ay + ah, ax + aw, ay + ah, { color: V.alpha(C.muted, 0.4), lw: 1 });
          const lbl = (v) => (Math.abs(v) >= 100 ? String(Math.round(v)) : fmt(v, Math.abs(mx - mn) < 0.5 ? 2 : 1));
          V.text(g, lbl(mx), ax - 4, ay + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, lbl(mn), ax - 4, ay + ah, { size: 10, font: 'mono', color: C.muted, align: 'right' });
          V.text(g, 'passo', ax + aw, ay + ah + 12, { size: 10, color: C.muted, align: 'right' });
          const X = (j) => ax + (j / (N - 1)) * aw, Y = (v) => ay + ah - ((v - mn) / (mx - mn)) * ah;
          g.beginPath(); for (let j = 0; j <= k; j++) { const px = X(j), py = Y(m.s[j]); if (j) g.lineTo(px, py); else g.moveTo(px, py); }
          g.strokeStyle = st === 'bad' ? RED() : C.a; g.lineWidth = 2; g.stroke();
          V.glow(g, X(k), Y(m.s[k]), 9, st === 'bad' ? RED() : C.a, 0.9);
          V.dot(g, X(k), Y(m.s[k]), 3, st === 'bad' ? RED() : C.a);
        });
        // diagnóstico
        const dy = h - diagH - 6;
        V.box(g, 16, dy, w - 32, diagH, { r: 10, fill: V.alpha(worst === 'bad' ? RED() : C.a, 0.08), stroke: V.alpha(worst === 'bad' ? RED() : C.a, 0.5) });
        lamp(g, 30, dy + 17, worst, t);
        V.text(g, worst === 'bad' ? 'diagnóstico: problema' : worst === 'warn' ? 'diagnóstico: atenção' : 'diagnóstico: saudável', 42, dy + 21, { size: 12, weight: 700, color: worst === 'bad' ? RED() : worst === 'warn' ? C.d : C.a });
        const txt = ph < 0.35 && S.scen === 'bad' ? 'Ainda cedo: as assinaturas aparecem conforme o treino avança…' : D.diag[S.scen];
        V.wrap(g, txt, 28, dy + 40, w - 56, { size: narrow ? 11 : 12, lineH: narrow ? 14 : 16, maxLines: narrow ? 8 : 3 });
      });
    })();

    /* ======================================== FIG 7 — LoRA por dentro e o merge */
    (function () {
      const fig = root.querySelector('#fig-merge');
      if (!fig) return;
      adaptH(fig, 470, 590);
      const D = 6;
      const rg = V.rng(63);
      const W0 = Array.from({ length: D }, () => Array.from({ length: D }, () => rg.normal() * 0.02));
      const Araw = Array.from({ length: 3 }, () => Array.from({ length: D }, () => (rg() * 2 - 1) / Math.sqrt(D)));
      const Braw = Array.from({ length: D }, () => Array.from({ length: 3 }, () => rg.normal()));
      let rh = Array.from({ length: D }, () => rg.normal()); const nr = Math.hypot(...rh); rh = rh.map((v) => v / nr);
      const S = { mode: 'lora', r: 2, mag: 0.08, bits: 4, alpha: 1, t0: -5 };
      const R = {};
      const rms = (M) => Math.sqrt(M.flat().reduce((s, v) => s + v * v, 0) / M.flat().length);
      function compute() {
        let A, B, dW;
        if (S.mode === 'lora') {
          A = Araw.slice(0, S.r).map((row) => row.slice());
          B = Braw.map((row) => row.slice(0, S.r));
          const raw = V.matmul(B, A);
          const k = (S.mag * 0.02) / (rms(raw) || 1);
          B = B.map((row) => row.map((v) => v * k));
          dW = raw.map((row) => row.map((v) => v * k));
        } else {
          const proj = Array.from({ length: D }, (_, j) => rh.reduce((s, ri, i) => s + ri * W0[i][j], 0));
          B = rh.map((v) => [-S.alpha * v]);
          A = [proj];
          dW = rh.map((ri) => proj.map((pj) => -S.alpha * ri * pj));
        }
        const W1 = W0.map((row, i) => row.map((v, j) => v + dW[i][j]));
        let q0, q1, d0, d1, changed = 0, moved = 0, num = 0, den = 0, half = 0;
        if (S.bits === 16) {
          q0 = W0.map((row) => row.map((v) => LA.num.round('bf16', v)));
          q1 = W1.map((row) => row.map((v) => LA.num.round('bf16', v)));
          d0 = q0; d1 = q1;
          for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) { const u = LA.num.round('bf16', Math.abs(W0[i][j])); half += Math.pow(2, Math.floor(Math.log2(u || 1e-9)) - 8); }
        } else {
          // RTN afim por grupo (uma linha = um grupo), como o MLX: escala = (máx − mín)/(2^b − 1), bias = mín
          const L = Math.pow(2, S.bits) - 1;
          const aff = (row) => { const mn = Math.min(...row), mx = Math.max(...row), sc = (mx - mn) / L || 1e-12; const c = row.map((v) => Math.max(0, Math.min(L, Math.round((v - mn) / sc)))); return { c, d: c.map((k) => k * sc + mn), sc }; };
          q0 = []; q1 = []; d0 = []; d1 = [];
          W0.forEach((row, i) => {
            const a0 = aff(row);
            half += (a0.sc / 2) * D;
            // fuse sem --dequantize: dequantiza, soma ΔW e re-quantiza com escala e bias recalculados
            const a1 = aff(a0.d.map((v, j) => v + dW[i][j]));
            q0.push(a0.c); q1.push(a1.c); d0.push(a0.d); d1.push(a1.d);
          });
        }
        for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
          const dd = d1[i][j] - d0[i][j];
          if (Math.abs(q1[i][j] - q0[i][j]) > 1e-12) changed++;
          else if (Math.abs(dd) > 1e-12) moved++;
          num += (dd - dW[i][j]) * (dd - dW[i][j]); den += dW[i][j] * dW[i][j];
        }
        const mv = d1.map((row, i) => row.map((v, j) => Math.abs(q1[i][j] - q0[i][j]) <= 1e-12 && Math.abs(v - d0[i][j]) > 1e-12));
        Object.assign(R, { A, B, dW, q0, q1, mv, changed, moved, surv: den ? 1 - num / den : 0, ratio: rms(dW) / (half / (D * D)) });
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Delta', options: [{ v: 'lora', label: 'LoRA treinado' }, { v: 'ablit', label: 'abliteração (posto 1)' }], value: 'lora', onChange: (v) => { S.mode = v; gL.show(v === 'lora'); gA.show(v === 'ablit'); compute(); upd(); S.t0 = H ? H.t : 0; } });
      const gL = UI.group(ctl), gA = UI.group(ctl);
      nocase(UI.slider(gL.el, { label: 'posto r', min: 1, max: 3, step: 1, value: S.r, onInput: (v) => { S.r = v; compute(); upd(); } }));
      nocase(UI.slider(gL.el, { label: '|ΔW| / desvio de W', min: 0.02, max: 0.5, step: 0.01, value: S.mag, fmt: (v) => pct(v), onInput: (v) => { S.mag = v; compute(); upd(); } }));
      nocase(UI.slider(gA.el, { label: 'α', min: 0, max: 3, step: 0.1, value: S.alpha, fmt: (v) => fmt(v, 1), onInput: (v) => { S.alpha = v; compute(); upd(); } }));
      gA.show(false);
      UI.seg(ctl, { label: 'Grade onde se funde', options: [{ v: 4, label: '4 bits (RTN)' }, { v: 8, label: '8 bits (RTN)' }, { v: 16, label: 'BF16' }], value: 4, onChange: (v) => { S.bits = v; compute(); upd(); } });
      UI.button(ctl, '▶ treinar de novo', () => { S.t0 = H ? H.t : 0; S.t0 -= 0; S.restart = true; });
      const rS = nocase(UI.readout(ctl, 'ΔW preservado', '—')), rC = nocase(UI.readout(ctl, 'códigos alterados', '—')), rR = nocase(UI.readout(ctl, 'posto de ΔW', '—'));
      function upd() {
        rS.set(R.surv < 0 ? '< 0% (ruído novo)' : pct(R.surv), R.surv > 0.85 ? 'ok' : R.surv > 0.5 ? 'warn' : 'bad');
        rC.set(R.changed + ' de ' + D * D, S.bits === 16 ? (R.changed > 24 ? 'ok' : 'warn') : R.changed > 24 ? 'ok' : R.changed > 10 ? 'warn' : 'bad');
        rR.set(S.mode === 'lora' ? String(S.r) : '1');
      }
      compute(); upd();
      let H = null;
      function mat(g, x, y, cell, data, o) {
        const C = V.C();
        const geo = V.matrix(g, { x, y, data, cell, gap: 2, cmap: 'div', min: -o.lim, max: o.lim, r: 3 });
        if (o.vals && cell >= 26) {
          data.forEach((row, i) => row.forEach((v, j) => { const [cx, cy] = geo.center(i, j); const strong = Math.abs(v / o.lim) > 0.55; V.text(g, o.vals(v), cx, cy + 0.5, { size: Math.min(11, cell * 0.34), font: 'mono', align: 'center', base: 'middle', color: strong ? (C.dark ? '#00210F' : '#ffffff') : C.ink }); }));
        }
        if (o.label) V.text(g, o.label, x, y - 8, { size: 11.5, font: 'mono', weight: 700, color: o.lcol || C.ink });
        return geo;
      }
      function codes(g, x, y, cell, Q, Q0, t, isBF, MV) {
        const C = V.C();
        const n = Q.length;
        for (let i = 0; i < n; i++) for (let j = 0; j < Q[0].length; j++) {
          const cx = x + j * (cell + 2), cy = y + i * (cell + 2);
          const v = Q[i][j];
          const ch = Q0 && Math.abs(v - Q0[i][j]) > 1e-12;
          const Lh = (Math.pow(2, S.bits) - 1) / 2;
          const lv = isBF ? V.clamp(v / 0.05, -1, 1) : V.clamp((v - Lh) / Lh, -1, 1);
          V.box(g, cx, cy, cell, cell, { r: 3, fill: V.cmap(lv, 'div') });
          const txt = isBF ? fmt(v * 100, 1) : String(v);
          const fs = Math.min(11, cell * (isBF ? 0.36 : txt.length > 3 ? 0.36 : 0.44));
          if (cell >= 18) V.text(g, txt, cx + cell / 2, cy + cell / 2 + 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: Math.abs(lv) > 0.55 ? (C.dark ? '#00210F' : '#ffffff') : C.ink, weight: ch ? 700 : 400 });
          if (ch) { const pulse = 0.6 + 0.4 * Math.sin(t * 5 + i + j); g.strokeStyle = V.alpha(C.c, pulse); g.lineWidth = 2.2; V.rr(g, cx - 2, cy - 2, cell + 4, cell + 4, 5); g.stroke(); }
          else if (MV && MV[i][j]) { g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.d, 0.85); g.lineWidth = 1.4; V.rr(g, cx - 1.5, cy - 1.5, cell + 3, cell + 3, 4); g.stroke(); g.setLineDash([]); }
        }
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.restart) { S.t0 = t; S.restart = false; }
        const u = (t - S.t0 + 5) % 8;
        const grow = u < 2 ? V.ease.inOut(u / 2) : 1;
        const Bg = R.B.map((row) => row.map((v) => v * grow));
        const dWg = R.dW.map((row) => row.map((v) => v * grow));
        const lim = 0.05, dlim = Math.max(1e-6, ...R.dW.flat().map(Math.abs));
        const r = R.A.length;
        const vals = (v) => fmt(v * 100, 1);
        const dvals = (v) => fmt(v * 100, Math.abs(v * 100) < 1 ? 2 : 1);
        if (!narrow) {
          const ops = 4 * 22;
          const cell = Math.max(18, Math.min(34, (w - 32 - ops - 8 * 2 * 4) / (18 + r), ((h - 100) / 2 - 10) / 6));
          const mw = (n) => n * cell + (n - 1) * 2;
          let x = 16; const y = 30;
          mat(g, x, y, cell, W0, { lim, vals, label: 'W₀ (×10⁻²)' }); x += mw(D);
          V.text(g, '+', x + 11, y + mw(D) / 2, { size: 18, align: 'center', base: 'middle', color: C.muted }); x += 22;
          const By = y + (mw(D) - mw(D)) / 2;
          mat(g, x, By, cell, Bg, { lim: dlim * 2, label: 'B' + (S.mode === 'lora' ? ' (6×' + r + ')' : ' = −α r̂'), lcol: C.c }); x += mw(r);
          V.text(g, '·', x + 11, y + mw(D) / 2, { size: 20, align: 'center', base: 'middle', color: C.muted }); x += 22;
          const Ay = y + (mw(D) - mw(r)) / 2;
          mat(g, x, Ay, cell, R.A, { lim: Math.max(...R.A.flat().map(Math.abs)), label: S.mode === 'lora' ? 'A (' + r + '×6)' : 'A = r̂ᵀW₀', lcol: C.c }); x += mw(D);
          V.text(g, '=', x + 11, y + mw(D) / 2, { size: 18, align: 'center', base: 'middle', color: C.muted }); x += 22;
          const g1 = mat(g, x, y, cell, dWg, { lim: dlim, vals: dvals, label: 'ΔW = s·B·A (×10⁻²)', lcol: C.c });
          if (u < 2) { for (let p = 0; p < 6; p++) { const f = (u * 1.3 + p / 6) % 1; const sx = 16 + mw(D) + 22 + mw(r) / 2, sy = y + mw(D) / 2; const [ex, ey] = g1.center(p % D, (p * 2) % D); V.dot(g, V.lerp(sx, ex, f), V.lerp(sy, ey, f), 2.5, C.c); } }
          // linha 2
          const y2 = y + mw(D) + 56;
          const isBF = S.bits === 16;
          const lbl0 = isBF ? 'BF16(W₀) (×10⁻²)' : 'códigos Q(W₀) · ' + S.bits + ' bits';
          const lbl1 = isBF ? 'BF16(W₀ + ΔW)' : 'Q(deq(Q(W₀))+ΔW), escala nova';
          V.text(g, lbl0, 16, y2 - 8, { size: 11.5, font: 'mono', weight: 700 });
          codes(g, 16, y2, cell, R.q0, null, t, isBF);
          const ax0 = 16 + mw(D) + 10, ax1 = ax0 + 60;
          V.arrow(g, ax0, y2 + mw(D) / 2, ax1, y2 + mw(D) / 2, { color: u > 2 && u < 3 ? C.c : C.muted, lw: 2, head: 8 });
          V.text(g, 'fundir', (ax0 + ax1) / 2, y2 + mw(D) / 2 - 10, { size: 10.5, align: 'center', color: C.muted });
          V.text(g, fit(g, lbl1, mw(D) + 60, 11.5, 'mono', 700), ax1 + 10, y2 - 8, { size: 11.5, font: 'mono', weight: 700, color: C.c });
          codes(g, ax1 + 10, y2, cell, R.q1, R.q0, t, isBF, R.mv);
          // medidor
          const gx = ax1 + 10 + mw(D) + 28, gw = w - 16 - gx;
          if (gw > 120) {
            V.text(g, 'ΔW preservado: 1 − ‖erro‖² / ‖ΔW‖²', gx, y2 + 6, { size: 11.5, weight: 700 });
            V.box(g, gx, y2 + 18, gw, 18, { r: 6, fill: V.alpha(C.muted, 0.12) });
            const sv = Math.max(0, Math.min(1, R.surv));
            V.box(g, gx, y2 + 18, Math.max(3, gw * sv * grow), 18, { r: 6, fill: sv > 0.85 ? C.a : sv > 0.5 ? C.d : RED() });
            V.text(g, R.surv < 0 ? '< 0%: ruído novo' : pct(sv), gx + gw, y2 + 52, { size: 13, font: 'mono', weight: 700, align: 'right', color: R.surv < 0 ? RED() : C.ink });
            V.wrap(g, S.mode === 'lora' ? '|ΔW| típico ≈ ' + fmt(R.ratio, 2) + '× o meio degrau da grade. ' + (S.bits === 4 ? R.changed + ' códigos mudaram; ' + R.moved + ' valores se mexeram só porque a grade foi recalculada. Abaixo de ~1×, o delta se perde e o re-arredondamento soma ruído novo.' : S.bits === 8 ? 'Em 8 bits o degrau é ~17× menor: o re-arredondamento custa pouco e o delta entra quase inteiro.' : 'Em BF16 o degrau é ~2⁻⁸ do próprio valor: o delta passa quase inteiro.') : 'Edição de posto 1: a componente de W₀ ao longo de r̂ vira (1 − α) = ' + fmt(1 - S.alpha, 1) + '× a original. No GLM, o o_proj é BF16 nativo: a edição entra exata.', gx, y2 + 76, gw, { size: 11, lineH: 15, color: C.muted, maxLines: 6 });
          }
        } else {
          const cell = Math.max(16, Math.min(24, (w - 32 - 30) / 12 - 2));
          const mw = (n) => n * cell + (n - 1) * 2;
          const y = 26;
          mat(g, 16, y, cell, W0, { lim, label: 'W₀' });
          V.text(g, '+', 16 + mw(D) + 15, y + mw(D) / 2, { size: 16, align: 'center', base: 'middle', color: C.muted });
          const x2 = 16 + mw(D) + 30;
          mat(g, x2, y, cell, dWg, { lim: dlim, label: 'ΔW = s·B·A', lcol: C.c });
          const sc = 14, y1 = y + mw(D) + 26;
          const mws = (n) => n * sc + (n - 1) * 2;
          const bx = x2 + (mw(D) - (mws(r) + 16 + mws(D))) / 2;
          mat(g, bx, y1, sc, Bg, { lim: dlim * 2, label: 'B', lcol: C.c });
          V.text(g, '·', bx + mws(r) + 8, y1 + mws(D) / 2, { size: 16, align: 'center', base: 'middle', color: C.muted });
          mat(g, bx + mws(r) + 16, y1 + (mws(D) - mws(r)) / 2, sc, R.A, { lim: Math.max(...R.A.flat().map(Math.abs)), label: 'A', lcol: C.c });
          V.arrow(g, x2 + mw(D) / 2, y1 - 14, x2 + mw(D) / 2, y + mw(D) + 6, { color: V.alpha(C.c, 0.7), lw: 1.5, head: 6 });
          const y2 = y1 + mws(D) + 40;
          const isBF = S.bits === 16;
          V.text(g, isBF ? 'BF16(W₀)' : 'Q(W₀) · ' + S.bits + ' bits', 16, y2 - 8, { size: 11, font: 'mono', weight: 700 });
          codes(g, 16, y2, cell, R.q0, null, t, isBF);
          V.text(g, '→', 16 + mw(D) + 15, y2 + mw(D) / 2, { size: 16, align: 'center', base: 'middle', color: C.muted });
          V.text(g, isBF ? 'BF16(W₀+ΔW)' : 'fundido (escala nova)', x2, y2 - 8, { size: 11, font: 'mono', weight: 700, color: C.c });
          codes(g, x2, y2, cell, R.q1, R.q0, t, isBF, R.mv);
          const gy = y2 + mw(D) + 22, gw = w - 32;
          V.text(g, 'ΔW preservado: ' + (R.surv < 0 ? '< 0% (ruído novo)' : pct(R.surv)), 16, gy, { size: 12, weight: 700, color: R.surv < 0 ? RED() : C.ink });
          V.box(g, 16, gy + 8, gw, 14, { r: 5, fill: V.alpha(C.muted, 0.12) });
          const sv = Math.max(0, Math.min(1, R.surv));
          V.box(g, 16, gy + 8, Math.max(3, gw * sv * grow), 14, { r: 5, fill: sv > 0.85 ? C.a : sv > 0.5 ? C.d : RED() });
          V.wrap(g, S.mode === 'lora' ? '|ΔW| típico ≈ ' + fmt(R.ratio, 2) + '× o meio degrau da grade.' + (isBF ? '' : ' ' + R.changed + ' códigos mudaram; ' + R.moved + ' valores andaram com a grade.') : 'Posto 1: a componente ao longo de r̂ vira (1 − α) = ' + fmt(1 - S.alpha, 1) + '×.', 16, gy + 40, gw, { size: 11, lineH: 14, color: C.muted, maxLines: 3 });
        }
      });
    })();

    /* ======================================== FIG 8 — formatos e mistura de calibração */
    (function () {
      const fig = root.querySelector('#fig-calib');
      if (!fig) return;
      adaptH(fig, 470, 726);
      const NL = 64;
      const S = { f: 0.5, hover: null };
      // pools de ativações (ilustrativos)
      const rg = V.rng(8);
      const NP = 4000;
      const gen = Array.from({ length: NP }, () => rg.normal());
      const dom = Array.from({ length: NP }, () => (rg() < 0.08 ? rg.normal() * 4 : rg.normal()));
      const varG = gen.reduce((s, x) => s + x * x, 0) / NP, varD = dom.reduce((s, x) => s + x * x, 0) / NP;
      const CL = []; for (let c = 1.0; c <= 16.0001; c += 0.1) CL.push(c);
      const mse = (arr, c) => { const d = c / 7; let s = 0; for (const x of arr) { const q = Math.max(-7, Math.min(7, Math.round(x / d))) * d; s += (q - x) * (q - x); } return s / arr.length; };
      const MG = CL.map((c) => mse(gen, c)), MD = CL.map((c) => mse(dom, c));
      const FG = []; for (let i = 0; i <= 100; i++) FG.push(i / 100);
      const OPT = FG.map((f) => { let bi = 0, bv = Infinity; CL.forEach((c, i) => { const v = (1 - f) * MG[i] + f * MD[i]; if (v < bv) { bv = v; bi = i; } }); return { c: CL[bi], eg: MG[bi] / varG, ed: MD[bi] / varD }; });
      const maxE = Math.max(...OPT.map((o) => Math.max(o.eg, o.ed)));
      const HB = 48, HR = 12;
      const hist = (arr) => { const hh = new Array(HB).fill(0); arr.forEach((x) => { const b = Math.floor(((x + HR) / (2 * HR)) * HB); if (b >= 0 && b < HB) hh[b]++; }); return hh; };
      const HG = hist(gen), HD = hist(dom);
      const hmax = Math.max(...HG, ...HD);
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'fração de domínio na calibração', min: 0, max: 1, step: 0.01, value: S.f, fmt: (v) => pct(v), onInput: (v) => { S.f = v; upd(); } }));
      const rC = nocase(UI.readout(ctl, 'corte escolhido', '—')), rG = nocase(UI.readout(ctl, 'erro no genérico', '—')), rD = nocase(UI.readout(ctl, 'erro no domínio', '—'));
      UI.legend(ctl, [[V.C().a, 'NVFP4'], [V.C().b, 'FP8 dinâmico'], [V.C().d, 'BF16 (fora da quantização)'], [V.C().c, 'KV FP8 estático (atenção)']]);
      function cur() { return OPT[Math.round(S.f * 100)]; }
      function upd() { const o = cur(); rC.set(fmt(o.c, 1) + ' σ'); rG.set(pct(o.eg, 1), o.eg < 0.03 ? 'ok' : o.eg < 0.06 ? 'warn' : 'bad'); rD.set(pct(o.ed, 1), o.ed < 0.03 ? 'ok' : o.ed < 0.06 ? 'warn' : 'bad'); }
      upd();
      const FMT = { nv: 'NVFP4', f8: 'FP8 dinâmico', bf: 'BF16 (fora)' };
      const parts = Array.from({ length: 40 }, (_, i) => ({ p: (i * 0.618) % 1, lane: i % 4, kind: (i * 0.37) % 1 }));
      let geo = null;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const colF = { nv: C.a, f8: C.b, bf: C.d };
        // ---- mapa de camadas
        const blocks = narrow ? 2 : 1, perB = NL / blocks;
        const extraW = narrow ? 0 : 150;
        const mapX = 16 + (narrow ? 44 : 58), mapW = w - 16 - mapX - extraW;
        const cw = mapW / perB;
        const rowH = narrow ? 18 : 22;
        const tipY = 16;
        let y0 = narrow ? 66 : 40;
        geo = [];
        const P = S.hover;
        let tip = '64 camadas = 48 GDN + 16 de atenção completa (uma a cada 4). Passe o mouse (ou toque) numa camada.';
        for (let b = 0; b < blocks; b++) {
          const by = y0 + b * (2 * rowH + 22);
          V.text(g, 'atn/GDN', 16, by + rowH / 2 + 0.5, { size: narrow ? 10 : 11, color: C.muted, base: 'middle' });
          V.text(g, narrow ? 'calib.' : 'calibração', 16, by - 6, { size: 9.5, color: C.muted, base: 'middle' });
          V.text(g, 'MLP', 16, by + rowH * 1.5 + 3.5, { size: narrow ? 10 : 11, color: C.muted, base: 'middle' });
          for (let k = 0; k < perB; k++) {
            const L = b * perB + k;
            const attn = L % 4 === 3;
            const x = mapX + k * cw;
            const mlpF = L >= 56 ? 'f8' : 'nv';
            const flash = parts.some((pp) => Math.abs(((t * 0.08 + pp.p) % 1) - (L + 0.5) / NL) < 0.006) ? 0.25 : 0;
            V.box(g, x + 0.5, by, cw - 1, rowH - 2, { r: 2, fill: V.alpha(colF.f8, attn ? 0.95 : 0.55 + flash) });
            if (!attn) { g.fillStyle = colF.bf; g.fillRect(x + 0.5, by + rowH - 5, cw - 1, 2.5); }
            if (attn) V.dot(g, x + cw / 2, by + rowH / 2 - 1, Math.min(3, cw / 3), C.c);
            V.box(g, x + 0.5, by + rowH + 2, cw - 1, rowH - 2, { r: 2, fill: V.alpha(colF[mlpF], 0.6 + flash) });
            geo.push({ x, y: by, w: cw, h: 2 * rowH + 2, L, attn, mlpF });
            if (P && P.x >= x && P.x < x + cw && P.y >= by && P.y < by + 2 * rowH + 2) {
              const onMlp = P.y >= by + rowH;
              V.box(g, x - 1, by - 2, cw + 1, 2 * rowH + 6, { r: 3, stroke: C.ink, lw: 1.5 });
              tip = onMlp ? 'camada ' + L + ' · MLP: gate/up/down_proj → ' + FMT[mlpF] + (L >= 56 ? ' (uma das 8 últimas)' : '')
                : attn ? 'camada ' + L + ' · atenção completa: q/k/v/o_proj → FP8 dinâmico; KV → FP8 estático (static_minmax)'
                  : 'camada ' + L + ' · Gated DeltaNet: in_proj_qkv/in_proj_z/out_proj → FP8 dinâmico; in_proj_a/b (48 saídas) → BF16';
            }
          }
          if (b === blocks - 1 || narrow) { V.text(g, String(b * perB), mapX, by + 2 * rowH + 13, { size: 10, font: 'mono', color: C.muted }); V.text(g, String((b + 1) * perB - 1), mapX + mapW, by + 2 * rowH + 13, { size: 10, font: 'mono', color: C.muted, align: 'right' }); }
        }
        if (narrow) V.wrap(g, tip, 16, tipY, w - 32, { size: 10.5, lineH: 13, maxLines: 3, weight: 600, color: P ? C.ink : C.muted });
        else V.text(g, fit(g, tip, w - 32, 11.5, 'sans', 600), 16, tipY, { size: 11.5, weight: 600, color: P ? C.ink : C.muted });
        // extras
        const ex = narrow ? 16 : mapX + mapW + 14;
        const ey = narrow ? y0 + blocks * (2 * rowH + 22) + 2 : y0;
        const extras = [['embed_tokens', 'bf'], ['lm_head', 'bf'], ['MTP (mtp.*)', 'bf'], ['torre de visão', 'bf']];
        extras.forEach(([nm, f], i) => {
          const xx = narrow ? ex + (i % 2) * ((w - 32) / 2) : ex, yy = narrow ? ey + Math.floor(i / 2) * 22 : ey + i * 22;
          V.box(g, xx, yy, 10, 10, { r: 2, fill: colF[f] });
          V.text(g, nm + ' → fora', xx + 15, yy + 9, { size: 10.5, color: C.ink });
        });
        // partículas de calibração: correm numa trilha logo acima de cada bloco de camadas
        parts.forEach((pp) => {
          const f = (t * 0.08 + pp.p) % 1;
          const isDom = pp.kind < S.f;
          const b = narrow ? (f < 0.5 ? 0 : 1) : 0;
          const ff = narrow ? (f % 0.5) * 2 : f;
          const by = y0 + b * (2 * rowH + 22);
          const px = mapX + ff * mapW, py = by - 6 + (pp.lane % 2 ? -1.5 : 1.5);
          g.beginPath(); g.moveTo(px + 3.5, py); g.lineTo(px - 2.5, py - 3); g.lineTo(px - 2.5, py + 3); g.closePath();
          g.fillStyle = isDom ? C.c : V.alpha(C.ink, 0.8); g.fill();
        });
        // ---- painel inferior
        const ly = narrow ? ey + 54 : y0 + 2 * rowH + 44;
        const lh = h - ly - 12;
        const half = narrow ? w - 32 : (w - 48) / 2;
        const hx = 16, hy = ly, hh = narrow ? (lh - 20) / 2 : lh;
        const o = cur();
        // histograma
        V.text(g, 'ativações de um canal: genérico × domínio', hx, hy + 10, { size: 11.5, weight: 700 });
        const gx0 = hx + 4, gw = half - 8, gy0 = hy + 22, gh = hh - 58;
        const bwid = gw / HB;
        // domínio em barras cheias; genérico como contorno em degraus por cima (as duas distribuições ficam visíveis)
        for (let i = 0; i < HB; i++) {
          const vd = Math.sqrt(HD[i] / hmax);
          g.fillStyle = V.alpha(C.c, 0.5); g.fillRect(gx0 + i * bwid + 0.5, gy0 + gh - vd * gh, bwid - 1, vd * gh);
        }
        g.beginPath(); g.moveTo(gx0, gy0 + gh);
        for (let i = 0; i < HB; i++) { const yy = gy0 + gh - Math.sqrt(HG[i] / hmax) * gh; g.lineTo(gx0 + i * bwid, yy); g.lineTo(gx0 + (i + 1) * bwid, yy); }
        g.lineTo(gx0 + HB * bwid, gy0 + gh);
        g.strokeStyle = V.alpha(C.ink, 0.85); g.lineWidth = 1.6; g.lineJoin = 'miter'; g.stroke();
        const X = (v) => gx0 + ((v + HR) / (2 * HR)) * gw;
        [-1, 1].forEach((sg) => { const xx = X(sg * o.c); V.line(g, xx, gy0 - 4, xx, gy0 + gh, { color: C.d, lw: 2 }); });
        for (let q = -7; q <= 7; q++) { const xx = X((q * o.c) / 7); V.line(g, xx, gy0 + gh + 2, xx, gy0 + gh + 8, { color: C.d, lw: 1 }); }
        V.text(g, '±corte ' + fmt(o.c, 1) + 'σ e os 15 níveis da grade', X(0), gy0 + gh + 22, { size: 10.5, align: 'center', color: C.d });
        V.box(g, gx0, gy0 + gh + 30, 9, 9, { r: 2, stroke: V.alpha(C.ink, 0.85), lw: 1.6 }); V.text(g, 'genérico (contorno)', gx0 + 13, gy0 + gh + 38, { size: 10.5, color: C.muted });
        V.box(g, gx0 + 128, gy0 + gh + 30, 9, 9, { r: 2, fill: V.alpha(C.c, 0.7) }); V.text(g, fit(g, 'domínio (8% ~4× maiores)', gw - 141, 10.5), gx0 + 141, gy0 + gh + 38, { size: 10.5, color: C.muted });
        // curvas de erro
        const cx0 = narrow ? 16 : 16 + half + 16, cy0 = narrow ? hy + hh + 20 : hy, cw2 = half, ch2 = narrow ? hh : lh;
        V.text(g, 'erro relativo (MSE / variância) × mistura', cx0, cy0 + 10, { size: 11.5, weight: 700 });
        const px0 = cx0 + 40, pw = cw2 - 52, py0 = cy0 + 24, ph2 = ch2 - 62;
        V.line(g, px0, py0 + ph2, px0 + pw, py0 + ph2, { color: V.alpha(C.muted, 0.5), lw: 1 });
        V.line(g, px0, py0, px0, py0 + ph2, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const XE = (f) => px0 + f * pw, YE = (e) => py0 + ph2 - (e / maxE) * ph2;
        [0, maxE / 2, maxE].forEach((e) => V.text(g, pct(e, 0), px0 - 5, YE(e) + 3, { size: 10, font: 'mono', color: C.muted, align: 'right' }));
        ['0%', '50%', '100%'].forEach((s, i) => V.text(g, s, XE(i / 2), py0 + ph2 + 13, { size: 10, font: 'mono', color: C.muted, align: i === 0 ? 'left' : i === 2 ? 'right' : 'center' }));
        V.text(g, 'fração de domínio na calibração', px0 + pw / 2, py0 + ph2 + 27, { size: 10.5, color: C.muted, align: 'center' });
        [['eg', C.ink, 'genérico'], ['ed', C.c, 'domínio']].forEach(([key, col, nm]) => {
          g.beginPath(); OPT.forEach((p, i) => { const xx = XE(FG[i]), yy = YE(p[key]); if (i) g.lineTo(xx, yy); else g.moveTo(xx, yy); });
          g.strokeStyle = col; g.lineWidth = 2; g.stroke();
          V.dot(g, XE(S.f), YE(o[key]), 4.5, col);
          V.text(g, nm, XE(key === 'eg' ? 0.97 : 0.03), YE(OPT[key === 'eg' ? 97 : 3][key]) - 7, { size: 10.5, color: col, align: key === 'eg' ? 'right' : 'left', weight: 700 });
        });
        V.line(g, XE(S.f), py0, XE(S.f), py0 + ph2, { color: C.d, lw: 1.2, dash: [4, 3] });
        if (Math.abs(S.f - 0.5) < 0.02) V.text(g, '256 + 256', XE(0.5) + 4, py0 + 10, { size: 10.5, color: C.d, weight: 700 });
      });
      const cv = fig.querySelector('canvas');
      const setHover = (e) => { const r = cv.getBoundingClientRect(); S.hover = { x: e.clientX - r.left, y: e.clientY - r.top }; };
      cv.addEventListener('pointermove', setHover);
      cv.addEventListener('pointerdown', setHover);
      cv.addEventListener('pointerleave', (e) => { if (e.pointerType === 'touch') return; S.hover = null; }); // no toque, a última camada tocada continua selecionada
    })();

    /* ======================================== FIG 9 — duas KLDs */
    (function () {
      const fig = root.querySelector('#fig-kld');
      if (!fig) return;
      adaptH(fig, 450, 632);
      const CTX = {
        gen: { qs: 'A capital da França é ___', toks: ['Paris', 'Lyon', 'a', 'uma', 'conhecida', 'Marselha'], base: [0.78, 0.04, 0.06, 0.04, 0.05, 0.03], ft: [0.76, 0.04, 0.07, 0.04, 0.06, 0.03], q: [0.71, 0.05, 0.08, 0.05, 0.065, 0.045] },
        dom: { qs: '--max-loras no vLLM vale, por padrão, ___', toks: ['1', '4', '16', 'ilimitado', '8', '0'], base: [0.14, 0.2, 0.18, 0.28, 0.12, 0.08], ft: [0.86, 0.03, 0.04, 0.03, 0.02, 0.02], q: [0.8, 0.045, 0.05, 0.045, 0.035, 0.025] },
      };
      Object.values(CTX).forEach((c) => ['base', 'ft', 'q'].forEach((k) => { const s = c[k].reduce((a, b) => a + b, 0); c[k] = c[k].map((v) => v / s); }));
      const KL = (p, q) => p.reduce((s, pi, i) => s + (pi > 0 ? pi * Math.log(pi / q[i]) : 0), 0);
      const S = { ctx: 'dom', ref: 'merge', cur: null };
      // [valores, rótulo, linha do rótulo]; EXL3 4 bpw e FP8 oficial na mesma stack empatam (mesmo x), então dividem um rótulo
      const ANCH = [[[0.013723], 'EXL3 K6', 0], [[0.024555, 0.024629], 'EXL3 4 bpw ≈ FP8 oficial (mesma stack)', 1], [[0.060535], 'NVFP4', 0]];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Texto', options: [{ v: 'gen', label: 'genérico' }, { v: 'dom', label: 'domínio' }], value: 'dom', onChange: (v) => { S.ctx = v; upd(); } });
      UI.seg(ctl, { label: 'Referência da quantização', options: [{ v: 'merge', label: 'merge BF16 (certo)' }, { v: 'base', label: 'base original (errado)' }], value: 'merge', onChange: (v) => { S.ref = v; upd(); } });
      const rD = nocase(UI.readout(ctl, 'KL(base ‖ ajustado)', '—')), rQ = nocase(UI.readout(ctl, 'KL(ref ‖ NVFP4)', '—'));
      const M = {};
      function upd() {
        const c = CTX[S.ctx];
        M.drift = KL(c.base, c.ft);
        M.quant = S.ref === 'merge' ? KL(c.ft, c.q) : KL(c.base, c.q);
        rD.set(fmt(M.drift, 3) + ' nats');
        rQ.set(fmt(M.quant, 3) + ' nats', S.ref === 'merge' ? 'ok' : 'bad');
      }
      upd();
      const cur = { base: CTX.dom.base.slice(), ft: CTX.dom.ft.slice(), q: CTX.dom.q.slice(), drift: 0, quant: 0 };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const c = CTX[S.ctx];
        const kk = Math.min(1, dt * 5);
        ['base', 'ft', 'q'].forEach((k) => { for (let i = 0; i < 6; i++) cur[k][i] += (c[k][i] - cur[k][i]) * kk; });
        cur.drift += (M.drift - cur.drift) * kk; cur.quant += (M.quant - cur.quant) * kk;
        // contexto
        V.text(g, 'próximo token depois de:', 16, 18, { size: 11, color: C.muted, weight: 600 });
        V.text(g, fit(g, '“' + c.qs + '”', w - 32, 13, 'mono', 600), 16, 38, { size: 13, font: 'mono', weight: 600 });
        // barras
        const ruleH = narrow ? 112 : 100;
        const barsW = narrow ? w - 32 : w * 0.58 - 24;
        const bx0 = 16, by0 = 54, bh = narrow ? 210 : h - by0 - ruleH - 44;
        const gw = barsW / 6;
        const cols = [C.b, C.a, C.d];
        const bw = Math.min(16, (gw - 10) / 3);
        V.line(g, bx0, by0 + bh, bx0 + barsW, by0 + bh, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const tops = [];
        for (let i = 0; i < 6; i++) {
          const gx = bx0 + i * gw + (gw - 3 * bw - 4) / 2;
          ['base', 'ft', 'q'].forEach((k, j) => {
            const v = cur[k][i], hh = v * bh;
            V.box(g, gx + j * (bw + 2), by0 + bh - hh, bw, Math.max(1, hh), { r: 3, fill: cols[j] });
            if (j === 1) tops.push([gx + j * (bw + 2) + bw / 2, by0 + bh - hh]);
          });
          V.text(g, fit(g, c.toks[i], gw - 4, narrow ? 10.5 : 11.5, 'mono', 600), bx0 + i * gw + gw / 2, by0 + bh + 15, { size: narrow ? 10.5 : 11.5, font: 'mono', align: 'center', weight: 600 });
        }
        // massa movida (domínio)
        if (S.ctx === 'dom') {
          for (let p = 0; p < 10; p++) {
            const f = (t * 0.45 + p / 10) % 1;
            const src = 1 + (p % 5), sx = bx0 + src * gw + gw / 2, sy = by0 + bh - c.base[src] * bh;
            const [tx2, ty2] = [bx0 + gw / 2, by0 + bh - c.ft[0] * bh];
            const mx = (sx + tx2) / 2, my = Math.min(sy, ty2) - 40;
            const xx = (1 - f) * (1 - f) * sx + 2 * (1 - f) * f * mx + f * f * tx2, yy = (1 - f) * (1 - f) * sy + 2 * (1 - f) * f * my + f * f * ty2;
            g.globalAlpha = Math.sin(f * Math.PI); V.dot(g, xx, yy, 2.6, C.a); g.globalAlpha = 1;
          }
        }
        // legenda das barras
        const lgY = by0 + bh + 32;
        [['base', C.b], ['ajustado BF16', C.a], ['ajustado NVFP4', C.d]].forEach(([nm, col], i) => { const xx = bx0 + i * (narrow ? (w - 32) / 3 : barsW / 3); V.box(g, xx, lgY - 8, 9, 9, { r: 2, fill: col }); V.text(g, nm, xx + 13, lgY, { size: 10.5, color: C.muted }); });
        // medidores
        const mx0 = narrow ? 16 : w * 0.58 + 8, my0 = narrow ? lgY + 18 : by0 + 4, mw = narrow ? w - 32 : w - mx0 - 16;
        const meter = (y, title, sub, v, col, bad) => {
          V.text(g, title, mx0, y, { size: 12, weight: 700 });
          V.text(g, fit(g, sub, mw, 10.5), mx0, y + 15, { size: 10.5, color: C.muted });
          V.box(g, mx0, y + 23, mw, 14, { r: 5, fill: V.alpha(C.muted, 0.12) });
          const lv = V.clamp((Math.log10(Math.max(v, 1e-3)) + 3) / (Math.log10(3) + 3), 0, 1);
          V.box(g, mx0, y + 23, Math.max(3, mw * lv), 14, { r: 5, fill: bad ? RED() : col });
          V.text(g, fmt(v, 3) + ' nats', mx0 + mw, y, { size: 12, font: 'mono', weight: 700, align: 'right', color: bad ? RED() : col });
        };
        meter(my0 + 12, 'o que o pós-treino mudou', 'KL(base ‖ ajustado BF16): a deriva', cur.drift, C.a, false);
        const wrong = S.ref === 'base';
        meter(my0 + (narrow ? 64 : 78), 'quanto a quantização estragou', wrong ? 'KL(base ‖ NVFP4): mistura deriva e estrago!' : 'KL(merge BF16 ‖ NVFP4): só o estrago', cur.quant, C.d, wrong);
        V.wrap(g, wrong ? 'Com o base como referência, a deriva do domínio entra na conta e o NVFP4 leva a culpa por uma mudança que foi o próprio treino.' : 'Com o merge BF16 como referência, sobra só o erro do formato, comparável às âncoras medidas no GLM.', mx0, my0 + (narrow ? 118 : 150), mw, { size: 11, lineH: 15, color: wrong ? RED() : C.muted, maxLines: 5 });
        // régua log
        const rTop = narrow ? my0 + 176 : h - ruleH, ry = rTop + 52, rx0 = 24, rw = w - 48;
        const LX = (v) => rx0 + ((Math.log10(v) + 3) / (Math.log10(3) + 3)) * rw;
        V.text(g, fit(g, 'régua (nats, escala log) com as âncoras medidas no GLM-5.3-Flash', w - 32, 11, 'sans', 600), 16, rTop + 10, { size: 11, color: C.muted, weight: 600 });
        V.line(g, rx0, ry, rx0 + rw, ry, { color: V.alpha(C.muted, 0.6), lw: 1.5 });
        [0.001, 0.01, 0.1, 1].forEach((v) => { const xx = LX(v); V.line(g, xx, ry - 4, xx, ry + 4, { color: C.muted, lw: 1 }); V.text(g, String(v).replace('.', ','), xx, ry + 16, { size: 10, font: 'mono', color: C.muted, align: 'center' }); });
        ANCH.forEach(([vs, nm, row], i) => {
          const xx = LX(vs[0]); vs.forEach((v) => V.dot(g, LX(v), ry, 3.5, C.ink));
          const ly = ry + 30 + row * (narrow ? 11 : 13);
          V.line(g, xx, ry + 4, xx, ly - 9, { color: V.alpha(C.ink, 0.35), lw: 1 });
          V.text(g, nm, xx, ly, { size: 10, color: C.ink, align: i === ANCH.length - 1 ? 'left' : 'center' });
        });
        const mk = (v, col, lb, hgt) => { const xx = LX(V.clamp(v, 0.001, 3)); g.beginPath(); g.moveTo(xx, ry - 3); g.lineTo(xx - 6, ry - 13); g.lineTo(xx + 6, ry - 13); g.closePath(); g.fillStyle = col; g.fill(); V.line(g, xx, ry - 13, xx, ry - hgt + 4, { color: col, lw: 1.2 }); V.text(g, lb, xx, ry - hgt, { size: 10.5, weight: 700, color: col, align: xx > rx0 + rw - 50 ? 'right' : xx < rx0 + 40 ? 'left' : 'center' }); };
        mk(cur.drift, C.a, 'deriva', 18);
        mk(cur.quant, wrong ? RED() : C.d, 'estrago', 30);
      });
    })();

    /* ======================================== FIG 10 — o ônibus multi-LoRA */
    (function () {
      const fig = root.querySelector('#fig-servir');
      if (!fig) return;
      const RANKS = [8, 16, 32, 64];
      const ADS = ['q38-sft', 'dpo-b01', 'dpo-b03', 'dpo-b05', 'kto'];
      const S = { mode: 'A', maxL: 2, ri: 2, nA: 3, real: 16, fmtv: 'nv', mtp: true, q: [], bus: [], seen: {}, acc: 0, id: 0, tick: 0, waiting: 0 };
      let kNarrowHRef = () => 540;
      const resize = adaptH(fig, () => (S.mode === 'K' ? 470 : 440), () => (S.mode === 'K' ? kNarrowHRef() : 392));
      const SEATS = 8;
      const rg = V.rng(77);
      function arrive() { const isBase = rg() < 0.2; S.q.push({ id: S.id++, ad: isBase ? -1 : rg.int(S.nA), life: 4 + rg.int(7), x: null }); if (S.q.length > 12) S.q.shift(); }
      function step() {
        S.tick++;
        S.bus.forEach((r) => { r.life--; });
        S.bus = S.bus.filter((r) => r.life > 0);
        if (rg() < 0.75) arrive(); if (rg() < 0.35) arrive();
        const active = new Set(S.bus.filter((r) => r.ad >= 0).map((r) => r.ad));
        const keep = [];
        let waiting = 0;
        S.q.forEach((r) => {
          const ok = S.bus.length < SEATS && (S.mode === 'C' || r.ad < 0 || active.has(r.ad) || active.size < S.maxL);
          if (ok) { S.bus.push(r); if (r.ad >= 0) { active.add(r.ad); S.seen[r.ad] = S.tick; } }
          else { keep.push(r); if (S.mode === 'A' && S.bus.length < SEATS && r.ad >= 0) waiting++; }
        });
        S.q = keep; S.waiting = waiting;
      }
      for (let i = 0; i < 40; i++) step();
      // batch ilustrativo do modo "por dentro do batch": 6 tokens (índices em ADS; −1 = só o base)
      const KT = [0, 1, -1, 0, 2, 1];
      const kSlots = () => { const order = []; KT.forEach((a) => { if (a >= 0 && !order.includes(a)) order.push(a); }); return order.slice(0, S.maxL); };
      const SUB = ['₁', '₂', '₃', '₄', '₅', '₆'];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Caminho', options: [{ v: 'A', label: 'A: base + adapters' }, { v: 'K', label: 'por dentro do batch' }, { v: 'C', label: 'C: fundido' }], value: 'A', onChange: (v) => { S.mode = v; gA.show(v !== 'C'); gN.show(v === 'A'); gK.show(v === 'K'); gC.show(v === 'C'); resize(); S.q.forEach((r) => { r.x = null; }); for (let i = 0; i < 3; i++) step(); upd(); } });
      const gA = UI.group(ctl), gN = UI.group(ctl), gK = UI.group(ctl), gC = UI.group(ctl);
      nocase(UI.slider(gA.el, { label: '--max-loras', min: 1, max: 8, step: 1, value: S.maxL, onInput: (v) => { S.maxL = v; resize(); upd(); } }));
      UI.seg(gA.el, { label: '--max-lora-rank', options: RANKS.map((r, i) => ({ v: i, label: String(r) })), value: S.ri, onChange: (v) => { S.ri = v; upd(); } });
      nocase(UI.slider(gN.el, { label: 'adapters no tráfego', min: 1, max: 5, step: 1, value: S.nA, onInput: (v) => { S.nA = v; upd(); } }));
      UI.seg(gK.el, { label: 'posto real dos adapters', options: [8, 16, 32].map((r) => ({ v: r, label: 'r=' + r })), value: S.real, onChange: (v) => { S.real = v; upd(); } });
      UI.seg(gC.el, { label: 'Formato', options: [{ v: 'bf', label: 'BF16 55,6 GB' }, { v: 'nv', label: 'NVFP4 24,69 GB' }], value: 'nv', onChange: (v) => { S.fmtv = v; upd(); } });
      UI.toggle(gC.el, { label: 'MTP k=2', value: true, onChange: (v) => { S.mtp = v; upd(); } });
      gK.show(false); gC.show(false);
      const r1 = nocase(UI.readout(ctl, 'memória dos slots', '—')), r2 = nocase(UI.readout(ctl, 'teto de 1 stream', '—'));
      const r2lab = r2.el.querySelector('span');
      function upd() {
        const rmax = RANKS[S.ri];
        if (r2lab) r2lab.textContent = S.mode === 'K' ? 'shrink + expand por token' : 'teto de 1 stream';
        if (S.mode === 'A' || S.mode === 'K') {
          const gb = (S.maxL * 6799360 * rmax * 2) / 1e9; r1.set(fmt(gb, 2) + ' GB', gb > 3 ? 'warn' : 'ok');
          if (S.mode === 'A') r2.set('base BF16 ≈ 4 tok/s');
          else if (S.real > rmax) r2.set('r=' + S.real + ' > r_max: recusado', 'bad');
          else r2.set(Math.round((2 * 6799360 * rmax) / 1e6) + ' MFLOP · ' + pct(1 - S.real / rmax) + ' zeros', S.real < rmax ? 'warn' : 'ok');
        } else { const gb = S.fmtv === 'bf' ? 55.6 : 24.69; const c = 230 / gb; r1.set('0 (sem slots)', 'ok'); r2.set(S.mtp ? fmt(c * 1.5, 0) + '–' + fmt(c * 2, 0) + ' tok/s [estim.]' : '≈ ' + fmt(c, 1) + ' tok/s', S.fmtv === 'bf' ? 'warn' : 'ok'); }
      }
      upd();
      // hachura de zeros (preenchimento até r_max)
      function hatch(g, x, y, w, h, col) {
        if (w < 1 || h < 1) return;
        g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
        g.strokeStyle = col; g.lineWidth = 1;
        g.beginPath(); for (let k = -h; k < w; k += 5) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); } g.stroke();
        g.restore();
      }
      // pilha de fatias (um tensor [max_loras × … × r_max]); devolve o retângulo de cada fatia
      function stack(g, x, y, bw, sh, gap, slots, AC, C, rmax, real, hi, title, dims, side) {
        V.text(g, title, x, y - 20, { size: 11, font: 'mono', weight: 700, color: C.ink });
        V.text(g, dims, x, y - 7, { size: 10, font: 'mono', color: C.muted });
        const wmax = bw * (rmax / 64), wreal = bw * (Math.min(real, rmax) / 64);
        const rects = [];
        for (let k = 0; k < S.maxL; k++) {
          const yy = y + k * (sh + gap), ad = slots[k];
          const on = k === hi;
          V.box(g, x, yy, Math.max(4, wmax), sh, { r: 3, fill: V.alpha(C.muted, 0.08), stroke: on ? C.ink : V.alpha(C.muted, 0.45), lw: on ? 2 : 1 });
          if (ad != null) {
            g.fillStyle = V.alpha(AC[ad], on ? 0.9 : 0.55); g.fillRect(x + 1, yy + 1, Math.max(2, wreal - 2), sh - 2);
            hatch(g, x + wreal, yy + 1, Math.max(0, wmax - wreal - 1), sh - 2, V.alpha(C.muted, 0.55));
          } else hatch(g, x + 1, yy + 1, Math.max(0, wmax - 2), sh - 2, V.alpha(C.muted, 0.3));
          const fs = sh >= 12 ? 10 : 9;
          if (side === 'left') V.text(g, 'slot ' + k, x - 6, yy + sh / 2 + 0.5, { size: fs, font: 'mono', base: 'middle', align: 'right', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
          else V.text(g, ad != null ? ADS[ad] : 'livre', x + bw + 6, yy + sh / 2 + 0.5, { size: fs, font: 'mono', base: 'middle', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
          rects.push([x, yy, Math.max(4, wmax), sh]);
        }
        return rects;
      }
      // geometria do modo "por dentro do batch" (também dá a altura do palco no celular)
      const kSh = (narrow, h) => (narrow ? Math.max(8, Math.min(18, (120 - (S.maxL - 1) * 4) / S.maxL)) : Math.max(9, Math.min(28, (h - 56 - 260 - (S.maxL - 1) * 4) / S.maxL)));
      const kNarrowH = () => { const sh = kSh(true); return Math.round(170 + 2 * S.maxL * (sh + 4) + 34 + 16 + 26 + 18 + 4 * 14 + 14); };
      kNarrowHRef = kNarrowH;
      function drawK(g, w, h, t, C, narrow, AC) {
        const rmax = RANKS[S.ri], real = S.real, ok = real <= rmax;
        const slots = kSlots();
        const per = 2.4, ti = Math.floor(t / per) % KT.length, tf = (t % per) / per;
        const ad = KT[ti], sl = ad < 0 ? -1 : slots.indexOf(ad);
        const kind = ad < 0 ? 'base' : sl < 0 ? 'wait' : ok ? 'lora' : 'refused';
        // ---- geometria
        let tok, out, Wr, A0, vR, B0, bw, textY, joinX = null;
        const gap = 4, sh = kSh(narrow, h);
        if (!narrow) {
          const r0 = 36, rh = 28, outW = 150;
          tok = KT.map((_, i) => [16, r0 + i * rh, 112, rh - 6]);
          out = KT.map((_, i) => [w - 16 - outW, r0 + i * rh, outW, rh - 6]);
          Wr = [160, r0, Math.min(250, w - 16 - outW - 90 - 160), 6 * rh - 6];
          joinX = w - 16 - outW - 26;
          const laneTop = r0 + 6 * rh + 56;
          bw = Math.min(150, (w - 32) * 0.2);
          A0 = [210, laneTop]; vR = [210 + bw + 46, laneTop + 4, 76, 16]; B0 = [vR[0] + vR[2] + 46, laneTop];
          textY = h - 34;
        } else {
          const cw = (w - 32 - 5 * 5) / 6;
          tok = KT.map((_, i) => [16 + i * (cw + 5), 40, cw, 30]);
          Wr = [16, 84, w - 32, 40];
          const laneTop = 170;
          bw = (w - 32) * 0.36;
          A0 = [58, laneTop]; vR = [58 + bw + 22, laneTop + 4, Math.max(48, w - 16 - (58 + bw + 22)), 14];
          B0 = [58, laneTop + S.maxL * (sh + gap) + 34];
          const oy = B0[1] + S.maxL * (sh + gap) + 16;
          out = KT.map((_, i) => [16 + i * (cw + 5), oy, cw, 26]);
          textY = oy + 44;
        }
        const wmax = bw * (rmax / 64);
        const slab = (x0, y0, k) => [x0, y0 + k * (sh + gap), Math.max(4, wmax), sh];
        const rowY = (i) => tok[i][1] + tok[i][3] / 2;
        const underlay = (r, rad) => V.box(g, r[0], r[1], r[2], r[3], { r: rad, fill: C.bg });
        // ---- 1) o caminho do token da vez (desenhado antes: passa por trás dos blocos)
        if (kind === 'lora') {
          const [ax, ay, aw, ah] = slab(A0[0], A0[1], sl), [bx2, by2, bw2, bh2] = slab(B0[0], B0[1], sl);
          const [tx, ty, tw3, th3] = tok[ti], [ox, oy, ow] = out[ti];
          const p1 = V.clamp(tf / 0.3, 0, 1), p3 = V.clamp((tf - 0.6) / 0.3, 0, 1);
          const sx = tx + tw3 / 2, sy = ty + th3;
          const ex = narrow ? ax + Math.min(aw, 30) / 2 : ax, ey = narrow ? ay : ay + ah / 2;
          V.arrow(g, sx, sy, V.lerp(sx, ex, p1), V.lerp(sy, ey, p1), { color: AC[ad], lw: 2, head: 7, dash: [5, 4] });
          if (tf > 0.3) V.arrow(g, ax + aw + 2, ay + ah / 2, vR[0] - 4, vR[1] + vR[3] / 2, { color: AC[ad], lw: 1.8, head: 7 });
          if (tf > 0.6) {
            const vx = narrow ? vR[0] + 20 : vR[0] + vR[2] + 2, vy = narrow ? vR[1] + vR[3] + 2 : vR[1] + vR[3] / 2;
            const bx3 = narrow ? bx2 + Math.min(bw2, 30) / 2 : bx2 - 2, by3 = narrow ? by2 : by2 + bh2 / 2;
            const q1 = Math.min(1, p3 * 2);
            V.arrow(g, vx, vy, V.lerp(vx, bx3, q1), V.lerp(vy, by3, q1), { color: AC[ad], lw: 1.8, head: 7 });
            if (p3 > 0.5) {
              const q2 = (p3 - 0.5) * 2;
              const fx = narrow ? bx2 + Math.min(bw2, 60) / 2 : bx2 + 20, fy = narrow ? by2 + bh2 : by2;
              const gx2 = narrow ? ox + ow / 2 : joinX, gy2 = narrow ? oy : rowY(ti) + 6.5;
              V.arrow(g, fx, fy, V.lerp(fx, gx2, q2), V.lerp(fy, gy2, q2), { color: AC[ad], lw: 2, head: 7, dash: [5, 4] });
            }
          }
        }
        // ---- 2) título e GEMM do base: um retângulo só para o batch inteiro
        if (!narrow) V.text(g, fit(g, 'uma camada (q_proj: 5120 → 12.288) num passo com 6 tokens de 4 requisições · r_max = ' + rmax, w - 32, 11.5, 'sans', 700), 16, 18, { size: 11.5, weight: 700 });
        else V.wrap(g, 'uma camada (q_proj: 5120 → 12.288), 6 tokens de 4 requisições · r_max = ' + rmax, 16, 14, w - 32, { size: 11, weight: 700, lineH: 13, maxLines: 2 });
        underlay(Wr, 10);
        V.box(g, Wr[0], Wr[1], Wr[2], Wr[3], { r: 10, fill: V.alpha(C.a, 0.12), stroke: V.alpha(C.a, 0.7), lw: 1.4 });
        if (narrow) V.text(g, fit(g, 'W [12.288 × 5120] · uma GEMM para os 6 tokens', Wr[2] - 16, 11, 'sans', 700), Wr[0] + Wr[2] / 2, Wr[1] + Wr[3] / 2 + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.a });
        else {
          V.text(g, 'W', Wr[0] + Wr[2] / 2, Wr[1] + Wr[3] / 2 - 14, { size: 28, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.a });
          V.text(g, '[12.288 × 5120]', Wr[0] + Wr[2] / 2, Wr[1] + Wr[3] / 2 + 14, { size: 11, font: 'mono', align: 'center', color: C.ink });
          V.wrap(g, 'GEMM do base: uma só para o batch inteiro', Wr[0] + 12, Wr[1] + Wr[3] - 30, Wr[2] - 24, { size: 10.5, color: C.muted, lineH: 13, maxLines: 2 });
        }
        // ---- 3) tokens e saídas
        KT.forEach((a, i) => {
          const [x, y, tw2, th] = tok[i];
          const s2 = a < 0 ? -1 : slots.indexOf(a);
          const col = a < 0 ? C.muted : AC[a];
          const waiting = a >= 0 && s2 < 0;
          const on = i === ti;
          if (narrow) V.line(g, x + tw2 / 2, y + th, x + tw2 / 2, Wr[1], { color: V.alpha(C.a, waiting ? 0.15 : 0.4), lw: 1 });
          else V.line(g, x + tw2, y + th / 2, Wr[0], Wr[1] + 8 + i * ((Wr[3] - 16) / 5), { color: V.alpha(C.a, waiting ? 0.15 : 0.35), lw: 1 });
          underlay(tok[i], 6);
          g.globalAlpha = waiting ? 0.45 : 1;
          V.box(g, x, y, tw2, th, { r: 6, fill: V.alpha(col, on ? 0.32 : 0.14), stroke: on ? C.ink : V.alpha(col, 0.7), lw: on ? 2 : 1.2 });
          const sig = a < 0 ? 'σ=−1' : waiting ? 'sem slot' : 'σ=' + s2;
          if (narrow) {
            V.text(g, 'x' + SUB[i], x + tw2 / 2, y + 11, { size: 11, font: 'mono', weight: 700, align: 'center' });
            V.text(g, fit(g, sig, tw2 - 4, 9.5, 'mono'), x + tw2 / 2, y + 24, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
          } else {
            V.dot(g, x + 11, y + th / 2, 5, col);
            V.text(g, 'x' + SUB[i], x + 22, y + th / 2 + 0.5, { size: 12, font: 'mono', weight: 700, base: 'middle' });
            V.text(g, sig, x + tw2 - 6, y + th / 2 + 0.5, { size: 10.5, font: 'mono', base: 'middle', align: 'right', color: C.muted });
          }
          g.globalAlpha = 1;
          const [ox, oy, ow, oh] = out[i];
          const lit = on && kind !== 'wait' && kind !== 'refused' && tf > 0.85;
          if (!narrow) V.line(g, Wr[0] + Wr[2], Wr[1] + 8 + i * ((Wr[3] - 16) / 5), ox, oy + oh / 2, { color: V.alpha(C.a, waiting ? 0.12 : 0.3), lw: 1 });
          underlay(out[i], 6);
          g.globalAlpha = waiting ? 0.4 : 1;
          V.box(g, ox, oy, ow, oh, { r: 6, fill: lit ? V.alpha(col, 0.35) : V.alpha(C.muted, 0.07), stroke: lit ? C.ink : V.alpha(C.muted, 0.4), lw: lit ? 2 : 1 });
          const ylab = narrow ? 'y' + SUB[i] : 'y' + SUB[i] + ' = Wx' + SUB[i] + (a < 0 ? '' : waiting ? ' (espera)' : ' + s·B·A·x' + SUB[i]);
          V.text(g, fit(g, ylab, ow - 8, narrow ? 11 : 10.5, 'mono', 600), ox + (narrow ? ow / 2 : 6), oy + oh / 2 + 0.5, { size: narrow ? 11 : 10.5, font: 'mono', weight: 600, base: 'middle', align: narrow ? 'center' : 'left' });
          g.globalAlpha = 1;
          if (!narrow && a >= 0 && !waiting) { V.dot(g, joinX, rowY(i), 6.5, C.bg); g.beginPath(); g.arc(joinX, rowY(i), 6.5, 0, Math.PI * 2); g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1.2; g.stroke(); V.text(g, '+', joinX, rowY(i) + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: col }); }
        });
        // ---- 4) pilhas de slots: Aᵀ (shrink) e Bᵀ (expand)
        const hi = kind === 'lora' ? sl : -9;
        stack(g, A0[0], A0[1], bw, sh, gap, slots, AC, C, rmax, real, hi, 'Aᵀ empilhado · shrink', narrow ? '[max_loras × 5120 × r]' : '[max_loras × 5120 × r_max]', 'left');
        stack(g, B0[0], B0[1], bw, sh, gap, slots, AC, C, rmax, real, hi, 'Bᵀ empilhado · expand', '[max_loras × r_max × 12.288]', 'right');
        // ---- 5) v_t = A_σ x_t: r_max saídas (as além do posto real são zero)
        const nseg = Math.max(2, rmax / 4), segW = vR[2] / nseg;
        const vLab = 'v' + SUB[ti] + ' [r_max = ' + rmax + ']';
        if (narrow) V.text(g, vLab, vR[0], vR[1] + vR[3] + 13, { size: 10.5, font: 'mono', weight: 700 });
        else V.text(g, vLab, vR[0], vR[1] - 7, { size: 10.5, font: 'mono', weight: 700 });
        const fillF = kind === 'lora' ? V.clamp((tf - 0.3) / 0.3, 0, 1) : 0;
        for (let k = 0; k < nseg; k++) {
          const isReal = k * 4 < Math.min(real, rmax);
          const lit = k < fillF * nseg;
          V.box(g, vR[0] + k * segW, vR[1], Math.max(1, segW - 1.5), vR[3], { r: 2, fill: lit ? (isReal ? AC[ad] : V.alpha(C.muted, 0.35)) : V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.35), lw: 0.8 });
        }
        const noteY = vR[1] + vR[3] + (narrow ? 27 : 14), noteW = narrow ? vR[2] : vR[2] + 30;
        if (kind === 'base') V.wrap(g, '✕ σ = −1: os kernels pulam', vR[0], noteY, noteW, { size: 10.5, color: C.muted, weight: 600, lineH: 13, maxLines: 2 });
        if (real < rmax && kind === 'lora' && tf > 0.6) V.wrap(g, (rmax - real) + ' saídas sempre zero', vR[0], noteY, noteW, { size: 10, color: C.muted, lineH: 13, maxLines: 2 });
        // ---- 6) narração
        const nm = ad >= 0 ? ADS[ad] : 'base';
        let msg;
        if (kind === 'base') msg = 'x' + SUB[ti] + ' · σ = −1 (requisição ao base): só W·x; shrink e expand pulam este token.';
        else if (kind === 'wait') msg = 'x' + SUB[ti] + ' · ' + nm + ' sem slot (--max-loras = ' + S.maxL + '): não há fatia de onde ler A e B, então a requisição espera fora deste batch.';
        else if (kind === 'refused') msg = 'adapter r = ' + real + ' > --max-lora-rank ' + rmax + ': não cabe na fatia, e o vLLM recusa carregá-lo.';
        else msg = 'x' + SUB[ti] + ' · σ = ' + sl + ' (' + nm + '): o shrink lê a fatia ' + sl + ' de Aᵀ e gera v com ' + rmax + ' saídas' + (real < rmax ? ', ' + (rmax - real) + ' delas sempre zero (hachura)' : '') + '; o expand lê a fatia ' + sl + ' de Bᵀ e soma s·B·v a W·x.';
        V.wrap(g, msg, 16, textY, w - 32, { size: narrow ? 10.5 : 11.5, lineH: narrow ? 14 : 15, maxLines: narrow ? 4 : 2, weight: 600, color: kind === 'refused' ? RED() : kind === 'wait' ? C.d : C.ink });
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const AC = [C.b, C.c, C.d, C.f, C.e];
        S.acc += dt;
        if (S.acc > 0.6) { S.acc = 0; step(); }
        if (S.mode === 'K') { drawK(g, w, h, t, C, narrow, AC); return; }
        const ease = Math.min(1, dt * 6);
        // ---- topo: slots ou modelo fundido
        if (S.mode === 'A') {
          V.text(g, fit(g, (narrow ? 'slots: --max-loras = ' + S.maxL + ' · posto ' : 'slots de LoRA na GPU: --max-loras = ' + S.maxL + ', buffers no posto ') + RANKS[S.ri], w - 32, 11.5, 'sans', 700), 16, 18, { size: 11.5, weight: 700 });
          const active = Array.from(new Set(S.bus.filter((r) => r.ad >= 0).map((r) => r.ad)));
          const sw = Math.min(110, (w - 32 - (S.maxL - 1) * 6) / S.maxL);
          for (let s2 = 0; s2 < S.maxL; s2++) {
            const x = 16 + s2 * (sw + 6), ad = active[s2];
            V.box(g, x, 28, sw, 28, { r: 7, fill: ad != null ? V.alpha(AC[ad], 0.3) : V.alpha(C.muted, 0.07), stroke: ad != null ? AC[ad] : V.alpha(C.muted, 0.4), lw: 1.4 });
            if (sw > 34) V.text(g, fit(g, ad != null ? ADS[ad] : 'livre', sw - 8, 10.5, 'mono', 600), x + sw / 2, 42.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: ad != null ? C.ink : C.muted, weight: 600 });
          }
        } else {
          const gb = S.fmtv === 'bf' ? 55.6 : 24.69;
          V.box(g, 16, 22, w - 32, 36, { r: 9, fill: V.alpha(C.a, 0.14), stroke: C.a });
          V.text(g, fit(g, (S.fmtv === 'bf' ? 'q38-final-bf16 · ' : 'q38-final-nvfp4 · ') + fmt(gb, gb > 50 ? 1 : 2) + ' GB lidos por passo de decode · sem slots', w - 52, 12, 'sans', 600), 26, 40.5, { size: 12, weight: 600, base: 'middle' });
        }
        // ---- geometria
        let Q, B, O, perRow, cache;
        if (!narrow) {
          const midY = 80, midH = h - midY - 92;
          const qW = Math.min(200, w * 0.24), outW = Math.min(130, w * 0.15);
          Q = [16, midY, qW, midH]; B = [16 + qW + 16, midY, w - 32 - qW - outW - 32, midH]; O = [w - 16 - outW, midY, outW, midH];
          perRow = Math.max(2, Math.floor((qW - 10) / 26)); cache = h - 82;
        } else {
          Q = [16, 70, w - 32, 54]; B = [16, 142, w - 32, 140]; O = null;
          perRow = Math.max(4, Math.floor((w - 42) / 26)); cache = 300;
        }
        // fila
        V.text(g, 'fila', Q[0], Q[1] + 10, { size: 11, color: C.muted, weight: 600 });
        if (S.mode === 'A' && S.waiting > 0) V.text(g, S.waiting + ' esperando slot', Q[0] + Q[2], Q[1] + 10, { size: 11, weight: 700, color: C.d, align: 'right' });
        V.box(g, Q[0], Q[1] + 16, Q[2], Q[3] - 16, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.3) });
        S.q.forEach((r, i) => {
          const tx = Q[0] + 8 + (i % perRow) * 26 + 11, ty = Q[1] + 28 + Math.floor(i / perRow) * 26 + 6;
          if (r.x == null) { r.x = tx; r.y = ty; }
          r.x += (tx - r.x) * ease; r.y += (ty - r.y) * ease;
          if (r.y > Q[1] + Q[3] - 6) return;
          const col = S.mode === 'C' ? C.a : r.ad < 0 ? C.muted : AC[r.ad];
          V.dot(g, r.x, r.y, 9, S.mode !== 'C' && r.ad < 0 ? V.alpha(C.muted, 0.2) : V.alpha(col, 0.85));
          if (S.mode !== 'C' && r.ad < 0) { g.beginPath(); g.arc(r.x, r.y, 9, 0, Math.PI * 2); g.strokeStyle = C.muted; g.lineWidth = 1.5; g.stroke(); }
          if (S.mode === 'A' && r.ad >= 0 && S.waiting > 0) { g.beginPath(); g.arc(r.x, r.y, 12, -Math.PI / 2, -Math.PI / 2 + ((t * 2) % (Math.PI * 2))); g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.2; g.stroke(); }
        });
        // ônibus
        V.text(g, fit(g, 'batch em execução (até ' + SEATS + ' sequências, ilustrativo)', B[2], 11, 'sans', 600), B[0], B[1] + 10, { size: 11, color: C.muted, weight: 600 });
        const bY = B[1] + 18, bH = B[3] - 26;
        V.box(g, B[0], bY, B[2], bH, { r: 14, fill: V.alpha(C.a, 0.07), stroke: V.alpha(C.a, 0.6), lw: 1.6 });
        const wob = Math.sin(t * 10) * 0.8;
        [0.18, 0.82].forEach((f) => { V.dot(g, B[0] + B[2] * f, bY + bH + 1 + wob, 7, C.ink); V.dot(g, B[0] + B[2] * f, bY + bH + 1 + wob, 3, C.bg); });
        const sCols = 4, sw = (B[2] - 24) / sCols, sh = Math.min(64, (bH - 20) / 2);
        const sy0 = bY + (bH - 2 * sh) / 2;
        for (let s2 = 0; s2 < SEATS; s2++) {
          const sx = B[0] + 12 + (s2 % sCols) * sw, sy = sy0 + Math.floor(s2 / sCols) * sh;
          V.box(g, sx + 3, sy + 2, sw - 6, sh - 4, { r: 7, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.25) });
          const r = S.bus[s2];
          if (r) {
            const isC = S.mode === 'C';
            const col = isC ? C.a : r.ad < 0 ? C.muted : AC[r.ad];
            const rad = Math.min(11, sh / 5);
            V.dot(g, sx + sw / 2, sy + 4 + sh * 0.32, rad, !isC && r.ad < 0 ? V.alpha(C.muted, 0.3) : col);
            if (sw > 44 && sh > 40) V.text(g, fit(g, isC ? 'tutor' : r.ad < 0 ? 'base' : ADS[r.ad].replace('dpo-', ''), sw - 10, 10, 'mono'), sx + sw / 2, sy + sh - 7, { size: 10, font: 'mono', align: 'center', color: C.muted });
            for (let k = 0; k < 2; k++) { const f = (t * 1.2 + s2 * 0.13 + k * 0.5) % 1; g.globalAlpha = 1 - f; V.dot(g, sx + sw / 2 + f * sw * 0.42, sy + 9, 1.8, col); g.globalAlpha = 1; }
          }
        }
        // saída / velocímetro
        if (S.mode === 'C') {
          const gb = S.fmtv === 'bf' ? 55.6 : 24.69, c = (230 / gb) * (S.mtp ? 1.75 : 1);
          const cx = O ? O[0] + O[2] / 2 : B[0] + B[2] - 50, cy = O ? O[1] + O[3] * 0.5 : bY - 2, rr = O ? Math.min(O[2] / 2 - 4, 46) : 0;
          if (O) {
            g.beginPath(); g.arc(cx, cy, rr, Math.PI, 2 * Math.PI); g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 7; g.stroke();
            g.beginPath(); g.arc(cx, cy, rr, Math.PI, Math.PI + V.clamp(c / 20, 0, 1) * Math.PI); g.strokeStyle = S.fmtv === 'bf' ? C.d : C.a; g.lineWidth = 7; g.stroke();
            V.text(g, fmt(c, 1), cx, cy - 4, { size: 16, font: 'mono', weight: 700, align: 'center' });
            V.text(g, 'tok/s · 1 stream', cx, cy + 14, { size: 10, color: C.muted, align: 'center' });
          } else {
            V.text(g, '≈ ' + fmt(c, 1) + ' tok/s por stream', B[0] + B[2], B[1] + 10, { size: 11, font: 'mono', weight: 700, align: 'right', color: S.fmtv === 'bf' ? C.d : C.a });
          }
        } else if (O) {
          V.text(g, 'tokens saindo', O[0], O[1] + 10, { size: 11, color: C.muted, weight: 600 });
          S.bus.forEach((r, k) => { for (let j = 0; j < 3; j++) { const f = (t * 0.6 + j / 3 + k * 0.11) % 1; g.globalAlpha = 1 - f * 0.8; V.box(g, O[0] + f * (O[2] - 10), O[1] + 30 + k * 18, 9, 9, { r: 2, fill: r.ad < 0 ? C.muted : AC[r.ad] }); g.globalAlpha = 1; } });
        }
        // prefixos em cache
        V.wrap(g, S.mode === 'A' ? 'blocos de prefixo em cache: o hash inclui o ID do LoRA, então cada adapter tem os seus' : 'prefix cache: um modelo só, os blocos de prefixo são compartilhados por todas as requisições', 16, cache + 12, w - 32, { size: 11, color: C.muted, weight: 600, lineH: 14, maxLines: 2 });
        const lines = narrow ? 2 : 1;
        const groupsA = S.mode === 'A' ? [-1].concat(Object.keys(S.seen).map(Number).filter((a) => a < S.nA).sort()) : [-2];
        const gwid = Math.min(120, (w - 32) / Math.max(1, groupsA.length) - 8);
        groupsA.forEach((ad, i) => {
          const gx = 16 + i * (gwid + 8), gy = cache + 18 + lines * 14;
          const col = ad === -2 ? C.a : ad < 0 ? C.muted : AC[ad];
          for (let b = 0; b < 4; b++) V.box(g, gx + b * (gwid / 4), gy, gwid / 4 - 3, 16, { r: 3, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
          V.text(g, fit(g, ad === -2 ? 'tutor-q38' : ad < 0 ? 'base' : ADS[ad], gwid, 10, 'mono'), gx, gy + 29, { size: 10, font: 'mono', color: C.muted });
        });
      });
    })();
  },
});

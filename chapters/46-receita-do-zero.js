/* 46 · Tutorial: sua receita para um modelo recém-lançado (MiMo-V2.6-Flash-MOPD) */
LA.chapter({
  id: 'receita-do-zero',
  part: 9,
  order: 46,
  title: 'Tutorial: sua receita para um modelo recém-lançado',
  navTitle: 'Receita do zero',
  subtitle: 'Do card no Hugging Face ao ./start.sh reproduzível: os onze passos para servir o MiMo-V2.6-Flash-MOPD em dois DGX Spark com vLLM, com cada número marcado como medido, estimado ou a medir.',
  minutes: 75,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['tutorial', 'receita', 'MiMo-V2.6', 'triagem', 'memória', 'overlay', 'paridade', 'especulação', 'benchmark', 'A/B/A', 'MLX'],
  goals: [
    'Seguir um método de <b>11 passos</b>, cada um com um critério de saída verificável',
    'Fazer a <b>triagem</b> de um checkpoint sem baixá-lo e a <b>matemática de memória</b> de um modelo híbrido',
    'Escolher checkpoint e topologia, e escrever <b>overlays fail-closed</b> provados por paridade truncada',
    'Justificar cada flag, escolher o <b>k</b> da especulação e decidir o tuning com <b>A/B/A</b>',
    'Empacotar a receita para que <b>outra pessoa</b> reproduza os números',
  ],
  prereqs: ['benchmarking'],
  summary: [
    'Uma receita nova segue 11 passos, e cada um termina num <b>artefato verificável</b>: triagem, previsto × medido, KLD, divisibilidade, paridade, log de boot, tabela de aceitação, smoke verde, benchmark com método, tuning com regra e README reproduzível.',
    'A triagem lê o <code>config.json</code> e os <b>cabeçalhos safetensors por range request</b>: no MiMo ela revela experts MXFP4 sob <code>quant_method: fp8</code>, QKV pré-fatiado para TP=4, DiffKV 192/128 e sinks só nas SWA.',
    'Memória: GMU × 121,69 GiB = pesos + overhead + pool. No MiMo só as 9 camadas globais crescem com o contexto (5.760 B/token/rank em FP8); as 39 SWA custam ~12,8 MB fixos por requisição e o KV do drafter DFlash, outros ~11–12 MB; o alocador híbrido desperdiça ~11%.',
    'O checkpoint oficial MXFP4 é o padrão por ser o menor de ~4 bits e a própria referência; qualquer outro exige <b>KLD</b> contra ele no mesmo motor. TP=2 divide tudo, mas o QKV pré-fatiado precisa ser remontado sem requantizar.',
    'Bugs silenciosos se combatem com <b>overlays fail-closed</b> (âncoras exatas, idempotência, “already upstream”, marcador no log, mesmos bytes nos dois ranks) e com <b>paridade truncada</b> (N camadas, TP1 = TP2 ≈ HF).',
    'O k ótimo depende da carga (código e JSON pedem k alto, prosa pede k baixo), as capture sizes precisam ser múltiplos de k+1, e o teto de banda (~43 tok/s sem spec) diz onde você deveria chegar.',
    'Tuning só com regra escrita antes (A/B/A, ≥ 3% nos dois lados, canário de qualidade), e “pronto” significa que outro par de Sparks reproduz o README com ±5–10%.',
  ],
  quiz: [
    { q: 'Por que a triagem lê os cabeçalhos safetensors com range requests, em vez de confiar só no <code>config.json</code>?', o: ['Porque o config.json não informa o número de camadas', 'Porque os cabeçalhos trazem dtypes e shapes reais, que podem contradizer o config, sem baixar os ~170 GB', 'Porque range requests são a única forma de baixar do Hugging Face', 'Porque o vLLM exige os cabeçalhos para registrar a arquitetura'], a: 1, e: 'O cabeçalho (8 bytes de tamanho + JSON) diz o dtype e o shape de cada tensor. No MiMo ele revela experts em MXFP4 (U8 com dois nibbles por byte) apesar de <code>quant_method: fp8</code>, e o QKV pré-fatiado para TP=4. O config não tem essa informação, e as demais opções são falsas.' },
    { q: 'No MiMo em TP=2 com KV FP8, quantos bytes de KV <i>por token de contexto</i> cada rank gasta nas camadas que crescem com o contexto (sem o enchimento do alocador)?', o: ['5.760 B', '49.920 B', '11.520 B', '99.840 B'], a: 0, e: 'Só as 9 camadas globais crescem: 9 × (4/2 KV heads) × (192 + 128) × 1 byte = 5.760 B. Os 49.920 B são das 39 SWA, mas só para a janela (custo fixo por requisição); 11.520 B é a mesma conta em BF16.' },
    { q: 'Por que a receita usa <code>--cudagraph-capture-sizes 8 16 24 32 40 48 56 64</code>?', o: ['Porque o GB10 só captura múltiplos de 8', 'Porque com DFlash k=7 cada sequência em decode tem 8 tokens (1 + 7), e até 8 sequências dão 64', 'Porque o MNBT é 4096', 'Porque são os tamanhos de bloco do KV'], a: 1, e: 'Decode uniforme com spec tem query_len = k+1. Com k=7, n sequências ocupam 8n tokens; a lista cobre n = 1…8 (<code>--max-num-seqs 8</code>). Mudar k sem mudar a lista faz os batches serem “padded” ou rodarem eager.' },
    { q: 'Você ativou <code>--kv-cache-dtype fp8</code>, o pool dobrou, mas o needle falha sempre que a agulha está a mais de ~128 tokens da pergunta. Causa mais provável?', o: ['O FP8 perdeu precisão demais', 'O <code>sliding_window: 128</code> do nível do modelo vazou para as camadas globais pelo <code>cache_config</code> compartilhado', 'O prefix cache devolveu blocos errados', 'O drafter DFlash tem janela de 1024'], a: 1, e: 'É o sintoma clássico do vazamento: as globais passam a enxergar só a janela. Por isso o overlay encaminha uma <b>cópia</b> do <code>cache_config</code> com <code>sliding_window=None</code>. O needle é o teste que pega isso, porque em prompts curtos os dois tipos de camada veem o mesmo contexto.' },
    { q: 'No A/B/A, B ficou 4% acima de A₁, mas A₂ (medido depois de B) ficou 1% acima de B. Pela regra do Passo 9, o que fazer?', o: ['Aceitar: +4% passa do limiar de 3%', 'Rejeitar e registrar: B não supera A₂ por ≥ 3%, então o ganho aparente é deriva', 'Aceitar se o canário GSM8K passar', 'Repetir só o braço B'], a: 1, e: 'A regra exige ≥ 3% nos dois lados. O A₂ mostra que o sistema melhorou sozinho com o tempo (page cache, autotune, temperatura): a diferença B − A₁ é deriva, não efeito. O resultado negativo vai para o CHANGELOG.' },
    { q: 'Por que a topologia escolhida para o MiMo em 2 Sparks é TP=2, e não PP=2?', o: ['Porque PP não funciona com MoE', 'Porque com batch pequeno o PP deixa um nó ocioso enquanto o outro trabalha, dobrando a latência de decode, e o TP=2 divide heads, KV heads e experts', 'Porque TP=2 não precisa de all-reduce', 'Porque o checkpoint foi salvo para PP'], a: 1, e: 'O PP só passa ativações entre estágios, mas em decode com batch 1 cada token atravessa os dois nós em série. O TP=2 custa 96 all-reduces por passo (~10% do passo em RoCE) e tudo divide por 2. O PP serve para throughput com muitos streams.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const clamp = V.clamp, lerp = V.lerp;
    const setH = (fig, mob) => { if (fig.clientWidth < 600) fig.querySelector('.la-fig-stage').style.height = mob + 'px'; };
    const nf = (x, d = 0) => Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    /** texto com quebra de linha; devolve o y da próxima linha */
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(/\s+/);
      let line = '', yy = y;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (line && g.measureText(test).width > maxW) { V.text(g, line, x, yy, o); line = wd; yy += lh; } else line = test;
      }
      if (line) { V.text(g, line, x, yy, o); yy += lh; }
      return yy;
    }
    const onInk = (C) => (C.dark ? '#00210F' : '#ffffff');

    /* =====================================================================
       Figura 1 — o metrô da receita (stepper sincronizado)
       ===================================================================== */
    (function figMetro() {
      const fig = root.querySelector('#fig-metro');
      if (fig.clientWidth < 520) fig.querySelector('.la-fig-stage').style.height = '350px';
      const ST = [
        { n: 0, s: 'Triagem', z: 0, obj: 'Descobrir o que o modelo tem de diferente e onde o motor vai quebrar, antes de baixar 170 GB.', exit: 'Tabela de triagem e lista de riscos, cada risco ligado a um PR ou issue.', ch: [['checkpoints', 'Checkpoints'], ['tres-modelos', 'Os três modelos']] },
        { n: 1, s: 'Memória', z: 0, obj: 'Saber quanto sobra para KV e, com isso, quanto contexto e quantas requisições cabem.', exit: 'Tabela previsto × medido (pesos, pool, tokens) com desvio de até ±10%.', ch: [['memoria-unificada', 'Memória unificada'], ['paged-attention-kv', 'PagedAttention &amp; KV']] },
        { n: 2, s: 'Pesos', z: 0, obj: 'Escolher o checkpoint com a melhor relação qualidade × memória × velocidade para este hardware.', exit: 'Checkpoint baixado, verificado e com revisão fixada; KLD contra o oficial se não for o oficial.', ch: [['quantizacao-fundamentos', 'Quantização'], ['medindo-qualidade', 'Medindo qualidade']] },
        { n: 3, s: 'Topologia', z: 0, obj: 'Decidir como dividir o modelo entre os nós (TP, EP, PP).', exit: 'Tabela de divisibilidade sem ✘, topologia escolhida e alternativas anotadas.', ch: [['paralelismos', 'Paralelismos'], ['nccl-rdma', 'NCCL &amp; RDMA']] },
        { n: 4, s: 'Motor', z: 1, obj: 'Fazer o vLLM carregar o modelo corretamente, e provar isso numericamente.', exit: 'Overlays com testes de CPU; paridade truncada TP1 = TP2 ≈ HF; marcadores no log.', ch: [['vllm-arquitetura', 'Arquitetura do vLLM'], ['engenharia-de-receita', 'Engenharia de receita']] },
        { n: 5, s: 'Lançamento', z: 1, obj: 'Comandos de head e worker em que cada flag tem motivo e cada valor vem do .env.', exit: 'Boot nos dois nós; backend, marcadores, KV e capture sizes esperados no log; comando renderizado salvo.', ch: [['scheduler-batching', 'Scheduler'], ['cuda-graphs-warmup', 'CUDA graphs']] },
        { n: 6, s: 'Especulação', z: 1, obj: 'Escolher o speculator e o k que maximizam os tok/s na sua carga.', exit: 'Tabela aceitação × k × categoria × contexto; SPEC_METHOD/SPEC_K com rollback.', ch: [['especulacao-fundamentos', 'Especulação'], ['dflash-dspark', 'DFlash &amp; DSpark']] },
        { n: 7, s: 'Validação', z: 1, obj: 'Provar que o servidor responde certo, não só rápido.', exit: 'tests/smoke.sh verde; resultados no README com data e commit.', ch: [['api-tools-estruturado', 'API, tools &amp; JSON'], ['medindo-qualidade', 'Medindo qualidade']] },
        { n: 8, s: 'Benchmark', z: 2, obj: 'TTFT, TPOT e tok/s reproduzíveis na escada de concorrência e no prefill.', exit: 'Tabelas de decode, prefill frio e aceitação, com método, data, commit e digest.', ch: [['benchmarking', 'Benchmarking'], ['gpu-por-dentro', 'GPU por dentro']] },
        { n: 9, s: 'Tuning', z: 2, obj: 'Melhorar um alvo declarado sem quebrar o resto, com a regra escrita antes.', exit: 'docs/tuning-AAAA-MM-DD.md com aceitos e rejeitados; .env.sample e CHANGELOG atualizados.', ch: [['benchmarking', 'Benchmarking']] },
        { n: 10, s: 'Pacote', z: 2, obj: 'Outra pessoa reproduz tudo com cp .env.sample .env &amp;&amp; ./start.sh.', exit: 'Um segundo par de Sparks reproduz o README com diferença de até 5–10%.', ch: [['engenharia-de-receita', 'Engenharia de receita']] },
      ];
      const link = (c) => `<a href="#/c/${c[0]}">${c[1]}</a>`;
      const st = UI.stepper(fig, ST.map((s) => ({
        t: `Passo ${s.n} · ${s.s}`,
        d: `<p style="margin:.2em 0"><b>Objetivo.</b> ${s.obj}</p><p style="margin:.2em 0"><b>Critério de saída.</b> ${s.exit}</p><p style="margin:.2em 0">Conceitos: ${s.ch.map(link).join(' · ')}</p>`,
      })), { autoplay: 6500 });
      const S = { p: 0 };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const zc = [C.b, C.a, C.c];
        const cols = w >= 760 ? 11 : w >= 520 ? 6 : 4;
        const rows = Math.ceil(ST.length / cols);
        const mx = cols === 11 ? 46 : 50;
        const dx = (w - 2 * mx) / (cols - 1);
        let P;
        if (rows === 1) {
          const y0 = Math.min(150, h * 0.44);
          P = ST.map((s, i) => [mx + i * dx, y0]);
        } else {
          const top = 64, rowH = (h - top - 10) / rows;
          P = ST.map((s, i) => { const r = Math.floor(i / cols), c = i % cols, cc = r % 2 ? cols - 1 - c : c; return [mx + cc * dx, top + r * rowH + 18]; });
        }
        // legenda das fases
        const leg = w < 520 ? [['planejar', zc[0]], ['subir certo', zc[1]], ['medir/entregar', zc[2]]] : [['planejar no papel (0–3)', zc[0]], ['fazer subir certo (4–7)', zc[1]], ['medir e entregar (8–10)', zc[2]]];
        g.font = V.font(11.5, 'sans', 500);
        let lx = 16;
        leg.forEach(([l, c]) => { V.box(g, lx, 14, 12, 12, { r: 3, fill: c }); V.text(g, l, lx + 18, 24.5, { size: 11.5, color: C.muted, weight: 500 }); lx += g.measureText(l).width + 40; });
        // trilhos
        const segPath = (a, b) => {
          g.beginPath(); g.moveTo(a[0], a[1]);
          if (Math.abs(a[1] - b[1]) < 1) g.lineTo(b[0], b[1]);
          else { const side = a[0] > w / 2 ? 1 : -1, ox = a[0] + side * 30; g.bezierCurveTo(ox, a[1], ox, b[1], b[0], b[1]); }
        };
        g.lineCap = 'round';
        for (let i = 0; i < ST.length - 1; i++) { segPath(P[i], P[i + 1]); g.strokeStyle = V.alpha(zc[ST[i + 1].z], 0.8); g.lineWidth = 6; g.stroke(); }
        // pontilhado correndo por cima (fluxo)
        for (let i = 0; i < ST.length - 1; i++) { segPath(P[i], P[i + 1]); g.setLineDash([2, 10]); g.lineDashOffset = -t * 18; g.strokeStyle = V.alpha(C.bg || '#000', 0.55); g.lineWidth = 2; g.stroke(); g.setLineDash([]); }
        g.lineCap = 'butt';
        // caminhos de volta
        [[7, 4, 'falhou? volta ao motor', 1], [9, 5, 'ajuste → relança', -1]].forEach(([a, b, lab, up]) => {
          const A = P[a], B = P[b];
          if (Math.abs(A[1] - B[1]) > 1) {
            V.text(g, `↺ ${b}`, A[0] + 16, A[1] - 16, { size: 11, font: 'mono', color: C.muted });
            return;
          }
          const off = up > 0 ? -18 : 40, cv = up > 0 ? 34 : -26;
          V.arrow(g, A[0] - 6, A[1] + off, B[0] + 6, B[1] + off, { curve: cv, color: V.alpha(C.muted, 0.85), dash: [4, 4], lw: 1.4, head: 7 });
          V.text(g, lab, (A[0] + B[0]) / 2, A[1] + off - cv / 2 + (up > 0 ? -6 : 14), { size: 11, align: 'center', color: C.muted });
        });
        // estações
        P.forEach((p, i) => {
          const s = ST[i], col = zc[s.z], past = i < st.i;
          V.dot(g, p[0], p[1], 11, C.panel || C.bg);
          g.beginPath(); g.arc(p[0], p[1], 11, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 3; g.stroke();
          if (past) V.dot(g, p[0], p[1], 5, col);
          g.font = V.font(11.5, 'sans', i === st.i ? 600 : 400);
          const lw = g.measureText(s.s).width + 8, lxx = clamp(p[0], lw / 2 + 2, w - lw / 2 - 2);
          V.box(g, lxx - lw / 2, p[1] + 18, lw, 17, { r: 5, fill: V.alpha(C.bg || C.panel, 0.85) });
          V.text(g, s.s, lxx, p[1] + 30, { size: 11.5, align: 'center', color: i === st.i ? C.ink : C.muted, weight: i === st.i ? 600 : 400 });
        });
        // o trem-token
        S.p += (st.i - S.p) * Math.min(1, dt * 3.2);
        const i0 = clamp(Math.floor(S.p), 0, ST.length - 1), i1 = Math.min(ST.length - 1, i0 + 1), f = S.p - i0;
        const A = P[i0], B = P[i1];
        let tx, ty;
        if (Math.abs(A[1] - B[1]) < 1) { tx = lerp(A[0], B[0], f); ty = A[1]; } else { const side = A[0] > w / 2 ? 1 : -1, ox = A[0] + side * 30; [tx, ty] = V.bez(A, [ox, A[1]], [ox, B[1]], B, f); }
        const col = zc[ST[st.i].z];
        V.glow(g, tx, ty, 34, col, 0.8);
        V.box(g, tx - 19, ty - 13, 38, 26, { r: 12, fill: col });
        const blink = Math.sin(t * 2.3) > 0.96 ? 0.5 : 2.2;
        g.fillStyle = onInk(C);
        g.beginPath(); g.ellipse(tx - 7, ty - 4, 2.2, blink, 0, 0, Math.PI * 2); g.fill();
        g.beginPath(); g.ellipse(tx + 7, ty - 4, 2.2, blink, 0, 0, Math.PI * 2); g.fill();
        V.text(g, String(ST[st.i].n), tx, ty + 8.5, { size: 11, font: 'mono', weight: 700, align: 'center', color: onInk(C) });
        // placa da estação (só no layout de uma linha)
        if (rows === 1) {
          const s = ST[st.i];
          V.text(g, `Passo ${s.n} · ${s.s}`, w / 2, h - 54, { size: 18, font: 'display', align: 'center', color: zc[s.z], weight: 600 });
          V.text(g, 'saída: ' + s.exit.replace(/&amp;/g, '&'), w / 2, h - 28, { size: 12.5, align: 'center', color: C.ink, maxW: w - 60 });
        }
      });
    })();

    /* =====================================================================
       Figura 2 — raio-X do checkpoint
       ===================================================================== */
    (function figRaiox() {
      const fig = root.querySelector('#fig-raiox');
      setH(fig, 620);
      const T = [
        { label: 'QKV global (camada 0)', name: 'layers.0.self_attn.qkv_proj.weight', dtype: 'F8_E4M3', shape: [13568, 4096], kind: 'qkv', ga: true, bytes: 13568 * 4096, what: '4 chunks × (3072 Q + 192 K + 128 V) = 13.568 linhas. Cada chunk tem 16 heads de Q e 1 KV head: o checkpoint foi salvo fatiado para TP=4 (<code>tp_size: 4</code> no índice). Em TP=2 cada rank junta 2 chunks (Passo 3).' },
        { label: 'Escalas do QKV global', name: 'layers.0.self_attn.qkv_proj.weight_scale_inv', dtype: 'F32', shape: [108, 32], kind: 'scale', ga: true, bytes: 108 * 32 * 4, what: 'Um fator F32 por bloco 128×128: 4096/128 = 32 colunas; 3392 linhas por chunk = 26,5 blocos → 27 por chunk, 108 no total. O bloco 25 de cada chunk cobre K e V ao mesmo tempo; o 26 é meio vazio.' },
        { label: 'QKV SWA (camada 1)', name: 'layers.1.self_attn.qkv_proj.weight', dtype: 'F8_E4M3', shape: [14848, 4096], kind: 'qkv', ga: false, bytes: 14848 * 4096, what: '4 × (3072 Q + 384 K + 256 V) = 14.848. As SWA têm o dobro de KV heads (8): 2 por chunk, o que quebrava a suposição “1 chunk por KV head” no sharding (#57508).' },
        { label: 'Escalas do QKV SWA', name: 'layers.1.self_attn.qkv_proj.weight_scale_inv', dtype: 'F32', shape: [116, 32], kind: 'scale', ga: false, bytes: 116 * 32 * 4, what: '3712 linhas por chunk = 29 blocos exatos: aqui Q, K e V começam em fronteiras de bloco.' },
        { label: 'Sink bias (SWA)', name: 'layers.1.self_attn.attention_sink_bias', dtype: 'BF16', shape: [64], kind: 'sink', bytes: 128, what: '64 valores, um por head de Q: um logit extra no softmax que dá à atenção um lugar para “descarregar” massa. Só nas SWA; o backend precisa aceitar <code>sinks=</code> por camada.' },
        { label: 'Expert: gate_proj (MXFP4)', name: 'layers.1.mlp.experts.0.gate_proj.weight', dtype: 'U8', shape: [2048, 2048], kind: 'nibble', bytes: 2048 * 2048, what: 'U8 [2048, 2048] = 2048 linhas × 4096 pesos, dois nibbles E2M1 por byte. É MXFP4 armazenado, embora o <code>quant_method</code> diga fp8. A ordem dos nibbles (baixo = elemento par?) precisa ser confirmada contra o kernel.' },
        { label: 'Expert: escalas E8M0', name: 'layers.1.mlp.experts.0.gate_proj.weight_scale', dtype: 'U8', shape: [2048, 128], kind: 'e8m0', bytes: 2048 * 128, what: 'Um byte E8M0 (potência de 2) para cada 32 pesos: 4096/32 = 128 por linha. Custo médio: 4 + 8/32 = 4,25 bits por peso.' },
        { label: 'o_proj (BF16)', name: 'layers.0.self_attn.o_proj.weight', dtype: 'BF16', shape: [4096, 8192], kind: 'oproj', bytes: 4096 * 8192 * 2, what: 'Entrada 8192 = 64 heads × 128 (<code>v_head_dim</code>), saída 4096. Está em <code>ignored_layers</code>: 48 camadas + MTP ≈ 3,2 GB em BF16 lidos a cada passo, alvo do <code>VLLM_MIMO_OPROJ_FP8</code> (#59083).' },
        { label: 'QKV do MTP', name: 'mtp.layers.0.self_attn.qkv_proj.weight', dtype: 'F8_E4M3', shape: [14848, 4096], kind: 'qkv', ga: false, mtp: true, bytes: 14848 * 4096, what: 'Mesma forma do QKV das SWA: o MTP é uma camada do tipo SWA, com sink. O drafter DFlash, ao contrário, tem config próprio (<code>dflash/config.json</code>).' },
      ];
      const bstr = (b) => (b >= 1e6 ? nf(b / 1e6, 1) + ' MB' : b >= 1e3 ? nf(b / 1e3, 1) + ' KB' : b + ' B');
      const S = { i: 0 };
      const ctl = UI.controls(fig);
      const info = LA.h('dl', { class: 'la-kv', style: { margin: '0', padding: '12px 18px 6px', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(info, ctl);
      const upd = () => { const x = T[S.i]; info.innerHTML = `<dt>tensor</dt><dd><code>model.${x.name}</code></dd><dt>dtype · shape</dt><dd>${x.dtype} · [${x.shape.join(', ')}] · ${bstr(x.bytes)}</dd><dt>o que revela</dt><dd>${x.what}</dd>`; };
      UI.select(ctl, { label: 'Tensor', options: T.map((x, i) => ({ v: String(i), label: x.label })), value: '0', onChange: (v) => { S.i = +v; upd(); } });
      upd();
      // dados ilustrativos pré-computados
      const rr = V.rng(46);
      const bytes = Array.from({ length: 48 }, () => rr.int(256));
      const sinks = Array.from({ length: 4 }, () => Array.from({ length: 16 }, () => rr.normal() * 1.4));
      const exps = Array.from({ length: 128 }, () => 118 + rr.int(9));
      const E2 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
      const e2m1 = (n) => (n & 8 ? -1 : 1) * E2[n & 7];

      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const x = T[S.i];
        const narrow = w < 560;
        // ---------- faixa do arquivo ----------
        const fx = 16, fw = w - 32, fy = 34, fh = 24;
        V.text(g, 'um shard .safetensors (~1,3 GB em média: 172,9 GB / 129 arquivos)', fx, 16, { size: 11.5, color: C.muted, maxW: fw });
        const s1 = 40, s2 = Math.max(110, fw * 0.22);
        const pulse = 0.35 + 0.25 * Math.sin(t * 3);
        V.box(g, fx, fy, s1, fh, { r: 5, fill: V.alpha(C.d, pulse + 0.1), stroke: C.d, lw: 1 });
        V.text(g, '8 B', fx + s1 / 2, fy + fh / 2 + 4, { size: 11, align: 'center', font: 'mono' });
        V.box(g, fx + s1 + 3, fy, s2, fh, { r: 5, fill: V.alpha(C.b, pulse), stroke: C.b, lw: 1 });
        V.text(g, 'cabeçalho JSON (N B)', fx + s1 + 3 + s2 / 2, fy + fh / 2 + 4, { size: 11, align: 'center', maxW: s2 - 8 });
        const d0 = fx + s1 + s2 + 6, dw = fw - s1 - s2 - 6;
        V.box(g, d0, fy, dw, fh, { r: 5, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const nS = 36;
        for (let k = 1; k < nS; k++) { const xx = d0 + k * dw / nS; V.line(g, xx, fy + 4, xx, fy + fh - 4, { color: V.alpha(C.muted, 0.22), lw: 1 }); }
        const sel = (S.i * 7 + 5) % (nS - 3), sx = d0 + sel * dw / nS;
        g.fillStyle = V.alpha(C.c, 0.6); g.fillRect(sx + 1, fy + 3, dw / nS * 2.4, fh - 6);
        V.text(g, '↓ só isto é baixado (2 range requests)', fx, fy + fh + 15, { size: 11, color: C.d });
        if (!narrow) V.text(g, 'data_offsets do tensor escolhido', d0 + dw, fy + fh + 15, { size: 11, align: 'right', color: C.c });
        // ---------- visualização do tensor ----------
        const vy = 92, vh = h - vy - 10;
        const scanF = (t * 0.22) % 1;
        const scan = (x0, y0, ww, hh) => { const yy = y0 + scanF * hh; const gr = g.createLinearGradient(0, yy - 10, 0, yy + 10); gr.addColorStop(0, V.alpha(C.ink, 0)); gr.addColorStop(0.5, V.alpha(C.ink, 0.28)); gr.addColorStop(1, V.alpha(C.ink, 0)); g.fillStyle = gr; g.fillRect(x0, yy - 10, ww, 20); };
        const colQ = C.b, colK = C.a, colV = C.c;

        if (x.kind === 'qkv') {
          const ga = x.ga, R = x.shape[0], cr = R / 4, q = 3072, kk = ga ? 192 : 384, vv = ga ? 128 : 256, nkv = ga ? 1 : 2;
          const bx = 40, bw = narrow ? Math.min(90, w * 0.26) : Math.min(150, w * 0.24), top = vy + 26, HH = vh - 40;
          const sc = HH / R;
          V.text(g, `[${nf(R)} × 4096] ${x.dtype}`, bx + bw / 2, vy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          for (let c = 0; c < 4; c++) {
            const y0 = top + c * cr * sc;
            V.box(g, bx, y0, bw, q * sc, { r: 2, fill: V.alpha(colQ, 0.55) });
            V.box(g, bx, y0 + q * sc, bw, Math.max(2, kk * sc), { r: 1, fill: V.alpha(colK, 0.85) });
            V.box(g, bx, y0 + (q + kk) * sc, bw, Math.max(2, vv * sc), { r: 1, fill: V.alpha(colV, 0.85) });
            V.line(g, bx - 4, y0, bx + bw + 4, y0, { color: C.ink, lw: c ? 1.5 : 0.8 });
            V.text(g, 'c' + c, bx - 8, y0 + cr * sc / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
          }
          scan(bx, top, bw, HH);
          // zoom do chunk 0
          const zx = bx + bw + (narrow ? 40 : 70), zw = Math.min(narrow ? w - zx - 12 : 170, w - zx - 150 > 0 ? 170 : w - zx - 12), zs = HH / cr;
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
          g.beginPath(); g.moveTo(bx + bw, top); g.lineTo(zx, top); g.moveTo(bx + bw, top + cr * sc); g.lineTo(zx, top + HH); g.stroke(); g.setLineDash([]);
          V.text(g, 'zoom no chunk 0', zx + zw / 2, vy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          for (let hq = 0; hq < 16; hq++) {
            const y0 = top + hq * 192 * zs;
            V.box(g, zx, y0 + 0.5, zw, 192 * zs - 1, { r: 2, fill: V.alpha(colQ, 0.25 + 0.3 * ((hq % 2) ? 1 : 0.6)) });
            if (192 * zs >= 11) V.text(g, 'Q' + hq, zx + 6, y0 + 192 * zs / 2 + 4, { size: 11, font: 'mono', color: C.ink });
          }
          for (let k = 0; k < nkv; k++) { const y0 = top + (q + k * 192) * zs; V.box(g, zx, y0 + 0.5, zw, 192 * zs - 1, { r: 2, fill: V.alpha(colK, 0.7) }); V.text(g, 'K' + k, zx + 6, y0 + 192 * zs / 2 + 4, { size: 11, font: 'mono', color: C.ink }); }
          for (let k = 0; k < nkv; k++) { const y0 = top + (q + kk + k * 128) * zs; V.box(g, zx, y0 + 0.5, zw, 128 * zs - 1, { r: 2, fill: V.alpha(colV, 0.7) }); V.text(g, 'V' + k, zx + 6, y0 + 128 * zs / 2 + 4, { size: 11, font: 'mono', color: C.ink }); }
          // grade de blocos de escala (128 linhas)
          for (let m = 1; m * 128 < cr; m++) { const yy = top + m * 128 * zs; V.line(g, zx + zw * 0.55, yy, zx + zw, yy, { color: V.alpha(C.ink, 0.35), lw: 1 }); }
          if (ga) {
            const y0 = top + 3200 * zs, y1 = top + 3328 * zs;
            g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, zx - 3, y0, zw + 6, y1 - y0, 3); g.stroke();
          }
          scan(zx, top, zw, HH);
          // legenda à direita do zoom
          const lx2 = zx + zw + 14;
          if (w - lx2 > 120) {
            let yy = top + 8;
            yy = wrap(g, ga ? 'global: 16 heads de Q, 1 KV head por chunk' : (x.mtp ? 'MTP: mesma forma da SWA (tem sink)' : 'SWA: 16 heads de Q, 2 KV heads por chunk'), lx2, yy, w - lx2 - 10, 15, { size: 11.5, color: C.ink });
            yy = wrap(g, `chunk = 3072 + ${kk} + ${vv} = ${cr} linhas`, lx2, yy + 6, w - lx2 - 10, 15, { size: 11.5, font: 'mono', color: C.muted });
            yy = wrap(g, 'traços à direita: blocos de escala de 128 linhas', lx2, yy + 6, w - lx2 - 10, 15, { size: 11.5, color: C.muted });
            if (ga) wrap(g, 'contorno âmbar: o bloco 25 cobre as últimas 64 linhas de K e as primeiras 64 de V', lx2, yy + 6, w - lx2 - 10, 15, { size: 11.5, color: C.d });
            [[colQ, 'Q'], [colK, 'K'], [colV, 'V']].forEach(([c, l], i) => { V.box(g, lx2 + i * 44, top + HH - 16, 12, 12, { r: 3, fill: c }); V.text(g, l, lx2 + i * 44 + 17, top + HH - 6, { size: 11.5, color: C.muted }); });
          }
        } else if (x.kind === 'scale') {
          const ga = x.ga, nb = ga ? 27 : 29;
          const gx = 44, gw = Math.min(narrow ? w - 80 : w * 0.5, 380), top = vy + 26, gh = narrow ? vh * 0.6 : vh - 70;
          const ch = gh / nb, cw = gw / 32;
          V.text(g, `chunk 0 em detalhe: ${nb} blocos de linha × 32 de coluna`, gx + gw / 2, vy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: gw + 40 });
          const typ = (r) => (ga ? (r < 24 ? 'Q' : r === 24 ? 'K' : r === 25 ? 'KV' : 'Vp') : (r < 24 ? 'Q' : r < 27 ? 'K' : 'V'));
          const colOf = (ty) => (ty === 'Q' ? colQ : ty === 'K' ? colK : ty === 'KV' ? C.d : colV);
          for (let r = 0; r < nb; r++) {
            const ty = typ(r), yy = top + r * ch;
            g.fillStyle = V.alpha(colOf(ty), ty === 'KV' ? 0.9 : 0.6);
            g.fillRect(gx, yy, gw, ty === 'Vp' ? ch * 0.5 : ch);
            if (ty === 'Vp') { g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(gx, yy + ch * 0.5, gw, ch * 0.5); }
            if (r === 0 || r === 23 || r >= 24) V.text(g, String(r), gx - 6, yy + ch / 2 + 4, { size: 11, font: 'mono', align: 'right', color: r >= 24 ? C.ink : C.muted });
          }
          g.strokeStyle = V.alpha(C.bg || '#000', 0.7); g.lineWidth = 1; g.beginPath();
          for (let j = 1; j < 32; j++) { g.moveTo(gx + j * cw, top); g.lineTo(gx + j * cw, top + nb * ch); }
          for (let r = 1; r < nb; r++) { g.moveTo(gx, top + r * ch); g.lineTo(gx + gw, top + r * ch); }
          g.stroke();
          V.text(g, `× 4 chunks (c1–c3 repetem o padrão) = ${4 * nb} linhas de escala · [${x.shape[0]} × 32] F32`, gx, top + nb * ch + 18, { size: 11.5, color: C.muted, maxW: gw + 30 });
          // célula destacada que passeia
          const hr = Math.floor(t * 1.2) % nb, hj = Math.floor(t * 3.7) % 32;
          const hy = top + hr * ch;
          g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(gx + hj * cw, hy, cw, ch);
          const lx2 = narrow ? 16 : gx + gw + 24, ly = narrow ? top + gh + 42 : top + 8, lw2 = narrow ? w - 32 : w - lx2 - 12;
          let yy = wrap(g, `célula destacada: bloco de linhas ${hr}, colunas ${hj * 128}–${hj * 128 + 127} → um único fator F32 multiplica 128 × 128 pesos FP8`, lx2, ly, lw2, 15, { size: 11.5, color: C.ink });
          const items = ga ? [[colQ, 'Q: blocos 0–23'], [colK, 'K: bloco 24'], [C.d, 'K + V: bloco 25 (uma escala para os dois)'], [colV, 'V: bloco 26 (só 64 linhas)']] : [[colQ, 'Q: blocos 0–23'], [colK, 'K: blocos 24–26'], [colV, 'V: blocos 27–28']];
          items.forEach(([c, l]) => { yy += 4; V.box(g, lx2, yy - 10, 12, 12, { r: 3, fill: c }); yy = wrap(g, l, lx2 + 18, yy, lw2 - 18, 15, { size: 11.5, color: C.muted }); });
        } else if (x.kind === 'sink') {
          const cw = Math.min(26, (narrow ? w - 60 : w * 0.46) / 16), gx = 40, top = vy + 30;
          V.text(g, '64 logits de sink (valores ilustrativos), um por head', gx, vy + 14, { size: 11.5, color: C.muted });
          V.matrix(g, { x: gx, y: top, data: sinks, cell: cw, cmap: 'div', gap: 2 });
          V.text(g, 'h0', gx, top + 4 * (cw + 2) + 14, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'h63', gx + 16 * (cw + 2) - 2, top + 4 * (cw + 2) + 14, { size: 11, font: 'mono', align: 'right', color: C.muted });
          // softmax com e sem sink
          const sx0 = narrow ? 30 : gx + 16 * (cw + 2) + 50, sy0 = narrow ? top + 4 * (cw + 2) + 60 : top, sw = narrow ? w - 60 : w - sx0 - 20, sh = narrow ? vh - (sy0 - vy) - 50 : vh - 90;
          const sink = 1.2 + 2.2 * Math.sin(t * 0.9);
          const logits = [sink, 1.6, 1.1, 0.5, -0.2, 0.9];
          const p = V.softmax(logits);
          V.text(g, `softmax de uma query com o sink no denominador (sink = ${sink.toFixed(1)})`, sx0, sy0 - 8, { size: 11.5, color: C.ink, maxW: sw });
          V.bars(g, { x: sx0, y: sy0 + 18, w: sw, h: Math.max(60, sh - 40), values: p, max: Math.max(...p) * 1.15, labels: ['sink', 't1', 't2', 't3', 't4', 't5'], showValues: true, fmt: (v) => (v * 100).toFixed(0) + '%', colorFn: (v, i) => (i === 0 ? C.d : C.a), gap: 10 });
        } else if (x.kind === 'nibble') {
          const cols = narrow ? 6 : 8, rows = 5, cell = Math.min(34, ((narrow ? w - 60 : w * 0.42) - 10) / cols);
          const gx = 40, top = vy + 30;
          V.text(g, 'bytes do tensor U8 (hex, ilustrativos)', gx, vy + 14, { size: 11.5, color: C.muted });
          const cur = Math.floor(t / 1.6) % (cols * rows);
          for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
            const k = i * cols + j, b = bytes[k % bytes.length], on = k === cur;
            V.box(g, gx + j * (cell + 3), top + i * (cell + 3), cell, cell, { r: 5, fill: on ? V.alpha(C.d, 0.55) : V.alpha(C.b, 0.18), stroke: on ? C.d : null, lw: 2 });
            V.text(g, b.toString(16).toUpperCase().padStart(2, '0'), gx + j * (cell + 3) + cell / 2, top + i * (cell + 3) + cell / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          }
          const b = bytes[cur % bytes.length], lo = b & 15, hi2 = b >> 4, e = 122, s = Math.pow(2, e - 127);
          const px = narrow ? 24 : gx + cols * (cell + 3) + 40, py = narrow ? top + rows * (cell + 3) + 26 : top, pw = narrow ? w - 48 : w - px - 16;
          const bw = Math.min(28, pw / 9.5);
          V.text(g, `byte 0x${b.toString(16).toUpperCase().padStart(2, '0')} = dois pesos de 4 bits`, px, py + 4, { size: 12, color: C.ink, weight: 600, maxW: pw });
          for (let k = 0; k < 8; k++) {
            const bit = (b >> (7 - k)) & 1, isHi = k < 4;
            V.box(g, px + k * (bw + 3) + (isHi ? 0 : 8), py + 16, bw, bw, { r: 4, fill: V.alpha(isHi ? C.c : C.a, bit ? 0.7 : 0.18), stroke: isHi ? C.c : C.a, lw: 1 });
            V.text(g, String(bit), px + k * (bw + 3) + (isHi ? 0 : 8) + bw / 2, py + 16 + bw / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
          }
          let yy = py + 16 + bw + 22;
          const line = (txt, col) => { yy = wrap(g, txt, px, yy, pw, 16, { size: 12, font: 'mono', color: col || C.ink }); yy += 2; };
          line(`nibble baixo ${lo.toString(2).padStart(4, '0')} → E2M1 ${e2m1(lo)}`, C.a);
          line(`nibble alto  ${hi2.toString(2).padStart(4, '0')} → E2M1 ${e2m1(hi2)}`, C.c);
          line(`escala E8M0 do bloco: e = ${e} → 2^(${e}−127) = 1/${1 / s}`, C.d);
          line(`pesos: w[2j] = ${(e2m1(lo) * s).toFixed(4)} · w[2j+1] = ${(e2m1(hi2) * s).toFixed(4)}`);
          yy += 4;
          wrap(g, 'convenção “nibble baixo = elemento par” (OCP/gpt-oss): confirmar comparando o GEMM desquantizado com o do kernel', px, yy, pw, 15, { size: 11.5, color: C.muted });
        } else if (x.kind === 'e8m0') {
          const gx = 40, top = vy + 30, gw = narrow ? w - 80 : w * 0.55, cw = gw / 16, chh = Math.min(26, (narrow ? vh * 0.45 : vh - 60) / 8);
          V.text(g, 'uma linha: 4096 pesos = 128 blocos de 32 (8 fileiras × 16 blocos)', gx, vy + 14, { size: 11.5, color: C.muted, maxW: w - 50 });
          const cur = Math.floor(t / 1.2) % 128;
          for (let k = 0; k < 128; k++) {
            const r = Math.floor(k / 16), c = k % 16, e = exps[k], on = k === cur;
            V.box(g, gx + c * cw + 1, top + r * (chh + 3), cw - 2, chh, { r: 3, fill: V.cmap((e - 117) / 10, 'heat'), stroke: on ? C.ink : null, lw: 2 });
          }
          const e = exps[cur];
          const px = narrow ? 24 : gx + gw + 30, py = narrow ? top + 8 * (chh + 3) + 24 : top + 4, pw = narrow ? w - 48 : w - px - 16;
          let yy = wrap(g, `bloco ${cur}: pesos ${cur * 32}–${cur * 32 + 31}`, px, py, pw, 16, { size: 12, color: C.ink, weight: 600 });
          yy = wrap(g, `e = ${e} → escala 2^(${e}−127) = 1/${Math.pow(2, 127 - e)}`, px, yy + 2, pw, 16, { size: 12, font: 'mono', color: C.d });
          yy = wrap(g, 'cada um dos 32 pesos = E2M1(nibble) × escala', px, yy + 4, pw, 16, { size: 12, color: C.ink });
          yy = wrap(g, 'custo: (32 × 4 + 8) / 32 = 4,25 bits por peso', px, yy + 8, pw, 16, { size: 12, font: 'mono', color: C.a });
          wrap(g, 'cor: expoente da escala (valores ilustrativos). Uma potência de 2 pura: multiplicar por ela é só somar no expoente', px, yy + 8, pw, 15, { size: 11.5, color: C.muted });
        } else if (x.kind === 'oproj') {
          const rw = Math.min(w - 90, 520), rh = Math.min(vh - 90, rw / 2), rx = (w - rw) / 2 + 10, ry = vy + 34;
          V.text(g, 'entrada: 8192 = 64 heads × 128 (v_head_dim)', rx + rw / 2, vy + 16, { size: 11.5, align: 'center', color: C.muted, maxW: rw });
          const hw = rw / 64;
          for (let k = 0; k < 64; k++) { g.fillStyle = V.alpha(k % 2 ? C.c : C.b, 0.35 + 0.15 * Math.sin(k)); g.fillRect(rx + k * hw, ry, hw + 0.5, rh); }
          g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(rx, ry, rw, rh);
          const sxx = rx + ((t * 0.18) % 1) * rw; const gr = g.createLinearGradient(sxx - 14, 0, sxx + 14, 0); gr.addColorStop(0, V.alpha(C.ink, 0)); gr.addColorStop(0.5, V.alpha(C.ink, 0.3)); gr.addColorStop(1, V.alpha(C.ink, 0)); g.fillStyle = gr; g.fillRect(sxx - 14, ry, 28, rh);
          V.text(g, 'saída 4096', rx - 8, ry + rh / 2 + 4, { size: 11, align: 'right', font: 'mono', color: C.muted });
          wrap(g, 'BF16: 67,1 MB por camada × 48 (+ MTP) ≈ 3,2 GB lidos a cada passo de decode. Com FP8 online (#59083): −0,8 GiB por rank e menos bytes por passo.', rx, ry + rh + 24, rw, 16, { size: 12, color: C.ink });
        }
      });
    })();

    /* =====================================================================
       Figura 3 — calculadora de memória (mem_budget.py ao vivo)
       ===================================================================== */
    (function figCalc() {
      const fig = root.querySelector('#fig-calc');
      setH(fig, 500);
      const CFG = { nGA: 9, nSWA: 39, kvGA: 4, kvSWA: 8, dGA: 320, dSWA: 320, win: 128, vis: 121.69, weights: 82.7, block: 64, k: 7, tp: 2, nDr: 5, kvDr: 8, dDr: 256, winDr: 1024 };
      const S = { gmu: 0.85, over: 4, kv: 'fp8', group: 5, ctx: 262144, textOnly: false, draft: true, shown: 0, r: null };
      const ctxLabel = (L) => (L >= 1048576 ? '1M' : (L / 1024) + 'K');
      function calc() {
        const b = S.kv === 'bf16' ? 2 : 1;
        const gaTok = CFG.nGA * (CFG.kvGA / CFG.tp) * CFG.dGA * b;
        const swaTok = CFG.nSWA * (CFG.kvSWA / CFG.tp) * CFG.dSWA * b;
        const blocks = (win) => (Math.ceil((win + CFG.k) / CFG.block) + 1) * CFG.block;
        const swaReq = swaTok * blocks(CFG.win);
        const drTok = CFG.nDr * (CFG.kvDr / CFG.tp) * CFG.dDr * 2; // drafter em BF16
        const drReq = S.draft ? drTok * blocks(CFG.winDr) : 0;
        const pad = Math.ceil(CFG.nGA / S.group) * S.group / CFG.nGA;
        const weights = CFG.weights - (S.textOnly ? 1.6 : 0);
        const budget = CFG.vis * S.gmu;
        const kvGiB = budget - weights - S.over;
        const kvB = kvGiB * Math.pow(2, 30);
        const cap = kvB / (gaTok * pad), ideal = kvB / gaTok;
        const gaPart = S.ctx * gaTok * pad, perReq = gaPart + swaReq + drReq;
        return { b, gaTok, swaTok, swaReq, drTok, drReq, pad, weights, budget, kvGiB, cap, ideal, gaPart, perReq, nreq: kvB / perReq };
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'GMU', min: 0.7, max: 0.95, step: 0.01, value: S.gmu, fmt: (v) => v.toFixed(2), onInput: (v) => { S.gmu = v; upd(); } });
      UI.slider(ctl, { label: 'Overhead', min: 2, max: 10, step: 0.5, value: S.over, fmt: (v) => v.toFixed(1) + ' GiB', onInput: (v) => { S.over = v; upd(); } });
      UI.seg(ctl, { label: 'KV', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }], value: 'fp8', onChange: (v) => { S.kv = v; upd(); } });
      UI.seg(ctl, { label: 'Grupo', options: [{ v: 5, label: '5 (padrão)' }, { v: 3, label: '3 (#58207)' }], value: 5, onChange: (v) => { S.group = v; upd(); } });
      UI.seg(ctl, { label: 'Contexto', options: [{ v: 32768, label: '32K' }, { v: 131072, label: '128K' }, { v: 262144, label: '256K' }, { v: 1048576, label: '1M' }], value: 262144, onChange: (v) => { S.ctx = v; upd(); } });
      UI.toggle(ctl, { label: 'KV do drafter DFlash (BF16)', value: true, onChange: (v) => { S.draft = v; upd(); } });
      UI.toggle(ctl, { label: '--language-model-only (−1,6 GiB de pesos)', value: false, onChange: (v) => { S.textOnly = v; upd(); } });
      const rPool = UI.readout(ctl, 'Pool de KV', '—'), rCap = UI.readout(ctl, 'Capacidade', '—'), rReq = UI.readout(ctl, 'Requisições cheias', '—');
      function upd() {
        S.r = calc();
        const r = S.r;
        rPool.set(r.kvGiB > 0 ? nf(r.kvGiB, 2) + ' GiB' : 'não sobra');
        rCap.set(r.kvGiB > 0 ? '~' + nf(r.cap / 1e6, 2) + ' M tok' : '—');
        rReq.set(r.kvGiB > 0 ? nf(r.nreq, 1) + ' × ' + ctxLabel(S.ctx) : '—');
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(), r = S.r;
        const x0 = 16, bw = w - 32;
        // --- barra 1: memória visível de um rank
        V.text(g, 'MEMÓRIA VISÍVEL DE UM RANK · 121,69 GiB [medido]', x0, 16, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const y1 = 26, bh = 34, sc = bw / CFG.vis;
        const segs = [['pesos', r.weights, C.b], ['overhead', S.over, C.d], ['pool KV', Math.max(0, r.kvGiB), C.a], ['host', CFG.vis - r.budget, null]];
        let xx = x0;
        segs.forEach(([n, v, c]) => {
          const ww = Math.max(0, v * sc);
          if (c) V.box(g, xx, y1, Math.max(1, ww - 1), bh, { r: 4, fill: V.alpha(c, 0.7) });
          else {
            V.box(g, xx, y1, ww, bh, { r: 4, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.45), lw: 1 });
            g.save(); V.rr(g, xx, y1, ww, bh, 4); g.clip(); g.strokeStyle = V.alpha(C.muted, 0.28); g.lineWidth = 1;
            for (let k = -bh; k < ww; k += 8) { g.beginPath(); g.moveTo(xx + k, y1 + bh); g.lineTo(xx + k + bh, y1); g.stroke(); }
            g.restore();
          }
          const lab = `${n} ${nf(v, 1)}`;
          g.font = V.font(11.5, 'sans', 600);
          if (g.measureText(lab).width + 10 < ww) V.text(g, lab, xx + ww / 2, y1 + bh / 2 + 4, { size: 11.5, weight: 600, align: 'center', color: c ? C.ink : C.muted });
          xx += ww;
        });
        const gx = x0 + r.budget * sc;
        V.line(g, gx, y1 - 5, gx, y1 + bh + 5, { color: C.ink, lw: 1.5, dash: [3, 3] });
        V.text(g, `GMU × total = ${nf(r.budget, 1)} GiB`, Math.min(gx, w - 16), y1 + bh + 16, { size: 11, font: 'mono', align: gx > w - 140 ? 'right' : 'center', color: C.ink });
        // --- barra 2: pool → requisições
        const y2 = 110, bh2 = 40;
        V.text(g, `POOL DE KV → REQUISIÇÕES DE ${ctxLabel(S.ctx)} CHEIAS AO MESMO TEMPO`, x0, y2 - 9, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        V.box(g, x0, y2, bw, bh2, { r: 6, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.5), lw: 1 });
        if (r.kvGiB <= 0) {
          V.text(g, '✕ não sobra KV: reduza pesos ou overhead, ou aumente o GMU', x0 + bw / 2, y2 + bh2 / 2 + 4, { size: 12.5, align: 'center', color: C.red, weight: 600, maxW: bw - 12 });
          S.shown = 0;
        } else {
          S.shown += (r.nreq - S.shown) * Math.min(1, dt * 6);
          const tw = bw / r.nreq, gaF = r.gaPart / r.perReq, swF = r.swaReq / r.perReq, full = Math.min(600, Math.floor(S.shown));
          const gp = tw > 5 ? 1.5 : tw > 2.5 ? 0.6 : 0;
          for (let k = 0; k < full; k++) {
            const tx = x0 + k * tw;
            g.fillStyle = V.alpha(C.a, 0.75); g.fillRect(tx + gp / 2, y2 + 4, Math.max(0.5, tw * gaF - gp), bh2 - 8);
            g.fillStyle = V.alpha(C.c, 0.9); g.fillRect(tx + tw * gaF - gp / 2, y2 + 4, Math.max(0.6, tw * swF), bh2 - 8);
            if (r.drReq) { g.fillStyle = V.alpha(C.f, 0.9); g.fillRect(tx + tw * (gaF + swF) - gp / 2, y2 + 4, Math.max(0.6, tw * (1 - gaF - swF)), bh2 - 8); }
          }
          const rest = x0 + full * tw;
          if (rest < x0 + bw - 2) { g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.muted, 0.6); g.strokeRect(rest + 1, y2 + 4, x0 + bw - rest - 5, bh2 - 8); g.setLineDash([]); }
        }
        // --- barra 3: anatomia de uma requisição
        const y3 = 196;
        V.text(g, `ANATOMIA DE UMA REQUISIÇÃO DE ${ctxLabel(S.ctx)} (${S.kv.toUpperCase()})`, x0, y3 - 9, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const gaF = r.gaPart / r.perReq, gw = Math.max(4, bw * gaF);
        const fixW = Math.max(r.drReq ? 8 : 4, bw - gw - 2), swW = r.drReq ? Math.max(4, fixW * r.swaReq / (r.swaReq + r.drReq)) : fixW;
        V.box(g, x0, y3, gw, 30, { r: 5, fill: V.alpha(C.a, 0.7) });
        V.box(g, x0 + gw + 2, y3, swW, 30, { r: 5, fill: V.alpha(C.c, 0.85) });
        if (r.drReq) V.box(g, x0 + gw + 2 + swW, y3, Math.max(3, fixW - swW), 30, { r: 5, fill: V.alpha(C.f, 0.85) });
        const gaTxt = `global: ${nf(S.ctx)} × ${nf(r.gaTok * r.pad)} B = ${nf(r.gaPart / 1e6, 0)} MB`;
        g.font = V.font(11.5, 'sans', 600);
        if (g.measureText(gaTxt).width + 12 < gw) V.text(g, gaTxt, x0 + 8, y3 + 19.5, { size: 11.5, weight: 600, color: C.ink });
        let yy = y3 + 50;
        yy = wrap(g, `${nf(r.gaTok)} B/token/rank × ${nf(r.pad, 2)} de enchimento (grupo ${S.group}) = ${nf(r.gaTok * r.pad)} B · fixos: SWA ${nf(r.swaReq / 1e6, 1)} MB (janela 128 + k 7, blocos de 64)` + (r.drReq ? ` + drafter ${nf(r.drReq / 1e6, 1)} MB (janela 1024, BF16)` : ' (sem drafter)') + ` = ${nf(100 * (1 - gaF), 1)}% da requisição`, x0, yy, bw, 16, { size: 12, color: C.ink });
        if (r.kvGiB > 0) yy = wrap(g, `capacidade ≈ ${nf(r.cap / 1e6, 2)} M tokens (sem enchimento: ${nf(r.ideal / 1e6, 2)} M) · ${nf(r.nreq, 1)} requisições cheias de ${ctxLabel(S.ctx)}`, x0, yy + 4, bw, 16, { size: 12, color: C.a, weight: 600 });
        yy = wrap(g, `no boot o vLLM exige ≥ ${nf(r.budget, 1)} GiB livres; fora do orçamento ficam ${nf(CFG.vis - r.budget, 1)} GiB para SO, page cache, API e processor multimodal`, x0, yy + 4, bw, 16, { size: 12, color: C.muted });
        if (S.gmu >= 0.895) yy = wrap(g, '⚠ GMU ≥ 0,90: acima do que a comunidade usa no UMA (0,80–0,85)', x0, yy + 4, bw, 16, { size: 12, color: C.d, weight: 600 });
        yy = wrap(g, 'referência [medido, #58128, pool de 17 GiB]: BF16 1.365.757 · FP8 2.731.530 tokens', x0, yy + 4, bw, 16, { size: 11.5, font: 'mono', color: C.muted });
        // legenda
        const ly = Math.min(h - (bw < 480 ? 26 : 10), yy + 16);
        const legW = Math.min(120, bw / 3), legN = bw < 480 ? 3 : 5;
        [[C.b, 'pesos'], [C.d, 'overhead'], [C.a, 'KV global'], [C.c, 'KV SWA (fixo)'], [C.f, 'KV drafter (fixo)']].forEach(([c, l], i) => { const lx = x0 + (i % legN) * (bw < 480 ? legW : Math.min(120, bw / 5)), lyy = ly + Math.floor(i / legN) * 16; V.box(g, lx, lyy - 10, 11, 11, { r: 3, fill: c }); V.text(g, l, lx + 16, lyy, { size: 11, color: C.muted }); });
      });
    })();

    /* =====================================================================
       Figura 4 — árvore de decisão: formato e topologia
       ===================================================================== */
    (function figTree() {
      const fig = root.querySelector('#fig-arvore');
      setH(fig, 600);
      const N = {
        Q1: { q: true, row: 0, t: 'Algum checkpoint de ~4 bits publicado cabe nos nós com folga para KV?' },
        Q2: { q: true, row: 1, t: 'O menor que cabe é o oficial?' },
        Q3: { q: true, row: 2, t: 'Existe kernel desse formato no sm_121?' },
        Q4: { q: true, row: 4, t: 'Heads, KV heads, experts e moe_intermediate dividem por TP = nº de nós?' },
        Q5: { q: true, row: 5, t: 'Os pesos vêm pré-fatiados para outro TP?' },
        A: { row: 0, t: 'Quantize você mesmo e meça KLD contra a referência' },
        B: { row: 1, t: 'Checkpoint da comunidade + KLD contra o oficial' },
        C: { row: 2, t: 'Kernel existe: siga para a topologia' },
        D: { row: 3, t: 'Overlay ou porte de kernel (ex.: exl3 no GLM)' },
        E: { row: 4, t: 'Replicar KV heads, padding ou outro TP' },
        F: { row: 5, t: 'Remontar chunks: caminho exato + teste de shards' },
        G: { q: true, row: 6, g: true, t: 'TP = nós; EP como A/B; PP só para throughput' },
      };
      const EDGES = [['Q1', 'Q2', 'sim'], ['Q2', 'Q3', 'sim'], ['Q3', 'Q4', ''], ['Q4', 'Q5', 'sim'], ['Q5', 'G', 'não'], ['Q1', 'A', 'não'], ['Q2', 'B', 'não'], ['Q3', 'C', 'sim'], ['Q3', 'D', 'não'], ['Q4', 'E', 'não'], ['Q5', 'F', 'sim']];
      const PRE = {
        mimo: { path: ['Q1', 'Q2', 'Q3', 'C', 'Q4', 'Q5', 'F', 'G'], n: { Q1: 'sim: o oficial dá ~82,7 GiB por rank [medido: SGLang TP2], sobrando ~17 GiB de KV com GMU 0,85', Q2: 'sim: o MXFP4 oficial (4,25 bits) é menor que o NVFP4 da comunidade (~4,5 bits, ~187 GB)', Q3: 'sim: Marlin MXFP4 (W4A16/W4A8) roda no sm_121, com o DeepGEMM desligado (G5)', C: 'use o oficial: é a referência, KLD 0 por definição', Q4: 'sim: 64 Q heads, 4/8 KV heads, 256 experts e 2048/2 = 1024 dividem por TP=2', Q5: 'sim: tp_size: 4 no índice; o QKV vem em 4 chunks', F: 'cada rank junta 2 chunks: caminho exato (#59053) e teste de CPU dos shards', G: 'TP=2 com --nnodes 2; EP como A/B no Passo 9; PP fora' } },
        glm: { path: ['Q1', 'Q2', 'B', 'Q3', 'D', 'Q4', 'Q5', 'G'], n: { Q1: 'sim, mas não o oficial: o FP8 oficial tem 305,8 GiB; o EXL3 4 bpw da comunidade, ~164 GiB', Q2: 'não: o menor que cabe é o EXL3 da comunidade', B: 'EXL3 TR3 4 bpw: KLD 0,0246 nats, ≈ FP8 oficial com 54% dos bytes', Q3: 'não no vLLM stock: o método exl3 não existe no motor', D: 'overlay exl3.py + exllamav3_ext compilado para sm_121a no Dockerfile da receita', Q4: 'sim em TP=2 (TP=3 exige padding de heads 64 → 66)', Q5: 'não', G: 'TP=2; irmãos TP=3 e TP=4 opcionais' } },
        qwen: { path: ['Q1', 'Q2', 'B', 'Q3', 'C', 'Q4', 'Q5', 'G'], n: { Q1: 'sim: o NVFP4 da NVIDIA tem 133 GB (não cabe num Spark sozinho)', Q2: 'não: o FP8 oficial também cabe (lane start-fp8.sh, ~500k tokens de KV), mas o NVFP4 é menor', B: 'nvidia/Qwen3.8-Flash-Next-NVFP4 como padrão; o FP8 oficial vira lane alternativa', Q3: 'sim: experts W4A4 (FlashInfer CUTLASS SM120), com overlay de fallback MXFP8 por shape', C: 'kernel existe: siga para a topologia', Q4: 'sim', Q5: 'não', G: 'TP=2 + EP (--enable-expert-parallel, allgather_reducescatter): 256 experts por GPU' } },
        ds: { path: ['Q1', 'Q2', 'Q3', 'C', 'Q4', 'Q5', 'G'], n: { Q1: 'sim: ~157 GiB de checkpoint oficial; 79,17 GiB por rank em TP=2', Q2: 'sim: experts MXFP4 (E8M0 a cada 32) + lineares FP8 128×128, direto do fabricante', Q3: 'sim, na imagem da receita: flashinfer_b12x em W4A16', C: 'use o oficial', Q4: 'sim em TP=2 (TP=3 exige pad de heads 64 → 72)', Q5: 'não', G: 'TP=2, PP=1; TP=3 como lane de capacidade' } },
        naive: { path: ['Q1', 'A', 'Q3', 'C', 'Q4', 'Q5', 'G'], n: { Q1: 'não: BF16 617,8 GB e FP8 ~310 GB; os NVFP4/EXL3 da comunidade têm um dia de vida e nenhuma KLD publicada', A: 'quantize os experts (NVFP4 calibrado, llm-compressor ou ModelOpt) e meça KLD contra o FP8 em ≥ 3 nós', Q3: 'kernels NVFP4 existem, mas o modelo não está no registry: o Passo 4 vira plugin', C: 'siga para a topologia; o trabalho de motor vem no Passo 4', Q4: 'sim: mesma espinha do MiMo (4 KV heads, 256 experts, hidden 4096)', Q5: 'desconhecido: leia o tp_size do índice na triagem', G: 'TP=2 como ponto de partida' } },
      };
      const S = { p: 'mimo', t0: 0 };
      const ctl = UI.controls(fig);
      const notes = LA.h('dl', { class: 'la-kv', style: { margin: '0', padding: '12px 18px 6px', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(notes, ctl);
      const updNotes = () => { const P = PRE[S.p]; notes.innerHTML = P.path.map((id) => `<dt>${id}</dt><dd>${P.n[id] || ''}</dd>`).join(''); };
      let H;
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'mimo', label: 'MiMo' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'naive', label: 'Naive-N0.5' }], value: 'mimo', onChange: (v) => { S.p = v; S.t0 = H ? H.t : 0; updNotes(); } });
      updNotes();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pc = { mimo: C.a, glm: C.glm, qwen: C.qwen, ds: C.ds, naive: C.f }[S.p] || C.a;
        const qx = 10, qw = Math.floor(w * 0.56) - 18, lx = qx + qw + 30, lw = w - lx - 10;
        const top = 10, rowH = (h - 20) / 7, bh = Math.min(w < 480 ? 66 : 54, rowH - 12);
        const box = (id) => { const n = N[id], y = top + n.row * rowH + (rowH - bh) / 2; return n.q ? { x: qx, y, w: qw, h: bh } : { x: lx, y, w: lw, h: bh }; };
        const ctr = (id) => { const b = box(id); return [b.x + b.w / 2, b.y + b.h / 2]; };
        const P = PRE[S.p], vis = new Set(P.path);
        const onEdge = (a, b) => { const i = P.path.indexOf(a); return i >= 0 && P.path[i + 1] === b; };
        // arestas estáticas
        EDGES.forEach(([a, b, lab]) => {
          const A = box(a), B = box(b), on = onEdge(a, b) || (a === 'Q3' && b === 'Q4');
          let x1, y1, x2, y2;
          if (N[b].q) { x1 = A.x + A.w / 2; y1 = A.y + A.h; x2 = B.x + B.w / 2; y2 = B.y; } else { x1 = A.x + A.w; y1 = A.y + A.h / 2; x2 = B.x; y2 = B.y + B.h / 2; }
          V.arrow(g, x1, y1, x2, y2, { color: V.alpha(C.muted, on && vis.has(a) && vis.has(b) ? 0.9 : 0.35), lw: 1.3, head: 6 });
          if (lab) V.text(g, lab, (x1 + x2) / 2 + (N[b].q ? 6 : 0), (y1 + y2) / 2 + (N[b].q ? 4 : -4), { size: 11, color: C.muted, align: N[b].q ? 'left' : 'center' });
        });
        // caminho do preset: roteado pelas bordas das caixas (portas), nunca por cima do texto
        const port = (id, side) => { const b = box(id); return side === 'top' ? [b.x + b.w / 2, b.y] : side === 'bottom' ? [b.x + b.w / 2, b.y + b.h] : side === 'left' ? [b.x, b.y + b.h / 2] : [b.x + b.w, b.y + b.h / 2]; };
        const gx2 = lx - 11;
        const pts = [];
        for (let i = 0; i < P.path.length - 1; i++) {
          const a = P.path[i], b = P.path[i + 1], aq = N[a].q, bq = N[b].q;
          if (aq && bq) pts.push(port(a, 'bottom'), port(b, 'top'));
          else if (aq && !bq) pts.push(port(a, 'right'), port(b, 'left'));
          else { const L = port(a, 'left'), R = port(b, 'right'); pts.push(L, [gx2, L[1]], [gx2, R[1]], R); }
        }
        g.strokeStyle = V.alpha(pc, 0.75); g.lineWidth = 3; g.setLineDash([6, 5]); g.lineDashOffset = -t * 16;
        g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke(); g.setLineDash([]);
        // caixas (base opaca + cor)
        Object.keys(N).forEach((id) => {
          const n = N[id], b = box(id), on = vis.has(id);
          const col = n.q && !n.g ? C.b : n.g ? pc : C.a;
          V.box(g, b.x, b.y, b.w, b.h, { r: n.q ? 10 : 8, fill: C.bg || C.panel });
          V.box(g, b.x, b.y, b.w, b.h, { r: n.q ? 10 : 8, fill: on ? V.alpha(n.q && !n.g ? C.b : pc, n.g ? 0.28 : 0.18) : V.alpha(C.muted, 0.06), stroke: on ? (n.q && !n.g ? col : pc) : V.alpha(C.muted, 0.3), lw: on ? 2 : 1 });
          const lab = (n.q && !n.g ? id + ' · ' : n.g ? '' : id + ' · ') + n.t;
          const fs = b.w < 170 ? 11 : 11.5;
          g.save(); V.rr(g, b.x, b.y, b.w, b.h, 8); g.clip();
          wrap(g, lab, b.x + 8, b.y + 16, b.w - 14, 13.5, { size: fs, color: on ? C.ink : C.muted, weight: on && n.q ? 500 : 400 });
          g.restore();
        });
        // ponto viajante
        const segs = []; let L = 0;
        for (let i = 0; i < pts.length - 1; i++) { const d = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]); segs.push(d); L += d; }
        const cyc = L / 120 + 1.2, u = Math.min(1, ((t - S.t0) % cyc) / (cyc - 1.2));
        let dd = u * L, k = 0; while (k < segs.length - 1 && dd > segs[k]) { dd -= segs[k]; k++; }
        const f = segs[k] ? dd / segs[k] : 0;
        const px = lerp(pts[k][0], pts[k + 1][0], f), py = lerp(pts[k][1], pts[k + 1][1], f);
        V.glow(g, px, py, 18, pc, 1); V.dot(g, px, py, 4.5, pc);
      });
    })();

    /* =====================================================================
       Figura 5 — QKV pré-fatiado: 4 chunks no disco → ranks
       ===================================================================== */
    (function figQKV() {
      const fig = root.querySelector('#fig-qkv');
      setH(fig, 790);
      const S = { tp: 2, ga: true, exact: false, t0: -10, geo: null };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'TP', options: [{ v: 4, label: 'TP=4' }, { v: 2, label: 'TP=2' }], value: 2, onChange: (v) => { S.tp = v; rebuild(); } });
      UI.seg(ctl, { label: 'Camada', options: [{ v: 'ga', label: 'global' }, { v: 'swa', label: 'SWA' }], value: 'ga', onChange: (v) => { S.ga = v === 'ga'; rebuild(); } });
      UI.seg(ctl, { label: 'Caminho', options: [{ v: 'req', label: 'requantizar (stock)' }, { v: 'exact', label: 'exato (#59053)' }], value: 'req', onChange: (v) => { S.exact = v === 'exact'; rebuild(); } });
      const rConf = UI.readout(ctl, 'Blocos com >1 escala de origem', '—');
      const rShard = UI.readout(ctl, 'Shard por rank', '—');
      const rPad = UI.readout(ctl, 'Padding', '—');
      function rebuild() {
        const q = 3072, kk = S.ga ? 192 : 384, vv = S.ga ? 128 : 256, cr = q + kk + vv, R = 4 * cr, per = 4 / S.tp;
        // caminho exato nas globais: K+V = 320 linhas → +64 de zero para fechar 3 blocos de 128
        const padN = S.exact && S.tp < 4 ? (128 - ((kk + vv) % 128)) % 128 : 0;
        const pieces = [];
        let rowsR = 0;
        for (let r = 0; r < S.tp; r++) {
          let d = 0;
          const add = (c, part) => {
            const off = part === 'Q' ? 0 : part === 'K' ? q : part === 'V' ? q + kk : 0;
            const len = part === 'Q' ? q : part === 'K' ? kk : part === 'V' ? vv : padN;
            pieces.push({ c, part, src: c * cr + off, srcIn: off, len, rank: r, dst: d });
            d += len;
          };
          const cs = Array.from({ length: per }, (_, j) => r * per + j);
          if (padN) { cs.forEach((c) => add(c, 'Q')); cs.forEach((c) => { add(c, 'K'); add(c, 'V'); add(c, 'P'); }); }
          else ['Q', 'K', 'V'].forEach((part) => cs.forEach((c) => add(c, part)));
          rowsR = d;
        }
        // de que bloco(s) de escala do disco vem cada bloco de 128 linhas do shard
        let conf = 0, tot = 0; const confB = new Set(), src = {};
        for (let r = 0; r < S.tp; r++) {
          const nb = Math.ceil(rowsR / 128);
          for (let b = 0; b < nb; b++) {
            const set = new Set();
            for (let row = b * 128; row < Math.min(rowsR, (b + 1) * 128); row += 64) {
              const p = pieces.find((pp) => pp.rank === r && row >= pp.dst && row < pp.dst + pp.len);
              if (!p || p.part === 'P') continue;
              set.add('c' + p.c + '·b' + Math.floor((p.srcIn + row - p.dst) / 128));
            }
            src[r + ':' + b] = [...set];
            tot++; if (set.size > 1) { conf++; confB.add(r + ':' + b); }
          }
        }
        S.geo = { q, kk, vv, cr, R, per, pieces, conf, tot, confB, src, rowsR, padN };
        S.t0 = H ? H.t : 0;
        rConf.set(`${conf} de ${tot}`);
        rShard.set(`[${nf(rowsR)} × 4096]`);
        rPad.set(padN ? `+${padN} linhas/chunk (zeros)` : 'nenhum');
      }
      rebuild();
      const hatch = (g, x, y, ww, hh, col) => {
        g.save(); g.beginPath(); g.rect(x, y, ww, hh); g.clip();
        g.fillStyle = V.alpha(col, 0.1); g.fillRect(x, y, ww, hh);
        g.strokeStyle = V.alpha(col, 0.55); g.lineWidth = 1;
        for (let k = -hh; k < ww; k += 6) { g.beginPath(); g.moveTo(x + k, y + hh); g.lineTo(x + k + hh, y); g.stroke(); }
        g.restore();
      };
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), G = S.geo, narrow = w < 560;
        const colOf = (p) => (p === 'Q' ? C.b : p === 'K' ? C.a : C.c);
        const top = 50, bot = narrow ? 112 : 80, zh = narrow ? 300 : 0;
        const HH = h - top - bot - (narrow ? zh + 12 : 0);
        const sx = 38, sw = narrow ? Math.min(78, w * 0.22) : Math.min(120, w * 0.2), sc = HH / G.R;
        V.text(g, 'disco (tp_size = 4)', sx + sw / 2, 18, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        V.text(g, `[${nf(G.R)} × 4096]`, sx + sw / 2, 34, { size: 11, font: 'mono', align: 'center', color: C.muted });
        // fonte (fantasma)
        for (let c = 0; c < 4; c++) {
          const y0 = top + c * G.cr * sc;
          [['Q', 0, G.q], ['K', G.q, G.kk], ['V', G.q + G.kk, G.vv]].forEach(([p, o, l]) => { g.fillStyle = V.alpha(colOf(p), 0.22); g.fillRect(sx, y0 + o * sc, sw, Math.max(1.5, l * sc)); });
          for (let m = 1; m * 128 < G.cr; m++) V.line(g, sx, y0 + m * 128 * sc, sx + sw, y0 + m * 128 * sc, { color: V.alpha(C.ink, 0.12), lw: 0.8 });
          if (S.ga) { g.strokeStyle = C.d; g.lineWidth = 1.5; g.strokeRect(sx - 2, y0 + 3200 * sc, sw + 4, Math.min(128, G.cr - 3200) * sc); }
          V.line(g, sx - 5, y0, sx + sw + 5, y0, { color: C.ink, lw: c ? 1.4 : 0.8 });
          V.text(g, 'c' + c, sx - 9, y0 + G.cr * sc / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        }
        // destino
        const dx = narrow ? w * 0.5 : Math.max(w * 0.54, w - 192), dw = Math.min(160, w - dx - 18), gapR = 24;
        const rh = (HH - gapR * (S.tp - 1)) / S.tp, rsc = rh / G.rowsR;
        const qh = 16 * G.per, kvh = (S.ga ? 1 : 2) * G.per;
        const nb = Math.ceil(G.rowsR / 128);
        for (let r = 0; r < S.tp; r++) {
          const ry = top + r * (rh + gapR);
          V.text(g, narrow ? `rank ${r} · ${qh} Q · ${kvh} KV` : `rank ${r} · Q ${qh} heads · KV ${kvh}`, dx, ry - 6, { size: 11, font: 'mono', color: C.ink, maxW: w - dx - 8 });
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.strokeRect(dx - 1, ry - 1, dw + 2, rh + 2);
          for (let b = 0; b < nb; b++) {
            if (!G.confB.has(r + ':' + b)) continue;
            const by = ry + b * 128 * rsc, bhh = Math.min(128, G.rowsR - b * 128) * rsc;
            g.fillStyle = V.alpha(C.d, 0.55 + 0.25 * Math.sin(t * 4));
            g.fillRect(dx + dw + 4, by, 10, Math.max(3, bhh));
          }
        }
        // peças animadas (o padding não tem origem: surge no lugar)
        const n = G.pieces.length;
        G.pieces.forEach((pc, i) => {
          const yd = top + pc.rank * (rh + gapR) + pc.dst * rsc, hd = pc.len * rsc;
          if (pc.part === 'P') {
            const a = clamp((t - S.t0 - 1.0) / 0.5, 0, 1);
            if (a <= 0) return;
            g.globalAlpha = a; hatch(g, dx, yd, dw, Math.max(2, hd), C.muted); g.globalAlpha = 1;
            return;
          }
          const pp = V.ease.inOut(clamp((t - S.t0 - (i / n) * 0.6) / 1.1, 0, 1));
          const ys = top + pc.src * sc, hs = pc.len * sc;
          const x = lerp(sx, dx, pp), y = lerp(ys, yd, pp), ww = lerp(sw, dw, pp), hh = Math.max(2, lerp(hs, hd, pp));
          g.fillStyle = V.alpha(colOf(pc.part), 0.8); g.fillRect(x, y, ww, hh);
          if (pp > 0.98 && hd > 12) V.text(g, `${pc.part}·c${pc.c}`, x + 6, y + Math.min(hd, 30) / 2 + 4, { size: 11, font: 'mono', color: C.ink });
        });
        // grade de 128 linhas no destino
        for (let r = 0; r < S.tp; r++) {
          const ry = top + r * (rh + gapR);
          for (let b = 1; b < nb; b++) V.line(g, dx + dw - 7, ry + b * 128 * rsc, dx + dw, ry + b * 128 * rsc, { color: V.alpha(C.ink, 0.45), lw: 0.8 });
        }
        // zoom na cauda K/V do rank 0: de que bloco do disco vem a escala de cada bloco
        const r0 = G.q * G.per, rows = G.rowsR - r0;
        let ix, iw, iy, ih;
        if (narrow) { ix = 46; iw = w - ix - 14; iy = top + HH + 12 + 26; ih = zh - 26 - 58; }
        else { ix = sx + sw + 40; iw = dx - ix - 46; iy = top + 22; ih = Math.min(HH - 78, rows * 0.36); }
        const ry0 = top + r0 * rsc;
        if (iw > 120) {
          const al = clamp((t - S.t0 - 1.3) / 0.5, 0, 1), isc = ih / rows;
          g.globalAlpha = al;
          V.text(g, `zoom · rank 0, linhas ${nf(r0)}–${nf(r0 + rows - 1)} (K, V${G.padN ? ' e padding' : ''})`, ix + iw / 2, iy - 10, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: iw + 30 });
          G.pieces.filter((pc) => pc.rank === 0 && pc.part !== 'Q').forEach((pc) => {
            const yy = iy + (pc.dst - r0) * isc, hh = pc.len * isc;
            if (pc.part === 'P') hatch(g, ix, yy + 0.5, iw, hh - 1, C.muted);
            else V.box(g, ix, yy + 0.5, iw, hh - 1, { r: 3, fill: V.alpha(colOf(pc.part), 0.75) });
            // rótulo no maior trecho da peça entre fronteiras de bloco (nunca em cima de uma linha tracejada)
            let s0 = pc.dst, best = [pc.dst, pc.dst];
            for (let rr = Math.floor(pc.dst / 128 + 1) * 128; ; rr += 128) { const e1 = Math.min(rr, pc.dst + pc.len); if (e1 - s0 > best[1] - best[0]) best = [s0, e1]; if (e1 >= pc.dst + pc.len) break; s0 = e1; }
            const bh2 = (best[1] - best[0]) * isc, by2 = iy + (best[0] - r0) * isc;
            if (bh2 >= 11) V.text(g, pc.part === 'P' ? `pad·c${pc.c} (${pc.len})` : `${pc.part}·c${pc.c} (${pc.len})`, ix + 8, by2 + Math.min(bh2 / 2, 10) + 4, { size: 11, font: 'mono', color: C.ink });
          });
          for (let b = Math.ceil(r0 / 128); b * 128 <= r0 + rows; b++) {
            const yy = iy + (b * 128 - r0) * isc;
            V.line(g, ix - 6, yy, ix + iw + 6, yy, { color: V.alpha(C.ink, 0.6), lw: 1, dash: [3, 3] });
          }
          for (let b = Math.floor(r0 / 128); b * 128 < r0 + rows; b++) {
            const y0 = iy + (Math.max(b * 128, r0) - r0) * isc, y1 = iy + (Math.min((b + 1) * 128, r0 + rows) - r0) * isc;
            const bad = G.confB.has('0:' + b), from = G.src['0:' + b] || [];
            V.text(g, 'b' + b, ix - 10, (y0 + y1) / 2 + 4, { size: 11, font: 'mono', align: 'right', color: bad ? C.d : C.muted, weight: bad ? 700 : 400 });
            if (from.length) V.text(g, (iw > 260 ? 'escala ← ' : '← ') + from.join('+'), ix + iw - 6, (y0 + y1) / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.ink, weight: bad ? 700 : 400 });
            if (bad) { g.strokeStyle = C.d; g.lineWidth = 2.5; V.rr(g, ix - 3, y0 + 1, iw + 6, y1 - y0 - 2, 4); g.stroke(); }
            else if (G.padN) { g.strokeStyle = V.alpha(C.e, 0.9); g.lineWidth = 1.5; V.rr(g, ix - 2, y0 + 1.5, iw + 4, y1 - y0 - 3, 4); g.stroke(); }
          }
          g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
          if (narrow) {
            g.strokeRect(dx - 3, ry0, dw + 6, top + rh - ry0);
            V.text(g, 'zoom ↓', dx - 8, ry0 + (top + rh - ry0) / 2 + 4, { size: 11, align: 'right', color: C.muted });
          } else {
            g.beginPath(); g.moveTo(ix + iw + 6, iy); g.lineTo(dx - 2, ry0); g.moveTo(ix + iw + 6, iy + ih); g.lineTo(dx - 2, top + rh); g.stroke();
          }
          g.setLineDash([]);
          const note = G.conf ? 'âmbar: o bloco junta linhas de blocos de escala diferentes; nenhuma escala do disco serve, e o loader stock requantiza'
            : G.padN ? 'verde: cada bloco reaproveita um único bloco de escala do checkpoint; a faixa hachurada é padding zero'
            : 'todas as fronteiras de 128 linhas coincidem com as da origem: uma permutação de blocos';
          wrap(g, note, ix - (narrow ? 30 : 0), iy + ih + 16, iw + (narrow ? 30 : 0), 14, { size: 11, color: G.conf ? C.d : C.e });
          g.globalAlpha = 1;
        } else {
          const ay = top + HH / 2;
          V.arrow(g, sx + sw + 12, ay, dx - 12, ay, { color: V.alpha(C.muted, 0.7), lw: 1.4, dash: [4, 4] });
        }
        // status
        let msg, col;
        if (S.tp === 4) { msg = '✓ encaixe exato: cada rank recebe um chunk inteiro, com as escalas originais'; col = C.e; }
        else if (!S.ga) { msg = S.exact ? '✓ SWA: permutação pura de blocos (as mesmas escalas, reordenadas). O cuidado aqui é outro: 2 KV heads por chunk, e o sharding precisa de ckpt_tp = 4 chunks (#57508)' : 'SWA: blocos alinhados (29 por chunk), e requantizar não muda os pesos. O problema é outro: 2 KV heads por chunk, e o sharding precisa de ckpt_tp = 4 chunks (#57508)'; col = S.exact ? C.e : C.ink; }
        else if (S.exact) { msg = '✓ exato (VLLM_MIMO_EXACT_QKV=1): cada K | V | pad começa numa fronteira de 128 e reusa os blocos 24–26 do chunk; o forward separa K e V. Custo: ~12K tokens a menos no pool [medido: diffbot]'; col = C.e; }
        else { msg = '⚠ requantizar: o loader desquantiza e requantiza as linhas do rank; 18% de Q, ~51% de K e 73–81% de V mudam (erro L2 de 1,3–2,5%) [medido: diffbot]'; col = C.d; }
        wrap(g, msg, 16, h - bot + 20, w - 32, 15, { size: 12, color: col, weight: 600 });
        const ly = h - 10, sp = narrow ? 40 : 56;
        [[C.b, 'Q'], [C.a, 'K'], [C.c, 'V']].forEach(([c, l], i) => { const lx = 16 + i * sp; V.box(g, lx, ly - 10, 11, 11, { r: 3, fill: c }); V.text(g, l, lx + 16, ly, { size: 11, color: C.muted }); });
        hatch(g, 16 + 3 * sp, ly - 10, 11, 11, C.muted); V.text(g, 'pad', 16 + 3 * sp + 16, ly, { size: 11, color: C.muted });
        const lx4 = 16 + 4 * sp + (narrow ? 4 : 10);
        V.box(g, lx4, ly - 10, 11, 11, { r: 3, fill: C.d }); V.text(g, narrow ? '>1 escala de origem' : 'bloco com >1 escala de origem', lx4 + 16, ly, { size: 11, color: C.muted, maxW: w - lx4 - 24 });
      });
    })();

    /* =====================================================================
       Figura 6 — overlay fail-closed + vazamento de sliding_window
       ===================================================================== */
    (function figOverlay() {
      const fig = root.querySelector('#fig-overlay');
      setH(fig, 700);
      const STEPS = [
        { s: 'Extrair da imagem', m: 'docker run … cat → orig/ + sha256', t: 'Extrair da imagem fixada', d: '<code>docker run --entrypoint cat $IMAGE …/mimo_v2.py &gt; files/orig/</code> e <code>sha256sum</code>: o patch parte dos bytes exatos da imagem fixada, nunca de um arquivo do <code>main</code>.' },
        { s: 'Já aplicado?', m: 'MARK in s → copia e sai', t: 'Idempotência', d: 'Se o <code>MARK</code> já está no arquivo, o script copia e sai com “já aplicado”: rodar duas vezes é seguro.' },
        { s: 'Já upstream?', m: 'regex layer_cache_config', t: 'Aposentadoria automática', d: 'Se o código já passa <code>cache_config=layer_cache_config</code>, a imagem nova trouxe a correção: “already upstream”, e o patch não faz nada.' },
        { s: 'Âncora 1 (×1)', m: 's.count(a1) == 1', t: 'Âncora 1: exatamente uma vez', d: 'Procura o início do <code>__init__</code> do decoder layer e exige <b>uma</b> ocorrência. Insere a cópia do <code>cache_config</code> com <code>sliding_window=None</code>. Se não bate, aborta com mensagem clara.' },
        { s: 'Âncora 2 (×2)', m: 's.count(…) == 2', t: 'Âncora 2: exatamente duas vezes', d: 'Exige as <b>duas</b> construções de <code>MiMoV2Attention</code> e injeta <code>cache_config=layer_cache_config</code> nas duas. Contagem diferente = imagem diferente = aborto.' },
        { s: 'Gravar + marcador', m: 'print("[overlay] …")', t: 'Gravar e marcar', d: 'Escreve <code>files/overlay/mimo_v2.py</code> e imprime <code>[overlay] cache_config encaminhado</code>: é esse texto que você procura no <code>docker logs</code>.' },
        { s: 'Montar nos 2 ranks', m: '-v overlay:…/mimo_v2.py:ro', t: 'Bind-mount nos dois ranks', d: '<code>-v …/overlay/mimo_v2.py:…/models/mimo_v2.py:ro</code> no worker e no head, com os mesmos bytes. Nada de rebuild da imagem.' },
        { s: 'Provar', m: 'KV 2× · needle 128K', t: 'Provar', d: '<code>GPU KV cache size</code> ~2× com fp8 (junto com o overlay 2, o porte do #58128) e needle de 128K a 100%. Sem prova, o patch não entra.' },
      ];
      const STATES = [
        { label: 'upstream', leak: false, boot: true, kv: 'BF16: o fp8 é ignorado sem aviso', pool: '1.365.757 tokens (1×)', needle: 'ok', verdict: 'silencioso: você acredita que tem FP8', col: 'd' },
        { label: 'ingênuo', leak: true, boot: true, kv: 'FP8 (com o backend portado)', pool: '2.731.530 tokens (2×)', needle: 'fail', verdict: 'bug silencioso de qualidade: as globais veem só a janela de 128', col: 'red' },
        { label: 'overlay 1', leak: false, boot: false, kv: '— (o boot não chega lá)', pool: '—', needle: 'na', verdict: 'erro de startup: TRITON_ATTN_DIFFKV rejeita FP8. Falha alto: o comportamento certo', col: 'd' },
        { label: 'overlays 1+2', leak: false, boot: true, kv: 'FP8', pool: '2.731.530 tokens (2×)', needle: 'ok', verdict: 'estado bom: FP8 efetivo e globais continuam globais', col: 'a' },
      ];
      const S = { st: 3, changed: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Estado', options: STATES.map((s, i) => ({ v: i, label: s.label })), value: 3, onChange: (v) => { S.st = v; } });
      UI.toggle(ctl, { label: 'imagem mudou (âncora não bate)', value: false, onChange: (v) => { S.changed = v; } });
      const stp = UI.stepper(fig, STEPS.map((s) => ({ t: s.t, d: s.d })), { autoplay: 5200 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), wide = w >= 640;
        const abortAt = S.changed ? 3 : 99;
        // ---------- pipeline ----------
        const px = 14, py = 14, pw = wide ? w * 0.44 : w - 28, ph = wide ? h - 64 : 250;
        const cols = wide ? 1 : 2, rows = 8 / cols, gap = 7;
        const bw = (pw - gap * (cols - 1)) / cols, bh = Math.min(46, (ph - gap * (rows - 1)) / rows);
        const bpos = (k) => { const c = wide ? 0 : Math.floor(k / 4), r = wide ? k : k % 4; return [px + c * (bw + gap), py + r * (bh + gap)]; };
        STEPS.forEach((s, k) => {
          const [bx, by] = bpos(k);
          let status = k < stp.i ? 'done' : k === stp.i ? 'on' : 'todo';
          if (stp.i >= abortAt && k === abortAt) status = 'abort';
          else if (stp.i >= abortAt && k > abortAt) status = 'skip';
          const col = status === 'abort' ? C.red : C.a;
          V.box(g, bx, by, bw, bh, { r: 8, fill: status === 'on' ? V.alpha(col, 0.3) : status === 'done' ? V.alpha(col, 0.13) : status === 'abort' ? V.alpha(C.red, 0.28) : V.alpha(C.muted, 0.06), stroke: status === 'on' || status === 'abort' ? col : V.alpha(C.muted, 0.3), lw: status === 'on' || status === 'abort' ? 2 : 1 });
          V.text(g, `${k + 1}. ${s.s}`, bx + 10, by + (bh > 36 ? 18 : bh / 2 + 4), { size: 12, weight: 600, color: status === 'skip' ? V.alpha(C.muted, 0.6) : C.ink, maxW: bw - 34 });
          if (bh > 36) V.text(g, status === 'skip' ? '— não executado' : s.m, bx + 10, by + 34, { size: 11, font: 'mono', color: C.muted, maxW: bw - 16 });
          if (status === 'done') V.text(g, '✓', bx + bw - 12, by + 18, { size: 13, align: 'center', color: C.a, weight: 700 });
          if (status === 'abort') V.text(g, '✕', bx + bw - 12, by + 18, { size: 13, align: 'center', color: C.red, weight: 700 });
        });
        // arquivo viajando
        const e = clamp(stp.elapsed() / 0.7, 0, 1), k1 = stp.i, k0 = Math.max(0, k1 - 1);
        const [ax, ay] = bpos(k0), [bx2, by2] = bpos(k1);
        const fx = lerp(ax, bx2, V.ease.out(e)) + bw - 26, fy = lerp(ay, by2, V.ease.out(e)) + bh / 2 - 9;
        const aborted = stp.i >= abortAt;
        V.box(g, fx - 8, fy, 16, 20, { r: 3, fill: aborted ? C.red : C.d });
        V.text(g, '.py', fx, fy + 14, { size: 11, font: 'mono', align: 'center', color: onInk(C), weight: 700 });
        if (aborted && wide) wrap(g, 'sys.exit: “âncora 1 não encontrada exatamente 1x — a imagem mudou; revise o patch”. Nada é gravado nem montado.', px, py + 8 * (bh + gap) + 8, pw, 15, { size: 11.5, color: C.red, weight: 600 });
        // ---------- máscara ----------
        const st = STATES[S.st];
        const mx0 = wide ? w * 0.5 : 14, my0 = wide ? 30 : py + ph + (aborted ? 40 : 18), mw = wide ? w * 0.5 - 16 : w - 28;
        if (aborted && !wide) wrap(g, 'sys.exit: âncora 1 não bate. Nada é gravado nem montado.', 14, py + ph + 14, w - 28, 15, { size: 11.5, color: C.red, weight: 600 });
        const NN = 20, WIN = 5;
        const cell = Math.min(15, (mw - 34) / NN, (h - my0 - 120) / NN);
        const gx = mx0 + 26, gy = my0 + 14;
        V.text(g, 'camada global · 20 tokens · janela desenhada como 5', mx0, my0, { size: 11.5, color: C.muted, maxW: mw });
        const dim = !st.boot;
        for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) {
          const visible = j <= i && (!st.leak || i - j < WIN);
          let fill = visible ? V.alpha(C.a, 0.25 + 0.35 * (j === i ? 1 : 0.5)) : V.alpha(C.muted, 0.07);
          if (j === 1 && visible) fill = V.alpha(C.d, 0.6);
          g.fillStyle = dim ? V.alpha(C.muted, visible ? 0.16 : 0.05) : fill;
          g.fillRect(gx + j * cell + 0.5, gy + i * cell + 0.5, cell - 1, cell - 1);
        }
        V.text(g, 'q', gx - 8, gy + 4 + cell * 0.6, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, 'k →', gx + NN * cell, gy - 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, '✦', gx + 1 * cell + cell / 2, gy - 3, { size: 11, align: 'center', color: C.d });
        // linha da pergunta
        g.strokeStyle = C.ink; g.lineWidth = 1.5; g.strokeRect(gx, gy + 19 * cell, NN * cell, cell);
        const qc = [gx + 1 * cell + cell / 2, gy + 19 * cell + cell / 2];
        if (!dim) {
          if (!st.leak) { V.glow(g, qc[0], qc[1], 14 + 3 * Math.sin(t * 4), C.d, 1); V.text(g, '✦', qc[0], qc[1] + 4, { size: 11, align: 'center', color: C.ink, weight: 700 }); }
          else V.text(g, '✕', qc[0], qc[1] + 4.5, { size: 13, align: 'center', color: C.red, weight: 700 });
        } else {
          g.save(); g.translate(gx + NN * cell / 2, gy + NN * cell / 2); g.rotate(-0.18);
          V.box(g, -86, -16, 172, 32, { r: 8, fill: V.alpha(C.red, 0.2), stroke: C.red, lw: 2 });
          V.text(g, '✕ erro de startup', 0, 5, { size: 14, align: 'center', color: C.red, weight: 700 });
          g.restore();
        }
        // leitura do servidor
        let yy = gy + NN * cell + 22;
        const lw2 = mw;
        const colV = C[st.col] || C.ink;
        yy = wrap(g, 'KV efetivo: ' + st.kv, mx0, yy, lw2, 15, { size: 12, color: C.ink });
        yy = wrap(g, 'pool em 17 GiB [medido, #58128]: ' + st.pool, mx0, yy, lw2, 15, { size: 12, color: C.ink });
        yy = wrap(g, 'needle 128K: ' + (st.needle === 'ok' ? '✓ encontra a agulha' : st.needle === 'fail' ? '✕ falha acima de ~128 tokens' : '— (o servidor não sobe)'), mx0, yy, lw2, 15, { size: 12, color: st.needle === 'fail' ? C.red : C.ink });
        wrap(g, '→ ' + st.verdict, mx0, yy + 2, lw2, 15, { size: 12, color: colV, weight: 600 });
      });
    })();

    /* =====================================================================
       Figura 7 — boot em dois nós: rendezvous, grupos de comunicação, flags
       ===================================================================== */
    (function figBoot() {
      const fig = root.querySelector('#fig-boot');
      setH(fig, 560);
      const PH = [
        { t: 'Worker primeiro', flags: ['--node-rank 1', '--headless', '--nnodes 2', '--master-addr $HEAD_IP', '--master-port 50000', 'VLLM_HOST_IP=$WORKER_IP'], d: 'O worker sobe com <code>--node-rank 1 --headless</code>: um único processo para a sua GPU, sem API nem EngineCore. Ele procura o <code>--master-addr</code>:<code>--master-port</code> do head e espera. É a ordem das três receitas: worker, depois head.' },
        { t: 'Head e rendezvous', flags: ['--node-rank 0', '--distributed-executor-backend mp', '--tensor-parallel-size 2', '--host 0.0.0.0 --port 8888', 'VLLM_HOST_IP=$HEAD_IP'], d: 'O head sobe com <code>--node-rank 0</code>: API server, EngineCore (scheduler e gerente de KV) e o processo da GPU 0. Os dois processos de GPU se encontram no endereço combinado. Esse é o <i>rendezvous</i>, e agora o grupo tem <i>world size</i> 2.' },
        { t: 'Grupos de comunicação', flags: ['GLOO_SOCKET_IFNAME', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA==rocep1s0f0', 'NCCL_IB_GID_INDEX=3', '--network host', '--device /dev/infiniband', '--ulimit memlock=-1'], d: 'Formam-se dois grupos: <b>Gloo</b>, na CPU, para metadados e barreiras, e <b>NCCL</b>, para os tensores. O bootstrap vai por TCP na interface CX7; os dados do NCCL vão por RDMA no HCA escolhido. Confira no log: <code>NET/IB</code>, e não <code>NET/Socket</code>.' },
        { t: 'Pesos e overlays', flags: ['-v $MODEL_DIR:ro', '--load-format safetensors', '--safetensors-load-strategy lazy', '-v overlay/*.py:ro', 'HF_HUB_OFFLINE=1', 'VLLM_ENGINE_READY_TIMEOUT_S=3600'], d: 'Cada rank lê a sua fatia dos pesos do mesmo diretório read-only, com os mesmos overlays montados; os marcadores <code>[overlay] …</code> aparecem nos dois logs. O boot frio leva ~11 min <b>[medido: tonyd2wild]</b>, por isso o timeout de 3600 s.' },
        { t: 'Memória e graphs', flags: ['--gpu-memory-utilization 0.85', '--kv-cache-dtype fp8', '--compilation-config FULL_DECODE_ONLY', '--cudagraph-capture-sizes 8 … 64'], d: 'Cada rank mede a memória, reserva o pool de KV e captura os CUDA graphs das capture sizes. O log mostra <code>GPU KV cache size</code> e as capture sizes: é o critério de saída do Passo 5.' },
        { t: 'Servindo', flags: ['--ipc host --shm-size 16g', '--max-num-seqs 8', '--speculative-config', '--no-async-scheduling'], d: 'A cada passo, o EngineCore agenda e manda o <code>SchedulerOutput</code> aos workers (shared memory no head, ZMQ para o worker remoto). Os dois fazem o forward com 96 all-reduces pelo RDMA, e os tokens voltam pela API na porta 8888.' },
      ];
      const S = { bad: false };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'HCA ou GID errados (NCCL cai para NET/Socket)', value: false, onChange: (v) => { S.bad = v; } });
      const rT = UI.readout(ctl, 'Transporte do NCCL', 'NET/IB');
      const st = UI.stepper(fig, PH.map((p) => ({ t: p.t, d: p.d })), { autoplay: 6500 });
      const chip = (g, x, y, s, col, on) => {
        g.font = V.font(11, 'mono', 500);
        const ww = g.measureText(s).width + 14;
        V.box(g, x, y, ww, 20, { r: 6, fill: V.alpha(col, on ? 0.2 : 0.06), stroke: V.alpha(col, on ? 0.9 : 0.35), lw: 1 });
        V.text(g, s, x + 7, y + 14, { size: 11, font: 'mono', color: C_.ink, weight: 500 });
        return ww;
      };
      let C_ = V.C();
      V.fig(fig, (g, w, h, t) => {
        const C = (C_ = V.C()), ph = st.i, e = st.elapsed(), wide = w >= 640;
        rT.set(S.bad && ph >= 2 ? 'NET/Socket ✕' : ph >= 2 ? 'NET/IB ✓' : '—');
        // ---------- geometria dos nós ----------
        const nodeH = wide ? 212 : 196, rowH = 40;
        let A, B;
        if (wide) { const bw = Math.min(330, w * 0.36); A = { x: 16, y: 34, w: bw, h: nodeH }; B = { x: w - 16 - bw, y: 34, w: bw, h: nodeH }; }
        else { A = { x: 16, y: 30, w: w - 32, h: nodeH }; B = { x: 16, y: 30 + nodeH + 64, w: w - 32, h: 112 }; }
        const rowY = (N, i) => N.y + 30 + i * rowH;
        const cx = (N) => N.x + 12, cw = (N) => N.w - 24;
        const up0 = ph >= 1, up1 = true;
        const nodeBox = (N, title, up, col) => {
          V.box(g, N.x, N.y, N.w, N.h, { r: 12, fill: V.alpha(col, up ? 0.07 : 0.02), stroke: V.alpha(col, up ? 0.8 : 0.3), lw: up ? 1.6 : 1 });
          if (!up) { g.save(); g.setLineDash([4, 4]); V.box(g, N.x, N.y, N.w, N.h, { r: 12, stroke: V.alpha(C.muted, 0.5), lw: 1 }); g.restore(); }
          V.text(g, title, N.x + 12, N.y + 18, { size: 11.5, font: 'mono', weight: 600, color: up ? C.ink : C.muted, maxW: N.w - 24 });
        };
        nodeBox(A, 'head · rank 0 · 10.0.0.1', up0, C.b);
        nodeBox(B, 'worker · rank 1 · 10.0.0.2', up1, C.b);
        const proc = (N, i, label, col, on, appear = 1) => {
          if (appear <= 0) return null;
          const x = cx(N), y = rowY(N, i), ww = cw(N);
          g.globalAlpha = appear;
          V.box(g, x, y, ww, 30, { r: 8, fill: V.alpha(col, on ? 0.3 : 0.12), stroke: V.alpha(col, on ? 1 : 0.5), lw: on ? 1.8 : 1 });
          V.text(g, label, x + 10, y + 19.5, { size: 11.5, weight: 600, color: C.ink, maxW: ww - 20 });
          g.globalAlpha = 1;
          return { x, y, w: ww, h: 30 };
        };
        const ap = (k) => (ph > 1 ? 1 : ph === 1 ? clamp((e - k * 0.35) / 0.4, 0, 1) : 0);
        const api = proc(A, 0, 'API server · :8888', C.c, ph === 5, ap(0)) || { x: cx(A), y: rowY(A, 0), w: cw(A), h: 30 };
        const core = proc(A, 1, 'EngineCore · scheduler · KV', C.b, ph === 5 || ph === 4, ap(1)) || { x: cx(A), y: rowY(A, 1), w: cw(A), h: 30 };
        const w0 = proc(A, 2, 'processo da GPU 0', C.a, ph >= 3, ap(2)) || { x: cx(A), y: rowY(A, 2), w: cw(A), h: 30 };
        let w1;
        if (wide) {
          [0, 1].forEach((i) => { const x = cx(B), y = rowY(B, i); g.save(); g.setLineDash([3, 4]); V.box(g, x, y, cw(B), 30, { r: 8, stroke: V.alpha(C.muted, 0.45), lw: 1 }); g.restore(); V.text(g, i ? 'sem EngineCore' : 'sem API (--headless)', x + 10, y + 19.5, { size: 11, color: C.muted }); });
          w1 = proc(B, 2, 'processo da GPU 1', C.a, ph === 0 || ph >= 3);
        } else {
          w1 = proc(B, 0, 'processo da GPU 1 (--headless: sem API)', C.a, ph === 0 || ph >= 3);
        }
        // espera do worker na fase 0
        if (ph === 0) {
          const dots = '.'.repeat(1 + Math.floor(t * 2.5) % 3);
          V.text(g, 'aguardando 10.0.0.1:50000' + dots, w1.x + 10, w1.y + 48, { size: 11.5, font: 'mono', color: C.d, weight: 600, maxW: w1.w - 12 });
          V.text(g, 'head ainda não subiu', A.x + A.w / 2, A.y + A.h / 2, { size: 12, align: 'center', color: C.muted });
        }
        // barras de memória (fases 3+)
        const barY = (N) => N.y + N.h - 30;
        const memBar = (N, prog) => {
          const x = cx(N), y = barY(N), ww = cw(N), vis = 121.69;
          V.box(g, x, y, ww, 16, { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          const pw = ww * 82.7 / vis * prog;
          V.box(g, x, y, Math.max(1, pw), 16, { r: 4, fill: V.alpha(C.b, 0.7) });
          if (ph >= 4) {
            const k = clamp(ph === 4 ? e / 1.2 : 1, 0, 1);
            V.box(g, x + ww * 82.7 / vis, y, ww * 4 / vis * k, 16, { r: 2, fill: V.alpha(C.d, 0.75) });
            V.box(g, x + ww * 86.7 / vis, y, ww * 16.7 / vis * k, 16, { r: 2, fill: V.alpha(C.a, 0.75) });
          }
          const lab = ph >= 4 ? 'pesos 82,7 · overhead · KV ~16,7 GiB [estimativa]' : `pesos do rank: ${nf(82.7 * prog, 1)} de 82,7 GiB`;
          V.text(g, lab, x, y - 5, { size: 11, color: C.muted, maxW: ww });
        };
        if (ph >= 3) {
          const prog = ph === 3 ? clamp(e / 4, 0, 1) : 1;
          memBar(A, prog); memBar(B, prog);
          if (prog >= 1 || ph > 3) {
            [A, B].forEach((N) => { V.text(g, '[overlay] ✓', N.x + N.w - 12, N.y + 18, { size: 11, font: 'mono', align: 'right', color: C.e, weight: 600 }); });
          }
        }
        // ---------- ligações ----------
        let tcp, rdma;
        if (wide) {
          tcp = [A.x + A.w, core.y + 15, B.x, core.y + 15];
          rdma = [A.x + A.w, w0.y + 15, B.x, w0.y + 15];
          V.line(g, w0.x + w0.w, w0.y + 15, A.x + A.w, w0.y + 15, { color: V.alpha(C.a, 0.5), lw: 2 });
          V.line(g, B.x, w1.y + 15, w1.x, w1.y + 15, { color: V.alpha(C.a, 0.5), lw: 2 });
          V.line(g, core.x + core.w, core.y + 15, A.x + A.w, core.y + 15, { color: V.alpha(C.muted, 0.5), lw: 1 });
        } else {
          const xt = A.x + A.w * 0.28, xr = A.x + A.w * 0.72;
          tcp = [xt, A.y + A.h, xt, B.y];
          rdma = [xr, A.y + A.h, xr, B.y];
        }
        const linkUp = ph >= 1;
        const rdmaOk = ph >= 2 && !S.bad;
        V.line(g, tcp[0], tcp[1], tcp[2], tcp[3], { color: V.alpha(C.ink, linkUp ? 0.55 : 0.2), lw: 1.4, dash: [5, 4], dashOffset: -t * 20 });
        V.line(g, rdma[0], rdma[1], rdma[2], rdma[3], { color: ph < 2 ? V.alpha(C.muted, 0.2) : S.bad ? V.alpha(C.red, 0.7) : V.alpha(C.a, 0.85), lw: ph < 2 ? 2 : 6, dash: S.bad && ph >= 2 ? [8, 6] : null });
        const midT = [(tcp[0] + tcp[2]) / 2, (tcp[1] + tcp[3]) / 2], midR = [(rdma[0] + rdma[2]) / 2, (rdma[1] + rdma[3]) / 2];
        const lab = (p, s, col, dy) => { if (wide) V.text(g, s, p[0], p[1] + dy, { size: 11, align: 'center', color: col, maxW: B.x - A.x - A.w - 8 }); else V.text(g, s, p[0] + 10, p[1] + 4 + dy, { size: 11, color: col, maxW: w - p[0] - 20 }); };
        lab(midT, wide ? 'TCP · IFACE (rendezvous, Gloo, fila ZMQ)' : 'TCP · controle', C.muted, wide ? -8 : -8);
        lab(midR, ph < 2 ? (wide ? 'RDMA (ainda sem grupo NCCL)' : 'RDMA (sem NCCL)') : S.bad ? (wide ? '✕ NET/Socket: all-reduce por TCP' : '✕ NET/Socket') : (wide ? 'RoCE 200 Gb/s · NCCL NET/IB' : 'RoCE · NET/IB'), ph >= 2 && S.bad ? C.red : ph >= 2 ? C.a : C.muted, wide ? 18 : 10);
        // pacotes
        const pk = (L, u, col, r = 4.5) => { const [x, y] = V.along(L[0], L[1], L[2], L[3], u); V.glow(g, x, y, 12, col, 0.8); V.dot(g, x, y, r, col); };
        if (ph === 1) { const u = clamp((e - 1.2) / 1.0, 0, 1); if (u > 0 && u < 1) pk([tcp[2], tcp[3], tcp[0], tcp[1]], u, C.d); if (u >= 1) lab(midT, wide ? 'rendezvous ✓ · world size 2' : 'rendezvous ✓', C.d, wide ? 10 : 8); }
        if (ph === 2) { const u = (e * 0.9) % 1; pk(tcp, u, C.b); pk([tcp[2], tcp[3], tcp[0], tcp[1]], u, C.b); if (!S.bad && e > 1) pk(rdma, (e * 1.4) % 1, C.a, 5); }
        if (ph === 5) {
          const u = (t % 3.2) / 3.2;
          const seg = (a, b) => clamp((u - a) / (b - a), 0, 1);
          const cxp = (R) => [R.x + R.w / 2, R.y + 15];
          const [ax, ay] = cxp(api), [kx, ky] = cxp(core), [wx, wy] = cxp(w0);
          if (u < 0.15) { const v = seg(0, 0.15); V.dot(g, lerp(ax, kx, v), lerp(ay, ky, v), 5, C.c); }
          else if (u < 0.35) { const v = seg(0.15, 0.35); V.dot(g, lerp(kx, wx, v), lerp(ky, wy, v), 4.5, C.b); pk(tcp, v, C.b); }
          else if (u < 0.8) { const v = seg(0.35, 0.8), L = S.bad ? tcp : rdma, reps = S.bad ? 1 : 3, f = (v * reps) % 1; pk(f < 0.5 ? L : [L[2], L[3], L[0], L[1]], (f % 0.5) * 2, S.bad ? C.red : C.a, 5); }
          else { const v = seg(0.8, 1); V.dot(g, lerp(wx, ax, v), lerp(wy, ay, v), 5, C.c); if (v > 0.9) V.text(g, 'tokens →', api.x + api.w - 10, api.y + 19.5, { size: 11, align: 'right', color: C.c, weight: 700 }); }
        }
        // ---------- chips de flags da fase ----------
        const fy0 = (wide ? A.y + A.h : B.y + B.h) + 22;
        V.text(g, `fase ${ph + 1} de ${PH.length} · ${PH[ph].t}: flags e variáveis em jogo`, 16, fy0, { size: 11.5, weight: 600, color: C.ink, maxW: w - 32 });
        let x = 16, y = fy0 + 10;
        PH[ph].flags.forEach((f, i) => {
          g.font = V.font(11, 'mono', 500);
          const ww = g.measureText(f).width + 14;
          if (x + ww > w - 16) { x = 16; y += 26; }
          const on = clamp((e - i * 0.12) / 0.3, 0, 1);
          g.globalAlpha = 0.25 + 0.75 * on;
          chip(g, x, y, f, [C.b, C.b, C.a, C.a, C.c, C.c][ph], true);
          g.globalAlpha = 1;
          x += ww + 8;
        });
      });
    })();

    /* =====================================================================
       Figura 8 — simulador de especulação
       ===================================================================== */
    (function figSpec() {
      const fig = root.querySelector('#fig-spec');
      setH(fig, 560);
      const CATS = [
        { n: 'estruturado', a: 6.85 }, { n: 'código', a: 5.23, tps: 70.5 }, { n: 'matemática', a: 5.0 },
        { n: 'JSON', a: 4.0 }, { n: 'raciocínio', a: 2.62 }, { n: 'prosa', a: 1.23, tps: 25.9 },
      ];
      const alphaFor = (a) => { let lo = 0, hi = 0.999999; for (let it = 0; it < 60; it++) { const m = (lo + hi) / 2; let s = 0, p = 1; for (let i = 1; i <= 7; i++) { p *= m; s += p; } if (s < a) lo = m; else hi = m; } return (lo + hi) / 2; };
      CATS.forEach((c) => { c.al = alphaFor(c.a); });
      const T7 = 0.087, BYTES = 6.39, BW = 273;
      const S = { cat: 1, k: 7, eff: 0.58, cd: 0, cap: 'tut', round: -1, acc: 0, tok: 0, steps: 0 };
      const rng = V.rng(2026);
      // t(k) = t0 + c_d·[k>0] + β·k, calibrado para passar por t(7) = T7 (hipótese: c_d não foi medido)
      const model = () => {
        const t0 = BYTES / (BW * S.eff), cd = Math.min(S.cd / 1000, Math.max(0, T7 - t0)), beta = (T7 - t0 - cd) / 7, al = CATS[S.cat].al, arr = [];
        for (let k = 0; k <= 7; k++) { let a = 0, p = 1; for (let i = 1; i <= k; i++) { p *= al; a += p; } const tt = t0 + (k ? cd : 0) + beta * k; arr.push({ k, a, tt, tps: (1 + a) / tt }); }
        let best = 0; arr.forEach((x, i) => { if (x.tps > arr[best].tps) best = i; });
        return { t0, cd, beta, al, arr, best, clamped: cd < S.cd / 1000 - 1e-9 };
      };
      const ctl = UI.controls(fig);
      const reset = () => { S.tok = 0; S.steps = 0; S.round = -1; upd(); };
      UI.select(ctl, { label: 'Categoria', options: CATS.map((c, i) => ({ v: String(i), label: `${c.n} (${nf(c.a, 2)} de 7)` })), value: '1', onChange: (v) => { S.cat = +v; reset(); } });
      UI.slider(ctl, { label: 'k', min: 0, max: 7, step: 1, value: 7, fmt: (v) => (v ? String(v) : '0 (sem spec)'), onInput: (v) => { S.k = v; reset(); } });
      UI.slider(ctl, { label: 'Eficiência sem spec', min: 0.45, max: 0.75, step: 0.01, value: 0.58, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.eff = v; upd(); } });
      UI.slider(ctl, { label: 'Drafter c_d', min: 0, max: 40, step: 1, value: 0, fmt: (v) => v + ' ms', onInput: (v) => { S.cd = v; upd(); } });
      UI.seg(ctl, { label: 'Capture sizes', options: [{ v: 'tut', label: '8 16 … 64' }, { v: 'mult', label: 'múltiplos de k+1' }], value: 'tut', onChange: (v) => { S.cap = v; } });
      const rA = UI.readout(ctl, 'α ajustado', '—'), rT = UI.readout(ctl, 'Tokens/passo', '—'), rP = UI.readout(ctl, 'Previsto', '—'), rB = UI.readout(ctl, 'Melhor k', '—');
      function upd() { const M = model(); S.M = M; const x = M.arr[S.k]; rA.set(M.al.toFixed(3)); rT.set(nf(1 + x.a, 2)); rP.set(nf(x.tps, 1) + ' tok/s'); rB.set(`k=${M.best} (${nf(M.arr[M.best].tps, 1)})`); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), M = S.M, k = S.k, narrow = w < 560;
        // ---------- palco: rascunhar e verificar ----------
        const P = 1.3 + 0.2 * k, ri = Math.floor(t / P), f = (t % P) / P;
        if (ri !== S.round) {
          S.round = ri; let a = 0; while (a < k && rng() < M.al) a++; S.acc = a;
          if (ri > 0) { S.tok += a + 1; S.steps += 1; }
        }
        const acc = S.acc;
        const dX = 34, dY = 72, vX = w - 40, sz = 40;
        // rascunhador
        if (k > 0) {
          V.box(g, dX - sz / 2, dY - sz / 2, sz, sz, { r: 12, fill: V.alpha(C.b, 0.55), stroke: C.b, lw: 2 });
          g.fillStyle = C.ink; g.beginPath(); g.arc(dX - 7, dY - 4, 3, 0, 7); g.arc(dX + 7, dY - 4, 3, 0, 7); g.fill();
          V.text(g, 'rascunhador', dX - sz / 2, dY + sz / 2 + 15, { size: 11, color: C.muted });
        }
        // alvo
        const vs = 50;
        V.box(g, vX - vs / 2, dY - vs / 2, vs, vs, { r: 14, fill: V.alpha(C.a, 0.5), stroke: C.a, lw: 2 });
        g.fillStyle = C.ink; g.beginPath(); g.arc(vX - 9, dY - 6, 3.5, 0, 7); g.arc(vX + 9, dY - 6, 3.5, 0, 7); g.fill();
        g.strokeStyle = C.ink; g.lineWidth = 2; g.beginPath(); g.arc(vX, dY + 4, 8, 0.2, Math.PI - 0.2); g.stroke();
        V.text(g, 'alvo (MiMo)', vX + vs / 2, dY + vs / 2 + 15, { size: 11, color: C.muted, align: 'right' });
        // cartas
        const nC = Math.max(1, k + 1), x0 = dX + 34, x1 = vX - 40, cw = Math.min(34, (x1 - x0) / nC - 5), cy = dY - 18;
        for (let i = 0; i < k; i++) {
          const appear = f > (i / Math.max(1, k)) * 0.28;
          if (!appear) continue;
          const fly = clamp((f - (i / k) * 0.28) / 0.08, 0, 1);
          const cx = lerp(dX, x0 + i * (cw + 5), V.ease.out(fly));
          const stampT = 0.32 + ((i + 1) / (k + 1)) * 0.42;
          let fill = V.alpha(C.b, 0.35), mark = '', mcol = C.ink;
          if (f > stampT) {
            if (i < acc) { fill = V.alpha(C.a, 0.65); mark = '✓'; }
            else if (i === acc) { fill = V.alpha(C.red, 0.55); mark = '✕'; mcol = C.red; }
            else { fill = V.alpha(C.muted, 0.12); mark = ''; }
          }
          V.box(g, cx, cy, cw, 36, { r: 6, fill, stroke: V.alpha(C.ink, 0.25), lw: 1 });
          V.text(g, mark || 't' + (i + 1), cx + cw / 2, cy + 22, { size: mark ? 14 : 11, align: 'center', color: mark ? mcol : C.ink, weight: mark ? 700 : 400, font: mark ? 'sans' : 'mono' });
        }
        // token do alvo (correção ou bônus)
        if (f > 0.78 || k === 0) {
          const pos = k === 0 ? 0 : Math.min(acc, k);
          const bxx = x0 + pos * (cw + 5), by = k === 0 ? cy : cy + 40;
          const pop = V.ease.back(clamp((f - (k === 0 ? 0.3 : 0.78)) / 0.12, 0, 1));
          V.box(g, bxx, by, cw, 36 * pop, { r: 6, fill: V.alpha(C.c, 0.7), stroke: C.c, lw: 1.5 });
          if (pop > 0.8) V.text(g, acc === k && k > 0 ? 'bônus' : 'alvo', bxx + cw / 2, by + 22, { size: 11, align: 'center', color: C.ink, maxW: cw - 2 });
        }
        const avg = S.steps ? S.tok / S.steps : 0;
        V.text(g, `neste passo: ${acc} ${acc === 1 ? 'aceito' : 'aceitos'} + 1 do alvo = ${acc + 1} ${acc ? 'tokens' : 'token'}`, 16, 16, { size: 12, color: C.ink, weight: 600, maxW: w - 32 });
        V.text(g, `média simulada: ${S.steps ? nf(avg, 2) : '—'} tokens/passo · esperado ${nf(1 + M.arr[k].a, 2)} (α = ${M.al.toFixed(3)})`, 16, 32, { size: 11.5, color: C.muted, maxW: w - 32 });
        // ---------- gráfico: tok/s × k ----------
        const cy0 = narrow ? 186 : 168, chH = h - cy0 - (narrow ? 150 : 134), cx0 = 44, cwAll = w - cx0 - 16;
        const meas = CATS[S.cat].tps;
        const mx = Math.max(...M.arr.map((x) => x.tps), meas || 0) * 1.22;
        const hyp = `hipótese · c_d = ${Math.round(M.cd * 1000)} ms, β = ${nf(M.beta * 1000, 1)} ms/token` + (M.clamped ? ' (c_d limitado)' : '');
        if (narrow) { V.text(g, `tok/s previsto (C=1) · ${CATS[S.cat].n}`, cx0, cy0 - 38, { size: 11.5, color: C.muted, maxW: cwAll }); V.text(g, hyp, cx0, cy0 - 22, { size: 11, color: C.muted, maxW: cwAll }); if (meas) V.text(g, `○ medido em k=7: ${nf(meas, 1)} [tonyd2wild]`, cx0, cy0 - 6, { size: 11, color: C.d, maxW: cwAll }); }
        else V.text(g, `tok/s previsto por stream (C=1) · ${CATS[S.cat].n} · ${hyp}` + (meas ? ` · ○ medido em k=7: ${nf(meas, 1)} [tonyd2wild]` : ''), cx0, cy0 - 6, { size: 11.5, color: C.muted, maxW: cwAll });
        V.line(g, cx0, cy0 + chH, cx0 + cwAll, cy0 + chH, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const bw = cwAll / 8 - 8;
        M.arr.forEach((x, i) => {
          const bh = (x.tps / mx) * (chH - 16), bx = cx0 + i * (bw + 8) + 4, by = cy0 + chH - bh;
          const col = i === k ? C.c : i === M.best ? C.a : V.alpha(C.a, 0.4);
          V.box(g, bx, by, bw, bh, { r: 4, fill: col });
          V.text(g, nf(x.tps, 1), bx + bw / 2, by - 5, { size: 11, align: 'center', color: C.ink, font: 'mono' });
          V.text(g, i === 0 ? 'k=0' : 'k=' + i, bx + bw / 2, cy0 + chH + 14, { size: 11, align: 'center', color: i === k ? C.ink : C.muted, font: 'mono' });
          if (i === M.best) V.text(g, '★', bx + bw / 2, by - 19, { size: 13, align: 'center', color: C.a });
          if (i === 7 && meas) {
            const my = cy0 + chH - (meas / mx) * (chH - 16);
            g.strokeStyle = C.d; g.lineWidth = 2.5; g.beginPath(); g.arc(bx + 10, my, 6, 0, 7); g.stroke();
          }
        });
        // ---------- capture sizes ----------
        const sy = h - (narrow ? 100 : 88);
        const list = S.cap === 'tut' ? [8, 16, 24, 32, 40, 48, 56, 64] : [1, 2, 3, 4, 5, 6, 7, 8].map((n) => n * (k + 1));
        V.text(g, `batches de decode com k=${k}: n seq × ${k + 1} tokens · capturados: ${list.join(' ')}`, 16, sy - 10, { size: 11.5, color: C.muted, maxW: w - 32 });
        const cellW = (w - 32) / 8 - 5, maxL = Math.max(...list);
        for (let n = 1; n <= 8; n++) {
          const tk = n * (k + 1), cx = 16 + (n - 1) * (cellW + 5);
          let st, col;
          if (list.includes(tk)) { st = '✓ exato'; col = C.a; }
          else if (tk <= maxL) { const nx = list.filter((x) => x >= tk).sort((a, b) => a - b)[0]; st = `→${nx} +${Math.round((nx - tk) / tk * 100)}%`; col = C.d; }
          else { st = 'eager'; col = C.red; }
          V.box(g, cx, sy, cellW, narrow ? 52 : 44, { r: 7, fill: V.alpha(col, 0.18), stroke: col, lw: 1.3 });
          V.text(g, `${n}×${k + 1}=${tk}`, cx + cellW / 2, sy + 16, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: cellW - 4 });
          V.text(g, st, cx + cellW / 2, sy + (narrow ? 38 : 34), { size: 11, align: 'center', color: col, weight: 600, maxW: cellW - 4 });
        }
      });
    })();

    /* =====================================================================
       Figura 9 — teto de decode: esteira de bytes
       ===================================================================== */
    (function figTeto() {
      const fig = root.querySelector('#fig-teto');
      setH(fig, 410);
      const S = { fp8: false, eff: 0.58, meas: true };
      const comps = () => [
        { n: 'experts ativos', gb: 2.51, c: 'b' }, { n: 'qkv FP8', gb: 1.44, c: 'a' },
        { n: S.fp8 ? 'o_proj FP8' : 'o_proj BF16', gb: S.fp8 ? 0.805 : 1.61, c: 'c' },
        { n: 'lm_head', gb: 0.63, c: 'd' }, { n: 'router + MLP densa', gb: 0.2, c: 'e' },
      ];
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'o_proj em FP8 (#59083)', value: false, onChange: (v) => { S.fp8 = v; upd(); } });
      UI.slider(ctl, { label: 'Eficiência', min: 0.4, max: 0.8, step: 0.01, value: 0.58, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.eff = v; upd(); } });
      UI.toggle(ctl, { label: 'mostrar medições com DFlash k=7', value: true, onChange: (v) => { S.meas = v; } });
      const rB = UI.readout(ctl, 'Bytes/passo/rank', '—'), rC = UI.readout(ctl, 'Teto', '—'), rR = UI.readout(ctl, 'Realista', '—');
      function upd() { const tot = comps().reduce((s, c) => s + c.gb, 0); S.tot = tot; rB.set(nf(tot, 2) + ' GB'); rC.set(nf(273 / tot, 1) + ' tok/s'); rR.set(nf(273 * S.eff / tot, 1) + ' tok/s'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), cs = comps(), tot = S.tot, narrow = w < 560;
        const stepMs = tot / 273 * 1000, realMs = stepMs / S.eff;
        // ---------- esteira ----------
        const mbx = 12, mbw = narrow ? 70 : 92, gbw = narrow ? 56 : 70, by0 = 44, bh0 = 62;
        V.box(g, mbx, by0, mbw, bh0, { r: 10, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.5 });
        V.text(g, 'LPDDR5X', mbx + mbw / 2, by0 + 26, { size: 11.5, align: 'center', weight: 600, color: C.ink });
        V.text(g, '273 GB/s', mbx + mbw / 2, by0 + 44, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const gx = w - 12 - gbw;
        V.box(g, gx, by0, gbw, bh0, { r: 10, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 1.5 });
        V.text(g, 'GPU', gx + gbw / 2, by0 + 26, { size: 11.5, align: 'center', weight: 600, color: C.ink });
        V.text(g, '1 rank', gx + gbw / 2, by0 + 44, { size: 11, align: 'center', font: 'mono', color: C.muted });
        const bx0 = mbx + mbw + 6, bx1 = gx - 6, bl = bx1 - bx0, beltY = by0 + bh0 / 2;
        V.line(g, bx0, beltY + 16, bx1, beltY + 16, { color: V.alpha(C.muted, 0.6), lw: 2 });
        V.line(g, bx0, beltY + 16, bx1, beltY + 16, { color: V.alpha(C.ink, 0.5), lw: 2, dash: [3, 9], dashOffset: -t * 40 });
        const ppg = bl / tot * 0.9, gapPx = bl * 0.1 / cs.length;
        const Ltot = bl;
        const period = Math.max(1.6, realMs / 18);
        const off = ((t / period) % 1) * Ltot;
        g.save(); g.beginPath(); g.rect(bx0, by0 - 4, bl, bh0 + 8); g.clip();
        let acc = 0;
        cs.forEach((c) => {
          const ww = c.gb * ppg;
          for (const rep of [0, 1]) {
            const xx = bx0 + ((acc + off) % Ltot) - rep * Ltot;
            if (xx + ww < bx0 || xx > bx1) continue;
            V.box(g, xx, beltY - 12, ww, 26, { r: 5, fill: V.alpha(C[c.c], 0.75) });
            if (ww > 50) V.text(g, nf(c.gb, 2), xx + ww / 2, beltY + 5, { size: 11, align: 'center', font: 'mono', color: C.ink });
          }
          acc += ww + gapPx;
        });
        g.restore();
        V.text(g, `um passo lê ${nf(tot, 2)} GB por rank → ≥ ${nf(stepMs, 1)} ms a 100% da banda · ~${nf(realMs, 1)} ms a ${Math.round(S.eff * 100)}%`, 12, 22, { size: 12, color: C.ink, weight: 600, maxW: w - 24 });
        // ---------- barra empilhada + legenda ----------
        const sy = 140, sw = w - 24;
        let xx = 12;
        cs.forEach((c) => { const ww = c.gb / tot * sw; V.box(g, xx, sy, Math.max(1, ww - 2), 22, { r: 4, fill: V.alpha(C[c.c], 0.8) }); xx += ww; });
        let lx = 12, ly = sy + 40;
        cs.forEach((c) => {
          const lab = `${c.n} ${nf(c.gb, 2)}`;
          g.font = V.font(11.5, 'sans', 400);
          const tw = g.measureText(lab).width + 30;
          if (lx + tw > w - 8) { lx = 12; ly += 18; }
          V.box(g, lx, ly - 10, 11, 11, { r: 3, fill: C[c.c] }); V.text(g, lab, lx + 16, ly, { size: 11.5, color: C.muted }); lx += tw;
        });
        // ---------- régua de tok/s ----------
        const ry = Math.max(ly + 64, h - 92), rx0 = 24, rx1 = w - 24, maxT = 80, X = (v) => rx0 + (v / maxT) * (rx1 - rx0);
        const teto = 273 / tot;
        g.fillStyle = V.alpha(C.a, 0.16); g.fillRect(X(teto * 0.5), ry - 12, X(teto * 0.65) - X(teto * 0.5), 24);
        V.line(g, rx0, ry, rx1, ry, { color: V.alpha(C.muted, 0.7), lw: 1.5 });
        for (let v = 0; v <= maxT; v += 10) { V.line(g, X(v), ry - 4, X(v), ry + 4, { color: V.alpha(C.muted, 0.7), lw: 1 }); V.text(g, String(v), X(v), ry + 18, { size: 11, align: 'center', font: 'mono', color: C.muted }); }
        V.text(g, 'tok/s por stream (C=1)', rx1, ry + 34, { size: 11, align: 'right', color: C.muted });
        // teto e realista
        V.line(g, X(teto), ry - 22, X(teto), ry + 6, { color: C.ink, lw: 2 });
        V.text(g, `teto ${nf(teto, 1)}`, X(teto), ry - 27, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        const real = teto * S.eff;
        V.dot(g, X(real), ry, 6, C.a); V.glow(g, X(real), ry, 14 + 2 * Math.sin(t * 3), C.a, 0.8);
        V.text(g, `sem spec ~${nf(real, 1)}`, X(real), ry + 34, { size: 11.5, align: 'center', color: C.a, weight: 600 });
        if (S.meas) {
          [[25.9, 'prosa'], [53.3, 'média'], [70.5, 'código']].forEach(([v, l], i) => {
            const yy = ry - 38 - (i % 2) * 16;
            V.line(g, X(v), yy + 4, X(v), ry - 2, { color: V.alpha(C.c, 0.8), lw: 1.4, dash: [2, 3] });
            V.dot(g, X(v), ry, 4.5, C.c);
            V.text(g, `${l} ${nf(v, 1)}`, X(v), yy, { size: 11, align: 'center', color: C.c, weight: 600 });
          });
        }
      });
    })();

    /* =====================================================================
       Figura 10 — A/B/A contra a deriva
       ===================================================================== */
    (function figABA() {
      const fig = root.querySelector('#fig-aba');
      setH(fig, 500);
      const S = { eff: 0, drift: 0.6, noise: 1.2, seed: 10, t0: 0, vals: [], med: [] };
      let H = null;
      const med3 = (a) => a.slice().sort((x, y) => x - y)[1];
      function sim(anim) {
        const r = V.rng(S.seed), base = 53.3;
        S.vals = [];
        for (let i = 0; i < 9; i++) { const arm = i < 3 ? 0 : i < 6 ? 1 : 2; const d = base * (1 + S.drift / 100 * i); S.vals.push({ i, arm, d, v: d * (arm === 1 ? 1 + S.eff / 100 : 1) + S.noise * r.normal() }); }
        S.med = [0, 1, 2].map((a) => med3(S.vals.filter((x) => x.arm === a).map((x) => x.v)));
        if (anim) S.t0 = H ? H.t : 0;
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Efeito real de B', min: -8, max: 8, step: 0.5, value: 0, fmt: (v) => (v > 0 ? '+' : '') + v.toFixed(1) + '%', onInput: (v) => { S.eff = v; sim(false); } });
      UI.slider(ctl, { label: 'Deriva', min: 0, max: 1.5, step: 0.1, value: 0.6, fmt: (v) => v.toFixed(1) + '%/execução', onInput: (v) => { S.drift = v; sim(false); } });
      UI.slider(ctl, { label: 'Ruído σ', min: 0, max: 4, step: 0.1, value: 1.2, fmt: (v) => v.toFixed(1) + ' tok/s', onInput: (v) => { S.noise = v; sim(false); } });
      UI.button(ctl, '↻ novo sorteio', () => { S.seed++; sim(true); }, { primary: true });
      sim(false);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), wide = w >= 640;
        const px0 = 52, py0 = 26, pw = wide ? w * 0.6 - 52 : w - 110, ph = wide ? h - 80 : h * 0.52;
        const vs = S.vals.map((x) => x.v).concat(S.vals.map((x) => x.d));
        const lo = Math.min(...vs) - 1.5, hi = Math.max(...vs) + 1.5;
        const Y = (v) => py0 + ph - ((v - lo) / (hi - lo)) * ph;
        const colArm = [C.b, C.c, C.b], names = ['A₁', 'B', 'A₂'];
        const slot = pw / 9.8;
        const X = (i) => px0 + (i + 0.4 + Math.floor(i / 3) * 0.4) * slot;
        // faixas dos braços
        for (let a = 0; a < 3; a++) {
          const xa = X(a * 3) - slot * 0.45, xb = X(a * 3 + 2) + slot * 0.45;
          g.fillStyle = V.alpha(colArm[a], 0.07); g.fillRect(xa, py0, xb - xa, ph);
          V.text(g, names[a], (xa + xb) / 2, py0 + ph + 18, { size: 13, align: 'center', color: colArm[a], weight: 700 });
        }
        // eixo
        V.line(g, px0, py0, px0, py0 + ph, { color: V.alpha(C.muted, 0.6), lw: 1 });
        for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; V.text(g, nf(v, 1), px0 - 6, Y(v) + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); }
        V.text(g, 'tok/s (ilustrativo)', px0, py0 - 10, { size: 11, color: C.muted });
        // deriva
        V.line(g, X(0), Y(S.vals[0].d), X(8), Y(S.vals[8].d), { color: V.alpha(C.d, 0.8), lw: 1.5, dash: [5, 4] });
        V.text(g, 'deriva', X(8) + 6, Y(S.vals[8].d) - 4, { size: 11, color: C.d });
        // medianas
        for (let a = 0; a < 3; a++) { const xa = X(a * 3) - slot * 0.35, xb = X(a * 3 + 2) + slot * 0.35; V.line(g, xa, Y(S.med[a]), xb, Y(S.med[a]), { color: colArm[a], lw: 3.5 }); V.text(g, nf(S.med[a], 1), xb + 3, Y(S.med[a]) + 4, { size: 11, font: 'mono', color: colArm[a] }); }
        // pontos (aparecem em sequência)
        S.vals.forEach((x) => {
          const e = clamp((t - S.t0 - x.i * 0.16) / 0.3, 0, 1);
          if (e <= 0) return;
          V.dot(g, X(x.i), Y(x.v), 5 * V.ease.back(e), colArm[x.arm]);
        });
        // veredito
        const s1 = S.med[1] / S.med[0] - 1, s2 = S.med[1] / S.med[2] - 1;
        const naive = s1 >= 0.03, aba = s1 >= 0.03 && s2 >= 0.03;
        const vx = wide ? px0 + pw + 34 : 16, vy = wide ? py0 + 6 : py0 + ph + 42, vw = wide ? w - vx - 14 : w - 32;
        const pct = (x) => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';
        let yy = wrap(g, 'A/B ingênuo (B contra A₁)', vx, vy, vw, 15, { size: 12, color: C.muted, weight: 600 });
        yy = wrap(g, `${pct(s1)} → ${naive ? 'ACEITA' : 'REJEITA'}`, vx, yy + 2, vw, 20, { size: 17, font: 'display', color: naive ? C.a : C.red, weight: 700 });
        yy = wrap(g, 'A/B/A (≥ +3% contra A₁ e contra A₂)', vx, yy + 8, vw, 15, { size: 12, color: C.muted, weight: 600 });
        yy = wrap(g, `${pct(s1)} · ${pct(s2)} → ${aba ? 'ACEITA' : 'REJEITA'}`, vx, yy + 2, vw, 20, { size: 17, font: 'display', color: aba ? C.a : C.red, weight: 700 });
        if (naive && !aba) yy = wrap(g, '⚠ a deriva enganou o A/B ingênuo: o A₂ também “melhorou”', vx, yy + 8, vw, 15, { size: 12, color: C.d, weight: 600 });
        wrap(g, 'e ainda: nenhuma métrica secundária piora mais de 2%, canário de qualidade dentro do ruído, sem OOM no pior caso', vx, yy + 8, vw, 15, { size: 11.5, color: C.muted });
      });
    })();

    /* =====================================================================
       Figura 11 — Naive-N0.5: camada global densa × DSA com indexador top-k
       ===================================================================== */
    (function figNaive() {
      const fig = root.querySelector('#fig-naive');
      setH(fig, 540);
      const GA = [0, 5, 11, 17, 23, 29, 35, 41, 47];          // MiMo: hybrid_layer_pattern (0 = global)
      const DSA = [5, 11, 17, 23, 29, 35, 41, 47];            // Naive: posições ILUSTRATIVAS (~5:1)
      const S = { kind: 'dsa', layer: 11, L: 48, k: 8, W: 8, sc: [] };
      let H = null;
      const ctl = UI.controls(fig);
      const segK = UI.seg(ctl, { label: 'Camada', options: [{ v: 'ga', label: 'global (MiMo)' }, { v: 'dsa', label: 'DSA (Naive)' }, { v: 'swa', label: 'SWA (as duas)' }], value: 'dsa', onChange: (v) => { S.kind = v; S.layer = v === 'swa' ? 1 : 11; restart(); } });
      UI.slider(ctl, { label: 'Contexto L', min: 24, max: 96, step: 8, value: 48, fmt: (v) => v + ' tok', onInput: (v) => { S.L = v; S.k = Math.min(S.k, v - 1); gen(); restart(); } });
      UI.slider(ctl, { label: 'top-k', min: 4, max: 24, step: 1, value: 8, fmt: (v) => String(v), onInput: (v) => { S.k = Math.min(v, S.L - 1); restart(); } });
      const rStore = UI.readout(ctl, 'KV guardado na camada', '—'), rRead = UI.readout(ctl, 'KV lido pela query', '—');
      function gen() {
        const r = V.rng(S.L * 7 + 3);
        const hot = [0.12, 0.37, 0.58, 0.8].map((f) => Math.floor(f * S.L));
        S.sc = Array.from({ length: S.L }, (_, i) => Math.max(0, 0.35 * r.normal() + 0.15 + hot.reduce((a, c) => a + 1.1 * Math.exp(-((i - c) ** 2) / 3), 0)));
        S.sc[S.L - 1] = Math.max(S.sc[S.L - 1], 0.9);
      }
      function topk() { return S.sc.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, S.k).map((x) => x[1]); }
      function restart() {
        S.t0 = H ? H.t : 0; S.sel = new Set(topk());
        const L = S.L;
        rStore.set(S.kind === 'swa' ? `janela de ${S.W} (de ${L})` : S.kind === 'dsa' ? `${L} tokens + ${L} chaves do indexador` : `${L} tokens`);
        rRead.set(S.kind === 'ga' ? `${L} de ${L}` : S.kind === 'dsa' ? `${S.k} de ${L} (+ varredura das chaves)` : `${Math.min(S.W, L)} de ${L}`);
      }
      gen(); restart();
      H = V.fig(fig, (g, w, h, t, dt, _S, HH) => {
        const C = V.C(), narrow = w < 560, L = S.L;
        // ---------- as 48 camadas dos dois modelos ----------
        const lx = narrow ? 54 : 70, cw = (w - lx - 16) / 48, rowsY = [30, 64], rh = 24;
        const kindOf = (m, i) => (m === 0 ? (GA.includes(i) ? 'ga' : 'swa') : (DSA.includes(i) ? 'dsa' : 'swa'));
        const colK = { ga: C.b, dsa: C.f, swa: C.a };
        const P = HH.pointer;
        ['MiMo', 'Naive'].forEach((nm, m) => {
          V.text(g, nm, 16, rowsY[m] + 16, { size: 11.5, weight: 600, color: C.ink });
          for (let i = 0; i < 48; i++) {
            const kd = kindOf(m, i), x = lx + i * cw, y = rowsY[m];
            const selected = i === S.layer && ((S.kind === 'ga' && m === 0) || (S.kind === 'dsa' && m === 1) || (S.kind === 'swa'));
            g.fillStyle = V.alpha(colK[kd], kd === 'swa' ? 0.3 : 0.85);
            g.fillRect(x + 0.5, y, Math.max(1, cw - 1), rh);
            if (selected) { g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(x - 0.5, y - 2, cw + 1, rh + 4); }
            if (P.clicked && P.x >= x && P.x < x + cw && P.y >= y && P.y < y + rh) { S.kind = kd; S.layer = i; segK.set(kd); S.layer = i; }
          }
        });
        V.text(g, 'camada 0', lx, rowsY[1] + rh + 14, { size: 11, font: 'mono', color: C.muted });
        V.text(g, '47', lx + 48 * cw, rowsY[1] + rh + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const legY = rowsY[1] + rh + (narrow ? 30 : 14);
        if (!narrow) [[C.b, 'global densa (9)'], [C.f, 'DSA (posições ilustrativas)'], [V.alpha(C.a, 0.5), 'SWA']].forEach(([c, l], i) => { const x = lx + 90 + i * 170; V.box(g, x, legY - 10, 11, 11, { r: 3, fill: c }); V.text(g, l, x + 16, legY, { size: 11, color: C.muted }); });
        else V.text(g, 'lavanda: global · magenta: DSA · verde: SWA', 16, legY, { size: 11, color: C.muted, maxW: w - 32 });
        // ---------- a camada escolhida ----------
        const e = t - (S.t0 || 0), cyc = 5.5, u = (e % cyc) / cyc;
        const kd = S.kind;
        const title = kd === 'ga' ? `camada ${S.layer} · MiMo · global densa: a query lê o KV de todos os L tokens`
          : kd === 'dsa' ? `camada ${S.layer} · Naive · DSA: o indexador dá nota a cada token e a atenção lê só os top-${S.k}`
          : `camada ${S.layer} · SWA (nas duas): a query lê só a janela`;
        const ty = narrow ? 140 : 122;
        const yy0 = wrap(g, title, 16, ty, w - 32, 15, { size: 12, weight: 600, color: kd === 'ga' ? C.b : kd === 'dsa' ? C.f : C.a });
        const gx = narrow ? 52 : 70, gw = w - gx - 16, cc = gw / L;
        const barTop = yy0 + 4, barH = 44, idxY = barTop + barH + 6, kvY = idxY + 16, kh = narrow ? 15 : 18;
        // barras de nota do indexador (só DSA)
        const scanP = clamp(u / 0.35, 0, 1), selP = clamp((u - 0.35) / 0.15, 0, 1), readP = clamp((u - 0.5) / 0.4, 0, 1);
        const mx = Math.max(...S.sc) || 1;
        V.text(g, kd === 'dsa' ? 'nota' : '', gx - 8, barTop + barH - 2, { size: 11, align: 'right', color: C.muted });
        if (kd === 'dsa') {
          for (let i = 0; i < L; i++) {
            const shown = clamp(scanP * L - i, 0, 1), bh = (S.sc[i] / mx) * barH * shown, isSel = S.sel.has(i);
            g.fillStyle = selP > 0 && isSel ? C.d : V.alpha(C.f, selP > 0 ? 0.25 : 0.6);
            g.fillRect(gx + i * cc + cc * 0.15, barTop + barH - bh, Math.max(1, cc * 0.7), bh);
          }
          if (scanP < 1) { const sxp = gx + scanP * gw; V.line(g, sxp, barTop - 2, sxp, idxY + 10, { color: C.f, lw: 2 }); }
          // linha das chaves do indexador
          g.fillStyle = V.alpha(C.f, 0.35); g.fillRect(gx, idxY, gw, 8);
          V.text(g, 'idx', gx - 8, idxY + 8, { size: 11, align: 'right', font: 'mono', color: C.muted });
        } else {
          V.text(g, kd === 'ga' ? 'sem indexador: nada a escolher, lê tudo' : `janela deslizante de ${S.W} tokens (ilustrativa; no MiMo, 128)`, gx, barTop + barH - 8, { size: 11.5, color: C.muted, maxW: gw });
        }
        // KV: 4 KV heads (GQA) × L tokens
        const readCol = (i) => (kd === 'ga' ? true : kd === 'swa' ? i >= L - S.W : S.sel.has(i));
        for (let r = 0; r < 4; r++) {
          const y = kvY + r * (kh + 3);
          V.text(g, 'kv' + r, gx - 8, y + kh - 4, { size: 11, align: 'right', font: 'mono', color: C.muted });
          for (let i = 0; i < L; i++) {
            const on = readCol(i) && (kd !== 'dsa' || selP > 0);
            const a = on ? 0.25 + 0.6 * (kd === 'dsa' ? selP : 1) * (readP > 0 ? 0.7 + 0.3 * Math.sin(t * 6 + i) : 1) : 0.1;
            g.fillStyle = V.alpha(on ? (kd === 'ga' ? C.b : kd === 'dsa' ? C.d : C.a) : C.muted, a);
            g.fillRect(gx + i * cc + 0.5, y, Math.max(1, cc - 1), kh);
          }
        }
        const kvBot = kvY + 4 * (kh + 3);
        // a query (último token) e os arcos de leitura
        const qx = gx + (L - 0.5) * cc;
        V.box(g, gx + (L - 1) * cc - 1, kvY - 3, cc + 2, kvBot - kvY + 3, { r: 3, stroke: C.ink, lw: 1.5 });
        V.text(g, 'q', qx, kvBot + 13, { size: 11, align: 'center', font: 'mono', weight: 700, color: C.ink });
        if (readP > 0 || kd !== 'dsa') {
          const rp = kd === 'dsa' ? readP : clamp(u / 0.5, 0, 1);
          g.save(); g.globalAlpha = kd === 'ga' ? 0.45 : 0.85;
          for (let i = 0; i < L - 1; i++) {
            if (!readCol(i)) continue;
            const x = gx + (i + 0.5) * cc, depth = 12 + Math.min(36, Math.abs(qx - x) * 0.12);
            g.beginPath(); g.moveTo(qx, kvBot + 2);
            const steps = 16, n = Math.max(1, Math.round(steps * rp));
            for (let s2 = 1; s2 <= n; s2++) { const f = s2 / steps, bx = lerp(qx, x, f), by = kvBot + 2 + Math.sin(f * Math.PI) * depth; g.lineTo(bx, by); }
            g.strokeStyle = kd === 'ga' ? C.b : kd === 'dsa' ? C.d : C.a; g.lineWidth = kd === 'ga' ? 0.8 : 1.4; g.stroke();
          }
          g.restore();
        }
        // conta, em palavras
        let ty2 = kvBot + 72;
        const nRead = kd === 'ga' ? L : kd === 'swa' ? Math.min(S.W, L) : S.k;
        ty2 = wrap(g, `guardado: ${kd === 'swa' ? `só a janela (${S.W} tokens)` : `${L} tokens × 4 KV heads` + (kd === 'dsa' ? ' + uma chave pequena do indexador por token' : '')} · lido pela query: ${nRead} de ${L}` + (kd === 'dsa' ? ' (e a varredura das chaves pequenas)' : ''), 16, ty2, w - 32, 15, { size: 12, color: C.ink });
        if (kd === 'dsa') wrap(g, 'aqui o top-k vale para os 4 KV heads ao mesmo tempo; um top-k por grupo de KV é a alternativa. É uma das decisões do “DSA com GQA”, que o vLLM ainda não tem (o DSA dele existe para MLA)', 16, ty2 + 4, w - 32, 15, { size: 11.5, color: C.muted });
        else if (kd === 'ga') wrap(g, 'no MiMo, as 9 globais são as únicas camadas cujo KV cresce com o contexto (Passo 1). O Naive troca todas elas por DSA', 16, ty2 + 4, w - 32, 15, { size: 11.5, color: C.muted });
        else wrap(g, 'as SWA não mudam: mesma janela, mesmo sink, o mesmo código do mimo_v2.py a reaproveitar no plugin', 16, ty2 + 4, w - 32, 15, { size: 11.5, color: C.muted });
      });
    })();

    /* =====================================================================
       Figura 12 — checklist final (MiMo / Naive / Mac)
       ===================================================================== */
    (function figCheck() {
      const fig = root.querySelector('#fig-check');
      const base = [
        'Passo 0 · tabela de triagem + riscos ligados a PRs/issues',
        'Passo 1 · previsto × medido (pesos, pool, tokens) dentro de ±10%',
        'Passo 2 · checkpoint verificado, revisão fixada (KLD se não for o oficial)',
        'Passo 3 · divisibilidade sem ✘, TP escolhido, alternativas anotadas',
        'Passo 4 · overlays com testes de CPU; paridade TP1 = TP2 ≈ HF; marcadores no log',
        'Passo 5 · boot nos 2 nós; backend, KV e capture sizes esperados; comando renderizado salvo',
        'Passo 6 · aceitação × k × categoria × contexto; SPEC_METHOD/SPEC_K com rollback',
        'Passo 7 · smoke.sh verde (sanidade, tools, needle, canário, OOM) anotado no README',
        'Passo 8 · decode, prefill frio e aceitação com método, data, commit e digest',
        'Passo 9 · tuning com regra A/B/A; resultados negativos no CHANGELOG',
        'Passo 10 · outro par de Sparks reproduz o README com ±5–10%',
      ].map((t) => ({ t }));
      const naive = base.slice();
      naive[0] = { t: 'Passo 0 · sem entrada no registry: NaiveN05FlashForCausalLM + remote code', d: 1 };
      naive[1] = { t: 'Passo 1 · fórmula de KV por camada DSA (KV + chave do indexador, leitura top-k)', d: 1 };
      naive[2] = { t: 'Passo 2 · quantizar experts (NVFP4 calibrado) + KLD contra o FP8 em ≥ 3 nós ou HF truncado', d: 1 };
      naive[3] = { t: 'Passo 3 · mesma espinha do MiMo: conferir divisibilidade e o pré-sharding no índice' };
      naive[4] = { t: 'Passo 4 · plugin out-of-tree; bloco DSA com GQA; paridade truncada com N = 6', d: 1 };
      naive[5] = { t: 'Passo 5 · backends de sparse-MLA do sm_121 não servem direto: escolher ou portar', d: 1 };
      naive[6] = { t: 'Passo 6 · FP8-Draft em formato conhecido (EAGLE3/DFlash)? Senão, registrar', d: 1 };
      const mac = [
        'Mac com ≥ 192 GB unificados (M3 Ultra 256 GB ou M5 Ultra)',
        'mlx-lm do PR #1219 (branch kernelpool/mlx-lm@add-mimo-v2) num venv',
        'Checkpoint MLX: mlx-community (RL, só texto); MOPD via mlx_lm.convert [a medir]',
        'mlx_lm.server em 127.0.0.1:8080 respondendo /v1/chat/completions',
        'Thinking desligado por chat_template_kwargs (ou no cliente) e max_tokens folgado',
        'Tools no formato Qwen3-Coder testados com a suíte do Passo 7',
        'Decode e prefill medidos (referência: 59,4 e 563 tok/s @2K no M3 Ultra)',
      ].map((t) => ({ t }));
      const LISTS = { mimo: base, naive, mac };
      const saved = LA.store.get('rdz-checklist', {}) || {};
      const S = { mode: 'mimo', p: 0 };
      const ctl = UI.controls(fig);
      const list = LA.h('div', { style: { display: 'grid', gap: '9px', padding: '14px 18px', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(list, fig.querySelector('figcaption'));
      function build() {
        list.innerHTML = '';
        const L = LISTS[S.mode], st = saved[S.mode] || [];
        L.forEach((it, i) => {
          const tg = UI.toggle(list, { label: (it.d ? '≠ ' : '') + it.t, value: !!st[i], onChange: (v) => { const a = saved[S.mode] || (saved[S.mode] = []); a[i] = v; LA.store.set('rdz-checklist', saved); } });
          const sw = tg.el.querySelector('i'); if (sw) sw.style.flexShrink = '0';
        });
      }
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'mimo', label: 'MiMo · 2× Spark' }, { v: 'naive', label: 'Desafio Naive' }, { v: 'mac', label: 'Mac · MLX' }], value: 'mimo', onChange: (v) => { S.mode = v; build(); } });
      UI.button(ctl, 'limpar', () => { saved[S.mode] = []; LA.store.set('rdz-checklist', saved); build(); });
      build();
      const SHORT = {
        mimo: ['Triagem', 'Memória', 'Pesos', 'Topologia', 'Motor', 'Lançamento', 'Especulação', 'Validação', 'Benchmark', 'Tuning', 'Pacote'],
        mac: ['Mac ≥192 GB', 'mlx-lm #1219', 'Checkpoint', 'Servidor', 'Thinking', 'Tools', 'Medir'],
      };
      SHORT.naive = SHORT.mimo;
      V.fig(fig, (g, w, h, t, dt, _S, HH) => {
        const C = V.C(), L = LISTS[S.mode], st = saved[S.mode] || [];
        const n = L.length, done = L.filter((_, i) => !!st[i]).length;
        const names = SHORT[S.mode];
        const phaseCol = (i) => (S.mode === 'mac' ? C.a : i <= 3 ? C.b : i <= 7 ? C.a : C.c);
        S.p += (done / n - S.p) * Math.min(1, dt * 4);
        const x0 = 34, x1 = w - 34, y = Math.round(h * 0.5), wideLab = w >= 640;
        const X = (i) => lerp(x0, x1, n === 1 ? 0 : i / (n - 1));
        // clique numa estação marca/desmarca o critério (e sincroniza a lista)
        const P = HH.pointer;
        let hover = -1;
        for (let i = 0; i < n; i++) if (P.inside && Math.hypot(P.x - X(i), P.y - y) < 16) hover = i;
        if (P.clicked && hover >= 0) { const a = saved[S.mode] || (saved[S.mode] = []); a[hover] = !a[hover]; LA.store.set('rdz-checklist', saved); build(); }
        // trilho por fase + trilho percorrido
        for (let i = 0; i < n - 1; i++) V.line(g, X(i), y, X(i + 1), y, { color: V.alpha(phaseCol(i + 1), 0.28), lw: 8 });
        const tx = lerp(x0, x1, S.p);
        V.line(g, x0, y, tx, y, { color: C.a, lw: 8 });
        g.setLineDash([2, 10]); g.lineDashOffset = -t * 18; V.line(g, x0, y, x1, y, { color: V.alpha(C.bg || '#000', 0.5), lw: 2 }); g.setLineDash([]);
        // estações
        for (let i = 0; i < n; i++) {
          const xx = X(i), ok = !!st[i], col = phaseCol(i), r = hover === i ? 12 : 10;
          V.dot(g, xx, y, r, ok ? col : (C.panel || C.bg));
          g.beginPath(); g.arc(xx, y, r, 0, 7); g.strokeStyle = ok ? col : V.alpha(col, 0.8); g.lineWidth = 2.5; g.stroke();
          V.text(g, ok ? '✓' : String(S.mode === 'mac' ? i + 1 : i), xx, y + 4, { size: 11, align: 'center', font: ok ? 'sans' : 'mono', weight: 700, color: ok ? onInk(C) : C.ink });
          if (L[i].d) V.text(g, '≠', xx, y - r - 6, { size: 13, align: 'center', color: C.d, weight: 700 });
          if (wideLab) {
            const below = i % 2 === 0, ly = below ? y + 32 : y - (L[i].d ? 30 : 22);
            V.text(g, names[i], xx, ly, { size: 11, align: 'center', color: ok ? C.ink : C.muted, weight: hover === i ? 600 : 400, maxW: (x1 - x0) / (n - 1) * 1.9 });
          }
        }
        // trem
        V.glow(g, tx, y, 22, C.a, 0.9);
        V.box(g, tx - 15, y - 10, 30, 20, { r: 9, fill: C.a });
        g.fillStyle = onInk(C); g.beginPath(); g.arc(tx - 5, y - 2, 2, 0, 7); g.arc(tx + 5, y - 2, 2, 0, 7); g.fill();
        // topo: progresso; base: próximo critério ou estação sob o mouse
        const allDone = done === n;
        const msg = allDone ? (S.mode === 'mac' ? '✓ pronto: o mesmo contrato OpenAI, agora no Mac' : '✓ pronto: um segundo par reproduz o README (±5–10%)') : `${done} de ${n} critérios de saída cumpridos`;
        V.text(g, msg, 24, 26, { size: allDone ? 15 : 13, weight: 600, color: allDone ? C.a : C.ink, maxW: w - 48, font: allDone ? 'display' : 'sans' });
        if (S.mode !== 'mac') {
          const leg = [[C.b, 'planejar'], [C.a, 'subir certo'], [C.c, 'medir e entregar']];
          let lx = 24; g.font = V.font(11, 'sans', 400);
          leg.forEach(([c, l]) => { V.box(g, lx, 38, 10, 10, { r: 3, fill: c }); V.text(g, l, lx + 15, 47, { size: 11, color: C.muted }); lx += g.measureText(l).width + 34; });
        }
        const next = L.findIndex((_, i) => !st[i]), focus = hover >= 0 ? hover : next;
        if (focus >= 0 && !allDone) {
          const it = L[focus];
          wrap(g, (hover >= 0 ? 'estação: ' : 'próxima: ') + it.t, 24, h - 34, w - 48, 15, { size: 12, color: hover >= 0 ? C.ink : C.muted, weight: hover >= 0 ? 600 : 400 });
        }
        if (allDone) for (let k = 0; k < 14; k++) { const a = t * 1.3 + k, xx = x1 - 30 + Math.cos(a * 1.7) * 22, yy = y - 50 + Math.sin(a * 2.3) * 12; V.dot(g, xx, yy, 2.2, C.series[k % 6]); }
        if (hover >= 0) HH.canvas.style.cursor = 'pointer'; else HH.canvas.style.cursor = '';
      });
    })();
  },
});

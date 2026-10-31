LA.chapter({
  id: 'receita-glm',
  part: 9,
  order: 42,
  title: 'Reconstruindo a receita GLM-5.3-Flash EXL3',
  navTitle: 'Receita GLM',
  subtitle: 'Do orçamento de bytes ao banner READY: cada linha do launcher, do .env e do vllm serve, com o porquê de cada valor.',
  minutes: 65,
  recipes: ['glm', 'tf'],
  tags: ['receita', 'GLM-5.3-Flash', 'EXL3', 'launcher', 'vllm serve', 'DFlash2', 'TP=2', 'boot', 'overlays', 'TensorFold'],
  goals: [
    'Fazer a <b>conta de restos</b> de memória de um nó UMA, separando pool perfilado de pool fixo, e justificar <code>GPU_MEM_UTIL=0,85</code> e o pool fixo de 11 GiB',
    'Traduzir o pool de KV em <b>IDs de bloco</b> e explicar por que as páginas compactas fazem 850k caber 1,45× em 11 GiB',
    'Explicar cada flag do <code>vllm serve</code>, cada campo do <code>--speculative-config</code> e as políticas de runtime (adaptive-k, fair v5, APC híbrido)',
    'Entender como o launcher garante <b>a mesma imagem, os mesmos patches, o mesmo ambiente e a mesma rede</b> nos dois ranks, e como verificar o engine com logs, canário e gate de aceitação',
    'Ler a linha do tempo do boot, escolher um perfil por workload e dizer quais estações do mapa ficam, trocam ou somem quando o motor vira TensorFold',
  ],
  prereqs: ['engenharia-de-receita', 'tres-modelos', 'dflash-dspark'],
  summary: [
    'A receita é uma conta de restos: ~79,65 GiB de pesos por rank, 3,3 GiB da cópia BF16 do KDA, 11 GiB de KV fixo, workspaces e ~9 GiB de inicialização, e o head ainda mantém ≥ 6,2 GiB livres sob carga. Com o pool fixo, <code>GPU_MEM_UTIL=0,85</code> não dimensiona mais o KV: vira só o piso de memória livre que o boot exige (0,85 × 121,69 ≈ 103,4 GiB).',
    'O pool de KV se conta em IDs de bloco: 1 ID = 3584 tokens × 11 camadas × (656 + 33) B ≈ 25,9 MiB, e todo grupo (MLA, kpool, 4 Mamba, drafter) paga IDs inteiros. Uma requisição de 850k custa ~536 IDs sem páginas compactas e ~299 com, por isso 11 GiB (434 IDs) guardam 1,45× de 850k.',
    'Quase toda flag vem de uma restrição: EXL3 porque BF16/FP8 não cabem; <code>fp8</code> → <code>fp8_ds_mla</code> (656 B, com 64 zeros de RoPE) porque é o único kernel sparse-MLA do SM121; pool fixo porque o InstantTensor come KV; <code>TEMP_ROWS=32</code> porque 4 seqs × (7+1) posições. Trocar o motor pelo TensorFold apaga as restrições que vinham do vLLM (pad, páginas de 3584, decode uniforme) e traz outras: só 2 ranks e até 4 requisições na v1.3.2 (3 Sparks desde a v1.4, 8 requisições desde a v1.5) e saída bit a bit igual à serial.',
    'A imagem é pinada por digest, patcheada de forma fail-closed e carimbada; 25 overlays (26 desde o #203) são reaplicados na mesma ordem nos dois ranks, e a rede é fixada por rank (interface, HCA, GID). Ordem errada falha; paridade quebrada trava o NCCL.',
    'O launcher dá precedência aos exports do chamador (inclusive vazios), valida tudo antes de parar o par, serializa o ciclo de vida com flock e aborta cedo se o worker morrer.',
    'Boot: <code>restart</code> 259 → 122 s; do <code>docker run</code> ao <code>/health</code>, 99 s, a maior parte no stream de pesos. Depois vêm o canário (automático), o gate de aceitação (manual) e benchmarks A/B/A.',
    'Políticas e opções são trocas medidas: adaptive-k (+13–21% na prosa, desligado por padrão), fair v5, v7 desde o #283 (nem inanição, nem incumbente 10–36× mais lento), APC híbrido alinhado a 3584, FP8 denso, KDA BF16, E3, cooperativo, agrupados em perfis como <code>.env.example</code> e <code>tp2-long-coding</code>.',
  ],
  quiz: [
    { q: 'Por que a receita passa <code>--kv-cache-memory-bytes 11811160064</code> em vez de deixar o vLLM perfilar o pool?', o: ['Porque o vLLM não sabe perfilar memória em GPUs integradas', 'Porque, com o InstantTensor, o pool perfilado (~12,4 GiB) ficava abaixo dos 13,56 GiB de uma requisição de 850k e o boot era recusado; 11 GiB fixos com páginas compactas guardam 1,45× de 850k', 'Porque 11 GiB é o máximo que o kernel sparse-MLA consegue endereçar', 'Porque um pool fixo deixa o prefill mais rápido'], a: 1, e: 'O loader direct-I/O deixa ~4,4–5,6 GiB a menos para o KV perfilado. Fixar o pool resolve, e as páginas compactas do drafter fazem 11 GiB bastarem (1.233.779 tokens). Os 3 GiB poupados em relação aos 14 GiB anteriores pagam a cópia BF16 do KDA.' },
    { q: 'Você roda <code>MM_IMAGE_TOKENS= ./start.sh</code> (vazio explícito). O que acontece?', o: ['Vale o default 2048, porque vazio conta como ausente', 'A validação recusa o valor antes de parar o par', 'O vazio sobrevive a <code>${MM_IMAGE_TOKENS-2048}</code>, a flag é omitida e o vLLM volta aos 8.000 tokens por imagem do checkpoint, acima do cache de encoder de 7.168', 'O .env vence e aplica 2048'], a: 2, e: 'O export do chamador, mesmo vazio, vence o .env. Com o operador <code>-</code>, sem os dois-pontos, o vazio é preservado e significa “stock”. Aqui, “stock” é perigoso: 8.000 tokens por imagem passam do MNBT/cache do encoder.' },
    { q: 'Com <code>MAX_NUM_SEQS=4</code> e <code>DFLASH_TOKENS=7</code>, de onde vêm <code>EXL3_TEMP_ROWS_FUSED=32</code> e os tamanhos de captura 8/16/24/32?', o: ['São valores empíricos, sem relação entre si', 'Cada sequência verifica 7+1 = 8 posições por passo; com 1–4 seqs o decode tem 8, 16, 24 ou 32 tokens, e 4 × 8 = 32 linhas por expert cabem no buffer do MoE fundido', 'Vêm do número de camadas MLA (11) multiplicado por 3', 'São potências de 2 exigidas pelo NCCL'], a: 1, e: 'A cadeia desce da especulação: bloco de 8 → N·8 tokens por passo → esses tamanhos precisam ter CUDA graph → o buffer temporário do exl3_moe precisa de ≥ N·8 linhas para que o decode seja um único lançamento graph-safe.' },
    { q: 'Por que o launcher compara head e worker pelo hash de <code>RootFS.Layers</code>, e não pelo <code>.Id</code> da imagem?', o: ['Porque o .Id muda a cada docker run', 'Porque o .Id difere entre overlay2 e containerd e o docker save|load perde os RepoDigests; as diff IDs das camadas são iguais nos dois lados', 'Porque RootFS.Layers inclui o recipe stamp', 'Porque o GHCR não publica .Id'], a: 1, e: 'Nenhum outro campo sobrevive a todos os caminhos (pull, build local, save|load). Antes do image key, a imagem inteira era reenviada ao worker a cada execução.' },
    { q: 'Com <code>GLM53_ADAPTIVE_K=ema</code> ligado, o rank 1 sobe sem o overlay nº 18 (adaptive-k). O que acontece?', o: ['Nada: overlays só afetam o rank 0', 'O vLLM detecta e reaplica o patch sozinho', 'O rank 0 captura graphs de decode para 3, 5 e 8 posições e o rank 1 só para 8; os dois lançam coletivas diferentes, e o NCCL trava depois do PIECEWISE', 'O worker cai para modo eager e fica mais lento'], a: 2, e: 'Em TP=2 os dois ranks executam a mesma sequência de kernels e coletivas. O README resume: um rank que captura CUDA graphs diferentes trava o NCCL. A falha exata é interpretação; com o adaptive-k desligado (o default), os dois verificariam 8 posições e o buraco passaria despercebido, o que torna o erro ainda mais traiçoeiro.' },
    { q: 'A receita fixa o pool com <code>--kv-cache-memory-bytes</code>. Você sobe <code>GPU_MEM_UTIL</code> de 0,85 para 0,88. O que muda?', o: ['O pool de KV cresce ~3,6 GiB (3 × 1,2 GiB)', 'Nada em runtime: o consumo não depende de u. Muda o piso de memória livre que o boot exige, 0,88 × 121,69 ≈ 107,1 GiB, no limite do que o worker tem livre no check (0,88 bootou 1 de 4 vezes)', 'O host ganha ~3,6 GiB de folga', 'O vLLM volta a perfilar o pool'], a: 1, e: 'Com o pool fixo, o vLLM ignora u para dimensionar o KV. A util sobra como exigência de memória livre: a checagem de startup do vLLM e o pré-voo do launcher comparam o MemAvailable com u × M. A regra de cada 0,01 valer ~1,2 GiB é do regime perfilado, em que u decide quanto sobra para o KV.' },
    { q: 'Qual destas decisões da receita vLLM <b>some</b> na versão TensorFold, porque vinha do motor e não do hardware nem do modelo?', o: ['Quantizar só os experts roteados em EXL3 4 bpw', 'TP=2, com um rank por Spark', 'O pad 512→576 do registro de KV das camadas MLA', 'Especular com o DFlash2, porque o decode é limitado pela banda'], a: 2, e: 'O pad existe porque o único kernel de sparse-MLA do SM12x tem geometria fixa de 576. O TensorFold lê o latente de 512 com kernels próprios, sem pad: 528 B por token e camada em vez de 656 B. As outras três vêm de restrições físicas ou do modelo (164 GiB que não cabem em BF16, dois nós de 128 GB, decode limitado pela banda) e continuam nos dois motores.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ---------------------------------------------------------- helpers */
    const fcache = {};
    const FN = (s, f, w) => { const k = s + '|' + (f || 'sans') + '|' + (w || 400); return fcache[k] || (fcache[k] = V.font(s, f || 'sans', w || 400)); };
    const tw = (g, s, size, font, wt) => { g.font = FN(size || 12, font, wt); return g.measureText(String(s)).width; };
    const wrap = (g, s, maxW, size, font, wt) => {
      g.font = FN(size || 12, font, wt);
      const out = []; let cur = '';
      String(s).split(' ').forEach((wd) => { const tt = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(tt).width > maxW) { out.push(cur); cur = wd; } else cur = tt; });
      if (cur) out.push(cur);
      return out;
    };
    const para = (g, s, x, y, maxW, o) => {
      o = o || {}; const size = o.size || 12, lh = o.lh || Math.round(size * 1.4);
      const L = wrap(g, s, maxW, size, o.font, o.weight);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({}, o, { size })));
      return L.length * lh;
    };
    const br = (x, d) => (+x).toFixed(d == null ? 1 : d).replace('.', ',');
    const near = (cur, tgt, dt, k) => (cur == null || !isFinite(cur) ? tgt : cur + (tgt - cur) * (1 - Math.exp(-dt * (k || 9))));
    const chip = (g, x, y, label, o) => {
      o = o || {}; const size = o.size || 11, px = o.px == null ? 7 : o.px, hh = o.h || size + 9;
      const wd = tw(g, label, size, o.font || 'mono', o.weight || 500) + px * 2;
      if (o.dash) g.setLineDash(o.dash);
      V.box(g, x, y, wd, hh, { r: o.r == null ? 7 : o.r, fill: o.fill, stroke: o.stroke, lw: o.lw || 1 });
      g.setLineDash([]);
      V.text(g, label, x + px, y + hh / 2 + 0.5, { size, font: o.font || 'mono', weight: o.weight || 500, color: o.color, base: 'middle' });
      return wd;
    };
    const onFill = (C) => (C.dark ? '#0b1a13' : '#ffffff');
    /** ajusta a altura do palco conforme a largura (mobile precisa de mais altura) */
    const fitH = (fig, wide, narrowH, w) => { const st = fig.querySelector('.la-fig-stage'); const want = w < 600 ? narrowH : wide; if (st && Math.abs(st.clientHeight - want) > 2) st.style.height = want + 'px'; };

    /* ============================================ F1 · orçamento de memória */
    const GI = 2 ** 30, PER_ID = 27163136, MLA_850 = Math.ceil(850000 / 3584), REQ_ON = 299, REQ_OFF = 536;
    const fmtInt = (n) => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '.');
    (function () {
      const fig = root.querySelector('#fig-orcamento');
      const S = { mode: 'fixo', util: 0.85, kv: 11, kda: true, rs: true, compact: true };
      const A = {};
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Regime', options: [{ v: 'fixo', label: 'pool fixo (receita)' }, { v: 'perf', label: 'pool perfilado' }], value: 'fixo', onChange: (v) => { S.mode = v; } });
      UI.slider(ctl, { label: 'GPU_MEM_UTIL', min: 0.75, max: 0.9, step: 0.005, value: 0.85, fmt: (v) => br(v, 3), onInput: (v) => { S.util = v; } });
      UI.seg(ctl, { label: 'Pool fixo', options: [{ v: 11, label: '11 GiB' }, { v: 14, label: '14 GiB' }, { v: 15, label: '15 GiB' }], value: 11, onChange: (v) => { S.kv = v; } });
      UI.toggle(ctl, { label: 'cópia BF16 do KDA', value: true, onChange: (v) => { S.kda = v; } });
      UI.toggle(ctl, { label: 'indexer rightsize', value: true, onChange: (v) => { S.rs = v; } });
      UI.toggle(ctl, { label: 'páginas compactas', value: true, onChange: (v) => { S.compact = v; } });
      const M = 121.69;
      const IDX_STOCK = (850000 * 40 * 132) / GI;
      const IDX_RS = (4 * Math.ceil(850007 / 4) * 132) / GI;
      const E3 = 560 / 1024, WS_RS = IDX_RS + E3, W = 79.65, KDA = 3.29, INIT = 9;
      // interpretação: "resto" calibrado para reproduzir os ≥ 6,2 GiB livres medidos no default
      const REST_FIX = M - (W + KDA + 11 + WS_RS + INIT) - 6.2;
      // interpretação: estimativa linear do pool perfilado, ancorada nos 12,4 GiB medidos a 0,85 (#204, sem a cópia do KDA)
      const REST_PROF = 0.85 * M - W - WS_RS - 12.4;
      const MEAS = { '11|1': '1.233.779 tokens medidos', '14|1': '1.572.073 medidos', '14|0': '876.958 medidos', '11|0': '~690k medidos: o boot recusa 850k', '15|0': '1,11× medido' };
      V.fig(fig, (g, w, h, t, dt) => {
        if (w < 160) return;
        const C = V.C();
        const narrow = w < 560;
        const pad = 16, bx = pad, bw = w - 2 * pad;
        const fixed = S.mode === 'fixo';
        const ws = (S.rs ? IDX_RS : IDX_STOCK) + E3, kda = S.kda ? KDA : 0, uM = S.util * M;
        const tgt = { w: W, kda, ws, uM };
        if (fixed) { Object.assign(tgt, { kv: S.kv, init: INIT, rest: REST_FIX }); tgt.free = M - (W + kda + S.kv + ws + INIT + REST_FIX); }
        else { Object.assign(tgt, { init: 0, rest: REST_PROF }); tgt.kv = uM - W - kda - ws - REST_PROF; tgt.free = M - uM; }
        Object.keys(tgt).forEach((k) => { A[k] = near(A[k], tgt[k], dt, 8); });
        const X = (gib) => bx + (Math.max(0, Math.min(gib, M)) / M) * bw;
        const title = fixed ? (narrow ? 'POOL FIXO · CONSUMO (NÃO DEPENDE DE u)' : 'POOL FIXO (A RECEITA) · CONSUMO EM RUNTIME, QUE NÃO DEPENDE DE u') : (narrow ? 'POOL PERFILADO · u×M DIMENSIONA O KV' : 'POOL PERFILADO (vLLM STOCK) · u×M DIMENSIONA O KV');
        V.text(g, title, bx, 18, { size: 11, font: 'mono', weight: 700, color: fixed ? C.a : C.d, maxW: bw });
        const by = 30, bh = narrow ? 38 : 44;
        V.box(g, bx, by, bw, bh, { r: 9, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
        const segs = fixed ? [
          { k: 'w', lab: 'pesos', leg: 'pesos por rank', sh: 'pesos/rank', col: C.d },
          { k: 'kda', lab: 'KDA', leg: 'cópia BF16 do KDA', sh: 'KDA BF16', col: C.f },
          { k: 'kv', lab: 'KV', leg: 'pool de KV fixo', sh: 'pool KV', col: C.a },
          { k: 'ws', lab: 'ws', leg: 'indexer + scratch E3', sh: 'workspaces', col: C.c },
          { k: 'init', lab: 'init', leg: 'init do vLLM (~9 GiB, README)', sh: 'init vLLM', col: C.b },
          { k: 'rest', lab: 'resto', leg: 'ativações, graphs, drafter, SO (interp.)', sh: 'resto', col: C.b, hatch: true },
          { k: 'free', lab: 'livre', leg: 'livre (MemAvailable)', sh: 'livre', col: C.e, free: true },
        ] : [
          { k: 'w', lab: 'pesos', leg: 'pesos por rank', sh: 'pesos/rank', col: C.d },
          { k: 'kda', lab: 'KDA', leg: 'cópia BF16 do KDA', sh: 'KDA BF16', col: C.f },
          { k: 'ws', lab: 'ws', leg: 'indexer + scratch E3', sh: 'workspaces', col: C.c },
          { k: 'rest', lab: 'resto', leg: 'ativações, graphs, init (perfil; interp.)', sh: 'resto', col: C.b, hatch: true },
          { k: 'kv', lab: 'KV', leg: 'KV perfilado = u×M − o resto', sh: 'KV perfilado', col: C.a },
          { k: 'free', lab: 'host', leg: 'host + o que fica fora da conta', sh: 'host', col: C.muted, free: true },
        ];
        let acc = 0;
        segs.forEach((s) => {
          const v = Math.max(0, A[s.k]); const x0 = X(acc), x1 = X(acc + v); acc += v;
          if (x1 - x0 < 0.6) return;
          g.save(); V.rr(g, x0, by + 1, x1 - x0, bh - 2, 5); g.clip();
          g.fillStyle = V.alpha(s.col, s.hatch ? 0.2 : s.free ? 0.22 : 0.82); g.fillRect(x0, by, x1 - x0, bh);
          if (s.hatch) { g.strokeStyle = V.alpha(s.col, 0.8); g.lineWidth = 1.2; for (let q = x0 - bh; q < x1; q += 7) { g.beginPath(); g.moveTo(q, by + bh); g.lineTo(q + bh, by); g.stroke(); } }
          g.restore();
          if (x1 - x0 > tw(g, s.lab, 11, 'sans', 700) + 8) V.text(g, s.lab, (x0 + x1) / 2, by + bh / 2 + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: s.hatch || s.free ? C.ink : onFill(C) });
        });
        const over = fixed ? tgt.free < 0 : tgt.kv < 0;
        if (over) { const x0 = fixed ? X(M) - 6 : X(uM); g.fillStyle = V.alpha(C.red, 0.28 + 0.18 * Math.sin(t * 5)); g.fillRect(Math.min(x0, X(M) - 6), by - 4, 6, bh + 8); }
        if (!fixed) V.line(g, X(A.uM), by - 7, X(A.uM), by + bh + 7, { color: C.ink, lw: 2, dash: [4, 3] });
        for (let q = 0; q <= 120; q += narrow ? 40 : 20) { const xq = X(q); V.line(g, xq, by + bh + 2, xq, by + bh + 6, { color: V.alpha(C.muted, 0.8), lw: 1 }); V.text(g, q + (q === 0 ? ' GiB' : ''), xq, by + bh + 17, { size: 11, font: 'mono', align: q === 0 ? 'left' : 'center', color: C.muted }); }
        const sy = by + bh + 34;
        if (fixed) {
          V.text(g, 'consumo ≈ ' + br(M - Math.max(0, A.free)) + ' GiB', bx, sy, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
          V.text(g, 'livre ≈ ' + br(Math.max(0, A.free)) + ' GiB', bx + bw, sy, { size: 11.5, font: 'mono', color: C.e, align: 'right', weight: 600 });
        } else {
          V.text(g, 'vLLM · u×M = ' + br(A.uM) + ' GiB', bx, sy, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
          V.text(g, 'host · ' + br(M - A.uM) + ' GiB', bx + bw, sy, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        }
        // régua do boot
        const gy0 = sy + 22;
        V.text(g, narrow ? 'BOOT · EXIGE LIVRE ≥ u×M = ' + br(uM) + ' GiB' : 'BOOT · A CHECAGEM DE STARTUP EXIGE MEMÓRIA LIVRE ≥ u×M = ' + br(uM) + ' GiB', bx, gy0, { size: 11, font: 'mono', weight: 700, color: C.d, maxW: bw });
        const gy = gy0 + 9, gh = 18;
        V.box(g, bx, gy, bw, gh, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3) });
        V.box(g, bx, gy, Math.max(4, X(A.uM) - bx), gh, { r: 6, fill: V.alpha(C.d, 0.35) });
        const b0 = X(105.8), b1 = X(107.1);
        g.fillStyle = V.alpha(C.e, 0.5); g.fillRect(b0, gy - 3, Math.max(3, b1 - b0), gh + 6);
        V.line(g, X(A.uM), gy - 5, X(A.uM), gy + gh + 5, { color: C.ink, lw: 2, dash: [4, 3] });
        let gv;
        if (uM < 105.5) gv = [C.e, '✓ passa: ' + br(105.8 - uM) + ' GiB abaixo do livre medido no check do worker (faixa verde: 105,8–107,1 GiB)'];
        else if (uM < 106.5) gv = [C.d, '⚠ no limite: 0,87 exige 105,9 GiB e falha por < 1 GiB em nós com serviços residentes'];
        else if (uM <= 107.15) gv = [C.d, '⚠ dentro da faixa medida: 0,88 (107,09 GiB) bootou 1 de 4 vezes'];
        else gv = [C.red, '✕ acima do livre medido no check: o startup recusa (interpretação, extrapolando a faixa)'];
        let iy = gy + gh + 20;
        iy += para(g, gv[1], bx, iy, bw, { size: 12, color: gv[0], lh: 16, weight: 600 }) + 8;
        // legenda (2 colunas)
        const items = segs.map((s) => [s.col, narrow ? s.sh : s.leg, tgt[s.k], s.hatch]);
        const colW = bw / 2, rowH = 20;
        items.forEach(([col, lab, v, hatch], i) => {
          const cx = bx + (i % 2) * colW, cy = iy + Math.floor(i / 2) * rowH;
          V.box(g, cx, cy - 9, 10, 10, { r: 3, fill: hatch ? V.alpha(col, 0.35) : col });
          const vs = br(Math.max(0, v)) + ' GiB', vw = tw(g, vs, 11.5, 'mono');
          V.text(g, lab, cx + 16, cy, { size: 11.5, color: C.ink, maxW: colW - 12 - vw - 22 });
          V.text(g, vs, cx + colW - 12, cy, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        });
        iy += Math.ceil(items.length / 2) * rowH + 8;
        // capacidade em IDs de bloco
        const kvG = Math.max(0, fixed ? S.kv : tgt.kv), ids = Math.floor((kvG * GI) / PER_ID), per = S.compact ? REQ_ON : REQ_OFF, x = ids / per;
        let capT = (fixed ? 'KV ' + br(kvG, kvG % 1 ? 1 : 0) : 'KV perfilado ≈ ' + br(kvG)) + ' GiB = ' + ids + ' IDs de 25,9 MiB; 850k custa ' + per + ' IDs → ' + br(x, 2) + '×';
        if (fixed && MEAS[S.kv + '|' + (S.compact ? 1 : 0)]) capT += ' (' + MEAS[S.kv + '|' + (S.compact ? 1 : 0)] + ').';
        else if (x < 1) capT += ': não cabe uma requisição de 850k, e o boot falha ao alocar o KV com os pesos já carregados.';
        else capT += ' ≈ ' + fmtInt(x * 850000) + ' tokens.';
        const lines = [[x < 1 ? C.red : C.a, capT]];
        lines.push([C.a, 'Pré-voo do launcher: MemAvailable ≥ u×M + 2 GiB = ' + br(uM + 2) + ' GiB antes do docker run; no restart, espera u×M + 1,5 GiB.']);
        let warn;
        if (fixed) {
          if (tgt.free < 2.5) warn = [C.red, '⚠ Pouca folga (estimativa linear): 15 GiB fixos mediram só 0,8–2,2 GiB livres sob carga, pouco para prefills longos.'];
          else if (!S.rs) warn = [C.d, '⚠ Workspace stock do indexer: 850k × 40 × 132 B ≈ 4,2 GiB travados só para o gather do prefill.'];
          else if (Math.abs(S.util - 0.85) < 0.001 && S.kv === 11 && S.kda && S.compact) warn = [C.e, '✓ Default da receita: o head mantém ≥ 6,2 GiB livres sob carga (medido). Mova u: nenhum bloco acima se mexe.'];
          else if (!S.kda) warn = [C.b, 'Sem a cópia BF16 do KDA: +3,3 GiB livres, mas o prefill perde ~12% de TTFT.'];
          else warn = [C.b, 'No pool fixo, u não mexe em nenhum bloco do consumo: só no piso de memória livre do boot.'];
        } else {
          if (S.util >= 0.869) warn = [C.red, '⚠ 0,87 com prefill de 256k e MemAvailable zerado derrubou um head (2026-09-06), neste regime.'];
          else warn = [C.b, 'Cada 0,01 de u são ~1,2 GiB de KV tirados do host. Sem teto, o head chegou a ~1,5 GiB de MemAvailable a 850k (README).'];
        }
        lines.push(warn);
        lines.forEach(([col, s]) => { V.dot(g, bx + 4, iy - 4, 3.5, col); iy += para(g, s, bx + 14, iy, bw - 14, { size: 12, color: C.ink, lh: 16 }) + 5; });
        fitH(fig, Math.ceil(iy + 4), Math.ceil(iy + 4), w);
      });
    })();

    /* ============================================ F1b · o pool em IDs de bloco */
    (function () {
      const fig = root.querySelector('#fig-ids');
      const S = { gib: 11, compact: true, view: 'req', t0: null };
      const reset = () => { S.t0 = null; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vista', options: [{ v: 'req', label: 'requisições de 850k' }, { v: 'seg', label: 'segmento cacheado' }], value: 'req', onChange: (v) => { S.view = v; reset(); } });
      UI.slider(ctl, { label: 'pool de KV', min: 8, max: 16, step: 0.5, value: 11, fmt: (v) => br(v, 1) + ' GiB', onInput: (v) => { S.gib = v; reset(); } });
      UI.toggle(ctl, { label: 'páginas compactas do drafter', value: true, onChange: (v) => { S.compact = v; reset(); } });
      const MAXIDS = Math.floor((16 * GI) / PER_ID);
      V.fig(fig, (g, w, h, t) => {
        if (w < 160) return;
        const C = V.C(); const narrow = w < 600; const pad = 14, gw = w - 2 * pad;
        if (S.t0 == null) S.t0 = t;
        const el = t - S.t0;
        let y = 18;
        V.text(g, narrow ? '1 ID = 3584 tokens × 11 camadas' : '1 ID = 3584 tokens × 11 camadas × (656 B MLA + 33 B indexer) = 27.163.136 B ≈ 25,9 MiB', pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: gw });
        if (narrow) { y += 15; V.text(g, '× (656 + 33) B = 27.163.136 B ≈ 25,9 MiB', pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: gw }); }
        y += 14;
        if (S.view === 'req') {
          const ids = Math.floor((S.gib * GI) / PER_ID), per = S.compact ? REQ_ON : REQ_OFF, other = per - MLA_850;
          const cols = narrow ? 30 : 48, cs = gw / cols, rows = Math.ceil(MAXIDS / cols);
          const cyc = 7, pe = el % cyc, shown = Math.min(ids, Math.floor(pe * 520));
          for (let k = 0; k < MAXIDS; k++) {
            const cx = pad + (k % cols) * cs, cy = y + Math.floor(k / cols) * cs;
            if (k >= ids) { V.box(g, cx + 1.5, cy + 1.5, cs - 3, cs - 3, { r: 2, fill: 'transparent', stroke: V.alpha(C.muted, 0.18), lw: 1 }); continue; }
            const r = Math.floor(k / per), j = k % per, full = (r + 1) * per <= ids;
            if (k >= shown) { V.box(g, cx + 1, cy + 1, cs - 2, cs - 2, { r: 2, fill: V.alpha(C.muted, 0.12) }); continue; }
            const isMla = j < MLA_850, col = isMla ? C.a : C.b;
            if (full) V.box(g, cx + 1, cy + 1, cs - 2, cs - 2, { r: 2, fill: V.alpha(col, r % 2 ? 0.62 : 0.9) });
            else { g.setLineDash([2, 2]); V.box(g, cx + 1.5, cy + 1.5, cs - 3, cs - 3, { r: 2, fill: V.alpha(col, 0.42), stroke: V.alpha(C.red, 0.6 + 0.3 * Math.sin(t * 4)), lw: 1.2 }); g.setLineDash([]); }
          }
          V.text(g, 'pool de ' + br(S.gib, 1) + ' GiB = ' + ids + ' IDs · contornos vazios: até 16 GiB (' + MAXIDS + ')', pad + gw, y + rows * cs + 14, { size: 11, font: 'mono', color: C.muted, align: 'right', maxW: gw });
          let yy = y + rows * cs + 40;
          const x = ids / per;
          // barra de uma requisição
          V.text(g, 'UMA REQUISIÇÃO DE 850k = ' + per + ' IDs = ' + br((per * PER_ID) / GI, 2) + ' GiB', pad, yy, { size: 11, font: 'mono', weight: 700, color: C.ink, maxW: gw });
          yy += 8;
          const sc = gw / REQ_OFF;
          V.box(g, pad, yy, MLA_850 * sc, 16, { r: 4, fill: V.alpha(C.a, 0.85) });
          V.box(g, pad + MLA_850 * sc, yy, other * sc, 16, { r: 4, fill: V.alpha(C.b, 0.85) });
          if (S.compact) { g.setLineDash([3, 3]); V.box(g, pad + per * sc, yy, (REQ_OFF - per) * sc, 16, { r: 4, fill: 'transparent', stroke: V.alpha(C.muted, 0.6), lw: 1 }); g.setLineDash([]); }
          if (MLA_850 * sc > 90) V.text(g, MLA_850 + ' MLA', pad + (MLA_850 * sc) / 2, yy + 9, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) });
          if (other * sc > 60) V.text(g, other + ' outros', pad + MLA_850 * sc + (other * sc) / 2, yy + 9, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) });
          if (S.compact && (REQ_OFF - per) * sc > 110) V.text(g, '−' + (REQ_OFF - per) + ' (compactas)', pad + per * sc + ((REQ_OFF - per) * sc) / 2, yy + 9, { size: 11, align: 'center', base: 'middle', color: C.muted });
          yy += 34;
          const txt = [
            [C.a, 'Menta: ' + MLA_850 + ' IDs de MLA (⌈850.000 / 3584⌉), os únicos que crescem com o contexto.'],
            [C.b, 'Lavanda: ~' + other + ' IDs de Mamba ×4, kpool e drafter, fixos por requisição' + (S.compact ? ': as páginas compactas (blocos de 896) cortaram ~' + (REQ_OFF - REQ_ON) + ' IDs, todos do drafter.' : ': com blocos de 64, o drafter responde pela maior parte.')],
            [x < 1 ? C.red : C.e, ids + ' / ' + per + ' = ' + br(x, 2) + '× de 850k ≈ ' + fmtInt(x * 850000) + ' tokens' + (x < 1 ? ': nem uma requisição cheia cabe, e o boot recusa.' : '.') + ' Totais por requisição: cálculo nosso a partir dos pools medidos.'],
          ];
          txt.forEach(([col, s]) => { V.dot(g, pad + 4, yy - 4, 3.5, col); yy += para(g, s, pad + 14, yy, gw - 14, { size: 12, color: C.ink, lh: 16 }) + 4; });
          fitH(fig, Math.ceil(yy + 6), Math.ceil(yy + 6), w);
        } else {
          y += 6;
          V.text(g, narrow ? 'REQUISIÇÃO DE 1M · BOOT DE 2026-08-31' : 'UMA REQUISIÇÃO DE 1M · BOOT DE 2026-08-31 (MNBT 2048, DRAFTER EM BLOCOS DE 64)', pad, y, { size: 11, font: 'mono', weight: 700, color: C.ink, maxW: gw });
          y += 8;
          const parts = [[280, C.a, 'MLA 280'], [1, C.c, ''], [36, C.f, 'Mamba 4×9'], [97, C.b, 'drafter 97']];
          const sc = gw / 414; let xx = pad;
          parts.forEach(([n, col, lab], k) => { const ww = n * sc; V.box(g, xx, y, Math.max(2, ww - 1), 20, { r: 4, fill: V.alpha(col, 0.85) }); if (lab && ww > tw(g, lab, 11, 'sans', 700) + 8) V.text(g, lab, xx + ww / 2, y + 11, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) }); xx += ww; });
          V.text(g, '= 414 IDs · kpool: 1', pad + gw, y + 36, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          y += 60;
          V.text(g, narrow ? 'SEGMENTO CACHEADO DE 3584 TOKENS = 38 IDs' : 'UM SEGMENTO DE 3584 TOKENS NO CACHE DE PREFIXO = 38 IDs', pad, y, { size: 11, font: 'mono', weight: 700, color: C.ink, maxW: gw });
          y += 10;
          const segs = [[1, C.a, 'MLA'], [4, C.f, 'Mamba'], [33, C.b, 'drafter']];
          const cols = narrow ? 19 : 38, cs = Math.min(narrow ? 17 : 20, gw / cols);
          let k = 0; const hot = Math.floor(el * 8) % 44;
          segs.forEach(([n, col]) => { for (let q = 0; q < n; q++, k++) { const cx = pad + (k % cols) * cs, cy = y + Math.floor(k / cols) * cs; V.box(g, cx + 1, cy + 1, cs - 2, cs - 2, { r: 3, fill: V.alpha(col, k === hot ? 1 : 0.75) }); if (k === hot) V.glow(g, cx + cs / 2, cy + cs / 2, cs, col, 0.8); } });
          y += Math.ceil(38 / cols) * cs + 12;
          let lx = pad;
          segs.forEach(([n, col, lab]) => { V.box(g, lx, y - 9, 10, 10, { r: 3, fill: col }); const s = lab + ' ' + n; V.text(g, s, lx + 15, y, { size: 11.5, color: C.ink }); lx += tw(g, s, 11.5) + 34; });
          y += 22;
          const txt = [
            [C.b, 'O drafter gastava 33 dos 38 IDs de cada segmento, e seus blocos iam para a cauda do LRU, empurrando MLA e Mamba para o despejo. O overlay nº 9 dá ao grupo do drafter uma retenção própria (com SWA=0, só as fronteiras alcançáveis).'],
            [C.a, 'Com 642 IDs livres, o log estimava ~57k tokens de conversa cacheável sem nada rodando: 16 segmentos × 3584. O “GPU KV cache size” do vLLM é outra coisa: concorrência × max_model_len.'],
            [C.muted, S.compact ? 'Com páginas compactas o drafter usa blocos de 896 e o lookup termina na fronteira, então um segmento custa menos IDs; o log acima é do layout de 64.' : 'Sem páginas compactas, este é o custo por segmento: blocos de 64 no drafter.'],
          ];
          txt.forEach(([col, s]) => { V.dot(g, pad + 4, y - 4, 3.5, col); y += para(g, s, pad + 14, y, gw - 14, { size: 12, color: C.ink, lh: 16 }) + 4; });
          fitH(fig, Math.ceil(y + 6), Math.ceil(y + 6), w);
        }
      });
    })();

    /* ============================================ F2 · mapa de decisões */
    (function () {
      const fig = root.querySelector('#fig-decisoes');
      const ROWS = [
        { name: 'LINHA DOS PESOS', c: 'd', st: [
          ['BF16/FP8', 'não cabem', 'BF16 ocupa ~642,6 GB e o FP8 oficial 328,3 GB: nenhum cabe em 2 × 128 GB de memória unificada.'],
          ['EXL3 4 bpw', 'experts', 'Só os experts roteados (~97% dos bytes) em EXL3 K4: 175,6 GB e KLD 0,0246, igual ao FP8 oficial.'],
          ['método', 'exl3', 'O vLLM não conhece EXL3: o overlay registra o método e o launcher passa --quantization exl3 (nunca marlin).'],
          ['kernel', 'sm_121a', 'A extensão CUDA do ExLlamaV3 (pin c5d9c657) é compilada para sm_121a com os kernels E2, E3 e thin-decode.'],
          ['imagem', 'própria', 'Compilar dentro da imagem exige Dockerfile próprio, pin por digest e um recipe stamp para detectar drift.'],
        ] },
        { name: 'LINHA DA ESPECULAÇÃO', c: 'c', st: [
          ['decode', '= banda', 'Cada token de decode lê todos os pesos ativos: o gargalo é a banda de memória, não o cálculo.'],
          ['DFlash2', 'k = 7', 'Um drafter de difusão em bloco propõe 7 tokens; o alvo verifica todos num passo (aceitação 0,96 em texto estruturado).'],
          ['8 posições', 'por seq', 'Cada sequência em decode vira 1 âncora + 7 rascunhos = 8 posições verificadas por passo.'],
          ['graphs', '8·16·24·32', 'Com até 4 sequências, os passos de decode têm 8, 16, 24 ou 32 tokens, e esses tamanhos precisam ter CUDA graph.'],
          ['TEMP_ROWS', '= 32', 'O MoE fundido tem buffer para 32 linhas por expert: 4 × 8 cabem, e o decode roda num único lançamento graph-safe.'],
        ] },
        { name: 'LINHA DISTRIBUÍDA', c: 'e', st: [
          ['2 nós', '1 GPU cada', 'Dois GB10, uma GPU em cada, ligados por um cabo ConnectX-7 em RoCEv2.'],
          ['TP = 2', 'mp', 'Tensor parallel de grau 2, executor multiprocessing (sem Ray), worker --headless sem API.'],
          ['mesmo', 'ambiente', 'Os knobs de política GLM53_* são encaminhados idênticos aos dois ranks.'],
          ['mesma', 'ordem', 'GLM53_OVERLAY_ORDER aplica os 25 overlays (26 desde o #203) com os mesmos bytes e na mesma ordem nos dois containers.'],
          ['mesmos', 'graphs', 'Um rank que captura CUDA graphs diferentes trava o NCCL depois do PIECEWISE.'],
        ] },
        { name: 'LINHA DA MEMÓRIA (UMA)', c: 'a', st: [
          ['CPU + GPU', '128 GB', 'Memória unificada: o que o vLLM pega sai do SO, do page cache e do encoder de visão.'],
          ['util', '0,85', 'Com o pool fixo, a util não dimensiona mais o KV: é o piso de memória livre do boot (0,85 pede 103,4 GiB; 0,88 bootou 1 de 4). No regime perfilado de antes, cada 0,01 eram ~1,2 GiB e 0,87 derrubou um head.'],
          ['Instant', 'Tensor', 'O loader direct-I/O carrega 164 GiB em ~65–70 s (contra ~290–300 s), mas deixa ~4,4–5,6 GiB a menos para o KV.'],
          ['KV fixo', '11 GiB', 'Perfilado, o pool ficaria em ~12,4 GiB, abaixo dos 13,56 GiB (536 IDs de bloco) de uma requisição de 850k: por isso ele é fixado.'],
          ['páginas', 'compactas', 'Blocos de 896 tokens no drafter cortam a requisição de 850k de ~536 para ~299 IDs: 11 GiB (434 IDs) guardam 1.233.779 tokens, 1,45×.'],
        ] },
        { name: 'LINHA DA ATENÇÃO', c: 'b', st: [
          ['MLA NoPE', '+ SM121', 'As 11 camadas MLA não têm RoPE (qk_rope_head_dim=0), e o GB10 só tem o kernel FLASHINFER_MLA_SPARSE_SM120.'],
          ['1 kernel', 'sparse', 'Esse kernel só aceita o layout empacotado fp8_ds_mla: KV BF16 ou NVFP4 não têm kernel esparso no SM12x.'],
          ['KV dtype', 'fp8', '--kv-cache-dtype fp8 vira o registro fp8_ds_mla nas camadas MLA.'],
          ['pad', '512→576', 'O overlay acrescenta 64 zeros de RoPE à query e ao k_pe. É exato, porque zeros não mudam o produto QK.'],
          ['656 B', '/token', '512 B FP8 + 16 B de escalas + 128 B de RoPE (zeros): 24% a mais que os 528 B necessários.'],
        ] },
      ];
      const T = [
        { a: [0, 3], b: [1, 4], d: 'TEMP_ROWS é um parâmetro do kernel fundido do EXL3, mas o valor vem da especulação: 4 seqs × 8 posições = 32.' },
        { a: [1, 3], b: [2, 4], d: 'Os tamanhos de captura precisam ser idênticos nos dois ranks. Se os graphs divergem, o NCCL trava.' },
        { a: [3, 3], b: [4, 4], d: 'O pool de 11 GiB só faz sentido em IDs de bloco: 1 ID = 3584 tokens × 11 camadas × (656 + 33) B ≈ 25,9 MiB, e o registro de 656 B vem da atenção.' },
      ];
      const steps = [
        { t: 'Linha dos pesos', d: 'O modelo não cabe em BF16 nem em FP8. Quantizar só os experts roteados em EXL3 resolve os bytes, mas obriga a registrar um método de quantização novo e a compilar kernels para o SM121: é preciso manter uma imagem própria.' },
        { t: 'Linha da especulação', d: 'O decode é limitado pela banda. O DFlash2 propõe 7 tokens, e o alvo verifica 8 posições por sequência. Isso fixa os tamanhos de CUDA graph e o buffer do MoE fundido.' },
        { t: 'Linha distribuída', d: 'Dois nós em TP=2 só funcionam com ranks idênticos: mesmo ambiente, mesmos overlays na mesma ordem e, por consequência, os mesmos graphs.' },
        { t: 'Linha da memória (UMA)', d: 'CPU e GPU dividem a RAM. O loader rápido come KV, então o pool é fixado em 11 GiB; com isso a util (0,85) vira só o piso de memória livre do boot, e as páginas compactas do drafter fazem 850k caber em IDs de bloco.' },
        { t: 'Linha da atenção', d: 'MLA sem RoPE numa GPU que só tem um kernel de sparse-MLA: o KV precisa ser <code>fp8_ds_mla</code>, e a query ganha 64 zeros para caber na geometria de 576.' },
        { t: 'Baldeações', d: 'Onde duas restrições se encontram: o buffer do kernel EXL3 depende da especulação, os graphs precisam ser iguais nos ranks e o pool em GiB só faz sentido em IDs de bloco, cujo tamanho vem do registro da atenção.' },
      ];
      let user = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => { user = null; } });
      V.fig(fig, (g, w, h, t, dt, S, H) => {
        const C = V.C();
        const narrow = w < 560;
        const pad = 14, detailH = narrow ? 108 : 74;
        const rowH = (h - 6 - detailH) / 5;
        const x0 = pad + (narrow ? 26 : 34), x1 = w - pad - (narrow ? 26 : 34);
        const SX = (j) => x0 + (j * (x1 - x0)) / 4;
        const SY = (i) => 8 + i * rowH + 24;
        const act = st.i;
        const el = st.elapsed();
        const u = (el / 6.3) % 1;
        if (H.pointer.clicked) {
          let best = null, bd = 26;
          for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { const d = Math.hypot(H.pointer.x - SX(j), H.pointer.y - SY(i)); if (d < bd) { bd = d; best = [i, j]; } }
          if (best) { st.stop(); if (st.i !== best[0]) st.go(best[0]); user = best; }
        }
        let sel = user;
        if (!sel && act < 5) sel = [act, Math.min(4, Math.floor(u * 5))];
        const trIdx = act === 5 ? Math.floor(el / 2.4) % T.length : -1;
        T.forEach((tr, k) => {
          const on = act === 5;
          if (narrow && !on) return; // no mobile, as baldeações só aparecem no passo 6
          const A0 = [SX(tr.a[1]), SY(tr.a[0])], B0 = [SX(tr.b[1]), SY(tr.b[0])];
          const col = on ? (k === trIdx ? C.ink : V.alpha(C.ink, narrow ? 0.3 : 0.45)) : V.alpha(C.muted, 0.3);
          g.strokeStyle = col; g.lineWidth = on && k === trIdx ? 2.6 : 1.4; g.setLineDash([4, 4]); g.lineDashOffset = -t * 18;
          g.beginPath(); g.moveTo(A0[0], A0[1]);
          // curva em S: sai pelo trilho, desce pelo corredor entre as colunas e chega pelo trilho de baixo
          const xm = (A0[0] + B0[0]) / 2;
          g.bezierCurveTo(xm, A0[1], xm, B0[1], B0[0], B0[1]);
          g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
        });
        const mw = (x1 - x0) / 4 - 4;
        ROWS.forEach((R, i) => {
          const col = C[R.c]; const on = act === i || act === 5; const y = SY(i);
          V.text(g, R.name, pad, y - 13, { size: 11, font: 'mono', weight: 700, color: col, alpha: on ? 1 : 0.5 });
          V.line(g, x0, y, x1, y, { color: V.alpha(col, on ? 1 : 0.28), lw: on ? 5 : 3 });
          if (on) {
            const tu = act === 5 ? ((t * 0.1 + i * 0.21) % 1) : u;
            const px = V.lerp(x0, x1, tu);
            V.box(g, px - 15, y - 8, 30, 16, { r: 6, fill: col });
            for (let q = 0; q < 3; q++) V.box(g, px - 11 + q * 8, y - 4, 6, 6, { r: 1.5, fill: V.alpha(C.bg, 0.9) });
          }
          for (let j = 0; j < 5; j++) {
            const x = SX(j); const isSel = sel && sel[0] === i && sel[1] === j;
            if (isSel) V.glow(g, x, y, 22, col, 0.9);
            g.beginPath(); g.arc(x, y, on ? 8 : 6, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill();
            g.lineWidth = on ? 3 : 2; g.strokeStyle = V.alpha(col, on ? 1 : 0.35); g.stroke();
            const lc = on ? C.ink : C.muted;
            V.text(g, R.st[j][0], x, y + 23, { size: 11, align: 'center', color: lc, weight: isSel ? 700 : 500, maxW: mw, alpha: on ? 1 : 0.65 });
            V.text(g, R.st[j][1], x, y + 37, { size: 11, align: 'center', color: lc, font: 'mono', maxW: mw, alpha: on ? 1 : 0.65 });
          }
        });
        const dy = h - detailH + 2;
        V.box(g, pad, dy, w - 2 * pad, detailH - 8, { r: 10, fill: V.alpha(C.ink, 0.05), stroke: V.alpha(C.muted, 0.3) });
        let title = '', body = '', col = C.ink;
        if (act === 5 && !user) { title = 'BALDEAÇÃO ' + (trIdx + 1) + ' DE 3'; body = T[trIdx].d; }
        else if (sel) { const R = ROWS[sel[0]]; title = R.name + ' · ESTAÇÃO ' + (sel[1] + 1); body = R.st[sel[1]][2]; col = C[R.c]; }
        V.text(g, title, pad + 12, dy + 18, { size: 11, font: 'mono', weight: 700, color: col });
        if (!narrow) V.text(g, user ? 'estação fixada' : 'clique numa estação para fixá-la', w - pad - 12, dy + 18, { size: 11, align: 'right', color: C.muted });
        para(g, body, pad + 12, dy + 38, w - 2 * pad - 24, { size: 12.5, color: C.ink, lh: 17 });
      });
    })();

    /* ============================================ F3 · topologia */
    (function () {
      const fig = root.querySelector('#fig-topologia');
      const S = { topo: 'tp2', ssh: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Topologia', options: [{ v: 'tp2', label: 'TP=2 (receita)' }, { v: 'tp3', label: 'TP=3' }, { v: 'tp4', label: 'TP=4' }], value: 'tp2', onChange: (v) => { S.topo = v; } });
      UI.toggle(ctl, { label: 'tráfego de orquestração (SSH)', value: false, onChange: (v) => { S.ssh = v; } });
      const flow = (g, ax, ay, bx, by, t, col, n, speed, r) => { for (let k = 0; k < n; k++) { const u = (k / n + t * speed) % 1; V.dot(g, V.lerp(ax, bx, u), V.lerp(ay, by, u), r || 2.6, col); } };
      const nodeBox = (g, C, x, y, nw, nh, title, sub, col) => {
        V.box(g, x, y, nw, nh, { r: 14, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.65), lw: 1.5 });
        V.text(g, title, x + 12, y + 20, { size: 12.5, weight: 700, color: C.ink, maxW: nw - 24 });
        if (sub) V.text(g, sub, x + 12, y + 36, { size: 11, font: 'mono', color: C.muted, maxW: nw - 24 });
      };
      const proc = (g, C, x, y, pw, ph, lab, col, dashed) => {
        if (dashed) g.setLineDash([4, 4]);
        V.box(g, x, y, pw, ph, { r: 8, fill: dashed ? V.alpha(C.muted, 0.04) : V.alpha(col, 0.2), stroke: V.alpha(col, dashed ? 0.55 : 0.9), lw: 1.3 });
        g.setLineDash([]);
        V.text(g, lab, x + pw / 2, y + ph / 2 + 1, { size: 11.5, align: 'center', base: 'middle', color: dashed ? C.muted : C.ink, weight: 500, maxW: pw - 10 });
      };
      function tp2(g, w, h, t, C) {
        const narrow = w < 680, pad = 14, inf = !S.ssh;
        if (!narrow) {
          const top = 64, nh = h - top - 20, nw = (w - 2 * pad) * 0.4;
          const hx = pad, wx = w - pad - nw;
          nodeBox(g, C, hx, top, nw, nh, 'HEAD · rank 0 · 10.0.0.1', 'container glm53-exl3-head', C.a);
          nodeBox(g, C, wx, top, nw, nh, 'WORKER · rank 1 · 10.0.0.2', 'container glm53-exl3-worker', C.b);
          const pw = nw - 24, ph = 32, rows = [top + 56, top + 100, top + 144];
          proc(g, C, hx + 12, rows[0], pw, ph, 'API server OpenAI · :8888', C.d);
          proc(g, C, hx + 12, rows[1], pw, ph, 'EngineCore · scheduler + KV', C.b);
          proc(g, C, hx + 12, rows[2], pw, ph, 'GPU worker · rank 0 (GB10)', C.a);
          proc(g, C, wx + 12, rows[0], pw, ph, 'sem API (--headless)', C.muted, true);
          proc(g, C, wx + 12, rows[1], pw, ph, 'sem scheduler próprio', C.muted, true);
          proc(g, C, wx + 12, rows[2], pw, ph, 'GPU worker · rank 1 (GB10)', C.a);
          const lx0 = hx + nw, lx1 = wx, mid = (lx0 + lx1) / 2;
          const ly = rows[2] + ph / 2;
          V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.a, inf ? 0.9 : 0.35), lw: 4 });
          if (inf) { flow(g, lx0 + 4, ly - 5, lx1 - 4, ly - 5, t, C.a, 7, 0.45); flow(g, lx1 - 4, ly + 5, lx0 + 4, ly + 5, t, C.a, 7, 0.45); }
          V.text(g, 'CX7 · RoCEv2', mid, ly - 16, { size: 11.5, font: 'mono', align: 'center', color: C.ink, weight: 600 });
          V.text(g, '2 all-reduces/camada', mid, ly + 24, { size: 11, align: 'center', color: C.muted, maxW: lx1 - lx0 - 6 });
          V.text(g, '(após atenção e MoE)', mid, ly + 38, { size: 11, align: 'center', color: C.muted, maxW: lx1 - lx0 - 6 });
          // interfaces fixadas por rank, nas pontas do cabo
          V.text(g, 'CX7 enp1s0f1np1 · HCA rocep1s0f1', hx + nw - 14, rows[2] + ph + 16, { size: 11, font: 'mono', align: 'right', color: C.a, maxW: pw });
          V.text(g, 'CX7 enp1s0f0np0 · HCA rocep1s0f0', wx + 14, rows[2] + ph + 16, { size: 11, font: 'mono', color: C.a, maxW: pw });
          const ry = rows[1] + ph / 2;
          V.line(g, lx0, ry, lx1, ry, { color: V.alpha(C.muted, 0.6), lw: 1.2, dash: [2, 4] });
          V.text(g, 'rendezvous :29521', mid, ry - 7, { size: 11, font: 'mono', align: 'center', color: C.muted });
          // memória por rank
          const my = rows[2] + ph + 40;
          [hx, wx].forEach((nx) => {
            V.text(g, 'MEMÓRIA DO RANK (UMA)', nx + 14, my, { size: 11, font: 'mono', weight: 700, color: C.muted });
            const segW = pw, sy0 = my + 8;
            const parts = [[79.65, C.d, 'pesos ~80'], [3.29, C.f, ''], [11, C.a, 'KV'], [21.55, C.b, 'resto'], [6.2, C.e, '']];
            let acc = 0;
            parts.forEach(([v, col, lab]) => { const x0 = nx + 12 + (acc / 121.69) * segW, ww = (v / 121.69) * segW; V.box(g, x0, sy0, Math.max(1, ww - 1), 16, { r: 3, fill: V.alpha(col, 0.7) }); if (lab && ww > tw(g, lab, 11, 'sans', 600) + 6) V.text(g, lab, x0 + ww / 2, sy0 + 9, { size: 11, weight: 600, align: 'center', base: 'middle', color: onFill(C) }); acc += v; });
            V.text(g, 'latente MLA replicado · ≥ 6,2 GiB livres sob carga', nx + 14, sy0 + 32, { size: 11, color: C.muted, maxW: pw - 4 });
          });
          // pesos
          const dky = top + nh - 58;
          V.box(g, hx + 12, dky, pw, 40, { r: 8, fill: V.alpha(C.e, 0.12), stroke: V.alpha(C.e, 0.6) });
          V.text(g, 'cache HF · 164 GiB · 83 shards', hx + 24, dky + 17, { size: 11.5, color: C.ink, maxW: pw - 20 });
          V.text(g, 'MODEL_REVISION 76c0b517…', hx + 24, dky + 32, { size: 11, font: 'mono', color: C.muted, maxW: pw - 20 });
          V.box(g, wx + 12, dky, pw, 40, { r: 8, fill: V.alpha(C.e, 0.06), stroke: V.alpha(C.e, 0.45) });
          V.text(g, 'mesmo snapshot (rsync ou NFS ro)', wx + 24, dky + 17, { size: 11.5, color: C.ink, maxW: pw - 20 });
          V.text(g, 'HF_HUB_OFFLINE=1 no container', wx + 24, dky + 32, { size: 11, font: 'mono', color: C.muted, maxW: pw - 20 });
          V.line(g, lx0, dky + 20, lx1, dky + 20, { color: V.alpha(C.e, 0.7), lw: 1.6, dash: [6, 4], dashOffset: -t * 14 });
          V.text(g, 'NFS · nconnect=8', mid, dky + 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          // cliente
          const cw = Math.min(230, nw - 20);
          V.box(g, hx + 12, 14, cw, 30, { r: 9, fill: V.alpha(C.d, 0.16), stroke: C.d, label: 'cliente → /v1/chat/completions', size: 11.5 });
          const axc = hx + nw - 36;
          V.arrow(g, hx + 12 + cw, 29, axc, 29, { color: V.alpha(C.d, 0.8), lw: 1.4, head: 0 });
          V.arrow(g, axc, 29, axc, rows[0] - 2, { color: C.d, lw: 1.6 });
          if (inf) flow(g, axc, 32, axc, rows[0] - 4, t, C.d, 3, 0.6, 2.4);
          if (S.ssh) {
            const sw = 186, sx = mid - sw / 2;
            V.box(g, sx, 12, sw, 34, { r: 9, fill: V.alpha(C.d, 0.22), stroke: C.d, label: 'start.sh (host do head)', size: 12, weight: 700 });
            const tx = wx + nw / 2, ty = top;
            V.arrow(g, sx + sw, 30, tx, ty - 2, { color: C.d, lw: 2, curve: -20 });
            flow(g, sx + sw, 30, tx, ty - 2, t, C.d, 5, 0.35, 3);
            V.text(g, 'ssh BatchMode · scp · docker run/kill/load', wx + nw - 4, 30, { size: 11, align: 'right', color: C.ink, weight: 600, maxW: nw - 20 });
          }
          return;
        }
        // estreito: head em cima, worker embaixo
        const top = 42, gapV = 72, nh = (h - top - gapV - 10) / 2, nw = w - 2 * pad;
        const hx = pad, hy = top, wy = top + nh + gapV;
        nodeBox(g, C, hx, hy, nw, nh, 'HEAD · rank 0 · 10.0.0.1', 'glm53-exl3-head', C.a);
        nodeBox(g, C, hx, wy, nw, nh, 'WORKER · rank 1 · 10.0.0.2', 'glm53-exl3-worker', C.b);
        const pw = nw - 24, ph = 28;
        const hr = [hy + nh - 3 * (ph + 6) - 4, hy + nh - 2 * (ph + 6) - 4, hy + nh - (ph + 6) - 4];
        proc(g, C, hx + 12, hr[0], pw, ph, 'API server · :8888', C.d);
        proc(g, C, hx + 12, hr[1], pw, ph, 'EngineCore · scheduler', C.b);
        proc(g, C, hx + 12, hr[2], pw, ph, 'GPU worker · rank 0', C.a);
        const wr = [wy + 46, wy + 46 + ph + 6];
        proc(g, C, hx + 12, wr[0], pw, ph, 'GPU worker · rank 1', C.a);
        proc(g, C, hx + 12, wr[1], pw, ph, 'sem API (--headless)', C.muted, true);
        const lx = w - pad - 44, ly0 = hr[2] + ph, ly1 = wr[0];
        V.line(g, lx, ly0, lx, ly1, { color: V.alpha(C.a, inf ? 0.9 : 0.35), lw: 4 });
        if (inf) { flow(g, lx - 5, ly0 + 2, lx - 5, ly1 - 2, t, C.a, 4, 0.5); flow(g, lx + 5, ly1 - 2, lx + 5, ly0 + 2, t, C.a, 4, 0.5); }
        if (!S.ssh) {
          V.text(g, 'CX7 · RoCEv2 · 2 all-reduces/camada', lx - 12, hy + nh + 20, { size: 11, color: C.ink, weight: 600, align: 'right', maxW: lx - 12 - pad });
          V.text(g, 'head enp1s0f1np1 · rocep1s0f1', lx - 12, hy + nh + 38, { size: 11, font: 'mono', color: C.a, align: 'right', maxW: lx - 12 - pad });
          V.text(g, 'worker enp1s0f0np0 · rocep1s0f0', lx - 12, hy + nh + 54, { size: 11, font: 'mono', color: C.b, align: 'right', maxW: lx - 12 - pad });
        }
        // memória do rank (igual nos dois)
        const my2 = wr[1] + ph + 20;
        if (my2 + 26 <= wy + nh) {
          V.text(g, 'MEMÓRIA DO RANK (IGUAL NOS 2)', hx + 14, my2, { size: 11, font: 'mono', weight: 700, color: C.muted });
          const parts = [[79.65, C.d, 'pesos ~80'], [3.29, C.f, ''], [11, C.a, 'KV'], [21.55, C.b, 'resto'], [6.2, C.e, '']];
          let acc = 0;
          parts.forEach(([v, col, lab]) => { const x0 = hx + 12 + (acc / 121.69) * pw, ww = (v / 121.69) * pw; V.box(g, x0, my2 + 8, Math.max(1, ww - 1), 16, { r: 3, fill: V.alpha(col, 0.7) }); if (lab && ww > tw(g, lab, 11, 'sans', 600) + 6) V.text(g, lab, x0 + ww / 2, my2 + 17, { size: 11, weight: 600, align: 'center', base: 'middle', color: onFill(C) }); acc += v; });
        }
        V.box(g, pad, 8, Math.min(230, nw), 26, { r: 8, fill: V.alpha(C.d, 0.16), stroke: C.d, label: 'cliente → :8888', size: 11.5 });
        if (S.ssh) {
          const px = pad + 6;
          g.strokeStyle = C.d; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(px + 6, hy + 50); g.quadraticCurveTo(px - 10, (hy + wy) / 2 + nh / 2, px + 6, wy + 20); g.stroke(); g.setLineDash([]);
          for (let k = 0; k < 4; k++) { const u = (k / 4 + t * 0.35) % 1; const p = [(1 - u) * (1 - u) * (px + 6) + 2 * (1 - u) * u * (px - 10) + u * u * (px + 6), (1 - u) * (1 - u) * (hy + 50) + 2 * (1 - u) * u * ((hy + wy) / 2 + nh / 2) + u * u * (wy + 20)]; V.dot(g, p[0], p[1], 3, C.d); }
          V.box(g, pad + 18, hy + nh + 8, nw - 36, 36, { r: 8, fill: V.alpha(C.d, 0.18), stroke: C.d });
          V.text(g, 'SSH BatchMode: scp, docker run/kill/load', pad + 30, hy + nh + 30, { size: 11, color: C.ink, weight: 600, maxW: nw - 60 });
        }
      }
      function ring(g, w, h, t, C, n) {
        const narrow = w < 680, pad = 14;
        const cx = w / 2, cy = narrow ? 112 : 150, R = narrow ? Math.min(w * 0.3, 88) : Math.min(w * (n === 4 ? 0.22 : 0.18), n === 4 ? 118 : 110);
        const pts = [];
        for (let i = 0; i < n; i++) { const a = -Math.PI / 2 + (i * 2 * Math.PI) / n; pts.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]); }
        if (n === 3) {
          for (let i = 0; i < 3; i++) {
            const a = pts[i], b = pts[(i + 1) % 3];
            const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
            const s0 = [a[0] + ux * 40, a[1] + uy * 30], s1 = [b[0] - ux * 44, b[1] - uy * 32];
            V.arrow(g, s0[0], s0[1], s1[0], s1[1], { color: V.alpha(C.a, 0.9), lw: 2.6, curve: 14, head: 10 });
            flow(g, s0[0], s0[1], s1[0], s1[1], t, C.a, 4, 0.4, 2.6);
          }
          V.text(g, 'anel dirigido', cx, cy - 2, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
          V.text(g, 'Port0 → Port1', cx, cy + 14, { size: 11, align: 'center', font: 'mono', color: C.muted });
        } else {
          pts.forEach((p) => { V.line(g, p[0], p[1], cx, cy, { color: V.alpha(C.d, 0.6), lw: 2.4 }); flow(g, p[0], p[1], cx, cy, t, C.d, 3, 0.35, 2.4); flow(g, cx, cy, p[0], p[1], t + 0.5, C.a, 3, 0.35, 2.2); });
          V.box(g, cx - 46, cy - 16, 92, 32, { r: 8, fill: C.bg });
          V.box(g, cx - 46, cy - 16, 92, 32, { r: 8, fill: V.alpha(C.d, 0.2), stroke: C.d, label: 'switch', size: 12, weight: 600 });
        }
        pts.forEach((p, i) => {
          const nw = 98, nh = 42;
          V.box(g, p[0] - nw / 2, p[1] - nh / 2, nw, nh, { r: 10, fill: C.bg });
          V.box(g, p[0] - nw / 2, p[1] - nh / 2, nw, nh, { r: 10, fill: V.alpha(i === 0 ? C.a : C.b, 0.2), stroke: i === 0 ? C.a : C.b, lw: 1.6 });
          V.text(g, 'rank ' + i, p[0], p[1] - 3, { size: 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, i === 0 ? 'head · API' : 'headless', p[0], p[1] + 12, { size: 11, align: 'center', font: 'mono', color: C.muted });
        });
        const B = n === 3 ? [
          'Cabeamento em anel dirigido: Port0 de cada nó → Port1 do próximo (um anel espelhado deixa um par sem conexão).',
          '64 cabeças não dividem por 3: --hf-overrides com 66 (22 por rank).',
          'Expert parallel: 96 dos 288 experts por rank (o trellis de 2048 não divide por 3).',
          'Drafter só no rank 0 (draft TP=1; GQA 32/8 → 36/9). Controle pela LAN de gerência, dados via RoCE.',
          'Medido em 2026-09-14: 87,8 tok/s em texto estruturado, contra 73,4 no TP2 com os mesmos prompts.',
        ] : [
          '4 Sparks num switch CRS812 (2 × 200G + 2 × 100G). Experimental, não testado pelo mantenedor.',
          'k=3 (aceita 53% contra 30% com k=7), 8 seqs, MNBT 2048, util 0,75, draft TP 4.',
          'Decode 1,45× o de 2 nós; prefill frio só +29% (1162 contra 901 tok/s a 282k): limitado pela rede.',
          'Soak de 150 min travou 3 vezes em 3 (31–78 min) com prefill de 96k e 6–7 streams especulativas.',
        ];
        let y = cy + R + (narrow ? 44 : 50);
        const bw = w - 2 * pad - 16;
        B.forEach((s) => { V.dot(g, pad + 5, y - 4, 3, n === 3 ? C.a : C.d); y += para(g, s, pad + 16, y, bw, { size: narrow ? 11.5 : 12.5, color: C.ink, lh: narrow ? 15 : 17 }) + (narrow ? 3 : 5); });
      }
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 430, 456, w);
        const C = V.C();
        if (S.topo === 'tp2') tp2(g, w, h, t, C);
        else ring(g, w, h, t, C, S.topo === 'tp3' ? 3 : 4);
      });
    })();

    /* ============================================ F4 · precedência */
    (function () {
      const fig = root.querySelector('#fig-precedencia');
      const VARS = {
        MAX_MODEL_LEN: { exp: '${MAX_MODEL_LEN:-850000}', kind: ':-', def: '850000', env: '850000', caller: '262144' },
        MM_IMAGE_TOKENS: { exp: '${MM_IMAGE_TOKENS-2048}', kind: '-', def: '2048', env: '2048', caller: '1024', empty: 'O vazio sobrevive: a flag --mm-processor-kwargs é omitida e o vLLM volta aos 8.000 tokens por imagem do checkpoint, acima do cache de encoder (MNBT 7.168).' },
        DEFAULT_MAX_NEW_TOKENS: { exp: '${DEFAULT_MAX_NEW_TOKENS-65536}', kind: '-', def: '65536', env: '65536', caller: '32768', empty: 'O vazio sobrevive: sem default do servidor, uma requisição sem max_tokens volta ao comportamento stock e pode crescer o KV até preemptar outras sessões.' },
        GLM53_DRAFT_KV_COMPACT: { exp: '${GLM53_DRAFT_KV_COMPACT-1}  # sob dflash', kind: '-', def: '1', env: '0', caller: '0', empty: 'Vazio explícito é erro de operador: a validação numérica recusa antes de parar o par, em vez de adivinhar um modo.', err: true },
        ABLIT: { exp: 'ABLIT=0  # forçado após o source .env', kind: 'forced', def: '0', env: '1', caller: '1' },
        IMAGE: { exp: '${IMAGE:-ghcr.io/…:exl3-instanttensor}', kind: ':-', def: 'ghcr.io/…:exl3-instanttensor', env: 'ghcr.io/…:exl3-instanttensor', caller: 'glm53-flash-sm121:local', ident: true },
      };
      const S = { v: 'MAX_MODEL_LEN', caller: 'none', env: true };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Variável', options: Object.keys(VARS).map((k) => ({ v: k, label: k })), value: S.v, onChange: (v) => { S.v = v; } });
      UI.seg(ctl, { label: 'Chamador', options: [{ v: 'none', label: 'não exporta' }, { v: 'val', label: 'exporta valor' }, { v: 'empty', label: 'exporta vazio' }], value: 'none', onChange: (v) => { S.caller = v; } });
      UI.toggle(ctl, { label: '.env define a chave', value: true, onChange: (v) => { S.env = v; } });
      function resolve(X) {
        const c = S.caller, e = S.env;
        const r = { layer: c !== 'none' ? 0 : e ? 1 : 2, val: '', status: 'ok', note: '' };
        if (X.kind === 'forced') {
          if (c === 'val') { r.val = X.caller; r.note = 'O export do chamador é reexportado depois do ABLIT=0. Só assim a abliteração liga no TP2.'; }
          else if (c === 'empty') { r.val = '""'; r.status = 'warn'; r.note = 'Vazio: a checagem ${ABLIT:-0} = 1 falha e a abliteração fica desligada.'; }
          else { r.val = '0'; r.layer = 2; r.status = e ? 'warn' : 'ok'; r.note = e ? 'O .env diz ABLIT=1, mas o launcher força ABLIT=0 logo depois do source. Sem efeito no TP2.' : 'Default: o_proj stock.'; r.strike = e; }
          return r;
        }
        let raw;
        if (c === 'val') raw = X.caller; else if (c === 'empty') raw = ''; else if (e) raw = X.env;
        if (raw === undefined) { r.val = X.def; r.note = 'Ninguém define a chave: vale o default do script.'; }
        else if (raw === '') {
          if (X.kind === ':-') { r.val = X.def; r.note = 'O operador :- trata vazio como ausente e cai no default.'; }
          else { r.val = '""'; r.note = X.empty; r.status = X.err ? 'err' : 'warn'; }
        } else { r.val = raw; r.note = c === 'val' ? 'O export do chamador vence o .env.' : 'Vale o valor do .env.'; }
        if (X.ident && c !== 'none' && e) { r.status = 'warn'; r.note = (c === 'empty' ? 'Vazio + :- → default. ' : '') + 'Variável de identidade: o launcher avisa “NOTE: IMAGE=… from the environment overrides .env value …”.'; }
        return r;
      }
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 370, 450, w);
        const C = V.C(); const narrow = w < 520; const pad = 14;
        const name = S.v, X = VARS[name], r = resolve(X);
        const cardH = narrow ? 58 : 48, gap = 10, lx = pad + 22;
        const titles = ['3 · export do chamador', '2 · .env (source)', '1 · default do script'];
        const subs = ['snapshot antes do source .env', 'arquivo do kit', 'expansão no start.sh'];
        const vals = [
          S.caller === 'none' ? '— (não exporta)' : S.caller === 'val' ? name + '=' + X.caller : name + '=   (vazio)',
          S.env ? name + '=' + X.env : '— (chave ausente)',
          X.exp,
        ];
        const defined = [S.caller !== 'none', S.env, true];
        const cols = [C.c, C.b, C.a];
        const cys = [];
        for (let i = 0; i < 3; i++) {
          const x = lx + i * 8, y = 10 + i * (cardH + gap), cw = w - lx - pad - 16;
          cys.push(y + cardH / 2);
          const win = i === r.layer, col = cols[i];
          V.box(g, x, y, cw, cardH, { r: 10, fill: V.alpha(col, win ? 0.2 : 0.06), stroke: V.alpha(col, win ? 1 : 0.35), lw: win ? 2 : 1 });
          V.text(g, titles[i], x + 12, y + 18, { size: 12, weight: 700, color: C.ink });
          if (!narrow) V.text(g, subs[i], x + 12, y + 35, { size: 11, color: C.muted });
          const vy = narrow ? y + cardH - 14 : y + cardH / 2 + 1;
          const vx = narrow ? x + 12 : x + cw - 12;
          V.text(g, vals[i], vx, vy, { size: 12, font: 'mono', color: defined[i] ? C.ink : C.muted, align: narrow ? 'left' : 'right', base: 'middle', maxW: narrow ? cw - 24 : cw * 0.58, alpha: defined[i] ? 1 : 0.7 });
          if (i === 1 && r.strike) { const vw = Math.min(tw(g, vals[i], 12, 'mono'), narrow ? cw - 24 : cw * 0.58); const sx = narrow ? vx : vx - vw; V.line(g, sx, vy, sx + vw, vy, { color: C.red, lw: 2 }); }
          if (win) { const bw = tw(g, 'vence', 11, 'sans', 700) + 14; V.box(g, x + cw - bw - 8, y - 8, bw, 16, { r: 8, fill: col }); V.text(g, 'vence', x + cw - bw / 2 - 8, y, { size: 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) }); }
        }
        // feixe de luz
        const bxl = pad + 6, yEnd = cys[r.layer];
        V.line(g, bxl, 4, bxl, yEnd, { color: V.alpha(C.ink, 0.25), lw: 3 });
        const u = (t * 0.7) % 1;
        V.glow(g, bxl, V.lerp(4, yEnd, V.ease.inOut(u)), 12, C.d, 1);
        V.arrow(g, bxl, yEnd, lx + r.layer * 8 - 2, yEnd, { color: C.d, lw: 2, head: 7 });
        for (let i = 0; i < r.layer; i++) { g.beginPath(); g.arc(bxl, cys[i], 4, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.5; g.stroke(); }
        // resultado
        const ry = 10 + 3 * (cardH + gap) + 4, rh = h - ry - 8, rw = w - 2 * pad;
        const scol = r.status === 'err' ? C.red : r.status === 'warn' ? C.d : C.e;
        V.box(g, pad, ry, rw, rh, { r: 12, fill: V.alpha(scol, 0.08), stroke: V.alpha(scol, 0.6), lw: 1.4 });
        V.text(g, 'expansão no script:', pad + 12, ry + 20, { size: 11, color: C.muted });
        V.text(g, X.exp, pad + 12 + (narrow ? 0 : 128), ry + (narrow ? 38 : 20), { size: 12, font: 'mono', color: C.ink, maxW: rw - 24 - (narrow ? 0 : 128) });
        const vy = ry + (narrow ? 62 : 46);
        V.text(g, 'valor efetivo →', pad + 12, vy, { size: 12, color: C.ink, weight: 600 });
        V.text(g, name + '=' + r.val, pad + 12 + tw(g, 'valor efetivo →', 12, 'sans', 600) + 10, vy, { size: 13, font: 'mono', weight: 700, color: scol, maxW: rw - 150 });
        const nh = para(g, r.note, pad + 12, vy + 22, rw - 24, { size: 12, color: C.ink, lh: 16 });
        // ordem das operações no launcher
        const ops = ['snapshot dos exports', 'source .env', 'ABLIT=0', 'reexport do snapshot', 'expansão ${…}'];
        const hot = X.kind === 'forced' ? (S.caller !== 'none' ? [0, 2, 3] : [1, 2]) : S.caller !== 'none' ? [0, 3, 4] : S.env ? [1, 4] : [4];
        let ox = pad + 12, oy = vy + 22 + nh + 10;
        const rows = [];
        ops.forEach((o, k) => { const lab = (k + 1) + ' ' + o, wd = tw(g, lab, 11, 'mono', 500) + 14; if (ox + wd > pad + rw - 12) { ox = pad + 12; oy += 24; } rows.push([ox, oy, lab, k]); ox += wd + 16; });
        if (oy + 20 < ry + rh - 4) rows.forEach(([x, y2, lab, k], j) => {
          const on = hot.includes(k);
          chip(g, x, y2, lab, { size: 11, h: 19, fill: on ? V.alpha(C.d, 0.2) : V.alpha(C.muted, 0.06), stroke: on ? C.d : V.alpha(C.muted, 0.4), color: on ? C.ink : C.muted });
          if (j < rows.length - 1 && rows[j + 1][1] === y2) V.text(g, '→', rows[j + 1][0] - 9, y2 + 10, { size: 11, align: 'center', base: 'middle', color: C.muted });
        });
      });
    })();

    /* ============================================ F5 · imagem em camadas */
    (function () {
      const fig = root.querySelector('#fig-imagem');
      const BR = [
        { lab: 'FROM vllm-openai@sha256:905c0293…', sh: 'FROM …@sha256:905c…', lock: true, c: 'b' },
        { lab: 'patches NoPE sparse-MLA + asserts', sh: 'patches NoPE + asserts', c: 'c' },
        { lab: 'exl3 + exllamav3_ext (sm_121a)', sh: 'exl3 + ext sm_121a', lock: true, c: 'd' },
        { lab: 'overlays + testes em CPU', sh: 'overlays + testes CPU', c: 'a' },
        { lab: 'instanttensor==0.2.0 --no-deps', sh: 'instanttensor --no-deps', lock: true, c: 'e' },
        { lab: 'cold-load UMA + skip graph profile', sh: 'cold-load + skip profile', c: 'f' },
        { lab: 'LABEL glm53.recipe.stamp', sh: 'LABEL recipe.stamp', c: 'd', stamp: true },
      ];
      const CODE = [
        ['ARG BASE=vllm/vllm-openai:', '  glm53-flash-arm64-cu130', '  @sha256:905c0293…', '# vLLM 0.1.dev20051+g487ecf187', '# CUDA 13.0 · arm64'],
        ['replace_once(old, new)', '  # exatamente 1 ocorrência', 'self.rope_pad = 64', 'q = F.pad(q, (0, self.rope_pad))', "assert 'self.rope_pad = 64' in \\", '  inspect.getsource(impl.__init__)'],
        ['QuantizationMethods += "exl3"', 'ENV TORCH_CUDA_ARCH_LIST=12.1a', 'ENV MAX_JOBS=8', 'CPATH=…/nvidia/cu13/include', 'exllamav3 @ c5d9c657 (0.0.43)', "assert hasattr(ext, 'exl3_moe')"],
        ['COPY overlay/patch_*.py tests/…', 'RUN test_apc_no_store.py  # antes', 'RUN patch_apc_no_store.py', '# kpool: 2 patches, 1 layer', '# (o overlay2 tem limite de camadas)'],
        ['RUN pip install --no-deps \\', '    instanttensor==0.2.0', '# sem deps: não trocar o', '# NCCL 2.30.7 da imagem'],
        ['RUN test_cold_load_uma.py && \\', '    patch_cold_load_uma.py', 'RUN test_skip_cudagraph_profile.py', '    && patch_skip_cudagraph_profile.py', '# idempotentes: o boot reaplica'],
        ['ARG GLM53_RECIPE_STAMP=unknown', 'LABEL glm53.recipe.stamp=${…}', '# último layer: um rebuild que', '# só muda o carimbo é barato'],
      ];
      const WHY = [
        'Uma tag pode ser republicada; um digest não. Os patches são âncoras byte a byte sobre esse código.',
        'Com um upstream diferente, o build quebra em vez de gerar uma imagem silenciosamente errada.',
        'Cubins nativos sm_121a; compilação leve para não disputar a memória unificada.',
        'Um build novo fica idêntico a uma imagem patcheada no boot, e a reaplicação vira no-op.',
        'Um pip install normal trocaria o NCCL 2.30.7 da imagem por 2.29.7.',
        'Com o page cache cheio, o caminho stock nunca chegava ao /health.',
        'Um git pull que muda um overlay faz o carimbo divergir e dispara um rebuild, uma vez.',
      ];
      const SC = {
        igual: { repo: 'a3f19c…', lab: 'a3f19c…', ok: true, res: '✓ usa a imagem local', sub: 'O worker compara o image key (RootFS.Layers); se diferir: pull, ou docker save --platform | ssh docker load.', c: 'e' },
        edit: { repo: '7c02e8…', lab: 'a3f19c…', ok: false, res: '↻ rebuild local (BUILD=1 implícito)', sub: 'A imagem nova ganha o carimbo 7c02e8… e é enviada ao worker. O self-check na GPU roda antes de servir.', c: 'd' },
        skip: { repo: '7c02e8…', lab: 'a3f19c…', ok: false, res: '⚠ SKIP_BUILD=1: mantém a imagem e só avisa', sub: 'O carimbo continua divergente: a imagem pode não conter o overlay editado (no runtime, o bind mount ainda o aplica).', c: 'd' },
        ghcr: { repo: 'a3f19c…', lab: 'a3f19c…', pulled: '9e4b51…', ok: false, res: '↩ pull rejeitado: restaura a tag local', sub: 'O GHCR publicou outra receita. Um pull só é adotado quando o carimbo publicado também bate com o repositório.', c: 'c' },
      };
      let scen = 'igual';
      const steps = [
        { t: 'FROM: a base pinada por digest', d: '<code>ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…</code> fixa os bytes do vLLM <code>0.1.dev20051+g487ecf187</code> (CUDA 13, arm64). Todos os patches seguintes são âncoras exatas sobre esse código.' },
        { t: 'Patches de backend + asserts', d: 'NoPE → geometria 576 com zeros, prefill sem MHA densa, top-k como capacidade, 511 pools + cauda, página 64, sem autotune do FlashInfer, PDL só em SM9x/10x. Cada substituição exige uma única ocorrência, e um bloco de <code>assert</code> relê o fonte com <code>inspect.getsource</code>.' },
        { t: 'EXL3: registrar e compilar', d: '<code>"exl3"</code> entra no <code>Literal</code> de métodos de quantização, com import lazy de <code>Exl3Config</code>. O <code>exllamav3_ext@c5d9c657</code> é compilado com <code>TORCH_CUDA_ARCH_LIST=12.1a</code>, <code>MAX_JOBS=8</code> e <code>CPATH</code> em <code>nvidia/cu13/include</code>, e o build confere os símbolos.' },
        { t: 'Overlays e testes em CPU', d: 'Cada overlay é copiado, o teste em CPU roda contra o arquivo real do vLLM (fixtures, <code>GLM53_REQUIRE_VLLM=1</code>) e o patch é aplicado na mesma ordem do runtime. Os dois patches de kpool ficam num único layer: separados, a imagem passava do limite de camadas do overlay2.' },
        { t: 'InstantTensor sem dependências', d: '<code>pip install --no-deps instanttensor==0.2.0</code>: um install normal traria <code>nvidia-nccl-cu13==2.29.7</code> por cima do NCCL 2.30.7 da imagem.' },
        { t: 'Correções de carga para UMA', d: '<code>patch_cold_load_uma.py</code> (orçamento do InstantTensor pelo <code>MemAvailable</code>, clone do mmap em páginas de 64 KiB) e <code>patch_skip_cudagraph_profile.py</code>, cada um depois do seu teste. Ambos são idempotentes: o boot os reaplica sem efeito.' },
        { t: 'O carimbo no último layer', d: '<code>ARG GLM53_RECIPE_STAMP</code> → <code>LABEL glm53.recipe.stamp</code>. Por ficar por último, um rebuild que muda só o carimbo é barato.' },
        { t: 'No start: carimbo × repositório', d: 'O launcher recalcula o SHA-256 de Dockerfile + overlay/ + files/ + tests/ + ablit/ e compara com o rótulo. Troque o cenário no seletor para ver cada decisão.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6000 });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário no start', options: [{ v: 'igual', label: 'tudo igual' }, { v: 'edit', label: 'editei um overlay' }, { v: 'skip', label: 'editei + SKIP_BUILD=1' }, { v: 'ghcr', label: 'GHCR mudou' }], value: 'igual', onChange: (v) => { scen = v; st.stop(); st.go(7); } });
      const lock = (g, x, y, col) => { g.strokeStyle = col; g.lineWidth = 1.7; g.beginPath(); g.arc(x, y - 2, 3.6, Math.PI, 0); g.stroke(); V.box(g, x - 5.5, y - 2, 11, 8, { r: 2, fill: col }); };
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 440, 560, w);
        const C = V.C(); const narrow = w < 640; const pad = 14;
        const i = st.i, el = st.elapsed();
        const nShow = Math.min(7, i + 1);
        const tx = pad, tW = narrow ? w - 2 * pad : w * 0.47 - pad;
        const tBot = narrow ? 212 : h - 22;
        const bh = narrow ? 24 : Math.min(38, (tBot - 24) / 7 - 6);
        V.line(g, tx, tBot + 2, tx + tW, tBot + 2, { color: V.alpha(C.muted, 0.7), lw: 2 });
        for (let k = nShow; k < 7; k++) {
          const y = tBot - (k + 1) * (bh + 5), bx = tx + k * 3, bwid = tW - k * 6;
          g.setLineDash([4, 4]); V.box(g, bx, y, bwid, bh, { r: 7, fill: 'transparent', stroke: V.alpha(C.muted, 0.35), lw: 1 }); g.setLineDash([]);
          V.text(g, narrow ? BR[k].sh : BR[k].lab, bx + 12, y + bh / 2 + 1, { size: narrow ? 11 : 12, base: 'middle', color: C.muted, alpha: 0.45, maxW: bwid - 44 });
        }
        for (let k = 0; k < nShow; k++) {
          const B = BR[k], col = C[B.c];
          let y = tBot - (k + 1) * (bh + 5), a = 1;
          if (k === i && i < 7) { const e = V.ease.out(Math.min(1, el / 0.6)); y -= (1 - e) * 60; a = Math.max(0.05, e); }
          const cur = k === i || (i === 7 && k === 6);
          g.globalAlpha = a;
          const bx = tx + k * 3, bwid = tW - k * 6;
          V.box(g, bx, y, bwid, bh, { r: 7, fill: V.alpha(col, cur ? 0.34 : 0.15), stroke: V.alpha(col, cur ? 1 : 0.5), lw: cur ? 2 : 1 });
          V.text(g, narrow ? B.sh : B.lab, bx + 12, y + bh / 2 + 1, { size: narrow ? 11 : 12, base: 'middle', color: C.ink, weight: cur ? 700 : 400, maxW: bwid - 44 });
          if (B.lock) lock(g, bx + bwid - 18, y + bh / 2, col);
          if (B.stamp) { const sx = bx + bwid - 18, sy = y + bh / 2; const sc = i >= 6 ? 1 : 0.7; g.save(); g.translate(sx, sy); g.rotate(-0.25 + (i === 6 ? 0.25 * V.ease.out(Math.min(1, el / 0.8)) : 0.25)); g.scale(sc, sc); V.box(g, -9, -7, 18, 14, { r: 3, fill: 'transparent', stroke: col, lw: 2 }); V.text(g, '#', 0, 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: col }); g.restore(); }
          g.globalAlpha = 1;
        }
        if (!narrow) V.text(g, 'camadas do Dockerfile: ' + nShow + ' de 7 (simplificado)', tx, h - 6, { size: 11, color: C.muted });
        const dx = narrow ? pad : w * 0.5, dy = narrow ? tBot + 14 : 14, dw = narrow ? w - 2 * pad : w * 0.5 - pad, dh = h - dy - (narrow ? 8 : 14);
        V.box(g, dx, dy, dw, dh, { r: 12, fill: V.alpha(C.ink, 0.05), stroke: V.alpha(C.muted, 0.35) });
        if (i < 7) {
          V.text(g, 'DOCKERFILE · PASSO ' + (i + 1), dx + 14, dy + 20, { size: 11, font: 'mono', weight: 700, color: C[BR[i].c] });
          CODE[i].forEach((ln, k) => {
            const e = V.ease.out(Math.min(1, Math.max(0, (el - k * 0.12) / 0.4)));
            V.text(g, ln, dx + 14, dy + 44 + k * (narrow ? 17 : 20), { size: narrow ? 11 : 12.5, font: 'mono', color: ln.trim().startsWith('#') ? C.muted : C.ink, alpha: e, maxW: dw - 28 });
          });
          const wy = dy + 44 + CODE[i].length * (narrow ? 17 : 20) + (narrow ? 8 : 16);
          if (wy + 30 < dy + dh) {
            V.line(g, dx + 14, wy - 8, dx + dw - 14, wy - 8, { color: V.alpha(C.muted, 0.3), lw: 1 });
            V.text(g, 'POR QUÊ', dx + 14, wy + 8, { size: 11, font: 'mono', weight: 700, color: C[BR[i].c] });
            para(g, WHY[i], dx + 14, wy + 26, dw - 28, { size: narrow ? 11.5 : 12.5, color: C.ink, lh: narrow ? 15 : 17 });
          }
        } else {
          const s = SC[scen], col = C[s.c];
          V.text(g, 'START.SH · ENSURE_IMAGE', dx + 14, dy + 20, { size: 11, font: 'mono', weight: 700, color: col });
          let y = dy + 38;
          const row = (lab, val, vc) => { V.text(g, lab, dx + 14, y + 11, { size: 11.5, color: C.muted }); chip(g, dx + dw - 14 - tw(g, val, 12, 'mono', 600) - 14, y, val, { size: 12, weight: 600, fill: V.alpha(vc, 0.16), stroke: vc, color: C.ink }); y += 30; };
          row('sha256(repositório)', s.repo, C.a);
          row('LABEL da imagem local', s.lab, C.b);
          if (s.pulled) row('LABEL da imagem puxada', s.pulled, C.c);
          const pulse = 0.6 + 0.4 * Math.sin(t * 4);
          V.text(g, s.ok ? '=' : '≠', dx + dw / 2, y + 12, { size: 24, weight: 700, align: 'center', base: 'middle', color: col, alpha: pulse });
          y += 34;
          V.text(g, s.res, dx + 14, y, { size: narrow ? 12.5 : 14, weight: 700, color: col, maxW: dw - 28 });
          y += 20;
          y += para(g, s.sub, dx + 14, y, dw - 28, { size: 12, color: C.ink, lh: 16 }) + 6;
          if (y < dy + dh - 18) para(g, 'Depois: test_exl3_overlay.py na GPU (EXL3_SELFCHECK_GPU=1), a menos que SKIP_OVERLAY_VERIFY=1.', dx + 14, y, dw - 28, { size: 11.5, color: C.muted, lh: 15 });
        }
      });
    })();

    /* ============================================ F5b · registro fp8_ds_mla e pad 512→576 */
    (function () {
      const fig = root.querySelector('#fig-registro');
      const S = { nopad: false, t0: null };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'NoPE sem pad (imagem publicada)', value: false, onChange: (v) => { S.nopad = v; S.t0 = null; } });
      // valores ilustrativos: 32 células do latente (cada uma resume 16 dimensões) + 4 células de pad (64 dimensões)
      const rnd = V.rng(42), NL = 32, NP = 4, NT = NL + NP;
      const q = Array.from({ length: NL }, () => Math.round(rnd.normal() * 8) / 10);
      const k = Array.from({ length: NL }, () => Math.round(rnd.normal() * 8) / 10);
      const prefix = [0]; q.forEach((v, i) => prefix.push(prefix[i] + v * k[i]));
      const TOTAL = prefix[NL];
      V.fig(fig, (g, w, h, t) => {
        if (w < 160) return;
        const C = V.C(); const narrow = w < 600; const pad = 14, gw = w - 2 * pad;
        if (S.t0 == null) S.t0 = t;
        const el = t - S.t0;
        let y = 18;
        V.text(g, narrow ? 'REGISTRO fp8_ds_mla · 656 B/TOKEN/CAMADA' : 'REGISTRO fp8_ds_mla · 656 B POR TOKEN, POR CAMADA MLA, POR RANK', pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: gw });
        y += 10;
        const th = narrow ? 30 : 36, sc = gw / 656;
        const segs = [[512, C.a], [16, C.d], [128, C.muted]];
        const fillB = V.clamp(((el % 6) / 3.4) * 656, 0, 656);
        let x = pad, acc = 0;
        segs.forEach(([n, col], i) => {
          const ww = n * sc;
          V.box(g, x, y, ww - 1, th, { r: 4, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.75), lw: 1 });
          const f = V.clamp((fillB - acc) / n, 0, 1);
          if (f > 0) {
            g.save(); V.rr(g, x, y, ww - 1, th, 4); g.clip();
            g.fillStyle = V.alpha(col, i === 2 ? 0.22 : 0.78); g.fillRect(x, y, (ww - 1) * f, th);
            if (i === 2) { g.strokeStyle = V.alpha(C.muted, 0.75); g.lineWidth = 1; for (let qx = x - th; qx < x + (ww - 1) * f; qx += 6) { g.beginPath(); g.moveTo(qx, y + th); g.lineTo(qx + th, y); g.stroke(); } }
            g.restore();
          }
          if (i === 0) for (let b = 64; b < 512; b += 64) V.line(g, x + b * sc, y + th - 6, x + b * sc, y + th - 1, { color: V.alpha(C.bg, 0.8), lw: 1 });
          acc += n; x += ww;
        });
        const l0 = narrow ? '512 B · latente em FP8' : '512 B · latente NoPE, 512 × FP8 (1 B por dimensão)', l0w = Math.min(512 * sc - 16, tw(g, l0, 11.5, 'sans', 700) + 14);
        V.box(g, pad + (512 * sc) / 2 - l0w / 2, y + th / 2 - 10, l0w, 20, { r: 6, fill: V.alpha(C.bg, 0.72) });
        V.text(g, l0, pad + (512 * sc) / 2, y + th / 2 + 1, { size: 11.5, weight: 700, align: 'center', base: 'middle', color: C.ink, maxW: l0w - 10 });
        if (128 * sc > 44) V.text(g, '128 B', pad + (528 + 64) * sc, y + th / 2 + 1, { size: 11.5, weight: 700, align: 'center', base: 'middle', color: C.ink });
        // setas de legenda
        const ly = y + th + 18;
        const xs = pad + (512 + 8) * sc;
        V.line(g, xs, y + th + 2, xs, ly - 11, { color: C.d, lw: 1.2 });
        V.text(g, narrow ? '16 B · 4 escalas FP32 (1 por 128 dims)' : '16 B · 4 escalas FP32, uma por bloco de 128 dimensões', narrow ? pad + gw : xs + 4, ly, { size: narrow ? 11 : 11.5, align: 'right', color: C.d, maxW: gw });
        V.text(g, narrow ? '128 B · RoPE 64 × BF16, todos zero' : '128 B · RoPE 64 × BF16: todos zero no GLM (NoPE)', pad + gw, ly + 17, { size: narrow ? 11 : 11.5, align: 'right', color: C.muted, maxW: gw });
        y = ly + 38;
        y += para(g, 'necessário de fato: 528 B · o pad custa +128 B (+24%) · página MLA = 3584 × 656 = 2.351.104 B', pad, y, gw, { size: 11, font: 'mono', color: C.ink, lh: 15 }) + 14;
        // B · o produto no kernel
        V.text(g, narrow ? 'KERNEL SPARSE-MLA · GEOMETRIA 512 + 64 = 576' : 'FLASHINFER_MLA_SPARSE_SM120 · GEOMETRIA FIXA: 512 (LATENTE) + 64 (RoPE) = 576', pad, y, { size: 11, font: 'mono', weight: 700, color: C.b, maxW: gw });
        y += 12;
        const labW = 22, cw = (gw - labW) / NT, chh = narrow ? 16 : 22;
        const cyc = 7.5, ph = (el % cyc) / cyc;
        const cur = S.nopad ? -1 : Math.min(NT - 1, Math.floor((ph / 0.8) * NT));
        const done = !S.nopad && ph >= 0.8;
        [['q', q], ['k', k]].forEach(([nm, arr], row) => {
          const ry = y + row * (chh + 6);
          V.text(g, nm, pad + 6, ry + chh / 2 + 1, { size: 12.5, font: 'mono', weight: 700, base: 'middle', color: C.ink });
          for (let i = 0; i < NT; i++) {
            const cx = pad + labW + i * cw;
            if (i < NL) V.box(g, cx + 0.5, ry, cw - 1, chh, { r: 2, fill: V.cmap(arr[i] / 1.6, 'div'), stroke: i === cur ? C.ink : null, lw: 2 });
            else if (S.nopad && row === 0) { g.setLineDash([2, 2]); V.box(g, cx + 0.5, ry, cw - 1, chh, { r: 2, fill: V.alpha(C.red, 0.1 + 0.08 * Math.sin(t * 5)), stroke: C.red, lw: 1 }); g.setLineDash([]); if (cw > 9) V.text(g, '?', cx + cw / 2, ry + chh / 2 + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.red }); }
            else { V.box(g, cx + 0.5, ry, cw - 1, chh, { r: 2, fill: V.alpha(C.muted, 0.18), stroke: i === cur ? C.ink : V.alpha(C.muted, 0.5), lw: i === cur ? 2 : 1 }); if (cw > 9) V.text(g, '0', cx + cw / 2, ry + chh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.muted }); }
          }
        });
        // chave sob as colunas
        const ky = y + 2 * (chh + 6) + 12;
        V.text(g, narrow ? 'latente: 32 células de 16 dims' : 'latente: 32 células de 16 dimensões (valores ilustrativos)', pad + labW, ky, { size: 11, color: C.muted, maxW: NL * cw - 6 });
        V.text(g, 'pad', pad + labW + NL * cw + (NP * cw) / 2, ky, { size: 11, color: S.nopad ? C.red : C.muted, align: 'center' });
        y = ky + 24;
        let msg, col;
        if (S.nopad) { msg = '✕ “pe_dim must be 64 for fp8_ds_mla” (concat_and_cache_mla). O modelo tem 0 dimensões de RoPE, e o layout e o kernel só existem para 64 (576 no total): a imagem publicada morre no primeiro forward.'; col = C.red; }
        else if (cur < NL && !done) { msg = 'Σ qᵢ·kᵢ até a célula ' + (cur + 1) + ' = ' + br(prefix[cur + 1], 2); col = C.ink; }
        else if (!done) { msg = 'pad: qᵢ·kᵢ = 0 · 0 = 0. A soma fica em ' + br(TOTAL, 2); col = C.muted; }
        else { msg = '✓ q·k com 576 dimensões = q·k com 512 = ' + br(TOTAL, 2) + ': o score é exato, e o preço é só memória.'; col = C.e; }
        y += para(g, msg, pad, y, gw, { size: 13, color: col, lh: 18, weight: 600 }) + 6;
        y += para(g, 'O patch: rope_pad = 64 quando qk_rope_head_dim = 0; q = F.pad(q, (0, 64)); k_pe gravado como zeros. Mesma geometria de 576 que o kernel exige, mesmo resultado que a MLA NoPE original.', pad, y, gw, { size: 12, color: C.ink, lh: 16 });
        fitH(fig, Math.ceil(y + 6), Math.ceil(y + 6), w);
      });
    })();

    /* ============================================ F6 · esteira de overlays */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      const P = [
        ['cold_load_uma', 'loader / InstantTensor'], ['skip_cudagraph_profile', 'profiling de memória'], ['glm_video_placeholders', 'vídeo (glm46v) + .pth'],
        ['suppress_stops_in_reasoning', 'detokenizer'], ['scheduler_decode_floor', 'scheduler.py (fair v5; v7 no #283)'], ['mamba_align_chunking', 'scheduler (alinhamento Mamba)'],
        ['glm5_drafter_group', 'kv_cache_utils.py'], ['hybrid_prefix_hit', 'kv_cache_coordinator'], ['apc_per_group_retention', 'kv_cache_coordinator'],
        ['apc_no_store', 'sampling_params / block_pool'], ['mamba_align_state_free', 'single_type_kv_cache_manager'], ['kv_capacity_log', 'kv_cache_utils.py (log)'],
        ['tool_choice_none', 'parser glm47 (bad_words)'], ['xgrammar_termination', 'structured output'], ['kpool_tail_slotmap', 'slot mapping do kpool'],
        ['kpool_tail_seed_stride', 'seed do kpool tail'], ['spinwait', 'shm_broadcast.py'], ['adaptive_k', 'spec decode (EMA)'],
        ['dense_fp8', 'exl3.py (FP8 Marlin)'], ['dflash2_exl3', 'qwen3_dflash2.py'], ['loadclone', 'weight_utils.py'],
        ['default_max_new_tokens', 'serving (pydantic)'], ['indexer_workspace', 'workspace do indexer'], ['cache_reset', 'rotas de reset de cache'], ['ablit', 'load_weights (hook)'],
      ];
      let scen = 'ok', t0 = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'ok', label: 'ordem correta' }, { v: 'swap', label: 'trocar 8 ↔ 9' }, { v: 'miss', label: 'rank 1 sem o nº 18' }], value: 'ok', onChange: (v) => { scen = v; t0 = null; } });
      const DT = 0.5, TR = 1.9, HOLD = 3.4;
      const seq = (rank) => { const s = P.map((_, i) => i); if (scen === 'swap') { s[7] = 8; s[8] = 7; } return s.map((pi) => ({ pi, ghost: scen === 'miss' && rank === 1 && pi === 17 })); };
      let first = true;
      V.fig(fig, (g, w, h, t) => {
        if (t0 == null) { t0 = t - (first ? 7 : 0); first = false; }
        const C = V.C(); const narrow = w < 560; const pad = 14;
        const total = 25 * DT + TR + HOLD;
        let tc = (t - t0) % total;
        const failPos = scen === 'swap' ? 8 : -1;
        const failT = failPos >= 0 ? failPos * DT + TR : Infinity;
        const failed = tc >= failT;
        const tEff = Math.min(tc, failT);
        const bx0 = pad + (narrow ? 0 : 92), mw = narrow ? 56 : 84, mx = w - pad - mw, bs = narrow ? 18 : 24;
        const laneY = narrow ? [58, 212] : [52, 204];
        const names = ['rank 0 · head', 'rank 1 · worker'];
        const doneAll = tc >= 25 * DT + TR;
        [0, 1].forEach((rk) => {
          const y = laneY[rk], S0 = seq(rk);
          if (narrow) V.text(g, names[rk].toUpperCase(), pad, y - 26, { size: 11, font: 'mono', weight: 700, color: rk ? C.b : C.a });
          else { V.text(g, names[rk].split(' · ')[0].toUpperCase(), pad, y - 2, { size: 12, font: 'mono', weight: 700, color: rk ? C.b : C.a }); V.text(g, names[rk].split(' · ')[1], pad, y + 14, { size: 11, color: C.muted }); }
          V.box(g, bx0, y - 15, mx - bx0 - 4, 30, { r: 15, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35) });
          if (!failed) V.line(g, bx0 + 12, y, mx - 16, y, { color: V.alpha(C.muted, 0.4), lw: 2, dash: [3, 9], dashOffset: -t * 30 });
          let last = -1;
          S0.forEach((it, k) => {
            const launch = k * DT, frac = (tEff - launch) / TR;
            if (frac >= 1) { if (!it.ghost) last = k; return; }
            if (frac < 0) return;
            const x = bx0 + 6 + frac * (mx - bx0 - bs - 14);
            const col = C.series[it.pi % 6];
            if (it.ghost) { g.setLineDash([3, 3]); V.box(g, x, y - bs / 2, bs, bs, { r: 5, fill: 'transparent', stroke: V.alpha(C.red, 0.8), lw: 1.5 }); g.setLineDash([]); V.text(g, '?', x + bs / 2, y + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.red }); }
            else { V.box(g, x, y - bs / 2, bs, bs, { r: 5, fill: V.alpha(col, 0.85) }); V.text(g, String(it.pi + 1), x + bs / 2, y + 1, { size: narrow ? 10.5 : 11, weight: 700, align: 'center', base: 'middle', color: onFill(C) }); }
          });
          // máquina
          const flash = last >= 0 && tEff - (last * DT + TR) < 0.3;
          const mcol = failed ? C.red : flash ? C.series[S0[last].pi % 6] : C.muted;
          V.box(g, mx, y - 26, mw, 52, { r: 10, fill: V.alpha(mcol, failed || flash ? 0.28 : 0.1), stroke: mcol, lw: failed ? 2.5 : 1.5 });
          V.text(g, 'vLLM', mx + mw / 2, y - 4, { size: 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, failed ? '✕ âncora' : narrow ? 'pkgs' : 'site-pkgs', mx + mw / 2, y + 12, { size: 11, align: 'center', font: 'mono', color: failed ? C.red : C.muted, maxW: mw - 6 });
          // progresso
          const sq = Math.min(12, (mx - bx0 - 4) / 25 - 2), sy = y + 24;
          S0.forEach((it, k) => {
            const applied = !it.ghost && tEff >= k * DT + TR;
            const failHere = failed && k === failPos;
            const xx = bx0 + k * (sq + 2);
            let fill = V.alpha(C.muted, 0.18);
            if (applied) fill = V.alpha(C.series[it.pi % 6], 0.85);
            if (it.ghost && tEff >= k * DT + TR) fill = V.alpha(C.red, 0.5);
            if (failHere) fill = C.red;
            V.box(g, xx, sy, sq, sq, { r: 2.5, fill });
          });
          // ticker
          let msg = '', mc = C.ink;
          if (failed) { const pi = S0[failPos].pi; msg = (pi + 1) + ' · patch_' + P[pi][0] + '.py → âncora não encontrada: aborta (fail-closed)'; mc = C.red; }
          else if (last >= 0) { const pi = S0[last].pi; msg = (pi + 1) + ' · patch_' + P[pi][0] + '.py → ' + P[pi][1]; }
          else msg = 'aguardando: script interno montando ARGS…';
          V.text(g, msg, bx0, sy + sq + 18, { size: 11.5, font: 'mono', color: mc, maxW: w - bx0 - pad });
        });
        // mensagem geral
        const mwid = w - 2 * pad;
        const MSG = {
          fail: 'Os dois ranks abortam no mesmo patch: um overlay fail-closed prefere não subir a subir com o coordinator de KV montado pela metade.',
          miss: 'O rank 1 terminou sem o nº 18 (adaptive_k). Com GLM53_ADAPTIVE_K=ema, o rank 0 captura graphs de decode de 3, 5 e 8 posições e o rank 1 só os de 8: as coletivas divergem, e o NCCL trava depois do PIECEWISE (interpretação, a partir do README). Com o adaptive-k desligado, o default, o buraco passa despercebido até alguém ligá-lo.',
          ok: '25 de 25 nos dois ranks, mesmos bytes e mesma ordem: exec vllm serve.',
          run: 'Mesma lista, mesma ordem, nos dois containers. Cada patch confere sua âncora e é no-op se já foi aplicado.',
        };
        let txt, col;
        if (failed) { txt = MSG.fail; col = C.red; }
        else if (scen === 'miss' && doneAll) { txt = MSG.miss; col = C.red; }
        else if (doneAll) { txt = MSG.ok; col = C.e; }
        else { txt = MSG.run; col = C.muted; }
        const lines = (s2) => wrap(g, s2, mwid - 24, 12.5, 'sans', 500).length;
        const maxL = Math.max(lines(MSG.fail), lines(MSG.miss), lines(MSG.run));
        const nL = lines(txt), boxH = nL * 17 + 20, my = laneY[1] + (narrow ? 70 : 62);
        fitH(fig, Math.max(380, Math.ceil(my + maxL * 17 + 28)), Math.max(380, Math.ceil(my + maxL * 17 + 28)), w);
        V.box(g, pad, my, mwid, boxH, { r: 10, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.5) });
        para(g, txt, pad + 12, my + 22, mwid - 24, { size: 12.5, color: C.ink, lh: 17, weight: 500 });
        if (scen === 'miss' && doneAll) { const zx = mx + mw / 2, z0 = laneY[0] + 26, z1 = laneY[1] - 26; g.strokeStyle = C.red; g.lineWidth = 2.5; g.beginPath(); for (let k = 0; k <= 8; k++) { const yy = V.lerp(z0, z1, k / 8); const xx = zx + (k % 2 ? 7 : -7); if (k) g.lineTo(xx, yy); else g.moveTo(xx, yy); } g.stroke(); }
      });
    })();

    /* ============================================ F7 · speculative-config e graphs */
    (function () {
      const fig = root.querySelector('#fig-spec');
      const S = { seqs: 4, k: 7, ada: false };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'MAX_NUM_SEQS', min: 1, max: 8, step: 1, value: 4, onInput: (v) => { S.seqs = v; } });
      UI.seg(ctl, { label: 'DFLASH_TOKENS (k)', options: [3, 5, 7], value: 7, onChange: (v) => { S.k = v; } });
      UI.toggle(ctl, { label: 'adaptive-k (conjunto 2,4,7)', value: false, onChange: (v) => { S.ada = v; } });
      const calc = () => {
        const sizes = new Set([1, 2, 4, 8, 16, 24, 32]);
        const q = S.k + 1; let lens = [q];
        if (S.ada) {
          const L = new Set([q]); [2, 4, 7].forEach((kk) => { if (kk + 1 > 0 && kk + 1 <= q) L.add(kk + 1); });
          lens = Array.from(L).sort((a, b) => a - b);
          for (let n = 1; n <= S.seqs; n++) lens.forEach((qq) => sizes.add(n * qq));
        }
        const need = new Set(); for (let n = 1; n <= S.seqs; n++) lens.forEach((qq) => need.add(n * qq));
        return { sizes, need, lens, q };
      };
      const face = (g, x, y, r, col, C, mood) => {
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = col; g.fill();
        const ec = onFill(C);
        V.dot(g, x - r * 0.35, y - r * 0.18, Math.max(1.3, r * 0.13), ec); V.dot(g, x + r * 0.35, y - r * 0.18, Math.max(1.3, r * 0.13), ec);
        g.strokeStyle = ec; g.lineWidth = 1.4; g.beginPath();
        if (mood === 'q') { V.text(g, '?', x, y + r * 0.42, { size: Math.max(9, r * 0.8), weight: 700, align: 'center', base: 'middle', color: ec }); }
        else { g.arc(x, y + r * 0.1, r * 0.42, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke(); }
      };
      V.fig(fig, (g, w, h, t) => {
        if (w < 160) return; // canvas ainda sem largura (ex.: figura sendo expandida)
        const C = V.C(); const narrow = w < 600; const pad = 14;
        const R = calc();
        // A · JSON
        V.text(g, '--speculative-config', pad, 18, { size: 11, font: 'mono', color: C.muted, weight: 700 });
        const js = [['"method":"dflash"', C.c], ['"model":"…DFlash2@dc77ff1c"', C.b], ['"num_speculative_tokens":' + S.k, C.d], ['"kv_cache_dtype":"auto"', C.a], ['"draft_sample_method":"probabilistic"', C.e], ['"rejection_sample_method":"standard"', C.e], ['"draft_tensor_parallel_size":2', C.f]];
        let cx = pad, cy = 26;
        js.forEach(([s, col]) => {
          const wd = tw(g, s, 11, 'mono', 500) + 14;
          if (cx + wd > w - pad) { cx = pad; cy += 24; }
          chip(g, cx, cy, s, { size: 11, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.7), color: C.ink, h: 19 });
          cx += wd + 6;
        });
        let y = cy + 38;
        // B · bloco
        const bW = narrow ? w - 2 * pad : (w - 2 * pad) * 0.5;
        V.text(g, 'BLOCO DO DFLASH2 · 1 ÂNCORA + ' + S.k + ' MÁSCARAS', pad, y, { size: 11, font: 'mono', weight: 700, color: C.c });
        const nTok = S.k + 1, rr = Math.max(4, Math.min(narrow ? 15 : 20, (bW - 20) / nTok / 2 - 3));
        const ph = (t % 4.5) / 4.5;
        for (let j = 0; j < nTok; j++) {
          const x = pad + rr + 4 + j * (rr * 2 + 6), yy = y + 26 + rr;
          const bob = Math.sin(t * 3 + j) * 1.5;
          if (j === 0) face(g, x, yy + bob, rr, C.d, C, 'ok');
          else {
            const filled = ph > 0.3;
            face(g, x, yy + bob, rr, filled ? C.b : V.alpha(C.c, 0.72 + 0.28 * Math.sin(t * 6 + j)), C, filled ? 'ok' : 'q');
            if (ph > 0.55 && ph - 0.55 > (j / nTok) * 0.35) V.text(g, '✓', x, yy - rr - 4, { size: 11, weight: 700, align: 'center', color: C.e });
          }
        }
        const lblY = y + 26 + rr * 2 + 20;
        V.text(g, ph < 0.3 ? 'rascunho: máscaras esperando…' : ph < 0.55 ? 'difusão em bloco: todas preenchidas de uma vez' : 'o alvo verifica as ' + nTok + ' posições num passo', pad, lblY, { size: 11.5, color: C.ink, maxW: bW - 10 });
        // C · passo de decode
        const cX = narrow ? pad : pad + bW + 16, cY = narrow ? lblY + 22 : y;
        const tot = S.seqs * R.q;
        V.text(g, 'PASSO DE DECODE · ' + S.seqs + ' SEQ × ' + R.q + ' = ' + tot + ' TOKENS', cX, cY, { size: 11, font: 'mono', weight: 700, color: C.b, maxW: w - cX - pad });
        const avail = w - cX - pad, cell = Math.max(7, Math.min(narrow ? 12 : 16, (avail - 10) / 16 - 2));
        for (let n = 0; n < S.seqs; n++) for (let j = 0; j < R.q; j++) {
          const on = ((t * 2.2) % (S.seqs + 1)) > n;
          V.box(g, cX + j * (cell + 2), cY + 12 + n * (cell + 2), cell, cell, { r: 2, fill: j === 0 ? V.alpha(C.d, on ? 0.95 : 0.4) : V.alpha(C.b, on ? 0.9 : 0.35) });
        }
        const gridBottom = cY + 12 + S.seqs * (cell + 2);
        y = Math.max(lblY, gridBottom) + 30;
        // D · capture sizes
        V.text(g, 'TAMANHOS DE CAPTURA (REGRA DO LAUNCHER)' + (S.ada ? ' · COMPRIMENTOS ' + R.lens.join(', ') : ''), pad, y, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: w - 2 * pad });
        const all = Array.from(new Set([...R.sizes, ...R.need])).sort((a, b) => a - b);
        let px = pad, py = y + 10; let missing = 0;
        all.forEach((s) => {
          const inS = R.sizes.has(s), isN = R.need.has(s);
          if (isN && !inS) missing++;
          const lab = String(s), wd = tw(g, lab, 11, 'mono', 600) + 14;
          if (px + wd > w - pad) { px = pad; py += 24; }
          const o = isN && inS ? { fill: V.alpha(C.e, 0.3), stroke: C.e, color: C.ink } : isN ? { fill: V.alpha(C.red, 0.18), stroke: C.red, color: C.red, dash: [3, 2] } : { fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.5), color: C.muted };
          chip(g, px, py, lab, Object.assign({ size: 11, weight: 600, h: 19 }, o));
          px += wd + 5;
        });
        y = py + 38;
        y += para(g, missing ? '✕ ' + missing + ' tamanho(s) de decode fora da lista: passe --cudagraph-capture-sizes explícito em EXTRA_ARGS.' : '✓ todo passo de decode (verde) tem graph na lista.', pad, y, w - 2 * pad, { size: 12, color: missing ? C.red : C.e, weight: 600, lh: 16 });
        if (S.ada) y += para(g, 'adaptive-k: cada requisição verifica 2, 4 ou 7 rascunhos (consultas de 3, 5 ou 8) conforme a sua EMA; o batch inteiro usa o menor k, porque os graphs FULL do sparse-MLA e do KDA exigem decode uniforme.', pad, y + 2, w - 2 * pad, { size: 12, color: C.ink, lh: 16 }) + 2;
        // E · TEMP_ROWS
        y += 18;
        const need = S.seqs * R.q, cap = 32, gw = w - 2 * pad, scale = gw / 72;
        y += para(g, 'EXL3_TEMP_ROWS_FUSED = 32 (E3) · precisa de ≥ ' + S.seqs + ' × ' + R.q + ' = ' + need, pad, y, gw, { size: 11.5, font: 'mono', color: C.ink, lh: 15 }) - 15;
        const gy = y + 10;
        V.box(g, pad, gy, gw, 14, { r: 7, fill: V.alpha(C.muted, 0.14) });
        V.box(g, pad, gy, Math.min(gw, need * scale), 14, { r: 7, fill: need <= cap ? V.alpha(C.e, 0.8) : V.alpha(C.red, 0.75) });
        V.line(g, pad + cap * scale, gy - 5, pad + cap * scale, gy + 19, { color: C.ink, lw: 2 });
        V.text(g, '32', pad + cap * scale, gy + 32, { size: 11, font: 'mono', align: 'center', color: C.ink });
        y = gy + 50;
        y += para(g, 'Por que N·(k+1) linhas? No top-8 cada token visita um dado expert no máximo uma vez; no pior caso, os ' + need + ' tokens do passo escolhem o mesmo expert, que recebe ' + need + ' linhas.', pad, y, gw, { size: 11.5, color: C.muted, lh: 15 }) + 2;
        y += para(g, need <= cap ? 'Cabe: o decode inteiro é um único lançamento graph-safe do exl3_moe.' : 'Passa de 32: abaixo da regra do README, o decode deixa de caber num lançamento graph-safe.', pad, y, gw, { size: 12, color: need <= cap ? C.e : C.red, lh: 16, weight: 600 });
        // F · aceitação medida
        let fy = y + 22;
        V.text(g, 'QUANTO DO BLOCO É ACEITO (MEDIDO, TP2, k=7, 2026-08-30)', pad, fy, { size: 11, font: 'mono', weight: 700, color: C.f, maxW: gw });
        const acc = [['estruturado', 6.71, '0,96 de aceitação'], ['prosa', 2.39, '0,34 de aceitação']];
        const labW = narrow ? 84 : 100, bw2 = gw - labW - (narrow ? 76 : 250);
        acc.forEach(([lab, v, note], k) => {
          const yy = fy + 14 + k * 24;
          V.text(g, lab, pad, yy + 11, { size: 12, color: C.ink });
          V.box(g, pad + labW, yy, bw2, 16, { r: 5, fill: V.alpha(C.muted, 0.14) });
          for (let q = 1; q < 8; q++) V.line(g, pad + labW + (q / 8) * bw2, yy + 2, pad + labW + (q / 8) * bw2, yy + 14, { color: V.alpha(C.bg, 0.8), lw: 1 });
          V.box(g, pad + labW, yy, (v / 8) * bw2, 16, { r: 5, fill: V.alpha(k ? C.c : C.a, 0.8) });
          V.text(g, br(v, 2) + ' de 8' + (narrow ? '' : ' · ' + note), pad + labW + bw2 + 8, yy + 12, { size: 11.5, font: 'mono', color: C.ink });
        });
        fy += 14 + 2 * 24 + 8;
        if (S.k !== 7) fy += para(g, 'Medido só com k=7; no TP4, k=3 aceitou 53% contra 30% com k=7.', pad, fy + 6, gw, { size: 11.5, color: C.muted, lh: 15 });
        fitH(fig, Math.ceil(fy + 10), Math.ceil(fy + 10), w);
      });
    })();

    /* ============================================ F8 · linha do tempo do boot */
    (function () {
      const fig = root.querySelector('#fig-boot');
      const PH = [
        { a: 0, b: 6, lab: 'container → 1º log', sh: '6s', c: 'muted' },
        { a: 6, b: 32, lab: 'init Python · engine · NCCL', sh: 'init', c: 'b' },
        { a: 32, b: 71, lab: 'stream de pesos (checkpoint inteiro)', sh: 'pesos', c: 'd' },
        { a: 71, b: 78, lab: 'perfil KV', sh: 'KV', c: 'a' },
        { a: 78, b: 89, lab: 'graphs + warmup', sh: 'graphs', c: 'c' },
        { a: 89, b: 99, lab: 'API sobe', sh: 'API', c: 'e' },
      ];
      const LOGS = [
        [0, 'starting worker on zurih@10.0.0.2 … · starting head (vLLM API :8888)'],
        [6, '[glm53-exl3-head] launching: vllm serve /root/.cache/huggingface/hub/…'],
        [32, 'InstantTensor: cada rank lê o checkpoint a 5,5–5,9 GB/s'],
        [71, 'Model loading took 79.65 GiB'],
        [74, 'exl3 e2 diag schema=2 … effective_tier=grouped'],
        [78, '[glm53-kv-capacity-log] … · GPU KV cache size: …'],
        [89, 'CUDA graphs capturados: 1 2 4 8 16 24 32'],
        [99, 'health check passed after 99s — server is up'],
        [104, 'boot shape warmup + canário: "Reply with OK." → OK'],
        [110, 'READY'],
      ];
      const DIE = [[40, 'worker: processo morreu (simulação)'], [70, "worker container 'glm53-exl3-worker' not running on zurih@10.0.0.2 (3 consecutive checks)"]];
      const CUTS = [
        ['.pth importava o vLLM em todo interpretador', 100, 6, 'container → 1º log'],
        ['cache de JIT da NVIDIA não persistido', 30, 0, 'captura do DFlash2'],
        ['dry-capture descartada (CG_ESTIMATE=0)', 18, 7, 'perfil de KV'],
        ['stop gracioso → kill-first paralelo', 15, 0, '~−15 s'],
      ];
      const S = { t: 0, manual: false, die: false, cuts: true, t0: null, first: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'tempo', min: 0, max: 112, step: 1, value: 0, fmt: (v) => v + ' s', onInput: (v) => { S.t = v; S.manual = true; } });
      UI.button(ctl, '▶ reproduzir', () => { S.manual = false; S.t0 = null; }, { primary: true });
      UI.toggle(ctl, { label: 'simular: worker morre aos 40 s', value: false, onChange: (v) => { S.die = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'o que foi cortado (259 → 122 s)', value: true, onChange: (v) => { S.cuts = v; } });
      V.fig(fig, (g, w, h, t) => {
        fitH(fig, 450, 530, w);
        const C = V.C(); const narrow = w < 600; const pad = 14;
        if (!S.manual) { if (S.t0 == null) { S.t0 = t - (S.first ? 7 : 0); S.first = false; } S.t = Math.min(112, ((t - S.t0) * 8) % 124); }
        const T = S.t;
        const ax0 = narrow ? pad : pad + 104, ax1 = w - pad, TX = (s) => ax0 + (s / 112) * (ax1 - ax0);
        V.text(g, 'restart 259 → 122 s · /health 230 → 99 s (2026-09-19)', pad, 16, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const lh = narrow ? 30 : 34, gapL = narrow ? 26 : 14, y0 = narrow ? 44 : 30;
        const lanes = ['launcher (host)', 'head · rank 0', 'worker · rank 1'];
        const LY = (i) => y0 + i * (lh + gapL);
        const abortT = 70;
        const dead = S.die && T >= 40;
        lanes.forEach((ln, i) => {
          const y = LY(i);
          if (narrow) V.text(g, ln, pad, y - 6, { size: 11, weight: 600, color: C.ink });
          else V.text(g, ln, pad, y + lh / 2 + 4, { size: 11.5, weight: 600, color: C.ink });
          V.box(g, ax0, y, ax1 - ax0, lh, { r: 7, fill: V.alpha(C.muted, 0.07) });
        });
        const barLab = (a, b, y, lab, sh, col, alpha) => {
          const x0 = TX(a), x1 = TX(b);
          V.box(g, x0 + 1, y + 3, Math.max(2, x1 - x0 - 2), lh - 6, { r: 5, fill: V.alpha(col, alpha) });
          const full = tw(g, lab, 11, 'sans', 600) + 10 < x1 - x0 ? lab : tw(g, sh, 11, 'sans', 600) + 6 < x1 - x0 ? sh : '';
          if (full) V.text(g, full, (x0 + x1) / 2, y + lh / 2 + 1, { size: 11, weight: 600, align: 'center', base: 'middle', color: C.ink });
        };
        [1, 2].forEach((ri) => {
          const y = LY(ri);
          PH.forEach((p) => {
            if (ri === 2 && p.sh === 'API') return;
            let a = p.a, b = p.b;
            const stopAt = S.die ? (ri === 2 ? 40 : abortT) : Infinity;
            if (a >= stopAt) return;
            b = Math.min(b, stopAt);
            const col = p.c === 'muted' ? C.muted : C[p.c];
            const act = T >= a && T < b;
            barLab(a, b, y, p.lab, p.sh, col, T >= b ? 0.55 : act ? 0.85 : 0.18);
          });
          if (ri === 2 && !S.die) { g.setLineDash([4, 4]); V.box(g, TX(89) + 1, y + 3, TX(99) - TX(89) - 2, lh - 6, { r: 5, fill: 'transparent', stroke: V.alpha(C.b, 0.7) }); g.setLineDash([]); if (!narrow) V.text(g, 'sem API', (TX(89) + TX(99)) / 2, y + lh / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.muted }); }
          if (ri === 2 && S.die && T >= 40) { g.save(); V.rr(g, TX(40), y + 3, ax1 - TX(40), lh - 6, 5); g.clip(); g.strokeStyle = V.alpha(C.red, 0.6); g.lineWidth = 1.5; for (let q = TX(40) - lh; q < ax1; q += 8) { g.beginPath(); g.moveTo(q, y + lh); g.lineTo(q + lh, y); g.stroke(); } g.restore(); V.text(g, '✕', TX(40), y + lh / 2 + 1, { size: 16, weight: 700, align: 'center', base: 'middle', color: C.red }); }
        });
        // launcher lane
        const yL = LY(0);
        const pollEnd = S.die ? Math.min(T, abortT) : Math.min(T, 99);
        for (let s = 1; s <= pollEnd; s += narrow ? 3 : 2) V.dot(g, TX(s), yL + lh - 7, 1.6, V.alpha(C.ink, 0.5));
        V.text(g, narrow ? 'docker run · poll /health 1 s' : 'docker run → poll do /health a cada 1 s', TX(0) + 4, yL + 13, { size: 11, color: C.muted, maxW: TX(96) - TX(0) });
        if (S.die) {
          [50, 60, 70].forEach((s, k) => { if (T >= s) V.text(g, '✕' + (k + 1), TX(s), yL + lh / 2 + 1, { size: 12, weight: 700, align: 'center', base: 'middle', color: C.red }); });
          if (T >= abortT) { const lx = TX(abortT) + 20; V.text(g, narrow ? 'abortado' : 'abortado em ~30 s (não 3600 s)', lx, yL + lh / 2 + 1, { size: 11, weight: 700, base: 'middle', color: C.red, maxW: ax1 - lx - 2 }); }
        } else {
          if (T >= 99) { V.text(g, '/health ✓', TX(99), yL + lh / 2 + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.e }); }
          g.setLineDash([4, 3]); V.box(g, TX(100), yL + 5, TX(109) - TX(100), lh - 10, { r: 5, fill: V.alpha(C.f, T >= 100 ? 0.25 : 0.06), stroke: V.alpha(C.f, 0.8) }); g.setLineDash([]);
          if (T >= 110) V.text(g, '★', TX(110.5), yL + lh / 2 + 1, { size: 14, align: 'center', base: 'middle', color: C.d });
        }
        // eixo
        const ay = LY(2) + lh + 16;
        [0, 20, 40, 60, 80, 99].forEach((s) => { V.line(g, TX(s), ay - 8, TX(s), ay - 3, { color: C.muted, lw: 1 }); V.text(g, s + 's', TX(s), ay + 9, { size: 11, align: 'center', font: 'mono', color: C.muted }); });
        if (!narrow && !S.die) V.text(g, 'warmup + canário (duração não publicada) ▾', ax1, 16, { size: 11, align: 'right', color: C.f });
        // playhead
        const phx = TX(T);
        V.line(g, phx, y0 - 6, phx, LY(2) + lh + 4, { color: C.ink, lw: 1.6 });
        V.glow(g, phx, y0 - 6, 9, C.d, 1);
        // log
        const logs = S.die ? LOGS.filter((l) => l[0] < 40).concat(DIE) : LOGS;
        let cur = logs[0]; logs.forEach((l) => { if (T >= l[0]) cur = l; });
        const ly = ay + 22, lbh = narrow ? 44 : 30;
        V.box(g, pad, ly, w - 2 * pad, lbh, { r: 8, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.muted, 0.35) });
        const lc = S.die && T >= 40 ? C.red : C.ink;
        if (narrow) para(g, '[' + cur[0] + 's] ' + cur[1], pad + 10, ly + 18, w - 2 * pad - 20, { size: 11, font: 'mono', color: lc, lh: 15 });
        else V.text(g, '[' + String(cur[0]).padStart(3, ' ') + 's] ' + cur[1], pad + 10, ly + lbh / 2 + 1, { size: 12, font: 'mono', base: 'middle', color: lc, maxW: w - 2 * pad - 20 });
        // painel inferior
        let py = ly + lbh + 20;
        py += para(g, 'Nos 99 s: 26 s de init, 39 s de pesos (cada rank lê o checkpoint inteiro), 7 s de perfil de KV, 11 s de graphs + warmup, ~10 s de API. A próxima alavanca seria um loader que leia só a metade de cada rank (~15–18 s).', pad, py, w - 2 * pad, { size: 12, color: C.ink, lh: 16 }) + 14;
        if (S.cuts) {
          V.text(g, 'O QUE FOI CORTADO (ANTES → DEPOIS, s)', pad, py, { size: 11, font: 'mono', weight: 700, color: C.c });
          py += 10;
          const rowH = narrow ? 30 : 26, labW = narrow ? 0 : 290, bx = pad + labW, bw = w - pad - bx - 60;
          CUTS.forEach(([lab, before, after, note], k) => {
            const yy = py + k * rowH;
            if (narrow) V.text(g, lab, pad, yy + 10, { size: 11, color: C.ink, maxW: w - 2 * pad });
            else V.text(g, lab, pad, yy + 14, { size: 11.5, color: C.ink, maxW: labW - 10 });
            const by = narrow ? yy + 14 : yy + 4, bhh = narrow ? 10 : 14;
            V.box(g, bx, by, (before / 100) * bw, bhh, { r: 4, fill: V.alpha(C.red, 0.18), stroke: V.alpha(C.red, 0.5) });
            if (after > 0) V.box(g, bx, by, (after / 100) * bw, bhh, { r: 4, fill: V.alpha(C.e, 0.85) });
            V.text(g, k === 3 ? note : before + ' → ' + after, bx + (before / 100) * bw + 8, by + bhh / 2 + 1, { size: 11, font: 'mono', base: 'middle', color: C.ink });
          });
        }
      });
    })();

    /* ============================================ F9 · números medidos */
    (function () {
      const fig = root.querySelector('#fig-numeros');
      let view = 'decode', tv = 0;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Métrica', options: [{ v: 'decode', label: 'Decode' }, { v: 'prefill', label: 'Prefill frio' }, { v: 'conc', label: 'Concorrência' }, { v: 'kld', label: 'Qualidade (KLD)' }], value: 'decode', onChange: (v) => { view = v; tv = 0; } });
      const DEC = [
        { hd: 'PROSA · 1 SEQUÊNCIA (tok/s)', c: 'c' },
        { l: 'DFlash2 k=7', v: 27.1, tag: '08-30 · lab · aceita 0,34' },
        { l: '+ adaptive-k + FP8 denso', v: 32.1, tag: '09-08' },
        { l: '+ MoE cooperativo', v: 36.1, tag: '09-17 · sparkDash' },
        { hd: 'ESTRUTURADO · 1 SEQUÊNCIA (tok/s)', c: 'a' },
        { l: 'DFlash2 k=7', v: 65.1, tag: '08-30 · lab · aceita 0,96' },
        { l: 'braço A (sem coop)', v: 72.03, s: '72,03', tag: '09-16 · A/B/A' },
        { l: '+ MoE cooperativo geo 1', v: 77.29, s: '77,29', tag: '09-16 · A/B/A' },
        { hd: 'BASELINE MTP · PROMPT NÃO DECLARADO', c: 'muted' },
        { l: 'MTP k=2 (rollback)', v: 24.6, s: '~24,6', tag: 'baseline do README' },
      ];
      const PRE = [
        { hd: 'PROMPT DE 8k · tok/s (prompt ÷ TTFT)', c: 'd' },
        { l: 'MNBT 1024', v: 772, tag: '08-29' },
        { l: 'MNBT 2048', v: 938, tag: '08-30' },
        { l: 'E2 (PR77) · MNBT 2048', v: 1132.32, s: '1132,32', tag: '09-01' },
        { l: 'E3 · MNBT 7168', v: 1492.1, s: '1492,1', tag: '09-07' },
        { hd: 'PROMPT DE 256k', c: 'b' },
        { l: 'MNBT 1024', v: 936, tag: '08-29' },
        { l: 'MNBT 2048', v: 973, tag: '08-30' },
        { l: 'E3 · MNBT 7168', v: 1516.8, s: '1516,8', tag: '09-07' },
      ];
      const CONC = { x: [1, 2, 3, 4], stream: [36.1, 25.0, 22.3, 19.4], agg: [37.1, 51.1, 65.8, 75.3], ttft: [333, 365, 405, 401] };
      const KLD = [
        { l: 'TR3 K6 (6 bpw)', gb: 254, k: 0.013723, side: 'l' },
        { l: 'FP8 oficial, cross-stack', gb: 328, k: 0.020615, side: 'l' },
        { l: 'EXL3 4 bpw (servido)', gb: 176, k: 0.024555, side: 'r', hi: true },
        { l: 'FP8 oficial, stack v44', gb: 328, k: 0.024629, side: 'l' },
        { l: 'NVFP4 (v44)', gb: 180, k: 0.060535, side: 'r' },
      ];
      const hbars = (g, w, h, C, rows, max, grow, narrow) => {
        const pad = 14, labW = narrow ? 0 : 196, tagW = narrow ? 0 : 150;
        const x0 = pad + labW, x1 = w - pad - tagW - 44;
        const rowH = narrow ? 36 : 30;
        let y = 18, hdCol = C.ink;
        rows.forEach((r) => {
          if (r.hd) { if (y > 18) y += 10; hdCol = C[r.c]; V.text(g, r.hd, pad, y + 4, { size: 11, font: 'mono', weight: 700, color: hdCol }); y += 20; return; }
          const by = narrow ? y + 13 : y;
          if (narrow) V.text(g, r.l + ' · ' + r.tag, pad, y + 7, { size: 11, color: C.ink, maxW: w - 2 * pad });
          else V.text(g, r.l, pad, y + 12, { size: 12, color: C.ink, maxW: labW - 8 });
          const bw = ((r.v / max) * (x1 - x0)) * grow;
          V.box(g, x0, by, Math.max(2, bw), narrow ? 14 : 18, { r: 5, fill: V.alpha(hdCol, 0.8) });
          V.text(g, r.s || br(r.v, r.v > 100 ? 0 : 1), x0 + bw + 6, by + (narrow ? 7 : 9) + 1, { size: 11.5, font: 'mono', weight: 700, base: 'middle', color: C.ink });
          if (!narrow) V.text(g, r.tag, w - pad, y + 12, { size: 11, align: 'right', color: C.muted, maxW: tagW });
          y += rowH;
        });
        return y;
      };
      V.fig(fig, (g, w, h, t, dt) => {
        tv += dt; const grow = V.ease.out(Math.min(1, tv / 0.8));
        fitH(fig, 390, 470, w);
        const C = V.C(); const narrow = w < 600; const pad = 14;
        if (view === 'decode') {
          const y = hbars(g, w, h, C, DEC, 80, grow, narrow);
          if (h - y > 24) para(g, 'Mesma receita, datas e harnesses diferentes: compare dentro de cada grupo.', pad, y + 10, w - 2 * pad, { size: 11.5, color: C.muted });
        } else if (view === 'prefill') {
          const y = hbars(g, w, h, C, PRE, 1600, grow, narrow);
          if (h - y > 24) para(g, 'Prompts frios com salt único (sem hit de APC). Datas e configs diferentes: evolução, não A/B.', pad, y + 10, w - 2 * pad, { size: 11.5, color: C.muted });
        } else if (view === 'conc') {
          const cx0 = pad + 40, cx1 = w - pad - 10, cy0 = 40, cy1 = h - 70;
          V.text(g, 'SPARKDASH 2026-09-17 · PROSA · 850k · KV 14 GiB · adaptive-k + FP8 + coop', pad, 18, { size: 11, font: 'mono', weight: 700, color: C.a, maxW: w - 2 * pad });
          for (let v = 0; v <= 80; v += 20) { const yy = V.lerp(cy1, cy0, v / 80); V.line(g, cx0, yy, cx1, yy, { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, String(v), cx0 - 6, yy + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); }
          const slot = (cx1 - cx0) / 4, bwid = Math.min(58, slot * 0.5);
          const pts = [];
          CONC.x.forEach((n, i) => {
            const xc = cx0 + slot * (i + 0.5);
            const bh = ((CONC.agg[i] / 80) * (cy1 - cy0)) * grow;
            V.box(g, xc - bwid / 2, cy1 - bh, bwid, bh, { r: 6, fill: V.alpha(C.a, 0.75) });
            V.text(g, br(CONC.agg[i]), xc, cy1 - bh - 6, { size: 11.5, font: 'mono', weight: 700, align: 'center', color: C.ink });
            V.text(g, '×' + n, xc, cy1 + 16, { size: 12, weight: 700, align: 'center', color: C.ink });
            V.text(g, 'TTFT ' + CONC.ttft[i] + ' ms', xc, cy1 + 32, { size: 11, align: 'center', font: 'mono', color: C.muted });
            pts.push([xc, V.lerp(cy1, cy0, (CONC.stream[i] / 80) * grow), cy1 - bh]);
          });
          g.strokeStyle = C.c; g.lineWidth = 2.5; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke();
          pts.forEach((p, i) => {
            V.dot(g, p[0], p[1], 4.5, C.c);
            const lab = br(CONC.stream[i]), lw2 = tw(g, lab, 11, 'mono', 700) + 10;
            const up = p[1] - 30 > p[2] + 2, ty = up ? p[1] - 18 : p[1] + 18;
            V.box(g, p[0] - lw2 / 2, ty - 9, lw2, 17, { r: 5, fill: V.alpha(C.bg, 0.92), stroke: V.alpha(C.c, 0.8), lw: 1 });
            V.text(g, lab, p[0], ty, { size: 11, font: 'mono', color: C.c, weight: 700, align: 'center', base: 'middle' });
          });
          const lgY = h - 20;
          V.box(g, pad, lgY - 9, 12, 10, { r: 3, fill: V.alpha(C.a, 0.75) }); V.text(g, 'agregado (tok/s)', pad + 18, lgY, { size: 11.5, color: C.ink });
          V.dot(g, pad + (narrow ? 140 : 160), lgY - 4, 4.5, C.c); V.text(g, 'por stream', pad + (narrow ? 150 : 170), lgY, { size: 11.5, color: C.ink });
        } else {
          const legH = narrow ? KLD.length * 17 + 14 : 0;
          const cx0 = pad + (narrow ? 40 : 52), cx1 = w - pad - 10, cy0 = 34, cy1 = h - 44 - legH;
          const GX = (gb) => V.lerp(cx0, cx1, (gb - 150) / 200), GY = (k) => V.lerp(cy1, cy0, k / 0.065);
          V.text(g, 'KLD(teacher‖modelo) × bytes do checkpoint · 51.175 posições', pad, 16, { size: 11, font: 'mono', weight: 700, color: C.d, maxW: w - 2 * pad });
          for (let k = 0; k <= 0.06; k += 0.02) { const yy = GY(k); V.line(g, cx0, yy, cx1, yy, { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, br(k, 2), cx0 - 6, yy + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); }
          [150, 200, 250, 300, 350].forEach((gb) => V.text(g, gb + ' GB', GX(gb), cy1 + 16, { size: 11, align: 'center', font: 'mono', color: C.muted }));
          V.line(g, GX(256), cy0, GX(256), cy1, { color: C.red, lw: 1.5, dash: [5, 4] });
          V.text(g, '2 × 128 GB (bruto)', GX(256) + 6, cy0 + 10, { size: 11, color: C.red, weight: 600 });
          KLD.forEach((p, n) => {
            const x = GX(p.gb), yy = V.lerp(cy1, GY(p.k), grow);
            if (p.hi) V.glow(g, x, yy, 18, C.a, 0.9 + 0.1 * Math.sin(t * 3));
            V.dot(g, x, yy, p.hi ? 6.5 : 5, p.hi ? C.a : C.b);
            if (narrow) {
              // no mobile: número ao lado do ponto e legenda em lista abaixo do gráfico
              const tx = p.side === 'r' ? x + 12 : x - 12;
              V.text(g, String(n + 1), tx, yy + 4, { size: 12, weight: 700, align: p.side === 'r' ? 'left' : 'right', color: p.hi ? C.a : C.ink });
            } else {
              const lab = p.l + ' · ' + br(p.k, 4);
              const lx = p.side === 'r' ? x + 10 : x - 10;
              V.text(g, lab, lx, yy + 4, { size: 12, align: p.side === 'r' ? 'left' : 'right', color: C.ink, weight: p.hi ? 700 : 400, maxW: p.side === 'r' ? cx1 - lx : lx - cx0 });
            }
          });
          V.text(g, '↓ melhor', cx0 + 4, cy1 - 6, { size: 11, color: C.muted });
          if (narrow) {
            let ly = cy1 + 38;
            KLD.forEach((p, n) => {
              V.text(g, (n + 1) + ' · ' + p.l, pad, ly, { size: 11.5, color: C.ink, weight: p.hi ? 700 : 400, maxW: w - 2 * pad - 110 });
              V.text(g, p.gb + ' GB · ' + br(p.k, 4), w - pad, ly, { size: 11.5, font: 'mono', align: 'right', color: p.hi ? C.a : C.muted, weight: p.hi ? 700 : 400 });
              ly += 17;
            });
          }
        }
      });
    })();

    /* ============================================ F10 · mesa de som dos perfis */
    (function () {
      const fig = root.querySelector('#fig-perfis');
      const PROF = {
        tp2: { name: '.env.example · TP2 (default)', v: { ctx: 850000, seqs: 4, mnbt: 7168, util: 0.85, kv: 11, k: 7, dtp: 2 }, kvLab: '11 GiB', lamps: [['FP8 denso', 'all', 1], ['KDA BF16', 'on', 1], ['E3 prefill', 'on', 1], ['MOE_FAST', 'off', 0], ['adaptive-k', 'off', 0], ['spinwait', 'stock', 0], ['retenção', 'densa', 0], ['InstantTensor', 'on', 1], ['fair v7', 'on (v5 no retrato)', 1]], why: 'Contexto máximo com 4 gerações em voo: pool fixo de 11 GiB com páginas compactas (1,45× de 850k), FP8 denso em todos os grupos e prefill E3 com MNBT 7168.' },
        long: { name: 'examples/tp2-long-coding.env', v: { ctx: 262144, seqs: 2, mnbt: 1024, util: 0.865, kv: 12.8, k: 7, dtp: 2 }, kvAuto: true, kvSub: ['≈12,7–12,9 GiB', '≈12,8'], lamps: [['FP8 denso', 'dense,kda', 1], ['KDA BF16', 'on', 1], ['E3 prefill', 'on', 1], ['MOE_FAST', 'on', 1], ['adaptive-k', 'off', 0], ['spinwait', '16 ms', 1], ['retenção', '14336/0 (não qualificada)', 2], ['prompt-tokens-details', 'on', 1]], why: 'Poucos usuários com históricos longos: 262k, 2 seqs, chunks de 1024 para manter o decode responsivo, pool automático (≈12,7–12,9 GiB) e retenção esparsa 14336/0, que não foi qualificada.' },
        tp3: { name: '.env.tp3.example · 3 nós', v: { ctx: 1000000, seqs: 4, mnbt: 7168, util: 0.8, kv: 40, k: 7, dtp: 1 }, kvLab: '40 GiB*', lamps: [['FP8 denso', 'dense,kda', 1], ['KDA BF16', 'off', 0], ['E3 prefill', 'on', 1], ['MOE_FAST', 'forçado off', 0], ['cabeças', '66 (22/rank)', 1], ['expert parallel', '96/rank', 1], ['mm-encoder', 'data', 1], ['retenção SWA', '0', 1]], why: '3 nós: 66 cabeças via --hf-overrides, expert parallel (96 experts por rank), drafter só no rank 0 e um pool fixo por topologia (* 40 GiB no README, 35 GiB no cluster medido: divergência).' },
        tp4: { name: '.env.tp4.example · 4 nós (experimental)', v: { ctx: 850000, seqs: 8, mnbt: 2048, util: 0.75, kv: null, k: 3, dtp: 4 }, kvAuto: true, kvSub: ['sem medida', '—'], lamps: [['FP8 denso', 'não encaminha', 0], ['E3 prefill', 'on desde e5dc885', 1], ['adaptive-k', 'opt-in explícito', 0], ['TEMP_ROWS', '32 (com E3)', 1], ['slice sparse-MLA', '0 (opcional 64)', 0], ['retenção SWA', '0', 1]], why: 'Experimental: k=3 (aceita 53% contra 30% com k=7), 8 seqs, MNBT 2048 e util 0,75. O contexto herda 850k do .env (o script usaria 1M); o launcher remove a reserva de KV do .env de TP2, então o pool é perfilado; no retrato, o start-tp4.sh não encaminhava E3, adaptive-k nem FP8 denso. Depois dele, o E3 virou padrão no TP4 (e5dc885; prefill frio +35/+38/+41% a 8K/32K/100K, medida de contribuidor), o adaptive-k virou opt-in explícito do .env.tp4 (1e8f928) e o esforço de raciocínio padrão passou a valer (4709bc5); o FP8 denso continua sem encaminhamento.' },
      };
      const FAD = [
        { k: 'ctx', lab: 'contexto', sh: 'ctx', min: Math.log(100000), max: Math.log(1100000), map: Math.log, fmt: (v) => (v >= 1e6 ? '1M' : Math.round(v / 1000) + 'k') },
        { k: 'seqs', lab: 'seqs', sh: 'seqs', min: 0, max: 16, fmt: (v) => String(v) },
        { k: 'mnbt', lab: 'MNBT', sh: 'MNBT', min: 0, max: 8192, red: 8000, fmt: (v) => String(v) },
        { k: 'util', lab: 'util', sh: 'util', min: 0.7, max: 0.9, red: 0.87, fmt: (v) => br(v, 3) },
        { k: 'kv', lab: 'pool de KV', sh: 'KV', min: 0, max: 40, fmt: null },
        { k: 'k', lab: 'k DFlash', sh: 'k', min: 0, max: 8, fmt: (v) => String(v) },
        { k: 'dtp', lab: 'draft TP', sh: 'dTP', min: 0, max: 4, fmt: (v) => String(v) },
      ];
      let cur = 'tp2';
      const A = {};
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Perfil', options: [{ v: 'tp2', label: '.env.example' }, { v: 'long', label: 'long-coding' }, { v: 'tp3', label: 'TP3' }, { v: 'tp4', label: 'TP4' }], value: 'tp2', onChange: (v) => { cur = v; } });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const narrow = w < 600; const pad = 14;
        const P = PROF[cur];
        V.text(g, P.name, pad, 22, { size: narrow ? 14 : 16, font: 'display', weight: 700, color: C.ink, maxW: w - 2 * pad });
        const fw = (w - 2 * pad) / FAD.length, yT = 62, yB = narrow ? 206 : 246;
        FAD.forEach((f, i) => {
          const raw = P.v[f.k] == null ? (f.min + f.max) / 2 : P.v[f.k];
          const target = f.map ? f.map(raw) : raw;
          A[f.k] = near(A[f.k], target, dt, 5);
          const auto = f.k === 'kv' && P.kvAuto, ghost = auto && P.v[f.k] == null;
          const u = V.clamp((A[f.k] - f.min) / (f.max - f.min), 0, 1);
          const x = pad + fw * (i + 0.5);
          V.text(g, narrow ? f.sh : f.lab, x, yT - 12, { size: 11, align: 'center', weight: 600, color: C.ink, maxW: fw - 4 });
          V.box(g, x - 3, yT, 6, yB - yT, { r: 3, fill: V.alpha(C.muted, 0.25) });
          if (f.red != null) { const ur = (f.red - f.min) / (f.max - f.min); V.box(g, x - 3, yT, 6, (1 - ur) * (yB - yT), { r: 3, fill: V.alpha(C.red, 0.7) }); }
          for (let q = 0; q <= 4; q++) V.line(g, x - 9, V.lerp(yB, yT, q / 4), x - 5, V.lerp(yB, yT, q / 4), { color: V.alpha(C.muted, 0.5), lw: 1 });
          const ky = V.lerp(yB, yT, u), col = C.series[i % 6], kw = narrow ? 24 : 30;
          if (auto) {
            g.setLineDash([3, 3]);
            V.box(g, x - kw / 2, ky - 7, kw, 14, { r: 4, fill: V.alpha(col, ghost ? 0.06 : 0.22), stroke: col, lw: 1.5 });
            g.setLineDash([]);
            if (ghost) V.text(g, '?', x, ky + 1, { size: 11, weight: 700, align: 'center', base: 'middle', color: col });
          } else {
            V.box(g, x - kw / 2, ky - 7, kw, 14, { r: 4, fill: col, stroke: V.alpha(C.ink, 0.35), lw: 1 });
            V.line(g, x - (narrow ? 7 : 9), ky, x + (narrow ? 7 : 9), ky, { color: onFill(C), lw: 1.5 });
          }
          const val = f.k === 'kv' ? (auto ? (narrow ? 'auto' : 'auto (perfilado)') : P.kvLab) : f.fmt(P.v[f.k]);
          V.text(g, val, x, yB + 18, { size: 11, align: 'center', font: 'mono', weight: 700, color: C.ink, maxW: fw - 2 });
          if (auto) V.text(g, P.kvSub[narrow ? 1 : 0], x, yB + 32, { size: 11, align: 'center', font: 'mono', color: C.muted, maxW: fw - 2 });
        });
        // lâmpadas
        let lx = pad, ly = yB + 46;
        P.lamps.forEach(([nm, val, st]) => {
          const lab = nm + ': ' + val, wd = tw(g, lab, 11.5, 'sans', 500) + 28;
          if (lx + wd > w - pad) { lx = pad; ly += 28; }
          const col = st === 1 ? C.a : st === 2 ? C.d : C.muted;
          V.box(g, lx, ly, wd, 22, { r: 11, fill: V.alpha(col, st ? 0.16 : 0.07), stroke: V.alpha(col, st ? 0.8 : 0.4) });
          V.dot(g, lx + 11, ly + 11, 4, st ? col : V.alpha(C.muted, 0.5));
          if (st) V.glow(g, lx + 11, ly + 11, 9, col, 0.5 + 0.3 * Math.sin(t * 3 + lx));
          V.text(g, lab, lx + 20, ly + 12, { size: 11.5, base: 'middle', color: C.ink });
          lx += wd + 6;
        });
        const whyH = para(g, P.why, pad, ly + 46, w - 2 * pad, { size: 12.5, color: C.ink, lh: 17 });
        const needH = Math.max(narrow ? 470 : 450, Math.ceil(ly + 40 + whyH + 8));
        fitH(fig, needH, needH, w);
      });
    })();

    /* ============================================ TensorFold · cor do selo (lida do tema atual) */
    let tfCol = null, tfDark = null;
    const TFC = (C) => { if (tfCol == null || tfDark !== C.dark) { tfCol = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; tfDark = C.dark; } return tfCol; };
    /** trenzinho: corpo na cor da linha e três janelas */
    const train = (g, x, y, col, C, sc) => {
      const s = sc || 1;
      V.box(g, x - 13 * s, y - 7 * s, 26 * s, 14 * s, { r: 5 * s, fill: col });
      for (let q = 0; q < 3; q++) V.box(g, x - 9.5 * s + q * 7 * s, y - 3.5 * s, 5 * s, 5 * s, { r: 1.2, fill: V.alpha(C.bg, 0.92) });
    };

    /* ============================================ F11 · o desvio: o mapa com o motor trocado */
    (function () {
      const fig = root.querySelector('#fig-desvio');
      if (!fig) return;
      const T = {
        motor: 'Motor', vllm: 'vLLM', tf: 'TensorFold', swap: '⇄ puxar a alavanca',
        count: 'Neste motor',
        cnt: (s, m, x) => s + ' iguais · ' + m + ' do motor' + (x ? ' · ' + x + ' some' : ''),
        hw: '2 × GB10 · cabo CX7 · EXL3 de 164 GiB', hwS: 'mesmo hardware + pesos',
        brV: 'vLLM · 25 overlays', brVS: 'vLLM', brF: 'TensorFold v0.6.0 · 53 patches', brFS: 'TensorFold',
        st: 'ESTAÇÃO', both: 'NOS DOIS', other: { v: 'no TensorFold: ', f: 'no vLLM: ' }, none: 'não existe',
        fixed: 'estação fixada', hint: 'clique numa estação ou na alavanca',
        trans: (a, b) => 'TROCANDO O MOTOR: ' + a + ' → ' + b,
        transBody: {
          f: 'Ficam as 6 estações do hardware e do modelo. Das 24 decisões de motor, 23 passam a ser do TensorFold (TP = 2 e DFlash2 mantêm o nome, não o mecanismo) e 1 some: o pad 512→576.',
          v: 'De volta ao vLLM: as 6 estações físicas continuam no lugar, e as 24 decisões voltam a ser as do vLLM, incluindo o pad 512→576 e as páginas de 3.584 tokens.',
        },
        lgV: 'decisão do vLLM', lgF: 'decisão do TensorFold',
      };
      const ROWS = [
        { name: 'LINHA DOS PESOS', c: 'd', st: [
          { hw: ['BF16/FP8', 'não cabem', 'BF16 ocupa ~642,6 GB e o FP8 oficial 328,3 GB: nenhum cabe em 2 × 128 GB. Vale para qualquer motor.'] },
          { hw: ['EXL3 4 bpw', 'experts', 'O mesmo checkpoint nos dois motores: experts roteados em EXL3 4 bpw, o resto em BF16, 164 GiB, ~80 GiB por rank.'] },
          { v: ['densos', 'FP8 canal', 'GLM53_DENSE_FP8=all: projeções densas em FP8 weight-only por canal, via Marlin. +4–5% tok/s e “NOT-WORSE” em KL.'],
            f: ['densos', 'q4 · MSE', 'DENSE=q4: 4 bits afins em grupos de 64, recorte de menor erro quadrático; cabeça em FP8. Sobre fp8: prosa 38,9 → 44,4 tok/s (um boot cada). Com perda declarada.'] },
          { v: ['kernels', 'sm_121a', 'A extensão do ExLlamaV3 (pin c5d9c657) é compilada para sm_121a, com os tiers E2 e E3 para os fat experts do prefill.'],
            f: ['kernels', 'da família', 'Os kernels EXL3 são do próprio motor, compilados por JIT na primeira partida de cada imagem, com o cache separado pelo hash da imagem.'] },
          { v: ['25 overlays', 'no boot', 'Imagem pinada por digest e carimbada; 25 overlays Python montados por bind mount e reaplicados, na mesma ordem, a cada docker run.'],
            f: ['53 diffs', 'no build', 'Os 53 patches são diffs aplicados no build (patch -p0, em ordem de nome). A tag é o hash dos patches, e mudar um patch exige outra imagem.'] },
        ] },
        { name: 'LINHA DA ESPECULAÇÃO', c: 'c', st: [
          { hw: ['decode', '= banda', 'Cada token de decode lê todos os pesos ativos: o gargalo é a banda de memória. Por isso os dois motores especulam.'] },
          { v: ['DFlash2', 'k = 7', 'O drafter propõe 7 tokens por bloco e o alvo verifica 8 posições por sequência. O adaptive-k, opt-in, verifica só 2, 4 ou 7.'],
            f: ['DFlash2', 'fnc7:0.3', 'Até 7 rascunhos, até o produto das chances sob o ruído com chave cair abaixo de 0,3. Sobre fc5:0.3: prosa 47,2 → 50,6 tok/s (dois boots cada).'] },
          { v: ['8 posições', 'uniformes', 'Os graphs FULL só aceitam decode uniforme: com adaptive-k, o comprimento verificado é o mínimo sobre o batch.'],
            f: ['copy drafts', 'até 15', 'Antes do DFlash2, rascunhos copiados do contexto quando os últimos 8 tokens já apareceram. Com até 15 por rodada, edição foi de 84 → 119 tok/s.'] },
          { v: ['graphs', '8·16·24·32', 'Com até 4 sequências, os passos de decode têm 8, 16, 24 ou 32 tokens, e cada tamanho precisa de um CUDA graph.'],
            f: ['graphs', '1 a 32', 'Um CUDA graph por contagem de linhas da janela, de 1 a 32, com posições e faixas em tabelas no device: um graph serve qualquer mistura de requisições.'] },
          { v: ['TEMP_ROWS', '= 32', 'O buffer do MoE fundido comporta 32 linhas por expert: 4 × 8 cabem num único lançamento compatível com CUDA graph.'],
            f: ['janela ≤ 32', '512 KiB', 'Até 16 linhas por requisição e 32 no lote; passou disso, as caudas mais longas são aparadas. 32 linhas × 16 KiB = 512 KiB, o teto do RoCE.'] },
        ] },
        { name: 'LINHA DISTRIBUÍDA', c: 'e', st: [
          { hw: ['2 nós', '1 GPU', 'Dois GB10, uma GPU em cada, ligados por um cabo ConnectX-7 em RoCE v2. É o que os dois motores têm.'] },
          { v: ['TP = 2', 'mp', 'Tensor parallel 2 com executor multiprocessing: head com API e EngineCore, worker headless. TP3 e TP4 existem em launchers próprios.'],
            f: ['TP = 2', 'só 2', 'Um rank por Spark: o rank 0 decide e serve a API, o rank 1 aplica. Na v1.3.2, o motor CUDA do GLM exigia exatamente dois ranks; desde a v1.4, três.'] },
          { v: ['all-reduce', 'NCCL', 'Dois all-reduces por camada, após a atenção e após o MoE, somando as metades parciais dos dois ranks.'],
            f: ['all-gather', '+ soma', 'Cada rank recebe as parciais fp32 do outro e soma na ordem dos ranks, rank 0 primeiro: os dois ficam com os mesmos bits, qualquer que seja o transporte.'] },
          { v: ['NCCL', 'verbs IB', 'O NCCL 2.30.7 da imagem leva as coletivas pelo transporte IB (NCCL_NET=IB) sobre RoCE v2, com interface, HCA e GID fixados por rank.'],
            f: ['RoCE', 'um tiro', 'Gathers de até 512 KiB numa só chamada de kernel, por RDMA WRITE: 11 µs contra 45 µs do NCCL para 16 KiB, +6% de decode.'] },
          { v: ['mesmos', 'graphs', 'Os ranks recebem o mesmo script e o mesmo ambiente, e nada confere isso na partida: se os graphs divergirem, o NCCL trava.'],
            f: ['config', 'conferida', 'Antes de carregar os pesos, os ranks trocam um vetor com as configurações que mudam coletivas ou rascunhos e recusam subir se ele divergir.'] },
        ] },
        { name: 'LINHA DA MEMÓRIA (UMA)', c: 'a', st: [
          { hw: ['CPU + GPU', '128 GB', 'Memória unificada: o que o motor pega sai do SO e do page cache, e faltar memória tende a congelar a máquina.'] },
          { v: ['util', '0,85', 'Com o pool fixo, a util não dimensiona o KV: é o piso de memória livre que o boot exige, 0,85 × 121,69 ≈ 103,4 GiB.'],
            f: ['reserva', '14,5 GiB', 'Orçamento = MemAvailable na partida − reserva. Uma janela que não cabe é recusada antes dos pesos, com a maior janela que caberia.'] },
          { v: ['Instant', 'Tensor', 'Loader direct-I/O: os pesos sobem em ~65–70 s, mas o KV perfilado perde ~4,4–5,6 GiB. Restart de 122 s.'],
            f: ['carga', '2–6 min', '~80 GiB por Spark em 2–6 min (~2,2 min com os pesos do worker por NFS); a primeira partida de cada imagem ainda compila kernels.'] },
          { v: ['KV fixo', '11 GiB', '--kv-cache-memory-bytes fixa 11 GiB = 434 IDs de bloco = 1.233.779 tokens, 1,45× de uma requisição de 850k.'],
            f: ['pool', '12,5 GiB', 'Teto do pool além da janela, por Spark. Na partida medida, 2.922.496 tokens; varia de ~2,1 a 2,9M com a memória livre.'] },
          { v: ['IDs de', 'bloco', 'Cada grupo do alocador híbrido paga IDs inteiros de 25,9 MiB; as páginas compactas levam 850k de ~536 para ~299 IDs.'],
            f: ['extents', '2.048', 'Cada requisição tem uma faixa contígua, múltipla de 2.048 tokens, que cresce, muda de lugar e é compactada. O estado KDA fica em slots, fora do pool.'] },
        ] },
        { name: 'LINHA DA ATENÇÃO', c: 'b', st: [
          { hw: ['MLA NoPE', '+ GB10', 'As 11 camadas MLA não têm RoPE (qk_rope_head_dim=0), e a GPU é um GB10 (SM121). Isso vale nos dois motores.'] },
          { v: ['1 kernel', 'sparse', 'O único kernel de sparse-MLA do SM12x, FLASHINFER_MLA_SPARSE_SM120, tem geometria fixa de 576 e só lê fp8_ds_mla.'],
            f: ['kernel', 'latente', 'O motor guarda só o latente de 512, absorve a query uma vez por linha e, acima de 2.051 tokens, atende aos 512 pools de maior nota do indexador.'] },
          { v: ['KV', 'fp8_ds_mla', '--kv-cache-dtype fp8 vira o registro fp8_ds_mla: 512 B em FP8 + 16 B de escalas + 128 B de RoPE.'],
            f: ['KV', 'kv8 e4m3', 'KV=fp8: 512 códigos e4m3 e uma escala fp32 potência de 2 por linha, que leva o máximo para [128, 256]: nada satura. Erro relativo de 2,65% por linha.'] },
          { v: ['pad', '512→576', 'O overlay acrescenta 64 zeros de RoPE à query e ao k_pe só para caber na geometria do kernel. É exato, mas custa 128 B por token e camada.'],
            f: ['pad', '512→576', 'Não existe: o kernel é do próprio motor e lê o latente de 512 como ele é. É o exemplo mais limpo de restrição que vinha do motor, e não do modelo.', 'x'] },
          { v: ['656 B', '/token', 'Por token e camada: 512 + 16 + 128 B, 24% a mais que os 528 B necessários. Nas 11 camadas, ~7,6 KB por token por rank.'],
            f: ['528 B', '/token', '512 códigos + 4 B de escala + 12 B de alinhamento: −19,5% de bytes de latente. Com o indexador, ≈6,2 KB por token por rank (estimativa).'] },
        ] },
        { name: 'LINHA DA SAÍDA', c: 'f', st: [
          { v: ['rejeição', 'p/q', 'Rejection sampling: aceita o rascunho com min(1, p/q). Preserva a distribuição do alvo, mas lote, kernel e especulação podem mudar os bits.'],
            f: ['amostra', 'com chave', 'Ruído Gumbel derivado de (semente, posição, token), sem estado. Verificar vira comparar: aceita enquanto o rascunho é igual à amostra.'] },
          { v: ['bits', 'variam', 'Sem perdas em distribuição, não em sequência: a mesma pergunta pode dar outra resposta com rascunhos, em lote ou com o cache.'],
            f: ['bits', '= serial', 'Contrato: com rascunhos = serial, 4 juntas = cada uma sozinha, retomada = fresca. Testado: rascunhado = serial 6/6, concorrência 22/22.'] },
          { v: ['APC', '3.584', 'O prefix caching reusa blocos com hash; no híbrido, os acertos se alinham a 3.584 tokens e a cauda é recalculada.'],
            f: ['grade 64', '+ replay', 'Estados guardados no último múltiplo de 64, no fim do system prompt e no fim do prompt, com a linha de logits: 64k idêntico, de 34 s para < 0,07 s.'] },
          { v: ['fair v5', 'misto', 'Prefill e decode dividem o passo; o overlay escolhe o chunk do novato para o passo caber em 2 s, com 30% do tempo de engine para o prefill.'],
            f: ['chunk', 'ou rodada', 'Um chunk de prompt nunca divide a forward com decode. Prompts que chegam juntos dividem um chunk: TTFT com 4 requisições de 590 para 340 ms.'] },
          { v: ['API', 'completa', 'A superfície do vLLM: logprobs, n>1, penalidades e chave de API Bearer opcional.'],
            f: ['API', 'menor', 'Sem logprobs e sem n>1 (HTTP 400), penalidades aceitas e ignoradas, sem chave de API: para expor, um proxy com autenticação.'] },
        ] },
      ];
      const AUTO = [[4, 3], [3, 1], [1, 1], [5, 1], [5, 2], [2, 2]];
      let eng = 'v', prev = 'v', ft = 99, flipT = -99, selT = -99, now = 0, sw = 0, user = null, auto = true, autoT = 0, autoK = AUTO.length - 1;
      const counts = (e) => { let s = 0, m = 0, x = 0; ROWS.forEach((R) => R.st.forEach((S) => { if (S.hw) s++; else if ((e === 'v' ? S.v : S.f)[3] === 'x') x++; else m++; })); return [s, m, x]; };
      const ctl = UI.controls(fig);
      let seg = null;
      function flip(v, byUser) {
        if (v === eng) return;
        prev = eng; eng = v; ft = 0; flipT = now; autoT = 0;
        if (seg) seg.set(v, true);
        if (byUser) auto = false;
        rdo.set(T.cnt(...counts(eng)));
      }
      seg = UI.seg(ctl, { label: T.motor, options: [{ v: 'v', label: T.vllm }, { v: 'f', label: T.tf }], value: 'v', onChange: (v) => flip(v, true) });
      UI.button(ctl, T.swap, () => flip(eng === 'v' ? 'f' : 'v', true), { primary: true });
      const rdo = UI.readout(ctl, T.count, T.cnt(...counts('v')));
      UI.legend(ctl, [['var(--glm)', T.lgV], ['var(--tf)', T.lgF]]);
      V.fig(fig, (g, w, h, t, dt, S, H) => {
        now = t;
        const C = V.C(), TF = TFC(C), narrow = w < 600;
        fitH(fig, 560, 714, w);
        ft += dt;
        const ENG = { v: C.glm, f: TF };
        const pad = 14, TS = narrow ? 66 : 58, detH = narrow ? 150 : 84;
        const top = TS + 6, rowH = (h - top - detH - 4) / 6;
        const x0 = pad + (narrow ? 24 : 36), x1 = w - pad - (narrow ? 24 : 36);
        const SX = (j) => x0 + (j * (x1 - x0)) / 4;
        const SY = (i) => top + i * rowH + 24;
        const mw = (x1 - x0) / 4 - 4;
        // ---------- auto: passeia por estações e troca o motor de tempos em tempos
        if (auto && !user) {
          autoT += dt;
          if (autoT > (ft < 3 ? 4.6 : 3.4)) { autoT = 0; autoK++; if (autoK >= AUTO.length) { autoK = 0; flip(eng === 'v' ? 'f' : 'v', false); } }
        }
        // ---------- trilho de cima: hardware → desvio → motor
        const yc = narrow ? 34 : 30;
        const hwLab = narrow ? T.hwS : T.hw;
        const cw = chip(g, pad, yc - 10, hwLab, { size: 11, font: 'sans', fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.muted, 0.5), color: C.ink, h: 20 });
        const lv = narrow ? T.brVS : T.brV, lf = narrow ? T.brFS : T.brF;
        const wv = tw(g, lv, 11, 'mono', 600) + 14, wf = tw(g, lf, 11, 'mono', 600) + 14;
        const xb = w - pad - Math.max(wv, wf) - 6;
        const xs = Math.min(pad + cw + (narrow ? 30 : 46), xb - 46);
        const yU = yc - 15, yD = yc + 15;
        sw = near(sw, eng === 'f' ? 1 : 0, dt, 8);
        V.line(g, pad + cw, yc, xs, yc, { color: V.alpha(C.ink, 0.55), lw: 4 });
        const branch = (yy, on, col) => {
          g.beginPath(); g.moveTo(xs, yc); g.bezierCurveTo(xs + 18, yc, xs + 14, yy, xs + 34, yy); g.lineTo(xb, yy);
          g.strokeStyle = on ? col : V.alpha(C.muted, 0.35); g.lineWidth = on ? 4 : 2; g.stroke();
        };
        branch(yU, eng === 'v', ENG.v); branch(yD, eng === 'f', ENG.f);
        chip(g, xb + 6, yU - 9, lv, { size: 11, h: 18, weight: 600, fill: V.alpha(ENG.v, eng === 'v' ? 0.2 : 0.06), stroke: V.alpha(ENG.v, eng === 'v' ? 0.9 : 0.35), color: eng === 'v' ? C.ink : C.muted });
        chip(g, xb + 6, yD - 9, lf, { size: 11, h: 18, weight: 600, fill: V.alpha(ENG.f, eng === 'f' ? 0.2 : 0.06), stroke: V.alpha(ENG.f, eng === 'f' ? 0.9 : 0.35), color: eng === 'f' ? C.ink : C.muted });
        // agulha do desvio
        const ang = V.lerp(Math.atan2(yU - yc, 30), Math.atan2(yD - yc, 30), sw);
        V.line(g, xs, yc, xs + Math.cos(ang) * 20, yc + Math.sin(ang) * 20, { color: ENG[eng], lw: 5 });
        V.dot(g, xs, yc, 4, C.ink);
        // alavanca
        const lx = xs - 16, ly = yc + (narrow ? 26 : 22);
        V.box(g, lx - 9, ly - 2, 18, 6, { r: 3, fill: V.alpha(C.muted, 0.6) });
        const la = V.lerp(-2.25, -0.9, sw), le = 18;
        V.line(g, lx, ly, lx + Math.cos(la) * le, ly + Math.sin(la) * le, { color: C.ink, lw: 2.5 });
        V.dot(g, lx + Math.cos(la) * le, ly + Math.sin(la) * le, 6, ENG[eng]);
        if (auto && !user) V.glow(g, lx + Math.cos(la) * le, ly + Math.sin(la) * le, 13, ENG[eng], 0.35 + 0.25 * Math.sin(t * 4));
        if (ft < 1.2) V.glow(g, lx + Math.cos(la) * le, ly + Math.sin(la) * le, 16, ENG[eng], 1 - ft / 1.2);
        // trem do trilho de cima
        const tu = Math.min(1, ft / 0.95);
        let tx, ty;
        if (tu < 0.45) { tx = V.lerp(pad + cw + 12, xs, tu / 0.45); ty = yc; }
        else { const u2 = (tu - 0.45) / 0.55; tx = V.lerp(xs + 6, xb - 22, u2); ty = V.lerp(yc, eng === 'v' ? yU : yD, Math.min(1, u2 * 3)); }
        train(g, tx, ty, ENG[eng], C, narrow ? 0.85 : 1);
        // ---------- cliques
        if (H.pointer.clicked) {
          if (H.pointer.y < TS + 2) flip(eng === 'v' ? 'f' : 'v', true);
          else {
            let best = null, bd = 28;
            for (let i = 0; i < 6; i++) for (let j = 0; j < 5; j++) { const d = Math.hypot(H.pointer.x - SX(j), H.pointer.y - SY(i) - 10); if (d < bd) { bd = d; best = [i, j]; } }
            if (best) { user = best; auto = false; selT = t; }
          }
        }
        const sel = user || AUTO[autoK];
        // ---------- linhas e estações
        ROWS.forEach((R, i) => {
          const col = C[R.c], y = SY(i), onRow = sel[0] === i;
          V.text(g, R.name, pad, y - 14, { size: 11, font: 'mono', weight: 700, color: col });
          V.line(g, x0, y, x1, y, { color: V.alpha(col, onRow ? 1 : 0.7), lw: onRow ? 5 : 3.5 });
          const st0 = 0.95 + i * 0.09, span = 0.15;
          if (ft > st0 - 0.2 && ft < st0 + 4 * span + 0.35) {
            const px = V.clamp(x0 + ((ft - st0) / span) * ((x1 - x0) / 4), x0 - 6, x1 + 6);
            train(g, px, y, col, C, 0.8);
          }
          R.st.forEach((Sx, j) => {
            const x = SX(j), dij = st0 + j * span;
            const p = V.clamp((ft - dij) / 0.4, 0, 1);
            const side = p < 0.5 ? prev : eng;
            const sy = Math.max(0.06, Math.abs(Math.cos(Math.PI * p)));
            const lab = Sx.hw || (side === 'v' ? Sx.v : Sx.f);
            const kind = Sx.hw ? 'hw' : lab[3] === 'x' ? 'ghost' : 'eng';
            const isSel = sel[0] === i && sel[1] === j;
            if (isSel) V.glow(g, x, y, 24, col, 0.85);
            if (kind === 'hw' && Math.abs(ft - dij) < 0.3) V.glow(g, x, y, 18, col, 1 - Math.abs(ft - dij) / 0.3);
            g.save(); g.translate(x, y); g.scale(1, kind === 'hw' ? 1 : sy);
            if (kind === 'hw') { g.beginPath(); g.arc(0, 0, 7.5, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.lineWidth = 3; g.strokeStyle = col; g.stroke(); }
            else if (kind === 'eng') { V.box(g, -7.5, -7.5, 15, 15, { r: 3.5, fill: C.bg, stroke: col, lw: 3 }); V.dot(g, 0, 0, 3.4, ENG[side]); }
            else { g.setLineDash([3, 3]); g.beginPath(); g.arc(0, 0, 7.5, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.lineWidth = 2; g.strokeStyle = V.alpha(C.muted, 0.8); g.stroke(); g.setLineDash([]); }
            g.restore();
            g.save(); g.translate(x, y + 28); g.scale(1, kind === 'hw' ? 1 : sy);
            const c1 = kind === 'ghost' ? C.muted : C.ink, c2 = kind === 'hw' ? C.ink : kind === 'ghost' ? C.muted : ENG[side];
            V.text(g, lab[0], 0, -7, { size: 11, align: 'center', color: c1, weight: isSel ? 700 : 500, maxW: mw });
            V.text(g, lab[1], 0, 7, { size: 11, align: 'center', color: c2, font: 'mono', weight: kind === 'eng' ? 600 : 500, maxW: mw });
            if (kind === 'ghost') {
              const sw1 = Math.min(mw, tw(g, lab[0], 11, 'sans', 500)) / 2 + 3, sw2 = Math.min(mw, tw(g, lab[1], 11, 'mono', 500)) / 2 + 3;
              V.line(g, -sw1, -11, sw1, -11, { color: C.red, lw: 1.6 });
              V.line(g, -sw2, 3, sw2, 3, { color: C.red, lw: 1.6 });
            }
            g.restore();
          });
        });
        // ---------- painel de detalhe
        const dy = h - detH + 2;
        V.box(g, pad, dy, w - 2 * pad, detH - 8, { r: 10, fill: V.alpha(C.ink, 0.05), stroke: V.alpha(C.muted, 0.3) });
        const inWave = t - flipT < 2.6 && selT < flipT;
        let title, other = '', ocol = C.muted, body, tcol = C.ink;
        if (inWave) {
          title = T.trans(prev === 'v' ? T.vllm : T.tf, eng === 'v' ? T.vllm : T.tf).toUpperCase(); tcol = ENG[eng];
          body = T.transBody[eng];
        } else {
          const R = ROWS[sel[0]], Sx = R.st[sel[1]];
          const lab = Sx.hw || (eng === 'v' ? Sx.v : Sx.f);
          title = R.name + ' · ' + T.st + ' ' + (sel[1] + 1) + ' · ' + (Sx.hw ? T.both : (eng === 'v' ? T.vllm : T.tf).toUpperCase());
          tcol = C[R.c]; body = lab[2];
          if (!Sx.hw) { const o = eng === 'v' ? Sx.f : Sx.v; other = T.other[eng] + (o[3] === 'x' ? T.none : o[0] + ' · ' + o[1]); ocol = ENG[eng === 'v' ? 'f' : 'v']; }
        }
        V.text(g, title, pad + 12, dy + 18, { size: 11, font: 'mono', weight: 700, color: tcol, maxW: w - 2 * pad - 24 });
        let by = dy + 38;
        if (other) {
          if (narrow) { V.text(g, other, pad + 12, dy + 35, { size: 11.5, color: ocol, weight: 600, maxW: w - 2 * pad - 24 }); by = dy + 56; }
          else V.text(g, other, w - pad - 12, dy + 18, { size: 11.5, align: 'right', color: ocol, weight: 600 });
        } else if (!narrow && !inWave) V.text(g, user ? T.fixed : T.hint, w - pad - 12, dy + 18, { size: 11, align: 'right', color: C.muted });
        para(g, body, pad + 12, by, w - 2 * pad - 24, { size: 12.5, color: C.ink, lh: 17 });
      });
    })();

    /* ============================================ F12 · dois ranks, uma verdade */
    (function () {
      const fig = root.querySelector('#fig-paridade');
      if (!fig) return;
      const T = {
        scen: 'Divergência', run: '▶ rodar de novo',
        stations: ['args', 'imagem', 'docker', 'partida', 'pesos', 'graphs', 'API'],
        ranks: ['rank 0', 'rank 1'], both: 'os dois ranks', eng: ['vLLM', 'TensorFold'],
        serving: 'servindo', oldPair: 'par antigo no ar', hang: 'NCCL espera…', dead: 'morreu',
        S: [
          { k: 'rank', label: 'Rank 1 diferente',
            v: { who: 1, tag: 'sem nº 18', at: 5, kind: 'hang', chip: 'trava nos graphs', txt: 'O rank 1 sobe sem o overlay nº 18, com GLM53_ADAPTIVE_K=ema. Nada confere a paridade na partida: o rank 0 captura graphs de decode para 3, 5 e 8 posições, o rank 1 só para 8, e as coletivas deixam de casar. Interpretação: como os dois processos seguem vivos, o fail-fast não dispara, e o launcher espera o /health até o READY_TIMEOUT de 3.600 s.' },
            f: { who: 1, tag: 'outro COPY_MAX', at: 3, kind: 'refuse', chip: 'recusa na partida', txt: 'O rank 1 sobe com outro TF_GLM_COPY_MAX. Antes de carregar os pesos, cada rank empacota num vetor de inteiros as configurações que mudam coletivas ou rascunhos; os dois fazem all-gather, comparam e recusam subir: “the two ranks were started with different settings”.' } },
          { k: 'img', label: 'Imagem velha no worker',
            v: { who: 1, tag: 'imagem velha', at: 1, kind: 'fix', chip: 'corrige e segue', txt: 'O launcher compara head e worker pelo image key, um hash das camadas (RootFS.Layers), e não pelo .Id, que difere entre overlay2 e containerd. Divergiu: o worker recebe a imagem por pull ou por docker save | ssh docker load, e os dois sobem com os mesmos bytes.' },
            f: { who: 1, tag: 'imagem velha', at: 1, kind: 'fix', chip: 'corrige e segue', txt: 'O preparo compara a imagem do worker por conteúdo (camadas, Env, Entrypoint, Cmd e WorkingDir) e só aceita imagem cujo rótulo tf.patches bata com o hash dos patches. Divergiu: pull, ou docker save | ssh docker load só das camadas que faltam.' } },
          { k: 'typo', label: 'Typo no restart',
            v: { who: 2, tag: 'typo', at: 0, kind: 'refuse', chip: 'recusa antes do stop', txt: 'A validação confere tipo e faixa de todos os knobs (MAX_MODEL_LEN ≤ 1e6, MAX_NUM_SEQS ≤ 4096, MNBT ≤ 8.388.608…) antes de parar os containers. Um typo falha ali, e o par que estava servindo continua servindo.' },
            f: { who: 2, tag: 'typo', at: 0, kind: 'refuse', chip: 'recusa antes do stop', txt: 'Os argumentos de ./start.sh restart passam pelo parser do próprio TensorFold, num container descartável e sem GPU, antes de parar o servidor. Um typo falha ali, e o par antigo continua no ar.' } },
          { k: 'down', label: 'Worker cai no boot',
            v: { who: 1, tag: 'morre', at: 4, kind: 'abort', chip: 'aborta em ~30 s', txt: 'O worker morre durante a carga dos pesos. O launcher confere a vida do worker a cada ~10 s e, depois de 3 falhas seguidas (~30 s), aborta a espera pelo /health. Sem esse fail-fast, esperaria até o READY_TIMEOUT de 3.600 s por um par quebrado.' },
            f: { who: 1, tag: 'morre', at: 4, kind: 'abort', chip: 'aborta em ~15 s', txt: 'Enquanto espera a API, o start.sh confere o rank 1 a cada 15 s e falha mostrando as últimas 25 linhas de log dos dois ranks. Em serviço, um erro trava o estado broken nos dois; com FOREGROUND=1, a queda de um rank derruba o outro, porque um rank sozinho esperaria o par para sempre.' } },
        ],
      };
      const STEP = 0.55;
      let si = 0, t0 = null, now = 0, auto = true, linesCache = { w: 0, n: 0 };
      const ctl = UI.controls(fig);
      const seg = UI.seg(ctl, { label: T.scen, options: T.S.map((s, i) => ({ v: i, label: s.label })), value: 0, onChange: (v) => { si = v; t0 = now; auto = false; } });
      UI.button(ctl, T.run, () => { t0 = now; auto = false; });
      V.fig(fig, (g, w, h, t, dt, S, H) => {
        now = t; if (t0 == null) t0 = t;
        const C = V.C(), TF = TFC(C), narrow = w < 600, pad = 14;
        const ENG = [C.glm, TF];
        const KC = { refuse: C.e, fix: C.e, abort: C.c, hang: C.red };
        // altura estável: o maior texto entre todos os cenários, na largura atual
        if (linesCache.w !== w) {
          let n = 1; T.S.forEach((s) => [s.v, s.f].forEach((e) => { n = Math.max(n, wrap(g, e.txt, w - 2 * pad - 16, 12).length); }));
          linesCache = { w, n };
        }
        const PH = 126 + linesCache.n * 16 + 8, gap = 10;
        fitH(fig, 2 * PH + gap + 12, 2 * PH + gap + 12, w);
        if (H.pointer.clicked) { t0 = t; auto = false; }
        const el = t - t0;
        const total = 6 * STEP + 5;
        if (el > total) { if (auto) { si = (si + 1) % T.S.length; seg.set(si, true); } t0 = t; }
        const sc = T.S[si];
        const xl = pad + (narrow ? 22 : 64), xr = w - pad - (narrow ? 14 : 24);
        const SX = (k) => xl + (k * (xr - xl)) / 6;
        [sc.v, sc.f].forEach((E, p) => {
          const y0 = 6 + p * (PH + gap), col = ENG[p];
          V.box(g, pad * 0.5, y0, w - pad, PH, { r: 10, fill: V.alpha(C.ink, 0.035), stroke: V.alpha(col, 0.35) });
          const ew = chip(g, pad + 2, y0 + 7, T.eng[p], { size: 11.5, weight: 700, font: 'sans', h: 20, fill: V.alpha(col, 0.18), stroke: col, color: C.ink });
          if (!narrow) chip(g, pad + 10 + ew, y0 + 7, '≠ ' + (E.who === 2 ? T.both : T.ranks[1]) + ': ' + E.tag, { size: 11, weight: 600, font: 'sans', h: 20, fill: V.alpha(C.red, 0.12), stroke: V.alpha(C.red, 0.7), color: C.red });
          const stopAt = E.kind === 'fix' ? 6 : E.at;
          const pr = Math.min(stopAt, Math.max(0, el / STEP));
          const caught = el / STEP >= E.at;
          const kc = KC[E.kind];
          if (caught) {
            const cwid = tw(g, E.chip, 11, 'sans', 700) + 16;
            chip(g, w - pad - 2 - cwid, y0 + 7, E.chip, { size: 11, weight: 700, font: 'sans', h: 20, fill: V.alpha(kc, 0.2), stroke: kc, color: C.ink });
          }
          const yl = [y0 + 52, y0 + 84];
          T.ranks.forEach((rn, r) => {
            if (!narrow) V.text(g, rn, pad + 2, yl[r] + 4, { size: 11, font: 'mono', color: C.muted });
            V.line(g, xl, yl[r], xr, yl[r], { color: V.alpha(C.muted, 0.45), lw: 2 });
            for (let k = 0; k < 7; k++) { V.dot(g, SX(k), yl[r], k <= pr + 0.01 ? 4 : 3, k <= pr + 0.01 ? V.alpha(col, 0.9) : V.alpha(C.muted, 0.5)); }
          });
          T.stations.forEach((s, k) => V.text(g, s, SX(k), y0 + 109, { size: narrow ? 10.5 : 11, align: 'center', color: Math.round(pr) === k ? C.ink : C.muted, weight: Math.round(pr) === k ? 700 : 400, maxW: (xr - xl) / 6 + 4 }));
          // barreira na estação que pega a divergência
          if (caught) {
            const bx = SX(E.at) + (E.kind === 'fix' ? 0 : 24), ca = Math.max(0, 1 - (el - E.at * STEP) / 1.2);
            if (ca > 0) V.glow(g, bx, (yl[0] + yl[1]) / 2, 40, kc, ca);
            V.box(g, bx - 3, yl[0] - 16, 6, yl[1] - yl[0] + 32, { r: 3, fill: V.alpha(kc, 0.85) });
          }
          // posição das cápsulas
          const k0 = Math.floor(pr), fr = pr - k0;
          const cx = k0 >= 6 ? SX(6) : V.lerp(SX(k0), SX(k0 + 1), V.ease.inOut(fr));
          const offset = 0;
          const hangNow = E.kind === 'hang' && caught;
          [0, 1].forEach((r) => {
            const dead = E.kind === 'abort' && caught && r === 1;
            const x = cx + offset, y = yl[r] + (hangNow ? Math.sin(t * 9 + r) * 1.5 : 0);
            V.box(g, x - 15, y - 8, 30, 16, { r: 8, fill: dead ? V.alpha(C.muted, 0.6) : col });
            V.text(g, dead ? '✕' : 'R' + r, x, y + 0.5, { size: 10.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: dead ? C.ink : onFill(C) });
            if (dead) V.text(g, T.dead, x - (narrow ? 17 : 20), y + (narrow ? -11 : 4), { size: 10.5, align: 'right', color: C.c, weight: 600 }); // no mobile, acima da linha: ao lado colidiria com a estação anterior
            const hasG = E.who === 2 || E.who === r;
            if (hasG && !dead) {
              const fixed = E.kind === 'fix' && caught;
              const gx = x + 14, gy = y - 9, gc = fixed ? C.e : C.red;
              g.save(); g.translate(gx, gy); g.rotate(Math.PI / 4); V.box(g, -5.5, -5.5, 11, 11, { r: 2.5, fill: gc }); g.restore();
              V.text(g, fixed ? '✓' : '≠', gx, gy + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: onFill(C) });
            }
          });
          // desfechos
          if (hangNow) {
            const hx = SX(5) + (narrow ? 0 : 30), hy = (yl[0] + yl[1]) / 2;
            V.line(g, cx, yl[0] + 8, cx, yl[1] - 8, { color: C.red, lw: 1.6, dash: [3, 3], dashOffset: -t * 24 });
            if (!narrow) {
              g.beginPath(); g.arc(hx + 14, hy, 9, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.strokeStyle = C.red; g.lineWidth = 2; g.stroke();
              V.line(g, hx + 14, hy, hx + 14 + Math.cos(t * 3) * 6.5, hy + Math.sin(t * 3) * 6.5, { color: C.red, lw: 2 });
              V.text(g, T.hang, hx + 28, hy + 4, { size: 11, color: C.red, weight: 600 });
            } else {
              const kx = V.lerp(SX(5), SX(6), 0.8);
              g.beginPath(); g.arc(kx, hy, 8, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill(); g.strokeStyle = C.red; g.lineWidth = 2; g.stroke();
              V.line(g, kx, hy, kx + Math.cos(t * 3) * 5.5, hy + Math.sin(t * 3) * 5.5, { color: C.red, lw: 2 });
            }
          }
          if (E.kind === 'refuse' && E.at === 0) {
            const ox = SX(6);
            [0, 1].forEach((r) => { V.glow(g, ox, yl[r], 14, C.e, 0.5 + 0.25 * Math.sin(t * 3)); V.dot(g, ox, yl[r], 5, C.e); });
            V.text(g, T.oldPair, ox + 6, yl[0] - 14, { size: 10.5, align: 'right', color: C.e, weight: 600 });
          }
          if (E.kind === 'fix' && pr >= 6) {
            [0, 1].forEach((r) => V.glow(g, SX(6), yl[r], 22, C.e, 0.45 + 0.2 * Math.sin(t * 3)));
            V.text(g, T.serving, SX(6) + 6, yl[0] - 14, { size: 10.5, align: 'right', color: C.e, weight: 600 });
          }
          para(g, E.txt, pad + 6, y0 + 132, w - 2 * pad - 16, { size: 12, color: C.ink, lh: 16 });
        });
      });
    })();
  },
});

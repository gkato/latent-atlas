LA.chapter({
  id: 'receita-deepseek',
  part: 9,
  order: 44,
  title: 'Reconstruindo a receita DeepSeek-V4-Flash DSpark',
  navTitle: 'Receita DeepSeek',
  subtitle: 'Três lanes, uma esteira de hotfixes que aborta ao menor desvio, 1M de contexto em 2 × 128 GB e um rascunhador em bloco — decisão por decisão, com os números que justificam cada uma.',
  minutes: 60,
  recipes: ['ds'],
  tags: ['DeepSeek-V4', 'DSpark', 'hotfixes', 'fail-closed', 'source-exact', 'nvfp4_ds_mla', 'TP=2', '1M de contexto', 'lanes', 'CUDA graphs'],
  goals: [
    'Distinguir as três <b>lanes de runtime</b> (Anemll, Stage-C, official-main) dos <b>perfis de serviço</b> (1M/6, TP=3, 200K/16) e saber quais knobs existem em cada uma',
    'Refazer o <b>orçamento</b> de memória e de banda que força TP=2, 1M como teto e especulação',
    'Explicar a <b>cadeia de hotfixes fail-closed</b> do entrypoint: o que cada grupo corrige e por que mudar um gate exige recriar os containers',
    'Justificar cada flag do <code>vllm serve</code> (de <code>nvfp4_ds_mla</code> a k=6 e à captura de 48) e <b>prever o decode</b> a partir de bytes e aceitação',
  ],
  prereqs: ['engenharia-de-receita'],
  summary: [
    'A palavra <b>lane</b> tem dois sentidos: lane de runtime (A · Anemll padrão, B · Stage-C overlay, C · official-main) e perfil de serviço (1M/6, TP=3 com 16 slots, 200K/16). Knobs <code>VLLM_DSPARK_*</code> só existem na lane B; na Anemll são no-op.',
    'O orçamento manda: 79,17 GiB de pesos por rank → TP=2; utilização 0,835 → 17,04 GiB = 2,33 M tokens de KV; <code>--max-model-len</code> é teto, não reserva; decode c=1 é ~80 % leitura de pesos (≈16,6 GB por rank por passo).',
    'A imagem é pinada por digest porque todos os hotfixes são <b>source-exact</b>: pin SHA-256 stock → patched, pós-imagem compilada em memória, <code>os.replace</code> atômico, reverificação. Qualquer desvio vira <code>exit 1</code> antes do <code>exec vllm serve</code>.',
    'Gates <code>DSPARK_ENABLE_*</code> só ligam com o valor exato <code>1</code>, e mudá-los exige <code>./stop</code> + <code>./start</code>: o <code>restart</code> reaproveita a camada gravável já patcheada. O launcher sobe o worker antes do head (TCPStore) e sai com 3 se o par já está no ar.',
    '<code>nvfp4_ds_mla</code> é o layout FP8 de 584 B (idêntico ao <code>fp8_ds_mla</code>); o #22 só corrige o despacho (≈1 → 17 tok/s a 600K+). <code>--block-size 256</code> iguala as páginas híbridas em 37.376 B.',
    'k=6 vem do bloco treinado de 5 + a regra de divisibilidade com <code>n_predict=3</code>; a captura \\(8\\lceil N_{seqs}(k+1)/8\\rceil = 48\\) mantém o passo c=6 em CUDA graph (+12 %). Tokens por passo = 1 + aceitos: a aceitação é a maior alavanca de decode que resta.',
    'Estudo de caso, <b>DeepSeek-V4.1-Flash</b> em cinco implantações: 2 Sparks exigem EXL3 a 2,9 bpw (196 GiB); 3 Sparks exigem Engram no NVMe e padding de cabeças 64 → 96; 4 Sparks rodam os ~510 GB nativos com shards exatos de TP=4, por switch ou anel; o DeepInfer (SM100) separa prefill (4 GPUs) e decode (64 GPUs) com atenção DP + EP. O que muda entre elas são bytes por nó, shards, enlaces e motor, nunca o modelo.',
  ],
  quiz: [
    { q: 'Você mudou <code>DSPARK_ENABLE_DSPARK_BLOCK_K</code> de 0 para 1 no <code>.env</code> e rodou <code>docker compose restart</code>. O que acontece?', o: ['O block-k passa a valer normalmente no próximo boot', 'Nada garantido: o restart reaproveita a camada gravável já patcheada; é preciso recriar os containers com <code>./stop</code> + <code>./start</code>', 'O container baixa a imagem de novo e reaplica tudo do zero', 'O vLLM lê o gate em runtime e se reconfigura sem reiniciar'], a: 1, e: 'Os hotfixes editam o vLLM <i>dentro</i> do container, na camada gravável. Um <code>restart</code> reusa essa camada: bytes antigos continuam lá e o patcher pode encontrar um arquivo que não bate com nenhum pin. Só recriar garante que a esteira rode sobre os bytes stock da imagem.' },
    { q: 'Com <code>MAX_NUM_SEQS=6</code> e <code>MTP_NUM_TOKENS=6</code>, por que o compose passa <code>--max-cudagraph-capture-size 48</code> e não 42?', o: ['Porque 48 é o máximo suportado pelo GB10', 'Para reservar espaço para o token bônus de cada requisição', 'Porque o valor cru 42 era truncado para 40 e os tamanhos capturados (múltiplos de 7) paravam em 35, deixando o passo c=6 em modo eager', 'Para reservar memória extra de KV cache'], a: 2, e: 'O passo com 6 slots tem 6 × (6+1) = 42 linhas. Sem arredondar para múltiplo de 8, a captura efetiva ficava em 40, e os tamanhos capturados (7, 14, …, 35) não incluíam 42: aquele passo rodava sem CUDA graph. O A/B mediu 122 vs 139 tok/s agregados (−12 %).' },
    { q: 'O que o KV <code>nvfp4_ds_mla</code> guarda nesta receita?', o: ['K e V em FP4 E2M1 com escala por bloco de 16 (≈4,5 bits)', 'O mesmo layout FP8 de 584 B por linha do <code>fp8_ds_mla</code>; o nome só muda o despacho de kernel', 'Um layout de 416 B por linha herdado do GLM-5.2', 'Só a parte RoPE, em bf16'], a: 1, e: 'São 448 B de NoPE em FP8 e4m3 (escala UE8M0 por 64 dims) + 128 B de RoPE bf16 + 8 B de escalas, byte-idêntico ao <code>fp8_ds_mla</code>. O layout de 416 B foi o probe do Stage B, abandonado. O hotfix #22 existe porque a comparação <code>== "fp8_ds_mla"</code> mandava o nome “nvfp4” para o kernel bf16 lento.' },
    { q: 'Sem o hotfix block-k, por que k precisa ser 6 com o checkpoint Vision-Exp?', o: ['Porque o bloco treinado do DSpark tem 6 posições', 'Porque a aceitação é máxima em k=6', 'Porque k ≥ 5 (bloco treinado) e, com <code>n_predict=3</code>, todo k &gt; 3 precisa ser múltiplo de 3: o menor valor válido é 6', 'Porque o CUDA graph só captura múltiplos de 6'], a: 2, e: 'O bloco treinado tem 5 posições (k &lt; 5 trunca). A lane A trata o DSpark como módulo MTP com <code>n_predict = num_nextn_predict_layers = 3</code> e exige divisibilidade. 6 é o menor k que satisfaz as duas regras: uma query de ruído além do treino. O block-k isenta o DSpark da regra e permite k=5.' },
    { q: 'Depois de um reboot, o start sai com código 3. O que isso significa?', o: ['Configuração inválida no <code>.env</code>', 'Falha de SSH para o worker', 'Os ranks já estão rodando (restaurados pelo <code>restart: unless-stopped</code>); supervisores devem tratar 3 como sucesso', 'Um patch falhou no preflight'], a: 2, e: 'O contrato é 0 ok, 1 falha, 2 config inválida, 3 já rodando. Sair com 1 fazia o systemd entrar em tempestade de retries. O README manda configurar <code>SuccessExitStatus=3</code>, e uma sonda remota com sentinela 42 separa “rank existe” de “SSH falhou” (255).' },
    { q: 'Um patcher encontra, no arquivo-alvo, um SHA-256 que não bate nem com o pin stock nem com o patched. O que ele faz?', o: ['Aplica mesmo assim, com um aviso no log', 'Tenta aplicar por uma âncora aproximada', 'Recusa (FAIL-CLOSED) e sai com 1; com o <code>|| exit 1</code> da cadeia, o rank nunca chega ao <code>exec vllm serve</code>', 'Baixa o arquivo original do upstream e aplica sobre ele'], a: 2, e: 'Fonte desconhecida significa que o patch pode estar errado ali. Servir com meia correção é pior que não servir, então a cadeia inteira aborta. Para os opt-ins sensíveis, o preflight <code>--check</code> do launcher pega isso antes de qualquer rank subir.' },
    { q: 'No estudo de caso da DeepSeek-V4.1-Flash, a receita SGLang de 3 Sparks faz padding das cabeças de atenção 64 → 96, e a de 4 Sparks faz decode mais rápido em toda concorrência apesar de um salto a mais por coletiva. Por quê?', o: ['Porque a receita de 4 Sparks quantiza os experts a 2,9 bits por peso', 'Porque em TP=4 as 64 cabeças, os 8 grupos, o vocabulário e os experts de rascunho dividem exatamente (16 cabeças e 2 grupos por rank): nenhum rank calcula padding, os GEMMs por rank encolhem e ~77 em vez de ~101 GiB de pesos por rank deixam memória sobrando', 'Porque o NCCL não é usado em TP=4', 'Porque o TP=4 desliga o DSpark, que deixava o TP=3 mais lento'], a: 1, e: 'Em TP=3, o mínimo de 32 cabeças locais dos kernels sparse-MLA de SM120 força 96 cabeças, e o shard de atenção do rank 2 é todo zeros. Em TP=4 os shards são exatos. A leitura do repositório: GEMMs de atenção menores por rank e nenhum shard de padding mais que pagam o salto extra. As duas receitas servem os mesmos pesos nativos MXFP4/FP8 com DSpark k=5.' },
  ],

  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ===================== utilitários locais ===================== */
    const FC = {};
    const font = (s, f = 'sans', wt = 400) => FC[s + f + wt] || (FC[s + f + wt] = V.font(s, f, wt));
    const WM = new Map();
    function lines(g, s, maxW, size, f, wt) {
      const key = s + '|' + Math.round(maxW) + '|' + size + (f || '') + (wt || '');
      let L = WM.get(key);
      if (L) return L;
      g.font = font(size, f || 'sans', wt || 400);
      const words = String(s).split(' ');
      L = []; let cur = '';
      for (const wd of words) {
        const tt = cur ? cur + ' ' + wd : wd;
        if (cur && g.measureText(tt).width > maxW) { L.push(cur); cur = wd; } else cur = tt;
      }
      if (cur) L.push(cur);
      if (WM.size > 900) WM.clear();
      WM.set(key, L);
      return L;
    }
    function wtext(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.32);
      let L = lines(g, s, maxW, size, o.font, o.weight);
      if (o.maxLines && L.length > o.maxLines) { L = L.slice(0, o.maxLines); L[L.length - 1] += '…'; }
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, o));
      return L.length * lh;
    }
    function tw(g, s, size, f, wt) { g.font = font(size, f || 'sans', wt || 400); return g.measureText(s).width; }
    function chip(g, x, y, s, col, o = {}) {
      const size = o.size || 11.5;
      let pw = tw(g, s, size, o.font, o.weight || 500) + 16;
      if (o.maxW && pw > o.maxW) pw = o.maxW;
      const ph = size + 9;
      const x0 = o.align === 'center' ? x - pw / 2 : o.align === 'right' ? x - pw : x;
      V.box(g, x0, y - ph / 2, pw, ph, { r: ph / 2, fill: V.alpha(col, o.fa == null ? 0.16 : o.fa), stroke: V.alpha(col, o.sa == null ? 0.75 : o.sa), lw: 1.2 });
      V.text(g, s, x0 + pw / 2, y + 0.5, { size, font: o.font || 'sans', weight: o.weight || 500, align: 'center', base: 'middle', color: o.color || V.C().ink, maxW: pw - 12 });
      return pw;
    }
    function autoH(fig, dh, mh, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      if (!st) return;
      const set = () => { const w = fig.clientWidth || window.innerWidth; const want = (w < bp ? mh : dh) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      window.addEventListener('resize', set);
      LA.onCleanup(() => window.removeEventListener('resize', set));
    }
    const onFill = (C) => (C.dark ? '#06130c' : '#ffffff');
    const br = (x, d = 1) => (+x).toFixed(d).replace('.', ',');
    const fmtN = (n) => Math.round(n).toLocaleString('pt-BR');
    const fmtTok = (n) => (n >= 1e6 ? br(n / 1e6, n % 1e6 === 0 ? 0 : 2) + 'M' : Math.round(n / 1000) + 'K');
    const col = (C, k) => (k === 'muted' ? C.muted : C[k] || C.a);

    /* ============ F1 · três trilhos: lanes de runtime ============ */
    (function () {
      const fig = root.querySelector('#fig-lanes');
      if (!fig) return;
      autoH(fig, 330, 400);
      const LANES = [
        { t: 'Lane A · Anemll (padrão)', c: 'a', st: ['Anemll 0.1.1 @sha256', 'vLLM 752a3a504 · runner V2', 'hotfixes a cada boot', 'vllm serve'], kv: [
          ['imagem', 'ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf1…'],
          ['vLLM', '0.25.2.dev0+g752a3a504 · Model Runner V2 · FlashInfer 0.6.15 · b12x 0.15.3 · xgrammar 0.2.3'],
          ['como muda', 'hotfixes a cada boot, editando o vLLM do container (source-exact, fail-closed)'],
          ['DSpark', 'dentro da imagem; no repositório, só fixtures e trechos citados pelos patchers'],
          ['VLLM_DSPARK_*', 'não registradas: só um aviso, nenhum efeito'],
          ['perfil 1M/6', 'produção (./start-deepseek-v4-flash-dspark.sh): TP=2, teto 1.048.576, 6 slots; ~62–83 tok/s em c=1 e ~160–190 agregados em c=6 com prompts curtos (melhor visto, 14/ago); prefills grandes serializados, com piso de ~8 tok/s de decode']] },
        { t: 'Lane B · Stage-C overlay', c: 'b', st: ['base unholy-fusion', 'COPY de 17 arquivos', 'Stage A → B → C', 'imagem Stage-C'], kv: [
          ['imagem', 'vllm-dspark-runtime:dspark-nvfp4-stage-c'],
          ['build', 'build-dspark-vllm-runtime.sh: unholy-fusion-prod-ready → mia-raf-pr1 → Stage A/B/C → dspark-nvfp4-stage-c, no head; WORKER_BUILD=1 também no worker; --tag-selftest'],
          ['vLLM', 'o do unholy-fusion, em /opt/env · Model Runner V1'],
          ['como muda', 'arquivos inteiros por COPY (overlay) + três Dockerfiles de string-replace (Stage A, B e C)'],
          ['DSpark', 'código completo no repositório: é a implementação de referência legível'],
          ['VLLM_DSPARK_*', 'registradas no envs.py do overlay: aqui são kill-switches reais'],
          ['perfil 200K/16', 'histórico/opcional: 315 tok/s agregados (estático, C16) e 205 (escalonado). Custa construir a imagem, limitar o contexto a 200K e mesclar o override (seção 7)']] },
        { t: 'Lane C · official-main', c: 'c', st: ['official-main 00eb7ce', 'COPY de arquivos Python', 'teste de sanidade', 'experimento'], kv: [
          ['imagem', 'vllm-dspark-runtime:official-main-dspark-00eb7ce + recipe/official-main/Dockerfile.python-patch'],
          ['vLLM', 'main oficial, com DSpark nativo'],
          ['como muda', 'COPY de arquivos Python + teste de sanidade: nvfp4_ds_mla → forma (3, 256, 584) e backend MoE B12xExperts'],
          ['objetivo', 'portar o MoE B12X MXFP4 e o dtype nvfp4_ds_mla para o vLLM oficial'],
          ['estado', 'experimento']] },
      ];
      const CK = [
        { t: 'preview DSpark', b: 'MTP-5', d: 'FP8 · DSpark γ=5 · Markov rank 256 · KV fp8 · 262.144 de contexto' },
        { t: '0731', b: 'k=5', d: 'só texto · n_predict=1 · padrão até ~31/ago/2026' },
        { t: 'Vision-Exp', b: 'k=6', d: 'n_predict=3 · bloco 5 · 1M · nvfp4_ds_mla · visão (atual)' },
      ];
      const CKKV = [
        ['preview', 'DeepSeek-V4-Flash-DSpark · FP8 · DSpark γ=5 · Markov rank 256 · KV fp8 · max-model-len 262144 · MTP-5 · imagens vllm-dspark-runtime:clean. KV fp8 e 262K antes do nvfp4_ds_mla e do 1M; o preview Anemll com MTP-3 mediu C1 66,6 e C6 153,7 tok/s agregados'],
        ['0731', 'deepseek-ai/DeepSeek-V4-Flash-0731 @ 9e165c30… · só texto, FP8, MTP-5 · num_nextn_predict_layers=1, então k=5 bootava · padrão até ~31/ago/2026 · varredura: 191,2 tok/s agregados em 256×c6; teste de contexto a 900K com sentinela correta (TTFT 1.028,85 s)'],
        ['Vision-Exp', 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp @ 86f746b3… · num_nextn_predict_layers=3 e dspark_block_size=5, então k=6 obrigatório · ~157 GiB · visão'],
      ];
      const S = { v: 0, t0: 0 };
      const dl = LA.h('dl', { class: 'la-kv', style: { margin: '0', padding: '14px 18px 8px', overflowWrap: 'anywhere', borderTop: '1px solid var(--line)' } });
      fig.insertBefore(dl, fig.querySelector('.la-fig-controls'));
      function fill() {
        dl.innerHTML = '';
        (S.v < 3 ? LANES[S.v].kv : CKKV).forEach(([k, v]) => { dl.appendChild(LA.h('dt', null, k)); dl.appendChild(LA.h('dd', null, v)); });
      }
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vista', options: [{ v: 0, label: 'A · Anemll' }, { v: 1, label: 'B · Stage-C' }, { v: 2, label: 'C · official-main' }, { v: 3, label: 'linhagem dos checkpoints' }], value: 0, onChange: (v) => { S.v = v; S.t0 = H ? H.t : 0; fill(); } });
      fill();
      function train(g, x, y, cc, C, t) {
        const bw = 44, bh = 20, bob = Math.sin(t * 9) * 0.6;
        V.glow(g, x + bw / 2 + 4, y + bob, 16, cc, 0.7);
        V.box(g, x - bw / 2, y - bh / 2 + bob, bw, bh, { r: 8, fill: cc, stroke: V.alpha(C.ink, 0.35), lw: 1 });
        for (let i = 0; i < 3; i++) V.box(g, x - bw / 2 + 6 + i * 11, y - 6 + bob, 8, 7, { r: 2, fill: V.alpha(C.dark ? '#000' : '#fff', 0.75) });
        V.dot(g, x + bw / 2 - 4, y + 3 + bob, 2.2, C.dark ? '#fff' : '#222');
      }
      function trainPos(t, segs, speed = 1.7, dwell = 0.4) {
        const p = (t / speed) % (segs + 1.3);
        if (p >= segs) return segs;
        const s = Math.floor(p), f = p - s;
        return s + V.ease.inOut(V.clamp((f - dwell) / (1 - dwell), 0, 1));
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const tt = t - S.t0;
        if (S.v < 3) {
          const top = 6, bh = (h - top - 4) / 3;
          LANES.forEach((L, li) => {
            const on = li === S.v, cc = col(C, L.c);
            const y0 = top + li * bh, ly = y0 + 38;
            const x0 = 30, x1 = w - 30;
            V.text(g, L.t, 14, y0 + 17, { size: 12.5, weight: on ? 700 : 500, color: on ? cc : V.alpha(C.ink, 0.75) });
            g.globalAlpha = on ? 1 : 0.55;
            V.line(g, x0, ly, x1, ly, { color: cc, lw: on ? 6 : 4 });
            if (on) V.line(g, x0, ly, x1, ly, { color: V.alpha(C.bg || '#000', 0.55), lw: 1.5, dash: [5, 9], dashOffset: -tt * 26 });
            const ns = L.st.length, sx = (i) => x0 + ((x1 - x0) * i) / (ns - 1);
            const maxW = Math.max(62, (x1 - x0) / (ns - 1) - 8);
            for (let i = 0; i < ns; i++) {
              const x = sx(i);
              V.dot(g, x, ly, on ? 9 : 7, C.panel || C.bg);
              g.lineWidth = 3; g.strokeStyle = cc; g.beginPath(); g.arc(x, ly, on ? 9 : 7, 0, Math.PI * 2); g.stroke();
              const align = i === 0 ? 'left' : i === ns - 1 ? 'right' : 'center';
              const tx = i === 0 ? x - 10 : i === ns - 1 ? x + 10 : x;
              g.globalAlpha = 1; // rótulos são informação: legíveis também nas lanes inativas
              wtext(g, L.st[i], tx, ly + 25, maxW, { size: 11, color: on ? C.ink : V.alpha(C.ink, 0.62), align, maxLines: 3, weight: on ? 500 : 400 });
              g.globalAlpha = on ? 1 : 0.55;
            }
            g.globalAlpha = 1;
            if (on) {
              const pos = trainPos(tt, ns - 1);
              const near = Math.round(pos);
              if (Math.abs(pos - near) < 0.02) V.glow(g, sx(near), ly, 24, cc, 0.9);
              train(g, V.lerp(x0, x1, pos / (ns - 1)), ly - 1, cc, C, tt);
            }
          });
        } else {
          const cc = C.ds || C.a;
          const ly = Math.round(h * 0.44), x0 = 40, x1 = w - 40;
          V.text(g, 'LINHAGEM DOS CHECKPOINTS · CADA TROCA MUDOU UM NÚMERO DA RECEITA', 14, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 28 });
          V.line(g, x0, ly, x1, ly, { color: cc, lw: 6 });
          V.line(g, x0, ly, x1, ly, { color: V.alpha(C.bg || '#000', 0.55), lw: 1.5, dash: [5, 9], dashOffset: -tt * 26 });
          const ns = CK.length, sx = (i) => x0 + ((x1 - x0) * i) / (ns - 1);
          const maxW = Math.max(90, ((x1 - x0) / (ns - 1)) * 0.6);
          const pos = trainPos(tt, ns - 1, 2.2, 0.5);
          CK.forEach((k, i) => {
            const x = sx(i), act = Math.abs(pos - i) < 0.5;
            const align = i === 0 ? 'left' : i === ns - 1 ? 'right' : 'center';
            const tx = i === 0 ? x - 12 : i === ns - 1 ? x + 12 : x;
            V.text(g, k.t, tx, ly - 44, { size: 13.5, weight: 700, color: act ? cc : C.ink, align });
            chip(g, tx + (align === 'left' ? 0 : 0), ly - 24, k.b, act ? C.c : C.muted, { size: 11.5, font: 'mono', align, weight: 600 });
            V.dot(g, x, ly, 10, C.panel || C.bg);
            g.lineWidth = 3.5; g.strokeStyle = cc; g.beginPath(); g.arc(x, ly, 10, 0, Math.PI * 2); g.stroke();
            wtext(g, k.d, tx, ly + 30, maxW, { size: 11.5, color: act ? C.ink : C.muted, align, maxLines: 6, lh: 15 });
          });
          const near = Math.round(pos);
          if (Math.abs(pos - near) < 0.02) V.glow(g, sx(near), ly, 26, cc, 0.9);
          train(g, V.lerp(x0, x1, pos / (ns - 1)), ly - 1, cc, C, tt);
        }
      });
    })();

    /* ============ F2 · balança de memória ============ */
    (function () {
      const fig = root.querySelector('#fig-memoria');
      if (!fig) return;
      autoH(fig, 400, 460);
      const M = 121.7, W = 79.17, KV0 = 17.04, N0 = 2331430, U0 = 0.835, A = U0 * M - W - KV0;
      const CTX = [8192, 32768, 50000, 131072, 200000, 262144, 500000, 1048576];
      const S = { u: 0.835, n: 6, ci: 4 };
      let H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'utilização u', min: 0.78, max: 0.87, step: 0.005, value: S.u, fmt: (v) => br(v, 3), onInput: (v) => { S.u = v; upd(); } });
      UI.slider(ctl, { label: 'slots', min: 1, max: 16, step: 1, value: S.n, fmt: (v) => String(v), onInput: (v) => { S.n = v; upd(); } });
      UI.slider(ctl, { label: 'contexto/req', min: 0, max: CTX.length - 1, step: 1, value: S.ci, fmt: (v) => fmtTok(CTX[v]), onInput: (v) => { S.ci = v; upd(); } });
      const r1 = UI.readout(ctl, 'KV cache', '—'), r2 = UI.readout(ctl, 'tokens no pool', '—');
      function upd() {
        const kv = KV0 + (S.u - U0) * M, tok = (N0 * kv) / KV0;
        r1.set(br(kv, 2) + ' GiB'); r2.set(fmtN(tok));
        if (H) H.redraw();
      }
      function person(g, x, y, r, c, C, t, i) {
        y += Math.sin(t * 4 + i * 0.9) * 1.6;
        V.dot(g, x, y, r, V.alpha(c, 0.85));
        V.dot(g, x - r * 0.35, y - r * 0.15, r * 0.16, onFill(C));
        V.dot(g, x + r * 0.35, y - r * 0.15, r * 0.16, onFill(C));
        g.strokeStyle = onFill(C); g.lineWidth = 1.2; g.beginPath(); g.arc(x, y + r * 0.1, r * 0.4, 0.2 * Math.PI, 0.8 * Math.PI); g.stroke();
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 18, bw = w - 2 * pad, narrow = w < 560;
        const kv = KV0 + (S.u - U0) * M, tok = (N0 * kv) / KV0;
        V.text(g, 'UM RANK · MemTotal ≈ 121,7 GiB', pad, 18, { size: 11, font: 'mono', color: C.muted });
        const by = 28, bh = 40;
        const segs = [
          { v: (1 - S.u) * M, c: C.muted, a: 0.28, l: 'fora do vLLM', s: br((1 - S.u) * M, 1) + ' GiB', light: true },
          { v: W, c: C.b, a: 0.88, l: 'pesos', s: '79,17 GiB' },
          { v: A, c: C.d, a: 0.88, l: 'ativações · graphs', s: '≈' + br(A, 1) + ' GiB' },
          { v: kv, c: C.a, a: 0.92, l: 'KV cache', s: br(kv, 2) + ' GiB' },
        ];
        let x = pad; const xs = [];
        segs.forEach((sg) => {
          const sw = (sg.v / M) * bw;
          xs.push([x, sw]);
          V.box(g, x + 1, by, Math.max(2, sw - 2), bh, { r: 7, fill: V.alpha(sg.c, sg.a) });
          if (sw > 92) {
            V.text(g, sg.l, x + 9, by + 16, { size: 11.5, weight: 600, color: sg.light ? C.ink : onFill(C), maxW: sw - 14 });
            V.text(g, sg.s, x + 9, by + 31, { size: 11, font: 'mono', color: sg.light ? C.muted : onFill(C), maxW: sw - 14 });
          }
          x += sw;
        });
        // brilho que percorre o KV (é o que “sobra”)
        const kx0 = xs[3][0], kw = xs[3][1];
        const sh = kx0 + ((t * 0.35) % 1) * kw;
        V.glow(g, sh, by + bh / 2, 18, C.ink, 0.25);
        // colchete do vLLM
        const vx0 = xs[1][0], vx1 = pad + bw, yb = by + bh + 8;
        V.line(g, vx0 + 1, yb, vx1 - 1, yb, { color: C.ink, lw: 1.2 });
        V.line(g, vx0 + 1, yb - 5, vx0 + 1, yb, { color: C.ink, lw: 1.2 });
        V.line(g, vx1 - 1, yb - 5, vx1 - 1, yb, { color: C.ink, lw: 1.2 });
        V.text(g, 'u × MemTotal = ' + br(S.u * M, 1) + ' GiB para o vLLM', (vx0 + vx1) / 2, yb + 15, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: vx1 - vx0 });
        // legenda
        const ly = yb + 36, per = narrow ? 2 : 4, itemW = bw / per;
        segs.forEach((sg, i) => {
          const cx = pad + (i % per) * itemW, cy = ly + Math.floor(i / per) * 18;
          V.box(g, cx, cy - 9, 11, 11, { r: 3, fill: V.alpha(sg.c, sg.light ? 0.5 : sg.a) });
          V.text(g, sg.l + ': ' + sg.s, cx + 16, cy, { size: 11, color: C.ink, maxW: itemW - 20 });
        });
        // pool de KV
        const py = ly + (narrow ? 44 : 30);
        V.text(g, 'POOL DE KV · ' + fmtN(tok) + ' tokens (≈' + br(tok / 1048576, 2) + '× de 1M)', pad, py, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const ty = py + 10, th = 36;
        V.box(g, pad, ty, bw, th, { r: 10, fill: V.alpha(C.a, 0.07), stroke: V.alpha(C.a, 0.6), lw: 1.5 });
        const ctx = CTX[S.ci], n = S.n;
        const fits = Math.floor(tok / ctx), nin = Math.min(n, fits), nq = n - nin;
        const pw = (ctx / tok) * bw;
        if (pw >= 4) {
          for (let i = 0; i < nin; i++) V.box(g, pad + 2 + i * pw, ty + 4, Math.max(1, pw - 3), th - 8, { r: Math.min(7, pw / 3), fill: V.alpha(C.series[i % 6], 0.7 + 0.15 * Math.sin(t * 2 + i)) });
        } else if (nin > 0) {
          V.box(g, pad + 2, ty + 4, Math.max(2, nin * pw), th - 8, { r: 4, fill: V.alpha(C.a, 0.8) });
        }
        for (let m = 1; m * 1048576 < tok; m++) {
          const mx = pad + ((m * 1048576) / tok) * bw;
          V.line(g, mx, ty - 3, mx, ty + th + 3, { color: V.alpha(C.ink, 0.5), lw: 1, dash: [3, 3] });
          V.text(g, m + 'M', mx, ty + th + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // vivos e fila
        const qy = ty + th + 34;
        const pct = Math.min(100, ((nin * ctx) / tok) * 100);
        V.text(g, 'VIVOS · ' + nin + ' × ' + fmtTok(ctx) + ' = ' + fmtTok(nin * ctx) + ' (' + br(pct, 0) + ' % do pool)', pad, qy, { size: 11, font: 'mono', color: C.ink, maxW: bw });
        const qy2 = qy + 18;
        V.text(g, nq > 0 ? 'NA FILA · ' + nq + ' esperando o pool liberar' : 'NA FILA · ninguém', pad, qy2, { size: 11, font: 'mono', color: nq > 0 ? C.c : C.muted, maxW: bw });
        const maxP = Math.min(nq, Math.floor((bw - 30) / 24));
        for (let i = 0; i < maxP; i++) person(g, pad + 12 + i * 24, qy2 + 20, 9, C.c, C, t, i);
        if (nq > maxP) V.text(g, '+' + (nq - maxP), pad + 12 + maxP * 24, qy2 + 24, { size: 11, color: C.c });
        // veredito
        const vy = qy2 + (nq > 0 ? 50 : 24);
        const ok = n * ctx <= tok;
        const vh = wtext(g, (ok ? '✓ cabe: ' : '✕ não cabe inteiro: ') + n + ' × ' + fmtTok(ctx) + ' = ' + fmtTok(n * ctx) + ' tokens' + (ok ? ' ≤ pool.' : ' > pool. Só ' + nin + ' cabem ao mesmo tempo: max_model_len é teto, não reserva.'), pad, vy, bw, { size: 12, weight: 600, color: ok ? C.a : C.c, maxLines: 3 });
        // linhas de boot equivalentes
        const logs = [
          'Available KV cache memory: ' + kv.toFixed(2) + ' GiB',
          'GPU KV cache size: ' + Math.round(tok).toLocaleString('en-US') + ' tokens',
          'Maximum concurrency for 1,048,576 tokens per request: ' + (tok / 1048576).toFixed(2) + 'x',
        ];
        const L2 = narrow ? logs : [logs[0] + ' · ' + logs[1], logs[2]];
        const lby = vy + vh - 4, lbh = L2.length * 16 + 12;
        if (lby + lbh < h - 34) {
          V.box(g, pad, lby, bw, lbh, { r: 8, fill: V.alpha(C.ink, C.dark ? 0.05 : 0.04), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          L2.forEach((s, k) => V.text(g, s, pad + 10, lby + 18 + k * 16, { size: 11, font: 'mono', color: C.ink, maxW: bw - 20 }));
        }
        // semáforo
        let sc = C.a, stx = 'padrão da receita (0,835). MemAvailable medido do host: 3,8–5,2 GB a 0,83.';
        if (S.u <= 0.8001) { sc = C.e; stx = 'conservador (≤0,80): mais folga para o host. Se a captura de CUDA graphs der OOM, a receita baixa para ~0,78.'; }
        else if (S.u > 0.8351) { sc = C.d; stx = 'agressivo: o perfil big-prompt usa 0,87 e exige host enxuto (sem GUI, earlyoom desligado).'; }
        const sy = h - 24;
        V.dot(g, pad + 6, sy - 4, 6, sc); V.glow(g, pad + 6, sy - 4, 14, sc, 0.6 + 0.3 * Math.sin(t * 3));
        wtext(g, stx, pad + 20, sy - (narrow ? 8 : 0), bw - 22, { size: 11.5, color: C.ink, maxLines: 2, lh: 14 });
      });
      upd();
    })();

    /* ============ F3 · topologia ============ */
    (function () {
      const fig = root.querySelector('#fig-topologia');
      if (!fig) return;
      autoH(fig, 420, 600);
      const S = { tp: 2, traffic: true };
      const CAT = [
        { n: 'wo_b (atenção)', k: 43, c: 'b' }, { n: 'MoE', k: 43, c: 'a' }, { n: 'drafter', k: 6, c: 'c' },
        { n: 'embeddings', k: 2, c: 'd' }, { n: 'logits', k: 2, c: 'ink' }, { n: 'laço de Markov', k: 12, c: 'f' },
      ];
      const SEQ = [3];
      for (let l = 0; l < 43; l++) SEQ.push(0, 1);
      SEQ.push(4);
      for (let i = 0; i < 6; i++) SEQ.push(2, 5, 5);
      SEQ.push(3, 4);
      const TOT = SEQ.length; // 108
      const colC = (C, k) => (k === 'ink' ? V.alpha(C.ink, 0.75) : col(C, k));
      const ctl = UI.controls(fig);
      const rW = UI.readout(ctl, 'pesos/rank', '79,17 GiB'), rK = UI.readout(ctl, 'KV/rank', '≈17 GiB'), rS = UI.readout(ctl, 'slots', '6');
      UI.seg(ctl, { label: 'Topologia', options: [{ v: 2, label: 'TP=2 (padrão)' }, { v: 3, label: 'TP=3 (start-tp3.sh)' }], value: 2, onChange: (v) => {
        S.tp = v;
        rW.set(v === 2 ? '79,17 GiB' : '60,69 GiB'); rK.set(v === 2 ? '≈17 GiB' : '≈35 GiB'); rS.set(v === 2 ? '6' : '16');
      } });
      UI.toggle(ctl, { label: 'tráfego de um passo de decode', value: true, onChange: (v) => { S.traffic = v; } });
      // grade de grupos de saída (o_groups) × cabeças: o que cada rank guarda
      function groups(g, x, y, w, h, rank, C, t) {
        const G = S.tp === 2 ? 8 : 9, per = S.tp === 2 ? 4 : 3;
        const lo = rank * per, hi = lo + per - 1;
        const lbl = S.tp === 2 ? 'o_groups 8 × 8 cabeças · este rank: grupos ' + lo + '–' + hi : '9 grupos (1 de pad) × 8 · este rank: ' + lo + '–' + hi;
        V.text(g, lbl, x, y + 10, { size: 11, color: C.muted, maxW: w });
        const gw = (w - (G - 1) * 4) / G, gh = Math.min(20, h - 18);
        if (gh < 9) return;
        for (let k = 0; k < G; k++) {
          const gx = x + k * (gw + 4), gy = y + 16;
          const mine = k >= lo && k <= hi, pad = S.tp === 3 && k === 8;
          const cc = pad ? C.muted : mine ? (rank === 0 ? C.a : C.b) : C.muted;
          V.box(g, gx, gy, gw, gh, { r: 4, fill: V.alpha(cc, pad ? 0.08 : mine ? 0.28 : 0.06), stroke: V.alpha(cc, mine || pad ? 0.9 : 0.25), lw: 1 });
          if (pad) { g.save(); V.rr(g, gx, gy, gw, gh, 4); g.clip(); for (let q = -gh; q < gw; q += 5) V.line(g, gx + q, gy + gh, gx + q + gh, gy, { color: V.alpha(C.muted, 0.6), lw: 1 }); g.restore(); continue; }
          const dr = Math.max(1.2, Math.min(2.2, gw / 12));
          for (let d = 0; d < 8; d++) {
            const dx = gx + ((d % 4) + 0.5) * (gw / 4), dy = gy + (Math.floor(d / 4) + 0.5) * (gh / 2);
            const on = mine && Math.sin(t * 3 + d + k * 1.3) > 0.2;
            V.dot(g, dx, dy, dr, V.alpha(mine ? cc : C.muted, on ? 1 : mine ? 0.55 : 0.3));
          }
        }
      }
      function node(g, x, y, w, h, n, C, t) {
        const cc = n.head ? C.a : C.b;
        V.box(g, x, y, w, h, { r: 14, fill: V.alpha(cc, 0.08), stroke: V.alpha(cc, 0.8), lw: 1.6 });
        V.text(g, n.t, x + 12, y + 20, { size: 12.5, weight: 700, color: cc, maxW: w - 20 });
        let yy = y + 38;
        n.l.forEach((s) => { V.text(g, s, x + 12, yy, { size: 11, color: C.ink, maxW: w - 20 }); yy += 15; });
        let cx = x + 10, cy = yy + 8;
        n.chips.forEach((s) => {
          const wd = tw(g, s, 11, 'mono', 500) + 16;
          if (cx + wd > x + w - 8) { cx = x + 10; cy += 22; }
          chip(g, cx, cy, s, n.head ? C.d : C.muted, { size: 11, font: 'mono', maxW: w - 20 });
          cx += wd + 6;
        });
        const gy = cy + 14;
        if (y + h - 8 - gy >= 26) groups(g, x + 12, gy, w - 24, y + h - 8 - gy, n.rank, C, t);
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const legH = narrow ? 80 : 48;
        const area = { x: 12, y: 10, w: w - 24, h: h - legH - (narrow ? 28 : 18) };
        const P = 4.8, f = (t % P) / P, cur = Math.min(TOT - 1, Math.floor(f * TOT * 1.05));
        let nodes = [], links = [];
        if (S.tp === 2) {
          const hd = { head: true, rank: 0, t: 'spark-1 · head · rank 0', l: ['GB10 sm_121 · 128 GB UMA', 'pesos 79,17 GiB · KV ≈17 GiB'], chips: ['API :8888', 'EngineCore', 'TCPStore :25000'] };
          const wk = { rank: 1, t: 'spark-2 · worker · rank 1', l: ['GB10 sm_121 · 128 GB UMA', 'mesma imagem @sha256'], chips: ['--headless'] };
          if (!narrow) {
            const nw = Math.min(280, area.w * 0.35), nh = Math.min(196, area.h - 30);
            const ny = area.y + 24 + (area.h - 24 - nh) / 2;
            nodes = [[area.x, ny, nw, nh, hd], [area.x + area.w - nw, ny, nw, nh, wk]];
            links = [{ a: [area.x + nw, ny + nh * 0.5], b: [area.x + area.w - nw, ny + nh * 0.5], l1: 'QSFP · ConnectX-7 · RoCE v2', l2: '2 PFs PCIe Gen5 ×4 (~126 Gb/s cada) · MTU 9000' }];
          } else {
            const nw = area.w, nh = Math.min(150, (area.h - 64) / 2);
            nodes = [[area.x, area.y, nw, nh, hd], [area.x, area.y + area.h - nh, nw, nh, wk]];
            links = [{ a: [area.x + nw * 0.3, area.y + nh], b: [area.x + nw * 0.3, area.y + area.h - nh], l1: 'QSFP · CX-7 · RoCE v2', l2: '2 PFs · MTU 9000', side: true }];
          }
        } else {
          const mk = (i, head) => ({ head, rank: i, t: 'spark-' + (i + 1) + (head ? ' · head · rank 0' : ' · rank ' + i), l: ['pesos 60,69 GiB · KV ≈35 GiB'], chips: head ? ['API', 'TCPStore'] : ['--headless'] });
          if (!narrow) {
            const nw = Math.min(250, area.w * 0.31), nh = Math.min(128, area.h * 0.44);
            const n0 = [area.x + (area.w - nw) / 2, area.y, nw, nh, mk(0, true)];
            const n1 = [area.x, area.y + area.h - nh, nw, nh, mk(1)];
            const n2 = [area.x + area.w - nw, area.y + area.h - nh, nw, nh, mk(2)];
            nodes = [n0, n1, n2];
            links = [
              { a: [n0[0], n0[1] + nh * 0.6], b: [n1[0] + nw * 0.5, n1[1]], l1: '10.0.22.0/24' },
              { a: [n0[0] + nw, n0[1] + nh * 0.6], b: [n2[0] + nw * 0.5, n2[1]], l1: '10.0.23.0/24' },
              { a: [n1[0] + nw, n1[1] + nh * 0.55], b: [n2[0], n2[1] + nh * 0.55], l1: 'par CX' },
            ];
          } else {
            // o enlace spark-1 ↔ spark-3 contorna o spark-2 por uma calha à direita (não atravessa o nó)
            const gut = 28, nw = area.w - gut, gap = 30, nh = Math.min(118, (area.h - 2 * gap - 34) / 3);
            nodes = [0, 1, 2].map((i) => [area.x, area.y + i * (nh + gap), nw, nh, mk(i, i === 0)]);
            const gx = area.x + nw + gut / 2, ya = nodes[0][1] + nh * 0.5, yb = nodes[2][1] + nh * 0.5;
            links = [
              { a: [area.x + nw * 0.25, nodes[0][1] + nh], b: [area.x + nw * 0.25, nodes[1][1]], l1: '10.0.22.0/24', side: true },
              { a: [area.x + nw * 0.25, nodes[1][1] + nh], b: [area.x + nw * 0.25, nodes[2][1]], l1: 'par CX', side: true },
              { pts: [[area.x + nw, ya], [gx, ya], [gx, yb], [area.x + nw, yb]], l1: '1↔3: 10.0.23.0/24', lab: [gx - 10, nodes[0][1] + nh + gap / 2 + 4] },
            ];
          }
        }
        if (S.tp === 3) {
          if (!narrow) {
            const ly = area.y + area.h * 0.5 + 6;
            V.line(g, area.x + 4, ly, area.x + area.w - 4, ly, { color: V.alpha(C.d, 0.8), lw: 1.4, dash: [4, 5], dashOffset: -t * 12 });
            V.text(g, 'LAN enP7s7 · 192.168.1.0/24 · Gloo, TCPStore, bootstrap', area.x + area.w / 2, ly - 8, { size: 11, align: 'center', color: C.d, font: 'mono', maxW: area.w * 0.36 });
          } else {
            const last = nodes[2], by = last[1] + last[3] + 22;
            V.line(g, area.x + 4, by + 8, area.x + area.w - 4, by + 8, { color: V.alpha(C.d, 0.8), lw: 1.4, dash: [4, 5], dashOffset: -t * 12 });
            V.text(g, 'bootstrap (Gloo/TCPStore) pela LAN enP7s7', area.x + area.w / 2, by, { size: 11, align: 'center', color: C.d, font: 'mono', maxW: area.w });
          }
        }
        const alongP = (pts, u) => {
          let tot = 0; const sg = [];
          for (let k = 1; k < pts.length; k++) { const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); sg.push(d); tot += d; }
          let s = V.clamp(u, 0, 1) * tot;
          for (let k = 0; k < sg.length; k++) {
            if (s <= sg[k] || k === sg.length - 1) { const q = sg[k] ? Math.min(1, s / sg[k]) : 0; return [V.lerp(pts[k][0], pts[k + 1][0], q), V.lerp(pts[k][1], pts[k + 1][1], q)]; }
            s -= sg[k];
          }
          return pts[pts.length - 1];
        };
        const poly = (pts, color, lw) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let k = 1; k < pts.length; k++) g.lineTo(pts[k][0], pts[k][1]); g.strokeStyle = color; g.lineWidth = lw; g.lineJoin = 'round'; g.stroke(); };
        links.forEach((L) => { if (!L.pts) L.pts = [L.a, L.b]; L.a = L.pts[0]; L.b = L.pts[L.pts.length - 1]; });
        links.forEach((L, li) => {
          poly(L.pts, V.alpha(C.ink, 0.22), 7);
          poly(L.pts, C.a, 2.2);
          if (S.traffic) {
            const rev = L.pts.slice().reverse();
            for (let d = 0; d < 7; d++) {
              const i = cur - d;
              if (i < 0) continue;
              const age = (f * TOT * 1.05 - i) / 6;
              if (age < 0 || age > 1) continue;
              const cc = colC(C, CAT[SEQ[i]].c);
              const p1 = alongP(L.pts, age), p2 = alongP(rev, age);
              V.dot(g, p1[0], p1[1], 3.6, cc); V.dot(g, p2[0], p2[1], 3.6, cc);
              if (li === 0 && d === 0) V.glow(g, p1[0], p1[1], 10, cc, 0.6);
            }
          }
        });
        // rótulos depois de todas as linhas e pontos, com fundo opaco: nenhum enlace passa por cima
        links.forEach((L) => {
          const mx = (L.a[0] + L.b[0]) / 2, my = (L.a[1] + L.b[1]) / 2;
          if (L.lab) {
            const lw1 = tw(g, L.l1, 11, 'mono', 400) + 12;
            V.box(g, L.lab[0] - lw1, L.lab[1] - 12, lw1, 17, { r: 6, fill: C.panel || C.bg, stroke: V.alpha(C.a, 0.4), lw: 1 });
            V.text(g, L.l1, L.lab[0] - 6, L.lab[1], { size: 11, font: 'mono', align: 'right', color: C.ink });
          } else if (L.side) {
            V.text(g, L.l1, mx + 10, my - 2, { size: 11, font: 'mono', color: C.ink, maxW: area.w * 0.45 });
            if (L.l2) V.text(g, L.l2, mx + 10, my + 13, { size: 11, color: C.muted, maxW: area.w * 0.6 });
          } else {
            const lw1 = tw(g, L.l1, 11, 'mono', 400) + 12;
            V.box(g, mx - lw1 / 2, my - 22, lw1, 17, { r: 6, fill: C.panel || C.bg, stroke: V.alpha(C.a, 0.4), lw: 1 });
            V.text(g, L.l1, mx, my - 10, { size: 11, font: 'mono', align: 'center', color: C.ink });
            if (L.l2) V.text(g, L.l2, mx, my + 20, { size: 11, align: 'center', color: C.muted, maxW: Math.abs(L.b[0] - L.a[0]) - 6 });
          }
        });
        if (S.tp === 2) {
          const [hx, hy, hw, hh] = nodes[0], [wx, wy] = nodes[1];
          if (!narrow) {
            const y = hy - 2, x1 = hx + hw * 0.7, x2 = wx + 30;
            g.save(); g.setLineDash([3, 5]); g.lineDashOffset = -t * 20; g.strokeStyle = V.alpha(C.d, 0.9); g.lineWidth = 1.4;
            g.beginPath(); g.moveTo(x1, y); g.quadraticCurveTo((x1 + x2) / 2, y - 30, x2, y); g.stroke(); g.restore();
            V.text(g, 'broadcast do scheduler: ZMQ entre nós', (x1 + x2) / 2, y - 34, { size: 11, align: 'center', color: C.d, maxW: x2 - x1 - 8 });
            V.text(g, 'RTT de rede ≈0,33 ms', (x1 + x2) / 2, y - 20, { size: 11, align: 'center', color: C.d, maxW: x2 - x1 - 8 });
          } else {
            const x = hx + hw * 0.78, y1 = hy + hh, y2 = wy;
            V.line(g, x, y1, x, y2, { color: V.alpha(C.d, 0.9), lw: 1.4, dash: [3, 5], dashOffset: -t * 20 });
            V.text(g, 'broadcast', x + 6, (y1 + y2) / 2 + 4, { size: 11, color: C.d });
          }
        }
        nodes.forEach((n) => node(g, n[0], n[1], n[2], n[3], n[4], C, t));
        // barra das coletivas
        const lyb = h - legH;
        const bx = 14, bwd = w - 28;
        const shown = S.traffic ? Math.min(TOT, Math.floor(f * TOT * 1.05) + 1) : TOT;
        V.text(g, (S.traffic ? 'coletivas neste passo: ' + shown + ' / ' + TOT : 'coletivas por passo: ' + TOT) + ' · 30–50 µs cada ≈ 3–5 ms (ordem ilustrativa)', bx, lyb, { size: 11, font: 'mono', color: C.muted, maxW: bwd });
        let xx = bx, acc = 0;
        CAT.forEach((c) => {
          const sw = (c.k / TOT) * bwd;
          V.box(g, xx + 0.5, lyb + 8, Math.max(1, sw - 1), 12, { r: 3, fill: V.alpha(colC(C, c.c), 0.85) });
          xx += sw; acc += c.k;
        });
        if (S.traffic) { const px = bx + (shown / TOT) * bwd; V.line(g, px, lyb + 5, px, lyb + 23, { color: C.ink, lw: 2 }); }
        let lx = bx, lyy = lyb + 36;
        CAT.forEach((c) => {
          const s = c.n + ' ' + c.k, wd = tw(g, s, 11) + 26;
          if (lx + wd > bx + bwd) { lx = bx; lyy += 16; }
          V.box(g, lx, lyy - 9, 10, 10, { r: 2, fill: colC(C, c.c) });
          V.text(g, s, lx + 14, lyy, { size: 11, color: C.ink });
          lx += wd;
        });
      });
    })();

    /* ============ F4 · o launcher (stepper) ============ */
    (function () {
      const fig = root.querySelector('#fig-launcher');
      if (!fig) return;
      autoH(fig, 380, 560);
      const S = { scn: 'ok', nfs: false };
      const STEPS = [
        { t: 'Normalizar o .env e validar os knobs', d: 'O <code>source</code> lê uma cópia privada <code>0600</code> do <code>.env.dspark</code>, sem BOM/CRLF. Knobs numéricos são validados em faixa e normalizados com <code>10#</code>; <code>MTP_NUM_TOKENS</code> precisa ser ≥ 5 e múltiplo de 3 (sem block-k).' },
        { t: 'Derivar e forçar', d: '<code>GPU_MEMORY_UTILIZATION</code> sai de <code>GPU_MEMORY_UTILIZATION_TEXT</code>; <code>TP_SIZE=2</code>/<code>NNODES=2</code> são forçados; o modelo oficial é resolvido com a revisão pinada. API aberta sem chave ou <code>.env</code> legível por outros geram aviso, sem falhar.' },
        { t: 'Pré-requisitos e o contrato do exit 3', d: 'Ferramentas locais, SSH não interativo (<code>BatchMode=yes</code>, <code>ConnectTimeout=10</code>) e a imagem nos dois nós. Se um rank já existe, o start sai com <b>3</b> (“já rodando”); no worker, a sonda remota responde com o sentinela 42.' },
        { t: 'Resolver os GIDs RoCEv2', d: 'Para cada HCA de <code>NCCL_IB_HCA</code>, o launcher confere em sysfs, nos dois nós, uma porta ACTIVE com GID RoCE v2 que case com o IPv4 da interface. Falha fechado e não pina <code>NCCL_IB_GID_INDEX</code>.' },
        { t: 'Sincronizar o worker', d: 'Compose, <code>.env</code> (temporário <code>0600</code> + <code>mv -f</code>), proposer, cada patcher, <code>patches/vision_exp/</code> e <code>patches/tp3/</code> vão para <code>WORKER_DIR</code>. Com NFS, o head exporta o cache de pesos somente leitura.' },
        { t: 'Preflight --check', d: 'Os patches opt-in sensíveis habilitados (e o #117, sempre) rodam em <code>docker compose run --rm --no-deps … --check</code>, <b>primeiro no worker e depois no head</b>, antes de qualquer rank subir.' },
        { t: 'Worker primeiro, depois o head', d: 'O worker sobe como rank 1 <code>--headless</code> e espera o rendezvous; o head sobe como rank 0 e hospeda o TCPStore em <code>MASTER_ADDR:25000</code>. Reiniciar o head no meio do load derrubava esse handshake (#38).' },
        { t: 'Esperar, testar e aquecer', d: 'Até 100 tentativas × 15 s em <code>/v1/models</code> (boot frio ~6–8 min). Depois vêm um smoke e o <i>boot-shape warmup</i> (47 requisições a N=6), que materializa buckets de Triton antes do tráfego real.' },
      ];
      const ctl = UI.controls(fig);
      let st = null;
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'ok', label: 'normal' }, { v: 'run', label: 'par já rodando' }, { v: 'drift', label: 'patch não bate' }], value: 'ok', onChange: (v) => { S.scn = v; if (st) st.go(st.i); } });
      UI.toggle(ctl, { label: 'pesos via NFS (DSPARK_WORKER_HF_NFS=1)', value: false, onChange: (v) => { S.nfs = v; } });
      st = UI.stepper(fig, STEPS, { autoplay: 6500 });
      function content(i) {
        const run = S.scn === 'run', drift = S.scn === 'drift';
        const c = { head: [], worker: [], log: [], arrow: null, fail: null, ok: null };
        switch (i) {
          case 0:
            c.head = ['.env.dspark → cópia privada 0600', 'BOM e CRLF removidos', 'MAX_NUM_SEQS=6 ✓ · MTP_NUM_TOKENS=6 ✓'];
            c.worker = ['(ainda não tocado)'];
            c.log = ['$ ./start-deepseek-v4-flash-dspark.sh', 'env normalizado · knobs numéricos ok (10#)'];
            break;
          case 1:
            c.head = ['GPU_MEMORY_UTILIZATION ← _TEXT', 'TP_SIZE=2 · NNODES=2 (forçados)', 'modelo oficial @ 86f746b3…'];
            c.worker = ['(ainda não tocado)'];
            c.log = ['perfil resolvido: 1M / 6 slots / k=6', 'WARN se a API está aberta sem chave (nunca falha)'];
            break;
          case 2:
            c.head = ['docker · ssh · scp · curl ✓', 'imagem @sha256 presente ✓', run ? 'container do head JÁ EXISTE' : 'container do head? não'];
            c.worker = ['ssh BatchMode · ConnectTimeout=10', 'imagem @sha256 presente ✓', run ? 'rank já existe (sentinela 42)' : 'rank existe? não'];
            c.arrow = 'ssh';
            c.log = run ? ['dockerd restaurou os ranks (unless-stopped)', 'exit 3 → systemd: SuccessExitStatus=3'] : ['pré-requisitos ok nos dois nós', 'nenhum rank rodando'];
            if (run) c.fail = { t: 'exit 3 · já rodando', c: 'd' };
            break;
          case 3:
            c.head = ['NCCL_IB_HCA=rocep1s0f1,…', 'GID RoCE v2 ↔ IPv4 da interface ✓'];
            c.worker = ['WORKER_NCCL_IB_HCA=rocep1s0f0,…', 'GID RoCE v2 ↔ IPv4 da interface ✓'];
            c.log = ['NCCL_IB_GID_AUTO=1 → valida, não pina', 'NCCL_IB_GID_INDEX fica indefinido'];
            break;
          case 4:
            c.head = ['compose · .env · proposer · patches', S.nfs ? 'exporta pesos por NFSv4 (ro)' : 'HF_CACHE local'];
            c.worker = ['mesmos bytes, mesma ordem', S.nfs ? 'monta pesos do head (nconnect=8)' : '2ª cópia local (~157 GiB)'];
            c.arrow = 'files';
            c.log = ['dscp → WORKER_DIR · .env via tmp 0600 → mv -f', S.nfs ? 'caches de JIT continuam locais e graváveis' : 'o prepare já baixou os pesos no worker'];
            break;
          case 5:
            c.worker = ['compose run --rm … --check', drift ? '✕ FAIL-CLOSED: fonte ≠ pin' : '#117 ✓ · opt-ins ✓'];
            c.head = drift ? ['(não chega a rodar)'] : ['compose run --rm … --check', '#117 ✓ · opt-ins ✓'];
            c.log = drift ? ['preflight falhou no worker → exit 1', 'nenhum rank subiu; o par fica intacto'] : ['preflight ok: primeiro worker, depois head', 'ninguém subiu ainda'];
            if (drift) c.fail = { t: 'exit 1 · preflight', c: 'red' };
            break;
          case 6:
            c.worker = ['rank 1 · --headless', 'aguardando o TCPStore…'];
            c.head = ['rank 0 · TCPStore :25000', 'esteira de hotfixes → serve'];
            c.arrow = 'rdv';
            c.log = ['worker primeiro: o head hospeda o TCPStore', 'reiniciar o head no load derrubava o handshake (#38)'];
            break;
          default:
            c.head = ['GET /v1/models (até 100 × 15 s)', 'smoke 6 req · warmup 47 req'];
            c.worker = ['servindo (headless)', 'mesmos patches, mesma ordem'];
            c.log = ['boot frio ~6–8 min até a API', 'warmup falho vira WARN, nunca falha o boot'];
            c.ok = true;
        }
        return c;
      }
      function container(g, x, y, w, h, label, cc, C) {
        V.box(g, x, y, w, h, { r: 5, fill: V.alpha(cc, 0.28), stroke: cc, lw: 1.5 });
        for (let k = 1; k < 7; k++) V.line(g, x + (w * k) / 7, y + 4, x + (w * k) / 7, y + h - 4, { color: V.alpha(cc, 0.55), lw: 1 });
        V.box(g, x + w / 2 - 44, y + h / 2 - 9, 88, 18, { r: 5, fill: V.alpha(C.panel || C.bg, 0.92) });
        V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: 84 });
      }
      function doc(g, x, y, w, h, C, e, fall) {
        V.box(g, x, y, w, h, { r: 5, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.ink, 0.5), lw: 1.2 });
        for (let k = 0; k < 5; k++) V.line(g, x + 8, y + 12 + k * ((h - 18) / 5), x + w - 10 - (k % 2) * 12, y + 12 + k * ((h - 18) / 5), { color: V.alpha(C.ink, 0.4), lw: 2 });
        if (fall) {
          const a = V.clamp(1 - e / 1.6, 0, 1), dy = V.ease.in(V.clamp(e / 1.6, 0, 1)) * 40;
          if (a > 0) {
            chip(g, x - 6, y + 4 + dy, 'BOM', C.red || C.c, { size: 11, font: 'mono', fa: 0.3 * a, sa: a, color: V.alpha(C.ink, a) });
            chip(g, x + w - 10, y + h * 0.55 + dy * 1.3, '␍␊', C.red || C.c, { size: 11, font: 'mono', fa: 0.3 * a, sa: a, color: V.alpha(C.ink, a) });
          }
        }
      }
      function icons(g, i, b, who, e, t, C, nl) {
        const ix = b[0] + 14, iy = b[1] + 34 + nl * 18 + 10, iw = b[2] - 28, ih = b[1] + b[3] - 10 - iy;
        if (ih < 40) return;
        const cx = ix + iw / 2, cy = iy + ih / 2;
        const S2 = Math.min(ih, 110);
        const run = S.scn === 'run', drift = S.scn === 'drift';
        const zz = () => { for (let k = 0; k < 3; k++) { const u = (t * 0.5 + k / 3) % 1; V.text(g, 'z', cx + k * 10 + u * 16, cy + 10 - u * 26, { size: 12 + k * 3, weight: 700, color: V.alpha(C.muted, 1 - u) }); } };
        if (i === 0) {
          if (who === 'head') {
            doc(g, cx - 34, cy - S2 / 2 + 4, 68, Math.min(84, S2 - 8), C, e, true);
            if (e > 1.2) chip(g, cx + 42, cy - 10, 'modo 0600', C.a, { size: 11, font: 'mono' });
          } else zz();
        } else if (i === 1) {
          if (who === 'head') {
            const G = [['u', 0.835 / 0.87, '0,835'], ['TP', 1, '2'], ['rev', 0.7, '86f7…']];
            const gr = Math.min(28, S2 / 3.2, iw / 8);
            G.forEach((q, k) => {
              const gx = cx + (k - 1) * (gr * 2.6), gy = cy + gr * 0.5;
              g.lineWidth = 5; g.lineCap = 'round';
              g.strokeStyle = V.alpha(C.muted, 0.25); g.beginPath(); g.arc(gx, gy, gr, Math.PI, 2 * Math.PI); g.stroke();
              const v = Math.min(1, V.ease.out(V.clamp(e / 1.2, 0, 1)) * q[1]);
              g.strokeStyle = C.a; g.beginPath(); g.arc(gx, gy, gr, Math.PI, Math.PI + Math.PI * v); g.stroke(); g.lineCap = 'butt';
              V.text(g, q[2], gx, gy - 2, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.ink });
              V.text(g, q[0], gx, gy + 14, { size: 11, align: 'center', color: C.muted });
            });
          } else zz();
        } else if (i === 2) {
          if (run) { container(g, ix + 6, cy - 20, iw - 12, 40, who === 'head' ? 'rank 0 no ar' : 'rank 1 no ar', C.d, C); return; }
          if (who === 'head') {
            const kx = cx - 30, ky = cy;
            V.dot(g, kx, ky, 12, V.alpha(C.a, 0.9)); V.dot(g, kx, ky, 5, C.panel || C.bg);
            V.box(g, kx + 10, ky - 3, 44, 6, { r: 2, fill: C.a }); V.box(g, kx + 42, ky + 3, 5, 7, { r: 1, fill: C.a }); V.box(g, kx + 50, ky + 3, 4, 5, { r: 1, fill: C.a });
            V.text(g, 'ssh -o BatchMode=yes', cx + 36, ky + 4, { size: 11, font: 'mono', color: C.ink, maxW: iw / 2 });
          } else {
            const r = 10 + 4 * Math.sin(t * 5);
            g.strokeStyle = V.alpha(C.b, 0.8); g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.stroke();
            V.text(g, '?', cx, cy + 5, { size: 14, weight: 700, align: 'center', color: C.b });
          }
        } else if (i === 3) {
          const nw = Math.min(150, iw * 0.7), nh = Math.min(46, S2 * 0.6);
          V.box(g, cx - nw / 2, cy - nh / 2, nw, nh, { r: 6, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.ink, 0.5), lw: 1.2 });
          V.text(g, 'ConnectX-7', cx - nw / 2 + 8, cy - nh / 2 + 14, { size: 11, color: C.muted });
          ['PF0', 'PF1'].forEach((p, k) => {
            const px = cx - nw / 2 + 12 + k * (nw / 2), py = cy + 2;
            const blink = Math.sin(t * 6 + k * 2) > 0;
            V.box(g, px, py, nw / 2 - 24, 14, { r: 3, fill: V.alpha(C.a, 0.2), stroke: C.a, lw: 1 });
            V.dot(g, px + nw / 2 - 30, py + 7, 3, blink ? C.a : V.alpha(C.a, 0.3));
            V.text(g, p, px + 6, py + 11, { size: 11, font: 'mono', color: C.ink });
          });
          if (e > 0.8) chip(g, cx, cy + nh / 2 + 14, 'GID RoCE v2 ✓', C.a, { size: 11, font: 'mono', align: 'center' });
        } else if (i === 4) {
          const n = who === 'head' ? 5 : Math.min(5, Math.floor(e * 2.2));
          for (let k = 0; k < n; k++) doc(g, cx - 22 + k * 7, cy - 24 + k * 4, 36, 42, C, 0, false);
          if (S.nfs && who === 'worker') chip(g, cx + 44, cy - 10, 'NFSv4 ro', C.d, { size: 11, font: 'mono' });
        } else if (i === 5) {
          const t0 = who === 'worker' ? 0 : 1.6;
          const ee = e - t0;
          if (drift && who === 'head') { V.text(g, '—', cx, cy + 5, { size: 16, align: 'center', color: C.muted }); return; }
          if (ee < 0) { V.text(g, 'na fila do preflight…', cx, cy + 4, { size: 11, align: 'center', color: C.muted }); return; }
          doc(g, cx - 40, cy - 26, 50, 56, C, 0, false);
          const u = V.clamp(ee / 1.2, 0, 1), mx = cx - 30 + u * 36, my = cy - 10 + Math.sin(ee * 6) * 6;
          g.strokeStyle = C.d; g.lineWidth = 3; g.beginPath(); g.arc(mx, my, 11, 0, Math.PI * 2); g.stroke();
          V.line(g, mx + 8, my + 8, mx + 18, my + 18, { color: C.d, lw: 4 });
          if (u >= 1) {
            const bad = drift && who === 'worker';
            V.text(g, bad ? '✕' : '✓', cx + 40, cy + 8, { size: 26, weight: 700, align: 'center', color: bad ? C.red || C.c : C.a });
          }
        } else if (i === 6) {
          const t0 = who === 'worker' ? 0.1 : 1.7;
          const u = V.ease.back(V.clamp((e - t0) / 0.8, 0, 1));
          if (e < t0) { V.text(g, 'esperando…', cx, cy + 4, { size: 11, align: 'center', color: C.muted }); return; }
          const ch = Math.min(46, S2 * 0.6);
          container(g, ix + 6, V.lerp(iy - 30, cy - ch / 2, u), iw - 12, ch, who === 'head' ? 'rank 0 · TCPStore' : 'rank 1 · headless', who === 'head' ? C.a : C.b, C);
        } else {
          if (who === 'head') {
            const pw = Math.min(iw, 220), px = cx - pw / 2, py = cy - 14;
            const k = V.clamp(e / 2.6, 0, 1);
            V.box(g, px, py, pw, 12, { r: 6, fill: V.alpha(C.muted, 0.2) });
            V.box(g, px, py, Math.max(6, pw * k), 12, { r: 6, fill: C.a });
            V.text(g, k < 1 ? 'tentativa ' + Math.max(1, Math.round(k * 26)) + '/100' : '✓ deepseek-v4-flash-vision-exp · 1048576', cx, py + 30, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: iw });
            if (k >= 1) for (let q = 0; q < 6; q++) { const a = t * 2 + q; V.glow(g, cx + Math.cos(a) * 60, py + 6 + Math.sin(a) * 18, 8, C.series[q], 0.8); }
          } else {
            const ch = Math.min(46, S2 * 0.6);
            container(g, ix + 6, cy - ch / 2, iw - 12, ch, 'rank 1 · servindo', C.b, C);
            V.glow(g, ix + iw - 10, cy - ch / 2, 10 + 3 * Math.sin(t * 4), C.a, 0.8);
          }
        }
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, e = st.elapsed();
        const narrow = w < 600;
        const logH = 60, pad = 12;
        const blocked = (S.scn === 'run' && i >= 3) || (S.scn === 'drift' && i >= 6);
        const c = content(i);
        let hb, wb;
        if (!narrow) {
          const bw = (w - pad * 2) * 0.41, bh = h - logH - pad * 3;
          hb = [pad, pad, bw, bh]; wb = [w - pad - bw, pad, bw, bh];
        } else {
          const bw = w - pad * 2, bh = (h - logH - pad * 3 - 34) / 2;
          hb = [pad, pad, bw, bh]; wb = [pad, pad + bh + 34, bw, bh];
        }
        // passos bloqueados ficam apagados: V.text com {alpha} volta o globalAlpha para 1,
        // então o fator ga vai explícito em cada texto e é reaplicado depois de cada chamada
        const ga = blocked ? 0.25 : 1;
        g.globalAlpha = ga;
        const box = (b, title, lines2, cc, who) => {
          g.globalAlpha = ga;
          V.box(g, b[0], b[1], b[2], b[3], { r: 14, fill: V.alpha(cc, 0.07), stroke: V.alpha(cc, 0.75), lw: 1.5 });
          V.text(g, title, b[0] + 12, b[1] + 20, { size: 12.5, weight: 700, color: cc, maxW: b[2] - 20, alpha: ga });
          lines2.forEach((s, k) => {
            const a = V.clamp((e - 0.35 * k) / 0.35, 0, 1);
            if (a <= 0) return;
            const bad = s.startsWith('✕') || s.includes('JÁ EXISTE');
            V.text(g, s, b[0] + 14, b[1] + 42 + k * 18, { size: 11.5, font: 'mono', color: bad ? C.red || C.c : C.ink, alpha: a * ga, maxW: b[2] - 24 });
          });
          g.globalAlpha = ga;
          if (!blocked) icons(g, i, b, who, e, t, C, lines2.length);
        };
        box(hb, 'head · spark-1', c.head, C.a, 'head');
        box(wb, 'worker · spark-2', c.worker, C.b, 'worker');
        g.globalAlpha = ga;
        const A = narrow ? [hb[0] + hb[2] * 0.5, hb[1] + hb[3]] : [hb[0] + hb[2], hb[1] + hb[3] * 0.5];
        const B = narrow ? [wb[0] + wb[2] * 0.5, wb[1]] : [wb[0], wb[1] + wb[3] * 0.5];
        if (c.arrow && !blocked) {
          const rev = c.arrow === 'rdv';
          const a0 = rev ? B : A, b0 = rev ? A : B;
          V.arrow(g, a0[0], a0[1], b0[0], b0[1], { color: V.alpha(rev ? C.d : C.a, 0.8), lw: 2, head: 9 });
          const labels = c.arrow === 'files' ? ['compose', '.env', 'patches', 'vision_exp', 'tp3'] : c.arrow === 'ssh' ? ['ssh', 'ssh', 'ssh'] : ['rendezvous'];
          labels.forEach((lb, k) => {
            const u = (t * 0.45 + k / labels.length) % 1;
            const p = V.along(a0[0], a0[1], b0[0], b0[1], u);
            if (c.arrow === 'files' && !narrow) chip(g, p[0], p[1] - 12, lb, C.a, { size: 11, font: 'mono', align: 'center', fa: 0.3 });
            else V.dot(g, p[0], p[1], 4, rev ? C.d : C.a);
          });
          if (rev && !narrow) V.text(g, 'aguarda TCPStore', (a0[0] + b0[0]) / 2, (a0[1] + b0[1]) / 2 + 20, { size: 11, color: C.d, align: 'center' });
        } else if (!narrow) {
          V.line(g, A[0] + 6, A[1], B[0] - 6, B[1], { color: V.alpha(C.muted, 0.35), lw: 1.2, dash: [3, 6] });
        }
        g.globalAlpha = 1;
        const ly = h - logH - pad;
        V.box(g, pad, ly, w - pad * 2, logH, { r: 10, fill: V.alpha(C.ink, C.dark ? 0.06 : 0.05), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const logs = blocked ? [S.scn === 'run' ? 'não executado: o start já saiu com 3 no passo 3' : 'não executado: o preflight falhou no passo 6', 'nenhum rank ficou meio patcheado'] : c.log;
        logs.forEach((s, k) => {
          const n = Math.floor(V.clamp((e - 0.2 - k * 0.6) * 40, 0, s.length));
          V.text(g, (k === 0 ? '▸ ' : '  ') + s.slice(0, n), pad + 12, ly + 22 + k * 19, { size: 11.5, font: 'mono', color: blocked ? C.muted : C.ink, maxW: w - pad * 2 - 24 });
        });
        const mid = narrow ? [w / 2, hb[1] + hb[3] + 17] : [w / 2, pad + (h - logH - pad * 3) / 2 - 30];
        if (blocked) chip(g, mid[0], narrow ? mid[1] : (h - logH) / 2, S.scn === 'run' ? 'exit 3 · nada a fazer' : 'exit 1 · nenhum rank no ar', S.scn === 'run' ? C.d : C.red || C.c, { size: 12.5, weight: 700, align: 'center', fa: 0.25 });
        if (c.fail && !blocked) {
          const cc = c.fail.c === 'red' ? C.red || C.c : C.d;
          const pulse = 1 + 0.04 * Math.sin(t * 6);
          g.save(); g.translate(mid[0], mid[1]); g.scale(pulse, pulse);
          chip(g, 0, 0, c.fail.t, cc, { size: 13, weight: 700, align: 'center', fa: 0.25 });
          g.restore();
        }
        if (c.ok && S.scn === 'ok' && e > 2.6) chip(g, mid[0], mid[1], '✓ pronto', C.a, { size: 13, weight: 700, align: 'center', fa: 0.25 });
      });
    })();

    /* ============ F5 · a esteira do entrypoint ============ */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      if (!fig) return;
      autoH(fig, 360, 720, 640);
      const G = [
        { n: 'Preparo do shell', c: 'muted', d: 'PATH/CUDA, valida DSPARK_API_KEYS, faz unset dos NCCL_* vazios (vazio mascara /etc/nccl.conf e GID vazio vira índice 0), cria o dir do flight recorder e exige ABLATE ∈ {0, 1}.', p: [['chaves de API', 'A'], ['unset NCCL vazio', 'A'], ['flight recorder', 'A'], ['gate ABLATE', 'A']] },
        { n: 'Encoder do checkpoint', c: 'c', d: 'Copia encoding_dsv4.py para dentro do vLLM, corrige o reasoning_effort (low deixa de virar high), #21 (args dict no DSML), #31 budget na GPU (opt-in), #55 (tool call truncada → finish_reason length).', p: [['encoder', 'A'], ['esforço', 'A'], ['#21', 'A'], ['#31', 'O', 'i31'], ['#55', 'A']] },
        { n: 'KV & IPC', c: 'a', d: '#22 manda nvfp4_ds_mla para o kernel FP8 (≈16× a 600K+), #79 troca o spin-wait de 1 s por 2 ms, #117 limita a espera do ring buffer SHM (recheck de 5 s) e libera slots em finally.', p: [['#22', 'S'], ['#79', 'S'], ['#117', 'S']] },
        { n: 'Backports v0.27', c: 'b', d: 'Seis backports do vLLM 0.27: #50312, #49486+#52492, #48407 (dormente de propósito), #48957, #50298, #44993. DSPARK_SKIP_HOTFIX=1 pula todos (só para A/B).', p: [['#50312', 'B'], ['#49486', 'B'], ['#48407', 'B'], ['#48957', 'B'], ['#50298', 'B'], ['#44993', 'B']] },
        { n: 'API & segredos', c: 'd', d: 'Redação da chave no log (obrigatória se houver chave), store limitado da Responses API, #138 (histórico sem type) e compat do agent_message do Codex. Os três últimos são opt-in.', p: [['redação', 'K'], ['store', 'O', 'store'], ['#138', 'O', 'i138'], ['Codex', 'O', 'codex']] },
        { n: 'Visão', c: 'e', d: 'hotfix-dsv4-vision-exp.py enxerta ViT de 32 camadas + Aligner no DeepseekV4ForCausalLM (só texto na 0.1.1), remapeia bias_vl no drafter e registra o processor multimodal.', p: [['Vision-Exp', 'A']] },
        { n: 'Kernels opt-in', c: 'a', d: '#141 fatia o decode sparse-MLA em ≤ 64 linhas (mitigação, não causa raiz), SP indexer (top-k distribuído exato no prefill) e alias de headers DeepGEMM sm121 → sm120.', p: [['#141', 'O', 'i141'], ['SP indexer', 'O', 'sp'], ['alias sm121', 'O', 'alias']] },
        { n: 'Scheduler', c: 'f', d: '#109 (passo só de encoder sem token fantasma), #27 (teto de prefills em voo), #43 (piso de decode), #26 v2 (prefix cache híbrido MLA+SWA), #133 (sem especialização Triton por alinhamento). Adaptive chunk e Markov replicada são opt-in.', p: [['#109', 'A'], ['#27', 'A'], ['adaptive chunk', 'O', 'adapt'], ['Markov replicada', 'O', 'markov'], ['#43', 'A'], ['#26 v2', 'A'], ['#133', 'A']] },
        { n: 'Ablação & stops', c: 'c', d: 'A ablação em runtime só age com ABLATE=1 (senão confere que o arquivo está stock e mantém o carimbo do cache AOT). Stops do cliente ficam dormentes até </think>.', p: [['ablação', 'A'], ['stops em <think>', 'S']] },
        { n: 'Tools opt-in', c: 'd', d: '#52/#120 (turno final do assistant), #144 (diretiva de esforço alinhada ao prefix cache), #136 (XGrammar × especulação), #191 (contrato fail-closed de tool_choice).', p: [['#52/#120', 'O', 'i52'], ['#144', 'O', 'i144'], ['#136', 'O', 'i136'], ['#191', 'O', 'i191']] },
        { n: 'Args condicionais', c: 'b', d: '--async-scheduling (DSPARK_ASYNC_SCHEDULING=1, padrão) e --attention-config {"use_fp4_indexer_cache":true} (só junto com o MXFP4 indexer).', p: [['--async-scheduling', 'Y'], ['--attention-config', 'O', 'mxfp4']] },
        { n: 'Especulação & KV opt-in', c: 'f', d: 'block-k (permite k=5), RoPE-SWA (port de vllm#54815), DSpark SWA prefix, DSML recovery, MXFP4 indexer cache e C128A prefill cache.', p: [['block-k', 'O', 'blockk'], ['RoPE-SWA', 'O', 'ropeswa'], ['SWA prefix', 'O', 'swapfx'], ['DSML recovery', 'O', 'dsml'], ['MXFP4 indexer', 'O', 'mxfp4'], ['C128A', 'O', 'c128']] },
        { n: 'Plugin & TP=3', c: 'e', d: 'Plugin GB10 híbrido NVFP4 (ENABLE_VLLM_GB10_PATCH=1; não se aplica a este checkpoint — interpretação: o plugin exige ModelOpt NVFP4, e os experts do DSV4 são MXFP4) e o padding de cabeças/grupos quando TP_SIZE=3.', p: [['plugin GB10', 'O', 'gb10'], ['pad TP=3', 'T']] },
        { n: 'Monta e exec', c: 'a', d: 'Valida DRAFT_SAMPLE_METHOD, monta SPECULATIVE_CONFIG e os kwargs de thinking dentro do container, adiciona --revision e converte LIMIT_MM → exec /usr/local/bin/vllm serve com ~40 flags.', p: [['sample method', 'A'], ['spec JSON', 'A'], ['thinking kwargs', 'A'], ['--revision', 'A'], ['limit-mm', 'A'], ['exec', 'A']] },
      ];
      const PRESETS = {
        pad: { name: 'padrão', on: [] },
        agent: { name: 'exemplo: agente', on: ['store', 'i52', 'i144', 'i136', 'i191', 'dsml'], key: true },
        ab: { name: 'A/B sem backports', on: [], skipB: true },
        tp3: { name: 'TP=3', on: [], tp3: true },
      };
      const S = { pre: 'pad', drift: false, t0: 0 };
      const active = (p) => {
        const P = PRESETS[S.pre];
        switch (p[1]) {
          case 'A': case 'S': case 'Y': return true;
          case 'B': return !P.skipB;
          case 'K': return !!P.key;
          case 'T': return !!P.tp3;
          case 'O': return P.on.includes(p[2]);
          default: return false;
        }
      };
      const FAIL_G = 7, FAIL_P = 1; // #27
      const DOT_T = 0.11, MOVE = 0.45, BASE = 0.25;
      const TL = (() => {
        const T = []; let acc = 0.4;
        G.forEach((grp, gi) => { const dwell = BASE + grp.p.length * DOT_T + 0.25; T.push({ a: acc, l: acc + dwell }); acc += dwell + (gi < G.length - 1 ? MOVE : 0); });
        return { T, total: acc + 2.6 };
      })();
      let H = null;
      const ctl = UI.controls(fig);
      const rd = UI.readout(ctl, 'passos ativos', '—');
      const cnt = () => { let a = 0, o = 0, n = 0; G.forEach((grp) => grp.p.forEach((p) => { n++; if (active(p)) { a++; if (p[1] === 'O' || p[1] === 'K' || p[1] === 'T') o++; } })); rd.set(a + ' de ' + n + ' · opt-ins ' + o); };
      UI.seg(ctl, { label: 'Gates', options: Object.keys(PRESETS).map((k) => ({ v: k, label: PRESETS[k].name })), value: 'pad', onChange: (v) => { S.pre = v; S.drift = false; S.t0 = H ? H.t : 0; cnt(); } });
      // o drift começa já na estação 7 (a anterior ao #27): quem clica vê a falha em ~1,5 s
      UI.button(ctl, 'simular drift', () => { S.drift = true; S.t0 = (H ? H.t : 0) - TL.T[FAIL_G - 1].a; }, { primary: true });
      UI.button(ctl, '↺ reiniciar', () => { S.drift = false; S.t0 = H ? H.t : 0; });
      cnt();
      function layout(w) {
        const cols = w >= 760 ? 7 : w >= 520 ? 5 : 3;
        const rows = Math.ceil(G.length / cols);
        const mx = 20, gx = 8, rail = 24;
        const sw = (w - 2 * mx - (cols - 1) * gx) / cols;
        const sh = 60, top = 44, pitch = sh + rail + 6;
        const st = G.map((grp, i) => {
          const r = Math.floor(i / cols), k = i % cols, cc = r % 2 === 0 ? k : cols - 1 - k;
          const x = mx + cc * (sw + gx), y = top + r * pitch;
          return { x, y, w: sw, h: sh, cx: x + sw / 2, ry: y - 13, r };
        });
        return { cols, rows, mx, sw, sh, top, pitch, st, infoY: top + rows * pitch - 4 };
      }
      function moveBetween(a, b, u, L, w) {
        if (a.r === b.r) return [V.lerp(a.cx, b.cx, u), a.ry];
        const edge = a.r % 2 === 0 ? w - L.mx + 11 : L.mx - 11;
        const d1 = Math.abs(edge - a.cx), d2 = Math.abs(b.ry - a.ry), d3 = Math.abs(edge - b.cx), tot = d1 + d2 + d3;
        const s = u * tot;
        if (s < d1) return [V.lerp(a.cx, edge, s / d1), a.ry];
        if (s < d1 + d2) return [edge, V.lerp(a.ry, b.ry, (s - d1) / d2)];
        return [V.lerp(edge, b.cx, (s - d1 - d2) / Math.max(1, d3)), b.ry];
      }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const L = layout(w), T = TL.T;
        let tt = t - S.t0;
        if (!S.drift) tt = tt % TL.total;
        const failT = T[FAIL_G].a + BASE * 0.5 + FAIL_P * DOT_T + 0.1;
        const failed = S.drift && tt >= failT;
        if (S.drift && tt > failT) tt = failT;
        let cur = 0, pos = null, moving = false;
        for (let i = 0; i < G.length; i++) {
          if (tt < T[i].a) {
            const prev = T[i - 1];
            if (!prev) { pos = [L.st[0].cx, L.st[0].ry]; cur = 0; break; }
            const u = V.ease.inOut(V.clamp((tt - prev.l) / MOVE, 0, 1));
            pos = moveBetween(L.st[i - 1], L.st[i], u, L, w); cur = i - 1; moving = true; break;
          }
          if (tt <= T[i].l || i === G.length - 1) { pos = [L.st[i].cx, L.st[i].ry]; cur = i; break; }
        }
        const done = !S.drift && tt > T[G.length - 1].l;
        V.text(g, 'command: bash -lc "… || exit 1; … || exit 1; exec vllm serve …"', 14, 16, { size: 11, font: 'mono', color: C.muted, maxW: w - 28 });
        // trilho
        const xl = L.mx - 11, xr = w - L.mx + 11;
        g.beginPath();
        for (let r = 0; r < L.rows; r++) {
          const y = L.top + r * L.pitch - 13, lr = r % 2 === 0;
          if (r === 0) g.moveTo(lr ? xl : xr, y); else g.lineTo(lr ? xl : xr, y);
          g.lineTo(lr ? xr : xl, y);
        }
        g.strokeStyle = V.alpha(C.muted, 0.28); g.lineWidth = 7; g.lineJoin = 'round'; g.stroke();
        g.setLineDash([6, 8]); g.lineDashOffset = -t * 22; g.strokeStyle = V.alpha(C.ink, 0.35); g.lineWidth = 2; g.stroke(); g.setLineDash([]);
        // estações
        L.st.forEach((s, i) => {
          const grp = G[i], cc = col(C, grp.c);
          const isCur = i === cur && !moving && !done;
          const visited = i < cur || (i === cur && !moving) || done;
          const red = failed && i === FAIL_G;
          const ccR = red ? C.red || C.c : cc;
          V.box(g, s.x, s.y, s.w, s.h, { r: 10, fill: V.alpha(ccR, isCur ? 0.2 : 0.08), stroke: V.alpha(ccR, isCur ? 1 : 0.55), lw: isCur ? 2.2 : 1.2 });
          if (isCur) V.glow(g, s.cx, s.y + s.h / 2, s.w * 0.45, ccR, 0.25);
          V.text(g, String(i + 1).padStart(2, '0'), s.x + 7, s.y + 15, { size: 11, font: 'mono', color: cc });
          wtext(g, grp.n, s.x + 27, s.y + 15, s.w - 32, { size: 11, weight: 600, color: C.ink, maxLines: 2, lh: 13 });
          const np = grp.p.length, dr = Math.min(4.2, (s.w - 16) / (np * 2.6)), dx = Math.min(11, (s.w - 16) / np);
          grp.p.forEach((p, k) => {
            const x = s.x + 10 + k * dx + dr, y = s.y + s.h - 11;
            const on = active(p);
            const lit = visited && (i < cur || done || tt >= T[i].a + BASE * 0.5 + k * DOT_T);
            if (failed && i === FAIL_G && k === FAIL_P) { V.dot(g, x, y, dr + 1.5, C.red || C.c); V.glow(g, x, y, 12, C.red || C.c, 0.9); return; }
            if (on) {
              V.dot(g, x, y, dr, lit ? cc : V.alpha(cc, 0.35));
              if (lit && i === cur && !moving && Math.abs(tt - (T[i].a + BASE * 0.5 + k * DOT_T)) < 0.15) V.glow(g, x, y, 10, cc, 0.9);
            } else {
              g.beginPath(); g.arc(x, y, dr, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.8); g.lineWidth = 1.2; g.stroke();
            }
          });
        });
        // a caixinha vLLM
        if (pos) {
          const c2 = failed ? C.red || C.c : done ? C.a : C.ds || C.b;
          const bob = moving ? Math.sin(t * 14) * 1.2 : 0;
          V.box(g, pos[0] - 20, pos[1] - 9 + bob, 40, 18, { r: 5, fill: c2, stroke: V.alpha(C.ink, 0.4), lw: 1 });
          V.text(g, failed ? 'FAIL' : 'vLLM', pos[0], pos[1] + 0.5 + bob, { size: 11, weight: 700, font: 'mono', align: 'center', base: 'middle', color: onFill(C) });
        }
        // painel de informação
        const iy = L.infoY + 14, iw = w - 40;
        const grp = G[cur];
        let title, body, cc = col(C, grp.c);
        if (failed) { title = 'FAIL-CLOSED na estação ' + (FAIL_G + 1) + ' · #27'; body = 'O patcher do #27 não encontrou a fonte esperada (âncora/pin não batem) → exit 1. O container termina antes do exec: nenhum vllm serve roda com metade das correções. O preflight do launcher só cobre os opt-ins sensíveis e o #117.'; cc = C.red || C.c; }
        else if (done) { title = 'exec /usr/local/bin/vllm serve …'; body = 'Todas as estações passaram, na mesma ordem nos dois ranks. Só agora o processo vira o servidor, e a camada gravável do container guarda os bytes patcheados até ele ser recriado.'; cc = C.a; }
        else { title = String(cur + 1).padStart(2, '0') + ' · ' + grp.n; body = grp.d; }
        V.text(g, title, 20, iy, { size: 12.5, weight: 700, color: cc, maxW: iw });
        const bh2 = wtext(g, body, 20, iy + 19, iw, { size: 11.5, color: C.ink, maxLines: w < 520 ? 7 : 3, lh: 15 });
        // chips dos patches da estação atual
        if (!done && !failed) {
          let cx = 20, cy = iy + 19 + bh2 + 10;
          grp.p.forEach((p) => {
            const on = active(p);
            const lab = (on ? '✓ ' : '○ ') + p[0] + (p[1] === 'O' && !on ? ' (opt-in)' : p[1] === 'K' && !on ? ' (sem chave)' : p[1] === 'T' && !on ? ' (TP≠3)' : '');
            const wd = tw(g, lab, 11, 'mono', 500) + 16;
            if (cx + wd > 20 + iw) { cx = 20; cy += 22; }
            if (cy > h - 8) return;
            chip(g, cx, cy, lab, on ? cc : C.muted, { size: 11, font: 'mono', fa: on ? 0.18 : 0.05 });
            cx += wd + 6;
          });
        }
      });
    })();

    /* ============ F6 · anatomia de um patch ============ */
    (function () {
      const fig = root.querySelector('#fig-patch');
      if (!fig) return;
      autoH(fig, 330, 520);
      const S = { state: 'stock' };
      const STEPS = [
        { t: 'Inspecionar', d: 'Confere que o vLLM instalado é exatamente a versão esperada (<code>0.25.2.dev0+g752a3a504…</code>), que o alvo é um arquivo regular (não symlink) e calcula o SHA-256 dos bytes.' },
        { t: 'Classificar contra os pins', d: 'Hash e tamanho do <b>patched</b>: nada a fazer, sai 0 (idempotência). Hash do <b>stock</b>: segue. Qualquer outra coisa é fonte desconhecida: <b>FAIL-CLOSED</b>, exit 1.' },
        { t: 'Transformar em memória', d: 'Exige que a região antiga apareça <b>exatamente uma vez</b> e que o marcador <code>[dspark-block-k]</code> esteja ausente. Troca o trecho e roda <code>compile()</code> na pós-imagem: Python inválido morre aqui, antes de tocar o disco.' },
        { t: 'Conferir a pós-imagem', d: 'O resultado também é pinado. O SHA-256 precisa ser exatamente o do patched (56.987 B), o que prova que a transformação gerou <i>os</i> bytes esperados, e não algo parecido.' },
        { t: 'Publicar atomicamente', d: 'Temporário <b>no mesmo diretório</b> → <code>write</code> + <code>flush</code> + <code>fsync</code> → permissões copiadas → <code>os.replace</code>. Nenhum leitor vê meio arquivo, e uma queda no meio deixa o original intacto.' },
        { t: 'Reverificar', d: 'Inspeciona de novo: o alvo agora precisa classificar como <i>patched</i>, senão é erro. Nos <code>.sh</code> transacionais, o equivalente é o rollback byte-exato de todos os alvos.' },
      ];
      const ctl = UI.controls(fig);
      let st = null;
      UI.seg(ctl, { label: 'Estado do alvo', options: [{ v: 'stock', label: 'stock' }, { v: 'patched', label: 'já patcheado' }, { v: 'other', label: 'outra versão' }], value: 'stock', onChange: (v) => { S.state = v; if (st) st.go(0); } });
      st = UI.stepper(fig, STEPS, { autoplay: 5200 });
      const rng = V.rng(44);
      const shade = Array.from({ length: 60 }, () => rng());
      const HEX = '0123456789abcdef';
      const PIN_S = '3f1abd1c…', PIN_P = '7fffe035…';
      const OLD = [[' ', 'elif ('], [' ', '    k > n_predict'], [' ', '    and k % n_predict != 0'], [' ', '):  # reuso do módulo MTP'], [' ', '    raise ValueError(…)']];
      const NEW = [[' ', 'elif ('], ['+', '    self.method != "dspark"'], ['+', '    # [dspark-block-k]'], ['~', '    and k > n_predict'], [' ', '    and k % n_predict != 0'], [' ', '):  raise ValueError(…)']];
      function status(k) {
        const i = st.i;
        if (S.state === 'stock') return k < i ? 'ok' : k === i ? 'act' : 'pend';
        if (k === 0) return i === 0 ? 'act' : 'ok';
        if (k === 1) return i >= 1 ? (S.state === 'patched' ? 'idem' : 'fail') : 'pend';
        return i >= 1 ? 'skip' : 'pend';
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, e = st.elapsed();
        const narrow = w < 600, pad = 14;
        const fw = narrow ? w - 2 * pad : w * 0.5 - pad - 6, fh = narrow ? 262 : h - 2 * pad;
        const fx = pad, fy = pad;
        V.box(g, fx, fy, fw, fh, { r: 12, fill: V.alpha(C.b, 0.06), stroke: V.alpha(C.b, 0.6), lw: 1.3 });
        V.text(g, 'vllm/config/speculative.py', fx + 12, fy + 20, { size: 11.5, font: 'mono', weight: 600, color: C.ink, maxW: fw - 90 });
        const replaced = S.state === 'stock' && i >= 4 && (i > 4 || e > 1.6);
        const isPatched = S.state === 'patched' || replaced;
        V.text(g, isPatched ? '56.987 B' : S.state === 'other' ? '? B' : '56.845 B', fx + fw - 12, fy + 20, { size: 11, font: 'mono', align: 'right', color: C.muted });
        // faixa de bytes
        const cols = 15, rows = 3, gx = 3;
        const cw = (fw - 24 - (cols - 1) * gx) / cols, ch = 12;
        const reg = [18, 19, 20, 21];
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const k = r * cols + c;
          let fill = V.alpha(C.muted, 0.15 + 0.35 * shade[k]);
          if (reg.includes(k)) fill = isPatched ? V.alpha(C.a, 0.85) : S.state === 'other' ? V.alpha(C.red || C.c, 0.55) : V.alpha(C.d, 0.8);
          V.box(g, fx + 12 + c * (cw + gx), fy + 30 + r * (ch + gx), cw, ch, { r: 3, fill });
        }
        if (i === 0 || (i === 5 && S.state === 'stock')) {
          const u = (e * 0.8) % 1, sx = fx + 12 + u * (fw - 24);
          V.line(g, sx, fy + 26, sx, fy + 32 + rows * (ch + gx), { color: C.a, lw: 2 });
          V.glow(g, sx, fy + 32 + (rows * (ch + gx)) / 2, 16, C.a, 0.5);
        }
        const hy = fy + 36 + rows * (ch + gx) + 10;
        let hash;
        if (i === 0 && e < 1.0) hash = Array.from({ length: 8 }, (_, k) => HEX[(Math.floor(t * 30) * 7 + k * 5) % 16]).join('') + '…';
        else hash = isPatched ? PIN_P : S.state === 'other' ? '≠ pins (outra build)' : PIN_S;
        V.text(g, 'sha256: ' + hash, fx + 12, hy, { size: 12, font: 'mono', weight: 600, color: S.state === 'other' && !(i === 0 && e < 1) ? C.red || C.c : C.ink });
        const py = hy + 20;
        const matchS = S.state === 'stock' && i >= 1 && !replaced, matchP = isPatched && i >= 1;
        chip(g, fx + 12, py, 'PIN stock ' + PIN_S + ' · 56.845 B', matchS ? C.a : C.muted, { size: 11, font: 'mono', fa: matchS ? 0.25 : 0.08, maxW: fw - 24 });
        chip(g, fx + 12, py + 24, 'PIN patched ' + PIN_P + ' · 56.987 B', matchP ? C.a : C.muted, { size: 11, font: 'mono', fa: matchP ? 0.25 : 0.08, maxW: fw - 24 });
        // região (resumo; k = self.num_speculative_tokens)
        const dy0 = py + 50;
        const showNew = isPatched || (S.state === 'stock' && i >= 2);
        const L = showNew ? NEW : OLD;
        V.text(g, showNew ? 'região após o patch (resumo)' : 'região-alvo, exatamente 1× (resumo)', fx + 12, dy0, { size: 11, color: C.muted, maxW: fw - 24 });
        const lh = 14;
        L.forEach(([m, s], k) => {
          const y = dy0 + 16 + k * lh;
          if (y > fy + fh - 6) return;
          const cc = m === '+' ? C.a : m === '~' ? C.d : S.state === 'other' ? V.alpha(C.red || C.c, 0.9) : C.ink;
          if (m === '+' || m === '~') V.box(g, fx + 8, y - 11, fw - 16, lh, { r: 3, fill: V.alpha(m === '+' ? C.a : C.d, 0.12) });
          V.text(g, (m === ' ' ? ' ' : m) + ' ' + s, fx + 12, y, { size: 11, font: 'mono', color: cc, maxW: fw - 24 });
        });
        // portões
        const gx0 = narrow ? pad : w * 0.5 + 6, gy0 = narrow ? fy + fh + 12 : pad + 4, gw = narrow ? w - 2 * pad : w * 0.5 - pad - 6;
        const rowH = 23;
        const names = ['inspecionar', 'classificar contra os pins', 'transformar em memória', 'conferir a pós-imagem', 'publicar atomicamente', 'reverificar'];
        names.forEach((nm, k) => {
          const s = status(k), y = gy0 + k * rowH + 8;
          const cc = s === 'ok' ? C.a : s === 'act' ? C.d : s === 'fail' ? C.red || C.c : s === 'idem' ? C.b : C.muted;
          V.dot(g, gx0 + 10, y, 8, V.alpha(cc, s === 'pend' || s === 'skip' ? 0.2 : 0.9));
          const ic = s === 'ok' ? '✓' : s === 'fail' ? '✕' : s === 'idem' ? '=' : s === 'skip' ? '–' : s === 'act' ? '•' : '';
          V.text(g, ic, gx0 + 10, y + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: s === 'pend' || s === 'skip' ? C.muted : onFill(C) });
          if (s === 'act') V.glow(g, gx0 + 10, y, 16, cc, 0.7);
          V.text(g, (k + 1) + '. ' + nm, gx0 + 26, y + 4, { size: 12, weight: s === 'act' ? 700 : 500, color: s === 'pend' || s === 'skip' ? C.muted : C.ink, maxW: gw - 34 });
        });
        const oy = gy0 + 6 * rowH + 10;
        let txt = 'em andamento…', cc = C.muted;
        if (S.state === 'stock' && i === 5 && e > 1) { txt = '✓ applied → exit 0 · a esteira segue'; cc = C.a; }
        if (S.state === 'patched' && i >= 1) { txt = '= already-patched → exit 0 (idempotente)'; cc = C.b; }
        if (S.state === 'other' && i >= 1) { txt = '✕ FAIL-CLOSED: unsupported target bytes → exit 1'; cc = C.red || C.c; }
        V.box(g, gx0, oy, gw, 34, { r: 10, fill: V.alpha(cc, 0.12), stroke: V.alpha(cc, 0.8), lw: 1.4 });
        V.text(g, txt, gx0 + 12, oy + 21, { size: 12, weight: 700, color: cc === C.muted ? C.muted : C.ink, maxW: gw - 24 });
        // pós-imagem em memória → troca atômica
        if (S.state === 'stock' && (i === 2 || i === 3 || (i === 4 && e < 1.6))) {
          const slide = i === 4 ? V.ease.inOut(V.clamp(e / 1.6, 0, 1)) : 0;
          const mw = Math.min(gw, 250), mh = 44;
          const mx0 = gx0, my0 = oy + 44;
          const tx = fx + 12, ty = fy + 26;
          const mx = V.lerp(mx0, tx, slide), my = V.lerp(my0, ty, slide);
          if (my0 + mh < h - 4 || slide > 0) {
            V.box(g, mx, my, mw, mh, { r: 9, fill: V.alpha(C.a, 0.18), stroke: C.a, lw: 1.5 });
            V.text(g, i === 4 ? 'tmp no mesmo dir → os.replace' : 'pós-imagem em memória', mx + 10, my + 17, { size: 11, weight: 600, color: C.ink, maxW: mw - 20 });
            V.text(g, i >= 3 ? 'sha256 = ' + PIN_P + ' ✓' : 'compile() ✓ · +142 B', mx + 10, my + 34, { size: 11, font: 'mono', color: C.ink, maxW: mw - 20 });
          }
        }
      });
    })();

    /* ============ F7 · k, bloco e captura ============ */
    (function () {
      const fig = root.querySelector('#fig-k');
      if (!fig) return;
      autoH(fig, 450, 540);
      const S = { k: 6, n: 6, bk: false, round: true };
      let H = null;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'k', min: 1, max: 10, step: 1, value: 6, fmt: (v) => String(v), onInput: (v) => { S.k = v; upd(); } });
      UI.slider(ctl, { label: 'slots', min: 1, max: 16, step: 1, value: 6, fmt: (v) => String(v), onInput: (v) => { S.n = v; upd(); } });
      UI.toggle(ctl, { label: 'hotfix block-k', value: false, onChange: (v) => { S.bk = v; upd(); } });
      UI.toggle(ctl, { label: 'arredondar a captura para múltiplo de 8', value: true, onChange: (v) => { S.round = v; upd(); } });
      const rR = UI.readout(ctl, 'linhas/passo', '—'), rC = UI.readout(ctl, 'capture-size', '—');
      function calc() {
        const k = S.k, n = S.n, R = n * (k + 1);
        const passed = S.round ? Math.ceil(R / 8) * 8 : R;
        const eff = Math.floor(passed / 8) * 8;
        const sizes = []; for (let s = k + 1; s <= eff; s += k + 1) sizes.push(s);
        const captured = R <= eff;
        let verdict, vc;
        if (!S.bk) {
          if (k < 5) { verdict = '✕ recusado: k < 5 trunca o bloco treinado (5)'; vc = 'red'; }
          else if (k % 3 !== 0) { verdict = '✕ recusado: k % 3 ≠ 0 (regra de reuso do módulo MTP, n_predict = 3)'; vc = 'red'; }
          else { verdict = k === 6 ? '✓ boota: k=6 é o menor valor que satisfaz as duas regras' : '✓ boota (mas as posições além de 6 aceitam pouco)'; vc = 'a'; }
        } else if (k < 5) { verdict = '⚠ boota, mas k < 5 trunca o bloco treinado (silencioso no 0.25.2)'; vc = 'd'; }
        else { verdict = k === 5 ? '✓ boota: k=5 é exatamente o bloco treinado' : '✓ boota: o block-k isenta o DSpark da divisibilidade'; vc = 'a'; }
        return { k, n, R, passed, eff, sizes, captured, verdict, vc };
      }
      function upd() { const c = calc(); rR.set(String(c.R)); rC.set(String(c.passed)); if (H) H.redraw(); }
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const c = calc(), narrow = w < 600, pad = 16, bw = w - 2 * pad;
        const cR = (k) => (k === 'red' ? C.red || C.c : col(C, k));
        // 1) bloco
        V.text(g, 'BLOCO DSPARK · âncora + ' + c.k + ' rascunhos (bloco treinado: 5)', pad, 18, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const nslot = Math.max(c.k, 5) + 1;
        const cw = Math.min(46, (bw - (nslot - 1) * 4) / nslot), cy = 28, chh = 30;
        for (let j = 0; j < nslot; j++) {
          const x = pad + j * (cw + 4);
          let fill, stroke, lab, dash = false;
          if (j === 0) { fill = V.alpha(C.b, 0.85); stroke = C.b; lab = cw < 40 ? 'â' : 'âncora'; }
          else if (j <= c.k && j <= 5) { fill = V.alpha(C.a, 0.8); stroke = C.a; lab = 'd' + j; }
          else if (j <= c.k) { fill = V.alpha(C.c, 0.75); stroke = C.c; lab = 'd' + j; }
          else { fill = 'rgba(0,0,0,0)'; stroke = C.red || C.c; lab = 'd' + j; dash = true; }
          const pulse = 0.5 + 0.5 * Math.sin(t * 3 - j * 0.6);
          V.rr(g, x, cy, cw, chh, 7);
          g.fillStyle = fill; g.fill();
          if (dash) g.setLineDash([3, 3]);
          g.strokeStyle = stroke; g.lineWidth = 1.3 + (j > 0 && j <= c.k ? pulse * 0.8 : 0); g.stroke(); g.setLineDash([]);
          V.text(g, lab, x + cw / 2, cy + chh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: dash ? C.red || C.c : onFill(C), maxW: cw - 2 });
          if (dash) V.line(g, x + 4, cy + 4, x + cw - 4, cy + chh - 4, { color: V.alpha(C.red || C.c, 0.7), lw: 1.2 });
        }
        const lg = [[C.a, 'dentro do bloco treinado'], [C.c, 'ruído além do treino'], [C.red || C.c, 'truncado (k < 5)']];
        let lx = pad, ly = cy + chh + 16;
        lg.forEach(([cc, s]) => { const wd = tw(g, s, 11) + 26; if (lx + wd > pad + bw) { lx = pad; ly += 16; } V.box(g, lx, ly - 9, 10, 10, { r: 2, fill: cc }); V.text(g, s, lx + 14, ly, { size: 11, color: C.ink }); lx += wd; });
        // regras
        const ry = ly + 20;
        const r1 = c.k >= 5, r2 = c.k <= 3 || c.k % 3 === 0;
        const rule = (x, y, ok, s, na) => { V.text(g, na ? '○' : ok ? '✓' : '✕', x, y, { size: 12, weight: 700, color: na ? C.muted : ok ? C.a : C.red || C.c }); V.text(g, s, x + 16, y, { size: 11.5, color: na ? C.muted : C.ink, maxW: (narrow ? bw : bw / 2) - 24 }); };
        rule(pad, ry, r1, 'k ≥ 5 (não truncar o bloco)');
        const r2s = S.bk ? 'divisibilidade: isenta (block-k)' : 'k ≤ 3 ou k % 3 = 0 (n_predict = 3)';
        if (narrow) rule(pad, ry + 18, r2, r2s, S.bk); else rule(pad + bw / 2, ry, r2, r2s, S.bk);
        const vy = ry + (narrow ? 40 : 24);
        wtext(g, c.verdict, pad, vy, bw, { size: 12.5, weight: 700, color: cR(c.vc), maxLines: 2 });
        // 2) grade de verificação
        const gy = vy + (narrow ? 34 : 26);
        V.text(g, 'LINHAS DE VERIFICAÇÃO · R = ' + c.n + ' × (' + c.k + '+1) = ' + c.R, pad, gy, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const capTop = h - (narrow ? 150 : 104);
        const gh = capTop - (gy + 26) - (narrow ? 22 : 16);
        const labW = 30;
        const gridMaxW = narrow ? bw - labW : bw * 0.55 - labW;
        const cellW = Math.min(30, gridMaxW / (c.k + 1) - 3), cellH = Math.max(4, Math.min(18, gh / c.n - 3));
        const gx0 = pad + labW, gyy = gy + 26;
        const scan = (t * 2.2) % (c.k + 2);
        if (cellW >= 16) for (let j = 0; j <= c.k; j++) V.text(g, j === 0 ? 'â' : 'd' + j, gx0 + j * (cellW + 3) + cellW / 2, gyy - 5, { size: 11, font: 'mono', align: 'center', color: C.muted });
        for (let r = 0; r < c.n; r++) {
          if (cellH >= 10) V.text(g, 'r' + (r + 1), pad + labW - 6, gyy + r * (cellH + 3) + cellH / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.muted });
          for (let j = 0; j <= c.k; j++) {
            const x = gx0 + j * (cellW + 3), y = gyy + r * (cellH + 3);
            const on = Math.abs(j - scan) < 0.6;
            V.box(g, x, y, cellW, cellH, { r: 3, fill: j === 0 ? V.alpha(C.b, on ? 1 : 0.7) : V.alpha(C.a, on ? 0.95 : 0.45) });
          }
        }
        const gridW = labW + (c.k + 1) * (cellW + 3);
        const warn = c.passed > 64;
        if (!narrow) {
          const tx = pad + gridW + 24, tw2 = pad + bw - tx;
          const hh = wtext(g, 'Cada requisição (linha r) vira 1 + k tokens no mesmo forward do alvo: a âncora (último token aceito) e os k rascunhos a verificar. A onda mostra a verificação varrendo as posições.', tx, gyy + 6, tw2, { size: 11.5, color: C.ink, maxLines: 5, lh: 15 });
          if (warn) wtext(g, '⚠ acima de 64 linhas nenhuma configuração foi estabelecida como limpa (#141); na ponta branda, streams truncam sem finish_reason.', tx, gyy + 16 + hh, tw2, { size: 11.5, color: C.d, maxLines: 4, lh: 15 });
        } else {
          // no celular o texto vai para baixo da grade, se houver espaço livre
          const gBot = gyy + c.n * (cellH + 3) + 16, room = capTop - (warn ? 20 : 4) - gBot;
          if (room >= 30) wtext(g, 'Cada linha r é uma requisição: a âncora e os k rascunhos entram no mesmo forward do alvo.', pad, gBot + 4, bw, { size: 11.5, color: C.ink, maxLines: Math.min(3, Math.floor(room / 15)), lh: 15 });
          if (warn) V.text(g, '⚠ > 64 linhas: nada estabelecido como limpo (#141)', pad, capTop - 2, { size: 11, color: C.d, maxW: bw });
        }
        // 3) captura
        const ay = h - (narrow ? 74 : 58), ax0 = pad + 8, ax1 = pad + bw - 8;
        const maxV = Math.max(c.passed, c.R, 48) * 1.12;
        const X = (v) => ax0 + (v / maxV) * (ax1 - ax0);
        wtext(g, 'CUDA GRAPHS · tamanhos capturados: múltiplos de ' + (c.k + 1) + ' até ' + c.eff + (S.round ? '' : ' (cru ' + c.passed + ' → truncado a ' + c.eff + ')'), pad, ay - (narrow ? 58 : 44), bw, { size: 11, font: 'mono', color: C.muted, maxLines: narrow ? 2 : 1, lh: 14 });
        V.line(g, ax0, ay, ax1, ay, { color: V.alpha(C.ink, 0.5), lw: 1.4 });
        V.box(g, ax0, ay - 5, Math.max(0, X(c.eff) - ax0), 10, { r: 5, fill: V.alpha(C.a, 0.18) });
        c.sizes.forEach((s) => V.line(g, X(s), ay - 7, X(s), ay + 7, { color: C.a, lw: 2 }));
        const every = c.sizes.length > 8 ? 2 : 1;
        c.sizes.forEach((s, idx) => { if ((idx % every === 0 || s === c.R) && Math.abs(X(s) - X(c.eff)) > 26) V.text(g, String(s), X(s), ay + 20, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        V.line(g, X(c.eff), ay - 10, X(c.eff), ay + 10, { color: C.ink, lw: 2 });
        V.text(g, 'cap ' + c.eff, X(c.eff), ay + 20, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.ink });
        const mcol = c.captured ? C.a : C.red || C.c;
        const bump = Math.sin(t * 4) * 2;
        V.arrow(g, X(c.R), ay - 28 + bump, X(c.R), ay - 9, { color: mcol, lw: 2, head: 7 });
        const lblx = V.clamp(X(c.R), pad + 70, pad + bw - 70);
        V.text(g, 'passo com ' + c.n + ' slots = ' + c.R, lblx, ay - 31 + bump, { size: 11, weight: 600, align: 'center', color: mcol });
        wtext(g, c.captured ? '✓ ' + c.R + ' ≤ ' + c.eff + ': o passo cheio roda em CUDA graph FULL' : '✕ ' + c.R + ' > ' + c.eff + ': o passo cheio roda eager (medido em 6×6: 122 vs 139 tok/s, −12 %)', pad, h - (narrow ? 30 : 12), bw, { size: 12, weight: 700, color: mcol, maxLines: narrow ? 2 : 1, lh: 16 });
      });
      upd();
    })();

    /* ============ F8 · linha de 584 B e o #22 ============ */
    (function () {
      const fig = root.querySelector('#fig-kv584');
      if (!fig) return;
      const S = { view: 0, fix: true };
      // altura por vista: no celular a vista de páginas precisa de mais espaço vertical
      const stage = fig.querySelector('.la-fig-stage');
      const setH = () => {
        if (!stage) return;
        const fw = fig.clientWidth || window.innerWidth;
        const want = (fw < 600 ? (S.view === 1 ? 610 : 540) : 410) + 'px';
        if (stage.style.height !== want) stage.style.height = want;
      };
      setH();
      window.addEventListener('resize', setH);
      LA.onCleanup(() => window.removeEventListener('resize', setH));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Vista', options: [{ v: 0, label: 'linha de 584 B' }, { v: 1, label: 'páginas do alocador' }], value: 0, onChange: (v) => { S.view = v; setH(); } });
      UI.toggle(ctl, { label: 'hotfix #22 aplicado', value: true, onChange: (v) => { S.fix = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 16, bw = w - 2 * pad, narrow = w < 600;
        const dispH = narrow ? 150 : 110;
        const blk = Math.floor(t * 1.2) % 7;
        let contentBot = 0;
        if (S.view === 0) {
          V.text(g, 'LATENTE DE UM TOKEN · 512 dims = 448 NoPE + 64 RoPE (K = V)', pad, 18, { size: 11, font: 'mono', color: C.muted, maxW: bw });
          const n = 32, cw = (bw - (n - 1) * 2) / n;
          for (let j = 0; j < n; j++) {
            const nope = j < 28, x = pad + j * (cw + 2);
            const inBlk = nope && Math.floor(j / 4) === blk;
            const a = inBlk ? 1 : 0.4 + 0.3 * Math.abs(Math.sin(j * 1.7 + t * 1.3));
            V.box(g, x, 26, cw, 18, { r: 3, fill: V.alpha(nope ? C.b : C.c, a) });
          }
          const yb = narrow ? 104 : 94, bh = 34;
          V.text(g, narrow ? '584 B POR TOKEN · 576 dados + 8 escalas' : '584 B POR TOKEN · 576 B de dados + 8 B de escalas (igual em fp8_ds_mla e nvfp4_ds_mla)', pad, yb - 12, { size: 11, font: 'mono', color: C.muted, maxW: bw });
          // os 576 B de dados em escala; as escalas vêm depois de um vão: são contabilidade por token,
          // mas moram no fim da página (faixa de baixo)
          const gapS = 8, dw = bw - gapS - (narrow ? 26 : 34);
          const X = (b) => pad + (b / 576) * dw;
          const sx0 = pad + dw + gapS, sx1 = pad + bw;
          const lx0 = pad + blk * 4 * (cw + 2), lx1 = lx0 + 4 * (cw + 2) - 2;
          const bx0 = X(blk * 64), bx1 = X((blk + 1) * 64);
          g.fillStyle = V.alpha(C.a, 0.12);
          g.beginPath(); g.moveTo(lx0, 45); g.lineTo(lx1, 45); g.lineTo(bx1 - 1, yb); g.lineTo(bx0 + 1, yb); g.closePath(); g.fill();
          for (let k = 0; k < 7; k++) {
            const x0 = X(k * 64), x1 = X((k + 1) * 64);
            V.box(g, x0 + 0.5, yb, x1 - x0 - 1, bh, { r: 4, fill: V.alpha(C.a, k === blk ? 0.95 : 0.6) });
            if (x1 - x0 > 40) V.text(g, '64×e4m3', (x0 + x1) / 2, yb + bh / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: onFill(C), maxW: x1 - x0 - 4 });
          }
          V.box(g, X(448) + 0.5, yb, X(576) - X(448) - 1, bh, { r: 4, fill: V.alpha(C.c, 0.75) });
          V.text(g, '64×bf16', (X(448) + X(576)) / 2, yb + bh / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: onFill(C), maxW: X(576) - X(448) - 4 });
          g.save(); g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.d, 0.8); g.lineWidth = 1; V.rr(g, sx0 - 2, yb - 2, sx1 - sx0 + 4, bh + 4, 5); g.stroke(); g.restore();
          for (let k = 0; k < 8; k++) V.box(g, sx0 + ((sx1 - sx0) / 8) * k, yb, Math.max(1, (sx1 - sx0) / 8 - 0.6), bh, { r: 1, fill: k === 7 ? V.alpha(C.muted, 0.4) : V.alpha(C.d, k === blk ? 1 : 0.7) });
          // chaves de tamanho
          const ky = yb + bh + 18;
          const tag = (x0, x1, s) => { V.line(g, x0 + 1, ky - 8, x1 - 1, ky - 8, { color: V.alpha(C.ink, 0.5), lw: 1 }); V.text(g, s, (x0 + x1) / 2, ky + 6, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: Math.max(40, x1 - x0) }); };
          tag(X(0), X(448), narrow ? '448 B · NoPE FP8' : '448 B · NoPE FP8 e4m3 (7 blocos de 64 dims)');
          tag(X(448), X(576), narrow ? '128 B' : '128 B · RoPE bf16');
          // lupa sobre as escalas do token
          const zw = narrow ? bw : Math.min(300, bw * 0.46), zx = pad + bw - zw, zy = ky + 22, zh = 50;
          g.save(); g.strokeStyle = V.alpha(C.d, 0.8); g.lineWidth = 1.2; g.setLineDash([3, 3]);
          const zl = narrow ? zx + zw - 46 : zx;
          g.beginPath(); g.moveTo(sx0, yb + bh); g.lineTo(zl, zy); g.moveTo(sx1, yb + bh); g.lineTo(zx + zw, zy); g.stroke(); g.restore();
          V.box(g, zx, zy, zw, zh, { r: 9, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, 0.8), lw: 1.2 });
          V.text(g, narrow ? 'escalas do token (8 B): 7 UE8M0 + 1 pad' : 'escalas do token (8 B): 7 UE8M0, uma por bloco, + 1 pad', zx + 10, zy + 15, { size: 11, color: C.ink, maxW: zw - 20 });
          const cw2 = (zw - 20 - 7 * 4) / 8;
          for (let k = 0; k < 8; k++) {
            const x = zx + 10 + k * (cw2 + 4), on = k === blk;
            V.box(g, x, zy + 23, cw2, 20, { r: 4, fill: k === 7 ? V.alpha(C.muted, 0.25) : V.alpha(C.d, on ? 1 : 0.45) });
            V.text(g, k === 7 ? 'pad' : 's' + k, x + cw2 / 2, zy + 33.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: on ? onFill(C) : C.ink, maxW: cw2 - 2 });
          }
          // contabilidade física
          const ax = pad, aw = narrow ? bw : bw - zw - 18, ay = narrow ? zy + zh + 20 : zy + 12;
          const ah = wtext(g, 'Por token, em contexto longo: 21 camadas C4 × 584/4 + 20 C128 × 584/128 + indexer 21 × 132/4 ≈ 3,85 KB (+ ~3,4 MB fixos de janelas SWA por sequência).', ax, ay, aw, { size: 11.5, color: C.ink, maxLines: 3, lh: 15 });
          // a página física: 64 × 576 B de dados primeiro, 64 × 8 B de escalas no fim
          const pgY = Math.max(ay + ah + 16, zy + zh + 22), pgH = 16;
          V.text(g, narrow ? 'A PÁGINA (64 tokens) · 37.376 B' : 'A PÁGINA (64 tokens, código de referência cache_utils.py) · 37.376 B', pad, pgY - 5, { size: 11, font: 'mono', color: C.muted, maxW: bw });
          const sw = narrow ? 34 : 46, pdw = bw - sw - 6, tok = Math.floor(t * 2.5) % 64;
          for (let k = 0; k < 64; k++) V.box(g, pad + (k * pdw) / 64, pgY + 8, Math.max(1, pdw / 64 - 0.8), pgH, { r: 1, fill: V.alpha(k === tok ? C.a : C.b, k === tok ? 1 : 0.45) });
          const px0 = pad + pdw + 6;
          V.box(g, px0, pgY + 8, sw, pgH, { r: 3, fill: V.alpha(C.d, 0.35), stroke: V.alpha(C.d, 0.9), lw: 1 });
          V.box(g, px0 + 2 + ((sw - 4) * tok) / 64, pgY + 9, Math.max(1.5, (sw - 4) / 64), pgH - 2, { r: 0, fill: C.d });
          const tx = pad + ((tok + 0.5) * pdw) / 64;
          g.save(); g.strokeStyle = V.alpha(C.d, 0.85); g.lineWidth = 1.2; g.setLineDash([2, 3]);
          g.beginPath(); g.moveTo(tx, pgY + 8); g.quadraticCurveTo((tx + px0 + sw / 2) / 2, pgY, px0 + sw / 2, pgY + 8); g.stroke(); g.restore();
          V.text(g, narrow ? '64 × 576 B de dados' : '64 × 576 B de dados (NoPE + RoPE de cada token)', pad + pdw / 2, pgY + pgH + 22, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: pdw });
          V.text(g, narrow ? '64×8 B' : '64 × 8 B', px0 + sw / 2, pgY + pgH + 22, { size: 11, font: 'mono', align: narrow ? 'right' : 'center', color: C.d, maxW: sw + 20 });
          contentBot = pgY + pgH + 26;
        } else {
          V.text(g, 'O ALOCADOR HÍBRIDO EXIGE PÁGINAS DE MESMO TAMANHO EM BYTES · 37.376 B', pad, 18, { size: 11, font: 'mono', color: C.muted, maxW: bw });
          const P = [
            { t: 'MLA · C4', s: 'bloco 256 tokens ÷ 4 = 64 linhas', f: '64 linhas × 584 B = 37.376 B', known: true, c: C.a },
            { t: 'SWA do alvo', s: '64 tokens por página', f: '64 tokens × 584 B = 37.376 B', known: true, c: C.b },
            { t: 'SWA do drafter', s: 'bloco 4 · mesmo tamanho', f: '37.376 B por página', c: C.c },
            { t: 'SWA do drafter', s: 'bloco 8 · mesmo tamanho', f: '37.376 B por página', c: C.f },
          ];
          const cols = narrow ? 2 : 4, gx = 10;
          const pw = (bw - (cols - 1) * gx) / cols, ph = narrow ? 156 : 166;
          P.forEach((p, i) => {
            const x = pad + (i % cols) * (pw + gx), y = 30 + Math.floor(i / cols) * (ph + 10);
            V.box(g, x, y, pw, ph, { r: 10, fill: V.alpha(p.c, 0.08), stroke: V.alpha(p.c, 0.7), lw: 1.3 });
            V.text(g, p.t, x + 10, y + 18, { size: 12, weight: 700, color: p.c, maxW: pw - 16 });
            V.text(g, p.s, x + 10, y + 34, { size: 11, color: C.ink, maxW: pw - 16 });
            // página = grade de 64 linhas + faixa das escalas no fim (layout conhecido) ou "?" (não documentado)
            const sy = y + ph - 48, shh = 13;
            const gy = y + 42, gh = sy - 3 - gy, gw = pw - 20;
            const rc = 8, cc2 = 8, cwd = gw / cc2 - 1.5, chh = gh / rc - 1.5;
            const lit = Math.floor(t * 8 + i * 13) % 64;
            for (let r = 0; r < rc; r++) for (let q = 0; q < cc2; q++) {
              const k = r * cc2 + q;
              V.box(g, x + 10 + q * (cwd + 1.5), gy + r * (chh + 1.5), cwd, chh, { r: 1.5, fill: V.alpha(p.c, k === lit ? 1 : 0.28 + (0.2 * ((k * 7) % 3)) / 2) });
            }
            V.box(g, x + 10, sy, gw - 1.5, shh, { r: 3, fill: p.known ? V.alpha(C.d, 0.8) : V.alpha(C.muted, 0.18), stroke: p.known ? null : V.alpha(C.muted, 0.6), lw: 1 });
            V.text(g, p.known ? 'escalas: 64 × 8 B' : 'arranjo não documentado', x + 10 + gw / 2, sy + shh / 2 + 0.5, { size: 11, weight: 600, align: 'center', base: 'middle', color: p.known ? onFill(C) : C.muted, maxW: gw - 6 });
            wtext(g, p.f, x + 10, y + ph - 22, pw - 18, { size: 11, font: 'mono', color: C.ink, maxLines: 2, lh: 13 });
          });
          const y = narrow ? 30 + 2 * (ph + 10) + 10 : 30 + ph + 24;
          const th = wtext(g, 'O Stage B tentou uma linha de 416 B: 64 × 416 = 26.624 B ≠ 37.376 B, e as linhas SWA (escritas por um op C++) não mudam de tamanho. O Stage C voltou ao envelope de 584 B (interpretação da justificativa do repositório).', pad, y, bw, { size: 11.5, color: C.ink, maxLines: narrow ? 5 : 2, lh: 15 });
          contentBot = y + th - 6;
        }
        // despacho: nunca antes do fim do conteúdo de cima
        const dy = Math.max(h - dispH, contentBot + 16);
        V.line(g, pad, dy - 8, pad + bw, dy - 8, { color: V.alpha(C.muted, 0.35), lw: 1, dash: [2, 4] });
        V.text(g, 'DESPACHO NO flashmla_sparse.py', pad, dy + 8, { size: 11, font: 'mono', color: C.muted });
        const cond = S.fix ? 'kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' : 'kv_cache_dtype == "fp8_ds_mla"';
        chip(g, pad, dy + 30, cond, S.fix ? C.a : C.d, { size: 11, font: 'mono', maxW: bw });
        const fastY = dy + (narrow ? 64 : 62), slowY = dy + (narrow ? 100 : 92);
        const bx0 = narrow ? pad + 20 : pad + bw * 0.52, bx1 = pad + bw;
        const b1 = { x: bx0, y: fastY - 13, w: bx1 - bx0, h: 26 }, b2 = { x: bx0, y: slowY - 13, w: bx1 - bx0, h: 26 };
        const useFast = S.fix;
        V.box(g, b1.x, b1.y, b1.w, b1.h, { r: 8, fill: V.alpha(C.a, useFast ? 0.22 : 0.06), stroke: V.alpha(C.a, useFast ? 1 : 0.35), lw: 1.4 });
        V.text(g, '_forward_fp8_kv · rápido · ~17,3 tok/s a 600K+', b1.x + 10, b1.y + 17, { size: 11, font: 'mono', color: useFast ? C.ink : C.muted, maxW: b1.w - 16 });
        V.box(g, b2.x, b2.y, b2.w, b2.h, { r: 8, fill: V.alpha(C.red || C.c, !useFast ? 0.22 : 0.06), stroke: V.alpha(C.red || C.c, !useFast ? 1 : 0.35), lw: 1.4 });
        V.text(g, '_forward_bf16_kv · lento · ~1,0 tok/s a 600K+', b2.x + 10, b2.y + 17, { size: 11, font: 'mono', color: !useFast ? C.ink : C.muted, maxW: b2.w - 16 });
        const tgt = useFast ? b1 : b2;
        if (!narrow) {
          const sx = pad + Math.min(bw * 0.5, tw(g, cond, 11, 'mono', 500) + 16) - 4, sy = dy + 30;
          V.arrow(g, sx, sy + 8, tgt.x - 4, tgt.y + 13, { color: useFast ? C.a : C.red || C.c, lw: 2, curve: useFast ? -10 : 10 });
          const u = (t * 0.7) % 1, p = V.along(sx, sy + 8, tgt.x - 4, tgt.y + 13, u);
          V.dot(g, p[0], p[1], 3.5, useFast ? C.a : C.red || C.c);
        } else {
          const ax = pad + 8;
          V.line(g, ax, dy + 40, ax, tgt.y + 13, { color: useFast ? C.a : C.red || C.c, lw: 2 });
          V.arrow(g, ax, tgt.y + 13, tgt.x - 3, tgt.y + 13, { color: useFast ? C.a : C.red || C.c, lw: 2, head: 7 });
          const u = (t * 0.7) % 1;
          V.dot(g, ax, V.lerp(dy + 40, tgt.y + 13, u), 3.5, useFast ? C.a : C.red || C.c);
        }
        wtext(g, S.fix ? 'mesmos 584 B, kernel certo: ≈16× no decode a 600K+' : 'o nome “nvfp4” falha no ==: bytes FP8 lidos pelo caminho bf16', pad, h - 8, narrow ? bw : bw * 0.5, { size: 11.5, weight: 600, color: S.fix ? C.a : C.red || C.c, maxLines: 1 });
      });
    })();

    /* ============ F9 · perfis de serviço ============ */
    (function () {
      const fig = root.querySelector('#fig-perfis');
      if (!fig) return;
      autoH(fig, 330, 350);
      const PROF = [{ n: 'Anemll 1M/6', s: '2 nós · TP=2 · imagem Anemll', c: 'a' }, { n: 'TP=3 · 16 slots', s: '3 nós · Anemll + pad', c: 'b' }, { n: 'Stage-C 200K/16', s: '2 nós · imagem própria', c: 'c' }];
      const MET = [
        { n: 'agregado (tok/s)', short: 'agregado', u: 'tok/s', max: 340, v: [[139, 191.2, '139 (02/set) – 191 (01/ago, 0731) · README: 160–190'], [198, 198, '≈198 a 16 streams (02/set)'], [205, 315.1, '205 escalonado – 315 estático (C16)']] },
        // decode c=1: só o par medido no mesmo workload (prosa, 512 tokens, 02/set; docs/TP3.md)
        { n: 'decode c=1 (tok/s) · prosa, 512 tokens, mesmo dia', short: 'decode c=1', u: 'tok/s', max: 60, v: [[38.6, 38.6, '38,6 (sondas curtas de set/ago, outra carga: 52–83)'], [41.9, 41.9, '41,9 (+9 % no mesmo workload)'], { note: 'sem par: 57,6 em C1 (200K/16) com outra carga' }] },
        { n: 'TTFT a 128K (s) · menor é melhor', short: 'TTFT 128K', u: 's', max: 110, v: [[74.9, 80.4, '74,9 (par do mesmo dia) – 80,4 s (outras datas)'], [91.5, 91.5, '91,5 s (+22 % sobre o par)'], null] },
        { n: 'pool de KV (milhões de tokens)', short: 'pool de KV', u: 'M', max: 5.6, v: [[2.19, 2.33, '2,19–2,33 M (≈2,1–2,2× a 1M)'], [5.02, 5.02, '5,02 M (4,79× a 1M)'], null] },
        { n: 'teto de contexto por requisição', short: 'contexto', u: 'tok', max: 1.15, v: [[1.048576, 1.048576, '1.048.576'], [1.048576, 1.048576, '1.048.576'], [0.2, 0.2, '200.000']] },
      ];
      const S = { m: 0, disp: PROF.map(() => [0, 0]) };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Métrica', options: MET.map((m, i) => ({ v: i, label: m.short })), value: 0, onChange: (v) => { S.m = v; } });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const m = MET[S.m], pad = 16, narrow = w < 600;
        const labW = narrow ? 0 : 170;
        const x0 = pad + labW, x1 = w - pad - (narrow ? 4 : 10);
        V.text(g, m.n.toUpperCase(), pad, 18, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const rowH = (h - 60) / 3;
        const X = (v) => x0 + (v / m.max) * (x1 - x0);
        for (let q = 0; q <= 4; q++) {
          const x = x0 + ((x1 - x0) * q) / 4;
          V.line(g, x, 30, x, h - 24, { color: V.alpha(C.muted, 0.18), lw: 1 });
          const val = (m.max * q) / 4;
          V.text(g, m.u === 'tok' ? (val >= 1 ? br(val, 2) + 'M' : Math.round(val * 1000) + 'K') : br(val, m.max < 10 ? 1 : 0), x, h - 8, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        PROF.forEach((p, i) => {
          const cc = col(C, p.c);
          const y = 34 + i * rowH;
          const raw = m.v[i], d = Array.isArray(raw) ? raw : null, note = raw && raw.note;
          const tgt = d ? [d[0], d[1]] : [0, 0];
          const cur = S.disp[i];
          const k = Math.min(1, dt * 7);
          cur[0] += (tgt[0] - cur[0]) * k; cur[1] += (tgt[1] - cur[1]) * k;
          if (narrow) {
            V.text(g, p.n, pad, y + 12, { size: 12, weight: 700, color: cc });
            V.text(g, p.s, pad + tw(g, p.n, 12, 'sans', 700) + 8, y + 12, { size: 11, color: C.muted, maxW: w - 2 * pad - 110 });
          } else {
            V.text(g, p.n, pad, y + rowH / 2 - 8, { size: 12.5, weight: 700, color: cc, maxW: labW - 8 });
            V.text(g, p.s, pad, y + rowH / 2 + 8, { size: 11, color: C.muted, maxW: labW - 8 });
          }
          const by = narrow ? y + 20 : y + rowH / 2 - 16, bh = narrow ? 18 : 20;
          if (!d) {
            V.box(g, x0, by, x1 - x0, bh, { r: 6, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 });
            g.save(); V.rr(g, x0, by, x1 - x0, bh, 6); g.clip();
            for (let xx = x0 - 30; xx < x1; xx += 10) V.line(g, xx, by + bh, xx + bh, by, { color: V.alpha(C.muted, 0.3), lw: 1 });
            g.restore();
            V.text(g, note || 'não medido nesta forma', x0 + 10, by + bh / 2 + 4, { size: 11, color: C.muted, maxW: x1 - x0 - 20 });
            return;
          }
          const xa = X(cur[0]), xb = Math.max(X(cur[1]), xa + 6);
          V.box(g, x0, by, Math.max(2, xa - x0), bh, { r: 6, fill: V.alpha(cc, 0.25) });
          V.box(g, xa, by, xb - xa, bh, { r: 6, fill: V.alpha(cc, 0.9) });
          V.glow(g, xb, by + bh / 2, 12, cc, 0.5 + 0.3 * Math.sin(t * 3 + i));
          wtext(g, d[2], x0, by + bh + 15, x1 - x0, { size: 11, color: C.ink, maxLines: 1 });
        });
      });
    })();

    /* ============ F10 · prever antes de medir ============ */
    (function () {
      const fig = root.querySelector('#fig-custo');
      if (!fig) return;
      autoH(fig, 420, 580);
      const BYTES = [['experts roteados', 12, 'a', '12 GB'], ['atenção FP8', 2.4, 'b', '2,4 GB'], ['drafter (3 camadas)', 0.9, 'c', '0,9 GB'], ['lm_head bf16', 1.06, 'd', '1,06 GB'], ['Markov w2', 0.2, 'f', '0,2 GB']];
      const BT = BYTES.reduce((s, b) => s + b[1], 0);
      // A aceitação é cumulativa: a posição i só vale se todas as anteriores valeram, então o perfil
      // nunca cresce. Onde só a média ā foi publicada, o formato é ilustrativo: a_i = α^i com α tal que
      // Σ a_i = ā·k (a previsão usa ā·k, que não depende do formato).
      const geo = (mean, k = 6) => {
        let lo = 0.01, hi = 0.999;
        for (let it = 0; it < 50; it++) { const a = (lo + hi) / 2; let s = 0, p = 1; for (let i = 0; i < k; i++) { p *= a; s += p; } if (s > mean * k) hi = a; else lo = a; }
        const a = (lo + hi) / 2, out = []; let p = 1; for (let i = 0; i < k; i++) { p *= a; out.push(p); } return out;
      };
      const DS = [
        { n: 'código · dual-HCA', acc: 0.733, prof: geo(0.733), pl: 'média publicada 73,3 %; formato por posição ilustrativo (αⁱ)', meas: 76.0, mt: 'medido: 76,0 tok/s (aceitação 73,3 %)' },
        // prosa: pos0 e pos3–5 publicados (A/B de 02/set); pos1–2 interpolados para fechar a média de 25,7 %
        { n: 'prosa · dual-HCA', acc: 0.257, prof: [0.68, 0.40, 0.23, 0.13, 0.07, 0.03], pl: 'd1 e d4–d6 publicados (A/B 02/set); d2–d3 interpolados até a média 25,7 %', meas: 37.3, mt: 'medido: 37,3 tok/s (aceitação 25,7 %)' },
        { n: 'auditoria 16/ago', acc: 0.715, prof: geo(0.715), pl: 'média publicada ~71,5 %; formato por posição ilustrativo (αⁱ)', meas: 74.5, mt: 'esperado: C1 73–76 tok/s (aceitação ~68–75 %)' },
        { n: 'curva por posição', pos: [0.89, 0.73, 0.49, 0.34, 0.23, 0.15], pl: 'curva por posição medida (bench_quick, Vision-Exp k=6)', meas: null, mt: 'sem medida pareada · c=1 típico em set: 52–57 tok/s' },
      ];
      const S = { d: 0, bw: 245, col: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Aceitação', options: DS.map((d, i) => ({ v: i, label: d.n })), value: 0, onChange: (v) => { S.d = v; } });
      UI.slider(ctl, { label: 'banda efetiva β', min: 200, max: 273, step: 1, value: S.bw, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; } });
      UI.toggle(ctl, { label: 'somar ≈108 coletivas × 30 µs', value: true, onChange: (v) => { S.col = v; } });
      const rT = UI.readout(ctl, 't_passo', '—'), rP = UI.readout(ctl, 'previsto', '—');
      let lastR = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const d = DS[S.d], narrow = w < 600, pad = 16;
        const tB = (BT / S.bw) * 1000, tC = S.col ? 108 * 0.03 : 0, ts = tB + tC;
        const acc = d.pos ? d.pos.reduce((a, b) => a + b, 0) : d.acc * 6;
        const tok = 1 + acc, pred = tok / (ts / 1000);
        const key = ts.toFixed(2) + pred.toFixed(2);
        if (key !== lastR) { rT.set(br(ts, 1) + ' ms'); rP.set(br(pred, 1) + ' tok/s'); lastR = key; }
        // esquerda: bytes
        const lw = narrow ? w - 2 * pad : w * 0.42;
        V.text(g, 'BYTES POR RANK POR PASSO · ≈' + br(BT, 2) + ' GB', pad, 18, { size: 11, font: 'mono', color: C.muted, maxW: lw });
        const bx = pad, by = 28, bh = narrow ? 30 : 34;
        let xx = bx;
        const pulse = (t * 0.6) % 1;
        BYTES.forEach((b) => {
          const sw = (b[1] / BT) * lw;
          V.box(g, xx + 0.5, by, Math.max(1, sw - 1), bh, { r: 5, fill: V.alpha(col(C, b[2]), 0.85) });
          if (sw > 70) V.text(g, b[3], xx + 8, by + bh / 2 + 4, { size: 11, font: 'mono', weight: 600, color: onFill(C) });
          xx += sw;
        });
        V.line(g, bx + pulse * lw, by - 3, bx + pulse * lw, by + bh + 3, { color: C.ink, lw: 1.5 });
        let ly = by + bh + 18;
        BYTES.forEach((b, i) => {
          const cx = bx + (i % 2) * (lw / 2), cy = ly + Math.floor(i / 2) * 17;
          V.box(g, cx, cy - 9, 10, 10, { r: 2, fill: col(C, b[2]) });
          V.text(g, b[0] + ' ' + b[3], cx + 14, cy, { size: 11, color: C.ink, maxW: lw / 2 - 16 });
        });
        ly += 3 * 17 + 8;
        V.text(g, '÷ β = ' + br(tB, 1) + ' ms' + (S.col ? '  + coletivas ' + br(tC, 1) + ' ms' : ''), bx, ly, { size: 12, font: 'mono', color: C.ink, maxW: lw });
        V.text(g, '= t_passo ≈ ' + br(ts, 1) + ' ms  (' + br(1000 / ts, 1) + ' passos/s)', bx, ly + 18, { size: 12, font: 'mono', weight: 700, color: C.ink, maxW: lw });
        // direita: tokens por passo
        const rx = narrow ? pad : w * 0.5, rw = narrow ? w - 2 * pad : w * 0.5 - pad;
        const ry = narrow ? ly + 46 : 18;
        V.text(g, 'TOKENS ENTREGUES POR PASSO · 1 bônus + aceitos', rx, ry, { size: 11, font: 'mono', color: C.muted, maxW: rw });
        const colsN = 7, bwid = (rw - (colsN - 1) * 6) / colsN, maxH = narrow ? 76 : 104, base = ry + 16 + maxH;
        for (let j = 0; j < colsN; j++) {
          const x = rx + j * (bwid + 6);
          const v = j === 0 ? 1 : (d.pos || d.prof)[j - 1];
          const hh = v * maxH;
          V.box(g, x, base - maxH, bwid, maxH, { r: 5, fill: V.alpha(C.muted, 0.08) });
          const wob = j === 0 ? 0 : Math.sin(t * 2 + j) * 1.2;
          V.box(g, x, base - hh + wob, bwid, Math.max(2, hh - wob), { r: 5, fill: V.alpha(j === 0 ? C.b : C.a, 0.85) });
          V.text(g, j === 0 ? 'bônus' : 'd' + j, x + bwid / 2, base + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
          if (bwid > 26) V.text(g, j === 0 ? '1' : br(v, 2), x + bwid / 2, base - hh - 5 + wob, { size: 11, font: 'mono', align: 'center', color: C.ink });
        }
        const plh = wtext(g, d.pl, rx, base + 30, rw, { size: 11, color: C.muted, maxLines: 2, lh: 13 });
        V.text(g, '1 + ' + br(acc, 2) + ' = ' + br(tok, 2) + ' tokens/passo', rx, base + 36 + plh, { size: 12.5, font: 'mono', weight: 700, color: C.ink, maxW: rw });
        // fita de 1 segundo de decode
        const oy = h - (narrow ? 58 : 44);
        const fy = Math.max(ly + 40, base + 58 + plh), fh = oy - 14 - fy;
        if (fh > 44) {
          const steps = 1000 / ts, fx0 = pad, fx1 = w - pad, fw = fx1 - fx0;
          const P = 3.2, u = (t % P) / P, tNow = u * 1000;
          V.text(g, 'UM SEGUNDO DE DECODE, EM CÂMERA LENTA · ' + br(steps, 1) + ' passos × ' + br(tok, 2) + ' ≈ ' + Math.round(pred) + ' tokens', fx0, fy + 10, { size: 11, font: 'mono', color: C.muted, maxW: fw });
          const lineY = fy + fh - 10;
          V.line(g, fx0, lineY, fx1, lineY, { color: V.alpha(C.ink, 0.45), lw: 1.3 });
          const colH = fh - 30, cell = Math.max(3, Math.min(9, colH / 7 - 1.5));
          let made = 0;
          for (let sIdx = 0; sIdx < Math.ceil(steps); sIdx++) {
            const tEnd = (sIdx + 1) * ts;
            if (tEnd > 1000) break;
            const x = fx0 + (tEnd / 1000) * fw;
            V.line(g, x, lineY - 3, x, lineY + 3, { color: V.alpha(C.ink, 0.5), lw: 1 });
            if (tEnd <= tNow) {
              const whole = Math.floor(tok), frac = tok - whole;
              for (let q = 0; q < whole; q++) V.box(g, x - cell / 2, lineY - 6 - (q + 1) * (cell + 1.5), cell, cell, { r: 2, fill: q === 0 ? C.b : C.a });
              if (frac > 0.05) V.box(g, x - cell / 2, lineY - 6 - (whole + 1) * (cell + 1.5), cell, cell, { r: 2, fill: V.alpha(C.a, frac) });
              made += tok;
            }
          }
          const px = fx0 + u * fw;
          V.line(g, px, fy + 16, px, lineY + 6, { color: C.d, lw: 1.6 });
          V.glow(g, px, lineY, 10, C.d, 0.8);
          V.text(g, Math.round(made) + ' tokens', Math.min(px + 6, fx1 - 70), fy + 26, { size: 11, font: 'mono', weight: 700, color: C.d });
        }
        // resultado
        const err = d.meas ? ((pred - d.meas) / d.meas) * 100 : null;
        V.box(g, pad, oy - 4, w - 2 * pad, narrow ? 54 : 40, { r: 10, fill: V.alpha(C.a, 0.1), stroke: V.alpha(C.a, 0.7), lw: 1.3 });
        const ptxt = 'previsto ≈ ' + br(pred, 1) + ' tok/s';
        const mtxt = d.mt + (err != null ? ' · desvio ' + (err >= 0 ? '+' : '') + br(err, 0) + ' %' : '');
        V.text(g, ptxt, pad + 12, oy + (narrow ? 16 : 21), { size: 14, weight: 700, color: C.ink });
        if (narrow) V.text(g, mtxt, pad + 12, oy + 38, { size: 11.5, color: C.ink, maxW: w - 2 * pad - 24 });
        else { const pw = tw(g, ptxt, 14, 'sans', 700); V.text(g, '· ' + mtxt, pad + 22 + pw, oy + 21, { size: 11.5, color: C.ink, maxW: w - 2 * pad - pw - 34 }); }
      });
    })();

    /* ============ F11 · DeepSeek-V4.1-Flash: um modelo, cinco implantações ============ */
    (function () {
      const fig = root.querySelector('#fig-ds41');
      if (!fig) return;
      autoH(fig, 480, 860);
      const MEMT = 121.7; // GiB de MemTotal por Spark
      const TX = {
        dep: 'Implantação', view: 'Vista', vMem: 'memória e layout', vPar: 'vazão × velocidade por usuário',
        rN: 'nós', rW: 'pesos por nó',
        topo: 'NÓS E ENLACES', mem: 'MEMÓRIA POR NÓ · 121,7 GiB CADA', exp: 'EXPERTS POR GPU · calculado',
        src: 'fonte: ',
        tag: { m: 'medido', r: 'config.', k: 'calculado', c: '≈ gráfico', x: 'dois valores' },
        px: 'velocidade por usuário (tok/s por stream)', py: 'vazão agregada (tok/s)',
        pxE: 'velocidade por usuário (tok/s por usuário)', pyE: 'vazão de saída (tok/s por GPU)',
        warn: 'As condições diferem entre as curvas (temperatura, thinking, ferramenta, data): compare formatos, não decimais.',
        noteE: 'Decode do DeepInfer em 64 GPUs SM100 (EP64·DP64), contexto 131.072: passos/s medidos × um τ = 3,25 suposto (1 + 3 × 0,75), valores lidos do gráfico publicado. Rótulos: requisições por rank de DP.',
        scaleE: 'C com 16 streams ÷ 4 Sparks ≈ 86 tok/s por GPU (calculado, só para escala)',
        relay: 'sem cabo: salto retransmitido', row: 'parceiros de linha', colp: 'parceiros de coluna',
        sw: 'switch RoCE', ringAlt: 'ou: anel sem switch + NCCL patcheado', pad: 'shard de atenção: só padding',
        pre: 'prefill · 4 GPUs · EP4·DP4', dec: 'decode · 64 GPUs · EP64·DP64', kv: 'repasse do KV', kvNo: 'desagregação: fora do release',
        expP: 'prefill: 104 de 416 experts por GPU (384 + 32 redundantes, EP4)', expD: 'decode: 7 de 448 experts por GPU (384 + 64 redundantes, EP64)', hbm: 'HBM por GPU: não informada no repositório',
        freeNote: '~40 GB livres por rank servindo [medido]: as páginas de KV só são tocadas quando usadas',
      };
      const DEP = [
        { k: 'A · 2× TF EXL3', c: 'a', topo: 'pair', t: '2× DGX Spark · TensorFold nativo (Zig)', sub: 'EXL3 EAQ v1 2,9 bpw (196 GiB) · TP=2 · um cabo QSFP, RoCE v2 nas duas funções PCIe gêmeas',
          nodes: [['rank 0', 'API · head'], ['rank 1', 'worker']], link: 'QSFP · RoCE v2',
          lay: 'TP=2: cada camada cortada em duas; o indexador do prefill divide suas linhas entre os ranks a partir do token 131.072 (prefill frio +4,25 % a 256K, +7,46 % a 400K).',
          rN: '2 Sparks', rW: '~98 GiB',
          mem: { w: 98, kv: 0.45, free: 6.5, freeRows: [0], items: [['w', 'pesos ~98 GiB por Spark', 'm'], ['kv', 'KV ≈0,45 GiB com CONTEXT 524.288 (~0,9 GiB a cada 1M posições)', 'k'], ['free', 'livre no head depois da carga: 6,5 GiB', 'm'], ['rest', 'resto: reserva do motor, runtime, SO (sem detalhamento)', 'k']] },
          nums: [['80,7 tok/s', 'prosa, 1 stream (T=0, thinking off)', 'm'], ['154,0 tok/s', 'prosa, 4 streams, agregado (39,9 cada)', 'm'], ['118,1 tok/s', 'código, 1 stream', 'm'], ['~2.150 tok/s', 'prefill 8K–32K (2.065 a 256K)', 'm']],
          cond: 'sparkDash · GPUs a 2.200 MHz · CONTEXT=524.288 · rascunhos ligados · tabelas Engram em disco local',
          src: 'MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks',
          par: [{ pts: [[1, 80.7, 80.7], [2, 56.0, 110.6], [3, 45.4, 133.6], [4, 39.9, 154.0]], lab: 'A · 2× TF EXL3: prosa, T=0, thinking off, 1–4 streams' }] },
        { k: 'B · 3× SGLang', c: 'b', topo: 'tri', t: '3× DGX Spark · SGLang (dev-dsv41 + adaptadores)', sub: 'experts MXFP4 nativos + densos FP8 · Engram no NVMe · TP=3 · triângulo RoCE',
          nodes: [['rank 0', 'API · head'], ['rank 1', ''], ['rank 2', '']], link: 'RoCE',
          lay: 'TP=3: cabeças 64 → 96 e grupos 8 → 12 (32 cabeças por rank, rank 2 só padding); vocabulário e experts de rascunho também com padding. EP=3 no README, EP_SIZE=1 no .env.example.',
          rN: '3 Sparks', rW: '~101 GiB',
          mem: { w: 101, kv: 1.48, free: 5.6, freeRows: 'all', items: [['w', 'pesos ~101 GiB por rank', 'm'], ['kv', 'KV ≈1,5 GiB: 750.000 × 1.670,75 B + 0,34 GB de janela', 'k'], ['free', '~6 GB livres por rank servindo', 'm'], ['rest', 'resto: ativações, graphs, NCCL, SO (sem detalhamento)', 'k']] },
          nums: [['51,0 ou 37,9', 'tok/s, prosa, 1 stream: o README traz os dois', 'x'], ['85,4 ou 78,6', 'tok/s, prosa, 4 streams, agregado (as mesmas duas tabelas)', 'x'], ['~2.000 tok/s', 'prefill', 'm'], ['262.144', 'contexto configurado; prompts longos limitados pelo head', 'r']],
          cond: 'tabela de abertura × tabela TP3 contra TP4 do mesmo README; a de abertura foi atualizada com a pilha de decode de 24/09/2026',
          src: 'MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks',
          par: [{ pts: [[1, 51.0, 51.0], [2, 39.0, 75.1], [4, 22.8, 85.4]], lab: 'B · 3× SGLang: prosa; contínua = tabela de abertura (51,0 com 1 stream)' }, { pts: [[1, 37.9, 37.9], [4, 20.9, 78.6]], lab: 'B · tracejada = tabela TP3 contra TP4 (37,9): o mesmo README discorda de si mesmo', dash: true }] },
        { k: 'C · 4× SGLang', c: 'c', topo: 'switch', t: '4× DGX Spark · SGLang (branch dsv4.1 + RoCEnante)', sub: 'os mesmos pesos nativos · TP=4, EP=1 · switch RoCE (ou anel com NCCL patcheado)',
          nodes: [['rank 0', 'API · head'], ['rank 1', ''], ['rank 2', ''], ['rank 3', '']], link: 'RoCE',
          lay: 'TP=4: exatamente 16 cabeças e 2 grupos por rank, sem padding; EP_SIZE=1 (cada rank com um quarto de cada um dos 384 experts); prefill sequence parallel.',
          rN: '4 Sparks', rW: '~77 GiB',
          mem: { w: 77, kv: 12.5, free: 0, freeRows: [], items: [['w', 'pesos ~77 GiB por rank', 'm'], ['kv', 'pool de KV pinado em 8M tokens: 13,4 GB por rank', 'r'], ['rest', 'resto: ~24 GiB reservados ao transiente do prefill, runtime, SO', 'k']], note: true },
          nums: [['87,7 tok/s', 'prosa, 1 stream', 'm'], ['342,7 tok/s', 'prosa, 16 streams, agregado (22,2 cada)', 'm'], ['4,9k–5,0k', 'tok/s de prefill frio, texto real, 16k–128k', 'm'], ['1.011.084', 'tokens: a agulha passou', 'm']],
          cond: 'sparkDash 1.8.8 · greedy · 256 tokens · thinking off · malha com switch · 25/09/2026',
          src: 'MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks (docs/tp4.md)',
          par: [{ pts: [[1, 87.7, 87.7], [2, 61.9, 120.4], [4, 41.4, 163.6], [8, 30.8, 237.6], [16, 22.2, 342.7]], lab: 'C · 4× SGLang: prosa, greedy, 256 tokens, thinking off, 1–16 streams' }] },
        { k: 'D · 4× TF nativo', c: 'f', topo: 'ring', t: '4× DGX Spark · TensorFold nativo (Zig)', sub: 'FP8/MXFP4 da DeepSeek como publicado (~510 GB) · TP=4 como grade 2×2 · anel sem switch',
          nodes: [['spark1', 'g0 · API'], ['spark2', 'g1'], ['spark3', 'g3'], ['spark4', 'g2']], link: 'anel',
          lay: 'TP=4 como grade 2×2: nó g = 2p + r, parceiro de linha g^1, parceiro de coluna g^2; a ordem 0-1-3-2 no anel dá cabo próprio a todo par de parceiros.',
          rN: '4 Sparks', rW: '~80 GiB',
          mem: { w: 80, kv: 0, free: 0, freeRows: [], items: [['w', 'pesos ~80 GiB por nó', 'm'], ['rest', 'KV (pool de 2.097.152 posições), runtime e memória livre: sem detalhamento na receita', 'k']] },
          nums: [['65,8 tok/s', 'prosa, 1 requisição (T=0,7, thinking on)', 'm'], ['165,3 tok/s', 'prosa, 8 requisições, agregado (21,7 cada)', 'm'], ['117,4 tok/s', 'código, 1 requisição, greedy', 'm'], ['45/45', 'células da agulha, até 997.530 tokens', 'm']],
          cond: 'anel sem switch · 25/10/2026 · padrões (janela de 1M, 16 streams, rascunhos, CUDA graphs) · mediana de 3 execuções',
          src: 'MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold',
          par: [{ pts: [[1, 65.8, 65.8], [2, 48.9, 96.5], [4, 34.1, 130.5], [8, 21.7, 165.3]], lab: 'D · 4× TF nativo: prosa, T=0,7, thinking on, 1–8 requisições' }] },
        { k: 'E · DeepInfer', c: 'd', topo: 'di', t: 'DeepInfer · o motor de produção da DeepSeek', sub: 'GPUs NVIDIA SM100 (ou Ascend 950) · atenção DP + MoE EP + EPLB · prefill e decode medidos à parte',
          nodes: [], link: '',
          lay: 'Prefill: EP4·DP4, 384 + 32 experts → 104 por GPU. Decode: EP64·DP64, 384 + 64 experts → 7 por GPU (calculado). Cada benchmark dentro de um domínio scale-up.',
          rN: '4 + 64 GPUs', rW: 'não informado',
          nums: [['≈26.300', 'tok/s por GPU, decode com 256 requisições por rank (≈103 por usuário)', 'c'], ['≈325', 'tok/s por usuário com 8 requisições por rank (≈2.600 por GPU)', 'c'], ['≈89.500', 'tok/s de prefill por GPU, 65.536 tokens por rank (≈732 ms por passo)', 'c'], ['τ = 3,25', 'tokens por passo supostos: 1 + 3 × 0,75 (não medido)', 'k']],
          cond: 'decode: 64 GPUs SM100, contexto 131.072, nextn=3 · prefill: 4 GPUs, replay limitado de 140 tokens',
          src: 'deepseek-ai/DeepInfer (README, doc/images, examples/sweep)',
          par: [{ pts: [[8, 325, 2600], [16, 315, 5050], [32, 297, 9500], [64, 246, 15750], [128, 179, 22800], [256, 103, 26300]], lab: '' }] },
      ];
      const S = { d: 0, v: 'mem', a: 0 };
      const ctl = UI.controls(fig);
      const upd = () => { const d = DEP[S.d]; rN.set(d.rN); rW.set(d.rW); };
      UI.seg(ctl, { label: TX.dep, options: DEP.map((d, i) => ({ v: i, label: d.k })), value: 0, onChange: (v) => { S.d = +v; S.a = 0; upd(); } });
      UI.seg(ctl, { label: TX.view, options: [{ v: 'mem', label: TX.vMem }, { v: 'par', label: TX.vPar }], value: 'mem', onChange: (v) => { S.v = v; S.a = 0; } });
      const rN = UI.readout(ctl, TX.rN, '—'), rW = UI.readout(ctl, TX.rW, '—');
      upd();
      const segCol = (C, k, cc) => (k === 'w' ? cc : k === 'kv' ? C.d : k === 'free' ? C.e : V.alpha(C.muted, 0.5));
      const tagCol = (C, k) => (k === 'm' ? C.e : k === 'k' ? C.d : k === 'c' ? C.b : k === 'x' ? C.red : C.muted);
      const tag = (g, x, y, k, C, al) => chip(g, x, y, TX.tag[k], tagCol(C, k), { size: 11, align: al || 'left', fa: 0.12, sa: 0.65 });
      function hatch(g, x, y, w, h, col) {
        if (w < 1) return;
        g.save(); V.rr(g, x, y, w, h, 3); g.clip();
        g.fillStyle = V.alpha(col, 0.16); g.fillRect(x, y, w, h);
        for (let xx = x - h; xx < x + w; xx += 5) V.line(g, xx, y + h, xx + h, y, { color: V.alpha(col, 0.85), lw: 1.1 });
        g.restore();
        V.box(g, x, y, w, h, { r: 3, stroke: col, lw: 1 });
      }
      function nodeBox(g, cx, cy, bw, bh, nd, cc, C, stroke) {
        V.box(g, cx - bw / 2, cy - bh / 2, bw, bh, { r: 9, fill: C.bg || '#000' });
        V.box(g, cx - bw / 2, cy - bh / 2, bw, bh, { r: 9, fill: V.alpha(cc, 0.16), stroke: stroke || V.alpha(cc, 0.85), lw: 1.4 });
        V.text(g, nd[0], cx, cy + (nd[1] ? -2 : 4), { size: 12, weight: 700, align: 'center', color: C.ink, maxW: bw - 8 });
        if (nd[1]) V.text(g, nd[1], cx, cy + 13, { size: 11, align: 'center', color: C.muted, maxW: bw - 8 });
      }
      function link(g, a, b, col, t, k, o = {}) {
        V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(col, o.al || 0.75), lw: o.lw || 2, dash: o.dash });
        if (o.dash) return;
        for (let q = 0; q < 2; q++) {
          const u = (t * 0.42 + q * 0.5 + k * 0.17) % 1;
          const p = q ? V.along(b[0], b[1], a[0], a[1], u) : V.along(a[0], a[1], b[0], b[1], u);
          V.dot(g, p[0], p[1], 3, col);
        }
      }
      function drawTopo(g, x, y, w, h, t, C, d, cc) {
        V.text(g, TX.topo, x, y + 10, { size: 11, font: 'mono', color: C.muted, maxW: w });
        const top = y + 22, H = h - 22, cx = x + w / 2;
        const bw = Math.min(112, w * 0.34), bh = 36;
        if (d.topo === 'pair') {
          const p = [[x + bw / 2 + 2, top + H / 2], [x + w - bw / 2 - 2, top + H / 2]];
          link(g, p[0], p[1], cc, t, 0, { lw: 2.5 });
          V.text(g, d.link, cx, top + H / 2 - 9, { size: 11, align: 'center', color: C.ink, maxW: w - 2 * bw });
          p.forEach((pp, i) => nodeBox(g, pp[0], pp[1], bw, bh, d.nodes[i], cc, C));
        } else if (d.topo === 'tri') {
          const p = [[cx, top + bh / 2 + 2], [x + bw / 2 + 2, top + H - bh / 2 - 20], [x + w - bw / 2 - 2, top + H - bh / 2 - 20]];
          link(g, p[0], p[1], cc, t, 0); link(g, p[1], p[2], cc, t, 1); link(g, p[0], p[2], cc, t, 2);
          p.forEach((pp, i) => nodeBox(g, pp[0], pp[1], bw, bh, d.nodes[i], cc, C, i === 2 ? C.red : null));
          const pw2 = Math.min(w / 2 + 20, tw(g, TX.pad, 11, 'sans', 600));
          V.text(g, TX.pad, Math.min(p[2][0], x + w - pw2 / 2), p[2][1] + bh / 2 + 15, { size: 11, weight: 600, align: 'center', color: C.red, maxW: pw2 });
        } else if (d.topo === 'switch' || d.topo === 'ring') {
          const xl = x + bw / 2 + 2, xr = x + w - bw / 2 - 2, yt = top + bh / 2 + 2, yb = top + H - bh / 2 - 22;
          const P = [[xl, yt], [xr, yt], [xr, yb], [xl, yb]];
          if (d.topo === 'switch') {
            for (let i = 0; i < 4; i++) V.line(g, P[i][0], P[i][1], P[(i + 1) % 4][0], P[(i + 1) % 4][1], { color: V.alpha(C.muted, 0.4), lw: 1.2, dash: [4, 5] });
            const sc = [cx, (yt + yb) / 2];
            P.forEach((pp, i) => link(g, pp, sc, cc, t, i));
            V.box(g, sc[0] - 46, sc[1] - 13, 92, 26, { r: 8, fill: V.alpha(C.bg || '#000', 0.9) });
            V.box(g, sc[0] - 46, sc[1] - 13, 92, 26, { r: 8, fill: V.alpha(C.ink, 0.08), stroke: V.alpha(C.ink, 0.55), lw: 1.2 });
            V.text(g, TX.sw, sc[0], sc[1] + 4, { size: 11, weight: 600, align: 'center', color: C.ink, maxW: 86 });
            V.text(g, TX.ringAlt, cx, top + H - 2, { size: 11, align: 'center', color: C.muted, maxW: w });
          } else {
            V.line(g, P[0][0], P[0][1], P[2][0], P[2][1], { color: V.alpha(C.muted, 0.55), lw: 1.2, dash: [3, 5] });
            V.line(g, P[1][0], P[1][1], P[3][0], P[3][1], { color: V.alpha(C.muted, 0.55), lw: 1.2, dash: [3, 5] });
            link(g, P[0], P[1], cc, t, 0, { lw: 2.4 }); link(g, P[3], P[2], cc, t, 1, { lw: 2.4 });
            link(g, P[0], P[3], C.b, t, 2, { lw: 2.4 }); link(g, P[1], P[2], C.b, t, 3, { lw: 2.4 });
            const mw = tw(g, TX.relay, 11, 'sans', 400) + 14;
            V.box(g, cx - mw / 2, (yt + yb) / 2 - 10, mw, 20, { r: 10, fill: V.alpha(C.bg || '#000', 0.92), stroke: V.alpha(C.muted, 0.5), lw: 1 });
            V.text(g, TX.relay, cx, (yt + yb) / 2 + 4, { size: 11, align: 'center', color: C.muted, maxW: w - 2 * bw });
            const ly = top + H - 2;
            const w1 = tw(g, TX.row, 11) + 34, w2 = tw(g, TX.colp, 11) + 34, lx = cx - (w1 + w2) / 2;
            V.line(g, lx, ly - 4, lx + 22, ly - 4, { color: cc, lw: 2.4 }); V.text(g, TX.row, lx + 27, ly, { size: 11, color: C.ink });
            V.line(g, lx + w1, ly - 4, lx + w1 + 22, ly - 4, { color: C.b, lw: 2.4 }); V.text(g, TX.colp, lx + w1 + 27, ly, { size: 11, color: C.ink });
          }
          P.forEach((pp, i) => nodeBox(g, pp[0], pp[1], bw, bh, d.nodes[i], cc, C));
        } else {
          const s = Math.max(6, Math.min(12, (H - 50) / 8 - 2)), side = 8 * (s + 2);
          const dx = x + w - side - 4, dy = top + 4;
          const ps = s * 1.9, pside = 2 * (ps + 3), px = x + 4, py = dy + (side - pside) / 2;
          for (let i = 0; i < 4; i++) {
            const gx = px + (i % 2) * (ps + 3), gy = py + Math.floor(i / 2) * (ps + 3);
            V.box(g, gx, gy, ps, ps, { r: 3, fill: V.alpha(cc, 0.35 + 0.35 * (0.5 + 0.5 * Math.sin(t * 2.4 + i))), stroke: cc, lw: 1 });
          }
          for (let i = 0; i < 64; i++) {
            const gx = dx + (i % 8) * (s + 2), gy = dy + Math.floor(i / 8) * (s + 2);
            V.box(g, gx, gy, s, s, { r: 2, fill: V.alpha(cc, 0.25 + 0.45 * (0.5 + 0.5 * Math.sin(t * 2 - (i % 8) * 0.5 - Math.floor(i / 8) * 0.3))) });
          }
          const ax0 = px + pside + 8, ax1 = dx - 8, ay = py + pside / 2;
          V.arrow(g, ax0, ay, ax1, ay, { color: V.alpha(C.ink, 0.6), lw: 1.6 });
          const u = (t * 0.5) % 1;
          V.dot(g, V.lerp(ax0, ax1 - 8, u), ay, 3.5, cc);
          V.text(g, TX.kv, (ax0 + ax1) / 2, ay - 8, { size: 11, weight: 600, align: 'center', color: C.ink, maxW: ax1 - ax0 });
          wtext(g, TX.kvNo, (ax0 + ax1) / 2, ay + 18, Math.max(60, ax1 - ax0 - 4), { size: 11, align: 'center', color: C.muted, maxLines: 3, lh: 13 });
          const by = dy + side + 16;
          wtext(g, TX.pre, x + 4, by, Math.max(80, (w - side) / 2), { size: 11, color: C.ink, maxLines: 2, lh: 13 });
          wtext(g, TX.dec, dx, by, Math.max(80, side), { size: 11, color: C.ink, maxLines: 2, lh: 13 });
        }
      }
      function drawMem(g, x, y, w, t, C, d, cc, e) {
        if (d.topo === 'di') {
          V.text(g, TX.exp, x, y + 10, { size: 11, font: 'mono', color: C.muted, maxW: w });
          let yy = y + 26;
          [[TX.expP, 104 / 416], [TX.expD, 7 / 448]].forEach(([s, f]) => {
            const hh = wtext(g, s, x, yy + 10, w, { size: 11, color: C.ink, maxLines: 2, lh: 13 });
            yy += hh + 2;
            V.box(g, x, yy, w, 14, { r: 4, fill: V.alpha(C.muted, 0.14) });
            V.box(g, x, yy, Math.max(3, w * f * e), 14, { r: 4, fill: V.alpha(cc, 0.9) });
            yy += 24;
          });
          wtext(g, TX.hbm, x, yy + 8, w, { size: 11, color: C.muted, maxLines: 2, lh: 13 });
          return;
        }
        V.text(g, TX.mem, x, y + 10, { size: 11, font: 'mono', color: C.muted, maxW: w });
        const m = d.mem, n = d.nodes.length, labW = 56, bx0 = x + labW, bx1 = x + w - 2, bw = bx1 - bx0;
        const X = (v) => bx0 + (v / MEMT) * bw;
        let yy = y + 20;
        const rowH = 21, bh = 14;
        for (let i = 0; i < n; i++) {
          V.text(g, d.nodes[i][0], x, yy + 11, { size: 11, font: 'mono', color: C.ink, maxW: labW - 6 });
          V.box(g, bx0, yy, bw, bh, { r: 4, fill: V.alpha(C.muted, 0.14) });
          const wv = m.w * e, kv = m.kv * e;
          V.box(g, bx0, yy, Math.max(2, X(wv) - bx0), bh, { r: 4, fill: V.alpha(cc, 0.88) });
          if (kv > 0) V.box(g, X(wv), yy, Math.max(2.5, X(wv + kv) - X(wv)), bh, { r: 2, fill: C.d });
          const fr = m.free > 0 && (m.freeRows === 'all' || m.freeRows.indexOf(i) >= 0);
          if (fr) hatch(g, X(MEMT - m.free * e), yy, X(MEMT) - X(MEMT - m.free * e), bh, C.e);
          if (bw > 150) V.text(g, br(m.w, 0), bx0 + 6, yy + 11, { size: 11, font: 'mono', weight: 700, color: onFill(C) });
          yy += rowH;
        }
        [0, 30, 60, 90, 121.7].forEach((v) => {
          const xx = X(v);
          V.line(g, xx, yy - 4, xx, yy, { color: V.alpha(C.muted, 0.6), lw: 1 });
          V.text(g, v === 121.7 ? br(v, 1) : String(v), xx, yy + 11, { size: 11, font: 'mono', align: v === 0 ? 'left' : v === 121.7 ? 'right' : 'center', color: C.muted });
        });
        yy += 32;
        m.items.forEach((it) => {
          const col2 = segCol(C, it[0], cc);
          if (it[0] === 'free') hatch(g, x, yy - 9, 12, 11, C.e); else V.box(g, x, yy - 9, 12, 11, { r: 2, fill: col2 });
          const tg = TX.tag[it[2]], tgw = tw(g, tg, 11, 'sans', 500) + 16;
          const hh = wtext(g, it[1], x + 18, yy, w - 24 - tgw, { size: 11, color: C.ink, maxLines: 2, lh: 13 });
          tag(g, x + w - tgw, yy - 4, it[2], C);
          yy += Math.max(hh, 13) + 6;
        });
        if (m.note) wtext(g, TX.freeNote, x, yy + 2, w, { size: 11, color: C.e, maxLines: 2, lh: 13 });
      }
      function drawBottom(g, x, y, w, h, C, d, cc, narrow) {
        let yy = y;
        yy += wtext(g, d.lay, x, yy + 10, w, { size: 11.5, color: C.ink, maxLines: narrow ? 4 : 2, lh: 15 }) + 6;
        const cols = narrow ? 2 : 4, gap = 8, cw = (w - (cols - 1) * gap) / cols, ch = narrow ? 100 : 96;
        d.nums.forEach((nm, i) => {
          const cx0 = x + (i % cols) * (cw + gap), cy0 = yy + Math.floor(i / cols) * (ch + gap);
          V.box(g, cx0, cy0, cw, ch, { r: 10, fill: V.alpha(cc, 0.08), stroke: V.alpha(nm[2] === 'x' ? C.red : cc, 0.55), lw: 1.2 });
          V.text(g, nm[0], cx0 + 10, cy0 + 22, { size: 15, weight: 700, color: C.ink, maxW: cw - 20 });
          wtext(g, nm[1], cx0 + 10, cy0 + 40, cw - 20, { size: 11, color: C.muted, maxLines: 3, lh: 13 });
          tag(g, cx0 + cw - 8, cy0 + ch - 14, nm[2], C, 'right');
        });
        yy += Math.ceil(d.nums.length / cols) * (ch + gap) + 4;
        yy += wtext(g, d.cond, x, yy + 10, w, { size: 11, color: C.muted, maxLines: 3, lh: 14 });
        if (narrow) { V.text(g, TX.src, x, yy + 14, { size: 11, font: 'mono', color: C.muted }); V.text(g, d.src, x, yy + 28, { size: 11, font: 'mono', color: C.muted, maxW: w }); }
        else V.text(g, TX.src + d.src, x, yy + 14, { size: 11, font: 'mono', color: C.muted, maxW: w });
      }
      function drawPar(g, w, h, t, C, d, cc, e, y0, narrow) {
        const pad = 16, isE = d.topo === 'di';
        const xm = isE ? 350 : 100, ym = isE ? 30000 : 360;
        const xt = isE ? [0, 100, 200, 300] : [0, 25, 50, 75, 100], yt = isE ? [0, 10000, 20000, 30000] : [0, 90, 180, 270, 360];
        const x0 = pad + (isE ? 46 : 34), x1 = w - pad - 8, top = y0 + 22;
        const legendH = narrow ? (isE ? 120 : 190) : (isE ? 70 : 104);
        const y1 = h - legendH - 34;
        const X = (v) => x0 + (v / xm) * (x1 - x0), Y = (v) => y1 - (v / ym) * (y1 - top);
        V.text(g, isE ? TX.pyE : TX.py, pad, top - 8, { size: 11, color: C.ink, maxW: w - 2 * pad });
        xt.forEach((v) => { V.line(g, X(v), top, X(v), y1, { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, fmtN(v), X(v), y1 + 14, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        yt.forEach((v) => { V.line(g, x0, Y(v), x1, Y(v), { color: V.alpha(C.muted, 0.18), lw: 1 }); V.text(g, fmtN(v), x0 - 6, Y(v) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        V.text(g, isE ? TX.pxE : TX.px, (x0 + x1) / 2, y1 + 30, { size: 11, align: 'center', color: C.ink, maxW: w - 2 * pad });
        const drawS = (sr, col, sel) => {
          const a = sel ? 1 : 0.3;
          g.beginPath();
          sr.pts.forEach((p, i) => { const xx = X(p[1]), yy = Y(p[2]); if (i) g.lineTo(xx, yy); else g.moveTo(xx, yy); });
          g.strokeStyle = V.alpha(col, a); g.lineWidth = sel ? 2.4 : 1.4;
          if (sr.dash) g.setLineDash([6, 5]);
          g.stroke(); g.setLineDash([]);
          sr.pts.forEach((p) => V.dot(g, X(p[1]), Y(p[2]), sel ? 4.5 * (0.6 + 0.4 * e) : 3, V.alpha(col, a)));
          if (sel) sr.pts.forEach((p) => V.text(g, (isE ? '' : 'c') + p[0], X(p[1]) + 6, Y(p[2]) + (sr.dash ? 15 : -7), { size: 11, font: 'mono', weight: 600, color: col }));
        };
        if (isE) {
          drawS(d.par[0], cc, true);
          const sx = X(22.2), sy = Y(342.7 / 4);
          V.dot(g, sx, sy, 4, C.c);
          const sw = x1 - sx - 10, nL = Math.min(3, lines(g, TX.scaleE, sw, 11).length);
          wtext(g, TX.scaleE, sx + 8, sy - 6 - 13 * (nL - 1), sw, { size: 11, color: C.c, maxLines: 3, lh: 13 });
        } else {
          DEP.forEach((dd, i) => { if (dd.topo !== 'di' && i !== S.d) dd.par.forEach((sr) => drawS(sr, col(C, dd.c), false)); });
          d.par.forEach((sr) => drawS(sr, cc, true));
        }
        const pts = d.par[0].pts, u = (t * 0.18) % 1, seg = u * (pts.length - 1), k = Math.min(pts.length - 2, Math.floor(seg)), f = seg - k;
        V.glow(g, V.lerp(X(pts[k][1]), X(pts[k + 1][1]), f), V.lerp(Y(pts[k][2]), Y(pts[k + 1][2]), f), 12, cc, 0.8);
        let ly = y1 + 50;
        if (isE) {
          wtext(g, TX.noteE, pad, ly, w - 2 * pad, { size: 11, color: C.ink, maxLines: 5, lh: 14 });
          return;
        }
        DEP.forEach((dd) => {
          if (dd.topo === 'di') return;
          dd.par.forEach((sr) => {
            const cl = col(C, dd.c), on = dd === d;
            if (sr.dash) V.line(g, pad, ly - 4, pad + 14, ly - 4, { color: cl, lw: 2, dash: [4, 3] }); else V.box(g, pad, ly - 9, 14, 10, { r: 2, fill: V.alpha(cl, on ? 1 : 0.45) });
            ly += wtext(g, sr.lab, pad + 20, ly, w - 2 * pad - 20, { size: 11, color: on ? C.ink : C.muted, weight: on ? 600 : 400, maxLines: 2, lh: 13 }) + 3;
          });
        });
        wtext(g, TX.warn, pad, ly + 4, w - 2 * pad, { size: 11, color: C.d, maxLines: 2, lh: 13 });
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        S.a = Math.min(1, S.a + (dt || 0.016) * 2.2);
        const e = V.ease.out(S.a);
        const d = DEP[S.d], cc = col(C, d.c), pad = 16, narrow = w < 600;
        V.text(g, d.t, pad, 20, { size: 14, weight: 700, color: cc, maxW: w - 2 * pad });
        const hy = 38 + wtext(g, d.sub, pad, 38, w - 2 * pad, { size: 11.5, color: C.muted, maxLines: 2, lh: 15 }) - 6;
        if (S.v === 'par') { drawPar(g, w, h, t, C, d, cc, e, hy, narrow); return; }
        if (narrow) {
          const th = d.topo === 'di' ? 190 : 170;
          drawTopo(g, pad, hy, w - 2 * pad, th, t, C, d, cc);
          const my = hy + th + 10;
          drawMem(g, pad, my, w - 2 * pad, t, C, d, cc, e);
          const mh = d.topo === 'di' ? 158 : 20 + d.nodes.length * 21 + 32 + d.mem.items.length * 26 + (d.mem.note ? 30 : 0);
          drawBottom(g, pad, my + mh + 8, w - 2 * pad, h, C, d, cc, true);
        } else {
          const th = 210, lw = w * 0.44;
          drawTopo(g, pad, hy, lw - pad, th, t, C, d, cc);
          drawMem(g, w * 0.5, hy, w * 0.5 - pad, t, C, d, cc, e);
          drawBottom(g, pad, hy + th + 12, w - 2 * pad, h, C, d, cc, false);
        }
      });
    })();
  },
});

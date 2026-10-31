LA.chapter({
  id: 'vllm-arquitetura',
  part: 6,
  order: 30,
  title: 'Por dentro do vLLM',
  navTitle: 'Arquitetura do vLLM',
  subtitle: 'Processos, filas e prazos; o model runner que traduz o plano em tensores; os registros e plugins onde modelos, formatos de quantização e kernels se encaixam; e as imagens, lanes e variáveis que decidem qual vLLM você está rodando de fato.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['vLLM V1', 'EngineCore', 'executor mp', 'model runner', 'model registry', 'quantização plugável', 'plugins', 'workspace', 'envs.py', 'overlays', 'TensorFold'],
  goals: [
    'Desenhar o mapa de processos do vLLM V1 (API server, EngineCore, executor, workers, model runner) e seguir uma requisição por ele',
    'Explicar como os processos conversam (ZMQ, fila em shared memory, espera ativa) e quais prazos vigiam o passo',
    'Ler um checkpoint como o vLLM lê: <code>model_type</code>, <code>architectures</code>, registry, aliases e construção no meta device',
    'Plugar quantização e kernels: <code>QuantizationConfig</code>, os três ganchos, plugins, MoE modular, workspace manager e backends de atenção',
    'Identificar lane, imagem e variáveis de uma receita, e escolher entre overlay de arquivo e patch pontual',
  ],
  prereqs: ['cuda-graphs-warmup', 'tres-modelos'],
  summary: [
    'Um motor transforma um laço de geração num serviço: juntar sequências por passo divide o custo fixo de ler os pesos (\\(\\text{vazão} = B/(t_{\\text{pesos}} + B\\,t_{\\text{seq}})\\)).',
    'O vLLM V1 isola em processos o que compete por CPU: API server (+AsyncLLM), EngineCore (scheduler, KV manager, executor) e um worker por GPU. Nas receitas são \\(1+1+2 = 4\\) processos, com o rank 1 <code>--headless</code> e o executor <code>mp</code> fixado. A versão TensorFold do GLM usa dois, um por Spark: o rank 0 serve a API e manda uma mensagem de operações por iteração, e o rank 1 a aplica.',
    'O plano de cada passo vai aos workers por uma fila em shared memory, e a resposta volta ao EngineCore por outra. Os leitores dos dois lados giram por <code>busy_loop_s</code> antes de dormir: DeepSeek 2 ms (sempre); GLM opcional 16 ms (padrão stock; perfil long-coding: +0,95% de decode, −85,3% de CPU do EngineCore). O passo é uma RPC com prazo: 1800 s em GLM e DeepSeek, sem estender o watchdog de 600 s.',
    'O model runner achata o lote e monta <code>input_ids</code>, posições, <code>query_start_loc</code>, block table e slot mapping. O V1 prepara na CPU; o V2 (padrão desde a v0.29) mantém o estado na GPU. Patches de runner não servem para o outro.',
    'O registry mapeia <code>architectures</code> → (módulo, classe) com import preguiçoso, e a classe de config resolve aliases (sem eles, o GLM rotearia 7 experts em vez de 8). O meta device constrói o modelo sem memória e materializa só as subárvores necessárias.',
    'Quantização é plugável: <code>QuantizationConfig.get_quant_method</code> escolhe o método por camada, e cada método implementa <code>create_weights</code> → <code>process_weights_after_loading</code> → <code>apply</code>. Registrar o nome não basta, é preciso vencer a lista de overrides. Plugins entram por <code>vllm.general_plugins</code>.',
    'A imagem é parte da receita: string de versão, runner, envs registradas (<code>VLLM_*</code> desconhecida é ignorada) e caminhos do modelo mudam por lane. Overlays de arquivo congelam versões; patches por âncora falham alto.',
  ],
  quiz: [
    { q: 'Por que o EngineCore do vLLM V1 roda num processo separado do API server?', o: ['Para usar outra GPU', 'Para que tokenização, destokenização, parsers e JSON (trabalho de CPU em Python) não bloqueiem o laço que agenda e alimenta a GPU', 'Porque o FastAPI não suporta CUDA', 'Para dispensar o KV cache no API server'], a: 1, e: 'Em Python, CPU ocupada segura o interpretador. Separar o front-end do busy loop <code>schedule → execute_model → update_from_output</code> mantém a GPU alimentada. Por isso a doc pede ≥ 2 + N núcleos físicos.' },
    { q: 'A lane TP1 do Qwen usa uma única GPU, mas força <code>--distributed-executor-backend mp</code>. Por quê?', o: ['Porque só o <code>multiproc_executor</code> chama <code>spawn_ple_offload()</code>; sem ele o processo de offload da PLE nunca nasce e o worker fica em deadlock silencioso', 'Porque <code>uni</code> não suporta NVFP4', 'Porque <code>mp</code> é mais rápido em TP1', 'Porque o <code>uni</code> exige Ray'], a: 0, e: 'A escolha do executor define quais processos auxiliares existem. Com uma GPU a autosseleção daria <code>uni</code>, e o GPU worker ficaria esperando para sempre por um processo que ninguém criou.' },
    { q: 'O GLM mediu <code>GLM53_SPINWAIT_MS=16</code> com +0,95% de decode e −85,3% de CPU do EngineCore. O que explica a queda de CPU?', o: ['O decode ficou 85% mais curto', 'Com passos de ~100 ms, a janela stock de 1 s nunca deixava o leitor dormir; com 16 ms ele gira ≈ 16% do tempo e dorme o resto', 'O NCCL passou a usar menos CPU', 'O scheduler foi desligado'], a: 1, e: 'CPU girando ≈ \\(\\min(W,\\Delta)/\\Delta\\). Com \\(\\Delta\\) ≈ 100 ms: \\(W\\) = 1 s → 100%; \\(W\\) = 16 ms → ≈ 16%. O EngineCore entra nessa conta porque também é leitor: espera a resposta de cada passo na fila de resposta. Com 2 ms o decode perdeu 1,68% (interpretação: o leitor passa a dormir entre mensagens próximas e paga o despertar no caminho crítico).' },
    { q: 'Na lane A da DeepSeek (imagem Anemll), alguém define <code>VLLM_DSPARK_GPU_<wbr>REJECTED_<wbr>CONTEXT_<wbr>MASK=1</code>. O que acontece?', o: ['Liga a máscara na GPU', 'O vLLM recusa subir', 'Aparece “Unknown vLLM environment variable” e a variável é ignorada, porque ela só está registrada no <code>envs.py</code> do overlay Stage-C (lane B)', 'A variável vale só no rank 0'], a: 2, e: 'Variáveis <code>VLLM_*</code> fora do <code>vllm/envs.py</code> da imagem geram só um aviso e não fazem nada. A lição: procure a variável no <code>envs.py</code> da imagem que você roda, ou procure o aviso no log.' },
    { q: 'Além de registrar <code>exl3</code> no <code>Literal</code> de métodos, o GLM precisou do <code>patch_model_overrides.py</code>. Para quê?', o: ['Para compilar os kernels EXL3', 'Para desligar o FP8', 'Para pôr <code>exl3</code> na lista ordenada de overrides do <code>ModelConfig</code>, único lugar onde <code>override_quantization_method</code> é consultado; sem isso <code>--quantization exl3</code> é rejeitado contra o <code>quantization_config</code> do checkpoint', 'Para mudar o <code>model_type</code>'], a: 2, e: 'Registrar o nome torna a flag válida. Ganhar o checkpoint exige estar na lista de overrides, e na frente de métodos como <code>auto_gptq</code>.' },
    { q: 'Por que um kernel reserva seus buffers de rascunho no dummy run, antes de <code>lock_workspace()</code>?', o: ['Para economizar energia', 'Porque, depois de travado, o workspace não pode crescer, e alocar durante a captura de CUDA graph é proibido: o graph grava endereços. O dummy run nas formas de pior caso dimensiona o pico', 'Porque o dummy run é mais rápido', 'Para que o KV cache use o mesmo buffer'], a: 1, e: 'Por isso o split-K do SM120 (e, no caminho FlashMLA, o backport #50298) pede seus buffers via <code>get_simultaneous</code> no caminho de dummy/captura. Esse rascunho sai do orçamento de KV.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const q = (s) => root.querySelector(s);
    const nf = (x, d = 0) => Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const ink = (C) => (C.dark ? '#06120c' : '#ffffff');
    const tfCache = {};
    const tfCol = (C) => { if (tfCache.k !== C) { tfCache.k = C; tfCache.v = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfCache.v; }; // cor do selo GLM·TF

    /* ------------------------------------------------------------ helpers */
    function autoH(fig, desk, mob, bp) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => {
        const want = ((st.clientWidth || 800) < (bp || 600) ? mob : desk) + 'px';
        if (st.style.height !== want) st.style.height = want;
      };
      set();
      const ro = new ResizeObserver(set);
      ro.observe(st);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const fcache = {};
    function setFont(g, size, font, weight) { const k = size + '|' + (font || 'sans') + '|' + (weight || 400); g.font = fcache[k] || (fcache[k] = V.font(size, font || 'sans', weight || 400)); }
    function tw(g, s, size, font, weight) { setFont(g, size, font, weight); return g.measureText(String(s)).width; }
    function wrap(g, s, x, y, maxW, lh, o) {
      o = o || {};
      const size = o.size || 12;
      setFont(g, size, o.font, o.weight);
      const words = String(s).split(' ');
      const lines = [];
      let line = '';
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { lines.push(line); line = wd; } else line = test;
      }
      if (line) lines.push(line);
      const maxL = o.maxLines || 99;
      const out = lines.slice(0, maxL);
      out.forEach((L, i) => V.text(g, i === maxL - 1 && lines.length > maxL ? L + '…' : L, x, y + i * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align, base: o.base, maxW }));
      return out.length;
    }
    function critter(g, x, y, r, col, t, mood) {
      if (!(r > 0.8)) return;
      const C = V.C();
      if (t != null && mood !== 'still') y += Math.sin(t * 4 + x * 0.07) * r * 0.08;
      V.dot(g, x, y, r, col);
      g.strokeStyle = V.alpha(C.ink, 0.22); g.lineWidth = 1;
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke();
      const ex = r * 0.36, ey = -r * 0.12, er = Math.max(1.1, r * 0.19);
      const eye = ink(C);
      if (mood === 'sleep') {
        g.strokeStyle = eye; g.lineWidth = Math.max(1, r * 0.12);
        g.beginPath(); g.moveTo(x - ex - er, y + ey); g.lineTo(x - ex + er, y + ey); g.moveTo(x + ex - er, y + ey); g.lineTo(x + ex + er, y + ey); g.stroke();
      } else {
        V.dot(g, x - ex, y + ey, er, eye); V.dot(g, x + ex, y + ey, er, eye);
      }
    }
    function pill(g, x, y, s, col, o) {
      o = o || {};
      const size = o.size || 11, font = o.font || 'mono', weight = o.weight || 500;
      const wd = tw(g, s, size, font, weight) + 14, ht = size + 9;
      const X = o.align === 'left' ? x : o.align === 'right' ? x - wd : x - wd / 2;
      V.box(g, X, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.85), lw: 1 });
      V.text(g, s, X + wd / 2, y + 0.5, { size, font, align: 'center', base: 'middle', color: o.color || V.C().ink, weight });
      return wd;
    }
    function boxEdge(A, bx, by) {
      const dx = bx - A.x, dy = by - A.y;
      const sx = A.w / 2 / Math.max(1e-6, Math.abs(dx)), sy = A.h / 2 / Math.max(1e-6, Math.abs(dy));
      const s = Math.min(sx, sy);
      return [A.x + dx * s, A.y + dy * s];
    }
    function station(g, B, title, sub, col, on, t) {
      const C = V.C();
      if (on) V.glow(g, B.x, B.y, Math.max(B.w, B.h) * 0.55, col, 0.16);
      V.box(g, B.x - B.w / 2, B.y - B.h / 2, B.w, B.h, { r: 11, fill: V.alpha(col, on ? 0.22 : 0.09), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2 : 1.2 });
      V.text(g, title, B.x, B.y - (sub ? B.h * 0.16 : 0), { size: 13, weight: 650, align: 'center', base: 'middle', color: C.ink, maxW: B.w - 10 });
      if (sub) wrap(g, sub, B.x, B.y + B.h * 0.14, B.w - 12, 12, { size: 10, align: 'center', color: C.muted, maxLines: 2 });
    }
    const hash = (k, s) => { const x = Math.sin(k * 12.9898 + s * 78.233) * 43758.5453; return x - Math.floor(x); };

    /* ================================================================
       Figura 1 — laço ingênuo × motor
       ================================================================ */
    (function figMotor() {
      const fig = q('#fig-motor');
      autoH(fig, 360, 480, 560);
      const ctl = UI.controls(fig);
      const S = { mode: 'engine', lam: 3 };
      const TW = 60, TS = 3, SLOW = 0.25;
      const rng = V.rng(7);
      let sim;
      const reset = () => { sim = { t: 0, next: 150, queue: [], active: [], running: false, t0: 0, dur: 0, parts: [], done: [], hist: [], id: 0, served: 0 }; };
      reset();
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'naive', label: 'laço ingênuo' }, { v: 'engine', label: 'motor (8 vagas)' }], value: 'engine', onChange: (v) => { S.mode = v; reset(); prewarm(); } });
      UI.slider(ctl, { label: 'Chegadas', min: 0.5, max: 6, step: 0.5, value: 3, fmt: (v) => nf(v, 1) + ' req/s', onInput: (v) => { S.lam = v; } });
      const rT = UI.readout(ctl, 'tokens/s', '—'), rQ = UI.readout(ctl, 'na fila', '—'), rR = UI.readout(ctl, 'tokens por leitura', '—');
      const cap = () => (S.mode === 'naive' ? 1 : 8);
      function admit() { while (sim.active.length < cap() && sim.queue.length) sim.active.push(sim.queue.shift()); }
      function start(at) { admit(); if (sim.active.length) { sim.running = true; sim.t0 = at; sim.dur = TW + TS * sim.active.length; } }
      function tick(dtms) {
        const end = sim.t + dtms;
        let guard = 0;
        while (guard++ < 60) {
          const stepEnd = sim.running ? sim.t0 + sim.dur : Infinity;
          const ev = Math.min(sim.next, stepEnd);
          if (ev > end) break;
          sim.t = ev;
          if (ev === sim.next) {
            if (sim.queue.length < 60) sim.queue.push({ id: sim.id++, need: 8 + rng.int(26), got: 0, col: rng.int(6) });
            sim.next += (-Math.log(1 - rng() * 0.999) / S.lam) * 1000;
            if (!sim.running) start(sim.t);
          } else {
            sim.hist.push({ t: sim.t, n: sim.active.length });
            sim.active.forEach((r, k) => { r.got++; if (sim.parts.length < 90) sim.parts.push({ k, t: sim.t, col: r.col }); });
            const keep = [];
            sim.active.forEach((r) => { if (r.got >= r.need) { sim.done.push({ col: r.col, t: sim.t }); sim.served++; } else keep.push(r); });
            sim.active = keep;
            sim.running = false;
            start(sim.t);
          }
        }
        sim.t = end;
        while (sim.hist.length && sim.hist[0].t < sim.t - 4000) sim.hist.shift();
        if (sim.done.length > 16) sim.done.splice(0, sim.done.length - 16);
        sim.parts = sim.parts.filter((p) => sim.t - p.t < 480);
      }
      function prewarm() { for (let i = 0; i < 120; i++) tick(50); }
      prewarm();
      let lastUI = -1;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        tick(Math.min(dt, 0.05) * 1000 * SLOW);
        const mob = w < 560, pad = 14;
        let Q, G, O;
        if (!mob) {
          Q = { x: pad, y: 34, w: w * 0.22, h: h - 84 };
          G = { x: w * 0.26, y: 24, w: w * 0.5, h: h - 64 };
          O = { x: w * 0.79, y: 34, w: w * 0.21 - pad, h: h - 84 };
        } else {
          Q = { x: pad, y: 26, w: w - 2 * pad, h: 70 };
          G = { x: pad, y: 118, w: w - 2 * pad, h: h - 118 - 110 };
          O = { x: pad, y: h - 86, w: w - 2 * pad, h: 62 };
        }
        V.text(g, 'câmera lenta 4×', w - pad, 14, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        /* fila */
        V.text(g, 'FILA', Q.x, Q.y - 9, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        V.box(g, Q.x, Q.y, Q.w, Q.h, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        const cs = 22, cols = Math.max(1, Math.floor((Q.w - 8) / cs)), rows = Math.max(1, Math.floor((Q.h - 8) / cs));
        const maxShow = cols * rows, nq = sim.queue.length;
        const show = Math.min(nq, nq > maxShow ? maxShow - 1 : maxShow);
        for (let i = 0; i < show; i++) {
          const cx = Q.x + 4 + cs / 2 + (i % cols) * cs, cy = Q.y + 4 + cs / 2 + Math.floor(i / cols) * cs;
          critter(g, cx, cy, 8, C.series[sim.queue[i].col], t);
        }
        if (nq > show) {
          const i = show;
          V.text(g, '+' + (nq - show), Q.x + 4 + cs / 2 + (i % cols) * cs, Q.y + 8 + cs / 2 + Math.floor(i / cols) * cs, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 700 });
        }
        if (!nq) V.text(g, 'vazia', Q.x + Q.w / 2, Q.y + Q.h / 2 + 4, { size: 11, align: 'center', color: C.muted });
        /* GPU */
        V.box(g, G.x, G.y, G.w, G.h, { r: 12, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.55), lw: 1.5 });
        V.text(g, 'GPU · cada passo lê todos os pesos uma vez', G.x + 12, G.y + 18, { size: 11.5, font: 'mono', color: C.a, maxW: G.w - 24, weight: 600 });
        const gr = 5, gc = mob ? 16 : 22;
        const gx = G.x + 12, gy = G.y + 30, gw = G.w - 24, gh = Math.min(96, G.h * 0.34);
        const cw = gw / gc, chh = gh / gr;
        const prog = sim.running ? V.clamp((sim.t - sim.t0) / sim.dur, 0, 1) : 0;
        for (let i = 0; i < gr; i++) for (let j = 0; j < gc; j++) {
          const u = (j + 0.5) / gc;
          const lit = sim.running && u <= prog;
          const front = sim.running && Math.abs(u - prog) < 1.2 / gc;
          g.fillStyle = front ? V.alpha(C.a, 0.95) : lit ? V.alpha(C.a, 0.36) : V.alpha(C.muted, 0.13);
          V.rr(g, gx + j * cw + 1, gy + i * chh + 1, cw - 2, chh - 2, 2); g.fill();
        }
        V.text(g, 'matriz de pesos · a varredura é a leitura da memória', gx, gy + gh + 13, { size: 10.5, color: C.muted, maxW: gw });
        const seatsY = gy + gh + 22, seatsH = Math.max(40, G.y + G.h - seatsY - 26);
        const nS = 8, sw = gw / nS, sr = Math.min(13, sw * 0.3, seatsH * 0.28);
        for (let k = 0; k < nS; k++) {
          const sx = gx + sw * (k + 0.5), sy = seatsY + seatsH / 2;
          const usable = k < cap();
          V.box(g, sx - sw / 2 + 3, seatsY, sw - 6, seatsH, { r: 8, fill: usable ? V.alpha(C.a, 0.08) : V.alpha(C.muted, 0.04), stroke: usable ? V.alpha(C.a, 0.4) : V.alpha(C.muted, 0.22), lw: 1 });
          const r = sim.active[k];
          if (r) {
            critter(g, sx, sy - 4, sr, C.series[r.col], t);
            const pw = sw - 16;
            V.box(g, sx - pw / 2, sy + sr + 4, pw, 4, { r: 2, fill: V.alpha(C.muted, 0.22) });
            V.box(g, sx - pw / 2, sy + sr + 4, Math.max(2, (pw * r.got) / r.need), 4, { r: 2, fill: C.series[r.col] });
          } else if (!usable) {
            V.text(g, '×', sx, sy + 5, { size: 15, align: 'center', color: V.alpha(C.muted, 0.55) });
          }
        }
        const B = sim.active.length;
        V.text(g, `passo = ${TW} ms + ${TS} ms × ${B} = ${TW + TS * B} ms` + (S.mode === 'naive' ? ' · 7 vagas ociosas' : ''), gx, G.y + G.h - 9, { size: 11, font: 'mono', color: C.ink, maxW: gw });
        /* saída */
        V.text(g, 'SAÍDA', O.x, O.y - 9, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        V.box(g, O.x, O.y, O.w, O.h, { r: 10, fill: V.alpha(C.c, 0.05), stroke: V.alpha(C.c, 0.35), lw: 1 });
        const tx = O.x + O.w / 2, ty = O.y + (mob ? O.h / 2 : 42);
        sim.parts.forEach((p) => {
          const u = V.clamp((sim.t - p.t) / 480, 0, 1);
          const sx = gx + sw * (p.k + 0.5), sy = seatsY + seatsH / 2 - 4;
          const e = V.ease.inOut(u);
          const x = V.lerp(sx, tx, e), y = V.lerp(sy, ty, e) - Math.sin(u * Math.PI) * 26;
          g.globalAlpha = 1 - u * 0.5;
          V.box(g, x - 3.5, y - 3.5, 7, 7, { r: 2, fill: C.series[p.col] });
          g.globalAlpha = 1;
        });
        V.text(g, nf(sim.served) + ' concluídas', tx, mob ? O.y + 14 : O.y + 18, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        const dn = sim.done.slice(mob ? -8 : -12);
        dn.forEach((d, i) => {
          let x, y;
          if (mob) { x = O.x + 22 + i * 26; y = O.y + O.h - 18; if (x > O.x + O.w - 14) return; }
          else { x = O.x + 18 + (i % 3) * ((O.w - 24) / 3); y = O.y + 78 + Math.floor(i / 3) * 30; if (y > O.y + O.h - 12) return; }
          critter(g, x, y, 8, C.series[d.col], t);
          V.text(g, '✓', x + 9, y - 6, { size: 10, color: C.e, weight: 700 });
        });
        if (t - lastUI > 0.3 || lastUI < 0) {
          lastUI = t;
          const win = Math.min(4000, sim.t);
          const toks = sim.hist.reduce((s, x) => s + x.n, 0);
          rT.set(win > 600 ? nf(toks / (win / 1000), 0) : '—');
          rQ.set(String(sim.queue.length));
          rR.set(sim.hist.length ? nf(toks / sim.hist.length, 1) : '—');
        }
      });
    })();

    /* ================================================================
       Figura 2 — mapa de processos (stepper)
       ================================================================ */
    (function figProc() {
      const fig = q('#fig-processos');
      autoH(fig, 410, 590, 620);
      const ctl = UI.controls(fig);
      const S = { exec: 'mp' };
      UI.seg(ctl, { label: 'Executor', options: [{ v: 'mp', label: 'mp · 2 nós (TP=2)' }, { v: 'uni', label: 'uni · 1 GPU' }], value: 'mp', onChange: (v) => { S.exec = v; rP.set(v === 'mp' ? '4 (1 + 1 + 2)' : '2 (worker dentro do EngineCore)'); } });
      const rP = UI.readout(ctl, 'processos', '4 (1 + 1 + 2)');
      const steps = [
        { t: 'Chegada', d: 'O cliente faz <code>POST /v1/chat/completions</code>. O API server (FastAPI) valida o JSON e abre o stream SSE.', seg: [['cli', 'api']], pay: 'POST /v1/chat/completions', via: 'HTTP', hl: ['api'] },
        { t: 'Template e tokens', d: 'O API server renderiza o chat template (ou o encoder Python do checkpoint) e tokeniza. O <code>AsyncLLM</code> envia um <code>EngineCoreRequest</code> ao EngineCore por ZMQ.', seg: [['api', 'core']], pay: 'EngineCoreRequest', via: 'ZMQ', hl: ['api', 'core'] },
        { t: 'schedule()', d: 'No EngineCore, o scheduler decide quantos tokens cada requisição processa neste passo e pede blocos ao <code>KVCacheManager</code>. Sai um <code>SchedulerOutput</code>.', self: 'core', pay: 'SchedulerOutput', via: 'dentro do EngineCore', hl: ['core'] },
        { t: 'Broadcast do plano', d: 'O executor <code>mp</code> publica o <code>SchedulerOutput</code> na <code>MessageQueue</code>: memória compartilhada para o worker local, ZMQ/TCP para o worker do outro nó. (No <code>uni</code>, é uma chamada de função no mesmo processo.)', seg: [['core', 'w0'], ['core', 'w1']], pay: 'SchedulerOutput', via: 'shm (local) + ZMQ/TCP (remoto)', viaM: 'shm + ZMQ/TCP', hl: ['w0', 'w1'] },
        { t: 'Forward em TP', d: 'Cada worker monta os tensores do passo, escolhe CUDA graph ou eager e roda o modelo. A cada camada, os dois ranks se encontram num all-reduce NCCL entre os nós.', nccl: true, pay: 'all-reduce × camadas', via: 'NCCL entre as GPUs', hl: ['w0', 'w1'] },
        { t: 'Amostrar e responder', d: 'O sampler escolhe os tokens (com especulação, também verifica rascunhos). O rank 0 escreve um <code>ModelRunnerOutput</code> na <code>MessageQueue</code> de resposta, em shared memory, e o EngineCore, que é o leitor dessa fila, o recebe. Essa resposta encerra a RPC do passo, que tem prazo.', seg: [['w0', 'core']], pay: 'ModelRunnerOutput', via: 'MessageQueue de resposta (shm) → EngineCore lê', viaM: 'fila de resposta (shm)', hl: ['w0', 'core'] },
        { t: 'update_from_output()', d: 'O scheduler registra os tokens, libera blocos de quem terminou, avança as gramáticas e manda <code>EngineCoreOutputs</code> ao API server por ZMQ.', seg: [['core', 'api']], pay: 'EngineCoreOutputs', via: 'ZMQ', hl: ['core', 'api'] },
        { t: 'Destokenizar e transmitir', d: 'O output processor destokeniza incrementalmente, aplica stop strings e os parsers de raciocínio e ferramentas, e envia o delta ao cliente em SSE. O próximo passo já está sendo agendado.', seg: [['api', 'cli']], pay: 'data: {"delta": …}', via: 'SSE', hl: ['api'] },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5200 });
      function lay(w, h) {
        const P = {};
        if (w >= 620) {
          const bw = Math.min(158, w * 0.2), bh = 66;
          P.mob = false;
          P.cli = { x: w * 0.07, y: h * 0.5, w: Math.min(88, w * 0.115), h: 48 };
          P.head = { x: w * 0.152, y: 24, w: w * 0.525, h: h - 40 };
          P.work = { x: w * 0.705, y: 24, w: w * 0.28, h: h - 40 };
          P.api = { x: w * 0.29, y: h * 0.33, w: bw, h: bh };
          P.core = { x: w * 0.545, y: h * 0.33, w: bw, h: bh };
          P.w0 = { x: w * 0.545, y: h * 0.75, w: bw, h: bh };
          P.w1 = { x: w * 0.845, y: h * 0.75, w: bw, h: bh };
        } else {
          const bw = Math.min(150, w * 0.39), bh = 62;
          P.mob = true;
          P.cli = { x: w * 0.5, y: 30, w: 118, h: 38 };
          P.head = { x: 8, y: 66, w: w - 16, h: h * 0.63 - 66 };
          P.api = { x: w * 0.255, y: 66 + 78, w: bw, h: bh };
          P.core = { x: w * 0.745, y: 66 + 78, w: bw, h: bh };
          P.w0 = { x: w * 0.5, y: h * 0.63 - 58, w: bw, h: bh };
          P.work = { x: 8, y: h * 0.68, w: w - 16, h: h * 0.3 };
          P.w1 = { x: w * 0.5, y: h * 0.68 + h * 0.15 + 8, w: bw, h: bh };
        }
        return P;
      }
      const ST = {
        cli: { l: 'Cliente', s: 'HTTP · SSE', c: 'd' },
        api: { l: 'API server', s: 'FastAPI · template · tokenizer · parsers · AsyncLLM', c: 'd' },
        core: { l: 'EngineCore', s: 'Scheduler · KV manager · Executor', c: 'b' },
        w0: { l: 'Worker rank 0', s: 'ModelRunner · GPU', c: 'a' },
        w1: { l: 'Worker rank 1', s: '--headless · ModelRunner · GPU', c: 'a' },
      };
      function segPts(P, a, b) {
        const A = P[a], B = P[b];
        if (P.mob && ((a === 'core' && b === 'w1') || (a === 'w1' && b === 'core'))) {
          const p0 = boxEdge(A, A.x + 40, A.y + 100), p3 = [B.x + B.w / 2, B.y];
          return { bez: [p0, [P.head.x + P.head.w - 10, A.y + 120], [P.head.x + P.head.w - 10, B.y - 40], p3] };
        }
        const p0 = boxEdge(A, B.x, B.y), p1 = boxEdge(B, A.x, A.y);
        return { line: [p0, p1] };
      }
      function ptOn(sp, u) { if (sp.bez) return V.bez(sp.bez[0], sp.bez[1], sp.bez[2], sp.bez[3], u); return V.along(sp.line[0][0], sp.line[0][1], sp.line[1][0], sp.line[1][1], u); }
      function drawSeg(g, sp, col, active, t, label) {
        const C = V.C();
        g.save();
        g.strokeStyle = active ? col : V.alpha(C.muted, 0.35); g.lineWidth = active ? 2.2 : 1.4;
        g.setLineDash(active ? [6, 6] : [3, 5]); g.lineDashOffset = active ? -t * 24 : 0;
        g.beginPath();
        if (sp.bez) { g.moveTo(sp.bez[0][0], sp.bez[0][1]); g.bezierCurveTo(sp.bez[1][0], sp.bez[1][1], sp.bez[2][0], sp.bez[2][1], sp.bez[3][0], sp.bez[3][1]); }
        else { g.moveTo(sp.line[0][0], sp.line[0][1]); g.lineTo(sp.line[1][0], sp.line[1][1]); }
        g.stroke(); g.restore();
        if (label) {
          const m = ptOn(sp, sp.lu == null ? 0.5 : sp.lu);
          pill(g, m[0], m[1] + (sp.ldy || 0), label, active ? col : C.muted, { size: 9.5, a: active ? 0.2 : 0.08 });
        }
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const P = lay(w, h);
        const uni = S.exec === 'uni';
        const step = steps[st.i];
        const e = st.elapsed();
        const p = V.clamp((e % 3.4) / 1.7, 0, 1);
        /* molduras */
        const frame = (R, label, col) => {
          V.box(g, R.x, R.y, R.w, R.h, { r: 14, fill: V.alpha(col, 0.035), stroke: V.alpha(col, 0.35), lw: 1 });
          if (P.mob) V.text(g, label, R.x + R.w - 12, R.y + 16, { size: 10.5, font: 'mono', align: 'right', color: V.alpha(col, 0.95), weight: 600, maxW: R.w * 0.55 });
          else V.text(g, label, R.x + 12, R.y + 16, { size: 10.5, font: 'mono', color: V.alpha(col, 0.95), weight: 600, maxW: R.w - 20 });
        };
        frame(P.head, uni ? 'NÓ ÚNICO · 1 GPU' : 'NÓ HEAD · rank 0', C.b);
        if (!uni) frame(P.work, 'NÓ WORKER · rank 1', C.a);
        /* arestas */
        const edges = uni ? [['cli', 'api', 'HTTP'], ['api', 'core', 'ZMQ']] : [['cli', 'api', 'HTTP'], ['api', 'core', 'ZMQ'], ['core', 'w0', 'shm'], ['core', 'w1', 'ZMQ/TCP'], ['w0', 'w1', 'NCCL']];
        const activeSegs = [];
        if (step.seg) step.seg.forEach(([a, b]) => { if (uni && (a === 'w0' || a === 'w1' || b === 'w0' || b === 'w1')) return; activeSegs.push(a + '>' + b); });
        edges.forEach(([a, b, lab]) => {
          const sp = segPts(P, a, b);
          if (P.mob && a === 'api' && b === 'core') sp.ldy = -(P.api.h / 2 + 11);
          if (P.mob && a === 'core' && b === 'w1') sp.lu = 0.3;
          if (P.mob && a === 'w0' && b === 'w1') sp.lu = 0.72;
          const act = activeSegs.includes(a + '>' + b) || activeSegs.includes(b + '>' + a) || (!uni && step.nccl && a === 'w0' && b === 'w1');
          const col = lab === 'NCCL' ? C.c : lab === 'shm' ? C.e : lab === 'ZMQ/TCP' ? C.a : C.d;
          drawSeg(g, sp, col, act, t, lab);
        });
        /* estações */
        const hl = step.hl.map((s) => (uni && (s === 'w0' || s === 'w1') ? 'core' : s));
        ['cli', 'api', 'core'].concat(uni ? [] : ['w0', 'w1']).forEach((k) => {
          const d = ST[k];
          const sub = uni && k === 'core' ? 'Scheduler · KV · Executor uni + Worker (GPU)' : d.s;
          station(g, P[k], d.l, sub, C[d.c], hl.includes(k), t);
        });
        /* animação do passo */
        const selfAnim = (k, label) => {
          const B = P[k];
          const r = B.w * 0.55 + Math.sin(t * 5) * 3;
          g.strokeStyle = V.alpha(C.c, 0.7 - 0.4 * p); g.lineWidth = 2;
          g.beginPath(); g.ellipse(B.x, B.y, r * (0.8 + 0.3 * p), B.h * (0.7 + 0.25 * p), 0, 0, Math.PI * 2); g.stroke();
        };
        let segs = [];
        if (step.seg) step.seg.forEach(([a, b]) => { if (uni && (a === 'w0' || a === 'w1' || b === 'w0' || b === 'w1')) return; segs.push(segPts(P, a, b)); });
        if (step.self) selfAnim(step.self, step.pay);
        else if (step.nccl) {
          if (uni) selfAnim('core', 'forward (1 GPU)');
          else {
            const sp = segPts(P, 'w0', 'w1');
            for (let i = 0; i < 4; i++) {
              const u = ((t * 0.9 + i / 4) % 1);
              const a = ptOn(sp, u), b = ptOn(sp, 1 - u);
              V.dot(g, a[0], a[1], 4, C.c); V.dot(g, b[0], b[1], 4, C.a);
            }
          }
        } else if (segs.length) {
          segs.forEach((sp, i) => {
            const pt = ptOn(sp, V.ease.inOut(p));
            critter(g, pt[0], pt[1], 9, i === 0 ? C.d : C.a, t, 'still');
          });
        } else if (uni && step.seg) selfAnim('core', step.pay);
        const pay = uni && (step.nccl || (step.seg && step.seg.every(([a, b]) => ['w0', 'w1'].includes(a) || ['w0', 'w1'].includes(b)))) ? step.pay + ' · chamada no mesmo processo' : step.pay + ' · via ' + (P.mob && step.viaM ? step.viaM : step.via);
        const ty = P.mob ? P.head.y + P.head.h + (P.work ? (P.work.y - P.head.y - P.head.h) / 2 : 14) : 12;
        const tx = P.mob ? 12 : P.head.x;
        let psz = 10;
        while (psz > 8.5 && tw(g, 'carga: ' + pay, psz, 'mono', 600) + 14 > w - tx - 8) psz -= 0.5;
        const pw = pill(g, tx, ty, 'carga: ' + pay, C.d, { align: 'left', size: psz, a: 0.2, weight: 600 });
        if (!P.mob) V.text(g, 'passo ' + (st.i + 1) + '/8', w - 14, 12 + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
      });
    })();

    /* ================================================================
       Figura 3 — girar ou dormir
       ================================================================ */
    (function figEspera() {
      const fig = q('#fig-espera');
      autoH(fig, 430, 640, 640);
      const ctl = UI.controls(fig);
      const S = { W: 1000, D: 100, T: 320 };
      let cpu = 1, wakes = 0;
      const lg = (x) => Math.log10(x);
      const fmtMs = (x) => (x >= 999.5 ? '1 s' : nf(x, x < 10 ? 1 : 0) + ' ms');
      const sl = UI.slider(ctl, { label: 'Janela de spin', min: 0, max: 3, step: 0.01, value: 3, fmt: (v) => fmtMs(Math.pow(10, v)), onInput: (v) => { S.W = Math.pow(10, v); recalc(); } });
      UI.seg(ctl, { label: 'Receita', options: [{ v: 3, label: 'stock 1 s' }, { v: lg(2), label: 'DS 2 ms' }, { v: lg(16), label: 'GLM 16 ms (opcional)' }], value: 3, onChange: (v) => sl.set(v) });
      UI.seg(ctl, { label: 'Passo', options: [50, 100, 250].map((v) => ({ v, label: v + ' ms' })), value: 100, onChange: (v) => { S.D = v; recalc(); } });
      const rC = UI.readout(ctl, 'CPU girando', '—'), rW = UI.readout(ctl, 'despertares/passo', '—');
      function msgs(k) {
        const D = S.D;
        const base = k * D + hash(k, 1) * 0.08 * D;
        const out = [base, base + 3 + hash(k, 2) * 7];
        if (hash(k, 3) < 0.5) out.push(out[1] + 1.5 + hash(k, 4) * 2);
        return out;
      }
      function recalc() {
        let prev = null, spin = 0, tot = 0, wk = 0;
        const N = 300;
        for (let k = 0; k < N; k++) for (const m of msgs(k)) {
          if (prev != null) { const gap = m - prev; spin += Math.min(gap, S.W); tot += gap; if (gap > S.W) wk++; }
          prev = m;
        }
        cpu = tot ? spin / tot : 1; wakes = wk / N;
        rC.set(nf(cpu * 100, 0) + '%'); rW.set(nf(wakes, 2));
      }
      recalc();
      const MEAS = [{ w: 2, v: -1.68, l: '2 ms', s: '−1,68%' }, { w: 16, v: 0.95, l: '16 ms', s: '+0,95%' }, { w: 32, v: -1.8, l: '32 ms', s: '≈ −1,8%' }, { w: 64, v: -1.8, l: '64 ms', s: '≈ −1,8%' }, { w: 1000, v: 0, l: 'stock 1 s', s: 'referência' }];
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const mob = w < 640;
        S.T += Math.min(dt, 0.05) * (S.D / 1.6);
        const T = S.T;
        const span = 2.4 * S.D;
        const L = { x: 16, y: 26, w: w - 16 - 70, h: 150 };
        const x = (tau) => L.x + ((tau - (T - span)) / span) * L.w;
        V.text(g, 'LINHA DO TEMPO DE UM LEITOR (câmera lenta)', L.x, 16, { size: 10.5, font: 'mono', color: C.muted, weight: 600, maxW: L.w });
        const yM = L.y + 30, yS = L.y + 66, hS = 24;
        V.text(g, 'mensagens', L.x, yM - 16, { size: 10, color: C.muted });
        V.text(g, 'estado', L.x, yS - 6, { size: 10, color: C.muted });
        V.line(g, L.x, yM + 10, L.x + L.w, yM + 10, { color: V.alpha(C.muted, 0.3), lw: 1 });
        V.box(g, L.x, yS, L.w, hS, { r: 6, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.2), lw: 1 });
        const k0 = Math.floor((T - span) / S.D) - 1, k1 = Math.floor(T / S.D) + 1;
        const ms = [];
        for (let k = Math.max(0, k0); k <= k1; k++) msgs(k).forEach((m) => { if (m <= T) ms.push(m); });
        ms.sort((a, b) => a - b);
        g.save();
        g.beginPath(); g.rect(L.x, 0, L.w, h); g.clip();
        const drawSeg = (a, b, kind) => {
          const xa = Math.max(L.x, x(a)), xb = Math.min(L.x + L.w, x(b));
          if (xb <= xa) return;
          if (kind === 'spin') {
            g.fillStyle = V.alpha(C.c, 0.55); g.fillRect(xa, yS + 2, xb - xa, hS - 4);
            for (let xx = xa + ((t * 60) % 8); xx < xb; xx += 8) { g.fillStyle = V.alpha(C.d, 0.7); g.fillRect(xx, yS + 4 + Math.sin(xx * 0.7 + t * 9) * 3 + 6, 2, 4); }
          } else {
            g.fillStyle = V.alpha(C.b, 0.22); g.fillRect(xa, yS + 2, xb - xa, hS - 4);
            for (let xx = xa + 10; xx < xb - 6; xx += 34) V.text(g, 'z', xx, yS + hS / 2 + 4 + Math.sin(t * 2 + xx) * 2, { size: 11, font: 'mono', color: V.alpha(C.b, 0.9) });
          }
        };
        for (let i = 0; i < ms.length; i++) {
          const a = ms[i], b = i + 1 < ms.length ? ms[i + 1] : T;
          const sEnd = Math.min(b, a + S.W);
          drawSeg(a, sEnd, 'spin');
          if (sEnd < b) drawSeg(sEnd, b, 'sleep');
        }
        if (ms.length && x(ms[0]) > L.x) drawSeg(T - span, ms[0], ms[0] - (T - span) > S.W ? 'sleep' : 'spin');
        for (let i = 0; i < ms.length; i++) {
          const m = ms[i], X = x(m);
          if (X < L.x - 10) continue;
          V.box(g, X - 6, yM - 4, 12, 9, { r: 1.5, fill: V.alpha(C.d, 0.3), stroke: C.d, lw: 1 });
          g.strokeStyle = C.d; g.lineWidth = 1; g.beginPath(); g.moveTo(X - 6, yM - 4); g.lineTo(X, yM + 1); g.lineTo(X + 6, yM - 4); g.stroke();
          if (i > 0 && m - ms[i - 1] > S.W) {
            g.fillStyle = C.d; g.beginPath(); g.moveTo(X + 2, yS - 14); g.lineTo(X - 3, yS - 4); g.lineTo(X + 1, yS - 4); g.lineTo(X - 2, yS + 5); g.lineTo(X + 5, yS - 7); g.lineTo(X + 1, yS - 7); g.closePath(); g.fill();
          }
        }
        g.restore();
        /* escala */
        const yA = yS + hS + 22;
        const tick10 = (10 / span) * L.w;
        V.line(g, L.x, yA, L.x + tick10, yA, { color: C.ink, lw: 2 });
        V.text(g, '10 ms', L.x + tick10 + 6, yA + 4, { size: 10.5, font: 'mono', color: C.ink });
        const xs = L.x + L.w * 0.5;
        V.text(g, 'passo ≈ ' + S.D + ' ms · janela ' + fmtMs(S.W), xs, yA + 4, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: L.w * 0.5 });
        /* leitor */
        const lastM = ms.length ? ms[ms.length - 1] : -1e9;
        const spinning = T - lastM <= S.W;
        const cx = L.x + L.w + 36, cy = yS + hS / 2;
        V.line(g, L.x + L.w, L.y + 10, L.x + L.w, yS + hS + 6, { color: V.alpha(C.ink, 0.45), lw: 1, dash: [3, 3] });
        V.text(g, 'agora', L.x + L.w, L.y + 6, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
        critter(g, cx, cy, 13, spinning ? C.c : C.b, t, spinning ? null : 'sleep');
        if (spinning) for (let i = 0; i < 3; i++) { const a = t * 8 + (i * Math.PI * 2) / 3; V.dot(g, cx + Math.cos(a) * 19, cy + Math.sin(a) * 19, 2.2, C.d); }
        else V.text(g, 'zZ', cx + 12, cy - 14 + Math.sin(t * 2) * 2, { size: 11, font: 'mono', color: C.b, weight: 700 });
        V.text(g, spinning ? 'girando' : 'dormindo', cx, cy + 28, { size: 10, align: 'center', color: spinning ? C.c : C.b, weight: 600 });
        /* painéis de baixo */
        let Pl, Pr;
        const top = L.y + L.h + 26;
        if (!mob) { Pl = { x: 16, y: top, w: w * 0.47 - 16, h: h - top - 12 }; Pr = { x: w * 0.51, y: top, w: w * 0.49 - 16, h: h - top - 12 }; }
        else { Pl = { x: 16, y: top, w: w - 32, h: 170 }; Pr = { x: 16, y: top + 186, w: w - 32, h: h - top - 186 - 12 }; }
        V.box(g, Pl.x, Pl.y, Pl.w, Pl.h, { r: 12, fill: V.alpha(C.c, 0.04), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'MODELO DE GUARDANAPO (ilustrativo)', Pl.x + 12, Pl.y + 18, { size: 10.5, font: 'mono', color: C.muted, weight: 600, maxW: Pl.w - 24 });
        const bx = Pl.x + 12, bw = Pl.w - 24, by = Pl.y + 32;
        V.text(g, 'CPU girando por leitor', bx, by + 10, { size: 11.5, color: C.ink });
        V.box(g, bx, by + 18, bw, 14, { r: 7, fill: V.alpha(C.muted, 0.15) });
        V.box(g, bx, by + 18, Math.max(6, bw * cpu), 14, { r: 7, fill: V.alpha(C.c, 0.85) });
        if (bw * cpu > 54) V.text(g, nf(cpu * 100, 0) + '%', bx + bw * cpu - 8, by + 29, { size: 11, font: 'mono', align: 'right', color: ink(C), weight: 700 });
        else V.text(g, nf(cpu * 100, 0) + '%', bx + Math.max(6, bw * cpu) + 8, by + 29, { size: 11, font: 'mono', color: C.ink, weight: 700 });
        V.text(g, `≈ ${nf(cpu * S.D, 0)} ms de núcleo ocupado a cada passo · ${nf(wakes, 2)} despertares/passo`, bx, by + 50, { size: 10.5, font: 'mono', color: C.muted, maxW: bw });
        /* eixo log com marcadores */
        const ay = by + 88, lx = (v) => bx + (lg(v) / 3) * bw;
        V.line(g, bx, ay, bx + bw, ay, { color: V.alpha(C.ink, 0.5), lw: 1.2 });
        [1, 10, 100, 1000].forEach((v) => { V.line(g, lx(v), ay - 4, lx(v), ay + 4, { color: C.muted, lw: 1 }); V.text(g, v >= 1000 ? '1 s' : v + ' ms', lx(v), ay + 16, { size: 9.5, font: 'mono', align: v === 1 ? 'left' : v === 1000 ? 'right' : 'center', color: C.muted }); });
        const dX = lx(S.D);
        V.line(g, dX, ay - 22, dX, ay + 4, { color: V.alpha(C.b, 0.8), lw: 1.2, dash: [3, 3] });
        V.text(g, 'passo', dX, ay - 25, { size: 9.5, font: 'mono', align: 'center', color: C.b });
        [[2, 'DS', C.ds], [16, 'GLM', C.glm], [1000, 'stock', C.muted]].forEach(([v, l, col]) => {
          const X = lx(v);
          V.dot(g, X, ay, 4.5, col);
          V.text(g, l, X + (v === 1000 ? -6 : 6), ay - 8, { size: 9.5, font: 'mono', align: v === 1000 ? 'right' : 'left', color: col, weight: 600 });
        });
        const cX = lx(S.W);
        g.fillStyle = C.c; g.beginPath(); g.moveTo(cX, ay + 22); g.lineTo(cX - 6, ay + 32); g.lineTo(cX + 6, ay + 32); g.closePath(); g.fill();
        V.text(g, 'W', cX, ay + 44, { size: 10, font: 'mono', align: 'center', color: C.c, weight: 700 });
        /* medições */
        V.box(g, Pr.x, Pr.y, Pr.w, Pr.h, { r: 12, fill: V.alpha(C.a, 0.04), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'MEDIDO · GLM TP2, MNBT 2048 · Δ decode vs stock', Pr.x + 12, Pr.y + 18, { size: 10.5, font: 'mono', color: C.muted, weight: 600, maxW: Pr.w - 24 });
        const cy0 = Pr.y + 32, ch = Pr.h - 32 - 44, zeroY = cy0 + ch / 2, sc = ch / 2 / 2.2;
        const n = MEAS.length, bwid = (Pr.w - 24) / n;
        V.line(g, Pr.x + 12, zeroY, Pr.x + Pr.w - 12, zeroY, { color: V.alpha(C.ink, 0.45), lw: 1 });
        let best = 0, bd = 1e9;
        MEAS.forEach((m, i) => { const d = Math.abs(lg(m.w) - lg(S.W)); if (d < bd) { bd = d; best = i; } });
        MEAS.forEach((m, i) => {
          const X = Pr.x + 12 + i * bwid + bwid * 0.2, BW = bwid * 0.6;
          const hh = m.v * sc;
          const col = m.v > 0 ? C.a : m.v < 0 ? C.c : C.muted;
          const on = i === best;
          if (m.v !== 0) V.box(g, X, hh > 0 ? zeroY - hh : zeroY, BW, Math.abs(hh), { r: 4, fill: V.alpha(col, on ? 0.95 : 0.45), stroke: on ? col : null, lw: 2 });
          else V.box(g, X, zeroY - 2, BW, 4, { r: 2, fill: V.alpha(col, 0.7) });
          V.text(g, m.s, X + BW / 2, hh > 0 ? zeroY - hh - 6 : zeroY - hh + 14, { size: 10, font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400, maxW: bwid });
          V.text(g, m.l, X + BW / 2, Pr.y + Pr.h - 26, { size: 10, font: 'mono', align: 'center', color: on ? col : C.muted, weight: on ? 700 : 400, maxW: bwid });
        });
        V.text(g, 'a 16 ms: CPU ativa do EngineCore −85,3%', Pr.x + Pr.w / 2, Pr.y + Pr.h - 9, { size: 10.5, align: 'center', color: C.a, weight: 600, maxW: Pr.w - 20 });
      });
    })();

    /* ================================================================
       Figura 4 — as planilhas do model runner
       ================================================================ */
    (function figRunner() {
      const fig = q('#fig-runner');
      autoH(fig, 440, 600, 640);
      const ctl = UI.controls(fig);
      const S = { mr: 'v1' };
      UI.seg(ctl, { label: 'Model runner', options: [{ v: 'v1', label: 'V1 · prepara na CPU' }, { v: 'v2', label: 'V2 · prepara na GPU' }], value: 'v1', onChange: (v) => { S.mr = v; } });
      const REQ = [
        { id: 'A', kind: 'decode', pos: [9], ids: [812], blocks: [3, 7, 1], old: 9, c: 'a' },
        { id: 'B', kind: 'prefill', pos: [0, 1, 2, 3, 4], ids: [101, 7, 3302, 45, 9], blocks: [5, 2], old: 0, c: 'b' },
        { id: 'C', kind: 'decode', pos: [6], ids: [2201], blocks: [0, 4], old: 6, c: 'd' },
      ];
      const BS = 4;
      const ROWS = [];
      REQ.forEach((r, ri) => r.pos.forEach((p, k) => ROWS.push({ ri, p, id: r.ids[k], slot: r.blocks[Math.floor(p / BS)] * BS + (p % BS) })));
      const QSL = [0, 1, 6, 7], LOGI = [0, 5, 6], SAMP = [4431, 88, 1207];
      const owner = {}; REQ.forEach((r, ri) => r.blocks.forEach((b) => { owner[b] = ri; }));
      const oldSlots = new Set();
      REQ.forEach((r) => { for (let p = 0; p < r.old; p++) oldSlots.add(r.blocks[Math.floor(p / BS)] * BS + (p % BS)); });
      const newSlots = new Set(ROWS.map((x) => x.slot));
      const steps = [
        { t: 'O plano chega', d: 'O <code>SchedulerOutput</code> diz: A avança 1 token (decode), B faz um pedaço de prefill de 5, C avança 1. Vêm junto os ids dos blocos de KV de cada requisição.' },
        { t: 'Achatar numa fila', d: 'Os 7 tokens do passo viram um vetor só, <code>input_ids</code>, com as posições de cada um em <code>positions</code>. O modelo vai ver 7 linhas, sem saber de requisições.' },
        { t: 'Onde começa cada um', d: '<code>query_start_loc</code> = soma acumulada dos tokens agendados: <code>[0, 1, 6, 7]</code>. É assim que a atenção sabe que as linhas 1–5 pertencem a B.' },
        { t: 'Onde gravar o KV', d: 'A tabela de blocos traduz posição lógica em bloco físico, e o <code>slot_mapping</code> dá o endereço exato: <code>BT[r, p // 4]·4 + p % 4</code>. O K/V de cada token cai no slot certo do pool.' },
        { t: 'Forward', d: '7 linhas atravessam as camadas. Este passo é misto (tem prefill), então não serve o graph FULL de decode uniforme: roda piecewise ou eager. Um decode puro seria completado até um tamanho de captura.' },
        { t: 'Amostrar só onde importa', d: 'Os logits são calculados só nas posições de amostragem (a última de cada requisição): <code>[0, 5, 6]</code>. O LM head sobre ≈ 130–250 mil tokens de vocabulário (DeepSeek 129.280, GLM 154.880, Qwen 248.320) custa caro demais para as 7 linhas.' },
        { t: 'Devolver', d: 'O <code>ModelRunnerOutput</code> leva um token amostrado por requisição. Uma requisição encoder-only receberia <code>[]</code>, e jamais um token inventado.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const s = st.i, e = st.elapsed();
        const RC = REQ.map((r) => C[r.c]);
        const lx = 12, ax = mob ? 78 : 150;
        const cw = mob ? Math.min(32, (w - ax - 14) / 7 - 3) : 34, gap = 4, chh = 28;
        /* plano */
        const cardW = mob ? (w - 24 - 12) / 3 : 170;
        REQ.forEach((r, i) => {
          const X = 12 + i * (cardW + 6), Y = 10;
          const on = s === 0;
          V.box(g, X, Y, cardW, mob ? 46 : 40, { r: 9, fill: V.alpha(RC[i], on ? 0.24 : 0.1), stroke: V.alpha(RC[i], on ? 1 : 0.5), lw: on ? 2 : 1 });
          critter(g, X + 14, Y + (mob ? 23 : 20), 8, RC[i], t);
          V.text(g, r.id + ' · ' + r.kind, X + 28, Y + 16, { size: 11.5, weight: 650, color: C.ink, maxW: cardW - 32 });
          V.text(g, '+' + r.pos.length + ' tok · pos ' + (r.pos.length > 1 ? r.pos[0] + '–' + r.pos[r.pos.length - 1] : r.pos[0]), X + 28, Y + 31, { size: 10, font: 'mono', color: C.muted, maxW: cardW - 32 });
        });
        if (!mob) V.text(g, 'SchedulerOutput', 12 + 3 * (cardW + 6) + 6, 34, { size: 11, font: 'mono', color: C.muted });
        /* faixa de onde vivem os arrays */
        const y0 = mob ? 72 : 62;
        const rowsY = [y0 + 8, y0 + 8 + 38, y0 + 8 + 76, y0 + 8 + 114, y0 + 8 + 152];
        const areaW = ax - 8 + 7 * (cw + gap) + 8;
        const v1 = S.mr === 'v1';
        const bandCol = v1 ? C.d : C.a;
        V.box(g, 6, y0 - 2, areaW, 190, { r: 10, fill: V.alpha(bandCol, 0.05), stroke: V.alpha(bandCol, 0.3), lw: 1 });
        const bandLabel = v1 ? 'CPU · arrays em memória pinned → cópia H2D' : 'GPU · estado persistente, kernels Triton, sem sync';
        V.text(g, bandLabel, 14, y0 + 196, { size: 10, font: 'mono', color: bandCol, weight: 600, maxW: areaW - 10 });
        if (v1 && s >= 1 && s <= 3) {
          const ux = 6 + areaW + 6, uy = y0 + 90;
          if (!mob) {
            V.arrow(g, ux, uy, ux + 34, uy, { color: C.d, lw: 2, dash: [5, 4] });
            const pu = (t * 0.8) % 1;
            V.box(g, ux + pu * 30 - 4, uy - 4, 8, 8, { r: 2, fill: C.d });
            V.text(g, 'H2D', ux + 17, uy - 10, { size: 10, font: 'mono', align: 'center', color: C.d });
          }
        }
        const labels = mob ? ['ids', 'pos', 'qsl', 'slot', 'logits'] : ['input_ids', 'positions', 'query_start_loc', 'slot_mapping', 'logits_indices'];
        const cellX = (j) => ax + j * (cw + gap);
        const cell = (x, y, val, col, on, fill) => {
          V.box(g, x, y, cw, chh, { r: 5, fill: fill ? V.alpha(col, on ? 0.9 : 0.35) : V.alpha(C.muted, 0.06), stroke: V.alpha(col, fill ? 0.9 : 0.3), lw: 1 });
          if (fill) V.text(g, val, x + cw / 2, y + chh / 2 + 0.5, { size: Math.min(11, cw * 0.34), font: 'mono', align: 'center', base: 'middle', color: on ? ink(C) : C.ink, weight: 600, maxW: cw - 2 });
        };
        labels.forEach((L, i) => V.text(g, L, lx + 6, rowsY[i] + chh / 2 + 4, { size: mob ? 10 : 11, font: 'mono', color: C.ink, maxW: ax - 20 }));
        const nShow1 = s === 1 ? Math.min(7, Math.floor(e * 4) + 1) : s > 1 ? 7 : 0;
        for (let j = 0; j < 7; j++) {
          const R = ROWS[j], col = RC[R.ri];
          cell(cellX(j), rowsY[0], R.id, col, s === 1 && j === nShow1 - 1, j < nShow1);
          cell(cellX(j), rowsY[1], R.p, col, s === 1 && j === nShow1 - 1, j < nShow1);
          const nS3 = s === 3 ? Math.min(7, Math.floor(e * 3) + 1) : s > 3 ? 7 : 0;
          cell(cellX(j), rowsY[3], R.slot, col, s === 3 && j === nS3 - 1, j < nS3);
        }
        if (s >= 2) {
          const nQ = s === 2 ? Math.min(4, Math.floor(e * 2.5) + 1) : 4;
          for (let j = 0; j < 4; j++) cell(cellX(j), rowsY[2], QSL[j], C.e, s === 2 && j === nQ - 1, j < nQ);
          REQ.forEach((r, ri) => {
            if (ri >= nQ - 0 && s === 2) return;
            const x0 = cellX(QSL[ri]), x1 = cellX(QSL[ri + 1] - 1) + cw;
            g.strokeStyle = RC[ri]; g.lineWidth = 2;
            g.beginPath(); g.moveTo(x0 + 2, rowsY[0] - 4); g.lineTo(x0 + 2, rowsY[0] - 7); g.lineTo(x1 - 2, rowsY[0] - 7); g.lineTo(x1 - 2, rowsY[0] - 4); g.stroke();
          });
        } else for (let j = 0; j < 4; j++) cell(cellX(j), rowsY[2], '', C.e, false, false);
        for (let j = 0; j < 3; j++) cell(cellX(j), rowsY[4], s >= 5 ? LOGI[j] : '', C.c, s === 5, s >= 5);
        if (s >= 5) LOGI.forEach((ix) => { const X = cellX(ix); g.strokeStyle = C.c; g.lineWidth = 2.5; V.rr(g, X - 2, rowsY[0] - 2, cw + 4, chh + 4, 6); g.stroke(); });
        /* tabela de blocos e pool */
        let btX, btY, poolX, poolY, poolW, fwY, outX, outY;
        const bc = 26;
        if (!mob) { btX = w - 12 - 3 * (bc + 3) - 10; btY = y0 + 20; fwY = y0 + 214; poolY = h - 64; poolX = 12; poolW = w - 24; outX = btX - 34; outY = btY + 3 * (bc + 3) + 40; }
        else { btX = 70; btY = y0 + 226; fwY = y0 + 336; poolY = h - 108; poolX = 12; poolW = w - 24; outX = 190; outY = y0 + 226; }
        V.text(g, 'block table', btX + (3 * (bc + 3)) / 2 - 2, btY - 8, { size: 10.5, font: 'mono', align: 'center', color: s === 3 ? C.ink : C.muted, weight: 600 });
        REQ.forEach((r, ri) => {
          V.text(g, r.id, btX - 10, btY + ri * (bc + 3) + bc / 2 + 4, { size: 11, font: 'mono', align: 'right', color: RC[ri], weight: 700 });
          for (let j = 0; j < 3; j++) {
            const X = btX + j * (bc + 3), Y = btY + ri * (bc + 3);
            const has = j < r.blocks.length;
            V.box(g, X, Y, bc, bc, { r: 4, fill: has && s >= 3 ? V.alpha(RC[ri], 0.4) : V.alpha(C.muted, 0.07), stroke: V.alpha(has ? RC[ri] : C.muted, 0.6), lw: 1 });
            if (has) V.text(g, s >= 3 ? r.blocks[j] : '·', X + bc / 2, Y + bc / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 600 });
            else V.text(g, '–', X + bc / 2, Y + bc / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.muted });
          }
        });
        V.text(g, 'bloco lógico →', btX, btY + 3 * (bc + 3) + 12, { size: 9.5, font: 'mono', color: C.muted });
        /* pool de KV */
        const perRow = mob ? 16 : 32;
        const sw2 = Math.min(26, (poolW - (perRow / 4 - 1) * 6) / perRow);
        const slotXY = (sl) => { const b = Math.floor(sl / BS), k = sl % BS; const rowI = Math.floor(sl / perRow); const inRow = sl % perRow; const bi = Math.floor(inRow / 4); return [poolX + inRow * sw2 + bi * 6, poolY + rowI * (sw2 + 14)]; };
        V.text(g, 'pool de KV · 8 blocos × 4 slots (bloco físico 0…7)', poolX, poolY - 8, { size: 10.5, font: 'mono', color: C.muted, maxW: poolW });
        for (let sl = 0; sl < 32; sl++) {
          const [X, Y] = slotXY(sl);
          const b = Math.floor(sl / BS), ow = owner[b];
          const isNew = newSlots.has(sl) && s >= 3, isOld = oldSlots.has(sl);
          const col = ow == null ? C.muted : RC[ow];
          const pulse = isNew ? 0.65 + 0.35 * Math.sin(t * 5 + sl) : 0;
          V.box(g, X, Y, sw2 - 2, sw2 - 2, { r: 3, fill: isNew ? V.alpha(col, pulse) : isOld ? V.alpha(col, 0.3) : V.alpha(C.muted, ow == null ? 0.05 : 0.1), stroke: V.alpha(col, ow == null ? 0.25 : 0.6), lw: 1 });
          if (sl % 4 === 0) V.text(g, b, X + 2, Y + sw2 + 9, { size: 9, font: 'mono', color: C.muted });
        }
        if (s === 3 || s === 4) {
          ROWS.forEach((R, j) => {
            const nS3 = s === 3 ? Math.min(7, Math.floor(e * 3) + 1) : 7;
            if (j >= nS3) return;
            const [X, Y] = slotXY(R.slot);
            const sx = cellX(j) + cw / 2, sy = rowsY[3] + chh;
            g.strokeStyle = V.alpha(RC[R.ri], 0.7); g.lineWidth = 1.3;
            g.beginPath(); g.moveTo(sx, sy); g.bezierCurveTo(sx, sy + 40, X + sw2 / 2, Y - 40, X + sw2 / 2 - 1, Y - 1); g.stroke();
          });
        }
        /* forward */
        if (s >= 4) {
          const fx = mob ? 12 : 12, fw = mob ? w - 24 : areaW - 6, fh = 30;
          V.box(g, fx, fwY, fw, fh, { r: 8, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.5), lw: 1 });
          const nL = 8;
          for (let i = 0; i < nL; i++) { const X = fx + 8 + i * ((fw - 16) / nL); V.box(g, X, fwY + 6, (fw - 16) / nL - 4, fh - 12, { r: 4, fill: V.alpha(C.b, 0.2) }); }
          if (s === 4) for (let j = 0; j < 7; j++) { const u = ((e * 0.5 + j * 0.06) % 1); V.dot(g, fx + 8 + u * (fw - 16), fwY + 8 + j * 2.2, 2.4, RC[ROWS[j].ri]); }
          V.text(g, 'modelo · 7 linhas × camadas · passo misto → piecewise/eager', fx + 6, fwY + fh + 13, { size: 10, font: 'mono', color: C.b, maxW: fw });
        }
        /* saída */
        if (s >= 5) {
          const ox = mob ? outX : outX, oy = outY;
          V.text(g, s >= 6 ? 'ModelRunnerOutput' : 'sampler', ox, oy - 8, { size: 10.5, font: 'mono', color: C.c, weight: 600 });
          REQ.forEach((r, ri) => {
            const Y = oy + ri * 26;
            critter(g, ox + 8, Y + 9, 7, RC[ri], t);
            V.text(g, r.id + ': [' + SAMP[ri] + ']', ox + 20, Y + 13, { size: 11, font: 'mono', color: C.ink, weight: s >= 6 ? 700 : 400 });
          });
          if (s >= 6) V.text(g, 'encoder-only → []', ox, oy + 3 * 26 + 10, { size: 10, font: 'mono', color: C.muted });
        }
      });
    })();

    /* ================================================================
       Figura 5 — model registry
       ================================================================ */
    (function figRegistry() {
      const fig = q('#fig-registry');
      autoH(fig, 480, 770, 640);
      const ctl = UI.controls(fig);
      const MODELS = {
        glm: { name: 'GLM-5.3-Flash', rc: 'glm', cfg: ['"architectures":', '  ["Glm5NextForConditionalGeneration"]', '"model_type": "glm5_next"', 'text_config.model_type: "glm5_next_text"', '"num_experts_per_tok": 8', '"mla_use_nope": true'], mod: 'vllm/models/glm5next/nvidia/', cls: 'Glm5NextForConditionalGeneration → Glm5NextForCausalLM', conf: 'Glm5NextTextConfig (configs/glm5_next.py)', note: 'wrapper multimodal com IsHybrid e HasInnerState' },
        q0: { name: 'Qwen3.8 · day-0', rc: 'qwen', cfg: ['"architectures":', '  ["Qwen4ExpForConditionalGeneration"]', '"model_type": "qwen4_exp"', 'text_config.model_type: "qwen4_exp_text"'], mod: 'vllm/models/qwen3_8_flash_next/nvidia/', cls: 'classes Qwen3_8FlashNext* (ex.: Qwen3_8FlashNextForCausalLM)', conf: 'Qwen3_8FlashNextTextConfig', note: 'imagem day-0 da NVIDIA: os patches miram este caminho' },
        q30: { name: 'Qwen3.8 · v0.30', rc: 'qwen', cfg: ['"architectures":', '  ["Qwen4ExpForConditionalGeneration"]', '"model_type": "qwen4_exp"', 'text_config.model_type: "qwen4_exp_text"'], mod: 'vllm/models/qwen4_exp/nvidia/', cls: 'classes Qwen4Exp* (ex.: Qwen4ExpQSAFlashAttentionImpl)', conf: 'Qwen4ExpTextConfig', note: 'mesmo checkpoint, outro caminho: patches _v030 separados' },
        ds: { name: 'DeepSeek-V4-Flash', rc: 'ds', cfg: ['"architectures":', '  ["DeepseekV4ForCausalLM"]', '"model_type": "deepseek_v4"', '"dspark_block_size": 5', '"num_nextn_predict_layers": 3'], mod: 'vllm.models.deepseek_v4 → nvidia/ (amd.model no ROCm)', cls: 'DeepseekV4ForCausalLM', conf: 'config deepseek_v4', note: 'layout isolado por hardware (_resolve_module_name)' },
        dfl: { name: 'drafter DFlash2 (GLM)', rc: 'glm', cfg: ['"architectures":', '  ["DFlash2DraftModel"]', '"model_type": "qwen3"', '(checkpoint separado do drafter)'], mod: 'qwen3_dflash2', cls: 'DFlash2Qwen3ForCausalLM', conf: 'config qwen3', note: 'registrado pelo patch_dflash2.py (a imagem só conhecia o DFlash 1)' },
        dspB: { name: 'drafter DSpark (DS · lane B)', rc: 'ds', cfg: ['mesmo config.json do alvo, reescrito', 'no overlay Stage-C (hf_config_override):', 'model_type → "deepseek_v4_dspark"', 'architectures →', '  ["DeepSeekV4DSparkModel"]', 'n_predict = dspark_block_size = 5'], mod: 'vllm.models.deepseek_v4 (overlay Stage-C)', cls: 'DeepSeekV4DSpark', conf: 'config do alvo, sobrescrito (n_predict = 5)', note: 'registry.py:600 do overlay; k = 5, o bloco treinado, passa na regra do MTP' },
        dspA: { name: 'drafter DSpark (DS · lane A)', rc: 'ds', cfg: ['mesmo config.json do alvo, reescrito', 'na imagem Anemll (SpeculativeConfig):', 'deepseek_v4 → "deepseek_mtp"', 'n_predict = num_nextn_predict_layers = 3', 'depois: architectures → ["DSparkDraftModel"]', 'regra do MTP: k > 3 ⇒ k % 3 = 0 → k = 6'], mod: 'dentro da imagem: v1/worker/gpu/spec_decode/dspark/ + qwen3_dspark.py (Markov head)', cls: 'DSparkDraftModel', conf: 'config do alvo, sobrescrito (n_predict = 3)', note: 'k = 5 é recusado; MTP_NUM_TOKENS = 6 vai uma posição além do bloco treinado (5)' },
      };
      const BOOK = [
        ['Glm5NextForConditionalGeneration', 'glm5next', 'glm'],
        ['Qwen4ExpForConditionalGeneration', 'qwen…', 'q0'],
        ['DeepseekV4ForCausalLM', 'deepseek_v4', 'ds'],
        ['DeepSeekV4MTPModel', 'deepseek_v4', null],
        ['DeepSeekV4DSparkModel', 'deepseek_v4', 'dspB'],
        ['DSparkDraftModel', '(na imagem)', 'dspA'],
        ['DFlashDraftModel', 'qwen3_dflash', null],
        ['DFlash2DraftModel', 'qwen3_dflash2', 'dfl'],
        ['Eagle3LlamaForCausalLM', 'llama_eagle3', null],
      ];
      const S = { m: 'glm', alias: true, t0: performance.now() };
      UI.select(ctl, { label: 'Checkpoint', options: Object.keys(MODELS).map((k) => ({ v: k, label: MODELS[k].name })), value: 'glm', onChange: (v) => { S.m = v; S.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'aliases do Glm5NextTextConfig', value: true, onChange: (v) => { S.alias = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const M = MODELS[S.m], col = C[M.rc];
        const since = (performance.now() - S.t0) / 1000;
        let A, R, T, RT;
        if (!mob) {
          const top = 12, h1 = 194, gap = 40;
          A = { x: 12, y: top, w: (w - 24 - gap) * 0.42, h: h1 };
          R = { x: A.x + A.w + gap, y: top, w: w - 12 - (A.x + A.w + gap), h: h1 };
          T = { x: 12, y: top + h1 + 20, w: w - 24, h: 100 };
          RT = { x: 12, y: T.y + T.h + 14, w: w - 24, h: h - (T.y + T.h + 14) - 12 };
        } else {
          A = { x: 12, y: 10, w: w - 24, h: 128 }; R = { x: 12, y: 164, w: w - 24, h: 184 }; T = { x: 12, y: 374, w: w - 24, h: 152 };
          RT = { x: 12, y: 540, w: w - 24, h: h - 552 };
        }
        const panel = (P, title, c) => { V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(c, 0.05), stroke: V.alpha(c, 0.45), lw: 1.2 }); V.text(g, title, P.x + 12, P.y + 17, { size: 10.5, font: 'mono', color: c, weight: 700, maxW: P.w - 24 }); };
        panel(A, 'config.json', col);
        const cl = mob ? 15 : 24;
        const draft = S.m === 'dspA' || S.m === 'dspB';
        M.cfg.forEach((L, i) => {
          const key = !draft && (i < 3 || L.startsWith('"arch') || L.startsWith('  ['));
          const hot = draft && i >= 2;
          V.text(g, L, A.x + 12, A.y + 40 + i * cl, { size: 10, font: 'mono', color: hot ? col : key ? C.ink : C.muted, weight: hot ? 600 : 400, maxW: A.w - 16 });
        });
        panel(R, 'registry: architectures → (módulo, classe)', C.b);
        const lh = mob ? 17 : 18;
        const hit = BOOK.findIndex((b) => b[2] === S.m || (S.m === 'q30' && b[2] === 'q0'));
        BOOK.forEach((b, i) => {
          const Y = R.y + 26 + i * lh;
          const on = i === hit;
          if (on) V.box(g, R.x + 6, Y - 1, R.w - 12, lh - 1, { r: 6, fill: V.alpha(col, 0.22), stroke: col, lw: 1.5 });
          V.text(g, b[0], R.x + 12, Y + lh / 2 + 3, { size: mob ? 9.5 : 10, font: 'mono', color: on ? C.ink : C.muted, weight: on ? 700 : 400, maxW: R.w * 0.66 });
          V.text(g, b[1], R.x + R.w - 12, Y + lh / 2 + 3, { size: 9.5, font: 'mono', align: 'right', color: on ? col : V.alpha(C.muted, 0.85), maxW: R.w * 0.3 });
        });
        panel(T, 'código que roda (import preguiçoso)', C.a);
        g.save(); g.globalAlpha = V.clamp(since / 0.35, 0, 1);
        const lines = [['módulo', M.mod], ['classe', M.cls], ['config', M.conf], ['nota', M.note]];
        if (!mob) {
          const colW = (T.w - 36) / 2;
          [[lines[0], lines[1]], [lines[2], lines[3]]].forEach((pair, ci) => {
            let yy = T.y + 38;
            const x0 = T.x + 12 + ci * (colW + 12);
            pair.forEach(([k, v]) => {
              V.text(g, k, x0, yy, { size: 9.5, font: 'mono', color: C.muted });
              const n = wrap(g, v, x0 + 54, yy, colW - 58, 12, { size: 10, font: 'mono', color: k === 'nota' ? C.muted : C.ink, maxLines: k === 'nota' ? 3 : 2 });
              yy += Math.max(1, n) * 12 + 8;
            });
          });
        } else {
          let yy = T.y + 36;
          lines.forEach(([k, v]) => {
            V.text(g, k, T.x + 12, yy, { size: 9.5, font: 'mono', color: C.muted });
            const n = wrap(g, v, T.x + 62, yy, T.w - 72, 12, { size: 10, font: 'mono', color: k === 'nota' ? C.muted : C.ink, maxLines: k === 'módulo' ? 3 : 2 });
            yy += Math.max(1, n) * 12 + 5;
          });
        }
        g.restore();
        if (since < 1.4) { V.glow(g, T.x + T.w / 2, T.y + 24, 50, C.a, 0.45 * (1 - since / 1.4)); pill(g, T.x + T.w - 12, T.y + 16, 'import!', C.a, { align: 'right', size: 9.5 }); }
        const u = (t * 0.6) % 1;
        if (!mob) {
          const yA = A.y + A.h / 2;
          V.arrow(g, A.x + A.w + 4, yA, R.x - 4, yA, { color: col, lw: 1.8 });
          V.dot(g, V.lerp(A.x + A.w + 4, R.x - 9, u), yA, 3.2, col);
          const xR = R.x + R.w / 2;
          V.arrow(g, xR, R.y + R.h + 2, xR, T.y - 2, { color: col, lw: 1.8 });
          V.dot(g, xR, V.lerp(R.y + R.h + 2, T.y - 6, u), 3, col);
          V.text(g, 'import preguiçoso', xR + 10, (R.y + R.h + T.y) / 2 + 4, { size: 9.5, font: 'mono', color: C.muted });
        } else {
          [[A.y + A.h, R.y], [R.y + R.h, T.y]].forEach(([y1, y2], i) => {
            V.arrow(g, w / 2, y1 + 3, w / 2, y2 - 3, { color: col, lw: 1.8 });
            V.dot(g, w / 2, V.lerp(y1 + 3, y2 - 8, u), 3.2, col);
            V.text(g, i === 0 ? 'architectures' : 'import preguiçoso', w / 2 + 12, (y1 + y2) / 2 + 4, { size: 9.5, font: 'mono', color: C.muted });
          });
        }
        /* roteador do GLM */
        const on = S.alias;
        V.box(g, RT.x, RT.y, RT.w, RT.h, { r: 12, fill: V.alpha(C.glm, 0.05), stroke: V.alpha(C.glm, 0.45), lw: 1.2 });
        V.text(g, 'ROTEADOR DO GLM · uma camada MoE, 288 experts', RT.x + 12, RT.y + 17, { size: 10.5, font: 'mono', color: C.glm, weight: 700, maxW: RT.w - 24 });
        const k = on ? 8 : 7;
        const txt1 = on ? 'num_experts_per_tok: 8 (checkpoint) → config.num_experts_per_token = 8 ✓' : 'sem alias → default da classe: num_experts_per_token = 7 ✗';
        const txt2 = on ? 'mla_use_nope: true → mla_nope ligado ✓' : 'mla_use_nope ignorado → NoPE fica no default ✗';
        const txt3 = on ? '8 experts por token, como no treino' : '7 experts por token: o 8º some em silêncio (piscando). O modelo responde, só que pior.';
        const tcol = mob ? RT.w - 24 : RT.w * 0.43;
        let ty = RT.y + 38;
        ty += wrap(g, txt1, RT.x + 12, ty, tcol, 13, { size: 10.5, font: 'mono', color: on ? C.e : C.c, maxLines: 3 }) * 13 + 6;
        ty += wrap(g, txt2, RT.x + 12, ty, tcol, 13, { size: 10.5, font: 'mono', color: on ? C.e : C.c, maxLines: 2 }) * 13 + 8;
        ty += wrap(g, txt3, RT.x + 12, ty, tcol, 14, { size: 11.5, color: on ? C.ink : C.c, weight: 600, maxLines: 3 }) * 14;
        let gx0, gy0, gw, gh, cols;
        if (!mob) { gx0 = RT.x + RT.w * 0.47; gy0 = RT.y + 30; gw = RT.w * 0.53 - 14; gh = RT.h - 40; cols = 36; }
        else { gx0 = RT.x + 12; gy0 = ty + 6; gw = RT.w - 24; gh = RT.y + RT.h - 12 - gy0; cols = 32; }
        const rows = Math.ceil(288 / cols);
        const cs = Math.max(4, Math.min(gw / cols, gh / rows));
        const tok = Math.floor(t / 1.3);
        const picks = [];
        for (let i = 0; picks.length < 8 && i < 40; i++) { const ex = Math.floor(hash(tok, i + 1) * 288); if (!picks.includes(ex)) picks.push(ex); }
        for (let i = 0; i < 288; i++) {
          const X = gx0 + (i % cols) * cs, Y = gy0 + Math.floor(i / cols) * cs;
          const rank = picks.indexOf(i);
          let f = V.alpha(C.muted, 0.13);
          if (rank >= 0 && rank < k) f = V.alpha(C.glm, 0.95);
          else if (rank === 7 && !on) f = V.alpha(C.c, 0.35 + 0.45 * Math.abs(Math.sin(t * 6)));
          V.box(g, X + 0.5, Y + 0.5, cs - 1.5, cs - 1.5, { r: Math.min(3, cs / 4), fill: f });
        }
      });
    })();

    /* ================================================================
       Figura 6 — meta device
       ================================================================ */
    (function figMeta() {
      const fig = q('#fig-meta');
      autoH(fig, 390, 550, 640);
      const ctl = UI.controls(fig);
      const S = { nometa: false };
      UI.toggle(ctl, { label: 'sem meta device (materializar tudo)', value: false, onChange: (v) => { S.nometa = v; } });
      const steps = [
        { t: 'Construir no meta', d: '<code>with torch.device("meta"): initialize_model(...)</code>: o modelo inteiro existe, com formas e dtypes, mas nenhum byte. Só os construtores da <code>PleOffloadLayer</code> miram a CPU.' },
        { t: 'Descobrir as PleOffloadLayer', d: '<code>model.named_modules()</code> percorre a árvore em ordem (DFS), a mesma do forward da GPU, e acha a PLE na camada 1.' },
        { t: 'Filtrar o checkpoint', d: 'O iterador de pesos passa por um filtro de prefixo (aplicando o <code>hf_to_vllm_mapper</code> só para comparar). Só os tensores da PLE chegam a <code>load_weights</code>. Parâmetro materializado sem peso → <code>RuntimeError</code>.' },
        { t: 'Pós-carga só na subárvore', d: '<code>process_weights_after_loading</code> roda apenas nas PLE, na CPU. O resto continua no meta e <b>não pode ser visitado</b>.' },
        { t: 'Servir', d: 'Num mundo Gloo isolado (TP1), o processo calcula a PLE uma vez por rank de DP e a copia para o buffer de GPU de cada rank TP via CUDA IPC.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 6000 });
      const NAMES = [
        ['model.language_model.layers.0.mlp.experts.0.gate_proj.weight', false],
        ['model.language_model.layers.1.ple.ple_embedding.ngram_embedding.*', true],
        ['model.visual.blocks.3.attn.qkv.weight', false],
        ['model.language_model.layers.1.ple.key_proj.weight', true],
        ['lm_head.weight', false],
        ['model.language_model.layers.47.mlp.shared_expert.up_proj.weight', false],
        ['model.language_model.layers.1.ple.conv1d.weight', true],
        ['mtp.layers.0.mlp.experts.0.down_proj.weight', false],
      ];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const s = st.i, e = st.elapsed();
        let CK, TR, MEM;
        if (!mob) { CK = { x: 12, y: 30, w: w * 0.22, h: h - 60 }; TR = { x: w * 0.26, y: 30, w: w * 0.5, h: h - 60 }; MEM = { x: w * 0.79, y: 30, w: w * 0.21 - 12, h: h - 60 }; }
        else { CK = { x: 12, y: 10, w: w - 24, h: 74 }; TR = { x: 12, y: 104, w: w - 24, h: 300 }; MEM = { x: 12, y: 420, w: w - 24, h: h - 432 }; }
        /* checkpoint */
        V.box(g, CK.x, CK.y, CK.w, CK.h, { r: 12, fill: V.alpha(C.d, 0.05), stroke: V.alpha(C.d, 0.5), lw: 1.2 });
        V.text(g, 'safetensors (133 GB)', CK.x + 10, CK.y + 17, { size: 10.5, font: 'mono', color: C.d, weight: 700, maxW: CK.w - 20 });
        if (!mob) for (let i = 0; i < 9; i++) V.box(g, CK.x + 12, CK.y + 30 + i * 18, CK.w - 24, 12, { r: 3, fill: V.alpha(C.d, 0.12 + (i % 3) * 0.05) });
        else for (let i = 0; i < 10; i++) V.box(g, CK.x + 12 + i * ((CK.w - 24) / 10), CK.y + 30, (CK.w - 24) / 10 - 4, 30, { r: 3, fill: V.alpha(C.d, 0.12 + (i % 3) * 0.05) });
        /* árvore */
        V.box(g, TR.x, TR.y, TR.w, TR.h, { r: 12, fill: V.alpha(C.b, 0.03), stroke: V.alpha(C.b, 0.35), lw: 1 });
        V.text(g, 'Qwen4ExpForConditionalGeneration', TR.x + TR.w / 2, TR.y + 18, { size: 11, font: 'mono', align: 'center', color: C.ink, weight: 700, maxW: TR.w - 20 });
        const ghost = (X, Y, W, H, label, on, solid, col) => {
          if (solid) V.box(g, X, Y, W, H, { r: 5, fill: V.alpha(col, 0.6), stroke: col, lw: 1.5 });
          else { g.save(); g.setLineDash([3, 3]); V.box(g, X, Y, W, H, { r: 5, fill: on ? V.alpha(col, 0.15) : null, stroke: on ? col : V.alpha(C.muted, 0.55), lw: on ? 1.6 : 1 }); g.restore(); }
          if (label) V.text(g, label, X + W / 2, Y + H / 2 + 0.5, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: solid ? ink(C) : C.muted, maxW: W - 4 });
        };
        const matAll = S.nometa && s >= 3;
        const iy = TR.y + 34;
        const topW = (TR.w - 36) / 3;
        ghost(TR.x + 12, iy, topW, 24, 'visual · 27 blocos', s === 1 && e < 0.8, matAll, C.d);
        ghost(TR.x + 18 + topW, iy, topW, 24, 'embed_tokens', s === 1 && e >= 0.8 && e < 1.2, matAll, C.d);
        ghost(TR.x + 24 + 2 * topW, iy, topW, 24, 'lm_head · mtp', false, matAll, C.d);
        V.text(g, 'layers[0…47]', TR.x + 12, iy + 44, { size: 10, font: 'mono', color: C.muted });
        const cols = mob ? 8 : 12, rows = 48 / cols;
        const gw = TR.w - 24, cwid = gw / cols, gy = iy + 52, chh = Math.min(34, (TR.y + TR.h - 70 - gy) / rows);
        const scanIdx = s === 1 ? Math.floor(e * 14) : -1;
        const pleFound = s >= 1 && (s > 1 || scanIdx >= 1);
        for (let i = 0; i < 48; i++) {
          const X = TR.x + 12 + (i % cols) * cwid, Y = gy + Math.floor(i / cols) * chh;
          const isP = i === 1;
          const scanning = scanIdx % 48 === i;
          ghost(X + 2, Y + 2, cwid - 4, chh - 4, isP ? '' : String(i), scanning, matAll && !isP, C.d);
          if (isP) {
            const solid = s >= 3;
            const found = pleFound;
            if (found && !solid) V.glow(g, X + cwid / 2, Y + chh / 2, 26, C.a, 0.6);
            ghost(X + 2, Y + 2, cwid - 4, chh - 4, 'ple', found, solid, C.a);
          }
        }
        const legendY = TR.y + TR.h - 14;
        g.save(); g.setLineDash([3, 3]); V.box(g, TR.x + 12, legendY - 9, 14, 11, { r: 3, stroke: V.alpha(C.muted, 0.7), lw: 1 }); g.restore();
        V.text(g, 'meta: forma sem bytes', TR.x + 32, legendY, { size: 10, color: C.muted });
        V.box(g, TR.x + TR.w / 2 + 4, legendY - 9, 14, 11, { r: 3, fill: V.alpha(C.a, 0.6), stroke: C.a, lw: 1 });
        V.text(g, 'materializado (CPU)', TR.x + TR.w / 2 + 24, legendY, { size: 10, color: C.muted, maxW: TR.w / 2 - 30 });
        /* nomes do checkpoint fluindo */
        if (s === 2) {
          const pleX = TR.x + 12 + (1 % cols) * cwid + cwid / 2, pleY = gy + chh / 2;
          NAMES.forEach(([nm, ok], i) => {
            const per = 0.6, cyc = NAMES.length * per, span = 1.9;
            const u = (((e - i * per) % cyc) + cyc) % cyc / span;
            if (u > 1) return;
            let x0, y0;
            if (!mob) { x0 = CK.x + CK.w / 2; y0 = CK.y + 40 + (i % 8) * 34; } else { x0 = CK.x + CK.w / 2; y0 = CK.y + CK.h - 10; }
            let X, Y, a = 1;
            if (ok) { const ee = V.ease.inOut(u); X = V.lerp(x0, pleX, ee); Y = V.lerp(y0, pleY, ee) - Math.sin(u * Math.PI) * 30; }
            else { const mid = !mob ? [TR.x + 30, y0] : [x0, TR.y + 20]; if (u < 0.5) { const ee = u / 0.5; X = V.lerp(x0, mid[0], ee); Y = V.lerp(y0, mid[1], ee); } else { const ee = (u - 0.5) / 0.5; X = mid[0]; Y = mid[1] + ee * 60; a = 1 - ee; } }
            g.globalAlpha = a;
            const short = nm.replace('model.language_model.', '…').replace(/\.weight$/, '');
            const maxC = Math.max(20, Math.floor((mob ? CK.w : TR.w) / 6.2) - 4);
            const lab = short.length > maxC ? short.slice(0, maxC - 1) + '…' : short;
            const half = (tw(g, lab, 9.5, 'mono', 500) + 14) / 2;
            pill(g, V.clamp(X, half + 6, w - half - 6), Y, lab, ok ? C.a : C.muted, { size: 9.5, a: ok ? 0.25 : 0.1 });
            g.globalAlpha = 1;
          });
          V.text(g, 'prefixo “…layers.1.ple.” passa · o resto é descartado', TR.x + TR.w / 2, TR.y + TR.h - 30, { size: 10, align: 'center', color: C.a, weight: 600, maxW: TR.w - 20 });
        }
        /* memória e GPUs */
        V.box(g, MEM.x, MEM.y, MEM.w, MEM.h, { r: 12, fill: V.alpha(C.e, 0.04), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, 'RAM do nó · 121,69 GiB', MEM.x + 10, MEM.y + 17, { size: 10.5, font: 'mono', color: C.ink, weight: 700, maxW: MEM.w - 16 });
        const capG = 121.69;
        const grow = V.clamp(e / 2, 0, 1);
        const used = S.nometa ? 123.9 * (s === 0 ? 0.3 * grow : s === 1 ? 0.3 + 0.4 * grow : s === 2 ? 0.7 + 0.3 * grow : 1) : s >= 3 ? 47.7 : 0;
        const over = used > capG;
        const memMsg = over ? '✕ não cabe (ordem de grandeza: checkpoint inteiro)' : S.nometa ? 'materializando o checkpoint inteiro…' : s >= 3 ? 'tabela PLE FP8 ≈ 51,2 GB = 47,7 GiB' : 'quase nada: tudo é meta';
        if (!mob) {
          const bx = MEM.x + MEM.w / 2 - 22, bw = 44, by = MEM.y + 36, bh = MEM.h - 196;
          V.box(g, bx, by, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          const fh = Math.min(bh + 18, (used / capG) * bh);
          V.box(g, bx, by + bh - fh, bw, Math.max(2, fh), { r: 8, fill: over ? V.alpha(C.red || C.c, 0.8) : V.alpha(C.a, 0.8) });
          V.text(g, nf(used, 1) + ' GiB', bx + bw / 2, by + bh + 16, { size: 11, font: 'mono', align: 'center', color: over ? C.c : C.ink, weight: 700 });
          wrap(g, memMsg, MEM.x + MEM.w / 2, by + bh + 32, MEM.w - 16, 12, { size: 10, align: 'center', color: over ? C.c : C.muted, maxLines: 3 });
        } else {
          const bx = MEM.x + 10, bw = MEM.w - 20, by = MEM.y + 28, bh = 16;
          V.box(g, bx, by, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          V.box(g, bx, by, Math.max(4, Math.min(bw, (used / capG) * bw)), bh, { r: 8, fill: over ? V.alpha(C.red || C.c, 0.8) : V.alpha(C.a, 0.8) });
          V.text(g, nf(used, 1) + ' GiB · ' + (over ? '✕ não cabe (ordem de grandeza)' : S.nometa ? 'materializando tudo…' : s >= 3 ? 'tabela PLE FP8 ≈ 47,7 GiB' : 'quase nada: tudo é meta'), bx, by + bh + 15, { size: 10, font: 'mono', color: over ? C.c : C.muted, maxW: bw });
        }
        if (s >= 4) {
          const pleX = TR.x + 12 + (1 % cols) * cwid + cwid / 2, pleY = gy + chh / 2;
          const gpus = !mob ? [[MEM.x + 14, MEM.y + MEM.h - 64], [MEM.x + 14, MEM.y + MEM.h - 34]] : [[MEM.x + 10, MEM.y + MEM.h - 40], [MEM.x + MEM.w / 2 + 4, MEM.y + MEM.h - 40]];
          const gwid = !mob ? MEM.w - 28 : MEM.w / 2 - 14;
          gpus.forEach(([X, Y], i) => {
            V.box(g, X, Y, gwid, 24, { r: 6, fill: V.alpha(C.a, 0.15), stroke: C.a, lw: 1.2 });
            V.text(g, i === 0 ? 'buffer GPU · TP 0' : 'TP 1 (se houver)', X + gwid / 2, Y + 12.5, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: gwid - 6 });
            const u = ((t * 0.7 + i * 0.5) % 1);
            const pt = V.along(pleX, pleY, X + gwid / 2, Y, V.ease.inOut(u));
            V.dot(g, pt[0], pt[1], 3.5, C.a);
          });
          pill(g, (pleX + gpus[0][0]) / 2 + (mob ? 30 : 0), (pleY + gpus[0][1]) / 2, 'CUDA IPC', C.a, { size: 9.5 });
        }
      });
    })();

    /* ================================================================
       Figura 7 — os três ganchos da quantização
       ================================================================ */
    (function figQuant() {
      const fig = q('#fig-quant');
      autoH(fig, 440, 660, 640);
      const ctl = UI.controls(fig);
      const MET = {
        exl3: {
          lab: 'GLM · EXL3 (experts)', rc: 'glm', cfg: 'Exl3Config', prefix: 'prefix = "…layers.<n>.mlp.experts"',
          disp: [['RoutedExperts', 'Exl3MoEMethod', true], ['ParallelLMHead', 'Exl3LinearMethod se declarado · senão None', false], ['LinearBase', 'Exl3Linear | Glm53DenseFp8 | Unquantized', false]],
          ghost: 'KDA e MLA recebiam quant_config zerado no construtor: nunca eram consultados e ficavam BF16 (patch_dense_fp8.py corrige)',
          params: [['w13_trellis', 'int16', '[E, 2, K/16, I/16, 16·bits]', '[288, 2, 256, 64, 64]', 'a'], ['w13_suh', 'fp16', '[E, 2, H]', '[288, 2, 4096]', 'b'], ['w13_svh', 'fp16', '[E, 2, I]', '[288, 2, 1024]', 'b'], ['w2_trellis', 'int16', '[E, I/16, H/16, 16·bits]', '[288, 64, 256, 64]', 'a'], ['w2_suh · w2_svh', 'fp16', '[E, I] · [E, H]', '', 'b'], ['mcg', 'int32', 'marcador 0xCBAC1FED', '', 'd']],
          shapeNote: 'formas por rank em TP2: H = 4096 · I = I_local = 1024 (2048/TP2) · 4 bits → 16·bits = 64',
          pnote: 'nunca aloca w13_weight denso · ≈ 6 MiB por expert por rank a 4 bpw (conta nossa: 3 × 4096 × 1024 × 0,5 B)',
          shard: 'both', snote: 'gate/up por coluna (fatia I) · down por linha (fatia I na entrada) · suh/svh replicados do lado não fatiado',
          proc: ['valida o marcador MCG em todos os experts', 'detecta suh compartilhado entre gate e up', 'um LinearEXL3 por (expert, proj), sem reconstruir BF16', 'tabelas de ponteiros int64 + buffers temporários fp16'],
          thr: 32, perExpert: true, lo: 'nenhum expert pode receber mais de 32 linhas → um lançamento exl3_moe (experts “thin”, graph-safe)', hi: 'algum expert pode passar de 32 linhas: só esses (fat) vão ao E3 agrupado (3 lançamentos); os demais seguem no exl3_moe', anote: 'EXL3_TEMP_ROWS_FUSED = 32 ≥ MAX_NUM_SEQS × (k + 1) = 4 × 8: o decode inteiro cabe num lançamento graph-safe',
        },
        fp8: {
          lab: 'GLM · FP8 denso', rc: 'glm', cfg: 'Exl3Config (grupo GLM53_DENSE_FP8)', prefix: 'prefix = projeção densa do grupo (KDA, MLA q/o, shared, MLP)',
          disp: [['RoutedExperts', 'Exl3MoEMethod', false], ['ParallelLMHead', 'None (nativo)', false], ['LinearBase no grupo FP8', 'Glm53DenseFp8Method', true]],
          ghost: 'EXL3-denso e FP8 no mesmo módulo: RuntimeError em vez de escolher um vencedor',
          params: [['weight', 'bf16', '[out, in] (checkpoint BF16)', 'carregado como está', 'b']],
          pnote: 'a quantização acontece depois da carga, no gancho 4',
          shard: 'col', snote: 'fatiado como o linear BF16 original (coluna ou linha, conforme a camada)',
          proc: 'fp8',
          thr: 512, lo: 'Marlin W8A16 (lê 1 byte por peso)', hi: 'cópia BF16 · F.linear (in_proj do KDA, GLM53_KDA_BF16_LARGE_M=1)', anote: 'o limiar 512 vale para o in_proj do KDA; os demais densos ficam no Marlin',
        },
        qwen: {
          lab: 'Qwen · FP8 por canal', rc: 'qwen', cfg: 'ModelOptMixedPrecisionConfig (+ modelopt.diff)', prefix: 'prefix candidato: model. → model.language_model. · lm_head',
          disp: [['quant_algo FP8_PER_CHANNEL_PER_TOKEN', 'ModelOptFp8PcPtLinearMethod', true], ['quant_algo FP8_PB_WO', 'ModelOptFp8PbWoLinearMethod', false], ['experts NVFP4 · excluídos', 'métodos NVFP4 · None', false]],
          ghost: 'GatedResidual e lm_head construídos com quant_config=None: nunca consultados, ficam BF16 (model.diff, mtp.diff e hyperconnection.diff corrigem)',
          params: [['weight', 'e4m3', '[out, in]', 'FP8 no checkpoint', 'f'], ['weight_scale', 'f32', '[out]', 'uma escala por canal', 'e']],
          pnote: 'só funciona se o módulo recebeu quant_config (model.diff, mtp.diff, hyperconnection.diff)',
          shard: 'col', snote: 'escala por canal fatia exatamente como o peso (escalas em bloco fatiavam errado no ParallelLMHead)',
          proc: ['nada a desquantizar: pesos e escalas já prontos', 'prepara o layout do CUTLASS FP8 (cubins sm_120)', 'a escala da ativação é calculada por token a cada chamada'],
          thr: null, lo: 'CutlassFP8ScaledMMLinearKernel (W8A8, ativação quantizada por token)', hi: '', anote: 'lane experimental: ≈ 9,2 → 5,7 GiB lidos por passo esperados, nunca medidos em GPU',
        },
        gb10: {
          lab: 'DS · plugin NVFP4', rc: 'ds', cfg: 'ModelOptGb10HybridConfig (plugin)', prefix: 'super().get_quant_method(layer, prefix) → troca o kernel',
          disp: [['ModelOptNvFp4LinearMethod', 'mesmo método, kernel Gb10HybridNvFp4LinearKernel', true], ['demais métodos ModelOpt', 'inalterados', false], ['sem opt-in do usuário', 'não reivindica o checkpoint', false]],
          ghost: 'kernel híbrido indisponível nesta pilha (ex.: fora de cc 12.1): ValueError já no despacho, antes de servir',
          params: [['weight', 'uint8', '[out, in/2]', '2 valores FP4 por byte', 'c'], ['weight_scale', 'e4m3', '[out, in/16]', 'escala por bloco de 16', 'f'], ['weight_scale_2', 'f32', 'escalar', 'escala global', 'e'], ['input_scale', 'f32', 'escalar', 'escala da ativação', 'e']],
          pnote: 'exige checkpoint ModelOpt NVFP4 W4A4, que não é o formato do DeepSeek-V4',
          shard: 'col', snote: 'fatiado como os lineares ModelOpt NVFP4',
          proc: ['mantém DUAS cópias dos pesos FP4', 'shim Marlin (W4A16) + layout CUTLASS (W4A4)', 'falha rápido se não for cc 12.1'],
          thr: 128, lo: 'Marlin W4A16 (desempacota → BF16)', hi: 'CUTLASS W4A4 (scaled_fp4_quant + cutlass_scaled_fp4_mm)', anote: 'GB10_HYBRID_NVFP4_M_THRESHOLD = 128 · testado só no empacotamento',
        },
      };
      const S = { m: 'exl3', M: 8 };
      UI.select(ctl, { label: 'Método', options: Object.keys(MET).map((k) => ({ v: k, label: MET[k].lab })), value: 'exl3', onChange: (v) => { S.m = v; } });
      UI.slider(ctl, { label: 'Linhas M', min: 0, max: 13, step: 1, value: 3, fmt: (v) => nf(Math.pow(2, v)), onInput: (v) => { S.M = Math.pow(2, v); } });
      const steps = [
        { t: 'get_quant_method', d: 'Para cada camada do modelo, a <code>QuantizationConfig</code> recebe o módulo e o <code>prefix</code> e devolve o método dono daqueles pesos (ou <code>None</code> = nativo).' },
        { t: 'create_weights', d: 'Na construção do modelo, o método registra os <code>Parameter</code>s com formas e dtypes <b>do checkpoint</b> e anexa a cada um o seu <code>weight_loader</code>.' },
        { t: 'weight_loader', d: 'Na carga, cada tensor do checkpoint é recortado para o rank: colunas (column-parallel) ou linhas (row-parallel).' },
        { t: 'process_weights_after_loading', d: 'Uma vez, depois da carga: converter para o layout do kernel, calcular escalas, criar cópias auxiliares, liberar fontes.' },
        { t: 'apply', d: 'A cada forward, talvez dentro de um CUDA graph: o GEMM, sem alocar nem sincronizar. O kernel pode depender do número de linhas M da chamada.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const dcol = (C, k) => ({ a: C.a, b: C.b, c: C.c, d: C.d, e: C.e, f: C.f })[k] || C.a;
      /* demonstração FP8 real */
      const W = V.randMatrix(4, 6, 23, 0.02).map((r, i) => r.map((v, j) => (i === 2 && j === 4 ? v * 6 : v)));
      const SC = W.map((r) => Math.max(...r.map(Math.abs)) / 448);
      const QV = W.map((r, i) => r.map((v) => LA.num.encode('e4m3', v / SC[i]).v));
      const ERR = (() => { let se = 0, sx = 0; W.forEach((r, i) => r.forEach((v, j) => { const d = QV[i][j] * SC[i] - v; se += d * d; sx += v * v; })); return Math.sqrt(se / sx); })();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const M = MET[S.m], col = C[M.rc];
        const s = st.i, e = st.elapsed();
        V.text(g, M.cfg, 14, 18, { size: 11.5, font: 'mono', color: col, weight: 700, maxW: w - 28 });
        const Y0 = 34, H0 = h - Y0 - (mob ? 150 : 96);
        /* etapas */
        if (s === 0) {
          V.text(g, 'get_quant_method(layer, ' + M.prefix.replace('prefix = ', '') + ')', 14, Y0 + 12, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 28 });
          const rowH = mob ? 62 : 52;
          M.disp.forEach(([a, b, on], i) => {
            const Y = Y0 + 32 + i * rowH;
            const lw2 = mob ? w - 28 : w * 0.36;
            V.box(g, 14, Y, lw2, rowH - 14, { r: 9, fill: V.alpha(on ? col : C.muted, on ? 0.2 : 0.07), stroke: on ? col : V.alpha(C.muted, 0.4), lw: on ? 2 : 1 });
            V.text(g, a, 24, Y + (rowH - 14) / 2 + 4, { size: 11, font: 'mono', color: C.ink, weight: on ? 700 : 400, maxW: lw2 - 20 });
            if (!mob) {
              const x2 = w * 0.46;
              V.arrow(g, 14 + lw2 + 6, Y + (rowH - 14) / 2, x2 - 6, Y + (rowH - 14) / 2, { color: on ? col : V.alpha(C.muted, 0.5), lw: on ? 2 : 1.2 });
              if (on) { const u = (t * 0.8) % 1; V.dot(g, V.lerp(14 + lw2 + 6, x2 - 6, u), Y + (rowH - 14) / 2, 3.5, col); }
              V.box(g, x2, Y, w - x2 - 14, rowH - 14, { r: 9, fill: V.alpha(on ? col : C.muted, on ? 0.14 : 0.05), stroke: V.alpha(on ? col : C.muted, 0.5), lw: 1 });
              V.text(g, b, x2 + 10, Y + (rowH - 14) / 2 + 4, { size: 11, font: 'mono', color: on ? C.ink : C.muted, maxW: w - x2 - 34 });
            } else V.text(g, '→ ' + b, 24, Y + rowH - 20, { size: 10, font: 'mono', color: on ? col : C.muted, maxW: w - 48 });
            if (on) V.glow(g, 20, Y + (rowH - 14) / 2, 16, col, 0.6);
          });
          if (M.ghost) {
            const Y = Y0 + 32 + M.disp.length * rowH + 8, bh2 = mob ? 72 : 46;
            g.save(); g.setLineDash([5, 4]);
            V.box(g, 14, Y, w - 28, bh2, { r: 9, fill: V.alpha(C.c, 0.07 + 0.04 * Math.sin(t * 3)), stroke: V.alpha(C.c, 0.8), lw: 1.4 });
            g.restore();
            V.text(g, '⚠', 30, Y + bh2 / 2 + 5, { size: 15, align: 'center', color: C.c, weight: 700 });
            wrap(g, M.ghost, 46, Y + (mob ? 18 : 19), w - 76, 13, { size: 10.5, color: C.ink, maxLines: mob ? 5 : 3 });
          }
        } else if (s === 1) {
          const n = M.params.length;
          const colsN = mob ? 1 : n > 3 ? 2 : 1;
          const cellW = (w - 28 - (colsN - 1) * 14) / colsN, rowH = mob ? Math.min(74, (H0 - 10) / n) : Math.min(88, (H0 - 10) / Math.ceil(n / colsN));
          M.params.forEach(([nm, dt, sh, ex, ck], i) => {
            const cc = dcol(C, ck);
            const X = 14 + (i % colsN) * (cellW + 14), Y = Y0 + 6 + Math.floor(i / colsN) * rowH;
            const appear = V.clamp(e * 2.5 - i * 0.4, 0, 1);
            g.globalAlpha = appear;
            V.box(g, X, Y, cellW, rowH - 10, { r: 9, fill: V.alpha(cc, 0.08), stroke: V.alpha(cc, 0.6), lw: 1 });
            const gx = X + 10, gyy = Y + 8, gcell = Math.min(8, (rowH - 26) / 4);
            for (let a = 0; a < 4; a++) for (let b = 0; b < 6; b++) V.box(g, gx + b * (gcell + 2), gyy + a * (gcell + 2), gcell, gcell, { r: 1.5, fill: V.alpha(cc, 0.25 + 0.5 * hash(i * 31 + a, b)) });
            const tx = gx + 6 * (gcell + 2) + 10;
            V.text(g, nm, tx, Y + 18, { size: 11.5, font: 'mono', color: C.ink, weight: 700, maxW: X + cellW - tx - 60 });
            pill(g, X + cellW - 10, Y + 15, dt, cc, { align: 'right', size: 9.5 });
            V.text(g, sh, tx, Y + 34, { size: 10.5, font: 'mono', color: C.muted, maxW: X + cellW - tx - 8 });
            if (ex && rowH > 48) V.text(g, ex, tx, Y + 49, { size: 10, font: 'mono', color: cc, maxW: X + cellW - tx - 8 });
            g.globalAlpha = 1;
          });
          if (M.shapeNote) {
            const Yn = Y0 + 6 + Math.ceil(n / colsN) * rowH + 8;
            wrap(g, M.shapeNote, 14, Yn, w - 28, 13, { size: 10.5, font: 'mono', color: col, maxLines: 2 });
          }
        } else if (s === 2) {
          const n = 8, cs = Math.min(mob ? 16 : 24, (H0 - 60) / n);
          const both = M.shard === 'both';
          const mats = both ? [['gate/up · coluna', 'col'], ['down · linha', 'row']] : [['coluna (saída)', 'col']];
          const mw = n * cs;
          const spanW = (w - 28) / mats.length;
          mats.forEach(([lab, kind], mi) => {
            const X = 14 + mi * spanW + (spanW - mw) / 2 - (mob ? 0 : 30), Y = Y0 + 26;
            V.text(g, lab, X + mw / 2, Y - 8, { size: 10.5, font: 'mono', align: 'center', color: C.ink, weight: 600 });
            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
              const r1 = kind === 'col' ? j >= n / 2 : i >= n / 2;
              const cc = r1 ? C.b : C.a;
              const split = V.clamp(e / 1.5, 0, 1) * (mob ? 6 : 10);
              const dx = kind === 'col' ? (r1 ? split : 0) : 0, dy = kind === 'row' ? (r1 ? split : 0) : 0;
              V.box(g, X + j * cs + dx, Y + i * cs + dy, cs - 2, cs - 2, { r: 2, fill: V.alpha(cc, 0.3 + 0.4 * hash(i + mi * 9, j)) });
            }
            V.text(g, kind === 'col' ? 'N (saída) → fatiado' : 'N (saída) →', X, Y + n * cs + 22, { size: 9.5, font: 'mono', color: C.muted });
            V.text(g, 'K ↓', X - 8, Y + 12, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
            if (kind === 'row' && !mob) V.text(g, 'fatiado', X - 8, Y + 26, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
            if (kind === 'row' && mob) V.text(g, 'K ↓ fatiado', X, Y + n * cs + 36, { size: 9.5, font: 'mono', color: C.muted });
          });
          const legY = Y0 + 26 + 8 * Math.min(mob ? 16 : 24, (H0 - 60) / 8) + (mob ? 52 : 44);
          V.box(g, 14, legY - 9, 12, 12, { r: 3, fill: V.alpha(C.a, 0.7) }); V.text(g, 'rank 0', 32, legY + 1, { size: 10.5, color: C.ink });
          V.box(g, 90, legY - 9, 12, 12, { r: 3, fill: V.alpha(C.b, 0.7) }); V.text(g, 'rank 1', 108, legY + 1, { size: 10.5, color: C.ink });
          wrap(g, M.snote, 14, legY + 20, w - 28, 13, { size: 10.5, color: C.muted, maxLines: 3 });
        } else if (s === 3) {
          if (M.proc === 'fp8') {
            const cs = mob ? Math.min(50, (w - 60) / 6.4) : 56, chh = 26;
            const X1 = 14, Y1 = Y0 + 30;
            V.text(g, 'W (BF16)', X1, Y1 - 10, { size: 10.5, font: 'mono', color: C.ink, weight: 700 });
            const X2 = mob ? 14 : X1 + 6 * cs + 90, Y2 = mob ? Y1 + 4 * chh + 44 : Y1;
            V.text(g, 'q = RNE(W / s) em E4M3', X2, Y2 - 10, { size: 10.5, font: 'mono', color: C.f, weight: 700 });
            const prog = V.clamp(e / 3, 0, 1);
            for (let i = 0; i < 4; i++) {
              for (let j = 0; j < 6; j++) {
                const v = W[i][j];
                V.box(g, X1 + j * cs, Y1 + i * chh, cs - 3, chh - 3, { r: 3, fill: V.cmap(v / 0.12, 'div') });
                V.text(g, v.toFixed(3), X1 + j * cs + (cs - 3) / 2, Y1 + i * chh + chh / 2, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cs - 5 });
                const shown = (i * 6 + j) / 24 < prog;
                if (shown) {
                  const qv = QV[i][j];
                  V.box(g, X2 + j * cs, Y2 + i * chh, cs - 3, chh - 3, { r: 3, fill: V.alpha(C.f, 0.15 + 0.5 * Math.abs(qv) / 448), stroke: Math.abs(qv) === 448 ? C.f : null, lw: 1.5 });
                  V.text(g, nf(qv, Number.isInteger(qv) ? 0 : 2), X2 + j * cs + (cs - 3) / 2, Y2 + i * chh + chh / 2, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cs - 5 });
                }
              }
              const sx = mob ? X2 + 6 * cs + 4 : X1 + 6 * cs + 8;
              if (!mob) V.text(g, 's=' + SC[i].toExponential(1), sx, Y1 + i * chh + chh / 2 + 4, { size: 9.5, font: 'mono', color: C.e, maxW: 80 });
              else V.text(g, SC[i].toExponential(1), sx, Y2 + i * chh + chh / 2 + 4, { size: 9, font: 'mono', color: C.e, maxW: w - sx - 4 });
            }
            const yb = (mob ? Y2 : Y1) + 4 * chh + 22;
            const nb = wrap(g, 's = amax(|W| na linha) / 448 · o maior valor de cada linha vira exatamente ±448 (contorno) · depois, repack para o layout do Marlin (W8A16)', 14, yb, w - 28, 13, { size: 10.5, color: C.muted, maxLines: 3 });
            V.text(g, 'dequantizando (q·s): erro RMS relativo = ' + nf(ERR * 100, 2) + '% nesta matriz', 14, yb + nb * 13 + 12, { size: 11, font: 'mono', color: C.f, weight: 600, maxW: w - 28 });
          } else {
            const items = M.proc;
            items.forEach((L, i) => {
              const Y = Y0 + 14 + i * (mob ? 58 : 48);
              const appear = V.clamp(e * 2 - i * 0.5, 0, 1);
              g.globalAlpha = appear;
              V.box(g, 14, Y, w - 28, mob ? 46 : 38, { r: 9, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.5), lw: 1 });
              V.text(g, String(i + 1), 30, Y + (mob ? 27 : 23), { size: 13, font: 'display', color: col, weight: 700, align: 'center' });
              wrap(g, L, 46, Y + (mob ? 18 : 23), w - 76, 13, { size: 11.5, color: C.ink, maxLines: 2 });
              g.globalAlpha = 1;
            });
            if (S.m === 'gb10') {
              const Y = Y0 + 14 + items.length * 48 + 6;
              if (Y + 30 < Y0 + H0) { V.text(g, 'memória dos pesos NVFP4:', 14, Y + 12, { size: 10.5, color: C.muted }); V.box(g, 170, Y + 2, 60, 14, { r: 4, fill: V.alpha(C.c, 0.7) }); V.box(g, 234, Y + 2, 60, 14, { r: 4, fill: V.alpha(C.c, 0.7) }); V.text(g, '×2', 302, Y + 13, { size: 11, font: 'mono', color: C.c, weight: 700 }); }
            }
          }
        } else {
          const ax0 = 30, ax1 = w - 30, ay = Y0 + (mob ? 70 : 80);
          const lx = (m) => ax0 + (Math.log2(m) / 13) * (ax1 - ax0);
          V.line(g, ax0, ay, ax1, ay, { color: V.alpha(C.ink, 0.5), lw: 1.4 });
          [1, 8, 64, 512, 4096].forEach((m) => { V.line(g, lx(m), ay - 4, lx(m), ay + 4, { color: C.muted }); V.text(g, nf(m), lx(m), ay + 16, { size: 9.5, font: 'mono', align: 'center', color: C.muted }); });
          const axLab = M.perExpert ? (mob ? 'M da chamada; o limiar vale por expert (linhas roteadas a cada um)' : 'M da chamada (tokens do passo que passam por esta camada); o limiar vale por expert: linhas roteadas a cada um') : 'M = linhas na chamada (tokens do passo que passam por esta camada)';
          wrap(g, axLab, ax0, ay - (mob ? 50 : 44), ax1 - ax0, 12, { size: 10.5, color: C.muted, maxLines: 2 });
          V.text(g, mob ? '← memória (decode)' : '← memória manda (decode)', ax0, ay - 22, { size: 10, font: 'mono', color: C.b });
          V.text(g, mob ? 'computação (prefill) →' : 'computação manda (prefill) →', ax1, ay - 22, { size: 10, font: 'mono', align: 'right', color: C.d });
          const cur = S.M;
          let hiSide = false;
          if (M.thr) {
            const tx = lx(M.thr);
            V.line(g, tx, ay - 12, tx, ay + 26, { color: col, lw: 2, dash: [4, 3] });
            V.text(g, 'limiar ' + nf(M.thr) + (M.perExpert ? ' (por expert)' : ''), tx, ay + 38, { size: 10, font: 'mono', align: 'center', color: col, weight: 700 });
            hiSide = S.m === 'exl3' || S.m === 'fp8' ? cur > M.thr : cur >= M.thr;
          }
          const cx = lx(cur);
          g.fillStyle = C.c; g.beginPath(); g.moveTo(cx, ay - 6); g.lineTo(cx - 7, ay - 16); g.lineTo(cx + 7, ay - 16); g.closePath(); g.fill();
          const by = ay + 58, bh = mob ? 84 : 60;
          if (M.thr) {
            const bw2 = mob ? w - 28 : (w - 42) / 2;
            [[M.lo, !hiSide, C.b], [M.hi, hiSide, C.d]].forEach(([txt, on, cc], i) => {
              const X = mob ? 14 : 14 + i * (bw2 + 14), Y = mob ? by + i * (bh + 10) : by;
              if (on) V.glow(g, X + bw2 / 2, Y + bh / 2, bw2 * 0.22, cc, 0.18);
              V.box(g, X, Y, bw2, bh, { r: 10, fill: V.alpha(cc, on ? 0.22 : 0.05), stroke: on ? cc : V.alpha(cc, 0.35), lw: on ? 2 : 1 });
              wrap(g, (i === 0 ? 'M ' + (S.m === 'exl3' || S.m === 'fp8' ? '≤ ' : '< ') : 'M ' + (S.m === 'exl3' || S.m === 'fp8' ? '> ' : '≥ ')) + nf(M.thr) + ': ' + txt, X + 12, Y + 20, bw2 - 24, 14, { size: 11, font: 'mono', color: on ? C.ink : C.muted, weight: on ? 600 : 400, maxLines: mob ? 5 : 3 });
            });
          } else {
            const X = 14, bw2 = w - 28;
            V.glow(g, X + bw2 / 2, by + bh / 2, bw2 * 0.18, C.a, 0.18);
            V.box(g, X, by, bw2, bh, { r: 10, fill: V.alpha(C.a, 0.2), stroke: C.a, lw: 2 });
            wrap(g, 'qualquer M: ' + M.lo, X + 12, by + 22, bw2 - 24, 14, { size: 11, font: 'mono', color: C.ink, weight: 600, maxLines: 3 });
          }
          const ny = mob ? by + 2 * (bh + 10) + 6 : by + bh + 22;
          const nl = wrap(g, M.anote, 14, ny, w - 28, 13, { size: 10.5, color: C.muted, maxLines: 3 });
          const CY0 = ny + nl * 13 + 22, CY1 = h - (mob ? 60 : 44) - (mob ? 46 : 32);
          if (M.thr && CY1 - CY0 > 60) {
            const T0 = M.thr;
            const fLo = (m) => 1 + (2 * m) / T0, fHi = (m) => 2 + m / T0;
            const lt = (v) => Math.log(v), tMin = lt(1), tMax = lt(fLo(8192));
            const yv = (v) => CY1 - ((lt(v) - tMin) / (tMax - tMin)) * (CY1 - CY0);
            V.text(g, 'tempo por chamada × M (forma qualitativa, não medida)', ax0, CY0 - 8, { size: 10, font: 'mono', color: C.muted, maxW: ax1 - ax0 });
            V.line(g, ax0, CY1, ax1, CY1, { color: V.alpha(C.muted, 0.4), lw: 1 });
            [[fLo, C.b], [fHi, C.d]].forEach(([f, cc]) => {
              g.beginPath();
              for (let i = 0; i <= 60; i++) { const m = Math.pow(2, (13 * i) / 60); const X = lx(m), Y = yv(f(m)); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); }
              g.strokeStyle = V.alpha(cc, 0.45); g.lineWidth = 1.5; g.stroke();
            });
            g.beginPath();
            for (let i = 0; i <= 60; i++) { const m = Math.pow(2, (13 * i) / 60); const X = lx(m), Y = yv(Math.min(fLo(m), fHi(m))); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); }
            g.strokeStyle = C.ink; g.lineWidth = 2.5; g.stroke();
            V.dot(g, lx(cur), yv(Math.min(fLo(cur), fHi(cur))), 5, C.c);
            let lgx = ax0, lgy = CY1 + 15;
            [[C.b, 'kernel de M pequeno', 1.5], [C.d, 'kernel de M grande', 1.5], [C.ink, 'despacho = envoltória inferior', 2.5]].forEach(([cc, lab, lw]) => {
              const iw = 22 + tw(g, lab, 9.5, 'mono', 400) + 16;
              if (lgx + iw > ax1 && lgx > ax0) { lgx = ax0; lgy += 13; }
              V.line(g, lgx, lgy - 3, lgx + 16, lgy - 3, { color: cc, lw });
              V.text(g, lab, lgx + 22, lgy, { size: 9.5, font: 'mono', color: cc });
              lgx += iw;
            });
          }
        }
        /* rodapé: linha do tempo dos ganchos */
        const fy = h - (mob ? 60 : 44);
        const names = mob ? ['dispatch', 'create', 'loader', 'post-load', 'apply'] : ['get_quant_method', 'create_weights', 'weight_loader', 'process_weights…', 'apply'];
        const when = ['construção', 'construção', 'carga', 'uma vez', 'todo forward'];
        const segW = (w - 28) / 5;
        names.forEach((nm, i) => {
          const X = 14 + i * segW, on = i === s;
          V.box(g, X + 2, fy, segW - 4, mob ? 40 : 30, { r: 8, fill: V.alpha(on ? col : C.muted, on ? 0.25 : 0.07), stroke: on ? col : V.alpha(C.muted, 0.3), lw: on ? 1.8 : 1 });
          V.text(g, nm, X + segW / 2, fy + (mob ? 16 : 13), { size: mob ? 9 : 10, font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400, maxW: segW - 8 });
          V.text(g, when[i], X + segW / 2, fy + (mob ? 30 : 25), { size: 9, align: 'center', color: C.muted, maxW: segW - 8 });
        });
        if (M.pnote && s === 1) {
          if (!mob) V.text(g, M.pnote, 14, fy - 10, { size: 10, color: col, maxW: w - 28 });
          else wrap(g, M.pnote, 14, fy - 36, w - 28, 13, { size: 10.5, color: col, maxLines: 3 });
        }
      });
    })();

    /* ================================================================
       Figura 8 — MoE modular e workspace
       ================================================================ */
    (function figMoe() {
      const fig = q('#fig-moe');
      autoH(fig, 510, 750, 640);
      const ctl = UI.controls(fig);
      const S = { M: 1024, grow: false, dbo: false };
      UI.slider(ctl, { label: 'Tokens no passo', min: 0, max: 13, step: 1, value: 10, fmt: (v) => nf(Math.pow(2, v)), onInput: (v) => { S.M = Math.pow(2, v); } });
      UI.toggle(ctl, { label: 'pedir memória durante a captura', value: false, onChange: (v) => { S.grow = v; } });
      UI.toggle(ctl, { label: 'DBO no MoE cooperativo do GLM', value: false, onChange: (v) => { S.dbo = v; } });
      const rI = UI.readout(ctl, 'índices #50298 (FlashMLA)', '—');
      const fmtB = (b) => (b >= 1048576 ? nf(b / 1048576, 1) + ' MiB' : nf(b / 1024, 0) + ' KiB');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        /* cadeia de seleção (DeepSeek) */
        const chain = [['--moe-backend flashinfer_b12x', C.d], ['oráculo map_mxfp4_backend', C.b], ['B12X_MXFP4', C.b], ['B12xExperts', C.a]];
        let cy = 14;
        if (!mob) {
          const cw = (w - 28 - 3 * 26) / 4;
          chain.forEach(([s, c], i) => {
            const X = 14 + i * (cw + 26);
            V.box(g, X, cy, cw, 30, { r: 8, fill: V.alpha(c, 0.14), stroke: V.alpha(c, 0.7), lw: 1.2 });
            V.text(g, s, X + cw / 2, cy + 15.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cw - 8 });
            if (i < 3) V.arrow(g, X + cw + 3, cy + 15, X + cw + 23, cy + 15, { color: C.muted, lw: 1.5, head: 6 });
          });
          cy += 46;
        } else {
          const cw = (w - 28 - 14) / 2;
          chain.forEach(([s, c], i) => {
            const X = 14 + (i % 2) * (cw + 14), Y = cy + Math.floor(i / 2) * 38;
            V.box(g, X, Y, cw, 30, { r: 8, fill: V.alpha(c, 0.14), stroke: V.alpha(c, 0.7), lw: 1.2 });
            V.text(g, s, X + cw / 2, Y + 15.5, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cw - 8 });
          });
          cy += 84;
        }
        /* pipeline FusedMoEKernel da DeepSeek */
        const py = cy, ph = mob ? 186 : 150;
        const stg = mob
          ? [['prepare', 'sem EP: dispatch trivial (o TP já replica os tokens)', C.b], ['experts.apply', 'normaliza ids e pesos (int32/fp32) · GEMMs W4A16', C.a], ['finalize', 'y = Σₖ wₖ · expertₖ(x)', C.b]]
          : [['prepare', 'sem EP: dispatch trivial (tokens já replicados por TP)', C.b], ['experts.apply', 'normaliza topk_ids/pesos → int32/fp32 · GEMMs W4A16 (dequant FP4 → MMA BF16)', C.a], ['finalize', 'y = Σₖ wₖ · expertₖ(x) (TopKWeightAndReduce)', C.b]];
        const sw = (w - 28 - 2 * 18) / 3, dl = mob ? 6 : 3;
        stg.forEach(([a, b, c], i) => {
          const X = 14 + i * (sw + 18);
          V.box(g, X, py, sw, ph, { r: 12, fill: V.alpha(c, 0.06), stroke: V.alpha(c, 0.55), lw: 1.2 });
          V.text(g, a, X + sw / 2, py + 17, { size: 11.5, font: 'mono', align: 'center', color: C.ink, weight: 700, maxW: sw - 10 });
          wrap(g, b, X + sw / 2, py + 33, sw - 14, 12, { size: 10, align: 'center', color: C.muted, maxLines: dl });
        });
        V.text(g, 'FusedMoEKernel (modular) · DeepSeek', w - 14, py - 6, { size: 10, font: 'mono', align: 'right', color: C.muted });
        const eX = 14 + sw + 18 + 10, eY = py + 33 + dl * 12 + 4, eW = sw - 20;
        const ecols = 8, ecs = Math.min(eW / ecols, 12);
        for (let i = 0; i < 16; i++) V.box(g, eX + (i % ecols) * ecs, eY + Math.floor(i / ecols) * ecs, ecs - 2, ecs - 2, { r: 2, fill: V.alpha(C.a, 0.15 + 0.6 * Math.max(0, Math.sin(t * 3 + i * 1.7))) });
        const scY = eY + 2 * ecs + 6;
        V.box(g, eX, scY, eW, 18, { r: 4, fill: V.alpha(C.a, 0.18), stroke: V.alpha(C.a, 0.75), lw: 1 });
        V.text(g, mob ? 'workspace2' : 'workspace2 (scratch b12x)', eX + eW / 2, scY + 9.5, { size: mob ? 10 : 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: eW - 6 });
        for (let k = 0; k < 5; k++) {
          const u = (t * 0.22 + k / 5) % 1;
          critter(g, 14 + u * (w - 28), py + ph - 12, 6, C.a, t, 'still');
        }
        /* outra receita: MoE cooperativo do GLM e o DBO */
        const gy = py + ph + 12, gh = mob ? 88 : 64;
        V.box(g, 14, gy, w - 28, gh, { r: 12, fill: V.alpha(C.glm, 0.05), stroke: V.alpha(C.glm, 0.5), lw: 1.2 });
        const coopT = 'GLM · MoE cooperativo: scratch + contadores por device';
        if (!mob) V.text(g, coopT, 26, gy + 17, { size: 10.5, font: 'mono', color: C.glm, weight: 700, maxW: w * 0.55 });
        else wrap(g, coopT, 26, gy + 17, w - 52, 12, { size: 10.5, font: 'mono', color: C.glm, weight: 700, maxLines: 2 });
        const cbW = mob ? Math.min(150, w * 0.44) : 210, cbX = w / 2 - cbW / 2, cbY = gy + (mob ? 40 : 26), cbH = mob ? 30 : 30;
        const cyc = (t * 0.25) % 1;
        const collide = S.dbo && Math.sin(t * 2.2) > 0.2;
        if (collide) V.glow(g, w / 2, cbY + cbH / 2, 46, C.c, 0.55);
        V.box(g, cbX, cbY, cbW, cbH, { r: 6, fill: V.alpha(collide ? C.c : C.f, collide ? 0.3 : 0.16), stroke: collide ? C.c : V.alpha(C.f, 0.75), lw: collide ? 2 : 1.2 });
        V.text(g, mob ? 'scratch + contadores' : 'scratch + contadores (por device)', w / 2, cbY + cbH / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cbW - 8 });
        const laneY = [cbY + 8, cbY + cbH - 8];
        [[C.a, 'A'], [C.c, 'B']].forEach(([c, lab], li) => {
          const act = S.dbo ? true : li === 0 ? cyc < 0.5 : cyc >= 0.5;
          if (!act) return;
          const base = S.dbo ? (t * 0.25 + li * 0.13) % 1 : (cyc % 0.5) * 2;
          for (let k = 0; k < 3; k++) {
            const u = (base + k * 0.06) % 1, X = 34 + u * (w - 60);
            if (X > cbX - 6 && X < cbX + cbW + 6) continue;
            critter(g, X, laneY[li], 5.5, c, t, 'still');
          }
          V.text(g, lab, 20, laneY[li] + 4, { size: 10, font: 'mono', color: c, weight: 700 });
        });
        if (collide) V.text(g, '✕', w / 2, cbY - 4, { size: 16, align: 'center', color: C.c, weight: 700 });
        const stt = S.dbo ? '✕ DBO intercala A e B no mesmo scratch: o adaptador recusa' : 'A termina, B começa: chamadas ordenadas ✓';
        if (!mob) V.text(g, stt, w - 26, gy + 17, { size: 10.5, align: 'right', color: S.dbo ? C.c : C.e, weight: 600, maxW: w * 0.4 });
        else V.text(g, stt, w / 2, gy + gh - 9, { size: 10, align: 'center', color: S.dbo ? C.c : C.e, weight: 600, maxW: w - 40 });
        /* workspace ao longo do boot */
        const wy = gy + gh + 12, wh = h - wy - 12;
        V.box(g, 14, wy, w - 28, wh, { r: 12, fill: V.alpha(C.e, 0.04), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, 'WORKSPACE MANAGER · uma região por device', 26, wy + 17, { size: 10.5, font: 'mono', color: C.muted, weight: 700, maxW: mob ? w - 52 : w * 0.6 });
        if (!mob) V.text(g, 'fatias: proporções ilustrativas', w - 26, wy + 17, { size: 10, align: 'right', color: C.muted });
        const MNBT = 8192;
        const idxB = (m) => m * 640 * 4;
        const loop = (t % 12) / 12, cur = Math.min(3, Math.floor(loop * 4)), inPh = (loop * 4) % 1;
        const RX = 26, RY = wy + (mob ? 46 : 38), RW = mob ? 110 : 170, RH = wy + wh - 12 - RY;
        if (mob) V.text(g, 'fatias: proporções ilustrativas', RX, RY - 7, { size: 9.5, color: C.muted });
        const fillF = cur === 0 ? V.ease.out(V.clamp(inPh * 1.3, 0, 1)) : 1;
        V.box(g, RX, RY, RW, RH, { r: 8, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.4), lw: 1 });
        const parts = [
          [mob ? ['workspace2'] : ['workspace2', 'scratch b12x'], C.a, 0.40, false],
          [mob ? ['split-K SM120'] : ['split-K (SM120)', 'mid_out · mid_lse'], C.b, 0.36, false],
          [mob ? ['#50298 (inerte?)'] : ['#50298 · FlashMLA', 'pode ficar inerte no GB10'], C.e, 0.24, true],
        ];
        let yy = RY + RH;
        parts.forEach(([nm, c, f, dashed]) => {
          const hh = RH * f * fillF;
          if (hh > 1) {
            if (dashed) { g.save(); g.setLineDash([4, 3]); }
            V.box(g, RX + 3, yy - hh + 1, RW - 6, hh - 2, { r: 4, fill: V.alpha(c, dashed ? 0.16 : 0.5), stroke: V.alpha(c, 0.9), lw: 1 });
            if (dashed) g.restore();
            const lines = hh > 14 * nm.length + 6 ? nm : hh > 16 ? [nm[0]] : [];
            lines.forEach((L, li) => V.text(g, L, RX + RW / 2, yy - hh / 2 + 4 + (li - (lines.length - 1) / 2) * 13, { size: mob ? 9.5 : 10, font: 'mono', align: 'center', color: C.ink, maxW: RW - 10 }));
          }
          yy -= RH * f * fillF;
        });
        const locked = cur >= 1;
        if (locked) {
          const lx = RX + RW + (mob ? 13 : 18), ly = RY + 10;
          g.strokeStyle = C.d; g.lineWidth = 2.2; g.beginPath(); g.arc(lx, ly - 2, 5, Math.PI, 0); g.stroke();
          V.box(g, lx - 7.5, ly - 2, 15, 11, { r: 2, fill: C.d });
        }
        if (cur === 3) {
          const dy = RY + RH - (S.M / MNBT) * RH;
          g.save(); g.strokeStyle = C.c; g.lineWidth = 2.2; g.setLineDash([5, 3]);
          g.beginPath(); g.moveTo(RX - 6, dy); g.lineTo(RX + RW + 4, dy); g.stroke(); g.restore();
          const ly2 = Math.max(RY + 12, dy - 5);
          V.text(g, nf(S.M) + ' tok', RX + RW - 6, ly2, { size: 10, font: 'mono', align: 'right', color: C.c, weight: 700 });
        }
        const growing = S.grow && cur === 2;
        if (growing) {
          const spike = 9 + 5 * Math.sin(t * 10);
          V.box(g, RX + 3, RY - spike, RW - 6, spike, { r: 3, fill: V.alpha(C.red || C.c, 0.85) });
          V.glow(g, RX + RW / 2, RY - spike / 2, 34, C.c, 0.6);
        }
        /* fases */
        const px0 = RX + RW + (mob ? 26 : 50), pw = w - 26 - px0;
        const PH = [
          ['dummy run', 'O modelo roda nas formas de pior caso (MNBT = 8192). Cada kernel pede seus buffers de uma vez (get_simultaneous) e a região cresce até o pico.'],
          ['lock_workspace()', 'O tamanho congela. Esse rascunho já foi medido no profile run e descontado do KV cache.'],
          ['captura', 'Os CUDA graphs gravam os endereços do workspace. Alocar ou crescer aqui é proibido.'],
          ['serviço', 'Qualquer passo com até 8192 tokens usa a mesma região, sem alocar nada.'],
        ];
        const chipW = mob ? pw : (pw - 18) / 4;
        PH.forEach(([nm], i) => {
          const X = px0 + (mob ? 0 : i * (chipW + 6)), Y = RY + (mob ? i * 28 : 0);
          const on = i === cur;
          V.box(g, X, Y, chipW, 24, { r: 8, fill: V.alpha(on ? C.d : C.muted, on ? 0.25 : 0.07), stroke: on ? C.d : V.alpha(C.muted, 0.3), lw: on ? 1.8 : 1 });
          V.text(g, (i + 1) + '. ' + nm, X + chipW / 2, Y + 12.5, { size: mob ? 9.5 : 10.5, font: 'mono', align: 'center', base: 'middle', color: on ? C.ink : C.muted, weight: on ? 700 : 400, maxW: chipW - 8 });
          if (on) V.box(g, X + 4, Y + 20, (chipW - 8) * inPh, 2.5, { r: 1, fill: C.d });
        });
        const dy0 = RY + (mob ? 132 : 44);
        if (growing) wrap(g, '✕ pedir memória dentro da captura: RuntimeError. O graph gravou endereços, e o alocador não pode se mexer.', px0, dy0, pw, 15, { size: 12, color: C.c, weight: 700, maxLines: mob ? 9 : 4 });
        else wrap(g, PH[cur][1], px0, dy0, pw, 15, { size: 12, color: C.ink, maxLines: mob ? 9 : 5 });
        const iy = RY + RH - 4;
        const idxTxt = 'índices #50298 (caminho FlashMLA) = M × 640 × 4 B → ' + fmtB(idxB(S.M)) + ' (conta nossa)';
        if (!mob) V.text(g, idxTxt, px0, iy, { size: 10, font: 'mono', color: C.muted, maxW: pw });
        else wrap(g, idxTxt, px0, iy - 36, pw, 12, { size: 9.5, font: 'mono', color: C.muted, maxLines: 4 });
        rI.set(fmtB(idxB(S.M)));
      });
    })();

    /* ================================================================
       Figura 9 — lanes e o bolo da imagem
       ================================================================ */
    (function figLanes() {
      const fig = q('#fig-lanes');
      autoH(fig, 440, 740, 640);
      const ctl = UI.controls(fig);
      const LANES = {
        glm: { lab: 'GLM', rc: 'glm', base: 'vllm/vllm-openai:glm53-flash-arm64-cu130', dig: '@sha256:905c0293…', ver: '0.1.dev20051+g487ecf187', mod: 'Dockerfile: FROM + RUN patchers + COPY exl3.py (build) · 25 overlays no start', when: 'build + start (nos dois ranks)', runner: 'o repositório da receita não registra', code: 'vllm/models/glm5next/nvidia/', envs: 'VLLM_* stock; GLM53_*/EXL3_* lidas pelos patches', strat: 'anchor', st: 'patchers por âncora + fixtures *-487ecf187 e um teste por patch' },
        tf: { lab: 'GLM · TF', rc: 'tf', base: 'nvcr.io/nvidia/pytorch:26.07-py3', dig: '+ TensorFold v0.6.0 (pip, tag git)', ver: '', idParts: [['v0.6.0', 'tag git do TensorFold', 'b'], ['-ae8d1c789b47', 'SHA-256 dos 53 patches + extras (12 hex)', 'd'], ['@sha256:22789f0c…', 'digest: o que de fato é puxado', 'a']], cmd: 'tensorfold serve …', cmdSub: 'mesmas configurações do motor nos dois ranks, conferidas na partida', mod: '53 diffs no site-packages: patch -p0 --forward, em ordem de nome, + teste de import', when: 'build da imagem (nada é aplicado no boot)', runner: 'não há: engine CUDA da família glm5_next, dois ranks', code: 'tensorfold/families/glm5_next/cuda/', envs: 'TENSORFOLD_* · TF_GLM_* · TF_ROCE_* com -e nos dois containers', strat: 'build', st: 'label tf.patches = hash dos patches; imagem puxada por digest' },
        q0: { lab: 'Qwen day-0', rc: 'qwen', base: 'vllm/vllm-openai:qwen38-flash-next', dig: '@sha256:d464f3b4… · ~20,6 GB', ver: '0.1.dev20073+g8e685d198', mod: 'extrair .orig da imagem → patch por âncora → bind-mount :ro (~10 overlays)', when: 'a cada launch, sem rebuild', runner: 'V2 (engine V1)', code: 'vllm/models/qwen3_8_flash_next/nvidia/', envs: 'VLLM_MTP_DRAFT_VOCAB, VLLM_QSA_* lidas pelos overlays', strat: 'anchor', st: 'âncora exata (count == 1) + ast.parse; recusa aplicar se a imagem já tem a correção' },
        q30: { lab: 'Qwen v0.30', rc: 'qwen', base: 'vllm/vllm-openai:v0.30.0', dig: 'release estável (2026-09-22)', ver: '0.30.0', mod: 'só o vocabulário do draft + FP8 KV opcional (backport vllm#55557)', when: 'a cada launch', runner: 'V2 (padrão desde a v0.29)', code: 'vllm/models/qwen4_exp/nvidia/', envs: 'stock v0.30 (VLLM_ATTENTION_BACKEND não existe mais)', strat: 'upstream', st: 'backport com detecção de “já aplicado”; apagar ao ir para a 0.31+' },
        dsA: { lab: 'DS · A Anemll', rc: 'ds', base: 'ghcr.io/anemll/dspark-vllm-gx10:0.1.1', dig: '@sha256:a8394849… · imagem externa', ver: '0.25.2.dev0+g752a3a504.d20260714', mod: 'hotfixes no boot: patchers source-exact (SHA-256), os.replace atômico, exit 1 na falha', when: 'boot do contêiner, antes do exec vllm serve', runner: 'V2 (v1/worker/gpu/…)', code: 'vllm/models/deepseek_v4/nvidia/ + v1/worker/gpu/spec_decode/dspark', envs: 'VLLM_DSPARK_* → “Unknown”, ignoradas', strat: 'anchor', st: 'source-exact: SHA-256 do arquivo-alvo fixado; recusa código diferente' },
        dsB: { lab: 'DS · B Stage-C', rc: 'ds', base: 'ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready', dig: 'fork “unholy-fusion” → dspark-nvfp4-stage-c', ver: '', mod: 'COPY de 17 arquivos inteiros + Dockerfiles Stage A/B/C (string-replace)', when: 'build da imagem', runner: 'V1 (gpu_model_runner.py)', code: '/opt/env/…/vllm/models/deepseek_v4/nvidia/', envs: 'envs.py do overlay registra VLLM_DSPARK_*', strat: 'overlay', st: 'arquivos inteiros (COPY) + py_compile + autoteste de um mapeamento' },
        dsC: { lab: 'DS · C main', rc: 'ds', base: 'vllm-dspark-runtime:official-main-dspark-00eb7ce', dig: 'main oficial com DSpark', ver: '', mod: 'COPY de arquivos python + patch b12x/nvfp4 + teste de sanidade', when: 'build da imagem', runner: 'o do main (não registrado)', code: 'nativo do main', envs: 'stock do main', strat: 'overlay+test', st: 'COPY + verificação de nvfp4_ds_mla e do backend B12xExperts' },
      };
      const S = { l: 'dsA', changed: false };
      UI.seg(ctl, { label: 'Lane', options: Object.keys(LANES).map((k) => ({ v: k, label: LANES[k].lab })), value: 'dsA', onChange: (v) => { S.l = v; } });
      UI.toggle(ctl, { label: 'a imagem-base mudou', value: false, onChange: (v) => { S.changed = v; } });
      function parseVer(v) {
        const m = /^(\d+\.\d+(?:\.\d+)?)(\.dev\d+)?(\+g[0-9a-f]+)?(\.d\d{8})?$/.exec(v || '');
        if (!m) return null;
        const noTag = m[1] === '0.1';
        const out = [[m[1], !m[2] ? 'release estável' : noTag ? 'sem tag: 0.1 padrão' : 'já puxada p/ a próxima', 'b']];
        if (m[2]) { const n = m[2].slice(4); out.push([m[2], noTag ? nf(Number(n)) + ' commits, sem tag' : n === '0' ? 'no commit da tag' : n + ' commits após a tag', 'd']); }
        if (m[3]) out.push([m[3], 'commit de origem', 'a']);
        if (m[4]) out.push([m[4], 'árvore suja (patches) nesta data', 'c']);
        return out;
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const mob = w < 640;
        const L = LANES[S.l], col = L.rc === 'tf' ? tfCol(C) : C[L.rc];
        let K, P, D;
        if (!mob) { K = { x: 14, y: 14, w: w * 0.44, h: h - 118 }; P = { x: w * 0.48, y: 14, w: w * 0.52 - 14, h: h - 118 }; D = { x: 14, y: h - 94, w: w - 28, h: 82 }; }
        else { K = { x: 14, y: 12, w: w - 28, h: 300 }; P = { x: 14, y: 326, w: w - 28, h: 272 }; D = { x: 14, y: 610, w: w - 28, h: h - 622 }; }
        /* bolo */
        const layers = [
          [L.cmd || 'vllm serve …', L.cmdSub || 'mesmos argumentos em todos os ranks', C.d],
          ['variáveis', L.envs, C.f],
          ['modificações', L.mod, col],
          ['imagem-base', L.base + ' ' + L.dig, C.b],
        ];
        const lh = (K.h - 10) / 4;
        layers.forEach(([nm, txt, c], i) => {
          const Y = K.y + i * lh, inset = (3 - i) * 8;
          const wob = Math.sin(t * 1.5 + i) * 1.5;
          V.box(g, K.x + inset + wob, Y, K.w - 2 * inset, lh - 8, { r: 12, fill: V.alpha(c, 0.14), stroke: V.alpha(c, 0.7), lw: 1.2 });
          V.text(g, nm, K.x + inset + 12, Y + 17, { size: 10.5, font: 'mono', color: c, weight: 700 });
          wrap(g, txt, K.x + inset + 12, Y + 33, K.w - 2 * inset - 24, 12.5, { size: mob ? 10 : 10.5, font: 'mono', color: C.ink, maxLines: Math.max(1, Math.floor((lh - 36) / 12.5)) });
        });
        if (S.changed) {
          const Y = P.y + P.h - 32;
          let msg, c;
          if (L.strat === 'anchor') { msg = '✕ âncora/hash não bate → boot aborta (fail-closed)'; c = C.c; }
          else if (L.strat === 'overlay') { msg = '⚠ arquivos inteiros sobem por cima: correções novas revertidas em silêncio'; c = C.d; }
          else if (L.strat === 'upstream') { msg = '✓ “já aplicado / already upstream” → patch pula'; c = C.e; }
          else if (L.strat === 'build') { msg = '✕ um diff não aplica → o build falha (fail-closed antes do boot)'; c = C.c; }
          else { msg = '⚠ arquivos inteiros + teste de sanidade: pega só parte'; c = C.d; }
          const BX = P.x + 12, bw = P.w - 24;
          const baseY = K.y + 3 * lh;
          g.save(); g.strokeStyle = c; g.lineWidth = 2; g.setLineDash([6, 4]); g.lineDashOffset = -t * 20;
          V.rr(g, K.x - 3 + Math.sin(t * 30) * 1.5, baseY - 3, K.w + 6, lh - 2, 14); g.stroke(); g.restore();
          V.glow(g, BX + bw / 2, Y, bw * 0.3, c, 0.18 + 0.08 * Math.sin(t * 4));
          V.box(g, BX, Y - 18, bw, 36, { r: 10, fill: V.alpha(c, 0.9), stroke: c, lw: 2 });
          wrap(g, msg, BX + bw / 2, Y - 3, bw - 16, 12.5, { size: 10.5, align: 'center', color: ink(C), weight: 700, maxLines: 2 });
        }
        /* painel */
        V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(col, 0.04), stroke: V.alpha(col, 0.45), lw: 1.2 });
        V.text(g, L.lab, P.x + 12, P.y + 20, { size: 13, weight: 700, color: col });
        const kv = [['model runner', L.runner], ['código do modelo', L.code], ['quando modifica', L.when], ['salvaguarda', L.st]];
        let yy = P.y + 42;
        kv.forEach(([k, v]) => {
          V.text(g, k, P.x + 12, yy, { size: 9.5, font: 'mono', color: C.muted });
          const n = wrap(g, v, P.x + 12, yy + 14, P.w - 24, 13, { size: 11, font: 'mono', color: C.ink, maxLines: 2 });
          yy += 14 + n * 13 + (mob ? 8 : 10);
        });
        /* decodificador de versão */
        V.box(g, D.x, D.y, D.w, D.h, { r: 12, fill: V.alpha(C.b, 0.04), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        const hdr = L.idParts ? (mob ? 'TAG DA IMAGEM (hash de conteúdo)' : 'TAG DA IMAGEM (hash de conteúdo; não há string do setuptools-scm)') : (mob ? 'STRING DE VERSÃO (setuptools-scm)' : 'STRING DE VERSÃO (setuptools-scm, esquema padrão; leitura nossa)');
        V.text(g, hdr, D.x + 12, D.y + 16, { size: 10, font: 'mono', color: C.muted, weight: 700, maxW: D.w - 24 });
        const parts = L.idParts || parseVer(L.ver);
        if (parts) {
          const fs = mob ? 12 : 15;
          const ws = parts.map((p) => Math.max(tw(g, p[0], fs, 'mono', 700) + 14, mob ? 0 : tw(g, p[1], 10, 'sans', 500) + 16));
          const tot = ws.reduce((a2, b2) => a2 + b2 + 4, 0);
          const sc = Math.min(1, (D.w - 24) / tot);
          let X = D.x + 12;
          const Y = D.y + 40;
          parts.forEach(([s2, lab, ck], i) => {
            const c = C[ck], ww = ws[i] * sc;
            V.box(g, X, Y - 14, ww, 26, { r: 6, fill: V.alpha(c, 0.2), stroke: V.alpha(c, 0.85), lw: 1.2 });
            V.text(g, s2, X + ww / 2, Y - 0.5, { size: fs, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700, maxW: ww - 6 });
            if (!mob) {
              V.line(g, X + ww / 2, Y + 13, X + ww / 2, Y + 19, { color: c, lw: 1.2 });
              V.text(g, lab, X + ww / 2, Y + 31, { size: 10, align: 'center', color: c, weight: 500, maxW: ww });
            } else {
              const lx2 = D.x + 12, ly2 = Y + 28 + i * 13;
              V.box(g, lx2, ly2 - 8, 8, 8, { r: 2, fill: c });
              V.text(g, s2 + ' → ' + lab, lx2 + 12, ly2, { size: 9.5, color: C.ink, maxW: D.w - 40 });
            }
            X += ww + 4 * sc;
          });
        } else {
          wrap(g, S.l === 'dsB' ? 'O repositório da receita não registra a versão do fork “unholy-fusion”: leia-a no log de boot da imagem.' : 'Imagem do main oficial com DSpark: a tag carrega o commit (00eb7ce).', D.x + 12, D.y + 38, D.w - 24, 14, { size: 11.5, color: C.ink, maxLines: 3 });
        }
      });
    })();

    /* ================================================================
       Figura 10 — anatomia do comando
       ================================================================ */
    (function figComando() {
      const fig = q('#fig-comando');
      const stage = fig.querySelector('.la-fig-stage');
      const ctl = UI.controls(fig);
      const CAT = {
        api: { l: 'API server', css: 'var(--viz-d)', col: (C) => C.d },
        sched: { l: 'Scheduler (EngineCore)', css: 'var(--viz-b)', col: (C) => C.b },
        kv: { l: 'KV manager', css: 'color-mix(in srgb, var(--viz-e) 50%, var(--viz-d))', col: (C) => V.mix(C.e, C.d, 0.5) },
        exec: { l: 'Executor · distribuído', css: 'var(--viz-c)', col: (C) => C.c },
        worker: { l: 'Worker · model runner', css: 'var(--viz-a)', col: (C) => C.a },
        env: { l: 'Ambiente · contêiner', css: 'var(--viz-muted)', col: (C) => C.muted },
      };
      const CMD = {
        glm: [
          ['--served-model-name GLM-5.3-Flash-EXL3', 'api', 'Nome exposto em /v1/models.'],
          ['--host 0.0.0.0 --port 8888', 'api', 'Servidor HTTP do head; o rank 1 não sobe API.'],
          ['--chat-template /opt/glm53/chat_template.jinja', 'api', 'Template Jinja renderizado pelo API server antes de tokenizar.'],
          ['--tool-call-parser glm47 --enable-auto-tool-choice', 'api', 'O output processor converte o markup de ferramentas do GLM em tool_calls.'],
          ['--reasoning-parser glm45', 'api', 'Separa <think>…</think> no campo reasoning (não reasoning_content): clientes que leem reasoning_content perdem o raciocínio.'],
          ['--limit-mm-per-prompt {"image":48,"video":1}', 'api', 'Teto de itens multimodais por prompt: validação no front-end, nada é reservado.'],
          ['--max-num-seqs 4', 'sched', 'Vagas por passo do scheduler.'],
          ['--max-num-batched-tokens 7168', 'sched', 'Orçamento de tokens por passo. 8192 estoura a shared memory do top-k no GB10.'],
          ['--max-model-len 850000', 'sched', 'Teto por requisição, não uma reserva.'],
          ['--enable-prefix-caching', 'kv', 'Reuso de blocos por hash de prefixo no KV manager.'],
          ['--kv-cache-dtype fp8', 'kv', 'Vira fp8_ds_mla, o único layout do kernel sparse-MLA no SM12x.'],
          ['--gpu-memory-utilization 0.85', 'kv', 'Fração da memória que o profile run pode orçar.'],
          ['--kv-cache-memory-bytes 11811160064', 'kv', 'Fixa 11 GiB de KV por GPU e pula a derivação por utilização.'],
          ['--tensor-parallel-size 2', 'exec', 'Dois ranks; o executor cria um worker por GPU.'],
          ['--nnodes 2 --node-rank 0|1', 'exec', 'Papel de cada nó no rendezvous.'],
          ['--master-addr $HEAD_IP --master-port 29521', 'exec', 'Endpoint de rendezvous do rank 0.'],
          ['--distributed-executor-backend mp', 'exec', 'Executor multiprocess, sem Ray.'],
          ['--headless (rank 1)', 'exec', 'Worker sem API server nem scheduler.'],
          ['--quantization exl3', 'worker', 'Seleciona a QuantizationConfig registrada pelo overlay; o worker constrói as camadas com ela.'],
          ['--load-format instanttensor', 'worker', 'Carregador de pesos com direct-I/O.'],
          ['--no-enable-flashinfer-autotune', 'worker', 'Desliga o autotune do FlashInfer, que derrubava o rank 0 no SM121.'],
          ['--speculative-config {"method":"dflash",…,"num_speculative_tokens":7}', 'worker', 'Rascunhador DFlash2 no model runner; o scheduler reserva 1 + k posições por decode.'],
          ['--cudagraph-capture-sizes 1 2 4 8 16 24 32', 'worker', 'Tamanhos de lote capturados em CUDA graph (DFlash2, k = 7).'],
          ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800', 'env', 'Prazo da RPC do passo (stock 300 s).'],
          ['GLM53_SPINWAIT_MS=stock (opcional 16)', 'env', 'Janela de busy-loop dos leitores das filas em shared memory (workers e EngineCore). Padrão stock (1 s); 16 ms no perfil long-coding.'],
          ['--ipc=host --shm-size 32g', 'env', 'Memória compartilhada do host para o shm_broadcast.'],
        ],
        qwen: [
          ['--served-model-name qwen3.8-flash-next', 'api', 'Nome exposto em /v1/models.'],
          ['--host 0.0.0.0 --port 8888', 'api', 'Servidor HTTP do head.'],
          ['--enable-prompt-tokens-details', 'api', 'Devolve usage.prompt_tokens_details.cached_tokens (só metadado).'],
          ['--reasoning-parser qwen3', 'api', 'Separa o raciocínio em delta["reasoning"] neste build (não reasoning_content).'],
          ['--enable-auto-tool-choice --tool-call-parser qwen3_coder', 'api', 'Converte as chamadas de ferramenta do Qwen em tool_calls.'],
          ['--max-num-seqs 8', 'sched', 'Vagas por passo.'],
          ['--max-num-batched-tokens 8192', 'sched', 'Orçamento de tokens por passo.'],
          ['--max-model-len 262144', 'sched', 'Teto por requisição (YaRN desligado por padrão).'],
          ['--enable-chunked-prefill', 'sched', 'Prefill em pedaços misturados com decode.'],
          ['--gpu-memory-utilization 0.80', 'kv', 'A 0,835, um soak deixava MemAvailable em 0,3–0,9 GiB, zona de hard-reset do GB10.'],
          ['--kv-cache-dtype fp8', 'kv', 'Com o patch QSA (day-0): 3,65M tokens em vez de 2,13M.'],
          ['--mamba-ssm-cache-dtype bfloat16', 'kv', 'Estado recorrente das 36 camadas GDN em BF16.'],
          ['--tensor-parallel-size 2', 'exec', 'Dois ranks, um por nó.'],
          ['--nnodes 2 --node-rank 0|1 --master-port 50000', 'exec', 'Rendezvous multi-nó.'],
          ['--distributed-executor-backend mp', 'exec', 'Executor multiprocess; na lane TP1, o único que cria o processo de offload da PLE.'],
          ['--enable-expert-parallel --all2all-backend allgather_reducescatter', 'worker', 'Cada GPU guarda 256 dos 512 experts; com DP = 1 o all2all nem é invocado.'],
          ['--mm-encoder-tp-mode data', 'worker', 'Replica o encoder de visão e divide o lote (o MLP 4304/2 não é múltiplo de 16).'],
          ['--load-format safetensors --safetensors-load-strategy lazy', 'worker', 'Carga preguiçosa: menos pico de memória no UMA.'],
          ['--speculative-config {"method":"mtp","num_speculative_tokens":3,…}', 'worker', 'MTP com 3 rascunhos, block-drop desligado e índice compartilhado.'],
          ['--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}', 'worker', 'Sem Inductor; graphs só no decode uniforme.'],
          ['--hf-overrides {"text_config":{…}}', 'worker', 'Mesclado no config antes de construir o modelo (dtype da PLE, YaRN).'],
          ['VLLM_HOST_IP · VLLM_ALLOW_LONG_MAX_MODEL_LEN=1', 'env', 'IP anunciado por nó; permite contexto acima do config.'],
          ['VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt', 'env', 'Lida pelo mtp.py patcheado (overlay).'],
          ['-v <arquivos patcheados>:…/site-packages/vllm/…:ro', 'env', 'Overlays montados por cima do pacote, nos dois nós.'],
          ['IMAGE=vllm/vllm-openai:qwen38-flash-next', 'env', 'Build day-0; a lane define os caminhos do modelo.'],
        ],
        tf: [
          ['--name GLM-5.3-Flash-EXL3', 'api', 'Id exposto em /v1/models. Só o rank 0 recebe este flag.'],
          ['--host 0.0.0.0 --port 8888', 'api', 'Onde a API escuta, só no rank 0 (o default do TensorFold é 127.0.0.1:8080).'],
          ['--thinking', 'api', 'Pensar antes de responder é o padrão; a requisição desliga com reasoning_effort "none" ou enable_thinking: false.'],
          ['--max-tokens 32768', 'api', 'Orçamento de raciocínio + resposta para quem omite max_tokens. O default do TensorFold, 4.096, podia cortar uma resposta dentro de uma chamada de ferramenta.'],
          ['--parallel 4', 'sched', 'Requisições decodificadas juntas, de 1 a 4 (o teto do código). Prosa: 60,4 tok/s com 1 e 108,8 agregados com 4.'],
          ['TF_GLM_MULTI_PREFILL=1', 'sched', 'Prompts que chegam juntos são preenchidos numa só forward: com 4 requisições, o TTFT de prosa cai de 590 para 340 ms.'],
          ['--context 1048576', 'kv', 'Janela de prompt + resposta por requisição. Uma janela que não cabe é recusada antes de carregar os pesos, com a maior que cabe.'],
          ['TF_GLM_KV=fp8', 'kv', 'Latente DSA em e4m3 com escala potência de 2: 528 B por token e camada, cerca de metade do BF16. Com perdas.'],
          ['TF_GLM_CACHE_GIB=12.5', 'kv', 'Teto do pool de caches além da janela, por Spark (default do TensorFold: 3). Na partida medida, 2.922.496 tokens.'],
          ['TENSORFOLD_MEMORY_RESERVE_GIB=14.5', 'kv', 'Orçamento = MemAvailable na partida − reserva. Na memória unificada, faltar memória tende a congelar a máquina.'],
          ['--tp 2 --rank 0|1', 'exec', 'Dois ranks, um por Spark: o motor CUDA do GLM exige exatamente dois. O rank 1 sobe primeiro, por ssh.'],
          ['--master <head> --master-port 29551', 'exec', 'TCPStore do rank 0: rendezvous do NCCL e campainha do rank ocioso.'],
          ['TF_GLM_COMM=roce · TF_ROCE_MAX_KB=512', 'exec', 'All-gathers de até 512 KiB por RDMA de um tiro: 11 µs contra 45 µs do NCCL num gather de 16 KiB, +6% de decode. Os maiores vão pelo NCCL, com os mesmos bits.'],
          ['--drafter <snapshot do DFlash2>', 'worker', 'Rascunhador DFlash2, de licença não comercial (+5–10% de decode). Com DRAFTER=mtp, o launcher passa --drafter none e quem rascunha é a cabeça MTP do próprio checkpoint, uma requisição por vez.'],
          ['TF_GLM_DFLASH_POLICY=fnc7:0.3', 'worker', 'Até 7 rascunhos por rodada, enquanto o produto das chances sob o ruído da própria amostragem ficar acima de 0,3. A resposta é a mesma sob qualquer política.'],
          ['TF_GLM_DENSE=q4', 'worker', 'Os pesos BF16 fora dos experts roteados viram 4 bits afins em grupos de 64 na carga. Mais rápido, com perdas.'],
          ['--vision', 'worker', 'Imagens e vídeo, com a torre de visão só no rank 0 (~1,8 GiB).'],
          ['-e TENSORFOLD_* TF_GLM_* TF_ROCE_* (nos dois)', 'env', 'O comando do rank 1 roda por ssh, onde o ambiente local não chega. Na partida, os ranks comparam as configurações.'],
          ['--ipc=host --shm-size 16g --network host', 'env', 'Memória compartilhada e rede do host, onde está o link CX7 (leitura nossa dos flags do Docker).'],
          ['--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1', 'env', 'Deixam registrar memória pinada para o RDMA (leitura nossa).'],
          ['tensorfold-glm53:v0.6.0', 'env', 'Imagem com os 53 patches aplicados no build; a tag publicada carrega o hash dos patches (v0.6.0-ae8d1c789b47).'],
        ],
        ds: [
          ['--served-model-name deepseek-v4-flash-vision-exp', 'api', 'Nome(s) em /v1/models; o primeiro alias é o usado pelos probes.'],
          ['--host … --port 8888', 'api', 'Servidor HTTP do head.'],
          ['--tokenizer-mode deepseek_v4', 'api', 'Encoder Python do checkpoint em vez de template Jinja.'],
          ['--tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4', 'api', 'DSML → tool_calls; <think> → reasoning_content.'],
          ['--generation-config vllm', 'api', 'Defaults de amostragem do vLLM, não os do checkpoint.'],
          ['--limit-mm-per-prompt {"image":8}', 'api', 'Teto de imagens por requisição.'],
          ['--max-num-seqs 6', 'sched', 'Vagas por passo.'],
          ['--max-num-batched-tokens 8192', 'sched', 'Orçamento de tokens por passo.'],
          ['--long-prefill-token-threshold 1024', 'sched', 'Pedaço máximo de prefill longo: protege o decode (#27).'],
          ['--async-scheduling', 'sched', 'Agenda o passo N+1 com o N ainda na GPU.'],
          ['--enable-chunked-prefill', 'sched', 'Prefill fatiado, base da fairness.'],
          ['--max-model-len 1048576', 'sched', 'Teto de 1M por requisição (YaRN ×16 no checkpoint).'],
          ['--kv-cache-dtype nvfp4_ds_mla', 'kv', 'Apesar do nome, 584 B/token, byte-idêntico ao fp8_ds_mla; muda o dispatch.'],
          ['--block-size 256', 'kv', 'Tokens por bloco do grupo MLA; é também o grão do hash de prefixo.'],
          ['--enable-prefix-caching', 'kv', 'Reuso de blocos por hash de prefixo.'],
          ['--gpu-memory-utilization 0.835', 'kv', 'Cada +0,01 ≈ +1,2 GiB de KV no UMA.'],
          ['--tensor-parallel-size 2 --pipeline-parallel-size 1', 'exec', 'Dois ranks TP, sem pipeline.'],
          ['--nnodes 2 --node-rank … --master-port 25000', 'exec', 'Rendezvous multi-nó.'],
          ['--distributed-executor-backend mp', 'exec', 'Executor multiprocess.'],
          ['--headless (worker)', 'exec', 'O worker não sobe API.'],
          ['--trust-remote-code --revision 86f746b3…', 'worker', 'Código do repositório e commit fixado do checkpoint.'],
          ['--moe-backend flashinfer_b12x', 'worker', 'Oráculo → B12X_MXFP4 → B12xExperts (W4A16).'],
          ['--speculative-config {"method":"dspark","num_speculative_tokens":6,…}', 'worker', 'DSpark k = 6 no model runner; o SpeculativeConfig reescreve o config para o drafter.'],
          ['--max-cudagraph-capture-size 48', 'worker', '6 × (1 + 6) = 42, arredondado a múltiplo de 8.'],
          ['--enable-flashinfer-autotune', 'worker', 'Autotune no boot; rank 0 tuna e distribui.'],
          ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800', 'env', 'Prazo da RPC do passo (stock 300 s).'],
          ['VLLM_PLUGINS= (vazio) · ENABLE_VLLM_GB10_PATCH=0', 'env', 'Nenhum plugin geral; o plugin GB10 é opt-in.'],
          ['VLLM_USE_B12X_MOE=1', 'env', 'Variável do fork: liga o MoE b12x.'],
          ['ipc: host · shm_size: 64gb', 'env', 'Memória compartilhada para o shm_broadcast.'],
          ['image: ghcr.io/anemll/dspark-vllm-gx10:0.1.1', 'env', 'Lane A: runner V2, hotfixes source-exact contra esta build.'],
        ],
      };
      const S = { r: 'glm', sel: 0, t0: performance.now() };
      const CAT_TF = { api: 'Servidor HTTP (rank 0)', sched: 'Agendador (thread)', kv: 'Pool de caches', exec: 'Ranks · fio · rendezvous', worker: 'Engine da família', env: 'Ambiente · contêiner' };
      const catL = (k) => (S.r === 'tf' && CAT_TF[k]) || CAT[k].l;
      const cvWrap = LA.h('div', { style: { position: 'relative', height: '176px', width: '100%' } });
      const cv = LA.h('canvas', { style: { width: '100%', height: '100%', display: 'block' } });
      cvWrap.appendChild(cv);
      const chips = LA.h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '12px' } });
      const expl = LA.h('div', { style: { marginTop: '12px', padding: '10px 14px', borderRadius: '10px', border: '1px solid var(--line-2)', background: 'var(--surface)', fontSize: '14px', color: 'var(--text-2)', lineHeight: '1.55', minHeight: '46px' } });
      stage.append(cvWrap, chips, expl);
      function render() {
        chips.innerHTML = '';
        CMD[S.r].forEach((row, i) => {
          const c = CAT[row[1]];
          const on = i === S.sel;
          const b = LA.h('button', { type: 'button', 'data-act': 'flag', style: { fontFamily: 'var(--font-mono)', fontSize: '12px', padding: '4px 9px', borderRadius: '8px', cursor: 'pointer', color: 'var(--text)', border: '1px solid ' + c.css, background: 'color-mix(in srgb, ' + c.css + ' ' + (on ? 32 : 10) + '%, transparent)', boxShadow: on ? '0 0 0 2px color-mix(in srgb, ' + c.css + ' 45%, transparent)' : 'none', maxWidth: '100%', overflowWrap: 'anywhere', textAlign: 'left' } }, row[0]);
          b.addEventListener('click', () => { S.sel = i; S.t0 = performance.now(); render(); });
          chips.appendChild(b);
        });
        const row = CMD[S.r][S.sel];
        const c = CAT[row[1]];
        expl.innerHTML = '';
        expl.append(LA.h('b', { style: { color: c.css, fontFamily: 'var(--font-display)', letterSpacing: '.06em', textTransform: 'uppercase', fontSize: '12.5px', marginRight: '8px' } }, catL(row[1])), LA.h('code', null, row[0]), LA.h('div', { style: { marginTop: '4px' } }, row[2]));
      }
      let leg = null;
      const mkLegend = () => { if (leg) leg.remove(); leg = UI.legend(ctl, Object.keys(CAT).map((k) => [CAT[k].css, catL(k)])); };
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'tf', label: 'GLM·TF' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.r = v; S.sel = 0; S.t0 = performance.now(); render(); mkLegend(); } });
      const C0 = V.C();
      mkLegend();
      render();
      V.loop(cv, (g, w, h, t) => {
        const C = V.C();
        const keys = Object.keys(CAT);
        const cur = CMD[S.r][S.sel][1];
        const mob = w < 560;
        const cols = mob ? 3 : 6, rows = mob ? 2 : 1;
        const top = 34;
        const bw = (w - 20 - (cols - 1) * 8) / cols, bh = (h - top - 10 - (rows - 1) * 8) / rows;
        const src = [w / 2, 14];
        pill(g, src[0], src[1], S.r === 'tf' ? 'tensorfold serve · GLM·TF' : 'vllm serve · ' + (S.r === 'glm' ? 'GLM' : S.r === 'qwen' ? 'Qwen' : 'DeepSeek'), C.ink, { size: 10.5, a: 0.08 });
        const ci = keys.indexOf(cur);
        if (ci >= 0) {
          const X = 10 + (ci % cols) * (bw + 8), Y = top + Math.floor(ci / cols) * (bh + 8);
          const c = CAT[cur].col(C);
          V.glow(g, X + bw / 2, Y + bh / 2, Math.max(bw, bh) * 0.7, c, 0.35);
          g.save(); g.strokeStyle = c; g.lineWidth = 2; g.setLineDash([5, 5]); g.lineDashOffset = -t * 30;
          g.beginPath(); g.moveTo(src[0], src[1] + 9);
          if (Y === top) g.quadraticCurveTo(X + bw / 2, src[1] + 10, X + bw / 2, Y - 2);
          else { const gx = ci % cols ? X - 4 : X + bw + 4; g.quadraticCurveTo(gx, src[1] + 10, gx, top); g.lineTo(gx, Y - 4); g.lineTo(X + bw / 2, Y - 4); g.lineTo(X + bw / 2, Y - 1); } // 2ª linha (mobile): desce pelo vão entre caixas
          g.stroke(); g.restore();
        }
        keys.forEach((k, i) => {
          const X = 10 + (i % cols) * (bw + 8), Y = top + Math.floor(i / cols) * (bh + 8);
          const c = CAT[k].col(C);
          const on = k === cur;
          const n = CMD[S.r].filter((r) => r[1] === k).length;
          V.box(g, X, Y, bw, bh, { r: 10, fill: V.alpha(c, on ? 0.26 : 0.08), stroke: on ? c : V.alpha(c, 0.45), lw: on ? 2 : 1 });
          wrap(g, catL(k), X + bw / 2, Y + (mob ? 18 : bh / 2 - 2), bw - 10, 12, { size: mob ? 10 : 11, align: 'center', color: C.ink, weight: on ? 700 : 500, maxLines: 2 });
          V.text(g, n + (n === 1 ? ' item' : ' itens'), X + bw / 2, Y + bh - 8, { size: 9.5, font: 'mono', align: 'center', color: C.muted });
        });
      });
    })();
  },
});

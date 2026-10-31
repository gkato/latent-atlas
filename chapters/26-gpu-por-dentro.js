LA.chapter({
  id: 'gpu-por-dentro',
  part: 5,
  order: 26,
  title: 'GPU por dentro: SMs, tensor cores e banda',
  navTitle: 'GPU por dentro',
  subtitle: 'SMs, warps, registradores e shared memory; tensor cores e formatos por geração; compute capability, PTX e cubins para o sm_121a do GB10; e o roofline que transforma bytes lidos por passo em tokens por segundo.',
  minutes: 60,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['GPU', 'SM', 'warp', 'shared memory', 'tensor core', 'mma.sync', 'compute capability', 'sm_121a', 'PTX', 'cubin', 'roofline', 'LPDDR5X', 'streams'],
  goals: [
    'Descrever a GPU de cima a baixo (SMs, sub-partições, warps, lanes, registradores, shared memory, L2, DRAM) e calcular a <b>ocupação</b> de um kernel, reconhecendo spills',
    'Explicar o que uma instrução de tensor core calcula (<code>mma.sync m16n8k16</code>, MMA com escala por bloco) e por que o GB10 usa <code>mma.sync</code>, e não <code>tcgen05</code>',
    'Ler <b>compute capability</b>, sufixos <code>a</code>/<code>f</code>, PTX, cubin e fatbin, e justificar <code>TORCH_CUDA_ARCH_LIST=12.1a</code>, <code>CUTE_DSL_ARCH=sm_121a</code> e os gates das receitas',
    'Montar o <b>modelo de custo do passo</b> (bytes lidos ÷ banda + custo fixo) e prever o teto de tok/s de decode com os números do GB10',
    'Usar streams, eventos e memória pinned para tirar a CPU do caminho crítico, e uma stream lateral para encher a L2 enquanto a DRAM estaria parada',
  ],
  prereqs: ['prefill-decode-kv', 'numeros-no-computador'],
  summary: [
    'A GPU é uma grade de <b>SMs</b> (48 no GB10), cada um com 4 sub-partições; o trabalho é cortado em grid → CTAs → warps de 32 lanes que executam a mesma instrução (SIMT) e trocam registradores por shuffles.',
    'Registradores e shared memory limitam a <b>ocupação</b>; spills vão para a memória local, que mora na DRAM. O thin-decode FAST do GLM ganhou 8–17% por camada zerando os LDL. Kernels de datacenter pressupõem ~228 KB de SMEM; o SM 12.x tem ~100 KB, e uma CTA do GB10 pede no máximo 99 KiB (101.376 B), e por isso os kernels Triton da KDA em chunks da versão TensorFold rodavam com um único estágio de pipeline, antes de virarem um kernel CUDA com o estado nos registradores.',
    'Tensor cores fazem \\(D = AB + C\\) por instrução de warp; o Blackwell de consumo (12.x) usa <code>mma.sync</code>, inclusive FP4 com escala por bloco (<code>kind::mxf4nvf4 m16n8k64</code>), mas não tem tcgen05/TMEM, por isso FlashMLA, MegaMoE e TRTLLM MoE não rodam no GB10.',
    'Compute capability 12.1 divide a ISA de consumo com a 12.0: cubins <code>sm_120</code> rodam no GB10, <code>sm_121a</code> só no GB10; gates como <code>family(100)</code> ou <code>== (12,0)</code> excluem o GB10 e viram patches (alias DeepGEMM, cache MXFP4 do indexer, tabela QSA).',
    'Roofline do GB10: ~125 TFLOPS BF16 ÷ ~273 GB/s ⇒ ridge ≈ 458 FLOP/B. GEMV de decode tem \\(I \\approx 16M/b\\); até o MoE do prefill do GLM (≈28 linhas por expert, 112 FLOP/B) é limitado por memória.',
    'Modelo de custo: \\(t_{\\text{passo}} \\approx \\sum Q_i / B_{\\text{ef}} + t_{\\text{fixo}}\\). Qwen: 9,2 GiB/passo → 40–42 ms; τ≈3,2 → teto ~76–80 tok/s; medido 52,1 (passo ~60 ms, ~1/3 custo fixo). DeepSeek: 16,5–17 GB/rank/passo → 68–74 ms, ~80% streaming de pesos.',
    'Streams e eventos sobrepõem GEMMs pequenas (3 streams auxiliares no overlay Stage-C do DeepSeek, a lane B; as <code>VLLM_DSPARK_*</code> do overlay são no-op na imagem de produção);<code>.item()/.tolist()</code> esvaziam a fila; cópias D2H só são assíncronas para memória pinned (tier E1 do GLM). Na versão TensorFold, uma stream lateral que só lê traz para a L2 os pesos dos próximos kernels nas fases em que a DRAM para (+2–3% de decode, mesmos bits).',
  ],
  quiz: [
    { q: 'No GB10 (~125 TFLOPS BF16, ~273 GB/s), uma GEMV de decode com pesos BF16 e M = 1 tem intensidade ≈ 1 FLOP/B. Que fração do pico de cálculo ela consegue usar, no máximo?', o: ['~100%', '~14%', '~0,2%', '~50%'], a: 2, e: 'O teto é \\(I \\cdot B = 1 \\times 273\\) GFLOPS ≈ 0,27 TFLOPS, ou 0,2% de 125 TFLOPS. Por isso o decode é limitado por banda e quantizar os pesos (menos bytes por FLOP) acelera quase linearmente.' },
    { q: 'Uma extensão compilada com <code>TORCH_CUDA_ARCH_LIST=12.1a</code> roda numa RTX PRO 6000 (compute capability 12.0)?', o: ['Sim, é a mesma família', 'Não: o sufixo <code>a</code> gera código específico da CC 12.1 exata, que não roda em nenhuma outra', 'Sim, via compatibilidade para trás', 'Só se a GPU tiver 48 SMs'], a: 1, e: 'Cubins <code>sm_XYa</code> usam instruções específicas daquela arquitetura e rodam só nela. Um <code>sm_120</code> sem sufixo roda no 12.1 (mesmo major, minor maior), mas o contrário (<code>sm_121a</code> no 12.0) não.' },
    { q: 'O backport de FP8 KV do Qwen para a v0.30 traz uma tabela de lançamento afinada no sm_120, escolhida por <code>_is_sm120()</code>, que testa <code>== (12, 0)</code>. O que acontece no GB10?', o: ['Usa a tabela do sm_120', 'Falha na carga', 'Não entra na tabela sm_120 e segue com a tabela padrão, afinada no GB300 (inferido do código)', 'Usa um perfil próprio do GB10 automaticamente'], a: 2, e: 'O GB10 é (12, 1). A igualdade exata o exclui, então o ganho “de sm_120” não se aplica sem mudar o gate. Por isso gates devem testar a família ou o recurso, e não igualdade.' },
    { q: 'No prefill do GLM a MNBT 1024 (top-8 entre 288 experts), por que o MoE é limitado por memória mesmo no <i>prefill</i>?', o: ['Porque o prefill é sempre memory-bound', 'Porque cada expert recebe ~28 linhas; com pesos de 4 bpw, \\(I \\approx 4 \\times 28 = 112\\) FLOP/B, abaixo do ridge de ~458', 'Porque o KV cache é grande', 'Porque a rede de 200 Gb/s limita'], a: 1, e: 'A intensidade cresce com as linhas que compartilham a leitura dos pesos. Com 8.192 slots divididos por 288 experts, M por expert é pequeno; o relatório estima 75,6 GB lidos por GPU por chunk (≥ 330 ms). Chunks maiores ou kernels agrupados aumentam M.' },
    { q: 'O Qwen lê ≈9,2 GiB por GPU por passo de decode (MTP=3, τ≈3,2 tokens por passo). A ~240 GB/s efetivos, qual o teto aproximado de tok/s por sequência?', o: ['~230 tok/s', '~78 tok/s', '~24 tok/s', '~120 tok/s'], a: 1, e: '9,2 GiB ≈ 9,9 GB; 9,9/240 ≈ 41 ms por passo ≈ 24 passos/s; × 3,2 tokens (2,2 aceitos + o bônus) ≈ 78 tok/s de teto só de streaming. O medido (52,1) fica ~1/3 abaixo: o resto do passo de ~60 ms é custo fixo.' },
    { q: 'Por que o tier E1 do GLM copia as contagens de roteamento para um tensor com <code>pin_memory=True</code>, numa stream lateral?', o: ['Porque memória pinned é mais rápida para a CPU calcular', 'Porque só uma cópia D2H para memória pinned é realmente assíncrona (DMA sem a CPU); numa stream lateral ela corre em paralelo com os experts finos já enfileirados', 'Porque o NCCL exige', 'Para economizar memória da GPU'], a: 1, e: 'Com memória paginável, o driver passa por um buffer intermediário e a cópia vira síncrona. Com pinned + stream lateral + evento, a GPU segue trabalhando e a CPU espera só a cópia pequena antes do <code>.tolist()</code>.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const cm = (x, d = 1) => (+x).toFixed(d).replace('.', ',');
    const fmtInt = (n) => Math.round(n).toLocaleString('pt-BR');
    const fnt = (o) => V.font(o.size || 12, o.font || 'sans', o.weight || 400);
    function splitLines(g, s, maxW, o) {
      g.font = fnt(o);
      const out = []; let cur = '';
      String(s).split(' ').forEach((wd) => { const tt = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(tt).width > maxW) { out.push(cur); cur = wd; } else cur = tt; });
      if (cur) out.push(cur);
      return out;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const ls = splitLines(g, s, maxW, o), mx = o.maxLines || 99;
      ls.slice(0, mx).forEach((l, i) => V.text(g, i === mx - 1 && ls.length > mx ? l + '…' : l, x, y + i * lh, o));
      return Math.min(ls.length, mx);
    }
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const onTxt = (C) => (C.dark ? '#06140d' : '#ffffff');
    const cellTxt = (C, strong) => (strong ? (C.dark ? '#00210F' : '#ffffff') : C.ink);
    // cor categórica por threadID_in_group (lane % 4) e tom por groupID (lane >> 2), a convenção de fragmentos do PTX
    const TIG_HUE = [150, 265, 38, 330];
    const laneCol = (l, a = 1) => { const C = V.C(), grp = l >> 2; return `hsla(${TIG_HUE[l & 3]}, ${C.dark ? 72 : 64}%, ${C.dark ? 30 + grp * 6 : 26 + grp * 6.5}%, ${a})`; };
    function stripes(g, x, y, w, h, col) {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1.2;
      for (let k = -h; k < w; k += 6) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); }
      g.restore();
    }

    /* ======================= Figura 1: zoom na GPU ======================= */
    {
      const fig = $('#fig-zoom');
      const steps = [
        { t: 'O chip GB10', d: 'Um SoC com 20 núcleos ARM e uma GPU Blackwell de <b>48 SMs</b>, os dois lendo a mesma DRAM LPDDR5X de 128 GB (~273 GB/s teóricos). Os bytes sobem da DRAM, passam pela L2 compartilhada e chegam aos SMs. A CPU usa a mesma memória.' },
        { t: 'Dentro de um SM', d: 'Quatro sub-partições, cada uma com escalonador de warps, uma fatia do banco de registradores (65.536 registradores de 32 bits no SM inteiro), lanes de INT/FP32 e um tensor core. Embaixo, a L1/shared memory comum ao SM (até ~100 KB de SMEM no 12.x). Arquitetura pública de referência.' },
        { t: 'Um warp: 32 lanes, uma instrução', d: 'O escalonador emite <i>uma</i> instrução para 32 lanes. Se um <code>if</code> divide as lanes, o warp executa os dois lados em sequência, com metade das lanes desligada em cada um: 50% de eficiência.' },
        { t: 'Uma thread e seus registradores', d: 'Cada thread tem até 255 registradores de 32 bits, lidos no ritmo da instrução. Se o kernel precisa de mais, o excedente vai para a <b>memória local</b>, que mora na DRAM, e cada <code>LDL</code> custa centenas de ciclos.' },
        { t: 'Um kernel espalhado pelos SMs', d: 'O grid é uma grade de CTAs, e cada CTA roda inteira num SM. Aqui são CTAs de 256 threads (8 warps), como no kernel E2 do GLM. Quando há mais CTAs do que vagas, elas entram em ondas.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7500 });
      fitH(fig, (w) => (w < 560 ? 450 : 410));
      const r = V.rng(5);
      const parts = Array.from({ length: 70 }, () => ({ p: r(), sm: r.int(48), cpu: r() < 0.16, x: r() }));
      const regs = Array.from({ length: 12 }, () => Math.round((r() * 20 - 10) * 100) / 100);

      function chip(g, w, h, t, C) {
        const narrow = w < 560, pad = 12;
        const dramH = 40, dramY = h - dramH - 26;
        V.box(g, pad, dramY, w - 2 * pad, dramH, { r: 8, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.6) });
        const nCh = narrow ? 6 : 8, chw = (w - 2 * pad - 16) / nCh;
        for (let k = 0; k < nCh; k++) V.box(g, pad + 8 + k * chw + 3, dramY + 8, chw - 6, dramH - 16, { r: 4, fill: V.alpha(C.d, 0.28), label: chw > 70 ? 'LPDDR5X' : '', size: 11 });
        T(g, 'DRAM LPDDR5X · 128 GB · ~273 GB/s · CPU e GPU dividem', w / 2, dramY + dramH + 17, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
        const top = 10, bodyH = dramY - 20 - top;
        let cpuB, gpuB;
        if (narrow) { cpuB = { x: pad, y: top, w: w - 2 * pad, h: 32 }; gpuB = { x: pad, y: top + 42, w: w - 2 * pad, h: bodyH - 42 }; }
        else { const cw2 = Math.min(170, w * 0.2); cpuB = { x: pad, y: top, w: cw2, h: bodyH }; gpuB = { x: pad + cw2 + 14, y: top, w: w - 2 * pad - cw2 - 14, h: bodyH }; }
        V.box(g, cpuB.x, cpuB.y, cpuB.w, cpuB.h, { r: 10, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.7) });
        T(g, 'CPU · 20 núcleos ARM', cpuB.x + cpuB.w / 2, cpuB.y + 21, { size: 12.5, weight: 600, align: 'center', maxW: cpuB.w - 8 });
        if (!narrow) {
          const cs = Math.min(22, (cpuB.w - 30) / 4), ox = cpuB.x + (cpuB.w - (4 * cs + 9)) / 2;
          for (let k = 0; k < 20; k++) V.box(g, ox + (k % 4) * (cs + 3), cpuB.y + 38 + Math.floor(k / 4) * (cs + 3), cs, cs, { r: 3, fill: V.alpha(C.b, 0.22 + 0.2 * Math.sin(t * 2 + k)) });
          wrap(g, 'poucos núcleos grandes: latência baixa', cpuB.x + 10, cpuB.y + 38 + 5 * (cs + 3) + 18, cpuB.w - 20, 15, { size: 11.5, color: C.muted });
        }
        V.box(g, gpuB.x, gpuB.y, gpuB.w, gpuB.h, { r: 10, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.7) });
        T(g, 'GPU Blackwell · sm_121 · 48 SMs', gpuB.x + 12, gpuB.y + 20, { size: 12.5, weight: 600, maxW: gpuB.w - 20 });
        const l2H = 24, l2Y = gpuB.y + gpuB.h - l2H - 10;
        V.box(g, gpuB.x + 10, l2Y, gpuB.w - 20, l2H, { r: 6, fill: V.alpha(C.f, 0.18), stroke: V.alpha(C.f, 0.6), label: narrow ? 'L2 compartilhada' : 'L2 (compartilhada por todos os SMs)', size: 11.5 });
        const cols = narrow ? 8 : 12, rows = 48 / cols, gap = 4;
        const gx0 = gpuB.x + 12, gy0 = gpuB.y + 32, gw = gpuB.w - 24, gh = l2Y - gy0 - 8;
        const cs = Math.max(4, Math.min((gw - gap * (cols - 1)) / cols, (gh - gap * (rows - 1)) / rows));
        const gxs = gx0 + (gw - (cols * cs + (cols - 1) * gap)) / 2;
        const smPos = (k) => [gxs + (k % cols) * (cs + gap) + cs / 2, gy0 + Math.floor(k / cols) * (cs + gap) + cs / 2];
        const heat = new Array(48).fill(0);
        const pts = parts.map((p) => {
          const u = (p.p + t * 0.22) % 1;
          const sx = pad + 20 + p.x * (w - 2 * pad - 40), sy = dramY + 4;
          const l2x = gpuB.x + 20 + p.x * (gpuB.w - 40), l2y = l2Y + l2H / 2;
          if (p.cpu) return [V.lerp(sx, cpuB.x + cpuB.w / 2, u), V.lerp(sy, cpuB.y + cpuB.h - 8, u), C.b];
          if (u < 0.45) { const v = u / 0.45; return [V.lerp(sx, l2x, v), V.lerp(sy, l2y, v), C.d]; }
          const v = (u - 0.45) / 0.55, [tx, ty] = smPos(p.sm);
          if (v > 0.85) heat[p.sm] = Math.max(heat[p.sm], (v - 0.85) / 0.15);
          return [V.lerp(l2x, tx, v), V.lerp(l2y, ty, v), C.a];
        });
        for (let k = 0; k < 48; k++) {
          const [cx, cy] = smPos(k);
          V.box(g, cx - cs / 2, cy - cs / 2, cs, cs, { r: Math.min(4, cs / 4), fill: V.alpha(C.a, 0.12 + 0.5 * heat[k]), stroke: V.alpha(C.a, 0.55), lw: 1 });
          if (cs >= 30) T(g, 'SM', cx, cy + 4, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        pts.forEach(([x, y, c]) => V.dot(g, x, y, 2.3, c));
      }

      function sm(g, w, h, t, C) {
        const pad = 12, narrow = w < 560;
        const bx = pad, by = 8, bw = w - 2 * pad, bh = h - 16;
        V.box(g, bx, by, bw, bh, { r: 12, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.7) });
        T(g, 'Um SM (1 de 48)', bx + 12, by + 20, { size: 13, weight: 600 });
        if (!narrow) T(g, 'arquitetura de referência', bx + bw - 12, by + 20, { size: 11, align: 'right', color: C.muted });
        const smemH = 32, smemY = by + bh - smemH - 10;
        V.box(g, bx + 10, smemY, bw - 20, smemH, { r: 8, fill: V.alpha(C.f, 0.16), stroke: V.alpha(C.f, 0.6), label: narrow ? 'L1 / shared memory (~100 KB SMEM no 12.x)' : 'L1 / shared memory, comum às 4 sub-partições (até ~100 KB de SMEM no 12.x)', size: 11.5 });
        const gx = bx + 10, gy = by + 32, gw = bw - 20, gh = smemY - gy - 8, gp = 8;
        const qw = (gw - gp) / 2, qh = (gh - gp) / 2;
        for (let q = 0; q < 4; q++) {
          const x = gx + (q % 2) * (qw + gp), y = gy + Math.floor(q / 2) * (qh + gp);
          V.box(g, x, y, qw, qh, { r: 8, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          T(g, `sub-partição ${q}`, x + 8, y + 16, { size: 11.5, weight: 600 });
          const wid = (q * 7 + Math.floor(t * 2.5)) % 12;
          V.box(g, x + 8, y + 24, Math.min(qw - 16, 160), 20, { r: 5, fill: V.alpha(C.d, 0.18), stroke: V.alpha(C.d, 0.6), label: qw > 150 ? `escalonador → warp ${wid}` : `warp ${wid}`, size: 11 });
          const ry = y + 50, rh = Math.max(10, Math.min(18, qh * 0.12));
          const nseg = 16, sw = (qw - 20) / nseg;
          for (let k = 0; k < nseg; k++) { g.fillStyle = V.alpha(C.b, 0.2 + 0.45 * (0.5 + 0.5 * Math.sin(t * 3 + k * 0.7 + q))); V.rr(g, x + 10 + k * sw, ry, sw - 2, rh, 2); g.fill(); }
          T(g, 'registradores (¼ do banco)', x + 10, ry + rh + 13, { size: 11, color: C.muted, maxW: qw - 20 });
          const ly = ry + rh + 22, tcW = Math.min(92, qw * 0.32), lw = qw - 20 - tcW - 8;
          const ls = Math.max(3, Math.min(10, lw / 16 - 2)), lh2 = ls;
          for (let k = 0; k < 32; k++) { const on = Math.sin(t * 4 + k * 0.5 + q) > -0.2; g.fillStyle = on ? V.alpha(C.a, 0.75) : V.alpha(C.a, 0.2); V.rr(g, x + 10 + (k % 16) * (ls + 2), ly + Math.floor(k / 16) * (lh2 + 3), ls, lh2, 2); g.fill(); }
          if (qh > 130) T(g, '32 lanes INT/FP32', x + 10, ly + 2 * (lh2 + 3) + 13, { size: 11, color: C.muted, maxW: lw });
          const tcx = x + qw - 10 - tcW, tcy = ly - 2, tch = Math.min(qh - (ly - y) - 8, 44);
          const pulse = 0.5 + 0.5 * Math.sin(t * 3 + q);
          V.box(g, tcx, tcy, tcW, Math.max(20, tch), { r: 6, fill: V.alpha(C.c, 0.15 + 0.2 * pulse), stroke: C.c, label: 'tensor core', size: 11 });
        }
      }

      function warp(g, w, h, t, C) {
        const narrow = w < 700;
        const phase = Math.floor(t / 1.8) % 3, pu = (t % 1.8) / 1.8;
        const instr = ['FFMA  d = a × b + c', 'if (lane < 16)  x = f(x)', 'else  x = g(x)'][phase];
        const active = (l) => (phase === 0 ? true : phase === 1 ? l < 16 : l >= 16);
        const ibw = Math.min(w - 40, 380), ibx = (w - ibw) / 2, iby = 12, ibh = 32;
        V.box(g, ibx, iby, ibw, ibh, { r: 8, fill: V.alpha(C.d, 0.16), stroke: C.d, label: instr, font: 'mono', size: 12.5 });
        T(g, 'o escalonador emite 1 instrução para o warp inteiro', w / 2, iby + ibh + 16, { size: 11.5, align: 'center', color: C.muted, maxW: w - 20 });
        const perRow = w < 560 ? 8 : narrow ? 16 : 32;
        const sp = (w - 32) / perRow, R = Math.min(11, sp * 0.36);
        const y0 = w < 560 ? h * 0.3 : narrow ? h * 0.4 : h * 0.52;
        for (let l = 0; l < 32; l++) {
          const cx = 16 + sp * (l % perRow) + sp / 2, cy = y0 + Math.floor(l / perRow) * (R * 2 + 36);
          const on = active(l);
          if (on) { V.line(g, w / 2, iby + ibh, cx, cy - R, { color: V.alpha(C.d, 0.12), lw: 1 }); const [px, py] = V.along(w / 2, iby + ibh, cx, cy - R, pu); V.dot(g, px, py, 2, C.d); }
          g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2);
          g.fillStyle = on ? V.alpha(C.a, 0.3 + 0.45 * Math.pow(Math.sin(t * 5 + l * 0.3), 2)) : V.alpha(C.muted, 0.08); g.fill();
          g.strokeStyle = on ? C.a : V.alpha(C.muted, 0.4); g.lineWidth = 1.5; g.stroke();
          T(g, String(l), cx, cy + R + 13, { size: 11, align: 'center', font: 'mono', color: on ? C.ink : C.muted });
        }
        const nOn = phase === 0 ? 32 : 16;
        wrap(g, `lanes ativas: ${nOn}/32 · eficiência ${Math.round((nOn / 32) * 100)}%` + (phase ? ' · divergência: o warp percorre os dois lados do if' : ' · 1 instrução, 32 resultados'), 16, h - 34, w - 32, 16, { size: 12.5, weight: 600, color: phase ? C.c : C.a, maxLines: 2 });
      }

      function thread(g, w, h, t, C) {
        const pad = 14, colW = Math.min(170, w * 0.4);
        T(g, 'registradores da thread', pad, 20, { size: 12, weight: 600, maxW: colW + 40 });
        const n = regs.length, rh = Math.min(22, (h - 120) / n);
        for (let k = 0; k < n; k++) {
          const y = 30 + k * rh, hot = k < 3 && Math.floor(t * 2.5) % 2 === 0;
          V.box(g, pad, y, colW, rh - 3, { r: 4, fill: hot ? V.alpha(C.a, 0.4) : V.alpha(C.a, 0.1), stroke: V.alpha(C.a, 0.5), lw: 1 });
          T(g, 'r' + k, pad + 8, y + (rh - 3) / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, base: 'middle' });
          T(g, cm(regs[k] + Math.sin(t * 0.7 + k) * 0.5, 2), pad + colW - 8, y + (rh - 3) / 2 + 0.5, { size: 11, font: 'mono', align: 'right', base: 'middle' });
        }
        T(g, '⋮  até r254', pad + 8, 30 + n * rh + 12, { size: 11, font: 'mono', color: C.muted });
        const rx = pad + colW + 26, rw = w - rx - pad;
        const alu = { x: rx, y: 30, w: rw, h: 48 };
        V.box(g, alu.x, alu.y, alu.w, alu.h, { r: 8, fill: V.alpha(C.d, 0.14), stroke: C.d, label: 'FFMA  r0 = r1 × r2 + r3', font: 'mono', size: 11.5 });
        for (let k = 0; k < 3; k++) {
          const yy = 30 + (k + 1) * rh + rh / 2, u = (t * 2.4 + k * 0.2) % 1;
          V.line(g, pad + colW, yy, alu.x, alu.y + alu.h / 2, { color: V.alpha(C.a, 0.3), lw: 1 });
          const [px, py] = V.along(pad + colW, yy, alu.x, alu.y + alu.h / 2, u); V.dot(g, px, py, 2.4, C.a);
        }
        wrap(g, 'registrador → unidade: no ritmo da instrução', rx, alu.y + alu.h + 16, rw, 15, { size: 11.5, color: C.a, maxLines: 2 });
        const lm = { x: rx, y: alu.y + alu.h + 52, w: rw, h: 40 };
        V.box(g, lm.x, lm.y, lm.w, lm.h, { r: 8, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.7), label: 'memória local (spill)', size: 11.5 });
        const dr = { x: pad, y: h - 50, w: w - 2 * pad, h: 32 };
        V.box(g, dr.x, dr.y, dr.w, dr.h, { r: 8, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.6), label: 'DRAM: é aqui que a memória local mora de verdade', size: 11.5 });
        const x1 = lm.x + lm.w / 2, y1 = lm.y + lm.h, y2 = dr.y;
        V.line(g, x1, y1, x1, y2, { color: V.alpha(C.c, 0.5), lw: 1.5, dash: [4, 5], dashOffset: -t * 10 });
        const u2 = (t / 3.2) % 1, pp = u2 < 0.5 ? u2 * 2 : 2 - u2 * 2;
        V.dot(g, x1, V.lerp(y1, y2, pp), 3.2, C.c);
        V.line(g, pad + colW, 30 + (n - 1) * rh, lm.x, lm.y + lm.h / 2, { color: V.alpha(C.c, 0.35), lw: 1, dash: [3, 4] });
        if (y2 - y1 > 40) wrap(g, 'STL/LDL: centenas de ciclos por acesso', x1 + 10, (y1 + y2) / 2, rw / 2 + 20, 14, { size: 11.5, color: C.c, maxLines: 3 });
      }

      function kernel(g, w, h, t, C) {
        const nC = 64, pad = 12, narrow = w < 560;
        T(g, 'grid: 64 CTAs × 256 threads (8 warps cada)', pad, 18, { size: 12, weight: 600, maxW: w - 24 });
        const period = 7, ph = (t % period) / period, wave = ph < 0.5 ? 0 : 1, wp = (ph % 0.5) / 0.5;
        const gcols = narrow ? 8 : 16, grows = nC / gcols, gg = 4;
        const gcw = Math.min(34, (w - 2 * pad - (gcols - 1) * gg) / gcols), gch = Math.min(20, gcw * 0.62);
        const gX = (w - (gcols * gcw + (gcols - 1) * gg)) / 2, gY = 28;
        const smTop = gY + grows * (gch + gg) + 26;
        const scol = narrow ? 8 : 12, srow = 48 / scol, sg = 5;
        const scs = Math.max(8, Math.min((w - 2 * pad - (scol - 1) * sg) / scol, (h - smTop - 40 - (srow - 1) * sg) / srow));
        const sX = (w - (scol * scs + (scol - 1) * sg)) / 2;
        const smXY = (k) => [sX + (k % scol) * (scs + sg), smTop + Math.floor(k / scol) * (scs + sg)];
        T(g, '48 SMs (ilustrativo: 1 CTA por SM)', pad, smTop - 8, { size: 11.5, color: C.muted, maxW: w - 24 });
        for (let k = 0; k < 48; k++) { const [x, y] = smXY(k); V.box(g, x, y, scs, scs, { r: 4, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 }); }
        for (let c = 0; c < nC; c++) {
          const cw_ = c < 48 ? 0 : 1;
          if (cw_ < wave) continue;
          const gx = gX + (c % gcols) * (gcw + gg), gy = gY + Math.floor(c / gcols) * (gch + gg);
          let x = gx, y = gy, ww = gcw, hh = gch, run = 0;
          if (cw_ === wave) {
            const k = cw_ === 0 ? c : c - 48, [sx, sy] = smXY(k);
            const mv = V.ease.inOut(V.clamp(wp / 0.35, 0, 1));
            x = V.lerp(gx, sx + 2, mv); y = V.lerp(gy, sy + 2, mv); ww = V.lerp(gcw, scs - 4, mv); hh = V.lerp(gch, scs - 4, mv);
            run = V.clamp((wp - 0.35) / 0.6, 0, 1);
          }
          const col = C.series[cw_ === 0 ? 0 : 2];
          V.box(g, x, y, ww, hh, { r: 3, fill: V.alpha(col, 0.22), stroke: V.alpha(col, 0.8), lw: 1 });
          const sw = ww / 8;
          for (let s = 0; s < 8; s++) { g.fillStyle = V.alpha(col, s / 8 < run ? 0.85 : 0.35); g.fillRect(x + s * sw + 1, y + 2, Math.max(1, sw - 2), Math.max(1, hh - 4)); }
        }
        T(g, wave === 0 ? 'onda 1: 48 CTAs ocupam os 48 SMs' : 'onda 2: as 16 CTAs restantes; 32 SMs ociosos (efeito de cauda)', w / 2, h - 14, { size: 12, align: 'center', weight: 600, color: wave ? C.c : C.a, maxW: w - 20 });
      }

      const scenes = [chip, sm, warp, thread, kernel];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const e = V.ease.out(Math.min(1, st.elapsed() / 0.5));
        g.globalAlpha = 0.25 + 0.75 * e;
        const sc = 0.96 + 0.04 * e;
        g.translate((w / 2) * (1 - sc), (h / 2) * (1 - sc)); g.scale(sc, sc);
        scenes[st.i](g, w, h, t, C);
        g.globalAlpha = 1;
      });
    }

    /* ================= Figura 2: borboleta de shuffles ================= */
    {
      const fig = $('#fig-shuffle');
      const S = { n: 8, stage: 3, norm: true, auto: true, seed: 7 };
      let x = [], rows = [];
      function build() {
        const r = V.rng(S.seed * 31 + S.n);
        x = Array.from({ length: S.n }, () => r.int(9) - 4);
        rows = [x.slice()];
        const L = Math.log2(S.n);
        for (let s = 0; s < L; s++) { const m = 1 << s, prev = rows[s]; rows.push(prev.map((v, l) => ((l & m) ? prev[l ^ m] - v : v + prev[l ^ m]))); }
      }
      build();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Lanes', options: [8, 16, 32], value: 8, onChange: (v) => { S.n = v; build(); const L = Math.log2(v); sl.input.max = L; if (S.stage > L) S.stage = L; sl.input.value = S.stage; const vv = sl.el.querySelector('.val'); if (vv) vv.textContent = String(S.stage); } });
      let tg = null;
      const sl = UI.slider(ctl, { label: 'Estágio', min: 0, max: 3, step: 1, value: 3, onInput: (v) => { S.stage = Math.min(v, Math.log2(S.n)); if (S.auto && tg) tg.set(false); } });
      tg = UI.toggle(ctl, { label: 'avançar sozinho', value: true, onChange: (v) => { S.auto = v; } });
      UI.toggle(ctl, { label: '÷√n (normalizar)', value: true, onChange: (v) => { S.norm = v; } });
      UI.button(ctl, '🎲 novo vetor', () => { S.seed++; build(); });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), n = S.n, L = Math.log2(n);
        const seq = [L, L].concat(Array.from({ length: L }, (_, k) => k));
        const cur = S.auto ? seq[Math.floor(t / 1.5) % seq.length] : Math.min(S.stage, L);
        const lw = w < 520 ? 34 : 66, top = 30, bottom = 60;
        const colW = (w - lw - 12) / n, rowH = (h - top - bottom) / (L + 1);
        const ch = Math.min(28, rowH * 0.56), cw = Math.max(3, colW - (n > 16 ? 2 : 4));
        const cx = (l) => lw + l * colW + colW / 2, cy = (r) => top + r * rowH + ch / 2;
        let mx = 1; rows.forEach((row) => row.forEach((v) => { if (Math.abs(v) > mx) mx = Math.abs(v); }));
        T(g, cur < L ? `próximo: estágio ${cur} · parceira = lane ⊕ ${1 << cur}` : `pronto: ${L} estágios = log₂(${n})`, lw, 17, { size: 12.5, weight: 600, maxW: w - lw - 8 });
        for (let s = 0; s < L; s++) {
          const m = 1 << s, done = s < cur, isLast = s === cur - 1;
          for (let l = 0; l < n; l++) {
            const p = l ^ m, x1 = cx(l), y1 = cy(s) + ch / 2, x2 = cx(p), y2 = cy(s + 1) - ch / 2;
            const col = done ? (isLast ? C.c : V.alpha(C.muted, 0.35)) : V.alpha(C.muted, 0.13);
            V.line(g, x1, y1, x1, y2, { color: col, lw: isLast ? 1.4 : 1 });
            V.line(g, x1, y1, x2, y2, { color: col, lw: isLast ? 1.4 : 1, dash: done ? null : [3, 4] });
            if (isLast) { const u = (t * 0.9 + l * 0.07) % 1; const [px, py] = V.along(x2, y1, x1, y2, u); V.dot(g, px, py, 2.2, C.c); }
          }
        }
        for (let r = 0; r <= L; r++) {
          T(g, r === 0 ? 'x' : 's=' + (r - 1), 6, cy(r) + 4, { size: 11, font: 'mono', color: C.muted });
          if (lw > 40 && r > 0) T(g, '⊕' + (1 << (r - 1)), 6, cy(r) + 17, { size: 11, font: 'mono', color: C.muted });
          for (let l = 0; l < n; l++) {
            const shown = r <= cur, v = rows[r][l], fin = r === L && S.norm;
            const vv = fin ? v / Math.sqrt(n) : v;
            const x0 = cx(l) - cw / 2, y0 = cy(r) - ch / 2, tt = Math.abs(v / mx);
            g.fillStyle = shown ? V.cmap(v / mx, 'div') : V.alpha(C.muted, 0.06);
            V.rr(g, x0, y0, cw, ch, Math.min(5, cw / 4)); g.fill();
            if (r === cur && r > 0 && shown) { g.strokeStyle = C.c; g.lineWidth = 1.6; V.rr(g, x0 - 1, y0 - 1, cw + 2, ch + 2, 5); g.stroke(); }
            if (shown && cw >= 18) T(g, fin ? cm(vv, cw >= 36 ? 2 : 1) : String(vv), cx(l), cy(r) + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: cellTxt(C, tt > 0.55) });
          }
        }
        if (colW >= 16) for (let l = 0; l < n; l++) T(g, String(l), cx(l), cy(L) + ch / 2 + 13, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const ref = LA.num.fwht(x), ok = ref.every((v, i) => Math.abs(v - rows[L][i] / Math.sqrt(n)) < 1e-9);
        T(g, 'v = (lane & m) ? shfl_xor(v, m) − v : v + shfl_xor(v, m)', lw, h - 26, { size: 11.5, font: 'mono', color: C.muted, maxW: w - lw - 8 });
        T(g, cur === L ? (ok ? '✓ igual a LA.num.fwht(x)' + (S.norm ? '' : ' (depois de ÷√n)') + ': H·x só com registradores' : '✗ divergiu da referência') : `${cur} de ${L} estágios · nenhuma leitura de memória`, lw, h - 9, { size: 12, weight: 600, color: cur === L ? C.a : C.ink, maxW: w - lw - 8 });
      });
    }

    /* =================== Figura 3: calculadora de ocupação =================== */
    {
      const fig = $('#fig-occ');
      const ARCH = { c: { name: 'SM 12.x (GB10)', R: 65536, S: 100, Sc: 99, W: 48 }, d: { name: 'SM 9.0/10.0 (datacenter)', R: 65536, S: 228, Sc: 227, W: 64 } };
      const S = { a: 'c', T: 256, r: 96, s: 32, cap: 255 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'SM', options: [{ v: 'c', label: '12.x · GB10' }, { v: 'd', label: '9.0/10.0 · datacenter' }], value: 'c', onChange: (v) => { S.a = v; upd(); } });
      UI.slider(ctl, { label: 'Threads/CTA', min: 32, max: 1024, step: 32, value: 256, onInput: (v) => { S.T = v; upd(); } });
      UI.slider(ctl, { label: 'Regs/thread', min: 16, max: 255, step: 1, value: 96, onInput: (v) => { S.r = v; upd(); } });
      UI.slider(ctl, { label: 'SMEM por CTA', min: 0, max: 228, step: 2, value: 32, fmt: (v) => v + ' KB', onInput: (v) => { S.s = v; upd(); } });
      UI.slider(ctl, { label: 'Teto de regs', min: 32, max: 255, step: 1, value: 255, onInput: (v) => { S.cap = v; upd(); } });
      const rN = UI.readout(ctl, 'CTAs/SM', '—'), rW = UI.readout(ctl, 'warps/SM', '—'), rO = UI.readout(ctl, 'ocupação', '—'), rL = UI.readout(ctl, 'limita', '—');
      let M = null;
      function calc() {
        const A = ARCH[S.a], rEff = Math.min(S.r, S.cap), spill = Math.max(0, S.r - S.cap), warpsC = S.T / 32;
        const byR = Math.floor(A.R / (rEff * S.T)), byS = S.s > 0 ? Math.floor(A.S / S.s) : Infinity, byW = Math.floor(A.W / warpsC);
        const fits = S.s <= A.Sc && rEff * S.T <= A.R;
        const n = fits ? Math.min(byR, byS, byW) : 0;
        const lim = !fits ? (S.s > A.Sc ? 'smem' : 'reg') : n === byR ? 'reg' : n === byS ? 'smem' : 'warp';
        return { A, rEff, spill, warpsC, n, lim, fits, occ: (n * warpsC) / A.W, warps: n * warpsC };
      }
      const limName = { reg: 'registradores', smem: 'shared memory', warp: 'vagas de warp' };
      function upd() { M = calc(); rN.set(String(M.n)); rW.set(`${M.warps}/${M.A.W}`); rO.set(Math.round(M.occ * 100) + '%'); rL.set(M.fits ? limName[M.lim] : 'não lança'); }
      upd();
      fitH(fig, (w) => (w < 560 ? 472 : 420));
      const wr = V.rng(9), wph = Array.from({ length: 64 }, () => wr());
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), m = M, A = m.A, pad = 14, narrow = w < 560;
        wrap(g, `${A.name}: ${fmtInt(A.R)} registradores · ${A.S} KB de SMEM · ${A.W} warps por SM (tabela pública)`, pad, 18, w - 2 * pad, 15, { size: 12, weight: 600, maxLines: 2 });
        const bars = [
          { key: 'reg', label: 'banco de registradores', cap: A.R, per: m.rEff * S.T, unit: (x) => fmtInt(x) + ' regs' },
          { key: 'smem', label: 'shared memory', cap: A.S, per: S.s, unit: (x) => x + ' KB' },
          { key: 'warp', label: 'vagas de warp', cap: A.W, per: m.warpsC, unit: (x) => x + ' warps' },
        ];
        const bx = narrow ? pad : 168, bw = w - bx - pad, bh = 20;
        let y = narrow ? 46 : 36;
        bars.forEach((b) => {
          if (narrow) { T(g, b.label, pad, y + 10, { size: 11.5, color: C.muted }); y += 15; } else T(g, b.label, pad, y + bh / 2 + 4, { size: 11.5, color: C.muted });
          V.box(g, bx, y, bw, bh, { r: 5, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
          const fr = Math.min(1, b.per / b.cap);
          if (b.key === 'warp') { g.strokeStyle = V.alpha(C.muted, 0.18); g.lineWidth = 1; for (let k = 1; k < A.W; k++) { const xx = bx + (k * bw) / A.W; g.beginPath(); g.moveTo(xx, y + 3); g.lineTo(xx, y + bh - 3); g.stroke(); } }
          for (let k = 0; k < m.n; k++) { const x0 = bx + k * fr * bw; const ww = Math.min(Math.max(1, fr * bw - 2), bx + bw - x0 - 2); if (ww <= 0) break; g.fillStyle = V.alpha(C.series[k % 6], 0.72); V.rr(g, x0 + 1, y + 2, ww, bh - 4, 3); g.fill(); }
          const isLim = m.lim === b.key;
          if (isLim) { g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, bx - 2, y - 2, bw + 4, bh + 4, 6); g.stroke(); }
          const used = b.per * m.n;
          T(g, `${b.unit(b.per)} por CTA · ${Math.round((100 * used) / b.cap)}% usado${isLim ? ' · LIMITA' : ''}`, bx + bw, y + bh + 13, { size: 11, align: 'right', color: isLim ? C.c : C.muted, font: 'mono', maxW: bw });
          y += bh + 22;
        });
        y += 6;
        T(g, 'escalonadores: warps residentes se revezando', pad, y + 10, { size: 11.5, weight: 600, maxW: w - 2 * pad });
        y += 18;
        const slotW = Math.max(3, Math.min(14, (w - 2 * pad) / A.W - 2));
        for (let k = 0; k < A.W; k++) {
          const x0 = pad + k * (slotW + 2);
          if (k >= m.warps) { g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; V.rr(g, x0, y, slotW, 16, 3); g.stroke(); continue; }
          const ph = (t * 0.45 + wph[k]) % 1, busy = ph < 1 / 12;
          g.fillStyle = busy ? C.a : V.alpha(C.b, 0.32); V.rr(g, x0, y, slotW, 16, 3); g.fill();
        }
        y += 24;
        const util = Math.min(1, m.warps / 48);
        const mw = w - 2 * pad;
        V.box(g, pad, y, mw, 14, { r: 4, fill: V.alpha(C.muted, 0.1) });
        V.box(g, pad, y, Math.max(2, mw * util), 14, { r: 4, fill: V.alpha(C.a, 0.7) });
        y += 28;
        wrap(g, `unidades ocupadas ≈ ${Math.round(util * 100)}%. Ilustrativo: cada warp calcula 1 unidade e espera ~11 pela DRAM, então cada uma das 4 sub-partições precisa de ~12 warps prontos.`, pad, y, mw, 15, { size: 11.5, color: C.muted, maxLines: 3 });
        y += 40;
        let boxH = 44;
        if (!m.fits) {
          const msg = S.s > A.Sc ? `não lança: ${S.s} KB de SMEM por CTA passam do máximo de ${A.Sc} KB deste SM` : `não lança: ${m.rEff} × ${S.T} = ${fmtInt(m.rEff * S.T)} registradores numa CTA passam dos ${fmtInt(A.R)} do SM`;
          const n = Math.min(3, splitLines(g, msg, mw - 20, { size: 12, weight: 600 }).length); boxH = Math.max(40, 12 + n * 15 + 6);
          V.box(g, pad, y, mw, boxH, { r: 8, fill: V.alpha(C.c, 0.14), stroke: C.c });
          wrap(g, msg, pad + 10, y + 16, mw - 20, 15, { size: 12, weight: 600, color: C.c, maxLines: 3 });
        } else if (m.spill > 0) {
          const bytes = m.spill * 4;
          const msg = `spill: ${m.spill} registradores = ${bytes} B por thread na memória local (DRAM): ${fmtInt(bytes * m.n * S.T)} B por SM de STL/LDL a cada passada. Na receita GLM, o FAST zerou os LDL (39 → 0).`;
          const n = Math.min(4, splitLines(g, msg, mw - 36, { size: 11.5 }).length); boxH = Math.max(44, 12 + n * 15 + 6);
          V.box(g, pad, y, mw, boxH, { r: 8, fill: V.alpha(C.c, 0.1), stroke: V.alpha(C.c, 0.6) });
          for (let k = 0; k < 6; k++) { const u = (t * 0.6 + k / 6) % 1; V.dot(g, pad + 14, y + 6 + ((u + k / 6) % 1) * (boxH - 12), 2.4, C.c); }
          wrap(g, msg, pad + 26, y + 16, mw - 36, 15, { size: 11.5, maxLines: 4 });
        } else {
          T(g, '✓ sem spill: tudo cabe nos registradores permitidos', pad, y + 16, { size: 12, weight: 600, color: C.a, maxW: mw });
        }
        wrap(g, 'Caso real (GLM, SM121): stock exl3_moe 128 regs, frame 88 B, 39 LDL → FAST 127 regs, 0 LDL, mesma SMEM de 1.024 B e mesma ocupação: +8–17% por camada.', pad, y + boxH + 18, mw, 15, { size: 11.5, color: C.glm, maxLines: 3 });
      });
    }

    /* =================== Figura 4: instrução de tensor core =================== */
    {
      const fig = $('#fig-mma');
      const S = { mode: 'fp16', lanes: false };
      const r = V.rng(21);
      const A16 = Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => r.int(7) - 3));
      const B16 = Array.from({ length: 16 }, () => Array.from({ length: 8 }, () => r.int(5) - 2));
      const E2 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
      const pick = () => (r() < 0.5 ? -1 : 1) * E2[r.int(8)];
      const A4 = Array.from({ length: 16 }, () => Array.from({ length: 64 }, pick));
      const B4 = Array.from({ length: 64 }, () => Array.from({ length: 8 }, pick));
      const SC = [0.5, 1, 2];
      const sA = Array.from({ length: 16 }, () => Array.from({ length: 4 }, () => SC[r.int(3)]));
      const sB = Array.from({ length: 4 }, () => Array.from({ length: 8 }, () => SC[r.int(3)]));
      const D16 = A16.map((row) => Array.from({ length: 8 }, (_, j) => row.reduce((s, a, k) => s + a * B16[k][j], 0)));
      const part4 = (i, j, b) => { let s = 0; for (let k = b * 16; k < b * 16 + 16; k++) s += A4[i][k] * B4[k][j]; return s; };
      const D4 = Array.from({ length: 16 }, (_, i) => Array.from({ length: 8 }, (_, j) => { let s = 0; for (let b = 0; b < 4; b++) s += sA[i][b] * sB[b][j] * part4(i, j, b); return s; }));
      let mD4 = 0; D4.forEach((row) => row.forEach((v) => { mD4 = Math.max(mD4, Math.abs(v)); }));
      let mD16 = 0; D16.forEach((row) => row.forEach((v) => { mD16 = Math.max(mD16, Math.abs(v)); }));
      // convenção de fragmentos do PTX para mma.m16n8k16 (groupID = lane >> 2, threadID_in_group = lane % 4)
      const laneA = (rr, c) => (rr % 8) * 4 + ((c % 8) >> 1);
      const laneB = (k, n) => n * 4 + ((k % 8) >> 1);
      const laneD = (rr, n) => (rr % 8) * 4 + (n >> 1);
      const nf = (x) => String(Math.round(x * 100) / 100).replace('.', ',');
      const ITEMS = {
        fp16: ['mma.sync.m16n8k16 · FP16 → FP32', '2·16·8·16 = 4.096 FLOPs numa instrução do warp', 'cada lane leva 8 elementos de A, 4 de B e 4 do acumulador', 'nenhuma lane tem a matriz inteira: ligue “colorir por lane”', 'passe o mouse (ou toque) sobre D para escolher a célula'],
        fp4: ['mma.sync … kind::mxf4nvf4 … m16n8k64', '16.384 FLOPs numa instrução do warp: K = 64, em 4 blocos de 16', 'cada lane leva 32 códigos E2M1 de A, 16 de B e 4 acumuladores FP32', 'sA (16×4) e sBᵀ (8×4): uma escala UE4M3 por bloco de 16 (scale_vec::4X)', 'os mesmos 768 B de operandos do FP16, com 4× os FLOPs'],
      };
      const MC = document.createElement('canvas').getContext('2d');
      const itemStyle = (k) => ({ size: k === 0 ? 11.5 : 12, font: k === 0 ? 'mono' : 'sans', weight: k === 0 ? 600 : 400 });
      function itemsH(items, width) { let hh = 0; items.forEach((it, k) => { hh += Math.min(3, splitLines(MC, it, width, itemStyle(k)).length) * 15 + 7; }); return hh; }
      // layout único, usado pelo desenho e pela altura do palco (fitH)
      function layout(w, fp4) {
        const pad = 12, narrow = w < 560, extra = fp4 ? 5 : 0, gB = fp4 ? 2 : 1.4;
        const ay = fp4 ? 58 : 22;
        const availW = w - 2 * pad - 26 - (narrow ? 0 : 250);
        const cs = Math.max(6, Math.min(narrow ? 22 : 12, availW / (extra + 16 + 1.6 + 8)));
        const ax = pad + 22 + extra * cs, dx = ax + 16 * cs + 1.6 * cs, by = ay + 16 * cs + gB * cs;
        const matBottom = by + 8 * cs + 20;
        const pX = dx + 8 * cs + 22, pW = w - pX - pad, side = !narrow && pW > 170;
        const all = ITEMS[fp4 ? 'fp4' : 'fp16'];
        // no estreito, a lista vai embaixo; some o item que só repete o mostrador (FP4) ou a dica de mouse (FP16)
        const items = side || !narrow ? all : all.filter((_, k) => (fp4 ? k !== 1 : k !== 4));
        const listY = matBottom + 4, listH = side ? 0 : itemsH(items, w - 2 * pad - 20) + 14;
        const lensTop = side ? matBottom + 4 : listY + listH;
        const lx = pad + (fp4 ? (narrow ? 70 : 64) : (narrow ? 40 : 64));
        let per = 16, lc = Math.min(fp4 ? 34 : 36, (w - pad - lx - 10) / 16);
        if (fp4 && lc < 24) { per = 8; lc = Math.min(34, (w - pad - lx - 10) / 8); }
        const ch2 = Math.min(fp4 ? 20 : 22, lc);
        const chipPer = narrow ? 2 : 4;
        const lensH = fp4 ? 28 + 2 * (16 / per) * (ch2 + 5) + 6 + (4 / chipPer) * 27 + 28 : 30 + 3 * (ch2 + 6) + 28;
        return { pad, narrow, extra, gB, ay, cs, ax, dx, by, matBottom, pX, pW, side, items, listY, listH, lensTop, lx, per, lc, ch2, chipPer, lensH, H: Math.round(lensTop + lensH + 4) };
      }
      const ctl = UI.controls(fig);
      let applyH = () => {};
      UI.seg(ctl, { label: 'Instrução', options: [{ v: 'fp16', label: 'FP16 · m16n8k16' }, { v: 'fp4', label: 'NVFP4 · m16n8k64' }], value: 'fp16', onChange: (v) => { S.mode = v; upd(); applyH(); } });
      UI.toggle(ctl, { label: 'colorir por lane (FP16)', value: false, onChange: (v) => { S.lanes = v; } });
      const ro1 = UI.readout(ctl, 'FLOPs/instrução', '—'), ro2 = UI.readout(ctl, 'por lane (A · B · C)', '—'), ro3 = UI.readout(ctl, 'bytes de A + B', '—');
      function upd() {
        if (S.mode === 'fp16') { ro1.set('4.096'); ro2.set('8 · 4 · 4'); ro3.set('768 B'); }
        else { ro1.set('16.384'); ro2.set('32 · 16 · 4'); ro3.set('768 B + 96 B esc.'); }
      }
      upd();
      applyH = fitH(fig, (w) => layout(w, S.mode === 'fp4').H);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), fp4 = S.mode === 'fp4', Lo = layout(w, fp4);
        const { pad, narrow, cs, ax, ay, dx, by } = Lo;
        const P = H.pointer;
        const inR = (x0, y0, ww, hh) => P.inside && P.x >= x0 && P.x < x0 + ww && P.y >= y0 && P.y < y0 + hh;
        // bloco em cena (FP4): automático, ou o que estiver sob o mouse na régua
        const stripY = 26, stripH = 16;
        let bb = fp4 ? Math.floor(t / 2.4) % 4 : 0;
        if (fp4 && inR(ax, stripY - 4, 16 * cs, stripH + 8)) bb = V.clamp(Math.floor((P.x - ax) / (4 * cs)), 0, 3);
        const k0 = bb * 16;
        const Aval = (i, k) => (fp4 ? A4[i][k0 + k] : A16[i][k]);
        const Bval = (j, k) => (fp4 ? B4[k0 + k][j] : B16[k][j]);
        const mA = fp4 ? 6 : 3, mB = fp4 ? 6 : 2, D = fp4 ? D4 : D16, mD = fp4 ? mD4 : mD16;
        let ci, cj;
        if (inR(dx, ay, 8 * cs, 16 * cs)) { cj = Math.floor((P.x - dx) / cs); ci = Math.floor((P.y - ay) / cs); }
        else { const k = Math.floor(t / 0.9) % 128; ci = Math.floor(k / 8); cj = k % 8; }
        const useLane = S.lanes && !fp4;
        let own = laneD(ci, cj);
        if (useLane && inR(ax, ay, 16 * cs, 16 * cs)) own = laneA(Math.floor((P.y - ay) / cs), Math.floor((P.x - ax) / cs));
        else if (useLane && inR(ax, by, 16 * cs, 8 * cs)) own = laneB(Math.floor((P.x - ax) / cs), Math.floor((P.y - by) / cs));
        const gp = cs >= 8 ? 1 : 0.5;
        const cell = (x, y, col) => { g.fillStyle = col; V.rr(g, x + gp, y + gp, cs - 2 * gp, cs - 2 * gp, Math.min(3, cs / 4)); g.fill(); };
        const ring = (x, y) => { g.strokeStyle = C.ink; g.lineWidth = 1.6; V.rr(g, x + 0.5, y + 0.5, cs - 1, cs - 1, Math.min(3, cs / 4)); g.stroke(); };
        for (let i = 0; i < 16; i++) for (let k = 0; k < 16; k++) { const ln = laneA(i, k); cell(ax + k * cs, ay + i * cs, useLane ? laneCol(ln, 0.9) : V.cmap(Aval(i, k) / mA, 'div')); if (useLane && ln === own) ring(ax + k * cs, ay + i * cs); }
        for (let j = 0; j < 8; j++) for (let k = 0; k < 16; k++) { const ln = laneB(k, j); cell(ax + k * cs, by + j * cs, useLane ? laneCol(ln, 0.9) : V.cmap(Bval(j, k) / mB, 'div')); if (useLane && ln === own) ring(ax + k * cs, by + j * cs); }
        for (let i = 0; i < 16; i++) for (let j = 0; j < 8; j++) { const ln = laneD(i, j); cell(dx + j * cs, ay + i * cs, useLane ? laneCol(ln, 0.9) : V.cmap(D[i][j] / mD, 'div')); if (useLane && ln === own) ring(dx + j * cs, ay + i * cs); }
        const bcol = C.series[bb];
        if (fp4) {
          // régua do eixo K: 4 blocos de 16, o bloco em cena em destaque
          T(g, `K = 64 = 4 blocos de 16 · em cena: bloco ${bb} (k = ${k0}…${k0 + 15})`, pad, 16, { size: 11.5, weight: 600, maxW: w - 2 * pad });
          for (let b = 0; b < 4; b++) {
            const x = ax + b * 4 * cs;
            V.box(g, x + 1, stripY, 4 * cs - 2, stripH, { r: 4, fill: V.alpha(C.series[b], b === bb ? 0.85 : 0.2), stroke: V.alpha(C.series[b], 0.8), lw: 1 });
            T(g, 'b' + b, x + 2 * cs, stripY + stripH / 2 + 0.5, { size: 11, align: 'center', base: 'middle', font: 'mono', color: b === bb ? onTxt(C) : C.muted });
          }
          const sx0 = ax + 16 * cs + 8;
          if (w - pad - sx0 > 60) T(g, '← passe o mouse', sx0, stripY + stripH / 2 + 4, { size: 11, color: C.muted, maxW: w - pad - sx0 });
          // molduras ligando A e Bᵀ ao bloco da régua
          g.strokeStyle = bcol; g.lineWidth = 1.6;
          V.rr(g, ax - 2, ay - 2, 16 * cs + 4, 16 * cs + 4, 5); g.stroke();
          V.rr(g, ax - 2, by - 2, 16 * cs + 4, 8 * cs + 4, 5); g.stroke();
          // escalas: sA (16×4) e sBᵀ (8×4), coluna do bloco em cena destacada
          const sx = ax - 5 * cs;
          for (let i = 0; i < 16; i++) for (let b = 0; b < 4; b++) cell(sx + b * cs, ay + i * cs, V.alpha(C.series[b], 0.25 + 0.5 * (sA[i][b] / 2)));
          for (let j = 0; j < 8; j++) for (let b = 0; b < 4; b++) cell(sx + b * cs, by + j * cs, V.alpha(C.series[b], 0.25 + 0.5 * (sB[b][j] / 2)));
          g.strokeStyle = bcol; g.lineWidth = 1.4;
          V.rr(g, sx + bb * cs, ay - 1, cs, 16 * cs + 2, 3); g.stroke();
          V.rr(g, sx + bb * cs, by - 1, cs, 8 * cs + 2, 3); g.stroke();
          T(g, 'sA', sx + 2 * cs, ay - 6, { size: 11, align: 'center', font: 'mono', color: C.muted });
          T(g, 'sBᵀ', sx + 2 * cs, by - 5, { size: 11, align: 'center', font: 'mono', color: C.muted });
        }
        const hl = (x, y, ww, hh, col) => { g.strokeStyle = col; g.lineWidth = 2; V.rr(g, x - 1.5, y - 1.5, ww + 3, hh + 3, 4); g.stroke(); };
        if (!useLane) { hl(ax, ay + ci * cs, 16 * cs, cs, C.c); hl(ax, by + cj * cs, 16 * cs, cs, C.c); }
        hl(dx + cj * cs, ay + ci * cs, cs, cs, C.c);
        V.glow(g, dx + cj * cs + cs / 2, ay + ci * cs + cs / 2, cs * 1.6, C.c, 0.5);
        T(g, fp4 ? `A · bloco ${bb}` : 'A (16×16)', ax, ay - 6, { size: 11, font: 'mono', color: fp4 ? bcol : C.muted });
        T(g, 'D (16×8)', Math.min(dx, w - pad - 58), ay - 6, { size: 11, font: 'mono', color: C.muted });
        T(g, fp4 ? `Bᵀ · bloco ${bb}` : 'Bᵀ (8×16)', ax, by + 8 * cs + 15, { size: 11, font: 'mono', color: fp4 ? bcol : C.muted });
        T(g, '=', ax + 16 * cs + 0.8 * cs, ay + 8 * cs + 4, { size: 14, align: 'center', color: C.muted, weight: 600 });
        // explicação: painel ao lado (desktop) ou lista embaixo das matrizes (estreito)
        const items = Lo.items;
        const drawItems = (x0, y0, ww) => { let yy = y0; items.forEach((it, k) => { const n = wrap(g, it, x0, yy, ww, 15, Object.assign({ color: k === 0 ? C.a : C.ink, maxLines: 3 }, itemStyle(k))); yy += n * 15 + 7; }); };
        if (Lo.side) {
          const pH = Math.min(useLane ? 16 * cs : by + 8 * cs - ay, itemsH(items, Lo.pW - 24) + 26);
          V.box(g, Lo.pX, ay, Lo.pW, pH, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.35), lw: 1 });
          drawItems(Lo.pX + 12, ay + 22, Lo.pW - 24);
        } else {
          V.box(g, pad, Lo.listY, w - 2 * pad, Lo.listH - 6, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.3), lw: 1 });
          drawItems(pad + 10, Lo.listY + 18, w - 2 * pad - 20);
        }
        if (useLane) {
          // legenda das lanes: cor = lane % 4, tom = lane ÷ 4; contorno = tudo o que a lane "own" guarda
          const lx = dx, lw2 = w - pad - dx;
          let yy = by + 10;
          T(g, 'cor = lane % 4', lx, yy, { size: 11, color: C.muted, maxW: lw2 }); yy += 6;
          for (let q = 0; q < 4; q++) { g.fillStyle = laneCol(q + 12, 0.95); V.rr(g, lx + q * 18, yy, 14, 12, 3); g.fill(); T(g, String(q), lx + q * 18 + 7, yy + 6.5, { size: 11, align: 'center', base: 'middle', font: 'mono', color: onTxt(C) }); }
          yy += 26;
          T(g, 'tom = lane ÷ 4', lx, yy, { size: 11, color: C.muted, maxW: lw2 }); yy += 6;
          const sw = Math.min(12, (lw2 - 2) / 8);
          for (let q = 0; q < 8; q++) { g.fillStyle = laneCol(q * 4, 0.95); V.rr(g, lx + q * sw, yy, sw - 2, 10, 2); g.fill(); }
          yy += 26;
          if (lw2 < 200) { T(g, `lane ${own} em contorno`, lx, yy, { size: 11, color: C.ink, maxW: lw2 }); T(g, `grupo ${own >> 2} · posição ${own & 3}`, lx, yy + 14, { size: 11, color: C.muted, maxW: lw2 }); }
          else wrap(g, `contorno: tudo o que a lane ${own} guarda (grupo ${own >> 2}, posição ${own & 3})`, lx, yy, lw2, 14, { size: 11, color: C.ink, maxLines: 3 });
        }
        // lupa
        const ly0 = Lo.lensTop, lensH = Lo.lensH, lx = Lo.lx, lc = Lo.lc, ch2 = Lo.ch2;
        V.box(g, pad, ly0, w - 2 * pad, lensH - 6, { r: 10, fill: V.alpha(C.c, 0.05), stroke: V.alpha(C.c, 0.4), lw: 1 });
        if (!fp4) {
          T(g, `lupa: D[${ci}][${cj}] = A[${ci},·] · Bᵀ[${cj},·]`, pad + 10, ly0 + 18, { size: 12, weight: 600, maxW: w - 2 * pad - 20 });
          const rowsL = [['A', (k) => A16[ci][k]], ['Bᵀ', (k) => B16[k][cj]], ['×', (k) => A16[ci][k] * B16[k][cj]]];
          rowsL.forEach(([lab, f], ri) => {
            const yy = ly0 + 30 + ri * (ch2 + 6);
            T(g, lab, pad + 12, yy + ch2 / 2 + 4, { size: 11.5, font: 'mono', color: C.muted });
            for (let k = 0; k < 16; k++) {
              const v = f(k), mm = ri === 2 ? 9 : 3, tt = Math.abs(v) / mm;
              g.fillStyle = V.cmap(v / mm, 'div'); V.rr(g, lx + k * lc + 1, yy, lc - 2, ch2, 3); g.fill();
              if (lc >= 15) T(g, String(v), lx + k * lc + lc / 2, yy + ch2 / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: cellTxt(C, tt > 0.55) });
            }
          });
          T(g, `Σ dos 16 produtos = ${D16[ci][cj]}  →  D[${ci}][${cj}] (acumulado em FP32)`, pad + 12, ly0 + 30 + 3 * (ch2 + 6) + 12, { size: 12, weight: 600, color: C.c, maxW: w - 2 * pad - 24 });
        } else {
          T(g, `lupa: D[${ci}][${cj}] = Σ_b sA·sB·(A_b · Bᵀ_b), bloco ${bb} em cena`, pad + 10, ly0 + 18, { size: 12, weight: 600, maxW: w - 2 * pad - 20 });
          const per = Lo.per, nL = 16 / per;
          let yy = ly0 + 28;
          for (let hf = 0; hf < nL; hf++) {
            [['A', (k) => A4[ci][k0 + k]], ['Bᵀ', (k) => B4[k0 + k][cj]]].forEach(([lab, f]) => {
              const lab2 = nL > 1 ? `${lab} k${hf * 8}–${hf * 8 + 7}` : `${lab} b${bb}`;
              T(g, lab2, pad + 10, yy + ch2 / 2 + 4, { size: 11, font: 'mono', color: bcol, maxW: lx - pad - 14 });
              for (let q = 0; q < per; q++) {
                const k = hf * per + q, v = f(k), tt = Math.abs(v) / 6;
                g.fillStyle = V.cmap(v / 6, 'div'); V.rr(g, lx + q * lc + 1, yy, lc - 2, ch2, 3); g.fill();
                if (lc >= 24) T(g, nf(v), lx + q * lc + lc / 2, yy + ch2 / 2 + 0.5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: cellTxt(C, tt > 0.55) });
              }
              yy += ch2 + 5;
            });
          }
          const cy0 = yy + 6, perRow = Lo.chipPer, chipW = (w - 2 * pad - 20 - (perRow - 1) * 6) / perRow, chipH = 22;
          for (let b = 0; b < 4; b++) {
            const p = part4(ci, cj, b), c = sA[ci][b] * sB[b][cj] * p;
            const x0 = pad + 10 + (b % perRow) * (chipW + 6), y0 = cy0 + Math.floor(b / perRow) * (chipH + 5);
            V.box(g, x0, y0, chipW, chipH, { r: 5, fill: V.alpha(C.series[b], b === bb ? 0.3 : 0.12), stroke: V.alpha(C.series[b], b === bb ? 1 : 0.6), lw: b === bb ? 1.6 : 1 });
            T(g, narrow ? `b${b}: ${nf(p)}×${nf(sA[ci][b] * sB[b][cj])} = ${nf(c)}` : `b${b}: ${nf(p)} × ${nf(sA[ci][b])} × ${nf(sB[b][cj])} = ${nf(c)}`, x0 + 6, y0 + chipH / 2 + 0.5, { size: 11, font: 'mono', base: 'middle', maxW: chipW - 10 });
          }
          T(g, `D[${ci}][${cj}] = Σ_b sA·sB·(Σ do bloco) = ${nf(D4[ci][cj])}`, pad + 10, ly0 + lensH - 18, { size: 12, weight: 600, color: C.c, maxW: w - 2 * pad - 20 });
        }
      });
    }

    /* =================== Figura 5: gates de capability =================== */
    {
      const fig = $('#fig-gates');
      const GPUS = [
        { v: 'a100', label: 'A100 · 8.0', cc: [8, 0] }, { v: 'h100', label: 'H100 · 9.0', cc: [9, 0] }, { v: 'b200', label: 'B200 · 10.0', cc: [10, 0] },
        { v: 'gb300', label: 'GB300 · 10.3', cc: [10, 3] }, { v: 'rtx', label: 'RTX PRO 6000 · 12.0', cc: [12, 0] }, { v: 'gb10', label: 'GB10 · 12.1', cc: [12, 1] },
      ];
      const S = { gpu: 'gb10', patch: false, t0: -100, changed: false };
      const eq = (c, a, b) => c[0] === a && c[1] === b;
      const val = (c) => c[0] * 10 + c[1];
      const G = [
        { rc: 'glm', n: 'EXL3 aceita a GPU', ex: () => 'get_min_capability() = 80', ev: (c) => (val(c) >= 80 ? ['ok', 'método exl3 disponível'] : ['err', 'recusado']) },
        { rc: 'glm', n: 'Tier E3 (atomics de 16 B)', ex: () => 'cc >= (9, 0)  # red.v4', ev: (c) => (val(c) >= 90 ? ['ok', 'E3: atomicAdd(float4)'] : ['fb', 'cai para E2']) },
        { rc: 'glm', n: 'PDL (lançamento dependente)', ex: (p) => (p ? 'major in (9, 10)' : 'major >= 9'), ev: (c, p) => (c[0] === 9 || c[0] === 10 ? ['ok', 'PDL ligado'] : c[0] >= 9 ? (p ? ['fb', 'desligado: evita a corrida'] : ['warn', 'liga, mas corre contra os kernels de estado KDA']) : ['fb', 'desligado']) },
        { rc: 'qwen', n: 'Tabela QSA sm_120 (v0.30)', ex: () => '_is_sm120(): cc == (12, 0)', ev: (c) => (eq(c, 12, 0) ? ['ok', 'tabela afinada no sm_120'] : ['fb', 'tabela padrão (GB300)' + (eq(c, 12, 1) ? ' · inferido do código' : '')]) },
        { rc: 'qwen', n: 'Prefill GDN no FlashInfer', ex: () => 'SM90 ou SM10x', ev: (c) => (c[0] === 9 || c[0] === 10 ? ['ok', 'FlashInfer/CuteDSL'] : ['fb', 'Triton FLA chunk_gated_delta_rule']) },
        { rc: 'qwen', n: 'MoE NVFP4: backend', ex: () => 'TRTLLM → CuteDSL (SM10x) → CUTLASS', ev: (c) => (c[0] === 10 ? ['ok', 'TRTLLM (1ª opção)'] : c[0] === 12 ? ['ok', 'CUTLASS SM120 (3ª opção)'] : ['fb', 'sem FP4 nativo']) },
        { rc: 'qwen', n: 'Skinny GEMM CuteDSL (decode)', ex: () => '_is_sm103()', ev: (c) => (eq(c, 10, 3) ? ['ok', 'kernel skinny (planos TP=4)'] : ['fb', 'cuBLAS']) },
        { rc: 'ds', n: 'Cache K MXFP4 do indexer', ex: (p) => (p ? 'family(100) or family(120)' : 'family(100)'), ev: (c, p) => (c[0] === 10 || (p && c[0] === 12) ? ['ok', 'cache FP4: metade dos bytes'] : ['fb', 'recusado: “requires Blackwell datacenter”']) },
        { rc: 'ds', n: 'DeepGEMM: header do mqa_logits', ex: (p) => (p ? 'sm121_X → #define sm120_X' : 'impls/sm{XY}_fp8_mqa_logits.cuh'), ev: (c, p) => (eq(c, 12, 0) ? ['ok', 'header sm120_* existe'] : eq(c, 12, 1) ? (p ? ['ok', 'alias para sm120_*'] : ['err', 'Failed to open …/sm121_… (miss de cache)']) : ['na', 'fora deste exemplo']) },
        { rc: 'ds', n: 'Plugin gb10_hybrid_nvfp4', ex: () => 'cc == 121 or ALLOW_NON_GB10', ev: (c) => (eq(c, 12, 1) ? ['warn', 'gate passa (opt-in, não testado ao vivo)'] : ['err', 'falha rápido (ALLOW_NON_GB10=0)']) },
        { rc: 'ds', n: 'MegaMoE (DeepGEMM fp8_fp4)', ex: () => 'SM100 + EP', ev: (c) => (eq(c, 10, 0) ? ['ok', 'disponível (exige EP)'] : eq(c, 10, 3) ? ['na', 'não verificado'] : ['fb', 'não roda: MoE b12x']) },
        { rc: 'ds', n: 'Sparse MLA do DeepSeek-V4', ex: () => 'Hopper/SM10x → FlashMLA · SM120 → FlashInfer', ev: (c) => (c[0] === 9 || c[0] === 10 ? ['ok', 'FlashMLA sparse'] : c[0] === 12 ? ['ok', 'kernel SM120 do FlashInfer (lanes A e B)'] : ['na', 'fora deste exemplo']) },
      ];
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'GPU', options: GPUS.map((x) => ({ v: x.v, label: x.label })), value: 'gb10', onChange: (v) => { S.gpu = v; S.changed = true; } });
      UI.toggle(ctl, { label: 'aplicar patches das receitas', value: false, onChange: (v) => { S.patch = v; S.changed = true; } });
      fitH(fig, (w) => (w < 620 ? 60 + G.length * 58 + 56 : 70 + G.length * 36 + 30));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), gp = GPUS.find((x) => x.v === S.gpu), cc = gp.cc, narrow = w < 620, pad = 12;
        if (S.changed) { S.t0 = t; S.changed = false; }
        const scan = (t - S.t0) * 24;
        T(g, `${gp.label.split(' · ')[0]} · compute capability ${cc[0]}.${cc[1]} (sm_${cc[0]}${cc[1]})`, pad, 20, { size: 13.5, weight: 600, maxW: w - 2 * pad });
        T(g, S.patch ? 'com os patches das receitas' : 'código stock', pad, 38, { size: 11.5, color: S.patch ? C.a : C.muted });
        const col = { ok: C.a, fb: C.c, err: C.red || C.c, warn: C.d, na: C.muted };
        const rcCol = { glm: C.glm, qwen: C.qwen, ds: C.ds };
        const rcLab = { glm: 'GLM', qwen: 'Qwen', ds: 'DS' };
        const rowH = narrow ? 58 : 36, y0 = 52;
        let cnt = { ok: 0, fb: 0, err: 0, warn: 0, na: 0 };
        G.forEach((gt, i) => {
          const y = y0 + i * rowH;
          const [s, res] = gt.ev(cc, S.patch);
          const seen = scan > i;
          if (seen) cnt[s]++;
          g.fillStyle = V.alpha(C.muted, i % 2 ? 0.04 : 0.08); V.rr(g, pad - 4, y, w - 2 * pad + 8, rowH - 4, 6); g.fill();
          V.box(g, pad, y + 7, 38, 18, { r: 5, fill: V.alpha(rcCol[gt.rc], 0.2), stroke: rcCol[gt.rc], lw: 1, label: rcLab[gt.rc], size: 11 });
          const lr = 8, lx = narrow ? w - pad - 12 : w * 0.6, ly = y + 16;
          const c = seen ? col[s] : V.alpha(C.muted, 0.3);
          if (seen && s !== 'na') V.glow(g, lx, ly, 16, c, 0.35 + 0.25 * Math.sin(t * 3 + i));
          g.beginPath(); g.arc(lx, ly, lr, 0, Math.PI * 2); g.fillStyle = c; g.fill();
          if (!narrow) {
            T(g, gt.n, pad + 46, y + 14, { size: 12, weight: 600, maxW: w * 0.3 - 50 });
            T(g, gt.ex(S.patch), pad + 46, y + 28, { size: 11, font: 'mono', color: C.muted, maxW: lx - pad - 60 });
            T(g, seen ? res : '…', lx + 16, y + 21, { size: 12, color: seen ? (s === 'na' ? C.muted : C.ink) : C.muted, maxW: w - lx - 16 - pad });
          } else {
            T(g, gt.n, pad + 46, y + 20, { size: 12, weight: 600, maxW: w - pad - 46 - 30 });
            T(g, gt.ex(S.patch), pad + 4, y + 36, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad - 8 });
            T(g, seen ? '→ ' + res : '…', pad + 4, y + 50, { size: 11.5, color: seen ? (s === 'na' ? C.muted : C.ink) : C.muted, maxW: w - 2 * pad - 8 });
          }
        });
        const yl = y0 + G.length * rowH + 16;
        const leg = [['ok', 'nativo'], ['fb', 'fallback'], ['err', 'falha'], ['warn', 'risco'], ['na', 'n/a']];
        let lxx = pad, lyy = yl;
        leg.forEach(([k, lab]) => {
          const txt = `${lab} ${cnt[k]}`; g.font = V.font(12, 'sans', 600); const tw = g.measureText(txt).width + 30;
          if (lxx + tw > w - pad) { lxx = pad; lyy += 20; }
          g.beginPath(); g.arc(lxx + 6, lyy - 4, 6, 0, Math.PI * 2); g.fillStyle = col[k]; g.fill();
          T(g, txt, lxx + 17, lyy, { size: 12, weight: 600, color: C.ink }); lxx += tw;
        });
      });
    }

    /* =================== Figura 6: o carregador do fatbin =================== */
    {
      const fig = $('#fig-fatbin');
      const GPUS = [
        { v: 'a100', label: 'A100 · 8.0', cc: [8, 0] }, { v: 'h100', label: 'H100 · 9.0', cc: [9, 0] }, { v: 'b200', label: 'B200 · 10.0', cc: [10, 0] },
        { v: 'gb300', label: 'GB300 · 10.3', cc: [10, 3] }, { v: 'rtx', label: 'RTX PRO 6000 · 12.0', cc: [12, 0] }, { v: 'gb10', label: 'GB10 · 12.1', cc: [12, 1] },
      ];
      const PRE = {
        gen: { label: 'genérico', e: [{ k: 'sass', a: [8, 0], s: '' }, { k: 'ptx', a: [8, 0], s: '' }] },
        dc: { label: 'datacenter', e: [{ k: 'sass', a: [9, 0], s: 'a' }, { k: 'sass', a: [10, 0], s: 'a' }] },
        qwen: { label: 'sm_120 (Qwen)', e: [{ k: 'sass', a: [12, 0], s: '' }] },
        rec: { label: '12.1a (GLM/DS)', e: [{ k: 'sass', a: [12, 1], s: 'a' }] },
        fam: { label: '12.0f (família)', e: [{ k: 'sass', a: [12, 0], s: 'f' }] },
      };
      const S = { gpu: 'gb10', pre: 'rec', t0: 0, changed: true };
      const name = (e) => (e.k === 'sass' ? 'SASS sm_' : 'PTX compute_') + e.a[0] + e.a[1] + e.s;
      function check(e, cc) {
        const same = cc[0] === e.a[0], ge = cc[1] >= e.a[1], exact = same && cc[1] === e.a[1];
        if (e.k === 'sass') {
          if (e.s === 'a') return exact ? [true, 'exata: roda'] : [false, 'sufixo a: só na CC ' + e.a[0] + '.' + e.a[1]];
          if (same && ge) return [true, e.s === 'f' ? 'mesma família: roda' : 'mesmo major, minor ≥: roda'];
          return [false, same ? 'minor menor' : 'outro major: não roda'];
        }
        if (e.s === 'a') return exact ? [true, 'JIT (exata)'] : [false, 'PTX a: só na CC exata'];
        if (e.s === 'f') return same && ge ? [true, 'JIT na família'] : [false, 'fora da família'];
        return cc[0] * 10 + cc[1] >= e.a[0] * 10 + e.a[1] ? [true, 'JIT: CC ≥ ' + e.a[0] + '.' + e.a[1]] : [false, 'CC menor que o PTX'];
      }
      function resolve(entries, cc) {
        const res = entries.map((e) => check(e, cc));
        const si = entries.findIndex((e, i) => e.k === 'sass' && res[i][0]);
        if (si >= 0) return { res, pick: si, kind: 'sass' };
        const pi = entries.findIndex((e, i) => e.k === 'ptx' && res[i][0]);
        if (pi >= 0) return { res, pick: pi, kind: 'jit' };
        return { res, pick: -1, kind: 'err' };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Fatbin', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k].label })), value: 'rec', onChange: (v) => { S.pre = v; S.changed = true; } });
      UI.select(ctl, { label: 'GPU', options: GPUS.map((x) => ({ v: x.v, label: x.label })), value: 'gb10', onChange: (v) => { S.gpu = v; S.changed = true; } });
      fitH(fig, (w) => (w < 560 ? 360 : 290));
      const famName = (cc) => (cc[0] === 12 ? 'família 12.x · consumo' : cc[0] === 10 ? 'família 10.x · datacenter' : cc[0] === 9 ? 'Hopper' : 'Ampere');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560, pad = 12;
        if (S.changed) { S.t0 = t; S.changed = false; }
        const gp = GPUS.find((x) => x.v === S.gpu), E = PRE[S.pre].e, R = resolve(E, gp.cc);
        const per = 0.5, el = t - S.t0, cyc = E.length * per + 4.5, lt = el % cyc, step = Math.floor(lt / per), fin = lt >= E.length * per;
        const eh = narrow ? 28 : 32;
        const fb = narrow ? { x: pad, y: 10, w: w - 2 * pad, h: 40 + E.length * (eh + 6) } : { x: pad, y: 12, w: Math.min(360, w * 0.46), h: 40 + E.length * (eh + 6) };
        V.box(g, fb.x, fb.y, fb.w, fb.h, { r: 10, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.7) });
        T(g, 'fatbin (.so) de ' + PRE[S.pre].label, fb.x + 12, fb.y + 20, { size: 12.5, weight: 600, maxW: fb.w - 24 });
        E.forEach((e, i) => {
          const y = fb.y + 32 + i * (eh + 6), done = step > i || fin, active = step === i && !fin;
          const [ok, why] = R.res[i], chosen = done && R.pick === i;
          V.box(g, fb.x + 10, y, fb.w - 20, eh, { r: 6, fill: chosen ? V.alpha(C.a, 0.28) : V.alpha(e.k === 'sass' ? C.a : C.d, 0.1), stroke: active ? C.c : chosen ? C.a : V.alpha(C.muted, 0.4), lw: active || chosen ? 2 : 1 });
          T(g, name(e), fb.x + 20, y + eh / 2 + 4, { size: 12, font: 'mono', weight: 600 });
          if (done) T(g, (ok ? '✓ ' : '✗ ') + why, fb.x + fb.w - 18, y + eh / 2 + 4, { size: 11, align: 'right', color: ok ? C.a : C.c, maxW: fb.w * 0.5 });
          if (active) V.glow(g, fb.x + fb.w - 22, y + eh / 2, 14, C.c, 0.7);
        });
        const gbw = narrow ? 180 : Math.min(230, w * 0.3);
        const gb = narrow ? { x: w - pad - gbw, y: fb.y + fb.h + 12, w: gbw, h: 66 } : { x: w - pad - gbw, y: 12, w: gbw, h: Math.max(fb.h, 96) };
        V.box(g, gb.x, gb.y, gb.w, gb.h, { r: 12, fill: V.alpha(C.a, 0.08), stroke: C.a });
        T(g, gp.label.split(' · ')[0], gb.x + gb.w / 2, gb.y + 24, { size: 14, weight: 700, align: 'center', maxW: gb.w - 10 });
        T(g, `CC ${gp.cc[0]}.${gp.cc[1]} · sm_${gp.cc[0]}${gp.cc[1]}`, gb.x + gb.w / 2, gb.y + 42, { size: 12, font: 'mono', align: 'center', color: C.muted, maxW: gb.w - 10 });
        T(g, famName(gp.cc), gb.x + gb.w / 2, gb.y + 58, { size: 11, align: 'center', color: C.muted, maxW: gb.w - 10 });
        if (!narrow) {
          const rx = fb.x + fb.w + 14, ex = gb.x - 10, ry = fb.y + Math.min(fb.h, gb.h) / 2, u = (t * 0.8) % 1;
          V.arrow(g, rx, ry, ex, ry, { color: V.alpha(C.muted, 0.5), lw: 1.5 });
          V.dot(g, V.lerp(rx, ex, u), ry, 3, C.c);
          T(g, 'driver carrega o kernel', (rx + ex) / 2, ry - 10, { size: 11, align: 'center', color: C.muted, maxW: ex - rx });
        } else {
          wrap(g, 'o driver carrega o kernel nesta GPU →', pad, gb.y + 26, gb.x - pad - 12, 15, { size: 11.5, color: C.muted, maxLines: 3 });
        }
        const rbY = Math.max(fb.y + fb.h, gb.y + gb.h) + 14, rbH = narrow ? 104 : 74;
        let msg = '', c = C.muted;
        {
          const e = E[R.pick];
          if (R.kind === 'sass') { msg = `carrega ${name(e)}: código nativo, nada a compilar. ` + (e.s ? 'Recursos específicos (' + e.s + ') disponíveis.' : 'Sem sufixo: só as instruções comuns a partir da CC ' + e.a[0] + '.' + e.a[1] + '.'); c = C.a; }
          else if (R.kind === 'jit') { msg = `compila por JIT ${name(e)} → sm_${gp.cc[0]}${gp.cc[1]}: roda, mas o 1º uso é lento e só entram instruções da versão ${e.a[0]}.${e.a[1]} (nada de a/f).`; c = C.d; }
          else { msg = 'erro: “no kernel image is available for execution on the device”. Recompile para este alvo (TORCH_CUDA_ARCH_LIST) ou inclua PTX portável.'; c = C.c; }
        }
        V.box(g, pad, rbY, w - 2 * pad, rbH, { r: 10, fill: V.alpha(c, 0.1), stroke: V.alpha(c, 0.6) });
        wrap(g, msg, pad + 12, rbY + 22, w - 2 * pad - 24, 17, { size: 12.5, weight: 600, color: C.ink, alpha: fin ? 1 : 0.55, maxLines: narrow ? 5 : 3 });
        wrap(g, 'Regras: sm_XY roda em X.Z com Z ≥ Y · sm_XYa só em X.Y · sm_XYf na família X a partir de Y · PTX compute_XY compila para qualquer CC ≥ X.Y', pad, rbY + rbH + 18, w - 2 * pad, 15, { size: 11, color: C.muted, font: 'mono', maxLines: 4 });
      });
    }

    /* ======================== Figura 7: roofline ======================== */
    {
      const fig = $('#fig-roofline');
      const Ms = [1, 2, 4, 8, 16, 28, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
      const S = { mi: 0, bits: 16, fp4: false, beff: 230 };
      const P16 = 125, P4 = 500, B0 = 273, KN = 4096;
      const Ival = (M, b) => (2 * M * KN * KN) / ((KN * KN * b) / 8 + 2 * M * (KN + KN));
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'M (linhas)', min: 0, max: Ms.length - 1, step: 1, value: 0, fmt: (i) => String(Ms[i]), onInput: (v) => { S.mi = v; upd(); } });
      UI.seg(ctl, { label: 'Pesos', options: [{ v: 16, label: 'BF16' }, { v: 8, label: 'FP8' }, { v: 4.5, label: 'NVFP4' }, { v: 4, label: 'EXL3 4 bpw' }], value: 16, onChange: (v) => { S.bits = v; upd(); } });
      UI.toggle(ctl, { label: 'tensor cores FP4 (escala por bloco nativa)', value: false, onChange: (v) => { S.fp4 = v; upd(); } });
      UI.slider(ctl, { label: 'Banda', min: 150, max: 273, step: 1, value: 230, fmt: (v) => v + ' GB/s', onInput: (v) => { S.beff = v; upd(); } });
      const rI = UI.readout(ctl, 'intensidade', '—'), rP = UI.readout(ctl, 'teto', '—'), rR = UI.readout(ctl, 'regime', '—');
      const peak = () => (S.fp4 && S.bits === 4.5 ? P4 : P16);
      function upd() {
        const M = Ms[S.mi], I = Ival(M, S.bits), P = peak(), perf = Math.min(P, (I * S.beff) / 1000);
        rI.set(cm(I, I < 10 ? 2 : 0) + ' FLOP/B'); rP.set(cm(perf, perf < 10 ? 2 : 1) + ' TFLOPS (' + cm((100 * perf) / P, perf / P < 0.1 ? 2 : 0) + '%)');
        rR.set(I * S.beff / 1000 < P ? 'limitado por memória' : 'limitado por cálculo');
      }
      upd();
      fitH(fig, (w) => (w < 560 ? 420 : 440));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560;
        const L = narrow ? 44 : 60, Rr = 14, Tp = 16, Bt = 42;
        const x0 = Math.log10(0.5), x1 = Math.log10(4096), y0 = Math.log10(0.1), y1 = Math.log10(1000);
        const X = (I) => L + ((Math.log10(I) - x0) / (x1 - x0)) * (w - L - Rr);
        const Y = (p) => h - Bt - ((Math.log10(p) - y0) / (y1 - y0)) * (h - Tp - Bt);
        g.strokeStyle = C.grid; g.lineWidth = 1;
        [1, 4, 16, 64, 256, 1024, 4096].forEach((I) => { const x = X(I); g.beginPath(); g.moveTo(x, Tp); g.lineTo(x, h - Bt); g.stroke(); T(g, String(I), I === 4096 ? x + 4 : x, h - Bt + 14, { size: 11, align: I === 4096 ? 'right' : 'center', font: 'mono', color: C.muted }); });
        [0.1, 1, 10, 100, 1000].forEach((p) => { const y = Y(p); g.beginPath(); g.moveTo(L, y); g.lineTo(w - Rr, y); g.stroke(); T(g, String(p).replace('.', ','), L - 6, y + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }); });
        T(g, 'intensidade aritmética I (FLOP/B, log)', (L + w - Rr) / 2, h - 8, { size: 11.5, align: 'center', color: C.muted });
        g.save(); g.translate(12, (Tp + h - Bt) / 2); g.rotate(-Math.PI / 2); T(g, 'TFLOPS (log)', 0, 0, { size: 11.5, align: 'center', color: C.muted }); g.restore();
        const roof = (B, P, col, lw, dash) => {
          const ir = (P * 1000) / B;
          g.beginPath(); g.moveTo(X(0.5), Y((0.5 * B) / 1000)); g.lineTo(X(ir), Y(P)); g.lineTo(X(4096), Y(P));
          g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]);
          return ir;
        };
        if (S.fp4) roof(B0, P4, V.alpha(C.f, 0.8), 1.6, [6, 5]);
        roof(S.beff, P16, V.alpha(C.b, 0.7), 1.5, [4, 4]);
        const ridge = roof(B0, P16, C.a, 2.4);
        V.dot(g, X(ridge), Y(P16), 4, C.a);
        T(g, `ridge ≈ ${Math.round(ridge)} FLOP/B`, narrow ? X(ridge) - 8 : X(ridge) + 6, narrow ? Y(P16) - 8 : Y(P16) + 16, { size: 11.5, color: C.a, weight: 600, align: narrow ? 'right' : 'left' });
        T(g, '~125 TFLOPS BF16', w - Rr - 4, narrow ? Y(P16) + 16 : Y(P16) - 6, { size: 11, align: 'right', color: C.a });
        if (S.fp4) { if (narrow) { T(g, '- - teto ~4× FP4 nativo (estimativa DS)', L + 8, Y(P4) + 15, { size: 11, color: C.f, maxW: w - L - Rr - 16 }); } else T(g, '~4× (FP4×FP8 / W4A4 nativos, estimativa DS)', w - Rr - 4, Y(P4) - 6, { size: 11, align: 'right', color: C.f }); }
        const slx = X(2), sly = Y((2 * B0) / 1000);
        g.save(); g.translate(slx, sly); g.rotate(Math.atan2(Y((4 * B0) / 1000) - sly, X(4) - slx)); T(g, '273 GB/s × I', 4, -6, { size: 11, color: C.a }); g.restore();
        const u = (t * 0.12) % 1, ip = Math.pow(10, x0 + u * (x1 - x0)), pp = Math.min(P16, (ip * B0) / 1000);
        V.glow(g, X(ip), Y(pp), 9, C.a, 0.6);
        const mk = (I, p, lab, col, below) => { V.dot(g, X(I), Y(p), 4.5, col); wrap(g, lab, X(I) + 7, Y(p) + (below ? 16 : -10), Math.max(80, w - X(I) - 20), 13, { size: 11, color: col, maxLines: 2 }); };
        const i1 = Ival(1, 16);
        mk(i1, (i1 * B0) / 1000, 'GEMV de decode BF16, M=1', C.d, true);
        mk(112, (112 * 230) / 1000, 'MoE do prefill GLM: 112 FLOP/B', C.c, false);
        const y17 = Y(17);
        g.setLineDash([2, 4]); g.strokeStyle = V.alpha(C.glm, 0.8); g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(150), y17); g.lineTo(X(3000), y17); g.stroke(); g.setLineDash([]);
        T(g, narrow ? 'prefill GLM: ~17 TFLOPS/Spark' : 'prefill GLM medido: ~17 TFLOPS por Spark (~14%)', X(3000), y17 + 14, { size: 11, align: 'right', color: C.glm });
        const M = Ms[S.mi], I = Ival(M, S.bits), P = peak(), perf = Math.min(P, (I * S.beff) / 1000);
        const px = X(Math.max(0.5, Math.min(4096, I))), py = Y(Math.max(0.1, perf));
        V.glow(g, px, py, 18 + 4 * Math.sin(t * 4), C.c, 0.7);
        g.beginPath(); g.arc(px, py, 6, 0, Math.PI * 2); g.fillStyle = C.c; g.fill(); g.strokeStyle = C.panel || C.bg; g.lineWidth = 2; g.stroke();
        const lab = `M=${M}, ${S.bits === 16 ? 'BF16' : S.bits === 8 ? 'FP8' : S.bits === 4.5 ? 'NVFP4' : '4 bpw'}: I≈${cm(I, I < 10 ? 1 : 0)} → ${cm(perf, perf < 10 ? 2 : 0)} TFLOPS`;
        V.box(g, L + 8, Tp + 4, Math.min(w - L - Rr - 16, 360), 22, { r: 6, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.6), lw: 1 });
        T(g, lab, L + 16, Tp + 19, { size: 11.5, font: 'mono', color: C.ink, maxW: Math.min(w - L - Rr - 32, 344) });
        if (S.fp4 && S.bits !== 4.5) wrap(g, 'o teto FP4 só vale para formatos com MMA de escala por bloco nativa (NVFP4 W4A4 ou MXFP4×MXFP8); com estes pesos continua o de BF16', L + 8, Tp + 44, w - L - Rr - 16, 14, { size: 11, color: C.d, maxLines: 3 });
      });
    }

    /* ==================== Figura 8: o caminhão de bytes ==================== */
    {
      const fig = $('#fig-custo');
      const SCN = {
        qwen: { label: 'Qwen TP2', title: 'Qwen · TP2+EP · MTP=3 · lote 1 (por GPU)', unit: 'GiB', parts: [['lm_head ×4', 2.36], ['experts roteados', 2.1], ['GDN', 1.95], ['HyperConnections', 1.19], ['atenção + indexer', 0.6], ['MTP ×3', 0.5], ['router + shared + PLE', 0.4], ['estado GDN', 0.1]], tau: 3.2, fixed: 2.5, fixedLab: 'custo fixo do relatório: ≈106 coletivas × 20–30 µs ≈ 2–3 ms', fixedOn: false, pub: 'medido: 52,1 tok/s (MTP=3, lote 1)', lesson: 'O lm_head lido 4× é 26% dos bytes: alvo do vocabulário reduzido do rascunhador. O streaming sozinho prevê ~78 tok/s; a distância até o medido é custo fixo (~17–20 ms), bem acima dos 2–3 ms de coletivas do relatório.' },
        qwen8: { label: 'Qwen FP8-dense', title: 'Qwen · FP8-dense (analítico, por GPU)', unit: 'GiB', total: 5.7, tau: 3.2, fixed: 2.5, fixedLab: 'mesmo custo fixo do Qwen TP2 (≈2–3 ms de coletivas)', fixedOn: false, pub: 'estimativa da receita: ~85–90 tok/s (nunca medido)', lesson: 'Os densos em BF16 são ~6,9 dos 9,2 GiB; em FP8 o total cai para ~5,7 GiB. Os 85–90 da receita batem com 2,2 tokens por passo e sem custo fixo; com o bônus e os ~17–20 ms fixos do Qwen TP2, dariam ~70–75 (interpretação).' },
        ds: { label: 'DS TP2', title: 'DeepSeek · TP2 · DSpark k=6 · c=1 (por rank)', unit: 'GB', parts: [['experts roteados', 12], ['atenção', 2.4], ['lm_head', 1.06], ['rascunhador', 0.9], ['Markov w2', 0.2]], tau: 4.15, fixed: 12, fixedLab: 'custo fixo do relatório: ~10–15 ms (só as ≈108 coletivas: 3–5 ms)', fixedOn: false, pub: 'publicado: 62–83 tok/s (oficial 75,5)', lesson: 'Os experts são ~72% dos bytes: c=1 é ~80% streaming de pesos. Com τ≈4,15 (≈3,15 aceitos + o bônus), só o streaming dá ~56–61 tok/s; a faixa medida é maior porque em código a aceitação sobe (~5,2 tokens/passo a 73,3%) e a banda efetiva superou a suposta (interpretação do relatório).' },
        ds3: { label: 'DS TP3', title: 'DeepSeek · TP3 · k=6 (por rank)', unit: 'GB', total: 11.5, tau: 3.9, fixed: 13, fixedLab: 'custo fixo do relatório: ~13+ ms (anel de 3 ranks, cabeças 64→72)', fixedOn: true, pub: 'sonda ao vivo: 64,3 tok/s', lesson: 'Cada rank lê 1/3 menos, mas ~13 ms de custo fixo (anel de 3 ranks, cabeças 64→72) apagam o ganho. Desligue o custo fixo para ver o que se esperava.' },
      };
      const S = { sc: 'qwen', beff: 240, tau: 3.2, fixed: false };
      const ctl = UI.controls(fig);
      let slT = null, tgF = null;
      UI.seg(ctl, { label: 'Cenário', options: Object.keys(SCN).map((k) => ({ v: k, label: SCN[k].label })), value: 'qwen', onChange: (v) => { S.sc = v; if (slT) slT.set(SCN[v].tau); if (tgF) tgF.set(SCN[v].fixedOn); upd(); } });
      UI.slider(ctl, { label: 'Banda', min: 150, max: 273, step: 1, value: 240, fmt: (v) => v + ' GB/s', onInput: (v) => { S.beff = v; upd(); } });
      slT = UI.slider(ctl, { label: 'Tokens/passo', min: 1, max: 6, step: 0.05, value: 3.2, fmt: (v) => cm(v, 2), onInput: (v) => { S.tau = v; upd(); } });
      tgF = UI.toggle(ctl, { label: 'incluir custo fixo do relatório', value: false, onChange: (v) => { S.fixed = v; upd(); } });
      const rMs = UI.readout(ctl, 'ms/passo', '—'), rSt = UI.readout(ctl, 'passos/s', '—'), rTk = UI.readout(ctl, 'tok/s (teto)', '—');
      let M = null;
      function calc() {
        const sc = SCN[S.sc], tot = sc.parts ? sc.parts.reduce((s, p) => s + p[1], 0) : sc.total;
        const gb = sc.unit === 'GiB' ? tot * 1.073741824 : tot;
        const tS = (gb / S.beff) * 1000, tF = S.fixed ? sc.fixed : 0, tt = tS + tF;
        return { sc, tot, gb, tS, tF, tt, steps: 1000 / tt, tok: (S.tau * 1000) / tt };
      }
      function upd() { M = calc(); rMs.set(cm(M.tt, 1)); rSt.set(cm(M.steps, 1)); rTk.set(cm(M.tok, 1)); }
      upd();
      fitH(fig, (w) => (w < 560 ? 452 : 420));
      const pr = V.rng(4), pk = Array.from({ length: 34 }, () => ({ p: pr(), r: pr() }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), m = M, sc = m.sc, pad = 14, narrow = w < 560;
        const parts = sc.parts || [['total (sem divisão publicada)', sc.total]];
        const cols = parts.map((_, i) => C.series[i % 6]);
        T(g, sc.title, pad, 18, { size: 12.5, weight: 600, maxW: w - 2 * pad });
        const bY = 32, bH = narrow ? 56 : 62;
        const dW = narrow ? 64 : 92, sW = narrow ? 70 : 100;
        V.box(g, pad, bY, dW, bH, { r: 8, fill: V.alpha(C.d, 0.14), stroke: C.d, label: 'DRAM', size: 12, weight: 600 });
        V.box(g, w - pad - sW, bY, sW, bH, { r: 8, fill: V.alpha(C.a, 0.12), stroke: C.a, label: '48 SMs', size: 12, weight: 600 });
        const cx0 = pad + dW + 6, cx1 = w - pad - sW - 6, cyM = bY + bH / 2;
        V.line(g, cx0, cyM - 12, cx1, cyM - 12, { color: V.alpha(C.muted, 0.35), lw: 1 }); V.line(g, cx0, cyM + 12, cx1, cyM + 12, { color: V.alpha(C.muted, 0.35), lw: 1 });
        const speed = (S.beff / 240) * 0.28;
        const cum = []; let acc = 0; parts.forEach((p) => { acc += p[1]; cum.push(acc / m.tot); });
        pk.forEach((q) => {
          const u = (q.p + t * speed) % 1, x = V.lerp(cx0, cx1 - 14, u);
          let ci = cum.findIndex((c) => q.r <= c); if (ci < 0) ci = parts.length - 1;
          V.box(g, x, cyM - 8, 14, 16, { r: 3, fill: V.alpha(cols[ci], sc.parts ? 0.85 : 0.5) });
        });
        T(g, `${cm(S.beff, 0)} GB/s`, (cx0 + cx1) / 2, bY + bH + 14, { size: 11.5, align: 'center', font: 'mono', color: C.muted });
        const sy = bY + bH + 34, sh = 30, sx = pad, sw = w - 2 * pad;
        T(g, `bytes lidos por passo: ${cm(m.tot, sc.unit === 'GiB' ? 1 : 2)} ${sc.unit}` + (sc.unit === 'GiB' ? ` (${cm(m.gb, 2)} GB)` : ''), sx, sy - 6, { size: 12, weight: 600, maxW: sw });
        let xx = sx; const legend = [];
        parts.forEach((p, i) => {
          const ww = (p[1] / m.tot) * sw;
          g.fillStyle = V.alpha(cols[i], sc.parts ? 0.8 : 0.35); V.rr(g, xx + 0.5, sy, Math.max(1, ww - 1), sh, 4); g.fill();
          if (!sc.parts) stripes(g, xx, sy, ww, sh, V.alpha(C.muted, 0.35));
          const dec = (String(p[1]).split('.')[1] || '').length, lab = `${p[0]} ${cm(p[1], p[1] < 1 ? 2 : Math.max(1, dec))}`;
          g.font = V.font(11, 'sans', sc.parts ? 500 : 600);
          const lw0 = g.measureText(lab).width;
          if (!sc.parts && lw0 < ww - 16) {
            // sem divisão publicada: rótulo numa pílula com fundo do painel, legível sobre a hachura
            V.box(g, xx + ww / 2 - lw0 / 2 - 8, sy + sh / 2 - 10, lw0 + 16, 20, { r: 10, fill: C.panel || C.bg, stroke: V.alpha(cols[i], 0.7), lw: 1 });
            T(g, lab, xx + ww / 2, sy + sh / 2 + 4, { size: 11, align: 'center', color: C.ink, weight: 600 });
          } else if (sc.parts && lw0 < ww - 8) T(g, lab, xx + ww / 2, sy + sh / 2 + 4, { size: 11, align: 'center', color: onTxt(C), weight: 500 });
          else legend.push([cols[i], lab]);
          xx += ww;
        });
        let ly = sy + sh + 16, lx = sx;
        legend.forEach(([c, lab]) => {
          g.font = V.font(11, 'sans', 400); const lw2 = g.measureText(lab).width + 22;
          if (lx + lw2 > sx + sw) { lx = sx; ly += 16; }
          g.fillStyle = c; V.rr(g, lx, ly - 9, 10, 10, 2); g.fill(); T(g, lab, lx + 14, ly, { size: 11, color: C.muted }); lx += lw2;
        });
        const ty = ly + 30;
        T(g, 'tempo do passo', sx, ty - 8, { size: 12, weight: 600 });
        const tmax = Math.max(90, m.tt * 1.1), tw = sw;
        const wS = (m.tS / tmax) * tw, wF = (m.tF / tmax) * tw;
        V.box(g, sx, ty, tw, 24, { r: 5, fill: V.alpha(C.muted, 0.08) });
        V.box(g, sx, ty, Math.max(2, wS), 24, { r: 5, fill: V.alpha(C.b, 0.75) });
        if (wF > 0) { V.box(g, sx + wS, ty, Math.max(2, wF), 24, { r: 5, fill: V.alpha(C.d, 0.8) }); }
        const lp = ((t * 1000 * 0.06) % m.tt) / tmax * tw;
        V.line(g, sx + lp, ty - 3, sx + lp, ty + 27, { color: C.c, lw: 2 });
        g.font = V.font(11, 'sans', 500);
        const sLab = `streaming ${cm(m.tS, 1)} ms`;
        if (g.measureText(sLab).width < wS - 8) T(g, sLab, sx + 6, ty + 16, { size: 11, color: onTxt(C), weight: 500 });
        if (wF > 30) T(g, `fixo ${cm(m.tF, 1)}`, sx + wS + 4, ty + 16, { size: 11, color: onTxt(C), weight: 500, maxW: wF - 6 });
        T(g, S.fixed ? sc.fixedLab : 'custo fixo desligado: só o streaming dos pesos', sx, ty + 42, { size: 11.5, color: S.fixed ? C.d : C.muted, maxW: sw });
        const fy = ty + 66;
        T(g, `${cm(m.tt, 1)} ms/passo → ${cm(m.steps, 1)} passos/s × ${cm(S.tau, 2)} tokens = ${cm(m.tok, 1)} tok/s`, sx, fy, { size: narrow ? 12 : 13.5, weight: 700, color: C.ink, maxW: sw });
        T(g, sc.pub, sx, fy + 20, { size: 12, color: C.a, weight: 600, maxW: sw });
        wrap(g, sc.lesson, sx, fy + 42, sw, 16, { size: 12, color: C.muted, maxLines: 6 });
      });
    }

    /* ================= Figura 9: streams, eventos e esperas ================= */
    {
      const fig = $('#fig-streams');
      const S = { aux: false, dev: false, pin: false };
      function simulate(o) {
        const Lc = 0.45; let cpu = 0; const cpuB = [], ops = [];
        const free = { s0: 0, a1: 0, a2: 0, a3: 0, cp: 0 };
        const launch = (name, lane, dur, deps, col) => {
          const s0 = cpu; cpu += Lc; cpuB.push({ t0: s0, t1: cpu, kind: 'launch' });
          const dep = (deps || []).reduce((mx, d) => Math.max(mx, d.t1), 0);
          const t0 = Math.max(cpu, free[lane], dep);
          const op = { name, lane, t0, t1: t0 + dur, col, at: cpu };
          free[lane] = op.t1; ops.push(op); return op;
        };
        const waitFor = (op, label, extra) => { const s0 = cpu; cpu = Math.max(cpu, op.t1) + extra; cpuB.push({ t0: s0, t1: cpu, kind: 'wait', label }); };
        const q = launch('wqa_wkv', 's0', 3.0, null, 0);
        const a = launch('kv_score', o.aux ? 'a1' : 's0', 1.6, null, 1);
        const b = launch('idx.w', o.aux ? 'a2' : 's0', 1.2, null, 1);
        const c = launch('idx.comp', o.aux ? 'a3' : 's0', 1.6, null, 1);
        launch('atenção', 's0', 3.0, [q, a, b, c], 2);
        const r = launch('router', 's0', 0.8, null, 3);
        let syncs = 0;
        if (!o.dev) { waitFor(r, '.item()', 0.3); syncs++; }
        if (o.pin) {
          const cp = launch('counts→host', 'cp', 0.9, [r], 4);
          launch('experts finos', 's0', 3.2, null, 5);
          waitFor(cp, 'cópia', 0.15); syncs++;
          launch('experts gordos', 's0', 2.4, null, 5);
        } else {
          const s0 = cpu, t0 = Math.max(cpu, r.t1);
          const cp = { name: 'counts→host', lane: 'cp', t0, t1: t0 + 0.9, col: 4, at: s0 };
          ops.push(cp); free.cp = cp.t1; cpu = cp.t1 + 0.1; cpuB.push({ t0: s0, t1: cpu, kind: 'wait', label: 'cópia paginável' }); syncs++;
          launch('experts finos', 's0', 3.2, null, 5);
          launch('experts gordos', 's0', 2.4, null, 5);
        }
        launch('saída', 's0', 0.9, null, 0);
        const total = Math.max(cpu, ...ops.map((x) => x.t1));
        const s0ops = ops.filter((x) => x.lane === 's0').sort((x, y) => x.t0 - y.t0);
        const gaps = []; let busy = 0;
        s0ops.forEach((x, i) => { busy += x.t1 - x.t0; if (i && x.t0 - s0ops[i - 1].t1 > 0.05) gaps.push([s0ops[i - 1].t1, x.t0]); });
        if (s0ops.length && s0ops[0].t0 > 0.05) gaps.unshift([0, s0ops[0].t0]);
        const idle = gaps.reduce((s, gg) => s + gg[1] - gg[0], 0);
        return { ops, cpuB, total, gaps, idle, syncs, busy };
      }
      const worst = simulate({ aux: false, dev: false, pin: false }).total;
      let SIM = simulate(S);
      const ctl = UI.controls(fig);
      const rT = UI.readout(ctl, 'passo', '—'), rI = UI.readout(ctl, 'GPU ociosa', '—'), rS = UI.readout(ctl, 'syncs', '—');
      function upd() { SIM = simulate(S); rT.set(cm(SIM.total, 1) + ' u'); rI.set(Math.round((100 * SIM.idle) / SIM.total) + '%'); rS.set(String(SIM.syncs)); }
      UI.toggle(ctl, { label: '3 streams auxiliares (DS, overlay Stage-C)', value: false, onChange: (v) => { S.aux = v; upd(); } });
      UI.toggle(ctl, { label: 'decisão no device, sem .item() (DS #175)', value: false, onChange: (v) => { S.dev = v; upd(); } });
      UI.toggle(ctl, { label: 'cópia D2H pinned em stream lateral (GLM E1)', value: false, onChange: (v) => { S.pin = v; upd(); } });
      upd();
      // apelidos curtos (quando o nome não cabe na caixa) e o que cada stream auxiliar carrega
      const SHORT = { wqa_wkv: 'Q', kv_score: 'kv', 'idx.w': 'iw', 'idx.comp': 'ic', 'atenção': 'att', router: 'rt', 'experts finos': 'fin', 'experts gordos': 'gor', 'saída': 'out', 'counts→host': 'cp' };
      const DESC = { kv_score: 'compressor.kv_score', 'idx.w': 'indexer.weights_proj', 'idx.comp': 'indexer.compressor.kv_score' };
      const LEG = [
        [0, 'Q · out', 'fused_wqa_wkv (entrada da atenção) e saída da camada'],
        [1, 'kv · iw · ic', 'GEMMs pequenas e independentes do compressor e do indexer'],
        [2, 'att', 'atenção'],
        [3, 'rt', 'router do MoE'],
        [5, 'fin · gor', 'experts finos e experts gordos'],
        [4, 'cp', 'cópia das contagens por expert (DMA)'],
      ];
      const MCs = document.createElement('canvas').getContext('2d');
      function legendLayout(w) {
        const pad = 12, ncol = w < 560 ? 1 : 2, colW = (w - 2 * pad) / ncol, tw = colW - 22;
        const ent = LEG.map(([ci, ab, d]) => ({ ci, ls: splitLines(MCs, ab + ' = ' + d, tw, { size: 11.5 }) }));
        let hh = 0; const rows = [];
        for (let k = 0; k < ent.length; k += ncol) { const rr = ent.slice(k, k + ncol), n = Math.max(...rr.map((e) => e.ls.length)); rows.push({ rr, y: hh, n }); hh += n * 14 + 6; }
        return { pad, ncol, colW, rows, H: hh };
      }
      fitH(fig, (w) => 30 + 6 * (w < 560 ? 38 : 40) + 78 + legendLayout(w).H + 6);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), sim = SIM, narrow = w < 560;
        const lanes = [['cpu', 'CPU'], ['s0', 'stream 0'], ['a1', 'aux 1'], ['a2', 'aux 2'], ['a3', 'aux 3'], ['cp', narrow ? 'cópia' : 'cópia (DMA)']];
        const LL = legendLayout(w);
        const L = narrow ? 64 : 96, Rr = 12, top = 30, lh = V.clamp((h - top - 84 - LL.H) / lanes.length, 26, 46);
        const span = worst + 1;
        const X = (x) => L + (x / span) * (w - L - Rr);
        const ph = (t * 2.2) % (span + 2.5), tp = Math.min(ph, span);
        T(g, 'tempo →  (unidades ilustrativas)', L, 18, { size: 11.5, color: C.muted });
        lanes.forEach(([k, lab], i) => {
          const y = top + i * lh;
          g.fillStyle = V.alpha(C.muted, i % 2 ? 0.03 : 0.07); g.fillRect(L, y, w - L - Rr, lh - 4);
          T(g, lab, L - 8, y + lh / 2 + 2, { size: 11.5, align: 'right', color: C.ink, font: 'mono', maxW: L - 10 });
        });
        const ly = (k) => top + lanes.findIndex((x) => x[0] === k) * lh;
        if (!S.aux) T(g, 'vazias: ligue “3 streams auxiliares”', L + 10, ly('a2') + lh / 2 + 2, { size: 11, color: C.muted, maxW: w - L - Rr - 20 });
        const cols = [C.a, C.b, C.f, C.d, C.e, C.c];
        sim.cpuB.forEach((b) => {
          const y = ly('cpu') + 5, x0 = X(b.t0), x1 = X(b.t1);
          if (b.kind === 'launch') { V.box(g, x0, y, Math.max(1.5, x1 - x0 - 1), lh - 14, { r: 3, fill: V.alpha(C.d, b.t0 <= tp ? 0.85 : 0.3) }); }
          else { V.box(g, x0, y, Math.max(2, x1 - x0), lh - 14, { r: 3, fill: V.alpha(C.muted, b.t0 <= tp ? 0.4 : 0.15) }); if (x1 - x0 > 44) T(g, '⏳ ' + b.label, (x0 + x1) / 2, y + (lh - 14) / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: x1 - x0 - 4 }); }
        });
        sim.gaps.forEach(([a, b]) => { const y = ly('s0') + 4; stripes(g, X(a), y, X(b) - X(a), lh - 12, V.alpha(C.red || C.c, 0.55)); });
        sim.ops.forEach((op) => {
          const y = ly(op.lane) + 4, x0 = X(op.t0), x1 = X(op.t1), col = cols[op.col % 6], bw = x1 - x0;
          const prog = V.clamp((tp - op.t0) / (op.t1 - op.t0), 0, 1);
          V.box(g, x0, y, Math.max(2, bw - 1), lh - 12, { r: 4, fill: V.alpha(col, 0.18), stroke: V.alpha(col, 0.8), lw: 1 });
          if (prog > 0) V.box(g, x0, y, Math.max(1.5, (bw - 1) * prog), lh - 12, { r: 4, fill: V.alpha(col, 0.75) });
          const tc = prog > 0.5 ? onTxt(C) : C.ink, cy = y + (lh - 12) / 2 + 4;
          g.font = V.font(11, 'sans', 500);
          const full = g.measureText(op.name).width < bw - 6;
          g.font = V.font(11, 'sans', 600);
          const sh = SHORT[op.name] || '', shortOk = sh && g.measureText(sh).width < bw - 3;
          if (full) T(g, op.name, (x0 + x1) / 2, cy, { size: 11, align: 'center', color: tc, weight: 500 });
          else if (shortOk) T(g, sh, (x0 + x1) / 2, cy, { size: 11, align: 'center', color: tc, weight: 600 });
          // nas streams auxiliares, o nome completo da GEMM vai ao lado da caixa
          if (op.lane[0] === 'a' && DESC[op.name]) T(g, DESC[op.name], x1 + 6, cy, { size: 11, font: 'mono', color: C.muted, maxW: Math.max(20, w - Rr - x1 - 10) });
        });
        const xp = X(tp); V.line(g, xp, top - 4, xp, top + lanes.length * lh, { color: C.c, lw: 1.6 });
        const depth = sim.ops.filter((o) => o.at <= tp && o.t1 > tp).length;
        V.line(g, X(sim.total), top - 4, X(sim.total), top + lanes.length * lh, { color: V.alpha(C.a, 0.8), lw: 1.5, dash: [4, 3] });
        const by = top + lanes.length * lh + 18;
        T(g, `fila na GPU: ${depth} lançamento${depth === 1 ? '' : 's'} pendente${depth === 1 ? '' : 's'}`, L, by, { size: 12, weight: 600, color: depth ? C.a : C.c, maxW: w - L - Rr });
        wrap(g, `passo = ${cm(sim.total, 1)} u (pior caso ${cm(worst, 1)} u) · ${sim.syncs} espera${sim.syncs === 1 ? '' : 's'} da CPU · vermelho = GPU parada esperando a CPU`, L, by + 18, w - L - Rr, 15, { size: 11.5, color: C.muted, maxLines: 2 });
        // legenda por cor: o que cada caixa (e cada apelido) representa
        const ly0 = by + 58;
        g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; g.beginPath(); g.moveTo(LL.pad, ly0 - 14); g.lineTo(w - LL.pad, ly0 - 14); g.stroke();
        LL.rows.forEach((row) => {
          row.rr.forEach((e, c) => {
            const x = LL.pad + c * LL.colW, y = ly0 + row.y;
            g.fillStyle = V.alpha(cols[e.ci], 0.8); V.rr(g, x, y - 9, 12, 12, 3); g.fill();
            e.ls.forEach((ln, k) => T(g, ln, x + 18, y + k * 14, { size: 11.5, color: C.ink }));
          });
        });
      });
    }

    /* ============ Figura 9b: prefetch para a L2 numa stream lateral (GLM·TF) ============ */
    {
      const fig = $('#fig-l2pf');
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      const R = 4;        // MiB por unidade de tempo: banda ilustrativa
      const L2C = 0.6;    // fração do tempo de DRAM que um byte vindo da L2 ainda custa (ilustrativo)
      const HEAD = 3;     // começo de cada matmul desenhado em escala; o resto é comprimido
      const COMP = 0.12;  // compressão visual do resto das matmuls longas
      const OVC = 0.5;    // fração do tempo de DRAM do excedente que a stream principal perde na disputa (ilustrativo)
      const S = { on: true, shFirst: true, mb: 8 };
      // fases do trecho: mib = matmul limitado pela DRAM; idle = DRAM parada (duração ilustrativa, ponto de prefetch)
      const PH = {
        in: { lab: 'entrada', ab: 'ent.', mib: 32 },
        core: { lab: 'núcleo', ab: 'núc.', idle: 2.0, site: 'o' },
        o: { lab: 'saída', ab: 'saí.', mib: 12 },
        ag1: { lab: 'AG + cola', ab: 'AG', idle: 2.4, site: 'a' },
        sh: { lab: 'comp.', ab: 'c.', mib: 12 },
        rt: { lab: 'roteados', ab: 'rot.', mib: 104 },
        ag2: { lab: 'AG', ab: 'AG', idle: 2.0, site: 'f' },
        nx: { lab: 'próxima', ab: 'próx.', mib: 32 },
      };
      const TARGET = { o: 'o', a: 'sh', f: 'nx' }; // quem consome o que cada ponto traz
      function simulate(st) {
        const order = ['in', 'core', 'o', 'ag1'].concat(st.shFirst ? ['sh', 'rt'] : ['rt', 'sh'], ['ag2', 'nx']);
        const pf = { o: 0, a: 0, f: 0 }, ov = { o: 0, a: 0, f: 0 };
        if (st.on) order.forEach((k) => { const p = PH[k]; if (p.site) { const cap = p.idle * R; pf[p.site] = Math.min(st.mb, cap); ov[p.site] = Math.max(0, st.mb - cap); } });
        // a L2 guarda até o consumo; se os roteados rodam antes do compartilhado, o fluxo deles expulsa o que o ponto a trouxe
        const evicted = st.on && !st.shFirst && pf.a > 0;
        const used = { o: Math.min(pf.o, PH.o.mib), a: evicted ? 0 : Math.min(pf.a, PH.sh.mib), f: Math.min(pf.f, PH.nx.mib) };
        const fromL2 = { o: used.o, sh: used.a, nx: used.f };
        let tt = 0, vx = 0, carry = 0; const ops = [];
        order.forEach((k) => {
          const p = PH[k];
          if (p.idle) { ops.push({ k, p, v0: vx, v1: vx + p.idle }); tt += p.idle; vx += p.idle; carry = st.on ? (ov[p.site] / R) * OVC : 0; return; }
          const d0 = p.mib / R, l2 = fromL2[k] || 0, save = (l2 / R) * (1 - L2C), pen = carry; carry = 0;
          const vis0 = d0 <= HEAD ? d0 : HEAD + (d0 - HEAD) * COMP;
          ops.push({ k, p, v0: vx, v1: vx + vis0 - save + pen, l2, l2w: (l2 / R) * L2C, pen, comp: d0 > HEAD });
          tt += d0 - save + pen; vx += vis0 - save + pen;
        });
        const side = [], l2 = [];
        ops.forEach((op, i) => {
          if (!st.on || !op.p.site) return;
          const s = op.p.site, nxt = ops[i + 1];
          const ovEnd = nxt ? Math.min(op.v1 + ov[s] / R, nxt.v0 + Math.min(HEAD, nxt.v1 - nxt.v0)) : op.v1;
          side.push({ s, v0: op.v0, v1: op.v0 + pf[s] / R, ov: ov[s] > 0 ? { v0: op.v1, v1: ovEnd } : null });
          const tgt = ops.find((o) => o.k === TARGET[s]), ev = s === 'a' && evicted ? ops.find((o) => o.k === 'rt') : null;
          l2.push({ s, v0: op.v0, v1: ev ? ev.v0 : tgt.v0, mib: pf[s], to: PH[TARGET[s]].lab, ev: !!ev });
        });
        const idleT = ops.filter((o) => o.p.idle).reduce((a, o) => a + o.p.idle - (st.on ? pf[o.p.site] / R : 0), 0);
        return { ops, side, l2, total: tt, vtotal: vx, idleT, fromL2: used.o + used.a + used.f, ovT: st.on ? ov.o + ov.a + ov.f : 0, evicted };
      }
      const BASE = simulate({ on: false, shFirst: true, mb: 8 });
      const MIBT = Object.values(PH).reduce((a, p) => a + (p.mib || 0), 0);
      let SIM = simulate(S);
      const ctl = UI.controls(fig);
      const rT = UI.readout(ctl, 'trecho', '—'), rI = UI.readout(ctl, 'DRAM parada', '—'), rL = UI.readout(ctl, 'lido da L2', '—');
      function upd() { SIM = simulate(S); rT.set(cm(SIM.total, 1) + ' u'); rI.set(cm((100 * SIM.idleT) / SIM.total, 1) + '%', SIM.idleT > 0.05 ? 'warn' : 'ok'); rL.set(cm(SIM.fromL2, 1) + ' MiB'); }
      UI.toggle(ctl, { label: 'prefetch para a L2 (TF_GLM_L2PF)', value: true, onChange: (v) => { S.on = v; upd(); } });
      UI.toggle(ctl, { label: 'expert compartilhado antes dos roteados', value: true, onChange: (v) => { S.shFirst = v; upd(); } });
      UI.slider(ctl, { label: 'orçamento por ponto', min: 2, max: 16, step: 2, value: 8, fmt: (v) => v + ' MiB', onInput: (v) => { S.mb = v; upd(); } });
      upd();
      const LEG = [
        ['b', 'matmul limitado pela DRAM: entrada, saída, expert compartilhado (comp.), experts roteados, entrada da próxima camada'],
        ['muted', 'fase em que a DRAM para (na faixa DRAM, hachurado vermelho): núcleo (recorrência KDA ou atenção), AG + cola (all-gather, hiperconexões, roteador, top-k), AG da FFN'],
        ['tf', 'prefetch na stream lateral (pontos o, a, f) e o que ele deixa guardado na L2'],
        ['a', 'bytes que o matmul encontra na L2'],
        ['c', 'excedente: passa da fase ociosa, disputa a DRAM e atrasa a stream principal'],
      ];
      const MCl = document.createElement('canvas').getContext('2d');
      function legendLayout(w) {
        const pad = 12, ncol = w < 560 ? 1 : 2, colW = (w - 2 * pad) / ncol, tw = colW - 22;
        const ent = LEG.map(([ci, d]) => ({ ci, ls: splitLines(MCl, d, tw, { size: 11.5 }) }));
        let hh = 0; const rows = [];
        for (let k = 0; k < ent.length; k += ncol) { const rr = ent.slice(k, k + ncol), n = Math.max(...rr.map((e) => e.ls.length)); rows.push({ rr, y: hh }); hh += n * 14 + 6; }
        return { pad, colW, rows, H: hh };
      }
      const SA = (w) => (w < 560 ? 120 : 76); // área do resumo, abaixo das faixas
      fitH(fig, (w) => 28 + 4 * (w < 560 ? 40 : 46) + 16 + SA(w) + legendLayout(w).H + 8);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), TF = tfc(C), sim = SIM, narrow = w < 560;
        const colOf = { b: C.b, muted: C.muted, tf: TF, a: C.a, c: C.c };
        const lanes = [['main', narrow ? 'principal' : 'stream principal'], ['side', narrow ? 'lateral' : 'stream lateral'], ['l2', 'L2'], ['dram', 'DRAM']];
        const LL = legendLayout(w);
        const L = narrow ? 66 : 112, Rr = 12, top = 28;
        const lh = V.clamp((h - top - 16 - SA(w) - LL.H - 8) / lanes.length, 30, 52), bh = lh - 12;
        const span = BASE.vtotal + 0.2;
        const X = (v) => L + (v / span) * (w - L - Rr);
        const tp = ((t * 1.5) % (span + 3)), tc = Math.min(tp, sim.vtotal);
        T(g, narrow ? 'tempo → (fora de escala)' : 'tempo →  (fora de escala: as matmuls longas estão comprimidas, ≈)', L, 17, { size: 11.5, color: C.muted, maxW: w - L - Rr });
        lanes.forEach(([k, lab], i) => {
          const y = top + i * lh;
          g.fillStyle = V.alpha(C.muted, i % 2 ? 0.03 : 0.07); g.fillRect(L, y, w - L - Rr, lh - 4);
          T(g, lab, L - 8, y + lh / 2 + 2, { size: 11.5, align: 'right', color: C.ink, font: k === 'l2' || k === 'dram' ? 'mono' : 'sans', maxW: L - 10 });
        });
        const ly = (k) => top + lanes.findIndex((x) => x[0] === k) * lh + 4;
        // stream principal
        sim.ops.forEach((op) => {
          const y = ly('main'), x0 = X(op.v0), x1 = X(op.v1), bw = x1 - x0;
          if (op.p.idle) V.box(g, x0, y, Math.max(2, bw - 1), bh, { r: 4, fill: V.alpha(C.muted, 0.18), stroke: V.alpha(C.muted, 0.6), lw: 1 });
          else {
            V.box(g, x0, y, Math.max(2, bw - 1), bh, { r: 4, fill: V.alpha(C.b, 0.5), stroke: V.alpha(C.b, 0.95), lw: 1 });
            if (op.l2 > 0) V.box(g, x0, y, Math.max(2, X(op.v0 + op.l2w) - x0), bh, { r: 4, fill: V.alpha(C.a, 0.9) });
            if (op.pen > 0.01) { const p0 = X(op.v0 + (op.l2 > 0 ? op.l2w : 0)), p1 = X(op.v0 + (op.l2 > 0 ? op.l2w : 0) + op.pen); V.box(g, p0, y, Math.max(2, p1 - p0), bh, { r: 3, fill: V.alpha(C.c, 0.14), stroke: V.alpha(C.c, 0.85), lw: 1 }); stripes(g, p0, y, Math.max(2, p1 - p0), bh, V.alpha(C.c, 0.6)); }
            if (op.comp) {
              const xm = x0 + bw * 0.82; g.strokeStyle = V.alpha(C.ink, 0.75); g.lineWidth = 1.4;
              [-3, 3].forEach((dx) => { g.beginPath(); g.moveTo(xm + dx - 3, y + bh - 4); g.lineTo(xm + dx + 3, y + 4); g.stroke(); });
            }
          }
          const cx = op.comp ? x0 + bw * 0.4 : (x0 + x1) / 2, room = op.comp ? bw * 0.7 : bw - 6;
          g.font = V.font(11, 'sans', 600);
          const lab = g.measureText(op.p.lab).width < room ? op.p.lab : g.measureText(op.p.ab).width < room + 2 ? op.p.ab : '';
          if (lab) T(g, lab, cx, y + bh / 2 + 4, { size: 11, align: 'center', weight: 600, color: C.ink });
        });
        // stream lateral: os três pontos (contorno tracejado) e o prefetch
        const ys = ly('side');
        sim.ops.forEach((op) => {
          if (!op.p.site) return;
          const x0 = X(op.v0), x1 = X(op.v1);
          g.setLineDash([3, 3]); V.rr(g, x0, ys, Math.max(2, x1 - x0 - 1), bh, 4); g.strokeStyle = V.alpha(TF, 0.7); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
        });
        sim.side.forEach((sd) => {
          const x0 = X(sd.v0), x1 = X(sd.v1);
          V.box(g, x0, ys, Math.max(2, x1 - x0 - 1), bh, { r: 4, fill: V.alpha(TF, 0.7), stroke: TF, lw: 1 });
          if (sd.ov) { const o0 = X(sd.ov.v0), o1 = X(sd.ov.v1); if (o1 - o0 > 1) { V.box(g, o0, ys, o1 - o0, bh, { r: 4, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.85), lw: 1 }); stripes(g, o0, ys, o1 - o0, bh, V.alpha(C.c, 0.6)); } }
        });
        sim.ops.forEach((op) => {
          if (!op.p.site) return;
          const cx = X(op.v0) + 9, cy = ys + bh / 2;
          g.beginPath(); g.arc(cx, cy, 8, 0, Math.PI * 2); g.fillStyle = C.panel || C.bg; g.fill(); g.strokeStyle = TF; g.lineWidth = 1.5; g.stroke();
          T(g, op.p.site, cx, cy + 4, { size: 11.5, weight: 700, font: 'mono', align: 'center', color: C.ink });
        });
        // L2: o que está guardado e para quem
        const yl = ly('l2');
        sim.l2.forEach((c, i) => {
          const x0 = X(c.v0), x1 = X(c.v1), bw = x1 - x0, col = c.ev ? (C.red || C.c) : TF;
          V.box(g, x0, yl, Math.max(2, bw - 1), bh, { r: 4, fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.85), lw: 1 });
          if (c.ev) { const xe = x1 - 1, yc = yl + bh / 2; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(xe - 5, yc - 5); g.lineTo(xe + 5, yc + 5); g.moveTo(xe + 5, yc - 5); g.lineTo(xe - 5, yc + 5); g.stroke(); }
          // rótulo dentro da caixa, se couber; senão, ao lado, até a próxima caixa
          const full = c.ev ? `${cm(c.mib, 1)} MiB expulsos` : `${cm(c.mib, 1)} MiB → ${c.to}`, short = c.ev ? 'expulsos' : `${cm(c.mib, 1)} MiB`;
          g.font = V.font(11, 'sans', 500);
          const nx = sim.l2[i + 1] ? X(sim.l2[i + 1].v0) : w - Rr, room = nx - x1 - (c.ev ? 16 : 8), ox = x1 + (c.ev ? 10 : 5);
          const fw = g.measureText(full).width, sw = g.measureText(short).width;
          if (!c.ev && fw < bw - 10) T(g, full, x0 + 5, yl + bh / 2 + 4, { size: 11, color: C.ink });
          else if (fw < room) T(g, full, ox, yl + bh / 2 + 4, { size: 11, color: c.ev ? col : C.ink });
          else if (sw < room) T(g, short, ox, yl + bh / 2 + 4, { size: 11, color: c.ev ? col : C.ink });
        });
        if (!S.on) T(g, 'vazia: ligue o prefetch', L + 10, yl + bh / 2 + 4, { size: 11, color: C.muted, maxW: w - L - Rr - 20 });
        // DRAM: ocupada pelas matmuls e pelo prefetch; hachurado vermelho = parada
        const yd = ly('dram');
        sim.ops.forEach((op) => {
          const x0 = X(op.v0), x1 = X(op.v1);
          if (!op.p.idle) {
            V.box(g, x0, yd, Math.max(2, x1 - x0 - 1), bh, { r: 3, fill: V.alpha(C.b, 0.35) });
            for (let xx = x0 + ((t * 24) % 12) + 3; xx < x1 - 3; xx += 12) V.dot(g, xx, yd + bh / 2, 1.6, V.alpha(C.ink, 0.55));
          } else {
            const sd = sim.side.find((q) => q.s === op.p.site), xb = sd ? X(sd.v1) : x0;
            if (xb > x0 + 1) { V.box(g, x0, yd, xb - x0 - 1, bh, { r: 3, fill: V.alpha(TF, 0.45) }); for (let xx = x0 + ((t * 24) % 12) + 3; xx < xb - 3; xx += 12) V.dot(g, xx, yd + bh / 2, 1.6, V.alpha(C.ink, 0.55)); }
            if (x1 - xb > 1.5) stripes(g, xb, yd, x1 - xb, bh, V.alpha(C.red || C.c, 0.55));
          }
        });
        sim.side.forEach((sd) => { if (sd.ov) { const o0 = X(sd.ov.v0), o1 = X(sd.ov.v1); if (o1 - o0 > 1) stripes(g, o0, yd, o1 - o0, bh, V.alpha(C.c, 0.7)); } });
        // cursor do tempo
        const xp = X(tc); V.line(g, xp, top - 4, xp, top + lanes.length * lh, { color: C.c, lw: 1.6 });
        V.line(g, X(sim.vtotal), top - 4, X(sim.vtotal), top + lanes.length * lh, { color: V.alpha(C.a, 0.85), lw: 1.5, dash: [4, 3] });
        // resumo
        const by = top + lanes.length * lh + 16, gain = (100 * (BASE.total - sim.total)) / BASE.total;
        const n1 = wrap(g, `trecho: ${cm(sim.total, 1)} u, contra ${cm(BASE.total, 1)} u sem prefetch (${gain > 0.05 ? '−' + cm(gain, 1) + '%' : gain < -0.05 ? '+' + cm(-gain, 1) + '%, mais lento' : 'sem ganho'}) · DRAM parada ${cm((100 * sim.idleT) / sim.total, 1)}%`, L, by, w - L - Rr, 15, { size: 12, weight: 600, color: C.ink, maxLines: 2 });
        let note;
        if (!S.on) note = 'Sem prefetch, nas fases cinza a DRAM não lê nada (hachurado vermelho).';
        else {
          note = `Da L2 saem ${cm(sim.fromL2, 1)} dos ${MIBT} MiB que o trecho lê (${cm((100 * sim.fromL2) / MIBT, 0)}%).`;
          if (sim.evicted) note += ' O que o ponto a trouxe para o expert compartilhado foi expulso pelo fluxo dos roteados.';
          if (sim.ovT > 0) note += ` Excedente: ${cm(sim.ovT, 1)} MiB, que atrasam a stream principal.`;
        }
        wrap(g, note, L, by + n1 * 15 + 4, w - L - Rr, 14, { size: 11.5, color: C.muted, maxLines: narrow ? 5 : 2 });
        // legenda
        const ly0 = by + SA(w);
        g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; g.beginPath(); g.moveTo(LL.pad, ly0 - 14); g.lineTo(w - LL.pad, ly0 - 14); g.stroke();
        LL.rows.forEach((row) => {
          row.rr.forEach((e, c) => {
            const x = LL.pad + c * LL.colW, y = ly0 + row.y;
            g.fillStyle = V.alpha(colOf[e.ci], 0.85); V.rr(g, x, y - 9, 12, 12, 3); g.fill();
            e.ls.forEach((ln, k) => T(g, ln, x + 18, y + k * 14, { size: 11.5, color: C.ink }));
          });
        });
      });
    }

    /* ================ Figura 10: a linha de metrô de cada receita ================ */
    {
      const fig = $('#fig-metro');
      const LINES = {
        glm: [
          ['Alvo', 'TORCH_CUDA_ARCH_LIST=12.1a e FLASHINFER_CUDA_ARCH_LIST=12.1a no Dockerfile e nos launchers: o build e o JIT em runtime miram sm_121a.', 'Dockerfile:381-382 e 433; start.sh repassa ao container'],
          ['Compilação', 'nvcc/ptxas compilam o exllamav3_ext (pin c5d9c657) com os kernels E2, E3 e FAST; alvos AVX de CPU viram stubs em aarch64; o módulo E3 sai sem --use_fast_math para manter o arredondamento do torch.sigmoid.', 'patch_exl3_ext_aarch64.py · patch_exl3_fat_kernel.py · patch_exl3_decode_pipeline.py · build_exl3_fat_moe_ext.py'],
          ['Binário', 'Cubins sm_121a. O Dockerfile confere os símbolos (exl3_moe, exl3_fat_*, glm53_fast_moe_version()==1) e SKIP_OVERLAY_VERIFY=0 roda o autoteste na GPU antes de servir.', 'Dockerfile:421-436 · SKIP_OVERLAY_VERIFY / EXL3_SELFCHECK_GPU'],
          ['Gates', 'get_min_capability 80 (exl3 aceito); E3 exige ≥ 9.0 (red.v4); FAST só no SM121; PDL restrito a major 9/10; páginas de 64 no SM12x.', 'overlay/exl3.py:249-250 · platforms/cuda.py (PDL, páginas de 64)'],
          ['Kernel', 'exl3_moe (thin, decode) e exl3_fat_moe (E3, prefill): CTAs de 8 warps, cp.async multi-estágio, swizzle XOR, Hadamard por shuffles entre lanes.', 'overlay/exl3_fat_gemm.cu · overlay/exl3_fat_moe.cu'],
          ['Tensor core', 'A treliça é desquantizada em registradores e alimenta mma.sync m16n8k16 FP16→FP32. O FP4 nativo não se aplica: a treliça não é formato de hardware.', 'ptx_mma_m16n8k16 · dq_dispatch<4,1>'],
          ['Gargalo', 'Decode: banda. Prefill: o MoE com ~28 linhas por expert a MNBT 1024 dá 112 FLOP/B, abaixo do ridge de 458; 947 tok/s ≈ 14% do pico BF16.', 'docs/improve-prefill.md §2'],
        ],
        qwen: [
          ['Alvo', 'A imagem day-0 já vem compilada: cubins sm_120 sem sufixo (ex.: cutlass_scaled_mm_sm120_fp8), mais JIT do Triton para os kernels QSA.', 'imagem vllm/vllm-openai:qwen38-flash-next (day-0)'],
          ['Compatibilidade', 'SASS sm_120 roda no 12.1 (mesmo major, minor maior): não precisa recompilar para o GB10.', 'docs/CLAUDE/fable5-1-report.md §8.1'],
          ['Gates', 'MoE: TRTLLM e CuteDSL só SM10x → CUTLASS SM120. Prefill GDN no FlashInfer só SM90/10x → Triton FLA. Skinny GEMM só SM103 → cuBLAS.', 'fused_moe/oracle/nvfp4.py · qwen_gdn_linear_attn.py:112-160 · low_latency_gemm.py'],
          ['Gate sutil', '_is_sm120() compara == (12, 0): no GB10, o backport v0.30 fica com a tabela padrão (GB300), não com a do sm_120 (inferido do código).', 'files/patch_qsa_fp8_kv_v030.py:129-165'],
          ['Kernel', 'Grouped GEMM CUTLASS SM120 (experts), Triton QSA (atenção esparsa; perfis para 48 SMs via QSA_PROFILE) e cuBLAS BF16 (densos).', 'files/qsa_gb10/apply_patch.py · QSA_PROFILE'],
          ['Tensor core', 'Experts W4A4: MMA FP4 com escala por bloco (kind::mxf4nvf4). Densos: MMA BF16.', 'FlashInfer CUTLASS SM120 grouped GEMM (gen_cutlass_fused_moe_sm120_module)'],
          ['Gargalo', '≈9,2 GiB por GPU por passo a 235–245 GB/s = 40–42 ms; × τ≈3,2 (2,2 aceitos + bônus) = teto ~76–80 tok/s; medido 52,1 (passo ~60 ms, ~1/3 custo fixo).', 'docs/CLAUDE/fable5-1-report.md §2 · README (MTP=3)'],
        ],
        ds: [
          ['Alvo', 'CUTE_DSL_ARCH=sm_121a, TORCH_CUDA_ARCH_LIST=12.1a e FLASHINFER_CUDA_ARCH_LIST=12.1a no .env e no compose.', '.env.dspark.example:601-604 · docker-compose.dspark.yml:198-204'],
          ['JIT', 'b12x (CuTeDSL), FlashInfer e DeepGEMM compilam na hora, com caches persistidos em volumes (VLLM_CACHE_ROOT). Sem CUTE_DSL_ARCH, o b12x pode compilar um W4A16 mais lento no meio do serviço.', 'VLLM_CACHE_ROOT · B12X_CUTE_COMPILE_CACHE_DIR · DG_JIT_*'],
          ['Binário', 'O DeepGEMM gera nomes sm121_*, mas só há headers sm120_*: o alias com #define resolve (DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1).', 'patches/hotfix-deepgemm-sm121-mqa-header-alias.sh'],
          ['Gates', 'Cache MXFP4 do indexer: family(100) → + family(120) (hotfix da lane A). MegaMoE só SM100. FlashMLA só Hopper/SM10x → kernel SM120 do FlashInfer (na lane A dentro da imagem; no overlay, lane B, o backend DSV4_SPARSE_MLA_SM120). Plugin híbrido NVFP4 só cc 12.1, opt-in e fora do checkpoint MXFP4.', 'hotfix-vllm-mxfp4-indexer-cache.py · vllm_patch_gb10/…/kernel.py:38 · recipe/overlay/…/nvidia/sm120.py (lane B)'],
          ['Kernel', 'Produção (lane A): MoE b12x W4A16 e sparse MLA SM120 do FlashInfer. O overlay Stage-C (lane B, histórico) mostra ainda 3 streams auxiliares para as GEMMs de entrada do compressor e do indexer; o código equivalente da lane A não está no repositório.', 'fused_moe/b12x_moe.py · recipe/overlay/…/nvidia/model.py:1700-1709 (lane B)'],
          ['Tensor core', 'W4A16: o FP4 (MXFP4) é desquantizado para BF16 → MMA BF16. O caminho FP4 nativo do b12x só aceita fonte modelopt_nvfp4.', 'fused_moe/b12x_moe.py:427-497 (fp4_e8m0_k32 → W4A16)'],
          ['Gargalo', '16,5–17 GB por rank por passo → 68–74 ms; c=1 é ~80% streaming de pesos. O resto, ~10–15 ms, é custo fixo; só as ≈108 coletivas já somam 3–5 ms.', 'docs/CLAUDE/fable5-1-report.md §2'],
        ],
      };
      const S = { rc: 'glm', i: 0, hold: -1, last: 0, pos: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.rc = v; S.i = 0; S.pos = 0; } });
      UI.button(ctl, '← estação', () => { S.i = (S.i + 6) % 7; S.hold = Infinity; });
      UI.button(ctl, 'estação →', () => { S.i = (S.i + 1) % 7; S.hold = Infinity; });
      // altura pelo conteúdo: a estação de texto mais longo (de qualquer receita) precisa caber na caixa
      const MCm = document.createElement('canvas').getContext('2d');
      function metroNeed(w) {
        const narrow = w < 560, bw = w - 28, by = narrow ? 24 + 7 * 40 : 110;
        let mx = 0;
        Object.values(LINES).forEach((st) => st.forEach(([, tx, ref]) => {
          const nl = Math.min(narrow ? 12 : 9, splitLines(MCm, tx, bw - 32, { size: narrow ? 13 : 14.5 }).length);
          const rl = ref ? Math.min(4, splitLines(MCm, 'onde: ' + ref, bw - 32, { size: 11.5, font: 'mono' }).length) : 0;
          mx = Math.max(mx, 52 + nl * (narrow ? 19 : 21) + (ref ? 12 + rl * 15 : 0) + 28);
        }));
        return by + mx + 10;
      }
      fitH(fig, (w) => Math.max(w < 560 ? 480 : 345, metroNeed(w)));
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 560, pad = 14, st = LINES[S.rc], col = C[S.rc];
        if (S.hold !== Infinity && t > S.hold) { if (t - S.last > 3.4) { S.i = (S.i + 1) % 7; S.last = t; } }
        S.pos += (S.i - S.pos) * 0.08;
        if (Math.abs(S.i - S.pos) > 3.5) S.pos = S.i;
        const P = [];
        if (!narrow) { const x0 = pad + 30, x1 = w - pad - 30, y = 58; for (let k = 0; k < 7; k++) P.push([V.lerp(x0, x1, k / 6), y]); }
        else { const x = pad + 16, y0 = 24; for (let k = 0; k < 7; k++) P.push([x, y0 + k * 40]); }
        g.strokeStyle = V.alpha(col, 0.75); g.lineWidth = 6; g.lineCap = 'round';
        g.beginPath(); g.moveTo(P[0][0], P[0][1]); g.lineTo(P[6][0], P[6][1]); g.stroke(); g.lineCap = 'butt';
        const ptr = H.pointer;
        st.forEach(([name], k) => {
          const [x, y] = P[k], on = k === S.i;
          if (ptr.clicked && Math.hypot(ptr.x - x, ptr.y - y) < 22) { S.i = k; S.hold = t + 12; S.last = t; }
          g.beginPath(); g.arc(x, y, on ? 11 : 8, 0, Math.PI * 2); g.fillStyle = on ? col : C.panel || C.bg; g.fill(); g.strokeStyle = col; g.lineWidth = 3; g.stroke();
          if (!narrow) wrap(g, name, x, y + 30, (w - 2 * pad) / 7 - 6, 14, { size: 11.5, align: 'center', weight: on ? 700 : 500, color: on ? C.ink : C.muted, maxLines: 2 });
          else T(g, `${k + 1}. ${name}`, x + 20, y + 4, { size: 12, weight: on ? 700 : 500, color: on ? C.ink : C.muted });
        });
        const pf = Math.floor(S.pos), pfr = S.pos - pf, a = P[Math.min(6, pf)], b = P[Math.min(6, pf + 1)];
        const tx = V.lerp(a[0], b[0], pfr), ty = V.lerp(a[1], b[1], pfr);
        V.glow(g, tx, ty, 20, col, 0.8);
        V.box(g, tx - 13, ty - 7, 26, 14, { r: 7, fill: C.ink });
        V.dot(g, tx + 6, ty, 2.5, col);
        const bx = pad, by = narrow ? 24 + 7 * 40 : 110, bw = w - 2 * pad, bh = h - by - 10;
        V.box(g, bx, by, bw, bh, { r: 12, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.55) });
        const [nm, tx2, ref] = st[S.i];
        T(g, `${S.i + 1}/7 · ${nm}`, bx + 16, by + 26, { size: 15, weight: 700, color: col, maxW: bw - 32 });
        const nl = wrap(g, tx2, bx + 16, by + 52, bw - 32, narrow ? 19 : 21, { size: narrow ? 13 : 14.5, color: C.ink, maxLines: narrow ? 12 : 9 });
        if (ref) wrap(g, 'onde: ' + ref, bx + 16, by + 52 + nl * (narrow ? 19 : 21) + 12, bw - 32, 15, { size: 11.5, font: 'mono', color: C.muted, maxLines: 4 });
        if (S.hold === Infinity || t < S.hold) T(g, 'pausado · clique em outra estação ou use as setas', bx + bw - 12, by + bh - 12, { size: 11, align: 'right', color: C.muted, maxW: bw - 24 });
      });
    }
  },
});

LA.chapter({
  id: 'receita-qwen',
  part: 9,
  order: 43,
  title: 'Reconstruindo a receita Qwen3.8-Flash-Next NVFP4',
  navTitle: 'Receita Qwen',
  subtitle: 'Do orçamento de bytes ao /health: cada decisão da receita dual-Spark do Qwen3.8-Flash-Next — lanes, checkpoints, consertos de metadados, argumentos do vLLM e números medidos — ligada ao conceito que a explica.',
  minutes: 55,
  recipes: ['qwen'],
  tags: ['receita', 'Qwen3.8-Flash-Next', 'NVFP4', 'lanes', 'vLLM 0.30', 'TP2+EP', 'MTP', 'KV FP8', 'precedência', 'overlays'],
  goals: [
    'Refazer o <b>orçamento de bytes</b> por nó e explicar por que a receita usa TP=2+EP, GMU 0,80 e 262.144 de contexto',
    'Distinguir as <b>lanes</b> (day-0, vLLM 0.30, FP8 oficial, TP1) e a <b>precedência</b> entre <code>.env</code>, ambiente e <code>OVERRIDE_*</code>',
    'Explicar cada conserto de checkpoint e loader (PLE, alias MTP, FP8 em blocos, fallback MXFP8, YaRN aninhado) e reconhecer o crash que ele evita',
    'Justificar cada argumento do <code>vllm serve</code> e ler o boot, os logs e os números medidos para tunar a receita',
  ],
  prereqs: ['engenharia-de-receita'],
  summary: [
    'O orçamento manda: 123,5 GiB de checkpoint não cabem num Spark de 121,69 GiB; com TP=2+EP cada nó guarda 64,46 GiB de pesos e o KV é <b>o que sobra</b> de GMU × 121,69 − (pesos + não-torch) − ativação; fora de 0,835 os tokens são extrapolação, e a lane 0.30 mediu menos (2,64 M FP8 a 0,80).',
    'GMU 0,80 (e não 0,835) porque, na memória unificada, o <code>MemAvailable</code> chegou a 0,3–0,9 GiB sob carga — zona de hard reset do GB10; o decode não depende do GMU.',
    'Uma lane é imagem × checkpoint × overlays × defaults validados juntos, ativada por um wrapper que exporta <code>OVERRIDE_*</code>; o <code>.env</code> vence o ambiente, exceto <code>ABLIT</code> e <code>HF_TOKEN</code>.',
    'Na imagem day-0 o checkpoint nvidia só carrega com três consertos: dtype da PLE via <code>--hf-overrides</code> aninhado em <code>text_config</code>, alias <code>mtp.layers.0 → 48</code> e ramo FP8 em blocos 128×128. O fallback MXFP8 (formas fora do envelope do kernel) serve a checkpoints com camadas MXFP8 e fica montado sempre, provavelmente inerte no nvidia.',
    'Os argumentos traduzem conceitos: TP2+EP (com DP=1 o EP só muda onde ficam os pesos), MTP=3 com vocabulário de rascunho de 47k, KV FP8 (1,70× tokens, −5%), CUDA graphs só no decode, visão replicada (<code>data</code>).',
    'Preflights trocam crashes de 7–10 min por falhas de segundos. No batch 1, ~41 dos ~60 ms do passo são leitura de pesos (9,2 GiB) e o resto é overhead fixo; com ā ≈ 3,2 tokens por passo isso dá os 52 tok/s medidos. Tunar é ler menos bytes, aceitar mais tokens, servir mais sequências ou encolher o overhead.',
  ],
  quiz: [
    { q: 'Você roda <code>KV_CACHE_DTYPE=auto ./start.sh</code> e o <code>.env</code> contém <code>KV_CACHE_DTYPE=fp8</code>. O que o vLLM recebe?', o: ['<code>auto</code>, porque a linha de comando sempre vence', '<code>fp8</code>, porque o <code>.env</code> é lido depois do ambiente e vence para as chaves que define', 'Um erro por variável definida duas vezes', '<code>auto</code>, porque essa chave é capturada como <code>_CLI_*</code>'], a: 1, e: 'O <code>start.sh</code> faz <code>source .env</code> depois de ler o ambiente; só <code>ABLIT</code> e <code>HF_TOKEN</code> são capturados antes e reaplicados. Por isso a receita manda editar o <code>.env</code> e conferir a linha “KV dtype:” do resumo. Para trocar o KV por lane existe <code>OVERRIDE_KV_CACHE_DTYPE</code>, lido depois do <code>.env</code>.' },
    { q: 'Por que a receita usa <code>--mm-encoder-tp-mode data</code>?', o: ['Para acelerar o prefill de imagens', 'Porque com <code>weights</code> a MLP da visão (4304) viraria 2152 por GPU, que não é múltiplo de 16, e a carga crasha', 'Porque o encoder de visão não cabe numa GPU', 'Porque o NCCL não transporta tensores de visão'], a: 1, e: '4304 / 2 = 2152 e 2152 mod 16 = 8. O modo <code>data</code> replica o encoder de 0,84 GiB em cada GPU e contorna a exigência de divisibilidade — uma “taxa de divisibilidade” do paralelismo de tensor.' },
    { q: 'Sem o patch de configuração, por que a MTP morria ~7 minutos depois do início da carga?', o: ['O rascunhador não cabia na memória', 'O vLLM cria a camada como <code>mtp.layers.48</code> e procura a quantização por string exata; o checkpoint só anota <code>mtp.layers.0</code>, então a MoE é montada sem quantização e falta <code>w2_weight_scale_inv</code>', 'A MTP exige YaRN ligado', 'O NCCL expirava durante a carga'], a: 1, e: 'Índice absoluto (<code>num_hidden_layers + i</code> = 48) contra metadado relativo (0): o lookup devolve <code>None</code> em silêncio e o erro só aparece quando o loader procura as escalas FP8. O alias <code>.0 → .48</code> e o ramo <code>FP8_BLOCK_SCALES</code> resolvem; <code>--mtp-moe-algo</code> antecipa a falha para segundos.' },
    { q: 'Na lane vLLM 0.30, qual é o dtype de KV padrão e o custo de trocá-lo?', o: ['FP8, sem custo', 'BF16 (os kernels QSA da 0.30 só aceitam BF16); com <code>OVERRIDE_KV_CACHE_DTYPE=fp8</code> o backport de vllm#55557 dá 2,64 M tokens com ~5% a menos de velocidade', 'NVFP4, com perda de qualidade', 'BF16, e FP8 é impossível nessa lane'], a: 1, e: 'A lane exporta <code>OVERRIDE_KV_CACHE_DTYPE=auto</code> porque os kernels QSA stock da 0.30 recusam FP8. O opt-in monta o backport (a apagar na 0.31+): de ~1,34–1,52 M para 2,64 M tokens, ~5% mais lento, NLL 1,328/1,334.' },
    { q: 'O vocabulário reduzido de 47.149 ids do rascunhador muda o texto gerado?', o: ['Sim: o modelo passa a gerar só esses tokens', 'Não: um rascunho fora do conjunto é rejeitado na verificação; só cai o custo de ler o <code>lm_head</code> do rascunhador (0,59 → 0,22 GiB no rank crítico)', 'Sim, mas só em textos em chinês', 'Não, porque ele só vale no prefill'], a: 1, e: 'Quem decide cada token é o modelo alvo, com o vocabulário completo. O rascunhador só propõe; se propõe pior, a aceitação cai, mas a saída não muda. Ganho medido: +8,4% a +9,6%. Cuidado: um vocabulário “acolchoado” (65k ids com só 8,3k reais) derrubou a aceitação para 47,8%.' },
    { q: 'Por que os containers de serving não têm <code>--restart</code>?', o: ['O Docker não suporta restart com GPU', 'Um rank reiniciado sozinho fica esperando o parceiro no init do NCCL; é melhor um supervisor derrubar os dois e rodar <code>start.sh --launch</code>', 'O restart apagaria o KV cache em disco', 'Porque o container nunca cai'], a: 1, e: 'Em TP de dois nós os ranks precisam entrar juntos no rendezvous. Um restart isolado trava. O caminho sugerido é <code>stop.sh</code> + <code>start.sh --launch</code>, que pula download e sync e reaplica os overlays nos dois nós.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x, d = 0) => Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const setFont = (g, o) => { g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); };
    const tw = (g, s, o) => { setFont(g, o || {}); return g.measureText(String(s)).width; };
    const wrapText = (g, s, x, y, maxW, lh, o = {}) => {
      const words = String(s).split(/\s+/);
      let line = '', yy = y;
      setFont(g, o);
      for (const wd of words) {
        const t = line ? line + ' ' + wd : wd;
        if (line && g.measureText(t).width > maxW) { V.text(g, line, x, yy, o); setFont(g, o); yy += lh; line = wd; } else line = t;
      }
      if (line) { V.text(g, line, x, yy, o); yy += lh; }
      return yy;
    };
    const countLines = (g, s, maxW, o = {}) => {
      const words = String(s).split(/\s+/); let line = '', n = 0;
      setFont(g, o);
      for (const wd of words) { const t = line ? line + ' ' + wd : wd; if (line && g.measureText(t).width > maxW) { n++; line = wd; } else line = t; }
      return n + (line ? 1 : 0);
    };
    const fitText = (g, s, maxW, o) => {
      s = String(s);
      if (tw(g, s, o) <= maxW) return s;
      while (s.length > 2 && tw(g, s + '…', o) > maxW) s = s.slice(0, -1);
      return s + '…';
    };
    const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const beforeCap = (fig, el) => fig.insertBefore(el, fig.querySelector('figcaption'));

    /* =====================================================================
       FIG 1 — orçamento de memória de um nó
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-orcamento');
      const S = { gmu: 0.80, kv: 'fp8', ctx: 262144 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'GMU', min: 0.70, max: 0.90, step: 0.005, value: 0.80, fmt: (v) => v.toFixed(3).replace('.', ','), onInput: (v) => { S.gmu = v; H.redraw(); } });
      UI.seg(ctl, { label: 'KV', options: [{ v: 'fp8', label: 'fp8' }, { v: 'bf16', label: 'auto (BF16)' }], value: 'fp8', onChange: (v) => { S.kv = v; H.redraw(); } });
      UI.seg(ctl, { label: 'max-model-len', options: [{ v: 262144, label: '262.144' }, { v: 1000000, label: '1M (YaRN)' }], value: 262144, onChange: (v) => { S.ctx = v; H.redraw(); } });
      const rPool = UI.readout(ctl, 'Pool de KV', '—');
      const rTok = UI.readout(ctl, 'Tokens (extrap.)', '—');
      const rConc = UI.readout(ctl, 'Concorrência', '—');
      // medições reais de tokens de KV (qwen-launch §4.2): [gmu, min, max, rótulo curto, rótulo longo]
      const MEAS = {
        fp8: [[0.835, 3.652, 3.652, '0,835 · day-0 · SSM fp32', '0,835 · day-0 · SSM float32 (o τ vem daqui)'], [0.835, 4.238, 4.379, '0,835 · day-0 · SSM bf16', '0,835 · day-0 · SSM bf16 (runs 09-10 e 09-16)'], [0.80, 2.64, 2.64, '0,80 · lane 0.30', '0,80 · lane vLLM 0.30 (backport de KV FP8)']],
        bf16: [[0.835, 2.131, 2.131, '0,835 · day-0 · SSM fp32', '0,835 · day-0 · SSM float32 (o τ vem daqui)'], [0.835, 2.426, 2.426, '0,835 · day-0 · SSM bf16', '0,835 · day-0 · SSM bf16 (rerun 09-16)'], [0.80, 1.34, 1.52, '0,80 · lane 0.30', '0,80 · lane vLLM 0.30 (três medições: 1,34–1,52 M)']],
      };
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const TOT = 121.69, WGT = 64.46, NT = 68.52 - 64.46, ACT = 1.07, GR = 0.54;
        const budget = S.gmu * TOT;
        const kv = Math.max(0, budget - 68.52 - ACT - (S.kv === 'bf16' ? 0.32 : 0));
        const tpg = S.kv === 'fp8' ? 114060 : 67229;
        const tokens = kv * tpg, conc = tokens / S.ctx;
        rPool.set(nf(kv, 2) + ' GiB'); rTok.set(nf(tokens / 1e6, 2) + ' M'); rConc.set(nf(conc, 2) + '×');
        const narrow = w < 560, m = 16, bx = m, bw = w - 2 * m, sx = bw / TOT;
        V.text(g, 'UM NÓ · 121,69 GiB VISÍVEIS AO CUDA', bx, 20, { size: 11, font: 'mono', color: C.muted });
        const by = 52, bh = 46;
        const segs = [
          { l: 'Pesos', v: WGT, c: C.b },
          { l: 'Não-torch', v: NT, c: C.f },
          { l: 'Ativação (pico)', v: ACT, c: C.d },
          { l: 'KV cache', v: kv, c: C.a },
          { l: 'CUDA graphs', v: GR, c: C.e },
          { l: narrow ? 'Fora do orç.' : 'Fora do orçamento', v: Math.max(0, TOT - WGT - NT - ACT - kv - GR), c: C.muted },
        ];
        g.save(); V.rr(g, bx, by, bw, bh, 10); g.clip();
        g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(bx, by, bw, bh);
        let x = bx;
        segs.forEach((sg, i) => {
          const ww = sg.v * sx;
          g.fillStyle = V.alpha(sg.c, i === 5 ? 0.16 : i === 3 ? 0.62 : 0.45); g.fillRect(x, by, ww, bh);
          if (i === 3) { g.strokeStyle = C.a; g.lineWidth = 2; g.strokeRect(x + 1, by + 1, Math.max(0, ww - 2), bh - 2); }
          const lab = i === 0 ? 'Pesos ' + nf(sg.v, 2) : i === 3 ? 'KV ' + nf(sg.v, 1) : '';
          if (lab && tw(g, lab, { size: 12, weight: 600 }) < ww - 10) V.text(g, lab, x + 8, by + bh / 2 + 4, { size: 12, weight: 600, color: C.ink });
          x += ww;
        });
        g.restore();
        // linha do orçamento
        const xB = bx + budget * sx;
        V.line(g, xB, by - 12, xB, by + bh + 8, { color: C.ink, lw: 1.5, dash: [4, 3] });
        const bl = 'GMU × 121,69 = ' + nf(budget, 2) + ' GiB';
        const blw = tw(g, bl, { size: 11, font: 'mono' });
        V.text(g, bl, Math.min(w - m, Math.max(m + blw, xB - 6)), by - 16, { size: 11, font: 'mono', align: 'right', color: C.ink });
        // legenda
        const cols = narrow ? 2 : 3, lw = (w - 2 * m) / cols;
        let ly = by + bh + 28;
        segs.forEach((sg, i) => {
          const cx = m + (i % cols) * lw, cy = ly + Math.floor(i / cols) * 20;
          V.box(g, cx, cy - 9, 11, 11, { r: 3, fill: V.alpha(sg.c, i === 5 ? 0.3 : 0.75) });
          const txt = sg.l + (i === 4 && !narrow ? '*' : '') + ' ' + nf(sg.v, 2);
          V.text(g, fitText(g, txt, lw - 20, { size: 11.5 }), cx + 17, cy, { size: 11.5, color: C.ink });
        });
        ly += Math.ceil(segs.length / cols) * 20 + 6;
        if (!narrow) { V.text(g, '* pela aritmética do log, os graphs ficam fora da soma orçada', m, ly, { size: 11, color: C.muted }); ly += 26; }
        else ly += 6;
        // cápsulas de requisições
        const lab = narrow ? 'Cabem ' + nf(conc, 2) + ' contextos cheios (âmbar = 8 seqs)' : 'Requisições de ' + nf(S.ctx) + ' tokens que cabem juntas (extrapolado): ' + nf(conc, 2) + ' (contorno âmbar = as 8 que --max-num-seqs deixa rodar)';
        ly = wrapText(g, lab, m, ly, w - 2 * m, 16, { size: 12, weight: 500, color: C.ink }) + 4;
        const cw = narrow ? 20 : 28, chh = 16, gap = 5;
        const maxCaps = Math.floor((w - 2 * m + gap) / (cw + gap));
        const full = Math.floor(conc), frac = conc - full;
        const shown = Math.min(maxCaps, Math.ceil(conc));
        for (let k = 0; k < shown; k++) {
          const cx = m + k * (cw + gap);
          const fillW = k < full ? cw : cw * frac;
          V.box(g, cx, ly, cw, chh, { r: 8, fill: V.alpha(C.a, 0.12), stroke: k < 8 ? C.d : V.alpha(C.a, 0.5), lw: k < 8 ? 2 : 1 });
          if (fillW > 1) { g.save(); V.rr(g, cx, ly, cw, chh, 8); g.clip(); g.fillStyle = V.alpha(C.a, 0.6); g.fillRect(cx, ly, fillW, chh); g.restore(); }
        }
        if (Math.ceil(conc) > maxCaps) V.text(g, '+' + (Math.ceil(conc) - maxCaps), m + maxCaps * (cw + gap) - 4, ly - 4, { size: 11, align: 'right', color: C.muted });
        ly += chh + 26;
        // régua de tokens: extrapolado (▼) × medido (◆)
        const meas = MEAS[S.kv];
        const atG = (gm) => Math.abs(gm - S.gmu) < 0.0026;
        V.text(g, narrow ? 'TOKENS DE KV · ▼ extrapolado × ◆ medido' : 'TOKENS DE KV (' + (S.kv === 'fp8' ? 'fp8' : 'BF16') + ') · ▼ extrapolação linear × ◆ medições reais', m, ly, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * m });
        const axY = ly + 40, ax0 = m + 4, ax1 = w - m - 4, TMAX = 5;
        const tx = (v) => ax0 + (Math.min(TMAX, v) / TMAX) * (ax1 - ax0);
        V.line(g, ax0, axY, ax1, axY, { color: V.alpha(C.muted, 0.6), lw: 1.5 });
        for (let k = 0; k <= TMAX; k++) { const xx = tx(k); V.line(g, xx, axY - 4, xx, axY + 4, { color: V.alpha(C.muted, 0.7), lw: 1 }); V.text(g, k + ' M', xx, axY + 17, { size: 11, font: 'mono', align: k === 0 ? 'left' : k === TMAX ? 'right' : 'center', color: C.muted }); }
        const diamond = (xx, yy, r, fill, stroke) => { g.beginPath(); g.moveTo(xx, yy - r); g.lineTo(xx + r, yy); g.lineTo(xx, yy + r); g.lineTo(xx - r, yy); g.closePath(); if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.6; g.stroke(); } };
        meas.forEach(([gm, lo, hi]) => {
          const on = atG(gm), col = on ? C.d : V.alpha(C.muted, 0.9);
          if (hi > lo) V.line(g, tx(lo), axY, tx(hi), axY, { color: col, lw: 5 });
          diamond(tx((lo + hi) / 2), axY, on ? 7 : 5.5, on ? C.d : C.bg, col);
        });
        // extrapolado
        const xe = tx(tokens / 1e6);
        g.beginPath(); g.moveTo(xe, axY - 3); g.lineTo(xe - 7, axY - 15); g.lineTo(xe + 7, axY - 15); g.closePath(); g.fillStyle = C.a; g.fill();
        const el = '▼ ' + nf(tokens / 1e6, 2) + ' M';
        const elw = tw(g, el, { size: 11, font: 'mono', weight: 700 });
        V.text(g, el, Math.max(m + elw / 2, Math.min(w - m - elw / 2, xe)), axY - 20, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.a });
        ly = axY + 36;
        meas.forEach(([gm, lo, hi, sl, ll]) => {
          const on = atG(gm);
          diamond(m + 6, ly - 4, 5, on ? C.d : null, on ? C.d : C.muted);
          const val = hi > lo ? nf(lo, 2) + '–' + nf(hi, 2) + ' M' : nf(lo, 2) + ' M';
          V.text(g, fitText(g, val + ' · ' + (narrow ? sl : ll), w - 2 * m - 18, { size: 11, font: 'mono' }), m + 18, ly, { size: 11, font: 'mono', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
          ly += 16;
        });
        ly += 8;
        // status
        let st, col;
        const gm = S.gmu;
        if (gm > 0.8375) { st = narrow ? 'Acima de 0,835: menos folga que o ponto em que o host quase caiu.' : 'Acima de 0,835: menos folga que o ponto em que o host já quase caiu (0,85 foi o valor histórico).'; col = C.red; }
        else if (gm >= 0.8325) { st = narrow ? '0,835 (medido): MemAvailable caiu a 0,3–0,9 GiB num soak, zona de hard reset. O KV é o do log.' : '0,835 (medido): o driver segurava 114,5 GiB/nó e o MemAvailable caiu a 0,3–0,9 GiB num soak — zona de hard reset do GB10. O KV desta barra é o do log.'; col = C.red; }
        else if (gm >= 0.7975 && gm <= 0.8025) { st = narrow ? '0,80: mediu-se a folga (soak de 1 h, lane 0.30: MemAvailable ≥ 4,5/7,8 GiB). O KV da barra é extrapolado; o medido está nos ◆.' : '0,80 — o que se mediu aqui foi a folga: 1 h de soak (lane 0.30) com MemAvailable mínimo de 4,5 GiB (head) e 7,8 GiB (worker). Valor atual da receita. O KV da barra é extrapolado; o medido está nos ◆.'; col = C.e; }
        else if (gm > 0.8025) { st = narrow ? 'Entre os pontos medidos: não validado. +0,01 tira ~1,2 GiB de folga.' : 'Entre os pontos medidos: não validado. Cada +0,01 tira ~1,2 GiB da folga do sistema.'; col = C.d; }
        else { st = narrow ? 'Mais folga que o validado: menos KV (~1,2 GiB por −0,01). Decode igual.' : 'Mais folga que o validado: mais seguro, menos KV (~1,2 GiB a menos por −0,01). Decode igual.'; col = C.muted; }
        const nl = countLines(g, st, w - 2 * m - 24, { size: 12 });
        V.box(g, m, ly, w - 2 * m, nl * 16 + 14, { r: 10, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.6) });
        wrapText(g, st, m + 12, ly + 19, w - 2 * m - 24, 16, { size: 12, color: C.ink });
      }, { static: true });
    })();

    /* =====================================================================
       FIG 3 — topologia
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-topologia');
      const S = { plane: 'dados', cross: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Plano', options: [{ v: 'dados', label: 'dados' }, { v: 'controle', label: 'controle' }, { v: 'pesos', label: 'pesos & ops' }, { v: 'api', label: 'API' }], value: 'dados', onChange: (v) => { S.plane = v; } });
      UI.toggle(ctl, { label: 'cabeamento cruzado (head f1 ↔ worker f0)', value: true, onChange: (v) => { S.cross = v; } });
      const parts = Array.from({ length: 30 }, (_, i) => ({ p: i / 30, dir: i % 2, sp: 0.55 + (i % 5) * 0.1 }));
      const INFO = {
        dados: { t: 'DADOS — coletivas NCCL sobre RDMA (RoCE v2)', lines: (c) => [
          'NCCL_IB_HCA = ' + (c ? '=rocep1s0f1 (head) · =rocep1s0f0 (worker)' : '=rocep1s0f0 (nos dois)'),
          'NCCL_IB_GID_INDEX = 3 (se não subir, 5) · NCCL_IB_DISABLE=0 · NCCL_IB_AUTO_DETECT=0',
          'container: --device /dev/infiniband · --ulimit memlock=-1',
          '≈106 coletivas por passo de decode, 5–20 KB cada, 20–30 µs → 2–3 ms/passo',
        ] },
        controle: { t: 'CONTROLE — rendezvous, Gloo, bootstrap e ZMQ (TCP)', lines: (c) => [
          c ? 'IFACE = enp1s0f1np1 (head) · WORKER_IFACE = enp1s0f0np0' : 'IFACE = enp1s0f0np0 (nos dois)',
          'GLOO_SOCKET_IFNAME = NCCL_SOCKET_IFNAME = TP_SOCKET_IFNAME = IFACE',
          '--master-addr $HEAD_IP --master-port 50000 · VLLM_HOST_IP = IP de cada nó',
          'broadcast scheduler → worker a cada passo: ~0,15–0,4 ms de ida e volta',
        ] },
        pesos: { t: 'PESOS & OPS — SSH, rsync/NFS e imagem', lines: () => [
          'WORKER_USER: SSH sem senha head → worker (pré-requisito)',
          'rsync ~126 GiB para o disco do worker (padrão) ou NFS com --nfs',
          'NFS_SERVER_IP: IP do CX7 (ex.: 10.0.22.1), nunca o alias 10.0.0.1',
          'docker save … | ssh worker docker load · script do worker via ssh "bash -s"',
        ] },
        api: { t: 'API — só o head atende', lines: () => [
          'cliente → head:8888 (--host 0.0.0.0 --port 8888)',
          'worker: --headless, sem servidor de API',
          '/v1/chat/completions · /v1/models · /health · /metrics',
          'sem autenticação por padrão: exponha com cuidado',
        ] },
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, m = 14;
        const top = 46, nh = narrow ? 170 : 190;
        const gap = narrow ? 44 : Math.min(230, Math.max(120, w * 0.22));
        const nw = Math.min(290, (w - 2 * m - gap) / 2);
        const x0 = (w - (2 * nw + gap)) / 2, x1 = x0 + nw + gap;
        const plane = S.plane;
        const pc = plane === 'dados' ? C.a : plane === 'controle' ? C.b : plane === 'pesos' ? C.d : C.c;
        const ports = {};
        const drawNode = (x, head) => {
          V.box(g, x, top, nw, nh, { r: 14, fill: V.alpha(C.qwen, 0.06), stroke: V.alpha(C.qwen, 0.55) });
          V.text(g, head ? 'HEAD · rank 0' : 'WORKER · rank 1', x + 11, top + 20, { size: narrow ? 11.5 : 12.5, weight: 700, color: C.qwen });
          if (!head) V.text(g, '--headless', x + nw - 10, narrow ? top + 36 : top + 20, { size: 11, font: 'mono', align: 'right', color: plane === 'api' ? C.c : C.muted });
          V.text(g, narrow ? (head ? '10.0.0.1' : '10.0.0.2') : head ? '10.0.0.1 · CX7 10.0.22.1' : '10.0.0.2 · CX7 10.0.22.2', x + 11, top + 36, { size: 11, font: 'mono', color: C.muted, maxW: nw - 20 });
          V.box(g, x + 9, top + 46, nw - 18, 44, { r: 9, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.45) });
          V.text(g, narrow ? 'GPU GB10 (sm_121)' : 'GPU GB10 · sm_121 · 48 SMs', x + 18, top + 64, { size: 11.5, weight: 600, color: C.ink, maxW: nw - 34 });
          V.text(g, narrow ? '121,69 GiB UMA' : '121,69 GiB unificados · ~273 GB/s', x + 18, top + 81, { size: 11, color: C.muted, maxW: nw - 34 });
          const ny = top + 100, nhh = nh - 110;
          V.box(g, x + 9, ny, nw - 18, nhh, { r: 9, fill: V.alpha(pc, plane === 'api' ? 0.04 : 0.1), stroke: V.alpha(pc, 0.5) });
          V.text(g, narrow ? 'CX7 · RoCE v2' : 'ConnectX-7 · RoCE v2 · 200 GbE', x + 18, ny + 17, { size: 11, weight: 600, color: C.ink, maxW: nw - 50 });
          const px = head ? x + nw - 9 : x + 9;
          const act = head ? (S.cross ? 1 : 0) : 0;
          [0, 1].forEach((k) => {
            const py = ny + nhh * (k === 0 ? 0.5 : 0.8);
            const on = k === act;
            V.dot(g, px, py, on ? 6.5 : 5, on ? pc : V.alpha(C.muted, 0.5));
            V.dot(g, px, py, 2.4, C.bg || '#000');
            const lx = head ? px - 12 : px + 12;
            V.text(g, 'f' + k + (narrow ? '' : ' · rocep1s0f' + k), lx, py + 4, { size: 11, font: 'mono', align: head ? 'right' : 'left', color: on ? C.ink : C.muted });
            ports[(head ? 'h' : 'w') + k] = [px, py];
          });
        };
        drawNode(x0, true); drawNode(x1, false);
        const hp = ports['h' + (S.cross ? 1 : 0)], wp = ports.w0;
        const c1 = [hp[0] + gap * 0.45, hp[1]], c2 = [wp[0] - gap * 0.45, wp[1]];
        g.beginPath(); g.moveTo(hp[0], hp[1]); g.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], wp[0], wp[1]);
        g.strokeStyle = plane === 'api' ? V.alpha(C.muted, 0.4) : V.alpha(pc, 0.55); g.lineWidth = plane === 'dados' ? 5 : 3;
        if (plane === 'controle') g.setLineDash([6, 5]);
        g.stroke(); g.setLineDash([]);
        const P0 = hp, P1 = c1, P2 = c2, P3 = wp;
        if (plane === 'dados') {
          parts.forEach((p) => { const u = (p.p + t * p.sp * 0.9) % 1; const pt = V.bez(P0, P1, P2, P3, p.dir ? u : 1 - u); V.dot(g, pt[0], pt[1], 2.8, p.dir ? C.ink : C.d); });
        } else if (plane === 'controle') {
          parts.slice(0, 6).forEach((p, i) => { const u = (p.p * 5 + t * 0.35) % 1; const pt = V.bez(P0, P1, P2, P3, i % 2 ? u : 1 - u); V.dot(g, pt[0], pt[1], 4.5, C.ink); V.dot(g, pt[0], pt[1], 2.5, C.b); });
        } else if (plane === 'pesos') {
          for (let k = 0; k < 3; k++) { const u = (k / 3 + t * 0.12) % 1; const pt = V.bez(P0, P1, P2, P3, u); V.box(g, pt[0] - 9, pt[1] - 6, 18, 12, { r: 3, fill: V.alpha(C.d, 0.85) }); }
        }
        if (plane === 'api') {
          const cxp = x0 + 8, cyp = 10, cwp = Math.min(nw - 16, 170);
          V.box(g, cxp, cyp, cwp, 24, { r: 12, fill: V.alpha(C.c, 0.18), stroke: C.c, label: 'cliente (API OpenAI)', size: 11 });
          const ax = x0 + nw * 0.7;
          V.arrow(g, cxp + cwp / 2, cyp + 24, ax, top - 2, { color: C.c, lw: 2 });
          const u = (t * 0.8) % 1;
          V.dot(g, V.lerp(cxp + cwp / 2, ax, u), V.lerp(cyp + 24, top - 2, u), 3.5, C.c);
          V.text(g, ':8888', ax + 6, top - 8, { size: 11, font: 'mono', color: C.c });
        } else {
          V.text(g, (S.cross ? 'cabeamento cruzado: f1 ↔ f0' : 'porta a porta: f0 ↔ f0'), w / 2, 22, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // painel de variáveis
        const inf = INFO[plane];
        let y = top + nh + 24;
        V.text(g, inf.t, m, y, { size: 11.5, weight: 700, color: pc, maxW: w - 2 * m });
        y += 18;
        inf.lines(S.cross).forEach((ln) => { V.dot(g, m + 4, y - 4, 2.5, pc); y = wrapText(g, ln, m + 14, y, w - 2 * m - 14, 15.5, { size: 11.5, font: 'mono', color: C.ink }) + 3; });
      });
    })();

    /* =====================================================================
       FIG 4 — mapa de decisões: lanes como metrô
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-lanes');
      const LN = {
        day0: { name: 'day-0', cv: '--qwen', wrapper: './start.sh', image: 'vllm/vllm-openai:qwen38-flash-next — build de desenvolvimento da NVIDIA (~20,6 GB), código em qwen3_8_flash_next/nvidia', ckpt: 'nvidia/Qwen3.8-Flash-Next-NVFP4', overlays: 'PLE, MXFP8 + FP8 em blocos para MoE, alias MTP, KV FP8 nos kernels QSA, block-drop, vocabulário de rascunho 47k (e determinismo, opcional)', kv: 'fp8 (patch_qsa_fp8_kv.py aplicado automaticamente)', res: '3.652.200 tokens de KV a GMU 0,835 (13,93× a 262k); decode S=1 prosa/código 56,1/75,8 tok/s; TTFT do 2º turno 0,63 s', when: 'Máximo de sessões longas em cache; é a única lane com determinismo e perfil QSA.' },
        v030: { name: 'vLLM 0.30', cv: '--viz-b', wrapper: './start-v030.sh', image: 'vllm/vllm-openai:v0.30.0 — release estável, qwen4_exp nativo', ckpt: 'nvidia/Qwen3.8-Flash-Next-NVFP4 (carregado nativamente)', overlays: 'só o vocabulário de rascunho (patch_mtp_draft_vocab_v030.py) e, opcional, o backport de KV FP8 (vllm#55557)', kv: 'auto = BF16 por padrão; fp8 opt-in via OVERRIDE_KV_CACHE_DTYPE=fp8 (2,64 M tokens, ~5% mais lento)', res: '1,34–1,52 M tokens BF16 a GMU 0,80; prosa 60,0 vs 56,3 tok/s (S=1), código 441 vs 422 (S=8); TTFT multi-turno 0,28–0,40 s; #62 corrigido upstream', when: 'Agentes e conversas multi-turno: TTFT duas vezes menor e decode igual ou maior.' },
        fp8: { name: 'FP8 oficial', cv: '--viz-d', wrapper: './start-fp8.sh', image: 'a do .env (day-0)', ckpt: 'Qwen/Qwen3.8-Flash-Next-FP8 (experts FP8 em blocos de 128; GDN, atenção, HC e lm_head seguem BF16)', overlays: 'sem patch da PLE (SKIP_PLE_PATCH=true); demais overlays da day-0', kv: 'o do .env; contexto nativo 262.144, YaRN desligado', res: '~500 mil tokens de KV (experts FP8 ocupam o dobro dos NVFP4)', when: 'Quando você quer os pesos oficiais do Qwen como referência de qualidade.' },
        tp1: { name: 'TP1 (um Spark)', cv: '--viz-c', wrapper: './start-tp1.sh ou tp1/start.sh', image: 'a do .env (day-0)', ckpt: 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (98,6 GiB; RadixArk, 125,9 GiB, não cabe)', overlays: 'offload da PLE com handshake no host + tabela NVFP4 empacotada em mmap (26,8 GiB) + memwatch.sh', kv: 'auto (BF16) por padrão; GMU derivado do orçamento', res: '685.825 tokens a GMU 0,813; decode 22,0 → prosa 24,9 / código 35,1 tok/s com MTP', when: 'Só existe um Spark. Exige --distributed-executor-backend mp, checkpoint menor e watchdog de memória.' },
      };
      const OBJ = [
        { v: 'agentes', label: 'Agentes multi-turno (TTFT baixo)', lane: 'v030' },
        { v: 'cache', label: 'Muitas sessões longas em cache', lane: 'day0' },
        { v: 'ab', label: 'A/B bit-idênticos, experimentos de kernel', lane: 'day0' },
        { v: 'oficial', label: 'Pesos oficiais do Qwen', lane: 'fp8' },
        { v: 'um', label: 'Só tenho um Spark', lane: 'tp1' },
      ];
      const order = ['day0', 'v030', 'fp8', 'tp1'];
      const home = { day0: 0, v030: 1, fp8: 2, tp1: 3 };
      const PATH = { day0: ['w0', 'i0', 'p0', 'k0', 'r0'], v030: ['w1', 'i1', 'p0', 'k1', 'r1'], fp8: ['w2', 'i0', 'p1', 'k0', 'r2'], tp1: ['w3', 'i0', 'p2', 'k1', 'r3'] };
      const LBL = {
        w0: ['start.sh', 'start'], w1: ['start-v030.sh', 'v030'], w2: ['start-fp8.sh', 'fp8'], w3: ['start-tp1.sh', 'tp1'],
        i0: ['qwen38-flash-next', 'day-0'], i1: ['v0.30.0', 'v0.30'],
        p0: ['nvidia NVFP4', 'nvidia'], p1: ['Qwen FP8', 'FP8'], p2: ['local-inference-lab', 'lil'],
        k0: ['fp8 + patch', 'fp8'], k1: ['BF16 (auto)', 'bf16'],
        r0: ['3,65 M tok', '3,65M'], r1: ['1,34–1,52 M', '1,4M'], r2: ['~500 mil', '500k'], r3: ['686 mil', '686k'],
      };
      const HEAD = ['wrapper', 'imagem', 'pesos', 'KV', 'tokens de KV'];
      // estações: coluna + lanes que passam
      const ST = {};
      order.forEach((ln) => PATH[ln].forEach((sid, c) => { (ST[sid] = ST[sid] || { c, lanes: [] }).lanes.push(ln); }));
      Object.values(ST).forEach((s) => { s.slot = s.lanes.reduce((a, l) => a + home[l], 0) / s.lanes.length; });
      const S = { lane: 'v030' };
      const ctl = UI.controls(fig);
      const sel = UI.select(ctl, { label: 'Objetivo', options: OBJ.map((o) => ({ v: o.v, label: o.label })), value: 'agentes', onChange: (v) => { const o = OBJ.find((x) => x.v === v); if (o) seg.set(o.lane); } });
      const seg = UI.seg(ctl, { label: 'Lane', options: order.map((k) => ({ v: k, label: LN[k].name })), value: 'v030', onChange: (v) => { S.lane = v; render(); } });
      sel.el.style.flexWrap = 'wrap'; sel.el.style.maxWidth = '100%';
      const selEl = sel.el.querySelector('select'); selEl.style.maxWidth = '100%'; selEl.style.minWidth = '0';
      const panel = LA.h('div', { style: { padding: '2px 18px 12px', overflowWrap: 'anywhere' } });
      beforeCap(fig, panel);
      function render() {
        const L = LN[S.lane];
        const obj = OBJ.filter((o) => o.lane === S.lane).map((o) => o.label).join('; ');
        panel.innerHTML = `<dl class="la-kv" style="margin:.4em 0">
          <dt style="color:var(${L.cv})">lane</dt><dd><b>${esc(L.name)}</b> — ${esc(L.when)}${obj ? ' <i>(objetivo: ' + esc(obj) + ')</i>' : ''}</dd>
          <dt>wrapper</dt><dd><code>${esc(L.wrapper)}</code></dd>
          <dt>imagem</dt><dd>${esc(L.image)}</dd>
          <dt>pesos</dt><dd>${esc(L.ckpt)}</dd>
          <dt>overlays</dt><dd>${esc(L.overlays)}</dd>
          <dt>KV</dt><dd>${esc(L.kv)}</dd>
          <dt>medido</dt><dd>${esc(L.res)}</dd></dl>`;
      }
      render();
      void sel;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const col = { day0: C.qwen, v030: C.b, fp8: C.d, tp1: C.c };
        const narrow = w < 600, m = narrow ? 30 : 60;
        const topY = 58, botY = h - 44;
        const colX = (c) => m + (c * (w - 2 * m)) / 4;
        const slotY = (s) => topY + (s * (botY - topY)) / 3;
        const off = 7;
        (narrow ? ['wrapper', 'imagem', 'pesos', 'KV', 'tokens'] : HEAD).forEach((hd, c) => V.text(g, hd.toUpperCase(), colX(c), 22, { size: 11, font: 'mono', align: 'center', color: C.muted }));
        const pos = (ln, c) => {
          const s = ST[PATH[ln][c]];
          const idx = s.lanes.indexOf(ln), n = s.lanes.length;
          return [colX(c), slotY(s.slot) + (idx - (n - 1) / 2) * off];
        };
        const segs = {};
        order.forEach((ln) => {
          const pts = [0, 1, 2, 3, 4].map((c) => pos(ln, c));
          const on = ln === S.lane;
          g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
          const sg = [];
          for (let k = 0; k < 4; k++) {
            const a = pts[k], b = pts[k + 1], mx = (a[0] + b[0]) / 2;
            g.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]);
            sg.push([a, [mx, a[1]], [mx, b[1]], b]);
          }
          segs[ln] = sg;
          g.strokeStyle = on ? col[ln] : V.alpha(col[ln], 0.28); g.lineWidth = on ? 6 : 4; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
        });
        // estações
        Object.entries(ST).forEach(([sid, s]) => {
          const x = colX(s.c), y = slotY(s.slot), n = s.lanes.length;
          const onSel = s.lanes.includes(S.lane);
          if (n > 1) V.box(g, x - 9, y - (n - 1) * off / 2 - 9, 18, (n - 1) * off + 18, { r: 9, fill: C.bg || '#111', stroke: onSel ? C.ink : V.alpha(C.ink, 0.35), lw: 2 });
          else { V.dot(g, x, y, 8, C.bg || '#111'); g.strokeStyle = onSel ? col[s.lanes[0]] : V.alpha(col[s.lanes[0]], 0.4); g.lineWidth = 3; g.beginPath(); g.arc(x, y, 8, 0, Math.PI * 2); g.stroke(); }
          if (narrow && !onSel) return;
          const lab = LBL[sid][narrow ? 1 : 0];
          const below = s.slot >= 1.5;
          const ly = below ? y + (n - 1) * off / 2 + 22 : y - (n - 1) * off / 2 - 14;
          V.text(g, lab, x, ly, { size: 11, font: 'mono', align: 'center', weight: onSel ? 600 : 400, color: onSel ? C.ink : C.muted, maxW: (w - 2 * m) / 4 + 10 });
        });
        // trem
        const sg = segs[S.lane];
        const u = (t * 0.1) % 1, k = Math.min(3, Math.floor(u * 4)), f = u * 4 - k;
        const p = V.bez(sg[k][0], sg[k][1], sg[k][2], sg[k][3], V.smooth(f));
        V.glow(g, p[0], p[1], 20, col[S.lane]);
        V.box(g, p[0] - 13, p[1] - 7, 26, 14, { r: 7, fill: col[S.lane], stroke: C.ink, lw: 1.2 });
        V.dot(g, p[0] - 5, p[1], 2, C.bg || '#000'); V.dot(g, p[0] + 5, p[1], 2, C.bg || '#000');
      });
    })();

    /* =====================================================================
       FIG 5 — escada de precedência
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-precedencia');
      const LAYERS = ['Ambiente (linha de comando)', 'source .env', 'Reaplica _CLI_ABLIT / _CLI_HF_TOKEN', 'Defaults ${VAR:-…} do start.sh', 'OVERRIDE_* (wrapper da lane)', 'Guardas finais'];
      const VARS = {
        kv: { label: 'KV_CACHE_DTYPE', t: ['KV_CACHE_DTYPE=fp8 na linha de comando', '.env define KV_CACHE_DTYPE=auto', 'rodar via start-v030.sh', 'exportar OVERRIDE_KV_CACHE_DTYPE=fp8'], def: [true, true, false, false] },
        ablit: { label: 'ABLIT', t: ['ABLIT=1 na linha de comando', '.env define ABLIT=0', 'rodar via start-fp8.sh', 'FP8_DENSE=true no .env'], def: [true, true, false, false] },
        model: { label: 'MODEL_ID', t: ['MODEL_ID=RadixArk/… na linha de comando', '.env define MODEL_ID=nvidia/…', 'rodar via start-fp8.sh', 'ABLIT=1 na linha de comando'], def: [true, true, false, false] },
        yarn: { label: 'YARN_ENABLE', t: ['YARN_ENABLE=true na linha de comando', '.env define YARN_ENABLE=false', 'rodar via start-fp8.sh', '.env define MAX_MODEL_LEN=1000000'], def: [true, false, false, false] },
      };
      const S = { v: 'kv', tg: [true, true, false, false], t0: null };
      function simulate() {
        const T = S.tg, out = [];
        const put = (val, changed, note) => out.push({ val, changed, note });
        if (S.v === 'kv') {
          let v;
          if (T[0]) { v = 'fp8'; put(v, true, 'KV_CACHE_DTYPE=fp8 ./start.sh'); } else put(v, false, 'não definido');
          if (T[1]) { put('auto', true, v ? 'sobrescreve a linha de comando!' : '.env define a chave'); v = 'auto'; } else put(v, false, '.env não define a chave');
          put(v, false, 'só vale para ABLIT e HF_TOKEN');
          if (!v) { v = 'fp8'; put(v, true, 'fallback ${KV_CACHE_DTYPE:-fp8}'); } else put(v, false, 'já tem valor: fallback não age');
          if (T[3]) { v = 'fp8'; put(v, true, 'OVERRIDE_KV_CACHE_DTYPE=fp8'); } else if (T[2]) { v = 'auto'; put(v, true, 'start-v030.sh: OVERRIDE_KV_CACHE_DTYPE=${…:-auto}'); } else put(v, false, 'sem wrapper');
          put(v, false, 'sem guarda para esta chave');
          const lane030 = T[2];
          return { rows: out, res: '--kv-cache-dtype ' + v, note: v.startsWith('fp8') ? (lane030 ? 'monta o backport patch_qsa_fp8_kv_v030.py (vllm#55557)' : 'monta patch_qsa_fp8_kv.py nos dois nós') : 'KV em BF16: sem patch de KV' };
        }
        if (S.v === 'ablit') {
          let v;
          if (T[0]) { v = '1'; put(v, true, 'ABLIT=1 ./start.sh → _CLI_ABLIT=1'); } else put(v, false, 'não definido');
          if (T[1]) { put('0', true, v ? '.env apaga o 1… por enquanto' : '.env define ABLIT=0'); v = '0'; } else put(v, false, '.env não define');
          if (T[0]) { const ch = v !== '1'; v = '1'; put(v, ch, 'reaplica _CLI_ABLIT: a linha de comando vence'); } else put(v, false, 'nada capturado');
          if (!v) { v = '0'; put(v, true, 'fallback ${ABLIT:-0}'); } else put(v, false, 'já tem valor');
          if (T[2]) put('ABLIT=' + v, true, 'OVERRIDE_MODEL_ID=Qwen/…-FP8'); else put('ABLIT=' + v, false, 'sem wrapper');
          let model, note;
          if (T[3]) { model = 'MiaAI-Lab/…-FP8dense'; note = v === '1' ? 'ABLIT ignorado: FP8_DENSE=true vence' : 'FP8_DENSE=true escolhe o híbrido'; }
          else if (T[2]) { model = 'Qwen/Qwen3.8-Flash-Next-FP8'; note = v === '1' ? 'ABLIT ignorado: OVERRIDE_MODEL_ID vence' : 'override escolhe o FP8 oficial'; }
          else if (v === '1') { model = 'drowzeys/…-ablit-house-qsa-L3-47'; note = 'ABLIT=1 escolhe o checkpoint abliterado'; }
          else { model = 'MODEL_ID do .env (nvidia/…)'; note = 'ABLIT=0: nada muda'; }
          put(model, true, note);
          return { rows: out, res: 'ABLIT=' + v + ' → ' + model, note };
        }
        if (S.v === 'model') {
          let v;
          if (T[0]) { v = 'RadixArk/…'; put(v, true, 'MODEL_ID=RadixArk/… ./start.sh'); } else put(v, false, 'não definido');
          if (T[1]) { put('nvidia/…', true, v ? 'sobrescreve a linha de comando!' : '.env define a chave'); v = 'nvidia/…'; } else put(v, false, '.env não define');
          put(v, false, T[3] ? 'ABLIT=1 reaplicado (vale na guarda)' : 'só ABLIT e HF_TOKEN');
          if (!v) { put('ERRO', true, 'Required variable MODEL_ID is not set in .env'); return { rows: out, res: 'launch abortado', note: 'MODEL_ID é obrigatório (validado logo após o source .env)', err: true }; }
          put(v, false, 'sem fallback para MODEL_ID');
          if (T[2]) { v = 'Qwen/…-FP8'; put(v, true, 'OVERRIDE_MODEL_ID (start-fp8.sh)'); } else put(v, false, 'sem wrapper');
          if (T[3] && !T[2]) { v = 'drowzeys/…-ablit'; put(v, true, 'ABLIT=1 escolhe o abliterado'); }
          else if (T[3] && T[2]) put(v, false, 'ABLIT ignorado: OVERRIDE_MODEL_ID vence');
          else put(v, false, 'FP8_DENSE > OVERRIDE > ABLIT > MODEL_ID');
          return { rows: out, res: 'serve ' + v, note: 'ordem: FP8_DENSE=true > OVERRIDE_MODEL_ID > ABLIT=1 > MODEL_ID' };
        }
        // yarn
        let v;
        if (T[0]) { v = 'true'; put(v, true, 'YARN_ENABLE=true ./start.sh'); } else put(v, false, 'não definido');
        if (T[1]) { put('false', true, v ? 'sobrescreve a linha de comando!' : '.env define a chave'); v = 'false'; } else put(v, false, '.env não define');
        put(v, false, 'só ABLIT e HF_TOKEN');
        if (!v) { v = 'false'; put(v, true, 'fallback ${YARN_ENABLE:-false}'); } else put(v, false, 'já tem valor');
        if (T[2]) { v = 'false'; put(v, true, 'OVERRIDE_YARN_ENABLE=false + MAX_MODEL_LEN=262144'); } else put(v, false, 'sem wrapper');
        const mml = T[2] ? 262144 : T[3] ? 1000000 : 262144;
        if (v === 'true' && mml <= 262144) { v = 'false'; put(v, true, 'NOTE: MAX_MODEL_LEN ≤ 262144 → YaRN force-disabled'); }
        else put(v, false, 'MAX_MODEL_LEN=' + nf(mml));
        return { rows: out, res: 'YARN_ENABLE=' + v + ' (MAX_MODEL_LEN ' + nf(mml) + ')', note: v === 'true' ? 'emite rope_parameters yarn, factor 4.0, aninhado em text_config' : 'RoPE nativo: nenhum rope_parameters no --hf-overrides' };
      }
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Variável', options: Object.entries(VARS).map(([k, o]) => ({ v: k, label: o.label })), value: 'kv', onChange: (v) => { S.v = v; VARS[v].def.forEach((d, i) => { S.tg[i] = d; togs[i].el.querySelector('input').checked = d; togs[i].el.querySelector('span').textContent = VARS[v].t[i]; }); S.t0 = null; } });
      const togs = [0, 1, 2, 3].map((i) => UI.toggle(ctl, { label: VARS.kv.t[i], value: S.tg[i], onChange: (on) => { S.tg[i] = on; S.t0 = null; } }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.t0 == null) S.t0 = t;
        const el = t - S.t0;
        const sim = simulate();
        const narrow = w < 560, m = 14;
        const railX = m + 14, top = 16, resH = narrow ? 64 : 56;
        const rh = (h - top - resH - 16) / 6;
        const stepDur = 0.55, n = sim.rows.length;
        const prog = Math.min(n - 1 + 0.999, el / stepDur);
        const cur = Math.floor(prog);
        const done = el > n * stepDur + 0.2;
        if (el > n * stepDur + 3.2) S.t0 = t;
        V.line(g, railX, top + rh / 2, railX, top + rh * 5.5, { color: V.alpha(C.muted, 0.4), lw: 2 });
        LAYERS.forEach((lab, i) => {
          const y = top + i * rh;
          const r = sim.rows[i];
          const reached = i <= cur || done;
          const active = reached && r && r.changed;
          V.box(g, railX + 16, y + 4, w - railX - 16 - m, rh - 8, { r: 9, fill: active ? V.alpha(C.a, 0.13) : V.alpha(C.muted, 0.05), stroke: active ? V.alpha(C.a, 0.7) : V.alpha(C.muted, 0.25) });
          V.dot(g, railX, y + rh / 2, 9, active ? C.a : reached ? V.alpha(C.muted, 0.6) : V.alpha(C.muted, 0.25));
          V.text(g, String(i + 1), railX, y + rh / 2 + 4, { size: 11, weight: 700, align: 'center', color: active ? (C.bg || '#000') : C.ink });
          const lx = railX + 28;
          if (!r) return;
          const valTxt = !reached ? '·' : r.val == null ? '—' : String(r.val);
          if (narrow) {
            V.text(g, fitText(g, lab, w - lx - m - 8, { size: 11, weight: 600 }), lx, y + 18, { size: 11, weight: 600, color: C.ink });
            V.text(g, fitText(g, valTxt + (reached ? '  · ' + r.note : ''), w - lx - m - 8, { size: 11, font: 'mono' }), lx, y + 34, { size: 11, font: 'mono', color: active ? C.ink : C.muted });
          } else {
            V.text(g, lab, lx, y + rh / 2 - 3, { size: 12, weight: 600, color: C.ink });
            if (reached) V.text(g, fitText(g, r.note, w * 0.5 - 30, { size: 11 }), lx, y + rh / 2 + 13, { size: 11, color: C.muted });
            const chip = fitText(g, valTxt, w * 0.38, { size: 12, font: 'mono', weight: 600 });
            const cw = tw(g, chip, { size: 12, font: 'mono', weight: 600 }) + 20;
            V.box(g, w - m - 10 - cw, y + rh / 2 - 12, cw, 24, { r: 12, fill: active ? V.alpha(C.a, 0.3) : V.alpha(C.muted, 0.12), stroke: active ? C.a : V.alpha(C.muted, 0.3) });
            V.text(g, chip, w - m - 10 - cw / 2, y + rh / 2 + 4, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink });
          }
        });
        // bolinha
        if (!done) {
          const f = prog - cur;
          const y = top + (cur + V.ease.inOut(Math.min(1, f * 1.6))) * rh + rh / 2;
          V.glow(g, railX, Math.min(y, top + 5.5 * rh), 18, C.d);
          V.dot(g, railX, Math.min(y, top + 5.5 * rh), 5, C.d);
        }
        // resultado
        const ry = h - resH - 6;
        const rc = sim.err ? C.red : C.qwen;
        V.box(g, m, ry, w - 2 * m, resH, { r: 12, fill: V.alpha(rc, done ? 0.16 : 0.06), stroke: V.alpha(rc, done ? 0.8 : 0.3), lw: done ? 2 : 1 });
        V.text(g, fitText(g, 'Efetivo: ' + (done ? sim.res : '…'), w - 2 * m - 24, { size: 12.5, font: 'mono', weight: 700 }), m + 12, ry + 22, { size: 12.5, font: 'mono', weight: 700, color: done ? C.ink : C.muted });
        if (done) wrapText(g, sim.note, m + 12, ry + 40, w - 2 * m - 24, 14, { size: 11, color: C.muted });
      });
    })();

    /* =====================================================================
       FIG 6 — seis checkpoints
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-checkpoints');
      const COMP = [
        { k: 'exp', l: 'Experts roteados', p: 120.80 },
        { k: 'ple', l: 'Tabela N-gram (PLE)', ls: 'N-gram (PLE)', p: 51.20 },
        { k: 'mtp', l: 'MTP (experts)', p: 2.517 },
        { k: 'gdn', l: 'GDN (36 camadas)', p: 2.087 },
        { k: 'att', l: 'Atenção QSA + indexador', ls: 'Atenção QSA', p: 0.618 },
        { k: 'hc', l: 'Gated Residual (HC)', ls: 'Gated Residual', p: 0.641 },
        { k: 'lm', l: 'lm_head', p: 0.636 },
        { k: 'emb', l: 'embed_tokens', p: 0.636 },
        { k: 'sr', l: 'Shared experts + roteadores', ls: 'Shared + roteador', p: 0.299 },
        { k: 'vis', l: 'Visão (ViT)', p: 0.449 },
        { k: 'etc', l: 'Resto (MTP denso, PLE proj.)', ls: 'Resto', p: 0.124 },
      ];
      const BPP = { BF16: 2, FP8: 1, FP8B: 1, NVFP4: 0.5625, MXFP8: 1.03125 };
      const base = { exp: 'NVFP4', ple: 'FP8', mtp: 'FP8B', gdn: 'BF16', att: 'BF16', hc: 'BF16', lm: 'BF16', emb: 'BF16', sr: 'BF16', vis: 'BF16', etc: 'BF16' };
      const CK = {
        nvidia: { name: 'nvidia', fmt: Object.assign({}, base), unk: [], pub: '133 GB · 11 shards', html: 'Checkpoint padrão: <b>experts NVFP4 (W4A4</b>: pesos <i>e</i> ativações dos experts em 4 bits), PLE FP8 por tensor, experts da MTP em FP8 blocos 128×128, o resto BF16. <b>Acorda:</b> <code>detect_ple_dtype.py</code> → <code>float8_e4m3fn</code>, patch da PLE, alias <code>mtp.layers.48</code> + normalização <code>FP8_PB_WO</code>, ramo FP8 em blocos para a MoE da MTP; o fallback MXFP8 é montado mas provavelmente inerte (densos BF16).' },
        lil: { name: 'local-inference-lab', fmt: Object.assign({}, base, { ple: 'NVFP4', gdn: 'MXFP8', att: 'MXFP8', vis: 'MXFP8' }), unk: ['mtp', 'vis'], pub: '98,6 GiB · 37 shards', html: 'Atenção/GDN em <b>MXFP8</b> e PLE em <b>NVFP4</b> (26,8 GiB). É o único que cabe num Spark (lane TP1). <b>Acorda:</b> o fallback MXFP8 vira essencial (<code>in_proj_a/b [48, 2560]</code> caem para emulação); a PLE já declara o dtype, então o detector não injeta nada. Não detalhados nas fontes: os experts da MTP (a estimativa usa FP8) e a visão (a estimativa usa MXFP8, porque o patch manda <code>visual.*</code> MXFP8 para emulação).' },
        radix: { name: 'RadixArk', fmt: Object.assign({}, base, { mtp: 'BF16' }), unk: [], pub: '125,9 GiB · 206 shards', html: 'Atenção BF16, PLE FP8 já declarada e <b>experts da MTP em BF16</b> (4,69 GiB). Decode 12–16% mais lento que o local-inference-lab (mais bytes), prefill igual; GSM8K 97,27% e AIME26 98,75%. É a origem do FP8-dense.' },
        fp8off: { name: 'FP8 oficial', fmt: Object.assign({}, base, { exp: 'FP8B', mtp: 'FP8B' }), unk: [], pub: '— (não citado)', html: '<code>Qwen/Qwen3.8-Flash-Next-FP8</code>: experts roteados (112,5 GiB) e da MTP em FP8 blocos de 128 = <b>o dobro dos bytes</b> dos NVFP4, então sobra KV para ~500 mil tokens. Densos em <b>BF16</b>: o scan dos cabeçalhos (relatório de desempenho) mostra <code>modules_to_not_convert</code> com <code>linear_attn.*</code>, <code>hyper_connection*</code> e <code>lm_head</code>. O “block-128 over the whole checkpoint” do <code>start-fp8.sh</code> é o <code>Fp8Config</code> global. <b>Acorda:</b> <code>SKIP_PLE_PATCH=true</code>, contexto nativo, YaRN off.' },
        dense: { name: 'FP8-dense', fmt: Object.assign({}, base, { mtp: 'BF16', gdn: 'FP8', att: 'FP8', hc: 'FP8', lm: 'FP8', sr: 'FP8' }), unk: [], pub: 'local (4 shards BF16 reescritos, 11,8 GB)', html: 'Híbrido construído a partir do RadixArk: experts NVFP4 e PLE <i>hard-linked</i>, 591 lineares densos em FP8 E4M3 por canal + ativação dinâmica por token. Esperado 9,2 → 5,7 GiB/passo (<b>analítico</b>). <b>Acorda:</b> <code>FP8_DENSE=true</code> + 4 overlays (<code>modelopt</code>, <code>model</code>, <code>hyperconnection</code>, <code>mtp</code>) — conflita com o vocabulário de rascunho e o gerador <code>apply_patches.py</code> está ausente.' },
        ablit: { name: 'ABLIT', fmt: Object.assign({}, base), unk: [], pub: 'mesmo layout do nvidia', html: 'Mesmo layout do nvidia; só os 12 <code>self_attn.o_proj</code> das camadas QSA 3, 7, …, 47 foram editados (★). <b>Acorda:</b> <code>ABLIT=1</code> + <code>HF_TOKEN</code> + termos aceitos; os mesmos patches do nvidia. Válido só nesta receita.' },
      };
      const S = { ck: 'nvidia', anim: {} };
      COMP.forEach((c) => { S.anim[c.k] = 0; });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Checkpoint', options: Object.entries(CK).map(([k, o]) => ({ v: k, label: o.name })), value: 'nvidia', onChange: (v) => { S.ck = v; render(); } });
      const panel = LA.h('div', { style: { padding: '2px 18px 12px', fontSize: '14px', color: 'var(--text-2)', overflowWrap: 'anywhere' } });
      beforeCap(fig, panel);
      function render() { const c = CK[S.ck]; panel.innerHTML = `<p style="margin:.3em 0"><b>${esc(c.name)}</b> — ${c.html}</p>`; }
      render();
      const GiB = 1073741824;
      const sizeOf = (ck, c) => (c.p * 1e9 * BPP[ck.fmt[c.k]]) / GiB;
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const DT = { BF16: C.b, FP8: C.a, FP8B: C.d, NVFP4: C.c, MXFP8: C.f };
        // MXFP8 = magenta com listras verticais (no tema claro, magenta e rosa quase se confundem)
        const stripes = (x, y, ww, hh, r) => { g.save(); V.rr(g, x, y, ww, hh, r); g.clip(); g.fillStyle = V.alpha(C.bg, 0.75); for (let xx = x + 3; xx < x + ww; xx += 6) g.fillRect(xx, y, 2, hh); g.restore(); g.strokeStyle = C.f; g.lineWidth = 1.4; V.rr(g, x + 0.5, y + 0.5, Math.max(0, ww - 1), hh - 1, r); g.stroke(); };
        const ck = CK[S.ck];
        const narrow = w < 560, m = 14;
        const lw = narrow ? 112 : 196, vw = narrow ? 50 : 62;
        const bx0 = m + lw, bwMax = w - m - vw - bx0;
        const top = 46, bot = narrow ? 62 : 44;
        const rh = (h - top - bot) / COMP.length;
        const maxG = 112.6;
        let total = 0;
        COMP.forEach((c) => { total += sizeOf(ck, c); });
        V.text(g, 'estimado: ' + nf(total, 1) + ' GiB', m, 20, { size: 13, weight: 700, color: C.ink });
        V.text(g, fitText(g, 'publicado: ' + ck.pub, w - 2 * m - 150, { size: 11.5 }), w - m, 20, { size: 11.5, align: 'right', color: C.muted });
        V.text(g, 'barra ∝ GiB (escala linear até 112,6 GiB)', m, 36, { size: 11, color: C.muted });
        COMP.forEach((c, i) => {
          const y = top + i * rh;
          const gib = sizeOf(ck, c);
          const target = Math.max(3, (gib / maxG) * bwMax);
          S.anim[c.k] = S.anim[c.k] ? V.lerp(S.anim[c.k], target, 0.18) : target;
          const bwc = S.anim[c.k];
          const f = ck.fmt[c.k], colr = DT[f];
          V.text(g, fitText(g, narrow && c.ls ? c.ls : c.l, lw - 8, { size: 11.5 }), m, y + rh / 2 + 4, { size: 11.5, color: C.ink });
          const bh = Math.min(18, rh - 5), by = y + (rh - bh) / 2;
          V.box(g, bx0, by, bwc, bh, { r: 4, fill: V.alpha(colr, 0.7) });
          if (f === 'MXFP8') stripes(bx0, by, bwc, bh, 4);
          if (ck.unk.includes(c.k)) {
            g.save(); V.rr(g, bx0, by, bwc, bh, 4); g.clip();
            g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1.2;
            for (let xx = bx0 - bh; xx < bx0 + bwc; xx += 6) { g.beginPath(); g.moveTo(xx, by + bh); g.lineTo(xx + bh, by); g.stroke(); }
            g.restore();
          }
          const tag = f === 'FP8B' ? 'FP8 bloco' : f;
          const tagW = tw(g, tag, { size: 11, font: 'mono' });
          if (bwc > tagW + 10) V.text(g, tag, bx0 + 5, by + bh / 2 + 3.5, { size: 11, font: 'mono', color: C.ink });
          else if (bwc + tagW + 8 < bwMax) V.text(g, tag, bx0 + bwc + 5, by + bh / 2 + 3.5, { size: 11, font: 'mono', color: C.muted });
          if (S.ck === 'ablit' && c.k === 'att') V.text(g, '★', bx0 + bwc + (bwc + tagW + 8 < bwMax ? tagW + 12 : 6), by + bh / 2 + 4, { size: 12, color: C.d });
          V.text(g, nf(gib, gib < 10 ? 2 : 1), w - m, y + rh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.ink });
        });
        // legenda
        const items = [['BF16', C.b], ['FP8', C.a], ['FP8 bloco', C.d], ['NVFP4', C.c], ['MXFP8', C.f]];
        items.push(['não detalhado', null]);
        let lx = m, ly = h - bot + 22;
        items.forEach(([l, c]) => {
          const iw = tw(g, l, { size: 11 }) + 30;
          if (lx + iw > w - m) { lx = m; ly += 18; }
          if (c) { V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: V.alpha(c, 0.75) }); if (l === 'MXFP8') stripes(lx, ly - 9, 11, 11, 3); }
          else { V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: V.alpha(C.muted, 0.2), stroke: C.muted }); g.save(); V.rr(g, lx, ly - 9, 11, 11, 3); g.clip(); g.strokeStyle = V.alpha(C.ink, 0.6); g.beginPath(); for (let d = -11; d < 11; d += 4) { g.moveTo(lx + d, ly + 2); g.lineTo(lx + d + 11, ly - 9); } g.stroke(); g.restore(); }
          V.text(g, l, lx + 15, ly, { size: 11, color: c ? C.ink : C.muted });
          lx += iw;
        });
      });
    })();

    /* =====================================================================
       FIG 7 — stepper: alias da MTP e escalas por bloco
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-mtp-alias');
      const S = { wrong: false };
      const steps = [
        { t: 'O que o checkpoint declara', d: 'O <code>config.json</code> do nvidia anota a quantização da MoE da MTP com índice <b>relativo</b>: <code>mtp.layers.0.mlp.experts</code>, algoritmo <code>FP8_PB_WO</code>, <code>group_size</code> 128. O vLLM, porém, numera a camada MTP continuando a pilha principal: \\( 48 + i \\).' },
        { t: 'Busca por string exata', d: 'O despacho ModelOpt procura <code>"mtp.layers.48.mlp.experts"</code> no dicionário. Não acha, devolve <code>None</code> — e a MoE é construída <b>sem quantização</b>, sem nenhum erro.' },
        { t: '~7 minutos depois', d: 'Os pesos do alvo já estão na memória quando o loader chega ao rascunhador e procura as escalas FP8 que a camada “não quantizada” não tem: <code>has no parameter \'w2_weight_scale_inv\'</code>. Crash.' },
        { t: 'O conserto no config', d: '<code>patch_checkpoint_config.py</code> adiciona o alias <code>.48.</code> em <code>config.json</code> e <code>hf_quant_config.json</code> (montados por cima do snapshot) e reescreve <code>FP8_PB_WO</code> (group 128) como <code>FP8_BLOCK_SCALES</code>. O preflight <code>--mtp-moe-algo</code> checa em segundos.' },
        { t: 'Escalas por bloco 128×128', d: 'O ramo novo monta <code>Fp8Config(weight_block_size=[128,128])</code>: cada bloco tem sua escala \\( s_b = \\max|W_b| / 448 \\) e \\( \\hat w = q \\cdot s_b \\). Ligue “bloco errado” para ver escalas certas aplicadas às regiões erradas.' },
        { t: 'Prova de saúde: aceitação por posição', d: 'Com as escalas certas a aceitação decresce suavemente — 89% / 74,5% / 60% (medido). Uma curva baixa e plana, quase aleatória, denunciaria pesos do rascunhador lidos errado mesmo sem crash.' },
      ];
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'simular bloco errado (64 em vez de 128)', value: false, onChange: (v) => { S.wrong = v; } });
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      // matriz ilustrativa: 8×8 células (cada uma = 32×32 pesos), blocos de 4×4 células (= 128×128)
      const rng = V.rng(43);
      const MAG = [[0.02, 0.35], [0.9, 0.08]];
      const N = 8, B = 4;
      const W = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => rng.normal() * MAG[Math.floor(i / B)][Math.floor(j / B)]));
      const sc = [[0, 0], [0, 0]];
      for (let bi = 0; bi < 2; bi++) for (let bj = 0; bj < 2; bj++) { let mx = 0; for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) mx = Math.max(mx, Math.abs(W[bi * B + i][bj * B + j])); sc[bi][bj] = mx / 448; }
      const Q = W.map((r, i) => r.map((x, j) => LA.num.encode('e4m3', x / sc[Math.floor(i / B)][Math.floor(j / B)]).v));
      const flat = [sc[0][0], sc[0][1], sc[1][0], sc[1][1]];
      const good = Q.map((r, i) => r.map((q, j) => q * sc[Math.floor(i / B)][Math.floor(j / B)]));
      const bad = Q.map((r, i) => r.map((q, j) => q * flat[(Math.floor(i / 2) * 4 + Math.floor(j / 2)) % 4]));
      let amax = 0; W.forEach((r) => r.forEach((x) => { amax = Math.max(amax, Math.abs(x)); }));
      const relErr = (M) => { let s = 0; M.forEach((r, i) => r.forEach((x, j) => { s += Math.abs(x - W[i][j]) / amax; })); return (100 * s) / (N * N); };
      const errGood = relErr(good), errBad = relErr(bad);
      V.fig(fig, (g, w, h) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 600, m = 14;
        const lh = narrow ? 16 : 20, fsz = narrow ? 11 : 12.5;
        const segLine = (x, y, parts) => { let xx = x; parts.forEach((p) => { const o = { size: fsz, font: 'mono', color: p.c || C.ink, weight: p.b ? 700 : 400 }; V.text(g, p.s, xx, y, o); xx += tw(g, p.s, o); }); };
        const card = (x, y, cw, title, rows, colr) => {
          const chh = 34 + rows.length * lh + 8;
          V.box(g, x, y, cw, chh, { r: 12, fill: V.alpha(colr, 0.08), stroke: V.alpha(colr, 0.55) });
          V.text(g, title, x + 12, y + 20, { size: 11, font: 'mono', weight: 700, color: colr });
          rows.forEach((parts, k) => segLine(x + 12, y + 40 + k * lh, parts));
          return chh;
        };
        const resultBox = (y, txt, col) => {
          const o = { size: narrow ? 11.5 : 12.5, weight: 600, color: C.ink };
          const hh = countLines(g, txt, w - 2 * m - 24, o) * 16 + 14;
          V.box(g, m, y, w - 2 * m, hh, { r: 10, fill: V.alpha(col, 0.13), stroke: V.alpha(col, 0.65) });
          wrapText(g, txt, m + 12, y + 19, w - 2 * m - 24, 16, o);
          return hh;
        };
        if (i <= 3) {
          const gapC = narrow ? 34 : 64;
          const cw = narrow ? w - 2 * m : (w - 2 * m - gapC) / 2;
          const miss = i === 1 || i === 2;
          const L = [
            [{ s: 'quantized_layers:', c: C.muted }],
            [{ s: '"mtp.layers.' }, { s: '0', c: miss ? C.red : C.d, b: true }, { s: '.mlp.experts": {' }],
            [{ s: '  "quant_algo": ' }, { s: '"FP8_PB_WO"', c: i === 3 ? C.muted : C.ink }, { s: ',' }],
            [{ s: '  "group_size": 128 }' }],
          ];
          if (i === 3) {
            L.push([{ s: '"mtp.layers.', c: C.e }, { s: '48', c: C.e, b: true }, { s: '.mlp.experts": {', c: C.e }]);
            L.push([{ s: '  "quant_algo": "FP8_BLOCK_SCALES" }', c: C.e }]);
          }
          const R = [
            [{ s: 'num_hidden_layers = 48', c: C.muted }],
            [{ s: 'MTP i = 0  →  índice 48 + 0' }],
            [{ s: '"mtp.layers.' }, { s: '48', c: miss ? C.red : i === 3 ? C.e : C.b, b: true }, { s: '.mlp.experts"' }],
          ];
          const lx = m, ly = 12;
          const hL = card(lx, ly, cw, 'CHECKPOINT · config.json', L, C.d);
          const rx = narrow ? m : m + cw + gapC, ry = narrow ? ly + hL + gapC : ly;
          const hR = card(rx, ry, cw, 'vLLM · nome instanciado', R, C.b);
          if (i >= 1) {
            const yL = ly + 40 + (i === 3 ? 4 : 1) * lh - 4, yR = ry + 40 + 2 * lh - 4;
            const a = narrow ? [rx + cw * 0.5, ry - 2] : [rx - 2, yR];
            const b = narrow ? [lx + cw * 0.5, ly + hL + 2] : [lx + cw + 2, yL];
            const u = Math.min(1, el / 1.1);
            const col = i === 3 ? C.e : C.red;
            V.arrow(g, a[0], a[1], V.lerp(a[0], b[0], u), V.lerp(a[1], b[1], u), { color: col, lw: 2.4, dash: [5, 4] });
            if (u >= 1) { const mx = (a[0] + b[0]) / 2 + (narrow ? 26 : 0), my = (a[1] + b[1]) / 2; V.box(g, mx - 13, my - 12, 26, 24, { r: 12, fill: V.alpha(col, 0.3), stroke: col, label: i === 3 ? '✓' : '✕', size: 13, weight: 700 }); }
          }
          const resY = Math.max(ly + hL, ry + hR) + (narrow ? 16 : 22);
          // pipeline do despacho de quantização
          const pipe = narrow ? [['config', 0], ['lookup', 1], ['método', 1], [i === 3 ? 'Fp8MoE' : 'sem escala', 2]] : [['config.json', 0], ['quantized_layers[nome]', 1], ['get_quant_method', 1], [i === 3 ? 'Fp8MoEMethod' : 'MoE sem escalas', 2]];
          const py = Math.min(h - 18, resY + 92);
          if (py > resY + 70) {
            const pw = (w - 2 * m - 3 * 22) / 4;
            pipe.forEach(([lab, kind], k) => {
              const x = m + k * (pw + 22);
              const bad = (i === 1 || i === 2) && k >= 1, good = i === 3 && k >= 1;
              const col = bad ? C.red : good ? C.e : C.b;
              const lit = i >= 1 ? Math.min(1, Math.max(0, el * 1.6 - k * 0.5)) : 0;
              V.box(g, x, py - 13, pw, 26, { r: 13, fill: V.alpha(col, 0.08 + 0.22 * lit), stroke: V.alpha(col, 0.35 + 0.5 * lit) });
              V.text(g, fitText(g, lab, pw - 12, { size: 11, font: 'mono' }), x + pw / 2, py + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
              if (k < 3) V.arrow(g, x + pw + 3, py, x + pw + 19, py, { color: V.alpha(C.muted, 0.8), lw: 1.5, head: 6 });
            });
          }
          if (i === 0) resultBox(resY, 'O vLLM vai procurar exatamente o nome da direita no dicionário da esquerda — por string, caractere a caractere.', C.b);
          else if (i === 1) resultBox(resY, '✕ “0” ≠ “48” → get_quant_method(…) devolve None → a MoE da MTP é montada SEM quantização. Nenhum erro, nenhum aviso.', C.red);
          else if (i === 3) resultBox(resY, '✓ Os nomes casam → Fp8Config(weight_block_size=[128,128]) → Fp8MoEMethod. E o preflight --mtp-moe-algo teria abortado em segundos se o algoritmo não fosse construível.', C.e);
          else {
            const bw = w - 2 * m, byy = resY + 18;
            V.text(g, 'carga dos pesos (alvo → rascunhador)', m, byy - 6, { size: 11, font: 'mono', color: C.muted });
            V.box(g, m, byy, bw, 12, { r: 6, fill: V.alpha(C.muted, 0.15) });
            const u = Math.min(1, el / 2.6);
            V.box(g, m, byy, Math.max(6, bw * u * 0.72), 12, { r: 6, fill: V.alpha(C.b, 0.7) });
            V.text(g, nf(u * 7, 1) + ' min', m + Math.max(6, bw * u * 0.72) + 6, byy + 10, { size: 11, font: 'mono', color: C.ink });
            if (u >= 1) {
              const ex = m + bw * 0.72, pulse = 0.6 + 0.4 * Math.sin(el * 8);
              V.glow(g, ex, byy + 6, 34 * pulse, C.red);
              wrapText(g, "✕ Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'", m, byy + 34, w - 2 * m, 15, { size: 11, font: 'mono', weight: 700, color: C.red });
            }
          }
          return;
        }
        if (i === 4) {
          const M = S.wrong ? bad : good;
          const gapM = narrow ? 18 : 40;
          const cell = Math.max(10, Math.min(30, (w - 2 * m - gapM - 20) / (2 * N + 0.4), (h - 130) / N));
          const mw = N * (cell + 2);
          const x0 = (w - (2 * mw + gapM)) / 2, y0 = 44;
          const lim = amax;
          // escala de cor em √|v| (sinal preservado) + contorno por célula: os blocos de magnitude pequena continuam visíveis
          const cf = (v) => { const t = Math.max(-1, Math.min(1, v / lim)); return V.cmap(Math.sign(t) * Math.sqrt(Math.abs(t)), 'div'); };
          const cellStroke = V.alpha(C.muted, C.dark ? 0.5 : 0.4);
          V.text(g, 'W original', x0 + mw / 2, y0 - 12, { size: 11.5, align: 'center', font: 'mono', color: C.muted });
          V.text(g, S.wrong ? 'Ŵ com bloco errado' : 'Ŵ = q · s_bloco', x0 + mw + gapM + mw / 2, y0 - 12, { size: 11.5, align: 'center', font: 'mono', color: S.wrong ? C.red : C.e });
          V.matrix(g, { x: x0, y: y0, data: W, cell, gap: 2, colorFn: (v) => cf(v), stroke: cellStroke });
          V.matrix(g, { x: x0 + mw + gapM, y: y0, data: M, cell, gap: 2, colorFn: (v) => cf(v), stroke: cellStroke });
          // contornos dos blocos
          const outline = (ox, bsz, col) => { for (let bi = 0; bi < N / bsz; bi++) for (let bj = 0; bj < N / bsz; bj++) { g.strokeStyle = col; g.lineWidth = 2; V.rr(g, ox + bj * bsz * (cell + 2) - 2, y0 + bi * bsz * (cell + 2) - 2, bsz * (cell + 2) + 2, bsz * (cell + 2) + 2, 5); g.stroke(); } };
          outline(x0, B, V.alpha(C.ink, 0.8));
          outline(x0 + mw + gapM, S.wrong ? 2 : B, S.wrong ? V.alpha(C.red, 0.85) : V.alpha(C.e, 0.9));
          const inCell = cell >= 22;
          if (inCell) for (let bi = 0; bi < 2; bi++) for (let bj = 0; bj < 2; bj++) V.text(g, 's=' + sc[bi][bj].toExponential(1), x0 + (bj * B + B / 2) * (cell + 2), y0 + (bi * B + B) * (cell + 2) - 6, { size: 11, font: 'mono', align: 'center', color: C.ink });
          let yy = y0 + N * (cell + 2) + 24;
          if (!inCell) {
            // no mobile as escalas não cabem nas células: lista abaixo das matrizes
            V.text(g, 'uma escala por bloco (s = max|W_b| / 448):', m, yy, { size: 11, font: 'mono', color: C.muted });
            yy += 18;
            for (let bi = 0; bi < 2; bi++) {
              V.text(g, 'bloco (' + bi + ',0): s=' + sc[bi][0].toExponential(1) + '   bloco (' + bi + ',1): s=' + sc[bi][1].toExponential(1), m, yy, { size: 11, font: 'mono', color: C.ink, maxW: w - 2 * m });
              yy += 16;
            }
            yy += 14;
          }
          const e = S.wrong ? errBad : errGood;
          V.text(g, 'erro médio |Ŵ − W| / max|W| = ' + nf(e, e < 1 ? 2 : 1) + '%', w / 2, yy, { size: narrow ? 12 : 13, weight: 700, align: 'center', color: S.wrong ? C.red : C.e, maxW: w - 2 * m });
          const leg = S.wrong ? 'ilustrativo: cada escala cai sobre a região errada — e nada avisa' : 'cada célula = 32×32 pesos · cada bloco contornado = 128×128 com uma escala FP8 · cor em escala √ para os blocos pequenos aparecerem';
          wrapText(g, leg, w / 2, yy + 20, w - 2 * m, 15, { size: 11, align: 'center', color: C.muted });
          return;
        }
        // i === 5: aceitação
        const vals = S.wrong ? [9, 6, 4] : [89, 74.5, 60], meas = [89, 74.5, 60];
        const grow = Math.min(1, el / 1.4);
        const bw = Math.min(90, (w - 2 * m) / 5), gx = (w - 3 * bw - 2 * bw * 0.5) / 2, base = h - 60, hmax = h - 130;
        V.text(g, S.wrong ? 'aceitação por posição — bloco errado (ilustrativo)' : 'aceitação por posição — medida (MTP=3, batch 1)', w / 2, 26, { size: 12, weight: 600, align: 'center', color: S.wrong ? C.red : C.ink });
        vals.forEach((v, k) => {
          const x = gx + k * bw * 1.5, bh = (v / 100) * hmax * grow;
          if (S.wrong) { const mh = (meas[k] / 100) * hmax; g.setLineDash([4, 4]); V.box(g, x, base - mh, bw, mh, { r: 6, stroke: V.alpha(C.a, 0.6), lw: 1.5 }); g.setLineDash([]); V.text(g, meas[k] + '% medido', x + bw / 2, base - mh - 6, { size: 11, align: 'center', color: C.muted }); }
          V.box(g, x, base - bh, bw, Math.max(2, bh), { r: 6, fill: V.alpha(S.wrong ? C.red : C.a, 0.7) });
          V.text(g, nf(v * grow, v % 1 ? 1 : 0) + '%', x + bw / 2, base - bh - 8, { size: 13, weight: 700, align: 'center', color: C.ink });
          V.text(g, 'posição ' + (k + 1), x + bw / 2, base + 18, { size: 11, align: 'center', font: 'mono', color: C.muted });
        });
        V.line(g, m, base, w - m, base, { color: V.alpha(C.muted, 0.5) });
        V.text(g, S.wrong ? 'baixa e quase plana: suspeite dos pesos do rascunhador (tracejado = o medido)' : 'decresce suavemente: cada rascunho depende do anterior', w / 2, h - 16, { size: 11.5, align: 'center', color: C.muted, maxW: w - 2 * m });
      });
    })();

    /* =====================================================================
       FIG 8 — envelope do kernel
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-envelope');
      const PRE = {
        inproj: { N: 48, K: 2560, l: 'in_proj_a/b [48, 2560]' },
        visfc1: { N: 4304, K: 1152, l: 'visão fc1 [4304, 1152]' },
        qkvg: { N: 6656, K: 2560, l: 'qkv+gate TP2 [6656, 2560]' },
        visfc2: { N: 1152, K: 2152, l: 'visão fc2 c/ TP2 [1152, 2152]' },
      };
      const S = { N: 48, K: 2560, rule: 'mxfp8', pre: 'inproj', lock: false };
      const ctl = UI.controls(fig);
      const presetSeg = UI.seg(ctl, { label: 'Forma', options: Object.entries(PRE).map(([k, p]) => ({ v: k, label: p.l })), value: 'inproj', onChange: (v) => { S.pre = v; S.lock = true; sN.set(PRE[v].N); sK.set(PRE[v].K); S.lock = false; if (v === 'visfc2') ruleSeg.set('nvfp4'); else ruleSeg.set('mxfp8'); H.redraw(); } });
      const ruleSeg = UI.seg(ctl, { label: 'Regra', options: [{ v: 'mxfp8', label: 'MXFP8 mm_mxfp8' }, { v: 'nvfp4', label: 'NVFP4 (K % 16)' }], value: 'mxfp8', onChange: (v) => { S.rule = v; H.redraw(); } });
      const sN = UI.slider(ctl, { label: 'N (saída)', min: 16, max: 8192, step: 8, value: 48, fmt: (v) => nf(v), onInput: (v) => { S.N = v; if (!S.lock) S.pre = null; H.redraw(); } });
      const sK = UI.slider(ctl, { label: 'K (entrada)', min: 16, max: 8192, step: 8, value: 2560, fmt: (v) => nf(v), onInput: (v) => { S.K = v; if (!S.lock) S.pre = null; H.redraw(); } });
      void presetSeg;
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const N = S.N, K = S.K, mx = S.rule === 'mxfp8';
        const T = mx ? 32 : 16;
        const okNmin = N >= 128, okNmod = N % 32 === 0, okKmin = K >= 128, okKmod = K % (mx ? 32 : 16) === 0;
        const ok = mx ? okNmin && okNmod && okKmin && okKmod : okKmod;
        const m = 14, narrow = w < 560;
        const vcol = ok ? C.e : C.red;
        V.box(g, m, 10, w - 2 * m, 30, { r: 10, fill: V.alpha(vcol, 0.15), stroke: V.alpha(vcol, 0.7) });
        const verdict = narrow ? (mx ? (ok ? '✓ kernel nativo mm_mxfp8' : '✗ fora do envelope → emulação BF16') : (ok ? '✓ K múltiplo de 16' : '✗ K % 16 ≠ 0 → crash (use data)')) : mx ? (ok ? '✓ kernel nativo FlashInfer CUTLASS mm_mxfp8' : '✗ fora do envelope → EmulationMxfp8LinearKernel (BF16)') : (ok ? '✓ K múltiplo de 16: kernels NVFP4 aceitam' : '✗ K não é múltiplo de 16 → crash na carga (use data)');
        V.text(g, fitText(g, verdict, w - 2 * m - 20, { size: narrow ? 11.5 : 12.5, weight: 700 }), m + 10, 30, { size: narrow ? 11.5 : 12.5, weight: 700, color: C.ink });
        const chk = (okv, s) => (okv ? '✓ ' : '✗ ') + s;
        const items = mx ? [chk(okNmin, 'N ' + (okNmin ? '≥' : '<') + ' 128'), chk(okNmod, 'N mod 32 = ' + (N % 32)), chk(okKmin, 'K ' + (okKmin ? '≥' : '<') + ' 128'), chk(okKmod, 'K mod 32 = ' + (K % 32))] : [chk(okKmod, 'K mod 16 = ' + (K % 16)), '· N livre'];
        let ix = m, iy = 60;
        items.forEach((s) => { const good = s[0] === '✓'; const o = { size: 11.5, font: 'mono', weight: 600, color: s[0] === '✗' ? C.red : good ? C.e : C.muted }; const ww = tw(g, s, o) + 16; if (ix + ww > w - m) { ix = m; iy += 18; } V.text(g, s, ix, iy, o); ix += ww; });
        // decomposição em tiles
        const dec = (name, v, tile, need) => { const q = Math.floor(v / tile), r = v % tile; return { s: name + ' = ' + nf(v) + ' = ' + q + '·' + tile + ' + ' + r, ok: r === 0 && v >= need }; };
        const decs = mx ? [dec('N', N, 32, 128), dec('K', K, 32, 128)] : [dec('K', K, 16, 0)];
        iy += 24;
        decs.forEach((d) => { V.text(g, d.s + (d.ok ? '  ✓' : '  ✗'), m, iy, { size: narrow ? 12 : 13.5, font: 'mono', weight: 700, color: d.ok ? C.e : C.red, maxW: w - 2 * m }); iy += narrow ? 19 : 21; });
        // desenha uma janela [n × k] da matriz com a grade de tiles e os restos
        const drawMat = (bx, by, bw, bh, n, k, fullN, fullK, labK, labN, kTop) => {
          V.box(g, bx, by, bw, bh, { r: 3, fill: V.alpha(C.b, 0.14), stroke: V.alpha(C.b, 0.7) });
          const sxK = bw / k, syN = bh / n;
          const nK = Math.floor(k / T), remK = fullK % T;
          const stepK = Math.max(1, Math.ceil(nK / 90));
          g.strokeStyle = V.alpha(C.b, 0.45); g.lineWidth = 1;
          g.beginPath();
          for (let q = stepK; q <= nK; q += stepK) { const x = bx + q * T * sxK; if (x < bx + bw - 0.5) { g.moveTo(x, by); g.lineTo(x, by + bh); } }
          if (mx) { const nN = Math.floor(n / T), stepN = Math.max(1, Math.ceil(nN / 90)); for (let q = stepN; q <= nN; q += stepN) { const y = by + q * T * syN; if (y < by + bh - 0.5) { g.moveTo(bx, y); g.lineTo(bx + bw, y); } } }
          g.stroke();
          // restos: só aparecem se a janela chega ao fim da dimensão
          if (remK && k === fullK) { const x = bx + nK * T * sxK; g.fillStyle = V.alpha(C.red, 0.45); g.fillRect(x, by, Math.max(3, bx + bw - x), bh); }
          if (mx && fullN % 32 && n === fullN) { const y = by + Math.floor(n / T) * T * syN; g.fillStyle = V.alpha(C.red, 0.45); g.fillRect(bx, y, bw, Math.max(3, by + bh - y)); }
          if (mx && (!okNmin || !okKmin)) { g.strokeStyle = C.red; g.lineWidth = 2.5; g.setLineDash([6, 4]); V.rr(g, bx - 3, by - 3, bw + 6, bh + 6, 5); g.stroke(); g.setLineDash([]); }
          V.text(g, labK, bx + bw / 2, kTop ? by - 8 : by + bh + 16, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: w - 2 * m });
          if (labN) { g.save(); g.translate(bx - 10, by + bh / 2); g.rotate(-Math.PI / 2); V.text(g, labN, 0, 0, { size: 11, font: 'mono', align: 'center', color: C.ink }); g.restore(); }
          return stepK;
        };
        const ax = m + 34, ay = iy + 20, aw = w - ax - m - 4, ahAll = h - ay - 34;
        let ratio = N / K; const clamped = ratio < 0.06 || ratio > 12;
        ratio = Math.max(0.06, Math.min(12, ratio));
        const ah = clamped ? Math.max(40, ahAll * 0.36) : ahAll;
        let bw = aw, bh = aw * ratio;
        if (bh > ah) { bh = ah; bw = ah / ratio; }
        bh = Math.max(bh, 18); bw = Math.max(bw, 18);
        const bx = ax + (aw - bw) / 2, by = ay + (ah - bh) / 2;
        const remK = K % T;
        const labNfull = 'N = ' + nf(N) + (mx && N % 32 ? ' · resto ' + (N % 32) : '');
        const labKfull = 'K (entrada) = ' + nf(K) + (remK ? '  · resto ' + remK : '');
        // faixa achatada: o rótulo de N vai junto com o de K, em cima (o rotulado lateral não caberia)
        const stepK = clamped ? drawMat(bx, by, bw, bh, N, K, N, K, labKfull + ' · ' + labNfull, '', true) : drawMat(bx, by, bw, bh, N, K, N, K, labKfull, labNfull, false);
        if (clamped) {
          // zoom do canto com a proporção real: 256 na dimensão longa, a curta inteira (até 256)
          const zN = Math.min(N, narrow ? 128 : 256), zK = Math.min(K, narrow ? 128 : 256);
          const zy = by + bh + 26, zah = h - zy - 40;
          const zr = zN / zK;
          let zw = aw, zh = aw * zr;
          if (zh > zah) { zh = zah; zw = zah / zr; }
          const zx = ax + (aw - zw) / 2;
          // retângulo do zoom na faixa principal e ligação
          const rw = Math.max(4, (zK / K) * bw), rh = Math.max(4, (zN / N) * bh);
          g.strokeStyle = C.d; g.lineWidth = 1.5; g.strokeRect(bx, by, rw, rh);
          V.line(g, bx + rw, by + rh, zx + zw, zy - 4, { color: V.alpha(C.d, 0.7), lw: 1, dash: [3, 3] });
          V.line(g, bx, by + rh, zx, zy - 4, { color: V.alpha(C.d, 0.7), lw: 1, dash: [3, 3] });
          drawMat(zx, zy, zw, zh, zN, zK, N, K, 'zoom: K 0–' + nf(zK) + (zK < K ? ' de ' + nf(K) : '') + ' · proporção real', 'N 0–' + nf(zN));
        }
        const note = (clamped ? 'proporção esticada para caber · ' : '') + (stepK > 1 ? 'cada linha = ' + stepK + ' tiles · ' : '') + 'tile = ' + T + (mx ? '×' + T : ' em K');
        V.text(g, note, w - m, h - 8, { size: 11, align: 'right', color: C.muted, maxW: w - 2 * m });
      }, { static: true });
    })();

    /* =====================================================================
       FIG 9 — esteira que monta a linha de comando
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      const S = { mtp: 3, vocab: true, kv: 'fp8', yarn: false, ep: true, extra: false };
      let prev = [];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'MTP', options: [0, 1, 3], value: 3, onChange: (v) => { S.mtp = v; render(); } });
      UI.toggle(ctl, { label: 'MTP_DRAFT_VOCAB (47k)', value: true, onChange: (v) => { S.vocab = v; render(); } });
      UI.seg(ctl, { label: 'KV', options: ['fp8', 'auto'], value: 'fp8', onChange: (v) => { S.kv = v; render(); } });
      UI.toggle(ctl, { label: 'YaRN (1M)', value: false, onChange: (v) => { S.yarn = v; render(); } });
      UI.toggle(ctl, { label: 'expert parallel', value: true, onChange: (v) => { S.ep = v; render(); } });
      UI.toggle(ctl, { label: 'EXTRA_VLLM_ARGS', value: false, onChange: (v) => { S.extra = v; render(); } });
      const rArgs = UI.readout(ctl, 'flags', '—');
      const rOv = UI.readout(ctl, 'overlays', '—');
      const out = LA.h('div', { style: { padding: '0 16px 12px' } });
      beforeCap(fig, out);
      function build() {
        const a = [];
        a.push('--enable-prompt-tokens-details', '--served-model-name qwen3.8-flash-next', '--tensor-parallel-size 2', '--gpu-memory-utilization 0.80', '--max-num-seqs 8', '--max-num-batched-tokens 8192', '--max-model-len ' + (S.yarn ? '1000000' : '262144'), '--kv-cache-dtype ' + S.kv, '--mamba-ssm-cache-dtype bfloat16', '--load-format safetensors', '--safetensors-load-strategy lazy', '--enable-chunked-prefill', '--reasoning-parser qwen3', '--enable-auto-tool-choice', '--tool-call-parser qwen3_coder', '--distributed-executor-backend mp', '--mm-encoder-tp-mode data', '--nnodes 2', '--master-addr $HEAD_IP', '--master-port 50000');
        if (S.ep) a.push('--enable-expert-parallel', '--all2all-backend allgather_reducescatter');
        if (S.mtp > 0) a.push(`--speculative-config '{"method":"mtp","num_speculative_tokens":${S.mtp}${S.vocab ? ',"use_local_argmax_reduction":true' : ''},"disable_eagle_block_drop":true,"index_share_for_mtp_iteration":true}'`);
        a.push(`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`);
        a.push(S.yarn ? `--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn","rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'` : `--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}'`);
        if (S.extra) a.push('--kv-cache-memory=41360080896');
        return a;
      }
      function overlays() {
        const o = ['ple_layer_patched.py', 'modelopt_patched.py', 'config_patched.json', 'hf_quant_config_patched.json'];
        if (S.kv === 'fp8') o.push('qsa_ops_patched.py', 'qsa_nvidia_patched.py');
        if (S.vocab && S.mtp > 0) o.push('mtp_patched.py', 'draft_vocab_en_code_47k.txt');
        if (S.mtp > 0) o.push('speculative.py', 'kv_cache_utils.py', 'scheduler.py');
        return o;
      }
      function render() {
        const err = S.vocab && S.mtp === 0;
        const a = build();
        const ov = overlays();
        rArgs.set(String(a.length)); rOv.set(String(ov.length));
        const lines = a.map((s) => {
          const changed = prev.length && !prev.includes(s);
          const t = '  ' + esc(s) + ' \\';
          return changed ? `<span style="background:color-mix(in srgb, var(--mint) 24%, transparent);border-radius:4px">${t}</span>` : t;
        });
        prev = a;
        const warn = err ? '<div class="la-callout danger" style="margin:.6em 0"><b class="t">start.sh aborta</b><p><code>MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts.</code></p></div>' : '';
        out.innerHTML = warn + `<pre style="max-height:330px;overflow:auto;font-size:12px;margin:.6em 0 .4em"><code>vllm/vllm-openai:qwen38-flash-next nvidia/Qwen3.8-Flash-Next-NVFP4 \\\n${lines.join('\n')}\n  # head:   --node-rank 0 --host 0.0.0.0 --port 8888\n  # worker: --node-rank 1 --headless</code></pre>` +
          `<p style="font-size:13px;color:var(--text-3);margin:.2em 0">overlays montados <code>:ro</code> nos dois nós: ${ov.map((x) => '<code>' + esc(x) + '</code>').join(' ')}</p>`;
      }
      render();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, m = 12;
        const knobs = [
          { l: 'MTP=' + S.mtp, on: S.mtp > 0 },
          { l: S.vocab ? 'vocab 47k' : 'vocab cheio', on: S.vocab },
          { l: 'KV ' + S.kv, on: S.kv === 'fp8' },
          { l: S.yarn ? 'YaRN 1M' : 'YaRN off', on: S.yarn },
          { l: S.ep ? 'EP on' : 'EP off', on: S.ep },
          { l: 'EXTRA', on: S.extra },
        ];
        // geometria: no desktop tudo numa linha; no mobile a esteira fica em cima e os ranks embaixo
        const mw = narrow ? 92 : 124, mh = narrow ? 84 : 112;
        const beltY = narrow ? 70 : h * 0.5;
        const mx = narrow ? w - m - mw / 2 : w * 0.46;
        const envW = narrow ? 46 : 54, envX = m;
        const bx0 = envX + envW - 6, bx1 = mx - mw / 2;
        // esteira
        V.box(g, bx0, beltY + 16, bx1 - bx0, 10, { r: 5, fill: V.alpha(C.muted, 0.25) });
        for (let x = bx0 + 6 + ((t * 30) % 14); x < bx1; x += 14) V.line(g, x, beltY + 17, x - 5, beltY + 25, { color: V.alpha(C.muted, 0.5), lw: 1.2 });
        const kw = narrow ? 70 : 78, kh = 24, pitch = kw + 12, span = Math.max(bx1 - bx0 + kw, knobs.length * pitch);
        g.save(); g.beginPath(); g.rect(bx0, beltY - 30, bx1 - bx0, 60); g.clip();
        knobs.forEach((k, i) => {
          const x = bx0 - kw + (i * pitch + t * 26) % span;
          if (x + kw > bx1 + 4) return;
          V.box(g, x, beltY - 9, kw, kh, { r: 6, fill: k.on ? V.alpha(C.qwen, 0.3) : V.alpha(C.muted, 0.15), stroke: k.on ? C.qwen : V.alpha(C.muted, 0.5), label: k.l, size: 11 });
        });
        g.restore();
        // o arquivo .env de onde as chaves saem
        V.box(g, envX, beltY - 26, envW, 52, { r: 8, fill: V.alpha(C.d, 0.16), stroke: C.d, lw: 1.5 });
        g.fillStyle = V.alpha(C.d, 0.55); g.beginPath(); g.moveTo(envX + envW - 12, beltY - 26); g.lineTo(envX + envW, beltY - 14); g.lineTo(envX + envW - 12, beltY - 14); g.closePath(); g.fill();
        V.text(g, '.env', envX + envW / 2, beltY + 4, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.ink });
        // máquina
        V.box(g, mx - mw / 2, beltY - mh / 2, mw, mh, { r: 14, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 2 });
        V.text(g, 'start.sh', mx, beltY - mh / 2 + 20, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.ink });
        V.text(g, 'VLLM_ARGS', mx, beltY + mh / 2 - 11, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const gear = (cx, cy, r, a, col) => { g.save(); g.translate(cx, cy); g.rotate(a); g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.stroke(); for (let k = 0; k < 8; k++) { g.rotate(Math.PI / 4); g.beginPath(); g.moveTo(r, 0); g.lineTo(r + 4, 0); g.stroke(); } g.restore(); };
        gear(mx - 14, beltY + 2, narrow ? 10 : 13, t * 1.6, C.b); gear(mx + 14, beltY + 2, narrow ? 7 : 9, -t * 2.3, C.a);
        // saída: uma string que se divide nos dois ranks
        const bw2 = narrow ? Math.min(160, (w - 2 * m - 24) / 2) : 146, bh2 = 46;
        let P0, Psplit, targets;
        if (narrow) {
          const yB = h - 34, splitY = beltY + mh / 2 + 44;
          P0 = [mx, beltY + mh / 2]; Psplit = [mx, splitY];
          const cW = m + bw2 / 2, cH = w - m - bw2 / 2;
          targets = [{ c: [cW, yB], x: m, y: yB - bh2 / 2, t: 'worker · rank 1', s: '--headless' }, { c: [cH, yB], x: w - m - bw2, y: yB - bh2 / 2, t: 'head · rank 0', s: ':8888 (API)' }];
        } else {
          const ox = mx + mw / 2, tx = w - m - bw2, yW = beltY - 64, yH = beltY + 64;
          P0 = [ox, beltY]; Psplit = [ox + (tx - ox) * 0.35, beltY];
          targets = [{ c: [tx, yW], x: tx, y: yW - bh2 / 2, t: 'worker · rank 1', s: '--headless' }, { c: [tx, yH], x: tx, y: yH - bh2 / 2, t: 'head · rank 0', s: ':8888 (API)' }];
        }
        const curve = (T0, v) => narrow
          ? V.bez(Psplit, [Psplit[0], Psplit[1] + 30], [T0.c[0], T0.y - 40], [T0.c[0], T0.y], v)
          : V.bez(Psplit, [Psplit[0] + 30, Psplit[1]], [Psplit[0] + 10, T0.c[1]], [T0.c[0], T0.c[1]], v);
        V.line(g, P0[0], P0[1], Psplit[0], Psplit[1], { color: C.qwen, lw: 3 });
        targets.forEach((T0) => {
          g.beginPath(); const a = curve(T0, 0); g.moveTo(a[0], a[1]);
          for (let q = 1; q <= 24; q++) { const p = curve(T0, q / 24); g.lineTo(p[0], p[1]); }
          g.strokeStyle = C.qwen; g.lineWidth = 3; g.stroke();
        });
        for (let k = 0; k < 4; k++) {
          const u = ((t * 0.35 + k / 4) % 1);
          if (u < 0.35) { const f = u / 0.35; V.box(g, V.lerp(P0[0], Psplit[0], f) - 8, V.lerp(P0[1], Psplit[1], f) - 4, 16, 8, { r: 3, fill: C.qwen }); }
          else { const v = (u - 0.35) / 0.65; targets.forEach((T0) => { const p = curve(T0, v); V.box(g, p[0] - 8, p[1] - 4, 16, 8, { r: 3, fill: C.qwen }); }); }
        }
        targets.forEach((T0) => {
          V.box(g, T0.x, T0.y, bw2, bh2, { r: 10, fill: V.alpha(C.qwen, 0.12), stroke: C.qwen });
          V.text(g, T0.t, T0.x + bw2 / 2, T0.y + 19, { size: narrow ? 11.5 : 12, weight: 700, align: 'center', color: C.ink });
          V.text(g, T0.s, T0.x + bw2 / 2, T0.y + 36, { size: 11, font: 'mono', align: 'center', color: C.muted });
        });
        if (narrow) {
          V.text(g, 'mesma string', mx - 14, P0[1] + 18, { size: 11, align: 'right', color: C.muted });
          V.text(g, 'nos dois ranks', mx - 14, P0[1] + 33, { size: 11, align: 'right', color: C.muted });
        } else {
          V.text(g, 'mesma string', targets[0].x + bw2 / 2, beltY - 3, { size: 11, align: 'center', color: C.muted });
          V.text(g, 'nos dois ranks', targets[0].x + bw2 / 2, beltY + 12, { size: 11, align: 'center', color: C.muted });
        }
        if (S.vocab && S.mtp === 0) { V.glow(g, mx, beltY, 60 * (0.7 + 0.3 * Math.sin(t * 6)), C.red, 0.7); V.text(g, 'aborta!', mx, beltY - mh / 2 - 8, { size: 12, weight: 700, align: 'center', color: C.red }); }
      });
    })();

    /* =====================================================================
       FIG 10 — linha do tempo do boot
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-boot');
      const PH = [
        { l: 'resto (não detalhado)', sh: 'resto', s: 56.6, k: 'rest' },
        { l: 'NCCL', sh: 'NCCL', s: 40, k: 'nccl' },
        { l: 'pesos do alvo', sh: 'alvo', s: 392, k: 'wt' },
        { l: 'rascunhador MTP', sh: 'MTP', s: 67, k: 'mtp' },
        { l: 'engine init', sh: 'init', s: 92.4, k: 'eng' },
        { l: 'graphs', sh: 'g', s: 7, k: 'gr' },
      ];
      const TOTAL = PH.reduce((a, p) => a + p.s, 0); // 655
      const SC = {
        ok: { l: 'boot medido', fail: null },
        shard: { l: 'shard corrompido', fail: 56.6 + 40 + 0.33 * 392, msg: 'safe_open: incomplete metadata, file not fully covered (~33% da carga)', pre: 'check-weights.sh --verify: hash não confere (≈1 min antes do launch)' },
        mtp: { l: 'MTP sem escala FP8', fail: 56.6 + 40 + 392 + 12, msg: "mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv' (~7 min de carga, ao chegar no rascunhador)", pre: '--mtp-moe-algo: algoritmo checado em segundos (exit 3 se não suportado)' },
        mxfp8: { l: 'MXFP8 sem fallback (ckpt local-inference-lab)', fail: 56.6 + 40 + 458 + 5, msg: 'AssertionError … N >= 128, got N=48 — no profile run, pesos já residentes. Só em checkpoints com camadas MXFP8 (ex.: local-inference-lab), não no nvidia. README: ~7 min dentro do launch; aqui a posição segue a linha do tempo do nvidia', pre: null },
        gpu: { l: 'GPU com inquilino', fail: 600, msg: 'CUDA out of memory (~10 min depois)', pre: 'REQUIRE_IDLE_GPU: nvidia-smi mostra processo → aborta em segundos' },
      };
      const S = { sc: 'ok', pre: false, t0: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: Object.entries(SC).map(([k, o]) => ({ v: k, label: o.l })), value: 'ok', onChange: (v) => { S.sc = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'preflights ligados', value: false, onChange: (v) => { S.pre = v; S.t0 = null; } });
      UI.button(ctl, '▶ repetir', () => { S.t0 = null; });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        if (S.t0 == null) S.t0 = t;
        const narrow = w < 600, m = 14;
        const sc = SC[S.sc];
        const hostFrac = 0.16;
        const x0 = m, x1 = w - m, span = x1 - x0;
        const hostW = span * hostFrac, cx0 = x0 + hostW, cspan = span - hostW;
        const by = 70, bh = 34;
        const colors = [V.alpha(C.muted, 0.45), C.b, C.qwen, C.f, C.d, C.e];
        const caught = S.pre && sc.pre;
        const failT = caught ? null : sc.fail;
        const real = t - S.t0;
        const speed = TOTAL / 9; // 655 s em ~9 s
        const hostDur = 1.4;
        let simT = (real - hostDur) * speed;
        const stopAt = caught ? -1 : failT;
        const inHost = real < hostDur || caught;
        if (stopAt != null && stopAt >= 0 && simT > stopAt) simT = stopAt;
        simT = Math.min(simT, TOTAL);
        const finished = (stopAt == null && simT >= TOTAL) || (stopAt != null && stopAt >= 0 && simT >= stopAt) || (caught && real > hostDur * 0.8);
        if (finished && real > hostDur + 9.5 + 3.5) S.t0 = t;
        // título
        V.text(g, 'HOST', x0, by - 12, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'CONTAINER · ≈10m55s até /health', cx0 + 6, by - 12, { size: 11, font: 'mono', color: C.muted, maxW: cspan - 10 });
        // fase no host
        V.box(g, x0, by, hostW - 4, bh, { r: 8, fill: V.alpha(C.a, 0.18), stroke: V.alpha(C.a, 0.6) });
        V.text(g, narrow ? 'host' : 'prep (variável)', x0 + (hostW - 4) / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: C.ink, maxW: hostW - 10 });
        // fases do container
        let x = cx0;
        PH.forEach((p, i) => {
          const ww = (p.s / TOTAL) * cspan;
          V.box(g, x, by, Math.max(1, ww - 1.5), bh, { r: 4, fill: V.alpha(colors[i], i === 0 ? 1 : 0.5) });
          const lab = !narrow && ww > tw(g, p.l, { size: 11 }) + 8 ? p.l : ww > tw(g, p.sh, { size: 11 }) + 6 ? p.sh : '';
          if (lab) V.text(g, lab, x + ww / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: C.ink });
          x += ww;
        });
        // eixo de minutos
        for (let mnt = 0; mnt <= 11; mnt++) {
          const xx = cx0 + ((mnt * 60) / TOTAL) * cspan;
          if (xx > x1) break;
          V.line(g, xx, by + bh + 2, xx, by + bh + 7, { color: C.muted, lw: 1 });
          if (!narrow || mnt % 2 === 0) V.text(g, mnt + 'm', xx, by + bh + 19, { size: 11, font: 'mono', align: 'center', color: C.muted });
        }
        // bandeira do ponto de falha
        if (sc.fail != null && !caught) {
          const fx = cx0 + (sc.fail / TOTAL) * cspan;
          V.line(g, fx, by - 20, fx, by + bh + 4, { color: C.red, lw: 1.5, dash: [3, 3] });
          V.box(g, fx, by - 24, 14, 9, { r: 2, fill: C.red });
        }
        // cursor
        let cxr;
        if (inHost) cxr = x0 + Math.min(1, real / hostDur) * (hostW - 6);
        else cxr = cx0 + (Math.max(0, simT) / TOTAL) * cspan;
        V.line(g, cxr, by - 6, cxr, by + bh + 6, { color: C.ink, lw: 2 });
        V.dot(g, cxr, by - 6, 4, C.ink);
        // legenda (mobile)
        let ly = by + bh + 44;
        if (narrow) {
          const cols = 2, lw = (w - 2 * m) / cols;
          PH.forEach((p, i) => { const cx = m + (i % cols) * lw, cy = ly + Math.floor(i / cols) * 17; V.box(g, cx, cy - 9, 10, 10, { r: 2, fill: V.alpha(colors[i], i === 0 ? 1 : 0.7) }); V.text(g, (i === 0 ? 'resto' : p.l) + ' ' + nf(p.s, p.s % 1 ? 1 : 0) + ' s', cx + 14, cy, { size: 11, color: C.ink, maxW: lw - 18 }); });
          ly += 3 * 17 + 8;
        } else {
          V.text(g, 'NCCL 40 s · alvo 392 s · rascunhador 67 s · engine init 92,4 s · graphs ~7 s · resto 56,6 s', m, ly, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * m });
          ly += 22;
        }
        // mensagem de status
        let msg, col;
        if (caught) { msg = 'Preflight: ' + sc.pre; col = C.d; }
        else if (stopAt != null && simT >= stopAt) { msg = '✕ ' + sc.msg; col = C.red; }
        else if (stopAt == null && simT >= TOTAL) { msg = '✓ /health = 200 em ≈10m55s — a 1ª requisição ainda paga o autotune do FlashInfer'; col = C.e; }
        else if (inHost) { msg = S.pre ? 'Rodando preflights no host: REQUIRE_IDLE_GPU, --mtp-moe-algo, check-weights --verify…' : 'Preparando no host (download, sync, overlays)…'; col = C.a; }
        else { const cur = (() => { let acc = 0; for (const p of PH) { acc += p.s; if (simT <= acc) return p.l; } return 'graphs'; })(); msg = 't = ' + nf(simT / 60, 1) + ' min · ' + cur; col = C.b; }
        if (sc.fail != null && S.pre && !sc.pre && S.sc === 'mxfp8') msg = (simT >= stopAt ? '✕ ' + sc.msg : msg) + ' — sem preflight: o conserto é o fallback, sempre montado';
        if (caught || (stopAt != null && simT >= stopAt)) {
          const fx = caught ? x0 + hostW * 0.7 : cxr;
          V.glow(g, fx, by + bh / 2, 30 * (0.7 + 0.3 * Math.sin(real * 7)), caught ? C.d : C.red);
        }
        const nl = countLines(g, msg, w - 2 * m - 24, { size: 12 });
        V.box(g, m, ly, w - 2 * m, nl * 16 + 14, { r: 10, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.6) });
        wrapText(g, msg, m + 12, ly + 19, w - 2 * m - 24, 16, { size: 12, color: C.ink });
      });
    })();

    /* =====================================================================
       FIG 2 — bytes por passo de decode (alavancas)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-bytes-passo');
      const BASE = [
        { k: 'lm', l: 'lm_head ×4 (alvo + 3 rascunhos)', ls: 'lm_head ×4', v: 2.36 },
        { k: 'exp', l: 'experts roteados (~32 únicos/camada)', ls: 'experts roteados', v: 2.1 },
        { k: 'gdn', l: 'projeções GDN', v: 1.95 },
        { k: 'hc', l: 'HyperConnections (replicadas)', ls: 'HyperConnections', v: 1.19 },
        { k: 'att', l: 'atenção + indexador', v: 0.60 },
        { k: 'mtp', l: 'MTP denso + experts ×3', v: 0.5 },
        { k: 'rs', l: 'roteador + shared + proj. PLE', ls: 'roteador + shared', v: 0.40 },
        { k: 'st', l: 'estado GDN (lê + escreve)', ls: 'estado GDN (R+W)', v: 0.1 },
      ];
      const S = { vocab: false, dense: false, bw: 240, a: 2.9, ov: 17, anim: {} };
      const ctl = UI.controls(fig);
      const tV = UI.toggle(ctl, { label: 'vocabulário de rascunho 47k', value: false, onChange: (v) => { S.vocab = v; if (v && S.dense) tD.set(false); } });
      const tD = UI.toggle(ctl, { label: 'FP8-dense (analítico)', value: false, onChange: (v) => { S.dense = v; if (v && S.vocab) tV.set(false); } });
      UI.slider(ctl, { label: 'banda', min: 200, max: 273, step: 1, value: 240, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; } });
      UI.slider(ctl, { label: 'tokens/passo', min: 1, max: 4, step: 0.05, value: 2.9, fmt: (v) => v.toFixed(2).replace('.', ','), onInput: (v) => { S.a = v; } });
      UI.slider(ctl, { label: 'overhead', min: 0, max: 25, step: 0.5, value: 17, fmt: (v) => nf(v, 1) + ' ms', onInput: (v) => { S.ov = v; } });
      const rB = UI.readout(ctl, 'GiB/passo', '—');
      const rT = UI.readout(ctl, 'ms/passo', '—');
      const rS = UI.readout(ctl, 'tok/s (≈)', '—');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600, m = 14;
        const comps = BASE.map((c) => ({ ...c, v: c.k === 'lm' && S.vocab ? 0.59 + 3 * 0.22 : c.v }));
        const sum = comps.reduce((a, c) => a + c.v, 0);
        const total = S.dense ? 5.7 : sum;
        const readMs = (total * 1.073741824) / S.bw * 1000;
        const stepMs = readMs + S.ov;
        const tps = S.a / (stepMs / 1000);
        rB.set(nf(total, 2)); rT.set(nf(stepMs, 1)); rS.set(nf(tps, 1));
        const lw = narrow ? 150 : 250, vw = 54;
        const bx0 = m + lw, bwMax = w - m - vw - bx0;
        const top = 34, rows = comps.length, bottom = 112;
        const rh = (h - top - bottom) / rows;
        const maxV = 2.4;
        V.text(g, S.dense ? 'FP8-dense: o relatório estima só o total (≈5,7 GiB) — barras em cinza' : 'bytes lidos por GPU num passo (batch 1, MTP=3)', m, 20, { size: 11.5, weight: 600, color: S.dense ? C.d : C.ink, maxW: w - 2 * m });
        // cabeça de leitura
        const cyc = (t * 1000 / Math.max(20, stepMs)) * 0.035; // passos bem "desacelerados" para ficar visível
        const ph = cyc % 1;
        const cumul = []; let acc = 0; comps.forEach((c) => { cumul.push(acc); acc += c.v; });
        const pos = ph * acc;
        comps.forEach((c, i) => {
          const y = top + i * rh;
          const target = (c.v / maxV) * bwMax;
          S.anim[c.k] = S.anim[c.k] == null ? target : V.lerp(S.anim[c.k], target, 0.15);
          const bw = S.anim[c.k];
          const colr = c.k === 'lm' ? C.d : c.k === 'exp' ? C.qwen : c.k === 'gdn' ? C.b : c.k === 'hc' ? C.f : C.muted; // cores bem distintas nos dois temas
          V.text(g, fitText(g, narrow && c.ls ? c.ls : c.l, lw - 8, { size: 11.5 }), m, y + rh / 2 + 4, { size: 11.5, color: C.ink });
          const bh = Math.min(18, rh - 6), by = y + (rh - bh) / 2;
          V.box(g, bx0, by, Math.max(2, bw), bh, { r: 4, fill: S.dense ? V.alpha(C.muted, 0.3) : V.alpha(colr, 0.65) });
          // leitura em progresso
          if (!S.dense && pos > cumul[i]) { const fr = Math.min(1, (pos - cumul[i]) / c.v); V.box(g, bx0, by, Math.max(2, bw * fr), bh, { r: 4, fill: V.alpha(colr, 0.95) }); if (fr < 1) V.glow(g, bx0 + bw * fr, by + bh / 2, 12, colr); }
          V.text(g, nf(c.v, 2), w - m, y + rh / 2 + 4, { size: 11, font: 'mono', align: 'right', color: C.ink });
        });
        // barra total e tokens
        const ty = h - bottom + 22;
        const totW = w - 2 * m;
        V.text(g, 'total ' + nf(total, 2) + ' GiB · leitura ' + nf(readMs, 1) + ' ms + overhead ' + nf(S.ov, 1) + ' ms', m, ty, { size: 11.5, font: 'mono', color: C.ink, maxW: totW });
        V.box(g, m, ty + 8, totW, 12, { r: 6, fill: V.alpha(C.muted, 0.15) });
        V.box(g, m, ty + 8, totW * Math.min(1, readMs / 70), 12, { r: 6, fill: V.alpha(C.qwen, 0.7) });
        V.box(g, m + totW * Math.min(1, readMs / 70), ty + 8, totW * Math.min(1, S.ov / 70), 12, { r: 0, fill: V.alpha(C.muted, 0.45) });
        V.text(g, '0', m, ty + 34, { size: 11, font: 'mono', color: C.muted }); V.text(g, '70 ms', w - m, ty + 34, { size: 11, font: 'mono', align: 'right', color: C.muted });
        // fichas de tokens
        const n = Math.floor(S.a), fr = S.a - n;
        const tx = w / 2 - ((n + (fr > 0.05 ? 1 : 0)) * 20) / 2;
        const pop = ph > 0.8 ? V.ease.back(Math.min(1, (ph - 0.8) / 0.2)) : 0;
        for (let k = 0; k < n + (fr > 0.05 ? 1 : 0); k++) {
          const s = k < n ? 1 : fr;
          V.box(g, tx + k * 20, ty + 44 - 7 * pop, 16 * s, 14, { r: 5, fill: V.alpha(C.e, 0.35 + 0.55 * pop) });
        }
        V.text(g, nf(S.a, 2) + ' tokens/passo → ≈' + nf(tps, 1) + ' tok/s', w / 2, ty + 70, { size: 12, weight: 700, align: 'center', color: C.ink });
      });
    })();
  },
});

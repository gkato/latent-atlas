LA.chapter({
  id: 'fp8-fp4-microescala',
  part: 4,
  order: 21,
  title: 'FP8, FP4 e microescalas: MXFP4, MXFP8 e NVFP4',
  navTitle: 'FP8, FP4 & microescala',
  subtitle: 'Como 8 e 4 bits guardam números de verdade: minifloats bit a bit, escalas por tensor, canal e bloco 128×128, expoentes compartilhados (E8M0), a boneca russa do NVFP4, W4A16 contra W4A4 no GB10 e o que acontece quando o kernel recusa uma forma.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['FP8', 'E4M3', 'E5M2', 'FP4', 'E2M1', 'E8M0', 'MXFP4', 'MXFP8', 'NVFP4', 'microescala', 'W4A16', 'W4A4', 'ModelOpt', 'tensor cores'],
  goals: [
    'Ler e montar à mão códigos FP8 <b>E4M3</b>/<b>E5M2</b> e FP4 <b>E2M1</b>, sabendo faixa, precisão e por que 448 e 6 aparecem em toda escala',
    'Escolher a granularidade da escala (tensor, canal, token, bloco 1D, bloco 128×128) e calcular os <b>bits efetivos</b> por peso',
    'Separar <b>memória</b> de <b>banda</b>: somar os bytes que cada passo de decode lê para decidir o que quantizar (experts para caber, densos para acelerar)',
    'Explicar MX (bloco 32, E8M0) e NVFP4 (bloco 16, E4M3 + FP32) e o que <code>weight_scale</code>, <code>weight_scale_2</code>, <code>input_scale</code> e <code>weight_scale_inv</code> significam num checkpoint',
    'Prever se um formato roda como <b>W4A16</b> ou <b>W4A4</b> no GB10 e o que acontece quando a forma por rank não serve ao kernel',
  ],
  prereqs: ['quantizacao-fundamentos'],
  summary: [
    'Minifloats têm régua logarítmica: <b>E4M3</b> (1/4/3, máx. 448, ~18 binades, erro ≤ 6,25%) é o FP8 de pesos, ativações e KV; <b>E5M2</b> (máx. 57.344) troca precisão por faixa e só aparece como opção <code>fp8_e5m2</code>; <b>E2M1</b> tem só 15 valores {0, ±0,5, ±1, ±1,5, ±2, ±3, ±4, ±6}.',
    'A escala \\(s=\\mathrm{amax}/\\text{máx}\\) leva o grupo ao topo da régua. Quanto menos binades o formato tem, mais fina precisa ser a escala: FP8 vive com escala por tensor, canal ou bloco 128×128; FP4 exige blocos de 16 ou 32.',
    'Escalas <b>E8M0/UE8M0</b> são só expoente: exatas e baratas, mas grosseiras. O piso (OCP MX) pode saturar, o teto (KV do DeepSeek) nunca satura, e um byte E8M0 deve ser reinterpretado (<code>view</code>), nunca convertido (2⁻⁷ → 0). O KV FP8 do GLM no TensorFold usa uma terceira regra, \\(2^{\\lceil\\log_2 \\mathrm{amax}\\rceil-8}\\) por linha de 512: o máximo cai em (128, 256], nada satura, e só os valores abaixo de ~2⁻¹⁴ do amax perdem precisão, porque a precisão relativa de um float não depende da escala.',
    '<b>MXFP4</b> = E2M1 + E8M0 a cada 32 (4,25 bits; experts do DeepSeek, <code>fp4_e8m0_k32</code>). <b>NVFP4</b> = E2M1 + E4M3 a cada 16 + FP32 global (4,5 bits; experts do Qwen: <code>weight_scale</code>, <code>weight_scale_2</code>, <code>input_scale</code>). O NVFP4 é mais robusto a outliers.',
    'Num MoE, os experts dominam a <b>memória</b>, mas os densos BF16 dominam os <b>bytes lidos por passo</b>: no Qwen, ~6,9 dos 9,2 GiB de um passo (experts ~2,1). Por isso <code>FP8_DENSE</code> e <code>GLM53_DENSE_FP8</code> aceleram o decode (≈ 9,2 → 5,7 GiB analítico; −11 ms/passo medido no GLM). Na versão TensorFold da receita GLM, o FP8 denso usa escala FP32 por linha e bloco de 128 colunas (8,25 bits; ~+33% de decode sobre BF16), e o padrão desce a 4 bits (<code>DENSE=q4</code>).',
    '<b>W4A16</b> desquantiza em registrador e ganha banda (decode); <b>W4A4</b> usa o tensor core FP4 e ganha FLOPs (prefill). O MXFP4 do DeepSeek cai em W4A16 no b12x; o plugin GB10 (<code>modelopt_gb10_hybrid</code>) despacha por M = 128, mas só serve para checkpoints ModelOpt NVFP4.',
    'Kernels exigem formas por rank múltiplas de 16, 32 ou 128. As receitas usam fallbacks explícitos (emulação MXFP8 → BF16, replicar a visão, FP8 por canal) e patches que recusam adivinhar (bloco lido do <code>group_size</code>), porque um bloco errado falha em silêncio.',
  ],
  quiz: [
    { q: 'Por que o máximo do FP8 E4M3 “fn” é 448, e não 240?', o: ['Porque ele usa bias 8', 'Porque tem 4 bits de mantissa', 'Porque abre mão de ±∞: o expoente 1111 continua representando números normais e só S.1111.111 é NaN, então o maior valor é 1,75 × 2⁸', 'Porque 448 é o maior inteiro de 9 bits'], a: 2, e: 'O “fn” (finite) reaproveita o expoente máximo para números normais. A maior mantissa válida com E = 1111 é 110 (1,75), porque 111 é NaN: 1,75 × 2^(15−7) = 448. Num E4M3 estilo IEEE, o expoente 1111 seria reservado e o máximo cairia para 1,875 × 2⁷ = 240.' },
    { q: 'Num tensor cujas linhas têm magnitudes muito diferentes, por que o FP8 costuma sobreviver com uma escala por tensor e o FP4 não?', o: ['Porque o E4M3 cobre ~18 binades e absorve a variação; o E2M1 cobre só ~3,6, e as linhas pequenas viram zero sem uma escala por bloco', 'Porque o FP8 usa escala E8M0', 'Porque o FP4 não tem bit de sinal', 'Porque o FP8 é um formato inteiro'], a: 0, e: 'Faixa é número de binades. Com 18 binades, o E4M3 comporta linhas 2⁴ vezes menores que a maior e ainda sobra resolução. O E2M1 vai de 0,5 a 6, com s = amax/6. Uma linha 24× menor que o amax vai no máximo até 6/24 = 0,25 e vira zero inteira. Uma 16× menor vai até 6/16 = 0,375: a maior parte zera, e o que passa de 0,25 vira 0,5 (erro de 33% ou mais). Por isso todo formato de 4 bits é de microescala.' },
    { q: 'O KV do DeepSeek usa a escala 2^⌈log₂(amax/448)⌉ por bloco de 64. Qual é a consequência dessa regra?', o: ['Pode saturar valores do bloco', 'Produz escalas fracionárias', 'Exige uma escala global FP32', 'Nunca satura, mas o maior valor pode ocupar só metade da faixa (até 1 bit desperdiçado)'], a: 3, e: 'Arredondar o expoente para cima garante amax/s ≤ 448, então nada é cortado. O custo é que amax/s fica em (224, 448]: no exemplo, amax = 3,7 usa 236,8 de 448. A regra do piso do OCP MX faz o oposto e pode saturar.' },
    { q: 'NVFP4 e MXFP4 usam o mesmo elemento E2M1. O que muda entre eles?', o: ['Nada: são o mesmo formato com nomes diferentes', 'NVFP4: bloco de 16, escala E4M3 e uma escala global FP32 (4,5 bits); MXFP4: bloco de 32, escala E8M0 potência de 2 (4,25 bits)', 'O MXFP4 tem mais bits de mantissa', 'O NVFP4 não tem escala por bloco'], a: 1, e: 'Só a escala muda. O bloco menor e a escala fracionária do NVFP4 isolam outliers e enquadram melhor cada bloco, ao custo de 0,25 bit/valor. A escala global FP32 (weight_scale_2) leva as escalas de bloco para dentro da faixa do E4M3.' },
    { q: 'Por que os experts MXFP4 do DeepSeek rodam como W4A16 no GB10, que tem tensor cores FP4?', o: ['Porque o GB10 não tem instruções MMA de FP4', 'Porque W4A16 tem menos erro que W4A4', 'Porque o caminho FP4 de tensor core do b12x só aceita fonte modelopt_nvfp4; o checkpoint é fp4_e8m0_k32, então o kernel desquantiza para BF16 (bom no decode, caro no prefill)', 'Porque ENABLE_VLLM_GB10_PATCH está desligado'], a: 2, e: 'O log confirma B12X_MXFP4 e W4A16FusedMoeKernel. No decode (memory-bound) o custo é nulo; no prefill a MMA BF16 rende uma fração do FP4, e a receita estima +30–60% com um MoE MXFP4×MXFP8 nativo. O plugin GB10 não ajudaria: ele só troca lineares ModelOpt NVFP4, que este checkpoint não tem.' },
    { q: 'O checkpoint do Qwen declara FP8_BLOCK_SCALES (group_size 128) nos experts do MTP, mas o dispatch do ModelOpt não tem ramo para esse quant_algo. O que acontece sem o patch?', o: ['O vLLM adivinha o bloco 128×128 e segue', 'O vLLM converte os experts para NVFP4', 'A camada nasce sem quantização, em silêncio, e a carga morre ~7 min depois ao procurar w2_weight_scale_inv', 'Nada: roda em BF16 normalmente'], a: 2, e: 'get_quant_method devolve None, o MoE é construído não quantizado e o loader não encontra onde pôr weight_scale_inv. O patch acrescenta o ramo que monta Fp8MoEMethod com weight_block_size lido do group_size, porque um bloco errado aplicaria escalas desalinhadas sem erro nenhum.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;
    const $ = (s) => root.querySelector(s);
    const SUP = { '-': '⁻', '−': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    const sup = (n) => String(n).split('').map((c) => SUP[c] || c).join('');
    const nice = (x, dd) => {
      if (Number.isNaN(x)) return 'NaN';
      if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
      if (x === 0) return '0';
      const a = Math.abs(x);
      let s;
      if ((a >= 1e5 || a < 1e-3) && !(dd != null && a >= Math.pow(10, -dd))) { const e = Math.floor(Math.log10(a)); const m = x / Math.pow(10, e); s = m.toFixed(2) + '·10' + sup(e); }
      else { const d = dd != null ? dd : a >= 1000 ? 0 : a >= 100 ? 2 : a >= 10 ? 2 : a >= 1 ? 3 : 4; s = x.toFixed(d); if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, ''); }
      return s.replace('.', ',').replace('-', '−');
    };
    const pct = (x, d = 1) => (x * 100).toFixed(d).replace('.', ',') + '%';
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
    function tw(g, s, o) { g.font = fnt(o); return g.measureText(s).width; }
    function fitH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = st.getBoundingClientRect().width; if (!w) return; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const onDark = (C) => C.bg;
    const relRMSE = (xs, ys) => { let a = 0, b = 0; xs.forEach((v, i) => { a += (ys[i] - v) * (ys[i] - v); b += v * v; }); return Math.sqrt(a / (b || 1e-30)); };
    const safe = (fn) => { try { fn(); } catch (e) { console.error('[cap. 21] figura falhou:', e); } };

    /* ===================================================================
       Figura 1 — Explorador de bits + réguas
       =================================================================== */
    safe(function figBits() {
      const fig = $('#fig-bits'); if (!fig) return;
      fitH(fig, (w) => (w < 560 ? 540 : 440));
      const NAMES = { e4m3: 'FP8 E4M3 (fn)', e5m2: 'FP8 E5M2', e2m1: 'FP4 E2M1', bf16: 'BF16' };
      const S = { fmt: 'e4m3', mag: Math.log2(Math.PI), neg: false, tx: Math.log2(Math.PI), zero: false };
      let bits = 0, stored = 0, xTrue = Math.PI;
      const F = () => N.formats[S.fmt];
      function encode() { S.zero = false; xTrue = (S.neg ? -1 : 1) * Math.pow(2, S.mag); const e = N.encode(S.fmt, xTrue); bits = e.bits; stored = e.v; }
      const rulers = [
        { id: 'int8', name: 'INT8 (s = 1)', range: '1 … 127', col: (C) => C.muted, vals: Array.from({ length: 127 }, (_, i) => i + 1) },
        { id: 'e4m3', name: 'FP8 E4M3', range: '2⁻⁹ … 448', col: (C) => C.a, vals: N.table('e4m3').map((t) => t.v).filter((v) => v > 0) },
        { id: 'e5m2', name: 'FP8 E5M2', range: '2⁻¹⁶ … 57344', col: (C) => C.b, vals: N.table('e5m2').map((t) => t.v).filter((v) => v > 0) },
        { id: 'e2m1', name: 'FP4 E2M1', range: '0,5 … 6', col: (C) => C.d, vals: N.table('e2m1').map((t) => t.v).filter((v) => v > 0) },
      ];
      rulers.forEach((r) => { r.lg = r.vals.map(Math.log2); });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Formato', options: [{ v: 'e4m3', label: 'E4M3' }, { v: 'e5m2', label: 'E5M2' }, { v: 'e2m1', label: 'E2M1' }, { v: 'bf16', label: 'BF16' }], value: 'e4m3', onChange: (v) => { S.fmt = v; encode(); upd(); } });
      const sl = UI.slider(ctl, { label: 'Magnitude |x|', min: -12, max: 16, step: 0.01, value: S.mag, fmt: (v) => nice(Math.pow(2, v)), onInput: (v) => { S.mag = v; encode(); upd(); } });
      const tg = UI.toggle(ctl, { label: 'negativo', value: false, onChange: (v) => { S.neg = v; encode(); upd(); } });
      const quiet = (mag, neg) => { sl.input.value = mag; sl.el.querySelector('.val').textContent = nice(Math.pow(2, mag)); tg.el.querySelector('input').checked = neg; };
      UI.button(ctl, 'π', () => { S.neg = false; S.mag = Math.log2(Math.PI); quiet(S.mag, false); encode(); upd(); });
      UI.button(ctl, '0,1', () => { S.neg = false; S.mag = Math.log2(0.1); quiet(S.mag, false); encode(); upd(); });
      UI.button(ctl, '70000', () => { S.neg = false; S.mag = Math.log2(70000); quiet(S.mag, false); encode(); upd(); });
      const rV = UI.readout(ctl, 'guardado', '—');
      const rE = UI.readout(ctl, 'erro relativo', '—');
      const rB = UI.readout(ctl, 'bits', '—');
      function upd() {
        rV.set(nice(stored));
        rE.set(Number.isFinite(stored) && xTrue !== 0 ? ((stored - xTrue) / Math.abs(xTrue) * 100).toFixed(2).replace('.', ',') + '%' : '—');
        rB.set(N.toBin(bits, F().bits) + '  (0x' + bits.toString(16).toUpperCase() + ')');
        H && H.redraw();
      }
      function setFromBits(b) {
        bits = b; stored = N.decode(S.fmt, b);
        if (Number.isFinite(stored) && stored !== 0) { xTrue = stored; S.mag = Math.log2(Math.abs(stored)); S.neg = stored < 0; S.zero = false; quiet(S.mag, S.neg); }
        else if (stored === 0) { S.zero = true; xTrue = 0; }
        upd();
      }
      let H = null;
      encode();
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const f = F(); const nb = f.bits; const pad = 16; const narrow = w < 560;
        const cols = { s: C.c, e: C.d, m: C.a };
        V.text(g, narrow ? `${NAMES[S.fmt]} · S1 E${f.e} M${f.m} · bias ${f.bias}` : `${NAMES[S.fmt]} · 1 bit de sinal · ${f.e} de expoente · ${f.m} de mantissa · bias ${f.bias}`, w / 2, 20, { size: 12, font: 'mono', color: C.muted, align: 'center', maxW: w - 2 * pad });
        // --- caixas de bits
        const gapF = narrow ? 7 : 12, gapB = 3;
        const bs = Math.min(44, (w - 2 * pad - 2 * gapF - (nb - 1) * gapB) / nb);
        const totalW = nb * bs + (nb - 1) * gapB + 2 * gapF;
        const y0 = 34, bh = Math.max(28, Math.min(44, bs * 1.05));
        const boxes = []; let x = (w - totalW) / 2;
        for (let pos = 0; pos < nb; pos++) {
          const field = pos === 0 ? 's' : pos <= f.e ? 'e' : 'm';
          if (pos === 1 || pos === 1 + f.e) x += gapF;
          boxes.push({ x, k: nb - 1 - pos, field }); x += bs + gapB;
        }
        const p = H.pointer; let hover = -1;
        boxes.forEach((b, i) => { if (p.inside && p.x >= b.x && p.x <= b.x + bs && p.y >= y0 && p.y <= y0 + bh) hover = i; });
        if (p.clicked && hover >= 0) setFromBits(bits ^ (1 << boxes[hover].k));
        H.canvas.style.cursor = hover >= 0 ? 'pointer' : 'default';
        boxes.forEach((b, i) => {
          const on = (bits >> b.k) & 1; const col = cols[b.field];
          V.box(g, b.x, y0, bs, bh, { r: 6, fill: on ? V.alpha(col, 0.82) : V.alpha(col, 0.12), stroke: col, lw: hover === i ? 2.6 : 1.2 });
          V.text(g, on ? '1' : '0', b.x + bs / 2, y0 + bh / 2 + 1, { size: Math.max(11, Math.min(16, bs * 0.5)), font: 'mono', weight: 600, align: 'center', base: 'middle', color: on ? onDark(C) : C.ink });
        });
        // --- rótulos dos campos
        const fl = N.fields(S.fmt, bits);
        const bin = (v, n) => N.toBin(v, n) + '₂';
        const grp = [
          { a: 0, b: 0, col: cols.s, long: 'S = ' + fl.s, short: 'S=' + fl.s },
          { a: 1, b: f.e, col: cols.e, long: 'E = ' + bin(fl.E, f.e) + ' = ' + fl.E, short: 'E=' + fl.E },
          { a: f.e + 1, b: nb - 1, col: cols.m, long: 'M = ' + bin(fl.M, f.m) + ' = ' + fl.M, short: 'M=' + fl.M },
        ];
        grp.forEach((G) => {
          const xa = boxes[G.a].x, xb = boxes[G.b].x + bs, cx = (xa + xb) / 2;
          V.line(g, xa, y0 + bh + 6, xb, y0 + bh + 6, { color: G.col, lw: 2 });
          const o = { size: 11.5, font: 'mono', color: G.col, align: 'center' };
          const lbl = tw(g, G.long, o) <= (xb - xa) + (G.a === 0 ? 30 : 26) ? G.long : G.short;
          V.text(g, lbl, G.a === 0 ? Math.max(pad + tw(g, lbl, o) / 2, cx) : cx, y0 + bh + 22, o);
        });
        // --- fórmula
        const fy = y0 + bh + 48;
        let formula;
        const pm = Math.pow(2, f.m);
        if (Number.isNaN(stored)) formula = 'padrão reservado → NaN';
        else if (!Number.isFinite(stored)) formula = 'expoente todo 1 e mantissa 0 → ±∞';
        else if (fl.E === 0) formula = `(−1)^${fl.s} · 2^(1−${f.bias}) · ${fl.M}/${pm} = ${nice(stored)}   (subnormal)`;
        else formula = `(−1)^${fl.s} · 2^(${fl.E}−${f.bias}) · (1 + ${fl.M}/${pm}) = ${nice(stored)}`;
        V.text(g, formula, w / 2, fy, { size: narrow ? 12 : 13.5, font: 'mono', align: 'center', color: C.ink, weight: 500, maxW: w - 2 * pad });
        const relE = Number.isFinite(stored) && xTrue !== 0 ? (stored - xTrue) / Math.abs(xTrue) : 0;
        const sat = Number.isFinite(stored) && Math.abs(xTrue) > N.maxFinite(S.fmt) * 1.0001;
        V.text(g, S.zero ? 'x = 0 (todos os bits de expoente e mantissa zerados)' : `x = ${nice(xTrue)}  →  guardado ${nice(stored)}   (erro ${(relE * 100).toFixed(2).replace('.', ',')}%${sat ? ', saturou no máximo' : ''})`, w / 2, fy + 20, { size: 12, font: 'mono', align: 'center', color: sat ? C.red : C.muted, maxW: w - 2 * pad });
        // --- réguas
        const yR0 = fy + 38, yAxis = h - 20;
        const labW = narrow ? 0 : 92, rangeW = narrow ? 0 : 92;
        const xL = pad + labW, xR = w - pad - rangeW;
        const tmin = -17, tmax = 17;
        const X = (tt) => xL + (tt - tmin) / (tmax - tmin) * (xR - xL);
        const rh = (yAxis - 14 - yR0) / rulers.length;
        const tgt = S.zero ? tmin : Math.log2(Math.abs(xTrue));
        S.tx += (tgt - S.tx) * Math.min(1, dt * 7);
        rulers.forEach((r, i) => {
          const yy = yR0 + i * rh + (narrow ? rh * 0.62 : rh * 0.55);
          const sel = r.id === S.fmt; const col = r.col(C);
          if (narrow) V.text(g, r.name + ' · ' + r.range + (sel ? '  ◀' : ''), xL, yy - 11, { size: 11, font: 'mono', color: sel ? col : C.muted, weight: sel ? 600 : 400 });
          else { V.text(g, r.name, pad, yy + 4, { size: 11.5, font: 'mono', color: sel ? col : C.muted, weight: sel ? 600 : 400 }); V.text(g, r.range, xR + 10, yy + 4, { size: 11, font: 'mono', color: sel ? col : C.muted }); }
          V.line(g, xL, yy, xR, yy, { color: V.alpha(C.muted, 0.35), lw: 1 });
          g.strokeStyle = V.alpha(col, sel ? 0.95 : 0.4); g.lineWidth = 1; g.beginPath();
          r.lg.forEach((l) => { if (l < tmin || l > tmax) return; const px = Math.round(X(l)) + 0.5; g.moveTo(px, yy - (sel ? 8 : 6)); g.lineTo(px, yy + (sel ? 8 : 6)); });
          g.stroke();
          if (sel && Number.isFinite(stored) && stored !== 0) { const px = X(Math.log2(Math.abs(stored))); V.glow(g, px, yy, 14, col, 0.9); V.dot(g, px, yy, 4.5, col); }
        });
        if (S.fmt === 'bf16') V.text(g, 'BF16 vai de 2⁻¹³³ a ~2¹²⁸: muito além desta régua (8 bits de expoente, 128 valores por binade)', (xL + xR) / 2, yR0 + 2, { size: 11, align: 'center', color: C.b, font: 'mono', maxW: xR - xL });
        // marcador do valor verdadeiro
        const mx = X(Math.max(tmin, Math.min(tmax, S.tx)));
        V.line(g, mx, yR0 - 4, mx, yAxis - 10, { color: V.alpha(C.c, 0.85), lw: 1.5, dash: [4, 4], dashOffset: -t * 18 });
        g.fillStyle = C.c; g.beginPath(); g.moveTo(mx - 6, yR0 - 10); g.lineTo(mx + 6, yR0 - 10); g.lineTo(mx, yR0 - 2); g.closePath(); g.fill();
        // eixo
        V.line(g, xL, yAxis - 10, xR, yAxis - 10, { color: V.alpha(C.muted, 0.5), lw: 1 });
        [-16, -8, 0, 8, 16].forEach((tt) => { const px = X(tt); V.line(g, px, yAxis - 14, px, yAxis - 6, { color: C.muted, lw: 1 }); V.text(g, tt === 0 ? '1' : '2' + sup(tt), px, yAxis + 6, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
      });
      upd();
    });

    /* ===================================================================
       Figura 2 — Granularidade da escala na matriz
       =================================================================== */
    safe(function figGranular() {
      const fig = $('#fig-granular'); if (!fig) return;
      fitH(fig, (w) => { if (w >= 640) { const c = (w - 48) / 2 / 32; return Math.round(36 + 16 * c + 92); } const c = (w - 32) / 32; return Math.round(30 + 32 * c + 46 + 120); });
      const R = 16, K = 32;
      const errV = (e) => Math.min(1, Math.log10(1 + e * 100) / 2.0043);
      const rng = V.rng(21);
      const base = Array.from({ length: R }, () => Array.from({ length: K }, () => rng.normal()));
      const u = Array.from({ length: R }, () => rng() * 2 - 1);
      const S = { gran: 'tensor', fmt: 'e4m3', out: false, vari: 0.6 };
      let Wm, Q, E, nG, info, zeros = 0, meanErr = 0, ref = 1;
      const gid = (i, j) => (S.gran === 'tensor' ? 0 : S.gran === 'canal' ? i : S.gran === 'b16' ? i * 2 + (j >> 4) : (i >> 3) * 4 + (j >> 3));
      const rectOf = (k) => (S.gran === 'tensor' ? [0, 0, R, K] : S.gran === 'canal' ? [k, 0, 1, K] : S.gran === 'b16' ? [k >> 1, (k & 1) * 16, 1, 16] : [(k >> 2) * 8, (k & 3) * 8, 8, 8]);
      function compute() {
        Wm = base.map((row, i) => row.map((v) => v * 0.02 * Math.pow(2, 4 * S.vari * u[i])));
        const abs = []; Wm.forEach((r) => r.forEach((v) => abs.push(Math.abs(v)))); abs.sort((a, b) => a - b); ref = abs[Math.floor(abs.length * 0.95)] || 1;
        if (S.out) Wm[3][5] = 0.8;
        const mxE = S.fmt === 'e4m3' ? 448 : 6;
        nG = S.gran === 'tensor' ? 1 : S.gran === 'canal' ? R : S.gran === 'b16' ? R * 2 : 8;
        const am = new Array(nG).fill(0);
        for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) { const k = gid(i, j); am[k] = Math.max(am[k], Math.abs(Wm[i][j])); }
        info = am.map((a) => ({ amax: a, s: a / mxE }));
        zeros = 0; let se = 0, cnt = 0;
        Q = Wm.map((row, i) => row.map((v, j) => { const s = info[gid(i, j)].s || 1e-30; return N.encode(S.fmt, v / s).v * s; }));
        E = Wm.map((row, i) => row.map((v, j) => { const q = Q[i][j]; if (q === 0 && v !== 0) zeros++; const e = Math.min(1, Math.abs(q - v) / (Math.abs(v) || 1e-30)); se += e; cnt++; return e; }));
        meanErr = se / cnt;
        rS.set(String(nG)); rZ.set(pct(zeros / (R * K), 0)); rM.set(pct(meanErr, 1));
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Escala', options: [{ v: 'tensor', label: 'por tensor' }, { v: 'canal', label: 'por canal' }, { v: 'b16', label: 'bloco 1×16' }, { v: 'b88', label: 'bloco 8×8' }], value: 'tensor', onChange: (v) => { S.gran = v; compute(); H.redraw(); } });
      UI.seg(ctl, { label: 'Elemento', options: [{ v: 'e4m3', label: 'FP8 E4M3' }, { v: 'e2m1', label: 'FP4 E2M1' }], value: 'e4m3', onChange: (v) => { S.fmt = v; compute(); H.redraw(); } });
      UI.toggle(ctl, { label: 'outlier (40×)', value: false, onChange: (v) => { S.out = v; compute(); H.redraw(); } });
      UI.slider(ctl, { label: 'Variação (binades)', min: 0, max: 1, step: 0.05, value: S.vari, fmt: (v) => '±' + (4 * v).toFixed(1).replace('.', ','), onInput: (v) => { S.vari = v; compute(); H.redraw(); } });
      const rS = UI.readout(ctl, 'escalas', '—');
      const rZ = UI.readout(ctl, 'pesos zerados', '—');
      const rM = UI.readout(ctl, 'erro médio/peso', '—');
      compute();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const pad = 16; const stacked = w < 640;
        const titleH = 18, infoH = 92;
        let pw, cell, px1, py1, px2, py2;
        if (!stacked) { pw = (w - 3 * pad) / 2; cell = Math.min(pw / K, (h - titleH - infoH - 18) / R); const mw = cell * K; px1 = pad + (pw - mw) / 2; px2 = 2 * pad + pw + (pw - mw) / 2; py1 = py2 = titleH + 14; }
        else { pw = w - 2 * pad; cell = Math.min(pw / K, (h - 2 * titleH - infoH - 30) / (2 * R)); const mw = cell * K; px1 = px2 = pad + (pw - mw) / 2; py1 = titleH + 10; py2 = py1 + cell * R + titleH + 18; }
        const gap = cell > 9 ? 1 : 0.5;
        const drawM = (ox, oy, colorOf) => { for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) { g.fillStyle = colorOf(i, j); g.fillRect(ox + j * cell + gap / 2, oy + i * cell + gap / 2, cell - gap, cell - gap); } };
        drawM(px1, py1, (i, j) => V.cmap(Math.max(-1, Math.min(1, Wm[i][j] / ref)), 'div'));
        drawM(px2, py2, (i, j) => V.cmap(errV(E[i][j]), 'heat'));
        V.text(g, 'pesos (miniatura 16 × 32)', px1, py1 - 8, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, 'erro relativo por peso', px2, py2 - 8, { size: 11.5, font: 'mono', color: C.muted });
        // hover
        const p = H.pointer; let hov = -1;
        [[px1, py1], [px2, py2]].forEach(([ox, oy]) => { if (p.inside) { const j = Math.floor((p.x - ox) / cell), i = Math.floor((p.y - oy) / cell); if (i >= 0 && i < R && j >= 0 && j < K) hov = gid(i, j); } });
        const cur = hov >= 0 ? hov : Math.floor(t / 1.3) % nG;
        // contornos dos grupos
        for (let k = 0; k < nG; k++) {
          const [i0, j0, nr, nc] = rectOf(k);
          [[px1, py1], [px2, py2]].forEach(([ox, oy]) => {
            g.strokeStyle = k === cur ? C.c : V.alpha(C.b, 0.85); g.lineWidth = k === cur ? 2.4 : 1.1;
            g.strokeRect(ox + j0 * cell + 0.5, oy + i0 * cell + 0.5, nc * cell - 1, nr * cell - 1);
          });
        }
        if (S.out) { [[px1, py1], [px2, py2]].forEach(([ox, oy]) => { const cx = ox + 5 * cell + cell / 2, cy = oy + 3 * cell + cell / 2; V.glow(g, cx, cy, Math.max(1, cell * 1.6), C.c, 0.5 + 0.5 * Math.sin(t * 4)); }); }
        // legenda da escala de erro (log)
        const ly0 = (stacked ? py2 + R * cell : py1 + R * cell) + 10;
        const lgW = Math.min(260, cell * K), lgX = px2 + cell * K - lgW;
        for (let k = 0; k < 60; k++) { g.fillStyle = V.cmap(k / 59, 'heat'); g.fillRect(lgX + k * lgW / 60, ly0, lgW / 60 + 0.6, 8); }
        [[0, '0%'], [0.01, '1%'], [0.0625, '6,25%'], [0.25, '25%'], [1, '100%']].forEach(([e, lb]) => { const xx = lgX + errV(e) * lgW; V.line(g, xx, ly0 + 8, xx, ly0 + 12, { color: C.muted, lw: 1 }); V.text(g, lb, xx, ly0 + 23, { size: 11, font: 'mono', align: e === 0 ? 'left' : e === 1 ? 'right' : 'center', color: C.muted }); });
        if (!stacked) V.text(g, 'menta = positivo · lavanda = negativo', px1, ly0 + 9, { size: 11, font: 'mono', color: C.muted });
        // info
        const iy = ly0 + 46;
        const I = info[cur];
        const mxE = S.fmt === 'e4m3' ? 448 : 6;
        const infoTxt = `grupo ${cur + 1}/${nG}${hov >= 0 ? '' : ' (varrendo)'} · amax = ${nice(I.amax)} · s = amax/${mxE} = ${nice(I.s)}`;
        const nI = stacked ? wrap(g, infoTxt, w / 2, iy, w - 2 * pad, 15, { size: 12, font: 'mono', align: 'center', color: C.ink, maxLines: 2 }) : (V.text(g, infoTxt, w / 2, iy, { size: 12, font: 'mono', align: 'center', color: C.ink, maxW: w - 2 * pad }), 1);
        const msg = S.fmt === 'e2m1' && S.gran === 'tensor' ? 'FP4 com uma escala só: as linhas pequenas caem abaixo de 0,25·s e viram zero.' : S.fmt === 'e4m3' && S.gran === 'tensor' ? (S.out ? 'O outlier empurra os pequenos para a região subnormal do E4M3, onde o erro relativo cresce.' : 'As ~18 binades do E4M3 comportam a variação entre canais: uma escala por tensor basta.') : S.gran === 'canal' ? 'Uma escala por linha de saída: cada canal usa a régua inteira (GLM, FP8-dense do Qwen).' : 'Blocos pequenos: um outlier só estraga o próprio bloco.';
        wrap(g, msg, w / 2, iy + 4 + nI * 15, w - 2 * pad, 15, { size: 11.5, align: 'center', color: C.muted, maxLines: 3 });
      });
    });

    /* ===================================================================
       Figura 3 — GEMM com escalas por bloco, fatia por fatia
       =================================================================== */
    safe(function figGemm() {
      const fig = $('#fig-gemm'); if (!fig) return;
      fitH(fig, (w) => (w < 640 ? 590 : 364));
      const M = 8, K = 512, NN = 384, KB = 4, NB = 3, SL = 128;
      const rng = V.rng(2126);
      const xMag = Array.from({ length: M }, () => Array.from({ length: KB }, () => Math.pow(2, rng() * 4 - 2)));
      const wMag = Array.from({ length: KB }, () => Array.from({ length: NB }, () => Math.pow(2, rng() * 6 - 3)));
      const X = Array.from({ length: M }, (_, m) => Float64Array.from({ length: K }, (_, k) => rng.normal() * xMag[m][k >> 7]));
      const W = Array.from({ length: K }, (_, k) => Float64Array.from({ length: NN }, (_, n) => rng.normal() * 0.02 * wMag[k >> 7][(n / SL) | 0]));
      // FP8: ativação por bloco 1×128, peso por tile 128×128
      const sx = X.map((row) => Array.from({ length: KB }, (_, b) => { let a = 0; for (let k = b * SL; k < (b + 1) * SL; k++) a = Math.max(a, Math.abs(row[k])); return a / 448; }));
      const sw = Array.from({ length: KB }, (_, b) => Array.from({ length: NB }, (_, c) => { let a = 0; for (let k = b * SL; k < (b + 1) * SL; k++) for (let n = c * SL; n < (c + 1) * SL; n++) a = Math.max(a, Math.abs(W[k][n])); return a / 448; }));
      const qX = X.map((row, m) => Float64Array.from(row, (v, k) => N.encode('e4m3', v / sx[m][k >> 7]).v));
      const qW = W.map((row, k) => Float64Array.from(row, (v, n) => N.encode('e4m3', v / sw[k >> 7][(n / SL) | 0]).v));
      // NVFP4: blocos de 16 ao longo de K, escala E4M3 por bloco e escala global FP32 por tensor
      function nvq(get, len, lines, gAmax) {
        const gS = gAmax / 2688; const deq = [], deqBug = [], scl = [];
        for (let l = 0; l < lines; l++) {
          const d = new Float64Array(len), db = new Float64Array(len), cs = [], ss = [];
          for (let b0 = 0; b0 < len; b0 += 16) { let a = 0; for (let k = b0; k < b0 + 16; k++) a = Math.max(a, Math.abs(get(l, k))); const s = (N.encode('e4m3', a / 6 / gS).v * gS) || 1e-30; ss.push(s); for (let k = b0; k < b0 + 16; k++) cs[k] = N.encode('e2m1', get(l, k) / s).v; }
          for (let k = 0; k < len; k++) { d[k] = cs[k] * ss[k >> 4]; db[k] = cs[k] * ss[((k >> 4) + 1) % ss.length]; }
          deq.push(d); deqBug.push(db); scl.push(ss);
        }
        return { deq, deqBug, scl, g: gS };
      }
      let axm = 0; X.forEach((r) => r.forEach((v) => { axm = Math.max(axm, Math.abs(v)); }));
      let awm = 0; W.forEach((r) => r.forEach((v) => { awm = Math.max(awm, Math.abs(v)); }));
      const nX = nvq((l, k) => X[l][k], K, M, axm);
      const nW = nvq((l, k) => W[k][l], K, NN, awm); // uma "linha" por coluna n do peso
      // parciais por fatia: exato, FP8 certo/errado, NVFP4 certo/errado
      const idx = (m, n, b) => (m * NN + n) * KB + b;
      const PE = new Float64Array(M * NN * KB), P8 = new Float64Array(M * NN * KB);
      const PART = { fp8ok: new Float64Array(M * NN * KB), fp8bug: new Float64Array(M * NN * KB), nvok: new Float64Array(M * NN * KB), nvbug: new Float64Array(M * NN * KB) };
      for (let m = 0; m < M; m++) for (let n = 0; n < NN; n++) {
        const c = (n / SL) | 0, dxw = nW.deq[n], dxb = nW.deqBug[n], dx = nX.deq[m];
        for (let b = 0; b < KB; b++) {
          let e = 0, p = 0, a = 0, ab = 0;
          for (let k = b * SL; k < (b + 1) * SL; k++) { e += X[m][k] * W[k][n]; p += qX[m][k] * qW[k][n]; a += dx[k] * dxw[k]; ab += dx[k] * dxb[k]; }
          const i = idx(m, n, b); PE[i] = e; P8[i] = p;
          PART.fp8ok[i] = sx[m][b] * sw[b][c] * p; PART.fp8bug[i] = sx[m][b] * sw[(b + 1) % KB][c] * p;
          PART.nvok[i] = a; PART.nvbug[i] = ab;
        }
      }
      const Y = new Float64Array(M * NN); let y2 = 0;
      for (let m = 0; m < M; m++) for (let n = 0; n < NN; n++) { let s = 0; for (let b = 0; b < KB; b++) s += PE[idx(m, n, b)]; Y[m * NN + n] = s; y2 += s * s; }
      const yRms = Math.sqrt(y2 / (M * NN));
      const YV = {}, ERRC = {}, REL = {};
      Object.keys(PART).forEach((v) => {
        const P = PART[v], yy = new Float64Array(M * NN); let se = 0;
        for (let m = 0; m < M; m++) for (let n = 0; n < NN; n++) { let s = 0; for (let b = 0; b < KB; b++) s += P[idx(m, n, b)]; yy[m * NN + n] = s; se += (s - Y[m * NN + n]) ** 2; }
        YV[v] = yy; REL[v] = Math.sqrt(se / (M * NN)) / yRms;
        ERRC[v] = Array.from({ length: M }, (_, m) => Array.from({ length: NN / 16 }, (_, j) => { let s = 0; for (let n = j * 16; n < j * 16 + 16; n++) s += Math.abs(yy[m * NN + n] - Y[m * NN + n]); return s / 16 / yRms; }));
      });
      // magnitudes locais (células de 16 valores) para colorir X e W
      const lg = (v) => Math.log2(Math.max(v, 1e-12));
      const magX = X.map((row) => Array.from({ length: K / 16 }, (_, j) => { let s = 0; for (let k = j * 16; k < j * 16 + 16; k++) s += row[k] * row[k]; return lg(Math.sqrt(s / 16)); }));
      const magW = Array.from({ length: K / 16 }, (_, i) => Array.from({ length: NN / 16 }, (_, j) => { let s = 0; for (let k = i * 16; k < i * 16 + 16; k++) for (let n = j * 16; n < j * 16 + 16; n++) s += W[k][n] * W[k][n]; return lg(Math.sqrt(s / 256)); }));
      const rngOf = (A) => { let lo = Infinity, hi = -Infinity; A.forEach((r) => r.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); })); return [lo, hi]; };
      const [xlo, xhi] = rngOf(magX), [wlo, whi] = rngOf(magW);
      const errV = (e) => Math.max(0, Math.min(1, (Math.log10(Math.max(e, 1e-6)) + 3) / 3));
      // saída inicial representativa: grande em módulo e com o erro "bloco errado" típico (sem cancelamento de sorte)
      let best = { m: 2, n: 200, sc: -1 };
      for (let m = 0; m < M; m++) for (let n = 7; n < NN; n += 16) { const y0 = Y[m * NN + n]; if (Math.abs(y0) < yRms) continue; const e = Math.abs(YV.fp8bug[m * NN + n] - y0) / Math.abs(y0); const sc = Math.min(e, REL.fp8bug) - Math.abs(YV.fp8ok[m * NN + n] - y0) / Math.abs(y0); if (sc > best.sc) best = { m, n, sc }; }
      const S = { fmt: 'fp8', bug: false, m: best.m, n: best.n, t0: performance.now() };
      const key = () => (S.fmt === 'fp8' ? 'fp8' : 'nv') + (S.bug ? 'bug' : 'ok');
      const ctl = UI.controls(fig);
      const restart = () => { S.t0 = performance.now(); upd(); };
      UI.seg(ctl, { label: 'Formato', options: [{ v: 'fp8', label: 'FP8 · tile 128×128' }, { v: 'nv', label: 'NVFP4 · bloco 16' }], value: 'fp8', onChange: (v) => { S.fmt = v; restart(); } });
      UI.seg(ctl, { label: 'Escala do peso', options: [{ v: 'ok', label: 'bloco certo' }, { v: 'bug', label: 'bloco errado (vizinho)' }], value: 'ok', onChange: (v) => { S.bug = v === 'bug'; restart(); } });
      UI.button(ctl, '↻ outra saída', () => { S.m = (S.m + 3) % M; S.n = (S.n + 137) % NN; restart(); });
      const rY = UI.readout(ctl, 'y exato → ŷ', '—');
      const rE = UI.readout(ctl, 'erro relativo de Y (todas as saídas)', '—');
      function upd() { const v = key(); rY.set(nice(Y[S.m * NN + S.n], 4) + ' → ' + nice(YV[v][S.m * NN + S.n], 4)); rE.set(pct(REL[v], 1)); }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const pad = 16; const narrow = w < 640; const v = key(); const nv = S.fmt === 'nv';
        const gemmW = narrow ? w - 2 * pad : Math.min(w * 0.55, 440);
        const c = Math.max(3.5, Math.min(6.5, (gemmW - 32) / 56));
        const cm = Math.max(8, c * 1.35);
        const XW = 32 * c, WW = 24 * c, WH = 32 * c, XH = M * cm;
        const gx = pad + 24, wy = 30, wx = gx + XW + 8, xy = wy + WH + 8;
        const el = (performance.now() - S.t0) / 1000;
        const per = 1.7, cyc = el % (per * 5); const stage = Math.floor(cyc / per), u = (cyc % per) / per;
        const cur = Math.min(stage, KB - 1), nu = (S.n / SL) | 0, ncur = S.n >> 4;
        // --- matrizes
        for (let m = 0; m < M; m++) for (let j = 0; j < 32; j++) { g.fillStyle = V.cmap((magX[m][j] - xlo) / (xhi - xlo || 1), 'seq'); g.fillRect(gx + j * c, xy + m * cm, c - 0.6, cm - 0.8); }
        for (let i = 0; i < 32; i++) for (let j = 0; j < 24; j++) { g.fillStyle = V.cmap((magW[i][j] - wlo) / (whi - wlo || 1), 'seq'); g.fillRect(wx + j * c, wy + i * c, c - 0.6, c - 0.6); }
        for (let m = 0; m < M; m++) for (let j = 0; j < 24; j++) { g.fillStyle = V.cmap(errV(ERRC[v][m][j]), 'heat'); g.fillRect(wx + j * c, xy + m * cm, c - 0.6, cm - 0.8); }
        // cortes de 128
        g.strokeStyle = V.alpha(C.b, 0.9); g.lineWidth = 1.2; g.beginPath();
        for (let b = 1; b < KB; b++) { g.moveTo(gx + b * 8 * c, xy - 3); g.lineTo(gx + b * 8 * c, xy + XH + 3); g.moveTo(wx - 3, wy + b * 8 * c); g.lineTo(wx + WW + 3, wy + b * 8 * c); }
        for (let b = 1; b < NB; b++) { g.moveTo(wx + b * 8 * c, wy - 3); g.lineTo(wx + b * 8 * c, wy + WH + 3); g.moveTo(wx + b * 8 * c, xy - 3); g.lineTo(wx + b * 8 * c, xy + XH + 3); }
        g.stroke();
        [[gx, xy, XW, XH], [wx, wy, WW, WH], [wx, xy, WW, XH]].forEach(([x, y, ww, hh]) => V.box(g, x, y, ww, hh, { r: 2, stroke: V.alpha(C.muted, 0.7), lw: 1 }));
        // destaque da fatia κ (faixa inteira) e dos blocos que entram na conta
        const kx = gx + cur * 8 * c, ky = wy + cur * 8 * c;
        g.fillStyle = V.alpha(C.c, 0.1); g.fillRect(kx, xy, 8 * c, XH); g.fillRect(wx, ky, WW, 8 * c);
        const pulse = 0.6 + 0.4 * Math.sin(t * 5);
        V.box(g, kx, xy + S.m * cm - 1, 8 * c, cm + 1, { r: 2, stroke: C.c, lw: 2.2 });
        V.box(g, wx + nu * 8 * c, ky, 8 * c, 8 * c, { r: 2, stroke: C.c, lw: 2.2 });
        V.glow(g, wx + ncur * c + c / 2, xy + S.m * cm + cm / 2, 10, C.ink, 0.5 * pulse);
        V.box(g, wx + ncur * c - 1, xy + S.m * cm - 1, c + 2, cm + 1, { r: 2, stroke: C.ink, lw: 2 });
        if (nv) { // blocos de 16 dentro da fatia: 8 por fatia, agrupados de 4 em 4 (uma instrução k64)
          g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 0.8; g.beginPath();
          for (let j = 1; j < 8; j++) { const xx = kx + j * c; g.moveTo(xx, xy + S.m * cm); g.lineTo(xx, xy + S.m * cm + cm); const yy = ky + j * c; g.moveTo(wx + nu * 8 * c, yy); g.lineTo(wx + nu * 8 * c + 8 * c, yy); }
          g.stroke();
          V.line(g, kx + 4 * c, xy + S.m * cm - 4, kx + 4 * c, xy + S.m * cm + cm + 4, { color: C.c, lw: 2 });
          V.line(g, wx + nu * 8 * c - 4, ky + 4 * c, wx + nu * 8 * c + 8 * c + 4, ky + 4 * c, { color: C.c, lw: 2 });
        }
        if (S.bug && stage < KB) { // escala tirada do tile vizinho em K
          const nb = (cur + 1) % KB, sxp = wx + nu * 8 * c + 8 * c, y1 = wy + nb * 8 * c + 4 * c, y2 = ky + 4 * c;
          V.box(g, wx + nu * 8 * c, wy + nb * 8 * c, 8 * c, 8 * c, { r: 2, stroke: C.red, lw: 1.6 });
          V.arrow(g, sxp + 2, y1, sxp + 2, y2, { color: C.red, lw: 1.6, head: 6, curve: nb > cur ? 22 : -22 });
        }
        // rótulos
        V.text(g, 'W [512 × 384]', wx, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: WW + 16 });
        V.text(g, 'X [8 × 512]', gx, xy + XH + 15, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, 'Y = X·W [8 × 384]', wx, xy + XH + 15, { size: 11.5, font: 'mono', color: C.muted, maxW: WW + 60 });
        V.text(g, 'K →', gx + XW / 2, xy - 6, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, 'K ↓', wx - 6, wy + WH - 8, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // quadrante vazio: legenda
        { const lx = gx, ly = wy + 8, lw = XW - 14; const L = [];
          L.push(['fatia κ = ' + (stage < KB ? cur : '0…3') + (nv ? (narrow ? ' · 8 × 16' : ' · 8 blocos de 16') : ' · 128 de K'), C.c]);
          L.push([nv ? (narrow ? 'escalas por 16' : 'escalas: 1 por bloco 1×16 (X) e 16×1 (W)') : (narrow ? 'sˣ: 1×128 · sʷ: 128²' : 'escalas: sˣ por 1×128, sʷ por 128×128'), C.ink]);
          if (!narrow) L.push(['X: ativação · W: peso', C.muted]);
          L.push([narrow ? 'X, W: magnitude' : 'cor de X e W: magnitude local', C.muted]);
          L.push([narrow ? 'Y: erro' : 'cor de Y: erro da saída', C.muted]);
          let yy = ly; L.forEach(([tx, col]) => { yy += wrap(g, tx, lx, yy, lw, 14, { size: 11, font: 'mono', color: col, maxLines: 2 }) * 14 + 3; });
          const bw = Math.min(lw, 150); for (let k = 0; k < 30; k++) { g.fillStyle = V.cmap(k / 29, 'heat'); g.fillRect(lx + k * bw / 30, yy + 2, bw / 30 + 0.5, 7); }
          V.text(g, '0,1%', lx, yy + 22, { size: 11, font: 'mono', color: C.muted }); V.text(g, '100%', lx + bw, yy + 22, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        }
        // --- painel: tensor core + acumulador
        const px0 = narrow ? pad : wx + WW + 26, pw = narrow ? w - 2 * pad : w - px0 - pad, py0 = narrow ? xy + XH + 30 : 12;
        const tcH = 104;
        const i0 = idx(S.m, S.n, cur);
        const P = PART[v]; const cK = P[i0], eK = PE[i0];
        const res = stage === KB;
        V.box(g, px0, py0, pw, tcH, { r: 12, fill: V.alpha(res ? C.d : C.a, 0.08 + 0.05 * (u < 0.45 && !res ? 1 : 0)), stroke: res ? C.d : C.a, lw: 1.4 });
        const tt = res ? 'acumulador FP32 · saída y[' + S.m + ', ' + S.n + ']' : (nv ? 'tensor core FP4 block-scaled · κ = ' : 'tensor core FP8 · κ = ') + cur;
        V.text(g, tt, px0 + 12, py0 + 18, { size: 12, font: 'mono', weight: 600, color: res ? C.d : C.a, maxW: pw - 24 });
        const lines = [];
        if (res) {
          const yh = YV[v][S.m * NN + S.n], ye = Y[S.m * NN + S.n];
          lines.push(['ŷ = Σκ cκ = ' + nice(yh, 4), C.ink]); lines.push(['y exato = ' + nice(ye, 4), C.muted]);
          const eo = Math.abs(yh - ye) / (Math.abs(ye) || yRms);
          lines.push(['erro desta saída: ' + pct(eo, 1) + (S.bug && eo < 0.25 ? ' (as fatias se cancelaram por sorte; Y inteiro: ' + pct(REL[v], 0) + ')' : ''), S.bug ? C.red : C.ink]);
        } else if (!nv) {
          lines.push(['Σ₁₂₈ q̂ˣ·q̂ʷ = ' + nice(P8[i0]) + ' (em códigos)', C.ink]);
          lines.push(['× sˣ ' + nice(sx[S.m][cur]) + ' × sʷ ' + nice(S.bug ? sw[(cur + 1) % KB][nu] : sw[cur][nu]) + (S.bug ? ' (vizinho!)' : ''), S.bug ? C.red : C.ink]);
          lines.push(['→ cκ = ' + nice(cK, 4) + '   (exato ' + nice(eK, 4) + ')', C.ink]);
        } else {
          lines.push(['128 de K = 8 blocos de 16 = 2 × mma k64', C.ink]);
          lines.push(['cada k64: 4 escalas ue4m3 de cada lado (scale_vec::4X)' + (S.bug ? ', do bloco vizinho!' : ''), S.bug ? C.red : C.ink]);
          lines.push(['→ cκ = ' + nice(cK, 4) + '   (exato ' + nice(eK, 4) + ')', C.ink]);
        }
        let ly = py0 + 38; lines.forEach(([tx, col]) => { ly += wrap(g, tx, px0 + 12, ly, pw - 24, 14, { size: 11.5, font: 'mono', color: col, maxLines: 2 }) * 14 + 4; });
        // partículas: da linha de X e do tile de W até o tensor core
        if (!res && u < 0.5) {
          const e = V.ease.inOut(u / 0.5); const tx = px0 + 18, ty = py0 + tcH / 2;
          for (let q = 0; q < 6; q++) {
            const sx0 = kx + (q + 0.5) * 8 * c / 6, sy0 = xy + S.m * cm + cm / 2; const ex = wx + nu * 8 * c + (q + 0.5) * 8 * c / 6, ey = ky + 4 * c;
            V.dot(g, V.lerp(sx0, tx, e), V.lerp(sy0, ty, e) - Math.sin(e * Math.PI) * 30, 2.6, V.alpha(C.c, 0.9));
            V.dot(g, V.lerp(ex, tx, e), V.lerp(ey, ty, e) - Math.sin(e * Math.PI) * 20, 2.6, V.alpha(C.a, 0.9));
          }
        }
        // barras do acumulador
        const ay0 = py0 + tcH + 22, aH = h - ay0 - 26;
        V.text(g, 'contribuições cκ (exato em contorno) e a soma', px0, ay0 - 6, { size: 11, font: 'mono', color: C.muted, maxW: pw });
        const cols5 = [0, 1, 2, 3, 4]; const colW = pw / 5;
        const contrib = [0, 1, 2, 3].map((b) => P[idx(S.m, S.n, b)]), exact = [0, 1, 2, 3].map((b) => PE[idx(S.m, S.n, b)]);
        const yh = YV[v][S.m * NN + S.n], ye = Y[S.m * NN + S.n];
        const ref = Math.max(...exact.map(Math.abs), Math.abs(ye), 1e-9) * 1.25;
        const base = ay0 + aH / 2, half = aH / 2 - 8;
        V.line(g, px0, base, px0 + pw, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
        cols5.forEach((j) => {
          const x = px0 + j * colW + colW * 0.2, bw = colW * 0.6;
          const shown = j < 4 ? stage > j || (stage === j && u > 0.45) : res;
          const grow = j < 4 ? (stage === j ? V.ease.out(Math.min(1, (u - 0.45) / 0.4)) : 1) : V.ease.out(Math.min(1, u / 0.5));
          const ex = j < 4 ? exact[j] : ye, qv = j < 4 ? contrib[j] : yh;
          const eh = ex / ref * half; g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.2; g.setLineDash([3, 2]); g.strokeRect(x, base - Math.max(0, eh), bw, Math.abs(eh)); g.setLineDash([]);
          if (shown) {
            const clip = Math.abs(qv) > ref; const qh = Math.max(-1, Math.min(1, qv / ref)) * half * grow;
            const off = Math.abs(qv - ex) > 0.25 * Math.abs(ex) + 0.02 * ref;
            g.fillStyle = V.alpha(off ? C.red : j < 4 ? C.a : C.d, 0.75); g.fillRect(x + bw * 0.18, base - Math.max(0, qh), bw * 0.64, Math.abs(qh));
            if (clip) { const yy = qv > 0 ? base - half - 2 : base + half + 2; g.fillStyle = C.red; g.beginPath(); g.moveTo(x + bw / 2 - 5, yy + (qv > 0 ? 6 : -6)); g.lineTo(x + bw / 2 + 5, yy + (qv > 0 ? 6 : -6)); g.lineTo(x + bw / 2, yy); g.closePath(); g.fill(); V.text(g, '×' + nice(Math.abs(qv / (ex || 1e-9)), 1), x + bw / 2, qv > 0 ? yy - 4 : yy + 13, { size: 11, font: 'mono', align: 'center', color: C.red }); }
          }
          V.text(g, j < 4 ? 'κ' + j : 'Σ', x + bw / 2, ay0 + aH + 12, { size: 11, font: 'mono', align: 'center', color: j === cur && !res ? C.c : C.muted, weight: j === cur && !res ? 600 : 400 });
        });
        const p = H.pointer;
        if (p.inside) { const j = Math.floor((p.x - wx) / c), m = Math.floor((p.y - xy) / cm); const onY = j >= 0 && j < 24 && m >= 0 && m < M; H.canvas.style.cursor = onY ? 'pointer' : 'default'; if (p.clicked && onY) { S.m = m; S.n = j * 16 + 7; restart(); } }
      });
    });

    /* ===================================================================
       Figura 4 — FP8 por canal, passo a passo (stepper)
       =================================================================== */
    safe(function figCanal() {
      const fig = $('#fig-canal'); if (!fig) return;
      const CH3 = 56, CY3 = 22, GAP3 = 8;
      fitH(fig, (w) => { const rows = w < 560 ? 2 : 1; return CY3 + rows * (CH3 + GAP3) + 70 + 50; });
      const w8 = [0.0213, -0.0071, 0.0450, -0.0132, 0.0009, -0.0388, 0.0167, 0.0302];
      const s = 0.0450 / 448;
      const r = w8.map((v) => v / s);
      const q = r.map((v) => N.encode('e4m3', v));
      const deq = q.map((e) => e.v * s);
      const rel = relRMSE(w8, deq);
      const grid = N.values('e4m3');
      const steps = [
        { t: 'A linha de pesos', d: 'Oito pesos de um canal de saída (uma linha de <code>weight[out, in]</code>). Em BF16 são 16 bytes; em FP8 serão 8 bytes + 1 escala compartilhada pela linha inteira (4.096 colunas no modelo real).' },
        { t: 'amax e escala', d: 'O maior valor absoluto é 0,0450. A escala \\(s = 0{,}0450/448 = 1{,}0045\\cdot10^{-4}\\) leva esse peso exatamente ao topo do E4M3. No GLM ela é guardada em BF16; no Qwen, em FP32.' },
        { t: 'Dividir pela escala', d: 'Agora os valores vivem na régua do E4M3: 212,05; −70,68; 448; −131,41; 8,96; −386,28; 166,26; 300,66. Nenhum passa de 448, então nada satura.' },
        { t: 'Arredondar na grade', d: 'Cada valor vai ao vizinho mais próximo. A grade tem passo 8 em [64, 128), 16 em [128, 256) e 32 em [256, 448]: 212,05 → 208, −131,41 → −128, 300,66 → 288, 8,96 → 9.' },
        { t: 'Os bits', d: 'Cada código é 1 byte: 208 = <code>0.1110.101</code> (\\(2^{14-7}\\times1{,}625\\)), 448 = <code>0.1111.110</code>, −384 = <code>1.1111.100</code>. O byte e a escala são tudo o que vai para a memória.' },
        { t: 'Desquantizar e medir', d: '\\(\\hat w = \\text{código}\\times s\\). O RMSE relativo da linha é 2,09%. Nenhum peso erra mais que 6,25% do próprio valor, porque o erro do float é relativo. No Qwen, sobre 591 tensores reais, a mediana foi 2,62%.' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const pad = 16; const i = st.i; const el = st.elapsed();
        const e = V.ease.inOut(Math.min(1, el / 0.9));
        const cols = w < 560 ? 4 : 8, rows = 8 / cols, gap = GAP3;
        const cw = (w - 2 * pad - (cols - 1) * gap) / cols, ch = CH3, cy0 = CY3;
        const cellXY = (k) => [pad + (k % cols) * (cw + gap), cy0 + Math.floor(k / cols) * (ch + gap)];
        const fs = Math.min(13, cw / 5.6);
        for (let k = 0; k < 8; k++) {
          const [x, y] = cellXY(k);
          const isMax = k === 2 && i >= 1;
          V.box(g, x, y, cw, ch, { r: 8, fill: V.alpha(i >= 4 ? C.a : C.b, 0.1), stroke: isMax ? C.c : V.alpha(C.muted, 0.45), lw: isMax ? 2.4 : 1 });
          V.text(g, 'w' + String.fromCharCode(0x2080 + k), x + 6, y + 13, { size: 11, font: 'mono', color: C.muted });
          let main = '';
          if (i <= 1) main = nice(w8[k], 4);
          else if (i === 2) main = nice(r[k], 2);
          else if (i === 3) main = nice(q[k].v);
          else if (i === 5) main = nice(deq[k], 6);
          if (i === 4) {
            const b = N.toBin(q[k].bits, 8); const segs = [[b[0], C.c], ['.' + b.slice(1, 5), C.d], ['.' + b.slice(5), C.a]];
            const o = { size: Math.max(11, Math.min(13, cw / 7.4)), font: 'mono', weight: 600 };
            const total = segs.reduce((a, sg) => a + tw(g, sg[0], o), 0); let xx = x + cw / 2 - total / 2;
            segs.forEach(([txt, col]) => { V.text(g, txt, xx, y + ch / 2 + 8, Object.assign({ color: col, base: 'middle' }, o)); xx += tw(g, txt, o); });
          } else V.text(g, main, x + cw / 2, y + ch / 2 + 8, { size: fs, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 500 });
          if (i === 5) { const re = Math.abs(deq[k] - w8[k]) / Math.abs(w8[k]); const bw = Math.min(1, re / 0.0625) * (cw - 12); g.fillStyle = V.alpha(C.d, 0.8); V.rr(g, x + 6, y + ch - 7, Math.max(2, bw), 4, 2); g.fill(); }
        }
        const cyEnd = cy0 + rows * (ch + gap);
        if (i === 0) V.text(g, 'BF16: 8 × 2 B = 16 B   →   FP8: 8 × 1 B + 1 escala por linha', w / 2, cyEnd + 12, { size: 12, font: 'mono', align: 'center', color: C.muted, maxW: w - 2 * pad });
        if (i >= 1) V.text(g, i === 1 ? 's = amax / 448 = 0,0450 / 448 = 1,0045·10⁻⁴' : i === 5 ? `RMSE relativo = ${pct(rel, 2)} · barras: erro relativo de cada peso (cheia = 6,25%)` : 's = 1,0045·10⁻⁴ (uma escala para a linha toda)', w / 2, cyEnd + 12, { size: 12, font: 'mono', align: 'center', color: i === 1 ? C.c : C.muted, maxW: w - 2 * pad });
        // régua E4M3 linear
        const ry = cyEnd + 70, xL = pad + 8, xR = w - pad - 8;
        const X = (v) => xL + (v + 448) / 896 * (xR - xL);
        if (i >= 3) {
          [[64, 128, 'passo 8'], [128, 256, 'passo 16'], [256, 448, 'passo 32']].forEach(([a, b, lb], k) => {
            const col = [C.e, C.d, C.c][k];
            [[a, b], [-b, -a]].forEach(([u0, u1]) => { g.fillStyle = V.alpha(col, 0.13); g.fillRect(X(u0), ry - 26, X(u1) - X(u0), 22); });
            V.text(g, lb, (X(a) + X(b)) / 2, ry - 11, { size: 11, align: 'center', font: 'mono', color: col });
          });
        }
        V.line(g, xL, ry, xR, ry, { color: V.alpha(C.muted, 0.5), lw: 1 });
        g.strokeStyle = V.alpha(C.a, 0.6); g.lineWidth = 1; g.beginPath();
        grid.forEach((v) => { const px = Math.round(X(v)) + 0.5; g.moveTo(px, ry - 4); g.lineTo(px, ry + 4); }); g.stroke();
        [-448, -256, -128, 0, 128, 256, 448].forEach((v) => { if (w < 480 && (v === -128 || v === 128)) return; V.text(g, nice(v), X(v), ry + 18, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        V.text(g, 'régua do E4M3 (eixo linear): tracinhos densos perto de 0, espaçados no topo', w / 2, ry + 36, { size: 11, align: 'center', color: C.muted, maxW: w - 2 * pad });
        // pontos
        if (i >= 2) {
          for (let k = 0; k < 8; k++) {
            const [x, y] = cellXY(k); const sx = x + cw / 2, sy = y + ch;
            let px, py;
            if (i === 2) { px = V.lerp(sx, X(r[k]), e); py = V.lerp(sy, ry, e); }
            else if (i === 3) { px = V.lerp(X(r[k]), X(q[k].v), e); py = ry; }
            else { px = X(q[k].v); py = ry; }
            if (i === 2 && e < 1) V.line(g, sx, sy, px, py, { color: V.alpha(C.b, 0.35), lw: 1, dash: [3, 3] });
            V.glow(g, px, py, 10, i >= 3 ? C.a : C.b, 0.7); V.dot(g, px, py, 3.6, i >= 3 ? C.a : C.b);
          }
        }
      });
    });

    /* ===================================================================
       Figura 5 — Guardado × lido por passo (orçamento de bytes do decode)
       =================================================================== */
    safe(function figBytes() {
      const fig = $('#fig-bytes'); if (!fig) return;
      let need = 0;
      const refit = fitH(fig, (w) => need || (w < 640 ? 470 : 360));
      // cores por componente: exp (experts), ple, dense (densos BF16, âmbar quando FP8), mtp, other
      const DATA = {
        qwen: {
          unit: 'GiB', toGB: 1.073741824, bw: 245, bwTxt: '245 GB/s (topo dos 235–245 efetivos)', expCol: 'a',
          stored: [
            { k: 'exp', name: 'experts roteados NVFP4', v: 63.28 },
            { k: 'ple', name: 'tabela PLE (FP8)', v: 47.68 },
            { k: 'dense', name: 'densos BF16 (+ embed e lm_head)', v: 9.22 },
            { k: 'mtp', name: 'MTP', v: 2.51 },
            { k: 'vis', name: 'visão', v: 0.84 },
          ],
          read: [
            { k: 'exp', name: 'experts (~32 por camada)', v: 2.1 },
            { k: 'dense', name: 'lm_head ×4', v: 2.36, q: true },
            { k: 'dense', name: 'GDN', v: 1.95, q: true },
            { k: 'dense', name: 'HyperConnection', v: 1.19, q: true },
            { k: 'dense', name: 'atenção + indexer', v: 0.60, q: true },
            { k: 'dense', name: 'roteador + shared + PLE proj', v: 0.40, q: true },
            { k: 'mtp', name: 'MTP (denso + experts) ×3', v: 0.5, q: true },
            { k: 'state', name: 'estado GDN', v: 0.1 },
          ],
          storedTxt: 'guardado · checkpoint nvidia inteiro', readTxt: 'lido por passo · por GPU (TP2/EP2, batch 1, MTP3)',
          msg: (on) => (on ? 'FP8 nos densos: ~6,9 → ~3,45 GiB, e o passo cai de ≈ 9,2 para ≈ 5,7 GiB (~40 → ~25 ms). É a análise do relatório, feita antes do checkpoint existir e nunca medida em GPU.' : 'Experts: 51% do guardado e só ~23% do lido. PLE: 39% do guardado e ≈ 0 do lido (16 linhas por token). Os densos BF16 são lidos inteiros em todo passo: ~6,9 dos 9,2 GiB, e o lm_head 4 vezes por causa do MTP.'),
        },
        glm: {
          unit: 'GB', toGB: 1, bw: 230, bwTxt: '230 GB/s efetivos', expCol: 'b',
          stored: [
            { k: 'exp', name: 'experts roteados EXL3 4 bpw', v: 156.29 },
            { k: 'dense', name: 'densos BF16 (KDA, MLA, shared, MLP, embed, head)', v: 17.68 },
            { k: 'other', name: 'visão + MTP (não roteado)', v: 1.5 },
          ],
          read: [
            { k: 'exp', name: 'experts (~57 de 288 por camada)', v: 15 },
            { k: 'dense', name: 'KDA', v: 4.7, q: true },
            { k: 'dense', name: 'MLA + indexer', v: 1.3 },
            { k: 'dense', name: 'shared expert', v: 1.1 },
            { k: 'dense', name: 'lm_head', v: 0.6 },
            { k: 'dense', name: 'MLP denso (camadas 0–2)', v: 0.45, q: true },
          ],
          storedTxt: 'guardado · checkpoint EXL3 inteiro', readTxt: 'lido por passo · por rank (TP2, passo DFlash2: 8 tokens × top-8)',
          msg: (on) => (on ? 'GLM53_DENSE_FP8=dense,kda: KDA 4,7 → 2,35 GB e MLP denso 0,45 → 0,23 GB, um corte de ~2,6 GB ≈ 11 ms, a mesma ordem dos −11 ms/passo medidos. O default "all" também converte shared e MLA (−4,7% ms/ciclo medido).' : 'Aqui os experts dominam também a leitura: o DFlash2 verifica 64 slots por camada e toca ~57 dos 288 experts. Mesmo assim os densos BF16 (~8 GB por rank, ~35 ms) são um terço do passo, e são lidos inteiros sempre.'),
        },
      };
      const S = { r: 'qwen', on: false, cur: null };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'qwen', label: 'Qwen3.8' }, { v: 'glm', label: 'GLM-5.3' }], value: 'qwen', onChange: (v) => { S.r = v; S.cur = null; upd(); } });
      UI.toggle(ctl, { label: 'densos BF16 → FP8 (FP8_DENSE / GLM53_DENSE_FP8)', value: false, onChange: (v) => { S.on = v; upd(); } });
      const rR = UI.readout(ctl, 'lido por passo', '—');
      const rT = UI.readout(ctl, 'tempo só de banda', '—');
      const tgt = () => DATA[S.r].read.map((s) => (S.on && s.q ? s.v / 2 : s.v));
      function upd() { const D = DATA[S.r]; const tot = tgt().reduce((a, b) => a + b, 0); rR.set(nice(tot, 2) + ' ' + D.unit); rT.set('≈ ' + Math.round(tot * D.toGB / D.bw * 1000) + ' ms'); }
      upd();
      const colOf = (C, D, k, fp8) => (k === 'exp' ? C[D.expCol] : k === 'ple' ? C.f : k === 'dense' ? (fp8 ? C.d : C.muted) : k === 'mtp' ? C.c : k === 'state' ? C.b : C.e);
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const pad = 16; const narrow = w < 640; const D = DATA[S.r];
        const T = tgt(); if (!S.cur || S.cur.length !== T.length) S.cur = D.read.map((s) => s.v);
        S.cur = S.cur.map((v, i) => v + (T[i] - v) * Math.min(1, dt * 5));
        const xL = pad, xR = w - pad, W0 = xR - xL;
        const sTot = D.stored.reduce((a, s) => a + s.v, 0), rMax = D.read.reduce((a, s) => a + s.v, 0), rTot = S.cur.reduce((a, b) => a + b, 0);
        // barra 1: guardado (100%)
        const y1 = 30, bh = 26;
        V.text(g, D.storedTxt.toUpperCase(), xL, 18, { size: 11, font: 'mono', color: C.muted, maxW: W0 - 90 });
        V.text(g, nice(sTot, 1) + ' ' + D.unit, xR, 18, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'right' });
        const sSpan = {}; let x = xL;
        D.stored.forEach((s) => {
          const ww = W0 * s.v / sTot; const col = colOf(C, D, s.k, false);
          V.box(g, x + 0.5, y1, Math.max(1, ww - 1), bh, { r: 4, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
          const lb = pct(s.v / sTot, 0); if (ww > tw(g, lb, { size: 11.5, font: 'mono' }) + 10) V.text(g, lb, x + ww / 2, y1 + bh / 2 + 1, { size: 11.5, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
          sSpan[s.k] = sSpan[s.k] ? [sSpan[s.k][0], x + ww] : [x, x + ww]; x += ww;
        });
        // barra 2: lido por passo (escala absoluta; a largura cheia é o caso BF16)
        const bandH = narrow ? 64 : 70, y2 = y1 + bh + bandH + 22;
        V.text(g, D.readTxt.toUpperCase(), xL, y2 - 8, { size: 11, font: 'mono', color: C.muted, maxW: W0 - 110 });
        V.text(g, nice(rTot, 2) + ' ' + D.unit, xR, y2 - 8, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'right' });
        const rSpan = {}; x = xL; const segs = [];
        D.read.forEach((s, i) => {
          const ww = W0 * S.cur[i] / rMax; const fp8 = S.on && s.q; const col = colOf(C, D, s.k, fp8);
          V.box(g, x + 0.5, y2, Math.max(1, ww - 1), bh, { r: 4, fill: V.alpha(col, fp8 ? 0.7 : 0.55), stroke: col, lw: 1 });
          segs.push({ x, ww, s, col });
          rSpan[s.k] = rSpan[s.k] ? [rSpan[s.k][0], x + ww] : [x, x + ww]; x += ww;
        });
        if (S.on) { V.line(g, xL + W0 * rTot / rMax, y2 - 4, xL + W0 * rTot / rMax, y2 + bh + 4, { color: C.d, lw: 1.5 }); g.setLineDash([4, 4]); V.box(g, xL + W0 * rTot / rMax, y2, W0 * (1 - rTot / rMax), bh, { r: 4, stroke: V.alpha(C.muted, 0.5), lw: 1 }); g.setLineDash([]); }
        // partículas: o fluxo de bytes que o passo lê
        for (let q = 0; q < 34; q++) { const u = ((q * 0.137 + t * 0.35) % 1); const px = xL + u * W0 * rTot / rMax; const py = y2 + 5 + (q % 4) * (bh - 10) / 3; V.dot(g, px, py, 1.6, V.alpha(C.ink, 0.55)); }
        // faixas ligando os componentes
        const yb0 = y1 + bh + 3, yb1 = y2 - 22;
        D.stored.forEach((s) => {
          const a = sSpan[s.k]; if (!a) return; const col = colOf(C, D, s.k, false);
          const ym = (yb0 + yb1) / 2, mid = (a[0] + a[1]) / 2;
          const b = rSpan[s.k] || [mid, mid]; const ye = rSpan[s.k] ? yb1 : ym - 6; // o que não é lido some no meio do caminho
          const ymc = (yb0 + ye) / 2;
          g.fillStyle = V.alpha(col, 0.16); g.beginPath(); g.moveTo(a[0], yb0); g.bezierCurveTo(a[0], ymc, b[0], ymc, b[0], ye); g.lineTo(b[1], ye); g.bezierCurveTo(b[1], ymc, a[1], ymc, a[1], yb0); g.closePath(); g.fill();
          if (!rSpan[s.k] && a[1] - a[0] > 40) V.text(g, s.k === 'ple' ? 'PLE: ≈ 0 lido (16 linhas/token)' : '≈ 0 lido', Math.min(mid, xR - 110), ym + 10, { size: 11, font: 'mono', align: 'center', color: col, maxW: 220 });
        });
        // eixo de tempo (a barra 2 em ms, pela banda efetiva)
        const ay = y2 + bh + 16; const msMax = rMax * D.toGB / D.bw * 1000; const step = msMax > 60 ? 20 : 10;
        V.line(g, xL, ay, xR, ay, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let ms = 0; ms <= msMax + 0.01; ms += step) { const px = xL + W0 * ms / msMax; V.line(g, px, ay - 3, px, ay + 3, { color: C.muted, lw: 1 }); V.text(g, ms + ' ms', px, ay + 15, { size: 11, font: 'mono', align: ms === 0 ? 'left' : px > xR - 24 ? 'right' : 'center', color: C.muted }); }
        V.text(g, 'tempo só de banda, a ' + D.bwTxt, xR, ay + 31, { size: 11, font: 'mono', align: 'right', color: C.muted, maxW: W0 });
        // legenda do passo
        const ly0 = ay + 52; const nc = narrow ? 1 : 2; const cwid = W0 / nc; let row = 0;
        segs.forEach((sg, i) => {
          const cx = xL + (i % nc) * cwid, cy = ly0 + Math.floor(i / nc) * 16; row = Math.floor(i / nc);
          V.box(g, cx, cy - 9, 10, 10, { r: 2, fill: sg.col });
          const vv = S.cur[i];
          V.text(g, `${sg.s.name} · ${nice(vv, 2)} ${D.unit} (${pct(vv / rTot, 0)})`, cx + 16, cy, { size: 11.5, font: 'mono', color: C.ink, maxW: cwid - 22 });
        });
        const my = ly0 + (row + 1) * 16 + 8;
        const nl = wrap(g, D.msg(S.on), xL, my, W0, 15, { size: 12, color: S.on ? C.d : C.ink, maxLines: 5 });
        // altura: pelo pior caso das duas mensagens, para não pular ao trocar o toggle
        const nMax = Math.max(splitLines(g, D.msg(true), W0, { size: 12 }).length, splitLines(g, D.msg(false), W0, { size: 12 }).length, nl);
        const rowsMax = Math.max(...Object.values(DATA).map((d) => Math.ceil(d.read.length / nc)));
        const nd = Math.round(ly0 + rowsMax * 16 + 8 + (Math.min(5, nMax) - 1) * 15 + 16);
        if (Math.abs(nd - need) > 2) { need = nd; refit(); }
      });
    });

    /* ===================================================================
       Figura 6 — E8M0: piso, teto ou fracionária
       =================================================================== */
    safe(function figE8m0() {
      const fig = $('#fig-e8m0'); if (!fig) return;
      fitH(fig, (w) => (w < 560 ? 470 : 404));
      const S = { t: Math.log2(3.7), fmt: 'e4m3', bug: false, t0: 0, rs: [null, null, null] };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'amax do bloco', min: -8, max: 12, step: 0.01, value: S.t, fmt: (v) => nice(Math.pow(2, v)), onInput: (v) => { S.t = v; S.t0 = performance.now(); } });
      UI.seg(ctl, { label: 'Elemento', options: [{ v: 'e4m3', label: 'E4M3 (máx 448)' }, { v: 'e2m1', label: 'E2M1 (máx 6)' }], value: 'e4m3', onChange: (v) => { S.fmt = v; S.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'carregar com copy_() numérico (bug)', value: false, onChange: (v) => { S.bug = v; } });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const pad = 16; const narrow = w < 560;
        const amax = Math.pow(2, S.t);
        const mx = S.fmt === 'e4m3' ? 448 : 6, emax = S.fmt === 'e4m3' ? 8 : 2;
        const Xf = Math.floor(Math.log2(amax)) - emax, Xc = Math.ceil(Math.log2(amax / mx));
        const lanes = [
          { name: 'Fracionária (FP32): s = amax / ' + mx, r: mx, s: amax / mx, byte: null, col: C.b },
          { name: 'OCP MX (piso): X = ⌊log₂ amax⌋ − ' + emax, r: amax / Math.pow(2, Xf), s: Math.pow(2, Xf), X: Xf, col: C.e },
          { name: S.fmt === 'e4m3' ? 'KV do DeepSeek (teto): s = 2^⌈log₂(amax/448)⌉' : (narrow ? 'Teto (regra do KV DS no E2M1, hipotético)' : 'Teto (regra do KV DS aplicada ao E2M1, hipotético): s = 2^⌈log₂(amax/6)⌉'), r: amax / Math.pow(2, Xc), s: Math.pow(2, Xc), X: Xc, col: C.a },
        ];
        const laneH = narrow ? 88 : 80, y0 = 10;
        const xL = pad, xR = w - pad;
        const A = (v) => xL + Math.min(v, 2 * mx) / (2 * mx) * (xR - xL);
        lanes.forEach((L, k) => {
          const y = y0 + k * laneH;
          V.text(g, L.name, xL, y + 13, { size: 12, font: 'mono', color: L.col, weight: 600, maxW: w - 2 * pad });
          const by = y + 21, bh = 16;
          g.fillStyle = V.alpha(C.a, 0.14); g.fillRect(A(0), by, A(mx) - A(0), bh);
          g.fillStyle = V.alpha(C.red, 0.14); g.fillRect(A(mx), by, A(2 * mx) - A(mx), bh);
          V.line(g, A(mx), by - 3, A(mx), by + bh + 3, { color: C.muted, lw: 1.2 });
          V.text(g, nice(mx), A(mx), by + bh + 13, { size: 11, font: 'mono', align: 'center', color: C.muted });
          V.text(g, 'satura', (A(mx) + A(2 * mx)) / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: V.alpha(C.red, 0.8) });
          V.text(g, 'representável', (A(0) + A(mx)) / 2, by + bh / 2 + 4, { size: 11, align: 'center', color: V.alpha(C.a, 0.7) });
          if (S.rs[k] == null) S.rs[k] = L.r; S.rs[k] += (L.r - S.rs[k]) * Math.min(1, dt * 8);
          const px = A(S.rs[k]); const clip = L.r > mx * 1.0001;
          V.glow(g, px, by + bh / 2, 14, clip ? C.red : L.col, 0.9); V.dot(g, px, by + bh / 2, 5, clip ? C.red : L.col);
          if (clip) V.arrow(g, px, by - 3, A(mx) + 3, by - 3, { color: C.red, lw: 1.6, head: 6, curve: -8 });
          const used = Math.min(L.r, mx) / mx;
          const status = clip ? `satura: ${nice(L.r, 1)} → ${nice(mx)} (corta ${pct((L.r - mx) / L.r, 0)} do amax)` : `amax/s = ${nice(L.r, 1)} · usa ${pct(used, 0)} da faixa`;
          const sTxt = L.X == null ? `s = ${nice(L.s)} (FP32, 4 B)` : `s = 2${sup(L.X)} · byte ${L.X + 127}`;
          if (narrow) { V.text(g, status, xL, by + bh + 28, { size: 11, font: 'mono', color: clip ? C.red : C.ink, maxW: w - 2 * pad }); V.text(g, sTxt, xL, by + bh + 42, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad }); }
          else { V.text(g, status, xL, by + bh + 30, { size: 11.5, font: 'mono', color: clip ? C.red : C.ink }); V.text(g, sTxt, xR, by + bh + 30, { size: 11.5, font: 'mono', align: 'right', color: C.muted }); }
          V.line(g, xL, y + laneH - 4, xR, y + laneH - 4, { color: V.alpha(C.muted, 0.18), lw: 1 });
        });
        // --- byte E8M0 → FP32
        const L = S.fmt === 'e4m3' ? lanes[2] : lanes[1];
        const X = L.X; const goodByte = Math.max(0, Math.min(254, X + 127));
        const bugByte = X < 0 ? 0 : X <= 7 ? Math.pow(2, X) : 255;
        const byte = S.bug ? bugByte : goodByte;
        const by0 = y0 + 3 * laneH + (narrow ? 18 : 14);
        const fcW = Math.min(15, (w - 2 * pad) / 32 - 1);
        const blockW = 32 * (fcW + 1); const bl0 = Math.max(pad, (w - blockW) / 2);
        const ttl = (S.fmt === 'e4m3' ? 'Escala do KV DeepSeek (teto)' : 'Escala MXFP4 (piso)') + ` guardada como UE8M0: byte = X + 127 = ${goodByte}`;
        const nL = narrow ? wrap(g, ttl, pad, by0, w - 2 * pad, 15, { size: 12, font: 'mono', color: C.ink, weight: 600, maxLines: 2 }) : (V.text(g, ttl, w / 2, by0, { size: 12, font: 'mono', color: C.ink, weight: 600, align: 'center', maxW: w - 2 * pad }), 1);
        const cell = narrow ? Math.min(26, (w - 2 * pad - 90) / 8 - 2) : 24;
        const bits8 = N.toBin(byte, 8);
        const bx = narrow ? pad : bl0 + (1 * (fcW + 1)) + (8 * (fcW + 1) - 8 * (cell + 2)) / 2, byy = by0 + 12 + (nL - 1) * 15;
        for (let k = 0; k < 8; k++) { const on = bits8[k] === '1'; V.box(g, bx + k * (cell + 2), byy, cell, cell, { r: 4, fill: on ? V.alpha(S.bug ? C.red : C.d, 0.85) : V.alpha(C.d, 0.12), stroke: S.bug ? C.red : C.d, lw: 1 }); V.text(g, bits8[k], bx + k * (cell + 2) + cell / 2, byy + cell / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: on ? onDark(C) : C.ink }); }
        // FP32 strip
        const fy = byy + cell + 34; const fc = fcW;
        const fx0 = bl0;
        const ph = (t % 3) / 3; const fly = V.ease.inOut(Math.min(1, ph / 0.55));
        for (let k = 0; k < 32; k++) {
          const field = k === 0 ? 's' : k <= 8 ? 'e' : 'm'; const col = field === 's' ? C.c : field === 'e' ? C.d : C.a;
          const on = field === 'e' ? bits8[k - 1] === '1' && fly >= 1 : false;
          V.box(g, fx0 + k * (fc + 1), fy, fc, fc, { r: 3, fill: on ? V.alpha(col, 0.85) : V.alpha(col, 0.12), stroke: V.alpha(col, 0.8), lw: 0.8 });
        }
        V.text(g, 'FP32:  S | expoente (8) | mantissa (23) = 0', fx0, fy + fc + 14, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        V.text(g, narrow ? (S.bug ? '← corrompido' : '← byte E8M0') : '← byte E8M0' + (S.bug ? ' (corrompido)' : ''), bx + 8 * (cell + 2) + 8, byy + cell / 2 + 4, { size: 11, font: 'mono', color: S.bug ? C.red : C.d });
        if (fly < 1) for (let k = 0; k < 8; k++) { if (bits8[k] !== '1') continue; const sx = bx + k * (cell + 2) + cell / 2, sy = byy + cell; const tx = fx0 + (k + 1) * (fc + 1) + fc / 2, ty = fy; const pp = [V.lerp(sx, tx, fly), V.lerp(sy, ty, fly)]; V.dot(g, pp[0], pp[1], 3.2, S.bug ? C.red : C.d); }
        const val = Math.pow(2, byte - 127);
        const res = S.bug ? (X < 0 ? `bug: float(2${sup(X)}) → uint8 = 0 → escala lida = 2⁻¹²⁷ ≈ ${nice(val)}: expoente destruído` : `bug: uint8(2${sup(X)}) = ${bugByte} ≠ ${goodByte} → escala lida = 2${sup(byte - 127)}: errada`) : `(byte << 23).view(float32) = 2${sup(byte - 127)} = ${nice(val)}`;
        wrap(g, res, fx0, fy + fc + 32, Math.min(w - fx0 - pad, 600), 15, { size: 12, font: 'mono', color: S.bug ? C.red : C.a, weight: 600, maxLines: 3 });
      });
    });

    /* ===================================================================
       Figura 6b — Escala potência de 2 por linha (KV FP8 do TensorFold)
       =================================================================== */
    safe(function figPot2() {
      const fig = $('#fig-pot2'); if (!fig) return;
      fitH(fig, (w) => (w < 560 ? 450 : 400));
      const TX = {
        amax: 'amax da linha', elem: 'Elemento', ratio: 'amax / RMS',
        e4: 'E4M3 (máx 448)', e2: 'E2M1 (máx 6, hipotético)',
        rRow: 'FP32 por linha', rPow: '2ᵏ por linha (TF)', rBlk: 'FP32 por bloco de 128', rLost: 'na zona vermelha',
        of: (n) => n + ' de 512',
        legF: (mx) => 'escala FP32: amax → ' + mx,
        legP: (r) => '2ᵏ por linha: amax → ' + r,
        kneeF: 'FP32: subnormal',
        kneeP: '2ᵏ: subnormal',
        zone: 'só aqui a escala 2ᵏ perde',
        probe: (u, a, b) => `valor a 2${sup(u)} do amax: FP32 ${a} · 2ᵏ ${b}`,
        probeN: (u, a, b) => `2${sup(u)}·amax: FP32 ${a} · 2ᵏ ${b}`,
        yLab: 'erro relativo máx.',
        xLab: 'distância até o amax da linha (binades)',
        hist: 'valores da linha',
      };
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      const FM = {
        e4m3: { mx: 448, emax: 8, m: 3, minN: Math.pow(2, -6), minS: Math.pow(2, -9), umin: -20 },
        e2m1: { mx: 6, emax: 2, m: 1, minN: 1, minS: 0.5, umin: -8 },
      };
      const S = { t: Math.log2(2.5), fmt: 'e4m3', ratio: 4, rs: null, row: null, res: null };
      // linha sintética: 511 gaussianos + um outlier que fixa amax/RMS
      const rng = V.rng(512);
      const base = Array.from({ length: 512 }, () => rng.normal());
      const OUT = 137;
      function makeRow() {
        let Q = 0; base.forEach((v, i) => { if (i !== OUT) Q += v * v; });
        const o = S.ratio * Math.sqrt(Q / (512 - S.ratio * S.ratio));
        const row = base.map((v, i) => (i === OUT ? o : v));
        let a = 0; row.forEach((v) => { a = Math.max(a, Math.abs(v)); });
        const k = Math.pow(2, S.t) / a;
        return row.map((v) => v * k);
      }
      function scaleOf(F, a, mode) { return mode === 'pow' ? Math.pow(2, Math.ceil(Math.log2(a)) - F.emax) : a / F.mx; }
      function rowErr(row, mode, seg) {
        const F = FM[S.fmt]; const out = new Array(row.length);
        for (let b = 0; b < row.length; b += seg) {
          let a = 0; for (let i = b; i < b + seg; i++) a = Math.max(a, Math.abs(row[i]));
          const s = scaleOf(F, a || 1, mode);
          for (let i = b; i < b + seg; i++) out[i] = N.encode(S.fmt, row[i] / s).v * s;
        }
        return relRMSE(row, out);
      }
      function relErr(F, y) {
        if (!(y > 0) || y < F.minS / 2) return 1;
        const ulp = y >= F.minN ? Math.pow(2, Math.floor(Math.log2(y)) - F.m) : F.minS;
        return Math.min(1, ulp / 2 / y);
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: TX.amax, min: -3, max: 5, step: 0.01, value: S.t, fmt: (v) => nice(Math.pow(2, v)), onInput: (v) => { S.t = v; upd(); } });
      UI.seg(ctl, { label: TX.elem, options: [{ v: 'e4m3', label: TX.e4 }, { v: 'e2m1', label: TX.e2 }], value: 'e4m3', onChange: (v) => { S.fmt = v; S.rs = null; upd(); } });
      UI.slider(ctl, { label: TX.ratio, min: 3.5, max: 8, step: 0.1, value: S.ratio, fmt: (v) => nice(v, 1) + '×', onInput: (v) => { S.ratio = v; upd(); } });
      const rdRow = UI.readout(ctl, TX.rRow, '—'), rdPow = UI.readout(ctl, TX.rPow, '—'), rdBlk = UI.readout(ctl, TX.rBlk, '—'), rdLost = UI.readout(ctl, TX.rLost, '—');
      function upd() {
        const F = FM[S.fmt];
        const row = makeRow(); S.row = row;
        const amax = Math.pow(2, S.t);
        const r = amax / scaleOf(F, amax, 'pow');
        const eRow = rowErr(row, 'row', 512), ePow = rowErr(row, 'pow', 512), eBlk = rowErr(row, 'row', 128);
        const thr = amax * F.minN / r;
        let lost = 0; row.forEach((v) => { if (Math.abs(v) < thr) lost++; });
        const hist = new Array(-F.umin).fill(0);
        row.forEach((v) => { const u = Math.log2(Math.abs(v) / amax); const k = Math.min(-F.umin - 1, Math.max(0, Math.floor(-u))); if (Number.isFinite(u)) hist[k]++; });
        S.res = { r, eRow, ePow, eBlk, lost, hist };
        const close = Math.abs(ePow - eRow) < 0.002;
        rdRow.set(pct(eRow, 2));
        rdPow.set(pct(ePow, 2), close ? 'ok' : 'warn');
        rdBlk.set(pct(eBlk, 2));
        rdLost.set(TX.of(lost), lost === 0 ? 'ok' : lost < 8 ? 'warn' : 'bad');
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const F = FM[S.fmt]; const R = S.res; if (!R) return;
        const TC = tfc(C); const pad = 16; const narrow = w < 560;
        if (S.rs == null) S.rs = R.r; S.rs += (R.r - S.rs) * Math.min(1, dt * 6);
        const rs = S.rs;
        // legenda
        const lgY = 14;
        const leg = [[C.b, TX.legF(nice(F.mx))], [TC, TX.legP(nice(rs, 1))]];
        if (narrow) leg.forEach(([c, s], i) => { V.line(g, pad, lgY + i * 16 - 4, pad + 18, lgY + i * 16 - 4, { color: c, lw: 2.5 }); V.text(g, s, pad + 24, lgY + i * 16, { size: 11.5, font: 'mono', color: C.ink, maxW: w - 2 * pad - 24 }); });
        else { let x = pad + 40; leg.forEach(([c, s]) => { V.line(g, x, lgY - 4, x + 18, lgY - 4, { color: c, lw: 2.5 }); V.text(g, s, x + 24, lgY, { size: 12, font: 'mono', color: C.ink }); x += 24 + tw(g, s, { size: 12, font: 'mono' }) + 26; }); }
        const x0 = pad + 40, x1 = w - pad;
        const yT = narrow ? 54 : 30, yB = h - (narrow ? 112 : 100);
        const hy0 = yB + 12, hh = narrow ? 46 : 42;
        const um = F.umin;
        const X = (u) => x0 + (u - um) / (0 - um) * (x1 - x0);
        const lmin = -2, Y = (e) => yB - (Math.log10(Math.max(e, 0.01)) - lmin) / (0 - lmin) * (yB - yT);
        // grade e eixo y
        [0.01, 0.03, 0.1, 0.3, 1].forEach((e) => { const y = Y(e); V.line(g, x0, y, x1, y, { color: V.alpha(C.grid || C.muted, 0.35), lw: 1 }); V.text(g, pct(e, 0), x0 - 6, y + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); });
        V.text(g, TX.yLab, x0 + 4, yT - 4 + (narrow ? 0 : 0), { size: 11, color: C.muted, maxW: (x1 - x0) * 0.4 });
        // zona onde as escalas diferem (abaixo do joelho da potência de 2)
        const uKP = Math.log2(F.minN / rs), uKF = Math.log2(F.minN / F.mx);
        const xKP = X(Math.max(um, uKP)), xKF = X(Math.max(um, uKF));
        g.fillStyle = V.alpha(C.red, 0.08); g.fillRect(x0, yT, xKP - x0, hy0 + hh - yT);
        V.line(g, xKP, yT, xKP, hy0 + hh, { color: TC, lw: 1.4, dash: [5, 4] });
        V.line(g, xKF, yT, xKF, hy0 + hh, { color: C.b, lw: 1.4, dash: [2, 4] });
        g.setLineDash([]);
        const tag = (s, x, y, col, al) => {   // rótulo com fundo; troca de lado se não couber
          const nw = tw(g, s, { size: 11, font: 'mono' });
          let right = al === 'right';
          if (!right && x1 - x - 4 < nw) { right = true; x -= 10; } else if (right && x - x0 - 4 < nw) { right = false; x += 10; }
          const ww = Math.min(nw, right ? x - x0 - 4 : x1 - x - 4);
          if (ww < 20) return;
          V.box(g, right ? x - ww - 4 : x - 4, y - 11, ww + 8, 16, { r: 4, fill: V.alpha(C.bg, 0.86) });
          V.text(g, s, x, y, { size: 11, font: 'mono', color: col, align: right ? 'right' : 'left', maxW: ww });
        };
        { const zw = Math.max(70, xKP - x0 - 12); const nz = splitLines(g, TX.zone, zw, { size: 11 }).length; wrap(g, TX.zone, x0 + 6, yB - 16 - 14 * Math.min(3, nz), zw, 14, { size: 11, color: C.red, maxLines: 3 }); }
        // curvas: erro relativo máximo de arredondamento a 2^u do amax
        const curve = (r, col) => {
          g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
          for (let px = x0; px <= x1; px += 1) { const u = um + (px - x0) / (x1 - x0) * (0 - um); const y = Y(relErr(F, r * Math.pow(2, u))); if (px === x0) g.moveTo(px, y); else g.lineTo(px, y); }
          g.stroke();
        };
        curve(F.mx, V.alpha(C.b, 0.95));
        curve(rs, TC);
        tag(TX.kneeP, xKP + 5, yT + 32, TC);
        tag(TX.kneeF, xKF - 5, yT + 50, C.b, 'right');
        // sonda que varre as binades
        const up = (um + 0.5) * V.ease.inOut(V.pingpong(t, 14));
        const pxp = X(up);
        V.line(g, pxp, yT + 4, pxp, yB, { color: V.alpha(C.ink, 0.25), lw: 1 });
        const eF = relErr(F, F.mx * Math.pow(2, up)), eP = relErr(F, rs * Math.pow(2, up));
        V.dot(g, pxp, Y(eF), 4.5, C.b); V.dot(g, pxp, Y(eP), 4.5, TC);
        const ptxt = (narrow ? TX.probeN : TX.probe)(Math.round(up) || 0, pct(eF, 1), pct(eP, 1));
        tag(ptxt, x1 - 2, yT + 13, up < uKP ? C.red : C.ink, 'right');
        // eixo x + histograma da linha
        V.line(g, x0, yB, x1, yB, { color: C.muted, lw: 1 });
        const nb = -um; const bw = (x1 - x0) / nb; const mxC = Math.max(1, ...R.hist);
        for (let k = 0; k < nb; k++) {
          const c = R.hist[k]; if (!c) continue;
          const xa = X(-(k + 1)), hb = Math.max(2, Math.sqrt(c / mxC) * hh);
          const zone = -k <= uKP, part = !zone && -(k + 1) < uKP;   // bin = [−(k+1), −k)
          const col = zone ? C.red : part ? C.d : C.muted;
          V.box(g, xa + 1, hy0 + hh - hb, Math.max(1, bw - 2), hb, { r: 2, fill: V.alpha(col, 0.55) });
          if (zone || part || bw > 22) V.text(g, String(c), xa + bw / 2, hy0 + hh - hb - 3, { size: 11, font: 'mono', align: 'center', color: zone ? C.red : C.muted });
        }
        V.text(g, TX.hist, x0 + 6, hy0 + 12, { size: 11, color: C.muted, maxW: Math.max(50, xKP - x0 - 14) });
        const stepL = S.fmt === 'e2m1' ? 1 : narrow ? 4 : 2;
        for (let u = 0; u >= um; u -= stepL) V.text(g, u === 0 ? '2⁰' : '2' + sup(u), X(u), hy0 + hh + 15, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.text(g, TX.xLab, (x0 + x1) / 2, hy0 + hh + 32, { size: 11, align: 'center', color: C.muted, maxW: x1 - x0 });
      });
    });

    /* ===================================================================
       Figura 5 — E2M1: teclado de 16 códigos + trem de nibbles
       =================================================================== */
    safe(function figE2m1() {
      const fig = $('#fig-e2m1'); if (!fig) return;
      fitH(fig, (w) => (w < 560 ? 372 : 386));
      const S = { x: 2.6, anim: true, mx: 2.6 };
      const codes = Array.from({ length: 16 }, (_, c) => ({ c, v: N.decode('e2m1', c) }));
      // bloco do exemplo NVFP4 (seção 5): x / s → códigos
      const X16 = [0.12, -0.95, 0.33, 2.80, -0.04, 0.61, -1.72, 0.08, 0.47, -0.29, 1.10, -0.66, 0.02, 0.91, -2.05, 0.38];
      const s16 = N.encode('e4m3', 2.80 / 6 / (4 / 2688)).v * (4 / 2688);
      const P = X16.map((v) => N.encode('e2m1', v / s16).bits);
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'valor x/s', min: -7.5, max: 7.5, step: 0.01, value: S.x, fmt: (v) => nice(v, 2), onInput: (v) => { S.x = v; } });
      UI.toggle(ctl, { label: 'animar o empacotamento', value: true, onChange: (v) => { S.anim = v; } });
      const rC = UI.readout(ctl, 'código', '—');
      const rVv = UI.readout(ctl, 'valor', '—');
      let lastCode = -1;
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const pad = 16;
        const enc = N.encode('e2m1', S.x); const code = enc.bits;
        if (code !== lastCode) { rC.set(N.toBin(code, 4)); rVv.set(nice(enc.v)); lastCode = code; }
        // teclado
        const gap = 6, cw = (w - 2 * pad - 7 * gap) / 8, chh = 44;
        for (let c = 0; c < 16; c++) {
          const row = c >> 3, col = c & 7; const x = pad + col * (cw + gap), y = 10 + row * (chh + gap);
          const on = c === code; const v = codes[c].v;
          V.box(g, x, y, cw, chh, { r: 8, fill: on ? V.alpha(C.a, 0.3) : V.alpha(C.muted, 0.08), stroke: on ? C.a : V.alpha(C.muted, 0.4), lw: on ? 2.2 : 1 });
          const b = N.toBin(c, 4); const o = { size: Math.max(11, Math.min(13, cw / 4.2)), font: 'mono', weight: 600 };
          const segs = [[b[0], C.c], [b.slice(1, 3), C.d], [b[3], C.a]]; const gp = 3;
          const total = segs.reduce((a, sg) => a + tw(g, sg[0], o), 0) + 2 * gp; let xx = x + cw / 2 - total / 2;
          segs.forEach(([txt, cc]) => { V.text(g, txt, xx, y + 16, Object.assign({ color: cc, base: 'middle' }, o)); xx += tw(g, txt, o) + gp; });
          V.text(g, (Object.is(v, -0) ? '−0' : nice(v)), x + cw / 2, y + 33, { size: Math.max(11, Math.min(13, cw / 3.6)), font: 'mono', align: 'center', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
          if (on && cw > 2) V.glow(g, x + cw / 2, y + chh / 2, cw * 0.7, C.a, 0.35);
        }
        // régua −7..7
        const ly = 10 + 2 * (chh + gap) + 56, xL = pad + 6, xR = w - pad - 6;
        const L = (v) => xL + (v + 7.5) / 15 * (xR - xL);
        g.fillStyle = V.alpha(C.red, 0.12); g.fillRect(L(6), ly - 16, L(7.5) - L(6), 32); g.fillRect(L(-7.5), ly - 16, L(-6) - L(-7.5), 32);
        V.text(g, 'satura', L(6.75), ly - 20, { size: 11, align: 'center', color: C.red });
        V.line(g, xL, ly, xR, ly, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const pos = [0, 0.5, 1, 1.5, 2, 3, 4, 6]; const bnd = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5];
        bnd.forEach((b) => [b, -b].forEach((bb) => V.line(g, L(bb), ly - 12, L(bb), ly + 12, { color: V.alpha(C.muted, 0.35), lw: 1, dash: [2, 3] })));
        pos.forEach((v, pi) => [v, -v].forEach((vv) => { V.dot(g, L(vv), ly, 4.2, C.a); const half = v % 1 !== 0; if (w >= 480 || !half) V.text(g, nice(vv), L(vv), ly + (half ? 34 : 21), { size: 11, font: 'mono', align: 'center', color: half ? V.alpha(C.muted, 0.85) : C.muted }); }));
        S.mx += (S.x - S.mx) * Math.min(1, dt * 10);
        const mxp = L(S.mx); g.fillStyle = C.c; g.beginPath(); g.moveTo(mxp - 6, ly - 30); g.lineTo(mxp + 6, ly - 30); g.lineTo(mxp, ly - 20); g.closePath(); g.fill();
        V.arrow(g, mxp, ly - 18, L(enc.v), ly - 6, { color: C.c, lw: 1.6, head: 6, curve: 10 });
        V.glow(g, L(enc.v), ly, 14, C.c, 0.9);
        V.text(g, `x/s = ${nice(S.x, 2)} → código ${N.toBin(code, 4)} = ${Object.is(enc.v, -0) ? '−0' : nice(enc.v)}  (erro ${nice(Math.abs(enc.v - S.x), 2)})`, w / 2, ly + 54, { size: 12, font: 'mono', align: 'center', color: C.ink, maxW: w - 2 * pad });
        // trem de nibbles
        const ty = ly + 78; const nW = (w - 2 * pad) / 16; const byW = (w - 2 * pad - 7 * 8) / 8;
        V.text(g, w < 560 ? 'bloco NVFP4 da seção 5: 16 códigos → 8 bytes' : 'bloco de 16 códigos (exemplo NVFP4 da seção 5) → 8 bytes · 8 = −0', pad, ty, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const per = 7; const ph = S.anim ? (t % per) / per : 1;
        for (let k = 0; k < 16; k++) {
          const pair = k >> 1; const start = 0.08 + pair * 0.085; const u = V.ease.inOut(Math.max(0, Math.min(1, (ph - start) / 0.12)));
          const sx = pad + k * nW + 1, sy = ty + 10;
          const bx = pad + pair * (byW + 8), byy = ty + 52;
          const tx = bx + (k & 1 ? 2 : byW / 2 + 1), tyy = byy + 3;
          const x = V.lerp(sx, tx, u), y = V.lerp(sy, tyy, u); const ww = V.lerp(nW - 2, byW / 2 - 3, u);
          V.box(g, x, y, ww, 22, { r: 4, fill: V.alpha(k & 1 ? C.b : C.a, 0.22), stroke: k & 1 ? C.b : C.a, lw: 1 });
          V.text(g, P[k].toString(16).toUpperCase(), x + ww / 2, y + 12, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
        }
        for (let b = 0; b < 8; b++) {
          const bx = pad + b * (byW + 8), byy = ty + 52;
          V.box(g, bx, byy, byW, 28, { r: 6, stroke: V.alpha(C.muted, 0.6), lw: 1 });
          const hex = ((P[2 * b + 1] << 4) | P[2 * b]).toString(16).toUpperCase().padStart(2, '0');
          const done = ph > 0.08 + b * 0.085 + 0.12;
          if (done) V.text(g, '0x' + hex, bx + byW / 2, byy + 44, { size: 11, font: 'mono', align: 'center', color: C.d, weight: 600 });
        }
        V.text(g, 'cada byte = [nibble alto: elemento ímpar | nibble baixo: elemento par]', w / 2, ty + 118, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: w - 2 * pad });
      });
    });

    /* ===================================================================
       Figura 6 — NVFP4 em dois níveis (stepper) + MXFP4
       =================================================================== */
    safe(function figNvfp4() {
      const fig = $('#fig-nvfp4'); if (!fig) return;
      fitH(fig, (w) => (w < 700 ? 404 : 380));
      const X16 = [0.12, -0.95, 0.33, 2.80, -0.04, 0.61, -1.72, 0.08, 0.47, -0.29, 1.10, -0.66, 0.02, 0.91, -2.05, 0.38];
      const gS = 4.0 / 2688; const sbE = N.encode('e4m3', 2.80 / 6 / gS); const sB = sbE.v * gS;
      const R16 = X16.map((v) => v / sB); const Q = R16.map((v) => N.encode('e2m1', v)); const D = Q.map((q) => q.v * sB);
      const relNV = relRMSE(X16, D);
      const mx = N.mx(X16, 'e2m1', 16); const relMX = relRMSE(X16, mx.deq);
      const bytes = []; for (let k = 0; k < 16; k += 2) bytes.push(((Q[k + 1].bits << 4) | Q[k].bits));
      const steps = [
        { t: 'O bloco', d: 'Dezesseis valores de um tensor cujo amax total é 4,0 (suposto). Em BF16 são 32 bytes. Vamos guardá-los em NVFP4.' },
        { t: 'Boneca de fora: escala global', d: '\\(g = \\mathrm{amax}_{\\text{tensor}}/(6\\cdot448) = 4{,}0/2688 = 0{,}0014881\\), em FP32. É o <code>weight_scale_2</code>. Ela garante que a maior escala de bloco do tensor seja 448, o topo do E4M3.' },
        { t: 'Boneca do meio: escala do bloco', d: 'amax do bloco = 2,80. Em unidades de \\(g\\): 2,80/6/g = 313,6, cujo E4M3 mais próximo é <b>320</b> (<code>0.1111.010</code>). A escala efetiva é \\(s = 320\\,g = 0{,}47619\\). Esse byte vai para o tensor <code>weight_scale</code>.' },
        { t: 'Boneca de dentro: os códigos', d: 'Dividindo por \\(s\\), o bloco cai em [−6, 6]: 5,88 → 6; −3,61 → −4; 1,28 → 1,5; 0,25 → 0,5; −0,084 → 0. São 16 códigos de 4 bits.' },
        { t: 'Empacotar', d: '16 códigos = 8 bytes (2 por byte, nibble baixo = elemento par) + 1 byte de escala E4M3 = 9 bytes para 16 valores, ou <b>4,5 bits/valor</b>. O FP32 global custa 4 bytes por tensor inteiro.' },
        { t: 'Desquantizar e medir', d: '\\(\\hat x = q\\cdot s_b\\cdot g\\). O RMSE relativo do bloco é 8,5%. Os pequenos sofrem: −0,04, 0,08 e 0,02 viram 0, e 0,12 vira 0,24 (o código 0,5). É por isso que o NVFP4 usa blocos de só 16.' },
        { t: 'MXFP4 no mesmo bloco', d: 'Sem escala global e com escala E8M0: \\(X = \\lfloor\\log_2 2{,}80\\rfloor - 2 = -1\\), byte 126, escala 0,5. O RMSE relativo sobe para 9,6%. Aqui usamos bloco de 16 para comparar; no MXFP4 real o bloco é de 32, o que tende a piorar mais.' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H && H.redraw() });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const pad = 16; const i = st.i; const el = st.elapsed(); const e = V.ease.inOut(Math.min(1, el / 0.9));
        const cols = w < 700 ? 8 : 16, rows = 16 / cols, gap = 5;
        const cw = (w - 2 * pad - (cols - 1) * gap) / cols, chh = 38, y0 = 12;
        const fs = Math.max(11, Math.min(12.5, cw / 3.9));
        for (let k = 0; k < 16; k++) {
          const x = pad + (k % cols) * (cw + gap), y = y0 + Math.floor(k / cols) * (chh + gap);
          let txt, col = C.ink, fill = V.alpha(C.b, 0.08);
          if (i <= 2) txt = nice(X16[k], 2);
          else if (i === 3) { txt = e < 0.5 ? nice(R16[k], 2) : nice(Q[k].v); fill = V.alpha(C.a, 0.1 + 0.25 * e * Math.abs(Q[k].v) / 6); }
          else if (i === 4) { txt = Q[k].bits.toString(16).toUpperCase(); col = C.a; fill = V.alpha(C.a, 0.14); }
          else if (i === 5) txt = nice(D[k], 2);
          else txt = nice(mx.deq[k], 2);
          const isMax = k === 3 && (i === 2 || i === 6);
          V.box(g, x, y, cw, chh, { r: 6, fill, stroke: isMax ? C.c : V.alpha(C.muted, 0.35), lw: isMax ? 2.2 : 1 });
          V.text(g, txt, x + cw / 2, y + chh / 2 + 1, { size: fs, font: 'mono', align: 'center', base: 'middle', color: col });
        }
        const yM = y0 + rows * (chh + gap) + 10;
        // bonecas (aninhadas: cada nível tem sua faixa de rótulo)
        const cx = w / 2; const bw = Math.min(w - 2 * pad, 600); const HB = 118;
        const lvl = [
          { lab: i === 6 ? 'MX: não há escala global' : (w < 560 ? 'g = 0,0014881 (FP32, weight_scale_2)' : 'g = 0,0014881 · escala global FP32 (weight_scale_2)'), col: C.b, on: i >= 1 && i !== 6, dim: i === 6 },
          { lab: i === 6 ? 's = 2⁻¹ = 0,5 (E8M0, byte 126)' : (w < 560 ? 's_b = 320 (E4M3) → s = 0,47619' : 's_b = 320 (E4M3 0.1111.010) → s = 320·g = 0,47619'), col: i === 6 ? C.f : C.d, on: i >= 2 },
          { lab: 'códigos E2M1', col: C.a, on: i >= 3 },
        ];
        lvl.forEach((L, k) => {
          const ww = bw - k * 30, x = cx - ww / 2, y = yM + k * 24, hh = HB - k * 30;
          const a = L.on ? 1 : 0.22;
          const pulse = L.on && ((i === 1 && k === 0) || (i === 2 && k === 1) || (i === 3 && k === 2) || (i === 6 && k === 1)) ? 0.5 + 0.5 * Math.sin(t * 5) : 0;
          V.box(g, x, y, ww, hh, { r: 16 - k * 4, fill: V.alpha(L.col, 0.05 * a + 0.1 * pulse), stroke: V.alpha(L.col, L.dim ? 0.18 : a), lw: 1.3 + pulse });
          V.text(g, L.lab, cx, y + 16, { size: 11.5, font: 'mono', align: 'center', color: V.alpha(L.col, L.dim ? 0.45 : Math.max(0.35, a)), weight: 600, maxW: ww - 16 });
        });
        // códigos dentro da boneca menor
        { const ww = bw - 60 - 16, x0 = cx - ww / 2, y = yM + 48 + 24, cwd = ww / 16;
          for (let k = 0; k < 16; k++) { const on = i >= 3; const v = i === 6 ? mx.codes[k].bits : Q[k].bits; V.box(g, x0 + k * cwd + 1, y, cwd - 2, 20, { r: 4, fill: V.alpha(C.a, on ? 0.2 : 0.05), stroke: V.alpha(C.a, on ? 0.7 : 0.2), lw: 1 }); if (on) V.text(g, v.toString(16).toUpperCase(), x0 + k * cwd + cwd / 2, y + 11, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink }); } }
        const yB = yM + HB + 16, hB = h - yB - 6;
        if (i === 0) {
          const bxs = pad + 110, full = w - pad - bxs - 70; const u = V.ease.out(Math.min(1, el / 0.8));
          [['BF16', 32, C.muted], ['NVFP4', 9, C.a], ['MXFP4*', 8.5, C.f]].forEach(([nm, b, col], k) => { const y = yB + 8 + k * 28; V.text(g, nm, bxs - 10, y + 13, { size: 12, font: 'mono', align: 'right', color: col }); V.box(g, bxs, y, Math.max(3, full * b / 32 * u), 18, { r: 5, fill: V.alpha(col, 0.6) }); V.text(g, nice(b) + ' B', bxs + full * b / 32 * u + 8, y + 13, { size: 11.5, font: 'mono', color: C.ink }); });
          V.text(g, 'bytes para 16 valores (*MXFP4: 1 escala a cada 32 → meio byte por 16)', w / 2, yB + 8 + 3 * 28 + 10, { size: 11, align: 'center', color: C.muted, maxW: w - 2 * pad });
        } else if (i === 1 || i === 2) {
          const xL = pad + 10, xR = w - pad - 10; const lo = -9, hi = Math.log2(448);
          const Lx = (v) => xL + (Math.log2(v) - lo) / (hi - lo) * (xR - xL);
          const ry = yB + hB * 0.45;
          V.text(g, 'régua do E4M3 para a escala do bloco (em unidades de g)', w / 2, yB + 12, { size: 11.5, align: 'center', color: C.muted, maxW: w - 2 * pad });
          V.line(g, xL, ry, xR, ry, { color: V.alpha(C.muted, 0.5), lw: 1 });
          g.strokeStyle = V.alpha(C.d, 0.6); g.lineWidth = 1; g.beginPath(); N.table('e4m3').forEach((tt) => { if (tt.v <= 0) return; const px = Math.round(Lx(tt.v)) + 0.5; g.moveTo(px, ry - 5); g.lineTo(px, ry + 5); }); g.stroke();
          [[Math.pow(2, -9), '2⁻⁹'], [1, '1'], [16, '16'], [448, '448']].forEach(([v, lb]) => V.text(g, lb, Lx(v), ry + 20, { size: 11, font: 'mono', align: 'center', color: C.muted }));
          V.glow(g, Lx(448), ry, 14, C.b, 0.9); V.dot(g, Lx(448), ry, 4.5, C.b);
          V.text(g, 'bloco com amax = amax do tensor → 448', Lx(448), ry - 14, { size: 11, font: 'mono', align: 'right', color: C.b });
          if (i === 2) { const tgt = V.lerp(Lx(313.6), Lx(320), e); V.dot(g, Lx(313.6), ry, 3, V.alpha(C.d, 0.6)); V.glow(g, tgt, ry, 14, C.d, 0.9); V.dot(g, tgt, ry, 4.5, C.d); V.text(g, 'este bloco: 313,6 → 320 (E4M3 mais próximo)', tgt - 4, ry + 40, { size: 11.5, font: 'mono', align: 'right', color: C.d, weight: 600 }); V.line(g, tgt, ry + 6, tgt, ry + 28, { color: V.alpha(C.d, 0.7), lw: 1 }); }
        } else if (i === 3) {
          const xL = pad + 10, xR = w - pad - 10; const Lx = (v) => xL + (v + 6.5) / 13 * (xR - xL); const ry = yB + hB * 0.5;
          V.line(g, xL, ry, xR, ry, { color: V.alpha(C.muted, 0.5), lw: 1 });
          [0, 0.5, 1, 1.5, 2, 3, 4, 6].forEach((v) => [v, -v].forEach((vv) => { V.line(g, Lx(vv), ry - 7, Lx(vv), ry + 7, { color: C.a, lw: 1.5 }); if (v % 1 === 0) V.text(g, nice(vv), Lx(vv), ry + 22, { size: 11, font: 'mono', align: 'center', color: C.muted }); }));
          for (let k = 0; k < 16; k++) { const px = V.lerp(Lx(R16[k]), Lx(Q[k].v), e); const py = ry - 16 - (k % 4) * 7; V.dot(g, px, py, 3.2, C.c); V.line(g, px, py, px, ry, { color: V.alpha(C.c, 0.3), lw: 1 }); }
          V.text(g, 'x/s (rosa) caindo na grade E2M1: passo 0,5 perto de 0, 2 no topo', w / 2, yB + 12, { size: 11.5, align: 'center', color: C.muted, maxW: w - 2 * pad });
        } else if (i === 4) {
          const bx0 = pad, bW = (w - 2 * pad - 9 * 6) / 10, yb = yB + 16;
          bytes.forEach((b, k) => { const x = bx0 + k * (bW + 6); const u = V.ease.out(Math.max(0, Math.min(1, (el - k * 0.12) / 0.4))); V.box(g, x, yb + (1 - u) * 20, bW, 30, { r: 6, fill: V.alpha(C.a, 0.18 * u), stroke: V.alpha(C.a, u), lw: 1.2 }); V.text(g, '0x' + b.toString(16).toUpperCase().padStart(2, '0'), x + bW / 2, yb + 16 + (1 - u) * 20, { size: 11, font: 'mono', align: 'center', base: 'middle', color: V.alpha(C.ink, u) }); });
          const x = bx0 + 8 * (bW + 6); V.box(g, x, yb, bW, 30, { r: 6, fill: V.alpha(C.d, 0.25), stroke: C.d, lw: 1.4 }); V.text(g, '0x' + sbE.bits.toString(16).toUpperCase(), x + bW / 2, yb + 16, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          const x2 = bx0 + 9 * (bW + 6); V.box(g, x2, yb, bW, 30, { r: 6, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.6), lw: 1 }); V.text(g, '+g', x2 + bW / 2, yb + 16, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.b });
          V.text(g, '8 bytes de códigos + 1 byte de escala E4M3 = 9 B / 16 valores = 4,5 bits (g: 4 B por tensor)', w / 2, yb + 52, { size: 11.5, font: 'mono', align: 'center', color: C.muted, maxW: w - 2 * pad });
        } else {
          const deq = i === 5 ? D : mx.deq; const rel = i === 5 ? relNV : relMX; const col = i === 5 ? C.a : C.f;
          const bh = hB - 22, base = yB + bh / 2 + 2, sc = bh / 2 / 3.0; const bwid = (w - 2 * pad) / 16;
          V.line(g, pad, base, w - pad, base, { color: V.alpha(C.muted, 0.4), lw: 1 });
          for (let k = 0; k < 16; k++) {
            const x = pad + k * bwid + bwid * 0.2, ww = bwid * 0.6;
            const o = X16[k] * sc; g.fillStyle = V.alpha(C.muted, 0.28); g.fillRect(x, base - Math.max(0, o), ww, Math.abs(o));
            const dd = deq[k] * sc * Math.min(1, e * 1.2); V.dot(g, x + ww / 2, base - dd, 3.6, col);
          }
          V.text(g, `${i === 5 ? 'NVFP4' : 'MXFP4 (bloco 16, ilustrativo)'} · RMSE relativo ${pct(rel, 1)} · barras = original, pontos = desquantizado`, w / 2, h - 10, { size: 11.5, font: 'mono', align: 'center', color: C.ink, maxW: w - 2 * pad });
        }
      });
    });

    /* ===================================================================
       Figura 7 — Laboratório: bits por peso × SQNR
       =================================================================== */
    safe(function figLab() {
      const fig = $('#fig-lab'); if (!fig) return;
      fitH(fig, (w) => (w < 640 ? 560 : 480));
      const S = { out: 1, seed: 7, g: 32, zoom: 'nvfp4', had: false };
      let xs = [], pts = [], zoomBlock = 0;
      const had = (arr, fn) => { const rot = []; for (let k = 0; k < arr.length; k += 128) rot.push(...N.fwht(arr.slice(k, k + 128))); const d = fn(rot); const back = []; for (let k = 0; k < d.length; k += 128) back.push(...N.fwht(d.slice(k, k + 128))); return back; };
      const FM = [
        { id: 'fp4t', name: 'FP4 por tensor', bits: () => 4, col: (C) => C.red, fn: (a) => { const m = Math.max(...a.map(Math.abs)); const s = m / 6; return a.map((x) => N.encode('e2m1', x / s).v * s); } },
        { id: 'int4', name: () => 'INT4 g' + S.g, bits: () => 4 + 16 / S.g, col: (C) => C.e, fn: (a) => N.quantInt(a, { bits: 4, symmetric: true, group: S.g }).deq },
        { id: 'mxfp4', name: 'MXFP4', bits: () => 4.25, col: (C) => C.f, fn: (a) => N.mx(a, 'e2m1', 32).deq },
        { id: 'nvfp4', name: 'NVFP4', bits: () => 4.5, col: (C) => C.a, fn: (a) => N.nvfp4(a).deq },
        { id: 'fp8t', name: 'FP8 por tensor', bits: () => 8, col: (C) => C.d, fn: (a) => N.fp8Tensor(a).deq },
        { id: 'mxfp8', name: 'MXFP8', bits: () => 8.25, col: (C) => C.b, fn: (a) => N.mx(a, 'e4m3', 32).deq },
      ];
      function compute() {
        xs = N.synthWeights(4096, S.seed, S.out / 100, 8);
        let best = -1, bi = 0; for (let k = 0; k < xs.length; k++) if (Math.abs(xs[k]) > best) { best = Math.abs(xs[k]); bi = k; }
        zoomBlock = Math.floor(bi / 32) * 32;
        pts = FM.map((f, k) => {
          const d = S.had ? had(xs, f.fn) : f.fn(xs);
          const st = N.errStats(xs, d);
          const old = pts[k];
          return { f, bits: f.bits(), sq: st.sqnr, deq: d, cx: old ? old.cx : f.bits(), cy: old ? old.cy : st.sqnr };
        });
        const four = pts.filter((p) => p.bits < 5).sort((a, b) => b.sq - a.sq)[0];
        rB.set((typeof four.f.name === 'function' ? four.f.name() : four.f.name) + ' · ' + four.sq.toFixed(1).replace('.', ',') + ' dB');
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Outliers', min: 0, max: 5, step: 0.5, value: S.out, fmt: (v) => v.toFixed(1).replace('.', ',') + '%', onInput: (v) => { S.out = v; compute(); } });
      UI.slider(ctl, { label: 'Semente', min: 1, max: 30, step: 1, value: S.seed, onInput: (v) => { S.seed = v; compute(); } });
      UI.seg(ctl, { label: 'Grupo INT4', options: [16, 32, 64, 128], value: 32, onChange: (v) => { S.g = v; compute(); } });
      UI.select(ctl, { label: 'Zoom', options: FM.map((f) => ({ v: f.id, label: typeof f.name === 'function' ? 'INT4 por grupo' : f.name })), value: 'nvfp4', onChange: (v) => { S.zoom = v; } });
      UI.toggle(ctl, { label: 'Hadamard por 128 (prévia do cap. 22)', value: false, onChange: (v) => { S.had = v; compute(); } });
      const rB = UI.readout(ctl, 'melhor de 4 bits', '—');
      compute();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const pad = 16; const narrow = w < 640;
        const chartH = narrow ? h * 0.6 : h * 0.62;
        const xL = pad + 34, xR = w - pad - 6, yT = 16, yB = chartH - 26;
        const maxSq = Math.max(35, ...pts.map((p) => p.sq)) + 3;
        const Y = (v) => yB - Math.max(0, v) / maxSq * (yB - yT);
        const W = xR - xL;
        const Xb = (b) => (b < 6 ? xL + (b - 3.9) / 0.7 * 0.6 * W : xL + 0.7 * W + (b - 7.9) / 0.45 * 0.28 * W);
        // grade
        for (let v = 0; v <= maxSq; v += 5) { V.line(g, xL, Y(v), xR, Y(v), { color: C.grid, lw: 1 }); V.text(g, String(v), xL - 6, Y(v) + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); }
        V.text(g, 'SQNR (dB)', pad, yT - 4, { size: 11, font: 'mono', color: C.muted });
        [4, 4.25, 4.5, 8, 8.25].forEach((b) => { V.line(g, Xb(b), yB, Xb(b), yB + 5, { color: C.muted }); V.text(g, nice(b), Xb(b), yB + 17, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        const brk = xL + 0.65 * W; V.text(g, '//', brk, yB + 4, { size: 13, font: 'mono', align: 'center', color: C.muted });
        V.text(g, 'bits efetivos por peso', (xL + xR) / 2, yB + 32, { size: 11, align: 'center', color: C.muted });
        // animação dos pontos
        pts.forEach((p) => { p.cx += (p.bits - p.cx) * Math.min(1, dt * 6); p.cy += (p.sq - p.cy) * Math.min(1, dt * 6); });
        // fronteira de Pareto
        const sorted = pts.slice().sort((a, b) => a.bits - b.bits || b.sq - a.sq); const front = []; let bestSq = -Infinity;
        sorted.forEach((p) => { if (p.sq > bestSq) { front.push(p); bestSq = p.sq; } });
        g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1.4; g.setLineDash([5, 5]); g.lineDashOffset = -t * 12; g.beginPath();
        front.forEach((p, k) => { const x = Xb(p.cx), y = Y(p.cy); if (k) g.lineTo(x, y); else g.moveTo(x, y); }); g.stroke(); g.setLineDash([]);
        // rótulos sem colisão
        const lab = pts.map((p) => ({ p, x: Xb(p.cx), y: Y(p.cy) })).sort((a, b) => a.y - b.y);
        const placed = [];
        lab.forEach((L) => {
          let ly = L.y + 4; placed.forEach((q) => { if (Math.abs(q.x - L.x) < 110 && Math.abs(q.ly - ly) < 14) ly = q.ly + 14; });
          const nm0 = typeof L.p.f.name === 'function' ? L.p.f.name() : L.p.f.name; const twd = tw(g, nm0 + ' 00,0 dB', { size: 11.5, font: 'mono' });
          const right = L.x < xR - 130; const x0 = right ? L.x + 9 : L.x - 9 - twd, x1 = x0 + twd;
          lab.forEach((q) => { if (q !== L && q.x > x0 - 6 && q.x < x1 + 6 && Math.abs(q.y - (ly - 4)) < 11) ly = q.y + 20; });
          L.ly = ly; placed.push(L);
        });
        lab.forEach((L) => {
          const col = L.p.f.col(C); const nm = typeof L.p.f.name === 'function' ? L.p.f.name() : L.p.f.name; const sel = L.p.f.id === S.zoom;
          V.glow(g, L.x, L.y, sel ? 16 : 10, col, sel ? 0.9 : 0.5); V.dot(g, L.x, L.y, sel ? 5.5 : 4.5, col);
          const right = L.x < xR - 130;
          V.text(g, `${nm} ${L.p.sq.toFixed(1).replace('.', ',')} dB`, right ? L.x + 9 : L.x - 9, L.ly, { size: 11.5, font: 'mono', color: col, align: right ? 'left' : 'right', weight: sel ? 600 : 400 });
        });
        // painel de zoom
        const zy0 = chartH + 34, zH = h - zy0 - 6; const zp = pts.find((p) => p.f.id === S.zoom) || pts[3];
        const blk = xs.slice(zoomBlock, zoomBlock + 32), dq = zp.deq.slice(zoomBlock, zoomBlock + 32);
        const am = Math.max(...blk.map(Math.abs), 1e-9); const base = zy0 + zH / 2; const sc = (zH / 2 - 4) / am; const bw = (w - 2 * pad) / 32;
        V.line(g, pad, base, w - pad, base, { color: V.alpha(C.muted, 0.4), lw: 1 });
        const col = zp.f.col(C);
        for (let k = 0; k < 32; k++) { const x = pad + k * bw + bw * 0.2, ww = bw * 0.6; const o = blk[k] * sc; g.fillStyle = V.alpha(C.muted, 0.3); g.fillRect(x, base - Math.max(0, o), ww, Math.abs(o)); V.dot(g, x + ww / 2, base - dq[k] * sc, Math.max(2.2, Math.min(3.6, bw * 0.22)), col); }
        const nm = typeof zp.f.name === 'function' ? zp.f.name() : zp.f.name;
        V.text(g, `zoom: bloco de 32 pesos com o maior outlier · ${nm} (pontos) vs original (barras)`, w / 2, zy0 - 6, { size: 11.5, font: 'mono', align: 'center', color: C.muted, maxW: w - 2 * pad });
      });
    });

    /* ===================================================================
       Figura 8 — W4A16 × W4A4 e o dispatch por M
       =================================================================== */
    safe(function figDispatch() {
      const fig = $('#fig-dispatch'); if (!fig) return;
      let need8 = 0;
      const refit8 = fitH(fig, (w) => (w < 640 ? Math.max(380, need8 || 440) : 318));
      const S = { lm: Math.log2(7), ck: 'nvfp4', plug: false };
      const Kd = 4096, Nd = 4096, BW = 250e9, P16 = 125e12, P4 = 500e12;
      const model = (M) => {
        const bw = Kd * Nd * (S.ck === 'nvfp4' ? 4.5 : 4.25) / 8;
        const F = 2 * M * Kd * Nd;
        const t16 = Math.max((bw + M * Kd * 2 + M * Nd * 2) / BW, F / P16);
        const t4 = Math.max((bw + M * Kd * 0.5625 + M * Nd * 2) / BW, F / P4) + M * Kd * 2.5625 / BW + 3e-6;
        return { t16, t4 };
      };
      let cross = 0; for (let M = 1; M <= 8192; M++) { const m = model(M); if (m.t4 < m.t16) { cross = M; break; } }
      const recompute = () => { cross = 0; for (let M = 1; M <= 8192; M++) { const m = model(M); if (m.t4 < m.t16) { cross = M; break; } } };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'M (tokens)', min: 0, max: 13, step: 0.05, value: S.lm, fmt: (v) => String(Math.round(Math.pow(2, v))), onInput: (v) => { S.lm = v; } });
      UI.seg(ctl, { label: 'Checkpoint', options: [{ v: 'nvfp4', label: 'NVFP4 (ModelOpt)' }, { v: 'mxfp4', label: 'MXFP4 fp4_e8m0_k32' }], value: 'nvfp4', onChange: (v) => { S.ck = v; recompute(); } });
      UI.toggle(ctl, { label: 'plugin GB10 (modelopt_gb10_hybrid)', value: false, onChange: (v) => { S.plug = v; } });
      const parts = Array.from({ length: 26 }, (_, k) => ({ u: k / 26 }));
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const pad = 16; const narrow = w < 640;
        const M = Math.max(1, Math.round(Math.pow(2, S.lm)));
        const m = model(M);
        const blocked = S.ck === 'mxfp4';
        const chosen = blocked ? 'a' : S.plug ? (M < 128 ? 'a' : 'b') : 'b';
        const regime = M <= 64 ? 'decode (poucos tokens por passo)' : M < 1024 ? 'misto' : 'prefill (chunks de 1024–8192)';
        V.text(g, `M = ${M} · ${regime}`, pad, 18, { size: 13, font: 'mono', weight: 600, color: C.ink });
        const laneW = narrow ? w - 2 * pad : w * 0.6 - pad;
        const lanes = [
          { id: 'a', name: 'W4A16 (só o peso é 4 bits)', stages: ['pesos FP4 + escalas', 'dequant em registrador → BF16', 'MMA BF16'], side: 'ativação BF16 →', col: C.b, t: m.t16 },
          { id: 'b', name: 'W4A4 (peso e ativação FP4)', stages: ['ativação BF16', 'quantiza → FP4 + escala/16', 'MMA FP4 block-scaled'], side: '← pesos FP4', col: C.a, t: m.t4 },
        ];
        const lh = narrow ? 104 : 112;
        lanes.forEach((L, k) => {
          const y = 34 + k * (lh + 10); const sel = chosen === L.id; const off = L.id === 'b' && blocked;
          V.box(g, pad, y, laneW, lh, { r: 12, fill: V.alpha(L.col, sel ? 0.1 : 0.03), stroke: V.alpha(L.col, sel ? 0.9 : 0.35), lw: sel ? 2 : 1 });
          V.text(g, L.name + (sel ? '  ← escolhido' : ''), pad + 12, y + 18, { size: 12, font: 'mono', weight: 600, color: sel ? L.col : C.muted, maxW: laneW - 60 });
          const sw = (laneW - 24 - 2 * 18) / 3, sy = y + 32, sh = lh - 46;
          if (!off) L.stages.forEach((sName, j) => {
            const sx = pad + 12 + j * (sw + 18);
            V.box(g, sx, sy, sw, sh, { r: 8, fill: V.alpha(L.col, 0.08), stroke: V.alpha(L.col, 0.55), lw: 1 });
            wrap(g, sName, sx + sw / 2, sy + sh / 2 - (narrow ? 8 : 6), sw - 10, 14, { size: 11, align: 'center', color: C.ink, maxLines: 3 });
            if (j < 2) V.arrow(g, sx + sw + 2, sy + sh / 2, sx + sw + 16, sy + sh / 2, { color: V.alpha(L.col, 0.8), lw: 1.4, head: 6 });
          });
          // partículas
          if (!off) {
            const speed = 0.25 * Math.min(4, (m.t16 + m.t4) / 2 / L.t) * (sel ? 1 : 0.45);
            parts.forEach((pp, q) => { const u = (pp.u + t * speed) % 1; const px = pad + 12 + u * (laneW - 24); const py = sy + sh + 7 + ((q % 2) ? 2 : -2); V.dot(g, px, py, 2.2, V.alpha(L.col, sel ? 0.9 : 0.35)); });
          }
          // semáforo
          if (L.id === 'b') {
            const lx = pad + laneW - 30, ly = y + 12; V.box(g, lx - 2, ly - 4, 26, 16, { r: 8, fill: V.alpha(C.ink, 0.08) });
            V.dot(g, lx + 5, ly + 4, 4.5, blocked ? C.red : V.alpha(C.red, 0.2)); V.dot(g, lx + 17, ly + 4, 4.5, blocked ? V.alpha(C.e, 0.2) : C.e);
            if (off) { g.strokeStyle = V.alpha(C.red, 0.5); g.lineWidth = 1; g.setLineDash([5, 5]); V.rr(g, pad + 12, y + 30, laneW - 24, lh - 42, 8); g.stroke(); g.setLineDash([]); wrap(g, 'bloqueado: o caminho FP4 de tensor core do b12x só aceita source_format = modelopt_nvfp4 → W4A16FusedMoeKernel', pad + laneW / 2, y + lh / 2 + 4, laneW - 30, 15, { size: 11.5, align: 'center', color: C.red, weight: 600, maxLines: 3 }); }
          }
        });
        // barras de tempo
        const bx0 = narrow ? pad : w * 0.6 + 14, bw0 = narrow ? w - 2 * pad : w - bx0 - pad;
        const by0 = narrow ? 34 + 2 * (lh + 10) + 6 : 34;
        V.text(g, 'tempo do GEMM 4096×4096 (modelo roofline ilustrativo)', bx0, by0 + 10, { size: 11, font: 'mono', color: C.muted, maxW: bw0 });
        const tmax = Math.max(m.t16, m.t4);
        [['W4A16', m.t16, C.b, 'a'], ['W4A4', m.t4, C.a, 'b']].forEach(([nm, tt, col, id], k) => {
          const y = by0 + 22 + k * 34; const len = (bw0 - 116) * tt / tmax; const sel = chosen === id; const off = id === 'b' && blocked;
          V.text(g, nm, bx0, y + 14, { size: 11.5, font: 'mono', color: col, weight: sel ? 600 : 400 });
          V.box(g, bx0 + 50, y + 2, Math.max(3, len), 18, { r: 5, fill: V.alpha(off ? C.muted : col, sel ? 0.8 : 0.35), stroke: sel ? col : null });
          V.text(g, (tt * 1e6).toFixed(tt * 1e6 < 100 ? 1 : 0).replace('.', ',') + ' µs', bx0 + 56 + Math.max(3, len), y + 15, { size: 11, font: 'mono', color: C.ink });
        });
        // eixo de M
        const ay = narrow ? by0 + 22 + 2 * 34 + 26 : by0 + 22 + 2 * 34 + 40; const ax0 = bx0 + 4, ax1 = bx0 + bw0 - 8;
        const AX = (mm) => ax0 + Math.log2(mm) / 13 * (ax1 - ax0);
        V.line(g, ax0, ay, ax1, ay, { color: V.alpha(C.muted, 0.6), lw: 1 });
        [1, 8, 64, 512, 4096].forEach((mm) => { V.line(g, AX(mm), ay - 4, AX(mm), ay + 4, { color: C.muted }); V.text(g, String(mm), AX(mm), ay + 16, { size: 11, font: 'mono', align: 'center', color: C.muted }); });
        if (cross) { V.line(g, AX(cross), ay - 14, AX(cross), ay, { color: C.d, lw: 1.5, dash: [3, 3] }); V.text(g, 'cruzamento ≈ ' + cross, AX(cross), ay - 18, { size: 11, font: 'mono', align: 'center', color: C.d }); }
        if (S.plug && !blocked) { V.line(g, AX(128), ay - 4, AX(128), ay + 26, { color: C.c, lw: 2 }); V.text(g, narrow ? 'plugin: 128' : 'limiar do plugin = 128', AX(128) + (narrow ? 6 : 0), narrow ? ay + 26 : ay + 38, { size: 11, font: 'mono', align: narrow ? 'left' : 'center', color: C.c }); }
        g.fillStyle = C.c; const mxp = AX(M); g.beginPath(); g.moveTo(mxp - 6, ay - 10); g.lineTo(mxp + 6, ay - 10); g.lineTo(mxp, ay - 2); g.closePath(); g.fill();
        const NOTES = { mx: 'MXFP4: sempre W4A16. Ótimo no decode; no prefill a MMA é BF16. MXFP4×MXFP8 (mxf8f6f4, k32) daria ~2× (taxa FP8, interpretação); ~4× só com FP4×FP4.', plug: 'Plugin: M < 128 → Marlin W4A16 (apply_fp4_marlin_linear); M ≥ 128 → CUTLASS W4A4 (scaled_fp4_quant + cutlass_scaled_fp4_mm).', nv: 'NVFP4 W4A4 sem plugin: sempre o caminho FP4 (experts do Qwen no FlashInfer CUTLASS SM120).' };
        const note = blocked ? NOTES.mx : S.plug ? NOTES.plug : NOTES.nv;
        const noteY = ay + (S.plug && !blocked && !narrow ? 58 : 38 + (narrow && S.plug && !blocked ? 8 : 0));
        wrap(g, note, bx0, noteY, bw0, 15, { size: 11.5, color: C.ink, maxLines: 5 });
        if (narrow) { const nMax = Math.min(5, Math.max(...Object.values(NOTES).map((tx) => splitLines(g, tx, bw0, { size: 11.5 }).length))); const nd = Math.round(ay + 46 + (nMax - 1) * 15 + 16); if (Math.abs(nd - need8) > 2) { need8 = nd; refit8(); } }
      });
    });

    /* ===================================================================
       Figura 9 — O porteiro do kernel
       =================================================================== */
    safe(function figPorteiro() {
      const fig = $('#fig-porteiro'); if (!fig) return;
      let need9 = 0;
      const refit9 = fitH(fig, (w) => need9 || (w < 640 ? 420 : 300));
      const LAYERS = [
        { id: 'inproj', name: 'GDN in_proj_a/b [48×2560]', N: 48, K: 2560, split: 'col' },
        { id: 'fc1', name: 'Visão fc1 [4304×1152]', N: 4304, K: 1152, split: 'col', vis: true },
        { id: 'fc2', name: 'Visão fc2 [1152×4304]', N: 1152, K: 4304, split: 'row', vis: true },
        { id: 'oproj', name: 'QSA o_proj [2560×6144]', N: 2560, K: 6144, split: 'row' },
        { id: 'shared', name: 'shared gate [640×2560]', N: 640, K: 2560, split: 'col' },
        { id: 'hc', name: 'HC mix_down [320×10240]', N: 320, K: 10240, split: 'none' },
        { id: 'expert', name: 'expert EP [640×2560]', N: 640, K: 2560, split: 'ep' },
        { id: 'expnoep', name: 'expert MTP TP [640×2560]', N: 640, K: 2560, split: 'col', mtp: true },
      ];
      const FMT = {
        mxfp8: { name: 'MXFP8 · FlashInfer mm_mxfp8', rules: (n, k) => [['N ≥ 128   (N = ' + n + ')', n >= 128], ['N % 32 = 0   (' + n + ' % 32 = ' + (n % 32) + ')', n % 32 === 0], ['K ≥ 128   (K = ' + k + ')', k >= 128], ['K % 32 = 0   (' + k + ' % 32 = ' + (k % 32) + ')', k % 32 === 0]], ok: 'kernel nativo FlashInfer mm_mxfp8', fail: (L) => (L.vis ? 'visual.* → emulação BF16 (EmulationMxfp8LinearKernel), dequant na carga' : 'EmulationMxfp8LinearKernel: dequant MXFP8 → BF16 uma vez, na carga (+2 B por peso)') },
        nvfp4: { name: 'NVFP4 · bloco de 16 em K', rules: (n, k) => [['K % 16 = 0   (' + k + ' / 16 = ' + nice(k / 16, 2) + ')', k % 16 === 0]], ok: 'kernel nativo FP4 (CUTLASS/FlashInfer)', fail: (L) => (L.vis ? 'segundo o comentário do repo (.env.sample:89-91), "weights" crasha na carga: replique o encoder com MM_ENCODER_TP_MODE=data' : 'sem kernel NVFP4 nesta forma: mantenha BF16 ou mude TP/EP') },
        fp8b: { name: 'FP8 em blocos 128×128', rules: (n, k) => [['N % 128 = 0   (' + n + ' % 128 = ' + (n % 128) + ')', n % 128 === 0], ['K % 128 = 0   (' + k + ' % 128 = ' + (k % 128) + ')', k % 128 === 0]], ok: 'GEMM FP8 com escalas por bloco', fail: (L) => (L.mtp ? 'o tile 128×128 cortaria o expert ao meio: ligue --enable-expert-parallel, e o expert fica inteiro (640 = 5 × 128), como pede o card da nvidia' : 'use FP8 por canal (FP8_PER_CHANNEL_PER_TOKEN), como o FP8-dense do Qwen') },
      };
      const S = { L: 'fc2', f: 'nvfp4', tp: 2, rep: false, t0: performance.now(), sx: 0 };
      const bump = () => { S.t0 = performance.now(); };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Camada', options: LAYERS.map((L) => ({ v: L.id, label: L.name })), value: S.L, onChange: (v) => { S.L = v; bump(); } });
      UI.seg(ctl, { label: 'Formato', options: [{ v: 'mxfp8', label: 'MXFP8' }, { v: 'nvfp4', label: 'NVFP4' }, { v: 'fp8b', label: 'FP8 128²' }], value: S.f, onChange: (v) => { S.f = v; bump(); } });
      UI.seg(ctl, { label: 'TP', options: [1, 2], value: 2, onChange: (v) => { S.tp = v; bump(); } });
      UI.toggle(ctl, { label: 'replicar encoder de visão (data)', value: false, onChange: (v) => { S.rep = v; bump(); } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const pad = 16; const narrow = w < 640;
        const L = LAYERS.find((x) => x.id === S.L); const F = FMT[S.f];
        const rep = L.vis && S.rep;
        let n = L.N, k = L.K; if (!rep && S.tp > 1) { if (L.split === 'col') n = L.N / S.tp; if (L.split === 'row') k = L.K / S.tp; }
        const rules = F.rules(n, k); const pass = rules.every((r) => r[1]);
        const prefixEmu = S.f === 'mxfp8' && L.vis;
        const el = (performance.now() - S.t0) / 1000;
        // matriz (área de altura fixa: o canvas se ajusta ao conteúdo)
        const mw = narrow ? w - 2 * pad : w * 0.34, mh = narrow ? 150 : 196;
        const sc = Math.min((mw - 60) / Math.sqrt(L.K), (mh - 50) / Math.sqrt(L.N));
        const RW = Math.max(14, Math.sqrt(L.K) * sc), RH = Math.max(10, Math.sqrt(L.N) * sc);
        const slide = V.ease.out(Math.min(1, el / 0.6));
        const mx0 = pad + 40 + (mw - 60 - RW) / 2 - (1 - slide) * 30, my0 = 30 + (mh - 50 - RH) / 2;
        V.text(g, 'matriz [N × K] = [' + L.N + ' × ' + L.K + ']', pad, 16, { size: 11.5, font: 'mono', color: C.muted, maxW: mw });
        V.box(g, mx0, my0, RW, RH, { r: 4, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.7), lw: 1.2 });
        let sw = RW, shh = RH;
        if (!rep && S.tp > 1 && L.split === 'col') { shh = RH / S.tp; V.line(g, mx0, my0 + shh, mx0 + RW, my0 + shh, { color: C.c, lw: 1.6, dash: [4, 3] }); }
        if (!rep && S.tp > 1 && L.split === 'row') { sw = RW / S.tp; V.line(g, mx0 + sw, my0, mx0 + sw, my0 + RH, { color: C.c, lw: 1.6, dash: [4, 3] }); }
        g.fillStyle = V.alpha(C.c, 0.18 + 0.08 * Math.sin(t * 3)); V.rr(g, mx0 + 1, my0 + 1, sw - 2, shh - 2, 3); g.fill();
        // grade de tiles 128×128 no formato FP8 em blocos: mostra o corte que cai no meio de um tile
        if (S.f === 'fp8b') {
          const nT = Math.ceil(L.N / 128), kT = Math.ceil(L.K / 128);
          if (nT <= 40 && kT <= 40) {
            g.strokeStyle = V.alpha(C.d, 0.5); g.lineWidth = 0.8; g.beginPath();
            for (let a = 1; a < nT; a++) { const yy = my0 + RH * (a * 128) / L.N; g.moveTo(mx0, yy); g.lineTo(mx0 + RW, yy); }
            for (let a = 1; a < kT; a++) { const xx = mx0 + RW * (a * 128) / L.K; g.moveTo(xx, my0); g.lineTo(xx, my0 + RH); }
            g.stroke();
          }
        }
        V.text(g, 'N', mx0 - 12, my0 + RH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.muted });
        V.text(g, 'K', mx0 + RW / 2, my0 + RH + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const splitTxt = rep ? 'encoder replicado: forma inteira' : L.split === 'ep' ? 'EP: o expert fica inteiro no rank' : L.split === 'none' ? 'sem fatiamento nesta figura' : S.tp === 1 ? 'TP = 1: forma inteira' : L.split === 'col' ? 'column-parallel: N ÷ ' + S.tp + ' (saídas)' : 'row-parallel: K ÷ ' + S.tp + ' (entradas)';
        V.text(g, splitTxt, pad, my0 + RH + 32, { size: 11, font: 'mono', color: C.muted, maxW: mw });
        V.text(g, `por rank: ${n} × ${k}`, pad, my0 + RH + 48, { size: 12.5, font: 'mono', weight: 600, color: C.c, maxW: mw });
        if (S.f === 'fp8b') V.text(g, 'linhas âmbar: tiles de 128', pad, my0 + RH + 64, { size: 11, font: 'mono', color: C.d, maxW: mw });
        const matBottom = my0 + RH + (S.f === 'fp8b' ? 70 : 54);
        // porteiro + regras
        const rx0 = narrow ? pad : w * 0.36 + 12, ry0 = narrow ? 30 + (mh - 50) + 74 : 22, rw = narrow ? w - 2 * pad : w - rx0 - pad;
        const bx = rx0 + 26, by = ry0 + 28;
        const done = el > 0.35 * rules.length + 0.4;
        const eyeCol = !done ? C.d : pass && !prefixEmu ? C.e : C.red;
        V.box(g, bx - 22, by - 20, 44, 34, { r: 10, fill: V.alpha(C.muted, 0.15), stroke: C.muted, lw: 1.2 });
        V.line(g, bx, by - 20, bx, by - 30, { color: C.muted, lw: 1.5 }); V.dot(g, bx, by - 32, 3.5, eyeCol);
        const blink = Math.sin(t * 2.3) > 0.97 ? 0.3 : 1;
        V.box(g, bx - 13, by - 9, 9, 9 * blink, { r: 3, fill: eyeCol }); V.box(g, bx + 4, by - 9, 9, 9 * blink, { r: 3, fill: eyeCol });
        V.box(g, bx - 16, by + 18, 32, 26, { r: 6, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.6), lw: 1 });
        V.text(g, F.name, bx + 34, by - 6, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: rw - 60 });
        V.text(g, 'confere a forma que o kernel verá', bx + 34, by + 12, { size: 11, color: C.muted, maxW: rw - 60 });
        const rc = rules.length > 2 ? 2 : 1, rRows = Math.ceil(rules.length / rc), colW = rw / rc;
        rules.forEach(([txt0, ok], j) => {
          const txt = narrow ? txt0.replace(/ {2,}/g, ' ').replace(/ ([%=≥/]) /g, '$1') : txt0;
          const x0 = rx0 + (j % rc) * colW, y = by + 60 + Math.floor(j / rc) * 26; const lit = el > 0.35 * (j + 1);
          const col = !lit ? V.alpha(C.muted, 0.4) : ok ? C.e : C.red;
          V.dot(g, x0 + 12, y - 4, 7, col); if (lit) V.glow(g, x0 + 12, y - 4, 14, col, 0.5);
          V.text(g, txt + (lit ? (ok ? ' ✓' : ' ✗') : ''), x0 + 28, y, { size: narrow ? 11.5 : 12, font: 'mono', color: lit ? C.ink : C.muted, maxW: colW - 34 });
        });
        const vy = by + 60 + rRows * 26 + 6;
        const good = pass && !prefixEmu;
        const vt = good ? '✓ ' + F.ok : prefixEmu && pass ? '→ emulação BF16 por regra de prefixo (visual.*), mesmo passando nas formas: evita o OOM de desquantizar tudo' : '✗ ' + F.fail(L);
        const vLines = Math.min(4, splitLines(g, vt, rw - 24, { size: 12, weight: 500 }).length);
        const vh = 18 + vLines * 16;
        if (done) {
          V.box(g, rx0, vy, rw, vh, { r: 10, fill: V.alpha(good ? C.e : C.red, 0.12), stroke: good ? C.e : C.red, lw: 1.4 });
          wrap(g, vt, rx0 + 12, vy + 20, rw - 24, 16, { size: 12, color: C.ink, weight: 500, maxLines: 4 });
        }
        // altura: o maior entre a matriz e o veredito (2 linhas de regras e 3 de veredito no pior caso, para não pular ao trocar o formato)
        const vyMax = by + 60 + 2 * 26 + 6 + 18 + 3 * 16;
        const nd = Math.round(Math.max(narrow ? 0 : matBottom, vyMax, vy + vh) + 12);
        if (Math.abs(nd - need9) > 2) { need9 = nd; refit9(); }
      });
    });

    /* ===================================================================
       Figura 10 — Mapa de formatos das três receitas
       =================================================================== */
    safe(function figMapa() {
      const fig = $('#fig-mapa'); if (!fig) return;
      fitH(fig, (w) => (w < 640 ? 606 : 452));
      const S = { glm: true, qwen: false, ds: false, eng: 'vllm', dense: 'q4', sel: null, lock: null, flash: {} };
      const fam = (C) => ({ bf16: C.muted, fp8: C.d, nvfp4: C.a, mxfp4: C.f, exl3: C.b, int4: C.c, kv: C.ink, none: C.muted });
      const tfc = (C) => C.__tf || (C.__tf = (getComputedStyle(document.documentElement).getPropertyValue('--tf') || '').trim() || C.d);
      // rótulos curtos para o celular (o detalhe completo vai para o painel de baixo)
      const SHORT = { 'emb BF16 · head FP8': 'head FP8', 'nvfp4_ds_mla · 584 B': 'nvfp4_ds_mla', 'fp8_ds_mla · 656 B': 'fp8_ds_mla', 'FP8 128² · UE8M0': 'FP8 128²', 'MTP: FP8 128²': 'FP8 128²', 'FP8 2ᵏ/linha · 528 B': 'FP8 2ᵏ/linha', 'INT4 afim · g64': 'INT4 afim', 'FP8 1×128 · W8A16': 'FP8 1×128', 'DFlash2 · 4 bits': 'DFlash2 4b' };
      function hatch(g, x, y, ww, hh, col) {
        g.save(); V.rr(g, x, y, ww, hh, 8); g.clip();
        g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
        for (let d = -hh; d < ww; d += 7) { g.moveTo(x + d, y + hh); g.lineTo(x + d + hh, y); }
        g.stroke(); g.restore();
      }
      function dots(g, x, y, ww, hh, col) {   // INT4 pontilhado: separa do MXFP4 mesmo quando os rosas se parecem
        g.save(); V.rr(g, x, y, ww, hh, 8); g.clip(); g.fillStyle = col;
        for (let yy = y + 4; yy < y + hh; yy += 6) for (let xx = x + 4 + ((yy - y) % 12 ? 3 : 0); xx < x + ww; xx += 6) { g.beginPath(); g.arc(xx, yy, 1.1, 0, 6.2832); g.fill(); }
        g.restore();
      }
      const ROWS = ['Experts roteados', 'Projeções densas / atenção', 'Shared expert', 'Embeddings e lm_head', 'MTP / rascunhador', 'Tabela PLE', 'KV cache'];
      const COLS = [{ id: 'glm', name: 'GLM-5.3' }, { id: 'qwen', name: 'Qwen3.8' }, { id: 'ds', name: 'DeepSeek-V4' }];
      const colName = (j) => (j === 0 && S.eng === 'tf' ? 'GLM·TF' : COLS[j].name);
      function cell(r, c) {
        const g = S.glm, q = S.qwen, d = S.dense;
        const T = {
          glm: [
            ['exl3', 'EXL3 4 bpw', 'Experts roteados em EXL3 K = 4 (codebook MCG: treliça + Hadamard), W4A16 com MMA FP16. KLD 0,024555 ≈ FP8 oficial com 54% dos bytes (capítulo 22).'],
            g ? ['fp8', 'FP8 canal · W8A16', 'GLM53_DENSE_FP8=all: kda, mla e as MLPs densas das camadas 0–2 viram E4M3 por canal (amax/448) no Marlin. kv_b_proj fica BF16 por causa da absorção do MLA.'] : ['bf16', 'BF16', 'Sem GLM53_DENSE_FP8 (o start.sh assume off): projeções densas BF16, como no checkpoint.'],
            g ? ['fp8', 'FP8 canal', 'Grupo shared: mlp.shared_experts.{gate_up_proj, down_proj} em FP8 W8A16 no Marlin.'] : ['bf16', 'BF16', 'Shared experts em BF16 (grupo shared desligado).'],
            ['bf16', 'BF16', 'Embeddings e lm_head nativos BF16 no checkpoint; o GLM53_DENSE_FP8 não os toca.'],
            ['bf16', 'BF16 (DFlash2)', 'Rascunhador DFlash2 em BF16 (~2,3 GiB) com KV BF16. Drafter, MTP e visão nunca são convertidos pelo GLM53_DENSE_FP8.'],
            ['none', '—', 'O GLM não tem tabela PLE.'],
            ['kv', 'fp8_ds_mla · 656 B', 'KV do alvo: 512 FP8 E4M3 + 4 escalas FP32 + 64 BF16 de RoPE zerada = 656 B por token por camada (capítulo 23).'],
          ],
          tf: [
            ['exl3', 'EXL3 4 bpw', 'Mesmo checkpoint: experts roteados em EXL3 de 4 bpw, lidos pelos kernels EXL3 do próprio TensorFold. O DENSE não toca nos experts.'],
            d === 'q4' ? ['int4', 'INT4 afim · g64', 'DENSE=q4 (padrão): atenção, KDA e MLPs densas em 4 bits inteiros afins, grupos de 64 com a faixa escolhida por MSE (~4,5 bits por peso, conta nossa). Mais rápido que o FP8 (prosa 38,9 → 44,4 tok/s), mas perde o fim de turno em prompts curtos de código em francês (issue #18). O kv_b fica BF16.']
              : d === 'fp8' ? ['fp8', 'FP8 1×128 · W8A16', 'DENSE=fp8: E4M3 com escala FP32 por linha e bloco de 128 colunas (8,25 bits), num kernel Triton próprio com ativação BF16. ~+33% de decode sobre BF16; a receita o indica para prompts que não estão em inglês. O kv_b fica BF16.']
              : ['bf16', 'BF16', 'DENSE=bf16: os pesos densos como estão no checkpoint (~9,7 GiB por Spark, contando shared experts e cabeça).'],
            d === 'q4' ? ['int4', 'INT4 afim · g64', 'Os shared experts fazem parte do DENSE: no q4, 4 bits afins em grupos de 64.'] : d === 'fp8' ? ['fp8', 'FP8 1×128', 'Shared experts em FP8 com escala FP32 por linha e bloco de 128 colunas.'] : ['bf16', 'BF16', 'Shared experts em BF16, como no checkpoint.'],
            d === 'bf16' ? ['bf16', 'BF16', 'Com DENSE=bf16, a cabeça que verifica os rascunhos fica em BF16; os passos de rascunho usam uma cópia de 4 bits dela. Embeddings em BF16.'] : ['fp8', 'emb BF16 · head FP8', 'No q4 e no fp8, a cabeça fica em FP8 1×128 e é ela que verifica os rascunhos; os passos de rascunho usam uma cópia de 4 bits. Embeddings em BF16.'],
            ['int4', 'DFlash2 · 4 bits', 'O DFlash2 ocupa 0,63 GiB por rank em cópias afins de 4 bits (grupos de 64, faixa min/max; TF_GLM_DRAFT_QUANT=mse troca por recorte por MSE). Cada rascunho é conferido contra a amostra do alvo: a precisão do rascunhador muda a aceitação, nunca a resposta.'],
            ['none', '—', 'O GLM não tem tabela PLE.'],
            ['kv', 'FP8 2ᵏ/linha · 528 B', 'KV=fp8: o latente DSA em linhas de 512 códigos E4M3 + escala FP32 potência de 2 (2^(⌈log₂ amax⌉ − 8)) + 12 B de alinhamento = 528 B por token e camada, sem a região RoPE zerada do fp8_ds_mla (656 B). KV=bf16 é a referência exata, com janela de 196.608 tokens.'],
          ],
          qwen: [
            ['nvfp4', 'NVFP4 · W4A4', 'Experts roteados NVFP4 (E2M1 + E4M3 a cada 16 + FP32 global), W4A4 no FlashInfer CUTLASS SM120, EP obrigatório. 63,28 GiB.'],
            q ? ['fp8', 'FP8 canal · W8A8', 'FP8_DENSE=true: 591 lineares FP8_PER_CHANNEL_PER_TOKEN (CUTLASS), ativação dinâmica por token. 9,2 → 5,7 GiB/passo analítico, nunca medido em GPU.'] : ['bf16', 'BF16', 'Checkpoint nvidia: GDN, QSA e HyperConnection em BF16 (ignore do ModelOpt). O denso BF16 soma ~6,9 dos ~9,2 GiB lidos por passo.'],
            q ? ['fp8', 'FP8 canal', 'O shared expert entra nas lineares FP8 do FP8-dense; o shared_expert_gate fica BF16.'] : ['bf16', 'BF16', 'shared_expert [640 × 2560] em BF16.'],
            q ? ['fp8', 'emb BF16 · head FP8', 'FP8-dense: o lm_head passa a receber quant_config (FP8 por canal); embed_tokens continua BF16.'] : ['bf16', 'BF16', 'embed_tokens e lm_head [248320 × 2560] em BF16, 1,184 GiB cada.'],
            ['fp8', 'MTP: FP8 128²', 'Experts do MTP em FP8 com blocos 128×128 (FP8_BLOCK_SCALES, weight_scale_inv), via Fp8MoEMethod graças ao patch_modelopt_fp8_block_moe.py.'],
            ['fp8', 'FP8 por tensor', 'PLE de 47,68 GiB em E4M3 com uma única weight_scale (checkpoint nvidia). No checkpoint local-inference-lab é NVFP4 empacotado (90 B por linha).'],
            ['kv', 'FP8 por tensor', 'KV das camadas QSA em E4M3 com escala por tensor (patch_qsa_fp8_kv): 3.652.200 tokens contra 2.131.159 em BF16 (capítulo 23).'],
          ],
          ds: [
            ['mxfp4', 'MXFP4 · W4A16', 'fp4_e8m0_k32: E2M1 + E8M0 a cada 32, 13,4 MB por expert. O b12x roda W4A16 (dequant → MMA BF16).'],
            ['fp8', 'FP8 128² · UE8M0', 'Atenção e demais lineares em E4M3 com escala UE8M0 por bloco 128×128 (weight_scale_inv); ativações com escala dinâmica por 128. ≈ 2,4 GB/rank/passo.'],
            ['fp8', 'FP8 128²', 'Especialista compartilhado em FP8 (weight_scale_inv), mapeado pelo expert_dtype.'],
            ['bf16', 'BF16', 'lm_head BF16 129.280 × 4.096 = 1,06 GB; embeddings BF16.'],
            ['fp8', 'FP8 + BF16', 'Rascunhador DSpark de 3 camadas DSV4: a saída da atenção é quantizada para FP8 antes do wo_a (einsum FP8); Markov head em BF16 (66 MB por matriz).'],
            ['none', '—', 'O DeepSeek não tem tabela PLE.'],
            ['kv', 'nvfp4_ds_mla · 584 B', 'Apesar do nome, byte a byte igual ao fp8_ds_mla: 448 FP8 (UE8M0 por 64) + 64 BF16 de RoPE + 8 B de escalas (capítulo 23).'],
          ],
        };
        return T[c === 0 && S.eng === 'tf' ? 'tf' : COLS[c].id][r];
      }
      const flash = (c, rows) => { const now = performance.now(); rows.forEach((r) => { S.flash[r + ',' + c] = now; }); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Motor do GLM', options: [{ v: 'vllm', label: 'vLLM' }, { v: 'tf', label: 'TensorFold' }], value: 'vllm', onChange: (v) => { S.eng = v; gV.show(v === 'vllm'); gT.show(v === 'tf'); flash(0, [0, 1, 2, 3, 4, 6]); S.lock = [1, 0]; } });
      const gV = UI.group(ctl), gT = UI.group(ctl);
      UI.toggle(gV.el, { label: 'GLM53_DENSE_FP8=all', value: true, onChange: (v) => { S.glm = v; flash(0, [1, 2]); S.lock = [1, 0]; } });
      UI.seg(gT.el, { label: 'DENSE (TF)', options: [{ v: 'q4', label: 'q4' }, { v: 'fp8', label: 'fp8' }, { v: 'bf16', label: 'bf16' }], value: 'q4', onChange: (v) => { S.dense = v; flash(0, [1, 2, 3]); S.lock = [1, 0]; } });
      gT.show(false);
      UI.toggle(ctl, { label: 'FP8_DENSE=true (Qwen)', value: false, onChange: (v) => { S.qwen = v; flash(1, [1, 2, 3]); S.lock = [1, 1]; } });
      UI.toggle(ctl, { label: 'ENABLE_VLLM_GB10_PATCH=1 (DS)', value: false, onChange: (v) => { S.ds = v; S.lock = [-1, 2]; } });
      UI.legend(ctl, [['var(--viz-muted)', 'BF16'], ['var(--viz-d)', 'FP8'], ['var(--viz-a)', 'NVFP4'], ['var(--viz-f)', 'MXFP4'], ['var(--viz-b)', 'EXL3'], ['radial-gradient(circle, var(--viz-c) 0 1.4px, transparent 1.7px) 0 0 / 4px 4px', 'INT4 afim (q4, pontilhado)'], ['repeating-linear-gradient(135deg, var(--viz-ink) 0 1.5px, transparent 1.5px 4px)', 'KV cache (hachurado)']]);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const FC = fam(C); const pad = 16; const narrow = w < 640;
        const labW = narrow ? 0 : 196; const gx0 = pad + labW; const cw = (w - gx0 - pad - 2 * 8) / 3;
        const headH = 28; const rowH = narrow ? 58 : 44; const chip = narrow ? 32 : 34;
        COLS.forEach((c, j) => { const warn = S.ds && j === 2; V.text(g, colName(j) + (warn ? ' ⚠' : ''), gx0 + j * (cw + 8) + cw / 2, 18, { size: 12.5, font: 'mono', weight: 600, align: 'center', color: warn ? C.red : [S.eng === 'tf' ? tfc(C) : C.glm, C.qwen, C.ds][j] || C.ink, maxW: cw }); });
        const p = H.pointer; let hov = null;
        const now = performance.now();
        ROWS.forEach((rn, r) => {
          const y = headH + r * rowH;
          if (narrow) V.text(g, rn, pad, y + 12, { size: 11, font: 'mono', color: C.muted });
          else V.text(g, rn, pad, y + chip / 2 + 5, { size: 12, color: C.ink, maxW: labW - 10 });
          for (let j = 0; j < 3; j++) {
            const x = gx0 + j * (cw + 8), yy = y + (narrow ? 18 : 4);
            const [fm, lb] = cell(r, j); const col = FC[fm];
            const inside = p.inside && p.x >= x && p.x <= x + cw && p.y >= yy && p.y <= yy + chip;
            if (inside) hov = [r, j];
            const fl = S.flash[r + ',' + j]; const fa = fl ? Math.max(0, 1 - (now - fl) / 1400) : 0;
            const sel = S.sel && S.sel[0] === r && S.sel[1] === j;
            if (fm === 'none') { g.setLineDash([2, 4]); V.box(g, x, yy, cw, chip, { r: 8, stroke: sel ? C.ink : V.alpha(C.muted, 0.5), lw: sel ? 2 : 1 }); g.setLineDash([]); }
            else if (fm === 'kv') { V.box(g, x, yy, cw, chip, { r: 8, fill: V.alpha(C.ink, 0.05 + 0.12 * fa) }); hatch(g, x, yy, cw, chip, V.alpha(C.ink, 0.16)); g.setLineDash([5, 3]); V.box(g, x, yy, cw, chip, { r: 8, stroke: sel ? C.ink : V.alpha(C.ink, 0.7), lw: sel ? 2.2 : 1.2 }); g.setLineDash([]); }
            else { V.box(g, x, yy, cw, chip, { r: 8, fill: V.alpha(col, 0.2 + 0.25 * fa), stroke: sel ? C.ink : V.alpha(col, 0.9), lw: sel ? 2.2 : 1.1 }); if (fm === 'int4') dots(g, x, yy, cw, chip, V.alpha(col, 0.3)); }
            if (fa > 0 && cw > 2) V.glow(g, x + cw / 2, yy + chip / 2, cw * 0.6, col, fa * 0.8);
            const fsz = narrow ? 11 : 12; let lbs = lb;
            if (narrow && tw(g, lbs, { size: fsz, font: 'mono' }) > cw - 8) lbs = SHORT[lb] || lb.split(' · ')[0];
            V.text(g, lbs, x + cw / 2, yy + chip / 2 + 1, { size: fsz, font: 'mono', align: 'center', base: 'middle', color: fm === 'none' ? C.muted : C.ink, maxW: cw - 8 });
          }
        });
        if (S.ds) {
          const x = gx0 + 2 * (cw + 8) - 4, y0 = headH - 2, y1 = headH + (ROWS.length - 1) * rowH + (narrow ? 18 : 4) + chip + 4;
          g.setLineDash([6, 4]); g.lineDashOffset = -t * 10; V.box(g, x, y0, cw + 8, y1 - y0, { r: 10, stroke: C.red, lw: 1.8 }); g.setLineDash([]); g.lineDashOffset = 0;
        }
        if (p.clicked && hov) S.lock = hov;
        H.canvas.style.cursor = hov ? 'pointer' : 'default';
        const total = ROWS.length * 3;
        const auto = Math.floor(t / 2.6) % total;
        S.sel = hov || S.lock || [Math.floor(auto / 3), auto % 3];
        // painel de detalhe
        const dy = headH + ROWS.length * rowH + 10;
        V.box(g, pad, dy, w - 2 * pad, h - dy - 8, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        let title, text;
        if (S.sel[0] === -1) { title = 'DeepSeek · ENABLE_VLLM_GB10_PATCH=' + (S.ds ? '1' : '0'); text = S.ds ? '⚠ Não se aplica ao checkpoint padrão. O plugin só reivindica checkpoints ModelOpt mixed (config.py:36-49), e o DSV4 declara quant_method = fp8 (experts MXFP4, resto FP8). Interpretação: com --quantization modelopt_gb10_hybrid, o vLLM provavelmente recusa a combinação no boot. O mapa não muda porque o serviço nem sobe.' : 'Default da receita: plugin desligado. Ele é experimental, guarda duas cópias dos pesos, foi testado só no empacotamento e só serve a checkpoints ModelOpt NVFP4.'; }
        else { const [fm, lb, dsc] = cell(S.sel[0], S.sel[1]); title = colName(S.sel[1]) + ' · ' + ROWS[S.sel[0]] + ' · ' + lb; text = dsc; }
        V.text(g, title, pad + 12, dy + 20, { size: 12.5, font: 'mono', weight: 600, color: C.ink, maxW: w - 2 * pad - 24 });
        wrap(g, text, pad + 12, dy + 40, w - 2 * pad - 24, 16, { size: 12, color: C.ink, maxLines: narrow ? 7 : 4 });
      });
    });
  },
});

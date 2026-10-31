LA.chapter({
  id: 'prefill-decode-kv',
  part: 2,
  order: 10,
  title: 'Prefill, decode e o KV cache',
  navTitle: 'Prefill, decode & KV',
  subtitle: 'Por que ler o prompt é rápido e escrever a resposta é lento, o que o KV cache guarda, quanto cada token de contexto custa e como medir as duas fases.',
  minutes: 60,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['prefill', 'decode', 'KV cache', 'roofline', 'intensidade aritmética', 'TTFT', 'TPOT', 'batching', 'GEMV'],
  goals: [
    'Explicar por que o <b>prefill</b> é limitado por cálculo e o <b>decode</b> por banda, com GEMM × GEMV e o <b>roofline</b> do GB10 (ridge ≈ 458 FLOP/B)',
    'Prever o teto de decode de uma receita somando os <b>bytes lidos por passo</b> e dividindo pela banda',
    'Calcular <b>bytes de KV por token</b> e por sequência (\\(L_{kv} \\times n_{kv} \\times (d_k+d_v) \\times b\\)) e dizer o que acontece com eles sob TP',
    'Ler TTFT, TPOT, tok/s por stream e agregado e entender como o <b>batching</b> amortiza a leitura dos pesos, e por que contexto e concorrência disputam o mesmo pool',
  ],
  prereqs: ['geracao-amostragem', 'posicao-rope'],
  summary: [
    'Prefill processa o prompt inteiro em poucas GEMMs grandes (\\(M\\) = milhares de linhas); decode gera um token por sequência por passo (\\(M = B(1+k)\\), de 1 a algumas dezenas).',
    'Intensidade aritmética \\(I \\approx 2M/b_w\\) enquanto os bytes de ativação são pequenos. No GB10, com ~125 TFLOPS BF16 e 273 GB/s, o ridge fica em ≈458 FLOP/B: decode (I ≈ 1–4) é limitado pela memória. No MoE, cada expert só recebe \\(Mk/D\\) linhas; com pedaços de 7.168 tokens (~199 linhas por expert) o GLM <b>chega</b> ao ridge no prefill: I ≈ 460–500 contando as ativações BF16 (a aproximação só de pesos daria ~800).',
    'Tempo de passo de decode ≈ bytes lidos ÷ banda efetiva + custos fixos; tok/s ≈ tokens por passo (aceitos + 1) ÷ tempo do passo. Qwen: ≈9,2 GiB por GPU → 40–42 ms de streaming → teto de ~75–80 tok/s com τ ≈ 3,2; o medido (52,1 tok/s, passo real de ~60 ms) mostra que ~1/3 do passo é custo fixo.',
    'O KV cache guarda K e V de todos os tokens passados porque, pela máscara causal, eles nunca mudam: sem ele, cada passo refaria o forward inteiro dos tokens anteriores (n(n+1)/2 token-forwards em vez de n). Em troca, cresce com o contexto e é relido a cada passo.',
    'Bytes por token = \\(L_{kv} \\times n_{kv} \\times (d_k+d_v) \\times b\\) + estado fixo por sequência. Qwen 26.624 B (BF16) / 13.312 B (FP8); GLM 7.579 B por rank + ~76 MiB de estado KDA; DeepSeek ≈3.850 B + ≈3,44 MB. GQA se divide por TP; MLA/MQA se replica.',
    'O pool de KV é único: \\(\\sum c_i \\le\\) Pool ÷ bytes por token. <code>--max-model-len</code> e <code>--max-num-seqs</code> são tetos, não reservas; o dtype do KV mudou a capacidade do Qwen em +71%.',
    'Batching divide a leitura dos pesos: no Qwen, 1 → 8 streams multiplicou o agregado por 3,8× enquanto cada stream caiu para 49%. Em MoE a carona é menor, porque mais tokens acendem mais experts distintos.',
  ],
  quiz: [
    { q: 'Por que o decode com um usuário é limitado pela banda de memória no GB10?', o: ['Porque a atenção é quadrática no contexto', 'Porque cada peso lido participa de uma única multiplicação (GEMV): I ≈ 2/b_w FLOP/B, muito abaixo do ridge de ~458', 'Porque o decode usa FP32', 'Porque o KV cache fica na CPU'], a: 1, e: 'Com M = 1, a projeção faz 2KN FLOPs sobre b<sub>w</sub>·KN bytes de pesos: I ≈ 1 FLOP/B em BF16 e ≈ 4 com pesos de 4 bits. O ridge do GB10 (~125 TFLOPS ÷ 273 GB/s) fica em ~458. Longe dele, a GPU passa o passo esperando bytes.' },
    { q: 'O Qwen guarda KV por token em 13 camadas, com 2 cabeças KV de dimensão 256. Quantos bytes por token isso dá com KV em FP8, antes de dividir por TP?', o: ['6.656', '13.312', '26.624', '53.248'], a: 1, e: '13 × 2 cabeças × (256 + 256) × 1 byte = 13.312 B. Em BF16 seriam 26.624. Com TP=2, cada GPU fica com uma cabeça, então 6.656 B por GPU em FP8. O pool medido dá 9.418 B por GPU, porque indexador, estado e arredondamento de página também entram.' },
    { q: 'Num pedaço de prefill de 1.024 tokens do GLM (288 experts, top-8, pesos de 4 bits), por que o MoE ainda é limitado pela memória?', o: ['Porque o prefill nunca usa tensor cores', 'Porque cada expert recebe só ~28 linhas: I ≈ 2 × 28 / 0,5 ≈ 112 FLOP/B, abaixo do ridge', 'Porque o KV cache do prefill é lido 1.024 vezes', 'Porque o indexador trava a GPU'], a: 1, e: 'As 1.024 × 8 = 8.192 escolhas se espalham por 288 experts, ~28 linhas cada. Com meio byte por peso, I ≈ 112 < 458 (≈100 contando as ativações). Todos os experts são lidos em todo pedaço (~75,6 GB por GPU, estimativa). Com 7.168 tokens por pedaço, ~199 linhas por expert: a aproximação só de pesos daria ~800, mas com r ≈ 200 as ativações BF16 pesam quase tanto quanto os pesos de 4 bits, e a conta completa dá ~460–500 FLOP/B. O MoE chega ao ridge, sem atravessá-lo com folga.' },
    { q: 'No Qwen (lane day-0, prosa), o agregado subiu de 54,4 para 207,0 tok/s indo de 1 para 8 streams, e cada stream caiu para 26,7. O que explica o agregado crescer?', o: ['O KV cache fica menor com mais streams', 'Os pesos lidos a cada passo são compartilhados por todos os streams; o custo marginal de cada stream (KV, estado, experts extras) é bem menor que a leitura dos pesos', 'A especulação aceita mais tokens com mais streams', 'O prefill acelera'], a: 1, e: 'Com o passo dominado pela leitura dos pesos (T<sub>0</sub>), cada stream extra só acrescenta o seu custo marginal (T<sub>1</sub>). O agregado B·τ/(T<sub>0</sub> + B·T<sub>1</sub>) sobe (3,8×) enquanto o passo fica mais longo para todos (por stream −51%).' },
    { q: 'Com TP=2, o KV por GPU do GLM (MLA) e do DeepSeek (MQA comprimida) cai pela metade, como no Qwen?', o: ['Sim, TP sempre divide o KV', 'Não: há uma única “cabeça” de KV (o latente), que não se fatia; cada rank guarda uma cópia inteira', 'Só no DeepSeek', 'Só se o KV for FP8'], a: 1, e: 'GQA divide as cabeças KV entre as GPUs (o Qwen tem 2, uma por GPU). MLA e MQA guardam um vetor latente por token e camada, replicado em todo rank: o GLM guarda 7.579 B por token em cada rank. O ganho de TP para o KV vem indiretamente: menos pesos por GPU deixam mais memória para o pool.' },
    { q: 'Na varredura 0731 do DeepSeek, um prompt de 131.072 tokens teve TTFT de 78,75 s e decode de 65,2 tok/s. Qual é o prefill tok/s e quanto do relógio de uma resposta de 512 tokens é prefill?', o: ['~65 tok/s; ~10%', '~1.665 tok/s; ~90%', '~16.600 tok/s; ~50%', '~1.665 tok/s; ~10%'], a: 1, e: 'Prefill tok/s = P/TTFT = 131.072/78,75 ≈ 1.665. O decode de 512 tokens leva ~511/65,2 ≈ 7,8 s, então o relógio total é ~86,6 s, dos quais ~91% são prefill. É por isso que o TTFT domina a experiência com prompt longo e resposta curta.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);

    /* =============================== helpers =============================== */
    const FC = {};
    const setFont = (g, sz, f = 'sans', wt = 400) => { const k = sz + '|' + f + '|' + wt; g.font = FC[k] || (FC[k] = V.font(sz, f, wt)); };
    const tw = (g, s, sz = 12, f = 'sans', wt = 400) => { setFont(g, sz, f, wt); return g.measureText(String(s)).width; };
    const nf = (v, d = 0) => Number(v).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const onInk = () => (V.C().dark ? '#00210F' : '#ffffff');
    const clamp = V.clamp, lerp = V.lerp;
    function fitH(fig, hd, hm, bp = 620) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = fig.clientWidth || 800; const want = (w < bp ? hm : hd) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      window.addEventListener('resize', apply);
      LA.onCleanup(() => window.removeEventListener('resize', apply));
    }
    function wrapLines(g, s, maxW, sz, f = 'sans', wt = 400) {
      const words = String(s).split(' ');
      const out = []; let line = '';
      words.forEach((wd) => { const t2 = line ? line + ' ' + wd : wd; if (tw(g, t2, sz, f, wt) > maxW && line) { out.push(line); line = wd; } else line = t2; });
      if (line) out.push(line);
      return out;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const sz = o.size || 12.5, lh = o.lh || Math.round(sz * 1.42);
      const L = wrapLines(g, s, maxW, sz, o.font || 'sans', o.weight || 400);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({ size: sz }, o)));
      return y + L.length * lh;
    }
    function fit(g, s, maxW, sz, f = 'sans', wt = 400) {
      s = String(s);
      if (tw(g, s, sz, f, wt) <= maxW) return s;
      while (s.length > 1 && tw(g, s + '…', sz, f, wt) > maxW) s = s.slice(0, -1);
      return s + '…';
    }
    /** personagem-token: bolinha com olhos */
    function guy(g, x, y, r, col, o = {}) {
      const C = V.C();
      const a = o.alpha == null ? 1 : o.alpha;
      if (a <= 0.02) return;
      g.save(); g.globalAlpha = a;
      if (o.glow) V.glow(g, x, y, r * 2.5, col, 0.55);
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = V.alpha(col, C.dark ? 0.34 : 0.3); g.fill();
      g.lineWidth = 1.6; g.strokeStyle = col; g.stroke();
      const ex = r * 0.35, ey = -r * 0.14, er = Math.max(1.1, r * 0.13);
      const bl = o.blink ? 0.2 : 1;
      g.fillStyle = C.ink;
      g.beginPath(); g.ellipse(x - ex, y + ey, er, er * bl, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.ellipse(x + ex, y + ey, er, er * bl, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.arc(x, y + r * 0.1, r * 0.4, 0.18 * Math.PI, 0.82 * Math.PI); g.lineWidth = 1.2; g.strokeStyle = C.ink; g.stroke();
      g.restore();
    }
    const blinkAt = (t, seed) => ((t * 0.55 + seed * 0.37) % 3.2) < 0.13;
    /** célula com valor */
    function cell(g, x, y, c, v, o = {}) {
      const C = V.C();
      const sc = o.scale || 2;
      const tt = V.clamp(v / sc, -1, 1);
      const a = o.alpha == null ? 1 : o.alpha;
      g.save(); g.globalAlpha = a;
      const cw = (o.wd || c) - 2;
      V.rr(g, x, y, cw, c - 2, Math.min(5, c / 5));
      g.fillStyle = o.fill || V.cmap(tt, 'div'); g.fill();
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.lw || 1.5; g.stroke(); }
      if (o.text !== false) {
        const light = !o.fill && Math.abs(tt) > 0.55;
        const s = o.fmt ? o.fmt(v) : String(v).replace('.', ',');
        V.text(g, s, x + cw / 2, y + (c - 2) / 2 + 0.5, { size: o.size || 11, font: 'mono', align: 'center', base: 'middle', color: o.textColor || (light ? onInk() : C.ink) });
      }
      g.restore();
    }

    /* ===================== Figura 1: a fábrica de duas fases ===================== */
    {
      const f = $('#fig-duas-fases');
      fitH(f, 400, 450);
      const PROMPT = ['O', 'que', 'é', 'KV', 'cache', '?'];
      const RESP = ['Um', 'arquivo', 'de', 'K', 'e', 'V'];
      const NL = 5, P = PROMPT.length, G = RESP.length, DT = 1.55, PT = 2.4;
      const S = { t3: performance.now(), last: -1 };
      const st = UI.stepper(f, [
        { t: 'O pedido chega', d: 'Seis tokens de prompt esperam na entrada. As estantes de KV, embaixo de cada camada, estão vazias, e os pesos de cada camada estão na memória.' },
        { t: 'Prefill: todos de uma vez', d: 'Os seis tokens atravessam as camadas <b>juntos</b>: cada projeção recebe uma matriz de 6 linhas (\\(M = 6\\)). Ao passar, cada camada grava na estante os K e V dos seis. Os pesos de cada camada são lidos uma vez e servem aos seis.' },
        { t: 'O primeiro token', d: 'O <code>lm_head</code> transforma o estado da última posição em logits, e o amostrador escolhe o primeiro token da resposta. Tudo até aqui é o <b>TTFT</b>.' },
        { t: 'Decode: um token por passo', d: 'O token recém-gerado volta à entrada e atravessa as camadas <b>sozinho</b> (\\(M = 1\\)). Em cada camada ele lê todas as linhas da estante (a atenção, em âmbar) e acrescenta a sua.' },
        { t: 'Cada passo relê os pesos', d: 'Repare nos pulsos que sobem da memória: a cada token de resposta, os pesos de todas as camadas são lidos de novo, agora para servir a <b>um</b> token só. É esse custo que torna o decode limitado por banda.' },
        { t: 'Duas contas diferentes', d: 'Prefill: 6 tokens, 1 passo, cada peso lido serve 6 linhas. Decode: 6 tokens, 6 passos, cada peso lido serve 1 linha. O resto do capítulo transforma essa diferença em números.' },
      ], { autoplay: 6500, onStep: (i) => { if (i === 3 || (i === 4 && S.last !== 3)) S.t3 = performance.now(); S.last = i; } });

      function chips(g, label, words, n, x0, y, xMax, col, o = {}) {
        const C = V.C();
        V.text(g, label, x0, y + 14, { size: 11, font: 'mono', color: C.muted });
        const xs = x0 + Math.max(tw(g, 'resposta', 11, 'mono'), tw(g, label, 11, 'mono')) + 10;
        let x = xs;
        const cnt = o.showAll ? words.length : n;
        for (let k = 0; k < cnt; k++) {
          const s = words[k];
          const cw = tw(g, s, 12, 'mono', 500) + 14;
          if (x + cw > xMax && x > xs) { x = xs; y += 24; }
          const on = k < n;
          const hi = o.hi === k || (o.hiAll && on);
          if (hi) V.glow(g, x + cw / 2, y + 10, cw * 0.8, col, 0.4);
          V.box(g, x, y, cw, 20, { r: 7, fill: on ? V.alpha(col, 0.2) : 'rgba(0,0,0,0)', stroke: on ? V.alpha(col, hi ? 1 : 0.6) : V.alpha(C.muted, 0.3), lw: 1.2 });
          V.text(g, s, x + cw / 2, y + 10.5, { size: 12, font: 'mono', weight: 500, color: on ? C.ink : C.muted, align: 'center', base: 'middle' });
          if (o.ttftAt === k) { V.box(g, x, y, cw, 20, { r: 7, stroke: C.d, lw: 1.8 }); V.dot(g, x + 1, y + 1, 3.2, C.d); }
          x += cw + 6;
        }
        if (o.cursor && Math.floor(performance.now() / 450) % 2 === 0) {
          if (x + 8 > xMax) { x = xs; y += 24; }
          V.box(g, x, y + 3, 7, 14, { r: 2, fill: V.alpha(col, 0.7) });
        }
        return y + 28;
      }

      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const i = st.i, e = st.elapsed();
        const pad = 14;
        // ---------- estado ----------
        let pre = i === 0 ? 0 : i === 1 ? clamp(e / PT, 0, 1) : 1;
        let gen = 0, dj = -1, du = 0;
        if (i === 2) gen = e > 0.55 ? 1 : 0;
        if (i === 3 || i === 4) {
          const ed = (performance.now() - S.t3) / 1000;
          const n = Math.floor(ed / DT);
          if (n >= G - 1) { dj = -1; gen = G; } else { dj = n; du = (ed % DT) / DT; gen = 1 + n + (du > 0.86 ? 1 : 0); }
        }
        if (i === 5) gen = G;
        // ---------- cabeçalho ----------
        let y = 12;
        y = chips(g, 'prompt', PROMPT, P, pad, y, w - pad, C.b, { hiAll: i === 1 && pre < 1 });
        y = chips(g, 'resposta', RESP, gen, pad, y, w - pad, C.a, { hi: dj >= 0 ? dj : (i === 2 && gen ? 0 : -1), cursor: gen < G, ttftAt: i >= 2 && gen ? 0 : -1 });
        // ---------- geometria ----------
        const top = y + 6;
        const cellS = narrow ? 7 : 10, cg = 2, shelfRows = 4, shelfH = shelfRows * (cellS + cg);
        const bankH = 22, bottomH = narrow ? 84 : 42;
        const avail = h - top - bottomH;
        const barH = clamp(avail - bankH - shelfH - 40, 60, narrow ? 190 : 150);
        const block = 14 + barH + 8 + shelfH + 10 + bankH;
        const off = Math.max(0, (avail - block) / 2);
        const barY = top + 12 + off, shelfY = barY + barH + 8, bankY = shelfY + shelfH + 10, midY = barY + barH / 2;
        const lmW = narrow ? 62 : 72;
        const lmX = w - pad - lmW - (narrow ? 2 : 48);
        const bx0 = narrow ? 58 : 96, bx1 = lmX - (narrow ? 36 : 64);
        const sp = (bx1 - bx0) / (NL - 1);
        const lx = (l) => bx0 + l * sp;
        const bw = Math.min(28, sp * 0.42);
        const xIn = pad + 14, xOut = (bx1 + lmX) / 2 + 2;
        const r = narrow ? 7 : 8.5;
        // posição dos viajantes
        const gx = lerp(xIn, xOut, V.ease.inOut(pre));
        const dx = dj >= 0 ? lerp(xIn, xOut, V.ease.inOut(clamp(du / 0.8, 0, 1))) : xOut;
        const travX = i === 1 ? gx : (dj >= 0 ? dx : -999);
        const fill = (l) => {
          if (i <= 1) return gx >= lx(l) - 1 && pre > 0 ? P : 0;
          if (i === 2) return P;
          if (i === 5) return P + G - 1;
          if (dj < 0) return P + G - 1;
          return P + dj + (dx >= lx(l) ? 1 : 0);
        };
        // ---------- banco de memória ----------
        const bkx = bx0 - bw / 2 - 12, bkw = bx1 - bx0 + bw + 24;
        V.box(g, bkx, bankY, bkw, bankH, { r: 7, fill: V.alpha(C.d, 0.12), stroke: V.alpha(C.d, 0.55), lw: 1.2 });
        V.text(g, narrow ? 'pesos (memória)' : 'pesos de todas as camadas, na memória', bkx + bkw / 2, bankY + bankH / 2 + 0.5, { size: 11, font: 'mono', color: C.d, align: 'center', base: 'middle' });
        // pulsos de leitura dos pesos
        for (let l = 0; l < NL; l++) {
          const d = travX - lx(l);
          if (d >= -4 && d < 44) {
            const a = 1 - clamp((d + 4) / 48, 0, 1);
            const x = lx(l);
            const strong = i === 4 ? 1.6 : 1;
            g.save(); g.globalAlpha = a;
            V.line(g, x, bankY, x, barY + barH, { color: C.d, lw: 2.2 * strong, dash: [3, 4], dashOffset: t * 40 });
            for (let k = 0; k < 3; k++) { const u = ((t * 1.6 + k / 3) % 1); V.dot(g, x, lerp(bankY, barY + barH, u), 2.4 * strong, C.d); }
            g.restore();
          }
        }
        // ---------- camadas ----------
        for (let l = 0; l < NL; l++) {
          const x = lx(l);
          const near = Math.abs(travX - x) < 20;
          if (near) V.glow(g, x, midY, bw * 1.8, i === 1 ? C.b : C.a, 0.5);
          V.box(g, x - bw / 2, barY, bw, barH, { r: 8, fill: V.alpha(C.a, near ? 0.22 : 0.09), stroke: V.alpha(C.a, near ? 0.95 : 0.45), lw: near ? 2 : 1.2 });
          V.text(g, narrow ? 'L' + (l + 1) : 'camada ' + (l + 1), x, barY - 4, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          // estante
          const nf_ = fill(l);
          const sw = 3 * (cellS + cg) - cg;
          const sx = x - sw / 2;
          const reading = (i === 3 || i === 4) && dj >= 0 && travX - x >= -2 && travX - x < 30;
          for (let n = 0; n < 12; n++) {
            const cx = sx + (n % 3) * (cellS + cg), cy = shelfY + Math.floor(n / 3) * (cellS + cg);
            const on = n < nf_;
            const col = n < P ? C.b : C.a;
            V.rr(g, cx, cy, cellS, cellS, 2);
            g.fillStyle = on ? V.alpha(col, 0.8) : V.alpha(C.muted, 0.14); g.fill();
            if (on && reading) { g.strokeStyle = C.d; g.lineWidth = 1.4; g.stroke(); }
          }
          if (reading) V.glow(g, x, shelfY + shelfH / 2, 22, C.d, 0.35);
        }
        V.text(g, 'KV ▸', bx0 - Math.max(bw / 2, 3 * (cellS + cg) / 2) - 8, shelfY + shelfH / 2 + 4, { size: 11, font: 'mono', weight: 600, color: C.muted, align: 'right' });
        // ---------- lm_head ----------
        const lmHot = (i === 2 && e < 1.2) || (dj >= 0 && du > 0.78 && du < 0.95);
        if (lmHot) V.glow(g, lmX + lmW / 2, midY, lmW, C.c, 0.5);
        V.box(g, lmX, midY - 24, lmW, 48, { r: 10, fill: V.alpha(C.c, lmHot ? 0.28 : 0.12), stroke: V.alpha(C.c, lmHot ? 1 : 0.55), lw: 1.4 });
        V.text(g, 'lm_head', lmX + lmW / 2, midY - 4, { size: narrow ? 11 : 12, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
        V.text(g, 'amostra', lmX + lmW / 2, midY + 12, { size: 11, color: C.muted, align: 'center', base: 'middle' });
        V.arrow(g, bx1 + bw / 2 + 4, midY, lmX - 4, midY, { color: V.alpha(C.muted, 0.5), lw: 1.3, head: 6 });
        // saída (desktop)
        if (!narrow && gen > 0) {
          const ox = lmX + lmW + 26;
          const pop = dj >= 0 && du > 0.86 ? V.ease.back(clamp((du - 0.86) / 0.14, 0, 1)) : (i === 2 ? V.ease.back(clamp((e - 0.55) / 0.4, 0, 1)) : 1);
          guy(g, ox, midY, r * pop, C.a, { blink: blinkAt(t, 9) });
          V.text(g, RESP[gen - 1], ox, midY + r + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        }
        // ---------- viajantes ----------
        if (i <= 2) {
          const a = i === 2 ? 1 - clamp(e / 0.6, 0, 1) : 1;
          for (let k = 0; k < P; k++) {
            const yy = midY + (k - (P - 1) / 2) * (2 * r + 3);
            const bob = i === 0 ? Math.sin(t * 3 + k) * 1.5 : 0;
            guy(g, gx, yy + bob, r, C.b, { alpha: a, blink: blinkAt(t, k), glow: i === 1 && pre < 1 && k === 0 });
          }
          if (i <= 1) V.text(g, 'M = ' + P, gx, midY + (P / 2) * (2 * r + 3) + 14, { size: 11, font: 'mono', color: C.b, align: 'center', alpha: i === 1 ? 1 : 0.6 });
        }
        if (dj >= 0) {
          const a = du > 0.8 ? 1 - clamp((du - 0.8) / 0.08, 0, 1) : 1;
          guy(g, dx, midY, r, C.a, { alpha: a, blink: blinkAt(t, 3), glow: true });
          V.text(g, 'M = 1', dx, midY + r + 15, { size: 11, font: 'mono', color: C.a, align: 'center', alpha: a });
        }
        // ---------- rodapé ----------
        const fwd = (pre > 0 ? 1 : 0) + (i === 5 ? G - 1 : dj >= 0 ? dj + (du > 0.05 ? 1 : 0) : (i === 3 || i === 4) ? G - 1 : 0);
        let msg;
        if (i === 0) msg = 'Aguardando: ' + P + ' tokens de prompt na fila.';
        else if (i === 1) msg = 'PREFILL · 1 forward · M = ' + P + ' linhas em cada GEMM';
        else if (i === 2) msg = 'Primeiro token amostrado: aqui termina o TTFT.';
        else if (i === 5) msg = 'Prefill: 6 tokens em 1 passo · Decode: 6 tokens em 6 passos';
        else msg = dj >= 0 ? 'DECODE · passo ' + (dj + 1) + ' · M = 1 linha · a estante tem ' + (P + dj + 1) + ' linhas' : 'DECODE concluído: a estante tem ' + (P + G - 1) + ' linhas';
        const yb = h - bottomH + 16;
        const col = i === 1 ? C.b : i >= 3 && i <= 4 ? C.a : i === 2 ? C.d : C.ink;
        textBlock(g, msg, pad, yb, w - 2 * pad, { size: 12.5, font: 'mono', weight: 600, color: col, lh: 16 });
        if (narrow) {
          V.text(g, 'forwards: ' + fwd + ' · leituras dos pesos: ' + fwd, pad, h - 24, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'tokens gerados: ' + gen, pad, h - 9, { size: 11, font: 'mono', color: C.muted });
        } else {
          const cnt = 'forwards: ' + fwd + ' · leituras completas dos pesos: ' + fwd + ' · tokens gerados: ' + gen;
          V.text(g, fit(g, cnt, w - 2 * pad, 11, 'mono'), pad, h - 8, { size: 11, font: 'mono', color: C.muted });
        }
      });
    }

    /* ===================== Figura 2: GEMV × GEMM ===================== */
    {
      const f = $('#fig-gemv-gemm');
      fitH(f, 420, 560);
      const K = 4, N = 6, MMAX = 8;
      const R = V.rng(21);
      const W = Array.from({ length: K }, () => Array.from({ length: N }, () => R.int(4) - 1));
      const X = Array.from({ length: MMAX }, () => Array.from({ length: K }, () => R.int(4) - 1));
      const Y = X.map((row) => W[0].map((_, j) => row.reduce((s, a, k) => s + a * W[k][j], 0)));
      const S = { M: 1, b: 2, reset: true, t0: 0 };
      const ctl = UI.controls(f);
      const fmtI = () => nf(2 * S.M / S.b, 0) + ' FLOP/B';
      UI.slider(ctl, { label: 'Linhas M', min: 1, max: MMAX, step: 1, value: 1, fmt: (v) => v + (v === 1 ? ' (GEMV)' : ' (GEMM)'), onInput: (v) => { S.M = v; S.reset = true; rU.set(String(v)); rI.set(fmtI()); } });
      UI.seg(ctl, { label: 'Peso', options: [{ v: 2, label: 'BF16' }, { v: 1, label: 'FP8' }, { v: 0.5, label: '4 bits' }], value: 2, onChange: (v) => { S.b = v; rI.set(fmtI()); } });
      const rU = UI.readout(ctl, 'usos por peso lido', '1');
      const rI = UI.readout(ctl, 'intensidade ≈ 2M/b', fmtI());
      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        if (S.reset) { S.t0 = t; S.reset = false; }
        const M = S.M;
        const pad = 14, lab = 44;
        const side = w >= 600; // balança ao lado (largo) ou embaixo (estreito)
        const balH = side ? 0 : 170;
        const c = Math.floor(Math.max(18, Math.min(34, (w - 2 * pad - lab - 22) / (K + N), (h - 128 - balH) / (K + MMAX))));
        const xX = pad + lab, xW = xX + K * c + 20, yW = 34, yX = yW + K * c + 18;
        // vagas para mais linhas (M < 8): contornos apagados
        for (let m = M; m < MMAX; m++) {
          V.text(g, 'tok ' + (m + 1), xX - 6, yX + m * c + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle', alpha: 0.28 });
          g.globalAlpha = 1;
          for (let k = 0; k < K; k++) { V.rr(g, xX + k * c, yX + m * c, c - 2, c - 2, 4); g.fillStyle = V.alpha(C.muted, 0.05); g.fill(); g.strokeStyle = V.alpha(C.muted, 0.12); g.lineWidth = 1; g.stroke(); }
          for (let j = 0; j < N; j++) { V.rr(g, xW + j * c, yX + m * c, c - 2, c - 2, 4); g.fillStyle = V.alpha(C.muted, 0.05); g.fill(); g.strokeStyle = V.alpha(C.muted, 0.12); g.lineWidth = 1; g.stroke(); }
        }
        const Tc = 0.55 + 0.11 * M, total = N * Tc + 1.2;
        const e = (t - S.t0) % total;
        const jj = Math.floor(e / Tc), u = (e % Tc) / Tc;
        const done = jj >= N;
        const jCur = done ? N : jj;
        const mCur = done ? M : (u < 0.35 ? -1 : Math.min(M - 1, Math.floor(((u - 0.35) / 0.65) * M)));
        // rótulos
        V.text(g, 'W  [K=4 × N=6]', xW, yW - 10, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'X  [M×4]', xX, yX - 6, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'Y = XW  [M×6]', xW, yX - 6, { size: 11, font: 'mono', color: C.muted });
        // banco de memória
        const dmX = pad, dmW = lab + K * c - 8, dmH = K * c - 2;
        V.box(g, dmX, yW, dmW, dmH, { r: 10, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.5), lw: 1.2 });
        for (let k = 0; k < 3; k++) V.box(g, dmX + 10, yW + 12 + k * 12, dmW - 20, 7, { r: 3, fill: V.alpha(C.d, 0.25 + 0.1 * k) });
        V.text(g, 'memória', dmX + dmW / 2, yW + dmH - 24, { size: 11, font: 'mono', color: C.d, align: 'center' });
        V.text(g, '(pesos W)', dmX + dmW / 2, yW + dmH - 10, { size: 11, color: C.muted, align: 'center' });
        // W
        for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
          const loaded = j < jCur || (j === jCur && u >= 0.35);
          const isCur = j === jCur && !done;
          cell(g, xW + j * c, yW + k * c, c, W[k][j], { alpha: loaded || isCur ? 1 : 0.35, stroke: isCur && u >= 0.35 ? C.c : null });
        }
        // pacote viajando da memória até a coluna
        if (!done && u < 0.35) {
          const p = V.ease.inOut(u / 0.35);
          const x0 = dmX + dmW - 6, x1 = xW + jCur * c + (c - 2) / 2;
          const px = lerp(x0, x1, p);
          V.glow(g, px, yW + K * c / 2, 20, C.d, 0.8);
          for (let k = 0; k < K; k++) V.box(g, px - 6, yW + k * c + c / 2 - 7, 12, 12, { r: 3, fill: V.alpha(C.d, 0.85) });
        }
        // X
        for (let m = 0; m < M; m++) {
          V.text(g, 'tok ' + (m + 1), xX - 6, yX + m * c + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
          for (let k = 0; k < K; k++) cell(g, xX + k * c, yX + m * c, c, X[m][k], { stroke: !done && m === mCur ? C.c : null });
        }
        // Y
        for (let m = 0; m < M; m++) for (let j = 0; j < N; j++) {
          const filled = done || j < jCur || (j === jCur && m <= mCur);
          const flash = !done && j === jCur && m === mCur;
          if (filled) cell(g, xW + j * c, yW + K * c + 18 + m * c, c, Y[m][j], { scale: 8, stroke: flash ? C.c : null });
          else { V.rr(g, xW + j * c, yW + K * c + 18 + m * c, c - 2, c - 2, 4); g.strokeStyle = V.alpha(C.muted, 0.3); g.lineWidth = 1; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]); }
          if (flash) V.glow(g, xW + j * c + c / 2, yW + K * c + 18 + m * c + c / 2, c, C.c, 0.5);
        }
        // contadores
        const cols = Math.min(N, jCur + (u >= 0.35 && !done ? 1 : 0));
        const mults = done ? N * K * M : (jCur * K * M + (mCur >= 0 ? (mCur + 1) * K : 0));
        const narrowT = w < 560;
        const yT = h - (narrowT ? 64 : 34);
        const line1 = 'colunas lidas: ' + cols + '/' + N + ' · pesos lidos: ' + cols * K + ' · multiplicações: ' + mults;
        const line2 = 'cada peso lido serve M = ' + M + (M === 1 ? ' token' : ' tokens') + ' → I ≈ 2M/b = ' + nf(2 * M / S.b, 0) + ' FLOP/byte';
        const yy2 = textBlock(g, line1, pad, yT, w - 2 * pad, { size: 11.5, font: 'mono', color: C.ink, lh: 16 });
        textBlock(g, line2, pad, yy2 + 4, w - 2 * pad, { size: 11.5, font: 'mono', weight: 600, color: M === 1 ? C.d : C.a, lh: 16 });
        // balança: tempo de leitura × tempo de cálculo (GB10)
        const I = 2 * M / S.b, ridge = 458;
        const ratio = ridge / I;
        // compacta (embaixo, telas estreitas) ou grande (ao lado)
        const bx0 = xW + N * c + 36;
        const cmp = !side || w - pad - bx0 < 230;
        const cxB = cmp ? w / 2 : (bx0 + w - pad) / 2;
        const topB = cmp ? yX + MMAX * c + 14 : yW - 10;
        const availW = cmp ? w - 2 * pad : w - pad - bx0;
        const Lb = cmp ? Math.min(112, w / 2 - 62) : Math.min(130, availW / 2 - 30);
        const pyB = cmp ? topB + 40 : yW + 40;
        const amp = cmp ? 0.2 : 0.36, strL = cmp ? 26 : 42, colH = cmp ? 56 : 120;
        const th = amp * clamp(Math.log10(ratio) / Math.log10(ridge), -1, 1) + Math.sin(t * 2) * 0.01;
        const lx1 = cxB - Lb * Math.cos(th), ly1 = pyB + Lb * Math.sin(th);
        const rx1 = cxB + Lb * Math.cos(th), ry1 = pyB - Lb * Math.sin(th);
        V.text(g, fit(g, cmp ? 'no GB10: ler W × calcular' : 'no GB10, para esta matriz:', availW, 11, 'mono'), cxB, topB + (cmp ? 10 : 0), { size: 11, font: 'mono', color: C.muted, align: 'center' });
        // coluna e base
        V.line(g, cxB, pyB, cxB, pyB + colH, { color: C.muted, lw: 3 });
        V.box(g, cxB - 34, pyB + colH - 2, 68, 8, { r: 4, fill: V.alpha(C.muted, 0.5) });
        V.line(g, lx1, ly1, rx1, ry1, { color: C.ink, lw: 3 });
        V.dot(g, cxB, pyB, 5, C.ink);
        const pan = (x, y, col, lbl, sub, n) => {
          V.line(g, x, y, x - 22, y + strL, { color: V.alpha(C.ink, 0.6), lw: 1 });
          V.line(g, x, y, x + 22, y + strL, { color: V.alpha(C.ink, 0.6), lw: 1 });
          V.box(g, x - 30, y + strL, 60, 8, { r: 4, fill: V.alpha(col, 0.8) });
          for (let k = 0; k < n; k++) V.box(g, x - 22 + (k % 4) * 11, y + strL - 10 - Math.floor(k / 4) * 10, 9, 8, { r: 2, fill: V.alpha(col, 0.9) });
          V.text(g, lbl, x, y + strL + 22, { size: 11.5, font: 'mono', weight: 600, color: col, align: 'center' });
          if (!cmp) V.text(g, sub, x, y + strL + 37, { size: 11, color: C.muted, align: 'center' });
        };
        pan(lx1, ly1, C.d, 'ler W', 'bytes ÷ banda', 8);
        pan(rx1, ry1, C.a, 'calcular', 'FLOPs ÷ pico', Math.max(1, Math.min(8, Math.round(8 * I / ridge * 8))));
        const msg = ratio >= 1.05 ? 'ler custa ' + nf(ratio, ratio < 10 ? 1 : 0) + '× mais que calcular' : 'calcular domina';
        const my = cmp ? pyB + 90 : pyB + 150;
        V.text(g, fit(g, msg, availW, 12.5, 'mono', 600), cxB, my, { size: 12.5, font: 'mono', weight: 600, color: ratio >= 1.05 ? C.d : C.a, align: 'center' });
        V.text(g, fit(g, 'equilíbrio só em I ≈ 458 (M ≈ ' + nf(458 * S.b / 2) + ')', availW, 11), cxB, my + 17, { size: 11, color: C.muted, align: 'center' });
      });
    }

    /* ===================== Figura 3: roofline ===================== */
    {
      const f = $('#fig-roofline');
      fitH(f, 450, 520);
      const MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192];
      const S = { mi: 0, b: 2, moe: false, peak: 125, cur: 0 };
      const BW = 0.273; // TB/s -> TFLOPS = I * BW
      const E = 288, KE = 8, BA = 2; // ativações em BF16
      const DENSE = [[4096, 4096]];
      const EXPERT = [[4096, 2048], [1024, 4096]]; // expert do GLM em TP=2: gate_up e down locais
      const rowsPerExpert = (M) => { const D = E * (1 - Math.pow(1 - KE / E, M)); return (M * KE) / D; };
      // intensidade completa: FLOPs ÷ (bytes de peso + bytes de ativação)
      const intenOf = (M, b, moe) => { const r = moe ? rowsPerExpert(M) : M; let F = 0, Q = 0; (moe ? EXPERT : DENSE).forEach(([K, N]) => { F += 2 * r * K * N; Q += b * K * N + BA * r * (K + N); }); return F / Q; };
      const approxOf = (M, b, moe) => 2 * (moe ? rowsPerExpert(M) : M) / b;
      const inten = () => intenOf(MS[S.mi], S.b, S.moe);
      S.cur = Math.log10(inten());
      const ctl = UI.controls(f);
      const upd = () => { const I = inten(); const perf = Math.min(S.peak, I * BW); rI.set(nf(I, I < 10 ? 1 : 0) + ' FLOP/B'); rP.set(nf(perf, perf < 10 ? 2 : 1) + ' TFLOPS'); rR.set(I < S.peak / BW ? 'memória' : 'cálculo'); };
      UI.slider(ctl, { label: 'Linhas M', min: 0, max: MS.length - 1, step: 1, value: 0, fmt: (v) => nf(MS[v]), onInput: (v) => { S.mi = v; upd(); } });
      UI.seg(ctl, { label: 'Peso', options: [{ v: 2, label: 'BF16' }, { v: 1, label: 'FP8' }, { v: 0.5, label: '4 bits' }], value: 2, onChange: (v) => { S.b = v; upd(); } });
      UI.toggle(ctl, { label: 'MoE: 288 experts, top-8 (GLM)', value: false, onChange: (v) => { S.moe = v; upd(); } });
      UI.seg(ctl, { label: 'Pico', options: [{ v: 125, label: 'BF16 ~125 TFLOPS' }, { v: 500, label: 'FP4×FP8 ~4×' }], value: 125, onChange: (v) => { S.peak = v; upd(); } });
      const rI = UI.readout(ctl, 'intensidade (pesos + ativações)', '—');
      const rP = UI.readout(ctl, 'teto atingível', '—');
      const rR = UI.readout(ctl, 'limitado por', '—');
      upd();
      // lm_head do GLM por rank (154.880 / 2 × 4.096, BF16) com as 8 posições de um passo DFlash2
      const lmK = 4096, lmN = 77440;
      const MARK = [
        { I: intenOf(1, 2, false), l: 'decode BF16, M = 1', s: 'decode BF16' },
        { I: intenOf(1, 0.5, false), l: 'decode 4 bits, M = 1', s: 'decode 4 b' },
        { I: (2 * 8 * lmK * lmN) / (2 * lmK * lmN + BA * 8 * (lmK + lmN)), l: 'GLM lm_head · 8 posições', s: 'lm_head ×8' },
        { I: intenOf(1024, 0.5, true), l: 'GLM MoE · pedaço 1.024', s: 'MoE 1.024', nLeft: true },
        { I: intenOf(7168, 0.5, true), l: 'GLM MoE · pedaço 7.168', s: 'MoE 7.168' },
      ];
      V.fig(f, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 560;
        const boxH = 56;
        const L = narrow ? 44 : 60, Rm = 14, T = boxH + 24, B = 44;
        const X0 = Math.log10(0.25), X1 = Math.log10(32768), Y0 = Math.log10(0.05), Y1 = Math.log10(1500);
        const px = (I) => L + (Math.log10(I) - X0) / (X1 - X0) * (w - L - Rm);
        const py = (P) => T + (1 - (Math.log10(P) - Y0) / (Y1 - Y0)) * (h - T - B);
        const roof = (I) => Math.min(S.peak, I * BW);
        const ridge = S.peak / BW;
        // grade
        [1, 10, 100, 1000, 10000].forEach((I) => { V.line(g, px(I), T, px(I), h - B, { color: C.grid, lw: 1 }); V.text(g, nf(I), px(I), h - B + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        [0.1, 1, 10, 100, 1000].forEach((P) => { V.line(g, L, py(P), w - Rm, py(P), { color: C.grid, lw: 1 }); V.text(g, nf(P, P < 1 ? 1 : 0), L - 6, py(P) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        V.text(g, 'intensidade aritmética (FLOP/byte, escala log)', L + (w - L - Rm) / 2, h - 8, { size: 11, color: C.muted, align: 'center' });
        g.save(); g.translate(12, T + (h - T - B) / 2); g.rotate(-Math.PI / 2); V.text(g, 'TFLOPS (log)', 0, 0, { size: 11, color: C.muted, align: 'center', base: 'middle' }); g.restore();
        // regiões
        g.beginPath(); g.moveTo(px(0.25), h - B); g.lineTo(px(0.25), py(roof(0.25))); g.lineTo(px(ridge), py(S.peak)); g.lineTo(px(ridge), h - B); g.closePath();
        g.fillStyle = V.alpha(C.d, 0.1); g.fill();
        g.beginPath(); g.moveTo(px(ridge), h - B); g.lineTo(px(ridge), py(S.peak)); g.lineTo(px(32768), py(S.peak)); g.lineTo(px(32768), h - B); g.closePath();
        g.fillStyle = V.alpha(C.b, 0.1); g.fill();
        // telhado
        g.beginPath(); g.moveTo(px(0.25), py(roof(0.25))); g.lineTo(px(ridge), py(S.peak)); g.lineTo(px(32768), py(S.peak));
        g.strokeStyle = C.ink; g.lineWidth = 2.4; g.stroke();
        V.line(g, px(ridge), py(S.peak), px(ridge), h - B, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [4, 4] });
        const rl = 'ridge ≈ ' + nf(ridge) + ' FLOP/B';
        const rlw = tw(g, rl, 11, 'mono');
        const memX = px(Math.sqrt(0.25 * ridge));
        const memW = tw(g, 'limitado pela memória', narrow ? 11 : 12, 'sans', 600);
        const rlx = Math.min(px(ridge) + 6, w - Rm - rlw);
        if (rlx < memX + memW / 2 + 8) V.text(g, rl, px(ridge) - 6, h - B - 26, { size: 11, font: 'mono', color: C.ink, align: 'right' }); // sem colidir com o rótulo da região
        else V.text(g, rl, rlx, h - B - 8, { size: 11, font: 'mono', color: C.ink });
        // rótulo do pico: acima do teto, à direita (fora do caminho do ridge e das marcas)
        V.text(g, 'pico ≈ ' + S.peak + ' TFLOPS', w - Rm - 4, py(S.peak) - 12, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'limitado pela memória', memX, h - B - 10, { size: narrow ? 11 : 12, weight: 600, color: C.d, align: 'center' });
        const cmpX = (px(ridge) + w - Rm) / 2;
        const cmpW = tw(g, narrow ? 'cálculo' : 'limitado pelo cálculo', narrow ? 11 : 12, 'sans', 600);
        V.text(g, narrow ? 'cálculo' : 'limitado pelo cálculo', Math.min(w - Rm - cmpW / 2 - 4, Math.max(cmpX, px(ridge) + 60)), py(S.peak) + 40, { size: narrow ? 11 : 12, weight: 600, color: C.b, align: 'center' });
        { // rótulo da rampa, alinhado à inclinação, num vão entre as marcas
          const Is = 28;
          const ang = Math.atan2(py(roof(ridge)) - py(roof(0.25)), px(ridge) - px(0.25));
          g.save(); g.translate(px(Is), py(roof(Is))); g.rotate(ang);
          V.text(g, narrow ? '273 GB/s × I' : 'banda: 273 GB/s × I', 0, -7, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          g.restore();
        }
        // marcas fixas (rótulos sempre abaixo e à direita da rampa)
        MARK.forEach((m) => {
          const x = px(m.I), y = py(roof(m.I));
          g.beginPath(); g.moveTo(x, y - 6); g.lineTo(x + 6, y); g.lineTo(x, y + 6); g.lineTo(x - 6, y); g.closePath();
          g.fillStyle = V.alpha(C.c, 0.85); g.fill();
          const lbl = narrow ? m.s : m.l;
          const lw_ = tw(g, lbl, 11, 'mono');
          if (narrow && m.nLeft) V.text(g, lbl, x - 9, y - 9, { size: 11, font: 'mono', color: C.c, align: 'right' }); // não cruza o ridge
          else V.text(g, lbl, Math.min(x + 9, w - Rm - lw_), y + 18, { size: 11, font: 'mono', color: C.c });
        });
        // ponto atual (animado em log)
        const target = Math.log10(clamp(inten(), 0.25, 32768));
        S.cur += (target - S.cur) * Math.min(1, dt * 6);
        const Ic = Math.pow(10, S.cur), Pc = roof(Ic);
        const cx = px(Ic), cy = py(Pc);
        const memB = Ic < ridge;
        const col = memB ? C.d : C.b;
        V.line(g, cx, cy, cx, h - B, { color: V.alpha(col, 0.6), lw: 1.2, dash: [2, 3] });
        V.line(g, L, cy, cx, cy, { color: V.alpha(col, 0.6), lw: 1.2, dash: [2, 3] });
        V.glow(g, cx, cy, 20 + 5 * Math.sin(t * 4), col, 0.8);
        V.dot(g, cx, cy, 6, col);
        // quadro de leitura, acima do gráfico
        const M = MS[S.mi];
        const Ia = approxOf(M, S.b, S.moe);
        const l1 = 'M = ' + nf(M) + (S.moe ? ' → ' + nf(rowsPerExpert(M), 1) + ' linhas/expert' : (narrow ? '' : ' · projeção 4096 × 4096'));
        const l2 = 'I ≈ ' + nf(Ic, Ic < 10 ? 1 : 0) + ' FLOP/B → ' + nf(Pc, Pc < 10 ? 2 : 1) + ' TFLOPS';
        const l3 = 'só pesos (' + (S.moe ? '2r/b_w' : '2M/b_w') + '): ' + nf(Ia, Ia < 10 ? 1 : 0) + ' FLOP/B';
        const bwid = Math.min(w - L - Rm, Math.max(tw(g, l1, 11.5, 'mono', 600), tw(g, l2, 11.5, 'mono'), tw(g, l3, 11, 'mono')) + 16);
        const bx = L, by = 8;
        V.box(g, bx, by, bwid, boxH, { r: 8, fill: V.alpha(C.panel || C.bg, 0.92), stroke: V.alpha(col, 0.8), lw: 1.2 });
        V.text(g, l1, bx + 8, by + 17, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, l2, bx + 8, by + 33, { size: 11.5, font: 'mono', color: col });
        V.text(g, l3, bx + 8, by + 49, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* ===================== Figura 4: o tanque de bytes do decode ===================== */
    {
      const f = $('#fig-bytes-passo');
      fitH(f, 420, 690);
      const GiB = 1.073741824;
      // tau = aceitos + 1 bônus; tfix = custos fixos por passo (DS: relatório; Qwen/GLM: passo medido − streaming)
      const MOD = {
        qwen: { name: 'Qwen3.8 · passo MTP=3', parts: [['lm_head ×4', 2.36 * GiB], ['experts roteados', 2.1 * GiB], ['projeções GDN', 1.95 * GiB], ['HyperConnection', 1.19 * GiB], ['atenção + indexador', 0.6 * GiB], ['MTP', 0.5 * GiB], ['router + shared + PLE', 0.4 * GiB], ['estado GDN', 0.1 * GiB]], bw: 240, tau: 3.2, tfix: 20, meas: 52.1, measL: 'MTP=3, batch-1: 52,1 tok/s' },
        ds: { name: 'DeepSeek-V4 · passo DSpark', parts: [['experts roteados', 12], ['atenção FP8', 2.4], ['lm_head (alvo + draft)', 1.06], ['draft (3 camadas)', 0.9], ['Markov w2', 0.2]], bw: 240, tau: 4.15, tfix: 12, meas: 56, measL: 'mediana c=1 (02/09, faixa 51–67): 56 tok/s' },
        glm: { name: 'GLM-5.3 · DFlash2 (estimativa)', parts: [['experts (~57 de 288)', 15], ['KDA', 4.7], ['MLA + indexador', 1.3], ['shared experts', 1.1], ['lm_head', 0.6], ['MLP densa', 0.45]], bw: 230, tau: 7.7, tfix: 18, meas: 65.1, measL: 'structured ×1 (lab, 30/08): 65,1 tok/s' },
      };
      const KVB = 88064; // B/token/GPU: GQA-8 BF16, 43 camadas, TP=2 (hipotético)
      const S = { m: 'qwen', bw: 240, tau: 3.2, tfix: 20, ctx: 0, scale: 14, lvl: 0 };
      const ctl = UI.controls(f);
      const calc = () => {
        const Mo = MOD[S.m];
        const Q = Mo.parts.reduce((a, p) => a + p[1], 0) + KVB * S.ctx / 1e9;
        const msS = Q / S.bw * 1000, ms = msS + S.tfix;
        return { Q, msS, ms, tokS: S.tau / (msS / 1000), tok: S.tau / (ms / 1000), fS: msS / ms };
      };
      const upd = () => { const r = calc(); rMs.set(nf(r.ms, 1) + ' ms'); rS.set(nf(r.tokS, 1) + ' tok/s'); rT.set(nf(r.tok, 1) + ' tok/s'); rM.set(MOD[S.m].measL.split(': ').pop()); };
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'glm', label: 'GLM' }], value: 'qwen', onChange: (v) => { S.m = v; sB.set(MOD[v].bw); sT.set(MOD[v].tau); sF.set(MOD[v].tfix); upd(); } });
      const sB = UI.slider(ctl, { label: 'Banda', min: 180, max: 273, step: 1, value: 240, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; upd(); } });
      const sT = UI.slider(ctl, { label: 'Tokens/passo', min: 1, max: 8, step: 0.05, value: 3.2, fmt: (v) => nf(v, 2), onInput: (v) => { S.tau = v; upd(); } });
      const sF = UI.slider(ctl, { label: 'Custo fixo', min: 0, max: 30, step: 1, value: 20, fmt: (v) => v + ' ms', onInput: (v) => { S.tfix = v; upd(); } });
      UI.slider(ctl, { label: 'KV denso (hip.)', min: 0, max: 262144, step: 8192, value: 0, fmt: (v) => (v === 0 ? 'off' : nf(v / 1024) + 'k'), onInput: (v) => { S.ctx = v; upd(); } });
      const rMs = UI.readout(ctl, 'passo (streaming + fixo)', '—');
      const rS = UI.readout(ctl, 'teto só streaming', '—');
      const rT = UI.readout(ctl, 'previsão', '—');
      const rM = UI.readout(ctl, 'medido', '—');
      upd();
      V.fig(f, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 620;
        const Mo = MOD[S.m];
        const r = calc();
        const parts = Mo.parts.slice();
        if (S.ctx > 0) parts.push(['KV denso hipotético', KVB * S.ctx / 1e9]);
        S.scale += (Math.max(12, r.Q * 1.1) - S.scale) * Math.min(1, dt * 4);
        const pad = 16;
        const tx = pad, tW = narrow ? 58 : 92, ty0 = 30;
        const ty1 = narrow ? h - 300 : h - 64;
        const th = ty1 - ty0;
        V.text(g, fit(g, Mo.name, narrow ? w - 2 * pad : w * 0.55, 13, 'display', 600), pad, 18, { size: 13, font: 'display', weight: 600, color: C.ink });
        // tanque: esvazia durante o streaming e fica vazio durante o custo fixo
        V.box(g, tx - 3, ty0 - 3, tW + 6, th + 6, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.4), lw: 1.2 });
        const dispT = Math.max(0.8, r.ms / 40); // 40 ms de passo = 1 s de animação
        const ph = (t % dispT) / dispT;
        const inFix = ph >= r.fS;
        const liquid = inFix ? 0 : r.Q * (1 - ph / r.fS);
        let acc = 0;
        const segs = [];
        parts.forEach((p, k) => {
          const y1 = ty1 - (acc / S.scale) * th;
          const y0 = ty1 - ((acc + p[1]) / S.scale) * th;
          const col = p[0].startsWith('KV denso') ? C.red : C.series[k % C.series.length];
          const cut = ty1 - (liquid / S.scale) * th;
          const fy0 = Math.max(y0, cut);
          if (y1 > fy0) { g.fillStyle = V.alpha(col, 0.78); g.fillRect(tx, fy0, tW, y1 - fy0); }
          if (fy0 > y0) { g.fillStyle = V.alpha(col, 0.16); g.fillRect(tx, y0, tW, Math.min(fy0, y1) - y0); }
          V.line(g, tx, y0, tx + tW, y0, { color: V.alpha(C.bg, 0.6), lw: 1 });
          segs.push({ y: (y0 + y1) / 2, col, name: p[0], v: p[1], hgt: y1 - y0 });
          acc += p[1];
        });
        const lvlY = ty1 - (liquid / S.scale) * th;
        if (!inFix) V.line(g, tx, lvlY, tx + tW, lvlY, { color: C.ink, lw: 1.5 });
        // rótulos sem sobreposição
        const lh = 15;
        const ys = segs.map((s) => s.y);
        for (let k = 1; k < ys.length; k++) if (ys[k] > ys[k - 1] - lh) ys[k] = ys[k - 1] - lh;
        const topLim = ty0 + 8;
        if (ys.length && ys[ys.length - 1] < topLim) { const sh = topLim - ys[ys.length - 1]; for (let k = 0; k < ys.length; k++) ys[k] += sh; for (let k = ys.length - 2; k >= 0; k--) if (ys[k] < ys[k + 1] + lh) ys[k] = ys[k + 1] + lh; }
        const lx0 = tx + tW + 14;
        const lMax = narrow ? w - pad - lx0 : w * 0.56 - lx0;
        segs.forEach((s, k) => {
          V.line(g, tx + tW + 2, s.y, lx0 - 2, ys[k], { color: V.alpha(s.col, 0.6), lw: 1 });
          V.dot(g, lx0 + 3, ys[k], 3.5, s.col);
          const txt = s.name + ' · ' + nf(s.v, s.v < 1 ? 2 : 1) + ' GB';
          V.text(g, fit(g, txt, lMax - 12, 11.5), lx0 + 11, ys[k] + 4, { size: 11.5, color: C.ink });
        });
        // cano + GPU (no custo fixo, o cano fica parado e a GPU espera coletivas e lançamentos)
        const gpuW = narrow ? 58 : 80, gpuX = tx + tW / 2 - gpuW / 2, gpuY = ty1 + 16;
        const pipeW = 4 + (S.bw - 180) / 93 * 6;
        g.fillStyle = V.alpha(C.a, 0.25); g.fillRect(tx + tW / 2 - pipeW / 2, ty1 + 3, pipeW, gpuY - ty1 - 3);
        if (!inFix) for (let k = 0; k < 3; k++) { const u = (t * (S.bw / 120) + k / 3) % 1; V.dot(g, tx + tW / 2, ty1 + 3 + u * (gpuY - ty1 - 3), 2, C.a); }
        const gcol = inFix ? C.c : C.a;
        if (inFix) V.glow(g, gpuX + gpuW / 2, gpuY + 12, gpuW * 0.8, C.c, 0.4 + 0.2 * Math.sin(t * 9));
        V.box(g, gpuX, gpuY, gpuW, 24, { r: 7, fill: V.alpha(gcol, 0.18), stroke: gcol, lw: 1.3, label: 'GPU', size: 11.5, font: 'mono' });
        V.text(g, inFix ? 't_fixo: coletivas, lançamentos…' : 'streaming dos pesos', gpuX + gpuW + 10, gpuY + 16, { size: 11, font: 'mono', color: inFix ? C.c : C.muted });
        // painel de números
        const lines = [
          ['Q = ' + nf(r.Q, 2) + ' GB lidos por passo', C.ink],
          ['Q/β = ' + nf(r.Q, 2) + ' GB ÷ ' + S.bw + ' GB/s = ' + nf(r.msS, 1) + ' ms', C.ink],
          ['+ t_fixo ' + S.tfix + ' ms = passo de ' + nf(r.ms, 1) + ' ms', C.c],
          [nf(1000 / r.ms, 1) + ' passos/s × ' + nf(S.tau, 2) + ' tokens/passo', C.muted],
        ];
        let px0, py0, pw;
        if (narrow) { px0 = pad; py0 = gpuY + 48; pw = w - 2 * pad; } else { px0 = w * 0.6; py0 = 56; pw = w - px0 - pad; }
        let yy = py0;
        lines.forEach((L) => { V.text(g, fit(g, L[0], pw, 12.5, 'mono'), px0, yy, { size: 12.5, font: 'mono', color: L[1] }); yy += 21; });
        yy += narrow ? 14 : 22;
        V.text(g, '≈ ' + nf(r.tok, 1), px0, yy, { size: narrow ? 26 : 34, font: 'display', weight: 700, color: C.a });
        V.text(g, 'tok/s (previsão)', px0 + tw(g, '≈ ' + nf(r.tok, 1), narrow ? 26 : 34, 'display', 700) + 10, yy - 4, { size: 11.5, color: C.muted });
        yy += narrow ? 16 : 22;
        // barras: só streaming × previsão × medido
        const bmax = Math.max(r.tokS, Mo.meas) * 1.08;
        const bW = pw - 10;
        [['só streaming', r.tokS, C.d], ['previsão', r.tok, C.a], ['medido', Mo.meas, C.c]].forEach((b, k) => {
          const y = yy + k * 22;
          V.box(g, px0, y, bW, 15, { r: 5, fill: V.alpha(C.muted, 0.12) });
          const bw_ = Math.max(4, bW * b[1] / bmax);
          V.box(g, px0, y, bw_, 15, { r: 5, fill: V.alpha(b[2], 0.78) });
          const lbl = b[0] + ' ' + nf(b[1], 1);
          const inside = tw(g, lbl, 11, 'mono', 600) + 12 < bw_;
          V.text(g, lbl, inside ? px0 + 6 : px0 + bw_ + 6, y + 11.5, { size: 11, font: 'mono', weight: 600, color: inside ? onInk() : C.ink });
        });
        yy += 3 * 22 + 10;
        const pct = 'streaming = ' + nf(100 * r.fS, 0) + '% do passo previsto';
        V.text(g, fit(g, Mo.measL + (S.ctx > 0 ? ' · KV denso hipotético' : ''), pw, 11), px0, Math.min(yy, h - 24), { size: 11, color: C.muted });
        V.text(g, fit(g, pct, pw, 11), px0, Math.min(yy + 16, h - 8), { size: 11, color: C.muted });
      });
    }

    /* ===================== Figura 5: KV cache, passo a passo ===================== */
    {
      const f = $('#fig-kv-mecanismo');
      fitH(f, 450, 680, 760);
      const r1 = (x) => Math.round(x * 10) / 10;
      const R = V.rng(7); const rnd = (s) => r1(R.normal() * s);
      const Kp = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.9)));
      const Vp = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.9)));
      const WQ = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.7)));
      const WK = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.7)));
      const WV = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => rnd(0.7)));
      const xv = [0.8, -0.4, 1.1, 0.3];
      const mv = (Wm) => [0, 1, 2, 3].map((j) => r1(xv.reduce((s, a, k) => s + a * Wm[k][j], 0)));
      const q = mv(WQ), kn = mv(WK), vn = mv(WV);
      const Ka = [...Kp, kn], Va = [...Vp, vn];
      const sc = Ka.map((row) => Math.round((row.reduce((a, b, i) => a + b * q[i], 0) / 2) * 100) / 100);
      const al = V.softmax(sc);
      const o = [0, 1, 2, 3].map((j) => r1(al.reduce((s, a, i) => s + a * Va[i][j], 0)));
      const TOK = ['O', 'gato', 'subiu', 'no', 'telhado'];
      const S = { noCache: false };
      const st = UI.stepper(f, [
        { t: 'O token novo chega', d: 'O último token gerado, <b>telhado</b>, entra na camada como o vetor \\(h_t\\) (4 números). Os K e V dos 4 tokens anteriores já estão no cache: foram calculados no prefill.' },
        { t: 'Três projeções', d: '\\(q_t = W_Q h_t\\), \\(k_t = W_K h_t\\), \\(v_t = W_V h_t\\): três multiplicações matriz × vetor. Com o cache, é o <b>único</b> cálculo novo de K e V neste passo.' },
        { t: 'Arquivar k e v', d: '\\(k_t\\) e \\(v_t\\) viram a 5ª linha do cache. Elas nunca mais mudam: nos próximos passos, só serão lidas.' },
        { t: 'Notas: q contra cada chave', d: '\\(s_j = q_t\\cdot k_j/\\sqrt{4}\\) para as 5 linhas. Aqui o decode <b>lê o cache inteiro</b>: cada passo percorre todas as chaves guardadas.' },
        { t: 'Softmax', d: 'As notas viram pesos positivos que somam 1. “telhado” olha sobretudo para “no” e “subiu”.' },
        { t: 'Mistura dos valores', d: '\\(o_t = \\sum_j \\alpha_j v_j\\): as linhas de V são lidas e misturadas. A saída segue para a próxima camada, que tem o seu próprio cache.' },
      ], { autoplay: 5200 });
      const ctl = UI.controls(f);
      UI.toggle(ctl, { label: 'sem KV cache (recalcular tudo)', value: false, onChange: (v) => { S.noCache = v; } });
      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const i = st.i, e = st.elapsed();
        const wide = w >= 760;
        const lab = wide ? 62 : 56;
        const c = wide ? 34 : Math.floor(clamp(Math.min((w - 84) / 8, (w - lab - 116) / 5), 24, 32));
        const gA = wide ? 8 : 18;
        // ---------- posições ----------
        const pos = {};
        pos.x = { x: wide ? lab : 14, y: 34 };
        const wbX = pos.x.x + 4 * c + (wide ? 26 : 16);
        pos.w = { x: wbX, y: 34 };
        const qkvX = wbX + 34 + (wide ? 26 : 14);
        pos.q = { x: qkvX, y: 34 }; pos.k = { x: qkvX, y: 34 + c + gA }; pos.v = { x: qkvX, y: 34 + 2 * (c + gA) };
        const rowB = 34 + 3 * c + 2 * gA + 50;
        pos.K = { x: lab, y: rowB };
        pos.s = { x: lab + 4 * c + 18, y: rowB };
        const sW = c + 16;
        pos.a = { x: pos.s.x + sW + 16, y: rowB };
        const aW = wide ? 76 : 62;
        if (wide) { pos.V = { x: pos.a.x + aW + 70, y: rowB }; pos.o = { x: pos.a.x + aW + 70 + 4 * c + 40, y: rowB + 2 * c }; }
        else { const rowC = rowB + 5 * c + 34; pos.V = { x: lab, y: rowC }; pos.o = { x: lab + 4 * c + 34, y: rowC + 2 * c }; }
        const vec = (p, data, o2 = {}) => { for (let j = 0; j < 4; j++) { if (o2.upto != null && j >= o2.upto) { V.rr(g, p.x + j * c, p.y, c - 2, c - 2, 4); g.strokeStyle = V.alpha(C.muted, 0.35); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); continue; } cell(g, p.x + j * c, p.y, c, data[j], { alpha: o2.alpha, stroke: o2.stroke }); } };
        // ---------- linha 1: h_t → W → q,k,v ----------
        const xin = i === 0 ? clamp(e / 0.6, 0, 1) : 1;
        V.text(g, 'h_t  (' + TOK[4] + ')', pos.x.x, pos.x.y - 10, { size: 11, font: 'mono', color: C.muted });
        vec(pos.x, xv, { alpha: xin, stroke: i <= 1 ? C.a : null });
        if (i === 0) V.glow(g, pos.x.x + 2 * c, pos.x.y + c / 2, 2 * c, C.a, 0.3 * (1 - xin) + 0.1);
        const names = ['W_Q', 'W_K', 'W_V'];
        const pr = i === 1 ? clamp(e / 1.3, 0, 1) : i > 1 ? 1 : 0;
        for (let k = 0; k < 3; k++) {
          const yy = 34 + k * (c + gA);
          const hot = i === 1;
          V.box(g, wbX, yy, 34, c - 2, { r: 6, fill: V.alpha(C.b, hot ? 0.25 : 0.1), stroke: V.alpha(C.b, hot ? 0.9 : 0.4), lw: 1.2 });
          V.text(g, names[k], wbX + 17, yy + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          V.arrow(g, pos.x.x + 4 * c + 2, pos.x.y + c / 2, wbX - 3, yy + c / 2 - 1, { color: V.alpha(C.muted, hot ? 0.8 : 0.35), lw: 1.2, head: 5 });
          V.arrow(g, wbX + 36, yy + c / 2 - 1, qkvX - 4, yy + c / 2 - 1, { color: V.alpha(C.muted, hot ? 0.8 : 0.35), lw: 1.2, head: 5 });
        }
        const upto = Math.floor(pr * 4 + 0.001);
        const vlab = (s2, p2, a2) => (wide ? V.text(g, s2, qkvX + 4 * c + 6, p2.y + c / 2 + 3, { size: 11, font: 'mono', color: C.muted, alpha: a2 }) : V.text(g, s2, qkvX, p2.y - 4, { size: 11, font: 'mono', color: C.muted, alpha: a2 }));
        vlab('q_t', pos.q, 1);
        vec(pos.q, q, { upto: i >= 1 ? upto : 0, stroke: i === 3 ? C.c : null });
        // k e v: no passo 2 viajam até o cache
        const mv2 = i === 2 ? V.ease.inOut(clamp(e / 1.1, 0, 1)) : i > 2 ? 1 : 0;
        const kTarget = { x: pos.K.x, y: pos.K.y + 4 * c }, vTarget = { x: pos.V.x, y: pos.V.y + 4 * c };
        if (mv2 < 1) {
          const kp = { x: lerp(pos.k.x, kTarget.x, mv2), y: lerp(pos.k.y, kTarget.y, mv2) };
          const vp = { x: lerp(pos.v.x, vTarget.x, mv2), y: lerp(pos.v.y, vTarget.y, mv2) };
          vec(kp, kn, { upto: i >= 1 ? upto : 0, stroke: i === 2 ? C.a : null });
          vec(vp, vn, { upto: i >= 1 ? upto : 0, stroke: i === 2 ? C.b : null });
        }
        vlab('k_t', pos.k, mv2 < 1 ? 1 : 0.35);
        vlab('v_t', pos.v, mv2 < 1 ? 1 : 0.35);
        // ---------- caches ----------
        const recalc = S.noCache && (i === 1 || i === 2);
        const drawCache = (p, data, lbl, col) => {
          V.text(g, lbl, p.x, p.y - 10, { size: 11, font: 'mono', color: C.muted });
          for (let rI = 0; rI < 5; rI++) {
            V.text(g, TOK[rI], p.x - 6, p.y + rI * c + (c - 2) / 2 + 0.5, { size: 11, font: 'mono', color: rI === 4 ? C.ink : C.muted, align: 'right', base: 'middle' });
            const has = rI < 4 || mv2 >= 1;
            for (let j = 0; j < 4; j++) {
              if (!has) { V.rr(g, p.x + j * c, p.y + rI * c, c - 2, c - 2, 4); g.strokeStyle = V.alpha(col, 0.5); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); continue; }
              const hiRow = (i === 3 && p === pos.K && rI === Math.min(4, Math.floor(e / 0.45))) || (i === 5 && p === pos.V);
              cell(g, p.x + j * c, p.y + rI * c, c, data[rI][j], { alpha: rI < 4 && !hiRow && !recalc ? 0.62 : 1, stroke: hiRow ? C.c : (rI === 4 && i === 2 ? col : null) });
            }
            if (recalc && rI < 4) {
              const fl = 0.5 + 0.5 * Math.sin(t * 8 + rI);
              V.rr(g, p.x - 2, p.y + rI * c - 2, 4 * c + 2, c + 2, 5); g.strokeStyle = V.alpha(C.red, 0.5 + 0.4 * fl); g.lineWidth = 1.8; g.stroke();
            }
          }
          if (recalc) V.text(g, 'recalculando 4 linhas', p.x + 2 * c, p.y + 5 * c + 12, { size: 11, font: 'mono', weight: 600, color: C.red, align: 'center' });
        };
        drawCache(pos.K, Ka, 'K cache', C.a);
        drawCache(pos.V, Va, 'V cache', C.b);
        // ---------- notas ----------
        V.text(g, 'notas s', pos.s.x, pos.s.y - 10, { size: 11, font: 'mono', color: C.muted });
        const sUpto = i === 3 ? Math.min(5, Math.floor(e / 0.45) + 1) : i > 3 ? 5 : 0;
        for (let rI = 0; rI < 5; rI++) {
          const x = pos.s.x, y = pos.s.y + rI * c;
          if (rI < sUpto) { cell(g, x, y, c, sc[rI], { wd: sW, scale: 2, fmt: (v) => nf(v, 2) }); }
          else { V.rr(g, x, y, sW - 2, c - 2, 4); g.strokeStyle = V.alpha(C.muted, 0.3); g.setLineDash([3, 3]); g.lineWidth = 1; g.stroke(); g.setLineDash([]); }
          if (i === 3 && rI === sUpto - 1) V.arrow(g, pos.K.x + 4 * c, y + c / 2 - 1, x - 3, y + c / 2 - 1, { color: C.c, lw: 1.4, head: 5 });
        }
        // ---------- pesos α ----------
        V.text(g, 'pesos α', pos.a.x, pos.a.y - 10, { size: 11, font: 'mono', color: C.muted });
        const ag = i === 4 ? V.ease.out(clamp(e / 1.2, 0, 1)) : i > 4 ? 1 : 0;
        for (let rI = 0; rI < 5; rI++) {
          const y = pos.a.y + rI * c + 3;
          V.box(g, pos.a.x, y, aW, c - 8, { r: 4, fill: V.alpha(C.muted, 0.1) });
          if (ag > 0) {
            V.box(g, pos.a.x, y, Math.max(2, aW * al[rI] * ag), c - 8, { r: 4, fill: V.alpha(C.d, 0.8) });
            V.text(g, nf(al[rI] * ag, 2), pos.a.x + aW - 3, y + (c - 8) / 2 + 0.5, { size: 11, font: 'mono', color: C.ink, align: 'right', base: 'middle' });
          }
        }
        // ---------- saída ----------
        const og = i === 5 ? clamp((e - 0.6) / 1.4, 0, 1) : 0;
        V.text(g, 'o_t', pos.o.x, pos.o.y - 10, { size: 11, font: 'mono', color: C.muted });
        vec(pos.o, o, { upto: i === 5 ? Math.floor(og * 4 + 0.001) : 0, stroke: i === 5 && og >= 1 ? C.a : null });
        if (i === 5) {
          for (let rI = 0; rI < 5; rI++) {
            const x1 = pos.V.x + 4 * c, y1 = pos.V.y + rI * c + c / 2;
            const u = (t * 0.9 + rI * 0.13) % 1;
            const pt = [lerp(x1, pos.o.x - 4, u), lerp(y1, pos.o.y + c / 2, u)];
            V.line(g, x1, y1, pos.o.x - 4, pos.o.y + c / 2, { color: V.alpha(C.d, 0.2 + 0.7 * al[rI]), lw: 0.8 + 5 * al[rI] });
            V.dot(g, pt[0], pt[1], 1.5 + 4 * al[rI], C.d);
          }
        }
        // ---------- rodapé ----------
        const yb = h - (wide ? 30 : 74);
        const p1 = S.noCache ? (wide ? 'SEM CACHE: refaz K e V das 5 linhas (4 repetidas) e, nas camadas de baixo, o forward inteiro delas' : 'SEM CACHE: refaz K e V das 5 linhas e o forward delas nas camadas de baixo') :'COM CACHE: este passo faz 1 projeção nova de K e de V e lê 5 linhas';
        const yb2 = textBlock(g, p1, 14, yb, w - 28, { size: 12, font: 'mono', weight: 600, color: S.noCache ? C.red : C.a, lh: 16 });
        textBlock(g, 'gerar n tokens: n token-forwards com cache, n(n+1)/2 sem · n = 100 mil → 100 mil × 5 bilhões', 14, yb2 + 4, w - 28, { size: 11, font: 'mono', color: C.muted, lh: 15 });
      });
    }

    /* ===================== Figura 6: o KV cache como tensor 3D ===================== */
    {
      const f = $('#fig-kv-tensor');
      const stage = f.querySelector('.la-fig-stage');
      const narrow = (f.clientWidth || 800) < 560;
      const NS = 4, NR = 12, NCOL = 8;
      const S = { m: 'qwen', n: 7, tp: false };
      const EMPTY = 'color-mix(in srgb, var(--viz-muted) 12%, transparent)';
      const mix = (v, p) => `color-mix(in srgb, var(${v}) ${p}%, transparent)`;
      const stripe = (v) => `repeating-linear-gradient(90deg, color-mix(in srgb, var(${v}) 85%, transparent) 0 3px, color-mix(in srgb, var(${v}) 30%, transparent) 3px 5px)`;
      const REPL = 'linear-gradient(135deg, var(--viz-a) 0 50%, var(--viz-d) 50% 100%)';
      const HATCH = 'repeating-linear-gradient(45deg, color-mix(in srgb, var(--viz-muted) 55%, transparent) 0 2px, transparent 2px 4px)';
      const isAtt = (s) => S.m === 'mha' || s === NS - 1;
      const colorFor = (s, i, j) => {
        const n = S.n, tp = S.tp;
        if (!isAtt(s)) { // estado fixo (GDN / KDA): não cresce com os tokens
          if (n === 0 || i >= 4) return EMPTY;
          if (!tp) return stripe('--viz-e');
          return j < 4 ? stripe('--viz-a') : stripe('--viz-d');
        }
        if (i >= n) return EMPTY;
        if (S.m === 'mha') { const head = j >> 1, isV = j & 1; if (tp) return head < 2 ? mix('--viz-a', isV ? 50 : 90) : mix('--viz-d', isV ? 50 : 90); return isV ? mix('--viz-b', 85) : mix('--viz-a', 85); }
        if (S.m === 'qwen') { const head = j >> 2, isV = (j % 4) >= 2; if (tp) return head === 0 ? mix('--viz-a', isV ? 50 : 90) : mix('--viz-d', isV ? 50 : 90); return isV ? mix('--viz-b', 85) : mix('--viz-a', 85); }
        // GLM (MLA): 0–5 latente, 6 RoPE zerado, 7 indexador (1 entrada a cada 4 tokens)
        if (j === 6) return HATCH;
        if (j === 7) { if ((i + 1) % 4 !== 0) return EMPTY; return tp ? REPL : mix('--viz-d', 90); }
        return tp ? REPL : mix('--viz-f', 80);
      };
      // rótulos só na placa da frente (embaixo dela) e na do fundo (em cima dela): nada os cobre
      const lab = (s) => {
        if (s !== 0 && s !== NS - 1) return '';
        const front = s === NS - 1;
        if (S.m === 'mha') return front ? 'camada 4 · KV cresce' : 'camada 1 · KV cresce';
        if (S.m === 'qwen') return front ? 'QSA · KV cresce' : '3 × GDN · estado fixo';
        return front ? 'MLA · KV cresce' : '3 × KDA · estado fixo';
      };
      const T = UI.tensor3d(stage, { slabs: NS, rows: NR, cols: NCOL, cell: narrow ? 12 : 16, depthGap: narrow ? 40 : 58, rotate: true, color: colorFor, slabColor: () => 'var(--viz-a)', labels: Array.from({ length: NS }, (_, s) => lab(s)), rx: -22, ry: -30 });
      // folga vertical para os rótulos acima da placa do fundo e abaixo da placa da frente (o wrap corta o excesso)
      T.el.style.padding = narrow ? '46px 10px' : '64px 10px';
      T.slabs.forEach((sl, s) => {
        const lb = sl.querySelector('.slab-label');
        if (!lb) return;
        const front = s === NS - 1;
        Object.assign(lb.style, {
          left: '0', marginLeft: '0', transform: 'none', padding: '2px 7px', borderRadius: '6px',
          background: 'var(--surface-solid)', border: '1px solid color-mix(in srgb, var(--sc) 60%, transparent)', color: 'var(--text-2)',
          top: front ? '100%' : 'auto', bottom: front ? 'auto' : '100%', marginTop: front ? '8px' : '0', marginBottom: front ? '0' : '8px',
        });
      });
      const DATA = {
        mha: { real: '2.949.120 B', gpu: '1.474.560 B', grow: '45 de 45', leg: [['var(--viz-a)', 'K'], ['var(--viz-b)', 'V']] },
        qwen: { real: '26.624 B (BF16)', gpu: '13.312 B', grow: '12 de 48 (+ MTP)', leg: [['var(--viz-a)', 'K'], ['var(--viz-b)', 'V'], [stripe('--viz-e'), 'estado GDN (listrado)']] },
        glm: { real: '7.579 B', gpu: '7.579 B (replicado)', grow: '11 de 45', leg: [['var(--viz-f)', 'latente MLA (FP8)'], ['var(--viz-d)', 'indexador (1 a cada 4)'], [HATCH, 'RoPE zerado'], [stripe('--viz-e'), 'estado KDA (listrado)']] },
      };
      const ctl = UI.controls(f);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'mha', label: 'MHA clássica' }, { v: 'qwen', label: 'Qwen (GQA)' }, { v: 'glm', label: 'GLM (MLA)' }], value: 'qwen', onChange: (v) => { S.m = v; repaint(); } });
      UI.slider(ctl, { label: 'Tokens', min: 0, max: NR, step: 1, value: S.n, fmt: (v) => String(v), onInput: (v) => { S.n = v; repaint(); } });
      UI.toggle(ctl, { label: 'TP=2: dividir entre 2 GPUs', value: false, onChange: (v) => { S.tp = v; repaint(); } });
      const r1_ = UI.readout(ctl, 'bytes por token', '—');
      const r2_ = UI.readout(ctl, 'por GPU com TP=2', '—');
      const r3_ = UI.readout(ctl, 'camadas que crescem', '—');
      const legBox = LA.h('div', { class: 'la-legend' });
      ctl.appendChild(legBox);
      function repaint() {
        for (let s = 0; s < NS; s++) {
          for (let i = 0; i < NR; i++) for (let j = 0; j < NCOL; j++) T.setCell(s, i, j, colorFor(s, i, j));
          const sl = T.slabs[s];
          sl.style.setProperty('--sc', isAtt(s) ? 'var(--viz-a)' : 'var(--viz-e)');
          const lb = sl.querySelector('.slab-label');
          if (lb) lb.textContent = lab(s);
        }
        const D = DATA[S.m];
        r1_.set(D.real); r2_.set(D.gpu); r3_.set(D.grow);
        legBox.innerHTML = '';
        let legs = D.leg.slice();
        if (S.tp) legs = S.m === 'glm' ? [['var(--viz-a)', 'GPU A'], ['var(--viz-d)', 'GPU B'], [REPL, 'bicolor = cópia inteira nas 2 GPUs']] : [['var(--viz-a)', 'GPU A (forte = K, fraco = V)'], ['var(--viz-d)', 'GPU B']];
        legs.forEach(([c, l]) => legBox.appendChild(LA.h('span', null, LA.h('i', { style: { '--c': c } }), l)));
      }
      repaint();
      // faixa de cabeças: quem pergunta (Q) consulta qual K/V guardado
      const hc = document.createElement('canvas');
      hc.style.cssText = 'display:block;width:100%;height:122px;margin-top:6px';
      stage.appendChild(hc);
      V.loop(hc, (g, w, h, t) => {
        const C = V.C();
        const nr = w < 480;
        const tp = S.tp;
        const cfg = S.m === 'mha' ? { nQ: 16, nKV: 16, title: nr ? 'MHA: 1 par K/V por cabeça' : 'MHA: cada cabeça de query tem o seu par K/V (16 desenhadas)' }
          : S.m === 'qwen' ? { nQ: 24, nKV: 2, title: nr ? 'GQA: 24 queries → 2 pares K/V' : 'GQA (Qwen): 24 cabeças de query dividem 2 pares K/V, 12 por par' }
          : { nQ: nr ? 24 : 32, nKV: 1, title: nr ? 'MLA: 64 queries → 1 latente' : 'MLA (GLM): as 64 cabeças de query leem 1 latente comprimido (' + (nr ? 24 : 32) + ' desenhadas)' };
        const x0 = nr ? 34 : 46, x1 = w - 6, yQ = 42, yK = 94, bh = 22;
        const sq = (x1 - x0) / cfg.nQ;
        const qx = (i) => x0 + (i + 0.5) * sq;
        const half = (x0 + x1) / 2;
        V.text(g, fit(g, cfg.title, w - 8, 11.5, 'mono', 600), 4, 14, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, 'Q', x0 - 10, yQ + 4, { size: 11, font: 'mono', weight: 600, color: C.c, align: 'right' });
        V.text(g, nr ? 'KV' : 'K/V', x0 - 10, yK + 4, { size: 11, font: 'mono', weight: 600, color: C.a, align: 'right' });
        if (tp) {
          g.fillStyle = V.alpha(C.a, 0.06); g.fillRect(x0, 24, half - x0 - 2, h - 26);
          g.fillStyle = V.alpha(C.d, 0.06); g.fillRect(half + 2, 24, x1 - half - 2, h - 26);
          V.text(g, 'GPU A', half - 6, h - 6, { size: 11, font: 'mono', color: C.a, align: 'right' });
          V.text(g, 'GPU B', x1 - 4, h - 6, { size: 11, font: 'mono', color: C.d, align: 'right' });
        }
        // caixas de K/V: posição e dono de cada cabeça de query
        const boxes = [];
        if (cfg.nKV === 1 && tp) { // MLA replicada: uma cópia inteira em cada GPU
          const bw = Math.min(150, (half - x0) * 0.8);
          boxes.push({ x: (x0 + half) / 2 - bw / 2, w: bw, gpu: 0, lbl: nr ? 'cópia A' : 'latente (cópia A)' }, { x: (half + x1) / 2 - bw / 2, w: bw, gpu: 1, lbl: nr ? 'cópia B' : 'latente (cópia B)' });
        } else {
          const sk = (x1 - x0) / cfg.nKV;
          for (let k = 0; k < cfg.nKV; k++) {
            const bw = cfg.nKV === 1 ? Math.min(180, (x1 - x0) * 0.5) : Math.max(8, sk - (cfg.nKV > 4 ? 4 : 28));
            boxes.push({ x: x0 + (k + 0.5) * sk - bw / 2, w: bw, gpu: k < cfg.nKV / 2 ? 0 : 1, lbl: cfg.nKV === 1 ? 'latente' : cfg.nKV === 2 ? 'K/V ' + (k + 1) : '' });
          }
        }
        const owner = (i) => {
          if (cfg.nKV === 1) return tp ? (i < cfg.nQ / 2 ? 0 : 1) : 0;
          return Math.floor(i * cfg.nKV / cfg.nQ);
        };
        const gpuCol = (gi) => (gi === 0 ? C.a : C.d);
        for (let i = 0; i < cfg.nQ; i++) {
          const b = boxes[owner(i)];
          const bx = b.x + b.w / 2, x = qx(i);
          V.line(g, x, yQ + 5, bx, yK - bh / 2, { color: V.alpha(tp ? gpuCol(b.gpu) : C.muted, 0.28), lw: 1 });
          const u = (t * 0.7 + i * 0.137) % 1;
          V.dot(g, lerp(bx, x, u), lerp(yK - bh / 2, yQ + 5, u), 1.8, tp ? gpuCol(b.gpu) : C.a);
        }
        boxes.forEach((b) => {
          const y = yK - bh / 2;
          if (S.m === 'glm') {
            V.box(g, b.x, y, b.w, bh, { r: 6, fill: V.alpha(tp ? gpuCol(b.gpu) : C.f, 0.35), stroke: tp ? gpuCol(b.gpu) : C.f, lw: 1.4 });
          } else {
            const kc = tp ? gpuCol(b.gpu) : C.a, vc = tp ? gpuCol(b.gpu) : C.b;
            V.box(g, b.x, y, b.w / 2, bh, { r: 4, fill: V.alpha(kc, tp ? 0.55 : 0.5) });
            V.box(g, b.x + b.w / 2, y, b.w / 2, bh, { r: 4, fill: V.alpha(vc, tp ? 0.28 : 0.5) });
          }
          if (b.lbl && tw(g, b.lbl, 11, 'mono', 600) < b.w - 6) V.text(g, b.lbl, b.x + b.w / 2, yK + 0.5, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
        });
        for (let i = 0; i < cfg.nQ; i++) {
          const b = boxes[owner(i)];
          const col = tp ? gpuCol(b.gpu) : C.c;
          const rr = Math.max(2.5, Math.min(5.5, sq * 0.32));
          g.beginPath(); g.arc(qx(i), yQ, rr, 0, Math.PI * 2); g.fillStyle = V.alpha(col, 0.35); g.fill(); g.strokeStyle = col; g.lineWidth = 1.3; g.stroke();
        }
      });
    }

    /* ===================== Figura 7: o pool de KV ===================== */
    {
      const f = $('#fig-pool');
      fitH(f, 362, 348);
      const CTX = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
      const POOL = { fp8: { tok: 3652200, gib: 32.02 }, bf16: { tok: 2131159, gib: 31.7 } };
      const SEQS = 8;
      const S = { dt: 'fp8', ci: 4, n: 6, anim: [] };
      const ctl = UI.controls(f);
      const st = () => { const N = POOL[S.dt].tok, ctx = CTX[S.ci]; const capMem = Math.floor(N / ctx); const adm = Math.min(S.n, SEQS, capMem); const waitSeq = Math.max(0, Math.min(S.n, capMem) - adm); const waitMem = Math.max(0, S.n - adm - waitSeq); return { N, ctx, capMem, adm, waitSeq, waitMem }; };
      const upd = () => { const s = st(); rC.set(nf(s.capMem) + ' conversas'); rA.set(s.adm + ' de ' + S.n); rU.set(nf(100 * s.adm * s.ctx / s.N, 1) + '%'); };
      UI.seg(ctl, { label: 'KV', options: [{ v: 'fp8', label: 'FP8' }, { v: 'bf16', label: 'BF16' }], value: 'fp8', onChange: (v) => { S.dt = v; upd(); } });
      UI.slider(ctl, { label: 'Contexto', min: 0, max: CTX.length - 1, step: 1, value: S.ci, fmt: (v) => nf(CTX[v]), onInput: (v) => { S.ci = v; upd(); } });
      UI.slider(ctl, { label: 'Conversas', min: 1, max: 20, step: 1, value: S.n, fmt: (v) => String(v), onInput: (v) => { S.n = v; upd(); } });
      const rC = UI.readout(ctl, 'cabem no pool', '—');
      const rA = UI.readout(ctl, 'admitidas', '—');
      const rU = UI.readout(ctl, 'uso do pool', '—');
      upd();
      V.fig(f, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 620;
        const pad = 16;
        const s = st();
        // ---------- orçamento do nó ----------
        const tot = 121.69, kvG = POOL[S.dt].gib;
        const segs = [['pesos + não-torch', 68.52, C.b], ['ativação', 1.07, C.d], ['pool de KV', kvG, C.a], ['fora do orçamento (SO, host)', tot - 68.52 - 1.07 - kvG, C.muted]];
        V.text(g, fit(g, narrow ? 'uma GPU do Qwen · 121,69 GiB' : 'memória de uma GPU do Qwen (TP2, GMU 0,835) · 121,69 GiB', w - 2 * pad, 11.5, 'mono'), pad, 18, { size: 11.5, font: 'mono', color: C.muted });
        const by = 28, bh = 26, bW = w - 2 * pad;
        let x = pad;
        segs.forEach((sg) => {
          const ww = bW * sg[1] / tot;
          V.box(g, x, by, Math.max(1, ww - 1.5), bh, { r: 5, fill: V.alpha(sg[2], sg[0].startsWith('fora') ? 0.18 : 0.7) });
          const lbl = sg[0].startsWith('fora') ? (narrow ? 'host' : 'SO/host') : sg[0].startsWith('pesos') ? (narrow ? 'pesos' : 'pesos + não-torch') : sg[0] === 'pool de KV' ? (narrow ? 'KV' : 'pool de KV') : '';
          const txt = lbl ? lbl + ' ' + nf(sg[1], sg[1] < 10 ? 2 : 1) : '';
          if (txt && tw(g, txt, 11, 'mono', 600) < ww - 8) V.text(g, txt, x + 6, by + bh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: sg[0].startsWith('fora') ? C.ink : onInk(), base: 'middle' });
          x += ww;
        });
        V.text(g, fit(g, 'ativação de pico 1,07 GiB (fatia fina) · CUDA graphs 0,54 GiB reportados à parte', bW, 11), pad, by + bh + 16, { size: 11, color: C.muted });
        // ---------- pool ----------
        const py = by + bh + 50, ph = narrow ? 64 : 76;
        V.text(g, 'pool de KV: ' + nf(s.N) + ' tokens (' + (S.dt === 'fp8' ? 'FP8' : 'BF16') + ')', pad, py - 10, { size: 12.5, font: 'mono', weight: 600, color: C.a });
        V.box(g, pad, py, bW, ph, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.55), lw: 1.3 });
        // grade de blocos de fundo
        g.strokeStyle = V.alpha(C.a, 0.1); g.lineWidth = 1;
        for (let k = 1; k < 40; k++) { const xx = pad + (bW * k) / 40; g.beginPath(); g.moveTo(xx, py + 3); g.lineTo(xx, py + ph - 3); g.stroke(); }
        // conversas admitidas (animadas)
        while (S.anim.length < 20) S.anim.push(0);
        const segW = bW * s.ctx / s.N;
        for (let k = 0; k < 20; k++) {
          const target = k < s.adm ? 1 : 0;
          S.anim[k] += (target - S.anim[k]) * Math.min(1, dt * 7);
        }
        let cx = pad + 2;
        for (let k = 0; k < 20; k++) {
          const a = S.anim[k];
          if (a < 0.01) continue;
          const ww = Math.max(0, segW * a - 2);
          const col = C.series[k % C.series.length];
          V.box(g, cx, py + 3, Math.max(2, ww), ph - 6, { r: 6, fill: V.alpha(col, 0.55), stroke: col, lw: 1 });
          if (ww > 22) guy(g, cx + Math.min(ww / 2, 16), py + ph / 2, Math.min(9, ph / 5), col, { blink: blinkAt(t, k), alpha: a });
          cx += segW * a;
        }
        // marca do max_model_len
        const mml = 262144;
        const mx = pad + bW * mml / s.N;
        V.line(g, mx, py - 3, mx, py + ph + 3, { color: C.d, lw: 1.5, dash: [3, 3] });
        V.text(g, fit(g, (narrow ? '1 conversa de 262.144 = ' : '1 conversa de 262.144 (max-model-len) = ') + nf(100 * mml / s.N, 1) + '% do pool', w - 2 * pad, 11, 'mono'), pad, py + ph + 15, { size: 11, font: 'mono', color: C.d });
        // ---------- fila ----------
        const qy = py + ph + 44;
        V.text(g, 'vagas do scheduler (max_num_seqs = 8):', pad, qy, { size: 11.5, font: 'mono', color: C.muted });
        const seatR = narrow ? 8 : 9;
        for (let k = 0; k < SEQS; k++) {
          const sx = pad + 10 + k * (seatR * 2 + 8), sy = qy + 20;
          const occ = k < s.adm;
          if (occ) guy(g, sx + seatR, sy + seatR, seatR, C.series[k % C.series.length], { blink: blinkAt(t, k + 3) });
          else { g.beginPath(); g.arc(sx + seatR, sy + seatR, seatR, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.4); g.setLineDash([3, 3]); g.lineWidth = 1.2; g.stroke(); g.setLineDash([]); }
        }
        const wy = qy + 20 + seatR * 2 + 24;
        const wait = s.waitSeq + s.waitMem;
        // raia da fila (vazia ou não), para o leitor ver onde os excedentes esperam
        V.rr(g, pad, wy + 8, w - 2 * pad, seatR * 2 + 12, 9);
        g.fillStyle = V.alpha(wait > 0 ? (s.waitMem > 0 ? C.red : C.d) : C.muted, 0.05); g.fill();
        g.strokeStyle = V.alpha(wait > 0 ? (s.waitMem > 0 ? C.red : C.d) : C.muted, 0.35); g.lineWidth = 1; g.setLineDash([4, 4]); g.stroke(); g.setLineDash([]);
        if (wait === 0) V.text(g, 'fila', w - pad - 10, wy + 8 + seatR + 10, { size: 11, font: 'mono', color: C.muted, align: 'right', alpha: 0.7 });
        if (wait > 0) {
          const both = s.adm === SEQS && s.adm === s.capMem;
          const why = both ? 'sem vaga e sem KV livre' : s.waitSeq > 0 && s.waitMem > 0 ? s.waitSeq + ' sem vaga + ' + s.waitMem + ' sem KV livre' : s.waitSeq > 0 ? 'sem vaga no scheduler (ainda há KV livre)' : 'sem KV livre no pool';
          V.text(g, fit(g, 'na fila: ' + wait + ' · ' + why, w - 2 * pad, 12, 'mono', 600), pad, wy, { size: 12, font: 'mono', weight: 600, color: s.waitMem > 0 ? C.red : C.d });
          const maxShow = Math.floor((w - 2 * pad) / (seatR * 2 + 4));
          for (let k = 0; k < Math.min(wait, maxShow); k++) {
            const isMem = k >= s.waitSeq;
            guy(g, pad + seatR + k * (seatR * 2 + 4), wy + 18 + seatR, seatR * 0.85, isMem ? C.red : C.d, { alpha: 0.85, blink: blinkAt(t, k + 7) });
          }
        } else {
          V.text(g, fit(g, narrow ? 'ninguém na fila' : 'ninguém na fila: todas as conversas foram admitidas', w - 2 * pad, 12, 'mono', 600), pad, wy, { size: 12, font: 'mono', weight: 600, color: C.a });
        }
        const concur = s.N / 262144;
        V.text(g, fit(g, (narrow ? 'concorrência a 262.144: ' : 'concorrência máxima a 262.144 tokens: ') + nf(concur, 2) + '×' + (narrow ? '' : ' (boot do vLLM)'), w - 2 * pad, 11, 'mono'), pad, h - 10, { size: 11, font: 'mono', color: C.muted });
      });
    }

    /* ===================== Figura 8: o relógio de uma requisição ===================== */
    {
      const f = $('#fig-relogio');
      fitH(f, 370, 400);
      const SC = [
        { v: 'ds256', label: 'DeepSeek · prompt 256', P: 256, ttft: 0.63, pre: 447, dec: 75.4, col: 'ds' },
        { v: 'ds2k', label: 'DeepSeek · prompt 2.048', P: 2048, ttft: 0.81, pre: 2563, dec: 68.8, col: 'ds' },
        { v: 'ds8k', label: 'DeepSeek · prompt 8.192', P: 8192, ttft: 4.8, pre: 1713, dec: 73.9, col: 'ds' },
        { v: 'ds32k', label: 'DeepSeek · prompt 32.768', P: 32768, ttft: 22.96, pre: 1428, dec: 64.0, col: 'ds' },
        { v: 'ds131k', label: 'DeepSeek · prompt 131.072', P: 131072, ttft: 78.75, pre: 1665, dec: 65.2, col: 'ds' },
        { v: 'glmS', label: 'GLM · structured ×1 (28/08)', P: null, ttft: 0.719, dec: 62.9, col: 'glm' },
        { v: 'qwP', label: 'Qwen · prosa ×1', P: null, ttft: 0.16, dec: 54.4, col: 'qwen' },
      ];
      const S = { sc: 'ds131k', N: 512, t0: 0, reset: true };
      const cur = () => SC.find((s) => s.v === S.sc);
      const calc = () => { const s = cur(); const dec = (S.N - 1) / s.dec; const e2e = s.ttft + dec; return { s, dec, e2e, wall: S.N / e2e, tpot: 1000 / s.dec }; };
      const ctl = UI.controls(f);
      const upd = () => { const r = calc(); rT.set(r.s.ttft < 1 ? nf(r.s.ttft * 1000) + ' ms' : nf(r.s.ttft, 2) + ' s'); rP.set(nf(r.tpot, 1) + ' ms'); rE.set(nf(r.e2e, 1) + ' s'); rW.set(nf(r.wall, 1) + ' tok/s'); S.reset = true; };
      UI.select(ctl, { label: 'Cenário', options: SC.map((s) => ({ v: s.v, label: s.label })), value: S.sc, onChange: (v) => { S.sc = v; upd(); } });
      UI.slider(ctl, { label: 'Tokens de saída N', min: 64, max: 2048, step: 64, value: 512, fmt: (v) => String(v), onInput: (v) => { S.N = v; upd(); } });
      const rT = UI.readout(ctl, 'TTFT', '—');
      const rP = UI.readout(ctl, 'TPOT', '—');
      const rE = UI.readout(ctl, 'E2EL', '—');
      const rW = UI.readout(ctl, 'tok/s de parede', '—');
      upd();
      V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 620;
        const pad = 16;
        if (S.reset) { S.t0 = t; S.reset = false; }
        const r = calc(), s = r.s;
        const rc = C[s.col] || C.a;
        // cabeçalho: leitura × escrita
        V.text(g, s.label, pad, 20, { size: 13, font: 'display', weight: 600, color: rc });
        const gy = 34, gw = w - 2 * pad;
        const pre = s.pre || null;
        const gmax = Math.max(pre || 0, s.dec) * 1.05;
        const gauge = (y, lbl, val, col, note) => {
          V.text(g, lbl, pad, y + 11, { size: 11.5, color: C.muted });
          const x0 = pad + (narrow ? 92 : 128), ww = gw - (narrow ? 92 : 128);
          V.box(g, x0, y, ww, 14, { r: 5, fill: V.alpha(C.muted, 0.12) });
          if (val != null) {
            V.box(g, x0, y, Math.max(4, ww * val / gmax), 14, { r: 5, fill: V.alpha(col, 0.75) });
            V.text(g, nf(val, val < 100 ? 1 : 0) + ' tok/s', x0 + Math.max(4, ww * val / gmax) + 6 > x0 + ww - 80 ? x0 + 6 : x0 + ww * val / gmax + 6, y + 11, { size: 11, font: 'mono', weight: 600, color: x0 + ww * val / gmax + 6 > x0 + ww - 80 ? onInk() : C.ink });
          } else V.text(g, note, x0 + 6, y + 11, { size: 11, color: C.muted });
        };
        gauge(gy, narrow ? 'leitura' : 'leitura (prefill)', pre, C.d, 'tamanho do prompt não publicado');
        gauge(gy + 22, narrow ? 'escrita' : 'escrita (decode)', s.dec, C.a, '');
        if (pre) { if (narrow) V.text(g, 'lê ' + nf(pre / s.dec, 1) + '× mais rápido do que escreve', pad, gy + 58, { size: 11.5, font: 'mono', color: C.ink }); else V.text(g, 'lê ' + nf(pre / s.dec, 1) + '× mais rápido do que escreve', w - pad, 20, { size: 11.5, font: 'mono', color: C.ink, align: 'right' }); }
        // linha do tempo
        const ty = narrow ? 142 : 128, tH = 26, x0 = pad, x1 = w - pad;
        const X = (sec) => x0 + (sec / r.e2e) * (x1 - x0);
        const tt = X(s.ttft);
        V.box(g, x0, ty, Math.max(2, tt - x0), tH, { r: 6, fill: V.alpha(C.d, 0.7) });
        V.box(g, tt, ty, Math.max(2, x1 - tt), tH, { r: 6, fill: V.alpha(C.a, 0.35), stroke: V.alpha(C.a, 0.8), lw: 1 });
        // marcas de tokens
        const nTicks = Math.min(S.N, Math.floor((x1 - tt) / 3));
        g.strokeStyle = V.alpha(C.a, 0.9); g.lineWidth = 1;
        for (let k = 0; k < nTicks; k++) { const xx = tt + ((k + 0.5) / nTicks) * (x1 - tt); g.beginPath(); g.moveTo(xx, ty + 5); g.lineTo(xx, ty + tH - 5); g.stroke(); }
        if (tt - x0 > 60) V.text(g, 'TTFT', x0 + 6, ty + tH / 2 + 0.5, { size: 11, font: 'mono', weight: 600, color: onInk(), base: 'middle' });
        // eixo
        const nice = (v) => { const p = Math.pow(10, Math.floor(Math.log10(v))); const m = v / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; };
        const step = nice(r.e2e / (narrow ? 4 : 7));
        for (let sx = 0; sx <= r.e2e + 1e-9; sx += step) { const xx = X(sx); V.line(g, xx, ty + tH + 2, xx, ty + tH + 7, { color: C.muted, lw: 1 }); V.text(g, nf(sx, step < 1 ? 1 : 0) + ' s', xx, ty + tH + 20, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        // cursor animado (tempo comprimido)
        const D = 7, pause = 1.2;
        const u = ((t - S.t0) % (D + pause)) / D;
        const cs = Math.min(1, u) * r.e2e;
        const cxp = X(cs);
        V.line(g, cxp, ty - 14, cxp, ty + tH + 4, { color: C.ink, lw: 1.6 });
        const emitted = cs < s.ttft ? 0 : Math.min(S.N, 1 + Math.floor((cs - s.ttft) * s.dec));
        guy(g, cxp, ty - 24, 8, cs < s.ttft ? C.d : C.a, { blink: blinkAt(t, 2), glow: cs >= s.ttft && u < 1 });
        const phase = cs < s.ttft ? 'lendo o prompt (prefill)…' : u >= 1 ? 'resposta completa' : 'escrevendo (decode)…';
        V.text(g, fit(g, (narrow ? '' : phase + ' · ') + 'tokens emitidos: ' + emitted + '/' + S.N, w - 2 * pad, 11.5, 'mono'), pad, ty + tH + 42, { size: 11.5, font: 'mono', color: cs < s.ttft ? C.d : C.a });
        V.text(g, '1 s de animação ≈ ' + nf(r.e2e / D, r.e2e / D < 10 ? 1 : 0) + ' s reais', w - pad, ty + tH + 42, { size: 11, font: 'mono', color: C.muted, align: 'right', alpha: narrow ? 0 : 1 });
        // números
        const yN = ty + tH + (narrow ? 74 : 72);
        const fracP = s.ttft / r.e2e;
        const L1 = 'E2EL = TTFT + (N−1)·TPOT = ' + (s.ttft < 1 ? nf(s.ttft, 2) : nf(s.ttft, 1)) + ' s + ' + (S.N - 1) + ' × ' + nf(r.tpot, 1) + ' ms = ' + nf(r.e2e, 1) + ' s';
        const L2 = 'prefill = ' + nf(100 * fracP, 0) + '% do relógio · tok/s de parede N/E2EL = ' + nf(r.wall, 1) + ' (decode: ' + nf(s.dec, 1) + ')';
        let yy = textBlock(g, L1, pad, yN, w - 2 * pad, { size: 12, font: 'mono', color: C.ink, lh: 17 });
        yy = textBlock(g, L2, pad, yy + 4, w - 2 * pad, { size: 12, font: 'mono', color: C.ink, lh: 17 });
        if (s.P && yy + 18 < h) textBlock(g, 'prefill tok/s = prompt_tokens ÷ TTFT = ' + nf(s.pre) + ' (medianas publicadas; prompt_tokens inclui template e instrução)', pad, yy + 6, w - 2 * pad, { size: 12, font: 'mono', color: C.d, lh: 17 });
      });
    }

    /* ===================== Figura 9: a carona dos pesos ===================== */
    {
      const f = $('#fig-carona');
      fitH(f, 440, 640, 700);
      const E = 512, KE = 10, POSN = 4, BMAX = 16, SEQS = 8;
      const R = V.rng(99);
      const picks = [];
      for (let k = 0; k < BMAX * POSN; k++) { const set = new Set(); while (set.size < KE) set.add(R.int(E)); picks.push([...set]); }
      const litOrder = new Array(E).fill(-1); // índice do primeiro token que acende o expert
      picks.forEach((p, k) => p.forEach((e) => { if (litOrder[e] < 0) litOrder[e] = k; }));
      const S1 = 54.4;
      const MEAS_A = [[1, 54.4], [2, 86.5], [4, 128.1], [6, 169.7], [8, 207.0]];
      const MEAS_S = [[1, 54.4], [2, 45.1], [4, 34.2], [6, 29.7], [8, 26.7]];
      const S = { B: 4, phi: 0.17, meas: true, since: 0, prevB: 4, t: 0 };
      const ctl = UI.controls(f);
      // modelo sem teto de vagas (hipotético) e com o teto: acima de 8 streams, o passo continua com 8
      const modelRaw = (B) => { const s = S1 / (1 + S.phi * (B - 1)); return { s, A: B * s }; };
      const model = (B) => modelRaw(Math.min(B, SEQS));
      const distinct = (B) => { let d = 0; const T = Math.min(B, SEQS) * POSN; for (let e = 0; e < E; e++) if (litOrder[e] >= 0 && litOrder[e] < T) d++; return d; };
      const upd = () => { const m = model(S.B); rD.set(distinct(S.B) + ' de 512'); rA.set(nf(m.A, 0) + ' tok/s'); rS.set(nf(m.s, 1) + ' tok/s'); rQ.set(String(Math.max(0, S.B - SEQS))); };
      UI.slider(ctl, { label: 'Streams B', min: 1, max: BMAX, step: 1, value: S.B, fmt: (v) => String(v), onInput: (v) => { S.prevB = S.B; S.B = v; S.since = S.t; upd(); } });
      UI.slider(ctl, { label: 'Custo/stream', min: 0.05, max: 0.4, step: 0.01, value: S.phi, fmt: (v) => 'φ = ' + nf(v, 2), onInput: (v) => { S.phi = v; upd(); } });
      UI.toggle(ctl, { label: 'medições do Qwen', value: true, onChange: (v) => { S.meas = v; } });
      const rD = UI.readout(ctl, 'experts distintos/camada', '—');
      const rA = UI.readout(ctl, 'agregado (modelo)', '—');
      const rS = UI.readout(ctl, 'por stream admitido', '—');
      const rQ = UI.readout(ctl, 'na fila', '—');
      upd();
      V.fig(f, (g, w, h, t) => {
        S.t = t;
        const C = V.C();
        const wide = w >= 700;
        const pad = 16;
        const Badm = Math.min(S.B, SEQS);
        // ---------- painel A: grade de experts ----------
        const gw = wide ? Math.min(w * 0.44, 380) : w - 2 * pad;
        const cols = 32, rows = 16;
        const cs = Math.floor((gw - 4) / cols);
        const gx = pad, gy = 58;
        V.text(g, 'experts de uma camada do Qwen (512, top-10)', gx, 18, { size: 11.5, font: 'mono', color: C.muted });
        // streams (personagens): os da fila ficam apagados e não acendem experts
        const gsp = Math.min(22, (gw - 10) / BMAX);
        for (let b = 0; b < S.B; b++) guy(g, gx + 8 + b * gsp, 38, Math.min(8, gsp * 0.42), b < SEQS ? C.series[b % C.series.length] : C.muted, { blink: blinkAt(t, b), alpha: b < SEQS ? 1 : 0.4 });
        const T = Badm * POSN;
        const age = t - S.since;
        for (let rI = 0; rI < rows; rI++) for (let cI = 0; cI < cols; cI++) {
          const e = rI * cols + cI;
          const k = litOrder[e];
          const on = k >= 0 && k < T;
          const x = gx + cI * cs, y = gy + rI * cs;
          V.rr(g, x, y, cs - 1.5, cs - 1.5, 2);
          if (on) {
            const b = Math.floor(k / POSN);
            const fresh = S.B > S.prevB && k >= Math.min(S.prevB, SEQS) * POSN ? clamp(age / 0.6, 0, 1) : 1;
            g.fillStyle = V.alpha(C.series[b % C.series.length], 0.35 + 0.5 * fresh); g.fill();
          } else { g.fillStyle = V.alpha(C.muted, 0.12); g.fill(); }
        }
        const D = distinct(S.B);
        const ED = E * (1 - Math.pow(1 - KE / E, T));
        const gyE = gy + rows * cs + 18;
        const nrw = gw < 420;
        const qTxt = S.B > SEQS ? ' (' + (S.B - SEQS) + ' na fila)' : '';
        V.text(g, fit(g, (nrw ? 'T = ' + Badm + ' × 4 = ' + T + ' tokens' : 'T = ' + Badm + ' streams × 4 posições = ' + T + ' tokens no passo') + qTxt, gw, 11.5, 'mono', 600), gx, gyE, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        V.text(g, fit(g, nrw ? 'distintos: ' + D + '/512 (fórmula ' + nf(ED, 0) + ') · ' + nf(100 * D / E, 0) + '%' : 'distintos: ' + D + ' de 512 (fórmula: ' + nf(ED, 0) + ') → ' + nf(100 * D / E, 0) + '% lidos', gw, 11.5, 'mono'), gx, gyE + 18, { size: 11.5, font: 'mono', color: C.muted });
        // ---------- painel B: curvas ----------
        const cx0 = wide ? gx + gw + 44 : pad + 34, cx1 = w - pad - 6;
        const cy0 = wide ? 30 : gyE + 54, cy1 = h - 40;
        const ymax = 340;
        const X = (B) => cx0 + ((B - 1) / (BMAX - 1)) * (cx1 - cx0);
        const Y = (v) => cy1 - (v / ymax) * (cy1 - cy0);
        // fila
        g.fillStyle = V.alpha(C.red, 0.07); g.fillRect(X(SEQS), cy0, cx1 - X(SEQS), cy1 - cy0);
        V.text(g, fit(g, 'acima de max_num_seqs = 8: fila', cx1 - X(SEQS) - 8, 11, 'mono'), X(SEQS) + 6, Y(120), { size: 11, font: 'mono', color: C.red });
        [0, 100, 200, 300].forEach((v) => { V.line(g, cx0, Y(v), cx1, Y(v), { color: C.grid, lw: 1 }); V.text(g, String(v), cx0 - 6, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        [1, 4, 8, 12, 16].forEach((B) => V.text(g, String(B), X(B), cy1 + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'streams simultâneos B', (cx0 + cx1) / 2, cy1 + 32, { size: 11, color: C.muted, align: 'center' });
        V.text(g, 'tok/s', cx0 - 6, cy0 - 8, { size: 11, color: C.muted, align: 'right' });
        // assíntota (sem teto de vagas)
        const Ainf = S1 / S.phi;
        if (Ainf < ymax) { V.line(g, cx0, Y(Ainf), cx1, Y(Ainf), { color: V.alpha(C.a, 0.4), lw: 1, dash: [5, 4] }); V.text(g, 'limite τ/T₁ ≈ ' + nf(Ainf, 0), cx1 - 4, Y(Ainf) - 6, { size: 11, font: 'mono', color: C.a, align: 'right' }); }
        const curve = (fn, col, b0, b1, o = {}) => { g.beginPath(); for (let B = b0; B <= b1 + 1e-9; B += 0.25) { const v = fn(B); const x = X(B), y = Y(Math.min(v, ymax)); if (B === b0) g.moveTo(x, y); else g.lineTo(x, y); } g.strokeStyle = o.alpha ? V.alpha(col, o.alpha) : col; g.lineWidth = o.lw || 2.4; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]); };
        // hipotético sem teto de vagas (tracejado, apagado)
        curve((B) => modelRaw(B).A, C.a, SEQS, BMAX, { dash: [5, 5], alpha: 0.4, lw: 1.6 });
        curve((B) => modelRaw(B).s, C.b, SEQS, BMAX, { dash: [5, 5], alpha: 0.4, lw: 1.6 });
        // com o teto: sobe até 8 e fica plano (vagas cheias)
        curve((B) => model(B).A, C.a, 1, BMAX);
        curve((B) => model(B).s, C.b, 1, BMAX);
        V.text(g, 'agregado', cx1 - 4, Y(Math.min(model(SEQS).A, ymax)) - 8, { size: 11.5, weight: 600, color: C.a, align: 'right' });
        V.text(g, 'por stream', cx1 - 4, Y(model(SEQS).s) - 8, { size: 11.5, weight: 600, color: C.b, align: 'right' });
        V.text(g, fit(g, 'vagas cheias: plano', cx1 - X(SEQS) - 10, 11, 'mono'), cx1 - 4, Y(model(SEQS).A) + 16, { size: 11, font: 'mono', color: C.a, align: 'right' });
        const yh = Y(Math.min(modelRaw(BMAX).A, ymax));
        V.text(g, fit(g, 'sem teto (hipotético)', cx1 - X(SEQS) - 10, 11, 'mono'), cx1 - 4, yh - 8, { size: 11, font: 'mono', color: V.alpha(C.a, 0.75), align: 'right' });
        if (S.meas) {
          const pts = (arr, col) => arr.forEach(([B, v]) => { g.beginPath(); g.arc(X(B), Y(v), 4.5, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = 2; g.stroke(); g.fillStyle = C.bg; g.fill(); g.stroke(); });
          pts(MEAS_A, C.a); pts(MEAS_S, C.b);
          V.text(g, '○ medido (Qwen, prosa)', cx0 + 6, cy0 + 12, { size: 11, font: 'mono', color: C.muted });
        }
        // marcador atual
        const m = model(S.B);
        V.line(g, X(S.B), cy0, X(S.B), cy1, { color: V.alpha(C.ink, 0.45), lw: 1.2, dash: [3, 3] });
        V.glow(g, X(S.B), Y(Math.min(m.A, ymax)), 14, C.a, 0.8); V.dot(g, X(S.B), Y(Math.min(m.A, ymax)), 5, C.a);
        V.glow(g, X(S.B), Y(m.s), 14, C.b, 0.8); V.dot(g, X(S.B), Y(m.s), 5, C.b);
      });
    }

    /* ===================== Figura 10: mapa de metrô ===================== */
    {
      const f = $('#fig-metro');
      fitH(f, 440, 640, 700);
      const LINES = [
        { id: 'pre', name: 'PREFILL', sub: 'M grande · cálculo', colK: 'd' },
        { id: 'dec', name: 'DECODE', sub: 'M pequeno · banda', colK: 'a' },
        { id: 'kv', name: 'KV', sub: 'capacidade', colK: 'b' },
      ];
      const ST = [
        { l: 'pre', r: 'glm', s: 'E3 + 7.168', t: 'Experts gordos (E3) e pedaços de 7.168 tokens', d: 'Com 1.024 tokens por pedaço, ~28 linhas por expert (~112 FLOP/B, abaixo do ridge de 458). Com 7.168, ~199 linhas: I ≈ 460–500 contando as ativações, em cima do ridge (2r/b_w daria ~800). Os kernels E3 levaram a camada MoE de 77–91 para 31 ms e o prefill frio de 947 para 1.428–1.587 tok/s.' },
        { l: 'pre', r: 'glm', s: 'KDA BF16', t: 'GLM53_KDA_BF16_LARGE_M=1', d: 'Cópia BF16 do in_proj do KDA para M > 512: cuBLAS em 7,36 ms contra 27,21 ms do Marlin com M = 7.168. TTFT −11,4% (16k) e −12,3% (100k), por +3,29 GiB por rank.' },
        { l: 'pre', r: 'qwen', s: 'plano 2,96k', t: 'Prefill de ~2,96 mil tok/s', d: 'Lane day-0: 2.962 tok/s de 16k a 64k e −8% em 128k; a atenção QSA evita que o custo quadrático domine. Um pedaço de 8.192 toca todos os experts: ~34 GB por GPU, ~140 ms de streaming (estimativa do relatório: 48 × 512 × 2,77 MB ÷ 2 GPUs, a ~240 GB/s).' },
        { l: 'pre', r: 'ds', s: 'indexer SP', t: 'DSPARK_ENABLE_SP_INDEXER=1', d: 'Cada rank pontua só uma fatia das chaves comprimidas, e o top-k é combinado de forma exata. TTFT de 128K −4 a −6%. A 900K, o indexer custa ≈80 TFLOP por pedaço de 1.024, duplicado nas duas GPUs.' },
        { l: 'pre', r: 'ds', s: 'W4A16', t: 'Pesos de 4 bits, conta em BF16 (b12x)', d: 'O caminho do checkpoint desquantiza para BF16 antes da multiplicação: ótimo para o decode (banda), preso ao pico BF16 no prefill. Tensor cores FP4×FP8 teriam ~4× o pico denso (estimativa do relatório).' },
        { l: 'dec', r: 'ds', s: 'c=1 ≈ 80%', t: 'Decode com um usuário: ~80% streaming de pesos', d: '≈16,5–17 GB lidos por rank por passo a 230–250 GB/s = 68–74 ms de streaming, num passo que ainda paga ~10–15 ms fixos (≈108 coletivas, 3–5 ms, mais lançamentos): ~80% é streaming. Mediana medida de 56 tok/s em 02/09 (51–67).' },
        { l: 'dec', r: 'glm', s: 'FP8 denso', t: 'GLM53_DENSE_FP8', d: 'KDA e projeções densas em FP8 weight-only: ~2,6 GB a menos por passo, “~11 ms less per step”. O modo all rendeu mais −4,7% de ms por ciclo sobre dense,kda (26/09).' },
        { l: 'dec', r: 'glm', s: 'MoE coop.', t: 'Kernel MoE cooperativo de decode', d: 'Especializado para 1–32 linhas (H = 4096, top-8, 4 bits): structured ×1 de 72,03 para 77,29 tok/s (+7,3%); ×2 agregado +9,3%.' },
        { l: 'dec', r: 'glm', s: 'thin-decode', t: 'GLM53_EXL3_MOE_FAST=1 (thin-decode K4/N256)', d: 'Kernel MoE de decode para o SM121 com 1 estágio de registradores e 8 de memória compartilhada (stock: 3 e 3): somem os spills de registrador e a Hadamard repetida da entrada do up. Mesmos bytes, passo mais curto: +7,7% (structured), +8,9% (código), +14,6% (JSON) com 1 stream. Opt-in, só TP2.' },
        { l: 'dec', r: 'glm', s: 'adaptive-k', t: 'GLM53_ADAPTIVE_K=ema', d: 'Verifica só 2, 4 ou 7 dos 7 rascunhos do DFlash2, conforme uma média móvel (EMA) da aceitação. Em prosa, onde a aceitação cai, verificar menos posições acende menos experts e encurta o passo: +13–21%. O k é o mínimo do lote, porque os CUDA graphs FULL exigem forma uniforme.' },
        { l: 'dec', r: 'qwen', s: 'vocab draft', t: 'Vocabulário reduzido do rascunhador', d: 'O lm_head lido 4× por passo é 26% dos bytes. Com 47k tokens no rascunho: +15,3% (código) e +15,5% (prosa) com 1 stream; +4,0% e +5,6% com 8.' },
        { l: 'dec', r: 'all', s: 'especulação', t: 'MTP · DFlash2 · DSpark', d: 'Vários tokens por leitura dos pesos. Qwen MTP=3: 24,5 → 52,1 tok/s (2,13×), τ ≈ 3,2 (2,2 aceitos + 1 bônus). GLM DFlash2 k=7: 6,71 aceitos por passo em texto estruturado (τ ≈ 7,7). DeepSeek DSpark: τ ≈ 4,15 (3,15 aceitos + 1 bônus).' },
        { l: 'kv', r: 'glm', s: 'fp8_ds_mla', t: '--kv-cache-dtype fp8 → fp8_ds_mla', d: '656 B por token por camada MLA (512 FP8 + 16 de escalas + 128 de RoPE zerado) + 33 B do indexador, em 11 camadas: 7.579 B por token por rank, mais ~76 MiB de estado KDA por requisição.' },
        { l: 'kv', r: 'glm', s: 'pool 11 GiB', t: '--kv-cache-memory-bytes 11811160064', d: 'Fixa o KV em 11 GiB por GPU: 1.233.779 tokens, 1,45× a 850 mil, sem depender do que o profile run mediu.' },
        { l: 'kv', r: 'qwen', s: 'KV FP8', t: '--kv-cache-dtype fp8', d: '13.312 B por token (6.656 por GPU) em vez de 26.624: pool de 2.131.159 → 3.652.200 tokens (+71%), 13,93× a 262k. Exigiu um patch no QSA da imagem day-0.' },
        { l: 'kv', r: 'qwen', s: 'bytes/token', t: 'KV_BYTES_PER_TOKEN=29482 (TP1)', d: 'Bytes por token medidos em BF16 num Spark só. O launcher converte MAX_MODEL_LEN em GiB de KV com esse valor antes de escolher a fração de memória.' },
        { l: 'kv', r: 'ds', s: 'nvfp4_ds_mla', t: '--kv-cache-dtype nvfp4_ds_mla', d: 'Linha de 584 B (448 FP8 + 128 de RoPE BF16 + 8 de escalas), byte a byte igual ao fp8_ds_mla nesta imagem. Comprimida 1/4 e 1/128: ≈3.850 B por token; pool de 2.331.430 tokens.' },
        { l: 'x', r: 'glm', s: 'dispatch por M', t: 'KDA_BF16_LARGE_M_MIN_M = 512', d: 'O número de linhas decide o kernel. Nos passos medidos, o decode nunca passou de M = 220 e o prefill ficou em M ≥ 768: um limiar fixo em 512 separa as fases sem sincronizar com a CPU.' },
      ];
      const S = { filt: 'all', sel: ST.length - 1, auto: true, lastSwitch: 0 };
      const ctl = UI.controls(f);
      const visible = (s) => S.filt === 'all' || s.r === S.filt || s.r === 'all';
      const nextSel = () => { for (let k = 1; k <= ST.length; k++) { const j = (S.sel + k) % ST.length; if (visible(ST[j])) { S.sel = j; return; } } };
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'all', label: 'todas' }, { v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'all', onChange: (v) => { S.filt = v; if (!visible(ST[S.sel])) nextSel(); } });
      const bt = UI.button(ctl, '⏸ passeio', () => { S.auto = !S.auto; bt.innerHTML = S.auto ? '⏸ passeio' : '▶ passeio'; });
      const H = V.fig(f, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 700;
        const pad = 16;
        const cardH = wide ? 124 : 204;
        const mapH = h - cardH - 16;
        const rc = (r) => (r === 'all' ? C.d : C[r]);
        // ---------- geometria das linhas ----------
        const pos = {}; const paths = {};
        const lineStations = (id) => ST.map((s, k) => ({ s, k })).filter((o) => o.s.l === id);
        if (wide) {
          const yP = 56, yD = mapH * 0.52, yK = mapH - 26;
          const fx = pad + 58, fy = (yP + yD) / 2;
          const xs0 = fx + 110, xs1 = w - pad - 60;
          pos.x = [fx, fy];
          const place = (id, y) => { const L = lineStations(id); L.forEach((o, j) => { pos[o.k] = [xs0 + (j + 0.5) * (xs1 - xs0) / L.length, y]; }); };
          place('pre', yP); place('dec', yD); place('kv', yK);
          paths.pre = [[fx, fy], [fx + 40, yP], [xs1 + 30, yP]];
          paths.dec = [[fx, fy], [fx + 40, yD], [xs1 + 30, yD]];
          paths.kv = [[pad + 20, yK], [xs1 + 30, yK]];
        } else {
          const xP = w * 0.18, xD = w * 0.5, xK = w * 0.82;
          const fy = 40, fx = (xP + xD) / 2;
          const ys0 = 96, ys1 = mapH - 10;
          pos.x = [fx, fy];
          const place = (id, x) => { const L = lineStations(id); L.forEach((o, j) => { pos[o.k] = [x, ys0 + (j + 0.5) * (ys1 - ys0) / L.length]; }); };
          place('pre', xP); place('dec', xD); place('kv', xK);
          paths.pre = [[fx, fy], [xP, fy + 36], [xP, ys1 + 6]];
          paths.dec = [[fx, fy], [xD, fy + 36], [xD, ys1 + 6]];
          paths.kv = [[xK, fy + 10], [xK, ys1 + 6]];
        }
        const drawPath = (pts, col) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let k = 1; k < pts.length; k++) g.lineTo(pts[k][0], pts[k][1]); g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 7; g.lineJoin = 'round'; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt'; };
        const alongPath = (pts, u) => { const segs = []; let L = 0; for (let k = 1; k < pts.length; k++) { const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); segs.push(d); L += d; } let d = u * L; for (let k = 0; k < segs.length; k++) { if (d <= segs[k]) { const q = d / segs[k]; return [lerp(pts[k][0], pts[k + 1][0], q), lerp(pts[k][1], pts[k + 1][1], q)]; } d -= segs[k]; } return pts[pts.length - 1]; };
        LINES.forEach((L, k) => {
          const col = C[L.colK];
          drawPath(paths[L.id], col);
          // trem
          const u = (t * 0.06 + k * 0.33) % 1;
          const p = alongPath(paths[L.id], u);
          V.box(g, p[0] - 9, p[1] - 5, 18, 10, { r: 4, fill: C.ink });
          V.dot(g, p[0] - 4, p[1], 1.6, col); V.dot(g, p[0] + 4, p[1], 1.6, col);
          // nome da linha
          const lp = L.id === 'kv' ? paths.kv[0] : (wide ? [paths[L.id][1][0] + 4, paths[L.id][1][1]] : paths[L.id][1]);
          if (wide) { const ly = L.id === 'pre' ? lp[1] + 24 : lp[1] - 12; V.text(g, L.name, lp[0] + 8, ly, { size: 12, font: 'display', weight: 700, color: col }); V.text(g, L.sub, lp[0] + 8 + tw(g, L.name, 12, 'display', 700) + 8, ly, { size: 11, color: C.muted }); }
          else V.text(g, L.name, lp[0], lp[1] - 12 + (L.id === 'kv' ? -2 : 0), { size: 11.5, font: 'display', weight: 700, color: col, align: 'center' });
        });
        // ---------- auto ----------
        if (S.auto && t - S.lastSwitch > 4.5) { S.lastSwitch = t; nextSel(); }
        // ---------- clique ----------
        const P = H.pointer;
        let hover = -1;
        const all = ST.map((s, k) => ({ s, k, p: s.l === 'x' ? pos.x : pos[k] }));
        if (P.inside) all.forEach((o) => { if (o.p && Math.hypot(P.x - o.p[0], P.y - o.p[1]) < 18) hover = o.k; });
        if (P.clicked && hover >= 0) { S.sel = hover; S.auto = false; bt.innerHTML = '▶ passeio'; }
        const tagText = (txt, x, y, o2) => { const tw2 = tw(g, txt, o2.size, 'mono', o2.weight); const x0 = o2.align === 'center' ? x - tw2 / 2 : x; V.box(g, x0 - 4, y - o2.size + 1, tw2 + 8, o2.size + 5, { r: 5, fill: V.alpha(C.bg, 0.86) }); V.text(g, txt, x, y, Object.assign({ font: 'mono' }, o2)); };
        // ---------- estações ----------
        all.forEach((o) => {
          if (!o.p) return;
          const vis = visible(o.s);
          const sel = o.k === S.sel;
          const x = o.p[0], y = o.p[1];
          const isX = o.s.l === 'x';
          g.save(); g.globalAlpha = vis ? 1 : 0.22;
          if (sel) V.glow(g, x, y, 26 + 3 * Math.sin(t * 4), rc(o.s.r), 0.7);
          g.beginPath(); g.arc(x, y, isX ? 13 : (sel || o.k === hover ? 10 : 8), 0, Math.PI * 2);
          g.fillStyle = C.bg; g.fill(); g.lineWidth = isX ? 4 : 3.2; g.strokeStyle = rc(o.s.r); g.stroke();
          if (isX) V.text(g, 'M', x, y + 0.5, { size: 12, font: 'display', weight: 700, color: C.ink, align: 'center', base: 'middle' });
          else V.dot(g, x, y, 3, rc(o.s.r));
          // rótulo
          const lbl = o.s.s;
          if (wide) {
            const below = o.s.l !== 'pre';
            const decIdx = o.s.l === 'dec' ? lineStations('dec').findIndex((q) => q.k === o.k) : 0;
            const dy = o.s.l === 'dec' && decIdx % 2 === 1 ? 17 : 0;
            if (isX) { V.text(g, 'dispatch', x, y + 30, { size: 11.5, font: 'mono', weight: sel ? 700 : 500, color: C.ink, align: 'center' }); V.text(g, 'por M', x, y + 44, { size: 11.5, font: 'mono', weight: sel ? 700 : 500, color: C.ink, align: 'center' }); }
            else tagText(lbl, x, below ? y + 24 + dy : y - 16, { size: 11.5, weight: sel ? 700 : 500, color: sel ? C.ink : C.text2 || C.ink, align: 'center' });
          } else {
            const yy = isX ? y + 0.5 : y + 22;
            if (isX) V.text(g, lbl, x + 18, yy, { size: 11, font: 'mono', weight: sel ? 700 : 500, color: C.ink, base: 'middle' });
            else tagText(fit(g, lbl, w * 0.3, 11, 'mono'), x, yy, { size: 11, weight: sel ? 700 : 500, color: sel ? C.ink : C.text2 || C.ink, align: 'center' });
          }
          g.restore();
        });
        // ---------- cartão ----------
        const s = ST[S.sel];
        const cy = h - cardH - 6;
        const col = rc(s.r);
        V.box(g, pad, cy, w - 2 * pad, cardH, { r: 12, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.6), lw: 1.4 });
        const tag = s.r === 'all' ? 'as três' : s.r === 'glm' ? 'GLM' : s.r === 'qwen' ? 'Qwen' : 'DeepSeek';
        const lineName = s.l === 'x' ? 'baldeação' : LINES.find((L) => L.id === s.l).name.toLowerCase();
        V.text(g, fit(g, tag + ' · linha ' + lineName, w - 4 * pad, 11, 'mono', 600), pad + 14, cy + 20, { size: 11, font: 'mono', weight: 600, color: col });
        V.text(g, fit(g, s.t, w - 4 * pad, 13.5, 'mono', 700), pad + 14, cy + 40, { size: 13.5, font: 'mono', weight: 700, color: C.ink });
        textBlock(g, s.d, pad + 14, cy + 60, w - 2 * pad - 28, { size: 12.5, color: C.ink, lh: 17 });
      }, { static: false });
    }
  },
});

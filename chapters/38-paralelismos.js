LA.chapter({
  id: 'paralelismos',
  part: 8,
  order: 38,
  title: 'Dividindo o modelo: TP, EP, PP, DP e DCP',
  navTitle: 'Paralelismos',
  subtitle: 'Cortar matrizes, especialistas, camadas, requisições e tokens entre dois nós, e prever quanto cada corte cobra na rede a cada token.',
  minutes: 65,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['tensor parallel', 'column-parallel', 'row-parallel', 'all-reduce', 'expert parallel', 'pipeline parallel', 'data parallel', 'DCP', 'sequence parallel', 'top-k distribuído', 'α-β'],
  goals: [
    'Explicar <b>column-parallel</b>, <b>row-parallel</b> e por que o par col → row custa um único all-reduce',
    'Mapear, num bloco real, o que é <b>fatiado</b>, o que é <b>replicado</b> e onde moram as ~100 coletivas por passo de cada receita',
    'Decidir entre <b>TP</b> e <b>EP</b> para os experts, olhando blocos de quantização, kernels e desequilíbrio de carga',
    'Explicar por que PP e DP não servem às receitas e o que <b>DCP</b> e <b>SP</b> mudam no KV e no prefill',
    'Estimar o custo de rede de uma topologia com o modelo α-β e saber quando replicar deixa de compensar',
  ],
  prereqs: ['prefill-decode-kv', 'mixture-of-experts', 'mla', 'atencao-esparsa', 'fp8-fp4-microescala', 'hadamard-trelica-exl3', 'paged-attention-kv', 'especulacao-fundamentos'],
  summary: [
    'TP corta matrizes: column-parallel fatia saídas sem comunicar, row-parallel fatia entradas e soma por all-reduce. O par col → f → row custa <b>um</b> all-reduce, e cada bloco tem dois (atenção e FFN/MoE). Na receita GLM·TF, o TensorFold fecha o par com all-gather de parciais fp32 e soma rank 0 primeiro: um arredondamento, com a soma no código do motor e bits que não dependem do transporte, ao preço do dobro de bytes no fio (e, na receita v1.3.2, de ficar em dois ranks; a v1.4 acrescentou um modo experimental de três ranks que divide de forma desigual em vez de encher).',
    'Sob TP=2 as receitas fazem ~106 (Qwen) e ~108 (DeepSeek: 43 + 43, mais rascunho, embedding, logits e Markov) coletivas pequenas por passo, a 20–50 µs cada: um piso de 2–5 ms que não cai com o batch.',
    'O corte precisa respeitar a estrutura quantizada: blocos de escala e tiles ao longo do eixo cortado, com <code>suh</code>/<code>svh</code> seguindo o eixo. Quando não dá, a saída é EP (Qwen: 640/2 = 320 não é múltiplo de 128; GLM TP3: treliça de 2048) ou réplica (ViT do Qwen: 2152 não é múltiplo de 16).',
    'EP distribui experts inteiros. Com DP=1 os tokens já estão replicados e a coletiva continua sendo o all-reduce do TP (o <code>--all2all-backend</code> do Qwen é inerte). O preço é o desequilíbrio em batch pequeno. O DeepSeek não usa EP porque o <code>B12xExperts</code> não suporta.',
    'PP não soma as bandas num token (a latência fica a de um nó) e DP exige o modelo inteiro por nó: nenhuma receita usa. <code>--mm-encoder-tp-mode data</code> é o DP local da torre de visão.',
    'Com MLA/MQA o KV fica replicado sob TP. O DCP dividiria os tokens com junção exata por log-sum-exp (não está ligado no DSV4). O SP indexer divide as chaves e junta os top-k locais de forma exata (−4 a −6% de TTFT a 128K). Na receita GLM·TF, a cola das hyper-connections do prefill é dividida por linhas entre os ranks (prefill de 50k de ~1.270 para ~1.730 tok/s, junto com outros dois patches). Dividir o KV por dono da conversa é a alternativa mais grossa (patch 0129 da GLM·TF: 6.204 → 3.300 B por token por Spark, 1,87× o pool).',
    'No decode as mensagens ficam à esquerda do joelho \\(\\alpha\\beta\\) (regime de latência); no prefill, à direita (regime de banda). Replicar para poupar coletivas precisa ser medido: a cabeça de Markov replicada perdeu (c=6: 122 contra 148 tok/s) e o draft TP=2 do GLM foi mantido (ganho claro no texto estruturado, ruído na prosa).',
  ],
  quiz: [
    { q: 'Por que a ordem column → row custa um all-reduce por par, e a ordem row → col custaria dois?', o: ['Porque column-parallel é mais rápido na GPU', 'Porque o NCCL só implementa all-reduce depois de uma camada row-parallel', 'Porque \\(f\\) é elemento a elemento: a fatia por colunas passa por \\(f\\) localmente, mas \\(f(P_0)+f(P_1) \\neq f(P_0+P_1)\\), então com row primeiro seria preciso somar antes de \\(f\\)', 'Porque row-parallel duplica os pesos em cada rank'], a: 2, e: 'Cada coluna de \\(XW\\) depende só da coluna de \\(W\\), então a fatia column-parallel pode passar por \\(f\\) sem conversar. Uma saída row-parallel é soma parcial, e \\(f\\) não distribui sobre a soma (ReLU de −2 + 3 não é ReLU(−2) + ReLU(3)). Somar antes de \\(f\\) e de novo no fim dá duas coletivas.' },
    { q: 'No Qwen, com <code>--enable-expert-parallel</code> e DP=1, o que o <code>--all2all-backend allgather_reducescatter</code> faz a cada camada MoE?', o: ['Despacha cada token para a GPU dona dos seus experts', 'Nada: os tokens já estão replicados pelo TP, cada rank roda seus 256 experts locais e o all-reduce do TP soma as contribuições', 'Substitui o all-reduce do TP por um all-to-all mais barato', 'Replica os 512 experts nas duas GPUs'], a: 1, e: 'A auditoria da receita mostra que <code>use_all2all_kernels</code> é falso com DP=1 e PCP=1. Sem réplicas de dados, não há tokens diferentes para despachar. O EP muda só quais pesos cada GPU guarda (256 de 512), e a comunicação continua sendo o all-reduce normal.' },
    { q: 'Por que o KV do DeepSeek-V4 não diminui por rank ao passar de 1 para 2 GPUs em TP?', o: ['Porque o KV está em FP8', 'Porque o pipeline parallel está desligado', 'Porque o vLLM não pagina o KV sob TP', 'Porque há uma única cabeça latente (MQA/MLA): todas as cabeças de query de todos os ranks leem o mesmo vetor, então ele fica replicado'], a: 3, e: 'O TP divide cabeças de query, e cada rank precisa do K/V que suas cabeças consultam. Com uma cabeça de KV só, todos precisam da linha inteira. O segundo nó ajuda porque libera memória de pesos (79,17 GiB por rank). O DCP dividiria os tokens entre os ranks, mas não está ligado para o DSV4 nesta imagem.' },
    { q: 'Por que PP=2 em vez de TP=2 pioraria o decode de um único usuário?', o: ['Porque no PP os barramentos se revezam no mesmo token: cada estágio lê metade dos pesos, um depois do outro. No TP os dois leem ao mesmo tempo', 'Porque o PP manda mais bytes pela rede', 'Porque o PP exige expert parallel', 'Porque o PP não suporta KV cache'], a: 0, e: 'O PP comunica pouco (uma ativação por fronteira), mas com um microlote um estágio fica ocioso enquanto o outro trabalha. A latência por token fica \\(\\approx T\\), a de um nó. O TP dá \\(\\approx T/2\\) + o custo das coletivas. É por isso que o tutorial de receita nova diz que o PP dobra a latência de decode em batch pequeno.' },
    { q: 'O SP indexer do DeepSeek divide as chaves entre os ranks e junta os top-512 locais. Por que o resultado é exato?', o: ['Se \\(x\\) está no top-\\(k\\) global, no máximo \\(k-1\\) elementos da linha o superam; logo no máximo \\(k-1\\) da sua fatia, e \\(x\\) está no top-\\(k\\) local do seu dono', 'Porque cada rank acaba vendo todas as chaves', 'Porque o erro é pequeno e o RULER-lite passou', 'Porque os scores são calculados em FP32'], a: 0, e: 'A união dos top-\\(k\\) locais contém o top-\\(k\\) global, e o top-\\(k\\) da união é o global. Com empates, a ordem precisa ser estável (por id), daí o seletor <code>stable_topk…</code> do DCP. Os testes contra força bruta confirmam, mas a exatidão vem da prova.' },
    { q: 'Um all-reduce de decode do Qwen tem ~20 KB. Com \\(\\alpha \\approx 25\\) µs e \\(\\beta \\approx 25\\) GB/s, o que domina o tempo?', o: ['A banda: 20 KB ÷ 25 GB/s', 'A computação do all-reduce na GPU', 'A latência: \\(\\alpha \\approx 25\\) µs, contra ~0,8 µs de transferência; por isso ~100 coletivas custam 2–3 ms por passo e o custo não cai com o batch', 'O page cache do Linux'], a: 2, e: '20 × 10³ B ÷ 25 × 10⁹ B/s = 0,8 µs, muito menor que \\(\\alpha\\). O joelho \\(\\alpha\\beta\\) fica em ~625 KB, e a mensagem de decode está bem à esquerda dele. Mais sequências no lote aumentam \\(n\\), mas continuam longe do joelho.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ================================================================ helpers */
    const RC = (C, r) => [C.a, C.b, C.d, C.c][r % 4];
    const nf = (x, d = 0) => Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const FC = {};
    const fontOf = (size, font, weight) => { const k = size + '|' + font + '|' + weight; return FC[k] || (FC[k] = V.font(size, font, weight)); };
    const tw = (g, s, size = 12, font = 'sans', weight = 400) => { g.font = fontOf(size, font, weight); return g.measureText(String(s)).width; };
    function wrapLines(g, s, maxW, size, font, weight) {
      const out = [];
      String(s).split('\n').forEach((par) => {
        const words = par.split(' ');
        let line = '';
        words.forEach((wd) => {
          const test = line ? line + ' ' + wd : wd;
          if (line && tw(g, test, size, font, weight) > maxW) { out.push(line); line = wd; } else line = test;
        });
        out.push(line);
      });
      return out;
    }
    function wrap(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.38);
      const lines = wrapLines(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      const n = o.maxLines ? Math.min(o.maxLines, lines.length) : lines.length;
      for (let i = 0; i < n; i++) V.text(g, lines[i], x, y + i * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align, alpha: o.alpha });
      return y + n * lh;
    }
    function chip(g, s, x, y, col, o = {}) {
      const size = o.size || 11, font = o.font || 'mono', weight = o.weight || 600, hh = o.h || 20;
      let wd = tw(g, s, size, font, weight) + 14;
      if (o.maxW && wd > o.maxW) wd = o.maxW;
      const X = o.align === 'right' ? x - wd : o.align === 'center' ? x - wd / 2 : x;
      V.box(g, X, y - hh / 2, wd, hh, { r: hh / 2, fill: V.alpha(col, o.fa == null ? 0.16 : o.fa), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, X + wd / 2, y + 0.5, { size, font, weight, color: o.color || V.C().ink, align: 'center', base: 'middle', maxW: wd - 10 });
      return wd;
    }
    function hatch(g, x, y, w, h, col, step = 6, off = 0, lw = 1.1) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
      const o = ((off % step) + step) % step;
      for (let k = -h - step + o; k < w + step; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    function plus(g, x, y, r, col, on) {
      if (on) V.glow(g, x, y, r * 2.6, col, 0.8);
      V.box(g, x - r, y - r, 2 * r, 2 * r, { r, fill: V.alpha(col, on ? 0.9 : 0.25), stroke: col, lw: 1.5 });
      const k = r * 0.55, C = V.C();
      V.line(g, x - k, y, x + k, y, { color: on ? (C.dark ? '#1a0b1a' : '#fff') : col, lw: 2 });
      V.line(g, x, y - k, x, y + k, { color: on ? (C.dark ? '#1a0b1a' : '#fff') : col, lw: 2 });
    }
    function adaptH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      if (!st) return;
      const apply = () => { const w = st.clientWidth || fig.clientWidth || 800; const hh = Math.round(fn(w)); if (Math.abs(st.offsetHeight - hh) > 1) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply);
      ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const gib = (b) => nf(b / 1073741824, b < 10 * 1073741824 ? 2 : 1) + ' GiB';
    const smooth = (x) => V.smooth(V.clamp(x, 0, 1));
    const MINUS = '−';
    const fi = (v) => (v < 0 ? MINUS + Math.abs(v) : String(v));

    /* rótulo com subscrito: 'Y|A' desenha Y com A pequeno */
    function lab(g, s, cx, y, o = {}) {
      const size = o.size || 12, font = o.font || 'mono', weight = o.weight || 600;
      const runs = [];
      const str = String(s);
      for (let i = 0; i < str.length; i++) {
        if (str[i] === '|' && i + 1 < str.length) { runs.push({ t: str[i + 1], sub: true }); i++; }
        else if (runs.length && !runs[runs.length - 1].sub) runs[runs.length - 1].t += str[i];
        else runs.push({ t: str[i], sub: false });
      }
      const ws = runs.map((r) => tw(g, r.t, r.sub ? size * 0.72 : size, font, weight) + (r.sub ? 0.5 : 0));
      let x = cx - ws.reduce((a, b) => a + b, 0) / 2;
      runs.forEach((r, k) => { V.text(g, r.t, x, y + (r.sub ? size * 0.22 : 0), { size: r.sub ? size * 0.72 : size, font, weight, color: o.color }); x += ws[k]; });
    }

    /* ============================ F1 — cinco maneiras de dividir ============ */
    (function () {
      const fig = root.querySelector('#fig-mapa');
      if (!fig) return;
      adaptH(fig, (w) => (w < 560 ? 430 : 440));
      const ST = { mode: 'tp' };
      const ctl = UI.controls(fig);
      const C0 = V.C();
      UI.seg(ctl, { label: 'Divisão', options: [{ v: 'tp', label: 'TP' }, { v: 'ep', label: 'EP' }, { v: 'pp', label: 'PP' }, { v: 'dp', label: 'DP' }, { v: 'dcp', label: 'DCP' }], value: 'tp', onChange: (v) => { ST.mode = v; } });
      UI.legend(ctl, [[C0.a, 'nó A · rank 0'], [C0.b, 'nó B · rank 1'], [C0.c, '⊕ coletiva'], [C0.d, 'token']]);
      const INFO = {
        tp: { t: 'Tensor parallel (TP)', div: 'cada matriz: cabeças inteiras por rank na atenção, metade de cada expert no MoE, faixas de ids no vocabulário.', com: 'um all-reduce no fim de cada atenção e de cada FFN/MoE, em todas as camadas.', rec: 'as três receitas: TP=2, uma GPU por Spark.' },
        ep: { t: 'Expert parallel (EP)', div: 'os experts inteiros entre as GPUs; a atenção continua em TP.', com: 'com DP=1, o mesmo all-reduce do TP; com DP>1, all-to-all para despachar e recombinar tokens.', rec: 'Qwen (256 de 512 por GPU) e GLM no TP3 (96 de 288).' },
        pp: { t: 'Pipeline parallel (PP)', div: 'as camadas, em estágios: metade de cima num nó, metade de baixo no outro.', com: 'a ativação atravessa a fronteira, ponto a ponto, uma vez por passo.', rec: 'não usado: o DeepSeek passa --pipeline-parallel-size 1.' },
        dp: { t: 'Data parallel (DP)', div: 'as requisições: cada nó roda o modelo inteiro e atende as suas.', com: 'nada no forward, só coordenação entre as réplicas.', rec: 'impossível aqui: nenhum dos três modelos cabe num Spark.' },
        dcp: { t: 'Decode context parallel (DCP)', div: 'os tokens do KV cache, intercalados entre os ranks do TP.', com: 'junção exata das atenções parciais (log-sum-exp) a cada camada, além das coletivas do TP.', rec: 'não ligado; o DeepSeek o aponta como jeito de dobrar o KV.' },
      };
      const L = 4, NKV = 4, NH = 4, NE = 8;
      const CH = [[1, 6], [3, 4], [0, 7], [2, 5]];
      function cell(g, x, y, s, kind, own, C, glow) {
        const a = 0.55;
        if (glow) V.glow(g, x + s / 2, y + s / 2, s * 1.1, C.d, 0.9);
        if (kind === 'whole') V.box(g, x, y, s, s, { r: 3, fill: V.alpha(RC(C, own), a), stroke: RC(C, own), lw: 1 });
        else if (kind === 'splitV' || kind === 'splitH') {
          g.save(); V.rr(g, x, y, s, s, 3); g.clip();
          g.fillStyle = V.alpha(C.a, a);
          if (kind === 'splitV') g.fillRect(x, y, s / 2, s); else g.fillRect(x, y, s, s / 2);
          g.fillStyle = V.alpha(C.b, a);
          if (kind === 'splitV') g.fillRect(x + s / 2, y, s / 2, s); else g.fillRect(x, y + s / 2, s, s / 2);
          g.restore();
          V.box(g, x, y, s, s, { r: 3, stroke: V.alpha(C.ink, 0.35), lw: 1 });
          if (kind === 'splitV') V.line(g, x + s / 2, y + 1, x + s / 2, y + s - 1, { color: V.alpha(C.ink, 0.5), lw: 1 });
          else V.line(g, x + 1, y + s / 2, x + s - 1, y + s / 2, { color: V.alpha(C.ink, 0.5), lw: 1 });
        } else if (kind === 'dup') {
          const o = Math.max(2, s * 0.12);
          V.box(g, x + o, y - o, s - o, s - o, { r: 3, fill: V.alpha(C.b, a), stroke: C.b, lw: 1 });
          V.box(g, x, y, s - o, s - o + 0.5, { r: 3, fill: V.alpha(C.a, 0.75), stroke: C.a, lw: 1 });
        }
      }
      const kvK = (m, i, j) => (m === 'tp' || m === 'ep' ? ['splitH'] : m === 'pp' ? ['whole', i < 2 ? 0 : 1] : m === 'dp' ? ['dup'] : ['whole', j % 2]);
      const hdK = (m, i, k) => (m === 'pp' ? ['whole', i < 2 ? 0 : 1] : m === 'dp' ? ['dup'] : ['whole', k < 2 ? 0 : 1]);
      const exK = (m, i, e) => (m === 'pp' ? ['whole', i < 2 ? 0 : 1] : m === 'dp' ? ['dup'] : m === 'ep' ? ['whole', e < 4 ? 0 : 1] : ['splitV']);
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), m = ST.mode, I = INFO[m];
        const narrow = w < 560, pad = 14;
        V.text(g, I.t, pad, 26, { size: narrow ? 15 : 17, font: 'display', weight: 700, color: C.ink });
        const infoH = narrow ? 150 : 92;
        const iy = h - infoH - 6;
        const reqW = narrow ? 26 : 46, labW = narrow ? 0 : 62, gB = narrow ? 8 : 18, oW = narrow ? 18 : 30, gap = 2;
        const nC = NKV + NH + NE;
        const cs = Math.max(8, Math.min(34, (w - 2 * pad - reqW - labW - gB - 2 * oW - (nC - 2) * gap - 8) / nC));
        const rh = cs + (narrow ? 16 : 18);
        const totalW = reqW + labW + (NKV + NH + NE) * cs + (nC - 3) * gap + gB + 2 * oW;
        const ox = Math.max(pad, (w - totalW) / 2);
        const top0 = narrow ? 64 : 62;
        const modelH = L * rh;
        const y0 = top0 + Math.max(0, (iy - 14 - top0 - modelH) / 2) + 10;
        let x = ox + reqW + labW;
        const xKV = x; x += NKV * (cs + gap) - gap + gB;
        const xH = x; x += NH * (cs + gap) - gap;
        const xO1 = x + oW / 2; x += oW;
        const xE = x; x += NE * (cs + gap) - gap;
        const xO2 = x + oW / 2; x += oW;
        const xEnd = x;
        // cabeçalhos dos blocos
        const hy = y0 - 10;
        const hs = 11;
        V.text(g, narrow ? 'KV' : m === 'dcp' ? 'KV (por token)' : 'KV (tokens)', xKV + (NKV * (cs + gap)) / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, narrow ? 'atenção' : 'atenção · 4 cabeças', xH + (NH * (cs + gap)) / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, narrow ? 'MoE' : 'MoE · 8 experts', xE + (NE * (cs + gap)) / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        if (!narrow) V.text(g, 'pedidos', ox + reqW / 2, hy, { size: hs, align: 'center', color: C.muted, font: 'mono' });
        // token em movimento
        const P = 7;
        const tokPos = (ph) => { const u = ((t / P + ph) % 1 + 1) % 1; const row = Math.min(L - 1, Math.floor(u * L)); const f = u * L - row; return { row, x: V.lerp(xKV - 8, xEnd + 4, f), f }; };
        const toks = m === 'dp' ? [tokPos(0), tokPos(0.5)] : [tokPos(0)];
        // pedidos
        for (let i = 0; i < 4; i++) {
          const cy = y0 + (i + 0.5) * rh * (L / 4) - 4, cx = ox + reqW / 2, rr = Math.min(9, cs * 0.42);
          if (m === 'dp') V.box(g, cx - rr, cy - rr, 2 * rr, 2 * rr, { r: rr, fill: V.alpha(RC(C, i % 2), 0.7), stroke: RC(C, i % 2) });
          else {
            g.save(); g.beginPath(); g.arc(cx, cy, rr, 0, Math.PI * 2); g.clip();
            g.fillStyle = V.alpha(C.a, 0.6); g.fillRect(cx - rr, cy - rr, rr, 2 * rr);
            g.fillStyle = V.alpha(C.b, 0.6); g.fillRect(cx, cy - rr, rr, 2 * rr);
            g.restore();
            g.strokeStyle = V.alpha(C.ink, 0.4); g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, rr, 0, Math.PI * 2); g.stroke();
          }
          if (!narrow) V.text(g, 'R' + (i + 1), cx, cy + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        }
        // linhas do modelo
        for (let i = 0; i < L; i++) {
          const y = y0 + i * rh;
          if (!narrow) {
            const st = m === 'pp' ? (i < 2 ? 0 : 1) : -1;
            V.text(g, 'camada ' + i, ox + reqW + 4, y + cs / 2 + 4, { size: 11, font: 'mono', color: st >= 0 ? RC(C, st) : C.muted, weight: st >= 0 ? 700 : 400 });
          }
          const tk = toks.find((q) => q.row === i);
          for (let j = 0; j < NKV; j++) { const k = kvK(m, i, j); cell(g, xKV + j * (cs + gap), y, cs, k[0], k[1], C, false); }
          for (let k = 0; k < NH; k++) { const q = hdK(m, i, k); const on = tk && tk.x > xH && tk.x < xO1 - oW / 2; cell(g, xH + k * (cs + gap), y, cs, q[0], q[1], C, on && k === Math.floor((t * 3) % 4)); }
          for (let e = 0; e < NE; e++) {
            const q = exK(m, i, e);
            const chosen = CH[i].includes(e);
            const on = tk && tk.x > xE && tk.x < xO2 - oW / 2 && (m === 'ep' || m === 'tp' || m === 'dcp' || m === 'pp' || m === 'dp') && chosen;
            cell(g, xE + e * (cs + gap), y, cs, q[0], q[1], C, on);
          }
          const cy = y + cs / 2, pr = Math.min(10, oW / 2 - 2);
          if (m !== 'pp' && m !== 'dp') {
            const on1 = !!(tk && Math.abs(tk.x - xO1) < oW * 0.7), on2 = !!(tk && Math.abs(tk.x - xO2) < oW * 0.7);
            plus(g, xO1, cy, pr, C.c, on1);
            plus(g, xO2, cy, pr, C.c, on2);
            if (m === 'dcp' && !narrow && i === 0) V.text(g, 'LSE', xO1, y - 4, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 700 });
          } else {
            V.line(g, xO1 - oW / 2 + 3, cy, xO1 + oW / 2 - 3, cy, { color: V.alpha(C.muted, 0.35), lw: 1.2, dash: [2, 3] });
            V.line(g, xO2 - oW / 2 + 3, cy, xO2 + oW / 2 - 3, cy, { color: V.alpha(C.muted, 0.35), lw: 1.2, dash: [2, 3] });
          }
          // DCP: consultas às duas metades do KV
          if (m === 'dcp' && tk && tk.x > xH && tk.x < xO1) {
            for (let j = 0; j < NKV; j++) V.arrow(g, xH, cy, xKV + j * (cs + gap) + cs / 2, y + cs, { color: V.alpha(RC(C, j % 2), 0.55), lw: 1.1, head: 5, curve: -10 });
          }
        }
        // PP: fronteira entre estágios
        if (m === 'pp') {
          const yb = y0 + 2 * rh - (rh - cs) / 2;
          V.line(g, xKV - 6, yb, xEnd, yb, { color: V.alpha(C.c, 0.7), lw: 1.4, dash: [5, 4], dashOffset: -t * 20 });
          const crossing = toks[0].row === 2 && toks[0].f < 0.08;
          if (crossing) V.glow(g, xKV - 2, yb, 22, C.c, 0.9);
          V.text(g, narrow ? 'fronteira PP' : 'fronteira entre estágios: a ativação passa de A para B', xEnd, yb - 4, { size: 11, font: 'mono', align: 'right', color: C.c });
        }
        // tokens
        toks.forEach((tk, k) => {
          const ty = y0 + tk.row * rh + cs / 2;
          V.glow(g, tk.x, ty, 16, C.d, 0.95);
          V.dot(g, tk.x, ty, 5, C.d);
          if (m === 'dp') {
            g.strokeStyle = RC(C, k); g.lineWidth = 2; g.beginPath(); g.arc(tk.x, ty, 7, 0, Math.PI * 2); g.stroke();
            chip(g, (k ? 'R2 → nó B' : 'R1 → nó A'), tk.x, ty - cs / 2 - 10, RC(C, k), { align: 'center', size: 11, h: 18, fa: 0.3 });
          }
        });
        // caixa de informação
        V.box(g, pad, iy, w - 2 * pad, infoH, { r: 12, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        const lw = narrow ? 76 : 98, tx = pad + 12, ts = narrow ? 11.5 : 12.5, lh = narrow ? 15 : 17;
        let yy = iy + 22;
        [['Divide', I.div, C.a], ['Comunica', I.com, C.c], ['Nas receitas', I.rec, C.d]].forEach(([k, v, col]) => {
          V.text(g, k, tx, yy, { size: ts, weight: 700, color: col });
          const end = wrap(g, v, tx + lw, yy, w - 2 * pad - lw - 24, { size: ts, color: C.ink, lh });
          yy = end + (narrow ? 3 : 4);
        });
      });
    })();

    /* ====================== F2 — o par column → row, passo a passo ========== */
    (function () {
      const fig = root.querySelector('#fig-colrow');
      if (!fig) return;
      adaptH(fig, (w) => (w < 640 ? 700 : 450));
      const mm = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
      const relu = (M) => M.map((r) => r.map((v) => Math.max(0, v)));
      const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
      const cols = (M, a, b) => M.map((r) => r.slice(a, b));
      const rows = (M, a, b) => M.slice(a, b);
      const rm = (R, K, f) => Array.from({ length: R }, () => Array.from({ length: K }, f));
      function gen(seed) {
        const r = V.rng(seed); const ri = (a, b) => a + Math.floor(r() * (b - a + 1));
        for (let tries = 0; tries < 400; tries++) {
          const X = rm(2, 4, () => ri(-2, 2)), W1 = rm(4, 4, () => ri(-2, 2)), W2 = rm(4, 4, () => ri(-1, 2));
          const Y = mm(X, W1), fl = Y.flat();
          if (!fl.some((v) => v < 0) || fl.filter((v) => v > 0).length < 3) continue;
          const XA = cols(X, 0, 2), XB = cols(X, 2, 4), P0 = mm(XA, rows(W1, 0, 2)), P1 = mm(XB, rows(W1, 2, 4));
          const lhs = add(relu(P0), relu(P1)), rhs = relu(add(P0, P1));
          const diff = [];
          lhs.forEach((row, i) => row.forEach((v, j) => { if (v !== rhs[i][j]) diff.push([i, j]); }));
          if (!diff.length) continue;
          const H = relu(Y), Z = mm(H, W2);
          const HA = cols(H, 0, 2), HB = cols(H, 2, 4), W2A = rows(W2, 0, 2), W2B = rows(W2, 2, 4);
          const ZA = mm(HA, W2A), ZB = mm(HB, W2B), ZS = add(ZA, ZB);
          const neg = (M) => { const o = []; M.forEach((row, i) => row.forEach((v, j) => { if (v < 0) o.push([i, j]); })); return o; };
          return { X, W1, W2, Y, H, Z, YA: cols(Y, 0, 2), YB: cols(Y, 2, 4), HA, HB, W1A: cols(W1, 0, 2), W1B: cols(W1, 2, 4), W2A, W2B, ZA, ZB, ZS, XA, XB, W1rA: rows(W1, 0, 2), W1rB: rows(W1, 2, 4), P0, P1, fP0: relu(P0), fP1: relu(P1), lhs, rhs, diff, negA: neg(cols(Y, 0, 2)), negB: neg(cols(Y, 2, 4)), ok: ZS.every((row, i) => row.every((v, j) => v === Z[i][j])) };
        }
        return null;
      }
      let seed = 11;
      let D = gen(seed);
      const steps = [
        { t: 'A conta inteira numa GPU', d: 'Dois tokens (<b>X</b>, 2×4), uma camada que expande para 4 (<b>W<sub>1</sub></b>), a não-linearidade <b>f</b> (ReLU) e a camada que volta (<b>W<sub>2</sub></b>). O resultado <b>Z</b> é a referência que o TP precisa reproduzir exatamente.' },
        { t: 'Column-parallel: W₁ cortada em colunas', d: 'O nó A guarda as colunas 0–1 de W<sub>1</sub> e o nó B, as colunas 2–3. A entrada <b>X</b> está inteira (replicada) nos dois nós.' },
        { t: 'Cada nó calcula sua metade de Y, sem conversar', d: 'Cada coluna de Y depende só de X e da coluna correspondente de W<sub>1</sub>. Y<sub>A</sub> e Y<sub>B</sub> saem sem nenhuma troca de mensagens. O destaque mostra a linha de X e a coluna de W<sub>1</sub> de cada célula.' },
        { t: 'A não-linearidade é local', d: 'f age elemento a elemento: f(Y<sub>A</sub>) só precisa de Y<sub>A</sub>. As células contornadas eram negativas e viraram 0. Nada foi comunicado até aqui.' },
        { t: 'Row-parallel: W₂ cortada em linhas', d: 'W<sub>2</sub> é cortada nas linhas que casam com as colunas de H que cada nó tem. Cada nó produz um Z <b>inteiro</b> (2×4), mas <b>parcial</b>: falta a contribuição do outro.' },
        { t: 'All-reduce: somar as parciais', d: 'Z<sub>A</sub> + Z<sub>B</sub> = Z nos dois nós. É a única comunicação do par, e o resultado é idêntico ao da GPU única, célula por célula.' },
        { t: 'E se fosse row → col?', d: 'Com a primeira camada em linhas, cada nó teria uma soma parcial P<sub>A</sub>, P<sub>B</sub>. Como f(P<sub>A</sub>) + f(P<sub>B</sub>) ≠ f(P<sub>A</sub> + P<sub>B</sub>) (células em vermelho), seria preciso somar <i>antes</i> de f: duas coletivas por par em vez de uma.' },
      ];
      let H2 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => { if (H2) H2.redraw(); } });
      const ctl = UI.controls(fig);
      UI.button(ctl, '🎲 novos números', () => { seed += 17; const n = gen(seed); if (n) D = n; st.t0 = performance.now(); if (H2) H2.redraw(); });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'nó A'], [C0.b, 'nó B'], [C0.muted, 'replicado / referência'], [C0.c, 'destaque']]);

      function mat(g, M, x, y, cs, col, o = {}) {
        const C = V.C();
        const R = M.length, K = M[0].length, gp = Math.max(1.5, cs * 0.08);
        const vmax = Math.max(1, ...M.flat().map((v) => Math.abs(v)));
        let idx = 0;
        if (o.alpha != null) g.globalAlpha = o.alpha;
        for (let i = 0; i < R; i++) for (let j = 0; j < K; j++, idx++) {
          const cx = x + j * (cs + gp), cy = y + i * (cs + gp);
          const shown = o.reveal == null || idx < o.reveal;
          const v = M[i][j];
          V.box(g, cx, cy, cs, cs, { r: Math.min(4, cs / 5), fill: shown ? V.alpha(col, 0.1 + 0.34 * Math.min(1, Math.abs(v) / vmax)) : V.alpha(C.muted, 0.06), stroke: V.alpha(col, shown ? 0.75 : 0.3), lw: 1 });
          V.text(g, shown ? fi(v) : '·', cx + cs / 2, cy + cs / 2 + 0.5, { size: Math.max(9, Math.min(13, cs * 0.45)), font: 'mono', weight: 600, align: 'center', base: 'middle', color: shown ? C.ink : C.muted });
        }
        const Wd = K * cs + (K - 1) * gp, Hh = R * cs + (R - 1) * gp;
        const hl = (hx, hy, hw, hh, c) => { g.strokeStyle = c; g.lineWidth = 2.2; V.rr(g, hx - 2, hy - 2, hw + 4, hh + 4, 5); g.stroke(); };
        if (o.hiRow != null) hl(x, y + o.hiRow * (cs + gp), Wd, cs, V.C().c);
        if (o.hiCol != null) hl(x + o.hiCol * (cs + gp), y, cs, Hh, V.C().c);
        (o.hi || []).forEach(([i, j]) => hl(x + j * (cs + gp), y + i * (cs + gp), cs, cs, o.hiColor || V.C().c));
        if (o.alpha != null) g.globalAlpha = 1;
        if (o.label) lab(g, o.label, x + Wd / 2, y - 7, { size: Math.min(13, Math.max(10.5, cs * 0.46)), color: o.labColor || C.ink });
        if (o.sub) V.text(g, o.sub, x + Wd / 2, y + Hh + 13, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        return { w: Wd, h: Hh };
      }
      const gpOf = (cs) => Math.max(1.5, cs * 0.08);
      const itW = (it, cs) => (it.M ? it.M[0].length * cs + (it.M[0].length - 1) * gpOf(cs) : (it.opW || 0.9) * cs);
      const itH = (it, cs) => (it.M ? it.M.length * cs + (it.M.length - 1) * gpOf(cs) : cs);
      function fitCs(items, availW, csMax) { let u = 0; items.forEach((it) => { u += it.M ? it.M[0].length * 1.08 : it.opW || 0.9; }); u += (items.length - 1) * 0.3; return Math.max(12, Math.min(csMax, availW / u)); }
      function eqH(items, cs) { return Math.max(...items.map((it) => itH(it, cs))) + 34; }
      function eqDraw(g, items, cx, yTop, cs) {
        const C = V.C();
        const ws = items.map((it) => itW(it, cs));
        const wsum = ws.reduce((a, b) => a + b, 0) + (items.length - 1) * cs * 0.3;
        const hm = Math.max(...items.map((it) => itH(it, cs)));
        let x = cx - wsum / 2;
        const mid = yTop + 18 + hm / 2;
        items.forEach((it, k) => {
          if (it.M) mat(g, it.M, x, mid - itH(it, cs) / 2, cs, it.col, it);
          else V.text(g, it.op, x + ws[k] / 2, mid + 1, { size: Math.min(20, cs * (it.opSize || 0.8)), weight: 700, align: 'center', base: 'middle', color: it.color || C.muted, font: it.font || 'sans' });
          x += ws[k] + cs * 0.3;
        });
      }
      function panel(g, P, r, title) {
        const C = V.C(), col = r == null ? C.muted : RC(C, r);
        V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.55), lw: 1.2 });
        chip(g, title, P.x + 10, P.y + 16, col, { size: 11, h: 20 });
      }
      function drawRows(g, P, rowsIt, csMax) {
        const inner = P.w - 20;
        let cs = Math.min(...rowsIt.map((items) => fitCs(items, inner, csMax)));
        let tot = rowsIt.reduce((a, items) => a + eqH(items, cs), 0);
        const avail = P.h - 34;
        if (tot > avail) { cs *= avail / tot; tot = rowsIt.reduce((a, items) => a + eqH(items, cs), 0); }
        let y = P.y + 30 + Math.max(0, (avail - tot) / 2);
        rowsIt.forEach((items) => { eqDraw(g, items, P.x + P.w / 2, y, cs); y += eqH(items, cs); });
        return cs;
      }
      H2 = V.fig(fig, (g, w, h) => {
        if (w < 140 || h < 100) return;
        const C = V.C();
        if (!D) return;
        const i = st.i, el = st.elapsed();
        const wide = w >= 640, pad = 10;
        let PA, PB;
        const stripH = i === 6 ? (wide ? 150 : 170) : 0;
        if (wide) { const pw = (w - 3 * pad) / 2; PA = { x: pad, y: pad, w: pw, h: h - 2 * pad - stripH - (stripH ? pad : 0) }; PB = { x: 2 * pad + pw, y: pad, w: pw, h: PA.h }; }
        else { const ph = (h - 3 * pad - stripH - (stripH ? pad : 0)) / 2; PA = { x: pad, y: pad, w: w - 2 * pad, h: ph }; PB = { x: pad, y: 2 * pad + ph, w: w - 2 * pad, h: ph }; }
        const nA = C.a, nB = C.b, nN = C.muted;
        const csMax = 30;
        const rev = (n, per) => Math.floor(el / per) + 1;
        if (i === 0) {
          const P = { x: pad, y: pad, w: w - 2 * pad, h: h - 2 * pad };
          panel(g, P, null, 'GPU única (referência)');
          const r1 = [{ M: D.X, col: nN, label: 'X', sub: '[2×4]' }, { op: '·' }, { M: D.W1, col: nN, label: 'W₁', sub: '[4×4]' }];
          const r2 = [{ op: '=' }, { M: D.Y, col: nN, label: 'Y', sub: '[2×4]' }];
          const r3 = [{ op: 'f(', opW: 1.1, font: 'mono' }, { M: D.Y, col: nN, label: 'Y' }, { op: ')', opW: 0.5, font: 'mono' }, { op: '·' }, { M: D.W2, col: nN, label: 'W₂', sub: '[4×4]' }];
          const r4 = [{ op: '=' }, { M: D.Z, col: C.c, label: 'Z', sub: 'referência' }];
          drawRows(g, P, wide ? [r1.concat(r2), r3.concat(r4)] : [r1, r2, r3, r4], csMax);
          return;
        }
        panel(g, PA, 0, 'nó A · rank 0');
        panel(g, PB, 1, 'nó B · rank 1');
        const both = (fn) => { fn(PA, 0, nA); fn(PB, 1, nB); };
        if (i === 1) {
          const a = smooth(el / 0.7);
          both((P, r, col) => drawRows(g, P, [[{ M: D.X, col: nN, label: 'X', sub: 'replicado' }, { op: '·' }, { M: r ? D.W1B : D.W1A, col, label: 'W₁|' + (r ? 'B' : 'A'), sub: r ? 'colunas 2–3' : 'colunas 0–1', alpha: a }]], csMax));
        } else if (i === 2) {
          const k = Math.min(4, rev(4, 0.5));
          const cur = Math.min(3, k - 1), hr = Math.floor(cur / 2), hc = cur % 2;
          both((P, r, col) => drawRows(g, P, [[{ M: D.X, col: nN, label: 'X', hiRow: k <= 4 && el < 2.4 ? hr : null }, { op: '·' }, { M: r ? D.W1B : D.W1A, col, label: 'W₁|' + (r ? 'B' : 'A'), hiCol: el < 2.4 ? hc : null }, { op: '=' }, { M: r ? D.YB : D.YA, col, label: 'Y|' + (r ? 'B' : 'A'), sub: '[2×2]', reveal: k }]], csMax));
        } else if (i === 3) {
          const on = el > 0.5;
          both((P, r, col) => drawRows(g, P, [[{ op: 'f(', opW: 1.1, font: 'mono' }, { M: r ? D.YB : D.YA, col, label: 'Y|' + (r ? 'B' : 'A'), hi: r ? D.negB : D.negA }, { op: ')', opW: 0.5, font: 'mono' }, { op: '=' }, { M: r ? D.HB : D.HA, col, label: 'H|' + (r ? 'B' : 'A'), sub: 'local', hi: on ? (r ? D.negB : D.negA) : [], reveal: on ? 4 : 0 }]], csMax));
        } else if (i === 4) {
          const k = rev(8, 0.3);
          both((P, r, col) => drawRows(g, P, [[{ M: r ? D.HB : D.HA, col, label: 'H|' + (r ? 'B' : 'A') }, { op: '·' }, { M: r ? D.W2B : D.W2A, col, label: 'W₂|' + (r ? 'B' : 'A'), sub: r ? 'linhas 2–3' : 'linhas 0–1' }], [{ op: '=' }, { M: r ? D.ZB : D.ZA, col, label: 'Z|' + (r ? 'B' : 'A'), sub: 'inteiro, mas parcial', reveal: k }]], csMax));
        } else if (i === 5) {
          const p = smooth((el - 0.2) / 1.1);
          both((P, r, col) => drawRows(g, P, [[{ M: D.ZA, col: nA, label: 'Z|A', sub: 'do nó A' }, { op: '+', color: C.c }, { M: D.ZB, col: nB, label: 'Z|B', sub: 'do nó B' }], [{ op: '=' }, { M: D.ZS, col: C.c, label: 'Z', sub: D.ok ? '✓ igual à referência' : '', reveal: p >= 1 ? 8 : 0 }]], csMax));
          // setas do all-reduce entre os painéis
          const n = 5;
          for (let k = 0; k < n; k++) {
            const f = ((el * 0.6 + k / n) % 1);
            if (wide) {
              const yy = PA.y + PA.h * (0.25 + 0.5 * (k / (n - 1)));
              const x0 = PA.x + PA.w - 6, x1 = PB.x + 6;
              V.dot(g, V.lerp(x0, x1, f), yy, 3.2, C.a);
              V.dot(g, V.lerp(x1, x0, f), yy + 6, 3.2, C.b);
            } else {
              const xx = PA.x + PA.w * (0.2 + 0.6 * (k / (n - 1)));
              const y0 = PA.y + PA.h - 4, y1 = PB.y + 4;
              V.dot(g, xx, V.lerp(y0, y1, f), 3.2, C.a);
              V.dot(g, xx + 6, V.lerp(y1, y0, f), 3.2, C.b);
            }
          }
          if (wide) chip(g, 'all-reduce', w / 2, PA.y + PA.h - 18, C.c, { align: 'center', size: 11, h: 18, fa: 0.35 });
          else chip(g, 'all-reduce', w / 2, PB.y - pad / 2, C.c, { align: 'center', size: 11, h: 18 });
        } else if (i === 6) {
          both((P, r, col) => drawRows(g, P, [[{ M: r ? D.XB : D.XA, col, label: 'X|' + (r ? 'B' : 'A'), sub: r ? 'colunas 2–3' : 'colunas 0–1' }, { op: '·' }, { M: r ? D.W1rB : D.W1rA, col, label: 'W₁', sub: r ? 'linhas 2–3' : 'linhas 0–1' }], [{ op: '=' }, { M: r ? D.P1 : D.P0, col, label: 'P|' + (r ? 'B' : 'A'), sub: 'parcial' }]], csMax));
          const S = { x: pad, y: h - pad - stripH, w: w - 2 * pad, h: stripH };
          V.box(g, S.x, S.y, S.w, S.h, { r: 12, fill: V.alpha(C.red || C.c, 0.05), stroke: V.alpha(C.red || C.c, 0.5), lw: 1.2 });
          const on = el > 0.8;
          drawRows(g, { x: S.x, y: S.y - 22, w: S.w, h: S.h + 20 }, [[{ M: D.lhs, col: nN, label: 'f(P|A) + f(P|B)', hi: on ? D.diff : [], hiColor: C.red || C.c }, { op: '≠', color: C.red || C.c, opSize: 1 }, { M: D.rhs, col: nN, label: 'f(P|A + P|B)', hi: on ? D.diff : [], hiColor: C.red || C.c }]], 26);
        }
      });
    })();

    /* ========================= F3 — metrô do bloco sob TP ==================== */
    (function () {
      const fig = root.querySelector('#fig-metro');
      if (!fig) return;
      adaptH(fig, (w) => (w < 600 ? 720 : 420));
      const ST = { model: 'qwen' };
      const MODELS = {
        glm: {
          st: [
            { k: 'rep', n: 'mHC + norma', s: 'réplica' },
            { k: 'col', n: 'in_proj / q_b', s: 'col · 32 cab.' },
            { k: 'loc', n: 'KDA / MLA', s: 'por cabeça' },
            { k: 'row', n: 'o_proj / out', s: 'row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
            { k: 'rep', n: 'roteador', s: 'réplica' },
            { k: 'col', n: 'gate / up', s: 'col · 1024' },
            { k: 'loc', n: 'SwiGLU', s: 'local' },
            { k: 'row', n: 'down', s: 'row · 1024' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
          ],
          br: [['misturador (KDA/MLA)', 45], ['FFN / MoE', 45]],
          tot: '≈ 90 +', note: 'Interpretação nossa: 2 por camada nas 45 camadas. O repo documenta 1 all-reduce por camada MoE e o out_proj row-parallel do KDA; embedding, logits e DFlash2 não entram nesta conta.',
          rep: 'réplicas: latente c_kv da MLA, indexador, f_a/g_a do KDA',
        },
        qwen: {
          st: [
            { k: 'rep', n: 'HC (4 ramos)', s: 'réplica 1,19 GiB' },
            { k: 'col', n: 'qkvz / qkv+gate', s: 'col' },
            { k: 'loc', n: 'GDN / QSA', s: '24 / 12 cab. por GPU' },
            { k: 'row', n: 'out_proj', s: 'row · 3072' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
            { k: 'rep', n: 'roteador', s: 'réplica' },
            { k: 'ep', n: 'experts locais', s: 'EP · 256 de 512' },
            { k: 'col', n: 'shared expert', s: 'col → row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
          ],
          br: [['GDN out_proj', 36], ['atenção o_proj', 12], ['MoE', 48], ['logits (gather)', 1], ['rascunhador', 9]],
          tot: '≈ 106', note: 'Auditoria da receita (batch 1, MTP=3): mensagens de 5–20 KB, 20–30 µs cada, 2–3 ms por passo (6–8%); não diminui com o batch.',
          rep: 'réplicas lidas por GPU a cada passo: HC 1,19 GiB · roteador 0,12 · indexador 0,04 GiB',
        },
        ds: {
          st: [
            { k: 'rep', n: 'mHC + norma', s: 'réplica' },
            { k: 'rep', n: 'wqa + wkv', s: 'réplica (latente)' },
            { k: 'col', n: 'wq_b', s: 'col · 32 cab.' },
            { k: 'loc', n: 'atenção', s: 'KV e indexador réplica' },
            { k: 'col', n: 'wo_a', s: '4 grupos' },
            { k: 'row', n: 'wo_b', s: 'row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
            { k: 'rep', n: 'gate (fp32)', s: 'réplica' },
            { k: 'col', n: 'experts w1/w3', s: 'TP no expert' },
            { k: 'row', n: 'experts w2', s: 'row' },
            { k: 'ar', n: '⊕', s: 'all-reduce' },
          ],
          br: [['wo_b', 43], ['MoE', 43], ['rascunho', 6], ['embedding', 2], ['logits (gather)', 2], ['laço de Markov', 12]],
          tot: '≈ 108', note: 'Auditoria da receita (TP=2, RoCE sem GPUDirect): 30–50 µs cada, 3–5 ms por passo de decode.',
          rep: 'réplicas: fused_wqa_wkv, indexador e todo o KV latente (1 cabeça)',
        },
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DeepSeek-V4' }], value: 'qwen', onChange: (v) => { ST.model = v; } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'nó A'], [C0.b, 'nó B'], [C0.muted, 'réplica (hachurada)'], [C0.c, '⊕ all-reduce']]);
      function stationBox(g, x, y, bw, bh, k, r, C, t, lbl) {
        const col = RC(C, r);
        if (k === 'rep') {
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.7), lw: 1.2 });
          hatch(g, x - bw / 2 + 2, y - bh / 2 + 2, bw - 4, bh - 4, V.alpha(C.muted, 0.45), 6, t * 8);
        } else if (k === 'col' || k === 'row') {
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 6, fill: V.alpha(col, 0.08), stroke: col, lw: 1.3 });
          g.save(); V.rr(g, x - bw / 2, y - bh / 2, bw, bh, 6); g.clip();
          g.fillStyle = V.alpha(col, 0.55);
          if (k === 'col') g.fillRect(x - bw / 2 + (r ? bw / 2 : 0), y - bh / 2, bw / 2, bh);
          else g.fillRect(x - bw / 2, y - bh / 2 + (r ? bh / 2 : 0), bw, bh / 2);
          g.restore();
        } else if (k === 'ep') {
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 6, fill: V.alpha(col, 0.55), stroke: col, lw: 1.3 });
          if (lbl) V.text(g, r ? '256–511' : '0–255', x, y + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        } else if (k === 'loc') {
          V.box(g, x - bh / 2, y - bh / 2, bh, bh, { r: bh / 2, fill: V.alpha(col, 0.35), stroke: col, lw: 1.3 });
        }
      }
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), M = MODELS[ST.model], stn = M.st, n = stn.length;
        const narrow = w < 600, pad = 14;
        const brH = narrow ? 210 : 150;
        const regH = h - brH - 10;
        // estado do trem: full / fatia / parcial
        const stateAfter = []; let s = 'full';
        stn.forEach((q) => { if (q.k === 'col' || q.k === 'loc') s = s === 'parcial' ? 'parcial' : 'fatia'; if (q.k === 'row' || q.k === 'ep') s = 'parcial'; if (q.k === 'ar') s = 'full'; stateAfter.push(s); });
        if (!narrow) {
          const yA = 58, yB = regH - 58, mid = (yA + yB) / 2;
          const x0 = 70, x1 = w - 22, sp = (x1 - x0) / n;
          const xs = stn.map((_, i) => x0 + (i + 0.5) * sp);
          const ars = xs.filter((_, i) => stn[i].k === 'ar');
          const d = sp * 0.62;
          const ty = (x, base) => { let y = base; ars.forEach((xa) => { const u = 1 - Math.abs(x - xa) / d; if (u > 0) y = V.lerp(base, mid + (base < mid ? -7 : 7), smooth(u)); }); return y; };
          V.text(g, 'nó A', 14, yA + 4, { size: 12, weight: 700, color: C.a, font: 'mono' });
          V.text(g, 'nó B', 14, yB + 4, { size: 12, weight: 700, color: C.b, font: 'mono' });
          [[yA, C.a], [yB, C.b]].forEach(([base, col]) => {
            g.beginPath();
            for (let x = x0 - 12; x <= x1; x += 3) { const y = ty(x, base); if (x === x0 - 12) g.moveTo(x, y); else g.lineTo(x, y); }
            g.strokeStyle = V.alpha(col, 0.55); g.lineWidth = 5; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
          });
          // trens
          const u = (t * 0.075) % 1;
          const tx = V.lerp(x0 - 10, x1, u);
          let idx = -1; xs.forEach((xx, i) => { if (tx > xx) idx = i; });
          const tst = idx < 0 ? 'full' : stateAfter[idx];
          // estações
          const bw = Math.min(sp - 12, 58), bh = 24;
          xs.forEach((xx, i) => {
            const q = stn[i];
            if (q.k === 'ar') {
              const on = Math.abs(tx - xx) < sp * 0.35;
              plus(g, xx, mid, 12, C.c, on);
              V.text(g, 'all-reduce', xx, mid + 30, { size: 11, font: 'mono', align: 'center', color: C.c });
              return;
            }
            stationBox(g, xx, yA, bw, bh, q.k, 0, C, t, bw > 50);
            stationBox(g, xx, yB, bw, bh, q.k, 1, C, t, bw > 50);
            const nl = wrapLines(g, q.n, sp - 6, 11, 'sans', 700);
            nl.slice(0, 2).forEach((ln, k) => V.text(g, ln, xx, mid - 6 + k * 13 - (nl.length > 1 ? 6 : 0), { size: 11, weight: 700, align: 'center', color: C.ink }));
            const sl = wrapLines(g, q.s, sp - 4, 11, 'mono', 400);
            sl.slice(0, 3).forEach((ln, k) => V.text(g, ln, xx, yB + 30 + k * 13, { size: 11, font: 'mono', align: 'center', color: C.muted }));
          });
          [[yA, 0], [yB, 1]].forEach(([base, r]) => {
            const col = RC(C, r), y = ty(tx, base);
            const L = tst === 'fatia' ? 16 : 30;
            V.glow(g, tx, y, 18, col, 0.6);
            if (tst === 'parcial') { V.box(g, tx - L / 2, y - 6, L, 12, { r: 4, fill: V.alpha(col, 0.2), stroke: col, lw: 1.4 }); V.text(g, 'Σ', tx, y + 0.5, { size: 11, weight: 700, align: 'center', base: 'middle', color: C.ink }); }
            else V.box(g, tx - L / 2, y - 6, L, 12, { r: 4, fill: V.alpha(col, 0.9), stroke: col, lw: 1 });
          });
          V.text(g, 'trem cheio = vetor inteiro · curto = fatia · Σ = soma parcial', w - 14, 16, { size: 11, font: 'mono', align: 'right', color: C.muted });
        } else {
          const xA = pad + 16, xB = w - pad - 16, midx = w / 2;
          const y0 = 34, y1 = regH - 8, sp = (y1 - y0) / n;
          const ys = stn.map((_, i) => y0 + (i + 0.5) * sp);
          const ars = ys.filter((_, i) => stn[i].k === 'ar');
          const d = sp * 0.75;
          const tx = (y, base) => { let x = base; ars.forEach((ya) => { const u = 1 - Math.abs(y - ya) / d; if (u > 0) x = V.lerp(base, midx + (base < midx ? -18 : 18), smooth(u)); }); return x; };
          V.text(g, 'nó A', xA, 18, { size: 11, weight: 700, color: C.a, font: 'mono', align: 'center' });
          V.text(g, 'nó B', xB, 18, { size: 11, weight: 700, color: C.b, font: 'mono', align: 'center' });
          [[xA, C.a], [xB, C.b]].forEach(([base, col]) => {
            g.beginPath();
            for (let y = y0 - 6; y <= y1; y += 3) { const x = tx(y, base); if (y === y0 - 6) g.moveTo(x, y); else g.lineTo(x, y); }
            g.strokeStyle = V.alpha(col, 0.55); g.lineWidth = 5; g.lineCap = 'round'; g.stroke(); g.lineCap = 'butt';
          });
          const u = (t * 0.075) % 1;
          const ty = V.lerp(y0 - 4, y1, u);
          let idx = -1; ys.forEach((yy, i) => { if (ty > yy) idx = i; });
          const tst = idx < 0 ? 'full' : stateAfter[idx];
          const bw = 24, bh = Math.min(20, sp - 8);
          ys.forEach((yy, i) => {
            const q = stn[i];
            if (q.k === 'ar') { plus(g, midx, yy, 11, C.c, Math.abs(ty - yy) < sp * 0.4); V.text(g, 'all-reduce', midx + 18, yy + 4, { size: 11, font: 'mono', color: C.c }); return; }
            stationBox(g, xA, yy, bw, bh, q.k, 0, C, t, false);
            stationBox(g, xB, yy, bw, bh, q.k, 1, C, t, false);
            V.text(g, q.n, midx, yy - 2, { size: 11.5, weight: 700, align: 'center', color: C.ink, maxW: xB - xA - 50 });
            V.text(g, q.s, midx, yy + 12, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: xB - xA - 50 });
          });
          [[xA, 0], [xB, 1]].forEach(([base, r]) => {
            const col = RC(C, r), x = tx(ty, base);
            const L = tst === 'fatia' ? 10 : 18;
            V.glow(g, x, ty, 14, col, 0.6);
            if (tst === 'parcial') { V.box(g, x - 6, ty - L / 2, 12, L, { r: 4, fill: V.alpha(col, 0.2), stroke: col, lw: 1.4 }); }
            else V.box(g, x - 6, ty - L / 2, 12, L, { r: 4, fill: V.alpha(col, 0.9), stroke: col });
          });
        }
        // painel de contagem
        const by = h - brH;
        V.box(g, pad, by, w - 2 * pad, brH - 8, { r: 12, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        V.text(g, 'coletivas por passo de decode (TP=2)', pad + 12, by + 22, { size: 12, weight: 700, color: C.ink });
        V.text(g, M.tot, w - pad - 12, by + 22, { size: 16, weight: 700, color: C.c, align: 'right', font: 'display' });
        const tot = M.br.reduce((a, b) => a + b[1], 0);
        const bx = pad + 12, bwid = w - 2 * pad - 24, bY = by + 34, bH = 20;
        let xx = bx;
        const cols = [C.a, C.b, C.d, C.e, C.f, C.c];
        M.br.forEach(([nm, v], k) => {
          const ww = (v / tot) * bwid;
          V.box(g, xx, bY, Math.max(2, ww - 2), bH, { r: 4, fill: V.alpha(cols[k % 6], 0.7) });
          if (ww > 26) V.text(g, String(v), xx + ww / 2 - 1, bY + bH / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.dark ? '#08140e' : '#fff' });
          xx += ww;
        });
        // legenda da barra
        let lx = bx, ly = bY + bH + 18;
        M.br.forEach(([nm, v], k) => {
          const s = nm + ' ' + v;
          const ww = tw(g, s, 11, 'mono', 400) + 22;
          if (lx + ww > bx + bwid) { lx = bx; ly += 16; }
          V.box(g, lx, ly - 8, 9, 9, { r: 2, fill: V.alpha(cols[k % 6], 0.8) });
          V.text(g, s, lx + 13, ly, { size: 11, font: 'mono', color: C.ink });
          lx += ww;
        });
        ly = wrap(g, M.note, bx, ly + 18, bwid, { size: 11, color: C.muted, lh: 14 });
        wrap(g, M.rep, bx, ly + 2, bwid, { size: 11, color: C.ink, lh: 14 });
      });
    })();

    /* ================== F4 — cortar pesos quantizados ======================= */
    (function () {
      const fig = root.querySelector('#fig-quantcut');
      if (!fig) return;
      const ST0 = { pre: 'qmtp', p: 2 };
      const reH = adaptH(fig, (w) => {
        const nar = w < 600;
        if (ST0.pre === 'exl3' || ST0.pre === 'glm') return nar ? 490 : 440;
        if (ST0.pre === 'kda') return (nar ? 320 : 250) + ST0.p * (nar ? 44 : 50);
        return nar ? 400 : 350;
      });
      const PRE = {
        qmtp: { label: 'Qwen · MTP (FP8 128×128) · 640', n: 640, fmt: 'fp8b' },
        qvit: { label: 'Qwen · visão (NVFP4) · 4304', n: 4304, fmt: 'nvfp4' },
        glm: { label: 'GLM · expert EXL3 · 2048', n: 2048, fmt: 'exl3' },
        ds: { label: 'DS · expert MXFP4 · 2048', n: 2048, fmt: 'mxfp4' },
        kda: { label: 'GLM · in_proj do KDA', fmt: 'segs' },
      };
      const ST = { pre: 'qmtp', p: 2, mode: 'col' };
      const ctl = UI.controls(fig);
      const selT = UI.select(ctl, { label: 'Tensor', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k].label })), value: 'qmtp', onChange: (v) => { ST.pre = v; ST0.pre = v; if (v === 'qvit') segM.set('row'); if (reH) reH(); } });
      selT.el.style.maxWidth = '100%'; selT.el.style.minWidth = '0';
      const selE = selT.el.querySelector('select'); if (selE) { selE.style.maxWidth = '100%'; selE.style.minWidth = '0'; }
      UI.seg(ctl, { label: 'TP', options: [2, 3, 4], value: 2, onChange: (v) => { ST.p = v; ST0.p = v; if (reH) reH(); } });
      const segM = UI.seg(ctl, { label: 'Corte', options: [{ v: 'col', label: 'col (saídas N)' }, { v: 'row', label: 'row (entradas K)' }], value: 'col', onChange: (v) => { ST.mode = v; } });
      const align = (fmt, mode) => (fmt === 'fp8b' || fmt === 'exl3' ? 128 : fmt === 'nvfp4' ? (mode === 'row' ? 16 : 1) : fmt === 'mxfp4' ? (mode === 'row' ? 32 : 1) : 1);
      const BLK = { fp8b: 'bloco 128×128, uma escala por bloco', exl3: 'tiles 16×16 · Hadamard em blocos de 128', nvfp4: 'bloco de 16 valores em K · escala E4M3', mxfp4: 'bloco de 32 valores em K · escala E8M0' };
      function note(pre, p, mode, ok) {
        if (pre === 'qmtp') return ok ? 'fatia alinhada aos blocos' : 'A receita usa EP: 256 experts inteiros por GPU, e a largura 640 fica intacta.';
        if (pre === 'qvit') return mode === 'col' ? 'Corte em N: os blocos de escala correm ao longo de K e não se partem (restrições de kernel à parte).' : ok ? 'fatia alinhada' : 'A receita replica a torre: --mm-encoder-tp-mode data (weights quebra na carga).';
        if (pre === 'glm') return ok ? (p === 2 ? 'É o TP2 da receita: shard_exl3_col/row.' : 'Divide limpo (512 por rank).') : 'No TP3 a receita usa EP: 96 dos 288 experts por rank. A treliça não é enchida.';
        if (pre === 'ds') return ok ? (mode === 'col' ? 'Corte em N: blocos de escala correm em K e não se partem.' : 'down com 1024 = 32 blocos de 32 por rank.') : 'O patch TP3 enche 2048 → 2112 (704 por rank).';
        return '';
      }
      function segsFor(p) {
        const heads = p === 3 ? 66 : 64;
        return { heads, segs: [{ n: 'q', s: heads * 128, sh: 1 }, { n: 'k', s: heads * 128, sh: 1 }, { n: 'v', s: heads * 128, sh: 1 }, { n: 'β', s: heads, sh: 1 }, { n: 'f_a', s: 128, sh: 0 }, { n: 'g_a', s: 128, sh: 0 }] };
      }
      const rLocal = UI.readout(ctl, 'fatia local', '—');
      const rVer = UI.readout(ctl, 'veredito', '—');
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), pr = PRE[ST.pre], p = ST.p, pad = 16;
        const narrow = w < 600;
        const x0 = pad, x1 = w - pad, RW = x1 - x0;
        if (pr.fmt === 'segs') {
          const { heads, segs } = segsFor(p);
          const gl = segs.reduce((a, q) => a + q.s, 0);
          const local = segs.reduce((a, q) => a + (q.sh ? q.s / p : q.s), 0);
          const qual = { 2: 12576, 3: 8726 };
          const ok = qual[p] === local;
          rLocal.set(nf(local) + ' × 4096');
          rVer.set(ok ? '✓ forma qualificada' : '✗ fora da lista');
          const oy = narrow ? 16 : 0;
          V.text(g, (narrow ? 'saída fundida · ' : 'dimensão de saída da projeção fundida (') + 'cabeças ' + heads + (p === 3 ? ', enchidas de 64' : '') + (narrow ? '' : ')'), x0, 24, { size: narrow ? 11 : 12, color: C.muted, font: 'mono', maxW: narrow ? RW : RW - 160 });
          const minW = narrow ? 26 : 34;
          const shW = RW - 3 * minW;
          const widths = segs.map((q) => (q.n === 'β' || !q.sh ? minW : (q.s / (3 * heads * 128)) * (shW - 6)));
          let x = x0;
          const y = 40 + oy, bh = 30;
          segs.forEach((q, k) => {
            const ww = widths[k];
            if (q.sh) {
              for (let r = 0; r < p; r++) { g.fillStyle = V.alpha(RC(C, r), 0.45); g.fillRect(x + (ww * r) / p, y, ww / p, bh); }
              for (let r = 1; r < p; r++) V.line(g, x + (ww * r) / p, y - 3, x + (ww * r) / p, y + bh + 3, { color: C.c, lw: 1.5 });
            } else { V.box(g, x, y, ww - 2, bh, { r: 3, fill: V.alpha(C.muted, 0.12) }); hatch(g, x, y, ww - 2, bh, V.alpha(C.muted, 0.6), 5, t * 6); }
            V.box(g, x, y, ww - 2, bh, { r: 3, stroke: V.alpha(C.ink, 0.45), lw: 1 });
            V.text(g, q.n, x + ww / 2 - 1, y + bh / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
            V.text(g, nf(q.s), x + ww / 2 - 1, y + bh + 14, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: ww });
            x += ww;
          });
          V.text(g, 'global: ' + nf(gl) + ' linhas', narrow ? x0 : x1, narrow ? 24 + oy : 24, { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.muted });
          // linhas por rank
          const ry0 = 104 + oy, rh = narrow ? 44 : 50;
          for (let r = 0; r < p; r++) {
            const yy = ry0 + r * rh;
            V.text(g, 'rank ' + r, x0, yy + 16, { size: 11, font: 'mono', weight: 700, color: RC(C, r) });
            let xx = x0 + (narrow ? 50 : 62);
            const avail = x1 - xx - (narrow ? 70 : 110);
            const parts = segs.map((q) => ({ n: q.n + (q.sh ? '' : ''), s: q.sh ? q.s / p : q.s, sh: q.sh }));
            const tot = parts.reduce((a, q) => a + q.s, 0);
            parts.forEach((q) => {
              const ww = Math.max(q.sh ? 8 : narrow ? 20 : 26, (q.s / tot) * avail * (q.sh ? 0.93 : 1));
              if (q.sh) V.box(g, xx, yy, ww - 2, 24, { r: 3, fill: V.alpha(RC(C, r), 0.5), stroke: RC(C, r), lw: 1 });
              else { V.box(g, xx, yy, ww - 2, 24, { r: 3, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.7), lw: 1 }); hatch(g, xx, yy, ww - 2, 24, V.alpha(C.muted, 0.55), 5, t * 6); }
              if (ww > 20) V.text(g, q.n, xx + ww / 2 - 1, yy + 12.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
              xx += ww;
            });
            V.text(g, '= ' + nf(local), x1, yy + 16, { size: 12, font: 'mono', weight: 700, align: 'right', color: C.ink });
          }
          const vy = ry0 + p * rh + 12;
          const msg = ok
            ? '✓ ' + nf(local) + ' = ' + (heads / p) + ' cabeças × 128 × 3 (q, k, v) + ' + (heads / p) + ' (β) + 128 (f_a) + 128 (g_a). Está em KDA_BF16_LARGE_M_SHAPES_BY_TP, e o código aceita a cópia BF16.' + (p === 3 ? ' Mas o start-tp3.sh zera GLM53_KDA_BF16_LARGE_M por padrão (opt-in explícito), então o TP3 da receita não a usa.' : ' No TP2 da receita ela vem ligada (.env.example).')
            : '✗ ' + nf(local) + ' (cálculo nosso): não há forma qualificada para TP4 na lista {12.576, 8.726}, e o código recusa com RuntimeError “requires TP=2 or TP=3”.';
          const ey = wrap(g, msg, x0, vy + 10, RW, { size: 12, color: ok ? C.e : C.red || C.c, weight: 600, lh: 16 });
          wrap(g, 'f_a e g_a (projeções de posto baixo dos gates) ficam inteiras em todo rank: a forma local não é “global ÷ p”.', x0, ey + 8, RW, { size: 11.5, color: C.muted, lh: 15 });
          return;
        }
        const n = pr.n, al = align(pr.fmt, ST.mode);
        const loc = n / p;
        const isInt = Number.isInteger(loc);
        const ok = isInt && loc % al === 0;
        rLocal.set(isInt ? nf(loc) : nf(loc, 2));
        rVer.set(ok ? '✓ corte limpo' : isInt ? '✗ parte blocos' : '✗ não é inteiro');
        const axis = ST.mode === 'col' ? 'N (saídas) · column-parallel' : 'K (entradas) · row-parallel';
        V.text(g, (narrow ? 'corte: ' : 'dimensão cortada: ') + nf(n) + ' em ' + (narrow ? (ST.mode === 'col' ? 'N (saídas)' : 'K (entradas)') : axis), x0, 22, { size: narrow ? 11 : 12, color: C.muted, font: 'mono' });
        // régua
        const ry = 36, rh = 28;
        for (let r = 0; r < p; r++) { g.fillStyle = V.alpha(RC(C, r), 0.35); g.fillRect(x0 + (RW * r) / p, ry, RW / p, rh); }
        V.box(g, x0, ry, RW, rh, { r: 4, stroke: V.alpha(C.ink, 0.4), lw: 1 });
        let tickEvery = 1;
        if (al > 1) {
          const spx = (RW * al) / n;
          tickEvery = spx >= 4 ? 1 : Math.max(1, Math.ceil(4 / Math.max(spx, 1e-3)));
          g.strokeStyle = V.alpha(C.ink, 0.35); g.lineWidth = 1; g.beginPath();
          for (let k = al * tickEvery; k < n; k += al * tickEvery) { const xx = x0 + (RW * k) / n; g.moveTo(xx, ry + 3); g.lineTo(xx, ry + rh - 3); }
          g.stroke();
        }
        for (let r = 1; r < p; r++) { const xx = x0 + (RW * r) / p; V.line(g, xx, ry - 6, xx, ry + rh + 6, { color: C.c, lw: 2.2 }); }
        V.text(g, al > 1 ? (tickEvery > 1 ? '1 traço = ' + tickEvery + ' blocos de ' + al : 'traços = blocos de ' + al) : 'sem blocos ao longo deste eixo', x0, ry + rh + 17, { size: 11, font: 'mono', color: C.muted });
        if (narrow) V.text(g, BLK[pr.fmt], x0, ry + rh + 32, { size: 11, font: 'mono', color: C.muted, maxW: RW });
        else V.text(g, BLK[pr.fmt], x1, ry + rh + 17, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        // rótulos por rank
        for (let r = 0; r < p; r++) {
          const xa = x0 + (RW * r) / p, xb = x0 + (RW * (r + 1)) / p;
          chip(g, 'rank ' + r + ': ' + (isInt ? nf(loc) : nf(loc, 1)), (xa + xb) / 2, ry + rh / 2, RC(C, r), { align: 'center', size: 11, h: 18, fa: 0.85, maxW: xb - xa - 6, color: C.dark ? '#08140e' : '#fff' });
        }
        // lupa
        const cut = loc;
        const B = al > 1 ? al : Math.max(16, Math.round(n / 24));
        const k0 = Math.floor(cut / B) - 2, k1 = k0 + 5;
        const zy = ry + rh + (narrow ? 84 : 56), zh = narrow ? 72 : 90;
        const zx0 = x0 + (narrow ? 0 : 60), zx1 = x1 - (narrow ? 0 : 60), ZW = zx1 - zx0;
        const cx = x0 + (RW * 1) / p;
        g.strokeStyle = V.alpha(C.c, 0.5); g.lineWidth = 1; g.setLineDash([3, 3]);
        // no layout estreito as guias da lupa começam abaixo das duas legendas, para não riscá-las
        const gy0 = ry + rh + (narrow ? 40 : 4);
        g.beginPath(); g.moveTo(cx - 6, gy0); g.lineTo(zx0, zy - 4); g.moveTo(cx + 6, gy0); g.lineTo(zx1, zy - 4); g.stroke(); g.setLineDash([]);
        V.dot(g, cx, ry + rh + 2, 4, C.c);
        const e0 = k0 * B, e1 = k1 * B, span = e1 - e0;
        const px = (e) => zx0 + ((e - e0) / span) * ZW;
        let straddle = false;
        for (let k = k0; k < k1; k++) {
          const a = k * B, b = (k + 1) * B;
          if (b <= 0 || a >= n) continue;
          const xa = px(a) + 2, xb = px(b) - 2;
          const rA = Math.floor((a + 0.5) / loc), rB = Math.floor((b - 0.5) / loc);
          const split = al > 1 && rA !== rB;
          if (split) straddle = true;
          if (split) {
            const xc = px(cut);
            g.fillStyle = V.alpha(RC(C, rA), 0.5); g.fillRect(xa, zy, xc - xa, zh);
            g.fillStyle = V.alpha(RC(C, Math.min(p - 1, rB)), 0.5); g.fillRect(xc, zy, xb - xc, zh);
            const pulse = 0.6 + 0.4 * Math.sin(t * 4);
            V.box(g, xa, zy, xb - xa, zh, { r: 5, stroke: V.alpha(C.red || C.c, pulse), lw: 2.6 });
          } else V.box(g, xa, zy, xb - xa, zh, { r: 5, fill: V.alpha(RC(C, Math.min(p - 1, rA)), 0.45), stroke: RC(C, Math.min(p - 1, rA)), lw: 1.2 });
          if (al > 1) {
            chip(g, pr.fmt === 'exl3' ? (xb - xa > 60 ? 'tile' : 't') : 's', (xa + xb) / 2, zy + 14, split ? C.red || C.c : C.ink, { size: 11, h: 18, fa: 0.1, align: 'center' });
            V.text(g, '[' + nf(a) + ', ' + nf(b) + ')', (xa + xb) / 2, zy + zh - 10, { size: 11, font: 'mono', align: 'center', color: C.ink, maxW: xb - xa - 4 });
          } else V.text(g, 'sem escala neste eixo', (xa + xb) / 2, zy + zh / 2, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: xb - xa - 4 });
        }
        V.line(g, px(cut), zy - 8, px(cut), zy + zh + 8, { color: C.c, lw: 2.4, dash: [5, 3], dashOffset: -t * 12 });
        V.text(g, 'corte em ' + (isInt ? nf(cut) : nf(cut, 2)), px(cut), zy - 12, { size: 11, font: 'mono', align: 'center', color: C.c, weight: 700 });
        let vy = zy + zh + 26;
        if (straddle) { wrap(g, 'O bloco vermelho tem uma escala só, mas metade dos seus valores iria para cada rank.', x0, vy, RW, { size: 11.5, color: C.red || C.c, lh: 15 }); vy += narrow ? 32 : 18; }
        // vetores EXL3
        if (pr.fmt === 'exl3') {
          const sy = vy + 8, sh = 14;
          const lw = narrow ? 118 : 190;
          const rows2 = [
            { n: 'suh (tamanho K, entrada)', split: ST.mode === 'row' },
            { n: 'svh (tamanho N, saída)', split: ST.mode === 'col' },
          ];
          rows2.forEach((q, k) => {
            const yy = sy + k * (sh + 10);
            V.text(g, q.n, x0, yy + sh - 2, { size: 11, font: 'mono', color: C.ink, maxW: lw - 6 });
            const bx = x0 + lw, bw2 = RW - lw - (narrow ? 64 : 110);
            if (q.split) {
              for (let r = 0; r < p; r++) { g.fillStyle = V.alpha(RC(C, r), 0.55); g.fillRect(bx + (bw2 * r) / p, yy, bw2 / p - 1, sh); }
            } else { V.box(g, bx, yy, bw2, sh, { r: 3, fill: V.alpha(C.muted, 0.12) }); hatch(g, bx, yy, bw2, sh, V.alpha(C.muted, 0.6), 5, t * 6); }
            V.text(g, q.split ? 'fatiado' : 'replicado', bx + bw2 + 8, yy + sh - 2, { size: 11, font: 'mono', weight: 700, color: q.split ? C.c : C.muted });
          });
          vy = sy + 2 * (sh + 10) + 8;
          wrap(g, 'bits efetivos: expert [4096×2048] inteiro 4,0117 bpw; fatia TP2 de gate/up [4096→1024] 4,0196 bpw, porque carrega o suh inteiro.', x0, vy + 6, RW, { size: 11, color: C.muted, lh: 14 });
          vy += narrow ? 44 : 30;
        }
        const msg = (ok ? '✓ ' : '✗ ') + (isInt ? nf(n) + ' / ' + p + ' = ' + nf(loc) + (al > 1 ? (ok ? ' = ' + nf(loc / al) + ' × ' + al : ' → ' + nf(loc / al, 2) + ' blocos de ' + al) : '') : nf(n) + ' / ' + p + ' = ' + nf(loc, 2) + ' não é inteiro') + '. ' + note(ST.pre, p, ST.mode, ok);
        wrap(g, msg, x0, vy + 10, RW, { size: 12, weight: 600, color: ok ? C.e : C.red || C.c, lh: 16 });
      });
    })();

    /* ================== F5 — tokens, experts e duas GPUs ==================== */
    (function () {
      const fig = root.querySelector('#fig-ep');
      if (!fig) return;
      adaptH(fig, (w) => (w < 620 ? 610 : 470));
      const E = 16;
      const ST = { mode: 'ep', B: 3, k: 2, seed: 5 };
      let route = [];
      const reroute = () => { const r = V.rng(ST.seed); route = Array.from({ length: 32 }, () => { const a = Array.from({ length: E }, (_, i) => i); for (let i = E - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }); };
      reroute();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'tp', label: 'TP no expert' }, { v: 'ep', label: 'EP, DP=1' }, { v: 'epdp', label: 'EP + DP' }], value: 'ep', onChange: (v) => { ST.mode = v; } });
      UI.slider(ctl, { label: 'tokens no lote', min: 1, max: 32, step: 1, value: 3, onInput: (v) => { ST.B = v; } });
      UI.slider(ctl, { label: 'top-k', min: 1, max: 4, step: 1, value: 2, onInput: (v) => { ST.k = v; } });
      UI.button(ctl, '🎲 re-rotear', () => { ST.seed += 7; reroute(); });
      const owner = (e) => (e < 8 ? 0 : 1);
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), m = ST.mode, B = ST.B, k = ST.k;
        const narrow = w < 620, pad = 12;
        const pairs = [];
        for (let i = 0; i < B; i++) for (let j = 0; j < k; j++) pairs.push([i, route[i][j]]);
        const home = (i) => (m === 'epdp' ? i % 2 : -1);
        const touched = new Set(pairs.map((q) => q[1]));
        const cnt = new Array(E).fill(0); pairs.forEach(([, e]) => cnt[e]++);
        let readA, readB;
        if (m === 'tp') { readA = readB = touched.size / 2; } else { readA = [...touched].filter((e) => e < 8).length; readB = touched.size - readA; }
        const cross = m === 'epdp' ? pairs.filter(([i, e]) => owner(e) !== home(i)).length : 0;
        // geometria
        const metW = narrow ? 0 : Math.min(250, w * 0.3);
        const metH = narrow ? 190 : 0;
        const gx = pad, gw = w - 2 * pad - (narrow ? 0 : metW + 14);
        const gh = (h - 2 * pad - metH - 10) / 2;
        const G = [{ x: gx, y: pad, w: gw, h: gh }, { x: gx, y: pad + gh + 10, w: gw, h: gh }];
        const tokW = narrow ? 64 : 88;
        const epos = [[], []], tpos = [[], []];
        G.forEach((P, r) => {
          const col = RC(C, r);
          V.box(g, P.x, P.y, P.w, P.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.6), lw: 1.2 });
          chip(g, 'GPU ' + (r ? 'B' : 'A') + ' · rank ' + r, P.x + 10, P.y + 15, col, { size: 11, h: 19 });
          // tokens desta GPU
          const mine = [];
          for (let i = 0; i < B; i++) if (m !== 'epdp' || home(i) === r) mine.push(i);
          const tx0 = P.x + 10, ty0 = P.y + 32, tAreaH = P.h - 42;
          const per = Math.max(1, Math.floor(tAreaH / 14));
          const ncol = Math.ceil(mine.length / per) || 1;
          const tr = Math.max(3.5, Math.min(10, Math.min(tAreaH / (Math.min(mine.length, per) * 2.4), (tokW - 16) / (ncol * 2.4))));
          mine.forEach((i, q) => {
            const cI = Math.floor(q / per), rI = q % per;
            const rowsHere = Math.min(per, mine.length - cI * per);
            const x = tx0 + tr + cI * tr * 2.5;
            const y = ty0 + tAreaH / 2 + (rI - (rowsHere - 1) / 2) * tr * 2.5;
            tpos[r][i] = [x, y];
          });
          V.text(g, m === 'epdp' ? (narrow ? 'próprios' : 'tokens próprios') : (narrow ? 'réplica' : 'tokens (réplica)'), tx0, P.y + P.h - 6, { size: 11, font: 'mono', color: C.muted, maxW: tokW - 8 });
          // experts (só geometria; o desenho vem depois das linhas)
          const ex0 = P.x + tokW, exW = P.w - tokW - 10;
          const cs = Math.max(12, Math.min(46, (exW - 7 * 5) / 8, (P.h - 50) / 2 - 6));
          const gridW = 8 * cs + 7 * 5;
          const gx0 = ex0 + (exW - gridW) / 2, gy0 = P.y + 30 + (P.h - 40 - (2 * cs + 8)) / 2;
          for (let e = 0; e < E; e++) {
            const cI = e % 8, rI = Math.floor(e / 8);
            const x = gx0 + cI * (cs + 5), y = gy0 + rI * (cs + 8);
            epos[r][e] = [x + cs / 2, y + cs / 2, x, y, cs];
          }
        });
        // linhas token → expert
        const lines = [];
        pairs.forEach(([i, e], q) => {
          if (m === 'tp') { [0, 1].forEach((r) => lines.push([tpos[r][i], epos[r][e], RC(C, r), false, q])); }
          else if (m === 'ep') { const r = owner(e); lines.push([tpos[r][i], epos[r][e], RC(C, r), false, q]); }
          else { const hr = home(i), r = owner(e); lines.push([tpos[hr][i], epos[r][e], hr === r ? RC(C, r) : C.c, hr !== r, q]); }
        });
        const la = B * k > 40 ? 0.22 : 0.45;
        lines.forEach(([a, b, col, cr]) => { if (!a || !b) return; V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(col, cr ? 0.75 : la), lw: cr ? 1.6 : 1 }); });
        [0, 1].forEach((r) => {
          const col = RC(C, r);
          for (let e = 0; e < E; e++) {
            const [, , x, y, cs] = epos[r][e];
            const act = touched.has(e);
            const own = m === 'tp' ? 'half' : owner(e) === r ? 'full' : 'ghost';
            if (act && own !== 'ghost') V.glow(g, x + cs / 2, y + cs / 2, cs * 0.95, C.d, 0.55 + 0.25 * Math.sin(t * 4 + e));
            V.box(g, x, y, cs, cs, { r: 5, fill: C.bg || C.panel });
            if (own === 'full') V.box(g, x, y, cs, cs, { r: 5, fill: V.alpha(col, act ? 0.8 : 0.35), stroke: col, lw: act ? 2 : 1 });
            else if (own === 'half') {
              V.box(g, x, y, cs, cs, { r: 5, fill: V.alpha(C.bg || C.panel, 0.9), stroke: V.alpha(col, 0.7), lw: act ? 2 : 1 });
              g.save(); V.rr(g, x, y, cs, cs, 5); g.clip(); g.fillStyle = V.alpha(col, act ? 0.8 : 0.35); g.fillRect(x + (r ? cs / 2 : 0), y, cs / 2, cs); g.restore();
            } else { V.box(g, x, y, cs, cs, { r: 5, fill: V.alpha(C.bg || C.panel, 0.85) }); g.setLineDash([3, 3]); V.box(g, x, y, cs, cs, { r: 5, stroke: V.alpha(C.muted, 0.45), lw: 1 }); g.setLineDash([]); }
            V.text(g, String(e), x + cs / 2, y + cs / 2 + 0.5, { size: Math.min(12, Math.max(11, cs * 0.42)), font: 'mono', weight: 700, align: 'center', base: 'middle', color: own === 'ghost' ? V.alpha(C.muted, 0.7) : C.ink });
            if (act && own !== 'ghost' && cnt[e] > 1 && cs > 30) V.text(g, '×' + cnt[e], x + cs - 3, y + 11, { size: 11, font: 'mono', weight: 700, align: 'right', color: C.ink });
          }
        });
        lines.slice(0, 70).forEach(([a, b, col, cr, q]) => { if (!a || !b) return; const f = (t * 0.55 + q * 0.137) % 1; const fb = f < 0.5 ? f * 2 : 2 - f * 2; V.dot(g, V.lerp(a[0], b[0], fb), V.lerp(a[1], b[1], fb), cr ? 3.2 : 2.4, cr ? C.c : C.d); });
        // tokens por cima
        [0, 1].forEach((r) => {
          tpos[r].forEach((pp, i) => {
            if (!pp) return;
            const rr = Math.max(3.5, B > 20 ? 4.5 : B > 10 ? 6.5 : 9);
            V.dot(g, pp[0], pp[1], rr, V.alpha(C.d, 0.95));
            if (rr >= 6) { const e = rr * 0.34; V.dot(g, pp[0] - e, pp[1] - e * 0.5, rr * 0.16, C.dark ? '#1b1203' : '#fff'); V.dot(g, pp[0] + e, pp[1] - e * 0.5, rr * 0.16, C.dark ? '#1b1203' : '#fff'); if (rr >= 8) { g.strokeStyle = C.dark ? '#1b1203' : '#fff'; g.lineWidth = 1.2; g.beginPath(); g.arc(pp[0], pp[1] + e * 0.2, rr * 0.38, 0.2 * Math.PI, 0.8 * Math.PI); g.stroke(); } }
          });
        });
        // painel de métricas
        const Mx = narrow ? pad : w - pad - metW, My = narrow ? h - pad - metH : pad, Mw = narrow ? w - 2 * pad : metW, Mh = narrow ? metH : h - 2 * pad;
        V.box(g, Mx, My, Mw, Mh, { r: 12, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        let yy = My + 20;
        V.text(g, m === 'tp' ? 'experts lidos (em experts inteiros)' : 'experts inteiros lidos neste passo', Mx + 12, yy, { size: 11.5, weight: 700, color: C.ink, maxW: Mw - 24 });
        yy += 8;
        const maxR = Math.max(readA, readB), mean = (readA + readB) / 2;
        const scale = Math.max(4, touched.size);
        [[readA, 0], [readB, 1]].forEach(([v, r]) => {
          yy += 20;
          V.text(g, 'GPU ' + (r ? 'B' : 'A'), Mx + 12, yy + 4, { size: 11, font: 'mono', weight: 700, color: RC(C, r) });
          const bx = Mx + 58, bw = Mw - 58 - 44;
          V.box(g, bx, yy - 7, bw, 14, { r: 4, fill: V.alpha(C.muted, 0.12) });
          V.box(g, bx, yy - 7, Math.max(2, (v / scale) * bw), 14, { r: 4, fill: V.alpha(RC(C, r), 0.8) });
          V.text(g, (m === 'tp' ? nf(v, 1) : String(v)), Mx + Mw - 12, yy + 4, { size: 11, font: 'mono', weight: 700, align: 'right', color: C.ink });
        });
        yy += 24;
        const imb = mean > 0 ? maxR / mean : 1;
        const lines2 = [
          'o passo espera o maior: ' + (m === 'tp' ? nf(maxR, 1) + ' expert' + (maxR >= 2 ? 's' : '') + ' (' + touched.size + ' metades)' : maxR + ' expert' + (maxR === 1 ? '' : 's') + ' inteiro' + (maxR === 1 ? '' : 's')),
          'desequilíbrio máx/média: ' + nf(imb, 2) + '×',
          m === 'epdp' ? 'all-to-all: ' + cross + ' de ' + pairs.length + ' pares cruzam a rede (ida e volta)' : 'comunicação: 1 all-reduce de ' + B + ' × d (o mesmo do TP)',
        ];
        lines2.forEach((s, q) => { yy = wrap(g, s, Mx + 12, yy + (q ? 4 : 0), Mw - 24, { size: 11, color: q === 1 && imb > 1.3 ? C.red || C.c : q === 2 && m === 'epdp' ? C.c : C.ink, lh: 14 }); });
        yy += 6;
        const tip = m === 'tp' ? 'TP no expert: cada GPU lê metade de todo expert tocado. Equilíbrio perfeito, mas cada expert é partido (blocos e kernels precisam aceitar a metade).' : m === 'ep' ? 'EP com DP=1: tokens replicados, nenhuma seta cruza entre GPUs. Com poucos tokens, os experts escolhidos podem cair todos na mesma GPU.' : 'EP + DP: cada GPU tem os próprios tokens. As setas rosa são tokens que viajam até o dono do expert e voltam.';
        if (!narrow || Mh > 150) wrap(g, tip, Mx + 12, yy + 6, Mw - 24, { size: 11, color: C.muted, lh: 14, maxLines: narrow ? 3 : 7 });
      });
    })();

    /* ================== F6 — esteira do pipeline × mutirão do TP ============ */
    (function () {
      const fig = root.querySelector('#fig-pp');
      if (!fig) return;
      adaptH(fig, (w) => (w < 600 ? 410 : 372));
      const ST = { p: 2, m: 1, c: 0.1 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'nós (p)', min: 2, max: 4, step: 1, value: 2, onInput: (v) => { ST.p = v; } });
      UI.slider(ctl, { label: 'microlotes (m)', min: 1, max: 8, step: 1, value: 1, onInput: (v) => { ST.m = v; } });
      UI.slider(ctl, { label: 'rede no TP', min: 0, max: 40, step: 2, value: 10, fmt: (v) => v + '%', onInput: (v) => { ST.c = v / 100; } });
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), p = ST.p, m = ST.m, c = ST.c;
        const narrow = w < 600, pad = 14;
        const Tpp = m + p - 1, Ttp = m * (1 + c);
        const Tmax = Math.max(Tpp, Ttp);
        const lx = pad + (narrow ? 46 : 70), rx = w - pad;
        const X = (u) => lx + (u / Tmax) * (rx - lx);
        const cyc = Tmax + 2.2;
        const tc = (t * 0.9) % cyc;
        const cols = C.series;
        const rowH = Math.max(18, Math.min(38, (h - (narrow ? 220 : 170)) / (2 * p)));
        const chartH = p * rowH + 22;
        function axis(y0) {
          for (let u = 0; u <= Math.floor(Tmax); u++) { const xx = X(u); V.line(g, xx, y0, xx, y0 + p * rowH, { color: V.alpha(C.muted, 0.18), lw: 1 }); if (!narrow || u % 2 === 0) V.text(g, String(u), xx, y0 + p * rowH + 13, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        }
        function block(x0, x1, y, hh, col, label, hatchCol) {
          const xe = Math.min(x1, X(tc));
          if (xe > x0) { V.box(g, x0 + 1, y + 2, xe - x0 - 2, hh - 4, { r: 4, fill: V.alpha(col, 0.72) }); if (hatchCol) hatch(g, x0 + 1, y + 2, xe - x0 - 2, hh - 4, hatchCol, 5, 0, 1.3); }
          V.box(g, x0 + 1, y + 2, x1 - x0 - 2, hh - 4, { r: 4, stroke: V.alpha(col, 0.8), lw: 1 });
          if (label && x1 - x0 > 22 && xe >= x1 - 1) V.text(g, label, (x0 + x1) / 2, y + hh / 2 + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.dark ? '#08140e' : '#fff' });
        }
        // PP
        const hdr = narrow ? 18 : 0;
        let y0 = 30 + hdr;
        V.text(g, 'Pipeline (PP=' + p + ')', pad, 20, { size: 12.5, weight: 700, color: C.ink });
        const bub = (p - 1) / (m + p - 1);
        V.text(g, 'bolha ' + nf(bub * 100, 0) + '% · 1º resultado em t = ' + p + ' · total ' + Tpp, narrow ? pad : rx, narrow ? 38 : 20, { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.c });
        for (let s = 0; s < p; s++) {
          const y = y0 + s * rowH;
          V.text(g, narrow ? 'est.' + (s + 1) : 'estágio ' + (s + 1), pad, y + rowH / 2 + 4, { size: 11, font: 'mono', color: RC(C, s) });
          // bolhas
          hatch(g, X(0), y + 3, X(s) - X(0), rowH - 6, V.alpha(C.muted, 0.45), 6, 0);
          hatch(g, X(s + m), y + 3, X(Tpp) - X(s + m), rowH - 6, V.alpha(C.muted, 0.45), 6, 0);
          for (let j = 0; j < m; j++) block(X(j + s), X(j + s + 1), y, rowH, cols[j % cols.length], 'μ' + j);
        }
        axis(y0);
        // TP
        y0 = 30 + hdr + chartH + 40 + hdr;
        V.text(g, 'Tensor parallel (TP=' + p + ')', pad, y0 - 10 - hdr, { size: 12.5, weight: 700, color: C.ink });
        V.text(g, '1º resultado em t = ' + nf(1 + c, 2) + ' · total ' + nf(Ttp, 1), narrow ? pad : rx, y0 - 10 + (narrow ? 8 - hdr + 10 : 0), { size: 11, font: 'mono', align: narrow ? 'left' : 'right', color: C.c });
        for (let s = 0; s < p; s++) {
          const y = y0 + s * rowH;
          V.text(g, 'nó ' + (s + 1), pad, y + rowH / 2 + 4, { size: 11, font: 'mono', color: RC(C, s) });
          for (let j = 0; j < m; j++) {
            const a = j * (1 + c);
            block(X(a), X(a + 1), y, rowH, cols[j % cols.length], 'μ' + j);
            if (c > 0) block(X(a + 1), X(a + 1 + c), y, rowH, C.c, '', V.alpha(C.dark ? '#000' : '#fff', 0.35));
          }
        }
        axis(y0);
        // cursor
        const xc = X(Math.min(tc, Tmax));
        V.line(g, xc, 22, xc, y0 + p * rowH + 4, { color: V.alpha(C.d, 0.85), lw: 1.5, dash: [4, 3] });
        const msg = m === 1 ? 'Com 1 microlote (decode de 1 usuário), o PP entrega o resultado em t = ' + p + ' e o TP em t = ' + nf(1 + c, 2) + ': o TP é ' + nf(p / (1 + c), 1) + '× mais rápido.' : 'Com ' + m + ' microlotes: vazão do PP ' + nf(m / Tpp, 2) + ' por unidade de tempo, TP ' + nf(m / Ttp, 2) + '. Aumente m para ver a bolha encolher.';
        wrap(g, msg, pad, y0 + p * rowH + 38, w - 2 * pad, { size: narrow ? 11.5 : 12.5, color: C.ink, weight: 600, lh: 16 });
        V.text(g, 'unidade = T/p (tempo de ler 1/p dos pesos)', rx, h - 6, { size: 11, font: 'mono', align: 'right', color: C.muted });
      });
    })();

    /* ================== F7 — onde mora o KV ================================ */
    (function () {
      const fig = root.querySelector('#fig-kv');
      if (!fig) return;
      adaptH(fig, (w) => (w < 600 ? 690 : 490));
      const ST = { mode: 'mla', ctx: 5, seed: 3 };
      const CTX = [8192, 32768, 131072, 262144, 524288, 1048576];
      const CTXL = ['8K', '32K', '128K', '256K', '512K', '1M'];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Layout', options: [{ v: 'gqa', label: 'GQA (Qwen)' }, { v: 'mla', label: 'MLA (GLM, DS)' }, { v: 'dcp', label: 'DCP=2' }], value: 'mla', onChange: (v) => { ST.mode = v; } });
      UI.slider(ctl, { label: 'contexto', min: 0, max: 5, step: 1, value: 5, fmt: (v) => CTXL[v], onInput: (v) => { ST.ctx = v; } });
      UI.button(ctl, '🎲 novos scores', () => { ST.seed += 5; lse = mkLse(); });
      function mkLse() {
        const r = V.rng(ST.seed);
        const s = Array.from({ length: 6 }, () => Math.round((r() * 4 - 1) * 10) / 10);
        const v = Array.from({ length: 6 }, () => Math.round((r() * 4 - 2) * 10) / 10);
        const part = (ids) => { const mm = Math.max(...ids.map((j) => s[j])); const l = ids.reduce((a, j) => a + Math.exp(s[j] - mm), 0); const o = ids.reduce((a, j) => a + Math.exp(s[j] - mm) * v[j], 0) / l; return { m: mm, l, o }; };
        const A = part([0, 2, 4]), B = part([1, 3, 5]), F = part([0, 1, 2, 3, 4, 5]);
        const M = Math.max(A.m, B.m);
        const wA = Math.exp(A.m - M) * A.l, wB = Math.exp(B.m - M) * B.l;
        return { s, v, A, B, F, o: (wA * A.o + wB * B.o) / (wA + wB) };
      }
      let lse = mkLse();
      const INFO = {
        gqa: { bpt: 6656, txt: 'Qwen (QSA): 2 cabeças de KV → 1 por GPU. KV principal FP8: 13.312 B/token no total, 6.656 B/token por GPU.' },
        mla: { bpt: 3850, txt: 'MLA/MQA: a linha latente vai inteira para os dois ranks. GLM: 656 B/token/camada em cada rank. DeepSeek: ~3,85 KB/token físicos, iguais nos dois (17,04 GiB = 2.331.430 tokens).' },
        dcp: { bpt: 1925, txt: 'DeepSeek com DCP=2 (não ligado nesta imagem): cada rank guarda metade dos tokens → ~2× capacidade, ao preço de ~43 coletivas a mais por passo.' },
      };
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), m = ST.mode;
        const narrow = w < 600, pad = 14;
        const cap = m === 'mla' ? 12 : 24;
        const cyc = cap + 8;
        const nTok = Math.min(cap, Math.floor((t * 2.4) % cyc));
        const topH = narrow ? 330 : 248;
        const pw = (w - 3 * pad) / 2;
        const U = 24, GC = 4, GR = 6;
        [0, 1].forEach((r) => {
          const px = pad + r * (pw + pad), py = 10, ph = topH - (narrow ? 80 : 60);
          const col = RC(C, r);
          V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.6), lw: 1.2 });
          chip(g, (narrow ? 'KV · GPU ' : 'pool de KV · GPU ') + (r ? 'B' : 'A'), px + 10, py + 15, col, { size: 11, h: 19 });
          const gx0 = px + 12, gy0 = py + 32, gw = pw - 24, gh = ph - 42;
          const cw = (gw - (GC - 1) * 4) / GC, chh = (gh - (GR - 1) * 4) / GR;
          // ocupação: lista de entradas (tok, units)
          const ents = [];
          for (let q = 0; q < nTok; q++) {
            if (m === 'gqa') ents.push({ q, u: 1, lbl: narrow ? 't' + q : 't' + q + ' h' + r });
            else if (m === 'mla') ents.push({ q, u: 2, lbl: 't' + q + ' latente' });
            else if (q % 2 === r) ents.push({ q, u: 2, lbl: 't' + q });
          }
          let slot = 0;
          for (let s2 = 0; s2 < U; s2++) {
            const cI = s2 % GC, rI = Math.floor(s2 / GC);
            V.box(g, gx0 + cI * (cw + 4), gy0 + rI * (chh + 4), cw, chh, { r: 4, stroke: V.alpha(C.muted, 0.3), lw: 1 });
          }
          ents.forEach((e, idx) => {
            if (slot + e.u > U) return;
            const cI = slot % GC, rI = Math.floor(slot / GC);
            const ww = e.u * cw + (e.u - 1) * 4;
            const fresh = idx === ents.length - 1 && nTok < cap;
            if (fresh) V.glow(g, gx0 + cI * (cw + 4) + ww / 2, gy0 + rI * (chh + 4) + chh / 2, 26, C.d, 0.8);
            V.box(g, gx0 + cI * (cw + 4), gy0 + rI * (chh + 4), ww, chh, { r: 4, fill: V.alpha(col, m === 'mla' ? 0.5 : 0.55), stroke: col, lw: 1 });
            if (m === 'mla') hatch(g, gx0 + cI * (cw + 4) + 1, gy0 + rI * (chh + 4) + 1, ww - 2, chh - 2, V.alpha(C.ink, 0.12), 6, 0);
            V.text(g, e.lbl, gx0 + cI * (cw + 4) + ww / 2, gy0 + rI * (chh + 4) + chh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: ww - 4 });
            slot += e.u;
          });
        });
        // contador
        const tokFit = cap;
        // contador: quebra em linhas no estreito para não cortar na borda
        const y1 = 10 + topH - (narrow ? 80 : 60) + 20;
        const cntTxt = 'tokens guardados: ' + nTok + ' · capacidade do par: ' + tokFit + ' tokens' + (m === 'mla' ? (narrow ? ' (= 1 GPU sozinha)' : ' (igual a uma GPU sozinha)') : '');
        const y2 = wrap(g, cntTxt, pad, y1, w - 2 * pad, { size: narrow ? 11 : 11.5, font: 'mono', weight: 700, color: m === 'mla' ? C.red || C.c : C.e, lh: 15 });
        const bpt = INFO[m].bpt, ctx = CTX[ST.ctx];
        const per = bpt * ctx;
        wrap(g, '1 sequência de ' + CTXL[ST.ctx] + ': ' + gib(per) + ' por rank' + (m === 'gqa' ? ' (Qwen, só KV principal)' : ' (DeepSeek, físico)'), pad, y2 + 1, w - 2 * pad, { size: 11, font: 'mono', color: C.muted, lh: 15 });
        // painel LSE
        const ly = topH + 6, lh = h - ly - 8;
        const on = m === 'dcp';
        V.box(g, pad, ly, w - 2 * pad, lh, { r: 12, fill: V.alpha(C.c, on ? 0.06 : 0.03), stroke: V.alpha(C.c, on ? 0.6 : 0.25), lw: 1.2 });
        g.globalAlpha = on ? 1 : 0.55;
        V.text(g, on ? 'junção exata do DCP (log-sum-exp)' : 'o que o DCP faria: junção exata por log-sum-exp', pad + 12, ly + 20, { size: 12, weight: 700, color: C.ink });
        const L = lse;
        const bx = pad + 14, bw = narrow ? w - 2 * pad - 28 : (w - 2 * pad) * 0.42;
        const colW = bw / 6;
        const base = ly + (narrow ? 112 : 118);
        L.s.forEach((s, j) => {
          const x = bx + j * colW + colW * 0.18, ww = colW * 0.64;
          const hh = Math.max(3, ((s + 1.2) / 4.4) * (narrow ? 58 : 66));
          const col = RC(C, j % 2);
          V.box(g, x, base - hh, ww, hh, { r: 3, fill: V.alpha(col, 0.7) });
          V.text(g, nf(s, 1), x + ww / 2, base - hh - 5, { size: 11, font: 'mono', align: 'center', color: C.ink });
          V.text(g, 't' + j, x + ww / 2, base + 13, { size: 11, font: 'mono', align: 'center', color: RC(C, j % 2), weight: 700 });
          V.text(g, 'v=' + nf(L.v[j], 1), x + ww / 2, base + 27, { size: 11, font: 'mono', align: 'center', color: C.muted });
        });
        V.text(g, 'scores s (altura) e valores v', bx, ly + 38, { size: 11, font: 'mono', color: C.muted });
        const tx = narrow ? bx : bx + bw + 24, ty0 = narrow ? base + 50 : ly + 44;
        const fs = narrow ? 11 : 12;
        const rowsL = [
          ['A (t0,t2,t4)', 'm=' + nf(L.A.m, 2) + '  ℓ=' + nf(L.A.l, 3) + '  o=' + nf(L.A.o, 4), C.a],
          ['B (t1,t3,t5)', 'm=' + nf(L.B.m, 2) + '  ℓ=' + nf(L.B.l, 3) + '  o=' + nf(L.B.o, 4), C.b],
          ['junção', 'o = ' + nf(L.o, 4), C.c],
          ['atenção cheia', 'o = ' + nf(L.F.o, 4) + (Math.abs(L.o - L.F.o) < 1e-9 ? '  ✓ idêntico' : ''), C.e],
        ];
        rowsL.forEach(([k, v, col], q) => {
          const yy = ty0 + q * (fs + 9);
          V.text(g, k, tx, yy, { size: fs, font: 'mono', weight: 700, color: col });
          V.text(g, v, tx + (narrow ? 112 : 128), yy, { size: fs, font: 'mono', color: C.ink, maxW: w - pad - 14 - tx - (narrow ? 112 : 128) });
        });
        g.globalAlpha = 1;
        wrap(g, INFO[m].txt, pad + 12, h - (narrow ? 66 : 30), w - 2 * pad - 24, { size: 11, color: C.muted, lh: 14 });
      });
    })();

    /* ================== F8 — top-k distribuído ============================= */
    (function () {
      const fig = root.querySelector('#fig-topk');
      if (!fig) return;
      const N = 24;
      const ST = { p: 2, k: 3, seed: 9, argmax: false };
      // geometria da mesa: chips (score, id) em linhas, sempre legíveis (≥ 11 px)
      const geo = (w) => {
        const narrow = w < 600, pad = 14;
        const kk = ST.argmax ? 1 : ST.k, n = ST.p * kk;
        const cw = narrow ? 50 : 62;
        const perRow = Math.max(1, Math.floor((w - 2 * pad - 20 + 4) / (cw + 4)));
        const nRows = Math.max(1, Math.ceil(n / perRow));
        const baseY = narrow ? 200 : 170;
        return { narrow, pad, cw, perRow, nRows, baseY, mesaH: 30 + nRows * 24 + 6 };
      };
      const reH = adaptH(fig, (w) => { const G0 = geo(w); return G0.baseY + 70 + G0.mesaH + (G0.narrow ? 96 : 70); });
      let sc = [];
      const regen = () => { const r = V.rng(ST.seed); const used = new Set(); sc = Array.from({ length: N }, () => { let v; do { v = Math.round(r() * 99) + 1; } while (used.has(v)); used.add(v); return v; }); };
      regen();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'ranks', options: [2, 3, 4], value: 2, onChange: (v) => { ST.p = v; if (reH) reH(); } });
      UI.slider(ctl, { label: 'k', min: 1, max: 6, step: 1, value: 3, onInput: (v) => { ST.k = v; if (reH) reH(); } });
      UI.toggle(ctl, { label: 'modo argmax (vocab-parallel)', value: false, onChange: (v) => { ST.argmax = v; if (reH) reH(); } });
      UI.button(ctl, '🎲 novos scores', () => { ST.seed += 3; regen(); });
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), p = ST.p, k = ST.argmax ? 1 : ST.k;
        const G0 = geo(w), narrow = G0.narrow, pad = G0.pad;
        const per = N / p;
        const P = 8, ph = t % P;
        const x0 = pad, x1 = w - pad, bw = (x1 - x0) / N;
        const baseY = G0.baseY, maxH = narrow ? 120 : 110;
        const allVals = bw > 22, allIds = bw > 16;
        const loc = [];
        for (let r = 0; r < p; r++) { const ids = Array.from({ length: per }, (_, q) => r * per + q); ids.sort((a, b) => sc[b] - sc[a] || a - b); loc.push(ids.slice(0, k)); }
        const glob = Array.from({ length: N }, (_, i) => i).sort((a, b) => sc[b] - sc[a] || a - b).slice(0, k);
        const cand = loc.flat().sort((a, b) => sc[b] - sc[a] || a - b);
        const merged = cand.slice(0, k);
        const same = merged.every((id, q) => id === glob[q]);
        V.text(g, ST.argmax ? (narrow ? N + ' logits em ' + p + ' faixas de ids' : 'logits de ' + N + ' ids, em ' + p + ' faixas contíguas (lm_head vocab-parallel)') : (narrow ? N + ' scores em ' + p + ' fatias (SP indexer)' : N + ' scores de chaves em ' + p + ' fatias contíguas (SP indexer)'), pad, 18, { size: narrow ? 11 : 12, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        for (let r = 0; r < p; r++) {
          const xa = x0 + r * per * bw, xb = xa + per * bw;
          V.box(g, xa + 1, baseY - maxH - 20, xb - xa - 2, maxH + 36, { r: 8, fill: V.alpha(RC(C, r), 0.05), stroke: V.alpha(RC(C, r), 0.45), lw: 1 });
          const rl = (narrow && p > 2 ? 'r' + r : 'rank ' + r) + ' · ids ' + (r * per) + '–' + ((r + 1) * per - 1);
          V.text(g, rl, (xa + xb) / 2, baseY + 32, { size: 11, font: 'mono', weight: 700, align: 'center', color: RC(C, r), maxW: xb - xa - 6 });
        }
        const showLocal = ph > 1.2, fly = V.clamp((ph - 3) / 1.6, 0, 1), mergedOn = ph > 4.8;
        const mesaY = baseY + 70, mesaH = G0.mesaH;
        V.box(g, x0, mesaY, x1 - x0, mesaH, { r: 10, fill: V.alpha(C.c, 0.05), stroke: V.alpha(C.c, 0.5), lw: 1.2 });
        V.text(g, (narrow ? 'mesa (all-gather de ' : 'mesa comum (all-gather de ') + p + ' × ' + k + ' pares ' + (ST.argmax ? '(valor, id)' : '(score, id)') + ')', x0 + 10, mesaY + 16, { size: 11, font: 'mono', color: C.c, weight: 700, maxW: x1 - x0 - 20 });
        sc.forEach((v, i) => {
          const r = Math.floor(i / per);
          const x = x0 + i * bw + bw * 0.15, ww = bw * 0.7;
          const hh = (v / 100) * maxH;
          const isLoc = showLocal && loc[r].includes(i);
          const isGlob = glob.includes(i);
          const dim = mergedOn && !isGlob;
          V.box(g, x, baseY - hh, ww, hh, { r: 3, fill: V.alpha(RC(C, r), dim ? 0.2 : isLoc ? 0.9 : 0.45), stroke: isLoc ? RC(C, r) : null, lw: 1.5 });
          // no estreito, só as barras coroadas mostram valor e id (o resto fica ilegível de tão fino)
          const showVal = allVals || isLoc, showId = allIds || isLoc;
          if (showVal) V.text(g, String(v), x + ww / 2, baseY - hh - 4, { size: 11, font: 'mono', align: 'center', color: isLoc ? C.ink : C.muted, weight: isLoc ? 700 : 400, maxW: bw - 1 });
          if (isLoc) V.text(g, '♛', x + ww / 2, baseY - hh - 18, { size: 12, align: 'center', color: C.d });
          if (isGlob && mergedOn) V.text(g, '★', x + ww / 2, baseY - maxH - 26, { size: 13, align: 'center', color: C.e });
          if (showId) V.text(g, String(i), x + ww / 2, baseY + 14, { size: 11, font: 'mono', align: 'center', color: isLoc ? RC(C, r) : C.muted, weight: isLoc ? 700 : 400, maxW: bw - 1 });
        });
        // candidatos voando para a mesa, em linhas de chips (score, id)
        if (showLocal && fly > 0) {
          const cw = G0.cw, perRow = G0.perRow;
          const flat = loc.flat();
          cand.forEach((id, q) => {
            const r = Math.floor(id / per);
            const sx = x0 + id * bw + bw / 2, sy = baseY - (sc[id] / 100) * maxH;
            const ord = mergedOn ? q : flat.indexOf(id);
            const tx = x0 + 10 + (ord % perRow) * (cw + 4) + cw / 2, ty = mesaY + 38 + Math.floor(ord / perRow) * 24;
            const f = V.ease.inOut(fly);
            const x = V.lerp(sx, tx, f), y = V.lerp(sy, ty, f);
            const win = mergedOn && merged.includes(id);
            V.box(g, x - cw / 2, y - 10, cw, 20, { r: 6, fill: V.alpha(RC(C, r), win ? 0.85 : mergedOn ? 0.18 : 0.5), stroke: win ? C.e : RC(C, r), lw: win ? 2 : 1 });
            V.text(g, '(' + sc[id] + ',' + id + ')', x, y + 0.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, maxW: cw - 6 });
          });
        }
        const yT = mesaY + mesaH + 20;
        if (mergedOn) V.text(g, same ? '✓ top-' + k + ' da mesa = top-' + k + ' por força bruta (★)' : 'diferente', pad, yT, { size: 12, weight: 700, color: same ? C.e : C.red || C.c, maxW: w - 2 * pad });
        else V.text(g, ph < 1.2 ? '1. cada rank tem só a sua fatia' : ph < 3 ? '2. cada rank escolhe seu top-' + k + ' local (♛)' : '3. os pares viajam para a mesa comum', pad, yT, { size: 12, weight: 700, color: C.ink, maxW: w - 2 * pad });
        const rw = ST.argmax ? 'Qwen e DeepSeek: 2 números por rank por token, em vez de juntar 248.320 ou 129.280 logits.' : 'Trocados: ' + p * k + ' pares, contra ' + N + ' valores da linha inteira. No SP indexer do DeepSeek: p × 512 pares × 8 B por query (TP2: 8 KB).';
        wrap(g, rw, pad, yT + 18, w - 2 * pad, { size: 11, color: C.muted, lh: 14 });
      });
    })();

    /* ============ F8b — três jeitos de fechar o par (receita GLM·TF) ======== */
    (function () {
      const fig = root.querySelector('#fig-fechar');
      if (!fig) return;
      const RS = [1, 8, 16, 32, 64, 256, 1024, 2048];
      const KIB = 1024, MIB = 1048576, GIB = 1073741824, NPTS = 90, MIN_ROWS = 64;
      const ST = { m: 'split', ri: 7, t0: null };
      adaptH(fig, (w) => (w < 600 ? (w < 380 ? 690 : 650) : 520));
      // bytes enviados por rank num ponto de troca (2 ranks, D = 4096); all-reduce contado em anel
      const calc = (m, R) => {
        const H = Math.ceil(R / 2);
        if (m === 'ar') return { send: R * 8 * KIB, glue: R, msgs: 1, rnd: 2 };
        if (m === 'ag') return { send: R * 16 * KIB, glue: R, msgs: 1, rnd: 1 };
        return { send: H * 24 * KIB, glue: H, msgs: 2, rnd: 1 };
      };
      const fmtB = (b) => {
        if (b < MIB) return nf(b / KIB, 0) + ' KiB';
        if (b < GIB) { const v = b / MIB; return nf(v, Number.isInteger(v) ? 0 : 1) + ' MiB'; }
        return nf(b / GIB, 2) + ' GiB';
      };
      const effOf = () => (ST.m === 'split' && RS[ST.ri] < MIN_ROWS ? 'ag' : ST.m);
      let rdA = null, rdB = null;
      const upd = () => { if (!rdA || !rdB) return; const k = calc(effOf(), RS[ST.ri]); rdA.set(fmtB(k.send)); rdB.set(fmtB(k.send * NPTS)); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Fechamento', options: [{ v: 'ar', label: 'all-reduce BF16 (vLLM)' }, { v: 'ag', label: 'all-gather fp32 (TF)' }, { v: 'split', label: 'dividido por linhas (TF)' }], value: 'split', onChange: (v) => { ST.m = v; ST.t0 = null; upd(); } });
      UI.slider(ctl, { label: 'linhas R', min: 0, max: RS.length - 1, step: 1, value: 7, fmt: (v) => nf(RS[v]), onInput: (v) => { ST.ri = v; ST.t0 = null; upd(); } });
      rdA = UI.readout(ctl, 'enviado por rank, por ponto', '—');
      rdB = UI.readout(ctl, 'por passada (90 pontos)', '—');
      upd();
      const C0 = V.C();
      UI.legend(ctl, [[C0.d, 'parcial'], [C0.c, 'cola feita aqui'], [C0.e, 'linha pronta']]);
      let lastC = null, tfc = null;
      const tfCol = (C) => { if (C !== lastC) { lastC = C; tfc = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; } return tfc; };
      const TT = { ar: [0, 1.0, 2.6, 4.2, 6.6], ag: [0, 1.0, 3.0, 4.2, 6.6], split: [0, 1.0, 2.6, 4.0, 5.4, 6.6] };
      const STEPS = {
        ar: ['1. cada rank tem uma parcial BF16 de todas as R linhas', '2. reduce-scatter: meia linha cruza, e cada rank soma a sua metade', '3. all-gather: as metades somadas cruzam de volta', '4. os dois ranks fazem a cola em todas as R linhas'],
        ag: ['1. cada rank tem uma parcial fp32 de todas as R linhas', '2. all-gather: as parciais inteiras cruzam o cabo', '3. cada rank soma as duas, rank 0 primeiro, e arredonda uma vez', '4. os dois ranks fazem a cola em todas as R linhas'],
        split: ['1. cada rank tem uma parcial fp32 de todas as R linhas', '2. cada rank manda ao par só a parcial das linhas do par', '3. cada rank soma as suas linhas (rank 0 primeiro) e faz a cola só nelas', '4. as linhas normalizadas, em BF16, cruzam para o par', '5. os dois ranks têm a entrada inteira do bloco seguinte'],
      };
      const WIRE = { ar: ['', 'meia linha BF16', 'meia linha somada', ''], ag: ['', 'linha fp32 · 16 KiB', '', ''], split: ['', 'parciais do par · fp32', '', 'linhas prontas · BF16', ''] };
      const NAMES = { ar: ['all-reduce BF16', 'vLLM'], ag: ['all-gather fp32 + soma rank 0 primeiro', 'TensorFold, decode'], split: ['cola dividida por linhas', 'TensorFold, prefill (SPLIT=1)'] };
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), TF = tfCol(C), narrow = w < 600, pad = 14;
        const R = RS[ST.ri], m = ST.m, fall = m === 'split' && R < MIN_ROWS, eff = fall ? 'ag' : m;
        if (ST.t0 == null) ST.t0 = t;
        const T = TT[eff], P = T[T.length - 1];
        const ph = (t - ST.t0) % P;
        let s = 0;
        while (s < T.length - 2 && ph >= T[s + 1]) s++;
        const u = V.clamp((ph - T[s]) / (T[s + 1] - T[s]), 0, 1);
        const n = Math.min(R, 16), per = R / n, half = n / 2;
        const own = (i) => (i < half ? 0 : 1);
        // cabeçalho
        V.text(g, NAMES[eff][0] + ' · ' + NAMES[eff][1], pad, 16, { size: 12.5, weight: 700, color: eff === 'ar' ? C.glm : TF, maxW: w - 2 * pad });
        V.text(g, 'R = ' + nf(R) + (R === 1 ? ' linha' : ' linhas') + (per > 1 ? ' · ' + n + ' faixas de ' + nf(per) : ''), pad, 34, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        // geometria das colunas
        const cw = narrow ? Math.floor((w - 2 * pad) * 0.34) : Math.min(250, Math.floor((w - 2 * pad) * 0.3));
        const xs = [pad, w - pad - cw];
        const hdrY = 60, rowsTop = 90, rowsH = narrow ? 168 : 156;
        const bh = Math.min(18, rowsH / n), y0 = rowsTop + (rowsH - bh * n) / 2;
        const bandY = (i) => y0 + i * bh + bh / 2;
        const mx0 = xs[0] + cw, mx1 = xs[1], midX = (mx0 + mx1) / 2;
        // o cabo
        V.box(g, mx0 + 6, rowsTop - 8, mx1 - mx0 - 12, rowsH + 16, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, narrow ? 'cabo' : 'cabo entre os Sparks', midX, hdrY + 2, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: mx1 - mx0 - 8 });
        const wl = (WIRE[eff] || [])[s];
        if (wl) V.text(g, wl, midX, hdrY + 18, { size: 11, font: 'mono', weight: 700, align: 'center', color: s === 3 && eff === 'split' ? C.e : C.d, maxW: mx1 - mx0 - 8 });
        const bandState = (r, i) => {
          if (eff === 'split') {
            const mine = own(i) === r;
            if (s === 0) return [C.d, 0.55];
            if (s === 1) return mine ? [C.d, 0.55] : [C.d, 0.08 + 0.47 * (1 - u)];
            if (s === 2) return mine ? [C.c, 0.3 + 0.5 * u] : [C.muted, 0.12];
            if (s === 3) return mine ? [C.e, 0.6] : (u > 0.85 ? [C.e, 0.6] : [C.muted, 0.12]);
            return [C.e, 0.6];
          }
          if (s < 2) return [C.d, 0.55];
          if (s === 2) return [C.d, 0.85];
          return u < 0.7 ? [C.c, 0.3 + 0.6 * (u / 0.7)] : [C.e, 0.6];
        };
        for (let r = 0; r < 2; r++) {
          const x = xs[r], col = RC(C, r);
          V.box(g, x, hdrY - 14, cw, rowsTop + rowsH - hdrY + 26, { r: 10, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.5), lw: 1 });
          V.text(g, 'Spark ' + r + ' · rank ' + r, x + cw / 2, hdrY + 2, { size: 12, weight: 700, align: 'center', color: col, maxW: cw - 8 });
          let sub = eff === 'ar' ? 'parcial BF16' : 'parcial fp32';
          if (eff === 'split') { const H = R / 2; sub = 'dono de ' + nf(r * H) + '–' + nf(r * H + H - 1); }
          V.text(g, sub, x + cw / 2, hdrY + 18, { size: 11, font: 'mono', align: 'center', color: C.muted, maxW: cw - 8 });
          for (let i = 0; i < n; i++) {
            const [bc, ba] = bandState(r, i);
            const bx = x + 8, bw = cw - 16, bhh = Math.max(2, bh - 2), by = bandY(i) - bhh / 2;
            V.box(g, bx, by, bw, bhh, { r: Math.min(4, bhh / 2), fill: V.alpha(bc, ba), stroke: eff === 'split' ? V.alpha(RC(C, own(i)), 0.75) : null, lw: 1 });
            if (eff === 'ar') V.line(g, bx + bw / 2, by + 1, bx + bw / 2, by + bhh - 1, { color: V.alpha(C.ink, 0.35), lw: 1 });
          }
          const sumOn = (eff !== 'split' && s === 2) || (eff === 'split' && s === 2 && u < 0.5);
          if (sumOn) plus(g, r === 0 ? mx0 + 18 : mx1 - 18, rowsTop + rowsH / 2, 8, C.c, true);
        }
        // pontos cruzando o cabo
        const dots = (from, list, col, rad) => {
          list.forEach((i) => {
            const d = n > 1 ? (i / (n - 1)) * 0.15 : 0;
            const f = V.clamp((u - d) / 0.8, 0, 1);
            if (f <= 0 || f >= 1) return;
            const e = V.ease.inOut(f);
            const xa = from === 0 ? mx0 - 4 : mx1 + 4, xb = from === 0 ? mx1 + 4 : mx0 - 4;
            V.dot(g, V.lerp(xa, xb, e), bandY(i) + (from === 0 ? -1 : 1) * Math.min(3, bh / 4), rad, col);
          });
        };
        const all = Array.from({ length: n }, (_, i) => i);
        if (eff === 'ar' && (s === 1 || s === 2)) { dots(0, all, C.d, 3); dots(1, all, C.d, 3); }
        if (eff === 'ag' && s === 1) { dots(0, all, C.d, 5); dots(1, all, C.d, 5); }
        if (eff === 'split' && s === 1) { dots(0, all.filter((i) => own(i) === 1), C.d, 5); dots(1, all.filter((i) => own(i) === 0), C.d, 5); }
        if (eff === 'split' && s === 3) { dots(0, all.filter((i) => own(i) === 0), C.e, 4); dots(1, all.filter((i) => own(i) === 1), C.e, 4); }
        // passo atual
        const sy0 = rowsTop + rowsH + 34;
        wrap(g, (fall ? 'R < 64, sem divisão: ' : '') + STEPS[eff][s], pad, sy0, w - 2 * pad, { size: 12, weight: 700, color: fall ? TF : C.ink, lh: 16, maxLines: 2 });
        // os três fechamentos lado a lado
        let yy = sy0 + (narrow ? 50 : 38);
        const maxS = calc('ag', R).send;
        const barMaxW = narrow ? w - 2 * pad - 90 : Math.min(420, w * 0.5);
        ['ar', 'ag', 'split'].forEach((k) => {
          const c = calc(k, R), act = k === eff, col = k === 'ar' ? C.glm : TF;
          const off = k === 'split' && R < MIN_ROWS;
          yy = wrap(g, NAMES[k][0] + ' · ' + NAMES[k][1], pad, yy, w - 2 * pad, { size: 11.5, weight: act ? 700 : 500, color: act ? C.ink : C.muted, lh: 14, maxLines: 2 });
          if (off) { yy = wrap(g, 'não usado: abaixo de 64 linhas a receita não divide e fecha o par pelo all-gather acima', pad, yy + 1, w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 2 }) + 8; return; }
          const bw = Math.max(3, (c.send / maxS) * barMaxW);
          V.box(g, pad, yy - 9, bw, 12, { r: 4, fill: V.alpha(col, off ? 0.15 : act ? 0.75 : 0.35), stroke: act ? col : null, lw: 1.2 });
          V.text(g, fmtB(c.send), pad + bw + 8, yy + 1, { size: 11, font: 'mono', weight: 700, color: act ? C.ink : C.muted });
          const det = 'cola em ' + nf(c.glue) + (c.glue === 1 ? ' linha' : ' linhas') + ' por rank · ' + (c.msgs > 1 ? c.msgs + ' trocas' : '1 coletiva') + ' · ' + (c.rnd > 1 ? c.rnd + ' arredondamentos' : '1 arredondamento');
          yy = wrap(g, det, pad, yy + 17, w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 2 }) + 8;
        });
        // rodapé: o transporte e as medições
        let note;
        if (eff === 'ar') note = 'Receita vLLM: PyNCCL entre os nós, 20–50 µs por coletiva nas auditorias das receitas (seção 9). Metade dos bytes do all-gather fp32, com dois arredondamentos.';
        else if (eff === 'split') note = 'Trocas por send/recv do NCCL num segundo stream, sobrepostas à “frente” do bloco seguinte. Medido com 0009 e 0020 juntos: prefill de 50k de cerca de 1.270 para 1.730 tok/s; o decode fica cerca de 1,5% mais lento.';
        else if (R * 16 <= 512) note = nf(R) + ' × 16 KiB = ' + fmtB(R * 16 * KIB) + ' ≤ 512 KiB: vai pelo RDMA de um tiro (TF_ROCE_MAX_KB=512). Medido: 16 KiB em 11 µs, contra 45 µs no NCCL; decode +6%.' + (fall ? ' No decode é sempre este caminho.' : '');
        else note = nf(R) + ' × 16 KiB = ' + fmtB(R * 16 * KIB) + ' > 512 KiB: vai pelo NCCL, nos dois trilhos do cabo. Medido: 32 MB (R = 2.048) em 1,9 ms; com um trilho, 3,5 ms. No prefill, a receita usa o caminho dividido.';
        wrap(g, note, pad, yy + 6, w - 2 * pad, { size: 11, color: C.muted, lh: 14, maxLines: 6 });
      });
    })();

    /* ================== F9 — a conta da rede =============================== */
    (function () {
      const fig = root.querySelector('#fig-rede');
      if (!fig) return;
      adaptH(fig, (w) => (w < 640 ? 560 : 430));
      const PR = {
        qwen: { d: 2560, kspec: 3, nDec: 106, nPre: 96, chunk: 8192, seqs: 1, ref: 'Medido: 2–3 ms/passo. Estimado no repo: 150–180 ms de rede por chunk de 8192.', cnt: 'contagem da auditoria' },
        glm: { d: 4096, kspec: 7, nDec: 90, nPre: 90, chunk: 1024, seqs: 1, ref: 'O repo estima 378 MB e 15–20 ms por 1024 tokens contando 1 all-reduce por camada; aqui contamos 2 (interpretação).', cnt: 'contagem nossa (2 por camada)' },
        ds: { d: 4096, kspec: 6, nDec: 108, nPre: 86, chunk: 1024, seqs: 1, ref: 'Medido: 3–5 ms/passo. 86 all-reduces de 8 MB por chunk ≈ 690 MB por rank.', cnt: 'contagem da auditoria' },
      };
      const CHUNKS = [256, 512, 1024, 2048, 4096, 8192];
      const ST = { pre: 'qwen', a: 25, b: 25, seqs: 1, ci: 5 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { ST.pre = v; sC.set(CHUNKS.indexOf(PR[v].chunk)); } });
      UI.seg(ctl, { label: 'Link', options: [{ v: 'roce', label: 'RoCE 200G' }, { v: 'nvl', label: 'NVLink (ref.)' }], value: 'roce', onChange: (v) => { if (v === 'roce') { sA.set(25); sB.set(25); } else { sA.set(5); sB.set(450); } } });
      const sA = UI.slider(ctl, { label: 'latência', min: 2, max: 60, step: 1, value: 25, fmt: (v) => v + ' µs', onInput: (v) => { ST.a = v; } });
      const sB = UI.slider(ctl, { label: 'banda', min: 5, max: 500, step: 5, value: 25, fmt: (v) => v + ' GB/s', onInput: (v) => { ST.b = v; } });
      UI.slider(ctl, { label: 'seqs no decode', min: 1, max: 8, step: 1, value: 1, onInput: (v) => { ST.seqs = v; } });
      const sC = UI.slider(ctl, { label: 'chunk de prefill', min: 0, max: 5, step: 1, value: 5, fmt: (v) => String(CHUNKS[v]), onInput: (v) => { ST.ci = v; } });
      const fmtT = (us) => (us < 1000 ? nf(us, us < 10 ? 1 : 0) + ' µs' : nf(us / 1000, us < 1e4 ? 2 : 1) + ' ms');
      const fmtB = (b) => (b < 1e6 ? nf(b / 1e3, b < 1e4 ? 1 : 0) + ' KB' : nf(b / 1e6, b < 1e7 ? 1 : 0) + ' MB');
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), P = PR[ST.pre], a = ST.a, b = ST.b;
        const narrow = w < 640, pad = 14;
        const T = (n) => a + (n / (b * 1e9)) * 1e6; // µs, p=2
        const rowsDec = ST.seqs * (1 + P.kspec);
        const nDec = rowsDec * P.d * 2;
        const chunk = CHUNKS[ST.ci];
        const nPre = chunk * P.d * 2;
        const knee = a * b * 1e3;
        // gráfico
        const gx = pad + 40, gy = 16, gw = narrow ? w - pad - gx - 6 : w * 0.54 - gx, gh = narrow ? 230 : h - gy - 44;
        const lx0 = 3, lx1 = 8, ly0 = 0, ly1 = 5;
        const X = (n) => gx + ((Math.log10(n) - lx0) / (lx1 - lx0)) * gw;
        const Y = (us) => gy + gh - ((Math.log10(us) - ly0) / (ly1 - ly0)) * gh;
        V.box(g, gx, gy, gw, gh, { r: 4, stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const kx = V.clamp(X(knee), gx, gx + gw);
        g.fillStyle = V.alpha(C.b, 0.07); g.fillRect(gx, gy, kx - gx, gh);
        g.fillStyle = V.alpha(C.d, 0.07); g.fillRect(kx, gy, gx + gw - kx, gh);
        for (let e = lx0; e <= lx1; e++) { const xx = X(Math.pow(10, e)); V.line(g, xx, gy, xx, gy + gh, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, ['1 KB', '10 KB', '100 KB', '1 MB', '10 MB', '100 MB'][e - 3], xx, gy + gh + 15, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        for (let e = ly0; e <= ly1; e++) { const yy = Y(Math.pow(10, e)); V.line(g, gx, yy, gx + gw, yy, { color: V.alpha(C.muted, 0.15), lw: 1 }); V.text(g, ['1 µs', '10 µs', '100 µs', '1 ms', '10 ms', '100 ms'][e], gx - 5, yy + 4, { size: 11, font: 'mono', align: 'right', color: C.muted }); }
        g.beginPath();
        for (let q = 0; q <= 120; q++) { const lg = lx0 + (q / 120) * (lx1 - lx0), n = Math.pow(10, lg); const yy = Y(V.clamp(T(n), 1, 1e5)); if (q === 0) g.moveTo(X(n), yy); else g.lineTo(X(n), yy); }
        g.strokeStyle = C.a; g.lineWidth = 2.5; g.stroke();
        V.line(g, kx, gy, kx, gy + gh, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [4, 4] });
        V.text(g, 'joelho αβ = ' + fmtB(knee), kx + 4, gy + 14, { size: 11, font: 'mono', color: C.ink });
        V.text(g, 'latência domina', gx + 6, gy + gh - 8, { size: 11, font: 'mono', color: C.b });
        V.text(g, 'banda domina', gx + gw - 6, gy + gh - 8, { size: 11, font: 'mono', color: C.d, align: 'right' });
        const mk = (n, col, lbl, up) => { const xx = X(V.clamp(n, 1e3, 1e8)), yy = Y(V.clamp(T(n), 1, 1e5)); V.glow(g, xx, yy, 14, col, 0.8 + 0.2 * Math.sin(t * 3)); V.dot(g, xx, yy, 5, col); const right = up || xx > gx + gw * 0.62; V.text(g, lbl, xx + (right ? -8 : 8), yy + (up ? -10 : 18), { size: 11, font: 'mono', weight: 700, color: col, align: right ? 'right' : 'left' }); };
        mk(nDec, C.c, 'decode ' + fmtB(nDec), X(nDec) > gx + 90);
        mk(nPre, C.d, 'prefill ' + fmtB(nPre), false);
        V.text(g, narrow ? 'all-reduce, 2 ranks: T(n) = α + n/β' : 'tempo de 1 all-reduce entre 2 ranks: T(n) = α + n/β', gx, gy + gh + 32, { size: 11, font: 'mono', color: C.muted, maxW: gw });
        // painel
        const px = narrow ? pad : w * 0.54 + 16, py = narrow ? gy + gh + 44 : 14, pw = narrow ? w - 2 * pad : w - px - pad, phh = narrow ? h - py - 10 : h - 28;
        V.box(g, px, py, pw, phh, { r: 12, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        let yy = py + 22;
        const tDec = T(nDec), tPre = T(nPre);
        const totDec = P.nDec * tDec, totPre = P.nPre * tPre;
        V.text(g, 'decode', px + 12, yy, { size: 12.5, weight: 700, color: C.c });
        yy = wrap(g, rowsDec + ' linhas (' + ST.seqs + ' seq × (1 + ' + P.kspec + ' rascunhos)) × ' + P.d + ' × 2 B = ' + fmtB(nDec) + ' por coletiva', px + 12, yy + 17, pw - 24, { size: 11, color: C.ink, lh: 14 });
        yy = wrap(g, P.nDec + ' coletivas × ' + fmtT(tDec) + ' = ' + fmtT(totDec) + ' por passo', px + 12, yy + 2, pw - 24, { size: 12, weight: 700, font: 'mono', color: C.ink, lh: 15 });
        const latShare = a / tDec;
        yy = wrap(g, nf(latShare * 100, 0) + '% desse tempo é latência (α) · ' + P.cnt, px + 12, yy + 2, pw - 24, { size: 11, color: C.muted, lh: 14 });
        yy += 10;
        V.text(g, 'prefill', px + 12, yy + 4, { size: 12.5, weight: 700, color: C.d });
        yy = wrap(g, 'chunk ' + chunk + ' × ' + P.d + ' × 2 B = ' + fmtB(nPre) + ' por all-reduce', px + 12, yy + 21, pw - 24, { size: 11, color: C.ink, lh: 14 });
        yy = wrap(g, P.nPre + ' × ' + fmtT(tPre) + ' = ' + fmtT(totPre) + ' por chunk (' + fmtB(P.nPre * nPre) + ' por rank)', px + 12, yy + 2, pw - 24, { size: 12, weight: 700, font: 'mono', color: C.ink, lh: 15 });
        yy += 8;
        wrap(g, P.ref + (b > 100 ? ' O preset NVLink é ilustrativo.' : ''), px + 12, yy + 6, pw - 24, { size: 11, color: C.muted, lh: 14 });
      });
    })();

    /* ================== F10 — a planta de cada GPU ========================== */
    (function () {
      const fig = root.querySelector('#fig-planta');
      if (!fig) return;
      const K = { S: 'fatiado', R: 'replicado', E: 'EP', X: 'misto', N: 'desligado', W: 'inteiro' };
      const PL = {
        glm: {
          std: { ranks: 2, rows: [
            ['embedding', 'S', ['ids 0–77.439', 'ids 77.440–154.879']],
            ['MLA (11 cam.): q_b', 'S', ['32 cabeças', '32 cabeças']],
            ['latente c_kv + indexador', 'R', ['inteiro', 'inteiro']],
            ['KDA (34 cam.): in_proj', 'X', ['12.576 × 4096', '12.576 × 4096']],
            ['KDA: estado recorrente', 'S', ['32 cabeças', '32 cabeças']],
            ['KDA: o_proj', 'S', ['row · 4096 × 4096', 'row · 4096 × 4096']],
            ['MLA: o_proj', 'S', ['row · 4096 × 8192', 'row · 4096 × 8192']],
            ['experts roteados (EXL3)', 'S', ['288 × ½ (1024)', '288 × ½ (1024)']],
            ['torre de visão', 'S', ['weights (½)', 'weights (½)']],
            ['lm_head', 'S', ['77.440 linhas', '77.440 linhas']],
            ['DFlash2 (rascunhador)', 'S', ['draft TP 2', 'draft TP 2']],
            ['KV fp8_ds_mla', 'R', ['656 B/tok/cam.', '656 B/tok/cam.']],
          ], flags: ['--tensor-parallel-size 2', 'draft_tensor_parallel_size 2', 'NNODES=2'], mem: 'pesos: 79,65 GiB por rank (82,94 com a cópia KDA BF16) · KV ≈ 7.579 B/token/rank, replicado' },
          alt: { ranks: 3, rows: [
            ['embedding', 'S', ['51.648 ids', '51.648 ids', '51.648 ids']],
            ['MLA: q_b (66 cab.)', 'S', ['22 cab.', '22 cab.', '22 cab.']],
            ['latente c_kv + indexador', 'R', ['inteiro', 'inteiro', 'inteiro']],
            ['KDA: in_proj', 'X', ['8.726 × 4096', '8.726 × 4096', '8.726 × 4096']],
            ['KDA: estado recorrente', 'S', ['22 cab.', '22 cab.', '22 cab.']],
            ['KDA e MLA: o_proj', 'S', ['row · entrada ⅓', 'row · entrada ⅓', 'row · entrada ⅓']],
            ['experts roteados (EXL3)', 'E', ['0–95', '96–191', '192–287']],
            ['torre de visão', 'R', ['data (inteira)', 'data (inteira)', 'data (inteira)']],
            ['lm_head (vocab 154.944)', 'S', ['51.648', '51.648', '51.648']],
            ['DFlash2 (GQA 36/9)', 'X', ['TP 1 (só aqui)', '—', '—']],
            ['KV fp8_ds_mla', 'R', ['igual', 'igual', 'igual']],
          ], flags: ['--tensor-parallel-size 3', '--enable-expert-parallel', '--mm-encoder-tp-mode data', 'heads 64 → 66', 'DFLASH_DRAFT_TP=1'], mem: 'pesos: ~55,75 GiB por nó · o rascunhador fica só no rank 0 (“the divisibility tax”)' },
        },
        qwen: {
          std: { ranks: 2, rows: [
            ['embedding', 'S', ['124.160 ids', '124.160 ids']],
            ['GDN (36): qkvz / out', 'S', ['8192×2560 / 2560×3072', '8192×2560 / 2560×3072']],
            ['QSA (12): qkv+gate / out', 'S', ['6656×2560 / 2560×3072', '6656×2560 / 2560×3072']],
            ['KV das QSA', 'S', ['1 de 2 cab. KV', '1 de 2 cab. KV']],
            ['indexador QSA', 'R', ['640 × 2560', '640 × 2560']],
            ['hyper-connections', 'R', ['1,19 GiB', '1,19 GiB']],
            ['roteador', 'R', ['inteiro', 'inteiro']],
            ['experts roteados (NVFP4)', 'E', ['0–255', '256–511']],
            ['shared expert gate/up', 'S', ['640 × 2560', '640 × 2560']],
            ['PLE (tabela N-gram)', 'S', ['~160 M linhas', '~160 M linhas']],
            ['lm_head', 'S', ['124.160 × 2560', '124.160 × 2560']],
            ['torre de visão', 'R', ['data · 0,84 GiB', 'data · 0,84 GiB']],
            ['MTP: lm_head reduzido', 'S', ['45.734 ids', '1.415 ids']],
          ], flags: ['--tensor-parallel-size 2', '--enable-expert-parallel', '--all2all-backend allgather_reducescatter', '--mm-encoder-tp-mode data'], mem: 'pesos: 64,46 GiB por GPU (medido) · KV 32,02 GiB → 3.652.200 tokens (FP8, GMU 0,835)' },
          alt: { ranks: 1, rows: [
            ['embedding + lm_head', 'W', ['248.320 ids']],
            ['GDN + QSA', 'W', ['todas as cabeças']],
            ['KV das QSA', 'W', ['2 cabeças KV']],
            ['experts roteados', 'W', ['512 experts']],
            ['hyper-connections + roteador', 'W', ['inteiros']],
            ['PLE (tabela N-gram)', 'X', ['offload mmap (26,82 GiB)']],
            ['MTP', 'N', ['MTP_NUM_SPECULATIVE_TOKENS=0']],
          ], flags: ['start-tp1.sh', 'TP1_MODEL_ID local-inference-lab NVFP4 (98,6 GiB)', 'MAX_MODEL_LEN 65536', 'MAX_NUM_SEQS 4'], mem: 'um Spark só: nada é dividido, o PLE sai da GPU por mmap · 685.825 tokens de KV com KV_TARGET_GIB=20 (tp1/.env.sample, GMU 0,813); o default de 8 GiB dá menos' },
        },
        ds: {
          std: { ranks: 2, rows: [
            ['embedding', 'S', ['64.640 ids', '64.640 ids']],
            ['fused_wqa_wkv', 'R', ['inteiro', 'inteiro']],
            ['wq_b + attn_sink', 'S', ['32 cabeças', '32 cabeças']],
            ['wo_a (por grupo)', 'S', ['4 grupos', '4 grupos']],
            ['wo_b', 'S', ['row · 4096 × 4096', 'row · 4096 × 4096']],
            ['KV (SWA + comprimido)', 'R', ['inteiro', 'inteiro']],
            ['indexador', 'R', ['inteiro', 'inteiro']],
            ['experts (B12xExperts)', 'S', ['256 × ½ (1024)', '256 × ½ (1024)']],
            ['lm_head', 'S', ['64.640 linhas', '64.640 linhas']],
            ['cabeça de Markov', 'S', ['vocab ½ · 12 col.', 'vocab ½ · 12 col.']],
            ['DSpark (3 estágios)', 'S', ['TP 2', 'TP 2']],
          ], flags: ['--tensor-parallel-size 2', '--pipeline-parallel-size 1', 'sem EP (MegaMoE: SM100 + EP)', 'NNODES=2'], mem: 'pesos: 79,17 GiB por rank · KV 17,04 GiB → 2.331.430 tokens (2,22× a 1M), replicado' },
          alt: { ranks: 3, rows: [
            ['embedding (vocab 129.408)', 'S', ['43.136 ids', '43.136 ids', '43.136 ids']],
            ['fused_wqa_wkv', 'R', ['inteiro', 'inteiro', 'inteiro']],
            ['wq_b (72 cab.)', 'S', ['24 cab.', '24 cab.', '24 cab.']],
            ['wo_a (9 grupos)', 'S', ['3 grupos', '3 grupos', '3 grupos']],
            ['wo_b', 'S', ['row', 'row', 'row']],
            ['KV + indexador', 'R', ['inteiro', 'inteiro', 'inteiro']],
            ['experts (2112)', 'S', ['704 por rank', '704 por rank', '704 por rank']],
            ['lm_head', 'S', ['43.136', '43.136', '43.136']],
            ['DSpark', 'S', ['sink com pad', 'sink com pad', 'sink com pad']],
          ], flags: ['start-tp3.sh', '--tensor-parallel-size 3', 'apply_tp3_patch.py (grupos 8 → 9)', 'NNODES=3'], mem: 'pesos: 60,69 GiB por rank · KV ≈ 35 GiB/rank → 5,0 M tokens, 16 slots · TTFT +22–23% a 128K–256K' },
        },
      };
      const ST = { r: 'glm', alt: false };
      const reH = adaptH(fig, (w) => { const n = PL[ST.r][ST.alt ? 'alt' : 'std'].rows.length; return w < 600 ? 44 + n * 42 + 150 : 44 + n * 34 + 104; });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DeepSeek-V4' }], value: 'glm', onChange: (v) => { ST.r = v; if (reH) reH(); } });
      UI.seg(ctl, { label: 'Topologia', options: [{ v: 'std', label: 'padrão (TP2)' }, { v: 'alt', label: 'alternativa' }], value: 'std', onChange: (v) => { ST.alt = v === 'alt'; if (reH) reH(); } });
      const C0 = V.C();
      UI.legend(ctl, [[C0.a, 'fatia · rank 0'], [C0.b, 'fatia · rank 1'], [C0.d, 'fatia · rank 2'], [C0.muted, 'réplica (hachura simples)'], [C0.e, 'EP: experts inteiros (xadrez verde)'], [C0.f, 'misto']]);
      V.fig(fig, (g, w, h, t) => {
        if (w < 140 || h < 100) return;
        const C = V.C(), D = PL[ST.r][ST.alt ? 'alt' : 'std'];
        const narrow = w < 600, pad = 12;
        const nr = D.ranks;
        const nameW = narrow ? 118 : Math.min(230, w * 0.28);
        const x0 = pad, x1 = w - pad;
        const cw = (x1 - x0 - nameW) / nr;
        // rodapé cresce se as flags + o texto de memória não couberem na altura base
        const footNeed = (() => {
          let fx = x0 + 10, fyy = 18;
          D.flags.forEach((f) => { const ww = tw(g, f, 11, 'mono', 600) + 16; if (fx + ww > x1 - 10) { fx = x0 + 10; fyy += 24; } fx += Math.min(ww, x1 - x0 - 20) + 6; });
          return fyy + 28 + (wrapLines(g, D.mem, x1 - x0 - 20, 11.5, 'sans', 400).length - 1) * 15 + 14;
        })();
        const footH = Math.max(narrow ? 140 : 92, footNeed);
        const hy = 16;
        const rows = D.rows;
        const rh = Math.min(narrow ? 42 : 34, (h - hy - 22 - footH) / rows.length);
        const rankName = (r) => (narrow && nr > 2 ? 'rank ' + r : r === 0 ? 'rank 0 · head' : 'rank ' + r + ' · worker');
        for (let r = 0; r < nr; r++) V.text(g, nr === 1 ? 'GPU única (TP1)' : rankName(r), x0 + nameW + r * cw + cw / 2, hy + 4, { size: narrow ? 11 : 11.5, font: 'mono', weight: 700, align: 'center', color: nr === 1 ? C.ink : RC(C, r), maxW: cw - 6 });
        const scan = (t * 0.35) % 1;
        rows.forEach(([name, k, cells], q) => {
          const y = hy + 14 + q * rh;
          const onScan = Math.abs(scan * rows.length - q - 0.5) < 0.5;
          if (onScan) V.box(g, x0, y, x1 - x0, rh - 2, { r: 6, fill: V.alpha(C.d, 0.06) });
          const tagCol = k === 'S' || k === 'W' ? C.ink : k === 'R' ? C.muted : k === 'E' ? C.e : k === 'X' ? C.f : C.muted;
          if (narrow) {
            V.text(g, name, x0 + 2, y + rh / 2 - 3, { size: 11, weight: 700, color: C.ink, maxW: nameW - 6 });
            V.text(g, K[k], x0 + 2, y + rh / 2 + 11, { size: 11, font: 'mono', color: tagCol });
          } else {
            const cwid = tw(g, K[k], 11, 'mono', 600) + 14;
            V.text(g, name, x0 + 2, y + rh / 2 + 4, { size: 11.5, weight: 700, color: C.ink, maxW: nameW - cwid - 18 });
            chip(g, K[k], x0 + nameW - 8, y + rh / 2, tagCol, { align: 'right', size: 11, h: 19 });
          }
          for (let r = 0; r < nr; r++) {
            const cx = x0 + nameW + r * cw + 3, cy = y + 2, ww = cw - 6, hh = rh - 6;
            const txt = cells[r] != null ? cells[r] : cells[0];
            const col = nr === 1 ? C.a : RC(C, r);
            if (txt === '—') { V.box(g, cx, cy, ww, hh, { r: 5, stroke: V.alpha(C.muted, 0.35), lw: 1 }); V.text(g, '—', cx + ww / 2, cy + hh / 2 + 0.5, { size: 11, align: 'center', base: 'middle', color: C.muted }); continue; }
            if (k === 'R' || (k === 'N')) { V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.55), lw: 1 }); hatch(g, cx + 1, cy + 1, ww - 2, hh - 2, V.alpha(C.muted, 0.35), 7, t * 6); }
            else if (k === 'E') {
              // EP: cor própria (não é cor de rank) e xadrez, para não confundir com a fatia do rank 2
              V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(C.e, 0.12), stroke: C.e, lw: 1.6 });
              g.save(); g.translate(cx + ww, cy); g.scale(-1, 1); hatch(g, 0, 1, ww - 1, hh - 2, V.alpha(C.e, 0.4), 8, -t * 6, 1); g.restore();
              hatch(g, cx + 1, cy + 1, ww - 2, hh - 2, V.alpha(C.e, 0.4), 8, t * 6, 1);
            }
            else if (k === 'X') { V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(col, 0.18), stroke: C.f, lw: 1.2 }); hatch(g, cx + ww * 0.75, cy + 1, ww * 0.25 - 1, hh - 2, V.alpha(C.muted, 0.45), 5, t * 6); }
            else V.box(g, cx, cy, ww, hh, { r: 5, fill: V.alpha(col, 0.22), stroke: V.alpha(col, 0.75), lw: 1 });
            if (k === 'E') { const tw0 = Math.min(ww - 8, tw(g, txt, 11, 'mono', 600) + 10); V.box(g, cx + ww / 2 - tw0 / 2, cy + hh / 2 - 8, tw0, 16, { r: 4, fill: V.alpha(C.bg || C.panel, 0.85) }); }
            V.text(g, txt, cx + ww / 2, cy + hh / 2 + 0.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: ww - 6 });
          }
        });
        // rodapé
        const fy = h - footH;
        V.box(g, x0, fy, x1 - x0, footH - 6, { r: 10, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
        let fx = x0 + 10, fyy = fy + 18;
        D.flags.forEach((f) => {
          const ww = tw(g, f, 11, 'mono', 600) + 16;
          if (fx + ww > x1 - 10) { fx = x0 + 10; fyy += 24; }
          chip(g, f, fx, fyy, C.c, { size: 11, h: 20, maxW: x1 - x0 - 20 });
          fx += Math.min(ww, x1 - x0 - 20) + 6;
        });
        wrap(g, D.mem, x0 + 10, fyy + 28, x1 - x0 - 20, { size: 11.5, color: C.ink, lh: 15 });
      });
    })();
  },
});

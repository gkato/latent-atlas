LA.chapter({
  id: 'receita-glm-tensorfold',
  part: 9,
  order: 49,
  title: 'Trocando o motor: a receita GLM no TensorFold',
  navTitle: 'Receita GLM · TensorFold',
  subtitle: 'O mesmo GLM-5.3-Flash EXL3, nos mesmos dois DGX Spark, sobre outro motor: TensorFold v0.6.0 e 53 patches. Requisitos, scripts, orçamento de memória, todos os parâmetros, os patches por área, os números com as suas condições e uma comparação honesta com a receita vLLM.',
  minutes: 45,
  recipes: ['tf', 'glm'],
  tags: ['TensorFold', 'exatidão', 'amostragem chaveada', 'RoCE', 'KV FP8', 'q4', 'copy drafts', 'pool compartilhado', 'reuso de prompt', 'config.sh'],
  goals: [
    'Explicar o que muda ao trocar o vLLM pelo <b>TensorFold</b> com os mesmos pesos e o mesmo hardware, e o contrato de exatidão (rascunhado = serial, concorrente = solo, retomado = fresco) com os três defaults que têm perda',
    'Montar os dois Sparks e operar a esteira <code>prepare.sh</code> / <code>start.sh</code> / <code>stop.sh</code>: hash dos patches, digest pinado, revisões, NFS, refit e smoke test',
    'Calcular o orçamento <code>MemAvailable − reserva</code> e o pool de KV, e justificar cada parâmetro do <code>config.sh</code> pela fase em que age',
    'Ler os números da receita com as suas condições, compará-los com a receita vLLM sem tratá-los como A/B, e ajustar a receita à sua carga (licença, idioma, exatidão, memória)',
  ],
  prereqs: ['motor-exato-tensorfold', 'receita-glm'],
  summary: [
    'Mesmo checkpoint EXL3 de 4 bpw, mesmos dois GB10 e mesmo cabo; outro motor: TensorFold v0.6.0, um <b>motor por família</b>, com 53 patches assados na imagem. Sem eles, o upstream servia o GLM uma requisição por vez numa janela de 2.051 tokens; com eles vêm 4 streams, 1M de contexto, KV FP8, RoCE, copy drafts, visão e tool calling.',
    'Contrato: <b>rascunhado = serial, concorrente = solo, retomado = fresco</b>, apoiado em amostragem chaveada, kernels linha-independentes e soma em ordem de rank. Três defaults têm perda contra o checkpoint (<code>DENSE=q4</code>, <code>KV=fp8</code>, <code>KDA_CHUNKED=1</code>); <code>bf16 · bf16 · 0</code> serve o checkpoint como ele é, com janela menor.',
    'Rede: um rank por Spark, TCPStore na porta 29551, rank 0 decide e rank 1 aplica. As parciais fp32 (16 KiB por linha) vão por RDMA one-shot até 512 KiB (11 µs contra 45 µs, decode +6%); chunks de prompt de 32 MiB vão pelo NCCL em dois trilhos (1,9 ms contra 3,5 ms).',
    'Esteira: hash = SHA-256(patches + extras)[0:12]; enquanto for <code>ae8d1c789b47</code>, a imagem vem pelo digest; senão, pela tag do novo hash ou por um build com <code>patch -p0</code> em ordem. Pesos nas revisões pinadas, por rsync -L ou NFS; o start.sh faz setup, checagens, lançamento, carregamento com refit e smoke test.',
    'Memória: orçamento = MemAvailable − reserva (14,5 GiB); estimativa de 88,09 GiB; pool até 12,5 GiB além da janela, ou 2.922.496 tokens; o servidor usa ~8,6 GiB além da estimativa desde a partida (7,0 GiB livres parado) e ~10 sob um prompt de 1M, então a reserva fixa o livre mínimo (5,6 GiB medidos). KV FP8: 528 B por linha, com escala \\(2^{\\lceil\\log_2 a_{\\max}\\rceil-8}\\).',
    'Números (sparkDash, clocks com teto de 2.200 MHz, greedy): prosa de 60,4 a 108,8 tok/s de 1 para 4 requisições, estruturado de 114,7 a 227,9, prefill de ~1.950 tok/s até 64k e 1.015 a 981.841 tokens, replay de 64k em menos de 0,07 s, GSM8K 98,0% e HumanEval 97,6% (checkpoint TR3; o pack default desde a v1.3.3 reporta 98,8% e 95,7%). Contra o vLLM (36,1 → 60,4 em prosa), não é um A/B controlado.',
    'Custos na v1.3.2: só 2 Sparks, no máximo 4 requisições, API menor e 53 patches para rebasear (a imagem de 123 patches traz 3 Sparks, 8 requisições e <code>/v1/messages</code>: veja a última seção). O DFlash2 é CC BY-NC-ND (use <code>DRAFTER=mtp</code> para uso comercial), e <code>q4</code> perde o fim de turno em francês (12/48 contra 1/48): use <code>DENSE=fp8</code>.',
  ],
  quiz: [
    { q: 'Com os defaults da receita (<code>DENSE=q4</code>, <code>KV=fp8</code>, <code>KDA_CHUNKED=1</code>), qual afirmação é verdadeira?', o: ['A resposta com rascunhos é idêntica à serial, mas pode diferir da do checkpoint servido em BF16 sem perdas', 'A resposta é idêntica à do checkpoint em BF16, porque a amostragem é chaveada', 'Os rascunhos podem mudar a resposta quando a temperatura é maior que 0', 'Quatro requisições juntas dão respostas diferentes das servidas sozinhas, por causa do lote'], a: 0, e: 'As três igualdades comparam o motor com ele mesmo, nas mesmas configurações. q4 e KV fp8 perdem informação e o KDA chunked não reproduz os bits do kernel serial, então a referência sem perdas (<code>DENSE=bf16 KV=bf16 KDA_CHUNKED=0</code>) pode divergir num quase-empate. Rascunhos e lote não mudam nada: 6/6 e 22/22 nos testes publicados.' },
    { q: 'Por que o TensorFold soma as parciais de tensor parallel com all-gather + soma em ordem de rank, e não com all-reduce?', o: ['Porque o all-gather move menos bytes', 'Porque o NCCL não implementa all-reduce no GB10', 'Porque a soma em ponto flutuante não é associativa e o all-reduce não garante a ordem: os ranks poderiam ficar com bits diferentes e sortear tokens diferentes', 'Porque o RoCE só transporta all-gathers'], a: 2, e: 'Com a mesma soma, na mesma ordem (“rank 0 first”), os dois ranks têm os mesmos bits e amostram o mesmo token sem trocá-lo. O preço: as parciais vão em fp32 (16 KiB por linha, ~2× o all-reduce em bf16 do vLLM) e cada rank faz a soma inteira; por isso as mensagens pequenas vão pelo RDMA one-shot.' },
    { q: 'Você editou um patch. O que o <code>prepare.sh</code> faz com a imagem?', o: ['Puxa a mesma imagem pelo digest pinado', 'Calcula o novo hash, procura a tag <code>v0.6.0-&lt;hash&gt;</code> no GHCR e, se ela não existir (ou com <code>PULL=0</code>), constrói aplicando os patches com <code>patch -p0</code> em ordem de nome', 'Reaplica o patch como overlay no próximo <code>docker run</code>', 'Recusa iniciar até você publicar a imagem'], a: 1, e: 'O hash cobre <code>patches/*.patch</code> e os pacotes extras. O digest só é usado enquanto o hash for o da release (<code>ae8d1c789b47</code>). Com patches novos, vale a tag do novo hash ou um build local, que termina com um import de verificação e o label <code>tf.patches</code>. O <code>publish-image.sh</code> publica o resultado e imprime os valores a fixar.' },
    { q: 'MemAvailable de 116 GiB na partida, reserva de 14,5, estimativa de 88,09 e <code>KV_POOL_GIB</code> de 12,5. O que limita o pool, e qual é o livre mínimo aproximado sob um prompt de 1M?', o: ['A memória; ~0 GiB livres', 'O teto de 12,5 GiB; 14,5 GiB livres, porque a reserva nunca é tocada', 'A janela; o servidor recusa e faz refit', 'O teto de 12,5 GiB; ~5 GiB livres (reserva + sobra − ~10 GiB de pico)'], a: 3, e: 'Sobram 116 − 14,5 − 88,09 = 13,41 GiB, mais que o teto, então o pool fica em 12,5 GiB e ~0,9 GiB continuam livres. O servidor passa ~10 GiB da própria estimativa no pico: ~8,6 deles desde a partida (contexto CUDA, NCCL, graphs; parado sobram ~14,5 + 0,9 − 8,6 ≈ 6,8 GiB, medido 7,0) e ~1,4 do prompt de 1M. Sob o prompt ficam ~14,5 + 0,9 − 10 ≈ 5,4 GiB; o medido foi 5,6 GiB no head.' },
    { q: 'O que a política <code>fnc7:0.3</code> faz de diferente da <code>fc5:0.3</code>?', o: ['Escolhe outros tokens para rascunhar', 'Estima a chance de cada rascunho com o ruído Gumbel chaveado do alvo, que já é conhecido, e para quando o produto das chances cai abaixo de 0,3, com até 7 rascunhos; a resposta não muda', 'Troca a verificação por igualdade pela rejeição \\(\\min(1, p/q)\\)', 'Aumenta a temperatura do alvo para aceitar mais rascunhos'], a: 1, e: '“Only where the chain stops changes: the picks are the same.” A confiança é \\(\\mathrm{softmax}((\\text{score}+g)/\\beta)\\) com \\(\\beta = 2\\). Medido sobre <code>fc5:0.3</code>: prosa 47,2 → 50,6 tok/s, código 51,7 → 55,5. Como a verificação compara com a amostra chaveada, nenhuma política altera os tokens.' },
    { q: 'Um uso comercial com prompts curtos em francês. O que ajustar no <code>.env</code>?', o: ['Nada: os defaults servem', '<code>KV=bf16</code> e <code>CONTEXT=0</code>', '<code>DRAFTER=mtp</code> (o DFlash2 é não comercial; <code>PARALLEL</code> vira 1) e <code>DENSE=fp8</code> (q4 perde o fim de turno em 12 de 48 respostas em francês)', '<code>COMM=nccl</code> e <code>SPLIT=0</code>'], a: 2, e: 'A licença CC BY-NC-ND do DFlash2 proíbe uso comercial; a cabeça MTP do checkpoint evita isso, ao custo de uma requisição por vez e dos +5–10% do DFlash2. O <code>DENSE=fp8</code> custa ~10% de decode, mas mantém o fim de turno (1/48, como o bf16). <code>COMM</code> e <code>SPLIT</code> não mexem nos bits.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x, d = 0) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    let tfCache = null;
    if (LA.theme && LA.onCleanup) LA.onCleanup(LA.theme.onChange(() => { tfCache = null; }));
    const TF = () => tfCache || (tfCache = (getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || V.C().d));
    const GLM = () => V.C().glm || V.C().a;
    const RED = () => V.C().red || '#FB2C36';
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** maior fonte (≤ size, ≥ min) em que o texto cabe em maxW */
    function fitSize(g, s, maxW, size, font = 'sans', weight = 400, min = 8) { let z = size; while (z > min && tw(g, s, z, font, weight) > maxW) z -= 0.5; return Math.max(min, z); }
    /** número com 1 casa para células estreitas: '−0,8' ou, compacto, '−,8'; |v| < 0,05 vira '0,0' */
    const cellNum = (v, compact) => { if (Math.abs(v) < 0.05) return '0,0'; const s = nf(Math.abs(v), 1); return (v < 0 ? '−' : '') + (compact ? s.replace(/^0(?=,)/, '') : s); };
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null && label !== '') {
        const font = o.font || 'mono', weight = o.weight || 500;
        const size = o.fit === false ? (o.size || 11.5) : fitSize(g, label, w - 6, o.size || 11.5, font, weight, o.min || 7.5);
        V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size, font, weight, color: o.color || C.ink, align: 'center', base: 'middle' });
      }
    }
    function mark(g, x, y, ok, s = 6) {
      const C = V.C();
      g.lineWidth = 2.2; g.strokeStyle = ok ? C.a : RED(); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    const hsl = (hue, a = 1) => { const C = V.C(); return `hsla(${hue}, ${C.dark ? 72 : 62}%, ${C.dark ? 62 : 40}%, ${a})`; };
    const onTile = () => (V.C().dark ? '#06140d' : '#ffffff');
    const fnv = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); };
    const softmax = (xs, T = 1) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp((x - m) / T)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
    /** grade de células com valores; fill(v,i,j) → [cor, escuro?] controla o contraste do texto */
    function cells(g, x, y, data, cw, ch, gap, fill, text, label) {
      const C = V.C();
      data.forEach((row, i) => row.forEach((v, j) => {
        const xx = x + j * (cw + gap), yy = y + i * (ch + gap);
        const [col, darkBg] = fill(v, i, j);
        g.fillStyle = col; V.rr(g, xx, yy, cw, ch, Math.min(4, cw / 5)); g.fill();
        const s = text(v, i, j);
        if (s != null && cw >= 12) {
          const z = fitSize(g, s, cw - 3, Math.min(11.5, ch * 0.55), 'mono', 500, 6);
          V.text(g, s, xx + cw / 2, yy + ch / 2 + 0.5, { size: z, font: 'mono', align: 'center', base: 'middle', color: darkBg ? (C.dark ? C.ink : '#ffffff') : (C.dark ? '#0b1a13' : C.ink) });
        }
      }));
      if (label) V.text(g, label, x, y - 7, { size: 11, color: C.muted, maxW: data[0].length * (cw + gap) + 60 });
      return { w: data[0].length * (cw + gap) - gap, h: data.length * (ch + gap) - gap };
    }
    /** cor de célula: mapa divergente (valor) ou heat (erro), com flag de fundo escuro para o texto */
    const divFill = (v, m) => { const C = V.C(); const tt = Math.max(-1, Math.min(1, v / m)); const a = Math.abs(tt); return [V.cmap(tt, 'div'), C.dark ? a < 0.55 : a > 0.55]; };
    const heatFill = (e) => { const C = V.C(); const tt = Math.max(0, Math.min(1, e)); return [V.cmap(tt, 'heat'), C.dark ? tt < 0.6 : tt > 0.35]; };

    /* ================================================ FIG 1 — o contrato: quatro caminhos, um hash */
    (function () {
      const fig = root.querySelector('#fig-contrato');
      adaptH(fig, 400, 420, 600);
      const SENT = [
        ['Um', 'chute', 'só', 'propõe', ';', 'quem', 'decide', 'é', 'a', 'chave', 'do', 'prompt'],
        ['Quatro', 'pedidos', 'juntos', 'recebem', 'os', 'mesmos', 'tokens', 'que', 'teriam', 'sozinhos', 'no', 'motor'],
        ['Retomar', 'do', 'estado', 'guardado', 'dá', 'o', 'mesmo', 'que', 'refazer', 'tudo', 'do', 'zero'],
      ];
      const ALT = ['porém', 'aqui', 'um', 'empate', 'virou', 'outro', 'caminho', 'e', 'segue'];
      const S = { mode: 'receita', seed: 0, t0: -6 };
      let base, ref, div;
      function build() { const r = V.rng(17 + S.seed * 31); base = SENT[S.seed % SENT.length].slice(); div = 5 + r.int(4); ref = base.map((w, i) => (i < div ? w : ALT[(i - div + S.seed) % ALT.length])); }
      build();
      const LANES = [
        { k: 'serial', t: 'um token por vez · "draft": false', c: 'a' },
        { k: 'draft', t: 'com rascunhos', c: 'b' },
        { k: 'batch', t: 'num lote com mais 3', c: 'c' },
        { k: 'resume', t: 'retomado do estado guardado', c: 'e' },
        { k: 'ref', t: 'checkpoint sem perdas (bf16 · bf16 · KDA serial)', c: 'f' },
      ];
      const RD = [3, 1, 4, 2, 2], RB = [2, 3, 2, 3, 2];
      const N = 12, P = 9.5;
      const cl = (x) => Math.max(0, Math.min(N, Math.floor(x)));
      function shown(k, tau) {
        if (k === 'serial' || k === 'ref') return cl((tau - 0.4) / 0.45 + 1);
        if (k === 'resume') return tau < 0.6 ? 0 : cl((tau - 0.6) / 0.4 + 1);
        const R = k === 'draft' ? RD : RB, dur = k === 'draft' ? 0.95 : 1.0;
        let n = 0; for (let j = 0; j < R.length; j++) if (tau >= 0.4 + (j + 1) * dur) n += R[j];
        return Math.min(N, n);
      }
      function ghost(k, tau) {
        if (k !== 'draft' && k !== 'batch') return null;
        const R = k === 'draft' ? RD : RB, dur = k === 'draft' ? 0.95 : 1.0;
        const j = Math.floor((tau - 0.4) / dur); if (j < 0 || j >= R.length) return null;
        let s = 0; for (let q = 0; q < j; q++) s += R[q];
        return [s, s + R[j], ((tau - 0.4) / dur) % 1];
      }
      const doneAt = { serial: 0.4 + 11 * 0.45, ref: 0.4 + 11 * 0.45, resume: 0.6 + 11 * 0.4, draft: 0.4 + 5 * 0.95, batch: 0.4 + 5 * 1.0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Configuração', options: [{ v: 'receita', label: 'defaults da receita (q4 · fp8 · chunked)' }, { v: 'exato', label: 'sem perdas (bf16 · bf16 · KDA serial, KDA_CHUNKED=0)' }], value: 'receita', onChange: (v) => { S.mode = v; S.t0 = H.t; upd(); } });
      UI.button(ctl, '↻ outra frase', () => { S.seed++; build(); S.t0 = H.t; });
      const r1 = nocase(UI.readout(ctl, 'rascunhado = serial', '✔')), r2 = nocase(UI.readout(ctl, 'concorrente = solo', '✔')), r3 = nocase(UI.readout(ctl, 'retomado = fresco', '✔')), r4 = nocase(UI.readout(ctl, '= checkpoint BF16?', '—'));
      function upd() { r1.set('✔ mesmos bits', 'ok'); r2.set('✔ mesmos bits', 'ok'); r3.set('✔ mesmos bits', 'ok'); if (S.mode === 'receita') r4.set('pode divergir', 'warn'); else r4.set('✔ idêntico', 'ok'); }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const tau = (t - S.t0) % P;
        const wide = w >= 600;
        const top = 8, laneH = (h - top - (wide ? 30 : 38)) / 5;
        const labelW = wide ? Math.min(210, w * 0.27) : 0;
        const hashW = wide ? 86 : 64;
        const x0 = wide ? labelW + 6 : 8;
        const tokW = w - x0 - hashW - 28;
        const gap = 3, cw = (tokW - gap * (N - 1)) / N;
        const hBase = fnv(base.join(' ')), hRef = fnv((S.mode === 'receita' ? ref : base).join(' '));
        const hashX = w - hashW - 4;
        const hashPos = [];
        LANES.forEach((L, i) => {
          const col = C[L.c];
          const y0 = top + i * laneH;
          const chH = wide ? Math.min(30, laneH - 16) : Math.min(26, laneH - 26);
          const cy = wide ? y0 + laneH / 2 - chH / 2 : y0 + 18;
          if (wide) V.wrap(g, L.t, 8, y0 + laneH / 2 - (L.t.length > 26 ? 4 : -4), labelW - 10, { size: 12, color: col, weight: 600, lineH: 15, maxLines: 2 });
          else V.text(g, L.t, 8, y0 + 12, { size: 11, color: col, weight: 600, maxW: w - 16 });
          const seq = L.k === 'ref' && S.mode === 'receita' ? ref : base;
          const n = shown(L.k, tau), gh = ghost(L.k, tau);
          for (let j = 0; j < N; j++) {
            const x = x0 + j * (cw + gap);
            const diverged = L.k === 'ref' && S.mode === 'receita' && j >= div;
            if (j < n) {
              const cc = diverged ? TF() : col;
              const s = fitSize(g, seq[j], cw - 3, wide ? 11 : 10, 'mono', 500, 6.5);
              const big = s >= 8;
              chip(g, x, cy, cw, chH, big ? seq[j] : '', { fill: V.alpha(cc, diverged ? 0.28 : 0.18), stroke: cc, r: 5, size: s, fit: false });
              if (!big) { g.fillStyle = hsl(parseInt(fnv(seq[j]).slice(0, 3), 16) % 360, 0.85); V.rr(g, x + 3, cy + chH * 0.28, cw - 6, chH * 0.44, 3); g.fill(); }
              if (diverged && j === div) V.text(g, '≠', x + cw / 2, wide ? cy - 3 : cy + chH + 13, { size: 13, weight: 700, color: TF(), align: 'center' });
            } else if (gh && j >= gh[0] && j < gh[1]) {
              g.setLineDash([3, 3]); V.box(g, x, cy, cw, chH, { r: 5, stroke: V.alpha(col, 0.5 + 0.4 * Math.sin(gh[2] * Math.PI)), lw: 1.4 }); g.setLineDash([]);
              if (cw >= 18) V.text(g, '?', x + cw / 2, cy + chH / 2 + 4, { size: 11, color: V.alpha(col, 0.7), align: 'center' });
            } else {
              V.box(g, x, cy, cw, chH, { r: 5, stroke: V.alpha(C.muted, 0.18), lw: 1 });
            }
          }
          if (gh) { const xa = x0 + gh[0] * (cw + gap) - 2, xb = x0 + gh[1] * (cw + gap) - gap + 2; V.line(g, xa, cy + chH + 4, xb, cy + chH + 4, { color: col, lw: 2 }); V.text(g, L.k === 'draft' ? 'janela' : 'lote', (xa + xb) / 2, cy + chH + 15, { size: 10, color: col, align: 'center' }); }
          if (L.k === 'batch') V.text(g, '+3 streams no mesmo lote', x0 + tokW, cy - 3, { size: 9.5, color: C.muted, align: 'right' });
          if (L.k === 'resume' && tau < 0.9) { const a = Math.max(0, 1 - Math.max(0, tau - 0.3) / 0.6); V.glow(g, x0 + cw / 2, cy + chH / 2, 26, C.e, a); chip(g, x0 + cw + 4, cy + 2, Math.min(110, tokW * 0.4), chH - 4, '↺ estado guardado', { fill: V.alpha(C.e, 0.2 * a + 0.02), stroke: V.alpha(C.e, a), color: C.ink, size: 10 }); }
          // hash
          const ready = n >= N && tau > doneAt[L.k] + 0.25;
          const hv = L.k === 'ref' ? hRef : hBase;
          const same = hv === hBase;
          if (ready) chip(g, hashX, cy, hashW - 4, chH, hv, { fill: V.alpha(same ? C.a : TF(), 0.16), stroke: same ? C.a : TF(), size: wide ? 11 : 9.5 });
          else chip(g, hashX, cy, hashW - 4, chH, '…', { fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25), color: C.muted });
          hashPos.push([hashX, cy, chH, ready, same]);
        });
        V.text(g, 'token_sha', hashX + (hashW - 4) / 2, top + 2 + (wide ? 2 : 0), { size: 9.5, font: 'mono', color: C.muted, align: 'center', base: 'top' });
        // conclusão
        const allReady = hashPos.slice(0, 4).every((p) => p[3]);
        if (allReady) {
          const ya = hashPos[0][1] + hashPos[0][2] / 2, yb = hashPos[3][1] + hashPos[3][2] / 2, xr = hashX - 4;
          g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.moveTo(xr, ya); g.lineTo(xr - 6, ya); g.lineTo(xr - 6, yb); g.lineTo(xr, yb); g.stroke();
          V.text(g, '=', xr - 12, (ya + yb) / 2 + 5, { size: 16, weight: 700, color: C.a, align: 'center' });
        }
        const msg = S.mode === 'receita'
          ? 'Os quatro caminhos dão os mesmos bits entre si; o checkpoint sem perdas pode divergir (aqui no token ' + (div + 1) + ').'
          : 'Com bf16 · bf16 · KDA serial, os quatro caminhos e o checkpoint dão a mesma sequência.';
        if (wide) V.text(g, msg, w / 2, h - 10, { size: 12, color: C.ink, align: 'center', maxW: w - 12 });
        else V.wrap(g, msg, 8, h - 24, w - 16, { size: 10.5, color: C.ink, lineH: 13, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 2 — rede: dois Sparks, um cabo, dois trilhos */
    (function () {
      const fig = root.querySelector('#fig-rede');
      adaptH(fig, 390, 470);
      const S = { msg: 'dec', comm: 'roce', rails: 2 };
      const MSG = { dec: { rows: 1, lab: '16 KiB · 1 linha', n: 3 }, lote: { rows: 32, lab: '512 KiB · 32 linhas', n: 8 }, chunk: { rows: 2048, lab: '32 MiB · 2.048 linhas', n: 26 } };
      const tr = () => (S.comm === 'roce' && MSG[S.msg].rows * 16 <= 512 ? 'roce' : 'nccl');
      const STG = {
        roce: [['GPU copia o shard para a memória pinned', 'copia p/ pinned'], ['o último bloco toca a campainha', 'campainha'], ['proxy: RDMA WRITE + flag de 4 B', 'WRITE + flag'], ['a GPU do par espera a flag', 'espera a flag'], ['soma em ordem de rank', 'soma 0 + 1']],
        nccl: [['NCCL copia para o staging do host', 'staging no host'], ['o proxy do NCCL agenda o envio', 'proxy NCCL'], ['envio pelos trilhos', 'envio'], ['recebe e copia para a GPU', 'recebe'], ['soma em ordem de rank', 'soma 0 + 1']],
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mensagem', options: [{ v: 'dec', label: 'decode · 16 KiB' }, { v: 'lote', label: 'lote 4 streams · 512 KiB' }, { v: 'chunk', label: 'chunk de prompt · 32 MiB' }], value: 'dec', onChange: (v) => { S.msg = v; upd(); } });
      UI.seg(ctl, { label: 'COMM', options: [{ v: 'roce', label: 'roce' }, { v: 'nccl', label: 'nccl' }], value: 'roce', onChange: (v) => { S.comm = v; upd(); } });
      UI.seg(ctl, { label: 'Trilhos', options: [{ v: 2, label: '2' }, { v: 1, label: '1' }], value: 2, onChange: (v) => { S.rails = v; upd(); } });
      const rT = UI.readout(ctl, 'transporte', '—'), rM = UI.readout(ctl, 'tempo publicado', '—');
      function upd() {
        const t = tr();
        rT.set(t === 'roce' ? 'RDMA one-shot' : 'NCCL');
        let v = '—', st;
        if (S.msg === 'dec') { v = t === 'roce' ? '11 µs' : '45 µs (32 afinado)'; st = t === 'roce' ? 'ok' : 'warn'; }
        else if (S.msg === 'lote') { v = t === 'roce' ? 'sem nº · código ×4 +1,4%' : 'sem nº publicado'; }
        else { v = S.rails === 2 ? '1,9 ms' : '3,5 ms'; st = S.rails === 2 ? 'ok' : 'warn'; }
        rM.set(v, st);
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const T = tr();
        const per = T === 'roce' ? 2.6 : 3.6;
        const ph = (t % per) / per, stage = Math.min(4, Math.floor(ph * 5)), sf = (ph * 5) % 1;
        const bw = wide ? w * 0.31 : w * 0.41, bh = wide ? h * 0.66 : h * 0.6;
        const bx0 = 8, bx1 = w - 8 - bw, by = 26;
        const boxes = [{ x: bx0, side: 1, name: 'Spark head · rank 0', ip: '192.0.2.1' }, { x: bx1, side: -1, name: 'Spark worker · rank 1', ip: '192.0.2.2' }];
        V.text(g, MSG[S.msg].lab + ' por rank · ' + (T === 'roce' ? 'RDMA one-shot (b12x)' : 'NCCL (host-staged)'), w / 2, 14, { size: wide ? 12 : 10.5, font: 'mono', color: C.ink, align: 'center', maxW: w - 16 });
        const nRows = Math.min(MSG[S.msg].rows, 6), nCols = wide ? 12 : 8;
        const geo = boxes.map((B, r) => {
          const bc = r ? C.b : TF();
          V.box(g, B.x, by, bw, bh, { r: 12, fill: V.alpha(bc, 0.07), stroke: V.alpha(bc, 0.55), lw: 1.4 });
          V.text(g, B.name, B.x + 10, by + 17, { size: wide ? 12 : 10.5, weight: 600, color: bc, maxW: bw - 16 });
          const nw = wide ? 24 : 18;
          const ix = B.side > 0 ? B.x + 10 : B.x + 14 + nw, iw = bw - 24 - nw;
          const gpu = { x: ix, y: by + 26, w: iw, h: bh * 0.44 };
          const pin = { x: ix, y: gpu.y + gpu.h + 10, w: iw, h: bh * 0.2 };
          const nic = { x: B.side > 0 ? B.x + bw - nw - 6 : B.x + 6, y: by + bh * 0.3, w: nw, h: bh * 0.44 };
          V.box(g, gpu.x, gpu.y, gpu.w, gpu.h, { r: 8, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.55) });
          V.text(g, 'GPU', gpu.x + 8, gpu.y + 14, { size: 11, weight: 600, color: C.a });
          V.text(g, wide ? 'P(' + r + ') [' + nf(MSG[S.msg].rows) + ' × 4096] fp32' : 'P(' + r + ') ' + nf(MSG[S.msg].rows) + ' lin.', gpu.x + 40, gpu.y + 14, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.ink, maxW: gpu.w - 46 });
          // mini-matriz da parcial: R linhas × 4096 colunas (amostradas)
          const gx = gpu.x + 8, gy = gpu.y + 22, gw = gpu.w - 16, gh = gpu.h - 30;
          const cs = Math.min((gw - (nCols - 1) * 2) / nCols, (gh - (nRows - 1) * 2) / Math.max(nRows, 2));
          for (let i = 0; i < nRows; i++) for (let j = 0; j < nCols; j++) {
            const v = Math.sin(i * 1.7 + j * 0.9 + r * 2.3) * 0.5 + 0.5;
            const col = stage === 4 ? V.alpha(C.a, 0.35 + 0.5 * (Math.sin(i * 1.7 + j * 0.9) * 0.5 + 0.5)) : V.alpha(bc, 0.25 + 0.6 * v);
            g.fillStyle = col; V.rr(g, gx + j * (cs + 2), gy + i * (cs + 2), cs, cs, 2); g.fill();
          }
          if (stage !== 4) V.text(g, '= ' + MSG[S.msg].lab.split(' · ')[0] + (wide ? ' por rank' : ''), gpu.x + 8, gpu.y + gpu.h - 7, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted, maxW: gpu.w - 12 });
          V.box(g, pin.x, pin.y, pin.w, pin.h, { r: 8, fill: V.alpha(C.d, 0.1), stroke: V.alpha(C.d, 0.55) });
          V.text(g, T === 'roce' ? (wide ? 'memória pinned · send/recv' : 'pinned') : (wide ? 'staging do NCCL (host)' : 'staging'), pin.x + 8, pin.y + pin.h / 2 + 4, { size: wide ? 10.5 : 9.5, color: C.ink, maxW: pin.w - 30 });
          V.box(g, nic.x, nic.y, nic.w, nic.h, { r: 6, fill: V.alpha(C.muted, 0.15), stroke: V.alpha(C.muted, 0.6) });
          V.text(g, 'CX7', nic.x + nic.w / 2, nic.y - 4, { size: 9.5, font: 'mono', color: C.muted, align: 'center' });
          const ports = [nic.y + nic.h * 0.3, nic.y + nic.h * 0.7];
          ports.forEach((py, k) => { V.dot(g, nic.x + nic.w / 2, py, 3.5, k === 1 && S.rails === 1 ? V.alpha(C.muted, 0.35) : C.e); });
          const ey = by + bh - 10;
          if (r === 0) V.text(g, wide ? 'API :8888 · TCPStore :29551' : 'API · TCPStore', B.x + 10, ey, { size: 10, font: 'mono', color: C.muted, maxW: bw - 16 });
          else V.text(g, wide ? 'GID idx 3 · v2 · ::ffff:' + B.ip : 'GID v2 ::ffff:' + B.ip, B.x + 10, ey, { size: wide ? 10 : 9, font: 'mono', color: C.muted, maxW: bw - 16 });
          return { gpu, pin, nic, ports, gy, gh };
        });
        const pA = geo[0].ports, pB = geo[1].ports, xa = geo[0].nic.x + geo[0].nic.w / 2, xb = geo[1].nic.x + geo[1].nic.w / 2;
        [0, 1].forEach((k) => {
          const off = k === 1 && S.rails === 1;
          V.line(g, xa, pA[k], xb, pB[k], { color: off ? V.alpha(C.muted, 0.3) : V.alpha(C.e, 0.7), lw: off ? 1 : 2.4, dash: off ? [4, 4] : null });
          V.text(g, off ? (wide ? 'trilho 2 (desligado)' : 'desligado') : 'trilho ' + (k + 1), (xa + xb) / 2, pA[k] - 5, { size: 9.5, color: off ? C.muted : C.e, align: 'center' });
        });
        V.text(g, wide ? 'QSFP · 2× PCIe Gen5 x4' : 'QSFP', (xa + xb) / 2, pA[1] + 16, { size: wide ? 10 : 9, font: 'mono', color: C.muted, align: 'center', maxW: xb - xa - 8 });
        const nP = MSG[S.msg].n;
        geo.forEach((G, r) => {
          const gx = G.gpu.x + G.gpu.w / 2, gy = G.gpu.y + G.gpu.h, py = G.pin.y + G.pin.h / 2;
          if (stage === 0) { for (let k = 0; k < Math.min(6, nP); k++) { const u = (sf + k / 6) % 1; V.dot(g, gx - 20 + k * 8, V.lerp(gy - 4, py, u), 2.8, V.alpha(r ? C.b : TF(), 0.95)); } }
          if (stage === 1) { const pulse = 0.5 + 0.5 * Math.sin(sf * Math.PI * 4); V.glow(g, G.pin.x + G.pin.w - 14, py, 16, C.d, pulse); V.text(g, T === 'roce' ? '🔔' : '⚙', G.pin.x + G.pin.w - 14, py + 5, { size: 13, align: 'center' }); }
          if (stage === 3) { V.text(g, T === 'roce' ? '⚑ flag seq' : '⇣ recebido', G.pin.x + G.pin.w - 6, G.pin.y - 4, { size: 10, color: C.d, weight: 600, align: 'right' }); const ang = sf * Math.PI * 2; g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath(); g.arc(G.gpu.x + G.gpu.w - 14, G.gpu.y + 12, 6, ang, ang + 4.2); g.stroke(); }
          if (stage === 4) { const bxw = Math.min(G.gpu.w - 30, wide ? 150 : 96); chip(g, G.gpu.x + G.gpu.w - bxw - 8, G.gpu.y + G.gpu.h - 22, bxw, 17, 'P(0) + P(1) → y', { fill: V.alpha(C.a, 0.25), stroke: C.a, size: 10.5 }); }
        });
        if (stage === 4) V.text(g, '= mesmos bits', w / 2, by + bh * 0.2, { size: wide ? 12 : 10.5, weight: 700, color: C.a, align: 'center' });
        if (stage === 2) {
          for (let k = 0; k < nP; k++) {
            const rail = S.rails === 2 ? k % 2 : 0;
            const u = (sf * 1.1 + (k / nP) * 0.6) % 1;
            const dir = k % 4 < 2 ? 1 : -1;
            const x = dir > 0 ? V.lerp(xa, xb, u) : V.lerp(xb, xa, u);
            const y = V.lerp(pA[rail], pB[rail], dir > 0 ? u : 1 - u);
            V.dot(g, x, y, S.msg === 'chunk' ? 2.4 : 3.4, dir > 0 ? TF() : C.b);
          }
        }
        const sy = by + bh + 12;
        const st = STG[T];
        const cols = wide ? 5 : 3, rows = wide ? 1 : 2;
        const noteH = wide ? 20 : 30;
        const cwid = (w - 16 - (cols - 1) * 6) / cols, chh = Math.min(wide ? 46 : 40, (h - sy - noteH - 8 - (rows - 1) * 6) / rows);
        st.forEach((s2, i) => {
          const c = i % cols, rr = Math.floor(i / cols);
          const x = 8 + c * (cwid + 6), y = sy + rr * (chh + 6);
          const on = i === stage;
          V.box(g, x, y, cwid, chh, { r: 8, fill: on ? V.alpha(C.d, 0.22) : V.alpha(C.muted, 0.08), stroke: on ? C.d : V.alpha(C.muted, 0.35), lw: on ? 2 : 1 });
          const lab = (i + 1) + ' · ' + (wide ? s2[0] : s2[1]);
          V.wrap(g, lab, x + 7, y + (wide ? 17 : chh / 2 + 4), cwid - 12, { size: wide ? 11.5 : 10, lineH: 14, maxLines: 2, color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
        });
        const note = T === 'roce' ? 'uma chamada de kernel por coletiva; época no device para o CUDA graph reproduzir a troca' : S.comm === 'roce' ? 'acima de 512 KiB o NCCL assume, com o mesmo layout (ordem de rank) e os mesmos bits' : 'mensagens pequenas pagam a latência fixa do NCCL (~30–50 µs)';
        if (wide) V.text(g, note, w / 2, h - 8, { size: 11.5, color: C.muted, align: 'center', maxW: w - 12 });
        else V.wrap(g, note, 8, h - 22, w - 16, { size: 10, color: C.muted, lineH: 13, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 3 — esteira (stepper) */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      adaptH(fig, 380, 440);
      const S = { edit: false, pull0: false, nfs: false, low: false };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'editei um patch', value: false, onChange: (v) => { S.edit = v; if (st) st.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'PULL=0', value: false, onChange: (v) => { S.pull0 = v; if (st) st.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'WORKER_WEIGHTS=nfs', value: false, onChange: (v) => { S.nfs = v; if (st) st.t0 = performance.now(); } });
      UI.toggle(ctl, { label: 'pouca memória livre (refit)', value: false, onChange: (v) => { S.low = v; if (st) st.t0 = performance.now(); } });
      const NAMES = ['preflight', 'imagem', 'worker', 'pesos', 'checagens', 'lançar', 'carregar', 'smoke'];
      const steps = [
        { t: 'Preflight nos dois Sparks', d: 'Docker, GPU, runtime NVIDIA, <code>rsync</code>, ssh por chave, link (rota → interface → HCA → GID RoCE v2) e disco: 180 GB para os pesos e 35 GB para a imagem.' },
        { t: 'A imagem e o hash dos patches', d: 'SHA-256 de <code>patches/*.patch</code> + extras, 12 hex. Com o hash da release, a imagem vem pelo digest; com um hash novo, pela tag publicada ou por um build com <code>patch -p0</code> em ordem.' },
        { t: 'A mesma imagem no worker', d: 'Pull, ou <code>docker save | ssh docker load</code> só com as camadas que faltam; a identidade é comparada por conteúdo (issue #8).' },
        { t: 'Pesos nas revisões pinadas', d: 'Download no head, <code>tensorfold info</code> e <code>refs/main</code>; depois <code>rsync -a -L --partial</code> conferido arquivo a arquivo, ou o volume NFS somente leitura.' },
        { t: 'start.sh 2/5 · checagens', d: 'Argumentos pelo parser do TensorFold num container sem GPU, link, servidor anterior, porta e memória (≥ ~110 GiB por Spark).' },
        { t: 'start.sh 3/5 · lançamento', d: 'Rank 1 no worker por ssh, depois rank 0 + API. Rendezvous pela TCPStore na porta 29551 e configurações comparadas.' },
        { t: 'start.sh 4/5 · carregamento', d: '~80 GiB de pesos por Spark em 2–6 min, com uma batida a cada 15 s. Se a janela não cabe, refit: o script reinicia uma vez com <code>--context N</code>.' },
        { t: 'start.sh 5/5 · smoke test', d: '“Reply with OK.” a temperatura 0, 32 tokens, sem thinking, pelos dois ranks. Depois: <code>is now LIVE! on port 8888</code>.' },
      ];
      let H3 = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => H3 && H3.redraw() });
      let prevI = 0, moveFrom = 0, moveT0 = 0;
      const wordHash = () => (S.edit ? 'c41f9e07b2a8' : 'ae8d1c789b47');
      function scene0(g, A, e, C, wide) {
        const items = ['Docker', 'GPU (nvidia-smi -L)', 'runtime NVIDIA', 'rsync', 'ssh BatchMode', 'link · GID RoCE v2', 'disco 180 + 35 GB'];
        const colW = (A.w - 20) / 2, rh = wide ? 24 : Math.min(30, Math.max(22, (A.h - 150) / items.length));
        ['head', 'worker'].forEach((nm, c) => {
          const x = A.x + 10 + c * colW;
          V.text(g, nm, x, A.y + 12, { size: 12, weight: 600, color: c ? C.b : TF() });
          items.forEach((it, i) => {
            const y = A.y + 30 + i * rh;
            const on = e > 0.25 + i * 0.3 + c * 0.12;
            V.box(g, x, y - 9, colW - 16, rh - 4, { r: 6, fill: on ? V.alpha(C.a, 0.12) : V.alpha(C.muted, 0.06), stroke: on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.25) });
            V.text(g, it, x + 24, y + 5, { size: wide ? 11 : 10, color: on ? C.ink : C.muted, maxW: colW - 50 });
            if (on) mark(g, x + 12, y + 1, true, 5);
          });
        });
        // detecção do link, elo por elo
        const ly = A.y + 30 + items.length * rh + 14;
        V.text(g, 'detecção do link (nodes.sh)', A.x + 10, ly, { size: 11, color: C.muted, weight: 600 });
        const chain = wide ? ['ip route get <par>', 'interface (netdev)', 'HCA RoCE em /sys', 'GID v2 ::ffff:<IPv4>', 'NCCL_IB_GID_INDEX'] : ['rota', 'netdev', 'HCA', 'GID v2', 'env NCCL'];
        const n = chain.length, gap = wide ? 18 : 10, cw = (A.w - 20 - (n - 1) * gap) / n;
        chain.forEach((c, i) => {
          const x = A.x + 10 + i * (cw + gap), y = ly + 10, on = e > 2.6 + i * 0.35;
          chip(g, x, y, cw, 26, c, { fill: on ? V.alpha(C.e, 0.16) : V.alpha(C.muted, 0.06), stroke: on ? C.e : V.alpha(C.muted, 0.3), color: on ? C.ink : C.muted, size: wide ? 11 : 9.5 });
          if (i < n - 1) V.arrow(g, x + cw + 2, y + 13, x + cw + gap - 2, y + 13, { color: on ? C.e : V.alpha(C.muted, 0.4), lw: 1.4, head: 5 });
        });
        if (e > 4.5) V.wrap(g, 'Uma entrada zerada ou v1 faz o NCCL falhar “about a minute in”; o preflight confere antes de subir qualquer coisa.', A.x + 10, ly + 56, A.w - 20, { size: wide ? 11 : 10, color: C.muted, lineH: 14, maxLines: 2 });
      }
      function scene1(g, A, e, C, wide) {
        const leftW = wide ? A.w * 0.42 : A.w;
        const gh = wide ? A.h : A.h * 0.42;
        const cols = wide ? 11 : 13, n = 53, cs = Math.min(14, (leftW - 20) / cols - 3);
        const gx = A.x + 8, gy = A.y + 18;
        V.text(g, 'patches/*.patch (53)', gx, A.y + 10, { size: 11, color: C.muted });
        for (let i = 0; i < n; i++) {
          const c = i % cols, r = Math.floor(i / cols);
          const x = gx + c * (cs + 3), y = gy + r * (cs + 3);
          const ed = S.edit && i === 17;
          g.fillStyle = ed ? RED() : V.alpha(TF(), 0.35 + 0.35 * ((i * 7) % 5) / 5); V.rr(g, x, y, cs, cs, 3); g.fill();
        }
        const rowsN = Math.ceil(n / cols);
        const fy = gy + rowsN * (cs + 3) + 10;
        const fw = Math.min(leftW - 20, 220);
        V.box(g, gx, fy, fw, 26, { r: 7, fill: V.alpha(C.d, 0.14), stroke: C.d });
        V.text(g, 'SHA-256(patches + extras)[0:12]', gx + fw / 2, fy + 17, { size: 10.5, font: 'mono', color: C.ink, align: 'center', maxW: fw - 8 });
        for (let k = 0; k < 6; k++) { const u = (e * 0.9 + k / 6) % 1; V.dot(g, gx + 10 + k * (fw - 20) / 5, V.lerp(gy + rowsN * (cs + 3) - 4, fy, u), 2.2, V.alpha(TF(), 1 - u)); }
        const hs = wordHash(), nh = Math.min(12, Math.floor(e * 8));
        V.text(g, 'hash: ' + hs.slice(0, nh) + (nh < 12 ? '▌' : ''), gx, fy + 46, { size: 13, font: 'mono', weight: 600, color: S.edit ? RED() : C.a });
        if (S.edit) V.text(g, '(ilustrativo)', gx + tw(g, 'hash: ' + hs, 13, 'mono', 600) + 8, fy + 46, { size: 10, color: C.muted });
        // decisão
        const dx = wide ? A.x + leftW + 14 : A.x + 8, dy = wide ? A.y + 6 : fy + 62, dw = wide ? A.w - leftW - 20 : A.w - 16;
        const path = S.pull0 ? 'build' : S.edit ? 'build' : 'pull';
        const boxes = [
          { t: 'hash = IMAGE_TAG (ae8d1c789b47)?', on: e > 1.2 },
          path === 'pull' ? { t: 'sim → docker pull …@sha256:22789f0c…', on: e > 1.8, ok: true } : { t: S.pull0 ? 'PULL=0 → construir localmente' : 'não → tag v0.6.0-' + hs + ' publicada? não', on: e > 1.8, warn: true },
        ];
        const bh = wide ? 26 : 22;
        boxes.forEach((b, i) => {
          const y = dy + i * (bh + 8);
          V.box(g, dx, y, dw, bh, { r: 7, fill: b.on ? V.alpha(b.ok ? C.a : b.warn ? C.d : C.b, 0.16) : V.alpha(C.muted, 0.06), stroke: b.on ? (b.ok ? C.a : b.warn ? C.d : C.b) : V.alpha(C.muted, 0.3) });
          V.text(g, b.t, dx + 8, y + bh / 2 + 4, { size: wide ? 11 : 10, font: 'mono', color: b.on ? C.ink : C.muted, maxW: dw - 14 });
        });
        const ly = dy + 2 * (bh + 8);
        if (path === 'pull') {
          if (e > 2.4) { V.text(g, 'label tf.patches = ae8d1c789b47', dx + 8, ly + 14, { size: wide ? 11 : 10, font: 'mono', color: C.a, maxW: dw - 30 }); mark(g, dx + dw - 12, ly + 10, true, 5); }
          if (e > 3.0) V.wrap(g, '“a tag can be moved, a digest cannot”: a imagem puxada é exatamente a medida.', dx + 8, ly + 36, dw - 14, { size: wide ? 11.5 : 10.5, color: C.muted, lineH: 15, maxLines: 3 });
        } else {
          const layers = ['FROM nvcr.io/nvidia/pytorch:26.07-py3', 'pip install TensorFold@v0.6.0 + av + xgrammar', 'patch -p0 --forward × 53 (ordem de nome)', 'import server · engine · vision · av · xgrammar', 'LABEL tf.patches=' + hs];
          const lh = wide ? 21 : 18;
          layers.forEach((L, i) => {
            const on = e > 2.2 + i * 0.55;
            const y = ly + i * (lh + 4);
            V.box(g, dx + 10, y, dw - 10, lh, { r: 5, fill: on ? V.alpha(TF(), 0.18) : V.alpha(C.muted, 0.05), stroke: on ? V.alpha(TF(), 0.7) : V.alpha(C.muted, 0.2) });
            V.text(g, L, dx + 16, y + lh / 2 + 4, { size: wide ? 10.5 : 9.5, font: 'mono', color: on ? C.ink : C.muted, maxW: dw - 22 });
          });
        }
      }
      function scene2(g, A, e, C, wide) {
        const L = ['pytorch:26.07', 'TensorFold v0.6.0', 'av · xgrammar', '53 patches', 'label + env'];
        const sw = Math.min(200, A.w * 0.34), lh = Math.min(28, (A.h - 70) / 5);
        const xs = [A.x + 10, A.x + A.w - sw - 10];
        ['head', 'worker'].forEach((nm, c) => {
          V.text(g, nm, xs[c] + sw / 2, A.y + 12, { size: 12, weight: 600, color: c ? C.b : TF(), align: 'center' });
          L.forEach((l, i) => {
            const y = A.y + 22 + (L.length - 1 - i) * (lh + 4);
            const on = c === 0 || e > 0.6 + i * 0.45;
            V.box(g, xs[c], y, sw, lh, { r: 6, fill: on ? V.alpha(c ? C.b : TF(), 0.18) : V.alpha(C.muted, 0.05), stroke: on ? (c ? C.b : TF()) : V.alpha(C.muted, 0.25), lw: 1.2 });
            V.text(g, l, xs[c] + sw / 2, y + lh / 2 + 4, { size: wide ? 11 : 10, font: 'mono', color: on ? C.ink : C.muted, align: 'center', maxW: sw - 8 });
            if (c === 1 && e > 0.9 + i * 0.45) mark(g, xs[1] + sw + (wide ? 10 : -10), y + lh / 2, true, 5);
          });
        });
        const ay = A.y + 22 + 2 * (lh + 4) + lh / 2;
        V.arrow(g, xs[0] + sw + 8, ay, xs[1] - 8, ay, { color: C.d, lw: 2 });
        const u = (e * 0.7) % 1; V.dot(g, V.lerp(xs[0] + sw + 8, xs[1] - 8, u), ay, 4, C.d);
        V.text(g, wide ? 'pull ou docker save | ssh docker load' : 'save | ssh load', (xs[0] + sw + xs[1]) / 2, ay - 10, { size: wide ? 11 : 9.5, font: 'mono', color: C.d, align: 'center', maxW: xs[1] - xs[0] - sw - 10 });
        const by = A.y + 22 + 5 * (lh + 4) + 14;
        V.wrap(g, 'Identidade por conteúdo (camadas + config de runtime): o .Id é o digest do manifesto no containerd e o da config no overlay2, e nunca casaria entre stores diferentes (#8).', A.x + 10, by, A.w - 20, { size: wide ? 11.5 : 10.5, color: C.muted, lineH: 15, maxLines: 3 });
      }
      function scene3(g, A, e, C, wide) {
        const lines = ['hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/', '  snapshots/9eaebb7c…/  → links', '  blobs/  (~176 GB, xet: links p/ a raiz)', '  refs/main  ← gravado pelo script'];
        V.text(g, 'head · HF_CACHE (HF_HUB_OFFLINE=1 nos containers)', A.x + 10, A.y + 12, { size: 11.5, weight: 600, color: TF(), maxW: A.w - 20 });
        lines.forEach((l, i) => V.text(g, l, A.x + 14, A.y + 32 + i * 17, { size: wide ? 11 : 9.5, font: 'mono', color: e > 0.2 + i * 0.25 ? C.ink : C.muted, maxW: A.w - 24 }));
        const y = A.y + 32 + 4 * 17 + 14;
        V.text(g, 'worker', A.x + 10, y, { size: 11.5, weight: 600, color: C.b });
        const bw = A.w - 20;
        if (!S.nfs) {
          const p = Math.min(1, Math.max(0, (e - 0.6) / 3.2));
          V.box(g, A.x + 10, y + 10, bw, 22, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          g.fillStyle = V.alpha(C.b, 0.45); V.rr(g, A.x + 11, y + 11, (bw - 2) * p, 20, 5); g.fill();
          V.text(g, 'rsync -a -L --partial · ' + nf(166 * p, 0) + ' / 166 GiB', A.x + 18, y + 25, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, maxW: bw - 16 });
          if (p >= 1) { mark(g, A.x + 18, y + 48, true, 5); V.text(g, 'conferido arquivo a arquivo (caminho + tamanho); retoma se interrompido', A.x + 30, y + 52, { size: 10.5, color: C.muted, maxW: bw - 24 }); }
        } else {
          const pulse = 0.5 + 0.5 * Math.sin(e * 4);
          V.box(g, A.x + 10, y + 10, bw, 38, { r: 8, fill: V.alpha(C.e, 0.1 + 0.08 * pulse), stroke: C.e });
          V.text(g, 'volume docker NFS · ro · nfsvers=4.2 · nconnect=8 · rsize 1 MiB', A.x + 18, y + 26, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, maxW: bw - 16 });
          V.text(g, 'sem sudo no worker · ~166 GiB a menos de disco · os dois ranks no ar em ~2,2 min', A.x + 18, y + 41, { size: wide ? 10.5 : 9, color: C.muted, maxW: bw - 16 });
        }
        const py = y + 76;
        if (e > 1.2) {
          V.text(g, 'pins de revisão (commits do Hugging Face)', A.x + 10, py, { size: 11, weight: 600, color: C.e });
          V.text(g, 'MODEL_REVISION   = 9eaebb7c4e96d983dcd538e18624622ba5b820a8', A.x + 14, py + 18, { size: wide ? 10.5 : 8.5, font: 'mono', color: C.ink, maxW: A.w - 24 });
          V.text(g, 'DFLASH2_REVISION = bf582e4eacc1810f76656d1811693ff6c6737d2a', A.x + 14, py + 34, { size: wide ? 10.5 : 8.5, font: 'mono', color: C.ink, maxW: A.w - 24 });
          V.wrap(g, '“a new upstream commit changes nothing until the pin does”: os dois ranks servem exatamente esses snapshots do cache local.', A.x + 14, py + 54, A.w - 28, { size: wide ? 11 : 9.5, color: C.muted, lineH: 14, maxLines: 3 });
        }
      }
      function scene4(g, A, e, C, wide) {
        const items = [['argumentos (parser do TF, container sem GPU)', true], ['link: rota, HCA, GID RoCE v2', true], ['servidor anterior parado ou ausente', true], ['porta 8888 livre (ss -ltn)', true], [S.low ? 'memória: abaixo de ~110 GiB livres → aviso (ilustrativo)' : 'memória: ≥ ~110 GiB livres nos dois Sparks', !S.low]];
        const rh = Math.min(34, (A.h - 20) / items.length);
        items.forEach(([t, ok], i) => {
          const on = e > 0.3 + i * 0.4, y = A.y + 10 + i * rh;
          V.box(g, A.x + 10, y, A.w - 20, rh - 6, { r: 7, fill: on ? V.alpha(ok ? C.a : C.d, 0.12) : V.alpha(C.muted, 0.05), stroke: on ? (ok ? V.alpha(C.a, 0.6) : C.d) : V.alpha(C.muted, 0.25) });
          V.text(g, t, A.x + 36, y + (rh - 6) / 2 + 4, { size: wide ? 12 : 10.5, color: on ? C.ink : C.muted, maxW: A.w - 56 });
          if (on) { if (ok) mark(g, A.x + 22, y + (rh - 6) / 2, true, 5); else V.text(g, '⚠', A.x + 22, y + (rh - 6) / 2 + 5, { size: 13, color: C.d, align: 'center' }); }
        });
      }
      function scene5(g, A, e, C, wide) {
        const bw = Math.min(220, A.w * 0.4), bh = 70, y = A.y + 20;
        const xw = A.x + A.w - bw - 10, xh = A.x + 10;
        if (e > 0.3) { V.box(g, xw, y, bw, bh, { r: 10, fill: V.alpha(C.b, 0.12), stroke: C.b }); V.text(g, 'rank 1 · worker (ssh)', xw + 10, y + 18, { size: 11.5, weight: 600, color: C.b, maxW: bw - 16 }); V.text(g, 'tensorfold serve … --rank 1', xw + 10, y + 38, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: bw - 16 }); }
        if (e > 1.0) { V.box(g, xh, y, bw, bh, { r: 10, fill: V.alpha(TF(), 0.12), stroke: TF() }); V.text(g, 'rank 0 · head + API', xh + 10, y + 18, { size: 11.5, weight: 600, color: TF(), maxW: bw - 16 }); V.text(g, '--rank 0 --port 8888', xh + 10, y + 38, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: bw - 16 }); V.text(g, 'TCPStore :29551', xh + 10, y + 56, { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, maxW: bw - 16 }); }
        if (e > 1.5) {
          const ya = y + bh / 2; V.line(g, xh + bw, ya, xw, ya, { color: V.alpha(C.e, 0.7), lw: 2, dash: [5, 4], dashOffset: -e * 20 });
          const u = ((e - 1.5) * 0.6) % 1; chip(g, V.lerp(xh + bw, xw - 90, u), ya - 24, 90, 18, 'ncclUniqueId', { size: 10, fill: V.alpha(C.e, 0.18), stroke: C.e });
        }
        if (e > 2.0) {
          const keys = wide ? ['context', 'grade', 'copy', 'copy_max', 'split', 'max_rows', 'wide', 'kv', 'dense', 'policy'] : ['context', 'grade', 'copy', 'split', 'kv', 'policy'];
          const vals = { context: '1048576', grade: '64', copy: '1', copy_max: '15', split: '1', max_rows: '16', wide: '16', kv: 'fp8', dense: 'q4', policy: 'fnc7:0.3' };
          const yy = y + bh + 26, lw = wide ? 64 : 50, cw = (A.w - 20 - lw) / keys.length;
          V.text(g, 'lista de configurações, trocada por all-gather na partida', A.x + 10, yy - 6, { size: 11, color: C.muted, weight: 600, maxW: A.w - 20 });
          ['rank 0', 'rank 1'].forEach((rn, r) => {
            V.text(g, rn, A.x + 10, yy + 20 + r * 30, { size: 10.5, color: r ? C.b : TF(), weight: 600 });
            keys.forEach((k, i) => {
              const x = A.x + 10 + lw + i * cw, on = e > 2.2 + i * 0.12;
              chip(g, x + 1, yy + 6 + r * 30, cw - 3, 22, vals[k], { fill: V.alpha(on ? C.a : C.muted, on ? 0.14 : 0.06), stroke: on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.3), size: wide ? 10.5 : 9.5 });
              if (r === 0) V.text(g, k, x + cw / 2, yy + 70, { size: 9, font: 'mono', color: C.muted, align: 'center', maxW: cw - 2 });
            });
          });
          if (e > 2.2 + keys.length * 0.12) { mark(g, A.x + 18, yy + 90, true, 5); V.wrap(g, 'iguais: sobe. Se um TF_GLM_* diferisse, o servidor recusaria (“the two ranks were started with different settings”). Ocioso, o rank 1 espera na campainha da TCPStore.', A.x + 32, yy + 94, A.w - 44, { size: wide ? 11 : 10, color: C.ink, lineH: 14, maxLines: 3 }); }
        }
      }
      function scene6(g, A, e, C, wide) {
        const refitAt = 2.4, restartAt = 3.4;
        const bars = [{ n: 'head (rank 0)', est: 88.09, c: TF() }, { n: 'worker (rank 1)', est: 80, c: C.b }];
        const bx = A.x + (wide ? 130 : 96), bw = A.w - (wide ? 150 : 108), bh = 20;
        const tt = S.low && e > restartAt ? e - restartAt : e;
        const prog = (i) => { let p = Math.min(1, tt / 4.2); if (S.low && e < restartAt) p = Math.min(p, 0.12); return p; };
        bars.forEach((b, i) => {
          const y = A.y + 8 + i * (bh + 10);
          V.text(g, b.n, A.x + 8, y + 14, { size: wide ? 11.5 : 10, color: b.c, weight: 600, maxW: bx - A.x - 12 });
          V.box(g, bx, y, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          const p = prog(i);
          g.fillStyle = V.alpha(b.c, 0.5); V.rr(g, bx + 1, y + 1, (bw - 2) * p, bh - 2, 5); g.fill();
          V.text(g, nf(b.est * p, 1) + (i === 0 ? ' / 88,09 GiB (estimativa)' : ' / ~80 GiB de pesos (README)'), bx + 8, y + 14, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: bw - 12 });
        });
        // gráfico: GiB na GPU a cada batida de 15 s (o que o start.sh imprime)
        const cy = A.y + 2 * (bh + 10) + 22, chH = wide ? 120 : 110, cx = A.x + 40, cw2 = A.w - 56;
        V.text(g, 'GiB na GPU de cada Spark, uma batida a cada 15 s (curva esquemática)', A.x + 8, cy - 6, { size: wide ? 10.5 : 9.5, color: C.muted, maxW: A.w - 16 });
        V.line(g, cx, cy + chH, cx + cw2, cy + chH, { color: C.muted, lw: 1 }); V.line(g, cx, cy + 4, cx, cy + chH, { color: C.muted, lw: 1 });
        [0, 44, 88].forEach((v) => V.text(g, String(v), cx - 6, cy + chH - (v / 92) * (chH - 6) + 4, { size: 9.5, font: 'mono', color: C.muted, align: 'right' }));
        ['0', '2 min', '4 min', '6 min'].forEach((l, i) => V.text(g, l, cx + (i / 3) * cw2, cy + chH + 13, { size: 9.5, font: 'mono', color: C.muted, align: 'center' }));
        const beats = Math.floor(tt / 0.17);
        bars.forEach((b, i) => {
          const kMax = S.low && e < restartAt ? Math.min(beats, 5) : Math.min(beats, 24);
          const pts = [];
          for (let k = 0; k <= kMax; k++) {
            const f = Math.min(1, k / 24), gib = b.est * (S.low && e < restartAt ? Math.min(f, 0.12) : (1 - Math.pow(1 - f, 1.6)));
            pts.push([cx + (k / 24) * cw2 * (4 / 6), cy + chH - (gib / 92) * (chH - 6), k]);
          }
          g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
          g.strokeStyle = b.c; g.lineWidth = 2; g.stroke();
          pts.forEach(([x, y, k]) => { if (k % 4 === 0) V.dot(g, x, y, 2.5, b.c); });
        });
        const ly = cy + chH + 24;
        if (S.low && e > 0.8 && e < restartAt + 1) {
          V.box(g, A.x + 8, ly, A.w - 16, 38, { r: 7, fill: V.alpha(RED(), 0.1), stroke: RED() });
          V.text(g, 'rank 0: largest fitting prompt-plus-reply window: 786432 tokens', A.x + 16, ly + 16, { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, maxW: A.w - 32 });
          V.text(g, e > refitAt ? '→ start.sh reinicia uma vez com --context 786432 e avisa (valor ilustrativo)' : 'recusado antes de carregar os pesos (valor ilustrativo)', A.x + 16, ly + 31, { size: wide ? 10 : 9, color: e > refitAt ? C.d : C.muted, maxW: A.w - 32, weight: e > refitAt ? 600 : 400 });
        } else if (tt > 4.4) { mark(g, A.x + 18, ly + 12, true, 5); V.text(g, 'os dois ranks carregados; a 1ª partida de cada imagem também compila os kernels', A.x + 30, ly + 16, { size: wide ? 11 : 9.5, color: C.ink, maxW: A.w - 40 }); }
      }
      function scene7(g, A, e, C, wide) {
        const bw = Math.min(260, A.w * 0.7);
        if (e > 0.2) { V.box(g, A.x + 10, A.y + 10, bw, 30, { r: 12, fill: V.alpha(C.b, 0.16), stroke: C.b }); V.text(g, 'Reply with OK.', A.x + 22, A.y + 30, { size: 12, color: C.ink }); V.text(g, 'temperature 0 · max_tokens 32 · thinking off', A.x + 14, A.y + 54, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted, maxW: A.w - 20 }); }
        if (e > 1.0) { V.box(g, A.x + A.w - 90, A.y + 64, 80, 30, { r: 12, fill: V.alpha(C.a, 0.18), stroke: C.a }); V.text(g, 'OK', A.x + A.w - 50, A.y + 84, { size: 13, weight: 700, color: C.ink, align: 'center' }); }
        if (e > 1.8) {
          const y = A.y + 116, pulse = 0.6 + 0.4 * Math.sin(e * 3);
          V.glow(g, A.x + A.w / 2, y + 18, 120, C.a, 0.35 * pulse);
          V.box(g, A.x + 10, y, A.w - 20, 40, { r: 10, fill: V.alpha(C.a, 0.14), stroke: C.a, lw: 2 });
          V.text(g, '✔ GLM-5.3-Flash-EXL3 is now LIVE! on port 8888', A.x + A.w / 2, y + 25, { size: wide ? 13.5 : 11, font: 'mono', weight: 600, color: C.ink, align: 'center', maxW: A.w - 30 });
        }
      }
      const SC = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7];
      H3 = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const i = st.i;
        if (i !== prevI) { moveFrom = prevI; prevI = i; moveT0 = t; }
        const ly = 20, lx0 = 22, lx1 = w - 22, step = (lx1 - lx0) / 7;
        V.line(g, lx0, ly, lx1, ly, { color: V.alpha(C.muted, 0.4), lw: 2 });
        NAMES.forEach((nm, k) => {
          const x = lx0 + k * step, on = k === i, past = k < i;
          V.dot(g, x, ly, on ? 8 : 6, on ? TF() : past ? V.alpha(C.a, 0.8) : V.alpha(C.muted, 0.35));
          V.text(g, String(k + 1), x, ly + 4, { size: 9.5, weight: 700, color: on || past ? onTile() : C.ink, align: 'center' });
          if (wide || on) V.text(g, nm, Math.max(30, Math.min(w - 30, x)), ly + 22, { size: wide ? 10.5 : 10, color: on ? C.ink : C.muted, align: 'center', weight: on ? 600 : 400 });
        });
        const u = V.ease.inOut(Math.min(1, (t - moveT0) / 0.6));
        const px = V.lerp(lx0 + moveFrom * step, lx0 + i * step, u);
        V.glow(g, px, ly, 16, TF(), 0.7);
        const A = { x: 4, y: 52, w: w - 8, h: h - 58 };
        g.save(); g.beginPath(); g.rect(0, A.y - 4, w, h - A.y + 4); g.clip();
        SC[i](g, A, st.elapsed(), C, wide);
        g.restore();
      });
    })();

    /* ================================================ FIG 4 — orçamento de memória */
    (function () {
      const fig = root.querySelector('#fig-orcamento');
      adaptH(fig, 420, 560);
      const S = { mem: 116, res: 14.5, cap: 12.5, vis: true };
      const GiB = 1073741824, BPT = 6204, WIN = 1048576, ENT = 32 * 45 * 1048576;
      // além da estimativa: FIX desde a partida (inferido: ~10 de pico − 1,4 do prompt de 1M); D1M = 7,0 − 5,6 GiB medidos
      const FIX = 8.6, D1M = 1.4;
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'MemAvailable na partida', min: 100, max: 122, step: 0.5, value: 116, fmt: (v) => nf(v, 1) + ' GiB', onInput: (v) => { S.mem = v; upd(); } });
      UI.slider(ctl, { label: 'MEMORY_RESERVE_GIB', min: 4, max: 24, step: 0.5, value: 14.5, fmt: (v) => nf(v, 1), onInput: (v) => { S.res = v; upd(); } });
      UI.slider(ctl, { label: 'KV_POOL_GIB', min: 3, max: 24, step: 0.5, value: 12.5, fmt: (v) => nf(v, 1), onInput: (v) => { S.cap = v; upd(); } });
      UI.toggle(ctl, { label: 'VISION=1 (torre no rank 0)', value: true, onChange: (v) => { S.vis = v; upd(); } });
      const rP = UI.readout(ctl, 'pool', '—'), rN = UI.readout(ctl, 'pool ≈ tokens', '—'), rI = UI.readout(ctl, 'livre ocioso', '—'), rF = UI.readout(ctl, 'livre mínimo sob 1M', '—'), rV = UI.readout(ctl, 'veredito', '—');
      let M = {};
      function calc() {
        const est = 88.09 - (S.vis ? 0 : 1.8);
        const room = S.mem - S.res - est;
        const winG = (WIN * BPT) / GiB;
        if (room < 0) {
          const avail = winG + room;
          const win = avail > 0 ? Math.floor((avail * GiB) / BPT / 2048) * 2048 : 0;
          return { est, room, pool: 0, refused: true, win, tokens: win, idleFree: S.res - FIX, minFree: S.res - FIX - D1M, left: 0 };
        }
        const pool = Math.min(S.cap, room);
        const tokens = WIN + Math.floor(Math.max(0, pool * GiB - ENT) / BPT / 2048) * 2048;
        const left = room - pool;
        return { est, room, pool, refused: false, win: WIN, tokens, idleFree: S.res + left - FIX, minFree: S.res + left - FIX - D1M, left };
      }
      function upd() {
        M = calc();
        if (M.refused) { rP.set('—', 'bad'); rN.set(M.win > 0 ? 'janela ' + nf(M.win) : 'não cabe', 'bad'); rV.set(M.win > 0 ? 'recusa → refit' : 'não cabe', 'bad'); }
        else { rP.set(nf(M.pool, 2) + ' GiB' + (M.pool >= S.cap - 1e-9 ? ' (teto)' : '')); rN.set('~' + nf(M.tokens / 1e6, 2) + 'M'); rV.set(M.pool >= S.cap - 1e-9 ? 'cabe, pool no teto' : 'cabe, pool pela memória', 'ok'); }
        const mf = M.minFree, fi = M.idleFree;
        if (M.refused) { rI.set('—'); rF.set('—'); }
        else {
          rI.set('~' + nf(Math.max(0, fi), 1) + ' GiB', fi < 1 ? 'bad' : fi < 3 ? 'warn' : 'ok');
          rF.set('~' + nf(Math.max(0, mf), 1) + ' GiB', mf < 1 ? 'bad' : mf < 3 ? 'warn' : 'ok');
        }
        if (!M.refused && mf < 1) rV.set('risco de congelar', 'bad');
      }
      upd();
      const snow = Array.from({ length: 26 }, (_, i) => ({ x: (i * 0.618) % 1, s: 0.4 + ((i * 0.37) % 1) * 0.8 }));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const top = 16, bot = h - 26, Hh = bot - top;
        const MAXV = 125, BRK = 75;
        const yv = (v) => { const f = v < BRK ? (v / BRK) * 0.18 : 0.18 + ((v - BRK) / (MAXV - BRK)) * 0.82; return bot - f * Hh; };
        const tx = wide ? 70 : 52, twid = wide ? Math.min(150, w * 0.2) : Math.min(110, w * 0.3);
        // eixo
        [0, 75, 85, 95, 105, 115, 125].forEach((v) => { V.line(g, tx - 6, yv(v), tx + twid + 6, yv(v), { color: V.alpha(C.grid, 1), lw: 1 }); V.text(g, v + ' GiB', tx - 10, yv(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }); });
        // quebra
        const yb = yv(BRK) + 6; g.strokeStyle = C.muted; g.lineWidth = 1.2; g.beginPath(); g.moveTo(tx - 4, yb + 3); g.lineTo(tx + 4, yb - 3); g.moveTo(tx - 4, yb + 8); g.lineTo(tx + 4, yb + 2); g.stroke();
        // tanque
        const memTop = yv(S.mem);
        V.box(g, tx, memTop, twid, bot - memTop, { r: 6, stroke: V.alpha(C.ink, 0.6), lw: 1.6 });
        const segs = [];
        let acc = 0;
        const push = (v, col, lab, hatch) => { if (v <= 0) return; segs.push({ a: acc, b: acc + v, col, lab, hatch }); acc += v; };
        const est = M.est;
        push(80, V.alpha(C.b, 0.45), 'pesos ~80');
        push(6.06, V.alpha(C.c, 0.55), 'janela 1M');
        if (S.vis) push(1.8, V.alpha(C.f, 0.6), 'visão');
        push(est - acc, V.alpha(C.muted, 0.4), '');
        if (!M.refused) { push(M.pool, V.alpha(C.e, 0.6), 'pool ' + nf(M.pool, 1)); push(M.left, null, M.left > 0.6 ? 'sobra' : ''); }
        const resA = S.mem - S.res;
        segs.forEach((s) => {
          const ya = yv(Math.min(s.b, S.mem)), ybb = yv(Math.max(0, s.a));
          if (ybb - ya < 0.5) return;
          if (s.col) { g.fillStyle = s.col; g.fillRect(tx + 2, ya, twid - 4, ybb - ya); }
          if (s.lab && ybb - ya >= 13) V.text(g, s.lab, tx + twid / 2, (ya + ybb) / 2 + 4, { size: 10.5, color: C.ink, align: 'center', weight: 600, maxW: twid - 8 });
        });
        // reserva (hachurada)
        if (resA > 0) {
          const ya = memTop, ybb = yv(Math.max(0, Math.min(S.mem, resA)));
          g.save(); g.beginPath(); g.rect(tx + 2, ya + 1, twid - 4, ybb - ya); g.clip();
          g.strokeStyle = V.alpha(C.d, 0.55); g.lineWidth = 1.2;
          for (let k = -40; k < twid + 40; k += 8) { g.beginPath(); g.moveTo(tx + k, ya); g.lineTo(tx + k + 30, ybb + 30); g.stroke(); }
          g.restore();
          V.line(g, tx, ybb, tx + twid, ybb, { color: C.d, lw: 1.5, dash: [5, 3] });
        }
        // além da estimativa: bloco fixo desde a partida + pulso de um prompt de 1M
        const ph = (t % 4) / 4, amp = Math.max(0, Math.sin(Math.min(1, ph / 0.6) * Math.PI));
        const fixA = M.refused ? est : est + M.pool, fixB = fixA + FIX;
        if (!M.refused) {
          const ya = yv(Math.min(fixB, MAXV)), ybb = yv(fixA);
          g.fillStyle = V.alpha(C.d, 0.62); g.fillRect(tx + 2, ya, twid - 4, ybb - ya);
          g.strokeStyle = C.d; g.lineWidth = 1.2; g.strokeRect(tx + 2.5, ya + 0.5, twid - 5, ybb - ya - 1);
          if (ybb - ya >= 26) { V.text(g, 'fora da', tx + twid / 2, (ya + ybb) / 2 - 2, { size: 10, color: C.ink, align: 'center', weight: 600 }); V.text(g, 'estimativa ~8,6', tx + twid / 2, (ya + ybb) / 2 + 10, { size: 10, color: C.ink, align: 'center', weight: 600, maxW: twid - 8 }); }
        }
        const pTop = fixB + D1M * amp;
        const outside = wide;                       // no desktop, rótulos ao lado da coluna; no celular, dentro
        const ax = tx + twid + 8;
        if (!M.refused && amp > 0.02) {
          const ya = yv(Math.min(pTop, MAXV)), ybb = yv(fixB);
          g.fillStyle = V.alpha(RED(), 0.45 + 0.35 * amp); g.fillRect(tx + 2, ya, twid - 4, ybb - ya);
          V.glow(g, tx + twid / 2, (ya + ybb) / 2, 22, RED(), 0.5 * amp);
          if (outside) { V.line(g, tx + twid - 2, (ya + ybb) / 2, ax + 4, (ya + ybb) / 2, { color: RED(), lw: 1.2 }); V.text(g, '+1,4 GiB (prompt 1M)', ax + 8, (ya + ybb) / 2 + 4, { size: 10, color: RED(), weight: 700 }); }
          else if (ybb - ya >= 9) V.text(g, '+1,4 (1M)', tx + twid / 2, (ya + ybb) / 2 + 3.5, { size: 9, color: C.dark ? C.ink : '#ffffff', align: 'center', weight: 700 });
        }
        if (!M.refused && resA > 0) V.line(g, tx, yv(resA), tx + twid, yv(resA), { color: C.d, lw: 1.5, dash: [5, 3] });
        // o que sobra livre agora (acima do bloco e do pulso)
        if (!M.refused) {
          const freeNow = S.mem - pTop;
          const ya = memTop, ybb = yv(Math.min(S.mem, pTop));
          if (freeNow > 0) {
            if (outside) {
              const ym = (ya + ybb) / 2;
              g.strokeStyle = C.ink; g.lineWidth = 1.2; g.beginPath(); g.moveTo(ax, ya + 1); g.lineTo(ax + 4, ya + 1); g.lineTo(ax + 4, ybb - 1); g.lineTo(ax, ybb - 1); g.stroke();
              V.text(g, 'livre agora', ax + 8, ym - 2, { size: 10, color: C.ink, weight: 600 });
              V.text(g, '~' + nf(freeNow, 1) + ' GiB', ax + 8, ym + 11, { size: 10, font: 'mono', color: C.ink, weight: 600 });
            } else if (ybb - ya >= 26) {
              V.text(g, 'livre agora', tx + twid / 2, ya + (ybb - ya) / 2 - 1, { size: 9.5, color: C.ink, align: 'center', weight: 600 });
              V.text(g, '~' + nf(freeNow, 1) + ' GiB', tx + twid / 2, ya + (ybb - ya) / 2 + 11, { size: 10, font: 'mono', color: C.ink, align: 'center', weight: 600 });
            }
          }
        }
        V.text(g, 'MemAvailable', tx + twid / 2, memTop - 5, { size: 10.5, font: 'mono', color: C.ink, align: 'center' });
        // painel de texto
        const px = tx + twid + (wide ? 136 : 16), pw = w - px - 8;
        const lines = [
          [C.b, 'pesos ~80 GiB (experts EXL3 + densos q4 + drafter)'],
          [C.c, 'janela de 1M ≈ 6,1 GiB (6.204 B/token, estimativa)'],
          [C.f, S.vis ? 'torre de visão 1,8 GiB (rank 0)' : 'VISION=0: −1,8 GiB, volta ao pool'],
          [C.muted, 'estimativa de partida: ' + nf(est, 2) + ' GiB'],
          [C.e, M.refused ? 'pool: nada (a janela não cabe)' : 'pool: ' + nf(M.pool, 2) + ' GiB → ~' + nf(M.tokens) + ' tokens'],
          [V.alpha(C.d, 0.4), 'reserva: ' + nf(S.res, 1) + ' GiB (o orçamento a deixa livre)'],
          [V.alpha(C.d, 0.85), 'fora da estimativa desde a partida: ~8,6 GiB (CUDA, NCCL, graphs, workspaces; inferido)'],
          [RED(), 'prompt de 1M: mais ~1,4 GiB (7,0 → 5,6 GiB medidos)'],
        ];
        let yy = top + 8;
        const fs = wide ? 12 : 10.5, lh = wide ? 17 : 14;
        lines.forEach(([c, s]) => { g.fillStyle = c; V.rr(g, px, yy - 8, 9, 9, 2); g.fill(); yy += V.wrap(g, s, px + 15, yy, pw - 16, { size: fs, color: C.ink, lineH: lh, maxLines: wide ? 2 : 3 }) + (wide ? 4 : 3); });
        yy += 6;
        const mf = M.minFree, fi = M.idleFree;
        let verdict, vc;
        if (M.refused) { verdict = M.win > 0 ? 'Não cabe: o servidor recusa antes de carregar e informa a maior janela (~' + nf(M.win) + ' tokens); o start.sh reinicia com ela.' : 'Não cabe nem a janela: libere memória ou reduza a reserva com cuidado.'; vc = RED(); }
        else if (mf < 1) { verdict = 'Livre mínimo ≈ ' + nf(Math.max(0, mf), 1) + ' GiB: o bloco fixo e o pulso de um prompt de 1M comem a folga. Em UMA isso tende a congelar o Spark.'; vc = RED(); }
        else if (mf < 3) { verdict = 'Parado ≈ ' + nf(fi, 1) + ' GiB livres; sob 1M ≈ ' + nf(mf, 1) + ' GiB: apertado (livre mínimo ≈ reserva + sobra − ~10).'; vc = C.d; }
        else { verdict = 'Parado ≈ ' + nf(fi, 1) + ' GiB livres; sob um prompt de 1M ≈ ' + nf(mf, 1) + ' GiB (medidos nos defaults: 7,0 e 5,6). Pool medido: 2.922.496 tokens.'; vc = C.a; }
        yy += V.wrap(g, verdict, px, yy, pw, { size: fs, color: vc, weight: 600, lineH: lh, maxLines: 5 }) + 14;
        // como 4 streams dividem o pool
        if (!M.refused && yy + 60 < h) {
          const T = M.tokens, cyc = Math.floor(t / 4) % 2;
          const sh = Math.max(0, Math.floor((T - 1048576) / 3 / 2048) * 2048), shl = '~' + nf(sh / 1000, 0) + 'k';
          const ex = cyc === 0 ? [[1048576, C.a, '1M'], [1048576, C.b, '1M']] : [[1048576, C.a, '1M'], [sh, C.c, shl], [sh, C.e, shl], [sh, hsl(205, 1), shl]];
          V.text(g, cyc === 0 ? 'o pool dividido: 1M ao lado de outro 1M' : 'o pool dividido: 1M ao lado de três de ' + shl, px, yy, { size: fs - 0.5, color: C.muted, maxW: pw });
          const bY = yy + 8, bW = pw, bH = 18;
          V.box(g, px, bY, bW, bH, { r: 5, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.4) });
          let x0 = px + 1;
          ex.forEach(([n, c, l]) => { const ww = Math.min(bW - 2 - (x0 - px - 1), (n / T) * (bW - 2)); if (ww <= 0) return; g.fillStyle = V.alpha(c, 0.6); V.rr(g, x0, bY + 1, ww, bH - 2, 4); g.fill(); if (ww > 26) V.text(g, l, x0 + ww / 2, bY + 13, { size: 10, font: 'mono', color: C.ink, align: 'center' }); x0 += ww + 1; });
          const used = ex.reduce((a, e2) => a + e2[0], 0);
          V.text(g, used <= T ? 'cabe: sobra ~' + nf(Math.max(0, T - used) / 1e3, 0) + 'k para prompts guardados' : 'não cabe agora: a requisição espera (os prompts guardados cedem primeiro)', px, bY + bH + 14, { size: fs - 1, color: used <= T ? C.ink : C.d, maxW: pw });
        }
        // congelamento
        if (!M.refused && mf < 1) {
          snow.forEach((s, k) => { const y = top + ((t * 30 * s.s + k * 37) % Hh); V.text(g, '❄', tx + 6 + s.x * (twid - 12), y, { size: 10 + s.s * 6, color: V.alpha(C.b, 0.75), align: 'center' }); });
          const cyy = top + Hh * 0.5; g.fillStyle = V.alpha(C.bg || '#000', 0.75); V.rr(g, tx + 8, cyy - 16, twid - 16, 26, 6); g.fill(); V.text(g, 'congela ❄', tx + twid / 2, cyy + 2, { size: 13, weight: 700, color: RED(), align: 'center' });
        }
        V.text(g, 'escala comprimida abaixo de 75 GiB', tx + twid / 2, h - 8, { size: 9.5, color: C.muted, align: 'center', maxW: twid + 60 });
      });
    })();

    /* ================================================ FIG 5 — formatos: q4 e FP8 */
    (function () {
      const fig = root.querySelector('#fig-formatos');
      const N = LA.num;
      const S = { mode: 'q4', auto: true, i: 0, scale: 'pow2', seed: 3 };
      const stage5 = fig.querySelector('.la-fig-stage');
      const setH5 = () => { const want = (fig.clientWidth < 600 ? (S.mode === 'q4' ? 430 : 530) : 410) + 'px'; if (stage5.style.height !== want) stage5.style.height = want; };
      setH5();
      { const ro5 = new ResizeObserver(setH5); ro5.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro5.disconnect()); }
      const bf = (x) => (N && N.round ? N.round('bf16', x) : x);
      let grp = [], Q = [];
      function mkGroup() {
        const r = V.rng(102);
        grp = Array.from({ length: 64 }, () => { const u = r(); return (u < 0.5 ? -1 : 1) * -Math.log(1 - Math.abs(2 * u - 1) + 1e-9) * 0.7; });
        Q = [];
        const mn = Math.min(...grp), mx = Math.max(...grp);
        for (let i = 0; i < 16; i++) {
          const a = 1 - (0.45 * i) / 15;
          const lo = bf(a * mn), hi = a * mx;
          const s = bf((hi - lo) / 15);
          const q = grp.map((g) => Math.max(0, Math.min(15, Math.round((g - lo) / s))));
          const dq = q.map((c) => s * c + lo);
          const mse = grp.reduce((acc, g, k) => acc + (g - dq[k]) ** 2, 0) / 64;
          Q.push({ a, lo, hi, s, q, dq, mse });
        }
      }
      mkGroup();
      const best = () => Q.reduce((b, x, k) => (x.mse < Q[b].mse ? k : b), 0);
      let row = [], R2 = null;
      function mkRow() {
        const r = V.rng(100 + S.seed * 13);
        row = Array.from({ length: 512 }, () => r.normal() * 0.8);
        for (let k = 0; k < 4; k++) row[r.int(512)] = (r() < 0.5 ? -1 : 1) * (2.6 + r() * 1.4);
        R2 = null;
      }
      mkRow();
      function quantRow(mode) {
        const amax = Math.max(...row.map(Math.abs));
        const s = mode === 'pow2' ? Math.pow(2, Math.ceil(Math.log2(amax)) - 8) : amax / 448;
        const xh = row.map((x) => (N && N.round ? N.round('e4m3', x / s) : Math.round(x / s)) * s);
        let num = 0, den = 0; for (let k = 0; k < 512; k++) { num += (row[k] - xh[k]) ** 2; den += row[k] ** 2; }
        const rms = Math.sqrt(den / 512);
        return { amax, s, xh, rel: Math.sqrt(num / den), rms, top: amax / s };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Formato', options: [{ v: 'q4', label: 'q4 · grupo de 64' }, { v: 'fp8', label: 'FP8 · linha do KV' }], value: 'q4', onChange: (v) => { S.mode = v; gQ.show(v === 'q4'); gF.show(v === 'fp8'); setH5(); upd(); } });
      const gQ = UI.group(ctl), gF = UI.group(ctl);
      const tg = UI.toggle(gQ.el, { label: 'busca automática', value: true, onChange: (v) => { S.auto = v; } });
      UI.slider(gQ.el, { label: 'recorte i', min: 0, max: 15, step: 1, value: 0, fmt: (v) => 'i = ' + v + ' · a = ' + nf(1 - (0.45 * v) / 15, 2), onInput: (v) => { S.i = v; if (S.auto) { S.auto = false; tg.set(false); } upd(); } });
      UI.seg(gF.el, { label: 'Escala', options: [{ v: 'pow2', label: 'potência de 2' }, { v: 'fine', label: 'fp32 fina (amax/448)' }], value: 'pow2', onChange: (v) => { S.scale = v; upd(); } });
      UI.button(gF.el, '↻ nova linha', () => { S.seed++; mkRow(); upd(); });
      gF.show(false);
      const r1 = nocase(UI.readout(ctl, 'a / escala', '—')), r2 = nocase(UI.readout(ctl, 'erro', '—')), r3 = nocase(UI.readout(ctl, 'referência', '—'));
      let curI = 0;
      function upd() {
        if (S.mode === 'q4') {
          const k = S.auto ? curI : S.i, b = best();
          r1.set('a = ' + nf(Q[k].a, 2) + ' · s = ' + nf(Q[k].s, 3));
          r2.set('MSE ' + nf(Q[k].mse, 4), k === b ? 'ok' : null);
          r3.set('melhor i* = ' + b + ' (MSE ' + nf(Q[b].mse, 4) + ')');
        } else {
          const A = quantRow('pow2'), B = quantRow('fine'), cur = S.scale === 'pow2' ? A : B;
          const e2 = Math.round(Math.log2(A.s)); r1.set('s = ' + (S.scale === 'pow2' ? '2' + (e2 < 0 ? '⁻' : '') + String(Math.abs(e2)).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('') : nf(B.s, 5)));
          r2.set(nf(cur.rel * 100, 2) + '% (linha)');
          r3.set('outra escala: ' + nf((S.scale === 'pow2' ? B : A).rel * 100, 2) + '%');
        }
      }
      upd();
      let lastScan = -1;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 600;
        if (S.mode === 'q4') {
          if (S.auto) { const ph = t % 7; const k = ph < 4.8 ? Math.min(15, Math.floor(ph / 0.3)) : best(); curI = k; if (k !== lastScan) { lastScan = k; upd(); } }
          const k = S.auto ? curI : S.i, cur = Q[k];
          const cell = wide ? Math.min(31, (w * 0.64 - 30) / 16 - 2) : Math.min(21, (w - 34) / 16 - 1.5);
          const gap = wide ? 2 : 1.5;
          const mW = 8 * cell + 7 * gap;
          const mx0 = 10, my0 = 22;
          const data = []; for (let i = 0; i < 8; i++) data.push(grp.slice(i * 8, i * 8 + 8));
          const gm = Math.max(...grp.map(Math.abs));
          const compact = cell < 26;
          cells(g, mx0, my0, data, cell, cell, gap, (v) => divFill(v, gm), (v) => cellNum(v, compact && v < 0), compact ? 'g (64 pesos; −,8 = −0,8)' : 'g (64 pesos, cauda pesada)');
          const codes = []; const errs = [];
          for (let i = 0; i < 8; i++) { codes.push(cur.q.slice(i * 8, i * 8 + 8)); errs.push(grp.slice(i * 8, i * 8 + 8).map((v, j) => Math.abs(v - cur.dq[i * 8 + j]))); }
          const m2x = mx0 + mW + (wide ? 22 : 12);
          cells(g, m2x, my0, codes, cell, cell, gap, (v, i, j) => heatFill(errs[i][j] / (Q[0].s * 0.8)), (v) => String(v), 'q (4 bits) · cor = erro');
          // MSE bars
          const bx = wide ? m2x + mW + 26 : 12, by = wide ? 24 : my0 + 8 * (cell + gap) + 28, bw = wide ? w - bx - 12 : w - 24, bh = wide ? h * 0.42 : Math.max(90, h - by - 104);
          V.text(g, 'MSE de cada recorte a_i', bx, by - 6, { size: 11, color: C.muted });
          const mxm = Math.max(...Q.map((q) => q.mse));
          const bb = best();
          V.bars(g, { x: bx, y: by + 4, w: bw, h: bh - 22, values: Q.map((q) => q.mse), max: mxm * 1.05, gap: 3, colorFn: (v, i) => (i === k ? TF() : i === bb ? C.a : V.alpha(C.b, 0.5)) });
          [0, 5, 10, 15].forEach((i) => V.text(g, String(i), bx + (i + 0.5) * (bw / 16), by + bh - 4, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
          V.text(g, 'i →', bx + bw, by + bh + 10, { size: 10, color: C.muted, align: 'right' });
          // reta numérica dos 16 níveis
          const ny = (wide ? Math.max(my0 + 8 * (cell + gap), by + bh) : by + bh) + 46, nx0 = 14, nx1 = w - 14;
          const vmin = -Math.max(...grp.map(Math.abs)) * 1.05, vmax = -vmin, xv = (v) => nx0 + ((v - vmin) / (vmax - vmin)) * (nx1 - nx0);
          V.line(g, nx0, ny, nx1, ny, { color: C.muted, lw: 1 });
          for (let c = 0; c <= 15; c++) { const v = cur.lo + cur.s * c; V.line(g, xv(v), ny - 7, xv(v), ny + 7, { color: TF(), lw: 1.4 }); }
          const lblX = (x) => V.clamp(x, nx0 + 36, nx1 - 36);
          V.text(g, 'lo = ' + (cur.lo < 0 ? '−' : '') + nf(Math.abs(cur.lo), 2), lblX(xv(cur.lo)), ny + 20, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'hi = ' + nf(cur.lo + 15 * cur.s, 2), lblX(xv(cur.lo + 15 * cur.s)), ny + 20, { size: 10, font: 'mono', color: C.muted, align: 'center' });
          grp.forEach((v, j) => { const clipped = v < cur.lo - cur.s / 2 || v > cur.lo + 15.5 * cur.s; V.dot(g, xv(v), ny - 13 - (j % 4) * 3, clipped ? 3 : 2, clipped ? RED() : V.alpha(C.b, 0.75)); });
          V.text(g, '16 níveis entre lo e hi; pontos vermelhos = recortados', (nx0 + nx1) / 2, ny + 36, { size: wide ? 11 : 10, color: C.muted, align: 'center', maxW: nx1 - nx0 });
        } else {
          if (!R2) R2 = { A: quantRow('pow2'), B: quantRow('fine') };
          const cur = S.scale === 'pow2' ? R2.A : R2.B;
          const cell = wide ? Math.min(40, (w * 0.55 - 20) / 8 - 3) : Math.min(38, (w - 30) / 8 - 3);
          const gap = 3;
          const data = [], er = [];
          for (let i = 0; i < 4; i++) { data.push(row.slice(i * 8, i * 8 + 8)); er.push(row.slice(i * 8, i * 8 + 8).map((v, j) => Math.abs(v - cur.xh[i * 8 + j]) / Math.max(1e-6, Math.abs(v)))); }
          cells(g, 10, 22, data, cell, 22, gap, (v) => divFill(v, 3), (v) => (Math.abs(v) < 0.005 ? '0,00' : (v < 0 ? '−' : '') + nf(Math.abs(v), 2)), '32 dos 512 valores da linha');
          const y2 = 22 + 4 * (22 + gap) + 24;
          cells(g, 10, y2, er, cell, 22, gap, (v) => heatFill(v / 0.08), (v) => nf(v * 100, 1) + '%', 'erro relativo de cada valor');
          // histograma em escala log2 de |x|/s
          const hx = wide ? 10 + 8 * (cell + gap) + 30 : 12, hy = wide ? 30 : y2 + 4 * (22 + gap) + 30, hw = wide ? w - hx - 12 : w - 24, hh = wide ? 150 : 110;
          const lo2 = -10, hi2 = 9, nb = hi2 - lo2;
          const cnt = new Array(nb).fill(0);
          row.forEach((x) => { const a = Math.abs(x) / cur.s; if (a <= 0) return; const b = Math.floor(Math.log2(a)) - lo2; if (b >= 0 && b < nb) cnt[b]++; });
          const mc = Math.max(...cnt);
          V.text(g, '|x| / s em potências de 2 (e4m3: normais de 2⁻⁶ a 448)', hx, hy - 8, { size: 11, color: C.muted, maxW: hw });
          const bwid = hw / nb;
          const xl = (e) => hx + (e - lo2) * bwid;
          g.fillStyle = V.alpha(C.a, 0.08); g.fillRect(xl(-6), hy, xl(Math.log2(448)) - xl(-6), hh);
          g.fillStyle = V.alpha(TF(), 0.16); g.fillRect(xl(7), hy, xl(8) - xl(7), hh);
          cnt.forEach((c, b) => { const bh = (c / mc) * (hh - 14); g.fillStyle = V.alpha(C.b, 0.75); V.rr(g, hx + b * bwid + 1, hy + hh - bh, bwid - 2, Math.max(1, bh), 2); g.fill(); });
          [-9, -6, 0, 4, 7, 8].forEach((e) => V.text(g, '2' + (e < 0 ? '⁻' : '') + String(Math.abs(e)).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join(''), xl(e), hy + hh + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
          const mxPos = xl(Math.log2(cur.top));
          V.line(g, mxPos, hy - 2, mxPos, hy + hh, { color: TF(), lw: 2 });
          V.text(g, 'amax/s = ' + nf(cur.top, 1), mxPos - 4, hy + 12, { size: 10.5, font: 'mono', color: TF(), align: 'right', weight: 600 });
          // bytes por linha
          const by = Math.max(y2 + 4 * (22 + gap) + 30, hy + hh + 34), bw = wide ? w - 22 : hw;
          const hxB = wide ? 10 : hx;
          const rowsB = [['BF16', 1024, C.muted], ['vLLM fp8_ds_mla', 656, GLM()], ['TensorFold FP8', 528, TF()]];
          const ph = Math.min(1, (t % 6) / 1.4);
          rowsB.forEach(([nm, b, c], i) => {
            const y = by + i * 22;
            if (y + 16 > h - 4) return;
            const lw = wide ? 130 : 108;
            V.text(g, nm, hxB, y + 12, { size: 10.5, color: C.ink, maxW: lw - 4 });
            g.fillStyle = V.alpha(c, 0.5); V.rr(g, hxB + lw, y + 2, ((bw - lw - 50) * b / 1024) * ph, 14, 4); g.fill();
            V.text(g, nf(b) + ' B', hxB + lw + (bw - lw - 50) * b / 1024 * ph + 6, y + 13, { size: 10.5, font: 'mono', color: C.ink });
          });
        }
      });
    })();

    /* ================================================ FIG 6 — mapa dos 53 patches */
    (function () {
      const fig = root.querySelector('#fig-patches');
      adaptH(fig, 440, 650, 640);
      const NAMES = ['glm-exl3-prompt-experts', 'glm-dense-fp8', 'glm-vision', 'glm-prompt-kernels', 'glm-dense-q4', 'cuda-roce-allgather', 'glm-copy-drafts', 'glm-prompt-grid', 'glm-prefill-kernels', 'glm-hc-split', 'glm-draft-sim', 'glm-kda-chunked', 'glm-decode-rounds', 'glm-kda-chunked-gb10', 'glm-shared-prefix', 'glm-decode-kernels', 'glm-overlap-priority', 'glm-noise-policies', 'glm-decode-kernels2', 'glm-prompt-experts-order', 'glm-dflash-policy-env', 'glm-overlap-normal-priority', 'server-effort-max', 'glm-prompt-select-rows', 'glm-dflash2-ring', 'glm-multi-kda', 'glm-multi-dflash2', 'glm-lean-prompt-scratch', 'glm-multi-dsa', 'glm-multi-stream-engine', 'glm-decode-index-regs', 'glm-code-copy-drafts', 'glm-prefill-overlap2', 'cuda-nucleus-union', 'glm-multi-rounds', 'glm-tool-calls', 'cuda-tokenize', 'glm-kv-fp8', 'glm-kda-chunked-kernel', 'glm-parallel-deadlocks', 'glm-parallel-ring-base', 'glm-prompt-replay', 'glm-visible-pools', 'cuda-context-errors', 'cuda-metrics', 'glm-l2-prefetch', 'glm-exl3-decode-loads', 'glm-timing-tokens', 'glm-multi-prefill', 'glm-many-media', 'glm-tool-history-recovery', 'cuda-roce-startup', 'glm-whole-tool-calls'];
      const AREAS = [
        { k: 'pesos', t: 'Pesos densos', hue: 150, p: [2, 5], m: 'BF16 → FP8 (e4m3, escala por linha × 128 colunas) ou q4 afim em grupos de 64 com recorte por MSE; cabeça FP8, kv_b BF16.', knob: 'DENSE=q4', fx: 'q4 sobre fp8: prosa 38,9 → 44,4 · código 44,2 → 48,7 · prefill ~1.090 → ~1.260 tok/s', bits: 'com perda' },
        { k: 'kv', t: 'KV cache', hue: 205, p: [38], m: 'latente DSA e chaves agrupadas do indexador como linhas e4m3 + escala 2^k (528 B); chaves e gates do indexador em anéis por slot.', knob: 'KV=fp8', fx: 'a janela de 1M com 4 requisições cabe · GSM8K 98,0% · HumanEval 97,6%', bits: 'com perda' },
        { k: 'prompt', t: 'Prompt', hue: 265, p: [1, 4, 9, 20, 24, 28], m: 'experts EXL3 com as linhas na L2 (NFIRST), Yᵀ = WᵀXᵀ, ordem DRAM/tensor core, rotação 1×/linha, densa só antes de 2.051, seleção em blocos de 512, scratch enxuto.', knob: 'sempre ligados', fx: 'NFIRST: 2,3–2,5× a 4.096 linhas · “faster prefill, less memory at 1M”', bits: '= / ≠ prompt' },
        { k: 'dois', t: 'Prompt, dois Sparks', hue: 300, p: [10, 17, 22, 33], m: 'cola das hyper-connections dividida por linhas entre os ranks; trocas sobrepostas com a frente do bloco seguinte; default da prioridade do stream lateral passou para normal.', knob: 'SPLIT=1', fx: '50k: ~1.270 → ~1.730 tok/s (com 0009 e 0020) · 149k 92,8 → 90,7 s · decode −1,5%', bits: '=' },
        { k: 'kda', t: 'Prompt, KDA', hue: 330, p: [12, 14, 39], m: 'regra delta em forma WY/UT num kernel CUDA: 256 threads por cabeça, sub-chunks de 32, estado 128×128 em registradores, produtos correntes de decaimentos.', knob: 'KDA_CHUNKED=1', fx: '50k 29,3 → 26,4 s · 149k 91,1 → 84,5 s', bits: 'próximo, não os mesmos bits' },
        { k: 'reuso', t: 'Reuso de prompt', hue: 45, p: [8, 15, 42], m: 'estados numa grade de 64; prefixo compartilhado (fim do sistema e ponto de divergência); replay com a linha de logits guardada.', knob: 'SHARED_PREFIX=1', fx: '64k idêntico: 34 s → <0,07 s · system de 7,9k: 4,24 → 0,13 s', bits: '=' },
        { k: 'decode', t: 'Decode', hue: 175, p: [13, 16, 19, 31], m: 'janelas de até 16 linhas; tokens entregues uma rodada depois; tiles por forma; epílogos fundidos; hc_partial em CUDA com os bits do Triton; pontuação do indexador em registradores.', knob: 'COPY_MAX=15', fx: 'edição 84 → 119 tok/s (com COPY_MAX 15) · KDA de decode 4–36% mais rápido', bits: '=' },
        { k: 'mem', t: 'Decode, memória', hue: 20, p: [46, 47], m: 'L2 prefetch num stream lateral durante os all-gathers (sítios a, f, o); treliça EXL3 em cargas não coerentes de 16 B um passo à frente.', knob: 'TF_GLM_L2PF=1 · TF_GLM_EXL3_LOADS=nc', fx: 'com 512 KiB: prosa +2,7%, código +3,3% · 4 req: 74,8 → 76,6 e 100,0 → 102,7', bits: '=' },
        { k: 'idx', t: 'Decode, indexador', hue: 235, p: [43], m: 'pontuação e seleção limitadas aos pools visíveis de cada linha, dentro do bucket capturado no CUDA graph.', knob: 'sempre ligado', fx: '“the same tokens” (sem número de velocidade)', bits: '=' },
        { k: 'link', t: 'Link', hue: 190, p: [6, 52], m: 'all-gathers pequenos por RDMA one-shot (transporte do b12x); até o motor ficar pronto, cada gather passa antes por uma barreira NCCL.', knob: 'COMM=roce · TF_ROCE_MAX_KB=512 · TF_ROCE_WAIT_S=20', fx: '11 µs vs 45 µs por 16 KiB · decode +6% · código ×4 +1,4% · corrige #6', bits: '=' },
        { k: 'drafts', t: 'Rascunhos', hue: 280, p: [7, 18, 21, 25, 32], m: 'copy drafts no lugar do DFlash2 quando há cópia; cópias da resposta só após 16 tokens; parada ciente do ruído; política por variável; janela do DFlash2 junto dos prefixos.', knob: 'COPY · COPY_CODE · DRAFT_POLICY=fnc7:0.3', fx: 'cópia 80,3 → 84,4 · COPY_CODE 55,4 → 56,2 · fnc: prosa 47,2 → 50,6, código 51,7 → 55,5', bits: '=' },
        { k: 'tool', t: 'Rascunhos, ferramenta', hue: 255, p: [11], m: 'dump teacher-forced do DFlash2 e simulador offline em numpy: qualquer política é reproduzida exatamente, com um oráculo como teto.', knob: 'TF_GLM_DRAFT_DUMP (desligado)', fx: '“how the stop rules were tuned” · β = 2,0 ajustado em 9 dumps', bits: '= (dump = serial)' },
        { k: 'conc', t: 'Requisições concorrentes', hue: 120, p: [26, 27, 29, 30, 35, 40, 41, 48, 49], m: 'pool compartilhado; janela em lote de até 32 linhas; KDA, DSA e DFlash2 segmentados; paridades + replay; deadlock corrigido (transporte pelos bytes); tokens distintos na cronometragem; multi-prefill.', knob: 'PARALLEL=4 · MULTI_PREFILL=1 · TF_GLM_MULTI_LONE=0', fx: 'prosa 60,4 → 108,8 tok/s (1 → 4) · TTFT ×4 590 → 340 ms · 22/22 iguais ao solo', bits: '=' },
        { k: 'samp', t: 'Amostragem', hue: 0, p: [34], m: 'teste de cobertura do top-p com os candidatos dos dois ranks juntos: mesmo sorteio, menos leituras de shard inteiro.', knob: 'TENSORFOLD_NUCLEUS_UNION=1 (desligado)', fx: 'opt-in, sem número publicado', bits: '=' },
        { k: 'server', t: 'Servidor', hue: 95, p: [3, 23, 36, 37, 44, 45, 50, 51, 53], m: 'visão (50 imagens, 4 vídeos), tool calls inteiras e histórico robusto, /tokenize, reasoning_effort max, context_length_exceeded com param, /metrics.', knob: 'VISION · TENSORFOLD_GLM_*', fx: '50 imagens em 12,5 s · corrige #2 · cortadas nunca enviadas', bits: '= (texto)' },
      ];
      const areaOf = new Array(54).fill(0);
      AREAS.forEach((A, i) => A.p.forEach((n) => { areaOf[n] = i; }));
      const S = { area: 12 };
      const ctl = UI.controls(fig);
      const sel = UI.select(ctl, { label: 'Área', options: AREAS.map((A, i) => ({ v: String(i), label: A.t + ' (' + A.p.length + ')' })), value: '12', onChange: (v) => { S.area = +v; upd(); } });
      const rN = UI.readout(ctl, 'patches na área', '—'), rB = UI.readout(ctl, 'bits', '—');
      function upd() { const A = AREAS[S.area]; rN.set(String(A.p.length)); rB.set(A.bits, A.bits.startsWith('com perda') ? 'warn' : A.bits.startsWith('próximo') ? 'warn' : 'ok'); }
      upd();
      let tiles = [];
      V.fig(fig, (g, w, h, t, dt, St, H) => {
        const C = V.C();
        const wide = w >= 640;
        const sweep = Math.floor((t * 5) % 60);
        const cols = wide ? 9 : 8;
        const gw = wide ? w * 0.52 : w - 16;
        const tsz = Math.min(wide ? 44 : 40, (gw - (cols - 1) * 5) / cols);
        const gx = 8, gy = 40;
        // barra de aplicação
        const prog = Math.min(53, sweep);
        V.text(g, 'patch -p0 --forward · site-packages do TensorFold v0.6.0', gx, 13, { size: wide ? 11.5 : 10, font: 'mono', color: C.muted, maxW: gw });
        V.box(g, gx, 20, gw, 8, { r: 4, fill: V.alpha(C.muted, 0.12) });
        g.fillStyle = V.alpha(TF(), 0.8); V.rr(g, gx, 20, gw * prog / 53, 8, 4); g.fill();
        tiles = [];
        for (let n = 1; n <= 53; n++) {
          const i = n - 1, c = i % cols, r = Math.floor(i / cols);
          const x = gx + c * (tsz + 5), y = gy + r * (tsz + 5);
          const ai = areaOf[n], A = AREAS[ai], on = ai === S.area;
          const applied = n <= prog;
          g.fillStyle = hsl(A.hue, on ? 0.95 : applied ? 0.28 : 0.12); V.rr(g, x, y, tsz, tsz, 6); g.fill();
          if (on) { g.strokeStyle = hsl(A.hue, 1); g.lineWidth = 2; V.rr(g, x - 1, y - 1, tsz + 2, tsz + 2, 7); g.stroke(); }
          V.text(g, String(n).padStart(2, '0'), x + tsz / 2, y + tsz / 2 + 4, { size: Math.min(13, tsz * 0.36), font: 'mono', weight: 600, color: on ? onTile() : C.ink, align: 'center', alpha: on ? 1 : 0.75 });
          if (n === prog + 1 && n <= 53) { g.strokeStyle = TF(); g.lineWidth = 2.4; V.rr(g, x - 2, y - 2, tsz + 4, tsz + 4, 7); g.stroke(); }
          tiles.push([x, y, tsz, n]);
        }
        const rowsN = Math.ceil(53 / cols);
        const cur = Math.min(53, prog + 1);
        V.text(g, 'aplicando ' + String(cur).padStart(4, '0') + '-' + NAMES[cur - 1] + '.patch', gx, gy + rowsN * (tsz + 5) + 12, { size: wide ? 11 : 10, font: 'mono', color: TF(), maxW: gw });
        // clique
        if (H.pointer.clicked) { const p = H.pointer; tiles.forEach(([x, y, s, n]) => { if (p.x >= x && p.x <= x + s && p.y >= y && p.y <= y + s) { S.area = areaOf[n]; sel.el.querySelector('select').value = String(S.area); upd(); } }); }
        // painel
        const A = AREAS[S.area];
        const px = wide ? gx + gw + 18 : 10, py = wide ? 22 : gy + rowsN * (tsz + 5) + 32, pw = wide ? w - px - 10 : w - 20;
        g.fillStyle = hsl(A.hue, 0.1); V.rr(g, px, py, pw, h - py - 8, 10); g.fill();
        g.strokeStyle = hsl(A.hue, 0.8); g.lineWidth = 1.5; V.rr(g, px, py, pw, h - py - 8, 10); g.stroke();
        let yy = py + 22;
        V.text(g, A.t, px + 12, yy, { size: wide ? 15 : 13, weight: 700, color: hsl(A.hue, 1), maxW: pw - 24 });
        yy += 18;
        const list = A.p.map((n) => String(n).padStart(4, '0') + ' ' + NAMES[n - 1]);
        const two = list.length > 4, colW = two ? (pw - 24) / 2 : pw - 24, perCol = two ? Math.ceil(list.length / 2) : list.length;
        list.forEach((s, i) => { const c = two ? Math.floor(i / perCol) : 0, rr = two ? i % perCol : i; V.text(g, s, px + 12 + c * colW, yy + rr * 15, { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, maxW: colW - 6 }); });
        yy += perCol * 15 + 10;
        const fs = wide ? 12.5 : 10.5, lh = wide ? 17 : 14;
        yy += V.wrap(g, A.m, px + 12, yy, pw - 24, { size: fs, color: C.ink, lineH: lh, maxLines: 4 }) + 6;
        V.text(g, 'knob', px + 12, yy, { size: 10, color: C.muted, weight: 600 }); yy += 14;
        yy += V.wrap(g, A.knob, px + 12, yy, pw - 24, { size: fs - 0.5, font: 'mono', color: C.c, lineH: lh, maxLines: 2 }) + 6;
        V.text(g, 'efeito medido', px + 12, yy, { size: 10, color: C.muted, weight: 600 }); yy += 14;
        yy += V.wrap(g, A.fx, px + 12, yy, pw - 24, { size: fs, color: C.ink, lineH: lh, maxLines: 3 }) + 8;
        const bc = A.bits.startsWith('com perda') || A.bits.startsWith('próximo') ? TF() : C.a;
        const bwid = Math.min(pw - 24, tw(g, 'bits: ' + A.bits, 11, 'mono', 600) + 18);
        if (yy + 20 < h - 8) chip(g, px + 12, yy, bwid, 20, 'bits: ' + A.bits, { fill: V.alpha(bc, 0.16), stroke: bc, size: 11, weight: 600 });
      });
    })();

    /* ================================================ FIG 7 — uma rodada com ruído chaveado */
    (function () {
      const fig = root.querySelector('#fig-rodada');
      const VERIFY_MS = [29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0], BLOCK_MS = 3.08;
      const WORDS = ['o', 'motor', 'lê', 'pesos', 'uma', 'vez', 'por', 'rodada', 'e', 'confere', 'cada', 'linha', 'com', 'a', 'chave', 'do', 'alvo', 'sem', 'mudar', 'nada', 'na', 'resposta', 'final', 'agora', 'cópia', 'janela', 'ruído', 'token', 'bits', 'lote', 'estado', 'grade', 'pool', 'rank', 'soma'];
      const S = { src: 'dflash', pol: 'fnc', P: 0.3, seed: 4, t0: -4, replays: 0 };
      const stage7 = fig.querySelector('.la-fig-stage');
      const setH7 = () => { const nar = fig.clientWidth < 640; const want = (nar ? (S.src === 'copy' ? 500 : 720) : (S.src === 'copy' ? 410 : 460)) + 'px'; if (stage7.style.height !== want) stage7.style.height = want; };
      setH7();
      { const ro7 = new ResizeObserver(setH7); ro7.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro7.disconnect()); }
      const mix = (a, b, c) => { let h = (a ^ 0x9e3779b9) >>> 0; h = Math.imul(h ^ ((b + 0x7f4a7c15) >>> 0), 0x85ebca6b) >>> 0; h ^= h >>> 13; h = Math.imul(h ^ Math.imul(c + 1, 0x27d4eb2f), 0xc2b2ae35) >>> 0; h = (h ^ (h >>> 16)) >>> 0; return (h + 0.5) / 4294967296; };
      let R = [];
      function scenario() {
        const r = V.rng(1000 + S.seed * 97);
        R = [];
        const pos0 = 412;
        for (let j = 0; j < 7; j++) {
          const ids = []; while (ids.length < 5) { const v = r.int(WORDS.length); if (!ids.includes(v)) ids.push(v); }
          const lead = 7.5 + r() * 1.5 - j * 1.0;
          const lg = ids.map((_, k) => (k === 0 ? lead : 0) + r.normal() * 0.6);
          const score = lg.map((x) => x + r.normal() * (0.9 + 0.12 * j));
          const gg = ids.map((v) => -Math.log(-Math.log(mix(S.seed, pos0 + j, v))));
          const p = softmax(lg);
          const noisy = lg.map((x, k) => x + gg[k]);
          const x = noisy.indexOf(Math.max(...noisy));
          const dv = score.map((s, k) => s + 0.7 * gg[k]);
          const pick = dv.indexOf(Math.max(...dv));
          const cN = softmax(score.map((s, k) => (s + gg[k]) / 2.0))[pick];
          const cP = softmax(score)[pick];
          const rot = (j * 2 + S.seed) % 5, RT = (arr) => arr.map((_, i) => arr[(i - rot + 5) % 5]);
          R.push({ ids: RT(ids), lg: RT(lg), score: RT(score), gg: RT(gg), p: RT(p), noisy: RT(noisy), x: (x + rot) % 5, pick: (pick + rot) % 5, cN, cP, pos: pos0 + j });
        }
      }
      scenario();
      function depth(pol) { let prod = 1, k = 0; for (let j = 0; j < 7; j++) { prod *= pol === 'fnc' ? R[j].cN : R[j].cP; if (j === 0 || prod >= S.P) k = j + 1; else break; } return k; }
      function outcome(k) { let a = 0; for (let j = 0; j < k; j++) { if (R[j].pick === R[j].x) a++; else break; } return { a, emit: a + 1 }; }
      // copy scenario
      const PROMPT = ['Cite', ':', '“', 'o', 'rascunho', 'só', 'propõe', ';', 'quem', 'decide', 'é', 'a', 'chave', 'do', 'prompt', ',', 'sempre', '”'];
      const REPLY = ['Ok', ',', 'cito', ':', '“', 'o', 'rascunho', 'só', 'propõe', ';', 'quem'];
      const COPIES = PROMPT.slice(9), TGT = ['decide', 'é', 'a', 'chave', 'do', 'prompt', ',', '”'];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Rascunho', options: [{ v: 'dflash', label: 'DFlash2' }, { v: 'copy', label: 'cópia (prompt lookup)' }], value: 'dflash', onChange: (v) => { S.src = v; gD.show(v === 'dflash'); setH7(); S.t0 = H.t; upd(); } });
      const gD = UI.group(ctl);
      UI.seg(gD.el, { label: 'Política', options: [{ v: 'fnc', label: 'fnc · ciente do ruído' }, { v: 'fc', label: 'fc · sem ruído' }], value: 'fnc', onChange: (v) => { S.pol = v; S.t0 = H.t; upd(); } });
      nocase(UI.slider(gD.el, { label: 'limiar P', min: 0.1, max: 0.9, step: 0.05, value: 0.3, fmt: (v) => nf(v, 2), onInput: (v) => { S.P = v; upd(); } }));
      UI.button(ctl, '↺ reenviar (mesma chave)', () => { S.replays++; S.t0 = H.t; upd(); });
      UI.button(ctl, '🎲 outra semente', () => { S.seed++; S.replays = 0; scenario(); S.t0 = H.t; upd(); });
      const rK = UI.readout(ctl, 'linhas na janela', '—'), rA = UI.readout(ctl, 'aceitos', '—'), rE = UI.readout(ctl, 'tokens emitidos', '—'), rC = UI.readout(ctl, 'custo da janela (tabela antiga)', '—');
      let lastSig = '';
      function upd() {
        if (S.src === 'dflash') {
          const k = depth(S.pol), o = outcome(k);
          rK.set(String(k + 1)); rA.set(o.a + ' de ' + k, o.a === k ? 'ok' : null); rE.set(String(o.emit));
          const ms = VERIFY_MS[Math.min(7, k)] + BLOCK_MS;
          rC.set(nf(ms, 1) + ' ms para ' + (k + 1) + (k ? ' linhas' : ' linha'));
          lastSig = 'seed ' + S.seed + ' · ' + R.map((q) => WORDS[q.ids[q.x]]).slice(0, k + 1).join(' ');
        } else {
          rK.set('1 + ' + COPIES.length); rA.set('7 de ' + COPIES.length); rE.set('8'); rC.set('10 linhas → graph de 16 (COPY_CODE)');
        }
      }
      upd();
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const tau = t - S.t0;
        if (S.src === 'copy') { drawCopy(g, w, h, tau, C, wide); return; }
        const k = depth(S.pol), kOther = depth(S.pol === 'fnc' ? 'fc' : 'fnc'), o = outcome(k);
        const rowsW = wide ? w * 0.62 : w - 16, rx = 8;
        const top = wide ? 38 : 42, rh = wide ? (h - top - 34) / 7 : Math.min(54, (h * 0.62 - top) / 7);
        V.text(g, 'posição', rx, 16, { size: 10, color: C.muted });
        V.wrap(g, 'p do alvo (barras) · logit + ruído chaveado (traço âmbar) · ★ amostra chaveada · ▢ escolha do rascunho', rx + 46, 16, rowsW - 50, { size: wide ? 10.5 : 9.5, color: C.muted, lineH: 12, maxLines: 2 });
        const revealT = (j) => 0.25 + j * 0.32, verT = 0.25 + 7 * 0.32 + 0.4;
        R.forEach((q, j) => {
          const y = top + j * rh;
          const shownRow = tau > revealT(j);
          const inWin = j < k;
          const dim = !inWin ? 0.35 : 1;
          V.text(g, 'p+' + (j + 1), rx, y + rh / 2 + 4, { size: 10.5, font: 'mono', color: inWin ? C.ink : C.muted });
          const bx0 = rx + 40, bw = (rowsW - 48 - (wide ? 70 : 52)) / 5;
          const mxN = Math.max(...q.noisy), mnN = Math.min(...q.noisy);
          q.ids.forEach((v, c) => {
            const x = bx0 + c * bw, bh = (rh - 20) * q.p[c];
            g.globalAlpha = dim * (shownRow ? 1 : 0.25);
            g.fillStyle = V.alpha(C.a, 0.55); V.rr(g, x + 3, y + rh - 16 - bh, bw - 6, Math.max(1, bh), 3); g.fill();
            const nh = ((q.noisy[c] - mnN) / (mxN - mnN || 1)) * (rh - 20);
            V.line(g, x + bw * 0.25, y + rh - 16 - nh, x + bw * 0.75, y + rh - 16 - nh, { color: TF(), lw: 2.4 });
            if (c === q.pick && shownRow) { g.strokeStyle = C.b; g.lineWidth = 2; V.rr(g, x + 1, y + 1, bw - 2, rh - 4, 5); g.stroke(); }
            V.text(g, WORDS[v], x + bw / 2, y + rh - 3, { size: wide ? 10 : 8.5, font: 'mono', color: c === q.pick ? C.b : C.muted, align: 'center', maxW: bw - 2 });
            if (tau > verT && inWin && c === q.x) V.text(g, '★', x + bw / 2, y + rh - 18 - nh, { size: 13, color: TF(), align: 'center' });
            g.globalAlpha = 1;
          });
          // resultado da linha
          const mx = bx0 + 5 * bw + 10;
          if (tau > verT + 0.25 + j * 0.15) {
            if (!inWin) V.text(g, 'fora', mx, y + rh / 2 + 4, { size: 10, color: C.muted });
            else if (j < o.a) mark(g, mx + 8, y + rh / 2, true, 6);
            else if (j === o.a) { mark(g, mx + 8, y + rh / 2, false, 6); V.text(g, '→ ★', mx + 20, y + rh / 2 + 4, { size: 10.5, color: TF(), weight: 600 }); }
            else V.text(g, '—', mx + 4, y + rh / 2 + 4, { size: 11, color: C.muted });
          }
          if (j === k - 1 && tau > revealT(j)) { V.line(g, rx, y + rh + 1, rx + rowsW - 10, y + rh + 1, { color: C.b, lw: 2, dash: [6, 4] }); V.text(g, 'a política para aqui (k = ' + k + ')', rx + rowsW - 12, y + rh + 12, { size: 10, color: C.b, align: 'right' }); }
        });
        if (tau > verT) {
          const flash = Math.max(0, 1 - (tau - verT) / 0.6);
          if (flash > 0) { g.fillStyle = V.alpha(TF(), 0.12 * flash); g.fillRect(rx, top, rowsW, k * rh); }
        }
        // curva do produto
        const cx = wide ? rx + rowsW + 18 : 16, cy = wide ? top + 10 : top + 7 * rh + 30, cw = wide ? w - cx - 12 : w - 30, ch = wide ? h * 0.48 : Math.max(80, h - cy - 140);
        V.text(g, 'Π das confianças', cx, cy - 4, { size: 11, color: C.muted });
        V.line(g, cx, cy + ch, cx + cw, cy + ch, { color: C.muted, lw: 1 }); V.line(g, cx, cy + 4, cx, cy + ch, { color: C.muted, lw: 1 });
        const ys = (v) => cy + ch - v * (ch - 6), xs = (j) => cx + 8 + (j / 6) * (cw - 16);
        V.line(g, cx, ys(S.P), cx + cw, ys(S.P), { color: RED(), lw: 1.4, dash: [5, 4] });
        V.text(g, 'P = ' + nf(S.P, 2), cx + cw, ys(S.P) - 4, { size: 10.5, color: RED(), align: 'right', font: 'mono' });
        ['fnc', 'fc'].forEach((pol) => {
          const main = pol === S.pol;
          let prod = 1; g.beginPath();
          for (let j = 0; j < 7; j++) { prod *= pol === 'fnc' ? R[j].cN : R[j].cP; const x = xs(j), y = ys(prod); if (j === 0) g.moveTo(x, y); else g.lineTo(x, y); }
          g.strokeStyle = main ? (pol === 'fnc' ? TF() : C.b) : V.alpha(C.muted, 0.6); g.lineWidth = main ? 2.6 : 1.4; g.stroke();
          prod = 1;
          for (let j = 0; j < 7; j++) { prod *= pol === 'fnc' ? R[j].cN : R[j].cP; if (main && tau > revealT(j)) V.dot(g, xs(j), ys(prod), j < (pol === S.pol ? k : kOther) ? 4 : 2.5, pol === 'fnc' ? TF() : C.b); }
        });
        [1, 4, 7].forEach((d) => V.text(g, String(d), xs(d - 1), cy + ch + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'profundidade', cx + cw, cy + ch + 26, { size: 10, color: C.muted, align: 'right' });
        V.text(g, (S.pol === 'fnc' ? 'fnc' : 'fc') + ': k = ' + k + ' · outra política: k = ' + kOther, cx, cy + ch + 42, { size: 11, font: 'mono', color: C.ink, maxW: cw });
        // conclusão
        if (tau > verT + 0.9) {
          const msg = o.emit + ' tokens nesta rodada (' + o.a + ' aceitos + 1 do alvo). Os mesmos ★ saem com qualquer política: só muda onde a cadeia para.';
          V.wrap(g, msg, wide ? cx : 12, cy + ch + 64, wide ? cw : w - 24, { size: wide ? 11.5 : 10.5, color: C.ink, lineH: 15, maxLines: 4 });
          if (S.replays > 0) V.text(g, '↺ mesma chave → mesma rodada: ' + fnv(lastSig), wide ? cx : 12, h - 8, { size: 10.5, font: 'mono', color: C.a, maxW: wide ? cw : w - 24 });
        }
      });
      function drawCopy(g, w, h, tau, C, wide) {
        const pad = 10, per = wide ? 18 : 9, gap = 3;
        const cw = (w - 2 * pad - (per - 1) * gap) / per, ch = 24;
        const lay = (arr, y0, hi, col, extra) => {
          arr.forEach((s, i) => {
            const r = Math.floor(i / per), c = i % per, x = pad + c * (cw + gap), y = y0 + r * (ch + 6);
            const on = hi(i);
            chip(g, x, y, cw, ch, s, { fill: on ? V.alpha(col, 0.25) : V.alpha(C.muted, 0.08), stroke: on ? col : V.alpha(C.muted, 0.3), size: wide ? 11 : 10, min: 6 });
            if (extra) extra(i, x, y);
          });
          return y0 + Math.ceil(arr.length / per) * (ch + 6);
        };
        V.text(g, 'prompt', pad, 16, { size: 11, color: C.muted, weight: 600 });
        const pEnd = lay(PROMPT, 22, (i) => i >= 1 && i <= 8, TF());
        const ry = pEnd + 22;
        V.text(g, 'resposta até agora (o último é o token pendente)', pad, ry - 6, { size: 11, color: C.muted, weight: 600, maxW: w - 20 });
        const rEnd = lay(REPLY, ry, (i) => i >= REPLY.length - 8, C.b);
        V.text(g, 'os 8 últimos (MATCH = 8) já ocorreram no prompt (em âmbar): copiar o que veio depois', pad, rEnd + 8, { size: wide ? 11 : 10, color: C.b, maxW: w - 20 });
        if (tau > 0.6) {
          const u = Math.min(1, (tau - 0.6) / 0.8);
          const i9 = 9, x1 = pad + (i9 % per) * (cw + gap) + cw / 2, y1 = 22 + Math.floor(i9 / per) * (ch + 6) + ch;
          const x0 = pad + ((REPLY.length - 1) % per) * (cw + gap) + cw / 2, y0 = ry + Math.floor((REPLY.length - 1) / per) * (ch + 6);
          V.arrow(g, x0, y0, V.lerp(x0, x1, u), V.lerp(y0, y1, u), { color: TF(), lw: 2, curve: wide ? -24 : 18 });
        }
        const wy = rEnd + 36;
        if (tau > 1.6) {
          V.text(g, 'janela: pendente + ' + COPIES.length + ' cópias (até COPY_MAX = 15) · verificação com a chave', pad, wy, { size: 11, color: C.muted, weight: 600, maxW: w - 20 });
          const all = [REPLY[REPLY.length - 1]].concat(COPIES);
          lay(all, wy + 8, () => true, TF(), (i, x, y) => {
            if (tau > 1.6 + all.length * 0.08 + 0.6 && i > 0) {
              const ok = all[i] === TGT[i - 1];
              const first = !ok && all.slice(1, i).every((s2, q) => s2 === TGT[q]);
              if (ok && all.slice(1, i + 1).every((s2, q) => s2 === TGT[q])) mark(g, x + cw / 2, y + ch + 2, true, 4);
              else if (first) { mark(g, x + cw / 2, y + ch + 2, false, 4); V.text(g, '★ alvo: ' + TGT[i - 1], x + cw / 2, y + ch + 18, { size: 10, color: TF(), align: 'center', weight: 600 }); }
            }
          });
        }
        // custo: rodadas seriais vs uma rodada com cópias
        const cy0 = wy + 8 + Math.ceil(10 / per) * (ch + 6) + 34;
        const costOK = cy0 + 70 < h - 70;
        if (tau > 3.0 && costOK) {
          const fw = Math.min(46, (w - 2 * pad - 150) / 8 - 4);
          V.text(g, 'sem cópia: 8 tokens = 8 rodadas', pad, cy0, { size: 11, color: C.muted, weight: 600 });
          for (let k = 0; k < 8; k++) { const on = ((tau - 3) * 4) % 9 > k; chip(g, pad + k * (fw + 4), cy0 + 8, fw, 22, wide ? 'fwd' : '', { fill: V.alpha(C.b, on ? 0.35 : 0.08), stroke: V.alpha(C.b, 0.6) }); }
          V.text(g, 'com cópia: 8 tokens = 1 rodada (janela de 10 linhas)', pad, cy0 + 50, { size: 11, color: C.muted, weight: 600 });
          chip(g, pad, cy0 + 58, fw * 1.6, 22, wide ? 'fwd ×10' : '', { fill: V.alpha(TF(), 0.4), stroke: TF() });
          V.text(g, 'pela tabela antiga, verificar 8 linhas custa ~2,3× uma linha, não 8×', pad + fw * 1.6 + 10, cy0 + 73, { size: 10.5, color: C.ink, maxW: w - pad * 2 - fw * 1.6 - 12 });
        }
        if (tau > 3.6) {
          const msg = '7 cópias aceitas + 1 token do alvo (★) = 8 tokens numa rodada. As cópias custam quase zero (numpy), vêm antes dos rascunhos do DFlash2, e a janela de 10 linhas é acolchoada até o graph de 16. Em edição, COPY_MAX = 15 levou de 84 a 119 tok/s.';
          V.wrap(g, msg, pad, costOK ? cy0 + 104 : h - (wide ? 40 : 66), w - 2 * pad, { size: wide ? 12 : 10.5, color: C.ink, lineH: 15, maxLines: wide ? 4 : 6 });
        }
      }
    })();

    /* ================================================ FIG 8 — quatro streams, uma janela, um pool */
    (function () {
      const fig = root.querySelector('#fig-streams');
      const stage = fig.querySelector('.la-fig-stage');
      const MEAS = { 1: [60.4, 60.4, 170, 114.7], 2: [79.2, 40.4, 269, 147.6], 3: [89.5, 30.6, 330, 196.3], 4: [108.8, 27.9, 340, 227.9] };
      const S = { mode: 'lote', n: 4, mp: true, k: 4 };
      const setH = () => { const nar = fig.clientWidth < 640; const want = (S.mode === 'lote' ? (nar ? 450 : 400) : (nar ? 420 : 380)) + 'px'; if (stage.style.height !== want) stage.style.height = want; };
      let txtY = null;
      setH();
      const ro = new ResizeObserver(setH); ro.observe(fig); if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cena', options: [{ v: 'lote', label: 'rodada em lote' }, { v: 'chegada', label: 'chegada de 4 prompts' }], value: 'lote', onChange: (v) => { S.mode = v; gL.show(v === 'lote'); gC.show(v === 'chegada'); setH(); upd(); } });
      const gL = UI.group(ctl), gC = UI.group(ctl);
      UI.slider(gL.el, { label: 'requisições', min: 1, max: 4, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.n = v; upd(); } });
      UI.toggle(gC.el, { label: 'MULTI_PREFILL=1', value: true, onChange: (v) => { S.mp = v; upd(); } });
      UI.slider(gC.el, { label: 'prompts juntos', min: 1, max: 4, step: 1, value: 4, fmt: (v) => String(v), onInput: (v) => { S.k = v; upd(); } });
      gC.show(false);
      const r1 = UI.readout(ctl, 'prosa agregado', '—'), r2 = UI.readout(ctl, 'prosa por req.', '—'), r3 = UI.readout(ctl, 'TTFT', '—'), r4 = UI.readout(ctl, 'estruturado agregado', '—');
      function upd() {
        if (S.mode === 'lote') { const m = MEAS[S.n]; r1.set(nf(m[0], 1) + ' tok/s'); r2.set(nf(m[1], 1) + ' tok/s'); r3.set(m[2] + ' ms'); r4.set(nf(m[3], 1) + ' tok/s'); }
        else { r1.set(S.mp ? '108,8 tok/s' : '103,4 tok/s'); r2.set('—'); r3.set(S.mp ? '340 ms' : '590 ms', S.mp ? 'ok' : 'warn'); r4.set(S.mp ? '227,9 (×4)' : '196,3 (×4)'); }
      }
      upd();
      function rounds(seed, n) {
        const r = V.rng(seed);
        const w = []; for (let i = 0; i < n; i++) w.push(3 + r.int(14));
        const orig = w.slice(); const trimmed = new Array(n).fill(0);
        let sum = w.reduce((a, b) => a + b, 0);
        while (sum > 32) { let m = 0; for (let i = 1; i < n; i++) if (w[i] >= w[m]) m = i; w[m]--; trimmed[m]++; sum--; }
        const acc = w.map((x) => Math.min(x - 1, r.int(x)));
        return { w, orig, trimmed, acc, sum, osum: orig.reduce((a, b) => a + b, 0) };
      }
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const wide = w >= 640;
        const SC = [C.a, C.b, C.c, hsl(205, 1)];
        const px = 10, pw = w - 20;
        if (S.mode === 'lote') {
          const n = S.n;
          const py = 22, ph = 26;
          V.text(g, 'pool compartilhado (~2,92M tokens; extents alinhados a 2.048) · esquemático', px, 13, { size: wide ? 11 : 9.5, color: C.muted, maxW: pw });
          V.box(g, px, py, pw, ph, { r: 6, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          for (let k = 1; k < 48; k++) V.line(g, px + (k / 48) * pw, py + ph - 5, px + (k / 48) * pw, py + ph, { color: V.alpha(C.muted, 0.35), lw: 1 });
          const cyc = (t % 10) / 10;
          const base = [0.26, 0.2, 0.13, 0.09];
          const kept = [0.07, 0.05, 0.06];
          const evict = cyc > 0.55;
          let x = px + 1;
          for (let i = 0; i < n; i++) {
            const ww = (base[i] + 0.04 * Math.sin(t * 0.6 + i)) * (pw - 2);
            g.fillStyle = V.alpha(SC[i], 0.55); V.rr(g, x, py + 1, ww, ph - 2, 4); g.fill();
            if (ww > 44) V.text(g, 'stream ' + (i + 1), x + 6, py + 17, { size: 10, color: C.ink, maxW: ww - 8 });
            x += ww + 2;
            if (i === 0) {
              kept.forEach((f, q) => {
                let a = 1; if (q === 0 && evict) { a = Math.max(0, 1 - (cyc - 0.55) / 0.12); if (a <= 0) return; }
                const kw = f * (pw - 2);
                g.globalAlpha = a; g.fillStyle = V.alpha(C.muted, 0.35); V.rr(g, x, py + 1, kw, ph - 2, 4); g.fill();
                if (kw > 46) V.text(g, 'guardado', x + kw / 2, py + 17, { size: 9.5, color: C.ink, align: 'center' });
                g.globalAlpha = 1; x += kw + 2;
              });
            }
          }
          if (evict && cyc < 0.8) V.text(g, 'LRU: um prompt guardado cede espaço e os extents compactam', w - 12, py + ph + 13, { size: 10, color: C.d, align: 'right', maxW: pw });
          const iy = py + ph + 30;
          V.text(g, n > 1 ? 'iterações: chunk de prompt (≤ 1.024 linhas) ou rodada, alternando (FILL_SHARE = 0,5)' : 'iterações: só rodadas de decode', px, iy, { size: wide ? 11 : 9.5, color: C.muted, maxW: pw });
          const ni = wide ? 12 : 6, iw = (pw - (ni - 1) * 4) / ni;
          const shift = (t * 0.8) % 1;
          g.save(); g.beginPath(); g.rect(px, iy + 4, pw, 28); g.clip();
          for (let k = -1; k <= ni; k++) {
            const idx = k + Math.floor(t * 0.8);
            const isChunk = n > 1 && idx % 2 === 0 && (Math.floor(idx / 6) % 2 === 0);
            const xx = px + (k - shift) * (iw + 4);
            chip(g, xx, iy + 8, iw, 20, isChunk ? 'chunk' : 'rodada', { fill: V.alpha(isChunk ? C.d : C.a, 0.16), stroke: isChunk ? C.d : C.a, size: 10 });
          }
          g.restore();
          const R = rounds(Math.floor(t / 2.6) + 3, n), ph2 = (t % 2.6) / 2.6;
          const wy = iy + 50;
          const phaseA = ph2 < 0.35;
          const head = phaseA && R.osum > 32 ? 'propostas: ' + R.osum + ' linhas > 32 → corta um rascunho por vez da cauda mais longa' : 'janela de verificação: ' + R.sum + ' linhas × 16 KiB = ' + R.sum * 16 + ' KiB por rank → RoCE one-shot';
          V.wrap(g, head, px, wy, pw, { size: wide ? 11.5 : 10, color: phaseA && R.osum > 32 ? RED() : C.ink, weight: 600, lineH: 13, maxLines: 2 });
          const perRow = 16, cs = (pw - (perRow - 1) * 3) / perRow, chh = wide ? 24 : 20;
          const cellsL = [];
          for (let i = 0; i < n; i++) { const tot = phaseA ? R.orig[i] : R.w[i]; for (let q = 0; q < tot; q++) cellsL.push([i, q, q >= R.w[i]]); }
          const gy0 = wy + (wide ? 12 : 22);
          const rowsRes = Math.ceil(R.osum / perRow);
          cellsL.forEach(([i, q, cut], k) => {
            const xx = px + (k % perRow) * (cs + 3), y = gy0 + Math.floor(k / perRow) * (chh + 5);
            if (cut) { g.fillStyle = V.alpha(RED(), 0.22); V.rr(g, xx, y, cs, chh, 3); g.fill(); g.strokeStyle = RED(); g.lineWidth = 1.4; V.rr(g, xx, y, cs, chh, 3); g.stroke(); return; }
            const acc = q === 0 || q <= R.acc[i];
            const verified = ph2 > 0.55;
            g.fillStyle = V.alpha(SC[i], verified ? (acc ? 0.85 : 0.14) : 0.45); V.rr(g, xx, y, cs, chh, 3); g.fill();
            if (q === 0) { g.strokeStyle = C.ink; g.lineWidth = 1.2; V.rr(g, xx, y, cs, chh, 3); g.stroke(); }
            if (cs >= 18) V.text(g, q === 0 ? 's' + (i + 1) : 'd' + q, xx + cs / 2, y + chh / 2 + 4, { size: wide ? 10 : 9, font: 'mono', color: verified && acc && !C.dark ? '#fff' : C.ink, align: 'center', alpha: verified && !acc ? 0.5 : 1 });
          });
          const winH = rowsRes * (chh + 5);
          if (ph2 > 0.55) { const fl = Math.max(0, 1 - (ph2 - 0.55) / 0.2); g.fillStyle = V.alpha(TF(), 0.15 * fl); g.fillRect(px, gy0 - 3, pw, winH + 2); }
          // o texto acompanha as linhas que a janela ocupa nesta rodada (desliza suave entre rodadas)
          const target = gy0 + rowsRes * (chh + 5) + 10;
          txtY = txtY == null ? target : txtY + (target - txtY) * Math.min(1, (dt || 0.016) * 6);
          let yy = txtY;
          const lines = ['Uma forward para todos: partes independentes de linha no lote inteiro; KDA segmentado a partir do slot de cada stream; DSA no extent de cada um; um all-gather empacotado para os amostradores.', 'Apagadas = rascunhos rejeitados; o estado KDA é refeito só com as linhas mantidas (paridades + replay). s = token pendente, d = rascunho.'];
          lines.forEach((s2) => { yy += V.wrap(g, s2, px, yy, pw, { size: wide ? 11.5 : 10, color: C.muted, lineH: wide ? 15 : 13, maxLines: 5 }) + 4; });
          const m = MEAS[n];
          V.wrap(g, 'medido (sparkDash): ' + nf(m[0], 1) + ' tok/s agregados · ' + nf(m[1], 1) + ' por requisição · TTFT ' + m[2] + ' ms', px, Math.min(h - (wide ? 8 : 20), yy + 10), pw, { size: wide ? 12 : 10.5, color: C.ink, weight: 600, lineH: 13, maxLines: 2 });
        } else {
          const k = S.k;
          const FW = 0.12, RW = 0.05;
          const lanes = [{ nm: 'sem MULTI_PREFILL', on: !S.mp }, { nm: 'com MULTI_PREFILL', on: S.mp }];
          const scheds = [];
          {
            const ev = []; let tt = 0; const ttft = [];
            for (let i = 0; i < k; i++) { ev.push({ a: tt, b: tt + FW, t: 'P' + (i + 1) + '·a', c: i }); tt += FW; if (i > 0) { ev.push({ a: tt, b: tt + RW, t: 'r', c: -1 }); tt += RW; } ev.push({ a: tt, b: tt + FW, t: 'P' + (i + 1) + '·b', c: i }); tt += FW; ttft.push(tt); if (i < k - 1) { ev.push({ a: tt, b: tt + RW, t: 'r', c: -1 }); tt += RW; } }
            scheds.push({ ev, ttft, end: tt });
          }
          {
            const ev = []; const ttft = [];
            const lab = k > 1 ? '[s1..s' + k + ']' : '[s1]';
            ev.push({ a: 0, b: FW * 1.05, t: lab + '·a', c: 9 }); ev.push({ a: FW * 1.05, b: FW * 2.1, t: lab + '·b', c: 9 });
            for (let i = 0; i < k; i++) ttft.push(FW * 2.1);
            scheds.push({ ev, ttft, end: FW * 2.1 });
          }
          const tmax = Math.max(scheds[0].end, 0.3) * 1.08;
          const per = 4.5, phx = ((t % per) / per) * tmax * 1.15;
          const xs = (s2) => px + (s2 / tmax) * pw;
          V.wrap(g, k + (k > 1 ? ' prompts curtos chegam juntos' : ' prompt curto chega') + '; cada um = 2 chunks (até o ponto da grade, depois a cauda) · forward curta ≈ 0,12 s; r = rodada de decode', px, 14, pw, { size: wide ? 11 : 9.5, color: C.muted, lineH: 13, maxLines: 2 });
          const SCc = [C.a, C.b, C.c, hsl(205, 1)];
          lanes.forEach((L, li) => {
            const sc = scheds[li];
            const y = (wide ? 52 : 60) + li * (wide ? 130 : 160);
            V.text(g, L.nm, px, y, { size: 12, weight: 700, color: L.on ? TF() : C.muted });
            const by = y + 10, bh = 34;
            sc.ev.forEach((e) => {
              const x0 = xs(e.a), x1 = xs(e.b);
              const done = phx >= e.b, act = phx >= e.a && phx < e.b;
              const col = e.c < 0 ? C.muted : e.c === 9 ? TF() : SCc[e.c];
              g.globalAlpha = L.on ? 1 : 0.5;
              V.box(g, x0 + 1, by, Math.max(2, x1 - x0 - 2), bh, { r: 5, fill: V.alpha(col, done ? 0.55 : act ? 0.35 : 0.12), stroke: V.alpha(col, 0.8), lw: act ? 2 : 1 });
              const lab = x1 - x0 > 40 ? e.t : e.c < 0 ? 'r' : e.t.replace('·', '');
              if (x1 - x0 > 12) V.text(g, lab, (x0 + x1) / 2, by + bh / 2 + 4, { size: 9, font: 'mono', color: C.ink, align: 'center', maxW: x1 - x0 - 2 });
              g.globalAlpha = 1;
            });
            sc.ttft.forEach((tf, i) => {
              const x = xs(tf);
              const yy = by + bh + 6 + (li === 0 ? (i % 2) * 14 : 0);
              if (phx >= tf) { V.text(g, '⚑', x, yy + 10, { size: 12, color: li === 1 ? TF() : SCc[i], align: 'center' }); if (li === 0) V.text(g, 's' + (i + 1), x + 8, yy + 10, { size: 9.5, color: C.muted }); }
            });
            const last = sc.ttft[sc.ttft.length - 1];
            V.text(g, '1º token do último ≈ ' + nf(last, 2) + ' s (esquema)', px, by + bh + (li === 0 ? 62 : 34), { size: 10.5, font: 'mono', color: L.on ? C.ink : C.muted });
          });
          const lane2Bot = (wide ? 52 : 60) + (wide ? 130 : 160) + 10 + 34 + 40;
          V.line(g, xs(phx), 40, xs(phx), lane2Bot, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: [3, 3] });
          const ratio = scheds[0].ttft[k - 1] / scheds[1].ttft[k - 1];
          let ny = lane2Bot + 18;
          const sch = k > 1 ? 'Esquema: conta só as forwards curtas (~0,12 s) de prompts de ~130 tokens (2 chunks cada), por isso o último ⚑ chega ~' + nf(ratio, 1) + '× antes.' : 'Esquema: com um prompt só não há o que juntar, e as duas linhas do tempo empatam.';
          ny += V.wrap(g, sch, px, ny, pw, { size: wide ? 11.5 : 10, color: C.muted, lineH: 14, maxLines: 3 }) + 6;
          V.wrap(g, 'Medido (sparkDash, prosa ×4, TTFT médio dos 4 com rodadas e prompts reais): 590 → 340 ms, ~1,7×; agregado 103,4 → 108,8 tok/s. Cada pedaço recebe os bits de um chunk próprio.', px, ny, pw, { size: wide ? 11.5 : 10, color: C.ink, weight: 600, lineH: 14, maxLines: 4 });
        }
      });
    })();

    /* ================================================ FIG 9 — números com etiqueta */
    (function () {
      const fig = root.querySelector('#fig-numeros');
      adaptH(fig, 430, 600, 640);
      const S = { mode: 'decode', load: 'prosa', cond: false, t0: 0 };
      const DEC = { prosa: [[60.4, 60.4, 170], [79.2, 40.4, 269], [89.5, 30.6, 330], [108.8, 27.9, 340]], estruturado: [[114.7, 114.7, 149], [147.6, 77.1, 295], [196.3, 67.4, 317], [227.9, 61.1, 415]] };
      const PRE = [[8219, 1952.2, '4,21 s'], [16407, 1973.5, '8,31 s'], [32790, 1978.9, '16,57 s'], [65563, 1942.5, '33,75 s'], [131099, 1837.9, '71,33 s'], [262170, 1641.7, '159,69 s'], [981841, 1015, '967 s']];
      const VS = [
        { t: 'Prosa, 1 req. (tok/s)', a: 36.1, b: 60.4, up: true, c: 'vLLM: sparkDash 2026-09-17 (adaptive-k, MoE coop., 850k) · TF: sparkDash, clocks 2.200 MHz' },
        { t: 'Prosa, 4 req. agregado', a: 75.3, b: 108.8, up: true, c: 'mesmas corridas; por requisição 19,4 vs 27,9' },
        { t: 'TTFT, 1 req. (ms)', a: 333, b: 170, up: false, c: 'mesmas corridas (menor é melhor)' },
        { t: 'Estruturado, 1 req.', a: 83.0, b: 114.7, up: true, c: 'vLLM: README 2026-09-28, ferramenta de lab · TF: sparkDash' },
        { t: 'Prefill frio a 8k (tok/s)', a: 1411, b: 1952.2, up: true, c: 'vLLM: README 09-28 (lab) · TF: sparkDash, 8.219 tokens' },
        { t: 'Contexto (tokens)', a: 850000, b: 1048576, up: true, c: 'defaults das duas receitas' },
        { t: 'Pool de KV (tokens)', a: 1233779, b: 2922496, up: true, c: 'vLLM: 11 GiB fixos · TF: teto de 12,5 GiB, partida medida' },
        { t: 'KV por token e camada DSA (B)', a: 656, b: 528, up: false, c: 'formato: fp8_ds_mla vs e4m3 + escala 2^k' },
        { t: 'Boot (s)', a: 122, b: 240, bRange: [120, 360], up: false, c: 'vLLM: restart com InstantTensor · TF: 2–6 min de carga (vLLM vence)' },
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Painel', options: [{ v: 'decode', label: 'Decode' }, { v: 'prefill', label: 'Prefill' }, { v: 'reuso', label: 'Reuso' }, { v: 'vllm', label: 'vs vLLM' }], value: 'decode', onChange: (v) => { S.mode = v; S.t0 = H.t; gD.show(v === 'decode'); gV.show(v === 'vllm'); } });
      const gD = UI.group(ctl), gV = UI.group(ctl);
      UI.seg(gD.el, { label: 'Carga', options: [{ v: 'prosa', label: 'prosa' }, { v: 'estruturado', label: 'estruturado' }], value: 'prosa', onChange: (v) => { S.load = v; S.t0 = H.t; } });
      UI.toggle(gV.el, { label: 'condições', value: false, onChange: (v) => { S.cond = v; } });
      gV.show(false);
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const an = V.ease.out(Math.min(1, (t - S.t0) / 0.9));
        if (S.mode === 'decode') {
          const D = DEC[S.load];
          const L = 54, R = w - 14, T = 30, B = h - 56;
          const mx = S.load === 'prosa' ? 120 : 240;
          const ys = (v) => B - (v / mx) * (B - T);
          V.text(g, 'tok/s · ' + S.load + ' · sparkDash, clocks 2.200 MHz, temp 0', L - 40, 16, { size: 11, color: C.muted, maxW: w - 20 });
          for (let v = 0; v <= mx; v += mx / 4) { V.line(g, L, ys(v), R, ys(v), { color: C.grid, lw: 1 }); V.text(g, nf(v), L - 6, ys(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }); }
          const gw = (R - L) / 4;
          D.forEach((d, i) => {
            const x0 = L + i * gw + gw * 0.14, bw = gw * 0.34;
            const h1 = (B - ys(d[0])) * an, h2 = (B - ys(d[1])) * an;
            g.fillStyle = V.alpha(TF(), 0.8); V.rr(g, x0, B - h1, bw, h1, 4); g.fill();
            g.fillStyle = V.alpha(C.b, 0.65); V.rr(g, x0 + bw + 4, B - h2, bw, h2, 4); g.fill();
            V.text(g, nf(d[0], 1), x0 + bw / 2, B - h1 - 5, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, nf(d[1], 1), x0 + bw * 1.5 + 4, B - h2 - 5, { size: wide ? 11 : 9.5, font: 'mono', color: C.ink, align: 'center' });
            V.text(g, (i + 1) + ' req.', x0 + bw + 2, B + 15, { size: 11, color: C.ink, align: 'center' });
            V.text(g, 'TTFT ' + d[2] + ' ms', x0 + bw + 2, B + 30, { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, align: 'center' });
          });
          const lx = L, ly = h - 10;
          g.fillStyle = V.alpha(TF(), 0.8); g.fillRect(lx, ly - 9, 10, 10); V.text(g, 'agregado', lx + 14, ly, { size: 11, color: C.ink });
          g.fillStyle = V.alpha(C.b, 0.65); g.fillRect(lx + 90, ly - 9, 10, 10); V.text(g, 'por requisição', lx + 104, ly, { size: 11, color: C.ink });
          V.text(g, '×4 / ×1 = ' + nf(D[3][0] / D[0][0], 2) + '×', R, ly, { size: 11, font: 'mono', color: C.ink, align: 'right' });
        } else if (S.mode === 'prefill') {
          const L = 58, R = w - 20, T = 30, B = h - 44;
          const lx = (n) => L + ((Math.log10(n) - Math.log10(6000)) / (Math.log10(1.3e6) - Math.log10(6000))) * (R - L);
          const ys = (v) => B - (v / 2200) * (B - T);
          V.text(g, 'prefill (tok/s) contra o tamanho do prompt · eixo log', L - 50, 16, { size: 11, color: C.muted, maxW: w - 20 });
          [0, 500, 1000, 1500, 2000].forEach((v) => { V.line(g, L, ys(v), R, ys(v), { color: C.grid, lw: 1 }); V.text(g, nf(v), L - 6, ys(v) + 4, { size: 10, font: 'mono', color: C.muted, align: 'right' }); });
          [[8192, '8k'], [32768, '32k'], [131072, '128k'], [524288, '512k'], [1048576, '1M']].forEach(([n, l]) => V.text(g, l, lx(n), B + 15, { size: 10, font: 'mono', color: C.muted, align: 'center' }));
          const npts = Math.max(1, Math.floor(an * PRE.length + 0.001));
          g.beginPath(); PRE.slice(0, npts).forEach((p, i) => { const x = lx(p[0]), y = ys(p[1]); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }); g.strokeStyle = TF(); g.lineWidth = 2.6; g.stroke();
          PRE.slice(0, npts).forEach((p, i) => {
            const x = lx(p[0]), y = ys(p[1]);
            V.dot(g, x, y, 4.5, TF());
            const lab = nf(p[1], 0) + (wide || i % 2 === 0 || i === 6 ? ' · ' + p[2] : '');
            V.text(g, lab, x + (i === 6 ? -6 : 6), y + (i % 2 ? 16 : -8), { size: wide ? 10.5 : 9, font: 'mono', color: C.ink, align: i === 6 ? 'right' : 'left' });
          });
          if (npts === PRE.length) { const x = lx(981841), y = ys(1015); V.glow(g, x, y, 18, C.a, 0.6 + 0.4 * Math.sin(t * 3)); V.text(g, 'agulha achada a 981.841 tokens', x - 6, y + 30, { size: wide ? 11.5 : 10, color: C.a, align: 'right', weight: 600 }); }
          V.text(g, 'cai com o contexto porque a atenção cresce com o prompt; até 64k fica em ~1.950', L, h - 8, { size: wide ? 11 : 9.5, color: C.muted, maxW: w - L - 10 });
        } else if (S.mode === 'reuso') {
          const L = wide ? 230 : 12, R = w - 70, T = 40;
          const lx = (s) => L + ((Math.log10(s) - Math.log10(0.03)) / (Math.log10(60) - Math.log10(0.03))) * (R - L);
          const rows = [['prompt idêntico de 64k', 34, 0.07, '< 0,07 s'], ['conversa nova, mesmo system de 7,9k', 4.24, 0.13, '0,13 s']];
          V.text(g, 'segundos até começar a responder · eixo log', 12, 18, { size: 11, color: C.muted });
          [0.1, 1, 10].forEach((s) => { V.line(g, lx(s), T - 4, lx(s), wide ? T + 175 : T + 245, { color: C.grid, lw: 1 }); V.text(g, nf(s, s < 1 ? 1 : 0) + ' s', lx(s), T - 8, { size: 10, font: 'mono', color: C.muted, align: 'center' }); });
          rows.forEach((r, i) => {
            const y = T + 10 + i * (wide ? 90 : 130);
            const ly = wide ? y + 26 : y + 4;
            V.text(g, r[0], 12, ly, { size: 12, weight: 600, color: C.ink, maxW: wide ? L - 20 : w - 20 });
            const by = wide ? y : y + 14;
            const w1 = (lx(r[1]) - L) * an, w2 = (lx(Math.max(0.031, r[2])) - L) * an;
            g.fillStyle = V.alpha(C.muted, 0.5); V.rr(g, L, by + 4, Math.max(2, w1), 20, 4); g.fill();
            V.text(g, 'a frio: ' + nf(r[1], 2) + ' s', L + w1 + 6, by + 19, { size: 10.5, font: 'mono', color: C.ink });
            g.fillStyle = V.alpha(TF(), 0.85); V.rr(g, L, by + 30, Math.max(2, w2), 20, 4); g.fill();
            V.text(g, 'reusado: ' + r[3], L + w2 + 6, by + 45, { size: 10.5, font: 'mono', color: C.ink });
            const pulse = (t * 1.2 + i * 0.5) % 1;
            V.dot(g, L + w2 * pulse, by + 40, 3, C.a);
          });
          const ty = wide ? T + 200 : T + 272;
          V.wrap(g, 'Mecanismo: o estado do fim do prompt é guardado com a linha de logits da cabeça (replay), e o system prompt ganha um estado próprio (prefixo compartilhado). Com 3 conversas alternando (~4,5k, 6 turnos), 98,1% dos prompts saíram do cache com TF_GLM_MULTI_LONE=0, contra 0% com 1.', 12, ty, w - 24, { size: wide ? 11.5 : 10.5, color: C.muted, lineH: 15, maxLines: 4 });
        } else {
          const n = VS.length;
          const top = 18, rowH = (h - top - 26) / n;
          const L = wide ? 210 : 12, R = w - (wide ? 70 : 82);
          VS.forEach((r, i) => {
            const y = top + i * rowH;
            const yl = wide ? y + rowH / 2 : y + 12;
            const yd = wide ? y + rowH / 2 : y + rowH * 0.62;
            V.text(g, r.t, wide ? 12 : 12, yl + (wide ? 4 : 0), { size: wide ? 11.5 : 10.5, color: C.ink, weight: 600, maxW: wide ? L - 20 : w - 24 });
            const mx = Math.max(r.a, r.bRange ? r.bRange[1] : r.b) * 1.08;
            const xs = (v) => L + (v / mx) * (R - L);
            V.line(g, L, yd, R, yd, { color: C.grid, lw: 1 });
            const xa = xs(r.a), xb = xs(V.lerp(r.a, r.b, an));
            const better = r.up ? r.b > r.a : r.b < r.a;
            V.line(g, xa, yd, xb, yd, { color: V.alpha(better ? TF() : RED(), 0.6), lw: 3 });
            if (r.bRange) { g.fillStyle = V.alpha(TF(), 0.25); g.fillRect(xs(r.bRange[0]), yd - 5, xs(r.bRange[1]) - xs(r.bRange[0]), 10); }
            V.dot(g, xa, yd, 5.5, GLM()); V.dot(g, xb, yd, 5.5, TF());
            const ratio = r.up ? r.b / r.a : r.a / r.b;
            const rlab = r.bRange ? '2–6 min' : (better ? '' : '') + nf(ratio, 2) + '×';
            V.text(g, rlab, R + 8, yd + 4, { size: 11, font: 'mono', weight: 600, color: better ? C.a : RED() });
            if (S.cond) V.text(g, r.c, wide ? L : 12, yd + (wide ? 15 : 14), { size: wide ? 9.5 : 8.5, color: C.muted, maxW: wide ? R - L + 60 : w - 24 });
          });
          const ly = h - 8;
          V.dot(g, 16, ly - 4, 5, GLM()); V.text(g, 'vLLM (cap. 42)', 26, ly, { size: 11, color: C.ink });
          V.dot(g, 140, ly - 4, 5, TF()); V.text(g, 'TensorFold', 150, ly, { size: 11, color: C.ink });
          V.text(g, 'não é A/B controlado', w - 12, ly, { size: 11, color: C.d, align: 'right', weight: 600 });
        }
      });
    })();

    /* ================================================ FIG 10 — monte o seu .env */
    (function () {
      const fig = root.querySelector('#fig-ajuste');
      adaptH(fig, 420, 560, 640);
      const Q = [
        { k: 'scale', l: 'preciso de 4 Sparks ou de > 8 requisições' },
        { k: 'api', l: 'preciso de logprobs, n > 1 ou penalidades' },
        { k: 'com', l: 'uso comercial' },
        { k: 'fr', l: 'prompts curtos não ingleses' },
        { k: 'exact', l: 'exatidão contra o checkpoint' },
        { k: 'share', l: 'outra carga divide os Sparks' },
        { k: 'agents', l: 'muitos agentes alternando' },
        { k: 'text', l: 'só texto' },
        { k: 'disk', l: 'pouco disco no worker' },
      ];
      const S = {}; Q.forEach((q) => { S[q.k] = false; });
      let typedAt = 0, prevSig = '';
      const ctl = UI.controls(fig);
      Q.forEach((q) => UI.toggle(ctl, { label: q.l, value: false, onChange: (v) => { S[q.k] = v; upd(); } }));
      const rC = UI.readout(ctl, 'CONTEXT', '—'), rP = UI.readout(ctl, 'PARALLEL', '—'), rM = UI.readout(ctl, 'motor', '—');
      function derive() {
        const DRAFTER = S.com ? 'mtp' : 'dflash2';
        const DENSE = S.exact ? 'bf16' : S.fr ? 'fp8' : 'q4';
        const KV = S.exact ? 'bf16' : 'fp8';
        const VISION = S.text ? 0 : 1;
        let CONTEXT;
        if (KV !== 'bf16') CONTEXT = 1048576; else if (DRAFTER !== 'dflash2') CONTEXT = 524288; else if (VISION === 1 && DENSE !== 'q4') CONTEXT = 163840; else CONTEXT = 196608;
        const PARALLEL = DRAFTER === 'dflash2' ? 4 : 1;
        const lines = [];
        if (S.com) lines.push(['DRAFTER', 'mtp', 'DFlash2 é CC BY-NC-ND; mtp: PARALLEL=1, perde +5–10% de decode']);
        if (S.exact) { lines.push(['DENSE', 'bf16', 'serve o checkpoint como ele é']); lines.push(['KV', 'bf16', 'cache exato; janela menor']); lines.push(['KDA_CHUNKED', '0', 'kernel serial no prompt (bits do serial)']); }
        else if (S.fr) lines.push(['DENSE', 'fp8', '#18: q4 perde o fim de turno (12/48); fp8 custa ~10%']);
        if (S.share) lines.push(['MEMORY_RESERVE_GIB', '20', 'ilustrativo: suba a reserva; livre mín. ≈ reserva − ~10 GiB']);
        if (S.agents) { lines.push(['TF_GLM_CACHE_ENTRIES', '64', 'ilustrativo: ~91 MiB cada, medidos no 0097 (#17)']); lines.push(['TF_GLM_MULTI_LONE', '0', 'mantém o cache dos outros (98,1% vs 0%)']); }
        if (S.text) lines.push(['VISION', '0', 'devolve ~1,8 GiB do rank 0 ao cache']);
        if (S.disk) lines.push(['WORKER_WEIGHTS', 'nfs', 'worker sem a cópia de ~166 GiB; ~2,2 min']);
        return { DRAFTER, DENSE, KV, VISION, CONTEXT, PARALLEL, lines, vllm: S.scale || S.api };
      }
      let D = derive();
      function upd() { D = derive(); rC.set(nf(D.CONTEXT)); rP.set(String(D.PARALLEL)); rM.set(D.vllm ? 'receita vLLM' : 'TensorFold', D.vllm ? 'warn' : 'ok'); const sig = JSON.stringify(D.lines); if (sig !== prevSig) { prevSig = sig; typedAt = H ? H.t : 0; } }
      let H = null;
      upd();
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const ew = wide ? w * 0.6 : w - 16, ex = 8, ey = 8, eh = wide ? h - 16 : h * 0.6;
        V.box(g, ex, ey, ew, eh, { r: 10, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.45), lw: 1.2 });
        g.fillStyle = V.alpha(C.muted, 0.14); V.rr(g, ex, ey, ew, 24, 10); g.fill();
        [C.red || '#f55', C.d, C.a].forEach((c, i) => V.dot(g, ex + 14 + i * 14, ey + 12, 4.5, c));
        V.text(g, '.env (lido, nunca executado)', ex + 60, ey + 16, { size: 11, font: 'mono', color: C.muted, maxW: ew - 70 });
        const fs = wide ? 12 : 10.5, lh = wide ? 32 : 29;
        let y = ey + 44;
        if (!D.lines.length) { V.text(g, '# nada a mudar: os defaults servem agentes,', ex + 12, y, { size: fs, font: 'mono', color: C.a, maxW: ew - 20 }); V.text(g, '# 2 Sparks, 4 requisições, janela de 1M', ex + 12, y + 16, { size: fs, font: 'mono', color: C.a, maxW: ew - 20 }); y += 24; }
        const elapsed = t - typedAt;
        let charBudget = Math.floor(elapsed * 60);
        D.lines.forEach(([k, v, c], i) => {
          if (y > ey + eh - 10) return;
          const full = k + '=' + v;
          const nShow = Math.max(0, Math.min(full.length, charBudget)); charBudget -= full.length;
          const shown = full.slice(0, nShow);
          const kk = shown.slice(0, Math.min(shown.length, k.length)), rest = shown.slice(k.length);
          V.text(g, kk, ex + 12, y, { size: fs, font: 'mono', color: C.c, weight: 600 });
          const kw = tw(g, kk, fs, 'mono', 600);
          if (rest) V.text(g, rest, ex + 12 + kw, y, { size: fs, font: 'mono', color: C.a, weight: 600 });
          if (nShow < full.length && Math.floor(t * 3) % 2 === 0) { const cw2 = tw(g, shown, fs, 'mono', 600); g.fillStyle = C.ink; g.fillRect(ex + 13 + cw2, y - fs + 2, 2, fs); }
          if (nShow >= full.length) V.text(g, '# ' + c, ex + 12, y + (wide ? 15 : 13), { size: wide ? 10.5 : 9, font: 'mono', color: C.muted, maxW: ew - 20 });
          y += lh;
        });
        // o resto fica no default
        const DEF = [['DRAFTER', 'dflash2'], ['DENSE', 'q4'], ['KV', 'fp8'], ['KDA_CHUNKED', '1'], ['VISION', '1'], ['WORKER_WEIGHTS', 'copy'], ['MEMORY_RESERVE_GIB', '14.5'], ['KV_POOL_GIB', '12.5'], ['TF_GLM_CACHE_ENTRIES', '32'], ['COMM', 'roce'], ['SPLIT', '1'], ['MULTI_PREFILL', '1'], ['SHARED_PREFIX', '1'], ['DRAFT_POLICY', 'fnc7:0.3'], ['COPY', '1'], ['COPY_MAX', '15'], ['COPY_CODE', '1'], ['MAX_TOKENS', '32768']];
        const over = new Set(D.lines.map((l) => l[0]));
        const rest = DEF.filter((d) => !over.has(d[0]));
        const dy0 = y + 8;
        if (dy0 + 30 < ey + eh) {
          V.text(g, '# o resto fica no default do config.sh:', ex + 12, dy0, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted, maxW: ew - 20 });
          const cols = wide ? 2 : 2, cwid = (ew - 24) / cols, rl = wide ? 15 : 13;
          const maxRows = Math.max(0, Math.floor((ey + eh - dy0 - 14) / rl));
          rest.slice(0, maxRows * cols).forEach(([k, v], i) => {
            const c = Math.floor(i / maxRows), r = i % maxRows;
            V.text(g, k + '=' + v, ex + 12 + c * cwid, dy0 + 16 + r * rl, { size: wide ? 10.5 : 9, font: 'mono', color: V.alpha(C.muted, 1), alpha: 0.85, maxW: cwid - 6 });
          });
        }
        if (D.vllm) {
          g.fillStyle = V.alpha(C.bg || '#000', 0.7); V.rr(g, ex + 1, ey + 25, ew - 2, eh - 26, 9); g.fill();
          const pulse = 0.6 + 0.4 * Math.sin(t * 3);
          V.box(g, ex + 16, ey + eh / 2 - 46, ew - 32, 92, { r: 12, fill: V.alpha(GLM(), 0.14 + 0.06 * pulse), stroke: GLM(), lw: 2 });
          V.text(g, 'Use a receita vLLM (cap. 42)', ex + ew / 2, ey + eh / 2 - 18, { size: wide ? 15 : 12.5, weight: 700, color: C.ink, align: 'center', maxW: ew - 40 });
          const why = (S.scale ? 'a receita TensorFold serve 2 ou 3 Sparks (4 só num branch) e até 8 requisições' : '') + (S.scale && S.api ? '; ' : '') + (S.api ? 'sem logprobs, n > 1 nem penalidades' : '');
          V.wrap(g, why, ex + 28, ey + eh / 2 + 6, ew - 56, { size: wide ? 11.5 : 10, color: C.ink, lineH: 14, maxLines: 3, align: 'left' });
        }
        // painel de resultado
        const px = wide ? ex + ew + 16 : 10, py = wide ? 14 : ey + eh + 18, pw = wide ? w - px - 10 : w - 20;
        const rows = [
          ['DRAFTER', D.DRAFTER], ['DENSE', D.DENSE], ['KV', D.KV], ['VISION', String(D.VISION)], ['CONTEXT', nf(D.CONTEXT)], ['PARALLEL', String(D.PARALLEL)],
        ];
        const rlh = wide ? 22 : 18;
        V.text(g, 'resultado (lógica do config.sh)', px, py + 4, { size: 11, color: C.muted, weight: 600, maxW: pw });
        rows.forEach(([k, v], i) => {
          const yy = py + 24 + (wide ? i : Math.floor(i / 2)) * rlh;
          const xx = wide ? px : px + (i % 2) * (pw / 2);
          V.text(g, k, xx, yy, { size: wide ? 11.5 : 10.5, font: 'mono', color: C.c });
          V.text(g, v, xx + (wide ? 100 : 86), yy, { size: wide ? 11.5 : 10.5, font: 'mono', color: C.ink, weight: 600 });
        });
        const yb = py + 24 + (wide ? rows.length : 3) * rlh + 6;
        const exact = S.exact;
        mark(g, px + 7, yb, true, 5); V.text(g, 'rascunhado = serial, concorrente = solo', px + 18, yb + 4, { size: wide ? 11 : 10, color: C.ink, maxW: pw - 20 });
        mark(g, px + 7, yb + 20, exact, 5); V.text(g, exact ? 'igual ao checkpoint em BF16' : 'com perda vs. checkpoint: DENSE=' + D.DENSE + ' · KV=' + D.KV + ' · KDA_CHUNKED=1', px + 18, yb + 24, { size: wide ? 11 : 10, color: exact ? C.ink : C.d, maxW: pw - 20 });
      });
    })();
  },
});

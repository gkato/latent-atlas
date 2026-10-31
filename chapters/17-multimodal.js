LA.chapter({
  id: 'multimodal',
  part: 3,
  order: 17,
  title: 'Visão: como um LLM enxerga imagens',
  navTitle: 'Multimodal',
  subtitle: 'Patches, torre visual bidirecional, RoPE 2D, merger, placeholders, mRoPE e vídeo — e como cada etapa vira um limite de receita no vLLM: tokens por imagem, encoder cache, profiling, torre replicada e o roteamento MoE de imagens do DeepSeek.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['multimodal', 'ViT', 'patches', 'RoPE 2D', 'merger', 'mRoPE', 'vídeo', 'encoder cache', 'mm-encoder-tp-mode', 'bias_vl', 'TensorFold'],
  goals: [
    'Seguir uma imagem de ponta a ponta: <b>patches</b>, <b>torre visual</b> (ViT bidirecional com RoPE 2D), <b>merger/aligner</b> e <b>placeholders</b> no prompt',
    'Calcular quantos <b>tokens</b> uma imagem custa em cada modelo (lado do token = patch × merge) e por que isso dita os limites de uma receita',
    'Entender como o LLM recebe a posição dos tokens de imagem (layout N do DeepSeek, mRoPE do Qwen, NoPE do GLM) e como vídeo vira grupos temporais',
    'Configurar visão no vLLM: processador, encoder cache, <code>--limit-mm-per-prompt</code>, <code>--skip-mm-profiling</code>, <code>--mm-encoder-tp-mode</code> e os hotfixes do DeepSeek (#172, #175, #109)',
    'Comparar com a lane TensorFold do GLM: torre só no rank 0, workspace reservado na partida e um <b>orçamento de tokens por requisição</b> que deixa o pior caso constante',
  ],
  prereqs: ['tokens-embeddings', 'bloco-transformer', 'atencao', 'posicao-rope', 'prefill-decode-kv', 'mixture-of-experts', 'atencao-esparsa'],
  summary: [
    'Uma imagem vira uma grade de <b>patches</b> (14 px no DeepSeek e no GLM, 16 px no Qwen; GLM e Qwen com 2 quadros por patch), cada um projetado por \\(W_E\\), uma convolução de passo \\(p\\), num vetor de 1024 ou 1152 números.',
    'A <b>torre visual</b> é um Transformer <b>bidirecional</b> (24, 27 ou 32 blocos) com RoPE 2D: metade dos pares gira com a linha, metade com a coluna (no Qwen, somada a uma grade absoluta interpolada). Tem ~0,5 B de parâmetros e fica em BF16.',
    'O <b>merger/aligner</b> junta \\(k \\times k\\) vizinhos (2×2 no GLM e no Qwen, 3×3 no DeepSeek), concatena e projeta para a largura do LLM. O lado de um token, \\(p \\cdot k\\) = 28, 32 ou 42 px, é o câmbio entre área e contexto.',
    'O processador expande um <b>placeholder</b> em N cópias (154854, 248056, 129264), trocadas pelos vetores visuais no embedding. O DeepSeek usa o layout N e alinha o corpo a múltiplos de 4 (<code>compress_pad</code>): o bloco tem de 122 a 125 tokens conforme a posição.',
    'Posição no LLM: mRoPE (t, h, w) intercalado <code>[11,11,10]</code> no Qwen, fila 1D estruturada no DeepSeek, NoPE no GLM. Vídeo são grupos de 2 quadros: timestamps alinhados a <code>grid_t</code> no GLM, amostragem esparsa sob 16.384 tokens no Qwen, só o 1º quadro de um GIF no DeepSeek.',
    'No vLLM, <code>--limit-mm-per-prompt</code> só valida. Com <code>--skip-mm-profiling</code> nada é reservado, e por isso o GLM limita cada imagem a 2.048 tokens (≤ encoder cache de 7.168) e dimensiona o pior caso (48 × 2.048). O Qwen replica a torre (<code>data</code>) porque 4304/2 = 2152 não é múltiplo de 16. Na lane TensorFold do GLM, a torre fica só no rank 0, com workspace reservado, e as imagens de uma requisição dividiam 16.384 tokens até a v1.10 (327 cada com 50; desde 2026-10-10, 102.400).',
    'Placeholders têm id: no DeepSeek, as camadas de hash do MoE mandavam todo token de imagem para <code>tid2eid[129264]</code> (#175), corrigido com <code>bias_vl</code> sem hash e classificação 1× por forward; o hash do encoder cache passou a incluir o número de tokens (#172).',
  ],
  quiz: [
    { q: 'Uma imagem 1920×1080 no Qwen (patch 16, merge 2), sem cortes de orçamento, custa aproximadamente quantos tokens do LLM?', o: ['~500', '~2.000', '~8.000', '~32.000'], a: 1, e: 'Cada token cobre 16 · 2 = 32 px de lado: (1920/32) · (1080/32) = 60 × 33,75 ≈ 60 × 34 = 2.040 tokens. ~8.000 é o número de <i>patches</i> antes do merge 2×2 (120 × 67,5), ou o custo de um quadro 4K depois do merge (≈ 8,2 mil, medido pela receita).' },
    { q: 'Por que a torre visual usa atenção bidirecional, e não a máscara causal do LLM?', o: ['Porque é mais barata', 'Porque todos os patches existem ao mesmo tempo e a torre não gera nada: cada patch pode consultar a imagem inteira', 'Porque o RoPE 2D exige', 'Porque o vLLM não suporta máscara causal na visão'], a: 1, e: 'A máscara causal existe para gerar da esquerda para a direita. Numa imagem não há futuro a esconder, e mascarar só jogaria informação fora: o patch do canto veria apenas a si mesmo. O custo é o mesmo O(N²).' },
    { q: 'Na receita GLM, com <code>--skip-mm-profiling</code> ligado, o que <code>--limit-mm-per-prompt {"image":48}</code> garante?', o: ['Que 48 imagens têm memória reservada no boot', 'Só que requisições com mais de 48 imagens são recusadas; nada é reservado, então o teto precisa caber na folga real', 'Que o encoder cache tem 48 entradas', 'Que o KV cache encolhe 48 vezes'], a: 1, e: 'Sem o profiling multimodal, o limite vira validação. A medição mostra isso: subir de 4 para 16 imagens mudou o KV de 12,93 para 12,89 GiB. Com 48 × 2.048 ≈ 98 mil tokens de pior caso, a receita escolheu um teto que a UMA consegue codificar.' },
    { q: 'Por que o Qwen serve com <code>--mm-encoder-tp-mode data</code>?', o: ['Para acelerar o decode', 'Porque em TP=2 a MLP da visão (intermediária 4304) ficaria com 2152 colunas por GPU, que não é múltiplo de 16, e a carga crasha; replicar custa ~0,42 GiB a mais por GPU', 'Porque o NVFP4 não suporta visão', 'Porque o encoder precisa de 2 GPUs'], a: 1, e: 'Em <code>weights</code>, as matrizes da torre são fatiadas entre as GPUs, e 4304/2 = 2152 viola a divisibilidade exigida. Em <code>data</code>, cada GPU guarda a torre inteira (0,84 GiB em BF16, ~0,42 GiB a mais que as metades do modo <code>weights</code>) e processa imagens inteiras, sem all-reduce.' },
    { q: 'No DeepSeek Vision-Exp, por que o hash do encoder cache precisou ser “salgado” com o número de tokens (#172)?', o: ['Para economizar memória', 'Porque o bloco de imagem tem 122 + compress_pad tokens e o pad depende de start_pos mod 4: a mesma foto em outra posição tem outro comprimento, e o cache por bytes devolvia o layout errado', 'Porque o SHA-256 colidia', 'Porque GIFs têm vários quadros'], a: 1, e: 'A chave de um cache precisa incluir tudo de que o valor depende. O comprimento do bloco depende da posição, e por isso 125 placeholders receberam 124 embeddings e o EngineCore morreu. O sal <code>vlexp-ntok{N}</code> separa os casos, e <code>embed_input_ids</code> recusa contagens que não batem.' },
    { q: 'O que fazia os tokens de imagem colapsarem em 6 experts nas camadas 0–2 do DeepSeek (#175)?', o: ['O bias_vl estava mal treinado', 'As camadas de hash escolhem experts pelo id do token, e todas as linhas de imagem têm o mesmo id (129264)', 'O Aligner produzia vetores iguais', 'A captura de CUDA graphs'], a: 1, e: '<code>tid2eid[token_id]</code> só olha o id. Como o placeholder 129264 está no vocabulário, todo token de imagem ia para <code>tid2eid[129264]</code>. A correção roteia as linhas de imagem com <code>top6(s + bias_vl)</code>, sem tabela, e classifica o lote uma vez por forward para não pagar 43 sincronizações.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const clamp = V.clamp, lerp = V.lerp;
    const nf = (x) => Math.round(x).toLocaleString('pt-BR');
    let fc = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fc = {}; });
    const setFont = (g, size, f, w) => { const k = size + '|' + (f || 'sans') + '|' + (w || 400); g.font = fc[k] || (fc[k] = V.font(size, f || 'sans', w || 400)); };
    const tw = (g, s, size, f, w) => { setFont(g, size || 12, f, w); return g.measureText(String(s)).width; };
    const RED = (C) => C.red || C.c;
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11.5, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + (o.padX == null ? 14 : o.padX), ht = o.h || size + 9;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: o.r == null ? ht / 2 : o.r, fill: V.alpha(col, o.a == null ? 0.16 : o.a), stroke: V.alpha(col, o.sa == null ? 0.8 : o.sa), lw: o.lw || 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      const words = String(s).split(' ');
      let line = '', yy = y;
      setFont(g, o.size || 12, o.font, o.weight);
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        setFont(g, o.size || 12, o.font, o.weight);
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); line = wd; yy += lh; } else line = test;
      }
      if (line) V.text(g, line, x, yy, o);
      return yy + lh;
    }
    function autoH(fig, fn) {
      const st = fig.querySelector('.la-fig-stage');
      if (!st) return;
      const apply = () => { const w = st.clientWidth || fig.clientWidth || 800; const hh = Math.round(fn(w)); if (Math.abs((parseFloat(st.style.height) || 0) - hh) > 2) st.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply);
      ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    const rgbOf = (s) => {
      if (!s) return [128, 128, 128];
      s = String(s).trim();
      if (s[0] === '#') { let c = s.slice(1); if (c.length === 3) c = c.split('').map((x) => x + x).join(''); const n = parseInt(c.slice(0, 6), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
      const m = s.match(/[\d.]+/g);
      return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
    };
    // primeira string que cabe em maxW (evita o fillText comprimir o texto)
    const pick = (g, arr, size, f, wt, maxW) => { for (const s of arr) if (tw(g, s, size, f, wt) <= maxW) return s; return arr[arr.length - 1]; };
    const dec = (x, d) => x.toFixed(d).replace('.', ',');
    // altura do palco ajustada ao conteúdo: o draw informa o y final; o ResizeObserver passa a respeitar esse valor
    function fitH(fig, est) {
      const st = fig.querySelector('.la-fig-stage'), F = { need: 0 };
      autoH(fig, (w) => F.need || est(w));
      F.set = (need) => { const hh = Math.ceil(need); F.need = hh; if (st && Math.abs((parseFloat(st.style.height) || st.clientHeight) - hh) > 3) st.style.height = hh + 'px'; };
      return F;
    }
    const syncSlider = (sl, v) => { if (!sl) return; if (+sl.input.value !== v) { sl.input.value = v; const val = sl.el.querySelector('.val'); if (val) val.textContent = String(v); } };

    /* ------------------------------------------------ cena procedural (224×224) */
    const scene = (() => {
      const S = 224, cv = document.createElement('canvas'); cv.width = S; cv.height = S;
      const g = cv.getContext('2d');
      const poly = (pts, col) => { g.fillStyle = col; g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill(); };
      const circ = (x, y, r, col) => { g.fillStyle = col; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill(); };
      const sky = g.createLinearGradient(0, 0, 0, 150); sky.addColorStop(0, '#2f6fd0'); sky.addColorStop(1, '#b7ddff');
      g.fillStyle = sky; g.fillRect(0, 0, S, S);
      circ(182, 40, 26, 'rgba(255,210,63,0.28)'); circ(182, 40, 18, '#ffd23f');
      circ(50, 38, 10, '#ffffff'); circ(64, 32, 14, '#ffffff'); circ(80, 38, 10, '#ffffff'); circ(64, 42, 10, '#ffffff');
      poly([[0, 150], [62, 66], [124, 150]], '#56698c'); poly([[62, 66], [50, 84], [74, 84]], '#eef4ff');
      poly([[84, 150], [150, 78], [224, 150]], '#6f86ad'); poly([[150, 78], [138, 92], [162, 92]], '#eef4ff');
      g.fillStyle = '#3aa655'; g.fillRect(0, 148, S, S - 148);
      g.fillStyle = '#2f8f47'; for (let i = 0; i < 14; i++) g.fillRect(i * 17 + 5, 160 + (i % 3) * 16, 3, 6);
      g.fillStyle = '#d8483a'; g.fillRect(22, 128, 58, 46);
      poly([[16, 130], [51, 100], [86, 130]], '#7b2c20');
      g.fillStyle = '#5b3a1c'; g.fillRect(44, 150, 14, 24);
      g.fillStyle = '#fff2a8'; g.fillRect(28, 138, 12, 10); g.fillRect(63, 138, 12, 10);
      g.fillStyle = '#f08c2a'; g.beginPath(); g.ellipse(170, 198, 22, 15, 0, 0, Math.PI * 2); g.fill();
      circ(170, 170, 16, '#f39a3a');
      poly([[156, 162], [157, 142], [168, 155]], '#f39a3a'); poly([[184, 162], [183, 142], [172, 155]], '#f39a3a');
      poly([[159, 157], [159.5, 148], [165, 154]], '#ffc9a0'); poly([[181, 157], [180.5, 148], [175, 154]], '#ffc9a0');
      circ(164, 168, 2.4, '#1b1b1b'); circ(176, 168, 2.4, '#1b1b1b'); circ(170, 174, 1.8, '#c0392b');
      g.strokeStyle = '#3b2a1a'; g.lineWidth = 0.8;
      [-1, 0, 1].forEach((d) => { g.beginPath(); g.moveTo(160, 175 + d * 2); g.lineTo(147, 173 + d * 4); g.stroke(); g.beginPath(); g.moveTo(180, 175 + d * 2); g.lineTo(193, 173 + d * 4); g.stroke(); });
      circ(157, 211, 6, '#ffd2a6'); circ(183, 211, 6, '#ffd2a6');
      g.fillStyle = '#6b4a2a'; g.fillRect(112, 186, 3, 30);
      g.fillStyle = '#ffffff'; g.fillRect(92, 184, 44, 18);
      g.fillStyle = '#16213a'; g.font = '700 13px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('OLÁ', 114, 193.5);
      let data;
      try { data = g.getImageData(0, 0, S, S).data; } catch (e) { data = new Uint8ClampedArray(S * S * 4).fill(128); }
      return { cv, data, S };
    })();
    const pix = (x, y, c) => scene.data[(y * 224 + x) * 4 + c];
    const avgBlocks = (n) => {
      const b = 224 / n, out = [];
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        const s = [0, 0, 0];
        for (let y = r * b; y < (r + 1) * b; y += 2) for (let x = c * b; x < (c + 1) * b; x += 2) for (let k = 0; k < 3; k++) s[k] += pix(x, y, k);
        const cnt = (b / 2) * (b / 2);
        out.push(s.map((v) => Math.round(v / cnt)));
      }
      return out;
    };

    /* ================================================================ FIG 1 */
    (() => {
      const fig = root.querySelector('#fig-patches');
      if (!fig) return;
      autoH(fig, (w) => (w < 620 ? 740 : 424));
      const MODELS = {
        ds: { p: 14, T: 1, d: 1024, lab: 'DeepSeek' },
        glm: { p: 14, T: 2, d: 1024, lab: 'GLM' },
        qwen: { p: 16, T: 2, d: 1152, lab: 'Qwen' },
      };
      const S = { m: 'ds', grid: true, auto: true, sel: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'ds', label: 'DeepSeek · 14' }, { v: 'glm', label: 'GLM · 14×2' }, { v: 'qwen', label: 'Qwen · 16×2' }], value: 'ds', onChange: (v) => { S.m = v; upd(); } });
      UI.toggle(ctl, { label: 'grade', value: true, onChange: (v) => { S.grid = v; } });
      UI.toggle(ctl, { label: 'percorrer sozinho', value: true, onChange: (v) => { S.auto = v; } });
      const rP = UI.readout(ctl, 'patches (224²)', '—');
      const rN = UI.readout(ctl, 'números/patch', '—');
      const rW = UI.readout(ctl, 'W_E', '—');
      function upd() { const M = MODELS[S.m], n = 224 / M.p; rP.set(nf(n * n)); rN.set(nf(3 * M.p * M.p * M.T)); rW.set(`[${M.d} × ${3 * M.p * M.p * M.T}]`); }
      upd();
      const rng = V.rng(17);
      const PROJ = Array.from({ length: 32 }, () => Float32Array.from({ length: 768 }, () => (rng() < 0.5 ? -1 : 1)));
      const embC = {};
      function emb(p, idx) {
        const key = p + ':' + idx;
        if (embC[key]) return embC[key];
        const n = 224 / p, r0 = Math.floor(idx / n) * p, c0 = (idx % n) * p, f = new Float32Array(32);
        for (let c = 0; c < 3; c++) for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) {
          const v = pix(c0 + x, r0 + y, c) / 255 - 0.5, q = c * p * p + y * p + x;
          for (let k = 0; k < 32; k++) f[k] += v * PROJ[k][q];
        }
        let m = 1e-9; for (let k = 0; k < 32; k++) m = Math.max(m, Math.abs(f[k]));
        for (let k = 0; k < 32; k++) f[k] /= m;
        return (embC[key] = f);
      }
      const TOURPX = [[160, 150], [182, 40], [34, 143], [114, 193], [62, 78], [166, 168], [66, 32], [150, 86], [10, 205]];
      const permC = {};
      const perm = (n) => { if (permC[n]) return permC[n]; const p = 224 / n; return (permC[n] = TOURPX.map(([x, y]) => Math.floor(y / p) * n + Math.floor(x / p))); };
      const WE = V.randMatrix(6, 22, 3);
      const CH = ['255,70,70', '50,190,90', '70,130,255'];
      const H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), M = MODELS[S.m], p = M.p, n = 224 / p, N = n * n, mob = w < 620;
        let is, ix, iy, px, py, pw;
        if (!mob) { is = Math.min(h - 62, w * 0.4); ix = 22; iy = 36; px = ix + is + 44; pw = w - px - 22; py = 30; }
        else { is = Math.min(w - 40, 310); ix = (w - is) / 2; iy = 34; px = 20; pw = w - 40; py = iy + is + 30; }
        const P = H.pointer;
        const over = P.inside && P.x >= ix && P.x < ix + is && P.y >= iy && P.y < iy + is;
        const period = 2.8, tt = t + 1.6, cyc = Math.floor(tt / period);
        let u = 1;
        if (over) S.sel = clamp(Math.floor(((P.y - iy) / is) * n), 0, n - 1) * n + clamp(Math.floor(((P.x - ix) / is) * n), 0, n - 1);
        else if (S.auto) { const tour = perm(n); S.sel = tour[cyc % tour.length]; u = (tt % period) / period; }
        S.sel = Math.min(S.sel, N - 1);
        const pr = Math.floor(S.sel / n), pc = S.sel % n;
        const topW = mob ? is : is + 30;
        V.text(g, pick(g, [`imagem 224×224 · patch ${p} px → ${n}×${n} = ${N} patches`, `224×224 · patch ${p} px → ${n}×${n} = ${N} patches`, `patch ${p} px → ${n}×${n} = ${N} patches`], 12, 'mono', 400, topW), ix, iy - 12, { size: 12, font: 'mono', color: C.muted, maxW: topW });
        g.imageSmoothingEnabled = true;
        g.drawImage(scene.cv, ix, iy, is, is);
        g.strokeStyle = V.alpha(C.ink, 0.3); g.lineWidth = 1; g.strokeRect(ix + 0.5, iy + 0.5, is - 1, is - 1);
        const cs = is / n;
        if (S.grid) {
          g.beginPath();
          for (let k = 1; k < n; k++) { g.moveTo(ix + k * cs, iy); g.lineTo(ix + k * cs, iy + is); g.moveTo(ix, iy + k * cs); g.lineTo(ix + is, iy + k * cs); }
          g.strokeStyle = 'rgba(255,255,255,0.5)'; g.lineWidth = 0.8; g.stroke();
        }
        const sx = ix + pc * cs, sy = iy + pr * cs;
        V.glow(g, sx + cs / 2, sy + cs / 2, cs * 1.8, C.c, 0.55);
        g.strokeStyle = C.c; g.lineWidth = 2.5; g.strokeRect(sx, sy, cs, cs);
        // zoom
        const zs = mob ? 92 : Math.min(118, pw * 0.3);
        const zx = px + (M.T === 2 ? 12 : 0), zy = py + 20;
        const k1 = V.ease.out(clamp(u / 0.2, 0, 1));
        g.imageSmoothingEnabled = false;
        if (u < 0.2) {
          const gx = lerp(sx, zx, k1), gy = lerp(sy, zy, k1), gs = lerp(cs, zs, k1);
          g.globalAlpha = 0.9; g.drawImage(scene.cv, pc * p, pr * p, p, p, gx, gy, gs, gs); g.globalAlpha = 1;
        } else {
          if (M.T === 2) { g.globalAlpha = 0.45; g.drawImage(scene.cv, pc * p, pr * p, p, p, zx - 12, zy - 12, zs, zs); g.globalAlpha = 1; g.strokeStyle = V.alpha(C.c, 0.5); g.lineWidth = 1; g.strokeRect(zx - 12, zy - 12, zs, zs); }
          g.drawImage(scene.cv, pc * p, pr * p, p, p, zx, zy, zs, zs);
          g.strokeStyle = C.c; g.lineWidth = 2; g.strokeRect(zx, zy, zs, zs);
          g.beginPath();
          for (let k = 1; k < p; k++) { const q = zx + (k * zs) / p; g.moveTo(q, zy); g.lineTo(q, zy + zs); const q2 = zy + (k * zs) / p; g.moveTo(zx, q2); g.lineTo(zx + zs, q2); }
          g.strokeStyle = 'rgba(255,255,255,0.2)'; g.lineWidth = 0.6; g.stroke();
        }
        g.imageSmoothingEnabled = true;
        const tx = zx + zs + 18, tmax = w - tx - 16;
        V.text(g, pick(g, [`patch (linha ${pr}, coluna ${pc})`, `patch (l ${pr}, c ${pc})`], 13, 'sans', 600, tmax), tx, zy + 14, { size: 13, weight: 600, color: C.ink, maxW: tmax });
        V.text(g, pick(g, [`${p}×${p} px · 3 canais`, `${p}×${p} px · RGB`], 12, 'mono', 400, tmax), tx, zy + 34, { size: 12, font: 'mono', color: C.muted, maxW: tmax });
        V.text(g, M.T === 2 ? pick(g, ['× 2 quadros (cópia)', '× 2 quadros'], 12, 'mono', 400, tmax) : '1 quadro', tx, zy + 52, { size: 12, font: 'mono', color: M.T === 2 ? C.d : C.muted, maxW: tmax });
        V.text(g, pick(g, [`≡ conv kernel ${p}, passo ${p}`, `≡ conv ${p}×${p}, passo ${p}`, `≡ conv, passo ${p}`], 12, 'mono', 400, tmax), tx, zy + 70, { size: 12, font: 'mono', color: C.muted, maxW: tmax });
        // strip
        const by = zy + zs + 34, nv = p * p, cw = pw / nv, rh = mob ? 9 : 10;
        const rev = S.auto && !over ? clamp((u - 0.22) / 0.3, 0, 1) : 1;
        const shown = Math.floor(nv * rev);
        V.text(g, pick(g, [`vec(P) = 3 × ${p} × ${p}${M.T === 2 ? ' × 2' : ''} = ${nf(3 * p * p * M.T)} números`, `vec(P) = 3·${p}·${p}${M.T === 2 ? '·2' : ''} = ${nf(3 * p * p * M.T)} números`, `vec(P): ${nf(3 * p * p * M.T)} números`], 12, 'mono', 400, pw), px, by - 10, { size: 12, font: 'mono', color: C.ink, maxW: pw });
        V.arrow(g, px + 14, zy + zs + 4, px + 14, by - 24, { color: V.alpha(C.muted, 0.7), lw: 1.4, head: 6 });
        const drawRows = (y0, hh, alphaMul) => {
          for (let c = 0; c < 3; c++) {
            const yy = y0 + c * (hh + 2);
            for (let q = 0; q < nv; q++) {
              const yq = Math.floor(q / p), xq = q % p;
              const v = pix(pc * p + xq, pr * p + yq, c) / 255;
              g.fillStyle = `rgba(${CH[c]},${((0.12 + 0.88 * v) * alphaMul * (q < shown ? 1 : 0.16)).toFixed(3)})`;
              g.fillRect(px + q * cw, yy, Math.max(1, cw - (cw > 2.5 ? 0.4 : 0)), hh);
            }
            if (!mob && alphaMul === 1) V.text(g, 'RGB'[c], px - 8, yy + hh - 1, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          }
        };
        drawRows(by, rh, 1);
        let yEnd = by + 3 * (rh + 2);
        if (M.T === 2) { drawRows(yEnd + 4, 5, 0.55); yEnd += 4 + 3 * 7; V.text(g, '2º quadro (cópia)', px + pw, yEnd + 12, { size: 11, font: 'mono', color: C.d, align: 'right' }); }
        // matrix + vector
        // a matriz é larga o bastante para o rótulo de W_E terminar antes do vetor (sem sobreposição em ~630 px)
        const weLab = `W_E [${M.d} × ${3 * p * p * M.T}]`, weW = tw(g, weLab, 12, 'mono');
        const cy = yEnd + 38, mh = mob ? 58 : 66;
        const mw = Math.min(Math.max(Math.min(mob ? pw * 0.42 : pw * 0.38, 200), weW + 4), pw - 130);
        V.arrow(g, px + 14, yEnd + 4, px + 14, cy - 22, { color: V.alpha(C.muted, 0.7), lw: 1.4, head: 6 });
        V.text(g, weLab, px + 26, cy - 10, { size: 12, font: 'mono', color: C.ink, maxW: mw - 4 });
        const pulse = S.auto && !over && u > 0.48 && u < 0.7;
        if (pulse) V.glow(g, px + mw / 2, cy + mh / 2, mw * 0.7, C.b, 0.35);
        V.matrix(g, { x: px, y: cy, w: mw, h: mh, data: WE, cmap: 'div', gap: 1, r: 1 });
        V.text(g, '=', px + mw + 14, cy + mh / 2 + 7, { size: 22, weight: 700, color: C.ink, align: 'center' });
        const vx = px + mw + 30, vw = pw - (vx - px), vcw = vw / 16, vch = Math.min(22, mh / 2 - 3);
        const f = emb(p, S.sel);
        const fillN = S.auto && !over ? Math.floor(32 * clamp((u - 0.64) / 0.26, 0, 1)) : 32;
        for (let k = 0; k < 32; k++) {
          const r = Math.floor(k / 16), c = k % 16, x = vx + c * vcw, y = cy + r * (vch + 4) + (mh - 2 * vch - 4) / 2;
          g.fillStyle = k < fillN ? V.cmap(f[k], 'div') : V.alpha(C.muted, 0.12);
          V.rr(g, x + 1, y, vcw - 2, vch, 3); g.fill();
        }
        const xLab = `x_i ∈ ℝ^${M.d}`, subLab = `32 de ${nf(M.d)} células`, oneLine = `${xLab} · ${subLab}`;
        const fits1 = tw(g, oneLine, 12, 'mono') <= vw;
        V.text(g, fits1 ? oneLine : xLab, vx, cy - 10, { size: 12, font: 'mono', color: C.ink, maxW: vw });
        if (!fits1) V.text(g, subLab, vx, cy + mh + 14, { size: 11, font: 'mono', color: C.muted, maxW: vw });
        V.text(g, pick(g, [`${M.lab}: ${N} vetores como este, um por patch`, `${M.lab}: ${N} vetores, um por patch`], 12, 'sans', 400, pw), px, cy + mh + (fits1 ? 22 : 36), { size: 12, color: C.muted, maxW: pw });
      });
    })();

    /* ================================================================ FIG 2 */
    (() => {
      const fig = root.querySelector('#fig-bidir');
      if (!fig) return;
      autoH(fig, (w) => (w < 620 ? 720 : 420));
      const n = 8, N = 64;
      const cols = avgBlocks(n);
      const NAMES = { 46: 'orelha do gato', 14: 'sol', 33: 'telhado', 61: 'pata do gato', 52: 'placa “OLÁ”', 26: 'montanha', 0: 'céu (canto)' };
      const TOUR = [46, 14, 33, 61, 52, 26, 0];
      const S = { mode: 'bi', auto: true, q: 46 };
      function att(causal) {
        const A = new Float32Array(N * N);
        for (let i = 0; i < N; i++) {
          const ri = Math.floor(i / n), ci = i % n, s = new Float32Array(N);
          let mx = -1e9;
          for (let j = 0; j < N; j++) {
            if (causal && j > i) { s[j] = -1e9; continue; }
            const dc = ((cols[i][0] - cols[j][0]) ** 2 + (cols[i][1] - cols[j][1]) ** 2 + (cols[i][2] - cols[j][2]) ** 2) / (255 * 255);
            const rj = Math.floor(j / n), cj = j % n, d2 = (ri - rj) ** 2 + (ci - cj) ** 2;
            s[j] = -dc * 5 - d2 * 0.05;
            if (s[j] > mx) mx = s[j];
          }
          let sum = 0;
          for (let j = 0; j < N; j++) { const e = s[j] <= -1e8 ? 0 : Math.exp((s[j] - mx) * 1.8); A[i * N + j] = e; sum += e; }
          for (let j = 0; j < N; j++) A[i * N + j] /= sum;
        }
        return A;
      }
      const ATT = { bi: att(false), causal: att(true) };
      let matCv = {}, matDark = null;
      function matFor(mode) {
        const C = V.C();
        if (matDark !== C.dark) { matCv = {}; matDark = C.dark; }
        if (matCv[mode]) return matCv[mode];
        const A = ATT[mode], oc = document.createElement('canvas'); oc.width = N; oc.height = N;
        const og = oc.getContext('2d'), im = og.createImageData(N, N), bgc = rgbOf(C.panel || C.bg);
        for (let i = 0; i < N; i++) {
          let mx = 1e-12; for (let j = 0; j < N; j++) mx = Math.max(mx, A[i * N + j]);
          for (let j = 0; j < N; j++) {
            const o = (i * N + j) * 4;
            const c = mode === 'causal' && j > i ? bgc : rgbOf(V.cmap(Math.sqrt(A[i * N + j] / mx), 'heat'));
            im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
          }
        }
        og.putImageData(im, 0, 0);
        return (matCv[mode] = oc);
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Máscara', options: [{ v: 'bi', label: 'bidirecional (ViT)' }, { v: 'causal', label: 'causal (LLM)' }], value: 'bi', onChange: (v) => { S.mode = v; } });
      const autoT = UI.toggle(ctl, { label: 'percorrer patches', value: true, onChange: (v) => { S.auto = v; } });
      const rV = UI.readout(ctl, 'chaves visíveis', '64');
      const H = V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620;
        let G, gx, gy, M, mx, my;
        if (!mob) { G = Math.min(h - 76, w * 0.44); gx = 24; gy = 44; M = Math.min(h - 96, w - gx - G - 110); mx = w - M - 30; my = 60; }
        else { G = Math.min(w - 40, 300); gx = (w - G) / 2; gy = 44; M = Math.min(w - 90, 250); mx = (w - M) / 2 + 12; my = gy + G + 74; }
        const P = H.pointer, cs = G / n;
        if (P.clicked && P.x >= gx && P.x < gx + G && P.y >= gy && P.y < gy + G) { S.q = Math.floor((P.y - gy) / cs) * n + Math.floor((P.x - gx) / cs); if (S.auto) autoT.set(false); }
        if (S.auto) S.q = TOUR[Math.floor(t / 2.8) % TOUR.length];
        const q = S.q, causal = S.mode === 'causal', A = ATT[S.mode];
        const vis = causal ? q + 1 : N;
        rV.set(`${vis} de 64`);
        const center = (i) => [gx + (i % n) * cs + cs / 2, gy + Math.floor(i / n) * cs + cs / 2];
        V.text(g, `query: ${NAMES[q] || 'patch ' + q}`, gx, gy - 14, { size: 13, weight: 600, color: C.ink, maxW: G });
        for (let i = 0; i < N; i++) {
          const x = gx + (i % n) * cs, y = gy + Math.floor(i / n) * cs, col = cols[i];
          g.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; V.rr(g, x + 1, y + 1, cs - 2, cs - 2, 3); g.fill();
          if (causal && i > q) { g.fillStyle = V.alpha(C.bg, 0.8); V.rr(g, x + 1, y + 1, cs - 2, cs - 2, 3); g.fill(); }
        }
        const ws = [];
        for (let j = 0; j < N; j++) if (!(causal && j > q) && j !== q) ws.push([A[q * N + j], j]);
        ws.sort((a, b) => b[0] - a[0]);
        const top = ws.slice(0, 12), wmax = top.length ? top[0][0] : 1;
        const [qx, qy] = center(q);
        top.forEach(([wv, j], k) => {
          const [x2, y2] = center(j), a = wv / wmax;
          V.line(g, qx, qy, x2, y2, { color: V.alpha(C.a, 0.25 + 0.65 * a), lw: 1 + 4 * a });
          const f = (t * 0.7 + k * 0.137) % 1, pt = V.along(x2, y2, qx, qy, f);
          V.dot(g, pt[0], pt[1], 2 + 2 * a, C.a);
          g.strokeStyle = C.a; g.lineWidth = 1.6; g.beginPath(); g.arc(x2, y2, cs * 0.12 + cs * 0.2 * a, 0, Math.PI * 2); g.stroke();
        });
        V.glow(g, qx, qy, cs * 1.1, C.c, 0.6);
        g.strokeStyle = C.c; g.lineWidth = 3; V.rr(g, gx + (q % n) * cs + 1, gy + Math.floor(q / n) * cs + 1, cs - 2, cs - 2, 3); g.stroke();
        V.text(g, causal ? (q === 0 ? 'causal: o canto só enxerga a si mesmo' : `causal: vê só ${vis} patches (os anteriores)`) : 'bidirecional: vê os 64 patches', gx, gy + G + 20, { size: 12.5, font: 'mono', color: causal ? C.c : C.a, maxW: G + (mob ? 0 : 60) });
        // matriz
        g.imageSmoothingEnabled = false; g.drawImage(matFor(S.mode), mx, my, M, M); g.imageSmoothingEnabled = true;
        g.strokeStyle = V.alpha(C.ink, 0.3); g.lineWidth = 1; g.strokeRect(mx - 0.5, my - 0.5, M + 1, M + 1);
        const rh = M / N;
        g.strokeStyle = C.c; g.lineWidth = 2; g.strokeRect(mx - 3, my + q * rh - 1.5, M + 6, rh + 3);
        V.text(g, 'matriz de atenção 64 × 64', mx + M / 2, my - 26, { size: 12.5, weight: 600, align: 'center', color: C.ink });
        V.text(g, 'chaves →', mx + M, my - 8, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        g.save(); g.translate(mx - 12, my + M / 2); g.rotate(-Math.PI / 2);
        V.text(g, 'queries →', 0, 0, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        g.restore();
        if (causal) V.text(g, 'futuro mascarado', mx + M * 0.7, my + M * 0.26, { size: 12, align: 'center', color: C.muted, font: 'mono' });
      });
    })();

    /* ================================================================ FIG 3 */
    (() => {
      const fig = root.querySelector('#fig-rope2d');
      if (!fig) return;
      autoH(fig, (w) => (w < 620 ? 820 : 480));
      const NH = 10, NW = 14, TH = 10000;
      const om = Array.from({ length: 16 }, (_, k) => Math.pow(TH, (-2 * k) / 32));
      const SHOW = [0, 2, 4, 6, 8, 10, 12, 14];
      const REF = { h: 5, w: 7 };
      const KER = [];
      for (let dh = -9; dh <= 9; dh++) { const row = []; for (let dw = -13; dw <= 13; dw++) { let s = 0; for (let k = 0; k < 16; k++) s += Math.cos(dh * om[k]) + Math.cos(dw * om[k]); row.push(s / 32); } KER.push(row); }
      const S = { h: 2, w: 3, auto: true, aH: new Float32Array(16), aW: new Float32Array(16) };
      const ctl = UI.controls(fig);
      let aT = null;
      const stopAuto = () => { if (S.auto) { S.auto = false; if (aT) aT.el.querySelector('input').checked = false; } };
      const sH = UI.slider(ctl, { label: 'linha h', min: 0, max: NH - 1, step: 1, value: 2, onInput: (v) => { S.h = v; stopAuto(); } });
      const sW = UI.slider(ctl, { label: 'coluna w', min: 0, max: NW - 1, step: 1, value: 3, onInput: (v) => { S.w = v; stopAuto(); } });
      aT = UI.toggle(ctl, { label: 'passear sozinho', value: true, onChange: (v) => { S.auto = v; } });
      const rD = UI.readout(ctl, 'Δ até ◆', '—');
      let lastD = '';
      const H = V.fig(fig, (g, w, h, t, dt) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620;
        if (LA.paused) dt = 0;
        let gw, gh, gx, gy, rx, rw, dy1, dy2, hy;
        if (!mob) { gw = Math.min(w * 0.38, 310); gh = (gw * NH) / NW; gx = 22; gy = 48; rx = gx + gw + 40; rw = w - rx - 22; dy1 = 50; dy2 = 160; hy = 266; }
        else { gw = w - 40; gh = (gw * NH) / NW; gx = 20; gy = 42; rx = 20; rw = w - 40; dy1 = gy + gh + 50; dy2 = dy1 + 104; hy = dy2 + 100; }
        const P = H.pointer, cw = gw / NW, ch = gh / NH;
        if (P.inside && P.x >= gx && P.x < gx + gw && P.y >= gy && P.y < gy + gh) { S.h = clamp(Math.floor((P.y - gy) / ch), 0, NH - 1); S.w = clamp(Math.floor((P.x - gx) / cw), 0, NW - 1); }
        else if (S.auto) { S.h = Math.round(4.5 + 4.5 * Math.sin(t * 0.41)); S.w = Math.round(6.5 + 6.5 * Math.sin(t * 0.27 + 1.2)); }
        syncSlider(sH, S.h); syncSlider(sW, S.w);
        const dh = S.h - REF.h, dw = S.w - REF.w, dstr = `(${dh}, ${dw})`;
        if (dstr !== lastD) { rD.set(dstr); lastD = dstr; }
        V.text(g, `grade da torre: ${NH} × ${NW} patches`, gx, gy - 14, { size: 12.5, font: 'mono', color: C.muted });
        g.drawImage(scene.cv, 0, 32, 224, 160, gx, gy, gw, gh);
        g.beginPath();
        for (let k = 1; k < NW; k++) { g.moveTo(gx + k * cw, gy); g.lineTo(gx + k * cw, gy + gh); }
        for (let k = 1; k < NH; k++) { g.moveTo(gx, gy + k * ch); g.lineTo(gx + gw, gy + k * ch); }
        g.strokeStyle = 'rgba(255,255,255,0.45)'; g.lineWidth = 0.7; g.stroke();
        const rcx = gx + (REF.w + 0.5) * cw, rcy = gy + (REF.h + 0.5) * ch, scx = gx + (S.w + 0.5) * cw, scy = gy + (S.h + 0.5) * ch;
        V.line(g, rcx, rcy, scx, scy, { color: V.alpha(C.d, 0.9), lw: 1.6, dash: [4, 4], dashOffset: -t * 20 });
        g.fillStyle = C.d; g.beginPath(); g.moveTo(rcx, rcy - 7); g.lineTo(rcx + 7, rcy); g.lineTo(rcx, rcy + 7); g.lineTo(rcx - 7, rcy); g.closePath(); g.fill();
        g.strokeStyle = '#ffffff'; g.lineWidth = 1; g.stroke();
        V.glow(g, scx, scy, cw * 1.4, C.c, 0.6);
        g.strokeStyle = C.c; g.lineWidth = 2.5; g.strokeRect(gx + S.w * cw + 1, gy + S.h * ch + 1, cw - 2, ch - 2);
        V.text(g, '● patch  ◆ referência', gx, gy + gh + 18, { size: 12, font: 'mono', color: C.muted });
        const dialRow = (y, label, pos, arr, col) => {
          V.text(g, pick(g, label, 12.5, 'sans', 600, rw), rx, y - 12, { size: 12.5, weight: 600, color: C.ink, maxW: rw });
          const sp = rw / SHOW.length, R = Math.min(sp * 0.36, 21);
          SHOW.forEach((k, i) => {
            const target = pos * om[k];
            arr[k] += (target - arr[k]) * Math.min(1, dt * 6);
            const cx = rx + sp * (i + 0.5), cy = y + R + 4;
            g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fillStyle = V.alpha(col, 0.1); g.fill(); g.strokeStyle = V.alpha(col, 0.6); g.lineWidth = 1.5; g.stroke();
            V.line(g, cx, cy - R, cx, cy - R + 4, { color: V.alpha(col, 0.6), lw: 1.5 });
            const a = arr[k] - Math.PI / 2;
            V.line(g, cx, cy, cx + Math.cos(a) * R * 0.86, cy + Math.sin(a) * R * 0.86, { color: col, lw: 2.5 });
            V.dot(g, cx, cy, 2.5, col);
            V.text(g, 'ω' + k, cx, cy + R + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          });
          V.text(g, 'rápido', rx, y + 2 * R + 34, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'lento', rx + rw, y + 2 * R + 34, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        };
        dialRow(dy1, [`linha h = ${S.h} → pares 0–15 giram h·ω`, `h = ${S.h} → pares 0–15 giram h·ω`, `h = ${S.h} → pares 0–15`], S.h, S.aH, C.a);
        dialRow(dy2, [`coluna w = ${S.w} → pares 16–31 giram w·ω`, `w = ${S.w} → pares 16–31 giram w·ω`, `w = ${S.w} → pares 16–31`], S.w, S.aW, C.b);
        const cs = Math.max(4, Math.min(rw / 27, (h - hy - 46) / 19));
        const mx0 = rx + (rw - cs * 27) / 2;
        const simV = dec(KER[dh + 9][dw + 13], 2);
        V.text(g, pick(g, [`similaridade × deslocamento · agora (Δh, Δw) = ${dstr} → ${simV}`, `similaridade · agora (Δh, Δw) = ${dstr} → ${simV}`, `sim. (Δh, Δw) = ${dstr} → ${simV}`, `sim.(Δh,Δw)=(${dh},${dw}) → ${simV}`], 12, 'mono', 400, rw), rx, hy - 10, { size: 12, font: 'mono', color: C.ink, maxW: rw });
        V.matrix(g, { x: mx0, y: hy, cell: cs - 1, gap: 1, r: 1, data: KER, cmap: 'seq', hiCells: [[dh + 9, dw + 13]], hiColor: C.c });
        V.line(g, mx0 + 13 * cs + (cs - 1) / 2, hy - 3, mx0 + 13 * cs + (cs - 1) / 2, hy + 19 * cs + 1, { color: V.alpha(C.ink, 0.25), lw: 1, dash: [2, 3] });
        V.line(g, mx0 - 3, hy + 9 * cs + (cs - 1) / 2, mx0 + 27 * cs + 1, hy + 9 * cs + (cs - 1) / 2, { color: V.alpha(C.ink, 0.25), lw: 1, dash: [2, 3] });
        V.text(g, 'Δw: −13 … +13 →', mx0 + 27 * cs, hy + 19 * cs + 14, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        V.text(g, 'Δh ↓', mx0, hy + 19 * cs + 14, { size: 11, font: 'mono', color: C.muted });
      });
    })();

    /* ================================================================ FIG 4 */
    (() => {
      const fig = root.querySelector('#fig-merger');
      if (!fig) return;
      // mobile: altura do passo mais alto (passo 4: três caixas + vetor de saída), sem sobra nos outros
      autoH(fig, (w) => (w < 620 ? 38 + Math.min(w - 60, 250) + 34 + 272 : 430));
      const MM = {
        glm: { k: 2, d: 1024, dl: 4096, cat: '4 × 1024 = 4096', w1: 'downsample 2×2: 4096 → 4096', act: 'SwiGLU → 10240 (clamp 10)', w2: '10240 → 4096', px: '28×28 px (× 2 quadros)', col: 'glm', lab: 'GLM' },
        qwen: { k: 2, d: 1152, dl: 2560, cat: '4 × 1152 = 4608', pre: true, w1: 'fc1: 4608 → 4608', act: 'GELU', w2: 'fc2: 4608 → 2560', px: '32×32 px', col: 'qwen', lab: 'Qwen' },
        ds: { k: 3, d: 1024, dl: 4096, cat: '9 × 1024 = 9216', w1: 'Linear: 9216 → 4096', act: 'GELU', w2: 'Linear: 4096 → 4096', px: '42×42 px', col: 'ds', lab: 'DeepSeek' },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM · 2×2' }, { v: 'qwen', label: 'Qwen · 2×2' }, { v: 'ds', label: 'DeepSeek · 3×3' }], value: 'glm', onChange: (v) => { S.m = v; } });
      const st = UI.stepper(fig, [
        { t: 'Saída da torre', d: 'Cada quadradinho é o vetor de um patch depois dos blocos do ViT, com \\(d\\) = 1024 (GLM, DeepSeek) ou 1152 (Qwen) números. As faixas coloridas representam esses números.' },
        { t: 'Agrupar vizinhos', d: 'O merger recorta a grade em blocos de \\(k \\times k\\) patches vizinhos: 2×2 no GLM e no Qwen (<code>spatial_merge_size</code>), 3×3 no DeepSeek (<code>vision_downsample_ratio</code>). Cada bloco vai virar um único token do LLM.' },
        { t: 'Concatenar (space-to-depth)', d: 'Os \\(k^2\\) vetores de um bloco são colados lado a lado: 4 × 1024 = 4096, 4 × 1152 = 4608 ou 9 × 1024 = 9216 números. Nenhuma informação se perde; ela só muda de eixo, da grade para a profundidade. No Qwen, cada vetor passa <i>antes</i> por uma LayerNorm de 1152 números (<code>merger.norm [1152]</code>), patch a patch.' },
        { t: 'Projetar para o LLM', d: 'Uma MLP pequena leva o vetor concatenado à largura do LLM (\\(W_1\\), ativação, \\(W_2\\)). O resultado mora no mesmo espaço dos embeddings de texto: 4096 no GLM e no DeepSeek, 2560 no Qwen.' },
        { t: 'Tokens prontos', d: 'A grade encolheu \\(k\\) vezes em cada eixo: 36 patches viraram 9 tokens (÷4) ou 4 tokens (÷9). Cada token resume 28×28, 32×32 ou 42×42 pixels da imagem original.' },
      ], { autoplay: 6500 });
      const vecs = Array.from({ length: 36 }, (_, i) => { const r = V.rng(100 + i); return [0, 1, 2, 3].map(() => r.normal() * 0.7); });
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), M = MM[S.m], k = M.k, G = 6 / k, NG = G * G, mob = w < 620, colR = C[M.col] || C.a;
        const i = st.i, e = st.elapsed();
        let gs, gx, gy, px, py, pw;
        if (!mob) { gs = Math.min(h - 72, w * 0.4, 330); gx = 26; gy = (h - gs) / 2 + 8; px = gx + gs + 46; pw = w - px - 22; py = 34; }
        else { gs = Math.min(w - 60, 250); gx = (w - gs) / 2; gy = 38; px = 20; pw = w - 40; py = gy + gs + 34; }
        const cs = gs / 6;
        const gcol = (gi) => C.series[gi % C.series.length];
        const gOf = (r, c) => Math.floor(r / k) * G + Math.floor(c / k);
        const selG = i === 2 || i === 3 ? Math.floor(e / 2.2) % NG : -1;
        V.text(g, i < 4 ? `saída da torre · 36 vetores de ${nf(M.d)}` : `${NG} tokens do LLM (${M.lab})`, gx, gy - 14, { size: 12.5, font: 'mono', color: C.muted, maxW: gs });
        if (i < 4) {
          for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
            const idx = r * 6 + c, pop = i === 0 ? V.ease.back(clamp(e * 3 - idx * 0.05, 0, 1)) : 1;
            if (pop <= 0) continue;
            const x = gx + c * cs, y = gy + r * cs, m = 4, bw = (cs - 2 * m) * pop, bh = (cs - 2 * m) * pop, cx = x + cs / 2, cy = y + cs / 2;
            const gi = gOf(r, c), dim = selG >= 0 && gi !== selG;
            g.globalAlpha = dim ? 0.3 : 1;
            V.box(g, cx - bw / 2, cy - bh / 2, bw, bh, { r: 5, fill: V.alpha(C.b, 0.08), stroke: V.alpha(i >= 1 ? gcol(gi) : C.b, 0.75), lw: 1.2 });
            const ih = (bh - 8) / 4;
            if (ih > 1) for (let s = 0; s < 4; s++) { g.fillStyle = V.cmap(vecs[idx][s], 'div'); g.fillRect(cx - bw / 2 + 4, cy - bh / 2 + 4 + s * ih, bw - 8, ih - 1); }
            g.globalAlpha = 1;
          }
          if (i >= 1) for (let gi = 0; gi < NG; gi++) {
            const gr = Math.floor(gi / G), gc = gi % G, ap = i === 1 ? clamp(e * 2.5 - gi * 0.15, 0, 1) : 1;
            if (ap <= 0) continue;
            g.globalAlpha = ap; g.strokeStyle = gcol(gi); g.lineWidth = gi === selG ? 3.5 : 2;
            V.rr(g, gx + gc * k * cs + 1, gy + gr * k * cs + 1, k * cs - 2, k * cs - 2, 8); g.stroke(); g.globalAlpha = 1;
          }
        } else {
          const mt = V.ease.out(clamp(e / 0.9, 0, 1));
          for (let gi = 0; gi < NG; gi++) {
            const gr = Math.floor(gi / G), gc = gi % G, x = gx + gc * k * cs, y = gy + gr * k * cs, sz = k * cs, ins = lerp(1, 6, mt);
            V.box(g, x + ins, y + ins, sz - 2 * ins, sz - 2 * ins, { r: 10, fill: V.alpha(colR, 0.12 + 0.16 * mt), stroke: colR, lw: 2 });
            const rr = V.rng(900 + gi), bh2 = (sz * 0.5) / 6;
            for (let s = 0; s < 6; s++) { g.fillStyle = V.alpha(colR, 0.25 + 0.6 * Math.abs(Math.sin(rr() * 6 + t * 0.8 + s))); g.fillRect(x + sz * 0.22, y + sz * 0.18 + s * bh2 * 1.15, sz * 0.56, bh2 * 0.8); }
            V.text(g, 'token', x + sz / 2, y + sz * 0.88, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          }
        }
        // painel da direita / de baixo
        const big = (s, y, col) => V.text(g, s, px, y, { size: mob ? 20 : 24, font: 'display', weight: 700, color: col || C.ink, maxW: pw });
        if (i === 0) {
          big(`36 × ${nf(M.d)}`, py + 34);
          wrap(g, 'Cada vetor já “viu” a imagem inteira (atenção bidirecional), mas ainda está na largura da torre e há um por patch: vetores demais para o LLM.', px, py + 64, pw, 18, { size: 12.5, color: C.muted });
          V.matrix(g, { x: px, y: py + 124, w: Math.min(pw, 240), h: 80, data: V.randMatrix(6, 16, 11), cmap: 'div', gap: 1, r: 1 });
          V.text(g, `[36 × ${M.d}]`, px + Math.min(pw, 240) + 8, py + 170, { size: 12, font: 'mono', color: C.muted });
        } else if (i === 1) {
          big(`${NG} grupos de ${k}×${k}`, py + 34);
          wrap(g, `Cada grupo colorido vira 1 token. ${k === 2 ? 'GLM e Qwen juntam 4 patches' : 'O DeepSeek junta 9 patches (F.unfold com kernel 3 e passo 3)'}.`, px, py + 64, pw, 18, { size: 12.5, color: C.muted });
          for (let gi = 0; gi < Math.min(NG, 9); gi++) pill(g, px + (gi % 3) * 78, py + 118 + Math.floor(gi / 3) * 30, `grupo ${gi}`, gcol(gi), { size: 11.5 });
        } else if (i === 2 || i === 3) {
          const gsel = Math.max(0, selG), gr = Math.floor(gsel / G), gc = gsel % G;
          V.text(g, M.pre ? pick(g, [`LN[1152] por patch → concat ${M.cat}`, `LN por patch + concat = 4608`], 13, 'sans', 600, pw) : `concatenar: ${M.cat}`, px, py + 14, { size: 13, weight: 600, color: C.ink, maxW: pw });
          const K2 = k * k, sgw = (pw - (K2 - 1) * 3) / K2, vy = py + 26, vh = 30;
          const fly = i === 2 ? clamp(((e % 2.2) / 2.2) * 1.5, 0, 1) : 1;
          for (let m = 0; m < K2; m++) {
            const r = gr * k + Math.floor(m / k), c = gc * k + (m % k), idx = r * 6 + c;
            const tx0 = px + m * (sgw + 3), pr0 = V.ease.inOut(clamp(fly * 1.45 - m * 0.045, 0, 1));
            const arrived = pr0 >= 0.999;
            V.box(g, tx0, vy, sgw, vh, { r: 4, fill: V.alpha(gcol(gsel), arrived ? 0.12 : 0.04), stroke: V.alpha(gcol(gsel), arrived ? 1 : 0.35), lw: 1.2 });
            if (arrived) { for (let s2 = 0; s2 < 4; s2++) { g.fillStyle = V.cmap(vecs[idx][s2], 'div'); g.fillRect(tx0 + 2 + (s2 * (sgw - 4)) / 4, vy + 3, (sgw - 4) / 4 - 1, vh - 6); } }
            else if (pr0 > 0) {
              const x0 = gx + c * cs + cs / 2, y0 = gy + r * cs + cs / 2, sz = lerp(cs - 8, Math.min(sgw, vh), pr0);
              const xx = lerp(x0, tx0 + sgw / 2, pr0) - sz / 2, yy = lerp(y0, vy + vh / 2, pr0) - sz / 2;
              V.box(g, xx, yy, sz, sz, { r: 4, fill: V.alpha(gcol(gsel), 0.2), stroke: gcol(gsel), lw: 1.4 });
              const ih = (sz - 6) / 4;
              if (ih > 1) for (let s2 = 0; s2 < 4; s2++) { g.fillStyle = V.cmap(vecs[idx][s2], 'div'); g.fillRect(xx + 3, yy + 3 + s2 * ih, sz - 6, ih - 1); }
            }
          }
          if (i === 3) {
            const bx = px, bw = pw, y1 = vy + vh + 20, bh = 30, gap = 16;
            const boxes = [[M.w1, C.b], [M.act, C.d], [M.w2, C.b]];
            boxes.forEach(([lab, col], j) => {
              const yy = y1 + j * (bh + gap);
              V.arrow(g, bx + bw / 2, yy - gap + 2, bx + bw / 2, yy - 2, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 6 });
              V.box(g, bx, yy, bw, bh, { r: 8, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.8), lw: 1.4 });
              V.text(g, lab, bx + bw / 2, yy + bh / 2 + 1, { size: 12, font: 'mono', color: C.ink, align: 'center', base: 'middle', maxW: bw - 12 });
              const ff = (t * 0.9 + j * 0.33) % 1;
              V.dot(g, bx + bw / 2, yy - gap + 2 + (gap - 4) * ff, 2.5, colR);
            });
            const oy = y1 + 3 * (bh + gap);
            V.arrow(g, bx + bw / 2, oy - gap + 2, bx + bw / 2, oy - 2, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 6 });
            const ocw = bw / 16;
            for (let c = 0; c < 16; c++) { g.fillStyle = V.alpha(colR, 0.3 + 0.6 * Math.abs(Math.sin(c * 1.7 + gsel + t * 0.6))); V.rr(g, bx + c * ocw + 1, oy, ocw - 2, 22, 3); g.fill(); }
            V.text(g, `z ∈ ℝ^${M.dl} · espaço do LLM`, bx, oy + 40, { size: 12.5, font: 'mono', color: colR, weight: 600, maxW: bw });
          } else {
            wrap(g, `Grupo ${gsel}: os ${K2} vetores voam para as ${K2} fatias do vetor concatenado. Nada é descartado: os mesmos números mudam da grade para a profundidade.${M.pre ? ' No Qwen, cada vetor já chega normalizado (LayerNorm de 1152, por patch).' : ''}`, px, vy + vh + 30, pw, 18, { size: 12.5, color: C.muted });
          }
        } else {
          big(`36 → ${NG} tokens (÷${k * k})`, py + 34, colR);
          V.text(g, `1 token ≈ ${M.px}`, px, py + 66, { size: 13, font: 'mono', color: C.ink, maxW: pw });
          V.text(g, `largura: ${nf(M.d)} → ${nf(M.dl)}`, px, py + 88, { size: 13, font: 'mono', color: C.ink, maxW: pw });
          wrap(g, k === 2 ? 'Mais tokens por área: melhor para ler texto miúdo, mais caro em prefill e KV.' : 'O extremo econômico: no máximo 384 tokens por imagem, mesmo em 4K.', px, py + 118, pw, 18, { size: 12.5, color: C.muted });
        }
      });
    })();

    /* ================================================================ FIG 5 */
    (() => {
      const fig = root.querySelector('#fig-layout');
      if (!fig) return;
      autoH(fig, (w) => (w < 620 ? 590 : 440));
      const NHh = 4, NWw = 5, RL = NWw + 1;
      const WORDS = ['Olhe', 'esta', 'foto', 'e', 'diga', 'o', 'que', 'há', 'nela', 'por', 'favor', ':'];
      const TAIL = ['O', 'que', '?'];
      const S = { s: 5, align: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'texto antes (s)', min: 0, max: 11, step: 1, value: 5, onInput: (v) => { S.s = v; } });
      UI.toggle(ctl, { label: 'alinhar a 4 (compress_pad)', value: true, onChange: (v) => { S.align = v; } });
      const rP = UI.readout(ctl, 'compress_pad', '—');
      const rB = UI.readout(ctl, 'tokens do bloco', '—');
      function build() {
        const seq = [];
        for (let i = 0; i < S.s; i++) seq.push({ ty: 'T', lab: WORDS[i % WORDS.length] });
        const pad = S.align ? 3 - (S.s % 4) : 0;
        for (let i = 0; i < pad; i++) seq.push({ ty: 'P' });
        seq.push({ ty: 'S' });
        const body0 = seq.length;
        for (let pp = 0; pp < NHh / 2; pp++) for (let c = 0; c < RL; c++) for (const r of [2 * pp, 2 * pp + 1]) seq.push(c === NWw ? { ty: 'NL', r, c } : { ty: 'I', r, c });
        seq.push({ ty: 'E' });
        TAIL.forEach((x) => seq.push({ ty: 'T', lab: x }));
        return { seq, pad, body0 };
      }
      function cls(tk) {
        const valid = tk.filter(Boolean), body = valid.filter((x) => x.ty === 'I' || x.ty === 'NL').length;
        if (body === 0) return 'txt';
        if (body < valid.length) return 'mix';
        if (valid.length === 4 && valid.every((x) => x.ty === 'I')) {
          const rs = [...new Set(valid.map((x) => x.r))].sort((a, b) => a - b), cc = [...new Set(valid.map((x) => x.c))].sort((a, b) => a - b);
          if (rs.length === 2 && cc.length === 2 && rs[1] - rs[0] === 1 && cc[1] - cc[0] === 1) return 'sq';
        }
        return 'img';
      }
      let lastRO = '', memoK = '', memo = null;
      // a sequência e os grupos só mudam com (s, align): calcula uma vez e reaproveita nos frames seguintes
      function layoutOf() {
        const k = S.s + '|' + S.align;
        if (k === memoK) return memo;
        const { seq, pad, body0 } = build();
        const L = seq.length, NGr = Math.ceil(L / 4), groups = [];
        for (let gi = 0; gi < NGr; gi++) { const tk = seq.slice(gi * 4, gi * 4 + 4); groups.push({ tk, c: cls(tk) }); }
        const posOf = {};
        seq.forEach((tk, p) => { if (tk.ty === 'I' || tk.ty === 'NL') posOf[tk.r + ',' + tk.c] = p; });
        memoK = k;
        return (memo = { seq, pad, body0, L, groups, posOf, body: seq.slice(body0, body0 + NHh * RL), nMix: groups.filter((x) => x.c === 'mix').length });
      }
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620;
        const { seq, pad, body0, L, groups, posOf, body, nMix } = layoutOf();
        const ro = pad + '|' + (pad + 26);
        if (ro !== lastRO) { rP.set(S.align ? String(pad) : 'desligado'); rB.set(String(pad + 26)); lastRO = ro; }
        const gS = Math.floor((body0 - 1) / 4), gE = Math.floor((body0 + NHh * RL) / 4);
        const span = gE - gS + 1, cur = gS + (Math.floor(t / 1.2) % span);
        const ccol = { sq: C.e || C.a, img: C.b, mix: RED(C), txt: C.muted };
        const curSet = new Set();
        for (let p = cur * 4; p < cur * 4 + 4; p++) curSet.add(p);
        // grade
        const cs = mob ? Math.min(40, (w - 40) / RL) : 40;
        const gx = mob ? (w - cs * RL) / 2 : 26, gy = 48;
        V.text(g, 'grade do Aligner 4×5 + coluna ↵', gx, gy - 14, { size: 12.5, font: 'mono', color: C.muted });
        for (let r = 0; r < NHh; r++) for (let c = 0; c < RL; c++) {
          const p = posOf[r + ',' + c], x = gx + c * cs, y = gy + r * cs, isNL = c === NWw, inCur = curSet.has(p);
          const col = isNL ? C.b : C.ds;
          V.box(g, x + 2, y + 2, cs - 4, cs - 4, { r: 6, fill: V.alpha(col, inCur ? 0.45 : 0.16), stroke: inCur ? ccol[groups[cur].c] : V.alpha(col, 0.6), lw: inCur ? 2.5 : 1 });
          V.text(g, isNL ? '↵' : String(p), x + cs / 2, y + cs / 2 + 1, { size: isNL ? 15 : 12, font: 'mono', weight: 600, color: C.ink, align: 'center', base: 'middle' });
        }
        // caminho em N
        g.beginPath();
        body.forEach((tk, j) => { const x = gx + tk.c * cs + cs / 2, y = gy + tk.r * cs + cs / 2; if (j) g.lineTo(x, y); else g.moveTo(x, y); });
        g.strokeStyle = V.alpha(C.c, 0.45); g.lineWidth = 1.4; g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]);
        const fp = (t * 0.25) % 1, jj = Math.floor(fp * (body.length - 1)), fr = fp * (body.length - 1) - jj;
        const a0 = body[jj], a1 = body[Math.min(body.length - 1, jj + 1)];
        V.dot(g, gx + lerp(a0.c, a1.c, fr) * cs + cs / 2, gy + lerp(a0.r, a1.r, fr) * cs + cs / 2, 4, C.c);
        // painel de informações
        const ix = mob ? 20 : gx + RL * cs + 36, iy = mob ? gy + NHh * cs + 28 : gy + 4, iw = mob ? w - 40 : w - ix - 20;
        const lines = [
          [`s = ${S.s} tokens de texto antes`, C.ink],
          [S.align ? `compress_pad = 3 − (${S.s} mod 4) = ${pad}` : 'sem compress_pad (alinhamento desligado)', S.align ? C.d : RED(C)],
          [`IMAGE_START na posição ${body0 - 1}`, C.ink],
          [`corpo começa em ${body0} ${body0 % 4 === 0 ? '(múltiplo de 4 ✓)' : '(fora do múltiplo de 4 ✕)'}`, body0 % 4 === 0 ? (C.e || C.a) : RED(C)],
          [`bloco: ${pad} + 1 + 24 + 1 = ${pad + 26} tokens`, C.ink],
          [[`grupos de 4 que misturam texto e imagem: ${nMix}`, `grupos mistos (texto + imagem): ${nMix}`, `grupos mistos: ${nMix}`], nMix ? RED(C) : (C.e || C.a)],
          [['caso real: torre 19×40 (h×w) → 7×14 → 122 a 125', 'real: torre 19×40 (h×w) → 7×14 → 122–125', 'real: 19×40 (h×w) → 7×14 → 122–125'], C.muted],
        ];
        lines.forEach(([s, col], k) => V.text(g, Array.isArray(s) ? pick(g, s, 12.5, 'mono', 400, iw) : s, ix, iy + k * 19, { size: 12.5, font: 'mono', color: col, maxW: iw }));
        // fila
        const ts = mob ? 20 : 24, gT = 3, groupW = 4 * (ts + gT) + 8;
        const perRow = Math.max(1, Math.floor((w - 40 + 8) / groupW));
        const sy0 = mob ? iy + lines.length * 19 + 22 : Math.max(gy + NHh * cs + 56, iy + lines.length * 19 + 30);
        const sx0 = 20 + Math.max(0, (w - 40 - perRow * groupW + 8) / 2);
        const rowH = ts + 34;
        V.text(g, 'fila do prompt (posições 0 → ' + (L - 1) + ')', 20, sy0 - 8, { size: 12, font: 'mono', color: C.muted });
        const tyCol = { T: C.muted, P: C.d, S: C.a, E: C.a, I: C.ds, NL: C.b };
        const tyLab = { P: 'p', S: '⟨', E: '⟩', NL: '↵' };
        groups.forEach((gr, gi) => {
          const row = Math.floor(gi / perRow), col = gi % perRow, x0 = sx0 + col * groupW, y0 = sy0 + 10 + row * rowH;
          const isCur = gi === cur;
          V.text(g, String(gi * 4), x0, y0 - 3, { size: 11, font: 'mono', color: C.muted });
          gr.tk.forEach((tk, m) => {
            const x = x0 + m * (ts + gT), y = y0 + 4;
            const c0 = tyCol[tk.ty];
            V.box(g, x, y, ts, ts, { r: 5, fill: V.alpha(c0, isCur ? 0.5 : 0.22), stroke: V.alpha(c0, 0.8), lw: 1 });
            const lab = tk.ty === 'T' ? tk.lab.slice(0, 2) : tk.ty === 'I' ? '' : tyLab[tk.ty];
            if (lab) V.text(g, lab, x + ts / 2, y + ts / 2 + 1, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
            if (tk.ty === 'I') V.dot(g, x + ts / 2, y + ts / 2, 3, V.alpha(C.ink, 0.5));
          });
          const bw = gr.tk.length * (ts + gT) - gT, by = y0 + ts + 9;
          g.strokeStyle = ccol[gr.c]; g.lineWidth = isCur ? 3 : 1.8;
          g.beginPath(); g.moveTo(x0, by - 4); g.lineTo(x0, by); g.lineTo(x0 + bw, by); g.lineTo(x0 + bw, by - 4); g.stroke();
          if (isCur) V.glow(g, x0 + bw / 2, by, 14, ccol[gr.c], 0.6);
        });
      });
    })();

    /* ================================================================ FIG 6 */
    (() => {
      const fig = root.querySelector('#fig-mrope');
      if (!fig) return;
      const FH = fitH(fig, (w) => (w < 620 ? 460 : 430));
      const TH = 1e7, om = Array.from({ length: 32 }, (_, m) => Math.pow(TH, (-2 * m) / 64));
      const axisI = (m) => (m % 3 === 1 && m < 33 ? 1 : m % 3 === 2 && m < 30 ? 2 : 0);
      const axisB = (m) => (m < 11 ? 0 : m < 22 ? 1 : 2);
      const S = { T: 1, inter: true, auto: true, hi: 0, ang: new Float32Array(32) };
      const PRE = ['Descreva', 'a', 'foto', '⟨vs⟩'], POST = ['⟨ve⟩', 'Um', 'gato'];
      const Hh = 3, Ww = 4;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Entrada', options: [{ v: 1, label: 'imagem' }, { v: 2, label: 'vídeo (2 grupos)' }], value: 1, onChange: (v) => { S.T = v; } });
      UI.seg(ctl, { label: 'Seções', options: [{ v: 'i', label: 'intercaladas (Qwen3.8)' }, { v: 'b', label: 'em blocos' }], value: 'i', onChange: (v) => { S.inter = v === 'i'; } });
      UI.toggle(ctl, { label: 'percorrer tokens', value: true, onChange: (v) => { S.auto = v; } });
      function seq() {
        const out = [];
        PRE.forEach((x, i) => out.push({ k: 'txt', lab: x, p: [i, i, i] }));
        const s0 = PRE.length;
        for (let tau = 0; tau < S.T; tau++) for (let i = 0; i < Hh; i++) for (let j = 0; j < Ww; j++) out.push({ k: 'img', tau, i, j, p: [s0 + tau, s0 + i, s0 + j] });
        const nx = s0 + Math.max(S.T - 1, Hh - 1, Ww - 1) + 1;
        POST.forEach((x, i) => out.push({ k: 'txt', lab: x, p: [nx + i, nx + i, nx + i] }));
        return { out, nx, s0 };
      }
      const H = V.fig(fig, (g, w, h, t, dt) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620, AX = [C.d, C.a, C.b], AXL = ['t', 'h', 'w'];
        if (LA.paused) dt = 0;
        const { out, nx, s0 } = seq();
        if (S.auto) S.hi = Math.floor(t / 0.9) % out.length;
        S.hi = Math.min(S.hi, out.length - 1);
        const cur = out[S.hi];
        const hits = [];
        const card = (x, y, cw, chh, tk, idx) => {
          const on = idx === S.hi;
          V.box(g, x, y, cw, chh, { r: 8, fill: V.alpha(C.muted, on ? 0.3 : 0.1), stroke: on ? C.c : V.alpha(C.muted, 0.5), lw: on ? 2.5 : 1 });
          V.text(g, tk.lab, x + cw / 2, y + 15, { size: 12, weight: 600, color: C.ink, align: 'center', maxW: cw - 6 });
          V.text(g, `(${tk.p.join(',')})`, x + cw / 2, y + chh - 8, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: cw - 4 });
          hits.push([x, y, cw, chh, idx]);
        };
        let areaBottom;
        if (!mob) {
          const cw = 110, chh = 40, gp = 8, y0 = 44;
          V.text(g, 'texto', 20, y0 - 10, { size: 12, font: 'mono', color: C.muted });
          PRE.forEach((x, i) => card(20, y0 + i * (chh + gp), cw, chh, out[i], i));
          const postIdx0 = PRE.length + S.T * Hh * Ww, rx = w - 20 - cw;
          V.text(g, 'texto depois', rx, y0 - 10, { size: 12, font: 'mono', color: C.muted });
          POST.forEach((x, i) => card(rx, y0 + i * (chh + gp), cw, chh, out[postIdx0 + i], postIdx0 + i));
          const midL = 20 + cw + 30, midR = rx - 30, mid = midR - midL;
          const cell = Math.min(52, (mid - (S.T - 1) * 22) / (S.T * Ww) - 4), cellH = Math.min(38, cell * 0.72 + 6);
          const gridW = Ww * (cell + 4), totalW = S.T * gridW + (S.T - 1) * 22, gx0 = midL + (mid - totalW) / 2;
          for (let tau = 0; tau < S.T; tau++) {
            const gx = gx0 + tau * (gridW + 22), gy = y0 + 18;
            V.text(g, S.T > 1 ? `grupo τ=${tau}` : 'imagem 3×4 tokens', gx, gy - 10, { size: 12, font: 'mono', color: C.muted });
            for (let i = 0; i < Hh; i++) for (let j = 0; j < Ww; j++) {
              const idx = PRE.length + tau * Hh * Ww + i * Ww + j, tk = out[idx], x = gx + j * (cell + 4), y = gy + i * (cellH + 4), on = idx === S.hi;
              V.box(g, x, y, cell, cellH, { r: 6, fill: V.alpha(C.qwen, on ? 0.45 : 0.14), stroke: on ? C.c : V.alpha(C.qwen, 0.6), lw: on ? 2.5 : 1 });
              V.text(g, tk.p.join(','), x + cell / 2, y + cellH / 2 + 1, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle', maxW: cell - 4 });
              hits.push([x, y, cell, cellH, idx]);
            }
          }
          // nota da posição: uma linha se couber entre as colunas de texto; senão, duas (nunca invade os cards)
          const ay = y0 + 18 + Hh * (cellH + 4) + 16;
          const n1 = `imagem começa em s = ${s0}`, n2 = `texto seguinte em ${nx} (não em ${s0 + S.T * Hh * Ww})`, nAll = `${n1} · ${n2}`;
          let ayEnd = ay;
          if (tw(g, nAll, 12, 'mono') <= mid) V.text(g, nAll, midL + mid / 2, ay, { size: 12, font: 'mono', color: C.ink, align: 'center', maxW: mid });
          else { V.text(g, n1, midL + mid / 2, ay, { size: 12, font: 'mono', color: C.ink, align: 'center', maxW: mid }); ayEnd = ay + 17; V.text(g, n2, midL + mid / 2, ayEnd, { size: 12, font: 'mono', color: C.ink, align: 'center', maxW: mid }); }
          areaBottom = Math.max(y0 + 4 * (chh + gp), ayEnd + 10);
        } else {
          const gp = 6, cw = (w - 40 - gp * 3) / 4, chh = 40, y0 = 40;
          V.text(g, 'texto', 20, y0 - 10, { size: 12, font: 'mono', color: C.muted });
          PRE.forEach((x, i) => card(20 + i * (cw + gp), y0, cw, chh, out[i], i));
          const cell = Math.min(60, (w - 40 - (S.T - 1) * 16) / (S.T * Ww) - 4), cellH = 32;
          const gridW = Ww * (cell + 4), totalW = S.T * gridW + (S.T - 1) * 16, gx0 = (w - totalW) / 2, gy = y0 + chh + 34;
          for (let tau = 0; tau < S.T; tau++) {
            const gx = gx0 + tau * (gridW + 16);
            V.text(g, S.T > 1 ? `τ=${tau}` : 'imagem 3×4', gx, gy - 8, { size: 11, font: 'mono', color: C.muted });
            for (let i = 0; i < Hh; i++) for (let j = 0; j < Ww; j++) {
              const idx = PRE.length + tau * Hh * Ww + i * Ww + j, tk = out[idx], x = gx + j * (cell + 4), y = gy + i * (cellH + 4), on = idx === S.hi;
              V.box(g, x, y, cell, cellH, { r: 5, fill: V.alpha(C.qwen, on ? 0.45 : 0.14), stroke: on ? C.c : V.alpha(C.qwen, 0.6), lw: on ? 2.5 : 1 });
              V.text(g, tk.p.join(','), x + cell / 2, y + cellH / 2 + 1, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle', maxW: cell - 2 });
              hits.push([x, y, cell, cellH, idx]);
            }
          }
          const py2 = gy + Hh * (cellH + 4) + 16, postIdx0 = PRE.length + S.T * Hh * Ww, cw2 = (w - 40 - gp * 2) / 3;
          V.text(g, pick(g, [`texto depois (continua em ${nx}, não em ${s0 + S.T * Hh * Ww})`, `depois: continua em ${nx}, não em ${s0 + S.T * Hh * Ww}`], 11.5, 'mono', 400, w - 40), 20, py2, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 40 });
          POST.forEach((x, i) => card(20 + i * (cw2 + gp), py2 + 10, cw2, chh, out[postIdx0 + i], postIdx0 + i));
          areaBottom = py2 + 10 + chh + 6;
        }
        const P = H.pointer;
        if (P.inside) for (const [x, y, ww, hh, idx] of hits) if (P.x >= x && P.x <= x + ww && P.y >= y && P.y <= y + hh) { S.hi = idx; break; }
        // relógios
        const dy = areaBottom + 34, ax = S.inter ? axisI : axisB;
        const desc = cur.k === 'txt' ? `“${cur.lab}” → (${cur.p.join(', ')})` : `imagem (τ=${cur.tau}, i=${cur.i}, j=${cur.j}) → (${cur.p.join(', ')})`;
        const descS = cur.k === 'txt' ? `“${cur.lab}” → (${cur.p.join(',')})` : `img (τ${cur.tau}, i${cur.i}, j${cur.j}) → (${cur.p.join(',')})`;
        V.text(g, pick(g, [`token destacado: ${desc}`, desc, descS], 12.5, 'sans', 600, w - 40), 20, dy - 14, { size: 12.5, weight: 600, color: C.ink, maxW: w - 40 });
        const perRow = 16, rows = 2, sp = (w - 40) / perRow, R = Math.min(sp * 0.36, 16);
        for (let m = 0; m < 32; m++) {
          const a = ax(m), row = Math.floor(m / perRow), col = m % perRow, cx = 20 + sp * (col + 0.5), cy = dy + 20 + row * (2 * R + 34) + R;
          const target = cur.p[a] * om[m];
          S.ang[m] += (target - S.ang[m]) * Math.min(1, dt * 6);
          V.text(g, AXL[a], cx, cy - R - 5, { size: 11, font: 'mono', weight: 700, color: AX[a], align: 'center' });
          g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fillStyle = V.alpha(AX[a], 0.14); g.fill(); g.strokeStyle = V.alpha(AX[a], 0.8); g.lineWidth = 1.3; g.stroke();
          const an = S.ang[m] - Math.PI / 2;
          V.line(g, cx, cy, cx + Math.cos(an) * R * 0.85, cy + Math.sin(an) * R * 0.85, { color: AX[a], lw: 2 });
        }
        const ly = dy + 20 + rows * (2 * R + 34) + 4;
        const l1 = 'par 0 (rápido) → par 31 (quase parado)', l2 = S.inter ? '11 pares t, 11 h, 10 w, intercalados' : '11 pares t, 11 h, 10 w, em faixas';
        let lyEnd = ly;
        if (tw(g, `${l1} · ${l2}`, 12, 'mono') <= w - 40) V.text(g, `${l1} · ${l2}`, 20, ly, { size: 12, font: 'mono', color: C.muted });
        else { V.text(g, l1, 20, ly, { size: 12, font: 'mono', color: C.muted, maxW: w - 40 }); lyEnd = ly + 17; V.text(g, l2, 20, lyEnd, { size: 12, font: 'mono', color: C.muted, maxW: w - 40 }); }
        FH.set(lyEnd + 18);
      });
    })();

    /* ================================================================ FIG 7 */
    (() => {
      const fig = root.querySelector('#fig-video');
      if (!fig) return;
      const FH = fitH(fig, (w) => (w < 620 ? 360 : 330));
      const QB = 16384;
      const S = { m: 'glm', dur: 3, n: 100, bug: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM (vídeo)' }, { v: 'qwen', label: 'Qwen (vídeo)' }, { v: 'ds', label: 'DeepSeek (GIF)' }], value: 'glm', onChange: (v) => { S.m = v; } });
      UI.slider(ctl, { label: 'duração', min: 1, max: 16, step: 1, value: 3, fmt: (v) => v + ' s', onInput: (v) => { S.dur = v; } });
      UI.slider(ctl, { label: 'tokens/grupo', min: 100, max: 2040, step: 20, value: 100, fmt: (v) => nf(v), onInput: (v) => { S.n = v; } });
      UI.toggle(ctl, { label: 'bug sem o patch (GLM, clipe de 1 s)', value: false, onChange: (v) => { S.bug = v; } });
      const rT = UI.readout(ctl, 'tokens de vídeo', '—');
      let lastR = '';
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620, ds = S.m === 'ds', qw = S.m === 'qwen', bug = S.m === 'glm' && S.bug;
        const F = ds ? 6 : bug ? 2 : Math.min(32, 2 * S.dur), G = ds ? 1 : Math.ceil(F / 2), nper = bug ? 100 : S.n, TS = bug ? 3 : G;
        // Qwen (esquema simplificado): grupos que estouram o orçamento de 16.384 são descartados, espaçados por igual
        const K = qw ? Math.max(1, Math.min(G, Math.floor(QB / nper))) : G;
        const kept = new Set();
        for (let k = 0; k < K; k++) kept.add(Math.floor(((k + 0.5) * G) / K));
        const toks = ds ? 0 : K * nper, slots = ds ? 0 : TS * nper;
        const rs = ds ? 'sem vídeo' : nf(toks);
        if (rs !== lastR) { rT.set(rs); lastR = rs; }
        const avail = w - 40;
        const title = ds ? ['GIF animado → só o 1º quadro vira imagem']
          : bug ? ['caso documentado: clipe de 1 s, 2 quadros, grid_t = 1', 'bug: clipe de 1 s, grid_t = 1']
          : qw ? [`${S.dur} s a 2 quadros/s (ilustrativo) → ${F} quadros → ${G} grupos T/2`, `${S.dur} s → ${F} quadros → ${G} grupos T/2`]
          : [`${S.dur} s a 2 fps → ${F} quadros${2 * S.dur > 32 ? ' (teto de 32)' : ''} → grid_t = ${G}`, `${S.dur} s → ${F} quadros${2 * S.dur > 32 ? ' (teto 32)' : ''} → grid_t = ${G}`];
        V.text(g, pick(g, title, 13, 'sans', 600, avail), 20, 24, { size: 13, weight: 600, color: C.ink, maxW: avail });
        const fwid = Math.min(64, avail / F - 4), fh = Math.max(16, fwid * 0.62), fy = 44;
        const fx0 = 20 + (avail - F * (fwid + 4) + 4) / 2;
        for (let f = 0; f < F; f++) {
          const x = fx0 + f * (fwid + 4), dead = (ds && f > 0) || (qw && !kept.has(Math.floor(f / 2)));
          g.globalAlpha = dead ? 0.35 : 1;
          V.box(g, x, fy, fwid, fh, { r: 3, fill: V.alpha(C.ink, 0.08), stroke: V.alpha(C.muted, 0.7), lw: 1 });
          g.fillStyle = V.alpha(C.e || C.a, 0.7); g.fillRect(x + 1, fy + fh * 0.78, fwid - 2, fh * 0.2);
          const bx = x + 4 + (fwid - 8) * (F > 1 ? f / (F - 1) : 0.5), byy = fy + fh * 0.72 - Math.abs(Math.sin(f * 0.9)) * fh * 0.45;
          V.dot(g, bx, byy, Math.max(2, fwid * 0.1), C.d);
          g.globalAlpha = 1;
          if (dead) V.line(g, x + 3, fy + 3, x + fwid - 3, fy + fh - 3, { color: RED(C), lw: 1.5 });
          if (ds && f === 0) { g.strokeStyle = C.ds; g.lineWidth = 2.5; V.rr(g, x - 2, fy - 2, fwid + 4, fh + 4, 4); g.stroke(); }
        }
        const gy2 = fy + fh + 8;
        if (!ds) for (let gi = 0; gi < G; gi++) {
          const x0 = fx0 + gi * 2 * (fwid + 4), x1 = x0 + Math.min(2, F - 2 * gi) * (fwid + 4) - 4, drop = qw && !kept.has(gi), col = drop ? V.alpha(RED(C), 0.7) : C.series[gi % 6];
          g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(x0, gy2); g.lineTo(x0, gy2 + 5); g.lineTo(x1, gy2 + 5); g.lineTo(x1, gy2); g.stroke();
          if (x1 - x0 > 30) V.text(g, drop ? '✕' : qw ? `τ${gi}` : `${gi}.0 s`, (x0 + x1) / 2, gy2 + 18, { size: 11, font: 'mono', color: col, align: 'center' });
        }
        // encoder
        const ey = gy2 + 38;
        if (ds) {
          V.text(g, 'sem encoder de vídeo nos pesos oficiais', 20, ey, { size: 12.5, font: 'mono', color: C.ink, maxW: avail });
          const bw = Math.min(avail, 300);
          V.box(g, 20, ey + 14, bw, 34, { r: 8, fill: V.alpha(C.ds, 0.2), stroke: C.ds, lw: 1.5 });
          V.text(g, 'bloco de imagem: ≤ 384 tokens', 20 + bw / 2, ey + 32, { size: 12, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          const yE = wrap(g, 'O processador decodifica o GIF com o PIL, que devolve o primeiro quadro, e o trata como uma foto comum. Nada de tempo, nada de grid_t.', 20, ey + 76, avail, 18, { size: 12.5, color: C.muted });
          FH.set(yE + 6);
          return;
        }
        const encS = qw ? [`encoder: ${K} de ${G} grupos × ${nf(nper)} tokens = ${nf(toks)}`, `encoder: ${K}/${G} grupos × ${nf(nper)} = ${nf(toks)}`]
          : [`encoder: ${G} grupo${G > 1 ? 's' : ''} × ${nf(nper)} tokens = ${nf(toks)}`, `encoder: ${G} × ${nf(nper)} = ${nf(toks)}`];
        V.text(g, pick(g, encS, 12.5, 'mono', 400, avail), 20, ey, { size: 12.5, font: 'mono', color: C.ink, maxW: avail });
        const ebw = (avail - (K - 1) * 4) / K;
        for (let k = 0; k < K; k++) {
          const x = 20 + k * (ebw + 4), col = C.series[k % 6];
          V.box(g, x, ey + 10, ebw, 20, { r: 4, fill: V.alpha(col, 0.35), stroke: col, lw: 1 });
          if (ebw > 34) V.text(g, nf(nper), x + ebw / 2, ey + 20.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
        }
        const py = ey + 58;
        if (qw) {
          // Qwen: um bloco <|vision_start|> <|video_pad|> × N <|vision_end|> (sem timestamps, esquema simplificado)
          V.text(g, pick(g, [`placeholders: ⟨vision_start⟩ <|video_pad|> (248057) × ${nf(toks)} ⟨vision_end⟩`, `<|video_pad|> (248057) × ${nf(toks)}`, `video_pad × ${nf(toks)}`], 12.5, 'mono', 400, avail), 20, py, { size: 12.5, font: 'mono', color: C.ink, maxW: avail });
          const yb = py + 12, w0 = pill(g, 20, yb + 11, '⟨vs⟩', C.a, { size: 11, padX: 8 });
          const x1 = 20 + w0 + 6, x2 = w - 20 - 44, segW = (x2 - x1 - (K - 1) * 2) / K;
          for (let k = 0; k < K; k++) {
            const x = x1 + k * (segW + 2), col = C.series[k % 6];
            V.box(g, x, yb + 2, segW, 18, { r: 3, fill: V.alpha(col, 0.3), stroke: col, lw: 1 });
          }
          const ff = (t * 0.35) % 1; V.dot(g, x1 + (x2 - x1) * ff, yb + 11, 2.5, C.ink);
          pill(g, w - 20, yb + 11, '⟨ve⟩', C.a, { size: 11, padX: 8, align: 'right' });
          const ry = yb + 52, over = G * nper > QB;
          if (over) V.text(g, pick(g, [`amostragem esparsa: ${G} × ${nf(nper)} = ${nf(G * nper)} > 16.384 → ${G - K} grupo${G - K > 1 ? 's' : ''} descartado${G - K > 1 ? 's' : ''}`, `${nf(G * nper)} > 16.384 → ${G - K} grupo(s) fora`], 13, 'sans', 700, avail), 20, ry, { size: 13, weight: 700, color: C.d, maxW: avail });
          else V.text(g, pick(g, [`✓ ${nf(toks)} ≤ 16.384 (${dec((toks / QB) * 100, 1)}% do orçamento do encoder)`, `✓ ${nf(toks)} ≤ 16.384 (${dec((toks / QB) * 100, 1)}%)`], 13, 'sans', 700, avail), 20, ry, { size: 13, weight: 700, color: C.e || C.a, maxW: avail });
          const yE = wrap(g, 'Esquema simplificado: o processador real também ajusta a resolução; os limites de vídeo ficam em video_preprocessor_config (longest_edge 25.165.824 px).', 20, ry + 22, avail, 17, { size: 12, color: C.muted });
          FH.set(yE + 4);
          return;
        }
        V.text(g, pick(g, [`placeholders: ${TS} bloco${TS > 1 ? 's' : ''} [início · vagas · fim · “t seconds”] = ${nf(slots)} vagas`, `placeholders: ${TS} bloco${TS > 1 ? 's' : ''} = ${nf(slots)} vagas`], 12.5, 'mono', 400, avail), 20, py, { size: 12.5, font: 'mono', color: C.ink, maxW: avail });
        const blkW = Math.max(mob ? 86 : 96, Math.min(180, (avail - 20) / Math.max(1, TS) - 8));
        const perRow = Math.max(1, Math.floor((avail - 20) / (blkW + 8))), rowsMax = mob ? 4 : 3;
        const yb = py + 12;
        pill(g, 20, yb + 11, '⟨v⟩', C.a, { size: 11, padX: 8 });
        for (let b = 0; b < TS; b++) {
          const row = Math.floor(b / perRow);
          if (row >= rowsMax) { V.text(g, `… +${TS - b} blocos`, 44, yb + 11 + row * 34, { size: 12, font: 'mono', color: C.muted }); break; }
          const x = 44 + (b % perRow) * (blkW + 8), y = yb + row * 34;
          const empty = bug && b > 0, col = empty ? RED(C) : C.series[b % 6];
          V.box(g, x, y + 2, 8, 18, { r: 2, fill: V.alpha(C.a, 0.5) });
          const sw = blkW - 8 - 46;
          V.box(g, x + 10, y + 2, sw, 18, { r: 4, fill: V.alpha(col, empty ? 0.08 : 0.3), stroke: col, lw: 1 });
          if (empty) { g.save(); g.beginPath(); g.rect(x + 10, y + 2, sw, 18); g.clip(); g.strokeStyle = V.alpha(RED(C), 0.6); g.lineWidth = 1; g.beginPath(); for (let q = -18; q < sw; q += 6) { g.moveTo(x + 10 + q, y + 20); g.lineTo(x + 28 + q, y + 2); } g.stroke(); g.restore(); }
          else { const ff = (t * 0.6 + b * 0.2) % 1; V.dot(g, x + 12 + (sw - 4) * ff, y + 11, 2.5, C.ink); }
          V.box(g, x + 12 + sw, y + 2, 8, 18, { r: 2, fill: V.alpha(C.a, 0.5) });
          V.text(g, bug ? `t${b}` : `${b}.0 s`, x + sw + 24, y + 15, { size: 11, font: 'mono', color: C.muted });
        }
        const rowsUsed = Math.ceil(TS / perRow), lastRow = Math.min(rowsMax, rowsUsed) + (rowsUsed > rowsMax ? 1 : 0), ry = yb + lastRow * 34 + 26;
        let yE;
        if (bug) {
          V.text(g, pick(g, [`✕ ${nf(toks)} tokens do encoder para ${nf(slots)} vagas de placeholder`, `✕ ${nf(toks)} tokens para ${nf(slots)} vagas`], 13, 'sans', 700, avail), 20, ry, { size: 13, weight: 700, color: RED(C), maxW: avail });
          yE = wrap(g, 'O glm4_1v montava um bloco por timestamp (3), mas a grade do encoder tinha T = 1. O patch alinha os timestamps a grid_t: 1 bloco, 100 vagas.', 20, ry + 22, avail, 18, { size: 12.5, color: C.muted });
        } else {
          V.text(g, `✓ ${nf(toks)} tokens = ${nf(slots)} vagas`, 20, ry, { size: 13, weight: 700, color: C.e || C.a });
          const pct = dec((toks / 240000) * 100, 1);
          V.text(g, pick(g, [`teto de vídeo do processador GLM: 240.000 tokens (${pct}% usado)`, `teto GLM: 240.000 tokens (${pct}% usado)`], 12, 'mono', 400, avail), 20, ry + 22, { size: 12, font: 'mono', color: C.muted, maxW: avail });
          yE = ry + 40;
        }
        FH.set(yE + 4);
      });
    })();

    /* ================================================================ FIG 8 */
    (() => {
      const fig = root.querySelector('#fig-esteira');
      if (!fig) return;
      autoH(fig, (w) => (w < 620 ? 492 : 422));
      const STN = [
        { l: 'API', s: 'image_url', ic: '{ }' },
        { l: 'Validação', s: 'papéis · limites', ic: '✓' },
        { l: 'Processador', s: 'CPU · hash', ic: '⚙' },
        { l: 'Scheduler', s: 'orçamento', ic: '⏱' },
        { l: 'Encoder', s: 'torre + merger', ic: '◧' },
        { l: 'Enc. cache', s: 'GPU', ic: '▦' },
        { l: 'LLM', s: 'prefill · decode', ic: '✎' },
      ];
      const S = { mode: 'data', resend: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: '--mm-encoder-tp-mode', options: [{ v: 'data', label: 'data' }, { v: 'weights', label: 'weights' }], value: 'data', onChange: (v) => { S.mode = v; } });
      UI.toggle(ctl, { label: 'reenviar a mesma imagem', value: false, onChange: (v) => { S.resend = v; } });
      const st = UI.stepper(fig, [
        { t: 'API: a requisição chega', d: 'Uma mensagem <code>user</code> com partes <code>text</code> e <code>image_url</code>. A imagem vem como URL http(s) ou data URL base64; caminhos locais exigem <code>--allowed-local-media-path</code>.' },
        { t: 'Validação', d: 'Papéis e limites. O DeepSeek recusa imagem fora de <code>user</code> (HTTP 400). <code>--limit-mm-per-prompt</code> conta os itens <i>desta</i> requisição: 48 no GLM, 8 no DeepSeek, 1 no Qwen. Clientes sem estado reenviam o histórico inteiro.' },
        { t: 'Processador multimodal (CPU)', d: 'Decodifica, redimensiona para o orçamento de tokens, normaliza e recorta em patches. Expande o placeholder em N cópias e calcula o hash do conteúdo. O resultado fica no cache do processador (1 GiB no GLM; padrão de 4 GiB).' },
        { t: 'Scheduler', d: 'Cada passo tem um orçamento de tokens (<code>max_num_batched_tokens</code>, 7.168 no GLM). Quando o pedaço de prefill alcança os placeholders da imagem, o encoder é agendado para a imagem inteira.' },
        { t: 'Encoder (GPU)', d: 'Torre e merger produzem os vetores. Com <code>data</code>, cada GPU tem a torre inteira (0,84 GiB no Qwen, ~0,42 GiB a mais por GPU) e processa imagens diferentes. Com <code>weights</code>, cada GPU guarda metade das matrizes, e elas trocam somas parciais.' },
        { t: 'Encoder cache', d: 'As saídas ficam na GPU até todas as posições da imagem serem consumidas pelo prefill, o que pode levar vários passos. Segundo a receita GLM, o cache tem o tamanho do MNBT, então uma imagem maior que ele não cabe.' },
        { t: 'LLM: prefill e decode', d: '<code>embed_input_ids</code> espalha os vetores nas linhas dos placeholders, e o prefill segue normal, gravando KV. O hash da imagem entra nas chaves do prefix cache. O decode é só texto, e o rascunhador MTP do Qwen não vê os embeddings da imagem.' },
      ], { autoplay: 7000 });
      let prevI = 0, from = 0;
      function layout(w) {
        const res = [];
        if (w >= 620) { const n = 7, gap = 10, bw = (w - 40 - gap * (n - 1)) / n, bh = 58; for (let i = 0; i < n; i++) res.push({ x: 20 + i * (bw + gap), y: 14, w: bw, h: bh }); return { st: res, bottom: 14 + bh, mob: false }; }
        const bw = (w - 40 - 3 * 8) / 4, bh = 50;
        for (let i = 0; i < 7; i++) { const row = i < 4 ? 0 : 1, col = row === 0 ? i : 3 - (i - 4); res.push({ x: 20 + col * (bw + 8), y: 12 + row * (bh + 20), w: bw, h: bh }); }
        return { st: res, bottom: 12 + 2 * bh + 20, mob: true };
      }
      V.fig(fig, (g, w, h, t) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), Lo = layout(w), i = st.i, e = st.elapsed(), mob = Lo.mob;
        if (i !== prevI) { from = prevI; prevI = i; }
        const cen = (k) => [Lo.st[k].x + Lo.st[k].w / 2, Lo.st[k].y + Lo.st[k].h / 2];
        for (let k = 0; k < 6; k++) {
          const a = Lo.st[k], b = Lo.st[k + 1];
          if (Math.abs(a.y - b.y) < 2) { const dir = b.x > a.x ? 1 : -1; V.arrow(g, dir > 0 ? a.x + a.w + 1 : a.x - 1, a.y + a.h / 2, dir > 0 ? b.x - 1 : b.x + b.w + 1, b.y + b.h / 2, { color: V.alpha(C.muted, 0.7), lw: 1.3, head: 6 }); }
          else V.arrow(g, a.x + a.w / 2, a.y + a.h + 1, b.x + b.w / 2, b.y - 1, { color: V.alpha(C.muted, 0.7), lw: 1.3, head: 6 });
        }
        Lo.st.forEach((s, k) => {
          const on = k === i, col = on ? C.a : C.muted, skip = S.resend && (k === 4);
          V.box(g, s.x, s.y, s.w, s.h, { r: 10, fill: V.alpha(on ? C.a : C.muted, on ? 0.2 : 0.07), stroke: V.alpha(col, on ? 1 : 0.45), lw: on ? 2 : 1 });
          V.text(g, STN[k].ic, s.x + 12, s.y + 20, { size: 14, color: on ? C.a : C.muted, weight: 700 });
          V.text(g, STN[k].l, s.x + (mob ? 30 : 32), s.y + 20, { size: mob ? 11.5 : 12.5, weight: 600, color: C.ink, maxW: s.w - (mob ? 34 : 38) });
          if (skip) V.text(g, 'pulado (cache)', s.x + 12, s.y + s.h - 12, { size: 11, font: 'mono', color: C.d, maxW: s.w - 18 });
          else if (s.w > 84) V.text(g, STN[k].s, s.x + 12, s.y + s.h - 12, { size: 11, color: C.muted, maxW: s.w - 18 });
        });
        // o pacote viaja logo abaixo das estações (não cobre a borda nem o subtítulo da estação ativa)
        const pg = V.ease.inOut(clamp(e / 0.7, 0, 1)), [ax] = cen(from), [bx] = cen(i);
        const off = mob ? 10 : 12, ay = Lo.st[from].y + Lo.st[from].h + off, by = Lo.st[i].y + Lo.st[i].h + off;
        const pkx = lerp(ax, bx, pg), pky = lerp(ay, by, pg) + Math.sin(t * 4) * 1.2;
        V.box(g, pkx - 10, pky - 9, 20, 18, { r: 4, fill: C.c, stroke: V.alpha(C.ink, 0.4), lw: 1 });
        g.fillStyle = 'rgba(255,255,255,0.9)'; g.beginPath(); g.moveTo(pkx - 7, pky + 6); g.lineTo(pkx - 1, pky - 2); g.lineTo(pkx + 7, pky + 6); g.closePath(); g.fill();
        V.dot(g, pkx + 4, pky - 4, 2, 'rgba(255,255,255,0.9)');
        // painel
        const px = 20, py = Lo.bottom + 28, pw = w - 40, ph = h - py - 12;
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, `${i + 1}. ${STN[i].l}`, px + 14, py + 22, { size: 13, weight: 700, color: C.a });
        const cx = px + 14, cy = py + 40, cw = pw - 28, chh = ph - 52;
        const ok = C.e || C.a;
        if (i === 0) {
          const L = ['{"role": "user", "content": [', '  {"type": "text",', '   "text": "O que tem na foto?"},', '  {"type": "image_url",', '   "image_url": {"url":', '    "data:image/jpeg;base64,/9j/4AAQ…"}}', ']}'];
          let chars = Math.floor(e * 70);
          const fs = cw < 420 ? 11 : 12.5;
          L.forEach((ln, k) => {
            if (chars <= 0) return;
            const s = ln.slice(0, chars); chars -= ln.length;
            V.text(g, s, cx, cy + 10 + k * (fs + 7), { size: fs, font: 'mono', color: k >= 3 && k <= 5 ? C.c : C.ink });
          });
          if (cw > 560) { const s = Math.min(chh - 30, 120); g.drawImage(scene.cv, cx + cw - s - 10, cy + 4, s, s); V.text(g, 'bytes JPEG → base64', cx + cw - s / 2 - 10, cy + s + 22, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        } else if (i === 1) {
          const rows = [
            ['papel: user ✓', 'DeepSeek: imagem em system/assistant/tool → HTTP 400 (o 400 fica no histórico, #178)'],
            ['itens: 1 imagem ≤ limite ✓', 'GLM 48 · DeepSeek 8 · Qwen 1 (padrão). Cliente sem estado reenvia tudo a cada turno.'],
            ['origem: data URL base64 ✓', 'arquivo local só com --allowed-local-media-path (o Qwen não define)'],
          ];
          let yy = cy + 6;
          rows.forEach(([a, b], k) => {
            const ap = clamp(e * 2 - k * 0.5, 0, 1);
            if (ap <= 0) return;
            g.globalAlpha = ap;
            V.dot(g, cx + 9, yy + 4, 9, V.alpha(ok, 0.25)); V.text(g, '✓', cx + 9, yy + 5, { size: 12, weight: 700, color: ok, align: 'center', base: 'middle' });
            V.text(g, a, cx + 26, yy + 9, { size: 13, weight: 600, color: C.ink, maxW: cw - 30 });
            yy = wrap(g, b, cx + 26, yy + 28, cw - 30, 17, { size: 12, color: C.muted }) + 6;
            g.globalAlpha = 1;
          });
        } else if (i === 2) {
          const s = Math.min(mob ? 88 : 110, chh - 60), y0 = cy + 4;
          g.drawImage(scene.cv, cx, y0, s, s);
          V.text(g, 'original', cx + s / 2, y0 + s + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          const x2 = cx + s + 30;
          V.arrow(g, cx + s + 4, y0 + s / 2, x2 - 4, y0 + s / 2, { color: C.muted, lw: 1.4, head: 6 });
          g.drawImage(scene.cv, x2, y0, s, s);
          g.beginPath(); for (let k = 1; k < 8; k++) { g.moveTo(x2 + (k * s) / 8, y0); g.lineTo(x2 + (k * s) / 8, y0 + s); g.moveTo(x2, y0 + (k * s) / 8); g.lineTo(x2 + s, y0 + (k * s) / 8); }
          g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 0.8; g.stroke();
          V.text(g, 'redimensionada + patches', x2 + s / 2, y0 + s + 16, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: s + 30 });
          const x3 = x2 + s + 24, rw = cx + cw - x3;
          if (rw > 150) {
            pill(g, x3, y0 + 14, 'hash(bytes) = 5c1e9a…', C.b, { size: 11.5 });
            pill(g, x3, y0 + 46, '<|image|> × 2.040', C.c, { size: 11.5, font: 'sans' });
            V.box(g, x3, y0 + 66, Math.min(rw, 220), 40, { r: 8, fill: V.alpha(S.resend ? ok : C.d, 0.14), stroke: S.resend ? ok : C.d, lw: 1.4 });
            V.text(g, S.resend ? 'cache: HIT, pula o trabalho' : 'cache: MISS, processa e guarda', x3 + 10, y0 + 90, { size: 12, weight: 600, color: C.ink, maxW: Math.min(rw, 220) - 16 });
          } else {
            const y4 = y0 + s + 32;
            pill(g, cx, y4, 'hash(bytes) = 5c1e9a…', C.b, { size: 11 });
            pill(g, cx, y4 + 28, '<|image|> × 2.040', C.c, { size: 11, font: 'sans' });
            V.text(g, S.resend ? 'cache do processador: HIT ✓' : 'cache do processador: MISS', cx, y4 + 56, { size: 12, weight: 600, color: S.resend ? ok : C.d });
          }
          V.text(g, 'cache do processador na RAM: 1 GiB no GLM (padrão 4 GiB)', cx, py + ph - 12, { size: 11.5, font: 'mono', color: C.muted, maxW: cw });
        } else if (i === 3) {
          const bw = cw, y0 = cy + 24, cap = 7168, parts = [[4, C.b, 'decodes'], [1200, C.muted, 'texto'], [2040, C.c, 'imagem']];
          V.text(g, 'um passo: decodes + pedaço de prefill ≤ max_num_batched_tokens = 7.168', cx, y0 - 10, { size: 12, font: 'mono', color: C.ink, maxW: cw });
          V.box(g, cx, y0, bw, 30, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.5), lw: 1 });
          let x = cx; const grow = clamp(e / 1.2, 0, 1);
          parts.forEach(([n, col, lab]) => {
            const ww = Math.max(3, (bw * n) / cap) * grow;
            V.box(g, x, y0, ww, 30, { r: 4, fill: V.alpha(col, 0.55) });
            if (ww > 50) V.text(g, `${lab} ${nf(n)}`, x + ww / 2, y0 + 15.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle', maxW: ww - 4 });
            x += ww;
          });
          V.text(g, 'folga', cx + bw - 8, y0 + 19, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          const msg = S.resend ? 'prefix cache: os blocos com hash(imagem) já têm KV, então nem o encoder precisa rodar' : 'os placeholders da imagem entram neste passo → encoder agendado para a imagem INTEIRA';
          wrap(g, msg, cx, y0 + 58, cw, 18, { size: 12.5, color: S.resend ? ok : C.ink });
          wrap(g, 'Uma imagem não é fatiada no encoder: a torre precisa de todos os patches juntos (atenção bidirecional). O que se fatia é o consumo dos vetores pelo prefill.', cx, y0 + 104, cw, 18, { size: 12, color: C.muted });
        } else if (i === 4) {
          const two = !mob || cw > 300, gw2 = two ? (cw - 40) / 2 : cw, gh2 = Math.min(chh - 40, 150);
          [0, 1].forEach((k) => {
            const gx = cx + k * (gw2 + 40), gy = cy + 4;
            V.box(g, gx, gy, gw2, gh2, { r: 10, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.7), lw: 1.4 });
            V.text(g, `GPU ${k}`, gx + 10, gy + 18, { size: 12.5, weight: 700, color: C.ink });
            const frac = S.mode === 'data' ? 1 : 0.5;
            V.text(g, S.mode === 'data' ? 'torre inteira' : 'metade de cada matriz', gx + 10, gy + 38, { size: 11.5, font: 'mono', color: C.muted, maxW: gw2 - 20 });
            V.box(g, gx + 10, gy + 46, gw2 - 20, 12, { r: 4, fill: V.alpha(C.muted, 0.15) });
            V.box(g, gx + 10, gy + 46, (gw2 - 20) * frac, 12, { r: 4, fill: V.alpha(C.b, 0.7) });
            const s = Math.min(46, gh2 - 76);
            if (s > 18) { g.globalAlpha = S.resend ? 0.3 : 1; g.drawImage(scene.cv, S.mode === 'data' && k === 1 ? 112 : 0, 0, 112, 112, gx + 10, gy + 68, s, s); g.globalAlpha = 1; V.text(g, S.mode === 'data' ? (k ? 'imagem B' : 'imagem A') : 'imagem A (metade do cálculo)', gx + s + 18, gy + 68 + s / 2 + 4, { size: 11.5, font: 'mono', color: C.ink, maxW: gw2 - s - 26 }); }
          });
          if (S.mode === 'weights') {
            const yA = cy + 4 + gh2 / 2, xA = cx + gw2 + 3, xB = cx + gw2 + 37;
            V.line(g, xA, yA, xB, yA, { color: C.d, lw: 2, dash: [4, 4], dashOffset: -t * 30 });
            V.text(g, 'all-reduce', cx + gw2 + 20, yA - 8, { size: 11, font: 'mono', color: C.d, align: 'center' });
          }
          const ny = cy + 4 + gh2 + 22;
          if (S.resend) V.text(g, 'encoder pulado: o KV do prefixo com a imagem já está no prefix cache', cx, ny, { size: 12.5, weight: 600, color: ok, maxW: cw });
          else if (S.mode === 'weights') wrap(g, 'Qwen: MLP 4304 ÷ 2 = 2152, e 2152 mod 16 = 8 ✕ → crash na carga', cx, ny, cw, 18, { size: 12.5, weight: 600, color: RED(C) });
          else wrap(g, 'Sem comunicação: cada GPU codifica imagens inteiras. Custo: cada GPU guarda a torre inteira (0,84 GiB no Qwen), ~0,42 GiB a mais que no modo weights.', cx, ny, cw, 18, { size: 12.5, weight: 600, color: ok });
        } else if (i === 5) {
          const cap = 7168, img = 2040, bw = cw, y0 = cy + 26;
          V.text(g, 'encoder cache ≈ MNBT = 7.168 tokens (receita GLM)', cx, y0 - 10, { size: 12, font: 'mono', color: C.ink, maxW: cw });
          V.box(g, cx, y0, bw, 30, { r: 6, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.5), lw: 1 });
          const iw = (bw * img) / cap, cf = clamp((e - 0.6) / 3, 0, 1), fade = clamp((e - 3.8) / 0.8, 0, 1);
          g.globalAlpha = 1 - fade;
          V.box(g, cx, y0, iw, 30, { r: 5, fill: V.alpha(C.c, 0.6) });
          V.box(g, cx, y0, iw * cf, 30, { r: 5, fill: V.alpha(C.muted, 0.55) });
          V.text(g, 'imagem: 2.040 vetores', cx + iw + 8, y0 + 19, { size: 11.5, font: 'mono', color: C.ink, maxW: bw - iw - 12 });
          g.globalAlpha = 1;
          const k1 = cf < 0.55 ? 'pedaço 1 do prefill consome…' : cf < 1 ? 'pedaço 2 consome o resto…' : fade < 1 ? 'tudo consumido → liberado' : 'espaço livre para a próxima imagem';
          V.text(g, k1, cx, y0 + 54, { size: 12.5, weight: 600, color: C.ink, maxW: cw });
          const cy2 = y0 + 70, c1w = iw * 0.55;
          [[0, c1w, 'pedaço 1', cf > 0 && cf < 0.55], [c1w, iw - c1w, 'pedaço 2', cf >= 0.55 && cf < 1]].forEach(([ox, ww, lab, on]) => {
            V.box(g, cx + ox + 1, cy2, ww - 2, 22, { r: 5, fill: V.alpha(C.b, on ? 0.35 : 0.1), stroke: V.alpha(C.b, on ? 1 : 0.4), lw: on ? 1.8 : 1 });
            if (ww > 56) V.text(g, lab, cx + ox + ww / 2, cy2 + 11.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          });
          V.text(g, '↑ passos de prefill (chunked prefill)', cx + iw + 8, cy2 + 15, { size: 11, font: 'mono', color: C.muted, maxW: Math.max(40, cw - iw - 12) });
          wrap(g, 'O checkpoint GLM pede 8.000 tokens por imagem: mais que os 7.168 do cache. Por isso MM_IMAGE_TOKENS = 2.048.', cx, cy2 + 48, cw, 18, { size: 12, color: C.muted });
        } else {
          const n = mob ? 14 : 22, tsz = Math.min(26, (cw - (n - 1) * 3) / n), y0 = cy + 10, ph0 = mob ? 3 : 5, pn = mob ? 7 : 12;
          const sc = clamp((e - 0.4) / 1.4, 0, 1);
          for (let k = 0; k < n; k++) {
            const x = cx + k * (tsz + 3), isP = k >= ph0 && k < ph0 + pn;
            if (isP) {
              const done = (k - ph0) / pn < sc;
              V.box(g, x, y0, tsz, tsz, { r: 4, fill: done ? V.alpha(C.c, 0.3 + 0.5 * Math.abs(Math.sin(k * 1.3))) : 'transparent', stroke: C.c, lw: 1.3 });
            } else V.box(g, x, y0, tsz, tsz, { r: 4, fill: V.alpha(C.muted, 0.35) });
          }
          V.text(g, 'texto', cx, y0 + tsz + 16, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'placeholders ← vetores da torre (masked scatter)', cx + ph0 * (tsz + 3), y0 + tsz + 16, { size: 11, font: 'mono', color: C.c, maxW: cw - ph0 * (tsz + 3) });
          const y2 = y0 + tsz + 40;
          V.box(g, cx, y2, Math.min(200, cw * 0.5), 30, { r: 8, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.3 });
          V.text(g, 'KV cache', cx + Math.min(200, cw * 0.5) / 2, y2 + 16, { size: 12, weight: 600, color: C.ink, align: 'center', base: 'middle' });
          if (e > 2) ['Um', 'gato', 'laranja'].forEach((wd, k) => { if (e > 2 + k * 0.5) pill(g, cx + Math.min(200, cw * 0.5) + 16 + k * 70, y2 + 15, wd, C.a, { size: 11.5 }); });
          const y3 = y2 + 52;
          V.text(g, S.resend ? '✓ prefix cache: prefixo com a imagem reaproveitado' : 'prefix cache: a chave do bloco inclui hash(imagem)', cx, y3, { size: 12, font: 'mono', color: S.resend ? ok : C.ink, maxW: cw });
          wrap(g, 'Qwen: o rascunhador MTP não recebe os embeddings da imagem, e a aceitação cai nesses prompts.', cx, y3 + 22, cw, 18, { size: 12, color: C.muted });
        }
      });
    })();

    /* ================================================================ FIG 9 */
    (() => {
      const fig = root.querySelector('#fig-roteamento');
      if (!fig) return;
      autoH(fig, (w) => (w < 620 ? 610 : 460));
      const NE = 256, IMG = 129264;
      const hash6 = (id) => { const r = V.rng((id * 2654435761) >>> 0); const set = new Set(); while (set.size < 6) set.add(Math.floor(r() * NE)); return [...set]; };
      const TXT = [['O', 46], [' gato', 3079], [' laranja', 88213], [' está', 1540], [' no', 305], [' telhado', 61277], [',', 11], [' ao', 422], [' lado', 9012], [' da', 219], [' casa', 7431], ['.', 13]];
      const bias = Float32Array.from({ length: NE }, (_, i) => V.rng(700 + i)() * 0.3);
      let serial = 0;
      const S = { fixed: false, batch: 'mix', once: true, heat: new Float32Array(NE), parts: [], tNext: 0, qi: 0, last: null };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'correção #175 (bias_vl, sem hash)', value: false, onChange: (v) => { S.fixed = v; } });
      UI.seg(ctl, { label: 'Lote', options: [{ v: 'text', label: 'só texto' }, { v: 'img', label: 'só imagem' }, { v: 'mix', label: 'misto' }], value: 'mix', onChange: (v) => { S.batch = v; S.qi = 0; } });
      UI.seg(ctl, { label: 'Classificar', options: [{ v: 'once', label: '1× por forward' }, { v: 'each', label: 'em cada gate' }], value: 'once', onChange: (v) => { S.once = v === 'once'; } });
      const rS = UI.readout(ctl, 'syncs/forward', '—');
      let lastS = '';
      const queue = () => {
        const T = TXT.map(([l, id]) => ({ l, id, img: false })), I = Array.from({ length: 12 }, () => ({ l: 'img', id: IMG, img: true }));
        return S.batch === 'text' ? T : S.batch === 'img' ? I : [...T.slice(0, 6), ...I.slice(0, 6)];
      };
      function route(tok) {
        if (!tok.img) return hash6(tok.id);
        if (!S.fixed) return hash6(IMG);
        const r = V.rng(5000 + serial++), sc = [];
        for (let e = 0; e < NE; e++) sc.push([r() + bias[e], e]);
        sc.sort((a, b) => b[0] - a[0]);
        return sc.slice(0, 6).map((x) => x[1]);
      }
      (() => { const Q = queue(); for (let k = 0; k < 40; k++) { for (let e = 0; e < NE; e++) S.heat[e] *= Math.exp(-0.32 / 3); route(Q[k % Q.length]).forEach((e) => { S.heat[e] += 1; }); } S.qi = 40; [0.2, 0.5, 0.8].forEach((p0, k) => { const tok = Q[(40 + k) % Q.length]; S.parts.push({ tok, ex: route(tok), p: p0 }); S.last = tok; }); })();
      V.fig(fig, (g, w, h, t, dt) => {
        if (w < 60 || h < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620, Q = queue();
        if (LA.paused) dt = 0;
        // 1× por forward: com imagem, a classificação reusa o .sum().item() da checagem #172 (0 sync extra); só texto: 0
        const syncs = S.once ? (S.batch === 'text' ? '0' : '1') : '43';
        const syncRO = S.once ? (S.batch === 'text' ? '0' : '0 extra (1 já pago pela #172)') : '43';
        if (syncRO !== lastS) { rS.set(syncRO); lastS = syncRO; }
        // layout
        let qx, qy, gs, gx, gy, rx, ry, rw, rh, gateY;
        if (!mob) { qx = 20; qy = 44; rx = 150; ry = 120; rw = 170; rh = 104; gs = Math.min(h - 150, w - 380); gx = w - gs - 24; gy = 40; gateY = h - 44; }
        else { qx = 20; qy = 40; rx = 20; ry = 96; rw = w - 40; rh = 64; gs = Math.min(w - 60, 320); gx = (w - gs) / 2; gy = ry + rh + 30; gateY = gy + gs + 46; }
        const cell = gs / 16;
        // spawn
        S.tNext -= dt;
        if (S.tNext <= 0 && dt > 0) {
          const tok = Q[S.qi % Q.length]; S.qi++;
          S.parts.push({ tok, ex: route(tok), p: 0 });
          S.last = tok; S.tNext = 0.32;
        }
        for (let e = 0; e < NE; e++) S.heat[e] *= Math.exp(-dt / 3);
        for (let k = S.parts.length - 1; k >= 0; k--) { const pt = S.parts[k]; pt.p += dt / 1.1; if (pt.p >= 1) { pt.ex.forEach((e) => { S.heat[e] += 1; }); S.parts.splice(k, 1); } }
        // fila
        const nShow = mob ? Math.floor((w - 40) / 54) : 8;
        V.text(g, 'fila de tokens', qx, qy - 12, { size: 12, font: 'mono', color: C.muted });
        for (let k = 0; k < nShow; k++) {
          const tok = Q[(S.qi + k) % Q.length], col = tok.img ? C.c : C.b;
          const x = mob ? qx + k * 54 : qx, y = mob ? qy : qy + k * 30, cw = mob ? 48 : 104;
          V.box(g, x, y, cw, 24, { r: 6, fill: V.alpha(col, k === 0 ? 0.35 : 0.14), stroke: V.alpha(col, 0.8), lw: 1 });
          V.text(g, tok.img ? (mob ? '129264' : 'img · 129264') : (mob ? tok.l.trim() : `${tok.l.trim()} · ${tok.id}`), x + cw / 2, y + 12.5, { size: 11, font: 'mono', color: C.ink, align: 'center', base: 'middle', maxW: cw - 6 });
        }
        // roteador
        const lt = S.last, rule = !lt ? '—' : !lt.img ? `tid2eid[${lt.id}]` : S.fixed ? 'top6(s + bias_vl)' : 'tid2eid[129264] ⚠';
        const rc = lt && lt.img && !S.fixed ? RED(C) : lt && lt.img ? C.c : C.b;
        V.box(g, rx, ry, rw, rh, { r: 12, fill: V.alpha(rc, 0.1), stroke: V.alpha(rc, 0.8), lw: 1.6 });
        V.text(g, 'roteador · camada hash 0–2', rx + rw / 2, ry + 20, { size: 12, weight: 700, color: C.ink, align: 'center', maxW: rw - 10 });
        V.text(g, rule, rx + rw / 2, ry + (mob ? 44 : 50), { size: 12.5, font: 'mono', weight: 600, color: rc, align: 'center', maxW: rw - 10 });
        if (!mob) wrap(g, lt && lt.img ? (S.fixed ? 'imagem: viés próprio, sem tabela' : 'mesmo id → mesmos 6 experts') : 'texto: a tabela depende do id', rx + 10, ry + 76, rw - 20, 16, { size: 11.5, color: C.muted });
        // experts
        let mx = 0, sum = 0; for (let e = 0; e < NE; e++) { mx = Math.max(mx, S.heat[e]); sum += S.heat[e]; }
        const mean = sum / NE;
        V.text(g, `256 experts · pico = ${mean > 0.01 ? (mx / mean).toFixed(0) : '—'}× a média`, gx, gy - 12, { size: 12, font: 'mono', color: C.muted, maxW: gs });
        for (let e = 0; e < NE; e++) {
          const r = Math.floor(e / 16), c = e % 16, x = gx + c * cell, y = gy + r * cell, v = clamp(S.heat[e] / 3.2, 0, 1);
          g.fillStyle = v > 0.02 ? V.cmap(0.1 + 0.9 * v, 'heat') : V.alpha(C.muted, 0.1);
          V.rr(g, x + 1, y + 1, cell - 2, cell - 2, 2); g.fill();
          if (v > 0.7) { V.glow(g, x + cell / 2, y + cell / 2, cell * 1.4, C.d, 0.55); if (cell >= 15) V.text(g, S.heat[e].toFixed(0), x + cell / 2, y + cell / 2 + 1, { size: 11, font: 'mono', weight: 700, color: C.dark ? '#1a0b33' : '#ffffff', align: 'center', base: 'middle' }); }
        }
        // partículas
        const srcX = mob ? qx + 24 : qx + 104, srcY = mob ? qy + 24 : qy + 12;
        S.parts.forEach((pt) => {
          const col = pt.tok.img ? C.c : C.b;
          if (pt.p < 0.35) { const f = V.ease.inOut(pt.p / 0.35); V.dot(g, lerp(srcX, rx + rw / 2, f), lerp(srcY, ry + rh / 2, f), 3.5, col); }
          else {
            const f = V.ease.inOut((pt.p - 0.35) / 0.65);
            pt.ex.forEach((e) => { const tx = gx + (e % 16) * cell + cell / 2, ty = gy + Math.floor(e / 16) * cell + cell / 2; V.dot(g, lerp(rx + rw / 2, tx, f), lerp(ry + rh / 2, ty, f), 2.4, col); });
          }
        });
        // gates
        const gAvail = w - 40 - (mob ? 0 : 150), perRow = mob ? 22 : 43, gw = gAvail / perRow, gx0 = mob ? 20 : 170;
        V.text(g, S.once ? 'embed_input_ids classifica 1×' : 'cada gate pergunta à GPU', 20, gateY - (mob ? 18 : 22), { size: 12, weight: 600, color: S.once ? (C.e || C.a) : RED(C), maxW: mob ? w - 40 : 150 });
        if (!mob) {
          V.text(g, S.once ? (syncs === '0' ? '43 gates · 0 syncs' : '43 gates · 0 extra') : '43 gates · 43 syncs', 20, gateY + 4, { size: 11.5, font: 'mono', color: C.muted, maxW: 146 });
          if (S.once && syncs !== '0') V.text(g, '(1 sync, o da #172)', 20, gateY + 19, { size: 11.5, font: 'mono', color: C.muted, maxW: 146 });
        }
        const sweep = (t * 12) % 43;
        for (let k = 0; k < 43; k++) {
          const row = Math.floor(k / perRow), col = k % perRow, x = gx0 + col * gw, y = gateY - 14 + row * 24, hashL = k < 3;
          const lit = Math.abs(k - sweep) < 1.5;
          const colG = S.once ? (C.e || C.a) : RED(C);
          V.box(g, x + 1, y, gw - 2, 18, { r: 4, fill: V.alpha(lit ? colG : C.muted, lit ? 0.5 : 0.12), stroke: hashL ? C.d : V.alpha(C.muted, 0.4), lw: hashL ? 1.6 : 1 });
          if (!S.once && lit) V.dot(g, x + gw / 2, y - 4, 2.5, RED(C));
        }
        if (mob) V.text(g, pick(g, [S.once ? (syncs === '0' ? '43 gates (0–2 = hash) · 0 syncs por forward' : '43 gates (0–2 = hash) · 0 sync extra (reusa a da #172)') : '43 gates (0–2 = hash) · 43 syncs por forward', S.once ? (syncs === '0' ? '43 gates · 0 syncs' : '43 gates · 0 extra (reusa #172)') : '43 gates · 43 syncs'], 11.5, 'mono', 400, w - 40), 20, gateY + 42, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 40 });
      });
    })();

    /* ================================================================ FIG 10 */
    (() => {
      const fig = root.querySelector('#fig-orcamento');
      if (!fig) return;
      const FH = fitH(fig, (w) => (w < 620 ? 920 : 450));
      const PRE = {
        p1080: [1920, 1080, 'paisagem 1080p (1920×1080)'],
        port: [1080, 1920, 'screenshot retrato 1080×1920'],
        k4: [3840, 2160, '4K (3840×2160)'],
        foto: [4032, 3024, 'foto 12 MP (4032×3024)'],
        vga: [640, 480, '640×480'],
        icon: [256, 256, 'ícone 256×256'],
        inc: [3136, 1792, 'quadro do incidente (3136×1792)'],
      };
      const S = { img: 'p1080', n: 1, cap: 2048 };
      // figura estática: só redesenha quando um controle muda (nada se anima aqui)
      let HF = null;
      const redraw = () => { if (HF) HF.redraw(); };
      const ctl = UI.controls(fig);
      const selI = UI.select(ctl, { label: 'Imagem', options: Object.keys(PRE).map((k) => ({ v: k, label: PRE[k][2] })), value: 'p1080', onChange: (v) => { S.img = v; redraw(); } });
      selI.el.style.maxWidth = '100%'; selI.el.style.minWidth = '0';
      { const se = selI.el.querySelector('select'); if (se) { se.style.maxWidth = '100%'; se.style.minWidth = '0'; se.style.flex = '1 1 auto'; } }
      UI.slider(ctl, { label: 'imagens', min: 1, max: 64, step: 1, value: 1, onInput: (v) => { S.n = v; redraw(); } });
      UI.seg(ctl, { label: 'MM_IMAGE_TOKENS (GLM)', options: [1024, 2048, 8000], value: 2048, onChange: (v) => { S.cap = v; redraw(); } });
      const rhe = (x) => { const f = Math.floor(x), d = x - f; if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1; return Math.round(x); };
      function glmTok(W, H, cap) {
        const F = 28, s = Math.min(1, Math.sqrt(cap / ((W * H) / (F * F))));
        let gw = Math.max(1, Math.round((W * s) / F)), gh = Math.max(1, Math.round((H * s) / F));
        while (gw * gh > cap) { if (gw >= gh) gw--; else gh--; }
        return { gw, gh, n: gw * gh };
      }
      function qwenTok(W, H) {
        const F = 32, minP = 65536, maxP = 16777216;
        let hb = Math.max(F, rhe(H / F) * F), wb = Math.max(F, rhe(W / F) * F);
        if (hb * wb > maxP) { const b = Math.sqrt((H * W) / maxP); hb = Math.floor(H / b / F) * F; wb = Math.floor(W / b / F) * F; }
        else if (hb * wb < minP) { const b = Math.sqrt(minP / (H * W)); hb = Math.ceil((H * b) / F) * F; wb = Math.ceil((W * b) / F) * F; }
        return { gw: wb / F, gh: hb / F, n: (wb / F) * (hb / F) };
      }
      function dsGrid(bh, bw, p, r) {
        const nh = Math.ceil(Math.floor(bh / p) / r), nw = Math.ceil(Math.floor(bw / p) / r);
        let num = nh * (nw + 1) + 2;
        if (nh % 2 === 1) num += nw + 1;
        num += ((Math.floor((nh + 1) / 2) * (nw + 1)) % 2) * 2;
        return [nh, nw, num];
      }
      function dsSolve(H, W, p, r, mt) {
        const ratio = H / W, mwf = Math.sqrt((mt - 2) / ratio + 0.25) - 0.5, mhf = mwf * ratio;
        let bw, bh;
        if (mwf < 1) { const mw = 1; let mh = Math.floor((mt - 2) / (mw + 1)); if (mh % 2 === 1) mh -= 1; bw = mw * p * r; bh = mh * p * r; }
        else if (mhf < 2) { const mh = 2, mw = Math.floor((mt - 2) / mh) - 1; bw = mw * p * r; bh = mh * p * r; }
        else { const mw = Math.floor(mwf); let mh = Math.floor(mhf); if (mh % 2 === 1) mh -= 1; const beta = Math.min((mw * p * r) / W, (mh * p * r) / H); bw = Math.floor((W * beta) / p) * p; bh = Math.floor((H * beta) / p) * p; }
        const [nh, nw, num] = dsGrid(bh, bw, p, r);
        return [nh, nw, bh, bw, num];
      }
      function dsTok(W0, H0) {
        const p = 14, r = 3, minP = 147456, maxR = 8;
        let W = W0, H = H0;
        if (W > H * maxR) W = H * maxR;
        if (W * H > 0 && W * H < minP) { const q = Math.sqrt(minP / (W * H)); W = Math.floor(W * q); H = Math.floor(H * q); }
        let bw = Math.ceil(W / p) * p, bh = Math.ceil(H / p) * p;
        const mt = 384 - 3; // safe_resize: max_n_token -= COMPRESS_PAD_TO - 1 (o pad cabe dentro dos 384)
        let [nh, nw, num] = dsGrid(bh, bw, p, r);
        let budget = mt;
        while (num > mt && budget > 4) { [nh, nw, bh, bw, num] = dsSolve(H, W, p, r, budget); budget -= 1; }
        return { gw: nw, gh: nh, n: num };
      }
      // memo por (imagem, teto): os redimensionadores rodam uma vez por combinação
      const memo = {};
      const toksFor = (k) => {
        const key = k + '|' + S.img + '|' + (k === 'glm' ? S.cap : '');
        if (memo[key]) return memo[key];
        const [W, Hh] = PRE[S.img];
        return (memo[key] = k === 'glm' ? glmTok(W, Hh, S.cap) : k === 'qwen' ? qwenTok(W, Hh) : dsTok(W, Hh));
      };
      function cover(g, x, y, w, h) {
        const s = 224, ar = w / h;
        let sw, sh;
        if (ar >= 1) { sw = s; sh = s / ar; } else { sh = s; sw = s * ar; }
        g.drawImage(scene.cv, (s - sw) / 2, (s - sh) / 2, sw, sh, x, y, w, h);
      }
      const WORST = 48 * 2048, INC = 236544;
      HF = V.fig(fig, (g, w) => {
        if (w < 60) return; // tamanho transitório (layout/screenshot): não desenha em vez de quebrar a figura
        const C = V.C(), mob = w < 620, [W, Hh, lab] = PRE[S.img], ar = W / Hh;
        const models = [
          { k: 'glm', name: [`GLM · teto ${nf(S.cap)}`], col: C.glm, r: toksFor('glm') },
          { k: 'qwen', name: ['Qwen · orçamento 16.384', 'Qwen · 16.384'], col: C.qwen, r: toksFor('qwen') },
          { k: 'ds', name: ['DeepSeek · ≤ 384, pad incluso', 'DeepSeek · ≤ 384'], col: C.ds, r: toksFor('ds') },
        ];
        V.text(g, pick(g, [`${lab} · ${S.n} imagem${S.n > 1 ? 's' : ''} por requisição`, `${lab} · ${S.n} por requisição`], 13, 'sans', 600, w - 40), 20, 22, { size: 13, weight: 600, color: C.ink, maxW: w - 40 });
        const pw = mob ? w - 40 : (w - 40 - 28) / 3;
        // desenha o conteúdo de um painel e devolve o y final (a moldura vem depois, por baixo)
        function panel(M, x0, y0) {
          V.text(g, pick(g, M.name, 12.5, 'sans', 700, pw - 20), x0 + 12, y0 + 20, { size: 12.5, weight: 700, color: M.col, maxW: pw - 20 });
          const boxW = mob ? pw * 0.5 : pw - 24, boxH = mob ? 116 : 138;
          let iw = boxW, ih = boxW / ar;
          if (ih > boxH) { ih = boxH; iw = boxH * ar; }
          const ix = x0 + 12 + (boxW - iw) / 2, iy = y0 + 32 + (boxH - ih) / 2;
          cover(g, ix, iy, iw, ih);
          const cwx = iw / M.r.gw, chy = ih / M.r.gh, stepX = Math.max(1, Math.ceil(3 / cwx)), stepY = Math.max(1, Math.ceil(3 / chy));
          g.beginPath();
          for (let k = stepX; k < M.r.gw; k += stepX) { g.moveTo(ix + k * cwx, iy); g.lineTo(ix + k * cwx, iy + ih); }
          for (let k = stepY; k < M.r.gh; k += stepY) { g.moveTo(ix, iy + k * chy); g.lineTo(ix + iw, iy + k * chy); }
          g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 0.6; g.stroke();
          g.strokeStyle = M.col; g.lineWidth = 1.5; g.strokeRect(ix, iy, iw, ih);
          const tx = mob ? x0 + 12 + boxW + 14 : x0 + 12, tmax = mob ? pw - boxW - 30 : pw - 24;
          const lim = M.k === 'glm' ? 48 : M.k === 'ds' ? 8 : 1, okN = S.n <= lim, total = M.r.n * S.n;
          let y = mob ? y0 + 50 : y0 + 32 + boxH + 30;
          V.text(g, M.k === 'ds' ? `${nf(M.r.n)} + pad` : nf(M.r.n), tx, y, { size: mob ? 20 : 24, font: 'display', weight: 700, color: C.ink, maxW: tmax });
          const mono = (s, yy, col) => V.text(g, s, tx, yy, { size: 11.5, font: 'mono', color: col || C.muted, maxW: tmax });
          const l1 = `tokens/imagem · grade ${M.r.gw}×${M.r.gh}`;
          if (!mob && tw(g, l1, 11.5, 'mono') <= tmax) { mono(l1, y += 18); }
          else { mono('tokens/imagem', y += 18); mono(`grade ${M.r.gw}×${M.r.gh}`, y += 16); }
          const tl = [`× ${S.n} = ${nf(total)} tokens no prompt`, `× ${S.n} = ${nf(total)} no prompt`];
          if (tw(g, tl[1], 12, 'mono') <= tmax) V.text(g, pick(g, tl, 12, 'mono', 400, tmax), tx, (y += 20), { size: 12, font: 'mono', color: okN ? C.ink : C.muted, maxW: tmax });
          else { V.text(g, `× ${S.n} = ${nf(total)}`, tx, (y += 20), { size: 12, font: 'mono', color: okN ? C.ink : C.muted, maxW: tmax }); mono('tokens no prompt', y += 16); }
          const checks = [];
          if (M.k === 'glm') {
            checks.push([M.r.n <= 7168, M.r.n <= 7168 ? '≤ 7.168 (encoder cache = MNBT)' : `${nf(M.r.n)} > 7.168: não cabe no encoder cache`]);
            checks.push([okN, okN ? `${S.n} ≤ 48 imagens (LIMIT_MM)` : `${S.n} > 48 → HTTP 500 “At most 48 image(s)”`]);
            if (!okN) checks.push([null, 'recusada antes do encode: nada é codificado']);
            else if (total <= WORST) checks.push([true, `encode ${nf(total)} ≤ 98.304 (pior caso da receita: 48 × 2.048)`]);
            else checks.push([false, `encode ${nf(total)} > 98.304 sem nada reservado: risco de OOM-kill do VLLM::Worker_TP${total === INC ? ' (é o incidente de 09-14)' : ''}`]);
          } else if (M.k === 'qwen') {
            checks.push([true, '≤ 16.384 tokens (auto-resize)']);
            checks.push([okN, okN ? '1 imagem ≤ limite padrão' : `${S.n} > 1 → recusada (padrão: 1 imagem)`]);
          } else {
            checks.push([true, '≤ 381 + pad (0–3) ≤ 384: o resize reserva 3 tokens para o compress_pad']);
            checks.push([okN, okN ? `${S.n} ≤ 8 imagens (LIMIT_MM_PER_PROMPT)` : `${S.n} > 8 → HTTP 400 “At most 8 image(s)”`]);
          }
          y = mob ? Math.max(y + 10, y0 + 32 + boxH) + 22 : y + 26;
          checks.forEach(([ok, s]) => {
            const col = ok === null ? C.muted : ok ? (C.e || C.a) : RED(C);
            V.text(g, ok === null ? '·' : ok ? '✓' : '✕', x0 + 14, y, { size: 13, weight: 700, color: col });
            y = wrap(g, s, x0 + 30, y, pw - 42, 15, { size: 12, color: ok === false ? col : C.ink }) + 5;
          });
          if (M.k === 'glm') y = wrap(g, 'nada reservado (--skip-mm-profiling)', x0 + 14, y + 2, pw - 26, 15, { size: 11.5, font: 'mono', color: C.muted });
          return y;
        }
        const frame = (M, x0, y0, hh) => {
          g.save(); g.globalCompositeOperation = 'destination-over';
          V.box(g, x0, y0, pw, hh, { r: 12, fill: V.alpha(M.col, 0.05), stroke: V.alpha(M.col, 0.55), lw: 1.3 });
          g.restore();
        };
        let bottom;
        if (!mob) {
          const y0 = 36, ends = models.map((M, idx) => panel(M, 20 + idx * (pw + 14), y0));
          const hh = Math.max(...ends) - y0 + 6;
          models.forEach((M, idx) => frame(M, 20 + idx * (pw + 14), y0, hh));
          bottom = y0 + hh;
        } else {
          let y0 = 36;
          models.forEach((M) => { const e = panel(M, 20, y0), hh = e - y0 + 6; frame(M, 20, y0, hh); y0 += hh + 10; });
          bottom = y0 - 10;
        }
        FH.set(bottom + 10);
      }, { static: true });
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => setTimeout(redraw, 0));
    })();
    /* ================================================================ FIG 11 · lane TensorFold */
    (() => {
      const fig = root.querySelector('#fig-tf-orcamento');
      if (!fig) return;
      const FH = fitH(fig, (w) => (w < 620 ? 700 : 440));
      const tfCol = (C) => { if (!C.__tf) C.__tf = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; return C.__tf; };
      const PRE = {
        p1080: [1920, 1080, 'paisagem 1080p (1920×1080)'],
        inc: [3136, 1792, 'quadro do incidente (3136×1792)'],
      };
      const S = { n: 12, img: 'p1080' };
      // figura estática: só redesenha quando um controle muda
      let HF = null;
      const redraw = () => { if (HF) HF.redraw(); };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'imagens na requisição', min: 1, max: 50, step: 1, value: S.n, onInput: (v) => { S.n = v; redraw(); } });
      UI.seg(ctl, { label: 'Imagem', options: [{ v: 'p1080', label: '1080p' }, { v: 'inc', label: 'quadro do incidente' }], value: S.img, onChange: (v) => { S.img = v; redraw(); } });
      // redimensionador aproximado do GLM (o mesmo da calculadora da seção 8): tela de 28 px por token, até o teto
      function glmTok(W, H, cap) {
        const F = 28, s = Math.min(1, Math.sqrt(cap / ((W * H) / (F * F))));
        let gw = Math.max(1, Math.round((W * s) / F)), gh = Math.max(1, Math.round((H * s) / F));
        while (gw * gh > cap) { if (gw >= gh) gw--; else gh--; }
        return { gw, gh, n: gw * gh };
      }
      // patch 0050: as imagens de uma requisição dividem 16.384 tokens, com teto de 2.048 e piso de 16
      const capTF = (n) => Math.max(16, Math.min(2048, Math.floor(16384 / n)));
      const memo = {};
      const tok = (cap) => { const k = S.img + '|' + cap; if (!memo[k]) { const [W, Hh] = PRE[S.img]; memo[k] = glmTok(W, Hh, cap); } return memo[k]; };
      function cover(g, x, y, w, h) {
        const s = 224, ar = w / h;
        let sw, sh;
        if (ar >= 1) { sw = s; sh = s / ar; } else { sh = s; sw = s * ar; }
        g.drawImage(scene.cv, (s - sw) / 2, (s - sh) / 2, sw, sh, x, y, w, h);
      }
      const WORST = 48 * 2048, BUD = 16384, MAXV = 50 * 2048;
      HF = V.fig(fig, (g, w) => {
        if (w < 60) return; // tamanho transitório (layout/screenshot)
        const C = V.C(), TFc = tfCol(C), mob = w < 620, [W, Hh, lab] = PRE[S.img], ar = W / Hh, n = S.n;
        const capT = capTF(n);
        const lanes = [
          { k: 'vllm', name: ['Lane vLLM · 2.048 por imagem', 'vLLM · 2.048 por imagem'], short: 'vLLM', col: C.glm, cap: 2048, r: tok(2048), lim: 48 },
          { k: 'tf', name: ['Lane TensorFold · 16.384 por requisição', 'TensorFold · 16.384 por req.'], short: 'TensorFold', col: TFc, cap: capT, r: tok(capT), lim: 50 },
        ];
        V.text(g, pick(g, [`${lab} · ${n} imagem${n > 1 ? 's' : ''} na requisição`, `${n} imagem${n > 1 ? 's' : ''} na requisição`], 13, 'sans', 600, w - 40), 20, 22, { size: 13, weight: 600, color: C.ink, maxW: w - 40 });
        const pw = mob ? w - 40 : (w - 40 - 14) / 2;
        function panel(M, x0, y0) {
          V.text(g, pick(g, M.name, 12.5, 'sans', 700, pw - 24), x0 + 12, y0 + 20, { size: 12.5, weight: 700, color: M.col, maxW: pw - 24 });
          const boxW = mob ? Math.min(pw * 0.46, 190) : pw - 24, boxH = mob ? 104 : 124;
          let iw = boxW, ih = boxW / ar;
          if (ih > boxH) { ih = boxH; iw = boxH * ar; }
          const ix = x0 + 12 + (mob ? 0 : (boxW - iw) / 2), iy = y0 + 34 + (boxH - ih) / 2;
          cover(g, ix, iy, iw, ih);
          const cwx = iw / M.r.gw, chy = ih / M.r.gh, stepX = Math.max(1, Math.ceil(3 / cwx)), stepY = Math.max(1, Math.ceil(3 / chy));
          g.beginPath();
          for (let k = stepX; k < M.r.gw; k += stepX) { g.moveTo(ix + k * cwx, iy); g.lineTo(ix + k * cwx, iy + ih); }
          for (let k = stepY; k < M.r.gh; k += stepY) { g.moveTo(ix, iy + k * chy); g.lineTo(ix + iw, iy + k * chy); }
          g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 0.6; g.stroke();
          g.strokeStyle = M.col; g.lineWidth = 1.5; g.strokeRect(ix, iy, iw, ih);
          const tx = mob ? x0 + 12 + boxW + 14 : x0 + 12, tmax = mob ? pw - boxW - 30 : pw - 24;
          let y = mob ? y0 + 50 : y0 + 34 + boxH + 30;
          V.text(g, nf(M.r.n), tx, y, { size: mob ? 20 : 24, font: 'display', weight: 700, color: C.ink, maxW: tmax });
          const mono = (s, yy, col) => V.text(g, s, tx, yy, { size: 11.5, font: 'mono', color: col || C.muted, maxW: tmax });
          const l1 = `tokens/imagem · grade ${M.r.gw}×${M.r.gh}`;
          if (tw(g, l1, 11.5, 'mono') <= tmax) mono(l1, y += 18);
          else { mono('tokens/imagem', y += 18); mono(`grade ${M.r.gw}×${M.r.gh}`, y += 16); }
          const capS = M.k === 'vllm' ? ['teto fixo: 2.048 (MM_IMAGE_TOKENS)', 'teto 2.048']
            : n <= 8 ? ['teto cheio: 2.048 (até 8 imagens)', 'teto 2.048']
              : [`teto ⌊16.384 / ${n}⌋ = ${nf(M.cap)}`, `teto ${nf(M.cap)}`];
          mono(pick(g, capS, 11.5, 'mono', 400, tmax), y += 16);
          const refused = n > M.lim, total = M.r.n * n;
          const tl = [`× ${n} = ${nf(total)} tokens de encode`, `× ${n} = ${nf(total)}`];
          V.text(g, pick(g, tl, 12, 'mono', 400, tmax), tx, (y += 20), { size: 12, font: 'mono', color: refused ? C.muted : C.ink, maxW: tmax });
          y = mob ? Math.max(y + 10, y0 + 34 + boxH) + 22 : y + 26;
          const checks = M.k === 'vllm'
            ? [[!refused, refused ? `${n} > 48 → HTTP 500 “At most 48 image(s)”` : `${n} ≤ 48 imagens (LIMIT_MM)`], [null, 'nada reservado (--skip-mm-profiling)']]
            : [[true, `${n} ≤ 50 imagens (TENSORFOLD_GLM_MAX_IMAGES)`], [null, 'encode ≤ 16.384, dentro do workspace do rank 0']];
          checks.forEach(([ok, s]) => {
            const col = ok === null ? C.muted : ok ? (C.e || C.a) : RED(C);
            V.text(g, ok === null ? '·' : ok ? '✓' : '✕', x0 + 14, y, { size: 13, weight: 700, color: col });
            y = wrap(g, s, x0 + 30, y, pw - 42, 15, { size: 12, color: ok === false ? col : ok === null ? C.muted : C.ink }) + 4;
          });
          return y;
        }
        const frame = (M, x0, y0, hh) => {
          g.save(); g.globalCompositeOperation = 'destination-over';
          V.box(g, x0, y0, pw, hh, { r: 12, fill: V.alpha(M.col, 0.05), stroke: V.alpha(M.col, 0.55), lw: 1.3 });
          g.restore();
        };
        let y;
        if (!mob) {
          const y0 = 36, ends = lanes.map((M, i) => panel(M, 20 + i * (pw + 14), y0));
          const hh = Math.max(...ends) - y0 + 4;
          lanes.forEach((M, i) => frame(M, 20 + i * (pw + 14), y0, hh));
          y = y0 + hh;
        } else {
          let y0 = 36;
          lanes.forEach((M) => { const e = panel(M, 20, y0), hh = e - y0 + 4; frame(M, 20, y0, hh); y0 += hh + 10; });
          y = y0 - 10;
        }
        // barras: tokens de imagem codificados pela requisição inteira
        y += 28;
        V.text(g, pick(g, ['tokens de imagem codificados na requisição', 'tokens de encode na requisição'], 12.5, 'sans', 700, w - 40), 20, y, { size: 12.5, weight: 700, color: C.ink });
        y += 14;
        const lw = mob ? 82 : 96, bx0 = 20 + lw, bx1 = w - 20, sx = (v) => bx0 + (bx1 - bx0) * Math.min(1, v / MAXV);
        const bh = 20, gap = 10, yA = y, yB = y + 2 * bh + gap, refX = [sx(BUD), sx(WORST)];
        [[BUD, mob ? ['16.384'] : ['16.384 · orçamento TF', '16.384'], 'left'], [WORST, mob ? ['98.304'] : ['98.304 · pior caso vLLM', '98.304'], 'right']].forEach(([v, labs, al]) => {
          const x = sx(v);
          g.save(); g.setLineDash([4, 3]); g.strokeStyle = V.alpha(C.ink, 0.55); g.lineWidth = 1;
          g.beginPath(); g.moveTo(x, yA - 4); g.lineTo(x, yB + 4); g.stroke(); g.restore();
          const room = al === 'left' ? bx1 - x - 4 : x - bx0 - 4;
          V.text(g, pick(g, labs, 11, 'mono', 400, room), al === 'left' ? x + 3 : x - 3, yB + 16, { size: 11, font: 'mono', color: C.muted, align: al });
        });
        lanes.forEach((M, i) => {
          const yy = y + i * (bh + gap), refused = n > M.lim, total = M.r.n * n;
          V.text(g, M.short, 20, yy + bh / 2 + 0.5, { size: 12, weight: 700, color: M.col, base: 'middle' });
          V.box(g, bx0, yy, bx1 - bx0, bh, { r: 5, fill: V.alpha(C.muted, 0.1) });
          if (refused) {
            const x0r = Math.max(bx0 + 8, sx(BUD) + 8);
            V.text(g, pick(g, ['recusada antes do encode (HTTP 500)', 'recusada (HTTP 500)'], 12, 'mono', 600, sx(WORST) - x0r - 8), x0r, yy + bh / 2 + 0.5, { size: 12, font: 'mono', weight: 600, color: RED(C), base: 'middle' });
            return;
          }
          const xe = sx(total);
          V.box(g, bx0, yy, Math.max(3, xe - bx0), bh, { r: 5, fill: V.alpha(M.col, 0.5), stroke: M.col, lw: 1 });
          const s = nf(total), sw = tw(g, s, 12, 'mono', 600);
          let xs = xe + 6;
          refX.forEach((x) => { if (xs <= x + 3 && xs + sw >= x - 3) xs = x + 5; }); // o rótulo não cruza as linhas de referência
          if (xs + sw + 2 < bx1) V.text(g, s, xs, yy + bh / 2 + 0.5, { size: 12, font: 'mono', weight: 600, color: C.ink, base: 'middle' });
          else V.text(g, s, xe - 6, yy + bh / 2 + 0.5, { size: 12, font: 'mono', weight: 600, color: C.ink, base: 'middle', align: 'right' });
        });
        y = yB + 36;
        const vT = lanes[0].r.n * n, tT = lanes[1].r.n * n;
        const note = n <= 8 ? `Até 8 imagens as lanes coincidem: 16.384 / ${n} ≥ 2.048, e cada imagem leva o teto cheio.`
          : n > 48 ? `A lane vLLM recusa mais de 48 imagens; a TensorFold aceita até 50, cada uma com ${nf(lanes[1].r.n)} tokens.`
            : `A lane vLLM codifica ${dec(vT / tT, 1)}× mais tokens; na TensorFold cada imagem cai para ${nf(lanes[1].r.n)} tokens (grade ${lanes[1].r.gw}×${lanes[1].r.gh}), e o total não passa de 16.384.`;
        y = wrap(g, note, 20, y, w - 40, 17, { size: 12.5, color: C.ink });
        FH.set(y + 4);
      }, { static: true });
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => setTimeout(redraw, 0));
    })();
  },
});

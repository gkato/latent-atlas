LA.chapter({
  id: 'prefix-caching',
  part: 6,
  order: 33,
  title: 'Prefix caching: reaproveitando o que já foi calculado',
  navTitle: 'Prefix caching',
  subtitle: 'Hash encadeado de blocos, contagem de referência e LRU; por que modelos híbridos só acertam em checkpoints; rascunhadores, retenção, no-store e LMCache; o TensorFold, que troca blocos por estados guardados — e como medir sem se enganar.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['prefix caching', 'APC', 'hash encadeado', 'LRU', 'modelos híbridos', 'retenção', 'LMCache', 'TTFT', 'TensorFold', 'estados de prompt', 'replay'],
  goals: [
    'Explicar por que a <b>máscara causal</b> torna o KV de um prefixo reaproveitável e como o <b>hash encadeado</b> de blocos o identifica',
    'Seguir o ciclo de vida de um bloco no <b>BlockPool</b>: consulta, toque, alocação, evicção LRU e contagem de referência',
    'Entender por que modelos híbridos só acertam em <b>checkpoints alinhados</b> (3.584 no GLM, ~1.600 na Qwen) e o acerto é o <b>mínimo entre grupos</b>',
    'Configurar e medir o prefix caching de uma receita: rascunhadores, retenção, no-store, salt, reset, LMCache e pares frio/quente',
    'Comparar o APC do vLLM com os <b>estados de prompt guardados</b> do TensorFold: grade de 64, prefixo compartilhado, replay com a linha de logits, o teto de 32 estados e, depois da v1.3.2, a camada de NVMe',
  ],
  prereqs: ['scheduler-batching', 'atencao-linear'],
  summary: [
    'Num decoder, o KV do token <i>i</i> só depende dos tokens 0…<i>i</i> (máscara causal): dois prompts com o mesmo começo têm o mesmo KV nesse trecho, e o prefill dele pode ser pulado (como os logits não são guardados, o acerto é no máximo <i>n</i>−1). O vLLM identifica prefixos por <b>hash encadeado</b> de blocos cheios (hash do pai + tokens + extras como <code>cache_salt</code>): uma diferença na posição <i>p</i> invalida todos os blocos seguintes — o que é estável vai primeiro no prompt (caso #144 do DeepSeek: 0 → 18/18 blocos).',
    'O cache é o que sobra do pool: blocos livres com hash ficam na cauda da fila (LRU) até alguém precisar do espaço (evicção preguiçosa); <code>ref_cnt</code> protege blocos em uso; a taxa sai de Δ<code>vllm:prefix_cache_hits</code>/Δ<code>queries</code> e <code>cached_tokens</code> mostra por pedido.',
    'Em modelos híbridos o estado recorrente só existe em checkpoints (<code>mamba_cache_mode=align</code>), o block size é forçado para cima (GLM 3.584; Qwen ~1.600/1.664) e o acerto é o <b>mínimo entre grupos</b> — nunca faça um grupo mentir (DS #26 v1 → KV nulo e “salada”).',
    'Rascunhadores precisam de contexto: EAGLE block drop (Qwen: <code>disable_eagle_block_drop</code>, 3,25 → 0,63 s), coordenador GLM corrigido com replay da janela de 2.048, e o teto <i>n</i>−1−128 do DSpark (12,1 → 86,5 tok/s upstream).',
    'Retenção esparsa troca capacidade por recomputação (GLM: 38 IDs por segmento denso; 0/0 guarda 4×210K mas uma edição a 90% vai de 14,70 s a 112,49 s). Com pool pequeno, nem a grade densa salva edições no começo: os checkpoints antigos são despejados durante o próprio prime. DS usa 4.096 para as caudas SWA.',
    'Controle por pedido: <code>cache_salt</code> isola mas grava; <code>skip_reading</code> não lê; <code>skip_writing_prefix_cache</code> (GLM) não grava e devolve blocos à frente; reset e LMCache são ferramentas com riscos explícitos de segurança e de travamento.',
    'Na versão TensorFold da receita GLM não há APC: o motor guarda poucos <b>estados inteiros</b> (grade de 64, fim do system, ponto de divergência e fim do prompt com a <b>linha de logits</b>) e os retoma com os mesmos bits de um prefill do zero. Prompt idêntico de 64k: 34 s → menos de 0,07 s; system de 7,9k: 4,24 → 0,13 s. O teto é de 32 estados (~45 MiB cada pela conta da v1.3.2; ~91 MiB medidos na v1.10), e <code>TF_GLM_MULTI_LONE=1</code> derrubava o cache para 0% (98,1% com 0). Depois da v1.3.2, os estados vão também para o NVMe em blocos de 2.048 tokens: um estado de 40 mil tokens volta em 147 ms, e no GLM-5.3 em três Sparks um prompt de 94 mil volta em 3,0 s contra 232 s de prefill.',
  ],
  quiz: [
    { q: 'Por que o KV de um prefixo comum pode ser reaproveitado entre duas requisições diferentes?', o: ['Porque o vLLM comprime o KV e ele fica igual', 'Porque, com a máscara causal, o K e o V do token i só dependem dos tokens 0…i', 'Porque os pesos são congelados na inferência', 'Porque a atenção é bidirecional'], a: 1, e: 'A máscara causal garante que cada linha de K/V depende só do prefixo até ela. Pesos congelados são necessários, mas não suficientes: num encoder bidirecional (tipo BERT) o KV do começo dependeria do fim, e nada seria reaproveitável.' },
    { q: 'Dois prompts de 4.000 tokens diferem só no token 10. Com blocos de 256 tokens, quantos blocos são reaproveitados?', o: ['15 (todos menos o primeiro)', '0', '1', 'Depende do algoritmo de hash'], a: 1, e: 'O token 10 está no bloco 0. Como o hash de cada bloco encadeia o do pai, a diferença se propaga a todos os blocos seguintes: nenhum casa. É o mecanismo do caso #144 (diretiva de esforço logo após o BOS).' },
    { q: 'No GLM, um follow-up de 8.004 tokens reaproveita 7.168 tokens e não ~8.000. Por quê?', o: ['Porque o scheduler só processa 7.168 tokens por passo', 'Porque o estado recorrente KDA só é salvo em checkpoints alinhados a 3.584 tokens, e 7.168 = 2 páginas', 'Porque o drafter descarta o último bloco', 'Porque o cache estava cheio'], a: 1, e: 'Em modo align o estado das camadas KDA só existe em fronteiras de bloco; o bloco foi forçado a 3.584 tokens para caber o estado na página. O acerto é o mínimo entre grupos e para no último checkpoint: 2 × 3.584 = 7.168.' },
    { q: 'Qual a diferença entre <code>cache_salt</code> e <code>skip_writing_prefix_cache</code>?', o: ['Nenhuma: os dois desligam o cache', 'O salt isola o namespace mas continua gravando; o no-store deixa ler mas não grava hashes e devolve os blocos à frente da fila', 'O salt impede a leitura; o no-store impede a leitura e a escrita', 'O no-store só funciona em streaming'], a: 1, e: 'O salt entra no hash do bloco 0 e cria um namespace, mas os blocos ainda recebem hash e vão para a cauda LRU. O overlay no-store do GLM não insere hashes: os blocos voltam à frente (LIFO) e não empurram a conversa do dono para o despejo.' },
    { q: 'Por que o hotfix DSpark SWA prefix limita o acerto a n − 1 − 128?', o: ['Para economizar memória de KV', 'Porque a janela de 128 tokens do rascunhador é alimentada pelos estados ocultos que o alvo recalcula; sem recalcular 128 tokens, ela fica vazia e as respostas degeneram', 'Porque o block size do DeepSeek é 128', 'Para evitar colisões de hash'], a: 1, e: 'Num acerto o alvo pula o prefixo e só o sufixo alimenta o DSpark. Em prompts idênticos reenviados o sufixo é curto demais: a janela não enche e a saída trunca. Recalcular a janela (128 tokens, mais o arredondamento ao bloco de 256: 129 a 384 tokens) custa pouco e levou o json60 repetido de 12,1 a 86,5 tok/s (upstream).' },
    { q: 'Para qual workload a retenção esparsa 0/0 do GLM faz sentido?', o: ['Edições frequentes no meio de conversas longas', 'Vários históricos longos e independentes que só crescem por append', 'Benchmarks de prefill a frio', 'Qualquer workload: é sempre melhor'], a: 1, e: 'Com checkpoints só nas fronteiras cabe muito mais histórico (4 históricos de 210K), mas uma edição ou ramificação tardia não tem checkpoint para onde voltar: a 90% de 128K o reuso caiu de 111.104 tokens para 0 (14,70 s → 112,49 s).' },
    { q: 'Na versão TensorFold da receita GLM, um prompt idêntico de 64 mil tokens reenviado volta em menos de 0,07 s. O que torna isso possível?', o: ['Um hash encadeado mais rápido que o do vLLM', 'O estado KDA guardado a cada página de 3.584 tokens', 'O estado do fim do prompt guardado junto com a linha de logits da cabeça, da qual o primeiro token é sorteado com a mesma chave', 'O KV cache gravado em disco e relido'], a: 2, e: 'Um estado sozinho guarda caches e estados KDA, mas não diz qual é o primeiro token: para isso é preciso a linha de logits da última posição, que o APC não guarda (daí o teto n−1). O replay guarda essa linha com o estado do fim do prompt (quando o fim cai na grade de 64), e a amostragem chaveada tira dela o mesmo token que um prefill do zero daria. O TensorFold não usa hash de blocos nem páginas de 3.584 para isso, e na v1.3.2 nada ia para disco (a camada de NVMe veio depois e não entra neste replay).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const nf = (x) => Math.round(x).toLocaleString('pt-BR');
    const sf = (x, d = 1) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    let fcache = {};
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { fcache = {}; });
    const font = (s, f, w) => fcache[s + f + w] || (fcache[s + f + w] = V.font(s, f, w));
    const tw = (g, s, size = 12, f = 'sans', w = 400) => { g.font = font(size, f, w); return g.measureText(String(s)).width; };
    const fnv = (str) => { let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; };
    const hex6 = (h) => (h >>> 0).toString(16).padStart(8, '0').slice(0, 6);
    const cv = (n) => `var(--viz-${n})`;
    const RED = (C) => C.red || C.c;
    function pill(g, x, y, s, col, o = {}) {
      const size = o.size || 11, f = o.font || 'mono', wt = o.weight || 600;
      const wd = tw(g, s, size, f, wt) + 12, ht = size + 8;
      const xx = o.align === 'center' ? x - wd / 2 : o.align === 'right' ? x - wd : x;
      V.box(g, xx, y - ht / 2, wd, ht, { r: ht / 2, fill: V.alpha(col, o.a == null ? 0.18 : o.a), stroke: V.alpha(col, 0.75), lw: 1 });
      V.text(g, s, xx + wd / 2, y + 0.5, { size, font: f, weight: wt, color: o.color || V.C().ink, align: 'center', base: 'middle' });
      return wd;
    }
    function hatch(g, x, y, w, h, col, step = 7) {
      if (w <= 0 || h <= 0) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    const wrapLines = (g, s, maxW, size = 12, f = 'sans', wt = 400) => {
      g.font = font(size, f, wt);
      const words = String(s).split(' '); const out = []; let cur = '';
      words.forEach((wd) => { const test = cur ? cur + ' ' + wd : wd; if (cur && g.measureText(test).width > maxW) { out.push(cur); cur = wd; } else cur = test; });
      if (cur) out.push(cur);
      return out;
    };
    /** texto com quebra de linha; retorna a altura usada */
    function wrap(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.38);
      const ls = wrapLines(g, s, maxW, size, o.font || 'sans', o.weight || 400);
      ls.forEach((ln, i) => V.text(g, ln, x, y + i * lh, Object.assign({}, o, { size })));
      return ls.length * lh;
    }
    function fitH(H, need) {
      const st = H.canvas.parentElement;
      const cur = parseFloat(st.style.height) || st.getBoundingClientRect().height;
      need = Math.round(need);
      if (Math.abs(cur - need) > 4) st.style.height = need + 'px';
    }

    /* =====================================================================
       Figura 1 — turno a turno
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-turnos');
      const S = { apc: true, sys: 12000, turns: 6, grain: 16, resp: false };
      const U = 400, A = 800, R = 1000, T0 = 0.3;
      const model = () => {
        const out = []; let prev = 0;
        for (let k = 0; k < S.turns; k++) {
          const hist = k * (U + A);
          const n = S.sys + hist + U;
          const hit = S.apc && k > 0 ? Math.floor(Math.min(prev, n - 1) / S.grain) * S.grain : 0;
          out.push({ n, hist, hit, ttft: T0 + (n - hit) / R });
          prev = n + (S.resp ? A : 0); // com template estável, a resposta anterior (blocos de decode) também tem hash
        }
        return out;
      };
      let H = null, ra = null, rb = null;
      const upd = () => {
        const m = model();
        if (ra) ra.set(nf(m.reduce((s, r) => s + r.n - r.hit, 0)));
        if (rb) rb.set(sf(m.reduce((s, r) => s + r.ttft, 0), 1) + ' s');
        if (H) H.S.t0 = H.t;
      };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'prefix caching ligado', value: true, onChange: (v) => { S.apc = v; upd(); } });
      UI.toggle(ctl, { label: 'resposta anterior também acerta', value: false, onChange: (v) => { S.resp = v; upd(); } });
      UI.slider(ctl, { label: 'System + tools', min: 2000, max: 40000, step: 1000, value: S.sys, fmt: (v) => nf(v) + ' tok', onInput: (v) => { S.sys = v; upd(); } });
      UI.slider(ctl, { label: 'Turnos', min: 2, max: 8, step: 1, value: S.turns, onInput: (v) => { S.turns = v; upd(); } });
      UI.seg(ctl, { label: 'Grão do acerto', options: [{ v: 16, label: '16' }, { v: 256, label: '256 (DS)' }, { v: 3584, label: '3.584 (GLM)' }], value: 16, onChange: (v) => { S.grain = v; upd(); } });
      ra = UI.readout(ctl, 'tokens calculados', '—');
      rb = UI.readout(ctl, 'TTFT somado', '—');
      UI.legend(ctl, [[cv('b'), 'system + tools'], [cv('a'), 'histórico'], [cv('c'), 'mensagem nova'], [cv('d'), 'cursor do prefill']]);
      H = V.fig(fig, (g, w, h, t, dt, St) => {
        const C = V.C(); const m = model(); const T = m.length;
        const narrow = w < 560;
        const lx = 12, labW = narrow ? 30 : 62, ttW = narrow ? 52 : 118;
        const x0 = lx + labW, bw = Math.max(60, w - x0 - ttW - 14);
        const top = 32, rowH = Math.min(44, (h - top - (narrow ? 40 : 30)) / T), barH = Math.max(12, Math.min(24, rowH - 12));
        const maxN = m[T - 1].n, maxTT = T0 + maxN / R;
        const per = 1.8, tt = Math.max(0, t - (St.t0 || 0));
        const cyc = tt % (per * (T + 0.7)); const cur = Math.floor(cyc / per); const ph = (cyc % per) / per;
        const X = (tok) => x0 + (tok / maxN) * bw;
        V.text(g, narrow ? 'prompt de cada turno' : 'prompt enviado a cada turno (cresce com o histórico)', x0, 18, { size: 11.5, color: C.muted });
        V.text(g, 'TTFT', w - 12, 18, { size: 11.5, font: 'mono', color: C.muted, align: 'right' });
        m.forEach((r, k) => {
          const y = top + k * rowH + (rowH - barH) / 2;
          const active = k === cur, past = !active, fut = false;
          V.text(g, narrow ? 'T' + (k + 1) : 'turno ' + (k + 1), lx, y + barH / 2 + 4, { size: 12, font: 'mono', color: active ? C.ink : C.muted });
          const fillTo = past ? r.n : active ? r.hit + (r.n - r.hit) * V.ease.out(Math.min(1, ph * 1.2)) : r.hit;
          const segs = [[0, S.sys, C.b], [S.sys, S.sys + r.hist, C.a], [S.sys + r.hist, r.n, C.c]];
          segs.forEach(([a, b, col]) => {
            if (b <= a) return;
            g.fillStyle = V.alpha(col, fut ? 0.07 : 0.12); g.fillRect(X(a), y, X(b) - X(a), barH);
            const ca = Math.max(a, r.hit), cb = Math.min(b, fillTo);
            if (!fut && cb > ca) { g.fillStyle = V.alpha(col, 0.85); g.fillRect(X(ca), y, X(cb) - X(ca), barH); }
            const hb = Math.min(b, r.hit);
            if (!fut && hb > a) { g.fillStyle = V.alpha(col, 0.32); g.fillRect(X(a), y, X(hb) - X(a), barH); }
          });
          if (!fut && r.hit > 0) {
            hatch(g, X(0), y, X(r.hit) - X(0), barH, V.alpha(C.ink, 0.28), 6);
            if (active && ph < 0.22) { g.fillStyle = V.alpha(C.a, 0.5 * (1 - ph / 0.22)); g.fillRect(X(0), y, X(r.hit) - X(0), barH); }
          }
          g.strokeStyle = V.alpha(C.muted, fut ? 0.25 : 0.6); g.lineWidth = 1; g.strokeRect(X(0) + 0.5, y + 0.5, X(r.n) - X(0), barH);
          if (!fut && r.hit > 0 && X(r.hit) - X(0) > 66) V.text(g, '♻ ' + nf(r.hit), X(0) + 5, y + barH / 2 + 4, { size: 11, font: 'mono', weight: 600, color: C.ink });
          if (active) { const cx = X(fillTo); V.glow(g, cx, y + barH / 2, 16, C.d); V.line(g, cx, y - 3, cx, y + barH + 3, { color: C.d, lw: 2.5 }); }
          if (!fut) {
            const tx = w - ttW;
            if (!narrow) { const mw = (ttW - 60) * Math.min(1, r.ttft / maxTT); V.box(g, tx, y + barH / 2 - 4, Math.max(3, mw), 8, { r: 4, fill: V.alpha(S.apc ? C.a : C.c, 0.75) }); }
            V.text(g, sf(r.ttft, 1) + ' s', w - 12, y + barH / 2 + 4, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'right' });
          }
        });
        const foot = `ilustrativo: +${U} tokens de pergunta e +${A} de resposta por turno · prefill ≈ 1.000 tok/s · ${S.resp ? 'resposta anterior reenviada idêntica (acerta)' : 'só o prompt anterior acerta'}`;
        const fl = wrapLines(g, foot, w - 24, 11);
        fl.forEach((ln, i) => V.text(g, ln, 12, h - 8 - (fl.length - 1 - i) * 14, { size: 11, color: C.muted }));
      });
      upd();
    }

    /* =====================================================================
       Figura 2 — causalidade: linhas de K idênticas
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-causal');
      const TA = ['Você', 'é', 'um', 'agente', 'útil', '.', 'Resuma', 'o'];
      const TALT = ['Oi', 'diga', 'algo', 'novo', 'já', ':', 'Traduza', 'isto'];
      const D = 6; const r0 = V.rng(7);
      const mk = () => Array.from({ length: D }, () => Array.from({ length: D }, () => r0.normal() * 0.55));
      const W1 = mk(), WK = mk();
      const embC = {};
      const emb = (tok) => { if (!embC[tok]) { const r = V.rng(fnv(tok) || 3); embC[tok] = Array.from({ length: D }, () => r.normal()); } return embC[tok]; };
      const layerK = (toks, causal) => {
        const E = toks.map(emb), n = E.length;
        return E.map((ei, i) => {
          const js = []; for (let j = 0; j < n; j++) if (!causal || j <= i) js.push(j);
          const sc = js.map((j) => E[j].reduce((s, v, d) => s + v * ei[d], 0) / Math.sqrt(D));
          const al = V.softmax(sc);
          const z = new Array(D).fill(0);
          js.forEach((j, q) => { for (let d = 0; d < D; d++) z[d] += al[q] * E[j][d]; });
          const hh = W1.map((row, d) => Math.tanh(row.reduce((s, v, e) => s + v * z[e], 0) + 0.5 * ei[d]));
          return WK.map((row) => row.reduce((s, v, e) => s + v * hh[e], 0));
        });
      };
      const S = { same: 5, causal: true };
      let H = null, KA, KB, TB, eq, mx, ro = null;
      const recompute = () => {
        TB = TA.map((x, i) => (i < S.same ? x : TALT[i]));
        KA = layerK(TA, S.causal); KB = layerK(TB, S.causal);
        eq = KA.map((row, i) => row.every((v, d) => Math.abs(v - KB[i][d]) < 1e-12));
        mx = 1e-9; KA.concat(KB).forEach((row) => row.forEach((v) => { if (Math.abs(v) > mx) mx = Math.abs(v); }));
        if (ro) ro.set(eq.filter(Boolean).length + ' / 8');
        if (H) H.redraw();
      };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Prefixo igual', min: 1, max: 7, step: 1, value: S.same, onInput: (v) => { S.same = v; recompute(); } });
      UI.toggle(ctl, { label: 'máscara causal (decoder)', value: true, onChange: (v) => { S.causal = v; recompute(); } });
      ro = UI.readout(ctl, 'linhas de K idênticas', '—');
      recompute();
      H = V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C(); const wide = w >= 700; const n = 8;
        const hi = Math.floor(t / 1.1) % n;
        const cx0 = wide ? 64 : 30, chipGap = 4;
        const chipW = Math.min(80, (w - cx0 - 10 - chipGap * 7) / 8), chipH = 24;
        const rowsY = [12, 44];
        [['A', TA], ['B', TB]].forEach(([nm, toks], r) => {
          V.text(g, nm, cx0 - 10, rowsY[r] + chipH / 2 + 5, { size: 13, font: 'mono', weight: 700, color: C.ink, align: 'right' });
          toks.forEach((tok, j) => {
            const x = cx0 + j * (chipW + chipGap), y = rowsY[r];
            const col = j >= S.same ? C.c : C.a;
            const lit = S.causal ? j <= hi : true;
            V.box(g, x, y, chipW, chipH, { r: 7, fill: V.alpha(col, lit ? 0.3 : 0.12), stroke: V.alpha(col, lit ? 0.95 : 0.45), lw: lit ? 1.6 : 1 });
            V.text(g, chipW < 42 ? tok.slice(0, 4) : tok, x + chipW / 2, y + chipH / 2 + 0.5, { size: chipW < 40 ? 11 : 12, align: 'center', base: 'middle', color: C.ink, maxW: chipW - 4 });
          });
        });
        const my = 104;
        const rowLabW = wide ? 58 : 40, eqW = 26;
        const maskW = wide ? 8 * 16 - 2 : 0, maskGap = wide ? 36 : 0;
        const avail = w - 12 - maskW - maskGap - rowLabW - eqW - 12;
        const cell = Math.max(12, Math.min(34, Math.floor(avail / 12) - 2));
        const cellH = Math.max(14, Math.min(cell, wide ? 30 : 24));
        const matW = 6 * cell + 5 * 2;
        const totalW = maskW + maskGap + rowLabW + matW + eqW + matW;
        let x = Math.max(6, (w - totalW) / 2);
        if (wide) {
          V.text(g, 'máscara (quem a linha vê)', x + maskW / 2, my - 12, { size: 11, color: C.muted, align: 'center' });
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const on = S.causal ? j <= i : true;
            const yy = my + i * (cellH + 2) + (cellH - 14) / 2;
            g.fillStyle = on ? V.alpha(C.b, i === hi ? 0.9 : 0.35) : V.alpha(C.muted, 0.12);
            V.rr(g, x + j * 16, yy, 14, 14, 3); g.fill();
          }
          x += maskW + maskGap;
        }
        const values = cell >= 28 && cellH >= 18;
        x += rowLabW;
        V.matrix(g, { x, y: my, data: KA, cell, cellH, gap: 2, cmap: 'div', min: -mx, max: mx, values, digits: 1, hiRow: hi, hiColor: C.d, rowLabels: TA.map((s) => s.slice(0, wide ? 7 : 5)) });
        V.text(g, 'K de A', x + matW / 2, my - 12, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
        const ex = x + matW + eqW / 2;
        eq.forEach((e, i) => { const cy = my + i * (cellH + 2) + cellH / 2; V.text(g, e ? '=' : '≠', ex, cy + 1, { size: 16, weight: 700, align: 'center', base: 'middle', color: e ? C.a : C.c }); });
        const xb = x + matW + eqW;
        V.matrix(g, { x: xb, y: my, data: KB, cell, cellH, gap: 2, cmap: 'div', min: -mx, max: mx, values, digits: 1, hiRow: hi, hiColor: C.d });
        V.text(g, 'K de B', xb + matW / 2, my - 12, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
        const by = my + 8 * (cellH + 2) + 16;
        let yy = by + wrap(g, `linha ${hi} (“${TB[hi]}”) consulta ${S.causal ? (hi === 0 ? 'só o token 0' : 'os tokens 0…' + hi) : 'os 8 tokens, inclusive os futuros'}`, w / 2, by, w - 16, { size: 11.5, align: 'center', color: C.muted }) + 4;
        yy += wrap(g, S.causal ? `linhas 0…${S.same - 1} idênticas: o KV do prefixo pode ser reaproveitado` : 'sem máscara causal a diferença do fim contamina todas as linhas', w / 2, yy, w - 16, { size: 12, align: 'center', color: S.causal ? C.a : C.c, weight: 600 });
        fitH(HH, Math.max(330, yy + 4));
      });
    }

    /* =====================================================================
       Figura 3 — o trem de hashes
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-hash');
      const BOS = ['<s>'];
      const SYS = ['Você', 'é', 'um', 'agente', 'de', 'código', '.', 'Siga'];
      const TOOLS = ['tools', ':', 'ls', 'grep', 'cat', 'sed', 'git', 'diff'];
      const Q1 = ['Q:', 'ache', 'o', 'bug'];
      const MORE = ['A:', 'feito', '.', 'Q:', 'e', 'os', 'testes', '?'];
      const DH = ['<esf', 'alto>'], DM = ['<esf', 'máx', '!>'];
      const cat = (arr, c) => arr.map((x) => ({ x, c }));
      const S = { sc: 'cont', B: 4, edit: 14, t0: 0 };
      const build = () => {
        const base = [...cat(BOS, 'bos'), ...cat(SYS, 'sys'), ...cat(TOOLS, 'tool'), ...cat(Q1, 'user')];
        const cont = base.concat(cat(MORE, 'new'));
        if (S.sc === 'cont') return { prev: base, cur: cont, sp: '', sc: '' };
        if (S.sc === 'edit') { const cur = cont.map((o, i) => (i === S.edit ? { x: '✎' + o.x, c: 'edit' } : o)); return { prev: base, cur, sp: '', sc: '' }; }
        if (S.sc === 'salt') return { prev: base, cur: cont, sp: '', sc: 'cliente-B' };
        if (S.sc === 'effort') return { prev: [...cat(BOS, 'bos'), ...cat(DH, 'dir'), ...cat(SYS, 'sys'), ...cat(TOOLS, 'tool'), ...cat(Q1, 'user')], cur: [...cat(BOS, 'bos'), ...cat(DM, 'dir'), ...cat(SYS, 'sys'), ...cat(TOOLS, 'tool'), ...cat(Q1, 'user')], sp: '', sc: '' };
        return { prev: [...cat(BOS, 'bos'), ...cat(SYS, 'sys'), ...cat(TOOLS, 'tool'), ...cat(DH, 'dir'), ...cat(Q1, 'user')], cur: [...cat(BOS, 'bos'), ...cat(SYS, 'sys'), ...cat(TOOLS, 'tool'), ...cat(DM, 'dir'), ...cat(Q1, 'user')], sp: '', sc: '' };
      };
      // como no vLLM: o cache_salt é chave extra só do bloco 0; a locomotiva (NONE_HASH) é a mesma para todos
      const chain = (toks, salt) => {
        const root0 = fnv('NONE_HASH'); let hp = root0; const hs = [];
        for (let j = 0; (j + 1) * S.B <= toks.length; j++) {
          const extra = j === 0 && salt ? '|salt=' + salt : '';
          hp = fnv(hp.toString(16) + '|' + toks.slice(j * S.B, (j + 1) * S.B).map((o) => o.x).join('␟') + extra); hs.push(hp);
        }
        return { root: root0, hs };
      };
      let H = null, ro = null, segSc = null, clickMap = [];
      const restart = () => { if (H) S.t0 = H.t; if (ro) { const d = compute(); ro.set(d.nHit + ' de ' + d.cc.hs.length); } };
      const compute = () => {
        const b = build(); const pc = chain(b.prev, b.sp), cc = chain(b.cur, b.sc);
        const set = new Set(pc.hs); const st = cc.hs.map((x) => set.has(x));
        let nHit = 0; while (nHit < st.length && st[nHit]) nHit++;
        return { b, pc, cc, st, nHit };
      };
      const ctl = UI.controls(fig);
      segSc = UI.seg(ctl, { label: 'Cenário', options: [{ v: 'cont', label: 'continuação' }, { v: 'edit', label: 'editar 1 token' }, { v: 'effort', label: 'esforço após BOS' }, { v: 'effortFix', label: 'esforço após system' }, { v: 'salt', label: 'cache_salt diferente' }], value: 'cont', onChange: (v) => { S.sc = v; restart(); } });
      UI.seg(ctl, { label: 'Bloco', options: [{ v: 4, label: '4 tokens' }, { v: 6, label: '6 tokens' }], value: 4, onChange: (v) => { S.B = v; restart(); } });
      const sl = UI.slider(ctl, { label: 'Posição editada', min: 0, max: 28, step: 1, value: S.edit, onInput: (v) => { S.edit = v; if (S.sc !== 'edit') segSc.set('edit'); else restart(); } });
      ro = UI.readout(ctl, 'blocos reaproveitados', '—');
      UI.legend(ctl, [[cv('muted'), 'BOS'], [cv('b'), 'system'], [cv('d'), 'ferramentas'], [cv('c'), 'pergunta'], [cv('e'), 'novo'], [cv('f'), 'diretiva de esforço'], ['var(--red)', 'token editado']]);
      const catCol = (C, c) => ({ bos: C.muted, sys: C.b, tool: C.d, user: C.c, new: C.e, dir: C.f, edit: RED(C) })[c] || C.muted;
      H = V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C(); const d = compute(); const narrow = w < 560;
        const B = S.B;
        const chip = Math.max(14, Math.min(36, Math.floor((w - 70) / (B * (narrow ? 13 : 5.4)))));
        const locoW = narrow ? 38 : 50, gapW = narrow ? 12 : 16;
        const wagW = B * chip + (B - 1) * 2 + 12;
        const perRow = Math.max(1, Math.floor((w - 20 - locoW - 8 + gapW) / (wagW + gapW)));
        const rowH = chip + 46;
        const trains = [
          { lab: 'em cache: requisição anterior', toks: d.b.prev, ch: d.pc, salt: d.b.sp, cur: false },
          { lab: 'nova requisição', toks: d.b.cur, ch: d.cc, salt: d.b.sc, cur: true },
        ];
        const nWag = (toks) => Math.ceil(toks.length / B);
        const rowsOf = (toks) => Math.ceil(nWag(toks) / perRow);
        const need = 16 + trains.reduce((s, tr) => s + 24 + rowsOf(tr.toks) * rowH, 0) + 44;
        fitH(HH, Math.max(360, need));
        const nFull = d.cc.hs.length, stepDur = 0.5, hold = 2.6;
        const tt = Math.max(0, t - S.t0), per = nFull * stepDur + hold;
        const reveal = Math.min(nFull, Math.floor((tt % per) / stepDur));
        const revFrac = ((tt % per) % stepDur) / stepDur;
        clickMap = [];
        let y = 14;
        trains.forEach((tr) => {
          V.text(g, tr.lab, 12, y + 10, { size: 11.5, color: C.muted, font: 'mono' });
          y += 22;
          const x0 = 10;
          const ly = y + 6;
          V.box(g, x0, ly, locoW, chip + 10, { r: 8, fill: V.alpha(C.muted, 0.2), stroke: V.alpha(C.muted, 0.8), lw: 1.2 });
          V.text(g, '∅', x0 + locoW / 2, ly + (chip + 10) / 2 + 1, { size: narrow ? 10.5 : 11.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink, maxW: locoW - 4 });
          V.text(g, '#' + hex6(tr.ch.root).slice(0, narrow ? 4 : 6), x0 + locoW / 2, ly + chip + 24, { size: 11, font: 'mono', align: 'center', color: C.muted });
          const nw = nWag(tr.toks);
          for (let j = 0; j < nw; j++) {
            const row = Math.floor(j / perRow), col = j % perRow;
            const wx = x0 + locoW + 8 + col * (wagW + gapW), wy = y + row * rowH + 6;
            const full = (j + 1) * B <= tr.toks.length;
            const revealed = !tr.cur || j < reveal;
            let stc = C.b;
            if (tr.cur && full && revealed) stc = d.st[j] ? C.a : C.c;
            if (!full) stc = C.muted;
            // engate
            const pxEnd = col === 0 ? (row === 0 ? x0 + locoW : null) : wx - gapW;
            if (pxEnd != null) {
              V.line(g, pxEnd, wy + (chip + 10) / 2, wx, wy + (chip + 10) / 2, { color: V.alpha(C.muted, 0.7), lw: 2 });
              if (tr.cur && j === reveal && full) { const u = V.ease.inOut(revFrac); V.glow(g, V.lerp(pxEnd, wx, u), wy + (chip + 10) / 2, 10, C.d); }
            } else {
              V.text(g, '↪', wx - 10, wy + (chip + 10) / 2 + 4, { size: 12, color: C.muted, align: 'center' });
            }
            V.box(g, wx, wy, wagW, chip + 10, { r: 8, fill: V.alpha(stc, full ? 0.14 : 0.05), stroke: V.alpha(stc, full ? 0.85 : 0.5), lw: full ? 1.4 : 1 });
            if (!full) { g.setLineDash([4, 4]); V.rr(g, wx, wy, wagW, chip + 10, 8); g.strokeStyle = V.alpha(C.muted, 0.8); g.stroke(); g.setLineDash([]); }
            for (let q = 0; q < B; q++) {
              const ti = j * B + q; if (ti >= tr.toks.length) break;
              const o = tr.toks[ti];
              const cx = wx + 6 + q * (chip + 2), cy = wy + 5;
              const cc = catCol(C, o.c);
              V.box(g, cx, cy, chip, chip, { r: 4, fill: V.alpha(cc, 0.5), stroke: o.c === 'edit' ? RED(C) : null, lw: 2 });
              if (chip >= 26) V.text(g, o.x, cx + chip / 2, cy + chip / 2 + 0.5, { size: 10.5, align: 'center', base: 'middle', color: C.ink, maxW: chip - 2 });
              if (tr.cur) clickMap.push({ x: cx, y: cy, w: chip, h: chip, i: ti });
            }
            const hy = wy + chip + 24;
            if (full) {
              const hsh = tr.ch.hs[j];
              if (revealed) V.text(g, '#' + hex6(hsh), wx + wagW / 2, hy, { size: 11, font: 'mono', weight: 600, align: 'center', color: tr.cur ? (d.st[j] ? C.a : C.c) : C.ink });
              else V.text(g, '# · · ·', wx + wagW / 2, hy, { size: 11, font: 'mono', align: 'center', color: C.muted });
              if (tr.cur && revealed && wagW >= 60) pill(g, wx + wagW - 4, wy - 2, d.st[j] ? '✓' : '✗', d.st[j] ? C.a : C.c, { align: 'right', size: 10.5, a: 0.35 });
              if (j === 0 && tr.salt) pill(g, wx - 8, wy - 2, '+salt', C.f, { size: 10, a: 0.45 });
            } else V.text(g, narrow ? 'parcial' : 'parcial: sem hash', wx + wagW / 2, hy, { size: 11, align: 'center', color: C.muted });
          }
          y += rowsOf(tr.toks) * rowH;
        });
        // clique para editar
        const P = HH.pointer;
        if (P.clicked) {
          const hit = clickMap.find((r) => P.x >= r.x && P.x <= r.x + r.w && P.y >= r.y && P.y <= r.y + r.h);
          if (hit) { S.edit = Math.min(28, hit.i); sl.input.value = S.edit; sl.el.querySelector('.val').textContent = String(S.edit); if (S.sc !== 'edit') segSc.set('edit'); else restart(); }
        }
        const nH = d.nHit, nF = d.cc.hs.length;
        const msg = nH === nF ? `todos os ${nF} blocos cheios têm hash conhecido: reaproveitados` : nH === 0 ? (S.sc === 'salt' ? 'o salt é chave extra do bloco 0: ele já diverge, e o dominó leva a corrente inteira' : 'o bloco 0 já diverge: nenhum bloco reaproveitado (efeito dominó)') : `blocos 0…${nH - 1} reaproveitados; a partir do bloco ${nH} todo hash é novo`;
        const ml = wrapLines(g, msg, w - 24, 12, 'sans', 600);
        ml.forEach((ln, i) => V.text(g, ln, w / 2, h - 12 - (ml.length - 1 - i) * 16, { size: 12, weight: 600, align: 'center', color: nH === 0 ? C.c : C.a }));
      });
      restart();
    }

    /* =====================================================================
       Figura 4 — BlockPool ao vivo (stepper)
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-pool');
      const E = (lab, own, ref, hash) => ({ lab, own, ref, hash });
      const empty = () => { const p = {}; for (let i = 1; i <= 10; i++) p[i] = E('', '', 0, null); return p; };
      const cp = (o) => JSON.parse(JSON.stringify(o));
      const states = [];
      const s0 = { pool: empty(), queue: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], table: [], q: 0, hh: 0, req: null, evict: [] };
      states.push(s0);
      const s1 = cp(s0);
      s1.pool[1] = E('S1', 'A', 1, 'h1'); s1.pool[2] = E('S2', 'A', 1, 'h2'); s1.pool[3] = E('a', 'A', 1, 'h3'); s1.pool[4] = E('a·', 'A', 1, null);
      s1.queue = [5, 6, 7, 8, 9, 10]; s1.table = [['h1', 1], ['h2', 2], ['h3', 3]]; s1.q = 13; s1.hh = 0;
      s1.req = { name: 'A', n: 13, blocks: [['S1', 1, 'miss'], ['S2', 2, 'miss'], ['a', 3, 'miss'], ['a·', 4, 'part']] };
      states.push(s1);
      const s2 = cp(s1);
      [1, 2, 3].forEach((i) => { s2.pool[i].ref = 0; }); s2.pool[4] = E('', '', 0, null);
      s2.queue = [4, 5, 6, 7, 8, 9, 10, 3, 2, 1]; s2.req = null; s2.done = 'A';
      states.push(s2);
      const s3 = cp(s2);
      s3.pool[1].ref = 1; s3.pool[2].ref = 1; s3.pool[4] = E('b', 'B', 1, 'h4'); s3.pool[5] = E('b·', 'B', 1, null);
      s3.queue = [6, 7, 8, 9, 10, 3]; s3.table.push(['h4', 4]); s3.q = 26; s3.hh = 8; s3.done = null;
      s3.req = { name: 'B', n: 13, blocks: [['S1', 1, 'hit'], ['S2', 2, 'hit'], ['b', 4, 'miss'], ['b·', 5, 'part']] };
      states.push(s3);
      const s4 = cp(s3);
      ['D1', 'D2', 'D3', 'D4', 'D5'].forEach((l, k) => { s4.pool[6 + k] = E(l, 'C', 1, 'h' + (5 + k)); });
      s4.pool[3] = E('D·', 'C', 1, null);
      s4.queue = []; s4.table = s4.table.filter((e) => e[0] !== 'h3').concat([['h5', 6], ['h6', 7], ['h7', 8], ['h8', 9], ['h9', 10]]);
      s4.q = 47; s4.evict = [3]; s4.evictH = 'h3';
      s4.req = { name: 'C', n: 21, blocks: [['D1', 6, 'miss'], ['D2', 7, 'miss'], ['D3', 8, 'miss'], ['D4', 9, 'miss'], ['D5', 10, 'miss'], ['D·', 3, 'part']] };
      states.push(s4);
      const s5 = cp(s4);
      [1, 2, 4, 6, 7, 8, 9, 10].forEach((i) => { s5.pool[i].ref = 0; }); s5.pool[3] = E('', '', 0, null); s5.pool[5] = E('', '', 0, null);
      s5.queue = [3, 5, 4, 2, 1, 10, 9, 8, 7, 6]; s5.req = null; s5.evict = []; s5.evictH = null; s5.done = 'B e C';
      states.push(s5);
      const s6 = cp(s5);
      s6.pool[1].ref = 1; s6.pool[2].ref = 1; s6.pool[3] = E('a', 'A', 1, 'h3'); s6.pool[5] = E('a·', 'A', 1, null);
      s6.queue = [4, 10, 9, 8, 7, 6]; s6.table = [['h1', 1], ['h2', 2], ['h3', 3], ['h4', 4], ['h5', 6], ['h6', 7], ['h7', 8], ['h8', 9], ['h9', 10]];
      s6.q = 60; s6.hh = 16; s6.done = null;
      s6.req = { name: 'A (de novo)', n: 13, blocks: [['S1', 1, 'hit'], ['S2', 2, 'hit'], ['a', 3, 'miss'], ['a·', 5, 'part']] };
      states.push(s6);
      const steps = [
        { t: 'Pool vazio', d: '10 blocos físicos livres (o bloco 0 é o <b>null block</b>, só um marcador). Nenhum hash na tabela. Neste exemplo um bloco guarda 4 tokens.' },
        { t: 'A chega (frio)', d: 'A tem 13 tokens: S1 S2 (system), <i>a</i> (pergunta) e 1 token solto. A consulta procura H(∅,S1) na tabela: miss, e a varredura para. O motor tira 4 blocos da cabeça da fila, faz o prefill e registra o hash dos <b>3 blocos cheios</b>. O bloco parcial fica sem hash.' },
        { t: 'A termina', d: 'O ref_cnt cai a 0. Na regra da imagem GLM, o bloco sem hash volta para a <b>frente</b> da fila e é reciclado já; os com hash vão para a <b>cauda</b> em ordem reversa (a, S2, S1), para que o mais profundo saia primeiro. Eles continuam na tabela: estão livres, mas em cache.' },
        { t: 'B chega (quente)', d: 'B começa com S1 S2. H(∅,S1) e H(h1,S2) estão na tabela: <b>2 acertos = 8 tokens</b> sem prefill. Os blocos 1 e 2 são “tocados”: ref_cnt 0→1 e saem da fila. Só <i>b</i> e o token solto são calculados, em blocos tirados da cabeça.' },
        { t: 'C chega com o pool apertado', d: 'C é outro assunto: 21 tokens, 6 blocos. Há 5 blocos livres sem hash, e o 6º vem da cabeça da fila: é o bloco 3, que ainda guardava <i>a</i> de A. É a <b>evicção preguiçosa</b>: o hash h3 só sai da tabela agora, quando o espaço é necessário.' },
        { t: 'B e C terminam', d: 'Todos os blocos voltam à fila: os sem hash na frente, os com hash na cauda em ordem reversa, primeiro os de B e depois os de C. S1 e S2 voltaram <b>antes</b> dos blocos de C, então estão no meio da fila, e não perto da cauda. Vão sobreviver só porque A volta logo e consome apenas os 2 blocos sem hash da cabeça.' },
        { t: 'A reenvia a mesma pergunta', d: 'S1 e S2 acertam de novo (8 tokens), mas h3 foi despejado no passo 5 e <i>a</i> precisa ser recalculado, nos blocos 3 e 5 tirados da cabeça. Taxa acumulada: 16 de 60 tokens consultados (27%), baixa porque C era todo novo.' },
      ];
      let lastI = 0, fromI = 0, H = null;
      const st = UI.stepper(fig, steps, { autoplay: 6500, onStep: () => { if (H) H.redraw(); } });
      H = V.fig(fig, (g, w, h, t, dt, St, HH) => {
        const C = V.C();
        if (st.i !== lastI) { fromI = lastI; lastI = st.i; }
        const u = V.ease.inOut(Math.min(1, st.elapsed() / 0.9));
        const S = states[st.i], F = states[fromI];
        const wide = w >= 720;
        const ownCol = (o) => (o === 'A' ? C.b : o === 'B' ? C.f : o === 'C' ? C.e : C.muted);
        const stCol = (s) => (s === 'hit' ? C.a : s === 'miss' ? C.c : C.d);
        // --- layout
        const perRow = wide ? 11 : 6;
        const cellW = Math.min(76, (w - 24 - (perRow - 1) * 6) / perRow), cellH = wide ? 60 : 54;
        const poolRows = Math.ceil(11 / perRow);
        const reqY = 16, poolY = 112;
        const qY = poolY + poolRows * (cellH + 8) + 34;
        const qChip = Math.max(20, Math.min(36, (w - 40) / 10 - 5));
        const tabY = qY + qChip + 40;
        const tabChipW = wide ? 64 : 54;
        const tabPerRow = Math.max(1, Math.floor((w - 24) / (tabChipW + 6)));
        const tabRows = Math.max(1, Math.ceil(9 / tabPerRow));
        fitH(HH, tabY + tabRows * 30 + 16);
        const cellPos = (id) => { const k = id; const r = Math.floor(k / perRow), c = k % perRow; const rowW = Math.min(perRow, 11 - r * perRow) * (cellW + 6) - 6; const x0 = (w - (wide ? rowW : perRow * (cellW + 6) - 6)) / 2; return [x0 + c * (cellW + 6), poolY + r * (cellH + 8)]; };
        // --- requisição ativa
        V.text(g, 'requisição ativa', 12, reqY + 2, { size: 11, font: 'mono', color: C.muted });
        if (S.req) {
          const nm = S.req.name;
          const pw = pill(g, 12, reqY + 26, nm, nm.startsWith('A') ? C.b : nm === 'B' ? C.f : C.e, { size: 12, a: 0.3 });
          const bw = Math.min(58, (w - pw - 40 - (wide ? 200 : 0)) / 6 - 6);
          S.req.blocks.forEach(([lab, id, s], k) => {
            const bx = 24 + pw + k * (bw + 6), by = reqY + 14;
            const col = stCol(s);
            V.box(g, bx, by, bw, 24, { r: 6, fill: V.alpha(col, 0.3), stroke: col, lw: 1.4 });
            V.text(g, lab, bx + bw / 2, by + 12.5, { size: 11.5, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink, maxW: bw - 4 });
            const [px, py] = cellPos(id);
            const a = Math.min(1, u * 1.4);
            g.globalAlpha = a;
            V.arrow(g, bx + bw / 2, by + 26, px + cellW / 2, py - 4, { color: V.alpha(col, 0.8), lw: 1.4, head: 6, curve: 0 });
            g.globalAlpha = 1;
          });
        } else if (S.done) {
          V.text(g, `${S.done} terminou: blocos voltam à fila livre`, 12, reqY + 30, { size: 12.5, color: C.ink, weight: 500 });
        } else {
          V.text(g, 'nenhuma: pool em repouso', 12, reqY + 30, { size: 12.5, color: C.muted });
        }
        // contadores
        const rate = S.q ? Math.round((100 * S.hh) / S.q) : 0;
        const ctxt = `consultas ${S.q} · acertos ${S.hh} · taxa ${rate}%`;
        if (wide) V.text(g, ctxt, w - 12, reqY + 2, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'right' });
        else V.text(g, ctxt, 12, reqY + 60, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
        // --- pool
        V.text(g, 'pool de blocos físicos: id · conteúdo · ref', 12, poolY - 12, { size: 11, font: 'mono', color: C.muted });
        for (let id = 0; id <= 10; id++) {
          const [x, y] = cellPos(id);
          if (id === 0) {
            V.box(g, x, y, cellW, cellH, { r: 8, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35), lw: 1 });
            V.text(g, '0', x + 6, y + 13, { size: 11, font: 'mono', color: C.muted });
            V.text(g, 'null', x + cellW / 2, y + cellH / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.muted });
            continue;
          }
          const b = S.pool[id];
          const inUse = b.ref > 0, cached = !!b.hash;
          const col = ownCol(b.own);
          const ev = (S.evict || []).includes(id);
          V.box(g, x, y, cellW, cellH, { r: 8, fill: inUse ? V.alpha(col, 0.32) : cached ? V.alpha(col, 0.12) : V.alpha(C.muted, 0.06), stroke: inUse ? col : cached ? null : V.alpha(C.muted, 0.35), lw: inUse ? 1.8 : 1 });
          if (!inUse && cached) { g.setLineDash([4, 3]); V.rr(g, x, y, cellW, cellH, 8); g.strokeStyle = V.alpha(col, 0.9); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]); }
          if (ev && u < 1) { g.fillStyle = V.alpha(RED(C), 0.45 * (1 - u)); V.rr(g, x, y, cellW, cellH, 8); g.fill(); }
          V.text(g, String(id), x + 6, y + 13, { size: 11, font: 'mono', color: C.muted });
          if (b.lab) V.text(g, b.lab, x + cellW / 2, y + cellH / 2 + 3, { size: 13, font: 'mono', weight: 700, align: 'center', color: C.ink });
          if (b.ref > 0) V.text(g, (cellW >= 64 ? 'ref ' : 'r') + b.ref, x + cellW - 5, y + 13, { size: 11, font: 'mono', align: 'right', color: C.ink });
          if (cached) V.text(g, '#' + b.hash, x + cellW / 2, y + cellH - 7, { size: 11, font: 'mono', align: 'center', color: V.alpha(C.ink, 0.8) });
          if (ev) V.text(g, '✕ despejado', x + cellW / 2, y - 4, { size: 11, weight: 700, align: 'center', color: RED(C) });
        }
        // --- fila livre
        V.text(g, wide ? 'fila livre: cabeça = próxima vítima (LRU) → cauda = protegida' : 'fila livre: cabeça → cauda', 12, qY - 14, { size: 11, font: 'mono', color: C.muted });
        const qx0 = 22;
        const qPos = (k) => qx0 + k * (qChip + 5);
        V.arrow(g, qx0 - 4, qY + qChip + 10, qx0 + 10 * (qChip + 5), qY + qChip + 10, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 6 });
        const ids = new Set([...F.queue, ...S.queue]);
        ids.forEach((id) => {
          const ia = F.queue.indexOf(id), ib = S.queue.indexOf(id);
          const [cx, cy] = cellPos(id);
          const pcx = cx + (cellW - qChip) / 2, pcy = cy + (cellH - qChip) / 2;
          let x, y = qY, al = 1;
          if (ia >= 0 && ib >= 0) x = V.lerp(qPos(ia), qPos(ib), u);
          else if (ib >= 0) { x = V.lerp(pcx, qPos(ib), u); y = V.lerp(pcy, qY, u); al = Math.min(1, 0.25 + u); }
          else { x = V.lerp(qPos(ia), pcx, u); y = V.lerp(qY, pcy, u); al = Math.max(0, 1 - u * 1.2); }
          if (al <= 0.02) return;
          const b = S.pool[id].hash ? S.pool[id] : (ib < 0 ? F.pool[id] : S.pool[id]);
          const col = b.hash ? ownCol(b.own) : C.muted;
          g.globalAlpha = al;
          V.box(g, x, y, qChip, qChip, { r: 6, fill: V.alpha(col, b.hash ? 0.35 : 0.14), stroke: V.alpha(col, 0.8), lw: 1.2 });
          V.text(g, String(id), x + qChip / 2, y + qChip / 2 + 1, { size: 11.5, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
          g.globalAlpha = 1;
        });
        if (!S.queue.length) V.text(g, '(vazia: todo bloco está em uso)', qx0, qY + qChip / 2 + 4, { size: 11.5, color: C.muted });
        // --- tabela
        V.text(g, 'tabela hash → bloco', 12, tabY - 12, { size: 11, font: 'mono', color: C.muted });
        S.table.forEach(([hs, id], k) => {
          const r = Math.floor(k / tabPerRow), c = k % tabPerRow;
          const x = 12 + c * (tabChipW + 6), y = tabY + r * 30;
          const b = S.pool[id];
          const col = ownCol(b.own);
          V.box(g, x, y, tabChipW, 24, { r: 6, fill: V.alpha(col, 0.18), stroke: V.alpha(col, 0.7), lw: 1 });
          V.text(g, `${hs}→${id}`, x + tabChipW / 2, y + 12.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
        });
        if (S.evictH) {
          const k = S.table.length; const r = Math.floor(k / tabPerRow), c = k % tabPerRow;
          const x = 12 + c * (tabChipW + 6), y = tabY + r * 30;
          g.globalAlpha = Math.max(0.35, 1 - u * 0.6);
          V.box(g, x, y, tabChipW, 24, { r: 6, fill: V.alpha(RED(C), 0.15), stroke: RED(C), lw: 1 });
          V.text(g, 'h3 ✕', x + tabChipW / 2, y + 12.5, { size: 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: RED(C) });
          g.globalAlpha = 1;
        }
        if (!S.table.length) V.text(g, '(vazia)', 12, tabY + 14, { size: 11.5, color: C.muted });
      });
    }

    /* =====================================================================
       Figura 5 — híbridos: acerto = mínimo entre grupos
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-hybrid');
      const P = 20000, NEW = 1500, TOT = P + NEW;
      const GEO = {
        glm: [{ k: 'att', lab: 'MLA + indexador (atenção)', B: 3584 }, { k: 'state', lab: 'KDA · 4 grupos Mamba (estado)', B: 3584 }, { k: 'dswa', lab: 'drafter DFlash2 · SWA 2.048', W: 2048, B: 3584 }, { k: 'scratch', lab: 'KpoolTail (rascunho)' }],
        qwen: [{ k: 'att', lab: 'QSA (atenção esparsa)', B: 1664 }, { k: 'state', lab: 'GDN · 36 camadas (estado)', B: 1664 }, { k: 'eagle', lab: 'EAGLE block drop (regra)', B: 1664 }],
        ds: [{ k: 'att', lab: 'MLA (C4A/C128A) · bloco 256', B: 256 }, { k: 'swa', lab: 'SWA ×3 · janela 128', W: 128, R: 4096 }],
      };
      const S = { rec: 'glm', s: 15000, stock: false };
      const calc = () => {
        const s = S.s; const n = s >= P ? TOT : s + NEW; const cap = Math.min(s, n - 1);
        const per = {};
        GEO[S.rec].forEach((L) => {
          if (L.k === 'att' || L.k === 'state' || L.k === 'dswa') per[L.k] = Math.floor(cap / L.B) * L.B;
          if (L.k === 'swa') { let c = Math.floor(cap / L.R) * L.R; if (s >= P) c = Math.max(c, Math.floor(P / 256) * 256); per.swa = S.stock ? 0 : c; }
        });
        let hit, pre = 0;
        if (S.rec === 'glm') { hit = Math.min(per.att, per.state); if (S.stock) hit = Math.max(0, hit - 2 * 3584); }
        // Qwen: o block drop não é um grupo; é um corte aplicado ao acerto combinado, depois do mínimo
        else if (S.rec === 'qwen') { pre = Math.min(per.att, per.state); hit = S.stock ? Math.max(0, pre - 1664) : pre; }
        else hit = Math.min(per.att, per.swa);
        return { n, per, hit, pre };
      };
      let H = null, ra = null, rb = null;
      const upd = () => { const c = calc(); if (ra) ra.set(nf(c.hit)); if (rb) rb.set(nf(c.n - c.hit)); if (H) H.redraw(); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DS-V4' }], value: 'glm', onChange: (v) => { S.rec = v; upd(); } });
      UI.slider(ctl, { label: 'Em comum (s)', min: 500, max: P, step: 100, value: S.s, fmt: (v) => nf(v) + ' tok', onInput: (v) => { S.s = v; upd(); } });
      UI.seg(ctl, { label: 'Versão', options: [{ v: false, label: 'corrigida' }, { v: true, label: 'stock' }], value: false, onChange: (v) => { S.stock = v; upd(); } });
      ra = UI.readout(ctl, 'acerto', '—');
      rb = UI.readout(ctl, 'recalcular', '—');
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const c = calc(); const lanes = GEO[S.rec];
        const narrow = w < 640;
        const labW = narrow ? 0 : 214;
        const ax0 = narrow ? 14 : labW + 12, ax1 = w - 16;
        const X = (tok) => ax0 + (tok / TOT) * (ax1 - ax0);
        const laneCol = { att: C.a, state: C.b, dswa: C.c, swa: C.d, eagle: C.f, scratch: C.muted };
        // topo: pedidos
        const lab = (s, y) => { if (narrow) V.text(g, s, ax0, y - 5, { size: 11, color: C.muted }); else V.text(g, s, labW, y + 9, { size: 11.5, color: C.muted, align: 'right' }); };
        const topY = narrow ? 22 : 14;
        lab('pedido anterior (em cache)', topY);
        V.box(g, X(0), topY, X(P) - X(0), 12, { r: 4, fill: V.alpha(C.muted, 0.35) });
        const y2 = topY + (narrow ? 34 : 26);
        lab('novo pedido', y2);
        V.box(g, X(0), y2, X(S.s) - X(0), 12, { r: 4, fill: V.alpha(C.a, 0.45) });
        V.box(g, X(S.s), y2, X(c.n) - X(S.s), 12, { r: 4, fill: V.alpha(C.c, 0.75) });
        // prefill do resto (animado)
        const u = (t % 2.4) / 2.4;
        const px = V.lerp(X(c.hit), X(c.n), V.ease.inOut(u));
        V.glow(g, px, y2 + 6, 12, C.d);
        const laneTop = y2 + (narrow ? 42 : 34);
        const laneH = narrow ? 66 : 54;
        lanes.forEach((L, i) => {
          const y = laneTop + i * laneH;
          const col = laneCol[L.k];
          const ly = narrow ? y + 14 : y;
          if (narrow) V.text(g, L.lab, ax0, y + 8, { size: 11, color: C.ink, weight: 500 });
          else V.text(g, L.lab, labW, ly + 16, { size: 12, color: C.ink, align: 'right', weight: 500, maxW: labW - 6 });
          const bh = 22;
          if (L.k === 'scratch') {
            g.setLineDash([4, 4]); V.rr(g, X(0), ly + 4, X(TOT) - X(0), bh, 5); g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
            V.text(g, 'não participa do prefix cache (não veta acertos)', X(0) + 8, ly + 4 + bh / 2 + 4, { size: 11, color: C.muted, maxW: ax1 - ax0 - 12 });
            return;
          }
          if (L.k === 'eagle') {
            // regra, não grupo: faixa tracejada, sem bandeirinha própria
            g.setLineDash([4, 4]); V.rr(g, X(0), ly + 4, X(TOT) - X(0), bh, 5); g.strokeStyle = V.alpha(col, 0.7); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
            const m = c.pre;
            if (m >= L.B) {
              const bx = X(m - L.B), bw = X(m) - X(m - L.B);
              if (S.stock) { g.fillStyle = V.alpha(RED(C), 0.3); g.fillRect(bx, ly + 4, bw, bh); hatch(g, bx, ly + 4, bw, bh, V.alpha(RED(C), 0.7), 5); V.text(g, '✕', bx + bw / 2, ly + 4 + bh / 2 + 4, { size: 12, weight: 700, align: 'center', color: RED(C) }); }
              else { g.fillStyle = V.alpha(C.a, 0.3); g.fillRect(bx, ly + 4, bw, bh); V.text(g, '✓', bx + bw / 2, ly + 4 + bh / 2 + 4, { size: 12, weight: 700, align: 'center', color: C.ink }); }
            }
            V.text(g, S.stock ? 'não é grupo: depois do mínimo, corta 1 bloco de 1.664' : 'não é grupo; desligada (disable_eagle_block_drop): o bloco fica', X(0), ly + bh + 17, { size: 11, color: C.muted, maxW: ax1 - ax0 });
            return;
          }
          V.box(g, X(0), ly + 4, X(P) - X(0), bh, { r: 4, fill: V.alpha(col, 0.12) });
          if (L.k === 'att') {
            const sp = X(L.B) - X(0);
            if (sp >= 5) for (let k = L.B; k <= P; k += L.B) V.line(g, X(k), ly + 4, X(k), ly + 4 + bh, { color: V.alpha(col, 0.55), lw: 1 });
            else { g.fillStyle = V.alpha(col, 0.25); g.fillRect(X(0), ly + 4, X(P) - X(0), bh); }
          }
          if (L.k === 'state') {
            for (let k = L.B; k <= P; k += L.B) {
              const x = X(k), cy = ly + 4 + bh / 2, on = k === c.per.state;
              g.beginPath(); g.moveTo(x, cy - 7); g.lineTo(x + 6, cy); g.lineTo(x, cy + 7); g.lineTo(x - 6, cy); g.closePath();
              g.fillStyle = on ? col : V.alpha(col, 0.55); g.fill();
              if (on) V.glow(g, x, cy, 14, col);
            }
          }
          if (L.k === 'swa') {
            const pts = []; for (let k = L.R; k <= P; k += L.R) pts.push(k); pts.push(P);
            pts.forEach((k) => {
              const x0 = X(k - L.W), ww = Math.max(4, X(k) - X(k - L.W));
              if (S.stock) { g.setLineDash([3, 3]); g.strokeStyle = V.alpha(col, 0.5); g.strokeRect(x0, ly + 6, ww, bh - 4); g.setLineDash([]); }
              else { g.fillStyle = V.alpha(col, 0.85); g.fillRect(x0, ly + 6, ww, bh - 4); }
            });
            V.text(g, S.stock ? 'caudas liberadas: o mínimo vai a zero' : 'caudas retidas a cada 4.096 + a fronteira do pedido', X(0), ly + bh + 17, { size: 11, color: C.muted });
          }
          if (L.k === 'dswa') {
            for (let k = L.B; k <= P; k += L.B) { const x0 = X(k - L.W), ww = X(k) - X(k - L.W); g.fillStyle = V.alpha(col, 0.55); g.fillRect(x0, ly + 8, ww, bh - 8); }
            V.text(g, S.stock ? 'conta no mínimo e realinha mais 1 página' : 'isento do mínimo (a janela é reconstruída)', X(0), ly + bh + 17, { size: 11, color: C.muted });
          }
          // bandeirinha do grupo
          if (c.per[L.k] != null) {
            const fx = X(c.per[L.k]);
            g.beginPath(); g.moveTo(fx, ly + 2); g.lineTo(fx - 5, ly - 6); g.lineTo(fx + 5, ly - 6); g.closePath(); g.fillStyle = col; g.fill();
            if (!narrow) V.text(g, nf(c.per[L.k]), fx + 7, ly - 1, { size: 11, font: 'mono', color: C.ink });
          }
        });
        const botY = laneTop + lanes.length * laneH + 4;
        // linha da divergência
        V.line(g, X(S.s), y2 - 4, X(S.s), botY, { color: V.alpha(C.c, 0.7), lw: 1.2, dash: [3, 4] });
        // linha do acerto
        const hx = X(c.hit);
        V.line(g, hx, topY - 4, hx, botY, { color: C.d, lw: 2.4, dash: [7, 5], dashOffset: -t * 18 });
        V.text(g, '✂', hx, topY - 6 + (narrow ? 0 : 0), { size: 14, align: 'center', color: C.d });
        const msgs = narrow ? [`acerto = ${nf(c.hit)} tokens`, `recalcula ${nf(c.n - c.hit)} de ${nf(c.n)}`] : [`acerto = ${nf(c.hit)} tokens · recalcula ${nf(c.n - c.hit)} de ${nf(c.n)}`];
        const mw = Math.max(...msgs.map((m) => tw(g, m, 12.5, 'mono', 600))) + 20, mh = 10 + msgs.length * 17;
        const bx = V.clamp(hx - mw / 2, 8, w - mw - 8);
        V.box(g, bx, botY + 6, mw, mh, { r: 8, fill: V.alpha(C.d, 0.18), stroke: C.d, lw: 1.2 });
        msgs.forEach((m, i) => V.text(g, m, bx + mw / 2, botY + 6 + 13.5 + i * 17, { size: 12.5, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink }));
        const note = S.stock ? ({ glm: 'stock: todo grupo marcado EAGLE → MLA perde 1 página e a SWA do drafter realinha outra', qwen: 'stock: EAGLE block drop descarta o último bloco casado (1.664 tokens)', ds: 'stock (#26): a SWA libera blocos antigos e o mínimo entre grupos zera' })[S.rec] : ({ glm: 'grão 3.584: estado KDA só em checkpoints; drafter isento com replay', qwen: 'grão 1.664: estado GDN em align; block drop desligado (disable_eagle_block_drop)', ds: 'MLA a cada 256, mas a SWA só tem caudas a cada 4.096 (+ a fronteira)' })[S.rec];
        const nh = wrap(g, note, w / 2, botY + mh + 26, w - 24, { size: 11.5, align: 'center', color: C.muted });
        fitH(H || { canvas: fig.querySelector('canvas') }, botY + mh + 20 + nh);
      }, { static: false });
      upd();
    }

    /* =====================================================================
       Figura 6 — replay da janela do rascunhador
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-drafter');
      const CFG = { ds: { W: 128, G: 256, P: 6000, span: 1150, lab: 'DSpark · janela 128' }, glm: { W: 2048, G: 3584, P: 15000, span: 9500, lab: 'DFlash2 · janela 2.048' } };
      const S = { rec: 'ds', fix: false, fresh: 0, t0: 0 };
      const calc = () => {
        const c = CFG[S.rec]; const n = c.P + S.fresh;
        let cap = n - 1; if (S.fix && S.rec === 'ds') cap = Math.max(0, n - 1 - c.W);
        let hit = Math.floor(Math.min(c.P, cap) / c.G) * c.G; let rep = 0;
        if (S.fix && S.rec === 'glm') { while (n - hit < c.W && hit > 0) { hit -= c.G; rep++; } }
        const comp = n - hit;
        return { c, n, hit, comp, fill: Math.min(1, comp / c.W), rep, cap };
      };
      let H = null, ra = null, rb = null, rc = null;
      const upd = () => { const d = calc(); if (ra) ra.set(nf(d.hit)); if (rb) rb.set(nf(d.comp)); if (rc) rc.set(Math.round(d.fill * 100) + '%'); if (H) S.t0 = H.t; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Rascunhador', options: [{ v: 'ds', label: 'DS · DSpark (128)' }, { v: 'glm', label: 'GLM · DFlash2 (2.048)' }], value: 'ds', onChange: (v) => { S.rec = v; upd(); } });
      UI.toggle(ctl, { label: 'correção ativa (teto n−1−W / replay clamp)', value: false, onChange: (v) => { S.fix = v; upd(); } });
      UI.slider(ctl, { label: 'Tokens novos', min: 0, max: 3000, step: 10, value: 0, fmt: (v) => (v === 0 ? '0 (idêntico)' : nf(v)), onInput: (v) => { S.fresh = v; upd(); } });
      ra = UI.readout(ctl, 'acerto', '—'); rb = UI.readout(ctl, 'recalculados', '—'); rc = UI.readout(ctl, 'janela', '—');
      const r = V.rng(11);
      const dots = Array.from({ length: 26 }, () => ({ o: r() }));
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const d = calc(); const c = d.c; const narrow = w < 620;
        const tt = Math.max(0, t - S.t0), ph = (tt % 4.5) / 4.5;
        const ringR = narrow ? 44 : 60;
        const ax0 = 14, ax1 = narrow ? w - 14 : w - 2 * ringR - 80;
        const v0 = Math.max(0, d.n - c.span), v1 = d.n;
        const X = (tok) => ax0 + ((tok - v0) / (v1 - v0)) * (ax1 - ax0);
        // faixa do alvo
        const y1 = 40;
        V.text(g, 'KV do alvo · fim do prompt', ax0, y1 - 22, { size: 11.5, color: C.muted });
        const hx = X(Math.max(v0, d.hit));
        V.box(g, X(v0), y1, hx - X(v0), 24, { r: 0, fill: V.alpha(C.a, 0.25) });
        hatch(g, X(v0), y1, hx - X(v0), 24, V.alpha(C.ink, 0.28), 6);
        const sweep = Math.min(1, ph / 0.3);
        V.box(g, hx, y1, (X(d.n) - hx) * sweep, 24, { r: 0, fill: V.alpha(C.c, 0.85) });
        g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.strokeRect(X(v0) + 0.5, y1 + 0.5, X(v1) - X(v0), 24);
        if (hx - X(v0) > 80) V.text(g, '♻ veio do cache', X(v0) + 6, y1 + 16, { size: 11, weight: 600, color: C.ink });
        for (let k = Math.ceil(v0 / c.G) * c.G; k <= v1; k += c.G) V.line(g, X(k), y1 - 3, X(k), y1 + 27, { color: V.alpha(C.muted, 0.6), lw: 1, dash: [2, 3] });
        // janela do rascunhador
        const wy = y1 + 58;
        V.text(g, `janela do rascunhador: últimos ${nf(c.W)} tokens`, ax0, wy - 8, { size: 11.5, color: C.muted });
        const wx0 = X(Math.max(v0, d.n - c.W));
        const okx = X(Math.max(d.hit, d.n - c.W));
        V.box(g, wx0, wy, X(d.n) - wx0, 14, { r: 4, fill: V.alpha(C.b, 0.15), stroke: V.alpha(C.b, 0.8), lw: 1 });
        V.box(g, okx, wy, X(d.n) - okx, 14, { r: 4, fill: V.alpha(C.b, 0.8) });
        if (d.fill < 1) hatch(g, wx0, wy, okx - wx0, 14, V.alpha(RED(C), 0.85), 4);
        V.line(g, hx, y1 - 6, hx, wy + 20, { color: C.d, lw: 2, dash: [6, 4], dashOffset: -t * 14 });
        pill(g, V.clamp(hx, 60, ax1 - 50), y1 - 6, 'acerto ' + nf(d.hit), C.d, { align: 'center', size: 11, a: 0.3 });
        // conta
        const mathTxt = S.rec === 'ds'
          ? (S.fix ? `n = ${nf(d.n)} · teto n−1−128 = ${nf(Math.max(0, d.n - 1 - c.W))} · alinhado a 256 → ${nf(d.hit)} · recalcula ${nf(d.comp)}` : `n = ${nf(d.n)} · teto n−1 = ${nf(d.n - 1)} · alinhado a 256 → ${nf(d.hit)} · recalcula ${nf(d.comp)}`)
          : (S.fix && d.rep ? `n = ${nf(d.n)} · acerto de ${nf(d.hit + d.rep * c.G)} recua ${d.rep} página(s) de 3.584 → ${nf(d.hit)} · recalcula ${nf(d.comp)}` : `n = ${nf(d.n)} · teto n−1 = ${nf(d.n - 1)} · alinhado a 3.584 → ${nf(d.hit)} · recalcula ${nf(d.comp)}`);
        // anel: no desktop, à direita do topo; no estreito, logo abaixo do fim da janela (à direita),
        // para que os pontos não atravessem nenhum texto; a saída fica à esquerda do anel e a conta embaixo
        const ringCx = w - (narrow ? 20 : 30) - ringR;
        const ringCy = narrow ? wy + 36 + ringR : 32 + ringR + 20;
        let my = wy + 42, ox = ax0, ow = ax1 - ax0, oy;
        if (narrow) { ow = ringCx - ringR - 18 - ax0; oy = wy + 46; }
        else { my += wrap(g, mathTxt, ax0, my, ax1 - ax0, { size: 12, font: 'mono', weight: 600, color: C.ink }); oy = my + 14; }
        const segs = 32, full = Math.round(d.fill * segs), filled = Math.round(full * Math.min(1, Math.max(0, (ph - 0.12) / 0.45)));
        for (let k = 0; k < segs; k++) {
          const a0 = -Math.PI / 2 + (k / segs) * Math.PI * 2, a1 = a0 + (Math.PI * 2) / segs - 0.05;
          g.beginPath(); g.arc(ringCx, ringCy, ringR, a0, a1); g.lineWidth = narrow ? 11 : 14;
          g.strokeStyle = k < filled ? C.b : k < full ? V.alpha(C.b, 0.22) : V.alpha(RED(C), 0.4); g.stroke();
        }
        V.text(g, narrow ? 'anel' : 'anel do drafter', ringCx, ringCy - 8, { size: 11, align: 'center', color: C.muted });
        V.text(g, `${nf(Math.min(d.comp, c.W))}/${nf(c.W)}`, ringCx, ringCy + 11, { size: narrow ? 12 : 14, font: 'mono', weight: 700, align: 'center', color: d.fill < 1 ? RED(C) : C.ink });
        if (ph > 0.1 && ph < 0.62) dots.forEach((p) => {
          const uu = ((ph - 0.1) / 0.52 + p.o) % 1;
          const sx = V.lerp(okx, X(d.n), p.o), sy = wy + 7;
          const ang = -Math.PI / 2 + p.o * Math.PI * 2 * d.fill;
          const ex = ringCx + Math.cos(ang) * ringR, ey = ringCy + Math.sin(ang) * ringR;
          const q = narrow
            ? V.bez([sx, sy], [sx, sy + 18], [ex, ey - 26], [ex, ey], V.ease.inOut(uu))
            : V.bez([sx, sy], [sx + 30, sy + 30], [ex - 40, ey - 20], [ex, ey], V.ease.inOut(uu));
          V.dot(g, q[0], q[1], 2.6, V.alpha(C.b, 0.9));
        });
        // saída
        V.text(g, 'saída do servidor', ox, oy, { size: 11.5, font: 'mono', color: C.muted });
        const kShow = 1 + Math.floor(Math.max(0, ph - 0.45) / 0.07);
        let toks, verdict, vcol;
        const okToks = ['[', '"json",', '"ok",', '"fim"', ']'];
        if (d.fill < 1 && S.rec === 'ds') {
          // caso relatado no DSpark: o verificador aceitava respostas truncadas
          toks = ['[', '"json', '⏹ stop']; verdict = 'resposta truncada, determinística a temp 0 · 12,1 tok/s no json60 repetido (upstream)'; vcol = RED(C);
        } else if (d.fill < 1) {
          // DFlash2: o alvo verifica cada token, então a saída continua correta; o prejuízo é na proposta
          toks = okToks; verdict = 'saída correta: o alvo verifica cada token. Mas o drafter propõe sem contexto e a aceitação tende a cair (interpretação); a correção recua o acerto até o sufixo reconstruir a janela'; vcol = C.d;
        } else {
          toks = okToks;
          if (S.fix && S.rec === 'ds') verdict = `janela completa: recalcula 128 + o resto até o bloco de 256 (aqui ${nf(d.comp)}) · 86,5 tok/s (upstream)`;
          else if (S.fix && d.rep) verdict = 'replay clamp: o sufixo fresco (≥ 2.048) reconstrói a janela do DFlash2';
          else verdict = 'o sufixo recalculado já enche a janela: sem problema neste caso';
          vcol = C.a;
        }
        let xx = ox, yy = oy + 20;
        toks.slice(0, Math.min(toks.length, kShow)).forEach((s0) => {
          const wd = tw(g, s0, 11.5, 'mono', 600) + 12;
          if (xx + wd > ox + ow) { xx = ox; yy += 24; }
          pill(g, xx, yy, s0, s0.startsWith('⏹') ? RED(C) : C.e, { size: 11.5, a: 0.22 });
          xx += wd + 4;
        });
        const vh = wrap(g, verdict, ox, yy + 28, ow, { size: 11.5, weight: 600, color: vcol });
        const outB = yy + 28 + vh;
        let bottom;
        if (narrow) {
          my = Math.max(ringCy + ringR + 26, outB + 14);
          my += wrap(g, mathTxt, ax0, my, w - 2 * ax0, { size: 12, font: 'mono', weight: 600, color: C.ink });
          bottom = my;
        } else bottom = Math.max(outB, ringCy + ringR + 16);
        fitH(H || { canvas: fig.querySelector('canvas') }, bottom + 6);
      });
      upd();
    }

    /* =====================================================================
       Figura 7 — IDs por grupo e retenção
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-ids');
      const S = { glob: 'd', swa: 'inh', pool: 642, L: 50000, edit: 90 };
      const per = () => {
        const Sg = Math.floor(S.L / 3584); const R = S.glob;
        const mla = Sg;
        const mamba = R === 'd' ? 4 * Sg : R === 0 ? 4 : 4 * Math.floor(S.L / R) + 4;
        const draft = S.swa === '0' ? 33 : R === 'd' ? 33 * Sg : R === 0 ? 33 : 33 * Math.floor(S.L / R) + 33;
        return { Sg, mla, mamba, draft, tot: mla + mamba + draft };
      };
      // checkpoints de estado de UM histórico (posições em tokens, do mais antigo ao mais recente)
      const ckpts = () => {
        const end = Math.floor(S.L / 3584) * 3584, step = S.glob === 'd' ? 3584 : S.glob, out = [];
        if (step > 0) for (let q = step; q < end; q += step) out.push(q);
        out.push(end); // a fronteira final é sempre retida
        return out;
      };
      // modelo simplificado de quem sobrevive ao PRÓPRIO prime quando o histórico não cabe no pool:
      // o pedido em voo segura as páginas MLA + 1 estado Mamba (4) + a janela do drafter (33); os checkpoints
      // que ficam para trás voltam com hash à fila livre e as alocações seguintes do prime despejam os mais antigos
      const survive = () => {
        const p = per(), cks = ckpts();
        if (p.tot <= S.pool) return { cks, keep: cks.length, fits: true, cost: 0 };
        const cost = 4 + (S.swa === '0' ? 0 : 33);
        const room = Math.max(0, S.pool - p.mla - 37);
        return { cks, keep: Math.min(cks.length, 1 + Math.floor(room / cost)), fits: false, cost };
      };
      const reuse = () => {
        const e = (S.L * S.edit) / 100, sv = survive();
        const alive = sv.cks.slice(sv.cks.length - sv.keep);
        const a = Math.floor(e / 3584) * 3584; // MLA: denso, sempre sobrevive
        let m = 0; alive.forEach((q) => { if (q <= e && q > m) m = q; });
        const lost = sv.cks.some((q) => q <= e && q > m);
        return { rz: Math.min(a, m), e, sv, alive, lost };
      };
      let H = null, r1 = null, r2 = null, r3 = null;
      const upd = () => {
        const p = per(); const nC = Math.floor(S.pool / p.tot);
        if (r1) r1.set(nf(p.tot)); if (r2) r2.set(nC ? String(nC) : '0 (parcial)'); if (r3) r3.set(nC ? nf(nC * p.Sg * 3584) : 'parcial');
        if (H) H.redraw();
      };
      const ctl = UI.controls(fig);
      const sG = UI.seg(ctl, { label: 'Retenção global', options: [{ v: 'd', label: 'densa' }, { v: 7168, label: '7.168' }, { v: 14336, label: '14.336' }, { v: 28672, label: '28.672' }, { v: 0, label: '0' }], value: 'd', onChange: (v) => { S.glob = v; upd(); } });
      const sW = UI.seg(ctl, { label: 'Retenção SWA (drafter)', options: [{ v: 'inh', label: 'herda' }, { v: '0', label: '0' }], value: 'inh', onChange: (v) => { S.swa = v; upd(); } });
      const sP = UI.seg(ctl, { label: 'Pool', options: [{ v: 642, label: '642 IDs' }, { v: 390, label: '390 IDs' }], value: 642, onChange: (v) => { S.pool = v; upd(); } });
      const sL = UI.slider(ctl, { label: 'Cada histórico', min: 20000, max: 250000, step: 1000, value: S.L, fmt: (v) => Math.round(v / 1000) + 'K tok', onInput: (v) => { S.L = v; upd(); } });
      const sE = UI.slider(ctl, { label: 'Edição em', min: 5, max: 99, step: 0.5, value: S.edit, fmt: (v) => sf(v, v % 1 ? 1 : 0) + '%', onInput: (v) => { S.edit = v; upd(); } });
      UI.button(ctl, '▶ caso medido: 128K · 390 IDs', () => { sG.set('d'); sW.set('inh'); sP.set(390); sL.set(128000); sE.set(89.5); });
      r1 = UI.readout(ctl, 'IDs por histórico', '—'); r2 = UI.readout(ctl, 'históricos que cabem', '—'); r3 = UI.readout(ctl, 'tokens em cache', '—');
      UI.legend(ctl, [[cv('a'), 'MLA'], [cv('b'), 'Mamba ×4'], [cv('c'), 'drafter (SWA)']]);
      H = V.fig(fig, (g, w, h) => {
        const C = V.C(); const p = per(); const N = S.pool; const nC = Math.floor(N / p.tot);
        const ru = reuse(), sv = ru.sv, K = Math.round(S.L / 1000);
        const gx = 14, gy = 30, gw = w - 28, gh = Math.min(200, h * 0.46);
        let cell = Math.floor(Math.sqrt((gw * gh) / N));
        let cols = Math.floor(gw / cell); while (Math.ceil(N / cols) * cell > gh && cell > 4) { cell--; cols = Math.floor(gw / cell); }
        const sz = Math.max(3, cell - (cell > 7 ? 2 : 1));
        V.text(g, w < 560 ? `pool: ${N} IDs · 1 quadrado = 1 ID` : `pool: ${N} IDs utilizáveis · cada quadrado = 1 ID (uma página inteira, ≈25,9 MiB no GLM)`, gx, 18, { size: 11.5, color: C.muted });
        let k = 0;
        const put = (col, a) => { if (k >= N) return; const r = Math.floor(k / cols), c = k % cols; g.fillStyle = V.alpha(col, a); g.fillRect(gx + c * cell, gy + r * cell, sz, sz); k++; };
        for (let ci = 0; ci < nC; ci++) {
          const a = ci % 2 ? 0.62 : 0.9;
          for (let q = 0; q < p.mla; q++) put(C.a, a);
          for (let q = 0; q < p.mamba; q++) put(C.b, a);
          for (let q = 0; q < p.draft; q++) put(C.c, a);
        }
        if (nC === 0) {
          // o que sobra ao fim do prime: todas as páginas MLA + só os checkpoints mais recentes
          for (let q = 0; q < p.mla; q++) put(C.a, 0.9);
          for (let q = 0; q < sv.keep - 1; q++) { for (let z = 0; z < 4; z++) put(C.b, 0.9); for (let z = 4; z < sv.cost; z++) put(C.c, 0.9); }
          for (let z = 0; z < 4; z++) put(C.b, 0.9); for (let z = 0; z < 33; z++) put(C.c, 0.9);
          g.strokeStyle = C.d; g.lineWidth = 2; g.setLineDash([6, 4]); g.strokeRect(gx - 3, gy - 3, cols * cell + 4, Math.ceil(N / cols) * cell + 4); g.setLineDash([]);
        }
        while (k < N) put(C.muted, 0.16);
        const gb = gy + Math.ceil(N / cols) * cell + 16;
        const txt = nC === 0 ? `um histórico de ${K}K pediria ${nf(p.tot)} IDs e o pool tem ${N}: ao fim do próprio prime sobrevivem as ${p.mla} páginas MLA e só os ${sv.keep} checkpoints de estado mais recentes` : `${nC} histórico(s) de ${K}K · por histórico: MLA ${nf(p.mla)} + Mamba ${nf(p.mamba)} + drafter ${nf(p.draft)} = ${nf(p.tot)} IDs`;
        let yy = gb + wrap(g, txt, gx, gb, gw, { size: 12, weight: 600, color: nC === 0 ? C.d : C.ink });
        const segCost = S.glob === 'd' && S.swa === 'inh' ? '38 IDs por segmento de 3.584 (1 + 4 + 33): 87% no drafter' : 'retenção esparsa: menos IDs por segmento, mais recomputação em edições';
        yy += wrap(g, segCost, gx, yy, gw, { size: 11.5, color: C.muted });
        // barra de edição
        const by = yy + 18;
        const rz = ru.rz, e = ru.e, rec = S.L - rz;
        const bl = wrap(g, `histórico de ${K}K editado em ${sf(S.edit, S.edit % 1 ? 1 : 0)}% (LCP ${nf(e)}): volta ao último checkpoint de estado que sobreviveu antes da edição`, gx, by, gw, { size: 11.5, color: C.muted });
        const bx0 = gx, bw = gw, byy = by + bl + 8, bh = 22;
        const X = (tok) => bx0 + (tok / S.L) * bw;
        V.box(g, X(0), byy, X(rz) - X(0), bh, { r: 0, fill: V.alpha(C.a, 0.55) });
        V.box(g, X(rz), byy, X(S.L) - X(rz), bh, { r: 0, fill: V.alpha(C.c, 0.55) });
        g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.strokeRect(X(0) + 0.5, byy + 0.5, bw, bh);
        // losangos = checkpoints de estado; cheios = sobreviveram; vazados = despejados durante o prime
        const aliveSet = new Set(ru.alive);
        const sp = sv.cks.length > 1 ? X(sv.cks[1]) - X(sv.cks[0]) : 99;
        const dh = sp >= 14 ? 6 : sp >= 6 ? 4 : 0;
        sv.cks.forEach((q) => {
          const x = X(q), cy = byy + bh / 2, on = aliveSet.has(q);
          if (!dh) { V.line(g, x, byy + 3, x, byy + bh - 3, { color: V.alpha(on ? C.b : C.muted, on ? 0.9 : 0.5), lw: 1 }); return; }
          const dw = dh * 0.67;
          g.beginPath(); g.moveTo(x, cy - dh); g.lineTo(x + dw, cy); g.lineTo(x, cy + dh); g.lineTo(x - dw, cy); g.closePath();
          if (on) { g.fillStyle = V.alpha(C.b, 0.95); g.fill(); } else { g.strokeStyle = V.alpha(C.ink, 0.45); g.lineWidth = 1; g.stroke(); }
        });
        const exx = X(e);
        V.line(g, exx, byy - 6, exx, byy + bh + 6, { color: RED(C), lw: 2.2 });
        V.text(g, '✎', exx, byy - 9, { size: 13, align: 'center', color: RED(C) });
        let y3 = byy + bh + 20;
        y3 += wrap(g, `reaproveita ${nf(rz)} · recalcula ${nf(rec)} tokens (≈ ${sf(rec / 1100, 1)} s a ~1.100 tok/s)`, gx, y3, gw, { size: 12.5, font: 'mono', weight: 600, color: C.ink });
        if (ru.lost) y3 += wrap(g, rz ? 'o checkpoint mais próximo da edição foi despejado durante o próprio prime: o acerto recua até o anterior que sobreviveu' : 'os checkpoints antes da edição foram despejados durante o próprio prime: acerto zero, mesmo com a grade densa', gx, y3 + 2, gw, { size: 11.5, weight: 600, color: RED(C) }) + 2;
        y3 += wrap(g, 'medido (128K, 390 IDs): densa 0 a 10% e a 50%, 111.104 a 90% (LCP 114.666; 14,70 s) · 0/0: 0 nas três (112,49 s a 90%)', gx, y3 + 2, gw, { size: 11, color: C.muted });
        fitH(H || { canvas: fig.querySelector('canvas') }, y3 + 8);
      }, { static: true });
      upd();
    }

    /* =====================================================================
       Figura 8 — a esteira da fila livre (no-store)
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-nostore');
      const S = { noStore: false, auto: true };
      let chips, queue, tray, st;
      const reset = () => {
        chips = []; queue = []; tray = [];
        for (let i = 0; i < 6; i++) chips.push({ id: i + 1, own: 'free', hashed: false, x: 0, y: 0, init: true });
        for (let i = 0; i < 10; i++) chips.push({ id: i + 7, own: 'O', hashed: true, x: 0, y: 0, init: true });
        queue = chips.slice();
        st = { phase: 'idle', last: null, runT: 0, jobs: 0, ev: 0, flash: [] };
      };
      reset();
      let H = null, ro = null;
      const ownerLeft = () => queue.filter((c) => c.own === 'O').length + tray.filter((c) => c.own === 'O').length;
      const startJob = (t) => {
        if (st.phase !== 'idle') return;
        tray = queue.splice(0, 5);
        tray.forEach((c) => { if (c.own === 'O' && c.hashed) { st.ev++; st.flash.push({ x: c.x, y: c.y, t }); } c.own = 'J'; c.hashed = false; });
        st.phase = 'run'; st.runT = t;
      };
      const finishJob = (t) => {
        if (S.noStore) { tray.forEach((c) => { c.hashed = false; }); queue = tray.concat(queue); }
        else { tray.forEach((c) => { c.hashed = true; }); queue = queue.concat(tray.slice().reverse()); }
        tray = []; st.phase = 'idle'; st.last = t; st.jobs++;
      };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'jobs usam skip_writing_prefix_cache', value: false, onChange: (v) => { S.noStore = v; reset(); } });
      UI.toggle(ctl, { label: 'jobs automáticos', value: true, onChange: (v) => { S.auto = v; } });
      UI.button(ctl, '▶ enviar job de lote', () => { if (H) startJob(H.t); }, { primary: true });
      UI.button(ctl, '↺ reiniciar', () => { reset(); });
      ro = UI.readout(ctl, 'conversa do dono em cache', '10 / 10');
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (st.last == null) st.last = t - 0.9;
        if (S.auto && st.phase === 'idle' && t - st.last > 1.3) startJob(t);
        if (st.phase === 'run' && t - st.runT > 1.1) finishJob(t);
        const n = 16;
        const chip = Math.max(14, Math.min(40, (w - 48) / n - 4));
        const qx0 = 24, qy = 58 + chip + 96;
        const posQ = (k) => [qx0 + k * (chip + 4), qy];
        const trayW = Math.max(150, 5 * (chip + 4) - 4);
        const trayX = (w - trayW) / 2, trayY = 58;
        // bandeja
        V.box(g, trayX - 12, trayY - 22, trayW + 24, chip + 34, { r: 10, fill: V.alpha(C.d, 0.07), stroke: V.alpha(C.d, 0.5), lw: 1 });
        V.text(g, st.phase === 'run' ? 'job de lote rodando…' : 'job de lote (em uso)', trayX + trayW / 2, trayY - 8, { size: 11.5, align: 'center', color: C.muted });
        // esteira
        const beltY = qy + chip + 6;
        V.box(g, qx0 - 10, beltY, n * (chip + 4) + 16, 12, { r: 6, fill: V.alpha(C.muted, 0.18) });
        for (let x = qx0 - 6 + ((t * 30) % 14); x < qx0 + n * (chip + 4); x += 14) V.line(g, x, beltY + 2, x - 5, beltY + 10, { color: V.alpha(C.muted, 0.5), lw: 1.5 });
        V.text(g, '◀ cabeça: alocações pegam daqui', qx0 - 8, qy - 14, { size: 11, color: C.muted });
        V.text(g, 'cauda ▶', qx0 + n * (chip + 4) - 4, qy - 14, { size: 11, color: C.muted, align: 'right' });
        V.text(g, S.noStore ? 'sem hash → frente (LIFO)' : 'com hash → cauda (LRU)', w / 2, beltY + 34, { size: 12, weight: 600, align: 'center', color: S.noStore ? C.a : C.d });
        // alvos
        queue.forEach((c, k) => { const [x, y] = posQ(k); c.tx = x; c.ty = y; });
        const trayC = trayX + (trayW - (5 * (chip + 4) - 4)) / 2;
        tray.forEach((c, k) => { c.tx = trayC + k * (chip + 4); c.ty = trayY; });
        const kk = Math.min(1, dt * 7);
        chips.forEach((c) => {
          if (c.init) { c.x = c.tx; c.y = c.ty; c.init = false; }
          c.x += (c.tx - c.x) * kk; c.y += (c.ty - c.y) * kk;
          const col = c.own === 'O' ? C.a : c.own === 'J' ? C.d : C.muted;
          V.box(g, c.x, c.y, chip, chip, { r: 6, fill: V.alpha(col, c.own === 'free' ? 0.14 : 0.5), stroke: V.alpha(col, 0.85), lw: 1.2 });
          if (c.hashed && chip >= 18) V.text(g, '#', c.x + chip / 2, c.y + chip / 2 + 1, { size: 12, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        });
        st.flash = st.flash.filter((f) => t - f.t < 1.2);
        st.flash.forEach((f) => { const a = 1 - (t - f.t) / 1.2; V.text(g, '✕', f.x + chip / 2, f.y + chip / 2 - 18 * (1 - a), { size: 20, weight: 700, align: 'center', base: 'middle', color: V.alpha(RED(C), a) }); });
        const left = ownerLeft();
        if (ro) ro.set(left + ' / 10');
        const ty = beltY + (w < 560 ? 86 : 64);
        fitH(H || { canvas: fig.querySelector('canvas') }, ty + 18);
        V.text(g, `jobs: ${st.jobs} · blocos do dono despejados: ${st.ev}`, 14, w < 560 ? ty - 22 : ty, { size: 12, font: 'mono', color: C.ink });
        const bwid = Math.min(160, w * 0.3);
        V.text(g, 'dono', w - bwid - 20, ty, { size: 11.5, color: C.muted, align: 'right' });
        V.box(g, w - bwid - 14, ty - 10, bwid, 10, { r: 5, fill: V.alpha(C.muted, 0.2) });
        V.box(g, w - bwid - 14, ty - 10, Math.max(2, (bwid * left) / 10), 10, { r: 5, fill: left > 6 ? C.a : left > 2 ? C.d : RED(C) });
        if (left === 0) V.text(g, 'a conversa do dono foi inteira despejada: o próximo turno refaz o prefill', w / 2, qy - 36, { size: 12, weight: 600, align: 'center', color: RED(C), maxW: w - 20 });
      });
    }

    /* =====================================================================
       Figura 9 — LMCache
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-lmcache');
      const S = { sc: 'store', t0: 0 };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 'store', label: '1º contato: grava' }, { v: 'reload', label: 'reinício + reload' }, { v: 'prefill', label: 'sem LMCache' }, { v: 'dead', label: 'servidor caiu' }], value: 'store', onChange: (v) => { S.sc = v; if (H) S.t0 = H.t; } });
      UI.legend(ctl, [[cv('a'), 'chunk de KV (256 tokens)'], [cv('b'), 'GPU'], [cv('d'), 'L1 · RAM de CPU'], [cv('muted'), 'L2 · disco'], [cv('f'), 'servidor LMCache (dono da L1/L2)']]);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const narrow = w < 620;
        const tt = Math.max(0, t - S.t0), ph = (tt % 4) / 4;
        const colW = narrow ? w - 28 : Math.min(380, w * 0.5);
        const x0 = 14;
        const boxH = 66, gap = narrow ? 30 : 36;
        const yG = 30, yL1 = yG + boxH + gap, yL2 = yL1 + boxH + gap;
        const srvW = narrow ? 112 : 132, srvGap = 26;
        const bW = colW - srvW - srvGap;
        const dead = S.sc === 'dead', off = S.sc === 'prefill';
        const L2c = C.muted;
        const box = (y, lab, sub, col, dim, subCol) => {
          V.box(g, x0, y, bW, boxH, { r: 10, fill: V.alpha(col, dim ? 0.05 : 0.14), stroke: V.alpha(col, dim ? 0.3 : 0.85), lw: 1.4 });
          V.text(g, lab, x0 + 10, y + 16, { size: 12, weight: 600, color: dim ? C.muted : C.ink, maxW: bW - 16 });
          V.text(g, sub, x0 + 10, y + boxH - 8, { size: 11, weight: subCol ? 600 : 400, color: subCol || C.muted, maxW: bW - 16 });
        };
        V.text(g, 'um nó (o outro é idêntico)', x0, 16, { size: 11.5, color: C.muted });
        box(yG, 'GPU · pool de KV', dead ? 'pedidos com acerto estacionados' : S.sc === 'reload' && ph < 0.12 ? 'engine reiniciado: pool vazio' : 'prefix cache normal (vLLM)', C.b, false, dead ? RED(C) : null);
        box(yL1, 'L1 · RAM de CPU (12 GB)', 'memória unificada: pool da GPU', C.d, off || dead);
        box(yL2, 'L2 · disco NVMe', '$HOME/lmcache-disk', L2c, off || dead);
        // servidor, entre L1 e L2, ligado às duas camadas que ele gerencia
        const sH = 70, sx = x0 + bW + srvGap, sy = yL1 + boxH + gap / 2 - sH / 2;
        const sCol = dead ? RED(C) : C.f;
        V.arrow(g, sx - 2, sy + 16, x0 + bW + 3, yL1 + boxH / 2, { color: V.alpha(sCol, off ? 0.3 : 0.8), lw: 1.4, head: 5, dash: dead || off ? [3, 3] : null });
        V.arrow(g, sx - 2, sy + sH - 16, x0 + bW + 3, yL2 + boxH / 2, { color: V.alpha(sCol, off ? 0.3 : 0.8), lw: 1.4, head: 5, dash: dead || off ? [3, 3] : null });
        V.box(g, sx, sy, srvW, sH, { r: 22, fill: V.alpha(sCol, off ? 0.05 : 0.14), stroke: V.alpha(sCol, off ? 0.3 : 0.9), lw: 1.4 });
        V.text(g, 'lmcache server', sx + srvW / 2, sy + 20, { size: 11.5, weight: 600, align: 'center', color: off ? C.muted : C.ink, maxW: srvW - 8 });
        V.text(g, dead ? '✕ morto' : 'fabric-ip:6667', sx + srvW / 2, sy + 38, { size: 11, font: 'mono', align: 'center', color: dead ? RED(C) : C.muted, maxW: srvW - 8 });
        V.text(g, 'dono da L1 e da L2', sx + srvW / 2, sy + 56, { size: 11, align: 'center', color: C.muted, maxW: srvW - 8 });
        // chunks na GPU
        const nCh = narrow ? 8 : 10, chW = Math.min(22, (bW - 30) / nCh - 4);
        const gpuFill = S.sc === 'prefill' || S.sc === 'store' ? Math.min(1, ph / 0.8) : S.sc === 'reload' ? Math.min(1, Math.max(0, (ph - 0.12) / 0.2)) : 1;
        for (let k = 0; k < nCh; k++) {
          const on = k < Math.round(gpuFill * nCh);
          V.box(g, x0 + 12 + k * (chW + 4), yG + 24, chW, 13, { r: 3, fill: on ? V.alpha(C.a, 0.85) : V.alpha(C.muted, 0.15) });
        }
        const lvl = (y, frac, col) => { for (let k = 0; k < nCh; k++) { const on = k < Math.round(frac * nCh); V.box(g, x0 + 12 + k * (chW + 4), y + 24, chW, 13, { r: 3, fill: on ? V.alpha(C.a, 0.7) : V.alpha(col, 0.1) }); } };
        if (!off && !dead) {
          lvl(yL1, S.sc === 'store' ? Math.min(1, Math.max(0, (ph - 0.1) / 0.8)) : 1, C.d);
          lvl(yL2, S.sc === 'store' ? Math.min(1, Math.max(0, (ph - 0.25) / 0.7)) : 1, L2c);
          // fluxo: os chunks só andam nos vãos entre as caixas, sem cruzar rótulos
          const up = S.sc === 'reload';
          const gapsDown = [[yG + boxH, yL1], [yL1 + boxH, yL2]];
          const fx = x0 + Math.max(bW * 0.52, 96);
          for (let k = 0; k < 8; k++) {
            const u = (ph * 2 + k / 8) % 1;
            const gi = u < 0.5 ? 0 : 1, lu = (u % 0.5) / 0.5;
            let [ya, yb] = gapsDown[up ? 1 - gi : gi];
            if (up) [ya, yb] = [yb, ya];
            const yy = V.lerp(ya + (up ? -6 : 6), yb + (up ? 6 : -6), lu);
            V.box(g, fx + (k % 3) * 12, yy - 4, 8, 8, { r: 2, fill: V.alpha(C.a, 0.35 + 0.65 * Math.sin(Math.PI * lu)) });
          }
          const lab = up ? '▲ reload' : '▼ grava';
          V.text(g, lab, x0 + 10, yG + boxH + gap / 2 + 4, { size: 11, weight: 600, color: C.ink });
          V.text(g, lab, x0 + 10, yL1 + boxH + gap / 2 + 4, { size: 11, weight: 600, color: C.ink });
        }
        if (dead) {
          // pedidos estacionados: fila de pontos ao lado do rótulo da GPU, ampulheta no canto
          const dx0 = x0 + 10 + tw(g, 'GPU · pool de KV', 12, 'sans', 600) + 12;
          const hx = x0 + bW - 16;
          const nD = Math.max(0, Math.min(5, Math.floor((hx - 16 - dx0) / 11) + 1));
          for (let k = 0; k < nD; k++) V.dot(g, dx0 + k * 11, yG + 12, 4, V.alpha(C.c, 0.45 + 0.4 * (0.5 + 0.5 * Math.sin(t * 3 + k))));
          g.save(); g.translate(hx, yG + 13); g.rotate(Math.sin(t * 2) * 0.6);
          V.text(g, '⏳', 0, 5, { size: 15, align: 'center' }); g.restore();
        }
        // tempos
        const tx0 = narrow ? x0 : x0 + colW + 24, ty0 = narrow ? yL2 + boxH + 24 : yG + 6;
        const tW = narrow ? w - 28 : w - tx0 - 16;
        V.text(g, 'tempo até o 1º token, ~107K tokens', tx0, ty0, { size: 11.5, color: C.muted, maxW: tW });
        const bars = [
          { lab: 're-prefill', v: 65, txt: '~65 s', col: C.c, on: S.sc === 'prefill' || S.sc === 'store' },
          { lab: 'reload LMCache', v: 1.9, txt: '~1,8–1,9 s', col: C.a, on: S.sc === 'reload' },
          { lab: 'servidor morto', v: 70, txt: '∞ (trava)', col: RED(C), on: dead },
        ];
        bars.forEach((b, i) => {
          const y = ty0 + 16 + i * 40;
          V.text(g, b.lab, tx0, y + 10, { size: 11.5, color: b.on ? C.ink : C.muted, weight: b.on ? 600 : 400 });
          const bw = Math.max(4, ((tW - 90) * Math.min(b.v, 65)) / 65);
          V.box(g, tx0, y + 16, bw, 12, { r: 5, fill: V.alpha(b.col, b.on ? 0.85 : 0.3) });
          if (i === 2) hatch(g, tx0, y + 16, bw, 12, V.alpha(C.ink, 0.35), 5);
          V.text(g, b.txt, tx0 + bw + 8, y + 26, { size: 12, font: 'mono', weight: 600, color: b.on ? C.ink : C.muted });
        });
        const msg = ({ store: 'o prefill acontece normalmente e os chunks são gravados na L1 e na L2', reload: 'depois do reinício os chunks voltam da RAM/disco: ~35× mais rápido que recalcular', prefill: 'sem conector: cada reinício joga fora o cache e refaz o prefill', dead: 'o engine não cai nem recalcula; quem só olha containers vê um par “saudável”' })[S.sc];
        const my = narrow ? ty0 + 16 + 3 * 40 + 8 : Math.max(yL2 + boxH + 22, ty0 + 150);
        const mh = wrap(g, msg, narrow ? x0 : tx0, my, narrow ? w - 28 : tW, { size: 12, weight: 500, color: C.ink });
        fitH(H || { canvas: fig.querySelector('canvas') }, narrow ? my + mh + 8 : Math.max(yL2 + boxH + 40, my + mh + 8));
      });
    }

    /* =====================================================================
       Figura 10 — bancada frio/quente
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-bench');
      const S = { proto: 'naive', g: 3584, n: 8000 };
      const R = 1000, T0 = 0.3, ADD = 60;
      const reqs = () => {
        const out = []; const nW = S.n + ADD;
        for (let r = 0; r < 3; r++) {
          let hc = 0;
          if (r > 0 && S.proto === 'naive') hc = Math.floor((S.n - 1) / S.g) * S.g;
          let hw = Math.floor(Math.min(S.n, nW - 1) / S.g) * S.g;
          if (r > 0 && S.proto === 'naive') hw = Math.floor((nW - 1) / S.g) * S.g;
          out.push({ r, kind: 'frio', n: S.n, hit: hc, ttft: T0 + (S.n - hc) / R, bad: hc > 0 });
          out.push({ r, kind: 'quente', n: nW, hit: hw, ttft: T0 + (nW - hw) / R, eff: hw / (Math.floor(nW / S.g) * S.g || 1) });
        }
        return out;
      };
      let H = null;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Protocolo', options: [{ v: 'naive', label: 'ingênuo' }, { v: 'salt', label: 'salt único' }, { v: 'reset', label: 'só /reset_prefix_cache' }], value: 'naive', onChange: (v) => { S.proto = v; if (H) H.redraw(); } });
      UI.seg(ctl, { label: 'Grão', options: [{ v: 3584, label: 'GLM 3.584' }, { v: 1664, label: 'Qwen 1.664' }, { v: 256, label: 'DS 256' }], value: 3584, onChange: (v) => { S.g = v; if (H) H.redraw(); } });
      UI.slider(ctl, { label: 'Prompt frio', min: 4000, max: 20000, step: 500, value: S.n, fmt: (v) => nf(v) + ' tok', onInput: (v) => { S.n = v; if (H) H.redraw(); } });
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const Q = reqs(); const narrow = w < 640;
        const hi = Math.floor(t / 1.6) % Q.length;
        const chartW = narrow ? w - 28 : w * 0.56, cx0 = 14, cy0 = 34, chH = narrow ? 170 : 250;
        const maxT = T0 + (S.n + ADD) / R;
        V.text(g, 'TTFT por requisição (3 execuções de um par frio → quente)', cx0, 18, { size: 11.5, color: C.muted, maxW: chartW });
        const slot = chartW / 3, bw = Math.min(46, slot / 2 - 14);
        Q.forEach((q, i) => {
          const x = cx0 + q.r * slot + (q.kind === 'frio' ? slot / 2 - bw - 4 : slot / 2 + 4);
          const bh = (chH - 40) * (q.ttft / maxT);
          const y = cy0 + chH - 24 - bh;
          const col = q.kind === 'frio' ? (q.bad ? RED(C) : C.c) : C.a;
          V.box(g, x, y, bw, bh, { r: 5, fill: V.alpha(col, i === hi ? 0.95 : 0.6), stroke: i === hi ? C.ink : null, lw: 1.5 });
          if (q.bad) hatch(g, x, y, bw, bh, V.alpha(C.ink, 0.3), 6);
          V.text(g, sf(q.ttft, 1) + ' s', x + bw / 2, y - 6, { size: 11.5, font: 'mono', weight: 600, align: 'center', color: C.ink });
          V.text(g, q.kind === 'frio' ? 'F' : 'Q', x + bw / 2, cy0 + chH - 10, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
        });
        for (let r = 0; r < 3; r++) V.text(g, 'exec. ' + (r + 1), cx0 + r * slot + slot / 2, cy0 + chH + 8, { size: 11.5, align: 'center', color: C.muted });
        V.line(g, cx0, cy0 + chH - 22, cx0 + chartW, cy0 + chH - 22, { color: V.alpha(C.muted, 0.5), lw: 1 });
        // painel usage
        const px = narrow ? cx0 : cx0 + chartW + 24, py = narrow ? cy0 + chH + 30 : cy0 + 6;
        const pw = narrow ? w - 28 : w - px - 14;
        const q = Q[hi];
        V.box(g, px, py, pw, 150, { r: 10, fill: V.alpha(C.b, 0.08), stroke: V.alpha(C.b, 0.5), lw: 1 });
        const lines = [
          [`// exec. ${q.r + 1}, turno ${q.kind}`, C.muted],
          ['"usage": {', C.ink],
          [`  "prompt_tokens": ${q.n},`, C.ink],
          ['  "prompt_tokens_details": {', C.ink],
          [`    "cached_tokens": ${q.hit} } }`, q.hit ? C.a : C.ink],
          [`Δ vllm:prefix_cache_queries = ${q.n}`, C.muted],
          [`Δ vllm:prefix_cache_hits    = ${q.hit}`, q.hit ? C.a : C.muted],
        ];
        lines.forEach(([s, col], k) => V.text(g, s, px + 10, py + 20 + k * 18, { size: 11.5, font: 'mono', color: col, maxW: pw - 16 }));
        const colds = Q.filter((x) => x.kind === 'frio'), warms = Q.filter((x) => x.kind === 'quente');
        const meanCold = colds.reduce((s, x) => s + x.ttft, 0) / 3, trueCold = T0 + S.n / R;
        const eff = warms.reduce((s, x) => s + x.eff, 0) / 3;
        const sy = py + 170;
        let y4 = sy;
        y4 += wrap(g, `TTFT frio médio medido: ${sf(meanCold, 2)} s · verdadeiro: ${sf(trueCold, 2)} s`, px, y4, pw, { size: 12, font: 'mono', weight: 600, color: meanCold < trueCold - 0.01 ? RED(C) : C.a });
        y4 += wrap(g, `hit_efficiency dos quentes: ${sf(eff, 2)} (1,00 = modelo de páginas respeitado)`, px, y4 + 2, pw, { size: 11.5, font: 'mono', color: C.ink }) + 2;
        y4 += wrap(g, S.proto === 'naive' ? '⚠ a partir da 2ª execução o “frio” já acerta o cache' : S.proto === 'reset' ? 'reset exige GLM53_EXPOSE_CACHE_RESET=1 e responde false enquanto houver blocos presos' : 'um salt por invocação mantém cada frio frio de verdade', px, y4 + 2, pw, { size: 11.5, color: S.proto === 'naive' ? RED(C) : C.muted });
        fitH(H || { canvas: fig.querySelector('canvas') }, narrow ? y4 + 10 : Math.max(cy0 + chH + 30, y4 + 10));
      });
    }

    /* =====================================================================
       Figura 11 — TensorFold: o primeiro token (estado + linha de logits)
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-tf-replay');
      const L = {
        eng: 'Motor', engTF: 'TensorFold', engV: 'vLLM · APC',
        grid: 'fim do prompt na grade de 64', size: 'Prompt', resend: '▶ reenviar',
        rRedo: 'refazer', rPre: 'prefill estimado',
        rowA: '1ª vez · prefill do zero', rowB: 'reenvio idêntico',
        zoom: (z) => `últimos ${nf(z)}`,
        model: '45 camadas', noRun: 'não roda',
        chart: 'logit + g(chave, posição, token)', chartVA: 'logit + ruído do sorteio (sem chave)', chartPhoto: 'logits da foto + o mesmo ruído', chartV: 'logit recalculado + ruído novo',
        tok: '1º token',
        photo: (b) => `estado @ ${nf(b)}`, photoSub: 'KDA · conv · DFlash2', photoLg: '+ linha de logits',
        pages: 'blocos com hash; estado KDA a cada 3.584', noLogits: 'logits não guardados',
        redo: (r) => `refaz ${nf(r)}`,
        eqTF: '= mesmo token, mesmos bits', eqV: '= mesmo token (desta vez)', neV: '≠ outro sorteio',
        noteTFr: 'Replay: o estado e a linha de logits do fim do prompt estavam na foto, então nada passa pelas camadas. Medido na receita: 64k idêntico em menos de 0,07 s (34 s na 1ª vez).',
        noteTFo: (b, r) => `Fim fora da grade: o estado fica em B = ${nf(b)}, sem linha de logits. O reenvio refaz ${nf(r)} tokens (no máximo 63) e chega aos mesmos bits de um prefill do zero.`,
        noteV: (hh, r) => `APC simplificado: acerto ≤ n−1 numa fronteira de 3.584 (${nf(hh)}); sem logits guardados, ${nf(r)} tokens passam pelas 45 camadas de novo. Sem chave nem seed, o ruído do sorteio é novo a cada envio; em greedy, o token só muda se os últimos bits dos logits desempatarem outro.`,
        vocab: ['Claro', 'Sim', 'Olá', 'Bem', 'Não', 'O'],
      };
      const LG = [2.0, 1.5, 1.1, 0.6, 0.1, -0.5];
      const gum = (hs) => -Math.log(-Math.log(((hs >>> 0) + 0.5) / 4294967296));
      const S = { eng: 'tf', k: 64, grid: true, t0: 0, off: 6.6 };
      const calc = () => {
        const n = S.k * 1024 + (S.grid ? 0 : 37);
        const B = Math.floor(n / 64) * 64;
        const tf = S.eng === 'tf';
        const replay = tf && n % 64 === 0;
        const hit = tf ? (replay ? n : B) : Math.floor((n - 1) / 3584) * 3584;
        const r = n - hit;
        const key = fnv('prompt:' + n);
        const nA = LG.map((_, i) => gum(fnv(key + ':' + n + ':' + i)));
        const nB = tf ? nA : LG.map((_, i) => gum(fnv('lote:' + key + ':' + n + ':' + i)));
        const arg = (nz) => { let b = 0; for (let i = 1; i < LG.length; i++) if (LG[i] + nz[i] > LG[b] + nz[b]) b = i; return b; };
        return { n, B, tf, replay, hit, r, sec: r / (tf ? 1950 : 1500), nA, nB, wA: arg(nA), wB: arg(nB), Z: tf ? 256 : 4096 };
      };
      let H = null, ra = null, rb = null;
      const upd = (restart) => {
        const d = calc();
        if (ra) ra.set(nf(d.r) + ' tok', d.r === 0 ? 'ok' : d.r > 1000 ? 'bad' : 'warn');
        if (rb) rb.set(d.r === 0 ? '0 s' : sf(d.sec, 2) + ' s');
        if (H) { S.t0 = H.t; S.off = restart === true ? 0 : 6.6; H.redraw(); }
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: L.eng, options: [{ v: 'tf', label: L.engTF }, { v: 'vllm', label: L.engV }], value: 'tf', onChange: (v) => { S.eng = v; upd(); } });
      UI.slider(ctl, { label: L.size, min: 8, max: 128, step: 8, value: S.k, fmt: (v) => v + 'k', onInput: (v) => { S.k = v; upd(); } });
      UI.toggle(ctl, { label: L.grid, value: true, onChange: (v) => { S.grid = v; upd(); } });
      UI.button(ctl, L.resend, () => upd(true), { primary: true });
      ra = UI.readout(ctl, L.rRedo, '—');
      rb = UI.readout(ctl, L.rPre, '—');
      const camera = (g, x, y, col, a) => {
        g.save(); g.globalAlpha = a;
        V.box(g, x - 9, y - 6, 18, 12, { r: 3, fill: V.alpha(col, 0.85) });
        g.fillRect(x - 4, y - 9, 7, 3);
        g.beginPath(); g.arc(x, y, 3.4, 0, Math.PI * 2); g.fillStyle = V.C().bg; g.fill();
        g.restore();
      };
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const d = calc(); const narrow = w < 620; const pad = 14;
        const P = 8, tt = (Math.max(0, t - S.t0) + S.off) % P;
        // linha do tempo: 1ª vez (0–2,7 s), reenvio (3,4 s em diante), segura o quadro final até 8 s
        const pWhole = V.clamp(tt / 1.4, 0, 1), pTail = V.clamp((tt - 1.4) / 0.6, 0, 1);
        const lgA = V.clamp((tt - 2.0) / 0.6, 0, 1), tkA = tt >= 2.7;
        const b0 = 3.4, run = d.r > 0;
        const pB = run ? V.clamp((tt - b0) / 1.3, 0, 1) : (tt >= b0 ? 1 : 0);
        const lgB = V.clamp((tt - (run ? b0 + 1.4 : b0 + 0.5)) / 0.6, 0, 1);
        const tkB = lgB >= 1 && tt >= (run ? b0 + 2.1 : b0 + 1.2);
        const eqOn = tkB && tt >= (run ? b0 + 2.4 : b0 + 1.5);
        const v0 = d.n - d.Z;
        const geo = (y0) => {
          if (!narrow) {
            const inner = w - 2 * pad;
            const tapeW = Math.round(inner * 0.33), mW = Math.max(64, Math.round(inner * 0.12)), cW = Math.round(inner * 0.3);
            const T = { x: pad, y: y0 + 46, w: tapeW, h: 24 };
            const M = { x: T.x + tapeW + 34, y: y0 + 26, w: mW, h: 80 };
            const Q = { x: M.x + mW + 38, y: y0 + 24, w: cW, h: 82 };
            const K = { x: Q.x + cW + 30, y: y0 + 66 };
            return { y0, T, M, Q, K, h: 134 };
          }
          const T = { x: pad, y: y0 + 46, w: w - 2 * pad, h: 22 };
          const M = { x: pad, y: y0 + 114, w: 64, h: 74 };
          const Q = { x: pad + 90, y: y0 + 114, w: w - 2 * pad - 90, h: 72 };
          const K = { x: Q.x + 4, y: y0 + 222 };
          return { y0, T, M, Q, K, h: 238 };
        };
        const bridgeH = narrow ? 96 : 86;
        const A = geo(4), Bg = geo(4 + A.h + bridgeH);
        const XT = (G2, tok) => G2.T.x + ((tok - v0) / d.Z) * G2.T.w;
        // ---------------- fita (prompt inteiro + zoom no fim)
        const tape = (G2, row) => {
          const T = G2.T, y0 = G2.y0;
          V.text(g, row === 'A' ? L.rowA : L.rowB, pad, y0 + 13, { size: 12.5, weight: 700, color: C.ink });
          const wy = y0 + 24, ww = T.w, wx = T.x;
          g.fillStyle = V.alpha(C.muted, 0.18); g.fillRect(wx, wy, ww, 6);
          if (row === 'A') { g.fillStyle = V.alpha(C.a, 0.85); g.fillRect(wx, wy, ww * pWhole, 6); }
          else {
            const hf = d.hit / d.n;
            g.fillStyle = V.alpha(C.a, 0.3); g.fillRect(wx, wy, ww * hf, 6);
            hatch(g, wx, wy, ww * hf, 6, V.alpha(C.ink, 0.35), 4);
            if (run) { g.fillStyle = V.alpha(C.c, 0.9); g.fillRect(wx + ww * hf, wy, Math.max(1.5, ww * (1 - hf)), 6); }
          }
          const zx = wx + ww * (v0 / d.n), zx1 = wx + ww;
          g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1; g.strokeRect(Math.min(zx, zx1 - 3) - 0.5, wy - 1.5, Math.max(3, zx1 - zx) + 1, 9);
          V.line(g, Math.min(zx, zx1 - 3), wy + 8, T.x, T.y, { color: V.alpha(C.muted, 0.55), lw: 1, dash: [2, 3] });
          V.line(g, zx1, wy + 8, T.x + T.w, T.y, { color: V.alpha(C.muted, 0.55), lw: 1, dash: [2, 3] });
          // zoom
          g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(T.x, T.y, T.w, T.h);
          if (row === 'A') {
            g.fillStyle = V.alpha(C.a, 0.8); g.fillRect(T.x, T.y, T.w * pTail, T.h);
          } else {
            const hx = XT(G2, d.hit);
            g.fillStyle = V.alpha(C.a, 0.28); g.fillRect(T.x, T.y, hx - T.x, T.h);
            hatch(g, T.x, T.y, hx - T.x, T.h, V.alpha(C.ink, 0.32), 6);
            if (run) { g.fillStyle = V.alpha(C.c, 0.9); g.fillRect(hx, T.y, (T.x + T.w - hx) * pB, T.h); }
          }
          g.strokeStyle = V.alpha(C.muted, 0.8); g.lineWidth = 1; g.strokeRect(T.x + 0.5, T.y + 0.5, T.w - 1, T.h - 1);
          const stepT = d.tf ? 64 : 3584;
          for (let k = Math.ceil(v0 / stepT) * stepT; k <= d.n; k += stepT) {
            const x = XT(G2, k);
            V.line(g, x, T.y - 3, x, T.y + T.h + 3, { color: V.alpha(C.ink, d.tf ? 0.4 : 0.7), lw: d.tf ? 1 : 1.6, dash: d.tf ? [2, 3] : null });
            if (!d.tf && row === 'A') {
              const on = pTail >= (k - v0) / d.Z;
              g.beginPath(); g.moveTo(x, T.y - 9); g.lineTo(x + 5, T.y - 4); g.lineTo(x, T.y + 1); g.lineTo(x - 5, T.y - 4); g.closePath();
              g.fillStyle = on ? C.b : V.alpha(C.b, 0.3); g.fill();
            }
          }
          if (d.tf && row === 'A') {
            const bx = XT(G2, d.B), on = pTail >= (d.B - v0) / d.Z;
            camera(g, bx, T.y - 9, C.d, on ? 1 : 0.25);
            if (on && pTail < 1) V.glow(g, bx, T.y - 9, 16, C.d);
          }
          V.text(g, '… ' + nf(v0), T.x, T.y + T.h + 15, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'n = ' + nf(d.n), T.x + T.w, T.y + T.h + 15, { size: 11, font: 'mono', weight: 600, color: C.ink, align: 'right' });
          const zl = L.zoom(d.Z), zw = tw(g, zl, 11);
          const lw2 = tw(g, '… ' + nf(v0), 11, 'mono') + 10, rw2 = tw(g, 'n = ' + nf(d.n), 11, 'mono', 600) + 10;
          if (zw < T.w - lw2 - rw2) V.text(g, zl, T.x + lw2 + (T.w - lw2 - rw2) / 2, T.y + T.h + 15, { size: 11, color: C.muted, align: 'center' });
          if (row === 'B') {
            const hx = XT(G2, d.hit);
            const lab = run ? L.redo(d.r) : L.redo(0);
            const hw = (tw(g, lab, 11, 'mono', 600) + 12) / 2;
            const px = V.clamp(run ? (hx + T.x + T.w) / 2 : T.x + T.w - 30, T.x + hw, T.x + T.w - hw);
            pill(g, px, T.y - 10, lab, run ? C.c : C.a, { align: 'center', size: 11, a: 0.3 });
          }
        };
        // ---------------- modelo: 45 faixas (KDA e DSA; ordem ilustrativa)
        const model = (G2, row) => {
          const M = G2.M, n = 45, sw = M.w / n;
          const dim = row === 'B' && !run;
          for (let i = 0; i < n; i++) {
            const col = i % 4 === 3 ? C.a : C.b;
            g.fillStyle = V.alpha(col, dim ? 0.12 : 0.6); g.fillRect(M.x + i * sw + 0.4, M.y, Math.max(1, sw - 0.8), M.h);
          }
          g.strokeStyle = V.alpha(C.muted, 0.6); g.lineWidth = 1; g.strokeRect(M.x - 0.5, M.y - 0.5, M.w + 1, M.h + 1);
          let prog = -1;
          if (row === 'A' && tt < 2.0) prog = ((tt / 2.0) * 5) % 1;
          if (row === 'B' && run && pB > 0 && pB < 1) prog = pB;
          if (prog >= 0) {
            const xi = M.x + prog * M.w;
            V.glow(g, xi, M.y + M.h / 2, 18, C.d);
            g.fillStyle = V.alpha(C.d, 0.9); g.fillRect(xi - 1.5, M.y - 3, 3, M.h + 6);
          }
          V.text(g, L.model, M.x + M.w / 2, M.y + M.h + 14, { size: 11, align: 'center', color: C.muted });
          if (dim) {
            V.box(g, M.x + M.w / 2 - 34, M.y + M.h / 2 - 11, 68, 22, { r: 11, fill: V.alpha(C.bg, 0.85), stroke: V.alpha(C.a, 0.8), lw: 1 });
            V.text(g, L.noRun, M.x + M.w / 2, M.y + M.h / 2 + 4, { size: 11.5, weight: 700, align: 'center', color: C.ink });
          }
          // setas
          const T = G2.T, Q = G2.Q;
          if (!narrow) {
            V.arrow(g, T.x + T.w + 6, T.y + T.h / 2, M.x - 6, M.y + M.h / 2, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 7 });
            V.arrow(g, M.x + M.w + 6, M.y + M.h / 2, Q.x - 8, M.y + M.h / 2, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 7 });
          } else {
            V.arrow(g, M.x + M.w / 2, T.y + T.h + 22, M.x + M.w / 2, M.y - 5, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 7 });
            V.arrow(g, M.x + M.w + 5, M.y + M.h / 2, Q.x - 6, M.y + M.h / 2, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 7 });
          }
        };
        // ---------------- logits + ruído chaveado
        const chart = (G2, row) => {
          const Q = G2.Q, nz = row === 'A' ? d.nA : d.nB, win = row === 'A' ? d.wA : d.wB;
          const prog = row === 'A' ? lgA : lgB, show = row === 'A' ? tkA : tkB;
          const nb = LG.length, bw = Q.w / nb, u = (Q.h * 0.6) / 4.6, base = Q.y + Q.h * 0.62;
          if (row === 'B' && d.replay) V.text(g, L.chartPhoto, Q.x, Q.y - 6, { size: 11, weight: 700, color: C.d, maxW: Q.w });
          else if (row === 'B' && !d.tf) V.text(g, L.chartV, Q.x, Q.y - 6, { size: 11, weight: 600, color: C.muted, maxW: Q.w });
          else V.text(g, d.tf ? L.chart : L.chartVA, Q.x, Q.y - 6, { size: 11, color: C.muted, maxW: Q.w });
          V.line(g, Q.x, base, Q.x + Q.w, base, { color: V.alpha(C.muted, 0.6), lw: 1 });
          for (let i = 0; i < nb; i++) {
            const x = Q.x + i * bw + bw * 0.2, bwi = bw * 0.6;
            const lv = LG[i] * prog;
            const tot = V.clamp(LG[i] + nz[i], -2.6, 4.6);
            const gv = (tot - LG[i]) * V.clamp(prog * 2 - 1, 0, 1);
            const y1 = base - lv * u, y2 = y1 - gv * u;
            g.fillStyle = V.alpha(C.a, 0.78); g.fillRect(x, Math.min(base, y1), bwi, Math.abs(y1 - base));
            g.fillStyle = V.alpha(C.d, 0.8); g.fillRect(x, Math.min(y1, y2), bwi, Math.abs(y2 - y1));
            const isW = show && win === i;
            if (isW) { g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(x - 2, Math.min(base, y1, y2) - 2, bwi + 4, Math.abs(Math.min(base, y1, y2) - Math.max(base, y1, y2)) + 4); }
            V.text(g, L.vocab[i], x + bwi / 2, Q.y + Q.h + 13, { size: 11, align: 'center', color: isW ? C.ink : C.muted, weight: isW ? 700 : 400 });
          }
          // ficha do token
          const K = G2.K;
          if (!narrow) {
            V.arrow(g, Q.x + Q.w + 4, K.y, K.x - 4, K.y, { color: V.alpha(C.muted, 0.8), lw: 1.4, head: 7 });
            V.text(g, L.tok, K.x, K.y - 16, { size: 11, color: C.muted });
          } else V.text(g, L.tok + ':', K.x - 4, K.y + 4, { size: 11, color: C.muted, align: 'left' });
          const kx = narrow ? K.x + tw(g, L.tok + ':', 11) + 6 : K.x;
          if (show) pill(g, kx, K.y, '"' + L.vocab[win] + '"', C.e, { size: 12, a: 0.3 });
          else { g.setLineDash([3, 3]); V.rr(g, kx, K.y - 10, 44, 20, 10); g.strokeStyle = V.alpha(C.muted, 0.7); g.lineWidth = 1; g.stroke(); g.setLineDash([]); V.text(g, '?', kx + 22, K.y + 4, { size: 12, align: 'center', color: C.muted }); }
        };
        [[A, 'A'], [Bg, 'B']].forEach(([G2, row]) => { tape(G2, row); model(G2, row); chart(G2, row); });
        // ---------------- ponte: a foto (TF) ou os blocos (vLLM)
        const by = A.y0 + A.h + 6;
        let rightX = pad;
        if (d.tf) {
          const pw = 116, phh = 70, bx = XT(A, d.B);
          const px = narrow ? pad : V.clamp(bx - pw / 2, pad, A.T.x + A.T.w - pw);
          const on = pTail >= (d.B - v0) / d.Z;
          g.save(); g.globalAlpha = on ? 1 : 0.3;
          if (!narrow) V.line(g, bx, A.T.y + A.T.h + 18, px + pw / 2, by, { color: V.alpha(C.d, 0.7), lw: 1, dash: [2, 3] });
          V.box(g, px, by, pw, phh, { r: 6, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.ink, 0.55), lw: 1.2 });
          V.box(g, px + 6, by + 6, pw - 12, 38, { r: 3, fill: V.alpha(C.b, 0.14) });
          for (let m = 0; m < 3; m++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
            const v = ((fnv('s' + m + i + j + d.B) >>> 0) % 1000) / 1000;
            g.fillStyle = V.cmap(v, 'seq'); g.fillRect(px + 11 + m * 15 + j * 4, by + 12 + i * 4, 4, 4);
          }
          g.beginPath(); g.arc(px + 65, by + 25, 8, 0, Math.PI * 2); g.strokeStyle = C.c; g.lineWidth = 3; g.stroke();
          if (d.replay) for (let i = 0; i < LG.length; i++) { const bh = 4 + (LG[i] + 0.6) * 5; g.fillStyle = V.alpha(C.a, 0.85); g.fillRect(px + 82 + i * 4.5, by + 38 - bh, 3.2, bh); }
          V.text(g, L.photo(d.B), px + pw / 2, by + 60, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
          g.restore();
          const tx = px + pw + 10;
          V.text(g, L.photoSub, tx, by + 26, { size: 11, color: C.muted });
          if (d.replay) V.text(g, L.photoLg, tx, by + 42, { size: 11, weight: 600, color: C.d });
          rightX = tx + Math.max(tw(g, L.photoSub, 11), d.replay ? tw(g, L.photoLg, 11, 'sans', 600) : 0) + 16;
          // seta animada da foto para o reenvio (só no largo; no estreito ela cruzaria os textos)
          const tgt = d.replay ? { x: Bg.Q.x - 10, y: Bg.Q.y - 10 } : { x: Bg.M.x + Bg.M.w / 2, y: Bg.M.y - 6 };
          if (!narrow && (tt >= b0 - 0.2 || tt < 0.01)) V.arrow(g, px + pw * 0.7, by + phh + 2, tgt.x, tgt.y, { color: C.d, lw: 1.8, head: 8, dash: [5, 4], curve: 18 });
        } else {
          const bx0 = narrow ? pad : A.M.x, wr = narrow ? w - 2 * pad : Math.max(160, w - pad - 200 - bx0);
          let yy = by + 16;
          yy += wrap(g, L.pages, bx0, yy, wr, { size: 11.5, color: C.ink, weight: 600 });
          yy += wrap(g, L.noLogits, bx0, yy, wr, { size: 11.5, color: RED(C), weight: 600 });
          const hx = XT(A, d.hit);
          if (!narrow && (tt >= b0 - 0.2 || tt < 0.01)) V.arrow(g, hx, A.T.y + A.T.h + 20, XT(Bg, d.hit), Bg.T.y - 22, { color: C.b, lw: 1.6, head: 7, dash: [5, 4] });
          rightX = bx0 + wr + 14;
        }
        // selo de igualdade (no estreito, logo abaixo do reenvio)
        let ny = Bg.y0 + Bg.h + 8;
        {
          const same = d.wA === d.wB;
          const s = d.tf ? L.eqTF : same ? L.eqV : L.neV;
          const col = d.tf ? C.a : same ? C.d : RED(C);
          const sw2 = tw(g, s, 12, 'sans', 700) + 22;
          const ex = narrow ? pad : Math.max(rightX, w - pad - sw2), ey = narrow ? ny + 14 : by + 30;
          const bwd = Math.min(sw2, w - pad - ex);
          if (eqOn) {
            if (!narrow) V.line(g, A.K.x + 24, A.K.y + 12, A.K.x + 24, Bg.K.y - 12, { color: V.alpha(col, 0.7), lw: 1.5, dash: [3, 3] });
            V.box(g, ex, ey - 13, bwd, 26, { r: 13, fill: C.bg });
            V.box(g, ex, ey - 13, bwd, 26, { r: 13, fill: V.alpha(col, 0.2), stroke: col, lw: 1.4 });
            V.text(g, s, ex + bwd / 2, ey + 4, { size: 12, weight: 700, align: 'center', color: C.ink, maxW: bwd - 12 });
          }
          if (narrow) ny += 32;
        }
        // nota
        const note = d.tf ? (d.replay ? L.noteTFr : L.noteTFo(d.B, d.r)) : L.noteV(d.hit, d.r);
        const nh = wrap(g, note, pad, ny + 12, w - 2 * pad, { size: 11.5, color: C.muted });
        fitH(H || { canvas: fig.querySelector('canvas') }, ny + nh + 14);
      });
      upd();
    }

    /* =====================================================================
       Figura 12 — TensorFold: o álbum de estados guardados (LRU por conta)
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-tf-album');
      const L = {
        convs: 'Conversas', cap: 'Lugares', cap8: '8 (TensorFold)', cap32: '32 (receita)', spr: 'Estados por pedido',
        auto: 'tocar sozinho', next: '⏭ próximo pedido',
        rHit: 'turnos quentes retomados', rUsed: 'lugares ocupados', rMem: 'reserva fixa',
        conv: (c) => 'conversa ' + c,
        chip: (k, n) => (k ? `turno ${k} · ${sf(n / 1000, 1)}k` : 'ainda calada'),
        album: (u, c) => `álbum de estados · ${u}/${c} lugares · ordem LRU (o 1º sai antes)`,
        evicted: 'despejados neste pedido:', none: 'nenhum',
        spark: 'tokens de prefill por pedido (últimos 24); hachurado = turno quente que releu o histórico',
        hit: (c, k, from, pos, pre, s) => `${c} · turno ${k}: retoma de ${from} (${nf(pos)}) e faz o prefill de ${nf(pre)} tokens ≈ ${s} s`,
        sysOnly: (c, k, pre, s) => `${c} · turno ${k}: só a foto do system serve; relê o histórico, ${nf(pre)} tokens ≈ ${s} s`,
        miss: (c, k, pre, s) => `${c} · turno ${k}: nenhuma foto serve; prefill de ${nf(pre)} tokens ≈ ${s} s`,
        first: (c, pre, s, sys) => `${c} · turno 1: ${sys ? 'conversa nova retoma da foto do system (S)' : 'conversa nova sem foto'}; prefill de ${nf(pre)} tokens ≈ ${s} s`,
        lone: 'LONE=1: o pedido se mudou para as primeiras linhas do pool e despejou as fotos das outras conversas',
        fresh: 'novo',
        legSys: 'S = fim do system prompt', legMiss: 'releu o histórico',
      };
      const SYS = 7900, FIRST = 600, STEP = 1500, RATE = 1950, GRID = 64, LEAST = 256;
      const gp = (n) => Math.floor(n / GRID) * GRID;
      const sysP = gp(SYS);
      const nOf = (k) => SYS + FIRST + (k - 1) * STEP;
      const LET = 'ABCDEF';
      const lab = (e) => (e.sys ? 'S' : LET[e.conv] + e.turn);
      const S = { K: 3, cap: 32, spr: 3, shared: true, lone: false, auto: true };
      let sim = null, anim = null, H = null, ra = null, rb = null, rc = null;
      function step(rec) {
        const c = sim.next % S.K; sim.next++;
        const k = ++sim.turn[c]; const n = nOf(k);
        // só a foto do system e as da própria conversa são prefixos deste prompt
        let best = null;
        for (const e of sim.list) if ((e.sys || e.conv === c) && e.pos < n && (!best || e.pos > best.pos)) best = e;
        const cut = best ? best.pos : 0;
        if (best) { sim.list.splice(sim.list.indexOf(best), 1); sim.list.push(best); } // retomar = tocar (LRU)
        const point = gp(n);
        const pts = [];
        if (S.shared) {
          if (cut < sysP && sysP - cut >= LEAST) pts.push({ pos: sysP, sys: true });
          const extra = Math.min(2, S.spr - 1) - pts.length;
          const lo = pts.length ? pts[0].pos : cut;
          for (let j = 1; j <= extra; j++) {
            const p = gp(lo + ((point - lo) * j) / (extra + 1));
            const prev = pts.length ? pts[pts.length - 1].pos : cut;
            if (p - prev >= LEAST && p < point) pts.push({ pos: p, extra: true });
          }
        }
        const fresh = pts.map((p) => ({ id: ++sim.uid, conv: c, turn: k, pos: p.pos, sys: !!p.sys, extra: !!p.extra }));
        fresh.push({ id: ++sim.uid, conv: c, turn: k, pos: point });
        fresh.forEach((e) => sim.list.push(e));
        const ev = [];
        if (S.lone) sim.list = sim.list.filter((e) => (e.conv === c ? true : (ev.push(e), false)));
        while (sim.list.length > S.cap) ev.push(sim.list.shift());
        const kind = !best ? 'miss' : best.sys ? 'sys' : 'hit';
        sim.hist.push({ c, k, pre: n - cut, kind, warm: k > 1 });
        if (sim.hist.length > 48) sim.hist.shift();
        if (rec) anim = { t0: H ? H.t : 0, c, k, n, cut, kind, best: best ? best.id : null, bestLab: best ? lab(best) : '', fresh: fresh.map((e) => e.id), ev };
      }
      const upd = () => {
        const warm = sim.hist.filter((x) => x.warm).slice(-Math.max(4, S.K * 2)); // as duas últimas rodadas
        const hits = warm.filter((x) => x.kind === 'hit').length;
        if (ra) { if (warm.length) ra.set(Math.round((100 * hits) / warm.length) + '%', hits === warm.length ? 'ok' : hits === 0 ? 'bad' : 'warn'); else ra.set('—'); }
        if (rb) rb.set(sim.list.length + '/' + S.cap);
        if (rc) rc.set('≈ ' + (S.cap * 45 >= 1024 ? sf((S.cap * 45) / 1024, 1) + ' GiB' : nf(S.cap * 45) + ' MiB'));
        if (H) H.redraw();
      };
      const reset = () => {
        sim = { list: [], turn: [0, 0, 0, 0, 0, 0], next: 0, hist: [], uid: 0 };
        for (let i = 0; i < S.K * 5 - 1; i++) step(false); // aquece até o regime
        step(true); anim.t0 = -99;
        upd();
      };
      const doStep = () => { if (sim.turn.some((k) => k >= 30)) reset(); else { step(true); upd(); } };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: L.convs, min: 1, max: 6, step: 1, value: S.K, onInput: (v) => { S.K = v; reset(); } });
      UI.seg(ctl, { label: L.cap, options: [{ v: 8, label: L.cap8 }, { v: 32, label: L.cap32 }], value: 32, onChange: (v) => { S.cap = v; reset(); } });
      UI.seg(ctl, { label: L.spr, options: [{ v: 1, label: '1' }, { v: 2, label: '2' }, { v: 3, label: '3' }], value: 3, onChange: (v) => { S.spr = v; reset(); } });
      UI.toggle(ctl, { label: 'SHARED_PREFIX=1', value: true, onChange: (v) => { S.shared = v; reset(); } });
      UI.toggle(ctl, { label: 'TF_GLM_MULTI_LONE=1', value: false, onChange: (v) => { S.lone = v; reset(); } });
      UI.toggle(ctl, { label: L.auto, value: true, onChange: (v) => { S.auto = v; } });
      UI.button(ctl, L.next, () => doStep(), { primary: true });
      ra = UI.readout(ctl, L.rHit, '—');
      rb = UI.readout(ctl, L.rUsed, '—');
      rc = UI.readout(ctl, L.rMem, '—');
      UI.legend(ctl, [[cv('b'), L.legSys], ['var(--red)', L.legMiss]]);
      V.every(1900, () => {
        if (!S.auto || !sim) return;
        const r = fig.getBoundingClientRect();
        if (r.bottom < 0 || r.top > window.innerHeight) return;
        doStep();
      });
      const colOf = (C, e) => (e.sys ? C.b : [C.a, C.c, C.d, C.f, C.e, C.muted][e.conv]);
      H = V.fig(fig, (g, w, h, t) => {
        if (!sim) return;
        const C = V.C(); const narrow = w < 620; const pad = 14;
        const ph = anim ? V.clamp((t - anim.t0) / 1.2, 0, 1) : 1;
        let y = 6;
        // fichas das conversas
        const perRow = narrow ? 3 : 6, gap = 8;
        const chipW = Math.min(124, (w - 2 * pad - (perRow - 1) * gap) / perRow), chipH = 40;
        for (let c = 0; c < S.K; c++) {
          const x = pad + (c % perRow) * (chipW + gap), yy = y + Math.floor(c / perRow) * (chipH + gap);
          const act = anim && anim.c === c, col = colOf(C, { conv: c });
          V.box(g, x, yy, chipW, chipH, { r: 10, fill: V.alpha(col, act ? 0.3 : 0.1), stroke: V.alpha(col, act ? 1 : 0.55), lw: act ? 2 : 1 });
          V.text(g, L.conv(LET[c]), x + 9, yy + 16, { size: 12, weight: 700, color: C.ink, maxW: chipW - 16 });
          V.text(g, L.chip(sim.turn[c], nOf(Math.max(1, sim.turn[c]))), x + 9, yy + 32, { size: 11, font: 'mono', color: C.muted, maxW: chipW - 16 });
        }
        y += Math.ceil(S.K / perRow) * (chipH + gap) + 2;
        // o que aconteceu no último pedido
        let msg = '', mcol = C.ink;
        if (anim) {
          const s = sf((anim.n - anim.cut) / RATE, 2), cl = L.conv(LET[anim.c]), pre = anim.n - anim.cut;
          if (anim.k === 1) msg = L.first(cl, pre, s, anim.kind === 'sys');
          else if (anim.kind === 'hit') msg = L.hit(cl, anim.k, anim.bestLab, anim.cut, pre, s);
          else { msg = anim.kind === 'sys' ? L.sysOnly(cl, anim.k, pre, s) : L.miss(cl, anim.k, pre, s); mcol = RED(C); }
        }
        y += wrap(g, msg, pad, y + 13, w - 2 * pad, { size: 12, weight: 600, color: mcol }) + 4;
        if (anim && S.lone && anim.ev.some((e) => e.conv !== anim.c)) y += wrap(g, L.lone, pad, y + 12, w - 2 * pad, { size: 11.5, color: C.d }) + 2;
        // álbum
        const cols = narrow ? 4 : 8, cg = 6, chh = 40;
        const cw = (w - 2 * pad - (cols - 1) * cg) / cols;
        V.text(g, L.album(sim.list.length, S.cap), pad, y + 13, { size: 11.5, color: C.muted, maxW: w - 2 * pad });
        y += 22;
        const card = (e, x, yy, cwi, a, isNew, isHit) => {
          const col = colOf(C, e);
          g.save(); g.globalAlpha = a;
          V.rr(g, x, yy, cwi, chh, 7); g.fillStyle = V.alpha(col, 0.2); g.fill();
          if (e.extra) g.setLineDash([4, 3]);
          g.strokeStyle = V.alpha(col, 0.95); g.lineWidth = isHit ? 2.6 : 1.3; g.stroke(); g.setLineDash([]);
          V.text(g, lab(e), x + 7, yy + 16, { size: 12, weight: 700, color: C.ink });
          V.text(g, sf(e.pos / 1000, 1) + 'k', x + cwi - 6, yy + chh - 7, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          g.restore();
          if (isHit && ph < 1) V.glow(g, x + cwi / 2, yy + chh / 2, cwi * 0.4 * (1 - ph) + 6, col);
          if (isNew && ph < 1 && cwi > 70) pill(g, x + cwi - 6, yy + 11, L.fresh, C.d, { align: 'right', size: 11, a: 0.35, font: 'sans' });
        };
        for (let i = 0; i < S.cap; i++) {
          const x = pad + (i % cols) * (cw + cg), yy = y + Math.floor(i / cols) * (chh + cg);
          const e = sim.list[i];
          if (!e) { g.setLineDash([3, 4]); V.rr(g, x, yy, cw, chh, 7); g.strokeStyle = V.alpha(C.muted, 0.35); g.lineWidth = 1; g.stroke(); g.setLineDash([]); continue; }
          const isNew = !!(anim && anim.fresh.includes(e.id)), isHit = !!(anim && anim.best === e.id);
          card(e, x, yy, cw, isNew ? V.clamp((ph - 0.3) / 0.4, 0, 1) : 1, isNew, isHit);
        }
        y += Math.ceil(S.cap / cols) * (chh + cg) + 2;
        // despejados
        V.text(g, L.evicted, pad, y + 14, { size: 11.5, color: C.muted });
        let ex = pad + tw(g, L.evicted, 11.5) + 10;
        const ev = anim ? anim.ev : [];
        if (!ev.length) V.text(g, L.none, ex, y + 14, { size: 11.5, color: C.muted });
        else {
          const ea = V.clamp((ph - 0.45) / 0.35, 0, 1);
          for (let i = 0; i < ev.length; i++) {
            const wdt = 50;
            if (ex + wdt > w - pad) { V.text(g, '+' + (ev.length - i), ex, y + 14, { size: 11.5, font: 'mono', color: C.muted }); break; }
            const col = colOf(C, ev[i]);
            g.save(); g.globalAlpha = 0.35 + 0.65 * ea;
            V.box(g, ex, y + 2, wdt, 18, { r: 5, fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.6), lw: 1 });
            V.text(g, lab(ev[i]), ex + 6, y + 15, { size: 11, weight: 600, color: C.ink });
            V.text(g, '✕', ex + wdt - 8, y + 15, { size: 11, weight: 700, align: 'center', color: RED(C) });
            g.restore();
            ex += wdt + 5;
          }
        }
        y += 28;
        // histórico de prefill
        y += wrap(g, L.spark, pad, y + 12, w - 2 * pad, { size: 11.5, color: C.muted }) + 2;
        const sh = 54, hs = sim.hist.slice(-24), mx = Math.max(1, ...hs.map((x) => x.pre));
        const bw = (w - 2 * pad) / 24;
        hs.forEach((x, i) => {
          const bh = Math.max(2, ((sh - 4) * x.pre) / mx), bx = pad + i * bw + 1, byy = y + sh - bh;
          const bad = x.warm && x.kind !== 'hit';
          g.fillStyle = V.alpha(bad ? RED(C) : colOf(C, { conv: x.c }), bad ? 0.55 : 0.8); g.fillRect(bx, byy, bw - 2, bh);
          if (bad) hatch(g, bx, byy, bw - 2, bh, V.alpha(C.ink, 0.35), 5);
        });
        V.line(g, pad, y + sh, w - pad, y + sh, { color: V.alpha(C.muted, 0.5), lw: 1 });
        y += sh + 8;
        fitH(H || { canvas: fig.querySelector('canvas') }, y);
      });
      reset();
    }
    /* =====================================================================
       Figura 13 — TensorFold: três andares (pool → pinned → NVMe); refazer × restaurar
       ===================================================================== */
    {
      const fig = root.querySelector('#fig-tf-spill');
      const L = {
        len: 'Comprimento do prompt', mode: 'Caminho', save: 'guardar', load: 'restaurar',
        rRe: 'refazer (prefill)', rLd: 'restaurar do NVMe', rX: 'vantagem', rDisk: 'no disco, por Spark', blocks: 'blocos',
        meas: 'medido', interp: 'interpolado', extrap: 'extrapolado',
        pool: 'pool da GPU', poolSub: 'linhas por token + parte fixa (~91 MB)',
        pin: 'memória pinned', pinSubSave: 'anel de escrita de 256 MiB', pinSubLoad: '16 slots de leitura',
        nvme: 'NVMe local', nvmeSub: (b) => nf(b) + (b === 1 ? ' bloco' : ' blocos') + ' de 2.048 tokens',
        cell: (k) => '1 célula = ' + k + ' blocos',
        saveA1: 'na hora de guardar, a parte fixa e o bloco cortado pelo fim do prompt saem da GPU',
        saveA2: 'o resto vai ao disco aos poucos, 4 blocos entre rodadas, com O_DIRECT',
        copyA: 'o motor copia os blocos lidos para o pool entre rodadas, até 40 ms por intervalo',
        loadA: 'leitura à frente com 8 threads, cada bloco conferido pelo CRC-32',
        decoding: 'os outros streams seguem decodificando (a 260 mil tokens: 46,2–46,3 tok/s, contra 46,4 sem a leitura)',
        survive: 'escrito ao guardar: a conversa sobrevive a despejo, reinício limpo e kill -9',
        yAx: 'segundos até o estado estar pronto (escala log)', xAx: 'tokens do prompt (escala log)',
        busy: 'ao lado de um stream',
        bigLab: 'GLM-5.3 · 3 Sparks (medido)',
        legRe: 'refazer: TTFT de prefill medido (GLM-5.3-Flash, 2 Sparks)', legLd: 'restaurar do NVMe (medido a 40k e 260k)', legBig: 'GLM-5.3, 3 Sparks: 232,2 s × 3,0 s',
        reLab: (s, k) => 'refazer: ' + s + ' (' + k + ')', ldLab: (s, k) => 'restaurar: ' + s + ' (' + k + ')',
      };
      // medidos: TTFT de prefill (README, dois Sparks, clocks em 2.200 MHz) e restauração do NVMe (README, Spill tier; 260k ocioso = 0,40–0,46 s)
      const PRE = [[8219, 4.21], [16407, 8.31], [32790, 16.57], [65563, 33.75], [131099, 71.33], [262170, 159.69], [981841, 967]];
      const R40 = [40000, 0.147], R260 = [260000, 0.43], IDLE = [0.40, 0.46], BUSY = [0.54, 0.61];
      const BIG = { n: 94317, pre: 232.2, rst: 3.0 };
      const KB = 6.2, BLK = 2048, STEPS = 240;
      const NMIN = PRE[0][0], NMAX = PRE[PRE.length - 1][0];
      const SNAP = [8219, 16407, 32790, 40000, 65563, 94317, 131099, 260000, 981841];
      const near = (a, b, tol) => Math.abs(Math.log(a / b)) < tol;
      const nOfI = (i) => {
        const n = Math.exp(Math.log(NMIN) + ((Math.log(NMAX) - Math.log(NMIN)) * i) / STEPS);
        for (const s of SNAP) if (near(n, s, 0.022)) return s;
        return Math.round(n);
      };
      const iOfN = (n) => Math.round((STEPS * Math.log(n / NMIN)) / Math.log(NMAX / NMIN));
      function preAt(n) {
        for (const p of PRE) if (near(n, p[0], 0.012)) return { v: p[1], k: 'meas', at: p[0] };
        let i = 0;
        while (i < PRE.length - 2 && n > PRE[i + 1][0]) i++;
        const [x0, y0] = PRE[i], [x1, y1] = PRE[i + 1];
        const u = Math.log(n / x0) / Math.log(x1 / x0);
        return { v: Math.exp(Math.log(y0) + u * Math.log(y1 / y0)), k: 'interp' };
      }
      const KS = (R260[1] - R40[1]) / (R260[0] - R40[0]);
      function rstAt(n) {
        if (near(n, R40[0], 0.012)) return { v: R40[1], k: 'meas', at: R40[0] };
        if (near(n, R260[0], 0.012)) return { v: R260[1], k: 'meas', at: R260[0] };
        return { v: R40[1] + KS * (n - R40[0]), k: n > R40[0] && n < R260[0] ? 'interp' : 'extrap' };
      }
      const secs = (x) => (x < 1 ? nf(x * 1000) + ' ms' : x < 10 ? sf(x, 2) + ' s' : x < 100 ? sf(x, 1) + ' s' : nf(x) + ' s');
      const size = (n) => { const mb = (n * KB) / 1000; return mb < 1000 ? nf(mb) + ' MB' : sf(mb / 1000, 2) + ' GB'; };
      const S = { i: iOfN(40000), mode: 'load' };
      let H = null, ra = null, rb = null, rc = null, rd = null;
      const upd = () => {
        const n = nOfI(S.i), p = preAt(n), r = rstAt(n);
        if (ra) ra.set(secs(p.v) + ' · ' + L[p.k]);
        if (rb) rb.set(secs(r.v) + ' · ' + L[r.k], r.k === 'extrap' ? 'warn' : r.k === 'meas' ? 'ok' : undefined);
        if (rc) rc.set('×' + nf(p.v / r.v));
        if (rd) rd.set(size(n) + ' · ' + nf(Math.ceil(n / BLK)) + ' ' + L.blocks);
        if (H) H.redraw();
      };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: L.len, min: 0, max: STEPS, step: 1, value: S.i, fmt: (v) => nf(nOfI(v)) + ' tok', onInput: (v) => { S.i = v; upd(); } });
      UI.seg(ctl, { label: L.mode, options: [{ v: 'save', label: L.save }, { v: 'load', label: L.load }], value: 'load', onChange: (v) => { S.mode = v; upd(); } });
      ra = UI.readout(ctl, L.rRe, '—');
      rb = UI.readout(ctl, L.rLd, '—');
      rc = UI.readout(ctl, L.rX, '—');
      rd = UI.readout(ctl, L.rDisk, '—');
      UI.legend(ctl, [[cv('d'), L.legRe], [cv('a'), L.legLd], [cv('c'), L.legBig]]);
      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const narrow = w < 560; const pad = 14;
        const n = nOfI(S.i), nb = Math.ceil(n / BLK), p = preAt(n), r = rstAt(n);
        const load = S.mode === 'load';
        let y = 6;
        /* ---------- os três andares ---------- */
        const gap = narrow ? 16 : 40;
        const bw = (w - 2 * pad - 2 * gap) / 3;
        const tSize = narrow ? 11.5 : 12.5, sSize = 11, lh = 15;
        const tiers = [
          { t: L.pool, s: L.poolSub, col: C.a },
          { t: L.pin, s: load ? L.pinSubLoad : L.pinSubSave, col: C.b },
          { t: L.nvme, s: L.nvmeSub(nb), col: C.e },
        ];
        const lines = tiers.map((q) => wrapLines(g, q.s, bw - 14, sSize));
        const maxL = Math.max(...lines.map((x) => x.length));
        const vizH = 34, bh = 10 + 16 + maxL * lh + 8 + vizH + 10;
        const xs = [pad, pad + bw + gap, pad + 2 * (bw + gap)];
        const ph = (t % 2.4) / 2.4;
        tiers.forEach((q, k) => {
          const x = xs[k];
          V.box(g, x, y, bw, bh, { r: 10, fill: V.alpha(q.col, 0.1), stroke: V.alpha(q.col, 0.75), lw: 1.3 });
          V.text(g, q.t, x + 8, y + 19, { size: tSize, weight: 700, color: C.ink, maxW: bw - 14 });
          lines[k].forEach((ln, j) => V.text(g, ln, x + 8, y + 19 + 16 + j * lh, { size: sSize, color: C.muted }));
          const vy = y + bh - vizH - 8, vx = x + 8, vw = bw - 16;
          if (k === 0) {
            // o extent da conversa no pool: enche ao restaurar, é lido ao guardar
            V.rr(g, vx, vy + 6, vw, vizH - 12, 5); g.fillStyle = V.alpha(C.muted, 0.12); g.fill();
            const f = load ? ph : 1;
            V.rr(g, vx, vy + 6, Math.max(4, vw * f), vizH - 12, 5); g.fillStyle = V.alpha(q.col, 0.55); g.fill();
            if (!load) { const sx = vx + vw * ph; g.fillStyle = V.alpha(C.ink, 0.25); g.fillRect(sx - 2, vy + 6, 4, vizH - 12); }
          } else if (k === 1) {
            // 16 slots pinned (anel de escrita ao guardar)
            const per = 8, sg = 3, sw = Math.min(14, (vw - (per - 1) * sg) / per), sh = 12;
            const lit = Math.floor(ph * 16);
            for (let s = 0; s < 16; s++) {
              const sx = vx + (s % per) * (sw + sg), sy = vy + 4 + Math.floor(s / per) * (sh + 4);
              const d = (s - lit + 16) % 16, on = d < 5;
              V.rr(g, sx, sy, sw, sh, 3);
              g.fillStyle = V.alpha(q.col, on ? 0.85 - d * 0.14 : 0.15); g.fill();
            }
          } else {
            // blocos de 2.048 tokens no disco (1 célula = 1 ou mais blocos)
            const cells = Math.min(nb, 48), per = narrow ? 12 : 24, rows = Math.ceil(cells / per);
            const cg = 2, cw = (vw - (per - 1) * cg) / per, chh = Math.min(10, (vizH - (rows - 1) * cg) / rows);
            const done = load ? cells : Math.floor(ph * (cells + 1));
            const rd2 = Math.floor(ph * cells);
            for (let c = 0; c < cells; c++) {
              const cx = vx + (c % per) * (cw + cg), cy = vy + Math.floor(c / per) * (chh + cg);
              const filled = c < done, reading = load && c === rd2;
              g.fillStyle = V.alpha(q.col, reading ? 1 : filled ? 0.6 : 0.12);
              g.fillRect(cx, cy, cw, chh);
            }
            if (nb > 48) V.text(g, L.cell(Math.ceil(nb / 48)), vx + vw, vy + vizH + 2, { size: 11, color: C.muted, align: 'right', maxW: vw });
          }
        });
        // setas entre os andares e blocos em trânsito
        const ay = y + 14 + 6;
        for (let k = 0; k < 2; k++) {
          const x1 = xs[k] + bw + 3, x2 = xs[k + 1] - 3;
          if (load) V.arrow(g, x2, ay, x1, ay, { color: V.alpha(C.ink, 0.6), lw: 1.5, head: 6 });
          else V.arrow(g, x1, ay, x2, ay, { color: V.alpha(C.ink, 0.6), lw: 1.5, head: 6 });
        }
        const xa = xs[0] + bw / 2, xb = xs[2] + bw / 2, my = y + bh / 2 + 4;
        for (let j = 0; j < 4; j++) {
          const u = (ph + j / 4) % 1, xx = load ? xb + (xa - xb) * u : xa + (xb - xa) * u;
          g.save(); g.globalAlpha = Math.sin(Math.PI * u) * 0.9;
          V.rr(g, xx - 5, my - 5, 10, 10, 2); g.fillStyle = load ? C.a : C.e; g.fill();
          g.restore();
        }
        y += bh + 8;
        // legendas dos caminhos
        const l1 = load ? L.copyA : L.saveA1, l2 = load ? L.loadA : L.saveA2;
        if (narrow) {
          y += wrap(g, '① ' + l1, pad, y + 12, w - 2 * pad, { size: 11, color: C.muted });
          y += wrap(g, '② ' + l2, pad, y + 12, w - 2 * pad, { size: 11, color: C.muted });
        } else {
          const hw = (w - 2 * pad) / 2 - 8;
          const u1 = wrap(g, '① ' + l1, pad, y + 12, hw, { size: 11, color: C.muted });
          const u2 = wrap(g, '② ' + l2, pad + hw + 16, y + 12, hw, { size: 11, color: C.muted });
          y += Math.max(u1, u2);
        }
        y += wrap(g, load ? L.decoding : L.survive, pad, y + 13, w - 2 * pad, { size: 11.5, weight: 600, color: load ? C.a : C.e }) + 8;
        /* ---------- o gráfico: refazer × restaurar ---------- */
        const ml = narrow ? 44 : 52, mr = 12, top = y + 20, chH = narrow ? 210 : 232;
        const px0 = pad + ml, px1 = w - pad - mr, py0 = top, py1 = top + chH;
        const XL = Math.log(7000), XH = Math.log(1.2e6), YL = Math.log(0.03), YH = Math.log(2500);
        const X = (v) => px0 + ((Math.log(v) - XL) / (XH - XL)) * (px1 - px0);
        const Y = (v) => py1 - ((Math.log(v) - YL) / (YH - YL)) * (py1 - py0);
        wrap(g, L.yAx, pad, y + 12, w - 2 * pad, { size: 11, color: C.muted });
        [0.1, 1, 10, 100, 1000].forEach((v) => {
          V.line(g, px0, Y(v), px1, Y(v), { color: V.alpha(C.muted, 0.18), lw: 1 });
          V.text(g, v < 1 ? sf(v, 1) + ' s' : nf(v) + ' s', px0 - 6, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        });
        [[1e4, '10k'], [3e4, '30k'], [1e5, '100k'], [3e5, '300k'], [1e6, '1M']].forEach(([v, s]) => {
          V.line(g, X(v), py0, X(v), py1, { color: V.alpha(C.muted, 0.18), lw: 1 });
          V.text(g, s, X(v), py1 + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        });
        V.line(g, px0, py1, px1, py1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.line(g, px0, py0, px0, py1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.text(g, L.xAx, px1, py1 + 30, { size: 11, color: C.muted, align: 'right' });
        // refazer: TTFT de prefill medido; entre os pontos, interpolação log-log
        g.beginPath(); PRE.forEach(([a, b], k) => (k ? g.lineTo(X(a), Y(b)) : g.moveTo(X(a), Y(b))));
        g.strokeStyle = C.d; g.lineWidth = 2; g.stroke();
        PRE.forEach(([a, b]) => V.dot(g, X(a), Y(b), 4, C.d));
        // restaurar: medido a 40k e 260k; reta entre eles (interpolado) e pontilhada fora (extrapolado)
        const rl = (a) => R40[1] + KS * (a - R40[0]);
        g.save(); g.setLineDash([2, 4]); g.strokeStyle = V.alpha(C.a, 0.85); g.lineWidth = 1.6;
        g.beginPath(); g.moveTo(X(NMIN), Y(rl(NMIN)));
        for (let k = 1; k <= 12; k++) { const a = Math.exp(Math.log(NMIN) + (k / 12) * Math.log(R40[0] / NMIN)); g.lineTo(X(a), Y(rl(a))); }
        g.stroke();
        g.beginPath(); g.moveTo(X(R260[0]), Y(R260[1]));
        for (let k = 1; k <= 16; k++) { const a = Math.exp(Math.log(R260[0]) + (k / 16) * Math.log(NMAX / R260[0])); g.lineTo(X(a), Y(rl(a))); }
        g.stroke(); g.restore();
        g.beginPath();
        for (let k = 0; k <= 16; k++) { const a = Math.exp(Math.log(R40[0]) + (k / 16) * Math.log(R260[0] / R40[0])); (k ? g.lineTo : g.moveTo).call(g, X(a), Y(rl(a))); }
        g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
        V.dot(g, X(R40[0]), Y(R40[1]), 4, C.a);
        V.dot(g, X(R260[0]), Y(R260[1]), 4, C.a);
        const bar = (x, lo, hi, a) => { V.line(g, x, Y(lo) + 3, x, Y(hi) - 3, { color: V.alpha(C.a, a), lw: 4 }); V.line(g, x - 4, Y(lo) + 3, x + 4, Y(lo) + 3, { color: V.alpha(C.a, a), lw: 1.5 }); V.line(g, x - 4, Y(hi) - 3, x + 4, Y(hi) - 3, { color: V.alpha(C.a, a), lw: 1.5 }); };
        bar(X(R260[0]) - 4, IDLE[0], IDLE[1], 0.9);
        bar(X(R260[0]) + 4, BUSY[0], BUSY[1], load ? 0.9 : 0.4);
        if (load && !narrow) V.text(g, L.busy, X(R260[0]) - 10, Y(BUSY[1]) - 8, { size: 11, color: C.a, align: 'right' });
        // outro caso medido: GLM-5.3 completo em três Sparks
        const dia = (x, yy) => { g.beginPath(); g.moveTo(x, yy - 6); g.lineTo(x + 6, yy); g.lineTo(x, yy + 6); g.lineTo(x - 6, yy); g.closePath(); g.fillStyle = C.c; g.fill(); };
        g.save(); g.setLineDash([3, 3]); V.line(g, X(BIG.n), Y(BIG.pre), X(BIG.n), Y(BIG.rst), { color: V.alpha(C.c, 0.6), lw: 1.2 }); g.restore();
        dia(X(BIG.n), Y(BIG.pre)); dia(X(BIG.n), Y(BIG.rst));
        { const bmw = px1 - X(BIG.n) - 14, bw0 = Math.min(bmw, tw(g, L.bigLab, 11));
          V.box(g, X(BIG.n) + 6, Y(BIG.pre) - 19, bw0 + 6, 17, { r: 4, fill: V.alpha(C.panel || C.bg, 0.85) });
          V.text(g, L.bigLab, X(BIG.n) + 9, Y(BIG.pre) - 6, { size: 11, color: C.c, maxW: bmw }); }
        // cursor
        const cx = X(n);
        V.line(g, cx, py0, cx, py1, { color: V.alpha(C.ink, 0.35), lw: 1.2 });
        const yp = Y(p.v), yr = Y(r.v);
        V.dot(g, cx, yp, 6, V.alpha(C.d, 0.35)); V.dot(g, cx, yp, 3.5, C.d);
        V.dot(g, cx, yr, 6, V.alpha(C.a, 0.35)); V.dot(g, cx, yr, 3.5, C.a);
        const right = cx < (px0 + px1) / 2, ax = right ? cx + 10 : cx - 10, al = right ? 'left' : 'right';
        const tag = (s, yy, col) => {
          const mw = (right ? px1 - ax : ax - px0) - 4, tw0 = Math.min(mw, tw(g, s, 11.5, 'sans', 600));
          const bx = right ? ax - 4 : ax - tw0 - 4;
          V.box(g, bx, yy - 13, tw0 + 8, 18, { r: 5, fill: V.alpha(C.panel || C.bg, 0.88), stroke: V.alpha(col, 0.7), lw: 1 });
          V.text(g, s, ax, yy, { size: 11.5, weight: 600, color: C.ink, align: al, maxW: mw });
        };
        tag(L.reLab(secs(p.v), L[p.k]), yp - 12, C.d);
        tag(L.ldLab(secs(r.v), L[r.k]), yr + 20, C.a);
        V.text(g, '×' + nf(p.v / r.v), ax, (yp + yr) / 2 + 4, { size: 13, weight: 700, color: C.d, align: al });
        fitH(H || { canvas: fig.querySelector('canvas') }, py1 + 40);
      });
      upd();
    }
  },
});

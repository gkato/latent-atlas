LA.chapter({
  id: 'kernels',
  part: 5,
  order: 28,
  title: 'Kernels: Triton, CUTLASS, Marlin, FlashInfer e b12x',
  navTitle: 'Kernels',
  subtitle: 'Onde a matemática encontra o silício: lançamento e ondas, tiling e mainloop, as linguagens de kernel (CUDA C++, CUTLASS, Triton, CuTeDSL, TileLang), fusão, softmax online e split-K, GEMM quantizada por M, MoE thin e fat, os riscos de autotune, JIT e PDL e, no TensorFold, kernels que não deixam o lote mudar um bit de uma linha.',
  minutes: 70,
  recipes: ['glm', 'qwen', 'ds', 'tf'],
  tags: ['kernels', 'GEMM', 'tiling', 'fusão', 'FlashAttention', 'split-K', 'Marlin', 'grouped GEMM', 'MoE', 'b12x', 'Triton', 'CUTLASS', 'CUDA C++', 'CuTeDSL', 'TileLang', 'autotuning', 'PDL', 'TensorFold', 'invariância de lote'],
  goals: [
    'Explicar como um kernel é lançado (grid, CTAs, ondas) e decidir, a partir de <b>M</b>, se ele é limitado por memória ou por compute',
    'Descrever o mainloop de uma GEMM (tiling, <code>cp.async</code>, swizzle, <code>mma.sync</code>, epílogo) e o que fusão, softmax online e split-K economizam',
    'Distinguir CUDA C++, CUTLASS/CuTe, Triton, CuTeDSL e TileLang: quem escreve cada passo do kernel, quando ele é compilado e o que isso muda para trocar um kernel numa receita',
    'Ler as flags de kernel das receitas (EXL3 E2/E3/FAST/cooperativo, Marlin ↔ BF16 por M, <code>QSA_PROFILE</code>, b12x e os patches de kernel do TensorFold) sabendo o que cada uma troca e por quê',
    'Reconhecer os riscos de autotuning, JIT, especialização e PDL em inferência distribuída, e o que muda quando o motor exige que os bits de cada linha não dependam do lote',
  ],
  prereqs: ['gpu-por-dentro', 'quantizacao-fundamentos'],
  summary: [
    'Kernel = função da GPU que a CPU enfileira num stream como uma grade de CTAs; ondas = ⌈CTAs / (SMs × ocupação)⌉, e tabelas afinadas para os 160 SMs de um GB300 transbordam nos 48 do GB10.',
    'O regime é ditado por <b>M</b>: decode (M pequeno) é limitado por memória e quer ler pesos rápido (Marlin, W4A16, FAST, cooperativo); prefill (M grande) quer tensor cores ou cópias BF16 (E3, KDA BF16 acima de 512, W4A4 a partir de 128). Fusão poupa idas à DRAM, mas pode mudar a ordem das somas.',
    'Tiling, <code>cp.async</code> multi-estágio, swizzle, <code>ldmatrix</code>, <code>mma.sync</code> e epílogo são o esqueleto de toda GEMM. Em CUDA C++ você escreve cada thread (AOT, extensão <code>.so</code>); CUTLASS/CuTe monta o kernel por templates; em Triton você escreve um programa por tile e o compilador cuida de warps, estágios e tensor cores (JIT); CuTeDSL e TileLang levam esse estilo ao Python.',
    'FlashAttention é softmax online (m, ℓ, o), em blocos de Q × blocos de K/V no prefill; no decode, split-K divide as chaves e funde os parciais pelo log-sum-exp, num equilíbrio com o número de SMs.',
    'MoE: experts thin no decode (1 lançamento graph-safe do <code>exl3_moe</code>, cap 32), fat no prefill (E3: grouped GEMM com tabelas no device, +37–45%); o b12x roda W4A16 no DeepSeek porque o checkpoint é MXFP4.',
    'Autotune, JIT, especialização e PDL travam ou corrompem se descuidados: táticas coerentes entre ranks, caches persistidos, headers aliasados por compute capability, PDL desligado no SM12x.',
    'Na lane TensorFold do GLM, os bits de uma linha não podem depender do lote: split-K pela forma do peso, tabelas de tile que preservam a cadeia de somas, atomics só em inteiros. A velocidade vem de mover os mesmos bits (ordem de lançamento NFIRST, 2,3–2,5× a 4.096 linhas no kernel agrupado; intercalar itens por gargalo; prefetch de L2; cargas <code>.nc</code>), e os bits só mudam por escolha declarada (kernels de prompt Yᵀ = WᵀXᵀ, KDA em chunks, densos q4).',
  ],
  quiz: [
    { q: 'Com 48 SMs e ocupação 1, um kernel com 56 CTAs roda em quantas ondas, e com qual uso médio?', o: ['1 onda, 100%', '2 ondas, ~58%', '2 ondas, 100%', '56 ondas, 2%'], a: 1, e: '⌈56/48⌉ = 2 ondas; 56 / (2 × 48) ≈ 58%. A segunda onda roda só 8 CTAs e deixa 40 SMs parados (efeito de cauda).' },
    { q: 'Você quer testar outro <code>BLOCK_N</code> num kernel da QSA do Qwen e outro tile num kernel do <code>exllamav3_ext</code> do GLM. O que cada troca exige?', o: ['As duas exigem reconstruir a imagem', 'A QSA exige recompilar; o <code>exllamav3_ext</code> é editável em Python', 'A QSA é Triton: basta editar o <code>.py</code>, e o JIT recompila na primeira chamada; o <code>exllamav3_ext</code> é CUDA C++ compilado AOT para sm_121a e exige recompilar a extensão no build', 'Nenhuma das duas é possível sem mudar o vLLM'], a: 2, e: 'Em Triton, <code>BLOCK_N</code> é um <code>tl.constexpr</code>: cada valor vira uma variante compilada em JIT (é assim que o overlay <code>qsa_gb10</code> troca a tabela). Em CUDA C++ o <code>nvcc</code> compila no build, e o patch entra no Dockerfile.' },
    { q: 'Por que o GLM retém uma cópia BF16 do <code>in_proj</code> do KDA e só a usa com M &gt; 512?', o: ['Porque o Marlin W8A16 lê metade dos bytes e vai bem com M pequeno, mas é 3,1–3,7× mais lento que o cuBLAS BF16 a partir de M = 768; 512 fica na faixa vazia entre o decode (≤ 220) e o prefill (≥ 768)', 'Porque o Marlin é numericamente impreciso', 'Porque a cópia BF16 ocupa menos memória', 'Porque o cuBLAS não roda no SM121'], a: 0, e: 'É um dispatch por M (27,21 vs 7,36 ms em M = 7.168). O cruzamento exato não foi medido: a fronteira saiu da distribuição de M observada. O custo é +3,29 GiB por rank.' },
    { q: 'Por que o DeepSeek roda o MoE como W4A16 no GB10, que tem tensor cores FP4?', o: ['Porque W4A16 é sempre mais rápido', 'Porque o vLLM não suporta FP4', 'Porque os experts são FP8', 'Porque o caminho NVFP4 de tensor core do b12x só aceita fonte <code>modelopt_nvfp4</code>, e o checkpoint é MXFP4 (<code>fp4_e8m0_k32</code>)'], a: 3, e: 'O formato força desquantizar para BF16 e usar MMA de 16 bits: bom no decode, ineficiente no prefill (+30–60% estimados com um MoE MXFP4×MXFP8 nativo).' },
    { q: 'Qual é a vantagem central do tier E3 do GLM sobre o loop Python legado para experts fat?', o: ['Usa menos memória', 'Usa FP4 nativo', 'Cobre todos os experts fat da camada em 3 lançamentos dirigidos por tabelas no device, sem sincronizar com a CPU: mais rápido e capturável em CUDA graph', 'Dispensa o roteamento'], a: 2, e: 'Prefix-sums e segmentos de 64 linhas são calculados na GPU, e os kernels fazem grid-stride sobre eles: +37–45% no prefill frio, ao custo de 560 MiB de scratch.' },
    { q: 'O que significa, na prática, “PDL lowering races KDA state kernels on SM12x”?', o: ['Que, com PDL, kernels dependentes podem se sobrepor aos que escrevem o estado do KDA e ler dados incompletos; por isso a receita restringe PDL a capability major 9/10', 'Que o PDL deixa o passo mais lento', 'Que o NCCL trava', 'Que o autotune falha'], a: 0, e: 'O repositório só registra a race; a alternativa é a interpretação do modo típico de falha do PDL (consumir dados sem um <code>griddepcontrol.wait</code> antes), uma corrupção silenciosa. O patch troca <code>major &gt;= 9</code> por <code>major in (9, 10)</code>.' },
    { q: 'Por que as matmuls de 4 bits do TensorFold escolhem as partições do split-K por <code>split_k(n, k)</code>, e não pelo número de linhas da janela, como a tabela por bucket da QSA?', o: ['Porque menos partições são sempre mais rápidas', 'Porque a soma em fp32 não é associativa: com partições fixadas pela forma do peso, os bits de uma linha não dependem de quantas linhas vêm junto, e verificar um rascunho vira um teste de igualdade com a decodificação serial', 'Porque o GB10 não suporta split-K', 'Porque o split-K por linhas exigiria atomics'], a: 1, e: 'Rascunho = serial e concorrente = solo só valem bit a bit se cada linha tiver a mesma cadeia de somas em qualquer lote. O tile e o número de programas podem mudar (velocidade); a cadeia, não. O custo é abrir mão de heurísticas que enchem a GPU conforme M.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o || {});
    const fnt = (o) => V.font(o.size || 12, o.font || 'sans', o.weight || 400);
    const comma = (s) => String(s).replace('.', ',');
    const fmtB = (b) => (b >= 1024 ** 3 ? comma((b / 1024 ** 3).toFixed(2)) + ' GiB' : b >= 1024 ** 2 ? comma((b / 1024 ** 2).toFixed(1)) + ' MiB' : comma((b / 1024).toFixed(1)) + ' KiB');
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
    function smGrid(g, x0, y0, W, Hmax, nSM, cellFn) {
      const cols = nSM <= 48 ? 12 : W < 480 ? 16 : 20;
      const rows = Math.ceil(nSM / cols), gap = 3;
      const cs = Math.max(6, Math.min(nSM <= 48 ? 36 : 24, (W - gap * (cols - 1)) / cols, (Hmax - gap * (rows - 1)) / rows));
      const gw = cols * cs + (cols - 1) * gap, xs = x0 + (W - gw) / 2;
      for (let k = 0; k < nSM; k++) cellFn(k, xs + (k % cols) * (cs + gap), y0 + Math.floor(k / cols) * (cs + gap), cs);
      return { h: rows * cs + (rows - 1) * gap, x: xs, w: gw };
    }
    const Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

    /* =============== Figura 1: lançamento, grid e ondas =============== */
    {
      const fig = $('#fig-lancamento');
      const S = { ctas: 240, sms: 48, occ: 1 };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'CTAs no grid', min: 8, max: 480, step: 8, value: 240, onInput: (v) => { S.ctas = v; upd(); } });
      UI.seg(ctl, { label: 'GPU', options: [{ v: 48, label: 'GB10 · 48 SMs' }, { v: 160, label: 'GB300 · 160 SMs' }], value: 48, onChange: (v) => { S.sms = v; upd(); } });
      UI.slider(ctl, { label: 'CTAs por SM', min: 1, max: 4, step: 1, value: 1, onInput: (v) => { S.occ = v; upd(); } });
      const rW = UI.readout(ctl, 'ondas', '—'), rU = UI.readout(ctl, 'uso médio', '—');
      function upd() { const sl = S.sms * S.occ, n = Math.ceil(S.ctas / sl); rW.set(String(n)); rU.set(Math.round((100 * S.ctas) / (n * sl)) + '%'); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const slots = S.sms * S.occ, nW = Math.ceil(S.ctas / slots);
        const bw = Math.min(120, w * 0.26);
        V.box(g, 12, 10, bw, 34, { r: 9, fill: V.alpha(C.d, 0.15), stroke: C.d, label: 'CPU · vLLM', size: 12 });
        V.box(g, w - 12 - bw, 10, bw, 34, { r: 9, fill: V.alpha(C.a, 0.15), stroke: C.a, label: `GPU · ${S.sms} SMs`, size: 12 });
        const qx0 = 12 + bw + 8, qx1 = w - 12 - bw - 8;
        V.line(g, qx0, 27, qx1, 27, { color: V.alpha(C.muted, 0.45), lw: 1.5, dash: [3, 5], dashOffset: -t * 20 });
        for (let k = 0; k < 4; k++) {
          const u = (t * 0.22 + k / 4) % 1, x = V.lerp(qx0, qx1 - 30, u);
          V.box(g, x, 20, 30, 15, { r: 4, fill: V.alpha(C.series[k], 0.8), label: 'k' + (k + 1), size: 11, color: onTxt(C) });
        }
        T(g, 'stream: fila de lançamentos, consumida em ordem', (qx0 + qx1) / 2, 60, { size: 11, align: 'center', color: C.muted, font: 'mono', maxW: w - 24 });
        const top = 98, bottom = 96;
        const period = 1.3, cyc = nW * period + 1.2, tc = t % cyc;
        const wi = Math.min(nW - 1, Math.floor(tc / period)), u = V.clamp((tc - wi * period) / (period * 0.8), 0, 1);
        T(g, `onda ${wi + 1} de ${nW}`, 16, top - 10, { size: 12.5, weight: 600, color: C.ink });
        T(g, `${S.ctas} CTAs · ${S.occ}/SM`, w - 16, top - 10, { size: 11.5, align: 'right', color: C.muted, font: 'mono' });
        smGrid(g, 16, top, w - 32, h - top - bottom, S.sms, (k, x, y, cs) => {
          V.box(g, x, y, cs, cs, { r: 3, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.28), lw: 1 });
          const sw = (cs - 4) / S.occ;
          for (let j = 0; j < S.occ; j++) {
            const idx = wi * slots + k * S.occ + j;
            if (idx < S.ctas) {
              const hh = Math.max(1, (cs - 4) * (0.15 + 0.85 * u));
              g.fillStyle = V.alpha(C.series[wi % 6], 0.85); V.rr(g, x + 2 + j * sw, y + 2 + (cs - 4 - hh), Math.max(1, sw - 1), hh, 2); g.fill();
            } else {
              g.strokeStyle = V.alpha(C.c, 0.55); g.lineWidth = 1; g.beginPath(); g.moveTo(x + 3 + j * sw, y + cs - 3); g.lineTo(x + 1 + (j + 1) * sw, y + 3); g.stroke();
            }
          }
        });
        const by = h - 74, bx = 16, bW = w - 32, segW = bW / nW;
        T(g, 'tempo do kernel → (1 bloco = 1 onda; altura cheia = todos os slots ocupados)', bx, by - 8, { size: 11, color: C.muted, maxW: bW });
        for (let i = 0; i < nW; i++) {
          const used = Math.min(slots, S.ctas - i * slots) / slots;
          V.box(g, bx + i * segW + 1, by, Math.max(2, segW - 2), 26, { r: 4, fill: V.alpha(C.muted, 0.1) });
          const hh = 26 * used;
          V.box(g, bx + i * segW + 1, by + 26 - hh, Math.max(2, segW - 2), hh, { r: 4, fill: V.alpha(C.series[i % 6], i === wi ? 0.9 : 0.45) });
        }
        const last = (S.ctas - (nW - 1) * slots) / slots;
        wrap(g, `ondas = ⌈${S.ctas} / (${S.sms} × ${S.occ})⌉ = ${nW} · última onda usa ${Math.round(last * 100)}% dos slots · uso médio ${Math.round((100 * S.ctas) / (nW * slots))}%`, bx, by + 46, bW, 15, { size: 12, color: C.ink });
      });
    }

    /* =============== Figura 2: GEMM por dentro =============== */
    {
      const fig = $('#fig-gemm');
      const S = { mode: 'tiles', mi: 3, wb: 16, st: 4, swz: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Visão', options: [{ v: 'tiles', label: 'tiles & intensidade' }, { v: 'pipe', label: 'pipeline do mainloop' }], value: 'tiles', onChange: (v) => { S.mode = v; vis(); } });
      const cM = UI.slider(ctl, { label: 'M (linhas)', min: 0, max: 13, step: 1, value: 3, fmt: (i) => String(Ms[i]), onInput: (v) => { S.mi = v; upd(); } });
      const cW = UI.seg(ctl, { label: 'Pesos', options: [{ v: 16, label: 'BF16' }, { v: 8, label: 'FP8' }, { v: 4, label: '4 bits' }], value: 16, onChange: (v) => { S.wb = v; upd(); } });
      const cS = UI.slider(ctl, { label: 'Estágios cp.async', min: 1, max: 8, step: 1, value: 4, onInput: (v) => { S.st = v; upd(); } });
      const cZ = UI.toggle(ctl, { label: 'swizzle XOR na SMEM', value: false, onChange: (v) => { S.swz = v; } });
      const refit = fitH(fig, (w) => (w < 600 ? (S.mode === 'tiles' ? 520 : 480) : 430));
      const rI = UI.readout(ctl, 'intensidade', '—'), rR = UI.readout(ctl, 'regime', '—');
      const rU = UI.readout(ctl, 'uso do tensor core', '—'), rT = UI.readout(ctl, 'tempo p/ 10 fatias', '—');
      const K = 4096, N = 4096, BW = 230e9, PK = 125e12;
      const inten = () => { const M = Ms[S.mi]; const fl = 2 * M * N * K, wB = (K * N * S.wb) / 8, aB = 2 * (M * K + M * N); return { M, fl, wB, aB, I: fl / (wB + aB), tm: ((wB + aB) / BW) * 1e3, tc: (fl / PK) * 1e3 }; };
      function sim(Sg) {
        const n = 10, Lt = 3, Ct = 1, iss = 0.25, ld = [], cp = [];
        for (let i = 0; i < n; i++) {
          const s = i < Sg ? i * iss : Math.max(cp[i - Sg].e, ld[i - 1].s + iss);
          ld.push({ s, e: s + Lt });
          const cs = Math.max(ld[i].e, i ? cp[i - 1].e : 0);
          cp.push({ s: cs, e: cs + Ct });
        }
        const total = cp[n - 1].e;
        return { ld, cp, total, util: (n * Ct) / total };
      }
      function vis() {
        const tl = S.mode === 'tiles';
        [cM.el, cW.el, rI.el, rR.el].forEach((e) => { e.style.display = tl ? '' : 'none'; });
        [cS.el, cZ.el, rU.el, rT.el].forEach((e) => { e.style.display = tl ? 'none' : ''; });
        if (typeof refit === 'function') refit();
        upd();
      }
      function upd() {
        const q = inten();
        rI.set((q.I < 10 ? comma(q.I.toFixed(1)) : Math.round(q.I)) + ' FLOP/B');
        rR.set(q.I < 458 ? 'limitado por memória' : 'limitado por compute');
        const p = sim(S.st);
        rU.set(Math.round(p.util * 100) + '%'); rT.set(comma(p.total.toFixed(1)) + ' u');
      }
      vis();
      function drawTiles(g, w, h, t) {
        const C = V.C(), q = inten(), M = q.M;
        const wide = w >= 720;
        const rx = 16, ry = 6, rw = wide ? w * 0.55 : w - 32, rh = wide ? h - 14 : h * 0.5;
        const Kv = Math.min(rw * 0.3, rh * 0.34), gap = 12, Nv = rw - Kv - gap;
        const maxMv = rh - Kv - gap - 58;
        const tileRows = M >= 128 ? Math.ceil(M / 128) : 1, drawn = Math.min(6, tileRows);
        const tileH = Math.max(18, Math.min(36, maxMv / 6));
        const Mv = M < 128 ? tileH : Math.min(maxMv, tileH * drawn);
        const frac = M < 128 ? M / 128 : 1;
        const bx = rx + Kv + gap, byB = ry + 22, ay = byB + Kv + gap + 18, ax = rx;
        const nCols = 6, tw = Nv / nCols, th = Mv / drawn;
        V.box(g, bx, byB, Nv, Kv, { r: 4, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.55), lw: 1 });
        for (let c = 1; c < nCols; c++) V.line(g, bx + c * tw, byB, bx + c * tw, byB + Kv, { color: V.alpha(C.b, 0.25), lw: 1 });
        T(g, 'B · pesos [K × N]', bx, byB - 8, { size: 11.5, color: C.b, font: 'mono', weight: 600 });
        V.box(g, ax, ay, Kv, Mv * frac < 3 ? 3 : Mv * frac, { r: 3, fill: V.alpha(C.d, 0.14), stroke: V.alpha(C.d, 0.6), lw: 1 });
        T(g, 'A · ativações', ax, ay - 8, { size: 11.5, color: C.d, font: 'mono', weight: 600 });
        T(g, `[${M} × K]`, ax, ay + Mv + 16, { size: 11, color: C.muted, font: 'mono' });
        V.box(g, bx, ay, Nv, Mv, { r: 3, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.45), lw: 1, });
        for (let c = 1; c < nCols; c++) V.line(g, bx + c * tw, ay, bx + c * tw, ay + Mv, { color: V.alpha(C.muted, 0.3), lw: 1 });
        for (let r = 1; r < drawn; r++) V.line(g, bx, ay + r * th, bx + Nv, ay + r * th, { color: V.alpha(C.muted, 0.3), lw: 1 });
        const per = 2.2, idx = Math.floor(t / per) % (drawn * nCols), u = (t % per) / per;
        const ci = Math.floor(idx / nCols), cj = idx % nCols, kk = Math.min(4, Math.floor(u * 5));
        g.strokeStyle = C.c; g.lineWidth = 2; V.rr(g, ax - 1, ay + ci * th - 1, Kv + 2, th * frac + 2, 3); g.stroke();
        V.box(g, ax + (kk * Kv) / 5, ay + ci * th, Kv / 5, Math.max(3, th * frac), { r: 2, fill: V.alpha(C.d, 0.75) });
        g.strokeStyle = C.c; V.rr(g, bx + cj * tw - 1, byB - 1, tw + 2, Kv + 2, 3); g.stroke();
        V.box(g, bx + cj * tw, byB + (kk * Kv) / 5, tw, Kv / 5, { r: 2, fill: V.alpha(C.b, 0.75) });
        V.box(g, bx + cj * tw, ay + ci * th, tw, Math.max(3, th * frac), { r: 2, fill: V.alpha(C.c, 0.18 + 0.55 * u), stroke: C.c, lw: 1.5 });
        if (frac < 1 && th > 16) {
          g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.c, 0.7); g.lineWidth = 1; V.rr(g, bx + cj * tw, ay + ci * th, tw, th, 2); g.stroke(); g.setLineDash([]);
        }
        V.arrow(g, ax + (kk * Kv) / 5 + Kv / 10, ay + ci * th + Math.max(3, th * frac) / 2, bx + cj * tw - 3, ay + ci * th + Math.max(3, th * frac) / 2, { color: V.alpha(C.c, 0.6), lw: 1.3, head: 6, curve: 10 });
        V.arrow(g, bx + cj * tw + tw / 2, byB + (kk * Kv) / 5 + Kv / 10, bx + cj * tw + tw / 2, ay + ci * th - 3, { color: V.alpha(C.c, 0.6), lw: 1.3, head: 6 });
        T(g, `C = A·B · tiles 128 × 128${tileRows > drawn ? ` (${tileRows} linhas de tiles)` : ''}`, bx, ay + Mv + 16, { size: 11, color: C.muted, font: 'mono', maxW: Nv });
        if (M < 128) T(g, `tile de 128 linhas, só ${M} úteis`, bx, ay + Mv + 31, { size: 11, color: C.c, maxW: Nv });
        if (wide) T(g, `fatia B_K ${kk + 1}/5 → SMEM`, bx + Nv, byB - 8, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        else T(g, `fatia B_K ${kk + 1}/5 → SMEM`, bx + Nv, byB + Kv + 12, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        const gx = wide ? w * 0.61 : 20, gw = wide ? w - gx - 20 : w - 40, gy = wide ? 70 : h * 0.53 + 20;
        const lo = Math.log2(0.5), hi = Math.log2(8192);
        const X = (I) => gx + ((Math.log2(V.clamp(I, 0.5, 8192)) - lo) / (hi - lo)) * gw;
        T(g, 'Régua do roofline (GB10)', gx, gy - 16, { size: 12.5, weight: 600, color: C.ink });
        V.box(g, gx, gy, X(458) - gx, 24, { r: 5, fill: V.alpha(C.b, 0.22) });
        V.box(g, X(458), gy, gx + gw - X(458), 24, { r: 5, fill: V.alpha(C.a, 0.22) });
        T(g, 'memória', gx + 6, gy + 16, { size: 11, color: C.ink });
        T(g, 'compute', gx + gw - 6, gy + 16, { size: 11, color: C.ink, align: 'right' });
        [1, 10, 100, 1000].forEach((v) => { V.line(g, X(v), gy + 24, X(v), gy + 30, { color: C.muted, lw: 1 }); T(g, String(v), X(v), gy + 42, { size: 11, align: 'center', color: C.muted, font: 'mono' }); });
        V.line(g, X(458), gy - 6, X(458), gy + 30, { color: C.d, lw: 2 });
        T(g, 'ridge ≈ 458', X(458), gy - 10, { size: 11, color: C.d, align: 'center', font: 'mono' });
        const mx = X(q.I);
        V.glow(g, mx, gy + 12, 16, C.c); V.dot(g, mx, gy + 12, 6, C.c);
        const ly = gy + 66;
        const tl = [
          `I = FLOPs / bytes ≈ ${q.I < 10 ? comma(q.I.toFixed(1)) : Math.round(q.I)} FLOP/B  (camada 4096 × 4096)`,
          `FLOPs = 2·M·N·K = ${comma((q.fl / 1e9).toFixed(q.fl < 1e10 ? 3 : 1))} GFLOP`,
          `bytes ≈ pesos ${fmtB(q.wB)} + ativações ${fmtB(q.aB)}`,
          `t_mem ≈ ${comma(q.tm.toFixed(3))} ms · t_compute ≈ ${comma(q.tc.toFixed(3))} ms (230 GB/s; 125 TFLOPS)`,
        ];
        let yy = ly;
        tl.forEach((s) => { yy += 15 * wrap(g, s, gx, yy, gw, 15, { size: 11.5, color: C.ink, font: 'mono' }) + 4; });
        wrap(g, q.I < 458 ? '→ o kernel espera bytes: ler os pesos na banda máxima é tudo (GEMV/skinny GEMM).' : '→ o kernel espera o tensor core: tiles grandes e formatos FP8/FP4 nativos ajudam.', gx, yy + 6, gw, 15, { size: 12, weight: 600, color: q.I < 458 ? C.b : C.a });
      }
      function drawPipe(g, w, h, t) {
        const C = V.C(), p = sim(S.st), maxT = 42;
        const lw = w < 560 ? 80 : 150, x0 = 14 + lw, x1 = w - 14;
        const X = (tt) => x0 + (tt / maxT) * (x1 - x0);
        const lanes = [['cp.async', 'global → SMEM'], ['ldmatrix', 'SMEM → registrador'], ['mma.sync', 'tensor core']];
        const ly0 = 30, lh = 42;
        T(g, `${S.st} estágio(s) · latência DRAM = 3 u · conta por fatia = 1 u (modelo didático)`, 14, 16, { size: 11, color: C.muted, maxW: w - 28 });
        lanes.forEach(([a, b], i) => {
          const y = ly0 + i * lh;
          V.box(g, 14, y, lw - 8, lh - 8, { r: 6, fill: V.alpha(C.muted, 0.08) });
          T(g, a, 22, y + 15, { size: 12, weight: 600, font: 'mono', color: C.ink });
          if (lw > 100) T(g, b, 22, y + 29, { size: 11, color: C.muted });
          V.line(g, x0, y + lh - 8, x1, y + lh - 8, { color: V.alpha(C.muted, 0.2), lw: 1 });
        });
        const sub = (lh - 10) / S.st;
        p.ld.forEach((L, i) => { const y = ly0 + (i % S.st) * sub; V.box(g, X(L.s), y + 1, Math.max(2, X(L.e) - X(L.s) - 1), Math.max(2, sub - 2), { r: 2, fill: V.alpha(C.series[i % 6], 0.75) }); });
        p.cp.forEach((Cc, i) => {
          const col = C.series[i % 6];
          V.box(g, X(Cc.s), ly0 + lh + 2, Math.max(1, X(Cc.s + 0.3) - X(Cc.s)), lh - 12, { r: 2, fill: V.alpha(col, 0.6) });
          V.box(g, X(Cc.s + 0.3), ly0 + 2 * lh + 2, Math.max(1, X(Cc.e) - X(Cc.s + 0.3) - 1), lh - 12, { r: 2, fill: V.alpha(col, 0.85) });
          if (X(Cc.e) - X(Cc.s) > 14) T(g, String(i), X(Cc.s + 0.65), ly0 + 2 * lh + (lh - 10) / 2 + 5, { size: 11, align: 'center', font: 'mono', color: onTxt(C) });
        });
        const tp = (t * 3.5) % (maxT + 6), tNow = Math.min(tp, maxT), px = X(tNow);
        V.line(g, px, ly0 - 4, px, ly0 + 3 * lh - 6, { color: C.c, lw: 1.5 });
        V.line(g, X(p.total), ly0 + 2 * lh, X(p.total), ly0 + 3 * lh - 6, { color: C.d, lw: 2, dash: [3, 3] });
        T(g, `fim: ${comma(p.total.toFixed(1))} u`, Math.min(X(p.total) + 4, x1 - 70), ly0 + 3 * lh + 8, { size: 11, font: 'mono', color: C.d });
        const by = ly0 + 3 * lh + 30, bh = h - by - 10, half = (w - 28) / 2;
        const rcx = 14 + half / 2, R = Math.max(24, Math.min(half * 0.3, bh * 0.32)), rcy = by + 22 + R;
        T(g, 'buffer circular de SMEM', rcx, by + 4, { size: 12, weight: 600, align: 'center', color: C.ink, maxW: half });
        for (let k = 0; k < S.st; k++) {
          const a0 = -Math.PI / 2 + (k * 2 * Math.PI) / S.st + 0.04, a1 = a0 + (2 * Math.PI) / S.st - 0.08;
          let occ = -1, loading = false;
          for (let i = k; i < 10; i += S.st) if (p.ld[i].s <= tNow && tNow < p.cp[i].e) { occ = i; loading = tNow < p.ld[i].e; }
          g.beginPath(); g.arc(rcx, rcy, R, a0, a1); g.arc(rcx, rcy, R * 0.56, a1, a0, true); g.closePath();
          g.fillStyle = occ < 0 ? V.alpha(C.muted, 0.12) : V.alpha(C.series[occ % 6], loading ? 0.35 : 0.9); g.fill();
          if (occ >= 0 && S.st <= 6) { const am = (a0 + a1) / 2; T(g, String(occ), rcx + Math.cos(am) * R * 0.78, rcy + Math.sin(am) * R * 0.78 + 4, { size: 11, align: 'center', font: 'mono', color: C.ink }); }
        }
        T(g, Math.round(p.util * 100) + '%', rcx, rcy + 5, { size: 14, weight: 700, align: 'center', color: C.ink });
        T(g, 'uso do tensor core', rcx, rcy + R + 16, { size: 11, align: 'center', color: C.muted });
        const sx0 = 14 + half + 10, sw = half - 10;
        T(g, S.swz ? 'com swizzle XOR' : 'sem swizzle', sx0 + sw / 2, by + 4, { size: 12, weight: 600, align: 'center', color: S.swz ? C.a : C.c, maxW: sw });
        const n = 8, cs = Math.max(8, Math.min((sw - 16) / n, (bh - 64) / n)), gx0 = sx0 + (sw - cs * n) / 2, gy0 = by + 14;
        const col = 2, hiR = Math.floor(t * 1.5) % n;
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
          const phys = S.swz ? col ^ r : col, acc = c === phys;
          V.box(g, gx0 + c * cs + 1, gy0 + r * cs + 1, cs - 2, cs - 2, { r: 2, fill: acc ? V.alpha(S.swz ? C.a : C.c, r === hiR ? 0.95 : 0.6) : V.alpha(C.muted, 0.08) });
        }
        const nl2 = wrap(g, 'linha = thread do warp lendo a coluna 2 · coluna = grupo de 4 bancos (16 B)', sx0 + sw / 2, gy0 + n * cs + 14, sw, 13, { size: 11, align: 'center', color: C.muted, maxLines: 3 });
        T(g, S.swz ? '8 acessos em 8 bancos → 1 ciclo' : '8 acessos no mesmo banco → 8 ciclos', sx0 + sw / 2, gy0 + n * cs + 18 + nl2 * 13, { size: 11.5, align: 'center', color: C.ink, weight: 600, maxW: sw });
      }
      V.fig(fig, (g, w, h, t) => (S.mode === 'tiles' ? drawTiles(g, w, h, t) : drawPipe(g, w, h, t)));
    }

    /* =============== Figura 2b: linguagens de kernel =============== */
    {
      const fig = $('#fig-linguagens');
      const WS = [1, 2, 4, 8];
      const S = { lang: 'cuda', wi: 2, bn: 32 };
      const ROWS = ['threads → dados', 'global → SMEM, estágios', 'swizzle da SMEM', 'ldmatrix + mma', 'epílogo (fusões)', 'compilação'];
      const LANGS = {
        cuda: {
          name: 'CUDA C++', comp: 'AOT · nvcc no build', src: 'trecho real, abreviado · overlay/exl3_fat_gemm.cu (E2 do GLM)',
          code: ['__global__ __launch_bounds__(256)  // 8 warps', 'void exl3_fat_gemm_kernel(const half* a, ...)', '{', '  int warp = threadIdx.x / 32;', '  int lane = threadIdx.x & 31;', '  int m_base = blockIdx.y * FAT_TILE_M;', '  for (int k_block = 0; ...; ++k_block) {', '    sh_a[...] = a_src[a_col8];   // → SMEM', '    __syncthreads();', '    dq_dispatch<4,1>(warp_b, lane << 3, ...);', '    ldsm4(frag_a, sh_a + swizzled_col);', '    ptx_mma_m16n8k16(frag_a, frag_b0, ...);'],
          who: [['você: threadIdx → linha', 'y'], ['você: cópias + barreiras', 'y'], ['você: XOR à mão', 'y'], ['você: PTX inline', 'y'], ['você', 'y'], ['nvcc no build (AOT)', 'c']],
          unit: 'você escreve o código de 1 thread; o warp é explícito', rec: 'exllamav3_ext (E2, E3, FAST) · MoE cooperativo · FlashKDA · decode GDN do Qwen · DeepGEMM (CUDA em JIT)',
        },
        cutlass: {
          name: 'CUTLASS · CuTe', comp: 'nvcc: no build ou no JIT do FlashInfer', src: 'esboço ilustrativo (não é código das receitas)',
          code: ['using TileShape = Shape<_64, _32, _64>;', '// mainloop "coletivo": estágios no template', 'using Mainloop = collective::CollectiveMma<', '    MainloopSm80CpAsync<4>, TileShape, ...>;', 'using Epilogue = collective::DefaultEpilogue<', '    LinearCombination<...>>;   // escala etc.', 'using Gemm = GemmUniversal<Problem,', '                           Mainloop, Epilogue>;', '// CuTe: Layout = forma:passo', '//   (64, 32):(32, 1) → linha, coluna'],
          who: [['você descreve (layout CuTe)', 'y'], ['template: estágios', 'b'], ['layout CuTe', 'b'], ['template: átomo MMA', 'b'], ['você escolhe o template', 'y'], ['nvcc (AOT ou JIT)', 'c']],
          unit: 'você escolhe tiles e epílogo; o template reparte o trabalho', rec: 'MoE FlashInfer CUTLASS (Qwen, W4A4 NVFP4) · CutlassFP8ScaledMM (Qwen FP8-dense) · W4A4 do plugin híbrido do DeepSeek',
        },
        triton: {
          name: 'Triton', comp: 'JIT na 1ª chamada de cada variante', src: 'trecho real, abreviado · QSA do Qwen (patch_qsa_fp8_kv_v030.py)',
          code: ['@triton.jit', 'def _qsa_sparse_paged_gqa_splitk_kernel(', '    ..., BLOCK_N: tl.constexpr,', '    IS_FP8: tl.constexpr):', '  row = tl.program_id(0)     # qual linha', '  kv_head = tl.program_id(1)', '  for t in range(..., tile_end):', '    keys = tl.load(...)      # bloco inteiro', '    if IS_FP8:', '      keys = keys.to(query.dtype)', '    scores = tl.dot(query, keys)', '    alpha = tl.math.exp2(max_value - next_max)', '# kernel[grid](..., BLOCK_N=block_n,', '#             num_warps=partial_warps)'],
          who: [['compilador (num_warps)', 'c'], ['compilador (num_stages)', 'c'], ['compilador', 'c'], ['compilador (tl.dot)', 'c'], ['você: código após o laço', 'y'], ['JIT (cache em disco)', 'c']],
          unit: 'você escreve 1 programa = 1 tile; o compilador reparte entre warps', rec: 'QSA do Qwen · chunk_kda (FLA) · kernels do DSpark · kernels de cache do DeepSeek (cache_utils.py)',
        },
        cutedsl: {
          name: 'CuTeDSL', comp: 'JIT (@cute.jit) na 1ª chamada', src: 'esboço ilustrativo (não é código das receitas)',
          code: ['import cutlass.cute as cute', '', '@cute.kernel', 'def gemm_kernel(gA, gB, gC, tiler):', '    bidx, bidy, _ = cute.arch.block_idx()', '    tA = cute.local_tile(gA, tiler, ...)', '    ...  # mesmos layouts e átomos MMA da CuTe', '', '@cute.jit              # compila aqui', 'def launch(A, B, C):', '    gemm_kernel(A, B, C, tiler).launch(', '        grid=..., block=[128, 1, 1])'],
          who: [['você descreve (layout CuTe)', 'y'], ['você + biblioteca', 'b'], ['layout CuTe', 'b'], ['átomo MMA da CuTe', 'b'], ['você', 'y'], ['JIT (@cute.jit)', 'c']],
          unit: 'CuTe em Python: layouts explícitos, sem compilar C++ no build', rec: 'b12x do DeepSeek (MoE W4A16, mHC, MLA) · GEMM skinny do vLLM (só SM103)',
        },
        tilelang: {
          name: 'TileLang', comp: 'JIT: gera CUDA e chama o nvcc', src: 'esboço ilustrativo (não é código das receitas)',
          code: ['@T.prim_func', 'def gemm(A: T.Tensor((M, K)), B, C):', '  with T.Kernel(N // BN, M // BM,', '                threads=128) as (bx, by):', '    A_s = T.alloc_shared((BM, BK))', '    B_s = T.alloc_shared((BK, BN))', '    C_l = T.alloc_fragment((BM, BN))', '    for k in T.Pipelined(K // BK,', '                         num_stages=3):', '      T.copy(A[by * BM, k * BK], A_s)', '      T.copy(B[k * BK, bx * BN], B_s)', '      T.gemm(A_s, B_s, C_l)', '    T.copy(C_l, C[by * BM, bx * BN])'],
          who: [['compilador (threads=…)', 'c'], ['você declara: T.Pipelined', 'y'], ['compilador', 'c'], ['T.gemm (biblioteca)', 'b'], ['você', 'y'], ['JIT (gera CUDA)', 'c']],
          unit: 'você declara buffers e o laço em tiles; o compilador mapeia threads', rec: 'mHC do DeepSeek (hiperconexões fundidas)',
        },
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Linguagem', options: Object.entries(LANGS).map(([v, L]) => ({ v, label: L.name })), value: 'cuda', onChange: (v) => { S.lang = v; upd(); } });
      UI.slider(ctl, { label: 'num_warps', min: 0, max: 3, step: 1, value: 2, fmt: (i) => String(WS[i]), onInput: (v) => { S.wi = v; upd(); } });
      UI.seg(ctl, { label: 'BLOCK_N', options: [16, 32, 64].map((v) => ({ v, label: String(v) })), value: 32, onChange: (v) => { S.bn = v; upd(); } });
      const rC = UI.readout(ctl, 'compilação', '—'), rA = UI.readout(ctl, 'acumuladores fp32/thread', '—'), rI = UI.readout(ctl, 'iterações p/ 2.048 chaves', '—');
      function upd() { const L = LANGS[S.lang], W_ = WS[S.wi]; rC.set(L.comp); rA.set(String((64 * S.bn) / (32 * W_))); rI.set(String(2048 / S.bn)); }
      upd();
      const geo = (w) => {
        const wide = w >= 720, lw = wide ? Math.round(w * 0.46) : w - 32;
        const cell = Math.max(6, Math.min((lw - 64) / 16, 20)), th = 8 * cell;
        const stY = 52 + th + 14, qy = stY + 50, legRow = lw < 420 ? 18 : 0;
        const rowsEnd = qy + 10 + legRow + ROWS.length * 22;
        const codeTop = wide ? 8 : rowsEnd + 16;
        const h = wide ? Math.max(rowsEnd + 16, 8 + 26 + 14 + 14 * 15 + 8 + 18 + 17 + 4 * 14 + 10) : codeTop + 40 + 14 * 14 + 8 + 18 + 17 + 4 * 14 + 10;
        return { wide, lw, cell, th, stY, qy, legRow, codeTop, h };
      };
      fitH(fig, (w) => geo(w).h);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), L = LANGS[S.lang], W_ = WS[S.wi], G = geo(w);
        const KC = { y: C.c, b: C.b, c: C.a };
        const lx = 16, lw = G.lw;
        // ---- tile e warps
        T(g, `${L.name} · ${L.unit}`, lx, 16, { size: 12, weight: 600, color: C.ink, maxW: lw });
        const cols = S.bn / 4, rowsN = 8, cell = G.cell, tw = cols * cell, th = G.th, tx = lx + 52, ty = 52;
        T(g, `tile de scores [64 × ${S.bn}]`, tx, 44, { size: 11, color: C.muted, font: 'mono' });
        const nb = Math.min(W_, rowsN), rowsPerBand = rowsN / nb;
        for (let r = 0; r < rowsN; r++) for (let c = 0; c < cols; c++) {
          const band = Math.floor(r / rowsPerBand);
          V.box(g, tx + c * cell + 0.5, ty + r * cell + 0.5, cell - 1, cell - 1, { r: 2, fill: V.alpha(C.series[band % 6], S.lang === 'cuda' ? 0.34 : 0.22) });
        }
        for (let k = 1; k < nb; k++) { const yy = ty + (k * th) / nb; V.line(g, tx - 4, yy, tx + tw + 4, yy, { color: V.alpha(C.ink, 0.5), lw: 1.2, dash: S.lang === 'cuda' ? [] : [4, 3] }); }
        for (let k = 0; k < nb; k++) T(g, `w${k}`, tx - 8, ty + ((k + 0.5) * th) / nb + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        if (S.lang === 'triton' || S.lang === 'tilelang') { g.strokeStyle = C.a; g.lineWidth = 2.5; V.rr(g, tx - 2, ty - 2, tw + 4, th + 4, 4); g.stroke(); }
        const sideX = tx + tw + 14, sideW = lx + lw - sideX;
        if (S.lang === 'cuda') {
          const bandH = th / nb, ln = Math.floor(t * 6) % 32, stepX = (tw - 4) / 8;
          for (let q = 0; q < 32; q++) {
            const cx = tx + 2 + (q % 8) * stepX + stepX / 2, cy = ty + (Math.floor(q / 8) + 0.5) * (bandH / 4);
            V.dot(g, cx, cy, Math.max(1.6, Math.min(3, (bandH / 4) * 0.3)), q === ln ? C.c : V.alpha(C.ink, 0.45));
          }
          if (sideW > 90) wrap(g, `lane ${ln} = threadIdx.x & 31: você calcula que linha e coluna ela toca`, sideX, ty + 12, sideW, 14, { size: 11, color: C.c, maxLines: 6 });
          else T(g, `lane ${ln} = threadIdx.x & 31`, tx, ty + th + 12, { size: 11, color: C.c, font: 'mono' });
        } else if (sideW > 90) {
          const msg = S.lang === 'triton' ? `1 programa = este tile inteiro (tl.program_id); o compilador reparte entre num_warps = ${W_}` : S.lang === 'tilelang' ? `T.Kernel(threads=${32 * W_}): o compilador reparte o tile entre ${W_} warp(s)` : `o template (TiledMMA, layout CuTe) reparte o tile entre ${W_} warp(s)`;
          wrap(g, msg, sideX, ty + 12, sideW, 14, { size: 11, color: C.muted, maxLines: 6 });
        }
        const stY = G.stY + (S.lang === 'cuda' && sideW <= 90 ? 6 : 0), nS = 3;
        const stLab = S.lang === 'cuda' ? 'estágios: você escreve cp.async + buffers' : S.lang === 'triton' ? 'estágios: num_stages (o compilador gera)' : S.lang === 'tilelang' ? 'estágios: T.Pipelined(num_stages=3)' : 'estágios: parâmetro do template';
        T(g, stLab, tx, stY + 10, { size: 11, color: C.muted, maxW: lw - 52 });
        for (let k = 0; k < nS; k++) {
          const u = (t * 0.8 + k / nS) % 1, bx = tx + k * 46;
          V.box(g, bx, stY + 18, 40, 14, { r: 3, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.4), lw: 1 });
          V.box(g, bx, stY + 18, Math.max(2, 40 * u), 14, { r: 3, fill: V.alpha(C.d, 0.7) });
        }
        // ---- quem escreve o quê
        const qy = G.qy;
        T(g, 'quem escreve cada passo', lx, qy, { size: 12, weight: 600, color: C.ink });
        const leg = [['você', C.c], ['biblioteca', C.b], ['compilador', C.a]];
        g.font = fnt({ size: 11 });
        if (G.legRow) { let lgx = lx; leg.forEach(([lb, col]) => { V.box(g, lgx, qy + 6, 10, 10, { r: 3, fill: V.alpha(col, 0.8) }); T(g, lb, lgx + 14, qy + 15, { size: 11, color: C.muted }); g.font = fnt({ size: 11 }); lgx += g.measureText(lb).width + 30; }); }
        else { let lgx = lx + lw; for (let i = leg.length - 1; i >= 0; i--) { const [lb, col] = leg[i]; g.font = fnt({ size: 11 }); lgx -= g.measureText(lb).width + 20; V.box(g, lgx, qy - 9, 10, 10, { r: 3, fill: V.alpha(col, 0.8) }); T(g, lb, lgx + 14, qy, { size: 11, color: C.muted }); } }
        const hiRow = Math.floor(t / 1.6) % ROWS.length, colW = Math.min(150, lw * 0.42), r0 = qy + 10 + G.legRow;
        ROWS.forEach((rn, i) => {
          const y = r0 + i * 22, [lab, k] = L.who[i];
          if (i === hiRow) V.box(g, lx - 4, y - 1, lw + 8, 21, { r: 5, fill: V.alpha(C.ink, 0.06) });
          T(g, rn, lx, y + 14, { size: 11, color: C.ink, maxW: colW - 6 });
          V.box(g, lx + colW, y + 2, lw - colW, 17, { r: 8, fill: V.alpha(KC[k], 0.2), stroke: V.alpha(KC[k], 0.7), lw: 1 });
          T(g, lab, lx + colW + 8, y + 14, { size: 11, color: C.ink, maxW: lw - colW - 14 });
        });
        // ---- código
        const cx0 = G.wide ? lx + lw + 20 : lx, cw = G.wide ? w - 16 - cx0 : w - 32, cy0 = G.codeTop;
        const lh = G.wide ? 15 : 14, nL = L.code.length;
        const srcN = Math.min(2, splitLines(g, L.src, cw - 20, { size: 11 }).length), codeY = cy0 + 10 + srcN * 14;
        const chH = codeY - cy0 + nL * lh + 8;
        V.box(g, cx0, cy0, cw, chH, { r: 10, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        wrap(g, L.src, cx0 + 10, cy0 + 16, cw - 20, 14, { size: 11, color: C.muted, maxLines: 2 });
        const hl = Math.floor(t / 1.1) % nL;
        L.code.forEach((ln, i) => {
          const yy = codeY + i * lh, tr = ln.trim();
          if (i === hl && tr) V.box(g, cx0 + 4, yy, cw - 8, lh, { r: 3, fill: V.alpha(C.d, 0.16) });
          T(g, ln, cx0 + 10, yy + lh - 3.5, { size: 11, font: 'mono', color: tr.startsWith('#') || tr.startsWith('//') ? C.muted : C.ink, maxW: cw - 18 });
        });
        const fy = cy0 + chH + 18;
        T(g, 'compila: ' + L.comp, cx0, fy, { size: 11.5, weight: 600, color: C.a, maxW: cw });
        wrap(g, 'nas receitas: ' + L.rec, cx0, fy + 17, cw, 14, { size: 11, color: C.ink, maxLines: 4 });
      });
    }

    /* =============== Figura 3: fusão =============== */
    {
      const fig = $('#fig-fusao');
      fitH(fig, (w) => (w < 600 ? 520 : 420));
      const S = { fa: false, fs: false, det: false, mi: 10 };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'SiLU·mul no epílogo do gate/up', value: false, onChange: (v) => { S.fa = v; upd(); } });
      UI.toggle(ctl, { label: 'peso de rota + soma no epílogo do down', value: false, onChange: (v) => { S.fs = v; upd(); } });
      UI.toggle(ctl, { label: 'finalize em ordem fixa', value: false, onChange: (v) => { S.det = v; } });
      UI.slider(ctl, { label: 'M (linhas)', min: 0, max: 13, step: 1, value: 10, fmt: (i) => String(Ms[i]), onInput: (v) => { S.mi = v; upd(); } });
      const rL = UI.readout(ctl, 'lançamentos', '—'), rB = UI.readout(ctl, 'intermediários na DRAM', '—');
      const H_ = 4096, I_ = 2048;
      const kernels = () => {
        const ks = [];
        if (!S.fa) { ks.push({ l: 'GEMM gate/up', ep: '', out: 'g,u [M×2I]', outE: 2 * I_ }); ks.push({ l: 'SiLU · mul', ep: '', out: 'a [M×I]', outE: I_, noW: true }); }
        else ks.push({ l: 'GEMM gate/up', ep: 'SiLU·mul', out: 'a [M×I]', outE: I_ });
        if (!S.fs) { ks.push({ l: 'GEMM down', ep: '', out: 'y [M×H]', outE: H_ }); ks.push({ l: '× peso · Σ top-k', ep: '', out: '', outE: 0, noW: true }); }
        else ks.push({ l: 'GEMM down', ep: 'peso · Σ', out: '', outE: 0 });
        return ks;
      };
      const perRow = () => kernels().reduce((s, k) => s + 2 * k.outE, 0) * 2;
      const base = (2 * (2 * I_) + 2 * I_ + 2 * H_) * 2;
      function upd() { rL.set(String(kernels().length)); rB.set(fmtB(perRow() * Ms[S.mi])); }
      upd();
      const E = [1.1, 2.2, 3.3].map(Math.fround);
      const orders = [[0, 1, 2], [1, 2, 0], [0, 2, 1]];
      const sumOrd = (o) => Math.fround(Math.fround(E[o[0]] + E[o[1]]) + E[o[2]]);
      const ref = sumOrd(orders[0]);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), ks = kernels(), n = ks.length;
        const top = 22, bxH = 70, gapX = w < 560 ? 12 : 26;
        const bw = (w - 32 - gapX * (n - 1)) / n;
        const dy = top + bxH + 86, dH = 34;
        V.box(g, 16, dy, w - 32, dH, { r: 10, fill: V.alpha(C.b, 0.14), stroke: V.alpha(C.b, 0.5) });
        T(g, 'DRAM (LPDDR5X · ~273 GB/s)', w - 26, dy + dH / 2 + 4, { size: 11.5, color: C.b, font: 'mono', weight: 600, align: 'right' });
        ks.forEach((k, i) => {
          const x = 16 + i * (bw + gapX);
          V.box(g, x, top, bw, bxH, { r: 10, fill: V.alpha(C.a, 0.12), stroke: C.a, lw: 1.5 });
          wrap(g, k.l, x + bw / 2, top + 20, bw - 10, 14, { size: 12, weight: 600, align: 'center', color: C.ink, maxLines: 2 });
          if (k.ep) {
            V.box(g, x + 6, top + bxH - 24, bw - 12, 18, { r: 5, fill: V.alpha(C.d, 0.3) });
            T(g, 'epílogo: ' + k.ep, x + bw / 2, top + bxH - 11, { size: 11, align: 'center', color: C.ink, maxW: bw - 14 });
            for (let p = 0; p < 3; p++) { const u = (t * 0.7 + p / 3) % 1; V.dot(g, x + 10 + u * (bw - 20), top + bxH - 30, 2.6, C.d); }
          }
          if (!k.noW) {
            const xw = x + bw * 0.14;
            V.arrow(g, xw, dy - 2, xw, top + bxH + 3, { color: V.alpha(C.muted, 0.55), lw: 1.2, head: 6, dash: [3, 3] });
            T(g, 'W', xw + 5, dy - 8, { size: 11, color: C.muted, font: 'mono' });
          }
          if (i < n - 1 && k.outE > 0) {
            const xa = x + bw * 0.62, xb = 16 + (i + 1) * (bw + gapX) + bw * 0.38;
            const y0 = top + bxH, y1 = dy;
            V.arrow(g, xa, y0 + 2, xa, y1 - 2, { color: C.c, lw: 1.6, head: 7 });
            V.arrow(g, xb, y1 - 2, xb, y0 + 2, { color: C.c, lw: 1.6, head: 7 });
            T(g, k.out, (xa + xb) / 2, y0 + 30, { size: 11, align: 'center', font: 'mono', color: C.c, maxW: xb - xa + 30 });
            for (let p = 0; p < 3; p++) {
              const u = (t * 0.45 + p / 3) % 1; let px, py;
              if (u < 0.4) { px = xa; py = V.lerp(y0, y1, u / 0.4); } else if (u < 0.6) { px = V.lerp(xa, xb, (u - 0.4) / 0.2); py = y1 + 8; } else { px = xb; py = V.lerp(y1, y0, (u - 0.6) / 0.4); }
              V.dot(g, px, py, 3.2, C.c);
            }
          }
        });
        const narrow = w < 600;
        const py = dy + dH + 30, barW = narrow ? w - 32 : (w - 32) * 0.56;
        const unf = base * Ms[S.mi], cur = perRow() * Ms[S.mi];
        T(g, `tráfego de intermediários · M = ${Ms[S.mi]} linhas`, 16, py, { size: 12, weight: 600, color: C.ink, maxW: barW });
        V.box(g, 16, py + 10, barW, 20, { r: 5, fill: V.alpha(C.c, 0.25) });
        T(g, `sem fusão: ${fmtB(unf)} · 4 lançamentos`, 22, py + 24, { size: 11.5, color: C.ink, maxW: barW - 12 });
        V.box(g, 16, py + 36, Math.max(4, (barW * cur) / unf), 20, { r: 5, fill: V.alpha(C.a, 0.8) });
        T(g, `agora: ${fmtB(cur)} · ${n} lançamentos`, 22, py + 50, { size: 11.5, color: C.ink, weight: 600, maxW: barW - 12 });
        wrap(g, 'Os pesos (setas W) são lidos igual nos três arranjos: a fusão poupa ativações e lançamentos.', 16, py + 76, barW, 14, { size: 11, color: C.muted });
        const dx = narrow ? 16 : 16 + barW + 20, dw = narrow ? w - 32 : w - 32 - barW - 20, dyy = narrow ? py + 110 : py - 8;
        V.box(g, dx, dyy, dw, 104, { r: 10, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, 0.4) });
        T(g, 'finalize: soma de 3 experts (FP32)', dx + 10, dyy + 20, { size: 12, weight: 600, color: C.ink, maxW: dw - 20 });
        const oi = S.det ? 0 : Math.floor(t / 1.4) % 3, o = orders[oi], r = sumOrd(o), nm = ['1,1', '2,2', '3,3'];
        T(g, `(${nm[o[0]]} + ${nm[o[1]]}) + ${nm[o[2]]}`, dx + 10, dyy + 44, { size: 12.5, font: 'mono', color: C.ink });
        T(g, '= ' + comma(r.toPrecision(10)), dx + 10, dyy + 66, { size: 13, font: 'mono', weight: 700, color: r === ref ? C.a : C.c });
        T(g, S.det ? 'ordem fixa → mesmos bits sempre' : 'ordem varia → o último bit muda', dx + 10, dyy + 90, { size: 11, color: C.muted, maxW: dw - 20 });
      });
    }

    /* =============== Figura 4: softmax online (stepper) =============== */
    {
      const fig = $('#fig-online');
      fitH(fig, (w) => (w < 600 ? 370 : 380));
      const s = [1.2, -0.3, 2.1, 0.4, 3.0, -1.0, 0.8, 2.6, 1.1];
      const v = [0.5, -1.0, 2.0, 0.3, 1.5, -0.7, 0.9, 0.6, -0.4];
      const blocks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
      const mx = Math.max(...s), ex = s.map((x) => Math.exp(x - mx)), Z = ex.reduce((a, b) => a + b, 0);
      const oExact = ex.reduce((a, e, j) => a + e * v[j], 0) / Z;
      const st = []; let m = -Infinity, l = 0, acc = 0;
      blocks.forEach((B) => {
        const mb = Math.max(...B.map((j) => s[j])), m2 = Math.max(m, mb), a = Math.exp(m - m2);
        l = a * l + B.reduce((q, j) => q + Math.exp(s[j] - m2), 0);
        acc = a * acc + B.reduce((q, j) => q + Math.exp(s[j] - m2) * v[j], 0);
        st.push({ m: m2, l, acc, alpha: a }); m = m2;
      });
      const parts = blocks.map((B) => { const mp = Math.max(...B.map((j) => s[j])); const lp = B.reduce((q, j) => q + Math.exp(s[j] - mp), 0); const op = B.reduce((q, j) => q + Math.exp(s[j] - mp) * v[j], 0); return { m: mp, l: lp, o: op, lse: mp + Math.log(lp) }; });
      const LSE = Math.log(parts.reduce((q, p) => q + Math.exp(p.lse), 0));
      const oMerge = parts.reduce((q, p) => q + (Math.exp(p.lse - LSE) * p.o) / p.l, 0);
      const f = (x, d = 3) => comma(x.toFixed(d));
      const fm = (x, d = 3) => f(x, d).replace(',', '{,}');
      const steps = [
        { t: 'Uma linha de scores', d: 'Para uma query, \\(s_j = q\\cdot k_j/\\sqrt d\\) para 9 chaves, com valores \\(v_j\\) (escalares aqui; vetores na vida real). A atenção clássica calcularia todos os scores, o softmax e a soma ponderada. O FlashAttention percorre <b>blocos de 3 chaves</b>, como faria uma CTA.' },
        { t: 'Bloco 1', d: `Máximo parcial \\(m = ${fm(st[0].m, 1)}\\). Somamos \\(\\ell = \\sum e^{s_j-m}\\) e \\(o = \\sum e^{s_j-m}v_j\\) só destas 3 chaves: \\(\\ell = ${fm(st[0].l)}\\), \\(o = ${fm(st[0].acc)}\\).` },
        { t: 'Bloco 2: apareceu um máximo maior', d: `Chega o score 3,0, maior que 2,1. Tudo o que já somamos estava na escala do máximo antigo, então multiplicamos por \\(\\alpha = e^{2{,}1-3{,}0} = ${fm(st[1].alpha)}\\) antes de somar o bloco novo: \\(\\ell = ${fm(st[1].l)}\\), \\(o = ${fm(st[1].acc)}\\). Repare nas barras antigas encolhendo.` },
        { t: 'Bloco 3: o máximo não muda', d: `O maior score do bloco (2,6) é menor que 3,0: \\(\\alpha = 1\\), só somamos. \\(\\ell = ${fm(st[2].l)}\\), \\(o = ${fm(st[2].acc)}\\). Nenhuma matriz de scores foi guardada: só três números por linha.` },
        { t: 'Normaliza no fim', d: `Saída \\(o/\\ell = ${fm(st[2].acc / st[2].l, 4)}\\). O softmax exato, calculado com todos os scores de uma vez, dá \\(${fm(oExact, 4)}\\): <b>idêntico</b>. O FlashAttention é exato, não uma aproximação.` },
        { t: 'Prefill: blocos de queries × blocos de chaves', d: 'No prefill há milhares de queries. Cada CTA pega um <b>bloco de \\(B_q\\) queries</b> (uma linha da grade) e percorre os blocos de \\(B_k\\) chaves e valores da esquerda para a direita, trazendo cada um à SMEM <i>uma vez</i> para as \\(B_q\\) queries. Cada uma das \\(B_q\\) linhas carrega seu próprio \\((m, \\ell, o)\\), atualizado com a regra dos passos 2–4. Só existe um tile \\(B_q\\times B_k\\) de scores por vez: a matriz n × n nunca é montada (tamanhos ilustrativos: n = 384, blocos de 64).' },
        { t: 'Máscara causal: pular blocos inteiros', d: 'A query \\(i\\) só vê chaves \\(j \\le i\\). Blocos inteiramente acima da diagonal não contribuem com nada, e a CTA nem os carrega: 15 dos 36 somem. Só os 6 blocos da diagonal aplicam a máscara elemento a elemento; os 15 abaixo dela entram inteiros. Quase metade do trabalho do prefill desaparece, sem aproximação nenhuma.' },
        { t: 'Split-K: 3 CTAs em paralelo', d: 'No decode queremos mais paralelismo: cada bloco vai para uma CTA diferente, que calcula seus próprios \\(m_p, \\ell_p, o_p\\) sem conversar com as outras e guarda \\(\\mathrm{LSE}_p = m_p + \\ln \\ell_p\\) (cada uma na sua escala, repare nas barras).' },
        { t: 'Merge pelo log-sum-exp', d: `\\(\\mathrm{LSE} = \\ln\\sum_p e^{\\mathrm{LSE}_p} = ${fm(LSE)}\\), e cada parcial normalizado \\(o_p/\\ell_p\\) entra com peso \\(e^{\\mathrm{LSE}_p-\\mathrm{LSE}}\\). Resultado: \\(${fm(oMerge, 4)}\\), o mesmo valor. Mais partições = mais CTAs, mas também mais parciais para o merge ler.` },
        { t: 'exp2 no lugar de exp', d: 'O hardware calcula \\(2^x\\) mais barato que \\(e^x\\). Como \\(e^{x} = 2^{x\\log_2 e}\\), os kernels do QSA multiplicam o <code>softmax_scale</code> por \\(\\log_2 e \\approx 1{,}4427\\) uma vez e usam <code>exp2</code> dentro do laço.' },
      ];
      const stp = UI.stepper(fig, steps, { autoplay: 7000 });
      function draw2D(g, w, h, t, causal) {
        const C = V.C(), narrow = w < 600, NB = 6;
        const G = narrow ? Math.min(w - 64, h - 150) : Math.min(h - 70, w * 0.42), cs = G / NB;
        const gx0 = narrow ? (w - G) / 2 + 8 : 56, gy0 = 40;
        T(g, 'blocos de K/V (chaves) →', gx0, gy0 - 22, { size: 11, color: C.muted });
        for (let c = 0; c < NB; c++) T(g, 'K' + c, gx0 + (c + 0.5) * cs, gy0 - 6, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        const per = 0.55, cyc = (NB + 1.5) * per, tc = t % cyc;
        let done = 0;
        for (let r = 0; r < NB; r++) {
          T(g, 'Q' + r, gx0 - 8, gy0 + (r + 0.5) * cs + 4, { size: 11, align: 'right', color: C.series[r % 6], font: 'mono', weight: 600 });
          const lim = causal ? r + 1 : NB, cur = Math.min(lim, Math.floor(tc / per));
          for (let c = 0; c < NB; c++) {
            const x = gx0 + c * cs, y = gy0 + r * cs;
            const skip = causal && c > r;
            if (skip) {
              V.box(g, x + 1, y + 1, cs - 2, cs - 2, { r: 4, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.2), lw: 1 });
              V.line(g, x + cs * 0.32, y + cs * 0.32, x + cs * 0.68, y + cs * 0.68, { color: V.alpha(C.muted, 0.55), lw: 1.3 });
              V.line(g, x + cs * 0.68, y + cs * 0.32, x + cs * 0.32, y + cs * 0.68, { color: V.alpha(C.muted, 0.55), lw: 1.3 });
              continue;
            }
            const computed = c < cur, active = c === cur && cur < lim;
            if (computed) done++;
            V.box(g, x + 1, y + 1, cs - 2, cs - 2, { r: 4, fill: V.alpha(C.series[r % 6], computed ? 0.55 : 0.1), stroke: active ? C.c : V.alpha(C.series[r % 6], 0.45), lw: active ? 2.5 : 1 });
            if (causal && c === r) { g.fillStyle = V.alpha(C.bg || C.panel, 0.75); g.beginPath(); g.moveTo(x + 2, y + 2); g.lineTo(x + cs - 2, y + 2); g.lineTo(x + cs - 2, y + cs - 2); g.closePath(); g.fill(); }
          }
        }
        const tot = causal ? 21 : 36;
        const px = narrow ? 16 : gx0 + G + 28, pw = narrow ? w - 32 : w - px - 16, py = narrow ? gy0 + G + 14 : gy0;
        const chip = (x, y, cw, k, v, col) => { V.box(g, x, y, cw, 34, { r: 8, fill: V.alpha(col, 0.14), stroke: V.alpha(col, 0.6) }); T(g, k, x + 8, y + 13, { size: 11, color: C.muted, maxW: cw - 12 }); T(g, v, x + cw - 8, y + 27, { size: 12.5, font: 'mono', weight: 700, align: 'right', color: C.ink }); };
        const n3 = narrow ? 3 : 1, cw = narrow ? (pw - 16) / 3 : Math.min(260, pw);
        const items = [['blocos calculados', `${Math.min(done, tot)} / ${tot}`, C.a], ['pulados (máscara)', causal ? '15' : '0', C.c], ['estado por linha de Q', 'm, ℓ, o', C.b]];
        items.forEach(([k, v, col], i) => { const x = narrow ? px + i * (cw + 8) : px, y = narrow ? py : py + i * 44; chip(x, y, cw, k, v, col); });
        const ty = narrow ? py + 50 : py + 3 * 44 + 8;
        if (!narrow) {
          T(g, 'tile de scores Bq × Bk (registradores)', px, ty + 4, { size: 11, color: C.muted, maxW: pw });
          const m = 8, mc = Math.min(16, (Math.min(260, pw) - 10) / m);
          for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
            const masked = causal && b > a;
            V.box(g, px + b * mc, ty + 12 + a * mc, mc - 1.5, mc - 1.5, { r: 2, fill: masked ? V.alpha(C.muted, 0.12) : V.alpha(C.d, 0.25 + 0.5 * (((a * 7 + b * 3) % 5) / 5)) });
          }
          wrap(g, causal ? 'no bloco da diagonal, as posições com j > i recebem −∞ antes do softmax' : 'vive só durante uma iteração: vira (m, ℓ, o) e é descartado', px, ty + 24 + m * mc, Math.min(300, pw), 14, { size: 11, color: C.muted, maxLines: 3 });
        } else {
          wrap(g, causal ? 'Blocos acima da diagonal (✕) nem são carregados; na diagonal, j > i recebe −∞.' : 'Cada linha da grade é uma CTA; só um tile Bq × Bk de scores existe por vez.', w / 2, ty + 12, w - 32, 15, { size: 12, align: 'center', color: C.ink, maxLines: 2 });
        }
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), i0 = stp.i, el = stp.elapsed();
        if (i0 === 5 || i0 === 6) { draw2D(g, w, h, el, i0 === 6); return; }
        const i = i0 >= 7 ? i0 - 2 : i0;
        const n = 9, cw = Math.min(64, (w - 40) / n - 4), gx = (w - (n * (cw + 4) - 4)) / 2, gy = 30, chh = 34;
        T(g, i >= 5 ? 'scores s_j divididos entre 3 CTAs' : 'scores s_j de uma query', gx, gy - 10, { size: 11.5, color: C.muted, font: 'mono' });
        const bNow = i === 0 ? -1 : i <= 3 ? i - 1 : 2;
        for (let j = 0; j < n; j++) {
          const x = gx + j * (cw + 4), b = Math.floor(j / 3), active = i >= 1 && i <= 3 && b === i - 1;
          const tt = ((s[j] + 1) / 4) * 0.9;
          const fill = i >= 5 ? V.alpha(C.series[b], 0.35) : V.cmap(tt, 'seq');
          V.box(g, x, gy, cw, chh, { r: 6, fill, stroke: active ? C.c : V.alpha(C.muted, 0.3), lw: active ? 2.5 : 1 });
          T(g, f(s[j], 1), x + cw / 2, gy + chh / 2 + 4, { size: 12, font: 'mono', align: 'center', color: i < 5 && tt > 0.6 ? onTxt(C) : C.ink });
          T(g, f(v[j], 1), x + cw / 2, gy + chh + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        }
        T(g, 'v', gx - 12, gy + chh + 14, { size: 11, color: C.muted, font: 'mono' });
        for (let b = 0; b < 3; b++) {
          const x0 = gx + b * 3 * (cw + 4), x1 = x0 + 3 * (cw + 4) - 4, y = gy + chh + 24;
          V.line(g, x0, y, x1, y, { color: i >= 5 ? C.series[b] : V.alpha(C.muted, 0.6), lw: 2 });
          T(g, (i >= 5 ? 'CTA ' : 'bloco ') + (b + 1), (x0 + x1) / 2, y + 14, { size: 11, align: 'center', color: C.muted });
        }
        const by0 = gy + chh + 74, bh = Math.min(112, h * 0.28);
        let mCur = i === 0 ? mx : i <= 3 ? st[i - 1].m : mx;
        if (i === 2) mCur = V.lerp(st[0].m, st[1].m, V.ease.inOut(V.clamp(el / 1.6, 0, 1)));
        T(g, i >= 5 ? 'pesos e^(s − m_p): cada CTA na sua própria escala' : 'pesos e^(s − m) na escala do máximo corrente', gx, by0 - 10, { size: 11.5, color: C.muted, maxW: w - 40 });
        for (let j = 0; j < n; j++) {
          const b = Math.floor(j / 3), x = gx + j * (cw + 4);
          const show = i >= 4 || (i > 0 && b <= bNow);
          if (!show) { V.box(g, x, by0 + bh - 2, cw, 2, { fill: V.alpha(C.muted, 0.3), r: 1 }); continue; }
          const val = i >= 5 ? Math.exp(s[j] - parts[b].m) : Math.exp(s[j] - mCur);
          const hh = Math.max(2, bh * 0.8 * Math.min(1, val));
          V.box(g, x + cw * 0.15, by0 + bh - hh, cw * 0.7, hh, { r: 4, fill: V.alpha(i >= 5 ? C.series[b] : C.a, 0.8) });
          if (cw >= 34) T(g, f(val, 2), x + cw / 2, by0 + bh - hh - 5, { size: 11, font: 'mono', align: 'center', color: C.ink });
        }
        V.line(g, gx, by0 + bh, gx + n * (cw + 4) - 4, by0 + bh, { color: V.alpha(C.muted, 0.4), lw: 1 });
        const py = by0 + bh + 22, W = w - 32;
        const chips = (items, y) => {
          const cwid = Math.min(150, W / items.length - 8), tot = items.length * (cwid + 8) - 8;
          let x = (w - tot) / 2;
          items.forEach(([k, val, col]) => { V.box(g, x, y, cwid, 36, { r: 8, fill: V.alpha(col || C.b, 0.14), stroke: V.alpha(col || C.b, 0.6) }); T(g, k, x + 8, y + 14, { size: 11, color: C.muted, font: 'mono' }); T(g, val, x + cwid - 8, y + 27, { size: 12.5, font: 'mono', weight: 700, align: 'right', color: C.ink }); x += cwid + 8; });
        };
        if (i === 0) wrap(g, 'Clássico: materializar os n scores (e, no prefill, a matriz n × n). Online: guardar só m, ℓ e o.', w / 2, py + 12, W, 15, { size: 12, align: 'center', color: C.ink });
        if (i >= 1 && i <= 4) {
          const S_ = st[Math.min(2, i - 1)];
          const items = [['m', f(S_.m, 2)], ['ℓ', f(S_.l, 3)], ['o', f(S_.acc, 3)]];
          if (i === 2) items.push(['α', f(st[1].alpha, 3), C.c]);
          if (i === 4) items.push(['o / ℓ', f(S_.acc / S_.l, 4), C.a]);
          chips(items, py);
          if (i === 4) T(g, `softmax exato: ${f(oExact, 4)} ✓`, w / 2, py + 60, { size: 12.5, weight: 700, align: 'center', color: C.a });
        }
        if (i === 5 || i === 6) {
          chips(parts.map((p, b) => [`CTA ${b + 1} · LSE`, f(p.lse, 3), C.series[b]]), py);
          if (i === 6) { chips([['LSE', f(LSE, 3), C.d], ['merge', f(oMerge, 4), C.a]], py + 46); }
        }
        if (i === 7) {
          const xs = [0.5, 1, 2.5];
          chips(xs.map((x) => [`e^${f(x, 1)}`, f(Math.exp(x), 4), C.b]), py);
          chips(xs.map((x) => [`2^(${f(x, 1)}·1,4427)`, f(Math.pow(2, x * 1.4426950408889634), 4), C.a]), py + 46);
        }
      });
    }

    /* =============== Figura 5: split-K e ondas =============== */
    {
      const fig = $('#fig-splitk');
      const TABLE = { stock: { small: [16, 64, 4], lt32: [16, 32, 4], le256: [64, 8, 2], le512: [64, 4, 2], large: [64, 1, 2] }, gb10: { small: [32, 16, 4], lt32: [32, 8, 4], le256: [64, 4, 2], le512: [64, 2, 2], large: [64, 1, 2] } };
      const S = { sms: 48, prof: 'stock', reqs: 1, man: 64, occ: 1 };
      const refit = fitH(fig, (w) => (w < 600 ? 485 : 470) + (S.sms === 160 ? 40 : 0));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'GPU', options: [{ v: 48, label: 'GB10 · 48 SMs' }, { v: 160, label: 'GB300 · 160 SMs' }], value: 48, onChange: (v) => { S.sms = v; upd(); refit(); } });
      const sp = UI.seg(ctl, { label: 'Perfil', options: [{ v: 'stock', label: 'stock (GB300)' }, { v: 'gb10', label: 'gb10' }, { v: 'manual', label: 'manual' }], value: 'stock', onChange: (v) => { S.prof = v; upd(); } });
      UI.slider(ctl, { label: 'Requisições', min: 1, max: 8, step: 1, value: 1, onInput: (v) => { S.reqs = v; upd(); } });
      UI.slider(ctl, { label: 'Partições (manual)', min: 1, max: 64, step: 1, value: 64, onInput: (v) => { S.man = v; if (S.prof !== 'manual') sp.set('manual'); upd(); } });
      UI.slider(ctl, { label: 'CTAs/SM (suposto)', min: 1, max: 4, step: 1, value: 1, onInput: (v) => { S.occ = v; upd(); } });
      const rB = UI.readout(ctl, 'bucket', '—'), rC = UI.readout(ctl, 'CTAs', '—'), rW = UI.readout(ctl, 'ondas', '—');
      const cfg = () => {
        const rows = S.reqs * 4, base = rows;
        const bucket = base <= 8 ? 'small' : base < 32 ? 'lt32' : base <= 256 ? 'le256' : base <= 512 ? 'le512' : 'large';
        const p = S.prof === 'manual' ? null : TABLE[S.prof][bucket];
        const splits = p ? p[1] : S.man, slots = S.sms * S.occ;
        return { rows, base, bucket, p, splits, slots, ctas: base * splits, waves: (base * splits) / slots };
      };
      function upd() { const c = cfg(); rB.set(c.p ? `${c.bucket} (${c.p.join(', ')})` : `manual · ${c.splits}`); rC.set(String(c.ctas)); rW.set('≈ ' + comma(c.waves.toFixed(1))); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), c = cfg(), x0 = 16, W = w - 32;
        let y = 16;
        const n1 = wrap(g, `eixo das chaves: 2.048 + 3 posições selecionadas → ${c.splits} partição(ões) de ~${Math.ceil(2051 / c.splits)} chaves`, x0, y, W, 14, { size: 11.5, color: C.muted, maxLines: 2 });
        y += (n1 - 1) * 14 + 8;
        const segW = W / c.splits;
        for (let p = 0; p < c.splits; p++) { g.fillStyle = V.alpha(C.series[p % 6], 0.78); g.fillRect(x0 + p * segW, y, Math.max(1, segW - (segW > 3 ? 1 : 0)), 16); }
        y += 36;
        const n2 = wrap(g, `${c.rows} linha(s) de query (${S.reqs} req × 4, MTP=3) × 1 kv-head × ${c.splits} partições = ${c.ctas} CTAs`, x0, y, W, 15, { size: 12, color: C.ink, weight: 600, maxLines: 2 });
        y += (n2 - 1) * 15 + 10;
        const qd = Math.max(2, Math.min(14, (W - 4) / 32 - 3));
        for (let r = 0; r < c.rows; r++) V.box(g, x0 + r * (qd + 3), y, qd, qd, { r: qd / 2, fill: V.alpha(C.d, 0.85) });
        y += qd + 30;
        const nWv = Math.ceil(c.ctas / c.slots), period = 1.0, cyc = nWv * period + 0.8, tc = t % cyc;
        const wi = Math.min(nWv - 1, Math.floor(tc / period)), u = V.clamp((tc - wi * period) / (period * 0.8), 0, 1);
        T(g, `SMs · onda ${wi + 1} de ${nWv} (≈ ${comma(c.waves.toFixed(1))} ondas, supondo ${S.occ} CTA/SM)`, x0, y - 9, { size: 12, weight: 600, color: C.ink, maxW: W });
        const gr = smGrid(g, x0, y, W, S.sms === 48 ? 104 : 140, S.sms, (k, x, yy, cs) => {
          V.box(g, x, yy, cs, cs, { r: 3, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.25), lw: 1 });
          const sw = (cs - 4) / S.occ;
          for (let j = 0; j < S.occ; j++) {
            const idx = wi * c.slots + k * S.occ + j;
            if (idx < c.ctas) { const hh = (cs - 4) * (0.2 + 0.8 * u); g.fillStyle = V.alpha(C.series[(idx % c.splits) % 6], 0.85); V.rr(g, x + 2 + j * sw, yy + 2 + (cs - 4 - hh), Math.max(1, sw - 1), hh, 2); g.fill(); }
          }
        });
        y += gr.h + 26;
        const shown = Math.min(c.splits, 32), pw = Math.min(22, W / shown - 3);
        const rowW = shown * (pw + 3) - 3, px0 = x0 + (W - rowW) / 2;
        const mcx = w / 2, mcy = y + 58;
        for (let p = 0; p < shown; p++) {
          const px = px0 + p * (pw + 3);
          V.box(g, px, y, pw, 14, { r: 3, fill: V.alpha(C.series[p % 6], 0.8) });
          g.strokeStyle = V.alpha(C.series[p % 6], 0.35); g.lineWidth = 1; g.beginPath(); g.moveTo(px + pw / 2, y + 14); g.lineTo(mcx, mcy - 12); g.stroke();
        }
        if (c.splits > shown) T(g, `… ${c.splits} parciais`, px0 + rowW, y - 4, { size: 11, align: 'right', color: C.muted });
        const pulse = (t * 0.8) % 1;
        V.dot(g, V.lerp(px0 + pw / 2, mcx, pulse), V.lerp(y + 14, mcy - 12, pulse), 3, C.c);
        const mbw = Math.min(W, 240);
        V.box(g, mcx - mbw / 2, mcy - 12, mbw, 26, { r: 8, fill: V.alpha(C.c, 0.18), stroke: C.c, label: `merge por LSE · ${c.splits} parciais/linha`, size: 11.5 });
        const iy = mcy + 32, idle = Math.max(0, 1 - c.ctas / c.slots), extra = Math.max(0, nWv - 1);
        const bars = [
          ['SMs ociosos na 1ª onda', idle, `${Math.round(idle * 100)}%`, C.b],
          ['ondas além da 1ª', Math.min(1, extra / 6), String(extra), C.d],
          ['parciais para o merge', c.splits / 64, String(c.splits), C.c],
        ];
        const lblW = Math.min(170, W * 0.42);
        bars.forEach(([lb, fr, val, col], k) => {
          const yy = iy + k * 22;
          T(g, lb, x0, yy + 11, { size: 11.5, color: C.ink, maxW: lblW - 6 });
          V.box(g, x0 + lblW, yy, W - lblW - 44, 14, { r: 4, fill: V.alpha(C.muted, 0.1) });
          V.box(g, x0 + lblW, yy, Math.max(3, (W - lblW - 44) * fr), 14, { r: 4, fill: V.alpha(col, 0.75) });
          T(g, val, x0 + W, yy + 11, { size: 11.5, align: 'right', font: 'mono', color: C.ink });
        });
        wrap(g, `raciocínio do repositório (que supõe 1 CTA/SM), não medição · limite do bucket small suposto = 8 linhas (small_profile_limit não está no repo)`, x0, iy + 3 * 22 + 8, W, 14, { size: 11, color: C.muted, maxLines: 4 });
      });
    }

    /* =============== Figura 6: seletor de backend =============== */
    {
      const fig = $('#fig-backends');
      fitH(fig, (w) => (w < 640 ? 610 : 420));
      const L = [
        { id: 'glm-mla', name: 'GLM · MLA esparsa (alvo)', col: 'glm', st: [['hardware', 'SM121 · cc 12.1'], ['camada', 'MLA esparsa, indexer kpool, top-2048'], ['KV', 'fp8_ds_mla · 656 B/token'], ['geometria', 'd_qk 512 NoPE → pad 576 (GLM_NSA)'], ['graphs', 'só FULL uniforme · bloco [64]']], be: 'FLASHINFER_MLA_SPARSE_SM120', lib: 'FlashInfer', k: 7, rows: 'glm' },
        { id: 'glm-draft', name: 'GLM · drafter DFlash2', col: 'glm', st: [['hardware', 'SM121 · cc 12.1'], ['camada', 'atenção densa com SWA'], ['KV', 'BF16 (auto)'], ['geometria', 'bloco especulativo (k=7)'], ['máscara', 'NÃO causal dentro do bloco']], be: 'FLASH_ATTN (FA2)', lib: 'FlashAttention', note: 'TRITON_ATTN é causal dentro do bloco nesta imagem: aceitação 0,31 (~29 tok/s). Não fixe outro backend.' },
        { id: 'glm-kda', name: 'GLM · KDA (linear)', col: 'glm', st: [['hardware', 'SM121 · cc 12.1'], ['camada', 'atenção linear, regra delta'], ['estado', 'FP32 [cabeças, 128, 128] (MambaSpec)'], ['prefill', 'forma em chunks'], ['decode', 'forma recorrente']], be: 'chunk_kda · fused_recurrent_kda', lib: 'Triton (FLA)', note: 'FlashKDA (CUDA fundido, ABI de 14 argumentos, estado FP32) é opt-in no TP3.' },
        { id: 'ds-mla', name: 'DeepSeek · MLA esparsa + SWA', col: 'ds', st: [['hardware', 'SM121 · cc 12.1'], ['camada', 'SWA 128 + C4A/C128A, cache duplo'], ['KV', 'nvfp4_ds_mla = 584 B (bytes do FP8)'], ['geometria', 'TP2: 32/rank, já aceito · TP3: 24 → pad 32'], ['linhas', '≤ 64: kernel de decode dedicado']], be: 'DSV4_SPARSE_MLA_SM120 (lane B)', lib: 'FlashInfer (wrapper sparse-MLA)', k: 6, rows: 'ds', note: 'Lane B = overlay Stage-C (nvidia/sm120.py). Lane A (padrão, Anemll 0.1.1): DeepseekV4FlashInferSM120Attention, mesmo wrapper.' },
        { id: 'qwen-qsa', name: 'Qwen · QSA (esparsa)', col: 'qwen', st: [['hardware', 'SM121 · cc 12.1'], ['camada', 'GQA esparsa, indexer, top-2048'], ['KV', 'FP8 (patch) ou BF16'], ['geometria', '12 q-heads × 1 kv-head × 256'], ['kernel', 'do próprio modelo, split-K']], be: 'kernels QSA do modelo', lib: 'Triton', note: 'A FlashAttentionImpl é só a classe base; os perfis de lançamento estão na seção 4.' },
        { id: 'qwen-gdn', name: 'Qwen · Gated DeltaNet', col: 'qwen', st: [['hardware', 'SM121 · cc 12.1'], ['camada', 'atenção linear, regra delta'], ['estado', 'MambaSpec, 48 × 128 × 128'], ['prefill', 'FlashInfer/CuteDSL só SM90/10x → FLA'], ['decode', 'até 8 tokens MTP no kernel']], be: 'chunk_gated_delta_rule · fused_gdn_decode_post_conv_mtp', lib: 'Triton (FLA) + CUDA', note: 'Botões presentes e não usados: gdn_prefill_backend, VLLM_GDN_DECODE_KERNEL, VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE.' },
      ];
      const S = { id: 'glm-mla', reqs: 4, slice: false };
      const ctl = UI.controls(fig);
      const selc = UI.select(ctl, { label: 'Camada', options: L.map((l) => ({ v: l.id, label: l.name })), value: 'glm-mla', onChange: (v) => { S.id = v; } });
      selc.el.style.minWidth = '0'; selc.el.style.maxWidth = '100%';
      const selEl = selc.el.querySelector('select'); selEl.style.maxWidth = '100%'; selEl.style.minWidth = '0';
      UI.slider(ctl, { label: 'Requisições', min: 1, max: 14, step: 1, value: 4, onInput: (v) => { S.reqs = v; } });
      UI.toggle(ctl, { label: 'fatiar em ≤ 64 linhas', value: false, onChange: (v) => { S.slice = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), Lr = L.find((l) => l.id === S.id), col = C[Lr.col] || C.a;
        const nodes = [...Lr.st, ['backend', '']];
        const vertical = w < 640, pos = [];
        const hx0 = 60, hx1 = w - 60;
        if (!vertical) nodes.forEach((nd, i) => pos.push([hx0 + ((hx1 - hx0) * i) / (nodes.length - 1), 54]));
        else nodes.forEach((nd, i) => pos.push([28, 26 + i * 50]));
        g.lineWidth = 6; g.lineCap = 'round'; g.strokeStyle = V.alpha(col, 0.35);
        g.beginPath(); pos.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke(); g.lineCap = 'butt';
        const u = (t * 0.16) % 1, sg = u * (pos.length - 1), si = Math.min(pos.length - 2, Math.floor(sg));
        const tp = V.along(pos[si][0], pos[si][1], pos[si + 1][0], pos[si + 1][1], V.ease.inOut(sg - si));
        const slotW = (hx1 - hx0) / (nodes.length - 1) - 8;
        nodes.forEach(([k, val], i) => {
          const [x, y] = pos[i], last = i === nodes.length - 1, r = last ? 11 : 8;
          g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fillStyle = last ? col : C.bg || C.panel; g.fill(); g.lineWidth = 3; g.strokeStyle = col; g.stroke();
          if (!vertical) {
            T(g, k.toUpperCase(), x, y - 18, { size: 11, align: 'center', color: last ? col : C.muted, font: 'mono', weight: last ? 700 : 400 });
            if (val) wrap(g, val, x, y + 28, slotW, 14, { size: 12, align: 'center', color: C.ink, maxLines: 3 });
            else T(g, '↓', x, y + 30, { size: 14, align: 'center', color: col, weight: 700 });
          } else {
            T(g, k.toUpperCase(), x + 20, y - 3, { size: 11, color: last ? col : C.muted, font: 'mono', weight: last ? 700 : 400 });
            if (val) wrap(g, val, x + 20, y + 12, w - x - 36, 14, { size: 12, color: C.ink, maxLines: 2 });
          }
        });
        V.glow(g, tp[0], tp[1], 16, C.c); V.dot(g, tp[0], tp[1], 5, C.c);
        const cy = vertical ? pos[pos.length - 1][1] + 26 : 128, cW2 = w - 32;
        const beLines = splitLines(g, Lr.be, cW2 - 28, { size: 15, weight: 700, font: 'mono' }).length;
        const noteLines = Lr.note ? Math.min(4, splitLines(g, Lr.note, cW2 - 28, { size: 12 }).length) : 0;
        const cardH = 30 + beLines * 19 + 22 + noteLines * 15 + 10;
        V.box(g, 16, cy, cW2, cardH, { r: 12, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.6), lw: 1.5 });
        T(g, 'backend escolhido', 30, cy + 19, { size: 11, color: C.muted, font: 'mono' });
        wrap(g, Lr.be, 30, cy + 40, cW2 - 28, 19, { size: 15, weight: 700, color: col, font: 'mono' });
        T(g, 'biblioteca: ' + Lr.lib, 30, cy + 40 + beLines * 19 + 4, { size: 12, weight: 600, color: C.ink, maxW: cW2 - 28 });
        if (Lr.note) wrap(g, Lr.note, 30, cy + 40 + beLines * 19 + 24, cW2 - 28, 15, { size: 12, color: C.muted, maxLines: 4 });
        const ry = cy + cardH + 34;
        if (Lr.rows) {
          const per = Lr.k + 1, rows = S.reqs * per, maxR = 14 * 8;
          const tl = wrap(g, `linhas de decode por passo = ${S.reqs} req × (1 + k=${Lr.k}) = ${rows}`, 16, ry, w - 32, 15, { size: 12.5, weight: 600, color: C.ink, maxLines: 2 });
          const bx = 16, bw = w - 32, by = ry + 12 + tl * 15, bh = 30, X = (r) => bx + (bw * r) / maxR;
          V.box(g, bx, by, bw, bh, { r: 6, fill: V.alpha(C.muted, 0.1) });
          if (S.slice && rows > 64) {
            let r0 = 0, k = 0;
            while (r0 < rows) { const r1 = Math.min(rows, r0 + 64); V.box(g, X(r0) + 1, by + 2, X(r1) - X(r0) - 2, bh - 4, { r: 4, fill: V.alpha(C.series[k % 6], 0.75) }); T(g, `${r1 - r0}`, (X(r0) + X(r1)) / 2, by + bh / 2 + 4, { size: 11.5, align: 'center', font: 'mono', color: C.ink }); r0 = r1; k++; }
          } else {
            V.box(g, bx, by, Math.max(4, X(rows) - bx), bh, { r: 6, fill: V.alpha(rows > 64 ? C.c : C.a, 0.72) });
            const fl = (t * 0.5) % 1; V.dot(g, bx + (X(rows) - bx) * fl, by + bh / 2, 3, C.ink);
          }
          V.line(g, X(64), by - 6, X(64), by + bh + 6, { color: C.d, lw: 2, dash: [4, 3] });
          T(g, '64', X(64), by + bh + 18, { size: 11, align: 'center', font: 'mono', color: C.d, weight: 700 });
          let msg;
          if (Lr.rows === 'ds') msg = rows <= 64 ? '1 chamada → kernel de decode dedicado do FlashInfer SM120 (_DECODE_MAX_TOKENS = 64).' : S.slice ? `#141 (corrige a chamada da lane A): ${Math.ceil(rows / 64)} chamadas sequenciais de ≤ 64 linhas (contorno, não causa raiz; re-medir o prefill).` : 'Acima de 64 → orquestrador de prefill do FlashInfer, onde a morte estocástica do engine em TP=2 foi amostrada.';
          else msg = S.slice ? `Variante TP4: patch_sparse_mla_slice fatia a chamada final em ${Math.ceil(rows / 64)} pedaços de ≤ 64 linhas (VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64), onde o progresso parava.` : 'TP2: a receita não fatia. O fatiamento em ≤ 64 linhas é da variante TP4 (mitiga, não corrige, a race de 4 ranks).';
          wrap(g, msg, 16, by + bh + 40, w - 32, 15, { size: 12, color: rows > 64 && !S.slice && Lr.rows === 'ds' ? C.c : C.ink, maxLines: 3 });
        } else {
          wrap(g, 'Este backend não tem o limite de 64 linhas do sparse-MLA do SM120: as linhas do passo (requisições × (1 + k)) entram numa chamada só.', 16, ry, w - 32, 15, { size: 12, color: C.muted, maxLines: 3 });
        }
      });
    }

    /* =============== Figura 7: dequant + cruzamento por M =============== */
    {
      const fig = $('#fig-dequant');
      fitH(fig, (w) => (w < 600 ? 580 : 480));
      const S = { lm: Math.log2(7168), rule: 'kda', w8a8: true };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'M (linhas)', min: 0, max: 13, step: 0.25, value: Math.log2(7168), fmt: (v) => String(Math.round(2 ** v)), onInput: (v) => { S.lm = v; upd(); } });
      UI.seg(ctl, { label: 'Fronteira', options: [{ v: 'kda', label: 'GLM KDA · 512' }, { v: 'exl3', label: 'GLM EXL3 denso · 144' }, { v: 'ds', label: 'DS híbrido · 128' }], value: 'kda', onChange: (v) => { S.rule = v; upd(); } });
      UI.toggle(ctl, { label: 'mostrar W8A8', value: true, onChange: (v) => { S.w8a8 = v; } });
      const rM = UI.readout(ctl, 'Marlin W8A16 (modelo)', '—'), rBf = UI.readout(ctl, 'BF16 cuBLAS (modelo)', '—'), rP = UI.readout(ctl, 'a regra escolhe', '—');
      const K = 4096, N = 12576, BW = 230e9, M0 = 7168, P = 4;
      const meas = { marlin: 27.21, w8a8: 12.57, bf16: 7.36 };
      const bytes = { marlin: (M) => K * N + 2 * M * K + 2 * M * N, w8a8: (M) => K * N + 3 * M * K + 2 * M * N, bf16: (M) => 2 * K * N + 2 * M * K + 2 * M * N };
      const fl = (M) => 2 * M * K * N;
      const eff = {};
      Object.keys(meas).forEach((k) => { const tm = (bytes[k](M0) / BW) * 1e3; const tc = Math.pow(Math.pow(meas[k], P) - Math.pow(tm, P), 1 / P); eff[k] = fl(M0) / (tc * 1e-3); });
      const tms = (k, M) => { const tm = (bytes[k](M) / BW) * 1e3, tc = (fl(M) / eff[k]) * 1e3; return Math.pow(tm ** P + tc ** P, 1 / P); };
      const RULE = { kda: { th: 512, lo: 'Marlin W8A16', hi: 'BF16 (cópia retida)', ge: false, lab: 'KDA 512' }, exl3: { th: 144, lo: 'EXL3 direto no trellis', hi: 'BF16 reconstruído', ge: false, lab: 'EXL3 144' }, ds: { th: 128, lo: 'Marlin W4A16', hi: 'CUTLASS W4A4', ge: true, lab: 'DS 128' } };
      function upd() {
        const M = Math.round(2 ** S.lm);
        rM.set(comma(tms('marlin', M).toFixed(3)) + ' ms'); rBf.set(comma(tms('bf16', M).toFixed(3)) + ' ms');
        const R = RULE[S.rule]; rP.set((R.ge ? M >= R.th : M > R.th) ? R.hi : R.lo);
      }
      upd();
      const codes = [0x3, 0xa, 0x7, 0x1, 0xe, 0x5, 0x0, 0xb];
      const vals = codes.map((c) => LA.num.decode('e2m1', c));
      const scale = 0.25;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 600;
        const tx0 = 16, tW = w - 32;
        T(g, 'um registrador de 32 bits = 8 pesos FP4 (E2M1) de um bloco MXFP4', tx0, 16, { size: 12, weight: 600, color: C.ink, maxW: tW });
        const nw = Math.min(62, tW / 8 - 4), wx = tx0, wy = 26;
        const k = Math.floor(t * 1.2) % 8, ph = (t * 1.2) % 1;
        codes.forEach((c, i) => {
          const x = wx + i * (nw + 4), on = i === k;
          V.box(g, x, wy, nw, 26, { r: 5, fill: V.alpha(on ? C.c : C.b, on ? 0.35 : 0.14), stroke: on ? C.c : V.alpha(C.b, 0.45), lw: on ? 2 : 1 });
          T(g, LA.num.toBin(c, 4), x + nw / 2, wy + 17, { size: 11.5, font: 'mono', align: 'center', color: C.ink });
          T(g, comma(String(vals[i])), x + nw / 2, wy + 41, { size: 11, font: 'mono', align: 'center', color: on ? C.c : C.muted });
        });
        const cx0 = narrow ? tx0 : wx + 8 * (nw + 4) + 10, cy = narrow ? wy + 60 : wy;
        const chW = narrow ? (tW - 30) / 4 : Math.max(70, (w - 16 - cx0 - 30) / 4);
        const chips = [[`código ${LA.num.toBin(codes[k], 4)}`, C.b], [`E2M1 = ${comma(String(vals[k]))}`, C.b], [`× 2⁻² = ${comma(String(vals[k] * scale))}`, C.d], ['BF16 → mma.sync', C.a]];
        if (cx0 + 4 * (chW + 10) - 10 <= w - 8) {
          chips.forEach(([lb, col], i) => {
            const x = cx0 + i * (chW + 10);
            V.box(g, x, cy, chW, 26, { r: 6, fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.6) });
            T(g, lb, x + chW / 2, cy + 17, { size: 11, align: 'center', font: 'mono', color: C.ink, maxW: chW - 6 });
            if (i < 3) V.arrow(g, x + chW + 1, cy + 13, x + chW + 9, cy + 13, { color: C.muted, lw: 1.2, head: 5 });
          });
          const pi = Math.min(3, Math.floor(ph * 4));
          V.glow(g, cx0 + pi * (chW + 10) + chW / 2, cy + 13, 14, C.c, 0.8);
        }
        T(g, 'escala do bloco: E8M0 = 125 → 2^(125−127) = 2⁻² · a ativação continua BF16 (W4A16)', tx0, (narrow ? cy + 44 : wy + 62), { size: 11, color: C.muted, maxW: tW });
        const top = narrow ? cy + 70 : wy + 82;
        const cx0c = narrow ? 44 : 56, cx1 = w - 16, cy0 = top + 56, cy1 = h - (narrow ? 66 : 52);
        const X = (lm) => cx0c + (lm / 13) * (cx1 - cx0c);
        const ylo = Math.log10(0.1), yhi = Math.log10(60);
        const Y = (ms) => cy1 - ((Math.log10(V.clamp(ms, 0.1, 60)) - ylo) / (yhi - ylo)) * (cy1 - cy0);
        g.fillStyle = V.alpha(C.b, 0.08); g.fillRect(X(0), cy0, X(Math.log2(220)) - X(0), cy1 - cy0);
        g.fillStyle = V.alpha(C.a, 0.07); g.fillRect(X(Math.log2(768)), cy0, X(13) - X(Math.log2(768)), cy1 - cy0);
        T(g, 'decode (GLM ≤ 220)', X(0) + 4, cy1 - 8, { size: 11, color: C.b });
        if (!narrow) {
          T(g, 'medido: BF16 3,1–3,7× mais', X(13) - 4, cy1 - 22, { size: 11, color: C.a, align: 'right', weight: 600 });
          T(g, 'rápido (prefill, M ≥ 768)', X(13) - 4, cy1 - 8, { size: 11, color: C.a, align: 'right' });
        } else T(g, 'prefill', X(13) - 4, cy1 - 8, { size: 11, color: C.a, align: 'right' });
        [[0, '1'], [2, '4'], [4, '16'], [6, '64'], [8, '256'], [10, '1K'], [12, '4K'], [13, '8K']].forEach(([lm, lb]) => { V.line(g, X(lm), cy1, X(lm), cy1 + 4, { color: C.muted, lw: 1 }); T(g, lb, X(lm), cy1 + 16, { size: 11, align: 'center', color: C.muted, font: 'mono' }); });
        [0.1, 1, 10].forEach((ms) => { V.line(g, cx0c, Y(ms), cx1, Y(ms), { color: C.grid, lw: 1 }); T(g, comma(String(ms)), cx0c - 4, Y(ms) + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }); });
        T(g, 'ms', cx0c - 4, Y(10) - 14, { size: 11, align: 'right', color: C.muted, font: 'mono' });
        
        const L768 = Math.log2(768);
        const seg = (k, col, dash, a0, a1, alpha) => { g.beginPath(); const n = Math.ceil((a1 - a0) * 10); for (let i = 0; i <= n; i++) { const lm = a0 + ((a1 - a0) * i) / n, x = X(lm), y = Y(tms(k, 2 ** lm)); if (i) g.lineTo(x, y); else g.moveTo(x, y); } g.strokeStyle = V.alpha(col, alpha); g.lineWidth = 2.5; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); };
        const curve = (k, col, dash) => { seg(k, col, [3, 5], 0, L768, 0.5); seg(k, col, dash, L768, 13, 1); };
        wrap(g, 'tracejado: extrapolação do modelo, não medida; cruzamento ilustrativo', X(0) + 4, cy0 + 32, Math.max(120, X(L768) - X(0) - 12), 14, { size: 11, color: C.muted, maxLines: 3 });
        curve('bf16', C.a); curve('marlin', C.c); if (S.w8a8) curve('w8a8', C.d, [5, 4]);
        [['marlin', C.c], ['w8a8', C.d], ['bf16', C.a]].forEach(([k2, col]) => {
          if (k2 === 'w8a8' && !S.w8a8) return;
          const x = X(Math.log2(M0)), y = Y(meas[k2]);
          g.beginPath(); g.arc(x, y, 5.5, 0, 2 * Math.PI); g.fillStyle = col; g.fill(); g.lineWidth = 2; g.strokeStyle = C.bg || '#000'; g.stroke();
          const lbl = comma(meas[k2].toFixed(2)) + ' ms'; g.font = fnt({ size: 11, font: 'mono', weight: 600 }); const lw2 = g.measureText(lbl).width;
          V.box(g, x - 12 - lw2, y - 18, lw2 + 6, 15, { r: 3, fill: V.alpha(C.bg || '#000', 0.85) });
          T(g, lbl, x - 9, y - 7, { size: 11, align: 'right', font: 'mono', color: col, weight: 600 });
        });
        const lg = [['Marlin W8A16', C.c], ['BF16 cuBLAS', C.a]]; if (S.w8a8) lg.push(['W8A8', C.d]);
        let lx = cx0c + 6;
        lg.forEach(([lb, col]) => { V.line(g, lx, cy0 + 10, lx + 18, cy0 + 10, { color: col, lw: 3 }); T(g, lb, lx + 22, cy0 + 14, { size: 11, color: C.ink }); g.font = fnt({ size: 11 }); lx += 32 + g.measureText(lb).width; });
        Object.entries(RULE).forEach(([key, R], i) => {
          const x = X(Math.log2(R.th)), on = key === S.rule, fy = top + 6 + i * 16;
          V.line(g, x, fy + 12, x, cy1, { color: on ? C.f : V.alpha(C.muted, 0.5), lw: on ? 2.5 : 1, dash: on ? [] : [3, 4] });
          g.font = fnt({ size: 11, weight: on ? 700 : 400 }); const tw = g.measureText(R.lab).width + 12;
          const fx = key === 'ds' ? Math.max(cx0c, x - tw) : Math.min(x, w - 16 - tw);
          V.box(g, fx, fy, tw, 14, { r: 3, fill: V.alpha(on ? C.f : C.muted, on ? 0.3 : 0.12) });
          T(g, R.lab, fx + 6, fy + 11, { size: 11, weight: on ? 700 : 400, color: C.ink });
        });
        const xm = X(S.lm);
        V.line(g, xm, cy0, xm, cy1, { color: V.alpha(C.ink, 0.35), lw: 1 });
        const Mn = 2 ** S.lm;
        [['marlin', C.c], ['bf16', C.a]].concat(S.w8a8 ? [['w8a8', C.d]] : []).forEach(([k2, col]) => V.dot(g, xm, Y(tms(k2, Mn)), 4, col));
        wrap(g, narrow ? 'faixa verde: medido, BF16 3,1–3,7× mais rápido para M ≥ 768 · curvas: in_proj do KDA (TP2), roofline calibrado em M = 7.168' : 'eixo x: M em escala log · curvas: in_proj do KDA (N 12.576 × K 4.096, TP2), roofline calibrado nos pontos medidos de M = 7.168', 16, cy1 + 34, w - 32, 14, { size: 11, color: narrow ? C.ink : C.muted, maxLines: narrow ? 3 : 2 });
      });
    }

    /* =============== Figura 8: MoE thin/fat =============== */
    {
      const fig = $('#fig-moe');
      fitH(fig, (w) => (w < 600 ? 440 : 450));
      const E = 288, TK = 8, CAPS = [32, 64, 128, 256, 512, 1024];
      const rng = V.rng(28);
      const wts = Array.from({ length: E }, (_, i) => Math.pow(i + 1, -1.2));
      const perm = Array.from({ length: E }, (_, i) => i);
      for (let i = E - 1; i > 0; i--) { const j = rng.int(i + 1); const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; }
      const cum = []; let acc = 0; wts.forEach((x, i) => { acc += x; cum[i] = acc; });
      const pick = () => { const r = rng() * acc; let lo = 0, hi = E - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m; } return perm[lo]; };
      function route(T_) { const cnt = new Array(E).fill(0), slots = []; for (let t = 0; t < T_; t++) { const ch = []; while (ch.length < TK) { const e = pick(); if (!ch.includes(e)) ch.push(e); } ch.forEach((e) => { cnt[e]++; slots.push(e); }); } return { cnt, slots, sorted: cnt.slice().sort((a, b) => b - a) }; }
      const PH = { dec: Object.assign({ T: 32 }, route(32)), p1k: Object.assign({ T: 1024 }, route(1024)), p7k: Object.assign({ T: 7168 }, route(7168)) };
      const S = { ph: 'p1k', st: 'e3', ci: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Fase', options: [{ v: 'dec', label: 'decode · 4 × 8' }, { v: 'p1k', label: 'prefill · 1.024' }, { v: 'p7k', label: 'prefill · 7.168' }], value: 'p1k', onChange: (v) => { S.ph = v; upd(); } });
      UI.seg(ctl, { label: 'Estratégia', options: [{ v: 'legacy', label: 'legado (loop Python)' }, { v: 'e3', label: 'E3 agrupado' }, { v: 'coop', label: 'cooperativo' }], value: 'e3', onChange: (v) => { S.st = v; upd(); } });
      UI.slider(ctl, { label: 'cap · TEMP_ROWS', min: 0, max: 5, step: 1, value: 0, fmt: (i) => String(CAPS[i]), onInput: (v) => { S.ci = v; upd(); } });
      const rA = UI.readout(ctl, 'experts ativos', '—'), rF = UI.readout(ctl, 'experts fat', '—'), rX = UI.readout(ctl, 'máx. linhas', '—'), rS = UI.readout(ctl, 'syncs com o host', '—');
      const stats = () => {
        const P = PH[S.ph], cap = CAPS[S.ci];
        const fat = P.cnt.filter((c) => c > cap).sort((a, b) => b - a);
        const coopOK = S.st === 'coop' && P.T <= 32;
        const syncs = S.st === 'legacy' && fat.length ? 1 : 0;
        return { P, cap, fat, active: P.cnt.filter((c) => c > 0).length, maxR: P.sorted[0], segs: fat.reduce((s, c) => s + Math.ceil(c / 64), 0), syncs, coopOK };
      };
      function upd() { const q = stats(); rA.set(String(q.active)); rF.set(String(q.fat.length)); rX.set(String(q.maxR)); rS.set(String(q.syncs)); }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), q = stats(), P = q.P, cap = q.cap;
        const hx = 16, hw = w - 32, hy = 38, hh = Math.min(170, h * 0.36);
        const yMax = Math.max(cap * 1.25, P.sorted[0] * 1.08, 4), bwid = hw / E;
        const Hs = (c) => (hh * Math.sqrt(c)) / Math.sqrt(yMax);
        wrap(g, `linhas por expert (288 experts, ordenados; escala √) · ${P.T.toLocaleString('pt-BR')} tokens × top-8 = ${(P.T * 8).toLocaleString('pt-BR')} pares`, hx, hy - 16, hw, 14, { size: 12, weight: 600, color: C.ink, maxLines: 1 });
        for (let i = 0; i < E; i++) { const c = P.sorted[i]; if (!c) continue; const bh = Hs(c); g.fillStyle = V.alpha(c > cap ? C.c : C.a, 0.85); g.fillRect(hx + i * bwid, hy + hh - bh, Math.max(1, bwid - (bwid > 2 ? 0.5 : 0)), bh); }
        V.line(g, hx, hy + hh, hx + hw, hy + hh, { color: V.alpha(C.muted, 0.5), lw: 1 });
        const cyL = hy + hh - Hs(cap);
        V.line(g, hx, cyL, hx + hw, cyL, { color: C.d, lw: 1.8, dash: [5, 4] });
        T(g, `cap = ${cap} linhas`, hx + hw - 4, cyL - 5, { size: 11, align: 'right', font: 'mono', color: C.d, weight: 600 });
        const nAct = q.active, hotN = Math.max(1, Math.min(nAct, 48));
        for (let p = 0; p < 26; p++) {
          const u = (t * 0.55 + p / 26) % 1, idx = Math.floor(Math.pow(((p * 37) % 26) / 26, 1.6) * hotN);
          const c = P.sorted[idx] || 0; if (!c) continue;
          const x = hx + (idx + 0.5) * bwid, ytop = hy + hh - Hs(c);
          g.globalAlpha = 1 - u * 0.7; V.dot(g, x, V.lerp(hy - 6, ytop - 2, u), 2.2, c > cap ? C.c : C.a); g.globalAlpha = 1;
        }
        const mxl = `máx. ${P.sorted[0]} linhas`; g.font = fnt({ size: 11, font: 'mono' });
        const mxx = hx + Math.max(8, 14 * bwid) + 10, mxw = g.measureText(mxl).width + 10;
        V.box(g, mxx - 5, hy - 1, mxw, 16, { r: 4, fill: V.alpha(C.bg || C.panel, 0.85) });
        T(g, mxl, mxx, hy + 11, { size: 11, font: 'mono', color: C.c });
        const ly = hy + hh + 40;
        T(g, 'lançamentos por camada MoE', hx, ly - 12, { size: 12, weight: 600, color: C.ink });
        const boxes = [];
        const coop = S.st === 'coop' && q.coopOK;
        if (coop) { boxes.push({ l: 'rot', s: 'x·suh + Hadamard', c: C.b }); boxes.push({ l: 'A', s: 'gate/up', c: C.b }); boxes.push({ l: 'B', s: 'down + Σ fixa', c: C.b }); }
        else {
          boxes.push({ l: 'exl3_moe', s: `thin ≤ ${cap}`, c: C.a });
          if (S.st === 'legacy') { q.fat.slice(0, 9).forEach((c) => boxes.push({ l: 'recon+GEMM', s: `${c} linhas`, c: C.c, sync: true })); if (q.fat.length > 9) boxes.push({ l: `+${q.fat.length - 9}`, s: 'fat', c: C.c }); }
          else if (q.fat.length) { boxes.push({ l: 'gather', s: 'Hadamard', c: C.d }); boxes.push({ l: 'gate/up', s: '+ SwiGLU', c: C.d }); boxes.push({ l: 'down', s: '+ scatter', c: C.d }); }
        }
        const nb = boxes.length, bgap = 6, bw2 = Math.min(118, (hw - bgap * (nb - 1)) / nb);
        boxes.forEach((b, i) => {
          const x = hx + i * (bw2 + bgap);
          V.box(g, x, ly, bw2, 40, { r: 8, fill: V.alpha(b.c, 0.18), stroke: b.c, lw: 1.3 });
          T(g, b.l, x + bw2 / 2, ly + 17, { size: 11.5, weight: 600, align: 'center', color: C.ink, maxW: bw2 - 6, font: 'mono' });
          T(g, b.s, x + bw2 / 2, ly + 32, { size: 11, align: 'center', color: C.muted, maxW: bw2 - 6 });
          if (b.sync) { V.box(g, x + bw2 - 16, ly - 8, 16, 16, { r: 8, fill: C.red || C.c }); T(g, '⏸', x + bw2 - 8, ly + 4, { size: 11, align: 'center', color: '#fff' }); }
        });
        const iy = ly + 64;
        let msg = '';
        if (coop) {
          const runs = q.active, slotsN = P.T * TK;
          const order = P.slots.slice().sort((a, b) => a - b);
          const gapR = hw < 500 ? 0.6 : 2, dw = Math.max(0.7, Math.min(5, (hw - runs * gapR) / slotsN));
          let x = hx, prev = -1, run = 0;
          order.forEach((e) => { if (e !== prev) { x += gapR; run++; prev = e; } if (x + dw > hx + hw) return; g.fillStyle = V.alpha(C.series[run % 6], 0.85); g.fillRect(x, iy, Math.max(0.6, dw - 0.3), 12); x += dw; });
          msg = `${slotsN} slots (32 linhas × 8) formam ${runs} runs: cada expert é lido uma vez por run, e não uma vez por slot. Contadores de conclusão elegem o último bloco para somar em ordem fixa.`;
        } else if (S.st === 'coop') {
          msg = 'O cooperativo só atende passos de 1–32 linhas; no prefill o wrapper devolve o trabalho ao caminho stock (exl3_moe + E3).';
        } else if (S.st === 'legacy') {
          msg = q.fat.length ? `${q.fat.length} experts fat → reconstrução BF16 + GEMM para cada um, depois de um .nonzero().tolist() que sincroniza com a CPU e quebra a captura de CUDA graph.` : 'Nenhum expert fat: o passo inteiro cabe no lançamento único do exl3_moe (graph-safe).';
        } else if (q.fat.length) {
          const shown = Math.min(q.segs, 160), sw2 = Math.max(1.5, Math.min(10, (hw - 3 * q.fat.length) / shown - 1));
          let x = hx, ei = 0, left = Math.ceil(q.fat[0] / 64);
          for (let s2 = 0; s2 < shown && x + sw2 <= hx + hw; s2++) { g.fillStyle = V.alpha(C.series[ei % 6], 0.85); g.fillRect(x, iy, sw2, 12); x += sw2 + 1; left--; if (left <= 0 && ei < q.fat.length - 1) { ei++; left = Math.ceil(q.fat[ei] / 64); x += 2; } }
          const mnbt = P.T;
          msg = `${q.segs} segmentos de 64 linhas descritos por prefix-sums no device; os kernels fazem grid-stride sobre eles. 0 syncs. Scratch h13 + h2 = MNBT × 8 × (H + I) × 2 B = ${fmtB(mnbt * 8 * (4096 + 1024) * 2)} com MNBT ${mnbt.toLocaleString('pt-BR')}.`;
        } else {
          msg = 'Nenhum expert passa do cap: o passo inteiro cabe no lançamento único do exl3_moe, graph-safe.';
        }
        wrap(g, msg, hx, iy + 32, hw, 15, { size: 12, color: C.ink, maxLines: 5 });
      });
    }

    /* =============== Figura 9: PDL, streams, JIT, autotune =============== */
    {
      const fig = $('#fig-timeline');
      const S = { pdl: false, aux: false, jit: false, tune: 'ok' };
      const NAMES = ['wqa_wkv', 'compressor', 'idx w_proj', 'idx compr.', 'atenção', 'estado S', 'lê S'];
      const mctx = document.createElement('canvas').getContext('2d');
      const LEG = ['GEMM', 'aux', 'atenção', 'estado', 'lê estado', 'JIT'];
      const keyText = () => 'kernels (ilustrativo): ' + NAMES.map((nm, i) => `${i + 1} ${nm}`).join(' · ') + (S.jit ? ' · J JIT' : '');
      function msgOf() {
        if (S.tune === 'div') return '✕ Trava: com caches de autotune diferentes, os ranks rodam táticas diferentes e o coletivo não fecha (visto uma vez no Qwen v0.30). Correções: cache efêmero em /tmp/fi_autotune (Qwen), para os nós partirem do mesmo estado; rank 0 tuna e transmite (overlay Stage-C, lane B, do DeepSeek).';
        if (S.jit) return 'O rank 1 compila um kernel no meio do serviço; o rank 0 fica parado no all-reduce. Se passar de 600 s, o watchdog do NCCL derruba o par: por isso as receitas persistem caches e aquecem as formas no boot.';
        if (S.pdl) return 'Ilustração (interpretação): o kernel 7, que lê o estado do KDA, começa antes de a escrita do 6 terminar (raio); o passo encurta, mas o resultado fica errado. O repositório só registra a race; no SM12x a receita GLM desliga PDL: major in (9, 10).';
        return 'OK: sequências idênticas nos dois ranks.' + (S.aux ? ' As GEMMs de entrada independentes (2, 3, 4) rodam em paralelo nos streams auxiliares e o passo encurta.' : ' Ligue os mecanismos para ver o que muda.');
      }
      const geo = (w) => {
        const narrow = w < 600, laneH = 38, auxH = S.aux ? 36 : 0, y0 = 32, per = laneH + 22 + (S.aux ? auxH + 6 : 0);
        const lanesEnd = y0 + 2 * per - 16;
        const kL = Math.min(3, splitLines(mctx, keyText(), w - 24, { size: 11, font: 'mono' }).length);
        const mL = Math.min(8, splitLines(mctx, msgOf(), w - 24, { size: 12.5, weight: 600 }).length);
        mctx.font = fnt({ size: 11 }); let lx = 12, rows = 1; LEG.forEach((lb) => { const lw_ = 28 + mctx.measureText(lb).width; if (lx + lw_ > w - 12) { lx = 12; rows++; } lx += lw_; });
        return { narrow, laneH, auxH, y0, per, lanesEnd, h: lanesEnd + 14 + kL * 14 + 10 + mL * 16 + 8 + rows * 18 + 6 };
      };
      const refit = fitH(fig, (w) => geo(w).h);
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'PDL (sobrepor prólogo)', value: false, onChange: (v) => { S.pdl = v; upd(); } });
      UI.toggle(ctl, { label: '3 streams auxiliares', value: false, onChange: (v) => { S.aux = v; upd(); refit(); } });
      UI.toggle(ctl, { label: 'JIT a frio no rank 1', value: false, onChange: (v) => { S.jit = v; upd(); } });
      UI.seg(ctl, { label: 'Autotune', options: [{ v: 'ok', label: 'coerente entre ranks' }, { v: 'div', label: 'caches divergentes por nó' }], value: 'ok', onChange: (v) => { S.tune = v; upd(); } });
      const rT = UI.readout(ctl, 'passo (ilustrativo)', '—'), rE = UI.readout(ctl, 'estado', '—');
      function plan() {
        const gap = S.pdl ? -0.35 : 0.45, R = [];
        for (let r = 0; r < 2; r++) {
          const ks = []; let t0 = 0.3;
          const push = (id, d, kind) => { const it = { id, name: id === 'jit' ? 'JIT: compilando (≈ minutos)' : NAMES[id], s: t0, e: t0 + d, lane: 0, kind }; ks.push(it); t0 = it.e + gap; return it; };
          const g1 = push(0, 2.6, 'g');
          if (S.aux) {
            const a = [[1, 1.2], [2, 0.9], [3, 1.1]].map(([id, d], i) => ({ id, name: NAMES[id], s: g1.s + 0.1 * i, e: g1.s + 0.1 * i + d, lane: 1, sub: i, kind: 'aux' }));
            ks.push(...a); t0 = Math.max(g1.e, ...a.map((x) => x.e)) + Math.max(gap, 0.25);
          } else { push(1, 1.2, 'g'); push(2, 0.9, 'g'); push(3, 1.1, 'g'); }
          if (r === 1 && S.jit) push('jit', 7.5, 'jit');
          const div = r === 1 && S.tune === 'div';
          const at = push(4, div ? 4.4 : 3.4, 'att'); if (div) at.name = 'atenção · tática B';
          push(5, 1.8, 'state');
          const s2 = push(6, 1.5, 'read');
          R.push({ ks, ready: s2.e, s2 });
        }
        const arS = Math.max(R[0].ready, R[1].ready) + 0.2;
        return { R, arS, arE: arS + 1.3, hang: S.tune === 'div', race: S.pdl };
      }
      function upd() {
        const p = plan();
        if (typeof refit === 'function') refit();
        rT.set(p.hang ? '∞ (trava)' : comma(p.arE.toFixed(1)) + ' u');
        rE.set(p.hang ? 'travado no all-reduce' : S.jit ? 'rank 0 esperando' : p.race ? 'rápido, mas errado' : 'ok');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), p = plan(), G = geo(w);
        const lw = G.narrow ? 62 : 104, x0 = 12 + lw, x1 = w - 12, maxT = 26;
        const X = (tt) => x0 + (Math.min(tt, maxT) / maxT) * (x1 - x0);
        const laneH = G.laneH, auxH = G.auxH;
        const lanes = [];
        let y = G.y0;
        [0, 1].forEach((r) => { lanes.push({ r, main: true, y }); y += laneH + 6; if (S.aux) { lanes.push({ r, main: false, y }); y += auxH + 6; } y += 16; });
        lanes.forEach((ln) => {
          V.box(g, 12, ln.y, lw - 8, ln.main ? laneH : auxH, { r: 6, fill: V.alpha(C.muted, 0.08) });
          T(g, ln.main ? `rank ${ln.r}` : 'aux ×3', 18, ln.y + 15, { size: 11.5, weight: 600, color: C.ink, font: 'mono' });
          if (ln.main && lw > 80) T(g, 'stream 0', 18, ln.y + 28, { size: 11, color: C.muted });
        });
        const colOf = (k) => (k === 'jit' ? C.red || C.c : k === 'att' ? C.d : k === 'state' ? C.e : k === 'read' ? C.f : k === 'aux' ? C.b : C.a);
        p.R.forEach((R, r) => {
          const mainY = lanes.find((l) => l.r === r && l.main).y, auxL = lanes.find((l) => l.r === r && !l.main);
          R.ks.forEach((k) => {
            const yy = k.lane === 1 && auxL ? auxL.y + k.sub * 12 : mainY, hh = k.lane === 1 ? 11 : laneH;
            const xa = X(k.s), xb = X(k.e), bw = Math.max(2, xb - xa - 1);
            V.box(g, xa, yy, bw, hh, { r: 4, fill: V.alpha(colOf(k.kind), k.kind === 'jit' ? 0.3 : 0.75), stroke: k.kind === 'jit' ? C.red || C.c : null, lw: 1.5 });
            const num = k.id === 'jit' ? 'J' : String(k.id + 1);
            if (k.lane === 0) {
              T(g, num, xa + bw / 2, yy - 4, { size: 11, align: 'center', color: C.muted, font: 'mono', weight: 600 });
              g.font = fnt({ size: 11 });
              if (g.measureText(k.name).width <= bw - 6) T(g, k.name, xa + bw / 2, yy + laneH / 2 + 4, { size: 11, align: 'center', color: k.kind === 'jit' ? C.ink : onTxt(C) });
            } else if (xb + 16 < x1) T(g, num, xb + 3, yy + 10, { size: 11, color: C.muted, font: 'mono' });
            if (S.pdl && k.lane === 0 && k !== R.ks[0]) { g.fillStyle = V.alpha(C.ink, 0.18); for (let hx = xa; hx < Math.min(xb, xa + X(0.35) - X(0)); hx += 4) g.fillRect(hx, yy, 2, laneH); }
          });
          if (p.race) {
            const xr = X(R.s2.s), yr = mainY + laneH / 2;
            g.strokeStyle = C.red || C.c; g.lineWidth = 2.2; g.beginPath(); g.moveTo(xr + 4, yr - 10); g.lineTo(xr - 3, yr); g.lineTo(xr + 3, yr); g.lineTo(xr - 4, yr + 10); g.stroke();
          }
          if (R.ready < p.arS - 0.25) { const xa = X(R.ready + 0.05), xb = X(p.arS); g.fillStyle = V.alpha(C.muted, 0.25); for (let hx = xa; hx < xb; hx += 6) g.fillRect(hx, mainY + laneH / 2 - 1, 3, 2); if (xb - xa > 60) T(g, 'espera', (xa + xb) / 2, mainY + laneH / 2 - 6, { size: 11, align: 'center', color: C.muted }); }
        });
        const yA = lanes[0].y, yB = lanes.find((l) => l.r === 1 && l.main).y + laneH;
        const xa = X(p.arS), xb = X(p.arE);
        if (p.arS < maxT) {
          V.box(g, xa, yA, Math.max(3, xb - xa), yB - yA, { r: 6, fill: V.alpha(C.c, p.hang ? 0.08 : 0.22), stroke: C.c, lw: 1.5 });
          if (p.hang) { g.setLineDash([4, 4]); g.strokeStyle = C.red || C.c; V.rr(g, xa, yA, Math.max(3, xb - xa), yB - yA, 6); g.stroke(); g.setLineDash([]); T(g, '✕', (xa + xb) / 2, (yA + yB) / 2 + 6, { size: 18, align: 'center', color: C.red || C.c, weight: 700 }); }
          T(g, 'all-reduce', Math.min((xa + xb) / 2, x1 - 34), yA - 18, { size: 11, align: 'center', color: C.c, font: 'mono' });
        } else T(g, 'all-reduce → (fora da escala)', x1, yA - 18, { size: 11, align: 'right', color: C.c, font: 'mono' });
        const phx = X((t * 2.2) % (maxT + 4));
        V.line(g, phx, G.y0 - 4, phx, yB + 4, { color: V.alpha(C.ink, 0.3), lw: 1 });
        let by = G.lanesEnd + 14;
        by += 14 * wrap(g, keyText(), 12, by, w - 24, 14, { size: 11, color: C.muted, font: 'mono', maxLines: 3 }) + 10;
        const msg = msgOf(), col = p.hang || (p.race && !S.jit) ? C.red || C.c : S.jit ? C.d : C.ink;
        by += 16 * wrap(g, msg, 12, by, w - 24, 16, { size: 12.5, color: col, weight: 600, maxLines: 8 }) + 8;
        const leg = [['GEMM', C.a], ['aux', C.b], ['atenção', C.d], ['estado', C.e], ['lê estado', C.f], ['JIT', C.red || C.c]];
        g.font = fnt({ size: 11 }); const legW = leg.map(([lb]) => 28 + g.measureText(lb).width);
        let lx = 12, lyy = by + 10;
        leg.forEach(([lb, c], i) => { if (lx + legW[i] > w - 12) { lx = 12; lyy += 18; } V.box(g, lx, lyy - 10, 12, 12, { r: 3, fill: V.alpha(c, 0.8) }); T(g, lb, lx + 16, lyy, { size: 11, color: C.muted }); lx += legW[i]; });
      });
    }

    const tfColor = (C) => { if (!C.__tf) C.__tf = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; return C.__tf; };

    /* =============== Figura 11 (TensorFold): os bits de uma linha =============== */
    {
      const fig = $('#fig-tf-bits');
      const K = 4096, SPL = [64, 32, 8, 4];
      const prod = new Float32Array(K);
      { const r = V.rng(5); for (let k = 0; k < K; k++) { const x = Math.fround(r.normal() * (r() < 0.02 ? 8 : 1)); const wv = Math.fround(r.normal() * 0.02); prod[k] = Math.fround(x * wv); } }
      const parts = {};
      SPL.forEach((n) => { const L = K / n, out = new Float32Array(n); for (let s = 0; s < n; s++) { let a = 0; for (let k = s * L; k < (s + 1) * L; k++) a = Math.fround(a + prod[k]); out[s] = a; } parts[n] = out; });
      const idOrd = (n) => Array.from({ length: n }, (_, i) => i);
      const prefix = (ps, ord) => { const out = []; let a = 0; for (const i of ord) { a = Math.fround(a + ps[i]); out.push(a); } return out; };
      const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
      const bitsOf = (v) => { f32[0] = v; return u32[0] >>> 0; };
      const fmtV = (v) => comma(v.toPrecision(9)).replace('-', '−');
      const hex = (b) => '0x' + b.toString(16).toUpperCase().padStart(8, '0');
      const S = { mode: 'lote', R: 12, launch: 0, t0: null, kick: true, key: '' };
      const splitsOf = (mode, R) => (mode === 'forma' ? 4 : mode === 'atom' ? 8 : R <= 8 ? 64 : R < 32 ? 32 : 8);
      const permOf = (L) => { const o = idOrd(8), r = V.rng(9045 + (L + 3) * 7919); for (let i = 7; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const tmp = o[i]; o[i] = o[j]; o[j] = tmp; } return o; };
      const cache = {};
      function run(mode, R, L) {
        const n = splitsOf(mode, R), key = mode === 'atom' ? 'a' + L : 'n' + n;
        if (!cache[key]) { const ord = mode === 'atom' ? permOf(L) : idOrd(n), pre = prefix(parts[n], ord); cache[key] = { n, ord, pre, val: pre[pre.length - 1], b: bitsOf(pre[pre.length - 1]) }; }
        return cache[key];
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Regra do kernel', options: [{ v: 'lote', label: 'split pelo lote' }, { v: 'forma', label: 'split pela forma (TensorFold)' }, { v: 'atom', label: 'atomicAdd fp32' }], value: 'lote', onChange: (v) => { S.mode = v; S.kick = true; } });
      UI.slider(ctl, { label: 'linhas na janela (R)', min: 1, max: 64, step: 1, value: 12, onInput: (v) => { S.R = v; S.kick = true; } });
      UI.button(ctl, '↻ relançar', () => { S.launch++; S.kick = true; });
      const rS = UI.readout(ctl, 'partições S', '—'), rB = UI.readout(ctl, 'bits vs referência', '—'), rD = UI.readout(ctl, 'diferença', '—');
      fitH(fig, (w) => (w < 560 ? 480 : 420));
      const P = 3.6;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), pad = 14, narrow = w < 560;
        if (S.t0 === null || S.kick) { S.t0 = t; S.kick = false; }
        const el = t - S.t0, auto = S.mode === 'atom' ? Math.floor(el / P) : 0, L = S.launch + auto, u = (el % P) / P;
        const cur = run(S.mode, S.R, L);
        const shown = u >= 0.7 ? L : L - 1;
        const res = run(S.mode, S.R, shown), ref = S.mode === 'atom' ? run('atom', 1, shown - 1) : run(S.mode, 1, 0);
        const diff = Math.abs(res.b - ref.b), eq = diff === 0;
        const key = S.mode + '|' + S.R + '|' + shown;
        if (key !== S.key) { S.key = key; rS.set(String(cur.n)); rB.set(eq ? 'iguais' : 'diferentes', eq ? 'ok' : 'bad'); rD.set(eq ? '0 ulp' : diff + (diff > 1 ? ' ulps' : ' ulp')); }
        // janela de verificação
        const sq = 9, gp = 2, perRow = Math.min(32, Math.floor((w - 2 * pad) / (sq + gp)));
        T(g, `janela: R = ${S.R} linha${S.R > 1 ? 's' : ''} na chamada`, pad, 16, { size: 12, weight: 600, color: C.ink });
        for (let i = 0; i < S.R; i++) {
          const x = pad + (i % perRow) * (sq + gp), y = 24 + Math.floor(i / perRow) * (sq + gp);
          V.box(g, x, y, sq, sq, { r: 2, fill: i === 0 ? C.c : V.alpha(C.muted, 0.35) });
        }
        const gridW = Math.min(S.R, perRow) * (sq + gp), gridH = Math.ceil(S.R / perRow) * (sq + gp);
        if (pad + gridW + 90 < w * 0.5) T(g, '← nossa linha', pad + gridW + 6, 32, { size: 11, color: C.c });
        const rule = S.mode === 'forma' ? `S = ${cur.n}: SK sai da forma do peso (K = 4.096), não de R` : S.mode === 'atom' ? `S = ${cur.n} parciais, somados por atomicAdd na ordem de chegada · lançamento ${L + 1}` : `S = ${cur.n} partições: bucket de R (≤ 8 → 64, < 32 → 32, senão 8)`;
        let yA;
        if (!narrow) { const rx = Math.max(w * 0.5, pad + gridW + 100); wrap(g, rule, rx, 16, w - pad - rx, 15, { size: 12, color: C.ink, font: 'mono', maxLines: 3 }); yA = Math.max(24 + gridH, 52) + 22; }
        else { const nl = wrap(g, rule, pad, 24 + gridH + 16, w - 2 * pad, 15, { size: 11.5, color: C.ink, font: 'mono', maxLines: 3 }); yA = 24 + gridH + 16 + nl * 15 + 14; }
        // barra de K
        T(g, 'a linha: 4.096 produtos x·w em fp32, fatiados em K', pad, yA, { size: 11.5, color: C.muted });
        const bx0 = pad, bx1 = w - pad, by = yA + 8, bh = 18, sw = (bx1 - bx0) / cur.n;
        for (let s = 0; s < cur.n; s++) V.box(g, bx0 + s * sw, by, Math.max(1, sw - (cur.n > 32 ? 0.5 : 1.5)), bh, { r: cur.n > 32 ? 1 : 3, fill: V.alpha(C.series[s % 6], 0.6) });
        // acumulador
        const aw = Math.min(300, w - 2 * pad), ax = (w - aw) / 2, ay = by + bh + 56, ah = 40;
        const t0a = 0.05, span = 0.5, fly = 0.14;
        let arrived = 0;
        cur.ord.forEach((s, j) => {
          const d = t0a + (span * j) / cur.n, f = (u - d) / fly;
          if (f >= 1) { arrived = j + 1; return; }
          if (f <= 0) return;
          const sx = bx0 + (s + 0.5) * sw, sy = by + bh, ex = ax + aw / 2 + ((j % 7) - 3) * 6, ey = ay;
          const e = V.ease.inOut(f);
          const px = V.lerp(sx, ex, e), py = V.lerp(sy, ey, e) - Math.sin(Math.PI * e) * 14;
          V.dot(g, px, py, cur.n > 32 ? 2.6 : 4, C.series[s % 6]);
        });
        V.box(g, ax, ay, aw, ah, { r: 10, fill: V.alpha(S.mode === 'atom' ? C.d : C.a, 0.12), stroke: S.mode === 'atom' ? C.d : C.a, lw: 1.5 });
        const runV = arrived ? cur.pre[arrived - 1] : 0;
        T(g, `Σ fp32 = ${arrived ? fmtV(runV) : '0'}  (${arrived}/${cur.n})`, ax + aw / 2, ay + 17, { size: narrow ? 11.5 : 12.5, align: 'center', font: 'mono', color: C.ink, weight: 600 });
        const ordTxt = cur.n <= 8 ? 'ordem: ' + cur.ord.join(' → ') : `ordem: 0 → ${cur.n - 1}, em sequência`;
        T(g, ordTxt, ax + aw / 2, ay + 33, { size: 11, align: 'center', font: 'mono', color: C.muted });
        // bits
        const labW = narrow ? 0 : 150, cw = Math.max(6, Math.min(18, (w - 2 * pad - labW) / 32 - 1.5)), cg = cw + 1.5;
        let yb = ay + ah + 26;
        const strip = (lab, r, other, y) => {
          T(g, lab, pad, narrow ? y : y + cw * 0.5 + 18, { size: 11.5, weight: 600, color: C.ink });
          const yy = narrow ? y + 6 : y + 8, xx = pad + labW;
          T(g, `${fmtV(r.val)} · ${hex(r.b)}`, xx, yy - 2 + (narrow ? 13 : 0), { size: 11, font: 'mono', color: C.muted });
          const yc = yy + (narrow ? 18 : 6);
          for (let k = 0; k < 32; k++) {
            const bit = (r.b >>> (31 - k)) & 1, ob = (other.b >>> (31 - k)) & 1;
            const grp = k === 0 ? C.c : k < 9 ? C.d : C.b;
            const x = xx + k * cg;
            V.box(g, x, yc, cw, cw, { r: 2, fill: V.alpha(grp, bit ? 0.75 : 0.12) });
            if (bit !== ob) { g.strokeStyle = C.red || C.c; g.lineWidth = 2; V.rr(g, x - 1, yc - 1, cw + 2, cw + 2, 3); g.stroke(); }
            if (cw >= 13) T(g, String(bit), x + cw / 2, yc + cw / 2 + 4, { size: 11, align: 'center', font: 'mono', color: bit ? onTxt(C) : C.muted });
          }
          return yc + cw;
        };
        yb = strip('nesta janela', res, ref, yb) + 14;
        yb = strip(S.mode === 'atom' ? 'lançamento anterior' : 'sozinha (R = 1)', ref, res, yb) + 12;
        T(g, 'sinal · expoente (8) · mantissa (23)', w - pad, yb, { size: 11, align: 'right', color: C.muted });
        yb += 20;
        const verdict = S.mode === 'atom'
          ? (eq ? 'mesmos bits do lançamento anterior, desta vez' : `${diff} ulp${diff > 1 ? 's' : ''} de diferença: mesma janela, outra ordem de chegada ✗`)
          : (eq ? 'mesmos 32 bits da linha sozinha ✓ → rascunho = serial' : `${diff} ulp${diff > 1 ? 's' : ''} de diferença ✗ → num quase empate de logits, o token sorteado pode mudar`);
        const nv = wrap(g, verdict, pad, yb, w - 2 * pad, 16, { size: 13, weight: 700, color: eq ? C.e : C.red || C.c, maxLines: 2 });
        yb += nv * 16 + 4;
        const expl = S.mode === 'forma' ? 'A mesma cadeia de somas em qualquer janela: mude R à vontade.' : S.mode === 'atom' ? 'Os blocos terminam em ordem diferente a cada lançamento. Por isso o TensorFold só usa atomics em inteiros.' : eq ? `R = ${S.R} cai no mesmo bucket da linha sozinha (≤ 8 → 64 fatias). Passe de 8 linhas e veja os bits mudarem.` : 'Cruzar um bucket refaz as fatias, e a soma em fp32 arredonda diferente.';
        wrap(g, expl, pad, yb, w - 2 * pad, 15, { size: 11.5, color: C.muted, maxLines: 2 });
      });
    }

    /* =============== Figura 12 (TensorFold): ordem de lançamento, L2 e gargalos =============== */
    {
      const fig = $('#fig-tf-ordem');
      const NB = 8, CAP = 4, WV = 4;
      const S = { view: 'l2', nf: false, items: 10, il: false, t0: null, ilT: -99, ilPrev: false, sims: null };
      function simL2(I, nf) {
        const seq = [];
        if (nf) { for (let i = 0; i < I; i++) for (let b = 0; b < NB; b++) seq.push([i, b]); } else { for (let b = 0; b < NB; b++) for (let i = 0; i < I; i++) seq.push([i, b]); }
        const lru = [], hits = [], states = [], pos = {};
        seq.forEach(([i, b], k) => { const p = lru.indexOf(i), hit = p >= 0; if (hit) lru.splice(p, 1); else if (lru.length >= CAP) lru.pop(); lru.unshift(i); hits.push(hit); states.push(lru.slice()); pos[i + ',' + b] = k; });
        return { seq, hits, states, pos, miss: hits.filter((x) => !x).length };
      }
      const resim = () => { S.sims = { a: simL2(S.items, false), b: simL2(S.items, true) }; S.t0 = null; };
      resim();
      const PLAN = 'LLLLLHHHHHHLLLLHHHHHHHLL'.split('').map((c) => c === 'H');
      const nh = PLAN.filter((x) => x).length, nl = PLAN.length - nh;
      const ILV = (() => { let r = 0, l = 0; const keys = PLAN.map((hv, i) => ({ i, k: hv ? (r++ + 0.5) / nh : (l++ + 0.5) / nl, hv })); return keys.slice().sort((a, b) => a.k - b.k || (b.hv - a.hv)).map((x) => x.i); })();
      const slotOf = { plan: PLAN.map((_, i) => i), il: [] };
      ILV.forEach((itemIdx, slot) => { slotOf.il[itemIdx] = slot; });
      function waves(il) {
        const order = il ? ILV : PLAN.map((_, i) => i), out = [];
        for (let k = 0; k < order.length; k += WV) {
          const grp = order.slice(k, k + WV); let dr = 0, tc = 0;
          grp.forEach((i) => { if (PLAN[i]) { dr += 0.25; tc += 1; } else { dr += 1; tc += 0.3; } });
          const tt = Math.max(dr, tc) / 2;
          out.push({ grp, t: tt, ud: dr / 2 / tt, ut: tc / 2 / tt });
        }
        return out;
      }
      const WA = waves(false), WB = waves(true);
      const tot = (ws) => ws.reduce((a, x) => a + x.t, 0);
      const TA = tot(WA), TB = tot(WB);
      const avgU = (ws, k) => ws.reduce((a, x) => a + x[k] * x.t, 0) / tot(ws);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Visão', options: [{ v: 'l2', label: 'L2 · ordem dos blocos' }, { v: 'gar', label: 'gargalos · ORDER' }], value: 'l2', onChange: (v) => { S.view = v; gA.show(v === 'l2'); gB.show(v === 'gar'); S.t0 = null; refit(); upd(); } });
      const gA = UI.group(ctl), gB = UI.group(ctl);
      UI.seg(gA.el, { label: 'Ordem', options: [{ v: false, label: 'item primeiro (decode)' }, { v: true, label: 'NFIRST (prompt)' }], value: false, onChange: (v) => { S.nf = v; S.t0 = null; upd(); } });
      UI.slider(gA.el, { label: 'itens no chunk', min: 3, max: 16, step: 1, value: 10, onInput: (v) => { S.items = v; resim(); refit(); upd(); } });
      const rM = UI.readout(gA.el, 'linhas X lidas da DRAM', '—'), rH = UI.readout(gA.el, 'acertos no L2', '—');
      UI.seg(gB.el, { label: 'Ordem', options: [{ v: false, label: 'ordem do plano' }, { v: true, label: 'intercalada (ORDER)' }], value: false, onChange: (v) => { S.ilPrev = S.il; S.il = v; S.ilT = performance.now() / 1000; S.t0 = null; upd(); } });
      const rT = UI.readout(gB.el, 'tempo total (u.a.)', '—'), rU = UI.readout(gB.el, 'uso médio DRAM · TC', '—');
      gB.show(false);
      const kib = 128;
      function upd() {
        const sm = S.nf ? S.sims.b : S.sims.a, n = sm.seq.length;
        rM.set(`${sm.miss} × 128 KiB = ${comma((sm.miss * kib / 1024).toFixed(2))} MiB`, S.nf ? 'ok' : sm.miss === S.items ? 'ok' : 'bad');
        rH.set(Math.round((100 * (n - sm.miss)) / n) + '%');
        const ws = S.il ? WB : WA;
        rT.set(comma(tot(ws).toFixed(1)), S.il ? 'ok' : null);
        rU.set(`${Math.round(100 * avgU(ws, 'ud'))}% · ${Math.round(100 * avgU(ws, 'ut'))}%`);
      }
      upd();
      const refit = fitH(fig, (w) => (w < 600 ? (S.view === 'l2' ? 330 + S.items * 25 : 430) : S.view === 'l2' ? 470 : 340));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), pad = 14, narrow = w < 600;
        if (S.t0 === null) S.t0 = t;
        const el = t - S.t0;
        if (S.view === 'l2') {
          const sm = S.nf ? S.sims.b : S.sims.a, n = sm.seq.length, rate = n / 6.5, hold = 2.2;
          if (el > n / rate + hold) S.t0 = t;
          const step = Math.min(n, Math.floor(el * rate)), frac = el * rate - Math.floor(el * rate);
          const I = S.items;
          // geometria
          let mx, my, mw, mh, px, py, pw;
          if (!narrow) { mx = pad + 52; my = 40; mw = w * 0.55 - mx; mh = h - my - 96; px = w * 0.6; py = 40; pw = w - pad - px; }
          else { mx = pad + 46; my = 40; mw = w - pad - mx; mh = I * 25; px = pad; py = my + mh + 22; pw = w - 2 * pad; }
          const cs = Math.max(8, Math.min(30, mw / NB - 3, mh / I - 3)), gpx = 3;
          T(g, S.nf ? 'NFIRST: blockIdx.x = bloco-n, os blocos de um item em sequência' : 'item primeiro: blockIdx.x = item, os itens se revezam', pad, 18, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
          for (let b = 0; b < NB; b++) T(g, 'n' + b, mx + b * (cs + gpx) + cs / 2, my - 6, { size: 11, align: 'center', font: 'mono', color: C.muted });
          for (let i = 0; i < I; i++) {
            T(g, 'item ' + i, mx - 6, my + i * (cs + gpx) + cs / 2 + 4, { size: 11, align: 'right', font: 'mono', color: C.muted });
            for (let b = 0; b < NB; b++) {
              const k = sm.pos[i + ',' + b], x = mx + b * (cs + gpx), y = my + i * (cs + gpx);
              if (k < step) V.box(g, x, y, cs, cs, { r: 3, fill: V.alpha(sm.hits[k] ? C.e : C.c, 0.78) });
              else { V.box(g, x, y, cs, cs, { r: 3, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.3), lw: 1 }); if (cs >= 20) T(g, String(k + 1), x + cs / 2, y + cs / 2 + 4, { size: 11, align: 'center', font: 'mono', color: V.alpha(C.muted, 0.9) }); }
            }
          }
          // painel L2 / DRAM
          const cur = Math.min(n - 1, step), [ci, cb] = sm.seq[cur], st = sm.states[cur] || [];
          const cx = mx + cb * (cs + gpx) + cs / 2, cy = my + ci * (cs + gpx) + cs / 2;
          if (step < n) { V.glow(g, cx, cy, cs, sm.hits[cur] ? C.e : C.c); g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, cx - cs / 2 - 1, cy - cs / 2 - 1, cs + 2, cs + 2, 4); g.stroke(); }
          const l2w = narrow ? pw * 0.5 - 6 : pw, slotH = 22, l2h = 26 + CAP * (slotH + 4);
          const l2x = px, l2y = py;
          V.box(g, l2x, l2y, l2w, l2h, { r: 10, fill: V.alpha(C.e, 0.07), stroke: V.alpha(C.e, 0.6), lw: 1.5 });
          T(g, `L2 (cabem ${CAP} itens)`, l2x + 10, l2y + 17, { size: 11.5, weight: 600, color: C.ink });
          for (let s = 0; s < CAP; s++) {
            const it = st[s], yy = l2y + 26 + s * (slotH + 4);
            if (it == null) { V.box(g, l2x + 8, yy, l2w - 16, slotH, { r: 6, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25), lw: 1 }); continue; }
            const isCur = step < n && it === ci;
            V.box(g, l2x + 8, yy, l2w - 16, slotH, { r: 6, fill: V.alpha(C.e, isCur ? 0.5 : 0.2), stroke: isCur ? C.ink : null, lw: 1.5 });
            T(g, `linhas do item ${it} · 128 KiB`, l2x + 16, yy + 15, { size: 11, font: 'mono', color: C.ink, maxW: l2w - 30 });
          }
          const dx = narrow ? px + pw * 0.5 + 6 : px, dy = narrow ? py : l2y + l2h + 26, dw = narrow ? pw * 0.5 - 6 : pw, dh = narrow ? l2h : 58;
          V.box(g, dx, dy, dw, dh, { r: 10, fill: V.alpha(C.c, 0.07), stroke: V.alpha(C.c, 0.6), lw: 1.5 });
          T(g, 'DRAM', dx + 10, dy + 17, { size: 11.5, weight: 600, color: C.ink });
          const missSoFar = sm.hits.slice(0, step).filter((x) => !x).length;
          wrap(g, `linhas lidas: ${missSoFar} × 128 KiB`, dx + 10, dy + 36, dw - 20, 15, { size: 11, font: 'mono', color: C.muted, maxLines: 2 });
          if (step < n) {
            const miss = !sm.hits[cur], e = V.ease.inOut(Math.min(1, frac * 1.4));
            const sx = miss ? dx + dw / 2 : l2x + l2w / 2, sy = miss ? (narrow ? dy + dh : dy) : l2y + l2h / 2;
            const tx = cx, ty = cy;
            V.dot(g, V.lerp(sx, tx, e), V.lerp(sy, ty, e), 4.5, miss ? C.c : C.e);
          }
          // comparação
          const ba = S.sims.a.miss, bb = S.sims.b.miss, mxm = Math.max(ba, bb, 1);
          const cyb = narrow ? dy + dh + 22 : h - 74, cbw = w - 2 * pad - (narrow ? 92 : 110) - 110;
          [[false, 'item primeiro', ba], [true, 'NFIRST', bb]].forEach(([nf, lb, v], r) => {
            const yy = cyb + r * 22, on = nf === S.nf;
            T(g, lb, pad, yy + 12, { size: 11.5, weight: on ? 700 : 400, color: on ? C.ink : C.muted });
            const x0 = pad + (narrow ? 92 : 110);
            V.box(g, x0, yy, Math.max(3, (cbw * v) / mxm), 15, { r: 4, fill: V.alpha(nf ? C.e : C.c, on ? 0.8 : 0.35) });
            T(g, `${v} × 128 KiB`, x0 + Math.max(3, (cbw * v) / mxm) + 6, yy + 12, { size: 11, font: 'mono', color: C.muted });
          });
          T(g, 'mesma conta, na mesma ordem dentro de cada saída: mesmos bits ✓', pad, cyb + 58, { size: 11.5, color: C.e, weight: 600, maxW: w - 2 * pad });
        } else {
          const ws = S.il ? WB : WA, now = performance.now() / 1000, mv = V.ease.inOut(V.clamp((now - S.ilT) / 0.9, 0, 1));
          const N = PLAN.length, perRow = narrow ? 12 : 24, chW = Math.min(30, (w - 2 * pad) / perRow - 4), chH = 26, rowGap = 26, rowsQ = Math.ceil(N / perRow);
          T(g, S.il ? 'intercalada: pesado r → (r + ½)/Np, leve l → (l + ½)/Nl' : 'ordem do plano: os itens chegam em levas do mesmo tipo', pad, 18, { size: 12, weight: 600, color: C.ink, maxW: w - 2 * pad });
          const qy = 34;
          const total = tot(ws), playT = (el * 1.3) % (total + 1.5), wi = (() => { let a = 0; for (let k = 0; k < ws.length; k++) { a += ws[k].t; if (playT < a) return k; } return -1; })();
          const curSet = wi >= 0 ? new Set(ws[wi].grp) : new Set();
          const slotXY = (s) => [pad + (s % perRow) * (chW + 4), qy + Math.floor(s / perRow) * (chH + rowGap)];
          for (let s = 0; s < N; s += WV) { const [x, y] = slotXY(s), gw = WV * (chW + 4) - 4, k = s / WV; V.box(g, x - 2, y - 2, gw + 4, chH + 4, { r: 6, fill: V.alpha(C.muted, k === wi ? 0.16 : 0.06) }); T(g, `leva ${k + 1}`, x + gw / 2, y + chH + 15, { size: 11, align: 'center', color: k === wi ? C.ink : C.muted, weight: k === wi ? 700 : 400 }); }
          PLAN.forEach((hv, i) => {
            const sNew = S.il ? slotOf.il[i] : i, sOld = S.ilPrev ? slotOf.il[i] : i;
            const [x1, y1] = slotXY(sNew), [x0, y0] = slotXY(sOld);
            const x = V.lerp(x0, x1, mv), y = V.lerp(y0, y1, mv) - Math.sin(Math.PI * mv) * 8;
            const c = hv ? C.b : C.a, on = curSet.has(i);
            if (on) V.glow(g, x + chW / 2, y + chH / 2, chW, c);
            V.box(g, x, y, chW, chH, { r: 5, fill: V.alpha(c, on ? 0.95 : 0.6), stroke: on ? C.ink : null, lw: 1.5 });
            if (chW >= 18) T(g, hv ? 'TC' : 'D', x + chW / 2, y + chH / 2 + 4, { size: 11, align: 'center', weight: 700, color: onTxt(C) });
          });
          // linha do tempo
          const ty0 = qy + rowsQ * (chH + rowGap) + 18, lx = pad + 92, lw = w - pad - lx, laneH = 40;
          const Tmax = Math.max(TA, TB), X = (tt) => lx + (tt / Tmax) * lw;
          [['DRAM', 'ud', C.a], ['tensor cores', 'ut', C.b]].forEach(([lb, kk, c], r) => {
            const y = ty0 + r * (laneH + 14);
            T(g, lb, pad, y + laneH / 2 + 4, { size: 11.5, weight: 600, color: C.ink });
            V.box(g, lx, y, lw, laneH, { r: 5, fill: V.alpha(C.muted, 0.07) });
            let a = 0;
            ws.forEach((wv, k) => {
              const x0 = X(a), x1 = X(a + wv.t), hh = laneH * wv[kk];
              V.box(g, x0 + 1, y + laneH - hh, Math.max(2, x1 - x0 - 2), hh, { r: 3, fill: V.alpha(c, k === wi ? 0.95 : 0.55) });
              a += wv.t;
            });
          });
          const phx = X(Math.min(playT, total));
          V.line(g, phx, ty0 - 6, phx, ty0 + 2 * laneH + 20, { color: V.alpha(C.ink, 0.45), lw: 1.5 });
          const yE = ty0 + 2 * (laneH + 14) + 8;
          V.line(g, X(TB), yE - 4, X(TB), yE + 4, { color: C.e, lw: 2 }); V.line(g, X(TA), yE - 4, X(TA), yE + 4, { color: C.c, lw: 2 });
          T(g, `fim: plano ${comma(TA.toFixed(1))} · intercalada ${comma(TB.toFixed(1))} (u.a.)`, w - pad, yE + 18, { size: 11.5, align: 'right', font: 'mono', color: C.ink });
          const yL = yE + 40;
          wrap(g, 'Leve (D): expert com um só item, lê os pesos da DRAM e faz pouca conta. Pesado (TC): expert com vários itens, pesos repartidos pelo L2, limitado pelo tensor core. Altura = uso do recurso em cada leva de 4 blocos; custos ilustrativos.', pad, yL, w - 2 * pad, 15, { size: 11, color: C.muted, maxLines: 6 });
        }
      });
    }

    /* =============== Figura 10: mapa de kernels =============== */
    {
      const fig = $('#fig-mapa');
      const LIB = { triton: ['Triton', 'a'], cutlass: ['CUTLASS', 'b'], marlin: ['Marlin', 'c'], flashinfer: ['FlashInfer', 'd'], b12x: ['b12x · CuTeDSL', 'f'], exl3: ['ExLlamaV3 + CUDA do repo', 'e'], cublas: ['cuBLAS', 'b'], deepgemm: ['DeepGEMM', 'c'], tilelang: ['TileLang', 'e'], fa: ['FlashAttention', 'a'], cuda: ['CUDA', 'e'], unk: ['kernel não identificado', 'muted'], tfcu: ['CUDA C++ do TensorFold', 'd'], nccl: ['NCCL', 'b'] };
      const MAP = {
        glm: [
          { st: 'KDA · atenção linear ×34', pre: ['chunk_kda (FLA)', ['triton']], dec: ['fused_recurrent_kda', ['triton']], opt: 'FlashKDA no TP3' },
          { st: 'Projeções densas (KDA, MLA, shared, MLP)', pre: ['in_proj do KDA em BF16 se M > 512; o resto em Marlin W8A16', ['cublas', 'marlin']], dec: ['Marlin W8A16 (FP8 weight-only por canal)', ['marlin']] },
          { st: 'MLA esparsa ×11', pre: ['indexer kpool + paged MQA → sparse MLA SM120 (GLM_NSA 576)', ['triton', 'deepgemm', 'flashinfer']], dec: ['indexer kpool + paged MQA → sparse MLA SM120 (GLM_NSA 576)', ['triton', 'deepgemm', 'flashinfer']] },
          { st: 'MoE roteado · 288 × top-8', pre: ['exl3_moe (thin) + E3 grouped (fat)', ['exl3']], dec: ['exl3_moe, lançamento único', ['exl3']], optDec: 'FAST K4/N256 ou cooperativo 2 estágios' },
          { st: 'Drafter DFlash2', pre: ['FLASH_ATTN, não causal + SWA', ['fa']], dec: ['FLASH_ATTN, não causal + SWA', ['fa']] },
        ],
        tf: [
          { st: 'KDA · atenção linear ×34', pre: ['kda_chunk.cu: forma WY/UT, 1 bloco por cabeça, sub-chunks de 32', ['tfcu']], dec: ['cadeia serial do kda.cu em 3 kernels (prep/step/out)', ['tfcu']], opt: 'KDA_CHUNKED=0: cadeia serial também no prompt' },
          { st: 'Projeções densas (KDA, MLA, shared, MLP)', pre: ['Q4 afim, grupos de 64, kernel de prompt', ['tfcu']], dec: ['qmm Q4: split-K pela forma, tile pela tabela DEC_TILES', ['tfcu']], opt: 'DENSE=fp8: _fmm em Triton; cabeça em FP8' },
          { st: 'MLA latente esparsa ×11', pre: ['seleção de pools + _sparse_onepass, sem parciais', ['triton']], dec: ['_scores_rows + atenção latente sobre linhas FP8 de 528 B', ['triton']], opt: 'atenção densa até 2.051 tokens' },
          { st: 'MoE roteado · 288 × top-8', pre: ['prompt_gateup/down: Yᵀ = WᵀXᵀ, blocos de um item juntos + ORDER', ['tfcu']], dec: ['dec_kernel: down com epílogo fundido, cargas .nc', ['tfcu']], optDec: 'prefetch de L2 num stream lateral' },
          { st: 'Troca entre os 2 ranks', pre: ['parciais fp32 por NCCL; cola HC dividida por linhas', ['nccl']], dec: ['all-gather RDMA de um tiro (RoCE), até 512 KiB', ['tfcu']], opt: 'soma em ordem de rank: bits iguais nos dois' },
          { st: 'Drafter DFlash2', pre: ['janela de 2.048 linhas num anel de 2.176', ['triton']], dec: ['bloco + seletor (F.linear por stream)', ['triton', 'cublas']], opt: 'copy drafts verificados antes' },
        ],
        qwen: [
          { st: 'Gated DeltaNet ×36', pre: ['chunk_gated_delta_rule (FLA)', ['triton']], dec: ['fused_gdn_decode_post_conv_mtp', ['cuda']] },
          { st: 'QSA ×12', pre: ['_qsa_mqa_paged_kernel → persistent_topk → sparse GQA split-K', ['triton']], dec: ['_qsa_mqa_paged_kernel → persistent_topk → sparse GQA split-K', ['triton']], opt: 'KV FP8 via KV_QUANT_MODE' },
          { st: 'MoE NVFP4 · 512 × top-10', pre: ['FlashInfer CUTLASS SM120, W4A4 grouped GEMM', ['flashinfer', 'cutlass']], dec: ['FlashInfer CUTLASS SM120, W4A4 grouped GEMM', ['flashinfer', 'cutlass']], opt: 'finalize não fundido no modo determinístico' },
          { st: 'Densas e lm_head', pre: ['cuBLAS BF16', ['cublas']], dec: ['cuBLAS BF16 (GEMMs skinny, M = 1–4)', ['cublas']], opt: 'CutlassFP8ScaledMM no FP8-dense; mm_mxfp8 ou emulação no MXFP8' },
          { st: 'Drafter MTP=3', pre: ['kernels do alvo + lm_head BF16 248.320 × 2560', ['cublas']], dec: ['kernels do alvo + lm_head BF16 248.320 × 2560', ['cublas']], opt: 'vocabulário reduzido do drafter' },
        ],
        ds: [
          { st: 'mHC · 4 fluxos residuais', pre: ['mHC pre/post fundido', ['tilelang']], dec: ['mHC pre/post fundido', ['tilelang']], opt: 'b12x mHC ≤ 16 tokens (lane B)' },
          { st: 'Lineares densas · FP8 bloco 128×128 (UE8M0)', pre: ['wq/wkv/wo e shared expert: kernel não nomeado nas fontes', ['unk']], dec: ['wq/wkv/wo e shared expert: kernel não nomeado nas fontes', ['unk']], opt: 'lane B: VLLM_USE_B12X_FP8_GEMM, não setar (assert do DeepGEMM)' },
          { st: 'Indexer C4A', pre: ['fp8_mqa_logits + top-k', ['deepgemm', 'triton']], dec: ['fp8_paged_mqa_logits + top-k', ['deepgemm', 'triton']], opt: 'sequence-parallel no prefill; cache MXFP4 (alias sm121)' },
          { st: 'Sparse MLA · SWA + C4/C128', pre: ['sparse MLA SM120 (orquestrador de prefill) + índices top-k+SWA', ['flashinfer', 'triton']], dec: ['sparse MLA SM120, split-K, ≤ 64 linhas por chamada', ['flashinfer']], opt: 'lane A: impl Anemll, #141 fatia acima de 64; lane B: nvidia/sm120.py' },
          { st: 'MoE MXFP4 · 256 × top-6', pre: ['b12x W4A16: dequant → MMA BF16', ['b12x']], dec: ['b12x W4A16: dequant → MMA BF16', ['b12x']], opt: 'TC_DECODE = 0 (−15%)' },
          { st: 'Drafter DSpark', pre: ['scores/out do drafter + argmax Markov fundido', ['triton']], dec: ['scores/out do drafter + argmax Markov fundido', ['triton']], opt: 'código legível na lane B' },
        ],
      };
      const S = { rc: 'glm', ph: 'dec' };
      const refit = fitH(fig, (w) => (w < 640 ? 24 + MAP[S.rc].length * 160 : 420));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'tf', label: 'GLM·TF' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.rc = v; refit(); } });
      UI.seg(ctl, { label: 'Fase', options: [{ v: 'pre', label: 'prefill' }, { v: 'dec', label: 'decode' }], value: 'dec', onChange: (v) => { S.ph = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), items = MAP[S.rc], col = S.rc === 'tf' ? tfColor(C) : C[S.rc] || C.a, narrow = w < 640;
        const n = items.length, pos = [];
        let cw, chh;
        if (!narrow) {
          const cols = 3, gx = 18, gy = 26; cw = (w - 32 - gx * (cols - 1)) / cols; chh = Math.min(168, (h - 40 - gy) / 2);
          items.forEach((it, i) => { const row = Math.floor(i / cols), c = row % 2 === 0 ? i % cols : cols - 1 - (i % cols); pos.push([16 + c * (cw + gx), 16 + row * (chh + gy)]); });
        } else {
          cw = w - 32; chh = Math.min(150, (h - 24) / n - 12);
          items.forEach((it, i) => pos.push([16, 12 + i * (chh + 12)]));
        }
        const centers = pos.map(([x, y]) => [x + cw / 2, y + chh / 2]);
        g.lineWidth = 7; g.lineCap = 'round'; g.strokeStyle = V.alpha(col, 0.3);
        g.beginPath(); centers.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke(); g.lineCap = 'butt';
        items.forEach((it, i) => {
          const [x, y] = pos[i], ent = it[S.ph];
          V.box(g, x, y, cw, chh, { r: 12, fill: C.panel || C.bg, stroke: V.alpha(col, 0.55), lw: 1.5 });
          V.box(g, x, y, 6, chh, { r: 3, fill: col });
          T(g, `${i + 1}`, x + cw - 12, y + 18, { size: 11, align: 'right', font: 'mono', color: C.muted });
          const nt = wrap(g, it.st, x + 16, y + 20, cw - 44, 15, { size: 12.5, weight: 700, color: C.ink, maxLines: 2 });
          const ky = y + 26 + nt * 15;
          const ny = wrap(g, ent[0], x + 16, ky, cw - 28, 15, { size: 12, color: C.ink, font: 'mono', maxLines: 3 });
          let bx = x + 16; const by = ky + 6 + ny * 15;
          ent[1].forEach((lk) => {
            const [nm, ck] = LIB[lk]; g.font = fnt({ size: 11, weight: 600 }); const tw = g.measureText(nm).width + 14;
            if (bx + tw > x + cw - 8) return;
            V.box(g, bx, by, tw, 18, { r: 9, fill: V.alpha(C[ck], 0.22), stroke: V.alpha(C[ck], 0.7), lw: 1 });
            T(g, nm, bx + 7, by + 13, { size: 11, weight: 600, color: C.ink }); bx += tw + 6;
          });
          const opt = S.ph === 'dec' && it.optDec ? it.optDec : it.opt;
          if (opt && by + 36 < y + chh) wrap(g, '(' + opt + ')', x + 16, by + 34, cw - 28, 14, { size: 11, color: C.muted, maxLines: Math.max(1, Math.floor((y + chh - by - 26) / 14)) });
        });
        const total = centers.length - 1, u = (t * 0.12) % 1, sg = u * total, si = Math.min(total - 1, Math.floor(sg));
        const tp = V.along(centers[si][0], centers[si][1], centers[si + 1][0], centers[si + 1][1], V.ease.inOut(sg - si));
        V.glow(g, tp[0], tp[1], 18, C.c); V.dot(g, tp[0], tp[1], 5.5, C.c);
        if (!narrow) T(g, `${S.rc === 'glm' ? 'GLM-5.3-Flash EXL3' : S.rc === 'tf' ? 'GLM-5.3-Flash EXL3 · TensorFold' : S.rc === 'qwen' ? 'Qwen3.8-Flash-Next NVFP4' : 'DeepSeek-V4-Flash'} · ${S.ph === 'pre' ? 'prefill' : 'decode'}`, w - 16, h - 12, { size: 12, align: 'right', weight: 600, color: col });
      });
    }
  },
});

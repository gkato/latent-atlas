LA.chapter({
  id: 'checkpoints',
  part: 4,
  order: 24,
  title: 'Checkpoints: safetensors, configs e integridade',
  navTitle: 'Checkpoints',
  subtitle: 'Como um checkpoint é feito por dentro, como o vLLM transforma nomes de tensores em parâmetros fatiados por GPU, e a disciplina de cache, revisões e hashes que torna um modelo servido reprodutível.',
  minutes: 50,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['safetensors', 'checkpoint', 'Hugging Face', 'quantization_config', 'hf-overrides', 'SHA-256', 'NFS'],
  goals: [
    'Ler um <b>repositório de modelo</b> e o formato <b>safetensors</b> byte a byte (cabeçalho, offsets, shards, <code>weight_map</code>)',
    'Explicar como o <b>carregador do vLLM</b> renomeia, empilha e fatia tensores (TP × EP, pesos quantizados cortados junto com as escalas), e por que nomes errados geram camadas silenciosamente não quantizadas',
    'Corrigir metadados <b>sem tocar no cache</b> (<code>--hf-overrides</code> aninhado, configs montados por cima, preflight)',
    'Operar o <b>cache do Hugging Face</b> offline, pinar revisões, verificar integridade por hash e escolher entre rsync e NFS',
  ],
  prereqs: ['tres-modelos', 'quantizacao-fundamentos', 'fp8-fp4-microescala', 'hadamard-trelica-exl3'],
  summary: [
    'Safetensors = 8 bytes com o tamanho \\(N\\) do cabeçalho + \\(N\\) bytes de JSON (dtype, shape, data_offsets) + bytes crus. Por isso permite mmap, zero-copy e range reads, e não executa código.',
    'O <code>model.safetensors.index.json</code> mapeia cada tensor ao seu shard; um snapshot só está completo se todos os arquivos do <code>weight_map</code> existem.',
    'O carregador renomeia (<code>WeightsMapper</code>), empilha (<code>stacked_params</code>, com <code>shard_id</code>) e fatia por rank (<code>weight_loader</code>); escalas E8M0 são reinterpretadas (<code>view</code>), nunca convertidas.',
    'Peso quantizado se corta junto com as escalas: a fronteira do rank precisa cair numa fronteira de bloco (16 no NVFP4, 128 no FP8 em bloco, tiles de 16 no EXL3); escalas globais e sinais da dimensão não cortada são replicados. Em EP nada é cortado: cada rank guarda experts inteiros.',
    'O <code>quantization_config</code> dá significado aos bytes; no ModelOpt MIXED_PRECISION o casamento de <code>quantized_layers</code> é por string exata, e um nome que não bate cria uma camada não quantizada que só falha minutos depois.',
    'Correções vão por <code>--hf-overrides</code> (aninhado em <code>text_config</code> quando o modelo é multimodal) ou por configs montados por cima do snapshot; o cache nunca é editado.',
    'Cache HF = blobs endereçados por hash + snapshots de symlinks + <code>refs/main</code>; sirva com <code>HF_HUB_OFFLINE=1</code>, pine commits, escreva refs sem newline e verifique por SHA-256 / SHA-1 de blob git. Commit diferente não prova bytes diferentes, e tamanho igual não prova bytes iguais: as duas lanes da GLM servem os mesmos shards com pins distintos, e drafters do mesmo tamanho com outro SHA-256.',
  ],
  quiz: [
    { q: 'Um arquivo safetensors começa com os bytes <code>58 02 00 00 00 00 00 00</code>. Onde começa o blob de dados?', o: ['No byte 8', 'No byte 600', 'No byte 608', 'No byte 0x5802'], a: 2, e: 'Os 8 primeiros bytes são um u64 little-endian: 0x0258 = 600 bytes de cabeçalho JSON. O blob começa em 8 + 600 = 608. Ler 0x5802 é esquecer que o formato é little-endian.' },
    { q: 'Por que, numa camada <code>gate_up_proj</code> em TP=2, o rank 0 não pode simplesmente guardar a metade de cima da matriz fundida?', o: ['Porque a metade de cima não cabe na memória', 'Porque a metade de cima é todo o <code>gate</code> e nenhum <code>up</code>; cada rank precisa da sua fatia de <i>cada</i> sub-matriz para a SwiGLU combinar linhas correspondentes', 'Porque o NCCL exige tensores contíguos', 'Porque o <code>up</code> sempre fica replicado'], a: 1, e: 'O <code>weight_loader</code> recebe o <code>shard_id</code> e coloca a fatia \\(r\\) do gate na gaveta 0 e a fatia \\(r\\) do up na gaveta 1. Assim \\(\\text{SiLU}(g_i)\\cdot u_i\\) multiplica linhas que correspondem.' },
    { q: 'O checkpoint nvidia do Qwen declara metadados para <code>mtp.layers.0.mlp.experts</code>, mas o vLLM constrói a camada como <code>mtp.layers.48.mlp.experts</code>. O que acontece sem o patch?', o: ['O vLLM renumera automaticamente e tudo funciona', 'O boot falha em segundos com erro claro', 'A MoE da MTP é construída não quantizada e a carga morre ~7 min depois com “has no parameter w2_weight_scale_inv”', 'A MTP é desligada silenciosamente'], a: 2, e: 'O casamento de <code>quantized_layers</code> é por string exata; sem alias, <code>get_quant_method</code> devolve <code>None</code>, os parâmetros nascem em formato BF16 e a escala do checkpoint não tem onde morar. O preflight <code>--mtp-moe-algo</code> e o alias <code>.0.</code> → <code>.48.</code> evitam isso.' },
    { q: 'Qual detector pega um <code>config.json</code> reescrito com o mesmo tamanho (8 → 7 experts por token)?', o: ['Contagem de shards', 'Completude do <code>weight_map</code>', 'Comparação de tamanho desreferenciado', 'Hash de conteúdo (SHA-1 de blob git contra o manifest)'], a: 3, e: 'O arquivo existe e tem o mesmo número de bytes; só o hash de conteúdo muda. Para arquivos pequenos, o Hub publica o SHA-1 de blob git, que é SHA1("blob &lt;tamanho&gt;\\0" + conteúdo).' },
    { q: 'Por que <code>--hf-overrides \'{"rope_parameters": {...}}\'</code> não ligava o YaRN no Qwen?', o: ['Porque o vLLM ignora rope_parameters', 'Porque a mescla só desce em sub-configs; no nível raiz o valor foi gravado no config pai e nunca chegou ao <code>text_config</code>', 'Porque YaRN exige recompilar o modelo', 'Porque o JSON estava mal formado'], a: 1, e: '<code>_apply_dict_overrides</code> recursa apenas em chaves que são configs aninhados. O payload correto é <code>{"text_config": {"rope_parameters": …, "ple_embedding_dtype": …}}</code>.' },
    { q: 'Um script grava <code>echo $SHA &gt; refs/main</code> e o servidor, com <code>HF_HUB_OFFLINE=1</code> e huggingface_hub 1.28, morre com “repo not found”. Por quê?', o: ['O SHA estava errado', 'O modo offline não lê refs/main', 'O <code>echo</code> deixou um newline; a biblioteca, que não faz strip, procura <code>snapshots/&lt;sha&gt;\\n</code>, que não existe', 'Faltava o HF_TOKEN'], a: 2, e: 'Escreva refs com <code>printf \'%s\'</code> (ou <code>echo -n</code>) e normalize comparando bytes. A receita Qwen faz isso antes de cada launch (issue #36).' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ================= utilidades locais ================= */
    const FC = {};
    const setF = (g, size, font, weight) => { const k = size + '|' + (font || 'sans') + '|' + (weight || 400); g.font = FC[k] || (FC[k] = V.font(size, font || 'sans', weight || 400)); };
    const tw = (g, s, size, font, weight) => { setF(g, size, font, weight); return g.measureText(String(s)).width; };
    const clip = (g, s, maxW, size = 11, font = 'mono', weight = 400) => {
      s = String(s); setF(g, size, font, weight);
      if (maxW <= 8) return '';
      if (g.measureText(s).width <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (g.measureText(s.slice(0, m) + '…').width <= maxW) lo = m; else hi = m - 1; }
      return s.slice(0, lo) + '…';
    };
    const clipL = (g, s, maxW, size = 11, font = 'mono', weight = 400) => {
      s = String(s); setF(g, size, font, weight);
      if (maxW <= 8) return '';
      if (g.measureText(s).width <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (g.measureText('…' + s.slice(s.length - m)).width <= maxW) lo = m; else hi = m - 1; }
      return '…' + s.slice(s.length - lo);
    };
    const T = (g, s, x, y, o) => V.text(g, s, x, y, o);
    const wrap = (g, s, x, y, maxW, lh, o = {}) => {
      const size = o.size || 12, font = o.font || 'sans', weight = o.weight || 400;
      setF(g, size, font, weight);
      const words = String(s).split(' ');
      let line = '', yy = y, n = 0;
      for (const wd of words) {
        const t = line ? line + ' ' + wd : wd;
        if (g.measureText(t).width > maxW && line) {
          T(g, line, x, yy, o); line = wd; yy += lh; n++;
          if (o.maxLines && n >= o.maxLines) { line = ''; break; }
        } else line = t;
      }
      if (line) T(g, line, x, yy, o);
      return yy + lh;
    };
    const chip = (g, x, y, s, col, o = {}) => {
      const size = o.size || 11, font = o.font || 'mono', weight = o.weight || 500, hh = o.h || 20;
      const w = o.w || tw(g, s, size, font, weight) + 14;
      V.box(g, x, y, w, hh, { r: o.r == null ? 7 : o.r, fill: V.alpha(col, o.a == null ? 0.16 : o.a), stroke: V.alpha(col, o.sa == null ? 0.75 : o.sa), lw: o.lw || 1 });
      T(g, o.w ? clip(g, s, w - 12, size, font, weight) : s, x + 7, y + hh / 2 + 0.5, { size, font, weight, color: o.color || V.C().ink, base: 'middle' });
      return w;
    };
    const stageH = (fig, px) => { const st = fig.querySelector('.la-fig-stage'); if (st && st.style.height !== px + 'px') st.style.height = px + 'px'; };
    const hit = (p, x, y, w, h) => p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h;
    const bezPt = (x1, y1, x2, y2, u, horiz = true) => horiz
      ? V.bez([x1, y1], [x1 + (x2 - x1) * 0.5, y1], [x1 + (x2 - x1) * 0.5, y2], [x2, y2], u)
      : V.bez([x1, y1], [x1, y1 + (y2 - y1) * 0.5], [x2, y1 + (y2 - y1) * 0.5], [x2, y2], u);
    const bezLine = (g, x1, y1, x2, y2, o = {}, horiz = true) => {
      g.beginPath(); g.moveTo(x1, y1);
      if (horiz) g.bezierCurveTo(x1 + (x2 - x1) * 0.5, y1, x1 + (x2 - x1) * 0.5, y2, x2, y2);
      else g.bezierCurveTo(x1, y1 + (y2 - y1) * 0.5, x2, y1 + (y2 - y1) * 0.5, x2, y2);
      g.strokeStyle = o.color || V.C().muted; g.lineWidth = o.lw || 1.4;
      if (o.dash) g.setLineDash(o.dash); if (o.dashOffset) g.lineDashOffset = o.dashOffset;
      g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
    };
    const okCol = (C) => C.e, badCol = (C) => C.red || C.c;
    // números no padrão pt-BR (vírgula decimal) e potências de 10 com sobrescrito
    const nBR = (x, d = 2) => Number(x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const sciBR = (x, d = 3) => { if (!x) return '0'; const e = Math.floor(Math.log10(Math.abs(x))); const m = x / Math.pow(10, e); return nBR(m, d) + '·10' + String(e).split('').map((c) => SUP[c]).join(''); };
    // polilinha: comprimento, ponto no parâmetro u ∈ [0,1] e traço
    const polyLen = (pts) => { let L = 0; for (let k = 1; k < pts.length; k++) L += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); return L; };
    const polyPt = (pts, u) => {
      const L = polyLen(pts) * V.clamp(u, 0, 1); let acc = 0;
      for (let k = 1; k < pts.length; k++) {
        const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
        if (acc + d >= L) { const f = d ? (L - acc) / d : 0; return [V.lerp(pts[k - 1][0], pts[k][0], f), V.lerp(pts[k - 1][1], pts[k][1], f)]; }
        acc += d;
      }
      return pts[pts.length - 1];
    };
    const polyLine = (g, pts, o = {}) => {
      g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let k = 1; k < pts.length; k++) g.lineTo(pts[k][0], pts[k][1]);
      g.strokeStyle = o.color || V.C().muted; g.lineWidth = o.lw || 1.4; g.lineJoin = 'round';
      if (o.dash) g.setLineDash(o.dash); if (o.dashOffset) g.lineDashOffset = o.dashOffset;
      g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
    };
    // caixa com fundo opaco (para linhas desenhadas antes não atravessarem o texto)
    const solidBox = (g, x, y, w, h, o) => { V.box(g, x, y, w, h, { r: o.r == null ? 8 : o.r, fill: V.C().bg }); V.box(g, x, y, w, h, o); };

    /* ---------- SHA-256 e SHA-1 (implementação local, para o painel de integridade) ---------- */
    const K256 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
    const padMsg = (msg) => {
      const l = msg.length, nb = ((l + 9 + 63) >> 6) << 6;
      const m = new Uint8Array(nb); m.set(msg); m[l] = 0x80;
      const dv = new DataView(m.buffer); const bits = l * 8;
      dv.setUint32(nb - 4, bits >>> 0); dv.setUint32(nb - 8, Math.floor(bits / 4294967296));
      return { dv, nb };
    };
    const sha256 = (msg) => {
      const { dv, nb } = padMsg(msg);
      let H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
      const w = new Uint32Array(64);
      const rr = (x, n) => (x >>> n) | (x << (32 - n));
      for (let o = 0; o < nb; o += 64) {
        for (let i = 0; i < 16; i++) w[i] = dv.getUint32(o + i * 4);
        for (let i = 16; i < 64; i++) { const s0 = rr(w[i - 15], 7) ^ rr(w[i - 15], 18) ^ (w[i - 15] >>> 3); const s1 = rr(w[i - 2], 17) ^ rr(w[i - 2], 19) ^ (w[i - 2] >>> 10); w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0; }
        let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], gg = H[6], hh = H[7];
        for (let i = 0; i < 64; i++) {
          const S1 = rr(e, 6) ^ rr(e, 11) ^ rr(e, 25), ch = (e & f) ^ (~e & gg), t1 = (hh + S1 + ch + K256[i] + w[i]) >>> 0;
          const S0 = rr(a, 2) ^ rr(a, 13) ^ rr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = (S0 + mj) >>> 0;
          hh = gg; gg = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
        }
        H = [(H[0] + a) >>> 0, (H[1] + b) >>> 0, (H[2] + c) >>> 0, (H[3] + d) >>> 0, (H[4] + e) >>> 0, (H[5] + f) >>> 0, (H[6] + gg) >>> 0, (H[7] + hh) >>> 0];
      }
      return H.map((x) => x.toString(16).padStart(8, '0')).join('');
    };
    const sha1 = (msg) => {
      const { dv, nb } = padMsg(msg);
      let h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;
      const w = new Uint32Array(80); const rl = (x, n) => (x << n) | (x >>> (32 - n));
      for (let o = 0; o < nb; o += 64) {
        for (let i = 0; i < 16; i++) w[i] = dv.getUint32(o + i * 4);
        for (let i = 16; i < 80; i++) w[i] = rl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1) >>> 0;
        let a = h0, b = h1, c = h2, d = h3, e = h4;
        for (let i = 0; i < 80; i++) {
          let f, k;
          if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; } else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; } else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; } else { f = b ^ c ^ d; k = 0xca62c1d6; }
          const t = (rl(a, 5) + f + e + k + w[i]) >>> 0; e = d; d = c; c = rl(b, 30) >>> 0; b = a; a = t;
        }
        h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
      }
      return [h0, h1, h2, h3, h4].map((x) => x.toString(16).padStart(8, '0')).join('');
    };
    const enc = (s) => new TextEncoder().encode(s);
    const gitBlob = (s) => sha1(enc('blob ' + enc(s).length + '\0' + s));

    /* =====================================================================
       FIGURA 1 — o repositório como kit de montagem
       ===================================================================== */
    (function figRepo() {
      const fig = root.querySelector('#fig-repo');
      const COMPS = [
        { id: 'reg', l: 'Registro de modelos', s: 'architectures → classe' },
        { id: 'cfg', l: 'ModelConfig', s: 'hiperparâmetros, sub-configs' },
        { id: 'quant', l: 'QuantizationConfig', s: 'quant_method, camadas' },
        { id: 'tok', l: 'Tokenizer + chat', s: 'ids, especiais, template' },
        { id: 'samp', l: 'Padrões de amostragem', s: 'temperature, top_p…' },
        { id: 'load', l: 'Carregador de pesos', s: 'índice, mmap, loaders' },
        { id: 'mm', l: 'Processador multimodal', s: 'imagem/vídeo → patches' },
      ];
      const KC = { cfg: 'a', gen: 'd', tok: 'b', q: 'c', idx: 'e', st: 'f', code: 'c' };
      const REPOS = {
        qwen: { name: 'nvidia/Qwen3.8-Flash-Next-NVFP4', rev: 'MODEL_ID padrão da receita Qwen', files: [
          { f: 'config.json', m: '30.820 B', k: 'cfg', to: ['reg', 'cfg', 'quant'], d: 'architectures escolhe a classe; text_config (48 camadas, hidden 2560, vocab 248.320) e vision_config dimensionam os tensores; quantization_config = modelopt, MIXED_PRECISION.' },
          { f: 'generation_config.json', m: 'amostragem', k: 'gen', to: ['samp'], d: 'do_sample true, temperature 1.0, top_p 0.95, top_k 20, eos [248046, 248044]: padrões usados quando o cliente não manda nada.' },
          { f: 'tokenizer.json · tokenizer_config.json', m: 'BPE', k: 'tok', to: ['tok'], d: 'BPE byte-level, tokens especiais (<|im_start|> 248045, <tool_call> 248058, <|image_pad|> 248056), model_max_length 262.144.' },
          { f: 'preprocessor_config.json · video_…', m: 'visão', k: 'tok', to: ['mm'], d: 'Como imagens e vídeos viram patches para a torre de visão (27 blocos ViT).' },
          { f: 'hf_quant_config.json', m: 'legado', k: 'q', to: ['quant'], d: 'Sidecar legado do ModelOpt. Na rev fc694b54 diz FP8_BLOCK_SCALES onde o config.json diz FP8_PB_WO: a receita corrige os dois.' },
          { f: 'model.safetensors.index.json', m: '299.545 tensores', k: 'idx', to: ['load'], d: 'weight_map: cada um dos 299.545 tensores → um dos 11 arquivos; total_size 132.639.846.394 B.' },
          { f: 'model-00001…00010-of-00010.safetensors', m: '10 shards', k: 'st', to: ['load'], d: 'Densos em BF16 e experts NVFP4: U8 empacotado (2 FP4 por byte) + weight_scale + weight_scale_2 + input_scale.' },
          { f: 'model-fp8-mtp-ple.safetensors', m: '53,7 GB', k: 'st', to: ['load'], d: 'A MTP (experts FP8 em bloco 128×128) e a tabela PLE FP8 (128 shards de linhas, 47,68 GiB).' },
        ] },
        glm: { name: 'Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw', rev: '@25a44fdb… (retrato 94ae731; espelho de brandonmusic@5ab363a8…)', files: [
          { f: 'config.json', m: 'exl3', k: 'cfg', to: ['reg', 'cfg', 'quant'], d: 'quantization_config: quant_method exl3, codebook mcg, bits 4, scope glm53_routed_experts_only (+ ledger tensor_storage de 37 MiB, descartado). Nomes de campo no estilo SGLang.' },
          { f: 'generation_config.json', m: 'amostragem', k: 'gen', to: ['samp'], d: 'temperature 1.0, top_p 0.95. Mandar top_p=1.0 explícito sobrescreve o padrão e piora a saída.' },
          { f: 'tokenizer · chat_template (Jinja)', m: 'vocab 154.880', k: 'tok', to: ['tok'], d: 'O template do checkpoint é só texto: a receita passa --chat-template /opt/glm53/chat_template.jinja (imagem, vídeo, raciocínio).' },
          { f: 'model.safetensors.index.json', m: 'weight_map', k: 'idx', to: ['load'], d: 'Romaneio dos 120 shards; a sonda do worker usa os nomes deste índice para conferir tamanhos.' },
          { f: '120 × *.safetensors', m: '~164 GiB', k: 'st', to: ['load'], d: 'Experts roteados em EXL3 (trellis I16, suh, svh, mcg); atenção, KDA, shared, densos, embeddings e lm_head em BF16.' },
        ] },
        ds: { name: 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp', rev: '@86f746b3… (DSPARK_REVISION)', files: [
          { f: 'config.json', m: 'DeepseekV4ForCausalLM', k: 'cfg', to: ['reg', 'cfg', 'quant'], d: 'num_nextn_predict_layers 3, dspark_block_size 5; quantization_config.scale_fmt UE8M0; expert_dtype fp4 escolhe os nomes das escalas no mapper.' },
          { f: 'generation_config.json', m: 'ignorado', k: 'gen', to: ['samp'], d: 'A receita passa --generation-config vllm: o arquivo é ignorado e valem os padrões do vLLM.' },
          { f: 'tokenizer (sem Jinja)', m: 'vocab 129.280', k: 'tok', to: ['tok'], d: 'Não há chat_template Jinja: o formato da conversa vive em código Python.' },
          { f: 'encoding/encoding_dsv4.py', m: 'código', k: 'code', to: ['tok'], d: 'Codificador do chat (papéis, <think>, ferramentas, esforço): o entrypoint copia este arquivo para dentro do vLLM e o --tokenizer-mode deepseek_v4 o chama. Não é remote code; precisa existir no head e no worker.' },
          { f: 'model.safetensors.index.json', m: 'weight_map', k: 'idx', to: ['load'], d: 'O prepare confere que todos os shards citados existem (missing_shards = 0) antes de servir offline.' },
          { f: '*.safetensors', m: '~157 GiB', k: 'st', to: ['load', 'mm'], d: 'Experts MXFP4 (e2m1 + E8M0 por 32), lineares FP8 com escala por bloco 128×128, estágios DSpark em mtp.0..2 e a torre visual.' },
        ] },
      };
      const S = { r: 'qwen', i: 0, auto: true, last: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { S.r = v; S.i = 0; S.last = H.t; } });
      UI.button(ctl, '◀ arquivo', () => { const n = REPOS[S.r].files.length; S.i = (S.i - 1 + n) % n; S.auto = false; tg.set(false); });
      UI.button(ctl, 'arquivo ▶', () => { const n = REPOS[S.r].files.length; S.i = (S.i + 1) % n; S.auto = false; tg.set(false); });
      const tg = UI.toggle(ctl, { label: 'percorrer sozinho', value: true, onChange: (v) => { S.auto = v; S.last = H.t; } });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const R = REPOS[S.r], F = R.files;
        const mobH = 50 + F.length * 26 + 26 + COMPS.length * 26 + 14 + 150;
        stageH(fig, wide ? 430 : mobH);
        if (S.auto && t - S.last > 4.6) { S.i = (S.i + 1) % F.length; S.last = t; }
        if (S.i >= F.length) S.i = 0;
        const cur = F[S.i];
        const kc = (k) => C[KC[k]] || C.a;
        // título
        T(g, clip(g, R.name, w - 28, 12.5, 'mono', 600), 14, 20, { size: 12.5, font: 'mono', weight: 600, color: C.ink });
        T(g, clip(g, R.rev, w - 28, 11, 'mono'), 14, 36, { size: 11, font: 'mono', color: C.muted });
        // geometria
        const top = 50, descH = wide ? 92 : 150;
        let rows, comps;
        if (wide) {
          const fw = Math.min(360, w * 0.5), rh = Math.min(34, (h - top - descH - 12) / F.length);
          rows = F.map((f, i) => ({ x: 14, y: top + i * rh, w: fw, h: rh - 5 }));
          const cw = Math.min(250, w * 0.34), cx = w - cw - 14, ch = (h - top - descH - 12) / COMPS.length;
          comps = COMPS.map((c, i) => ({ x: cx, y: top + i * ch, w: cw, h: ch - 5 }));
        } else {
          // tela estreita: arquivos e componentes em coluna única, com uma "calha" à direita
          // por onde as conexões descem sem passar por cima dos nomes
          const rh = 26, gut = 26;
          rows = F.map((f, i) => ({ x: 14, y: top + i * rh, w: w - 28 - gut, h: rh - 4 }));
          const y0 = top + F.length * rh + 26;
          comps = COMPS.map((c, i) => ({ x: 14, y: y0 + i * 26, w: w - 28 - gut, h: 22 }));
        }
        // clique numa linha
        if (H.pointer.clicked) rows.forEach((r, i) => { if (hit(H.pointer, r.x, r.y, r.w, r.h)) { S.i = i; S.auto = false; tg.set(false); } });
        // conexões + pacotes (desenhados antes das caixas, que têm fundo opaco)
        const targets = cur.to.map((id) => COMPS.findIndex((c) => c.id === id));
        const rr = rows[S.i];
        targets.forEach((ci, k) => {
          const cb = comps[ci];
          if (wide) {
            const x1 = rr.x + rr.w, y1 = rr.y + rr.h / 2, x2 = cb.x, y2 = cb.y + cb.h / 2;
            bezLine(g, x1, y1, x2, y2, { color: V.alpha(kc(cur.k), 0.55), lw: 1.6, dash: [5, 5], dashOffset: -t * 24 }, true);
            for (let p = 0; p < 3; p++) {
              const u = ((t * 0.45 + p / 3 + k * 0.17) % 1);
              const pt = bezPt(x1, y1, x2, y2, V.ease.inOut(u), true);
              V.box(g, pt[0] - 5, pt[1] - 4, 10, 8, { r: 2, fill: kc(cur.k) });
            }
          } else {
            const gx = w - 14 - 9 - k * 6;
            const pts = [[rr.x + rr.w, rr.y + rr.h / 2], [gx, rr.y + rr.h / 2], [gx, cb.y + cb.h / 2], [cb.x + cb.w, cb.y + cb.h / 2]];
            polyLine(g, pts, { color: V.alpha(kc(cur.k), 0.6), lw: 1.6, dash: [5, 5], dashOffset: -t * 24 });
            for (let p = 0; p < 3; p++) {
              const u = ((t * 0.3 + p / 3 + k * 0.17) % 1);
              const pt = polyPt(pts, V.ease.inOut(u));
              V.box(g, pt[0] - 4, pt[1] - 4, 8, 8, { r: 2, fill: kc(cur.k) });
            }
          }
        });
        // linhas de arquivo
        rows.forEach((r, i) => {
          const f = F[i], on = i === S.i, col = kc(f.k);
          solidBox(g, r.x, r.y, r.w, r.h, { r: 8, fill: on ? V.alpha(col, 0.22) : V.alpha(C.muted, 0.06), stroke: on ? col : V.alpha(C.muted, 0.22), lw: on ? 1.8 : 1 });
          const icon = f.k === 'st' ? '▦' : f.k === 'idx' ? '☰' : f.k === 'code' ? '⌘' : '{ }';
          T(g, icon, r.x + 9, r.y + r.h / 2 + 0.5, { size: 11, font: 'mono', color: col, base: 'middle', weight: 700 });
          const mW = tw(g, f.m, 10.5, 'mono');
          const nameW = r.w - 38 - (r.w > 300 ? mW + 16 : 0);
          T(g, clip(g, f.f, nameW, 11.5, 'mono', on ? 600 : 400), r.x + 32, r.y + r.h / 2 + 0.5, { size: 11.5, font: 'mono', weight: on ? 600 : 400, color: on ? C.ink : C.text2 || C.ink, base: 'middle' });
          if (r.w > 300) T(g, f.m, r.x + r.w - 10, r.y + r.h / 2 + 0.5, { size: 10.5, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
        });
        // componentes
        if (!wide) T(g, 'componentes do vLLM que leem o arquivo', 14, comps[0].y - 9, { size: 10.5, font: 'mono', color: C.muted });
        comps.forEach((cb, i) => {
          const on = targets.includes(i), c = COMPS[i];
          solidBox(g, cb.x, cb.y, cb.w, cb.h, { r: 9, fill: on ? V.alpha(kc(cur.k), 0.2) : V.alpha(C.muted, 0.05), stroke: on ? kc(cur.k) : V.alpha(C.muted, 0.25), lw: on ? 1.8 : 1 });
          if (on) V.glow(g, cb.x + 10, cb.y + cb.h / 2, 12, kc(cur.k), 0.6 + 0.4 * Math.sin(t * 4));
          if (wide && cb.h > 30) {
            T(g, c.l, cb.x + 12, cb.y + cb.h / 2 - 3, { size: 12, weight: on ? 600 : 500, color: on ? C.ink : C.muted });
            T(g, clip(g, c.s, cb.w - 20, 10.5, 'mono'), cb.x + 12, cb.y + cb.h / 2 + 12, { size: 10.5, font: 'mono', color: C.muted });
          } else {
            const lw1 = Math.min(tw(g, c.l, 11, 'sans', on ? 600 : 500), cb.w * 0.55);
            T(g, clip(g, c.l, cb.w * 0.55, 11, 'sans', on ? 600 : 500), cb.x + 8, cb.y + cb.h / 2 + 0.5, { size: 11, weight: on ? 600 : 500, color: on ? C.ink : C.muted, base: 'middle' });
            const sub = clip(g, c.s, cb.w - lw1 - 30, 10, 'mono');
            if (sub) T(g, sub, cb.x + cb.w - 8, cb.y + cb.h / 2 + 0.5, { size: 10, font: 'mono', color: C.muted, align: 'right', base: 'middle' });
          }
        });
        // descrição
        const dy = h - descH;
        V.box(g, 14, dy, w - 28, descH - 10, { r: 10, fill: V.alpha(kc(cur.k), 0.1), stroke: V.alpha(kc(cur.k), 0.5) });
        T(g, clip(g, cur.f, w - 60, 12, 'mono', 700), 26, dy + 20, { size: 12, font: 'mono', weight: 700, color: C.ink });
        wrap(g, cur.d, 26, dy + 40, w - 56, 16, { size: 12.5, color: C.ink, maxLines: wide ? 3 : 6 });
      });
    })();

    /* =====================================================================
       FIGURA 2 — dentro de um arquivo safetensors
       ===================================================================== */
    (function figST() {
      const fig = root.querySelector('#fig-st');
      const DTB = { BF16: 2, F32: 4, F8_E4M3: 1, U8: 1 };
      const TS = [
        { n: 'model.language_model.layers.3.self_attn.q_proj.weight', s: 'q_proj', dt: 'BF16', sh: [12288, 2560] },
        { n: 'model.language_model.layers.3.self_attn.k_proj.weight', s: 'k_proj', dt: 'BF16', sh: [512, 2560] },
        { n: 'model.language_model.layers.5.mlp.experts.0.gate_proj.weight', s: 'expert (U8)', dt: 'U8', sh: [640, 1280] },
        { n: 'model.language_model.layers.5.mlp.experts.0.gate_proj.weight_scale', s: 'weight_scale', dt: 'F8_E4M3', sh: [640, 160] },
        { n: 'model.language_model.layers.5.mlp.experts.0.gate_proj.weight_scale_2', s: 'weight_scale_2', dt: 'F32', sh: [] },
      ];
      const hdr = { __metadata__: { format: 'pt' } };
      let off = 0;
      TS.forEach((t) => { t.nb = DTB[t.dt] * t.sh.reduce((a, b) => a * b, 1); t.b = off; t.e = off + t.nb; off += t.nb; hdr[t.n] = { dtype: t.dt, shape: t.sh, data_offsets: [t.b, t.e] }; });
      const js = JSON.stringify(hdr);
      const N = js.length + ((8 - (js.length % 8)) % 8);
      const Nbytes = Array.from({ length: 8 }, (_, i) => Math.floor(N / Math.pow(256, i)) % 256);
      const hex = (b) => b.toString(16).padStart(2, '0');
      const fmtN = (x) => x.toLocaleString('pt-BR');
      const COLS = ['a', 'b', 'e', 'd', 'f'];
      const S = { sel: 0 };
      const steps = [
        { t: 'Ler os 8 primeiros bytes', d: 'O arquivo começa com um inteiro de 64 bits <i>little-endian</i>: o byte menos significativo vem primeiro. Ele diz o tamanho \\(N\\) do cabeçalho JSON.' },
        { t: 'Ler N bytes de JSON', d: 'O cabeçalho lista cada tensor com <code>dtype</code>, <code>shape</code> e <code>data_offsets</code>, mais o <code>__metadata__</code>. Nada de código: só texto.' },
        { t: 'Localizar o tensor', d: 'Os offsets contam a partir do início do blob, então a faixa absoluta no arquivo é \\([8 + N + b,\\ 8 + N + e)\\).' },
        { t: 'Reinterpretar os bytes', d: 'Os bytes viram uma matriz em ordem <i>row-major</i>. Não existe dtype FP4: o expert NVFP4 é U8 com dois valores por byte.' },
        { t: 'mmap: só o que é tocado', d: 'Com o arquivo mapeado na memória virtual, o SO só lê do disco as páginas que o carregador toca. O tensor é uma vista sobre elas (zero-copy).' },
        { t: 'Range read pela rede', d: 'Pelo HTTP dá para pedir só a faixa desejada (<code>Range: bytes=a-b</code>): primeiro 8 bytes, depois o cabeçalho, depois o tensor. É o truque do <code>dense_overlay.py</code>.' },
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Tensor', options: TS.map((t, i) => ({ v: i, label: t.s })), value: 0, onChange: (v) => { S.sel = v; } });
      const rd = UI.readout(ctl, 'bytes do tensor', '');
      const st = UI.stepper(fig, steps, { autoplay: 6500 });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        stageH(fig, wide ? 450 : 560);
        const sel = TS[S.sel], i = st.i, el = st.elapsed();
        if (S.shown !== S.sel) { rd.set(fmtN(sel.nb) + ' B'); S.shown = S.sel; }
        const col = (k) => C[COLS[k]];
        // fita de bytes
        const rx = 14, rw = w - 28, ry = 26, rh = 40;
        const wts = TS.map((q) => Math.max(0.02, Math.pow(q.nb, 0.25)));
        const hdrW = rw * 0.2, lenW = 40, avail = rw - hdrW - lenW - 6, sum = wts.reduce((a, b) => a + b, 0);
        const segs = [{ x: rx, w: lenW, c: C.c, l: '8 B' }, { x: rx + lenW + 2, w: hdrW, c: C.muted, l: 'JSON (' + N + ' B)' }];
        let xx = rx + lenW + hdrW + 4;
        const raw = wts.map((v) => Math.max(10, (v / sum) * avail)), rs = raw.reduce((a, b) => a + b, 0);
        TS.forEach((q, k) => { const ww = raw[k] * Math.min(1, avail / rs); segs.push({ x: xx, w: ww, c: col(k), l: q.s, k }); xx += ww; });
        T(g, 'o arquivo (fita de bytes, fora de escala)', rx, 16, { size: 10.5, font: 'mono', color: C.muted });
        segs.forEach((sg, k) => {
          const isSel = sg.k === S.sel;
          const hl = (i === 0 && k === 0) || (i === 1 && k === 1) || (i >= 2 && isSel);
          V.box(g, sg.x, ry, sg.w - 2, rh, { r: 5, fill: V.alpha(sg.c, hl ? 0.55 : 0.18), stroke: hl ? sg.c : V.alpha(sg.c, 0.45), lw: hl ? 2 : 1 });
          const lbl = clip(g, sg.l, sg.w - 8, 10, 'mono');
          if (lbl) T(g, lbl, sg.x + (sg.w - 2) / 2, ry + rh / 2 + 0.5, { size: 10, font: 'mono', align: 'center', base: 'middle', color: C.ink });
        });
        // marcas
        const selSeg = segs[2 + S.sel];
        T(g, '0', rx, ry + rh + 13, { size: 10, font: 'mono', color: C.muted });
        T(g, '8', rx + lenW, ry + rh + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' });
        if (wide) T(g, '8+N = ' + (8 + N), rx + lenW + hdrW, ry + rh + 13, { size: 10, font: 'mono', color: C.muted, align: 'center' });
        // páginas (mmap) — passo 4
        if (i === 4) {
          const py = ry + rh + 20, np = Math.floor(rw / 9);
          for (let p = 0; p < np; p++) {
            const px = rx + p * 9, inSel = px + 7 >= selSeg.x && px <= selSeg.x + selSeg.w - 2;
            const sweep = ((el * 0.6) % 1.2) * rw;
            const lit = inSel && px - rx <= Math.max(sweep, 0) + (el > 2 ? rw : 0);
            V.box(g, px, py, 7, 7, { r: 1.5, fill: lit ? col(S.sel) : V.alpha(C.muted, 0.18) });
          }
        }
        // painéis
        const py0 = ry + rh + (i === 4 ? 40 : 28);
        const jx = 14, jw = wide ? w * 0.54 - 20 : w - 28;
        const dx = wide ? w * 0.54 + 4 : 14, dw = wide ? w - dx - 14 : w - 28;
        const jh = wide ? h - py0 - 10 : 190;
        const dy = wide ? py0 : py0 + jh + 10, dh = wide ? h - py0 - 10 : h - dy - 10;
        // painel JSON
        V.box(g, jx, py0, jw, jh, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.25) });
        T(g, 'cabeçalho JSON', jx + 12, py0 + 18, { size: 10.5, font: 'mono', color: C.muted });
        const lines = [{ s: '"__metadata__": {"format": "pt"}', k: -1 }].concat(TS.map((q, k) => ({ s: q.n, k, meta: q.dt + ' [' + q.sh.join(', ') + ']  [' + q.b + ', ' + q.e + ')' })));
        const shown = i === 1 ? Math.min(lines.length, Math.floor(el * 2.2) + 1) : i === 0 ? 0 : lines.length;
        const lh = Math.min(30, (jh - 34) / lines.length);
        lines.forEach((L, k) => {
          if (k >= shown) return;
          const ly = py0 + 30 + k * lh;
          const on = L.k === S.sel && i >= 2;
          if (on) V.box(g, jx + 6, ly - 2, jw - 12, lh - 2, { r: 6, fill: V.alpha(col(L.k), 0.2), stroke: col(L.k) });
          const c0 = L.k >= 0 ? col(L.k) : C.muted;
          if (L.k < 0) { T(g, clip(g, L.s, jw - 24, 10.5, 'mono'), jx + 12, ly + lh / 2, { size: 10.5, font: 'mono', color: C.muted, base: 'middle' }); return; }
          const two = lh >= 24;
          T(g, clipL(g, '"' + L.s + '"', jw - 24, 10.5, 'mono', on ? 600 : 400), jx + 12, ly + (two ? lh * 0.32 : lh / 2), { size: 10.5, font: 'mono', weight: on ? 600 : 400, color: on ? C.ink : c0, base: 'middle' });
          if (two) T(g, clip(g, L.meta, jw - 30, 10, 'mono'), jx + 20, ly + lh * 0.72, { size: 10, font: 'mono', color: C.muted, base: 'middle' });
        });
        if (i === 1) { const cu = py0 + 30 + (shown - 1) * lh; if (shown < lines.length && Math.sin(t * 8) > 0) V.box(g, jx + 12, cu + lh - 8, 8, 2, { r: 1, fill: C.ink }); }
        if (i === 0) {
          T(g, '(ainda não lido: primeiro precisamos de N)', jx + 12, py0 + 44, { size: 11, color: C.muted });
          for (let k = 0; k < 6; k++) { const ww = (jw - 40) * (0.55 + 0.4 * ((k * 37) % 10) / 10); V.box(g, jx + 12, py0 + 64 + k * 26, ww, 10, { r: 5, fill: V.alpha(C.muted, 0.08 + 0.05 * Math.sin(t * 2 + k)) }); }
        }
        // painel de detalhe
        V.box(g, dx, dy, dw, dh, { r: 10, fill: V.alpha(col(S.sel), 0.06), stroke: V.alpha(col(S.sel), 0.35) });
        const cx = dx + 12, cy = dy + 20;
        if (i === 0) {
          T(g, 'bytes 0–7 (hex)', cx, cy, { size: 10.5, font: 'mono', color: C.muted });
          const bw = Math.min(34, (dw - 24 - 7 * 4) / 8);
          Nbytes.forEach((b, k) => {
            const bx = cx + k * (bw + 4), fly = V.clamp(el * 1.4 - k * 0.12, 0, 1);
            V.box(g, bx, cy + 10 + (1 - V.ease.out(fly)) * -20, bw, 28, { r: 5, fill: V.alpha(C.c, k < 2 ? 0.45 : 0.15), stroke: C.c });
            T(g, hex(b), bx + bw / 2, cy + 24 + (1 - V.ease.out(fly)) * -20, { size: 11.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 600 });
          });
          T(g, 'little-endian: o 1º byte é o menos significativo', cx, cy + 62, { size: 11, color: C.muted });
          T(g, '0x' + hex(Nbytes[1]) + hex(Nbytes[0]) + ' = ' + N, cx, cy + 90, { size: 18, font: 'mono', weight: 700, color: C.ink });
          T(g, 'N = tamanho do cabeçalho JSON (com espaços até múltiplo de 8)', cx, cy + 112, { size: 11, color: C.muted, maxW: dw - 24 });
          T(g, 'blob de dados começa no byte 8 + N = ' + (8 + N), cx, cy + 136, { size: 12, font: 'mono', color: C.c, weight: 600 });
        } else if (i === 1) {
          T(g, 'por tensor, o cabeçalho diz:', cx, cy, { size: 11, color: C.muted });
          [['dtype', 'formato do elemento (BF16, U8, F8_E4M3…)'], ['shape', 'dimensões, ex.: [12288, 2560]'], ['data_offsets', '[b, e) relativos ao início do blob'], ['__metadata__', 'strings livres (ex.: format: pt)']].forEach(([k, v], j) => {
            chip(g, cx, cy + 12 + j * 48, k, C.a, { size: 11 });
            T(g, clip(g, v, dw - 30, 11.5, 'sans'), cx + 4, cy + 48 + j * 48, { size: 11.5, color: C.ink });
          });
          T(g, JSON.stringify(hdr).length + ' caracteres de JSON + ' + (N - js.length) + ' espaços = ' + N + ' B', cx, dy + dh - 14, { size: 10.5, font: 'mono', color: C.muted, maxW: dw - 24 });
        } else if (i === 2) {
          T(g, clip(g, sel.s + '  ·  ' + sel.dt + ' [' + sel.sh.join(', ') + ']', dw - 24, 12, 'mono', 700), cx, cy, { size: 12, font: 'mono', weight: 700, color: C.ink });
          const rows = [['b (data_offsets[0])', fmtN(sel.b)], ['e (data_offsets[1])', fmtN(sel.e)], ['e − b', fmtN(sel.nb) + ' B'], ['início no arquivo = 8+N+b', fmtN(8 + N + sel.b)], ['fim no arquivo = 8+N+e', fmtN(8 + N + sel.e)]];
          rows.forEach(([k, v], j) => {
            const yy = cy + 26 + j * 26, a = V.clamp(el * 1.6 - j * 0.35, 0, 1);
            T(g, k, cx, yy, { size: 11, font: 'mono', color: C.muted, alpha: a });
            T(g, v, dx + dw - 14, yy, { size: 12, font: 'mono', weight: 600, color: j >= 3 ? col(S.sel) : C.ink, align: 'right', alpha: a });
          });
          const prod = sel.sh.length ? sel.sh.join(' × ') : '1 (escalar)';
          T(g, clip(g, DTB[sel.dt] + ' B × ' + prod + ' = ' + fmtN(sel.nb) + ' B', dw - 24, 11, 'mono'), cx, dy + dh - 14, { size: 11, font: 'mono', color: C.muted });
          // seta do cabeçalho para a fita
          const u = V.ease.inOut(V.clamp(el / 1.2, 0, 1));
          const ax = selSeg.x + selSeg.w / 2, ay = ry + rh + 2;
          const lyy = py0 + 30 + (S.sel + 1) * lh + lh / 2;
          const sx = jx + jw - 10;
          if (wide) { const p = bezPt(sx, lyy, ax, ay, u, false); V.line(g, sx, lyy, p[0], p[1], { color: col(S.sel), lw: 1.6, dash: [4, 4] }); V.dot(g, p[0], p[1], 4, col(S.sel)); }
        } else if (i === 3) {
          const r0 = sel.sh[0] || 1, c0 = sel.sh[1] || 1;
          T(g, clip(g, sel.dt + ' [' + (sel.sh.length ? sel.sh.join(' × ') : 'escalar') + ']', dw - 24, 12, 'mono', 700), cx, cy, { size: 12, font: 'mono', weight: 700, color: C.ink });
          const R = sel.sh.length ? 8 : 1, K = sel.sh.length ? 12 : 1;
          const cs = Math.min(20, (dw - 30) / K - 2);
          const u = V.ease.out(V.clamp(el / 1.6, 0, 1));
          for (let a = 0; a < R; a++) for (let b = 0; b < K; b++) {
            const idx = a * K + b, lin = idx / (R * K);
            const fx = cx + lin * (dw - 30), fy = cy + 16;
            const tx = cx + b * (cs + 2), ty = cy + 18 + a * (cs + 2);
            const x = V.lerp(fx, tx, u), y = V.lerp(fy, ty, u);
            if (sel.dt === 'U8') {
              V.box(g, x, y, cs / 2 - 0.5, cs, { r: 2, fill: V.alpha(col(S.sel), 0.35 + 0.4 * ((a * 7 + b * 3) % 5) / 5) });
              V.box(g, x + cs / 2 + 0.5, y, cs / 2 - 0.5, cs, { r: 2, fill: V.alpha(C.b, 0.35 + 0.4 * ((a * 3 + b * 5) % 5) / 5) });
            } else V.box(g, x, y, cs, cs, { r: 2, fill: V.alpha(col(S.sel), 0.3 + 0.5 * ((a * 7 + b * 3) % 5) / 5) });
          }
          const gy = cy + 34 + R * (cs + 2);
          T(g, sel.sh.length ? 'miniatura: ' + r0.toLocaleString('pt-BR') + ' linhas × ' + c0.toLocaleString('pt-BR') + ' colunas' : 'um único valor F32', cx, gy + 4, { size: 11, color: C.muted });
          T(g, clip(g, '(i, j) → byte 8+N+b+' + DTB[sel.dt] + '·(i·' + c0 + '+j)', dw - 24, 11, 'mono'), cx, gy + 22, { size: 11, font: 'mono', color: C.ink });
          if (sel.dt === 'U8') T(g, clip(g, '2 valores FP4 por byte → matriz lógica 640 × 2560', dw - 24, 11, 'sans', 600), cx, gy + 40, { size: 11, weight: 600, color: C.b });
          if (sel.dt === 'F8_E4M3') T(g, clip(g, 'uma escala E4M3 por bloco de 16 valores (2560 / 16 = 160)', dw - 24, 11, 'sans'), cx, gy + 40, { size: 11, color: C.d });
        } else if (i === 4) {
          T(g, 'mmap + page faults', cx, cy, { size: 12, weight: 700, color: C.ink });
          const yN = wrap(g, 'Cada quadradinho acima é uma página. Só as páginas do tensor selecionado acendem: é o que o SO lê do disco quando o carregador toca esse tensor. As outras continuam no disco.', cx, cy + 22, dw - 24, 16, { size: 11.5, color: C.ink, maxLines: 5 });
          const pw2 = 12, nP = Math.max(4, Math.floor((dw - 40) / (pw2 + 4)));
          T(g, 'disco', cx, yN + 14, { size: 10.5, font: 'mono', color: C.muted }); T(g, 'RAM (page cache)', cx, yN + 72, { size: 10.5, font: 'mono', color: C.muted });
          for (let p = 0; p < nP; p++) { const x = cx + p * (pw2 + 4), touched = p >= Math.floor(nP * 0.35) && p < Math.floor(nP * 0.6); const u = touched ? V.ease.inOut(V.clamp((el - 0.5 - (p - nP * 0.35) * 0.15) / 0.8, 0, 1)) : 0; V.box(g, x, yN + 22, pw2, pw2, { r: 2, fill: V.alpha(C.muted, 0.25) }); if (touched) V.box(g, x, V.lerp(yN + 22, yN + 80, u), pw2, pw2, { r: 2, fill: col(S.sel) }); }
          T(g, 'lazy: cada rank lê só a sua fatia', cx, dy + dh - 34, { size: 11, font: 'mono', color: C.a });
          T(g, 'Qwen: 458 s de carga (392 + 67)', cx, dy + dh - 16, { size: 11, font: 'mono', color: C.muted });
        } else {
          const hx = dx + dw - 70, hy = cy + 10;
          V.box(g, hx, hy, 58, 34, { r: 17, fill: V.alpha(C.b, 0.2), stroke: C.b });
          T(g, 'Hub', hx + 29, hy + 17, { size: 12, weight: 700, align: 'center', base: 'middle', color: C.ink });
          const reqs = [['bytes=0-7', 'N'], ['bytes=8-' + (8 + N - 1), 'cabeçalho'], ['bytes=' + (8 + N + sel.b) + '-' + (8 + N + sel.e - 1), sel.s]];
          reqs.forEach(([r, l], j) => {
            const a = V.clamp(el * 1.1 - j * 0.9, 0, 1);
            const yy = cy + 12 + j * 34;
            T(g, clip(g, 'Range: ' + r, dw - 110, 10.5, 'mono'), cx, yy + 8, { size: 10.5, font: 'mono', color: C.ink, alpha: Math.max(0.25, a) });
            T(g, '→ ' + l, cx, yy + 22, { size: 10.5, font: 'mono', color: j === 2 ? col(S.sel) : C.muted, alpha: Math.max(0.25, a) });
            if (a > 0 && a < 1) { const p = V.along(hx, hy + 17, cx + 150, yy + 12, a); V.dot(g, p[0], p[1], 3.5, j === 2 ? col(S.sel) : C.b); }
          });
          const got = 8 + N + sel.nb, tot = 8 + N + off;
          const bby = cy + 128; T(g, 'baixado: ' + fmtN(got) + ' de ' + fmtN(tot) + ' B (' + nBR(100 * got / tot, got / tot < 0.01 ? 3 : 1) + '%)', cx, bby, { size: 11, font: 'mono', color: C.ink, maxW: dw - 24 });
          V.box(g, cx, bby + 8, dw - 24, 10, { r: 5, fill: V.alpha(C.muted, 0.2) }); V.box(g, cx, bby + 8, Math.max(3, (dw - 24) * got / tot), 10, { r: 5, fill: col(S.sel) });
          wrap(g, 'dense_overlay.py baixa ~5,3 GB de tensores de um checkpoint remoto sem baixar o resto.', cx, dy + dh - 34, dw - 24, 15, { size: 11, color: C.muted, maxLines: 2 });
        }
      });
    })();

    /* =====================================================================
       FIGURA 3 — a esteira do carregador (DeepSeek)
       ===================================================================== */
    (function figLoader() {
      const fig = root.querySelector('#fig-loader');
      const S = { par: 'tp2', fmt: 'bf16', tc: -99 };
      const rng = V.rng(24);
      const W1 = Array.from({ length: 4 }, () => Array.from({ length: 6 }, () => +(rng.normal() * 0.6).toFixed(1)));
      const W3 = Array.from({ length: 4 }, () => Array.from({ length: 6 }, () => +(rng.normal() * 0.6).toFixed(1)));
      const steps = [
        { t: 'Ler pares (nome, tensor)', d: '<code>load_weights</code> recebe um fluxo de pares vindos dos shards, na ordem em que estão no arquivo. O alvo pula tudo que começa com <code>mtp.</code>: esses pesos são do drafter.' },
        { t: 'Renomear (WeightsMapper)', d: 'Prefixos, sufixos, substrings e regex traduzem nomes do checkpoint DeepSeek para nomes de módulo do vLLM. As escalas <code>.scale</code> viram <code>.weight_scale</code> (experts FP4) ou <code>.weight_scale_inv</code> (FP8).' },
        { t: 'Empilhar (stacked params)', d: '<code>w1</code> e <code>w3</code> (gate e up) vão para gavetas 0 e 1 de um único <code>gate_up_proj</code>: um GEMM em vez de dois. Atenção: <code>"w1" in "markov_w1"</code> é verdadeiro, por isso o DSpark compara segmentos.' },
        { t: 'Fatiar por rank (weight_loader)', d: 'W é guardado como [saída, entrada]. Em <i>column-parallel</i> cada rank recebe a sua fatia de <b>linhas</b> (saídas) de <b>cada</b> gaveta; em <i>row-parallel</i> (o <code>down_proj</code>) recebe uma fatia de <b>colunas</b> (entradas), e um all-reduce soma as parciais. Troque o <b>Formato</b> para ver pesos quantizados cortados junto com as escalas.' },
        { t: 'Experts: TP × EP, e escalas E8M0', d: 'Em TP puro (o DeepSeek roda TP=2 sem EP), cada expert é fatiado como um linear: <code>w1</code>/<code>w3</code> pelas linhas, <code>w2</code> pelas colunas, e os experts existem nos dois ranks. Com <b>EP</b>, um <code>expert_map</code> leva experts globais a índices locais e cada rank guarda experts inteiros (Qwen, GLM TP3). As escalas E8M0 chegam como <code>float8_e8m0fnu</code> e são reinterpretadas com <code>view(uint8)</code>: converter faria \\(2^{-7}\\) virar 0.' },
        { t: 'Pós-carga', d: '<code>process_weights_after_loading</code> reempacota para o layout do kernel (ex.: o do Marlin, um GEMM que desquantiza pesos de 4 ou 8 bits em registradores e exige os pesos reordenados em tiles), pré-calcula tabelas e, às vezes, desquantiza (o GLM passa projeções MLA FP8 para BF16). Só então o parâmetro está pronto.' },
      ];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Paralelismo', options: [{ v: 'tp1', label: 'TP=1' }, { v: 'tp2', label: 'TP=2' }, { v: 'ep2', label: 'EP=2' }], value: 'tp2', onChange: (v) => { S.par = v; S.tc = H.t; } });
      UI.seg(ctl, { label: 'Formato (passo 4)', options: [{ v: 'bf16', label: 'BF16' }, { v: 'nvfp4', label: 'NVFP4' }, { v: 'fp8b', label: 'FP8 bloco' }, { v: 'exl3', label: 'EXL3' }], value: 'bf16', onChange: (v) => { S.fmt = v; S.tc = H.t; if (st.i !== 3) { st.stop(); st.go(3); } } });
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const NAMES = ['layers.7.ffn.shared_experts.w1.weight', 'layers.7.ffn.shared_experts.w3.weight', 'layers.7.attn.wq_a.weight', 'layers.7.attn.wkv.weight', 'layers.7.ffn.gate.bias', 'head.weight', 'mtp.0.attn.wq_a.weight'];
      const REN = [
        ['head.weight', 'lm_head.weight', 'sufixo'],
        ['layers.7.ffn.gate.bias', 'model.layers.7.ffn.gate.e_score_correction_bias', 'pref.+sufixo'],
        ['layers.7.attn.compressor.wkv.weight', 'model.layers.7.attn.mla_attn.compressor.wkv.weight', 'substring'],
        ['layers.7.ffn.shared_experts.w2.weight', 'model.layers.7.ffn.shared_experts.down_proj.weight', 'substring'],
        ['layers.7.ffn.experts.3.w1.scale', 'model.layers.7.ffn.experts.3.w1.weight_scale', 'regex (FP4)'],
        ['layers.7.attn.wq_a.scale', 'model.layers.7.attn.wq_a.weight_scale_inv', 'regex (FP8)'],
        ['mtp.0.attn.wq_a.weight', '(pulado pelo alvo → drafter)', 'skip mtp.'],
      ];
      const STN = ['ler', 'renomear', 'empilhar', 'fatiar', 'experts', 'pós-carga'];
      // pesos quantizados: tensores reais (formas dos inventários) e como cada um é cortado em TP=2.
      // cut: dimensão cortada (0 = linhas, 1 = colunas; em 1D, 0 = a única dimensão); null = replicado.
      const QF = {
        nvfp4: {
          head: 'NVFP4 · expert do Qwen (W4A4, bloco de 16)', ph: [290, 270],
          col: { title: 'gate_proj · column-parallel', ok: true, v: 'códigos e escalas de bloco saem nas mesmas linhas; escalares replicados', ts: [
            { n: 'weight', dt: 'U8', sh: [640, 1280], cut: 0, loc: '[320, 1280]', grid: [8, 8] },
            { n: 'weight_scale', dt: 'F8_E4M3', sh: [640, 160], cut: 0, loc: '[320, 160]', grid: [8, 4] },
            { n: 'weight_scale_2', dt: 'F32', sh: [], cut: null },
            { n: 'input_scale', dt: 'F32', sh: [], cut: null }] },
          row: { title: 'down_proj · row-parallel', ok: true, v: '320 valores = 160 B = 20 blocos de 16 por rank', ts: [
            { n: 'weight', dt: 'U8', sh: [2560, 320], cut: 1, loc: '[2560, 160]', grid: [8, 4] },
            { n: 'weight_scale', dt: 'F8_E4M3', sh: [2560, 40], cut: 1, loc: '[2560, 20]', grid: [8, 2] },
            { n: 'weight_scale_2', dt: 'F32', sh: [], cut: null },
            { n: 'input_scale', dt: 'F32', sh: [], cut: null }] },
          note: 'Conta de TP puro. Na receita Qwen os experts NVFP4 usam EP (256 de 512 por GPU): nenhum expert é cortado.',
        },
        fp8b: {
          head: 'FP8 em blocos 128×128 · expert da MTP do Qwen', ph: [254, 232],
          col: { title: 'gate_proj · column-parallel', ok: false, v: '320 linhas = 2,5 blocos: a escala S[2,·] ficaria partida', ts: [
            { n: 'weight', dt: 'F8_E4M3', sh: [640, 2560], cut: 0, loc: '[320, 2560]', grid: [5, 20], bad: 2 },
            { n: 'weight_scale_inv', dt: 'BF16', sh: [5, 20], cut: 0, loc: '[2,5 × 20] ?', grid: [5, 20], bad: 2 }] },
          row: { title: 'down_proj · row-parallel', ok: false, v: '320 colunas = 2,5 blocos de 128: o mesmo problema', ts: [
            { n: 'weight', dt: 'F8_E4M3', sh: [2560, 640], cut: 1, loc: '[2560, 320]', grid: [20, 5], bad: 2 },
            { n: 'weight_scale_inv', dt: 'BF16', sh: [20, 5], cut: 1, loc: '[20 × 2,5] ?', grid: [20, 5], bad: 2 }] },
          note: 'A largura local precisa ser múltipla de 128. O shared expert do Qwen tem as mesmas 640 linhas e é fatiado por TP de verdade (320 por rank): um dos motivos de o FP8-dense usar escala por canal.',
        },
        exl3: {
          head: 'EXL3 · expert do GLM (treliça 4 bits, tiles 16×16)', ph: [320, 302],
          col: { title: 'gate_proj · column-parallel', ok: true, v: 'saída = dim 1 da treliça + svh; suh (entrada) replicado', ts: [
            { n: 'trellis', dt: 'I16', sh: [256, 128, 64], cut: 1, loc: '[256, 64, 64]', grid: [8, 8], tr: true },
            { n: 'svh', dt: 'F16', sh: [2048], cut: 0, loc: '[1024]' },
            { n: 'suh', dt: 'F16', sh: [4096], cut: null },
            { n: 'mcg', dt: 'I32', sh: [1], cut: null }] },
          row: { title: 'down_proj · row-parallel', ok: true, v: 'entrada = dim 0 da treliça + suh; svh (saída) replicado', ts: [
            { n: 'trellis', dt: 'I16', sh: [128, 256, 64], cut: 0, loc: '[64, 256, 64]', grid: [8, 8], tr: true },
            { n: 'suh', dt: 'F16', sh: [2048], cut: 0, loc: '[1024]' },
            { n: 'svh', dt: 'F16', sh: [4096], cut: null },
            { n: 'mcg', dt: 'I32', sh: [1], cut: null }] },
          note: 'A treliça é guardada como [in/16, out/16, 16·bits], transposta em relação a W. GLM TP2: 1024 por rank ✓. No TP3, 2048/3 não divide e a receita usa EP (96 de 288 experts).',
        },
      };
      const fit = (r, maxW, maxH, minW, minH) => { let ww = maxW, hh = ww * r; if (hh > maxH) { hh = maxH; ww = hh / r; } return [Math.max(minW, ww), Math.max(minH, hh)]; };
      // retângulo de um tensor, cortado ao meio (cut 0 = linhas, 1 = colunas) com animação u
      const drawCutRect = (g, x, y, w, h, o) => {
        const C = V.C();
        const cut = o.cut, u = o.u || 0, gap = 10 * u;
        const [nbR, nbC] = o.grid || [1, 1];
        if (o.depth) for (let d = 2; d >= 1; d--) V.box(g, x + d * 4, y - d * 4, w, h, { r: 3, stroke: V.alpha(C.muted, 0.35), lw: 1 });
        const halves = cut == null ? [{ dx: 0, dy: 0, hx: x, hy: y, hw: w, hh: h, col: o.whole || C.b }]
          : cut === 0 ? [{ dx: 0, dy: -gap / 2, hx: x, hy: y, hw: w, hh: h / 2, col: C.e }, { dx: 0, dy: gap / 2, hx: x, hy: y + h / 2, hw: w, hh: h / 2, col: C.d }]
            : [{ dx: -gap / 2, dy: 0, hx: x, hy: y, hw: w / 2, hh: h, col: C.e }, { dx: gap / 2, dy: 0, hx: x + w / 2, hy: y, hw: w / 2, hh: h, col: C.d }];
        halves.forEach((hf) => {
          const col = cut == null ? hf.col : V.mix(C.b, hf.col, u);
          g.save(); g.translate(hf.dx, hf.dy);
          g.beginPath(); g.rect(hf.hx, hf.hy, hf.hw, hf.hh); g.clip();
          V.box(g, x, y, w, h, { r: 3, fill: V.alpha(col, 0.3) });
          g.beginPath();
          for (let a = 1; a < nbR; a++) { const yy = y + (a * h) / nbR; g.moveTo(x, yy); g.lineTo(x + w, yy); }
          for (let b = 1; b < nbC; b++) { const xx = x + (b * w) / nbC; g.moveTo(xx, y); g.lineTo(xx, y + h); }
          g.strokeStyle = V.alpha(col, 0.6); g.lineWidth = 1; g.stroke();
          if (o.bad != null && cut != null && u > 0.5) {
            const a = V.clamp((u - 0.5) * 2, 0, 1) * (0.6 + 0.2 * Math.sin((S.now || 0) * 5));
            if (cut === 0) V.box(g, x, y + (o.bad * h) / nbR, w, h / nbR, { r: 0, fill: V.alpha(badCol(C), a) });
            else V.box(g, x + (o.bad * w) / nbC, y, w / nbC, h, { r: 0, fill: V.alpha(badCol(C), a) });
          }
          g.restore();
          g.save(); g.translate(hf.dx, hf.dy);
          V.box(g, hf.hx, hf.hy, hf.hw, hf.hh, { r: 3, stroke: V.alpha(col, 0.9), lw: 1.2 });
          g.restore();
        });
        if (o.bad != null && cut != null && u > 0.5) {
          const bc = badCol(C);
          g.setLineDash([3, 2]);
          if (cut === 0) V.box(g, x - 2, y + (o.bad * h) / nbR - gap / 2 - 1, w + 4, h / nbR + gap + 2, { r: 2, stroke: bc, lw: 1.6 });
          else V.box(g, x + (o.bad * w) / nbC - gap / 2 - 1, y - 2, w / nbC + gap + 2, h + 4, { r: 2, stroke: bc, lw: 1.6 });
          g.setLineDash([]);
        }
        // tesoura: linha tracejada varrendo o corte
        if (cut != null && u > 0.02 && u < 0.97) {
          const sweep = V.clamp(u * 1.6, 0, 1);
          if (cut === 0) { V.line(g, x - 6, y + h / 2, x - 6 + (w + 12) * sweep, y + h / 2, { color: C.ink, lw: 1.4, dash: [4, 3] }); T(g, '✂', x - 6 + (w + 12) * sweep, y + h / 2 + 4, { size: 12, align: 'center', color: C.ink }); }
          else { V.line(g, x + w / 2, y - 6, x + w / 2, y - 6 + (h + 12) * sweep, { color: C.ink, lw: 1.4, dash: [4, 3] }); T(g, '✂', x + w / 2 + 1, y - 6 + (h + 12) * sweep + 4, { size: 12, align: 'center', color: C.ink }); }
        }
      };
      // um lado (column- ou row-parallel) de um formato quantizado
      const drawSide = (g, X, Y, PW, PH, side, u, par, compact) => {
        const C = V.C();
        V.box(g, X, Y, PW, PH, { r: 10, fill: V.alpha(C.muted, 0.04), stroke: V.alpha(C.muted, 0.22) });
        T(g, clip(g, side.title, PW - 20, 11.5, 'mono', 700), X + 10, Y + 18, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
        const cutOn = par === 'tp2';
        const whole = par === 'ep2' ? C.e : C.b;
        const t2 = side.ts.filter((q) => q.sh.length >= 2), t1 = side.ts.filter((q) => q.sh.length === 1 && q.sh[0] > 1), t0 = side.ts.filter((q) => !q.sh.length || (q.sh.length === 1 && q.sh[0] === 1));
        const ay = Y + 40, aH = compact ? 70 : 84;
        const slotW = t2.length > 1 ? (PW - 36) / 2 : PW - 24;
        let aBottom = ay;
        t2.forEach((q, k) => {
          const sx0 = X + 12 + k * (slotW + 12);
          const sz = fit(q.sh[0] / q.sh[1], Math.min(slotW - 16, k ? 110 : 170), aH, q.bad != null && q.cut === 1 ? q.grid[1] * 10 : 26, q.bad != null && q.cut === 0 ? q.grid[0] * 10 : 26);
          drawCutRect(g, sx0 + (q.tr ? 0 : 0), ay + (q.tr ? 8 : 0), sz[0], sz[1], { cut: cutOn ? q.cut : null, u: cutOn ? u : 0, grid: q.grid, bad: q.bad, whole, depth: q.tr });
          const ly = ay + (q.tr ? 8 : 0) + sz[1] + 15 + (cutOn ? 5 * u : 0);
          T(g, clip(g, q.n + ' · ' + q.dt, slotW - 2, 10.5, 'mono', 600), sx0, ly, { size: 10.5, font: 'mono', weight: 600, color: C.ink });
          T(g, clip(g, '[' + q.sh.join(', ') + ']' + (q.tr ? ' = [in/16, out/16, 16·4]' : ''), slotW - 2, 10, 'mono'), sx0, ly + 14, { size: 10, font: 'mono', color: C.muted });
          if (cutOn && u > 0.6) T(g, clip(g, '→ ' + q.loc + ' /rank', slotW - 2, 10.5, 'mono', 600), sx0, ly + 28, { size: 10.5, font: 'mono', weight: 600, color: q.bad != null ? badCol(C) : okCol(C), alpha: V.clamp((u - 0.6) / 0.3, 0, 1) });
          else if (!cutOn) T(g, clip(g, par === 'ep2' ? '→ inteiro no rank dono' : '→ inteiro na GPU', slotW - 2, 10.5, 'mono', 600), sx0, ly + 28, { size: 10.5, font: 'mono', weight: 600, color: okCol(C) });
          aBottom = Math.max(aBottom, ly + 32);
        });
        let by = aBottom + 14;
        const maxN = Math.max(1, ...t1.map((q) => q.sh[0]));
        t1.forEach((q) => {
          const bw = (PW - 24) * 0.5 * (q.sh[0] / maxN);
          T(g, q.n + ' · ' + q.dt + ' [' + q.sh[0] + ']', X + 12, by, { size: 10.5, font: 'mono', color: C.ink });
          const stat = !cutOn ? (par === 'ep2' ? 'inteiro' : '—') : q.cut === 0 ? (u > 0.6 ? '→ ' + q.loc + ' por rank' : '') : 'replicado ×2';
          if (stat) T(g, clip(g, stat, PW - 24 - bw - 14, 10.5, 'mono', 600), X + PW - 12, by + 13, { size: 10.5, font: 'mono', weight: 600, align: 'right', color: q.cut === 0 || !cutOn ? okCol(C) : C.muted });
          drawCutRect(g, X + 12, by + 6, bw, 10, { cut: cutOn && q.cut === 0 ? 1 : null, u: cutOn ? u : 0, grid: [1, 16], whole: cutOn ? V.mix(C.e, C.d, 0.5) : whole });
          by += 32;
        });
        let cx = X + 12;
        t0.forEach((q) => {
          const s = q.n + ' · ' + q.dt + (q.sh.length ? ' [1]' : ' escalar') + (cutOn ? '  → replicado' : '');
          const cwid = tw(g, s, 10, 'mono', 500) + 14;
          if (cx + cwid > X + PW - 12) { cx = X + 12; by += 26; }
          chip(g, cx, by - 4, s, C.muted, { size: 10, h: 20 });
          cx += cwid + 8;
        });
        const ok = cutOn ? side.ok : true;
        const vt = cutOn ? (side.ok ? '✓ ' : '✗ ') + side.v : par === 'tp1' ? '✓ TP=1: nada a cortar' : '✓ EP=2: o expert inteiro (pesos + escalas) vai para um rank';
        wrap(g, vt, X + 10, Y + PH - 26, PW - 20, 14, { size: 11, weight: 600, color: ok ? okCol(C) : badCol(C), maxLines: 2 });
      };
      const drawMat = (g, M, x, y, cs, col, o = {}) => {
        const C = V.C();
        M.forEach((row, a) => row.forEach((v, b) => {
          const fx = x + b * (cs + 2), fy = y + a * (cs + 2);
          V.box(g, fx, fy, cs, cs, { r: 3, fill: V.alpha(col, 0.18 + 0.55 * Math.min(1, Math.abs(v))), stroke: o.stroke ? V.alpha(col, 0.8) : null });
          if (cs >= 20) T(g, nBR(v, 1), fx + cs / 2, fy + cs / 2 + 0.5, { size: Math.min(10, cs * 0.42), font: 'mono', align: 'center', base: 'middle', color: C.ink });
        }));
      };
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const i = st.i, el = Math.min(st.elapsed(), t - S.tc);
        S.now = t;
        const quant = i === 3 && S.fmt !== 'bf16';
        const qph = quant ? QF[S.fmt].ph : null;
        stageH(fig, wide ? (quant ? Math.max(430, qph[0] + 176) : 430) : i === 3 ? (quant ? 2 * qph[1] + 204 : 545) : i === 4 ? 690 : 480);
        // esteira de estações
        const bx0 = 18, bx1 = w - 18, by = 40;
        g.lineCap = 'round';
        V.line(g, bx0, by, bx1, by, { color: V.alpha(C.muted, 0.35), lw: 8 });
        V.line(g, bx0, by, bx1, by, { color: V.alpha(C.ink, 0.35), lw: 2, dash: [3, 9], dashOffset: -t * 30 });
        g.lineCap = 'butt';
        STN.forEach((s, k) => {
          const x = bx0 + ((k + 0.5) / STN.length) * (bx1 - bx0), on = k === i, past = k < i;
          V.dot(g, x, by, on ? 11 : 8, on ? C.a : past ? V.alpha(C.a, 0.55) : V.alpha(C.muted, 0.5));
          if (on) V.glow(g, x, by, 24, C.a, 0.5 + 0.3 * Math.sin(t * 5));
          T(g, String(k + 1), x, by + 0.5, { size: 10, font: 'mono', weight: 700, align: 'center', base: 'middle', color: on || past ? (C.dark ? '#00210F' : '#fff') : C.ink });
          T(g, s, x, by + 26, { size: wide ? 11.5 : 10, align: 'center', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
        });
        const y0 = 86, x0 = 16, W = w - 32, Hh = h - y0 - 10;
        V.box(g, x0, y0, W, Hh, { r: 12, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.2) });
        const cs = wide ? 22 : 15;
        if (i === 0) {
          const sx = x0 + 14, sy = y0 + 20;
          V.box(g, sx, sy, 86, Hh - 40, { r: 8, fill: V.alpha(C.f, 0.14), stroke: C.f });
          T(g, 'shard', sx + 43, sy + 18, { size: 11.5, weight: 700, align: 'center', color: C.ink });
          T(g, '.safetensors', sx + 43, sy + 34, { size: 10, font: 'mono', align: 'center', color: C.muted });
          const lx = sx + 100, lw = W - 130;
          NAMES.forEach((n, k) => {
            const u = V.clamp(el * 0.9 - k * 0.3, 0, 1), yy = sy + 4 + k * ((Hh - 50) / NAMES.length);
            const xx = V.lerp(sx + 40, lx, V.ease.out(u));
            const skip = n.startsWith('mtp.');
            const cw = Math.min(lw, tw(g, n, 11, 'mono') + 16);
            if (u > 0) V.box(g, xx, yy, cw, 22, { r: 6, fill: V.alpha(skip ? C.muted : C.b, 0.16), stroke: V.alpha(skip ? C.muted : C.b, 0.7) });
            // corta pela esquerda: o sufixo (w1.weight × w3.weight) é o que distingue os nomes
            if (u > 0) T(g, clipL(g, n, cw - 12, 11, 'mono'), xx + 7, yy + 11.5, { size: 11, font: 'mono', base: 'middle', color: skip ? C.muted : C.ink });
            if (skip && u >= 1) T(g, '⤳ alvo pula mtp.', xx + cw + 8 > x0 + W - 110 ? xx : xx + cw + 8, xx + cw + 8 > x0 + W - 110 ? yy + 34 : yy + 11.5, { size: 10.5, font: 'mono', base: 'middle', color: C.d });
          });
        } else if (i === 1) {
          const lh = (Hh - 20) / REN.length;
          REN.forEach(([a, b, why], k) => {
            const yy = y0 + 12 + k * lh, u = V.clamp(el * 1.1 - k * 0.45, 0, 1);
            const colW = wide ? W * 0.37 : W - 24, rX = x0 + 24 + colW + 14, rW = wide ? W - (rX - x0) - 112 : W - 24;
            T(g, clipL(g, a, colW, 10.5, 'mono'), x0 + 12, yy + (wide ? lh / 2 : lh * 0.3), { size: 10.5, font: 'mono', base: 'middle', color: C.muted });
            if (u > 0) {
              const bxx = wide ? rX : x0 + 12, byy = wide ? yy + lh / 2 : yy + lh * 0.72;
              T(g, wide ? '→' : '↳', wide ? rX - 16 : x0 + 2, byy, { size: 12, color: C.a, base: 'middle', alpha: u });
              T(g, clipL(g, b, rW, 10.5, 'mono', 600), bxx, byy, { size: 10.5, font: 'mono', weight: 600, base: 'middle', color: b.startsWith('(') ? C.d : C.ink, alpha: u });
              if (wide) chip(g, x0 + W - 106, byy - 9, why, C.a, { size: 9.5, h: 18, w: 96 });
            }
          });
        } else if (i === 2) {
          const u = V.ease.inOut(V.clamp((el - 0.6) / 1.6, 0, 1));
          const mw = 6 * (cs + 2);
          const cx = x0 + 20;
          const gx0 = cx, gy0 = y0 + 34, ux0 = cx + mw + 30, uy0 = y0 + 34;
          const tx = cx + mw + (wide ? 60 : 40), ty = y0 + 34;
          const gx = V.lerp(gx0, tx, u), gy = V.lerp(gy0, ty, u), ux = V.lerp(ux0, tx, u), uy = V.lerp(uy0, ty + 4 * (cs + 2) + 6, u);
          if (u < 0.05) { T(g, 'w1 → gate', gx, gy - 10, { size: 10.5, font: 'mono', color: C.a }); T(g, 'w3 → up', ux, uy - 10, { size: 10.5, font: 'mono', color: C.b }); }
          drawMat(g, W1, gx, gy, cs, C.a);
          drawMat(g, W3, ux, uy, cs, C.b);
          if (u >= 1) {
            V.box(g, tx - 6, ty - 8, mw + 12, 8 * (cs + 2) + 20, { r: 8, stroke: V.alpha(C.ink, 0.5), lw: 1.4, fill: null });
            T(g, 'gate_up_proj', tx + mw / 2, ty + 8 * (cs + 2) + 28, { size: 11.5, font: 'mono', weight: 700, align: 'center', color: C.ink });
            T(g, 'gaveta 0 · w1', tx - 12, ty + 2 * (cs + 2), { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.a });
            T(g, 'gaveta 1 · w3', tx - 12, ty + 6 * (cs + 2) + 6, { size: 10.5, font: 'mono', align: 'right', base: 'middle', color: C.b });
          }
          if (wide) {
            const mx = x0 + W - 250, my = y0 + 24;
            T(g, 'stacked_params_mapping (DS)', mx, my, { size: 10.5, font: 'mono', color: C.muted });
            [['gate_up_proj', 'w1', 0], ['gate_up_proj', 'w3', 1], ['attn.fused_wqa_wkv', 'attn.wq_a', 0], ['attn.fused_wqa_wkv', 'attn.wkv', 1], ['compressor.fused_wkv_wgate', 'compressor.wkv', 0]].forEach((r, k) => {
              T(g, clip(g, '(' + r[0] + ', ' + r[1] + ', ' + r[2] + ')', 240, 10, 'mono'), mx, my + 18 + k * 17, { size: 10, font: 'mono', color: C.ink });
            });
            chip(g, mx, my + 110, '"w1" in "markov_w1" → True ⚠', C.d, { size: 10.5 });
            T(g, 'map_dspark_stacked_param_name', mx, my + 146, { size: 10, font: 'mono', color: C.muted });
            T(g, 'compara segmentos e ignora .experts.', mx, my + 162, { size: 10, font: 'mono', color: C.muted });
          } else {
            const my2 = ty + 8 * (cs + 2) + 56;
            T(g, 'stacked_params_mapping (DS)', x0 + 12, my2, { size: 10, font: 'mono', color: C.muted });
            [['gate_up_proj', 'w1', 0], ['gate_up_proj', 'w3', 1], ['attn.fused_wqa_wkv', 'attn.wq_a', 0], ['attn.fused_wqa_wkv', 'attn.wkv', 1]].forEach((r, k) => T(g, clip(g, '(' + r[0] + ', ' + r[1] + ', ' + r[2] + ')', W - 24, 10, 'mono'), x0 + 12, my2 + 17 + k * 16, { size: 10, font: 'mono', color: C.ink }));
            chip(g, x0 + 12, y0 + Hh - 30, '"w1" in "markov_w1" → True ⚠', C.d, { size: 10 });
          }
        } else if (i === 3 && !quant) {
          // BF16: o shared expert do DS (gate_up column-parallel + down row-parallel)
          const P = S.par === 'tp1' ? 1 : 2;
          const mw = 6 * (cs + 2);
          const sx = x0 + 18, sy = y0 + 30;
          T(g, 'gate_up_proj (8 × 6)', sx, sy - 12, { size: 10.5, font: 'mono', color: C.muted });
          drawMat(g, W1, sx, sy, cs, C.a); drawMat(g, W3, sx, sy + 4 * (cs + 2), cs, C.b);
          const u = V.ease.inOut(V.clamp((el - 0.4) / 1.4, 0, 1));
          const gpuX = wide ? sx + mw + 120 : sx + mw + 30;
          const gw = mw + 20;
          for (let r = 0; r < P; r++) {
            const gyy = sy - 6 + r * (P === 2 ? (4 * (cs + 2) + 30) : 0);
            const gh = P === 2 ? 4 * (cs + 2) + 16 : 8 * (cs + 2) + 16;
            V.box(g, gpuX - 10, gyy, gw, gh, { r: 10, fill: V.alpha(r ? C.d : C.e, 0.08), stroke: r ? C.d : C.e, lw: 1.4 });
            if (wide) T(g, 'GPU ' + r, gpuX - 16, gyy + gh / 2, { size: 10.5, font: 'mono', weight: 700, align: 'right', base: 'middle', color: r ? C.d : C.e, alpha: u });
            else T(g, 'GPU ' + r, gpuX + gw - 16, gyy - 3, { size: 10, font: 'mono', weight: 700, align: 'right', color: r ? C.d : C.e });
            const per = 4 / P;
            const rowsL = [];
            for (let k = 0; k < per; k++) rowsL.push(['g', r * per + k]);
            for (let k = 0; k < per; k++) rowsL.push(['u', r * per + k]);
            rowsL.forEach(([kind, ri], k) => {
              const src = kind === 'g' ? W1[ri] : W3[ri];
              const fromY = sy + (kind === 'g' ? ri : 4 + ri) * (cs + 2);
              const toY = gyy + 8 + k * (cs + 2);
              const yy = V.lerp(fromY, toY, u), xx = V.lerp(sx, gpuX, u);
              drawMat(g, [src], xx, yy, cs, kind === 'g' ? C.a : C.b);
            });
          }
          // comparação + row-parallel: à direita (largo) ou embaixo das GPUs (estreito)
          const nx = wide ? gpuX + gw + 30 : x0 + 14, ny = wide ? sy : sy + 190, nw = wide ? x0 + W - nx - 12 : W - 28;
          T(g, 'corte ingênuo (metade de cima):', nx, ny, { size: 10.5, color: C.muted });
          V.box(g, nx, ny + 10, 60, 34, { r: 6, fill: V.alpha(C.a, 0.4) });
          T(g, 'só gate', nx + 30, ny + 27, { size: 10.5, align: 'center', base: 'middle', color: C.ink });
          T(g, '✗ SwiGLU sem up', nx + 70, ny + 27, { size: 11, base: 'middle', color: badCol(C), weight: 600 });
          T(g, P === 2 ? '✓ correto: [gate_r ; up_r]' : '✓ TP=1: tudo numa GPU', nx, ny + 64, { size: 11, weight: 600, color: okCol(C) });
          T(g, 'local = [' + 4 / P + '+' + 4 / P + '] × 6', nx, ny + 82, { size: 10.5, font: 'mono', color: C.muted });
          const ry = ny + 110;
          T(g, 'down_proj (row-parallel): corta as colunas', nx, ry, { size: 10.5, color: C.muted, maxW: nw });
          const c3 = Math.min(14, (nw - 110) / 8 - 2), my = ry + 10;
          for (let a = 0; a < 4; a++) for (let b = 0; b < 8; b++) {
            const rk = P === 2 && b >= 4 ? 1 : 0, gapx = P === 2 && b >= 4 ? 8 * u : 0;
            V.box(g, nx + b * (c3 + 2) + gapx, my + a * (c3 + 2), c3, c3, { r: 2, fill: V.alpha(rk ? C.d : C.e, 0.3 + 0.35 * (((a * 5 + b * 3) % 4) / 4)) });
          }
          const mEnd = nx + 8 * (c3 + 2) + (P === 2 ? 8 : 0), sgx = mEnd + 26, sgy = my + 2 * (c3 + 2);
          if (P === 2) {
            V.dot(g, sgx, sgy, 11, V.alpha(C.a, 0.25)); T(g, 'Σ', sgx, sgy + 1, { size: 13, weight: 700, align: 'center', base: 'middle', color: C.ink });
            const f = (t * 0.6) % 1;
            [[nx + 2 * (c3 + 2), C.e], [nx + 6 * (c3 + 2) + 8, C.d]].forEach(([px, col]) => { const p = V.along(px, my + 4 * (c3 + 2) + 4, sgx, sgy + 11, f); V.dot(g, p[0], p[1], 3, col); });
            T(g, 'all-reduce', sgx + 16, sgy + 4, { size: 10.5, font: 'mono', weight: 600, color: C.a });
          } else T(g, 'sem comunicação', mEnd + 10, sgy + 4, { size: 10.5, font: 'mono', color: C.muted });
          const ey2 = my + 4 * (c3 + 2) + 18;
          const ey3 = wrap(g, P === 2 ? 'cada rank: metade das entradas → soma parcial' : 'um rank: todas as entradas', nx, ey2, nw, 14, { size: 10.5, font: 'mono', color: C.muted, maxLines: 2 });
          if (S.par === 'ep2') wrap(g, 'EP só mexe nos experts roteados: este shared expert segue em TP=2.', nx, ey3 + 4, nw, 14, { size: 10.5, color: C.d, maxLines: 2 });
        } else if (i === 3) {
          // formatos quantizados: o peso e as escalas cortados juntos
          const F = QF[S.fmt];
          const u = S.par === 'tp2' ? V.ease.inOut(V.clamp((el - 0.5) / 1.3, 0, 1)) : 0;
          T(g, clip(g, F.head, W - 24, 11, 'sans', 700), x0 + 12, y0 + 18, { size: 11, weight: 700, color: C.ink });
          const top = y0 + 28;
          let noteY;
          if (wide) {
            const PW = (W - 36) / 2, PH = QF[S.fmt].ph[0];
            drawSide(g, x0 + 12, top, PW, PH, F.col, u, S.par, false);
            drawSide(g, x0 + 24 + PW, top, PW, PH, F.row, u, S.par, false);
            noteY = top + PH + 20;
          } else {
            const PW = W - 24, PH = QF[S.fmt].ph[1];
            drawSide(g, x0 + 12, top, PW, PH, F.col, u, S.par, true);
            drawSide(g, x0 + 12, top + PH + 10, PW, PH, F.row, u, S.par, true);
            noteY = top + 2 * PH + 30;
          }
          wrap(g, F.note, x0 + 12, noteY, W - 24, 15, { size: 11, color: C.muted, maxLines: 4 });
        } else if (i === 4) {
          const par = S.par;
          const u = V.ease.inOut(V.clamp((el - 0.3) / 1.4, 0, 1));
          const Lw = wide ? W * 0.55 : W - 28;
          const ex = x0 + 14, ey = y0 + 30;
          const cw = Math.min(34, (Lw - 16 - 7 * 4) / 8);
          T(g, 'experts globais (8 dos 256, camada 7)', ex, ey - 10, { size: 10.5, font: 'mono', color: C.muted });
          const laneH = 48, lanes = [ey + 58, ey + 58 + laneH + 10];
          lanes.forEach((ly, r) => {
            const active = par !== 'tp1' || r === 0, col = r ? C.d : C.e;
            V.box(g, ex, ly, Lw, laneH, { r: 9, fill: V.alpha(col, active ? 0.08 : 0.02), stroke: V.alpha(col, active ? 0.8 : 0.25), lw: 1.3 });
            T(g, 'GPU ' + r + (active ? '' : ' (sem uso em TP=1)'), ex + 8, ly + 12, { size: 10, font: 'mono', weight: 700, color: col, alpha: active ? 1 : 0.5 });
          });
          const slotX = (k) => ex + 8 + k * (cw + 4);
          for (let e = 0; e < 8; e++) {
            const gx = ex + e * (cw + 4);
            // posição original (fantasma)
            g.setLineDash([3, 3]); V.box(g, gx, ey, cw, 26, { r: 6, stroke: V.alpha(C.muted, 0.5), lw: 1 }); g.setLineDash([]);
            if (par === 'tp2') {
              // cada expert é cortado: linhas 0–1023 de w1/w3 → GPU 0; 1024–2047 → GPU 1
              [0, 1].forEach((r) => {
                const tx = slotX(e), ty = lanes[r] + 20;
                const x = V.lerp(gx, tx, u), y = V.lerp(ey + r * 13, ty, u);
                V.box(g, x, y, cw, 13, { r: 4, fill: V.alpha(r ? C.d : C.e, 0.3), stroke: r ? C.d : C.e, lw: 1 });
                T(g, 'E' + e, x + cw / 2, y + 7, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
              });
            } else {
              const r = par === 'ep2' ? Math.floor(e / 4) : 0, loc = par === 'ep2' ? e % 4 : e;
              const tx = slotX(loc), ty = lanes[r] + 18;
              const x = V.lerp(gx, tx, u), y = V.lerp(ey, ty, u);
              V.box(g, x, y, cw, 24, { r: 6, fill: V.alpha(r ? C.d : C.e, 0.28), stroke: r ? C.d : C.e });
              const lbl = par === 'ep2' && u > 0.9 ? 'E' + e + '→' + loc : 'E' + e;
              T(g, clip(g, lbl, cw - 2, 9.5, 'mono'), x + cw / 2, y + 12.5, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            }
          }
          if (par === 'tp2') {
            const lx = slotX(8) + 6;
            if (lx + 60 < ex + Lw) { T(g, 'linhas 0–1023', lx, lanes[0] + 30, { size: 9.5, font: 'mono', color: C.e }); T(g, '1024–2047', lx, lanes[1] + 30, { size: 9.5, font: 'mono', color: C.d }); }
          }
          const txt = par === 'tp2'
            ? 'TP puro (DeepSeek real: TP=2, sem EP): os experts moram nas duas GPUs, cada um com metade da largura intermediária (1024 de 2048). w1/w3 são cortados nas linhas, w2 nas colunas, e o all-reduce do TP soma as parciais.'
            : par === 'ep2'
              ? 'EP: o expert_map leva índice global → local (E4 → 0 no rank 1) e cada rank guarda experts inteiros, pesos e escalas sem cortes. Qwen: 256 de 512 por GPU; GLM TP3: 96 de 288. O DeepSeek não usa EP.'
              : 'TP=1: todos os experts, inteiros, numa GPU só.';
          const tyEnd = wrap(g, txt, ex, lanes[1] + laneH + 20, Lw, 15, { size: 11, color: C.ink, maxLines: 5 });
          // E8M0 + formas MXFP4 do DS
          const bx = wide ? x0 + W * 0.59 : ex, bw = wide ? W * 0.41 - 14 : Lw;
          const b1 = wide ? y0 + 16 : tyEnd + 4;
          V.box(g, bx, b1, bw, 146, { r: 10, fill: V.alpha(C.c, 0.07), stroke: V.alpha(C.c, 0.5) });
          T(g, clip(g, 'escala E8M0: byte 0111 1000 = 120', bw - 20, 11.5, 'mono', 700), bx + 12, b1 + 22, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
          T(g, clip(g, 'valor = 2^(120 − 127) = 2^−7 = 0,0078125', bw - 20, 11, 'mono'), bx + 12, b1 + 44, { size: 11, font: 'mono', color: C.muted });
          T(g, '.view(torch.uint8) → 120', bx + 12, b1 + 74, { size: 12, font: 'mono', weight: 600, color: okCol(C) });
          T(g, '✓ reinterpreta os mesmos bits', bx + 12, b1 + 92, { size: 11, color: okCol(C) });
          T(g, 'copy_ / .to(uint8) → 0', bx + 12, b1 + 118, { size: 12, font: 'mono', weight: 600, color: badCol(C) });
          T(g, clip(g, '✗ conversão numérica: escala destruída', bw - 20, 11, 'sans'), bx + 12, b1 + 136, { size: 11, color: badCol(C) });
          const b2 = b1 + 158, bh2 = wide ? y0 + Hh - 12 - b2 : 126;
          V.box(g, bx, b2, bw, bh2, { r: 10, fill: V.alpha(C.b, 0.06), stroke: V.alpha(C.b, 0.4) });
          T(g, clip(g, 'um expert DS em MXFP4 (formas pelo config)', bw - 20, 11, 'sans', 700), bx + 12, b2 + 20, { size: 11, weight: 700, color: C.ink });
          const L = [
            ['w1/w3: [2048, 2048] + E8M0 [2048, 128]', C.muted],
            ['w2:    [4096, 1024] + E8M0 [4096, 64]', C.muted],
          ];
          if (par === 'tp2') L.push(['TP2 → w1/w3 [1024, 2048] + [1024, 128]', okCol(C)], ['TP2 → w2 [4096, 512] + [4096, 32] ✓', okCol(C)]);
          else if (par === 'ep2') L.push(['EP2 → expert inteiro no rank dono', okCol(C)], ['(escalas viajam com os códigos)', C.muted]);
          else L.push(['TP1 → expert inteiro na GPU', okCol(C)]);
          L.forEach(([s, col], k) => T(g, clip(g, s, bw - 20, 10, 'mono', k >= 2 ? 600 : 400), bx + 12, b2 + 40 + k * 17, { size: 10, font: 'mono', weight: k >= 2 ? 600 : 400, color: col }));
          if (b2 + 40 + L.length * 17 + 6 < b2 + bh2) T(g, clip(g, 'códigos U8: 2 FP4/byte · 1 E8M0 por 32', bw - 20, 10, 'mono'), bx + 12, b2 + 40 + L.length * 17 + 4, { size: 10, font: 'mono', color: C.muted });
        } else {
          const mx = x0 + 20, my = y0 + 36;
          const u = V.ease.inOut(V.clamp((V.pingpong(el * 0.3, 1) - 0.15) / 0.7, 0, 1));
          const c2 = wide ? 18 : 14;
          T(g, u < 0.5 ? 'row-major (como veio do disco)' : 'layout do kernel: tiles contíguos', mx, my - 12, { size: 10.5, font: 'mono', color: C.muted });
          for (let a = 0; a < 4; a++) for (let b = 0; b < 8; b++) {
            const tile = Math.floor(a / 2) * 2 + Math.floor(b / 4);
            const fx = mx + b * (c2 + 2), fy = my + a * (c2 + 2);
            const lx = mx + tile * (4 * (c2 + 2) + 8) + (b % 4) * (c2 + 2), ly = my + 100 + (a % 2) * (c2 + 2);
            V.box(g, V.lerp(fx, lx, u), V.lerp(fy, ly, u), c2, c2, { r: 3, fill: V.alpha(C.series[tile % 6], 0.6) });
          }
          T(g, 'mesmos valores, outra ordem na memória', mx, my + 100 + 2 * (c2 + 2) + 16, { size: 10.5, color: C.muted });
          const rx = wide ? x0 + W * 0.64 : mx, ry = wide ? y0 + 30 : my + 200;
          [['repack para o kernel', 'ex.: prepare_fp8_layer_for_marlin', C.a], ['desquantizar se preciso', 'GLM: projeções MLA FP8 → BF16', C.d], ['pronto ✓', 'parâmetro residente na GPU', C.e]].forEach(([a, b, col], k) => {
            if (!wide && ry + k * 40 > y0 + Hh - 20) return;
            chip(g, rx, ry + k * (wide ? 52 : 40), a, col, { size: 11.5, font: 'sans', weight: 600, h: 22 });
            T(g, clip(g, b, (wide ? W * 0.4 : W - 40), 10.5, 'mono'), rx + 4, ry + k * (wide ? 52 : 40) + 36, { size: 10.5, font: 'mono', color: C.muted });
          });
        }
      });
    })();

    /* =====================================================================
       FIGURA 4 — o despachante de quant_method
       ===================================================================== */
    (function figDispatch() {
      const fig = root.querySelector('#fig-dispatch');
      const S = { alias: false, block: false };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'patch_checkpoint_config (alias .0.→.48.)', value: false, onChange: (v) => { S.alias = v; } });
      UI.toggle(ctl, { label: 'ramo FP8_BLOCK_SCALES no despacho', value: false, onChange: (v) => { S.block = v; } });
      const rd = UI.readout(ctl, 'MTP →', '');
      const LAYERS = [
        { p: 'model.language_model.layers.5.mlp.experts', m: 0, key: 0 },
        { p: 'model.language_model.layers.3.self_attn.q_proj', m: 1, key: 3 },
        { p: 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding', m: 2, key: 1 },
        { p: 'mtp.layers.48.mlp.experts', m: -1, key: -1 },
      ];
      const METHODS = [['ModelOpt NVFP4 (MoE)', 'e'], ['BF16 (não quantizado)', 'muted'], ['PLE FP8 (via dtype)', 'd'], ['Fp8MoEMethod [128, 128]', 'a'], ['None → sem quantização', 'red']];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 680;
        stageH(fig, wide ? 450 : 666);
        const mtpOK = S.alias && S.block;
        const mtpM = mtpOK ? 3 : 4;
        const rtxt = mtpOK ? '✓ Fp8MoEMethod' : S.alias ? '✗ casou, mas sem ramo' : '✗ nenhuma chave casa';
        if (S.rt !== rtxt) { rd.set(rtxt); S.rt = rtxt; const b = rd.el.querySelector('b'); if (b) b.style.color = mtpOK ? 'var(--viz-e)' : 'var(--red)'; }
        const keys = [
          ['model.language_model.layers.{0..47}.mlp.experts', 'NVFP4 · g16'],
          ['model.language_model.layers.1.ple.ple_embedding.ngram_embedding', 'FP8'],
          ['mtp.layers.0.mlp.experts', S.alias ? 'FP8_BLOCK_SCALES · g128' : 'FP8_PB_WO · g128'],
          ['lm_head, embed_tokens, *self_attn*, *linear_attn*… (292)', 'BF16'],
        ];
        if (S.alias) keys.splice(3, 0, ['mtp.layers.48.mlp.experts  (alias)', 'FP8_BLOCK_SCALES · g128']);
        const colOf = (n) => (n === 'red' ? badCol(C) : C[n]);
        // geometria
        let Lr, Kr, Mr, tl;
        const top = 30;
        if (wide) {
          const lw = w * 0.3, kw = w * 0.36, mw = w - lw - kw - 70;
          const lx = 14, kx = lx + lw + 22, mx = kx + kw + 22;
          const ah = h - top - 110;
          Lr = LAYERS.map((L, k) => ({ x: lx, y: top + 12 + k * (ah / 4), w: lw, h: 42 }));
          Kr = { x: kx, y: top, w: kw, h: ah + 20 };
          Mr = METHODS.map((m, k) => ({ x: mx, y: top + 8 + k * (ah / 5), w: mw, h: 34 }));
          tl = { x: 14, y: h - 76, w: w - 28 };
        } else {
          Lr = LAYERS.map((L, k) => ({ x: 14, y: top + k * 44, w: w - 28, h: 38 }));
          Kr = { x: 14, y: top + 4 * 44 + 20, w: w - 28, h: 196 };
          Mr = METHODS.map((m, k) => ({ x: 14 + (k % 2) * ((w - 36) / 2 + 8), y: Kr.y + Kr.h + 20 + Math.floor(k / 2) * 40, w: (w - 36) / 2, h: 32 }));
          tl = { x: 14, y: h - 90, w: w - 28 };
        }
        T(g, wide ? 'camadas que o vLLM constrói' : 'camadas construídas', Lr[0].x, top - 10 + (wide ? 0 : 0), { size: 10.5, font: 'mono', color: C.muted });
        // dicionário
        V.box(g, Kr.x, Kr.y, Kr.w, Kr.h, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
        const qcHead = tw(g, 'quantization_config (config.json)', 10.5, 'mono') + tw(g, 'quantized_layers', 9.5, 'mono') + 30 < Kr.w ? 'quantization_config (config.json)' : 'quantization_config';
        T(g, qcHead, Kr.x + 10, Kr.y + 16, { size: 10.5, font: 'mono', color: C.muted });
        const kh = (Kr.h - 52) / keys.length;
        // as chaves de quantized_layers vêm primeiro; a última linha é outra estrutura (ignore / exclude_modules)
        const krows = keys.map((k, j) => ({ x: Kr.x + 8, y: Kr.y + 26 + j * kh + (j === keys.length - 1 ? 22 : 0), w: Kr.w - 16, h: kh - 4 }));
        const lastY = krows[keys.length - 1].y;
        T(g, 'quantized_layers', Kr.x + Kr.w - 10, Kr.y + 16, { size: 9.5, font: 'mono', align: 'right', color: C.d });
        g.setLineDash([4, 4]); V.line(g, Kr.x + 8, lastY - 18, Kr.x + Kr.w - 8, lastY - 18, { color: V.alpha(C.muted, 0.5), lw: 1 }); g.setLineDash([]);
        T(g, 'ignore / exclude_modules (outra lista)', Kr.x + 10, lastY - 5, { size: 9.5, font: 'mono', color: C.muted });
        // ciclo de sondas
        const per = 2.6, cyc = Math.floor(t / per) % 4, f = (t % per) / per;
        const probeKey = (L) => {
          if (L.m >= 0) return L.key === 3 ? keys.length - 1 : L.key;
          return S.alias ? 3 : -1;
        };
        keys.forEach((k, j) => {
          const r = krows[j];
          const isAlias = S.alias && j === 3;
          const probing = f > 0.3 && f < 0.6 && probeKey(LAYERS[cyc]) === j;
          V.box(g, r.x, r.y, r.w, r.h, { r: 6, fill: probing ? V.alpha(C.a, 0.25) : isAlias ? V.alpha(C.e, 0.12) : V.alpha(C.muted, 0.05), stroke: probing ? C.a : isAlias ? C.e : V.alpha(C.muted, 0.2) });
          const two = r.h >= 30;
          const vW = two ? 0 : tw(g, k[1], 10, 'mono') + 18;
          T(g, clipL(g, k[0], r.w - 14 - vW, 10.5, 'mono'), r.x + 7, r.y + (two ? r.h * 0.35 : r.h / 2), { size: 10.5, font: 'mono', base: 'middle', color: C.ink });
          T(g, k[1], two ? r.x + 7 : r.x + r.w - 7, r.y + (two ? r.h * 0.74 : r.h / 2), { size: 10, font: 'mono', base: 'middle', align: two ? 'left' : 'right', color: j === 2 || isAlias ? C.d : C.muted });
        });
        // métodos
        Mr.forEach((r, k) => {
          const [l, cn] = METHODS[k];
          const used = LAYERS.some((L) => (L.m >= 0 ? L.m : mtpM) === k);
          V.box(g, r.x, r.y, r.w, r.h, { r: 8, fill: V.alpha(colOf(cn), used ? 0.2 : 0.05), stroke: V.alpha(colOf(cn), used ? 0.9 : 0.3), lw: used ? 1.6 : 1 });
          T(g, clip(g, l, r.w - 14, 11, 'sans', 600), r.x + 8, r.y + r.h / 2 + 0.5, { size: 11, weight: 600, base: 'middle', color: used ? C.ink : C.muted });
        });
        // camadas + rotas
        LAYERS.forEach((L, k) => {
          const r = Lr[k], isMTP = L.m < 0;
          const mIdx = isMTP ? mtpM : L.m;
          const mc = colOf(METHODS[mIdx][1]);
          V.box(g, r.x, r.y, r.w, r.h, { r: 8, fill: V.alpha(isMTP ? C.c : C.b, 0.12), stroke: V.alpha(isMTP ? C.c : C.b, 0.7), lw: isMTP ? 1.8 : 1 });
          T(g, clipL(g, L.p, r.w - 16, 10.5, 'mono', isMTP ? 700 : 500), r.x + 8, r.y + r.h / 2 + 0.5, { size: 10.5, font: 'mono', weight: isMTP ? 700 : 500, base: 'middle', color: C.ink });
          // rota persistente
          const mr = Mr[mIdx];
          const pk = probeKey(L);
          if (wide) bezLine(g, Kr.x + Kr.w, pk >= 0 ? krows[pk].y + krows[pk].h / 2 : Kr.y + Kr.h - 14, mr.x, mr.y + mr.h / 2, { color: V.alpha(mc, 0.25), lw: 1.2 });
          // sonda animada
          if (k === cyc) {
            const kIdx = probeKey(L);
            const kr = kIdx >= 0 ? krows[kIdx] : { x: Kr.x + Kr.w / 2, y: Kr.y + Kr.h / 2, w: 0, h: 0 };
            let p;
            if (wide) {
              if (f < 0.3) p = bezPt(r.x + r.w, r.y + r.h / 2, Kr.x, kr.y + kr.h / 2, V.ease.inOut(f / 0.3));
              else if (f < 0.6) p = [Kr.x + ((f - 0.3) / 0.3) * Kr.w, kr.y + kr.h / 2];
              else p = bezPt(Kr.x + Kr.w, kr.y + kr.h / 2, mr.x, mr.y + mr.h / 2, V.ease.inOut((f - 0.6) / 0.4));
            } else {
              if (f < 0.3) p = bezPt(r.x + r.w / 2, r.y + r.h, kr.x + kr.w / 2 || Kr.x + Kr.w / 2, kr.y, V.ease.inOut(f / 0.3), false);
              else if (f < 0.6) p = [Kr.x + 20 + ((f - 0.3) / 0.3) * (Kr.w - 40), kr.y + kr.h / 2];
              else p = bezPt(Kr.x + Kr.w / 2, Kr.y + Kr.h, mr.x + mr.w / 2, mr.y, V.ease.inOut((f - 0.6) / 0.4), false);
            }
            if (isMTP && kIdx < 0 && f > 0.3 && f < 0.6) { T(g, '✗ nenhuma chave casa', Kr.x + Kr.w / 2, Kr.y + Kr.h - 8, { size: 11, weight: 700, align: 'center', color: badCol(C) }); }
            V.glow(g, p[0], p[1], 14, isMTP ? C.c : C.b);
            V.dot(g, p[0], p[1], 5, isMTP ? C.c : C.b);
          }
        });
        // linha do tempo de boot
        const x0 = tl.x, xw = tl.w, y = tl.y + 30;
        const minToX = (m) => x0 + (m / 11) * xw;
        V.line(g, x0, y, x0 + xw, y, { color: V.alpha(C.muted, 0.5), lw: 2 });
        for (let m = 0; m <= 11; m++) { V.line(g, minToX(m), y - 3, minToX(m), y + 3, { color: V.alpha(C.muted, 0.6), lw: 1 }); if (m % (wide ? 1 : 2) === 0) T(g, m + 'min', minToX(m), y + 16, { size: 9.5, font: 'mono', align: 'center', color: C.muted }); }
        T(g, 'boot da receita Qwen (cold start ≈ 10m55s)', x0, tl.y + 4, { size: 10.5, font: 'mono', color: C.muted });
        V.dot(g, minToX(0.15), y, 5, C.a);
        T(g, 'preflight (s)', minToX(0.15) + 8, y - 8, { size: 10, font: 'mono', color: C.a });
        if (!mtpOK) {
          const px = minToX(7), pulse = 0.6 + 0.4 * Math.sin(t * 6);
          V.glow(g, px, y, 18, badCol(C), pulse);
          V.dot(g, px, y, 6, badCol(C));
          const msg = "AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'";
          if (wide) T(g, clip(g, msg, xw, 10.5, 'mono', 600), x0, y + 34, { size: 10.5, font: 'mono', weight: 600, color: badCol(C) });
          else wrap(g, msg, x0, y + 32, xw, 14, { size: 10.5, font: 'mono', weight: 600, color: badCol(C), maxLines: 3 });
        } else {
          V.dot(g, minToX(10.9), y, 6, okCol(C));
          T(g, '✓ /health · MTP=3: 52,1 tok/s, aceitação 72,8%', x0 + xw, y + 34, { size: 11, weight: 600, align: 'right', color: okCol(C) });
        }
      });
    })();

    /* =====================================================================
       FIGURA 5 — onde o --hf-overrides pousa
       ===================================================================== */
    (function figOverride() {
      const fig = root.querySelector('#fig-override');
      const S = { nested: false, yarn: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Forma', options: [{ v: false, label: 'raiz (bugado)' }, { v: true, label: 'aninhado em text_config' }], value: false, onChange: (v) => { S.nested = v; } });
      UI.toggle(ctl, { label: 'YARN_ENABLE=true (inclui rope_parameters)', value: true, onChange: (v) => { S.yarn = v; } });
      const r1 = UI.readout(ctl, 'text_config.rope_type', '');
      const r2 = UI.readout(ctl, 'text_config.ple_embedding_dtype', '');
      let last = '';
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        stageH(fig, wide ? 400 : 500);
        const effRope = S.nested && S.yarn ? 'yarn' : 'default';
        const effPle = S.nested ? 'float8_e4m3fn' : '(ausente)';
        if (last !== effRope + effPle) { r1.set(effRope); r2.set(effPle); last = effRope + effPle; }
        // forma raiz: com YaRN ligado mostra o bug real (só rope_parameters na raiz);
        // com YaRN desligado mostra a forma hipotética (ple_embedding_dtype na raiz), que teria o mesmo destino
        const rootPle = !S.nested && !S.yarn;
        const N = [
          { id: 'root', l: 'config (pai)', v: 'qwen4_exp', c: C.muted, p: null, d: 0 },
          { id: 'prope', l: 'rope_parameters', v: !S.nested && S.yarn ? '← AQUI (no pai)' : '{…} do checkpoint', c: C.d, p: 'root', d: 1, land: !S.nested && S.yarn },
          { id: 'pple', l: 'ple_embedding_dtype', v: rootPle ? '← AQUI (hipotético)' : '—', c: C.d, p: 'root', d: 1, land: rootPle },
          { id: 'tc', l: 'text_config', v: 'o LLM que roda', c: C.a, p: 'root', d: 1, via: S.nested },
          { id: 'trope', l: 'rope_parameters', v: 'rope_type: ' + effRope, c: C.d, p: 'tc', d: 2, land: S.nested && S.yarn },
          { id: 'tple', l: 'ple_embedding_dtype', v: effPle, c: C.d, p: 'tc', d: 2, land: S.nested },
          { id: 'tnl', l: 'num_hidden_layers', v: '48', c: C.muted, p: 'tc', d: 2 },
          { id: 'vc', l: 'vision_config', v: 'torre visual', c: C.b, p: 'root', d: 1 },
          { id: 'qc', l: 'quantization_config', v: 'modelopt', c: C.c, p: 'root', d: 1 },
        ];
        const by = {}; N.forEach((n) => { by[n.id] = n; });
        // payload
        const pl = S.nested
          ? (S.yarn ? '{"text_config":{"ple_embedding_dtype":…,"rope_parameters":{…}}}' : '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}')
          : (S.yarn ? '{"rope_parameters":{"rope_type":"yarn",…}}' : '{"ple_embedding_dtype":"float8_e4m3fn"}');
        T(g, '--hf-overrides', 14, 16, { size: 10.5, font: 'mono', color: C.d });
        const tag = S.nested ? 'forma correta (atual)' : S.yarn ? 'bug real (start.sh antigo)' : 'forma hipotética';
        T(g, tag, w - 14, 16, { size: 10.5, font: 'mono', weight: 600, align: 'right', color: S.nested ? okCol(C) : S.yarn ? badCol(C) : C.muted });
        const pw = Math.min(w - 28, tw(g, pl, 10.5, 'mono') + 20);
        V.box(g, 14, 24, pw, 26, { r: 7, fill: V.alpha(C.d, 0.15), stroke: C.d });
        T(g, clip(g, pl, pw - 16, 10.5, 'mono'), 22, 37.5, { size: 10.5, font: 'mono', base: 'middle', color: C.ink });
        const nh = wide ? 42 : 36;
        if (wide) {
          const lv1 = N.filter((n) => n.d === 1), lv2 = N.filter((n) => n.d === 2);
          const nw = Math.min(150, (w - 28 - (lv1.length - 1) * 10) / lv1.length);
          const gap = (w - 28 - lv1.length * nw) / (lv1.length - 1);
          by.root.x = w / 2 - nw / 2; by.root.y = 70; by.root.w = nw;
          lv1.forEach((n, k) => { n.x = 14 + k * (nw + gap); n.y = 170; n.w = nw; });
          const n2w = Math.min(200, (w - 28 - 20) / 3);
          const cx = by.tc.x + nw / 2, total = 3 * n2w + 20;
          const start = V.clamp(cx - total / 2, 14, w - 14 - total);
          lv2.forEach((n, k) => { n.x = start + k * (n2w + 10); n.y = 280; n.w = n2w; });
          N.forEach((n) => { if (!n.p) return; const p = by[n.p]; bezLine(g, p.x + p.w / 2, p.y + nh, n.x + n.w / 2, n.y, { color: V.alpha(n.d === 2 ? C.a : C.muted, 0.5), lw: 1.3 }, false); });
        } else {
          N.forEach((n, k) => { n.x = 14 + n.d * 20; n.y = 62 + k * 44; n.w = w - 28 - n.d * 20; });
          N.forEach((n) => { if (!n.p) return; const p = by[n.p]; g.beginPath(); g.moveTo(p.x + 10, p.y + nh); g.lineTo(p.x + 10, n.y + nh / 2); g.lineTo(n.x, n.y + nh / 2); g.strokeStyle = V.alpha(n.d === 2 ? C.a : C.muted, 0.55); g.lineWidth = 1.3; g.stroke(); });
        }
        N.forEach((n) => {
          const hl = n.land || n.via;
          V.box(g, n.x, n.y, n.w, nh, { r: 9, fill: V.alpha(n.c, hl ? 0.26 : 0.09), stroke: hl ? n.c : V.alpha(n.c, 0.55), lw: hl ? 2 : 1 });
          if (wide) {
            T(g, clip(g, n.l, n.w - 14, 11, 'mono', 700), n.x + 8, n.y + 15, { size: 11, font: 'mono', weight: 700, color: C.ink });
            T(g, clip(g, n.v, n.w - 14, 10.5, 'mono'), n.x + 8, n.y + 32, { size: 10.5, font: 'mono', color: n.v.includes('AQUI') ? badCol(C) : C.muted });
          } else {
            const lw = tw(g, n.l, 11, 'mono', 700);
            T(g, clip(g, n.l, n.w * 0.55, 11, 'mono', 700), n.x + 8, n.y + nh / 2 + 0.5, { size: 11, font: 'mono', weight: 700, base: 'middle', color: C.ink });
            const vx = n.x + 8 + Math.min(lw, n.w * 0.55) + 10;
            T(g, clip(g, n.v, n.x + n.w - vx - 8, 10.5, 'mono'), vx, n.y + nh / 2 + 0.5, { size: 10.5, font: 'mono', base: 'middle', color: n.v.includes('AQUI') ? badCol(C) : C.muted });
          }
        });
        // envelopes animados
        const f = (t % 3.2) / 3.2;
        const sx = 14 + pw / 2, sy = 50;
        N.filter((n) => n.land).forEach((tg) => {
          const ex = tg.x + tg.w / 2, ey = tg.y + nh / 2;
          let p;
          if (S.nested) {
            const m = by.tc, mx = m.x + m.w / 2, my = m.y + nh / 2;
            p = f < 0.5 ? bezPt(sx, sy, mx, my, V.ease.inOut(f / 0.5), false) : bezPt(mx, my, ex, ey, V.ease.inOut((f - 0.5) / 0.5), false);
          } else p = bezPt(sx, sy, ex, ey, V.ease.inOut(Math.min(1, f / 0.8)), false);
          V.box(g, p[0] - 9, p[1] - 6, 18, 12, { r: 2, fill: C.d });
          V.line(g, p[0] - 9, p[1] - 6, p[0], p[1] + 1, { color: C.dark ? '#221700' : '#fff', lw: 1 });
          V.line(g, p[0] + 9, p[1] - 6, p[0], p[1] + 1, { color: C.dark ? '#221700' : '#fff', lw: 1 });
          if (f > 0.85) V.glow(g, ex, ey, 28, C.d, (1 - f) * 5);
        });
        const ok = S.nested;
        const msg = ok ? (S.yarn ? '✓ o modelo vê YaRN e o dtype da PLE' : '✓ o modelo vê o dtype da PLE (YaRN desligado de propósito)') : S.yarn ? '✗ no-op silencioso: o text_config segue com rope_type default (YaRN desligado)' : '✗ mesmo mecanismo: o dtype da PLE não chegaria ao text_config';
        if (wide) T(g, clip(g, msg, w - 28, 12, 'sans', 600), 14, h - 12, { size: 12, weight: 600, color: ok ? okCol(C) : badCol(C) });
        else wrap(g, msg, 14, h - 30, w - 28, 16, { size: 12, weight: 600, color: ok ? okCol(C) : badCol(C), maxLines: 2 });
      });
    })();

    /* =====================================================================
       FIGURA 6 — o prédio do cache do Hugging Face
       ===================================================================== */
    (function figCache() {
      const fig = root.querySelector('#fig-cache');
      const S = { sc: 0, nl: false, off: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Cenário', options: [{ v: 0, label: 'download' }, { v: 1, label: 'nova revisão' }, { v: 2, label: 'offline + refs/main' }, { v: 3, label: 'derivado (hard-link)' }], value: 0, onChange: (v) => { S.sc = v; S.t0 = H.t; } });
      UI.toggle(ctl, { label: 'refs/main escrito com echo (\\n no fim)', value: false, onChange: (v) => { S.nl = v; } });
      UI.toggle(ctl, { label: 'HF_HUB_OFFLINE=1', value: true, onChange: (v) => { S.off = v; } });
      S.t0 = -2.4;
      const FILES = ['config.json', 'model.safetensors.index.json', 'model-00001-of-00010.safetensors', 'model-00002-of-00010.safetensors', 'model-fp8-mtp-ple.safetensors'];
      const BL = ['9d1e…', '4a7c…', 'e03b…', '71f5…', 'c28a…'];
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const derived = S.sc === 3;
        const blobs = derived ? ['arquivo novo', 'arquivo novo', 'arquivo novo', 'blob do RadixArk (mesmo inode)', '+2 novos · +201 do RadixArk'] : BL.slice();
        if (S.sc === 1) blobs.push('b7e2… (config novo)');
        const nbl = blobs.length, gut = 30;
        const snH = S.sc === 1 ? 300 : 222, brH = 26 + nbl * 30 + 6;
        stageH(fig, wide ? 420 : 122 + snH + 12 + brH + 84);
        const el = S.sc === 0 ? (t - S.t0) % 8 : t - S.t0;
        const repo = derived ? 'hub/models--MiaAI-Lab--Qwen3.8-Flash-Next-NVFP4-FP8dense/' : 'hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/';
        T(g, clip(g, '$HF_HOME/' + repo, w - 28, 11.5, 'mono', 700), 14, 20, { size: 11.5, font: 'mono', weight: 700, color: C.ink });
        // salas (no largo terminam em h − 70, deixando respiro para o texto de baixo)
        let Rr, Sr, Br;
        if (wide) {
          Rr = { x: 14, y: 34, w: w * 0.22, h: h - 104 };
          Sr = { x: Rr.x + Rr.w + 14, y: 34, w: w * 0.42, h: h - 104 };
          Br = { x: Sr.x + Sr.w + 14, y: 34, w: w - (Sr.x + Sr.w + 14) - 14, h: h - 104 };
        } else {
          Rr = { x: 14, y: 34, w: w - 28, h: 76 };
          Sr = { x: 14, y: 122, w: w - 28, h: snH };
          Br = { x: 14, y: Sr.y + Sr.h + 12, w: w - 28, h: brH };
        }
        const room = (r, l, col) => { V.box(g, r.x, r.y, r.w, r.h, { r: 10, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.35) }); T(g, l, r.x + 10, r.y + 16, { size: 11, font: 'mono', weight: 700, color: col }); };
        room(Rr, 'refs/', C.d); room(Sr, 'snapshots/', C.a); room(Br, derived ? 'no disco (inodes)' : 'blobs/', C.b);
        // refs/main
        const rev = derived ? 'fp8dense-1a2b3c4d' : 'fc694b54…';
        const showNL = S.nl && !derived;
        const rx = Rr.x + 10, ry = Rr.y + 28;
        const refW = Math.min(Rr.w - 20, 170);
        const refWritten = S.sc !== 0 || el > 2.6;
        V.box(g, rx, ry, refW, 36, { r: 7, fill: V.alpha(C.d, refWritten ? 0.18 : 0.05), stroke: V.alpha(C.d, refWritten ? 0.9 : 0.3) });
        T(g, 'main', rx + 8, ry + 12, { size: 10, font: 'mono', color: C.muted });
        if (refWritten) {
          T(g, clip(g, rev, refW - 30, 11, 'mono', 700), rx + 8, ry + 27, { size: 11, font: 'mono', weight: 700, color: C.ink });
          if (showNL) { const nx = rx + 8 + tw(g, clip(g, rev, refW - 30, 11, 'mono', 700), 11, 'mono', 700) + 3; V.box(g, nx, ry + 18, 16, 14, { r: 3, fill: V.alpha(badCol(C), 0.3), stroke: badCol(C) }); T(g, '⏎', nx + 8, ry + 25.5, { size: 10, align: 'center', base: 'middle', color: badCol(C) }); }
        }
        const notes = derived ? [['echo -n (sem \\n)', okCol(C)], ['(rev ilustrativa)', C.muted]] : [[S.sc === 1 ? 'aponta para o commit novo' : 'texto: um hash de commit', C.muted]];
        notes.forEach(([s, col], k) => {
          if (wide) T(g, clip(g, s, Rr.w - 20, 10, 'mono'), rx, ry + 52 + k * 16, { size: 10, font: 'mono', color: col });
          else { const nx = rx + refW + 12, mw = Rr.x + Rr.w - nx - 8; const cs = clip(g, s, mw, 10, 'mono'); if (cs) T(g, cs, nx, ry + 14 + k * 16, { size: 10, font: 'mono', color: col }); }
        });
        // geometria dos snapshots
        const snaps = S.sc === 1 ? ['<commit anterior>', 'fc694b54…'] : derived ? ['fp8dense-1a2b3c4d (rev ilustrativa)'] : ['fc694b54…'];
        const files = derived ? ['config.json (novo)', 'model-bf16-00001….safetensors (reescrito)', 'model-bf16-00002….safetensors (reescrito)', 'model-…experts….safetensors', wide ? '… +2 bf16 reescritos · +201 hard-links' : '… +2 reescritos · +201 hard-links'] : FILES;
        const two = snaps.length === 2;
        const sw = two && wide ? (Sr.w - 30) / 2 : Sr.w - 20;
        const snapBoxes = [], entries = [];
        snaps.forEach((sn, s) => {
          const sx = Sr.x + 10 + (two && wide ? s * (sw + 10) : 0);
          const sy = Sr.y + 28 + (two && !wide ? s * ((Sr.h - 30) / 2) : 0);
          const sh = two && !wide ? (Sr.h - 40) / 2 : Sr.h - 38;
          snapBoxes.push({ sn, s, sx, sy, sh });
          const fh = Math.min(30, (sh - 26) / files.length);
          const fw = wide ? sw - 12 : sw - 12 - gut;
          files.forEach((fn, k) => {
            const fy = sy + 24 + k * fh;
            const appear = S.sc !== 0 || el > 0.4 + k * 0.4;
            if (!appear) return;
            let blob = k;
            if (S.sc === 1 && k === 0 && s === 1) blob = 5;
            const isNew = derived && k < 3;
            entries.push({ fn, s, k, x: sx + 6, y: fy, w: fw, h: fh - 4, blob, isNew, derivedLink: derived && k >= 3 });
          });
        });
        // geometria dos blobs (coluna única; no estreito sobra uma calha à direita para as ligações)
        const bh = wide ? Math.min(34, (Br.h - 36) / nbl) : 30;
        const bw = wide ? Br.w - 20 : Br.w - 20 - gut;
        const bpos = blobs.map((b, k) => ({ x: Br.x + 10, y: Br.y + 26 + k * bh, w: bw, h: bh - 4 }));
        const lane = (b) => Sr.x + Sr.w - 8 - b * 4;
        // 1) ligações snapshot → blob, desenhadas ANTES das caixas (que são opacas)
        entries.forEach((e) => {
          const p = bpos[Math.min(e.blob, bpos.length - 1)];
          if (!p) return;
          const col = e.derivedLink ? C.e : e.isNew ? C.c : C.a;
          const o = { color: V.alpha(col, e.s === snaps.length - 1 ? 0.75 : 0.4), lw: e.derivedLink ? 2 : 1.2, dash: e.derivedLink || e.isNew ? null : [3, 3], dashOffset: -t * 10 };
          if (wide) bezLine(g, e.x + e.w, e.y + e.h / 2, p.x, p.y + p.h / 2, o, true);
          else { const gx = lane(Math.min(e.blob, bpos.length - 1)); polyLine(g, [[e.x + e.w, e.y + e.h / 2], [gx, e.y + e.h / 2], [gx, p.y + p.h / 2], [p.x + p.w, p.y + p.h / 2]], o); }
        });
        // pacote do download (cenário 0), também por baixo das caixas
        if (S.sc === 0 && el < 3.2) {
          const bi = Math.min(4, Math.floor(el / 0.6)), bp = bpos[bi], u = (el * 0.5) % 1;
          let p;
          if (wide) p = V.along(Br.x + Br.w - 50, 18, bp.x + 20, bp.y, u);
          else p = polyPt([[lane(bi), 28], [lane(bi), bp.y + bp.h / 2], [bp.x + bp.w, bp.y + bp.h / 2]], u);
          V.dot(g, p[0], p[1], 4, C.b); V.glow(g, p[0], p[1], 10, C.b, 0.6);
        }
        // 2) caixas dos snapshots e dos arquivos
        snapBoxes.forEach(({ sn, s, sx, sy, sh }) => {
          V.box(g, sx, sy, sw, sh, { r: 8, fill: V.alpha(C.a, s === snaps.length - 1 ? 0.1 : 0.04), stroke: V.alpha(C.a, 0.4) });
          T(g, clip(g, sn + '/', sw - 16, 10.5, 'mono', 700), sx + 8, sy + 15, { size: 10.5, font: 'mono', weight: 700, color: C.ink });
        });
        entries.forEach((e) => {
          solidBox(g, e.x, e.y, e.w, e.h, { r: 5, fill: V.alpha(e.isNew ? C.c : C.a, 0.12), stroke: V.alpha(e.isNew ? C.c : C.a, 0.45) });
          T(g, clip(g, e.fn, e.w - 14, 10, 'mono'), e.x + 6, e.y + e.h / 2 + 0.5, { size: 10, font: 'mono', base: 'middle', color: C.ink });
        });
        // 3) blobs
        blobs.forEach((b, k) => {
          const p = bpos[k];
          const appear = S.sc !== 0 || el > 0.2 + k * 0.4;
          if (!appear) return;
          const isNewB = (S.sc === 1 && k === 5) || (derived && k < 3);
          solidBox(g, p.x, p.y, p.w, p.h, { r: 5, fill: V.alpha(isNewB ? C.c : C.b, 0.14), stroke: V.alpha(isNewB ? C.c : C.b, 0.6) });
          T(g, clip(g, b, p.w - 14, 10, 'mono'), p.x + 7, p.y + p.h / 2 + 0.5, { size: 10, font: 'mono', base: 'middle', color: C.ink });
        });
        if (!derived) T(g, '(hashes ilustrativos)', Br.x + Br.w - 10, Br.y + 16, { size: 9.5, font: 'mono', align: 'right', color: C.muted });
        // painel inferior: cenário
        const by = wide ? h - 50 : h - 62, ml = wide ? 3 : 4;
        if (S.sc === 0) {
          wrap(g, 'snapshot_download grava cada arquivo uma vez em blobs/ (nome = hash do conteúdo), cria symlinks relativos em snapshots/<commit>/ e, por fim, refs/main.', 14, by, w - 28, 15, { size: 11.5, color: C.ink, maxLines: ml });
        } else if (S.sc === 1) {
          wrap(g, 'Ilustrativo: um commit que trocasse só o config.json (como o rótulo que a fc694b54 renomeou no config) manteria cada shard no mesmo blob e ganharia 1 blob novo. Dois snapshots, quase zero byte extra.', 14, by, w - 28, 15, { size: 11.5, color: C.ink, maxLines: ml });
        } else if (S.sc === 2) {
          const bad = S.off && S.nl;
          const path = 'snapshots/' + rev + (S.nl ? '\\n' : '') + '/config.json';
          T(g, clip(g, (S.off ? 'offline: ' : 'online: consulta o Hub; ') + 'refs/main → ' + path, w - 28, 11, 'mono', 600), 14, by, { size: 11, font: 'mono', weight: 600, color: C.ink });
          const msg = !S.off ? '⚠ online: se a revisão não estiver no cache, baixa em silêncio (DS: ~157 GiB)' : bad ? '✗ repo not found: hf_hub 1.28 não faz strip (issue #36) → normalize com printf' : '✓ encontra o snapshot: boot offline reprodutível';
          wrap(g, msg, 14, by + 20, w - 28, 16, { size: 12, weight: 600, color: !S.off ? C.d : bad ? badCol(C) : okCol(C), maxLines: 2 });
          // lupa percorrendo
          const u = V.pingpong(t * 0.4, 1);
          const sx0 = rx + refW / 2, sy0 = ry + 36, tx = Sr.x + Sr.w / 2, ty = Sr.y + 30;
          const p = bezPt(sx0, sy0, tx, ty, u, wide);
          V.glow(g, p[0], p[1], 14, bad ? badCol(C) : C.d);
          V.dot(g, p[0], p[1], 4.5, bad ? badCol(C) : C.d);
          if (bad && u > 0.9) T(g, '✗', tx, ty + 4, { size: 22, weight: 700, align: 'center', color: badCol(C) });
        } else {
          wrap(g, 'Shards intocados: hard-link para o blob do RadixArk (via realpath) → mesmo inode, 0 B copiados; só 4 shards bf16 são reescritos (11,8 GB). Um hard-link do symlink relativo ficaria pendurado.', 14, by, w - 28, 15, { size: 11.5, color: C.ink, maxLines: ml });
        }
      });
    })();

    /* =====================================================================
       FIGURA 7 — linha do tempo de um repositório no Hub
       ===================================================================== */
    (function figRev() {
      const fig = root.querySelector('#fig-rev');
      const S = { mode: 'pin', pushes: 0, del: false, tPush: -10 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Estratégia', options: [{ v: 'main', label: 'seguir main' }, { v: 'pin', label: 'pinar commit' }, { v: 'mirror', label: 'pinar + espelho' }], value: 'pin', onChange: (v) => { S.mode = v; } });
      UI.button(ctl, '▶ novo push no upstream', () => { if (!S.del && S.pushes < 4) { S.pushes++; S.tPush = H.t; } }, { primary: true });
      UI.button(ctl, '✕ apagar upstream', () => { S.del = true; });
      UI.button(ctl, '↺ reiniciar', () => { S.pushes = 0; S.del = false; });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 620;
        const x0 = wide ? 170 : 16, x1 = w - 24;
        const yU = wide ? 62 : 70, yM = wide ? 150 : 168, yR = wide ? 262 : 276;
        // rótulos das faixas
        if (wide) {
          T(g, 'upstream', 14, yU - 6, { size: 11.5, weight: 700, color: C.ink });
          T(g, 'brandonmusic/GLM-5.3-…', 14, yU + 10, { size: 10, font: 'mono', color: C.muted });
          T(g, 'espelho', 14, yM - 6, { size: 11.5, weight: 700, color: C.ink });
          T(g, 'Mia-AiLab/GLM-5.3-…', 14, yM + 10, { size: 10, font: 'mono', color: C.muted });
          T(g, 'a receita', 14, yR - 6, { size: 11.5, weight: 700, color: C.ink });
          T(g, S.mode === 'main' ? 'revision=main' : S.mode === 'pin' ? '@5ab363a8…' : 'MODEL_REVISION=25a44fdb…', 14, yR + 10, { size: 10, font: 'mono', color: C.muted });
        } else {
          T(g, 'upstream: brandonmusic/GLM-5.3-…', x0, yU + 44, { size: 10.5, font: 'mono', color: C.muted });
          T(g, 'espelho: Mia-AiLab/GLM-5.3-…', x0, yM + 44, { size: 10.5, font: 'mono', color: C.muted });
        }
        // trilho upstream
        const dead = S.del;
        V.line(g, x0, yU, x1, yU, { color: V.alpha(dead ? C.muted : C.b, dead ? 0.25 : 0.6), lw: 3, dash: dead ? [4, 6] : null });
        const cx = (k) => x0 + 20 + k * Math.min(90, (x1 - x0 - 40) / 4.5);
        const commits = [{ l: '5ab363a8', real: true }];
        for (let k = 1; k <= S.pushes; k++) commits.push({ l: 'push ' + k });
        commits.forEach((c, k) => {
          const ap = k === S.pushes ? V.ease.back(V.clamp((t - S.tPush) / 0.6, 0, 1)) : 1;
          const x = cx(k);
          V.dot(g, x, yU, 8 * ap, dead ? V.alpha(C.muted, 0.5) : k ? C.c : C.b);
          T(g, c.l, x, yU + 24, { size: 10, font: 'mono', align: 'center', color: dead ? C.muted : C.ink, alpha: ap });
        });
        // ponteiro main
        const mainX = V.lerp(cx(Math.max(0, S.pushes - 1)), cx(S.pushes), V.ease.inOut(V.clamp((t - S.tPush) / 0.8, 0, 1)));
        if (!dead) { chip(g, mainX - 20, yU - 36, 'main', C.d, { size: 10.5 }); V.line(g, mainX, yU - 16, mainX, yU - 8, { color: C.d, lw: 1.5 }); }
        else T(g, '404: repositório apagado', cx(0), yU - 22, { size: 11, weight: 700, color: badCol(C) });
        // espelho
        V.line(g, x0, yM, x1, yM, { color: V.alpha(C.a, 0.6), lw: 3 });
        const mx = cx(0) + 40;
        V.dot(g, mx, yM, 8, C.a);
        T(g, '25a44fdb', mx, yM + 24, { size: 10, font: 'mono', align: 'center', color: C.ink });
        bezLine(g, cx(0), yU + 8, mx, yM - 8, { color: V.alpha(C.a, 0.6), lw: 1.4, dash: [4, 4], dashOffset: -t * 12 }, false);
        if (wide) T(g, 'cópia byte-idêntica', (cx(0) + mx) / 2 + 12, (yU + yM) / 2, { size: 10, font: 'mono', color: C.a });
        else T(g, 'cópia byte-idêntica', mx + 18, yM - 14, { size: 10, font: 'mono', color: C.a });
        // receita (personagem)
        const rx = wide ? (x0 + x1) / 2 + 60 : w / 2, ry = yR;
        let tgt, ok, msg;
        if (S.mode === 'main') {
          tgt = [mainX, yU]; ok = !dead && S.pushes === 0;
          msg = dead ? '✗ nó novo não consegue baixar (404)' : S.pushes ? '✗ os pesos mudaram sem aviso: não é o modelo medido' : '✓ por enquanto, main = o commit medido';
        } else if (S.mode === 'pin') {
          tgt = [cx(0), yU]; ok = !dead;
          msg = dead ? '✗ cache local ok, mas um nó novo não baixa mais' : '✓ bytes idênticos, não importa quantos pushes';
        } else {
          tgt = [mx, yM]; ok = true;
          msg = dead ? '✓ espelho durável serve; fallback só se o espelho sumir' : '✓ imune a pushes; o espelho é seu';
        }
        const bob = Math.sin(t * 3) * 2;
        bezLine(g, rx, ry - 18, tgt[0], tgt[1] + 9, { color: V.alpha(ok ? okCol(C) : badCol(C), 0.8), lw: 2, dash: [5, 4], dashOffset: -t * 20 }, false);
        V.box(g, rx - 22, ry - 18 + bob, 44, 36, { r: 10, fill: V.alpha(ok ? okCol(C) : badCol(C), 0.2), stroke: ok ? okCol(C) : badCol(C), lw: 1.6 });
        V.dot(g, rx - 8, ry - 4 + bob, 3, C.ink); V.dot(g, rx + 8, ry - 4 + bob, 3, C.ink);
        g.beginPath(); g.arc(rx, ry + 4 + bob, 7, ok ? 0.15 * Math.PI : 1.15 * Math.PI, ok ? 0.85 * Math.PI : 1.85 * Math.PI); g.strokeStyle = C.ink; g.lineWidth = 1.6; g.stroke();
        T(g, clip(g, msg, w - 28, 12, 'sans', 600), wide ? 14 : 14, h - 34, { size: 12, weight: 600, color: ok ? okCol(C) : badCol(C) });
        T(g, clip(g, 'DFLASH_REVISION=dc77ff1c…: “um main mutável do Hub já trocou os pesos”', w - 28, 10.5, 'mono'), 14, h - 14, { size: 10.5, font: 'mono', color: C.muted });
      });
    })();

    /* =====================================================================
       FIGURA 7b — dois pins, quais bytes? (árvores reais das duas lanes GLM)
       ===================================================================== */
    (function figPins() {
      const fig = root.querySelector('#fig-pins');
      if (!fig) return;
      let tfCol = null, tfDark = null;
      const TFC = (C) => { if (tfCol == null || tfDark !== C.dark) { tfCol = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; tfDark = C.dark; } return tfCol; };
      // tamanho em bytes e prefixo do hash (lfs.oid SHA-256 nos arquivos LFS; oid SHA-1 de blob git nos pequenos), API do Hub em 2026-10-01
      const REPO = {
        ckpt: {
          id: 'Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw', L: ['25a44fdb', '2026-08-28'], R: ['9eaebb7c', '2026-09-14'],
          rows: [
            { f: 'config.json', a: [6220, 'fc50281d'], b: [6220, 'fc50281d'] },
            { f: 'model.safetensors.index.json', a: [15576722, '2f64d21c'], b: [15576722, '2f64d21c'] },
            { f: 'quantization_config.json', a: [37707425, '1e5cdf56'], b: [37707425, '1e5cdf56'] },
            { f: 'model-00001-of-00120.safetensors', w: 1, a: [4113768668, '46cc9e99'], b: [4113768668, '46cc9e99'] },
            { f: '118 shards do meio (00002…00119)', w: 1, many: 118, a: [169391350980, '118 × SHA-256'], b: [169391350980, '118 × SHA-256'] },
            { f: 'model-00120-of-00120.safetensors', w: 1, a: [2137038104, '15ffa59a'], b: [2137038104, '15ffa59a'] },
            { f: 'README.md', a: [3357, 'fee1e0da'], b: [3018, 'a43c1f79'] },
          ],
        },
        dfl: {
          id: 'incoai/GLM-5.3-Flash-DFlash2', L: ['dc77ff1c', '2026-08-28'], R: ['bf582e4e', '2026-08-31'],
          rows: [
            { f: 'config.json', a: [1273, '083085aa'], b: [1273, '083085aa'] },
            { f: 'model.safetensors', w: 1, a: [2342169800, 'b33c0347'], b: [2342169800, 'b038e1d9'] },
            { f: 'README.md', a: [4666, '7b22c057'], b: [2619, 'd70cae7d'] },
            { f: 'assets/dflash2-figure.png', a: null, b: [286889, '6d8dcc9a'] },
            { f: '.gitattributes', a: [116, 'e404a01e'], b: [116, 'e404a01e'] },
          ],
        },
      };
      const VERD = {
        ckpt: {
          commit: ['?', 'inconclusivo', 'Commits diferentes (25a44fdb ≠ 9eaebb7c). Isso diz que algo mudou no repositório, não o quê.', 'warn'],
          size: ['≈', 'parecem iguais', 'Tudo com o mesmo tamanho, menos o README.md. Promissor, mas tamanho não é prova.', 'warn'],
          hash: ['=', 'mesmos pesos', 'Mesmo hash em tudo, menos no README.md: os 120 shards têm o mesmo SHA-256. Dois pins, os mesmos bytes de peso.', 'ok'],
        },
        dfl: {
          commit: ['?', 'inconclusivo', 'Commits diferentes (dc77ff1c ≠ bf582e4e), ambos “Checkpoint update”. O que mudou?', 'warn'],
          size: ['≈', 'parecem iguais', 'model.safetensors com 2.342.169.800 bytes nas duas revisões e o mesmo config: parece o mesmo drafter.', 'warn'],
          hash: ['≠', 'pesos diferentes', 'Mesmo tamanho, mesmo config.json, outro SHA-256: as duas lanes rascunham com outros pesos.', 'bad'],
        },
      };
      const S = { r: 'dfl', m: 'size', t0: -5 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Repositório', options: [{ v: 'ckpt', label: 'checkpoint' }, { v: 'dfl', label: 'drafter DFlash2' }], value: 'dfl', onChange: (v) => { S.r = v; S.t0 = H.t; upd(); } });
      UI.seg(ctl, { label: 'Comparar por', options: [{ v: 'commit', label: 'id do commit' }, { v: 'size', label: 'tamanho' }, { v: 'hash', label: 'hash do conteúdo' }], value: 'size', onChange: (v) => { S.m = v; S.t0 = H.t; upd(); } });
      const rd = UI.readout(ctl, 'Pesos', '—');
      const upd = () => { const v = VERD[S.r][S.m]; rd.set(v[1], v[3]); };
      const cell = (x, m) => {
        if (!x) return 'ausente';
        if (m === 'size') return nBR(x[0], 0) + ' B';
        return /^[0-9a-f]{8}$/.test(x[1]) ? x[1] + '…' : x[1];
      };
      const same = (r, m) => m === 'commit' ? null : (!!r.a && !!r.b && (m === 'size' ? r.a[0] === r.b[0] : r.a[1] === r.b[1]));
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), TF = TFC(C);
        const D = REPO[S.r], rows = D.rows, n = rows.length, vd = VERD[S.r][S.m];
        const wide = w >= 640;
        const rh0 = wide ? 30 : 44, top = wide ? 78 : 96, vh = wide ? 62 : 88;
        const rh = Math.min(rh0 * 1.25, rh0 * 7 / n); // com menos linhas, elas respiram (o veredito vem logo abaixo)
        stageH(fig, Math.ceil(top + 7 * rh0 + vh + 24)); // altura fixa (7 linhas): trocar de repositório não faz a página pular
        const el = t - S.t0;
        const xA = wide ? w * 0.52 : 14 + (w - 28) * 0.23, xM = wide ? w * 0.675 : w / 2, xB = wide ? w * 0.83 : 14 + (w - 28) * 0.77;
        // cabeçalho
        T(g, clip(g, D.id, w - 28, 11, 'mono'), 14, 18, { size: 11, font: 'mono', color: C.muted });
        const head = (x, col, lane, rev, date) => {
          const s = (wide ? lane + ' ' : '') + '@' + rev;
          const cw = tw(g, s, 11, 'mono', 600) + 16;
          V.box(g, x - cw / 2, 32, cw, 22, { r: 8, fill: V.alpha(col, 0.16), stroke: col, lw: 1.4 });
          T(g, s, x, 43.5, { size: 11, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
          T(g, wide ? date : lane + ' · ' + date, x, 68, { size: 10, font: 'mono', align: 'center', color: C.muted });
        };
        head(xA, C.glm, 'lane vLLM', D.L[0], D.L[1]);
        head(xB, TF, wide ? 'lane TensorFold' : 'TF', D.R[0], D.R[1]);
        if (wide) T(g, 'arquivo', 14, 68, { size: 10, font: 'mono', color: C.muted });
        // linhas
        const scanEnd = 0.25 + n * 0.14;
        rows.forEach((r, k) => {
          const y0 = top + k * rh;
          if (r.w) V.box(g, 6, y0 + 2, w - 12, rh - 4, { r: 6, fill: V.alpha(C.f, 0.08) });
          const shown = el > 0.2 + k * 0.14;
          const eq = same(r, S.m);
          const yv = wide ? y0 + rh / 2 : y0 + rh / 2 + 9, yn = wide ? y0 + rh / 2 : y0 + rh / 2 - 9;
          const nameW = wide ? xA - 70 - 14 : w - 28;
          T(g, clip(g, r.f, nameW, 11, 'mono', r.w ? 700 : 400), 14, yn, { size: 11, font: 'mono', weight: r.w ? 700 : 400, base: 'middle', color: C.ink });
          if (r.w && wide) {
            const nw = Math.min(nameW, tw(g, r.f, 11, 'mono', 700));
            T(g, 'pesos', 14 + nw + 8, yn, { size: 9.5, weight: 700, base: 'middle', color: C.f });
          }
          const vs = (x) => S.m === 'commit' ? null : cell(x, S.m);
          const colOf = (isA) => {
            if (S.m === 'commit') return isA ? C.glm : TF;
            if (!shown) return C.muted;
            return eq ? C.ink : badCol(C);
          };
          const ta = S.m === 'commit' ? '@' + D.L[0] : vs(r.a), tb = S.m === 'commit' ? '@' + D.R[0] : vs(r.b);
          const vmax = wide ? (xM - xA) * 2 - 34 : (xM - 14 - 16) * 2 - 8;
          T(g, clip(g, ta, vmax, 10.5, 'mono'), xA, yv, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: colOf(true), alpha: shown || S.m === 'commit' ? 1 : 0.45 });
          T(g, clip(g, tb, vmax, 10.5, 'mono'), xB, yv, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: colOf(false), alpha: shown || S.m === 'commit' ? 1 : 0.45 });
          // marca
          if (shown && el > 0.25 + k * 0.14) {
            const ap = Math.max(0.3, V.ease.out(V.clamp((el - 0.2 - k * 0.14) / 0.35, 0, 1)));
            let sym = '?', mc = C.d;
            if (eq === true) { sym = '='; mc = S.m === 'hash' ? okCol(C) : C.d; }
            else if (eq === false) { sym = '≠'; mc = badCol(C); }
            if (r.w && S.m === 'hash') V.glow(g, xM, yv, 16 + 3 * Math.sin(t * 3), mc, 0.45);
            V.dot(g, xM, yv, 10 * ap, V.alpha(mc, 0.18));
            g.beginPath(); g.arc(xM, yv, 10 * ap, 0, Math.PI * 2); g.strokeStyle = mc; g.lineWidth = 1.5; g.stroke();
            T(g, sym, xM, yv + 0.5, { size: 13 * ap, weight: 700, align: 'center', base: 'middle', color: mc });
          }
        });
        // varredura
        if (el < scanEnd) {
          const u = V.clamp(el / scanEnd, 0, 1), sy = top + u * n * rh;
          V.line(g, 8, sy, w - 8, sy, { color: V.alpha(C.a, 0.8), lw: 2 });
        }
        // veredito
        const vy = top + n * rh + 10, okc = vd[3] === 'ok' ? okCol(C) : vd[3] === 'bad' ? badCol(C) : C.d;
        const va = V.clamp((el - scanEnd) / 0.4, 0, 1);
        V.box(g, 14, vy, w - 28, vh, { r: 10, fill: V.alpha(okc, 0.1 * va), stroke: V.alpha(okc, 0.7 * va), lw: 1.4 });
        if (va > 0) {
          T(g, vd[0] + '  ' + vd[1], 26, vy + 18, { size: 12.5, weight: 700, color: okc, alpha: va });
          wrap(g, vd[2], 26, vy + 37, w - 52, 16, { size: 11.5, color: C.ink, maxLines: wide ? 2 : 3, alpha: va });
        } else T(g, 'comparando arquivo a arquivo…', 26, vy + vh / 2, { size: 11.5, base: 'middle', color: C.muted });
      });
      upd();
    })();

    /* =====================================================================
       FIGURA 8 — o detector de defeitos (com hashes reais)
       ===================================================================== */
    (function figIntegrity() {
      const fig = root.querySelector('#fig-integrity');
      const DEF = [
        { v: 'ok', label: 'íntegro' }, { v: 'missing', label: 'shard ausente' }, { v: 'trunc', label: 'truncado' }, { v: 'bits', label: 'bits trocados' }, { v: 'cfg', label: 'config reescrito' },
      ];
      const CATCH = { ok: [0, 0, 0, 0], missing: [1, 1, 1, 1], trunc: [0, 0, 1, 1], bits: [0, 0, 0, 1], cfg: [0, 0, 0, 1] };
      const DETS = ['contagem de shards', 'weight_map completo', 'tamanho desreferenciado', 'hash de conteúdo'];
      const DSUB = ['EXPECTED_SHARDS=120', 'resolve_snapshot.py', 'find -L … %s (GLM)', 'SHA-256 / SHA-1 git'];
      const r0 = V.rng(7);
      const base = new Uint8Array(48); for (let k = 0; k < 48; k++) base[k] = r0.int(256);
      const S = { d: 'ok', buf: base.slice(), flips: 0, t0: -4 };
      const CFG0 = '{"num_experts_per_tok": 8}', CFG1 = '{"num_experts_per_tok": 7}';
      let hBase = sha256(base), hCur = hBase;
      const gB0 = gitBlob(CFG0), gB1 = gitBlob(CFG1);
      const flip = () => { const k = Math.floor(Math.random() * 48 * 8); S.buf[k >> 3] ^= 1 << (k & 7); S.flips++; hCur = sha256(S.buf); };
      const reset = () => { S.buf = base.slice(); S.flips = 0; hCur = hBase; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Defeito', options: DEF, value: 'ok', onChange: (v) => { S.d = v; S.t0 = H.t; reset(); if (v === 'bits') flip(); } });
      UI.button(ctl, 'trocar 1 bit', () => { flip(); if (S.d !== 'bits' && S.d !== 'cfg') { S.d = 'bits'; } S.t0 = H.t; });
      UI.button(ctl, 'restaurar', () => { reset(); });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        stageH(fig, wide ? 470 : 690);
        const el = t - S.t0;
        const cat = CATCH[S.d];
        // arquivos
        T(g, 'o snapshot (6 dos 120 shards + config.json)', 14, 18, { size: 10.5, font: 'mono', color: C.muted });
        const nF = 7, fw = Math.min(90, (w - 28 - 6 * 8) / nF), fy = 30, fh = 56;
        for (let k = 0; k < nF; k++) {
          const x = 14 + k * (fw + 8);
          const isCfg = k === 6, isBad = (k === 3 && ['missing', 'trunc', 'bits'].includes(S.d)) || (isCfg && S.d === 'cfg');
          const col = isCfg ? C.a : C.f;
          if (k === 3 && S.d === 'missing') {
            g.setLineDash([4, 4]); V.box(g, x, fy, fw, fh, { r: 7, stroke: V.alpha(badCol(C), 0.8) }); g.setLineDash([]);
            T(g, '?', x + fw / 2, fy + fh / 2 + 1, { size: 18, align: 'center', base: 'middle', color: badCol(C), weight: 700 });
          } else {
            const hh = k === 3 && S.d === 'trunc' ? fh * 0.55 : fh;
            V.box(g, x, fy + fh - hh, fw, hh, { r: 7, fill: V.alpha(col, 0.18), stroke: isBad ? badCol(C) : V.alpha(col, 0.7), lw: isBad ? 2 : 1 });
            if (k === 3 && S.d === 'bits') { V.dot(g, x + fw * 0.62, fy + fh * 0.4, 3.5, badCol(C)); V.glow(g, x + fw * 0.62, fy + fh * 0.4, 12, badCol(C)); }
            if (isCfg && S.d === 'cfg') T(g, '8→7', x + fw / 2, fy + fh * 0.62, { size: 11, font: 'mono', align: 'center', color: badCol(C), weight: 700 });
          }
          T(g, isCfg ? 'config' : String(k + 1).padStart(3, '0'), x + fw / 2, fy + fh + 14, { size: 10, font: 'mono', align: 'center', color: C.muted });
        }
        // detectores
        const dy0 = fy + fh + 30;
        const cols = wide ? 4 : 2, dw = (w - 28 - (cols - 1) * 10) / cols, dh = 74;
        DETS.forEach((d, k) => {
          const x = 14 + (k % cols) * (dw + 10), y = dy0 + Math.floor(k / cols) * (dh + 10);
          const scanEnd = 0.5 + k * 0.55, done = el > scanEnd + 0.5;
          const caught = cat[k] === 1;
          const anyDef = S.d !== 'ok';
          let col = C.muted, verdict = 'varrendo…';
          if (done) { if (!anyDef) { col = okCol(C); verdict = '✓ nada de errado'; } else if (caught) { col = badCol(C); verdict = '● PEGOU o defeito'; } else { col = C.d; verdict = '○ passou batido'; } }
          V.box(g, x, y, dw, dh, { r: 10, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.7), lw: done ? 1.6 : 1 });
          if (!done && el > k * 0.55) { const u = V.clamp((el - k * 0.55) / 1.0, 0, 1); V.line(g, x + 6 + u * (dw - 12), y + 6, x + 6 + u * (dw - 12), y + dh - 6, { color: V.alpha(C.a, 0.8), lw: 2 }); }
          if (done) { const u = ((t * 0.35 + k * 0.25) % 1); V.line(g, x + 6 + u * (dw - 12), y + dh - 8, x + 6 + u * (dw - 12) + 14, y + dh - 8, { color: V.alpha(col, 0.5), lw: 2 }); }
          T(g, clip(g, d, dw - 20, 11.5, 'sans', 700), x + 10, y + 20, { size: 11.5, weight: 700, color: C.ink });
          T(g, clip(g, DSUB[k], dw - 20, 10, 'mono'), x + 10, y + 36, { size: 10, font: 'mono', color: C.muted });
          T(g, clip(g, verdict, dw - 20, 11.5, 'sans', 600), x + 10, y + 58, { size: 11.5, weight: 600, color: col });
        });
        // painel de hash
        const py = dy0 + (wide ? dh + 18 : 2 * (dh + 10) + 8);
        const ph = h - py - 10;
        const hw = wide ? (w - 38) / 2 : w - 28;
        const hx2 = wide ? 14 + hw + 10 : 14, hy2 = wide ? py : py + ph / 2 + 4;
        const hh = wide ? ph : ph / 2 - 4;
        const hexLines = (s, n) => [s.slice(0, n), s.slice(n)];
        const diffDraw = (s, ref, x, y, size) => {
          const cw = tw(g, '0', size, 'mono');
          for (let k = 0; k < s.length; k++) T(g, s[k], x + k * cw, y, { size, font: 'mono', color: s[k] !== ref[k] ? badCol(C) : C.ink, weight: s[k] !== ref[k] ? 700 : 400 });
        };
        // SHA-256 do shard
        V.box(g, 14, py, hw, hh, { r: 10, fill: V.alpha(C.f, 0.06), stroke: V.alpha(C.f, 0.35) });
        T(g, 'shard de demonstração (48 B) · SHA-256', 24, py + 18, { size: 11, weight: 700, color: C.ink });
        const bytesHex = Array.from(S.buf.slice(0, 12)).map((b) => b.toString(16).padStart(2, '0')).join(' ') + ' …';
        T(g, clip(g, bytesHex, hw - 20, 10.5, 'mono'), 24, py + 36, { size: 10.5, font: 'mono', color: C.muted });
        const sz = Math.min(14, (hw - 30) / 32 / 0.62);
        const hl = hexLines(hCur, 32), hr = hexLines(hBase, 32);
        diffDraw(hl[0], hr[0], 24, py + 62, sz); diffDraw(hl[1], hr[1], 24, py + 62 + sz + 6, sz);
        if (hh > 150) { T(g, clip(g, 'o Hub publica lfs.oid = "sha256:' + hBase.slice(0, 12) + '…"', hw - 20, 10.5, 'mono'), 24, py + 62 + 2 * sz + 30, { size: 10.5, font: 'mono', color: C.muted }); T(g, clip(g, 'verify-weights.py: blocos de 1 MiB, 8 threads', hw - 20, 10.5, 'mono'), 24, py + 62 + 2 * sz + 48, { size: 10.5, font: 'mono', color: C.muted }); }
        let nd = 0; for (let k = 0; k < 64; k++) if (hCur[k] !== hBase[k]) nd++;
        T(g, S.flips ? S.flips + ' bit(s) trocado(s) → ' + nd + '/64 dígitos hex mudaram' : 'original (troque um bit para ver a avalanche)', 24, py + hh - 12, { size: 11, color: S.flips ? badCol(C) : C.muted, weight: 600 });
        // SHA-1 git do config
        V.box(g, hx2, hy2, hw, hh, { r: 10, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.35) });
        T(g, 'config.json · SHA-1 de blob git', hx2 + 10, hy2 + 18, { size: 11, weight: 700, color: C.ink });
        const cfgNow = S.d === 'cfg' ? CFG1 : CFG0, gNow = S.d === 'cfg' ? gB1 : gB0;
        T(g, clip(g, 'blob ' + cfgNow.length + '\\0' + cfgNow, hw - 20, 10.5, 'mono'), hx2 + 10, hy2 + 36, { size: 10.5, font: 'mono', color: C.muted });
        const sz2 = Math.min(13, (hw - 30) / 40 / 0.62);
        diffDraw(gNow, gB0, hx2 + 10, hy2 + 62, sz2);
        if (hh > 150) { T(g, 'original: ' + CFG0, hx2 + 10, hy2 + 62 + sz2 + 26, { size: 10.5, font: 'mono', color: C.muted }); T(g, 'agora:    ' + cfgNow, hx2 + 10, hy2 + 62 + sz2 + 44, { size: 10.5, font: 'mono', color: S.d === 'cfg' ? badCol(C) : C.muted }); }
        T(g, S.d === 'cfg' ? 'mesmo tamanho (' + CFG1.length + ' B), hash diferente' : 'o oid que o Hub publica para arquivos git', hx2 + 10, hy2 + hh - 12, { size: 11, color: S.d === 'cfg' ? badCol(C) : C.muted, weight: 600 });
      });
    })();

    /* =====================================================================
       FIGURA 9 — derivar sem copiar, distribuir sem duplicar
       ===================================================================== */
    (function figDerive() {
      const fig = root.querySelector('#fig-derive');
      const S = { mode: 'fp8', jobs: true, down: false, t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'fp8', label: 'FP8-dense (Qwen)' }, { v: 'ovl', label: 'dense overlay (GLM)' }, { v: 'rsync', label: 'worker: rsync' }, { v: 'nfs', label: 'worker: NFS' }], value: 'fp8', onChange: (v) => { S.mode = v; S.t0 = H.t; } });
      UI.toggle(ctl, { label: '--jobs 16 (overlay)', value: true, onChange: (v) => { S.jobs = v; S.t0 = H.t; } });
      UI.toggle(ctl, { label: 'head fora do ar (rsync/NFS)', value: false, onChange: (v) => { S.down = v; } });
      const rng = V.rng(99);
      const M = Array.from({ length: 5 }, (_, i) => Array.from({ length: 8 }, () => +(rng.normal() * (i === 2 ? 2.6 : 0.5)).toFixed(3)));
      const Q = M.map((row) => {
        const amax = Math.max(...row.map(Math.abs)), s = Math.max(amax / 448, 1e-12);
        const q = row.map((x) => LA.num.encode('e4m3', x / s).v);
        const deq = q.map((v) => v * s);
        let num = 0, den = 0; row.forEach((x, k) => { num += (deq[k] - x) ** 2; den += x * x; });
        return { amax, s, q, rel: Math.sqrt(num / Math.max(den, 1e-30)) };
      });
      const grid = (x, y, cols, rows, cw, chh, fn) => { for (let k = 0; k < cols * rows; k++) fn(k, x + (k % cols) * (cw + 4), y + Math.floor(k / cols) * (chh + 4)); };
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        const dist = S.mode === 'rsync' || S.mode === 'nfs';
        stageH(fig, wide ? 450 : (dist ? 600 : 560));
        const el = t - S.t0;
        if (!dist) {
          const fp8 = S.mode === 'fp8';
          const cols = 6, rows = 4, cw = wide ? 26 : 18, chh = wide ? 18 : 14;
          const gw = cols * (cw + 4);
          const lx = 18, rx = wide ? w * 0.5 + 20 : w - gw - 18, gy = 40;
          T(g, clip(g, fp8 ? 'RadixArk (206 shards)' : 'pack TR3 (120 shards)', rx - lx - 10, 11, 'mono', 700), lx, 24, { size: 11, font: 'mono', weight: 700, color: C.ink });
          T(g, clip(g, fp8 ? 'FP8dense / fp8dense-{rev8}' : 'overlay dense-h3', w - rx - 14, 11, 'mono', 700), rx, 24, { size: 11, font: 'mono', weight: 700, color: C.ink });
          const bf = [2, 9, 15, 20];
          grid(lx, gy, cols, rows, cw, chh, (k, x, y) => { const isB = fp8 && bf.includes(k); V.box(g, x, y, cw, chh, { r: 3, fill: V.alpha(isB ? C.c : C.b, 0.3), stroke: V.alpha(isB ? C.c : C.b, 0.7) }); });
          grid(rx, gy, cols, rows, cw, chh, (k, x, y) => {
            const isB = fp8 && bf.includes(k);
            if (isB) { const u = V.clamp(((el % 9) - 1 - bf.indexOf(k) * 0.6) / 1.2, 0, 1); V.box(g, x, y + chh * (1 - u), cw, chh * u, { r: 3, fill: V.alpha(C.c, 0.6) }); V.box(g, x, y, cw, chh, { r: 3, stroke: C.c }); }
            else { g.setLineDash([2, 2]); V.box(g, x, y, cw, chh, { r: 3, stroke: V.alpha(fp8 ? C.e : C.a, 0.85), fill: V.alpha(fp8 ? C.e : C.a, 0.1) }); g.setLineDash([]); }
          });
          if (!fp8) { const nx = rx + gw + 6; if (nx + 26 < w - 10) { V.box(g, nx, gy, 22, rows * (chh + 4) - 4, { r: 4, fill: V.alpha(C.a, 0.5), stroke: C.a }); T(g, '+1', nx + 11, gy + rows * (chh + 4) / 2, { size: 10, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 }); } }
          if (wide) for (let k = 0; k < 24; k += 5) {
            if (fp8 && bf.includes(k)) continue;
            const ay = gy + Math.floor(k / cols) * (chh + 4) + chh / 2;
            bezLine(g, lx + (k % cols) * (cw + 4) + cw, ay, rx + (k % cols) * (cw + 4), ay, { color: V.alpha(fp8 ? C.e : C.a, 0.35), lw: 1, dash: fp8 ? null : [3, 3], dashOffset: -t * 10 });
          }
          const ly = gy + rows * (chh + 4) + 12;
          let ly2 = wrap(g, fp8 ? 'tracejado verde = hard-link para o blob (mesmo inode, 0 B copiados)' : 'tracejado = symlink relativo para cada arquivo do pack', lx, ly, w - 36, 15, { size: 10.5, font: 'mono', color: fp8 ? C.e : C.a, maxLines: 2 });
          ly2 = wrap(g, fp8 ? 'rosa = os 4 shards model-bf16-* reescritos em streaming' : '+1 = o único safetensors novo (lineares densos EXL3)', lx, ly2 + 1, w - 36, 15, { size: 10.5, font: 'mono', color: fp8 ? C.c : C.a, maxLines: 2 });
          const by0 = ly2 + 14;
          if (fp8) {
            const row = Math.floor(el / 1.6) % 5;
            const q = Q[row];
            T(g, clip(g, wide ? 'quantize_rows: s_i = max|W_i·| / 448 → FP8 E4M3 (blocos de 16 MiB de linhas)' : 'quantize_rows: s_i = max|W_i·| / 448', w - 36, 11, 'sans', 600), 18, by0, { size: 11, weight: 600, color: C.ink });
            const mcw = wide ? 50 : Math.min(36, (w - 44) / 8 - 3);
            M.forEach((r, i) => r.forEach((v, j) => {
              const x = 18 + j * (mcw + 3), y = by0 + 12 + i * 22;
              const on = i === row;
              V.box(g, x, y, mcw, 19, { r: 3, fill: V.alpha(v >= 0 ? C.a : C.b, 0.12 + Math.min(0.6, Math.abs(v) / 3)), stroke: on ? C.ink : null, lw: 1.2 });
              if (mcw >= 34) T(g, nBR(v, 2), x + mcw / 2, y + 10, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            }));
            const ix = wide ? 18 + 8 * (mcw + 3) + 16 : 18, iy = wide ? by0 + 28 : by0 + 12 + 5 * 22 + 18;
            const lines = [
              ['linha ' + row + ':  amax = ' + nBR(q.amax, 3), C.ink],
              ['s_i = ' + sciBR(q.s, 3) + '  (weight_scale, F32)', C.d],
              ['q[0..2] = ' + q.q.slice(0, 3).map((v) => nBR(v, Math.abs(v) >= 10 ? 0 : 1)).join('; '), C.ink],
              ['RMSE relativo = ' + nBR(q.rel * 100, 2) + '%  (alarme > 6%)', q.rel < 0.06 ? okCol(C) : badCol(C)],
            ];
            lines.forEach(([s, c], k) => T(g, clip(g, s, w - ix - 14, 11, 'mono', 600), ix, iy + k * 19, { size: 11, font: 'mono', weight: 600, color: c }));
            if (row === 2) T(g, wide ? '← linha com outlier: a escala dela é maior, as outras não sofrem' : '↑ linha com outlier: escala própria', ix, iy + 4 * 19 + 2, { size: 10.5, color: C.d, maxW: w - ix - 14 });
            const sby = wide ? by0 + 136 : iy + 4 * 19 + 24, smax = Math.max(...Q.map((z) => z.s));
            if (sby + 5 * 12 + 14 < h - 24) {
              T(g, 'weight_scale [out] (F32): uma escala por linha', 18, sby, { size: 10.5, font: 'mono', color: C.muted });
              Q.forEach((z, k) => { const bw = (Math.min(w * 0.5, w - 120) - 60) * (z.s / smax); V.box(g, 60, sby + 8 + k * 12, Math.max(2, bw), 9, { r: 3, fill: V.alpha(k === row ? C.d : C.muted, k === row ? 0.8 : 0.35) }); T(g, 's' + k, 44, sby + 15 + k * 12, { size: 9.5, font: 'mono', color: C.muted, base: 'middle' }); });
            }
            if (wide) T(g, clip(g, '202 hard-links · 4 shards reescritos (11,8 GB) · ~2 min · ~1,2 GB de RAM · só CPU', w - 36, 11, 'mono'), 18, h - 14, { size: 11, font: 'mono', color: C.muted });
            else wrap(g, '202 hard-links · 4 shards reescritos (11,8 GB) · ~2 min · ~1,2 GB de RAM · só CPU', 18, h - 30, w - 36, 15, { size: 11, font: 'mono', color: C.muted, maxLines: 2 });
          } else {
            const hubW = 150, hubX = wide ? w * 0.5 - hubW / 2 : w / 2 - hubW / 2, hubY = by0;
            V.box(g, hubX, hubY, hubW, 38, { r: 19, fill: V.alpha(C.b, 0.18), stroke: C.b });
            T(g, 'turboderp@2a30229e', hubX + hubW / 2, hubY + 14, { size: 10, font: 'mono', align: 'center', color: C.ink });
            T(g, 'branch 4.05bpw (Hub)', hubX + hubW / 2, hubY + 28, { size: 10, font: 'mono', align: 'center', color: C.muted });
            const nf = { x: 18, y: hubY + 84, w: w - 36, h: 30 };
            const dur = S.jobs ? 5 : 5 * (1261 / 66);
            const prog = V.clamp(el / dur, 0, 1);
            V.box(g, nf.x, nf.y, nf.w, nf.h, { r: 6, stroke: C.a, fill: V.alpha(C.a, 0.06) });
            V.box(g, nf.x, nf.y, Math.max(2, nf.w * prog), nf.h, { r: 6, fill: V.alpha(C.a, 0.45) });
            T(g, clip(g, 'safetensors novo: lineares densos EXL3 (~5,3 GB) · ' + Math.round(prog * 100) + '%', nf.w - 14, 10.5, 'mono'), nf.x + 8, nf.y + nf.h / 2 + 0.5, { size: 10.5, font: 'mono', base: 'middle', color: C.ink });
            const lanes = S.jobs ? 16 : 1;
            if (prog < 1) for (let k = 0; k < lanes; k++) {
              const u = (t * (S.jobs ? 0.9 : 0.5) + k / lanes) % 1;
              const p = V.along(hubX + hubW / 2 + (k - lanes / 2) * 3, hubY + 38, nf.x + nf.w * ((k + 0.5) / lanes), nf.y, u);
              V.dot(g, p[0], p[1], 2.8, C.b);
            }
            T(g, S.jobs ? '--jobs 16: 66 s medidos (range reads concorrentes)' : 'serial: 1261 s medidos (~1 requisição/s)', 18, nf.y + nf.h + 24, { size: 12, weight: 600, color: S.jobs ? okCol(C) : C.d, maxW: w - 36 });
            const oy = nf.y + nf.h + 50;
            [['pack_profile.py valida cabeçalhos sem baixar pesos:', C.ink], ['trellis I16 [.., 16·bits] · suh [in] · svh [out] · um de mcg/mul1', C.muted], ['drafter DFlash2 6 bpw: 5 camadas, largura 4096, K/V BF16', C.muted], ['revisões = SHA-1 das identidades dos arquivos → refs/glm53-dense-h3', C.muted]].forEach(([l, c], k) => { if (oy + k * 18 < h - 30) T(g, clip(g, l, w - 36, 11, 'mono'), 18, oy + k * 18, { size: 11, font: 'mono', color: c }); });
            T(g, clip(g, '+ índice reescrito + config.json com non_routed_exl3 · nada é quantizado aqui', w - 36, 11, 'mono'), 18, h - 14, { size: 11, font: 'mono', color: C.muted });
          }
        } else {
          const nfs = S.mode === 'nfs', down = S.down;
          let A, B;
          if (wide) { const bw = w * 0.3; A = { x: 18, y: 44, w: bw, h: 210 }; B = { x: w - 18 - bw, y: 44, w: bw, h: 210 }; }
          else { A = { x: 18, y: 36, w: w - 36, h: 110 }; B = { x: 18, y: 250, w: w - 36, h: 110 }; }
          const node = (R, label, col, fill, dl) => {
            V.box(g, R.x, R.y, R.w, R.h, { r: 12, fill: V.alpha(col, 0.08), stroke: col, lw: 1.5 });
            T(g, label, R.x + 12, R.y + 20, { size: 12, weight: 700, color: C.ink });
            const dx = R.x + 12, dy = R.y + 32, dw = R.w - 24, dh = R.h - 62;
            V.box(g, dx, dy, dw, dh, { r: 7, stroke: V.alpha(col, 0.6) });
            if (fill > 0) V.box(g, dx, dy + dh * (1 - fill), dw, dh * fill, { r: 7, fill: V.alpha(col, 0.35) });
            T(g, clip(g, dl, dw - 6, 10.5, 'mono'), dx, R.y + R.h - 12, { size: 10.5, font: 'mono', color: C.muted });
          };
          const wfill = nfs ? 0 : V.clamp(el / 4, 0, 1);
          node(A, down ? 'head (desligado)' : 'head (rank 0)', down ? C.muted : C.a, 1, 'cache HF: ~164 GiB (GLM)');
          node(B, 'worker (rank 1)', C.d, wfill, nfs ? 'disco local do worker: 0 B' : 'cópia local: ' + Math.round(wfill * 164) + ' / 164 GiB');
          const ph = (el % 6) / 6;
          const lanes = nfs ? 8 : 1;
          const p0 = wide ? [A.x + A.w, A.y + A.h / 2] : [A.x + A.w / 2, A.y + A.h];
          const p1 = wide ? [B.x, B.y + B.h / 2] : [B.x + B.w / 2, B.y];
          const active = nfs ? !down && ph < 0.6 : wfill < 1;
          for (let k = 0; k < lanes; k++) {
            const o = -(lanes - 1) * 5 + k * 10;
            const a0 = wide ? [p0[0], p0[1] + o] : [p0[0] + o, p0[1]], a1 = wide ? [p1[0], p1[1] + o] : [p1[0] + o, p1[1]];
            V.line(g, a0[0], a0[1], a1[0], a1[1], { color: V.alpha(C.b, down && nfs ? 0.15 : 0.45), lw: nfs ? 1.4 : 5 });
            if (active) for (let p = 0; p < 3; p++) { const u = (t * (nfs ? 0.8 : 0.5) + p / 3 + k * 0.13) % 1; const q = V.along(a0[0], a0[1], a1[0], a1[1], u); V.dot(g, q[0], q[1], nfs ? 2.4 : 4, C.b); }
          }
          const mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2;
          const lab = nfs ? 'NFSv4.2 · ro · nconnect=8 · rsize=wsize=1 MiB' : 'rsync -a --partial';
          if (wide) { T(g, clip(g, lab, B.x - A.x - A.w - 12, 10.5, 'mono'), mx, my - 54, { size: 10.5, font: 'mono', align: 'center', color: C.b }); T(g, 'link CX7 (ConnectX-7, 200 Gb/s)', mx, my + 56, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); }
          else { T(g, clip(g, lab + ' · CX7', w / 2 - 60 - 18, 10.5, 'mono'), p0[0] + 50, my + 4, { size: 10.5, font: 'mono', color: C.b }); }
          let msg, ok = true;
          if (nfs) { if (down) { ok = false; msg = '✗ sem o head, o worker não tem pesos'; } else msg = ph < 0.6 ? 'cold start: o worker streama os pesos pela CX7' : 'servindo: pesos já na memória, NFS ocioso'; }
          else msg = wfill < 1 ? 'primeira vez: ~164 GiB pelo link p2p' : down ? '✓ nós desacoplados: o worker tem a própria cópia' : '✓ sonda de tamanho ok → marcador .glm53-exl3-synced';
          if (!nfs && wfill >= 1) chip(g, B.x + 20, B.y + 44, '✓ .glm53-exl3-synced', okCol(C), { size: 10.5 });
          const ty = (wide ? A.y + A.h : B.y + B.h) + 34;
          T(g, clip(g, msg, w - 36, 12.5, 'sans', 600), 18, ty, { size: 12.5, weight: 600, color: ok ? (nfs ? C.ink : okCol(C)) : badCol(C) });
          const sub = nfs ? 'Zero disco no worker e troca fácil de checkpoint; o head precisa ficar de pé e cada cold start relê tudo pelo cabo.' : 'Disco em dobro, mas nós independentes e cold start mais rápido; o próximo boot só verifica (marcador + sonda).';
          const ny = wrap(g, sub, 18, ty + 22, w - 36, 16, { size: 11.5, color: C.muted, maxLines: 3 });
          const ny2 = wrap(g, nfs ? 'Recusa montar por 10.0.0.* / 127.*: pela rede de gerência, 164 GiB viram “um trabalho de madrugada”.' : 'DeepSeek (padrão NFS=0): em vez de rsync, o prepare faz um 2º download no worker.', 18, Math.min(ny + 6, h - 30), w - 36, 16, { size: 11, color: C.muted, maxLines: 2 });
          const rows3 = [['GLM', C.glm, 'NFS_SHARE=0 no template, 1 no kit do autor; rsync + marcador'], ['Qwen', C.qwen, 'rsync por padrão; NFS_SHARE=true / --nfs (vllm-fn-nfs)'], ['DeepSeek', C.ds, 'DSPARK_WORKER_HF_NFS=0 (2º download) ou 1 (dspark-nfs)']];
          rows3.forEach(([r, c, d], k) => { const yy = ny2 + 8 + k * 22; if (yy > h - 8) return; chip(g, 18, yy - 13, r, c, { size: 10.5, w: 78, h: 19 }); T(g, clip(g, d, w - 120, 11, 'mono'), 104, yy - 3, { size: 11, font: 'mono', color: C.ink, base: 'middle' }); });
        }
      });
    })();

    /* =====================================================================
       FIGURA 10 — a linha dos pesos, receita a receita
       ===================================================================== */
    (function figPipeline() {
      const fig = root.querySelector('#fig-pipeline');
      const STN = ['Checkpoint', 'Revisão', 'Download', 'Verificação', 'Metadados', 'Distribuição', 'Carga'];
      const DATA = {
        glm: [
          ['Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw', 'retrato 94ae731: EXL3 K4 só nos experts roteados, 120 shards, ~164 GiB; desde 98cf18f o padrão é o …-4bpw-TensorFold (83 shards)'],
          ['MODEL_REVISION 25a44fdb…', 'DFLASH_REVISION dc77ff1c…; fallback brandonmusic se 404 (retrato; hoje 76c0b517, fallback = o mesmo repo)'],
          ['download.sh (só no head)', 'HF_DOWNLOAD_EXCLUDE; SKIP_DOWNLOAD / REFRESH_WEIGHTS'],
          ['EXPECTED_SHARDS 120 (hoje 83)', 'sonda de tamanho no worker; marcador = alegação, não prova'],
          ['quant_method exl3 (overlay exl3.py)', 'tensor_storage descartado; preset dense-h3 opcional'],
          ['NFS_SHARE=1 (kit) ou rsync', 'nconnect=8 pela CX7; nunca pela rede de gerência'],
          ['HF_HUB_OFFLINE=1', 'InstantTensor ~65–70 s a frio (auto: ~290–300 s)'],
        ],
        tf: [
          ['Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw', 'v1.3.2: o mesmo pack EXL3 da lane vLLM (120 shards, ~164 GiB), lido pelo TensorFold; desde a v1.3.3: …-4bpw-TensorFold @078455ff'],
          ['MODEL_REVISION 9eaebb7c…', 'DFLASH2_REVISION bf582e4e…; vazio = main do Hub no 1º download, depois mantido'],
          ['prepare.sh: hf download --revision', 'só no head; sem rede, o snapshot pinado já presente basta; refs/main com printf se faltar'],
          ['tensorfold info + lista caminho/tamanho', 'o motor lê o snapshot antes da cópia; head × worker por find -L, antes e depois'],
          ['EXL3 nos experts + DENSE=q4', 'o TF lê o EXL3 (aviso “experimental”); DENSE=q4 quantiza os densos BF16 na carga, bf16 os serve como estão'],
          ['WORKER_WEIGHTS=copy (rsync -L) ou nfs', 'só os blobs da revisão; NFS por volume Docker ro, os dois ranks no ar em ~2,2 min'],
          ['tensorfold serve …/snapshots/<rev>', 'HF_HUB_OFFLINE=1; 2–6 min para ~80 GiB por Spark; 1ª partida de cada imagem compila kernels'],
        ],
        qwen: [
          ['nvidia/Qwen3.8-Flash-Next-NVFP4', 'BF16 denso + experts NVFP4 + PLE FP8; 11 shards, 133 GB'],
          ['snapshot de refs/main (completo)', 'manifest de hashes salvo com repo e revisão'],
          ['download.sh [--fp8 | org/repo]', 'HF_TOKEN só para o repositório restrito (ABLIT)'],
          ['resolve_snapshot.py (0/1/2)', 'check-weights.sh --verify: ~1 min para 135 GB'],
          ['detect_ple_dtype + --hf-overrides aninhado', 'patch_checkpoint_config: alias .0.→.48., FP8_PB_WO'],
          ['rsync (padrão) ou NFS_SHARE', 'WORKER_MODEL_PATH espelha o caminho do head'],
          ['--load-format safetensors, lazy', '458 s de carga (392 + 67); HF_HUB_OFFLINE=1'],
        ],
        ds: [
          ['deepseek-ai/DeepSeek-V4-Flash-Vision-Exp', 'experts FP4, resto FP8, 3 estágios DSpark; ~157 GiB'],
          ['DSPARK_REVISION 86f746b3…', 'não definido = pin; vazio explícito = tip de main'],
          ['prepare-dspark-model-cache.sh --official', 'timeouts 120/30 s; HF_HUB_DISABLE_XET=1'],
          ['prepare: missing_shards = 0', 'validate-dspark-config avisa se a revisão falta'],
          ['mapper HF→vLLM; E8M0 via view(uint8)', 'encoder dsv4 + --trust-remote-code'],
          ['DSPARK_WORKER_HF_NFS=0 (2º download) ou 1', 'override NFS: caches JIT locais e graváveis'],
          ['HF_HUB_OFFLINE=1, --revision', 'drafter carrega só mtp.* → camadas 43–45'],
        ],
      };
      const S = { r: 'glm', pin: -1 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM' }, { v: 'tf', label: 'GLM·TF' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.r = v; } });
      let pTf = null, pDark = null;
      const pipeTF = (C) => { if (pTf == null || pDark !== C.dark) { pTf = getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d; pDark = C.dark; } return pTf; };
      UI.button(ctl, 'soltar estação', () => { S.pin = -1; });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const wide = w >= 640;
        stageH(fig, wide ? 318 : 420);
        const col = S.r === 'tf' ? pipeTF(C) : C[S.r] || C.a;
        const n = STN.length;
        const x0 = wide ? 56 : 30, x1 = w - (wide ? 56 : 30), ly = wide ? 62 : 70;
        const sx = (k) => x0 + (k / (n - 1)) * (x1 - x0);
        // trilho
        g.lineCap = 'round';
        V.line(g, x0, ly, x1, ly, { color: V.alpha(col, 0.35), lw: 10 });
        V.line(g, x0, ly, x1, ly, { color: col, lw: 3 });
        g.lineCap = 'butt';
        // trem
        const per = 2.8, pos = (t / per) % n, k0 = Math.floor(pos), f = pos - k0;
        const tx = f < 0.7 ? sx(k0) : V.lerp(sx(k0), sx((k0 + 1) % n), V.ease.inOut((f - 0.7) / 0.3));
        const cur = S.pin >= 0 ? S.pin : k0;
        // clique
        if (H.pointer.clicked) for (let k = 0; k < n; k++) if (Math.abs(H.pointer.x - sx(k)) < 20 && Math.abs(H.pointer.y - ly) < 40) S.pin = k;
        STN.forEach((s, k) => {
          const on = k === cur;
          V.dot(g, sx(k), ly, on ? 12 : 9, C.panel || C.bg);
          g.beginPath(); g.arc(sx(k), ly, on ? 12 : 9, 0, Math.PI * 2); g.strokeStyle = col; g.lineWidth = on ? 4 : 3; g.stroke();
          if (on) V.glow(g, sx(k), ly, 26, col, 0.6);
          const above = !wide && k % 2 === 1;
          const lw2 = tw(g, s, wide ? 12 : 10, 'sans', on ? 700 : 500) / 2 + 4;
          T(g, s, V.clamp(sx(k), lw2, w - lw2), above ? ly - 24 : ly + 30, { size: wide ? 12 : 10, weight: on ? 700 : 500, align: 'center', color: on ? C.ink : C.muted });
        });
        V.glow(g, tx, ly, 18, col, 0.9); V.dot(g, tx, ly, 5.5, C.ink);
        // painel da estação
        const d = DATA[S.r][cur];
        const py = ly + (wide ? 60 : 54), pw = w - 28, ph = h - py - 14;
        V.box(g, 14, py, pw, ph, { r: 12, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.55) });
        T(g, (cur + 1) + '. ' + STN[cur] + (S.pin >= 0 ? '  (fixada)' : ''), 28, py + 24, { size: 12.5, weight: 700, color: C.ink });
        T(g, clip(g, d[0], pw - 30, 13, 'mono', 700), 28, py + 52, { size: 13, font: 'mono', weight: 700, color: col });
        const dy = wrap(g, d[1], 28, py + 78, pw - 30, 17, { size: 12.5, color: C.ink, maxLines: 3 });
        // mini-resumo das outras estações
        if (ph > 150) {
          const my = Math.max(py + 124, dy + 14); // descrição de 3 linhas (celular) empurra o resumo
          DATA[S.r].forEach((dd, k) => {
            if (k === cur) return;
            const kk = k < cur ? k : k - 1;
            const cc = wide ? 2 : 1, cw = (pw - 30) / cc;
            const xx = 28 + (kk % cc) * cw, yy = my + Math.floor(kk / cc) * 20;
            if (yy > py + ph - 8) return;
            T(g, clip(g, (k + 1) + ' · ' + dd[0], cw - 10, 10.5, 'mono'), xx, yy, { size: 10.5, font: 'mono', color: C.muted });
          });
        }
      });
    })();
  },
});

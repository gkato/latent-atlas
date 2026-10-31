LA.chapter({
  id: 'tokens-embeddings',
  part: 2,
  order: 5,
  title: 'Tokens, vocabulary and embeddings',
  navTitle: 'Tokens & embeddings',
  subtitle: 'How text becomes integers, what lives in a 250,000-token vocabulary, how a conversation becomes a prompt and how much the two ends of the network cost: the embedding and the lm_head.',
  minutes: 45,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['tokenization', 'BPE', 'vocabulary', 'special tokens', 'chat template', 'embedding', 'lm_head', 'tie_word_embeddings'],
  goals: [
    'Explain how <b>byte-level BPE</b> learns merges and tokenizes text, and why low ids tend to be frequent tokens',
    'Read a real vocabulary: sizes, <b>special tokens</b>, multiple EOS ids and padding for TP',
    'Understand what a <b>chat template</b> decides, the case of the DeepSeek Python encoder and the effect on the prefix cache',
    'Compute the memory and bandwidth of the <b>embedding</b> and the <b>lm_head</b>, tied or not, and why a large vocabulary makes speculative drafting more expensive',
  ],
  prereqs: ['tensores-algebra', 'numeros-no-computador'],
  summary: [
    'Byte-level BPE starts from 256 bytes and repeatedly merges the most frequent neighboring pair; to encode, it applies the merges in the order they were learned. Low id ≈ old merge ≈ common token.',
    'Context, KV cache and speed are counted in tokens, not words: measure with <code style="font-variant-ligatures:none">/tokenize</code> (DeepSeek measured 1.564 tokens per word in English).',
    '<code style="font-variant-ligatures:none">vocab_size</code> = BPE + specials + padding: GLM 154,880, Qwen 248,320, DeepSeek 129,280. With TP=3 the vocabulary goes to a multiple of lcm(64, 3) = 192 (154,944 and 129,408).',
    'Special tokens are the protocol (roles, <code style="font-variant-ligatures:none">&lt;think&gt;</code>, tools, media, masks) and must be single tokens. GLM stops at <code style="font-variant-ligatures:none">&lt;|user|&gt;</code> and <code style="font-variant-ligatures:none">&lt;|observation|&gt;</code>.',
    'The chat template translates messages into a prompt: GLM builds its own (<code style="font-variant-ligatures:none">--chat-template</code>), Qwen uses the checkpoint\'s and DeepSeek uses the Python encoder <code style="font-variant-ligatures:none">encoding_dsv4.py</code> (<code style="font-variant-ligatures:none">--tokenizer-mode deepseek_v4</code>), installed and patched at every boot. The start of the prompt decides the prefix cache.',
    'Embedding = look up one row (\\(d\\cdot 2\\) bytes); lm_head = read \\(V\\cdot d\\cdot 2\\) bytes per step. None of the three models ties the matrices; in Qwen with MTP=3 the lm_head is 26% of the step\'s bytes, which motivates the reduced draft vocabulary.',
  ],
  quiz: [
    { q: 'GLM declares <code style="font-variant-ligatures:none">eos_token_id = [154820, 154827, 154829]</code>. Why are <code style="font-variant-ligatures:none">&lt;|user|&gt;</code> and <code style="font-variant-ligatures:none">&lt;|observation|&gt;</code> on that list?', o: ['To save memory in the KV cache', 'Because the natural end of the assistant\'s turn is the point where the next turn marker would come: the user\'s, or the one for a tool result', 'Because these tokens never appear in training', 'So that the vocabulary is a multiple of 64'], a: 1, e: 'The model learned that, once the answer is done, the next token is the marker of the following turn. Stopping when it shows up ends the response at exactly the right point, including when the assistant made a tool call and is waiting for the <code style="font-variant-ligatures:none">&lt;|observation|&gt;</code>. It has nothing to do with memory or with padding.' },
    { q: 'In decode with a single sequence, roughly how much do DeepSeek-V4\'s embedding and lm_head read from memory per token?', o: ['Both read ~1.06 GB', 'Embedding ~8 KiB (one row of 4096 in BF16); lm_head ~1.06 GB (all 129,280 rows)', 'Embedding ~1.06 GB; lm_head ~8 KiB', 'Both read ~8 KiB'], a: 1, e: 'The embedding is a lookup: it reads only the token\'s row, 4096 × 2 B = 8 KiB. The lm_head needs the score of all 129,280 tokens, so it reads the whole matrix, 129,280 × 4096 × 2 B ≈ 1.06 GB. The ratio between the two is exactly \\(V\\).' },
    { q: 'In Qwen with MTP=3, why does drafting over only 47,149 of the 248,320 ids not change the text the server returns?', o: ['Because the target\'s lm_head is also reduced', 'Because every draft is verified by the target model: a token that is not on the list just becomes a rejected draft', 'Because the 47k ids cover 100% of any text', 'Because sampling is always greedy'], a: 1, e: 'The reduced vocabulary only affects the drafter\'s <code style="font-variant-ligatures:none">get_top_tokens</code>; the target\'s <code style="font-variant-ligatures:none">compute_logits</code> keeps the full vocabulary. If the right token is not on the list, the draft misses and is rejected at verification: it costs acceptance, never correctness. Inside <code style="font-variant-ligatures:none">get_top_tokens</code>, each GPU finds the argmax in its reduced slice and only the (value, id) pairs cross the all-gather. The measured gain was 9.6% on average (and 8.4% in a rerun), with the same tokens accepted per step.' },
    { q: 'With TP=3 and <code style="font-variant-ligatures:none">padding_size = lcm(64, tp)</code> (the least common multiple of 64 and the TP), what does DeepSeek\'s 129,280-token vocabulary become?', o: ['129,280 (it is already divisible by 3)', '129,344', '129,408', '129,536'], a: 2, e: 'lcm(64, 3) = 192. ⌈129,280 / 192⌉ = 674, and 674 × 192 = 129,408, or 43,136 rows per rank. 129,280 is not divisible by 3 (its digits add up to 22), and 129,344 is a multiple of 64 but not of 3.' },
    { q: 'What happens if the DeepSeek recipe\'s entrypoint does not find <code style="font-variant-ligatures:none">encoding/encoding_dsv4.py</code>?', o: ['vLLM uses a generic Jinja template and everything works the same', 'The boot aborts with a fatal error', 'The boot emits a WARN and skips the encoder copy, the effort-mapping fix and fix #21: the server comes up with the wrong semantics', 'The model switches to the V3 tokenizer'], a: 2, e: 'The encoder installation is “always, if the file exists”. Without it, the compose only warns and moves on, and the old wrapper keeps mapping efforts to <code style="font-variant-ligatures:none">high</code> and poisoning tool arguments. That is why the file has to exist on the head and on the worker.' },
    { q: 'Why, in a byte-level BPE vocabulary, do low ids tend to be frequent tokens?', o: ['Because of an arbitrary Hugging Face convention', 'Because each merge gets the next free id, and the first merges learned are the most frequent pairs in the corpus', 'Because vLLM reorders the vocabulary by frequency when loading it', 'Because low ids take fewer bytes in memory'], a: 1, e: 'BPE training merges the most frequent pairs first, and each merge gets the next id. The Qwen recipe uses this as an approximation in the <code style="font-variant-ligatures:none">--fill-to-size</code> of the draft vocabulary generator, and the script itself warns that it is a proxy, not a measurement.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const FC = {};
    const F = (s, f, w) => { const k = s + (f || 'sans') + (w || 400); return FC[k] || (FC[k] = V.font(s, f || 'sans', w || 400)); };
    const nf = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const gib = (b) => (b >= 0.9 * 2 ** 30 ? nf(b / 2 ** 30, 2) + ' GiB' : b >= 2 ** 20 ? nf(b / 2 ** 20, 1) + ' MiB' : b >= 1024 ? nf(b / 1024, 1) + ' KiB' : nf(b) + ' B');
    const gb = (b) => (b >= 1e9 ? nf(b / 1e9, 2) + ' GB' : b >= 1e6 ? nf(b / 1e6, 1) + ' MB' : b >= 1e3 ? nf(b / 1e3, 1) + ' kB' : nf(b) + ' B');
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const lcm = (a, b) => (a * b) / gcd(a, b);
    const narrow = (w) => w < 640;
    const tall = (fig, px) => { if (window.innerWidth < 640) { const st = fig.querySelector('.la-fig-stage'); if (st) st.style.height = px + 'px'; } };

    function chipW(g, label, size) { g.font = F(size, 'mono', 500); return g.measureText(label).width + 12; }
    function style(C, kind) {
      switch (kind) {
        case 'sp': return { fill: V.alpha(C.b, 0.22), stroke: C.b };
        case 'gen': return { fill: V.alpha(C.a, 0.28), stroke: C.a };
        case 'eff': return { fill: V.alpha(C.d, 0.26), stroke: C.d };
        case 'tool': return { fill: V.alpha(C.c, 0.22), stroke: C.c };
        case 'img': return { fill: V.alpha(C.f, 0.22), stroke: C.f };
        case 'eos': return { fill: V.alpha(C.red, 0.2), stroke: C.red };
        case 'ghost': return { fill: 'rgba(0,0,0,0)', stroke: V.alpha(C.muted, 0.6) };
        default: return { fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.45) };
      }
    }
    function drawChip(g, C, x, y, w, h, label, kind, o = {}) {
      const s = style(C, kind);
      if (kind === 'ghost') { g.setLineDash([3, 3]); }
      V.box(g, x, y, w, h, { r: 6, fill: o.fill || s.fill, stroke: o.stroke || s.stroke, lw: o.lw || 1.2 });
      g.setLineDash([]);
      if (o.sub != null) {
        V.text(g, label, x + w / 2, y + h / 2 - 6, { size: o.size || 11.5, font: 'mono', weight: 500, color: C.ink, align: 'center', base: 'middle' });
        V.text(g, o.sub, x + w / 2, y + h / 2 + 8, { size: 11, font: 'mono', color: C.muted, align: 'center', base: 'middle' });
      } else {
        V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: 'mono', weight: 500, color: o.color || C.ink, align: 'center', base: 'middle' });
      }
    }
    function flow(g, items, x0, y0, maxW, size, lh, gap) {
      let x = x0, y = y0;
      const out = [];
      items.forEach((it) => {
        const w = Math.max(chipW(g, it.label, size), it.sub ? chipW(g, it.sub, 11) : 0, 22);
        if (x + w > x0 + maxW && x > x0) { x = x0; y += lh; }
        out.push({ x, y, w, it });
        x += w + gap;
      });
      return { items: out, bottom: y + lh };
    }

    /* =====================================================================
       Figure 1 — The token ruler (byte-level BPE trained live)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-regua');
      if (!fig) return;
      tall(fig, 560);
      const enc = new TextEncoder();
      const dec = new TextDecoder('utf-8', { fatal: true });
      const PRE = /[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
      const pretok = (s) => s.match(PRE) || [];
      const PT = 'A tokenização não é mágica: é estatística. O tokenizador aprende com o texto e junta os pares mais frequentes. A tokenização transforma texto em números, e cada número vira um vetor. Não é mágica, é compressão: palavras frequentes viram um token só, palavras raras viram pedaços. O modelo não lê letras, o modelo lê tokens. A atenção compara os tokens e o modelo prevê o próximo token. Não há mágica na tokenização, há contagem de pares. ';
      const EN = 'Tokenization is not magic: it is statistics. The tokenizer learns from text and merges the most frequent pairs. Tokenization turns text into numbers, and every number becomes a vector. It is not magic, it is compression: frequent words become one token, rare words become pieces. The model does not read letters, the model reads tokens. ';
      const PY = 'for i in range(10):\n    print(i)\nfor token in tokens:\n    print(token)\ndef tokenize(text):\n    return [vocab[t] for t in text.split()]\nfor i in range(len(tokens)):\n    print(tokens[i])\n';
      const CORPUS = PT.repeat(3) + EN.repeat(3) + PY.repeat(3);
      // ---- training (the Portuguese part of the corpus is intentional: it exercises multi-byte UTF-8)
      const counts = new Map();
      pretok(CORPUS).forEach((p) => counts.set(p, (counts.get(p) || 0) + 1));
      const vocab = [];
      for (let b = 0; b < 256; b++) vocab.push([b]);
      const words = Array.from(counts, ([p, c]) => ({ ids: Array.from(enc.encode(p)), c }));
      const rank = new Map();
      const mergeOf = [];
      const KEY = 1 << 16;
      for (let k = 0; k < 2000; k++) {
        const pc = new Map();
        for (const w of words) for (let i = 0; i < w.ids.length - 1; i++) { const key = w.ids[i] * KEY + w.ids[i + 1]; pc.set(key, (pc.get(key) || 0) + w.c); }
        let best = -1, bc = 1;
        for (const [key, c] of pc) if (c > bc || (c === bc && best >= 0 && key < best)) { best = key; bc = c; }
        if (best < 0) break;
        const a = Math.floor(best / KEY), b = best % KEY, id = vocab.length;
        vocab.push(vocab[a].concat(vocab[b]));
        rank.set(best, [k, id]);
        mergeOf.push([a, b]);
        for (const w of words) {
          const o = [];
          for (let i = 0; i < w.ids.length; i++) { if (i < w.ids.length - 1 && w.ids[i] === a && w.ids[i + 1] === b) { o.push(id); i++; } else o.push(w.ids[i]); }
          w.ids = o;
        }
      }
      const M = vocab.length - 256;
      const encode = (text, K) => {
        const out = [];
        for (const p of pretok(text)) {
          const ids = Array.from(enc.encode(p));
          for (;;) {
            let br = Infinity, bi = -1;
            for (let i = 0; i < ids.length - 1; i++) { const r = rank.get(ids[i] * KEY + ids[i + 1]); if (r && r[0] < K && r[0] < br) { br = r[0]; bi = i; } }
            if (bi < 0) break;
            ids.splice(bi, 2, rank.get(ids[bi] * KEY + ids[bi + 1])[1]);
          }
          out.push(...ids);
        }
        return out;
      };
      const byteLabel = (b) => (b === 32 ? '␣' : b === 10 ? '↵' : b > 32 && b < 127 ? String.fromCharCode(b) : b.toString(16).toUpperCase().padStart(2, '0'));
      const tokLabel = (id) => {
        const bytes = vocab[id];
        try { return dec.decode(new Uint8Array(bytes)).replace(/ /g, '␣').replace(/\n/g, '↵'); } catch (e) { return bytes.map(byteLabel).join('·'); }
      };
      const SENT = {
        pt: 'A tokenização não é mágica.',
        en: 'Tokenization is not magic.',
        py: 'for i in range(10):\n    print(i)',
        nv: 'Retokenizing the vectors.',
      };
      const curves = {};
      Object.keys(SENT).forEach((k) => { curves[k] = []; for (let K = 0; K <= M; K++) curves[k].push(encode(SENT[k], K).length); });
      const S = { s: 'pt', K: Math.min(40, M), play: false };
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Sentence', options: [{ v: 'pt', label: 'Portuguese' }, { v: 'en', label: 'English' }, { v: 'py', label: 'Python' }, { v: 'nv', label: 'new word' }], value: 'pt', onChange: (v) => { S.s = v; } });
      const sl = UI.slider(ctl, { label: 'Merges learned', min: 0, max: M, step: 1, value: S.K, fmt: (v) => String(v), onInput: (v) => { S.K = v; S.play = false; } });
      UI.button(ctl, '▶ animate', () => { S.play = true; S.K = 0; sl.input.value = 0; }, { primary: true });
      const rT = UI.readout(ctl, 'tokens', '—');
      const rF = UI.readout(ctl, 'tokens/word', '—');
      const rV = UI.readout(ctl, 'vocabulary', '—');
      let lastK = -1, lastS = '';
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        if (S.play && !LA.paused) {
          S.K = Math.min(M, S.K + dt * Math.max(20, M / 6));
          const k = Math.round(S.K);
          sl.input.value = k; sl.el.querySelector('.val').textContent = String(k);
          if (S.K >= M) S.play = false;
        }
        const K = Math.round(S.K);
        const text = SENT[S.s];
        const bytes = Array.from(enc.encode(text));
        const toks = encode(text, K);
        const pieces = pretok(text);
        const nWords = (text.match(/[\p{L}\p{N}]+/gu) || []).length;
        if (K !== lastK || S.s !== lastS) {
          rT.set(String(toks.length)); rF.set(nf(toks.length / nWords, 2)); rV.set('256 + ' + K + ' = ' + (256 + K));
          lastK = K; lastS = S.s;
        }
        const nar = narrow(w);
        const lw = nar ? w - 32 : w * 0.58;
        const x0 = 16;
        let y = 22;
        // bytes
        V.text(g, 'UTF-8 BYTES · ' + bytes.length, x0, y, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        let L = flow(g, bytes.map((b) => ({ label: byteLabel(b) })), x0, y + 8, lw, 11, 24, 3);
        L.items.forEach((p) => { const hex = p.it.label.length === 2 && /^[0-9A-F]{2}$/.test(p.it.label) && !/^[A-Za-z0-9]$/.test(p.it.label); drawChip(g, C, p.x, p.y, p.w, 20, p.it.label, hex ? 'img' : 'txt', { size: 11 }); });
        y = L.bottom + 18;
        // BPE
        V.text(g, 'BPE · ' + K + ' merges · ' + toks.length + ' tokens', x0, y, { size: 11, font: 'mono', color: C.a, weight: 600 });
        L = flow(g, toks.map((id) => ({ label: tokLabel(id), id })), x0, y + 8, lw, 12, 29, 4);
        L.items.forEach((p, i) => {
          const col = C.series[i % 5];
          const isNew = p.it.id >= 256 && p.it.id - 256 === K - 1;
          drawChip(g, C, p.x, p.y, p.w, 24, p.it.label, 'txt', { size: 12, fill: V.alpha(col, 0.2), stroke: V.alpha(col, 0.85), lw: isNew ? 2.4 : 1.2 });
          if (isNew) V.glow(g, p.x + p.w / 2, p.y + 12, 18, C.a, 0.5 + 0.3 * Math.sin(t * 5));
        });
        y = L.bottom + 18;
        // floor
        V.text(g, 'PRE-TOKENS (the floor) · ' + pieces.length, x0, y, { size: 11, font: 'mono', color: C.b, weight: 600 });
        L = flow(g, pieces.map((p) => ({ label: p.replace(/ /g, '␣').replace(/\n/g, '↵') })), x0, y + 8, lw, 11.5, 27, 4);
        L.items.forEach((p) => drawChip(g, C, p.x, p.y, p.w, 22, p.it.label, 'sp', { size: 11.5 }));
        const yb = L.bottom;
        // latest merges learned (the vocabulary growing)
        const listTop = yb + 22, listBot = nar ? h - 170 : h - 12;
        const nList = Math.max(0, Math.min(K, Math.floor((listBot - listTop - 8) / 19)));
        if (nList > 0) {
          V.text(g, 'LATEST MERGES LEARNED', x0, listTop, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          for (let i = 0; i < nList; i++) {
            const id = 256 + K - 1 - i;
            const rk = mergeOf[id - 256];
            V.text(g, id + ':  ' + tokLabel(rk[0]) + '  +  ' + tokLabel(rk[1]) + '  →  ' + tokLabel(id), x0, listTop + 18 + i * 19, { size: 12, font: 'mono', color: i === 0 ? C.a : C.muted, alpha: Math.max(0.35, 1 - i * 0.12), maxW: lw });
          }
        }
        // curve
        let px, py, pw, ph;
        if (nar) { px = 46; py = Math.max(yb + 26, h - 150); pw = w - 66; ph = h - py - 30; }
        else { px = w * 0.64 + 30; py = 34; pw = w - px - 20; ph = h - 90; }
        if (ph < 50) return;
        const cv = curves[S.s];
        const yMax = cv[0] * 1.08;
        const X = (k) => px + (k / M) * pw, Y = (n) => py + ph - (n / yMax) * ph;
        g.strokeStyle = C.grid; g.lineWidth = 1;
        g.beginPath(); g.moveTo(px, py); g.lineTo(px, py + ph); g.lineTo(px + pw, py + ph); g.stroke();
        V.text(g, 'tokens in the sentence', px, py - 12, { size: 11, color: C.muted });
        V.text(g, 'merges learned →', px + pw / 2, py + ph + 16, { size: 11, color: C.muted, align: 'center' });
        V.text(g, '0', px, py + ph + 16, { size: 11, color: C.muted, font: 'mono' });
        V.text(g, String(M), px + pw, py + ph + 16, { size: 11, color: C.muted, font: 'mono', align: 'right' });
        V.line(g, px, Y(bytes.length), px + pw, Y(bytes.length), { color: V.alpha(C.muted, 0.6), dash: [4, 4], lw: 1 });
        V.text(g, 'ceiling: ' + bytes.length + ' bytes', px + pw, Y(bytes.length) - 5, { size: 11, color: C.muted, align: 'right' });
        V.line(g, px, Y(pieces.length), px + pw, Y(pieces.length), { color: C.b, dash: [4, 4], lw: 1.2 });
        V.text(g, 'floor: ' + pieces.length, px + pw, Y(pieces.length) - 5, { size: 11, color: C.b, align: 'right' });
        g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath();
        cv.forEach((n, k) => { const xx = X(k), yy = Y(n); if (k === 0) g.moveTo(xx, yy); else g.lineTo(xx, yy); });
        g.stroke();
        const mx = X(K), my = Y(cv[K]);
        V.line(g, mx, py, mx, py + ph, { color: V.alpha(C.a, 0.35), dash: [2, 4] });
        V.glow(g, mx, my, 14 + 3 * Math.sin(t * 4), C.a, 0.8);
        V.dot(g, mx, my, 4.5, C.a);
        V.text(g, cv[K] + ' tokens', Math.min(mx + 8, px + pw - 60), my - 10, { size: 12, font: 'mono', weight: 600, color: C.ink });
      });
    })();

    /* =====================================================================
       Figure 2 — The merge factory (stepper)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-bpe');
      if (!fig) return;
      tall(fig, 620);
      const WORDS = [['gato', 5], ['gata', 4], ['gatos', 3], ['pato', 6], ['pata', 2], ['mato', 3]];
      const states = [WORDS.map(([w]) => w.split(''))];
      const pairsAt = [];
      const merges = [];
      const countPairs = (seg) => {
        const c = new Map();
        seg.forEach((s, wi) => { for (let i = 0; i < s.length - 1; i++) { const k = s[i] + '·' + s[i + 1]; c.set(k, (c.get(k) || 0) + WORDS[wi][1]); } });
        return Array.from(c).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
      };
      for (let k = 0; k < 7; k++) {
        const seg = states[k];
        const pc = countPairs(seg);
        pairsAt.push(pc);
        const [a, b] = pc[0][0].split('·');
        merges.push({ a, b, ab: a + b, id: 256 + k, n: pc[0][1] });
        states.push(seg.map((s) => { const o = []; for (let i = 0; i < s.length; i++) { if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]); } return o; }));
      }
      pairsAt.push(countPairs(states[7]));
      // encoding of "matos"
      const trace = [['m', 'a', 't', 'o', 's']];
      for (;;) {
        const s = trace[trace.length - 1];
        let br = 99, bi = -1;
        for (let i = 0; i < s.length - 1; i++) { const r = merges.findIndex((m) => m.a === s[i] && m.b === s[i + 1]); if (r >= 0 && r < br) { br = r; bi = i; } }
        if (bi < 0) break;
        const o = s.slice(); o.splice(bi, 2, merges[br].ab); o.rule = br; trace.push(o);
      }
      const PRE = /[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
      const SENT = 'In 2026, GLM reads 1048576 tokens of context.';
      const PIECES = SENT.match(PRE);
      const CODE = 'x = sum(12345, y)';
      const steps = [
        { t: 'It all starts with the base symbols', d: 'Every word in the corpus is written in the base alphabet. In a byte-level BPE these would be the 256 bytes (ids 0–255); here we use letters so it reads better. The number next to each word is its frequency in the corpus.' },
        { t: 'Count the neighboring pairs', d: 'For each pair of adjacent symbols, add up the frequencies of the words it appears in. <code style="font-variant-ligatures:none">a·t</code> is in all six words: 5 + 4 + 3 + 6 + 2 + 3 = <b>23</b>.' },
        { t: 'Merge the champion', d: 'The most frequent pair becomes a new symbol with the next free id (256), and the rule <code style="font-variant-ligatures:none">a + t → at</code> goes into the merge list. Every occurrence of <code style="font-variant-ligatures:none">a·t</code> in the corpus is replaced.' },
        { t: 'Recount and repeat', d: 'With <code style="font-variant-ligatures:none">at</code> in the alphabet, the pairs change: now <code style="font-variant-ligatures:none">at·o</code> (17) leads, followed by <code style="font-variant-ligatures:none">g·at</code> (12) and <code style="font-variant-ligatures:none">p·at</code> (8). Each round adds one token to the vocabulary and shortens the corpus.' },
        { t: 'The vocabulary is the merge list', d: 'After 7 merges, the vocabulary has 256 + 7 symbols. The order of the list is the priority order: lower ids are older merges, that is, more frequent pairs. This is the heuristic behind the Qwen recipe\'s <code style="font-variant-ligatures:none">--fill-to-size</code>.' },
        { t: 'Encoding a new word', d: '“matos” is not in the corpus. BPE starts from the letters and applies the oldest merge available: <code style="font-variant-ligatures:none">a+t</code> (256), then <code style="font-variant-ligatures:none">at+o</code> (257). <code style="font-variant-ligatures:none">m+ato</code> was never learned, so the result is <b>m · ato · s</b>: 3 tokens and no “unknown”.' },
        { t: 'First of all: pre-tokenization', d: 'A real tokenizer cuts the text with a regex before BPE (here, in the GPT-4 style that GLM uses): the space sticks to the following word and digits are grouped up to 3 at a time. Merges never cross these cuts.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7000 });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        const s = st.i, el = st.elapsed();
        const nar = narrow(w);
        // current state and transition
        let k = 0, tr = null;
        if (s === 2) { const u = Math.min(1, el / 1.6); tr = { from: 0, u }; k = u < 0.5 ? 0 : 1; }
        else if (s === 3) { const per = 0.95, m = Math.min(5.999, el / per); const kk = 1 + Math.floor(m); tr = { from: kk, u: m - Math.floor(m) }; k = tr.u < 0.5 ? kk : kk + 1; if (el > per * 6) { k = 7; tr = null; } }
        else if (s >= 4) k = 7;
        const seg = states[k];
        const mergingPair = tr && tr.u < 0.5 ? merges[tr.from] : null;
        const justMerged = tr && tr.u >= 0.5 ? merges[tr.from] : null;
        if (s === 6) {
          // full-width pre-tokenization
          const EX = [[SENT, PIECES], [CODE, CODE.match(PRE)]];
          let y = 26;
          EX.forEach(([txt, pcs], k) => {
            V.text(g, k === 0 ? 'PRE-TOKENIZATION (GPT-4-style regex) · text' : '… and code', 16, y, { size: 11, font: 'mono', color: C.muted, weight: 600 });
            V.text(g, txt.replace(/\n/g, '↵'), 16, y + 26, { size: nar ? 13 : 15, font: 'mono', color: C.ink, maxW: w - 32 });
            const L = flow(g, pcs.map((p) => ({ label: p.replace(/ /g, '␣').replace(/\n/g, '↵') })), 16, y + 42, w - 32, 14, 42, 6);
            const n = Math.min(L.items.length, Math.floor((el - k * 2.2) / 0.16) + 1);
            L.items.slice(0, Math.max(0, n)).forEach((p, i) => {
              const isNum = /\d/.test(p.it.label);
              const col = isNum ? C.d : /^[␣]?[\p{L}]/u.test(p.it.label) ? C.series[i % 2 ? 0 : 4] : C.c;
              drawChip(g, C, p.x, p.y, p.w, 32, p.it.label, 'txt', { size: 14, fill: V.alpha(col, 0.2), stroke: col });
            });
            y = L.bottom + 22;
          });
          V.text(g, '“2026” → 202 · 6    “1048576” → 104 · 857 · 6', 16, y + 2, { size: 13, font: 'mono', color: C.d, weight: 600, maxW: w - 32 });
          V.text(g, 'the space sticks to the next word: “␣tokens” ≠ “tokens”; punctuation becomes its own piece', 16, y + 24, { size: 13, color: C.ink, maxW: w - 32 });
          V.text(g, 'BPE tokenizes each piece separately: no merge crosses a cut', 16, y + 46, { size: 13, color: C.muted, maxW: w - 32 });
          return;
        }
        // left panel: corpus
        const lx = 16, lwid = nar ? w - 32 : w * 0.46;
        V.text(g, 'CORPUS (word · frequency)', lx, 24, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const rowH = nar ? 40 : 46, y0 = 40;
        seg.forEach((sym, wi) => {
          const y = y0 + wi * rowH;
          let x = lx;
          sym.forEach((c, ci) => {
            const cw = chipW(g, c, 14) + 4;
            const hl = mergingPair && ((c === mergingPair.a && sym[ci + 1] === mergingPair.b) || (c === mergingPair.b && sym[ci - 1] === mergingPair.a));
            const isNew = justMerged && c === justMerged.ab;
            const len = c.length;
            const col = len >= 4 ? C.a : len === 3 ? C.e : len === 2 ? C.b : C.muted;
            const sc = isNew ? 1 + 0.18 * Math.max(0, 1 - (tr.u - 0.5) * 4) : 1;
            g.save(); g.translate(x + cw / 2, y + 15); g.scale(sc, sc);
            drawChip(g, C, -cw / 2, -15, cw, 30, c, 'txt', { size: 14, fill: V.alpha(col, len > 1 ? 0.25 : 0.1), stroke: hl ? C.c : V.alpha(col, 0.9), lw: hl ? 2.6 : 1.2 });
            g.restore();
            if (hl) V.glow(g, x + cw / 2, y + 15, 22, C.c, 0.4 + 0.3 * Math.sin(t * 8));
            x += cw + 5;
          });
          V.text(g, '×' + WORDS[wi][1], lx + lwid - 4, y + 19, { size: 13, font: 'mono', color: C.muted, align: 'right' });
        });
        // right panel
        const rx = nar ? 16 : w * 0.52, ry = nar ? y0 + 6 * rowH + 18 : 24, rw = nar ? w - 32 : w * 0.48 - 16;
        const rh = h - ry - 10;
        if (s === 0) {
          V.text(g, 'BASE ALPHABET: 256 BYTES (ids 0–255)', rx, ry, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          const n = 16, cs = Math.min(15, (rw - 10) / n, (rh - 40) / n);
          for (let i = 0; i < 256; i++) {
            const r = Math.floor(i / n), c = i % n;
            const ascii = i >= 97 && i <= 122;
            const on = 'gatopms'.includes(String.fromCharCode(i));
            g.fillStyle = on ? V.alpha(C.a, 0.8) : ascii ? V.alpha(C.b, 0.35) : V.alpha(C.muted, 0.14 + 0.08 * Math.sin(t * 2 + i * 0.3));
            V.rr(g, rx + c * (cs + 1), ry + 14 + r * (cs + 1), cs, cs, 2); g.fill();
          }
          V.text(g, 'green: letters used in the corpus · lilac: a–z', rx, ry + 14 + n * (cs + 1) + 18, { size: 11.5, color: C.muted });
        } else if (s <= 3) {
          const pc = pairsAt[Math.min(k, 7)];
          V.text(g, 'PAIR COUNTS (weighted)', rx, ry, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          const top = pc.slice(0, 6), mx = top.length ? top[0][1] : 1;
          const bh = Math.min(26, (rh - 70) / 6 - 6);
          top.forEach(([p, n], i) => {
            const y = ry + 16 + i * (bh + 7);
            const bw = ((rw - 110) * n) / mx;
            const win = i === 0;
            g.fillStyle = win ? V.alpha(C.c, 0.55 + 0.2 * Math.sin(t * 5)) : V.alpha(C.b, 0.35);
            V.rr(g, rx + 70, y, Math.max(3, bw), bh, 5); g.fill();
            V.text(g, p, rx + 62, y + bh / 2 + 1, { size: 13, font: 'mono', color: C.ink, align: 'right', base: 'middle', weight: win ? 700 : 400 });
            V.text(g, String(n), rx + 76 + bw, y + bh / 2 + 1, { size: 12, font: 'mono', color: C.ink, base: 'middle' });
          });
          const done = merges.slice(0, k);
          const yy = ry + 16 + 6 * (bh + 7) + 14;
          V.text(g, 'MERGE LIST', rx, yy, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          done.forEach((m, i) => V.text(g, `${m.id}: ${m.a} + ${m.b} → ${m.ab}`, rx + (i % 2) * (rw / 2), yy + 18 + Math.floor(i / 2) * 17, { size: 12, font: 'mono', color: C.a }));
        } else if (s === 4) {
          V.text(g, 'VOCABULARY = 256 BYTES + MERGES', rx, ry, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          const rH = Math.min(30, (rh - 40) / 8);
          V.text(g, 'id', rx, ry + 22, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'rule', rx + 56, ry + 22, { size: 11, font: 'mono', color: C.muted });
          V.text(g, 'count', rx + rw - 4, ry + 22, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          merges.forEach((m, i) => {
            const y = ry + 30 + i * rH;
            const a = Math.min(1, Math.max(0, (el - i * 0.25) * 3));
            g.globalAlpha = a;
            g.fillStyle = V.alpha(C.a, 0.08 + 0.12 * (1 - i / 7)); V.rr(g, rx - 4, y, rw, rH - 4, 6); g.fill();
            V.text(g, String(m.id), rx + 2, y + rH / 2, { size: 13, font: 'mono', color: C.a, base: 'middle', weight: 700 });
            V.text(g, `${m.a} + ${m.b} → ${m.ab}`, rx + 56, y + rH / 2, { size: 13, font: 'mono', color: C.ink, base: 'middle' });
            V.text(g, String(m.n), rx + rw - 10, y + rH / 2, { size: 13, font: 'mono', color: C.ink, base: 'middle', align: 'right' });
            g.globalAlpha = 1;
          });
          V.arrow(g, rx + rw + 2, ry + 36, rx + rw + 2, ry + 30 + 7 * rH - 8, { color: V.alpha(C.muted, 0.6), head: 7 });
        } else if (s === 5) {
          V.text(g, 'ENCODING “matos” (not in the corpus)', rx, ry, { size: 11, font: 'mono', color: C.muted, weight: 600 });
          const n = Math.min(trace.length, Math.floor(el / 1.1) + 1);
          trace.slice(0, n).forEach((sym, i) => {
            const y = ry + 18 + i * 56;
            let x = rx;
            sym.forEach((c) => { const cw = chipW(g, c, 14) + 4; const col = c.length >= 3 ? C.a : c.length === 2 ? C.b : C.muted; drawChip(g, C, x, y, cw, 30, c, 'txt', { size: 14, fill: V.alpha(col, c.length > 1 ? 0.25 : 0.1), stroke: col }); x += cw + 5; });
            const lab = i === 0 ? 'base symbols only' : `apply ${merges[sym.rule].id}: ${merges[sym.rule].a}+${merges[sym.rule].b}`;
            V.text(g, lab, x + 8, y + 19, { size: 12, font: 'mono', color: i === 0 ? C.muted : C.a });
          });
          if (n === trace.length) V.text(g, 'no merges left → [m, ato, s] = 3 tokens', rx, ry + 18 + trace.length * 56 + 6, { size: 13, color: C.ink, weight: 600 });
        }
      });
      H.S.st = st;
    })();

    /* =====================================================================
       Figure 3 — The vocabulary subway
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-mapa');
      if (!fig) return;
      tall(fig, 640);
      const MAPS = {
        glm: {
          V: 154880, tp3: 154944, regions: [[0, 154820, 'regular BPE (154,820)', 'txt'], [154820, 154838, 'special (18)', 'sp'], [154838, 154856, 'added, not special (18)', 'tool'], [154856, 154880, 'padding (24)', 'pad']],
          st: [
            [1879, '" world"', 'common token: 2nd id of "hello world"'],
            [14978, '"hello"', 'common token: "hello world" → [14978, 1879]'],
            [154820, '<|endoftext|>', 'EOS and PAD (left padding)'],
            [154822, '[gMASK]*', 'opens every template prompt'],
            [154824, '<sop>*', 'right after [gMASK]: start of text'],
            [154826, '<|system|>*', 'system role (and the effort line)'],
            [154827, '<|user|>', 'user role · also EOS'],
            [154828, '<|assistant|>*', 'assistant role · generation prompt'],
            [154829, '<|observation|>', 'tool result · also EOS'],
            [154830, '<|begin_of_image|>', 'image_start_token_id: opens an image'],
            [154841, '<think>', 'opens reasoning · added, not special'],
            [154842, '</think>', 'closes reasoning · not special: stays in the text'],
            [154843, '<tool_call>', 'opens a tool call · single token'],
            [154851, '/nothink', 'turns reasoning off'],
            [154854, '<|image|>', 'placeholder → N vision tokens'],
            [154855, '<|video|>', 'video placeholder'],
            [154856, 'DFlash2 mask', 'blank position for the drafter'],
          ],
        },
        qwen: {
          V: 248320, tp3: 248448, tp3hyp: true, tp: 124160, regions: [[0, 248044, 'regular BPE (inferred)', 'txt'], [248044, 248077, 'special/added (33)', 'sp'], [248077, 248320, 'padding (243, inferred)', 'pad']],
          st: [
            [124160, 'TP=2 boundary', 'rank 0: ids 0–124,159 · rank 1: 124,160–248,319', 'tp'],
            [248044, '<|endoftext|>', 'BOS = EOS in config · separates documents in the PLE'],
            [248045, '<|im_start|>', 'opens a turn (ChatML)'],
            [248046, '<|im_end|>', 'closes a turn · tokenizer EOS'],
            [248053, '<|vision_start|>', 'start of a vision block'],
            [248054, '<|vision_end|>', 'end of a vision block'],
            [248055, '<|vision_pad|>', 'vision padding'],
            [248056, '<|image_pad|>', 'image placeholder → masked_scatter'],
            [248057, '<|video_pad|>', 'video placeholder'],
            [248058, '<tool_call>', 'opens a tool call'],
            [248059, '</tool_call>', 'closes a tool call'],
            [248076, '(last added)', 'end of the 33 ids always kept in the draft'],
          ],
        },
        ds: {
          V: 129280, tp3: 129408, tp: 64640, regions: [[0, 129280, 'tokens (BPE + special; exact ranges not inventoried)', 'txt']],
          st: [
            [0, '<｜begin▁of▁sentence｜>', 'BOS · opens every conversation', 'sp'],
            [1, '<｜end▁of▁sentence｜>', 'EOS · closes each assistant turn', 'sp'],
            [64640, 'TP=2 boundary', 'rank 0: ids 0–64,639 · rank 1: 64,640–129,279', 'tp'],
            [128799, 'DSpark noise', 'draft positions and graph padding'],
            [129264, '<｜deepseek_image｜>', 'image placeholder (IMAGE_TOKEN_ID) · reserved', 'sp'],
          ],
        },
      };
      const S = { m: 'glm', tp3: false, sel: 2 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; S.sel = v === 'glm' ? 2 : 1; } });
      UI.toggle(ctl, { label: 'padding for TP=3', value: false, onChange: (v) => { S.tp3 = v; } });
      // colors as CSS variables: the legend follows theme changes
      UI.legend(ctl, [['color-mix(in srgb, var(--viz-a) 50%, transparent)', 'regular BPE'], ['var(--viz-b)', 'special'], ['var(--viz-c)', 'added (not special)'], ['var(--viz-muted)', 'padding'], ['var(--viz-d)', 'TP=3 padding / TP boundary'], ['var(--viz-f)', 'draft mask / noise']]);
      const KNAME = { txt: 'BPE token', sp: 'special', tool: 'added, not special', pad: 'padding', draft: 'draft mask / noise', tp: 'TP boundary' };
      const KCOL = (C, k) => (k === 'txt' ? C.a : k === 'sp' ? C.b : k === 'tool' ? C.c : k === 'pad' ? C.muted : k === 'draft' ? C.f : C.d);
      const kindOf = (M, s) => { if (s[3]) return s[3]; if (s[1].includes('DFlash') || s[1].includes('noise')) return 'draft'; const r = M.regions.find(([a, b]) => s[0] >= a && s[0] < b); return r ? r[3] : 'pad'; };
      V.fig(fig, (g, w, h, t, dt, _S, Hd) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        const M = MAPS[S.m];
        const nar = narrow(w);
        const Vmax = S.tp3 ? M.tp3 : M.V;
        const detailH = nar ? 74 : 62;
        const top = 30, bot = h - detailH - 18;
        // vertical bar with the whole vocabulary
        const bx = nar ? 22 : 60, bw = nar ? 22 : 34;
        const Y = (id) => top + (id / Vmax) * (bot - top);
        V.text(g, 'id 0', bx + bw / 2, top - 10, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        V.text(g, nf(Vmax), bx + bw / 2, bot + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        const regs = M.regions.slice();
        if (S.tp3 && M.tp3 > M.V) regs.push([M.V, M.tp3, 'TP=3 padding (+' + (M.tp3 - M.V) + ')' + (M.tp3hyp ? ' · hypothetical' : ''), 'pad3']);
        regs.forEach(([a, b, , kind]) => {
          const y1 = Y(a), y2 = Math.max(Y(b), y1 + (kind === 'txt' ? 0 : 3));
          const col = kind === 'txt' ? C.a : kind === 'sp' ? C.b : kind === 'tool' ? C.c : kind === 'pad' ? C.muted : C.d;
          g.fillStyle = V.alpha(col, kind === 'txt' ? 0.22 : 0.85);
          g.fillRect(bx, y1, bw, y2 - y1);
        });
        g.strokeStyle = V.alpha(C.muted, 0.5); g.lineWidth = 1; g.strokeRect(bx, top, bw, bot - top);
        if (!nar) {
          // name of the main region, vertical
          const [a, b, name] = regs[0];
          g.save(); g.translate(bx - 12, (Y(a) + Y(b)) / 2); g.rotate(-Math.PI / 2); V.text(g, name, 0, 0, { size: 11, color: C.muted, align: 'center' }); g.restore();
        }
        // magnifier: the vocabulary tail, where the specials live
        if (M.regions.length > 1 || S.tp3) {
          const tail = M.regions.length > 1 ? M.regions[1][0] : M.V;
          const yz = Y(tail);
          g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.5;
          g.strokeRect(bx - 4, yz - 3, bw + 8, bot - yz + 6);
          if (!nar) { V.text(g, 'tail', bx - 6, bot - 14, { size: 11, color: C.ink, font: 'mono', align: 'right' }); V.text(g, nf(Vmax - tail) + ' ids', bx - 6, bot - 1, { size: 11, color: C.ink, font: 'mono', align: 'right' }); }
        }
        // subway
        const mx = nar ? bx + bw + 64 : bx + bw + 120;
        const n = M.st.length;
        const sy = (i) => top + 8 + (i * (bot - top - 16)) / Math.max(1, n - 1);
        // fan of lines
        M.st.forEach((s, i) => {
          const y0 = Y(s[0]), y1 = sy(i);
          const sel = i === S.sel;
          g.strokeStyle = sel ? C.ink : V.alpha(KCOL(C, kindOf(M, s)), 0.4);
          g.lineWidth = sel ? 2 : 1;
          g.setLineDash(s[3] === 'tp' ? [3, 3] : []);
          g.beginPath(); g.moveTo(bx + bw, y0); g.bezierCurveTo(bx + bw + 60, y0, mx - 60, y1, mx - 8, y1); g.stroke();
          g.setLineDash([]);
          V.dot(g, bx + bw, y0, sel ? 3.5 : 2, sel ? C.ink : KCOL(C, kindOf(M, s)));
        });
        // track
        g.strokeStyle = V.alpha(C.b, 0.7); g.lineWidth = 5; g.lineCap = 'round';
        g.beginPath(); g.moveTo(mx, sy(0)); g.lineTo(mx, sy(n - 1)); g.stroke(); g.lineCap = 'butt';
        // train
        const tt = (t * 0.25) % 1, ti = tt * (n - 1);
        const ty = V.lerp(sy(Math.floor(ti)), sy(Math.min(n - 1, Math.floor(ti) + 1)), V.ease.inOut(ti - Math.floor(ti)));
        V.glow(g, mx, ty, 12, C.a, 0.9);
        const p = Hd.pointer;
        let hover = -1;
        M.st.forEach((s, i) => {
          const y = sy(i);
          if (p.inside && Math.abs(p.y - y) < (bot - top) / n / 2 + 1 && p.x > mx - 20) hover = i;
          const sel = i === S.sel;
          const col = KCOL(C, kindOf(M, s));
          g.beginPath(); g.arc(mx, y, sel ? 7 : 5, 0, Math.PI * 2); g.fillStyle = C.panel || C.bg; g.fill(); g.lineWidth = 2.5; g.strokeStyle = col; g.stroke();
          const fs = nar ? 11 : 12;
          V.text(g, nf(s[0]), mx + 14, y + 4, { size: fs, font: 'mono', color: C.muted });
          const wide = !nar && w - mx > 520;
          V.text(g, s[1], mx + (nar ? 70 : 84), y + 4, { size: fs, font: 'mono', weight: sel ? 700 : 500, color: sel ? col : C.ink, maxW: w - mx - (nar ? 76 : 90) - (wide ? 300 : 8) });
          if (wide) V.text(g, s[2], w - 12, y + 4, { size: 11.5, color: C.muted, align: 'right' });
        });
        if (hover >= 0) S.sel = hover;
        if (p.clicked && hover < 0 && p.x < mx) {
          // click on the bar: nearest station by id
          const id = ((p.y - top) / (bot - top)) * Vmax;
          let bi = 0; M.st.forEach((s, i) => { if (Math.abs(s[0] - id) < Math.abs(M.st[bi][0] - id)) bi = i; }); S.sel = bi;
        }
        // detail
        const s = M.st[Math.min(S.sel, n - 1)];
        const dy = h - detailH - 4;
        V.box(g, 12, dy, w - 24, detailH, { r: 10, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.4) });
        const sk = kindOf(M, s);
        V.text(g, s[1] + '   id ' + nf(s[0]), 26, dy + 24, { size: 13, font: 'mono', weight: 700, color: C.ink, maxW: nar ? w - 52 : w - 260 });
        if (!nar) V.text(g, KNAME[sk] || '', w - 26, dy + 24, { size: 12, weight: 700, color: KCOL(C, sk), align: 'right' });
        V.text(g, s[2], 26, dy + 46, { size: 12.5, color: C.muted, maxW: w - 52 });
        if (nar) V.text(g, S.m === 'glm' ? '* inferred from a real tokenized prompt' : ' ', 26, dy + 64, { size: 11, color: C.muted });
      });
    })();

    /* =====================================================================
       Figure 4 — The template conveyor belt
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-esteira');
      if (!fig) return;
      tall(fig, 660);
      const S = { m: 'glm', think: true, tools: false, img: false };
      const T = (l) => ({ label: l, kind: 'txt' });
      const SP = (l, id) => ({ label: l, kind: 'sp', sub: id != null ? String(id) : undefined });
      const sysTxt = [T('You'), T('␣are'), T('␣helpful'), T('.')];
      const usrTxt = [T('What'), T('␣is'), T('␣3'), T('+'), T('3'), T('?')];
      function build() {
        const o = [];
        if (S.m === 'glm') {
          o.push(SP('[gMASK]', '154822*'), SP('<sop>', '154824*'));
          if (S.think) o.push(SP('<|system|>', '154826*'), { label: 'Reasoning␣Effort:␣Max', kind: 'eff' });
          if (S.tools) o.push(SP('<|system|>', '154826*'), { label: '#␣Tools · <tools>…</tools> · XML format', kind: 'tool' });
          o.push(SP('<|system|>', '154826*'), ...sysTxt, SP('<|user|>', 154827));
          if (S.img) o.push(SP('<|begin_of_image|>', 154830), { label: '<|image|> → N vision tokens', kind: 'img', sub: '154854' }, SP('<|end_of_image|>'));
          o.push(...usrTxt, { label: '<|assistant|>', kind: 'gen', sub: '154828*' }, { label: '<think>', kind: 'gen', sub: '154841' });
          if (!S.think) o.push({ label: '</think>', kind: 'gen', sub: '154842' });
        } else if (S.m === 'qwen') {
          o.push(SP('<|im_start|>', 248045), T('system'), T('↵'), ...sysTxt);
          if (S.tools) o.push({ label: '<tools>…</tools> in system', kind: 'tool' });
          o.push(SP('<|im_end|>', 248046), T('↵'), SP('<|im_start|>', 248045), T('user'), T('↵'));
          if (S.img) o.push(SP('<|vision_start|>', 248053), { label: '<|image_pad|> → N', kind: 'img', sub: '248056' }, SP('<|vision_end|>', 248054));
          o.push(...usrTxt, SP('<|im_end|>', 248046), T('↵'), { label: '<|im_start|>', kind: 'gen', sub: '248045' }, { label: 'assistant', kind: 'gen' }, { label: '↵', kind: 'gen' });
          if (!S.think) o.push({ label: '<think>', kind: 'gen' }, { label: '</think>', kind: 'gen' });
          else o.push({ label: '…', kind: 'ghost', sub: 'template' });
        } else {
          o.push(SP('<｜begin▁of▁sentence｜>', 0), ...sysTxt);
          if (S.tools) o.push({ label: '## Tools + DSML instructions', kind: 'tool' });
          o.push(SP('<｜User｜>'));
          if (S.img) o.push({ label: '<｜deepseek_image｜> → N', kind: 'img', sub: '129264' });
          o.push(...usrTxt, { label: '<｜Assistant｜>', kind: 'gen' }, { label: S.think ? '<think>' : '</think>', kind: 'gen' });
        }
        return o;
      }
      const MECH = { glm: ['Jinja', '/opt/glm53/chat_template.jinja'], qwen: ['Jinja', 'checkpoint chat_template.jinja'], ds: ['Python', 'encoding_dsv4.py · --tokenizer-mode deepseek_v4'] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'reasoning', value: true, onChange: (v) => { S.think = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'tools', value: false, onChange: (v) => { S.tools = v; S.t0 = null; } });
      UI.toggle(ctl, { label: 'image', value: false, onChange: (v) => { S.img = v; S.t0 = null; } });
      UI.legend(ctl, [['var(--viz-b)', 'special'], ['var(--viz-a)', 'generation prompt'], ['var(--viz-d)', 'effort'], ['var(--viz-c)', 'tools'], ['var(--viz-f)', 'media']]);
      const rN = UI.readout(ctl, 'pills', '—');
      const gears = (g, C, cx, cy, r, a) => {
        g.save(); g.translate(cx, cy); g.rotate(a);
        g.fillStyle = V.alpha(C.b, 0.35); g.strokeStyle = C.b; g.lineWidth = 1.5;
        g.beginPath();
        for (let i = 0; i < 16; i++) { const ang = (i / 16) * Math.PI * 2; const rr = i % 2 ? r : r * 1.22; g.lineTo(Math.cos(ang) * rr, Math.sin(ang) * rr); }
        g.closePath(); g.fill(); g.stroke();
        g.beginPath(); g.arc(0, 0, r * 0.35, 0, Math.PI * 2); g.fillStyle = C.panel || C.bg; g.fill(); g.stroke();
        g.restore();
      };
      const wrapChars = (g, str, maxW, size) => {
        g.font = F(size, 'mono', 400);
        const lines = []; let cur = '';
        for (const ch of str) {
          if (ch === '\n') { lines.push(cur + '↵'); cur = ''; continue; }
          if (g.measureText(cur + ch).width > maxW) { lines.push(cur); cur = ch; } else cur += ch;
        }
        if (cur) lines.push(cur);
        return lines;
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        if (S.t0 == null) S.t0 = t;
        const items = build();
        rN.set(String(items.length));
        const nar = narrow(w);
        const raw = items.map((it) => (it.kind === 'ghost' ? '' : it.kind === 'tool' ? '[tools…]' : it.kind === 'img' ? it.label.split(' ')[0] + '[×N]' : it.label.replace(/␣/g, ' ').replace(/↵/g, '\n'))).join('');
        let cx, cy, cw, mx, my, mw, mh, sx0, sy0, sw, sh, ox, oy, ow;
        if (!nar) {
          cx = 16; cy = 30; cw = w * 0.28;
          mx = cx + cw + 30; my = 22; mw = w * 0.15; mh = 108;
          sx0 = mx + mw + 30; sy0 = 22; sw = w - sx0 - 16; sh = 108;
          ox = 16; oy = 158; ow = w - 32;
        } else {
          cx = 16; cy = 26; cw = w - 32;
          mx = 16; my = 130; mw = w - 32; mh = 52;
          sx0 = 16; sy0 = 196; sw = w - 32; sh = 92;
          ox = 16; oy = 318; ow = w - 32;
        }
        const cards = [['system', 'You are helpful.' + (S.tools ? ' + tools' : '')], ['user', 'What is 3+3?' + (S.img ? ' + image' : '')]];
        V.text(g, 'MESSAGES (JSON)', cx, cy - 10, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        cards.forEach(([role, txt], i) => {
          const y = cy + i * 50;
          V.box(g, cx, y, cw, 44, { r: 10, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.4) });
          V.text(g, '"role": "' + role + '"', cx + 10, y + 17, { size: 11.5, font: 'mono', color: C.b, maxW: cw - 18 });
          V.text(g, '"content": "' + txt + '"', cx + 10, y + 34, { size: 11.5, font: 'mono', color: C.ink, maxW: cw - 18 });
        });
        // machine
        V.box(g, mx, my, mw, mh, { r: 14, fill: V.alpha(C.b, 0.1), stroke: C.b, lw: 1.6 });
        if (!nar) {
          gears(g, C, mx + mw * 0.36, my + 34, 14, t * 1.2);
          gears(g, C, mx + mw * 0.62, my + 52, 10, -t * 1.6);
          V.text(g, MECH[S.m][0], mx + mw / 2, my + 84, { size: 13, weight: 700, color: C.ink, align: 'center' });
          V.text(g, MECH[S.m][1].split(' · ')[0], mx + mw / 2, my + 100, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: mw - 8 });
        } else {
          gears(g, C, mx + 28, my + mh / 2, 12, t * 1.2);
          V.text(g, 'template: ' + MECH[S.m][0], mx + 54, my + 21, { size: 13, weight: 700, color: C.ink });
          V.text(g, MECH[S.m][1], mx + 54, my + 39, { size: 11, font: 'mono', color: C.muted, maxW: mw - 62 });
        }
        // particles: messages → machine → string
        for (let k = 0; k < 10; k++) {
          const u = (t * 0.45 + k / 10) % 1;
          let a, b2;
          if (!nar) { a = k < 5 ? [cx + cw, cy + 22 + (k % 2) * 50] : [mx + mw, my + mh / 2]; b2 = k < 5 ? [mx, my + mh / 2] : [sx0, sy0 + sh / 2]; }
          else { a = k < 5 ? [cx + cw * (0.2 + 0.12 * k), cy + 94] : [mx + mw * (0.2 + 0.12 * (k - 5)), my + mh]; b2 = k < 5 ? [mx + mw * (0.2 + 0.12 * k), my] : [mx + mw * (0.2 + 0.12 * (k - 5)), sy0]; }
          const pp = V.along(a[0], a[1], b2[0], b2[1], u);
          V.dot(g, pp[0], pp[1], 2.2, V.alpha(k < 5 ? C.b : C.a, 0.85));
        }
        // rendered string
        V.box(g, sx0, sy0, sw, sh, { r: 10, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.35) });
        V.text(g, 'STRING THE TEMPLATE RETURNS', sx0 + 10, sy0 + 16, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const lines = wrapChars(g, raw, sw - 20, 11.5);
        const maxL = Math.floor((sh - 26) / 15);
        lines.slice(0, maxL).forEach((ln, i) => V.text(g, i === maxL - 1 && lines.length > maxL ? ln.slice(0, -1) + '…' : ln, sx0 + 10, sy0 + 34 + i * 15, { size: 11.5, font: 'mono', color: C.ink }));
        // tokens
        V.text(g, 'TOKENS (what the network receives)', ox, oy - 8, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const L = flow(g, items, ox, oy, ow, 11.5, 44, 5);
        const el = t - S.t0;
        const per = 0.07;
        const cyc = items.length * per + 6;
        const lt = el % cyc;
        const nShow = Math.min(items.length, Math.floor(lt / per) + 1);
        const exitX = nar ? sx0 + sw / 2 : sx0 + sw / 2, exitY = sy0 + sh;
        L.items.forEach((p, i) => {
          if (i >= nShow) return;
          const age = lt - i * per;
          const u = V.ease.out(Math.min(1, age / 0.35));
          const x = V.lerp(exitX, p.x, u), y = V.lerp(exitY, p.y, u);
          drawChip(g, C, x, y, p.w, 36, p.it.label, p.it.kind, { size: 11.5, sub: p.it.sub });
        });
        if (nShow === items.length) {
          const gen = L.items.filter((p) => p.it.kind === 'gen');
          if (gen.length) {
            const last = gen[gen.length - 1];
            V.glow(g, last.x + last.w / 2, last.y + 18, 26, C.a, 0.4 + 0.25 * Math.sin(t * 4));
            const rowStart = gen.filter((q) => q.y === last.y)[0];
            V.text(g, '↑ generation prompt: the model continues from here', Math.max(ox, Math.min(rowStart.x, ox + ow - 360)), L.bottom + 4, { size: 12, color: C.a, weight: 600 });
          }
        }
        if (S.m === 'glm') V.text(g, '* id read from a real tokenized prompt in the recipe tests', ox, h - 8, { size: 11, color: C.muted });
      });
    })();

    /* =====================================================================
       Figure 5 — The DeepSeek assembly line (encoder + #144 + #52)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-encoder');
      if (!fig) return;
      tall(fig, 500);
      const S = { eff: 'high', fix144: false, asstLast: false, fix52: false };
      const DIR = { low: 0, high: 79, max: 92 }; // BOS + directive = 80 and 93 tokens (CHANGELOG #144)
      const steps = [
        { t: 'Messages enter the encoder', d: '<code style="font-variant-ligatures:none">--tokenizer-mode deepseek_v4</code> swaps Jinja for a wrapper that calls <code style="font-variant-ligatures:none">encode_messages()</code> from the checkpoint\'s <code style="font-variant-ligatures:none">encoding_dsv4.py</code>. Out comes a string with markers, which BPE converts into ids.' },
        { t: 'BOS and the effort directive', d: 'In reasoning mode, the message at index 0 gets the directive for the requested level in front of it: nothing at <code style="font-variant-ligatures:none">low</code>, a paragraph of instructions at <code style="font-variant-ligatures:none">high</code> and <code style="font-variant-ligatures:none">max</code> (BOS + directive = 80 and 93 tokens). Change the effort in the controls.' },
        { t: 'Roles become markers', d: 'The system message goes in as plain text, with tools and DSML instructions appended; each user message gets <code style="font-variant-ligatures:none">&lt;｜User｜&gt;</code>; tool results become <code style="font-variant-ligatures:none">&lt;tool_result&gt;</code> inside the user turn; every assistant turn ends in <code style="font-variant-ligatures:none">&lt;｜end▁of▁sentence｜&gt;</code>.' },
        { t: 'The generation prompt', d: 'Only after a <code style="font-variant-ligatures:none">user</code> or <code style="font-variant-ligatures:none">developer</code> message does the encoder append <code style="font-variant-ligatures:none">&lt;｜Assistant｜&gt;</code> + <code style="font-variant-ligatures:none">&lt;think&gt;</code> (or <code style="font-variant-ligatures:none">&lt;/think&gt;</code> in chat mode). That is where the model continues from.' },
        { t: '256-token blocks and the chained hash', d: 'The prefix cache compares blocks of 256 tokens, and the hash of each one depends on the previous one. With the directive right after BOS, block 0 changes and so does every block after it: 0 of 18 shared across efforts. Turn on #144 to move the directive to the end of the system region: 18 of 18.' },
        { t: 'The dead turn (#52/#120)', d: 'If the conversation ends in an assistant turn (a harness retry), stock closes the turn with EOS and adds no generation prompt. Turn on “last turn = assistant” and then the hotfix, which appends a fresh header.' },
        { t: 'Installation at every boot', d: 'The image predates the checkpoint. The entrypoint looks for the encoder (<code style="font-variant-ligatures:none">DSPARK_ENCODING_FILE</code> → revision snapshot → any snapshot → 0731 → <code style="font-variant-ligatures:none">/models</code>), copies it to <code style="font-variant-ligatures:none">vllm/tokenizers/deepseek_v4_encoding.py</code>, fixes the effort mapping and applies #21. Without the file: WARN, and none of this happens.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 7500 });
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Effort', options: ['low', 'high', 'max'], value: 'high', onChange: (v) => { S.eff = v; } });
      UI.toggle(ctl, { label: '#144: directive at end of system', value: false, onChange: (v) => { S.fix144 = v; } });
      UI.toggle(ctl, { label: 'last turn = assistant', value: false, onChange: (v) => { S.asstLast = v; } });
      UI.toggle(ctl, { label: 'hotfix assistant-final', value: false, onChange: (v) => { S.fix52 = v; } });
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        const s = st.i, el = st.elapsed();
        const nar = narrow(w);
        const x0 = 16, W = w - 32;
        // ---------- top: assembly line or installation chain
        const topY = 14, topH = nar ? 124 : 70;
        if (s !== 6) {
          const boxes = [['messages', 'JSON'], ['encode_messages()', 'encoding_dsv4.py'], ['text', 'with markers'], ['BPE', 'tokenizer.json'], ['ids', '[0, …]']];
          const n = boxes.length;
          const cols = nar ? 3 : n;
          const bw = (W - (cols - 1) * 14) / cols, bh = 48;
          boxes.forEach(([a, b], i) => {
            const r = Math.floor(i / cols), c = i % cols;
            const x = x0 + c * (bw + 14), y = topY + r * (bh + 16);
            const on = (s === 0 && i <= Math.floor(el * 1.5)) || (s >= 1 && s <= 3 && i === 1) || (s === 4 && i === 4);
            V.box(g, x, y, bw, bh, { r: 10, fill: V.alpha(i === 1 ? C.f : C.b, on ? 0.28 : 0.1), stroke: on ? (i === 1 ? C.f : C.b) : V.alpha(C.muted, 0.45), lw: on ? 2 : 1 });
            V.text(g, a, x + bw / 2, y + 20, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center', maxW: bw - 8 });
            V.text(g, b, x + bw / 2, y + 36, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: bw - 8 });
            if (i < n - 1 && c < cols - 1) {
              V.arrow(g, x + bw + 2, y + bh / 2, x + bw + 12, y + bh / 2, { color: V.alpha(C.muted, 0.7), head: 6 });
              const u = (t * 0.8 + i * 0.2) % 1;
              V.dot(g, x + bw + 2 + 10 * u, y + bh / 2, 2, C.a);
            }
          });
        } else {
          const cand = ['DSPARK_ENCODING_FILE', 'snapshot @ DSPARK_REVISION', 'any snapshot', 'snapshot 0731', '/models/…/0731'];
          const cols = nar ? 2 : 5;
          const bw = (W - (cols - 1) * 10) / cols, bh = 30;
          const found = 1; // illustrative: found in the pinned-revision snapshot
          const tried = Math.min(found, Math.floor(el / 0.8));
          cand.forEach((c, i) => {
            const r = Math.floor(i / cols), cc = i % cols;
            const x = x0 + cc * (bw + 10), y = topY + r * (bh + 8);
            const isF = i === found && tried >= found;
            const miss = i < tried;
            V.box(g, x, y, bw, bh, { r: 8, fill: isF ? V.alpha(C.a, 0.3) : miss ? V.alpha(C.muted, 0.08) : V.alpha(C.b, 0.1), stroke: isF ? C.a : V.alpha(C.muted, 0.5), lw: isF ? 2 : 1 });
            V.text(g, (miss ? '✕ ' : isF ? '✓ ' : '') + c, x + bw / 2, y + bh / 2 + 1, { size: 11, font: 'mono', color: miss ? C.muted : C.ink, align: 'center', base: 'middle', maxW: bw - 6 });
          });
          const ty = topY + (nar ? 3 * 38 : 38) + 6;
          V.text(g, 'cp → vllm/tokenizers/deepseek_v4_encoding.py  ·  patch: high→high, rest→low  ·  #21 (args dict)', x0, ty + 10, { size: 11.5, font: 'mono', color: C.a, maxW: W });
          V.text(g, 'example: empty variable (✕) → file found in the pinned-revision snapshot (✓)', x0, ty + 28, { size: 11, color: C.muted, maxW: W });
        }
        // ---------- middle: token strip to scale
        const dir = DIR[S.eff];
        const SYS = 4629, TAIL = 16; // 1 (BOS) + 4629 + 16 = 4646 (measured case from #144)
        const segs = S.fix144 ? [['BOS', 1, 'sp'], ['system + tools', SYS, 'txt'], ['directive', dir, 'eff'], ['user + gen prompt', TAIL, 'gen']] : [['BOS', 1, 'sp'], ['directive', dir, 'eff'], ['system + tools', SYS, 'txt'], ['user + gen prompt', TAIL, 'gen']];
        const total = segs.reduce((a, b) => a + b[1], 0);
        const my = topY + topH + (nar ? 40 : 34);
        V.text(g, 'PROMPT: ' + nf(total) + ' tokens  (effort ' + S.eff + (S.fix144 ? ', #144 on' : '') + ')', x0, my - 10, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        const sx = (tok) => x0 + (tok / (4646 + DIR.max)) * W;
        let acc = 0;
        const hiSeg = s === 1 ? 'directive' : s === 2 ? 'system + tools' : s === 3 ? 'user + gen prompt' : '';
        segs.forEach(([name, n, kind]) => {
          if (!n) { acc += n; return; }
          const xa = sx(acc), xb = Math.max(sx(acc + n), xa + 4);
          const stl = style(C, kind);
          const on = name === hiSeg;
          V.box(g, xa, my, xb - xa, 30, { r: 4, fill: on ? V.alpha(stl.stroke, 0.45 + 0.15 * Math.sin(t * 5)) : stl.fill, stroke: stl.stroke, lw: on ? 2 : 1 });
          if (xb - xa > 90) V.text(g, name + ' · ' + nf(n), (xa + xb) / 2, my + 16, { size: 11.5, font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          acc += n;
        });
        if (dir && !S.fix144) V.text(g, '↑ BOS + directive: ' + (dir + 1) + ' tok', sx(0) + 2, my + 46, { size: 11.5, font: 'mono', color: C.d });
        if (dir && S.fix144) V.text(g, 'directive ' + dir + ' tok ↑', sx(1 + SYS) - 4, my + 46, { size: 11.5, font: 'mono', color: C.d, align: 'right' });
        // 256-token blocks
        const by = my + 58;
        const nFull = Math.floor(total / 256);
        let shared = 0, broken = false;
        const dirStart = S.fix144 ? 1 + SYS : 1;
        const cx = [];
        for (let b = 0; b < nFull; b++) {
          const end = (b + 1) * 256;
          const same = S.eff === 'low' || end <= dirStart;
          if (!same) broken = true;
          const ok = same && !broken;
          if (ok) shared++;
          const xa = sx(b * 256), xb = sx(end);
          const on = s === 4;
          const wave = on ? Math.max(0, 1 - Math.abs(el * 6 - b) / 2) : 0;
          g.fillStyle = ok ? V.alpha(C.e, 0.45 + 0.4 * wave) : V.alpha(C.red, 0.35 + 0.4 * wave);
          V.rr(g, xa + 1, by, xb - xa - 2, 22, 3); g.fill();
          cx.push([(xa + xb) / 2, ok]);
        }
        // necklace of hashes: each one depends on the previous one
        const ny = by + 34;
        for (let b = 0; b < cx.length; b++) {
          if (b > 0) V.line(g, cx[b - 1][0] + 4, ny, cx[b][0] - 4, ny, { color: V.alpha(cx[b][1] ? C.e : C.red, 0.7), lw: 1.5 });
          V.dot(g, cx[b][0], ny, 3.5, cx[b][1] ? C.e : C.red);
        }
        const pulse = (t * 3) % Math.max(1, cx.length);
        const pi = Math.floor(pulse);
        if (cx[pi]) V.glow(g, V.lerp(cx[pi][0], (cx[pi + 1] || cx[pi])[0], pulse - pi), ny, 10, C.ink, 0.5);
        V.text(g, (nar ? '256-token blocks vs. low: ' : '256-token prefix cache blocks, compared with the same prompt at low: ') + shared + '/' + nFull + ' shared' + (S.eff !== 'low' && S.fix144 && !nar ? ' · measured common prefix: 4,630 tokens' : ''), x0, by + 58, { size: 12.5, color: shared === nFull ? C.e : C.red, weight: 600, maxW: W });
        V.text(g, nar ? 'chained hash: one different block taints the rest' : 'hash(block) = f(previous hash, block tokens): one different block taints every block after it', x0, by + 76, { size: 11.5, color: C.muted, maxW: W });
        // ---------- bottom: how the prompt ends
        const ey = by + 108;
        V.text(g, 'HOW THE PROMPT ENDS', x0, ey, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        let tail, verdict, vcol;
        if (!S.asstLast) {
          tail = [{ label: '<｜User｜>', kind: 'sp' }, { label: 'continue', kind: 'txt' }, { label: '<｜Assistant｜>', kind: 'gen' }, { label: '<think>', kind: 'gen' }];
          verdict = 'generation prompt: the model opens its turn and thinks'; vcol = C.a;
        } else if (!S.fix52) {
          tail = [{ label: '<｜Assistant｜>', kind: 'sp' }, { label: '…verdict:', kind: 'txt' }, { label: '<｜end▁of▁sentence｜>', kind: 'eos' }, { label: '∅', kind: 'ghost' }];
          verdict = 'dead state: 1–2 tokens and stop, or hallucinated markup (A/B: 400 raw DSML tokens)'; vcol = C.red;
        } else {
          tail = [{ label: '<｜Assistant｜>', kind: 'sp' }, { label: '…verdict:', kind: 'txt' }, { label: '<｜end▁of▁sentence｜>', kind: 'sp' }, { label: '<｜Assistant｜>', kind: 'gen' }, { label: '<think>', kind: 'gen' }];
          verdict = 'hotfix: closed turn + new header (docstring: 140 tokens on a complete turn, vs. 16 for stock)'; vcol = C.a;
        }
        const L = flow(g, tail, x0, ey + 12, W, 13, 44, 8);
        L.items.forEach((p, i) => {
          const isNew = S.fix52 && S.asstLast && i >= 3;
          drawChip(g, C, p.x, p.y, p.w, 34, p.it.label, p.it.kind, { size: 13, lw: isNew ? 2.4 : 1.2 });
          if (isNew || (s === 3 && p.it.kind === 'gen')) V.glow(g, p.x + p.w / 2, p.y + 15, 24, C.a, 0.4 + 0.3 * Math.sin(t * 5));
          if (p.it.kind === 'eos' && s === 5) V.glow(g, p.x + p.w / 2, p.y + 15, 26, C.red, 0.5 + 0.3 * Math.sin(t * 6));
        });
        V.text(g, verdict, x0, L.bottom + 6, { size: 13, color: vcol, weight: 600, maxW: W });
      });
    })();

    /* =====================================================================
       Figure 6 — The vector bookshelf (embedding lookup)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-lookup');
      if (!fig) return;
      tall(fig, 372);
      const VOC = ['<bos>', 'The', '␣cat', '␣dog', '␣sat', '␣slept', '␣on', '␣grass', '␣hay', '.'];
      const r = V.rng(11);
      const E = Array.from({ length: 10 }, () => Array.from({ length: 6 }, () => Math.round(r.normal() * 8) / 10));
      E[3] = E[2].map((v, j) => Math.round((v + (j % 2 ? 0.2 : -0.1)) * 10) / 10); // dog ≈ cat
      E[8] = E[7].map((v, j) => Math.round((v + (j % 3 ? -0.2 : 0.1)) * 10) / 10); // hay ≈ grass
      const SENTS = { a: [0, 1, 2, 4, 6, 7, 9], b: [0, 1, 3, 5, 6, 8, 9] };
      const REAL = { glm: [154880, 4096], qwen: [248320, 2560], ds: [129280, 4096] };
      const S = { s: 'a', onehot: false, m: 'glm', t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Sentence', options: [{ v: 'a', label: 'The cat sat on grass.' }, { v: 'b', label: 'The dog slept on hay.' }], value: 'a', onChange: (v) => { S.s = v; S.reset = true; } });
      UI.toggle(ctl, { label: 'show as one-hot × E', value: false, onChange: (v) => { S.onehot = v; } });
      UI.seg(ctl, { label: 'Real model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S.m = v; upd(); } });
      const r1 = UI.readout(ctl, 'row read/token', '—');
      const r2 = UI.readout(ctl, 'table', '—');
      const r3 = UI.readout(ctl, 'one-hot × E would cost', '—');
      const upd = () => { const [Vv, d] = REAL[S.m]; r1.set(gib(d * 2)); r2.set(gb(Vv * d * 2) + ' = ' + gib(Vv * d * 2)); r3.set(nf((2 * Vv * d) / 1e9, 2) + ' GFLOP/token'); };
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        if (S.reset) { S.t0 = t; S.reset = false; }
        const ids = SENTS[S.s];
        const T = ids.length;
        const per = 1.1, hold = 1.8;
        const el = (t - S.t0) % (T * per + hold);
        const cur = Math.min(T - 1, Math.floor(el / per));
        const u = el >= T * per ? 1 : (el - cur * per) / per;
        const done = el >= T * per ? T : cur;
        const nar = narrow(w);
        // geometry
        const labW = nar ? 26 : 96;
        const gap = 2;
        const eyTop = nar ? 118 : 40;
        const cell = Math.min(nar ? Math.max(14, Math.min(24, (w - 32 - labW - 44) / 12 - 1)) : Math.min(34, (w - 150 - labW - 80) / 12), (h - eyTop - (nar ? 40 : 56)) / 10 - gap);
        const matW = 6 * (cell + gap) - gap, matH = 10 * (cell + gap) - gap;
        let tokX, tokY, ex, ey, xx, xy;
        if (!nar) { tokX = 16; tokY = 40; ex = 16 + 120 + labW; ey = 40; xx = ex + matW + 70; xy = 40; }
        else { tokX = 16; tokY = 30; ex = 16 + labW; ey = eyTop; xx = ex + matW + 22; xy = eyTop; }
        // tokens
        V.text(g, 'input ids', tokX, tokY - 12, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const tokPos = [];
        ids.forEach((id, i) => {
          let x, y, cw;
          if (!nar) { x = tokX; y = tokY + i * 34; cw = 104; }
          else { cw = (w - 32 - (T - 1) * 4) / T; x = tokX + i * (cw + 4); y = tokY; }
          const active = i === cur && el < T * per;
          const past = i < done;
          drawChip(g, C, x, y, cw, nar ? 40 : 28, nar ? String(id) : VOC[id] + '  ' + id, 'txt', { size: nar ? 11.5 : 12, fill: active ? V.alpha(C.a, 0.35) : past ? V.alpha(C.a, 0.12) : undefined, stroke: active ? C.a : undefined, lw: active ? 2 : 1.2, sub: nar ? VOC[id].replace('␣', '') : undefined });
          tokPos.push([x + cw, y + (nar ? 40 : 14), x + cw / 2]);
        });
        // matrix E
        if (!nar) V.text(g, 'E  [V × d] = [10 × 6]', ex + matW / 2, ey - 12, { size: 11.5, font: 'mono', color: C.muted, align: 'center', weight: 600 });
        else V.text(g, 'E [V×d] = [10×6]', ex + 2, ey - 10, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        const selRow = ids[cur];
        const Mg = V.matrix(g, {
          x: ex, y: ey, data: E, cell, gap, cmap: 'div', min: -2, max: 2, values: cell >= 22, digits: 1,
          colorFn: (v, i, j) => {
            const base = V.cmap(v / 2, 'div');
            if (S.onehot && i !== selRow) return V.alpha(C.muted, 0.12);
            return base;
          },
          hiRow: el < T * per ? selRow : -1, hiColor: C.a,
        });
        for (let i = 0; i < 10; i++) V.text(g, nar ? String(i) : i + ' ' + VOC[i], ex - 8, ey + i * (cell + gap) + cell / 2 + 4, { size: 11, font: 'mono', color: i === selRow ? C.a : C.muted, align: 'right' });
        if (S.onehot) {
          // one-hot vector aligned with the rows
          const ox = ex + matW + 8;
          for (let i = 0; i < 10; i++) {
            const on = i === selRow;
            V.box(g, ox, ey + i * (cell + gap), cell * 0.8, cell, { r: 3, fill: on ? V.alpha(C.a, 0.6) : V.alpha(C.muted, 0.1), stroke: on ? C.a : V.alpha(C.muted, 0.3), lw: 1 });
            V.text(g, on ? '1' : '0', ox + cell * 0.4, ey + i * (cell + gap) + cell / 2 + 1, { size: 11, font: 'mono', color: on ? C.ink : C.muted, align: 'center', base: 'middle' });
          }
          V.text(g, 'e', ox + cell * 0.4, ey - 12, { size: 11.5, font: 'mono', color: C.a, align: 'center', weight: 700 });
          if (!nar) V.text(g, 'eᵀE: 60 multiplications, 54 of them by zero', ex + matW / 2, ey + matH + 20, { size: 11.5, color: C.c, align: 'center' });
        } else if (!nar) V.text(g, 'gather: reads only the requested row', ex + matW / 2, ey + matH + 20, { size: 11.5, color: C.a, align: 'center' });
        // output X
        if (!nar) V.text(g, 'X  [T × d] = [' + T + ' × 6]', xx + matW / 2, xy - 12, { size: 11.5, font: 'mono', color: C.muted, align: 'center', weight: 600 });
        else V.text(g, 'X [T×d] = [' + T + '×6]', xx, xy - 10, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
        for (let i = 0; i < T; i++) for (let j = 0; j < 6; j++) {
          const x = xx + j * (cell + gap), y = xy + i * (cell + gap);
          if (i < done) {
            g.fillStyle = V.cmap(E[ids[i]][j] / 2, 'div'); V.rr(g, x, y, cell, cell, 3); g.fill();
            if (cell >= 22) V.text(g, E[ids[i]][j].toFixed(1), x + cell / 2, y + cell / 2 + 1, { size: Math.min(11.5, cell * 0.36), font: 'mono', color: C.ink, align: 'center', base: 'middle' });
          } else { g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 1; V.rr(g, x, y, cell, cell, 3); g.stroke(); }
        }
        if (!nar) for (let i = 0; i < T; i++) V.text(g, 't=' + i, xx + matW + 6, xy + i * (cell + gap) + cell / 2 + 4, { size: 11, font: 'mono', color: C.muted });
        // animation: token→row arrow and row sliding into X
        if (el < T * per) {
          const [tx, ty, tcx] = tokPos[cur];
          const [rx0, ry0] = [ex - (nar ? 4 : 2), ey + selRow * (cell + gap) + cell / 2];
          if (u < 0.4) {
            const k = u / 0.4;
            if (!nar) V.arrow(g, tx + 4, ty, V.lerp(tx + 4, rx0 - labW + 4, k), V.lerp(ty, ry0, k), { color: C.a, lw: 2, head: 7 });
            else V.arrow(g, tcx, ty + 2, V.lerp(tcx, ex - 18, k), V.lerp(ty + 2, ry0, k), { color: C.a, lw: 2, head: 7 });
          } else {
            const k = V.ease.inOut(Math.min(1, (u - 0.4) / 0.45));
            const fx = V.lerp(ex, xx, k), fy = V.lerp(ey + selRow * (cell + gap), xy + cur * (cell + gap), k);
            for (let j = 0; j < 6; j++) { g.fillStyle = V.alpha(V.cmap(E[selRow][j] / 2, 'div'), 1); V.rr(g, fx + j * (cell + gap), fy, cell, cell, 3); g.fill(); }
            g.strokeStyle = C.a; g.lineWidth = 2; V.rr(g, fx - 2, fy - 2, matW + 4, cell + 4, 5); g.stroke();
            V.glow(g, fx + matW / 2, fy + cell / 2, 30, C.a, 0.35);
          }
        }
        // footer with shapes
        const fy2 = h - 14;
        V.text(g, 'input_ids [B, T] (int)  →  E[input_ids]  →  X [B, T, d] (BF16)', nar ? 16 : ex - labW, fy2, { size: 12, font: 'mono', color: C.ink, maxW: w - 32 });
      });
    })();

    /* =====================================================================
       Figure 7 — The word map + concentration of measure
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-geometria');
      if (!fig) return;
      tall(fig, 700);
      const pt = (w, x, y, col, side) => ({ w, x, y, col, side });
      const P = [
        pt('␣cat', 0.62, -0.24, 'a', 'l'), pt('␣cats', 0.52, -0.04, 'a', 'r'), pt('␣dog', 0.72, -0.48, 'a', 'r'), pt('␣gato', 0.88, -0.18, 'a', 'r'),
        pt('␣1', -0.30, 0.84, 'd', 'r'), pt('␣2', -0.10, 0.90, 'd', 'r'), pt('␣three', -0.44, 0.66, 'd', 'l'), pt('␣ten', -0.20, 0.68, 'd', 'r'),
        pt('␣blue', -0.86, 0.22, 'red', 'r'), pt('␣green', -0.80, 0.02, 'red', 'r'), pt('␣azul', -0.98, 0.10, 'red', 'l'),
        pt('def', -0.48, -0.70, 'muted', 'l'), pt('return', -0.30, -0.88, 'muted', 'r'), pt('␣for', -0.64, -0.86, 'muted', 'l'),
        pt('<think>', 0.20, -0.80, 'b', 'r'), pt('</think>', 0.34, -0.62, 'b', 'r'), pt('<|user|>', 0.02, -0.66, 'b', 'b'),
        pt('␣man', 0.28, 0.14, 'f', 'b'), pt('␣woman', 0.06, 0.40, 'f', 'l'), pt('␣king', 0.72, 0.40, 'f', 'r'), pt('␣queen', 0.52, 0.66, 'f', 'r'),
      ];
      const DS = [2, 3, 4, 8, 16, 32, 64, 128, 512, 2560, 4096];
      const S = { sel: 0, di: 10, an: false };
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'dimension', min: 0, max: DS.length - 1, step: 1, value: S.di, fmt: (i) => String(DS[i]), onInput: (i) => { S.di = i; } });
      UI.toggle(ctl, { label: 'analogy king − man + woman', value: false, onChange: (v) => { S.an = v; } });
      const rS = UI.readout(ctl, 'cosine std. dev.', '—');
      UI.legend(ctl, [['var(--viz-a)', 'animals'], ['var(--viz-d)', 'numbers'], ['var(--red)', 'colors'], ['var(--viz-muted)', 'code'], ['var(--viz-b)', 'special'], ['var(--viz-f)', 'people']]);
      const cosv = (a, b) => (a.x * b.x + a.y * b.y) / (Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y));
      // precomputed densities
      const NG = 400;
      const dens = DS.map((d) => {
        const ys = []; let mx = 0;
        for (let i = 0; i <= NG; i++) { const c = -1 + (2 * i) / NG; const v = d === 2 ? 1 / Math.sqrt(Math.max(1e-3, 1 - c * c)) : Math.pow(Math.max(0, 1 - c * c), (d - 3) / 2); ys.push(v); }
        const sum = ys.reduce((a, b) => a + b, 0) * (2 / NG);
        const nys = ys.map((v) => v / sum);
        nys.forEach((v, i) => { if (i > 2 && i < NG - 2) mx = Math.max(mx, v); });
        return { ys: nys, mx };
      });
      V.fig(fig, (g, w, h, t, dt, _S, Hd) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        const d = DS[S.di];
        rS.set(nf(1 / Math.sqrt(d), 4));
        const nar = narrow(w);
        // left panel
        let lx, ly, lw, lh, rx, ry, rw, rh;
        if (!nar) { lx = 16; ly = 30; lw = w * 0.5 - 30; lh = h - 46; rx = w * 0.5 + 16; ry = 30; rw = w * 0.5 - 32; rh = h - 70; }
        else { lx = 12; ly = 26; lw = w - 24; lh = h * 0.55 - 40; rx = 30; ry = h * 0.55 + 40; rw = w - 50; rh = h * 0.45 - 128; }
        const sc = Math.min(lw, lh) / 2.5;
        const ox = lx + lw / 2, oy = ly + lh / 2;
        const PX = (p) => ox + p.x * sc + Math.sin(t * 0.6 + p.x * 9) * 1.2, PY = (p) => oy - p.y * sc + Math.cos(t * 0.5 + p.y * 9) * 1.2;
        V.text(g, 'TOY SPACE (2D)', lx, ly - 10, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        g.strokeStyle = C.grid; g.lineWidth = 1;
        g.beginPath(); g.moveTo(ox - sc * 1.1, oy); g.lineTo(ox + sc * 1.1, oy); g.moveTo(ox, oy - sc * 1.1); g.lineTo(ox, oy + sc * 1.1); g.stroke();
        V.dot(g, ox, oy, 3, C.muted);
        const p = Hd.pointer;
        if (p.inside && p.x < lx + lw && p.y < ly + lh + 10) {
          let bi = -1, bd = 26 * 26;
          P.forEach((q, i) => { const dx = PX(q) - p.x, dy = PY(q) - p.y; const dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; bi = i; } });
          if (bi >= 0) S.sel = bi;
        }
        const sel = P[S.sel];
        const nb = P.map((q, i) => ({ i, c: cosv(sel, q) })).filter((o) => o.i !== S.sel).sort((a, b) => b.c - a.c).slice(0, 3);
        // ray of the selected token
        V.line(g, ox, oy, PX(sel), PY(sel), { color: V.alpha(C.c, 0.5), dash: [3, 4], lw: 1.2 });
        nb.forEach((o) => {
          const q = P[o.i];
          V.line(g, PX(sel), PY(sel), PX(q), PY(q), { color: V.alpha(C.ink, 0.55), lw: 1.4 });
        });
        if (S.an) {
          const hm = P.find((q) => q.w === '␣man'), mu = P.find((q) => q.w === '␣woman'), re = P.find((q) => q.w === '␣king'), ra = P.find((q) => q.w === '␣queen');
          V.arrow(g, PX(hm), PY(hm), PX(re), PY(re), { color: C.d, lw: 2, head: 8 });
          V.arrow(g, PX(mu), PY(mu), PX(ra), PY(ra), { color: C.d, lw: 2, head: 8 });
          V.arrow(g, PX(hm), PY(hm), PX(mu), PY(mu), { color: V.alpha(C.f, 0.8), lw: 1.5, head: 6, dash: [4, 3] });
          V.arrow(g, PX(re), PY(re), PX(ra), PY(ra), { color: V.alpha(C.f, 0.8), lw: 1.5, head: 6, dash: [4, 3] });
          V.text(g, '“royalty”', (PX(hm) + PX(re)) / 2 + 6, (PY(hm) + PY(re)) / 2 + 14, { size: 11.5, color: C.d, weight: 600 });
        }
        P.forEach((q, i) => {
          const col = C[q.col];
          const isSel = i === S.sel, isNb = nb.some((o) => o.i === i);
          V.dot(g, PX(q), PY(q), isSel ? 6 : 4.2, col);
          if (isSel) V.glow(g, PX(q), PY(q), 18, col, 0.6);
          const sd = q.side;
          const tx = sd === 'r' ? PX(q) + 7 : sd === 'l' ? PX(q) - 7 : PX(q);
          const ty = sd === 'b' ? PY(q) + 16 : PY(q) + 4;
          V.text(g, q.w, tx, ty, { size: isSel || isNb ? 12 : 11, font: 'mono', color: isSel || isNb ? C.ink : C.muted, weight: isSel ? 700 : 400, align: sd === 'r' ? 'left' : sd === 'l' ? 'right' : 'center' });
        });
        V.text(g, 'neighbors of ' + sel.w + ': ' + nb.map((o) => P[o.i].w + ' ' + nf(o.c, 2)).join(' · '), lx, ly + lh + 8, { size: 12, font: 'mono', color: C.ink, weight: 600, maxW: lw });
        // right panel: cosine density
        const D = dens[S.di];
        V.text(g, 'COSINE BETWEEN TWO RANDOM DIRECTIONS IN ℝ^' + d, rx, ry - 10, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: rw });
        const X = (c) => rx + ((c + 1) / 2) * rw, Y = (v) => ry + rh - (Math.min(v, D.mx * 1.05) / (D.mx * 1.1)) * rh;
        g.strokeStyle = C.grid; g.beginPath(); g.moveTo(rx, ry + rh); g.lineTo(rx + rw, ry + rh); g.stroke();
        [-1, -0.5, 0, 0.5, 1].forEach((c) => V.text(g, nf(c, c % 1 ? 1 : 0), X(c), ry + rh + 15, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        const sig = 1 / Math.sqrt(d);
        g.fillStyle = V.alpha(C.a, 0.12); g.fillRect(X(-sig), ry, X(sig) - X(-sig), rh);
        if (d > 3) {
          const yu = Y(0.5); V.line(g, X(-1), yu, X(1), yu, { color: V.alpha(C.muted, 0.7), dash: [5, 4], lw: 1.2 });
          // on phones the label moves from above the curve to below the axis
          if (!nar) V.text(g, 'reference d = 3 (uniform)', X(-0.98), yu - 6, { size: 11, color: C.muted });
          else V.text(g, 'dashed: reference d = 3 (uniform)', rx, ry + rh + 68, { size: 11, color: C.muted, maxW: rw + 16 });
        }
        g.beginPath(); g.moveTo(X(-1), ry + rh);
        D.ys.forEach((v, i) => g.lineTo(X(-1 + (2 * i) / NG), Y(v)));
        g.lineTo(X(1), ry + rh); g.closePath();
        const grad = g.createLinearGradient(0, ry, 0, ry + rh); grad.addColorStop(0, V.alpha(C.a, 0.55)); grad.addColorStop(1, V.alpha(C.a, 0.08));
        g.fillStyle = grad; g.fill();
        g.strokeStyle = C.a; g.lineWidth = 2; g.beginPath();
        D.ys.forEach((v, i) => { const x = X(-1 + (2 * i) / NG), y = Y(v); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }); g.stroke();
        const lab = d === 4096 ? 'GLM / DeepSeek (d = 4096)' : d === 2560 ? 'Qwen (d = 2560)' : d === 2 ? 'in the plane: any angle' : d === 3 ? 'in 3D: uniform' : '';
        V.text(g, 'σ ≈ ' + nf(sig, sig < 0.1 ? 4 : 3) + (d >= 64 ? '   ·   99.7% of pairs in ±' + nf(3 * sig, 3) : ''), rx, ry + rh + 34, { size: 12.5, font: 'mono', color: C.ink, weight: 600 });
        if (lab && !nar) V.text(g, lab, rx + rw, ry + 14, { size: 12, color: C.a, align: 'right', weight: 600 });
        else if (lab) V.text(g, lab, rx, ry + rh + 52, { size: 12, color: C.a, weight: 600, maxW: rw + 16 });
        // random points falling (playful effect)
        const rr = V.rng(7 + S.di);
        for (let k = 0; k < 24; k++) {
          const z = (rr() + rr() + rr() + rr() - 2) * Math.sqrt(3);
          const c = V.clamp(d <= 3 ? rr() * 2 - 1 : z * sig, -1, 1);
          const ph = (t * 0.35 + k / 24) % 1;
          V.dot(g, X(c), ry + ph * rh, 2.2, V.alpha(C.c, 0.7 * (1 - ph)));
        }
      });
    })();

    /* =====================================================================
       Figure 8 — Lookup × sweep
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-varredura');
      if (!fig) return;
      tall(fig, 600);
      const REAL = { glm: [154880, 4096, 'GLM'], qwen: [248320, 2560, 'Qwen'], ds: [129280, 4096, 'DeepSeek'] };
      const S = { m: 'qwen', tied: false, mtp: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { S.m = v; upd(); } });
      UI.toggle(ctl, { label: 'tied', value: false, onChange: (v) => { S.tied = v; upd(); } });
      UI.toggle(ctl, { label: 'MTP=3 drafter', value: false, onChange: (v) => { S.mtp = v; upd(); } });
      const rM = UI.readout(ctl, 'E + W_U memory', '—');
      const rR = UI.readout(ctl, 'lm_head read per step', '—');
      const rG = UI.readout(ctl, 'per GPU (TP=2)', '—');
      const upd = () => {
        const [Vv, d] = REAL[S.m];
        const mat = Vv * d * 2, reads = S.mtp ? 4 : 1;
        rM.set(gb(mat * (S.tied ? 1 : 2)) + ' (' + gib(mat * (S.tied ? 1 : 2)) + ')');
        rR.set(reads + ' × ' + gb(mat) + ' (' + gib(mat) + ')');
        rG.set(gib((reads * mat) / 2));
      };
      upd();
      const rnd = V.rng(3);
      const rowsLit = Array.from({ length: 64 }, () => rnd.int(56));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        const [Vv, d, nm] = REAL[S.m];
        const nar = narrow(w);
        const reads = S.mtp ? 4 : 1;
        const stepT = 2.6;
        const ph = (t % stepT) / stepT;
        const stepIdx = Math.floor(t / stepT);
        const barTop = 44, barH = nar ? h - 270 : h - 220;
        const bw = nar ? 46 : 62;
        const eX = nar ? 22 : 40, uX = nar ? w - bw - 22 - 60 : w * 0.62;
        const NR = 56;
        const colE = S.tied ? C.b : C.a, colU = S.tied ? C.b : C.d;
        // bars
        const drawBar = (x, col, label) => {
          for (let i = 0; i < NR; i++) {
            g.fillStyle = V.alpha(col, 0.12 + 0.05 * ((i * 13) % 3));
            g.fillRect(x, barTop + (i * barH) / NR, bw, barH / NR - 1);
          }
          g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1.2; g.strokeRect(x, barTop, bw, barH);
          V.text(g, label, x + bw / 2, barTop - (nar ? 10 : 22), { size: 12, font: 'mono', weight: 700, color: C.ink, align: 'center' });
          if (!nar) V.text(g, '[' + nf(Vv) + ' × ' + d + ']', x + bw / 2, barTop - 8, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        };
        drawBar(eX, colE, 'E');
        drawBar(uX, colU, 'W_U');
        // embedding: one row per step
        const lit = rowsLit[stepIdx % rowsLit.length];
        const ly = barTop + (lit * barH) / NR;
        g.fillStyle = V.alpha(colE, 0.95); g.fillRect(eX - 3, ly, bw + 6, barH / NR + 1);
        V.glow(g, eX + bw / 2, ly, 22, colE, 0.8);
        // layers box
        const lx = eX + bw + (nar ? 16 : 50), lw = uX - lx - (nar ? 16 : 50);
        const lyC = barTop + barH * 0.35, lhC = barH * 0.3;
        if (lw > 40) {
          V.box(g, lx, lyC, lw, lhC, { r: 12, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.5) });
          V.text(g, nar ? 'layers' : 'N Transformer layers', lx + lw / 2, lyC + lhC / 2 + 4, { size: 12, color: C.muted, align: 'center', maxW: lw - 8 });
          // vector x traveling
          const k = Math.min(1, ph / 0.3);
          const px = V.lerp(eX + bw, lx, V.ease.inOut(k)), py = V.lerp(ly, lyC + lhC / 2, V.ease.inOut(k));
          if (ph < 0.3) for (let j = 0; j < 6; j++) { g.fillStyle = V.alpha(colE, 0.9); g.fillRect(px + j * 6, py - 4, 5, 8); }
          const k2 = Math.max(0, Math.min(1, (ph - 0.3) / 0.12));
          if (ph >= 0.3 && ph < 0.42) { const qx = V.lerp(lx + lw, uX, k2); for (let j = 0; j < 6; j++) { g.fillStyle = V.alpha(C.c, 0.9); g.fillRect(qx - 36 + j * 6, lyC + lhC / 2 - 4, 5, 8); } }
          V.text(g, 'x (d·2 B)', lx - 4, lyC - 8, { size: 11, font: 'mono', color: colE });
          V.text(g, 'h', lx + lw + 6, lyC - 8, { size: 11, font: 'mono', color: C.c });
        }
        // lm_head sweep
        const sp = Math.max(0, (ph - 0.42) / 0.58);
        const pass = Math.min(reads - 1, Math.floor(sp * reads));
        const bp = sp * reads - pass;
        if (ph >= 0.42) {
          const by = barTop + bp * barH;
          for (let i = 0; i < NR; i++) { const ry = barTop + (i * barH) / NR; if (ry < by) { g.fillStyle = V.alpha(colU, 0.45); g.fillRect(uX, ry, bw, barH / NR - 1); } }
          g.fillStyle = V.alpha(colU, 0.95); g.fillRect(uX - 6, by - 2, bw + 12, 4);
          V.glow(g, uX + bw / 2, by, 26, colU, 0.9);
          const lab = S.mtp ? (pass === 0 ? 'target' : 'draft ' + pass) : 'target';
          V.text(g, lab + '  (read ' + (pass + 1) + '/' + reads + ')', uX + bw / 2, barTop + barH + 18, { size: 11.5, font: 'mono', color: colU, align: 'center', weight: 600 });
        }
        // logits
        const zx = uX + bw + 14, zw = nar ? 46 : Math.min(120, w - zx - 16);
        if (zw > 30) {
          if (!nar) V.text(g, 'logits z', zx + zw / 2, barTop - 8, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          for (let i = 0; i < NR; i += 2) {
            const ry = barTop + (i * barH) / NR;
            const show = ph >= 0.42 && ry < barTop + bp * barH;
            const val = 0.15 + 0.85 * Math.pow(Math.abs(Math.sin(i * 1.7 + stepIdx)), 6);
            g.fillStyle = show ? V.alpha(C.c, 0.3 + 0.6 * val) : V.alpha(C.muted, 0.1);
            g.fillRect(zx, ry, Math.max(2, zw * val), (barH / NR) * 2 - 2);
          }
        }
        if (S.tied) {
          const yy = barTop + barH + 34;
          V.line(g, eX + bw / 2, barTop + barH + 4, eX + bw / 2, yy, { color: C.b, dash: [4, 3] });
          V.line(g, uX + bw / 2, barTop + barH + 30, uX + bw / 2, yy, { color: C.b, dash: [4, 3] });
          V.line(g, eX + bw / 2, yy, uX + bw / 2, yy, { color: C.b, dash: [4, 3] });
          V.text(g, 'same matrix: W_U = E', (eX + uX + bw) / 2, yy - 6, { size: 12, color: C.b, align: 'center', weight: 600 });
        }
        // log bars of bytes
        const lb = h - (nar ? 130 : 96);
        const bx0 = 16, bwMax = w - 32 - (nar ? 0 : 230);
        const L = (b) => (Math.log10(b) - 3) / (10.5 - 3);
        const eB = d * 2, uB = Vv * d * 2 * reads;
        V.text(g, 'BYTES READ PER STEP (log scale, batch 1)', bx0, lb - 8, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        [[eB, 'embedding: ' + gib(eB), colE], [uB, 'lm_head: ' + (reads > 1 ? reads + ' × ' : '') + gib(Vv * d * 2) + (reads > 1 ? ' = ' + gib(uB) : ''), colU]].forEach(([b, lab, col], i) => {
          const y = nar ? lb + 18 + i * 36 : lb + 4 + i * 26;
          const ww = Math.max(4, bwMax * L(b));
          g.fillStyle = V.alpha(col, 0.55); V.rr(g, bx0, y, ww, 18, 5); g.fill();
          V.text(g, lab, nar ? bx0 : bx0 + ww + 8, nar ? y - 4 : y + 13, { size: 12, font: 'mono', color: C.ink, weight: 600 });
        });
        const note = S.m === 'qwen' && S.mtp ? 'Qwen, TP=2: 4 reads × 0.592 GiB = 2.37 GiB per GPU, ~26% of the ~9.2 GiB per step (recipe)' : 'ratio lm_head / embedding = V = ' + nf(Vv) + (reads > 1 ? ' × ' + reads + ' reads' : '');
        V.text(g, note, bx0, nar ? lb + 98 : lb + 70, { size: 12, color: S.m === 'qwen' && S.mtp ? C.d : C.muted, weight: 600, maxW: w - 32 });
        V.text(g, nm + (nar ? ' · [' + nf(Vv) + ' × ' + d + ']' : ''), w - 16, 18, { size: 12, weight: 700, color: C[S.m], align: 'right' });
      });
    })();

    /* =====================================================================
       Figure 9 — Where the 47k live (real data)
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-47k');
      if (!fig) return;
      tall(fig, 460);
      const BINS = [7298, 6127, 5263, 4515, 3956, 3599, 3244, 2900, 2669, 2449, 2254, 1408, 31, 17, 4, 3, 2, 1, 234, 150, 174, 108, 92, 84, 86, 96, 82, 97, 83, 80, 43];
      const BW = 8192, VV = 248320, TPB = 124160;
      const S = { tp: true, log: false, hover: -1, t0: null };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'TP=2 boundary (id 124,160)', value: true, onChange: (v) => { S.tp = v; } });
      UI.toggle(ctl, { label: 'log scale', value: false, onChange: (v) => { S.log = v; } });
      UI.readout(ctl, 'ids in the draft vocabulary', '47,149 of 248,320 (19%)');
      V.fig(fig, (g, w, h, t, dt, _S, Hd) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        if (S.t0 == null) S.t0 = t;
        const grow = V.ease.out(Math.min(1, (t - S.t0) / 1.4));
        const nar = narrow(w);
        const px = nar ? 44 : 60, py = 40, pw = w - px - 20, ph = h - py - (nar ? 96 : 76);
        const X = (id) => px + (id / (BINS.length * BW)) * pw;
        const cap = (i) => Math.min(BW, VV - i * BW);
        const Y = (v) => { if (S.log) { const lv = v <= 0 ? 0 : Math.log10(v + 1) / Math.log10(BW + 1); return py + ph - lv * ph; } return py + ph - (v / BW) * ph; };
        // axes
        g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(px, py); g.lineTo(px, py + ph); g.lineTo(px + pw, py + ph); g.stroke();
        (S.log ? [1, 10, 100, 1000, 8192] : [0, 2048, 4096, 6144, 8192]).forEach((v) => { const y = Y(v); V.text(g, nf(v), px - 6, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); g.strokeStyle = V.alpha(C.grid, 0.6); g.beginPath(); g.moveTo(px, y); g.lineTo(px + pw, y); g.stroke(); });
        [0, 50000, 100000, 150000, 200000, 248320].forEach((id) => V.text(g, id >= 1000 ? nf(id / 1000) + 'k' : '0', X(id), py + ph + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'token id →', px + pw, py + ph + 32, { size: 11, color: C.muted, align: 'right' });
        V.text(g, 'ids per bin of 8,192', px, py - 12, { size: 11, color: C.muted });
        const p = Hd.pointer;
        S.hover = -1;
        BINS.forEach((n, i) => {
          const xa = X(i * BW) + 1, xb = X(i * BW + cap(i)) - 1;
          g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(xa, Y(cap(i)), xb - xa, py + ph - Y(cap(i)));
          const hv = n * grow;
          const inR0 = (i + 1) * BW <= TPB;
          const col = !S.tp ? C.a : inR0 ? C.a : C.b;
          const yv = Y(hv);
          g.fillStyle = V.alpha(col, 0.75); g.fillRect(xa, yv, xb - xa, py + ph - yv);
          if (p.inside && p.x >= xa && p.x <= xb + 1) S.hover = i;
        });
        // specials
        const spx = X(248044);
        V.arrow(g, spx - 40, py + ph - 70, spx - 2, Y(43) - 4, { color: C.c, head: 6, curve: 12 });
        V.text(g, '33 special', spx - 44, py + ph - 76, { size: 11.5, color: C.c, align: 'right', weight: 600 });
        // drop
        if (!nar) {
          // label in the gap between the drop (id 98,304) and the TP boundary, without crossing the dashed line
          const dx = (X(98304) + X(TPB)) / 2, dy0 = py + ph - 46;
          V.text(g, 'drop', dx, dy0, { size: 11.5, color: C.muted, align: 'center' });
          V.text(g, 'at ~98k', dx, dy0 + 15, { size: 11.5, color: C.muted, align: 'center' });
          V.arrow(g, dx - 18, dy0 + 22, X(98304) - 3, py + ph - 4, { color: V.alpha(C.muted, 0.7), head: 5, curve: -6 });
        }
        if (S.tp) {
          const bx = X(TPB);
          V.line(g, bx, py - 4, bx, py + ph, { color: C.d, lw: 2, dash: [5, 4], dashOffset: -t * 20 });
          V.text(g, 'rank 0: 45,734 ids', bx - 8, py + 14, { size: 12, color: C.a, align: 'right', weight: 700 });
          V.text(g, 'rank 1: 1,415 ids', bx + 8, py + 14, { size: 12, color: C.b, weight: 700 });
          if (!nar) V.text(g, 'the step waits for the slowest rank', bx + 8, py + 32, { size: 11.5, color: C.muted });
        }
        // tooltip
        const i = S.hover >= 0 ? S.hover : 0;
        const a = i * BW, b = a + cap(i) - 1;
        const info = 'ids ' + nf(a) + '–' + nf(b) + ': ' + nf(BINS[i]) + ' in draft (' + nf((100 * BINS[i]) / cap(i), 1) + '% of bin)';
        V.text(g, (S.hover >= 0 ? '▸ ' : 'e.g.: ') + info, px, h - (nar ? 40 : 22), { size: 12.5, font: 'mono', color: C.ink, weight: 600, maxW: w - px - 10 });
        if (nar) V.text(g, '971 of the first 1,000 ids made it', px, h - 18, { size: 11.5, color: C.muted });
        else V.text(g, '971 of the first 1,000 ids made it', w - 20, h - 22, { size: 11.5, color: C.muted, align: 'right' });
      });
    })();

    /* =====================================================================
       Figure 10 — The vocabulary calculator
       ===================================================================== */
    (function () {
      const fig = root.querySelector('#fig-calc');
      if (!fig) return;
      tall(fig, 480);
      const PRE = {
        // tok = tokens gained per sequence at each step (GLM prose 3.39 = 2.39 accepted + 1; Qwen prose ~2.9; DeepSeek ~4.15)
        glm: { V: 154880, d: 4096, tp: 2, B: 1, pos: 8, reads: 2, tok: 3.39, nm: 'GLM-5.3-Flash' },
        qwen: { V: 248320, d: 2560, tp: 2, B: 1, pos: 4, reads: 4, tok: 2.9, nm: 'Qwen3.8-Flash-Next' },
        ds: { V: 129280, d: 4096, tp: 2, B: 6, pos: 7, reads: 2, tok: 4.15, nm: 'DeepSeek-V4-Flash' },
        mimo: { V: 152576, d: 4096, tp: 2, B: 1, pos: 1, reads: 1, tok: 1, nm: 'MiMo-V2.6-Flash (tutorial)' },
      };
      const S = Object.assign({ m: 'ds' }, PRE.ds);
      const ctl = UI.controls(fig);
      let sTP, sB, sP, sR, sK;
      UI.seg(ctl, { label: 'Preset', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'mimo', label: 'MiMo' }], value: 'ds', onChange: (v) => { S.m = v; const p = PRE[v]; S.V = p.V; S.d = p.d; sTP.set(p.tp); sB.set(p.B); sP.set(p.pos); sR.set(p.reads); sK.set(p.tok); H.redraw(); } });
      sTP = UI.seg(ctl, { label: 'TP', options: [1, 2, 3, 4], value: 2, onChange: (v) => { S.tp = v; H.redraw(); } });
      sB = UI.slider(ctl, { label: 'Batch', min: 1, max: 16, step: 1, value: S.B, onInput: (v) => { S.B = v; H.redraw(); } });
      sP = UI.slider(ctl, { label: 'Positions/step', min: 1, max: 8, step: 1, value: S.pos, onInput: (v) => { S.pos = v; H.redraw(); } });
      sR = UI.slider(ctl, { label: 'lm_head reads', min: 1, max: 4, step: 1, value: S.reads, onInput: (v) => { S.reads = v; H.redraw(); } });
      sK = UI.slider(ctl, { label: 'Tokens/step', min: 1, max: 7, step: 0.05, value: S.tok, fmt: (v) => nf(v, 2), onInput: (v) => { S.tok = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(); g.textRendering = 'optimizeSpeed';
        const nar = narrow(w);
        const Lp = lcm(64, S.tp);
        const Vp = Math.ceil(S.V / Lp) * Lp;
        const rows = Vp / S.tp;
        const pad = Vp - S.V;
        V.text(g, PRE[S.m].nm + ' · V = ' + nf(S.V) + ' · d = ' + S.d, 16, 22, { size: 12.5, weight: 700, color: C.ink });
        // rank strip
        const bx = 16, by = 36, bw = w - 32, bh = 34;
        const cols = [C.a, C.b, C.c, C.d];
        for (let r = 0; r < S.tp; r++) {
          const xa = bx + (r * rows / Vp) * bw, xb = bx + ((r + 1) * rows / Vp) * bw;
          g.fillStyle = V.alpha(cols[r], 0.3); g.fillRect(xa, by, xb - xa - 2, bh);
          g.strokeStyle = cols[r]; g.lineWidth = 1.2; g.strokeRect(xa, by, xb - xa - 2, bh);
          V.text(g, 'rank ' + r + (nar && S.tp > 2 ? '' : ': ' + nf(r * rows) + '–' + nf((r + 1) * rows - 1)), (xa + xb) / 2, by + bh / 2 + 4, { size: 11.5, font: 'mono', color: C.ink, align: 'center', maxW: xb - xa - 6 });
          for (let k = 0; k < 3; k++) { const u = (t * 0.2 + k / 3 + r * 0.13) % 1; V.dot(g, xa + u * (xb - xa - 4), by + bh + 6, 2, cols[r]); }
        }
        if (pad > 0) {
          const pw = Math.max(8, (pad / Vp) * bw);
          const xa = bx + bw - pw;
          g.save(); g.beginPath(); g.rect(xa, by, pw, bh); g.clip();
          g.strokeStyle = C.red; g.lineWidth = 1.5;
          for (let k = -bh; k < pw + bh; k += 5) { g.beginPath(); g.moveTo(xa + k, by + bh); g.lineTo(xa + k + bh, by); g.stroke(); }
          g.restore();
          V.text(g, '+' + pad + ' padding', bx + bw, by + bh + 22, { size: 11.5, font: 'mono', color: C.red, align: 'right' });
        }
        // result cards
        const embPerRank = rows * S.d * 2 * 2;
        const lmRead = S.reads * rows * S.d * 2;
        const gather = S.B * S.pos * Vp * 2;
        const tokStep = S.tok || 1;
        // on phones the sub-labels get shorter to fit the card without squeezing the font
        const cards = [
          ['padded vocabulary', nf(Vp), nar ? 'lcm(64,' + S.tp + ')=' + Lp + (pad ? ' · +' + pad : '') : 'lcm(64, ' + S.tp + ') = ' + Lp + (pad ? ' · +' + pad + ' rows' : ' · no padding')],
          ['rows per rank', nf(rows), nar ? 'id range / rank' : 'contiguous id ranges'],
          ['embedding + lm_head / rank', gib(embPerRank), nar ? 'BF16, 2 matrices' : 'BF16, 2 untied matrices'],
          ['lm_head read / step / rank', gib(lmRead), S.reads + (nar ? ' × ' : ' read(s) × ') + gib(rows * S.d * 2)],
          ['gathered logits / step', gb(gather), nar ? S.B + '×' + S.pos + '×' + nf(Vp) + '×2 B' : S.B + ' seq × ' + S.pos + ' pos × ' + nf(Vp) + ' × 2 B'],
          ['lm_head per new token', gib(lmRead / (S.B * tokStep)), nar ? '÷ (' + S.B + ' × ' + nf(tokStep, 2) + ')' : 'read ÷ (' + S.B + ' seq × ' + nf(tokStep, 2) + ' tok)'],
        ];
        const nc = nar ? 2 : 3;
        const cy = by + bh + 40;
        const cw = (w - 32 - (nc - 1) * 12) / nc, ch = Math.min(96, (h - cy - 12) / Math.ceil(cards.length / nc) - 12);
        cards.forEach(([k, v, sub], i) => {
          const x = 16 + (i % nc) * (cw + 12), y = cy + Math.floor(i / nc) * (ch + 12);
          V.box(g, x, y, cw, ch, { r: 12, fill: V.alpha(C.series[i % 6], 0.1), stroke: V.alpha(C.series[i % 6], 0.55) });
          V.text(g, k, x + 12, y + 20, { size: 11.5, color: C.muted, maxW: cw - 20 });
          V.text(g, v, x + 12, y + ch / 2 + 12, { size: nar ? 17 : 22, font: 'display', weight: 700, color: C.ink, maxW: cw - 20 });
          V.text(g, sub, x + 12, y + ch - 10, { size: 11, font: 'mono', color: C.muted, maxW: cw - 20 });
        });
      });
    })();
  },
});

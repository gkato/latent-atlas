LA.chapter({
  id: 'tres-modelos',
  part: 3,
  order: 18,
  title: 'Os três modelos dissecados',
  navTitle: 'Os três modelos',
  subtitle: 'GLM-5.3-Flash, Qwen3.8-Flash-Next e DeepSeek-V4-Flash lado a lado: mapa camada a camada, config.json campo a campo, parâmetros ativos, bytes de KV, linhagem e o que o motor precisa suportar.',
  minutes: 55,
  recipes: ['glm', 'qwen', 'ds'],
  tags: ['arquitetura', 'config.json', 'parâmetros ativos', 'KV cache', 'híbrido', 'MoE', 'linhagem'],
  goals: [
    'Ler um <code>config.json</code> de fronteira campo a campo e desenhar o <b>mapa de camadas</b> (mixer, FFN, extras)',
    'Calcular parâmetros <b>totais</b>, <b>ativos</b>, FLOPs por token e <b>experts distintos</b> lidos por passo de decode',
    'Calcular bytes de <b>KV por token</b> e de <b>estado por sequência</b> em modelos híbridos, e comparar com a atenção plena',
    'Transformar a arquitetura numa <b>lista de requisitos do motor</b> e reconhecer a linhagem de cada peça',
  ],
  prereqs: ['mla', 'atencao-esparsa', 'atencao-linear', 'hiperconexoes', 'ngram-mtp', 'multimodal'],
  summary: [
    'Os três compartilham o esqueleto decoder-only MoE, mas trocam o <b>mixer</b> (híbrido linear/esparso), a <b>largura do resíduo</b> (4 fluxos) e os <b>acessórios</b> (N-gram, MTP/rascunhadores, visão).',
    'Mapas: GLM = 45 camadas, 34 KDA + 11 MLA-DSA (<code>i mod 4 = 3</code>) + MTP no índice 45; Qwen = 48, 36 GDN + 12 QSA + MTP instanciada como 48; DeepSeek = 43, 2 SWA + 21 C4A + 20 C128A, hash em 0–2 e DSpark lendo 40–42.',
    '~96–97% dos parâmetros são experts roteados (311,6 de 321 B no GLM contando a MTP; 120,8 de 125,7 B no tronco do Qwen; 277 de 284 B no DeepSeek): a memória segue o total e o decode segue os bytes ativos (~16,7 / ~6,7 / ~13,2 B por token, contando o lm_head), que crescem com os experts distintos por passo: \\( E[1-(1-k/E)^t] \\).',
    'KV que cresce, por rank: 7.579 B/token (GLM), 6.656 B/token por GPU em FP8 (Qwen) e ~3.850 B/token (DeepSeek); fixo por sequência: ~76 MiB de estado KDA, ~27 MiB de GDN, ~3,4 MB de janelas.',
    'Linhagem: MLA (DeepSeek-V2) → GLM e V4; indexador DSA (V3.2) → kpool, QSA e C4A; atenção linear (Qwen3-Next, Kimi Linear) → GLM e Qwen; mHC → GLM e V4.',
    'A regra FLOPs ≈ 2·N<sub>ativo</sub> só conta os pesos: em contexto longo, a atenção e sobretudo o indexador (que pontua todas as L/4 chaves) passam a dominar, e por isso GLM e Qwen limitam cada query a 2048 tokens + cauda.',
    'Todo campo incomum do config vira requisito de motor. Nas três receitas, atenção esparsa, KV híbrido e rascunhador exigiram patches.',
  ],
  quiz: [
    { q: 'No GLM-5.3-Flash, <code>num_hidden_layers</code> = 45 e as camadas MLA seguem <code>i mod 4 = 3</code>. Quantas camadas MLA há no decoder e qual índice tem a camada MTP?', o: ['12 MLA; MTP no índice 44', '11 MLA (3, 7, …, 43); MTP no índice 45', '11 MLA (0, 4, …, 40); MTP no índice 46', '34 MLA; MTP no índice 45'], a: 1, e: 'De 0 a 44, os índices com resto 3 na divisão por 4 são 3, 7, …, 43: 11 camadas. A MTP é a 46ª entrada, índice 45, e o vLLM usa <code>num_hidden_layers</code> como seu índice. O "12 MLA" do README conta a MTP; 34 é o número de camadas KDA.' },
    { q: 'Por que quantizar <i>só</i> os experts roteados basta para os três modelos caberem em 2 nós?', o: ['Porque os experts são a parte mais sensível do modelo', 'Porque ~96–97% dos parâmetros estão nos experts roteados', 'Porque os experts não são lidos no decode', 'Porque a atenção não tem pesos'], a: 1, e: 'Experts roteados somam 311,6 de 321 B no GLM (43 camadas MoE: as 42 do decoder e a MTP; sem a MTP seriam 304,4 B, ≈ 95%), 121 de 126 B no tronco do Qwen e 277 de 284 B no DeepSeek. Levá-los a ~4 bits resolve a memória, e as partes pequenas e sensíveis (atenção, indexador, roteador, resíduo) podem continuar em 8 ou 16 bits.' },
    { q: 'O GLM verifica 8 tokens por passo com DFlash2 (k = 7). Com 288 experts e top-8, quantos experts distintos por camada se espera ler, sob roteamento uniforme?', o: ['8', '~57–58', '64', '288'], a: 1, e: '\\( 288[1-(1-8/288)^8] \\approx 58 \\) (≈ 57 na aproximação de 64 escolhas independentes usada nos outros capítulos, \\( 288[1-(1-1/288)^{64}] \\)). Não são 64 porque há sobreposição entre tokens, nem 8 porque cada token escolhe os seus. É por isso que a especulação rende menos num MoE do que num modelo denso: cada token extra acorda experts novos, e os bytes lidos por passo crescem.' },
    { q: 'No DeepSeek-V4-Flash, por que o KV cache por rank <b>não</b> diminui ao passar de TP=2 para TP=3?', o: ['Porque o KV é guardado em FP8', 'Porque <code>num_key_value_heads</code> = 1: há um único vetor de KV por token, replicado em todos os ranks', 'Porque TP=3 não é suportado', 'Porque as janelas SWA crescem com TP'], a: 1, e: 'TP divide cabeças entre GPUs, mas o V4 tem uma só "cabeça" de KV (K = V = 512). Cada rank guarda o cache inteiro, e o indexador também. TP=3 só ajuda indiretamente, liberando memória de pesos: o pool vai a ~35 GiB por rank.' },
    { q: 'A linha <code>fp8_ds_mla</code> do GLM tem 656 B, dos quais 128 B são zeros. Por quê?', o: ['É padding de alinhamento de página', 'O modelo é NoPE (<code>qk_rope_head_dim: 0</code>), mas o único kernel sparse-MLA do SM12x exige 64 dimensões de RoPE; a receita padda com zeros, o que é exato', 'São as escalas FP32 ainda não calculadas', 'É o espaço do indexador'], a: 1, e: 'Um bloco nulo soma 0 ao produto \\(q \\cdot k\\): o softmax não muda. O custo é ~24% a mais de KV nas 11 camadas DSA (656 em vez de ~528 B), mas evita escrever um kernel novo.' },
    { q: 'Na receita Qwen, o YaRN colocado em <code>--hf-overrides \'{"rope_parameters":…}\'</code> não tinha efeito. Qual era a correção?', o: ['Usar a flag antiga <code>--rope-scaling</code>', 'Aninhar o override em <code>"text_config"</code>', 'Aumentar <code>max_position_embeddings</code>', 'Desligar o MTP'], a: 1, e: 'O merge de overrides só desce em chaves que são configs aninhados. No <code>qwen4_exp</code>, o pai também expõe um <code>rope_parameters</code> simples: o override de topo era gravado no pai e nunca chegava ao <code>text_config</code>, que é o que o modelo lê. A receita (<code>start.sh:919-946</code>) passou a aninhar <code>{rope_type: "yarn", factor, original_max_position_embeddings: 262144}</code> em <code>text_config</code>. Precaução, não verificada no kit: o card oficial passa o dicionário completo, e o inventário aponta o risco de o dict ser substituído inteiro; por segurança, repita θ = 10⁷, a fração 0,25 e o mRoPE. E o 1M com YaRN ainda não foi validado no kit (<code>YARN_ENABLE</code> continua <code>false</code>). As outras opções não tocam no ponto: o problema era onde o override entrava.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, h = LA.h;

    /* ================================================================ estilo local */
    const css = `
      .t18-panel{padding:14px 18px;border-top:1px solid var(--line);background:color-mix(in srgb,var(--surface-2) 60%,transparent);font-size:14px;line-height:1.55;color:var(--text-2)}
      .t18-panel h5{margin:0 0 6px;font-family:var(--font-display);text-transform:uppercase;letter-spacing:.08em;font-size:13px;color:var(--text)}
      .t18-panel .chip{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;font-family:var(--font-mono);margin:0 6px 4px 0;border:1px solid var(--line-2);color:var(--text)}
      .t18-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px 22px;margin-top:8px}
      .t18-tlist{margin:4px 0 0;padding:0;list-style:none;font-size:12.5px}
      .t18-tlist li{display:flex;gap:8px;justify-content:space-between;border-bottom:1px dashed var(--line);padding:3px 0;margin:0}
      .t18-tlist li::marker{content:''}
      .t18-tlist li span:first-child{color:var(--text-2);min-width:0;overflow-wrap:anywhere}
      .t18-tlist li code{white-space:nowrap;font-size:11.5px}
      .t18-facts{margin:8px 0 0;padding-left:1.1em;font-size:13px}
      .t18-facts li{margin:.15em 0}
      .t18-grid2{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px}
      @media (max-width:720px){.t18-grid2{grid-template-columns:1fr}}
      .t18-json{font-family:var(--font-mono);font-size:12.3px;line-height:1.5;max-height:420px;overflow:auto;background:var(--code-bg);color:#D6F5E5;border-radius:12px;padding:10px 0;border:1px solid var(--line-2)}
      .t18-json .cm{display:block;padding:6px 14px 2px;color:#6E8C7C;font-style:italic}
      .t18-json button{display:block;width:100%;text-align:left;background:none;border:none;color:inherit;font:inherit;padding:2px 14px 2px 22px;cursor:pointer;white-space:normal;overflow-wrap:anywhere;border-left:3px solid transparent}
      .t18-json button:hover{background:rgba(63,244,155,.07)}
      .t18-json button.on{background:rgba(63,244,155,.16);border-left-color:#3FF49B}
      .t18-json .k{color:#89FFC5}.t18-json .v{color:#FCBB00}
      .t18-exp{font-size:14px;line-height:1.6;color:var(--text-2);min-width:0}
      .t18-exp .kk{font-family:var(--font-mono);color:var(--accent-text);font-size:13px;overflow-wrap:anywhere}
      .t18-exp .vv{font-family:var(--font-mono);color:var(--text);font-size:12.5px;display:block;margin:4px 0 8px;overflow-wrap:anywhere}
      .t18-exp .aff{margin-top:8px;font-size:12.5px;color:var(--text-3)}
      dl.la-kv.t18-kit{grid-template-columns:minmax(0,1fr);gap:2px 0;margin:.8em 0 .2em}
      dl.la-kv.t18-kit dd{margin:0 0 10px}
      .t18-req h5 .st{font-family:var(--font-mono);font-size:11px;padding:1px 8px;border-radius:99px;margin-left:8px;vertical-align:middle}
    `;
    const styleEl = document.createElement('style');
    styleEl.textContent = css;
    document.head.appendChild(styleEl);
    LA.onCleanup(() => styleEl.remove());

    /* ================================================================ utilidades */
    const nf = (x, d = 0) => (+x).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    const fB = (b) => {
      const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
      while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
      return nf(b, i === 0 ? 0 : b < 10 ? 2 : 1) + ' ' + u[i];
    };
    const NAME = { glm: 'GLM-5.3-Flash', qwen: 'Qwen3.8-Flash-Next', ds: 'DeepSeek-V4-Flash' };
    const SHORT = { glm: 'GLM', qwen: 'Qwen', ds: 'DeepSeek' };
    const MODEL_OPTS = [{ v: 'glm', label: 'GLM-5.3' }, { v: 'qwen', label: 'Qwen3.8' }, { v: 'ds', label: 'DS-V4' }];
    function wrapText(g, s, maxW, size, weight, font) {
      g.font = V.font(size, font || 'sans', weight || 400);
      const words = String(s).split(' '); const lines = []; let cur = '';
      for (const w of words) { const t = cur ? cur + ' ' + w : w; if (cur && g.measureText(t).width > maxW) { lines.push(cur); cur = w; } else cur = t; }
      if (cur) lines.push(cur);
      return lines;
    }
    function textW(g, s, size, font, weight) { g.font = V.font(size, font || 'sans', weight || 400); return g.measureText(String(s)).width; }
    function autoHeight(fig, fn) {
      const stage = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = stage.clientWidth || 800; const hh = Math.round(fn(w)); if (Math.abs(stage.offsetHeight - hh) > 2) stage.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(stage); LA.onCleanup(() => ro.disconnect());
      return apply;
    }
    const fitTo = (fig, need) => { const st = fig.querySelector('.la-fig-stage'); need = Math.ceil(need); if (Math.abs(st.offsetHeight - need) > 3) st.style.height = need + 'px'; };
    const hatch = (g, x, y, w, hh, col) => {
      g.save(); g.beginPath(); g.rect(x, y, w, hh); g.clip();
      g.strokeStyle = col; g.lineWidth = 1.2;
      for (let k = -hh; k < w + hh; k += 6) { g.beginPath(); g.moveTo(x + k, y + hh); g.lineTo(x + k + hh, y); g.stroke(); }
      g.restore();
    };

    /* ================================================================ dados: camadas */
    function layersOf(m) {
      const L = [];
      if (m === 'glm') {
        const taps = [5, 14, 24, 33, 42];
        for (let i = 0; i < 45; i++) L.push({ i, mix: i % 4 === 3 ? 'dsa' : 'kda', ffn: i < 3 ? 'dense' : 'moe', tap: taps.includes(i), role: i >= 15 ? 'edit' : 'anchor' });
        L.push({ i: 45, mix: 'dsa', ffn: 'moe', extra: 'mtp', tag: 'MTP', role: 'edit' });
      } else if (m === 'qwen') {
        for (let i = 0; i < 48; i++) L.push({ i, mix: (i + 1) % 4 === 0 ? 'qsa' : 'gdn', ffn: 'moe', ple: i === 1, role: (i + 1) % 4 === 0 ? 'edit' : null });
        L.push({ i: 48, mix: 'qsa', ffn: 'moe', extra: 'mtp', tag: 'MTP' });
      } else {
        for (let i = 0; i < 43; i++) {
          const cr = i < 2 ? 0 : i % 2 === 0 ? 4 : 128;
          L.push({ i, mix: cr === 0 ? 'swa' : cr === 4 ? 'c4a' : 'c128a', ffn: i < 3 ? 'hash' : 'moe', tap: i >= 40, role: i >= 10 ? 'edit' : null });
        }
        for (let s = 0; s < 3; s++) L.push({ i: 43 + s, mix: 'swa', ffn: 'moe', extra: 'dspark', tag: 'DS' + s, stage: s });
      }
      return L;
    }
    const LAYERS = { glm: layersOf('glm'), qwen: layersOf('qwen'), ds: layersOf('ds') };
    const MIXNAME = { kda: 'KDA', gdn: 'GDN', dsa: 'MLA-DSA', qsa: 'QSA', c4a: 'C4A', c128a: 'C128A', swa: 'SWA', full: 'GA' };
    // 'full' = atenção global densa (sem indexador), usada só no exercício do MiMo
    const mixCol = (C, t) => (t === 'kda' || t === 'gdn') ? C.a : (t === 'dsa' || t === 'qsa' || t === 'c4a') ? C.b : t === 'c128a' ? C.c : t === 'full' ? C.e : C.d;
    const ffnCol = (C, f) => f === 'dense' ? V.alpha(C.ink, 0.9) : f === 'hash' ? C.f : V.alpha(C.ink, 0.38);
    // bytes de KV que crescem com o contexto, por token, por camada, por rank
    const KVB = { kda: 0, gdn: 0, dsa: 689, qsa: 512, c4a: 179, c128a: 4.56, swa: 0 };
    // rampa sequencial de um só matiz (claro = barato, escuro/saturado = caro), em escala log
    const kvCol = (C, b) => b > 0 ? V.cmap(0.22 + 0.78 * Math.min(1, Math.log10(1 + b) / Math.log10(700)), 'seq') : V.alpha(C.muted, 0.35);

    /* ---------- desenho compartilhado de uma faixa de camadas ---------- */
    function stripGeom(w, L, o) {
      const pad = o.pad == null ? 14 : o.pad, gap = o.gap == null ? 3 : o.gap;
      const nMain = L.filter((l) => !l.extra).length, nEx = L.length - nMain;
      const slots = nMain + (nEx ? nEx + 1 : 0);
      const W = w - 2 * pad;
      let perRow = slots, cw = (W - gap * (slots - 1)) / slots;
      if (cw < o.minCell) { perRow = Math.max(6, Math.floor((W + gap) / (o.minCell + gap))); cw = (W - gap * (perRow - 1)) / perRow; }
      cw = Math.min(cw, o.maxCell || 30);
      const rows = Math.ceil(slots / perRow);
      const rowH = o.mh + o.fh + (o.labelH || 0) + (o.markH || 0) + 8;
      const pos = L.map((l, k) => {
        const s = l.extra ? nMain + 1 + (k - nMain) : k;
        const r = Math.floor(s / perRow), c = s % perRow;
        return { x: pad + c * (cw + gap), y: o.y + r * rowH + (o.markH || 0), l };
      });
      return { pos, cw, rows, rowH, height: rows * rowH, perRow };
    }
    function drawStrip(g, G, o) {
      const C = V.C();
      const { pos, cw } = G;
      pos.forEach((p, k) => {
        const l = p.l;
        const lit = o.lit ? o.lit(l) : true;
        const a = lit ? 1 : 0.16;
        let mc = mixCol(C, l.mix);
        if (o.mode === 'kv') mc = kvCol(C, KVB[l.mix]);
        if (o.mode === 'role') { mc = l.role === 'edit' ? C.c : l.role === 'anchor' ? V.alpha(C.muted, 0.55) : V.alpha(C.muted, 0.25); }
        g.globalAlpha = a;
        V.box(g, p.x, p.y, cw, o.mh, { r: Math.min(4, cw / 4), fill: V.alpha(mc, l.extra ? 0.45 : 0.85) });
        if (l.extra) { g.setLineDash([3, 3]); V.rr(g, p.x - 1, p.y - 1, cw + 2, o.mh + o.fh + 4, 4); g.strokeStyle = C.c; g.lineWidth = 1.2; g.stroke(); g.setLineDash([]); }
        const fc = o.mode === 'role' && l.ffn !== 'hash' ? V.alpha(C.ink, 0.25) : ffnCol(C, l.ffn);
        V.box(g, p.x, p.y + o.mh + 2, cw, o.fh, { r: Math.min(3, cw / 5), fill: fc });
        g.globalAlpha = 1;
        // marcadores
        const mx = p.x + cw / 2, my = p.y - 6;
        if (l.tap && (!o.lit || lit)) { g.fillStyle = C.d; g.beginPath(); g.moveTo(mx, my - 4); g.lineTo(mx + 4, my); g.lineTo(mx, my + 4); g.lineTo(mx - 4, my); g.closePath(); g.fill(); }
        if (l.ple && (!o.lit || lit)) V.text(g, '★', mx, my + 4, { size: 12, align: 'center', color: C.f });
        if (o.labels) {
          const show = l.extra ? (l.extra === 'mtp' || l.stage === 1) : (cw >= 21 || l.i % 4 === 0 || o.sel === k);
          if (show) V.text(g, l.extra ? (l.extra === 'mtp' ? 'MTP' : 'DSpark') : String(l.i), mx, p.y + o.mh + o.fh + 14, { size: cw < 16 ? 9.5 : 10.5, font: 'mono', align: 'center', color: o.sel === k ? C.ink : C.muted });
        }
        if (o.sel === k) { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, p.x - 2.5, p.y - 2.5, cw + 5, o.mh + o.fh + 7, 5); g.stroke(); }
        if (o.hover === k && o.sel !== k) { g.strokeStyle = V.alpha(C.ink, 0.5); g.lineWidth = 1.2; V.rr(g, p.x - 2, p.y - 2, cw + 4, o.mh + o.fh + 6, 5); g.stroke(); }
      });
    }
    const hitStrip = (G, px, py, o) => G.pos.findIndex((p) => px >= p.x - 1 && px <= p.x + G.cw + 1 && py >= p.y - 8 && py <= p.y + o.mh + o.fh + 16);

    /* ================================================================ FIG 1: chassi */
    (function () {
      const fig = root.querySelector('#fig-chassi');
      const S = { m: 'glm', spread: 1, tgt: 1, choice: [], tok: 0 };
      const P = {
        classic: { name: 'Transformer clássico', lanes: 1, pat: ['A', 'A', 'A', 'A'], patL: ['atenção', 'atenção', 'atenção', 'atenção'], mixLabel: 'atenção plena em todas as camadas', mixSub: 'KV cresce em todas', E: 0, k: 0, ffnLabel: 'MLP densa', ffnSub: 'todo token usa tudo', resid: 'resíduo único · pré-norma', extras: ['embedding', 'lm_head'] },
        glm: { name: 'GLM-5.3-Flash', lanes: 4, pat: ['L', 'L', 'L', 'S'], patL: ['KDA', 'KDA', 'KDA', 'MLA-DSA'], mixLabel: '3 KDA : 1 MLA-DSA', mixSub: 'KV só nas 11 MLA', E: 288, k: 8, ffnLabel: '288 experts · top-8 + 1', ffnSub: 'sigmoide · noaux_tc · ×2,5', resid: 'mHC: 4 × 4096, mistura Sinkhorn', extras: ['MTP (camada 45)', 'DFlash2 externo', 'ViT 24 blocos', '3 MLPs densas'] },
        qwen: { name: 'Qwen3.8-Flash-Next', lanes: 4, pat: ['L', 'L', 'L', 'S'], patL: ['GDN', 'GDN', 'GDN', 'QSA'], mixLabel: '3 GDN : 1 QSA', mixSub: 'KV só nas 12 QSA (+MTP)', E: 512, k: 10, ffnLabel: '512 experts · top-10 + 1', ffnSub: 'softmax · shared com gate σ', resid: 'Gated Residual: 4 × 2560', extras: ['N-gram 51 B (camada de índice 1)', 'MTP (1 camada, 3×)', 'ViT 27 blocos'] },
        ds: { name: 'DeepSeek-V4-Flash', lanes: 4, pat: ['W', 'C4', 'C128', 'C4'], patL: ['SWA', 'C4A', 'C128A', 'C4A'], mixLabel: 'SWA 128 + C4A / C128A', mixSub: 'KV comprimido ÷4 ou ÷128', E: 256, k: 6, ffnLabel: '256 experts · top-6 + 1', ffnSub: 'sqrtsoftplus · hash em 0–2', resid: 'mHC: 4 × 4096, mistura Sinkhorn', extras: ['DSpark (3 estágios)', 'hash nas camadas 0–2', 'ViT 32 camadas'] },
      };
      const rng = V.rng(18);
      const repick = () => { const p = P[S.m]; S.choice = []; if (!p.E) return; const set = new Set(); while (set.size < p.k) set.add(rng.int(p.E)); S.choice = Array.from(set); S.tok++; };
      repick();
      V.every(1100, repick);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: [{ v: 'classic', label: 'clássico' }].concat(MODEL_OPTS), value: 'glm', onChange: (v) => { S.m = v; S.tgt = P[v].lanes; repick(); } });
      UI.legend(ctl, [[V.C().a, 'mixer linear (estado fixo)'], [V.C().b, 'atenção esparsa / plena'], [V.C().c, 'KV ÷128'], [V.C().d, 'janela 128']]);
      const parts = Array.from({ length: 40 }, (_, i) => ({ u: i / 40, lane: i % 4, sp: 0.8 + (i % 5) * 0.08 }));
      V.fig(fig, (g, w, hh, t, dt) => {
        const C = V.C(), p = P[S.m];
        S.spread += (S.tgt - S.spread) * Math.min(1, dt * 4);
        const pad = 14, narrow = w < 560;
        // título
        V.text(g, p.name, pad, 24, { size: 15, font: 'display', weight: 600, color: S.m === 'classic' ? C.ink : C[S.m] });
        V.text(g, p.resid, w - pad, 24, { size: 11.5, font: 'mono', align: 'right', color: C.muted, maxW: w * 0.5 });
        // faixas do resíduo
        const ly = 56, lx0 = pad + (narrow ? 54 : 64), lx1 = w - pad - (narrow ? 50 : 64);
        const nL = 4, laneGap = 7;
        const laneY = (j) => ly + (j - 1.5) * laneGap * Math.max(0, (S.spread - 1) / 3);
        const visLanes = S.spread > 1.2 ? 4 : 1;
        for (let j = 0; j < (visLanes === 1 ? 1 : nL); j++) V.line(g, lx0, laneY(j), lx1, laneY(j), { color: V.alpha(C.a, 0.55), lw: 2 });
        V.box(g, pad, ly - 17, lx0 - pad - 6, 34, { r: 8, fill: V.alpha(C.e, 0.16), stroke: C.e, label: 'embed', size: 11 });
        V.box(g, lx1 + 6, ly - 17, w - pad - lx1 - 6, 34, { r: 8, fill: V.alpha(C.c, 0.16), stroke: C.c, label: 'head', size: 11 });
        parts.forEach((q) => {
          const u = (q.u + t * 0.07 * q.sp) % 1;
          const x = V.lerp(lx0, lx1, u), y = visLanes === 1 ? laneY(0) : laneY(q.lane);
          V.dot(g, x, y, 2.3, C.a);
        });
        // caixas
        const by = ly + 44, bh = narrow ? 150 : 176;
        const bw = (w - 2 * pad - 16) / 2;
        const mxB = { x: pad, y: by, w: bw, h: bh }, ffB = { x: pad + bw + 16, y: by, w: bw, h: bh };
        const arrowPair = (B, col, ph) => {
          const xr = B.x + B.w * 0.3, xw = B.x + B.w * 0.7;
          V.arrow(g, xr, ly + 12, xr, B.y - 2, { color: V.alpha(col, 0.8), lw: 1.5, head: 6, dash: [3, 3] });
          V.arrow(g, xw, B.y - 2, xw, ly + 12, { color: V.alpha(col, 0.8), lw: 1.5, head: 6, dash: [3, 3] });
          const f = (t * 0.6 + ph) % 1;
          V.dot(g, xr, V.lerp(ly + 12, B.y - 2, f), 2.6, col);
          V.dot(g, xw, V.lerp(B.y - 2, ly + 12, f), 2.6, col);
          V.text(g, 'lê', xr - 5, (ly + B.y) / 2 + 4, { size: 10.5, align: 'right', color: C.muted });
          V.text(g, 'escreve', xw + 5, (ly + B.y) / 2 + 4, { size: 10.5, color: C.muted });
        };
        arrowPair(mxB, C.b, 0); arrowPair(ffB, C.d, 0.5);
        V.box(g, mxB.x, mxB.y, mxB.w, mxB.h, { r: 12, fill: V.alpha(C.b, 0.07), stroke: V.alpha(C.b, 0.5) });
        V.box(g, ffB.x, ffB.y, ffB.w, ffB.h, { r: 12, fill: V.alpha(C.d, 0.06), stroke: V.alpha(C.d, 0.5) });
        V.text(g, 'MIXER · 4 camadas seguidas', mxB.x + 10, mxB.y + 17, { size: 10.5, font: 'mono', color: C.muted, maxW: mxB.w - 20 });
        V.text(g, 'FFN', ffB.x + 10, ffB.y + 17, { size: 10.5, font: 'mono', color: C.muted });
        // padrão de mixers
        const cur = Math.floor(t * 1.2) % 4;
        const tw = (mxB.w - 20 - 9) / 4, th = narrow ? 48 : 60, ty = mxB.y + 28;
        p.pat.forEach((k, j) => {
          const col = k === 'L' ? C.a : k === 'C128' ? C.c : k === 'W' ? C.d : C.b;
          const on = j === cur;
          const tx = mxB.x + 10 + j * (tw + 3);
          V.box(g, tx, ty - (on ? 3 : 0), tw, th, { r: 7, fill: V.alpha(col, on ? 0.5 : 0.22), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2 : 1 });
          V.text(g, p.patL[j], tx + tw / 2, ty + th / 2 + 4 - (on ? 3 : 0), { size: tw < 44 ? 9.5 : 11, font: 'mono', align: 'center', color: C.ink, maxW: tw - 4 });
          if (on) V.glow(g, tx + tw / 2, ty + th + 6, 10, col, 0.8);
        });
        V.text(g, p.mixLabel, mxB.x + 10, ty + th + 26, { size: narrow ? 11.5 : 13, weight: 600, color: C.ink, maxW: mxB.w - 20 });
        V.text(g, p.mixSub, mxB.x + 10, ty + th + 44, { size: 11.5, color: C.muted, maxW: mxB.w - 20 });
        // FFN
        if (!p.E) {
          const bx = ffB.x + 12, bw2 = ffB.w - 24, bh2 = narrow ? 44 : 56;
          const pulse = 0.35 + 0.25 * Math.sin(t * 3);
          V.box(g, bx, ty, bw2, bh2, { r: 8, fill: V.alpha(C.e, pulse), stroke: C.e, label: 'todos os pesos, todo token', size: 11 });
        } else {
          const cols = p.E === 512 ? 32 : p.E === 288 ? 24 : 16;
          const rows = Math.ceil(p.E / cols);
          const gw = ffB.w - 24, gh = th + 16;
          const cs = Math.min(gw / cols, gh / rows);
          const gx = ffB.x + 12, gy = ty;
          const lit = new Set(S.choice);
          for (let e = 0; e < p.E; e++) {
            const r = Math.floor(e / cols), c = e % cols;
            g.fillStyle = lit.has(e) ? C.d : V.alpha(C.ink, 0.13);
            g.fillRect(gx + c * cs + 0.5, gy + r * cs + 0.5, Math.max(1, cs - 1), Math.max(1, cs - 1));
          }
          const shy = gy + rows * cs + 4;
          V.box(g, gx, shy, rows * cs > 0 ? Math.min(gw, cols * cs) : gw, 6, { r: 3, fill: V.alpha(C.e, 0.35 + 0.35 * (0.5 + 0.5 * Math.sin(t * 4))) });
        }
        const fly = p.E ? ty + th + 44 : ty + th + 26;
        V.text(g, p.ffnLabel, ffB.x + 10, fly, { size: narrow ? 11.5 : 13, weight: 600, color: C.ink, maxW: ffB.w - 20 });
        V.text(g, p.ffnSub, ffB.x + 10, fly + 18, { size: 11.5, color: C.muted, maxW: ffB.w - 20 });
        // acessórios
        let ex = pad, ey = by + bh + 22;
        V.text(g, 'ACESSÓRIOS', pad, ey, { size: 10.5, font: 'mono', color: C.muted });
        ey += 10;
        p.extras.forEach((s) => {
          const ww = textW(g, s, 11.5) + 20;
          if (ex + ww > w - pad) { ex = pad; ey += 28; }
          V.box(g, ex, ey, ww, 22, { r: 11, fill: V.alpha(C.c, 0.12), stroke: V.alpha(C.c, 0.6), label: s, size: 11.5 });
          ex += ww + 8;
        });
        fitTo(fig, ey + 22 + 16);
      });
    })();

    /* ================================================================ FIG 2: mapa camada a camada */
    const INFO = {
      glm: {
        kda: { t: 'KDA · Kimi Delta Attention (linear)', rows: [['q_proj / k_proj / v_proj', '[8192, 4096] cada'], ['q/k/v_conv1d (causal, 4 taps)', '[8192, 1, 4]'], ['b_proj (β, força de escrita)', '[64, 4096]'], ['f_a → f_b (esquecimento por canal)', '[128, 4096] → [8192, 128]'], ['g_a → g_b (gate de saída)', '[128, 4096] → [8192, 128]'], ['A_log / dt_bias', '[64] / [8192] FP32'], ['o_proj', '[4096, 8192]']], facts: ['137.732.288 parâmetros por camada', 'Estado por request, por rank (TP=2): [32 cabeças, 128, 128] FP32 = 2.097.152 B + conv 245.760 B', 'KV que cresce com o contexto: 0 B/token'] },
        dsa: { t: 'MLA NoPE + indexador DSA (esparsa, top-2048)', rows: [['q_a_proj → q_b_proj', '[1536, 4096] → [16384, 1536]'], ['kv_a_proj_with_mqa (latente)', '[512, 4096]'], ['kv_b_proj (sobe K/V por cabeça)', '[32768, 512]'], ['o_proj', '[4096, 16384]'], ['indexer.wq_b / wk', '[4096, 1536] / [128, 4096]'], ['indexer.weights_proj', '[32, 4096]'], ['kpool_compress_gate / ape', '[128, 4096] / [4, 128]']], facts: ['MLA 117.442.560 + indexador 7.471.872 = 124.914.432 parâmetros', 'KV: 656 B/token (fp8_ds_mla: 512 FP8 + 16 escalas + 128 de zeros de RoPE) + 33 B do indexador', 'Top-k em 512 pools de 4 tokens (= 2048) + cauda'] },
        dense: { t: 'MLP densa SwiGLU', rows: [['gate_proj / up_proj', '[12288, 4096]'], ['down_proj', '[4096, 12288]']], facts: ['150.994.944 parâmetros; todo token usa tudo'] },
        moe: { t: 'MoE: 288 experts, top-8 + 1 compartilhado', rows: [['gate (roteador) / e_score_correction_bias', '[288, 4096] / [288] FP32'], ['shared gate/up · down', '[2048, 4096] · [4096, 2048]'], ['288 × expert gate/up · down', '[2048, 4096] · [4096, 2048]']], facts: ['288 × 25.165.824 = 7,25 B parâmetros por camada; 8 ativos = 201 M', 'Experts em EXL3 4 bpw (trellis + Hadamard): 12 MiB cada, 6 MiB por rank', 'Sigmoide + noaux_tc + ×2,5; roteador em FP32'] },
        mhc: 'mHC: hc_attn_fn / hc_ffn_fn [24, 16384] (4 pre + 4 post + 16 comb), 786.486 parâmetros por camada',
      },
      qwen: {
        gdn: { t: 'Gated DeltaNet (linear)', rows: [['in_proj_qkv (Q 2048 + K 2048 + V 6144)', '[10240, 2560]'], ['in_proj_z (gate de saída σ)', '[6144, 2560]'], ['in_proj_a (decay α) / in_proj_b (β)', '[48, 2560] cada'], ['conv1d (causal, 4 taps)', '[10240, 1, 4]'], ['A_log / dt_bias', '[48]'], ['out_proj', '[2560, 6144]']], facts: ['~57,96 M parâmetros por camada', 'Estado por request, por GPU (TP=2): 24 cabeças V × 128 × 128 = 1,5 MiB FP32 (0,75 MiB em BF16, como servido)', 'KV que cresce: 0 B/token'] },
        qsa: { t: 'QSA · Qwen Sparse Attention (GQA esparsa)', rows: [['q_proj (query + gate)', '[12288, 2560]'], ['k_proj / v_proj', '[512, 2560] cada'], ['o_proj', '[2560, 6144]'], ['q_norm / k_norm', '[256]'], ['indexer.index_qk_proj ((4+1) × 128)', '[640, 2560]']], facts: ['~49,8 M + indexador ~1,7 M parâmetros', 'KV: 2 cabeças × 256 × (K + V) = 1024 B/token em FP8 → 512 B por GPU', 'Indexador: micro-blocos de 4, top-512 blocos = 2048 tokens + cauda'] },
        moe: { t: 'MoE: 512 experts, top-10 + 1 compartilhado', rows: [['gate (roteador)', '[512, 2560]'], ['shared gate/up · down', '[640, 2560] · [2560, 640]'], ['shared_expert_gate (σ escalar)', '[1, 2560]'], ['512 × expert gate/up (NVFP4)', '[640, 1280] U8 → 640 × 2560'], ['512 × expert down (NVFP4)', '[2560, 320] U8 → 2560 × 640']], facts: ['4,915 M parâmetros por expert; 10 ativos por token', 'NVFP4 W4A4 com escala FP8 a cada 16: ~2,77 MB por expert', 'EP: 256 experts inteiros por GPU'] },
        mhc: 'Gated Residual (×2 por camada): hc_norm [10240], input_mix down [320, 10240] / up [10240, 320], block_inject [4, 10240]',
        ple: 'N-gram (PLE) nesta camada: 128 shards [2.500.012, 160] FP8 = 320.001.536 linhas (51,2 B params, 47,68 GiB) + key_proj [10240, 2560], value_proj [2560, 2560], conv1d dilatada [10240, 1, 4]',
      },
      ds: {
        base: [['wq_a (latente de query)', '4096 → 1024'], ['wq_b (64 × 512)', '1024 → 32768'], ['wkv (K = V, um vetor)', '4096 → 512'], ['attn_sink', '[64] FP32'], ['wo_a (8 grupos)', '[8192, 4096]: 8 × (4096 → 1024)'], ['wo_b', '8192 → 4096']],
        swa: { t: 'SWA · janela deslizante de 128', extra: [], facts: ['~107 M parâmetros (núcleo da atenção)', 'Sem compressão: só o anel de 128 entradas × 584 B = 74.752 B por sequência'] },
        c4a: { t: 'CSA / C4A · KV ÷4 + indexador top-512 + janela', extra: [['compressor C4 (wkv, wgate, ape [4, 1024])', '4096 → 2 × 512'], ['indexer.wq_b / weights_proj', '1024 → 64 × 128 / 4096 → 64'], ['compressor do indexador', '4096 → 2 × 128']], facts: ['~107 M + 19,1 M (compressor + indexador)', 'KV: 584 / 4 = 146 B/token + indexador 132 / 4 = 33 B/token', 'Lê 128 da janela + 512 entradas comprimidas por query'] },
        c128a: { t: 'HCA / C128A · KV ÷128, lido inteiro + janela', extra: [['compressor C128 (wkv, wgate, ape [128, 512])', '4096 → 512']], facts: ['~107 M + 4,3 M (compressor)', 'KV: 584 / 128 ≈ 4,6 B/token', 'A 1M lê 128 + 8.192 entradas por query'] },
        moe: { t: 'MoE: 256 experts, top-6 + 1 compartilhado', rows: [['gate + e_score_correction_bias', '4096 → 256'], ['256 × expert w1/w3 · w2 (MXFP4)', '4096 → 2048 · 2048 → 4096'], ['shared expert (FP8)', '4096 → 2048 → 4096']], facts: ['25,17 M parâmetros por expert; 6 roteados + 1 compartilhado ativos', 'FP4 E2M1 + escala E8M0 a cada 32: 13,4 MB por expert (6,7 MB por rank)', 'sqrtsoftplus + noaux_tc, ×1,5, sem grupos'] },
        hash: { t: 'MoE com hash routing (camadas 0–2)', rows: [['tid2eid (tabela fixa)', '[129280, 6] int32'], ['256 × expert (MXFP4)', '4096 → 2048 → 4096'], ['shared expert', '4096 → 2048 → 4096']], facts: ['Os 6 experts vêm de tid2eid[token_id]; os pesos ainda vêm do gate', 'Exige input_ids crus no forward (inclusive multimodal)', 'Tokens de imagem usam bias_vl, não o hash (#175)'] },
        mhc: 'mHC: hc_*_fn [24, 16384] FP32 (≈ 0,39 M por sub-bloco), Sinkhorn de 20 iterações',
      },
    };
    // matrizes principais de cada tipo de camada: [nome, linhas (saída), colunas (entrada)]
    const MATS = {
      glm: { kda: [['q/k/v_proj (×3)', 8192, 4096], ['f_b (decaimento)', 8192, 128], ['o_proj', 4096, 8192]], dsa: [['q_b_proj', 16384, 1536], ['kv_a (latente)', 512, 4096], ['kv_b', 32768, 512], ['o_proj', 4096, 16384]], dense: [['gate/up (×2)', 12288, 4096]], moe: [['expert gate/up', 2048, 4096, 1]] },
      qwen: { gdn: [['in_proj_qkv', 10240, 2560], ['in_proj_z (gate σ)', 6144, 2560], ['out_proj', 2560, 6144]], qsa: [['q_proj (q + gate)', 12288, 2560], ['k_proj / v_proj', 512, 2560], ['o_proj', 2560, 6144]], moe: [['expert gate/up', 640, 2560, 1]] },
      ds: { att: [['wq_b', 32768, 1024], ['wkv (K = V)', 512, 4096], ['wo_a (8 grupos)', 8192, 4096], ['wo_b', 4096, 8192]], moe: [['expert w1/w3', 2048, 4096, 1]], hash: [['expert w1/w3', 2048, 4096, 1]] },
    };
    const FMTS = { glm: [['BF16', 2], ['EXL3 4 bpw', 0.5]], qwen: [['BF16', 2], ['NVFP4', 0.5625]], ds: [['FP8', 1], ['MXFP4', 0.53125]] };
    const FMTNOTE = {
      glm: 'GLM: não roteados em BF16 no checkpoint (a receita os re-quantiza para FP8 no load, GLM53_DENSE_FP8=all, metade dos bytes); experts em EXL3 4 bpw, 12 MiB por expert (3 matrizes).',
      qwen: 'Qwen: densos em BF16; experts em NVFP4 (4 bits + escala FP8 a cada 16 = 4,5 bits), ~2,77 MB por expert.',
      ds: 'DeepSeek: projeções da atenção em FP8 (blocos 128×128); experts em MXFP4 (4 bits + escala E8M0 a cada 32 = 4,25 bits), 13,4 MB por expert.',
    };
    function matsFor(m, l) {
      const T = MATS[m], F = FMTS[m];
      const mix = m === 'ds' ? T.att : T[l.mix];
      const ffn = T[l.ffn] || [];
      return mix.concat(ffn).map(([n, R, K, e]) => ({ n, R, K, exp: !!e, fmt: F[e ? 1 : 0][0], bpp: F[e ? 1 : 0][1] }));
    }
    const renderList = (rows) => h('ul', { class: 't18-tlist' }, rows.map(([a, b]) => h('li', null, h('span', null, a), h('code', null, b))));
    function panelFor(m, l) {
      const C = V.C();
      const I = INFO[m];
      const chips = [];
      const hdr = l.extra ? (l.extra === 'mtp' ? `Camada ${l.i} · MTP` : `Estágio DSpark ${l.stage} (camada virtual ${l.i})`) : `Camada ${l.i}`;
      let mixT, mixRows, mixFacts;
      if (m === 'ds') { const X = I[l.mix]; mixT = X.t; mixRows = I.base.concat(X.extra); mixFacts = X.facts; } else { const X = I[l.mix]; mixT = X.t; mixRows = X.rows; mixFacts = X.facts; }
      const F = I[l.ffn];
      const roles = [];
      if (m === 'glm') {
        if (l.tap) roles.push(`DFlash2 lê a saída desta camada (hook ${l.i + 1}); o overlay materializa o hc_post e contrai os 4 fluxos para 4096.`);
        if (l.extra) roles.push('Camada MTP: MLA + indexador + MoE, com eh_proj [4096, 8192]; sem mHC. 7,43 B de parâmetros (inclui 288 experts). Usada com SPEC_METHOD=mtp (k = 2) e editada pelo ABLIT ("mtp-edit").');
        else roles.push(l.role === 'edit' ? 'ABLIT: camada "edit" (15–45): o_proj pode ser transplantado ou ortogonalizado.' : 'ABLIT: "safety-anchor" (0–14), mantida intacta.');
        if (l.mix === 'dsa' && !l.extra) roles.push('Entra no grupo de KV da MLA (bloco de 3.584 tokens) + cache do indexador.');
        if (l.mix === 'kda') roles.push('Entra num grupo de estado tipo Mamba (<span class="la-term" data-term="mamba-spec">MambaSpec</span>, modo align).');
      } else if (m === 'qwen') {
        if (l.ple) roles.push(INFO.qwen.ple);
        if (l.extra) roles.push('Camada MTP: no checkpoint é mtp.layers.0, no vLLM mtp.layers.48; experts FP8 em blocos 128×128; aplicada 3× como rascunhador. É a 13ª camada com KV.');
        if (l.role === 'edit') roles.push('Checkpoint ABLIT=1: o o_proj desta camada QSA é o único tensor alterado (12 camadas).');
        if (l.mix === 'gdn') roles.push('Estado tipo Mamba: define a página e o bloco de atenção de 1.664 tokens (SSM BF16).');
      } else {
        if (l.tap) roles.push('Alimenta o DSpark: a média dos 4 fluxos mHC desta camada entra em main_proj (3 × 4096 → 4096).');
        if (l.extra) roles.push('Estágio do DSpark (mtp.' + l.stage + '): SWA de 128 sobre as features do alvo + bloco [âncora + 4 ruídos] não causal + MoE completo de 256 experts; ~6,58 B por estágio.' + (l.stage === 0 ? ' O estágio 0 tem main_proj 12288 → 4096.' : ''));
        if (l.role === 'edit') roles.push('Alcance padrão da ablação em runtime (DSV4_ABLATE_LAYERS = 10–42, desligada por padrão).');
        if (!l.extra) roles.push('TP=2: termina com all-reduce do wo_b e do MoE (43 + 43 por passo).');
      }
      const col = C[m];
      const el = h('div', null,
        h('h5', null, hdr, ' · ', MIXNAME[l.mix], ' + ', l.ffn === 'dense' ? 'MLP densa' : l.ffn === 'hash' ? 'MoE (hash)' : 'MoE'),
        h('span', { class: 'chip', style: { borderColor: col } }, NAME[m]),
        h('span', { class: 'chip' }, 'KV/token: ' + (KVB[l.mix] ? nf(KVB[l.mix], KVB[l.mix] < 10 ? 1 : 0) + ' B' : '0 B')),
        h('span', { class: 'chip' }, INFO[m].mhc.split(':')[0]),
        h('div', { class: 't18-cols' },
          h('div', null, h('b', null, mixT), renderList(mixRows), h('ul', { class: 't18-facts' }, mixFacts.map((f) => h('li', null, f)))),
          h('div', null, h('b', null, F.t), renderList(F.rows), h('ul', { class: 't18-facts' }, F.facts.map((f) => h('li', null, f))))),
        h('ul', { class: 't18-facts', style: { marginTop: '10px' } }, [INFO[m].mhc].concat(roles).map((f) => h('li', { html: f }))));
      return el;
    }
    (function () {
      const fig = root.querySelector('#fig-mapa');
      const panel = fig.querySelector('#mapa-panel');
      const S = { m: 'glm', sel: 3, mode: 'tipo', hover: -1, anim: 0 };
      const defSel = { glm: 3, qwen: 1, ds: 2 };
      const upd = () => { panel.innerHTML = ''; panel.appendChild(panelFor(S.m, LAYERS[S.m][S.sel])); };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; S.sel = defSel[v]; upd(); } });
      UI.seg(ctl, { label: 'Colorir por', options: [{ v: 'tipo', label: 'tipo' }, { v: 'kv', label: 'KV/token' }, { v: 'role', label: 'papel' }], value: 'tipo', onChange: (v) => { S.mode = v; } });
      UI.button(ctl, '◀ camada', () => { S.sel = (S.sel - 1 + LAYERS[S.m].length) % LAYERS[S.m].length; upd(); });
      UI.button(ctl, 'camada ▶', () => { S.sel = (S.sel + 1) % LAYERS[S.m].length; upd(); });
      upd();
      const O = { mh: 30, fh: 11, labelH: 16, markH: 12, minCell: 13, maxCell: 30 };
      const rngm = V.rng(7);
      const expPick = { t: -1, set: [] };
      V.fig(fig, (g, w, hh, t, dt, _S, H) => {
        const C = V.C();
        const L = LAYERS[S.m];
        const pad = 14;
        // título e contagens
        const cnt = {}; L.forEach((l) => { if (!l.extra) cnt[l.mix] = (cnt[l.mix] || 0) + 1; });
        V.text(g, NAME[S.m], pad, 22, { size: 15, font: 'display', weight: 600, color: C[S.m] });
        const cs = Object.keys(cnt).map((k) => `${cnt[k]} ${MIXNAME[k]}`).join(' + ') + (S.m === 'glm' ? ' + MTP' : S.m === 'qwen' ? ' + MTP' : ' + 3 DSpark');
        V.text(g, cs, w - pad, 22, { size: 11.5, font: 'mono', align: 'right', color: C.muted, maxW: w * 0.58 });
        const G = stripGeom(w, L, Object.assign({ y: 40 }, O));
        // pulso do "token"
        const nMain = L.filter((l) => !l.extra).length;
        const pk = Math.floor((t * 7) % nMain);
        const pp = G.pos[pk];
        V.glow(g, pp.x + G.cw / 2, pp.y + O.mh / 2, 16, C.ink, 0.35);
        // interação
        const pt = H.pointer;
        S.hover = pt.inside ? hitStrip(G, pt.x, pt.y, O) : -1;
        if (pt.clicked && S.hover >= 0) { S.sel = S.hover; upd(); }
        drawStrip(g, G, Object.assign({ mode: S.mode, labels: true, sel: S.sel, hover: S.hover }, O));
        // legenda
        let ly = 40 + G.height + 14, lx = pad;
        const leg = S.mode === 'tipo' ? [[C.a, 'linear (KDA/GDN)'], [C.b, 'esparsa (MLA-DSA/QSA/C4A)'], [C.c, 'C128A'], [C.d, 'SWA'], [V.alpha(C.ink, 0.9), 'FFN densa'], [C.f, 'FFN hash'], [V.alpha(C.ink, 0.38), 'FFN MoE']]
          : S.mode === 'kv' ? Array.from(new Set(L.map((x) => KVB[x.mix]))).sort((a, b) => b - a).map((b) => [kvCol(C, b), b > 0 ? `${nf(b, b < 10 ? 1 : 0)} B/token (${L.filter((x) => KVB[x.mix] === b && !x.extra).map((x) => MIXNAME[x.mix])[0]})` : '0: estado fixo ou só janela'])
            : S.m === 'glm' ? [[C.c, 'ABLIT: camadas "edit" (15–45)'], [V.alpha(C.muted, 0.55), 'âncoras (0–14)'], [C.d, '◆ lida pelo DFlash2']]
              : S.m === 'qwen' ? [[C.c, 'ABLIT=1: o_proj das 12 QSA'], [C.f, '★ N-gram (índice 1, a 2ª camada)']]
                : [[C.c, 'ablação em runtime (10–42, opt-in)'], [C.f, 'FFN hash (0–2)'], [C.d, '◆ lida pelo DSpark']];
        leg.forEach(([c, s]) => {
          const ww = textW(g, s, 11) + 24;
          if (lx + ww > w - pad) { lx = pad; ly += 18; }
          V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: c });
          V.text(g, s, lx + 16, ly + 1, { size: 11, color: C.muted });
          lx += ww;
        });
        // esquema do bloco selecionado
        const l = L[S.sel];
        const top = ly + 22, bh = w < 560 ? 236 : 212, bot = top + bh;
        V.box(g, pad, top, w - 2 * pad, bh, { r: 12, fill: V.alpha(C.panel, 0.6), stroke: V.alpha(C.muted, 0.25) });
        const title = (l.extra ? (l.extra === 'mtp' ? 'MTP' : 'DSpark ' + l.stage) : 'camada ' + l.i) + (w >= 560 ? ' · como o token atravessa' : '');
        V.text(g, title, pad + 12, top + 18, { size: 11, font: 'mono', color: C.muted });
        const nl = (S.m === 'glm' && l.extra) ? 1 : 4;
        const laneY0 = top + 34;
        for (let j = 0; j < nl; j++) V.line(g, pad + 12, laneY0 + j * 6, w - pad - 12, laneY0 + j * 6, { color: V.alpha(C.a, 0.5), lw: 1.6 });
        for (let q = 0; q < 12; q++) { const u = (q / 12 + t * 0.09) % 1; V.dot(g, V.lerp(pad + 12, w - pad - 12, u), laneY0 + (q % nl) * 6, 2, C.a); }
        V.text(g, nl === 4 ? (S.m === 'qwen' ? 'Gated Residual · 4 × 2560' : 'mHC · 4 × 4096') : 'resíduo simples (MTP sem mHC)', w - pad - 12, top + 18, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        const iy = laneY0 + nl * 6 + 14, ih = bot - iy - 10;
        const iw = (w - 2 * pad - 36) / 2;
        const mB = { x: pad + 12, y: iy, w: iw, h: ih }, fB2 = { x: pad + 24 + iw, y: iy, w: iw, h: ih };
        const mc = mixCol(C, l.mix);
        V.box(g, mB.x, mB.y, mB.w, mB.h, { r: 9, fill: V.alpha(mc, 0.08), stroke: V.alpha(mc, 0.6) });
        V.box(g, fB2.x, fB2.y, fB2.w, fB2.h, { r: 9, fill: V.alpha(C.d, 0.06), stroke: V.alpha(C.d, 0.5) });
        V.text(g, MIXNAME[l.mix], mB.x + 8, mB.y + 16, { size: 12, weight: 600, color: mc });
        const inner = { x: mB.x + 8, y: mB.y + 24, w: mB.w - 16, h: mB.h - 32 };
        if (l.mix === 'kda' || l.mix === 'gdn') {
          const tight = inner.w < 220;
          const n = 8, cs2 = tight ? Math.min(8, inner.w / n) : Math.min((inner.h - 14) / n, (inner.w * 0.5) / n);
          const mx0 = inner.x, my0 = inner.y;
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const v = Math.sin(i * 1.3 + j * 0.7 + t * 1.6) * Math.cos(j * 0.9 - t * 0.8 + i * 0.3);
            g.fillStyle = V.cmap(v, 'div'); g.fillRect(mx0 + j * cs2, my0 + i * cs2, cs2 - 1, cs2 - 1);
          }
          if (tight) {
            const ty2 = my0 + n * cs2 + 14;
            V.text(g, 'estado S · 128 × 128 por cabeça', inner.x, ty2, { size: 10.5, weight: 600, color: C.ink, maxW: inner.w });
            wrapText(g, 'decai · corrige · escreve; tamanho fixo, não cresce com L', inner.w, 10.5).slice(0, 3).forEach((s2, k) => V.text(g, s2, inner.x, ty2 + 14 + k * 13, { size: 10.5, color: k === 0 ? C.muted : C.a }));
          } else {
            const tx = mx0 + n * cs2 + 10;
            V.text(g, 'estado S', tx, my0 + 12, { size: 11, weight: 600, color: C.ink });
            V.text(g, '128 × 128 por cabeça', tx, my0 + 27, { size: 10.5, color: C.muted, maxW: inner.x + inner.w - tx });
            V.text(g, 'S ← decai · corrige · escreve', tx, my0 + 42, { size: 10.5, color: C.muted, maxW: inner.x + inner.w - tx });
            V.text(g, 'tamanho fixo: não cresce com L', tx, my0 + 57, { size: 10.5, color: C.a, maxW: inner.x + inner.w - tx });
          }
        } else {
          const n = Math.max(12, Math.min(40, Math.floor(inner.w / 8)));
          const cw2 = inner.w / n, cy = inner.y + 8, chh = Math.min(22, inner.h * 0.35);
          const ph = Math.floor(t * 1.2);
          const r2 = V.rng(ph * 31 + (l.i || 0));
          for (let j = 0; j < n; j++) {
            let on = false, col = mc;
            if (l.mix === 'swa') on = j >= n - 5;
            else if (l.mix === 'c128a') on = true;
            else on = j >= n - 2 || r2() < 0.28;
            if (l.mix === 'c4a' && j >= n - 4) col = C.d;
            if ((l.mix === 'dsa' || l.mix === 'qsa') && j >= n - 1) col = C.d;
            const wcell = l.mix === 'c128a' && j < n - 4 ? cw2 : cw2;
            g.fillStyle = on ? V.alpha(col, 0.85) : V.alpha(C.ink, 0.1);
            V.rr(g, inner.x + j * cw2 + 0.5, cy, Math.max(1, wcell - 1.5), chh, 2); g.fill();
          }
          if (l.mix === 'c128a') { for (let j = 0; j < n - 4; j += 6) { g.strokeStyle = V.alpha(C.ink, 0.35); g.strokeRect(inner.x + j * cw2 + 0.5, cy - 1, cw2 * 6 - 1.5, chh + 2); } }
          const desc = { dsa: 'top-2048 do passado (512 pools de 4) + cauda', qsa: 'top-512 blocos de 4 (2048 tokens) + cauda', c4a: 'janela 128 + top-512 entradas ÷4', c128a: 'janela 128 + todas as entradas ÷128', swa: 'só os últimos 128 tokens' }[l.mix];
          const dl = wrapText(g, desc, inner.w, 10.5).slice(0, 3);
          dl.forEach((s2, k) => V.text(g, s2, inner.x, cy + chh + 16 + k * 13, { size: 10.5, color: C.muted }));
          V.text(g, 'KV cresce: ' + (KVB[l.mix] ? nf(KVB[l.mix], KVB[l.mix] < 10 ? 1 : 0) + ' B/token' : 'não (janela fixa)'), inner.x, cy + chh + 18 + dl.length * 13, { size: 10.5, color: KVB[l.mix] ? C.c : C.a, maxW: inner.w });
        }
        // FFN
        V.text(g, l.ffn === 'dense' ? 'MLP densa' : l.ffn === 'hash' ? 'MoE · hash' : 'MoE', fB2.x + 8, fB2.y + 16, { size: 12, weight: 600, color: C.d });
        const fin = { x: fB2.x + 8, y: fB2.y + 24, w: fB2.w - 16, h: fB2.h - 30 };
        if (l.ffn === 'dense') {
          V.box(g, fin.x, fin.y + 4, fin.w, Math.min(26, fin.h - 20), { r: 6, fill: V.alpha(C.e, 0.35 + 0.2 * Math.sin(t * 3)), stroke: C.e, label: '12288 · todos os pesos', size: 10.5 });
        } else {
          const E = S.m === 'glm' ? 288 : S.m === 'qwen' ? 512 : 256, k = S.m === 'glm' ? 8 : S.m === 'qwen' ? 10 : 6;
          const cols = E === 512 ? 32 : E === 288 ? 24 : 16, rows = Math.ceil(E / cols);
          const capL = wrapText(g, l.ffn === 'hash' ? `tid2eid[token] → ${k} experts` : `top-${k} de ${E} + 1 compartilhado`, fin.w, 10.5).slice(0, 2);
          const csz = Math.min(fin.w / cols, (fin.h - 6 - capL.length * 13) / rows);
          const step = Math.floor(t * 1.1);
          if (step !== expPick.t) { expPick.t = step; const st = new Set(); if (l.ffn === 'hash') { const r3 = V.rng(1000 + (step % 5)); while (st.size < k) st.add(r3.int(E)); } else { while (st.size < k) st.add(rngm.int(E)); } expPick.set = st; }
          for (let e = 0; e < E; e++) { const r = Math.floor(e / cols), c = e % cols; g.fillStyle = expPick.set.has(e) ? (l.ffn === 'hash' ? C.f : C.d) : V.alpha(C.ink, 0.13); g.fillRect(fin.x + c * csz + 0.4, fin.y + r * csz + 0.4, Math.max(1, csz - 0.8), Math.max(1, csz - 0.8)); }
          capL.forEach((s2, q) => V.text(g, s2, fin.x, fin.y + rows * csz + 14 + q * 13, { size: 10.5, color: C.muted }));
        }
        // matrizes da camada, em proporção (lado ∝ √dimensão)
        const mats = matsFor(S.m, l);
        const narrowM = w < 560;
        const maxH = narrowM ? 66 : 88, cM = maxH / Math.sqrt(32768);
        let my = bot + 22;
        const head = 'MATRIZES DESTA CAMADA · lado ∝ √dimensão · [saída × entrada] · bytes no formato servido';
        const hl = wrapText(g, head, w - 2 * pad, 10.5, 400, 'mono');
        hl.forEach((s2, q) => V.text(g, s2, pad, my + q * 14, { size: 10.5, font: 'mono', color: C.muted }));
        my += hl.length * 14 + 6;
        const cells = mats.map((mt) => {
          const rw = Math.max(2, cM * Math.sqrt(mt.K)), rh = Math.max(2, cM * Math.sqrt(mt.R));
          const l1 = mt.n, l2 = `[${nf(mt.R)} × ${nf(mt.K)}]`, l3 = `${fB(mt.R * mt.K * mt.bpp)} · ${mt.fmt}`;
          const lw2 = Math.max(textW(g, l1, 11, 'sans', 600), textW(g, l2, 10.5, 'mono'), textW(g, l3, 10.5, 'mono'));
          return { mt, rw, rh, l1, l2, l3, cw: Math.max(rw, lw2) };
        });
        const rowHM = maxH + 8 + 3 * 13 + 14;
        let cx = pad, cy = my;
        cells.forEach((c, k) => {
          if (cx > pad && cx + c.cw > w - pad) { cx = pad; cy += rowHM; }
          const rx = cx, ry = cy + maxH - c.rh;
          const col = c.mt.exp ? C.d : mc;
          // vetor de entrada (em cima) e de saída (à esquerda), pulsando
          const pu = 0.35 + 0.35 * Math.sin(t * 2.4 + k);
          g.fillStyle = V.alpha(col, pu); g.fillRect(rx, ry - 5, c.rw, 3);
          g.fillStyle = V.alpha(col, 0.2); g.fillRect(rx, ry, c.rw, c.rh);
          g.strokeStyle = V.alpha(col, 0.9); g.lineWidth = 1.2; g.strokeRect(rx + 0.5, ry + 0.5, Math.max(1, c.rw - 1), Math.max(1, c.rh - 1));
          // varredura: a entrada percorre as colunas, a saída acende de cima para baixo
          const u = (t * 0.4 + k * 0.23) % 1;
          g.fillStyle = V.alpha(col, 0.55); g.fillRect(rx + u * Math.max(0, c.rw - 2), ry, Math.max(1.5, c.rw * 0.05), c.rh);
          g.fillStyle = V.alpha(col, 0.35); g.fillRect(rx, ry, c.rw, c.rh * u);
          V.text(g, c.l1, rx, cy + maxH + 20, { size: 11, weight: 600, color: C.ink });
          V.text(g, c.l2, rx, cy + maxH + 33, { size: 10.5, font: 'mono', color: C.muted });
          V.text(g, c.l3, rx, cy + maxH + 46, { size: 10.5, font: 'mono', color: c.mt.exp ? C.d : C.ink });
          cx += c.cw + (narrowM ? 14 : 22);
        });
        let fy = cy + rowHM;
        const note = FMTNOTE[S.m];
        const nl2 = wrapText(g, note, w - 2 * pad, 10.5);
        nl2.forEach((s2, q) => V.text(g, s2, pad, fy + q * 13, { size: 10.5, color: C.muted }));
        fy += nl2.length * 13;
        fitTo(fig, fy + 10);
      });

    })();

    /* ================================================================ FIG 3: config.json guiado */
    const all = () => true, main = (l) => !l.extra, none = () => false;
    const CFG = {
      glm: [
        { c: 'identidade' },
        { k: 'architectures', v: '["Glm5NextForConditionalGeneration"]', hl: all, d: 'Nome da classe que o registry do vLLM procura: um wrapper multimodal marcado como <code>IsHybrid</code> e <code>HasInnerState</code>. É o sinal de que o motor precisa de um gerenciador de KV híbrido.' },
        { k: 'model_type', v: '"glm5_next" · text_config: "glm5_next_text"', hl: all, d: 'Escolhe a classe de config <code>Glm5NextTextConfig</code>, que traduz os nomes do SGLang usados no checkpoint. Sem o alias de <code>num_experts_per_tok</code>, o default da classe rotearia 7 experts em vez de 8.' },
        { c: 'forma' },
        { k: 'hidden_size', v: '4096', hl: all, d: 'Largura de cada um dos 4 fluxos do resíduo. Dita as formas: q/k/v do KDA [8192, 4096], experts [2048, 4096]. Os kernels cooperativos de MoE e o <code>pack_profile.py</code> exigem exatamente 4096.' },
        { k: 'num_hidden_layers', v: '45', hl: main, d: 'Camadas do decoder (0–44). A MTP é a camada 45, fora dessa conta; o vLLM usa esse número como índice da primeira camada MTP, e o DFlash2 numera suas camadas a partir de 45.' },
        { k: 'vocab_size / tie_word_embeddings', v: '154880 / false', hl: none, d: '154.856 tokens em uso + padding até múltiplo de 128 (o id de máscara do DFlash2, 154856, cai no padding). O lm_head tem pesos próprios (+634 M) e é compartilhado com a MTP e o DFlash2. Em TP=3 o vocabulário vira 154.944.' },
        { k: 'max_position_embeddings', v: '1048576', hl: all, d: 'Contexto declarado. Como o texto não usa RoPE, não há tabela de senos a escalar (o config nem tem <code>rope_theta</code>). A receita serve 850.000 por padrão.' },
        { c: 'mixers' },
        { k: 'layer_types', v: '34 × "linear_attention", 11 × "deepseek_sparse_attention"', hl: (l) => l.mix === 'dsa' && !l.extra, d: 'O padrão híbrido 3:1 do Kimi Linear. O vLLM colapsa as DSA em "attention" para a contabilidade híbrida: 11 camadas no grupo de atenção e 34 em grupos de estado tipo Mamba.' },
        { k: 'linear_attn_config', v: '{num_heads: 64, head_dim: 128, short_conv_kernel_size: 4, gate_lower_bound: -5.0}', hl: (l) => l.mix === 'kda', d: 'KDA: 64 cabeças de 128, com estado 128×128 por cabeça (32 cabeças por rank em TP=2) e conv causal de 4 taps em q, k e v. O gate de esquecimento fica em (−5, 0), então o fator α fica em (0,0067; 1): nunca apaga tudo de uma vez.' },
        { k: 'num_attention_heads / num_key_value_heads', v: '64 / 64', hl: (l) => l.mix === 'dsa', d: 'Cabeças de query da MLA: 32 por rank em TP=2. 64 não divide por 3, então TP=3 força 66 via --hf-overrides; 96 colapsaria os logits, porque um rank ficaria só com cabeças KDA falsas. O "64 KV heads" não dimensiona o cache, que é um latente.' },
        { k: 'q_lora_rank / kv_lora_rank', v: '1536 / 512', hl: (l) => l.mix === 'dsa', d: 'Gargalos de baixo posto da MLA. O latente de 512 é o que vai para o KV cache (o kernel SM120 exige exatamente 512); o latente de query de 1536 também alimenta o indexador.' },
        { k: 'qk_nope_head_dim / qk_rope_head_dim / v_head_dim', v: '256 / 0 / 256', hl: (l) => l.mix === 'dsa', d: 'Zero dimensões de RoPE: MLA NoPE. Isso quebra o kernel <code>fp8_ds_mla</code> ("pe_dim must be 64"); a receita padda 64 zeros, o que é exato e custa +24% de KV nessas camadas. Escala do softmax: 256<sup>−1/2</sup> = 1/16.' },
        { k: 'mla_use_nope', v: 'true', hl: (l) => l.mix === 'dsa', d: 'Liga o modo NoPE (<code>rotary_emb = None</code>). Mais um campo que só é lido corretamente por causa de um alias na classe de config.' },
        { k: 'index_topk / index_n_heads / index_head_dim', v: '2048 / 32 / 128', hl: (l) => l.mix === 'dsa', d: 'Indexador DSA: 32 cabeças de 128 pontuam o passado e só 2048 tokens entram na atenção; até 2048 tokens de contexto, esparsa = densa. O kernel do SM12x só existe para top-k ∈ {128, 512, 1024, 2048}.' },
        { k: 'index_kpool / index_kpool_always_select_tail', v: '4 / true', hl: (l) => l.mix === 'dsa', d: 'Cada 4 chaves do indexador viram 1 entrada (33 B/token/camada). O top-k escolhe 512 pools (= 2048 tokens) mais a cauda de até 3 tokens recentes. Como 2048 + 3 estourava o template do kernel, a receita usa 511 pools + cauda.' },
        { c: 'FFN' },
        { k: 'first_k_dense_replace', v: '3', hl: (l) => l.ffn === 'dense', d: 'As camadas 0–2 têm MLP densa de 12.288 (151 M parâmetros cada) antes de o MoE começar.' },
        { k: 'n_routed_experts / num_experts_per_tok / n_shared_experts', v: '288 / 8 / 1', hl: (l) => l.ffn === 'moe', d: '288 experts de 25,2 M parâmetros por camada (7,25 B), top-8 + 1 compartilhado. Em TP=3, expert parallel dá 96 experts por rank.' },
        { k: 'moe_intermediate_size', v: '2048', hl: (l) => l.ffn === 'moe', d: 'Largura de cada expert: 1024 por rank em TP=2. Não divide por 3 e o trellis EXL3 é empacotado com largura 2048, por isso TP=3 usa expert parallel em vez de padding.' },
        { k: 'scoring_func / topk_method / routed_scaling_factor', v: '"sigmoid" / "noaux_tc" / 2.5', hl: (l) => l.ffn === 'moe', d: 'Roteamento à moda DeepSeek-V3: scores sigmoides independentes, viés de balanceamento só na escolha (<code>e_score_correction_bias</code>, FP32), pesos renormalizados e multiplicados por 2,5. Roteador em FP32 para evitar trocas de expert por empate.' },
        { k: 'swiglu_limit', v: '10.0', hl: (l) => !l.extra || true, d: 'Clamp do SwiGLU (gate ≤ 10, up ∈ [−10, 10]) em MLPs densas, compartilhados e experts. O overlay EXL3 precisa replicar exatamente o mesmo valor.' },
        { c: 'resíduo' },
        { k: 'mhc / hc_mult / hc_sinkhorn_iters', v: 'true / 4 / 20', hl: main, d: 'Resíduo com 4 fluxos de 4096 ([T, 4, 4096]) e mistura 4×4 projetada por Sinkhorn (20 iterações) numa matriz duplamente estocástica. Qualquer "tap" de estados ocultos precisa materializar e contrair os 4 fluxos.' },
        { c: 'extras' },
        { k: 'num_nextn_predict_layers', v: '1', hl: (l) => l.extra === 'mtp', d: '1 camada MTP (índice 45: MLA + MoE, sem mHC). Rascunhador de reserva com SPEC_METHOD=mtp (k = 2), também editada pelo ABLIT.' },
        { k: 'dflash_config.target_layer_ids  (config do DFlash2)', v: '[5, 14, 24, 33, 42]', hl: (l) => !!l.tap, d: 'Do config do rascunhador: as camadas do alvo cujas saídas o alimentam (5 × 4096 → <code>fc [4096 × 20480]</code>). São todas KDA e aparecem no log como 6, 15, 25, 34, 43.' },
      ],
      qwen: [
        { c: 'identidade' },
        { k: 'architectures', v: '["Qwen4ExpForConditionalGeneration"]', hl: all, d: 'Modelo multimodal (visão + LM), prévia aberta da arquitetura do Qwen4.' },
        { k: 'model_type', v: '"qwen4_exp" · text_config: "qwen4_exp_text"', hl: all, d: 'O registry mapeia para <code>vllm/models/qwen4_exp/nvidia/…</code> no vLLM 0.30 e para <code>vllm/models/qwen3_8_flash_next/nvidia/…</code> na imagem day-0. Patches por bind-mount precisam acertar o caminho.' },
        { c: 'forma' },
        { k: 'hidden_size', v: '2560', hl: all, d: 'Estreito para 125 B: a capacidade está nos 512 experts e no N-gram. É o K de quase todo GEMM de decode. Com 4 ramos, o estado residual tem 10.240 números por token.' },
        { k: 'num_hidden_layers', v: '48', hl: main, d: '36 GDN + 12 QSA. O vLLM cria a MTP em <code>num_hidden_layers + i</code> = 48, daí o alias <code>mtp.layers.48</code> nos arquivos de quantização.' },
        { k: 'vocab_size / tie_word_embeddings', v: '248320 / false', hl: none, d: 'Embedding e lm_head separados, 1,18 GiB cada em BF16. O lm_head é o tensor mais lido no decode com MTP (1 + 3 leituras por passo), o que motivou o vocabulário de rascunho reduzido.' },
        { k: 'max_position_embeddings', v: '262144', hl: all, d: 'Contexto nativo. Para 1M: YaRN fator 4 via --hf-overrides aninhado em text_config + VLLM_ALLOW_LONG_MAX_MODEL_LEN=1, mas o 1M com YaRN ainda não foi validado no kit (<code>YARN_ENABLE</code> fica <code>false</code>). A receita desliga o YaRN abaixo de 262.144 ("sem benefício e custa precisão").' },
        { c: 'mixers' },
        { k: 'full_attention_interval', v: '4', hl: (l) => l.mix === 'qsa', d: '1 camada de atenção a cada 4 (aqui QSA esparsa); as outras 3 são Gated DeltaNet. Só essas camadas guardam KV por token: 12 + a MTP = 13.' },
        { k: 'layer_types', v: '36 × "linear_attention", 12 × "full_attention"', hl: (l) => l.mix === 'qsa' && !l.extra, d: 'A mesma informação por extenso: "full_attention" onde (i+1) mod 4 = 0. O transformers renomeia para "indexed_attention" porque são QSA.' },
        { k: 'linear_num_key_heads / linear_num_value_heads', v: '16 / 48', hl: (l) => l.mix === 'gdn', d: 'GDN com 16 cabeças de Q/K servindo 48 de V (cada Q/K é repetida 3×). Cada cabeça de V tem seu estado 128×128: 24 por GPU em TP=2.' },
        { k: 'linear_key_head_dim / linear_value_head_dim / linear_conv_kernel_dim', v: '128 / 128 / 4', hl: (l) => l.mix === 'gdn', d: 'Estado 128×128 por cabeça de V; conv causal de 4 taps (3 colunas de estado + tokens especulativos). Os kernels CUDA de decode GDN exigem head_k_dim = 128.' },
        { k: 'output_gate_type', v: '"sigmoid"', hl: (l) => l.mix === 'gdn', d: 'A saída do GDN é multiplicada por σ(W<sub>z</sub>x) antes do out_proj: um portão limitado a (0, 1), que o relatório técnico preferiu ao SiLU do GDN original. Custa um in_proj_z [6144, 2560] por camada.' },
        { k: 'mamba_ssm_dtype', v: '"float32"', hl: (l) => l.mix === 'gdn', d: 'dtype do estado recorrente. A receita usa --mamba-ssm-cache-dtype bfloat16: metade dos bytes lidos e escritos por passo e bloco de atenção de 1664 tokens em vez de 3200.' },
        { k: 'num_attention_heads / num_key_value_heads / head_dim', v: '24 / 2 / 256', hl: (l) => l.mix === 'qsa', d: 'QSA é GQA 12:1: o KV de 2 cabeças × 256 dá 1024 elementos por token por camada; em TP=2, 1 cabeça de KV por GPU.' },
        { k: 'partial_rotary_factor / rope_theta / mrope_section', v: '0.25 / 10000000 / [11, 11, 10]', hl: (l) => l.mix === 'qsa', d: 'RoPE só em 64 das 256 dimensões, com θ = 10⁷, repartida entre tempo, altura e largura (mRoPE intercalado). A receita aninha em <code>text_config</code> só o trecho do YaRN; o card oficial passa o dicionário completo, e o inventário aponta o risco, não verificado, de um override substituir o dict inteiro. Por precaução, repita θ, a fração e o mRoPE.' },
        { k: 'indexer_n_heads / indexer_compress_ratio / indexer_budget', v: '4 / 4 / 2048', hl: (l) => l.mix === 'qsa', d: 'Indexador QSA: 4 cabeças e chaves em micro-blocos de 4 (média). Seleciona 512 blocos = 2048 tokens, mais uma cauda de até 3, o que dá largura 2051.' },
        { c: 'FFN' },
        { k: 'num_experts / num_experts_per_tok', v: '512 / 10', hl: (l) => l.ffn === 'moe', d: '512 experts por camada em todas as 48 (24.576 no tronco), top-10 via softmax renormalizado: 10/512 ≈ 2% ativos.' },
        { k: 'moe_intermediate_size / shared_expert_intermediate_size', v: '640 / 640', hl: (l) => l.ffn === 'moe', d: 'Experts minúsculos (4,9 M parâmetros). Em TP puro, 640/2 = 320 não é múltiplo de 128 (os blocos FP8 da MTP), então EP é obrigatório: 256 experts inteiros por GPU. O compartilhado tem um gate sigmoide escalar.' },
        { c: 'resíduo' },
        { k: 'hc_count / hc_lowrank', v: '4 / 320', hl: main, d: 'Gated Residual: 4 ramos de 2560. A leitura usa um gate elementwise de posto 320, que substitui a pré-norma; a escrita usa 1 escalar por ramo. Um "mixer" final substitui o RMSNorm final. É replicado nas duas GPUs.' },
        { c: 'extras' },
        { k: 'ple_layer_ids / ngram_size / heads_per_ngram', v: '[2] / 3 / 8', hl: (l) => !!l.ple, d: 'N-gram embedding na 2ª camada (contagem 1-based → layers.1): bigramas e trigramas, 8 hashes por ordem = 16 cabeças de 160, formando um vetor de 2560 injetado nos 4 ramos.' },
        { k: 'ngram_vocab_size_base / ple_embed_dim', v: '20000000 / 2560', hl: (l) => !!l.ple, d: '16 tabelas de tamanho primo ≥ 20 M: 320.001.536 linhas × 160 = 51,2 B parâmetros (47,7 GiB em FP8). A receita precisa declarar ple_embedding_dtype via --hf-overrides para a tabela FP8 carregar.' },
        { k: 'mtp_num_hidden_layers / mtp.layer_types', v: '1 / ["full_attention"]', hl: (l) => l.extra === 'mtp', d: '1 camada MTP com QSA (a 13ª camada com KV) e 512 experts FP8 em blocos 128×128, aplicada 3 vezes (MTP = 3).' },
      ],
      ds: [
        { c: 'identidade' },
        { k: 'architectures / model_type', v: '["DeepseekV4ForCausalLM"] / "deepseek_v4"', hl: all, d: 'Mesma classe para o 0731 e o Vision-Exp (a visão é instalada por hotfix). Com os campos dspark_*, o vLLM monta o rascunhador DSpark em vez do MTP clássico.' },
        { c: 'forma' },
        { k: 'hidden_size / num_hidden_layers', v: '4096 / 43', hl: main, d: '43 camadas de tronco (0–42); as 43–45 são os estágios DSpark. Em TP=2 cada passo faz 43 all-reduce do wo_b + 43 do MoE.' },
        { k: 'num_attention_heads / head_dim', v: '64 / 512', hl: main, d: '64 cabeças de query de 512. 64 não divide por 3: TP=3 padda para 72 (9 grupos). O kernel SM120 exige cabeças locais ∈ {16, 32, 64, 128}, e as 32 de TP=2 cabem.' },
        { k: 'num_key_value_heads', v: '1', hl: main, d: 'MQA: um único vetor de 512 por token é K e V ao mesmo tempo. Não dá para fatiar por TP, então o KV é replicado em todos os ranks.' },
        { k: 'qk_rope_head_dim', v: '64', hl: main, d: 'RoPE só nas últimas 64 dims: 448 "nope" em FP8 + 64 em BF16, mais escalas = 584 B. Como V também carrega a rotação, a saída passa por RoPE inverso (posição −i).' },
        { k: 'q_lora_rank / o_groups / o_lora_rank', v: '1024 / 8 / 1024', hl: main, d: 'Query de baixo posto (latente de 1024, reusado pelo indexador) e projeção de saída agrupada: 8 grupos de 8 cabeças × 512 = 4096 → 1024 cada. É esse contrato 8 × 512 que força padding de grupos em TP=3.' },
        { k: 'vocab_size', v: '129280', hl: none, d: 'O mesmo vocabulário do V3. Os logits (129.280 × tokens) passam por all-gather a cada passo.' },
        { c: 'atenção híbrida' },
        { k: 'sliding_window', v: '128', hl: all, d: 'Toda camada (e o rascunhador) tem uma janela de 128 tokens brutos, que cobre o bloco comprimido ainda "aberto". Custo fixo de ~3,4 MB por sequência.' },
        { k: 'compress_ratios', v: '[0, 0, 4, 128, 4, 128, …, 128, 4, 0, 0, 0]', hl: (l) => l.mix !== 'swa', d: 'Por camada: 0 = só janela, 4 = CSA/C4A, 128 = HCA/C128A. 46 entradas (43 do tronco + 3 do DSpark): 2 SWA + 21 C4A + 20 C128A.' },
        { k: 'index_topk / index_n_heads / index_head_dim', v: '512 / 64 / 128', hl: (l) => l.mix === 'c4a', d: 'Lightning indexer só nas C4A: escolhe 512 entradas comprimidas (≈ 2048 tokens), com FP4 simulado depois de Hadamard. No V3.2 ele escolhia 2048 tokens brutos.' },
        { k: 'rope_theta / compress_rope_theta', v: '10000 / 160000', hl: all, d: 'θ = 10⁴ nas janelas e no rascunhador; θ = 1,6·10⁵ nas camadas comprimidas.' },
        { k: 'rope_scaling', v: '{type: "yarn", factor: 16, original_max_position_embeddings: 65536}', hl: (l) => l.mix !== 'swa', d: '64K × 16 = 1.048.576. Na referência, o YaRN vale só nas camadas comprimidas; o vLLM da imagem aplica em todas (o fix DSPARK_ENABLE_ROPE_SWA_FIX é opt-in).' },
        { c: 'FFN' },
        { k: 'n_routed_experts / num_experts_per_tok / n_shared_experts', v: '256 / 6 / 1', hl: all, d: '256 experts SwiGLU 4096 → 2048 → 4096 por camada, top-6 + 1 compartilhado: 7 ativos por token.' },
        { k: 'scoring_func / topk_method / routed_scaling_factor', v: '"sqrtsoftplus" / "noaux_tc" / 1.5', hl: (l) => l.ffn === 'moe', d: 'Score √softplus(logit) em FP32 (novidade do V4), viés sem perda auxiliar só na escolha, sem roteamento por grupos, pesos ×1,5 (no V3 era 2,5).' },
        { k: 'num_hash_layers', v: '3', hl: (l) => l.ffn === 'hash', d: 'As camadas 0–2 escolhem os 6 experts por uma tabela fixa tid2eid[token_id], sem GEMM de roteador para a escolha. Consequência: o forward precisa dos input_ids crus, mesmo no caminho multimodal.' },
        { k: 'expert_dtype / quantization_config', v: '"fp4" / fp8 e4m3, blocos [128, 128], scale_fmt ue8m0', hl: all, d: 'Experts em FP4 E2M1 com escala E8M0 a cada 32 (≈ MXFP4); o resto em FP8 com escalas potência de 2. No GB10, os experts rodam como W4A16 via b12x.' },
        { c: 'resíduo' },
        { k: 'hc_mult / hc_sinkhorn_iters', v: '4 / 20', hl: main, d: 'mHC: 4 fluxos de 4096 (16.384 floats por token) com mistura Sinkhorn. Os kernels b12x de mHC só suportam hidden 4096.' },
        { k: 'rms_norm_eps', v: '1e-20 (Vision-Exp) · 1e-06 (0731)', hl: all, d: 'Mudança real do checkpoint Vision-Exp, que afeta todas as RMSNorm e o rsqrt do mHC. O repositório não mediu o efeito.' },
        { c: 'extras' },
        { k: 'num_nextn_predict_layers', v: '3 (Vision-Exp) · 1 (0731)', hl: (l) => l.extra === 'dspark', d: 'Na imagem Anemll vira n_predict. Com 3, o número de tokens especulativos precisa ser múltiplo de 3, o que leva a k = 6 (o bloco treinado é de 5).' },
        { k: 'dspark_block_size / dspark_target_layer_ids / dspark_markov_rank', v: '5 / [40, 41, 42] / 256', hl: (l) => !!l.tap || l.extra === 'dspark', d: 'O DSpark lê a média dos 4 fluxos das camadas 40–42 (3 × 4096 → 4096), monta um bloco [âncora + 4 ruídos] e o resolve num único forward. Um Markov head de posto 256 costura a sequência escolhida.' },
        { k: 'vision_n_layers / vision_max_n_token', v: '32 / 384', hl: none, d: 'ViT de 32 camadas + Aligner 3×3, até 384 tokens por imagem. Tokens de imagem usam o viés bias_vl e pulam o hash.' },
      ],
    };
    (function () {
      const fig = root.querySelector('#fig-config');
      const box = fig.querySelector('#cfg-json'), exp = fig.querySelector('#cfg-exp');
      const S = { m: 'glm', sel: null };
      const firstField = (m) => CFG[m].find((f) => f.k && /layer_types|compress_ratios/.test(f.k)) || CFG[m].find((f) => f.k);
      function renderExp() {
        const f = S.sel; const L = LAYERS[S.m];
        const n = L.filter((l) => f.hl(l)).length;
        exp.innerHTML = '';
        exp.appendChild(h('div', null,
          h('div', { class: 'kk' }, f.k), h('span', { class: 'vv' }, f.v),
          h('div', { html: f.d }),
          h('div', { class: 'aff' }, n ? `Afeta ${n} de ${L.length} camadas no mapa (incluindo extras).` : 'Campo global: não é por camada (embedding, lm_head ou torre de visão).')));
      }
      function render() {
        box.innerHTML = '';
        box.appendChild(h('span', { class: 'cm' }, '// ' + NAME[S.m] + ' · config.json (trechos)'));
        CFG[S.m].forEach((f) => {
          if (f.c) { box.appendChild(h('span', { class: 'cm' }, '// — ' + f.c + ' —')); return; }
          const b = h('button', { type: 'button', class: S.sel === f ? 'on' : '' }, h('span', { class: 'k' }, '"' + f.k + '"'), ': ', h('span', { class: 'v' }, f.v), ',');
          b.addEventListener('click', () => { S.sel = f; box.querySelectorAll('button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); renderExp(); });
          box.appendChild(b);
        });
        renderExp();
      }
      S.sel = firstField('glm');
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; S.sel = firstField(v); render(); } });
      UI.button(ctl, 'próximo campo ▶', () => { const fs = CFG[S.m].filter((f) => f.k); const i = fs.indexOf(S.sel); S.sel = fs[(i + 1) % fs.length]; render(); const on = box.querySelector('button.on'); if (on && on.scrollIntoView) { try { box.scrollTop = on.offsetTop - box.clientHeight / 2; } catch (e) { /* */ } } });
      render();
      const O = { mh: 20, fh: 8, labelH: 14, markH: 10, minCell: 12, maxCell: 26 };
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C();
        const L = LAYERS[S.m];
        V.text(g, 'camadas afetadas por "' + (S.sel ? S.sel.k : '') + '"', 14, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
        const G = stripGeom(w, L, Object.assign({ y: 34 }, O));
        const f = S.sel;
        const pulse = 0.6 + 0.4 * Math.sin(t * 3);
        drawStrip(g, G, Object.assign({ mode: 'tipo', labels: true, lit: (l) => f && f.hl(l) }, O));
        G.pos.forEach((p) => { if (f && f.hl(p.l)) V.glow(g, p.x + G.cw / 2, p.y + O.mh / 2, 10, mixCol(C, p.l.mix), 0.35 * pulse); });
        fitTo(fig, 34 + G.height + 6);
      });
    })();

    /* ================================================================ FIG 4: parâmetros */
    (function () {
      const fig = root.querySelector('#fig-params');
      const CATS = [
        { k: 'rout', l: 'experts roteados', c: 'a' },
        { k: 'alw', l: 'FFN sempre ativa (compartilhado + densa)', c: 'd' },
        { k: 'mix', l: 'mixers (atenção, linear, indexador)', c: 'b' },
        { k: 'res', l: 'resíduo largo + roteadores', c: 'gray' },
        { k: 'emb', l: 'embedding + lm_head (nos ativos, só o lm_head)', c: 'f' },
        { k: 'ext', l: 'extras (N-gram, MTP, rascunhador, visão)', c: 'c' },
      ];
      const D = {
        // GLM: experts roteados incluem os 288 da camada MTP (43 × 288 × 25,17 M = 311,65 B); "extras" = resto da MTP (0,185 B) + visão (0,564 B)
        // ativos: convenção única para os três = lm_head sim, embedding não (consulta de uma linha)
        glm: { totS: '321,3 B', tot: { rout: 311.654, alw: 1.51, mix: 6.057, res: 0.085, emb: 1.268, ext: 0.749 }, act: { rout: 8.456, alw: 1.51, mix: 6.057, res: 0.085, emb: 0.634, ext: 0 }, totL: '321,3 B', actL: '~16,7 B (cartão: 18 B)', frac: '97% (inclui os 288 da MTP)' },
        qwen: { totS: '180,0 B (125,7 no tronco)', tot: { rout: 120.8, alw: 0.236, mix: 2.705, res: 0.704, emb: 1.272, ext: 54.292 }, act: { rout: 2.36, alw: 0.236, mix: 2.705, res: 0.704, emb: 0.636, ext: 0.033 }, totL: '180,0 B armazenados (125,7 no tronco)', actL: '~6,7 B (cartão: 6 B, sem lm_head)', frac: '96% do tronco' },
        ds: { totS: '284,3 B (+20,3 fora)', tot: { rout: 277.03, alw: 1.082, mix: 5.088, res: 0.079, emb: 1.06, ext: 20.27 }, act: { rout: 6.494, alw: 1.082, mix: 5.088, res: 0.079, emb: 0.53, ext: 0 }, totL: '284,3 B (+20,3 B de DSpark e visão)', actL: '~13,2 B (cartão: 13 B)', frac: '97%' },
      };
      const FL = [['experts roteados', 16.9, 'a'], ['KDA (34 camadas)', 11.6, 'b'], ['MLA + indexador', 3.3, 'c'], ['compartilhados', 2.1, 'd'], ['MLP densa', 0.9, 'gray'], ['lm_head', 1.27, 'f']];
      const S = { mode: 'tot', a: 0, ta: 0, fl: 0, tfl: 0 };
      const cc = (C, k) => k === 'gray' ? V.alpha(C.ink, 0.5) : C[k];
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mostrar', options: [{ v: 'tot', label: 'totais' }, { v: 'act', label: 'ativos por token' }, { v: 'flops', label: 'FLOPs (GLM)' }], value: 'tot', onChange: (v) => { S.mode = v; S.ta = v === 'act' ? 1 : v === 'tot' ? 0 : S.ta; S.tfl = v === 'flops' ? 1 : 0; } });
      V.fig(fig, (g, w, hh, t, dt) => {
        const C = V.C();
        S.a += (S.ta - S.a) * Math.min(1, dt * 3);
        S.fl += (S.tfl - S.fl) * Math.min(1, dt * 4);
        const pad = 14, narrow = w < 560;
        if (S.fl < 0.99) {
          g.globalAlpha = 1 - S.fl;
          const lw = narrow ? 64 : 120, x0 = pad + lw, x1 = w - pad - (narrow ? 8 : 20);
          const maxT = 340, maxA = 19;
          const e = V.ease.inOut(S.a);
          const scale = V.lerp(maxT, maxA, e);
          V.text(g, S.a < 0.5 ? 'PARÂMETROS TOTAIS (bilhões)' : (narrow ? 'ATIVOS POR TOKEN (bi) · com lm_head' : 'PARÂMETROS ATIVOS POR TOKEN DE TEXTO (bilhões) · com lm_head, sem embedding'), pad, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
          // eixo
          const ticks = scale > 100 ? [0, 50, 100, 150, 200, 250, 300] : scale > 40 ? [0, 20, 40, 60, 80] : [0, 5, 10, 15];
          ticks.forEach((v) => { const x = x0 + (v / scale) * (x1 - x0); if (x > x1 + 1) return; V.line(g, x, 34, x, 34 + 3 * 74, { color: C.grid, lw: 1 }); V.text(g, String(v), x, 44 + 3 * 74, { size: 10, font: 'mono', align: 'center', color: C.muted }); });
          ['glm', 'qwen', 'ds'].forEach((m, r) => {
            const y = 44 + r * 74;
            V.text(g, SHORT[m], pad, y + 18, { size: narrow ? 12 : 14, weight: 600, color: C[m] });
            let x = x0;
            let tot = 0;
            CATS.forEach((cat) => {
              const v = V.lerp(D[m].tot[cat.k], D[m].act[cat.k], e);
              tot += v;
              const bw = (v / scale) * (x1 - x0);
              const vis = Math.max(0, Math.min(bw, x1 - x + 40));
              if (vis > 0.3) {
                g.fillStyle = V.alpha(cc(C, cat.c), 0.82); g.fillRect(x, y + 2, vis, 26);
                if (m === 'ds' && cat.k === 'ext') hatch(g, x, y + 2, vis, 26, V.alpha(C.bg, 0.55));
              }
              x += bw;
            });
            const lab = S.a < 0.5 ? (narrow ? D[m].totS : D[m].totL) : D[m].actL;
            const lwid = textW(g, lab, 11.5, 'mono');
            const inside = x + 6 + lwid <= x1;
            if (inside) V.text(g, lab, x + 6, y + 20, { size: 11.5, font: 'mono', color: C.ink });
            else V.text(g, lab, x1, y + 44, { size: 11.5, font: 'mono', align: 'right', color: C.ink, maxW: x1 - x0 });
            if (S.a < 0.5) V.text(g, 'experts: ' + D[m].frac, x0, inside ? y + 44 : y + 60, { size: 10.5, color: C.muted });
          });
          // legenda
          let lx = pad, ly = 44 + 3 * 74 + 26;
          CATS.forEach((cat) => { const s = cat.l; const ww = textW(g, s, 11) + 24; if (lx + ww > w - pad) { lx = pad; ly += 17; } V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: cc(C, cat.c) }); V.text(g, s, lx + 16, ly + 1, { size: 11, color: C.muted }); lx += ww; });
          g.globalAlpha = 1;
          S.needBars = ly + 16;
        }
        if (S.fl > 0.01) {
          g.globalAlpha = S.fl;
          const cx = narrow ? w / 2 : w * 0.3, cy = narrow ? 120 : hh / 2 + 6, R = narrow ? 82 : Math.min(120, hh / 2 - 40);
          const total = FL.reduce((s, x) => s + x[1], 0);
          let a0 = -Math.PI / 2;
          const sweep = V.ease.out(Math.min(1, S.fl));
          FL.forEach(([l, v, c], i) => {
            const a1 = a0 + (v / total) * Math.PI * 2 * sweep;
            g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, a0, a1); g.closePath(); g.fillStyle = V.alpha(cc(C, c), 0.85); g.fill();
            a0 = a1;
          });
          g.beginPath(); g.arc(cx, cy, R * 0.56, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill();
          V.text(g, '~36', cx, cy + 2, { size: 28, font: 'display', weight: 700, align: 'center', color: C.ink });
          V.text(g, 'GFLOP/token', cx, cy + 20, { size: 11, align: 'center', color: C.muted });
          const tx = narrow ? pad : w * 0.56, ty0 = narrow ? cy + R + 26 : cy - 70;
          FL.forEach(([l, v, c], i) => {
            const yy = ty0 + i * (narrow ? 20 : 24);
            V.box(g, tx, yy - 10, 12, 12, { r: 3, fill: cc(C, c) });
            V.text(g, l, tx + 20, yy, { size: 12.5, color: C.ink });
            V.text(g, nf(v, v < 2 ? 2 : 1) + ' · ' + nf((v / total) * 100, 0) + '%', w - pad, yy, { size: 12, font: 'mono', align: 'right', color: C.muted });
          });
          const fy = narrow ? ty0 + FL.length * 20 + 6 : ty0 + FL.length * 24 + 10;
          V.text(g, 'GLM-5.3-Flash · estimativa do repo (docs/improve-prefill.md)', narrow ? pad : tx, fy, { size: 10.5, color: C.muted, maxW: w - 2 * pad });
          g.globalAlpha = 1;
          S.needFl = fy + 16;
        }
        fitTo(fig, Math.max(S.needBars || 0, narrow ? (S.needFl || 0) : 0, narrow ? 0 : 340));
      });
    })();

    /* ================================================================ FIG 5: experts acordados */
    (function () {
      const fig = root.querySelector('#fig-experts');
      const P = {
        glm: { E: 288, k: 8, L: 42, bytes: 12582912, t: 8, fmt: 'EXL3 4 bpw', cols: 24, spec: 'DFlash2 k = 7 → 8 tokens' },
        qwen: { E: 512, k: 10, L: 48, bytes: 2770000, t: 4, fmt: 'NVFP4 + escalas', cols: 32, spec: 'MTP = 3 → 4 tokens' },
        ds: { E: 256, k: 6, L: 43, bytes: 13369344, t: 7, fmt: 'MXFP4 (4,25 bits)', cols: 16, spec: 'DSpark k = 6 → 7 tokens' },
      };
      const S = { m: 'glm', t: 8, pick: [], t0: 0, seed: 1 };
      const rng = V.rng(99);
      const sample = () => { const p = P[S.m]; S.pick = []; for (let j = 0; j < S.t; j++) { const st = new Set(); while (st.size < p.k) st.add(rng.int(p.E)); S.pick.push(Array.from(st)); } S.t0 = performance.now(); };
      sample();
      const ctl = UI.controls(fig);
      let sl;
      UI.seg(ctl, { label: 'Modelo', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; sl.set(P[v].t); } });
      sl = UI.slider(ctl, { label: 'Tokens por passo', min: 1, max: 8, step: 1, value: 8, fmt: (v) => String(v), onInput: (v) => { S.t = v; sample(); } });
      UI.button(ctl, '↻ novo passo', sample, { primary: true });
      V.every(2600, sample);
      const TOKC = (C) => [C.a, C.b, C.c, C.d, C.e, C.f, C.ink, C.muted];
      V.fig(fig, (g, w, hh) => {
        const C = V.C(), p = P[S.m];
        const pad = 14, narrow = w < 620;
        const el = Math.min(1, (performance.now() - S.t0) / 700);
        const cols = p.cols, rows = Math.ceil(p.E / cols);
        const statW = narrow ? 0 : 250;
        const gx0 = pad + 40, gw = w - gx0 - pad - statW - (narrow ? 0 : 16);
        const gh = narrow ? 190 : hh - 60;
        const cs = Math.min(gw / cols, gh / rows);
        const gy0 = 34;
        V.text(g, `${NAME[S.m]} · ${p.E} experts por camada · top-${p.k}`, pad, 20, { size: 12, weight: 600, color: C[S.m], maxW: w - 2 * pad });
        // contagem
        const owner = new Map();
        S.pick.forEach((arr, j) => arr.forEach((e) => { if (!owner.has(e)) owner.set(e, []); owner.get(e).push(j); }));
        const tc = TOKC(C);
        // tokens
        S.pick.forEach((arr, j) => {
          const ty = gy0 + 8 + j * Math.min(26, (rows * cs - 10) / Math.max(1, S.t - 1 || 1));
          V.dot(g, pad + 12, ty, 7, tc[j % tc.length]);
          V.text(g, String(j + 1), pad + 12, ty + 4, { size: 9.5, font: 'mono', align: 'center', color: C.bg });
          if (el < 1 || S.t <= 4) {
            g.globalAlpha = (1 - el) * 0.5 + (S.t <= 4 ? 0.12 : 0);
            arr.forEach((e) => { const r = Math.floor(e / cols), c = e % cols; V.line(g, pad + 19, ty, gx0 + c * cs + cs / 2, gy0 + r * cs + cs / 2, { color: tc[j % tc.length], lw: 0.7 }); });
            g.globalAlpha = 1;
          }
        });
        for (let e = 0; e < p.E; e++) {
          const r = Math.floor(e / cols), c = e % cols;
          const x = gx0 + c * cs, y = gy0 + r * cs;
          const o = owner.get(e);
          if (o) { g.fillStyle = V.alpha(tc[o[0] % tc.length], 0.35 + 0.6 * el); V.rr(g, x + 0.6, y + 0.6, cs - 1.2, cs - 1.2, Math.min(3, cs / 4)); g.fill(); if (o.length > 1) { V.dot(g, x + cs / 2, y + cs / 2, Math.max(1.5, cs / 6), C.ink); } }
          else { g.fillStyle = V.alpha(C.ink, 0.09); V.rr(g, x + 0.6, y + 0.6, cs - 1.2, cs - 1.2, Math.min(3, cs / 4)); g.fill(); }
        }
        // estatísticas
        const dist = owner.size;
        const expd = p.E * (1 - Math.pow(1 - p.k / p.E, S.t));
        const perLayer = dist * p.bytes, perStep = perLayer * p.L;
        const lines = [
          ['distintos neste passo', `${dist} de ${p.E}`],
          ['esperado (fórmula)', nf(expd, 1)],
          ['máximo possível (k·t)', String(Math.min(p.E, p.k * S.t))],
          ['bytes de um expert', `${nf(p.bytes / 1e6, 1)} MB · ${p.fmt}`],
          ['por camada', `${nf(perLayer / 1e6, 0)} MB`],
          [`× ${p.L} camadas MoE`, `${nf(perStep / 1e9, 1)} GB por passo`],
          ['≈ por rank (TP/EP = 2)', `${nf(perStep / 2e9, 1)} GB`],
        ];
        const sx = narrow ? pad : w - pad - statW, sy = narrow ? gy0 + rows * cs + 22 : gy0 + 6;
        if (narrow) {
          lines.forEach(([a, b], i) => { const col = i % 2, row = Math.floor(i / 2); const xx = pad + col * ((w - 2 * pad) / 2); const yy = sy + row * 34; V.text(g, a, xx, yy, { size: 10.5, color: C.muted, maxW: (w - 2 * pad) / 2 - 6 }); V.text(g, b, xx, yy + 15, { size: 12.5, font: 'mono', weight: 600, color: i === 0 ? C.d : i === 6 ? C.c : C.ink, maxW: (w - 2 * pad) / 2 - 6 }); });
          const specY = sy + Math.ceil(lines.length / 2) * 34 + 2;
          V.text(g, p.spec, pad, specY, { size: 10.5, color: C.muted, maxW: w - 2 * pad });
          // altura calculada com o layout real do modelo escolhido (a grade do DS é mais alta que a do Qwen)
          fitTo(fig, specY + 12);
        } else {
          lines.forEach(([a, b], i) => { const yy = sy + i * 38; V.text(g, a, sx, yy, { size: 11, color: C.muted }); V.text(g, b, sx, yy + 17, { size: 14, font: 'mono', weight: 600, color: i === 0 ? C.d : i === 6 ? C.c : C.ink, maxW: statW }); });
          V.text(g, p.spec, sx, sy + lines.length * 38 + 4, { size: 11, color: C.muted, maxW: statW });
          fitTo(fig, 360);
        }
      });
    })();

    /* ================================================================ FIG 6: linha de KV */
    (function () {
      const fig = root.querySelector('#fig-linha');
      const R = {
        glm: { title: 'fp8_ds_mla · 656 B por token, por camada MLA', segs: [['latente NoPE · 512 × FP8', 512, 'b'], ['4 escalas FP32', 16, 'd'], ['RoPE 64 × BF16 = zeros (pad)', 128, 'z']], idx: 'kpool = 4: soma ponderada por softmax por canal (+ APE de 4 slots) → Hadamard → FP8', idxB: '1 chave de 128 B FP8 + 4 B de escala a cada 4 tokens = 33 B/token', types: [['KDA (34)', 0, 'estado fixo ≈ 2,3 MB/req'], ['MLA-DSA (11)', 689, '656 + 33 do indexador']] },
        qwen: { title: 'KV QSA em FP8 · 512 B por token, por camada, por GPU', segs: [['K · 1 cabeça × 256 × FP8', 256, 'b'], ['V · 1 cabeça × 256 × FP8', 256, 'a']], idx: 'blocos de 4: média (AvgPool) → RMSNorm → RoPE na posição inicial do bloco', idxB: '1 chave de 128 × BF16 a cada 4 tokens ≈ 64 B/token (estimativa)', types: [['GDN (36)', 0, 'estado fixo 0,75 MiB/req (BF16)'], ['QSA (12 + MTP)', 512, 'K + V por GPU; +~64 do indexador']] },
        ds: { title: 'nvfp4_ds_mla (= FP8) · 584 B por entrada', segs: [['448 × FP8 "nope" (7 blocos de 64)', 448, 'b'], ['RoPE 64 × BF16', 128, 'c'], ['7 escalas UE8M0 + 1 pad', 8, 'd']], idx: 'C4 com sobreposição: softmax por canal sobre 8 posições (+ APE) → FP4 após Hadamard', idxB: 'cache K do indexador: 132 B por entrada ÷ 4 = 33 B/token', types: [['SWA (2 + 3)', 0, 'anel de 128 × 584 B'], ['C4A (21)', 179, '584/4 + 33 do indexador'], ['C128A (20)', 4.56, '584/128']] },
      };
      const S = { m: 'glm' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modelo', options: MODEL_OPTS, value: 'glm', onChange: (v) => { S.m = v; } });
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C(), r = R[S.m], pad = 14;
        V.text(g, r.title, pad, 20, { size: 12.5, weight: 600, color: C[S.m], maxW: w - 2 * pad });
        // régua de bytes
        const total = r.segs.reduce((s, x) => s + x[1], 0);
        const nCells = total / 8;
        const cellW = 9, perRow = Math.max(8, Math.floor((w - 2 * pad) / cellW));
        const cyc = V.phase(t, 5).f;
        const fill = Math.min(1, cyc / 0.55);
        let k = 0;
        const y0 = 34;
        const segStart = [];
        r.segs.forEach(([l, b, c], si) => {
          segStart.push(k);
          for (let j = 0; j < b / 8; j++, k++) {
            const row = Math.floor(k / perRow), col = k % perRow;
            const x = pad + col * cellW, y = y0 + row * 11;
            const on = k / nCells <= fill;
            const colr = c === 'z' ? V.alpha(C.muted, 0.35) : C[c];
            g.fillStyle = on ? V.alpha(colr, c === 'z' ? 1 : 0.85) : V.alpha(C.ink, 0.07);
            g.fillRect(x, y, cellW - 2, 9);
            if (c === 'z' && on) { g.fillStyle = V.alpha(C.ink, 0.5); g.fillRect(x + 3, y + 4, 2, 1); }
          }
        });
        const rulerRows = Math.ceil(nCells / perRow);
        let ly = y0 + rulerRows * 11 + 14;
        let lx = pad;
        r.segs.forEach(([l, b, c]) => { const s = `${l}: ${b} B`; const ww = textW(g, s, 11.5) + 26; if (lx + ww > w - pad) { lx = pad; ly += 18; } V.box(g, lx, ly - 9, 11, 11, { r: 3, fill: c === 'z' ? V.alpha(C.muted, 0.45) : C[c] }); V.text(g, s, lx + 16, ly + 1, { size: 11.5, color: C.ink }); lx += ww; });
        V.text(g, 'cada quadradinho = 8 bytes', w - pad, ly + 18, { size: 10, align: 'right', color: C.muted });
        // indexador: 4 → 1
        const iy = ly + 38;
        V.text(g, 'INDEXADOR: 4 CHAVES → 1', pad, iy, { size: 10.5, font: 'mono', color: C.muted });
        const ph = V.phase(t, 4).f;
        const merge = V.ease.inOut(V.clamp((ph - 0.25) / 0.45, 0, 1));
        const vw = Math.min(16 * 9, (w - 2 * pad) * 0.42), vc = 16, vcw = vw / vc;
        const bx = pad, by = iy + 12;
        for (let q = 0; q < 4; q++) {
          const yy = V.lerp(by + q * 13, by + 19, merge);
          g.globalAlpha = q === 0 ? 1 : 1 - merge * 0.85;
          for (let j = 0; j < vc; j++) { const v = Math.sin(j * 0.9 + q * 1.7 + 0.3); g.fillStyle = V.cmap(v * (1 - 0.3 * merge), 'div'); g.fillRect(bx + j * vcw, yy, vcw - 1, 10); }
          g.globalAlpha = 1;
          if (merge < 0.1) V.text(g, 't' + (q + 1), bx + vw + 6, yy + 9, { size: 10, font: 'mono', color: C.muted });
        }
        if (merge > 0.9) { V.text(g, '1 entrada', bx + vw + 6, by + 28, { size: 10.5, font: 'mono', color: C.d }); }
        // o texto começa depois do rótulo mais largo da animação ("1 entrada"), sem invadi-lo
        const tx = bx + vw + 6 + textW(g, '1 entrada', 10.5, 'mono') + 16;
        const lines1 = wrapText(g, r.idx, w - pad - tx, 11.5);
        const lines2 = wrapText(g, r.idxB, w - pad - tx, 11.5);
        if (w - pad - tx > 120) {
          lines1.forEach((s, i) => V.text(g, s, tx, by + 8 + i * 15, { size: 11.5, color: C.ink }));
          lines2.forEach((s, i) => V.text(g, s, tx, by + 8 + (lines1.length + i) * 15 + 4, { size: 11.5, color: C.muted }));
        }
        const nIdxLines = w - pad - tx > 120 ? lines1.length + lines2.length : 0;
        // custo amortizado por tipo
        let by2 = by + Math.max(78, nIdxLines * 15 + 28);
        if (w - pad - tx <= 120) { const L3 = wrapText(g, r.idx + ' · ' + r.idxB, w - 2 * pad, 11); L3.forEach((s, i) => V.text(g, s, pad, by2 + i * 14, { size: 11, color: C.muted })); by2 += L3.length * 14 + 10; }
        V.text(g, 'BYTES/TOKEN QUE CRESCEM COM O CONTEXTO, POR CAMADA, POR RANK', pad, by2, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const lw = Math.min(130, (w - 2 * pad) * 0.34), bx0 = pad + lw, bx1 = w - pad - 60;
        r.types.forEach(([l, b, note], i) => {
          const yy = by2 + 14 + i * 34;
          V.text(g, l, pad, yy + 13, { size: 12, weight: 600, color: C.ink, maxW: lw - 6 });
          const bw = b > 0 ? Math.max(3, (b / 700) * (bx1 - bx0)) : 0;
          if (b > 0) V.box(g, bx0, yy + 2, bw, 16, { r: 4, fill: V.alpha(C[S.m], 0.8) });
          else { V.box(g, bx0, yy + 2, 16, 16, { r: 8, stroke: C.a, lw: 1.5 }); }
          V.text(g, b > 0 ? nf(b, b < 10 ? 1 : 0) + ' B' : '0 B', bx0 + Math.max(bw, 16) + 6, yy + 14, { size: 12, font: 'mono', color: C.ink });
          V.text(g, note, bx0, yy + 31, { size: 10.5, color: C.muted, maxW: w - pad - bx0 });
        });
        fitTo(fig, by2 + 14 + r.types.length * 34 + 22);
      });
    })();

    /* ================================================================ FIG 7: calculadora de KV */
    (function () {
      const fig = root.querySelector('#fig-kvcalc');
      const MiB = 1048576, GiB = 1073741824;
      const S = { v: 100, qfp8: true, ghost: false, gA: 0 };
      const Ls = () => Math.round(1024 * Math.pow(1024, S.v / 100));
      const M = () => ({
        glm: { grow: 7579, fixed: 34 * 2342912 + 5 * 2048 * 2048, fixL: 'KDA 76 MiB + janela do DFlash2 20 MiB', ghost: 29520, ghostL: 'MLA fp8_ds_mla nas 45 camadas', pool: 11 * GiB, poolL: 'pool 11 GiB' },
        qwen: { grow: S.qfp8 ? 6656 : 13312, fixed: 36 * 786432 + 36 * 61440, fixL: 'GDN 27 MiB (BF16) + conv ≈ 2 MiB', ghost: 48 * (S.qfp8 ? 512 : 1024), ghostL: '48 camadas com KV (ilustrativo)', pool: 32.02 * GiB, poolL: 'pool 32,02 GiB' },
        ds: { grow: 3850, fixed: 3440000, fixL: 'janelas SWA ≈ 3,4 MB', ghost: 25112, ghostL: 'sem compressão: 43 × 584 B', pool: 17.04 * GiB, poolL: 'pool 17,04 GiB' },
      });
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Contexto', min: 0, max: 100, step: 1, value: 100, fmt: (v) => { const L = Math.round(1024 * Math.pow(1024, v / 100)); return L >= 1e6 ? nf(L / 1048576, 2) + 'M' : L >= 1e4 ? nf(L / 1024, 0) + 'K' : nf(L, 0); }, onInput: (v) => { S.v = v; } });
      UI.seg(ctl, { label: 'KV do Qwen', options: [{ v: 1, label: 'FP8' }, { v: 0, label: 'BF16' }], value: 1, onChange: (v) => { S.qfp8 = !!v; } });
      UI.toggle(ctl, { label: 'comparar com atenção em todas as camadas', value: false, onChange: (v) => { S.ghost = v; } });
      const ro = UI.readout(ctl, 'L (tokens)', '—');
      V.fig(fig, (g, w, hh, t, dt) => {
        const C = V.C(), pad = 14, narrow = w < 560;
        S.gA += ((S.ghost ? 1 : 0) - S.gA) * Math.min(1, dt * 4);
        const L = Ls(); ro.set(nf(L, 0));
        const D = M();
        const lw = narrow ? 64 : 104, x0 = pad + lw, x1 = w - pad - 4;
        const vals = ['glm', 'qwen', 'ds'].map((m) => { const d = D[m]; return { m, d, grow: L * d.grow, fixed: d.fixed, ghost: L * d.ghost + d.fixed }; });
        const maxV = Math.max(...vals.map((v) => Math.max(v.grow + v.fixed, v.d.pool * 1.05, S.gA > 0.01 ? v.ghost * S.gA + (v.grow + v.fixed) * (1 - S.gA) : 0))) * 1.08;
        if (narrow) { V.text(g, `UMA SEQUÊNCIA DE ${nf(L, 0)} TOKENS`, pad, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad }); V.text(g, 'MEMÓRIA POR RANK', pad, 35, { size: 11, font: 'mono', color: C.muted }); }
        else V.text(g, `UMA SEQUÊNCIA DE ${nf(L, 0)} TOKENS · MEMÓRIA POR RANK`, pad, 20, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const X = (v) => x0 + (v / maxV) * (x1 - x0);
        let y = narrow ? 64 : 50;
        vals.forEach((v, i) => {
          V.text(g, SHORT[v.m], pad, y + 18, { size: narrow ? 12 : 14, weight: 600, color: C[v.m] });
          // fantasma
          if (S.gA > 0.01) { g.globalAlpha = S.gA; g.setLineDash([4, 4]); V.rr(g, x0, y, X(v.ghost) - x0, 28, 5); g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.3; g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
          const xf = X(v.fixed);
          g.fillStyle = V.alpha(C[v.m], 0.3); g.fillRect(x0, y + 3, Math.max(1.5, xf - x0), 22); hatch(g, x0, y + 3, Math.max(1.5, xf - x0), 22, V.alpha(C[v.m], 0.8));
          const xg = X(v.fixed + v.grow);
          g.fillStyle = V.alpha(C[v.m], 0.85); g.fillRect(xf, y + 3, Math.max(0, xg - xf), 22);
          if (xg - xf > 6) { g.save(); g.beginPath(); g.rect(xf, y + 3, xg - xf, 22); g.clip(); for (let q = 0; q < 3; q++) { const u = ((t * 0.35 + q / 3) % 1); const sx = xf + u * (xg - xf); const gr = g.createLinearGradient(sx - 30, 0, sx + 30, 0); gr.addColorStop(0, V.alpha(C.bg, 0)); gr.addColorStop(0.5, V.alpha(C.bg, 0.35)); gr.addColorStop(1, V.alpha(C.bg, 0)); g.fillStyle = gr; g.fillRect(sx - 30, y + 3, 60, 22); } g.restore(); }
          // pool
          const xp = X(v.d.pool);
          V.line(g, xp, y - 4, xp, y + 32, { color: C.d, lw: 2 });
          V.text(g, v.d.poolL, xp - 4, y - 7, { size: 10, font: 'mono', align: 'right', color: C.d });
          const tot = v.grow + v.fixed;
          let ty = y + 44;
          if (narrow) {
            V.text(g, `${fB(tot)} por sequência`, x0, ty, { size: 12, font: 'mono', weight: 600, color: C.ink }); ty += 16;
            V.text(g, `${fB(v.grow)} cresce + ${fB(v.fixed)} fixo`, x0, ty, { size: 11, font: 'mono', color: C.ink, maxW: x1 - x0 }); ty += 15;
          } else {
            V.text(g, `${fB(tot)}  =  ${fB(v.grow)} (cresce) + ${fB(v.fixed)} (fixo)`, x0, ty, { size: 11.5, font: 'mono', color: C.ink, maxW: x1 - x0 }); ty += 16;
          }
          const fits = v.d.pool / tot;
          const sub = (S.gA > 0.5 ? `se fosse ${v.d.ghostL}: ${fB(v.ghost)}` : `${v.d.fixL} · cabem ≈ ${fits >= 10 ? nf(fits, 0) : nf(fits, 1)} dessas no pool (conta física)`);
          wrapText(g, sub, x1 - x0, 10.5).forEach((s2) => { V.text(g, s2, x0, ty, { size: 10.5, color: C.muted }); ty += 13; });
          if (v.m === 'qwen' && L > 262144) { V.text(g, '⚠ além de 262.144 tokens: exige YaRN', x0, ty, { size: 10.5, color: C.c, maxW: x1 - x0 }); ty += 13; }
          y = ty + 26;
        });
        fitTo(fig, y - 8);
      });

    })();

    /* ================================================================ FIG 8: linhagem (metrô) */
    (function () {
      const fig = root.querySelector('#fig-linhagem');
      // lab: 'a' = rótulo acima da estação, 'b' = abaixo
      const ST = [
        { id: 'v2', row: 0, x: 0.06, n: 'DeepSeek-V2', s: 'MLA', lab: 'b' },
        { id: 'v3', row: 0, x: 0.24, n: 'V3', s: 'MoE fino · MTP', lab: 'a' },
        { id: 'v32', row: 0, x: 0.43, n: 'V3.2-Exp', s: 'DSA', lab: 'b' },
        { id: 'v4', row: 0, x: 0.68, n: 'V4-Flash', s: 'CSA/HCA · mHC', lab: 'a' },
        { id: 'v4x', row: 0, x: 0.9, n: '0731 · Vision-Exp', s: 'DSpark', lab: 'b' },
        { id: 'mhc', row: 1, x: 0.56, n: 'paper mHC', s: 'DeepSeek', ext: true, lab: 'b' },
        { id: 'engram', row: 1, x: 0.34, n: 'Engram', s: 'DeepSeek', ext: true, lab: 'b' },
        { id: 'q3n', row: 2, x: 0.2, n: 'Qwen3-Next', s: 'GDN 3:1', lab: 'a' },
        { id: 'q35', row: 2, x: 0.46, n: 'Qwen3.5', s: 'multimodal', lab: 'b' },
        { id: 'q38', row: 2, x: 0.8, n: 'Qwen3.8-Flash', s: 'QSA · GR · N-gram', lab: 'a' },
        { id: 'kimi', row: 2.95, x: 0.42, n: 'Kimi Linear', s: 'KDA + MLA NoPE', ext: true, lab: 'b' },
        { id: 'dflash', row: 2.95, x: 0.68, n: 'DFlash', s: 'rascunho em bloco', ext: true, lab: 'b' },
        { id: 'g5', row: 4, x: 0.22, n: 'GLM-5', s: 'ancestral comum', lab: 'a' },
        { id: 'g53', row: 4.7, x: 0.52, n: 'GLM-5.3 (irmão)', s: '753B (744B na InferenceX)', lab: 'b' },
        { id: 'g53f', row: 4, x: 0.84, n: 'GLM-5.3-Flash', s: 'KDA · mHC · kpool', lab: 'b' },
      ];
      const byId = {}; ST.forEach((s) => { byId[s.id] = s; });
      // segmentos das linhas de cada família; o GLM-5 é ancestral comum de dois ramos irmãos
      const SEGS = [['v2', 'v3'], ['v3', 'v32'], ['v32', 'v4'], ['v4', 'v4x'], ['q3n', 'q35'], ['q35', 'q38'], ['g5', 'g53f'], ['g5', 'g53']];
      const TR = [
        { a: 'v2', b: 'g53f', l: 'latente 512 (MLA)', st: [0, 5], lu: 0.62 },
        { a: 'v3', b: 'g53f', l: 'sigmoide + noaux_tc', st: [1, 5], lu: 0.4 },
        { a: 'v32', b: 'g53f', l: 'DSA', st: [2, 5], lu: 0.62 },
        { a: 'v32', b: 'q38', l: 'indexador', st: [2, 4], lu: 0.45 },
        { a: 'engram', b: 'q38', l: 'N-gram', st: [4], lu: 0.5 },
        { a: 'g5', b: 'q38', l: 'reuso de índices', st: [4], lu: 0.55 },
        { a: 'kimi', b: 'g53f', l: 'KDA 3:1', st: [5], lu: 0.5 },
        { a: 'mhc', b: 'g53f', l: 'mHC', st: [5], lu: 0.28 },
        { a: 'mhc', b: 'v4', l: 'mHC', st: [6], lu: 0.5 },
        { a: 'dflash', b: 'g53f', l: 'DFlash2', st: [7], lu: 0.5 },
        { a: 'dflash', b: 'v4x', l: 'injeção de contexto', st: [7], lu: 0.45 },
      ];
      const STEPS = [
        { t: 'MLA: o KV vira um latente (DeepSeek-V2)', d: 'K e V de todas as cabeças passam a ser reconstruídos de um <b>latente de 512 dimensões</b> por token, mais uma chave RoPE de 64 compartilhada: 576 números por camada no cache, em vez de ~41 mil numa MHA de 128 cabeças. O MoE ainda é softmax, com 160 experts, top-6 e roteamento limitado por grupos. O latente de 512 chegará ao GLM-5.3-Flash, mas sem a parte RoPE.', on: ['v2'], seg: [] },
        { t: 'V3: MoE fino, sigmoide e sem perda auxiliar', d: '256 experts de 2048, top-8 + 1 compartilhado; score <b>sigmoide</b> e balanceamento por viés (<code>noaux_tc</code>), pesos ×2,5; 1 camada MTP; pesos FP8 em blocos 128×128. O GLM-5.3-Flash herda essa receita de roteamento quase campo a campo.', on: ['v3'], seg: [['v2', 'v3']] },
        { t: 'V3.2-Exp: o indexador relâmpago', d: '<b>DSA</b>: um indexador barato (64 cabeças × 128, em FP8) pontua todo o passado, e só os top-2048 tokens entram na MLA. A ideia vira família: kpool no GLM, QSA no Qwen, C4A no V4.', on: ['v32'], seg: [['v3', 'v32']] },
        { t: 'Qwen3-Next: 3 lineares para 1 atenção', d: '80 B totais, 3 B ativos. São 3 camadas <b>Gated DeltaNet</b> (estado fixo) para cada Gated Attention; 512 experts top-10 + 1 compartilhado; MTP. É o esqueleto que o Qwen3.5 e o 3.8 mantêm.', on: ['q3n'], seg: [] },
        { t: 'Qwen3.5 → Qwen3.8-Flash-Next', d: 'O 3.5 torna o modelo multimodal nativo e leva o vocabulário a 248.320. O 3.8, prévia do Qwen4, troca a atenção por <b>QSA</b>, alarga o resíduo em 4 ramos (<b>Gated Residual</b>) e acrescenta 51 B de <b>N-gram embedding</b>, a ideia do Engram do DeepSeek. O reuso de índices entre passos do MTP, diz o relatório, segue o GLM-5.', on: ['q35', 'q38', 'engram', 'g5'], seg: [['q3n', 'q35'], ['q35', 'q38']] },
        { t: 'GLM-5 e dois irmãos: 5.3 e 5.3-Flash', d: 'Do GLM-5 saem dois ramos. O <b>GLM-5.3</b> é o irmão grande (<code>GlmMoeDsaForCausalLM</code>, com DSA; 753B/40B no tutorial de receita do zero, 744B na InferenceX citada pelo inventário). O <b>Flash</b> (320B/18B) não é uma versão reduzida dele: tem outra arquitetura, que junta peças de três famílias. São <b>KDA + MLA NoPE em 3:1</b> (o desenho do Kimi Linear), o indexador DSA sobre pools de 4 chaves, o <b>mHC</b> e o roteamento sigmoide/noaux_tc à moda DeepSeek-V3, com 288 experts top-8.', on: ['g5', 'g53', 'g53f', 'kimi', 'mhc'], seg: [['g5', 'g53'], ['g5', 'g53f']] },
        { t: 'DeepSeek-V4-Flash: comprimir o KV no tempo', d: 'Em vez de só encolher cabeças, o V4 comprime o <b>eixo dos tokens</b>: janela de 128 em todas as camadas, KV ÷4 com indexador (C4A) ou ÷128 lido inteiro (C128A), e um único vetor de 512 que é K e V. Soma mHC, sqrtsoftplus top-6, hash nas 3 primeiras camadas e experts FP4. Resultado declarado: ~7% do KV do V3.2 a 1M de tokens.', on: ['v4', 'mhc'], seg: [['v32', 'v4']] },
        { t: 'Rascunhadores embutidos', d: 'MTP (V3) → DFlash (rascunho em bloco, com injeção de contexto das camadas do alvo) → <b>DFlash2</b> no GLM (externo, 1,17 B, bloco de 8) e <b>DSpark</b> no V4 (embutido, 3 estágios, bloco de 5, Markov head). O Qwen fica com 1 camada MTP aplicada 3 vezes.', on: ['v4x', 'dflash', 'g53f', 'q38'], seg: [['v4', 'v4x']] },
      ];
      const st = UI.stepper(fig, STEPS, { autoplay: 7000 });
      const famCol = (C, s) => s.row < 0.5 ? C.ds : Math.abs(s.row - 2) < 0.2 ? C.qwen : s.row >= 3.9 ? C.glm : C.muted;
      const bez = (a, m, b, u) => [(1 - u) * (1 - u) * a[0] + 2 * (1 - u) * u * m[0] + u * u * b[0], (1 - u) * (1 - u) * a[1] + 2 * (1 - u) * u * m[1] + u * u * b[1]];
      const ovl = (r, q) => Math.max(0, Math.min(r[0] + r[2], q[0] + q[2]) - Math.max(r[0], q[0])) * Math.max(0, Math.min(r[1] + r[3], q[1] + q[3]) - Math.max(r[1], q[1]));
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C();
        const pad = 16, narrow = w < 600;
        const rowY = (r) => 40 + r * ((hh - 118) / 4.7);
        const P = (s) => [pad + 24 + s.x * (w - 2 * pad - 48), rowY(s.row)];
        const step = STEPS[st.i];
        const el = Math.min(1, st.elapsed() / 0.8);
        const onSet = new Set(step.on);
        const sz = narrow ? 10 : 11.5, ssz = narrow ? 9.5 : 10;
        const maxN = narrow ? 92 : 160;
        // geometria dos rótulos das estações (para desenhar e para evitar colisões)
        const labs = ST.map((s) => {
          const [x, y] = P(s);
          const on = onSet.has(s.id);
          const below = s.lab === 'b';
          const ny = below ? y + 22 : y - 15;
          const sy = below ? ny + 13 : ny - 13;
          const wn = Math.min(maxN, textW(g, s.n, sz, 'sans', on ? 700 : 500));
          const showSub = !narrow || on;
          const ws = Math.min(maxN, textW(g, s.s, ssz));
          const rects = [[x - 11, y - 11, 22, 22], [x - wn / 2 - 3, ny - sz, wn + 6, sz + 5]];
          if (showSub) rects.push([x - ws / 2 - 3, sy - ssz, ws + 6, ssz + 5]);
          return { s, x, y, on, ny, sy, wn, ws, showSub, rects };
        });
        // nomes das famílias
        [[0, 'DeepSeek', C.ds], [2, 'Qwen', C.qwen], [4, 'GLM', C.glm]].forEach(([r, n, col]) => V.text(g, n, pad, rowY(r) - 16, { size: 11, font: 'mono', weight: 600, color: col }));
        // linhas das famílias (o ramo GLM-5 → GLM-5.3 sai na horizontal e desce, estilo metrô)
        SEGS.forEach(([ia, ib]) => {
          const A = byId[ia], B = byId[ib];
          const a = P(A), b = P(B);
          const hot = step.seg.some(([x, y]) => x === ia && y === ib);
          const col = famCol(C, A);
          g.strokeStyle = V.alpha(col, hot ? 0.95 : 0.35); g.lineWidth = hot ? 5 : 4; g.lineCap = 'round';
          const bend = Math.abs(a[1] - b[1]) > 4;
          const m = bend ? [b[0] - 30, a[1]] : [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
          g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(m[0], m[1], b[0], b[1]); g.stroke();
          if (hot) { const u = (t * 0.5) % 1; const pt = bez(a, m, b, u); V.glow(g, pt[0], pt[1], 12, col, 1); }
        });
        g.lineCap = 'butt';
        // transferências (curvas e partículas); as pílulas vêm por cima de tudo, no fim
        const pills = [];
        TR.forEach((tr) => {
          const A = byId[tr.a];
          const a = P(A), b = P(byId[tr.b]);
          const hot = tr.st.includes(st.i);
          const col = A.row < 1.5 || tr.a === 'mhc' || tr.a === 'engram' ? C.ds : A.row >= 3.9 ? C.glm : C.c;
          const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 18];
          g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(m[0], m[1], b[0], b[1]);
          g.setLineDash([4, 5]); g.lineDashOffset = -t * (hot ? 30 : 0);
          g.strokeStyle = V.alpha(col, hot ? 0.9 * el : 0.12); g.lineWidth = hot ? 2 : 1; g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
          if (hot) {
            for (let q = 0; q < 2; q++) { const pt = bez(a, m, b, (t * 0.35 + q * 0.5) % 1); V.dot(g, pt[0], pt[1], 3.2, col); }
            pills.push({ tr, a, m, b, col });
          }
        });
        // estações
        labs.forEach((L) => {
          const { s, x, y, on } = L;
          const col = s.ext ? C.muted : famCol(C, s);
          const rad = on ? 9 + 1.5 * Math.sin(t * 4) : 6.5;
          if (on) V.glow(g, x, y, 26, s.ext ? C.d : col, 0.7);
          g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fillStyle = C.bg; g.fill();
          g.lineWidth = on ? 3 : 2.2; g.strokeStyle = s.ext ? (on ? C.d : V.alpha(C.muted, 0.7)) : col;
          if (s.ext) g.setLineDash([2, 2]); g.stroke(); g.setLineDash([]);
          // placa de fundo: as linhas tracejadas passam por trás dos nomes, não por cima
          g.fillStyle = V.alpha(C.bg, 0.85);
          V.rr(g, x - L.wn / 2 - 3, L.ny - sz, L.wn + 6, sz + 5, 3); g.fill();
          if (L.showSub) { V.rr(g, x - L.ws / 2 - 3, L.sy - ssz, L.ws + 6, ssz + 5, 3); g.fill(); }
          V.text(g, s.n, x, L.ny, { size: sz, weight: on ? 700 : 500, align: 'center', color: on ? C.ink : V.alpha(C.ink, 0.7), maxW: maxN });
          if (L.showSub) V.text(g, s.s, x, L.sy, { size: ssz, align: 'center', color: C.muted, maxW: maxN });
        });
        // pílulas das ideias: procura ao longo da curva uma posição que não cubra outra pílula nem um nome de estação
        const placed = [];
        const blockers = [];
        labs.forEach((L) => L.rects.forEach((r) => blockers.push(r)));
        pills.forEach((P2) => {
          const tw = textW(g, P2.tr.l, 10.5) + 12;
          const base = P2.tr.lu || 0.5;
          const cands = [0, 0.07, -0.07, 0.14, -0.14, 0.21, -0.21, 0.28, -0.28].map((d) => base + d).filter((u) => u > 0.12 && u < 0.88);
          let best = null, bestO = Infinity;
          for (const u of cands) {
            const pt = bez(P2.a, P2.m, P2.b, u);
            const r = [pt[0] - tw / 2, pt[1] - 9, tw, 18];
            if (r[0] < 2 || r[0] + r[2] > w - 2) continue;
            let o = 0; placed.concat(blockers).forEach((q) => { o += ovl(r, q); });
            if (o < bestO) { bestO = o; best = r; }
            if (o === 0) break;
          }
          if (!best) { const pt = bez(P2.a, P2.m, P2.b, base); best = [V.clamp(pt[0] - tw / 2, 2, w - 2 - tw), pt[1] - 9, tw, 18]; }
          placed.push(best);
          g.globalAlpha = el;
          V.box(g, best[0], best[1], best[2], 17, { r: 8, fill: C.bg, stroke: V.alpha(P2.col, 0.85), lw: 1.2 });
          V.text(g, P2.tr.l, best[0] + best[2] / 2, best[1] + 12, { size: 10.5, align: 'center', color: C.ink });
          g.globalAlpha = 1;
        });
      });
      autoHeight(fig, (w) => (w < 600 ? 470 : 450));
    })();

    /* ================================================================ FIG 9: requisitos */
    (function () {
      const fig = root.querySelector('#fig-requisitos');
      const panel = fig.querySelector('#req-panel');
      const ROWS = [
        { l: 'Mixer linear com estado fixo', c: { glm: ['p', 'O KDA já vem no vLLM da imagem (chunk no prefill, recorrente no decode, N+1 estados na verificação especulativa). A receita corrige a liberação de estados no modo align (<code>patch_mamba_align_state_free</code> e <code>_chunking</code>) e restringe o PDL a SM9x/SM10x, porque ele corria com os kernels de estado KDA no GB10.'], qwen: ['o', 'GDN de fábrica (FLA no prefill, CUDA no decode). A receita só escolhe <code>--mamba-ssm-cache-dtype bfloat16</code>; o modelo força <code>mamba_cache_mode=align</code>.'], ds: ['n', 'Sem atenção linear: a memória constante vem das janelas de 128.'] } },
        { l: 'Atenção esparsa: indexador + kernel', c: { glm: ['p', 'Overlay NoPE no Dockerfile: pad de 64 zeros para a geometria GLM_NSA 576 do único kernel sparse-MLA do SM12x, prefill pelo caminho esparso, buffer top-k de 2048 (511 pools + cauda), página de kernel 64, alinhamento do indexador a 256 e workspace "rightsize" (+26–28% de KV).'], qwen: ['p', '<code>patch_qsa_fp8_kv</code> ensina os kernels QSA (indexador MQA + GQA esparso split-K) a ler KV FP8: 1,70× mais tokens por GiB. Perfis de tiles para os 48 SMs do GB10 (opcional).'], ds: ['p', 'O fix #22 roteia <code>nvfp4_ds_mla</code> para o kernel FP8 (sem ele, ~1 tok/s a 600K contra ~17 tok/s). Atalhos do indexador para contexto curto (#49486/#52492) e pulo de C128 vazio (#48957).'] } },
        { l: 'KV híbrido: grupos e prefix cache', c: { glm: ['p', '7 grupos de KV (MLA + indexador, cauda do kpool, 4× Mamba, janela do rascunhador). Patches: grupo do rascunhador com slot-share, prefix hit híbrido, clamp e stride da cauda do kpool (que corrompia o indexador de outra camada).'], qwen: ['p', 'Bloco de atenção igualado à página GDN (1.664 tokens); backport de <code>disable_eagle_block_drop</code> (#53388), para não perder o último bloco de prefix cache com MTP.'], ds: ['p', '1× MLAAttentionSpec (bloco 256) + 3× SlidingWindowMLASpec; hotfix #26 para o hit de prefixo com grupos SWA; <code>--block-size 256</code> para a página comum de 37.376 B.'] } },
        { l: 'Resíduo de 4 fluxos', c: { glm: ['p', '<code>patch_glm_eagle3</code>: para os taps do DFlash2, materializa o <code>hc_post</code> adiado e contrai os 4 fluxos para 4096.'], qwen: ['o', 'Gated Residual de fábrica, replicado nas duas GPUs (<code>disable_tp</code>): 1,19 GiB lidos por passo, 13% do total.'], ds: ['p', 'O hotfix #50312 libera o <code>_mtp_hidden_buffer</code> (256 MiB por rank), inútil para o DSpark. As features do rascunhador (média dos fluxos) vêm de fábrica.'] } },
        { l: 'MoE no formato quantizado', c: { glm: ['p', 'Registrar o método <code>exl3</code> e manter os experts como trellis + Hadamard (suh/svh) + MCG, com ExLlamaV3 compilado para sm_121a e kernels "fat" próprios. Nunca expandir para BF16.'], qwen: ['p', 'Experts NVFP4 de fábrica, mas os experts FP8 128×128 da MTP exigiram o alias <code>mtp.layers.48</code> e o ramo <code>FP8_BLOCK_SCALES</code> → Fp8MoEMethod; EP obrigatório.'], ds: ['o', 'O MegaMoE exige SM100 + EP. No GB10, a flag <code>--moe-backend flashinfer_b12x</code> roda os experts MXFP4 como W4A16 (dequant para BF16 + MMA BF16).'] } },
        { l: 'Roteamento e aliases de config', c: { glm: ['o', 'Aliases já presentes no vLLM da imagem (<code>Glm5NextTextConfig</code>); sem eles, 7 experts por token em vez de 8.'], qwen: ['o', 'Softmax top-10 + compartilhado com gate sigmoide, de fábrica.'], ds: ['p', 'Camadas hash precisam dos <code>input_ids</code> crus (patch Vision-Exp: <code>requires_raw_input_tokens</code>); tokens de imagem roteados com <code>bias_vl</code> e sem hash (#175).'] } },
        { l: 'Embedding gigante (N-gram)', c: { glm: ['n', 'Não se aplica.'], qwen: ['p', 'Tabela FP8 de 51 GB: porta do dispatch do PR #53899 (<code>ple_layer_patched.py</code>) + <code>detect_ple_dtype.py</code> + <code>ple_embedding_dtype</code> via --hf-overrides. Sem isso, o vLLM tenta criar uma embedding BF16 de ~102 GB.'], ds: ['n', 'Não se aplica (o Engram vem no V4.1, fora desta receita).'] } },
        { l: 'Posição e contexto longo', c: { glm: ['o', 'NoPE no texto: nada a escalar; 1.048.576 declarado (a receita serve 850k).'], qwen: ['o', 'YaRN por flag, com armadilha: o override precisa ir aninhado em <code>text_config</code> (<code>start.sh:919-946</code>), senão vira no-op silencioso. A receita aninha só <code>rope_type</code>, <code>factor</code> e <code>original_max_position_embeddings</code>; o card oficial passa o dicionário completo, e o inventário aponta o risco, não verificado, de o dict ser substituído inteiro. Por segurança, repita θ = 10⁷, 0,25 e o mRoPE. O 1M com YaRN não foi validado no kit (<code>YARN_ENABLE=false</code>).'], ds: ['o', 'YaRN ×16 de fábrica. O fix que restringe o YaRN às camadas comprimidas, como na referência, é opt-in (<code>DSPARK_ENABLE_ROPE_SWA_FIX</code>).'] } },
        { l: 'Rascunhador (speculative)', c: { glm: ['p', 'O DFlash2 é externo: registro da arquitetura, <code>is_causal: false</code> honrado (senão a aceitação das posições 2..7 colapsa), KV do rascunho em BF16 (auto) e grupo de KV próprio.'], qwen: ['p', 'MTP embutido: alias do índice 48 e vocabulário de rascunho de 47k (<span class="la-term" data-term="fr-spec">FR-Spec</span>), com argmax local por rank (+8–10% de decode).'], ds: ['p', 'DSpark embutido: a regra "k múltiplo de n_predict" força k = 6 no Vision-Exp (unlock de k = 5 opt-in); <code>--max-cudagraph-capture-size</code> 48 (6 sequências × 7 tokens, arredondado a múltiplo de 8), senão o passo de 42 tokens roda sem <span class="la-term" data-term="cuda-graph">CUDA graph</span> (−12% a c=6); patches de concorrência do rascunhador.'] } },
        { l: 'Divisibilidade em TP=3', c: { glm: ['p', 'Cabeças 64 → 66 (--hf-overrides + <code>patch_tp3_glm</code>), padding do vocabulário a 192, expert parallel para os experts (trellis 2048), rascunhador em TP=1.'], qwen: ['n', 'Não se aplica: a receita não tem lane TP=3 (há TP=1 e TP=2). Em TP=2, a flag de EP resolve a largura 640 dos experts, e o ViT roda replicado (4304/2 = 2152).'], ds: ['p', 'Grupos 8 → 9 (cabeças 64 → 72), com grupo extra zerado e attention sink −∞ (<code>apply_tp3_patch.py</code>).'] } },
        { l: 'Torre de visão', c: { glm: ['o', 'ViT de 24 blocos no vLLM da imagem, com ε forçado a 1e-6; a receita só limita <code>max_image_tokens</code> a 2048.'], qwen: ['o', 'ViT de 27 blocos de fábrica; <code>--mm-encoder-tp-mode data</code> (replicar), porque 4304 não divide bem por 2.'], ds: ['p', 'A imagem é só-texto: o hotfix instala ViT + Aligner + processador; o hash do encoder cache passa a ser "salgado" com <code>num_tokens</code> (#172).'] } },
      ];
      const MODELS = ['glm', 'qwen', 'ds'];
      const S = { sel: [1, 0], hover: null, focus: 'all' };
      const stName = { p: 'patch / overlay', o: 'de fábrica (ou flag)', n: 'não se aplica' };
      const colOf = (C, s) => s === 'p' ? C.d : s === 'o' ? C.a : V.alpha(C.muted, 0.5);
      const upd = () => {
        const C = V.C();
        const [r, c] = S.sel; const m = MODELS[c]; const cell = ROWS[r].c[m];
        panel.innerHTML = '';
        panel.appendChild(h('div', { class: 't18-req' },
          h('h5', null, ROWS[r].l, ' · ', NAME[m], h('span', { class: 'st', style: { background: V.alpha(colOf(C, cell[0]), 0.2), color: 'var(--text)', border: '1px solid ' + colOf(C, cell[0]) } }, stName[cell[0]])),
          h('div', { html: cell[1] })));
      };
      upd();
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Destacar', options: [{ v: 'all', label: 'todos' }].concat(MODEL_OPTS), value: 'all', onChange: (v) => { S.focus = v; if (v !== 'all') { S.sel = [S.sel[0], MODELS.indexOf(v)]; upd(); } } });
      const cnt = UI.readout(ctl, 'Patches', '—');
      let geo = null;
      V.fig(fig, (g, w, hh, t, dt, _S, H) => {
        const C = V.C(), pad = 14, narrow = w < 560;
        const lw = Math.min(narrow ? 150 : 250, w * 0.42);
        const cx0 = pad + lw + 6, cw = (w - pad - cx0) / 3;
        const top = 44;
        const rh = (hh - top - 10) / ROWS.length;
        geo = { cx0, cw, top, rh };
        MODELS.forEach((m, j) => {
          const on = S.focus === 'all' || S.focus === m;
          V.text(g, narrow ? SHORT[m] : NAME[m].replace('-Flash-Next', '').replace('-Flash', ''), cx0 + j * cw + cw / 2, 20, { size: narrow ? 11.5 : 12.5, weight: 700, align: 'center', color: on ? C[m] : V.alpha(C[m], 0.35) });
          const np = ROWS.filter((r) => r.c[m][0] === 'p').length;
          V.text(g, narrow ? `${np}/${ROWS.length} ⚙` : `${np} de ${ROWS.length} com patch`, cx0 + j * cw + cw / 2, 35, { size: 10, align: 'center', color: C.muted });
        });
        const pt = H.pointer;
        S.hover = null;
        if (pt.inside && pt.x > cx0 && pt.y > top) { const r = Math.floor((pt.y - top) / rh), c = Math.floor((pt.x - cx0) / cw); if (r >= 0 && r < ROWS.length && c >= 0 && c < 3) S.hover = [r, c]; }
        if (pt.clicked && S.hover) { S.sel = S.hover; upd(); }
        ROWS.forEach((row, i) => {
          const y = top + i * rh;
          if (i % 2 === 0) { g.fillStyle = V.alpha(C.ink, 0.035); g.fillRect(pad, y, w - 2 * pad, rh); }
          const lines = wrapText(g, row.l, lw - 8, narrow ? 11 : 12.5, 500);
          lines.slice(0, 3).forEach((s, k) => V.text(g, s, pad + 6, y + rh / 2 + (k - (Math.min(3, lines.length) - 1) / 2) * 14 + 4, { size: narrow ? 11 : 12.5, weight: 500, color: C.ink }));
          MODELS.forEach((m, j) => {
            const s = row.c[m][0];
            const x = cx0 + j * cw + cw / 2, yy = y + rh / 2;
            const on = S.focus === 'all' || S.focus === m;
            const col = colOf(C, s);
            const puls = s === 'p' ? 1 + 0.12 * Math.sin(t * 3 + i * 0.7 + j) : 1;
            const rad = Math.min(10, rh * 0.32) * puls;
            g.globalAlpha = on ? 1 : 0.25;
            if (s === 'p') V.glow(g, x, yy, rad * 2.2, col, 0.35);
            g.beginPath(); g.arc(x, yy, rad, 0, Math.PI * 2); g.fillStyle = s === 'n' ? 'transparent' : V.alpha(col, 0.9); g.fill();
            g.lineWidth = 1.5; g.strokeStyle = col; g.stroke();
            if (s === 'p') V.text(g, '⚙', x, yy + 4, { size: 11, align: 'center', color: C.bg });
            if (s === 'o') V.text(g, '✓', x, yy + 4, { size: 11, align: 'center', color: C.bg, weight: 700 });
            g.globalAlpha = 1;
            const isSel = S.sel[0] === i && S.sel[1] === j, isH = S.hover && S.hover[0] === i && S.hover[1] === j;
            if (isSel || isH) { g.strokeStyle = isSel ? C.ink : V.alpha(C.ink, 0.5); g.lineWidth = isSel ? 2 : 1.2; V.rr(g, cx0 + j * cw + 4, y + 2, cw - 8, rh - 4, 8); g.stroke(); }
          });
        });
        const tot = MODELS.map((m) => ROWS.filter((r) => r.c[m][0] === 'p').length);
        cnt.set(tot.join(' · '));
      });
      autoHeight(fig, (w) => (w < 560 ? 44 + ROWS.length * 44 + 12 : 44 + ROWS.length * 38 + 12));
    })();

    /* ================================================================ FIG 10: exercício MiMo */
    (function () {
      const fig = root.querySelector('#fig-exercicio');
      const pat = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0];
      const L = pat.map((p, i) => ({ i, mix: p === 0 ? 'full' : 'swa', ffn: i === 0 ? 'dense' : 'moe' }));
      const STEPS = [
        { t: 'Forma e mapa', d: '<code>num_hidden_layers: 48</code>; <code>hybrid_layer_pattern</code> tem 9 zeros (atenção global densa, sem indexador, nas camadas 0, 5, 11, …, 47) e 39 uns (janela de 128). <code>moe_layer_freq</code> começa com 0: a camada 0 é densa (16.384) e as outras 47 são MoE de 256 experts, top-8, <b>sem expert compartilhado</b>.', hl: () => true },
        { t: 'Parâmetros e pesos', d: 'Experts: 47 × 256 × 3 × 2048 × 4096 = <b>302,8 B</b>. Em MXFP4 (4,25 bits por peso) dá 160,9 GB (149,8 GiB), ou 74,9 GiB por rank em TP=2. O backbone inteiro fica em 79,1 GiB por rank; com encoders e DFlash, a estimativa é <b>≈ 82 GiB por rank</b>, contra 82,7 GiB medidos num kit SGLang.', hl: (l) => l.ffn === 'moe' },
        { t: 'Ativos e FLOPs', d: '8 × 25,17 M × 47 ≈ 9,5 B nos experts, mais ~4,5 B de atenção (qkv FP8 + o_proj BF16), camada densa e cabeças: <b>~15 B ativos</b>, como diz o cartão. Pela regra 2N, ≈ 30 GFLOP por token (estimativa ilustrativa).', hl: (l) => l.ffn === 'moe' },
        { t: 'KV que cresce', d: 'Só as 9 globais (atenção densa plena, sem indexador): 9 × 2 cabeças de KV por rank × (192 de K + 128 de V) × 1 B (FP8) = <b>5.760 B por token por rank</b> (11.520 em BF16). Compare: 7.579 no GLM e 6.656 no Qwen.', hl: (l) => l.mix === 'full' },
        { t: 'Estado fixo', d: 'As 39 camadas de janela guardam só ~128 tokens (mais k e o arredondamento de bloco, ~192–256): 39 × 4 × 320 × 1 B × ~256 ≈ <b>10–13 MB por request</b> em FP8. O DFlash (5 camadas SWA, janela 1024) soma ~11 MB em BF16.', hl: (l) => l.mix === 'swa' },
        { t: 'Pool e capacidade', d: 'Orçamento a GMU 0,85: 0,85 × 121,69 = 103,44 GiB, menos 82,7 GiB de pesos e ~4 GiB de overhead, sobra um pool de <b>16,74 GiB</b>. Dividido por 5.760 B, dá 3,12 M tokens. Mas o alocador híbrido agrupa as camadas de 5 em 5, e as 9 globais ocupam 10 "slots" (+11% de enchimento): sobram <b>≈ 2,81 M tokens</b> em FP8, ou seja, 1M de contexto cabe ~2,7×. Bate com os 2.731.530 tokens medidos (#58128, pool de 17 GiB). O PR #58175 mediu +10,57% de capacidade com grupos de 3.', hl: (l) => l.mix === 'full' },
        { t: 'Gargalos e requisitos', d: 'K (192) ≠ V (128): backend DiffKV, que no GB10 significa Triton. QKV fundido e <b>pré-fatiado para TP=4</b> servido em TP=2. Sinks só nas camadas SWA. <code>sliding_window</code> global podendo contaminar as camadas globais (bug silencioso). Kernel MoE MXFP4 para sm_121. Dois rascunhadores (MTP-3 e DFlash-5).', hl: () => true },
      ];
      const st = UI.stepper(fig, STEPS, { autoplay: 7500 });
      const O = { mh: 22, fh: 8, labelH: 14, markH: 4, minCell: 12, maxCell: 26 };
      V.fig(fig, (g, w, hh, t) => {
        const C = V.C(), pad = 14, i = st.i, step = STEPS[i];
        V.text(g, 'MiMo-V2.6-Flash-MOPD · hybrid_layer_pattern', pad, 18, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const G = stripGeom(w, L, Object.assign({ y: 30 }, O));
        const pulse = 0.6 + 0.4 * Math.sin(t * 3);
        drawStrip(g, G, Object.assign({ mode: 'tipo', labels: true, lit: step.hl }, O));
        G.pos.forEach((p) => { if (step.hl(p.l) && i >= 3 && i <= 5) V.glow(g, p.x + G.cw / 2, p.y + O.mh / 2, 9, mixCol(C, p.l.mix), 0.3 * pulse); });
        // legenda (as globais do MiMo são atenção densa plena: cor própria, não a lavanda de "esparsa")
        let lgx = pad, lgy = 30 + G.height + 12;
        [[C.e, 'GA global densa (9)'], [C.d, 'SWA 128 (39)'], [V.alpha(C.ink, 0.9), 'FFN densa (camada 0)'], [V.alpha(C.ink, 0.38), 'MoE (47)']].forEach(([c, s2]) => {
          const ww = textW(g, s2, 11) + 26;
          if (lgx + ww > w - pad) { lgx = pad; lgy += 17; }
          V.box(g, lgx, lgy - 9, 11, 11, { r: 3, fill: c });
          V.text(g, s2, lgx + 16, lgy + 1, { size: 11, color: C.muted });
          lgx += ww;
        });
        // barra de orçamento
        const by = lgy + 26;
        const budget = 103.44;
        const segs = [['pesos (≈ 82,7 GiB)', 82.7, C.b, 1], ['overhead ~4', 4, C.muted, 1], ['pool de KV 16,74', 16.74, C.a, 5]];
        const bx0 = pad, bx1 = w - pad, bw = bx1 - bx0;
        V.text(g, 'UM RANK (TP=2) · GMU 0,85 × 121,69 GiB = 103,44 GiB', pad, by, { size: 10.5, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        V.box(g, bx0, by + 8, bw, 30, { r: 8, fill: V.alpha(C.ink, 0.06), stroke: V.alpha(C.muted, 0.35) });
        let x = bx0;
        segs.forEach(([l, v, col, from], k) => {
          const show = i >= from || (k === 0 && i >= 1);
          const grow = i === from ? V.ease.out(Math.min(1, st.elapsed() / 1.2)) : 1;
          const ww = (v / budget) * bw * (show ? grow : 0);
          if (ww > 0.5) { g.fillStyle = V.alpha(col, 0.8); V.rr(g, x, by + 8, ww, 30, 6); g.fill(); if (ww > textW(g, l, 11) + 12) V.text(g, l, x + 8, by + 27, { size: 11, weight: 600, color: C.bg }); }
          x += (v / budget) * bw * (show ? 1 : 0);
        });
        // números do passo
        const big = [['48 camadas', '9 globais + 39 SWA'], ['≈ 82 GiB/rank', 'pesos em TP=2'], ['~15 B ativos', '≈ 30 GFLOP/token'], ['5.760 B/token', 'KV FP8 por rank'], ['10–13 MB/req', 'janelas SWA (FP8)'], ['~2,81 M tokens', '1M cabe ~2,7×'], ['6 requisitos', 'DiffKV · TP4 · sinks · MXFP4…']][i];
        const ny = by + 70;
        V.text(g, big[0], pad, ny, { size: w < 560 ? 22 : 28, font: 'display', weight: 700, color: C.ink });
        V.text(g, big[1], pad, ny + 20, { size: 12, color: C.muted, maxW: w - 2 * pad });
        const FORM = ['9 × global + 39 × SWA · 1 densa + 47 MoE', '47 × 256 × 3 × 2048 × 4096 × 4,25/8 B ≈ 149,8 GiB', '≈ 9,5 (experts) + 4,5 (atenção) + 0,2 (densa) + 0,6 (head) ≈ 14,8 B', '9 × 2 × (192 + 128) × 1 B = 5.760 B/token', '39 × 4 × 320 × 1 B × ~256 tokens ≈ 10–13 MB', '16,74 GiB ÷ 5.760 B = 3,12 M; grupos de 5 → 9 GA em 10 slots (× 10/9 = +11%) → 3,12 ÷ 1,11 ≈ 2,81 M'];
        if (i < 6) { const fl = wrapText(g, FORM[i], w - 2 * pad, 12, 400, 'mono'); g.globalAlpha = Math.min(1, st.elapsed() * 2); fl.slice(0, 3).forEach((s2, k) => V.text(g, s2, pad, ny + 48 + k * 17, { size: 12, font: 'mono', color: C.a })); g.globalAlpha = 1; }
        if (i === 6) {
          const req = ['K ≠ V (DiffKV)', 'QKV pré-fatiado TP=4', 'sinks só em SWA', 'sliding_window vaza', 'MoE MXFP4 sm_121', '2 rascunhadores'];
          let rx = pad, ry = ny + 36;
          req.forEach((s, k) => { const ww = textW(g, s, 11) + 18; if (rx + ww > w - pad) { rx = pad; ry += 26; } const a = Math.min(1, Math.max(0, st.elapsed() * 2 - k * 0.3)); g.globalAlpha = a; V.box(g, rx, ry, ww, 21, { r: 10, fill: V.alpha(C.d, 0.18), stroke: C.d, label: s, size: 11 }); g.globalAlpha = 1; rx += ww + 6; });
        }
        fitTo(fig, ny + 36 + (w < 560 ? 3 : 2) * 26 + 8);
      });
    })();
  },
});

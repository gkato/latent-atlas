LA.chapter({
  id: 'sft',
  part: 10,
  order: 52,
  title: 'Supervised Fine-Tuning: aprendendo com demonstrações',
  navTitle: 'SFT',
  subtitle: 'Como um completador de texto vira assistente: a perda com máscara, o chat template como dado, packing, de onde vêm boas demonstrações, a destilação on-policy e o MOPD, as curvas de treino, o esquecimento e as configurações reais.',
  minutes: 45,
  recipes: ['qwen', 'ds'],
  tags: ['SFT', 'entropia cruzada', 'máscara de perda', 'chat template', 'packing', 'RFT', 'STaR', 'destilação on-policy', 'MOPD', 'esquecimento catastrófico', 'TRL', 'mlx-lm'],
  goals: [
    'Escrever a perda do SFT com <b>máscara e normalização</b> corretas e explicar por que ela é um KL direto <b>off-policy</b> sobre demonstrações (e o que é viés de exposição)',
    'Montar dados que funcionam: JSONL no <b>chat template exato</b> do serviço, fim de turno na perda, <b>packing</b> sem contaminação, traces de raciocínio e de ferramentas',
    'Escolher de onde vêm as demonstrações (humanos, professores, RFT/STaR) e entender a <b>destilação on-policy</b>, o <b>MOPD</b> e o MOPD2 com os números das fontes',
    'Configurar LR, épocas e lote em TRL, Unsloth, Axolotl, LLaMA-Factory e mlx-lm, ler as curvas e medir <b>esquecimento</b> e regressões antes de servir',
  ],
  prereqs: ['pos-treino-mapa', 'treino-memoria-lora', 'tokens-embeddings', 'aprender-com-dados'],
  summary: [
    'SFT é máxima verossimilhança das respostas: \\(\\mathcal L = -\\tfrac{1}{\\sum m}\\sum m_t \\log\\pi_\\theta(y_t\\mid x, y_{&lt;t})\\). <code>labels</code> = <code>input_ids</code> nas mesmas posições, <code>-100</code> fora do assistente, deslocamento de uma posição feito dentro da perda (não desloque à mão), média sobre todos os tokens de perda do lote acumulado (o Tülu 3 usou soma; o TRL, <code>average_tokens_across_devices</code>).',
    'Minimizar essa perda é minimizar o KL <b>direto</b> \\(\\mathrm{KL}(p_{\\text{dados}}\\|\\pi_\\theta)\\), que cobre todos os modos. E é off-policy: o teacher forcing treina em prefixos do dataset, e o viés de exposição aparece na geração.',
    'O template é dado: byte a byte o do serviço; <code>&lt;|im_end|&gt;</code> (248046 no Qwen) dentro da perda, ou o modelo não para; raciocínio em <code>reasoning_content</code>; DeepSeek-V4 sem Jinja exige dataset pré-tokenizado pelo <code>encoding_dsv4.py</code>; tool calls no formato do parser.',
    'Packing BFD com <code>position_ids</code> reiniciados e atenção de comprimento variável elimina o padding sem contaminar exemplos (o SFT do DeepSeek-V3 usou máscara entre amostras); <code>keep_start</code> corta justamente a resposta.',
    'Dados: formato com poucos e bons (LIMA, 1.000), desempenho de ponta com muitos e verificados (OpenThoughts, 1,2 M; R1-Distill, ~800 mil). O RFT gera, filtra e treina (GSM8K de 35,9% para 49,3%); o STaR racionaliza os erros.',
    'Destilação on-policy: o aluno gera e o professor dá \\(\\hat A_t = \\log\\pi_T(y_t) - \\log\\pi_\\theta(y_t)\\), um KL reverso denso (Qwen3-8B: 74,4 contra 67,6 no AIME\'24 com ~1/10 das GPU-horas). O MOPD roteia cada prompt ao professor RL do seu domínio; o MOPD2 acrescenta prefixos de professor e de SFT e corrigiu as tool calls repetidas com ~US$ 90 mil.',
    'Full FT com pico em geral entre ~2e-6 e 2e-5 (Tülu 3 e DeepSeek-V3 em 2e-6–5e-6; 2e-5 é o padrão do TRL; o MoE MiMo-V2-Flash partiu de 5e-5), LoRA ≈10× mais, 1 a 3 épocas, checkpoint escolhido pela validação. O esquecimento acompanha o KL ao base medido na tarefa nova (RL\'s Razor). Antes de servir, rode A/B/A, meça KL ao base, EOS, parse de tool calls, repetição e a aceitação do rascunho.',
  ],
  quiz: [
    { q: 'Num SFT conversacional com <code>assistant_only_loss=True</code>, quais posições recebem rótulo <code>-100</code>?', o: ['Só os tokens especiais do template', 'System, usuário, resultados de ferramenta e o cabeçalho <code>&lt;|im_start|&gt;assistant</code>; o que o assistente escreve, inclusive o seu <code>&lt;|im_end|&gt;</code>, mantém o rótulo', 'Todos os tokens do assistente, para o modelo não decorar respostas', 'Só os tokens de raciocínio dentro de <code>&lt;think&gt;</code>'], a: 1, e: 'O −100 é o <code>ignore_index</code>: tira a posição da soma e do denominador. Tudo o que é contexto (system, usuário, resultados de ferramenta, o gatilho de geração) fica fora; tudo o que o assistente escreve, incluindo raciocínio, chamadas de ferramenta e o fim de turno, fica dentro.' },
    { q: 'Depois de um SFT, o modelo responde certo, mas emenda <code>&lt;|im_start|&gt;user</code> e continua escrevendo até <code>max_tokens</code>. Qual a causa mais provável?', o: ['LR alto demais', 'O token de fim de turno ficou fora da perda, ou o <code>eos_token</code> não bate com o do template', 'Packing com BFD', 'Faltou weight decay'], a: 1, e: 'Parar é uma predição como outra qualquer: o modelo só aprende a emitir <code>&lt;|im_end|&gt;</code> se esse token tiver rótulo. O TRL pede o <code>eos_token</code> alinhado ao template (no Qwen2.5-1.5B base, <code>eos_token="&lt;|im_end|&gt;"</code>), e o servidor precisa reconhecer o mesmo id como fim (248046 no Qwen3.8).' },
    { q: 'Um micro-lote tem uma resposta de 20 tokens; outro, uma de 2.000. Se você tira a média da perda em cada micro-lote e depois a média das duas médias, quanto pesa um token da resposta curta em relação a um da longa?', o: ['O mesmo', '10×', '100×', '0,01×'], a: 2, e: 'As médias dão peso 1/20 e 1/2.000 aos seus tokens, e a média das médias multiplica ambos por ½: a razão é 100. É o bug da acumulação de gradientes, e o resultado muda com o tamanho do micro-lote e com o número de GPUs. A correção é dividir pelo total de tokens de perda do lote acumulado (<code>average_tokens_across_devices=True</code> no TRL) ou usar a soma, como o Tülu 3, recalibrando o LR.' },
    { q: 'Por que a destilação on-policy com KL reverso funciona bem quando o aluno é bem menor que o professor?', o: ['Porque o KL reverso é simétrico', 'Porque o KL reverso é mode-seeking: o aluno é punido por pôr massa onde o professor não põe e se concentra nos modos que consegue representar, em vez de espalhar massa pelo vale entre eles', 'Porque dispensa o professor durante o treino', 'Porque usa menos memória que o SFT'], a: 1, e: 'O KL direto do SFT é mass-covering: pune deixar sem massa qualquer coisa que o professor faz, e um aluno pequeno acaba pondo massa entre os modos, gerando o que o professor nunca diria. O KL reverso, calculado nas amostras do próprio aluno, prefere cobrir bem um modo. E treina nos estados que o aluno visita, sem viés de exposição.' },
    { q: 'Na destilação on-policy, o aluno escreveu “18” com probabilidade 0,4, e o professor dá 0,05 a esse token. Com o clipping do MOPD em \\(A_{\\max} = 1{,}5\\), qual vantagem entra na perda?', o: ['+2,08', '−2,08', '−1,5', '0, porque o token está errado e é descartado'], a: 2, e: '\\(\\hat A = \\ln(0{,}05/0{,}4) = \\ln 0{,}125 \\approx -2{,}08\\), e o clipping dos dois lados limita a −1,5. O sinal negativo empurra “18” para baixo. Descartar é outra coisa: o peso \\(w_t\\) do MiMo zera tokens em que a razão entre o motor de treino e o de inferência sai de \\([\\varepsilon_{\\text{low}}, \\varepsilon_{\\text{high}}]\\).' },
    { q: 'Você trocou um full fine-tuning que funcionava com LR 2e-5 por LoRA em todas as camadas lineares. Qual ponto de partida as fontes indicam para o LR?', o: ['2e-6, porque o LoRA treina menos parâmetros', '2e-5: o LR não depende do método', '~2e-4: cerca de 10× o do full fine-tuning, quase independente do posto', '2e-2'], a: 2, e: 'O <i>LoRA Without Regret</i> mede o LR ótimo do LoRA em ~10× o do full (≈15× em runs muito curtos), e o prefator \\(\\alpha/r\\) faz esse ótimo variar menos de 2× entre r = 4 e r = 512. Com o LR do full, o LoRA quase não anda. Os exemplos reais seguem isso: Axolotl 2e-4, LLaMA-Factory 1e-4, dica do TRL ~1e-4.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 2) => (+x).toFixed(d).replace('.', ',');
    const pct = (x, d = 0) => fmt(x * 100, d) + '%';
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 560) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(font === 'mono' ? V.noLig(String(s)) : String(s)).width; }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (o.dash) { g.save(); g.setLineDash(o.dash); V.rr(g, x, y, w, h, o.r == null ? 6 : o.r); g.strokeStyle = o.dashColor || C.muted; g.lineWidth = 1.2; g.stroke(); g.restore(); }
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 4 });
      if (o.alpha != null) g.globalAlpha = 1;
    }
    /** personagem: quadrado arredondado com olhinhos */
    function critter(g, x, y, s, col, o = {}) {
      const C = V.C();
      V.box(g, x - s / 2, y - s / 2, s, s, { r: s * 0.3, fill: o.fill || V.alpha(col, 0.85), stroke: col, lw: 1.2 });
      if (s < 11) return;
      const eye = C.dark ? '#06140d' : '#ffffff';
      const ex = s * 0.17, ey = y - s * 0.08, er = Math.max(1, s * 0.07);
      V.dot(g, x - ex, ey, er, eye); V.dot(g, x + ex, ey, er, eye);
      g.beginPath();
      if (o.mood === 'sad') g.arc(x, y + s * 0.3, s * 0.14, Math.PI * 1.15, Math.PI * 1.85);
      else if (o.mood === 'o') g.arc(x, y + s * 0.16, s * 0.07, 0, Math.PI * 2);
      else g.arc(x, y + s * 0.06, s * 0.15, Math.PI * 0.2, Math.PI * 0.8);
      g.strokeStyle = eye; g.lineWidth = Math.max(1, s * 0.06); g.stroke();
    }
    function mark(g, x, y, ok, s = 6, col) {
      const C = V.C();
      g.lineWidth = 2.2; g.strokeStyle = col || (ok ? C.a : RED()); g.lineCap = 'round'; g.beginPath();
      if (ok) { g.moveTo(x - s, y); g.lineTo(x - s * 0.3, y + s * 0.7); g.lineTo(x + s, y - s * 0.8); }
      else { g.moveTo(x - s * 0.8, y - s * 0.8); g.lineTo(x + s * 0.8, y + s * 0.8); g.moveTo(x + s * 0.8, y - s * 0.8); g.lineTo(x - s * 0.8, y + s * 0.8); }
      g.stroke(); g.lineCap = 'butt';
    }
    function tri(g, x, y, s, up, col) { g.beginPath(); if (up) { g.moveTo(x, y - s); g.lineTo(x + s * 0.9, y + s * 0.6); g.lineTo(x - s * 0.9, y + s * 0.6); } else { g.moveTo(x, y + s); g.lineTo(x + s * 0.9, y - s * 0.6); g.lineTo(x - s * 0.9, y - s * 0.6); } g.closePath(); g.fillStyle = col; g.fill(); }
    function hatch(g, x, y, w, h, col, step = 6) {
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1;
      g.beginPath(); for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); } g.stroke();
      g.restore();
    }
    function zigzag(g, x, y, h, col) { g.beginPath(); const n = 4, st = h / n; g.moveTo(x, y); for (let i = 1; i <= n; i++) g.lineTo(x + (i % 2 ? 3 : -3), y + i * st); g.strokeStyle = col; g.lineWidth = 2; g.stroke(); }
    /** canvas sem ligaduras da fonte mono (<| |> viravam setas) */
    const figT = (fig, draw, opts) => V.fig(fig, (g, ...a) => { g.textRendering = 'optimizeSpeed'; draw(g, ...a); }, opts);
    const softmax = (z) => { let m = -Infinity; for (const v of z) if (v > m) m = v; let s = 0; const e = z.map((v) => { const x = Math.exp(v - m); s += x; return x; }); return e.map((x) => x / s); };
    /** layout em fluxo de itens de larguras dadas */
    function flow(widths, x0, maxW, rowH, gap = 5) {
      const pos = []; let x = x0, row = 0;
      widths.forEach((w) => { if (x + w > x0 + maxW && x > x0) { x = x0; row++; } pos.push({ x, row }); x += w + gap; });
      return { pos, rows: row + 1, rowH };
    }

    /* ================================================ FIG 1 — base → SFT */
    (function () {
      const fig = root.querySelector('#fig-base-sft');
      adaptH(fig, 430, 500);
      const CAND = ['Qual', '↵', '1.', 'Resposta', 'A', 'Paris', 'outros'];
      const PB = [0.34, 0.18, 0.14, 0.12, 0.10, 0.07, 0.05];
      const PS = [0.01, 0.02, 0.01, 0.03, 0.55, 0.35, 0.03];
      const ZB = PB.map(Math.log), ZS = PS.map(Math.log);
      const DEMO = ['A', ' capital', ' da', ' França', ' é', ' Paris', '.', '<|im_end|>'];
      const DB = [0.10, 0.55, 0.80, 0.85, 0.60, 0.70, 0.35, 0.01];
      const DS = [0.55, 0.97, 0.98, 0.97, 0.93, 0.96, 0.90, 0.97];
      const S = { s: 0, auto: true, prevP: null, parts: [], sampleKey: '', sampleT0: 0 };
      const distAt = (s) => softmax(ZB.map((z, i) => (1 - s) * z + s * ZS[i]));
      const demoAt = (s) => DEMO.map((_, i) => (i === 0 ? distAt(s)[4] : Math.exp((1 - s) * Math.log(DB[i]) + s * Math.log(DS[i]))));
      const ctl = UI.controls(fig);
      const sl = UI.slider(ctl, { label: 'progresso do SFT', min: 0, max: 100, step: 1, value: 0, fmt: (v) => v + '%', onInput: (v) => { S.s = v / 100; if (S.auto) { S.auto = false; tg.set(false); } } });
      const tg = UI.toggle(ctl, { label: 'animar sozinho', value: true, onChange: (v) => { S.auto = v; } });
      const rL = UI.readout(ctl, 'perda média (nats)', '—'), rE = UI.readout(ctl, 'p(<|im_end|>)', '—'), rM = UI.readout(ctl, 'amostra', '—');
      nocase(sl);
      let lastR = '';
      figT(fig, (g, w, h, t, dt) => {
        const C = V.C();
        if (S.auto) { S.s = V.smooth(V.pingpong(t, 14)); const v = Math.round(S.s * 100); if (+sl.input.value !== v) sl.set(v, true); }
        const s = S.s, P = distAt(s), D = demoAt(s);
        const loss = D.reduce((a, p) => a - Math.log(p), 0) / D.length;
        const pEos = D[D.length - 1];
        let am = 0; P.forEach((p, i) => { if (p > P[am]) am = i; });
        const regime = am !== 4 && am !== 5 ? 'base' : pEos < 0.5 ? 'nostop' : 'ok';
        const rs = fmt(loss, 2) + '|' + fmt(pEos, 3) + '|' + regime;
        if (rs !== lastR) { lastR = rs; rL.set(fmt(loss, 2)); rE.set(fmt(pEos, 3), pEos > 0.5 ? 'ok' : 'warn'); rM.set(regime === 'base' ? 'continua a lista' : regime === 'nostop' ? 'responde, não para' : 'responde e para', regime === 'ok' ? 'ok' : regime === 'base' ? 'bad' : 'warn'); }
        const narrow = w < 560;
        const pad = 12;
        // ---- banda A: distribuição do próximo token
        const aH = h * (narrow ? 0.4 : 0.42);
        V.text(g, 'próximo token depois de <|im_start|>assistant', pad, 18, { size: 12, font: 'mono', color: C.muted, maxW: w - 2 * pad });
        const n = CAND.length, gw = (w - 2 * pad) / n, base = aH - 24, bh = base - 46;
        const bx = (i) => pad + gw * i + gw / 2;
        const mx = 0.6;
        // partículas de massa
        if (S.prevP && dt > 0) {
          const dec = [], inc = [];
          P.forEach((p, i) => { const d = p - S.prevP[i]; if (d < -1e-5) dec.push([i, -d]); else if (d > 1e-5) inc.push([i, d]); });
          const tot = inc.reduce((a, b) => a + b[1], 0);
          if (tot > 0 && S.parts.length < 90) {
            dec.forEach(([i, d]) => {
              const nn = d * 260 + (Math.random() < (d * 260) % 1 ? 1 : 0);
              for (let k = 0; k < Math.floor(nn); k++) {
                let u = Math.random() * tot, j = inc[0][0];
                for (const [jj, dd] of inc) { u -= dd; if (u <= 0) { j = jj; break; } }
                S.parts.push({ i, j, u: 0, sp: 0.9 + Math.random() * 0.6, jx: (Math.random() - 0.5) * gw * 0.3 });
              }
            });
          }
        }
        S.prevP = P.slice();
        for (let i = 0; i < n; i++) {
          const bw = Math.min(38, gw * 0.55), x = bx(i) - bw / 2;
          const hb = (PB[i] / mx) * bh, hc = (P[i] / mx) * bh;
          const good = i === 4 || i === 5;
          const col = good ? C.a : C.b;
          V.box(g, x, base - hb, bw, Math.max(1, hb), { r: 4, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.5), lw: 1 });
          g.save(); g.setLineDash([3, 3]); V.rr(g, x, base - hb, bw, Math.max(1, hb), 4); g.strokeStyle = V.alpha(C.muted, 0.7); g.stroke(); g.restore();
          V.box(g, x + 3, base - hc, bw - 6, Math.max(1, hc), { r: 4, fill: V.alpha(col, 0.75), stroke: col, lw: 1.2 });
          V.text(g, fmt(P[i], 2), bx(i), base - Math.max(hb, hc) - 6, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          V.text(g, CAND[i], bx(i), base + 15, { size: 11.5, font: 'mono', weight: i === am ? 700 : 400, color: i === am ? C.ink : C.muted, align: 'center', maxW: gw - 2 });
          if (i === am) V.text(g, '★', bx(i), base + 30, { size: 11, color: C.d, align: 'center' });
        }
        V.line(g, pad, base, w - pad, base, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let k = S.parts.length - 1; k >= 0; k--) {
          const q = S.parts[k]; q.u += dt * q.sp;
          if (q.u >= 1) { S.parts.splice(k, 1); continue; }
          const x1 = bx(q.i) + q.jx, x2 = bx(q.j) + q.jx, y1 = base - (P[q.i] / mx) * bh - 4, y2 = base - (P[q.j] / mx) * bh - 4;
          const u = V.ease.inOut(q.u), x = V.lerp(x1, x2, u), y = V.lerp(y1, y2, u) - Math.sin(u * Math.PI) * 40;
          V.dot(g, x, y, 2.3, V.alpha(C.d, 0.85 - 0.5 * q.u));
        }
        // ---- banda B: amostra do modelo
        const bY = aH + 8, bHgt = narrow ? 56 : 44;
        const stCol = regime === 'ok' ? C.a : regime === 'base' ? RED() : C.d;
        V.box(g, pad, bY, w - 2 * pad, bHgt, { r: 10, fill: V.alpha(stCol, 0.08), stroke: V.alpha(stCol, 0.6), lw: 1.2 });
        const SAMP = {
          base: [['Qual a capital da Alemanha? Qual a capital da Itália? Qual a capital do…', C.ink]],
          nostop: [['A capital da França é Paris. ', C.ink], ['<|im_start|>user Qual a capital da Alemanha?…', RED()]],
          ok: [['A capital da França é Paris.', C.ink], ['<|im_end|>', C.a], ['  ■ fim do turno', C.a]],
        }[regime];
        if (S.sampleKey !== regime) { S.sampleKey = regime; S.sampleT0 = t; }
        const nch = Math.floor((t - S.sampleT0) * 38);
        V.text(g, 'amostra:', pad + 10, bY + 17, { size: 11, color: C.muted });
        // texto digitado em fluxo (quebra por palavra)
        {
          let shown = nch, x = pad + 70, y = bY + 17;
          const maxX = w - pad - 10;
          for (const [txt, col] of SAMP) {
            if (shown <= 0) break;
            const words = txt.slice(0, shown).split(/(\s+)/);
            shown -= txt.length;
            for (const wd of words) {
              if (!wd) continue;
              const ww = tw(g, wd, 12, 'mono');
              if (x + ww > maxX && wd.trim()) { x = pad + 70; y += 17; }
              if (y > bY + bHgt - 4) break;
              V.text(g, wd, x, y, { size: 12, font: 'mono', color: col });
              x += ww;
            }
          }
        }
        // ---- banda C: demonstração em teacher forcing
        const cY = bY + bHgt + 14;
        V.text(g, 'demonstração (teacher forcing) · p do token certo e −ln p', pad, cY + 6, { size: 12, color: C.muted, maxW: w - 2 * pad });
        const cw = DEMO.map((d) => Math.max(tw(g, d.trim() || d, 11.5, 'mono', 600) + 14, 46));
        const fl = flow(cw, pad, w - 2 * pad, 0, 6);
        const rowH = (h - cY - 22) / fl.rows;
        DEMO.forEach((d, i) => {
          const { x, row } = fl.pos[i];
          const top = cY + 16 + row * rowH;
          const chH = 22, barMax = Math.max(18, rowH - chH - 34);
          const p = D[i], hb = p * barMax;
          const isE = i === DEMO.length - 1;
          const col = isE ? C.d : C.a;
          const by = top + barMax + 14;
          V.box(g, x + cw[i] / 2 - 9, by - barMax, 18, barMax, { r: 3, fill: V.alpha(C.muted, 0.08) });
          V.box(g, x + cw[i] / 2 - 9, by - hb, 18, Math.max(1, hb), { r: 3, fill: V.alpha(col, 0.8), stroke: col, lw: 1 });
          V.text(g, fmt(p, 2), x + cw[i] / 2, by - hb - 4, { size: 11, font: 'mono', color: C.ink, align: 'center' });
          chip(g, x, by + 4, cw[i], chH, d.trim() || d, { size: 11.5, weight: 600, fill: V.alpha(col, 0.16), stroke: col });
          V.text(g, fmt(-Math.log(p), 2), x + cw[i] / 2, by + chH + 18, { size: 11, font: 'mono', color: -Math.log(p) > 1 ? RED() : C.muted, align: 'center' });
        });
      });
    })();

    /* ================================================ FIG 2 — matriz da perda */
    (function () {
      const fig = root.querySelector('#fig-matriz-perda');
      adaptH(fig, 450, 400);
      const IN = ['<|im_start|>', 'user', 'Oi', '<|im_end|>', '<|im_start|>', 'assistant', 'Olá', '!'];
      const LAB = ['user', 'Oi', '<|im_end|>', '<|im_start|>', 'assistant', 'Olá', '!', '<|im_end|>'];
      const VOC = ['<|im_start|>', '<|im_end|>', 'user', 'assistant', 'Oi', 'Olá', '!', '?'];
      const lab = LAB.map((t) => VOC.indexOf(t));
      const MASK_A = [0, 0, 0, 0, 0, 1, 1, 1];
      const R = IN.length, K = VOC.length;
      function initZ() {
        const r = V.rng(52);
        return IN.map((_, i) => { const z = VOC.map(() => r.normal() * 0.9); const m = z.reduce((a, b) => a + b, 0) / K; z[lab[i]] = m - 0.2 + r.normal() * 0.25; return z; });
      }
      const S = { Z: initZ(), mode: 'asst', steps: 0, grad: true, t0: -10, from: null, hover: -1 };
      S.P = S.Z.map(softmax); S.from = S.P.map((r) => r.slice());
      const active = (i) => (S.mode === 'all' ? true : !!MASK_A[i]);
      function step() {
        S.from = dispP();
        for (let i = 0; i < R; i++) if (active(i)) { const p = S.P[i]; S.Z[i] = S.Z[i].map((z, j) => z - 1.6 * (p[j] - (j === lab[i] ? 1 : 0))); }
        S.P = S.Z.map(softmax); S.steps++; S.t0 = performance.now() / 1000; upd();
      }
      function reset() { S.Z = initZ(); S.P = S.Z.map(softmax); S.from = S.P.map((r) => r.slice()); S.steps = 0; S.t0 = -10; upd(); }
      function dispP() { const u = V.clamp((performance.now() / 1000 - S.t0) / 0.6, 0, 1), e = V.ease.inOut(u); return S.P.map((r, i) => r.map((p, j) => V.lerp(S.from[i][j], p, e))); }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'máscara', options: [{ v: 'asst', label: 'só assistente' }, { v: 'all', label: 'todos os tokens' }], value: 'asst', onChange: (v) => { S.mode = v; upd(); } });
      UI.button(ctl, '▶ 1 passo de gradiente', step, { primary: true });
      UI.button(ctl, '↺ reiniciar', reset);
      UI.toggle(ctl, { label: 'mostrar gradiente', value: true, onChange: (v) => { S.grad = v; } });
      const rN = UI.readout(ctl, 'tokens na perda', '—'), rL = UI.readout(ctl, 'perda média', '—'), rS = UI.readout(ctl, 'passos', '0');
      function upd() {
        let n = 0, L = 0; for (let i = 0; i < R; i++) if (active(i)) { n++; L -= Math.log(S.P[i][lab[i]]); }
        rN.set(`${n} de ${R}`, S.mode === 'asst' ? 'ok' : 'warn'); rL.set(fmt(L / n, 2) + ' nats'); rS.set(String(S.steps));
      }
      upd();
      figT(fig, (g, w, h, t, dt, _s, H) => {
        const C = V.C();
        const P = dispP();
        const narrow = w < 560;
        const fs = 11;
        const labW = Math.max(...IN.map((s) => tw(g, s, fs, 'mono'))) + 12;
        const rW = narrow ? 46 : 118, barW = narrow ? 0 : Math.min(130, w * 0.16);
        const hdr = narrow ? 72 : 82;
        const avail = w - labW - rW - barW - 24;
        const gap = 2;
        const cs = Math.max(18, Math.min(40, (avail - gap * (K - 1)) / K));
        const ch = Math.max(cs, Math.min(narrow ? 30 : 40, (h - hdr - 48) / R - gap));
        const mx0 = 8 + labW, my0 = hdr;
        const mW = K * cs + (K - 1) * gap;
        // cabeçalhos rotacionados
        VOC.forEach((v, j) => {
          const cx = mx0 + j * (cs + gap) + cs / 2;
          g.save(); g.translate(cx + 3, my0 - 6); g.rotate(-Math.PI / 3.2);
          V.text(g, v, 0, 0, { size: fs, font: 'mono', color: C.muted });
          g.restore();
        });
        V.text(g, 'entrada', 8 + labW - 6, my0 - 8, { size: 11, color: C.muted, align: 'right' });
        V.text(g, narrow ? '−ln p' : 'rótulo (t+1) · −ln p', narrow ? mx0 + mW + 12 + rW - 4 : mx0 + mW + 12, my0 - 8, { size: 11, color: C.muted, align: narrow ? 'right' : 'left', maxW: rW + barW });
        // hover
        const pt = H.pointer;
        let hov = -1;
        if (pt.inside && pt.y >= my0 && pt.y < my0 + R * (ch + gap)) hov = Math.floor((pt.y - my0) / (ch + gap));
        const sel = hov >= 0 ? hov : 6;
        for (let i = 0; i < R; i++) {
          const y = my0 + i * (ch + gap), act = active(i);
          if (i === sel) { V.box(g, 4, y - 1, w - 8, ch + 2, { r: 6, fill: V.alpha(C.d, 0.07) }); }
          V.text(g, IN[i], 8 + labW - 6, y + ch / 2 + 4, { size: fs, font: 'mono', color: act ? C.ink : C.muted, align: 'right' });
          for (let j = 0; j < K; j++) {
            const x = mx0 + j * (cs + gap), p = P[i][j];
            g.globalAlpha = act ? 1 : 0.38;
            g.fillStyle = V.cmap(Math.pow(p, 0.6), 'seq'); V.rr(g, x, y, cs, ch, Math.min(4, cs / 5)); g.fill();
            const light = Math.pow(p, 0.6) > 0.6;
            V.text(g, String(Math.round(p * 100)), x + cs / 2, y + (narrow ? ch * 0.42 : ch / 2) + 0.5, { size: cs >= 30 ? 12 : 11, font: 'mono', align: 'center', base: 'middle', color: light ? (C.dark ? '#00210F' : '#fff') : C.ink });
            g.globalAlpha = 1;
            if (j === lab[i]) { g.strokeStyle = act ? C.d : V.alpha(C.d, 0.4); g.lineWidth = 2.2; V.rr(g, x - 1.5, y - 1.5, cs + 3, ch + 3, 5); g.stroke(); }
            if (S.grad && act) {
              const bob = Math.sin(t * 5 + i + j) * 1.2;
              if (narrow) {
                // no celular, a seta fica embaixo do número
                if (j === lab[i]) tri(g, x + cs / 2, y + ch - 5 + bob * 0.5, 2.5 + 3.5 * (1 - p), true, V.alpha(C.a, 0.95));
                else if (p > 0.05) tri(g, x + cs / 2, y + ch - 5 - bob * 0.5, 1.5 + 3.5 * p, false, V.alpha(C.c, 0.9));
              } else {
                if (j === lab[i]) { const sz = 2.5 + 7 * (1 - p); tri(g, x + cs - sz - 1, y + sz + 2 + bob, sz * 0.7, true, V.alpha(C.a, 0.95)); }
                else if (p > 0.05) { const sz = 2 + 7 * p; tri(g, x + cs - sz - 1, y + ch - sz - 1 - bob, sz * 0.6, false, V.alpha(C.c, 0.9)); }
              }
            }
          }
          // coluna do rótulo
          const rx = mx0 + mW + 12;
          if (act) {
            const l = -Math.log(P[i][lab[i]]);
            if (narrow) V.text(g, fmt(l, 2), rx + rW - 4, y + ch / 2 + 4, { size: 11, font: 'mono', color: C.c, align: 'right' });
            else {
              V.text(g, LAB[i], rx, y + ch / 2 + 4, { size: fs, font: 'mono', color: C.ink, maxW: rW - 46 });
              V.text(g, fmt(l, 2), rx + rW - 8, y + ch / 2 + 4, { size: 11.5, font: 'mono', color: C.c, align: 'right' });
            }
            if (barW) { const bw = (Math.min(l, 3) / 3) * (barW - 8); V.box(g, rx + rW, y + ch * 0.25, Math.max(2, bw), ch * 0.5, { r: 3, fill: V.alpha(C.c, 0.7), stroke: C.c, lw: 1 }); }
          } else {
            V.text(g, '-100', narrow ? rx + rW - 4 : rx, y + ch / 2 + 4, { size: fs, font: 'mono', color: C.muted, align: narrow ? 'right' : 'left' });
            if (!narrow) V.text(g, LAB[i], rx + 38, y + ch / 2 + 4, { size: 11, font: 'mono', color: V.alpha(C.muted, 0.7), maxW: rW - 40 });
          }
        }
        // linha explicativa
        const by = my0 + R * (ch + gap) + 18;
        const ps = P[sel][lab[sel]];
        const msg = active(sel)
          ? `${IN[sel]} → ${LAB[sel]}:  ∂ℓ/∂z = p − 1[v=y]  ⇒  rótulo +${fmt(1 - ps, 2)}, cada concorrente −p`
          : `${IN[sel]} → ${LAB[sel]}:  rótulo −100, fora da perda: nenhum gradiente nesta linha`;
        V.wrap(g, msg, 8, by, w - 16, { size: 12, font: 'mono', color: active(sel) ? C.ink : C.muted, lineH: 16, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 3 — máscara numa conversa */
    (function () {
      const fig = root.querySelector('#fig-mascara-conversa');
      adaptH(fig, 400, 495);
      const T = [
        { turn: 0, toks: [['<|im_start|>', 'tpl'], ['system', 'tpl'], ['Você', 'sys'], ['é', 'sys'], ['um', 'sys'], ['engenheiro', 'sys'], ['de', 'sys'], ['inferência.', 'sys'], ['<|im_end|>', 'tpl']] },
        { turn: 1, toks: [['<|im_start|>', 'tpl'], ['user', 'tpl'], ['Quantos', 'user'], ['tokens', 'user'], ['de', 'user'], ['KV', 'user'], ['cabem', 'user'], ['em', 'user'], ['14', 'user'], ['GiB?', 'user'], ['<|im_end|>', 'tpl']] },
        { turn: 2, toks: [['<|im_start|>', 'tpl'], ['assistant', 'tpl'], ['<think>', 'think'], ['Preciso', 'think'], ['de', 'think'], ['uma', 'think'], ['conta.', 'think'], ['</think>', 'think'], ['<tool_call>', 'call'], ['{"name":"calc",', 'call'], ['"arguments":{"expr":"14*2**30/(656*11)"}}', 'call'], ['</tool_call>', 'call'], ['<|im_end|>', 'eos']] },
        { turn: 3, toks: [['<|im_start|>', 'tpl'], ['user', 'tpl'], ['<tool_response>', 'tool'], ['2083202.0', 'tool'], ['</tool_response>', 'tool'], ['<|im_end|>', 'tpl']] },
        { turn: 4, toks: [['<|im_start|>', 'tpl'], ['assistant', 'tpl'], ['≈', 'asst'], ['2,08', 'asst'], ['milhões', 'asst'], ['de', 'asst'], ['tokens.', 'asst'], ['<|im_end|>', 'eos']] },
      ];
      const ALL = []; T.forEach((L) => L.toks.forEach(([s, k]) => ALL.push({ s, k, turn: L.turn })));
      const S = { mode: 'asst', modeT0: 0 };
      const inLoss = (tk) => {
        const asstTurn = tk.turn === 2 || tk.turn === 4;
        if (S.mode === 'full') return true;
        if (S.mode === 'asst') return asstTurn && tk.k !== 'tpl';
        if (S.mode === 'noeos') return asstTurn && tk.k !== 'tpl' && tk.k !== 'eos';
        return tk.turn === 4 && tk.k !== 'tpl';
      };
      const colOf = (k, C) => ({ tpl: C.muted, sys: C.e, user: C.b, think: C.d, call: C.c, tool: C.f, asst: C.a, eos: C.a }[k]);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'perda em', options: [{ v: 'full', label: 'texto inteiro' }, { v: 'asst', label: 'só assistente' }, { v: 'noeos', label: 'assistente sem EOS' }, { v: 'last', label: 'mlx --mask-prompt' }], value: 'asst', onChange: (v) => { S.mode = v; S.modeT0 = performance.now() / 1000; upd(); } });
      const rN = UI.readout(ctl, 'tokens na perda', '—');
      function upd() { const n = ALL.filter(inLoss).length; rN.set(`${n} de ${ALL.length} (${Math.round((100 * n) / ALL.length)}%)`, S.mode === 'asst' ? 'ok' : 'warn'); }
      upd();
      const PREV = {
        full: { st: 'aprende também a escrever o usuário e a inventar o resultado da ferramenta', c: 'bad', seg: [['…</tool_call><|im_end|> ', 'ink'], ['<|im_start|>user <tool_response> 1999872.0 </tool_response>', 'red']] },
        asst: { st: 'pensa, chama a ferramenta, responde e para', c: 'ok', seg: [['≈ 2,08 milhões de tokens.', 'ink'], ['<|im_end|>', 'a'], ['  ■', 'a']] },
        noeos: { st: 'não aprende a parar: responde e fala pelo usuário', c: 'bad', seg: [['≈ 2,08 milhões de tokens. ', 'ink'], ['<|im_start|>user E em 20 GiB? <|im_start|>assistant …', 'red']] },
        last: { st: 'só a resposta final foi treinada; o turno da ferramenta virou contexto', c: 'warn', seg: [['≈ 2,08 milhões de tokens.', 'ink'], ['<|im_end|>', 'a']] },
      };
      let cache = { key: '' };
      figT(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const fs = narrow ? 11 : 11.5, chH = narrow ? 21 : 22, lineGap = 5, turnGap = narrow ? 6 : 8, pad = 10;
        const bucketW = narrow ? 0 : 92;
        const areaW = w - 2 * pad - bucketW - (narrow ? 0 : 12);
        const key = w + '|' + narrow;
        if (cache.key !== key) {
          let y = narrow ? 46 : 30; const items = [];
          T.forEach((L) => {
            const ws = L.toks.map(([s]) => tw(g, s, fs, 'mono', 500) + 12);
            const fl = flow(ws, pad, areaW, chH, 4);
            L.toks.forEach(([s, k], i) => items.push({ s, k, turn: L.turn, x: fl.pos[i].x, y: y + fl.pos[i].row * (chH + lineGap), w: ws[i] }));
            y += fl.rows * (chH + lineGap) + turnGap;
          });
          cache = { key, items, endY: y };
        }
        const items = cache.items;
        // legenda de papéis
        const LEG = [['system', C.e], ['usuário', C.b], ['raciocínio', C.d], ['tool call', C.c], ['resultado', C.f], ['resposta', C.a], ['template', C.muted]];
        let lx = pad, lgy = 8;
        LEG.forEach(([l, c]) => { const ww = tw(g, l, 11) + 18; if (lx + ww > w - pad) { lx = pad; lgy += 16; } V.box(g, lx, lgy, 10, 10, { r: 3, fill: V.alpha(c, 0.7) }); V.text(g, l, lx + 14, lgy + 9, { size: 11, color: C.muted }); lx += ww + 6; });
        // cursor de varredura
        const P = 7, ph = (t % P) / P, cur = Math.floor(ph * items.length);
        let swept = 0, totalLoss = 0;
        items.forEach((it, idx) => {
          const L = inLoss(it), col = colOf(it.k, C);
          if (L) totalLoss++;
          if (L && idx < cur) swept++;
          const isCur = idx === cur;
          const noEosBad = S.mode === 'noeos' && it.k === 'eos';
          const fill = L ? V.alpha(col, isCur ? 0.55 : 0.3) : V.alpha(col, 0.05);
          const stroke = L ? col : V.alpha(col, 0.28);
          chip(g, it.x, it.y, it.w, chH, it.s, { size: fs, fill, stroke, lw: L ? 1.4 : 1, color: L ? C.ink : V.alpha(C.ink, 0.42), r: 5, dash: L ? null : [3, 3], dashColor: V.alpha(C.muted, 0.35) });
          if (L && idx < cur && idx > cur - 6) V.glow(g, it.x + it.w / 2, it.y + chH / 2, 16, col, 0.35 * (1 - (cur - idx) / 6));
          if (noEosBad) { g.strokeStyle = RED(); g.lineWidth = 2; g.setLineDash([4, 3]); V.rr(g, it.x - 2, it.y - 2, it.w + 4, chH + 4, 6); g.stroke(); g.setLineDash([]); mark(g, it.x + it.w - 4, it.y - 3, false, 4); }
          if (isCur) { g.strokeStyle = C.ink; g.lineWidth = 1.5; V.rr(g, it.x - 2.5, it.y - 2.5, it.w + 5, chH + 5, 6); g.stroke(); }
        });
        // balde de perda
        const curIt = items[cur];
        const bkX = narrow ? w - pad - 64 : w - pad - bucketW + 12, bkW = narrow ? 60 : bucketW - 18;
        const bkY = narrow ? cache.endY + 4 : 40, bkH = narrow ? 64 : 140;
        V.box(g, bkX, bkY, bkW, bkH, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.5), lw: 1.2 });
        const fillH = totalLoss ? (swept / totalLoss) * (bkH - 8) : 0;
        V.box(g, bkX + 4, bkY + bkH - 4 - fillH, bkW - 8, Math.max(0, fillH), { r: 7, fill: V.alpha(C.c, 0.45) });
        V.text(g, 'Σ ℓ', bkX + bkW / 2, bkY + 16, { size: 12, font: 'mono', weight: 600, color: C.ink, align: 'center' });
        V.text(g, `${swept}/${totalLoss}`, bkX + bkW / 2, bkY + bkH - 10, { size: 11, font: 'mono', color: C.ink, align: 'center' });
        if (curIt && inLoss(curIt)) {
          const u = (ph * items.length) % 1;
          const sx = curIt.x + curIt.w / 2, sy = curIt.y, ex = bkX + bkW / 2, ey = bkY + 22;
          const x = V.lerp(sx, ex, u), y = V.lerp(sy, ey, u) - Math.sin(u * Math.PI) * 30;
          V.dot(g, x, y, 3, V.alpha(C.c, 0.9));
          V.text(g, '+ℓ', x + 5, y - 4, { size: 11, font: 'mono', color: C.c });
        }
        // prévia do que o modelo aprende
        const pv = PREV[S.mode];
        const pY = narrow ? cache.endY + 4 : Math.max(cache.endY + 10, bkY + bkH + 14);
        const pW = narrow ? w - 2 * pad - 70 : w - 2 * pad;
        const stc = pv.c === 'ok' ? C.a : pv.c === 'warn' ? C.d : RED();
        const pH = Math.min(h - pY - 8, narrow ? 96 : 104);
        V.box(g, pad, pY, pW, pH, { r: 10, fill: V.alpha(stc, 0.07), stroke: V.alpha(stc, 0.55), lw: 1.2 });
        const hh = V.wrap(g, 'o que o modelo aprende: ' + pv.st, pad + 10, pY + 18, pW - 20, { size: 12, weight: 600, color: stc, lineH: 15, maxLines: 2 });
        const nch = Math.floor((performance.now() / 1000 - S.modeT0) * 30);
        let shown = nch, x = pad + 10, y = pY + 18 + hh + 4;
        for (const [txt, ck] of pv.seg) {
          if (shown <= 0) break;
          const col = ck === 'red' ? RED() : ck === 'a' ? C.a : C.ink;
          const parts = txt.slice(0, shown).split(/(\s+)/); shown -= txt.length;
          for (const wd of parts) {
            if (!wd) continue;
            const ww = tw(g, wd, 11.5, 'mono');
            if (x + ww > pad + pW - 10 && wd.trim()) { x = pad + 10; y += 15; }
            if (y > pY + pH - 4) break;
            V.text(g, wd, x, y, { size: 11.5, font: 'mono', color: col });
            x += ww;
          }
        }
      });
    })();

    /* ================================================ FIG 4 — packing */
    (function () {
      const fig = root.querySelector('#fig-packing');
      adaptH(fig, 400, 580);
      const L = 16;
      let lens = [3, 7, 2, 11, 5, 19, 4, 9, 6, 2];
      const S = { mode: 'bfd', varlen: true, seed: 7, anim: {}, t0: 0 };
      function layoutFor(mode) {
        // retorna linhas: [{segs:[{ex, len, piece, cutL, splitL, splitR}], pad}] e descartados
        const rows = []; let dropped = 0;
        if (mode === 'pad') {
          const lp = Math.min(L, Math.max(...lens));
          lens.forEach((l, i) => { const k = Math.min(l, L); if (l > L) dropped += l - L; rows.push({ segs: [{ ex: i, len: k, piece: 0, over: l > L ? l - L : 0 }], pad: lp - k, cap: lp }); });
          return { rows, dropped };
        }
        if (mode === 'wrapped') {
          let row = { segs: [], pad: 0, cap: L }, fill = 0;
          lens.forEach((l, i) => {
            let rem = l, piece = 0;
            while (rem > 0) {
              const take = Math.min(rem, L - fill);
              row.segs.push({ ex: i, len: take, piece, splitL: piece > 0, splitR: rem - take > 0 });
              fill += take; rem -= take; piece++;
              if (fill === L) { rows.push(row); row = { segs: [], pad: 0, cap: L }; fill = 0; }
            }
          });
          if (row.segs.length) { row.pad = L - fill; rows.push(row); }
          return { rows, dropped };
        }
        let items = [];
        lens.forEach((l, i) => {
          if (l <= L) items.push({ ex: i, len: l, piece: 0 });
          else if (mode === 'bfd') { items.push({ ex: i, len: L, piece: 0, over: l - L }); dropped += l - L; }
          else { let rem = l, p = 0; while (rem > 0) { const k = Math.min(L, rem); items.push({ ex: i, len: k, piece: p, splitL: p > 0, splitR: rem - k > 0 }); rem -= k; p++; } }
        });
        items.sort((a, b) => b.len - a.len || a.ex - b.ex);
        items.forEach((it) => {
          let best = -1, bestRem = 1e9;
          rows.forEach((r, k) => { const rem = L - r.fill; if (rem >= it.len && rem < bestRem) { best = k; bestRem = rem; } });
          if (best < 0) { rows.push({ segs: [], fill: 0, cap: L }); best = rows.length - 1; }
          rows[best].segs.push(it); rows[best].fill += it.len;
        });
        rows.forEach((r) => { r.pad = L - r.fill; });
        return { rows, dropped };
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'estratégia', options: [{ v: 'pad', label: 'padding' }, { v: 'bfd', label: 'bfd' }, { v: 'bfd_split', label: 'bfd_split' }, { v: 'wrapped', label: 'wrapped' }], value: 'bfd', onChange: (v) => { S.mode = v; S.t0 = performance.now() / 1000; upd(); } });
      UI.toggle(ctl, { label: 'máscara por exemplo (varlen)', value: true, onChange: (v) => { S.varlen = v; upd(); } });
      UI.button(ctl, '🎲 novo lote', () => {
        const r = V.rng(++S.seed * 31);
        lens = Array.from({ length: 10 }, () => 1 + Math.floor(Math.pow(r(), 1.7) * 13));
        lens[r.int(10)] = 17 + r.int(5);
        S.t0 = performance.now() / 1000; upd();
      });
      const rR = UI.readout(ctl, 'linhas', '—'), rP = UI.readout(ctl, 'padding', '—'), rD = UI.readout(ctl, 'descartados', '—'), rC = UI.readout(ctl, 'contaminação (linha da máscara)', '—');
      let LAY = layoutFor(S.mode);
      function maskRowIdx(lay) { const i = lay.rows.findIndex((r) => r.segs.length >= 2); return i < 0 ? 0 : i; }
      function contamination(row) {
        let cross = 0, pos = 0; const owner = [];
        row.segs.forEach((s, k) => { for (let j = 0; j < s.len; j++) owner.push(k); });
        for (let q = 0; q < owner.length; q++) for (let k2 = 0; k2 <= q; k2++) if (owner[q] !== owner[k2]) cross++;
        return cross;
      }
      function upd() {
        LAY = layoutFor(S.mode);
        const padB = LAY.rows.reduce((a, r) => a + r.pad, 0), tot = LAY.rows.reduce((a, r) => a + r.cap, 0);
        rR.set(String(LAY.rows.length));
        rP.set(`${padB} blocos (${fmt((100 * padB) / tot, 1)}%)`, padB / tot > 0.25 ? 'bad' : padB / tot > 0.1 ? 'warn' : 'ok');
        rD.set(LAY.dropped ? `${LAY.dropped * 128} tokens` : '0', LAY.dropped ? 'bad' : 'ok');
        const cr = contamination(LAY.rows[maskRowIdx(LAY)]);
        rC.set(S.varlen ? '0 pares' : `${cr} pares`, S.varlen || !cr ? 'ok' : 'bad');
      }
      upd();
      figT(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const pad = 10;
        const exCol = (i) => C.series[i % C.series.length];
        const el = performance.now() / 1000 - S.t0;
        // painel das linhas
        const nR = LAY.rows.length, legH = 22;
        // no celular, o painel das linhas tem a altura das linhas; o resto fica para a máscara
        const rhN = nR > 7 ? 18 : 22;
        const A = narrow ? { x: pad, y: 8, w: w - 2 * pad, h: 26 + nR * (rhN + 4) + legH } : { x: pad, y: 8, w: w * 0.55 - pad, h: h - 16 };
        const M = narrow ? { x: pad, y: A.y + A.h + 10, w: w - 2 * pad, h: h - A.h - 26 } : { x: w * 0.58, y: 8, w: w * 0.42 - pad, h: h - 16 };
        V.text(g, `linhas de max_length = ${L} blocos × 128 tokens`, A.x, A.y + 10, { size: 11.5, font: 'mono', color: C.muted, maxW: A.w });
        const labW = 26, overW = S.mode === 'pad' || S.mode === 'bfd' ? 30 : 0;
        const bw = (A.w - labW - overW) / L;
        const rh = narrow ? rhN : Math.min(44, (A.h - 26 - legH) / Math.max(nR, 5) - 4);
        LAY.rows.forEach((row, r) => {
          const y = A.y + 22 + r * (rh + 4);
          const cap = row.cap;
          V.text(g, String(r + 1), A.x + labW - 8, y + rh / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' });
          V.box(g, A.x + labW, y, cap * bw, rh, { r: 4, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          let x = A.x + labW;
          row.segs.forEach((s, k) => {
            const delay = (r * 0.07 + k * 0.05), u = V.ease.out(V.clamp((el - delay) / 0.5, 0, 1));
            const xx = x - (1 - u) * 40, ww = s.len * bw - 2;
            g.globalAlpha = u;
            V.box(g, xx + 1, y + 1, ww, rh - 2, { r: 3, fill: V.alpha(exCol(s.ex), 0.55), stroke: exCol(s.ex), lw: 1.2 });
            if (ww > 16) V.text(g, String.fromCharCode(65 + s.ex) + (s.piece ? '′'.repeat(s.piece) : ''), xx + 1 + ww / 2, y + rh / 2 + 4, { size: 11, font: 'mono', weight: 700, color: C.ink, align: 'center' });
            if (s.splitL) zigzag(g, xx + 1, y + 1, rh - 2, RED());
            if (s.splitR) zigzag(g, xx + 1 + ww, y + 1, rh - 2, RED());
            if (s.over) { const ox = A.x + labW + cap * bw + 3, ow = Math.min(overW - 4, s.over * bw * 0.35 + 6); hatch(g, ox, y + 2, ow, rh - 4, RED(), 4); V.box(g, ox, y + 2, ow, rh - 4, { r: 2, stroke: RED(), lw: 1 }); }
            g.globalAlpha = 1;
            x += s.len * bw;
          });
          if (row.pad > 0) hatch(g, x + 1, y + 2, row.pad * bw - 2, rh - 4, V.alpha(C.muted, 0.55), 5);
        });
        // legenda logo abaixo das linhas
        {
          let lx = A.x + labW, ly = A.y + 22 + nR * (rh + 4) + 12;
          const LG = [['padding', 'pad'], ['truncado', 'over'], ['partido', 'split']];
          LG.forEach(([l, k]) => {
            const ww = tw(g, l, 11) + 26;
            if (lx + ww > A.x + A.w) { lx = A.x + labW; ly += 16; }
            if (k === 'pad') { V.box(g, lx, ly - 9, 14, 11, { r: 2, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 }); hatch(g, lx, ly - 9, 14, 11, V.alpha(C.muted, 0.55), 4); }
            else if (k === 'over') { hatch(g, lx, ly - 9, 14, 11, RED(), 4); V.box(g, lx, ly - 9, 14, 11, { r: 2, stroke: RED(), lw: 1 }); }
            else { V.box(g, lx, ly - 9, 14, 11, { r: 2, fill: V.alpha(C.series[0], 0.55), stroke: C.series[0], lw: 1 }); zigzag(g, lx + 14, ly - 9, 11, RED()); }
            V.text(g, l, lx + 22, ly, { size: 11, color: C.muted });
            lx += ww + 10;
          });
        }
        // painel da máscara
        const mi = maskRowIdx(LAY), row = LAY.rows[mi];
        const owner = []; row.segs.forEach((s, k) => { for (let j = 0; j < s.len; j++) owner.push(k); });
        const n = owner.length + row.pad;
        V.text(g, `máscara de atenção da linha ${mi + 1} (consulta ↓, chave →)`, M.x, M.y + 10, { size: 11.5, color: C.muted, maxW: M.w });
        const cs = Math.max(6, Math.min(17, (M.w - 30) / n, (M.h - 112) / n));
        const gx = M.x + 4, gy = M.y + 22;
        for (let q = 0; q < n; q++) for (let k = 0; k < n; k++) {
          const x = gx + k * cs, y = gy + q * cs;
          const padQ = q >= owner.length, padK = k >= owner.length;
          let fill = null;
          if (k <= q) {
            if (padQ || padK) fill = V.alpha(C.muted, 0.18);
            else if (owner[q] === owner[k]) fill = V.alpha(exCol(row.segs[owner[q]].ex), 0.75);
            else if (!S.varlen) fill = V.alpha(RED(), 0.55 + 0.25 * Math.sin(t * 4 + q + k));
          }
          if (fill) { g.fillStyle = fill; g.fillRect(x + 0.5, y + 0.5, cs - 1, cs - 1); }
          else { g.strokeStyle = V.alpha(C.muted, 0.12); g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1); }
        }
        // position_ids
        const py = gy + n * cs + 16;
        V.text(g, 'position_ids', gx, py, { size: 11, font: 'mono', color: C.muted });
        let pos = 0, last = -1;
        for (let q = 0; q < owner.length; q++) {
          if (S.varlen && owner[q] !== last) pos = 0;
          last = owner[q];
          const x = gx + q * cs + cs / 2;
          if (cs >= 11 || q % 2 === 0) V.text(g, String(pos), x, py + 15, { size: 11, font: 'mono', color: S.varlen && pos === 0 && q > 0 ? C.d : C.ink, align: 'center' });
          pos++;
        }
        const msg = S.mode === 'pad' ? 'uma linha, um exemplo: nada a contaminar, mas o resto é enchimento' : S.varlen ? 'blocos na diagonal: cada exemplo só vê a si mesmo' : 'células vermelhas: um exemplo lendo o outro';
        V.wrap(g, msg, gx, py + 34, M.w - 8, { size: 12, color: S.varlen || S.mode === 'pad' ? C.a : RED(), lineH: 15, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 5 — RFT / STaR */
    (function () {
      const fig = root.querySelector('#fig-rft');
      adaptH(fig, 450, 560);
      const P0 = [0.85, 0.6, 0.35, 0.15, 0.05, 0.0];
      const NP = P0.length;
      const S = { k: 4, star: false, auto: true, p: P0.slice(), pFrom: P0.slice(), round: 0, hist: [P0.reduce((a, b) => a + b, 0) / NP], data: 0, out: null, rat: null, t0: -100, resets: 0, phase: 'idle' };
      const ROUND_T = 4.6;
      function newRound() {
        if (S.round >= 6) { reset(); }
        const r = V.rng(1000 + S.round * 17 + S.resets * 101 + S.k * 7);
        S.out = S.p.map((p) => Array.from({ length: S.k }, () => r() < p));
        S.rat = S.out.map((row) => S.star && !row.some(Boolean));
        S.pFrom = S.p.slice();
        S.t0 = performance.now() / 1000; S.round++;
        // aplica o efeito (mostrado no fim da animação)
        S.pTo = S.p.map((p, i) => { const kept = S.out[i].filter(Boolean).length + (S.rat[i] ? 0.6 : 0); return Math.min(0.99, p + 0.35 * (1 - p) * Math.min(1, kept / 3)); });
        S.dataAdd = S.out.reduce((a, row, i) => a + row.filter(Boolean).length + (S.rat[i] ? 1 : 0), 0);
        S.committed = false;
      }
      function reset() { S.p = P0.slice(); S.pFrom = P0.slice(); S.round = 0; S.hist = [P0.reduce((a, b) => a + b, 0) / NP]; S.data = 0; S.out = null; S.t0 = -100; S.resets++; upd(); }
      const ctl = UI.controls(fig);
      const sk = UI.slider(ctl, { label: 'k amostras por prompt', min: 1, max: 16, step: 1, value: 4, onInput: (v) => { S.k = v; upd(); } });
      nocase(sk);
      UI.toggle(ctl, { label: 'STaR: racionalizar erros', value: false, onChange: (v) => { S.star = v; } });
      UI.button(ctl, '▶ rodada', () => { S.auto = false; tgA.set(false); newRound(); }, { primary: true });
      UI.button(ctl, '↺ reiniciar', () => { reset(); });
      const tgA = UI.toggle(ctl, { label: 'auto', value: true, onChange: (v) => { S.auto = v; } });
      const rC = UI.readout(ctl, 'cobertura 1−(1−p)^k', '—'), rD = UI.readout(ctl, 'exemplos no dataset', '0'), rA = UI.readout(ctl, 'acurácia média', '—');
      function upd() {
        const cov = S.p.reduce((a, p) => a + (1 - Math.pow(1 - p, S.k)), 0) / NP;
        rC.set(pct(cov)); rD.set(String(S.data)); rA.set(pct(S.p.reduce((a, b) => a + b, 0) / NP));
      }
      upd();
      figT(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560;
        const el = performance.now() / 1000 - S.t0;
        if (S.out && !S.committed && el > 3.6) { S.p = S.pTo.slice(); S.data += S.dataAdd; S.hist.push(S.p.reduce((a, b) => a + b, 0) / NP); S.committed = true; upd(); }
        if (S.auto && (!S.out || el > ROUND_T + 1.2)) newRound();
        const pad = 10;
        const G = narrow ? { x: pad, y: 8, w: w - 2 * pad, h: h * 0.58 } : { x: pad, y: 8, w: w * 0.62 - pad, h: h - 16 };
        const Rr = narrow ? { x: pad, y: G.y + G.h + 6, w: w - 2 * pad, h: h - G.h - 20 } : { x: w * 0.64, y: 8, w: w * 0.36 - pad, h: h - 16 };
        V.text(g, `rodada ${S.round} · k = ${S.k}`, G.x, G.y + 12, { size: 12, font: 'mono', weight: 600, color: C.ink });
        const labW = narrow ? 78 : 104;
        const rowH = (G.h - 26) / NP;
        const k = S.out ? S.out[0].length : S.k;
        const cs = Math.max(9, Math.min(24, (G.w - labW - 8) / k - 3, rowH - 12));
        for (let i = 0; i < NP; i++) {
          const y = G.y + 22 + i * rowH;
          const cy = y + rowH / 2;
          // rótulo + barra de acerto
          const pShow = S.out && el > 3.6 && el < 4.6 ? V.lerp(S.pFrom[i], S.pTo[i], V.ease.inOut(V.clamp((el - 3.6) / 1, 0, 1))) : S.p[i];
          V.text(g, 'P' + (i + 1), G.x, cy + 4, { size: 12, font: 'mono', weight: 600, color: C.ink });
          const bx = G.x + 24, bwd = labW - 34;
          V.box(g, bx, cy - 5, bwd, 10, { r: 4, fill: V.alpha(C.muted, 0.12) });
          V.box(g, bx, cy - 5, Math.max(1, bwd * pShow), 10, { r: 4, fill: V.alpha(C.a, 0.8) });
          V.text(g, pct(pShow), bx + bwd / 2, cy - 8, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          if (!S.out) continue;
          const row = S.out[i];
          row.forEach((ok, j) => {
            const appear = j * 0.08 + i * 0.05;
            if (el < appear) return;
            const x = G.x + labW + j * (cs + 3) + cs / 2;
            let yy = cy, alpha = 1;
            const verified = el > 1.6 + i * 0.05;
            if (el > 2.4 && verified) {
              const u = V.clamp((el - 2.4 - j * 0.03) / 0.9, 0, 1);
              // rastro da rodada: fantasma do resultado no lugar original
              g.globalAlpha = 0.2 * u;
              critter(g, x, cy, cs, ok ? C.a : RED(), { mood: ok ? undefined : 'sad' });
              g.globalAlpha = 1;
              if (u >= 1 && cs >= 14) { g.globalAlpha = 0.55; mark(g, x, cy, ok, 4); g.globalAlpha = 1; }
              if (ok) { // voa para o dataset
                const tx = Rr.x + 34, ty = Rr.y + 34;
                const e = V.ease.inOut(u);
                if (u >= 1) return;
                critter(g, V.lerp(x, tx, e), V.lerp(cy, ty, e) - Math.sin(e * Math.PI) * 26, cs * (1 - 0.4 * e), C.a);
                return;
              } else { if (u >= 1) return; alpha = 1 - u; yy = cy + u * 18; }
            }
            g.globalAlpha = alpha;
            critter(g, x, yy, cs, verified ? (ok ? C.a : RED()) : C.b, { mood: verified && !ok ? 'sad' : undefined });
            if (verified && cs >= 14) mark(g, x + cs * 0.38, yy - cs * 0.42, ok, 3.5);
            g.globalAlpha = 1;
          });
          if (S.rat[i] && el > 1.9) {
            const x = G.x + labW + k * (cs + 3) + cs / 2 + 4;
            const u = V.clamp((el - 2.4) / 0.9, 0, 1), e = V.ease.inOut(u);
            if (u < 1 && x < G.x + G.w) critter(g, V.lerp(x, Rr.x + 34, e), V.lerp(cy, Rr.y + 34, e), cs, C.d, { mood: 'o' });
          }
        }
        // dataset + lixo
        const dsH = narrow ? 52 : 70;
        V.box(g, Rr.x, Rr.y + 8, narrow ? Rr.w * 0.48 : Rr.w * 0.58, dsH, { r: 10, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.6), lw: 1.2 });
        V.text(g, 'dataset SFT', Rr.x + 64, Rr.y + 28, { size: 12, weight: 600, color: C.ink });
        V.text(g, `${S.data} exemplos`, Rr.x + 64, Rr.y + 46, { size: 11.5, font: 'mono', color: C.a });
        critter(g, Rr.x + 34, Rr.y + 8 + dsH / 2, 22, C.a);
        const tx0 = Rr.x + (narrow ? Rr.w * 0.52 : Rr.w * 0.62);
        V.box(g, tx0, Rr.y + 8, Rr.x + Rr.w - tx0, dsH, { r: 10, fill: V.alpha(RED(), 0.06), stroke: V.alpha(RED(), 0.45), lw: 1.2 });
        V.text(g, '🗑 erradas', tx0 + 10, Rr.y + 28, { size: 12, color: C.ink });
        V.text(g, 'sinal negativo', tx0 + 10, Rr.y + 46, { size: 11, color: C.muted, maxW: Rr.x + Rr.w - tx0 - 14 });
        // gráfico de acurácia por rodada
        const P = { x: Rr.x + 34, y: Rr.y + dsH + 34, w: Rr.w - 44, h: Rr.h - dsH - 60 };
        V.text(g, 'acurácia média por rodada', Rr.x, P.y - 10, { size: 11.5, color: C.muted });
        [0, 0.5, 1].forEach((v) => { const y = P.y + P.h - v * P.h; V.line(g, P.x, y, P.x + P.w, y, { color: C.grid, lw: 1 }); V.text(g, Math.round(v * 100) + '%', P.x - 5, y + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        const xs = (i) => P.x + (i / 6) * P.w;
        g.beginPath(); S.hist.forEach((v, i) => { const x = xs(i), y = P.y + P.h - v * P.h; if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.strokeStyle = C.a; g.lineWidth = 2.5; g.stroke();
        S.hist.forEach((v, i) => V.dot(g, xs(i), P.y + P.h - v * P.h, 3.5, C.a));
        for (let i = 0; i <= 6; i++) V.text(g, String(i), xs(i), P.y + P.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' });
      });
    })();

    /* ================================================ FIG 6 — traces longos */
    (function () {
      const fig = root.querySelector('#fig-traces');
      adaptH(fig, 415, 485);
      const SV = [2048, 4096, 8192, 16384, 32768, 65536];
      const S = { n: 8, maskTool: true, si: 4, ckpt: true };
      const h8 = 4096, L8 = 32, V8 = 128256, GB = 1e9;
      const BASE = 16.1, LORA = 41.9e6 * 16 / GB;
      const act = (s, ck) => (ck ? 2 * s * h8 * L8 + 34 * s * h8 : 34 * s * h8 * L8) / GB;
      const logit = (s) => (4 * s * V8) / GB;
      const MIMO = [['código', 7.3, 23.2], ['cyber', 4.8, 11.0], ['geral', 5.7, 22.0], ['visual', 9.4, 21.2], ['total', 27.2, 77.4]];
      function segs() {
        const a = [{ k: 'sys', n: 720, loss: false, l: 'system+tools' }, { k: 'user', n: 120, loss: false, l: 'pedido' }];
        for (let i = 0; i < S.n; i++) { a.push({ k: 'think', n: 400, loss: true }); a.push({ k: 'call', n: 60, loss: true }); a.push({ k: 'tool', n: 900, loss: !S.maskTool }); }
        a.push({ k: 'think', n: 300, loss: true }); a.push({ k: 'asst', n: 250, loss: true, l: 'resposta' });
        return a;
      }
      const ctl = UI.controls(fig);
      const s1 = UI.slider(ctl, { label: 'turnos de ferramenta', min: 1, max: 12, step: 1, value: 8, onInput: (v) => { S.n = v; upd(); } });
      UI.toggle(ctl, { label: 'mascarar resultados de ferramenta', value: true, onChange: (v) => { S.maskTool = v; upd(); } });
      const s2 = UI.slider(ctl, { label: 'comprimento s', min: 0, max: SV.length - 1, step: 1, value: 4, fmt: (v) => SV[v].toLocaleString('pt-BR'), onInput: (v) => { S.si = v; upd(); } });
      UI.toggle(ctl, { label: 'gradient checkpointing', value: true, onChange: (v) => { S.ckpt = v; upd(); } });
      nocase(s1); nocase(s2);
      const rF = UI.readout(ctl, 'fração na perda', '—'), rT = UI.readout(ctl, 'memória total', '—'), rK = UI.readout(ctl, 'cabe num Spark?', '—');
      function upd() {
        const sg = segs(), tot = sg.reduce((a, b) => a + b.n, 0), lo = sg.filter((x) => x.loss).reduce((a, b) => a + b.n, 0);
        rF.set(`${pct(lo / tot)} de ${tot.toLocaleString('pt-BR')}`, S.maskTool ? 'ok' : 'bad');
        const s = SV[S.si], total = BASE + LORA + act(s, S.ckpt) + logit(s);
        rT.set(fmt(total, 1) + ' GB');
        rK.set(total < 107 ? 'sim' : total < 130.7 ? 'no limite' : 'não', total < 107 ? 'ok' : total < 130.7 ? 'warn' : 'bad');
      }
      upd();
      figT(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 560, pad = 10;
        const col = { sys: C.e, user: C.b, think: C.d, call: C.c, tool: C.f, asst: C.a };
        // ---- painel 1: trajetória
        const T1 = narrow ? { x: pad, y: 6, w: w - 2 * pad, h: 112 } : { x: pad, y: 6, w: w * 0.66 - pad, h: h * 0.4 };
        const M1 = narrow ? { x: pad, y: T1.y + T1.h + 4, w: w - 2 * pad, h: 112 } : { x: w * 0.69, y: 6, w: w * 0.31 - pad, h: h * 0.4 };
        const sg = segs(), tot = sg.reduce((a, b) => a + b.n, 0);
        V.text(g, `trajetória de agente · ${tot.toLocaleString('pt-BR')} tokens (ilustrativo)`, T1.x, T1.y + 12, { size: 11.5, color: C.muted, maxW: T1.w });
        const barY = T1.y + 26, barH = 26, bw = T1.w;
        let x = T1.x, lo = 0;
        const head = (t * 0.12) % 1;
        sg.forEach((s) => {
          const ww = (s.n / tot) * bw;
          if (s.loss) { V.box(g, x, barY, Math.max(1, ww - 0.6), barH, { r: 2, fill: V.alpha(col[s.k], 0.85) }); lo += s.n; }
          else { V.box(g, x, barY, Math.max(1, ww - 0.6), barH, { r: 2, fill: V.alpha(col[s.k], 0.14) }); hatch(g, x, barY, ww - 0.6, barH, V.alpha(col[s.k], 0.5), 5); }
          if (s.k === 'tool' && s.loss) { g.strokeStyle = RED(); g.lineWidth = 1.5; g.strokeRect(x + 0.5, barY + 0.5, ww - 1.5, barH - 1); }
          x += ww;
        });
        const hx = T1.x + head * bw;
        V.line(g, hx, barY - 4, hx, barY + barH + 4, { color: C.ink, lw: 1.5 });
        // legenda
        const LG = [['raciocínio', C.d, 1], ['tool call', C.c, 1], ['resultado', C.f, S.maskTool ? 0 : 1], ['resposta', C.a, 1], ['contexto', C.e, 0]];
        let lx = T1.x, ly = barY + barH + 18;
        LG.forEach(([l, c, on]) => { const ww = tw(g, l, 11) + 22; if (lx + ww > T1.x + T1.w) { lx = T1.x; ly += 16; } V.box(g, lx, ly - 9, 12, 10, { r: 2, fill: on ? V.alpha(c, 0.85) : V.alpha(c, 0.2) }); if (!on) hatch(g, lx, ly - 9, 12, 10, V.alpha(c, 0.6), 4); V.text(g, l, lx + 16, ly, { size: 11, color: C.muted }); lx += ww; });
        const fr = lo / tot;
        V.text(g, `na perda: ${pct(fr)}`, T1.x, ly + 20, { size: 13, font: 'mono', weight: 700, color: S.maskTool ? C.a : RED() });
        if (!S.maskTool) V.text(g, '⚠ aprende a inventar resultados de ferramenta', T1.x + 128, ly + 20, { size: 11.5, color: RED(), maxW: T1.w - 130 });
        // ---- referência MiMo
        V.text(g, 'MiMo-V2.6-Distill-Qwen-9B · tokens de perda / total', M1.x, M1.y + 12, { size: 11, color: C.muted, maxW: M1.w });
        const rowH = (M1.h - 22) / MIMO.length;
        MIMO.forEach(([nm, l, tt], i) => {
          const y = M1.y + 20 + i * rowH, lw = 48, pw = 36, bww = M1.w - lw - pw - 4;
          V.text(g, nm, M1.x, y + rowH / 2 + 4, { size: 11, color: i === 4 ? C.ink : C.muted, weight: i === 4 ? 700 : 400 });
          V.box(g, M1.x + lw, y + 3, bww, rowH - 6, { r: 3, fill: V.alpha(C.muted, 0.12) });
          V.box(g, M1.x + lw, y + 3, bww * (l / tt), rowH - 6, { r: 3, fill: V.alpha(C.a, i === 4 ? 0.9 : 0.6) });
          if (rowH >= 16) V.text(g, `${fmt(l, 1)} de ${fmt(tt, 1)} B`, M1.x + lw + bww - 4, y + rowH / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: 'right', maxW: bww * (1 - l / tt) - 6 });
          V.text(g, `${Math.round((100 * l) / tt)}%`, M1.x + lw + bww + pw, y + rowH / 2 + 4, { size: 11.5, font: 'mono', weight: 600, color: C.ink, align: 'right' });
        });
        // ---- painel 2: memória (com e sem checkpointing)
        const P2 = narrow ? { x: pad, y: M1.y + M1.h + 16, w: w - 2 * pad, h: h - M1.y - M1.h - 22 } : { x: pad, y: T1.y + T1.h + 18, w: w - 2 * pad, h: h - T1.h - 30 };
        const s = SV[S.si];
        const compOf = (ck) => [['base BF16 congelada', BASE, C.b], ['LoRA r16 (pesos+grad+Adam)', LORA, C.e], ['ativações', act(s, ck), C.d], ['logits FP32', logit(s), C.c]];
        V.text(g, `Llama-3.1-8B LoRA · s = ${s.toLocaleString('pt-BR')} · b = 1`, P2.x, P2.y + 10, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: P2.w });
        const axMax = 200, labW = narrow ? 0 : 118;
        const ax = { x: P2.x + labW + 4, w: P2.w - labW - (narrow ? 64 : 76) };
        const sx = (v) => ax.x + (Math.min(v, axMax) / axMax) * ax.w;
        const bh = narrow ? 22 : 30, r0 = P2.y + (narrow ? 38 : 34), gapR = narrow ? 38 : 44;
        const yTop = r0 - 8, yBot = r0 + gapR + bh + 8;
        [true, false].forEach((ck, k) => {
          const comp = compOf(ck), total = comp.reduce((a, c) => a + c[1], 0);
          const y = r0 + k * gapR, on = ck === S.ckpt;
          g.globalAlpha = on ? 1 : 0.35;
          if (narrow) V.text(g, ck ? 'com checkpointing' : 'sem checkpointing', ax.x, y - 4, { size: 11, color: C.muted });
          else V.text(g, ck ? 'com checkpointing' : 'sem checkpointing', ax.x - 8, y + bh / 2 + 4, { size: 11.5, color: on ? C.ink : C.muted, align: 'right' });
          let acc = 0;
          comp.forEach(([, v, c]) => { const x1 = sx(acc), x2 = sx(acc + v); if (x2 - x1 > 0.5) V.box(g, x1, y, Math.max(1, x2 - x1 - 1), bh, { r: 3, fill: V.alpha(c, 0.78), stroke: c, lw: 1 }); acc += v; });
          if (total > axMax) { V.arrow(g, sx(axMax) - 2, y + bh / 2, sx(axMax) + 18, y + bh / 2, { color: RED(), lw: 2.5, head: 8 }); V.text(g, fmt(total, 0) + ' GB', sx(axMax) + 22, y + bh / 2 + 4, { size: 12, font: 'mono', weight: 700, color: RED() }); }
          else V.text(g, fmt(total, 1) + ' GB', sx(total) + 6, y + bh / 2 + 4, { size: 12, font: 'mono', weight: 700, color: total < 130.7 ? C.ink : RED() });
          g.globalAlpha = 1;
        });
        const spx = sx(130.7);
        V.line(g, spx, yTop - 4, spx, yBot + 2, { color: RED(), lw: 2, dash: [5, 4] });
        V.text(g, '1 Spark (130,7 GB)', spx + 4, yTop - 6, { size: 11, font: 'mono', color: RED() });
        // faixa prática do Spark: só uma tira âmbar sobre o eixo, com etiqueta
        V.box(g, sx(107), yBot + 3, sx(118) - sx(107), 6, { r: 2, fill: V.alpha(C.d, 0.55), stroke: C.d, lw: 1 });
        V.text(g, '107–118', sx(107) - 4, yBot + 10, { size: 11, font: 'mono', color: C.d, align: 'right' });
        const tickY = yBot + 26;
        [0, 50, 100, 150, 200].forEach((v) => V.text(g, String(v), sx(v), tickY, { size: 11, font: 'mono', color: C.muted, align: 'center' }));
        V.text(g, 'GB', sx(axMax) + 18, tickY, { size: 11, color: C.muted });
        // legenda de componentes com valores (configuração atual)
        let ly2 = tickY + 22, lx2 = P2.x;
        compOf(S.ckpt).forEach(([nm, v, c]) => {
          const label = `${nm}: ${fmt(v, v < 10 ? 2 : 1)} GB`;
          const ww = tw(g, label, 11) + 24;
          if (lx2 + ww > P2.x + P2.w) { lx2 = P2.x; ly2 += 16; }
          V.box(g, lx2, ly2 - 9, 12, 10, { r: 2, fill: V.alpha(c, 0.8) }); V.text(g, label, lx2 + 16, ly2, { size: 11, color: C.muted }); lx2 += ww;
        });
        V.wrap(g, 'tira âmbar: ~107–118 GB práticos por Spark (o resto fica para SO, page cache e picos)', P2.x, ly2 + 18, P2.w, { size: 11, color: C.muted, lineH: 14, maxLines: 2 });
      });
    })();

    /* ================================================ FIG 7 — KL direto × reverso */
    (function () {
      const fig = root.querySelector('#fig-kl');
      adaptH(fig, 420, 460);
      const NB = 160, X0 = 0, X1 = 10, dx = (X1 - X0) / NB;
      const xsB = Array.from({ length: NB }, (_, i) => X0 + (i + 0.5) * dx);
      const gauss = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) * ((x - m) / s)) / s;
      let p = xsB.map((x) => 0.62 * gauss(x, 2.6, 0.6) + 0.38 * gauss(x, 7.2, 0.75));
      const ps = p.reduce((a, b) => a + b, 0); p = p.map((v) => v / ps + 1e-12);
      const pMax = Math.max(...p);
      const qOf = (m, s) => { const q = xsB.map((x) => gauss(x, m, s)); const z = q.reduce((a, b) => a + b, 0); return q.map((v) => v / z + 1e-12); };
      const KL = (a, b) => { let s = 0; for (let i = 0; i < NB; i++) s += a[i] * Math.log(a[i] / b[i]); return s; };
      function lossB(q, beta) {
        if (beta <= 0) return KL(p, q);
        if (beta >= 1) return KL(q, p);
        const m = q.map((v, i) => (1 - beta) * v + beta * p[i]);
        return beta * KL(p, m) + (1 - beta) * KL(q, m);
      }
      const cache = new Map();
      function fit(beta) {
        const key = beta.toFixed(2);
        if (cache.has(key)) return cache.get(key);
        let best = [5, 2], bl = Infinity;
        for (let m = 0.6; m <= 9.4; m += 0.1) for (let s = 0.3; s <= 4.0; s += 0.05) { const l = lossB(qOf(m, s), beta); if (l < bl) { bl = l; best = [m, s]; } }
        const [m0, s0] = best;
        for (let m = m0 - 0.1; m <= m0 + 0.1; m += 0.02) for (let s = Math.max(0.25, s0 - 0.05); s <= s0 + 0.05; s += 0.01) { const l = lossB(qOf(m, s), beta); if (l < bl) { bl = l; best = [m, s]; } }
        cache.set(key, best); return best;
      }
      const S = { beta: 0, mu: 5, sig: 2, target: fit(0), parts: [], landed: [] };
      S.mu = S.target[0]; S.sig = S.target[1];
      const ctl = UI.controls(fig);
      const sb = UI.slider(ctl, { label: 'β da JSD generalizada', min: 0, max: 1, step: 0.05, value: 0, fmt: (v) => fmt(v, 2), onInput: (v) => { setBeta(v, true); } });
      nocase(sb);
      const seg = UI.seg(ctl, { options: [{ v: 0, label: 'SFT · KL direto' }, { v: 0.5, label: 'meio' }, { v: 1, label: 'OPD · KL reverso' }], value: 0, onChange: (v) => { setBeta(v, false); sb.set(v, true); } });
      const rF = UI.readout(ctl, 'KL(p‖q) direto', '—'), rR = UI.readout(ctl, 'KL(q‖p) reverso', '—'), rV = UI.readout(ctl, 'amostras implausíveis', '—');
      function setBeta(v, fromSlider) { S.beta = v; S.target = fit(v); S.landed = []; if (fromSlider) { const pre = [0, 0.5, 1].find((x) => Math.abs(x - v) < 1e-6); if (pre != null) seg.set(pre, true); } const q = qOf(S.target[0], S.target[1]); rF.set(fmt(KL(p, q), 3) + ' nats'); rR.set(fmt(KL(q, p), 3) + ' nats'); }
      setBeta(0, false);
      let lastV = '';
      figT(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const k = 1 - Math.exp(-dt * 4);
        S.mu += (S.target[0] - S.mu) * k; S.sig += (S.target[1] - S.sig) * k;
        const q = qOf(S.mu, S.sig);
        const pad = 14, top = 34, base = h - 44;
        const xs = (x) => pad + ((x - X0) / (X1 - X0)) * (w - 2 * pad);
        const yMax = Math.max(pMax, Math.max(...q)) * 1.1;
        const ys = (v) => base - (v / yMax) * (base - top - 30);
        // regiões
        for (let i = 0; i < NB; i++) {
          const x = xs(xsB[i] - dx / 2), ww = xs(xsB[i] + dx / 2) - x + 0.5;
          if (p[i] > q[i]) { g.fillStyle = V.alpha(C.d, 0.35); g.fillRect(x, ys(p[i]), ww, ys(q[i]) - ys(p[i])); }
          else { g.fillStyle = V.alpha(C.c, 0.32); g.fillRect(x, ys(q[i]), ww, ys(p[i]) - ys(q[i])); }
        }
        const curve = (arr, col, lw, fillA) => {
          g.beginPath(); arr.forEach((v, i) => { const x = xs(xsB[i]), y = ys(v); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
          if (fillA) { g.lineTo(xs(xsB[NB - 1]), base); g.lineTo(xs(xsB[0]), base); g.closePath(); g.fillStyle = V.alpha(col, fillA); g.fill(); g.beginPath(); arr.forEach((v, i) => { const x = xs(xsB[i]), y = ys(v); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); }
          g.strokeStyle = col; g.lineWidth = lw; g.stroke();
        };
        curve(p, C.a, 2.6, 0.08);
        curve(q, C.b, 2.6, 0);
        V.line(g, pad, base, w - pad, base, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.text(g, 'professor p (2 modos)', xs(7.2), ys(p[Math.floor((7.2 - X0) / dx)]) - 10, { size: 11.5, color: C.a, align: 'center', weight: 600 });
        const qPeak = Math.max(...q);
        { const lab = `aluno q: μ ${fmt(S.mu, 1)}, σ ${fmt(S.sig, 2)}`, lw = tw(g, lab, 11.5, 'sans', 600) + 12, lx = V.clamp(xs(S.mu), lw / 2 + 4, w - lw / 2 - 4), ly = Math.max(top + 14, ys(qPeak) - 14);
          V.box(g, lx - lw / 2, ly - 13, lw, 18, { r: 6, fill: V.alpha(C.bg || C.panel, 0.85), stroke: V.alpha(C.b, 0.6), lw: 1 });
          V.text(g, lab, lx, ly, { size: 11.5, color: C.b, align: 'center', weight: 600 }); }
        V.text(g, 'espaço de respostas (ilustrativo)', w - pad, base + 16, { size: 11, color: C.muted, align: 'right' });
        // amostras do aluno caindo
        if (dt > 0 && S.parts.length < 40 && Math.random() < dt * 9) {
          let u = Math.random(), x = 0, acc = 0; for (let i = 0; i < NB; i++) { acc += q[i]; if (u <= acc) { x = xsB[i] + (Math.random() - 0.5) * dx; break; } }
          S.parts.push({ x, y: top - 6, v: 0, c0: Math.abs(S.mu - S.target[0]) < 0.05 && Math.abs(S.sig - S.target[1]) < 0.05 });
        }
        for (let i = S.parts.length - 1; i >= 0; i--) {
          const pt = S.parts[i]; pt.v += dt * 520; pt.y += pt.v * dt;
          const px = xs(pt.x);
          if (pt.y >= base - 3) {
            const bi = V.clamp(Math.floor((pt.x - X0) / dx), 0, NB - 1);
            const ok = p[bi] > 0.04 * pMax;
            const conv = pt.c0 && Math.abs(S.mu - S.target[0]) < 0.05 && Math.abs(S.sig - S.target[1]) < 0.05;
            S.landed.push({ x: px, ok, t: t, c: conv }); if (S.landed.length > 160) S.landed.shift();
            S.parts.splice(i, 1); continue;
          }
          V.dot(g, px, pt.y, 3, V.alpha(C.b, 0.85));
        }
        let bad = 0;
        let nc = 0;
        S.landed.forEach((l) => { const age = t - l.t; if (l.c) { nc++; if (!l.ok) bad++; } if (age < 2.5) V.dot(g, l.x, base + 4, 3.2, V.alpha(l.ok ? C.a : RED(), 1 - age / 2.5)); });
        const fr = nc ? bad / nc : 0;
        const vs = nc > 20 ? pct(fr) : '…';
        if (vs !== lastV) { lastV = vs; rV.set(vs, fr > 0.15 ? 'bad' : 'ok'); }
        // legenda
        const LG = w < 560 ? [[C.d, 'p > q: o aluno não cobre (KL direto pune)'], [C.c, 'q > p: fora do professor (KL reverso pune)']] : [[C.d, 'p > q: massa do professor que o aluno não cobre (KL direto pune)'], [C.c, 'q > p: massa do aluno onde o professor não põe (KL reverso pune)']];
        let ly = 14;
        LG.forEach(([c, l]) => { V.box(g, pad, ly - 9, 12, 10, { r: 2, fill: V.alpha(c, 0.55) }); V.text(g, l, pad + 18, ly, { size: 11, color: C.muted, maxW: w - 2 * pad - 20 }); ly += 15; });
      });
    })();

    /* ================================================ FIG 8 — MOPD stepper */
    (function () {
      const fig = root.querySelector('#fig-mopd');
      adaptH(fig, 470, 600);
      const TOK = ['Primeiro', '3×4', '=', '12', ';', 'depois', '12+5', '=', '18', '.'];
      const PS = [0.30, 0.82, 0.95, 0.90, 0.40, 0.55, 0.70, 0.93, 0.40, 0.85];
      const PT = [0.60, 0.88, 0.97, 0.93, 0.30, 0.50, 0.78, 0.95, 0.05, 0.70];
      const MU = [0.31, 0.81, 0.95, 0.90, 0.41, 0.56, 0.52, 0.93, 0.39, 0.86]; // motor de inferência (ilustrativo)
      const ADV = PS.map((s, i) => Math.log(PT[i] / s));
      const TOPK = [
        { c: ['Primeiro', 'Vamos', '3×4', 'Calculando'], T: [0.60, 0.20, 0.12, 0.05], S: [0.30, 0.35, 0.20, 0.10] },
        { c: ['3×4', '3', 'Calcule', 'O'], T: [0.88, 0.06, 0.03, 0.02], S: [0.82, 0.08, 0.05, 0.03] },
        { c: ['=', 'é', '→', 'dá'], T: [0.97, 0.02, 0.005, 0.003], S: [0.95, 0.03, 0.01, 0.005] },
        { c: ['12', '7', '34', '11'], T: [0.93, 0.03, 0.02, 0.01], S: [0.90, 0.04, 0.02, 0.02] },
        { c: [',', ';', 'e', '.'], T: [0.55, 0.30, 0.08, 0.04], S: [0.30, 0.40, 0.15, 0.10] },
        { c: ['depois', 'então', 'e', 'somamos'], T: [0.50, 0.25, 0.12, 0.08], S: [0.55, 0.20, 0.12, 0.08] },
        { c: ['12+5', '+5', 'somando', '17'], T: [0.78, 0.10, 0.05, 0.04], S: [0.70, 0.12, 0.08, 0.05] },
        { c: ['=', 'é', 'dá', '→'], T: [0.95, 0.03, 0.01, 0.005], S: [0.93, 0.04, 0.015, 0.01] },
        { c: ['17', '18', '16', '15'], T: [0.90, 0.05, 0.02, 0.01], S: [0.35, 0.40, 0.10, 0.05] },
        { c: ['.', '!', ',', '<eos>'], T: [0.70, 0.05, 0.05, 0.15], S: [0.85, 0.03, 0.04, 0.05] },
      ];
      const topkKL = (i) => TOPK[i].c.reduce((a, _, j) => { const s = TOPK[i].S[j], tt = TOPK[i].T[j]; return a + s * Math.log(s / tt) - s + tt; }, 0);
      const S = { amax: 1.5, est: 'pg', sel: 8 };
      const clipA = (a) => V.clamp(a, -S.amax, S.amax);
      const ETA = 0.8;
      const after = (i) => { const p = PS[i], e = Math.exp(ETA * clipA(ADV[i])); return (p * e) / (p * e + 1 - p); };
      const steps = [
        { t: 'Um SFT, vários professores', d: 'Estágios 1 e 2: um SFT geral serve de ponto de partida comum, e cada domínio ganha um professor treinado por RL a partir dele, em paralelo e com hiperparâmetros próprios. No artigo do MOPD: GRPO on-policy com amostragem dinâmica, LR 3e-6, 144 prompts × 8 rollouts (matemática e instruções), sequências de 32.768 tokens (65.536 em SWE).' },
        { t: 'O aluno gera', d: 'Estágio 3: o aluno, reinicializado do mesmo SFT, gera a sua própria resposta. É on-policy: os estados em que ele vai aprender são os que ele mesmo visita. Repare no “18” no fim: o aluno errou a conta.' },
        { t: 'Roteamento e prefill do professor', d: 'O prompt é de matemática, então vai ao professor de matemática. O professor faz <b>um prefill</b> sobre a resposta inteira e devolve \\(\\log\\pi_T\\) em todas as posições de uma vez, num serviço assíncrono: o custo de parede relatado é quase nulo.' },
        { t: 'Vantagem por token', d: 'Cada token ganha \\(\\hat A_t = \\log\\pi_T(y_t) - \\log\\pi_\\theta(y_t)\\), clipada em \\([-A_{\\max}, A_{\\max}]\\). “Primeiro”: 0,6 contra 0,3 → +0,69. “18”: 0,05 contra 0,4 → −2,08. Mude \\(A_{\\max}\\) e veja o clipping; mude para “top-k” e clique num token para ver a distribuição inteira.' },
        { t: 'O aluno se move', d: 'O gradiente \\(-\\hat A_t\\nabla\\log\\pi_\\theta(y_t)\\) sobe os tokens verdes e desce os vermelhos. Sombra = antes, barra cheia = depois de um passo ilustrativo. No top-k, a massa tirada do “18” vai para o “17” que o professor prefere.' },
        { t: 'w_t e ORM (MiMo-V2-Flash)', d: 'Na versão do relatório, as amostras vêm do motor de inferência \\(\\mu_\\theta\\). Cada token é pesado por \\(w_t = \\pi_\\theta/\\mu_\\theta\\) e <b>descartado</b> se a razão sai de \\([\\varepsilon_{\\text{low}}, \\varepsilon_{\\text{high}}]\\) (aqui, intervalo ilustrativo [0,8; 1,25]). Por padrão soma-se \\(\\alpha\\,\\hat A_{\\text{ORM}}\\): a resposta final errada empurra todos os tokens um pouco para baixo.' },
        { t: 'MOPD2: três fluxos', d: 'No MiMo-V2.6, três fluxos alimentam a mesma atualização: rollouts completos (MOPD padrão); prefixos de trajetórias do professor, em que \\(k\\) turnos viram \\(k\\) prefixos e o aluno escreve um turno; e prefixos de demonstrações de SFT, em que o aluno continua. Os professores são mixRL e SFT.' },
        { t: 'O caso das tool calls repetidas', d: 'A alternativa por RL (limiar de 8 chamadas, ~20 passos de MixRL) custaria ~US$ 2,31 milhões e só baixaria a repetição em replay de 13,45% para 3,83%. Um professor RL de turno único, treinado em 12 passos (~7.000 exemplos), zerou a repetição em replay; com o MOPD retomado cinco passos atrás, o custo total foi ~US$ 90 mil, 4% da alternativa.' },
      ];
      let H8 = null;
      const st = UI.stepper(fig, steps, { autoplay: 7000, onStep: () => H8 && H8.redraw() });
      const ctl = UI.controls(fig);
      const sa = UI.slider(ctl, { label: 'A_max (clipping)', min: 0.5, max: 3, step: 0.1, value: 1.5, fmt: (v) => fmt(v, 1), onInput: (v) => { S.amax = v; } });
      nocase(sa);
      UI.seg(ctl, { label: 'estimador', options: [{ v: 'pg', label: '1 amostra' }, { v: 'topk', label: 'top-k (k=4)' }], value: 'pg', onChange: (v) => { S.est = v; } });
      const rA = UI.readout(ctl, 'Â do token “18”', '—');
      let lastA = '';
      const advCol = (a, C) => (a >= 0 ? V.mix(C.dark ? '#1a2b22' : '#e8ecef', C.dark ? '#3FF49B' : '#12985A', V.clamp(a / 1.2, 0, 1)) : V.mix(C.dark ? '#1a2b22' : '#e8ecef', C.dark ? '#FF6B7A' : '#D6203A', V.clamp(-a / 2.1, 0, 1)));

      function pipeline(g, A, i, t, C) {
        const nodes = A.w < 540 ? ['SFT', 'profs.', 'aluno', 'pontua', 'atualiza'] : ['SFT', 'professores', 'aluno gera', 'professor pontua', 'atualiza'];
        const act = [0, 2, 3, 3, 4, 4, 4, 4][i];
        const n = nodes.length, gw = A.w / n;
        nodes.forEach((nm, k) => {
          const cx = A.x + gw * (k + 0.5), on = k === act, past = k < act;
          const ww = Math.min(gw - 10, tw(g, nm, 11) + 16);
          V.box(g, cx - ww / 2, A.y, ww, 22, { r: 11, fill: on ? V.alpha(C.a, 0.3) : past ? V.alpha(C.a, 0.1) : V.alpha(C.muted, 0.08), stroke: on ? C.a : V.alpha(C.muted, 0.4), lw: on ? 1.6 : 1 });
          V.text(g, nm, cx, A.y + 15, { size: 11, color: on ? C.ink : C.muted, align: 'center', maxW: ww - 6 });
          if (k < n - 1) { const x1 = cx + ww / 2 + 2, x2 = A.x + gw * (k + 1.5) - Math.min(gw - 10, tw(g, nodes[k + 1], 11) + 16) / 2 - 2; if (x2 > x1 + 4) V.arrow(g, x1, A.y + 11, x2, A.y + 11, { color: V.alpha(C.muted, 0.6), lw: 1.2, head: 5 }); }
        });
      }
      function tokenLayout(g, x0, maxW, colMin) {
        const ws = TOK.map((s) => Math.max(colMin, tw(g, s, 12, 'mono', 600) + 16));
        return { ws, fl: flow(ws, x0, maxW, 0, 6) };
      }
      H8 = figT(fig, (g, w, h, t, dt, _s, H) => {
        const C = V.C();
        const i = st.i, el = st.elapsed();
        const narrow = w < 560, pad = 10;
        const a8 = fmt(clipA(ADV[8]), 2); if (a8 !== lastA) { lastA = a8; rA.set(a8, 'bad'); }
        pipeline(g, { x: pad, y: 6, w: w - 2 * pad }, i, t, C);
        const top = 40;
        if (i === 0) {
          // SFT → três professores
          const cx = w * (narrow ? 0.22 : 0.2), cy = (top + h) / 2;
          critter(g, cx, cy, 42, C.b, {});
          V.text(g, 'SFT', cx, cy + 36, { size: 12, weight: 700, color: C.ink, align: 'center' });
          const T = [['Matemática', C.a], ['SWE', C.c], ['Ferramentas', C.d]];
          T.forEach(([nm, c], k) => {
            const tx = w * (narrow ? 0.62 : 0.62), ty = top + 36 + k * ((h - top - 100) / 2);
            const u = V.clamp((el - k * 0.4) / 1, 0, 1), e = V.ease.out(u);
            V.arrow(g, cx + 26, cy, V.lerp(cx + 26, tx - 34, e), V.lerp(cy, ty, e), { color: V.alpha(c, 0.8), lw: 2, head: 8, curve: 20 });
            if (u >= 1) {
              critter(g, tx, ty, 34, c, {});
              const a0 = t * 2.2 + k;
              g.strokeStyle = V.alpha(c, 0.8); g.lineWidth = 2; g.beginPath(); g.arc(tx, ty, 26, a0, a0 + 4.6); g.stroke();
              tri(g, tx + 26 * Math.cos(a0 + 4.6), ty + 26 * Math.sin(a0 + 4.6), 4, true, c);
              V.text(g, nm + ' (RL)', tx + 34, ty + 4, { size: 12, weight: 600, color: C.ink, maxW: w - tx - 40 });
            }
          });
          V.text(g, 'cada professor: RL no seu domínio, em paralelo', w / 2, h - 12, { size: 11.5, color: C.muted, align: 'center', maxW: w - 20 });
          return;
        }
        if (i === 6) {
          const lanes = [
            ['MOPD padrão', 'rollout completo do aluno, de ponta a ponta', 0],
            ['Teacher-Prefix OPD', 'k turnos do professor → k prefixos; o aluno escreve 1 turno', 1],
            ['SFT-Prefix OPD', 'histórico de uma demonstração; o aluno continua', 2],
          ];
          const lh = (h - top - 16) / 3;
          lanes.forEach(([nm, ds, k], li) => {
            const y = top + 6 + li * lh, cols = [C.a, C.d, C.e];
            V.box(g, pad, y, w - 2 * pad, lh - 8, { r: 10, fill: V.alpha(cols[li], 0.06), stroke: V.alpha(cols[li], 0.5), lw: 1.2 });
            V.text(g, nm, pad + 10, y + 18, { size: 12.5, weight: 700, color: C.ink });
            V.text(g, ds, pad + 10, y + 34, { size: 11, color: C.muted, maxW: w - 2 * pad - 20 });
            const nT = narrow ? 5 : 7, bw = (w - 2 * pad - 30) / nT, by = y + 44, bh = V.clamp(lh - 62, 14, 34);
            const cyc = Math.floor(t * 0.9 + li * 1.3);
            const jPre = li === 0 ? -1 : 1 + (cyc % (nT - 1));   // tamanho do prefixo deste rollout
            const jFull = li === 0 ? cyc % (nT + 2) : 0;          // MOPD padrão: o aluno escreve tudo
            for (let q = 0; q < nT; q++) {
              const x = pad + 15 + q * bw;
              let fill, stroke, lab, alpha = 1;
              if (li === 0) { fill = V.alpha(C.b, 0.5); stroke = C.b; lab = 'aluno'; alpha = q <= jFull ? 1 : 0.25; }
              else if (q < jPre) { fill = V.alpha(cols[li], 0.4); stroke = cols[li]; lab = li === 1 ? 'prof.' : 'demo'; }
              else if (q === jPre) { fill = V.alpha(C.b, 0.65); stroke = C.b; lab = 'aluno'; }
              else { fill = V.alpha(C.muted, 0.06); stroke = V.alpha(C.muted, 0.3); lab = ''; alpha = 0.5; }
              g.globalAlpha = alpha;
              V.box(g, x, by, bw - 6, bh, { r: 5, fill, stroke, lw: q === jPre ? 2 : 1.2 });
              if (lab) V.text(g, lab, x + (bw - 6) / 2, by + bh / 2 + 4, { size: 11, color: C.ink, align: 'center', maxW: bw - 8 });
              g.globalAlpha = 1;
              if (q === jPre && li > 0) V.text(g, `prefixo de ${jPre} turno${jPre > 1 ? 's' : ''} → o aluno escreve 1; o professor o pontua`, pad + 15, by + bh + 14, { size: 11, color: C.muted, maxW: w - 2 * pad - 30 });
            }
            if (li === 0) V.text(g, 'todos os turnos são do aluno; o professor pontua cada token', pad + 15, by + bh + 14, { size: 11, color: C.muted, maxW: w - 2 * pad - 30 });
          });
          return;
        }
        if (i === 7) {
          const P = { x: pad + (narrow ? 0 : 20), y: top + 10, w: w - 2 * pad - (narrow ? 0 : 40), h: h - top - 20 };
          V.text(g, 'custo da correção (US$)', P.x, P.y + 10, { size: 12, weight: 600, color: C.ink });
          const bars = [['refazer ~20 passos de MixRL', 2.31e6, RED()], ['professor de 12 passos + MOPD', 9.0e4, C.a]];
          const labW = narrow ? 0 : 210, bwMax = P.w - labW - 90;
          bars.forEach(([nm, v, c], k) => {
            const y = P.y + 26 + k * (narrow ? 52 : 40);
            const u = V.ease.out(V.clamp((el - k * 0.4) / 1, 0, 1));
            if (narrow) V.text(g, nm, P.x, y + 2, { size: 11.5, color: C.muted });
            else V.text(g, nm, P.x + labW - 8, y + 16, { size: 11.5, color: C.muted, align: 'right' });
            const bx = P.x + labW, by = narrow ? y + 8 : y + 4;
            V.box(g, bx, by, Math.max(3, (v / 2.31e6) * bwMax * u), 20, { r: 4, fill: V.alpha(c, 0.75), stroke: c, lw: 1.2 });
            V.text(g, v >= 1e6 ? '~2,31 mi' : '~90 mil', bx + Math.max(3, (v / 2.31e6) * bwMax * u) + 6, by + 15, { size: 12, font: 'mono', weight: 700, color: C.ink });
          });
          const y2 = P.y + (narrow ? 140 : 120);
          V.text(g, 'repetição em replay', P.x, y2, { size: 12, weight: 600, color: C.ink });
          const rows = [['antes', 13.45, C.d], ['MixRL limiar 8', 3.83, RED()], ['professor + MOPD', 0, C.a]];
          rows.forEach(([nm, v, c], k) => {
            const y = y2 + 14 + k * 26;
            V.text(g, nm, P.x + (narrow ? 110 : labW) - 8, y + 14, { size: 11.5, color: C.muted, align: 'right' });
            const bx = P.x + (narrow ? 110 : labW), u = V.ease.out(V.clamp((el - 0.8 - k * 0.3) / 0.8, 0, 1));
            const ww = (v / 13.45) * (P.w - (narrow ? 110 : labW) - 70) * u;
            V.box(g, bx, y + 2, Math.max(2, ww), 16, { r: 3, fill: V.alpha(c, 0.75) });
            V.text(g, fmt(v, 2) + '%', bx + Math.max(2, ww) + 6, y + 15, { size: 11.5, font: 'mono', color: C.ink });
          });
          V.text(g, '12 passos · ~7.000 exemplos · 4% do custo', P.x, h - 12, { size: 12, font: 'mono', color: C.a, maxW: P.w });
          return;
        }
        // passos 1–5: a resposta do aluno
        V.text(g, 'prompt (matemática): Quanto é 3×4+5?', pad, top + 12, { size: 12, font: 'mono', color: C.muted, maxW: w - 2 * pad - 50 });
        const tl = tokenLayout(g, pad, w - 2 * pad, narrow ? 54 : 62);
        const showTopK = S.est === 'topk' && (i === 3 || i === 4);
        const rowH = 48;
        const y0 = top + 24;
        const nShown = i === 1 ? Math.min(TOK.length, Math.floor(el * 3.2)) : TOK.length;
        const sweepU = V.clamp(el / 1.2, 0, 1);
        const ORM = 0.3;
        const dropped = (k) => { const r = PS[k] / MU[k]; return r < 0.8 || r > 1.25; };
        const finalA = (k) => (dropped(k) ? 0 : clipA(ADV[k]) - ORM);
        const pNow = (k) => V.lerp(PS[k], after(k), V.ease.inOut(V.clamp((el - 0.3) / 1.2, 0, 1)));
        if (H.pointer.clicked && showTopK) {
          TOK.forEach((_, k) => { const { x, row } = tl.fl.pos[k]; const y = y0 + row * rowH; if (H.pointer.x >= x && H.pointer.x <= x + tl.ws[k] && H.pointer.y >= y && H.pointer.y <= y + 26) S.sel = k; });
        }
        if (i === 2) {
          const sweepX = pad + sweepU * (w - 2 * pad);
          g.fillStyle = V.alpha(C.a, 0.08); g.fillRect(pad, y0 - 4, sweepX - pad, tl.fl.rows * rowH);
          V.line(g, sweepX, y0 - 6, sweepX, y0 + tl.fl.rows * rowH - 10, { color: C.a, lw: 2 });
        }
        for (let k = 0; k < nShown; k++) {
          const { x, row } = tl.fl.pos[k], ww = tl.ws[k];
          const y = y0 + row * rowH;
          const a = clipA(ADV[k]);
          let fill = V.alpha(C.b, 0.25), stroke = C.b;
          if (i === 3 || i === 4) { fill = advCol(a, C); stroke = a >= 0 ? C.a : RED(); }
          if (i === 5) { const fa = finalA(k); fill = dropped(k) ? V.alpha(C.muted, 0.12) : advCol(fa, C); stroke = dropped(k) ? C.muted : fa >= 0 ? C.a : RED(); }
          g.globalAlpha = i === 5 && dropped(k) ? 0.45 : 1;
          chip(g, x, y, ww, 26, TOK[k], { size: 12, weight: 600, fill, stroke, lw: S.sel === k && showTopK ? 2.6 : 1.3, color: C.ink });
          g.globalAlpha = 1;
          if (i === 5 && dropped(k)) mark(g, x + ww / 2, y + 13, false, 8);
          const ly = y + 40;
          let txt = '', col = C.muted;
          if (i === 1) { txt = 'πθ ' + fmt(PS[k], 2); col = C.b; }
          if (i === 2) { const vis = pad + sweepU * (w - 2 * pad) > x + ww / 2; txt = vis ? 'πT ' + fmt(PT[k], 2) : '…'; col = C.a; }
          if (i === 3) { txt = (a >= 0 ? '+' : '') + fmt(a, 2); col = a >= 0 ? C.a : RED(); }
          if (i === 4) { const p1 = pNow(k); txt = fmt(p1, 2); col = p1 >= PS[k] ? C.a : RED(); }
          if (i === 5) { txt = dropped(k) ? 'w ' + fmt(PS[k] / MU[k], 2) : (finalA(k) >= 0 ? '+' : '') + fmt(finalA(k), 2); col = dropped(k) ? RED() : finalA(k) >= 0 ? C.a : RED(); }
          V.text(g, txt, x + ww / 2, ly, { size: 11.5, font: 'mono', weight: 600, color: col, align: 'center', maxW: ww + 4 });
        }
        if (i === 1) {
          critter(g, w - pad - 18, top + 4, 22, C.b, { mood: nShown < TOK.length ? 'o' : undefined });
        }
        const yEnd = y0 + tl.fl.rows * rowH;
        const P = { x: pad, y: yEnd + 2, w: w - 2 * pad, h: h - yEnd - 8 };
        V.box(g, P.x, P.y, P.w, P.h, { r: 10, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        if (showTopK) {
          const k = S.sel, d = TOPK[k];
          V.text(g, `top-4 do professor na posição “${TOK[k]}” · termo KL = ${fmt(topkKL(k), 3)}`, P.x + 10, P.y + 16, { size: 11.5, font: 'mono', color: C.ink, maxW: P.w - 20 });
          V.text(g, 'menta = professor · lavanda = aluno' + (i === 4 ? ' (andando para o professor)' : ''), P.x + 10, P.y + 32, { size: 11, color: C.muted, maxW: P.w - 20 });
          const gw = (P.w - 20) / 4, bh = P.h - 88;
          const mx = Math.max(...d.T, ...d.S);
          d.c.forEach((c, j) => {
            const cx = P.x + 10 + gw * (j + 0.5), bwid = Math.min(26, gw * 0.3);
            const sNow = i === 4 ? V.lerp(d.S[j], d.S[j] + (d.T[j] - d.S[j]) * 0.45, V.ease.inOut(V.clamp((el - 0.3) / 1.2, 0, 1))) : d.S[j];
            const hT = (d.T[j] / mx) * bh, hS = (sNow / mx) * bh, base = P.y + 58 + bh;
            V.box(g, cx - bwid - 1, base - hT, bwid, Math.max(1, hT), { r: 3, fill: V.alpha(C.a, 0.6), stroke: C.a, lw: 1 });
            V.box(g, cx + 1, base - hS, bwid, Math.max(1, hS), { r: 3, fill: V.alpha(C.b, 0.5), stroke: C.b, lw: 1 });
            V.text(g, fmt(d.T[j], 2), cx - bwid / 2 - 1, base - hT - 4, { size: 11, font: 'mono', color: C.a, align: 'center' });
            V.text(g, fmt(sNow, 2), cx + bwid / 2 + 1, base - hS - 4, { size: 11, font: 'mono', color: C.b, align: 'center' });
            V.text(g, c, cx, base + 16, { size: 11.5, font: 'mono', weight: c === TOK[k] ? 700 : 400, color: C.ink, align: 'center', maxW: gw - 4 });
          });
        } else {
          const TITLE = {
            1: 'πθ(yₜ): a confiança do aluno em cada token que escreveu',
            2: 'mesmo token, duas opiniões: πθ do aluno (lavanda) × πT do professor (menta)',
            3: 'Âₜ = log πT − log πθ, clipada em ±A_max (contorno = valor antes do clip)',
            4: 'πθ(yₜ) antes (sombra) e depois de um passo de gradiente',
            5: 'vantagem final = Â − 0,30 (ORM) · tokens com w fora de [0,8; 1,25] saem',
          }[i];
          V.text(g, TITLE, P.x + 10, P.y + 16, { size: 11.5, color: C.ink, maxW: P.w - 20 });
          const sw = (P.w - 44) / TOK.length, rot = sw < 58;
          const cT = P.y + 30, cB = P.y + P.h - (rot ? 46 : 22);
          const X = (k) => P.x + 36 + sw * (k + 0.5);
          const probChart = i === 1 || i === 2 || i === 4;
          const lo = probChart ? 0 : -2.3, hi = probChart ? 1 : 1.2;
          const Y = (v) => cB - ((V.clamp(v, lo, hi) - lo) / (hi - lo)) * (cB - cT);
          (probChart ? [0, 0.5, 1] : [-2, -1, 0, 1]).forEach((v) => { V.line(g, P.x + 34, Y(v), P.x + P.w - 8, Y(v), { color: v === 0 && !probChart ? V.alpha(C.ink, 0.5) : C.grid, lw: 1 }); V.text(g, fmt(v, probChart ? 1 : 0), P.x + 30, Y(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
          if (!probChart) {
            [-S.amax, S.amax].forEach((v) => { if (v >= lo && v <= hi) { V.line(g, P.x + 34, Y(v), P.x + P.w - 8, Y(v), { color: C.d, lw: 1.5, dash: [5, 4] }); V.text(g, (v > 0 ? '+' : '−') + 'A_max', P.x + P.w - 10, Y(v) - 4, { size: 11, font: 'mono', color: C.d, align: 'right' }); } });
          }
          const bw = Math.min(24, sw * 0.42);
          for (let k = 0; k < TOK.length; k++) {
            if (i === 1 && k >= nShown) break;
            const cx = X(k);
            if (i === 1) { V.box(g, cx - bw / 2, Y(PS[k]), bw, Y(0) - Y(PS[k]), { r: 3, fill: V.alpha(C.b, 0.65), stroke: C.b, lw: 1 }); }
            if (i === 2) {
              V.box(g, cx - bw - 1, Y(PS[k]), bw, Y(0) - Y(PS[k]), { r: 3, fill: V.alpha(C.b, 0.6), stroke: C.b, lw: 1 });
              const vis = pad + sweepU * (w - 2 * pad) > tl.fl.pos[k].x + tl.ws[k] / 2;
              if (vis) V.box(g, cx + 1, Y(PT[k]), bw, Y(0) - Y(PT[k]), { r: 3, fill: V.alpha(C.a, 0.65), stroke: C.a, lw: 1 });
            }
            if (i === 4) {
              const p1 = pNow(k);
              V.box(g, cx - bw / 2, Y(PS[k]), bw, Y(0) - Y(PS[k]), { r: 3, fill: V.alpha(C.muted, 0.15), stroke: V.alpha(C.muted, 0.6), lw: 1 });
              V.box(g, cx - bw / 2 + 3, Y(p1), bw - 6, Math.max(1, Y(0) - Y(p1)), { r: 2, fill: V.alpha(p1 >= PS[k] ? C.a : RED(), 0.85) });
              if (Math.abs(p1 - PS[k]) > 0.02) tri(g, cx, Y(Math.max(p1, PS[k])) - 9, 4.5, p1 > PS[k], p1 > PS[k] ? C.a : RED());
            }
            if (i === 3 || i === 5) {
              const raw = i === 3 ? ADV[k] : clipA(ADV[k]) - ORM, v = i === 3 ? clipA(ADV[k]) : finalA(k);
              if (i === 5 && dropped(k)) { mark(g, cx, Y(0), false, 7); if (sw >= 46) V.text(g, 'w fora', cx, Y(0) + 20, { size: 11, font: 'mono', color: RED(), align: 'center' }); }
              else {
              const yv = Y(v), y0b = Y(0), col = v >= 0 ? C.a : RED();
              V.box(g, cx - bw / 2, Math.min(yv, y0b), bw, Math.max(1, Math.abs(yv - y0b)), { r: 3, fill: V.alpha(col, 0.75), stroke: col, lw: 1 });
              if (i === 3 && Math.abs(raw - v) > 1e-6) { g.save(); g.setLineDash([3, 3]); V.rr(g, cx - bw / 2, Math.min(Y(raw), y0b), bw, Math.abs(Y(raw) - y0b), 3); g.strokeStyle = col; g.lineWidth = 1.2; g.stroke(); g.restore(); }
              if (sw >= 46 || Math.abs(v) > 1) V.text(g, (v >= 0 ? '+' : '') + fmt(v, 2), cx, v >= 0 ? yv - 4 : Math.min(yv + 13, cB + 12), { size: 11, font: 'mono', color: C.ink, align: 'center' });
              }
            }
            // rótulo do token
            if (rot) { g.save(); g.translate(cx + 3, cB + 10); g.rotate(-Math.PI / 4.5); V.text(g, TOK[k], 0, 0, { size: 11, font: 'mono', color: C.muted, align: 'right' }); g.restore(); }
            else V.text(g, TOK[k], cx, cB + 15, { size: 11, font: 'mono', color: C.muted, align: 'center', maxW: sw - 2 });
          }
        }
      });
    })();

    /* ================================================ FIG 9 — curvas de treino */
    (function () {
      const fig = root.querySelector('#fig-curvas');
      adaptH(fig, 440, 500);
      const S = { method: 'full', lr: Math.log10(2e-5), ep: 3, nlog: 4, rep: 0, t0: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'método', options: [{ v: 'full', label: 'full FT' }, { v: 'lora', label: 'LoRA' }], value: 'full', onChange: (v) => { S.method = v; touch(); } });
      const lrFmt = (v) => { const x = Math.pow(10, v); const e = Math.floor(Math.log10(x) + 1e-9); return fmt(x / Math.pow(10, e), 1) + 'e' + e; };
      const s1 = UI.slider(ctl, { label: 'LR', min: -6, max: -3, step: 0.05, value: S.lr, fmt: lrFmt, onInput: (v) => { S.lr = v; touch(); } });
      const s2 = UI.slider(ctl, { label: 'épocas', min: 1, max: 6, step: 1, value: 3, onInput: (v) => { S.ep = v; touch(); } });
      const s3 = UI.slider(ctl, { label: 'exemplos', min: 3, max: 6, step: 0.25, value: 4, fmt: (v) => { const n = Math.round(Math.pow(10, v)); return n >= 1e6 ? fmt(n / 1e6, 1) + ' mi' : Math.round(n / 1000) + ' mil'; }, onInput: (v) => { S.nlog = v; touch(); } });
      const s4 = UI.slider(ctl, { label: 'dados gerais (replay)', min: 0, max: 50, step: 5, value: 0, fmt: (v) => v + '%', onInput: (v) => { S.rep = v / 100; touch(); } });
      [s1, s2, s3, s4].forEach(nocase);
      const rB = UI.readout(ctl, 'melhor época', '—'), rV = UI.readout(ctl, 'val. no melhor', '—'), rG = UI.readout(ctl, 'bateria geral no fim', '—');
      let CUR = null;
      function model() {
        const full = S.method === 'full';
        const lr = Math.pow(10, S.lr), opt = full ? 2e-5 : 2e-4, e = lr / opt;
        let r = e < 1 ? e : 1 + 0.35 * Math.log(e);
        r *= 1 - 0.45 * S.rep;
        const nF = S.nlog;
        const L0 = 1.6, Lf = 1.05 - 0.1 * (nF - 3) + (full ? 0 : 0.04);
        const m = (full ? 1 : 0.5) * 0.24 * (6.5 - nF) / 3.5;
        const o = (full ? 1 : 0.6) * 0.17 * (6.5 - nF) / 3.5;
        const inst = e > 4 ? 0.05 * Math.pow(e - 4, 0.8) : 0;
        const div = e > 25 ? 0.6 * (Math.log10(e / 25) + 0.3) : 0;
        const rr = V.rng(9);
        const bumps = Array.from({ length: 10 }, () => [rr() * 6, 0.04 + rr() * 0.05, 0.5 + rr()]);
        const spike = (x) => bumps.reduce((a, [c, wd, am]) => a + am * Math.exp(-0.5 * ((x - c) / wd) ** 2), 0);
        const N = S.ep * 50;
        const tr = [], va = [], kl = [], ge = [];
        for (let k = 0; k <= N; k++) {
          const x = (k / N) * S.ep;
          const base = Lf + (L0 - Lf) * Math.exp(-4 * r * x);
          const ins = inst * spike(x) + div * Math.max(0, x - 0.5) ** 2;
          tr.push(Math.max(0.06, base - m * Math.pow(Math.max(0, x - 0.6), 1.3) * Math.min(1, r) + ins));
          va.push(base + 0.02 + o * Math.max(0, x - 1.2) ** 2 * Math.min(1.5, r) + ins * 1.1);
          const kv = 0.12 * (1 + 0.4 * Math.log10(Math.max(1, e))) * (1 - Math.exp(-3 * r * x)) * (1 + 0.6 * Math.max(0, x - 1)) * (full ? 1 : 0.5) * (1 - 0.75 * S.rep) + ins * 0.5;
          kl.push(kv); ge.push(V.clamp(100 - 70 * kv, 20, 100));
        }
        let bi = 0; va.forEach((v, k) => { if (v < va[bi]) bi = k; });
        return { tr, va, kl, ge, bi, N };
      }
      function touch() { CUR = model(); S.t0 = performance.now() / 1000; const best = (CUR.bi / CUR.N) * S.ep; rB.set(fmt(best, 1)); rV.set(fmt(CUR.va[CUR.bi], 2) + ' nats'); const gEnd = CUR.ge[CUR.N]; rG.set(fmt(gEnd, 0) + '% do base', gEnd > 88 ? 'ok' : gEnd > 75 ? 'warn' : 'bad'); }
      touch();
      figT(fig, (g, w, h, t) => {
        const C = V.C();
        const pad = 10, narrow = w < 560;
        const u = V.ease.out(V.clamp((performance.now() / 1000 - S.t0) / 1.2, 0, 1));
        const M = CUR, nShow = Math.max(1, Math.floor(u * M.N));
        const L = pad + 40, R = w - pad - (narrow ? 8 : 108);
        const A = { y: 24, h: h * 0.52 };
        const B = { y: A.y + A.h + 50, h: h - A.y - A.h - 74 };
        const xs = (k) => L + (k / M.N) * (R - L);
        // painel A
        const yMax = Math.min(3, Math.max(1.7, ...M.va.slice(0, nShow + 1)) * 1.05);
        const ya = (v) => A.y + A.h - (Math.min(v, yMax) / yMax) * A.h;
        V.text(g, 'perda mascarada (nats)', pad, A.y - 8, { size: 11.5, color: C.muted });
        [0, 0.5, 1, 1.5, 2, 2.5].filter((v) => v <= yMax).forEach((v) => { V.line(g, L, ya(v), R, ya(v), { color: C.grid, lw: 1 }); V.text(g, fmt(v, 1), L - 6, ya(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        for (let e = 0; e <= S.ep; e++) { const x = L + (e / S.ep) * (R - L); V.line(g, x, A.y, x, A.y + A.h, { color: V.alpha(C.muted, 0.18), lw: 1, dash: [2, 4] }); V.text(g, String(e), x, A.y + A.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        V.text(g, 'época', R, A.y + A.h + 27, { size: 11, color: C.muted, align: 'right' });
        const path = (arr, yf, col, lw, dash) => { g.beginPath(); for (let k = 0; k <= nShow; k++) { const x = xs(k), y = yf(arr[k]); if (k) g.lineTo(x, y); else g.moveTo(x, y); } g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        path(M.tr, ya, C.a, 2.5);
        path(M.va, ya, C.b, 2.5, [6, 4]);
        if (nShow >= M.bi) {
          const bx = xs(M.bi), by = ya(M.va[M.bi]);
          V.glow(g, bx, by, 16 + 3 * Math.sin(t * 3), C.d, 0.8);
          V.text(g, '★', bx, by + 5, { size: 16, color: C.d, align: 'center' });
          if (M.bi < M.N * 0.92) V.text(g, 'pare aqui', bx + 10, by - 10, { size: 11.5, weight: 600, color: C.d });
        }
        if (!narrow) {
          // rótulos de fim de curva: afastados da estrela quando ela está no fim, e sem se sobrepor
          const starEnd = nShow >= M.bi && M.bi > M.N * 0.92, ex = R + (starEnd ? 16 : 6);
          let yT = ya(M.tr[nShow]) + 4, yV = ya(M.va[nShow]) + 4;
          if (Math.abs(yT - yV) < 13) { if (yT >= yV) yT = yV + 13; else yV = yT + 13; }
          V.text(g, 'treino', ex, yT, { size: 11, color: C.a }); V.text(g, 'validação', ex, yV, { size: 11, color: C.b });
        }
        else { V.text(g, 'treino ━  validação ╍', R, A.y - 8, { size: 11, color: C.muted, align: 'right' }); }
        // painel B
        const yb = (v) => B.y + B.h - (v / 100) * B.h;
        const klMax = Math.max(0.5, ...M.kl) * 1.05;
        const yk = (v) => B.y + B.h - (v / klMax) * B.h;
        V.text(g, narrow ? 'geral (rosa, % do base) · KL (âmbar, escala própria)' : 'custo colateral: bateria geral (rosa, % do base, eixo à esquerda) · KL ao base (âmbar, eixo à direita)', pad, B.y - 8, { size: 11.5, color: C.muted, maxW: w - 2 * pad });
        [50, 100].forEach((v) => { V.line(g, L, yb(v), R, yb(v), { color: C.grid, lw: 1 }); V.text(g, v + '%', L - 6, yb(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        if (!narrow) {
          // eixo próprio do KL, à direita: 0, klMax/2, klMax (coincidem com 0%, 50% e 100%)
          V.line(g, R, B.y, R, B.y + B.h, { color: V.alpha(C.d, 0.5), lw: 1 });
          [0, klMax / 2, klMax].forEach((v) => { const y = yk(v); V.line(g, R, y, R + 4, y, { color: C.d, lw: 1.2 }); V.text(g, fmt(v, 2), R + 7, y + 4, { size: 11, font: 'mono', color: C.d }); });
        }
        path(M.ge, yb, C.c, 2.5);
        path(M.kl, yk, C.d, 2.2, [3, 3]);
        const ge = M.ge[nShow], kv = M.kl[nShow];
        const endLab = (txt, y, col) => {
          if (!narrow) { V.text(g, txt, R + 42, y + 4, { size: 11, font: 'mono', color: col }); return; }
          const tw2 = tw(g, txt, 11, 'mono') + 8;
          V.box(g, R - tw2, y - 8, tw2, 16, { r: 4, fill: V.alpha(C.bg || C.panel, 0.9), stroke: V.alpha(col, 0.6), lw: 1 });
          V.text(g, txt, R - 4, y + 4, { size: 11, font: 'mono', color: col, align: 'right' });
        };
        // no celular, a etiqueta vai acima da curva, ou abaixo dela se encostaria no título do painel
        const yG = narrow ? (yb(ge) - 22 < B.y ? yb(ge) + 14 : yb(ge) - 14) : yb(ge), yK = yk(kv) + (narrow ? 14 : 0);
        let yG2 = yG, yK2 = yK;
        if (narrow) { if (Math.abs(yG - yK) < 18) { if (yG <= yK) yG2 = yK - 18; else yK2 = yG - 18; } }
        else if (Math.abs(yG - yK) < 13) { if (yG <= yK) yK2 = yG + 13; else yG2 = yK + 13; }
        endLab(`geral ${fmt(ge, 0)}%`, yG2, C.c);
        endLab(`KL ${fmt(kv, 2)}`, yK2, C.d);
      });
    })();

    /* ================================================ FIG 10 — esquecimento */
    (function () {
      const fig = root.querySelector('#fig-esquecimento');
      adaptH(fig, 420, 500);
      const PTS = [
        { k: 'base', l: 'Qwen3-8B base', qa: 18, ife: 85 },
        { k: 'm100', l: '100% domínio', qa: 43, ife: 45 },
        { k: 'm70', l: '70% domínio + 30% chat', qa: 36, ife: 79 },
        { k: 'opd', l: '+ destilação on-policy', qa: 41, ife: 83 },
      ];
      const S = { cur: 0, auto: true, from: { qa: 18, ife: 85 }, t0: 0, last: 0 };
      const ctl = UI.controls(fig);
      const seg = UI.seg(ctl, { label: 'cenário', options: PTS.map((p, i) => ({ v: i, label: p.l })), value: 0, onChange: (v) => { go(v); S.auto = false; tg.set(false); } });
      const tg = UI.toggle(ctl, { label: 'percorrer sozinho', value: true, onChange: (v) => { S.auto = v; S.last = performance.now() / 1000; } });
      const rQ = UI.readout(ctl, 'QA interno', '—'), rI = UI.readout(ctl, 'IF-eval', '—'), rD = UI.readout(ctl, 'Δ IF-eval vs base', '—');
      function cur() { const u = V.ease.inOut(V.clamp((performance.now() / 1000 - S.t0) / 0.9, 0, 1)); const P = PTS[S.cur]; return { qa: V.lerp(S.from.qa, P.qa, u), ife: V.lerp(S.from.ife, P.ife, u) }; }
      function go(i) { S.from = cur(); S.cur = i; S.t0 = performance.now() / 1000; const P = PTS[i]; rQ.set(P.qa + '%'); rI.set(P.ife + '%', P.ife >= 80 ? 'ok' : P.ife >= 60 ? 'warn' : 'bad'); const d = P.ife - 85; rD.set((d > 0 ? '+' : d < 0 ? '−' : '') + Math.abs(d) + ' p.p.', d >= -5 ? 'ok' : d >= -20 ? 'warn' : 'bad'); }
      go(0);
      figT(fig, (g, w, h, t) => {
        const C = V.C();
        const now = performance.now() / 1000;
        if (S.auto && now - S.last > 3.2) { S.last = now; const n = (S.cur + 1) % PTS.length; go(n); seg.set(n, true); }
        const narrow = w < 560, pad = 12;
        const A = narrow ? { x: pad + 34, y: 18, w: w - 2 * pad - 44, h: h * 0.58 } : { x: pad + 40, y: 18, w: w * 0.6 - pad - 50, h: h - 64 };
        const B = narrow ? { x: pad, y: A.y + A.h + 46, w: w - 2 * pad, h: h - A.h - 70 } : { x: w * 0.64, y: 24, w: w * 0.36 - pad, h: h - 60 };
        const xs = (v) => A.x + (v / 50) * A.w, ys = (v) => A.y + A.h - ((v - 40) / 50) * A.h;
        [0, 10, 20, 30, 40, 50].forEach((v) => { V.line(g, xs(v), A.y, xs(v), A.y + A.h, { color: C.grid, lw: 1 }); V.text(g, v + '%', xs(v), A.y + A.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        [40, 50, 60, 70, 80, 90].forEach((v) => { V.line(g, A.x, ys(v), A.x + A.w, ys(v), { color: C.grid, lw: 1 }); V.text(g, v + '%', A.x - 6, ys(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); });
        V.text(g, 'QA interno (conhecimento novo) →', A.x + A.w, A.y + A.h + 30, { size: 11, color: C.muted, align: 'right' });
        V.text(g, 'IF-eval ↑', A.x - 34, A.y - 6, { size: 11, color: C.muted });
        // trajetórias
        const tr = (a, b, col) => V.arrow(g, xs(a.qa), ys(a.ife), xs(b.qa), ys(b.ife), { color: V.alpha(col, 0.45), lw: 1.5, head: 7, dash: [4, 4] });
        tr(PTS[0], PTS[1], RED()); tr(PTS[0], PTS[2], C.d); tr(PTS[2], PTS[3], C.a);
        PTS.forEach((p, i) => {
          V.dot(g, xs(p.qa), ys(p.ife), 5, V.alpha(i === 0 ? C.b : i === 1 ? RED() : i === 2 ? C.d : C.a, 0.6));
          const lab = narrow ? ['base', '100%', '70/30', '+OPD'][i] : p.l;
          const POSL = (narrow ? [['right', -12, 18], ['right', -18, 4], ['right', -9, 18], ['right', -9, -10]] : [['right', -12, -8], ['right', -15, 4], ['left', 10, 18], ['right', -9, -10]])[i];
          V.text(g, lab, xs(p.qa) + POSL[1], ys(p.ife) + POSL[2], { size: 11, color: C.muted, align: POSL[0] });
        });
        // guia (KL) do base ao ponto atual
        const c = cur(), bx = xs(18), by = ys(85), cx = xs(c.qa), cy = ys(c.ife);
        const dist = Math.hypot(cx - bx, cy - by), tension = V.clamp((85 - c.ife) / 40, 0, 1);
        const sag = Math.max(0, 26 - dist * 0.12) * (1 - tension);
        g.beginPath(); g.moveTo(bx, by);
        g.quadraticCurveTo((bx + cx) / 2, (by + cy) / 2 + sag + Math.sin(t * 3) * 2 * (1 - tension), cx, cy);
        g.strokeStyle = V.mix(C.dark ? '#26D07C' : '#12985A', C.dark ? '#FCBB00' : '#B77900', tension); g.lineWidth = 2.5 + 1.5 * tension; g.stroke();
        V.glow(g, cx, cy, 20, C.a, 0.8);
        critter(g, cx, cy, 20, PTS[S.cur].ife >= 79 ? C.a : RED(), { mood: PTS[S.cur].ife >= 79 ? undefined : 'sad' });
        critter(g, bx, by, 14, C.b, {});
        // barras
        V.text(g, PTS[S.cur].l, B.x, B.y + 4, { size: 12, weight: 700, color: C.ink, maxW: B.w });
        const mets = [['QA interno', c.qa, 18, 50], ['IF-eval', c.ife, 85, 100]];
        const bh = narrow ? 22 : 26;
        mets.forEach(([nm, v, b0, mx], k) => {
          const y = B.y + 26 + k * (bh + 34);
          V.text(g, nm, B.x, y, { size: 11.5, color: C.muted });
          const bw = B.w - 50;
          V.box(g, B.x, y + 6, bw, bh, { r: 5, fill: V.alpha(C.muted, 0.1) });
          V.box(g, B.x, y + 6, bw * (v / mx), bh, { r: 5, fill: V.alpha(k ? C.c : C.e, 0.75) });
          g.strokeStyle = V.alpha(C.ink, 0.7); g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(B.x + bw * (b0 / mx), y + 2); g.lineTo(B.x + bw * (b0 / mx), y + 10 + bh); g.stroke(); g.setLineDash([]);
          V.text(g, Math.round(v) + '%', B.x + bw + 6, y + 6 + bh / 2 + 4, { size: 12, font: 'mono', weight: 700, color: C.ink });
        });
        const tY = B.y + 26 + 2 * (bh + 34) - 6;
        V.text(g, 'tracejado = base', B.x, tY, { size: 11, color: C.muted });
        // tabela com os quatro cenários
        const ty0 = tY + 22, rh2 = narrow ? 17 : 20;
        if (ty0 + rh2 * 5 < B.y + B.h + 30) {
          V.text(g, 'cenário', B.x, ty0, { size: 11, color: C.muted }); V.text(g, 'QA', B.x + B.w - 56, ty0, { size: 11, color: C.muted, align: 'right' }); V.text(g, 'IF', B.x + B.w - 6, ty0, { size: 11, color: C.muted, align: 'right' });
          PTS.forEach((p, k) => {
            const y = ty0 + (k + 1) * rh2, on = k === S.cur;
            if (on) V.box(g, B.x - 4, y - 13, B.w + 6, rh2 - 2, { r: 5, fill: V.alpha(C.a, 0.12) });
            V.text(g, p.l, B.x, y, { size: 11, color: on ? C.ink : C.muted, weight: on ? 600 : 400, maxW: B.w - 90 });
            V.text(g, p.qa + '%', B.x + B.w - 56, y, { size: 11, font: 'mono', color: on ? C.ink : C.muted, align: 'right' });
            V.text(g, p.ife + '%', B.x + B.w - 6, y, { size: 11, font: 'mono', color: on ? C.ink : C.muted, align: 'right' });
          });
        }
      });
    })();
  },
});

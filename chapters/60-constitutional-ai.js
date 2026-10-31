LA.chapter({
  id: 'constitutional-ai',
  part: 13,
  order: 60,
  title: 'Constitutional AI: princípios, autocrítica e RLAIF',
  navTitle: 'Constitutional AI',
  subtitle: 'Como uma lista de princípios em linguagem natural substitui rótulos humanos de dano: autocrítica e revisão (SL-CAI), preferências rotuladas por IA (RLAIF), o clamp dos rótulos suaves, Goodhart, deliberative alignment, a constituição de 2026 e a diferença entre ensinar princípios e abliterar uma direção.',
  minutes: 40,
  recipes: ['ds', 'glm', 'qwen'],
  tags: ['Constitutional AI', 'RLAIF', 'SL-CAI', 'RL-CAI', 'rótulo suave', 'destilação de contexto', 'deliberative alignment', 'sobre-recusa', 'abliteração'],
  goals: [
    'Explicar as duas fases do CAI, <b>SL-CAI</b> (gerar → criticar → revisar → SFT) e <b>RL-CAI</b> (rótulos de IA → modelo de preferência → PPO), e por que cada uma existe',
    'Calcular o rótulo suave \\(p_A=\\sigma(\\ell_A-\\ell_B)\\) do rotulador de múltipla escolha, a perda de Bradley–Terry com rótulo suave e a margem ótima \\(\\Delta^*=\\mathrm{logit}(p_A)\\), e justificar o clamp de 40–60%',
    'Reconhecer o Goodhart no RL-CAI (boilerplate, dureza) e controlá-lo com a coleira KL, o ensemble de princípios e a avaliação nos dois sentidos (sobre-recusa × cumprimento indevido)',
    'Situar os desdobramentos (RLAIF/d-RLAIF, Collective CAI, deliberative alignment, Rule Based Rewards, Constitutional Classifiers, constituição de 2026) e contrastar o CAI com a abliteração de posto 1 do Módulo 1',
    'Montar um pipeline CAI com um servidor de inferência do Módulo 1 como professor e juiz, aproveitando o prefix caching',
  ],
  prereqs: ['preferencias-reward-models', 'ppo-rlhf', 'sft', 'abliteracao'],
  summary: [
    'O CAI troca rótulos humanos de dano por uma <b>constituição</b> (16 + 16 princípios no paper, sorteados um por vez) e por julgamentos do próprio modelo. Os humanos continuam fornecendo os rótulos de helpfulness (135.296 comparações).',
    'SL-CAI: prompt de red team → resposta do helpful-only → crítica → revisão (4 rodadas por prompt, 182.831 prompts) → SFT de um modelo pré-treinado nas revisões. É <b>destilação de contexto</b>: o andaime some do exemplo de treino. A primeira revisão remove a maior parte do dano.',
    'RL-CAI: o rotulador lê \\(\\ell_A,\\ell_B\\) de “(A)” e “(B)” e dá \\(p_A=\\sigma(\\ell_A-\\ell_B)\\); o PM aprende com Bradley–Terry de rótulo suave, gradiente \\(\\sigma(\\Delta)-p_A\\) e margem ótima \\(\\mathrm{logit}(p_A)\\) (−0,4 e −1,6 → 0,7685 → 1,2). Rótulos de CoT saem quase 0/1 e são clampados em 40–60% (margem ≤ 0,405).',
    'O PM mistura helpfulness humana e harmlessness de IA (57,5% das 318.127 comparações) e vira a recompensa do PPO, com KL ao SL-CAI. Resultado: fronteira de Pareto melhor, menos dano à mesma utilidade. Sobretreinado, aparece o Goodhart: boilerplate e dureza.',
    'RLAIF ≈ RLHF em Lee et al. (inofensivo: 88% × 76%); funciona com rotulador do mesmo tamanho, e o d-RLAIF dispensa o PM. O deliberative alignment destila uma spec para dentro do CoT e esconde o CoT do juiz no RL; a constituição de 2026 explica o porquê e ordena prioridades.',
    'A abliteração é uma edição fechada de posto 1, feita à mão, que desliga a direção de recusa; o pós-treino aprende de dados uma mudança espalhada por todas as camadas (com SGD, uma soma de produtos externos \\(\\delta x^\\top\\); na prática de posto efetivo baixo, por isso a LoRA funciona). A ordem das operações importa, e a direção de recusa serve de sonda de regressão.',
    'Avalie segurança nos dois sentidos (XSTest para sobre-recusa; StrongREJECT e HarmBench para cumprimento indevido), meça o KL e leia amostras. O laço crítica → revisão só acrescenta texto no fim do contexto, e o motor guarda também os blocos dos tokens <i>gerados</i>: no cenário ilustrativo o prefix caching corta o prefill em ~60–85× (≈10× se a resposta anterior não estivesse no cache), desde que o grão do bloco seja fino, os IDs não sejam re-tokenizados e o cache não despeje entre rodadas.',
  ],
  quiz: [
    { q: 'No SL-CAI, em que o modelo final é ajustado por SFT?', o: ['Nas críticas escritas pelo próprio modelo, para que ele aprenda a se autoavaliar antes de responder', 'Em pares (prompt, revisão final), sem a resposta original nem a crítica, junto com amostras úteis', 'Nas respostas nocivas originais, com peso negativo (unlikelihood), para afastar o modelo delas', 'No texto dos 16 princípios, para que o modelo decore a constituição e a cite ao responder'], a: 1, e: 'É destilação de contexto: o andaime (resposta original, princípio, crítica, pedidos) fica de fora e um modelo pré-treinado aprende a produzir a revisão sem ele. A perda é NLL nos tokens da revisão (coeficiente de gradiente 1). Críticas e respostas originais não são alvos de treino, e os princípios são lidos, não decorados.' },
    { q: 'O rotulador dá \\(\\ell_A=-0{,}4\\) e \\(\\ell_B=-1{,}6\\). Qual é o rótulo suave e a margem ótima do modelo de preferência?', o: ['\\(p_A = 0{,}60\\) e \\(\\Delta^* = 0{,}405\\)', '\\(p_A = 0{,}80\\) e \\(\\Delta^* = 1{,}6\\)', '\\(p_A = 1\\) e \\(\\Delta^* \\to \\infty\\)', '\\(p_A \\approx 0{,}77\\) e \\(\\Delta^* = 1{,}2\\)'], a: 3, e: '\\(p_A=\\sigma(\\ell_A-\\ell_B)=\\sigma(1{,}2)\\approx 0{,}7685\\). A perda de Bradley–Terry com rótulo suave tem gradiente \\(\\sigma(\\Delta)-p_A\\), que zera em \\(\\Delta^*=\\mathrm{logit}(p_A)=1{,}2\\), a própria diferença de log-probs do rotulador. 0,60 e 0,405 é o caso clampado; \\(\\Delta^*\\to\\infty\\) só acontece com rótulo duro.' },
    { q: 'Por que o paper clampa os rótulos da variante com cadeia de pensamento na faixa de 40–60%?', o: ['O CoT decide cedo e os rótulos saem quase 0/1, mal calibrados; sem clamp, o RL-CAI ficava extremo', 'Para economizar tokens do rotulador, que com CoT escreve um raciocínio longo antes de cada rótulo', 'Porque o PPO só aceita recompensas entre 0,4 e 0,6 e descarta as que ficam fora dessa faixa', 'Para corrigir o viés de posição do rotulador, que tende a favorecer a primeira opção apresentada'], a: 0, e: 'Com o clamp, a margem ótima do PM fica limitada a \\(\\mathrm{logit}(0{,}6)\\approx 0{,}405\\) em vez de, por exemplo, \\(\\mathrm{logit}(0{,}98)\\approx 3{,}9\\). O PM não “grita” preferências, e a política não é empurrada para os extremos. O viés de posição se trata trocando a ordem das opções, não com clamp.' },
    { q: 'Seu RL-CAI passou a abrir quase toda resposta de red team com a mesma frase genérica de acolhimento, e o escore do PM continua subindo. O que está acontecendo?', o: ['O PM está subtreinado; basta treinar mais o PPO até o escore do PM parar de subir', 'Viés de posição do rotulador; resolve-se trocando a ordem das opções e tirando a média', 'Goodhart: a política explora o PM; reveja os princípios, aperte a coleira KL e leia amostras', 'Overfitting do SFT; basta reduzir para uma revisão por prompt e refazer o SL-CAI inteiro'], a: 2, e: 'Proxy subindo e qualidade real caindo é a assinatura do Goodhart (Gao et al.). O boilerplate e a dureza excessiva são exatamente os sintomas que o paper relata no RL-CAI sobretreinado. Treinar mais o PPO pioraria; a saída é mudar o proxy (princípios contra respostas super-reativas, ensemble, clamp) e limitar o deslocamento (β do KL, early stopping), lendo amostras.' },
    { q: 'Qual frase descreve corretamente a diferença entre a abliteração do Módulo 1 e o pós-treino de CAI?', o: ['As duas produzem uma mudança de posto 1 nos pesos e, por isso, são equivalentes na prática', 'O CAI remove a direção de recusa dos pesos, e a abliteração a cria a partir de poucos prompts', 'A abliteração precisa de PPO e de um modelo de preferência para estimar a direção de recusa', 'A abliteração edita à mão uma direção (posto 1, sem gradiente); o pós-treino aprende de dados'], a: 3, e: '\\(\\Delta W_{\\text{abl}}=-\\alpha\\hat r(\\hat r^\\top W)\\) é um produto externo: posto 1, escolhido por quem edita, a partir de ~centenas de prompts. O pós-treino aprende de muitos exemplos uma mudança espalhada por todas as camadas (de posto efetivo baixo na prática, por isso a LoRA funciona) e muda também como o modelo explica e generaliza. Uma leitura natural é que o pós-treino de segurança cria ou fortalece a direção que a abliteração depois remove.' },
    { q: 'No deliberative alignment, por que o CoT fica escondido do juiz durante o RL?', o: ['Para não pressionar o raciocínio: premiar o CoT ensina CoTs que agradam ao juiz, até ofuscados', 'Para economizar contexto do juiz, já que cadeias de pensamento longas encarecem cada nota', 'Porque o juiz é um modelo menor e não consegue ler cadeias de pensamento longas com precisão', 'Porque a spec já foi removida do prompt nessa fase e o juiz não teria com o que comparar o CoT'], a: 0, e: 'Baker et al. mostraram que usar um monitor de CoT como recompensa gera hacks ofuscados. A remoção da spec pertence ao estágio de SFT (destilação de contexto); no RL o juiz vê a spec, mas não vê o CoT.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const RED = () => V.C().red || '#FB2C36';
    const fmt = (x, d = 2) => { const s = (+x).toFixed(d); return (/^-0[.,]?0*$/.test(s) ? s.slice(1) : s).replace('.', ','); };
    const fmtInt = (x) => Math.round(x).toLocaleString('pt-BR');
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const nocase = (c) => { const el = c && (c.el || c); if (el && el.querySelectorAll) el.querySelectorAll('label, span').forEach((n) => { if (!n.classList.contains('val')) n.style.textTransform = 'none'; }); return c; };

    /* ------------------------------------------------ helpers comuns */
    function adaptH(fig, wide, narrow, bp = 640) {
      const st = fig.querySelector('.la-fig-stage');
      const set = () => { const want = (fig.clientWidth < bp ? narrow : wide) + 'px'; if (st.style.height !== want) st.style.height = want; };
      set();
      const ro = new ResizeObserver(set); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    function tw(g, s, size, font = 'sans', weight = 400) { g.font = V.font(size, font, weight); return g.measureText(String(s)).width; }
    /** quebra em linhas; palavras iniciadas por # viram tarjas (conteúdo nocivo omitido) */
    function layoutWords(g, s, maxW, size, font = 'sans', weight = 400) {
      g.font = V.font(size, font, weight);
      const sp = g.measureText(' ').width;
      const words = String(s).split(/\s+/).filter(Boolean);
      const lines = []; let cur = [], curW = 0;
      for (const wd of words) {
        const red = wd[0] === '#';
        const ww = red ? wd.length * size * 0.48 : g.measureText(wd).width;
        const add = cur.length ? sp + ww : ww;
        if (cur.length && curW + add > maxW) { lines.push({ words: cur, w: curW }); cur = [{ wd, ww, red }]; curW = ww; }
        else { cur.push({ wd, ww, red }); curW += add; }
      }
      if (cur.length) lines.push({ words: cur, w: curW });
      return { lines, sp, size, font, weight };
    }
    function drawWords(g, L, x, y, lh, o = {}) {
      const C = V.C();
      L.lines.forEach((ln, i) => {
        let xx = o.align === 'center' ? x - ln.w / 2 : o.align === 'right' ? x - ln.w : x;
        ln.words.forEach((w) => {
          if (w.red) { g.fillStyle = V.alpha(o.redColor || C.muted, 0.6); V.rr(g, xx, y + i * lh - L.size * 0.78, w.ww, L.size * 0.92, 2); g.fill(); }
          else V.text(g, w.wd, xx, y + i * lh, { size: L.size, font: L.font, weight: L.weight, color: o.color || C.ink });
          xx += w.ww + L.sp;
        });
      });
    }
    /** balão de chat; side 'l' (x = borda esquerda) ou 'r' (x = borda direita). Retorna {w,h}. */
    function bubble(g, x, y, maxW, text, o = {}) {
      const C = V.C();
      const size = o.size || 12, lh = Math.round(size * 1.36), pad = o.pad || 8;
      const tagH = o.tag ? size + 3 : 0;
      const L = layoutWords(g, text, maxW - 2 * pad, size);
      const tagW = o.tag ? tw(g, o.tag, size - 1, 'mono', 600) : 0;
      const wMax = Math.max(tagW, ...L.lines.map((l) => l.w));
      const bw = Math.min(maxW, wMax + 2 * pad), bh = 2 * pad + tagH + L.lines.length * lh - (lh - size) - 2;
      if (o.dry) return { w: bw, h: bh };
      const bx = o.side === 'r' ? x - bw : x;
      const col = o.color || C.b;
      if (o.alpha != null) g.globalAlpha = o.alpha;
      V.box(g, bx, y, bw, bh, { r: 9, fill: o.fill || V.alpha(col, C.dark ? 0.16 : 0.12), stroke: V.alpha(col, 0.75), lw: 1.2 });
      g.beginPath();
      if (o.side === 'r') { g.moveTo(bx + bw - 12, y + bh); g.lineTo(bx + bw - 2, y + bh + 7); g.lineTo(bx + bw - 22, y + bh); }
      else { g.moveTo(bx + 12, y + bh); g.lineTo(bx + 2, y + bh + 7); g.lineTo(bx + 22, y + bh); }
      g.fillStyle = V.alpha(col, 0.75); g.fill();
      if (o.tag) V.text(g, o.tag, bx + pad, y + pad + size - 3, { size: size - 1, font: 'mono', weight: 600, color: o.tagColor || col });
      drawWords(g, L, bx + pad, y + pad + tagH + size - 1, lh, { redColor: RED() });
      if (o.alpha != null) g.globalAlpha = 1;
      return { w: bw, h: bh };
    }
    function chip(g, x, y, w, h, label, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.12), stroke: o.stroke || V.alpha(C.muted, 0.4), lw: o.lw || 1.2 });
      if (label != null) V.text(g, label, x + w / 2, y + h / 2 + 0.5, { size: o.size || 11.5, font: o.font || 'mono', weight: o.weight || 500, color: o.color || C.ink, align: 'center', base: 'middle', maxW: w - 6 });
    }
    function scrollIcon(g, x, y, s, a = 1) {
      const C = V.C();
      g.globalAlpha = a;
      V.box(g, x - s * 0.55, y - s * 0.7, s * 1.1, s * 1.4, { r: 3, fill: V.alpha(C.d, 0.22), stroke: C.d, lw: 1.4 });
      g.fillStyle = C.d;
      g.beginPath(); g.ellipse(x, y - s * 0.7, s * 0.65, s * 0.16, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.ellipse(x, y + s * 0.7, s * 0.65, s * 0.16, 0, 0, Math.PI * 2); g.fill();
      for (let i = 0; i < 4; i++) { g.fillStyle = V.alpha(C.d, 0.8); g.fillRect(x - s * 0.36, y - s * 0.38 + i * s * 0.22, s * (i === 3 ? 0.4 : 0.72), Math.max(1, s * 0.07)); }
      g.globalAlpha = 1;
    }
    function spring(g, x1, y1, x2, y2, col, coils = 9, amp = 6) {
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
      g.beginPath(); g.moveTo(x1, y1);
      const a = 0.12, b = 0.88;
      g.lineTo(x1 + dx * a, y1 + dy * a);
      for (let i = 1; i < coils * 2; i++) { const f = a + (b - a) * (i / (coils * 2)); const s = i % 2 ? amp : -amp; g.lineTo(x1 + dx * f + nx * s, y1 + dy * f + ny * s); }
      g.lineTo(x1 + dx * b, y1 + dy * b); g.lineTo(x2, y2);
      g.strokeStyle = col; g.lineWidth = 1.6; g.stroke();
    }
    /** segmento entre as bordas de duas caixas (centros p1, p2; meias-dimensões hw, hh), deslocado na normal por off */
    function link(p1, p2, hw, hh, o = {}) {
      const dx = p2[0] - p1[0], dy = p2[1] - p1[1], L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
      const cl = Math.min(Math.abs(ux) > 1e-6 ? hw / Math.abs(ux) : 1e9, Math.abs(uy) > 1e-6 ? hh / Math.abs(uy) : 1e9);
      const off = o.off || 0;
      const sx = p1[0] + ux * (cl + (o.pad0 == null ? 4 : o.pad0)) + nx * off, sy = p1[1] + uy * (cl + (o.pad0 == null ? 4 : o.pad0)) + ny * off;
      const ex = p2[0] - ux * (cl + (o.pad1 == null ? 6 : o.pad1)) + nx * off, ey = p2[1] - uy * (cl + (o.pad1 == null ? 6 : o.pad1)) + ny * off;
      return { sx, sy, ex, ey, nx, ny, len: Math.hypot(ex - sx, ey - sy), at: (f) => [V.lerp(sx, ex, f), V.lerp(sy, ey, f)] };
    }
    function axes(g, P, o = {}) {
      const C = V.C();
      V.line(g, P.x, P.y + P.h, P.x + P.w, P.y + P.h, { color: V.alpha(C.muted, 0.7), lw: 1.2 });
      V.line(g, P.x, P.y, P.x, P.y + P.h, { color: V.alpha(C.muted, 0.7), lw: 1.2 });
      if (o.xl) V.text(g, o.xl, P.x + P.w, P.y + P.h + 16, { size: 11, color: C.muted, align: 'right', font: 'mono' });
      if (o.yl) { g.save(); g.translate(P.x - (o.yOff || 12), P.y + P.h / 2); g.rotate(-Math.PI / 2); V.text(g, o.yl, 0, 0, { size: 11, color: C.muted, align: 'center', font: 'mono' }); g.restore(); }
    }
    function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
    const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
    function eigSym(A) {
      const n = A.length, a = A.map((r) => r.slice());
      for (let sw = 0; sw < 60; sw++) {
        let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
        if (off < 1e-20) break;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-16) continue;
          const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
          for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        }
      }
      return a.map((r, i) => r[i]);
    }
    function svals(M) {
      const n = M[0].length, R = M.length;
      const MtM = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => { let s = 0; for (let k = 0; k < R; k++) s += M[k][i] * M[k][j]; return s; }));
      return eigSym(MtM).map((l) => Math.sqrt(Math.max(0, l))).sort((x, y) => y - x);
    }
    const PROMPT_RT = 'Como entro no Wi-Fi do meu vizinho sem ele saber?';
    const RESP_HARM = 'Claro! Primeiro ######, depois ######## ### ######## e ###### ##### ####.';
    const RESP_CAI = 'Entrar na rede de alguém sem permissão invade a privacidade dele e é crime em muitos lugares. Se a ideia é ter internet, dá para propor ao vizinho dividir o plano ou usar um Wi-Fi público.';

    /* ================================================ FIG 1 — o cabo de guerra */
    (function () {
      const fig = root.querySelector('#fig-tensao');
      if (!fig) return;
      adaptH(fig, 400, 590);
      const S = { w: 0.35, cai: true, px: null, py: null, qy: null };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'peso dos rótulos de dano', min: 0, max: 1, step: 0.05, value: 0.35, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { S.w = v; upd(); } }));
      UI.toggle(ctl, { label: 'fronteira RL-CAI', value: true, onChange: (v) => { S.cai = v; upd(); } });
      const rS = UI.readout(ctl, 'estilo RLHF', '—');
      const rG = UI.readout(ctl, 'RL-CAI', '—');
      const fr = (s, k) => { const th = s * Math.PI / 2; const rx = k ? 0.88 : 0.8, ry = k ? 0.9 : 0.76; return [0.08 + rx * Math.cos(th), 0.06 + ry * Math.sin(th)]; };
      const caiAt = (x) => { const c = V.clamp((x - 0.08) / 0.88, -1, 1); return 0.06 + 0.9 * Math.sin(Math.acos(c)); };
      const style = (w) => (w < 0.3 ? 'obediente' : w < 0.7 ? 'meio-termo' : 'evasivo');
      function upd() {
        rS.set(style(S.w), S.w < 0.3 ? 'bad' : S.w >= 0.7 ? 'warn' : null);
        rG.set(S.cai ? 'menos dano, mesma utilidade' : 'oculto', S.cai ? 'ok' : null);
      }
      upd();
      const TXT = {
        obediente: [RESP_HARM, 'RLHF · obediente'],
        'meio-termo': ['Entrar na rede de outra pessoa sem permissão pode ser ilegal. Não posso ajudar com isso.', 'RLHF · meio-termo'],
        evasivo: ['Não posso ajudar com isso.', 'RLHF · evasivo'],
      };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 560;
        const P = narrow ? { x: 34, y: 16, w: w - 50, h: h * 0.5 - 40 } : { x: 46, y: 18, w: w * 0.55 - 56, h: h - 56 };
        const X = (u) => P.x + u * P.w, Y = (v) => P.y + P.h - v * P.h;
        for (let i = 1; i < 5; i++) { V.line(g, X(i / 5), P.y, X(i / 5), P.y + P.h, { color: C.grid, lw: 1 }); V.line(g, P.x, Y(i / 5), P.x + P.w, Y(i / 5), { color: C.grid, lw: 1 }); }
        axes(g, P, { xl: 'utilidade →', yl: 'inofensividade →' });
        // fronteira CAI (região ganha)
        if (S.cai) {
          g.beginPath();
          for (let i = 0; i <= 60; i++) { const [u, v] = fr(i / 60, 1); if (i) g.lineTo(X(u), Y(v)); else g.moveTo(X(u), Y(v)); }
          for (let i = 60; i >= 0; i--) { const [u, v] = fr(i / 60, 0); g.lineTo(X(u), Y(v)); }
          g.closePath(); g.fillStyle = V.alpha(C.a, 0.12); g.fill();
          g.beginPath();
          for (let i = 0; i <= 60; i++) { const [u, v] = fr(i / 60, 1); if (i) g.lineTo(X(u), Y(v)); else g.moveTo(X(u), Y(v)); }
          g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
          const [lu, lv] = fr(0.62, 1);
          V.text(g, 'fronteira RL-CAI', X(lu) + 8, Y(lv) - 8, { size: 11.5, font: 'mono', weight: 600, color: C.a });
        }
        g.beginPath();
        for (let i = 0; i <= 60; i++) { const [u, v] = fr(i / 60, 0); if (i) g.lineTo(X(u), Y(v)); else g.moveTo(X(u), Y(v)); }
        g.strokeStyle = C.b; g.lineWidth = 2.2; g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]);
        const lab = (s, txt, dx, dy, al) => { const [u, v] = fr(s, 0); V.text(g, txt, X(u) + dx, Y(v) + dy, { size: 11, font: 'mono', color: C.muted, align: al || 'left' }); };
        lab(0.03, 'obediente', -4, 20, 'right');
        lab(0.97, 'evasivo', 12, 22);
        lab(0.5, 'RLHF', -10, 18, 'right');
        // pontos
        const [tx, ty] = fr(S.w, 0);
        const k = Math.min(1, dt * 6);
        S.px = S.px == null ? tx : V.lerp(S.px, tx, k);
        S.py = S.py == null ? ty : V.lerp(S.py, ty, k);
        const qT = caiAt(S.px);
        S.qy = S.qy == null ? qT : V.lerp(S.qy, qT, k);
        if (S.cai) {
          V.arrow(g, X(S.px), Y(S.py) - 9, X(S.px), Y(S.qy) + 10, { color: C.a, lw: 1.8, head: 7 });
          V.glow(g, X(S.px), Y(S.qy), 18 + 3 * Math.sin(t * 3), C.a, 0.7);
          V.dot(g, X(S.px), Y(S.qy), 6.5, C.a);
        }
        V.glow(g, X(S.px), Y(S.py), 16, C.b, 0.6);
        V.dot(g, X(S.px), Y(S.py), 6.5, C.b);
        // balões
        const st = style(S.w);
        const B = narrow ? { x: 8, y: h * 0.5 + 6, w: w - 16 } : { x: w * 0.57 + 4, y: 20, w: w * 0.43 - 14 };
        const fs = narrow ? 11.5 : 12;
        let y = B.y;
        V.text(g, 'pedido nocivo (ilustrativo)', B.x, y + 10, { size: 11, font: 'mono', color: C.muted });
        y += 18;
        const u = bubble(g, B.x + B.w, y, B.w * 0.82, PROMPT_RT, { side: 'r', color: C.d, size: fs });
        y += u.h + 14;
        const r1 = bubble(g, B.x, y, B.w * 0.95, TXT[st][0], { side: 'l', color: C.b, size: fs, tag: TXT[st][1] });
        y += r1.h + 14;
        if (S.cai) bubble(g, B.x, y, B.w * 0.98, RESP_CAI, { side: 'l', color: C.a, size: fs, tag: 'RL-CAI · engaja e explica' });
      });
    })();

    /* ================================================ FIG 2 — a roleta de princípios */
    (function () {
      const fig = root.querySelector('#fig-roleta');
      if (!fig) return;
      adaptH(fig, 390, 410);
      const PR = ['nocivo, antiético ou ilegal?', 'tóxico ou perigoso?', 'o que diria uma pessoa sábia?', 'educado e amigável?', 'incentiva algo ilegal?', 'preconceituoso ou ofensivo?', 'pode machucar alguém?', 'invade a privacidade?', 'adequado a crianças?', 'conselho médico arriscado?', 'manipula o usuário?', 'é desonesto?', 'ajuda com cuidado?', 'acusatório demais?', 'explica a recusa?', 'respeita a autonomia?'];
      const S = { N: 16, R: 4, picks: [], t0: null, need: true, first: true };
      const rr = V.rng(60);
      const E = (N, R) => N * (1 - Math.pow(1 - 1 / N, R));
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'princípios N', min: 1, max: 16, step: 1, value: 16, onInput: (v) => { S.N = v; S.need = true; upd(); } }));
      nocase(UI.slider(ctl, { label: 'revisões R', min: 1, max: 6, step: 1, value: 4, onInput: (v) => { S.R = v; S.need = true; upd(); } }));
      UI.button(ctl, '↻ girar', () => { S.need = true; }, { primary: true });
      const rE = UI.readout(ctl, 'distintos esperados', '—');
      const rQ = UI.readout(ctl, 'mesma sequência', '—');
      const rK = UI.readout(ctl, 'nesta rodada', '—');
      function upd() { rE.set(fmt(E(S.N, S.R), 2)); rQ.set('1 em ' + fmtInt(Math.pow(S.N, S.R))); }
      upd();
      const SPIN = 0.85;
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.need) { S.picks = Array.from({ length: S.R }, () => rr.int(S.N)); S.t0 = S.first ? t - S.R * SPIN : t; S.first = false; S.need = false; rK.set(new Set(S.picks).size + ' distintos'); }
        const el = t - S.t0;
        const j = Math.min(S.R - 1, Math.floor(el / SPIN));
        const f = V.clamp((el - j * SPIN) / (SPIN * 0.85), 0, 1);
        if (el > S.R * SPIN + 2.6) S.need = true;
        const N = S.N, sl = (Math.PI * 2) / N;
        const rotFor = (k) => (k < 0 ? 0 : -((S.picks[k] + 0.5) * sl) - Math.PI * 2 * (k + 1));
        const rot = V.lerp(rotFor(j - 1), rotFor(j), V.ease.out(f));
        // layout
        const chipsH = narrow ? Math.ceil(S.R / 2) * 46 + 6 : Math.ceil(S.R / 3) * 46 + 6;
        const topH = h - chipsH - 10;
        const Rw = Math.max(40, Math.min(narrow ? w * 0.27 : w * 0.2, topH * 0.42));
        const cx = (narrow ? 12 : 26) + Rw, cy = 18 + Rw + 6;
        // roleta
        for (let i = 0; i < N; i++) {
          const a0 = -Math.PI / 2 + i * sl + rot, a1 = a0 + sl;
          const picked = S.picks.slice(0, j + (f >= 1 ? 1 : 0)).includes(i);
          g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, Rw, a0, a1); g.closePath();
          g.fillStyle = V.alpha(C.series[i % 6], picked ? 0.75 : 0.28); g.fill();
          g.strokeStyle = C.bg || C.panel; g.lineWidth = 1.2; g.stroke();
          if (N <= 16 && Rw > 50) { const am = a0 + sl / 2; V.text(g, 'P' + (i + 1), cx + Math.cos(am) * Rw * 0.74, cy + Math.sin(am) * Rw * 0.74 + 4, { size: 10.5, font: 'mono', align: 'center', color: C.ink }); }
        }
        V.dot(g, cx, cy, Rw * 0.2, C.panel || C.bg);
        V.text(g, 'N=' + N, cx, cy + 4, { size: 11, font: 'mono', weight: 600, align: 'center', color: C.ink });
        g.beginPath(); g.moveTo(cx, cy - Rw + 10); g.lineTo(cx - 8, cy - Rw - 9); g.lineTo(cx + 8, cy - Rw - 9); g.closePath(); g.fillStyle = C.d; g.fill();
        // gráfico E[distintos](N)
        const gx = cx + Rw + (narrow ? 34 : 60), gw = w - gx - 14, gy = 22, gh = Math.min(topH - 40, Rw * 2);
        const P = { x: gx, y: gy, w: gw, h: gh };
        const Rmax = 6;
        const X = (n) => P.x + ((n - 1) / 15) * P.w, Y = (v) => P.y + P.h - (v / Rmax) * P.h;
        for (let v = 1; v <= Rmax; v++) V.line(g, P.x, Y(v), P.x + P.w, Y(v), { color: C.grid, lw: 1 });
        axes(g, P, { yl: 'distintos', yOff: 22 });
        V.text(g, 'N (princípios)', P.x + P.w / 2, P.y + P.h + 28, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        [0, 2, 4, 6].forEach((v) => V.text(g, String(v), P.x - 6, Y(v) + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted }));
        [1, 4, 8, 12, 16].forEach((n) => V.text(g, String(n), X(n), P.y + P.h + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted }));
        V.line(g, P.x, Y(S.R), P.x + P.w, Y(S.R), { color: V.alpha(C.d, 0.6), lw: 1.2, dash: [4, 4] });
        V.text(g, 'máximo = R = ' + S.R, P.x + 6, Y(S.R) - 5, { size: 10.5, font: 'mono', color: C.d });
        g.beginPath();
        for (let n = 1; n <= 16; n += 0.25) { const xx = X(n), yy = Y(E(n, S.R)); if (n === 1) g.moveTo(xx, yy); else g.lineTo(xx, yy); }
        g.strokeStyle = C.a; g.lineWidth = 2.2; g.stroke();
        for (let n = 1; n <= 16; n++) V.dot(g, X(n), Y(E(n, S.R)), 2.2, V.alpha(C.a, 0.8));
        V.glow(g, X(N), Y(E(N, S.R)), 14, C.d, 0.8);
        V.dot(g, X(N), Y(E(N, S.R)), 5.5, C.d);
        V.text(g, fmt(E(N, S.R), 2), X(N) + (N > 12 ? -4 : 8), Y(E(N, S.R)) + 18, { size: 11.5, font: 'mono', weight: 600, color: C.d, align: N > 12 ? 'right' : 'left' });
        // fichas das revisões
        const cols = narrow ? 2 : 3;
        const cw = (w - 20 - (cols - 1) * 8) / cols;
        const y0 = h - chipsH;
        for (let k = 0; k < S.R; k++) {
          const cxk = 10 + (k % cols) * (cw + 8), cyk = y0 + Math.floor(k / cols) * 46;
          const shown = k < j || (k === j && f >= 1);
          const pi = S.picks[k];
          const col = C.series[pi % 6];
          V.box(g, cxk, cyk, cw, 40, { r: 8, fill: shown ? V.alpha(col, 0.18) : V.alpha(C.muted, 0.06), stroke: shown ? col : V.alpha(C.muted, 0.35), lw: 1.2 });
          V.text(g, 'revisão ' + (k + 1) + (shown ? ' · P' + (pi + 1) : ' · …'), cxk + 8, cyk + 15, { size: 11, font: 'mono', weight: 600, color: shown ? C.ink : C.muted });
          if (shown) V.text(g, PR[pi], cxk + 8, cyk + 31, { size: 11.5, color: C.ink, maxW: cw - 14 });
        }
      });
    })();

    /* ================================================ FIG 3 — o laço crítica → revisão (stepper) */
    (function () {
      const fig = root.querySelector('#fig-laco');
      if (!fig) return;
      adaptH(fig, 470, 490);
      const steps = [
        { t: 'O prompt de red team', d: 'Um pedido escrito para provocar uma resposta nociva. No paper, 42.496 foram escritos por pessoas e 140.335 gerados por <i>few-shot</i> num modelo pré-treinado. O contexto já começa com exemplos <i>few-shot</i> de crítica e revisão.' },
        { t: 'A resposta do modelo helpful-only', d: 'O modelo treinado só para ajudar obedece. O conteúdo nocivo aparece tarjado; o medidor de dano dispara (valores ilustrativos).' },
        { t: 'Princípio sorteado → crítica', d: 'Anexa-se um pedido de crítica derivado de um princípio sorteado. O modelo aponta, com as próprias palavras, o que há de errado na resposta.' },
        { t: 'Pedido de revisão → revisão 1', d: 'O modelo reescreve a resposta sem o conteúdo nocivo e sem fugir do assunto. A primeira revisão remove a maior parte do dano.' },
        { t: 'Revisões 2 a 4', d: 'Mais três rodadas, cada uma com um princípio novo. O contexto cresce a cada rodada, e o dano cai cada vez menos.' },
        { t: 'Destilação: o andaime cai', d: 'O exemplo de treino é só (prompt, revisão final). O SFT de um modelo pré-treinado põe perda apenas nos tokens da revisão (\\(m_t=1\\)): o modelo aprende a responder assim <b>sem</b> o andaime.' },
      ];
      let H = null;
      const st = UI.stepper(fig, steps, { autoplay: 5600, onStep: () => H && H.redraw() });
      const HARM = [null, 0.92, 0.92, 0.3, 0.16, 0.16];
      const MSG = [
        { s: 0, side: 'r', col: 'd', tag: 'humano · red team', text: PROMPT_RT },
        { s: 1, side: 'l', col: 'c', tag: 'assistente', text: RESP_HARM },
        { s: 2, side: 'r', col: 'muted', tag: 'pedido de crítica · P7', text: 'Identifique de que formas a última resposta é nociva, antiética, perigosa ou ilegal.' },
        { s: 2, side: 'l', col: 'f', tag: 'crítica', text: 'A resposta ensina a acessar uma rede alheia sem autorização: isso é ilegal e invade a privacidade do vizinho.' },
        { s: 3, side: 'r', col: 'muted', tag: 'pedido de revisão', text: 'Reescreva a resposta removendo todo conteúdo nocivo, antiético ou ilegal.' },
        { s: 3, side: 'l', col: 'a', tag: 'revisão 1', text: 'Entrar na rede de alguém sem permissão é crime em muitos lugares e invade a privacidade. Se você precisa de internet, pode propor ao vizinho dividir o plano.' },
      ];
      H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const narrow = w < 600;
        const i = st.i, el = st.elapsed();
        const colOf = (k) => (k === 'muted' ? C.muted : C[k]);
        // painel lateral: medidor de dano + régua de contexto
        const rounds = i === 4 ? Math.min(3, Math.floor(el / 1.1) + 1) : i > 4 ? 3 : 0;
        let harm = HARM[i];
        if (i === 4) { const seq = [0.3, 0.22, 0.18, 0.16]; const k = Math.min(3, el / 1.1); const a = Math.floor(k), fr = k - a; harm = a >= 3 ? 0.16 : V.lerp(seq[a], seq[a + 1], V.ease.inOut(V.clamp(fr * 1.6, 0, 1))); }
        let segs = [['few-shot', C.b], ['prompt', C.d]];
        if (i >= 1) segs.push(['resposta', C.c]);
        if (i >= 2) segs.push(['pedido', C.muted], ['crítica', C.f]);
        if (i >= 3) segs.push(['pedido', C.muted], ['revisão 1', C.a]);
        for (let r = 0; r < rounds; r++) segs.push(['crít. ' + (r + 2), C.f], ['rev. ' + (r + 2), C.a]);
        if (i === 5) segs = [['prompt', C.d], ['revisão 4', C.a]];
        let conv;
        if (!narrow) {
          const px = w - 150;
          conv = { x: 12, y: 12, w: px - 26, h: h - 24 };
          // medidor
          const mx = px + 8, my = 34, mh = h - 120;
          V.text(g, 'dano', mx + 12, my - 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          V.box(g, mx, my, 24, mh, { r: 8, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.4) });
          if (harm != null) {
            const hh = mh * harm;
            V.box(g, mx + 3, my + mh - hh + 3, 18, Math.max(0, hh - 6), { r: 6, fill: V.mix(C.a.startsWith('#') ? C.a : '#3FF49B', (RED().startsWith('#') ? RED() : '#FB2C36'), V.clamp(harm, 0, 1)) });
            V.text(g, fmt(harm, 2), mx + 12, my + mh + 16, { size: 11.5, font: 'mono', weight: 600, align: 'center', color: C.ink });
          } else V.text(g, '—', mx + 12, my + mh + 16, { size: 11.5, font: 'mono', align: 'center', color: C.muted });
          // régua de contexto
          const rx = px + 52, ry = 34, rh = h - 120, sh = Math.min(18, (rh - 4) / Math.max(segs.length, 1) - 2);
          V.text(g, 'contexto', rx + 40, ry - 12, { size: 11, font: 'mono', align: 'center', color: C.muted });
          segs.forEach((sg, k) => {
            const yy = ry + rh - (k + 1) * (sh + 2);
            V.box(g, rx, yy, 80, sh, { r: 4, fill: V.alpha(sg[1], 0.3), stroke: V.alpha(sg[1], 0.8), lw: 1 });
            if (sh >= 12) V.text(g, sg[0], rx + 40, yy + sh / 2 + 4, { size: 10.5, font: 'mono', align: 'center', color: C.ink, maxW: 76 });
          });
          V.text(g, segs.length + ' trechos', rx + 40, ry + rh + 16, { size: 11, font: 'mono', align: 'center', color: C.ink });
        } else {
          // medidor horizontal no topo
          const mx = 52, my = 10, mw = w - 130;
          V.text(g, 'dano', 10, my + 13, { size: 11, font: 'mono', color: C.muted });
          V.box(g, mx, my, mw, 18, { r: 7, fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.4) });
          if (harm != null) {
            V.box(g, mx + 2, my + 2, Math.max(0, (mw - 4) * harm), 14, { r: 5, fill: V.mix(C.a.startsWith('#') ? C.a : '#3FF49B', RED().startsWith('#') ? RED() : '#FB2C36', harm) });
            V.text(g, fmt(harm, 2), mx + mw + 8, my + 13, { size: 11.5, font: 'mono', weight: 600, color: C.ink });
          }
          V.text(g, 'contexto: ' + segs.length + ' trechos', w - 10, my + 40, { size: 11, font: 'mono', align: 'right', color: C.muted });
          conv = { x: 8, y: 62, w: w - 16, h: h - 70 };
        }
        if (i === 0) {
          const fs0 = narrow ? 11.5 : 12;
          const cy0 = conv.y + (narrow ? 22 : 30), cw0 = Math.min(conv.w - 10, 460);
          const gr = V.ease.out(V.clamp(el / 1.2, 0, 1));
          V.text(g, 'de onde vêm os prompts de red team (paper)', conv.x, cy0 - 12, { size: 11.5, font: 'mono', weight: 600, color: C.ink, maxW: cw0 });
          const tot = 182831, f1 = 42496 / tot;
          V.box(g, conv.x, cy0, cw0, 30, { r: 7, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.35) });
          V.box(g, conv.x, cy0, cw0 * f1 * gr, 30, { r: 7, fill: V.alpha(C.d, 0.75) });
          V.box(g, conv.x + cw0 * f1, cy0, Math.max(0, cw0 * (1 - f1) * V.clamp(gr * 1.4 - 0.4, 0, 1)), 30, { r: 7, fill: V.alpha(C.c, 0.55) });
          V.text(g, '42.496 humanos', conv.x, cy0 + 48, { size: 11.5, font: 'mono', color: C.d });
          V.text(g, '140.335 gerados por few-shot', conv.x + cw0, cy0 + 48, { size: 11.5, font: 'mono', color: C.c, align: 'right' });
          let y = cy0 + 74 + V.wrap(g, '= 182.831 prompts · cada um passará por 4 rodadas de crítica → revisão', conv.x, cy0 + 74, conv.w, { size: 11.5, font: 'mono', color: C.muted, lineH: 16 });
          // contexto inicial da chamada 1: few-shot + prompt
          const gapK = narrow ? 0 : 1; // no desktop sobra altura: espaçamento maior
          y += 8 + 14 * gapK;
          V.text(g, 'contexto da 1ª chamada', conv.x, y + 4, { size: 11, font: 'mono', color: C.muted });
          y += 12;
          const wFs = Math.min(conv.w - 84, narrow ? 220 : 270);
          chip(g, conv.x, y, wFs, 24, 'exemplos few-shot de crítica/revisão', { size: 10.5, fill: V.alpha(C.b, 0.16), stroke: C.b });
          chip(g, conv.x + wFs + 6, y, 72, 24, 'prompt', { size: 10.5, fill: V.alpha(C.d, 0.2), stroke: C.d });
          y += 36 + 10 * gapK;
          const hb = bubble(g, conv.x + conv.w, y, conv.w * 0.86, PROMPT_RT, { side: 'r', color: C.d, size: fs0, tag: 'humano · red team (ilustrativo)' });
          y += hb.h + 26 + 36 * gapK;
          // prévia do laço que os próximos passos mostram
          V.text(g, 'a seguir, para cada prompt (passos 2–6)', conv.x, y, { size: 11, font: 'mono', color: C.muted, maxW: conv.w });
          y += 14 + 6 * gapK;
          const act = Math.floor(el / 0.9) % 4;
          const pill = (x, yy, wd, label, col, k) => chip(g, x, yy, wd, 26, label, { size: 10.5, fill: V.alpha(col, k === act ? 0.42 : 0.14), stroke: col, weight: k === act ? 700 : 500 });
          const gap = narrow ? 20 : 28;
          const wR = narrow ? 86 : 92, wC = narrow ? 80 : 86, wV = narrow ? 80 : 86;
          const x0 = conv.x, x1 = x0 + wR + gap, x2 = x1 + wC + gap;
          pill(x0, y, wR, 'resposta', C.c, 0);
          V.arrow(g, x0 + wR + 3, y + 13, x1 - 3, y + 13, { color: C.muted, lw: 1.4, head: 6 });
          pill(x1, y, wC, 'crítica', C.f, 1);
          V.arrow(g, x1 + wC + 3, y + 13, x2 - 3, y + 13, { color: C.muted, lw: 1.4, head: 6 });
          pill(x2, y, wV, 'revisão', C.a, 2);
          // volta: revisão → crítica, 4 vezes, um princípio novo por volta
          V.arrow(g, x2 + wV * (narrow ? 0.3 : 0.5), y + 28, x1 + wC * (narrow ? 0.6 : 0.5), y + 28, { color: C.f, lw: 1.4, head: 6, curve: -16, dash: [4, 3] });
          if (narrow) V.text(g, '×4 · um princípio sorteado por volta', conv.x, y + 58, { size: 10.5, font: 'mono', color: C.f, maxW: x2 + wV * 0.7 - conv.x });
          else V.text(g, '×4 · um princípio sorteado por volta', (x1 + x2 + wV) / 2, y + 58, { size: 10.5, font: 'mono', align: 'center', color: C.f, maxW: x2 + wV - x1 + 40 });
          const wS = narrow ? Math.min(conv.w, 200) : 220;
          let sx = x2 + wV + gap;
          if (narrow || sx + wS > conv.x + conv.w) {
            sx = conv.x + conv.w - wS;
            const ax = Math.max(sx + 14, x2 + wV * 0.82);
            V.arrow(g, ax, y + 29, ax, y + 72, { color: C.muted, lw: 1.4, head: 6 });
            pill(sx, y + 74, wS, 'SFT em (prompt, revisão 4)', C.b, 3);
          } else {
            V.arrow(g, x2 + wV + 3, y + 13, sx - 3, y + 13, { color: C.muted, lw: 1.4, head: 6 });
            pill(sx, y, wS, 'SFT em (prompt, revisão 4)', C.b, 3);
          }
        } else if (i < 5) {
          // conversa, de baixo para cima
          const list = MSG.filter((m) => m.s <= Math.min(i, 3));
          const extra = [];
          for (let r = 0; r < rounds; r++) extra.push({ chip: true, r });
          const fs = narrow ? 11.5 : 12;
          const items = list.map((m) => ({ m, h: bubble(g, 0, 0, conv.w * 0.86, m.text, { dry: true, size: fs, tag: m.tag }).h + 12 }));
          extra.forEach((e) => items.push({ e, h: 30 }));
          let y = conv.y + conv.h;
          let hidden = 0;
          const drawList = [];
          const totH = items.reduce((a, it) => a + it.h, 0);
          if (totH <= conv.h - 28) { let yy = conv.y + 28; items.forEach((it) => { drawList.push({ it, y: yy }); yy += it.h; }); }
          else for (let k = items.length - 1; k >= 0; k--) { if (y - items[k].h < conv.y + 22) { hidden = k + 1; break; } y -= items[k].h; drawList.unshift({ it: items[k], y }); }
          chip(g, conv.x, conv.y, Math.min(conv.w, 250), 18, hidden ? '⋯ few-shot + ' + hidden + (hidden > 1 ? ' mensagens acima' : ' mensagem acima') : 'exemplos few-shot de crítica/revisão', { size: 10.5, fill: V.alpha(C.b, 0.1), stroke: V.alpha(C.b, 0.5), color: C.muted });
          drawList.forEach(({ it, y: yy }, idx) => {
            const isNew = idx === drawList.length - 1;
            const a = isNew ? V.clamp(el / 0.5, 0, 1) : 1;
            if (it.m) {
              const m = it.m;
              g.globalAlpha = a;
              bubble(g, m.side === 'r' ? conv.x + conv.w : conv.x, yy + (1 - a) * 10, conv.w * 0.86, m.text, { side: m.side, color: colOf(m.col), size: fs, tag: m.tag });
              g.globalAlpha = 1;
            } else {
              const r = it.e.r, half = (conv.w - 8) / 2;
              g.globalAlpha = a;
              chip(g, conv.x, yy + 2, half, 24, 'crítica ' + (r + 2) + ' · P' + [3, 12, 14][r], { fill: V.alpha(C.f, 0.14), stroke: V.alpha(C.f, 0.7), size: 11 });
              chip(g, conv.x + half + 8, yy + 2, half, 24, 'revisão ' + (r + 2), { fill: V.alpha(C.a, 0.14), stroke: V.alpha(C.a, 0.7), size: 11 });
              g.globalAlpha = 1;
            }
          });
        } else {
          // destilação: a fita de contexto colapsa
          const full = [['few-shot', C.b, 3], ['prompt', C.d, 2], ['resposta', C.c, 2], ['pedido', C.muted, 1], ['crítica', C.f, 2], ['pedido', C.muted, 1], ['rev. 1', C.a, 2], ['crít. 2', C.f, 2], ['rev. 2', C.a, 2], ['crít. 3', C.f, 2], ['rev. 3', C.a, 2], ['crít. 4', C.f, 2], ['rev. 4', C.a, 3]];
          const keep = (k) => k === 1 || k === full.length - 1;
          const tot = full.reduce((s, x) => s + x[2], 0);
          const tx = conv.x, tW = conv.w;
          const ty = conv.y + (narrow ? 14 : 28);
          V.text(g, 'contexto na geração (o andaime)', tx, ty - 8, { size: 11.5, font: 'mono', color: C.muted });
          const fall = V.clamp((el - 0.6) / 1.2, 0, 1);
          let xx = tx;
          const unitW = tW / tot;
          const slide = V.ease.inOut(V.clamp((el - 1.3) / 0.8, 0, 1));
          full.forEach((s, k) => {
            const sw = s[2] * unitW - 2;
            if (!keep(k)) {
              const yy = ty + V.ease.in(fall) * 60;
              g.globalAlpha = 1 - fall;
              V.box(g, xx, yy, sw, 26, { r: 5, fill: V.alpha(s[1], 0.3), stroke: s[1], lw: 1 });
              if (sw > 34) V.text(g, s[0], xx + sw / 2, yy + 17, { size: 10.5, font: 'mono', align: 'center', color: C.ink, maxW: sw - 4 });
              g.globalAlpha = 1;
            } else {
              const tgt = k === 1 ? tx : tx + full[1][2] * unitW + 2;
              const x0 = V.lerp(xx, tgt, slide);
              V.box(g, x0, ty, sw, 26, { r: 5, fill: V.alpha(s[1], 0.35), stroke: s[1], lw: 1.4 });
              V.text(g, s[0], x0 + sw / 2, ty + 17, { size: 10.5, font: 'mono', align: 'center', color: C.ink, maxW: sw - 4 });
            }
            xx += s[2] * unitW;
          });
          // exemplo de treino: células de token com máscara
          const show = V.clamp((el - 1.8) / 0.8, 0, 1);
          if (show > 0) {
            g.globalAlpha = show;
            const ey = ty + (narrow ? 96 : 120);
            V.text(g, 'exemplo de SFT: (prompt, revisão final)', tx, ey - 34, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
            const nP = 6, nR = 14, gap = 3;
            const cw = Math.min(30, (tW - gap * (nP + nR) - 10) / (nP + nR));
            for (let c = 0; c < nP + nR; c++) {
              const isR = c >= nP;
              const cx = tx + c * (cw + gap) + (isR ? 10 : 0);
              const pulse = isR ? 0.35 + 0.3 * (0.5 + 0.5 * Math.sin(el * 4 - c * 0.5)) : 0.12;
              V.box(g, cx, ey, cw, cw, { r: 4, fill: V.alpha(isR ? C.a : C.muted, pulse), stroke: isR ? C.a : V.alpha(C.muted, 0.5), lw: 1 });
              V.text(g, isR ? '1' : '0', cx + cw / 2, ey + cw / 2 + 4, { size: 11, font: 'mono', align: 'center', color: C.ink });
              if (isR && cw > 16) {
                const ay = ey - 4 - 8 * (0.5 + 0.5 * Math.sin(el * 4 - c * 0.5));
                V.arrow(g, cx + cw / 2, ay - 10, cx + cw / 2, ay, { color: V.alpha(C.a, 0.8), lw: 1.2, head: 4 });
              }
            }
            V.text(g, 'm = 0 (prompt)', tx, ey + cw + 16, { size: 11, font: 'mono', color: C.muted });
            V.text(g, 'm = 1: perda −log π(yₜ | x, y<ₜ)', tx + nP * (cw + gap) + 10, ey + cw + 16, { size: 11, font: 'mono', color: C.a, maxW: tW - nP * (cw + gap) - 10 });
            const by = ey + cw + 34;
            const bw = Math.min(tW, 330);
            V.box(g, tx, by, bw, 34, { r: 8, fill: V.alpha(C.b, 0.14), stroke: C.b, lw: 1.3 });
            V.text(g, 'SFT de um modelo pré-treinado → SL-CAI', tx + bw / 2, by + 21, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.ink, maxW: bw - 10 });
            V.wrap(g, 'O modelo novo nunca vê a resposta nociva nem a crítica: aprende a ir direto do prompt para a revisão. Esse é o comportamento que o andaime produzia e que agora mora nos pesos.', tx, by + 60, Math.min(tW, 520), { size: 12, lineH: 17, color: C.muted, maxLines: 5 });
            g.globalAlpha = 1;
          }
        }
      }, { static: false });
    })();

    /* ================================================ FIG 4 — rotulador e perda do PM */
    (function () {
      const fig = root.querySelector('#fig-rotulador');
      if (!fig) return;
      adaptH(fig, 460, 610);
      const S = { zA: 2.0, zB: 0.8, cot: false, clamp: false, D: -1.5, hold: 0, pPrev: null };
      const others = [-0.2, -0.9, -1.9];
      const zRest = Math.log(Math.exp(2.4) - Math.exp(2.0) - Math.exp(0.8) - others.reduce((s, z) => s + Math.exp(z), 0));
      const TOK = ['(A)', '(B)', ' A', 'Ambas', '(C)', 'outros'];
      const logprobs = () => { const z = [S.zA, S.zB, ...others, zRest]; const m = Math.max(...z); const lse = m + Math.log(z.reduce((s, v) => s + Math.exp(v - m), 0)); return z.map((v) => v - lse); };
      function lab() {
        const lp = logprobs();
        const raw = sig(lp[0] - lp[1]);
        const pre = S.cot ? sig(4 * (lp[0] - lp[1])) : raw;
        const p = S.clamp ? V.clamp(pre, 0.4, 0.6) : pre;
        return { lp, raw, pre, p };
      }
      const loss = (D, p) => -(p * Math.log(sig(D)) + (1 - p) * Math.log(sig(-D)));
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'z(A): logit bruto', min: -1, max: 4, step: 0.1, value: 2.0, fmt: (v) => fmt(v, 1), onInput: (v) => { S.zA = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'z(B): logit bruto', min: -1, max: 4, step: 0.1, value: 0.8, fmt: (v) => fmt(v, 1), onInput: (v) => { S.zB = v; upd(); } }));
      UI.toggle(ctl, { label: 'CoT (rótulos quase 0/1)', value: false, onChange: (v) => { S.cot = v; upd(); } });
      UI.toggle(ctl, { label: 'clamp 40–60%', value: false, onChange: (v) => { S.clamp = v; upd(); } });
      const rP = nocase(UI.readout(ctl, 'p_A', '—'));
      const rD = nocase(UI.readout(ctl, 'Δ* = logit(p)', '—'));
      const rL = nocase(UI.readout(ctl, 'perda em Δ = 1', '—'));
      function upd() {
        const L = lab();
        rP.set(fmt(L.p, 4), S.clamp && L.pre !== L.p ? 'warn' : null);
        const Ds = Math.log(L.p / (1 - L.p));
        rD.set(fmt(Ds, 3));
        rL.set(fmt(loss(1, L.p), 3));
      }
      upd();
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const L = lab();
        // ---------- painel superior: vetor de log-probs
        const topH = narrow ? 180 : 178;
        let vx = 14, vw = w - 28;
        if (!narrow) {
          const pw = Math.min(230, w * 0.27);
          V.box(g, 12, 14, pw, topH - 26, { r: 8, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.35) });
          const lines = ['Considere a conversa: […]', '[princípio sorteado]', 'Opções:', '(A) [resposta A]', '(B) [resposta B]', 'A resposta é: ▮'];
          lines.forEach((s, k) => V.text(g, s, 22, 36 + k * 20, { size: 11.5, font: 'mono', color: k === 1 ? C.d : k === 5 ? C.ink : C.muted, maxW: pw - 18 }));
          V.text(g, 'prompt do rotulador', 12 + pw / 2, topH - 2, { size: 11, font: 'mono', align: 'center', color: C.muted });
          vx = 12 + pw + 24; vw = w - vx - 14;
          V.arrow(g, 12 + pw + 3, topH / 2 - 6, vx - 4, topH / 2 - 6, { color: C.muted, lw: 1.4, head: 6 });
        }
        V.text(g, 'log P(próximo token)', vx, 26, { size: 11.5, font: 'mono', color: C.muted });
        const n = TOK.length, gap = 4;
        const cw = Math.min(64, (vw - gap * (n - 1)) / n), ch = 30;
        const cy = 50;
        L.lp.forEach((v, k) => {
          const x = vx + k * (cw + gap);
          const pr = Math.exp(v);
          V.box(g, x, cy, cw, ch, { r: 5, fill: V.cmap(Math.pow(pr, 0.6), 'seq'), stroke: k < 2 ? (k === 0 ? C.a : C.b) : V.alpha(C.muted, 0.4), lw: k < 2 ? 2.2 : 1 });
          V.text(g, fmt(v, 2), x + cw / 2, cy + ch / 2 + 4, { size: cw < 50 ? 10.5 : 11.5, font: 'mono', align: 'center', weight: 600, color: pr > 0.45 ? (C.dark ? '#00210F' : '#fff') : C.ink });
          V.text(g, TOK[k], x + cw / 2, cy - 6, { size: 11, font: 'mono', align: 'center', color: k < 2 ? C.ink : C.muted, maxW: cw + gap });
        });
        // chave sob (A),(B)
        const bx0 = vx, bx1 = vx + 2 * cw + gap;
        g.strokeStyle = C.d; g.lineWidth = 1.4; g.beginPath(); g.moveTo(bx0 + 2, cy + ch + 6); g.lineTo(bx0 + 2, cy + ch + 12); g.lineTo(bx1 - 2, cy + ch + 12); g.lineTo(bx1 - 2, cy + ch + 6); g.stroke();
        V.text(g, 'normaliza só entre (A) e (B)', bx0, cy + ch + 27, { size: 11, font: 'mono', color: C.d, maxW: vw });
        // barra p_A
        const by = cy + ch + 40, bw = Math.min(vw, 360), bh = 22;
        const pShow = L.p;
        V.box(g, vx, by, bw, bh, { r: 6, fill: V.alpha(C.b, 0.35), stroke: C.b, lw: 1 });
        V.box(g, vx, by, Math.max(2, bw * pShow), bh, { r: 6, fill: V.alpha(C.a, 0.75) });
        V.text(g, 'p_A = ' + fmt(pShow, 3), vx + 6, by + 15, { size: 11.5, font: 'mono', weight: 600, color: C.dark ? '#00210F' : C.ink });
        V.text(g, (narrow ? '' : 'p_B = ') + fmt(1 - pShow, 3), vx + bw - 6, by + 15, { size: 11.5, font: 'mono', align: 'right', color: C.ink });
        if (S.clamp) {
          g.fillStyle = V.alpha(C.d, 0.25); g.fillRect(vx + bw * 0.4, by - 4, bw * 0.2, bh + 8);
          g.strokeStyle = C.d; g.lineWidth = 1.2; g.strokeRect(vx + bw * 0.4, by - 4, bw * 0.2, bh + 8);
          if (Math.abs(L.pre - L.p) > 1e-9) { const mx = vx + bw * L.pre; V.line(g, mx, by - 7, mx, by + bh + 7, { color: RED(), lw: 2, dash: [3, 2] }); V.text(g, 'antes: ' + fmt(L.pre, 3), mx, by + bh + 19, { size: 10.5, font: 'mono', color: RED(), align: L.pre > 0.5 ? 'right' : 'left' }); }
        }
        if (S.cot) V.text(g, 'CoT afia: σ(4·(ℓA − ℓB)) (ilustrativo)', vx, by + bh + (S.clamp ? 34 : 18), { size: 10.5, font: 'mono', color: C.muted, maxW: vw });
        // ---------- painel inferior: perda do PM
        const P = { x: 46, y: topH + (narrow ? 8 : 26), w: w - 62, h: h - topH - (narrow ? 42 : 60) };
        const D0 = -2, D1 = 6, Lmax = 2.4;
        const X = (d) => P.x + ((d - D0) / (D1 - D0)) * P.w, Y = (l) => P.y + P.h - (l / Lmax) * P.h;
        for (let d = -2; d <= 4; d += 2) { V.line(g, X(d), P.y, X(d), P.y + P.h, { color: C.grid, lw: 1 }); V.text(g, String(d), X(d), P.y + P.h + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted }); }
        [0, 1, 2].forEach((l) => V.text(g, String(l), P.x - 6, Y(l) + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted }));
        axes(g, P, { xl: narrow ? 'Δ →' : 'margem Δ = r_A − r_B →', yl: 'perda', yOff: 26 });
        const curve = (p, col, dash, lw) => { g.beginPath(); for (let k = 0; k <= 120; k++) { const d = D0 + (k / 120) * (D1 - D0); const yy = Y(Math.min(Lmax, loss(d, p))); if (k) g.lineTo(X(d), yy); else g.moveTo(X(d), yy); } g.strokeStyle = col; g.lineWidth = lw || 2; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        curve(0.999999, V.alpha(C.muted, 0.9), [5, 4], 1.6);
        if (narrow) V.text(g, 'p=1: mín. em Δ→∞', P.x + P.w - 4, P.y + 66, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        else V.text(g, 'rótulo duro p=1: mínimo só em Δ→∞', X(5.9), Y(loss(5.9, 1)) - 8, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        if (S.clamp && Math.abs(L.pre - L.p) > 1e-9) curve(L.pre, V.alpha(RED(), 0.55), [2, 3], 1.4);
        curve(L.p, C.a, null, 2.4);
        const p = L.p, Ds = Math.log(p / (1 - p));
        const DsC = V.clamp(Ds, D0, D1);
        V.line(g, X(DsC), P.y, X(DsC), P.y + P.h, { color: C.d, lw: 1.4, dash: [4, 3] });
        const Hp = loss(Ds, p);
        V.text(g, 'Δ* = ' + fmt(Ds, 2), X(DsC) + (DsC > 4 ? -6 : 6), P.y + 12, { size: 11.5, font: 'mono', weight: 600, color: C.d, align: DsC > 4 ? 'right' : 'left' });
        V.text(g, 'mínimo = H(p) = ' + fmt(Hp, 3), X(DsC) + (DsC > 4 ? -6 : 6), P.y + 28, { size: 10.5, font: 'mono', color: C.d, align: DsC > 4 ? 'right' : 'left' });
        // bolinha descendo o gradiente
        if (S.pPrev == null || Math.abs(S.pPrev - p) > 1e-9) { S.pPrev = p; S.D = -1.5; S.hold = 0; }
        const gr = sig(S.D) - p;
        if (Math.abs(gr) > 0.002) S.D -= 5 * gr * dt * 1.6; else { S.hold += dt; if (S.hold > 1.6) { S.D = -1.5; S.hold = 0; } }
        const bxp = X(V.clamp(S.D, D0, D1)), byp = Y(Math.min(Lmax, loss(S.D, p)));
        V.glow(g, bxp, byp, 16, C.a, 0.8);
        V.dot(g, bxp, byp, 6, C.ink);
        if (Math.abs(gr) > 0.01) V.arrow(g, bxp, byp - 16, bxp - V.clamp(gr * 140, -60, 60), byp - 16, { color: C.d, lw: 1.8, head: 6 });
        V.text(g, 'gradiente σ(Δ) − p = ' + fmt(gr, 3), P.x + P.w - 4, P.y + 50, { size: 10.5, font: 'mono', color: C.ink, align: 'right' });
      });
    })();

    /* ================================================ FIG 5 — a fábrica RLAIF */
    (function () {
      const fig = root.querySelector('#fig-fabrica');
      if (!fig) return;
      adaptH(fig, 440, 520);
      const S = { mode: 'pm', swap: true };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Modo', options: [{ v: 'pm', label: 'RLAIF canônico (PM)' }, { v: 'd', label: 'd-RLAIF (juiz direto)' }], value: 'pm', onChange: (v) => { S.mode = v; gSwap.show(v === 'pm'); upd(); } });
      const gSwap = UI.group(ctl);
      UI.toggle(gSwap.el, { label: 'trocar ordem e tirar média', value: true, onChange: (v) => { S.swap = v; upd(); } });
      const rN = UI.readout(ctl, 'pares no PM', '—');
      const rL = UI.readout(ctl, 'rótulo deste par', '—');
      function upd() {
        if (S.mode === 'pm') rN.set('318.127 (57,5% IA)'); else rN.set('sem PM', 'warn');
        rL.set(S.mode === 'd' ? 'nota 1–10 no laço' : S.swap ? '(0,80 + 0,62)/2 = 0,71' : '0,80 (só uma ordem)');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        const dm = S.mode === 'd';
        const bw = narrow ? Math.min(92, (w - 60) / 3) : Math.min(124, w * 0.15), bh = 46;
        let B;
        if (!narrow) {
          const y1 = h * 0.3, y2 = h * 0.66;
          B = { pr: [w * 0.08, y1], sl: [w * 0.26, y1], lb: [w * 0.45, y1], pm: [w * 0.65, y1], hu: [w * 0.65, y2], po: [w * 0.88, y1], rf: [w * 0.88, y2] };
        } else {
          const y1 = 40, y2 = h * 0.42, y3 = h * 0.74;
          B = { pr: [w * 0.18, y1], sl: [w * 0.5, y1], lb: [w * 0.82, y1], hu: [w * 0.2, y2], pm: [w * 0.62, y2], po: [w * 0.3, y3], rf: [w * 0.78, y3] };
        }
        const off = dm ? 0.22 : 1;
        const box = (k, label, sub, col, a = 1) => {
          const [x, y] = B[k];
          g.globalAlpha = a;
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 9, fill: V.alpha(col, 0.16), stroke: col, lw: 1.5 });
          V.text(g, label, x, y - (sub ? 3 : -4), { size: 12, weight: 600, align: 'center', color: C.ink, maxW: bw - 8 });
          if (sub) V.text(g, sub, x, y + 13, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: bw - 8 });
          g.globalAlpha = 1;
        };
        const edge = (a, b, col, alpha = 1, dash) => {
          const k = link(B[a], B[b], bw / 2, bh / 2);
          g.globalAlpha = alpha;
          V.arrow(g, k.sx, k.sy, k.ex, k.ey, { color: col, lw: 1.6, head: 7, dash });
          g.globalAlpha = 1;
          return k.at;
        };
        // arestas
        const ePS = edge('pr', 'sl', C.muted, off);
        const eSL = edge('sl', 'lb', C.muted, off);
        const eLP = edge('lb', 'pm', C.muted, off);
        const eHP = edge('hu', 'pm', C.muted, off);
        const ePM = dm ? null : edge('pm', 'po', C.muted, 1);
        // caixas
        box('pr', 'prompts', '182.831 red team', C.d, off);
        box('sl', 'SL-CAI', '2 respostas', C.b, off);
        box('lb', 'rotulador', '+ princípio', C.f, off);
        box('hu', 'humanos', '135.296 pares', C.e, off);
        if (!dm) box('pm', 'modelo de pref.', 'r_φ(x, y)', C.c, 1);
        box('po', 'política π_θ', dm ? 'PPO/REINFORCE' : 'PPO', C.a, 1);
        box('rf', 'referência', 'SL-CAI congelado', C.b, 1);
        // mola KL
        const [px, py] = B.po, [rx, ry] = B.rf;
        const stretch = 0.5 + 0.5 * Math.sin(t * 1.6);
        if (!narrow) spring(g, px, py + bh / 2 + 2, rx, ry - bh / 2 - 2 - 6 * stretch, C.d, 8, 5 + 2 * stretch);
        else spring(g, px + bw / 2 + 2, py, rx - bw / 2 - 2 - 6 * stretch, ry, C.d, 8, 5 + 2 * stretch);
        V.text(g, 'mola KL (β)', narrow ? (px + rx) / 2 : px + 14, narrow ? py - 12 : (py + ry) / 2, { size: 11, font: 'mono', color: C.d, align: narrow ? 'center' : 'left' });
        // partículas do lado offline
        const T = 4.2, ph = (t % T) / T;
        if (!dm) {
          if (ph < 0.22) { const [x, y] = ePS(ph / 0.22); V.glow(g, x, y, 10, C.d, 1); V.dot(g, x, y, 4, C.d); }
          else if (ph < 0.45) { const f = (ph - 0.22) / 0.23; const [x, y] = eSL(f); chip(g, x - 26, y - 22, 22, 18, 'A', { fill: V.alpha(C.a, 0.3), stroke: C.a, size: 11 }); chip(g, x + 4, y - 22, 22, 18, 'B', { fill: V.alpha(C.b, 0.3), stroke: C.b, size: 11 }); }
          else if (ph < 0.66) {
            const [x, y] = B.lb; const yy = y + bh / 2 + 8;
            if (S.swap) {
              chip(g, x - 56, yy, 52, 20, '(A,B) .80', { size: 10.5, fill: V.alpha(C.f, 0.15), stroke: C.f });
              chip(g, x + 4, yy, 52, 20, '(B,A) .62', { size: 10.5, fill: V.alpha(C.f, 0.15), stroke: C.f });
              chip(g, x - 28, yy + 24, 56, 20, 'p = .71', { size: 10.5, fill: V.alpha(C.a, 0.2), stroke: C.a, weight: 700 });
            } else chip(g, x - 28, yy, 56, 20, 'p = .80', { size: 10.5, fill: V.alpha(C.a, 0.2), stroke: C.a, weight: 700 });
          } else { const f = (ph - 0.66) / 0.34; const [x, y] = eLP(f); chip(g, x - 22, y - 10, 44, 20, S.swap ? '.71' : '.80', { size: 10.5, fill: V.alpha(C.a, 0.3), stroke: C.a, weight: 700 }); }
          for (let k = 0; k < 3; k++) { const f = ((t * 0.35 + k / 3) % 1); const [x, y] = eHP(f); V.dot(g, x, y, 3.5, C.e); }
          // barra de proporção sob o PM
          const [mx0, my] = narrow ? B.hu : B.pm; const mx = mx0; const pw = bw + 20, pyb = narrow ? my + bh / 2 + 8 : my - bh / 2 - 40;
          const bx0 = narrow ? 12 : mx - pw / 2;
          V.box(g, bx0, pyb, pw, 12, { r: 4, fill: V.alpha(C.f, 0.7) });
          V.box(g, bx0, pyb, pw * 0.425, 12, { r: 4, fill: V.alpha(C.e, 0.85) });
          V.text(g, 'humano 42,5% · IA 57,5%', narrow ? 12 : mx, pyb + 26, { size: 10.5, font: 'mono', align: narrow ? 'left' : 'center', color: C.muted, maxW: Math.max(pw + 30, 190) });
          // laço RL: resposta → PM → recompensa
          const f2 = (t % 2.4) / 2.4;
          const [ax, ay] = B.po, [bx2, by2] = B.pm;
          const cyc = f2 < 0.5 ? f2 * 2 : 2 - f2 * 2;
          const yo = narrow ? -8 : -14;
          const x = V.lerp(ax, bx2, cyc), y = V.lerp(ay, by2, cyc) + yo;
          if (f2 < 0.5) { V.dot(g, x, y, 4.5, C.a); } else { chip(g, x - 26, y - 10, 52, 20, 'r = ' + ['1,3', '−0,4', '0,9'][Math.floor(t / 2.4) % 3], { size: 10.5, fill: V.alpha(C.c, 0.25), stroke: C.c }); }
        } else {
          // juiz dentro do laço
          const [ax, ay] = B.po;
          const [jx, jy] = B.pm;
          V.box(g, jx - bw / 2 - 6, jy - bh / 2 - 4, bw + 12, bh + 8, { r: 10, fill: V.alpha(C.f, 0.2), stroke: C.f, lw: 1.8 });
          V.text(g, 'juiz LLM', jx, jy - 3, { size: 12, weight: 600, align: 'center', color: C.ink });
          V.text(g, 'nota 1–10', jx, jy + 13, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
          scrollIcon(g, jx + bw / 2 + 2, jy - bh / 2 - 2, 10);
          const hw = bw / 2 + 6, hh = bh / 2 + 4;
          const k1 = link([ax, ay], [jx, jy], hw, hh, { off: 9 }), k2 = link([jx, jy], [ax, ay], hw, hh, { off: 9 });
          V.arrow(g, k1.sx, k1.sy, k1.ex, k1.ey, { color: C.a, lw: 1.5, head: 6 });
          V.arrow(g, k2.sx, k2.sy, k2.ex, k2.ey, { color: C.f, lw: 1.5, head: 6 });
          const f2 = (t % 2.4) / 2.4;
          if (f2 < 0.5) { const [x, y] = k1.at(f2 * 2); V.dot(g, x, y, 4.5, C.a); }
          else { const [x, y] = k2.at((f2 - 0.5) * 2); V.dot(g, x, y, 4, C.f); }
          const [mx2, my2] = k2.at(0.5);
          const sc = ['7', '4', '8', '6'][Math.floor(t / 2.4) % 4] + '/10';
          chip(g, mx2 - 22 + k2.nx * 14, my2 - 10 + k2.ny * 14, 44, 20, sc, { size: 10.5, fill: V.alpha(C.f, 0.25), stroke: C.f });
          if (narrow) V.text(g, 'sem PM: o juiz pontua cada rollout (Lee et al.)', w / 2, h - 14, { size: 11.5, font: 'mono', align: 'center', color: C.f, maxW: w - 20 });
        }
        if (!narrow) {
          const ly = h - 34;
          V.text(g, '① offline: pares de IA (rotulador) + pares humanos treinam o PM', 16, ly, { size: 11.5, font: 'mono', color: dm ? V.alpha(C.muted, 0.6) : C.muted, maxW: w * 0.62 });
          V.text(g, dm ? '② online: a política gera, o juiz dá nota 1–10, o RL atualiza (sem PM)' : '② online: a política gera, o PM pontua, o PPO atualiza; a mola KL prende ao SL-CAI', 16, ly + 18, { size: 11.5, font: 'mono', color: C.ink, maxW: w - 32 });
        } else if (!dm) V.text(g, 'offline: rótulos → PM · online: PPO', w / 2, h - 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
      });
    })();

    /* ================================================ FIG 6 — Goodhart e a coleira KL */
    (function () {
      const fig = root.querySelector('#fig-goodhart');
      if (!fig) return;
      adaptH(fig, 440, 540);
      const S = { beta: 0.25, clamp: false, t0: null, first: true };
      const ctl = UI.controls(fig);
      nocase(UI.slider(ctl, { label: 'β (coleira KL)', min: 0.06, max: 1, step: 0.01, value: 0.25, fmt: (v) => fmt(v, 2), onInput: (v) => { S.beta = v; S.t0 = null; upd(); } }));
      UI.toggle(ctl, { label: 'clamp nos rótulos (interpretação)', value: false, onChange: (v) => { S.clamp = v; S.t0 = null; upd(); } });
      const rD = nocase(UI.readout(ctl, 'd* = √KL', '—'));
      const rK = UI.readout(ctl, 'KL (nats)', '—');
      const rQ = UI.readout(ctl, 'qualidade real', '—');
      const rR = UI.readout(ctl, 'região', '—');
      const DMAX = 6.2;
      const aP = () => (S.clamp ? 0.5 : 1);
      const dstar = () => Math.min(DMAX, aP() / (2 * S.beta));
      const gold = (d) => (d <= 0 ? 0 : d * (1 - 0.5 * Math.log(d)));
      const region = (d) => (d < 1.5 ? ['perto do SL-CAI', null] : d < 3.8 ? ['ponto doce', 'ok'] : d < 5 ? ['Goodhart: boilerplate', 'warn'] : ['Goodhart: dureza', 'bad']);
      function upd() { const d = dstar(); rD.set(fmt(d, 2)); rK.set(fmt(d * d, 1)); rQ.set(fmt(gold(d), 3)); const r = region(d); rR.set(r[0], r[1]); }
      upd();
      const RESP = {
        'perto do SL-CAI': ['Isso pode ser arriscado e talvez ilegal. Melhor não fazer.', 'pouco treino: cauteloso, pouco útil'],
        'ponto doce': [RESP_CAI, 'ponto doce: engaja e explica'],
        'Goodhart: boilerplate': ['Entendo que você possa estar passando por um momento difícil, e quero que saiba que você é importante. Acessar redes alheias não é correto. Se precisar, procure ajuda.', 'boilerplate: acolhimento genérico'],
        'Goodhart: dureza': ['Esse pedido é inaceitável e revela intenção criminosa. Não vou participar disso.', 'sobretreino: duro e acusatório'],
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        if (S.t0 == null) { S.t0 = S.first ? t - 5 : t; S.first = false; }
        const el = t - S.t0;
        if (el > 7.5) S.t0 = t;
        const u = V.clamp(el / 5, 0, 1);
        const ds = dstar();
        const d = ds * V.ease.inOut(u);
        const P = narrow ? { x: 40, y: 34, w: w - 56, h: h * 0.58 - 98 } : { x: 46, y: 34, w: w * 0.6 - 60, h: h - 118 };
        const X = (x) => P.x + (x / DMAX) * P.w, Y = (v) => P.y + P.h - v * P.h;
        // regiões
        [[0, 1.5, C.muted, 'SL-CAI'], [1.5, 3.8, C.a, 'ponto doce'], [3.8, DMAX, RED(), 'Goodhart']].forEach(([a, b, col, lab]) => {
          g.fillStyle = V.alpha(col, 0.07); g.fillRect(X(a), P.y, X(b) - X(a), P.h);
          V.text(g, lab, (X(a) + X(b)) / 2, P.y - 10, { size: 11, font: 'mono', align: 'center', color: col === C.muted ? C.muted : col, maxW: X(b) - X(a) - 4 });
        });
        for (let k = 1; k <= 6; k++) V.text(g, String(k), X(k), P.y + P.h + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        axes(g, P, { yl: 'escore (escala livre)', yOff: 24 });
        V.text(g, 'd = √KL desde o SL-CAI →', P.x + P.w / 2, P.y + P.h + 30, { size: 11, font: 'mono', align: 'center', color: C.muted });
        const GN = 1.45;
        // proxy
        g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(DMAX), Y((aP() * DMAX) / 6.4)); g.strokeStyle = C.b; g.lineWidth = 2.2; g.stroke();
        
        // ouro
        g.beginPath(); for (let k = 0; k <= 120; k++) { const x = (k / 120) * DMAX; const yy = Y(gold(x) / GN); if (k) g.lineTo(X(x), yy); else g.moveTo(X(x), yy); }
        g.strokeStyle = C.a; g.lineWidth = 2.6; g.stroke();
        V.dot(g, X(Math.E), Y(gold(Math.E) / GN), 3.5, C.a);
        V.line(g, X(Math.E), P.y + P.h, X(Math.E), P.y + P.h + 5, { color: C.a, lw: 2 });
        V.text(g, 'e', X(Math.E), P.y + P.h + 15, { size: 11, font: 'mono', weight: 700, align: 'center', color: C.a });
        { const lx = P.x + 4, ly = P.y + P.h + 50;
          V.line(g, lx, ly - 4, lx + 18, ly - 4, { color: C.a, lw: 2.6 }); V.text(g, 'qualidade real', lx + 24, ly, { size: 11, font: 'mono', color: C.a });
          V.line(g, lx + 140, ly - 4, lx + 158, ly - 4, { color: C.b, lw: 2.2 }); V.text(g, 'PM (proxy)', lx + 164, ly, { size: 11, font: 'mono', color: C.b }); }
        // alvo d*
        V.line(g, X(ds), P.y + 22, X(ds), P.y + P.h, { color: C.d, lw: 1.3, dash: [4, 3] });
        V.text(g, 'd* = a/(2β)', X(ds) + (ds > 4.5 ? -5 : 5), P.y + P.h - 26, { size: 11, font: 'mono', color: C.d, align: ds > 4.5 ? 'right' : 'left' });
        // política
        const gy = Y(gold(d) / GN), py = Y((aP() * d) / 6.4);
        V.dot(g, X(d), py, 4.5, C.b);
        V.glow(g, X(d), gy, 16, C.a, 0.9);
        V.dot(g, X(d), gy, 6.5, C.ink);
        // mola: referência em d = 0
        const sy = P.y + P.h - 12;
        V.dot(g, X(0) + 4, sy, 4.5, C.b);
        if (X(d) - X(0) > 16) spring(g, X(0) + 8, sy, X(d) - 4, sy, C.d, Math.max(3, Math.round(d * 2.2)), 4);
        V.dot(g, X(d), sy, 4, C.d);
        // balão
        const reg = region(d)[0];
        const B = narrow ? { x: 10, y: h * 0.58 + 8, w: w - 20 } : { x: w * 0.6 + 6, y: 26, w: w * 0.4 - 18 };
        let y = B.y;
        V.text(g, 'mesmo pedido do começo do capítulo', B.x, y + 8, { size: 11, font: 'mono', color: C.muted, maxW: B.w });
        y += 16;
        const uu = bubble(g, B.x + B.w, y, B.w * 0.85, PROMPT_RT, { side: 'r', color: C.d, size: narrow ? 11.5 : 12 });
        y += uu.h + 14;
        const col = reg === 'ponto doce' ? C.a : reg.startsWith('Goodhart') ? RED() : C.b;
        bubble(g, B.x, y, B.w * 0.98, RESP[reg][0], { side: 'l', color: col, size: narrow ? 11.5 : 12, tag: RESP[reg][1] });
      });
    })();

    /* ================================================ FIG 7 — onde mora a constituição */
    (function () {
      const fig = root.querySelector('#fig-geracoes');
      if (!fig) return;
      adaptH(fig, 440, 620);
      const S = { mode: 'cai', s1: null, s2: null };
      const MODES = {
        cai: { lab: 'CAI 2022', where: 'prompts de crítica e de comparação', txt: 'Os princípios aparecem em dois lugares: nos pedidos de crítica e revisão que a própria política lê no SL, e no prompt de múltipla escolha do rotulador no RL. 16 + 16 princípios, um sorteado por vez. Sinal: SFT nas revisões + PM de rótulos suaves + PPO.' },
        rlaif: { lab: 'RLAIF 2023', where: 'preâmbulo do juiz', txt: 'Só o juiz lê os princípios. RLAIF canônico: rótulos → PM → RL, comparável ao RLHF (inofensivo: 88% × 76%). d-RLAIF: o juiz dá uma nota de 1 a 10 a cada resposta durante o RL, sem PM.' },
        ccai: { lab: 'Collective', where: 'texto escrito pelo público', txt: '1.002 adultos dos EUA deliberam no Polis: 1.127 afirmações e 38.252 votos. A constituição pública compartilha ~50% dos conceitos com a padrão. O modelo treinado com ela mostrou menos viés em 9 dimensões sociais (BBQ), com desempenho equivalente.' },
        delib: { lab: 'Deliberative 2024', where: 'no CoT da própria política', txt: 'SFT: a spec vai no system prompt, a política gera (CoT, resposta) citando a política, um juiz com a spec filtra pelo mínimo de k notas, e a spec é removida do exemplo. RL: o juiz vê a spec e dá a recompensa, mas o CoT fica escondido dele.' },
        c2026: { lab: 'Constituição 2026', where: 'documento longo e explicativo', txt: 'O documento explica o porquê, ordena prioridades (seguro > ético > diretrizes > útil) e fixa restrições rígidas. O próprio modelo o usa para gerar dados sintéticos: entender o texto, conversas relevantes, respostas alinhadas e rankings de respostas.' },
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { options: Object.keys(MODES).map((k) => ({ v: k, label: MODES[k].lab })), value: 'cai', onChange: (v) => { S.mode = v; rW.set(MODES[v].where); } });
      const rW = UI.readout(ctl, 'onde mora', MODES.cai.where);
      const rr = V.rng(7);
      const DOTS = Array.from({ length: 100 }, () => ({ x: rr(), y: rr(), ph: rr() * 6.28, v: rr() < 0.62 ? 1 : -1 }));
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 600;
        const m = S.mode;
        const bw = narrow ? Math.min(150, w * 0.42) : Math.min(150, w * 0.19), bh = narrow ? 48 : 56;
        const A = narrow ? { jz: [w * 0.24, 44], po: [w * 0.76, 44], dd: [w * 0.5, 146] } : { jz: [w * 0.17, 96], dd: [w * 0.5, 96], po: [w * 0.83, 96] };
        const band = narrow ? 186 : 96 + bh / 2 + 12;
        const textY = narrow ? h - 168 : h - 104;
        const actor = (k, label, sub, col, a = 1) => {
          const [x, y] = A[k];
          g.globalAlpha = a;
          V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 10, fill: V.alpha(col, 0.15), stroke: col, lw: 1.6 });
          V.text(g, label, x, y - 2, { size: narrow ? 12 : 13, weight: 600, align: 'center', color: C.ink, maxW: bw - 8 });
          V.text(g, sub, x, y + 15, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: bw - 8 });
          g.globalAlpha = 1;
        };
        const dimA = m === 'ccai' ? 0.3 : 1;
        actor('jz', m === 'delib' ? 'juiz G_RM' : 'juiz / rotulador', m === 'rlaif' ? 'rótulos ou nota' : 'lê princípios', C.f, dimA);
        actor('dd', 'dados de treino', m === 'c2026' ? 'sintéticos' : 'pares, revisões', C.b, dimA);
        actor('po', 'política', m === 'delib' ? 'raciocina no CoT' : 'π_θ', C.a, dimA);
        const legend = [];
        const flow = (a, b, label, col, off = 0, dash) => {
          if (narrow && label) { legend.push([label, col]); label = ''; }
          const k = link(A[a], A[b], bw / 2, bh / 2, { off });
          V.arrow(g, k.sx, k.sy, k.ex, k.ey, { color: col, lw: 1.6, head: 7, dash });
          const f = (t * 0.45 + (off > 0 ? 0.5 : 0)) % 1;
          const [mx, my] = k.at(f);
          V.dot(g, mx, my, 3.5, col);
          if (label) {
            const [cx, cy] = k.at(0.5);
            const sgn = off >= 0 ? 1 : -1;
            const horiz = Math.abs(k.ex - k.sx) > Math.abs(k.ey - k.sy);
            const lx = cx + k.nx * sgn * (horiz ? 12 : 8), ly = cy + k.ny * sgn * (horiz ? 12 : 8) + (horiz ? 4 : 0);
            V.text(g, label, lx, ly, { size: 10.5, font: 'mono', align: horiz ? 'center' : (k.nx * sgn > 0 ? 'left' : 'right'), color: col, maxW: horiz ? k.len + 10 : w * 0.42 });
          }
        };
        // alvos dos pergaminhos
        const corner = (k, side) => [A[k][0] + side * (bw / 2 - 6), A[k][1] - bh / 2 - 8];
        let t1 = null, t2 = null;
        if (m === 'cai' || m === 'delib') { t1 = corner('jz', 1); t2 = corner('po', -1); }
        if (m === 'rlaif') t1 = corner('jz', 1);
        if (m === 'ccai') t1 = [w / 2, band + 50];
        if (m === 'c2026') t1 = narrow ? [w / 2, band + 6] : [w / 2, band + 18];
        if (narrow && (m === 'cai' || m === 'delib' || m === 'rlaif')) { t1 = [A.jz[0] - bw / 2 + 10, A.jz[1] + bh / 2 + 14]; if (t2) t2 = [A.po[0] + bw / 2 - 10, A.po[1] + bh / 2 + 14]; }
        const k5 = Math.min(1, dt * 5);
        if (t1) S.s1 = S.s1 ? [V.lerp(S.s1[0], t1[0], k5), V.lerp(S.s1[1], t1[1], k5)] : t1.slice();
        if (t2) S.s2 = S.s2 ? [V.lerp(S.s2[0], t2[0], k5), V.lerp(S.s2[1], t2[1], k5)] : t2.slice(); else if (S.s2 && t1) S.s2 = [V.lerp(S.s2[0], t1[0], k5), V.lerp(S.s2[1], t1[1], k5)];
        /* mini-cenas: o laço SL e a rotulagem RL */
        function miniSL(cx, cy) {
          const r = narrow ? 42 : 48, names = ['resposta', 'crítica', 'revisão'], cols = [C.c, C.f, C.a];
          const act = Math.floor(t * 0.9) % 3;
          for (let i = 0; i < 3; i++) {
            const a0 = -Math.PI / 2 + (i * 2 * Math.PI) / 3, a1 = a0 + (2 * Math.PI) / 3;
            const p0 = [cx + Math.cos(a0) * r, cy + Math.sin(a0) * r], p1 = [cx + Math.cos(a1) * r, cy + Math.sin(a1) * r];
            V.arrow(g, V.lerp(p0[0], p1[0], 0.3), V.lerp(p0[1], p1[1], 0.3), V.lerp(p0[0], p1[0], 0.7), V.lerp(p0[1], p1[1], 0.7), { color: V.alpha(C.muted, 0.8), lw: 1.3, head: 5 });
          }
          for (let i = 0; i < 3; i++) {
            const a0 = -Math.PI / 2 + (i * 2 * Math.PI) / 3;
            const x = cx + Math.cos(a0) * r, y = cy + Math.sin(a0) * r;
            const cw = narrow ? 56 : 66;
            chip(g, x - cw / 2, y - 10, cw, 20, names[i], { size: 10.5, fill: V.alpha(cols[i], i === act ? 0.45 : 0.14), stroke: cols[i], weight: i === act ? 700 : 500 });
          }
          V.text(g, '×4 rodadas por prompt (SL)', cx, cy + r * 0.5 + 30, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        }
        function miniRL(cx, cy, swap) {
          const cw = narrow ? 44 : 50;
          chip(g, cx - cw - 4, cy - 30, cw, 20, '(A)', { size: 10.5, fill: V.alpha(C.a, 0.2), stroke: C.a });
          chip(g, cx + 4, cy - 30, cw, 20, '(B)', { size: 10.5, fill: V.alpha(C.b, 0.2), stroke: C.b });
          V.text(g, 'qual é menos nociva?', cx, cy + 2, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: narrow ? w * 0.46 : 200 });
          if (!swap) chip(g, cx - 50, cy + 12, 100, 22, 'p_A = σ(ℓA − ℓB)', { size: 10.5, fill: V.alpha(C.d, 0.18), stroke: C.d });
          else { chip(g, cx - 72, cy + 12, 70, 20, '(A,B) .80', { size: 10.5, fill: V.alpha(C.f, 0.15), stroke: C.f }); chip(g, cx + 2, cy + 12, 70, 20, '(B,A) .62', { size: 10.5, fill: V.alpha(C.f, 0.15), stroke: C.f }); chip(g, cx - 40, cy + 38, 80, 20, 'média .71', { size: 10.5, fill: V.alpha(C.d, 0.2), stroke: C.d, weight: 700 }); }
        }
        if (m === 'cai') {
          flow('po', 'dd', 'revisões (SL)', C.a, 9);
          flow('jz', 'dd', 'rótulos suaves', C.f, narrow ? -9 : 9);
          flow('dd', 'po', 'SFT + PM/PPO', C.b, 9);
          miniSL(A.po[0], band + (narrow ? 56 : 66));
          miniRL(A.jz[0], band + (narrow ? 56 : 66), false);
        } else if (m === 'rlaif') {
          miniRL(A.jz[0], band + (narrow ? 70 : 96), true);
          flow('jz', 'dd', 'rótulos → PM', C.f, narrow ? -9 : 9);
          flow('dd', 'po', 'RL', C.b, narrow ? 9 : -9);
          if (!narrow) {
            V.arrow(g, A.jz[0] + 20, 96 + bh / 2 + 6, A.po[0] - 20, 96 + bh / 2 + 6, { color: C.f, lw: 1.5, head: 7, dash: [5, 4], curve: 60 });
            V.text(g, 'd-RLAIF: nota 1–10 direto no laço de RL', w / 2, 96 + bh / 2 + 52, { size: 10.5, font: 'mono', align: 'center', color: C.f, maxW: w - 20 });
          } else {
            V.arrow(g, A.jz[0] + bw / 2 + 3, A.jz[1], A.po[0] - bw / 2 - 5, A.po[1], { color: C.f, lw: 1.5, head: 6, dash: [4, 3] });
            V.text(g, 'd-RLAIF: o juiz dá nota 1–10 direto no laço de RL', w / 2, band + 10, { size: 10.5, font: 'mono', align: 'center', color: C.f, maxW: w - 20 });
          }
        } else if (m === 'ccai') {
          const cx0 = narrow ? 12 : w * 0.06, cw0 = narrow ? w * 0.36 : w * 0.3, cy0 = band + 6, ch0 = narrow ? 96 : 104;
          DOTS.forEach((p) => {
            const x = cx0 + p.x * cw0, y = cy0 + p.y * ch0;
            const on = Math.sin(t * 2 + p.ph) > 0.6;
            V.dot(g, x, y, 2.6, on ? (p.v > 0 ? C.a : C.c) : V.alpha(C.muted, 0.5));
          });
          V.text(g, '1.002 pessoas (1 ponto ≈ 10)', cx0 + cw0 / 2, cy0 + ch0 + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: cw0 + 30 });
          V.arrow(g, cx0 + cw0 + 6, cy0 + ch0 / 2, w / 2 - 18, band + 50, { color: C.d, lw: 1.5, head: 7 });
          const vx = w * 0.8, vy = cy0 + ch0 / 2, vr = narrow ? 30 : 42;
          g.beginPath(); g.arc(vx - vr * 0.55, vy, vr, 0, Math.PI * 2); g.fillStyle = V.alpha(C.d, 0.2); g.fill(); g.strokeStyle = C.d; g.lineWidth = 1.4; g.stroke();
          g.beginPath(); g.arc(vx + vr * 0.55, vy, vr, 0, Math.PI * 2); g.fillStyle = V.alpha(C.b, 0.2); g.fill(); g.strokeStyle = C.b; g.stroke();
          V.text(g, '~50%', vx, vy + 4, { size: 11.5, font: 'mono', weight: 700, align: 'center', color: C.ink });
          V.text(g, 'pública', vx - vr * 1.0, vy + vr + 14, { size: 10.5, font: 'mono', align: 'center', color: C.d });
          V.text(g, 'padrão', vx + vr * 1.0, vy + vr + 14, { size: 10.5, font: 'mono', align: 'center', color: C.b });
          V.text(g, 'constituição pública', w / 2, band + 82, { size: 10.5, font: 'mono', align: 'center', color: C.d });
        } else if (m === 'delib') {
          flow('po', 'dd', '(CoT, resposta)', C.a, 9);
          flow('jz', 'dd', 'mín. de k notas', C.f, narrow ? -9 : 9);
          flow('dd', 'po', 'SFT sem a spec', C.b, 9);
          let jx, jy, tx2, ty2;
          if (!narrow) {
            V.arrow(g, A.jz[0] + 20, 96 + bh / 2 + 6, A.po[0] - 20, 96 + bh / 2 + 6, { color: C.f, lw: 1.5, head: 7, dash: [5, 4], curve: 60 });
            V.text(g, 'RL: recompensa do juiz (que vê a spec)', w / 2, 96 + bh / 2 + 52, { size: 10.5, font: 'mono', align: 'center', color: C.f, maxW: w - 20 });
            jx = A.jz[0] - 34; jy = 96 + bh / 2 + 22; tx2 = A.po[0] + 34; ty2 = 96 - bh / 2 - 30;
          } else {
            V.arrow(g, A.jz[0] + bw / 2 + 3, A.jz[1], A.po[0] - bw / 2 - 5, A.po[1], { color: C.f, lw: 1.5, head: 6, dash: [4, 3] });
            V.text(g, 'RL: recompensa do juiz (que vê a spec)', w / 2, band + 10, { size: 10.5, font: 'mono', align: 'center', color: C.f, maxW: w - 20 });
            jx = 30; jy = band + 40; tx2 = w - 56; ty2 = band + 40;
          }
          g.beginPath(); g.ellipse(jx, jy, 12, 7, 0, 0, Math.PI * 2); g.strokeStyle = C.ink; g.lineWidth = 1.5; g.stroke();
          V.dot(g, jx, jy, 3, C.ink);
          V.line(g, jx - 13, jy + 9, jx + 13, jy - 9, { color: RED(), lw: 2 });
          V.text(g, 'juiz não vê o CoT', jx - 14, jy + 22, { size: 10.5, font: 'mono', color: RED() });
          V.box(g, tx2 - 44, ty2 - 11, 88, 22, { r: 10, fill: V.alpha(C.a, 0.15), stroke: C.a, lw: 1 });
          V.text(g, '“a spec diz…”', tx2, ty2 + 4, { size: 10.5, font: 'mono', align: 'center', color: C.ink });
          V.text(g, 'CoT', tx2, narrow ? ty2 + 24 : ty2 - 16, { size: 10.5, font: 'mono', align: 'center', color: C.a });
        } else if (m === 'c2026') {
          const sx = narrow ? 14 : w * 0.06, sw = narrow ? w * 0.44 : w * 0.26, sy = narrow ? band + 40 : band + 8;
          const PRI = [['1 · amplamente seguro', C.a], ['2 · amplamente ético', C.b], ['3 · diretrizes', C.d], ['4 · genuinamente útil', C.c]];
          PRI.forEach(([l, c], i) => { V.box(g, sx, sy + i * 26, sw, 22, { r: 6, fill: V.alpha(c, 0.18), stroke: c, lw: 1.2 }); V.text(g, l, sx + 8, sy + i * 26 + 15, { size: 11, font: 'mono', color: C.ink, maxW: sw - 12 }); });
          V.box(g, sx - 6, sy - 4, 4, 4 * 26 + 2, { r: 2, fill: RED() });
          V.text(g, 'restrições rígidas', sx, sy + 4 * 26 + 14, { size: 10.5, font: 'mono', color: RED() });
          const types = ['entender', 'conversas', 'respostas', 'rankings'];
          const cols = narrow ? 2 : 4;
          const tx0 = narrow ? w * 0.52 : w * 0.42, tw0 = narrow ? w * 0.46 : w * 0.5;
          const cw = (tw0 - (cols - 1) * 6) / cols;
          types.forEach((ty, i) => {
            const f = (t * 0.4 + i * 0.25) % 1;
            const yy = sy + (narrow ? 20 : 60) + Math.floor(i / cols) * 30 - (narrow ? 0 : f * 16);
            g.globalAlpha = 0.45 + 0.55 * Math.sin(Math.PI * f);
            chip(g, tx0 + (i % cols) * (cw + 6), yy, cw, 22, ty, { size: 10.5, fill: V.alpha(C.b, 0.2), stroke: C.b });
            g.globalAlpha = 1;
          });
          V.text(g, 'a política gera dados sintéticos', tx0 + tw0 / 2, sy + (narrow ? 94 : 100), { size: 10.5, font: 'mono', align: 'center', color: C.muted, maxW: tw0 });
          flow('po', 'dd', '', C.a, 9);
          flow('dd', 'po', 'treino', C.b, 9);
        }
        if (S.s1) scrollIcon(g, S.s1[0], S.s1[1] + Math.sin(t * 2) * 2, m === 'c2026' ? 15 : 11);
        if (S.s2 && (m === 'cai' || m === 'delib')) scrollIcon(g, S.s2[0], S.s2[1] + Math.sin(t * 2 + 1) * 2, 11);
        if (m === 'c2026' && S.s1) V.text(g, 'constituição (CC0)', S.s1[0] + 20, S.s1[1] + 4, { size: 10.5, font: 'mono', color: C.d });
        legend.forEach(([l, c], i) => {
          const ly = textY - 22 - (legend.length - 1 - i) * 17;
          V.line(g, 14, ly - 4, 34, ly - 4, { color: c, lw: 2 }); V.dot(g, 34, ly - 4, 2.5, c);
          V.text(g, l, 42, ly, { size: 11, font: 'mono', color: c, maxW: w - 56 });
        });
        V.line(g, 12, textY - 14, w - 12, textY - 14, { color: C.grid, lw: 1 });
        V.text(g, MODES[m].lab, 14, textY + 2, { size: 12, font: 'mono', weight: 700, color: C.d });
        V.wrap(g, MODES[m].txt, 14, textY + 22, w - 28, { size: narrow ? 11.5 : 12.5, lineH: narrow ? 16 : 18, color: C.ink, maxLines: narrow ? 9 : 4 });
      });
    })();

    /* ================================================ FIG 8 — posto 1 × posto alto */
    (function () {
      const fig = root.querySelector('#fig-direcao');
      if (!fig) return;
      adaptH(fig, 480, 475);
      const S = { mode: 'abl', alpha: 1, K: 6, t0: null, first: true };
      const n = 6;
      const W = V.randMatrix(n, n, 19, 0.8);
      const rg = V.rng(23);
      let r = Array.from({ length: n }, () => rg.normal());
      const rn = Math.hypot(...r); r = r.map((v) => v / rn);
      const rW = Array.from({ length: n }, (_, j) => r.reduce((s, ri, i) => s + ri * W[i][j], 0));
      const DL = Array.from({ length: 12 }, () => Array.from({ length: n }, () => rg.normal()));
      const XS = Array.from({ length: 12 }, () => Array.from({ length: n }, () => rg.normal()));
      const ETA = 0.12;
      const dAbl = (a) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => -a * r[i] * rW[j]));
      const dTr = (kf) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => { let s = 0; for (let m = 0; m < Math.ceil(kf); m++) { const wgt = Math.min(1, kf - m); s += -ETA * wgt * DL[m][i] * XS[m][j]; } return s; }));
      const PTS = Array.from({ length: 36 }, (_, i) => ({ harm: i < 18, cr: (i < 18 ? 1.6 : -0.2) + rg.normal() * 0.32, co: rg.normal() * 0.35 }));
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Mudança', options: [{ v: 'abl', label: 'abliteração' }, { v: 'tr', label: 'pós-treino (CAI/SFT)' }], value: 'abl', onChange: (v) => { S.mode = v; gA.show(v === 'abl'); gT.show(v === 'tr'); S.t0 = null; } });
      const gA = UI.group(ctl), gT = UI.group(ctl);
      nocase(UI.slider(gA.el, { label: 'α (força)', min: 0, max: 3, step: 0.1, value: 1, fmt: (v) => fmt(v, 1), onInput: (v) => { S.alpha = v; S.t0 = null; } }));
      nocase(UI.slider(gT.el, { label: 'termos δxᵀ somados', min: 1, max: 12, step: 1, value: 6, onInput: (v) => { S.K = v; S.t0 = null; } }));
      gT.show(false);
      const rRank = UI.readout(ctl, 'posto de ΔW', '—');
      const rF = nocase(UI.readout(ctl, '‖ΔW‖_F', '—'));
      const rC = UI.readout(ctl, 'custo', '—');
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640;
        if (S.t0 == null) { S.t0 = S.first ? t - 1.6 : t; S.first = false; }
        const el = t - S.t0;
        let dW, prog;
        if (S.mode === 'abl') { const ramp = V.clamp(el / 1.5, 0, 1); if (el > 5) S.t0 = t; prog = V.ease.inOut(ramp); dW = dAbl(S.alpha * prog); }
        else { const kf = Math.min(S.K, el / 0.55 + 0.001); if (el > S.K * 0.55 + 3) S.t0 = t; prog = kf / 12; dW = dTr(kf); }
        const sv = svals(dW);
        const smax = Math.max(sv[0], 1e-9);
        const rank = sv.filter((s) => s > 1e-6 * Math.max(1, smax) && s > 1e-9).length;
        const fro = Math.sqrt(dW.reduce((s, row) => s + row.reduce((a, v) => a + v * v, 0), 0));
        rRank.set(String(rank), rank <= 1 ? 'warn' : 'ok');
        rF.set(fmt(fro, 2));
        rC.set(S.mode === 'abl' ? 'minutos, sem backprop' : 'horas–dias, muitos exemplos');
        const f1 = (v) => fmt(v, 1);
        // matrizes
        const cell = narrow ? Math.max(17, Math.floor((w - 60) / 12) - 2) : 26;
        const mW = n * (cell + 2) - 2;
        const mx1 = narrow ? 12 : 24, my = 28;
        const mx2 = mx1 + mW + (narrow ? 22 : 40);
        V.text(g, 'W', mx1 + mW / 2, my - 10, { size: 12, font: 'mono', weight: 600, align: 'center', color: C.muted });
        V.matrix(g, { x: mx1, y: my, data: W, cell, gap: 2, cmap: 'div', values: true, fmt: f1, min: -2, max: 2 });
        const dMax = Math.max(0.3, ...dW.map((row) => Math.max(...row.map(Math.abs))));
        V.text(g, S.mode === 'abl' ? 'ΔW = −α r̂(r̂ᵀW)' : 'ΔW = −η Σ δₖxₖᵀ', mx2 + mW / 2, my - 10, { size: 12, font: 'mono', weight: 600, align: 'center', color: S.mode === 'abl' ? C.c : C.a, maxW: mW + 20 });
        V.matrix(g, { x: mx2, y: my, data: dW, cell, gap: 2, cmap: 'div', values: true, fmt: f1, min: -dMax, max: dMax });
        V.text(g, '+', mx1 + mW + (mx2 - mx1 - mW) / 2, my + mW / 2 + 5, { size: 16, align: 'center', color: C.muted });
        if (S.mode === 'abl') {
          // r̂ como coluna à esquerda de ΔW? mostrar como fator
          const rx = mx2 + mW + 6;
          if (!narrow) { r.forEach((v, i) => { V.box(g, rx, my + i * (cell + 2), 12, cell, { r: 3, fill: V.cmap(v / 0.8, 'div') }); }); V.text(g, 'r̂', rx + 6, my + mW + 16, { size: 11, font: 'mono', align: 'center', color: C.muted }); }
        }
        // valores singulares
        let sx, sy, sw, sh;
        if (!narrow) { sx = mx2 + mW + 44; sy = my; sw = w - sx - 16; sh = mW; }
        else { sx = 30; sy = my + mW + 40; sw = w * 0.42 - 30; sh = 170; }
        V.text(g, 'valores singulares de ΔW', sx + sw / 2, sy - 10, { size: 11.5, font: 'mono', align: 'center', color: C.muted, maxW: sw + 20 });
        const SMX = S.mode === 'abl' ? Math.max(1, 3 * Math.hypot(...rW)) : 1.6;
        const bwid = (sw - 5 * 5) / 6;
        V.line(g, sx, sy + sh, sx + sw, sy + sh, { color: V.alpha(C.muted, 0.6), lw: 1 });
        sv.forEach((s, i) => {
          const bh = Math.min(sh, (s / SMX) * sh);
          const x = sx + i * (bwid + 5);
          V.box(g, x, sy + sh - bh, bwid, Math.max(1.5, bh), { r: 3, fill: s > 1e-6 ? (S.mode === 'abl' ? C.c : C.a) : V.alpha(C.muted, 0.3) });
          V.text(g, 'σ' + (i + 1), x + bwid / 2, sy + sh + 14, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
          if (bwid > 20) V.text(g, fmt(s, 2), x + bwid / 2, sy + sh - bh - 5, { size: 10, font: 'mono', align: 'center', color: C.ink });
        });
        // nuvens de ativações
        let qx, qy, qw, qh;
        if (!narrow) { qx = 40; qy = my + mW + 46; qw = w * 0.55 - 40; qh = h - qy - 28; }
        else { qx = w * 0.5 + 18; qy = sy; qw = w - qx - 12; qh = sh; }
        const XR = [-4.6, 3], YR = [-1.4, 2.4];
        const QX = (v) => qx + ((v - XR[0]) / (XR[1] - XR[0])) * qw, QY = (v) => qy + qh - ((v - YR[0]) / (YR[1] - YR[0])) * qh;
        V.box(g, qx, qy, qw, qh, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.3), lw: 1 });
        V.line(g, QX(0), qy + 4, QX(0), qy + qh - 4, { color: V.alpha(C.d, 0.7), lw: 1.2, dash: [4, 3] });
        V.text(g, 'componente em r̂ →', qx + qw - 4, qy + qh + 14, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
        V.text(g, 'r̂ = 0', QX(0) + 4, qy + 13, { size: 10.5, font: 'mono', color: C.d });
        PTS.forEach((p) => {
          let cr = p.cr, co = p.co;
          if (S.mode === 'abl') cr = (1 - S.alpha * prog) * p.cr;
          else { const q = Math.min(1, (prog * 12) / 6); if (p.harm) { co = p.co + 1.6 * q; cr = p.cr * (1 - 0.15 * q); } else co = p.co + 0.12 * q; }
          V.dot(g, QX(cr), QY(co), 3.6, p.harm ? C.c : C.a);
        });
        if (!narrow) {
          const lx = w * 0.58, ly = qy + 10, lw2 = w - lx - 16;
          const msg = S.mode === 'abl'
            ? 'A abliteração escala a componente em r̂ por (1 − α): com α = 1 as nuvens colapsam na linha r̂ = 0; com α = 3 (padrão do GLM) são refletidas. Uma única direção muda; o resto do espaço fica intacto.'
            : 'Com SGD puro, cada termo δₖxₖᵀ tem posto 1, e termos em direções diferentes se somam num ΔW de posto alto (na prática, com Adam e muitos dados, o posto efetivo fica baixo). As ativações dos prompts nocivos se deslocam numa direção nova (ilustrativo): o modelo continua reconhecendo o pedido, e passa a responder de outro jeito.';
          V.wrap(g, msg, lx, ly + 12, lw2, { size: 12, lineH: 17, color: C.ink, maxLines: 9 });
          V.dot(g, lx + 6, qy + qh - 30, 4, C.c); V.text(g, 'prompts nocivos', lx + 16, qy + qh - 26, { size: 11, font: 'mono', color: C.muted });
          V.dot(g, lx + 6, qy + qh - 12, 4, C.a); V.text(g, 'prompts inofensivos', lx + 16, qy + qh - 8, { size: 11, font: 'mono', color: C.muted });
        } else {
          const ly = Math.max(sy + sh + 32, qy + qh + 30);
          V.wrap(g, S.mode === 'abl' ? 'Abliteração: só σ1 ≠ 0. A componente em r̂ das ativações é escalada por (1 − α).' : 'Pós-treino (SGD puro): cada δxᵀ tem posto 1; a soma de muitos tem posto alto. Os pontos rosa (nocivos) se movem numa direção nova.', 12, ly, w - 24, { size: 11.5, lineH: 16, color: C.ink, maxLines: 5 });
        }
      });
    })();

    /* ================================================ FIG 9 — dois erros, um limiar */
    (function () {
      const fig = root.querySelector('#fig-seguranca');
      if (!fig) return;
      adaptH(fig, 450, 470);
      const PROF = {
        rlhf: { lab: 'RLHF evasivo', s: [0.38, 0.13], u: [0.72, 0.13], tau: 0.36 },
        help: { lab: 'helpful-only', s: [0.38, 0.13], u: [0.72, 0.13], tau: 0.82 },
        cai: { lab: 'tipo CAI', s: [0.3, 0.1], u: [0.78, 0.1], tau: 0.55 },
      };
      const NS = 250, NU = 200;
      const S = { prof: 'cai', tau: 0.55 };
      const ctl = UI.controls(fig);
      let sl;
      UI.seg(ctl, { label: 'Modelo', options: Object.keys(PROF).map((k) => ({ v: k, label: PROF[k].lab })), value: 'cai', onChange: (v) => { S.prof = v; S.tau = PROF[v].tau; sl.set(S.tau, true); upd(); } });
      sl = nocase(UI.slider(ctl, { label: 'limiar de recusa', min: 0.05, max: 0.95, step: 0.01, value: 0.55, fmt: (v) => fmt(v, 2), onInput: (v) => { S.tau = v; upd(); } }));
      const rO = UI.readout(ctl, 'sobre-recusa', '—');
      const rH = UI.readout(ctl, 'cumprimento indevido', '—');
      const cnt = (tau, P) => { const pr = PROF[P]; const over = NS * (1 - Phi((tau - pr.s[0]) / pr.s[1])); const harm = NU * Phi((tau - pr.u[0]) / pr.u[1]); return { over, harm }; };
      function upd() {
        const c = cnt(S.tau, S.prof);
        rO.set(Math.round(c.over) + '/250 (' + fmt((100 * c.over) / NS, 1) + '%)', c.over / NS > 0.1 ? 'bad' : c.over / NS > 0.03 ? 'warn' : 'ok');
        rH.set(Math.round(c.harm) + '/200 (' + fmt((100 * c.harm) / NU, 1) + '%)', c.harm / NU > 0.1 ? 'bad' : c.harm / NU > 0.03 ? 'warn' : 'ok');
      }
      upd();
      const rr = V.rng(91);
      const DROPS = Array.from({ length: 26 }, (_, i) => ({ safe: i % 9 < 5, z: rr.normal(), ph: rr() }));
      const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) * ((x - m) / s)) / (s * Math.sqrt(2 * Math.PI));
      V.fig(fig, (g, w, h, t, dt, _S, Hh) => {
        const C = V.C();
        const narrow = w < 640;
        const pr = PROF[S.prof];
        const P = narrow ? { x: 16, y: 26, w: w - 32, h: h * 0.5 - 30 } : { x: 20, y: 26, w: w * 0.58 - 30, h: h - 66 };
        const X = (v) => P.x + v * P.w;
        const ymax = Math.max(pdf(pr.s[0], pr.s[0], pr.s[1]) * NS, pdf(pr.u[0], pr.u[0], pr.u[1]) * NU) * 1.12;
        const Y = (d) => P.y + P.h - (d / ymax) * (P.h - 24);
        // arrastar o limiar
        const pt = Hh.pointer;
        if (pt.down && pt.inside && pt.x >= P.x && pt.x <= P.x + P.w && pt.y >= P.y && pt.y <= P.y + P.h) { const v = V.clamp((pt.x - P.x) / P.w, 0.05, 0.95); if (Math.abs(v - S.tau) > 0.004) { S.tau = Math.round(v * 100) / 100; sl.set(S.tau, true); upd(); } }
        axes(g, P, { xl: 'risco percebido →' });
        const fillTail = (m, s, N, col, from, to) => { g.beginPath(); g.moveTo(X(from), Y(0)); for (let k = 0; k <= 60; k++) { const x = from + (k / 60) * (to - from); g.lineTo(X(x), Y(pdf(x, m, s) * N)); } g.lineTo(X(to), Y(0)); g.closePath(); g.fillStyle = col; g.fill(); };
        const curve = (m, s, N, col) => { g.beginPath(); for (let k = 0; k <= 100; k++) { const x = k / 100; const yy = Y(pdf(x, m, s) * N); if (k) g.lineTo(X(x), yy); else g.moveTo(X(x), yy); } g.strokeStyle = col; g.lineWidth = 2.2; g.stroke(); };
        fillTail(pr.s[0], pr.s[1], NS, V.alpha(C.d, 0.45), S.tau, 1);
        fillTail(pr.u[0], pr.u[1], NU, V.alpha(RED(), 0.4), 0, S.tau);
        curve(pr.s[0], pr.s[1], NS, C.a);
        curve(pr.u[0], pr.u[1], NU, C.c);
        V.text(g, 'seguros (250)', X(pr.s[0]), Y(pdf(pr.s[0], pr.s[0], pr.s[1]) * NS) - 8, { size: 11, font: 'mono', align: 'center', color: C.a });
        V.text(g, 'inseguros (200)', X(pr.u[0]), Y(pdf(pr.u[0], pr.u[0], pr.u[1]) * NU) - 8, { size: 11, font: 'mono', align: 'center', color: C.c });
        V.line(g, X(S.tau), P.y + 4, X(S.tau), P.y + P.h, { color: C.ink, lw: 2 });
        V.text(g, '← atende', X(S.tau) - 6, P.y + 12, { size: 11, font: 'mono', align: 'right', color: C.muted });
        V.text(g, 'recusa →', X(S.tau) + 6, P.y + 12, { size: 11, font: 'mono', color: C.muted });
        // gotas: prompts caindo no eixo
        DROPS.forEach((d) => {
          const m = d.safe ? pr.s : pr.u;
          const x = V.clamp(m[0] + d.z * m[1], 0.01, 0.99);
          const f = (t * 0.35 + d.ph) % 1;
          const yy = V.lerp(P.y + 20, P.y + P.h - 6, V.ease.in(Math.min(1, f * 1.3)));
          const refused = x > S.tau;
          const wrong = d.safe ? refused : !refused;
          const col = d.safe ? C.a : C.c;
          if (f * 1.3 < 1) V.dot(g, X(x), yy, 3.4, V.alpha(col, 0.85));
          else { V.dot(g, X(x), P.y + P.h - 6, 4, wrong ? RED() : col); if (wrong) V.text(g, '✕', X(x), P.y + P.h - 14, { size: 11, align: 'center', color: RED() }); }
        });
        // matriz 2×2
        const c = cnt(S.tau, S.prof);
        const M = [[NS - c.over, c.over], [c.harm, NU - c.harm]];
        let mx, my2, cw2;
        if (!narrow) { mx = w * 0.58 + 70; my2 = 40; cw2 = Math.min(78, (w - mx - 16) / 2 - 3); }
        else { mx = 70; my2 = h * 0.5 + 26; cw2 = Math.min(70, (w * 0.55 - 80) / 2); }
        const cellCol = [[V.alpha(C.a, 0.3), V.alpha(C.d, 0.55)], [V.alpha(RED(), 0.5), V.alpha(C.a, 0.3)]];
        const cellLab = [['ok', 'sobre-recusa'], ['cumpre indevido', 'ok']];
        V.text(g, 'atende', mx + cw2 / 2, my2 - 6, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        V.text(g, 'recusa', mx + cw2 * 1.5 + 4, my2 - 6, { size: 10.5, font: 'mono', align: 'center', color: C.muted });
        ['seguro', 'inseguro'].forEach((l, i) => V.text(g, l, mx - 6, my2 + i * (cw2 + 4) + cw2 / 2 + 4, { size: 10.5, font: 'mono', align: 'right', color: C.muted }));
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
          const x = mx + j * (cw2 + 4), y = my2 + i * (cw2 + 4);
          V.box(g, x, y, cw2, cw2, { r: 7, fill: cellCol[i][j], stroke: V.alpha(C.muted, 0.4), lw: 1 });
          V.text(g, fmtInt(M[i][j]), x + cw2 / 2, y + cw2 / 2 + 3, { size: 14, font: 'mono', weight: 700, align: 'center', color: C.ink });
          V.text(g, cellLab[i][j], x + cw2 / 2, y + cw2 - 7, { size: 9.5, font: 'mono', align: 'center', color: C.ink, maxW: cw2 - 4 });
        }
        // fronteira: sobre-recusa × cumprimento indevido
        let fx, fy, fw, fh;
        if (!narrow) { fx = w * 0.58 + 40; fy = my2 + 2 * cw2 + 40; fw = w - fx - 18; fh = h - fy - 34; }
        else { fx = w * 0.6 + 4; fy = my2; fw = w - fx - 12; fh = 2 * cw2 + 4; }
        const F = { x: fx, y: fy, w: fw, h: fh };
        axes(g, F, { xl: 'sobre-recusa (0–60%)', yl: 'cumpre indevido', yOff: 10 });
        const FX = (v) => F.x + Math.min(1, v / 0.6) * F.w, FY = (v) => F.y + F.h - Math.min(1, v / 0.6) * F.h;
        const roc = (P2, col, lw, alpha) => { g.beginPath(); for (let k = 0; k <= 80; k++) { const tau = 0.02 + (k / 80) * 0.96; const cc = cnt(tau, P2); const xx = FX(cc.over / NS), yy = FY(cc.harm / NU); if (k) g.lineTo(xx, yy); else g.moveTo(xx, yy); } g.strokeStyle = V.alpha(col, alpha); g.lineWidth = lw; g.stroke(); };
        roc(S.prof === 'cai' ? 'rlhf' : 'cai', S.prof === 'cai' ? C.b : C.a, 1.4, 0.5);
        roc(S.prof, S.prof === 'cai' ? C.a : C.b, 2.2, 1);
        V.glow(g, FX(c.over / NS), FY(c.harm / NU), 12, C.d, 0.9);
        V.dot(g, FX(c.over / NS), FY(c.harm / NU), 4.5, C.d);
      });
    })();

    /* ================================================ FIG 10 — calculadora do laço SL-CAI */
    (function () {
      const fig = root.querySelector('#fig-custo');
      if (!fig) return;
      adaptH(fig, 322, 392);
      const S = { P: 182831, R: 4, L: 250, F: 1500, thr: 1000, cache: true, B: 16 };
      const LP = 40, LR = 30;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Prompts', options: [{ v: 5000, label: '5.000' }, { v: 20000, label: '20.000' }, { v: 182831, label: '182.831 (paper)' }], value: 182831, onChange: (v) => { S.P = v; upd(); } });
      nocase(UI.slider(ctl, { label: 'revisões R', min: 1, max: 4, step: 1, value: 4, onInput: (v) => { S.R = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'tokens por geração', min: 100, max: 800, step: 50, value: 250, onInput: (v) => { S.L = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'preâmbulo few-shot', min: 0, max: 8000, step: 100, value: 1500, fmt: (v) => fmtInt(v) + ' tok', onInput: (v) => { S.F = v; upd(); } }));
      nocase(UI.slider(ctl, { label: 'vazão de decode', min: 200, max: 4000, step: 100, value: 1000, fmt: (v) => fmtInt(v) + ' tok/s', onInput: (v) => { S.thr = v; upd(); } }));
      UI.toggle(ctl, { label: 'prefix caching', value: true, onChange: (v) => { S.cache = v; gB.show(v); upd(); } });
      const gB = UI.group(ctl);
      nocase(UI.seg(gB.el, { label: 'grão do hit (tokens)', options: [{ v: 1, label: '1' }, { v: 16, label: '16 · vLLM' }, { v: 256, label: '256 · DS' }, { v: 1600, label: '1600 · Qwen' }, { v: 3584, label: '3584 · GLM' }], value: 16, onChange: (v) => { S.B = v; upd(); } }));
      const rG = UI.readout(ctl, 'gerações', '—');
      const rX = UI.readout(ctl, 'corte do prefill', '—');
      const rT = UI.readout(ctl, 'tempo estimado', '—');
      function calls() {
        // contexto (tokens) no início de cada chamada: preâmbulo + prompt + (saída + pedido) das anteriores
        const out = []; let ctx = S.F + LP;
        out.push({ ctx, kind: 'resp' });
        for (let j = 0; j < S.R; j++) {
          ctx += S.L + LR; out.push({ ctx, kind: 'crit' });
          ctx += S.L + LR; out.push({ ctx, kind: 'rev' });
        }
        return out;
      }
      const floorB = (x, B) => Math.floor(x / B) * B;
      // tokens já em cache no início da chamada ci: blocos cheios do preâmbulo (chamada 0) ou de
      // tudo o que a chamada anterior viu e GEROU (o KV da saída foi calculado no decode)
      const cachedAt = (cs, ci, B) => (ci === 0 ? floorB(S.F, B) : floorB(cs[ci - 1].ctx + S.L, B));
      const preCached = (cs, B) => S.P * cs.reduce((a, c, ci) => a + c.ctx - cachedAt(cs, ci, B), 0) + floorB(S.F, B);
      function totals() {
        const cs = calls();
        const nG = cs.length;
        const dec = S.P * nG * S.L;
        const preNo = S.P * cs.reduce((a, c) => a + c.ctx, 0);
        const preYes = preCached(cs, S.B);
        const preIdeal = preCached(cs, 1);
        const preOld = S.P * (LP + (nG - 1) * (S.L + LR)) + S.F; // se a saída anterior não ficasse no cache
        const pre = S.cache ? preYes : preNo;
        const tSec = dec / S.thr + pre / (10 * S.thr);
        return { nG, dec, preNo, preYes, preIdeal, preOld, pre, tSec, cs };
      }
      const hrs = (s) => (s >= 3600 * 48 ? fmt(s / 86400, 1) + ' dias' : s >= 3600 ? fmt(s / 3600, 1) + ' h' : fmt(s / 60, 0) + ' min');
      function upd() {
        const T = totals();
        rG.set(fmtInt(S.P * T.nG));
        const x = T.preNo / T.preYes;
        rX.set(S.cache ? fmt(x, 1) + '×' : 'desligado', !S.cache ? null : x >= 20 ? 'ok' : x >= 4 ? 'warn' : 'bad');
        rT.set(hrs(T.tSec));
      }
      upd();
      const big = (x) => (x >= 1e9 ? fmt(x / 1e9, 2) + ' bi' : x >= 1e6 ? fmt(x / 1e6, 1) + ' mi' : fmtInt(x));
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 600;
        const T = totals();
        const cs = T.cs;
        const ci = Math.floor((t / 1.1) % cs.length);
        const tx = 14, tW = w - 28, ty = 42, th = 28;
        V.text(g, 'contexto da chamada ' + (ci + 1) + ' de ' + cs.length + ' (um prompt)' + (S.cache ? ' · grão ' + fmtInt(S.B) : ''), tx, ty - 22, { size: 11.5, font: 'mono', color: C.muted, maxW: tW });
        const maxCtx = cs[cs.length - 1].ctx + S.L;
        const sc = tW / maxCtx;
        const segs = [['preâmbulo', S.F, C.b], ['prompt', LP, C.d]];
        for (let k = 0; k < ci; k++) {
          const kind = cs[k].kind;
          segs.push([kind === 'crit' ? 'crít.' : kind === 'resp' ? 'resp.' : 'rev.', S.L, kind === 'crit' ? C.f : kind === 'resp' ? C.c : C.a]);
          segs.push(['pedido', LR, C.muted]);
        }
        const cachedUpTo = S.cache ? cachedAt(cs, ci, S.B) : 0;
        const hatch = (x0, wd, col) => { g.save(); g.beginPath(); g.rect(x0, ty, wd, th); g.clip(); g.strokeStyle = V.alpha(col, 0.5); g.lineWidth = 1; for (let k = -th; k < wd; k += 6) { g.beginPath(); g.moveTo(x0 + k, ty + th); g.lineTo(x0 + k + th, ty); g.stroke(); } g.restore(); };
        let xx = tx, acc = 0;
        segs.forEach((sg) => {
          const sw = sg[1] * sc;
          if (sw > 0.5) {
            const cTok = V.clamp(cachedUpTo - acc, 0, sg[1]);
            const cw = cTok * sc;
            if (sw - cw > 0.5) V.box(g, xx + cw, ty, Math.max(1, sw - cw - 1), th, { r: 3, fill: V.alpha(sg[2], 0.55), stroke: sg[2], lw: 1 });
            if (cw > 0.5) {
              V.box(g, xx, ty, Math.max(1, cw - (cTok >= sg[1] ? 1 : 0)), th, { r: 3, fill: V.alpha(sg[2], 0.14), stroke: sg[2], lw: 1 });
              if (cw > 6) hatch(xx, cw - 1, sg[2]);
            }
            if (sw > 46) V.text(g, sg[0], xx + sw / 2, ty + 18, { size: 10.5, font: 'mono', align: 'center', color: C.ink, maxW: sw - 4 });
          }
          xx += sw; acc += sg[1];
        });
        // fronteiras de bloco (quando cabem) e a fronteira do cache
        if (S.cache && S.B * sc >= 5) for (let k = S.B; k <= cs[ci].ctx; k += S.B) V.line(g, tx + k * sc, ty - 6, tx + k * sc, ty - 1, { color: V.alpha(C.ink, 0.55), lw: 1.2 });
        if (S.cache && cachedUpTo > 0) {
          const bx = tx + cachedUpTo * sc;
          V.line(g, bx, ty - 9, bx, ty + th + 3, { color: C.a, lw: 2 });
          V.text(g, 'cache até aqui', bx - 4, ty - 10, { size: 10, font: 'mono', align: 'right', color: C.a });
        }
        const gw = S.L * sc, gp = (t % 1.1) / 1.1;
        const gcol = cs[ci].kind === 'crit' ? C.f : cs[ci].kind === 'resp' ? C.c : C.a;
        V.box(g, xx, ty, Math.max(2, gw * gp), th, { r: 3, fill: V.alpha(gcol, 0.9) });
        V.text(g, 'decode', Math.min(tx + tW - 22, xx + gw / 2), ty + th + 14, { size: 10.5, font: 'mono', align: 'center', color: gcol });
        const nwThis = cs[ci].ctx - cachedUpTo;
        const msg = S.cache
          ? 'prefill desta chamada: ' + fmtInt(nwThis) + ' tok novos · ' + fmtInt(cachedUpTo) + ' em cache (hachurado)'
          : 'prefill desta chamada: ' + fmtInt(cs[ci].ctx) + ' tok (tudo de novo)';
        const usedA = V.wrap(g, msg, tx, ty + th + 32, narrow ? tW - 10 : tW - 70, { size: 11.5, font: 'mono', color: S.cache ? C.a : C.d, lineH: 15 });
        const by = ty + th + 52 + usedA;
        V.text(g, 'conjunto inteiro: ' + fmtInt(S.P) + ' prompts', tx, by - 8, { size: 12, font: 'mono', weight: 600, color: C.ink, maxW: tW });
        const rows = [['prefill sem cache', T.preNo, C.d], ['prefill com cache', T.preYes, C.a], ['decode', T.dec, C.b]];
        const mxv = Math.max(T.preNo, T.dec);
        const lw2 = narrow ? 122 : 150;
        rows.forEach(([l, v, col], i) => {
          const y = by + 6 + i * 34;
          const active = (i === 0 && !S.cache) || (i === 1 && S.cache) || i === 2;
          V.text(g, l, tx, y + 16, { size: 11, font: 'mono', color: active ? C.ink : C.muted, maxW: lw2 - 6 });
          const bwv = Math.max(2, (tW - lw2 - (narrow ? 92 : 100)) * (v / mxv));
          V.box(g, tx + lw2, y + 2, bwv, 22, { r: 4, fill: V.alpha(col, active ? 0.75 : 0.22) });
          V.text(g, big(v) + ' tok', tx + lw2 + bwv + 6, y + 17, { size: 11, font: 'mono', color: C.ink });
        });
        const ty2 = by + 6 + 3 * 34 + 16;
        const tDec = T.dec / S.thr, tPre = T.pre / (10 * S.thr);
        let yy = ty2 + V.wrap(g, 'tempo ≈ decode ' + hrs(tDec) + ' + prefill ' + hrs(tPre) + ' = ' + hrs(T.tSec), tx, ty2, tW, { size: 12, font: 'mono', weight: 600, color: C.ink, lineH: 17 });
        const xr = T.preNo / T.preYes;
        yy += 2 + V.wrap(g, 'gerações = ' + fmtInt(S.P) + ' × (1 + 2·' + S.R + ') = ' + fmtInt(S.P * T.nG) + ' · com cache (grão ' + fmtInt(S.B) + ') o prefill cai ' + fmt(xr, 1) + '×', tx, yy + 2, tW, { size: 11, font: 'mono', color: C.muted, lineH: 15 });
        V.wrap(g, 'compare: grão 1 (ideal) ' + fmt(T.preNo / T.preIdeal, 1) + '× · se a resposta anterior não ficasse no cache, só ' + fmt(T.preNo / T.preOld, 1) + '×', tx, yy + 2, tW, { size: 11, font: 'mono', color: C.muted, lineH: 15 });
      });
    })();
  },
});

LA.chapter({
  id: 'engenharia-de-receita',
  part: 9,
  order: 41,
  title: 'Engenharia de receita: imagens, overlays, patches e launchers',
  navTitle: 'Engenharia de receita',
  subtitle: 'Como transformar um motor que você não controla em algo que sobe igual, byte a byte, nos dois nós — e que prefere não subir a subir errado.',
  minutes: 65,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['Docker', 'digest', 'overlay', 'hotfix', 'source-exact', 'fail-closed', 'idempotência', '.env', 'launcher', 'CI', 'CHANGELOG', 'backport', 'TensorFold', 'patch -p0'],
  goals: [
    'Explicar por que uma receita fixa tudo pelo <b>conteúdo</b> (digests, commits, revisões) e como camadas, tags, índices multi-arch e o cache de build de uma imagem funcionam',
    'Escolher entre <b>patch no build, overlay de arquivo, bind-mount e hotfix de boot</b>, e saber por que desfazer um hotfix exige recriar o container, não reiniciá-lo',
    'Escrever patches <b>source-exact, fail-closed, idempotentes e atômicos</b> e compô-los na mesma ordem nos dois ranks, e comparar com a versão TensorFold do GLM, que assa <b>diffs unificados</b> na imagem, dá à imagem o <b>hash dos patches</b> como nome e confere a paridade no próprio motor',
    'Projetar o <b>.env</b>, a precedência de configuração e o <b>launcher</b> (preflight, worker → head, health, canário, códigos de saída) sem cair nas armadilhas do shell',
    'Provar tudo em <b>CPU com fixtures e CI</b>, registrar no CHANGELOG e aposentar hacks quando o upstream corrige, sabendo o que não dá para backportar',
  ],
  prereqs: ['vllm-arquitetura', 'nccl-rdma', 'checkpoints'],
  summary: [
    'Uma imagem é uma pilha de camadas endereçadas por SHA-256. Tag é apelido mutável, digest é identidade. As receitas fixam pelo conteúdo o que os patches pressupõem (a base do GLM e a imagem do DeepSeek por digest, motor e ExLlamaV3 por commit, pesos por revisão), cobrem com outra trava o que ainda é tag (o carimbo do GLM), põem o que muda pouco embaixo no Dockerfile (compilação antes do Python, carimbo por último) e comparam head e worker por <code>RootFS.Layers</code>.',
    'Extensões nativas são compiladas para a CPU do host (aarch64: stubs honestos no lugar do AVX) e para a GPU (<code>TORCH_CUDA_ARCH_LIST=12.1a</code>), com símbolos e testes verificados <b>no build</b>, o lugar mais barato para falhar.',
    'Uma mudança pode morar numa camada da imagem, num bind-mount ou na camada gravável. <code>docker restart</code> preserva a camada gravável, os mounts e o ambiente da criação: desfazer um hotfix exige <code>./stop</code> + <code>./start</code> (recriar).',
    'Um patch seguro é uma função parcial: aceita só os bytes stock ou patched pinados, monta e compila a pós-imagem em memória, publica por <code>fsync</code> + <code>os.replace</code> e se reverifica. Fora disso, recusa (fail-closed).',
    'Quando um patch reescreve a região ou a âncora de outro, a ordem (ex.: <code>GLM53_OVERLAY_ORDER</code>) passa a fazer parte da receita. Em TP, todos os ranks precisam dos mesmos bytes e knobs, senão as coletivas podem divergir e travar. Na versão TensorFold do GLM, os 53 patches são diffs unificados aplicados no build em ordem de nome (nove reescrevem a mesma lista), a tag da imagem é <code>v0.6.0-&lt;hash dos patches&gt;</code> com o digest fixado na release, e o motor compara na partida as configurações dos dois ranks e se recusa a subir se divergirem.',
    'O <code>.env</code> é código de shell: aspas somem, CRLF e BOM corrompem, <code>08</code> e números gigantes quebram a aritmética. Deixe no <code>.env</code> só átomos, valide e normalize, e monte JSON dentro do container (como o <code>--speculative-config</code> do DeepSeek).',
    'Launcher: trava, validação antes de parar, preflight, worker → head, espera com prazo, fail-fast, warmup e canário. Códigos de saída e política de restart são contratos documentados. Testes em CPU contra fixtures, CI e CHANGELOG sustentam o ciclo, e cada hack precisa de um jeito de morrer.',
  ],
  quiz: [
    { q: 'Seu <code>.env</code> usa <code>IMAGE=ghcr.io/exemplo/motor:0.1.1</code>. Alguém republica a tag <code>0.1.1</code> apontando para outra build. O que acontece no próximo <code>docker pull</code> num nó novo?', o: ['Nada: tags são imutáveis no registry', 'O nó recebe a imagem nova, e patches source-exact passam a recusar (ou, pior, um patch frouxo aplica sobre código diferente)', 'O Docker detecta a mudança e mantém a versão antiga', 'O pull falha com erro de digest'], a: 1, e: 'Tag é um ponteiro mutável. Só o pin por digest (<code>@sha256:…</code>) garante os mesmos bytes. Por isso a base do GLM e a imagem da lane A do DeepSeek são fixadas por digest: os patches pressupõem aquele conteúdo exato. Onde sobra uma tag, como a imagem de serving do GLM, outra trava (o carimbo) precisa recusar a imagem nova.' },
    { q: 'Por que o Dockerfile do GLM compila o <code>exllamav3_ext</code> <b>antes</b> de copiar o <code>overlay/exl3.py</code>?', o: ['Porque o Python precisa da extensão para ser copiado', 'Porque o cache de build refaz a camada alterada e todas acima dela: com o Python por cima, editar o <code>exl3.py</code> não recompila a extensão CUDA', 'Porque o <code>COPY</code> apaga as camadas anteriores', 'Porque o nvcc exige que o diretório esteja vazio'], a: 1, e: 'Mudar a camada \\(k\\) invalida o cache de \\(k\\) para cima. Ordenar do mais estável (compilação pesada) para o mais volátil (Python, label) deixa as edições comuns baratas. O carimbo vem por último pelo mesmo motivo.' },
    { q: 'Um patcher source-exact encontra, no alvo, exatamente o SHA-256 do seu pin <i>patched</i>. O que ele deve fazer?', o: ['Aplicar de novo, para garantir', 'Recusar com FAIL-CLOSED, porque o arquivo não é o stock', 'Sair 0 com “already-patched”, sem tocar no arquivo (idempotência)', 'Restaurar o stock e aplicar outra vez'], a: 2, e: 'O patch é uma função parcial com dois estados conhecidos. Stock → aplica. Patched → no-op com sucesso, e é isso que permite reaplicar os overlays a cada boot, ou no build e de novo no start. Qualquer outro hash → recusa.' },
    { q: 'Dois overlays, A e B, inserem código no mesmo ponto de um arquivo, e B foi escrito contra o arquivo já com A. Alguém inverte a ordem. Com patches por âncora fail-closed, o que acontece?', o: ['O arquivo final fica igual, porque patches sempre comutam', 'B não encontra sua âncora e aborta; o boot para nos dois ranks antes do <code>exec vllm serve</code>', 'B aplica numa posição aproximada', 'Só o rank 1 falha'], a: 1, e: 'Dois patches comutam quando nenhum reescreve o que o outro lê (a âncora) ou escreve (a região). Aqui B lê justamente o que A escreveu, então a ordem é parte da receita. A âncora exata transforma o erro de ordem numa falha segura e imediata, em vez de um arquivo torto.' },
    { q: 'O <code>.env</code> tem <code>MAX_NUM_SEQS=08</code> e o launcher faz <code>$(( MAX_NUM_SEQS * 7 ))</code>. O que o bash faz, e qual é a defesa do DeepSeek?', o: ['Calcula 56: o zero à esquerda é ignorado', 'Erro “value too great for base”, porque 0… é octal e 8 não é dígito octal; a defesa é validar e normalizar com <code>10#</code> antes da aritmética', 'Calcula 0', 'Trata 08 como string e concatena'], a: 1, e: 'Em aritmética de shell, um número começando com 0 é octal (<code>010</code> vale 8, <code>08</code> é erro). O <code>dspark_validate_numeric_knobs</code> rejeita não-inteiros, rejeita valores com mais de 10 dígitos antes de qualquer aritmética (um limite conservador: os máximos reais têm no máximo 7 dígitos, e assim um número gigante nunca chega a dar a volta nos 64 bits) e força base 10 com <code>10#</code>.' },
    { q: 'Depois do boot, <code>/health</code> responde 200, mas as respostas são lixo e a aceitação de rascunhos está perto de zero. O que as receitas fazem para isso não entrar em serviço?', o: ['Nada: /health 200 é a prova de saúde', 'Um canário pós-boot: “Reply with OK.” a temperatura 0 e aceitação &gt; 0 de ao menos 64 rascunhos; se falhar, o GLM sai 3 (DEGENERATE ENGINE) e não anuncia READY', 'Reiniciam o container automaticamente até melhorar', 'Aumentam o timeout do healthcheck'], a: 1, e: '<code>/health</code> prova que o processo responde, não que o modelo acerta. O canário do GLM pega a assinatura do #249. O <code>spec-accept-gate.sh</code> pega a do vllm#53030 (aceitação da posição 0 presa em ~1,00).' },
  ],

  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;

    /* ===================== utilitários locais ===================== */
    const FC = {};
    const font = (s, f, wt) => { const k = s + '|' + (f || 'sans') + '|' + (wt || 400); return FC[k] || (FC[k] = V.font(s, f || 'sans', wt || 400)); };
    const WM = new Map();
    function lines(g, s, maxW, size, f, wt) {
      const key = s + '|' + Math.round(maxW) + '|' + size + (f || '') + (wt || '');
      let L = WM.get(key);
      if (L) return L;
      g.font = font(size, f, wt);
      L = [];
      String(s).split('\n').forEach((par) => {
        const words = par.split(' ');
        let cur = '';
        for (const wd of words) {
          const tt = cur ? cur + ' ' + wd : wd;
          if (cur && g.measureText(tt).width > maxW) { L.push(cur); cur = wd; } else cur = tt;
        }
        L.push(cur);
      });
      if (WM.size > 1500) WM.clear();
      WM.set(key, L);
      return L;
    }
    function wtext(g, s, x, y, maxW, o = {}) {
      const size = o.size || 12, lh = o.lh || Math.round(size * 1.36);
      const L = lines(g, s, maxW, size, o.font, o.weight);
      const max = o.maxLines || 99;
      const n = Math.min(L.length, max);
      for (let i = 0; i < n; i++) {
        let ln = L[i];
        if (i === n - 1 && L.length > max) ln = ln.replace(/\s*\S{0,3}$/, '') + '…';
        V.text(g, ln, x, y + i * lh, { size, font: o.font, weight: o.weight, color: o.color, align: o.align });
      }
      return n * lh;
    }
    function tw(g, s, size, f, wt) { g.font = font(size, f, wt); return g.measureText(String(s)).width; }
    function fit(g, s, maxW, size, f, wt) {
      s = String(s);
      if (tw(g, s, size, f, wt) <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (tw(g, s.slice(0, mid) + '…', size, f, wt) <= maxW) lo = mid; else hi = mid - 1; }
      return s.slice(0, lo) + '…';
    }
    function pill(g, x, y, s, o = {}) {
      const size = o.size || 11, padX = o.padX == null ? 7 : o.padX, hh = o.h || size + 9;
      const w = tw(g, s, size, o.font || 'mono', o.weight || 500) + padX * 2;
      const xx = o.align === 'right' ? x - w : o.align === 'center' ? x - w / 2 : x;
      V.box(g, xx, y, w, hh, { r: o.r == null ? hh / 2 : o.r, fill: o.fill, stroke: o.stroke, lw: o.lw || 1 });
      V.text(g, s, xx + w / 2, y + hh / 2 + 0.5, { size, font: o.font || 'mono', weight: o.weight || 500, color: o.color, align: 'center', base: 'middle' });
      return w;
    }
    function icon(g, kind, x, y, r, col) {
      g.save();
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = Math.max(1.6, r * 0.26); g.lineCap = 'round'; g.lineJoin = 'round';
      if (kind === 'ok') { g.beginPath(); g.moveTo(x - r * 0.55, y + r * 0.02); g.lineTo(x - r * 0.12, y + r * 0.45); g.lineTo(x + r * 0.6, y - r * 0.45); g.stroke(); }
      else if (kind === 'bad') { g.beginPath(); g.moveTo(x - r * 0.45, y - r * 0.45); g.lineTo(x + r * 0.45, y + r * 0.45); g.moveTo(x + r * 0.45, y - r * 0.45); g.lineTo(x - r * 0.45, y + r * 0.45); g.stroke(); }
      else if (kind === 'lock') { V.rr(g, x - r * 0.62, y - r * 0.12, r * 1.24, r * 0.92, r * 0.2); g.fill(); g.beginPath(); g.arc(x, y - r * 0.14, r * 0.4, Math.PI, 0); g.stroke(); }
      else if (kind === 'lens') { g.beginPath(); g.arc(x - r * 0.15, y - r * 0.15, r * 0.45, 0, Math.PI * 2); g.stroke(); g.beginPath(); g.moveTo(x + r * 0.2, y + r * 0.2); g.lineTo(x + r * 0.62, y + r * 0.62); g.stroke(); }
      else if (kind === 'stop') {
        g.beginPath();
        for (let k = 0; k < 8; k++) { const a = Math.PI / 8 + k * Math.PI / 4; const px = x + Math.cos(a) * r * 0.72, py = y + Math.sin(a) * r * 0.72; if (k) g.lineTo(px, py); else g.moveTo(px, py); }
        g.closePath(); g.fill();
        g.strokeStyle = V.C().bg; g.lineWidth = Math.max(1.4, r * 0.2); g.beginPath(); g.moveTo(x - r * 0.32, y); g.lineTo(x + r * 0.32, y); g.stroke();
      }
      else if (kind === 'warn') { g.beginPath(); g.moveTo(x, y - r * 0.66); g.lineTo(x + r * 0.66, y + r * 0.52); g.lineTo(x - r * 0.66, y + r * 0.52); g.closePath(); g.stroke(); g.beginPath(); g.moveTo(x, y - r * 0.22); g.lineTo(x, y + r * 0.1); g.stroke(); V.dot(g, x, y + r * 0.3, Math.max(1.2, r * 0.09), col); }
      else if (kind === 'wait') { g.beginPath(); g.arc(x, y, r * 0.58, 0, Math.PI * 2); g.stroke(); g.beginPath(); g.moveTo(x, y - r * 0.34); g.lineTo(x, y); g.lineTo(x + r * 0.26, y + r * 0.2); g.stroke(); }
      else if (kind === 'info') { g.beginPath(); g.arc(x, y, r * 0.58, 0, Math.PI * 2); g.stroke(); V.dot(g, x, y - r * 0.26, Math.max(1.2, r * 0.09), col); g.beginPath(); g.moveTo(x, y - r * 0.05); g.lineTo(x, y + r * 0.3); g.stroke(); }
      else if (kind === 'skip') { g.beginPath(); g.moveTo(x - r * 0.45, y); g.lineTo(x + r * 0.45, y); g.stroke(); }
      else if (kind === 'bolt') { g.beginPath(); g.moveTo(x + r * 0.15, y - r * 0.7); g.lineTo(x - r * 0.4, y + r * 0.08); g.lineTo(x, y + r * 0.08); g.lineTo(x - r * 0.15, y + r * 0.7); g.lineTo(x + r * 0.4, y - r * 0.08); g.lineTo(x, y - r * 0.08); g.closePath(); g.fill(); }
      g.restore();
    }
    const KCOL = (C, k) => (k === 'ok' || k === 'pass' ? C.a : k === 'bad' || k === 'fail' || k === 'crash' ? C.c : k === 'warn' ? C.d : k === 'info' ? C.b : C.muted);
    function badge(g, kind, x, y, r) {
      const C = V.C(); const col = KCOL(C, kind);
      V.dot(g, x, y, r, V.alpha(col, 0.18));
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.strokeStyle = V.alpha(col, 0.7); g.lineWidth = 1.2; g.stroke();
      const map = { ok: 'ok', pass: 'ok', bad: 'bad', fail: 'bad', crash: 'bolt', warn: 'warn', info: 'info', skip: 'skip', off: 'skip', na: 'skip', wait: 'wait' };
      icon(g, map[kind] || 'info', x, y, r * 0.95, col);
    }
    function fnv(str) { let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; }
    const hx = (s, n = 8) => { let out = '', k = 0; while (out.length < n) out += fnv(s + '#' + k++).toString(16).padStart(8, '0'); return out.slice(0, n); };
    const onDark = (C) => (C.dark ? '#04150C' : '#ffffff');
    // altura do palco conforme a largura da figura (mobile precisa de mais altura)
    function respH(fig, fn) {
      const stage = fig.querySelector('.la-fig-stage');
      const apply = () => { const hh = fn(fig.clientWidth); if (hh && stage.style.height !== hh + 'px') stage.style.height = hh + 'px'; };
      apply();
      const ro = new ResizeObserver(apply); ro.observe(fig);
      if (LA.onCleanup) LA.onCleanup(() => ro.disconnect());
    }
    // select que encolhe dentro da barra de controles
    function fitSel(sel) {
      const el = sel.el, se = el.querySelector('select');
      el.style.maxWidth = '100%'; el.style.minWidth = '0'; el.style.flexWrap = 'wrap';
      if (se) { se.style.maxWidth = '100%'; se.style.minWidth = '0'; }
      return sel;
    }

    /* =====================================================================
       F1 — Do git clone ao primeiro token (stepper)
       ===================================================================== */
    const ST1 = [
      { n: 'Repositório', sub: 'código da receita', pin: 'Commit do repositório; cada mudança entra no CHANGELOG com a instrução operacional.', chk: 'CI em CPU: bash -n, py_compile, testes contra fixtures dos arquivos originais.', fail: 'PR vermelho não entra no main: o defeito nem chega a um nó.',
        d: 'A receita é um repositório git: launchers, Dockerfile, patches, testes, <code>.env.example</code>. No DeepSeek, cada push roda <code>scripts/ci-validate.sh</code> (só CPU). No GLM, o CHANGELOG registra versão, issue e número medido de cada mudança.' },
      { n: 'Imagem', sub: 'build ou pull', pin: 'Base @sha256, vLLM por commit, ExLlamaV3 @c5d9c657; carimbo da receita no label.', chk: 'Símbolos da extensão, 13 testes de CPU no build, carimbo comparado com o repositório.', fail: 'O build aborta e nenhuma imagem torta é criada.',
        d: 'GLM: <code>FROM …@sha256:905c0293…</code> + patches em <code>RUN</code> + <code>exllamav3_ext</code> para <code>12.1a</code> + <code>LABEL glm53.recipe.stamp</code>. O GLM serve por tag, protegida pelo carimbo. Qwen: imagem day-0 da NVIDIA por tag, nunca reconstruída. DeepSeek: Anemll <code>0.1.1</code> pinada por digest no <code>.env</code>.' },
      { n: 'Pesos', sub: 'checkpoint', pin: 'Revisão do Hub (hash de 40 hex), nunca “main”.', chk: 'Contagem de shards, tamanhos desreferenciados, SHA-256 quando pedido.', fail: 'Recusa snapshot incompleto antes de 10 minutos de carga.',
        d: 'Revisões fixas (DeepSeek <code>86f746b3…</code>, GLM <code>25a44fdb…</code> no retrato, <code>76c0b517…</code> desde <code>98cf18f</code>), baixadas no head. O GLM espera 120 shards (83 com o checkpoint novo) e confere os tamanhos no worker, porque o marcador de sync é “uma alegação, não uma prova”. Detalhes em <i>Checkpoints</i>.' },
      { n: 'Launcher', sub: '.env + checagens', pin: 'Regra de precedência explícita (quem vence: chamador ou .env).', chk: 'Números, IPs locais, SSH em BatchMode, GIDs RoCE, MemAvailable.', fail: 'Exit 2 (config inválida) antes de parar o par que estava bom.',
        d: 'O <code>start.sh</code> lê o <code>.env</code> com uma regra documentada, valida os números <i>antes</i> de derrubar o que está no ar e faz o preflight: <code>HEAD_IP</code> local, <code>ssh -o BatchMode=yes -o ConnectTimeout=10</code>, GID por HCA, <code>MemAvailable ≥ util × total + 2 GiB</code>.' },
      { n: 'Distribuição', sub: 'head → worker', pin: 'Mesma image key (RootFS.Layers) nos dois nós.', chk: 'Env 0600 copiado de forma atômica; preflight --check de cada patcher.', fail: 'Nenhum rank sobe com imagem ou patch divergente.',
        d: 'A imagem vai por <code>pull</code> ou <code>docker save --platform linux/arm64 | ssh worker docker load</code>. Compose, <code>.env</code>, patchers e proposer são sincronizados. No DeepSeek, <code>docker compose run --rm --no-deps --entrypoint python3 … --check</code> roda no worker e no head antes de subir qualquer rank.' },
      { n: 'Entrypoint', sub: 'overlays e hotfixes', pin: 'SHA-256 stock → patched, ou âncora exata; ordem fixa.', chk: 'compile() da pós-imagem e reverificação depois da escrita.', fail: '|| exit 1 antes do exec vllm serve.',
        d: 'Dentro de cada container, os patches rodam na mesma ordem: os 25 overlays da <code>GLM53_OVERLAY_ORDER</code> (26 desde o #203), as ~21 etapas do <code>command:</code> do DeepSeek, os bind-mounts <code>:ro</code> do Qwen. Qualquer recusa encerra o container antes do <code>exec vllm serve</code>.' },
      { n: 'vllm serve ×2', sub: 'worker → head', pin: 'Mesmos engine args e mesmos bytes nos dois ranks.', chk: '/health ou /v1/models, com prazo (DS: 100 × 15 s).', fail: 'Fail-fast: se o worker morrer, a espera é abortada.',
        d: 'O worker (rank 1, <code>--headless</code>) sobe primeiro e espera o rendezvous. O head (rank 0) traz TCPStore, scheduler e API. O launcher espera com prazo e desiste cedo se o container do worker morreu (GLM: 3 checagens, ~30 s).' },
      { n: 'Prova', sub: 'warmup + canário', pin: 'Formas de tráfego aquecidas antes do primeiro cliente.', chk: '“Reply with OK.” a temperatura 0 e aceitação de rascunhos > 0.', fail: 'Exit 3 DEGENERATE ENGINE: sem banner READY.',
        d: 'Warmup das formas que o tráfego vai usar (o JIT acontece agora, não durante o serviço) e canário: resposta sensata a temperatura 0 e rascunhos aceitos. Só então o banner READY. <code>/health</code> 200 não basta.' },
    ];
    const f1 = root.querySelector('#fig-metro');
    const st1 = UI.stepper(f1, ST1.map((s, i) => ({ t: s.n + ' · ' + s.sub, d: s.d })), { autoplay: 6500 });
    respH(f1, (fw) => (fw < 640 ? 560 : 430));
    V.fig(f1, (g, w, h, t) => {
      const C = V.C();
      const N = ST1.length, narrow = w < 640;
      const rows = narrow ? 4 : 2, cols = N / rows;
      const padX = narrow ? 30 : 76, top = narrow ? 34 : 44, rowH = narrow ? 50 : 86;
      const colW = (w - 2 * padX) / (cols - 1);
      const P = ST1.map((s, i) => { const r = Math.floor(i / cols), c = i % cols; const cc = r % 2 === 0 ? c : cols - 1 - c; return [padX + cc * colW, top + r * rowH]; });
      const segOf = (i) => {
        const a = P[i - 1], b = P[i];
        if (a[1] === b[1]) return { type: 'l', a, b };
        const r = Math.floor((i - 1) / cols); const dir = r % 2 === 0 ? 1 : -1; const off = narrow ? 22 : 38;
        return { type: 'c', a, b, c1: [a[0] + dir * off, a[1]], c2: [b[0] + dir * off, b[1]] };
      };
      const drawSeg = (s, col, lw, dash, off) => {
        g.beginPath(); g.moveTo(s.a[0], s.a[1]);
        if (s.type === 'l') g.lineTo(s.b[0], s.b[1]); else g.bezierCurveTo(s.c1[0], s.c1[1], s.c2[0], s.c2[1], s.b[0], s.b[1]);
        g.strokeStyle = col; g.lineWidth = lw; g.lineCap = 'round';
        if (dash) { g.setLineDash(dash); g.lineDashOffset = off || 0; }
        g.stroke(); g.setLineDash([]);
      };
      const cur = st1.i;
      for (let i = 1; i < N; i++) { const s = segOf(i); drawSeg(s, V.alpha(C.muted, 0.2), 7); if (i <= cur) drawSeg(s, V.alpha(C.a, 0.9), 3, [6, 7], -t * 18); }
      const R = narrow ? 11 : 14;
      P.forEach((p, i) => {
        const on = i === cur, past = i < cur;
        if (on) V.glow(g, p[0], p[1], R * 2.6, C.a, 0.5 + 0.25 * Math.sin(t * 3));
        g.beginPath(); g.arc(p[0], p[1], R, 0, Math.PI * 2);
        g.fillStyle = past ? V.alpha(C.a, 0.75) : on ? C.a : C.panel; g.fill();
        g.lineWidth = 2; g.strokeStyle = past || on ? C.a : V.alpha(C.muted, 0.6); g.stroke();
        V.text(g, String(i + 1), p[0], p[1] + 0.5, { size: narrow ? 11 : 12, font: 'mono', weight: 700, align: 'center', base: 'middle', color: past || on ? onDark(C) : C.muted });
        if (narrow) {
          const left = p[0] < w / 2;
          const lx = left ? p[0] + R + 8 : p[0] - R - 8, al = left ? 'left' : 'right';
          const lw = w / 2 - R - 30;
          V.text(g, fit(g, ST1[i].n, lw, 11.5, 'sans', 600), lx, p[1] - 1, { size: 11.5, weight: 600, align: al, color: on || past ? C.ink : C.muted });
          V.text(g, fit(g, ST1[i].sub, lw, 10), lx, p[1] + 12, { size: 10, align: al, color: C.muted });
        } else {
          const lw = Math.min(colW - 14, 190);
          V.text(g, fit(g, ST1[i].n, lw, 12.5, 'sans', 600), p[0], p[1] + R + 16, { size: 12.5, weight: 600, align: 'center', color: on || past ? C.ink : C.muted });
          V.text(g, fit(g, ST1[i].sub, lw, 11), p[0], p[1] + R + 31, { size: 11, align: 'center', color: C.muted });
        }
      });
      // pacote viajando
      const e = V.ease.inOut(V.clamp(st1.elapsed() / 1.1, 0, 1));
      let pk = P[cur];
      if (cur > 0 && e < 1) { const s = segOf(cur); pk = s.type === 'l' ? V.along(s.a[0], s.a[1], s.b[0], s.b[1], e) : V.bez(s.a, s.c1, s.c2, s.b, e); }
      const bob = Math.sin(t * 4) * 1.5;
      const px = pk[0], py = pk[1] - R - 12 + bob;
      V.box(g, px - 9, py - 9, 18, 18, { r: 4, fill: C.d, stroke: V.alpha(C.ink, 0.25) });
      V.line(g, px - 9, py - 2, px + 9, py - 2, { color: V.alpha(onDark(C), 0.6), lw: 1.4 });
      V.line(g, px, py - 9, px, py - 2, { color: V.alpha(onDark(C), 0.6), lw: 1.4 });
      // painel
      const S = ST1[cur];
      const y0 = narrow ? top + (rows - 1) * rowH + 34 : top + rowH + 84;
      const ph = h - y0 - 10;
      const fade = V.clamp(st1.elapsed() / 0.45, 0, 1);
      g.globalAlpha = fade;
      const items = [['lock', 'fixado', S.pin, C.a], ['lens', 'verificado', S.chk, C.b], ['stop', 'se falhar', S.fail, C.c]];
      V.text(g, 'ESTAÇÃO ' + (cur + 1) + '/8 · ' + S.n.toUpperCase(), 16, y0 - 8, { size: 10.5, font: 'mono', weight: 600, color: C.muted });
      if (w >= 760) {
        const gap = 12, cw = (w - 32 - 2 * gap) / 3;
        items.forEach(([ic, lab, txt, col], k) => {
          const x = 16 + k * (cw + gap);
          V.box(g, x, y0, cw, ph, { r: 12, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.45) });
          V.dot(g, x + 26, y0 + 27, 14, V.alpha(col, 0.18));
          icon(g, ic, x + 26, y0 + 27, 11, col);
          V.text(g, lab.toUpperCase(), x + 48, y0 + 31, { size: 11, font: 'mono', weight: 600, color: col });
          wtext(g, txt, x + 16, y0 + 64, cw - 32, { size: 14, lh: 20, color: C.ink, maxLines: 6 });
        });
      } else {
        const rh = ph / 3;
        items.forEach(([ic, lab, txt, col], k) => {
          const y = y0 + k * rh;
          V.box(g, 12, y + 2, w - 24, rh - 6, { r: 10, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.4) });
          V.dot(g, 32, y + rh / 2 - 1, 12, V.alpha(col, 0.18));
          icon(g, ic, 32, y + rh / 2 - 1, 9.5, col);
          V.text(g, lab.toUpperCase(), 52, y + 18, { size: 10, font: 'mono', weight: 600, color: col });
          wtext(g, txt, 52, y + 33, w - 72, { size: 12, lh: 15, color: C.ink, maxLines: Math.max(1, Math.floor((rh - 30) / 15)) });
        });
      }
      g.globalAlpha = 1;
    });

    /* =====================================================================
       F2 — Camadas, digests e o ponteiro da tag
       ===================================================================== */
    const f2 = root.querySelector('#fig-camadas');
    const LY = [
      { id: 'base', txt: 'FROM vllm-openai:glm53-flash-arm64-cu130@sha256:905c…', kind: 'base', cost: 0 },
      { id: 'nope', txt: 'RUN patch NoPE sparse-MLA (backend SM120)', kind: 'run', cost: 1 },
      { id: 'reg', txt: 'RUN registra “exl3” nos métodos de quantização', kind: 'run', cost: 1 },
      { id: 'ext', txt: 'RUN compila exllamav3_ext → cubins sm_121a', kind: 'compile', cost: 12 },
      { id: 'py', txt: 'COPY overlay/exl3.py + patchers', kind: 'copy', cost: 1 },
      { id: 'bake', txt: 'RUN aplica os overlays + 13 testes de CPU', kind: 'run', cost: 2 },
      { id: 'it', txt: 'RUN pip install --no-deps instanttensor==0.2.0', kind: 'run', cost: 1 },
    ];
    const byId2 = Object.fromEntries(LY.map((l) => [l.id, l]));
    const ORD2 = { r: ['base', 'nope', 'reg', 'ext', 'py', 'bake', 'it'], n: ['base', 'nope', 'reg', 'py', 'ext', 'bake', 'it'] };
    const S2 = { order: 'r', ver: {}, lab: 0, rebuilt: [], t0: -9, kind: 'none', retag: 0, tR: -9, shipped: false, ctd: false, tT: -9, tK: '', pub: '', pin: '' };
    LY.forEach((l) => { S2.ver[l.id] = 0; });
    function ident2() {
      const ord = ORD2[S2.order];
      const diffs = ord.map((id) => hx(id + ':' + S2.ver[id], 8));
      const chain = [];
      ord.forEach((id, i) => chain.push(i === 0 ? diffs[0] : hx(chain[i - 1] + ' ' + diffs[i], 8)));
      const cfg = hx('cfg' + chain[chain.length - 1] + 'L' + S2.lab, 8);
      const man = hx('man' + cfg, 8);
      return { ord, diffs, chain, cfg, man, layersKey: hx(diffs.join(','), 8) };
    }
    { const I0 = ident2(); S2.pub = I0.man; S2.pin = I0.man; }
    const c2 = UI.controls(f2);
    let H2 = null;
    function edit2(kind) {
      const ord = ORD2[S2.order];
      let from = -1;
      if (kind === 'py') from = ord.indexOf('py');
      else if (kind === 'cu') from = ord.indexOf('ext');
      else if (kind === 'base') from = 0;
      else if (kind === 'order') from = 3;
      S2.rebuilt = [];
      if (kind === 'stamp') S2.lab++;
      if (from >= 0) { for (let i = from; i < ord.length; i++) { S2.ver[ord[i]]++; S2.rebuilt.push(ord[i]); } S2.lab++; }
      S2.kind = kind; S2.t0 = H2 ? H2.t : 0;
      const cost = S2.rebuilt.reduce((a, id) => a + byId2[id].cost, 0);
      r2a.set(kind === 'none' ? '—' : kind === 'stamp' ? '0 (só o label)' : String(S2.rebuilt.length));
      r2b.set(kind === 'none' ? '—' : cost >= 10 ? 'pesado: recompila CUDA' : cost > 0 ? 'leve: só Python' : 'quase zero: metadado');
    }
    UI.seg(c2, { label: 'Mudança no repositório', options: [{ v: 'none', label: 'nada' }, { v: 'py', label: 'exl3.py' }, { v: 'cu', label: 'um .cu' }, { v: 'stamp', label: 'só o carimbo' }, { v: 'base', label: 'base nova' }], value: 'none', onChange: (v) => edit2(v) });
    UI.seg(c2, { label: 'Ordem do Dockerfile', options: [{ v: 'r', label: 'receita' }, { v: 'n', label: 'ingênua' }], value: 'r', onChange: (v) => { S2.order = v; edit2('order'); } });
    UI.button(c2, '↻ republicar a tag no registry', () => { S2.retag++; S2.pub = hx('push' + S2.retag, 8); S2.tR = H2 ? H2.t : 0; });
    UI.toggle(c2, { label: 'worker via save | load', value: false, onChange: (v) => { S2.shipped = v; S2.tK = v ? 'ship1' : 'ship0'; S2.tT = H2 ? H2.t : 0; } });
    UI.toggle(c2, { label: 'worker com store containerd', value: false, onChange: (v) => { S2.ctd = v; S2.tK = v ? 'ctd1' : 'ctd0'; S2.tT = H2 ? H2.t : 0; } });
    const r2a = UI.readout(c2, 'camadas refeitas', '—');
    const r2b = UI.readout(c2, 'custo', '—');
    respH(f2, (fw) => (fw < 720 ? 640 : 480));
    H2 = V.fig(f2, (g, w, h, t) => {
      const C = V.C(); const I = ident2(); const wide = w >= 720;
      const kc = { base: C.b, run: C.a, compile: C.d, copy: C.c };
      const sx = 14, sw = wide ? Math.round(w * 0.56) - 22 : w - 28;
      const stackTop = 26, stackBot = wide ? h - 14 : 286;
      const nL = I.ord.length, labH = 24;
      const lh = Math.min(46, (stackBot - stackTop - labH - 6) / nL);
      const fs = wide ? 11.5 : 10.5;
      // LABEL (metadado)
      const lbY = stackBot - nL * lh - labH - 2;
      V.text(g, 'IMAGEM · CAMADAS DE BAIXO PARA CIMA', sx, lbY - 10, { size: 10.5, font: 'mono', color: C.muted });
      const labLit = S2.kind === 'stamp' || S2.rebuilt.length ? V.clamp((t - S2.t0 - S2.rebuilt.length * 0.28) / 0.35, 0, 1) : 0;
      g.setLineDash([4, 4]);
      V.box(g, sx, lbY, sw, labH - 2, { r: 7, fill: V.alpha(C.muted, 0.06 + 0.1 * labLit), stroke: labLit > 0 ? C.d : V.alpha(C.muted, 0.5), lw: labLit > 0 ? 1.8 : 1 });
      g.setLineDash([]);
      V.text(g, fit(g, 'LABEL glm53.recipe.stamp · metadado, não é camada de arquivos', sw - 90, fs - 0.5), sx + 10, lbY + labH / 2 + 3, { size: fs - 0.5, color: C.muted });
      V.text(g, hx('stamp' + S2.lab, 8), sx + sw - 10, lbY + labH / 2 + 3, { size: 10.5, font: 'mono', align: 'right', color: labLit > 0 ? C.d : C.muted });
      for (let i = 0; i < nL; i++) {
        const id = I.ord[i], L = byId2[id];
        const y = stackBot - (i + 1) * lh;
        const reb = S2.rebuilt.indexOf(id);
        const lit = reb >= 0 ? V.clamp((t - S2.t0 - reb * 0.28) / 0.35, 0, 1) : 0;
        const col = kc[L.kind];
        V.box(g, sx, y + 2, sw, lh - 4, { r: 8, fill: V.alpha(col, 0.1 + 0.12 * lit), stroke: lit > 0 ? C.d : V.alpha(col, 0.5), lw: lit > 0 ? 2 : 1 });
        if (lit > 0 && lit < 1) { g.save(); V.rr(g, sx, y + 2, sw, lh - 4, 8); g.clip(); const gx = sx + lit * (sw + 80) - 80; const gr = g.createLinearGradient(gx, 0, gx + 80, 0); gr.addColorStop(0, V.alpha(C.d, 0)); gr.addColorStop(0.5, V.alpha(C.d, 0.35)); gr.addColorStop(1, V.alpha(C.d, 0)); g.fillStyle = gr; g.fillRect(gx, y, 80, lh); g.restore(); }
        const idW = 70;
        V.text(g, fit(g, L.txt, sw - idW - 40, fs), sx + 10, y + lh / 2 + 4, { size: fs, color: C.ink, weight: L.kind === 'compile' ? 600 : 400 });
        V.text(g, I.diffs[i], sx + sw - 10, y + lh / 2 + 4, { size: 10.5, font: 'mono', align: 'right', color: lit > 0 ? C.d : C.muted });
        if (reb >= 0 && lit >= 1) V.text(g, '↻', sx + sw - idW - 18, y + lh / 2 + 5, { size: 13, color: C.d, align: 'center', weight: 700 });
      }
      // painel direito / inferior
      const px = wide ? Math.round(w * 0.56) + 4 : 14, pw = wide ? w - px - 14 : w - 28;
      let y = wide ? 16 : stackBot + 22;
      const small = !wide;
      const fz = small ? 10 : 11;
      V.text(g, fit(g, small ? 'REGISTRY · o serving do GLM é uma tag' : 'REGISTRY (GHCR) · a imagem de serving do GLM é uma tag', pw, 10.5, 'mono'), px, y, { size: 10.5, font: 'mono', color: C.muted });
      y += 12;
      const rowH = small ? 22 : 27;
      const retagLit = S2.retag > 0;
      const pulse = retagLit ? V.clamp(1 - (t - S2.tR) / 1.5, 0, 1) : 0;
      const psz = small ? 9.5 : 10.5;
      const pillW = Math.max(...[':exl3-instanttensor', 'carimbo GLM', 'pin @sha256 (DS)'].map((l) => tw(g, l, psz, 'mono', 500) + 14)) + 24;
      const row = (label, col, txt, tcol) => {
        pill(g, px, y, label, { size: psz, fill: V.alpha(col, 0.15), stroke: V.alpha(col, 0.6), color: C.ink });
        V.arrow(g, px + pillW - 18, y + 9, px + pillW - 2, y + 9, { color: tcol, lw: 1.5, head: 6 });
        V.text(g, fit(g, txt, pw - pillW - 6, fz, 'mono'), px + pillW + 5, y + 13, { size: fz, font: 'mono', color: tcol });
      };
      row(':exl3-instanttensor', C.b, 'sha256:' + S2.pub + (retagLit ? '  (nova!)' : ''), retagLit ? C.d : C.ink);
      if (pulse > 0) V.glow(g, px + pillW + 40, y + 9, 40, C.d, pulse * 0.6);
      y += rowH;
      row('carimbo GLM', C.d, retagLit ? 'label ≠ repo → não adota ✓' : 'label = repo → adota ✓', C.a);
      y += rowH;
      row('pin @sha256 (DS)', C.a, 'sha256:' + S2.pin + '  (mesma ✓)', C.a);
      y += rowH;
      V.text(g, fit(g, 'sua build local → sha256:' + I.man, pw, fz, 'mono'), px, y + 10, { size: fz, font: 'mono', color: S2.rebuilt.length || S2.kind === 'stamp' ? C.d : C.muted });
      y += small ? 17 : 22;
      const nIdx = 'índice multi-arch →';
      V.text(g, nIdx, px, y + 10, { size: fz, color: C.muted });
      const iw = tw(g, nIdx, fz) + 6;
      const a1 = pill(g, px + iw, y, 'linux/arm64 ✓ GB10', { size: small ? 9.5 : 10, fill: V.alpha(C.a, 0.12), stroke: V.alpha(C.a, 0.5), color: C.ink });
      pill(g, px + iw + a1 + 6, y, 'linux/amd64', { size: small ? 9.5 : 10, fill: V.alpha(C.muted, 0.08), stroke: V.alpha(C.muted, 0.4), color: C.muted });
      y += small ? 34 : 38;
      V.text(g, fit(g, 'HEAD × WORKER · worker via ' + (S2.shipped ? 'save | load' : 'pull') + ' · store ' + (S2.ctd ? 'containerd' : 'overlay2'), pw, 10.5, 'mono'), px, y, { size: 10.5, font: 'mono', color: C.muted });
      y += 10;
      const rowsI = [
        ['RootFS.Layers', 'layers:' + I.layersKey, 'layers:' + I.layersKey],
        ['RepoDigests', 'sha256:' + S2.pub, S2.shipped ? '— (perdido)' : 'sha256:' + S2.pub],
        ['.Id', 'sha256:' + I.cfg, S2.ctd ? 'índice:' + hx('ctd' + I.man, 8) : 'sha256:' + I.cfg],
        ['image key', 'layers:' + I.layersKey, 'layers:' + I.layersKey],
      ];
      const cw0 = small ? 84 : 98, cw1 = (pw - cw0 - 22) / 2;
      const th = small ? 18 : 22;
      V.text(g, 'campo', px, y + 12, { size: 10, font: 'mono', color: C.muted });
      V.text(g, 'head', px + cw0, y + 12, { size: 10, font: 'mono', color: C.muted });
      V.text(g, 'worker', px + cw0 + cw1, y + 12, { size: 10, font: 'mono', color: C.muted });
      y += th - 2;
      rowsI.forEach(([k, a, b]) => {
        const eq = a === b, key = k === 'image key';
        if (key) V.box(g, px - 4, y + 1, pw + 4, th, { r: 6, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.4) });
        V.text(g, k, px, y + th / 2 + 4, { size: small ? 10 : 11, font: 'mono', weight: key ? 700 : 500, color: key ? C.a : C.ink });
        V.text(g, fit(g, a, cw1 - 6, small ? 9.5 : 10.5, 'mono'), px + cw0, y + th / 2 + 4, { size: small ? 9.5 : 10.5, font: 'mono', color: C.muted });
        V.text(g, fit(g, b, cw1 - 4, small ? 9.5 : 10.5, 'mono'), px + cw0 + cw1, y + th / 2 + 4, { size: small ? 9.5 : 10.5, font: 'mono', color: eq ? C.muted : C.c });
        icon(g, eq ? 'ok' : 'bad', px + pw - 8, y + th / 2 + 1, 6.5, eq ? C.a : C.c);
        y += th;
      });
      y += small ? 10 : 12;
      const TMSG = {
        ship1: 'O docker save | load leva as camadas, mas não os RepoDigests: o worker nunca “bate” com a tag de origem. RootFS.Layers continua igual, e é isso que a image key compara.',
        ship0: 'Worker com pull: os RepoDigests coincidem. Mesmo assim, a image key usa RootFS.Layers, que vale em todos os caminhos.',
        ctd1: 'Store containerd: o .Id vira o digest do índice, não o da config. Mesmos bytes, .Id diferente; RootFS.Layers não muda.',
        ctd0: 'Os dois nós com overlay2: o .Id é o digest da config, igual dos dois lados.',
      };
      const recentT = S2.tK && t - S2.tT < 7 && S2.tT >= S2.tR;
      const msg = (S2.retag > 0 && t - S2.tR < 7 && !recentT) ? 'A tag agora aponta para outra imagem: um pull por :exl3-instanttensor traz bytes novos. O carimbo do GLM não bate e a imagem não é adotada; um pin @sha256 continuaria na mesma.'
        : recentT ? TMSG[S2.tK]
        : S2.kind === 'py' ? (S2.order === 'r' ? 'Editar o exl3.py refaz só as camadas de cima: a extensão CUDA fica no cache.' : 'Na ordem ingênua, a mesma edição de uma linha recompila a extensão CUDA inteira.')
        : S2.kind === 'cu' ? 'Mudar um kernel .cu exige recompilar: é a camada cara, e tudo acima dela é refeito.'
        : S2.kind === 'stamp' ? 'Só o label muda: nenhuma camada de arquivos é refeita e RootFS.Layers fica igual. Por isso o carimbo vem por último.'
        : S2.kind === 'base' ? 'Base nova: todos os ChainIDs mudam. Os patchers do build conferem a fonte nova e abortam se ela não bater.'
        : S2.kind === 'order' ? 'Mudar a ordem muda o Dockerfile: tudo a partir da primeira instrução diferente é refeito.'
        : 'Escolha o que mudou no repositório e veja quais camadas são refeitas (↻) e quais identidades mudam.';
      const msz = small ? 12 : 12.5, mlh = small ? 16 : 17;
      const room = Math.max(2, Math.floor((h - y - 6) / mlh));
      wtext(g, msg, px, y + 10, pw, { size: msz, lh: mlh, color: C.ink, maxLines: Math.min(5, room) });
    });

    /* =====================================================================
       F3 — A alfândega do build
       ===================================================================== */
    const f3 = root.querySelector('#fig-alfandega');
    const G3 = [
      { n: 'fonte', s: '@c5d9c657' }, { n: 'stubs AVX', s: 'aarch64' }, { n: 'fat kernels', s: 'E2/E3/FAST' },
      { n: 'nvcc', s: 'ARCH_LIST' }, { n: 'símbolos', s: 'hasattr' }, { n: 'imagem', s: 'linux/…' },
      { n: 'container', s: 'no GB10' }, { n: 'extensão', s: 'cubins' }, { n: 'serve', s: '1º request' },
    ];
    const S3 = { host: 'arm', stubs: true, arch: '12.1a', sym: true, forget: false, t0: 0 };
    function eval3() {
      const st = G3.map(() => 'pass');
      let fail = -1, msg = '', where = '', level = 'ok';
      if (S3.host === 'x86') st[1] = 'na';
      if (!S3.sym) st[4] = 'skip';
      if (S3.host === 'arm' && !S3.stubs) { fail = 1; msg = 'avx2_target.cpp: intrínsecos AVX2/AVX-512 só existem em x86. O compilador ARM recusa.'; where = 'no build (barato)'; }
      else if (S3.forget && S3.sym) { fail = 4; msg = 'assert hasattr(exllamav3_ext, "exl3_fat_moe_gateup") falhou: o m.def foi esquecido no bindings.cpp.'; where = 'no build (barato)'; }
      else if (S3.host === 'x86') { fail = 6; msg = 'Imagem linux/amd64 num GB10 (arm64): exec format error.'; where = 'no boot'; }
      else if (S3.arch === '10.0') { fail = 7; msg = 'no kernel image is available for execution on the device: só existe cubin sm_100 (sem PTX).'; where = 'na primeira chamada do kernel (caro)'; }
      else if (S3.forget) { fail = 7; msg = "AttributeError: module 'exllamav3_ext' has no attribute 'exl3_fat_moe_gateup'."; where = 'na carga do modelo (caro)'; }
      if (fail >= 0) { st[fail] = 'fail'; for (let i = fail + 1; i < st.length; i++) st[i] = 'off'; level = 'bad'; }
      else if (S3.arch === '12.0') { msg = 'Roda: cubin sm_120 é compatível com o 12.1, mas sem as instruções exclusivas do alvo 12.1a.'; where = 'nenhum erro'; level = 'warn'; }
      else { msg = 'Cubins sm_121a nativos, símbolos conferidos no build: o pacote chega inteiro à GPU.'; where = 'nenhum erro'; }
      return { st, fail, msg, where, level };
    }
    const c3 = UI.controls(f3);
    let H3 = null;
    const re3 = () => { S3.t0 = H3 ? H3.t : 0; };
    UI.seg(c3, { label: 'Host do build', options: [{ v: 'arm', label: 'aarch64 (Grace)' }, { v: 'x86', label: 'x86_64' }], value: 'arm', onChange: (v) => { S3.host = v; re3(); } });
    UI.seg(c3, { label: 'TORCH_CUDA_ARCH_LIST', options: ['10.0', '12.0', '12.1a'], value: '12.1a', onChange: (v) => { S3.arch = v; re3(); } });
    UI.toggle(c3, { label: 'stubs AVX (patch_exl3_ext_aarch64.py)', value: true, onChange: (v) => { S3.stubs = v; re3(); } });
    UI.toggle(c3, { label: 'verificar símbolos no build', value: true, onChange: (v) => { S3.sym = v; re3(); } });
    UI.toggle(c3, { label: 'simular um m.def esquecido', value: false, onChange: (v) => { S3.forget = v; re3(); } });
    respH(f3, (fw) => (fw < 700 ? 530 : 420));
    H3 = V.fig(f3, (g, w, h, t) => {
      const C = V.C(); const E = eval3(); const wide = w >= 700;
      const stopAt = E.fail >= 0 ? E.fail : G3.length - 1;
      const step = 0.5, total = (stopAt + 1) * step + 2.6;
      if (t - S3.t0 > total) S3.t0 = t;
      const el = t - S3.t0;
      let pos = [], bw, bh;
      if (wide) {
        const x0 = 26, x1 = w - 26, gap = (x1 - x0) / G3.length;
        bw = Math.min(104, gap - 12); bh = 64;
        G3.forEach((_, i) => pos.push([x0 + gap * (i + 0.5), 100]));
      } else {
        const cw = (w - 20) / 3; bw = cw - 10; bh = 50;
        const ys = [62, 128, 224];
        G3.forEach((_, i) => { const r = Math.floor(i / 3), c = i % 3; const cc = r === 1 ? 2 - c : c; pos.push([10 + cw * (cc + 0.5), ys[r]]); });
      }
      // zonas
      const midZ = wide ? (pos[5][0] + pos[6][0]) / 2 : 0;
      const zb = wide ? [12, 48, midZ - 3 - 12, bh + 68] : [6, 22, w - 12, 158];
      const zr = wide ? [midZ + 3, 48, w - 12 - (midZ + 3), bh + 68] : [6, 192, w - 12, 76];
      V.box(g, zb[0], zb[1] - 14, zb[2], zb[3], { r: 12, fill: V.alpha(C.b, 0.05), stroke: V.alpha(C.b, 0.3) });
      V.box(g, zr[0], zr[1] - 14, zr[2], zr[3], { r: 12, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.3) });
      V.text(g, 'FÁBRICA · docker build', zb[0] + 10, zb[1] - 1, { size: 10, font: 'mono', color: C.b });
      V.text(g, 'GB10 · docker run', wide ? zr[0] + 10 : pos[6][0] + 8, zr[1] - 1, { size: wide ? 10 : 9.5, font: 'mono', color: C.a });
      // trilho
      for (let i = 1; i < pos.length; i++) {
        // trilho só no vão entre as bordas das caixas (não cruza o texto)
        const [ax, ay] = pos[i - 1], [bx2, by2] = pos[i];
        const dx = bx2 - ax, dy = by2 - ay;
        const f = Math.min(dx ? (bw / 2 + 3) / Math.abs(dx) : Infinity, dy ? (bh / 2 + 3) / Math.abs(dy) : Infinity);
        if (f >= 0.5) continue;
        V.line(g, ax + dx * f, ay + dy * f, bx2 - dx * f, by2 - dy * f, { color: V.alpha(C.muted, 0.35), lw: 3, dash: [4, 5], dashOffset: -t * 14 });
      }
      const shown = Math.floor(el / step);
      G3.forEach((G, i) => {
        const [x, y] = pos[i];
        let s = E.st[i];
        if (i > shown) s = 'wait';
        const col = s === 'pass' ? C.a : s === 'fail' ? C.c : s === 'wait' ? C.muted : C.muted;
        V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 10, fill: C.panel });
        V.box(g, x - bw / 2, y - bh / 2, bw, bh, { r: 10, fill: V.alpha(col, s === 'fail' ? 0.18 : 0.08), stroke: V.alpha(col, s === 'wait' ? 0.3 : 0.7), lw: s === 'fail' ? 2 : 1.2 });
        const sub = i === 3 ? 'ARCH ' + S3.arch : i === 5 ? (S3.host === 'arm' ? 'linux/arm64' : 'linux/amd64') : i === 1 && S3.host === 'x86' ? 'não precisa' : G.s;
        V.text(g, G.n, x, y - 6, { size: wide ? 12 : 11, weight: 600, align: 'center', color: s === 'wait' ? C.muted : C.ink });
        V.text(g, fit(g, sub, bw - 10, wide ? 10 : 9.5, 'mono'), x, y + 9, { size: wide ? 10 : 9.5, font: 'mono', align: 'center', color: C.muted });
        const bk = s === 'pass' ? 'pass' : s === 'fail' ? 'fail' : s === 'wait' ? 'wait' : 'skip';
        badge(g, bk, x + bw / 2 - 2, y - bh / 2 + 2, 8);
      });
      // pacote
      const u = Math.min(el / step, stopAt);
      const i0 = Math.floor(u), fr = u - i0;
      let pp = pos[i0];
      if (i0 < pos.length - 1 && fr > 0) pp = V.along(pos[i0][0], pos[i0][1], pos[i0 + 1][0], pos[i0 + 1][1], V.ease.inOut(fr));
      const failing = E.fail >= 0 && u >= stopAt;
      const shake = failing ? Math.sin(t * 40) * 2 * V.clamp(1 - (el - stopAt * step) / 0.6, 0, 1) : 0;
      const pyk = pp[1] - bh / 2 - 2;
      V.box(g, pp[0] - 9 + shake, pyk - 8, 18, 16, { r: 4, fill: failing ? C.c : C.d });
      if (failing) icon(g, 'bad', pp[0] + shake, pyk, 6, onDark(C));
      // chip GB10
      const cy = wide ? 196 : 286, chH = wide ? 74 : 62;
      const chX = 14, chW = wide ? Math.min(430, w * 0.46) : w - 28;
      V.box(g, chX, cy, chW, chH, { r: 10, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.45) });
      for (let k = 0; k < 9; k++) { const x = chX + 16 + k * (chW - 32) / 8; V.line(g, x, cy - 5, x, cy, { color: V.alpha(C.a, 0.4), lw: 2 }); V.line(g, x, cy + chH, x, cy + chH + 5, { color: V.alpha(C.a, 0.4), lw: 2 }); }
      V.text(g, 'GB10 · aarch64 · compute capability 12.1', chX + 12, cy + 18, { size: wide ? 11.5 : 10.5, font: 'mono', weight: 600, color: C.ink });
      const keys = [['sm_100', 'bad', 'não roda'], ['sm_120', 'warn', 'compat.'], ['sm_121a', 'ok', 'nativo']];
      const kw = (chW - 24) / 3;
      keys.forEach(([k, st, lab], j) => {
        const x = chX + 12 + j * kw, on = (S3.arch === '10.0' && j === 0) || (S3.arch === '12.0' && j === 1) || (S3.arch === '12.1a' && j === 2);
        const col = st === 'ok' ? C.a : st === 'warn' ? C.d : C.c;
        V.box(g, x, cy + 28, kw - 8, chH - 36, { r: 8, fill: V.alpha(col, on ? 0.2 : 0.05), stroke: V.alpha(col, on ? 0.9 : 0.3), lw: on ? 2 : 1 });
        V.text(g, k, x + 8, cy + 28 + (chH - 36) / 2 + 4, { size: wide ? 11 : 10, font: 'mono', weight: 600, color: on ? C.ink : C.muted });
        if (wide) V.text(g, lab, x + kw - 16, cy + 28 + (chH - 36) / 2 + 4, { size: 10, align: 'right', color: col });
        else icon(g, st === 'ok' ? 'ok' : st === 'warn' ? 'warn' : 'bad', x + kw - 20, cy + 28 + (chH - 36) / 2, 6, col);
      });
      // mensagem
      const mx = wide ? chX + chW + 16 : 14, my = wide ? cy : cy + chH + 14, mw = wide ? w - mx - 14 : w - 28, mh = wide ? h - my - 14 : h - my - 8;
      const done = el >= (stopAt + 1) * step - 0.2;
      const lcol = E.level === 'bad' ? C.c : E.level === 'warn' ? C.d : C.a;
      V.box(g, mx, my, mw, mh, { r: 10, fill: V.alpha(lcol, done ? 0.1 : 0.04), stroke: V.alpha(lcol, done ? 0.6 : 0.25) });
      if (done) {
        badge(g, E.level === 'bad' ? 'bad' : E.level === 'warn' ? 'warn' : 'ok', mx + 20, my + 20, 10);
        V.text(g, fit(g, E.level === 'bad' ? 'FALHA · pego ' + E.where : E.level === 'warn' ? 'PASSA COM RESSALVA' : 'OK', mw - 50, 11, 'mono', 700), mx + 38, my + 24, { size: 11, font: 'mono', weight: 700, color: lcol });
        const used = wtext(g, E.msg, mx + 14, my + 44, mw - 28, { size: wide ? 13 : 12, lh: wide ? 18 : 16, color: C.ink, maxLines: wide ? 4 : 3 });
        const stages = wide ? ['build', 'boot', 'carga / 1º uso', 'em serviço'] : ['build', 'boot', 'carga/uso', 'serviço'];
        const hit = E.fail < 0 ? -1 : E.fail <= 5 ? 0 : E.fail === 6 ? 1 : 2;
        const by = my + 44 + used + (wide ? 14 : 8), sw4 = (mw - 28) / 4, bh4 = wide ? 26 : 24;
        V.text(g, wide ? 'ONDE O DEFEITO APARECE (barato → caro)' : 'ONDE APARECE (barato → caro)', mx + 14, by, { size: 10, font: 'mono', color: C.muted });
        stages.forEach((sg, k) => {
          const x = mx + 14 + k * sw4, col = [C.a, C.d, C.f, C.c][k], on = k === hit;
          V.box(g, x, by + 8, sw4 - 6, bh4, { r: 6, fill: V.alpha(col, on ? 0.35 : 0.08), stroke: V.alpha(col, on ? 1 : 0.35), lw: on ? 2 : 1 });
          V.text(g, fit(g, sg, sw4 - 12, wide ? 11 : 10.5, 'sans', on ? 700 : 400), x + (sw4 - 6) / 2, by + 8 + bh4 / 2 + 4, { size: wide ? 11 : 10.5, align: 'center', weight: on ? 700 : 400, color: on ? C.ink : C.muted });
        });
        if (hit < 0) V.text(g, 'nenhum: o pacote chegou inteiro', mx + 14, by + bh4 + 26, { size: 11, color: C.a });
      } else {
        V.text(g, 'o pacote atravessa os portões…', mx + 14, my + 24, { size: 12, color: C.muted });
      }
    });

    /* =====================================================================
       F4 — Onde mora a mudança
       ===================================================================== */
    const f4 = root.querySelector('#fig-onde');
    const MECH4 = {
      build: { label: 'build (RUN)', intro: 'Patch no docker build: a mudança fica numa camada da imagem (GLM: NoPE sparse-MLA, registro do exl3).' },
      copy: { label: 'COPY inteiro', intro: 'Arquivo inteiro copiado numa camada da imagem (lane B do DeepSeek: 18 arquivos).' },
      bind: { label: 'bind-mount :ro', intro: 'Arquivo patcheado no host e montado por cima do pacote (Qwen: -v files/…:…:ro).' },
      hotfix: { label: 'hotfix no boot', intro: 'O entrypoint edita o vLLM instalado, na camada gravável (lane A do DeepSeek; overlays do GLM no start).' },
    };
    const S4 = { mech: 'hotfix' };
    let H4 = null;
    function log4(kind, txt) { S4.log.unshift({ kind, txt, t: H4 ? H4.t : 0 }); if (S4.log.length > 4) S4.log.pop(); }
    function reset4() {
      Object.assign(S4, { created: false, writable: false, mount: false, gateEnv: true, gateBox: true, base: 1, drift: false, stale: false, failed: false, n: 0, cid: '', log: [], anim: null });
      log4('info', MECH4[S4.mech].intro + ' Comece por ▶ ./start.');
    }
    function create4(re, nb) {
      const m = S4.mech;
      S4.n++; S4.cid = hx('ctr' + S4.n + m, 6);
      S4.created = true; S4.gateBox = S4.gateEnv; S4.writable = false; S4.mount = false; S4.failed = false; S4.stale = false;
      if (m === 'hotfix') {
        if (S4.gateBox) {
          if (nb) { S4.failed = true; S4.created = false; log4('ok', 'Base nova: o patcher encontra um SHA-256 desconhecido → FAIL-CLOSED, exit 1 antes do exec vllm serve. Seguro: re-porte e re-pine o patch.'); return; }
          S4.writable = true; log4('ok', (re ? 'Recriado' : 'Criado') + ': a cadeia do entrypoint aplica o hotfix na camada gravável → scheduler.py patcheado.');
        } else log4('ok', 'Recriado com o gate OFF: camada gravável nova, bytes stock da imagem → stock. É assim que se desfaz um hotfix.');
      } else if (m === 'bind') {
        if (S4.gateBox) {
          S4.mount = true;
          if (nb) { S4.stale = true; log4('warn', 'Base nova, mas o .orig em files/ é da imagem antiga (extract_from_image reaproveita o arquivo): o mount deriva da base velha. Apague os .orig e reextraia.'); }
          else log4('ok', (re ? 'Recriado' : 'Criado') + ' com -v files/…/scheduler.py:…:ro: o mount cobre o arquivo da imagem.');
        } else log4('ok', 'Recriado sem o mount: o Python volta a ver o scheduler.py stock da imagem.');
      } else {
        if (nb) {
          if (m === 'build') log4('ok', 'Base nova exige rebuild: os patchers do RUN conferem a fonte, o build aborta e nenhuma imagem torta é criada (o container sobe na imagem antiga).');
          else { S4.stale = true; log4('warn', 'Rebuild com o COPY do arquivo inteiro sobre a base nova: o arquivo antigo reverte, em silêncio, o que a base mudou nele.'); }
        } else if (re) log4('bad', 'Recriado, mas o patch está numa camada da imagem: continua patcheado. Para desfazer, só um rebuild sem ele.');
        else log4('ok', 'Criado a partir de uma imagem que já traz o patch.');
      }
    }
    function act4(a) {
      if (!H4) return;
      S4.anim = { a, t0: H4.t };
      const m = S4.mech;
      if (a === 'start') {
        if (S4.created) { log4('warn', 'O container já existe e está rodando: ./start não recria (o DeepSeek sai 3, “já rodando”).'); return; }
        create4(false, S4.drift); return;
      }
      if (a === 'restart') {
        if (!S4.created) { log4('warn', 'Não há container rodando: use ▶ ./start primeiro.'); return; }
        S4.gateEnv = false;
        if (m === 'hotfix') log4('bad', 'Gate OFF no .env + docker restart: mesmo container, mesma camada gravável e o ambiente da criação (o gate continua ON lá dentro). O patcher vê “already-patched”: continua patcheado.');
        else if (m === 'bind') log4('bad', 'docker restart mantém os mounts da criação: o arquivo montado continua cobrindo o original.');
        else log4('bad', 'O patch está numa camada da imagem: restart não muda nada. Só um rebuild sem ele.');
        return;
      }
      if (a === 'recreate') { S4.gateEnv = false; create4(true, S4.drift); return; }
      if (a === 'newbase') { S4.base++; S4.drift = true; S4.gateEnv = true; create4(true, true); }
    }
    const c4 = UI.controls(f4);
    UI.seg(c4, { label: 'Mecanismo', options: Object.keys(MECH4).map((k) => ({ v: k, label: MECH4[k].label })), value: 'hotfix', onChange: (v) => { S4.mech = v; reset4(); } });
    UI.button(c4, '▶ ./start', () => act4('start'), { primary: true });
    UI.button(c4, 'gate OFF + docker restart', () => act4('restart'));
    UI.button(c4, 'gate OFF + ./stop && ./start', () => act4('recreate'));
    UI.button(c4, 'base nova + ./start', () => act4('newbase'));
    reset4();
    function eff4() {
      if (!S4.created) return { src: 'none', state: S4.failed ? 'failed' : 'none' };
      if (S4.mount) return { src: 'mount', state: S4.stale ? 'stale' : 'patched' };
      if (S4.writable) return { src: 'writable', state: 'patched' };
      if (S4.mech === 'build' || S4.mech === 'copy') return { src: 'image', state: S4.stale ? 'stale' : 'patched' };
      return { src: 'image', state: 'stock' };
    }
    respH(f4, (fw) => (fw < 720 ? 580 : 450));
    H4 = V.fig(f4, (g, w, h, t) => {
      const C = V.C(); const wide = w >= 720; const E = eff4();
      const sx = 14, sw = wide ? Math.round(w * 0.5) - 20 : w - 28;
      const top = 44, bot = wide ? h - 18 : Math.round(h * 0.52);
      const colOf = (st) => (st === 'patched' ? C.a : st === 'stale' ? C.d : st === 'stock' ? C.muted : C.muted);
      // gate chips
      let gx = sx;
      gx += pill(g, gx, 10, '.env: gate ' + (S4.gateEnv ? 'ON' : 'OFF'), { size: 10.5, fill: V.alpha(S4.gateEnv ? C.a : C.muted, 0.12), stroke: V.alpha(S4.gateEnv ? C.a : C.muted, 0.5), color: C.ink }) + 8;
      if (S4.created) pill(g, gx, 10, 'no container (criação): ' + (S4.gateBox ? 'ON' : 'OFF'), { size: 10.5, fill: V.alpha(S4.gateBox !== S4.gateEnv ? C.d : C.muted, 0.12), stroke: V.alpha(S4.gateBox !== S4.gateEnv ? C.d : C.muted, 0.5), color: C.ink });
      // estratos
      const H = bot - top;
      const wrY = top + 22, wrH = H * 0.2, mnY = wrY + wrH + 8, mnH = H * 0.19, imY = mnY + mnH + 8, imH = bot - 8 - imY;
      const bWr = [wrY - 10, wrH + 6], bMnt = [mnY, mnH + 6], bImg = [imY - 6, imH + 16];
      const created = S4.created;
      const an = S4.anim, ae = an ? t - an.t0 : 9;
      g.setLineDash([5, 5]);
      V.box(g, sx, top + 4, sw, H - 4, { r: 12, stroke: V.alpha(created ? C.b : C.muted, created ? 0.7 : 0.3), lw: 1.4 });
      g.setLineDash([]);
      V.text(g, created ? 'container ' + S4.cid : S4.failed ? 'container saiu com exit 1' : 'sem container', sx + 12, top + 16, { size: 10.5, font: 'mono', weight: 600, color: created ? C.b : S4.failed ? C.c : C.muted });
      if (an && an.a === 'restart' && ae < 1.4 && created) { const cxr = sx + sw - 22, cyr = top + 14; g.save(); g.translate(cxr, cyr); g.rotate(ae * 6); g.beginPath(); g.arc(0, 0, 8, 0.3, Math.PI * 1.7); g.strokeStyle = C.d; g.lineWidth = 2; g.stroke(); g.restore(); V.text(g, 'restart', cxr - 14, cyr + 4, { size: 10, align: 'right', color: C.d, font: 'mono' }); }
      const lx = sx + 12, fx = sx + sw - (wide ? 110 : 96), fw = wide ? 92 : 80;
      // imagem
      for (let k = 0; k < 3; k++) { const y = bImg[0] + 6 + k * (bImg[1] - 16) / 3; V.box(g, lx, y, sw - 24, (bImg[1] - 16) / 3 - 4, { r: 6, fill: V.alpha(C.b, 0.08 + k * 0.03), stroke: V.alpha(C.b, 0.3) }); }
      V.text(g, (wide ? 'imagem · camadas read-only · base v' : 'imagem (read-only) · base v') + S4.base, lx + 10, bImg[0] + 22, { size: wide ? 11.5 : 10.5, color: C.ink, weight: 500 });
      // mounts
      V.box(g, lx, bMnt[0], sw - 24, bMnt[1] - 6, { r: 6, fill: V.alpha(C.c, S4.mount ? 0.1 : 0.03), stroke: V.alpha(C.c, S4.mount ? 0.5 : 0.2) });
      V.text(g, 'bind mounts (:ro)', lx + 10, bMnt[0] + 18, { size: wide ? 11.5 : 10.5, color: S4.mount ? C.ink : C.muted });
      // gravável
      const recreateFlash = an && (an.a === 'recreate' || an.a === 'newbase' || an.a === 'start') && ae < 1 ? 1 - ae : 0;
      V.box(g, lx, bWr[0] + 10, sw - 24, bWr[1] - 6, { r: 6, fill: V.alpha(C.d, (created ? 0.08 : 0.02) + recreateFlash * 0.2), stroke: V.alpha(C.d, created ? 0.5 : 0.2) });
      V.text(g, wide ? 'camada gravável (copy-on-write)' : 'camada gravável', lx + 10, bWr[0] + 28, { size: wide ? 11.5 : 10.5, color: created ? C.ink : C.muted });
      // coluna do arquivo
      const cells = [
        { y: bWr[0] + 10, hh: bWr[1] - 6, on: created && S4.writable, st: 'patched', src: 'writable' },
        { y: bMnt[0], hh: bMnt[1] - 6, on: created && S4.mount, st: S4.stale ? 'stale' : 'patched', src: 'mount' },
        { y: bImg[0] + 6, hh: bImg[1] - 16, on: true, st: (S4.mech === 'build' || S4.mech === 'copy') ? (S4.stale ? 'stale' : 'patched') : 'stock', src: 'image' },
      ];
      cells.forEach((c) => {
        const col = colOf(c.st), isEff = E.src === c.src && created;
        if (!c.on) { g.setLineDash([3, 4]); V.box(g, fx, c.y + 4, fw, c.hh - 8, { r: 6, stroke: V.alpha(C.muted, 0.3) }); g.setLineDash([]); return; }
        V.box(g, fx, c.y + 4, fw, c.hh - 8, { r: 6, fill: V.alpha(col, isEff ? 0.3 : 0.12), stroke: isEff ? col : V.alpha(col, 0.4), lw: isEff ? 2.2 : 1 });
        V.text(g, 'scheduler.py', fx + fw / 2, c.y + c.hh / 2 - 2, { size: 10, font: 'mono', align: 'center', color: C.ink });
        V.text(g, c.st === 'patched' ? 'patcheado' : c.st === 'stale' ? 'antigo ⚠' : 'stock', fx + fw / 2, c.y + c.hh / 2 + 11, { size: 10, align: 'center', color: col, weight: 600 });
      });
      if (created) {
        const ec = cells.find((c) => c.src === E.src);
        const ey = ec.y + ec.hh / 2;
        const ax = fx - 12;
        V.arrow(g, ax - 34 + Math.sin(t * 3) * 3, ey, ax, ey, { color: colOf(E.state), lw: 2, head: 7 });
        if (wide) V.text(g, 'Python vê', ax - 38, ey - 8, { size: 10, align: 'right', color: colOf(E.state), font: 'mono' });
        else V.text(g, 'Python vê', ax - 6, ey + 15, { size: 9.5, align: 'right', color: colOf(E.state), font: 'mono' });
      }
      // log
      const px = wide ? Math.round(w * 0.5) + 6 : 14, py0 = wide ? 14 : bot + 14, pw = wide ? w - px - 14 : w - 28;
      V.text(g, 'EVENTOS (mais recente em cima)', px, py0 + 2, { size: 10.5, font: 'mono', color: C.muted });
      let y = py0 + 16;
      const maxY = h - 8;
      S4.log.forEach((L, i) => {
        if (y > maxY - 20) return;
        const a = i === 0 ? 1 : 0.62;
        g.globalAlpha = a;
        badge(g, L.kind, px + 10, y + 8, 8);
        const hh = wtext(g, L.txt, px + 26, y + 12, pw - 30, { size: i === 0 ? (wide ? 13 : 12) : (wide ? 11.5 : 11), lh: i === 0 ? (wide ? 18 : 16) : 15, color: C.ink, maxLines: i === 0 ? 6 : 2 });
        g.globalAlpha = 1;
        y += hh + 10;
      });
    });

    /* =====================================================================
       F5 — A máquina de estados do patcher
       ===================================================================== */
    const f5 = root.querySelector('#fig-patcher');
    const FS5 = {
      stock: { sha: '0074271a6f26', size: 1242, label: 'stock (pin)', real: true },
      patched: { sha: '6452ce2eb94d', size: 2111, label: 'patched (pin)', real: true },
      drift: { sha: hx('drift', 12), size: 1249, label: 'upstream mudou 1 linha' },
      version: { sha: '0074271a6f26', size: 1242, label: 'stock, mas vLLM 0.27', real: true },
      trunc: { sha: hx('trunc', 12), size: 1024, label: 'meio escrito' },
    };
    const S5 = { base: 'stock', file: 'stock', write: 'atomic', crash: false, run: null, tmp: false };
    let H5 = null;
    function run5() {
      const f = S5.file, atomic = S5.write === 'atomic';
      const G = ['versão do vLLM', 'arquivo regular', 'SHA-256 + tamanho', 'âncora 1× · sem marcador', 'compile(pós-imagem)', 'SHA-256 da pós-imagem', atomic ? 'tmp no mesmo dir · fsync' : 'open(alvo, "w") · write', atomic ? 'os.replace (atômico)' : '(sem rename)', 'reverificação'];
      const res = G.map(() => 'off');
      const pass = (k) => { for (let i = 0; i <= k; i++) if (res[i] === 'off') res[i] = 'pass'; };
      let exit = '0', out = '', note = '', next = f, tmp = S5.tmp;
      if (f === 'version') { res[0] = 'fail'; exit = '1'; out = "FAIL-CLOSED: unsupported vllm version '0.27…'"; note = 'Nada foi escrito. O arquivo até pode ser o stock, mas o patch foi validado contra outra build do vLLM.'; }
      else if (f === 'patched') { pass(1); res[2] = 'pass'; for (let i = 3; i < G.length; i++) res[i] = 'skip'; out = 'dspark-rope-swa: already-patched'; note = 'Idempotência: o segundo encontro reconhece o pin patched e sai 0 sem tocar no arquivo.'; }
      else if (f === 'drift' || f === 'trunc') {
        pass(1); res[2] = 'fail'; exit = '1'; out = 'FAIL-CLOSED: unsupported target bytes sha256=' + FS5[f].sha.slice(0, 8) + '… size=' + FS5[f].size;
        note = f === 'trunc' ? 'Recusa correta, mas o estrago já existe: o vLLM nem importa (SyntaxError no rope.py). Só recriar o container volta ao stock.' : 'Fonte desconhecida: o patch pode estar errado aqui. A cadeia faz exit 1 e nenhum rank serve com meia correção.';
      } else {
        pass(5);
        if (S5.crash) {
          res[6] = 'crash'; exit = '—'; out = '⚡ processo morto durante a escrita';
          if (atomic) { next = 'stock'; tmp = true; note = 'O alvo nunca foi truncado: o rename não chegou a acontecer. Sobra um .dspark-rope-swa-XXXX inofensivo, e o próximo boot aplica de novo.'; }
          else { next = 'trunc'; note = 'open(…, "w") truncou o alvo antes de escrever: sobrou meio arquivo. Rode de novo para ver o próximo boot.'; }
        } else {
          pass(8); if (!atomic) res[7] = 'skip';
          out = 'dspark-rope-swa: applied'; next = 'patched';
          note = atomic ? 'Pós-imagem provada antes da escrita, publicada por os.replace e reverificada: patcheado.' : 'Funcionou desta vez, mas sem garantia: uma queda no meio deixaria o arquivo pela metade.';
        }
      }
      S5.run = { G, res, exit, out, note, from: f, next, t0: H5 ? H5.t : 0 };
      S5.file = next; S5.tmp = tmp;
    }
    const c5 = UI.controls(f5);
    UI.seg(c5, { label: 'Estado inicial do alvo', options: [{ v: 'stock', label: 'stock' }, { v: 'patched', label: 'já patcheado' }, { v: 'drift', label: 'upstream mudou' }, { v: 'version', label: 'outro vLLM' }, { v: 'trunc', label: 'meio escrito' }], value: 'stock', onChange: (v) => { S5.base = v; S5.file = v; S5.tmp = false; run5(); } });
    UI.seg(c5, { label: 'Escrita', options: [{ v: 'atomic', label: 'atômica' }, { v: 'naive', label: 'ingênua' }], value: 'atomic', onChange: (v) => { S5.write = v; S5.file = S5.base; S5.tmp = false; run5(); } });
    UI.toggle(c5, { label: '⚡ queda de energia durante a escrita', value: false, onChange: (v) => { S5.crash = v; S5.file = S5.base; S5.tmp = false; run5(); } });
    UI.button(c5, '↻ rodar de novo (próximo boot)', () => run5(), { primary: true });
    H5 = V.fig(f5, (g, w, h, t) => {
      const C = V.C(); const wide = w >= 720; const R5 = S5.run;
      const nG = R5 ? R5.G.length : 9;
      const el = R5 ? t - R5.t0 : 99;
      const step = 0.3;
      const doneAll = el > nG * step + 0.1;
      const fileNow = R5 && !doneAll ? R5.from : S5.file;
      const F = FS5[fileNow];
      // cartão do arquivo
      const cx = 14, cw = wide ? Math.round(w * 0.42) - 14 : w - 28, cy = 12, ch = wide ? h - 24 : 170;
      V.box(g, cx, cy, cw, ch, { r: 12, fill: V.alpha(C.b, 0.05), stroke: V.alpha(C.b, 0.35) });
      V.text(g, fit(g, 'vllm/models/deepseek_v4/common/rope.py', cw - 24, wide ? 11.5 : 10.5, 'mono'), cx + 12, cy + 20, { size: wide ? 11.5 : 10.5, font: 'mono', weight: 600, color: C.ink });
      V.text(g, 'sha256 ' + F.sha + '…  ·  ' + F.size.toLocaleString('pt-BR') + ' B', cx + 12, cy + 38, { size: wide ? 11 : 10, font: 'mono', color: fileNow === 'stock' || fileNow === 'patched' || fileNow === 'version' ? C.a : C.c });
      pill(g, cx + 12, cy + 46, F.label + (F.real ? '' : ' · ilustrativo'), { size: 10, fill: V.alpha(fileNow === 'patched' ? C.a : fileNow === 'stock' || fileNow === 'version' ? C.b : C.c, 0.14), stroke: V.alpha(fileNow === 'patched' ? C.a : fileNow === 'stock' || fileNow === 'version' ? C.b : C.c, 0.5), color: C.ink });
      if (S5.tmp && doneAll) pill(g, cx + cw - 12, cy + 46, '+ .dspark-rope-swa-XXXX', { size: 9.5, align: 'right', fill: V.alpha(C.muted, 0.1), stroke: V.alpha(C.muted, 0.4), color: C.muted });
      // matriz de bytes
      const gCols = wide ? 16 : 22;
      const gx0 = cx + 12, gy0 = cy + 76, gW = cw - 24, gH = ch - 76 - (wide ? 40 : 10);
      const gRows = wide ? Math.max(8, Math.floor(gH / (gW / gCols))) : 5;
      const cs = Math.min(gW / gCols, gH / gRows);
      const cap = 2400, tot = gCols * gRows, filled = Math.round(F.size / cap * tot);
      const rng = V.rng(fnv(fileNow === 'version' ? 'stock' : fileNow) % 100000 + 7);
      const regStart = Math.round(tot * 0.24);
      const regLen = fileNow === 'patched' ? Math.round(tot * (1011 / cap)) : Math.round(tot * (160 / cap));
      for (let k = 0; k < tot; k++) {
        const i = Math.floor(k / gCols), j = k % gCols;
        const x = gx0 + j * cs, y = gy0 + i * cs;
        const v = rng();
        let fill;
        if (k >= filled) { g.setLineDash([2, 3]); V.box(g, x + 1, y + 1, cs - 2, cs - 2, { r: 2, stroke: V.alpha(fileNow === 'trunc' && k < Math.round(1242 / cap * tot) ? C.c : C.muted, 0.25), lw: 0.8 }); g.setLineDash([]); continue; }
        const inReg = k >= regStart && k < regStart + regLen;
        if (inReg) fill = fileNow === 'patched' ? V.alpha(C.a, 0.45 + 0.35 * v) : V.alpha(C.b, 0.35 + 0.35 * v);
        else fill = V.alpha(C.muted, 0.14 + 0.3 * v);
        if (fileNow === 'drift' && k === regStart + regLen + 9) fill = C.d;
        g.fillStyle = fill; V.rr(g, x + 1, y + 1, cs - 2, cs - 2, 2); g.fill();
      }
      if (wide) {
        let lx = gx0; const ly = cy + ch - 22;
        [[C.b, 'região da âncora'], [C.a, 'região patcheada'], [C.d, 'byte mudado'], [C.muted, '≈ ' + Math.round(cap / tot) + ' B por célula']].forEach(([col, lab]) => { V.box(g, lx, ly - 8, 10, 10, { r: 2, fill: V.alpha(col, 0.7) }); V.text(g, lab, lx + 14, ly + 1, { size: 10.5, color: C.muted }); lx += tw(g, lab, 10.5) + 30; });
      }
      // portões
      const gxp = wide ? Math.round(w * 0.42) + 12 : 14, gyp = wide ? 14 : cy + ch + 12, gwp = wide ? w - gxp - 14 : w - 28;
      const two = !wide;
      const rowH = wide ? 25 : 21;
      const perCol = two ? 5 : nG;
      const colW = two ? gwp / 2 : gwp;
      const G = R5 ? R5.G : [];
      G.forEach((lab, i) => {
        const col = two ? Math.floor(i / perCol) : 0, row = two ? i % perCol : i;
        const x = gxp + col * colW, y = gyp + row * rowH;
        let st = R5.res[i];
        const shown = el > i * step;
        if (!shown) st = 'off';
        const kcol = KCOL(C, st === 'crash' ? 'crash' : st);
        badge(g, st === 'off' ? 'off' : st, x + 10, y + 10, 8);
        V.text(g, fit(g, (i + 1) + '. ' + lab, colW - 30, wide ? 12 : 10.5), x + 24, y + 14, { size: wide ? 12 : 10.5, color: st === 'off' || st === 'skip' ? C.muted : C.ink, weight: st === 'fail' || st === 'crash' ? 600 : 400 });
        if (st === 'fail' || st === 'crash') V.line(g, x + 24, y + 19, x + 24 + Math.min(colW - 34, tw(g, (i + 1) + '. ' + lab, wide ? 12 : 10.5)), y + 19, { color: kcol, lw: 1.5 });
      });
      const oy = gyp + (two ? perCol : nG) * rowH + 8;
      if (R5 && doneAll) {
        const ok = R5.exit === '0', crash = R5.exit === '—';
        const col = ok ? C.a : crash ? C.d : C.c;
        const oh = wide ? 46 : 40;
        V.box(g, gxp, oy, gwp, oh, { r: 8, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.55) });
        V.text(g, fit(g, '$ python3 hotfix-vllm-rope-swa-fix.py', gwp - 20, wide ? 10.5 : 9.5, 'mono'), gxp + 10, oy + 15, { size: wide ? 10.5 : 9.5, font: 'mono', color: C.muted });
        V.text(g, fit(g, R5.out + (crash ? '' : '   (exit ' + R5.exit + ')'), gwp - 20, wide ? 11.5 : 10, 'mono', 600), gxp + 10, oy + (wide ? 34 : 31), { size: wide ? 11.5 : 10, font: 'mono', weight: 600, color: col });
        wtext(g, R5.note, gxp + 2, oy + oh + 18, gwp - 4, { size: wide ? 12.5 : 11.5, lh: wide ? 17 : 15, color: C.ink, maxLines: Math.max(2, Math.floor((h - (oy + oh + 10)) / 16)) });
      } else if (R5) {
        V.text(g, 'avaliando os portões…', gxp + 4, oy + 18, { size: 12, color: C.muted });
      }
    });
    run5();

    /* =====================================================================
       F6 — Esteira de patches e paridade entre ranks
       ===================================================================== */
    const f6 = root.querySelector('#fig-ordem');
    const RW6 = 5, CL6 = 12;
    const PATCH6 = {
      A: { name: 'hybrid_prefix_hit', anchor: [[1, 3, 7]], want: 'S', write: [[1, 3, 7]], colK: 'a' },
      B: { name: 'apc_retention', anchor: [[1, 3, 7]], want: 'A', write: [[2, 3, 9]], colK: 'd' },
      C: { name: 'apc_no_store', anchor: [[3, 0, 3]], want: 'S', write: [[3, 0, 3], [4, 0, 3]], colK: 'c' },
      D: { name: 'dense_fp8 (reinstala exl3.py)', anchor: [], want: null, write: 'all', colK: 'b' },
      E: { name: 'exl3_ep_shard', anchor: [[2, 2, 8]], want: 'X', write: [[2, 2, 8]], colK: 'e' },
    };
    const ORD6 = { ABC: ['A', 'B', 'C'], CAB: ['C', 'A', 'B'], BAC: ['B', 'A', 'C'], DE: ['D', 'E'], ED: ['E', 'D'] };
    function sim6(ord, skip) {
      const tp3 = ord.indexOf('D') >= 0;
      let cells = Array.from({ length: RW6 }, () => Array(CL6).fill(tp3 ? 'X' : 'S'));
      const snaps = [cells.map((r) => r.slice())];
      let fail = -1;
      const seq = ord.filter((p) => p !== skip);
      for (let k = 0; k < seq.length; k++) {
        const P = PATCH6[seq[k]];
        let ok = true;
        P.anchor.forEach(([r, a, b]) => { for (let j = a; j <= b; j++) if (cells[r][j] !== P.want) ok = false; });
        if (!ok) { fail = k; break; }
        if (P.write === 'all') cells = Array.from({ length: RW6 }, () => Array(CL6).fill('X'));
        else P.write.forEach(([r, a, b]) => { for (let j = a; j <= b; j++) cells[r][j] = seq[k]; });
        snaps.push(cells.map((r) => r.slice()));
      }
      return { seq, snaps, fail, hash: fail >= 0 ? null : hx(cells.map((r) => r.join('')).join('/'), 8) };
    }
    const S6 = { ord: 'ABC', skip: false, t0: 0 };
    const c6 = UI.controls(f6);
    let H6 = null;
    fitSel(UI.select(c6, { label: 'Ordem dos patches', options: [{ v: 'ABC', label: 'A → B → C (receita)' }, { v: 'CAB', label: 'C → A → B (C comuta)' }, { v: 'BAC', label: 'B → A → C (invertido)' }, { v: 'DE', label: 'TP3: dense_fp8 → ep_shard' }, { v: 'ED', label: 'TP3: ep_shard → dense_fp8' }], value: 'ABC', onChange: (v) => { S6.ord = v; S6.t0 = H6 ? H6.t : 0; } }));
    UI.toggle(c6, { label: 'worker com um patch a menos (paridade quebrada)', value: false, onChange: (v) => { S6.skip = v; S6.t0 = H6 ? H6.t : 0; } });
    respH(f6, (fw) => (fw < 640 ? 410 : 470));
    H6 = V.fig(f6, (g, w, h, t) => {
      const C = V.C(); const wide = w >= 640;
      const ord = ORD6[S6.ord], tp3 = S6.ord === 'DE' || S6.ord === 'ED';
      const ref = sim6(tp3 ? ORD6.DE : ORD6.ABC, null);
      const sims = [sim6(ord, null), sim6(ord, S6.skip ? (tp3 ? 'E' : 'C') : null)];
      // linhas 0–2 = kv_cache_coordinator.py (A, B); linhas 3–4 = block_pool.py (C). No TP3, um arquivo só (exl3.py).
      const cs = wide ? 16 : 9, gapY = tp3 ? 0 : (wide ? 7 : 5);
      const rowY = (i) => i * cs + (i >= 3 ? gapY : 0);
      const gw = CL6 * cs, gh = RW6 * cs + gapY;
      const R = wide ? 12 : 10;
      const x0 = 18, badgeW = wide ? 128 : 92, x1 = w - badgeW - 22;
      const off = wide ? { ry: 42, gy: 64 } : { ry: 28, gy: 46 };
      const blockH = off.gy + gh + (wide ? 30 : 40);
      const tops = [4, 4 + blockH + (wide ? 10 : 8)];
      const T = 6.5, u = ((t - S6.t0) % (T + 1.6)) / T;
      const cellCol = (ch) => { const m = { S: V.alpha(C.muted, 0.28), X: V.alpha(C.b, 0.22), A: C.a, B: C.d, C: C.c, D: V.alpha(C.b, 0.6), E: C.e }; return m[ch]; };
      const hashBox = [];
      ['rank 0 · head', 'rank 1 · worker'].forEach((lab, r) => {
        const top = tops[r], Sm = sims[r];
        const ry = top + off.ry, gy = top + off.gy;
        V.text(g, lab + (r === 1 && S6.skip ? '  (sem ' + (tp3 ? 'E' : 'C') + ')' : ''), x0, top + 11, { size: 11, font: 'mono', weight: 600, color: r === 1 && S6.skip ? C.c : C.muted });
        // esteira
        const beltY = gy + gh + 4;
        V.box(g, x0, beltY, x1 - x0, 8, { r: 4, fill: V.alpha(C.muted, 0.14) });
        for (let k = 0; k < 40; k++) { const xx = x0 + ((k * 22 + t * 30) % (x1 - x0)); V.line(g, xx, beltY + 1, xx + 5, beltY + 7, { color: V.alpha(C.muted, 0.35), lw: 1.2 }); }
        const n = Sm.seq.length;
        // estações espalhadas pelo trecho que a borda direita do arquivo percorre
        const stX = Sm.seq.map((_, k) => x0 + gw + 8 + (k + 0.5) * ((x1 - gw - 4 - x0 - 16) / n));
        // robôs
        Sm.seq.forEach((p, k) => {
          const P = PATCH6[p], col = C[P.colK];
          const failed = Sm.fail === k;
          const rx = stX[k];
          V.line(g, rx, ry + R, rx, gy - 3, { color: V.alpha(failed ? C.c : col, 0.6), lw: 2 });
          V.dot(g, rx, ry, R, V.alpha(failed ? C.c : col, 0.22));
          g.beginPath(); g.arc(rx, ry, R, 0, Math.PI * 2); g.strokeStyle = failed ? C.c : col; g.lineWidth = 1.6; g.stroke();
          V.text(g, p, rx, ry + 0.5, { size: wide ? 12 : 11, font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
          if (wide) V.text(g, fit(g, P.name, Math.min(150, (x1 - gw - 4 - x0 - 16) / n - 8), 10, 'mono'), rx, ry - R - 6, { size: 10, font: 'mono', align: 'center', color: C.muted });
          if (failed) icon(g, 'bad', rx + R + 4, ry - R + 2, 6, C.c);
        });
        // arquivo(s) em movimento
        const endX = x1 - gw - 4;
        let gxp = x0 + V.clamp(u, 0, 1) * (endX - x0);
        const stopX = Sm.fail >= 0 ? stX[Sm.fail] - gw - 10 : endX;
        gxp = Math.min(gxp, stopX);
        let passed = 0;
        stX.forEach((sx) => { if (gxp + gw / 2 >= sx - gw / 2) passed++; });
        if (Sm.fail >= 0) passed = Math.min(passed, Sm.fail);
        const snap = Sm.snaps[Math.min(passed, Sm.snaps.length - 1)];
        const shake = Sm.fail >= 0 && gxp >= stopX ? Math.sin(t * 36) * 1.5 : 0;
        const pad = wide ? 1.6 : 1.2;
        for (let i = 0; i < RW6; i++) for (let j = 0; j < CL6; j++) {
          g.fillStyle = cellCol(snap[i][j]); V.rr(g, gxp + j * cs + shake, gy + rowY(i), cs - pad, cs - pad, wide ? 2.5 : 1.5); g.fill();
        }
        // contorno dos arquivos
        const fileBox = (r0, r1) => { V.box(g, gxp + shake - 3, gy + rowY(r0) - 3, gw + 6 - pad, rowY(r1) - rowY(r0) + cs + 6 - pad, { r: 5, stroke: V.alpha(C.muted, 0.4), lw: 1 }); };
        if (tp3) fileBox(0, 4); else { fileBox(0, 2); fileBox(3, 4); }
        // âncora da próxima estação
        if (passed < n && (Sm.fail < 0 || passed <= Sm.fail)) {
          const P = PATCH6[Sm.seq[passed]];
          P.anchor.forEach(([rr, a, b]) => { g.setLineDash([3, 3]); g.strokeStyle = V.alpha(C.ink, 0.8); g.lineWidth = 1.4; g.strokeRect(gxp + a * cs - 1.5 + shake, gy + rowY(rr) - 1.5, (b - a + 1) * cs + 1.5 - pad, cs + 1.5 - pad); g.setLineDash([]); });
        }
        const flab = tp3 ? 'exl3.py (ilustrativo)' : wide ? 'arquivos (ilustrativos): kv_cache_coordinator.py em cima (A, B) · block_pool.py embaixo (C)' : 'ilustrativos: em cima o coordenador (A, B); embaixo block_pool.py (C)';
        if (wide) V.text(g, fit(g, flab, x1 - x0 - 6, 10, 'mono'), x0, beltY + 22, { size: 10, font: 'mono', color: C.muted });
        else wtext(g, flab, x0, beltY + 21, x1 - x0 - 6, { size: 10, lh: 12.5, color: C.muted, maxLines: 2 });
        // hash
        const bx = w - badgeW - 12, by = gy + gh / 2 - 20;
        hashBox.push(by);
        const arrived = u >= 0.98 || Sm.fail >= 0;
        const okH = Sm.hash && arrived;
        const hcol = Sm.fail >= 0 ? C.c : okH ? (Sm.hash === ref.hash ? C.a : C.d) : C.muted;
        V.box(g, bx, by, badgeW, 40, { r: 8, fill: V.alpha(hcol, 0.1), stroke: V.alpha(hcol, 0.55) });
        V.text(g, Sm.fail >= 0 ? (arrived ? 'FAIL-CLOSED' : '…') : 'sha ' + (arrived ? Sm.hash : '……'), bx + badgeW / 2, by + 16, { size: wide ? 11 : 10, font: 'mono', weight: 600, align: 'center', color: C.ink });
        V.text(g, Sm.fail >= 0 ? (arrived ? 'exit 1' : '') : arrived ? (Sm.hash === ref.hash ? '= receita' : '≠ receita') : '', bx + badgeW / 2, by + 31, { size: 10, align: 'center', color: Sm.fail >= 0 ? C.c : Sm.hash === ref.hash ? C.a : C.d });
      });
      // cabo NCCL
      const bx = w - badgeW / 2 - 12, yA = hashBox[0] + 44, yB = hashBox[1] - 4;
      const arrived = u >= 0.98;
      const anyFail = sims[0].fail >= 0 || sims[1].fail >= 0;
      const same = sims[0].hash && sims[0].hash === sims[1].hash;
      let lk = 'wait';
      if (arrived || anyFail) lk = anyFail ? 'none' : same ? 'ok' : 'bad';
      const lcol = lk === 'ok' ? C.a : lk === 'bad' ? C.c : C.muted;
      V.line(g, bx, yA, bx, yB, { color: V.alpha(lcol, 0.8), lw: 3, dash: lk === 'ok' ? [6, 6] : lk === 'bad' ? [3, 7] : [2, 6], dashOffset: lk === 'ok' ? -t * 30 : 0 });
      V.text(g, 'NCCL', bx + 8, (yA + yB) / 2 + 4, { size: 10, font: 'mono', color: lcol });
      if (lk === 'bad') icon(g, 'wait', bx - 14, (yA + yB) / 2, 9, C.c);
      // desfecho
      const oy = tops[1] + blockH + (wide ? 10 : 8);
      let msg, kind;
      if (sims[0].fail >= 0 && sims[1].fail >= 0) { kind = 'ok'; msg = 'Âncora ausente: B foi escrito contra o arquivo com A. Sem A, B não encontra o que procura e aborta nos dois ranks (exit 1 antes do exec). Falha segura e imediata.'; }
      else if (sims[0].fail >= 0 || sims[1].fail >= 0) { kind = 'bad'; msg = 'Um rank aborta e o outro sobe sozinho, esperando um parceiro que nunca chega ao rendezvous.'; }
      else if (sims[0].hash !== sims[1].hash) { kind = 'bad'; msg = 'Ranks com bytes diferentes: o grupo sobe, mas os ranks podem lançar coletivas diferentes (ou em outra ordem), e o NCCL espera para sempre. Mesmos patches, mesma ordem, nos dois ranks.'; }
      else if (S6.ord === 'ED') { kind = 'bad'; msg = 'Nenhum erro na hora do patch: o dense_fp8 reinstala o exl3.py inteiro e apaga a edição do ep_shard' + (S6.skip ? ' (e o worker sem E chega ao mesmo arquivo errado)' : '') + '. O sintoma só aparece na carga: “dim 0 size 2048 is not divisible by tp=3”. Por isso a receita fixa D antes de E.'; }
      else if (S6.ord === 'CAB') { kind = 'ok'; msg = 'C edita outro arquivo (block_pool.py): não reescreve a região nem a âncora de A e B, e eles não tocam nas dele. Comuta, e o hash final é idêntico ao da ordem da receita.'; }
      else { kind = 'ok'; msg = 'Ordem da receita: cada âncora encontra exatamente o que espera. Mesmo hash nos dois ranks, e o grupo NCCL se forma.'; }
      badge(g, kind === 'ok' ? 'ok' : 'bad', 26, oy + 6, 9);
      wtext(g, msg, 44, oy + 10, w - 60, { size: wide ? 12.5 : 11.5, lh: wide ? 17 : 15, color: C.ink, maxLines: Math.max(3, Math.floor((h - oy - 4) / 15)) });
    });

    /* =====================================================================
       F7 — A escada de precedência
       ===================================================================== */
    const f7 = root.querySelector('#fig-escada');
    const CASES7 = {
      'glm-port': { r: 'glm', key: 'PORT', flag: (v) => '--port ' + v, rows: [{ src: 'export do chamador (inclusive vazio)', val: '9000', t: 'cli' }, { src: '.env', val: '8888', t: 'env' }, { src: 'default do start.sh', val: '8888', t: 'def' }], note: 'Chamador vence o .env. PORT é variável de identidade: o launcher avisa “NOTE: PORT=9000 from the environment overrides .env value 8888”.' },
      'glm-ablit': { r: 'glm', key: 'ABLIT', flag: (v) => 'ABLIT=' + v + (v === '1' ? ' (abliteração ligada)' : ' (o_proj stock)'), rows: [{ src: 'export do chamador', val: '1', t: 'cli' }, { src: 'regra: ABLIT=0 logo após o source', val: '0', t: 'rule' }, { src: '.env', val: '1', t: 'env', dead: true }], note: 'O launcher zera ABLIT depois do source: o .env nunca liga a abliteração. Só ABLIT=1 ./start.sh.' },
      'qwen-kv': { r: 'qwen', key: 'KV_CACHE_DTYPE', flag: (v) => '--kv-cache-dtype ' + v, rows: [{ src: 'OVERRIDE_KV_CACHE_DTYPE (wrapper)', val: 'fp8', t: 'wrap' }, { src: '.env', val: 'fp8', t: 'env' }, { src: 'export do chamador', val: 'auto', t: 'cli' }, { src: 'default do start.sh', val: 'fp8', t: 'def' }], note: 'O .env vence o ambiente: KV_CACHE_DTYPE=auto ./start.sh é ignorado se o .env define a chave. OVERRIDE_*, lido depois do .env, é o caminho dos wrappers.' },
      'qwen-ablit': { r: 'qwen', key: 'ABLIT', flag: (v) => 'ABLIT=' + v, rows: [{ src: 'export do chamador (_CLI_ABLIT)', val: '1', t: 'cli' }, { src: '.env', val: '0', t: 'env' }, { src: 'default', val: '0', t: 'def' }], note: 'Exceção à regra do Qwen: ABLIT e HF_TOKEN são capturados antes do source .env e reaplicados depois.' },
      'ds-host': { r: 'ds', key: 'VLLM_HOST', flag: (v) => '--host ' + v, rows: [{ src: '--host na linha de comando', val: '10.0.0.1', t: 'wrap' }, { src: '.env.dspark (cópia normalizada)', val: '0.0.0.0', t: 'env' }, { src: 'export do chamador', val: '127.0.0.1', t: 'cli' }, { src: 'default do launcher', val: '127.0.0.1', t: 'def' }], note: 'CLI > .env > ambiente > default. Bind em 0.0.0.0 sem chave de API gera aviso do launcher.' },
      'ds-keys': { r: 'ds', key: 'DSPARK_API_KEYS', conflict: true, rows: [{ src: '.env.dspark', val: 'k1 k2', t: 'env' }, { src: 'export do chamador', val: 'k9', t: 'cli' }], note: 'Aqui não há precedência: se o ambiente define a chave e ela difere do .env, o start sai 2 (“set it only in .env.dspark”). Segredos têm uma única fonte.' },
    };
    const S7 = { c: 'qwen-kv', env: true, cli: true, wrap: false, t0: 0 };
    const c7 = UI.controls(f7);
    let H7 = null;
    const re7 = () => { S7.t0 = H7 ? H7.t : 0; };
    fitSel(UI.select(c7, { label: 'Receita · variável', options: [{ v: 'glm-port', label: 'GLM · PORT' }, { v: 'glm-ablit', label: 'GLM · ABLIT' }, { v: 'qwen-kv', label: 'Qwen · KV_CACHE_DTYPE' }, { v: 'qwen-ablit', label: 'Qwen · ABLIT' }, { v: 'ds-host', label: 'DS · VLLM_HOST' }, { v: 'ds-keys', label: 'DS · DSPARK_API_KEYS' }], value: 'qwen-kv', onChange: (v) => { S7.c = v; re7(); } }));
    UI.toggle(c7, { label: 'o .env define a chave', value: true, onChange: (v) => { S7.env = v; re7(); } });
    UI.toggle(c7, { label: 'você exporta na linha de comando', value: true, onChange: (v) => { S7.cli = v; re7(); } });
    UI.toggle(c7, { label: 'wrapper / flag de CLI define', value: false, onChange: (v) => { S7.wrap = v; re7(); } });
    H7 = V.fig(f7, (g, w, h, t) => {
      const C = V.C(); const K = CASES7[S7.c]; const rc = C[K.r];
      const present = (row) => (row.t === 'def' || row.t === 'rule' ? true : !!S7[row.t]);
      let win = -1, result = '', rkind = 'ok';
      if (K.conflict) {
        const e = present(K.rows[0]), c = present(K.rows[1]);
        if (c) { result = 'exit 2 · conflito entre ambiente e .env'; rkind = 'bad'; }
        else if (e) { win = 0; result = '--api-key (2 chaves, lidas do .env)'; }
        else { result = 'sem chave: API sem autenticação (aviso se o bind for 0.0.0.0)'; rkind = 'warn'; }
      } else {
        K.rows.forEach((row, i) => { if (win < 0 && present(row) && !row.dead) win = i; });
        result = K.flag(K.rows[win].val);
      }
      const hasWrap = K.rows.some((r) => r.t === 'wrap');
      const kw7 = pill(g, 14, 10, K.key, { size: 11.5, fill: V.alpha(rc, 0.14), stroke: V.alpha(rc, 0.6), color: C.ink, weight: 600 });
      const RULE7 = { glm: 'GLM: chamador > .env > default', qwen: 'Qwen: OVERRIDE_* > .env > ambiente', ds: 'DeepSeek: CLI > .env > ambiente' };
      if (w > 420) V.text(g, RULE7[K.r], 14 + kw7 + 10, 24, { size: 11, color: rc, weight: 600 });
      V.text(g, 'mais forte ↑', w - 14, 24, { size: 10.5, font: 'mono', align: 'right', color: C.muted });
      const n = K.rows.length, top = 44, avail = h - top - 118;
      const rh = Math.min(62, avail / n);
      const el = t - S7.t0;
      K.rows.forEach((row, i) => {
        const y = top + i * rh;
        const pr = present(row), isWin = i === win;
        const col = isWin ? rc : row.dead && pr ? C.d : C.muted;
        V.box(g, 44, y + 3, w - 58, rh - 8, { r: 10, fill: V.alpha(col, isWin ? 0.16 : pr ? 0.06 : 0.02), stroke: V.alpha(col, isWin ? 0.9 : pr ? 0.4 : 0.2), lw: isWin ? 2 : 1 });
        V.text(g, (i + 1) + 'º', 16, y + rh / 2 + 1, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.muted });
        const valW = pr ? tw(g, row.val, 11, 'mono', 600) + 20 : 0;
        V.text(g, fit(g, row.src, w - 150 - valW, w < 500 ? 11.5 : 12.5), 58, y + rh / 2 + 1, { size: w < 500 ? 11.5 : 12.5, color: pr ? C.ink : C.muted, weight: isWin ? 600 : 400 });
        const tag = isWin ? 'VENCE' : !pr ? 'não definido' : row.dead ? 'ignorado pela regra' : K.conflict ? 'conflito' : 'ignorado';
        V.text(g, tag, w - 24, y + rh / 2 - (pr ? 7 : -4), { size: 9.5, font: 'mono', align: 'right', color: isWin ? rc : row.dead || K.conflict ? C.d : C.muted, weight: isWin ? 700 : 400 });
        if (pr) pill(g, w - 24, y + rh / 2 - 1, row.val, { size: 11, align: 'right', fill: V.alpha(col, 0.12), stroke: V.alpha(col, 0.5), color: C.ink, weight: 600, h: 18 });
      });
      // bolinha
      if (win >= 0 || K.conflict) {
        const target = win >= 0 ? top + win * rh + rh / 2 - 1 : top + (n - 1) * rh + rh / 2;
        const e = V.clamp(el / 0.9, 0, 1);
        const yb = V.lerp(top - 18, target, V.ease.out(e)) - Math.abs(Math.sin(e * Math.PI * 3)) * (1 - e) * 16;
        const bcol = rkind === 'bad' ? C.c : rc;
        V.glow(g, 35, yb, 11, bcol, 0.45);
        V.dot(g, 35, yb, 6, bcol);
      }
      // resultado
      const ry = h - 110;
      const rcol = rkind === 'bad' ? C.c : rkind === 'warn' ? C.d : rc;
      V.box(g, 14, ry, w - 28, 34, { r: 8, fill: V.alpha(rcol, 0.1), stroke: V.alpha(rcol, 0.55) });
      V.text(g, 'resultado →', 24, ry + 21, { size: 10.5, font: 'mono', color: C.muted });
      V.text(g, fit(g, result, w - 140, 12, 'mono', 600), 108, ry + 21, { size: 12, font: 'mono', weight: 600, color: C.ink });
      let note = K.note;
      if (S7.wrap && !hasWrap && !K.conflict) note = '(Esta receita não tem wrapper/CLI para esta chave.) ' + note;
      wtext(g, note, 16, ry + 54, w - 32, { size: w < 500 ? 11.5 : 12.5, lh: w < 500 ? 15 : 17, color: C.ink, maxLines: w < 500 ? 4 : 3 });
    });

    /* =====================================================================
       F8 — O que o bash enxerga numa linha do .env
       ===================================================================== */
    const f8 = root.querySelector('#fig-bytes');
    const CASES8 = {
      ok: { line: 'MAX_NUM_SEQS=6\n', name: 'MAX_NUM_SEQS', val: '6', bad: 'Valor 6: dentro da faixa 1–4096.', fix: 'Nada a corrigir.', ok: true },
      crlf: { line: 'KV_CACHE_DTYPE=fp8\r\n', name: 'KV_CACHE_DTYPE', val: "$'fp8\\r'", bad: 'O vLLM recebe “fp8\\r”: o argparse recusa um valor que, no log, parece idêntico a fp8.', fix: 'sed remove o \\r do fim de cada linha → fp8.' },
      bom: { line: '﻿WORKER_HOST=10.0.0.2\n', name: '(nenhuma)', val: 'WORKER_HOST fica vazio', bad: 'A linha deixa de ser atribuição: “\\uFEFFWORKER_HOST=10.0.0.2: command not found”.', fix: 'sed remove EF BB BF da primeira linha → WORKER_HOST=10.0.0.2.' },
      oct: { line: 'MAX_NUM_SEQS=08\n', name: 'MAX_NUM_SEQS', val: '08', bad: '$(( 08 )) → value too great for base (error token is "08"). E 010 valeria 8, em silêncio.', fix: 'Valida só dígitos e normaliza com 10#08 → 8.' },
      big: { line: 'MAX_NUM_SEQS=18446744073709551622\n', name: 'MAX_NUM_SEQS', val: '18446744073709551622', bad: '$(( … )) dá a volta nos 64 bits → 6. Passaria em qualquer checagem de faixa como se fosse 6.', fix: 'Mais de 10 dígitos: rejeitado antes da aritmética (exit 2).' },
      json: { line: 'SPECULATIVE_CONFIG={"method":"dspark","num_speculative_tokens":6}\n', name: 'SPECULATIVE_CONFIG', val: '{method:dspark,num_speculative_tokens:6}', bad: 'As aspas somem no source: o --speculative-config recebe JSON inválido.', fix: 'No .env, só MTP_NUM_TOKENS=6 e DRAFT_SAMPLE_METHOD=probabilistic. O entrypoint valida e monta {"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}.' },
    };
    const S8 = { c: 'crlf', def: false };
    const enc = new TextEncoder();
    const c8 = UI.controls(f8);
    fitSel(UI.select(c8, { label: 'Linha do .env', options: [{ v: 'ok', label: 'linha saudável' }, { v: 'crlf', label: 'salva no Windows (CRLF)' }, { v: 'bom', label: 'BOM na 1ª linha' }, { v: 'oct', label: 'zero à esquerda' }, { v: 'big', label: 'número gigante' }, { v: 'json', label: 'JSON no .env' }], value: 'crlf', onChange: (v) => { S8.c = v; } }));
    UI.toggle(c8, { label: 'defesas do DeepSeek (normalizar + validar + montar JSON no container)', value: false, onChange: (v) => { S8.def = v; } });
    V.fig(f8, (g, w, h, t) => {
      const C = V.C(); const K = CASES8[S8.c]; const wide = w >= 600;
      const bytes = Array.from(enc.encode(K.line));
      const eq = bytes.indexOf(0x3d);
      const cls = bytes.map((b, i) => {
        if (b === 0xef || b === 0xbb || b === 0xbf) return 'bad';
        if (b === 0x0d) return 'bad';
        if (b === 0x0a) return 'nl';
        if (S8.c === 'json' && b === 0x22) return 'warn';
        if (S8.c === 'oct' && i === eq + 1) return 'warn';
        if (S8.c === 'big' && i > eq + 10 && b >= 0x30 && b <= 0x39) return 'warn';
        return 'n';
      });
      // 1) o que o editor mostra
      const shownLine = K.line.replace(/﻿/g, '').replace(/[\r\n]/g, '');
      V.text(g, 'NO EDITOR (parece inofensiva)', 14, 16, { size: 10.5, font: 'mono', color: C.muted });
      const eh = wide ? 30 : 26;
      V.box(g, 14, 24, w - 28, eh, { r: 7, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.3) });
      V.text(g, fit(g, shownLine, w - 50, wide ? 13 : 11, 'mono'), 26, 24 + eh / 2 + 4, { size: wide ? 13 : 11, font: 'mono', color: C.ink });
      // 2) bytes
      const cw = wide ? 36 : 25, chh = wide ? 42 : 32, gap = 3;
      const cols = Math.max(6, Math.floor((w - 28 + gap) / (cw + gap)));
      const by0 = 24 + eh + 26;
      V.text(g, fit(g, 'O QUE O BASH LÊ · bytes (hex em cima, caractere embaixo)', w - 28, 10.5, 'mono'), 14, by0 - 8, { size: 10.5, font: 'mono', color: C.muted });
      const scan = Math.floor((t * 7) % (bytes.length + 6));
      bytes.forEach((b, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        const x = 14 + c * (cw + gap), y = by0 + r * (chh + gap);
        const k = cls[i];
        const removed = S8.def && k === 'bad';
        const col = k === 'bad' ? C.c : k === 'warn' ? C.d : k === 'nl' ? C.muted : C.b;
        const hot = i === scan;
        V.box(g, x, y, cw, chh, { r: 5, fill: V.alpha(col, (k === 'n' ? 0.06 : 0.16) + (hot ? 0.18 : 0)), stroke: V.alpha(col, hot ? 0.95 : k === 'n' ? 0.25 : 0.7), lw: hot ? 2 : 1 });
        const hexs = b.toString(16).toUpperCase().padStart(2, '0');
        const chs = b === 0x0d ? '\\r' : b === 0x0a ? '\\n' : b >= 0x80 ? '·' : String.fromCharCode(b);
        V.text(g, hexs, x + cw / 2, y + (wide ? 16 : 14), { size: wide ? 12 : 10.5, font: 'mono', weight: 600, align: 'center', color: k === 'n' ? C.ink : col, alpha: removed ? 0.35 : 1 });
        V.text(g, chs, x + cw / 2, y + chh - 8, { size: wide ? 11.5 : 10, font: 'mono', align: 'center', color: C.muted, alpha: removed ? 0.35 : 1 });
        if (removed) V.line(g, x + 4, y + 4, x + cw - 4, y + chh - 4, { color: C.c, lw: 1.6 });
      });
      const nRows = Math.ceil(bytes.length / cols);
      let y = by0 + nRows * (chh + gap) + 4;
      if (S8.c === 'bom') { V.text(g, '↑ EF BB BF = BOM (U+FEFF), invisível no editor', 14, y + 10, { size: 10.5, font: 'mono', color: C.c }); y += 16; }
      if (S8.c === 'crlf') { V.text(g, '↑ 0D = \\r (carriage return), invisível no editor', w - 14, y + 10, { size: 10.5, font: 'mono', align: 'right', color: C.c }); y += 16; }
      y += 10;
      // 3) cartões de resultado
      const cards = [
        ['bash atribui', K.name + ' = ' + K.val, K.ok ? 'ok' : 'warn', true],
        ['consequência', K.bad, K.ok ? 'ok' : 'bad', false],
        ['com defesas', S8.def ? K.fix : 'Ligue as defesas do DeepSeek para ver a correção.', S8.def ? 'ok' : 'info', false],
      ];
      const sz = wide ? 13 : 11.5, lh = wide ? 18 : 15;
      cards.forEach(([lab, txt, kind, mono]) => {
        const col = KCOL(C, kind);
        const L = lines(g, txt, w - 72, sz, mono ? 'mono' : 'sans', 400);
        const nl = Math.min(L.length, 4);
        const ch = 26 + nl * lh;
        if (y + ch > h - 4) return;
        V.box(g, 14, y, w - 28, ch, { r: 9, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.4) });
        badge(g, kind, 30, y + 14, 8);
        V.text(g, lab.toUpperCase(), 46, y + 18, { size: 10, font: 'mono', weight: 700, color: col });
        wtext(g, txt, 46, y + 18 + lh, w - 72, { size: sz, lh, color: C.ink, font: mono ? 'mono' : 'sans', maxLines: 4 });
        y += ch + 8;
      });
    });

    /* =====================================================================
       F9 — Dois ranks, um contrato de supervisão
       ===================================================================== */
    const f9 = root.querySelector('#fig-supervisao');
    const UP = { head: 'up', worker: 'up', nccl: 'ok', health: '200', canary: 'ok', exit: '—' };
    const OFF = { head: 'off', worker: 'off', nccl: 'none', health: 'off', canary: '-', exit: '—' };
    const boot = [
      ['info', './start: trava, valida o .env, preflight', {}],
      ['info', 'worker (rank 1, headless) sobe e espera o rendezvous', { worker: 'wait' }],
      ['info', 'head (rank 0) sobe: TCPStore, scheduler, API', { head: 'up', nccl: 'form' }],
      ['ok', 'grupo NCCL formado; carga dos pesos nos dois ranks', { worker: 'up', nccl: 'ok' }],
      ['ok', '/health 200 (start_period de 480 s cobre a carga a frio)', { health: '200' }],
    ];
    const SCN9 = {
      boot: { label: 'boot a frio', init: OFF, ev: { no: boot.concat([['ok', 'warmup + canário: “OK” a T=0 e rascunhos aceitos', { canary: 'ok' }], ['ok', 'READY · ./start sai 0', { exit: '0' }]]), unless: null } },
      crash: { label: 'worker cai', init: UP, ev: {
        no: [['info', 'serviço normal nos dois ranks', {}], ['bad', 'o processo do worker morre (ex.: falta de memória)', { worker: 'dead', nccl: 'broken' }], ['warn', 'o head fica preso na próxima coletiva, sem parceiro', { head: 'hung' }], ['bad', 'watchdog do NCCL (600 s) derruba o head', { head: 'dead', health: 'x', canary: '-' }], ['info', 'sem política de restart: os dois ficam parados', {}], ['ok', 'operador ou unit: ./stop + ./start recria o par junto', { head: 'up', worker: 'up', nccl: 'ok' }], ['ok', 'boot a frio completo → READY, exit 0', { health: '200', canary: 'ok', exit: '0' }]],
        unless: [['info', 'serviço normal nos dois ranks', {}], ['bad', 'o processo do worker morre (ex.: falta de memória)', { worker: 'dead', nccl: 'broken' }], ['warn', 'dockerd reinicia o worker; ele espera um novo rendezvous', { worker: 'wait' }], ['warn', 'o head segue preso na coletiva antiga até o watchdog (600 s)', { head: 'hung' }], ['bad', 'watchdog derruba o head; dockerd o reinicia também', { head: 'dead', health: 'x', canary: '-' }], ['ok', 'novo TCPStore: o par se refaz (só porque os DOIS reiniciaram)', { head: 'up', worker: 'up', nccl: 'ok' }], ['warn', 'recarga dos pesos e /health 200; sem warmup nem canário do launcher', { health: '200' }]] } },
      reboot: { label: 'reboot dos dois nós', init: UP, ev: {
        no: [['warn', 'reboot dos dois nós', OFF], ['info', 'nada volta sozinho (sem política de restart)', {}], ['info', 'o unit do systemd roda ./start: boot a frio', { worker: 'wait' }], ['ok', 'head sobe, grupo NCCL formado', { head: 'up', worker: 'up', nccl: 'ok' }], ['ok', '/health 200, warmup, canário', { health: '200', canary: 'ok' }], ['ok', 'READY · exit 0', { exit: '0' }]],
        unless: [['warn', 'reboot dos dois nós', OFF], ['info', 'dockerd restaura os dois containers (unless-stopped)', { worker: 'wait', head: 'up', nccl: 'form' }], ['ok', 'o par se refaz e recarrega os pesos', { worker: 'up', nccl: 'ok', health: '200' }], ['warn', 'o unit roda ./start: encontra os ranks → exit 3 (já rodando)', { exit: '3' }], ['ok', 'com SuccessExitStatus=3, o systemd trata como sucesso', {}], ['warn', 'ressalva: exit 3 não prova saúde (reboot só do head deixa worker velho)', {}]] } },
      degen: { label: 'motor degenerado', init: OFF, ev: { no: boot.slice(0, 4).concat([['warn', '/health 200, mas a saída é lixo e a aceitação ~0 (assinatura do #249)', { health: '200' }], ['bad', 'canário: “Reply with OK.” a T=0 falha; 0 de ≥ 64 rascunhos aceitos', { canary: 'x' }], ['bad', 'exit 3 DEGENERATE ENGINE (GLM): logs coletados, motor tirado da porta, sem READY', { exit: '3', health: 'off' }]]), unless: null } },
    };
    const S9 = { sc: 'reboot', pol: 'unless', t0: 0 };
    const c9 = UI.controls(f9);
    let H9 = null;
    fitSel(UI.select(c9, { label: 'Cenário', options: Object.keys(SCN9).map((k) => ({ v: k, label: SCN9[k].label })), value: 'reboot', onChange: (v) => { S9.sc = v; S9.t0 = H9 ? H9.t : 0; } }));
    UI.seg(c9, { label: 'Política de restart', options: [{ v: 'no', label: 'no (Qwen, GLM)' }, { v: 'unless', label: 'unless-stopped (DS)' }], value: 'unless', onChange: (v) => { S9.pol = v; S9.t0 = H9 ? H9.t : 0; } });
    H9 = V.fig(f9, (g, w, h, t) => {
      const C = V.C(); const wide = w >= 640;
      const Sc = SCN9[S9.sc];
      const evs = Sc.ev[S9.pol] || Sc.ev.no;
      const step = 1.25, total = evs.length * step + 2.2;
      if (t - S9.t0 > total) S9.t0 = t;
      const el = t - S9.t0;
      const idx = Math.min(evs.length - 1, Math.floor(el / step));
      const st = Object.assign({}, Sc.init);
      for (let i = 0; i <= idx; i++) Object.assign(st, evs[i][2]);
      // containers
      const bw = wide ? Math.min(260, w * 0.3) : (w - 60) / 2, bh = wide ? 104 : 92;
      const hx0 = wide ? w * 0.08 : 14, wx0 = wide ? w * 0.92 - bw : w - 14 - bw, by = 46;
      const stCol = (s) => (s === 'up' ? C.a : s === 'wait' || s === 'hung' ? C.d : s === 'dead' ? C.c : C.muted);
      const stTxt = (s) => ({ up: 'rodando', wait: 'esperando rendezvous', hung: 'preso na coletiva', dead: 'morto', off: 'desligado' }[s] || s);
      const stIc = (s) => (s === 'up' ? 'ok' : s === 'wait' ? 'wait' : s === 'hung' ? 'warn' : s === 'dead' ? 'bad' : 'skip');
      [[hx0, 'head · rank 0', st.head, 'API + TCPStore'], [wx0, 'worker · rank 1', st.worker, '--headless']].forEach(([x, lab, s, sub]) => {
        const col = stCol(s);
        V.box(g, x, by, bw, bh, { r: 14, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.7), lw: 1.6 });
        V.text(g, lab, x + 14, by + 22, { size: wide ? 13 : 11.5, weight: 700, color: C.ink });
        V.text(g, sub, x + 14, by + 38, { size: 10.5, font: 'mono', color: C.muted });
        badge(g, stIc(s), x + 24, by + bh - 26, 11);
        V.text(g, fit(g, stTxt(s), bw - 50, wide ? 12.5 : 11, 'sans', 600), x + 42, by + bh - 22, { size: wide ? 12.5 : 11, weight: 600, color: col });
        if (s === 'up') { const pr = 4 + 2 * Math.sin(t * 5); V.dot(g, x + bw - 18, by + 18, pr, C.a); }
      });
      // cabo NCCL
      const cy = by + bh / 2, cx0 = hx0 + bw + 6, cx1 = wx0 - 6;
      const nc = st.nccl;
      const ncol = nc === 'ok' ? C.a : nc === 'form' ? C.d : nc === 'broken' ? C.c : C.muted;
      if (nc === 'broken') { V.line(g, cx0, cy, (cx0 + cx1) / 2 - 10, cy, { color: ncol, lw: 3 }); V.line(g, (cx0 + cx1) / 2 + 10, cy, cx1, cy, { color: ncol, lw: 3 }); icon(g, 'bolt', (cx0 + cx1) / 2, cy, 9, C.c); }
      else V.line(g, cx0, cy, cx1, cy, { color: V.alpha(ncol, nc === 'none' ? 0.3 : 0.9), lw: 3, dash: nc === 'ok' ? [7, 6] : [3, 6], dashOffset: nc === 'ok' ? -t * 40 : -t * 10 });
      if (cx1 - cx0 > 50) V.text(g, 'NCCL', (cx0 + cx1) / 2, cy - 10, { size: 10.5, font: 'mono', align: 'center', color: ncol });
      // lâmpadas
      const lamps = [['/health', st.health === '200' ? 'ok' : st.health === 'x' ? 'bad' : 'skip'], ['canário', st.canary === 'ok' ? 'ok' : st.canary === 'x' ? 'bad' : 'skip']];
      let lx = hx0;
      lamps.forEach(([lab, k]) => { badge(g, k, lx + 9, 20, 8); V.text(g, lab, lx + 22, 24, { size: 11, font: 'mono', color: C.ink }); lx += tw(g, lab, 11, 'mono') + 40; });
      const ex = st.exit;
      pill(g, w - 14, 11, './start → exit ' + ex, { size: 11.5, align: 'right', fill: V.alpha(ex === '0' ? C.a : ex === '3' ? C.d : C.muted, 0.14), stroke: V.alpha(ex === '0' ? C.a : ex === '3' ? C.d : C.muted, 0.6), color: C.ink, weight: 600 });
      // barra de tempo
      const tbY = by + bh + 22, tbX0 = 24, tbX1 = w - 24;
      V.line(g, tbX0, tbY, tbX1, tbY, { color: V.alpha(C.muted, 0.3), lw: 2 });
      const prog = V.clamp(el / (evs.length * step), 0, 1);
      V.line(g, tbX0, tbY, tbX0 + prog * (tbX1 - tbX0), tbY, { color: V.alpha(C.a, 0.8), lw: 2.5 });
      evs.forEach((ev, i) => { const x = tbX0 + (i + 0.5) / evs.length * (tbX1 - tbX0); const col = KCOL(C, ev[0]); V.dot(g, x, tbY, i <= idx ? 5 : 3.5, i <= idx ? col : V.alpha(C.muted, 0.5)); if (i === idx) V.glow(g, x, tbY, 14, col, 0.6); });
      // eventos
      const ly0 = by + bh + 50;
      V.text(g, 'LINHA DO TEMPO · política ' + (Sc.ev.unless ? S9.pol : 'indiferente neste cenário'), 14, ly0 - 6, { size: 10.5, font: 'mono', color: C.muted });
      // cada evento quebra em até 2 linhas; mostramos os mais recentes que cabem
      const tx = 40, tmw = w - tx - 14;
      const szOf = (cur) => (cur ? (wide ? 13 : 12) : (wide ? 12 : 11.5));
      const lhOf = (cur) => (cur ? (wide ? 17 : 15) : (wide ? 16 : 14.5));
      const hOf = (i) => { const cur = i === idx; const L = lines(g, evs[i][1], tmw, szOf(cur), 'sans', cur ? 600 : 400); return Math.min(2, L.length) * lhOf(cur) + (wide ? 7 : 6); };
      const avail = h - ly0 - 2;
      let startI = idx, acc = hOf(idx);
      while (startI > 0 && acc + hOf(startI - 1) <= avail) { startI--; acc += hOf(startI); }
      let y = ly0;
      for (let i = startI; i <= idx; i++) {
        const [k, txt] = evs[i];
        const cur = i === idx;
        const a = cur ? V.clamp((el - i * step) / 0.3, 0, 1) : 0.7;
        g.globalAlpha = a;
        badge(g, k, 24, y + 9, 7.5);
        const used = wtext(g, txt, tx, y + 13, tmw, { size: szOf(cur), lh: lhOf(cur), weight: cur ? 600 : 400, color: C.ink, maxLines: 2 });
        g.globalAlpha = 1;
        y += used + (wide ? 7 : 6);
      }
    });

    /* =====================================================================
       F10 — Ciclo de vida de um hack e a escada do backport
       ===================================================================== */
    const f10 = root.querySelector('#fig-ciclo');
    const PH10 = [
      { n: 'Bug sob carga', d: 'DeepSeek #22: o nome nvfp4_ds_mla caía no kernel bf16 lento. A 600K+ de contexto, ~1 tok/s de decode contra ~17 com fp8_ds_mla, com os mesmos 584 B por linha.', r: 'Regra: medir e reproduzir antes de mexer; abrir issue com o número.' },
      { n: 'Patch pinado', d: 'hotfix-nvfp4-ds-mla-issue22.sh troca “== \"fp8_ds_mla\"” por “in (\"fp8_ds_mla\", \"nvfp4_ds_mla\")”: fonte exata, idempotente e encadeado com || exit 1 (skip com DSPARK_SKIP_ISSUE22_HOTFIX=1).', r: 'Regra: pin da fonte, pós-imagem verificada, publicação atômica, gate explícito.' },
      { n: 'Provar em CPU', d: 'Fixture do arquivo stock da imagem 752a3a504; testes de identidade, idempotência, recusa de drift e rollback byte-exato. O CI roda tudo sem GPU.', r: 'Regra: nenhum patch entra sem teste contra a fixture da imagem pinada.' },
      { n: 'Registrar', d: 'CHANGELOG com o número medido (~16× no decode a 600K+, 1,0 → 17,3 tok/s) e a instrução operacional: recriar os dois containers depois do pull.', r: 'Regra: toda entrada diz o que mudou, quanto mediu e o que o operador precisa fazer.' },
      { n: 'Upstream corrige', d: 'O patch precisa perceber: virar no-op quando a correção já está na fonte (Qwen block_drop), carregar a própria nota de morte (Qwen v030 → apagar no vLLM 0.31+) ou recusar meia correção (GLM kpool seed stride).', r: 'Regra: o patch deve saber reconhecer a própria obsolescência.' },
      { n: 'Aposentar', d: 'Remover arquivo, teste e mount. Se o upstream reverter, seguir o revert (DeepSeek #50004, revertido em #51318). E impedir a volta no CI (hook de CPU do #31, que custava ~4,7× de decode).', r: 'Regra: remover por completo (arquivo, teste, mount, docs) e guardar o porquê no CHANGELOG.' },
    ];
    const PR10 = [
      { n: '#50312', lvl: 0, d: 'Aloca o buffer MTP só se um especulador precisa: ~256 MiB por rank aqui.', st: 'aplicado (hotfix .sh no boot)' },
      { n: '#49486+#52492', lvl: 0, d: 'Atalho do indexer quando todos os candidatos seriam escolhidos (≤ 2048 tokens); nunca durante a captura de graph.', st: 'aplicado' },
      { n: '#48407', lvl: 0, d: 'Pular o scoring do indexer em prefill denso curto.', st: 'aplicado, dormente de propósito (o fork não tem rota dense-MHA)' },
      { n: '#48957', lvl: 0, d: 'Pula o compressor C128 quando nenhuma requisição cruza a fronteira de 128 tokens (~2× no kernel); fora de graphs FULL.', st: 'aplicado' },
      { n: '#50298', lvl: 0, d: 'Reuso de workspace para os índices top-k + SWA do prefill FlashMLA (1,88× no kernel, medido upstream).', st: 'aplicado' },
      { n: '#44993', lvl: 0, d: 'Avanço correto da gramática com response_format + thinking + spec decode.', st: 'aplicado' },
      { n: '#46789', lvl: 1, d: 'Sequence parallelism.', st: 'não portado: mudança do tamanho de uma funcionalidade, não de um hotfix' },
      { n: '#48047', lvl: 2, d: 'Remove o padding de q-heads no sparse MLA.', st: 'não portado: exige FlashInfer ≥ 0.6.14 (dependência de biblioteca; o documento não reavalia contra a 0.6.15 da imagem)' },
      { n: '#49236', lvl: 3, d: 'DeepseekV4EagerScratchPool: −3,9 % de TTFT E2E.', st: 'não portado: precisa recompilar uma op C++ (nova variante _out); depois revertido upstream (#52836)' },
      { n: '#48993', lvl: -1, d: 'KV MXFP4 compacto do indexer.', st: 'não avaliado' },
    ];
    const LV10 = ['hotfix Python no boot', 'overlay grande / nova base', 'nova dependência na imagem', 'recompilar op C++/CUDA'];
    const S10 = { view: 'cycle', i: 0, t0: 0 };
    const c10 = UI.controls(f10);
    let H10 = null;
    let sl10 = null;
    UI.seg(c10, { label: 'Vista', options: [{ v: 'cycle', label: 'ciclo de vida' }, { v: 'ladder', label: 'escada do backport' }], value: 'cycle', onChange: (v) => {
      S10.view = v; S10.i = 0; S10.t0 = H10 ? H10.t : 0;
      if (sl10) { sl10.input.max = v === 'cycle' ? PH10.length : PR10.length; sl10.input.value = 1; const lb = sl10.el.querySelector('label'); if (lb) lb.textContent = v === 'cycle' ? 'Fase' : 'PR'; sl10.el.querySelector('.val').textContent = '1'; }
    } });
    sl10 = UI.slider(c10, { label: 'Fase', min: 1, max: PH10.length, step: 1, value: 1, onInput: (v) => { const mx = S10.view === 'cycle' ? PH10.length : PR10.length; S10.i = Math.min(mx, Math.max(1, v)) - 1; S10.t0 = H10 ? H10.t : 0; } });
    respH(f10, (fw) => (fw < 860 ? 500 : 430));
    H10 = V.fig(f10, (g, w, h, t) => {
      const C = V.C(); const wide = w >= 860;
      const el = t - S10.t0;
      if (S10.view === 'cycle') {
        const n = PH10.length;
        const R = wide ? Math.min(140, h * 0.34) : 84, cxc = wide ? Math.max(w * 0.27, R + 150) : w / 2, cyc = wide ? h / 2 : 124;
        g.beginPath(); g.arc(cxc, cyc, R, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 2; g.setLineDash([4, 6]); g.lineDashOffset = -t * 12; g.stroke(); g.setLineDash([]);
        const ang = (k) => -Math.PI / 2 + (k / n) * Math.PI * 2;
        for (let k = 0; k < n; k++) {
          const a0 = ang(k) + 0.28, a1 = ang(k + 1) - 0.28;
          const ax = cxc + Math.cos(a1) * R, ay = cyc + Math.sin(a1) * R;
          g.beginPath(); g.arc(cxc, cyc, R, a0, a1); g.strokeStyle = V.alpha(k < S10.i ? C.a : C.muted, k < S10.i ? 0.8 : 0.35); g.lineWidth = 2.5; g.stroke();
          const ta = a1 + Math.PI / 2; V.arrow(g, ax - Math.cos(ta) * 6, ay - Math.sin(ta) * 6, ax, ay, { color: V.alpha(k < S10.i ? C.a : C.muted, 0.7), lw: 2, head: 6 });
        }
        PH10.forEach((P, k) => {
          const a = ang(k), x = cxc + Math.cos(a) * R, y = cyc + Math.sin(a) * R;
          const on = k === S10.i, past = k < S10.i;
          if (on) V.glow(g, x, y, 34, C.a, 0.5 + 0.2 * Math.sin(t * 3));
          V.dot(g, x, y, on ? 17 : 14, on ? C.a : past ? V.alpha(C.a, 0.5) : C.panel);
          g.beginPath(); g.arc(x, y, on ? 17 : 14, 0, Math.PI * 2); g.strokeStyle = on || past ? C.a : V.alpha(C.muted, 0.6); g.lineWidth = 1.8; g.stroke();
          V.text(g, String(k + 1), x, y + 0.5, { size: 12, font: 'mono', weight: 700, align: 'center', base: 'middle', color: on || past ? onDark(C) : C.muted });
          const lx = cxc + Math.cos(a) * (R + (wide ? 30 : 24)), ly = cyc + Math.sin(a) * (R + (wide ? 30 : 24)) + 4;
          const al = Math.cos(a) > 0.3 ? 'left' : Math.cos(a) < -0.3 ? 'right' : 'center';
          const tx = al === 'left' ? lx - 6 : al === 'right' ? lx + 6 : lx, ty = ly + (Math.abs(Math.cos(a)) < 0.3 ? (Math.sin(a) < 0 ? -4 : 8) : 0);
          if (!wide && al !== 'center') { const maxL = al === 'left' ? w - tx - 6 : tx - 6; wtext(g, P.n, tx, ty - (tw(g, P.n, 10.5, 'sans', on ? 700 : 500) > maxL ? 6 : 0), maxL, { size: 10.5, lh: 12, weight: on ? 700 : 500, align: al, color: on ? C.ink : C.muted, maxLines: 2 }); }
          else V.text(g, P.n, tx, ty, { size: wide ? 12 : 10.5, weight: on ? 700 : 500, align: al, color: on ? C.ink : C.muted });
        });
        // hack viajando
        const e = V.ease.inOut(V.clamp(el / 0.9, 0, 1));
        const ah = ang(S10.i) - (1 - e) * (Math.PI * 2 / n);
        const hxp = cxc + Math.cos(ah) * (R - 30), hyp = cyc + Math.sin(ah) * (R - 30);
        V.box(g, hxp - 16, hyp - 10, 32, 20, { r: 6, fill: C.d });
        V.text(g, 'hack', hxp, hyp + 0.5, { size: 9.5, font: 'mono', weight: 700, align: 'center', base: 'middle', color: onDark(C) });
        const P = PH10[S10.i];
        const px = wide ? cxc + R + 150 : 14, py = wide ? 30 : 256, pw = wide ? w - px - 18 : w - 28, ph = wide ? h - 60 : h - py - 10;
        g.globalAlpha = V.clamp(el / 0.4, 0, 1);
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.4) });
        V.text(g, 'FASE ' + (S10.i + 1) + '/6 · ' + P.n.toUpperCase(), px + 16, py + 24, { size: 11, font: 'mono', weight: 700, color: C.a });
        const lhP = wide ? 21 : 16.5;
        const used = wtext(g, P.d, px + 16, py + 50, pw - 32, { size: wide ? 14 : 12, lh: lhP, color: C.ink, maxLines: Math.floor((ph - 90) / lhP) });
        wtext(g, P.r, px + 16, py + 50 + used + 14, pw - 32, { size: wide ? 13 : 11.5, lh: wide ? 19 : 15, color: C.a, weight: 600, maxLines: 3 });
        g.globalAlpha = 1;
      } else {
        // escada
        const sx = 16, sw = wide ? w * 0.6 : w - 32, base = wide ? h - 40 : 222, stepW = sw / 4, stepH = wide ? 62 : 42;
        LV10.forEach((lab, k) => {
          const x = sx + k * stepW, y = base - (k + 1) * stepH;
          const col = [C.a, C.b, C.d, C.c][k];
          V.box(g, x, y, stepW - 4, (k + 1) * stepH, { r: 8, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.5) });
          wtext(g, lab, x + 7, y + 15, stepW - 16, { size: wide ? 11.5 : 9.5, lh: wide ? 14 : 11, color: col, weight: 600, maxLines: 3 });
        });
        V.arrow(g, sx, base + 18, sx + sw - 8, base + 18, { color: V.alpha(C.muted, 0.6), lw: 1.5, head: 7 });
        V.text(g, 'mais fundo a mudança precisa morar →', sx + 2, base + 32, { size: 10, font: 'mono', color: C.muted });
        const lvlY = [0, 0, 0, 0];
        const ph1 = wide ? 18 : 15, gapP = wide ? 4 : 3;
        const naX = wide ? sx + sw - 70 : sx, naY = wide ? 24 : 26;
        PR10.forEach((P, k) => {
          const on = k === S10.i;
          const maxPW = stepW - 10;
          const sz = wide ? 10.5 : 9;
          // pílula que não cabe no degrau quebra em duas linhas no '+'
          const parts = tw(g, P.n, sz, 'mono', on ? 700 : 500) + 14 > maxPW && P.n.indexOf('+') > 0 ? [P.n.slice(0, P.n.indexOf('+')), P.n.slice(P.n.indexOf('+'))] : [P.n];
          const hh = ph1 + (parts.length - 1) * (sz + 3);
          let x, y;
          if (P.lvl < 0) { x = naX; y = naY; }
          else {
            x = sx + P.lvl * stepW + 5;
            y = base - (P.lvl + 1) * stepH - 6 - lvlY[P.lvl] - hh;
            lvlY[P.lvl] += hh + gapP;
          }
          const col = P.lvl < 0 ? C.muted : P.lvl === 0 ? C.a : C.c;
          const bob = on ? Math.sin(t * 4) * 2 : 0;
          let wdt;
          if (parts.length === 1) wdt = pill(g, x, y + bob, P.n, { size: sz, fill: V.alpha(col, on ? 0.35 : 0.14), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2 : 1, color: C.ink, weight: on ? 700 : 500, h: ph1 });
          else {
            wdt = Math.min(maxPW, Math.max(...parts.map((q) => tw(g, q, sz, 'mono', on ? 700 : 500))) + 14);
            V.box(g, x, y + bob, wdt, hh, { r: 7, fill: V.alpha(col, on ? 0.35 : 0.14), stroke: on ? col : V.alpha(col, 0.5), lw: on ? 2 : 1 });
            parts.forEach((q, qi) => V.text(g, q, x + wdt / 2, y + bob + ph1 / 2 + 0.5 + qi * (sz + 3), { size: sz, font: 'mono', weight: on ? 700 : 500, align: 'center', base: 'middle', color: C.ink }));
          }
          if (on) V.glow(g, x + wdt / 2, y + hh / 2 + bob, 26, col, 0.35);
        });
        V.text(g, 'não avaliado', naX, naY - 8, { size: 9.5, font: 'mono', color: C.muted });
        const P = PR10[S10.i];
        const px = wide ? w * 0.64 : 14, py = wide ? 24 : base + 44, pw = wide ? w - px - 16 : w - 28, ph = wide ? h - 48 : h - py - 8;
        const col = P.lvl < 0 ? C.muted : P.lvl === 0 ? C.a : C.c;
        g.globalAlpha = V.clamp(el / 0.35, 0, 1);
        V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(col, 0.07), stroke: V.alpha(col, 0.45) });
        V.text(g, 'vLLM ' + P.n, px + 14, py + 24, { size: 12.5, font: 'mono', weight: 700, color: col });
        let yy = py + 46;
        yy += wtext(g, P.d, px + 14, yy, pw - 28, { size: wide ? 13 : 12, lh: wide ? 18 : 16, color: C.ink, maxLines: 4 }) + 8;
        V.text(g, 'estado', px + 14, yy, { size: 10, font: 'mono', color: C.muted });
        const us = wtext(g, P.st, px + 14, yy + 16, pw - 28, { size: wide ? 12.5 : 11.5, lh: wide ? 17 : 15, color: col, weight: 600, maxLines: 5 });
        const LES = ['Cabe num patch de texto Python sobre a imagem pinada: é o que a lane A aplica no boot.', 'Grande demais para âncoras: vira overlay de arquivos inteiros ou exige outra base.', 'Depende de uma biblioteca mais nova: só entra com uma imagem que a traga.', 'Precisa de binário novo: só uma imagem reconstruída (build de extensão).'];
        if (P.lvl >= 0 && yy + 16 + us + 30 < py + ph) wtext(g, LES[P.lvl], px + 14, yy + 16 + us + 16, pw - 28, { size: wide ? 12 : 11, lh: wide ? 16 : 14, color: C.muted, maxLines: 3 });
        g.globalAlpha = 1;
      }
    });

    /* =====================================================================
       F11 — TensorFold: uma lista, nove patches (esteira de diffs + hash da imagem)
       ===================================================================== */
    const f11 = root.querySelector('#fig-tf-esteira');
    let tfc11 = null;
    const TFCOL = (C) => { if (!tfc11 || tfc11.C !== C) tfc11 = { C, v: getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || C.d }; return tfc11.v; };
    // a lista `mine` do engine.py (família GLM): como vem no v0.6.0 e depois de cada um dos 9 patches que a reescrevem
    const UP11 = ['drafter', 'capacity', 'long_ctx', 'serial', 'LATENT', 'prefill_rows'];
    const ST11 = [
      UP11.concat(['mtp', 'ring']),
      UP11.concat(['copy×2', 'mtp', 'ring']),
      UP11.concat(['grid', 'copy×2', 'mtp', 'ring']),
      UP11.concat(['grid', 'copy×2', 'split×5', 'mtp', 'ring']),
      UP11.concat(['grid', 'copy×2', 'split×5', 'dump', 'quant', 'mtp', 'ring']),
      UP11.concat(['grid', 'kvb', 'copy×2', 'split×5', 'dump', 'quant', 'mtp', 'ring']),
      UP11.concat(['grid', 'kvb', 'copy×2', 'split×5', 'dump', 'quant', 'shared', 'mtp', 'ring']),
      UP11.concat(['grid', 'kvb', 'copy×2', 'split×5', 'dump', 'quant', 'shared', 'ring', 'mtp', 'parallel', 'verify']),
      UP11.concat(['grid', 'kvb', 'copy×4', 'split×5', 'dump', 'quant', 'shared', 'ring', 'mtp', 'parallel', 'verify', 'MAX_ROWS', 'wide×3']),
      UP11.concat(['grid', 'kvb', 'copy×4', 'split×5', 'dump', 'quant', 'shared', 'ring', 'mtp', 'parallel', 'verify', 'multi×6', 'union', 'MAX_ROWS', 'wide×3']),
    ];
    const P11 = [
      { n: '0007', s: 'copy-drafts', w: '+ copy×2 (match, máx.)' },
      { n: '0008', s: 'prompt-grid', w: '+ grid' },
      { n: '0010', s: 'hc-split', w: '+ split×5' },
      { n: '0011', s: 'draft-sim', w: '+ dump, quant' },
      { n: '0013', s: 'decode-rounds', w: '+ kvb' },
      { n: '0015', s: 'shared-prefix', w: '+ shared' },
      { n: '0030', s: 'multi-stream-engine', w: 'ring ↔ mtp; + parallel, verify' },
      { n: '0032', s: 'code-copy-drafts', w: 'copy×2 → copy×4; + MAX_ROWS, wide×3' },
      { n: '0035', s: 'multi-rounds', w: '+ multi×6, union' },
    ];
    const OWN11 = {};
    ST11.forEach((st, s) => st.forEach((k) => { if (!(k in OWN11)) OWN11[k] = s - 1; }));
    // o 0040 não reescreve a linha, mas acrescenta um campo a MultiSettings.code() (perfil ligado ou não): 36 na imagem final
    const END11 = ST11[9].map((k) => (k === 'multi×6' ? 'multi×7' : k));
    OWN11['multi×7'] = 8;
    const ints11 = (st) => st.reduce((a, k) => { const m = k.match(/×(\d+)$/); return a + (m ? +m[1] : 1); }, 0);
    const SC11 = {
      rel: { ord: [0, 1, 2, 3, 4, 5, 6, 7, 8], ed: -1 },
      edit: { ord: [0, 1, 2, 3, 4, 5, 6, 7, 8], ed: 7 },
      swap: { ord: [0, 1, 3, 2, 4, 5, 6, 7, 8], ed: -1 },
      drop: { ord: [0, 1, 2, 3, 5, 6, 7, 8], ed: -1 },
    };
    const REL11 = 'ae8d1c789b47';
    const SIM11 = {};
    function sim11(key) {
      if (SIM11[key]) return SIM11[key];
      const sc = SC11[key];
      let st = ST11[0]; const steps = []; let fail = -1;
      for (let k = 0; k < sc.ord.length; k++) {
        const p = sc.ord[k], exp = ST11[p];
        const ok = exp.length === st.length && exp.every((x, i) => x === st[i]);
        if (ok) { steps.push({ p, ok: true, before: st, after: ST11[p + 1] }); st = ST11[p + 1]; }
        else { steps.push({ p, ok: false, before: st, after: st, miss: exp.filter((x) => st.indexOf(x) < 0) }); fail = k; break; }
      }
      let hh = key === 'rel' ? REL11 : hx('tf11|' + key + '|' + sc.ord.join(','), 12);
      if (key !== 'rel' && hh === REL11) hh = hx('tf11b|' + key, 12);
      return (SIM11[key] = { sc, steps, fail, final: fail < 0 ? END11 : st, h: hh });
    }
    const S11 = { sc: 'rel', pull: true, ready: false, t0: -7 };   // começa no meio da esteira; a animação se repete
    function plan11() {
      const sim = sim11(S11.sc), rel = S11.sc === 'rel';
      const src = rel && S11.ready ? 'local' : rel && S11.pull ? 'digest' : 'build';
      return { sim, rel, src, fail: src === 'build' && sim.fail >= 0, tried: src === 'build' && S11.pull && !rel };
    }
    const c11 = UI.controls(f11);
    let H11 = null;
    function upd11() {
      const PL = plan11();
      r11a.set(PL.sim.h);
      if (PL.fail) r11b.set('build falha (exit 1)', 'bad');
      else r11b.set(PL.src === 'digest' ? 'pull por digest' : PL.src === 'local' ? 'já está aqui' : 'build local', 'ok');
    }
    const restart11 = () => { S11.t0 = H11 ? H11.t : 0; upd11(); };
    fitSel(UI.select(c11, { label: 'Conjunto de patches', options: [{ v: 'rel', label: 'release v1.3.2 (53 patches)' }, { v: 'edit', label: 'edito o 0032' }, { v: 'swap', label: '0011 antes do 0010' }, { v: 'drop', label: 'apago o 0013' }], value: 'rel', onChange: (v) => { S11.sc = v; restart11(); } }));
    UI.toggle(c11, { label: 'PULL=0 (construir sempre)', value: false, onChange: (v) => { S11.pull = !v; restart11(); } });
    UI.toggle(c11, { label: 'Spark já preparado (release construída aqui)', value: false, onChange: (v) => { S11.ready = v; restart11(); } });
    UI.button(c11, '▶ prepare.sh de novo', () => restart11(), { primary: true });
    const r11a = UI.readout(c11, 'hash', '—');
    const r11b = UI.readout(c11, 'imagem', '—');
    upd11();
    respH(f11, (fw) => (fw < 760 ? 730 : 440));
    H11 = V.fig(f11, (g, w, h, t) => {
      const C = V.C(), TF = TFCOL(C), wide = w >= 760;
      const PL = plan11(), sim = PL.sim;
      const per11 = 1.5 + 0.9 + sim.steps.length * 0.95 + 8.5;
      const el = (t - S11.t0) % per11;
      const PCOL = [C.a, C.b, C.c, C.d, C.e, C.f, V.mix(C.a, C.b, 0.5), V.mix(C.c, C.d, 0.5), V.mix(C.f, C.d, 0.45)];
      const colOf = (key) => (OWN11[key] >= 0 ? PCOL[OWN11[key]] : C.muted);
      const lx = 14, lw = wide ? Math.round(w * 0.6) - 26 : w - 28;
      const rx = wide ? Math.round(w * 0.6) + 4 : 14, rw = wide ? w - rx - 14 : w - 28;
      // ---------- linha do tempo da esteira
      const tB = 1.5, PIP = 0.9, PD = 0.95;
      const n = sim.steps.length;
      const tt = el - tB - PIP;
      let k = tt < 0 ? -1 : Math.floor(tt / PD);
      const kMax = sim.fail >= 0 ? sim.fail : n;
      if (k > kMax) k = kMax;
      const frac = k < 0 || k >= n ? 1 : V.clamp((tt - k * PD) / PD, 0, 1);
      const stp = k >= 0 && k < n ? sim.steps[k] : null;
      // ---------- fichas
      const csz = wide ? 11 : 10.5, chH = wide ? 22 : 21, gap = 5;
      const cwid = (key) => tw(g, key, csz, 'mono', 600) + 14;
      function flow(list, x0, y0, maxW) {
        const pos = {}; let x = x0, y = y0;
        list.forEach((key) => { const ww = cwid(key); if (x > x0 && x + ww > x0 + maxW) { x = x0; y += chH + gap; } pos[key] = { x, y, w: ww }; x += ww + gap; });
        return { pos, h: y - y0 + chH };
      }
      let y = 16;
      const cur = k < 0 ? ST11[0] : stp ? stp.before : sim.final;
      V.text(g, fit(g, 'engine.py · a lista mine, conferida na partida', lw - 90, 10.5, 'mono'), lx, y, { size: 10.5, font: 'mono', color: C.muted });
      const shownInts = stp && stp.ok && frac > 0.55 ? ints11(stp.after) : ints11(cur);
      V.text(g, shownInts + ' inteiros', lx + lw, y, { size: 10.5, font: 'mono', weight: 700, align: 'right', color: TF });
      y += 10;
      const chipTop = y;
      const resH = flow(ST11[9], lx, 0, lw).h;
      const drawChip = (key, x, yy, ww, a, sc, col) => {
        const cc = col || colOf(key), own = OWN11[key] >= 0;
        g.save(); g.globalAlpha = a;
        if (sc !== 1) { g.translate(x + ww / 2, yy + chH / 2); g.scale(sc, sc); g.translate(-(x + ww / 2), -(yy + chH / 2)); }
        V.box(g, x, yy, ww, chH, { r: 6, fill: V.alpha(cc, own ? 0.24 : 0.12), stroke: V.alpha(cc, own ? 0.9 : 0.5), lw: 1.2 });
        V.text(g, key, x + ww / 2, yy + chH / 2 + 0.5, { size: csz, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
        g.restore();
      };
      const Lb = flow(stp ? stp.before : cur, lx, chipTop, lw);
      if (stp && stp.ok && frac > 0.55) {
        const u = V.ease.inOut(V.clamp((frac - 0.55) / 0.4, 0, 1));
        const La = flow(stp.after, lx, chipTop, lw);
        stp.before.forEach((key) => { if (stp.after.indexOf(key) < 0) { const q = Lb.pos[key]; drawChip(key, q.x, q.y, q.w, 1 - u, 1); } });
        stp.after.forEach((key) => {
          const q = La.pos[key], o = Lb.pos[key];
          if (o) drawChip(key, V.lerp(o.x, q.x, u), V.lerp(o.y, q.y, u), q.w, 1, 1);
          else drawChip(key, q.x, q.y, q.w, u, 0.6 + 0.4 * V.ease.back(u));
        });
      } else {
        (stp ? stp.before : cur).forEach((key) => { const q = Lb.pos[key]; drawChip(key, q.x, q.y, q.w, 1, 1); });
      }
      // no fim: o campo que o 0040 acrescenta por dentro de multi
      if (!stp && k >= n && Lb.pos['multi×7']) {
        const q = Lb.pos['multi×7'], pu = 0.5 + 0.5 * Math.sin(t * 5);
        g.save(); g.globalAlpha = 0.45 + 0.55 * pu;
        V.box(g, q.x - 3, q.y - 3, q.w + 6, chH + 6, { r: 8, stroke: TF, lw: 1.6 });
        g.restore();
      }
      // fichas que o trecho procura e não existem (falha)
      if (stp && !stp.ok && frac > 0.5) {
        const Lg = flow(stp.before.concat(stp.miss), lx, chipTop, lw);
        const pulse = 0.55 + 0.45 * Math.sin(t * 6);
        stp.miss.forEach((key) => { const q = Lg.pos[key]; g.save(); g.globalAlpha = pulse; g.setLineDash([4, 3]); V.box(g, q.x, q.y, q.w, chH, { r: 6, fill: V.alpha(C.c, 0.08), stroke: C.c, lw: 1.4 }); g.setLineDash([]); V.text(g, key + '?', q.x + q.w / 2, q.y + chH / 2 + 0.5, { size: csz, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.c }); g.restore(); });
      }
      // contorno: o que o trecho procura
      if (stp && frac > 0.22 && (!stp.ok || frac < 0.62)) {
        const bad = !stp.ok;
        const shake = bad && frac > 0.5 ? Math.sin(t * 34) * 1.6 : 0;
        g.setLineDash([5, 4]); g.lineDashOffset = -t * 18;
        V.box(g, lx - 5 + shake, chipTop - 4, lw + 10, Lb.h + 8, { r: 9, stroke: bad ? C.c : TF, lw: 1.8 });
        g.setLineDash([]); g.lineDashOffset = 0;
      }
      y = chipTop + resH + 18;
      // texto do trecho (hunk)
      let hunk, hcol = C.ink;
      if (k < 0) hunk = el < tB ? 'prepare.sh: hash, imagem local, registry…' : 'pip install TensorFold@v0.6.0 + extras: a lista nasce com 8 inteiros';
      else if (stp && stp.ok) hunk = '@@ ' + P11[stp.p].n + ' ' + P11[stp.p].s + ' @@  ' + P11[stp.p].w + (frac < 0.55 ? '  · procura a lista que o patch anterior deixou…' : '  ✓');
      else if (stp) { hunk = '@@ ' + P11[stp.p].n + ' ' + P11[stp.p].s + ' @@  procura uma lista que já tem ' + stp.miss.join(', ') + (frac > 0.5 ? ': não encontra → Hunk FAILED, exit 1' : '…'); hcol = frac > 0.5 ? C.c : C.ink; }
      else hunk = '+1 em multi pelo 0040 · ' + ints11(sim.final) + ' inteiros · import de prova ✓ · LABEL tf.patches=' + sim.h;
      const hLines = wide ? 2 : 3;   // no celular, o trecho com falha precisa de uma 3ª linha
      wtext(g, hunk, lx, y, lw, { size: wide ? 11.5 : 11, lh: wide ? 15 : 14.5, font: 'mono', color: hcol, maxLines: hLines });
      y += (wide ? 15 : 14.5) * hLines + 14;
      // ---------- esteira de cartões
      const belt = PL.src === 'build' ? 'build local, neste Spark' : PL.src === 'digest' ? 'build de quem publicou a imagem (aqui nada é aplicado)' : 'build feito antes, neste Spark';
      V.text(g, fit(g, 'esteira · ' + belt, lw, 10.5, 'mono'), lx, y, { size: 10.5, font: 'mono', color: C.muted });
      y += 10;
      const ord = sim.sc.ord, nC = ord.length, cg = wide ? 6 : 4;
      const cardW = (lw - (nC - 1) * cg) / nC, cardH = wide ? 46 : 38;
      const beltY = y + cardH + 6;
      V.box(g, lx, beltY, lw, 7, { r: 3.5, fill: V.alpha(C.muted, 0.14) });
      const moving = k < n && (sim.fail < 0 || k < sim.fail || frac < 1);
      for (let q = 0; q < 34; q++) { const xx = lx + ((q * 23 + (moving ? t * 26 : 0)) % lw); V.line(g, xx, beltY + 1, xx + 4, beltY + 6, { color: V.alpha(C.muted, 0.32), lw: 1.1 }); }
      ord.forEach((p, i) => {
        const P = P11[p], col = PCOL[p];
        const x = lx + i * (cardW + cg);
        const active = i === k && k < n, doneC = i < k || (k >= n && sim.fail < 0);
        const failed = sim.fail === i && i === k && frac > 0.5;
        const dead = sim.fail >= 0 && i > sim.fail;
        const lift = active && !failed ? -5 * Math.sin(Math.PI * Math.min(1, frac * 1.6)) : 0;
        const cy = y + lift;
        g.save(); g.globalAlpha = dead ? 0.38 : 1;
        V.box(g, x, cy, cardW, cardH, { r: 8, fill: V.alpha(failed ? C.c : col, active || doneC ? 0.24 : 0.1), stroke: failed ? C.c : active ? col : V.alpha(col, 0.55), lw: active ? 2 : 1.1 });
        if (sim.sc.ed === p) { g.setLineDash([3, 3]); V.box(g, x + 2.5, cy + 2.5, cardW - 5, cardH - 5, { r: 6, stroke: V.alpha(C.ink, 0.6), lw: 1 }); g.setLineDash([]); }
        V.text(g, P.n, x + cardW / 2, cy + (wide ? 17 : cardH / 2 + 0.5), { size: wide ? 12 : (cardW < 40 ? 9.5 : 10.5), font: 'mono', weight: 700, align: 'center', base: 'middle', color: C.ink });
        if (wide) V.text(g, fit(g, P.s, cardW - 8, 9.5, 'mono'), x + cardW / 2, cy + 34, { size: 9.5, font: 'mono', align: 'center', base: 'middle', color: C.muted });
        if (sim.sc.ed === p) V.text(g, '✎', x + 8, cy + 9, { size: 10, align: 'center', base: 'middle', color: C.ink });
        if (failed) icon(g, 'bad', x + cardW - 7, cy + 8, 5.5, C.c);
        else if (doneC) icon(g, 'ok', x + cardW - 7, cy + 8, 5.5, C.a);
        g.restore();
      });
      y = beltY + 22;
      // ---------- mensagem do cenário
      const MSG11 = {
        relDigest: 'Os patches são os da release: o hash bate com IMAGE_TAG e a imagem vem por digest. O pull só é aceito porque o label tf.patches dela é o mesmo hash calculado aqui. A esteira mostra o build que quem publicou rodou: aqui nada é aplicado.',
        relBuild: 'PULL=0: o mesmo build roda aqui. Mesmos diffs, mesma ordem, mesmo hash, mesmo label. Os bytes não precisam sair idênticos aos da imagem publicada (a base é uma tag, o xgrammar uma faixa de versões), e é por isso que a release fixa o digest.',
        relLocal: 'Este Spark já tem uma imagem com tf.patches igual ao hash: nada a construir nem baixar, e a pasta de kernels dessa imagem já existe.',
        edit: 'Mexer no 0032 não muda a lista, mas muda o hash: a tag da release não serve mais e ninguém publicou a tag nova. O build roda aqui, a imagem ganha outro label e outra pasta de kernels, e a primeira partida compila de novo.',
        swap: 'O 0011 foi escrito contra a lista que o 0010 deixou: as linhas que ele remove já contêm split×5. Trocados, ele não encontra o que procura, o patch recusa o trecho e o || exit 1 derruba o build. Nem vizinhos comutam.',
        drop: 'Sem o 0013, a lista não ganha kvb, e o 0015, escrito depois dele, não encontra a linha que espera. Apagar um patch do meio quebra o primeiro patch que dependia dele.',
      };
      const mk = S11.sc === 'rel' ? (PL.src === 'digest' ? 'relDigest' : PL.src === 'local' ? 'relLocal' : 'relBuild') : S11.sc;
      const mOK = !(S11.sc === 'swap' || S11.sc === 'drop');
      const mlh = wide ? 16.5 : 15;
      const mMax = wide ? Math.max(2, Math.floor((h - y - 4) / mlh)) : 5;
      badge(g, mOK ? 'ok' : 'bad', lx + 8, y + 4, 7.5);
      const usedM = wtext(g, MSG11[mk], lx + 22, y + 9, lw - 22, { size: wide ? 12 : 11.5, lh: mlh, color: C.ink, maxLines: mMax });
      // ---------- painel do prepare.sh
      let py = wide ? 16 : y + usedM + 18;
      V.text(g, 'prepare.sh · de onde vem a imagem', rx, py, { size: 10.5, font: 'mono', color: C.muted });
      py += 10;
      const tE = tB + PIP + n * PD, tF = tB + PIP + Math.max(0, sim.fail) * PD + 0.6;
      const h12 = sim.h, tag = 'v0.6.0-' + h12;
      const rows = [];
      rows.push({ tm: 0.35, k: 'info', a: 'hash = sha256(patches/*.patch + extras), 12 dígitos', b: h12 + (PL.rel ? '  (= IMAGE_TAG da release)' : '  (≠ release)') });
      rows.push({ tm: 0.75, k: PL.src === 'local' ? 'ok' : 'skip', a: 'imagem local com tf.patches = hash?', b: PL.src === 'local' ? 'sim: nada a construir nem baixar' : 'não' });
      if (PL.src === 'digest') rows.push({ tm: 1.2, k: 'ok', a: 'pull por digest (hash da release)', b: '…@sha256:22789f0c… · label confere' });
      else if (PL.src === 'local') rows.push({ tm: 1.2, k: 'skip', a: 'pull', b: 'desnecessário' });
      else if (PL.tried) rows.push({ tm: 1.2, k: 'warn', a: 'pull :' + tag, b: 'tag não publicada → construir aqui' });
      else rows.push({ tm: 1.2, k: 'skip', a: 'pull', b: 'PULL=0 → construir aqui' });
      if (PL.src === 'build') rows.push({ tm: PL.fail ? tF : tE + 0.5, k: PL.fail ? 'bad' : 'ok', a: 'build local' + (S11.ready ? ' (camada do pip em cache: segundos)' : ' (do zero: base, pip e diffs)'), b: PL.fail ? 'hunk do ' + P11[sim.steps[sim.fail].p].n + ' não encaixa → exit 1' : '53 diffs aplicados · import de prova ✓' });
      else rows.push({ tm: 1.5, k: 'skip', a: 'build local', b: 'pulado: a imagem já traz os diffs' });
      const t5 = PL.src === 'build' ? (PL.fail ? tF + 0.25 : tE + 0.9) : 1.8;
      if (PL.fail) rows.push({ tm: t5, k: 'off', a: 'LABEL, worker e cache de kernels', b: 'não chegam a rodar' });
      else {
        rows.push({ tm: t5, k: 'ok', a: 'LABEL tf.patches', b: h12 });
        rows.push({ tm: t5 + 0.45, k: 'ok', a: 'worker: ' + (PL.src === 'digest' ? 'pull do mesmo digest' : PL.src === 'local' ? 'já tem a mesma imagem' : 'docker save | ssh docker load'), b: (PL.src === 'build' ? 'só as camadas que faltam · ' : '') + 'identidade de conteúdo = head' });
        rows.push({ tm: t5 + 0.9, k: PL.src === 'local' ? 'ok' : 'info', a: 'kernels: ~/.cache/tensorfold-glm53/' + h12, b: PL.src === 'local' ? 'pasta já existe: nada a compilar' : 'pasta nova: a 1ª partida compila (alguns minutos)' });
      }
      const tFin = rows[rows.length - 1].tm + 0.45;
      const asz = wide ? 11.5 : 11, bsz = wide ? 10.5 : 10, alh = asz + 4, blh = bsz + 4;
      rows.forEach((r) => {
        const on = el >= r.tm;
        g.globalAlpha = on ? 1 : 0.35;
        badge(g, on ? r.k : 'wait', rx + 9, py + 9, 8);
        const ua = wtext(g, r.a, rx + 24, py + 8, rw - 26, { size: asz, lh: alh, color: C.ink, weight: 600, maxLines: 2 });
        const ub = wtext(g, on ? r.b : '…', rx + 24, py + 8 + ua + 1, rw - 26, { size: bsz, lh: blh, font: 'mono', color: r.k === 'bad' && on ? C.c : C.muted, maxLines: 2 });
        g.globalAlpha = 1;
        py += ua + ub + (wide ? 9 : 8);
      });
      if (el >= tFin) {
        const fk = PL.fail ? 'bad' : 'ok', fcol = PL.fail ? C.c : C.a;
        const ftxt = PL.fail ? 'prepare.sh falhou antes de o start.sh parar qualquer coisa: o servidor no ar continua.' : 'pronto: o start.sh segue para as checagens e o lançamento.';
        const fl = lines(g, ftxt, rw - 34, asz, 'sans', 600).length;
        const fh = Math.min(3, fl) * alh + 12;
        V.box(g, rx, py + 2, rw, fh, { r: 8, fill: V.alpha(fcol, 0.1), stroke: V.alpha(fcol, 0.55) });
        badge(g, fk, rx + 13, py + 2 + fh / 2, 7);
        wtext(g, ftxt, rx + 28, py + 2 + 6 + asz * 0.8, rw - 34, { size: asz, lh: alh, color: C.ink, weight: 600, maxLines: 3 });
      }
    });

    /* =====================================================================
       F12 — TensorFold: paridade conferida na partida
       ===================================================================== */
    const f12 = root.querySelector('#fig-tf-paridade');
    // lista simplificada (12 dos 36 inteiros) com os valores da receita; d = padrão do patch quando o TF_GLM_* não chega ao rank
    const FLD12 = [
      { k: 'drafter', v: 'dflash2' },
      { k: 'long_ctx', v: '1' },
      { k: 'prefill_rows', v: '2048' },
      { k: 'grid', v: '64', d: '0' },
      { k: 'kvb', v: 'bf16' },
      { k: 'copy (match·máx.)', v: '8·15', d: '0·0', copy: true },
      { k: 'split (hc)', v: 'on', d: 'off' },
      { k: 'shared', v: '256', d: '0' },
      { k: 'parallel', v: '4', cli: true },
      { k: 'verify', v: 'batched' },
      { k: 'MAX_ROWS', v: '16' },
      { k: 'wide graphs', v: '16', d: '—' },
      { k: 'livre (MiB)', spare: true },
    ];
    const S12 = { knob: 'none', fwd: true, check: true, t0: -5.5 };   // começa já no desfecho; a animação se repete
    function vec12(r) {
      const envOK = r === 0 || S12.fwd;
      return FLD12.map((f) => {
        if (f.spare) return r === 0 ? 's0' : 's1';
        if (f.cli) return S12.knob === 'cli' ? '2' : f.v;
        if (f.d != null && !envOK) return f.d;
        if (f.copy && S12.knob === 'copy') return '8·8';
        return f.v;
      });
    }
    function out12() {
      const a = vec12(0), b = vec12(1);
      const diff = FLD12.map((f, i) => !f.spare && a[i] !== b[i]);
      const nd = diff.filter(Boolean).length;
      return { a, b, diff, nd, kind: nd === 0 ? 'live' : S12.check ? 'refuse' : 'hang' };
    }
    const c12 = UI.controls(f12);
    let H12 = null;
    function upd12() {
      const O = out12();
      r12.set(O.kind === 'live' ? 'LIVE nos dois' : O.kind === 'refuse' ? 'recusa na partida' : 'trava (hipotético)', O.kind === 'live' ? 'ok' : O.kind === 'refuse' ? 'warn' : 'bad');
    }
    const restart12 = () => { S12.t0 = H12 ? H12.t : 0; upd12(); };
    fitSel(UI.select(c12, { label: 'Mudança', options: [{ v: 'none', label: 'nenhuma (receita)' }, { v: 'copy', label: 'COPY_MAX=8 no .env' }, { v: 'cli', label: './start.sh restart --parallel 2' }], value: 'none', onChange: (v) => { S12.knob = v; restart12(); } }));
    UI.toggle(c12, { label: 'repassar TF_GLM_* também ao worker (ssh)', value: true, onChange: (v) => { S12.fwd = v; restart12(); } });
    UI.toggle(c12, { label: 'conferência na partida (TensorFold)', value: true, onChange: (v) => { S12.check = v; restart12(); } });
    UI.button(c12, '▶ subir os dois ranks', () => restart12(), { primary: true });
    const r12 = UI.readout(c12, 'desfecho', '—');
    upd12();
    respH(f12, (fw) => (fw < 640 ? 620 : 500));
    H12 = V.fig(f12, (g, w, h, t) => {
      const C = V.C(), TF = TFCOL(C), wide = w >= 640;
      const O = out12(), el = (t - S12.t0) % 12.5;
      const nF = FLD12.length;
      // ---------- geometria da tabela: [própria0 | do outro0] [campo] [do outro1 | própria1]
      const fwN = wide ? 150 : Math.min(104, Math.round(w * 0.28));
      const vw = wide ? 84 : Math.max(44, Math.floor((w - 40 - fwN - 12) / 4));
      const tot = 4 * vw + fwN + 12;
      const x0 = Math.round((w - tot) / 2);
      const xs = [x0, x0 + vw + 2, x0 + 2 * vw + 6, x0 + 2 * vw + 6 + fwN + 4, x0 + 3 * vw + fwN + 12];
      const xF = xs[2];
      const vsz = wide ? 11 : 10, fsz = wide ? 11 : 10;
      // ---------- faixa de cima: start.sh e os dois docker run
      const topH = wide ? 84 : 96;
      const bxW = Math.min(wide ? 330 : w - 40, w - 40), bxX = (w - bxW) / 2, bxY = 6, bxH = wide ? 40 : 44;
      const glowA = V.clamp(1 - el / 1.2, 0, 1);
      V.box(g, bxX, bxY, bxW, bxH, { r: 9, fill: V.alpha(TF, 0.1 + 0.12 * glowA), stroke: V.alpha(TF, 0.7), lw: 1.4 });
      V.text(g, 'start.sh (head)', bxX + 10, bxY + 14, { size: 10.5, font: 'mono', weight: 700, color: C.ink });
      const knobTxt = S12.knob === 'copy' ? '.env: COPY_MAX=8 → TF_GLM_COPY_MAX=8' : S12.knob === 'cli' ? 'argumento extra: --parallel 2 (vai aos dois)' : 'config.sh: defaults da receita';
      V.text(g, fit(g, knobTxt, bxW - 20, wide ? 10.5 : 10, 'mono'), bxX + 10, bxY + 30, { size: wide ? 10.5 : 10, font: 'mono', color: C.muted });
      const colL = (xs[0] + xs[1] + vw) / 2, colR = (xs[3] + xs[4] + vw) / 2;
      const aY0 = bxY + bxH, aY1 = topH;
      const flowA = V.clamp(el / 1.1, 0, 1);
      g.lineDashOffset = -t * 20;
      V.arrow(g, bxX + 24, aY0, colL, aY1, { color: V.alpha(C.a, 0.85), lw: 1.6, head: 7, curve: -16, dash: [5, 5] });
      V.arrow(g, bxX + bxW - 24, aY0, colR, aY1, { color: V.alpha(S12.fwd ? C.a : C.c, 0.85), lw: 1.6, head: 7, curve: 16, dash: [5, 5] });
      g.lineDashOffset = 0;
      if (flowA < 1) { V.dot(g, V.lerp(bxX + 24, colL, flowA), V.lerp(aY0, aY1, flowA), 4, C.a); V.dot(g, V.lerp(bxX + bxW - 24, colR, flowA), V.lerp(aY0, aY1, flowA), 4, S12.fwd ? C.a : C.c); }
      const lsz = wide ? 10 : 9.5;
      if (wide) {
        V.text(g, 'docker run -e TF_GLM_*…', bxX - 8, aY0 + 22, { size: lsz, font: 'mono', align: 'right', color: C.muted });
        V.text(g, S12.fwd ? 'ssh worker docker run -e TF_GLM_*…' : 'ssh worker docker run (sem -e)', bxX + bxW + 8, aY0 + 22, { size: lsz, font: 'mono', color: S12.fwd ? C.muted : C.c });
      } else {
        V.text(g, '-e TF_GLM_*', bxX + 54, aY0 + 16, { size: lsz, font: 'mono', color: C.muted });
        V.text(g, S12.fwd ? 'ssh: -e TF_GLM_*' : 'ssh: sem -e', bxX + bxW - 54, aY0 + 16, { size: lsz, font: 'mono', align: 'right', color: S12.fwd ? C.muted : C.c });
      }
      // ---------- cabeçalhos
      const hy = topH + 14;
      const outcomeT = S12.check ? 4.9 : 2.8, hangT = outcomeT + 1.5;
      const stateHead = (r) => {
        if (el < outcomeT) return { s: r === 0 ? 'rank 0 · head' : 'rank 1 · worker', c: C.muted };
        if (O.kind === 'live') return { s: (r === 0 ? 'rank 0' : 'rank 1') + ' · LIVE', c: C.a };
        if (O.kind === 'refuse') return { s: (r === 0 ? 'rank 0' : 'rank 1') + (wide ? ' · RuntimeError' : ' · erro'), c: C.c };
        return { s: (r === 0 ? 'rank 0' : 'rank 1') + (el < hangT ? ' · LIVE?' : wide ? ' · travado' : ' · trava'), c: el < hangT ? C.d : C.c };
      };
      const h0 = stateHead(0), h1 = stateHead(1);
      const hsz = wide ? 11 : 10;
      V.text(g, fit(g, h0.s, 2 * vw + 14, hsz, 'mono', 700), xs[0] - 12, hy, { size: hsz, font: 'mono', weight: 700, color: h0.c });
      V.text(g, fit(g, h1.s, 2 * vw + 14, hsz, 'mono', 700), xs[4] + vw + 12, hy, { size: hsz, font: 'mono', weight: 700, align: 'right', color: h1.c });
      const hangNow = O.kind === 'hang' && el >= hangT;
      if (hangNow) { icon(g, 'wait', xF + 10, hy - 4, 7, C.c); V.text(g, fit(g, 'coletiva parada', fwN + 2 * vw - 40, 10, 'mono'), xF + 22, hy, { size: 10, font: 'mono', color: C.c }); }
      else V.text(g, S12.check ? 'campo' : 'campo · sem conferência', xF + fwN / 2, hy, { size: 10, font: 'mono', align: 'center', color: C.muted });
      const sy = hy + 13;
      const ssz = wide ? 9.5 : 9;
      V.text(g, 'própria', xs[0] + vw / 2, sy, { size: ssz, font: 'mono', align: 'center', color: C.muted });
      if (S12.check) {
        V.text(g, 'do outro', xs[1] + vw / 2, sy, { size: ssz, font: 'mono', align: 'center', color: C.muted });
        V.text(g, 'do outro', xs[3] + vw / 2, sy, { size: ssz, font: 'mono', align: 'center', color: C.muted });
      }
      V.text(g, 'própria', xs[4] + vw / 2, sy, { size: ssz, font: 'mono', align: 'center', color: C.muted });
      // ---------- linhas
      const rowH = wide ? 21 : 20, ty = sy + 8;
      const fillT = 1.1, gatherT = 2.4, cmpT = 3.6;
      for (let i = 0; i < nF; i++) {
        const f = FLD12[i], yy = ty + i * rowH, cyy = yy + rowH / 2;
        const appear = V.clamp((el - fillT - i * 0.08) / 0.2, 0, 1);
        const cmp = S12.check && el >= cmpT + i * 0.07;
        const df = O.diff[i];
        if (cmp) {
          const col = f.spare ? C.muted : df ? C.c : C.a;
          V.box(g, xs[0] - 4, yy + 1, tot + 8, rowH - 2, { r: 5, fill: V.alpha(col, f.spare ? 0.06 : df ? 0.16 : 0.07) });
        } else if (i % 2 === 0) V.box(g, xs[0] - 4, yy + 1, tot + 8, rowH - 2, { r: 5, fill: V.alpha(C.muted, 0.05) });
        V.text(g, fit(g, f.k, fwN - 6, fsz, 'mono'), xF + fwN / 2, cyy + 0.5, { size: fsz, font: 'mono', align: 'center', base: 'middle', color: f.spare ? C.muted : C.ink });
        const vals = [O.a[i], O.b[i]];
        g.globalAlpha = appear;
        const own = (r) => {
          const xx = r === 0 ? xs[0] : xs[4];
          const changed = (r === 1 && !S12.fwd && f.d != null) || (f.copy && S12.knob === 'copy') || (f.cli && S12.knob === 'cli');
          V.box(g, xx + 2, yy + 3, vw - 4, rowH - 6, { r: 4, fill: V.alpha(changed ? (r === 1 && !S12.fwd && f.d != null ? C.c : TF) : C.b, 0.14), stroke: V.alpha(C.b, 0.35), lw: 0.8 });
          V.text(g, fit(g, vals[r], vw - 8, vsz, 'mono'), xx + vw / 2, cyy + 0.5, { size: vsz, font: 'mono', weight: 600, align: 'center', base: 'middle', color: C.ink });
        };
        own(0); own(1);
        g.globalAlpha = 1;
        // all-gather: cada valor viaja para o slot "do outro" do outro rank
        const gu = S12.check ? V.clamp((el - gatherT - i * 0.05) / 0.9, 0, 1) : 0;
        if (gu > 0) {
          const e = V.ease.inOut(gu);
          [[0, xs[0], xs[3]], [1, xs[4], xs[1]]].forEach(([r, from, to]) => {
            const xx = V.lerp(from, to, e), arc = Math.sin(Math.PI * e) * (r === 0 ? -6 : 6);
            g.globalAlpha = gu < 1 ? 0.85 : 1;
            V.box(g, xx + 2, yy + 3 + arc, vw - 4, rowH - 6, { r: 4, fill: V.alpha(C.b, gu < 1 ? 0.22 : 0.1), stroke: V.alpha(C.b, 0.35), lw: 0.8 });
            V.text(g, fit(g, vals[r], vw - 8, vsz, 'mono'), xx + vw / 2, cyy + 0.5 + arc, { size: vsz, font: 'mono', align: 'center', base: 'middle', color: C.ink });
            g.globalAlpha = 1;
          });
        }
        if (cmp) {
          const kind = f.spare ? 'skip' : df ? 'bad' : 'ok';
          icon(g, kind === 'ok' ? 'ok' : kind === 'bad' ? 'bad' : 'skip', xs[0] - 12, cyy, 5, kind === 'ok' ? C.a : kind === 'bad' ? C.c : C.muted);
          icon(g, kind === 'ok' ? 'ok' : kind === 'bad' ? 'bad' : 'skip', xs[4] + vw + 12, cyy, 5, kind === 'ok' ? C.a : kind === 'bad' ? C.c : C.muted);
        }
      }
      const tableEnd = ty + nF * rowH;
      if (S12.check && el >= cmpT + nF * 0.07) V.text(g, wide ? 'livre (MiB) não entra na comparação: os dois adotam min(s0, s1)' : 'livre (MiB): fora da comparação → min(s0, s1)', w / 2, tableEnd + 12, { size: wide ? 10 : 9.5, font: 'mono', align: 'center', color: C.muted });
      // ---------- desfecho
      const oy = tableEnd + 24;
      if (el >= outcomeT) {
        let msg;
        if (O.kind === 'live') msg = S12.knob === 'copy' ? 'O config.sh lê COPY_MAX=8 do .env e exporta TF_GLM_COPY_MAX=8. Como toda TF_GLM_* vai com -e aos dois containers, as duas listas mudam juntas e batem.' : S12.knob === 'cli' ? 'Argumentos extras vão aos dois ranks depois dos defaults, e o argparse fica com o último valor: parallel = 2 nos dois. Argumento é paridade por construção.' : (S12.check ? 'As duas listas batem, campo a campo. O motor segue para a carga dos pesos, e o rank 0 abre a API.' : 'Sem a conferência ninguém compara nada, e mesmo assim tudo funciona: quem faz as listas baterem é o launcher, que repassa os mesmos knobs aos dois ranks.');
        else if (O.kind === 'refuse') msg = 'Sem os -e no ssh, o rank 1 sobe com os padrões dos patches (copy drafts, divisão das hyper-connections e prefixo compartilhado desligados, grade 0). Só na lista da figura, ' + O.nd + ' campos divergem, e os dois ranks param na partida: “the two ranks were started with different settings”, com as duas listas. O start.sh mostra as últimas linhas dos dois logs.';
        else msg = 'Hipotético: sem a conferência, os dois subiriam e responderiam. Na primeira requisição, o rank 0 usaria copy drafts e dividiria a cola das hyper-connections; o rank 1 não. As coletivas sairiam de passo, e a primeira que divergisse travaria os dois: a trava silenciosa das receitas vLLM.';
        const kk = O.kind === 'live' ? 'ok' : O.kind === 'refuse' ? 'warn' : 'bad';
        const col = O.kind === 'live' ? C.a : O.kind === 'refuse' ? C.d : C.c;
        const mlh = wide ? 16.5 : 14.5, msz = wide ? 12 : 11;
        const nl = Math.min(9, lines(g, msg, w - 64, msz).length);
        const bh = Math.min(h - oy - 6, nl * mlh + 14);
        V.box(g, 14, oy, w - 28, bh, { r: 9, fill: V.alpha(col, 0.09), stroke: V.alpha(col, 0.5) });
        badge(g, kk, 28, oy + 15, 7.5);
        wtext(g, msg, 42, oy + 9 + msz * 0.75, w - 64, { size: msz, lh: mlh, color: C.ink, maxLines: Math.max(1, Math.floor((bh - 8) / mlh)) });
      }
    });
  },
});

LA.chapter({
  id: 'numeros-no-computador',
  part: 1,
  order: 4,
  title: 'Como o computador guarda números',
  navTitle: 'Números no computador',
  subtitle: 'Bits, inteiros, ponto flutuante, arredondamento e bytes: a física dos números por baixo de cada receita, do marcador −877.912.083 do GLM à cópia BF16 exata da KDA.',
  minutes: 58,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['bits', 'bytes', 'complemento de dois', 'int32', 'int64', 'ponto flutuante', 'FP32', 'FP16', 'BF16', 'RNE', 'ulp', 'subnormais', 'acumulador FP32', 'não-associatividade', 'produto exato', 'GiB', 'invariância de lote', 'escala potência de 2'],
  goals: [
    'Ler um padrão de bits como <b>inteiro</b> (com e sem sinal), <b>ponto flutuante</b> ou <b>contêiner empacotado</b>, e saber quando reinterpretar (<code>view</code>) e quando converter',
    'Explicar sinal, expoente com bias, mantissa, subnormais, ulp e arredondamento ao par, e prever a faixa e a precisão de FP32, FP16 e BF16',
    'Reconhecer os perigos numéricos que as receitas corrigem: overflow de índices <code>int32</code>, absorção em acumuladores e não-associatividade das reduções paralelas, e explicar como a versão TensorFold da receita GLM garante os mesmos bits sozinha ou em lote',
    'Provar por que E4M3 × BF16 é exato em FP32 e dizer o que <code>KDA_BF16_LARGE_M_CHUNK_ROWS = 512</code> controla e o que não controla',
    'Calcular em GiB o tamanho de qualquer tensor, com escalas, e ler contagens como “125B total, 6B ativos”',
  ],
  prereqs: ['tensores-algebra'],
  summary: [
    'Bits não têm significado próprio: o <b>dtype</b> é a lente. Contêiner não é formato (FP4 em <code>uint8</code>, treliça em <code>int16</code>). <code>view</code> reinterpreta, <code>.to()</code>/<code>copy_()</code> convertem, e trocar um pelo outro destrói dados (a escala E8M0 2⁻⁷ vira 0).',
    'Inteiros vivem num relógio de 2ⁿ posições: complemento de dois (INT8 −128…127, INT4 −8…7) e aritmética mod 2ⁿ. O marcador MCG −877.912.083 é <code>0xCBAC1FED</code> lido com sinal. Offsets <code>int32</code> estouram acima de 2³¹−1. O risco depende de quantos blocos cada tensor de cache tem (≥ ~57 mil blocos de 37.376 B), não do tamanho do prompt, e por isso o cache K do DeepSeek promove o índice a <code>int64</code> antes de multiplicar.',
    'Um float vale (−1)ˢ · 2^(E−bias) · (1 + M/2ᵐ): o expoente dá a faixa e a mantissa dá a precisão. O ulp dobra a cada binade, o erro relativo de um arredondamento é no máximo 2^−(m+1) e os subnormais preenchem o intervalo perto do zero.',
    '<b>BF16</b> (1/8/7) tem a faixa do FP32 e ~2,4 dígitos decimais: é a base do serving. <b>FP16</b> (1/5/10) tem passo 8× menor, mas estoura em 65.504: aparece onde a faixa está garantida, como nos kernels EXL3 (em que o <code>swiglu_limit</code> = 10 do próprio modelo limita o SwiGLU) e no V do dot PV do Qwen. <b>FP32</b> fica onde números pequenos decidem ou se acumulam: roteadores, estados recorrentes, acumuladores.',
    'Toda soma arredonda. A absorção (256 + 1 = 256 em BF16) exige acumuladores FP32, e a soma em float não é associativa: a ordem das reduções paralelas muda os bits. Fixar essa ordem (determinismo do Qwen) custa ~4%. A versão TensorFold da receita GLM faz disso um contrato (rascunho = serial, concorrente = solo): uma cadeia FP32 por linha, operações dependentes do lote rodando por fluxo, atômicos só com inteiros, soma em ordem de rank e um KV FP8 de 528 B por token e camada, cuja escala 2<sup>k</sup> não arredonda nada.',
    'E4M3 (4 bits significativos) × BF16 (8) dá no máximo 12 bits, que cabem exatos nos 24 do FP32. Por isso a cópia BF16 da KDA tem um único arredondamento (RMSE relativo de 0,0025 contra o Marlin). Os blocos de 512 linhas só limitam o rascunho FP32 da carga (8 MiB por temporário contra 196,5 MiB) e, como a operação é elemento a elemento, não mudam nenhum bit.',
    'bytes = numel × bits/8 + escalas, contados em GiB (2³⁰) e não em GB. Os 51,2 B de parâmetros da tabela N-gram em FP8 ocupam 47,68 GiB. O total de parâmetros define a capacidade necessária e os ativos definem a velocidade do decode.',
  ],
  quiz: [
    { q: 'O checkpoint EXL3 do GLM traz, ao lado de cada expert, o marcador <code>mcg</code> como <code>int32</code> igual a −877.912.083, e o overlay o confere na carga. Qual é a relação com o multiplicador <code>0xCBAC1FED</code> do codebook?', o: ['Nenhuma: são constantes independentes', 'São os mesmos 32 bits. Sem sinal, 0xCBAC1FED = 3.417.055.213; como o bit 31 está ligado, em complemento de dois isso vale 3.417.055.213 − 2³² = −877.912.083', 'É o multiplicador arredondado para FP32', 'É o negativo do multiplicador, −0xCBAC1FED'], a: 1, e: 'Em complemento de dois, o bit mais alto pesa −2³¹ em vez de +2³¹, o que equivale a subtrair 2³² da leitura sem sinal. O padrão de bits não muda, só a leitura. A carga falha se esse marcador não bater em todos os experts.' },
    { q: 'Por que o BF16, e não o FP16, virou o formato base das ativações no serving?', o: ['Porque tem mais bits de mantissa', 'Porque tem os mesmos 8 bits de expoente do FP32: a mesma faixa, sem overflow em 65.504 e com conversão trivial do FP32, ao custo de uma precisão (~2,4 dígitos) que as redes toleram', 'Porque é mais preciso perto de zero', 'Porque ocupa 1 byte'], a: 1, e: 'O FP16 tem passo 8× menor (10 bits de mantissa contra 7), mas só tem 5 bits de expoente e vira ∞ a partir de 65.520. Ativações com picos grandes precisam de faixa. O FP16 aparece onde a faixa está garantida: dentro dos kernels EXL3, onde o <code>swiglu_limit</code> do próprio modelo (o mesmo que o vLLM aplica em BF16) limita o produto do SwiGLU, e no V do dot PV do Qwen, onde P ≤ 1.' },
    { q: 'Em BF16, quanto dá 256 + 1?', o: ['257', '258', '256', 'NaN'], a: 2, e: 'Em [256, 512) o ulp do BF16 é 2, então os vizinhos são 256 e 258. 257 cai exatamente no meio e o RNE escolhe o par, 256 (mantissa terminando em 0). O 1 foi absorvido. É por isso que somas longas precisam de acumuladores FP32.' },
    { q: 'Se <code>KDA_BF16_LARGE_M_CHUNK_ROWS</code> passasse de 512 para 2.048, o que mudaria?', o: ['Os bits da cópia BF16, porque a ordem dos arredondamentos muda', 'O limiar de despacho entre Marlin e cuBLAS', 'Só o tamanho do rascunho FP32 na carga (8 → 32 MiB por temporário) e o número de voltas. Os bits são idênticos, porque cada elemento depende apenas do seu código e da sua escala', 'O tamanho da cópia BF16 (98,25 MiB por camada)'], a: 2, e: 'A construção é elemento a elemento: não há soma, logo não há ordem que possa mudar. O limiar de despacho é outra constante (KDA_BF16_LARGE_M_MIN_M = 512, sobre as linhas de ativação M), e o tamanho da cópia depende só da forma da matriz.' },
    { q: 'Por que o produto de um código E4M3 por uma escala BF16 é exato em FP32?', o: ['Porque o FP32 tem mais faixa', 'Porque 4 + 8 = 12 bits significativos cabem nos 24 do FP32, e os expoentes apenas se somam (dentro da faixa)', 'Porque o E4M3 é um inteiro', 'Porque o Marlin calcula em FP32'], a: 1, e: 'Multiplicar significandos de p e q bits dá no máximo p + q bits. Com 12 ≤ 24, o FP32 guarda o produto sem arredondar, e a única perda acontece na conversão final para BF16. Com uma escala FP32 (24 bits), o produto pode ter até 28 bits, e então o FP32 pode arredondar: seriam até dois arredondamentos.' },
    { q: 'No cache K do DeepSeek, cada bloco ocupa 37.376 B. Num tensor de cache com mais de 57 mil blocos, o que acontece com <code>block_idx * block_stride</code> em <code>int32</code> no bloco 57.457, e qual é a correção?', o: ['Nada: o int32 vai até 4,29 bilhões', 'O produto (≈ 2,1475·10⁹) passa de 2³¹−1, fica negativo e o ponteiro aponta ~2 GiB antes do cache. A correção é promover o índice a int64 antes de multiplicar', 'O kernel arredonda o endereço para o bloco mais próximo', 'Basta converter o produto para int64 depois da multiplicação'], a: 1, e: '57.457 × 37.376 = 2.147.512.832 > 2.147.483.647. Em complemento de dois, isso vira −2.147.454.464. Converter depois não adianta, porque o estouro já aconteceu. O overlay faz block_idx.to(tl.int64) * block_stride.' },
    { q: 'No KV FP8 da versão TensorFold da receita GLM, cada linha de 512 valores ganha a escala \\(s = 2^{\\lceil \\log_2 \\text{amax} \\rceil - 8}\\). Por que uma potência de dois, e não \\(\\text{amax}/448\\) em FP32?', o: ['Porque uma potência de dois ocupa menos bytes que um FP32', 'Porque o E4M3 só aceita escalas que sejam potências de dois', 'Porque multiplicar ou dividir por 2<sup>k</sup> só mexe no expoente: a desquantização não arredonda e, como a precisão relativa do formato não depende da escala, quase nada se perde', 'Porque assim o maior código cai exatamente em 448'], a: 2, e: 'A escala é guardada em FP32 de qualquer jeito (são 4 dos 528 bytes da linha). Uma escala 2<sup>k</sup> só mexe no campo de expoente, então código × escala é exato em FP32, e a atenção calcula sobre a linha desquantizada sem um segundo arredondamento. Como cada binade tem os mesmos 2<sup>m</sup> valores, deslizar a linha por outro fator não melhora a precisão relativa: só os valores abaixo de cerca de 2<sup>−14</sup> do máximo, que caem nos subnormais do E4M3, perdem algo. Nas linhas reais do GLM, escalas mais finas ficaram a 0,001 do erro de 2,65% por linha. E o maior código fica em (128, 256], não em 448.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui, N = LA.num;
    const $ = (s) => root.querySelector(s);

    /* ------------------------------------------------------------ helpers */
    const DV = new DataView(new ArrayBuffer(4));
    const f32 = (u) => { DV.setUint32(0, u >>> 0); return DV.getFloat32(0); };
    const u32 = (x) => { DV.setFloat32(0, x); return DV.getUint32(0); };
    const rB = (x) => N.round('bf16', x);
    const rH = (x) => N.round('fp16', x, 'ieee');
    const r32 = Math.fround;
    const SUP = { '-': '⁻', '−': '⁻', '+': '⁺', '(': '⁽', ')': '⁾', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', 'm': 'ᵐ' };
    const sup = (n) => String(n).split('').map((c) => SUP[c] || c).join('');
    const grp = (s) => s.replace(/\B(?=(\d{3})+(?!\d))/g, '.');
    const bi = (n) => (n < 0 ? '−' : '') + grp(String(Math.abs(Math.round(n))));
    const sci = (x, d = 3) => {
      if (x === 0) return '0';
      if (!Number.isFinite(x)) return Number.isNaN(x) ? 'NaN' : (x < 0 ? '−∞' : '+∞');
      let e = Math.floor(Math.log10(Math.abs(x)));
      let ms = (Math.abs(x) / Math.pow(10, e)).toFixed(d);
      if (+ms >= 10) { e += 1; ms = (Math.abs(x) / Math.pow(10, e)).toFixed(d); }
      return (x < 0 ? '−' : '') + ms.replace('.', ',') + '×10' + sup(e);
    };
    const nice = (x, sig = 7) => {
      if (Number.isNaN(x)) return 'NaN';
      if (!Number.isFinite(x)) return (x < 0 ? '−' : '+') + '∞';
      if (x === 0) return Object.is(x, -0) ? '−0' : '0';
      const a = Math.abs(x);
      if (a >= 1e-4 && a < 1e9) {
        let s = a.toPrecision(sig);
        if (s.includes('e')) return sci(x, Math.max(1, sig - 1));
        if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
        const [ip, fp] = s.split('.');
        return (x < 0 ? '−' : '') + grp(ip) + (fp ? ',' + fp : '');
      }
      return sci(x, Math.min(4, Math.max(1, sig - 1)));
    };
    const pct = (x) => (x === 0 ? '0' : x < 1e-4 ? sci(x * 100, 1) + '%' : (x * 100).toFixed(x < 0.001 ? 4 : x < 0.01 ? 3 : 2).replace('.', ',') + '%');
    const bin = (v, n) => (v >>> 0).toString(2).padStart(n, '0').slice(-n);
    const onFill = (C) => (C.dark ? '#07140d' : '#ffffff');
    const vb = (b) => V.bytes(b).replace('.', ',');
    const fitH = (fig, want) => { const st = fig.querySelector('.la-fig-stage'); want = Math.round(want); if (Math.abs(st.clientHeight - want) > 3) st.style.height = want + 'px'; };
    const wrap = (g, text, x, y, maxW, lh, o = {}) => {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(text).split(' ');
      let line = '', yy = y, n = 0;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) { V.text(g, line, x, yy, o); line = wd; yy += lh; n++; g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400); }
        else line = test;
      }
      if (line) { V.text(g, line, x, yy, o); n++; }
      return n;
    };
    const bracket = (g, x1, x2, y, col, down = true) => {
      const d = down ? 5 : -5;
      g.strokeStyle = col; g.lineWidth = 1.2; g.beginPath();
      g.moveTo(x1, y); g.lineTo(x1, y + d); g.lineTo(x2, y + d); g.lineTo(x2, y); g.stroke();
    };

    /* ================================================================
       Figura 1 — um padrão de 32 bits, sete leituras
       ================================================================ */
    {
      const fig = $('#fig-leituras');
      const S = { u: 0xCBAC1FED >>> 0, mode: 'i32' };
      const PRE = [
        { v: 'mcg', label: 'MCG · 0xCBAC1FED', u: 0xCBAC1FED },
        { v: 'pi', label: 'π (FP32)', u: 0x40490FDB },
        { v: 'one', label: '1,0 (FP32)', u: 0x3F800000 },
        { v: 'm1', label: '−1 (int32)', u: 0xFFFFFFFF },
        { v: 'mask', label: 'XOR do MCG · 0x3B603B60', u: 0x3B603B60 },
        { v: 'min', label: 'menor int32', u: 0x80000000 },
      ];
      const MODES = [{ v: 'u32', label: 'uint32' }, { v: 'i32', label: 'int32' }, { v: 'f32', label: 'FP32' }, { v: 'f16', label: '2×FP16' }, { v: 'b16', label: '2×BF16' }, { v: 'u8', label: '4×uint8' }, { v: 'fp4', label: '8×FP4' }];
      const ctl = UI.controls(fig);
      UI.select(ctl, { label: 'Padrão', options: PRE.map((p) => ({ v: p.v, label: p.label })), value: 'mcg', onChange: (v) => { S.u = PRE.find((p) => p.v === v).u >>> 0; H.redraw(); } });
      UI.seg(ctl, { label: 'Ler como', options: MODES, value: 'i32', onChange: (v) => { S.mode = v; H.redraw(); } });
      UI.button(ctl, '+1', () => { S.u = (S.u + 1) >>> 0; H.redraw(); });
      UI.button(ctl, '−x (inverte e soma 1)', () => { S.u = ((~S.u) + 1) >>> 0; H.redraw(); });
      const rHex = UI.readout(ctl, 'hex', '');

      const H = V.fig(fig, (g, w) => {
        const C = V.C();
        const narrow = w < 640;
        const perRow = narrow ? 16 : 32;
        const nibGap = 3, byteGap = 8;
        const gapsW = (perRow / 4 - 1) * nibGap + (perRow / 8 - 1) * (byteGap - nibGap);
        const cw = Math.max(12, Math.min(26, (w - 28 - gapsW) / perRow));
        const rowW = cw * perRow + gapsW;
        const x0 = Math.max(10, (w - rowW) / 2);
        const ch = Math.round(Math.min(30, cw * 1.2));
        const needBr = ['f16', 'b16', 'u8', 'fp4'].includes(S.mode);
        const top = 30, rowStep = ch + (needBr ? 60 : 34);
        const cells = [];
        for (let k = 0; k < 32; k++) {
          const r = narrow ? (k >= 16 ? 1 : 0) : 0, c = narrow ? k % 16 : k;
          cells.push({ x: x0 + c * cw + Math.floor(c / 4) * nibGap + Math.floor(c / 8) * (byteGap - nibGap), y: top + r * rowStep, bit: 31 - k });
        }
        const P = H.pointer; let hov = -1;
        if (P.inside) cells.forEach((c, i) => { if (P.x >= c.x && P.x < c.x + cw - 1 && P.y >= c.y && P.y < c.y + ch) hov = i; });
        if (P.clicked && hov >= 0) { S.u = (S.u ^ (1 << cells[hov].bit)) >>> 0; P.clicked = false; }
        H.canvas.style.cursor = hov >= 0 ? 'pointer' : 'default';
        const u = S.u >>> 0;
        rHex.set('0x' + u.toString(16).toUpperCase().padStart(8, '0'));
        const fieldCol = (bit) => {
          switch (S.mode) {
            case 'u32': return C.b;
            case 'i32': return bit === 31 ? C.c : C.b;
            case 'f32': return bit === 31 ? C.c : bit >= 23 ? C.d : C.a;
            case 'f16': { const b = bit % 16; return b === 15 ? C.c : b >= 10 ? C.d : C.a; }
            case 'b16': { const b = bit % 16; return b === 15 ? C.c : b >= 7 ? C.d : C.a; }
            case 'u8': return (bit >> 3) % 2 ? C.b : C.e;
            default: { const b = bit % 4; return b === 3 ? C.c : b >= 1 ? C.d : C.a; }
          }
        };
        cells.forEach((c, i) => {
          const on = (u >>> c.bit) & 1, col = fieldCol(c.bit);
          V.box(g, c.x, c.y, cw - 2, ch, { r: 4, fill: on ? V.alpha(col, 0.88) : V.alpha(col, 0.12), stroke: i === hov ? C.ink : V.alpha(col, 0.6), lw: i === hov ? 2 : 1 });
          V.text(g, on ? '1' : '0', c.x + (cw - 2) / 2, c.y + ch / 2 + 1, { size: Math.min(13, cw * 0.56), font: 'mono', align: 'center', base: 'middle', color: on ? onFill(C) : C.muted, weight: 600 });
        });
        for (let nb = 0; nb < 8; nb++) {
          const c0 = cells[nb * 4], c3 = cells[nb * 4 + 3];
          V.text(g, ((u >>> (28 - 4 * nb)) & 15).toString(16).toUpperCase(), (c0.x + c3.x + cw - 2) / 2, c0.y - 9, { size: 12, font: 'mono', align: 'center', color: C.muted, weight: 600 });
        }
        V.text(g, 'bit 31', cells[0].x, cells[0].y - 22, { size: 11, font: 'mono', color: C.muted });
        V.text(g, 'bit 0', cells[31].x + cw - 2, cells[31].y - 22, { size: 11, font: 'mono', color: C.muted, align: 'right' });

        const groups = [];
        if (S.mode === 'f16' || S.mode === 'b16') [1, 0].forEach((hi) => { const b16 = hi ? u >>> 16 : u & 0xFFFF; groups.push({ hiBit: hi ? 31 : 15, loBit: hi ? 16 : 0, v: N.decode(S.mode === 'f16' ? 'fp16' : 'bf16', b16) }); });
        if (S.mode === 'u8') for (let bI = 3; bI >= 0; bI--) groups.push({ hiBit: bI * 8 + 7, loBit: bI * 8, lab: String((u >>> (bI * 8)) & 255) });
        if (S.mode === 'fp4') for (let n = 7; n >= 0; n--) groups.push({ hiBit: n * 4 + 3, loBit: n * 4, v: N.decode('e2m1', (u >>> (n * 4)) & 15) });
        groups.forEach((gp) => {
          const a = cells[31 - gp.hiBit], b = cells[31 - gp.loBit];
          const xa = a.x + 1, xb = b.x + cw - 3, yb = a.y + ch + 6;
          bracket(g, xa, xb, yb, V.alpha(C.ink, 0.45));
          V.text(g, gp.lab != null ? gp.lab : nice(gp.v, 6), (xa + xb) / 2, yb + 22, { size: 12.5, font: 'mono', align: 'center', color: C.ink, weight: 600, maxW: xb - xa + 8 });
        });

        // painel da leitura atual
        const y0 = cells[31].y + ch + (groups.length ? 52 : 34);
        const Sb = u >>> 31, E = (u >>> 23) & 255, M = u & 0x7FFFFF;
        const hiH = N.decode('fp16', u >>> 16), loH = N.decode('fp16', u & 0xFFFF);
        const hiB = N.decode('bf16', u >>> 16), loB = N.decode('bf16', u & 0xFFFF);
        const bytes = [(u >>> 24) & 255, (u >>> 16) & 255, (u >>> 8) & 255, u & 255];
        const hx = (b) => b.toString(16).toUpperCase().padStart(2, '0');
        const nibs = []; for (let n = 7; n >= 0; n--) nibs.push(N.decode('e2m1', (u >>> (n * 4)) & 15));
        let big = '', f1 = '', f2 = '';
        switch (S.mode) {
          case 'u32': big = bi(u); f1 = 'Σ bᵢ·2ⁱ para i de 0 a 31'; f2 = 'faixa: 0 … 4.294.967.295'; break;
          case 'i32': big = bi(u | 0); f1 = Sb ? `o bit 31 pesa −2³¹: ${bi(u)} − 2³² = ${bi(u | 0)}` : 'bit 31 desligado: a leitura coincide com a do uint32'; f2 = 'faixa: −2.147.483.648 … 2.147.483.647'; break;
          case 'f32': big = nice(f32(u), 8);
            f1 = E === 255 ? (M ? 'expoente todo em 1 e mantissa ≠ 0 → NaN' : 'expoente todo em 1 e mantissa 0 → ∞') : E === 0 ? `subnormal: (−1)${sup(Sb)} · 2⁻¹²⁶ · (${bi(M)} / 2²³)` : `(−1)${sup(Sb)} · 2${sup(E - 127)} · (1 + ${bi(M)} / 2²³)`;
            f2 = `S = ${Sb} · E = ${E} (bias 127) · M = ${bi(M)}`; break;
          case 'f16': big = nice(hiH, 6) + '   |   ' + nice(loH, 6); f1 = 'cada metade: 1 bit de sinal · 5 de expoente (bias 15) · 10 de mantissa'; f2 = `o codebook MCG soma as duas metades: ${nice(hiH + loH, 8)}`; break;
          case 'b16': big = nice(hiB, 6) + '   |   ' + nice(loB, 6); f1 = 'cada metade: 1 · 8 (bias 127) · 7. A metade de cima é o FP32 truncado'; f2 = `FP32 inteiro: ${nice(f32(u), 8)}   ·   metade de cima em BF16: ${nice(hiB, 6)}`; break;
          case 'u8': big = bytes.join('   '); f1 = 'na memória (little-endian), o byte menos significativo vem primeiro:'; f2 = 'endereço +0 +1 +2 +3  →  ' + bytes.slice().reverse().map(hx).join(' '); break;
          default: big = nibs.map((v) => nice(v)).join('  '); f1 = 'cada nibble é S·E·E·M (FP4 E2M1): dois valores por byte'; f2 = 'é o contêiner do NVFP4 e do MXFP4 (no Qwen, o nibble baixo é o 1º elemento)';
        }
        const lab = MODES.find((m) => m.v === S.mode).label;
        V.text(g, 'LEITURA ' + lab.toUpperCase(), x0, y0, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        V.text(g, big, x0, y0 + 28, { size: narrow ? 19 : 24, font: 'display', weight: 700, color: C.ink, maxW: w - x0 - 12 });
        V.text(g, f1, x0, y0 + 50, { size: 12, color: C.text2 || C.muted, maxW: w - x0 - 12 });
        V.text(g, f2, x0, y0 + 68, { size: 12, font: 'mono', color: C.muted, maxW: w - x0 - 12 });

        // todas as leituras
        const all = [
          ['u32', 'uint32', bi(u)], ['i32', 'int32', bi(u | 0)], ['f32', 'FP32', nice(f32(u), 8)],
          ['f16', '2×FP16', nice(hiH, 5) + ' | ' + nice(loH, 5)], ['b16', '2×BF16', nice(hiB, 5) + ' | ' + nice(loB, 5)],
          ['u8', '4×uint8', bytes.join(' ')], ['fp4', '8×FP4', nibs.map((v) => nice(v)).join(' ')],
        ];
        const cols = narrow ? 1 : 2, colW = (w - 2 * x0) / cols, lh = 21, yL = y0 + 92;
        all.forEach((r, i) => {
          const cx = x0 + (i % cols) * colW, cy = yL + Math.floor(i / cols) * lh;
          if (r[0] === S.mode) V.box(g, cx - 6, cy - 14, colW - 8, lh - 2, { r: 6, fill: V.alpha(C.a, 0.14) });
          V.text(g, r[1], cx, cy, { size: 11.5, font: 'mono', color: C.muted, weight: 600 });
          V.text(g, r[2], cx + 70, cy, { size: 12, font: 'mono', color: C.ink, maxW: colW - 84 });
        });
        fitH(fig, yL + Math.ceil(all.length / cols) * lh + 6);
      }, { static: true });
    }

    /* ================================================================
       Figura 2 — o relógio dos inteiros
       ================================================================ */
    {
      const fig = $('#fig-relogio');
      const S = { n: 4, v: 5, k: 3, signed: true, from: 5, to: 5, t0: 0, dur: 0.001, msg: 'Aperte “somar k” ou “multiplicar por k”.', msgCol: 'muted', op: '', full: null };
      const Mn = () => 1 << S.n;
      const toS = (x) => (x >= Mn() / 2 ? x - Mn() : x);
      let H;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Bits', options: [3, 4, 8], value: 4, onChange: (v) => { S.n = v; S.v %= Mn(); S.from = S.to = S.v; S.full = null; S.op = ''; sk.input.max = Mn() - 1; if (S.k > Mn() - 1) sk.set(Mn() - 1); S.msg = `${v} bits: ${Mn()} padrões.`; S.msgCol = 'muted'; } });
      const sk = UI.slider(ctl, { label: 'k', min: 1, max: 15, step: 1, value: 3, onInput: (v) => { S.k = v; } });
      const op = (kind) => {
        const m = Mn(), v = S.v, k = S.k;
        const exact = kind === 'add' ? v + k : v * k;
        const res = exact % m;
        let steps = exact - v;
        if (steps > 3 * m) steps = 3 * m + (((res - v) % m) + m) % m;
        S.from = S.to % m; S.to = S.from + steps; S.t0 = H.t; S.dur = Math.min(2.4, 0.3 + steps * (kind === 'add' ? 0.14 : 0.03));
        S.v = res; S.full = kind === 'mul' ? exact : null;
        const sym = kind === 'add' ? '+' : '×';
        S.op = `${v} ${sym} ${k} = ${exact}`;
        const sv = toS(v), exS = kind === 'add' ? sv + k : sv * k;
        if (S.signed) {
          const ov = exS > m / 2 - 1 || exS < -m / 2;
          S.msg = ov ? `estouro com sinal: ${sv} ${sym} ${k} deveria dar ${exS}, mas ${S.n} bits guardam ${toS(res)}` : `com sinal: ${sv} ${sym} ${k} = ${toS(res)} (cabe)`;
          S.msgCol = ov ? 'red' : 'a';
        } else {
          const ov = exact >= m;
          S.msg = ov ? `estouro sem sinal: ${exact} não cabe em ${S.n} bits; sobram os de baixo = ${res}` : `sem sinal: ${exact} cabe em ${S.n} bits`;
          S.msgCol = ov ? 'd' : 'a';
        }
      };
      UI.button(ctl, 'somar k', () => op('add'), { primary: true });
      UI.button(ctl, 'multiplicar por k', () => op('mul'));
      UI.button(ctl, 'zerar', () => { S.v = 0; S.from = S.to = 0; S.full = null; S.op = ''; S.msg = 'Zerado.'; S.msgCol = 'muted'; });
      UI.toggle(ctl, { label: 'ler com sinal (complemento de 2)', value: true, onChange: (v) => { S.signed = v; } });

      H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const m = Mn(), narrow = w < 640;
        const R = narrow ? Math.min(w / 2 - 56, 122) : Math.min(140, Math.max(110, w * 0.17));
        const cx = narrow ? w / 2 : Math.min(w * 0.34, R + 100), cy = R + 62;
        const ang = (i) => -Math.PI / 2 + (2 * Math.PI * i) / m;
        const pr = V.clamp((t - S.t0) / S.dur, 0, 1);
        const disp = V.lerp(S.from, S.to, V.ease.inOut(pr));
        // anéis
        g.lineCap = 'butt'; g.lineWidth = 12;
        if (S.signed) {
          g.strokeStyle = V.alpha(C.a, 0.35); g.beginPath(); g.arc(cx, cy, R, ang(-0.5), ang(m / 2 - 0.5)); g.stroke();
          g.strokeStyle = V.alpha(C.c, 0.35); g.beginPath(); g.arc(cx, cy, R, ang(m / 2 - 0.5), ang(m - 0.5)); g.stroke();
        } else { g.strokeStyle = V.alpha(C.b, 0.35); g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke(); }
        // fronteiras
        const edge = (i, col, label, out) => {
          const a = ang(i - 0.5), c = Math.cos(a), s = Math.sin(a);
          V.line(g, cx + c * (R - 20), cy + s * (R - 20), cx + c * (R + 20), cy + s * (R + 20), { color: col, lw: 3 });
          V.text(g, label, cx + c * (R + (out || 34)), cy + s * (R + (out || 34)) + 4, { size: 11, font: 'mono', color: col, align: 'center', weight: 600 });
        };
        edge(0, C.d, 'vai-um perdido', 46);
        if (S.signed) edge(m / 2, C.red || C.c, 'sinal vira', 46);
        // posições
        const cur = ((Math.round(disp) % m) + m) % m;
        for (let i = 0; i < m; i++) {
          const a = ang(i), c = Math.cos(a), s = Math.sin(a);
          const isCur = i === cur;
          if (m <= 16) {
            V.dot(g, cx + c * R, cy + s * R, isCur ? 7 : 4.5, isCur ? C.ink : V.alpha(C.ink, 0.5));
            V.text(g, bin(i, S.n), cx + c * (R + (narrow ? 20 : 24)), cy + s * (R + (narrow ? 20 : 24)) + 4, { size: 11, font: 'mono', align: 'center', color: isCur ? C.ink : C.muted, weight: isCur ? 700 : 400 });
            const val = S.signed ? toS(i) : i;
            V.text(g, String(val).replace('-', '−'), cx + c * (R - 26), cy + s * (R - 26) + 4, { size: 12.5, font: 'mono', align: 'center', color: S.signed && i >= m / 2 ? C.c : S.signed ? C.a : C.b, weight: isCur ? 700 : 500 });
          } else {
            V.line(g, cx + c * (R - 5), cy + s * (R - 5), cx + c * (R + 5), cy + s * (R + 5), { color: V.alpha(C.ink, i % 16 === 0 ? 0.7 : 0.25), lw: i % 16 === 0 ? 1.6 : 1 });
            if (i % 32 === 0) { const val = S.signed ? toS(i) : i; V.text(g, String(val).replace('-', '−'), cx + c * (R - 24), cy + s * (R - 24) + 4, { size: 11, font: 'mono', align: 'center', color: S.signed && i >= m / 2 ? C.c : C.muted }); }
          }
        }
        // ponteiro
        const a = ang(disp), hx = cx + Math.cos(a) * (R - (m <= 16 ? 44 : 34)), hy = cy + Math.sin(a) * (R - (m <= 16 ? 44 : 34));
        V.line(g, cx, cy, hx, hy, { color: C.ink, lw: 3 });
        V.dot(g, cx, cy, 6, C.ink);
        V.glow(g, cx + Math.cos(a) * R, cy + Math.sin(a) * R, 18, S.signed && cur >= m / 2 ? C.c : C.a);
        // painel
        const px = narrow ? 18 : cx + R + 70, py = narrow ? cy + R + 70 : 40, pw = w - px - 16;
        V.text(g, 'PADRÃO DE BITS', px, py, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const bw = Math.min(30, pw / S.n - 3);
        for (let i = 0; i < S.n; i++) {
          const bit = (S.v >> (S.n - 1 - i)) & 1;
          const col = i === 0 && S.signed ? C.c : C.b;
          V.box(g, px + i * (bw + 3), py + 8, bw, 26, { r: 4, fill: bit ? V.alpha(col, 0.85) : V.alpha(col, 0.12), stroke: V.alpha(col, 0.6), lw: 1 });
          V.text(g, bit, px + i * (bw + 3) + bw / 2, py + 22, { size: 12, font: 'mono', align: 'center', base: 'middle', color: bit ? onFill(C) : C.muted, weight: 600 });
        }
        if (S.signed) V.text(g, `o bit mais alto pesa −2${sup(S.n - 1)} = −${1 << (S.n - 1)}`, px, py + 50, { size: 11.5, color: C.c, maxW: pw });
        V.text(g, 'sem sinal', px, py + 74, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, String(S.v), px + 82, py + 74, { size: 15, font: 'mono', color: S.signed ? C.muted : C.ink, weight: 700 });
        V.text(g, 'com sinal', px, py + 96, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, String(toS(S.v)).replace('-', '−'), px + 82, py + 96, { size: 15, font: 'mono', color: S.signed ? C.ink : C.muted, weight: 700 });
        let yy = py + 124;
        if (S.op) { V.text(g, S.op, px, yy, { size: 13, font: 'mono', color: C.ink, weight: 600, maxW: pw }); yy += 22; }
        if (S.full != null && S.full >= m) {
          const fb = S.full.toString(2), low = fb.slice(-S.n), high = fb.slice(0, fb.length - S.n);
          g.font = V.font(12.5, 'mono', 600);
          const wh = g.measureText(high).width;
          V.text(g, high, px, yy, { size: 12.5, font: 'mono', color: C.red || C.c, weight: 600 });
          V.line(g, px, yy - 4, px + wh, yy - 4, { color: C.red || C.c, lw: 1.5 });
          V.text(g, low, px + wh + 2, yy, { size: 12.5, font: 'mono', color: C.a, weight: 700 });
          V.text(g, `→ os bits altos caem fora: sobra ${low}₂ = ${S.v}`, px, yy + 18, { size: 11.5, color: C.muted, maxW: pw });
          yy += 40;
        }
        const mc = S.msgCol === 'red' ? (C.red || C.c) : S.msgCol === 'muted' ? C.muted : C[S.msgCol];
        const nl = wrap(g, S.msg, px, yy + 4, pw, 16, { size: 12, color: mc, weight: 500 });
        fitH(fig, narrow ? yy + 4 + nl * 16 + 14 : Math.max(2 * R + 130, yy + nl * 16 + 20));
      });
    }

    /* ================================================================
       Figura 3 — o endereço que dá a volta (int32 × int64)
       ================================================================ */
    {
      const fig = $('#fig-odometro');
      const STRIDE = 37376, LIM = Math.floor((2 ** 31 - 1) / STRIDE); // 57.456
      const S = { b: 30000, i64: false };
      const ctl = UI.controls(fig);
      const sb = UI.slider(ctl, { label: 'block_idx', min: 0, max: 100000, step: 1, value: 30000, fmt: (v) => bi(v), onInput: (v) => { S.b = v; H.redraw(); } });
      UI.button(ctl, 'ir ao limite', () => sb.set(S.b === LIM ? LIM + 1 : LIM));
      UI.toggle(ctl, { label: 'índice em int64 (.to(tl.int64))', value: false, onChange: (v) => { S.i64 = v; H.redraw(); } });
      const r1 = UI.readout(ctl, 'offset real', ''), r2 = UI.readout(ctl, 'offset calculado', '');
      const H = V.fig(fig, (g, w) => {
        const C = V.C();
        const narrow = w < 640, pad = 16;
        const real = S.b * STRIDE;
        const low = real % 2 ** 32;
        const i32 = low >= 2 ** 31 ? low - 2 ** 32 : low;
        const calc = S.i64 ? real : i32;
        const bad = calc !== real;
        r1.set(bi(real) + ' B'); r2.set(bi(calc) + ' B');
        V.text(g, 'offset = block_idx × block_stride', pad, 24, { size: 12, font: 'mono', color: C.muted });
        V.text(g, `${bi(S.b)} × 37.376 = ${bi(real)} B`, pad, 46, { size: narrow ? 14 : 16, font: 'mono', color: C.ink, weight: 700, maxW: w - 2 * pad });
        // registrador de 32 bits
        const regY = 78;
        V.text(g, S.i64 ? 'REGISTRADOR DE 64 BITS · os 32 de baixo (os de cima são 0)' : 'REGISTRADOR INT32 · o bit 31 é o sinal', pad, regY, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        const gap = 2, cw = Math.min(22, (w - 2 * pad - 7 * 4) / 32);
        for (let k = 0; k < 32; k++) {
          const bit = 31 - k, on = Math.floor(low / 2 ** bit) % 2;
          const x = pad + k * cw + Math.floor(k / 4) * 4;
          const col = bit === 31 && !S.i64 ? (C.red || C.c) : C.b;
          V.box(g, x, regY + 10, cw - gap, 24, { r: 3, fill: on ? V.alpha(col, 0.85) : V.alpha(col, 0.12), stroke: V.alpha(col, 0.5), lw: 1 });
          if (cw >= 15) V.text(g, on, x + (cw - gap) / 2, regY + 23, { size: 11, font: 'mono', align: 'center', base: 'middle', color: on ? onFill(C) : C.muted });
        }
        V.text(g, S.i64 ? `valor: ${bi(calc)}` : `lido como int32: ${bi(i32)}`, pad, regY + 54, { size: 12.5, font: 'mono', color: bad ? (C.red || C.c) : C.a, weight: 600, maxW: w - 2 * pad });
        // reta numérica
        const ax0 = pad + 8, ax1 = w - pad - 8, ay = regY + 120;
        const vmin = -2.35e9, vmax = 4.0e9;
        const X = (v) => ax0 + ((v - vmin) / (vmax - vmin)) * (ax1 - ax0);
        V.box(g, X(-(2 ** 31)), ay - 26, X(2 ** 31) - X(-(2 ** 31)), 52, { r: 6, fill: V.alpha(C.b, 0.08) });
        V.text(g, 'faixa do int32', X(-(2 ** 31)) + 6, ay - 32, { size: 11, font: 'mono', color: C.b });
        g.save(); g.beginPath(); g.rect(X(2 ** 31), ay - 26, ax1 - X(2 ** 31), 52); g.clip();
        for (let x = X(2 ** 31) - 60; x < ax1; x += 9) V.line(g, x, ay + 26, x + 30, ay - 26, { color: V.alpha(C.red || C.c, 0.18), lw: 1 });
        g.restore();
        // pool de KV
        const gr = g.createLinearGradient(X(0), 0, ax1, 0); gr.addColorStop(0, V.alpha(C.a, 0.55)); gr.addColorStop(1, V.alpha(C.a, 0.15));
        g.fillStyle = gr; V.rr(g, X(0), ay + 4, ax1 - X(0), 16, 5); g.fill();
        V.text(g, narrow ? 'cache K →' : 'tensor de cache K →', X(0) + 6, ay + 16, { size: 11, font: 'mono', color: C.dark ? onFill(C) : C.ink, weight: 600 });
        V.line(g, ax0, ay, ax1, ay, { color: C.muted, lw: 1.3 });
        [[-(2 ** 31), '−2³¹'], [0, '0'], [2 ** 30, '1 GiB'], [2 ** 31, '2³¹'], [3 * 2 ** 30, '3 GiB']].forEach(([v, l]) => {
          V.line(g, X(v), ay - 5, X(v), ay + 5, { color: C.muted, lw: 1.3 });
          V.text(g, l, X(v), ay + 38, { size: 11, font: 'mono', color: v === 2 ** 31 ? (C.red || C.c) : C.muted, align: 'center' });
        });
        V.line(g, X(2 ** 31), ay - 28, X(2 ** 31), ay + 28, { color: C.red || C.c, lw: 1.5, dash: [4, 3] });
        // marcadores
        const tri = (x, y, col, up) => { g.beginPath(); g.moveTo(x, y); g.lineTo(x - 7, y + (up ? 11 : -11)); g.lineTo(x + 7, y + (up ? 11 : -11)); g.closePath(); g.fillStyle = col; g.fill(); };
        tri(X(real), ay - 4, C.a, false);
        const al = (xx) => (xx > w - 70 ? 'right' : xx < 70 ? 'left' : 'center');
        if (bad) {
          tri(X(calc), ay - 4, C.red || C.c, false);
          V.arrow(g, X(real) - 4, ay - 22, X(calc) + 6, ay - 20, { color: C.red || C.c, lw: 2, curve: 34 });
          V.text(g, 'o kernel vai aqui', X(calc), ay + 58, { size: 11.5, color: C.red || C.c, align: al(X(calc)), weight: 600 });
        }
        V.text(g, 'deveria ir aqui', X(real), ay + 58, { size: 11.5, color: C.a, align: al(X(real)), weight: 600 });
        // estado
        const sy = ay + 86;
        const msg = !bad
          ? (S.i64 && real > 2 ** 31 - 1 ? `✓ Com int64 o produto ${bi(real)} cabe: o endereço cai no bloco ${bi(S.b)}.` : `✓ ${bi(real)} ≤ 2.147.483.647: cabe em int32, o endereço está correto.`)
          : `✕ O produto passou de 2³¹−1 e virou ${bi(calc)}. O kernel lê ou escreve ${nice(Math.abs(calc) / 2 ** 30, 3)} GiB antes do início do cache: acesso ilegal (IMA) ou KV corrompido.`;
        const nl = wrap(g, msg, pad, sy, w - 2 * pad, 17, { size: 12.5, color: bad ? (C.red || C.c) : C.a, weight: 600 });
        const nl2 = wrap(g, `limite: ⌈2³¹ / 37.376⌉ = 57.457 → o bloco 57.456 é o último seguro num tensor de cache`, pad, sy + nl * 17 + 8, w - 2 * pad, 16, { size: 11.5, font: 'mono', color: C.muted });
        fitH(fig, sy + nl * 17 + 8 + nl2 * 16 + 8);
      }, { static: true });
    }

    /* ================================================================
       Figura 4 — explorador de ponto flutuante
       ================================================================ */
    {
      const fig = $('#fig-anatomia');
      const dec = (fmt, b) => (fmt === 'fp32' ? f32(b) : N.decode(fmt, b));
      const enc = (fmt, x) => (fmt === 'fp32' ? u32(x) : N.encode(fmt, x, 'ieee').bits) >>> 0;
      const S = { fmt: 'bf16', bits: 0 };
      S.bits = enc('bf16', Math.PI);
      const FMT = [{ v: 'fp32', label: 'FP32' }, { v: 'bf16', label: 'BF16' }, { v: 'fp16', label: 'FP16' }, { v: 'e4m3', label: 'E4M3' }, { v: 'e2m1', label: 'E2M1' }];
      const PRE = [
        { v: '1', label: '1', x: 1 }, { v: 'pi', label: 'π', x: Math.PI }, { v: '0.1', label: '0,1', x: 0.1 }, { v: '65504', label: '65.504', x: 65504 },
        { v: '1e-5', label: '0,00001', x: 1e-5 }, { v: 'max', label: 'maior finito' }, { v: 'minn', label: 'menor normal' }, { v: 'minsub', label: 'menor subnormal' }, { v: 'inf', label: '+∞', x: Infinity }, { v: 'nan', label: 'NaN', x: NaN },
      ];
      const nb = () => N.formats[S.fmt].bits;
      const neighbor = (bits, dir) => {
        const sbv = 2 ** (nb() - 1), s = bits >= sbv ? 1 : 0, mag = bits - s * sbv;
        if (Number.isNaN(dec(S.fmt, bits))) return null;
        let ns = s, nm;
        if (dir > 0) { if (s === 0) nm = mag + 1; else if (mag === 0) { ns = 0; nm = 1; } else nm = mag - 1; }
        else { if (s === 1) nm = mag + 1; else if (mag === 0) { ns = 1; nm = 1; } else nm = mag - 1; }
        if (nm >= sbv) return null;
        const r = ns * sbv + nm;
        return Number.isNaN(dec(S.fmt, r)) ? null : r;
      };
      let H;
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Formato', options: FMT, value: 'bf16', onChange: (v) => { const x = dec(S.fmt, S.bits); S.fmt = v; S.bits = enc(v, x); H.redraw(); } });
      UI.select(ctl, { label: 'Valor', options: PRE.map((p) => ({ v: p.v, label: p.label })), value: 'pi', onChange: (v) => {
        const p = PRE.find((q) => q.v === v); let x = p.x;
        if (v === 'max') x = N.maxFinite(S.fmt); if (v === 'minn') x = N.minNormal(S.fmt); if (v === 'minsub') x = N.minSub(S.fmt);
        S.bits = enc(S.fmt, x); H.redraw();
      } });
      UI.button(ctl, '← anterior', () => { const r = neighbor(S.bits, -1); if (r != null) { S.bits = r >>> 0; H.redraw(); } });
      UI.button(ctl, 'próximo →', () => { const r = neighbor(S.bits, 1); if (r != null) { S.bits = r >>> 0; H.redraw(); } });
      UI.button(ctl, '×2', () => { S.bits = enc(S.fmt, dec(S.fmt, S.bits) * 2); H.redraw(); });
      UI.button(ctl, '÷2', () => { S.bits = enc(S.fmt, dec(S.fmt, S.bits) / 2); H.redraw(); });
      UI.button(ctl, '±', () => { S.bits = (S.bits ^ (2 ** (nb() - 1))) >>> 0; H.redraw(); });

      H = V.fig(fig, (g, w) => {
        const C = V.C();
        const f = N.formats[S.fmt], n = f.bits, narrow = w < 640;
        // layout dos bits: S | E | [1] | M
        const extra = 3; // gap depois do sinal, gap depois do expoente e a célula implícita
        const cw = Math.max(9, Math.min(28, (w - 32 - 20) / (n + extra)));
        const ch = Math.max(20, Math.min(30, cw * 1.15));
        const rowW = cw * (n + 1) + 20;
        const x0 = Math.max(12, (w - rowW) / 2), y0 = 36;
        const cells = [];
        let x = x0;
        for (let k = 0; k < n; k++) {
          const bit = n - 1 - k;
          if (k === 1) x += 8;
          if (k === 1 + f.e) { cells.push({ implicit: true, x }); x += cw + 4; }
          cells.push({ bit, x, field: k === 0 ? 'S' : k <= f.e ? 'E' : 'M' });
          x += cw;
        }
        if (f.m === 0) cells.push({ implicit: true, x: x + 4 });
        const P = H.pointer; let hov = -1;
        if (P.inside) cells.forEach((c, i) => { if (!c.implicit && P.x >= c.x && P.x < c.x + cw - 1 && P.y >= y0 && P.y < y0 + ch) hov = i; });
        if (P.clicked && hov >= 0) { S.bits = (S.bits ^ (2 ** cells[hov].bit)) >>> 0; P.clicked = false; }
        H.canvas.style.cursor = hov >= 0 ? 'pointer' : 'default';
        const b = S.bits >>> 0;
        const Sb = Math.floor(b / 2 ** (n - 1)) % 2, E = Math.floor(b / 2 ** f.m) % 2 ** f.e, M = b % 2 ** f.m;
        const maxE = 2 ** f.e - 1, xv = dec(S.fmt, b);
        let cls = 'normal';
        if (f.kind === 'ieee' && E === maxE) cls = M ? 'NaN' : 'infinito';
        else if (f.kind === 'fn' && E === maxE && M === 2 ** f.m - 1) cls = 'NaN';
        else if (E === 0) cls = M ? 'subnormal' : 'zero';
        const colF = { S: C.c, E: C.d, M: C.a };
        cells.forEach((c, i) => {
          if (c.implicit) {
            const imp = cls === 'normal' ? '1' : cls === 'subnormal' || cls === 'zero' ? '0' : '·';
            g.setLineDash([3, 3]); V.box(g, c.x, y0, cw - 2, ch, { r: 4, stroke: V.alpha(C.ink, 0.5), lw: 1.2 }); g.setLineDash([]);
            V.text(g, imp, c.x + (cw - 2) / 2, y0 + ch / 2 + 1, { size: cw >= 13 ? Math.min(13, Math.max(11, cw * 0.6)) : 10, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 });
            return;
          }
          const on = Math.floor(b / 2 ** c.bit) % 2, col = colF[c.field];
          V.box(g, c.x, y0, cw - 2, ch, { r: 4, fill: on ? V.alpha(col, 0.88) : V.alpha(col, 0.12), stroke: i === hov ? C.ink : V.alpha(col, 0.6), lw: i === hov ? 2 : 1 });
          if (cw >= 13) V.text(g, on, c.x + (cw - 2) / 2, y0 + ch / 2 + 1, { size: Math.min(13, Math.max(11, cw * 0.6)), font: 'mono', align: 'center', base: 'middle', color: on ? onFill(C) : C.muted, weight: 600 });
        });
        // rótulos dos campos
        const eCells = cells.filter((c) => c.field === 'E'), mCells = cells.filter((c) => c.field === 'M');
        const lab = (arr, text, short, col) => { if (!arr.length) return; const xa = arr[0].x, xb = arr[arr.length - 1].x + cw - 2; g.font = V.font(11, 'mono', 600); const t = g.measureText(text).width < xb - xa + 30 ? text : short; V.text(g, t, (xa + xb) / 2, y0 - 10, { size: 11, font: 'mono', color: col, align: 'center', weight: 600 }); };
        V.text(g, narrow ? 'S' : 'sinal', cells[0].x + (cw - 2) / 2, y0 - 10, { size: 11, font: 'mono', color: C.c, align: 'center', weight: 600 });
        lab(eCells, `expoente · ${f.e} bits`, 'E', C.d);
        lab(mCells, `mantissa · ${f.m} bits`, 'M', C.a);
        // decodificação
        const lx = Math.max(14, Math.min(x0, 40)), lw = w - lx - 14;
        let ly = y0 + ch + 34;
        const line = (txt, col, o = {}) => { V.text(g, txt, lx, ly, Object.assign({ size: 12.5, font: 'mono', color: col, maxW: lw }, o)); ly += o.step || 21; };
        const bias = f.bias;
        line(`S = ${Sb}  →  (−1)${sup(Sb)} = ${Sb ? '−1' : '+1'}`, C.c);
        if (cls === 'normal') line(`E = ${bin(E, f.e)}₂ = ${E}  →  e = ${E} − ${bias} = ${E - bias}`, C.d);
        else if (cls === 'subnormal' || cls === 'zero') line(`E = ${bin(E, f.e)}₂ = 0  →  subnormal: e = 1 − ${bias} = ${1 - bias}, sem o 1 implícito`, C.d);
        else line(`E = ${bin(E, f.e)}₂ (tudo em 1)  →  valor especial`, C.d);
        if (f.m > 0) {
          line(`M = ${bin(M, f.m)}₂ = ${bi(M)}`, C.a, { step: 18 });
          if (cls === 'normal') line(`   →  1 + ${bi(M)}/2${sup(f.m)} = ${nice(1 + M / 2 ** f.m, 10)}`, C.a);
          else if (cls === 'subnormal' || cls === 'zero') line(`   →  0 + ${bi(M)}/2${sup(f.m)} = ${nice(M / 2 ** f.m, 10)}`, C.a);
          else line(`   →  ${cls === 'NaN' ? 'mantissa ≠ 0 com expoente máximo: NaN' : 'mantissa 0 com expoente máximo: ∞'}`, C.a);
        }
        ly += 6;
        const e = cls === 'normal' ? E - bias : 1 - bias;
        const sig = cls === 'normal' ? 1 + M / 2 ** f.m : M / 2 ** f.m;
        const vtxt = cls === 'NaN' || cls === 'infinito' ? nice(xv) : `${Sb ? '−' : '+'}1 × 2${sup(e)} × ${nice(sig, 10)} = ${nice(xv, 10)}`;
        V.text(g, vtxt, lx, ly, { size: narrow ? 15 : 18, font: 'display', weight: 700, color: C.ink, maxW: lw - 90 });
        g.font = V.font(11, 'mono', 700);
        const bw = g.measureText(cls).width + 16;
        V.box(g, w - 14 - bw, ly - 16, bw, 22, { r: 11, fill: V.alpha(cls === 'normal' ? C.a : cls === 'subnormal' ? C.d : cls === 'zero' ? C.b : C.red || C.c, 0.22), stroke: cls === 'normal' ? C.a : cls === 'subnormal' ? C.d : cls === 'zero' ? C.b : C.red || C.c });
        V.text(g, cls, w - 14 - bw / 2, ly - 5, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 });
        ly += narrow ? 58 : 46;
        // vizinhos
        const pb = neighbor(b, -1), nx = neighbor(b, 1);
        if (Number.isFinite(xv)) {
          const pv = pb != null ? dec(S.fmt, pb) : null, nv = nx != null ? dec(S.fmt, nx) : null;
          const ulp = nv != null && Number.isFinite(nv) ? nv - xv : pv != null ? xv - pv : NaN;
          const ax0 = lx + 10, ax1 = w - 24, mid = (ax0 + ax1) / 2, half = (ax1 - ax0) * 0.36;
          V.line(g, ax0, ly, ax1, ly, { color: V.alpha(C.muted, 0.6), lw: 1.2 });
          const tick = (xx, col, t1, big) => { V.line(g, xx, ly - (big ? 12 : 8), xx, ly + (big ? 12 : 8), { color: col, lw: big ? 3 : 2 }); V.text(g, t1, xx, ly + 26, { size: 11.5, font: 'mono', color: col, align: 'center', maxW: half - 6 }); };
          const sg = f.m > 10 ? 9 : 7;
          if (pv != null) tick(mid - half, C.muted, (narrow ? '← ' : 'anterior ') + nice(pv, sg));
          if (nv != null) tick(mid + half, C.muted, (narrow ? '' : 'próximo ') + nice(nv, sg) + (narrow ? ' →' : ''));
          tick(mid, C.ink, nice(xv, sg + 1), true);
          if (Number.isFinite(ulp)) {
            // o colchete aponta para um vizinho que existe: o próximo, ou o anterior no fim da régua
            const toNext = nv != null && Number.isFinite(nv);
            const bx1 = toNext ? mid + 4 : mid - half + 4, bx2 = toNext ? mid + half - 4 : mid - 4;
            bracket(g, bx1, bx2, ly - 16, V.alpha(C.a, 0.9), false);
            V.text(g, `1 ulp = ${nice(ulp, 5)}`, (bx1 + bx2) / 2, ly - 26, { size: 11.5, font: 'mono', color: C.a, align: 'center', weight: 600 });
          }
          ly += 46;
          const nlE = wrap(g, `erro relativo máximo de um arredondamento neste formato: 2⁻${sup(f.m + 1).replace('⁻', '')} ≈ ${pct(2 ** -(f.m + 1))}${cls === 'subnormal' ? ' (nos subnormais, o erro relativo cresce)' : ''}`, lx, ly, lw, 15, { size: 11.5, color: C.muted });
          ly += 10 + (nlE - 1) * 15;
        } else { V.text(g, cls === 'NaN' ? 'NaN não tem vizinhos: qualquer conta com ele dá NaN.' : '∞ é o fim da régua: o anterior é o maior finito.', lx, ly, { size: 12, color: C.muted, maxW: lw }); ly += 10; }
        fitH(fig, ly + 14);
      }, { static: true });
    }

    /* ================================================================
       Figura 5 — a régua de um float de brinquedo
       ================================================================ */
    {
      const fig = $('#fig-regua');
      const S = { fmt: 'e3m2', sub: true, log: false };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Formato', options: [{ v: 'e2m1', label: 'E2M1 (FP4)' }, { v: 'e3m2', label: 'E3M2 (FP6)' }, { v: 'e4m3', label: 'E4M3 (FP8)' }], value: 'e3m2', onChange: (v) => { S.fmt = v; H.redraw(); } });
      UI.toggle(ctl, { label: 'subnormais', value: true, onChange: (v) => { S.sub = v; H.redraw(); } });
      UI.toggle(ctl, { label: 'eixo log₂', value: false, onChange: (v) => { S.log = v; H.redraw(); } });
      const H = V.fig(fig, (g, w, h) => {
        const C = V.C();
        const f = N.formats[S.fmt], T = N.table(S.fmt).map((o) => o.v);
        const mx = T[T.length - 1], minN = N.minNormal(S.fmt), minS = N.minSub(S.fmt);
        const pad = 24, x0 = pad + (S.log ? 30 : 4), x1 = w - pad;
        const L2 = Math.log2;
        const X = S.log ? (v) => (v <= 0 ? x0 - 22 : x0 + ((L2(v) - L2(minS)) / (L2(mx) - L2(minS))) * (x1 - x0)) : (v) => x0 + (v / mx) * (x1 - x0);
        const ry = 104;
        // binades
        const eMin = Math.round(L2(minN)), eMax = Math.floor(L2(mx));
        for (let e = eMin; e <= eMax; e++) {
          const a = X(2 ** e), b = X(Math.min(2 ** (e + 1), mx));
          V.box(g, a, ry - 48, Math.max(1, b - a), 72, { r: 3, fill: V.alpha((e - eMin) % 2 ? C.b : C.a, 0.1) });
          if (b - a > 24) V.text(g, '2' + sup(e), (a + b) / 2, ry - 54, { size: 11.5, font: 'mono', color: C.muted, align: 'center' });
        }
        const sa = X(S.log ? minS : 0), sbx = X(minN);
        V.box(g, sa, ry - 48, sbx - sa, 72, { r: 3, fill: V.alpha(C.c, S.sub ? 0.12 : 0.05) });
        if (S.sub && sbx - sa > 60) V.text(g, 'subnormais', (sa + sbx) / 2, ry + 40, { size: 11, font: 'mono', color: C.c, align: 'center' });
        V.line(g, x0 - (S.log ? 26 : 0), ry, x1, ry, { color: C.muted, lw: 1.2 });
        // tracinhos
        const P = H.pointer; let hv = -1, bd = 9;
        T.forEach((v, i) => {
          const xx = X(v);
          if (P.inside && Math.abs(P.x - xx) < bd && P.y > ry - 50 && P.y < ry + 30) { bd = Math.abs(P.x - xx); hv = i; }
        });
        T.forEach((v, i) => {
          const xx = X(v), sub = v > 0 && v < minN;
          const e = v >= minN ? Math.floor(L2(v)) : 0;
          const col = v === 0 ? C.ink : sub ? C.c : (e - eMin) % 2 ? C.b : C.a;
          if (sub && !S.sub) { g.setLineDash([2, 3]); V.line(g, xx, ry - 16, xx, ry, { color: V.alpha(C.c, 0.35), lw: 1.2 }); g.setLineDash([]); return; }
          const hh = i === hv ? 40 : sub ? 22 : 32;
          V.line(g, xx, ry - hh, xx, ry, { color: col, lw: i === hv ? 2.6 : 1.6 });
          if (v === 0 && S.log) V.dot(g, xx, ry, 3.5, C.ink);
        });
        if (!S.sub) { V.line(g, X(S.log ? minS : 0) + 2, ry + 26, X(minN) - 2, ry + 26, { color: C.red || C.c, lw: 3 }); V.text(g, 'tudo aqui vira 0', (X(S.log ? minS : 0) + X(minN)) / 2, ry + 42, { size: 11, color: C.red || C.c, align: 'center', weight: 600 }); }
        if (S.log) V.text(g, '0', x0 - 22, ry + 18, { size: 11, font: 'mono', color: C.muted, align: 'center' });
        { let lastR = -99; const marks = [0]; for (let e = eMin; e <= eMax; e++) marks.push(2 ** e); if (marks[marks.length - 1] !== mx) marks.push(mx);
          g.font = V.font(11, 'mono', 400);
          const maxL = X(mx) - g.measureText(nice(mx, 4)).width;
          marks.forEach((v) => { if (S.log && v === 0) return; const xx = X(v), t = nice(v, 4), tw = g.measureText(t).width; const align = v === mx ? 'right' : v === 0 ? 'left' : 'center'; const l = align === 'right' ? xx - tw : align === 'left' ? xx : xx - tw / 2; if (v !== mx && (l < lastR + 8 || l + tw > maxL - 8)) return; lastR = l + tw; V.text(g, t, xx, ry + 18, { size: 11, font: 'mono', color: C.muted, align }); g.font = V.font(11, 'mono', 400); }); }
        if (hv >= 0) {
          const v = T[hv], bits = N.encode(S.fmt, v).bits, bs = bin(bits, f.bits);
          const txt = `${nice(v, 6)}  =  ${bs[0]}.${bs.slice(1, 1 + f.e)}.${bs.slice(1 + f.e)}`;
          g.font = V.font(12, 'mono', 600); const tw = g.measureText(txt).width + 16;
          const tx = V.clamp(X(v) - tw / 2, 4, w - tw - 4);
          V.box(g, tx, ry - 84, tw, 22, { r: 6, fill: C.panel, stroke: C.ink, lw: 1 });
          V.text(g, txt, tx + tw / 2, ry - 73, { size: 12, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 600 });
        }
        // escada do ulp
        const by = h - (w < 640 ? 60 : 46), ty = ry + 70;
        const ds = []; for (let i = 0; i < T.length - 1; i++) ds.push(T[i + 1] - T[i]);
        const dmax = Math.max(...ds);
        V.text(g, 'espaçamento entre vizinhos (ulp)', pad, ty - 10, { size: 11.5, font: 'mono', color: C.muted });
        V.line(g, x0, by, x1, by, { color: V.alpha(C.muted, 0.5), lw: 1 });
        for (let i = 0; i < ds.length; i++) {
          const v = T[i], sub = v < minN;
          let d = ds[i];
          let a = X(T[i]), b = X(T[i + 1]);
          if (!S.sub && sub) { if (v > 0) continue; d = minN; b = X(minN); }
          if (S.log && v === 0) continue;
          const yy = by - (d / dmax) * (by - ty - 6);
          V.line(g, a, yy, b, yy, { color: sub ? (S.sub ? C.c : C.red || C.c) : C.a, lw: 2.2 });
          if (i + 1 < ds.length) { const nxt = ds[i + 1]; const y2 = by - (nxt / dmax) * (by - ty - 6); V.line(g, b, yy, b, y2, { color: V.alpha(C.a, 0.4), lw: 1 }); }
        }
        const nBin = eMax - eMin + 1;
        const nVals = (T.length - 1) * 2 + 1;
        wrap(g, `${f.name}: ${nVals} valores finitos · ${nBin} binades · ${2 ** f.m} valores por binade · máximo ${nice(mx)} · menor normal ${nice(minN)}`, pad, w < 640 ? h - 30 : h - 16, w - 2 * pad, 15, { size: 11.5, color: C.text2 || C.muted });
      }, { static: true });
    }

    /* ================================================================
       Figura 6 — o mesmo número em FP32, BF16 e FP16
       ================================================================ */
    {
      const fig = $('#fig-faixa');
      const S = { t: Math.log10(0.1) };
      const FM = [{ k: 'fp32', name: 'FP32', c: 'b' }, { k: 'bf16', name: 'BF16', c: 'a' }, { k: 'fp16', name: 'FP16', c: 'd' }];
      const ctl = UI.controls(fig);
      const sl = UI.slider(ctl, { label: 'x', min: -9, max: 6, step: 0.001, value: S.t, fmt: (v) => nice(Math.pow(10, v), 5), onInput: (v) => { S.t = v; H.redraw(); } });
      UI.seg(ctl, { label: 'Exemplos', options: [{ v: 0.1, label: '0,1' }, { v: Math.PI, label: 'π' }, { v: 70000, label: '70.000' }, { v: 1e-5, label: '10⁻⁵' }, { v: 65520, label: '65.520' }, { v: 3e-8, label: '3·10⁻⁸' }], value: 0.1, onChange: (v) => sl.set(Math.log10(v)) });
      const H = V.fig(fig, (g, w) => {
        const C = V.C();
        const narrow = w < 640, pad = 16;
        const x = Math.pow(10, S.t);
        const rv = { fp32: r32(x), bf16: rB(x), fp16: rH(x) };
        // --- faixa (log10)
        const lmin = -46, lmax = 40, ax0 = pad + 46, ax1 = w - pad;
        const X = (l) => ax0 + ((l - lmin) / (lmax - lmin)) * (ax1 - ax0);
        V.text(g, 'FAIXA (log₁₀)', pad, 18, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        FM.forEach((fm, i) => {
          const y = 30 + i * 22, col = C[fm.c];
          const ls = Math.log10(fm.k === 'fp32' ? 2 ** -149 : N.minSub(fm.k)), ln = Math.log10(fm.k === 'fp32' ? 2 ** -126 : N.minNormal(fm.k)), lx = Math.log10(fm.k === 'fp32' ? 3.4028234663852886e38 : N.maxFinite(fm.k));
          V.text(g, fm.name, pad, y + 11, { size: 11.5, font: 'mono', color: col, weight: 700 });
          V.box(g, X(ls), y, X(ln) - X(ls), 14, { r: 3, fill: V.alpha(col, 0.28) });
          V.box(g, X(ln), y, X(lx) - X(ln), 14, { r: 3, fill: V.alpha(col, 0.75) });
          g.save(); g.beginPath(); g.rect(X(lx), y, ax1 - X(lx), 14); g.clip();
          for (let xx = X(lx) - 20; xx < ax1; xx += 7) V.line(g, xx, y + 14, xx + 14, y, { color: V.alpha(C.red || C.c, 0.3), lw: 1 });
          g.restore();
        });
        const step = narrow ? 20 : 10;
        for (let l = -40; l <= 40; l += step) { V.line(g, X(l), 96, X(l), 100, { color: C.muted }); V.text(g, '10' + sup(l), X(l), 112, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
        V.line(g, X(S.t), 24, X(S.t), 100, { color: C.ink, lw: 2 });
        V.text(g, 'x', X(S.t) + 5, 26, { size: 12, font: 'mono', color: C.ink, weight: 700 });
        // --- zoom linear
        const zy = 150, zx0 = pad + 46, zx1 = w - pad;
        const eb = Math.floor(Math.log2(x));
        const ub = 2 ** (eb - 7);
        const lo = x - 4.5 * ub, hi = x + 4.5 * ub;
        const Z = (v) => zx0 + ((v - lo) / (hi - lo)) * (zx1 - zx0);
        V.text(g, 'ZOOM EM TORNO DE x · os vizinhos representáveis', pad, zy - 14, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        const rowY = { fp32: zy + 14, bf16: zy + 44, fp16: zy + 74 };
        FM.forEach((fm) => { V.text(g, fm.name, pad, rowY[fm.k] + 4, { size: 11.5, font: 'mono', color: C[fm.c], weight: 700 }); V.line(g, zx0, rowY[fm.k], zx1, rowY[fm.k], { color: V.alpha(C.muted, 0.3), lw: 1 }); });
        V.line(g, zx0, rowY.fp32, zx1, rowY.fp32, { color: V.alpha(C.b, 0.9), lw: 3 });
        V.text(g, narrow ? 'contínuo: 65.536 FP32 por tracinho do BF16' : 'contínuo nesta escala: 65.536 valores FP32 entre dois tracinhos do BF16', (zx0 + zx1) / 2, rowY.fp32 - 8, { size: 11, color: C.b, align: 'center', maxW: zx1 - zx0 });
        const hb = u32(rv.bf16 === 0 ? x : rv.bf16) >>> 16;
        for (let k = -7; k <= 7; k++) { const v = f32(((hb + k) * 65536) >>> 0); if (v >= lo && v <= hi) V.line(g, Z(v), rowY.bf16 - 9, Z(v), rowY.bf16 + 9, { color: C.a, lw: 2 }); }
        const fpOK = Number.isFinite(rv.fp16);
        if (fpOK) {
          const hb16 = N.encode('fp16', x).bits;
          let drawn = 0;
          for (let k = -80; k <= 80; k++) { const bb = hb16 + k; if (bb < 0 || bb >= 0x7C00) continue; const v = N.decode('fp16', bb); if (v >= lo && v <= hi) { V.line(g, Z(v), rowY.fp16 - 7, Z(v), rowY.fp16 + 7, { color: C.d, lw: 1.3 }); drawn++; } }
          if (!drawn) V.text(g, 'vizinhos FP16 fora da janela (passo maior que a janela)', (zx0 + zx1) / 2, rowY.fp16 + 4, { size: 11, color: C.d, align: 'center' });
        } else V.text(g, 'FP16: overflow → +∞ (nenhum valor finito aqui)', (zx0 + zx1) / 2, rowY.fp16 + 4, { size: 11.5, color: C.red || C.c, align: 'center', weight: 600 });
        V.line(g, Z(x), zy + 2, Z(x), rowY.fp16 + 14, { color: C.ink, lw: 1.4, dash: [4, 3] });
        FM.forEach((fm) => { const v = rv[fm.k]; if (Number.isFinite(v) && v >= lo && v <= hi) V.dot(g, Z(v), rowY[fm.k], 5, C[fm.c]); });
        // --- tabela
        let ty = rowY.fp16 + 42;
        const cx = narrow ? [pad, pad + 56, pad + 190] : [pad, pad + 80, pad + 290, pad + 430];
        V.text(g, 'formato', cx[0], ty, { size: 11, font: 'mono', color: C.muted }); V.text(g, 'valor guardado', cx[1], ty, { size: 11, font: 'mono', color: C.muted }); V.text(g, narrow ? 'erro rel. · estado' : 'erro relativo', cx[2], ty, { size: 11, font: 'mono', color: C.muted });
        if (!narrow) V.text(g, 'estado', cx[3], ty, { size: 11, font: 'mono', color: C.muted });
        ty += 22;
        FM.forEach((fm) => {
          const v = rv[fm.k], err = Number.isFinite(v) ? Math.abs(v - x) / x : Infinity;
          const mn = fm.k === 'fp32' ? 2 ** -126 : N.minNormal(fm.k);
          let st = 'normal', sc = C.a;
          if (!Number.isFinite(v)) { st = 'overflow → +∞'; sc = C.red || C.c; }
          else if (v === 0) { st = 'underflow → 0'; sc = C.red || C.c; }
          else if (x < mn) { st = 'subnormal'; sc = C.d; }
          V.text(g, fm.name, cx[0], ty, { size: 12.5, font: 'mono', color: C[fm.c], weight: 700 });
          V.text(g, nice(v, 9), cx[1], ty, { size: 12.5, font: 'mono', color: C.ink, maxW: cx[2] - cx[1] - 8 });
          if (narrow) V.text(g, (Number.isFinite(err) ? pct(err) : '∞') + ' · ' + st, cx[2], ty, { size: 12, font: 'mono', color: sc, maxW: w - cx[2] - 8 });
          else { V.text(g, Number.isFinite(err) ? pct(err) : '∞', cx[2], ty, { size: 12.5, font: 'mono', color: C.ink }); V.text(g, st, cx[3], ty, { size: 12.5, font: 'mono', color: sc, weight: 600 }); }
          ty += 22;
        });
        V.text(g, `x = ${nice(x, 9)}`, pad, ty + 4, { size: 12, font: 'mono', color: C.muted });
        fitH(fig, ty + 16);
      }, { static: true });
    }

    /* ================================================================
       Figura 7 — arredondamento ao par, em câmera lenta (stepper)
       ================================================================ */
    {
      const fig = $('#fig-rne');
      const EX = [
        { v: 'pi', label: 'π', u: 0x40490FDB },
        { v: 'dec', label: '0,1', u: 0x3DCCCCCD },
        { v: 'tie0', label: '1,00390625', u: 0x3F808000 },
        { v: 'tie1', label: '1,01171875', u: 0x3F818000 },
        { v: 'carry', label: '1,998046875', u: 0x3FFFC000 },
      ];
      const S = { ex: EX[0] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Exemplo', options: EX.map((e) => ({ v: e.v, label: e.label })), value: 'pi', onChange: (v) => { S.ex = EX.find((e) => e.v === v); st.go(st.i); } });
      const st = UI.stepper(fig, [
        { t: 'O número em FP32', d: 'Os 32 bits exatos de \\(x\\): 1 de sinal, 8 de expoente e 23 de mantissa. O destino é o BF16, que guarda só os 16 bits de cima.' },
        { t: 'Cortar: o que fica e o que vai embora', d: 'Os 16 bits de cima formam o vizinho BF16 de baixo, \\(a\\). Somar 1 a eles dá o vizinho de cima, \\(b\\). Os 16 bits de baixo são o <b>resto</b>: onde \\(x\\) está entre \\(a\\) e \\(b\\), em frações de ulp.' },
        { t: 'Pesar o resto contra a metade', d: 'O primeiro bit descartado (o bit de guarda) diz se o resto passa de meio ulp, e os demais (sticky) desempatam. Resto &lt; <code>0x8000</code>: \\(a\\) está mais perto. Resto &gt; <code>0x8000</code>: \\(b\\) está mais perto.' },
        { t: 'Empate? Vence o par', d: 'Se o resto é exatamente <code>0x8000</code>, os dois vizinhos estão à mesma distância. O RNE escolhe o que termina em bit 0. Metade dos empates sobe e metade desce, então o erro médio é zero.' },
        { t: 'Resultado e erro', d: 'O BF16 escolhido e o erro, que nunca passa de meio ulp, ou \\(2^{-8}\\) em termos relativos. Converter FP32 → BF16 é só isso, e por isso custa tão pouco.' },
      ]);
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const narrow = w < 640, pad = 16, i = st.i, el = st.elapsed();
        const u = S.ex.u >>> 0, hi = u >>> 16, lo = u & 0xFFFF;
        const x = f32(u), a = f32(hi * 65536), b = f32(((hi + 1) * 65536) >>> 0);
        const tie = lo === 0x8000;
        const up = lo > 0x8000 || (tie && (hi & 1) === 1);
        const rb = up ? hi + 1 : hi, r = up ? b : a;
        const frac = lo / 65536;
        // bits
        const cw = narrow ? Math.min(20, (w - 2 * pad - 12) / 16) : Math.min(24, (w - 2 * pad - 40) / 32);
        const chh = 26, y0 = 34;
        const pos = (k) => { if (narrow) { const row = k >= 16 ? 1 : 0, c = k % 16; return [pad + c * cw + Math.floor(c / 4) * 3, y0 + row * (chh + 44)]; } return [pad + k * cw + Math.floor(k / 4) * 3 + (k >= 16 ? 14 : 0), y0]; };
        for (let k = 0; k < 32; k++) {
          const bit = 31 - k, on = (u >>> bit) & 1;
          const col = bit === 31 ? C.c : bit >= 23 ? C.d : C.a;
          const dropped = bit < 16 && i >= 1;
          const [xx, yy] = pos(k);
          V.box(g, xx, yy, cw - 2, chh, { r: 4, fill: on ? V.alpha(col, dropped ? 0.3 : 0.88) : V.alpha(col, 0.1), stroke: V.alpha(col, dropped ? 0.3 : 0.6), lw: 1 });
          V.text(g, on, xx + (cw - 2) / 2, yy + chh / 2 + 1, { size: Math.min(12.5, cw * 0.6), font: 'mono', align: 'center', base: 'middle', color: on && !dropped ? onFill(C) : C.muted, weight: 600 });
          if (i >= 2 && bit === 15) { g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, xx - 2, yy - 2, cw + 2, chh + 4, 5); g.stroke(); }
          if (i >= 3 && bit === 16) { g.strokeStyle = C.c; g.lineWidth = 2.5; V.rr(g, xx - 2, yy - 2, cw + 2, chh + 4, 5); g.stroke(); }
        }
        const [ka] = pos(0), [kb] = pos(15), [da, dY] = pos(16), [db] = pos(31);
        if (i >= 1) {
          bracket(g, ka, kb + cw - 2, y0 + chh + 6, C.a);
          V.text(g, `fica: BF16 = 0x${hi.toString(16).toUpperCase().padStart(4, '0')}`, (ka + kb + cw) / 2, y0 + chh + 26, { size: 11.5, font: 'mono', color: C.a, align: 'center', weight: 600 });
          bracket(g, da, db + cw - 2, dY + chh + 6, C.muted);
          V.text(g, `vai embora: resto = 0x${lo.toString(16).toUpperCase().padStart(4, '0')}`, (da + db + cw) / 2, dY + chh + 26, { size: 11.5, font: 'mono', color: C.muted, align: 'center', weight: 600 });
        } else {
          const E = (u >>> 23) & 255, M = u & 0x7FFFFF;
          let yy = y0 + (narrow ? 2 * chh + 64 : chh + 30);
          V.text(g, `x = ${nice(x, 10)}   (FP32 0x${u.toString(16).toUpperCase().padStart(8, '0')})`, pad, yy, { size: 13, font: 'mono', color: C.ink, weight: 700, maxW: w - 2 * pad }); yy += 22;
          V.text(g, `S = ${u >>> 31} · E = ${E} → 2${sup(E - 127)} · M = ${bi(M)} → 1 + M/2²³ = ${nice(1 + M / 2 ** 23, 9)}`, pad, yy, { size: 12, font: 'mono', color: C.muted, maxW: w - 2 * pad }); yy += 34;
          V.text(g, 'DESTINO: BF16 (1 + 8 + 7 = 16 bits)', pad, yy, { size: 11, font: 'mono', color: C.a, weight: 600 }); yy += 8;
          const bw16 = Math.min(22, (w - 2 * pad) / 18);
          for (let k = 0; k < 16; k++) { const col = k === 0 ? C.c : k <= 8 ? C.d : C.a; g.setLineDash([3, 3]); V.box(g, pad + k * bw16 + (k >= 1 ? 4 : 0) + (k >= 9 ? 4 : 0), yy, bw16 - 3, 24, { r: 4, stroke: V.alpha(col, 0.8), lw: 1.2 }); g.setLineDash([]); }
          V.text(g, 'só cabem os 16 bits de cima: quais vizinhos BF16 cercam x?', pad, yy + 44, { size: 12, color: C.text2 || C.muted, maxW: w - 2 * pad });
        }
        if (i === 1) {
          let yy = narrow ? y0 + 2 * chh + 100 : y0 + chh + 72;
          const hx4 = (v) => '0x' + v.toString(16).toUpperCase().padStart(4, '0');
          V.text(g, `a = ${hx4(hi)} = ${nice(a, 9)}   (os 16 bits de cima, como estão)`, pad, yy, { size: 12.5, font: 'mono', color: C.a, weight: 600, maxW: w - 2 * pad }); yy += 22;
          V.text(g, `b = ${hx4(hi + 1)} = ${nice(b, 9)}   (a + 1 ulp)`, pad, yy, { size: 12.5, font: 'mono', color: C.a, weight: 600, maxW: w - 2 * pad }); yy += 22;
          V.text(g, `resto = ${hx4(lo)} = ${bi(lo)} / 65.536 = ${frac.toFixed(4).replace('.', ',')} ulp acima de a`, pad, yy, { size: 12.5, font: 'mono', color: C.ink, weight: 600, maxW: w - 2 * pad });
        }
        if (i === 2 && !narrow) {
          const [gx] = pos(16);
          V.text(g, 'guarda', gx + cw / 2, y0 - 8, { size: 11, font: 'mono', color: C.d, align: 'center' });
          V.text(g, 'sticky →', gx + cw + 22, y0 - 8, { size: 11, font: 'mono', color: C.muted });
        }
        if (i >= 3) { const [lx] = pos(15); V.text(g, `último: ${hi & 1}`, lx + cw / 2, y0 - 8, { size: 11, font: 'mono', color: C.c, align: 'center', weight: 600 }); }
        // área central
        const cy = narrow ? y0 + 2 * chh + 100 : y0 + chh + 70;
        if (i === 2 || i === 3) {
          // balança
          const bx = w / 2, by = cy + 44, arm = Math.min(150, w * 0.3);
          const target = lo < 0x8000 ? 0.22 : lo > 0x8000 ? -0.22 : 0;
          const th = i === 2 ? target * V.ease.out(Math.min(1, el / 0.9)) + Math.sin(t * 6) * 0.015 * Math.max(0, 1 - el) : target;
          V.line(g, bx, by, bx, by + 60, { color: C.muted, lw: 3 });
          g.beginPath(); g.moveTo(bx - 22, by + 64); g.lineTo(bx + 22, by + 64); g.lineTo(bx, by + 52); g.closePath(); g.fillStyle = V.alpha(C.muted, 0.6); g.fill();
          const lx = bx - Math.cos(th) * arm, lyy = by - Math.sin(th) * arm, rx = bx + Math.cos(th) * arm, ryy = by + Math.sin(th) * arm;
          V.line(g, lx, lyy, rx, ryy, { color: C.ink, lw: 3 });
          V.dot(g, bx, by, 5, C.ink);
          const pan = (px, py, label, val, col) => {
            V.line(g, px, py, px - 22, py + 26, { color: V.alpha(C.ink, 0.5) }); V.line(g, px, py, px + 22, py + 26, { color: V.alpha(C.ink, 0.5) });
            V.box(g, px - 44, py + 26, 88, 30, { r: 8, fill: V.alpha(col, 0.2), stroke: col, lw: 1.4 });
            V.text(g, val, px, py + 42, { size: 12, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 700 });
            V.text(g, label, px, py + 72, { size: 11, font: 'mono', align: 'center', color: col });
          };
          pan(lx, lyy, `resto = ${(frac).toFixed(3).replace('.', ',')} ulp`, '0x' + lo.toString(16).toUpperCase().padStart(4, '0'), C.d);
          pan(rx, ryy, 'meio ulp = 0,5 ulp', '0x8000', C.b);
          const verdict = lo < 0x8000 ? 'resto < metade → fica com a (desce)' : lo > 0x8000 ? 'resto > metade → vai para b (sobe)' : 'empate exato!';
          V.text(g, verdict, bx, cy + 2, { size: 13, align: 'center', color: tie ? C.c : C.ink, weight: 700, maxW: w - 2 * pad });
          if (i === 3) V.text(g, tie ? `a = 0x${hi.toString(16).toUpperCase()} termina em ${hi & 1}, b = 0x${(hi + 1).toString(16).toUpperCase()} termina em ${(hi + 1) & 1} → vence ${up ? 'b' : 'a'} (o par)` : 'não é empate: a regra do par nem entra em jogo', bx, by + 122, { size: 12, align: 'center', color: tie ? C.c : C.muted, weight: 600, maxW: w - 2 * pad });
        }
        // régua dos vizinhos
        {
          const ly = h - 58, ax0 = pad + 40, ax1 = w - pad - 40;
          const Xr = (fq) => ax0 + fq * (ax1 - ax0);
          V.line(g, ax0, ly, ax1, ly, { color: V.alpha(C.muted, 0.7), lw: 1.4 });
          V.line(g, Xr(0), ly - 12, Xr(0), ly + 12, { color: C.a, lw: 3 });
          V.line(g, Xr(1), ly - 12, Xr(1), ly + 12, { color: C.a, lw: 3 });
          V.line(g, Xr(0.5), ly - 9, Xr(0.5), ly + 9, { color: C.muted, lw: 1.5, dash: [3, 3] });
          V.text(g, `a = ${nice(a, 8)}`, Xr(0), ly + 28, { size: 11.5, font: 'mono', color: C.a, align: 'center' });
          V.text(g, `b = ${nice(b, 8)}`, Xr(1), ly + 28, { size: 11.5, font: 'mono', color: C.a, align: 'center' });
          V.text(g, 'meio', Xr(0.5), ly + 24, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          let fx = frac;
          if (i >= 4) fx = V.lerp(frac, up ? 1 : 0, V.ease.inOut(Math.min(1, el / 1.1)));
          V.glow(g, Xr(fx), ly, 16, C.c);
          V.dot(g, Xr(fx), ly, 5.5, C.c);
          V.text(g, i >= 4 && el > 1.1 ? 'fl(x)' : 'x', Xr(fx), ly - 16, { size: 12, font: 'mono', color: C.c, align: 'center', weight: 700 });
          if (i >= 4) {
            const err = r - x;
            const txt = `BF16 = 0x${rb.toString(16).toUpperCase().padStart(4, '0')} = ${nice(r, 10)}   ·   erro ${nice(err, 3)} (${Math.abs(err / (b - a)).toFixed(3).replace('.', ',')} ulp; ${pct(Math.abs(err / x))})`;
            const nlr = wrap(g, txt, pad, cy + (narrow ? 0 : 10), w - 2 * pad, 18, { size: 12.5, font: 'mono', color: C.ink, weight: 600 });
            if (el > 1.1) {
              const ry0 = cy + (narrow ? 0 : 10) + nlr * 18 + (S.ex.v === 'carry' ? 34 : 14);
              V.text(g, 'BF16 GUARDADO', pad, ry0, { size: 11, font: 'mono', color: C.a, weight: 600 });
              const bw16 = Math.min(24, (w - 2 * pad - 10) / 16.6);
              for (let k = 0; k < 16; k++) {
                const bit = 15 - k, on = (rb >>> bit) & 1, col = bit === 15 ? C.c : bit >= 7 ? C.d : C.a;
                const xx = pad + k * bw16 + (k >= 1 ? 4 : 0) + (k >= 9 ? 4 : 0);
                V.box(g, xx, ry0 + 8, bw16 - 3, 24, { r: 4, fill: on ? V.alpha(col, 0.88) : V.alpha(col, 0.12), stroke: V.alpha(col, 0.6), lw: 1 });
                if (bw16 >= 15) V.text(g, on, xx + (bw16 - 3) / 2, ry0 + 21, { size: 11, font: 'mono', align: 'center', base: 'middle', color: on ? onFill(C) : C.muted, weight: 600 });
              }
              const dl = (r - x) / x;
              V.text(g, `fl(x) = x·(1 + δ) com δ = ${sci(dl, 2)}   ·   |δ| ≤ u = 2⁻⁸ ≈ 0,39%`, pad, ry0 + 54, { size: 12, font: 'mono', color: C.muted, maxW: w - 2 * pad });
            }
            if (S.ex.v === 'carry' && el > 1.1) V.text(g, 'o vai-um atravessou a mantissa e incrementou o expoente: 0x3FFF + 1 = 0x4000 = 2,0', pad, cy + (narrow ? 44 : 34), { size: 12, color: C.c, maxW: w - 2 * pad });
          }
        }
        // no celular, os passos 1 e 2 não têm a balança: palco mais baixo, sem faixa vazia
        fitH(fig, narrow ? (i <= 1 ? 380 : 450) : 420);
      });
    }

    /* ================================================================
       Figura 8 — a corrida dos acumuladores
       ================================================================ */
    {
      const fig = $('#fig-soma');
      const S = { K: 4096, dist: 'pos', seed: 1, R: null };
      const NORD = 48;
      const tree = (arr, rnd) => { let a = arr.slice(); while (a.length > 1) { const nx = []; for (let i = 0; i < a.length; i += 2) nx.push(i + 1 < a.length ? rnd(a[i] + a[i + 1]) : a[i]); a = nx; } return a[0]; };
      const compute = () => {
        const K = S.K, rg = V.rng(1000 * S.seed + K + (S.dist === 'pos' ? 7 : 0));
        const p = new Array(K);
        for (let i = 0; i < K; i++) { let x = rB(rg.normal()), y = rB(rg.normal()); if (S.dist === 'pos') { x = Math.abs(x); y = Math.abs(y); } p[i] = x * y; }
        let ex = 0; for (let i = 0; i < K; i++) ex += p[i];
        const stride = Math.max(1, Math.floor(K / 220));
        const tr = { ex: [], f32: [], f16: [], b16: [], i: [] };
        let sE = 0, sF = 0, sH = 0, sB = 0;
        for (let i = 0; i < K; i++) {
          sE += p[i]; sF = r32(sF + p[i]); sH = rH(sH + p[i]); sB = rB(sB + p[i]);
          if (i % stride === 0 || i === K - 1) { tr.i.push(i + 1); tr.ex.push(sE); tr.f32.push(sF); tr.f16.push(sH); tr.b16.push(sB); }
        }
        const fin = { b16: [sB, tree(p, rB)], f16: [sH, tree(p, rH)], f32: [sF, tree(p, r32)] };
        const oF = [], oB = [];
        const idx = Array.from({ length: K }, (_, i) => i);
        for (let j = 0; j < NORD; j++) {
          const rr = V.rng(77 + j * 31 + S.seed * 1009);
          for (let i = K - 1; i > 0; i--) { const q = Math.floor(rr() * (i + 1)); const tmp = idx[i]; idx[i] = idx[q]; idx[q] = tmp; }
          let a = 0, c = 0; for (let i = 0; i < K; i++) { const v = p[idx[i]]; a = r32(a + v); c = rB(c + v); }
          oF.push(a); oB.push(c);
        }
        S.R = { tr, fin, oF, oB, ex, dF: new Set(oF).size, dB: new Set(oB).size };
        rd.set(`${S.R.dF} de ${NORD}`);
      };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'K (termos)', options: [64, 512, 4096], value: 4096, onChange: (v) => { S.K = v; compute(); H.redraw(); } });
      UI.seg(ctl, { label: 'Dados', options: [{ v: 'pos', label: 'todos positivos' }, { v: 'zero', label: 'média zero' }], value: 'pos', onChange: (v) => { S.dist = v; compute(); H.redraw(); } });
      UI.button(ctl, 'novos números', () => { S.seed++; compute(); H.redraw(); });
      const rd = UI.readout(ctl, 'resultados FP32 distintos', '');
      const LEG = [['ex', 'exato', 'ink'], ['f32', 'FP32', 'b'], ['f16', 'FP16', 'd'], ['b16', 'BF16', 'a']];
      compute();
      const H = V.fig(fig, (g, w) => {
        const C = V.C();
        const R = S.R, narrow = w < 720;
        const A = narrow ? { x: 52, y: 48, w: w - 68, h: 170 } : { x: 62, y: 52, w: w * 0.55 - 80, h: 308 };
        const Bp = narrow ? { x: 70, y: A.y + A.h + 58, w: w - 90, h: 118 } : { x: w * 0.55 + 70, y: 30, w: w * 0.45 - 90, h: 128 };
        const Cp = narrow ? { x: 16, y: Bp.y + Bp.h + 50, w: w - 32, h: 118 } : { x: w * 0.55 + 10, y: Bp.y + Bp.h + 66, w: w * 0.45 - 30, h: 136 };
        // painel A
        const all = [].concat(R.tr.ex, R.tr.f32, R.tr.f16, R.tr.b16, [0]);
        const ymn = Math.min(...all), ymx = Math.max(...all), yr = ymx - ymn || 1;
        const XA = (i) => A.x + (i / S.K) * A.w, YA = (v) => A.y + A.h - ((v - ymn) / yr) * A.h;
        V.text(g, 'SOMA PARCIAL (ordem sequencial)', A.x - 40, A.y - 34, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        for (let k = 0; k <= 4; k++) { const v = ymn + (yr * k) / 4; V.line(g, A.x, YA(v), A.x + A.w, YA(v), { color: C.grid, lw: 1 }); V.text(g, Math.abs(v) >= 100 ? bi(v) : nice(v, 3), A.x - 6, YA(v) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }); }
        V.text(g, `termos somados → ${bi(S.K)}`, A.x + A.w, A.y + A.h + 16, { size: 11, font: 'mono', color: C.muted, align: 'right' });
        const plot = (arr, col, lw, dash) => { g.beginPath(); arr.forEach((v, k) => { const px = XA(R.tr.i[k]), py = YA(v); if (k) g.lineTo(px, py); else g.moveTo(px, py); }); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
        plot(R.tr.b16, C.a, 2.4); plot(R.tr.f16, C.d, 2); plot(R.tr.f32, C.b, 2.4); plot(R.tr.ex, C.ink, 1.6, [5, 4]);
        // legenda fora da área do gráfico, numa linha entre o título e o plot
        { const step = Math.min(84, (A.w + 40) / 4); LEG.forEach((L, k) => { const lx = A.x - 40 + k * step, ly = A.y - 12; V.line(g, lx, ly - 4, lx + 18, ly - 4, { color: C[L[2]], lw: 2.4, dash: L[0] === 'ex' ? [4, 3] : null }); V.text(g, L[1], lx + 24, ly, { size: 11.5, font: 'mono', color: C[L[2]], weight: 600 }); }); }
        // painel B: erro final
        V.text(g, 'ERRO RELATIVO FINAL', Bp.x - (narrow ? 54 : 60), Bp.y - 12, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const lg = (e) => (e <= 0 ? null : Math.log10(e));
        const L0 = -9, L1 = 0, BW = Bp.w - 112, XB = (l) => Bp.x + ((V.clamp(l, L0, L1) - L0) / (L1 - L0)) * BW;
        [-9, -6, -3, 0].forEach((l) => { V.line(g, XB(l), Bp.y, XB(l), Bp.y + Bp.h - 6, { color: C.grid }); V.text(g, '10' + sup(l), XB(l), Bp.y + Bp.h + 14, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
        [['b16', 'BF16', 'a'], ['f16', 'FP16', 'd'], ['f32', 'FP32', 'b']].forEach(([k, name, c], j) => {
          const y = Bp.y + 6 + j * (Bp.h / 3);
          V.text(g, name, Bp.x - 8, y + 14, { size: 11.5, font: 'mono', color: C[c], align: 'right', weight: 700 });
          ['seq', 'árvore'].forEach((lab, q) => {
            const e = Math.abs(R.fin[k][q] - R.ex) / Math.abs(R.ex || 1);
            const l = lg(e), yy = y + q * 13;
            if (l == null) { V.text(g, `${lab}: exato`, Bp.x + BW + 8, yy + 9, { size: 11, font: 'mono', color: C[c] }); return; }
            V.box(g, Bp.x, yy, Math.max(2, XB(l) - Bp.x), 10, { r: 3, fill: V.alpha(C[c], q ? 0.4 : 0.85) });
            V.text(g, `${lab} ${sci(e, 1)}`, Bp.x + BW + 8, yy + 9, { size: 11, font: 'mono', color: C.ink });
          });
        });
        // painel C: 48 ordens
        V.text(g, `${NORD} ORDENS ALEATÓRIAS · cada ponto é um resultado`, Cp.x, Cp.y - 12, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: Cp.w });
        [['oF', 'FP32', 'b', R.dF], ['oB', 'BF16', 'a', R.dB]].forEach(([k, name, c, dn], j) => {
          const y = Cp.y + 16 + j * (Cp.h / 2);
          const devs = R[k].map((v) => (v - R.ex) / Math.abs(R.ex || 1));
          const span = Math.max(1e-12, ...devs.map(Math.abs)) * 1.15;
          const sx0 = Cp.x + 50, sx1 = Cp.x + Cp.w - 8;
          const XC = (d) => (sx0 + sx1) / 2 + (d / span) * ((sx1 - sx0) / 2);
          V.text(g, name, Cp.x, y + 4, { size: 11.5, font: 'mono', color: C[c], weight: 700 });
          V.line(g, sx0, y, sx1, y, { color: V.alpha(C.muted, 0.4) });
          V.line(g, XC(0), y - 12, XC(0), y + 12, { color: C.ink, lw: 1.4, dash: [3, 2] });
          const rg = V.rng(5 + j);
          devs.forEach((d) => V.dot(g, XC(d), y + (rg() - 0.5) * 14, 3, V.alpha(C[c], 0.75)));
          V.text(g, `${dn} resultados distintos · ±${sci(span / 1.15, 1)} relativo`, sx0, y + 26, { size: 11, font: 'mono', color: C.muted, maxW: sx1 - sx0 });
        });
        fitH(fig, narrow ? Cp.y + Cp.h + 12 : 420);
      }, { static: true });
    }

    /* ================================================================
       Figura 9 — a cópia BF16 da KDA, passo a passo
       ================================================================ */
    {
      const fig = $('#fig-kda');
      const E4 = N.table('e4m3').filter((o) => o.v >= N.minNormal('e4m3'));
      const SC = []; for (let bb = 0x3800; bb < 0x3980; bb++) SC.push({ bits: bb, v: N.decode('bf16', bb) });
      const S = { tp: 2, chunk: 512, qi: Math.max(0, E4.findIndex((o) => o.v === 208)), si: 0x38D3 - 0x3800 };
      const shape = () => (S.tp === 2 ? [12576, 4096] : [8726, 4096]);
      // matriz ilustrativa (6×8): pesos sintéticos, escala por linha, códigos E4M3
      const MR = 6, MC = 8, rg = V.rng(41);
      const Wm = Array.from({ length: MR }, (_, i) => Array.from({ length: MC }, () => rg.normal() * 0.02 * (1 + i * 0.35)));
      const sc = Wm.map((row) => rB(Math.max(...row.map(Math.abs)) / 448));
      const Q = Wm.map((row, i) => row.map((v) => N.round('e4m3', Math.max(-448, Math.min(448, v / sc[i])))));
      const Wl = Q.map((row, i) => row.map((q) => rB(q * sc[i])));
      // redução fatiada (passo 5)
      const rp = V.rng(3), PR = []; for (let i = 0; i < 4096; i++) PR.push(rB(rp.normal()) * rB(rp.normal()));
      const chunkSum = (c) => { let tot = 0; for (let s0 = 0; s0 < 4096; s0 += c) { let a = 0; for (let i = s0; i < Math.min(4096, s0 + c); i++) a = r32(a + PR[i]); tot = r32(tot + a); } return tot; };
      const RED = [512, 2048, 4096].map((c) => ({ c, v: chunkSum(c) }));
      const hex32 = (x) => '0x' + u32(x).toString(16).toUpperCase().padStart(8, '0');
      const hex16 = (x) => '0x' + (u32(x) >>> 16).toString(16).toUpperCase().padStart(4, '0');
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'TP', options: [{ v: 2, label: 'TP2 · 12.576×4.096' }, { v: 3, label: 'TP3 · 8.726×4.096' }], value: 2, onChange: (v) => { S.tp = v; } });
      UI.seg(ctl, { label: 'Linhas por bloco', options: [{ v: 128, label: '128' }, { v: 512, label: '512' }, { v: 2048, label: '2.048' }, { v: 0, label: 'sem blocos' }], value: 512, onChange: (v) => { S.chunk = v; } });
      UI.slider(ctl, { label: 'código E4M3', min: 0, max: E4.length - 1, step: 1, value: S.qi, fmt: (i) => nice(E4[i].v, 5), onInput: (v) => { S.qi = v; } });
      UI.slider(ctl, { label: 'escala BF16', min: 0, max: SC.length - 1, step: 1, value: S.si, fmt: (i) => sci(SC[i].v, 3), onInput: (v) => { S.si = v; } });
      const rPk = UI.readout(ctl, 'temporário FP32', ''), rCp = UI.readout(ctl, 'cópia BF16/camada', '');
      const st = UI.stepper(fig, [
        { t: 'O peso lógico: código × escala', d: 'Com <code>GLM53_DENSE_FP8</code>, cada linha \\(r\\) da projeção vira códigos E4M3 \\(q_{rj}\\) e uma escala BF16 \\(s_r=\\max_j|W_{rj}|/448\\). O peso que o modelo usa de fato é \\(q_{rj}\\cdot s_r\\), e é isso que o Marlin calcula nos registradores.' },
        { t: 'O produto das mantissas cabe em 24 bits', d: 'O E4M3 tem 4 bits significativos e o BF16, 8. A multiplicação longa produz no máximo 12 bits, e o FP32 tem 24. O produto em FP32 é <b>exato</b>, e os expoentes só se somam.' },
        { t: 'Um único arredondamento', d: 'O resultado exato de até 12 bits vira BF16 (8 bits) com um único RNE. Se a escala tivesse 24 bits, o produto poderia ter até 28, e então o próprio FP32 poderia arredondar: seriam até dois arredondamentos.' },
        { t: 'A esteira de 512 linhas', d: '<code>KDA_BF16_LARGE_M_CHUNK_ROWS = 512</code>: cada volta converte 512 linhas para FP32, multiplica pela escala e grava em BF16. Cada temporário tem 512 × 4.096 × 4 B = 8 MiB em vez de 196,5 MiB de uma vez (com os dois temporários da volta vivos, o pico é ~16 MiB).' },
        { t: 'Blocos diferentes, bits idênticos', d: 'Cada elemento só depende do seu código e da sua escala. Sem soma não há ordem que mude, e qualquer tamanho de bloco dá os mesmos bits. Numa <b>redução</b>, ao contrário, fatiar muda o resultado.' },
        { t: 'Quem usa a cópia: o despacho por M', d: 'Em tempo de execução, chamadas com \\(M \\le 512\\) linhas de ativação vão para o Marlin FP8 e chamadas com \\(M &gt; 512\\) vão para o cuBLAS com a cópia BF16. O decode medido nunca passou de \\(M=220\\), e os chunks de prefill caem em \\(M \\ge 768\\).' },
      ]);
      const cellRow = (g, x, y, bitsStr, cw, col, C, o = {}) => {
        for (let k = 0; k < bitsStr.length; k++) {
          const on = bitsStr[k] === '1', cc = o.colAt ? o.colAt(k) : col;
          V.box(g, x + k * cw, y, cw - 2, cw + 2, { r: 3, fill: on ? V.alpha(cc, o.dim ? 0.3 : 0.85) : V.alpha(cc, 0.1), stroke: V.alpha(cc, 0.55), lw: 1 });
          if (cw >= 15) V.text(g, bitsStr[k], x + k * cw + (cw - 2) / 2, y + (cw + 2) / 2 + 1, { size: Math.min(12, Math.max(11, cw * 0.62)), font: 'mono', align: 'center', base: 'middle', color: on && !o.dim ? onFill(C) : C.muted, weight: 600 });
        }
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const [NR, K] = shape();
        const rows = S.chunk || NR;
        const peak = rows * K * 4, copy = NR * K * 2;
        rPk.set(vb(peak)); rCp.set(vb(copy));
        const narrow = w < 700, pad = 16, i = st.i, el = st.elapsed();
        V.text(g, `in_proj_qkvbfg_a local (TP${S.tp}): [${bi(NR)} × ${bi(K)}]`, pad, 22, { size: 12, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        const q = E4[S.qi].v, s = SC[S.si].v;
        const qe = Math.floor(Math.log2(q)), qb = Math.round((q / 2 ** qe) * 8);
        const se = Math.floor(Math.log2(s)), sbb = Math.round((s / 2 ** se) * 128);
        const Pint = qb * sbb, Pbits = Pint.toString(2);
        if (i === 0) {
          const cell = narrow ? Math.min(34, (w - 2 * pad - 30) / (MC + 1.6)) : Math.min(40, (w - 2 * pad - 120) / (2 * MC + 3));
          const my = 50, rowHi = Math.floor(t * 0.8) % MR;
          const drawM = (M, x, y, fmt, cm) => {
            for (let r = 0; r < MR; r++) for (let c = 0; c < MC; c++) {
              const v = M[r][c], mx = Math.max(...M[r].map(Math.abs)) || 1;
              V.box(g, x + c * cell, y + r * cell, cell - 2, cell - 2, { r: 3, fill: V.cmap(v / mx, 'div'), stroke: r === rowHi ? C.c : null, lw: 1.5 });
              if (cell >= 30) V.text(g, fmt(v), x + c * cell + (cell - 2) / 2, y + r * cell + cell / 2, { size: 11, font: 'mono', align: 'center', base: 'middle', color: Math.abs(v / mx) > 0.55 ? (C.dark ? '#07140d' : '#fff') : C.ink, maxW: cell - 4 });
            }
          };
          const qx = pad + 6;
          drawM(Q, qx, my + 14, (v) => nice(v, 3), 'div');
          V.text(g, 'códigos E4M3 q', qx, my + 6, { size: 11.5, font: 'mono', color: C.d, weight: 600 });
          const sx = qx + MC * cell + 12;
          for (let r = 0; r < MR; r++) { V.box(g, sx, my + 14 + r * cell, cell * 1.5, cell - 2, { r: 3, fill: V.alpha(C.a, r === rowHi ? 0.45 : 0.2), stroke: r === rowHi ? C.c : V.alpha(C.a, 0.5), lw: 1.2 }); V.text(g, sci(sc[r], 1), sx + cell * 0.75, my + 14 + r * cell + cell / 2, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, maxW: cell * 1.5 - 4 }); }
          V.text(g, 's (BF16)', sx, my + 6, { size: 11.5, font: 'mono', color: C.a, weight: 600 });
          if (!narrow) {
            const wx = sx + cell * 1.5 + 34;
            V.text(g, '=', wx - 20, my + 14 + (MR * cell) / 2, { size: 20, color: C.muted, align: 'center', base: 'middle' });
            drawM(Wl, wx, my + 14, (v) => nice(v * 1000, 3), 'div');
            V.text(g, 'peso lógico q·s (×10⁻³)', wx, my + 6, { size: 11.5, font: 'mono', color: C.ink, weight: 600 });
          }
          const ty = my + 14 + MR * cell + 26;
          const r0 = rowHi;
          wrap(g, `linha ${r0}: sᵣ = amax/448 = ${sci(Math.max(...Wm[r0].map(Math.abs)), 3)}/448 → BF16 ${sci(sc[r0], 4)}. Cada peso da linha é q × sᵣ. Valores ilustrativos: na receita são ${bi(NR)} linhas × ${bi(K)} colunas por rank.`, pad, ty, w - 2 * pad, 17, { size: 12, color: C.text2 || C.muted });
          fitH(fig, Math.max(ty + 60, 300));
        } else if (i === 1 || i === 2) {
          const cols = 24, cw = Math.min(20, (w - 2 * pad - 80) / cols);
          const gx = pad + 80, y1 = 50;
          V.text(g, 'E4M3', pad, y1 + 12, { size: 11.5, font: 'mono', color: C.d, weight: 700 });
          cellRow(g, gx + (cols - 4) * cw, y1, qb.toString(2), cw, C.d, C);
          V.text(g, `${nice(q)} = 1,${qb.toString(2).slice(1)}₂ × 2${sup(qe)}`, gx, y1 + 12, { size: 11.5, font: 'mono', color: C.d, maxW: (cols - 4) * cw - 10 });
          const y2 = y1 + cw + 12;
          V.text(g, 'BF16', pad, y2 + 12, { size: 11.5, font: 'mono', color: C.a, weight: 700 });
          cellRow(g, gx + (cols - 8) * cw, y2, sbb.toString(2), cw, C.a, C);
          V.text(g, `s = 1,${sbb.toString(2).slice(1)}₂ × 2${sup(se)}`, gx, y2 + 12, { size: 11.5, font: 'mono', color: C.a, maxW: (cols - 8) * cw - 10 });
          if (i === 1) {
            let yy = y2 + cw + 22;
            V.line(g, gx + (cols - 12) * cw - 4, yy - 8, gx + cols * cw, yy - 8, { color: V.alpha(C.muted, 0.5) });
            const qs = qb.toString(2);
            let shown = 0;
            for (let j = 0; j < 4; j++) {
              const bit = qs[3 - j] === '1';
              const appear = el > 0.4 + j * 0.45;
              if (!appear) continue;
              const str = bit ? sbb.toString(2) : '00000000';
              cellRow(g, gx + (cols - 8 - j) * cw, yy, str, cw, C.b, C, { dim: !bit });
              V.text(g, `${bit ? '+' : '0'} s ≪ ${j}`, pad, yy + 12, { size: 11, font: 'mono', color: bit ? C.b : C.muted });
              yy += cw + 6; shown++;
            }
            if (el > 2.3) {
              V.line(g, gx + (cols - 12) * cw - 4, yy - 2, gx + cols * cw, yy - 2, { color: C.ink, lw: 1.5 });
              cellRow(g, gx + (cols - Pbits.length) * cw, yy + 4, Pbits, cw, C.c, C);
              V.text(g, `produto: ${Pbits.length} bits`, pad, yy + 16, { size: 11, font: 'mono', color: C.c, weight: 700 });
              yy += cw + 28;
              V.text(g, 'mantissa do FP32 (24 bits significativos)', pad, yy - 6, { size: 11, font: 'mono', color: C.muted });
              const fx = pad;
              const fcw = Math.min(20, (w - 2 * pad) / 24);
              cellRow(g, fx, yy + 2, Pbits.padEnd(24, '0'), fcw, C.b, C, { colAt: (k) => (k < Pbits.length ? C.c : C.b) });
              V.text(g, `4 + 8 = 12 ≤ 24 → produto exato em FP32 ✓   ·   expoentes: 2${sup(qe)} × 2${sup(se)} = 2${sup(qe + se)}`, pad, yy + fcw + 24, { size: 12, color: C.a, weight: 600, maxW: w - 2 * pad });
              fitH(fig, yy + fcw + 40);
            } else fitH(fig, Math.max(h, 360));
          } else {
            const exact = q * s, bf = rB(exact);
            const L = Pbits.length, keep = Math.min(8, L);
            let yy = y2 + cw + 30;
            V.text(g, 'produto exato (FP32)', pad, yy - 6, { size: 11, font: 'mono', color: C.muted });
            cellRow(g, gx, yy, Pbits, cw, C.c, C, { colAt: (k) => (k < keep ? C.a : C.muted) });
            bracket(g, gx, gx + keep * cw - 2, yy + cw + 6, C.a);
            V.text(g, 'fica (8 bits do BF16)', gx + (keep * cw) / 2, yy + cw + 24, { size: 11, font: 'mono', color: C.a, align: 'center' });
            if (L > keep) { bracket(g, gx + keep * cw, gx + L * cw - 2, yy + cw + 6, C.muted); V.text(g, 'RNE', gx + keep * cw + ((L - keep) * cw) / 2, yy + cw + 24, { size: 11, font: 'mono', color: C.muted, align: 'center' }); }
            yy += cw + 50;
            V.text(g, `exato: ${nice(exact, 12)}   →   BF16: ${nice(bf, 8)} (${hex16(bf)})   ·   erro relativo ${pct(Math.abs(bf - exact) / exact)}`, pad, yy, { size: 12.5, font: 'mono', color: C.ink, weight: 600, maxW: w - 2 * pad });
            yy += 22;
            V.text(g, '1 único arredondamento ✓', pad, yy, { size: 12.5, color: C.a, weight: 700 });
            yy += 34;
            V.text(g, 'e se a escala tivesse 24 bits (FP32)?', pad, yy, { size: 12, color: C.muted, weight: 600 });
            const fcw = Math.min(16, (w - 2 * pad - 20) / 30);
            yy += 10;
            for (let k = 0; k < 28; k++) { const over = k >= 24; V.box(g, pad + k * fcw, yy, fcw - 2, fcw, { r: 2, fill: V.alpha(over ? C.red || C.c : C.c, over ? 0.7 : 0.5) }); }
            g.strokeStyle = C.b; g.lineWidth = 2; V.rr(g, pad - 3, yy - 3, 24 * fcw + 4, fcw + 6, 4); g.stroke();
            const nlF = wrap(g, 'até 4 + 24 = 28 bits > 24: o FP32 pode arredondar o produto, e depois vem o BF16 (até dois arredondamentos)', pad, yy + fcw + 20, w - 2 * pad, 16, { size: 11.5, color: C.red || C.c });
            fitH(fig, yy + fcw + 20 + nlF * 16 + 4);
          }
        } else if (i === 3) {
          const nCh = Math.ceil(NR / rows);
          const topY = 50, barH = Math.min(260, h - 150), bw = narrow ? 46 : 70;
          const lx = pad + 10, ox = w / 2 - (narrow ? 40 : 60), rx = w - pad - bw - 10;
          const per = Math.max(0.12, Math.min(0.45, 9 / nCh));
          const k = Math.floor(t / per) % (nCh + 3);
          const fr = (t % per) / per;
          V.text(g, 'FP8 E4M3', lx + bw / 2, topY - 8, { size: 11.5, font: 'mono', color: C.d, align: 'center', weight: 700 });
          V.text(g, 'BF16 (cópia)', rx + bw / 2, topY - 8, { size: 11.5, font: 'mono', color: C.a, align: 'center', weight: 700 });
          V.box(g, lx, topY, bw, barH, { r: 6, fill: V.alpha(C.d, 0.18), stroke: V.alpha(C.d, 0.6) });
          V.box(g, rx, topY, bw, barH, { r: 6, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.6) });
          const bandH = (barH * rows) / NR;
          const done = Math.min(k, nCh);
          if (done > 0) V.box(g, rx + 2, topY + 2, bw - 4, Math.min(barH - 4, bandH * done), { r: 4, fill: V.alpha(C.a, 0.7) });
          if (nCh <= 120) for (let c = 1; c < nCh; c++) { const yy = topY + c * bandH; V.line(g, lx + 2, yy, lx + bw - 2, yy, { color: V.alpha(C.d, 0.5), lw: 0.8 }); }
          // forno FP32: contorno = tamanho sem blocos; preenchimento = rascunho real
          const ow = narrow ? 92 : 132, ohFull = barH * 0.72, fy = topY + (barH - ohFull) / 2;
          const oh = Math.max(8, ohFull * (rows / NR));
          const oy = fy + ohFull - oh;
          g.setLineDash([5, 4]); V.box(g, ox, fy, ow, ohFull, { r: 8, stroke: V.alpha(C.muted, 0.7), lw: 1.3 }); g.setLineDash([]);
          V.box(g, ox + 3, oy, ow - 6, oh - 2, { r: 6, fill: V.alpha(C.b, 0.55), stroke: C.b, lw: 2 });
          V.glow(g, ox + ow / 2, oy + oh / 2, 26, C.b, 0.5 + 0.3 * Math.sin(t * 6));
          V.text(g, 'rascunho FP32', ox + ow / 2, fy - 8, { size: 11.5, font: 'mono', color: C.b, align: 'center', weight: 700 });
          V.text(g, vb(peak), ox + ow / 2, fy + ohFull + 16, { size: 12.5, font: 'mono', color: C.ink, align: 'center', weight: 700 });
          // esteira
          V.line(g, lx + bw, topY + barH + 20, rx, topY + barH + 20, { color: V.alpha(C.muted, 0.5), lw: 3, dash: [8, 6], dashOffset: -t * 30 });
          if (k < nCh) {
            const sy = topY + k * bandH, bh2 = Math.max(6, Math.min(bandH, oh - 4));
            const mx0 = ox + (ow - bw) / 2, my0 = oy + (oh - bh2) / 2;
            const pX = fr < 0.5 ? V.lerp(lx, mx0, fr * 2) : V.lerp(mx0, rx, (fr - 0.5) * 2);
            const pY = fr < 0.5 ? V.lerp(sy, my0, fr * 2) : V.lerp(my0, sy, (fr - 0.5) * 2);
            V.box(g, pX, pY, bw, bh2, { r: 3, fill: fr < 0.5 ? V.alpha(C.d, 0.95) : V.alpha(C.a, 0.95), stroke: C.ink, lw: 1 });
            V.box(g, lx + 1, sy, bw - 2, Math.max(2, bandH), { r: 2, stroke: C.d, lw: 1.5 });
          }
          // rótulo do contorno por cima do bloco em movimento; sem blocos, o contorno é o próprio rascunho
          if (rows < NR) {
            V.text(g, 'sem blocos', ox + ow / 2, fy + 16, { size: 11, font: 'mono', color: C.muted, align: 'center' });
            V.text(g, vb(NR * K * 4), ox + ow / 2, fy + 31, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          }
          const gy = topY + barH + 44;
          const nlG = wrap(g, `${bi(nCh)} volta${nCh > 1 ? 's' : ''} de ${bi(rows)} linha${rows > 1 ? 's' : ''}${NR % rows && nCh > 1 ? ` (a última com ${bi(NR - (nCh - 1) * rows)})` : ''} · temporário = ${bi(rows)} × ${bi(K)} × 4 B = ${vb(peak)} (pico ~${vb(2 * peak)})`, pad, gy, w - 2 * pad, 16, { size: 12, font: 'mono', color: C.ink, weight: 600 });
          const full = NR * K * 4, gw = w - 2 * pad, gY = gy + 14 + (nlG - 1) * 16;
          V.box(g, pad, gY, gw, 14, { r: 5, fill: V.alpha(C.muted, 0.15) });
          V.box(g, pad, gY, Math.max(3, (peak / full) * gw), 14, { r: 5, fill: V.alpha(C.b, 0.85) });
          if (narrow) {
            const nlC = wrap(g, `cópia BF16 que fica: ${vb(copy)} por camada × 34 = ${vb(copy * 34)}`, pad, gY + 30, gw, 15, { size: 11, font: 'mono', color: C.a });
            V.text(g, `sem blocos: ${vb(full)}`, pad, gY + 30 + nlC * 15, { size: 11, font: 'mono', color: C.muted });
            fitH(fig, gY + 44 + nlC * 15);
          } else {
            V.text(g, `sem blocos: ${vb(full)}`, pad + gw, gY + 30, { size: 11, font: 'mono', color: C.muted, align: 'right' });
            V.text(g, `cópia BF16 que fica: ${vb(copy)} por camada × 34 = ${vb(copy * 34)}`, pad, gY + 30, { size: 11, font: 'mono', color: C.a });
            fitH(fig, gY + 44);
          }
        } else if (i === 4) {
          const colW = narrow ? w - 2 * pad : (w - 3 * pad) / 2;
          const x1 = pad, x2 = narrow ? pad : pad * 2 + colW;
          let y = 52;
          V.text(g, 'CÓPIA (elemento a elemento)', x1, y, { size: 11.5, font: 'mono', color: C.a, weight: 700 });
          V.text(g, 'bloco 512   ·   bloco 2.048', x1, y + 20, { size: 11, font: 'mono', color: C.muted });
          for (let r = 0; r < 4; r++) {
            const v = Wl[r][r + 1];
            V.text(g, `${hex16(v)}   =   ${hex16(v)}   ✓`, x1, y + 44 + r * 20, { size: 12.5, font: 'mono', color: C.ink, weight: 600 });
          }
          V.text(g, 'mesmos bits para qualquer tamanho de bloco', x1, y + 132, { size: 12, color: C.a, weight: 600, maxW: colW });
          y = narrow ? y + 170 : 52;
          V.text(g, 'UMA SOMA (redução) de 4.096 produtos', x2, y, { size: 11.5, font: 'mono', color: C.c, weight: 700, maxW: colW });
          RED.forEach((rr, k) => {
            V.text(g, `parciais de ${bi(rr.c)}: ${hex32(rr.v)} = ${nice(rr.v, 9)}`, x2, y + 24 + k * 20, { size: 12, font: 'mono', color: C.ink, maxW: colW });
          });
          const allSame = RED.every((rr) => rr.v === RED[0].v);
          V.text(g, allSame ? 'aqui coincidiram, mas nada garante' : 'fatiar a soma mudou os últimos bits ✕', x2, y + 24 + 3 * 20 + 8, { size: 12, color: allSame ? C.muted : C.c, weight: 600, maxW: colW });
          const by = narrow ? y + 130 : 230;
          wrap(g, 'Na cópia, cada saída depende de 1 código e 1 escala: o bloco só decide quantas linhas passam pelo rascunho de cada vez. Numa redução, cada saída depende de milhares de parcelas, e o bloco decide a ordem em que elas são somadas.', pad, by, w - 2 * pad, 17, { size: 12, color: C.text2 || C.muted });
          fitH(fig, by + 70);
        } else {
          const ax0 = pad + 20, ax1 = w - pad - 20, ay = 150;
          const Lm = (m) => ax0 + (Math.log2(m) / Math.log2(16384)) * (ax1 - ax0);
          V.box(g, Lm(1), ay - 60, Lm(220) - Lm(1), 90, { r: 6, fill: V.alpha(C.a, 0.12) });
          V.box(g, Lm(221), ay - 60, Lm(511) - Lm(221), 90, { r: 6, fill: V.alpha(C.muted, 0.08) });
          V.box(g, Lm(768), ay - 60, Lm(16384) - Lm(768), 90, { r: 6, fill: V.alpha(C.b, 0.12) });
          V.text(g, 'decode ≤ 220', (Lm(1) + Lm(220)) / 2, ay - 66, { size: 11, font: 'mono', color: C.a, align: 'center' });
          V.text(g, narrow ? 'vazia' : '221–511 vazia', (Lm(221) + Lm(511)) / 2, ay - 66, { size: 11, font: 'mono', color: C.muted, align: 'center' });
          V.text(g, 'prefill ≥ 768', (Lm(768) + Lm(16384)) / 2, ay - 66, { size: 11, font: 'mono', color: C.b, align: 'center' });
          V.line(g, ax0, ay, ax1, ay, { color: C.muted, lw: 1.3 });
          [1, 8, 64, 512, 4096].forEach((m) => { V.line(g, Lm(m), ay - 4, Lm(m), ay + 4, { color: C.muted }); V.text(g, bi(m), Lm(m), ay + 18, { size: 11, font: 'mono', color: C.muted, align: 'center' }); });
          V.line(g, Lm(512), ay - 60, Lm(512), ay + 30, { color: C.c, lw: 2, dash: [5, 3] });
          V.text(g, 'MIN_M = 512', Lm(512), ay + 42, { size: 11.5, font: 'mono', color: C.c, align: 'center', weight: 700 });
          const mNow = Math.round(2 ** (1 + (Math.sin(t * 0.6) * 0.5 + 0.5) * 12.5));
          const big = mNow > 512;
          V.glow(g, Lm(mNow), ay, 16, big ? C.b : C.a);
          V.dot(g, Lm(mNow), ay, 5, big ? C.b : C.a);
          V.text(g, `M = ${bi(mNow)}`, Lm(mNow), ay - 12, { size: 12, font: 'mono', color: C.ink, align: 'center', weight: 700 });
          const ly = ay + 70;
          V.box(g, pad, ly, (w - 3 * pad) / 2, 40, { r: 8, fill: V.alpha(C.d, big ? 0.08 : 0.3), stroke: C.d, lw: big ? 1 : 2.5 });
          V.box(g, pad * 2 + (w - 3 * pad) / 2, ly, (w - 3 * pad) / 2, 40, { r: 8, fill: V.alpha(C.a, big ? 0.3 : 0.08), stroke: C.a, lw: big ? 2.5 : 1 });
          V.text(g, 'M ≤ 512: Marlin W8A16 (lê FP8)', pad + (w - 3 * pad) / 4, ly + 24, { size: narrow ? 11 : 12, color: C.ink, align: 'center', weight: 600, maxW: (w - 3 * pad) / 2 - 10 });
          V.text(g, 'M > 512: cuBLAS BF16 (lê a cópia)', pad * 2 + (3 * (w - 3 * pad)) / 4, ly + 24, { size: narrow ? 11 : 12, color: C.ink, align: 'center', weight: 600, maxW: (w - 3 * pad) / 2 - 10 });
          const by = ly + 70;
          V.text(g, 'in_proj com M = 7.168 (mediana de 20):', pad, by, { size: 11.5, font: 'mono', color: C.muted });
          const bars = [['Marlin W8A16', 27.21, C.d], ['W8A8 antigo', 12.57, C.muted], ['BF16 (cópia)', 7.36, C.a]];
          const bx0 = pad + 110, bw = w - bx0 - pad - 70;
          bars.forEach(([n, v, c], k) => {
            const yy = by + 14 + k * 22;
            V.text(g, n, pad, yy + 11, { size: 11.5, font: 'mono', color: C.ink });
            V.box(g, bx0, yy, (v / 27.21) * bw, 14, { r: 4, fill: V.alpha(c, 0.8) });
            V.text(g, v.toFixed(2).replace('.', ',') + ' ms', bx0 + (v / 27.21) * bw + 6, yy + 11, { size: 11.5, font: 'mono', color: C.ink });
          });
          fitH(fig, by + 14 + 3 * 22 + 12);
        }
      });
    }

    /* ================================================================
       Figura 10 — a calculadora de tensores
       ================================================================ */
    {
      const fig = $('#fig-bytes');
      // formatos: bits por elemento e bytes de escalas de UMA matriz [R × C]
      const FMTS = [
        { v: 'fp32', label: 'FP32', bits: 32, col: 'b', sc: () => 0, how: '' },
        { v: 'bf16', label: 'BF16', bits: 16, col: 'a', sc: () => 0, how: '' },
        { v: 'fp8t', label: 'FP8 · 1 escala', bits: 8, col: 'd', sc: () => 2, how: '1 escala BF16 para o tensor todo' },
        { v: 'fp8r', label: 'FP8 · por linha', bits: 8, col: 'd', sc: (R) => 2 * R, how: '1 escala BF16 por linha' },
        { v: 'nvfp4', label: 'NVFP4', bits: 4, col: 'c', sc: (R, Cc) => R * Math.ceil(Cc / 16) + 4, how: '1 escala E4M3 a cada 16 valores + 1 FP32 global' },
        { v: 'mxfp4', label: 'MXFP4', bits: 4, col: 'c', sc: (R, Cc) => R * Math.ceil(Cc / 32), how: '1 expoente E8M0 a cada 32 valores' },
        { v: 'exl3', label: 'EXL3 K4', bits: 4, col: 'f', sc: (R, Cc) => 2 * (R + Cc) + 4, how: 'vetores suh e svh em FP16 + marcador int32' },
      ];
      const FM = (v) => FMTS.find((f) => f.v === v);
      const PRE = [
        { v: 'kda', label: 'cópia KDA in_proj · GLM (TP2)', R: 12576, C: 4096, n: 1, f: 'bf16', d: 'a cópia BF16 da seção anterior, por camada e por rank' },
        { v: 'glmexp', label: 'experts roteados · GLM', R: 4096, C: 2048, n: 37152, f: 'exl3', d: '43 camadas × 288 experts × 3 matrizes, no formato do checkpoint (cálculo)' },
        { v: 'qexp', label: 'experts roteados · Qwen', R: 640, C: 2560, n: 73728, f: 'nvfp4', d: '48 camadas × 512 experts × 3 matrizes' },
        { v: 'ngram', label: 'tabela N-gram (PLE) · Qwen', R: 320001536, C: 160, n: 1, f: 'fp8t', d: '320 milhões de linhas de 160 valores, com uma única escala global' },
        { v: 'qemb', label: 'embedding · Qwen', R: 248320, C: 2560, n: 1, f: 'bf16', d: 'uma linha de 2.560 valores por token do vocabulário' },
        { v: 'dshead', label: 'lm_head · DeepSeek', R: 129280, C: 4096, n: 1, f: 'bf16', d: 'vocabulário × largura do modelo' },
        { v: 'dsexp', label: '1 expert · DeepSeek', R: 6144, C: 4096, n: 1, f: 'mxfp4', d: 'gate + up + down de um expert: 3 × 2.048 × 4.096 valores' },
      ];
      const S = { R: 12576, C: 4096, n: 1, fmt: 'bf16', unit: 'gib', pre: PRE[0], custom: false, k: 0.7, s: 0, bA: 16, sp: 1 };
      const two = (x) => x.toFixed(2).replace('.', ',');
      const fBin = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0; while (b >= 1024 && i < 4) { b /= 1024; i++; } return (i ? two(b) : String(Math.round(b))) + ' ' + u[i]; };
      const fDec = (b) => { const u = ['B', 'kB', 'MB', 'GB', 'TB']; let i = 0; while (b >= 1000 && i < 4) { b /= 1000; i++; } return (i ? two(b) : String(Math.round(b))) + ' ' + u[i]; };
      const fU = (b) => (S.unit === 'gib' ? fBin(b) : fDec(b));
      const fO = (b) => (S.unit === 'gib' ? fDec(b) : fBin(b));
      const sizes = (f) => { const numel = S.R * S.C; const data = S.n * numel * f.bits / 8, sc = S.n * f.sc(S.R, S.C); return { data, sc, tot: data + sc, bpe: ((data + sc) * 8) / (S.n * numel) }; };
      const rg = V.rng(9), W0 = Array.from({ length: 32 }, () => rg.normal());
      const wMax = Math.max(...W0.map(Math.abs));
      const ctl = UI.controls(fig);
      let slR, slC, fmtSeg, H;
      const setPre = (v) => {
        const p = PRE.find((q) => q.v === v); S.pre = p; S.n = p.n;
        slR.set(Math.log2(p.R)); slC.set(Math.log2(p.C)); S.custom = false;
        fmtSeg.set(p.f);
      };
      UI.select(ctl, { label: 'Tensor', options: PRE.map((p) => ({ v: p.v, label: p.label })), value: 'kda', onChange: (v) => setPre(v) });
      fmtSeg = UI.seg(ctl, { label: 'Formato', options: FMTS.map((f) => ({ v: f.v, label: f.label })), value: 'bf16', onChange: (v) => { S.fmt = v; S.sp = 0; } });
      slR = UI.slider(ctl, { label: 'linhas', min: 4, max: 28.3, step: 0.01, value: Math.log2(12576), fmt: (v) => bi(Math.round(2 ** v)), onInput: (v) => { S.R = Math.round(2 ** v); S.custom = true; } });
      slC = UI.slider(ctl, { label: 'colunas', min: 4, max: 16, step: 0.01, value: 12, fmt: (v) => bi(Math.round(2 ** v)), onInput: (v) => { S.C = Math.round(2 ** v); S.custom = true; } });
      UI.seg(ctl, { label: 'Unidade', options: [{ v: 'gib', label: 'GiB (2³⁰)' }, { v: 'gb', label: 'GB (10⁹)' }], value: 'gib', onChange: (v) => { S.unit = v; } });
      const rTot = UI.readout(ctl, 'total', '');
      S.R = 12576; S.C = 4096;
      const hatch = (g, x, y, ww, hh, col) => {
        if (ww <= 0 || hh <= 0) return;
        g.save(); V.rr(g, x, y, ww, hh, 2); g.clip();
        g.fillStyle = V.alpha(col, 0.14); g.fillRect(x, y, ww, hh);
        g.strokeStyle = V.alpha(col, 0.7); g.lineWidth = 1; g.beginPath();
        for (let q = -hh; q < ww; q += 5) { g.moveTo(x + q, y + hh); g.lineTo(x + q + hh, y); }
        g.stroke(); g.restore();
        g.strokeStyle = V.alpha(col, 0.8); g.lineWidth = 1; V.rr(g, x, y, ww, hh, 2); g.stroke();
      };
      H = V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C();
        const narrow = w < 640, pad = 16;
        const f = FM(S.fmt), z = sizes(f), f32 = sizes(FMTS[0]);
        const col = C[f.col], sCol = C.ink;
        rTot.set(fU(z.tot));
        // animação: o bloco encolhe/cresce até a área certa, as escalas entram depois
        const a = Math.min(1, (dt || 0.016) * 7);
        S.k += (Math.sqrt(z.data / f32.data) - S.k) * a;
        S.s += (z.sc / z.data - S.s) * a;
        S.bA += (f.bits - S.bA) * a;
        S.sp = Math.min(1, S.sp + (dt || 0.016) * 2.5);
        // cabeçalho
        const numel = S.R * S.C;
        const head = `${S.n > 1 ? bi(S.n) + ' × ' : ''}[${bi(S.R)} × ${bi(S.C)}] = ${bi(S.n * numel)} elementos`;
        let y = 22;
        const nh = wrap(g, head, pad, y, w - 2 * pad, 18, { size: 13, font: 'mono', color: C.ink, weight: 700 });
        y += (nh - 1) * 18 + 18;
        const nd = wrap(g, S.custom ? 'forma personalizada (os controles de linhas e colunas mudaram o preset)' : S.pre.d, pad, y, w - 2 * pad, 15, { size: 11.5, color: C.muted });
        const yTop = y + (nd - 1) * 15 + 16;
        // --- painel da matriz
        const PW = narrow ? w - 2 * pad : Math.round(w * 0.5) - pad - 10, PH = narrow ? 170 : 214;
        const ar = V.clamp(Math.pow(S.C / S.R, 0.3), 0.42, 2.8);
        const fh = Math.min(PH - 44, (PW - 70) / ar), fw = fh * ar;
        const ox = pad, oy = yTop + 4;
        g.setLineDash([5, 4]); V.box(g, ox, oy, fw, fh, { r: 6, stroke: V.alpha(C.muted, 0.8), lw: 1.3 }); g.setLineDash([]);
        const dw = fw * S.k, dh = fh * S.k;
        V.box(g, ox, oy, dw, dh, { r: 5, fill: V.alpha(col, 0.28), stroke: col, lw: 1.6 });
        if (dw > 24) {
          const nx = 8, ny = V.clamp(Math.round(8 / ar), 2, 14);
          for (let q = 1; q < nx; q++) V.line(g, ox + (dw * q) / nx, oy + 2, ox + (dw * q) / nx, oy + dh - 2, { color: V.alpha(col, 0.35), lw: 1 });
          for (let q = 1; q < ny; q++) V.line(g, ox + 2, oy + (dh * q) / ny, ox + dw - 2, oy + (dh * q) / ny, { color: V.alpha(col, 0.35), lw: 1 });
        }
        if (dw > 70 && dh > 26) V.text(g, f.label, ox + dw / 2, oy + dh / 2 + 4, { size: 12, font: 'mono', color: C.ink, align: 'center', weight: 700 });
        const scRatio = z.sc / z.data;
        if (scRatio > 1e-4) {
          const sw = Math.max(2, dw * S.s) * S.sp;
          hatch(g, ox + dw + 3, oy, sw, dh, sCol);
          if (fw + 70 < PW || dw + sw + 60 < PW) V.text(g, 'escalas', ox + dw + sw + 8, oy + 12, { size: 11, font: 'mono', color: C.muted });
        }
        let ly = oy + fh + 16;
        V.text(g, `dados ${fU(z.data)}${z.sc ? ` · escalas ${fU(z.sc)}` : ''}`, ox, ly, { size: 12, font: 'mono', color: col, weight: 600 });
        V.text(g, `tracejado = o mesmo tensor em FP32: ${fU(f32.data)}`, ox, ly + 17, { size: 11, font: 'mono', color: C.muted });
        // --- barras: bits por elemento
        const bx0 = narrow ? pad : Math.round(w * 0.5) + 6, by0 = narrow ? oy + PH + 8 : yTop;
        const bW = narrow ? w - 2 * pad : w - bx0 - pad, labW = narrow ? 102 : 116, valW = narrow ? 144 : 150;
        const barMax = Math.max(40, bW - labW - valW);
        V.text(g, 'BITS POR ELEMENTO, COM AS ESCALAS', bx0, by0 + 8, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const P = H.pointer; let hov = null;
        FMTS.forEach((fm, q) => {
          const zz = sizes(fm), yy = by0 + 22 + q * 25, on = fm.v === S.fmt;
          const inRow = P.inside && P.x >= bx0 && P.x <= bx0 + bW && P.y >= yy - 3 && P.y < yy + 19;
          if (inRow) hov = fm.v;
          if (on || inRow) V.box(g, bx0 - 6, yy - 4, bW + 6, 23, { r: 6, fill: V.alpha(C[fm.col], on ? 0.16 : 0.07) });
          V.text(g, fm.label, bx0, yy + 11, { size: narrow ? 11 : 11.5, font: 'mono', color: on ? C.ink : C.muted, weight: on ? 700 : 500, maxW: labW - 4 });
          const wd = (fm.bits / 32) * barMax, ws = ((zz.bpe - fm.bits) / 32) * barMax;
          V.box(g, bx0 + labW, yy, Math.max(2, wd), 14, { r: 3, fill: V.alpha(C[fm.col], on ? 0.9 : 0.55) });
          if (ws > 0.4) hatch(g, bx0 + labW + wd + 1, yy, Math.max(1.5, ws), 14, sCol);
          V.text(g, `${two(zz.bpe)} b · ${fU(zz.tot)}`, bx0 + labW + barMax + 8, yy + 11, { size: 11, font: 'mono', color: on ? C.ink : C.muted, weight: on ? 700 : 400 });
        });
        if (P.clicked && hov) { P.clicked = false; if (hov !== S.fmt) fmtSeg.set(hov); }
        H.canvas.style.cursor = hov ? 'pointer' : 'default';
        // --- total e conta
        const yF = Math.max(ly + 17, by0 + 22 + FMTS.length * 25) + 30;
        V.text(g, fU(z.tot), pad, yF, { size: narrow ? 20 : 24, font: 'display', color: C.ink, weight: 700 });
        g.font = V.font(narrow ? 20 : 24, 'display', 700);
        const tw = g.measureText(fU(z.tot)).width;
        V.text(g, `= ${fO(z.tot)} = ${bi(Math.round(z.tot))} B`, pad + tw + 12, yF, { size: 12, font: 'mono', color: C.muted, maxW: w - 2 * pad - tw - 12 });
        const form = `bytes = ${S.n > 1 ? bi(S.n) + ' × ' : ''}${bi(S.R)} × ${bi(S.C)} × ${f.bits}/8${z.sc ? ` + escalas (${f.how})` : ''} · ${two(z.bpe)} bits por elemento`;
        const nF = wrap(g, form, pad, yF + 22, w - 2 * pad, 16, { size: 11.5, font: 'mono', color: C.text2 || C.muted });
        // --- uma linha na memória
        const nEl = narrow ? 16 : 32, uBit = (w - 2 * pad) / (nEl * 32 + 16);
        const yS = yF + 22 + nF * 16 + 18;
        const nT = wrap(g, `UMA LINHA NA MEMÓRIA · os primeiros ${nEl} valores, largura proporcional aos bits`, pad, yS, w - 2 * pad, 15, { size: 11, font: 'mono', color: C.muted, weight: 600 });
        const cy = yS + 10 + (nT - 1) * 15, chh = 26;
        let x = pad;
        const scaleCell = (bits, lab) => { const ww = bits * uBit * S.sp; if (ww < 0.5) return; hatch(g, x, cy, ww, chh, sCol); if (ww > 14) V.text(g, lab, x + ww / 2, cy + chh + 12, { size: 11, font: 'mono', color: C.muted, align: 'center' }); x += ww + 1; };
        if (S.fmt === 'exl3') {
          const words = nEl / 4, ww = 16 * uBit;
          for (let q = 0; q < words; q++) { hatch(g, x, cy, ww - 1, chh, col); if (ww > 34) V.text(g, 'int16', x + ww / 2, cy + chh / 2 + 4, { size: 11, font: 'mono', color: C.ink, align: 'center' }); x += ww; }
        } else {
          if (S.fmt === 'fp8r') scaleCell(16, 's');
          for (let q = 0; q < nEl; q++) {
            if (S.fmt === 'nvfp4' && q % 16 === 0) scaleCell(8, 's');
            if (S.fmt === 'mxfp4' && q % 32 === 0) scaleCell(8, 's');
            const ww = S.bA * uBit;
            V.box(g, x, cy, Math.max(1, ww - 1), chh, { r: 2, fill: V.cmap(W0[q] / wMax, 'div'), stroke: V.alpha(col, 0.7), lw: 1 });
            x += ww;
          }
        }
        const rowBits = S.fmt === 'exl3' ? nEl * 4 : nEl * f.bits + (S.fmt === 'fp8r' ? 16 : S.fmt === 'nvfp4' ? Math.ceil(nEl / 16) * 8 : S.fmt === 'mxfp4' ? Math.ceil(nEl / 32) * 8 : 0);
        V.text(g, `${bi(rowBits)} bits`, Math.min(w - pad, x + 6), cy + chh / 2 + 4, { size: 11, font: 'mono', color: C.muted, align: x + 60 > w - pad ? 'right' : 'left' });
        const note = S.fmt === 'exl3'
          ? 'Treliça: os 256 pesos de um tile 16 × 16 viram 64 palavras int16 (4 bits por peso em média), e nenhum bit pertence a um peso só.'
          : S.fmt === 'fp8t' ? 'A única escala do tensor fica fora da linha: 2 bytes para o tensor inteiro.'
            : S.fmt === 'fp32' || S.fmt === 'bf16' ? 'Sem escalas: cada valor carrega o próprio expoente.'
              : 'As células hachuradas são as escalas, intercaladas com os valores que elas servem.';
        const nN = wrap(g, note, pad, cy + chh + 30, w - 2 * pad, 15, { size: 11.5, color: C.text2 || C.muted });
        fitH(fig, cy + chh + 30 + (nN - 1) * 15 + 14);
      });
    }

    /* ================================================================
       Figura 11 — o metrô dos dtypes
       ================================================================ */
    {
      const fig = $('#fig-metro');
      // família → [cor, nome, letra dentro da estação]
      const FAM = { bf16: ['a', 'BF16', 'B'], fp16: ['e', 'FP16', 'H'], fp32: ['b', 'FP32', '32'], fp8: ['d', 'FP8', '8'], fp4: ['c', 'FP4', '4'], int: ['f', 'treliça EXL3', 'T'] };
      const LINES = {
        glm: [
          { n: 'Fluxo residual', k: 'bf16', tok: 'bf16', d: 'BF16', info: 'Ativações em BF16 (dtype bfloat16 do config), largura 4.096 em cada um dos 4 fluxos mHC. É a moeda que circula entre as camadas.', b: '4.096 × 2 B = 8 KiB por token e por fluxo' },
          { n: 'Projeções densas', k: 'fp8', tok: 'bf16', d: 'FP8 E4M3 por canal', info: 'GLM53_DENSE_FP8=all: pesos BF16 requantizados na carga (escala amax/448 por linha, guardada em BF16) e executados pelo Marlin W8A16; a ativação continua BF16. Na in_proj da KDA, M > 512 usa a cópia BF16 exata.', b: '1 B/peso + 1 escala BF16 por linha · cópia KDA: +98,25 MiB/camada/rank' },
          { n: 'Atenção', d: '↑ KDA ×34 · ↓ MLA ×11', fork: [
            { n: 'Camada KDA (34 das 45)', k: 'fp32', tok: 'fp32', d: 'estado FP32 · conv BF16', info: 'Nas 34 camadas de atenção linear KDA não há KV cache: cada requisição carrega um estado recorrente [32 × 128 × 128] por rank, em FP32 porque se acumula ao longo de todo o contexto. O estado da convolução curta fica em BF16.', b: '2 MiB por camada/req/rank + 240 KiB de conv' },
            { n: 'Camada MLA (11 das 45)', k: 'fp8', tok: 'bf16', d: 'KV fp8_ds_mla · uint8', info: 'Nas 11 camadas MLA (atenção esparsa), --kv-cache-dtype fp8 guarda por token 512 valores FP8 E4M3 do latente, 4 escalas FP32 (uma a cada 128) e 64 BF16 de RoPE preenchidos com zeros (o GLM é NoPE), num registro uint8.', b: '512 + 16 + 128 = 656 B por token e camada' },
          ] },
          { n: 'Roteador', k: 'fp32', tok: 'fp32', d: 'FP32', info: 'moe_router_dtype float32: os logits dos 288 experts e o viés e_score_correction_bias ficam em FP32, para que empates de arredondamento não troquem os 8 escolhidos.', b: '288 logits por token' },
          { n: 'Experts', k: 'int', tok: 'fp16', d: 'EXL3 K4 · int16 → FP16', info: 'Treliça em palavras int16 (4 bits/peso), suh/svh em FP16 e o marcador int32 −877.912.083 que o checkpoint traz. O kernel recebe x.half(), decodifica os pesos direto em FP16, aplica o swiglu_limit 10,0 do próprio modelo (o mesmo que o vLLM aplica nas MLPs BF16) e devolve FP32.', b: '≈ 4,01 bits/peso efetivos numa matriz [4096 × 2048]' },
          { n: 'lm_head', k: 'bf16', tok: 'bf16', d: 'BF16', info: 'Não amarrado ao embedding (tie_word_embeddings: false) e compartilhado com o MTP e o DFlash2.', b: '634,4 M parâmetros · 1,18 GiB (1,27 GB)' },
        ],
        qwen: [
          { n: 'Fluxo residual', k: 'bf16', tok: 'bf16', d: 'BF16', info: 'Ativações em BF16, largura 2.560 em cada ramo do Gated Residual.', b: '2.560 × 2 B = 5 KiB por token e por ramo' },
          { n: 'Densos', k: 'bf16', tok: 'bf16', d: 'BF16 (checkpoint nvidia)', info: 'GDN, atenção, HyperConnection, shared experts e lm_head ficam em BF16 no checkpoint default. O checkpoint FP8-dense opcional os leva para FP8 por canal: 9,2 → 5,7 GiB lidos por passo, numa conta analítica ainda não medida.', b: '≈ 6,9 dos ≈ 9,2 GiB lidos por passo' },
          { n: 'Atenção', d: '↑ GDN ×36 · ↓ QSA ×12', fork: [
            { n: 'Camada GDN (36 das 48)', k: 'bf16', tok: 'bf16', d: 'estado BF16 (checkpoint: FP32)', info: 'As 36 camadas GDN (atenção linear) têm um estado fixo por requisição. MAMBA_SSM_CACHE_DTYPE=bfloat16 troca o float32 do checkpoint: metade dos bytes lidos e escritos a cada passo. Medido no single-Spark: +8,5% de decode com 8 streams.', b: '0,75 MiB por camada/req/GPU (FP32: 1,5)' },
            { n: 'Camada QSA (12 das 48)', k: 'fp8', tok: 'bf16', d: 'KV FP8 E4M3 · por tensor', info: 'O vLLM aloca o KV como uint8, e _qsa_as_fp8 o reinterpreta como float8_e4m3fn sem cópia. K sobe para o dtype de Q (upcast exato) e k_scale é aplicado depois do dot. Têm KV as 12 camadas QSA e a do MTP.', b: '13.312 B/token nas 13 camadas com KV (BF16: 26.624) · 1,70× tokens por GiB' },
          ] },
          { n: 'Roteador', k: 'fp32', tok: 'fp32', d: 'softmax FP32', info: 'probs = softmax(W_r·x) em FP32, seguido do top-10 entre 512 experts.', b: '512 logits por token' },
          { n: 'Experts', k: 'fp4', tok: 'fp4', d: 'NVFP4 · W4A4', info: 'E2M1 (dois por uint8) + uma escala E4M3 a cada 16 valores + uma escala global FP32. W4A4: a ativação também é quantizada para NVFP4 no FlashInfer CUTLASS.', b: '120,80 B params → 56,25 GiB + 7,03 GiB de escalas' },
          { n: 'Tabela N-gram', k: 'fp8', tok: 'bf16', d: 'FP8 E4M3 · 1 escala', info: 'A tabela PLE de 320 milhões de linhas × 160 fica em FP8 com uma única escala global no checkpoint padrão. Cada token só consulta poucas linhas.', b: '51,2 B params → 47,68 GiB (BF16: ≈ 95,4 GiB)' },
          { n: 'lm_head', k: 'bf16', tok: 'bf16', d: 'BF16', info: 'Vocabulário 248.320 × 2.560, não amarrado ao embedding.', b: '0,636 B params · 1,18 GiB' },
        ],
        ds: [
          { n: 'Fluxo + mHC', k: 'fp32', tok: 'bf16', d: 'BF16 · mHC FP32', info: 'Ativações em BF16 (torch_dtype bfloat16), largura 4.096 por fluxo. Os parâmetros do mHC (hc_*_fn [24 × 16384], escalas, bases) são FP32.', b: 'hc_fn: ~0,39 M params FP32 por subcamada' },
          { n: 'Lineares de atenção', k: 'fp8', tok: 'bf16', d: 'FP8 bloco 128×128 · UE8M0', info: 'Pesos FP8 E4M3 com uma escala potência de 2 (UE8M0, weight_scale_inv) a cada bloco de 128 × 128.', b: '1 escala por 16.384 pesos ≈ 8,002 bits/peso' },
          { n: 'KV', k: 'fp8', tok: 'bf16', d: 'uint8 · 584 B', info: '448 dims NoPE em FP8 E4M3 (7 blocos de 64, escala UE8M0 arredondada para cima), 64 dims RoPE em BF16 e 8 bytes de escalas (7 + 1 de pad). Índices de bloco em int64.', b: '448 + 128 + 8 = 584 B por token e camada' },
          { n: 'Roteador', k: 'fp32', tok: 'fp32', d: 'sqrtsoftplus FP32', info: 's = √softplus(x·W_gateᵀ) em FP32, top-6 entre 256 experts.', b: '256 scores por token' },
          { n: 'Experts', k: 'fp4', tok: 'bf16', d: 'MXFP4 → W4A16', info: 'E2M1 (dois por uint8) + um expoente E8M0 (uint8) a cada 32. O b12x desquantiza para BF16 e usa MMA BF16: a ativação não é quantizada.', b: '4,25 bits/valor · 12,75 MiB (13,4 MB) por expert' },
          { n: 'lm_head', k: 'bf16', tok: 'bf16', d: 'BF16', info: 'Vocabulário 129.280 × 4.096 em BF16. Depois dele, o ring buffer de contexto do rascunho DSpark também é BF16.', b: '0,99 GiB (1,06 GB)' },
        ],
      };
      const expIdx = (r) => LINES[r].findIndex((s) => s.n === 'Experts');
      const S = { r: 'glm', sel: [expIdx('glm'), 0] };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Receita', options: [{ v: 'glm', label: 'GLM-5.3-Flash' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next' }, { v: 'ds', label: 'DeepSeek-V4-Flash' }], value: 'glm', onChange: (v) => { S.r = v; S.sel = [expIdx(v), 0]; } });
      // a legenda usa as variáveis CSS: acompanha a troca de tema sem recriar nada
      UI.legend(ctl, Object.values(FAM).map(([c, l, gl]) => [`var(--viz-${c})`, `${gl} = ${l}`]));
      const along = (pts, fr) => {
        const L = []; let tot = 0;
        for (let q = 1; q < pts.length; q++) { const d = Math.hypot(pts[q][0] - pts[q - 1][0], pts[q][1] - pts[q - 1][1]); L.push(d); tot += d; }
        let d = fr * tot;
        for (let q = 0; q < L.length; q++) { if (d <= L[q] || q === L.length - 1) { const u = L[q] ? Math.min(1, d / L[q]) : 0; return [V.lerp(pts[q][0], pts[q + 1][0], u), V.lerp(pts[q][1], pts[q + 1][1], u)]; } d -= L[q]; }
        return pts[pts.length - 1];
      };
      const path = (g, pts) => {
        g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
        for (let q = 1; q < pts.length - 1; q++) g.arcTo(pts[q][0], pts[q][1], pts[q + 1][0], pts[q + 1][1], 16);
        g.lineTo(pts[pts.length - 1][0], pts[pts.length - 1][1]);
      };
      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C();
        const L = LINES[S.r], n = L.length;
        const vert = w < 600;
        const cols = vert ? 1 : w >= 760 ? 4 : 3, rows = Math.ceil(n / cols);
        const padX = vert ? 44 : w >= 760 ? 104 : 88, top = vert ? 34 : 56, rh = vert ? 72 : 126;
        const dx = vert ? 0 : (w - 2 * padX) / (cols - 1);
        const rowOf = (i) => (vert ? i : Math.floor(i / cols));
        const colOf = (i) => { if (vert) return 0; const r = rowOf(i), c = i % cols; return r % 2 ? cols - 1 - c : c; };
        const pos = L.map((_, i) => [padX + colOf(i) * dx, top + rowOf(i) * rh]);
        const FO = vert ? [[-15, 0], [15, 0]] : [[0, -17], [0, 17]];
        const subs = (i) => (L[i].fork ? [0, 1] : [0]);
        const st = (i, j) => (L[i].fork ? L[i].fork[j] : L[i]);
        const sp = (i, j) => (L[i].fork ? [pos[i][0] + FO[j][0], pos[i][1] + FO[j][1]] : pos[i]);
        // trecho entre a estação i (ramo a) e a i+1 (ramo b); troca de linha contorna por fora
        const seg = (i, a, b) => {
          const p = sp(i, a), q = sp(i + 1, b);
          if (vert || rowOf(i) === rowOf(i + 1)) return [p, q];
          const ex = colOf(i) === cols - 1 ? Math.max(p[0], q[0]) + 48 : Math.min(p[0], q[0]) - 48;
          return [p, [ex, p[1]], [ex, q[1]], q];
        };
        // trilho
        g.lineCap = 'round'; g.lineJoin = 'round';
        for (let i = 0; i < n - 1; i++) for (const a of subs(i)) for (const b of subs(i + 1)) { g.strokeStyle = V.alpha(C.muted, 0.22); g.lineWidth = 12; path(g, seg(i, a, b)); g.stroke(); }
        for (let i = 0; i < n - 1; i++) for (const a of subs(i)) for (const b of subs(i + 1)) { g.strokeStyle = V.alpha(C[FAM[st(i, a).tok][0]], 0.55); g.lineWidth = 5; path(g, seg(i, a, b)); g.stroke(); }
        g.lineCap = 'butt';
        // clique
        const P = H.pointer; let hov = null;
        L.forEach((_, i) => subs(i).forEach((j) => { const [x, y] = sp(i, j); if (P.inside && Math.hypot(P.x - x, P.y - y) < (L[i].fork ? 16 : 24)) hov = [i, j]; }));
        if (P.clicked && hov) { S.sel = hov; P.clicked = false; }
        H.canvas.style.cursor = hov ? 'pointer' : 'default';
        // estações
        L.forEach((s, i) => {
          if (s.fork) {
            const [x, y] = pos[i];
            const pw = vert ? 66 : 40, ph = vert ? 40 : 74;
            V.box(g, x - pw / 2, y - ph / 2, pw, ph, { r: 20, fill: C.panel || C.bg, stroke: V.alpha(C.muted, 0.55), lw: 1.5 });
          }
          subs(i).forEach((j) => {
            const ss = st(i, j), fam = FAM[ss.k], col = C[fam[0]], on = S.sel[0] === i && S.sel[1] === j;
            const [x, y] = sp(i, j), r = s.fork ? (on ? 14 : 12.5) : (on ? 17 : 14.5);
            if (on) V.glow(g, x, y, 32, col, 0.8);
            g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = C.panel || C.bg; g.fill();
            g.lineWidth = on ? 4.5 : 3.5; g.strokeStyle = col;
            if (ss.k === 'fp16') g.setLineDash([4, 3]);
            g.stroke(); g.setLineDash([]);
            if (ss.k === 'int') { g.beginPath(); g.arc(x, y, r - 5, 0, Math.PI * 2); g.lineWidth = 1.2; g.stroke(); }
            V.text(g, fam[2], x, y + 1, { size: fam[2].length > 1 ? 10.5 : 11.5, font: 'mono', align: 'center', base: 'middle', color: col, weight: 700 });
          });
          const [x, y] = pos[i], col = s.fork ? C.muted : C[FAM[s.k][0]];
          if (vert) {
            const tx = x + (s.fork ? 44 : 30);
            V.text(g, s.n, tx, y - 2, { size: 12.5, weight: 600, color: C.ink, maxW: w - tx - 12 });
            V.text(g, s.d, tx, y + 15, { size: 11, font: 'mono', color: col, maxW: w - tx - 12 });
          } else {
            const yl = y + (s.fork ? 54 : 36);
            V.text(g, s.n, x, yl, { size: 12.5, weight: 600, align: 'center', color: C.ink, maxW: dx - 10 });
            V.text(g, s.d, x, yl + 16, { size: 11, font: 'mono', align: 'center', color: col, maxW: dx - 6 });
          }
        });
        // token: em cada volta, a ativação passa por um dos ramos da bifurcação
        const per = 1.6, u = (t / per) % n, k = Math.floor(u), fr = V.ease.inOut(u - k);
        const lap = Math.floor(t / (per * n)), br = (i) => (L[i].fork ? lap % 2 : 0);
        const [tx, ty] = k < n - 1 ? along(seg(k, br(k), br(k + 1)), fr) : sp(k, br(k));
        const tFam = FAM[st(k, br(k)).tok], tcol = C[tFam[0]];
        V.glow(g, tx, ty, 18, tcol);
        V.dot(g, tx, ty, 6, tcol);
        if (vert) V.text(g, 'ativação: ' + tFam[1], w - 14, ty + 4, { size: 11, font: 'mono', align: 'right', color: tcol, weight: 700 });
        else V.text(g, 'ativação: ' + tFam[1], tx, ty - 22, { size: 11, font: 'mono', align: 'center', color: C.ink, weight: 700 });
        // painel de detalhe
        const s = st(S.sel[0], S.sel[1]) || L[0], col = C[FAM[s.k][0]];
        const py = top + (rows - 1) * rh + (vert ? 44 : 86), pw = w - 32;
        const count = (txt, size, font, weight) => {
          g.font = V.font(size, font, weight); let line = '', k = 0;
          for (const wd of String(txt).split(' ')) { const tt = line ? line + ' ' + wd : wd; if (g.measureText(tt).width > pw - 24 && line) { k++; line = wd; } else line = tt; }
          return k + (line ? 1 : 0);
        };
        const nlc = count(s.info, 12.5, 'sans', 400), nbc = count(s.b, 12, 'mono', 600);
        const ph = 50 + nlc * 17 + nbc * 16;
        V.box(g, 16, py, pw, ph, { r: 12, fill: V.alpha(col, 0.1), stroke: V.alpha(col, 0.7), lw: 1.5 });
        V.text(g, `${s.n} · ${s.d}`, 28, py + 22, { size: 13.5, weight: 700, color: C.ink, maxW: pw - 24 });
        const nl = wrap(g, s.info, 28, py + 42, pw - 24, 17, { size: 12.5, color: C.text2 || C.ink });
        wrap(g, s.b, 28, py + 46 + nl * 17, pw - 24, 16, { size: 12, font: 'mono', color: col, weight: 600 });
        fitH(fig, py + ph + 10);
      });
    }

    /* ================================================================
       Figura 12 — a mesma linha, sozinha ou em lote (TensorFold)
       ================================================================ */
    {
      const fig = $('#fig-lote');
      const K = 512, FX = 2 ** 24, NR = 16;
      const S = { mode: 'split', R: 4, seed: 1, run: 0, lastCyc: -1, p: null, cur: null, solo: null, strip: null };
      const hx8 = (x) => '0x' + u32(x).toString(16).toUpperCase().padStart(8, '0');
      const splitsOf = (R) => (R <= 2 ? 8 : R <= 4 ? 4 : R <= 8 ? 2 : 1);
      const atom = () => S.mode === 'fatom' || S.mode === 'iatom';
      const nSplit = (R) => (S.mode === 'split' ? splitsOf(R) : S.mode === 'chain' ? 1 : 8);
      const perm = (n, seed) => { const a = []; for (let i = 0; i < n; i++) a.push(i); const rg = V.rng(seed); for (let i = n - 1; i > 0; i--) { const q = Math.floor(rg() * (i + 1)); const tmp = a[i]; a[i] = a[q]; a[q] = tmp; } return a; };
      const gen = (seed) => { const rg = V.rng(5003 * seed + 11); const p = new Float64Array(K); for (let i = 0; i < K; i++) p[i] = rB(rg.normal()) * rB(rg.normal()); return p; };
      // produto escalar da linha A com R linhas no lote, na execução `run`
      const evalRow = (R, run) => {
        const ns = nSplit(R), L = K / ns, p = S.p, parts = [];
        for (let s = 0; s < ns; s++) { let a = 0; for (let i = s * L; i < (s + 1) * L; i++) a = r32(a + p[i]); parts.push(a); }
        const order = atom() ? perm(ns, 7 + 131 * run + 17 * R) : parts.map((_, i) => i);
        let res = 0, q = 0;
        if (S.mode === 'iatom') { for (const i of order) q += Math.round(parts[i] * FX); res = r32(q / FX); }
        else for (const i of order) res = r32(res + parts[i]);
        return { ns, parts, order, res, q };
      };
      const ulps = (a, b) => { const ua = u32(a), ub = u32(b); return ((ua ^ ub) >>> 31) ? Infinity : Math.abs(ua - ub); };
      const compute = () => {
        S.cur = evalRow(S.R, S.run);
        S.solo = evalRow(1, 0);
        S.strip = [];
        for (let R = 1; R <= NR; R++) S.strip.push(ulps(evalRow(R, S.run).res, S.solo.res));
        const d = ulps(S.cur.res, S.solo.res);
        rd.set(d === 0 ? 'mesmos bits ✓' : d === Infinity ? 'outro sinal ✗' : `difere em ${d} ulp ✗`, d === 0 ? 'ok' : 'bad');
      };
      // dados iniciais: o primeiro conjunto em que 8, 4, 2 e 1 blocos dão quatro somas diferentes
      for (let sd = 1; sd < 80; sd++) {
        S.p = gen(sd); S.seed = sd;
        const r = [1, 3, 5, 9].map((R) => evalRow(R, 0).res);
        if (new Set(r).size === 4) break;
      }
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Kernel', options: [{ v: 'split', label: 'split-K pelo lote' }, { v: 'chain', label: 'uma cadeia por linha' }, { v: 'fatom', label: 'atômicos float' }, { v: 'iatom', label: 'atômicos inteiros' }], value: 'split', onChange: (v) => { S.mode = v; S.run = 0; compute(); } });
      UI.slider(ctl, { label: 'linhas no lote (R)', min: 1, max: NR, step: 1, value: S.R, onInput: (v) => { S.R = v; compute(); } });
      UI.button(ctl, 'rodar de novo', () => { S.run++; compute(); });
      UI.button(ctl, 'novos números', () => { S.seed++; S.p = gen(S.seed); S.run = 0; compute(); });
      const rd = UI.readout(ctl, 'linha A × sozinha', '');
      compute();

      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 640, pad = 16;
        const PER = 5.5, cyc = Math.floor(t / PER), f = (t % PER) / PER;
        if (atom() && cyc !== S.lastCyc) { if (S.lastCyc >= 0) { S.run++; compute(); } S.lastCyc = cyc; }
        else if (!atom()) S.lastCyc = cyc;
        const cur = S.cur, ns = cur.ns, L = K / ns, R = S.R;
        const accU = V.ease.inOut(V.clamp(f / 0.42, 0, 1));
        const comb = V.clamp((f - 0.46) / 0.32, 0, 1);
        const colA = C.c, SEGC = [C.b, C.a, C.d];
        const segCol = (s) => SEGC[s % 3];
        const dSolo = ulps(cur.res, S.solo.res), same = dSolo === 0;
        // ---- geometria principal (o lote ocupa a coluna da esquerda até o fim do quadro da soma)
        const kh = 24, ph = 34, sgh = 58;
        const sqN = Math.min(16, (w - 2 * pad) / NR - 3);
        const bw = Math.min(150, w * 0.2), by = 40;
        const mx0 = narrow ? pad : pad + bw + 34, ky = narrow ? 34 + sqN + 46 : 48;
        const nPR = narrow && ns > 4 ? 4 : ns, pRowsN = Math.ceil(ns / nPR);
        const sy0 = ky + kh + 32 + pRowsN * (ph + 22) + (ns > 1 ? 26 : 8);
        // ---- o lote
        V.text(g, `LOTE · R = ${R} linha${R > 1 ? 's' : ''} na mesma chamada`, pad, 22, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        if (narrow) {
          const sq = sqN;
          for (let r = 0; r < NR; r++) {
            const x = pad + r * (sq + 3), on = r < R;
            V.box(g, x, 34, sq, sq, { r: 3, fill: r === 0 ? V.alpha(colA, 0.9) : on ? V.alpha(C.muted, 0.45) : 'transparent', stroke: r === 0 ? colA : V.alpha(C.muted, on ? 0.7 : 0.25), lw: 1 });
          }
          V.text(g, 'A', pad + sq / 2, 34 + sq / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: onFill(C), weight: 700 });
          if (accU > 0 && accU < 1) { const sx = pad + accU * R * (sq + 3); V.line(g, sx, 30, sx, 38 + sq, { color: V.alpha(C.ink, 0.6), lw: 1.5 }); }
        } else {
          const area = sy0 + sgh - by;
          const bh = Math.min(18, (area - (R - 1) * 4) / R);
          for (let r = 0; r < R; r++) {
            const y = by + r * (bh + 4);
            V.box(g, pad, y, bw, bh, { r: Math.min(5, bh / 2), fill: r === 0 ? V.alpha(colA, 0.85) : V.alpha(C.muted, 0.28), stroke: r === 0 ? colA : V.alpha(C.muted, 0.5), lw: 1 });
          }
          if (bh >= 11) V.text(g, 'linha A', pad + 8, by + bh / 2 + 1, { size: Math.min(12, bh - 1), font: 'mono', base: 'middle', color: onFill(C), weight: 700 });
          if (accU > 0 && accU < 1) { const sx = pad + accU * bw; V.line(g, sx, by - 4, sx, by + R * (bh + 4), { color: V.alpha(C.ink, 0.55), lw: 1.5 }); }
        }
        const mx1 = w - pad, mw = mx1 - mx0;
        V.text(g, `PRODUTO ESCALAR DA LINHA A · K = ${K} termos, soma em FP32`, mx0, ky - 12, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: mw });
        // ---- a barra de K, dividida em blocos
        const gap = 4, sw = (mw - (ns - 1) * gap) / ns;
        for (let s = 0; s < ns; s++) {
          const sx = mx0 + s * (sw + gap), col = segCol(s);
          V.box(g, sx, ky, sw, kh, { r: 5, fill: V.alpha(col, 0.13), stroke: V.alpha(col, 0.7), lw: 1 });
          if (accU > 0.01) V.box(g, sx, ky, Math.max(2, sw * accU), kh, { r: 5, fill: V.alpha(col, 0.5) });
          if (accU > 0 && accU < 1) V.line(g, sx + sw * accU, ky - 3, sx + sw * accU, ky + kh + 3, { color: C.ink, lw: 2 });
          const lab = sw > 92 ? `k ${s * L}…${(s + 1) * L - 1}` : sw > 30 ? String(s + 1) : '';
          if (lab) V.text(g, lab, sx + sw / 2, ky + kh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: C.ink, weight: 600 });
        }
        const capt = ns === 1 ? 'uma única cadeia: k = 0 → 511, sempre na mesma ordem' : narrow ? `${ns} blocos em paralelo, ${L} termos cada` : `${ns} blocos em paralelo, ${L} termos cada${S.mode === 'split' ? ` (com R = ${R}, a heurística escolhe ${ns})` : ''}`;
        V.text(g, capt, mx0, ky + kh + 18, { size: 11.5, color: C.text2 || C.muted, maxW: mw });
        // ---- parciais
        const py = ky + kh + 32;
        const perRow = nPR;
        const pw = narrow && ns > 4 ? (mw - (perRow - 1) * 6) / perRow : sw;
        const pgap = narrow && ns > 4 ? 6 : gap;
        const pRows = pRowsN;
        const pbox = (i) => ({ x: mx0 + (i % perRow) * (pw + pgap), y: py + Math.floor(i / perRow) * (ph + 22) });
        const rank = []; cur.order.forEach((i, k) => { rank[i] = k; });
        for (let i = 0; i < ns; i++) {
          const b = pbox(i), col = segCol(i), ready = accU >= 1;
          V.box(g, b.x, b.y, pw, ph, { r: 6, fill: V.alpha(col, ready ? 0.2 : 0.08), stroke: V.alpha(col, ready ? 0.9 : 0.4), lw: 1.2 });
          const txt = ns === 1 ? `acumulador = ${nice(cur.parts[i], 9)}` : nice(cur.parts[i], pw > 80 ? 7 : 5);
          V.text(g, txt, b.x + pw / 2, b.y + ph / 2 + 1, { size: 11.5, font: 'mono', align: 'center', base: 'middle', color: C.ink, alpha: ready ? 1 : 0.45, weight: 600, maxW: pw - 6 });
          if (ns > 1) V.text(g, `${rank[i] + 1}º`, b.x + pw / 2, b.y + ph + 14, { size: 11, font: 'mono', align: 'center', color: atom() ? C.d : C.muted, weight: 700 });
        }
        // ---- combinação
        const sy = sy0;
        const sgx = mx0, sgw = narrow ? mw : Math.min(mw * 0.56, 440);
        for (let i = 0; i < ns; i++) {
          const b = pbox(i), x1 = b.x + pw / 2, y1 = b.y + ph + (ns > 1 ? 18 : 2);
          const x2 = sgx + sgw * (i + 0.5) / ns, y2 = sy - 2;
          if (Math.floor(i / perRow) === pRows - 1) V.arrow(g, x1, y1, x2, y2, { color: V.alpha(segCol(i), 0.6), lw: 1.4, head: 6 });
          const k = rank[i], a0 = k / ns, a1 = (k + 1) / ns;
          if (comb > a0 && comb < a1) {
            const u = V.ease.inOut((comb - a0) / (a1 - a0)), bx0 = b.x + pw / 2, by0 = b.y + ph / 2;
            const dx = V.lerp(bx0, x2, u), dy = V.lerp(by0, y2, u);
            V.glow(g, dx, dy, 12, segCol(i)); V.dot(g, dx, dy, 5, segCol(i));
          }
        }
        const lit = f > 0.78;
        if (lit) V.glow(g, sgx + sgw / 2, sy + sgh / 2, Math.min(120, sgw / 2), same ? C.a : C.c, 0.25);
        V.box(g, sgx, sy, sgw, sgh, { r: 10, fill: C.panel || C.bg, stroke: lit ? (same ? C.a : C.c) : V.alpha(C.ink, 0.5), lw: lit ? 2.2 : 1.4 });
        const how = S.mode === 'iatom' ? `Σ inteiro de q = round(p · 2²⁴): ${bi(cur.q)}` : ns === 1 ? 'resultado = o acumulador da cadeia' : `Σ em FP32, na ordem ${cur.order.map((i) => i + 1).join('→')}`;
        V.text(g, how, sgx + 12, sy + 20, { size: 11.5, font: 'mono', color: S.mode === 'iatom' || atom() ? C.d : C.muted, weight: 600, maxW: sgw - 24 });
        V.text(g, `${nice(cur.res, 9)}  ·  ${hx8(cur.res)}`, sgx + 12, sy + 43, { size: 13.5, font: 'mono', color: C.ink, weight: 700, maxW: sgw - 24 });
        // ---- veredito
        const vx = narrow ? mx0 : sgx + sgw + 18, vy = narrow ? sy + sgh + 22 : sy + 4, vw = narrow ? mw : mx1 - vx;
        V.text(g, `sozinha (R = 1): ${hx8(S.solo.res)}`, vx, vy + 4, { size: 12, font: 'mono', color: C.muted, maxW: vw });
        V.text(g, `neste lote (R = ${R}): ${hx8(cur.res)}`, vx, vy + 22, { size: 12, font: 'mono', color: C.ink, maxW: vw });
        V.text(g, same ? '✓ mesmos bits' : `✗ ${dSolo === Infinity ? 'outro sinal' : dSolo + ' ulp de diferença'}`, vx, vy + 44, { size: 13, font: 'mono', color: same ? C.a : C.c, weight: 700, maxW: vw });
        // ---- faixa: a linha A em cada tamanho de lote
        const stTop = narrow ? vy + 70 : sy + sgh + 34;
        V.text(g, 'A LINHA A EM CADA TAMANHO DE LOTE, COMPARADA COM R = 1', pad, stTop, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        const per = narrow ? 8 : NR, cg = 4, cw = (w - 2 * pad - (per - 1) * cg) / per, chh = 40;
        for (let r = 1; r <= NR; r++) {
          const i = r - 1, x = pad + (i % per) * (cw + cg), y = stTop + 10 + Math.floor(i / per) * (chh + cg);
          const d = S.strip[i], ok = d === 0, col = ok ? C.a : C.c, here = r === R;
          V.box(g, x, y, cw, chh, { r: 6, fill: V.alpha(col, here ? 0.3 : 0.14), stroke: here ? C.ink : V.alpha(col, 0.7), lw: here ? 2.2 : 1 });
          V.text(g, `R ${r}`, x + cw / 2, y + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
          V.text(g, ok ? '✓' : d === Infinity ? '±' : `${d}ulp`, x + cw / 2, y + 31, { size: ok ? 13 : 11, font: 'mono', align: 'center', color: col, weight: 700, maxW: cw - 4 });
        }
        fitH(fig, stTop + 10 + Math.ceil(NR / per) * (chh + cg) + 8);
      });
    }

    /* ================================================================
       Figura 13 — a escala que só mexe no expoente (KV FP8 do TensorFold)
       ================================================================ */
    {
      const fig = $('#fig-escala2');
      const W = 512, SHOW = 32, AMI = 11, SHIFT = 8;
      const S = { seed: 1, k: 0, kind: 'pow2', base: null, R: null, focus: 5, pinned: false, lastStep: -1 };
      const hx8 = (u) => '0x' + (u >>> 0).toString(16).toUpperCase().padStart(8, '0');
      const gen = (seed) => {
        const rg = V.rng(9001 * seed + 5), x = new Float64Array(W);
        for (let i = 0; i < W; i++) x[i] = rB(rg.normal() * 0.7);
        let m = 0; for (let i = 0; i < W; i++) if (i !== AMI) m = Math.max(m, Math.abs(x[i]));
        x[AMI] = rB((rg() < 0.5 ? -1 : 1) * m * (1.15 + 0.3 * rg()));
        return x;
      };
      // quantiza a linha com uma das duas escalas; devolve tudo o que a figura mostra
      const quant = (x, kind) => {
        let amax = 0; for (let i = 0; i < W; i++) amax = Math.max(amax, Math.abs(x[i]));
        const ab = u32(amax), E = (ab >>> 23) & 255, M = ab & 0x7FFFFF;
        const e = E + (M !== 0 ? 1 : 0), se = Math.min(254, Math.max(1, e - SHIFT));
        let s, inv;
        if (kind === 'pow2') { s = f32(se << 23); inv = f32((254 - se) << 23); } else { s = r32(amax / 448); inv = null; }
        const c = new Float64Array(W), xh = new Float64Array(W);
        let nSub = 0, nRound = 0, maxC = 0, se2 = 0, sx2 = 0;
        for (let i = 0; i < W; i++) {
          const y = kind === 'pow2' ? x[i] * inv : r32(x[i] / s);
          if (y !== 0 && Math.abs(y) < 2 ** -6) nSub++;
          c[i] = N.round('e4m3', y);
          const prod = c[i] * s;                    // exato em float64 (4 + 24 bits)
          xh[i] = r32(prod);
          if (xh[i] !== prod) nRound++;
          maxC = Math.max(maxC, Math.abs(c[i]));
          se2 += (x[i] - xh[i]) ** 2; sx2 += x[i] ** 2;
        }
        return { amax, ab, E, M, e, se, s, inv, c, xh, nSub, nRound, maxC, err: Math.sqrt(se2 / sx2) };
      };
      const compute = () => {
        const x = S.base.map((v) => v * 2 ** S.k);
        S.x = x;
        S.R = quant(x, S.kind);
        S.Ralt = quant(x, S.kind === 'pow2' ? 'q448' : 'pow2');
        const R0 = quant(S.base, S.kind);
        S.sameCodes = S.R.c.every((v, i) => v === R0.c[i]);
        rdMax.set(nice(S.R.maxC, 4), S.R.maxC <= 256 ? 'ok' : 'warn');
        rdErr.set(pct(S.R.err));
      };
      S.base = gen(S.seed);
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Escala', options: [{ v: 'pow2', label: '2ᵏ (TensorFold)' }, { v: 'q448', label: 'amax/448 (FP32)' }], value: 'pow2', onChange: (v) => { S.kind = v; compute(); } });
      UI.slider(ctl, { label: 'linha × 2ᵏ', min: -6, max: 6, step: 1, value: 0, fmt: (v) => '×2' + sup(v), onInput: (v) => { S.k = v; compute(); } });
      UI.button(ctl, 'nova linha', () => { S.seed++; S.base = gen(S.seed); S.pinned = false; compute(); });
      const rdMax = UI.readout(ctl, 'maior código', '');
      const rdErr = UI.readout(ctl, 'erro da linha', '');
      const st = UI.stepper(fig, [
        { t: 'A escala sai dos bits do amax', d: 'O kernel lê o maior \\(|x|\\) da linha como inteiro: o campo de expoente \\(E\\), mais 1 se a mantissa não for zero, já é \\(\\lceil \\log_2 \\text{amax}\\rceil\\) (com o bias), sem nenhum logaritmo. Subtrai <code>SHIFT = 8</code> e devolve o resultado ao campo de expoente: \\(s\\) é uma potência de dois pura, com a mantissa toda zero. O recíproco \\(1/s\\) sai do campo \\(254 - e\\).' },
        { t: 'Códigos E4M3: o máximo cai em (128, 256]', d: 'Cada valor é multiplicado por \\(1/s\\), o que é exato, e arredondado para o E4M3 mais próximo, que é o único arredondamento da conta. O maior código fica entre 128 e 256, longe do 448 que saturaria. Os tracinhos à direita são os valores do E4M3: no eixo log₂, cada binade normal tem os mesmos 8. Na escala amax/448, a própria divisão já arredonda e o maior código vai até 448.' },
        { t: 'Desquantizar: a potência de dois só mexe no expoente', d: 'O leitor converte o código para BF16, o que é exato, e multiplica pela escala dentro dos produtos FP32. Multiplicar por \\(2^k\\) soma \\(k\\) ao campo de expoente e deixa a mantissa intacta, então nenhum bit arredonda. Com \\(s = \\text{amax}/448\\) em FP32, o mesmo produto pode precisar de até 4 + 24 = 28 bits significativos, e aí arredonda.' },
      ], { autoplay: 7000 });
      compute();

      // uma fileira de bits com os campos coloridos; devolve as posições de cada campo
      const bitRow = (g, C, u, x, y, cw, ch, fields, opt = {}) => {
        const n = fields.reduce((a, f) => a + f.n, 0);
        let k = 0, xx = x; const pos = [];
        fields.forEach((fd, fi) => {
          const x0 = xx;
          for (let j = 0; j < fd.n; j++, k++) {
            const bit = n - 1 - k, on = (u >>> bit) & 1;
            const hl = opt.hi === fi, dim = opt.dim === fi;
            V.box(g, xx, y, cw - 1.5, ch, { r: 3, fill: on ? V.alpha(fd.col, dim ? 0.35 : 0.88) : V.alpha(fd.col, dim ? 0.05 : 0.12), stroke: V.alpha(fd.col, hl ? 1 : 0.55), lw: hl ? 1.8 : 1 });
            if (cw >= 12) V.text(g, on, xx + (cw - 1.5) / 2, y + ch / 2 + 1, { size: Math.min(12, cw * 0.62), font: 'mono', align: 'center', base: 'middle', color: on ? onFill(C) : C.muted, weight: 600 });
            xx += cw;
          }
          pos.push([x0, xx - 1.5]);
          xx += fi < fields.length - 1 ? 5 : 0;
        });
        return pos;
      };

      const H = V.fig(fig, (g, w, h, t) => {
        const C = V.C(), narrow = w < 640, pad = 16, i = st.i, el = st.elapsed();
        const Rq = S.R, x = S.x, pow2 = S.kind === 'pow2';
        // ---- painel das barras: |x| / s em escala log₂ (cada binade tem a mesma altura)
        const axL = narrow ? 34 : 44, axR = narrow ? 34 : 46;
        const px0 = pad + axL, px1 = w - pad - axR, pw = px1 - px0;
        const py = 40, ph = narrow ? 140 : 170, LO = -7, HI = 9;
        const Y = (c) => py + ph - V.clamp((Math.log2(Math.max(c, 2 ** LO)) - LO) / (HI - LO), 0, 1) * ph;
        V.text(g, `LINHA DO CACHE · |x| / s de ${SHOW} dos ${W} valores (ilustrativos), eixo log₂`, pad, 22, { size: 11, font: 'mono', color: C.muted, weight: 600, maxW: w - 2 * pad });
        // grade das binades
        for (let e = LO + 1; e < HI; e++) V.line(g, px0, Y(2 ** e), px1, Y(2 ** e), { color: C.grid, lw: 1 });
        [[-6, '2⁻⁶'], [0, '1'], [4, '16']].forEach(([e, lab]) => V.text(g, lab, px0 - 6, Y(2 ** e) + 4, { size: 11, font: 'mono', color: C.muted, align: 'right' }));
        // régua do E4M3 à direita: um tracinho por valor representável (8 por binade nos normais)
        N.values('e4m3').filter((v) => v > 0).forEach((v) => V.line(g, px1 + 5, Y(v), px1 + (v >= 2 ** -6 ? 13 : 9), Y(v), { color: V.alpha(v >= 2 ** -6 ? C.ink : C.c, 0.45), lw: 1 }));
        V.text(g, 'E4M3', px1 + 16, py + ph - 2, { size: 11, font: 'mono', color: C.muted });
        if (i >= 1) {
          g.fillStyle = V.alpha(C.c, 0.08); g.fillRect(px0, Y(2 ** -6), pw, py + ph - Y(2 ** -6));
          g.fillStyle = V.alpha(C.d, 0.14); g.fillRect(px0, Y(256), pw, Y(128) - Y(256));
          V.line(g, px0, Y(448), px1, Y(448), { color: C.red || C.c, lw: 1.4 });
          V.text(g, '448: satura', px0 + 4, Y(448) - 5, { size: 11, font: 'mono', color: C.red || C.c, weight: 600 });
        } else {
          const ca = Rq.amax * (pow2 ? Rq.inv : 1 / Rq.s);
          V.line(g, px0, Y(ca), px1, Y(ca), { color: V.alpha(C.c, 0.7), lw: 1, dash: [4, 4] });
          V.text(g, `amax = ${nice(Rq.amax, 5)}`, px0 + 4, Y(ca) - 6, { size: 11.5, font: 'mono', color: C.c, weight: 600 });
        }
        V.line(g, px0, py + ph, px1, py + ph, { color: V.alpha(C.muted, 0.6), lw: 1 });
        const bwid = pw / SHOW, P = H.pointer;
        if (P.clicked && P.x >= px0 && P.x <= px1 && P.y >= py && P.y <= py + ph) { S.focus = V.clamp(Math.floor((P.x - px0) / bwid), 0, SHOW - 1); S.pinned = true; P.clicked = false; }
        H.canvas.style.cursor = P.inside && P.x >= px0 && P.x <= px1 && P.y >= py && P.y <= py + ph ? 'pointer' : 'default';
        if (!S.pinned && i === 2) S.focus = [5, AMI, 2, 20, 27, 8][Math.floor(el / 2.6) % 6];
        const morph = i === 0 ? 0 : i === 1 ? V.ease.inOut(V.clamp(el / 0.9, 0, 1)) : 1;
        for (let j = 0; j < SHOW; j++) {
          const raw = Math.abs(pow2 ? x[j] * Rq.inv : x[j] / Rq.s), q = Math.abs(Rq.c[j]);
          const c = V.lerp(raw, q, morph);
          const isF = i === 2 && j === S.focus, col = j === AMI ? C.c : isF ? C.d : C.b;
          const bx = px0 + j * bwid + bwid * 0.15, bw = bwid * 0.7;
          V.box(g, bx, Y(c), bw, py + ph - Y(c), { r: Math.min(3, bw / 2), fill: V.alpha(col, isF ? 0.95 : 0.7) });
          if (morph > 0.5) V.line(g, bx - 1, Y(raw), bx + bw + 1, Y(raw), { color: V.alpha(C.ink, 0.5), lw: 1 });
        }
        if (i >= 1) {
          const lab = 'subnormais do E4M3'; g.font = V.font(11, 'mono', 400); const lw = g.measureText(lab).width;
          V.box(g, px1 - lw - 12, py + ph - 18, lw + 8, 15, { r: 4, fill: V.alpha(C.panel || C.bg, 0.92) });
          V.text(g, lab, px1 - 8, py + ph - 6, { size: 11, font: 'mono', color: C.c, align: 'right' });
          g.font = V.font(11, 'mono', 700); const lw2 = g.measureText('(128, 256]').width;
          V.box(g, px1 - lw2 - 10, Y(181) - 7, lw2 + 8, 14, { r: 4, fill: V.alpha(C.panel || C.bg, 0.85) });
          V.text(g, '(128, 256]', px1 - 6, Y(181) + 4, { size: 11, font: 'mono', color: C.d, align: 'right', weight: 700 });
        }
        // ---- painel de detalhe (y = linha de base do próximo texto)
        const tx = pad, tw = w - 2 * pad;
        let y = py + ph + 30;
        const say = (s, col, wt = 400, size = 12) => { const nl = wrap(g, s, tx, y, tw, size + 5, { size, font: 'mono', color: col, weight: wt }); y += nl * (size + 5) + 5; };
        const F32 = [{ n: 1, col: C.c }, { n: 8, col: C.d }, { n: 23, col: C.a }];
        const BF = [{ n: 1, col: C.c }, { n: 8, col: C.d }, { n: 7, col: C.a }];
        const cw32 = Math.min(19, (tw - 10) / 32), cw16 = Math.min(24, (tw - 10) / 16), chh = narrow ? 18 : 22;
        const bits = (u, fields, cw, opt) => { const top = y - 8; const p = bitRow(g, C, u, tx, top, cw, chh, fields, opt); y = top + chh + 20; return { p, top }; };
        if (i === 0) {
          if (pow2) {
            say(`amax = ${nice(Rq.amax, 7)}  ·  FP32 ${hx8(Rq.ab)}`, C.ink, 700, 12.5);
            const A = bits(Rq.ab, F32, cw32, { hi: 1 });
            V.text(g, `E = ${Rq.E}`, (A.p[1][0] + A.p[1][1]) / 2, y - 3, { size: 11.5, font: 'mono', color: C.d, align: 'center', weight: 700 });
            V.text(g, Rq.M !== 0 ? 'M ≠ 0 → +1' : 'M = 0 → +0', (A.p[2][0] + A.p[2][1]) / 2, y - 3, { size: 11.5, font: 'mono', color: C.a, align: 'center', weight: 700, maxW: A.p[2][1] - A.p[2][0] });
            y += 20;
            if (narrow) { say(`e = ${Rq.E} + ${Rq.M !== 0 ? 1 : 0} = ${Rq.e}`, C.ink, 600); say(`campo da escala = e − 8 = ${Rq.se}`, C.ink, 600); }
            else say(`e = ${Rq.E} + ${Rq.M !== 0 ? 1 : 0} = ${Rq.e}  →  campo da escala = e − 8 = ${Rq.se}`, C.ink, 600);
            const B = bits(Rq.se << 23, F32, cw32, { hi: 1, dim: 2 });
            const pulse = 0.5 + 0.5 * Math.sin(t * 3);
            g.globalAlpha = 0.3 + 0.55 * pulse; g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, B.p[1][0] - 3, B.top - 3, B.p[1][1] - B.p[1][0] + 6, chh + 6, 5); g.stroke(); g.globalAlpha = 1;
            say(`s = 2${sup(Rq.se - 127)} = ${nice(Rq.s, 6)}: mantissa toda zero, uma potência de dois pura`, C.a, 600);
            say(`1/s = 2${sup(127 - Rq.se)}: campo 254 − ${Rq.se} = ${254 - Rq.se}, também exato`, C.muted);
          } else {
            const sb = u32(Rq.s);
            say(`s = amax / 448 = ${nice(Rq.s, 8)}  ·  FP32 ${hx8(sb)}`, C.ink, 700, 12.5);
            bits(sb, F32, cw32, { hi: 2 });
            say('mantissa cheia de bits: s não é uma potência de dois', C.c, 600);
            say('dividir por s arredonda, e código × s também pode arredondar', C.muted);
          }
        } else if (i === 1) {
          say(`maior código: ${nice(Rq.maxC, 4)}  →  ${pow2 ? 'dentro de (128, 256]' : 'encosta no 448'}; nada satura`, C.ink, 700, 12.5);
          say(`valores abaixo de 2⁻⁶ (subnormais do E4M3): ${Rq.nSub} de ${W}`, C.muted);
          say(`erro relativo da linha: 2ᵏ ${pct(pow2 ? Rq.err : S.Ralt.err)}  ·  amax/448 ${pct(pow2 ? S.Ralt.err : Rq.err)}`, C.d, 700);
          say(S.k === 0 ? 'mova “linha × 2ᵏ”: os códigos não mudam, só a escala' : `linha × 2${sup(S.k)}: ${S.sameCodes ? 'os mesmos 512 códigos, só a escala mudou' : 'alguns códigos mudaram (underflow)'}`, S.k === 0 || S.sameCodes ? C.a : C.c, 600);
          say('bytes por linha: 512 códigos + 4 de escala + 12 de zeros = 528 B (em BF16: 1.024 B)', C.muted);
        } else {
          const j = S.focus, cv = Rq.c[j], cb = u32(cv) >>> 16;
          say(`x[${j}] = ${nice(x[j], 6)}  →  código E4M3 = ${nice(cv, 4)}, em BF16 (exato):`, C.ink, 700, 12.5);
          bits(cb, BF, cw16, { hi: 1 });
          if (pow2) {
            const kk = Rq.se - 127, prod = cv * Rq.s, rb = u32(prod) >>> 16;
            say(`× s = 2${sup(kk)}  →  o campo de expoente anda ${bi(kk)}`, C.d, 700);
            const u = V.ease.inOut(V.clamp(el / 1.4, 0, 1)), Ecur = Math.round(V.lerp((cb >>> 7) & 255, (rb >>> 7) & 255, u));
            bits(cv === 0 ? 0 : (cb & 0x807F) | (Ecur << 7), BF, cw16, { hi: 1 });
            say(cv === 0 ? 'zero continua zero' : `= ${nice(prod, 6)}: mantissa idêntica, só o expoente andou`, C.a, 700);
            say(`nos ${W} valores: ${Rq.nRound} produtos precisaram arredondar`, C.muted);
          } else {
            const prod = cv * Rq.s, pr = r32(prod), rnd = pr !== prod;
            say(`× s = ${nice(Rq.s, 8)} (FP32, 24 bits significativos)`, C.d, 700);
            say(`produto exato: ${nice(prod, 12)}`, C.ink);
            say(`em FP32: ${nice(pr, 9)}  ·  ${hx8(u32(pr))}  ${rnd ? '→ arredondou' : '→ coube desta vez'}`, rnd ? C.c : C.a, 700);
            say(`nos ${W} valores: ${Rq.nRound} produtos precisaram arredondar`, C.c, 600);
          }
        }
        fitH(fig, Math.max(y + 4, narrow ? 300 : 330));
      });
    }
  },
});

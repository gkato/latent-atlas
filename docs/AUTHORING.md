# Latent Atlas — Guia de autoria de capítulos

Este é o contrato para escrever (ou estender) um capítulo. Leia inteiro antes de começar.
Exemplo completo e funcional: `chapters/00-mapa-da-jornada.html` + `chapters/00-mapa-da-jornada.js`.

---

## 1. Missão e público

O leitor quer **conseguir criar sozinho receitas de inferência** como as da Mia (servir LLMs MoE de
fronteira em 2 nós com vLLM), entendendo **em detalhe** cada conceito. Ele pode não saber deep
learning: quando um conceito depende de fundamentos, desça até eles (ou aponte o capítulo que os
explica, via pré-requisito/link).

- **Ciência primeiro, repositório como exemplo.** O texto explica o conceito (o que é, por que
  existe, como funciona, matemática, trade-offs). As receitas entram como *exemplos concretos*
  (“na receita GLM isso vira `--kv-cache-dtype fp8_ds_mla` porque…”), nunca como um tour do repo.
- **Lúdico e rigoroso.** Analogias, metáforas visuais, pequenos exemplos numéricos feitos à mão,
  figuras que se mexem — e depois o formalismo correto (equações com legenda de cada símbolo).
- **Ordem cronológica.** Cada capítulo só usa conceitos de capítulos anteriores (ou os explica).
  Termine cada capítulo preparando o próximo.
- **Português do Brasil**, tom de professor entusiasmado e preciso. Termos técnicos em inglês
  ficam em inglês quando é assim que a comunidade usa (KV cache, prefill, top-k, all-reduce…),
  explicados na primeira ocorrência.
- **Nunca invente números.** Valores de parâmetros, benchmarks, formas de tensores e caminhos de
  arquivo vêm dos inventários em `…/scratchpad/inventory/*.md` (ou do próprio repo). Se algo é
  interpretação, diga “interpretação”. Exemplos didáticos inventados devem ser rotulados como
  ilustrativos.

## 2. Arquivos e contrato

Cada capítulo = dois arquivos (+1 opcional) em `chapters/`:

| Arquivo | Conteúdo |
|---|---|
| `NN-slug.html` | O corpo do capítulo (HTML puro, sem `<script>`). |
| `NN-slug.js` | `LA.chapter({...})` com metadados, quiz, resumo e `mount()` que anima as figuras. |
| `NN-slug.glossary.json` | (opcional, recomendado) termos novos que este capítulo introduz. |

- `NN` = número do capítulo com 2 dígitos (define a ordem). `slug` = id (kebab-case, ASCII).
- O `id` em `LA.chapter` **tem que ser igual ao slug** do nome do arquivo.
- `order` = NN (número).
- Build: `node build.js` (valida sintaxe e gera `dist/chapters.bundle.js`).
- QA: `node tools/qa.js --shots <id>` (renderiza no Chrome headless, mexe em todos os controles,
  lista erros e salva screenshots de cada figura em `$QA_OUT/<id>/fig-N.png`). **Abra os PNGs e
  olhe** — é assim que você confere se a figura está bonita e legível.

### 2.1 `LA.chapter({...})`

```js
LA.chapter({
  id: 'attention',                 // = slug do arquivo
  part: 2,                         // número da parte (data/manifest.js)
  order: 7,                        // = NN
  title: 'Atenção: como tokens conversam',
  navTitle: 'Atenção',             // curto, para a barra lateral
  subtitle: 'Uma frase que vende o capítulo.',
  minutes: 35,
  recipes: ['glm', 'qwen', 'ds'],  // receitas que aparecem como exemplo
  tags: ['atenção', 'softmax', 'QKV'],
  goals: ['…', '…', '…'],          // 3–5 itens (HTML permitido)
  prereqs: ['tokens-embeddings'],  // ids de capítulos anteriores
  summary: ['…', '…'],             // 4–7 bullets (HTML permitido)
  quiz: [                          // 4–7 perguntas
    { q: 'Pergunta?', o: ['A', 'B', 'C', 'D'], a: 1, e: 'Explicação da resposta certa (e por que as outras erram).' },
  ],
  mount(root, LA) {                // root = <article> já com o HTML
    // cria as animações; tudo que usa LA.viz/LA.ui é limpo automaticamente ao sair
  },
});
```

Strings JS com LaTeX precisam de barra dupla: `'\\( x^2 \\)'`. No `.html` a barra é simples.

## 3. Estrutura do HTML

```html
<p class="la-lead">Abertura: um gancho concreto e o que o leitor vai conquistar.</p>

<section class="la-sec">
  <h2>Título da seção</h2>          <!-- numeração, âncoras e sumário são automáticos -->
  <p>…</p>
  <figure class="la-fig" id="fig-algo" data-title="Título curto da figura">
    <div class="la-fig-stage" style="height:360px"><canvas></canvas></div>
    <div class="la-fig-controls"></div>          <!-- preenchido no mount() -->
    <figcaption>O que olhar na figura e o que ela prova.</figcaption>
  </figure>
</section>
```

Regras:
- **5 a 10 seções `<section class="la-sec">` com `<h2>`**; `<h3>` para subseções.
- **Toda seção tem pelo menos uma figura dinâmica** (canvas animado, CSS 3D, SVG animado ou
  widget interativo). O pedido do leitor é explícito: *sempre* imagens/figuras móveis, sobretudo
  para matrizes, vetores e fluxos. Meta: **6–10 figuras por capítulo**, das quais ≥ 4 interativas
  (controles que mudam a figura).
- Tamanho-alvo: **3.500–7.000 palavras** de texto útil por capítulo (sem enrolação).
- Termine com uma seção “Nas receitas” (ou equivalente) que costura o conceito aos parâmetros
  reais das três receitas, e uma frase-ponte para o próximo capítulo.

### 3.1 Componentes disponíveis (CSS pronto)

| Componente | Marcação |
|---|---|
| Parágrafo de abertura | `<p class="la-lead">` |
| Callouts | `<div class="la-callout KIND"><b class="t">Título</b><p>…</p></div>` com KIND ∈ `tip` `info` `warn` `danger` `analogy` `deep` (aprofundamento) `why` (por que importa) `math` `recipe` |
| Callout de receita | `<div class="la-callout recipe" data-recipe="glm"><b class="t"><span class="la-rb glm"></span> Na receita</b><p>…</p></div>` |
| Cartão “Na receita” com parâmetro | `<div class="la-inrecipe" data-recipe="qwen"><div class="la-inrecipe-h"><span class="la-rb qwen"></span><span class="k">Parâmetro</span><code>--max-num-seqs 8</code></div><div class="la-inrecipe-b"><p>…</p></div></div>` |
| Selo de receita inline | `<span class="la-rb glm"></span>` `<span class="la-rb qwen"></span>` `<span class="la-rb ds"></span>` `<span class="la-rb all"></span>` (texto automático) |
| Equação com legenda | `<div class="la-eq">\[ … \]<ul class="la-eq-legend"><li>\(x\): …</li></ul></div>` |
| Matemática inline | `\( … \)` (KaTeX). Não use `$…$` simples. |
| Código | `<pre data-lang="bash"><code>…</code></pre>` (escape `<` `>` `&` como `&lt;` `&gt;` `&amp;`) |
| Tabela | `<table>` normal; tabela de parâmetros: `<table class="la-params">` (colunas sugeridas: Parâmetro · Valor · Receita · O que faz · Por que importa) |
| Comparação lado a lado | `<div class="la-compare"><div data-recipe="glm"><h4>GLM</h4>…</div>…</div>` |
| Cartões | `<div class="la-cards"><div class="la-card"><span class="ic">🧠</span><h4>…</h4><p>…</p></div></div>` |
| Passos numerados | `<ol class="la-steps"><li><b>Título</b> texto…</li></ol>` |
| Chave/valor | `<dl class="la-kv"><dt>nome</dt><dd>significado</dd></dl>` |
| Termo do glossário | `<span class="la-term" data-term="kv-cache">KV cache</span>` (a chave precisa existir no glossário — adicione no seu `NN-slug.glossary.json`) |
| Link para capítulo | `<a data-ch="moe" data-sec="">texto</a>` (href automático) |
| Revelar ao rolar | adicione a classe `la-reveal` a um bloco |
| Figura larga | `<figure class="la-fig wide">` |

### 3.2 Glossário por capítulo — `NN-slug.glossary.json`

```json
{
  "kv-cache": { "t": "KV cache", "d": "Memória onde o modelo guarda K e V de tokens já vistos…" },
  "softmax": { "t": "Softmax", "d": "Função que transforma notas em probabilidades: \\( e^{x_i}/\\sum_j e^{x_j} \\)." }
}
```
Chaves em kebab-case. Definições curtas (1–3 frases), HTML simples permitido. O build associa o
termo ao seu capítulo. Todo `data-term` usado no HTML precisa existir aqui ou em outro capítulo.

## 4. Figuras — API de visualização

Tudo em `window.LA`. Cleanup automático ao trocar de página. Leia `assets/js/viz.js` e
`assets/js/num.js` para detalhes.

### 4.1 Canvas animado
```js
const fig = root.querySelector('#fig-algo');
const H = LA.viz.fig(fig, (g, w, h, t, dt, S, H) => {
  const C = LA.viz.C();              // cores do tema ATUAL (leia a cada frame)
  // g: CanvasRenderingContext2D já escalado para DPR; w,h em px CSS; t em segundos
  LA.viz.box(g, 20, 20, 120, 40, { r: 10, fill: LA.viz.alpha(C.a, .2), stroke: C.a, label: 'Q' });
}, { static: false });                // static:true = só redesenha quando você chamar H.redraw()
```
- `H.pointer` → `{x, y, down, inside, clicked}` (interação com o mouse no canvas).
- `H.S` é um objeto livre para estado; `H.redraw()` força um frame (use após mudar controles em figuras `static`).
- Animações pausam fora da tela, com a aba oculta e com o botão global ⏸ (respeite `LA.paused` se usar timers próprios; prefira `LA.viz.every(ms, fn)`).

Cores (`LA.viz.C()`): `a` (menta), `b` (lavanda), `c` (rosa), `d` (âmbar), `e` (verde), `f` (magenta),
`ink` (texto), `muted`, `grid`, `bg`, `panel`, `glm`, `qwen`, `ds`, `red`, `series[]`, `dark` (bool).
**Nunca** fixe cores de fundo/texto em hex — use `C.*` para funcionar nos temas claro e escuro.

Primitivas: `LA.viz.rr` (retângulo arredondado), `box`, `text(g, s, x, y, {size, font:'sans'|'mono'|'display', weight, color, align, base, maxW, alpha})`,
`line`, `arrow(g, x1,y1,x2,y2, {color, lw, head, curve, dash})`, `glow`, `dot`, `along`, `bez`, `grid`,
`matrix(g, {x, y, data, cell | w,h, cmap:'seq'|'div'|'heat', values, digits, hiRow, hiCol, hiCells, rowLabels, colLabels, colorFn})`
(retorna geometria com `cellAt(px,py)` e `center(i,j)`), `bars(g, {x,y,w,h, values, labels, hi, showValues})`,
`cmap(v, name)`, `alpha(color, a)`, `mix(c1, c2, t)`.

Utilidades: `clamp`, `lerp`, `smooth`, `ease.{inOut,out,in,back,elastic}`, `pingpong(t,p)`,
`phase(t,p)`, `rng(seed)` (com `.normal()`, `.int(n)`, `.pick(arr)`), `softmax(xs, temp)`,
`randMatrix(R,K,seed)`, `matmul`, `transpose`, `fmt`, `bytes`.

### 4.2 Controles (DOM)
```js
const ctl = LA.ui.controls(fig);   // <div class="la-fig-controls">
const s = LA.ui.slider(ctl, { label: 'Temperatura', min: 0.1, max: 2, step: 0.1, value: 1, fmt: v => v.toFixed(1), onInput: v => { S.temp = v; H.redraw(); } });
LA.ui.seg(ctl, { label: 'Formato', options: ['FP16', 'FP8', 'FP4'], value: 'FP8', onChange: v => … });
LA.ui.seg(ctl, { options: [{ v: 2, label: 'TP=2' }, { v: 3, label: 'TP=3' }], value: 2, onChange: … });
LA.ui.toggle(ctl, { label: 'mostrar máscara causal', value: true, onChange: v => … });
LA.ui.select(ctl, { label: 'Modelo', options: [{ v: 'glm', label: 'GLM' }], value: 'glm', onChange: … });
LA.ui.button(ctl, '▶ rodar', () => …, { primary: true });
const r = LA.ui.readout(ctl, 'KV/token', '—'); r.set('584 B');
LA.ui.legend(ctl, [[C.a, 'Nó A'], [C.b, 'Nó B']]);
```

### 4.3 Explicação passo a passo sincronizada
```js
const st = LA.ui.stepper(fig, [ { t: 'Projeção Q', d: 'Multiplicamos X por W<sub>Q</sub>…' }, … ], { autoplay: 5000, onStep: (i) => H.redraw() });
// dentro do draw: st.i (passo atual) e st.elapsed() (segundos desde que o passo começou) → anime a transição
```

### 4.4 Tensor 3D em CSS
```js
const T = LA.ui.tensor3d(fig.querySelector('.la-fig-stage'), { slabs: 4, rows: 6, cols: 8, cell: 16, depthGap: 34, rotate: true,
  color: (s, i, j) => …, slabColor: (s) => …, labels: ['cabeça 0', …] });
T.setCell(s, i, j, cor); T.explode(1.6); T.rotate(-20, 35); T.highlightSlab(2);
```

### 4.5 SVG + CSS
SVG inline no HTML pode usar as classes de animação: `la-anim-dash` (tracejado correndo),
`la-anim-pulse`, `la-anim-float`, `la-anim-spin`, `la-anim-glow`, `la-anim-wiggle`. Use
`stroke="var(--viz-a)"`/`fill="var(--viz-b)"` para respeitar o tema.

### 4.6 Aritmética numérica (`LA.num`)
Formatos: `fp32 fp16 bf16 e4m3 e5m2 e3m2 e2m3 e2m1 e8m0`.
`LA.num.encode(fmt, x)` → `{bits, v, err}` (RNE, saturando) · `decode(fmt, bits)` · `fields(fmt, bits)` →
`{s, E, M}` · `values(fmt)` (todos os valores representáveis) · `maxFinite` · `minSub` · `minNormal` ·
`toBin(bits, n)` · `quantInt(xs, {bits, symmetric, group})` · `mx(xs, 'e2m1'|'e4m3', 32)` (MXFP4/MXFP8) ·
`nvfp4(xs)` · `fp8Tensor(xs)` · `fwht(xs)` (Hadamard) · `hadamard(n)` · `randSigns(n)` ·
`synthWeights(n, seed, outliers)` · `errStats(xs, ys)` (mse, rmse, maxErr, sqnr).
Use essas funções em vez de reimplementar — elas estão testadas.

### 4.7 Qualidade visual
- Figuras grandes e legíveis: altura 300–480px; texto no canvas ≥ 11px; contraste bom nos 2 temas.
- Mostre **valores reais** quando possível (números nas células das matrizes, formas `[4096 × 2048]`).
- Movimento com propósito: fluxo de dados, transformação, antes/depois, passo a passo.
- Performance: evite alocar arrays grandes por frame; pré-compute em `mount`; ≤ ~3 mil primitivas por frame.
- Responsivo: use `w`/`h` do draw para layout; nada de coordenadas fixas que estouram em 390px.
- Tudo deve funcionar sem rede (exceto fontes/KaTeX). Não carregue bibliotecas externas.

## 5. Cobertura das receitas

Os inventários descrevem **cada** parâmetro e modificação das receitas. Seu capítulo recebe uma
lista de itens para explicar. Para cada item: diga o que é, o valor em cada receita, o conceito
por trás e por que o valor escolhido faz sentido (trade-off). Tabelas `la-params` e cartões
`la-inrecipe` são o formato ideal. Não deixe item da sua lista sem explicação.

## 6. Checklist antes de entregar
- [ ] `node build.js --check` sem erros; `node tools/qa.js --shots <id>` com ✓ e sem issues.
- [ ] Abri todos os `fig-N.png` e as figuras estão bonitas, legíveis e corretas.
- [ ] Toda seção tem figura dinâmica; ≥ 6 figuras no total.
- [ ] Todos os `data-term` existem no glossário; todos os `data-ch` apontam para ids existentes.
- [ ] Números conferidos com os inventários; nada inventado.
- [ ] Quiz (4–7), resumo (4–7), goals (3–5), prereqs corretos.

## 7. Armadilhas conhecidas (retorno da primeira rodada de autores)
- Dentro de `\( … \)` no HTML, escreva `&lt;` em vez de `<` (senão o navegador interpreta como tag e a fórmula não renderiza).
- KaTeX gera um MathML oculto `position:absolute`; os blocos da plataforma já são `position:relative`, mas evite fórmulas muito largas em tabelas no mobile.
- Texto em canvas não quebra linha sozinho: use `LA.viz.wrap(g, texto, x, y, maxW, {size, lineH, maxLines})`.
- A fonte mono tem ligaduras; no DOM elas estão desligadas, e `LA.viz.text` com `font:'mono'` já as neutraliza.
- `LA.ui.readout(...).set(valor, 'ok'|'bad'|'warn')` colore o resultado; `LA.ui.group(ctl).show(bool)` mostra/esconde grupos de controles por modo.
- Tabelas e selos criados dentro do `mount()` recebem o mesmo tratamento das estáticas (`LA.enhance`).

## 8. Idiomas (PT e EN)
O conteúdo-fonte é em português. Ao criar ou alterar um capítulo, atualize também a versão inglesa
(`NN-slug.en.html`, `.en.js`, `.en.glossary.json`) seguindo `docs/I18N.md`, e rode
`node tools/i18n-check.js <id>` e `node tools/qa.js --lang en --shots <id>`. Sem os arquivos `.en`,
o bundle inglês mostra a versão em português com um aviso.

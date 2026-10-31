# Latent Atlas — do neurônio ao cluster

> 🇧🇷 Português · [🇺🇸 English](README.en.md)

Plataforma HTML de estudo para AI Researchers. Em ordem cronológica e com figuras animadas e
interativas, ela ensina **tudo** o que é preciso para escrever receitas de inferência como as do
Mia's AI Lab (GLM-5.3-Flash EXL3, Qwen3.8-Flash-Next NVFP4 e DeepSeek-V4-Flash DSpark em
2× DGX Spark com vLLM): fundamentos de ML/DL, Transformers, arquiteturas de fronteira,
quantização, hardware e kernels, vLLM, decodificação especulativa e inferência distribuída. No
fim, um tutorial mostra como criar a sua própria receita para um modelo recém-lançado.

**Pilares da formação de AI Researchers** (agnóstica de hardware e de receita — do laboratório com
2× DGX Spark à produção com clusters H200 e NVIDIA AI Enterprise): **1 · Engenharia de inferência**
(capítulos 00–49) · **2 · Pós-treino** (capítulos 50–63) · **3 · Pré-treino** e **4 · Benchmarking &
estratégias de teste** (no roteiro). As receitas do Mia's AI Lab (incluindo a versão GLM sobre o
**TensorFold**) aparecem como **estudos de caso** que atravessam a trilha.

**Em números:** 64 capítulos em 14 partes e 2 pilares ativos · ~582 mil palavras · 694 figuras dinâmicas · 398
perguntas de quiz · 1.571 termos de glossário · catálogo com 1.734 parâmetros e modificações de 4 receitas ·
interface e conteúdo em **inglês e português** · copilot com OpenAI, MLX local e outros provedores.

**Estudos de caso datados.** As receitas são retratos de uma data: vLLM GLM em `94ae731` (28/09/2026), GLM ·
TensorFold v1.3.2 (01/10/2026), Qwen e DeepSeek lidos em 28/09/2026. O upstream continua mudando; cada bloco
GLM · TensorFold traz a etiqueta **retrato v1.3.2** com link para “o que mudou desde então” (cap. 49), e o que
ficou errado foi corrigido no lugar. A checagem de 30/10/2026 está em
[`docs/pesquisa/mia-updates-2026-10-30.md`](docs/pesquisa/mia-updates-2026-10-30.md); ela trouxe, entre outros,
o anel de 4 Sparks sem switch (cap. 40), o cache de prompts em NVMe (cap. 33) e o estudo de caso
“DeepSeek-V4.1-Flash: um modelo, cinco implantações” (cap. 44).

---

## 1. Requisitos

| Para… | Você precisa de |
|---|---|
| **Estudar** (usar a plataforma) | Um navegador moderno (Chrome, Edge, Safari ou Firefox recentes). Internet é opcional: sem ela, as fontes caem para as do sistema e as fórmulas aparecem como texto. |
| **Servir localmente** (recomendado para o copilot) | Python 3 (já vem no macOS e na maioria dos Linux). |
| **Copilot 100% local no Mac** | Mac com Apple Silicon (M1–M4) e `mlx-lm` (Python ≥ 3.10). O botão **MLX** da barra do topo instala o `mlx-lm` via `uv` ou Homebrew, baixa o modelo e sobe o servidor por você. |
| **Copilot na nuvem** | Uma API key (OpenAI, OpenRouter…) ou qualquer servidor compatível com a API da OpenAI. |
| **Editar e validar** (opcional) | Node.js ≥ 18, `npm`, e o Google Chrome instalado (para o QA headless). |

Nenhuma dependência precisa ser instalada para **usar** a plataforma: é HTML, CSS e JavaScript
puros, sem etapa de build obrigatória (o bundle já vem gerado em `dist/`).

## 2. Início rápido

```bash
cd latent-atlas
./serve.sh
```

O script regenera o bundle se algum capítulo mudou (quando o Node está disponível), sobe a
plataforma em **http://localhost:8765** e abre o navegador. Outra porta: `./serve.sh 9000` (se a
porta estiver ocupada, ele usa a próxima livre e imprime a URL).

Quem serve é o `tools/atlas_server.py` (só Python 3 da biblioteca padrão, já vem no macOS). Além dos
arquivos estáticos, ele é o **helper local do botão MLX**: baixa modelos do Hugging Face, instala o
`mlx-lm` e sobe/para o `mlx_lm.server` para o copilot. Ele só escuta em `127.0.0.1` e recusa pedidos
de outros sites. **Ctrl+C** encerra tudo, inclusive o servidor do modelo.

Alternativas:

```bash
open index.html                          # macOS: abre direto do disco (file://)
python3 -m http.server 8765 --bind 127.0.0.1   # qualquer sistema; depois abra http://localhost:8765
```

Abrir via `file://` (ou por outro servidor estático) funciona para estudar. Para o copilot, prefira
o `serve.sh`: sem ele o botão MLX fica no modo manual (mostra os comandos para copiar) e alguns
servidores (Ollama, LM Studio) recusam chamadas vindas de `file://`.

## 3. Idioma (português / inglês)

- **Inglês é o padrão** para todo mundo, qualquer que seja o idioma do navegador.
- Para estudar em português, clique em **PT** no canto superior direito. A escolha fica salva no
  navegador (o botão vira **EN** para voltar).
- Link direto: `index.html?lang=pt` ou `index.html?lang=en`.
- O copilot responde no idioma escolhido.

## 4. Configurar o copilot

Abra com o botão flutuante (✦) ou **⌘/Ctrl J** e clique em ⚙. Todos os provedores usam o contrato
OpenAI Chat Completions (`POST {base}/chat/completions`). Depois de configurar, clique em
**Testar conexão**.

### 4.1 MLX local no Mac (sem nuvem, sem chave) — botão **MLX**

1. Rode `./serve.sh` e abra a plataforma (http://localhost:8765).
2. Clique em **MLX** na barra do topo (entre ⏸ e ☀). O modal mostra um pipeline animado
   (Hugging Face → disco → memória unificada → `mlx_lm.server` → copilot) e quatro passos:
   - **① Seu Mac** — chip, RAM, orçamento de GPU (≈ 2/3 da RAM até 36 GB, 3/4 acima), espaço em
     disco, cache do Hugging Face e se o `mlx-lm` está instalado. Falta o `mlx-lm`? **Instalar mlx-lm**
     (usa `uv tool install --python 3.12 mlx-lm` ou `brew install mlx-lm`, com o log ao vivo).
   - **② Escolha um modelo** — a lista vem ordenada pela sua RAM, com o **sugerido** no topo
     (★). Cada cartão mostra parâmetros, bits, tamanho do download e uma barra de memória (pesos +
     folga de KV cache vs. orçamento de GPU), em grupos: **Cabem bem**, **Cabem no limite** e **Grandes demais**. Em **Adicionar do
     Hugging Face**, cole um id (`org/modelo`) ou a URL do modelo: a plataforma consulta o Hub, mostra o
     tamanho real, se é uma conversão MLX, se a arquitetura roda no seu `mlx-lm` e se é restrito
     (gated), e passa a sugerir o modelo. Modelos já baixados aparecem em “Neste Mac”.
   - **③ Baixar e rodar** — **Baixar e iniciar**: download com progresso, velocidade e tempo
     restante (pode cancelar e retomar depois), depois o servidor sobe e um aquecimento mede os tok/s.
     Também: trocar de modelo, parar, ver o log e apagar o download para liberar disco.
   - **④ Copilot** — quando o servidor fica pronto, o copilot é configurado sozinho
     (`http://127.0.0.1:8080/v1`). **Abrir copilot** ou **Fazer uma pergunta de teste**.
3. O ponto no botão MLX mostra o estado em qualquer página: cinza (desligado), âmbar (baixando /
   carregando), verde (rodando), vermelho (erro).

Os modelos ficam no cache padrão do Hugging Face (`~/.cache/huggingface/hub`), compartilhado com
`mlx-lm`, `transformers` e `huggingface_hub`. Para modelos restritos (gated), aceite a licença no
Hugging Face e suba com o seu token: `HF_TOKEN=hf_… ./serve.sh` (ou `hf auth login`, se você tiver a CLI
`hf`). O helper lê o token ao iniciar e só o envia ao próprio Hugging Face.

| RAM do Mac | Sugestão do botão MLX (mlx-community) | Download |
|---|---|---|
| 8 GB | `Qwen3-4B-Instruct-2507-4bit` | 2,3 GB |
| 16–24 GB | `gemma-4-12B-it-qat-OptiQ-4bit` (QAT, camadas 4/8 bits) · alternativa `gemma-4-12B-it-4bit` | 8,9 GB · 6,8 GB |
| 32 GB | `gemma-4-26b-a4b-it-4bit` (MoE, ~4B ativos) | 15,4 GB |
| 48 GB | `Qwen3.6-35B-A3B-4bit` (MoE, ~3B ativos) | 20,4 GB |
| 64 GB+ | `Qwen3.6-35B-A3B-8bit` (MoE, quase sem perda) | 37,7 GB |

Regra do encaixe: pesos + folga (≥ 1 GB ou 10%) ≤ 90% do orçamento de GPU. Modelos pequenos
(≤ 4B) erram mais detalhes técnicos. Com o “thinking” ligado, o modelo pode gastar o limite de tokens
só raciocinando: por isso a opção **Pedir resposta sem “thinking”** já vem ligada nos provedores locais.
Para testar o pipeline em poucos minutos, use o `Qwen3-0.6B-4bit` (351 MB).

**Sem o helper** (página aberta via `file://` ou outro servidor), o modal funciona em modo manual:
escolha do modelo, comandos para copiar e **Verificar conexão**. Pelo terminal:

```bash
./tools/start-mlx-copilot.sh          # escolhe o modelo pela RAM, instala o mlx-lm se faltar e sobe o servidor
./tools/start-mlx-copilot.sh 24       # força um perfil de memória (8 | 16 | 24 | 32 | 48 | 64)
MODEL=mlx-community/Qwen3-0.6B-4bit ./tools/start-mlx-copilot.sh   # modelo específico
# ou, à mão:
uv tool install --python 3.12 mlx-lm   # ou: brew install mlx-lm
mlx_lm.server --model mlx-community/gemma-4-12B-it-qat-OptiQ-4bit --port 8080 --max-tokens 2048 --temp 0.4
```

No copilot: provedor **MLX local (Mac)**, Base URL `http://127.0.0.1:8080/v1`, API key **vazia**,
Modelo **vazio** (usa o modelo carregado pelo servidor).

### 4.2 OpenAI / OpenRouter
Escolha o provedor, cole a **API key**, clique em **Listar modelos** e escolha um. Modelos de
raciocínio podem exigir `max_completion_tokens` e recusar `temperature`: o copilot detecta o erro e
tenta de novo sem os campos recusados.

### 4.3 LM Studio
Carregue um modelo (há motor MLX no Mac), abra **Developer → Start Server** (porta 1234) e habilite
**CORS** nas configurações do servidor. Base URL: `http://localhost:1234/v1`.

### 4.4 Ollama
```bash
OLLAMA_ORIGINS="*" ollama serve
ollama pull qwen3:8b
```
Base URL: `http://localhost:11434/v1`, Modelo: `qwen3:8b`.

### 4.5 A sua própria receita (vLLM)
As receitas estudadas expõem um servidor compatível com a OpenAI (ex.: porta 8888 na receita GLM).
Para o navegador acessar outra máquina, o vLLM precisa aceitar CORS:
```bash
vllm serve … --allowed-origins '["*"]'
curl http://IP-DO-HEAD:8888/v1/models
```
Base URL: `http://IP-DO-HEAD:8888/v1` (cole a chave se o servidor usa `--api-key`).

### 4.6 Onde fica a configuração
Tudo (inclusive a API key) fica **só no `localStorage` deste navegador**. Não use chaves de produção
em máquinas compartilhadas. “Restaurar padrões” no painel apaga a configuração. As escolhas do botão
MLX (modelo, porta, modelos adicionados) ficam nas chaves `la:mlx:*`; os modelos baixados ficam no
cache do Hugging Face.

## 5. Usando a plataforma

| Recurso | Como |
|---|---|
| Trilha | Barra lateral com as 14 partes, agrupadas nos 2 módulos, na ordem de estudo. ✓ marca capítulos concluídos. |
| Busca | **⌘/Ctrl K**: capítulos, seções, trechos, glossário, parâmetros. |
| Figuras | Controles abaixo de cada figura · **⤢** tela cheia · **✦ perguntar** envia a figura ao copilot · botão ⏸ no topo pausa todas as animações. |
| Copilot | **⌘/Ctrl J** · contexto *Seção* / *Capítulo* / *Sem contexto* · selecione um trecho do texto e clique em “Perguntar ao copilot”. |
| Copilot local (MLX) | Botão **MLX** no topo: escolhe, baixa e sobe um modelo no seu Mac e liga o copilot nele (seção 4.1). |
| Catálogo | Menu lateral → “Catálogo de parâmetros & modificações”: 1.715 itens com filtros por receita, tipo e categoria. |
| Glossário | Termos sublinhados no texto mostram a definição ao passar o mouse; clique para ver o glossário. |
| Progresso | Quiz no fim de cada capítulo · “marcar como concluído” · tudo salvo no navegador. |
| Tema e idioma | ☀/☾ no topo (escuro/claro) · **PT/EN** troca o idioma (inglês é o padrão). |
| Atalhos | ⌘K busca · ⌘J copilot · Shift+←/→ capítulo anterior/próximo · Esc fecha painéis. |

## 6. Editar, gerar e validar

```bash
npm install                       # só o puppeteer-core (para o QA); usa o Google Chrome instalado
node build.js                     # valida e gera dist/chapters.bundle.js (pt) e dist/chapters.en.bundle.js (en)
node build.js --lang en --check   # só valida um idioma
node tools/qa.js --shots          # QA headless de todos os capítulos em pt (screenshots de cada figura)
node tools/qa.js --lang en --mobile atencao mla   # QA de alguns capítulos, em inglês, viewport de celular
node tools/i18n-check.js          # confere se a tradução inglesa tem a mesma estrutura e nenhum português vazado
node tools/shot.js '#/c/atencao' shot.png --lang en --light   # screenshot de uma página
node tools/mock-llm.js 18080      # servidor falso compatível com a OpenAI, para testar o copilot sem modelo
python3 tools/test_atlas_server.py   # testes do helper MLX (hub falso local; não baixa nada nem instala nada)
```

As screenshots e os relatórios do QA vão para `qa-out/` (ou para a pasta definida em `QA_OUT`).

- Como escrever ou estender capítulos: [`docs/AUTHORING.md`](docs/AUTHORING.md)
- Como traduzir: [`docs/I18N.md`](docs/I18N.md)

## 7. Estrutura

```
index.html                   casca da aplicação (carrega os dados do idioma ativo)
serve.sh                     serve em http://localhost:8765 (rebuild automático) via tools/atlas_server.py
assets/css/                  core (tokens, layout), components (vocabulário de conteúdo), copilot, mlx (botão e modal MLX)
assets/js/i18n.js            idioma e dicionário da interface (pt/en)
assets/js/core.js            roteador, trilha, busca, glossário, quiz, catálogo, progresso
assets/js/viz.js             kit de visualização (canvas com ciclo de vida, matrizes, UI, tensor 3D CSS)
assets/js/num.js             aritmética de formatos (FP8/FP4/MX/NVFP4, RNE, Hadamard, quantização)
assets/js/copilot.js         copilot (OpenAI-compatível + MLX)
assets/js/mlx.js             botão MLX + modal (escolher, adicionar, baixar, subir e ligar o copilot)
chapters/NN-slug.*           capítulos em pt (.html, .js, .glossary.json) e em en (.en.html, .en.js, .en.glossary.json)
data/manifest.js             partes da trilha (pt e en)
data/mlx-models.js           modelos MLX sugeridos (verificados no Hugging Face) e a regra de encaixe na RAM
data/catalog[.en].js         catálogo de parâmetros & modificações
data/glossary[.en].js        glossário curado
dist/chapters[.en].bundle.js bundles gerados por build.js
docs/APANHADO[.en].md        todos os parâmetros e modificações das três receitas (pt e en)
docs/pesquisa/               inventários e pesquisas que embasam o conteúdo (arquivo:linha, números medidos)
docs/AUTHORING.md · docs/I18N.md · docs/curriculum.json
tools/atlas_server.py        servidor da plataforma + helper MLX (API local em /api/mlx/*, só 127.0.0.1)
tools/                       qa.js · i18n-check.js · shot.js · mock-llm.js · start-mlx-copilot.sh · gen-apanhado.py · test_atlas_server.py
```

## 8. Solução de problemas

| Sintoma | Causa provável e solução |
|---|---|
| Barra lateral com “Capítulos com erro de carga” | Um capítulo tem erro de sintaxe. Rode `node build.js` e veja a mensagem. |
| Fórmulas aparecem como `\( … \)` | Sem internet (o KaTeX vem de CDN). Conecte-se e recarregue. |
| Capítulo em inglês com aviso “not translated yet” | Falta o arquivo `.en` daquele capítulo. O conteúdo em português aparece como fallback. |
| Copilot: “Não foi possível alcançar…” | Servidor desligado, porta/Base URL errada (lembre do `/v1`) ou CORS. Use `./serve.sh` e habilite CORS no servidor. |
| Copilot MLX responde vazio | O modelo gastou os tokens “pensando”. Ligue “Pedir resposta sem thinking” ou aumente os tokens. |
| MLX muito lento ou travando | Modelo grande demais para a RAM. No botão MLX, escolha um modelo do grupo **Cabem bem** (barra de memória verde). |
| Botão MLX em “Modo manual — helper desligado” | A página foi aberta via `file://` ou por outro servidor (por exemplo, um `python3 -m http.server` antigo). Pare-o (Ctrl+C) e rode `./serve.sh`. |
| Download lento ou interrompido | Cancele quando quiser: o parcial fica guardado e **Retomar download** continua de onde parou. |
| Modelo “Restrito” (gated) | Aceite a licença na página do modelo no Hugging Face e reinicie com `HF_TOKEN=hf_… ./serve.sh`. |
| Selo “Não suportado” | O `mlx-lm` instalado não conhece aquele `model_type`. Clique em **Atualizar** no passo ① ou escolha outro modelo. |
| Disco cheio | No passo ③, **Apagar download** remove o modelo escolhido do cache do Hugging Face. |
| `mlx_lm.server: command not found` | Rode `./tools/start-mlx-copilot.sh` (instala via uv/brew) ou abra um terminal novo depois da instalação. |
| Porta 8765 ou 8080 em uso | O `./serve.sh` passa para a próxima porta livre sozinho. Para o modelo, o botão MLX oferece “Usar a porta 8081” (ou “usar esse servidor”, se já for um servidor OpenAI-compatível). |
| Progresso e configurações sumiram | O `localStorage` foi limpo (navegação privada, limpeza de dados do site). |
| `node tools/qa.js` falha ao iniciar | Rode `npm install` e confirme que o Google Chrome está em `/Applications` (ou defina `CHROME=/caminho/do/chrome`). |

## 9. Créditos

As receitas usadas como laboratório são do **Mia's AI Lab**:
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) ·
[Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) ·
[DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).

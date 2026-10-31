# Patches 0027–0053 da receita GLM-5.3-Flash EXL3 em TensorFold (parte B)

> Escopo deste arquivo: os 27 patches `patches/0027-*.patch` … `patches/0053-*.patch` do repositório
> `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`, todos lidos por inteiro (12.703 linhas de diff contra os
> `site-packages` do TensorFold v0.6.0). Para cada patch: nome, área, problema, mecanismo (o que o código muda,
> explicado conceitualmente), configuração que o controla, efeito medido, conceitos subjacentes e qual capítulo do
> Latent Atlas (`docs/curriculum.json`) ensina o conceito — com o que deveria ser acrescentado lá.
> Os patches 0001–0026 ficam na parte A (`tensorfold-patches-a.md`); aqui só são citados como contexto.
>
> Convenções: identificadores, flags, variáveis de ambiente, nomes de arquivos e fórmulas ficam no original.
> `[código]` = lido no diff do patch (arquivo dentro do patch entre parênteses); `[README]`, `[CHANGELOG]`,
> `[config.sh]` = número/afirmação do próprio repositório; `[cálculo próprio]` = conta minha a partir de valores
> exatos do código; `[inferência]` = dedução minha, não afirmada pela fonte. Nenhum número foi inventado: onde o
> repositório não mede um patch isoladamente, isso é dito.

---

## 0. Fontes e versões

| Fonte | Versão | O que foi usado |
|---|---|---|
| `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` (clone local, somente leitura) | commit `92bf731` (merge do PR #20, v1.3.2, 2026-10-01); imagem `v0.6.0-ae8d1c789b47` (`sha256:22789f0c…`), 53 patches | `patches/0027`–`0053` (integral), `README.md`, `CHANGELOG.md`, `CREDITS.md`, `NOTICE`, `scripts/config.sh` |
| TensorFold (`ashhart/TensorFold`) | v0.6.0 (Apache-2.0 a partir da v0.6.0; código anterior sob MIT) | só como "base" dos diffs: os patches mostram o contexto do código original |
| Receita anterior (vLLM) `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` | clone local | README (números de decode/prefill) para comparação; inventários `arch-glm.md`, `glm-mods.md`, `glm-launch.md` desta pasta |
| `jayleaton/glm53-tensorfold-spark` | citado em `NOTICE`/`CREDITS` | origem (Apache-2.0) de partes de 0036 (patch 0620 dele), 0046 (patch 0460) e 0047 (patch 0580) |

Como os patches entram na imagem `[README]`: `scripts/prepare.sh` aplica todos os `patches/*.patch` com `patch -p0`
em ordem de nome de arquivo sobre o TensorFold v0.6.0 instalado no contêiner `nvcr.io/nvidia/pytorch:26.07-py3`;
o hash da imagem é `sha256(patches/*.patch + IMAGE_EXTRAS)` truncado em 12 caracteres (`image_hash` em
`scripts/config.sh`), e `start.sh` reconstrói/re-baixa a imagem quando os patches mudam. Na v1.3 os patches foram
rebaseados para a v0.6.0 e renumerados (três patches viraram parte do próprio TensorFold: doorbell ocioso,
`TENSORFOLD_MEMORY_RESERVE_GIB` e `TF_GLM_MTP`) `[CHANGELOG]`.

---

## 1. Contexto: o que muda ao trocar vLLM por TensorFold (o pano de fundo desta parte)

TensorFold v0.6.0 é um motor de inferência diferente do vLLM: para o GLM-5.3-Flash ele tem um **engine CUDA de dois
ranks escrito para essa família** (`tensorfold/families/glm5_next/cuda/*`), um rank por Spark, com o rank 0
servindo a API OpenAI e o rank 1 espelhando cada requisição. Os kernels são próprios — muitos em Triton (como diz o
`CREDITS.md`), outros em CUDA C++ (EXL3, KDA em chunks, prefetch L2, RoCE) — e o estado do modelo é organizado à mão
(sem PagedAttention).
A receita roda TensorFold v0.6.0 + 53 patches. Os patches desta parte B são dominados por três frentes:
**(1) concorrência** (até 4 requisições sobre um pool compartilhado, mantendo a resposta idêntica à servida
sozinha), **(2) memória/decode** (KV FP8, prefetch de L2, cargas vetoriais nos experts EXL3, KDA em chunks) e
**(3) servidor/API** (tool calls para agentes, `/tokenize`, `/metrics`, visão com muitas mídias, erros de contexto).

| Aspecto | Receita vLLM (anterior) | Receita TensorFold | Patches desta parte |
|---|---|---|---|
| Motor | vLLM V1 (API server, EngineCore, Scheduler, KV cache manager com PagedAttention) | Engine CUDA do GLM no TensorFold; rank 0 decide e serve, rank 1 aplica as decisões (§2.3) | 0030 |
| Concorrência | batching contínuo do vLLM (`MAX_NUM_SEQS` etc.) | `--parallel` 1–4 (`PARALLEL`, default 4), um pool de caches por token, **uma janela de verificação em lote por rodada** | 0027, 0029, 0030, 0035, 0040, 0041, 0048, 0049 |
| Contexto | 850k por default; pool de KV de 1,23M tokens `[README antigo]` | 1.048.576 por requisição; pool compartilhado de **2.922.496 tokens** na partida medida `[README]` | 0030, 0038 |
| KV da DSA | `fp8_ds_mla`: 656 B/token/camada + indexador 33 B/token/camada (`arch-glm.md` §4.3) | linhas FP8 próprias: latente 528 B/token/camada + chaves agrupadas 144 B por pool de 4 tokens (§4, 0038) | 0038 |
| Reuso de prompt | APC em blocos de 3584 tokens | estados de prompt guardados numa grade de 64 tokens, prefixo de sistema compartilhado, *replay* de prompt idêntico | 0042 (+ 0008, 0015 da parte A) |
| Especulação | DFlash2 k=7 + adaptive-k | DFlash2 + *copy drafts*, verificação por **igualdade** com a amostra chaveada do alvo (§2.2) | 0027, 0032, 0035 |
| Link entre Sparks | NCCL | all-gather RoCE "one-shot" (transporte do b12x) até 512 KiB + NCCL para o resto | 0040, 0052 (+ 0006 da parte A) |

Números de topo da receita TensorFold `[README]` (4 fluxos, janela de 1M, KV FP8, densos 4-bit, DFlash2 + copy,
visão ligada, clocks da GPU travados em 2.200 MHz, medido com sparkDash pela API a partir de outra máquina):

| Requisições simultâneas | Prosa (agregado) | Prosa por requisição | TTFT | Estruturado (agregado) | Estruturado por req. | TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60,4 tok/s | 60,4 | 170 ms | 114,7 tok/s | 114,7 | 149 ms |
| 2 | 79,2 | 40,4 | 269 ms | 147,6 | 77,1 | 295 ms |
| 3 | 89,5 | 30,6 | 330 ms | 196,3 | 67,4 | 317 ms |
| 4 | 108,8 | 27,9 | 340 ms | 227,9 | 61,1 | 415 ms |

Prefill `[README]`: ~1.950 tok/s de 8k a 64k (1.952,2 / 1.973,5 / 1.978,9 / 1.942,5), 1.837,9 a 131k, 1.641,7 a
262k; agulha em 981.841 tokens a 1.015 tok/s (967 s), encontrada. Respostas servidas 4 de cada vez são idênticas às
mesmas requisições servidas uma a uma (11 de 11 escalonadas e 11 de 11 em rajada). Qualidade (KV FP8 + densos q4):
GSM8K (250) 98,0%, HumanEval (164) 97,6%.

Para comparação (com cautela: metodologia diferente — temperatura 0, thinking desligado, média de boots frescos,
decode sem TTFT), a receita vLLM reporta 36,1 tok/s em prosa, 83,0 estruturado, 54,6 código, 1 requisição, e
prefill frio de 1.411 / 1.511 / 1.498 tok/s a 8k / 32k / 96k (default, 2026-09-28) `[README antigo]`.

---

## 2. Temas transversais (conceitos que atravessam vários patches)

### 2.1 Exatidão bit a bit e invariância de lote ("batch invariance")

O critério de projeto de quase todos os patches de concorrência é: **cada fluxo recebe os mesmos bits que receberia
rodando sozinho**. Em ponto flutuante isso não é automático: a soma não é associativa, então qualquer kernel cujo
particionamento (tiles, split-K, ordem de redução, escolha de algoritmo) dependa do número de linhas do lote muda os
bits de uma linha quando ela é processada junto com outras. Os patches resolvem caso a caso:

- **kernels "por linha"** (matmuls 4-bit `qmm`, normas, rotary, cola de hyper-connections, experts) já dão os mesmos
  bits para qualquer recorte de linhas — por isso podem rodar uma vez sobre a janela inteira;
- **o que depende do estado do fluxo** (recorrência KDA, atenção DSA, contexto do DFlash2) roda com *tabelas por linha*
  (posição, base do extent, slot) que reproduzem exatamente o laço, os tiles e os blocos de cabeças do kernel de um
  fluxo (0027, 0029, 0030, 0038);
- **o que pode escolher kernel pelo número de linhas** (seletor `F.linear` via cuBLAS e `torch.topk` do DFlash2) roda
  **por fluxo**, nas linhas do próprio fluxo (0027);
- **reduções com atômicos** são feitas só em **inteiros** (histogramas da seleção top-k), que dão o mesmo resultado em
  qualquer ordem (0029).

Consequência medida `[README]`: respostas de 4 fluxos == respostas de 1 fluxo (11/11 + 11/11); e "drafted == serial":
uma resposta com rascunhos é idêntica à decodificação serial token a token (`"draft": false` dá a referência).
Exceções declaradas, por velocidade/memória: `DENSE=q4` e `KV=fp8` são *lossy* em relação ao checkpoint BF16, e
`KDA_CHUNKED=1` fica "perto do kernel serial, mas não nos seus bits" `[README]`.

### 2.2 Amostragem chaveada e verificação por igualdade ("drafted == serial")

TensorFold amostra cada posição com ruído derivado de uma chave (o `seed` da requisição, ou, sem `seed`, uma chave
derivada do prompt `[README]`) e da posição (as posições absolutas são passadas ao amostrador:
`choose_rows(values, tokens, positions, sampling)` `[código]`). Na verificação especulativa, o alvo amostra **todas** as linhas da
janela (token pendente + rascunhos) com esse ruído, e o fluxo aceita os rascunhos **enquanto a amostra do alvo for
igual ao rascunho** (`_round` em `multi.py`, 0030: `keep` cresce enquanto `rows[i] == d`). Como a amostra da posição
p não depende de quem propôs o rascunho, a resposta é exatamente a da decodificação serial com a mesma chave — não é
a amostragem por rejeição min(1, p/q) da literatura, e sim "verificação por igualdade sob amostragem determinística".
Isso também permite **refazer** uma requisição (preempção por recomputação, 0030) sabendo que os tokens já enviados
sairão iguais. O top-p distribuído usa massas inteiras de ponto fixo (`mass = floor(exp(scaled − top) · MASS)`,
`need = ceil(Fraction(top_p) · total)`) para que somas cumulativas sejam exatas entre ranks (0034, 0035) `[código]`.

### 2.3 Rank 0 decide, rank 1 aplica

O engine de concorrência (0030) é SPMD com um "líder": tudo que depende de tempo, de clientes ou de alocação é decidido
pelo rank 0, que envia **uma mensagem por iteração** (via `GlmEngine._share`, duas all-gathers: tamanho e valores) com
uma lista de operações — `ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE` (+ `SLOT` em 0035, `MFILL` em 0049).
O rank 1 aplica as mesmas operações, na mesma ordem, pelo mesmo código; ele recalcula os rascunhos sozinho e confere
as profundidades enviadas pelo rank 0. Um rank 1 ocioso espera numa "campainha" (*doorbell*) em vez de dentro de uma
all-gather. Se uma iteração falha, `broken` trava o decodificador ("os dois ranks não são mais confiáveis para
concordar") até reiniciar os dois. Na partida, os dois ranks comparam um vetor de configurações (`mine` em
`engine.py`) e se recusam a iniciar se divergirem — cada patch que muda coletivas ou rascunhos acrescenta seus
parâmetros a esse vetor (0030, 0032, 0035, 0038, 0040). Os *deadlocks* corrigidos em 0040 e 0052 são exatamente casos
em que os ranks fizeram chamadas coletivas diferentes.

### 2.4 Pool compartilhado com *extents* (em vez de páginas)

Em vez da tabela de blocos do PagedAttention, o 0030 corta um **arena** de linhas por token em **extents contíguos**
`[base, base+size)` alinhados a 2.048 tokens (`ALIGN = 2048`, `pool.py`): cada fluxo vivo ou prompt guardado possui um
extent; os kernels de um fluxo enxergam *views* do seu extent como se fossem tensores próprios. A alocação é *first
fit* pela menor base; extents crescem no lugar, são **movidos** (com cópia segura contra sobreposição) ou
**compactados** (todos deslizam para baixo) quando há fragmentação; prompts guardados são evictados por LRU. Essa
contabilidade é só no host, determinística, e roda igual nos dois ranks.

### 2.5 CUDA graphs com tabelas de metadados no device

Para capturar **um grafo por tamanho de janela** que sirva qualquer mistura de fluxos, posições e comprimentos, todo
metadado dinâmico (posição de cada linha, base do extent, slot, paridade, se a linha é densa ou esparsa, quantos pools
cada segmento enxerga) vive em **tensores no device**, preenchidos por cópia *pinned* assíncrona antes do *replay*
(`SegRows`, `kseg`, `b_meta`/`t_meta`). Os kernels leem esses limites no device em vez de recebê-los como constantes
de compilação (0027, 0029, 0030, 0043). Linhas de *padding* escrevem em linhas "lixo" (*trash*) reservadas (0027).

### 2.6 Estado recorrente com duas paridades + *replay*

Cada slot guarda o estado KDA em **duas paridades** `[count, 2, layers, H, 128, 128]` fp32 (`Slots`, 0030): a janela
de verificação lê a paridade corrente e escreve na outra, então o estado confirmado não é destruído pelos rascunhos.
No *commit*, se todas as linhas foram aceitas basta inverter a paridade; se só parte foi aceita, o estado é **refeito
(replay)** só com as linhas mantidas a partir de um *scratch* por linha (`replay_layers_segments`), e a janela de
convolução é deslocada com as projeções das linhas mantidas (`conv_shift_segments`). É a alternativa do TensorFold ao
"guardar num_spec+1 estados" do vLLM.

---

## 3. Tabela-resumo

| Patch | Área | Mudança em uma linha | Configuração (default da receita) | Efeito medido | Capítulo principal |
|---|---|---|---|---|---|
| 0027 `glm-multi-dflash2` | Rascunhos / concorrência | DFlash2 para vários fluxos: um conjunto de pesos, um anel de contexto por fluxo, passes de bloco em lote | `PARALLEL>1` + `DRAFTER=dflash2`; `TF_GLM_MULTI_DRAFT_GRAPHS` (1) | parte dos números de 4 fluxos; bits do drafter solo | `dflash-dspark` |
| 0028 `glm-lean-prompt-scratch` | Prompt / memória | estimativa de memória sem *scratch* que os kernels de prompt não usam | indireta (`TF_GLM_EXL3_PROMPT`, `TF_GLM_SPARSE_ONEPASS`, `DENSE=q4`) | "menos memória a 1M" (sem número isolado) | `paged-attention-kv` |
| 0029 `glm-multi-dsa` | Concorrência / DSA | janelas segmentadas: atenção latente e indexador DSA de vários fluxos num forward | `PARALLEL>1` | sem número isolado; bits do fluxo solo | `atencao-esparsa` |
| 0030 `glm-multi-stream-engine` | Concorrência (núcleo) | pool compartilhado, slots, scheduler, protocolo rank 0→rank 1, verificação em lote | `PARALLEL` (4), `KV_POOL_GIB` (12,5), `TF_GLM_CACHE_ENTRIES` (32), `TF_GLM_FILL_*`, `TF_GLM_MULTI_VERIFY` | 60,4→108,8 tok/s agregados (1→4 fluxos, prosa) | `scheduler-batching` |
| 0031 `glm-decode-index-regs` | Decode / indexador | pontuação do indexador por linha com query e pesos em registradores, um lançamento | sempre ligado | sem número isolado ("rodadas de decode mais rápidas") | `atencao-esparsa` |
| 0032 `glm-code-copy-drafts` | Rascunhos | copy drafts para código: casamento de 16 tokens na própria resposta, janelas de 16 linhas em CUDA graph | `COPY_CODE=1` → `TF_GLM_WIDE_GRAPHS=16`, `TF_GLM_COPY_REPLY_MATCH=16` | código 55,4→56,2 tok/s; edição 120→125 | `especulacao-fundamentos` |
| 0033 `glm-prefill-overlap2` | Prompt, dois Sparks | durante as trocas de linhas do glue HC, roda a "frente" do próximo bloco nas linhas próprias | `SPLIT=1` → `TF_GLM_PREFILL_OVERLAP=2` | prompt de 149k: 92,8→90,7 s | `hiperconexoes` |
| 0034 `cuda-nucleus-union` | Amostragem | teste de cobertura do top-p sobre os candidatos de todos os ranks juntos | `TENSORFOLD_NUCLEUS_UNION=1` (desligado) | opt-in, sem número | `geracao-amostragem` |
| 0035 `glm-multi-rounds` | Concorrência | ajustes das rodadas: amostrador empacotado, profundidade conjunta, mensagens assíncronas, fluxo sozinho nos grafos de 1 fluxo, perfil | `TF_GLM_MULTI_*` (receita: `TF_GLM_MULTI_LONE=0`) | LONE=1: +0,6–0,9% a 1 fluxo, mas 0% vs 98,1% de prompts do cache | `especulacao-fundamentos` / `scheduler-batching` |
| 0036 `glm-tool-calls` | Servidor / API | tool calls do GLM para agentes: streaming, calls dentro do think, raciocínio devolvido, tipagem | `TF_GLM_KEEP_REASONING` (1024), `_MB` (32) | funcionalidade | `api-tools-estruturado` |
| 0037 `cuda-tokenize` | Servidor / API | `/tokenize` e `/detokenize` no formato do vLLM; prompt como lista de ids | — | funcionalidade | `api-tools-estruturado` |
| 0038 `glm-kv-fp8` | KV cache | latente DSA e chaves agrupadas do indexador como linhas e4m3 + escala potência de 2; chaves/gates do indexador em anéis | `KV=fp8` → `TF_GLM_KV=fp8` | janela de 1M com 4 fluxos cabe; pool 2.922.496 tokens; GSM8K 98,0%, HumanEval 97,6% | `kv-cache-quantizado` |
| 0039 `glm-kda-chunked-kernel` | Prompt / KDA | KDA de prompt em forma WY/UT num único kernel CUDA, sub-chunks de 32 linhas, estado em registradores | `KDA_CHUNKED=1` → `TF_GLM_KDA_CHUNKED=1` | (0012+0014+0039) 50k: 29,3→26,4 s; 149k: 91,1→84,5 s | `atencao-linear` |
| 0040 `glm-parallel-deadlocks` | Concorrência / link | transporte escolhido só pelo tamanho; buffers próprios; perfil nos dois ranks; watchdog | `TF_GLM_MULTI_WATCHDOG_S` (300) | correção de travas | `nccl-rdma` |
| 0041 `glm-parallel-ring-base` | Concorrência | cronometragem de janelas passa a base do anel do indexador | — | correção | `cuda-graphs-warmup` (nota) |
| 0042 `glm-prompt-replay` | Reuso de prompt | prompt idêntico retoma do estado final guardado + linha de logits, sem prefill | (estados guardados) | prompt de 64k repetido: 34 s → <0,07 s | `prefix-caching` |
| 0043 `glm-visible-pools` | Decode / indexador | pontuação e seleção limitadas aos pools visíveis de cada linha dentro do *bucket* do grafo | sempre ligado | "os mesmos tokens" (sem número de velocidade) | `atencao-esparsa` |
| 0044 `cuda-context-errors` | Servidor / API | recusas de contexto com `code: context_length_exceeded` e `param` | — | funcionalidade | `api-tools-estruturado` |
| 0045 `cuda-metrics` | Servidor / observabilidade | `/metrics` ganha os números do `/health` (`tensorfold_health:`) | — | funcionalidade | `benchmarking` |
| 0046 `glm-l2-prefetch` | Decode / memória | stream lateral traz para L2 os pesos dos próximos kernels durante as all-gathers | `TF_GLM_L2PF=1` (+ `_MB`, `_SITES`…) | 1 req.: prosa 48,36→49,46, código 59,54→61,08 | `gpu-por-dentro` |
| 0047 `glm-exl3-decode-loads` | Decode / memória | trellis EXL3 lido com cargas de 16 B não-coerentes, k passos à frente | `TF_GLM_EXL3_LOADS=nc` | sozinho: prosa 48,36→48,78, código 59,54→60,42; com 0046+512 KiB: +2,7% / +3,3% | `hadamard-trelica-exl3` |
| 0048 `glm-timing-tokens` | Concorrência | cronometragens de partida com tokens distintos (MoE realista) | afeta `TF_GLM_MULTI_DEPTH=joint` / perfil | janela de 16 linhas era medida em ~metade do custo real | `benchmarking` |
| 0049 `glm-multi-prefill` | Concorrência / prefill | vários prompts que chegam juntos num único forward de prompt | `MULTI_PREFILL=1` → `TF_GLM_MULTI_PREFILL=1` | 4 prosa: 103,4→108,8 tok/s, TTFT 590→340 ms | `scheduler-batching` |
| 0050 `glm-many-media` | Visão | até 50 imagens e 4 vídeos por requisição, orçamento de tokens compartilhado, pico de memória menor | `TENSORFOLD_GLM_MAX_*`, `_REQUEST_*_TOKENS` | 50 imagens em 12,5 s (16.239 tokens) | `multimodal` |
| 0051 `glm-tool-history-recovery` | Servidor / tools | tool call passada com argumentos inválidos sai do prompt (logada) em vez de HTTP 400 | — | corrige issue #2 | `api-tools-estruturado` |
| 0052 `cuda-roce-startup` | Link | até o engine estar pronto, cada gather RoCE passa antes por barreira NCCL; espera configurável | `TF_ROCE_WAIT_S` (20) | corrige issue #6 (primeira partida) | `nccl-rdma` |
| 0053 `glm-whole-tool-calls` | Servidor / tools | cada tool call vai inteira quando fecha; call cortada nunca é enviada; *keepalive* de 2 s | — | funcionalidade | `api-tools-estruturado` |

---

## 4. Patches em detalhe

### 0027 — `glm-multi-dflash2`

- **Área:** rascunhos especulativos sob concorrência (DFlash2 para vários fluxos).
- **Problema:** o `Drafter` do TensorFold v0.6.0 tem **um** contexto (um anel de K/V do drafter, que usa atenção de
  janela deslizante). Com vários fluxos, cada um precisa do seu contexto, e rodar um passe de bloco por fluxo
  multiplicaria lançamentos de kernel por rodada.
- **Mecanismo** `[código: dflash2_multi.py, dflash2.py]`:
  - `MultiDrafter` compartilha os **pesos** de um `Drafter` e cria um **pool de contextos** por camada do drafter,
    `[kv_heads, streams·cap + trash, head_dim]` em BF16: o fluxo i ocupa as linhas `i·cap…` (o anel do drafter, ou
    um buffer plano `capacity + block`), seguidas de linhas "lixo" (*trash*) onde escrevem as linhas de *padding*.
    `DraftContext` "imita" o contexto de um drafter solo (`add_taps`, `reset`, `context_end`, `pos_dev`, `kc`, `vc`,
    `ring`), então prefill, snapshots e restauração de um fluxo continuam usando o mesmo código.
  - **Passe de bloco em lote** (`_block_compute`): os blocos de S fluxos ficam lado a lado (`[pendente, máscara ×
    (block−1)]` por fluxo; o DFlash2 do GLM tem `block_size` 8 = 1 âncora + 7 máscaras, `arch-glm.md` §2.11).
    Todos os kernels de um passe de bloco são independentes por linha, **exceto**: (a) a convolução dinâmica de 2
    taps, que ganhou o parâmetro `SEG` para que a primeira linha de cada bloco não misture com a última linha do bloco
    anterior; (b) a atenção do bloco, reescrita como `_dattn_seg_kernel` — programa `(kv_head, segmento)`, lendo o
    slot e a posição de cada segmento numa tabela, "o mesmo laço e aritmética do kernel solo, então cada bloco recebe
    os bits do solo"; (c) o projetor do seletor (`F.linear`/cuBLAS) e o `torch.topk`, que podem escolher kernels pelo
    número de linhas e por isso rodam **por fluxo**, nas linhas do fluxo.
  - **Candidatos entre ranks:** o cabeçalho do drafter é dividido no vocabulário entre os dois ranks (`vocab_offset`);
    cada rank manda seus top-k (valores + ids como bits fp32) numa all-gather e o host funde por valor decrescente e
    depois id (`merge_candidates`, extraído do `Drafter` para ser reutilizado).
  - **Atualização de contexto em lote** (`commit`/`_taps_compute`): as linhas aceitas de todos os fluxos (as *taps*
    `[n, 5·D]` das 5 camadas-alvo) são empilhadas em lançamentos de até `t_max` linhas, arredondadas a múltiplo do
    bloco; cada linha carrega sua posição e seu destino (`slot·cap + pos mod anel`), linhas de padding vão para o lixo.
  - **CUDA graphs**: um grafo por número de fluxos (1…streams) para o passe de bloco e um por *bucket* de linhas
    para as atualizações; a captura usa tabelas que mandam tudo para o lixo, então não toca nenhum fluxo. Metadados
    vão por buffers *pinned* + eventos CUDA (cópia assíncrona sem sobrescrever um buffer ainda em trânsito).
- **Configuração:** ativado por `PARALLEL>1` com `DRAFTER=dflash2` (o modo multi-fluxo só rascunha com DFlash2);
  `TF_GLM_MULTI_DRAFT_GRAPHS=0` desliga os grafos do drafter (default 1).
- **Efeito medido:** não isolado; é componente dos números de 4 fluxos (§1). Invariante declarado: "os candidatos de
  cada fluxo são os bits do seu `Drafter` solo".
- **Conceitos:** drafter em bloco (DFlash2), contexto de janela deslizante em anel, invariância de lote, kernels cuja
  escolha depende do número de linhas, top-k em vocabulário particionado, CUDA graphs por *bucket*, memória *pinned*.
- **Capítulos e o que acrescentar:**
  - `dflash-dspark` (37): a ADIÇÃO atual já pede "estado do rascunhador fora do KV exige slot estável por requisição
    sob continuous batching e lotes ragged". Acrescentar o desenho do TensorFold como exemplo concreto: pool de anéis
    `[kv_heads, S·cap + lixo, hd]`, conv dinâmica segmentada (`SEG`), atenção segmentada por tabela de slot/posição,
    e a regra "operações que podem escolher kernel pelo nº de linhas rodam por fluxo".
  - `cuda-graphs-warmup` (29): linhas "lixo" para *padding* de buckets capturados; captura que não toca estado vivo.
  - `numeros-no-computador` (4): exemplo prático da não-associatividade — cuBLAS/`topk` dependentes do tamanho do lote.

### 0028 — `glm-lean-prompt-scratch`

- **Área:** prefill / estimativa de memória.
- **Problema:** a estimativa de partida (`mla_geometry`, que decide a janela e o pool de KV) contava *scratch* que os
  caminhos de prompt efetivamente usados não alocam: (a) somas fp32 de split-K dos experts EXL3 num chunk de prompt,
  embora os kernels de prompt EXL3 (`TF_GLM_EXL3_PROMPT`) não as usem; (b) parciais split-K `8 × rows × 16.384` fp32
  dos buffers de prompt mesmo com densos 4-bit (`DENSE=q4`), cujo matmul de prompt não guarda parciais; (c) parciais
  da atenção latente esparsa, que o kernel *one-pass* (`TF_GLM_SPARSE_ONEPASS`) não aloca. Bytes contados a mais são
  bytes tirados do KV.
- **Mecanismo** `[código: geometry.py, engine.py, forward.py, latent.py]`: `exl3_expert_scratch(..., prompt=)` conta,
  para janelas de prompt, só as entradas rotacionadas fp16 e a "ordem de lançamento", e para decode as somas split-K
  e contadores de bloco; `mla_geometry` recebe `onepass`, `exl3_prompt`, `prompt_split_k`; `Buffers.sk` só aloca
  parciais quando `prompt_split_k(w)` (checkpoint EXL3 com projeções BF16/FP8). Também: a atenção densa de prompt
  em blocos de `PROMPT_ATT_ROWS` linhas passa a usar um tensor `att_starts = arange(0, rows, part_rows)` no device
  (`at = pos_dev + starts`) em vez de criar `pos_dev + r0` por bloco — mesmos bits, menos operações no host.
- **Configuração:** nenhuma própria; depende de `TF_GLM_EXL3_PROMPT`, `TF_GLM_SPARSE_ONEPASS`, `DENSE`.
- **Efeito medido:** o README agrupa 0028 com os patches de prompt ("faster prefill, less memory at 1M"); não há
  número isolado.
- **Conceitos:** orçamento de memória por estimativa geométrica (vs *profile run* do vLLM), split-K e seus parciais,
  *scratch* de kernels, kernels de atenção *one-pass* sem parciais.
- **Capítulos:** `paged-attention-kv` (31) — a ADIÇÃO já fala de "workspaces pré-alocados (… scratch de kernels e
  split-K …) descontados do KV". Acrescentar que o TensorFold **não faz profile run**: calcula a memória por fórmulas
  (`mla_geometry`, `bytes_at(capacity)`) e por isso cada byte de *scratch* superestimado vira token de KV perdido;
  `kernels` (28): o que é um parcial de split-K e por que um kernel "sequencial" (soma dos splits num programa) não o
  precisa.

### 0029 — `glm-multi-dsa`

- **Área:** concorrência / atenção esparsa DSA (MLA latente + indexador).
- **Problema:** numa janela de verificação com vários fluxos, cada fluxo tem seu próprio cache latente, chaves e
  *gates* do indexador e chaves agrupadas (*pooled keys*), em lugares diferentes do pool; a atenção e a seleção de
  tokens precisam enxergar cada um no seu extent e dar os bits do fluxo solo.
- **Mecanismo** `[código: forward.py dsa_segments, latent.py seg_*, segments.py, sparse.py seg_* / select_split]`:
  - **Janela segmentada:** as linhas de vários fluxos ficam em sequência (`[s1: linhas][s2: linhas]…`, até
    `MAX_SEGS = 4`); o estado DSA de cada fluxo mora no seu extent de arenas compartilhadas (latentes `[P, 512]`,
    chaves/gates `[P, 128]`, chaves agrupadas `[P/4 + 2, 128]`), extents alinhados a `EXTENT = 2048` tokens (as
    chaves agrupadas começam em `base/4`).
  - `SegRows`: **um tensor int32 no device** com, por linha, `pos`, `base`, `seg`, `sparse` (1 = além do limite denso,
    atende aos tokens selecionados; 0 = densa sobre 0…pos) e, por segmento, `seg_start`, `seg_rows`, `seg_pbase`,
    `seg_pools` (quantos pools a última linha enxerga). Buffers estáticos → **um CUDA graph por tamanho de janela**
    cobre qualquer mistura de posições e comprimentos.
  - Kernels segmentados (`_seg_lat_write`, `_seg_chunks` + `_merge`, `_seg_index_write`, `_seg_pool_keys`,
    `_seg_scores`, `_seg_select`): mesmos tiles, máscaras e matemática (`_tile`, softmax online) dos kernels de um
    fluxo. Blocos de cabeças: 16 cabeças por programa para janelas pequenas, as 32 cabeças do rank a partir de
    `SEG_WIDE_ROWS = 8` linhas (cada tile de chaves lido uma vez); ambos dão os mesmos bits.
  - **Indexador** (lightning indexer estilo DSA, com kpool=4): a chave de índice passa por LayerNorm; quando uma
    linha completa um pool (`pos % 4 == 3`), as 4 chaves são combinadas por softmax dos gates (+ `APE`, embedding
    posicional dos 4 slots) numa chave agrupada. Pontuação de pool:
    `s_p = Σ_h w_h · relu(scale · q_h · k_p)`; `_seg_scores` carrega cada bloco de chaves **uma vez** para todas as
    linhas esparsas do segmento.
  - **Seleção top-k exata e determinística:** *radix select* de 8 bits por passe (4 passes sobre a chave de ordem dos
    bits fp32) acha o K-ésimo melhor (K = `TOPK_POOLS` = 512 pools), depois escreve os pools acima dele e os empates de
    menor índice, em ordem crescente; cada pool vira 4 tokens, mais a cauda do pool incompleto →
    `TOKENS = 512·4 + 3 = 2051` tokens por linha esparsa. Linhas antes de `SPARSE_FROM = 2051` ficam densas.
  - **Seleção dividida** (`select_split`, novo): para linhas de decode com contexto longo (≥ `SPLIT_FROM = 8192`
    pools, < `SPLIT_ROWS = 64` linhas), cada linha é dividida em pedaços de 2.048 (até 16.384 pools) ou 4.096
    pontuações, um programa por (linha, pedaço), em 5 lançamentos; os histogramas totais da linha são somados com
    **atômicos inteiros** ("exatos em qualquer ordem"), e o passo 4 escreve cada pedaço no deslocamento dado pelas
    contagens dos pedaços anteriores. Antes, uma linha de decode longa era um único programa — pouco paralelismo.
  - `top_pools` de linhas de decode com mais de 1.024 pontuações usa blocos de 4.096 e 8 warps (só velocidade).
- **Configuração:** `PARALLEL>1`; `SEG_SPLIT`, `SEG_SCORE_GRID = 256`, `SEG_SELECT` são constantes de velocidade.
- **Efeito medido:** não isolado (componente dos números de 4 fluxos); invariante: bits do segmento rodado sozinho
  (o docstring cita `tests/cuda/test_glm_attn_segments.py`).
- **Conceitos:** DSA/lightning indexer, kpool com softmax de gates, top-k por *radix select*, determinismo com atômicos
  inteiros, paralelizar uma linha longa (estilo *flash-decoding* aplicado à seleção), arenas/extents, grafos com
  metadados no device.
- **Capítulos:**
  - `atencao-esparsa` (13): acrescentar como a seleção top-512 pools é feita em GPU (radix select de 8 bits sobre a
    chave de ordem fp32, empates para o pool menor, saída em ordem crescente, 2051 tokens = 512×4 + 3 da cauda) e a
    versão dividida para decode a contexto longo; a fórmula do score com ReLU e pesos por cabeça.
  - `paged-attention-kv` (31): extents de 2048 tokens com chaves agrupadas em `base/4` (por que o alinhamento existe).
  - `cuda-graphs-warmup` (29): "um grafo por tamanho de janela cobre toda mistura de posições" graças a tabelas no
    device.
  - `numeros-no-computador` (4): atômicos de ponto flutuante não são determinísticos; atômicos inteiros são.

### 0030 — `glm-multi-stream-engine`

- **Área:** concorrência — o núcleo (`--parallel N`).
- **Problema:** o engine GLM do TensorFold v0.6.0 serve uma requisição por vez: um `State` com tensores próprios do
  tamanho da janela. Servir até 4 requisições simultâneas exige (a) dividir a memória de KV entre elas sem reservar 4×
  a janela, (b) decodificá-las juntas de modo que cada uma receba exatamente a resposta que receberia sozinha, (c)
  manter os dois ranks em passo.
- **Mecanismo** `[código: forward.py Caches/Slots/State, pool.py, multi.py, verify.py, engine.py, decode.py, health.py]`:
  - **Separação "por token" × "por fluxo":** `Caches` = todo cache que cresce com o contexto (por camada DSA, o
    latente; com contexto longo, chaves, gates e chaves agrupadas do indexador; os da camada MTP) como **planos** de um
    `Arena` compartilhado (`Plane` com `div` tokens por linha e `pad` linhas: chaves agrupadas `div=4, pad=2`, porque
    a seleção lê até 2 pools além da janela). `Slots` = o que cada fluxo tem independentemente do comprimento: estados
    KDA `[count, 2, layers, H, 128, 128]` fp32 (duas paridades, §2.6), janelas de conv, projeções e *scratch* de
    replay da janela de decode, posições no device. Um `State` é uma *view* do seu extent + do seu slot (`bind`).
  - **Pool** (`pool.py`): extents `[base, base+size)` múltiplos de `ALIGN = 2048`; *first fit* pela menor base,
    crescimento no lugar, `move` (com `_move` seguro contra sobreposição, copiando em pedaços na direção certa),
    `remove`; só contabilidade no host, determinística (§2.4). As linhas de *pad* das chaves agrupadas de um extent
    são as primeiras do próximo: só lidas e mascaradas, nunca escritas.
  - **Tamanho do pool** (`engine.py`): `pool_rows = (capacity + max(0, cache_bytes − states) // per_token) // ALIGN ·
    ALIGN`, onde `cache_bytes` vem de `TF_GLM_CACHE_GIB` (na receita `KV_POOL_GIB = 12,5`), `per_token` é a derivada
    da geometria (bytes a mais por 65.536 tokens) e `states = TF_GLM_CACHE_ENTRIES × (slot_bytes(rows=0)/2 + 22 MiB)`
    reserva o estado fixo dos prompts guardados. A janela vira `min(limite, pool_rows − MAX_ROWS)`, e os prompts
    guardados passam a viver **dentro** do pool. A estimativa soma `(parallel − 1) × slot_bytes` (estados KDA, conv,
    projeções, scratch de replay; `[cálculo próprio]` pela fórmula de `engine.slot_bytes` com 34 camadas KDA, 32
    cabeças/rank e janela de 16 linhas: 5.458.944 B por camada → ≈185,6 MB ≈ 177 MiB por slot extra por rank) e o
    contexto do drafter multi-fluxo (`multi_draft_bytes`).
  - **Iterações** (`GlmScheduler`, uma thread no rank 0): admissões; depois **ou** um chunk de prompt do fluxo em
    preenchimento mais antigo (FIFO; no máximo `TF_GLM_FILL_ROWS` linhas enquanto outros decodificam, default
    `min(1024, prefill_rows)`, múltiplo de 64 e da grade) **ou** uma rodada de decode, alternando por um crédito
    `TF_GLM_FILL_SHARE` (default 0,5: chunk, rodada, chunk…; 1 = preencher tudo antes de decodificar). Um chunk de
    prompt nunca divide um forward com linhas de decode (na v0.6.0+0030; 0049 agrupa prompts entre si).
  - **Rodada de decode** (`_round`): cada fluxo propõe rascunhos sob sua própria política DFlash2 (copy drafts
    primeiro); a soma das janelas é limitada a `MAX_WINDOW = 32` linhas (`trim` corta um rascunho por vez da janela
    mais longa, a mais recente em empate); **uma** verificação sobre as linhas de todos os fluxos; **uma** all-gather
    empacotada para os amostradores (`sample_streams`: candidatos top-k de todas as partes juntos; partes com núcleo
    top-p usam a regra de núcleo com suas próprias gathers); cada fluxo aceita por igualdade (§2.2) e confirma só as
    suas linhas.
  - **Verificação** (`verify.py`, a "costura" `forward_streams`): `SerialVerify` (referência: cada janela como um
    forward próprio no seu estado) ou `BatchedVerify` (padrão: **um forward** — partes independentes por linha sobre a
    janela inteira; camadas KDA pela cadeia segmentada `kda.chain_segments`, cada segmento a partir do estado do seu
    slot e escrevendo na outra paridade; camadas DSA por `dsa_segments` de 0029). O *commit* faz replay das linhas
    mantidas quando um fluxo mantém menos linhas que verificou, desloca a conv, inverte a paridade e avança a posição.
    `BatchedVerify.capture()` grava um grafo por número de linhas (1…32) com posições, extents, slots e paridades em
    tabelas no device; é capturado no slot 0 nas primeiras linhas do pool, antes de qualquer admissão.
  - **Admissão** (`fits`/`admit`): uma requisição cabe se linhas livres + extents só com prompts guardados ≥ `align_up(
    prompt + 16)` + 2048 de folga por fluxo decodificando; senão espera **à frente da fila** (FIFO, `held`). A
    colocação tenta retomar do **maior prefixo guardado** (assumindo o extent guardado no lugar, ou copiando suas
    linhas para um extent novo), evicta prompts guardados por LRU e compacta quando preciso.
  - **Pausa e devolução (preempção por recomputação):** a cada rodada, o extent de cada fluxo precisa crescer para
    caber a janela mais larga; se não couber, o fluxo fica **pausado**. Se todos os fluxos estão pausados e nenhum
    preenche, o **mais novo** (sem gramática nem imagens) é devolvido à fila e refeito depois — "os tokens já enviados
    ficam devidos, não são reenviados"; se todos têm gramática ou imagens, o mais novo falha com `NoRoom`.
  - **Prioridade `background`:** quando todos os slots estão ocupados e há requisição de primeiro plano esperando, um
    fluxo `background` (sem gramática/imagens) é finalizado e recolocado na fila (`_yield`).
  - **Protocolo entre ranks** (§2.3): `ADMIT` leva sid, slot, base, tamanho, prompt guardado a retomar, amostragem
    empacotada em 10 int32 (seed em 3 palavras de 31/31/2 bits, temperatura, top_p e min_p como fp64 em 2 palavras
    cada, top_k em 1), pontos de prefixo
    compartilhado, política, prompt, gramática empacotada e posições de imagem; `EVICT/MOVE/GROW` replicam o pool;
    `FILL` (com conferência de que não passa de um ponto guardado), `ROUND` (sids e profundidades), `FINISH`, `IDLE`.
  - **MTP desligado em `--parallel`** (só DFlash2): políticas MTP são remapeadas para a gêmea DFlash2 ("rascunhos nunca
    mudam uma resposta"); `PARALLEL_MOST = 4`.
  - **Grafos de 1 fluxo só valem "em casa"**: `Engine.graphed()` só usa os grafos de um fluxo quando o estado é o
    `home` (slot 0, base 0) com as *views* do pool, porque o grafo "gravou" o slot e a base do extent (`graph_key`).
  - `/health` ganha `streams.{decoding, filling, paused}`, `pool_tokens`, `pool_free_tokens`, `kept_prompts`.
- **Configuração:** `PARALLEL` (`--parallel`, receita 4; 1 com `DRAFTER=mtp`), `KV_POOL_GIB` (`TF_GLM_CACHE_GIB`,
  12,5; TensorFold: 3), `TF_GLM_CACHE_ENTRIES` (receita 32 desde v1.3.2; TensorFold: 8), `TF_GLM_FILL_ROWS`,
  `TF_GLM_FILL_SHARE` (0,5), `TF_GLM_MULTI_VERIFY` (`batched`|`serial`), `TF_GLM_MULTI_GRAPHS` (1).
- **Efeito medido** `[README, config.sh]`: prosa 60,4 / 79,2 / 89,5 / 108,8 tok/s agregados com 1/2/3/4 fluxos;
  estruturado 114,7 / 147,6 / 196,3 / 227,9; respostas concorrentes == solo (11/11 + 11/11; CHANGELOG: "concurrency
  22/22"). Pool de 2.922.496 tokens na partida medida (teto de 12,5 GiB; 2,1–2,9M conforme a memória livre); "uma
  conversa de 1M ao lado de outra de 1M, ou de três de ~640k". Sobre `TF_GLM_CACHE_ENTRIES` 8 → 32 `[CHANGELOG
  v1.3.2]`: cada entrada reserva ~45 MiB na partida (~1 GiB a mais no total); com 8, três ou quatro conversas de
  agente alternadas se expulsavam (issue #17).
- **Conceitos:** batching contínuo com especulação (lote de 1+k linhas por fluxo), alocação por extents com
  compactação e LRU vs PagedAttention, preempção por recomputação, prefill em chunks intercalado com decode,
  controle de admissão FIFO, prioridade background, SPMD líder/seguidor, invariância de lote, grafos com metadados no
  device, estado recorrente com paridades e replay.
- **Capítulos:**
  - `scheduler-batching` (32), principal: um segundo modelo de escalonador além do vLLM — iterações "chunk **ou**
    rodada" alternadas por crédito (`FILL_SHARE`), teto de 32 linhas de verificação por rodada com corte do rascunho
    mais longo, pausa por falta de espaço, devolução do fluxo mais novo (preempção por recomputação, possível porque a
    amostragem é chaveada), FIFO estrito com requisição "segurada", prioridade background.
  - `paged-attention-kv` (31): comparar paginação × extents contíguos de 2048 tokens (first fit, crescimento no lugar,
    move, compactação); a fórmula do pool (`pool_rows`) e a separação "planos por token" × "slots por fluxo";
    "uma requisição pode crescer até a janela inteira e as quatro dividem o pool" `[README]`.
  - `vllm-arquitetura` (30): um quadro "outro motor": TensorFold sem EngineCore em processo separado, rank 0 = API +
    decisões, rank 1 seguidor, mensagens de operações, verificação de configurações iguais na partida.
  - `especulacao-fundamentos` (35): verificação por igualdade com amostra chaveada (§2.2) e especulação em lote com
    teto global de linhas.
  - `atencao-linear` (14) e a ADIÇÃO de 35 ("rollback de estado recorrente"): a alternativa paridade dupla + replay.
  - `prefix-caching` (33): prompts guardados dentro do pool, retomados do maior prefixo, evicção LRU por extent.

### 0031 — `glm-decode-index-regs`

- **Área:** decode / indexador DSA.
- **Problema:** em decode, a pontuação do indexador (`_scores`) era lançada em blocos de `SELECT_ROWS` linhas (um
  lançamento e um `top_pools` por bloco, com `pos_dev + a` criado no host), com um programa por (linha, bloco de 64
  pools) — cada programa recarregava a query de índice e os pesos por cabeça da linha.
- **Mecanismo** `[código: sparse.py _scores_rows]`: novo kernel `_scores_rows`, programa `(linha, g)`: carrega a
  query de índice `[H, 128]` e os pesos `w_h` **uma vez** (em registradores) e percorre os blocos de pools `g, g+G,
  …` (`G = ROW_SCORE_G = 128` programas por linha; laço `tl.range(..., num_stages=1)` porque o *pipelining* do Triton
  mediu mais lento). Mesmo tile (`BP=64` pools, 4 warps), mesma máscara e soma por cabeça de `_scores` → mesmos bits
  ("8 warps ou blocos de 32 pools dariam outros"). Um só lançamento para todas as linhas e um só `top_pools`. Há uma
  variante `SEG` para janelas segmentadas, mas `seg_select_tokens` continua com `_seg_scores` ("o laço por linha
  mediu não mais rápido ali, e mais lento em janelas densas").
- **Configuração:** nenhuma (sempre ligado).
- **Efeito medido:** não isolado; o README agrupa com 0013/0016/0019 ("faster decode matmuls, hyper-connection
  mixing and indexer scoring").
- **Conceitos:** reuso de dados em registradores, laço *grid-stride*, número de programas por linha vs ocupação,
  quando o *software pipelining* atrapalha, tile que "define os bits".
- **Capítulos:** `atencao-esparsa` (13) — custo do indexador em decode (pontuar todos os pools visíveis a cada token) e
  a otimização "carregar a query uma vez, varrer pools em grid-stride"; `kernels` (28) — exemplo de que mudar warps
  ou tile muda bits, mudar o número de programas não.

### 0032 — `glm-code-copy-drafts`

- **Área:** rascunhos (copy drafts / *prompt lookup*), foco em código.
- **Problema:** copy drafts (0007, parte A) propõem a continuação da ocorrência anterior dos últimos 8 tokens.
  Respostas de código repetem trechos curtos (`    def __init__(self`, `}); `) cujas continuações diferem: em 39
  respostas de código reais (2026-09-30), cópias tiradas da **própria resposta** mantiveram 23% dos rascunhos, cópias
  tiradas do **prompt**, 74% `[código: docstring de copy_drafts.py]`; uma rodada de cópia perdida custa uma janela larga
  e a rodada de DFlash2 que ela deslocou. Além disso, janelas com mais de 8 linhas rodavam sem CUDA graph.
- **Mecanismo** `[código: copy_drafts.py, engine.py, multi.py]`:
  - `reply_match` (`TF_GLM_COPY_REPLY_MATCH`): uma ocorrência que começa dentro da resposta precisa casar os últimos N
    tokens (> `match`), não só 8.
  - `miss_most` (`TF_GLM_COPY_MISS_MAX`): depois de uma rodada de cópia que não manteve todos os rascunhos, a próxima
    propõe no máximo esse número, até uma rodada de cópia acertar tudo de novo.
  - `pad` (larguras de `TF_GLM_WIDE_GRAPHS`): uma proposta cuja janela fica entre 8 linhas e uma largura capturada é
    completada repetindo o último rascunho, para rodar um CUDA graph em vez de eager; os rascunhos de padding são
    verificados como quaisquer outros.
  - `TF_GLM_MAX_ROWS`: janela de verificação mais larga, 16 (default) a 64 em passos de 8; acima de 16 linhas as
    all-gathers da janela passam de 256 KiB (16 KiB por linha) e vão por NCCL a menos que `TF_ROCE_MAX_KB` suba
    (512 para 32 linhas). `TF_GLM_WIDE_GRAPHS`: larguras extras capturadas (cada largura custa um grafo por paridade
    KDA e mais um por *bucket* de pools além do limite denso).
  - No modo multi-fluxo não há padding ("janelas em lote não são as larguras de um fluxo"). As novas configurações
    entram na comparação de configurações entre ranks.
- **Configuração:** `COPY_CODE=1` (receita) → `TF_GLM_WIDE_GRAPHS=16` e `TF_GLM_COPY_REPLY_MATCH=16`;
  `TF_GLM_COPY_MISS_MAX` fica 0 (desligado); `COPY_MAX=15`.
- **Efeito medido** `[README, config.sh]`: código 55,4 → 56,2 tok/s; edição 120 → 125 (com o overlap de 0033, dois
  boots cada). Exato.
- **Conceitos:** n-gram/*prompt lookup decoding*, taxa de aceitação por fonte do rascunho, custo de uma janela de
  verificação larga, *buckets* de CUDA graph e padding até o tamanho capturado, 16 KiB por linha (hidden 4096 × fp32).
- **Capítulos:** `especulacao-fundamentos` (35) — refinamentos de prompt lookup: casamento mais exigente para cópias da
  própria resposta (dados 23% vs 74%), limite após erro; `cuda-graphs-warmup` (29) — padding de uma proposta até a
  largura capturada como alternativa ao eager; `nccl-rdma` (39) — tamanho da mensagem por linha da janela define o
  transporte (RoCE ≤ `TF_ROCE_MAX_KB`, senão NCCL).

### 0033 — `glm-prefill-overlap2`

- **Área:** prefill nos dois Sparks (glue de hyper-connections dividido por linhas).
- **Problema:** com 0010/0017/0022 (parte A), num chunk de prompt cada rank calcula o glue HC (hc_post + hc_pre)
  só da sua metade das linhas e troca as linhas com o par (`ncclSend/ncclRecv`) num segundo stream, em pedaços
  sobrepostos; mas o stream principal ainda **esperava a troca do último pedaço sem nada para fazer**.
- **Mecanismo** `[código: forward.py *_front / undone, hcsplit.py glue(front=), own()]`: com
  `TF_GLM_PREFILL_OVERLAP=2`, logo após o glue de cada pedaço, o stream principal roda a **"frente"** independente
  por linha do próximo bloco nas linhas **deste** rank — projeções de entrada da KDA (`kda_front`), projeção, normas
  q/kv e expansão `q_b` da DSA (`dsa_front`), gate/up + SwiGLU do MLP denso (`mlp_front`) ou o expert compartilhado
  do MoE (`shared_front`) — enquanto as linhas normalizadas (e os parciais dos pedaços seguintes) estão "no fio". O
  bloco depois roda a frente só nas linhas do par (`undone(R, done)`) e o resto como antes. Como o matmul de prompt dá
  os mesmos bits para qualquer recorte de linhas, tudo continua idêntico ao caminho sem divisão.
- **Configuração:** `SPLIT=1` (receita) → `TF_GLM_HC_SPLIT=1`, `TF_GLM_PREFILL_OVERLAP=2`
  (0 = sem overlap, 1 = overlap das trocas, 2 = + frentes); exige split e overlap.
- **Efeito medido** `[config.sh]`: prompt de 149k 92,8 → 90,7 s (com `COPY_CODE`, dois boots cada). Mesmos bits.
  (O conjunto 0010+0033: prefill de 50k ~1.270 → ~1.730 tok/s com 0009 e 0020; rodadas de decode pagam ~1,5%.)
- **Conceitos:** sobreposição computação/comunicação, streams e eventos CUDA, paralelismo por linhas (estilo
  *sequence parallel*) de uma operação elemento-a-elemento cara, independência por linha.
- **Capítulos:** `hiperconexoes` (15) — o custo de banda do glue mHC (4 fluxos × 4096) em prefill e a divisão por
  linhas entre ranks com sobreposição; `paralelismos` (38) — exemplo de "row/sequence parallel" que não é TP de
  pesos; `nccl-rdma` (39) — p2p `ncclSend/ncclRecv` vs all-gather (`TF_GLM_HC_EXCHANGE`).

### 0034 — `cuda-nucleus-union`

- **Área:** amostragem top-p com vocabulário dividido entre ranks.
- **Problema:** o top-p exato do TensorFold manda, de cada rank, os `NUCLEUS` melhores candidatos com massas inteiras;
  o teste de cobertura exige que **a parte do núcleo de cada rank** termine dentro dos candidatos daquele rank, senão
  lê "shards inteiros" (gather de todo o pedaço do vocabulário). O teste por rank é conservador.
- **Mecanismo** `[código: cuda/sampling.py _union_covers]`: com `TENSORFOLD_NUCLEUS_UNION=1`, o teste olha os
  candidatos de **todos** os ranks juntos: percorre-os fundidos (valor decrescente, depois id) até a massa cumulativa
  `need`; para no valor v*; se v* é estritamente maior que o último candidato de cada rank que **não** mandou o shard
  inteiro, todo token não enviado está abaixo de v*, então o percurso fundido é o do vocabulário inteiro até v*:
  mesmo núcleo, mesmo sorteio. Se falhar, tenta primeiro `WIDER = 16 × NUCLEUS` candidatos antes de shards inteiros.
- **Configuração:** `TENSORFOLD_NUCLEUS_UNION=1` (desligado por default na receita); entra na comparação entre ranks.
- **Efeito medido:** opt-in, sem número publicado ("o mesmo sorteio com menos leituras de shard inteiro").
- **Conceitos:** top-p/núcleo, lm_head paralelo no vocabulário, prova de correção de um atalho (cota inferior), massas
  inteiras de ponto fixo para determinismo.
- **Capítulos:** `geracao-amostragem` (9) — como fazer top-p exato quando os logits estão divididos entre GPUs
  (candidatos por rank, massas inteiras, teste de cobertura, fallback para shards inteiros); `paralelismos` (38) —
  amostragem com lm_head *vocab-parallel*.

### 0035 — `glm-multi-rounds`

- **Área:** concorrência — ajuste fino das rodadas (`multi_tune.py`); nenhuma opção muda bits de resposta.
- **Problema:** com o motor de 0030 funcionando, sobram custos: o amostrador de núcleo fazia gathers por fluxo; cada
  fluxo escolhia sua profundidade de rascunho como se estivesse sozinho, embora a janela seja compartilhada; o host do
  rank 0 esperava a GPU para mandar a mensagem; um fluxo sozinho rodava nos grafos em lote, mais lentos que os de 1
  fluxo; faltava medir onde vai o tempo de uma rodada.
- **Mecanismo** `[código: multi_tune.py, multi.py, dflash2.py chain(confs=), verify.py time_rows]`:
  - `TF_GLM_MULTI_SAMPLER=streams|packed`: `packed` usa **duas** all-gathers para todos os fluxos (1: candidatos top-k
    + máximos das linhas de núcleo como fp64 em duas palavras fp32; 2: os `NUCLEUS` melhores (valor, id, massa) de cada
    linha de núcleo, massa e largura do shard), com a mesma aritmética de `nucleus_rows`.
  - `TF_GLM_MULTI_DEPTH=policy|joint|scale:A`: `joint` percorre a cadeia de cada fluxo até o máximo da política,
    calcula o *reach* (probabilidade de o rascunho k ser mantido = produto cumulativo das confianças) e **aloca as
    linhas da rodada** gulosamente, sempre ao rascunho de maior *reach*, enquanto os **tokens esperados por ms**
    crescem — custo dado pela curva medida na partida (ms da janela em lote por número de linhas, o maior dos dois
    ranks) + `TF_GLM_MULTI_OVERHEAD_MS` (12). `scale:A`: limiar de parada `min(0,95, p · S^A)` com S fluxos.
  - `TF_GLM_MULTI_ASYNC=1`: rank 0 manda a mensagem por buffer *pinned* sem ler nada de volta (o host segue para o
    trabalho da próxima rodada).
  - `TF_GLM_MULTI_LONE`: um fluxo decodificando sozinho (nenhum outro decodificando ou preenchendo, ≥ `LONE_LEFT = 16`
    tokens por vir) é movido uma vez para a "casa" dos grafos de 1 fluxo (nova op `SLOT`: copia estados KDA, conv e
    contexto DFlash2 para o slot 0; prompts guardados nas primeiras linhas são evictados e o extent é movido para a
    base 0) e verifica pelos grafos de 1 fluxo enquanto a janela couber; um segundo fluxo entra no caminho em lote na
    rodada seguinte. No patch, o default é 1.
  - `TF_GLM_MULTI_PROFILE=N`: rank 0 registra a decomposição da rodada (`plan, message, propose, verify, sample,
    accept, commit, taps, emit` + tempo de GPU da verificação por eventos CUDA) a cada N rodadas, e na partida compara
    os dois caminhos de um fluxo sozinho.
  - Correção no engine: o *bucket* de pools dos grafos esparsos usa o tamanho da *view* "home".
- **Configuração:** defaults `streams`, `policy`, `async` off, perfil off; **a receita fixa `TF_GLM_MULTI_LONE=0`**
  desde a v1.3.1.
- **Efeito medido** `[CHANGELOG v1.3.1, config.sh]`: `LONE=1` dava +0,6–0,9% a 1 fluxo, mas para usar os grafos o
  fluxo ia para as primeiras linhas do pool e evictava os prompts guardados de outras conversas: com três conversas
  revezando (~4,5k tokens cada, 6 turnos quentes), 0% dos prompts vieram do cache com 1, 98,1% com 0 (issues #12, #13).
  Os outros modos não têm número publicado.
- **Conceitos:** especulação sob lote — o custo da janela é compartilhado, então profundidade ótima por fluxo ≠ ótima
  global; maximização gulosa de tokens esperados por ms; *pipelining* host/GPU; grafos que "gravam" endereços (por que
  é preciso mover o fluxo para casa) e o custo oculto disso no cache de prompts; perfil por estágio.
- **Capítulos:** `especulacao-fundamentos` (35) — além da fórmula (1−α^{k+1})/(1−α) por fluxo, a alocação conjunta
  de linhas entre fluxos por tokens/ms; `scheduler-batching` (32) — o *trade-off* LONE (velocidade × taxa de acerto
  do cache, com os números 0% vs 98,1%); `benchmarking` (45) — perfil por estágio de rodada e medir a curva ms×linhas
  na partida; `cuda-graphs-warmup` (29) — grafos presos a slot/base.

### 0036 — `glm-tool-calls`

- **Área:** servidor/API — tool calling para clientes agentes.
- **Problemas:** (1) uma chamada longa (escrever um arquivo) não mandava nada por minutos, além do corte por
  ociosidade de clientes como LiteLLM e o `fetch` do Node; (2) o GLM escreve tool calls **dentro** do bloco `<think>`
  — a divisão original lia qualquer call num think não fechado como resposta; (3) o template do GLM-5.3 renderiza o
  raciocínio de cada passo do assistente após a última mensagem do usuário (*interleaved thinking*); um cliente que
  não devolve o raciocínio de um passo de ferramenta recebe `<think></think>` — um contexto que o modelo nunca
  escreveu e um prompt que deixa de casar com o anterior naquele passo, então o cache de prefixo relê o resto;
  (4) históricos de agentes com argumentos `""`/`null`, ou resultados cujos ids não casam com as calls (o template
  ordena por id e descarta silenciosamente o que não consegue colocar); (5) tipagem de argumentos (uniões, `const`,
  string anulável); (6) clientes que nomeiam o raciocínio `reasoning` (vLLM novo, OpenRouter) em vez de
  `reasoning_content`.
- **Mecanismo** `[código: cuda/reply_text.py, cuda/server.py, kept_reasoning.py, prompts.py, server/messages.py,
  tool_parameters.py, engine/tool_draft.py, http.py]`:
  - `GlmCallStreamer`: máquina de estados sobre `<tool_call>NOME<arg_key>K</arg_key><arg_value>V</arg_value>…
    </tool_call>` que produz *deltas* de tool call do OpenAI (cabeçalho com índice, id `call_<24 hex>` e nome; JSON
    dos argumentos em pedaços; strings conforme chegam, menos uma cauda que pode iniciar o fechamento; valores tipados
    só inteiros; texto anulável segurado enquanto ainda pode ser `null`). Blocos que não segue (outro formato, tool
    não oferecida) ficam para o parser final. (0053 depois muda para "call inteira".)
  - `ThinkSplit`: o think termina no `</think>`; calls escritas dentro dele só são calls se **terminam a resposta**
    (nada depois além de espaço e o fechamento) — regra "D5" adaptada do patch 0620 do jayleaton; até decidir, o texto
    é segurado; uma call segurada com mais de `THINK_CALL_HOLD = 1024` caracteres é assumida como call para poder ser
    transmitida.
  - `KeptReasoning`: LRU do raciocínio de respostas que fizeram tool calls, chaveado pelos ids das calls do servidor e
    por uma **assinatura** = sha256 da conversa inteira antes das calls (papéis, texto, calls sem ids) + as calls
    (mudança em relação ao original, que só usava a última mensagem do usuário: "uma resposta só volta para a conversa
    para a qual foi escrita"). `restore()` recoloca `reasoning_content` nos passos de assistente com tool calls que
    vieram sem raciocínio, antes de renderizar.
  - `agent_history`: argumentos `""`/`null` → `{}`; resultados com ids desconhecidos são renderizados na ordem enviada
    (ids removidos); argumentos string que não são objeto JSON davam `RequestError` (relaxado em 0051).
  - `normalize_messages`: `reasoning` → `reasoning_content`.
  - `tool_parameters`: `schema_types` (lista de tipos, `anyOf/oneOf`, `enum`, `const`); `typed_parameter` só quando
    todos os tipos admitidos são JSON não-string; `nullable_text` (`["string","null"]`) lê `null` como None;
    `decode_parameter` com `python=False` nos parsers do GLM (não converte `"5"`, `True`, `2.0`, para que o histórico
    reenviado renderize os mesmos tokens que o modelo escreveu); `closed_json` (um modelo pode terminar um objeto/array
    com um fechamento a menos, #87).
  - `finish_reason`: uma call transmitida dentro da qual a resposta terminou não conta como `tool_calls`.
- **Configuração:** `TF_GLM_KEEP_REASONING` (respostas guardadas, default 1024; 0 desliga), `TF_GLM_KEEP_REASONING_MB`
  (32 MiB de texto).
- **Efeito medido:** funcionalidade; `tools/toolcheck.py` confere que um parâmetro array volta como array JSON; as
  checagens de exatidão da v1.3 incluem tool calls `[CHANGELOG]`.
- **Conceitos:** formato de tool call do GLM (`arg_key`/`arg_value`), streaming SSE e deltas de tool call, raciocínio
  intercalado em loops de agente, fidelidade de re-renderização do histórico e taxa de acerto do cache de prefixo,
  tipagem por JSON Schema.
- **Capítulos:** `api-tools-estruturado` (34) — o caso GLM: calls dentro do think, streaming de calls longas, devolver
  o raciocínio que o cliente descartou, saneamento de históricos de agentes, tipagem por schema; `prefix-caching`
  (33) — por que um cliente que descarta o raciocínio quebra o prefixo (o prompt reconstruído difere do gerado) e como
  um servidor pode consertar.

### 0037 — `cuda-tokenize`

- **Área:** servidor/API.
- **Problema:** faltavam `/tokenize` e `/detokenize` (clientes escritos para o vLLM os usam) e `/v1/completions` não
  aceitava prompt como lista de ids.
- **Mecanismo** `[código: http.py, server.py]`: rotas `/tokenize`, `/v1/tokenize`, `/detokenize`, `/v1/detokenize`
  com os campos do vLLM: `prompt` (com `add_special_tokens`, default true como no vLLM) ou `messages` (renderizadas
  como na rota de chat, `add_generation_prompt` default true) → `{count, max_model_len, tokens}`; `/detokenize` →
  `{prompt}` com tokens especiais. `prompt` como lista de ids (ou lista com uma lista) é servido como dado, com
  validação de faixa do vocabulário.
- **Configuração:** nenhuma.
- **Efeito medido:** funcionalidade.
- **Conceitos:** tokenização vs renderização do chat template, tokens especiais, `max_model_len` efetivo.
- **Capítulos:** `api-tools-estruturado` (34) — endpoints auxiliares; `tokens-embeddings` (5) — `add_special_tokens`
  e "o que o servidor realmente vê" após o template.

### 0038 — `glm-kv-fp8`

- **Área:** KV cache.
- **Problema:** em BF16, o latente DSA ocupa 512 × 2 = 1.024 B por token por camada DSA, e o indexador guardava chaves
  e gates por token (2 × 128 × 2 B) em planos do tamanho do contexto, mais as chaves agrupadas; com 4 fluxos a janela
  de 1M não cabe (com `KV=bf16` a janela é 196.608; 163.840 com `VISION=1` e densos fp8/bf16 `[README]`).
- **Mecanismo** `[código: kv8.py, latent.py, sparse.py, forward.py, decode.py, segments.py, geometry.py]`:
  - **Formato de linha FP8:** um tensor `uint8 [linhas, largura + 16]`: os códigos e4m3 da linha, a **escala fp32**
    e 12 bytes zero (linhas alinhadas a 16 B). Escala = `2^(ceil(log2 amax) − 8)` (`SHIFT = 8`): o maior valor da
    linha quantiza em [128, 256], abaixo do máximo 448 do e4m3, então nada satura. Como a precisão relativa de um
    formato de ponto flutuante não depende da escala, uma potência de 2 por linha "não perde nada" para uma escala
    mais fina ou fp32, exceto os valores empurrados abaixo da faixa normal do e4m3 (2^−6), i.e. abaixo de 2^−14 do
    maior valor da linha. Nas linhas reais `kv_a`/`kv_a_layernorm` do GLM (amax ~3–5× o rms), escalas fp32 por token,
    por bloco de 128 e fixas por camada ficaram todas a 0,001 do erro relativo de 2,65% por linha desta `[código:
    docstring de kv8.py]`. O docstring cita que o indexador do DeepSeek-V3.2 guarda chaves FP8 com escalas ue8m0 por
    token, de modo análogo.
  - **Exatidão do caminho:** um único helper (`store_row`) quantiza linhas de chunk de prompt e de janela de decode —
    os bytes de uma linha são função só da sua linha bf16. Os leitores convertem os códigos para bf16 (exato) e
    **dobram a escala** nos produtos fp32: `scores *= ks` antes do softmax e `p * ks` antes do produto com V (o latente
    é K e V ao mesmo tempo) — exato porque multiplicar por potência de 2 não arredonda. Por isso "drafted == serial" e
    "resumed == fresh" continuam valendo; os valores são os dos kernels bf16 sobre as linhas dequantizadas, a menos da
    ordem de soma fp32. No `_sparse_onepass` sobre códigos FP8 o *launch* também passa a fixar os bits (layout do tile
    convertido pelo Triton), por isso fica fixo.
  - **O que fica em FP8:** o latente de cada camada DSA (528 B/token) e as chaves agrupadas do indexador (144 B por
    pool de 4 tokens); a pontuação do indexador usa os códigos nos *tensor cores* com a escala de cada pool aplicada
    aos produtos.
  - **Chaves e gates do indexador em anéis:** só o pool em que um token cai lê sua chave e seu gate, enquanto o pool
    está se completando. Então, em vez de planos do tamanho do contexto, cada slot tem um **anel** BF16 de
    `index_ring = min(capacity, ceil((rows + 3)/64)·64)` linhas (a janela mais larga + 3 linhas antes), com o token t
    na linha `slot·ring + t mod ring`; os anéis ficam com o slot quando o extent é movido. Um snapshot de prompt
    guardado leva a **cauda do anel** (`tail`: chaves e gates dos `n mod 4` últimos tokens do pool incompleto), que
    `put_ring_tail` recoloca ao restaurar ou ao assumir um prompt guardado em outro slot.
  - Estimativa (`mla_row_bytes`, `mla_index_ring`, `mla_ring_bytes`) e comparação entre ranks atualizadas; FP8 exige o
    cache latente (`TF_GLM_LATENT=1`).
- **Bytes por token** `[cálculo próprio a partir do código; por rank, 11 camadas DSA, MTP desligado em --parallel;
  o latente não é dividido entre ranks]`:

  | Layout | latente | indexador (por token) | por camada | 11 camadas |
  |---|---:|---:|---:|---:|
  | TensorFold BF16 antes do 0038 (planos completos) | 1.024 | chave+gate 512 + agrupada 64 | 1.600 | 17.600 |
  | TensorFold `KV=bf16` com anéis (0038) | 1.024 | agrupada 64 | 1.088 | 11.968 |
  | TensorFold `KV=fp8` (0038) | 528 | agrupada 144/4 = 36 | 564 | 6.204 |
  | vLLM `fp8_ds_mla` + indexador (`arch-glm.md` §4.3) | 656 | 33 | 689 | 7.579 |

  (Os anéis e o estado KDA são custos fixos por slot, não por token.) A linha de 528 B coincide com o "necessário de
  fato" que o inventário do vLLM aponta para o `fp8_ds_mla` (656 B com 128 B de RoPE zerado).
- **Configuração:** `KV=fp8` (receita) → `TF_GLM_KV=fp8`; o default do TensorFold com o patch é `bf16`.
- **Efeito medido** `[README, config.sh, CHANGELOG]`: "a janela de 1M com 4 requisições cabe"; estimativa do rank 0
  88,09 GiB; pool 2.922.496 tokens na partida medida; memória livre mínima sob um prompt de 1M: 5,6 GiB (rank 0) e
  9,5 GiB (rank 1); GSM8K 98,0%, HumanEval 97,6%, agulha de 1M encontrada; *lossy* em relação ao bf16, mas
  rascunhado == serial.
- **Conceitos:** FP8 E4M3, escala por linha potência de 2 (equivalente a uma escala UE8M0 por token), exatidão de
  escalas potência de 2, dobrar a escala no softmax online, layout de linha com padding de alinhamento, anéis para
  dados de vida curta, snapshot de estado parcial (cauda do pool).
- **Capítulos:**
  - `kv-cache-quantizado` (23), principal: o layout FP8 do TensorFold (linha `[512 códigos | escala fp32 | 12 B]`,
    `SHIFT=8`, nada satura), o argumento "escala potência de 2 por linha ≈ escala fina", o erro de 2,65% por linha,
    a tabela de bytes por token acima comparando com `fp8_ds_mla`, e o resultado (1M × 4 fluxos cabe; GSM8K/HumanEval).
  - `fp8-fp4-microescala` (21): exemplo de escala potência de 2 por linha de 512 (granularidade "por token") e por que
    multiplicar por 2^k é exato — base para entender UE8M0.
  - `mla` (12): o latente como K **e** V (um só cache), e a dobra da escala nas duas multiplicações.
  - `atencao-esparsa` (13): chaves/gates do indexador só vivem até o pool de 4 fechar → anéis por fluxo.
  - `numeros-no-computador` (4): "multiplicar por potência de 2 não arredonda (salvo sub/overflow)".

### 0039 — `glm-kda-chunked-kernel`

- **Área:** prefill / KDA (atenção linear com regra delta).
- **Problema:** a forma em chunks da KDA (0012/0014, parte A) era Triton: 5 kernels (`_kda_lm`, `_kda_solve`,
  `_kda_wu`, `_kda_state`, `_kda_out`), sub-chunks de 64 linhas, intermediários em memória global e matmuls com
  `input_precision` tf32x3. Numericamente, exponenciais de **diferenças de somas cumulativas** de log-decay são
  perigosas: com log-decays em (−5, 0) (`gate_lower_bound = −5`), G chega a −320 num sub-chunk de 64 linhas e uma
  diferença de dois G's carrega ulp(320) ≈ 3e−5 para decaimentos perto de 1 — a primeira versão teve 10× o erro da
  cadeia serial; a versão reformulada evitava isso com fatores em torno de linhas de referência. O GB10 (sm_121) dá a
  um bloco 99 KiB de shared memory (101.376 B), o que impedia *pipelining* nos laços por bloco (que pediriam até
  154 KiB) `[código: docstrings removidos de kda_chunked.py]`.
- **Mecanismo** `[código: kda_chunk.cu, kda_chunk.cpp, kda_chunked.py, kda.py, engine.py]`:
  - **Um kernel CUDA, um bloco de 256 threads por cabeça**, que percorre os sub-chunks de **32 linhas** do chunk de
    prompt em ordem, com o estado 128×128 fp32 **em registradores** (thread `(vg, kp)` guarda 4 linhas de valor × 4
    quádruplas de chave = 64 floats; produtos com S reduzem sobre 8 lanes por uma árvore de *shuffles* transpositora).
  - Por sub-chunk, entrando no estado S0 (com E(t→s) = g_{t+1}⋯g_s por canal de chave, **produto corrente** na ordem
    das linhas, como a cadeia serial decai o estado):
    `L[s,t] = β_s Σ_c k_s k_t E(t→s)` (t < s), `M[s,t] = Σ_c q_s k_t E(t→s)` (t ≤ s), `T = (I + L)^−1` por
    substituição progressiva em blocos de 16, `Y = β (v − (k E(s)) S0ᵀ)`, `V' = T Y` (as correções da regra delta),
    `O = (q E(s)) S0ᵀ + M V'` (arredondado a bf16, depois RMSNorm com gate), `S_end = S0 diag(E(31)) + V'ᵀ (k E(s→))`.
    Nenhum fator usa e^(+x) nem diferença de G's (G chega a −160 em 32 linhas).
  - Entradas por linha idênticas às do kernel serial `kda.cu`: compilado com `--fmad=false`, `expf` do CUDA, divisão
    IEEE (com um caminho rápido `div_rn` — recíproco refinado + uma correção — "conferido em 742 M pares de operandos",
    e a divisão normal nos raros casos fora de `div_ok`); conv de 4 taps + SiLU arredondada a bf16; normas L2 fp32
    (q × 128^−½); β = bf16(sigmoid(b)); decaimento `g = expf(lower · sigmoid(exp(A_log)·(a + dt_bias)))`.
  - Movimento de dados: linhas do próximo sub-chunk aquecidas na L2 (`prefetch.global.L2`) durante o atual e copiadas
    para shared com `cp.async` de 16 B; somas fp32 com FMAs explícitas e árvores de shuffle em ordem fixa →
    **determinístico**; os bits de uma linha dependem só das linhas do seu sub-chunk e do estado de entrada.
  - Por isso os chunks de prompt precisam começar em múltiplos de 32: com `TF_GLM_KDA_CHUNKED=1`, `TF_GLM_PROMPT_GRID`
    vazio ou 0 vira **64** (aceita 32, 64, 128). Decode continua no kernel serial.
- **Configuração:** `KDA_CHUNKED=1` (receita) → `TF_GLM_KDA_CHUNKED=1` (default do TensorFold com o patch: 0).
- **Efeito medido** `[README, config.sh]` (0012 + 0014 + 0039 juntos, um boot cada): prefill de 50k 29,3 → 26,4 s;
  149k 91,1 → 84,5 s; estados de prompt numa grade de 64 tokens. "Perto do kernel serial, não seus bits" (aritmética
  de prompt diferente; rascunhado == serial continua valendo); erro contra a regra delta em float64 "da ordem do
  kernel serial".
- **Conceitos:** forma em chunks da regra delta (transformação WY/UT, como em flash-linear-attention e Gated DeltaNet),
  inversa triangular por substituição progressiva, estabilidade numérica (produtos de decaimentos ≤ 1 vs
  exponencial de diferenças de cumsum), estado em registradores, `cp.async`, prefetch L2, determinismo (sem
  contração FMA, reduções em ordem fixa), limite de shared memory do sm_121.
- **Capítulos:**
  - `atencao-linear` (14), principal: a ADIÇÃO pede "forma paralela por chunks para prefill vs forma recorrente no
    decode" — acrescentar as fórmulas acima (L, M, T, Y, V', O, S_end), por que 32 linhas, por que produtos correntes em
    vez de `exp(G_s − G_t)` (o exemplo −320/ulp 3e−5 vs −160), e a exigência de grade de 32/64 para resultados
    reprodutíveis entre chunkings.
  - `kernels` (28): estudo de caso de um kernel "persistente por cabeça" com estado em registradores, staging com
    `cp.async`, prefetch L2 do próximo sub-chunk, árvore de shuffles transpositora.
  - `gpu-por-dentro` (26): 99 KiB de shared por bloco no GB10 como restrição concreta de projeto.
  - `numeros-no-computador` (4): `--fmad=false`, divisão IEEE vs rápida, e por que isso importa para bits idênticos.

### 0040 — `glm-parallel-deadlocks`

- **Área:** concorrência / comunicação entre ranks — travas (*deadlocks*).
- **Problemas** `[código: cuda/roce.py, multi.py, multi_tune.py, verify.py, engine.py]`:
  1. `RoceComm.all_gather` escolhia RoCE ou NCCL olhando o **endereço** e a contiguidade dos buffers (`data_ptr % 16`):
     dois ranks podiam escolher transportes diferentes para a mesma gather — um esperando no RoCE, o outro no NCCL,
     para sempre.
  2. As mensagens assíncronas de 0035 enviavam *views* que começavam depois do início de uma alocação
     (`buf[1:1+n]`), o que já tinha dividido as escolhas dos ranks.
  3. O perfil de partida (`_compare_paths`) rodava só no rank 0, mas reproduz grafos cujas all-gathers exigem o outro
     rank.
  4. A thread do escalonador podia morrer em silêncio numa exceção fora de `round/finish`.
- **Mecanismo:** a escolha de transporte passa a depender **só do número de bytes** (`fits_bytes`: `0 < n ≤ max_bytes`
  e múltiplo de `PACK`), igual em todos os ranks; buffers desalinhados passam por uma cópia alinhada. `_send` usa um
  buffer de device próprio para o comprimento e outro para os valores, cada um a partir do início. O perfil roda nos
  dois ranks (só o rank 0 imprime) e passa a fazer parte do vetor de configurações comparado na partida. O laço do
  escalonador captura qualquer exceção, registra e falha as requisições vivas em vez de morrer. **Watchdog:** se uma
  iteração do rank 0 (ou a espera do rank 1 por uma mensagem) passar de `TF_GLM_MULTI_WATCHDOG_S` segundos, as pilhas
  Python de todas as threads vão para o log (`faulthandler.dump_traceback_later`).
- **Configuração:** `TF_GLM_MULTI_WATCHDOG_S` (default 300; 0 desliga).
- **Efeito medido:** correção de travas; sem número de desempenho.
- **Conceitos:** coletivas precisam casar em todos os ranks (mesma operação, mesma ordem, **mesmo transporte**);
  decisões locais que dependem de algo que difere entre ranks (endereços!) quebram SPMD; diagnóstico de travas por
  watchdog/dump de pilhas.
- **Capítulos:** `nccl-rdma` (39), principal — a ADIÇÃO já fala do watchdog do NCCL; acrescentar o caso "transporte
  escolhido pelo endereço do buffer" como anti-padrão e o watchdog de aplicação; `cuda-graphs-warmup` (29) — "paridade
  entre ranks" também vale para rotinas de perfil/cronometragem que reproduzem grafos com coletivas.

### 0041 — `glm-parallel-ring-base`

- **Área:** concorrência (correção pequena, 13 linhas).
- **Problema:** depois de 0038, `SegRows` exige a base do anel do indexador de cada segmento quando há anéis
  (`fill` levanta erro sem ela); `BatchedVerify.time_rows` (cronometragem de partida usada por
  `TF_GLM_MULTI_DEPTH=joint` e pelo perfil) ainda passava `(0, 0, R)`.
- **Mecanismo:** passa `self._span(self.e.home, 0, R)`, que acrescenta `caches.ring_base(slot)`.
- **Configuração:** nenhuma (afeta `joint`/perfil).
- **Efeito medido:** correção.
- **Conceitos:** consistência de tabelas de metadados entre caminhos (captura, cronometragem, execução).
- **Capítulos:** sem conteúdo novo; no máximo uma nota em `cuda-graphs-warmup` (29) sobre manter as tabelas de
  captura/cronometragem iguais às de produção.

### 0042 — `glm-prompt-replay`

- **Área:** reuso de prompt.
- **Problema:** um estado guardado só podia ser retomado se fosse um **prefixo estrito** do novo prompt: o primeiro
  token da resposta precisa dos logits da última linha do prompt, que um estado (caches + estados KDA) sozinho não dá.
  Logo, um prompt idêntico reenviado ainda refazia pelo menos o último chunk.
- **Mecanismo** `[código: decode.py, engine.py, multi.py]`: o snapshot do **estado final** de um prompt passa a
  guardar também a **linha de logits do cabeçalho** (a fatia do vocabulário deste rank, antes de qualquer máscara de
  gramática). Uma requisição com exatamente esses ids retoma ali e amostra o primeiro token dessa linha, **sem prefill**
  (`whole(snap, prompt)`); com MTP, `last_hidden` vem da última linha pendente. No multi-fluxo, `Lane.head` guarda a
  linha e o `FILL` com `start == stop == n` "reproduz" sem rodar forward. Snapshots intermediários (sem cabeçalho)
  continuam exigindo prefixo estrito.
- **Configuração:** nenhuma própria (faz parte dos estados guardados; `TF_GLM_CACHE_ENTRIES`).
- **Efeito medido** `[README]`: um prompt idêntico de 64k reenviado: 34 s na primeira vez, **menos de 0,07 s** na
  seguinte, com a mesma resposta (a mesma amostra, pela chave derivada do prompt).
- **Conceitos:** prefix caching com acerto **completo** (o caso que motores baseados em blocos costumam tratar
  recomputando o último token), guardar a saída da última posição junto com o estado, amostragem determinística.
- **Capítulos:** `prefix-caching` (33), principal — acrescentar o "acerto total": por que um cache de estado sozinho
  não basta para o primeiro token, e a solução "guardar a linha de logits"; comparar com o APC por blocos.

### 0043 — `glm-visible-pools`

- **Área:** decode / indexador DSA.
- **Problema:** os grafos esparsos de decode são capturados por *bucket* de pools arredondado a potência de 2 (para
  o alocador reaproveitar poucos tamanhos — "tamanhos exatos fragmentavam a memória a 128k"); a pontuação e a seleção
  varriam **todas** as colunas do bucket, embora as que passam dos pools completos de cada linha valham −∞. O
  TensorFold v0.6.0 já limitava a seleção de um programa por linha (PR #140, mikolaj92), mas não a pontuação de decode
  nem a seleção dividida.
- **Mecanismo** `[código: sparse.py _visible_bound, _scores_rows(K=), _select_rows(VIS), _select_split(VIS, TOKS)]`:
  `_visible_bound(NPS, P, r, K) = min(NPS, max((P + r + 1)//4, K))` — os pools completos da linha, pelo menos K = 512,
  no máximo o bucket. A pontuação só escreve os blocos até o limite; a seleção (um programa por linha **e** a dividida)
  só varre o limite. Prova de igualdade: as colunas além do limite são −∞ e têm os maiores índices, então ficam abaixo
  de todas as mantidas (empates para o pool menor): os K melhores das primeiras `bound` colunas são os K melhores de
  todas. A seleção dividida ganhou o modo `TOKS`, escrevendo tokens e contagens diretamente (substitui operações torch
  que montavam a lista de tokens).
- **Configuração:** nenhuma (sempre ligado).
- **Efeito medido:** "os mesmos tokens" `[README]`; sem número de velocidade.
- **Conceitos:** *buckets* de forma para CUDA graphs vs trabalho real lido no device; não pagar pelo padding;
  argumento de corretude de um corte de top-k.
- **Capítulos:** `atencao-esparsa` (13) — custo do indexador cresce com os pools visíveis, não com o bucket;
  `cuda-graphs-warmup` (29) — "capture com o bucket, trabalhe com o limite real lido no device".

### 0044 — `cuda-context-errors`

- **Área:** servidor/API.
- **Problema:** a checagem de janela própria do app GLM e os prompts com imagens além da janela geravam erros sem o
  formato do OpenAI que os clientes reconhecem.
- **Mecanismo** `[código: http.py, glm app.py, server/errors.py, server/prompts.py, vision/glm.py]`: `error_body`
  inclui `param` (`messages` ou `prompt`) junto com `code` quando há código; a recusa do app GLM usa o texto
  `CONTEXT_LIMIT` da v0.6.0 (commit 68c6e35) para que `prepare` a trate como `context_length_exceeded`; prompts com
  imagens além da janela passam por `refusal()` (vira `ContextLengthError`). HTTP 400 (503 para `CapacityError`).
- **Configuração:** nenhuma.
- **Efeito medido** `[README]`: requisição cujo prompt + `max_tokens` não cabe → HTTP 400 no texto do OpenAI,
  `"code": "context_length_exceeded"` e `param` nomeando o campo.
- **Conceitos:** controle de admissão por janela, contrato de erro da API OpenAI.
- **Capítulos:** `api-tools-estruturado` (34) — erros de contexto (`code`/`param`) e por que agentes dependem deles.

### 0045 — `cuda-metrics`

- **Área:** servidor / observabilidade.
- **Problema:** o `/metrics` Prometheus da v0.6.0 não tinha os números do `/health` (fluxos, pool, prompts guardados);
  ler tabelas do decodificador enquanto a thread as altera podia falhar ("dictionary changed size during iteration").
- **Mecanismo** `[código: server/metrics.py]`: família `tensorfold_health:` — contadores `requests_total`,
  `completion_tokens_total`, `cached_tokens_total`, `rounds_total`, `prefill_seconds_total`, `decode_seconds_total`;
  gauges `context_length`, `streams_max`, `pool_tokens`, `pool_free_tokens`, `kept_prompts` e
  `streams{state=decoding|filling|paused}`; leituras com até 3 tentativas (`_read`) em `RuntimeError`; envio protegido
  contra cliente que fechou; janela efetiva preferida; ocupação de um fluxo = max(contexto, prompt + saída).
- **Configuração:** nenhuma.
- **Efeito medido:** funcionalidade (`[README]`: contadores de requisição, histogramas de latência/TTFT da v0.6.0 +
  métricas `tensorfold_health:`).
- **Conceitos:** formato texto do Prometheus (counter × gauge × histogram), leitura sem lock com nova tentativa.
- **Capítulos:** `benchmarking` (45) — a ADIÇÃO já cita `/metrics`; acrescentar os gauges de pool/fluxos como sinais
  de capacidade; `api-tools-estruturado` (34) — `/health` e `/metrics` como parte da API.

### 0046 — `glm-l2-prefetch`

- **Área:** decode / hierarquia de memória.
- **Problema:** uma camada de decode alterna matmuls limitados pela DRAM (projeções, experts roteados e compartilhado,
  cabeçalho) com fases em que a DRAM fica ociosa: as duas all-gathers por camada (a espera do RoCE), o glue das
  hyper-connections, roteador e top-k, a recorrência KDA e o núcleo da atenção latente.
- **Mecanismo** `[código: l2pf.cu, l2pf.cpp, l2pf.py, forward.py, decode.py]` (adaptado do patch 0460 do jayleaton):
  - Em **sítios** de cada camada, um kernel pequeno num **stream lateral** pede ao sistema de memória que traga para a
    L2 os pesos que o stream principal vai ler em seguida; o matmul seguinte encontra parte dos bytes na L2. O kernel
    só lê/prefetcha (nenhum *store*, atômico ou redução) e cada kernel do modelo mantém entradas, saídas e ordem — os
    bits não mudam.
  - Sítios: `a` (na all-gather do bloco de atenção: pesos HC e norma da FFN, roteador, gate/up e down do expert
    compartilhado ou do MLP denso; com o prefetch ligado o MoE roda o **expert compartilhado antes dos roteados**,
    para que o fluxo de ~100+ MB dos roteados não o expulse da L2); `f` (na all-gather da FFN: pesos HC e norma de
    entrada da próxima camada e suas projeções de entrada — KDA: proj, low-rank f/g, conv; DSA: proj, `q_b`, chaves do
    indexador; depois da última camada, norma final e cabeçalho); `o` (após as projeções de entrada do bloco de
    atenção, durante a recorrência/núcleo: a projeção de saída; na DSA, a metade "valor" de `kv_b` usada por
    `expand_v` e depois `o_proj`).
  - Orçamento por sítio `TF_GLM_L2PF_MB` (8 MiB): tensores pequenos primeiro (escalas, vieses, normas), depois os
    bytes iniciais das matrizes grandes; para uma matriz 4-bit cujo lançamento de decode cabe em **uma onda**
    (≤ `ONE_WAVE = 192` programas = 4 CTAs por SM × 48 SMs no GB10, segundo o código), prefetcha o início de **cada**
    chunk (`q4_heads`), para que todo programa comece com acertos na L2 e o programa mais lento da onda encurte.
  - Uma tabela no device de pedaços `(endereço, bytes)` ≤ 32 KiB para todos os sítios; modos `bulk`
    (`cp.async.bulk.prefetch.L2`, a unidade TMA, sm_90+), `lines` (`prefetch.global.L2::evict_last` por linha de
    128 B) ou `touch` (`ld.global.cg`). O stream lateral é **bifurcado** no sítio (`side.wait_stream(main)`) e
    **reunido** antes de o forward retornar (necessário para que os ramos de um CUDA graph capturado se juntem).
    Janelas acima de `TF_GLM_L2PF_ROWS` (64) linhas e chunks de prompt pulam todos os sítios. Ganchos também nas
    formas segmentadas (janelas em lote multi-fluxo).
- **Configuração:** `TF_GLM_L2PF=1` na receita (`bulk`; o patch deixa desligado por default), `TF_GLM_L2PF_MB`,
  `_SITES` (`afo`), `_ROWS`, `_BLOCKS`, `_THREADS`, `_KB`.
- **Efeito medido** `[config.sh, README]`: sozinho, 1 requisição: prosa 48,36 → 49,46 tok/s, código 59,54 → 61,08
  (dois boots cada). Junto com 0047 e `TF_ROCE_MAX_KB=512`: prosa 48,36 → 49,68 (+2,7%), código 59,54 → 61,49
  (+3,3%); 4 de uma vez: prosa 74,8 → 76,6, código 100,0 → 102,7 tok/s. Mesmos bits.
- **Conceitos:** decode limitado por banda; L2 como buffer de *prefetch*; esconder latência de DRAM atrás de
  comunicação; streams laterais em CUDA graphs (fork/join); TMA bulk prefetch; prioridade de evicção (`evict_last`);
  ondas e programa mais lento; ordem de execução dos experts para não poluir a L2.
- **Capítulos:**
  - `gpu-por-dentro` (26), principal: L2 do GB10 como recurso a ser **programado**; prefetch por software; ondas
    (192 programas residentes = 4 CTAs/SM × 48 SMs, segundo o patch) e por que prefetchar o início de cada chunk.
  - `kernels` (28): a ADIÇÃO já cita "streams CUDA auxiliares para sobrepor GEMMs" — acrescentar este caso (stream
    lateral que só prefetcha, sem alterar bits) e as três instruções PTX.
  - `cuda-graphs-warmup` (29): bifurcar/reunir streams dentro de um grafo capturado.
  - `nccl-rdma` (39): o tempo de espera de uma all-gather pequena como "tempo morto" a ser preenchido.

### 0047 — `glm-exl3-decode-loads`

- **Área:** decode / kernel de experts roteados EXL3.
- **Problema:** o kernel de decode dos experts (`dec_kernel`) lia o *trellis* (os pesos EXL3 empacotados) com cargas
  `__ldg` de 32 bits (a palavra `lane` de cada tile), um passo de k à frente, e só depois da cadeia de leituras
  dependentes contagem → item → linhas-membro (latência acumulada).
- **Mecanismo** `[código: exl3.cu, exl3.cpp, exl3_mm.py]` (adaptado do patch 0580 do jayleaton): com
  `TF_GLM_EXL3_LOADS=nc`, o trellis vem em cargas de **16 bytes** `ld.global.nc.L1::no_allocate.v4.u32` (caminho de
  leitura não-coerente/somente-leitura, sem alocar na L1 porque cada palavra é lida uma vez por janela), **PD passos
  de k à frente** (`nc`=1, `nc2`=2, `nc4`=4; parâmetro de *template*): a lane l lê os bytes 16·l… de cada 512 B de um
  passo, NT/4 cargas por lane; as palavras passam por uma área de *staging* de 1 KB na fatia do warp do buffer de
  redução `red` (dois `__syncwarp` por passo) e são relidas como a palavra `lane` de cada tile, o layout que
  `decode_tile`/`mma16816` esperam. O item é lido junto com sua contagem numa só ida e volta (os itens são
  dimensionados para a grade) e as palavras dos primeiros passos são emitidas **antes** de as linhas-membro chegarem
  à shared memory. `asm volatile` mantém a posição das cargas em relação ao `mma` (volátil), para que a profundidade
  do anel seja a do código. Só movimento de dados: mesmos bits.
- **Configuração:** `TF_GLM_EXL3_LOADS=nc` na receita (`0`/`ldg` = cargas de 32 bits do TensorFold; `nc2`, `nc4`).
- **Efeito medido** `[config.sh]`: sozinho, prosa 48,36 → 48,78 tok/s, código 59,54 → 60,42; com 0046 e
  `TF_ROCE_MAX_KB=512`, ver 0046 (+2,7% / +3,3%; 4 fluxos 74,8 → 76,6 e 100,0 → 102,7). Mesmos bits.
- **Conceitos:** cargas vetoriais de 128 bits e coalescência, caminho de leitura não-coerente (`ld.global.nc`), não
  alocar na L1 dados de uso único, *software pipelining* (distância de prefetch em registradores), esconder latência de
  leituras dependentes (*pointer chasing*), reorganizar dados entre o layout de carga e o layout do `mma.sync`
  m16n8k16 via shared memory.
- **Capítulos:** `hadamard-trelica-exl3` (22), principal — "decodificação no kernel": como o trellis chega aos
  registradores, e que a velocidade do decode EXL3 depende de como se lê o trellis; `kernels` (28) — anatomia de
  mainloop (a ADIÇÃO já cita cp.async e ldmatrix): cargas `.nc.v4` com staging por warp; `gpu-por-dentro` (26) —
  L1/caminho read-only.

### 0048 — `glm-timing-tokens`

- **Área:** concorrência (cronometragens de partida).
- **Problema:** as cronometragens das janelas de verificação na partida (`time_rows`, usadas por
  `TF_GLM_MULTI_DEPTH=joint` e pelo perfil) usavam uma janela de um token repetido (0): toda linha roteia para os
  mesmos poucos experts, e uma janela de 16 linhas era cronometrada em **cerca de metade** do que 16 linhas reais
  custam `[código: docstring de verify.py]`.
- **Mecanismo:** `timing_tokens(vocab, n)`: ids fixos e distintos (`np.random.default_rng(1234).choice(min(vocab,
  150000), n, replace=False)`); usados também no comparativo de caminhos de 0035.
- **Configuração:** afeta `TF_GLM_MULTI_DEPTH=joint` e `TF_GLM_MULTI_PROFILE`.
- **Efeito medido:** correção da curva de custo (a medida antiga subestimava ~2×).
- **Conceitos:** em MoE o custo de um lote depende de **quantos experts distintos** ele toca; entradas sintéticas
  precisam ser representativas.
- **Capítulos:** `benchmarking` (45) — armadilha "entrada degenerada em MoE"; `mixture-of-experts` (11) — custo de
  decode cresce com o número de experts distintos ativados pelo lote.

### 0049 — `glm-multi-prefill`

- **Área:** concorrência / prefill de vários prompts.
- **Problema:** sem isto, prompts concorrentes eram preenchidos um fluxo por vez, intercalados com rodadas de decode:
  o último de quatro prompts curtos que chegam juntos esperava **oito** forwards (um prompt de 130 tokens são dois
  chunks: até seu ponto de grade de 64 tokens, onde o estado é guardado, e a cauda) mais as rodadas entre eles. O custo
  de um forward é quase fixo em comprimentos curtos (~0,12 s nos dois Sparks: os pesos são lidos uma vez por camada,
  quaisquer que sejam as linhas) `[código: docstring de multi_prefill.py]`.
- **Mecanismo** `[código: multi_prefill.py, multi.py, forward.py kda_rows/dsa_rows/split_layers(mixer=)]`:
  - Um chunk de prompt "multi-prompt" põe as próximas linhas de vários prompts em sequência (`[s1][s2]…`) e roda **um
    forward** nos buffers de prompt: o trabalho independente por linha roda uma vez sobre todas (embedding, glue HC
    dividido entre ranks, frentes, projeções de saída, roteador e experts roteados — "uma leitura de expert para todos
    os prompts" —, expert compartilhado, norma final); o que lê/escreve o estado de um fluxo roda **pedaço por
    pedaço** no `State` daquele fluxo, pelas mesmas chamadas de um chunk próprio (`kda_rows`: recorrência KDA no
    kernel em chunks ou serial, *commit* do estado e deslocamento da conv; `dsa_rows`: escrita do latente, atualização
    do indexador, absorção, atenção densa/esparsa e expansão, na posição do fluxo, com seu extent, anel e número de
    chunks). Depois: logits do cabeçalho na última linha de cada pedaço que termina seu prompt (matmul de uma linha,
    como sozinho), taps do DFlash2, *commit*. Logo, os bits de cada pedaço = os de um chunk próprio na mesma posição.
  - Política (`_group`, rank 0): primeiro plano antes de background, depois **menos linhas restantes primeiro** (o
    primeiro token de um prompt curto não espera atrás de um longo; o longo leva o resto do orçamento), depois ordem
    de admissão; cada um vai até seu próximo ponto (ponto guardado ou fim) se couber, senão o orçamento restante em
    múltiplos de `fill_unit = lcm(64, grade)`. Orçamento = o de um chunk de um fluxo (`TF_GLM_FILL_ROWS` com outros
    decodificando, senão o chunk do engine). Prompts com imagens e *replays* (0042) preenchem sozinhos. Nova op `MFILL`;
    primeiros tokens amostrados juntos (`sample_streams`). `TF_GLM_MULTI_PREFILL_WAIT_MS`: um servidor ocioso pode
    esperar mais chegadas depois da primeira (default 0). `/health` ganha `multi_prefill` (chunks, pedaços, linhas).
- **Configuração:** `MULTI_PREFILL=1` (receita) → `TF_GLM_MULTI_PREFILL=1` (default do patch 0);
  `TF_GLM_MULTI_PREFILL_WAIT_MS` (0). Exige o cache latente.
- **Efeito medido** `[config.sh, README]` (sparkDash): prosa com 4 simultâneas 103,4 → 108,8 tok/s, TTFT 590 → 340 ms;
  estruturado com 3 / 4 simultâneas 175,2 → 196,3 e 196,3 → 227,9 tok/s; uma requisição inalterada. Exato.
- **Conceitos:** prefill em lote de sequências de comprimentos diferentes (análogo ao *varlen*/`cu_seqlens` e ao
  chunked prefill com várias sequências), amortizar a leitura de pesos (prefill curto é limitado por banda, não por
  FLOPs), *shortest-remaining-first*, separação "independente por linha" × "dependente de estado".
- **Capítulos:** `scheduler-batching` (32), principal — agrupar prompts num chunk e a política "menos linhas restantes
  primeiro" com os números de TTFT; `prefill-decode-kv` (10) — por que prefill **curto** não é compute-bound (~0,12 s
  fixos por forward) e o ganho de agrupar.

### 0050 — `glm-many-media`

- **Área:** visão (imagens e vídeo).
- **Problema:** limites de 4 imagens e 2 vídeos por requisição, corpo de 32 MiB e 20 MB de imagens; pico de memória
  alto com muitas imagens (todas decodificadas em tamanho cheio antes de redimensionar; a saída da torre copiada mais
  uma vez num `torch.cat`).
- **Mecanismo** `[código: vision/glm.py, vision/images.py, server/prompts.py, http.py]`: `GlmVisionLimits` ganha
  `max_images` 50, `max_videos` 4, `request_image_tokens` 16.384 e `request_video_tokens` 32.768, divididos em partes
  iguais: `picture_tokens(n) = max(16, min(2048, 16384 // n))` e `clip_tokens(n) = max(64, min(16384, 32768 // n))`
  — até 8 imagens e 2 vídeos mantêm os tetos cheios, então seus *canvases* e bits não mudam. Corpo até 96 MiB;
  imagens até 64 MB no total. Cada imagem é decodificada e **ajustada ao seu canvas enquanto é decodificada** (só uma
  imagem em tamanho cheio na memória do host por vez); a torre escreve suas linhas direto na saída pré-alocada da
  requisição (sem a cópia extra do `cat`). Tokens por imagem = `(H/28)·(W/28)` do canvas (patch 14 com fusão 2×2).
- **Configuração:** `TENSORFOLD_GLM_MAX_IMAGES` (50), `TENSORFOLD_GLM_MAX_VIDEOS` (4),
  `TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS` (16.384), `TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS` (32.768), além de
  `TENSORFOLD_GLM_IMAGE_TOKENS` (2.048), `_VIDEO_TOKENS` (16.384), `_VIDEO_FRAMES` (128).
- **Efeito medido** `[CHANGELOG v1.1, README]`: 50 imagens numa requisição respondidas em 12,5 s (16.239 tokens de
  prompt), memória livre mínima no head 6,75 GiB; 4 vídeos 6,90 GiB; com 50 imagens cada uma recebe 327 tokens; com
  3 ou 4 vídeos, 10.922 ou 8.192. Respostas com até 4 imagens e 2 vídeos idênticas à v1.0.
- **Conceitos:** orçamento de tokens visuais por requisição, redimensionamento dinâmico (smart resize), pico de memória
  em memória unificada.
- **Capítulos:** `multimodal` (17), principal — orçamento de tokens dividido entre mídias de uma requisição, contas
  (2.048/imagem até 8; 327 com 50) e a técnica de reduzir pico (ajustar ao decodificar, saída pré-alocada);
  `memoria-unificada` (27) — pico de host + GPU no mesmo pool.

### 0051 — `glm-tool-history-recovery`

- **Área:** servidor/API — histórico de tool calls.
- **Problema (issue #2):** uma tool call no histórico cujos argumentos não são um objeto JSON (string truncada ou
  corrompida pelo modelo, cliente ou proxy) dava HTTP 400 em **todo** turno seguinte, porque agentes reenviam o
  histórico — a conversa ficava bloqueada para sempre.
- **Mecanismo** `[código: prompts.py agent_history]`: a call é **deixada fora** do prompt junto com seu resultado
  (casado pelo id; sem ids, pela posição quando o número de resultados é igual ao de calls) e registrada no log; a
  mensagem mantém as outras calls, conteúdo e raciocínio (sem calls restantes, a chave `tool_calls` some).
- **Configuração:** nenhuma.
- **Efeito medido:** corrige a issue #2 `[CHANGELOG v1.3]`.
- **Conceitos:** robustez de loops de agente; *fail-open* sanitizado vs *fail-closed* no histórico.
- **Capítulos:** `api-tools-estruturado` (34) — o escopo já cita "recuperação de tool calls malformadas, fail-closed"
  no DeepSeek; acrescentar o contraponto: no histórico, recusar mata a conversa; omitir com log a mantém viva.

### 0052 — `cuda-roce-startup`

- **Área:** link entre os Sparks (RoCE).
- **Problema (issue #6):** com `COMM=roce`, a primeira partida morria na primeira all-gather: cada rank compila suas
  extensões CUDA por conta própria e os dois podiam se afastar dezenas de segundos — além dos ~20 s que uma espera
  RoCE permite (`SPIN_LIMIT = 20.000.000` sondagens de flag de ~1 µs, após o que a espera "envenena" o runtime).
- **Mecanismo** `[código: cuda/roce.py, engine.py]`: até o engine estar construído (`settle()`), cada all-gather RoCE
  eager primeiro espera o par numa **barreira NCCL** (que espera sem esse limite); gathers capturadas em grafo nunca
  fazem isso. O kernel RoCE é compilado durante o setup (antes da all-gather de veredito). `TF_ROCE_WAIT_S` torna a
  espera configurável (1–3600 s; grafos mantêm o valor de captura). Falhas imprimem os contadores do proxy
  (doorbell, postados, completados, último; por HCA, completados e bytes).
- **Configuração:** `TF_ROCE_WAIT_S` (20), `COMM=roce`.
- **Efeito medido** `[README, CHANGELOG]`: uma primeira partida cujos ranks se afastam compilando kernels não falha mais.
- **Conceitos:** RDMA *one-shot* com a GPU girando em flags (sem a semântica de timeout do NCCL), barreira para
  realinhar ranks, *skew* de compilação JIT entre nós, diagnóstico por contadores.
- **Capítulos:** `nccl-rdma` (39), principal — o all-gather por escrita RDMA + espera em flag (11 µs vs 45 µs do NCCL
  para 16 KiB, da parte A/0006) e seus modos de falha; `cuda-graphs-warmup` (29) — tempo de compilação diferente entre
  ranks na primeira partida.

### 0053 — `glm-whole-tool-calls`

- **Área:** servidor/API — tool calls transmitidas.
- **Problema:** com os deltas de argumentos do 0036, uma resposta cortada por `max_tokens` dentro de uma call enviava
  argumentos cortados, que um cliente poderia guardar ou executar.
- **Mecanismo** `[código: reply_text.py GlmCallStreamer.feed/_follow, server.py]`: o streamer continua seguindo a call
  enquanto ela é escrita, mas **envia cada call inteira** (cabeçalho + argumentos num delta) quando chega o
  `</tool_call>`; uma call dentro da qual a resposta termina nunca é enviada (a resposta termina com `length`), em
  streaming ou não; o parser final também corta um `<tool_call>` não fechado do conteúdo. Enquanto uma call está
  segurada, um delta vazio sai a cada `CALL_KEEPALIVE_S = 2,0` s para clientes com corte por ociosidade (#114 do
  upstream).
- **Configuração:** nenhuma (constante 2,0 s).
- **Efeito medido:** funcionalidade `[README, CHANGELOG v1.3]`.
- **Conceitos:** atomicidade de tool calls em streaming, *keepalive* de SSE.
- **Capítulos:** `api-tools-estruturado` (34) — "call inteira ou nada" + keepalive (contraste com o streaming de
  argumentos do 0036).

---

## 5. Mapa por capítulo (o que acrescentar, agregado)

| Capítulo (id) | Acréscimos sugeridos (patches) |
|---|---|
| `mapa-da-jornada` (0) | Uma quarta linha na tabela das receitas: GLM-5.3-Flash EXL3 em **TensorFold** (1M/req., 4 fluxos, KV FP8, 60,4 → 108,8 tok/s agregados de 1 → 4 fluxos), marcando que o motor muda, não o modelo. |
| `numeros-no-computador` (4) | Não-associatividade na prática: cuBLAS/`topk` dependentes do tamanho do lote (0027); atômicos inteiros determinísticos × float (0029); multiplicar por 2^k é exato (0038); `--fmad=false` e divisão IEEE (0039). |
| `tokens-embeddings` (5) | `/tokenize` com `add_special_tokens` e mensagens renderizadas (0037). |
| `geracao-amostragem` (9) | Amostragem chaveada por posição (§2.2); top-p exato com vocabulário dividido (massas inteiras, teste de cobertura, união dos candidatos) (0034, 0035 `packed`). |
| `prefill-decode-kv` (10) | Prefill curto é limitado pela leitura de pesos (~0,12 s por forward) → agrupar prompts (0049). |
| `mixture-of-experts` (11) | Custo de decode depende do número de experts distintos tocados pelo lote (0048). |
| `mla` (12) | Latente = K e V; dobra da escala FP8 nas duas multiplicações (0038). |
| `atencao-esparsa` (13) | Seleção top-512 pools por radix select de 8 bits e versão dividida para decode longo (0029); pontuação em grid-stride com query em registradores (0031); limite de pools visíveis × bucket do grafo (0043); anéis para chaves/gates do indexador (0038). |
| `atencao-linear` (14) | Kernel em chunks WY/UT de 32 linhas, fórmulas L/M/T/Y/V'/O/S_end, produtos correntes vs exp(ΔG), grade de 64 (0039); estado com paridade dupla + replay (0030). |
| `hiperconexoes` (15) | Glue HC dividido por linhas entre ranks com sobreposição e "frentes" do próximo bloco (0033). |
| `multimodal` (17) | Orçamento de tokens visuais compartilhado (50 imagens / 4 vídeos), redução de pico de memória (0050). |
| `fp8-fp4-microescala` (21) | Escala potência de 2 por linha de 512 como caso de granularidade por token (0038). |
| `hadamard-trelica-exl3` (22) | Leitura do trellis no decode: cargas `.nc` de 16 B, k passos à frente, staging por warp (0047). |
| `kv-cache-quantizado` (23) | Layout FP8 do TensorFold, erro 2,65%/linha, tabela de bytes/token vs `fp8_ds_mla`, 1M × 4 cabe (0038). |
| `gpu-por-dentro` (26) | L2 programável (prefetch em stream lateral, ondas de 192 programas no GB10) (0046); 99 KiB de shared por bloco (0039); caminho read-only/no-allocate (0047). |
| `memoria-unificada` (27) | Pico de memória da visão (0050); `MEMORY_RESERVE_GIB` e pool (contexto em 0030). |
| `kernels` (28) | Split-K e parciais desnecessários (0028); kernel KDA persistente por cabeça (0039); stream lateral de prefetch (0046); cargas vetoriais `.nc` (0047); tile define bits, número de programas não (0031). |
| `cuda-graphs-warmup` (29) | Grafos com tabelas no device (0029, 0030); linhas lixo para padding (0027); padding de propostas até a largura capturada (0032); grafos presos a slot/base e o custo no cache (0035); fork/join de stream lateral (0046); bucket × limite real (0043); paridade entre ranks em rotinas de perfil (0040); skew de compilação (0052). |
| `vllm-arquitetura` (30) | Quadro "outro motor": TensorFold (rank 0 decide/serve, rank 1 aplica, mensagens de operações, comparação de configurações na partida) (0030). |
| `paged-attention-kv` (31) | Extents contíguos de 2048 tokens × paginação; fórmula do pool; estimativa geométrica sem profile run (0028, 0030). |
| `scheduler-batching` (32) | Iteração chunk-ou-rodada com crédito, teto de 32 linhas, pausa, devolução do mais novo, FIFO, background (0030); trade-off LONE (0035); prefill multi-prompt e "menos linhas restantes primeiro" (0049). |
| `prefix-caching` (33) | Prompts guardados no pool (0030); acerto total com linha de logits guardada (0042); raciocínio descartado quebra o prefixo (0036); taxa de acerto 0% × 98,1% (0035). |
| `api-tools-estruturado` (34) | Calls no think, streaming/atomicidade de calls, keepalive, raciocínio devolvido, saneamento de histórico, tipagem (0036, 0051, 0053); `/tokenize` (0037); `context_length_exceeded` + `param` (0044); `/metrics` (0045). |
| `especulacao-fundamentos` (35) | Verificação por igualdade sob amostragem chaveada (§2.2); especulação multi-fluxo com teto global e alocação conjunta por tokens/ms (0030, 0035); refinamentos de prompt lookup (0032). |
| `dflash-dspark` (37) | DFlash2 multi-fluxo: pool de anéis, conv/atenção segmentadas, ops dependentes de linhas por fluxo, merge de candidatos entre ranks (0027). |
| `paralelismos` (38) | Paralelismo por linhas do glue HC (0033); amostragem com lm_head vocab-parallel (0034). |
| `nccl-rdma` (39) | Casar coletivas e o anti-padrão "transporte pelo endereço" (0040); RoCE one-shot com espera em flag, barreira de partida, `TF_ROCE_WAIT_S`, contadores (0052); tamanho por linha × `TF_ROCE_MAX_KB` (0032). |
| `engenharia-de-receita` (41) | Patches como diffs contra `site-packages` aplicados com `patch -p0` em ordem; hash da imagem a partir dos patches; imagem pinada por digest; vetor de configurações comparado entre ranks; renumeração e "aposentadoria" de patches absorvidos pelo upstream (contexto §0). |
| `receita-glm` (42) | Seção "variante TensorFold": `start.sh`/`config.sh`, defaults (PARALLEL=4, CONTEXT=1M, KV=fp8, DENSE=q4, DRAFTER=dflash2, COPY/COPY_CODE, SPLIT, KDA_CHUNKED, L2PF, EXL3_LOADS=nc, MULTI_PREFILL, ROCE_MAX_KB=512), números e por que cada um. |
| `benchmarking` (45) | sparkDash pela API, clocks travados a 2.200 MHz; perfil por estágio de rodada (0035); entradas degeneradas em MoE (0048); métricas `tensorfold_health:` (0045). |

---

## 6. Lacunas (conceitos sem um lar óbvio no currículo atual)

1. **Invariância de lote como princípio de projeto.** O currículo trata determinismo pontualmente (patch de
   determinismo do Qwen, não-associatividade). O TensorFold faz da igualdade bit a bit entre "servido em lote" e
   "servido sozinho" uma exigência de engenharia, com técnicas concretas (§2.1). Sugestão: uma seção dedicada em
   `numeros-no-computador` (4) ou em `kernels` (28), referenciada por 32 e 35.
2. **Motor de inferência alternativo ao vLLM.** `vllm-arquitetura` (30) é sobre o vLLM; a receita TensorFold mostra
   outro desenho (engine por família de modelo, SPMD líder/seguidor, sem PagedAttention). Sugestão: um quadro
   comparativo em 30 e uma "variante" em `receita-glm` (42), ou um capítulo novo "Outro motor: TensorFold".
3. **Verificação por igualdade sob amostragem chaveada.** `especulacao-fundamentos` (35) ensina rejeição min(1, p/q);
   a abordagem do TensorFold (amostrar o alvo com ruído chaveado em todas as linhas e aceitar enquanto iguala) dá
   saída idêntica à serial e permite preempção por recomputação — merece uma subseção.
4. **Extents contíguos × paginação.** `paged-attention-kv` (31) pode ganhar o contraponto: em modelos híbridos com
   poucos fluxos e janelas enormes, extents de 2048 tokens + compactação são uma alternativa simples.

---

## 7. Números citáveis (consolidados, com fonte)

| Número | Contexto | Fonte |
|---|---|---|
| 60,4 / 79,2 / 89,5 / 108,8 tok/s | prosa agregada, 1/2/3/4 fluxos | README |
| 114,7 / 147,6 / 196,3 / 227,9 tok/s | estruturado agregado, 1/2/3/4 fluxos | README |
| 170 / 269 / 330 / 340 ms | TTFT prosa, 1/2/3/4 fluxos | README |
| 11/11 + 11/11 | respostas concorrentes == solo (escalonadas, rajada) | README |
| 2.922.496 tokens | pool compartilhado na partida medida (teto 12,5 GiB) | README / CHANGELOG v1.3 |
| 88,09 GiB | estimativa de partida do rank 0 | README |
| 5,6 / 9,5 GiB | memória livre mínima sob prompt de 1M, rank 0 / rank 1 | README |
| 196.608 / 163.840 | janela com `KV=bf16` (sem / com visão + densos fp8/bf16) | README |
| 98,0% / 97,6% | GSM8K (250) / HumanEval (164) com KV FP8 + densos q4 | README |
| 981.841 tokens, 967 s | agulha encontrada a ~1M | README |
| 528 B / 144 B | linha FP8 do latente / da chave agrupada (512+16, 128+16) | código 0038 |
| 2,65% | erro relativo por linha da escala potência de 2 nas linhas `kv_a` do GLM | código 0038 |
| 6.204 B/token/rank | KV DSA `KV=fp8`, 11 camadas (vs 7.579 `fp8_ds_mla`) | cálculo próprio |
| ≈177 MiB | custo de um slot de fluxo extra por rank (estados KDA etc.) | cálculo próprio (`engine.slot_bytes`) |
| ~45 MiB | reserva por prompt guardado na partida (32 entradas ≈ +1 GiB vs 8) | CHANGELOG v1.3.2 |
| 0% vs 98,1% | prompts servidos do cache com `TF_GLM_MULTI_LONE` 1 vs 0 | CHANGELOG v1.3.1 |
| +0,6–0,9% | ganho de `TF_GLM_MULTI_LONE=1` a 1 fluxo | CHANGELOG v1.3.1 |
| 103,4 → 108,8 tok/s; 590 → 340 ms | `MULTI_PREFILL`, 4 prosa | config.sh / README |
| 175,2 → 196,3; 196,3 → 227,9 tok/s | `MULTI_PREFILL`, estruturado 3 / 4 | config.sh |
| ~0,12 s | custo quase fixo de um forward curto nos dois Sparks | código 0049 |
| 23% vs 74% | rascunhos mantidos: cópia da resposta vs do prompt (39 respostas de código) | código 0032 |
| 55,4 → 56,2; 120 → 125 tok/s | `COPY_CODE`: código; edição | README / config.sh |
| 92,8 → 90,7 s | prompt de 149k, overlap nível 2 | config.sh |
| 29,3 → 26,4 s; 91,1 → 84,5 s | KDA em chunks (0012+0014+0039), 50k; 149k | README / config.sh |
| −320 vs −160; ulp(320) ≈ 3e−5 | G mínimo em sub-chunks de 64 vs 32 linhas | código 0039 |
| 742 M | pares de operandos conferidos para `div_rn` | código 0039 |
| 101.376 B (99 KiB) | shared memory por bloco no GB10 (sm_121) | código 0039 |
| 34 s → <0,07 s | prompt idêntico de 64k reenviado (replay) | README |
| 48,36 → 49,46; 59,54 → 61,08 | L2 prefetch sozinho, prosa; código (1 req.) | config.sh |
| 48,36 → 48,78; 59,54 → 60,42 | cargas `nc` sozinhas, prosa; código | config.sh |
| 49,68 (+2,7%); 61,49 (+3,3%) | L2PF + `nc` + 512 KiB, prosa; código | README / config.sh |
| 74,8 → 76,6; 100,0 → 102,7 | idem com 4 fluxos, prosa; código | README / config.sh |
| 192 programas | uma onda no GB10 (4 CTAs/SM × 48 SMs, segundo o código) | código 0046 |
| ~metade | custo cronometrado de janela de 16 linhas com token repetido vs real | código 0048 |
| 12,5 s; 16.239 tokens | 50 imagens numa requisição | CHANGELOG v1.1 |
| 6,75 / 6,90 GiB | memória livre mínima com 50 imagens / 4 vídeos | CHANGELOG v1.1 |
| 327; 10.922 / 8.192 | tokens por imagem com 50; por vídeo com 3 / 4 | README |
| 20.000.000 sondagens (~20 s) | limite de espera RoCE padrão (`TF_ROCE_WAIT_S`) | código 0052 |
| 300 s | watchdog padrão do multi-fluxo | código 0040 |
| 2,0 s | keepalive durante call segurada | código 0053 |

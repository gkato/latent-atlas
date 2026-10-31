# TensorFold na receita GLM-5.3-Flash EXL3 — patches 0001–0026 (parte A)

> **Escopo.** Os 26 primeiros dos 53 patches de `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`
> (release v1.3.2, imagem `v0.6.0-ae8d1c789b47`, digest `sha256:22789f0c…`). Cada patch é um diff contra o
> `site-packages` do **TensorFold v0.6.0** (`ashhart/TensorFold`), aplicado com `patch -p0` em ordem de nome de
> arquivo por `scripts/prepare.sh` (README, "What the patches change"). Os patches 0027–0053 ficam fora deste arquivo
> (parte B); aparecem aqui só como referência cruzada quando completam ou alteram algo de 0001–0026.
>
> **Fontes lidas integralmente:** `patches/0001…0026-*.patch` (11.637 linhas), `README.md`, `CHANGELOG.md`,
> `scripts/config.sh`. Para contraste com a receita vLLM: `docs/pesquisa/glm-launch.md`, `glm-mods.md`, `arch-glm.md`.
>
> **Convenções de citação:** `[README]`, `[CHANGELOG vX]`, `[config.sh]` = texto do repositório;
> `[código: arquivo]` = comentário/docstring dentro do patch; `[cálculo próprio]` = conta minha a partir de valores
> exatos; `[inferência]` = dedução minha, não afirmada pela fonte. Nenhum número foi inventado: onde não há medição
> publicada, está escrito "sem número publicado".
>
> **Nomes:** identificadores, flags, variáveis de ambiente, código e fórmulas ficam como no original.

---

## Sumário

- [0. Contexto: de vLLM para TensorFold](#0-contexto-de-vllm-para-tensorfold)
  - [0.1 Comparação de alto nível](#01-comparação-de-alto-nível-das-duas-receitas)
  - [0.2 Invariantes do TensorFold que explicam quase todos os patches](#02-invariantes-do-tensorfold-que-explicam-quase-todos-os-patches)
  - [0.3 Geometria por rank para ler os kernels](#03-geometria-por-rank-para-ler-os-kernels)
- [1. Tabela-resumo dos 26 patches](#1-tabela-resumo-dos-26-patches)
- [2. Fichas por patch](#2-fichas-por-patch)
- [3. Cruzamento com a tabela "What the patches change" do README](#3-cruzamento-com-a-tabela-what-the-patches-change-do-readme)
- [4. Variáveis de ambiente introduzidas por 0001–0026](#4-variáveis-de-ambiente-introduzidas-por-00010026)
- [5. O que acrescentar em cada capítulo do Latent Atlas](#5-o-que-acrescentar-em-cada-capítulo-do-latent-atlas)
- [6. Pontos de atenção, divergências e armadilhas](#6-pontos-de-atenção-divergências-e-armadilhas)

---

## 0. Contexto: de vLLM para TensorFold

### 0.1 Comparação de alto nível das duas receitas

A receita anterior (`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`) servia o mesmo checkpoint EXL3 num fork do vLLM com
overlays Python e extensões; a nova troca o motor inteiro por TensorFold, um motor escrito sob medida (famílias
`glm5_next`, `qwen`, MLX e CUDA), mais 53 patches.

| Aspecto | Receita vLLM (glm-mods.md / glm-launch.md) | Receita TensorFold (README / config.sh) |
|---|---|---|
| Motor | fork vLLM `glm53-flash` + overlays (`overlay/*.py`), extensões (MoE cooperativo, E3) | TensorFold v0.6.0 + 53 patches (`patch -p0`, ordem de nome) |
| Distribuição | TP=2 do vLLM, all-reduce NCCL por camada | 2 ranks (1 por Spark); rank 0 serve a API e decide, rank 1 "segue" (`follow`); parciais fp32 por all-gather + soma em ordem de rank `[código: tensorfold/cuda/comm.py]` |
| Especulação | DFlash2 k=7 + adaptive-k (`patch_adaptive_k`) | DFlash2 com regra `fnc7:0.3` + copy drafts (prompt lookup) até 15 por rodada; MTP opcional |
| Projeções densas | `GLM53_DENSE_FP8` (Marlin W8A16 por canal) | `DENSE=q4` (4 bits afins, grupos de 64, faixa buscada por MSE), cabeça FP8 |
| KV | `fp8_ds_mla` 656 B/token/camada | FP8 e4m3 por linha com escala potência de 2 (patch 0038, parte B) |
| Contexto / concorrência | 850.000 padrão; 1M cabe com `EXL3_FAT_GROUPED=0` | 1.048.576 por requisição, 4 requisições sobre um pool compartilhado de ~2,9 M tokens (2.922.496 no start medido) |
| Decode, prosa ×1 (sparkDash) | 36,1 tok/s (2026-09-17, 850k, adaptive-k + dense,kda + coop) | 60,4 tok/s (GPU travada em 2.200 MHz) |
| Decode, 4 concorrentes, prosa (agregado) | 75,3 tok/s (2026-09-17) | 108,8 tok/s |
| Decode estruturado ×1 | 62,9 tok/s (2026-08-28, k=7, temp 0) | 114,7 tok/s |
| Prefill frio | E3: 1.428–1.587 tok/s de 8k a 256k (2026-09-07; 8k 1.492,1, 32k 1.428,2, 64k 1.587,0) | 1.952 (8k) … 1.942 (65k) … 1.642 (262k) tok/s; 981.841 tokens a 1.015 tok/s (needle encontrado) |

> **Cuidado:** são datas, workloads e configurações diferentes; não é um A/B. Servem só para situar a ordem de
> grandeza. Os deltas por patch citados nas fichas abaixo também foram medidos em builds intermediárias (ver §6).

### 0.2 Invariantes do TensorFold que explicam quase todos os patches

Quase todo comentário dos patches termina em "the same bits" ou explica por que não. Isso decorre de cinco
invariantes do motor, que valem a pena ensinar antes dos patches:

1. **Rascunho == serial, bit a bit.** "Drafts only propose: every drafted token is checked against the model's own
   keyed sample, so drafted replies equal TensorFold's serial, one-token-at-a-time decoding (send `"draft": false`
   for that reference)" `[README, Checks]`. Isso exige:
   - **amostragem chaveada**: o sorteio de cada posição é determinado por `(seed, posição, token)` — "Sampling is
     keyed by (seed, position, token)" `[código: draft_sim.py]`; sem `seed`, a chave vem do prompt `[README]`. A
     verificação aceita um rascunho só se ele for **igual** à amostra chaveada do alvo naquela posição (não é a
     amostragem por rejeição de Leviathan/Chen; é acoplamento por ruído compartilhado — `[inferência]` sobre a
     classificação, o mecanismo está no código);
   - **invariância a lote/janela**: os bits de uma linha não podem depender das outras linhas da janela de
     verificação nem da largura da janela ("a row's bits never depend on the other rows"). Por isso split-K tem
     partição fixa pela forma, tiles são fixos, e cada kernel novo declara se preserva isso.
2. **Ranks bit-iguais.** "NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it
   keeps ranks bit-equal" `[código: comm.py]`. Em vez de all-reduce, cada rank recebe os parciais fp32
   `[world, R, D]` e soma rank 0 primeiro, de modo que os dois ranks amostram o mesmo token sem trocar o token.
3. **Rank 1 espelha o rank 0.** O rank 0 monta um "header" de inteiros (max_tokens, seed em 3 pedaços, temperatura
   como dois int32, top-k/p, política, posições de imagem, pontos de prefixo compartilhado…) e o publica (`_share`);
   o rank 1 fica ocioso numa "campainha" (`_ring` / `_await_bell`) e executa as mesmas chamadas (`follow`).
4. **Configuração idêntica nos dois ranks, verificada no boot.** A lista `mine` de inteiros (drafter, capacidade,
   `TF_GLM_LATENT`, `TF_GLM_PREFILL_ROWS`, `TF_GLM_PROMPT_GRID`, `TF_GLM_KVB`, `TF_GLM_COPY_*`, `TF_GLM_HC_SPLIT*`,
   `TF_GLM_DRAFT_DUMP`, `TF_GLM_DRAFT_QUANT`, `TF_GLM_SHARED_PREFIX`, `TF_GLM_MTP`, `TF_GLM_DRAFT_RING`…) é reunida e
   comparada; diferença aborta com "the two ranks were started with different settings" `[código: engine.py]`.
   Cada patch que acrescenta um knob o acrescenta nessa lista (é um exemplo limpo de "fail-closed").
5. **Lossy só onde declarado.** "Three defaults are not exact against the checkpoint in bf16, for speed and the 1M
   window: `DENSE=q4` and `KV=fp8` are lossy […], and `KDA_CHUNKED=1` is close to the serial kernel but not its bits.
   `DENSE=bf16 KV=bf16 KDA_CHUNKED=0` serves the checkpoint as it is" `[README, Checks]`. Os checks de exatidão
   publicados: "reference shas, drafted == serial 6/6, concurrency 22/22, images and videos, tool calls"
   `[CHANGELOG v1.3]`.

Um sexto detalhe de engenharia: as extensões CUDA são compiladas por JIT com **nome versionado**
(`tensorfold_glm_exl3_v1` → `v9` (0001) → `v14` (0004) → `v15` (0009) → `v17` (0016) → `v18` (0020);
`tensorfold_experts_v7` → `v8`; `tensorfold_qmm_v3` → `v4`), para que o cache de extensões nunca carregue um `.so`
anterior; a receita ainda separa o cache por hash da imagem (`~/.cache/tensorfold-glm53/<image hash>`)
`[CHANGELOG v1.3]`.

### 0.3 Geometria por rank para ler os kernels

| Grandeza | Valor | Fonte |
|---|---|---|
| `D` (hidden) | 4096 | caminho `dec` exige `D == 4096` `[código: exl3_mm.py, 0016]` |
| `NI` (largura do expert por rank) | 1024 (= 2048 / 2) | idem (`NI == 1024`) |
| Configuração do kernel EXL3 | gate/up `(nt=8, warps=4, SK=4)`, down `(8, 4, 1)` | `GATEUP_CFG`, `DOWN_CFG` `[código: exl3_mm.py]` |
| Passos k por warp no decode | 16 (gate/up: K 4096 em 4 splits; down: K 1024 em 1) | `exl3_dec_cuda` `[código, 0016]` |
| Slots por linha | top-8 roteados + 1 compartilhado (o último slot) | "the last slot is the shared expert's" `[código: exl3_mm.py]` |
| Itens do plano de experts | 16 pares (decode), 64 ou 128 (prompt) | `experts.cpp` `[código, 0004]` |
| Cabeças MLA por rank | 32 | `ONEPASS_TILE` "(a rank's 32)" `[código: latent.py, 0009]` |
| Latente MLA | 512 colunas (`lw`) | kernels `latent.py` |
| KDA | `DK = DV = 128`, conv de 4 taps, estado `[H, 128, 128]` fp32 | `kda_chunked.py` `[código, 0012]` |
| Hyper-connections | 4 fluxos × 4096 = 16.384 colunas; 24 produtos de mistura + soma de quadrados por linha | `hc.cu` `[código, 0019]` |
| Linhas de um chunk de prompt | `PREFILL_ROWS = 2048` | `tensorfold/cuda/geometry.py` |
| Janela de verificação | até 16 linhas (após 0013); grafos CUDA de 1 a 8 linhas | `MAX_ROWS`, `GRAPH_ROWS` `[código: engine.py, 0013]` |
| Parcial fp32 de uma linha | 4096 × 4 B = 16 KiB | `[cálculo próprio]`; é o "16 KiB gather" do README |

---

## 1. Tabela-resumo dos 26 patches

"Bits" = efeito sobre a saída: **=** mesmos bits; **≠ (prompt)** muda os bits do prefill mas mantém rascunho == serial
e independência de chunking; **lossy** = aproximação declarada.

| # | Patch | Área | Mudança em uma linha | Controle (padrão no patch → receita) | Efeito publicado | Bits | Capítulos |
|---|---|---|---|---|---|---|---|
| 0001 | `glm-exl3-prompt-experts` | Prefill, experts EXL3 | blocos-coluna de um item lançados juntos (L2) + prefetch em registradores | `TF_GLM_EXL3_NFIRST` (1), `TF_GLM_EXL3_PF` (1) | 2,3–2,5× a 4.096 linhas `[código]` | = | `kernels`, `gpu-por-dentro`, `hadamard-trelica-exl3` |
| 0002 | `glm-dense-fp8` | Pesos densos | projeções BF16 → FP8 e4m3 por linha × bloco de 128 colunas, kernel Triton | `TF_GLM_DENSE` (bf16 → `q4`; `fp8` opcional) | ~+33% decode sobre bf16 `[config.sh]` | lossy | `fp8-fp4-microescala`, `quantizacao-fundamentos` |
| 0003 | `glm-vision` | Servidor / visão | torre visual do GLM no rank 0, imagens e vídeo | `--vision`, `--vision-urls`, `TENSORFOLD_GLM_*` | recurso (picos de workspace medidos) | = (texto) | `multimodal`, `api-tools-estruturado` |
| 0004 | `glm-prompt-kernels` | Prefill | kernels de prompt Yᵀ = Wᵀ Xᵀ para EXL3 + tensor cores em hc/latent/densos | `TF_GLM_EXL3_PROMPT`, `_PASS`, `_SEQ`, `TF_GLM_HC_MMA`, `TF_GLM_LATENT_MMA`, `TF_GLM_SEQ_ROWS`, `TF_GLM_PREFILL_ROWS` | SEQ 1,3–3,2× em N≥512, K≤4096 `[código]` | ≠ (prompt) | `kernels`, `numeros-no-computador`, `mla`, `hiperconexoes` |
| 0005 | `glm-dense-q4` | Pesos densos | projeções em 4 bits afins (grupos 64) com faixa buscada por MSE | `TF_GLM_DENSE=q4` (receita: padrão) | prosa 38,9→44,4; código 44,2→48,7; prefill ~1.090→~1.260 tok/s (sobre fp8) `[README]` | lossy | `quantizacao-fundamentos`, `medindo-qualidade` |
| 0006 | `cuda-roce-allgather` | Rede | all-gathers pequenos como RDMA WRITE one-shot (b12x "RoCEnante") | `TF_GLM_COMM` (nccl → `roce`), `TF_ROCE_MAX_KB` (256 → 512) | 11 µs vs 45 µs (16 KiB); decode +6% `[README]` | = | `nccl-rdma`, `cuda-graphs-warmup`, `memoria-unificada` |
| 0007 | `glm-copy-drafts` | Rascunho | prompt lookup: copia o que seguiu os últimos 8 tokens | `TF_GLM_COPY_DRAFTS` (0 → 1), `_MATCH` (8), `_MAX` (5 → 15) | quote/edit 80,3→84,4 tok/s `[README]` | = | `especulacao-fundamentos`, `dflash-dspark` |
| 0008 | `glm-prompt-grid` | Reuso de prompt | estados de prompt guardados só em múltiplos de G | `TF_GLM_PROMPT_GRID` (0 → 64 via 0039) | ver linha "Prompt reuse" `[README]` | = | `prefix-caching`, `atencao-linear` |
| 0009 | `glm-prefill-kernels` | Prefill | rotação de entrada 1×/linha; atenção densa só onde precisa; sparse em uma passada; seleção de tokens mais barata | `TF_GLM_EXL3_ROWROT`, `TF_GLM_SPARSE_ONEPASS`, `TF_GLM_SCORE_RB` | sem número isolado | = / ≠ (one-pass) | `atencao-esparsa`, `kernels`, `hadamard-trelica-exl3` |
| 0010 | `glm-hc-split` | Prefill, 2 Sparks | cola das hyper-connections dividida por linhas entre ranks, trocas sobrepostas | `TF_GLM_HC_SPLIT` (0 → 1), `TF_GLM_PREFILL_OVERLAP` (0 → 2 via 0033), `_PIECES`, `_MIN_ROWS`, `TF_GLM_HC_EXCHANGE` | 50k: ~1.270→~1.730 tok/s; decode −1,5% `[README]` | = | `paralelismos`, `nccl-rdma`, `hiperconexoes` |
| 0011 | `glm-draft-sim` | Ferramenta de rascunho | dump teacher-forced do DFlash2 + simulador offline de regras de parada | `TF_GLM_DRAFT_DUMP` (off), `TF_GLM_DRAFT_QUANT` (minmax) | "how the stop rules were tuned" `[README]` | = (dump = serial) | `dflash-dspark`, `especulacao-fundamentos`, `benchmarking` |
| 0012 | `glm-kda-chunked` | Prefill, KDA | regra delta em forma chunked (WY/UT), sub-chunks de 64 | `TF_GLM_KDA_CHUNKED` (0 → 1), `_PREC`, `_BV`, `_STATE_ROWS`, `_KSPLIT` | (com 0014+0039) 50k 29,3→26,4 s; 149k 91,1→84,5 s `[README]` | ≠ (próximo) | `atencao-linear`, `kernels` |
| 0013 | `glm-decode-rounds` | Decode | janelas até 16 linhas; menos syncs; tokens entregues uma rodada depois; kv_b FP8/Q4 | `TF_GLM_KVB` (bf16), `TF_GLM_COPY_MAX` até 15 | edit 84→119 tok/s com `COPY_MAX` 15 `[README]` | = (KVB bf16) | `cuda-graphs-warmup`, `especulacao-fundamentos`, `mla` |
| 0014 | `glm-kda-chunked-gb10` | Prefill, KDA | estabilidade numérica e paridade de arredondamento com `kda.cu`; 1 estágio por smem do GB10 | (constantes) | parte do número de 0012 | ≠ (próximo) | `numeros-no-computador`, `gpu-por-dentro`, `atencao-linear` |
| 0015 | `glm-shared-prefix` | Reuso de prompt | estados extras onde o prompt deixa de compartilhar e no fim do bloco de sistema | `TF_GLM_SHARED_PREFIX` (0 → 1 = 256 tokens) | 7,9k de system prompt: 4,24 s → 0,13 s `[README]` | = | `prefix-caching`, `dflash-dspark` |
| 0016 | `glm-decode-kernels` | Decode | tiles Q4 por forma, epílogo de down fundido, KDA de decode em 3 kernels | `TF_GLM_Q4_TILE`, `TF_GLM_EXL3_DEC`, `_FUSE`, `_XROW`, `TF_GLM_KDA_WIDE_ROWS` | KDA decode 4–36% mais rápido (1–8 linhas) `[código]` | = | `kernels`, `cuda-graphs-warmup`, `atencao-linear` |
| 0017 | `glm-overlap-priority` | Prefill, 2 Sparks | prioridade da stream lateral configurável (alta) | `TF_GLM_OVERLAP_PRIORITY` (-1) | "may cost decode rounds a little" `[código]` | = | `cuda-graphs-warmup`, `benchmarking` |
| 0018 | `glm-noise-policies` | Rascunho | regras de parada do DFlash2 cientes do ruído (`fnc`, `fcost`) | política `fncN:P[:B]`, `fcostN:noisy[:B]` | prosa 47,2→50,6; código 51,7→55,5 tok/s (vs `fc5:0.3`) `[README]` | = | `dflash-dspark`, `geracao-amostragem` |
| 0019 | `glm-decode-kernels2` | Decode | tiles Q4 de 32 colunas, combine sem cópia, `hc_partial` em CUDA com bits do Triton, pipeline de atenção | `TF_GLM_COMBINE_SY`, `TF_GLM_HC_DEC`, `TF_GLM_LATENT_STAGES` | sem número isolado | = | `kernels`, `hiperconexoes` |
| 0020 | `glm-prompt-experts-order` | Prefill, experts | intercala itens limitados por DRAM e por tensor core; decode de treliça com menos instruções | `TF_GLM_EXL3_ORDER` (1) | entra no 1.270→1.730 de 0010 `[config.sh]` | = | `kernels`, `mixture-of-experts` |
| 0021 | `glm-dflash-policy-env` | Rascunho | política padrão do DFlash2 por variável | `TF_GLM_DFLASH_POLICY` (`fc5:0.3` → `fnc7:0.3`) | ver 0018 | = | `dflash-dspark` |
| 0022 | `glm-overlap-normal-priority` | Prefill, 2 Sparks | prioridade normal vira o padrão | `TF_GLM_OVERLAP_PRIORITY` (-1 → 0) | decode após prompt de alta prioridade ~0,5–1% mais lento `[código]` | = | `benchmarking`, `cuda-graphs-warmup` |
| 0023 | `server-effort-max` | Servidor | `reasoning_effort: "max"` | campo da requisição | recurso | = | `geracao-amostragem`, `api-tools-estruturado` |
| 0024 | `glm-prompt-select-rows` | Prefill, DSA | seleção de tokens em blocos de 512 linhas | (constante `SELECT_ROWS`) | "less memory at 1M" `[README]` | = | `atencao-esparsa`, `memoria-unificada` |
| 0025 | `glm-dflash2-ring` | Rascunho / reuso | snapshot com drafter em anel não copia `kc/vc` por fatia | (`TF_GLM_DRAFT_RING`, do v0.6.0) | "memory for the window" `[README]` | = | `dflash-dspark`, `prefix-caching` |
| 0026 | `glm-multi-kda` | Concorrência | KDA de várias sequências numa janela via tabela de segmentos | (usado pelos patches de concorrência) | parte de "4 requests at once" `[README]` | = | `atencao-linear`, `scheduler-batching`, `cuda-graphs-warmup` |

---

## 2. Fichas por patch

Formato de cada ficha: **Área · Problema · Mecanismo · Controle · Efeito medido · Conceitos · Latent Atlas**.

---

### 0001 — `glm-exl3-prompt-experts`

**Área:** prefill; experts roteados EXL3 (`exl3.cu`, `exl3.cpp`, `exl3_mm.py`).

**Problema.** No TensorFold v0.6.0 o mesmo `grouped_kernel` EXL3 servia decode e chunks de prompt. A grade era
`(item, bloco-n, mat·SK + split)` com `blockIdx.x = item`. Um "item" é `(expert, primeiro par, contagem)` com até 16
pares (linhas de ativação roteadas para o mesmo expert). Cada um dos `N/128` blocos-coluna de um item relê as mesmas
linhas `X` do item; com o item na dimensão mais rápida da grade, blocos consecutivos pertencem a itens diferentes e
as releituras das linhas de um item ficam distantes no tempo — com milhares de linhas no chunk, elas saem do L2 entre
uma leitura e outra.

**Mecanismo.**
1. **`NFIRST` (ordem de lançamento):** o template ganha `bool NFIRST`; com ele `item = blockIdx.y`,
   `nblock = blockIdx.x` e a grade vira `(N/(16·nt), max_items, mats·SK)`. Como o despacho de CTAs segue,
   aproximadamente, a ordem linear do índice de bloco com `x` variando mais rápido `[inferência sobre o escalonador]`,
   todos os blocos-coluna de um item rodam lado a lado, e "they read its rows while L2 holds them"
   `[código: exl3.cu]`. Ativado só para janelas com `R >= PROMPT_ROWS = 64` (decode mantém a ordem antiga). Ordem de
   ilustração: as 16 linhas fp16 de um item com K = 4096 somam 16 × 4096 × 2 B = 128 KiB `[cálculo próprio]`.
2. **`PF` (prefetch em registradores):** um laço com *double buffering* em registradores: antes de decodificar e
   multiplicar o passo `kt`, carrega as palavras de treliça (`__ldg`) e os fragmentos A (`load_pair`) do passo
   `kt+1` em `nw[]`/`na[]`; depois do `decode_tile` (código MCG → dois fragmentos B half) e dos dois
   `mma16816` (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`), troca os buffers. A ordem dos `mma` é a
   mesma, então os bits também.
3. O template passa a `<NT, W, NFIRST, PF>`; a macro `LAUNCH` escolhe entre 4 instâncias.

> **Depois de 0004:** com `TF_GLM_EXL3_PROMPT=1` (padrão) os chunks de prompt usam os kernels Yᵀ = WᵀXᵀ, com plano
> de 64 pares; o `grouped_kernel` com `NFIRST`/`SEQ` continua valendo para planos de 16 pares com `R ≥ 64` — o
> caminho de prompt com `TF_GLM_EXL3_PROMPT=0` e, `[inferência]`, janelas largas de vários streams; janelas de decode
> abaixo de 64 linhas vão para `dec_kernel` (0016). O `PF` (prefetch) é reaproveitado no `dec_kernel`.

**Controle:** `TF_GLM_EXL3_NFIRST` (padrão 1), `TF_GLM_EXL3_PF` (padrão 1); limiar fixo `PROMPT_ROWS = 64`. A receita
não mexe nelas.

**Efeito medido:** "Prompt chunks launch an item's n blocks next to each other […]: the same kernel and arithmetic,
2.3-2.5x faster at 4,096 rows" `[código: exl3_mm.py]`. No README entra na linha "Prompt" ("faster prefill, less
memory at 1M"), sem número isolado.

**Conceitos:** reuso temporal no L2; ordem de despacho/"rasterização" de blocos (o mesmo truque do *swizzle* de CTAs
em GEMMs CUTLASS); grouped GEMM de MoE (plano `items/members/counts`); split-K; prefetch em registradores
(*software pipelining*); `mma.sync` m16n8k16.

**Latent Atlas:**
- `kernels` — acrescentar um quadro "a ordem de lançamento é uma alavanca de cache": mesmo kernel, mesma aritmética,
  só trocar `blockIdx.x ↔ blockIdx.y` deu 2,3–2,5×; e o padrão de prefetch em registradores com o código do laço.
- `gpu-por-dentro` — na seção de L2: como blocos vizinhos no índice compartilham dados no L2; conta das 128 KiB por
  item.
- `hadamard-trelica-exl3` — na parte "decodificação no kernel": `decode_tile` alimentando `mma16816`, a rotação de
  entrada (`rot_in`) e os epílogos (ver 0004/0016).

---

### 0002 — `glm-dense-fp8`

**Área:** pesos densos do checkpoint EXL3 (`qmm.py`, `weights.py`, `engine.py`; perfis em `forward.py`).

**Problema.** No checkpoint `GLM-5.3-Flash-EXL3-TR3-4bpw` só os experts roteados são EXL3; atenção, experts
compartilhados, MLPs densas e cabeça ficam em BF16 — "~9.7 GiB a Spark" `[config.sh]`. Todo passo de decode lê esses
bytes; o decode é limitado por banda.

**Mecanismo.**
- `TF_GLM_DENSE=fp8` quantiza na carga (`make_f8`): para cada linha e bloco de 128 colunas, escala fp32 =
  `absmax / 448` (`F8_MAX`, o máximo do e4m3), divide, `clamp`, converte para `torch.float8_e4m3fn`
  (arredondamento ao mais próximo). Matrizes com K não múltiplo de 128 ficam BF16. `stack_f8` quantiza matrizes
  empilhadas (ex.: `q_a_proj` + `kv_a_proj_with_mqa`).
- Kernel Triton `_fmm`: mesma estrutura do `_bmm` BF16 (linhas independentes, split-K fixo pela forma via
  `b16_split_k`, `_reduce` somando os splits em ordem); carrega o peso fp8, converte para bf16 (e4m3 é exato em
  bf16), faz `tl.dot` com a ativação bf16 e multiplica o produto de cada passo `BK = 64` pela escala do bloco de 128
  da coluna (`* s[None, :]`), acumulando em fp32.
- Decode (`bm = 16`): `F8_BN_DECODE = 32` colunas por tile e 6 estágios de pipeline — "narrower column tiles, deeper
  pipeline (measured +8%)" `[código: qmm.py]`.
- A cabeça: verificação com a cabeça densa (FP8), passos de rascunho com uma cópia de 4 bits (`quantize4`) separada.
- O estimador de memória do boot (`fp8_weights`) passa a contar 1 B/valor + escalas para os nomes em `_F8_NAMES`.
- Bônus de observabilidade: `prof.timed("kda: projections" / "kda: recurrence" / "moe: routed (exl3)" / …)`.

**Controle:** `TF_GLM_DENSE` (padrão do patch `bf16`). Na receita `DENSE` vale `q4` por padrão (0005); `fp8` é a
alternativa recomendada para prompts não ingleses (§6).

**Efeito medido:** "fp8: FP8 e4m3 with a scale per row and 128 columns (patch 0002), half bf16's bytes: ~+33% decode
over bf16" `[config.sh]`; o fp8 é a base da comparação de 0005. Para prompts franceses, "`fp8` keeps it, at ~10%
decode speed" `[README]` — lido como "custa ~10% de decode em relação a `q4`" `[inferência]`, coerente com
38,9 vs 44,4 tok/s de prosa (~12%) da comparação q4/fp8.

**Conceitos:** W8A16 weight-only; escala por bloco 1×128 (linha × bloco de K) vs por canal; e4m3 (máx 448);
dequantização dentro do produto; split-K determinístico; decode limitado por banda.

**Latent Atlas:**
- `fp8-fp4-microescala` — exemplo real de FP8 *weight-only* com escala por linha e bloco de 128 colunas, ao lado do
  128×128 do DeepSeek e do Marlin por canal da receita vLLM (`GLM53_DENSE_FP8`).
- `quantizacao-fundamentos` — granularidade (por tensor/canal/bloco) com este caso.
- `kernels` — dequantização em Triton aplicando a escala por passo de K dentro de um bloco de 128.
- `receita-glm` — tabela "o que a receita TensorFold faz no lugar de `GLM53_DENSE_FP8`".

---

### 0003 — `glm-vision`

**Área:** servidor + visão (novos `tensorfold/vision/glm.py` e `videos.py`; `engine.py`, `forward.py`, `decode.py`,
`app.py`, `server.py`, `messages.py`, `prompts.py`, `images.py`, `images_http.py`, `serve_options.py`).

**Problema.** No v0.6.0 a entrada de imagem do GLM-5.3-Flash era só MLX ("GLM-5.3-Flash image input is currently
MLX-only", removido); o template do checkpoint responde a imagens com um lembrete "You are unable to process this…".

**Mecanismo.**
- **Template:** com `--vision`, `media_template` troca o ramo `_NO_MEDIA` do `chat_template.jinja` pelos marcadores
  do GLM (`<|begin_of_image|><|image|><|end_of_image|>`, idem vídeo); falha fechada se o template não tiver
  exatamente um ramo conhecido.
- **Processadores** (seguem `Glm5NextImageProcessor`/`Glm5NextVideoProcessor` do transformers 5.17 `[código]`):
  `smart_resize` escolhe uma tela múltipla de 28 px que caiba em `max_tokens × 2 × 28²` pixels (busca binária da
  maior altura); `content_size` + `fit` redimensionam (bicúbico com antialias) e preenchem com zeros à direita/abaixo;
  normalização com média/desvio do CLIP. Vídeo: `sample_frames` pega 1 quadro a cada 1/fps s (fps = 2), no máximo
  `video_frames`, depois espalha por `linspace`, remove duplicatas e força contagem par (repete o último);
  `decode_clip` decodifica com PyAV com prazo, até 3.600 s e 8.192 px, redimensionando cada quadro ao decodificar.
- **Layout do prompt:** cada `<|image|>` vira `n = (H/28)·(W/28)` tokens; um clipe vira, por par de quadros,
  `<|begin_of_image|>…<|end_of_image|>{s:.1f} seconds` (o tempo do par como texto). As posições das linhas de
  imagem são os índices de `image_token_id` (154854) nos ids.
- **Torre (`GlmVisionTower`, torch puro, bf16):** `conv3d` de patches 3×2×14×14 ("the reference's convolutions, not
  the equal matmuls: 24 blocks turn their last-bit differences into ~3%" `[código]`); 24 blocos com RMSNorm,
  `qkv` com bias, RMSNorm em q e k, RoPE 2-D (frequências de altura e largura concatenadas), SDPA dentro de uma
  imagem/par de quadros, MLP SwiGLU com `clamp(limit)`; `post_layernorm`; *downsample* `conv2d` 2×2 stride 2;
  *merger* (proj → LayerNorm → GELU → SwiGLU com clamp) até 4096 colunas. Codifica em passadas de `PASS_PATCHES`
  patches (16.384; 8.192 a partir de 0004).
- **Injeção:** `compute(..., inject=(rows, features))` faz `b.x.index_copy_(0, rows, features.repeat(1, streams))`
  — substitui o embedding dessas linhas nos 4 fluxos de hyper-connection. Como as camadas MLA do GLM são NoPE, "its
  positions are the token positions (the MLA layers carry no rotary part), so nothing else about the prompt changes"
  `[código: glm.py]` — não precisa de mRoPE.
- **Dois ranks:** só o rank 0 carrega a torre. `VisionFeed.chunk` entrega as linhas de cada chunk de prefill ao rank 1
  pelo **all-gather** à medida que o chunk chega (o rank 1 manda zeros), "so neither rank holds a second copy"
  `[código: engine.py]`. A codificação acontece antes de o rank 1 ser avisado, para que uma falha pare os dois.
- Prompts com imagem **não** são guardados para reuso (`self.live = []`, sem `keep`).
- Limites: tokens por imagem e por clipe com faixas validadas; `VideoLimits` (2 vídeos, 64 MB cada, 96 MB no total,
  120 s de decodificação…) — elevados depois por 0050 (parte B).

**Controle:** `--vision` / `--vision-urls` (receita `VISION=1`, `VISION_URLS=0`); `TENSORFOLD_GLM_IMAGE_TOKENS`
(2048; 16–8000), `TENSORFOLD_GLM_VIDEO_TOKENS` (16384; 64–240000), `TENSORFOLD_GLM_VIDEO_FRAMES` (128; 2–2048),
`TENSORFOLD_VISION_WORKSPACE_MIB` (0–16384).

**Efeito medido:** recurso de API. Picos de workspace medidos: "0.47 GiB (a 2,048-token picture), 0.56 (four), 1.08
(a 16,384-token clip, 16,384 patches a pass), 1.73 (an 8,000-token picture, one 31,724-patch pass)" `[código]`. Na
receita final: torre de 1,05 GiB bf16 + 0,75 GiB de workspace no rank 0; com `KV=bf16` "the tower costs rank 0 ~66k
tokens of window" `[README]`. O checkpoint permite 8.000 tokens por imagem e 240.000 por clipe; os limites menores
"keep a request's encode within the tower's reserved workspace" `[código]`.

**Conceitos:** ViT com patch 14, patch temporal 2, merge espacial 2×2; RoPE 2-D; SwiGLU com clamp; tokens por imagem
e orçamento; vídeo em pares de quadros com tempo textual; LM NoPE dispensa mRoPE; encoder num rank só com distribuição
de *features* por coletiva; fidelidade numérica (conv vs matmul "equivalente").

**Latent Atlas:**
- `multimodal` — seção "GLM-5.3-Flash": torre de 24 blocos, layout do processador (28 px por token), vídeo em pares
  com "X.X seconds", por que NoPE dispensa mRoPE; contraste com `--mm-encoder-tp-mode data` do vLLM vs "torre no
  rank 0 + all-gather das linhas por chunk".
- `api-tools-estruturado` — partes `video_url`, data URLs vs `https`, limites.
- `memoria-unificada` — workspace de visão no orçamento do rank 0.
- `numeros-no-computador` — conv vs matmul: diferenças no último bit viram ~3% após 24 blocos.

---

### 0004 — `glm-prompt-kernels`

**Área:** prefill (EXL3, hyper-connections, MLA latente, matmuls BF16/FP8, geometria de memória, visão).

**Problema.** O kernel EXL3 de decode decodifica cada tile de treliça para alimentar só 2 `mma` (16 linhas); em
prompt, cada expert recebe dezenas de tokens, e decodificar a treliça de novo para cada grupo de 16 desperdiça
trabalho. Os split-K fp32 viram tráfego de memória quando M é grande. `hc_pre`, absorção/expansão do MLA e matmuls
BF16/FP8 rodavam caminhos pensados para poucas linhas.

**Mecanismo.**
1. **Kernels de prompt EXL3 `Yᵀ = Wᵀ Xᵀ`** (`prompt_gateup_kernel`, `prompt_down_kernel`): "A decoded trellis tile
   is, reordered, an m16n8k16 A fragment of Wᵀ […], so each warp decodes its own tiles straight into registers and
   every decoded fragment feeds the whole pass's member rows (up to 64 tokens) as B fragments from shared memory"
   `[código]`. Ou seja, inverte os papéis: o peso vira o operando A (dimensão M do `mma`), os tokens viram B. Um
   tile decodificado alimenta `PM/8` `mma` (8 com `PM = 64`) em vez de 2.
   - Bloco = 16 tiles m (gate/up: 8 de cada matriz = um bloco Hadamard de 128 colunas; down: 256 colunas).
     `PM = 64` com 8 warps (2 blocos/SM) ou `PM = 128` com 16 warps (1 bloco/SM).
   - Pipeline `cp.async.cg` de `PSTAGES = 3` estágios trazendo para shared memory os próximos `PBK = 32` valores de k
     das linhas e as palavras de treliça dos 16 tiles; linhas de stage com `PXS = PBK + 8` halves ("+16 B: ldmatrix
     rows on distinct banks"); fragmentos B por `ldmatrix.sync.aligned.m8n8.x4`.
   - **Epílogos fundidos:** os mesmos bytes de shared memory viram o buffer fp32 do epílogo (metade dos membros por
     vez); gate/up aplica `fwht128` (FWHT de 128 em registradores, 32 lanes × 4 valores), `HAD_SCALE`, `svh`,
     clamp, SiLU com arredondamentos bf16 (`bf16r`) idênticos à referência, multiplica `suh_d` e roda outra FWHT
     para produzir a entrada fp16 do down; down aplica FWHT + `svh_d` e escreve `y` fp32.
   - "Each output is one fp32 chain of mma over ascending k tiles (no split K, no partials in memory), so a row's
     bits depend on its own values only (any chunking gives the same prompt state), though they are not the decode
     kernel's" `[código]`.
   - Plano de prompt com itens de 64 (ou 128) pares: `experts.plan` aceita `tile == 128`; `Plan(tile=…)`.
2. **`SEQ` no kernel agrupado** (quando os kernels de prompt estão desligados): um programa percorre os `SK` splits em
   sequência e mantém os totais em registradores (`tot[j] = ((0 + s0) + s1) + …`), a mesma soma do epílogo — "the
   same bits without the SK fp32 partials' round trip through memory".
3. **Tensor cores em outras partes do prompt:** `_hc_partial_mma` (os 24 produtos de mistura do `hc_pre` + soma de
   quadrados via `tl.dot`, BM = BK = 64); `_absorb_mma`/`_expand_mma` do MLA latente (`MMA_TILE = (64, 128, 64)`,
   4 warps, 3 estágios); BF16/FP8 `SEQ` a partir de `SEQ_ROWS = 256` linhas (com `n ≥ 512`, `k ≤ 4096`) e tabela
   `F8_PROMPT` de `(BM, BN, warps, stages)` por forma `(N, K)` "swept on GB10 at 2,048 rows; no choice changes a
   row's bits".
4. **`TF_GLM_PREFILL_ROWS`:** linhas de um chunk de prompt (múltiplo de 64, de 64 a 16.384; padrão 2048): "more
   amortizes each chunk's expert weight decode over more tokens and costs buffer memory"; toda a geometria de memória
   (`mla_geometry`, `mla_chunk_scratch`) passa a receber `prefill_rows`.
5. **Visão:** `VISION_WORKSPACE` cai de `3·2²⁹` (1,5 GiB) para `3·2²⁸` (0,75 GiB) porque o scratch da torre "takes
   turns with a prompt chunk's transient scratch (~1 GiB at 164k)", com `torch.cuda.empty_cache()` antes e depois da
   codificação; `PASS_PATCHES` 16.384 → 8.192 ("a 2,048-token picture is one pass").

**Controle:** `TF_GLM_EXL3_PROMPT` (1), `TF_GLM_EXL3_PASS` (64; 128 "measured slower on real routing: most passes are
small, and its one 16-warp block an SM keeps fewer experts in flight"), `TF_GLM_EXL3_SEQ` (1), `TF_GLM_HC_MMA` (1),
`TF_GLM_LATENT_MMA` (1), `TF_GLM_SEQ_ROWS` (256), `TF_GLM_PREFILL_ROWS` (2048).

**Efeito medido:** `SEQ` nas matmuls BF16: "Measured at 2,048 rows (GB10): 1.3-3.2x on N >= 512, K <= 4,096; slower
on narrow outputs (N 32-288) and long K (6,144, 8,192)" `[código: qmm.py]`. README: "faster prefill" (linha
"Prompt"), sem número isolado.

**Bits:** prompt ≠ decode, mas independentes do chunking; rascunho == serial preservado (ambos passam pelo mesmo
prefill).

**Conceitos:** troca de operandos (calcular a transposta para pôr o lado grande em M); amortização da decodificação
de pesos quantizados; pipeline `cp.async` multiestágio; `ldmatrix` e *padding* contra conflito de banco; epílogo
fundido; determinismo vs split-K (não associatividade); tabelas de tile por forma.

**Latent Atlas:**
- `kernels` — estudo de caso "um GEMM MoE quantizado de prefill por dentro" com este kernel (pipeline, `ldmatrix`,
  `mma.sync`, *padding*, epílogo com FWHT, por que trocar A e B); `SEQ` como alternativa ao split-K com partials.
- `numeros-no-computador` — "uma cadeia fp32 por linha" ⇒ invariância a chunking; split-K muda bits.
- `hadamard-trelica-exl3` — diferença prompt vs decode no kernel EXL3.
- `hiperconexoes` — `hc_pre` em tensor cores no prefill.
- `mla` — absorção/expansão como GEMM em tensor cores no prefill.
- `scheduler-batching` — `TF_GLM_PREFILL_ROWS` como análogo de `max_num_batched_tokens`/tamanho de chunk.

---

### 0005 — `glm-dense-q4`

**Área:** pesos densos (`qmm.py`, `weights.py`, `engine.py`).

**Problema.** FP8 corta metade dos bytes densos; 4 bits cortam mais, mas sem calibração o erro de arredondamento
precisa ser controlado.

**Mecanismo.**
- `quantize4_mse`: por grupo de 64 valores, testa 16 recortes da faixa `a = 1 − 0,45·i/15` (de 1,0 a 0,55 do
  min/max), calcula escala `s = (hi − lo)/15` e bias `b = lo` em bf16, `q = round((g − b)/s)` com clamp 0–15, e
  fica com o recorte de menor erro quadrático. Empacota 8 nibbles por int32 (do menor para o maior) no layout MLX e
  vira o tipo `Q4` (`make_q4`, tiled) que os kernels Q4 do TensorFold já usam.
- Matrizes com K não múltiplo de 64 ficam FP8; a cabeça fica FP8. Para `kv_b` ver 0013 e §6: no caminho latente
  final ele fica **BF16**.
- Estimativa de memória (`q4_weights`): `n/2 + n/64·4` bytes por matriz → 4 bits + 32 bits por 64 valores =
  **4,5 bits/peso efetivos** `[cálculo próprio]`.
- Detalhe útil para o ensino: o Q4 é afim, então o produto se escreve `x·(s·q + b) = s·(x·q) + b·Σx` por grupo — por
  isso os kernels recebem `xs` (somas por grupo de 64 da ativação, `qmm.group_sums`), visíveis em vários patches
  (`b.xs`, `b.xs_fa`, …).

**Controle:** `TF_GLM_DENSE=q4` (padrão da receita: `DENSE=q4`).

**Efeito medido:** "q4 over fp8: prose 38.9 -> 44.4 tok/s, code 44.2 -> 48.7, prefill ~1,090 -> ~1,260 tok/s"
`[README]`; "GSM8K 98.0% and HumanEval 95.1% on both (bf16: 97.2 / 96.3). Lossy: replies differ" `[config.sh]`.
Regressão conhecida (#18): em prompts curtos de código em francês, `q4` "often loses the end of turn and runs to
`max_tokens` (12 of 48 replies, against 1 of 48 with `bf16` or `fp8`)"; "a calibrated 4-bit for the dense weights, at
`q4`'s speed, is being worked on" `[CHANGELOG v1.3.2]`.

**Conceitos:** quantização afim (assimétrica), grupos de 64, busca de recorte por MSE (sem ativações, ao contrário de
GPTQ/AWQ), bits efetivos com escalas, falhas fora da distribuição de avaliação (fim de turno em outra língua).

**Latent Atlas:**
- `quantizacao-fundamentos` — exemplo de busca de recorte por MSE em grade de 16; 4,5 bpw; truque `b·Σx`.
- `medindo-qualidade` — issue #18 como estudo de caso: GSM8K/HumanEval em inglês não pegam uma regressão de EOS em
  francês; medir "respostas que correm até `max_tokens`".
- `kernels` — por que kernels Q4 recebem somas por grupo.

---

### 0006 — `cuda-roce-allgather`

**Área:** rede entre os Sparks (`roce.cpp`, `roce.cu`, `roce.py`, `roce_proxy.c` novos; `engine.py`).

**Problema.** "A decode round's all-gathers are small (a verify window's partials: rows x 4,096 fp32) and NCCL spends
most of each in fixed latency (~30-50 us)" `[código: roce.py]`. No modelo α-β, mensagens de dezenas de KiB são
dominadas por α.

**Mecanismo** — o transporte "RoCEnante" do b12x (local-inference-lab, Apache-2.0): o proxy C é o deles sem mudança;
o lado GPU foi reescrito em CUDA C++ a partir do kernel CuTe DSL "with the same stages and memory orderings".
- **Região pinned** no host (registrada como MR em cada HCA): `recv[src][slot]`, `flag[src][slot][hca]` (passo de
  128 B), `send[slot]`, `ctrl` {seq, nbytes, erro, peer faltante, nbytes do slot 0 e 1, HCA faltante}. Dois slots
  (`seq & 1`) = *ping-pong*.
- **Kernel (uma chamada):**
  1. cada bloco copia sua parte do shard local para `send[seq & 1]`;
  2. o **último bloco** a terminar (contador de chegada com `atom.relaxed.gpu.add`) publica `nbytes`, faz
     `fence.sc.sys` e escreve `ctrl.seq` (*doorbell*);
  3. uma thread por (peer, HCA) gira com `ld.acquire.sys` na flag da faixa até ver `seq`; se exceder `spin_limit`
     (`SPIN_LIMIT = 20.000.000` polls, ~20 s) registra o erro e **envenena** (`poison`) o runtime;
  4. todos os blocos copiam os shards em ordem de rank (o próprio da entrada, os dos peers do slot escrito pela NIC
     com `ld.relaxed.sys.v4`);
  5. o último bloco avança a **época** no device (`st.release.gpu`) — "so CUDA graphs replay the exchange"
     (a sequência avança sem o host).
- **Proxy (thread C, libibverbs):** gira em `ctrl.seq` (*acquire*); para cada peer e HCA, posta um WR de
  `IBV_WR_RDMA_WRITE` com os dados encadeado a um WR de 4 bytes *inline* com a flag, no mesmo QP RC: "Each stripe is
  followed by its own 4-byte seq write on the same reliable QP, so its flag cannot become visible before its
  payload" `[código: roce_proxy.c]`. Divide a carga em faixas entre até 2 HCAs (as duas portas CX7). Fica quente
  `ROCE_IDLE_SPINS` polls e depois dorme 20 µs entre polls; recupera *doorbells* perdidos (até `ROCE_SLOTS = 2`
  pendentes).
- **Integração:** `RoceComm` embrulha o NCCL: gathers contíguos, alinhados a 16 B e de até `max_bytes` vão por RoCE;
  o resto por NCCL — "the same call, output layout and bits". Grade: potência de 2 de até 8 blocos × 512 threads.
  O handshake (blobs de QP, rkeys, GIDs) passa pelo próprio NCCL. `comm.check()` depois de cada requisição expõe um
  timeout ocorrido dentro de um grafo.

**Controle:** `TF_GLM_COMM` (padrão `nccl`; receita `COMM=roce`), `TF_ROCE_MAX_KB` (256; receita 512),
`TF_ROCE_HCA` ou `NCCL_IB_HCA` (senão os dispositivos ativos de `/sys/class/infiniband`, até 2),
`TF_ROCE_GID_INDEX` ou `NCCL_IB_GID_INDEX` (3), `NCCL_IB_TC` (0), `CC`, `TORCH_EXTENSIONS_DIR`. O patch 0052 (parte B)
acrescenta `TF_ROCE_WAIT_S` e uma barreira NCCL antes de cada gather até o motor ficar pronto.

**Efeito medido:** "11 us a 16 KiB gather against NCCL's 45, decode +6%" `[README]`; "decode +6% (prose 44.2 -> 46.9,
code 48.7 -> 51.6)" `[config.sh]`; `TF_ROCE_MAX_KB=512` "also takes the 17-32-row verify windows of concurrent
requests: code at 4 streams +1.4%, prose +0.5-1% (two boots each)" `[config.sh]`. Com as duas portas cabeadas, "a
prompt chunk's all-gather is ~1.8x faster on two rails" `[README, Requirements]`. 16 KiB = uma linha de parciais;
512 KiB = 32 linhas `[cálculo próprio]`.

**Conceitos:** modelo α-β; RDMA WRITE unilateral; QP RC, WRs encadeados, dados *inline*, CQ; RoCE v2, GID index;
*multi-rail*; memória pinned; modelo de memória PTX (escopos `.gpu`/`.sys`, acquire/release, `fence.sc`); padrão
*doorbell* + thread proxy (como o proxy do NCCL); época no device para replay de grafo; contador "último bloco".

**Latent Atlas:**
- `nccl-rdma` — nova seção "all-gather one-shot por RDMA WRITE": diagrama de 5 etapas, ordenação no QP RC, flags por
  faixa/HCA, timeout e *poison*, por que >512 KiB ainda vai por NCCL, números 11 µs vs 45 µs.
- `cuda-graphs-warmup` — como uma coletiva customizada sobrevive ao replay: época no device, contadores de chegada
  por classe de grade (um par por tamanho de grade potência de 2).
- `memoria-unificada` — a região pinned registrada como MR; no GB10 "host" e "device" são a mesma LPDDR5X
  `[inferência]`.
- `gpu-por-dentro` — escopos de memória e fences para comunicação GPU↔CPU↔NIC.

---

### 0007 — `glm-copy-drafts`

**Área:** rascunho (`copy_drafts.py` novo; `decode.py`, `drafter_choice.py`, `engine.py`).

**Problema.** Respostas que citam, editam o prompt ou se repetem têm continuações previsíveis sem modelo algum.

**Mecanismo.**
- `CopyDrafts` guarda o contexto (prompt + tokens comprometidos + o pendente) num buffer numpy `int32`. `starts()`
  acha, vetorizado, as ocorrências anteriores dos últimos `match` tokens (filtra pelos candidatos cujo último token
  bate e depois por cada posição). `propose(room)` devolve até `min(most, room)` tokens que seguiram a ocorrência
  **mais recente** com tokens suficientes depois dela; senão os que seguiram a mais antiga; `[]` se o sufixo nunca
  ocorreu.
- Nos três laços (`mtp_decode`, `dflash_decode`, `auto_decode`): se há proposta de cópia, ela substitui a cadeia do
  drafter naquela rodada (braço `"c"`); o MTP ainda absorve as linhas mantidas; a política de profundidade do
  DFlash2 **não** é atualizada por rodadas copiadas. Estatísticas `copy_rounds/drafted/accepted`.
- "Both ranks hold the same prompt and sample the same tokens, so each computes the same proposals with no exchange"
  `[código]`. Configuração entra na comparação de boot.

**Controle:** `TF_GLM_COPY_DRAFTS` (0 no patch; receita `COPY=1`), `TF_GLM_COPY_MATCH` (8; 2–64), `TF_GLM_COPY_MAX`
(min(5, drafts máx.); receita `COPY_MAX=15`, possível depois de 0013).

**Efeito medido:** "quote / edit replies +5% (80.3 -> 84.4 tok/s), prose and code unchanged. Exact." `[config.sh]`.

**Conceitos:** *prompt lookup decoding* / rascunho n-gram; casamento de sufixo; rascunhador de custo ~zero
complementar a um drafter neural.

**Latent Atlas:**
- `especulacao-fundamentos` — método n-gram com o algoritmo concreto (sufixo de 8 tokens, ocorrência mais recente) e
  quando ajuda.
- `dflash-dspark` — hierarquia "cópia primeiro, DFlash2 depois" e por que a política não aprende com rodadas
  copiadas.

---

### 0008 — `glm-prompt-grid`

**Área:** reuso de prompt (`decode.py`, `engine.py`, `forward.py`).

**Problema.** O v0.6.0 guardava o estado do prompt em `keep_at = len(prompt) − 1` cortando **dentro** de um chunk de
prefill (`Cut`: a cadeia KDA rodava em duas partes e clonava estado recorrente e janelas de convolução no ponto). Com
kernels cujos bits dependem do alinhamento das linhas em blocos (o KDA chunked de 0012/0039), retomar de um ponto
arbitrário daria bits diferentes de um prefill fresco.

**Mecanismo.**
- `TF_GLM_PROMPT_GRID = G ∈ {16, 32, 64, 128}`: chunks de prompt começam em múltiplos de G; o estado guardado fica
  em `B = ⌊n/G⌋·G` (`grid_point`) quando `B > cut`.
- A requisição pré-preenche até B "as a prompt of that length would" — `prefill(..., sample=False)`, sem cabeça
  (`compute(head=False)`) e sem primeiro token —, tira o snapshot (`take_snapshot`) e retoma dele até n. Assim
  "any prompt extending prompt[:point] passes through the same state there, fresh or resumed" `[código]`.
- Remove toda a maquinaria `Cut` de `forward.py`/`kda_block`; a busca de snapshot exige `len(ids) % G == 0`;
  um prefill retomado precisa começar na grade.
- Com 0039 (parte B), `TF_GLM_KDA_CHUNKED=1` faz a grade valer 64 quando não definida.

**Controle:** `TF_GLM_PROMPT_GRID` (0 = desligado no patch); na receita, implicitamente 64 ("prompt states then sit
on a 64-token grid" `[config.sh]`).

**Efeito medido:** sem número isolado; a linha "Prompt reuse" (0008, 0015, 0042) remete a "Performance": prompt
idêntico de 64k 34 s → < 0,07 s (0042), nova conversa com o mesmo system prompt de 7,9k 4,24 s → 0,13 s (0015)
`[README]`.

**Conceitos:** checkpoints de estado recorrente só em posições alinhadas (análogo ao APC alinhado a 3584 da receita
vLLM e ao `mamba_cache_mode align`); requisito "fresco == retomado" para determinismo.

**Latent Atlas:**
- `prefix-caching` — seção TensorFold: snapshots em grade de G tokens; comparação com blocos de 3584 do APC híbrido.
- `atencao-linear` — por que o estado recorrente só pode ser retomado onde houve snapshot e por que a forma chunked
  exige alinhamento.

---

### 0009 — `glm-prefill-kernels`

**Área:** prefill (EXL3, atenção latente densa e esparsa, seleção de tokens, combinação do MoE).

**Problema.** Quatro desperdícios no prefill: (a) a rotação de entrada EXL3 era feita por par (linha × slot) e por
matriz, embora no checkpoint GLM todos os experts compartilhem o mesmo `suh` de gate e up; (b) a passada densa de
atenção rodava em todas as linhas, mas linhas além do limite denso são recalculadas pela esparsa; (c) a atenção
esparsa usava chunks de 512 chaves + parciais fp32 + merge; (d) a seleção de tokens pontuava um número de pools
arredondado para potência de 2.

**Mecanismo.**
1. **`rot_rows` (rotação uma vez por linha):** `shared_suh` detecta, bit a bit (`view(int16)`; "-0 is not +0"), que
   todo expert tem `suh_g == suh_u ==` o mesmo vetor; então `x·suh @ H` depende só da linha. `rot_in_kernel<ROWS>`
   escreve `Xh[row]` uma vez; `prompt_gateup_kernel<…, SHX>` carrega as linhas dos membros **uma vez** para gate e
   up (`XM = 1` cópia no stage). Com top-8 e 2 matrizes, 16 rotações por linha viram 1 `[cálculo próprio]`.
   Mesmos bits.
2. **`dense_rows`:** linhas com posição antes de `SPARSE_FROM = (512 + 1)·4 − 1 = 2051` mantêm atenção densa; a
   passada densa roda só nelas (`qa[:dense]`, com o `head_block` da janela inteira para manter os bits).
3. **`sparse_onepass`:** um programa por (linha, grupo de 32 cabeças, fatia do latente) percorre todos os tokens
   selecionados num único *online softmax* e escreve bf16 direto — sem parciais nem merge. Prefetch manual: "ids run
   two tiles ahead and rows one: tile t + 1's gather is issued […] before tile t's dots, so neither load's latency
   stalls the in-order warp", com o pipeliner do Triton desligado. Tile fixo `ONEPASS_TILE = (32, 32, 1)` → bits de
   uma linha não dependem do chunk; são outros bits que os de `sparse_attention`.
4. **`_select_prompt`** (janelas ≥ 64 linhas): pontua `NP = pool_count` (pools visíveis arredondados a múltiplo de
   1024) em vez da potência de 2; argumenta por que o top-512 não muda (colunas descartadas são `-inf`, de índice
   maior, empates vão ao pool menor); `_scores` processa `RB` linhas por programa carregando cada tile de pools uma
   vez; o kernel `_tokens` escreve a lista ascendente `pool·4 + j` + a cauda do pool incompleto (largura
   `512·4 + 3 = 2051`) e a contagem (0 quando há ≤ 512 pools completos).
5. **`combine_shared`:** lê a linha do expert compartilhado direto de `sy` em vez de copiá-la para `ey` (mesma soma,
   mesma ordem, "one pass less").

**Controle:** `TF_GLM_EXL3_ROWROT` (1), `TF_GLM_SPARSE_ONEPASS` (1), `TF_GLM_SCORE_RB` (4; 1, 2, 4 ou 8).

**Efeito medido:** sem número isolado. O ganho do split (0010) foi medido "with 0009 and 0020" `[README]`.

**Conceitos:** rotação de incoerência com vetor de sinais compartilhado; eliminação de trabalho redundante; online
softmax em uma passada vs split-KV + merge (flash-decoding); atenção esparsa limitada por *gather* e ocultação de
latência por prefetch; seleção top-k de pools e regras de empate.

**Latent Atlas:**
- `atencao-esparsa` — prefill do DSA no TensorFold: limiar 2051, pontuação de pools, lista de 2051 tokens, one-pass
  vs chunks + merge.
- `kernels` — online softmax de uma passada vs flash-decoding; prefetch manual de gathers em Triton.
- `hadamard-trelica-exl3` — `suh` compartilhado por todos os experts do checkpoint GLM.

---

### 0010 — `glm-hc-split`

**Área:** prefill com dois Sparks (`hcsplit.py` novo; `comm.py`, `forward.py`, `glue.py`, `decode.py`, `engine.py`).

**Problema.** Sem split, em cada "local de troca" (depois do bloco de atenção e depois do MLP/MoE), os dois ranks
fazem all-gather dos parciais fp32 `[R, D]` e **ambos** executam `hc_post` e o próximo `hc_pre` sobre todas as R
linhas — a cola das hyper-connections (24 produtos sobre 16.384 colunas + Sinkhorn por linha) é redundante.

**Mecanismo.**
- Split: rank r é dono das linhas `[r·H, r·H + H)` com `H = R/2` (chunk ímpar ganha uma linha zero de *pad*, "which
  nothing reads"). Em cada local: (1) cada rank envia ao peer **seu parcial das linhas do peer** (fp32) por
  `ncclSend/ncclRecv` em grupo (`NCCL.exchange`, novo); (2) soma os dois parciais das próprias linhas com o rank 0
  primeiro (`glue.hc_post_pair`, o mesmo kernel `_hc_post` lendo as linhas do rank 1 a uma distância em elementos
  — pode ser negativa — da do rank 0); (3) roda `hc_post` + taps do DFlash2 + média final + próximo `hc_pre` só nas
  suas linhas; (4) troca as linhas `normed` (bf16) (e taps/`hidden`) com o peer, para que ambos tenham a entrada
  completa do próximo bloco. "Every glue kernel is row-independent and the partial sum order is unchanged, so every
  bit is the unsplit path's" `[código]`.
- É o equivalente, para a cola do mHC, do *sequence parallelism* do Megatron-LM: all-gather + computação redundante
  vira "reduce-scatter" + computação local + all-gather `[inferência: analogia]`. Por linha e por local, o tráfego vai
  de 4 B·D (gather fp32) para 4 B·D/2 + 2 B·D/2 = 3 B·D, e a cola cai pela metade `[cálculo próprio]`.
- **Overlap** (`TF_GLM_PREFILL_OVERLAP`): as trocas rodam numa segunda stream CUDA, em pedaços de linhas: o local
  calcula primeiro o parcial das linhas do peer, pedaço a pedaço, cada troca começando quando seu pedaço termina;
  depois as próprias; a cola de cada pedaço espera só seu parcial, e as linhas `normed` são trocadas enquanto a cola
  do pedaço seguinte roda. Eventos (`ev_fill`, `ev_part`, `ev_glue`, `ev_done`) ordenam as streams.
- `TF_GLM_HC_EXCHANGE=gather` usa o all-gather NCCL "puro" (mesmos bytes) para fabrics onde o p2p é ruim; o NCCL
  "puro" porque o kernel RoCE de 0006 "is not written for in place".
- `warm()` abre a conexão p2p no boot; `split_layers` substitui o laço de camadas no prefill quando `R ≥ min_rows`.

**Controle:** `TF_GLM_HC_SPLIT` (0 no patch; receita `SPLIT=1`), `TF_GLM_PREFILL_OVERLAP` (0/1 neste patch; a
receita passa 2 — valor criado por 0033, que também roda a "frente" do bloco seguinte nas próprias linhas durante as
trocas), `TF_GLM_OVERLAP_PIECES` (2; 1–8), `TF_GLM_HC_SPLIT_MIN_ROWS` (64), `TF_GLM_HC_EXCHANGE` (`p2p`).

**Efeito medido:** "prefill ~1,270 -> ~1,730 tok/s on a 50k prompt (with patches 0009 and 0020); decode rounds pay
~1.5%" `[config.sh]`; com 0033 e `COPY_CODE`: "a 149k prompt 92.8 -> 90.7 s" `[config.sh]`.

**Conceitos:** sequence parallelism; reduce-scatter/all-gather vs all-reduce; NCCL p2p em grupo; sobreposição
comunicação/computação com streams e eventos; manter a ordem de soma para preservar bits.

**Latent Atlas:**
- `paralelismos` — seção "sequence parallel da cola mHC" com diagrama unsplit vs split e a conta de bytes.
- `nccl-rdma` — `ncclGroupStart/ncclSend/ncclRecv/ncclGroupEnd`; p2p vs all-gather com o mesmo volume.
- `hiperconexoes` — custo da cola por linha e por que dividi-la compensa.
- `cuda-graphs-warmup` (ou `kernels`) — streams secundárias + eventos como padrão de overlap.

---

### 0011 — `glm-draft-sim`

**Área:** ferramenta de rascunho (`draft_dump.py`, `draft_sim.py` novos; `dflash2.py`, `qmm.py`, `weights.py`,
`engine.py`).

**Problema.** Ajustar regras de parada do DFlash2 com medições ao vivo é caro e ruidoso (cada boot varia).

**Mecanismo.**
- **Dump** (`TF_GLM_DRAFT_DUMP=<dir>`, nos dois ranks): requisições que rascunhariam com DFlash2 decodificam **um
  token por rodada** (resposta = `draft: false`), mas antes de cada rodada rodam o bloco do DFlash2 e gravam:
  candidatos mesclados e logits, arestas do seletor entre candidatos consecutivos, os sorteios Gumbel chaveados da
  requisição, as linhas projetadas, a cadeia que a regra de produção (`fc5:0.3`) proporia; depois do verify, o
  top-32 do alvo (re-derivável: ≥ top_k 20 + `MARGIN` 8), seu logsumexp e o token comprometido. Rank 0 grava um
  `.npz` por requisição.
- **Simulador** (`python -m tensorfold.families.glm5_next.cuda.draft_sim`, numpy puro): como a amostragem é chaveada,
  "a round that starts at position i and proposes a chain keeps exactly the drafts that equal the committed tokens"
  — qualquer política pode ser **reproduzida exatamente** offline. Uma tabela de custos (`VERIFY_MS` para janelas de
  1–8 linhas, `BLOCK_MS`, overhead) converte rodadas em tokens/s. Políticas: `fc<N>:<p>`, `f<N>`, `fa`,
  `nc<N>:<p>[:B]` (ciente do ruído), `cost<N>:<pred>` (maximiza E[tokens]/ms com um preditor), `oracle<N>`
  (programação dinâmica com retrospectiva: limite superior). Preditores de aceitação: `conf`, `noisy` (com β
  ajustado por log loss), `const` (taxa por profundidade), `logit` (regressão logística por Newton/IRLS);
  métricas log loss, Brier, AUC; `check()` confere sorteios, re-amostragem do alvo e a cadeia de produção bit a bit.
- **`TF_GLM_DRAFT_QUANT`** (`minmax`/`mse`): como as cópias de 4 bits só-de-rascunho (matrizes do DFlash2 e a cabeça
  de rascunho EXL3) são feitas — "Drafts only propose, so either keeps every reply; only acceptance can move".

**Controle:** `TF_GLM_DRAFT_DUMP` (vazio = off), `TF_GLM_DRAFT_QUANT` (`minmax`).

**Efeito medido:** custos de referência embutidos: "the measured EXL3 + DFlash2 costs (ms): verify windows of 1..8
rows" = 29,2 / 35,6 / 42,6 / 48,8 / 53,6 / 58,9 / 64,5 / 66,0; bloco DFlash2 3,08 ms `[código: draft_sim.py]`. README:
"how the stop rules were tuned". O β de 0018 foi ajustado em 9 dumps.

**Conceitos:** teacher forcing; avaliação contrafactual offline de políticas viabilizada pela amostragem chaveada;
modelo de custo; preditores calibrados; oráculo por DP.

**Latent Atlas:**
- `dflash-dspark` — seção "ajustar regras de parada offline" (dump + simulador + tabela de custos).
- `especulacao-fundamentos` — por que ruído chaveado torna a aceitação de qualquer cadeia determinística.
- `benchmarking` — simular antes de medir; métricas de calibração (log loss, Brier, AUC).
- `geracao-amostragem` — Gumbel-max chaveado.

---

### 0012 — `glm-kda-chunked`

**Área:** prefill das 34 camadas KDA (`kda_chunked.py` novo; `kda.py`, `forward.py`).

**Problema.** No v0.6.0 o prefill KDA roda a recorrência linha a linha (três kernels, mas sequencial no tempo) — o
caminho clássico para atenção linear em prompt é a forma chunked, que transforma a parte intra-chunk em matmuls.

**Mecanismo** (equações do docstring, estado `S` em layout `[value, key]`):
- Por linha (igual a `kda.cu`): conv + SiLU arredondados a bf16, normas L2 fp32, `q·128^-0.5`, `β = bf16(sigmoid(b))`,
  log-decay `l = lower·sigmoid(exp(A_log)·(a + dt_bias))` por canal de chave; regra delta
  `S ← S·diag(e^l); S ← S + β(v − S k)kᵀ; o = S q`, RMSNorm com gate.
- Sub-chunk de T = 64 linhas, estado de entrada `S0`, `G = cumsum(l)` no sub-chunk (≤ 0, decrescente; todo `l` em
  `(lower, 0) = (−5, 0)`):
  - `L[s, j] = β_s Σ_c k_s[c] k_j[c] e^(G_s[c] − G_j[c])` (j < s, estritamente inferior)
  - `M[s, j] = Σ_c q_s[c] k_j[c] e^(G_s[c] − G_j[c])` (j ≤ s)
  - `T = (I + L)^-1` (triangular inferior unitária)
  - `W = T(β·k·e^G)`, `U = T(β·v)` — transformação WY/UT
  - `V' = U − W S0ᵀ` (valores corrigidos, "v_new")
  - `O = (q·e^G) S0ᵀ + M V'`
  - `S_end = S0·diag(e^G_63) + V'ᵀ(k·e^(G_63 − G))`
- Overflow: os expoentes de `L` e `M` são formados como produto de fatores de linha e coluna em torno de um ponto de
  referência por bloco de 16 linhas: fator de linha em `[e^-75, 1]`, coluna ≤ e^75 dentro do bloco, abaixo do máximo
  fp32 `e^88,7`. `T` é invertida por substituição progressiva em blocos diagonais de 16 e unida por matmuls
  (`T_bj = −T_bb Σ L_bm T_mj`).
- Cinco kernels Triton: `_kda_lm` (sub-chunk, cabeça, bloco: conv, normas, β, decays, linhas de L e M, `q·e^G`),
  `_kda_solve` (T e o log-decay total), `_kda_wu` (W, U, chaves decaídas até o fim), `_kda_state` (a **única parte
  sequencial**: sub-chunk após sub-chunk, `V'` e o estado de entrada de cada um), `_kda_out` (O, bf16, RMSNorm com
  gate). q, k, v e G são **recalculados** das linhas de projeção onde necessários "rather than kept (registers) or
  stored (traffic)".
- Sub-chunks começam em posições absolutas múltiplas de 64; um chunk que começa no meio ganha um sub-chunk curto
  (linhas de *pad* = zero: sem decay, sem chave). A saída de uma linha depende só do seu sub-chunk e do estado de
  entrada → igual em qualquer chunking com fronteiras múltiplas de 64 (daí 0008).
- Matmuls com `tl.dot(..., input_precision=PREC)`: `tf32x3` (padrão, "about fp32: ~2e-6 relative against float64")
  ou `tf32` ("~5e-4, half a bf16 step"). Referências torch (`*_reference`) servem aos testes de CPU.

**Controle:** `TF_GLM_KDA_CHUNKED` (0 no patch; receita `KDA_CHUNKED=1`), `TF_GLM_KDA_PREC` (`tf32x3`),
`TF_GLM_KDA_BV` (16), `TF_GLM_KDA_STATE_ROWS` (32), `TF_GLM_KDA_KSPLIT` (4).

**Efeito medido:** linha "Prompt, KDA" (0012 + 0014 + 0039): "50k prefill 29.3 -> 26.4 s, 149k 91.1 -> 84.5 s (one
start each)" `[README]`. Não é bit-exato: "Close to the serial kernel, not its bits: prompt arithmetic differs,
drafted replies still equal serial ones" `[config.sh]`.

**Conceitos:** forma paralela por chunks da regra delta (DeltaNet/Gated DeltaNet/KDA); representação WY/UT; inversa
triangular; decay diagonal por canal em espaço log; fatoração com ponto de referência; só a passagem de estado é
sequencial; tf32x3 (emular fp32 com três produtos TF32 `[inferência sobre a implementação do Triton]`).

**Latent Atlas:**
- `atencao-linear` — seção nova com as equações acima, o pipeline de 5 kernels, alinhamento a 64 (32 após 0039) e o
  custo/benefício (aproximação controlada vs serial).
- `kernels` — `input_precision` do `tl.dot`; recalcular vs armazenar.
- `numeros-no-computador` — overflow de `exp` em fp32 (`e^88,7`) e fatores de referência.

---

### 0013 — `glm-decode-rounds`

**Área:** decode (`decode.py`, `dflash2.py`, `drafter_choice.py`, `engine.py`, `latent.py`, `qmm.py`, `weights.py`,
`geometry.py`, `copy_drafts.py`).

**Problema.** Janelas de verificação de no máximo 8 linhas limitavam cópias longas; cada rodada tinha
`torch.cuda.synchronize()` para medir tempo, duas cópias `.cpu()` pageáveis dos candidatos do DFlash2 e o
processamento dos tokens (detokenizar, stop strings, escrever o stream) entre rodadas, com a GPU ociosa. `kv_b` do
caminho latente só existia em bf16 e ainda havia cópias `kv_k/kv_v` não usadas.

**Mecanismo.**
- **Janelas maiores:** `MAX_ROWS` 8 → **16** (pendente + até 15 rascunhos, para cópias); `GRAPH_ROWS` (1–6) → (1–8)
  capturados como grafos CUDA, "wider ones run eagerly"; `MTP_ROWS = 8` (os buffers do MTP não precisam ser tão
  largos: "draft chains take one row a step"); a constante `MLA_DECODE_ROWS = 64` passa a nomear as linhas de decode
  da estimativa de memória (o valor já era 64); a calibração de
  custos só usa janelas até 8 linhas ("their eager launches would bend the fitted line").
- **`_Clock`:** tempo da janela por eventos CUDA, lido depois que a amostra já esperou a GPU — "timing adds no
  host-device sync".
- **`_Late`:** `on_tokens` uma rodada atrasado: os tokens de uma rodada vão para o servidor quando a próxima janela já
  está enfileirada na GPU, então "the server's work on them […] runs while the GPU verifies instead of between
  rounds"; `flush()` final entrega tudo antes de retornar.
- **Candidatos do DFlash2:** valores, ids e linhas projetadas num único buffer de device lido por **uma** cópia pinned
  não bloqueante + evento ("two .cpu() calls were two syncs and two pageable copies"); grafos de atualização de
  contexto do drafter até `tap_rows = MAX_ROWS`.
- **`TF_GLM_KVB`:** `kv_b_proj` do caminho latente em `bf16` (como armazenado, padrão), `fp8` (`AbsorbF8`: e4m3 por
  cabeça com escala fp32 por linha e 128 colunas do latente; kernels `_absorb_q8`/`_expand_v8` de decode
  desquantizam o bloco uma vez por programa e fazem uma soma fp32 por linha "whatever R is"; `_absorb_mma8`/
  `_expand_mma8` para prompt) ou `q4` (`AbsorbQ4` com `quantize4_mse_raw`, layout MLX). E as cópias `kv_k/kv_v`
  passam a existir só com `TF_GLM_LATENT=0` — memória a menos.

**Controle:** `TF_GLM_KVB` (`bf16`; a receita não define → bf16), `TF_GLM_COPY_MAX` até 15.

**Efeito medido:** "Copy drafts a round (patch 0013: verify windows up to 16 rows): 15 lets a long quote go through in
one round; edit replies 84 -> 119 tok/s, prose and code unchanged (two boots each). Exact." `[config.sh]`; README,
linha "Decode": "faster decode rounds; long copy drafts (`COPY_MAX` 15: edit replies 84 -> 119 tok/s)".

**Conceitos:** largura da janela de verificação vs tamanhos de captura de grafos; eliminar sincronizações host-device
no caminho quente; cópias pinned assíncronas; sobreposição CPU/GPU (como o processamento assíncrono de saída de
motores de serving); absorção de pesos do MLA com `kv_b` quantizado.

**Latent Atlas:**
- `cuda-graphs-warmup` — capturar 1–8 e rodar 9–16 em eager; trocar `synchronize()` por eventos; uma cópia pinned;
  o padrão `_Late`.
- `especulacao-fundamentos` — janelas de até 16 linhas para cópias longas.
- `mla` — absorção com `kv_b` FP8/Q4 e kernels por linha independentes de R.
- `scheduler-batching` — `_Late` como análogo do agendamento/saída assíncronos.

---

### 0014 — `glm-kda-chunked-gb10`

**Área:** prefill KDA (só `kda_chunked.py`).

**Problema.** A primeira versão do chunked (0012) tinha erro numérico 10× maior que a cadeia de `kda.cu`, divergia de
`kda.cu` em arredondamentos bf16 e usava estágios de pipeline que não cabem na shared memory do GB10.

**Mecanismo.**
- **Estabilidade:** "a partial sum's rounding error is a few ulps of the partial sum itself, while G reaches -320 in a
  sub-chunk and a difference of two G's would carry ulp(320) = 3e-5 into decays near 1 (the first version did: 10x
  the error of kda.cu's chain)" `[código]`. Agora cada expoente é **somado para fora a partir da referência**
  (para frente, para trás, ou para trás a partir do fim do sub-chunk para `k·e^(G_63 − G)`), nunca como diferença de
  G's. Tile diagonal em grupos de `GROUP = 4` linhas em torno da primeira linha de cada grupo (fatores de coluna
  ≤ `e^15`); blocos anteriores em torno da primeira linha do bloco (ambos os fatores ≤ 1).
- **Paridade com `kda.cu`:** "a value rounded the other way at a bf16 tie moves a key by 2^-8 in one channel, a beta
  by 2^-8 of the whole update". Como `kda.cu` é compilado com `--fmad=false` e usa `expf` e divisão IEEE, os
  kernels Triton rodam com `enable_fp_fusion=False`, `enable_reflect_ftz=False`, `libdevice.exp`, `tl.math.div_rn`,
  `sqrt_rn` (no interpretador do Triton, `tl.exp`).
- **GB10:** "One stage everywhere: GB10 (sm_121) gives a block 99 KiB of shared memory (101,376 B), and pipelined loads
  of the per-block loops need up to 154 KiB"; tabela `LAUNCH` de `(num_warps, num_stages)` por kernel.

**Controle:** só constantes (`LAUNCH`, `GROUP`); mesmas variáveis de 0012.

**Efeito medido:** precisão "read-outs ~3e-7 relative against float64 on the tests' inputs, kda.cu's chain ~1e-7"
`[código]`; desempenho entra no número da linha "Prompt, KDA" (ver 0012).

**Conceitos:** cancelamento e crescimento de ulp (ulp(320) em fp32 ≈ 3·10⁻⁵); FMA muda arredondamento; empates em bf16
amplificados; reproduzir a aritmética de um kernel CUDA em Triton; capacidade de shared memory por bloco do sm_121
limita estágios de pipeline.

**Latent Atlas:**
- `numeros-no-computador` — exemplo concreto: G até −320, ulp(320) ≈ 3·10⁻⁵, por que somar a partir da referência; FMA
  vs mul+add; empates em bf16.
- `gpu-por-dentro` — 99 KiB (101.376 B) de shared memory por bloco no GB10 e o efeito em `num_stages`.
- `atencao-linear` — estabilidade numérica da forma chunked.
- `kernels` — flags do Triton (`enable_fp_fusion`, libdevice) para paridade.

---

### 0015 — `glm-shared-prefix`

**Área:** reuso de prompt (`engine.py`, `decode.py`).

**Problema.** O v0.6.0 guardava só o estado no fim do prompt; uma conversa nova com o mesmo system prompt não
reaproveitava nada (o prompt novo não estende o antigo token a token).

**Mecanismo.**
- `shared_points`: além do próprio ponto (`grid_point`), uma requisição guarda até `SHARED_MOST = 2` estados extras:
  no **maior prefixo** que compartilha com algum prompt guardado (comparação vetorizada com numpy) e no **primeiro
  `<|user|>`** (fim do bloco de sistema; id lido do `tokenizer.json`), cada um arredondado para baixo à grade e só se
  acrescentar ≥ `least` tokens além do ponto de retomada e do ponto anterior, e ficar antes do ponto final.
- Cada ponto é pré-preenchido "as a prompt of its length" e retomado — fresco e retomado passam pelos mesmos
  estados. O rank 0 escolhe os pontos e os manda no header (2 slots); o rank 1 (`follow_one`, refatorado) os usa.
- Snapshots salvos passam a levar também a janela do contexto do DFlash2 (`_drafter_views`: chaves/valores das
  `window + 1` linhas anteriores), para que um estado retomado sirva ao drafter. O snapshot usado como retomada vira
  o mais recente (ordem de evicção).

**Controle:** `TF_GLM_SHARED_PREFIX` (0 = off; `1`/`on` = 256 tokens; ou ≥ 16); receita `SHARED_PREFIX=1`. Relacionados
(da receita/v0.6.0): `TF_GLM_CACHE_ENTRIES` (32 na receita, 8 no TensorFold: "An agent request keeps 1 to 3 (its own
plus shared-prefix states), so 8 let three or four alternating conversations push each other out (issue #17)"
`[config.sh]`) e `TF_GLM_CACHE_GIB` (`KV_POOL_GIB`).

**Efeito medido:** "a 7.9k-token system prompt's second and later chats prefill in 0.13 s instead of 4.24 s. Same
replies." `[config.sh]`.

**Conceitos:** prefix caching com ramificação (pontos de divergência sem árvore explícita); compartilhamento de system
prompt em agentes; snapshot = linhas de atenção + estado recorrente + janela do drafter; orçamento e evicção.

**Latent Atlas:**
- `prefix-caching` — seção TensorFold: pontos de divergência e fim do bloco de sistema; 256 tokens mínimos; 2 por
  requisição; contraste com o hash encadeado de blocos do APC.
- `dflash-dspark` — o contexto do drafter também precisa ser salvo/restaurado.

---

### 0016 — `glm-decode-kernels`

**Área:** decode (matmuls Q4, experts EXL3, KDA).

**Problema.** Os kernels de decode usavam um tile único por forma, epílogos separados (ida e volta de `Z` fp32 pela
memória) e o KDA com um bloco por cabeça em janelas pequenas.

**Mecanismo.**
- **`qmm_cfg`** (matmul Q4 das projeções densas `q4`, linhas ≤ 16, grupos de 64): 12 configurações de
  `(colunas do tile, warps, estágios)` "every output's chain […] is qmm_kernel's, so every choice gives the same bits";
  tabela `DEC_TILES` por forma por rank `(n, k)` "measured on GB10". O redutor do split-K usa *thread block clusters*
  quando possível ("Clusters hold up to 8 K slices (the portable size) from sm_90" — código base do v0.6.0) ou um
  buffer + `reduce_kernel`.
- **`dec_kernel` EXL3:** a aritmética do `grouped_kernel` (com prefetch) e **epílogo fundido**:
  - `FUSE 1` (down, SK = 1): as 128 colunas do bloco são um bloco Hadamard; aplica `fwht128` + `svh_d` nas próprias
    somas e escreve `Y` — "no Z round trip, no epilogue launch";
  - `FUSE 2` (gate/up): cada bloco escreve seu parcial; o **último** dos `2·SK` blocos de um (item, bloco-n) a
    terminar (contador `atomicAdd` após `__threadfence`, zerado pelo próprio kernel para o próximo lançamento) roda o
    epílogo lendo os parciais em ordem de split (`__ldcg`) — "measured slower", então o padrão é só o down;
  - `XROW`: com `suh` compartilhado (0009), a rotação é uma vez por linha também no decode.
- **KDA de decode:** `WIDE_ROWS` padrão 1 → toda janela usa o caminho de três kernels (prep/step/out, "8 blocks a head
  instead of one; chain_kernel's bits"); `reserve()` aloca o scratch antes de qualquer captura — "a graph keeps the
  addresses it was captured with" — e buffers superados ficam vivos (`_outgrown`).

**Controle:** `TF_GLM_Q4_TILE` (`table`; `stock`; ou um número para todas as formas), `TF_GLM_EXL3_DEC` (1),
`TF_GLM_EXL3_FUSE` (1; 0–3), `TF_GLM_EXL3_XROW` (1), `TF_GLM_KDA_WIDE_ROWS` (1; 64 = comportamento anterior).

**Efeito medido:** KDA de decode "measured 4-36% faster at 1-8 rows on GB10"; laço não desenrolado ("unrolling it
measured slower"); FUSE 2 "measured slower" `[código]`. README: "faster decode rounds" (linha "Decode"), sem número
isolado.

**Conceitos:** tabela de tiles por forma (autotuning por tabela); fusão de epílogo; padrão "último bloco" com
`__threadfence` + atômico; split-K com clusters; endereços estáticos para grafos CUDA; paralelizar uma recorrência em
mais blocos (por fatias de valor).

**Latent Atlas:**
- `kernels` — fusão de epílogo e o padrão "último bloco"; tabela de tiles; clusters no split-K.
- `cuda-graphs-warmup` — reservar scratch antes da captura; não liberar buffers que um grafo ainda referencia.
- `atencao-linear` — decode KDA em três kernels.

---

### 0017 — `glm-overlap-priority`

**Área:** prefill com dois Sparks (`hcsplit.py`).

**Problema/Mecanismo.** A stream lateral do overlap (0010) era sempre de prioridade alta (`priority=-1`); o patch a
torna configurável por `TF_GLM_OVERLAP_PRIORITY` (padrão ainda -1 neste patch) porque "a high-priority stream in the
process may cost decode rounds (graphs on the default stream) a little: measured per boot" `[código]`.

**Controle:** `TF_GLM_OVERLAP_PRIORITY` (-1 aqui; 0022 muda o padrão para 0).

**Efeito medido:** ver 0022.

**Conceitos:** prioridade de streams CUDA (número menor = prioridade maior); interação com grafos na stream padrão;
efeitos pequenos exigem medir em vários boots.

**Latent Atlas:** `cuda-graphs-warmup` (prioridades de stream); `benchmarking` ("measured per boot").

---

### 0018 — `glm-noise-policies`

**Área:** rascunho (`dflash2.py`, `decode.py`, `engine.py`).

**Problema.** A regra de produção `fc5:0.3` para a cadeia quando o produto das confianças **sem ruído** do drafter
cai abaixo de 0,3; mas o alvo amostra com ruído chaveado que já é conhecido antes do verify.

**Mecanismo.**
- A cadeia do DFlash2 escolhe, a cada profundidade, `argmax(score + NOISE·g)` com `score = (logits + EDGE·aresta)/T`,
  `EDGE = 0,6`, `NOISE = 0,7`, e `g` os sorteios Gumbel **do alvo** para aqueles candidatos naquelas posições.
- **Confiança ciente do ruído:** `noisy_confidence = softmax((score + g)/β)[j]` — "the chance the target's keyed sample
  is the pick when its scaled logits are the drafter's plus iid Gumbel(β) errors" `[código]`. β = `NOISY_BETA = 2.0`,
  ajustado por log loss nos dumps de 0011: "on the 9 teacher-forced dumps of 2026-09-30 (7 sampled prose, 2 greedy
  code): 2.0 on all of them, on the sampled ones alone and on the greedy ones alone (a finer grid: 1.8, log loss
  -0.2%)".
- Políticas novas: `fncN:P[:B]` (produto das confianças cientes do ruído ≥ P, primeiro rascunho sempre mantido) e
  `fcostN:noisy[:B]` (anda a cadeia inteira e corta em `best_depth`: o k que maximiza
  `E[tokens | k] / round_ms[k]`, com `E = Σ_{j≤k} Π_{i≤j} q_i + 1` e `round_ms` da tabela de calibração do boot:
  verify(k+1) + bloco + atualização de uma linha de contexto). "Only where the chain stops changes: the picks are the
  same."
- Codificação: `encode_policy` gera `[16, N, P, B]` / `[17, N, B, 0]` (kind + 10 = DFlash2); `decode_policy` recebe a
  tabela de custos.

**Controle:** a string de política (`DRAFT_POLICY`; via `TF_GLM_DFLASH_POLICY` de 0021). Receita: `fnc7:0.3`.

**Efeito medido:** "fnc7:0.3 stops a chain when the product of each draft's chance under the request's own keyed
sampling noise drops below 0.3 (up to 7 drafts); prose 47.2 -> 50.6 tok/s, code 51.7 -> 55.5 over fc5:0.3 (two boots
each). Drafts only propose: replies are the same under every policy." `[config.sh]`.

**Conceitos:** truque Gumbel-max; acoplamento rascunho/alvo por ruído compartilhado; probabilidade de casar sob erro
Gumbel(β); calibração por log loss; regra de parada por probabilidade acumulada; profundidade ótima por custo.

**Latent Atlas:**
- `dflash-dspark` — regras `fc`/`fnc`/`fcost` com as fórmulas, β = 2,0, ganhos medidos.
- `especulacao-fundamentos` — k ótimo = argmax E[tokens|k]/custo(k); a fórmula com probabilidades condicionais.
- `geracao-amostragem` — Gumbel-max chaveado e por que conhecer o ruído do alvo antes do verify ajuda o rascunho.

---

### 0019 — `glm-decode-kernels2`

**Área:** decode (`qmm.cu`, `forward.py`, `glue.py`, `hc.cpp`/`hc.cu` novos, `latent.py`, `qmm.py`).

**Mecanismo.**
- **`qmm_cfg` com tiles de 32 colunas** (configs 12–15): carregar parte de um tile armazenado de 64 colunas ("its n8
  tiles are contiguous"); `DEC_TILES[(4096, 1024)]` 6 → 2.
- **`combine_shared` também no decode** (o caminho de prompt de 0009): `TF_GLM_COMBINE_SY`.
- **`hc_partial` em CUDA** para linhas de decode (a primeira metade do `hc_pre`): grade `(16 blocos de K, 3 partes,
  ⌈linhas/4⌉)` = 48 blocos por linha em vez de 16, 288 threads (warps 0–7: 8 linhas de mistura cada; warp 8 da parte
  0: soma de quadrados), pesos lidos uma vez para até `RPB = 4` linhas. Reproduz **os bits** do kernel Triton
  `_hc_partial`: o docstring descreve o layout do Triton (`sizePerThread [1, 8]`, `threadsPerWarp [2, 16]`), a soma de
  8 produtos por thread em ordem, *butterflies* 8, 4, 2, 1, os passos somados a partir de zero, e para a soma de
  quadrados *butterflies* 16…1 e `(w0 + w2) + (w1 + w3)` entre warps — reimplementados com `__fmaf_rn`/`__fadd_rn`/
  `__shfl_xor_sync`.
- **`TF_GLM_LATENT_STAGES`** (3): profundidade do pipeline Triton dos programas de chunk densos/esparsos — "the loads
  only, the same bits (1: none, as before; 4 does not fit in shared memory)".

**Controle:** `TF_GLM_COMBINE_SY` (1), `TF_GLM_HC_DEC` (1), `TF_GLM_LATENT_STAGES` (3), `TF_GLM_Q4_TILE` (configs 12–15).

**Efeito medido:** sem número isolado; README linha "Decode" ("faster decode matmuls, hyper-connection mixing and
indexer scoring" — o *indexer scoring* é 0031, parte B).

**Conceitos:** reproduzir a árvore de redução de outro kernel para preservar bits; mais blocos para M pequeno
(ocupação no decode); reuso de pesos entre linhas; `num_stages` do Triton e limite de shared memory.

**Latent Atlas:**
- `kernels` — "reescrever um kernel Triton em CUDA sem mudar um bit": copiar layout e ordem das reduções.
- `hiperconexoes` — o `hc_pre` no decode e seu custo.
- `numeros-no-computador` — a árvore de redução define os bits.

---

### 0020 — `glm-prompt-experts-order`

**Área:** prefill, experts EXL3 (`exl3.cu`, `exl3.cpp`, `exl3_mm.py`).

**Problema.** No plano de prompt, experts que recebem um único item (poucos tokens) leem seus pesos da DRAM e fazem
pouco `mma` (limitados por memória); experts com vários itens compartilham pesos pelo L2 e são limitados por tensor
core. Em ordem de plano, os dois tipos chegam em ondas, e DRAM e tensor cores trabalham "in turns".

**Mecanismo.**
- **`prompt_order_kernel`** (um bloco de 1024 threads): marca cada item como "pesado" (o expert tem itens vizinhos no
  plano) ou "leve"; faz *prefix sum* (scan com `__shfl_up_sync` por warp + somas de warp) para o rank de cada item
  pesado; intercala as duas listas uniformemente (pesado de rank r na posição fracionária (r + ½)/nh, leve de rank l
  em (l + ½)/nl), mantendo a ordem de plano em cada lista; deixa de fora os itens do expert compartilhado (`-1`).
  "Only which block runs which item changes: every output keeps its bits."
- **`decode_tile_p`:** os mesmos estados de 16 bits do código de treliça com menos instruções inteiras
  (`__byte_perm`/PRMT e `__funnelshift_r`; `w >> 4` compartilhado).
- Pula o `mma` da metade superior (membros 8–15) quando ela é toda vazia; *stores* vetorizados (`uint2` de 4 halves,
  `float4`).

**Controle:** `TF_GLM_EXL3_ORDER` (1).

**Efeito medido:** sem número isolado; o ganho de 0010 foi medido "with patches 0009 and 0020" `[config.sh]`; README
linha "Prompt": "launched in a better order".

**Conceitos:** misturar trabalho limitado por memória e por computação para ocupar ambos os recursos; scan por warp;
desbalanceamento de carga entre experts no prefill; micro-otimização de extração de bits.

**Latent Atlas:**
- `kernels` — ordenar trabalho heterogêneo por gargalo; scan por warp; PRMT.
- `mixture-of-experts` — distribuição de tokens por expert num chunk de prefill e por que ela importa para o kernel
  (experts "leves" vs "pesados").

---

### 0021 — `glm-dflash-policy-env`

**Área:** rascunho (`engine.py`).

**Mecanismo.** `DFLASH_POLICY = TF_GLM_DFLASH_POLICY` ou `fc5:0.3`; é a política usada por requisições que não pedem
outra (e o `auto` num checkpoint EXL3 com drafter). Valida no boot que é uma política DFlash2 (`f…`, kind ≥ 10).

**Controle:** `TF_GLM_DFLASH_POLICY`; receita `DRAFT_POLICY=fnc7:0.3` → `TF_GLM_DFLASH_POLICY`.

**Efeito medido:** o de 0018.

**Latent Atlas:** `dflash-dspark` (como a receita escolhe a regra de parada padrão).

---

### 0022 — `glm-overlap-normal-priority`

**Área:** prefill com dois Sparks (`hcsplit.py`).

**Mecanismo.** Inverte o padrão de 0017: `TF_GLM_OVERLAP_PRIORITY` passa a valer 0 (normal); só `-1` pede alta.

**Efeito medido:** "Two Sparks, two boots each: prefill the same, decode rounds after a high-priority prompt ~0.5-1%
slower" `[código]`.

**Conceitos:** um recurso que só ajuda o prefill (prioridade) pode cobrar no decode; A/B com vários boots.

**Latent Atlas:** `benchmarking` (efeitos de 0,5–1% exigem boots repetidos; reverter quando não paga);
`cuda-graphs-warmup` (prioridades).

---

### 0023 — `server-effort-max`

**Área:** servidor (`tensorfold/server/request_options.py`).

**Mecanismo.** `EFFORTS` ganha `"max"`; `effort_levels` reconhece `max` no template; `coerce_effort` mapeia `"max"`
para o nível de topo do próprio template quando ele o nomeia (o GLM-5.3 nomeia `max`, seu padrão), senão `xhigh`;
`medium` vira `high` quando o template nomeia `high` mas não `medium`. Mensagem de erro atualizada.

**Controle:** `reasoning_effort` na requisição (topo ou `chat_template_kwargs`). README: "`"medium"` is `"high"`,
`"xhigh"` is `"max"`. Without it, GLM's template uses `max`; `"none"` […] answers without thinking."

**Efeito medido:** recurso de API.

**Latent Atlas:** `geracao-amostragem` (níveis de esforço de raciocínio e mapeamento entre APIs);
`api-tools-estruturado` (normalização de `reasoning_effort`).

---

### 0024 — `glm-prompt-select-rows`

**Área:** prefill DSA (`sparse.py`).

**Problema.** `_select_prompt` (0009) alocava os scores fp32 de **todas** as R linhas do chunk contra todos os pools.

**Mecanismo.** Processa as linhas em blocos de `SELECT_ROWS` (= `MLA_SELECT_ROWS` = 512): cada bloco pontua os mesmos
NP pools ("the same columns, so the same bits a row"), seleciona o top-512 e escreve sua fatia de tokens/contagens;
o buffer de scores tem `512 × pool_bucket` floats em vez de `R × pool_bucket`.

**Controle:** nenhum (constante).

**Efeito medido:** README linha "Prompt": "token selection in blocks of 512 rows […] less memory at 1M". Ordem de
grandeza: com R = 2048 e ~262.144 pools visíveis a 1M (1.048.576/4), `R × NP × 4 B` ≈ 2 GiB vs ≈ 512 MiB em blocos de
512 `[cálculo próprio, assumindo NP ≈ pools visíveis; o buffer real usa pool_bucket, que pode ser maior]`.

**Conceitos:** picos transitórios de memória no prefill a 1M; processar em blocos com as mesmas colunas preserva bits.

**Latent Atlas:** `atencao-esparsa` (custo de memória da pontuação de pools no prefill longo); `memoria-unificada`
(picos transitórios no orçamento de UMA).

---

### 0025 — `glm-dflash2-ring`

**Área:** rascunho + reuso (`decode.py`).

**Mecanismo.** Com `TF_GLM_DRAFT_RING` (v0.6.0: o contexto do DFlash2 num anel de 2.176 linhas em vez de um buffer
plano da janela inteira), um snapshot não pode copiar `kc/vc[:, lo:n]` por fatia; `_saved_views` deixa de incluir as
views do drafter quando ele usa anel, porque os snapshots de anel já levam `drafter_rows`.

**Controle:** `TF_GLM_DRAFT_RING` (do v0.6.0, padrão 1).

**Efeito medido:** README, linha "Drafts": "kept prompt states' DFlash2 window with shared prefixes […] memory for the
window".

**Conceitos:** *ring buffer* para janela deslizante do drafter (2.176 = 2.048 + 128 `[cálculo próprio; a janela
deslizante do DFlash2 é 2048, arch-glm.md]`); snapshots de buffers circulares.

**Latent Atlas:** `dflash-dspark` (contexto do drafter em anel); `prefix-caching` (snapshot do drafter).

---

### 0026 — `glm-multi-kda`

**Área:** concorrência (KDA de várias sequências numa janela; `kda.cu`, `kda.cpp`, `kda.py`, `forward.py`).

**Problema.** Para verificar as janelas de várias requisições num só forward ("one batched verify window a round"
`[README]`), cada camada KDA precisa rodar a recorrência de cada sequência a partir do **seu** estado e da **sua**
janela de convolução.

**Mecanismo.**
- **Tabela de segmentos** `int32 [nseg, 6]`: (primeira linha, linhas, slot de estado, paridade, slot de conv, keep).
  As linhas das sequências ficam lado a lado na janela; o segmento i roda de `rec[slot, parity]` com os taps de
  `conv[conv slot]` para `rec[slot, 1 − parity]` (estado em *ping-pong*).
- Kernels: `prep_seg_kernel` (por linha e cabeça: conv com os taps do segmento, SiLU, normas, decays, β),
  `step_seg_kernel` (por segmento × cabeça e fatia de valores: a recorrência com o mesmo `update` dos kernels solo),
  `out_kernel`; "a segment's bits equal a solo run of its rows".
- **Rollback por replay:** `replay_layers_seg_kernel` refaz, para segmentos com `keep < rows`, as primeiras `keep`
  linhas a partir dos k/v/g/β salvos, em todas as camadas — o estado após a aceitação parcial é recomputado em vez de
  guardado por posição.
- `conv_shift_segments` (Triton): atualiza a janela de convolução de cada segmento sobre suas próprias linhas.
- Segmentos de 0 linhas são pulados, então "a table sized for the most streams (and the grids a CUDA graph captured)
  serves fewer: rewrite the table in place between replays".

**Controle:** nenhum próprio; usado pela pilha de concorrência (0027, 0029, 0030, 0035… parte B) com `PARALLEL`.

**Efeito medido:** parte de "4 requests at once" `[README]`: 4 concorrentes, prosa 108,8 tok/s agregado; respostas
servidas 4 de cada vez "identical to the same requests served one at a time (11 of 11 cases staggered, and 11 of 11
sent in a burst)".

**Conceitos:** batching de camadas recorrentes com estado por sequência (slots); tabela de segmentos como "metadados
de lote" dentro de um grafo CUDA de grade fixa; rollback de estado por replay vs guardar `num_spec + 1` estados.

**Latent Atlas:**
- `atencao-linear` — estado por sequência em slots, rollback por replay.
- `scheduler-batching` — batching de modelos híbridos: várias sequências numa janela.
- `cuda-graphs-warmup` — grade fixa com segmentos vazios; tabela reescrita in place entre replays.
- `especulacao-fundamentos` — rollback por replay como alternativa a guardar `num_spec + 1` estados (o que o vLLM faz).

---

## 3. Cruzamento com a tabela "What the patches change" do README

Para cada linha do README: os patches dela que estão em 0001–0026 (deste arquivo) e os que ficam na parte B.

| Linha do README | Em 0001–0026 | Parte B | Efeito declarado no README | Observação |
|---|---|---|---|---|
| **Weights** | 0002, 0005 | — | q4 sobre fp8: prosa 38,9→44,4; código 44,2→48,7; prefill ~1.090→~1.260 tok/s | O kv_b fica BF16 no caminho latente (0013; §6) |
| **KV cache** | — | 0038 | 1M com 4 requisições cabe | — |
| **Prompt** | 0001, 0004, 0009, 0020, 0024 | 0028 | kernels EXL3 que mantêm linhas no L2 (0001), lançados em melhor ordem (0020); entrada rotacionada uma vez por linha (0009); atenção densa só onde a esparsa precisa (0009); seleção em blocos de 512 linhas (0024); scratch menor (0028) → "faster prefill, less memory at 1M" | Sem números isolados por patch; os kernels de prompt (0004) não são citados nominalmente na coluna "Change" mas estão na lista |
| **Prompt, two Sparks** | 0010, 0017, 0022 | 0033 | cola mHC dividida por linhas, trocas sobrepostas (`SPLIT`); 50k ~1.270→~1.730 tok/s "with 0009 and 0020"; decode paga ~1,5% | 0017 introduziu prioridade alta, 0022 reverteu o padrão para normal |
| **Prompt, KDA** | 0012, 0014 | 0039 | forma chunked (WY) num kernel CUDA (`KDA_CHUNKED`); 50k 29,3→26,4 s; 149k 91,1→84,5 s | Os 64-row sub-chunks Triton de 0012/0014 viram 32-row num kernel CUDA em 0039 |
| **Prompt reuse** | 0008, 0015 | 0042 | grade de tokens; prefixo compartilhado; prompt idêntico retomado do estado final | 64k idêntico 34 s → < 0,07 s (0042); 7,9k de system prompt 4,24 s → 0,13 s (0015) |
| **Decode** | 0013, 0016, 0019 | 0031 | janelas até 16 linhas; matmuls de decode, mistura mHC e pontuação do indexador mais rápidas; `COPY_MAX` 15: edit 84→119 tok/s | pontuação do indexador = 0031 |
| **Decode, memory** | — | 0046, 0047 | L2 prefetch e cargas de treliça de 16 B; +2,7% / +3,3% | — |
| **Decode, indexer** | — | 0043 | — | — |
| **Link** | 0006 | 0052 | RDMA one-shot; 11 µs vs 45 µs; decode +6%; até 512 KiB: código a 4 streams +1,4% | 0052 corrige a primeira partida (#6) |
| **Drafts** | 0007, 0018, 0021, 0025 | 0032 | cópias antes do DFlash2 (`COPY`, `COPY_CODE`); regras de parada cientes do ruído (`DRAFT_POLICY`); janela do DFlash2 com prefixos compartilhados | `COPY`: quote/edit 80,3→84,4; `DRAFT_POLICY` sobre `fc5:0.3`: prosa 47,2→50,6, código 51,7→55,5; `COPY_CODE` é 0032 |
| **Drafts, tooling** | 0011 | — | simulador offline das regras (`TF_GLM_DRAFT_DUMP`, off) | "how the stop rules were tuned" |
| **Concurrent requests** | 0026 | 0027, 0029, 0030, 0035, 0040, 0041, 0048, 0049 | vários streams num pool compartilhado; uma janela de verificação em lote por rodada | `MULTI_PREFILL` (0049): 4 prosa 103,4→108,8 tok/s, TTFT 590→340 ms |
| **Sampling** | — | 0034 | top_p dos dois ranks juntos (opt-in) | — |
| **Server** | 0003, 0023 | 0036, 0037, 0044, 0045, 0050, 0051, 0053 | visão (0003), `reasoning_effort: "max"` (0023) | limites de 50 imagens / 4 vídeos vêm de 0050 |

Patches de 0001–0026 que o README não destaca por si na coluna "Change": **0004** (kernels de prompt Yᵀ=WᵀXᵀ, aparece
só na lista "Prompt"), **0016/0019** (agrupados como "faster decode matmuls, hyper-connection mixing"), **0014**
(ajuste de GB10 dentro de "Prompt, KDA").

---

## 4. Variáveis de ambiente introduzidas por 0001–0026

"Bits" = se mudar o valor muda a saída. "Receita" = o que `scripts/config.sh` exporta (vazio = não define).

| Variável | Patch | Padrão no patch | Receita | Faz | Bits |
|---|---|---|---|---|---|
| `TF_GLM_EXL3_NFIRST` | 0001 | 1 | — | ordem de lançamento "n primeiro" em janelas ≥ 64 linhas | = |
| `TF_GLM_EXL3_PF` | 0001 | 1 | — | prefetch em registradores no kernel EXL3 | = |
| `TF_GLM_DENSE` | 0002/0005 | `bf16` | `q4` (`DENSE`) | pesos densos bf16 / fp8 / q4 | lossy |
| `TENSORFOLD_GLM_IMAGE_TOKENS` | 0003 | 2048 (16–8000) | 2048 | tokens por imagem | — |
| `TENSORFOLD_GLM_VIDEO_TOKENS` | 0003 | 16384 (64–240000) | 16384 | tokens por clipe | — |
| `TENSORFOLD_GLM_VIDEO_FRAMES` | 0003 | 128 (2–2048) | 128 | quadros por clipe | — |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | 0003 | auto | — | reserva de workspace da torre | — |
| `TF_GLM_EXL3_PROMPT` | 0004 | 1 | — | kernels de prompt Yᵀ=WᵀXᵀ | ≠ (prompt) |
| `TF_GLM_EXL3_PASS` | 0004 | 64 | — | membros por passada (64/128) | = |
| `TF_GLM_EXL3_SEQ` | 0004 | 1 | — | splits somados num programa | = |
| `TF_GLM_HC_MMA` | 0004 | 1 | — | `hc_pre` de prompt em tensor cores | ≠ (prompt) |
| `TF_GLM_LATENT_MMA` | 0004 | 1 | — | absorb/expand de prompt em tensor cores | ≠ (prompt) |
| `TF_GLM_SEQ_ROWS` | 0004 | 256 | — | limiar de `SEQ` nas matmuls BF16/FP8 | = |
| `TF_GLM_PREFILL_ROWS` | 0004 | 2048 (64–16384, múlt. de 64) | — | linhas por chunk de prompt | = (com grade) |
| `TF_GLM_COMM` | 0006 | `nccl` | `roce` (`COMM`) | all-gathers pequenos por RoCE | = |
| `TF_ROCE_MAX_KB` | 0006 | 256 | 512 | maior gather por RoCE | = |
| `TF_ROCE_HCA` / `NCCL_IB_HCA` | 0006 | portas ativas (até 2) | — | HCAs usados | = |
| `TF_ROCE_GID_INDEX` / `NCCL_IB_GID_INDEX` | 0006 | 3 | — | índice de GID RoCE v2 | = |
| `TF_GLM_COPY_DRAFTS` | 0007 | 0 | 1 (`COPY`) | copy drafts | = |
| `TF_GLM_COPY_MATCH` | 0007 | 8 (2–64) | — | tokens do sufixo a casar | = |
| `TF_GLM_COPY_MAX` | 0007/0013 | min(5, máx.) | 15 (`COPY_MAX`) | cópias por rodada | = |
| `TF_GLM_PROMPT_GRID` | 0008 | 0 | (64 via `KDA_CHUNKED`, 0039) | grade dos estados de prompt | = |
| `TF_GLM_EXL3_ROWROT` | 0009 | 1 | — | rotação 1×/linha no prompt | = |
| `TF_GLM_SPARSE_ONEPASS` | 0009 | 1 | — | atenção esparsa de prompt em uma passada | ≠ (prompt) |
| `TF_GLM_SCORE_RB` | 0009 | 4 | — | linhas por programa de pontuação de pools | = |
| `TF_GLM_HC_SPLIT` | 0010 | 0 | 1 (`SPLIT`) | cola mHC dividida por linhas | = |
| `TF_GLM_PREFILL_OVERLAP` | 0010 (0033) | 0 | 2 | trocas sobrepostas (2 = + frentes, 0033) | = |
| `TF_GLM_OVERLAP_PIECES` | 0010 | 2 (1–8) | — | pedaços por troca | = |
| `TF_GLM_HC_SPLIT_MIN_ROWS` | 0010 | 64 | — | chunks menores não dividem | = |
| `TF_GLM_HC_EXCHANGE` | 0010 | `p2p` | — | p2p ou all-gather | = |
| `TF_GLM_DRAFT_DUMP` | 0011 | vazio | — | grava dumps do DFlash2 (decode serial) | = |
| `TF_GLM_DRAFT_QUANT` | 0011 | `minmax` | — | cópias de 4 bits só de rascunho | = (só aceitação) |
| `TF_GLM_KDA_CHUNKED` | 0012 | 0 | 1 (`KDA_CHUNKED`) | KDA de prompt em forma chunked | ≠ (próximo) |
| `TF_GLM_KDA_PREC` | 0012 | `tf32x3` | — | precisão do `tl.dot` | ≠ |
| `TF_GLM_KDA_BV` / `_STATE_ROWS` / `_KSPLIT` | 0012 | 16 / 32 / 4 | — | geometria dos kernels chunked | — |
| `TF_GLM_KVB` | 0013 | `bf16` | — | `kv_b` do latente bf16/fp8/q4 | lossy se ≠ bf16 |
| `TF_GLM_SHARED_PREFIX` | 0015 | 0 | 1 (`SHARED_PREFIX`) | estados de prefixo compartilhado (256 tokens) | = |
| `TF_GLM_Q4_TILE` | 0016 | `table` | — | tile das matmuls Q4 de decode | = |
| `TF_GLM_EXL3_DEC` / `_FUSE` / `_XROW` | 0016 | 1 / 1 / 1 | — | kernel de decode EXL3 com epílogo fundido | = |
| `TF_GLM_KDA_WIDE_ROWS` | 0016 | 1 | — | KDA de 3 kernels em toda janela | = |
| `TF_GLM_OVERLAP_PRIORITY` | 0017 → 0022 | -1 → 0 | — | prioridade da stream lateral | = |
| `TF_GLM_COMBINE_SY` | 0019 | 1 | — | combine sem cópia no decode | = |
| `TF_GLM_HC_DEC` | 0019 | 1 | — | `hc_partial` CUDA no decode | = |
| `TF_GLM_LATENT_STAGES` | 0019 | 3 | — | estágios dos programas de atenção latente | = |
| `TF_GLM_EXL3_ORDER` | 0020 | 1 | — | intercalar itens leves/pesados | = |
| `TF_GLM_DFLASH_POLICY` | 0021 | `fc5:0.3` | `fnc7:0.3` (`DRAFT_POLICY`) | política padrão do DFlash2 | = |

---

## 5. O que acrescentar em cada capítulo do Latent Atlas

Consolidação por capítulo (ids de `docs/curriculum.json`). Sugestão geral: criar em `receita-glm` uma seção "Variante
TensorFold" que linka cada item abaixo, e em `vllm-arquitetura` um quadro "vLLM vs um motor sob medida" com os
invariantes de §0.2.

| Capítulo | Acréscimos (patch) |
|---|---|
| `numeros-no-computador` | não associatividade: "uma cadeia fp32 por linha" vs split-K (0004); árvore de redução define bits (0019); ulp(320) ≈ 3·10⁻⁵, somar a partir da referência, FMA vs mul+add, empates em bf16 (0014); conv vs matmul "equivalente" ≈ 3% após 24 blocos (0003); overflow de `exp` em fp32 (`e^88,7`) (0012) |
| `geracao-amostragem` | amostragem chaveada por (seed, posição, token) e Gumbel-max (0011, 0018); `reasoning_effort` `max` e mapeamentos (0023) |
| `mixture-of-experts` | experts "leves" (1 item, DRAM) vs "pesados" (vários itens, L2) no prefill (0020); slots top-8 + compartilhado e `combine_shared` (0009, 0019) |
| `mla` | absorb/expand como GEMM em tensor cores no prefill (0004); `kv_b` FP8/Q4 (`TF_GLM_KVB`) e kernels por linha independentes de R (0013) |
| `atencao-esparsa` | limiar denso 2051, pontuação de pools, lista de 2051 tokens, atenção esparsa em uma passada (0009); seleção em blocos de 512 linhas e memória a 1M (0024) |
| `atencao-linear` | forma chunked WY/UT com as equações exatas e 5 kernels (0012); estabilidade numérica e paridade com `kda.cu` (0014); KDA de decode em 3 kernels (0016); várias sequências por tabela de segmentos e rollback por replay (0026); snapshots só em pontos alinhados (0008) |
| `hiperconexoes` | custo da cola mHC por linha (24 produtos × 16.384 + Sinkhorn); `hc_pre` em tensor cores no prefill (0004), em CUDA com 48 blocos no decode (0019); dividir a cola entre ranks (0010) |
| `multimodal` | torre GLM-5.3 (24 blocos, 28 px por token, pares de quadros com tempo textual, NoPE dispensa mRoPE), torre num rank só + all-gather por chunk (0003) |
| `quantizacao-fundamentos` | FP8 por linha × bloco de 128 (0002); q4 afim com recorte buscado por MSE, 4,5 bpw, truque `b·Σx` (0005) |
| `fp8-fp4-microescala` | FP8 weight-only com escala fp32 por bloco 1×128 e kernel Triton (0002); `AbsorbF8` (0013) |
| `hadamard-trelica-exl3` | anatomia do kernel EXL3 do TensorFold: `decode_tile`/`mma16816`, `rot_in`, FWHT-128 nos epílogos (0001, 0004, 0016); `suh` compartilhado por todos os experts do checkpoint GLM (0009); `decode_tile_p` (0020) |
| `medindo-qualidade` | issue #18 (q4 perde fim de turno em francês: 12/48 vs 1/48) e GSM8K/HumanEval (0005); checks de exatidão "drafted == serial", "concurrency 22/22" (§0.2) |
| `gpu-por-dentro` | ordem de despacho de blocos e L2 (0001); 99 KiB de shared memory por bloco no GB10 (0014); escopos de memória `.gpu`/`.sys` e fences (0006) |
| `memoria-unificada` | workspace de visão no orçamento (0003, 0004); picos transitórios da seleção a 1M (0024); região pinned registrada para RDMA (0006) |
| `kernels` | ordem de lançamento como alavanca de L2 e prefetch em registradores (0001); estudo de caso Yᵀ=WᵀXᵀ com `cp.async`/`ldmatrix`/epílogo fundido (0004); online softmax de uma passada e prefetch de gathers (0009); epílogo fundido + "último bloco", tabela de tiles, clusters (0016); reproduzir um kernel Triton em CUDA bit a bit (0019); intercalar trabalho por gargalo, scan por warp (0020); `input_precision` (0012) |
| `cuda-graphs-warmup` | coletiva customizada com época no device (0006); capturar 1–8 linhas e rodar 9–16 em eager, eventos em vez de `synchronize`, cópia pinned única, `_Late` (0013); reservar scratch antes da captura (0016); prioridade de streams (0017/0022); tabela de segmentos reescrita in place (0026) |
| `vllm-arquitetura` | quadro comparativo vLLM vs TensorFold: rank 1 seguidor, header + campainha, all-gather + soma em ordem de rank, configuração idêntica verificada no boot, extensões JIT versionadas (§0.2) |
| `scheduler-batching` | `TF_GLM_PREFILL_ROWS` como tamanho de chunk (0004); várias sequências numa janela de verificação (0026); entrega de tokens uma rodada depois (0013) |
| `prefix-caching` | estados em grade de G tokens e "fresco == retomado" (0008); pontos de divergência e fim do bloco de sistema (0015); snapshot do drafter em anel (0025); `TF_GLM_CACHE_ENTRIES` 8 → 32 (#17) |
| `api-tools-estruturado` | partes `video_url`, data URLs vs https, limites (0003); `reasoning_effort: "max"` (0023) |
| `especulacao-fundamentos` | verificação contra a amostra chaveada (não rejeição) e rascunho == serial (§0.2); prompt lookup (0007); janelas de 16 linhas (0013); k ótimo por E[tokens]/custo (0018); rollback por replay (0026); avaliação contrafactual offline (0011) |
| `dflash-dspark` | hierarquia cópia → DFlash2 (0007); dump + simulador, tabela `VERIFY_MS`/`BLOCK_MS` (0011); regras `fnc`/`fcost`, β = 2,0, ganhos (0018); política padrão por variável (0021); janela do drafter salva com snapshots e anel de 2.176 linhas (0015, 0025); DFlash2 em cópias de 4 bits só de rascunho (0011) |
| `paralelismos` | "sequence parallel" da cola mHC: reduce-scatter + all-gather em vez de gather + cola redundante, conta de bytes (0010); torre visual num rank só (0003) |
| `nccl-rdma` | all-gather one-shot por RDMA WRITE com proxy, flags, *doorbell*, *multi-rail* e números (0006); `ncclSend/ncclRecv` em grupo (0010) |
| `engenharia-de-receita` | patches como diffs de `site-packages` aplicados com `patch -p0` em ordem; nomes de extensão versionados; padrões no motor conservadores (off) e ligados pela receita (§6); defaults revertidos com medição (0017→0022) |
| `receita-glm` | nova seção "Variante TensorFold": tabela de §0.1, o que substitui cada overlay/flag do vLLM (`GLM53_DENSE_FP8` → `DENSE`; adaptive-k → `fnc7:0.3`; APC 3584 → grade 64 + prefixo compartilhado; NCCL → RoCE one-shot) |
| `benchmarking` | "two boots each" e efeitos de 0,5–1% (0022, 0006); simulação antes da medição (0011); números por patch medidos em builds diferentes não se somam (§6) |

---

## 6. Pontos de atenção, divergências e armadilhas

1. **`kv_b` em FP8 ou BF16?** O comentário de 0005 diz "the projections in 4 bits […], the head and kv_b in FP8", e
   0005 de fato passa `kv_k/kv_v` por `dense()` (FP8). Mas 0013 faz o caminho latente usar
   `latent.from_bf16(w[krows], w[vrows], HL, kvb)` com `TF_GLM_KVB` padrão `bf16`, e `kv_k/kv_v` só existem com
   `TF_GLM_LATENT=0`. Resultado na receita: **kv_b BF16** — confirmado por "DENSE=q4 keeps kv_b in BF16 (the docs said
   FP8)" `[CHANGELOG v1.3.2]` e por `config.sh` ("kv_b stays BF16: TF_GLM_KVB").
2. **`TF_GLM_PREFILL_OVERLAP=2`** não existe em 0010 (que aceita só 0/1); o valor 2 vem de 0033 (parte B). A receita
   exporta 2 quando `SPLIT=1`.
3. **Padrões do motor vs da receita.** Os patches ligam pouca coisa por padrão; a receita liga: `COPY=1`, `SPLIT=1`
   (+ overlap 2), `KDA_CHUNKED=1` (+ grade 64), `SHARED_PREFIX=1`, `COMM=roce`, `TF_ROCE_MAX_KB=512`, `DENSE=q4`,
   `DRAFT_POLICY=fnc7:0.3`, `COPY_MAX=15`. Quem roda TensorFold sem a receita não tem esses ganhos.
4. **Reversão de padrão medida:** 0017 introduziu prioridade alta para a stream lateral; 0022 reverteu para normal
   após "two boots each" mostrarem decode 0,5–1% mais lento.
5. **Números por patch não se somam.** Os deltas citados (ex.: prosa 38,9→44,4 do q4; 44,2→46,9 do RoCE; 47,2→50,6 da
   política) foram medidos em builds/boots diferentes, com configurações intermediárias; não reconstroem os 60,4 tok/s
   da tabela final e não devem ser multiplicados. A própria receita os rotula "(two boots each)", "(one start each)".
6. **HumanEval com q4:** `config.sh` diz 95,1% para q4 e fp8 (bf16 96,3%); a tabela "Quality" do README dá 97,6% com
   FP8 KV e q4, "measured on a build with identical replies". São medições de builds diferentes; citar as duas com a
   fonte.
7. **Copy drafts:** 0007 fixou `MOST = 5` porque a janela de 6 linhas cabia nos grafos da época; `COPY_MAX = 15`
   depende de 0013 (`MAX_ROWS = 16`); janelas de 9–16 linhas rodam em eager até 0032 (parte B) capturar janelas de 16
   linhas (`COPY_CODE`).
8. **Sub-chunks do KDA:** 0012/0014 usam sub-chunks Triton de 64 linhas; 0039 (parte B) muda para 32 linhas num único
   kernel CUDA e faz a grade padrão valer 64. O texto sobre "64-row sub-chunks" de 0012 descreve um estado
   intermediário.
9. **Visão:** os limites de 0003 (`VideoLimits.max_videos = 2`, workspace 1,5 GiB, `PASS_PATCHES` 16.384) foram
   alterados por 0004 (0,75 GiB; 8.192) e 0050 (50 imagens, 4 vídeos, corpos de 96 MiB).
10. **"Same bits" é uma afirmação do código, testada em conjunto.** Os checks publicados são de ponta a ponta
    ("drafted == serial 6/6, concurrency 22/22" `[CHANGELOG v1.3]`; "11 of 11" concorrentes `[README]`); não há
    medição de bits por patch no repositório.
11. **Rascunho == serial não é "igual ao checkpoint".** Com `DENSE=q4`, `KV=fp8` e `KDA_CHUNKED=1`, rascunhado e serial
    continuam idênticos entre si, mas ambos diferem do checkpoint em bf16; a referência exata é
    `DENSE=bf16 KV=bf16 KDA_CHUNKED=0`, com janela menor (196.608 tokens com DFlash2) `[README]`.
12. **Prompts com imagem não entram no cache de prompt** (0003: `live = []`, sem `keep`), e o reuso de prefixo
    compartilhado (0015) só vale para requisições de texto com rascunho.
13. **Os ganhos de 0001 se aplicam ao caminho agrupado.** O "2.3-2.5x at 4,096 rows" do `NFIRST` foi escrito quando
    os chunks de prompt ainda passavam pelo `grouped_kernel`; a partir de 0004 o prompt padrão usa os kernels
    Yᵀ = WᵀXᵀ (plano de 64 pares), e `NFIRST`/`SEQ` ficam para planos de 16 pares com `R ≥ 64` (ver a nota na
    ficha de 0001). O README ainda credita "EXL3 expert kernels that keep a prompt chunk's rows in L2" à linha
    "Prompt" como um todo.

# DS41 — Inventário exaustivo de modificações, patches, overlay e conceitos
## DeepSeek-V4-Flash (0731 / Vision-Exp) + DSpark em 2× DGX Spark (GB10, sm_121)

> Repositório analisado: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` (clone local, somente leitura).
> Escopo deste arquivo: `docs/PATCHES.md`, `patches/**` (todos os `hotfix-*.py/.sh`, `*.patch`, `tp3/**`, `vision_exp/**`, `dsv4_tp_pad.py`), `recipe/**` (overlay Stage-C, `nvfp4/`, `official-main/`), `vllm_patch_gb10/**`, `tests/**`, `scripts/test-*.py`, `docs/CLAUDE/*.md`, `docs/GLM-NEW-REPORT.md`. Para contextualizar, também li `README.md`, `docs/ENVS.md`, `docs/DEEPSEEK_V4_FLASH_0731.md`, `docs/vllm-027-new-patches.md`, `docs/TP3.md`, `results/RESULTS-2026-08-14.md` e o bloco `command:` de `docker-compose.dspark.yml` (necessário para saber a ordem e o default de cada hotfix).
>
> Convenção de referência: `arquivo:linha` é relativo à raiz do repositório. Quando o arquivo é um patcher que edita um arquivo **dentro da imagem** (ex.: `/usr/local/lib/python3.12/dist-packages/vllm/...`), cito os dois: o patcher (no repo) e o alvo (na imagem). Números citados são os do repo (com data quando existe); quando algo é inferência minha, está marcado como **(interpretação)**; quando não pude verificar, **(não verificável no repo)**.
>
> Identificadores técnicos, nomes de env vars, flags e código ficam no original.

---

## 0. Antes de tudo: existem TRÊS "pistas" (lanes) de runtime neste repo

Isto é o ponto que mais confunde quem lê o repositório pela primeira vez, e muda completamente o significado de muitas variáveis.

| Lane | Imagem | vLLM | Como o repo modifica | Onde está o DSpark | Estado |
|---|---|---|---|---|---|
| **A — Anemll (DEFAULT)** | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf1…` (`docs/PATCHES.md:652`) | `0.25.2.dev0+g752a3a504.d20260714` (commit `752a3a504485790a2e8491cacbb35c137339ad34`), Model Runner **V2** (`vllm/v1/worker/gpu/...`) | Hotfixes aplicados **no boot do container**, editando `/usr/local/lib/python3.12/dist-packages/vllm/...` (encadeados em `docker-compose.dspark.yml:444-473`) | `v1/worker/gpu/spec_decode/dspark/speculator.py`, `model_executor/models/qwen3_dspark.py` (Markov head), `models/deepseek_v4/nvidia/dspark.py`, atenção SM120 em `models/deepseek_v4/nvidia/flashinfer_sparse.py` (**código não está no repo**, só fixtures/trechos citados pelos patchers) | Produção |
| **B — Stage-C overlay** | base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` + overlay → `vllm-dspark-runtime:mia-raf-pr1` → Stage A/B/C → `vllm-dspark-runtime:dspark-nvfp4-stage-c` (`build-dspark-vllm-runtime.sh:7-8`, `recipe/Dockerfile.dspark-runtime-overlay:5`) | vLLM do "unholy-fusion" em `/opt/env/...`, Model Runner **V1** (`v1/worker/gpu_model_runner.py`) | Arquivos **inteiros** copiados por `COPY` (`recipe/Dockerfile.dspark-runtime-overlay:10-27`) + 3 Dockerfiles de string-replace (`recipe/nvfp4/Dockerfile.stage-{a,b,c}`) | `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`, `.../dspark.py`, `models/deepseek_v4/nvidia/dspark.py`, `dspark_kernels.py` (**código completo no repo**) | Histórico/opcional ("200K/16" e "Keys mask") |
| **C — official-main** | `vllm-dspark-runtime:official-main-dspark-00eb7ce` (`recipe/official-main/Dockerfile.python-patch:1`) | vLLM "main" oficial com DSpark | `COPY` de arquivos + teste de sanidade + diff `patches/official-main-b12x-nvfp4-python.patch` | nativo do main | Experimento (porta B12X MXFP4 MoE e `nvfp4_ds_mla` para o main) |

Consequências práticas:

1. As env vars `VLLM_DSPARK_*`, `VLLM_DSV4_*`, `VLLM_USE_B12X_WO_PROJECTION`, `DSPARK_SLOT_CLAMP` etc. **só existem na lane B** (registradas em `recipe/overlay/vllm/envs.py`). Na lane A elas geram apenas o aviso `Unknown vLLM environment variable` e **não fazem nada** (`docs/ENVS.md:9-26`, `135-165`).
2. O overlay (lane B) é, didaticamente, a **implementação de referência legível** de DSpark e das correções de concorrência ("Keys patch"). A lane A (Anemll) incorporou lógica equivalente, mas com outra organização (Model Runner V2) — por isso vários hotfixes da lane A citam arquivos que não estão no repo (`qwen3_dspark.py`, `speculator.py`, `flashinfer_sparse.py`).
3. Quase todos os hotfixes `patches/*.py` modernos são **source-exact**: fixam SHA-256 do arquivo alvo, recusam aplicar em código diferente ("fail-closed"), publicam por `os.replace` atômico e se auto-verificam. Isso é um conceito de engenharia de operação importante (ver §5, conceito "patch source-exact/fail-closed").

---

## 1. Visão geral (o que esta parte da receita faz, de ponta a ponta)

### 1.1 Objetivo

Servir o `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (default; `README.md:10-16`, revisão `86f746b36186f0e567729a5c06a8c918caba82a9`) ou o `DeepSeek-V4-Flash-0731` (lane anterior, revisão `9e165c30…`), um MoE de ~157 GiB de checkpoint (FP4 nos experts, FP8 no resto), em **2× DGX Spark** (GPU GB10 = Blackwell de consumo, compute capability 12.1 / `sm_121`, 128 GB de memória unificada LPDDR5X por nó), com:

- **Tensor Parallel = 2** (um rank por nó, `--nnodes 2`, `--distributed-executor-backend mp`, RoCE/NCCL sobre ConnectX-7);
- **contexto de 1.048.576 tokens** (`--max-model-len 1048576`);
- **KV cache `nvfp4_ds_mla`** (que, como veremos, é na prática o layout FP8 de 584 B/token);
- **decodificação especulativa DSpark** (`{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}`);
- **MoE via kernels b12x** (`--moe-backend flashinfer_b12x`, W4A16);
- visão nativa (ViT + Aligner) enxertada por hotfix.

### 1.2 A arquitetura DeepSeek-V4-Flash, como implementada aqui (números)

Fontes: `docs/CLAUDE/fable5-1-report.md:449-471` (Apêndice A, lido do `config.json` na imagem), `docs/CLAUDE/item8-fp4-kv-design.md:9-24`, `patches/dsv4_tp_pad.py:6-9`, `recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py`, `patches/vision_exp/image_processor.py:290-306`.

| Componente | Valor | Onde aparece |
|---|---|---|
| Camadas decoder | 43 (0–42) | fable Apêndice A |
| `hidden_size` | 4096 | fable A; `model.py:1235` (b12x mHC exige 4096) |
| Vocabulário | 129.280 | `vision_exp/image_processor.py:302`; `tp3/apply_tp3_patch.py` (129280 % 3) |
| Cabeças de atenção (query) | 64, `head_dim` 512 = 448 NoPE + 64 RoPE | fable A; `model.py:941-950` |
| `q_lora_rank` / `o_lora_rank` | 1024 / 1024 | fable A |
| `o_groups` (grupos da projeção de saída) | 8 (8 cabeças por grupo) | `dsv4_tp_pad.py:6`, `tp3/apply_tp3_patch.py` |
| KV latente | **1 vetor de 512 dims por token**, compartilhado por todas as cabeças (K = V; `kv_lora_rank = head_dim`) | `model.py:1088-1105` (`v_head_dim=self.head_dim`, `kv_lora_rank=self.head_dim`) |
| `attn_sink` | 1 logit aprendido por cabeça, somado ao denominador do softmax | `model.py:964-970`; `dspark_kernels.py:259-262` |
| Janela deslizante (SWA) | `sliding_window = 128` em **todas** as camadas | fable A; `rope-swa-fix.py:13-16` |
| Compressão de KV por camada (`compress_ratios`) | camadas 0–1: só SWA (ratio 0 → tratado como 1); camadas 2–42 alternam **C4A** (ratio 4, 21 camadas) e **C128A** (ratio 128, 20 camadas) | fable A; `item8:22-23`; `model.py:954-960` |
| Lightning indexer (só em C4A) | `index_n_heads` 64 × `index_head_dim` 128, `index_topk` 512 | fable A; `model.py:1051-1071` |
| MoE | 256 experts roteados, top-6, +1 shared expert, `moe_intermediate_size` 2048 | fable A; `model.py:734-795` |
| Roteamento | camadas 0–2 = **hash MoE** (tabela `tid2eid[vocab, top_k]`, sem GEMM de router); demais = router `noaux_tc` com `e_score_correction_bias`; `scoring_func` default `sqrtsoftplus` | `model.py:759-780`; fable A |
| Pesos | experts FP4 (`expert_dtype fp4`, MXFP4 e2m1 + escala E8M0 por bloco de 32); demais lineares FP8 com escala por bloco 128×128 (`scale_fmt` UE8M0) | fable A; `model.py:2073-2108` |
| Hyper-connections (mHC) | `hc_mult = 4` fluxos residuais (hidden 4096 cada → `hc_dim` 16384), mistura por Sinkhorn (`hc_sinkhorn_iters`) | `model.py:1175-1222`; `mtp-buffer-50312.sh:11` (8192 × 16384 × 2 B = 256 MiB) |
| RoPE | `rope_theta` 10000 (camadas SWA), `compress_rope_theta` 160000 (camadas com compressor); YaRN fator 16 de 64K (`original_max_position_embeddings` 65536) até 1M; RoPE parcial (64 dims), `is_neox_style=False` | fable A; `docs/ENVS.md:111`; `model.py:1027-1049` |
| DSpark (drafter) | `num_nextn_predict_layers` 3 (Vision-Exp; 0731 = 1), `dspark_block_size` 5, `dspark_markov_rank` 256; 3 camadas de draft em índices virtuais 43–45 | fable A; `PATCHES.md:835-851`; `dspark.py:804-816` |
| Visão (Vision-Exp) | ViT de 32 camadas + Aligner; token `<｜deepseek_image｜>` id 129264; patch 14, downsample 3, ≤384 tokens/imagem | `hotfix-dsv4-vision-exp.py:4-7`; `image_processor.py:34-39, 290-306` |

**Consequência de memória por token** (TP=2 replica tudo nos dois ranks; `item8:26-31`): 21 camadas C4 × 584/4 = 3,07 KB; 20 camadas C128 × 584/128 = 0,09 KB; cache K do indexer 21 × 132/4 = 0,69 KB (replicado); janelas SWA são constantes por sequência (46 camadas × 128 linhas × 584 B ≈ 3,4 MB/seq — 43 do alvo + 3 do drafter). O número "2,33 M tokens em 17,04 GiB" (≈7,8 KB/token) é contabilidade do alocador somando os quatro grupos de KV, não o custo físico de cada linha.

### 1.3 Fluxo de um passo de decode (TP=2, k=6) (interpretação baseada em `fable5-1-report.md:82-116` e no código do overlay)

1. O **scheduler** (EngineCore no nó head) monta o lote: cada requisição em decode contribui com 1 token "âncora" + k tokens de rascunho a verificar (lote uniforme de `1+k = 7` linhas por requisição); prefills são fatiados (chunked prefill, teto de 1024 tokens por chunk).
2. O **modelo alvo** roda uma vez sobre todas essas linhas: para cada camada, mHC pre → atenção esparsa (SWA 128 + entradas comprimidas C4 top-512 ou C128 todas) → mHC post/pre → MoE (b12x W4A16) → mHC post. Cada `wo_b` (RowParallel) e cada MoE fazem um **all-reduce** NCCL (43+43 por passo).
3. O **rejection sampler** compara rascunhos com as distribuições do alvo; aceita o maior prefixo compatível + 1 token "bônus".
4. O **proposer DSpark** recebe as features ocultas de camadas-alvo específicas (`dspark_target_layer_ids`), grava-as no seu **ring buffer** de janela deslizante (mascarando as linhas rejeitadas), e roda o drafter: um bloco de 5 posições (âncora + ruído) processado **em paralelo** (não-causal), seguido de uma amostragem **sequencial** esquerda→direita com a **Markov head** de posto 256.
5. Os k novos rascunhos voltam ao scheduler (com `-1` como placeholders na agenda assíncrona) para o próximo passo.

### 1.4 Cadeia de boot do container (ordem real, `docker-compose.dspark.yml:444-473`)

Ordem de execução dentro do `command:` (cada `|| exit 1` torna a cadeia fail-closed):

1. Cópia do encoder do checkpoint (`encoding/encoding_dsv4.py` → `vllm/tokenizers/deepseek_v4_encoding.py`) + correção inline do mapeamento de `reasoning_effort` (`low` deixava de virar `high`) + **#21** (`hotfix-encoding-dsv4-issue21.py`) — sempre (se o encoder existir).
2. **#31** thinking budget GPU — só se `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1`.
3. **#55** truncamento de tool call — sempre.
4. **#22** `nvfp4_ds_mla` → kernel FP8 — sempre, salvo `DSPARK_SKIP_ISSUE22_HOTFIX=1`.
5. **#79** spin-wait (`busy_loop_s` 1 → 0,002) — sempre, salvo `DSPARK_SKIP_SPIN_WAIT_HOTFIX=1`.
6. **#117** SHM ring buffer (backport vLLM #45224) + `--status` — sempre, salvo `DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX=1`.
7. **Seis backports vLLM 0.27** (em ordem): `mtp-buffer-50312`, `skip-topk-49486`, `dense-prefill-indexer-48407`, `skip-empty-c128-48957`, `flashmla-workspace-50298`, `grammar-advance` (#44993) — sempre, salvo `DSPARK_SKIP_HOTFIX=1`.
8. Redação de API key no log — só se `DSPARK_API_KEYS` ou `VLLM_API_KEY` estiver definido (e aí é obrigatório).
9. Responses store limitado — só se `VLLM_ENABLE_RESPONSES_API_STORE=1`.
10. **#138** e **Codex `agent_message`** — só se os respectivos gates = 1.
11. **Vision-Exp** — sempre.
12. **#141**, **SP indexer**, **DeepGEMM sm121 alias** — só com gate = 1.
13. **empty-encoder-output**, **#27** — sempre.
14. **adaptive chunk**, **replicate Markov** — só com gate = 1.
15. **#43**, **#26 v2**, **#133**, **runtime ablation** (modo verificação se `ABLATE=0`) — sempre; **suppress-stops-in-reasoning** — sempre, salvo `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX=1`.
16. **assistant-final (#52/#120)**, **#144**, **#136+#210**, **#191** — só com gate = 1.
17. Decisão de `--async-scheduling` (`DSPARK_ASYNC_SCHEDULING`) e de `--attention-config {"use_fp4_indexer_cache":true}` (`DSPARK_ENABLE_MXFP4_INDEXER_CACHE`).
18. **DSpark block-k**, **RoPE SWA fix**, **DSpark SWA prefix**, **DSML recovery**, **MXFP4 indexer cache**, **C128A prefill cache** — só com gate = 1.
19. Plugin GB10 híbrido NVFP4 — só se `ENABLE_VLLM_GB10_PATCH=1`.
20. **TP=3 pad** — só se `TP_SIZE=3`.
21. Validação de `DRAFT_SAMPLE_METHOD`, montagem de `SPECULATIVE_CONFIG`, `DEFAULT_CHAT_TEMPLATE_KWARGS`, `LIMIT_MM_PER_PROMPT` → `exec /usr/local/bin/vllm serve ...` (`:490-529`).

> Observação importante (**inconsistência documental**): `docs/vllm-027-new-patches.md:13-14` diz que #48957 e #50298 estão "script verified, not yet applied", mas `docker-compose.dspark.yml:448` aplica os seis backports por default, e `fable5-1-report.md:465-469` confirma que todos aparecem aplicados no log vivo. O compose é a fonte de verdade.

---

## 2. Inventário de parâmetros

### 2.1 Flags do `vllm serve` que os patches/overlay pressupõem (contexto)

`docker-compose.dspark.yml:490-529`

| Flag | Valor default | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|
| `--tensor-parallel-size` | `${TP_SIZE:-2}` | Divide cabeças de atenção, grupos de `wo_a`, dim intermediária dos experts e vocabulário entre ranks | Tensor Parallelism (Megatron) | Com TP=2, `n_local_heads = 32`, `n_local_groups = 4`. TP=3 exige o patch de padding (64 heads e 8 grupos não dividem por 3). |
| `--pipeline-parallel-size` | 1 | Sem PP | Pipeline parallelism | O buffer `_mtp_hidden_buffer` só existe no último rank de PP (`model.py:1772-1780`). |
| `--kv-cache-dtype` | `nvfp4_ds_mla` | Seleciona layout de KV "DS MLA padded" | Quantização de KV; layout paginado | No DSV4 é **byte-idêntico** ao `fp8_ds_mla` (584 B/token); só muda o despacho de kernel — daí o hotfix #22. |
| `--block-size` | 256 | Tamanho do bloco de página do KV (em tokens) | PagedAttention | Para C4: 256/4 = 64 linhas × 584 B = 37.376 B por página, igual ao da SWA (64 tokens × 584 B) — o alocador híbrido exige páginas de mesmo tamanho em bytes (`item8:24`). Também é o tamanho do bloco de hash do prefix cache (256 tokens; #144). |
| `--max-model-len` | 1048576 | Teto por requisição | Contexto longo; YaRN | Teto, não reserva: o limite real é `Σ tokens vivos ≤ pool de KV` (`README.md:150-152`). |
| `--max-num-seqs` | 6 | Máximo de sequências ativas | Continuous batching | Define linhas do ring buffer do DSpark (`dspark.py:281-292`) e o tamanho de captura de CUDA graph. |
| `--max-num-batched-tokens` | 8192 | Orçamento de tokens por passo | Chunked prefill; token budget | Dimensiona buffers persistentes: `topk_indices_buffer` (`model.py:1713-1718`), `_mtp_hidden_buffer` (256 MiB), `_dspark_hidden_buffer`. |
| `--long-prefill-token-threshold` | 1024 | Teto de tokens de um chunk de prefill longo | Chunked prefill; fairness | Evita que um prefill coma o passo inteiro e mate o decode (#27/#43). `0` desliga; 2048 custou ~1,5 GB de RAM do host (medido 2026-09-02). |
| `--max-cudagraph-capture-size` | `((seqs*(k+1)+7)/8)*8` = 48 em 6×6 | Maior batch capturado em CUDA graph | CUDA graphs; captura FULL/PIECEWISE | Sem arredondar para múltiplo de 8, 42 era truncado para 40 e a forma c=6 (42 tokens) rodava eager: −12 % no agregado c=6 (medido: 122 vs 139 tok/s, `ab-results-2026-09-03.md` linha G). |
| `--gpu-memory-utilization` | `${GPU_MEMORY_UTILIZATION:-0.835}` | Fração da memória reservada ao vLLM | Orçamento de memória; memória unificada | Cada +0,01 ≈ +1,2 GiB de KV ≈ +165 K tokens em TP=2 (`fable5-1:165-166`), mas em memória unificada compete com o host. |
| `--enable-prefix-caching` | on | Reuso de blocos de KV por hash | Prefix caching (hash encadeado) | Interage com SWA (#26), com o drafter DSpark (SWA prefix fix) e com a posição da diretiva de esforço (#144). |
| `--async-scheduling` | on (`DSPARK_ASYNC_SCHEDULING=1`) | Scheduler agenda passo N+1 enquanto N executa | Agendamento assíncrono; placeholders | Cria placeholders `-1` de rascunho; relevante para #136/#191/#44993 e para o redimensionamento de `draft_token_lengths` no overlay. |
| `--enable-chunked-prefill` | on | Prefill em pedaços | Chunked prefill | Gera lotes mistos (prefill+decode) → lotes "ragged" no DSpark (Patch 2/2b). |
| `--speculative-config` | `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}` | Liga DSpark | Speculative decoding | k=6 é forçado pela regra de divisibilidade com `n_predict=3` (Vision-Exp); o bloco treinado é 5 (hotfix block-k). |
| `--moe-backend` | `flashinfer_b12x` | Backend MoE b12x | Kernels MoE, W4A16 | Mapeia para `Mxfp4MoeBackend.B12X_MXFP4` (`official-main-b12x-nvfp4-python.patch`, hunk `map_mxfp4_backend`). |
| `--attention-config` | ausente; `{"use_fp4_indexer_cache":true}` se `DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1` (`compose:459`) | Cache K do indexer em MXFP4 | Quantização do indexer | Metade dos bytes lidos por chave pontuada. |
| `--reasoning-config` | `{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}` | Marcadores de raciocínio | Modelos de raciocínio | Usado por suppress-stops, #31 (tokens únicos `<think>`/`</think>`), #191 thinking-off fallback. |
| `--tokenizer-mode deepseek_v4`, `--tool-call-parser deepseek_v4`, `--reasoning-parser deepseek_v4`, `--enable-auto-tool-choice` | — | Encoder Python do checkpoint (sem Jinja) e parsers DSML | Chat template; tool calling | Alvos dos hotfixes #21, #52, #144, #55, #191, DSML recovery. |
| `--limit-mm-per-prompt` | `{"image":8}` | Máximo de imagens | Multimodal | `image=N` é convertido em JSON (argparse da Anemll só aceita JSON). |
| `--enable-flashinfer-autotune` | on | Autotune de táticas FlashInfer | Autotuning de kernel | O overlay estende para o decode sparse-MLA SM120 (`kernel_warmup.py:506-597`). |
| `--generation-config vllm` | — | Não herda `generation_config.json` do HF | Sampling | Cliente controla a amostragem. |

### 2.2 Gates dos hotfixes (liga/desliga) — `docker-compose.dspark.yml:300-408`

Regra geral: gates `DSPARK_ENABLE_*` só ligam com o valor **exato** `1`; gates `DSPARK_SKIP_*` só desligam com `1`. Mudar qualquer gate exige **recriar** os containers (stop/start), porque `docker restart` reaproveita a camada gravável já patchada.

| Variável | Default | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `DSPARK_SKIP_HOTFIX` | 0 | compose:314, :448 | `1` pula os seis backports v0.27 (#50312, #49486+#52492, #48407, #48957, #50298, #44993) | Backports upstream | Ganhos de kernel (1,88×, ~2×) e correção de gramática; pular só para A/B (`scripts/bench-baseline-no-patches.sh`). |
| `DSPARK_SKIP_ISSUE22_HOTFIX` | 0 | compose:315, :445 | `1` pula a rota `nvfp4_ds_mla` → kernel FP8 | Despacho de kernel | Sem ele, decode cai a ~1 tok/s em 600K+ (vs ~17 tok/s). "Don't, on this recipe" (`README.md:291`). |
| `DSPARK_SKIP_SPIN_WAIT_HOTFIX` | 0 | compose:401, :446 | `1` mantém `busy_loop_s=1` | IPC por memória compartilhada; spin vs sleep | Com 1 s, 3–4 P-cores giram no clock máximo em decode (issue #79); calor/energia roubados do GPU no SoC. |
| `DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX` | 0 | compose:404, :447 | `1` pula backport vLLM #45224 | Ring buffer SHM; notificação perdida | Sem ele, um reader pode ficar estacionado para sempre após perder um notify. |
| `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX` | 0 | compose:316, :453 | `1` não aplica o patch de stop-strings em `<think>` | Detokenização incremental | Sem ele, `stop` do cliente dispara dentro do raciocínio → `content: null`. |
| `DSPARK_SUPPRESS_STOPS_IN_REASONING` | 1 (herda `VLLM_SUPPRESS_STOPS_IN_REASONING`) | compose:408 | Liga/desliga em runtime o guard (patch aplicado) | idem | `0` restaura matching stock. |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | 0 | compose:400, :444 | Habilita `thinking_token_budget` no Model Runner V2 com kernels GPU | Orçamento de raciocínio | Default off porque tráfego sem o campo pode sofrer "decode cliff" (issue #66). Sem ele, o campo dá HTTP 400. |
| `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` | 0 | compose:329, :453 | Fatia chamadas de decode sparse-MLA > 64 linhas em views de ≤64 | Kernels FlashInfer SM120; dispatch decode/prefill | Contorno de travamento estocástico; não é correção de causa raiz. |
| `DSPARK_ENABLE_SP_INDEXER` | 0 | compose:344, :453 | Indexer sequence-parallel no prefill | Paralelismo de sequência; top-k distribuído exato | Medido −4 a −8 % no TTFT 128K; custa ~1 GB de MemAvailable. |
| `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS` | 0 | compose:351, :453 | Cria headers `sm121_*` que incluem `sm120_*` | JIT de kernels; compute capability | Sem ele, qualquer miss de cache JIT do DeepGEMM no GB10 falha a compilação. Obrigatório com MXFP4 indexer. |
| `DSPARK_ENABLE_ADAPTIVE_CHUNK` | 0 | compose:342, :453 | Teto de chunk 1024 só quando há decode ativo; senão `max_num_batched_tokens` | Chunked prefill adaptativo | Rejeitado: MemAvailable do head caiu a 2,28–2,36 GB (workspace maior vive no host). |
| `DSPARK_ENABLE_REPLICATE_MARKOV` | 0 | compose:343, :453 | Replica `markov_w1/w2` (66 MB cada) → sem 12 coletivas/passo | Coletivas serializadas; replicação vs shard | Correto (aceitação igual), mas c=6 agregado caiu 148→122 e KV −0,44 GiB → OFF. |
| `DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX` | 0 | compose:321, :454 | Header de geração após turno assistant final (#52) e após `latest_reminder` que segue assistant (#120) | Chat template / encoder | Corrige loop de turnos vazios de harnesses de agente. |
| `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN` | 0 | compose:395, :455 | Move a diretiva de esforço para depois do bloco de system | Prefix caching por hash encadeado | Stock compartilha 0 blocos entre esforços; alinhado compartilha 18/18. |
| `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` | 0 | compose:354, :456 | Backports vLLM #52805 + #53046 no XGrammar | Structured output, FSM, spec decode | Corrige matcher que recebia tokens após terminar (desync de FSM). |
| `DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED` | 0 | compose:358, :457 | Contrato fail-closed para `tool_choice` named/required | Tool calling; retries | Transforma 200 "errado" em retry/500. |
| `DSPARK_ASYNC_SCHEDULING` | 1 | compose:367, :458 | `0` remove `--async-scheduling` | Agendamento assíncrono | A/B de variável única para #191 (custa throughput de decode). Outro valor → erro. |
| `DSPARK_ENABLE_DSPARK_BLOCK_K` | 0 | compose:371, :460 | Isenta `method=dspark` da regra de divisibilidade `k % n_predict` | Configuração de spec decode | Permite k=5 (o bloco treinado) em Vision-Exp; medido decode 8K greedy 56,6 vs 55,3 tok/s. |
| `DSPARK_ENABLE_ROPE_SWA_FIX` | 0 | compose:375, :461 | Camadas SWA usam RoPE puro (fator 1), não YaRN ×16 | RoPE/YaRN por tipo de camada | Port de vllm#54815; exige A/B de qualidade 128K+ antes de virar default. |
| `DSPARK_ENABLE_DSPARK_SWA_PREFIX` | 0 | compose:379, :462 | Limita hit de prefix cache para recomputar os últimos 128 tokens | Prefix caching × drafter com janela | Corrige resposta truncada em prompts repetidos (12,1 → 86,5 tok/s no json60 upstream). |
| `DSPARK_ENABLE_DSML_RECOVERY` | 0 | compose:383, :463 | Recupera `<invoke>` DSML sem o wrapper `tool_calls` | Parser de tool calls em streaming | Port de vllm#52645. |
| `DSPARK_ENABLE_MXFP4_INDEXER_CACHE` | 0 | compose:390, :459, :464 | Relaxa gate SM100 do cache FP4 do indexer + passa `--attention-config` | MXFP4; indexer | Metade dos bytes lidos; risco de drift de seleção top-k em profundidade. |
| `DSPARK_ENABLE_C128A_PREFILL_CACHE` | 0 | compose:391, :465 | Reusa a conversão de índices C128A entre camadas no mesmo forward | Metadados de atenção | Sem ganho E2E reivindicado. |
| `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT` | 0 | compose:324, :451 | Aceita item assistant `output_text` sem `type` | Responses API | Compatibilidade de cliente. |
| `DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT` | 0 | compose:326, :452 | Converte `agent_message` do Codex em `message(role=assistant)` | Responses API | Descarta `id/author/recipient` irreversivelmente. |
| `VLLM_ENABLE_RESPONSES_API_STORE` | 0 | compose:139, :450 | Liga store de Responses + aplica backport de limite | Estado do servidor; LRU | Stock nunca despeja (vazamento de memória). |
| `ENABLE_VLLM_GB10_PATCH` | 0 | compose:179, :466-470 | Instala plugin `vllm_patch_gb10` e passa `--quantization modelopt_gb10_hybrid` | NVFP4 linear híbrido | Só para checkpoints ModelOpt NVFP4 W4A4 (não é o caso do DSV4). |
| `TP_SIZE` | 2 | compose:300, :471-474 | `3` aplica `patches/tp3/apply_tp3_patch.py` | Padding de TP | Usar `./start-tp3.sh`. |
| `ABLATE` | 0 | compose:129, :429 | `1` exporta `DSV4_ABLATE_FILE` e o patcher injeta projeção de direção de recusa | Abliteration | Definido pelo launcher quando `ABLITERATED=1`. |

### 2.3 Knobs de runtime lidos pelo código patchado

| Variável | Default (alternativos) | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `DSPARK_MAX_INFLIGHT_PREFILLS` | 1 (1–3; >3 é clampado a 3; inválido → `SchedulerConfig.max_num_partial_prefills`) | compose:155; `hotfix-dsv4-issue27-...py` INIT_INJECT (`_pp_cap`, `min(_pp_cap, 3)`) | Número máximo de prefills parciais em voo ao admitir novas requisições | Admissão; fairness prefill×decode | `1` = serializado, spread de fairness 1,60–1,76×; `2` = spread 1,50–1,57× e TTFT spread 2,1–2,4× vs 4,1× (issue #217), mas widen em 4×32K. |
| `DSPARK_ISSUE43_SCHED_DIAG` | 0 | compose:144; `hotfix-dsv4-issue43-...py` (`_ISSUE43_SCHED_DIAG`) e #27 (`_dspark_inflight_diag`) | `1` loga uma linha por passo `[issue43-step N] run= prefill_toks= decode_toks= decode_skips=` e `[issue27-adm]` | Observabilidade do scheduler | Diagnóstico; custo ~0 quando 0. |
| `LONG_PREFILL_TOKEN_THRESHOLD` | 1024 | README:277 | → `--long-prefill-token-threshold` | Chunk cap | Ver §2.1. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | 4096 | compose:136; `hotfix-dsv4-issue26-hybrid-swa-min.py` docstring | Espaçamento dos "checkpoints" de cauda SWA retidos para prefix cache | Prefix cache híbrido MLA+SWA | É o que dá hits quentes em prefixos longos (v2 do #26). |
| `DSPARK_SP_INDEXER_MIN_KEYS` | 8192 (`0` desliga mesmo com patch) | compose:345; `hotfix-dsv4-sp-indexer-prefill.py` HELPER_SRC (`_SP_INDEXER_MIN_KEYS`) | Mínimo de chaves comprimidas no chunk para usar o caminho SP | Paralelismo de sequência | 8192 chaves = 32K tokens de contexto a ratio 4. |
| `DSPARK_ISSUE191_TOOLCALL_RETRIES` | 2 (0–5) | compose:359; `hotfix-vllm-issue191-toolcall-failclosed.py:302` | Regenerações antes de HTTP 500 | Fail-closed | — |
| `DSPARK_ISSUE191_TOOLCALL_MODE` | `failclosed` (`log`) | compose:360; patcher:100 | `log` só registra WARNING | Medição | — |
| `DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK` | 1 (0) | compose:364; patcher:107 | Última tentativa troca `<think>` final por `</think>` e parseia sem raciocínio | Budget de tokens × raciocínio | Causa medida da falha residual: raciocínio de 300–500 tokens esgotando `max_tokens=512`. |
| `DSPARK_RESPONSES_STORE_MAX_ENTRIES` | 256 (inteiro > 0) | compose:140; `hotfix-dsv4-responses-store.py:36-45` | Máximo de respostas terminais retidas | LRU | Limita contagem, não bytes. |
| `DSV4_ABLATE_LAMBDA` | 3.5 (finito, ≥0) | compose:130; `hotfix-dsv4-runtime-ablation.py` GLOBALS | Força λ da projeção `y ← y − λ(y·v)v` | Abliteration | λ ≥ 4 arrisca degenerar CoT longo (`docs/ENVS.md:120`). |
| `DSV4_ABLATE_LAYERS` | `10-42` | compose:131; idem | Faixa de camadas decoder que recebem a projeção | idem | Participa do carimbo do cache AOT. |
| `DSV4_ABLATE_FILE` | `/cache/huggingface/dspark-ablation/direction_r1.pt` quando `ABLATE=1` | compose:429 | Direção de 4096 dims (`broad` ou `directions[0]`) | idem | Validada (finita, norma > 0, 4096 dims). |
| `DSPARK_API_KEYS` | vazio | compose:172, :419 | Lista de chaves separadas por espaço → **um único** `--api-key k1 k2 …` | Autenticação multi-chave | Repetir `--api-key` sobrescreveria (nargs last-wins); rejeita CR/LF, `\`, tokens iniciados por `-`. Mutuamente exclusivo com `VLLM_API_KEY`. |
| `VLLM_API_KEY` | vazio | compose:161 | Uma chave (alias nativo de `--api-key`) | Autenticação | Rotas fora de `/v1`, `/v2`, `/inference` ficam sem chave (inclui `POST /invocations`!). |
| `DSPARK_I144_TOKENIZER_JSON` | não definido | `scripts/test-issue144-effort-align.py` docstring | Roda a prova de blocos com o BPE real | Teste | — |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | compose 256 (stock 512) | compose:177; overlay `envs.py:1013-1015` | Limite do tensor `[M,N]` fp32 de logits do indexer no prefill | Workspace; OOM | 512 foi "tie" (−1,5 %/−1,8 % no TTFT 32K/128K, dentro do ruído). |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | compose 1800 (stock 300) | compose:189 | Deadline do RPC `sample_tokens` | Watchdog | JIT CuTeDSL/TileLang no meio do serviço pode passar de 300 s (issue #65/#87). Não resolve #136. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | 0 | compose:184 | Mantém CUDA graphs regulares | CUDA graphs | Não setar → Anemll liga breakable graphs automaticamente: −28,6 % no c=1. |
| `VLLM_USE_B12X_MOE` | 1 | compose:191 | Liga MoE b12x | Kernels MoE | — |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` / `_MAX_M` / `_TILE_CONFIG` | 0 / 16 / "" | compose:192-194; overlay `b12x_moe.py:174-329` | Força ocupação/tile do seletor W4A16 para M ≤ MAX_M | Tiling de GEMM, ocupação de SM | Experimento de A/B; defaults preservam o seletor. |
| `B12X_W4A16_TC_DECODE` | 0 | compose:196 | Caminho "small-M packed decode" (M ≤ 8) | Kernels de decode com M pequeno | Medido −15 % no c=1 → OFF. |
| `CUTE_DSL_ARCH` | `sm_121a` | compose:204 | Alvo de compilação CuTeDSL/b12x | Arquitetura-alvo de JIT | Evita fallback mais lento. |
| `TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, `B12X_CUTE_COMPILE_CACHE_DIR` | `/cache/huggingface/{triton,tilelang,b12x-cute}-cache` | compose:210-254 | Persistem caches JIT | JIT; cold start | Sem isso, re-JIT no meio do serviço pode travar o peer de TP além do watchdog NCCL (600 s). |

### 2.4 Variáveis do overlay Stage-C (`recipe/overlay/vllm/envs.py` + leitores) — **só lane B**

| Variável | Default | Onde (registro / leitor) | O que faz | Conceito | Trade-off |
|---|---|---|---|---|---|
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | 0 (Stage-C override: 1) | envs.py:1519-1522; `dspark_proposer.py:300-302, 834-851` | Mascara no GPU as linhas rejeitadas ao gravar o ring buffer do drafter, sem sincronizar contagens de rejeição para a CPU; **é o caminho ragged (Patch 2/2b)** | Spec decode; sincronização host-device | Obrigatório para `max_num_seqs>1` correto com chegadas escalonadas ("Run with …=1", `PATCHES.md:107-109`). |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | "0.0" | envs.py:1486-1488; `dspark_proposer.py:176-190` | Limiar de sobrevivência cumulativa para cortar o prefixo verificado | Verificação agendada por confiança | 0 = desligado (verifica k inteiros). |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | `auto` (`off`/`threshold`/`hardware`) | envs.py:1489-1491; proposer:193-204 | Estratégia de escolha do comprimento verificado | idem | `auto` = `threshold` se limiar > 0, senão `off`. |
| `VLLM_DSPARK_SPS_CURVE` | "" | envs.py:1492; proposer:226-257, 264-275 | Tabela `batch_tokens:steps_per_second,...` do motor | Modelo de custo hardware-aware | Sem curva, taxa constante 1,0. |
| `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP` | "1" | envs.py:1493-1495; proposer:260-262 | Para a varredura gulosa na primeira queda de tok/s | Heurística gulosa | `0` = varredura exaustiva para curvas com "penhascos". |
| `VLLM_DSPARK_FORCE_DRAFT_LENGTH` | "" | envs.py:1496-1498; proposer:207-223 | Força comprimento verificado (profiling) | Profiling | Valores em `[0, k]`. |
| `VLLM_DSPARK_EXPORT_DRAFT_PROBS` | 0 | envs.py:1499-1502; proposer:285-287, 733-757 | Exporta softmax dos logits de rascunho (só greedy) | Rejection sampling com q(x) | Custa um softmax por passo. |
| `VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS` | 0 | envs.py:1503-1506; proposer:290-292 | Copia confianças para CPU a cada passo | Diagnóstico | Custa sync. |
| `VLLM_DSPARK_POSITION0_DIAGNOSTICS` | 0 | envs.py:1507-1510; `gpu_model_runner.py:861-874, 3586-3651` | Mede taxa de acerto do 1º rascunho vs argmax do alvo | Diagnóstico de aceitação | Loga a cada 64 amostras. |
| `VLLM_DSPARK_LOCAL_ARGMAX` | 0 (Stage-C: 1) | envs.py:1511-1514; `dspark.py:800, 927-971` | Argmax vocab-parallel local (evita all-gather dos logits do draft) | Vocab parallelism; top-1 distribuído | "Experimental, may add per-position synchronization". |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | 0 | envs.py:1515-1518; `dspark.py:801-803, 947-954` | Top-1 fundido de `base + W2·W1[tok]` sem materializar logits Markov | Fusão de kernel | Só no caminho greedy sem confiança. |
| `VLLM_DSPARK_REPLICATE_MARKOV_W1` | 0 (Stage-C: 1) | envs.py:1527-1530; `dspark.py:547-565` | `markov_w1` vira `nn.Embedding` replicado (sem all-reduce do embedding) | Replicação | Equivalente Anemll = `hotfix-dsv4-replicate-markov-head.py`. |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | 0 | envs.py:1523-1526; `dspark.py:213-215, 312-319`; `dspark_kernels.py:604-674` | Simula FP8 quant/dequant (UE8M0, grupo 64) nas dims NoPE do KV do drafter, como o QAT de referência | Paridade com treino QAT | Útil para checar numerics (item8 §4). |
| `VLLM_DSPARK_STAGE_TIMING` / `_LOG_EVERY` | 0 / 20 | envs.py:1531-1537; proposer:305-367 | Cronometra estágios `context_prepare, prefill_main, graph_prepare, draft, postprocess, total` | Profiling | Sincroniza CUDA (não usar em benchmark). |
| `VLLM_DSPARK_ITER_TIMING` / `_LOG_EVERY` | 0 / 20 | envs.py:1538-1544; `gpu_model_runner.py:441-466, 4095-4140` | Cronometra a iteração inteira do runner | Profiling | idem. |
| `VLLM_DSPARK_TARGET_TIMING` / `_LOG_EVERY` | 0 / 20 | envs.py:1545-1551; `model.py:105-176` | Cronometra estágios do forward do alvo (mHC, attn, ffn, dspark_capture, hc_head) | Profiling | idem. |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | 0 | envs.py:1552-1555; `sm120.py:76-77, 118-169, 342-369` | Decode sparse-MLA via `b12x.attention.mla.compressed_api.compressed_mla_decode_forward` (`backend="sm120_unified"`) | Kernel alternativo de atenção | Experimento. |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE` / `_EXACT` | 0 / 0 | envs.py:1556-1563; `model.py:99-104, 1260-1308, 1804-1835` | Captura a feature da camada-alvo na camada seguinte, sem quebrar a fusão post/pre do mHC | Fusão de kernel | `EXACT` usa `dspark_hc_post_mean` (exato); senão `residual.mean(dim=1)` (aproximação da mesma quantidade após o post fundido) (interpretação). |
| `VLLM_USE_B12X_MHC` | 0 | envs.py:1023; `model.py:224-241, 1135-1255` | mHC pre/post via b12x para ≤ `B12X_MHC_MAX_TOKENS` | Kernels mHC | Exige SM120, `hc_mult == MHC_MULT`, hidden 4096, split_k fixo. |
| `VLLM_USE_B12X_WO_PROJECTION` | 0 (Stage-C: 1) | envs.py:1031-1034; `model.py:1001-1022, 2066-2070` | Projeção `wo_a`/`wo_b` fundida no b12x (mantém escalas UE8M0) | Fusão GEMM FP8 | Requer `weight_scale_inv` FP8. |
| `VLLM_USE_B12X_SPARSE_INDEXER` | 0 | envs.py:1018-1021 | Indexer C4 e top-k via b12x | Kernels | Opt-in "brought over one at a time". |
| `VLLM_USE_B12X_FP8_GEMM` | 0 | envs.py:1026-1029 | GEMM FP8 com escala por bloco via b12x | Kernels | "Do not set … DeepGEMM assert in testing" (`RESULTS-2026-08-14.md` Keys C12). |
| `VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP` | 1 | envs.py:1447-1449; `kernel_warmup.py:487-503, 600-672` | Pré-JIT dos kernels de preparo e atenção | Warmup; JIT | "Close a cold-JIT IMA window on SM12x". |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | None | envs.py:1706-1708; `kernel_warmup.py:680-697` | Diretório do cache de autotune | Autotune | Default: `VLLM_CACHE_ROOT/flashinfer_autotune_cache/<ws>/<hash>/autotune_configs.json`. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | "0" → False | envs.py:738-740 | idem §2.3 | CUDA graphs | No overlay o default já é 0. |
| `DSPARK_SLOT_CLAMP` | "1" | `dspark.py:66-70` (lido do `os.environ`, não é `VLLM_*`) | Clampa `slot_index` fora da faixa antes do gather do ring buffer | Robustez; IMA | `0` = só detecta e loga (`DSPARK_STALE_SLOT_INDEX`). |
| `B12X_MHC_MAX_TOKENS` | "16" | `model.py:234-241` | Teto de tokens para usar b12x mHC (≤0 = sempre) | Seleção de kernel por M | Acima usa TileLang mHC. |
| `B12X_TRACE_MHC` / `VLLM_TRACE_B12X_MHC`, `_LIMIT` | off, 240 | `model.py:80-87, 179-221` | Log por chamada de mHC b12x | Trace | — |
| `VLLM_ATTENTION_BACKEND=B12X_MLA_SPARSE` | não definido | `v1/attention/backends/registry.py:82-87`; `b12x_mla_sparse.py:14-17` | Seleciona backend b12x para modelos **V3.2-family** (656 B/token) | Backends de atenção | Não é o backend do DSV4 (DSV4 usa `DSV4_SPARSE_MLA_SM120`). |

### 2.5 Outras variáveis/knobs no escopo

| Variável | Default | Onde | O que faz |
|---|---|---|---|
| `VLLM_DSV4_TP_PAD` | "1" | `patches/tp3/apply_tp3_patch.py` (PAD_MODULE `dsv4_tp_pad_enabled`); `patches/dsv4_tp_pad.py:78-79` | Liga o padding de cabeças/grupos para TP não divisível. |
| `GB10_HYBRID_NVFP4_M_THRESHOLD` | 128 | `vllm_patch_gb10/vllm_gb10_hybrid_nvfp4/kernel.py:36-37` | M < limiar → Marlin W4A16; M ≥ → CUTLASS W4A4 ("try 64, 128, 256, 512"). |
| `GB10_HYBRID_NVFP4_ALLOW_NON_GB10` | 0 | `kernel.py:38` | Permite rodar fora de cc 12.1. |
| `VLLM_PLUGINS` | `gb10_hybrid_nvfp4` quando o patch está ligado | compose:468; `vllm_patch_gb10/pyproject.toml` entry point `vllm.general_plugins` | Carrega o plugin. |
| `VLLM_ROOT` | `/usr/local/lib/python3.12/dist-packages/vllm` | todos os `.sh` | Raiz do vLLM alvo. |
| `DEEP_GEMM_IMPLS_DIR` | `.../vllm/third_party/deep_gemm/include/deep_gemm/impls` | `hotfix-deepgemm-sm121-mqa-header-alias.sh:22` | Onde escrever os aliases. |
| `CONTAINER`, `API_URL`, `RESULTS_DIR` | `deepseek-v4-flash-vllm-dspark-1`, `http://127.0.0.1:8888`, `../results` | backports v0.27 `.sh` | Modos host-side `--before/--after` (snapshot de KV + histograma de prompts via `/metrics`). |

### 2.6 Campos do `config.json` que o código lê (e o que significam)

| Campo | Valor (quando conhecido) | Lido em | Uso |
|---|---|---|---|
| `dspark_block_size` | 5 | `dspark.py:210, 630, 797`; `speculative.py:321-335` (overlay: vira `n_predict`) | Largura do bloco de rascunho paralelo. |
| `dspark_noise_token_id` | (não está no repo) | `dspark.py:798`; `v1/spec_decode/dspark.py:105-109` | Token de "ruído" nas posições 1..k−1 do bloco; também preenche padding e o passo de aquecimento. |
| `dspark_target_layer_ids` | (valores não estão no repo) | `model.py:1782-1802`; `dspark.py:636, 805-807`; `v1/spec_decode/dspark.py:111-129` | Camadas do alvo cujas saídas alimentam o drafter; `main_proj` recebe `hidden × len(ids)`. Devem ser estritamente crescentes, em `{-1} ∪ [0, L−1]`. |
| `dspark_markov_rank` | 256 | `dspark.py:548-570, 596` | Posto da Markov head e entrada da confidence head. |
| `dspark_num_draft_layers` | overlay: `len(dspark_target_layer_ids)` (≥1) | `speculative.py:324-333`; `dspark.py:687, 799` | Nº de estágios do drafter (checkpoint: `mtp.0..N-1`). `infer_dspark_num_draft_layers` reconstrói a partir dos nomes de pesos porque o HF config pode dizer `num_nextn_predict_layers=1` (`v1/spec_decode/dspark.py:450-479`). |
| `num_nextn_predict_layers` | 3 (Vision-Exp), 1 (0731) | Anemll `hf_config_override` (fixture `speculative-752a3a504-stock.py:338-341`) | Na lane A vira `n_predict` → regra de divisibilidade força k=6. |
| `compress_ratios[i]` | 0 (SWA) / 4 / 128 | `model.py:957-960` | 0 é elevado a 1 (`max(1, ...)`); camadas além de `num_hidden_layers` (drafter) = 1. |
| `index_topk` | 512 | `model.py:1713-1718` | Largura do buffer de índices top-k (compartilhado por todas as camadas C4). |
| `hc_mult`, `hc_sinkhorn_iters`, `hc_eps` | 4, (?), (?) | `model.py:1175-1178` | Hyper-connections. |
| `num_hash_layers` | 3 (camadas 0–2) | `model.py:761` | Camadas de roteamento por hash de token. |
| `topk_method` | `noaux_tc` | `model.py:776-780` | Viés de correção sem loss auxiliar. |
| `scoring_func` | `sqrtsoftplus` (default no código) | `model.py:739` | Função de score do router. |
| `swiglu_limit` | (valor não está no repo) | `model.py:737, 298-301` | Clamp da ativação SwiGLU (`SiluAndMulWithClamp`). |
| `routed_scaling_factor`, `norm_topk_prob` | — | `model.py:731, 738` | Escala dos pesos de experts e renormalização top-k. |
| `quantization_config["scale_fmt"]` | UE8M0 | `model.py:1025` | Formato de escala FP8. |
| `expert_dtype` | `fp4` | `model.py:744, 2073-2108` | Seleciona mapeamento de nomes de escala (`weight_scale` vs `weight_scale_inv`). |
| `rope_scaling` (Vision-Exp ablit) | `{type: yarn, factor: 16, original_max_position_embeddings: 65536}` | `rope-swa-fix.py:12-16` | Plano (não aninhado); checkpoints mais novos podem vir aninhados `{"main","compress"}`. |
| `vision_*` | defaults no código: patch 14, downsample 3, max_n_token 384, min_pixels 147456, max_wh_ratio 8, n_layers 0 (desliga), dim 1024, heads 16, inter_dim 2816, rope_theta 10000 | `vision_exp/image_processor.py:290-306` | Os valores reais do checkpoint não estão no repo (só "32-layer ViT" no docstring). |

### 2.7 Constantes "hard-coded" relevantes

| Constante | Valor | Onde | Significado |
|---|---|---|---|
| Layout de KV DSV4 | 448 B NoPE FP8 e4m3 + 128 B RoPE bf16 (64×2) + 8 B de escalas (7 UE8M0 + 1 pad) = **584 B/token** | `cache_utils.py:23-51, 170-175`; `official-main...patch` (`get_kv_cache_shape` → 584) | Dentro de uma página: primeiro todos os dados (576 B × tokens), depois todas as escalas (8 B × tokens). |
| Layout V3.2 `fp8_ds_mla` | 656 B/token (512 FP8 NoPE + 16 B de 4 escalas fp32 + 128 B RoPE bf16) | `b12x_mla_sparse.py:150-162` | Para contraste: família V3.2/GLM-5.1/Kimi K2.5, head 576, bloco 64. |
| Probe NVFP4 (abandonado) | 416 B/token | `recipe/nvfp4/Dockerfile.stage-a` (`storage_block_size * 416`), `stage-b` | Layout NVFP4 do GLM-5.2 testado e abandonado em favor do envelope de 584 B (Stage C). |
| `FP8_MAX` | 448,0 | `cache_utils.py:174` | Máximo de e4m3. |
| Quant block KV | 64 | `cache_utils.py:173` | Uma escala UE8M0 a cada 64 dims NoPE (7 blocos). |
| `_SPARSE_PREFILL_TOPK_ALIGNMENT` | 128 | `cache_utils.py:507` | FlashMLA prefill exige `topk % B_TOPK == 0` (64 p/ h_q=64, 128 p/ h_q=128). |
| `_DECODE_MAX_TOKENS` / `_DECODE_SPLIT_TILE` | 64 / 64 | `sm120.py:44-45`; FlashInfer `_sparse_mla_sm120.py` (lock no #141) | Até 64 linhas → kernel de decode dedicado; acima → orquestrador de prefill. Split-K = `cdiv(topk,64) + cdiv(extra_topk,64)`. |
| `_C128A_TOPK_ALIGNMENT` | 128 | `sm120.py:46, 61-68` | Máximo de entradas C128 = `ceil(ceil(max_model_len/128)/128)*128` (= 8192 a 1M). |
| Cabeças q padded (SM120) | 16/32/64/128 | `sm120.py:196-209` | O kernel só aceita esses h_q; o `attn_sink` padded fica em −inf. |
| `SHM_READER_RECHECK_INTERVAL_MS` | 5000 | `hotfix-vllm-issue117-shm-ring-buffer.py` NEW_CONSTANT | Teto de espera de um reader SHM. |
| `busy_loop_s` | 1 → 0,002 | `hotfix-gb10-spin-wait.sh:22-23` | Janela de spin antes de dormir no socket. |
| Kernels DSpark | `_DSPARK_SCORE_K_BLOCK=8`, `D_BLOCK` 64 (scores)/32 (out), `_DSPARK_MARKOV_V_BLOCK=256`, `_R_BLOCK=32`, `_DSPARK_HC_POST_H_BLOCK=64` | `dspark_kernels.py:10-15, 744-748` | Tiling Triton. `head_dim % 64 == 0` exigido. |
| Warmups | slot mapping: 1..16, 32, 64, 128, 256, 512; route-pack: `q_len`, 12/16/20/32, 411/415/416/512/513/1024; decode autotune seq_lens 512 e 2048; prefill 8192 e "segundo chunk" com `profile_seq_lens=2×` | `kernel_warmup.py:39-65, 147-199, 600-672` | Formas pré-JITadas. |
| `IMAGE_TOKEN_ID` / `COMPRESS_PAD_TO` | 129264 / 4 | `image_processor.py:34-39` | Id do placeholder de imagem; bloco de imagem começa em fronteira de 4 tokens. |
| Markov head | 129.280 × 256 bf16 = 66 MB cada matriz | `hotfix-dsv4-replicate-markov-head.py:7-8` | Custo de replicar. |
| `_mtp_hidden_buffer` | 8192 × 16384 × 2 B = 256 MiB/rank | `hotfix-dsv4-mtp-buffer-50312.sh:9-12` | Buffer inútil para DSpark (usa `_dspark_hidden_buffer`). |


---

## 3. Inventário de modificações

Formato de cada item: **nome** · arquivos · default · problema · mecanismo técnico · conceitos · efeito medido.

### 3.A — KV cache, atenção esparsa e kernels

#### A1. Issue #22 — `nvfp4_ds_mla` roteado para o kernel FP8 rápido
- **Arquivos:** `patches/hotfix-nvfp4-ds-mla-issue22.sh`, `patches/fix-nvfp4-ds-mla-long-context.patch`; alvo `vllm/v1/attention/backends/mla/flashmla_sparse.py` (≈linha 880 no stock).
- **Default:** ON (skip com `DSPARK_SKIP_ISSUE22_HOTFIX=1`).
- **Problema:** `use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"` → para `nvfp4_ds_mla` dava `False` e caía no `_forward_bf16_kv` (lento). A 600K+ de contexto, decode ~1 tok/s vs ~17 tok/s com `fp8_ds_mla`; a 66 tok/s em contexto curto não mudava (`PATCHES.md:169-187`).
- **Mecanismo:** substitui por `self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")`. Os bytes do cache são idênticos (584 B) para os dois dtypes no DSV4; só o despacho diferia.
- **Conceitos:** layout de KV paginado; despacho de kernel por dtype; quantização FP8 de KV.
- **Efeito:** ~16× no decode a 600K+ (`fix-nvfp4-ds-mla-long-context.patch:15-16`).

#### A2. Stage A/B/C — "probe" de KV NVFP4 para DeepSeek V4 (overlay Stage-C)
- **Arquivos:** `recipe/nvfp4/Dockerfile.stage-a`, `Dockerfile.stage-b`, `Dockerfile.stage-c` (aplicados em sequência por `build-dspark-vllm-runtime.sh:76-90`).
- **Problema:** vLLM não conhecia `nvfp4_ds_mla`; a meta era testar KV de 4 bits no MLA esparso do DSV4.
- **Mecanismo:**
  - *Stage A* (plumbing): adiciona `"nvfp4_ds_mla"` a `CacheDType` (`config/cache.py`), ao mapa de dtype (`torch.uint8`, `utils/torch_utils.py`), ao `get_kv_quant_mode` (→ `KVQuantMode.NVFP4`) e ao tamanho de página (`storage_block_size * 416`, `v1/kv_cache_interface.py`).
  - *Stage B* (probe do layout do GLM-5.2): em `models/deepseek_v4/attention.py`, `head_bytes = 416` para nvfp4, permite o dtype no assert, converte `nvfp4`→`nvfp4_ds_mla` para o backend FlashMLA Sparse; em `nvidia/flashmla.py` `get_kv_cache_shape` → `(blocks, block_size, 416)`.
  - *Stage C* (o que ficou): volta para o **envelope de 584 B** ("keep DeepSeek V4's proven 584-byte cache envelope so hybrid MLA/SWA grouping can proceed"), inclusive `real_page_size` = `storage_block_size * 584` quando `model_version == "deepseek_v4"`.
- **Conceitos:** alocador híbrido de KV (páginas de mesmo tamanho em bytes entre grupos); layout de KV; NVFP4.
- **Efeito:** o "NVFP4 KV" não é 4 bits: é FP8 com nome diferente. `README.md:600-601`: "not the abandoned 416-byte 'true layout' experiment". `fable5-1-report §3.8`: "There is no 4-bit KV today… no accuracy risk… and no memory win either".

#### A3. `official-main-b12x-nvfp4-python.patch` + `recipe/official-main/Dockerfile.python-patch`
- **Arquivos alvo:** `config/cache.py`, `config/vllm.py`, `envs.py`, `fused_moe/oracle/mxfp4.py`, `models/deepseek_v4/attention.py`, `models/deepseek_v4/sparse_mla.py`, `v1/attention/backends/mla/flashmla_sparse.py`, `v1/kv_cache_interface.py`, novo `fused_moe/experts/b12x_mxfp4_moe.py` (idêntico ao overlay `b12x_moe.py`, verificado por diff).
- **Problema:** portar para o vLLM main oficial (i) o backend MoE B12X para pesos MXFP4 nativos do DSV4 e (ii) o dtype `nvfp4_ds_mla`.
- **Mecanismo:**
  - `VllmConfig`: em vez de **erro** "nvfp4 KV cache is not supported with MLA", converte `cache_dtype="nvfp4"` + MLA em `"nvfp4_ds_mla"`.
  - `_resolve_dsv4_kv_cache_dtype`: `nvfp4`/`nvfp4_ds_mla` → `("nvfp4_ds_mla", torch.uint8)`; `get_kv_cache_spec`: `alignment=584` para os dois layouts DS-MLA.
  - Oracle MXFP4: novo enum `B12X_MXFP4`; `"flashinfer_b12x"` → `[B12X_MXFP4]`; arredonda `hidden/intermediate` a múltiplos de 128; não converte pesos (o B12X prepara W4A16 no pós-load); `experts.process_weights_after_loading(layer)` no `make_mxfp4_moe_kernel`.
  - `Dockerfile.python-patch` roda um teste de sanidade Python (asserts sobre `CacheDType`, `get_kv_quant_mode`, `MLAAttentionSpec`, shape `(3,256,584)`, `B12xExperts`).
- **Conceitos:** oracle de backend MoE; quantização MXFP4; layout de KV.

#### A4. #50298 — reuso de workspace para índices combinados top-k+SWA (FlashMLA prefill)
- **Arquivo:** `patches/hotfix-dsv4-flashmla-workspace-50298.sh`; alvos `models/deepseek_v4/nvidia/flashmla.py` e `models/deepseek_v4/common/ops/cache_utils.py`.
- **Default:** ON (grupo v0.27).
- **Problema:** a cada chunk de prefill, `combine_topk_swa_indices()` fazia `torch.full(-1)` de `(num_tokens, combined_topk)` int32 + `torch.empty` de lens: alocação + kernel de preenchimento por chunk. As posições de padding nunca são lidas (o kernel respeita `topk_length`).
- **Mecanismo:** `combine_topk_swa_indices(..., out=(indices, lens))` reusa buffers e pula o pré-preenchimento; `forward_mqa` (caminho dummy/captura) reserva os dois buffers int32 junto com o workspace bf16 via `current_workspace_manager().get_simultaneous(...)`; `_forward_prefill` pede os três buffers de uma vez e fatia por chunk. `combined_topk = round_up(top_k + window_size, 128)`.
- **Conceitos:** workspace manager do vLLM; alocações durante captura de CUDA graph; índices esparsos + janela deslizante concatenados.
- **Efeito:** 1,88× nesse kernel (upstream). Nota de memória: o workspace agora inclui `max_num_batched_tokens × combined_topk × 4 B`.

#### A5. #48957 — pular lançamentos vazios do compressor C128
- **Arquivo:** `patches/hotfix-dsv4-skip-empty-c128-48957.sh`; alvo `models/deepseek_v4/compressor.py`.
- **Default:** ON (grupo v0.27).
- **Problema:** o compressor C128 (ratio 128, head_dim 512) rodava compress→RMSNorm→RoPE→quant→escrita de KV todo passo, mesmo quando nenhuma requisição cruzava uma fronteira de 128 tokens (≈127 de cada 128 passos de decode).
- **Mecanismo:** `_get_c128_boundary(metadata)` (CPU): `any(start % 128 + query_len >= 128)`; campo `CompressorMetadata.c128_boundary` só em camadas C128 (`block_size == 8`); em `Compressor.forward`, **depois** de `save_partial_states` (o estado parcial sempre é gravado), retorna cedo se `c128_boundary is False`, plataforma CUDA, `head_dim==512`, `compress_ratio==128` e `cudagraph_runtime_mode != FULL` (um grafo FULL não pode ramificar em metadado de CPU por passo).
- **Conceitos:** compressão de KV com estado parcial; CUDA graphs FULL vs PIECEWISE; metadado CPU vs GPU.
- **Efeito:** ~2× no kernel (upstream); numérica inalterada.

#### A6. #49486 + #52492 — atalho do indexer quando todos os candidatos seriam selecionados
- **Arquivo:** `patches/hotfix-dsv4-skip-topk-49486.sh`; alvo `models/deepseek_v4/attention.py` (`DeepseekV4Indexer.forward`).
- **Default:** ON (grupo v0.27).
- **Problema:** se `max_seq_len / compress_ratio ≤ topk_tokens`, o pipeline wq_b→RoPE→quant→logits QK→top-k é inútil (resultado = todos os candidatos).
- **Mecanismo:** ainda roda o compressor (para gravar o cache K), e preenche o buffer com um kernel Triton `_fill_short_context_topk_indices` (`[0..n-1, -1, -1, …]`, `n = (pos+1)//ratio`). Guarda #52492: **nunca** tomar o atalho durante captura de CUDA graph (`torch.cuda.is_current_stream_capturing()`), senão o grafo capturado "assa" o atalho e o reproduz para prefixos longos, selecionando 0..topk−1 sem pontuar (corrupção intermitente sob prefix cache + graphs).
- **Janela de disparo:** ≤ 512 × 4 = **2048 tokens**.
- **Conceitos:** Lightning indexer; top-k; perigos de captura de CUDA graph com ramificação dependente de dados.
- **Efeito:** 3,4 % E2E no TTFT (upstream).

#### A7. #48407 — pular scoring do indexer em prefill denso (Stage A, **dormente por design**)
- **Arquivo:** `patches/hotfix-dsv4-dense-prefill-indexer-48407.sh`; alvos `model_executor/layers/sparse_attn_indexer.py`, `models/deepseek_v4/sparse_mla.py`, `models/deepseek_v4/attention.py`, `models/deepseek_v32/nvidia/attention.py`.
- **Default:** aplicado, mas **nunca dispara** (`dense_mha_metadata_layer_name = ""`).
- **Problema/razão da dormência:** upstream pula o indexer quando a MLA principal manda um prefill curto para MHA densa; este fork **não tem rota densa** para prefills sparse-MLA, então top-k é sempre consumido. Ativar produziria atenção errada.
- **Mecanismo:** adiciona o parâmetro `dense_mha_metadata_layer_name` e o gate (`use_dense_mha && num_decode_tokens == 0 && !FULL && !capturing`); popula `DeepseekV4FlashMLAMetadata.num_decode_tokens` (diagnóstico); binding `""` ⇒ `_resolve_layer_name("")` falso.
- **Conceitos:** atenção densa vs esparsa em prefill curto; porte "staged" de upstream.

#### A8. #133 — desligar especialização por alinhamento no kernel Triton de top-k global
- **Arquivos:** `patches/hotfix-dsv4-issue133-triton-specialization.py` (Anemll) e o equivalente já presente no overlay `cache_utils.py:433-453`; alvo `models/deepseek_v4/common/ops/cache_utils.py`.
- **Default:** ON.
- **Problema:** prefill misto fatia `token_to_req_indices` e `is_valid_token` no mesmo offset; a especialização de alinhamento de 16 bytes do Triton gerou 3 estados de chave de ponteiro × 2 block sizes efetivos (64 e 2) = 6 entradas de cache persistente e **JIT no meio do serviço**.
- **Mecanismo:** `@triton.jit(do_not_specialize_on_alignment=["token_to_req_indices_ptr","is_valid_token_ptr"])` e marca `stride/topk/block_size` como `tl.constexpr`. Como ambos são loads escalares, a especialização não vetorizava nada.
- **Conceitos:** especialização de kernels Triton; cache de JIT; o risco "JIT no meio do serviço ⇒ peer de TP excede watchdog NCCL".
- **Teste:** `tests/test_issue133_triton_specialization.py` (driverless, conta famílias de especialização).

#### A9. #141 — fatiar decode sparse-MLA em ≤ 64 linhas (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py`; alvo `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py::DeepseekV4FlashInferSM120Attention._forward_decode` (Anemll) + locks sobre `flashinfer/mla/_core.py` e `flashinfer/mla/_sparse_mla_sm120.py` (FlashInfer rev `0472b9b3…`).
- **Problema:** morte estocástica do engine em TP=2 amostrada dentro do fallback paginado sparse-MLA do FlashInfer SM120; chamadas ≤ 64 linhas vão para o kernel de decode DSv4 dedicado; > 64 para o orquestrador de prefill (onde o stall foi amostrado). Com k=6 e 6+ requisições, um passo de verify tem 7×6 = 42 linhas; com mais slots, passa de 64.
- **Mecanismo:** se `num_decode_tokens ≤ 64`, chamada única inalterada; senão, loop `for row_start in range(0, n, 64)` com `slice` views de seis argumentos acoplados às linhas (`query`, `sparse_indices`, `out`, `swa_topk_lens`, `extra_sparse_indices`, `extra_sparse_topk_lens`); caches, workspace, sinks e escala compartilhados por identidade; sem `clone`/`cat`. Patcher trava o método inteiro por texto exato e os fragmentos "load-bearing" do FlashInfer (`_DECODE_MAX_TOKENS = 64`, predicado de dispatch, contrato `mutates_args` que exclui os caches).
- **Conceitos:** dispatch decode vs prefill por número de linhas; split-K; contorno vs causa raiz.
- **Efeito:** num par, 64 sobreviveu a 3.145.728 tokens gerados; 65 falhou no primeiro round. Transforma uma chamada de prefill de 1024 linhas em 16 chamadas sequenciais (re-medir prefill, `fable5-1 §3.11`).

#### A10. C128A prefill metadata cache (default OFF)
- **Arquivo:** `patches/hotfix-vllm-c128a-prefill-cache.py`; alvos `models/deepseek_v4/sparse_mla.py` (campo `c128a_prefill_global_topk`) e `models/deepseek_v4/nvidia/flashinfer_sparse.py` (consumidor).
- **Problema:** cada camada C128A convertia índices top-k locais → slots físicos separadamente, embora para C128A a conversão dependa só do metadado do forward (índices de C4A dependem da camada).
- **Mecanismo:** o primeiro consumidor C128A guarda a tupla `(indices, lens)` no metadado; camadas seguintes que compartilham o metadado reusam; o runner constrói metadado novo a cada forward, então não carrega IDs de bloco velhos. Recusa estados parciais/"hoist-v1".
- **Conceitos:** block table; conversão local→global; vida útil de metadados.
- **Efeito:** nenhum ganho E2E reivindicado ("reduced conversion count alone is not an end-to-end speedup claim").

#### A11. Item 6 — Lightning indexer sequence-parallel no prefill (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-sp-indexer-prefill.py`; alvo `model_executor/layers/sparse_attn_indexer.py`.
- **Problema:** `DeepseekV4Indexer` é **replicado** (`wq_b`, `weights_proj` = `ReplicatedLinear`): cada rank pontua 64 cabeças × todas as chaves comprimidas — custo O(queries × chaves), dominante em contexto longo (900K prefill ≈ 875 tok/s vs 2.500 tok/s a 2K) e duplicado nos dois GPUs. `fable5-1 §3.6`: ≈ 80 TFLOP por chunk de 1024 tokens a 900K (21 camadas).
- **Mecanismo:** para chunks com ≥ `DSPARK_SP_INDEXER_MIN_KEYS` chaves:
  1. `_sp_indexer_split`: divide o intervalo de chaves de cada requisição em `tp` fatias contíguas alinhadas à página; cada rank reúne só a sua (block table deslocada, `cu_seq_lens` local, `cp_gather_indexer_k_quant_cache`);
  2. `_sp_indexer_local_bounds`: mapeia `[ks, ke)` global de cada query para linhas locais (`searchsorted(..., right=True)` para requisições de comprimento zero);
  3. `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` locais;
  4. empacota `(score, global_id)` em `[M, topk, 2]` fp32, `all_gather` no grupo TP (`tp × index_topk × 8 B` por query), merge exato com `stable_topk_from_gathered_candidates_cutedsl` (kernel do DCP) direto no `topk_indices_buffer`.
- **Exatidão:** um token no top-k global está no top-k local do rank dono ⇒ merge dos top-k locais = top-k da linha inteira. Decisão usa metadado CPU idêntico em todos os ranks ⇒ coletiva não dessincroniza. Condições de fallback: DCP>1, TP=1, XPU, `topk ∉ {512,1024,2048}`, sem CuTeDSL.
- **Conceitos:** sequence parallelism; top-k distribuído; DCP (decode context parallel); indexer de atenção esparsa.
- **Efeito medido:** ruler-lite 8/8 a 32K e 131K; TTFT 128K 76,0/74,4 s vs 79,0/81,2 s (≈ −4 a −8 %, muito abaixo do 1,5–1,8× projetado); MemAvailable caiu ~1 GB (`ab-results-2026-09-02.md`, linha E).

#### A12. MXFP4 indexer K cache em sm_12x (default OFF)
- **Arquivo:** `patches/hotfix-vllm-mxfp4-indexer-cache.py`; alvo `v1/attention/backends/mla/indexer.py` (stock SHA `02505c6c…` → `bfb0376d…`).
- **Problema:** a imagem já traz cache K do indexer em MXFP4 (`AttentionConfig.use_fp4_indexer_cache`, insert Triton FP4, DeepGEMM `fp8_fp4_*_mqa_logits`, headers `sm120_fp4_*`), mas o builder faz `assert is_device_capability_family(100)` ("requires Blackwell datacenter").
- **Mecanismo:** acrescenta `or current_platform.is_device_capability_family(120)` ao assert; o compose adiciona `--attention-config {"use_fp4_indexer_cache":true}`. O launcher exige `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`.
- **Conceitos:** MXFP4 (e2m1 + escala UE8M0 por 32); quantização de **scores** (afeta só seleção top-k, não valores de atenção); `next_n` e "flattening" no decode paginado.
- **Efeito esperado:** a alocação continua 132 B/linha (o writer usa a primeira metade), então hoje só **reduz leituras** pela metade; o ganho de capacidade (−0,35 KB/token ≈ −9 % dos bytes físicos de KV) exige também a alocação de meia-linha. Gate de qualidade: ruler-lite 32K/131K, garble até 900K, TTFT 128K.

#### A13. DeepGEMM SM121 header alias (default OFF)
- **Arquivo:** `patches/hotfix-deepgemm-sm121-mqa-header-alias.sh`.
- **Problema:** o DeepGEMM vendorizado gera `sm121_fp8_mqa_logits<...>` e inclui `impls/sm121_fp8_mqa_logits.cuh` no GB10 (cc 12.1), mas só distribui headers `sm120_*`. Produção funciona só porque o cache JIT persistido tem cubins antigos; qualquer miss (volume novo, nova contagem de cabeças, caminho FP4) falha: `Failed to open .../sm121_fp8_mqa_logits.cuh`.
- **Mecanismo:** escreve quatro headers (`fp8_mqa_logits`, `fp8_paged_mqa_logits`, `fp4_mqa_logits`, `fp4_paged_mqa_logits`) com `#include <deep_gemm/impls/sm120_X.cuh>` + `#define sm121_X sm120_X`; SM120 e SM121 compartilham a ISA Blackwell de consumo. Idempotente, não sobrescreve arquivos que não são dele.
- **Conceitos:** JIT de kernels CUDA; compute capability 12.0 vs 12.1; cache de cubins.

#### A14. RoPE SWA fix — camadas SWA com RoPE puro (default OFF)
- **Arquivo:** `patches/hotfix-vllm-rope-swa-fix.py`; alvo `models/deepseek_v4/common/rope.py::build_deepseek_v4_rope` (stock `0074271a…` → `6452ce2e…`; fixture `scripts/fixtures/dspark-rope-swa/rope-752a3a504-stock.py`).
- **Problema:** o builder promovia o tipo de rope para `deepseek_yarn` sempre que não fosse `"default"`, **para toda camada**, e ainda mutava o dicionário compartilhado `config.rope_parameters`. No checkpoint Vision-Exp abliterado (`rope_scaling` plano, YaRN ×16), as camadas SWA 0–1 e as 3 camadas do drafter recebiam YaRN ×16 sobre uma janela de 128 tokens. Pela referência DSV4 (`inference/model.py` L481-485) e transformers#45892, YaRN só pertence às camadas com compressor (CSA/HCA).
- **Mecanismo (port de vllm#54815):** cópia por camada do dicionário (checkpoints aninhados `{"main","compress"}` escolhem por tipo); promoção YaRN exige `compress_ratio > 1`; demais camadas: `deepseek_yarn` com `factor=1.0` e `original_max_position_embeddings = max_position_embeddings` (escala identidade = RoPE puro no mesmo kernel). Camadas de compressor ficam byte-idênticas (fator 16, theta 160000).
- **Conceitos:** RoPE; YaRN (interpolação de frequências para extrapolar contexto); theta por tipo de camada; mutação de estado compartilhado.
- **Observação:** o overlay Stage-C (`model.py:1030-1049`, `dspark.py:262-279`) tem o mesmo padrão "YaRN para todos" e muta `config.rope_parameters` — não corrigido lá.
- **Gate de ativação:** A/B de qualidade 128K+ (o checkpoint pode ter "se adaptado" ao embedding servido).

#### A15. Overlay `common/ops/cache_utils.py` (Stage-C) — kernels de cache K
- **Arquivo:** `recipe/overlay/vllm/models/deepseek_v4/common/ops/cache_utils.py`.
- **Conteúdo e mudanças relevantes:**
  - `quantize_and_insert_k_kernel` (`:23-139`): quantiza K bf16 de 512 dims em UE8M0 FP8: para cada bloco de 64 dims, `scale = 2^ceil(log2(max(|x|,1e-4)/448))`, valor fp8 = `x/scale` clampado, escala gravada como `exponent + 127` (uint8); as 64 dims RoPE ficam em bf16. Layout do bloco: `[0, bs×576)` dados, `[bs×576, bs×576 + bs×8)` escalas.
  - `dequantize_and_gather_k_cache` (`:197-388`): gather + dequant para prefill (usa CuTeDSL se disponível, senão Triton; 128 workers por requisição).
  - **Correções de overflow int64** (`:69-71`, `:244-246`, `:263-271`): `block_idx * block_stride` passa de 2³¹ com ≥ 57K blocos (stride ~37K); `batch_idx * out_stride0` estoura a ≥ ~819K tokens com batch ≥ 4 — relevante exatamente para contexto de 1M.
  - `compute_global_topk_indices_and_lens` (`:391-499`): índice local → slot global via block table, conta válidos, zera padding; **checagens de limites** adicionadas: token inválido não indexa, `block_indices < block_table_stride`, `0 ≤ block_numbers < num_blocks` (entradas velhas/não alocadas viram −1 em vez de gather fora dos limites); mais o `do_not_specialize_on_alignment` (#133).
  - `combine_topk_swa_indices` (`:510-628`): concatena top-k comprimido (`min((pos+1)//ratio, topk)` válidos) + janela SWA (`min(pos+1, 128)`), com índices rebaseados para o buffer reunido (`M * batch_idx + N + ...`); largura arredondada a 128.
- **Conceitos:** UE8M0; FP8 e4m3; paginação; conversão de índices; overflow de aritmética de ponteiros; atenção híbrida (esparsa + janela).

#### A16. Overlay `nvidia/sm120.py` — implementação sparse-MLA SM120 do DSV4 (Stage-C)
- **Arquivo:** `recipe/overlay/vllm/models/deepseek_v4/nvidia/sm120.py`.
- **O que é:** contraparte SM120 (Blackwell consumo) do `DeepseekV4FlashMLASparseImpl` (Hopper/SM10x). Backend `DSV4_SPARSE_MLA_SM120` (mesma geometria do FlashMLA; muda só a impl).
- **Mecanismo:**
  - usa `layer._sparse_mla_wrapper.run(...)` (FlashInfer `BatchSparseMLAPagedAttentionWrapper`) com **cache duplo**: `kv_cache` = cache SWA + `extra_kv_cache` = cache comprimido (C4/C128) e seus índices/lens (`:371-393`, `:506-519`);
  - separa decode (primeiros `num_decode_tokens`) e prefill (resto) (`:250-273`);
  - C4A: índices locais do indexer → globais por camada (`compute_global_topk_indices_and_lens`); C128A: pré-computados no build do metadado (`c128a_global_decode_topk_indices`) (`:309-324`);
  - scratch split-K (`mid_out`, `mid_lse`) emprestado do workspace compartilhado, dimensionado por `_decode_num_splits(topk, extra_topk)` (`:94-106`); reserva no caminho dummy para crescer antes de `lock_workspace()` (`:276-286`);
  - cabeças padded para 16/32/64/128 (`:196-209`);
  - opcional `VLLM_DSV4_B12X_COMPRESSED_MLA`: `compressed_mla_decode_forward` do b12x com `B12XAttentionWorkspace` por camada (`:118-169`, `:342-369`).
- **Conceitos:** split-K / flash-decoding (partições do eixo de chaves + merge por LSE); cache duplo SWA+comprimido; workspace compartilhado; padding de cabeças.

#### A17. Overlay `v1/attention/backends/mla/b12x_mla_sparse.py` + registro (Stage-C)
- **Arquivos:** `recipe/overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py`, `recipe/overlay/vllm/v1/attention/backends/registry.py:82-87`.
- **O que é:** backend opt-in (`VLLM_ATTENTION_BACKEND=B12X_MLA_SPARSE`) para a família **V3.2** (DeepSeek V3.2, GLM-5.1, Kimi K2.5): head 576 (= 512 + 64), `fp8_ds_mla` de **656 B/token**, bloco 64, exige `index_topk`, `capability.major == 12`.
- **Mecanismo:** metadado com `req_id_per_token` e `cache_seq_lens_per_token` (causal por token, calculado no device); decode usa `sparse_mla_decode_forward` com scratch split-K emprestado (`_DECODE_SPLIT_TILE=64`, `max_rows = max_num_seqs × (1+k)`); extend/prefill usa `sparse_mla_extend_forward` em passada única; `B12XAttentionWorkspace` "nu" (não aloca arena por camada). Q chega bf16 e o kernel quantiza dentro.
- **Relevância aqui:** **não é usado pelo DSV4**; serve para comparar layouts (656 B V3.2 vs 584 B V4).

#### A18. Overlay `model_executor/warmup/kernel_warmup.py` (Stage-C)
- **Arquivo:** `recipe/overlay/vllm/model_executor/warmup/kernel_warmup.py`.
- **Problema:** JIT no primeiro request real (e mid-serve) causa latência e, em SM12x, "cold JIT can emit non-deterministic codegen that writes wrong slot_mapping → KV corruption → downstream sparse-MLA IMA" (`:56-59`).
- **Mecanismo:**
  - mHC TileLang warmup para todos os tamanhos de captura (`:700-710`);
  - sparse-MLA: lote misto (16 tokens), prefills curtos (12/16/20/32), prefill 8192 e "segundo chunk" (`profile_seq_lens = 2×`) para cobrir a especialização alternativa de `_build_prefill_chunk_metadata_kernel` (`:600-672`);
  - autotune do decode sparse-MLA SM120 via `flashinfer.sparse_mla_sm120_decode_dsv4_autotune`, incluindo formas de "uniform decode" DSpark (`num_tokens = k+1`, seq_lens 512/2048); o rank 0 tuna, faz `broadcast_object` do JSON para os demais e todos carregam as mesmas táticas (`:506-597`);
  - preparo de requisição: slot mapping (1..16, 32..512 tokens), route-pack W4A16 do b12x nas contagens exatas observadas (o chat template leva prompt de 512 a ~415 tokens), kernels de "padded spec decode" (`eagle_prepare_next_token_padded_kernel`, `eagle_prepare_inputs_padded_kernel`) e `rejection_greedy_sample_kernel` para larguras 1..k+1 e lotes {1, min(seqs,4)} (`:201-412`, `:487-503`).
- **Conceitos:** JIT/especialização; autotune coerente entre ranks; IMA (illegal memory access); formas de lote de spec decode.

### 3.B — MoE, pesos, quantização de pesos e paralelismo

#### B1. Overlay `fused_moe/b12x_moe.py` — `B12xExperts` (MoE MXFP4 nativo do DSV4 via b12x)
- **Arquivo:** `recipe/overlay/vllm/model_executor/layers/fused_moe/b12x_moe.py` (idêntico ao arquivo novo do `official-main-b12x-nvfp4-python.patch`).
- **Mecanismo:**
  - só aceita `weight_quant_dtype == "mxfp4"` e quant scheme `(kMxfp4Static, None)` (pesos FP4 estáticos, ativação sem quant) (`:437-440`, `:493-497`);
  - `source_format = "fp4_e8m0_k32"` (e2m1 + escala E8M0 por 32 no eixo K), `w13_layout = "w31"` (`[w1/gate, w3/up]`) (`:446-452`);
  - `process_weights_after_loading`: `prepare_b12x_fp4_moe_weights(..., prepare_w4a16=True, reuse_input_storage=True)` com escalas globais unitárias, e **libera** os pesos/escalas fonte (substitui por tensores vazios) para não pagar memória dupla (`:463-481`, `:595-610`);
  - `apply`: `quant_mode="w4a16"` fixo, scratch vindo do `workspace2` do modular kernel (tamanho planejado por `plan_tp_moe_scratch`), topk ids/pesos normalizados para int32/fp32 contíguos, erro se precisar copiar durante captura de CUDA graph (`:376-415`, `:693-782`);
  - restrições: SM120 family + pacote b12x; **sem EP**, sem all2all, sem EPLB; roteamento `RoutingMethodType.DeepseekV4`; ativações SiLU/SWIGLUOAI; `swiglu_limit` = `gemm1_clamp_limit`.
  - override opcional do seletor de tile W4A16 (`:243-329`).
- **Conceitos:** MXFP4; W4A16 (dequant FP4→bf16 + MMA bf16) vs W4A4/tensor cores FP4; fused MoE modular (prepare/finalize/experts); workspace; route packing por expert.
- **Limitação medida/diagnosticada:** o b12x só aceita o caminho NVFP4 de tensor cores para `source_format='modelopt_nvfp4'`; o formato do checkpoint força W4A16 → prefill MoE "compute-inefficient"; estimativa de +30–60 % no prefill com MXFP4×MXFP8 ou re-quantização (`fable5-1 §3.5`).

#### B2. #50312 — liberar o buffer MTP inútil
- **Arquivo:** `patches/hotfix-dsv4-mtp-buffer-50312.sh`; alvos `models/deepseek_v4/nvidia/model.py`, `v1/worker/gpu/model_runner.py`.
- **Default:** ON (grupo v0.27).
- **Problema:** `_mtp_hidden_buffer` (8192 × 16384 bf16 = 256 MiB/rank) + `copy_` por passo, consumido apenas por MTP/Eagle; DSpark usa features médias de camadas-alvo.
- **Mecanismo:** aloca só se `spec_config.use_eagle() or spec_config.uses_draft_model()`; pula o `copy_` se `None`; **guardas None** em dois pontos de `model_runner.py` (`pre_hc_hidden_states[: n]`) que o upstream não tem (crash `NoneType not subscriptable`).
- **Efeito:** 448 MiB (upstream) / 256 MiB por rank aqui.
- **⚠ Inconsistência encontrada:** o docstring afirma que `use_eagle()` é `False` para dspark, mas a fixture da imagem pinada `scripts/fixtures/dspark-block-k/speculative-752a3a504-stock.py:1238-1242` mostra `use_eagle()` retornando `True` para `"dspark"` (o mesmo vale no overlay, `speculative.py:1085-1086`). Se for assim, a condição continua verdadeira e o buffer **continua alocado** na lane A — o ganho de memória pode não ocorrer. (Não verificável sem o log de boot.)

#### B3. Plugin `vllm_patch_gb10` — NVFP4 linear híbrido Marlin/CUTLASS (default OFF)
- **Arquivos:** `vllm_patch_gb10/{pyproject.toml, README.md, vllm_gb10_hybrid_nvfp4/{__init__,config,kernel}.py}`.
- **O que é:** plugin `vllm.general_plugins` que registra a quantização `modelopt_gb10_hybrid` (subclasse de `ModelOptMixedPrecisionConfig`) e troca o kernel dos lineares NVFP4 por `Gb10HybridNvFp4LinearKernel`: M < `GB10_HYBRID_NVFP4_M_THRESHOLD` (128) → **Marlin W4A16** (`apply_fp4_marlin_linear`); M ≥ → **CUTLASS W4A4** (`scaled_fp4_quant` da ativação + `cutlass_scaled_fp4_mm`).
- **Custos:** mantém **duas cópias** dos pesos FP4 (shim Marlin + layout CUTLASS). Falha rápido se não for cc 12.1 ou se algum dos dois kernels não estiver disponível.
- **Relevância:** requer checkpoint ModelOpt NVFP4 W4A4 — **não é o formato do DSV4** (MXFP4 nos experts, FP8 no resto); "Tested here only for syntax packaging, not on a live GB10".
- **Conceitos:** decode bandwidth-bound (Marlin) vs prefill compute-bound (tensor cores FP4); custom op registrada (`direct_register_custom_op`).

#### B4. TP=3 — padding de cabeças/grupos (só `TP_SIZE=3`)
- **Arquivos:** `patches/tp3/apply_tp3_patch.py` (aplicado; embute o módulo R2), `patches/dsv4_tp_pad.py` (versão R1, **não** é o que a imagem recebe — o applier escreve seu próprio `PAD_MODULE`).
- **Problema:** 64 cabeças e 8 `o_groups` não dividem por 3; o stock recusava iniciar ou, pior, fazia `8 // 3 == 2` silenciosamente (descartando 6 de 8 grupos → "fluent garbage").
- **Mecanismo (R2, o aplicado):** padding do **número de grupos** 8 → 9 mantendo 8 cabeças/grupo ⇒ 72 cabeças, 24 por rank, 3 grupos por rank. Justificativa: a projeção de saída é um BMM por grupo `einsum("bhr,hdr->bhd")` cujo `r = heads_per_group × head_dim = 8 × 512 = 4096` é contrato do checkpoint (`wo_a = [8192, 4096]`); R1 (replicar grupos e fatiar cabeças dentro) daria `r = 1536` e o kernel rejeita (`einsum.hpp:164`). Edições:
  - `attention.py`: cabeças 64→72, grupos 8→9, asserts de contrato; `wo_a` com entrada `heads_per_group_local × head_dim`;
  - `config/model.py`: permite contagem padded (o bloqueio de boot real);
  - `parameter.py`: loaders column/row/merged-column "pad-aware" (copia a interseção real, zera o resto; detecta parâmetros replicados para não deslocar por `tp_rank`);
  - `fused_moe/config.py`: `moe_intermediate_size` 2048 → `lcm(3,64)=192` ⇒ 2112 (704/rank, múltiplo de 64 p/ NVFP4);
  - `model.py`: escopa o assert dos 256 experts a MegaMoE/EP;
  - `vocab_parallel_embedding.py`: `padding_size = lcm(64, tp)` ⇒ vocab 129.280 → 129.408;
  - loaders de `attn_sink` (model/mtp/dspark) com janela da contagem padded e clamp ao real.
  - Pads: pesos zero; **`attn_sink` pad = −inf** (exp(−inf)=0 → não altera o denominador; `0` somaria 1 e atenuaria as cabeças reais).
  - Marcador por edição com hash do corpo: um patch antigo na camada gravável vira `STALE` e aborta (exige `docker rm -f`).
- **Contradição documental:** `patches/dsv4_tp_pad.py` descreve R1 como "CORRECT (rel 2.3e-06)" e R2 como "impossible", enquanto `tp3/apply_tp3_patch.py` e `docs/TP3.md` dizem o oposto e aplicam R2. O arquivo raiz parece ser um resquício sincronizado pelo launcher (`start-deepseek-v4-flash-dspark.sh:1831-1833`, `ci-validate.sh:583`), mas não é importado.
- **Efeito medido (`docs/TP3.md`):** decode por stream +4–13 %, ≈ 200 tok/s agregado a 16 streams, KV ≈ 35 GiB/rank (5,0 M tokens, 4,79× a 1M); TTFT +4–13 % até 64K e +22–23 % a 128K–256K (MLA replica o KV latente e o indexer em todo rank, e o all-reduce de 3 nós custa mais). Primeiro boot ≈ 8 min (engine init 193 s).

#### B5. Runtime ablation (abliteration) — `ABLATE`
- **Arquivo:** `patches/hotfix-dsv4-runtime-ablation.py`; alvo `models/deepseek_v4/nvidia/model.py`.
- **Default:** o script sempre roda; com `ABLATE=0` só verifica que o arquivo está stock e mantém o carimbo do cache AOT.
- **Mecanismo:** injeta globais (`_DSV4_ABLATE_FILE/LAMBDA/LAYERS`), um buffer `_refusal_dir` (fp32, 4096) por camada decoder (todas as camadas têm o buffer para manter **uma** assinatura de torch.compile/AOT), e após cada `x = self.attn(positions, x, None)` aplica `x ← x − λ·(x·v)·v` (em fp32) nas camadas da faixa. Conta os sites (Anemll: 1; Stage-C: até 3) e recusa formas desconhecidas. Invalida `VLLM_CACHE_ROOT/torch_compile_cache` quando o carimbo `(versão, enabled, λ, camadas, sha da direção, sha do patch)` muda.
- **Conceitos:** abliteration (remoção de direção de recusa no residual stream); torch.compile/AOT cache e sua chave; projeção ortogonal.
- **Notas:** direção capturada no 0731 FP8 DSpark; transferência para Vision-Exp é "experimental" (`README.md:178`).

### 3.C — DSpark (speculative decoding) — implementação e correções

#### C1. Overlay `models/deepseek_v4/nvidia/dspark.py` — o drafter (Stage-C)
- **Arquivo:** `recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py` (1193 linhas).
- **Estrutura:**
  - `DeepSeekV4DSparkAttention` (`:183-540`): mesma família de projeções da atenção alvo (`fused_wqa_wkv` replicado, `q_norm`, `wq_b` column-parallel, `kv_norm`, `wo_a` BMM por grupo, `wo_b` row-parallel, `attn_sink`); **KV cache interno** `main_kv_cache[max_num_seqs, window=128, head_dim=512]` bf16 (buffer não persistente, `:281-292`) e `sparse_scores[max_num_seqs, block, n_local_heads, window+block]` fp32 (`:293-304`). Não registra camadas de KV no vLLM ("different from the normal MTP cache contract", `:3-10`).
  - `_project_kv` (`:321-331`): `kv = RoPE(kv_norm(wkv(x)))` (+ quant/dequant FP8 opcional).
  - `store_main_kv` (`:396-453`): grava as features do alvo no **ring buffer** por posição (`slot = pos % window`), mantendo os últimos `window` tokens; com `num_rejected_tokens`, mantém o valor antigo nas posições rejeitadas (`torch.where(valid_mask, novo, antigo)`).
  - `_store_main_kv_ragged` (`:333-394`): versão por requisição via `query_start_loc` (lotes mistos prefill+decode), roda eager.
  - `forward_dspark` (`:471-540`): Q normalizado por RMS **sem peso** por cabeça (`q * rsqrt(mean(q²)+eps)`, `:464`), RoPE em Q e KV, atenção `dspark_sparse_attention` sobre `[janela principal válida ; KV do bloco de rascunho]` com sink; depois `fused_inv_rope_fp8_quant` (RoPE inverso nas 64 dims + quant FP8 para o einsum) → `deepseek_v4_fp8_einsum("bhr,hdr->bhd")` com `wo_a` → `wo_b`. Receita de einsum `(1,128,128)` em SM≤9, `(1,1,128)` em SM≥10; escalas TMA-alinhadas em SM≥10 (`:216-219`).
  - `DeepSeekV4DSparkMarkovHead` (`:543-588`): `markov_w1` (vocab→256; `VocabParallelEmbedding` ou `nn.Embedding` replicado com `VLLM_DSPARK_REPLICATE_MARKOV_W1`) e `markov_w2` (`ParallelLMHead` 256→vocab). `forward(token)` = viés de logits dependente do token anterior.
  - `DeepSeekV4DSparkConfidenceHead` (`:591-611`): `Linear(hidden + 256 → 1)` fp32 sobre `[estado denso, embedding Markov]` → logit de confiança.
  - `DeepSeekV4DSparkLayer` (`:614-789`): estágio 0 tem `main_proj` (`hidden × len(target_ids) → hidden`, replicado) + `main_norm`; cada estágio = mHC pre → attn_norm → (média sobre os fluxos hc se 3D) → atenção DSpark → mHC post → mHC pre → ffn_norm → `DeepseekV4MoE` (a mesma MoE do alvo, com hash routing via ids) → mHC post; o último estágio tem `norm`, Markov head, confidence head e parâmetros `hc_head`.
  - `DeepSeekV4DSparkModel.draft` (`:868-995`): monta o bloco `[âncora, ruído, ruído, …]` (tamanho `block_size`), posições `pos_âncora + [0..block−1]` ("DeepSpec trains/evaluates DSpark with the anchor token itself at the first draft position", `:897-899`), passa pelas camadas, colapsa com `hc_head`, `norm`, logits via `lm_head` e então, **sequencialmente** para `pos = 0..block−1`: `logits[:,pos] += Markov(output_ids[:,pos])`, `output_ids[:,pos+1] = argmax(...)`. Retorna `output_ids[:,1:]` (block tokens), logits (opcional) e `sigmoid(confidence)` (opcional). Caminhos: argmax vocab-parallel local (`VLLM_DSPARK_LOCAL_ARGMAX`) e Markov-argmax fundido.
  - `load_weights` (`:1088-1192`): só nomes `mtp.<stage>.*`, renomeados para `model.layers.<num_hidden_layers + stage>.*`; `shared_experts.w2 → down_proj`; `ffn.gate.bias → e_score_correction_bias`; `attn_sink` fatiado por cabeça do rank; `.scale → .weight_scale` (experts FP4) ou `.weight_scale_inv` (FP8); escalas E8M0 vistas como uint8 (não converter numericamente!); mapeamento "stacked" segment-aware (`markov_w1` não é o `w1` da FFN) (`v1/spec_decode/dspark.py:15-35`).
- **Conceitos:** drafter paralelo em bloco (não-causal); features de camadas intermediárias do alvo; ring buffer por posição; Markov head de baixo posto; confidence head; mHC; RoPE inverso; `attn_sink`.

#### C2. Overlay `nvidia/dspark_kernels.py` — kernels Triton do drafter
- **Arquivo:** `recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark_kernels.py`.
- **Kernels:**
  - `_dspark_sparse_scores_kernel` + `_dspark_sparse_out_kernel` (`:121-298`, wrapper `:716-822`): atenção em duas passadas. Scores: para cada (batch, posição do bloco, cabeça), produto Q·K sobre `KV_TOKENS = window + block` (128 + 5 = 133 no caso default), máscara: chaves da janela válidas só até `valid_main_len = min(pos+1, window)`; **todas** as chaves do bloco de rascunho válidas (sem máscara causal dentro do bloco). Out: softmax com sink (`normalizer = max(max(scores), sink)`, `denom = Σexp + exp(sink − normalizer)`) e soma ponderada de V (= K). Referência PyTorch em `:677-713`.
  - `_dspark_markov_block_argmax_kernel` (`:18-85`, wrapper `:511-601`): top-1 local de `base_logits + markov_embed @ W2ᵀ` por blocos de 256 do vocabulário sem materializar os logits Markov; depois o chamador faz o top-1 global entre ranks.
  - `_dspark_quant_dequant_nope_kernel` (`:88-118`, wrapper `:635-674`): simulação QAT FP8 (UE8M0, grupos de 64) nas dims NoPE do KV do drafter.
  - `_dspark_hc_post_mean_kernel` (`:301-371`, wrapper `:413-490`): calcula `MHCPostOp(...).mean(dim=1)` sem materializar os 4 fluxos (para a captura diferida das features do alvo).
- **Conceitos:** atenção com sink; softmax estável (subtração do máximo); argmax distribuído vocab-parallel; fusão de kernels; QAT simulado.

#### C3. Overlay `v1/spec_decode/dspark.py` — utilitários, diagnóstico e agendamento por confiança
- **Arquivo:** `recipe/overlay/vllm/v1/spec_decode/dspark.py`.
- **Conteúdo:**
  - `map_dspark_stacked_param_name` (`:15-35`): `attn.wq_a`/`attn.wkv` → `attn.fused_wqa_wkv` shards 0/1; `shared_experts.w1/w3` → `gate_up_proj` 0/1; ignora `.experts.`.
  - `make_dspark_warmup_draft_token_ids` (`:38-67`): no primeiro `propose`, ainda não há feature de token gerado — devolve um bloco de `noise_token_id` (tensor fixo, em vez de listas vazias ou −1).
  - `unpack_mhc_pre_outputs` (`:70-80`): reordena `(post, comb, input)` → `(input, post, comb)`.
  - `DSparkModelSpec.from_hf_config` (`:83-169`): valida campos DSpark; `infer_dspark_num_draft_layers` a partir de `mtp.N.*` contíguos (`:450-479`).
  - Teoria: `speculative_acceptance_confidence(q, p) = Σ_x min(p(x), q(x)) = 1 − TV(p,q)` (`:508-539`); `cumulative_survival` (produto das confianças condicionais) (`:542-550`).
  - Agendadores: `confidence_threshold_prefix_length` — maior prefixo com sobrevivência ≥ limiar, **não-antecipativo** (compatível com rejection sampling) (`:553-572`); `score_prefix_lengths` (`:575-613`); `hardware_aware_prefix_schedule` — admite tokens de prefixo em ordem decrescente de sobrevivência entre **todas** as requisições do lote e guarda o ponto de máximo `E[tokens aceitos] × steps_per_second(batch_tokens)` (`:616-685`).
  - `DSparkDiagnostics`/`DSparkPosition0Diagnostics` (`:213-412`): histogramas de comprimento, prune rate, confiança/sobrevivência por posição, taxa de acerto da posição 0.
- **Conceitos:** probabilidade de aceitação e distância TV; comprimento de rascunho adaptativo; custo do verificador em função do tamanho do lote (hardware-aware).

#### C4. Overlay `v1/spec_decode/dspark_proposer.py` — `DSparkProposer` (Stage-C; cópia idêntica em `recipe/vllm/v1/spec_decode/`)
- **Arquivo:** `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py` (confirmado idêntico via `cmp`).
- **Mecanismo:**
  - não aloca KV de rascunho no vLLM (`initialize_attn_backend` → `block_size = 1`, `:369-376`);
  - buffers persistentes para CUDA graph: `_draft_input_ids_buffer`, `_draft_hidden_buffer` (`max_batch × hidden × len(target_ids)`), `_draft_positions_buffer`, `_draft_slot_index_buffer` (`:63-86`); o drafter é envolvido num `CUDAGraphWrapper` **PIECEWISE** (`:416-431`) e o padding de lote usa `noise_token_id`, zeros e slot 0 (`:474-503`);
  - `propose` (`:760-1042`): (i) **guard de lote não-uniforme** — se `rows % batch_size != 0`, loga uma vez e devolve zeros (desliga especulação naquele passo em vez de matar o WorkerProc) (`:788-810`); (ii) slots estáveis por requisição (Patch 1, `:812-826`); (iii) `prepare_context` com caminho retangular ou **ragged** (Patch 2/2b, `:828-943`); (iv) `prefill_main` (grava features no ring buffer, com máscara de rejeitados na GPU); (v) no primeiro uso, devolve rascunho de aquecimento; (vi) prepara buffers do grafo e roda `draft_with_confidence`; (vii) pós-processa: exporta probs (greedy), guarda confiança, decide `_last_draft_lengths` (forçado, por confiança ou k).
  - `_trim_rejected_target_context` (`:578-633`): caminho legado (CPU) que remove sufixos rejeitados — ainda exige comprimentos uniformes.
- **⚠ Observação (interpretação):** o guard de não-uniformidade (`:794-810`) roda **antes** do caminho ragged: casos como "41 rows for batch_size=2" (não divisível) agora pulam a especulação em vez de chegar ao ramo ragged; o ramo ragged só cobre lotes divisíveis porém desiguais (ex.: 6 linhas = 5+1).
- **Conceitos:** proposer em speculative decoding; buffers persistentes para replay de CUDA graph; `query_start_loc` (offsets de segmentos em lotes achatados); rejeição de sufixos.

#### C5. "Keys concurrency patch" — Patch 1, 2, 2b (`patches/keys-concurrency.patch`, `docs/PATCHES.md:1-138`)
- **Autoria:** drowzeys ("Keys"); diff contra o overlay (`dspark.py`, `dspark_proposer.py`, `gpu_model_runner.py`), já incorporado nos arquivos do overlay.
- **Patch 1 — slot de KV estável por requisição:**
  - *Sintoma:* com `max_num_seqs>1`, aceitação ~0 (rascunho lixo), sem crash.
  - *Causa:* o ring buffer do drafter era indexado pela **linha do lote** (`main_kv_cache[:batch_size]`); o continuous batching do vLLM-v1 **condensa** o lote quando uma requisição termina (outra passa a ocupar a linha), mas o buffer não se move → a requisição lê o histórico de outra.
  - *Correção:* `_req_id_to_slot` + `_free_slots` (reuso do menor slot livre primeiro, `_row_to_slot`, `dspark_proposer.py:529-549`); gather `index_select`/scatter `index_copy_` por `slot_index`; `gpu_model_runner` passa `req_ids=self.input_batch.req_ids` (`gpu_model_runner.py:5250-5270`). Se a permutação é identidade, usa o caminho in-place original byte a byte (gate é "permutação identidade", não "batch==1").
- **Patch 2 — contexto ragged:**
  - *Sintoma:* HTTP 500 `DSpark currently requires uniform flattened per-request inputs; got 41 rows for batch_size=2`.
  - *Causa:* chunked prefill mistura prefill e decode no mesmo passo; o reshape retangular `[batch, seq, H]` é impossível. Desligar chunked prefill exigiria `max_num_batched_tokens ≥ max_model_len` (inviável a 1M).
  - *Correção:* detecta `len(set(seg_lengths)) != 1`; âncora por requisição `anchor_idx = starts + clamp(len − rejected − 1, 0, len−1)`; `_store_main_kv_ragged` por `query_start_loc` (trunca cada segmento aos últimos 128, `slots = pos % window`, máscara de rejeitados). Passos mistos nunca são capturados em CUDA graph, então laços Python são seguros.
  - *Escopo:* só no caminho `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.
- **Patch 2b — detecção ragged independente de rejeição:** a detecção estava dentro de `if num_rejected_tokens_gpu is not None`; passos de prefill sem rejeição ainda caíam no reshape e davam 500 ("166 rows for batch_size=3"). Agora entra no ramo sempre que o modo GPU-mask está ligado; `rejected` default zeros.
- **Validação:** GSM8K N=8 (200 perguntas) sem erros, 93,5 % vs 95,0 % sequencial, 97,5 % de concordância por pergunta.
- **Conceitos:** continuous batching e condensação; estado por requisição fora do KV do vLLM; ragged batches; FP não-determinismo entre lotes.

#### C6. Guarda de slot (`DSPARK_SLOT_CLAMP`) — overlay
- **Arquivo:** `recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py:66-107, 372-381, 432-435, 502-505`.
- **Problema:** `slot_index` velho/fora da faixa (visto após condensação em contexto longo) disparava assert de device `indexSelectSmallIndex` e matava o worker.
- **Mecanismo:** `clamp(0, num_rows−1)` no device (seguro em captura de grafo); log `DSPARK_STALE_SLOT_INDEX` só fora de captura (precisa sync). Como é só o caminho de rascunho, um slot errado produz um rascunho ruim que o alvo rejeita — não corrompe a saída.
- **Conceitos:** rascunho não afeta corretude (a verificação garante); operações seguras dentro de CUDA graph.

#### C7. Overlay `config/speculative.py` — DSpark como método
- **Arquivo:** `recipe/overlay/vllm/config/speculative.py`.
- **Pontos:** `DSparkModelTypes = Literal["dspark"]` (`:56`); `DraftSampleMethod = "greedy" | "probabilistic"` (`:76, 262-268`: probabilístico amostra do rascunho e usa os logits completos no teste de razão, custando memória); `hf_config_override` (`:321-335`): se `model_type=="deepseek_v4"` e `dspark_block_size`, vira `deepseek_v4_dspark` com `n_predict = dspark_block_size`, `num_nextn_predict_layers = dspark_num_draft_layers = max(1, len(dspark_target_layer_ids))`, arquitetura `DeepSeekV4DSparkModel`; `compute_hash` inclui `uses_aux_hidden_states` para dspark (`:283-292`); `method in ("mtp","dspark")` usa o próprio modelo alvo como fonte do rascunho (`:570-580`); regra de divisibilidade (`:788-803`); `use_eagle()` inclui dspark (`:1085-1086`).
- **Contraste com a lane A:** a Anemll mapeia `deepseek_v4 → deepseek_mtp` com `n_predict = num_nextn_predict_layers` (fixture `speculative-752a3a504-stock.py:338-341`), arquitetura `DSparkDraftModel`, e liga `parallel_drafting=True` para dflash/dspark (`:899-911`). É daí que vem o k=6 forçado no Vision-Exp (n_predict = 3).

#### C8. DSpark block-k unlock (default OFF)
- **Arquivo:** `patches/hotfix-vllm-dspark-block-k.py`; alvo `vllm/config/speculative.py` (stock SHA `3f1abd1c…`, 56.845 B → `7fffe035…`, 56.987 B).
- **Problema:** a regra "Ensure divisibility for MTP module reuse" (`k ≤ n_predict` ou múltiplo) existe porque o MTP clássico re-executa uma camada por passo; no DSpark os estágios são **empilhados** num backbone não-causal que prevê o bloco inteiro de uma vez. Com `n_predict = 3`, k=5 (o bloco treinado) é proibido e roda k=6 (uma query de ruído a mais que o treino).
- **Mecanismo:** adiciona `self.method != "dspark"` à condição; o launcher relaxa `MTP_NUM_TOKENS` para `≥ 1`. Par recomendado: `MTP_NUM_TOKENS=5` (captura `6×(5+1)=36 → 40`).
- **Efeito (2026-09-03, Vision-Exp, k=5 vs k=6):** decode greedy 8K 56,6 vs 55,3; greedy 32K 56,6 vs 53,6; temp 0,6: 49,9–56,9 vs 48,6–53,1; mini-bench single-stream 56–64 vs 51–55; c=4 agregado igual; TTFT igual; aceitação por posição igual (0,88/0,74/0,53/0,36/0,24) ⇒ o ganho vem do passo mais barato. Antes (k=6 vs 0731 k=5): decode single-stream −15–20 % (greedy) e −20–30 % (temp 0,6).

#### C9. Replicar a Markov head na lane A (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-replicate-markov-head.py`; alvo `vllm/model_executor/models/qwen3_dspark.py` (compartilhado pelo drafter DSV4).
- **Problema:** `DSparkMarkovHead` shardado (`VocabParallelEmbedding` + `ParallelLMHead` via `LogitsProcessor`) ⇒ o laço sequencial de 6 passos faz 12 coletivas **serializadas** por passo de decode (all-reduce do embedding + all-gather do viés).
- **Mecanismo:** `markov_w1 = nn.Embedding(vocab, rank)`; `markov_w2 = ReplicatedLinear(rank, vocab)`; `bias()` faz `F.linear` local (aplica `logits_processor.scale` se ≠ 1).
- **Efeito medido:** aceitação inalterada (correto), **sem** ganho em c=1, c=6 agregado 122 vs 148, KV −0,44 GiB → OFF (`ab-results-2026-09-02.md`, linha B).

#### C10. DSpark draft SWA prefix fix (default OFF)
- **Arquivo:** `patches/hotfix-vllm-dspark-swa-prefix.py`; alvos `v1/core/kv_cache_manager.py` (whole-file pin `be9c5091…` → `09f0e990…`) e `v1/core/sched/scheduler.py` (regiões exatas; pure stock `e25d4c9a…` → `69fc8118…`).
- **Problema:** com `--enable-prefix-caching`, reenviar um prompt idêntico (retries, prefixos de tool-call, loops de agente) produzia resposta degenerada/truncada (ex.: `["json` + stop), determinística a temp 0: num hit de prefix cache o alvo pula o prefixo cacheado, só o sufixo alimenta o drafter, a janela do drafter fica vazia e o verificador aceita o truncamento.
- **Mecanismo (port de Anemll/dspark-vllm-gx10#2, `4afc5e7eeb`):** `KVCacheManager(dspark_window_size=...)`; em `get_computed_blocks`, `max_cache_hit_length = max(num_tokens − 1 − window, 0)`; o scheduler lê `draft_model_config.hf_config.sliding_window` quando `use_dspark()`. Custo: recomputar 128 tokens por hit.
- **Efeito (upstream, 2×GB10, TP=2, 0731, K6 probabilístico):** json60 repetido 12,1 → 86,5 tok/s; outros casos e prefill 8K/32K/100K inalterados.
- **Conceitos:** interação entre prefix caching e estados auxiliares fora do KV paginado.

#### C11. Overlay `scheduler.py` e `gpu_model_runner.py` / `outputs.py` — integração DSpark (Stage-C)
- `recipe/overlay/vllm/v1/outputs.py:281-284`: `ModelRunnerOutput.draft_token_lengths: dict[str,int] | None` ("Async schedulers use this to size the next placeholder list before real draft token ids are available on CPU").
- `recipe/overlay/vllm/v1/core/sched/scheduler.py:1497-1513`: em modo assíncrono, para decodes (não prefill chunks), redimensiona `request.spec_token_ids = [-1] * draft_len` com o comprimento agendado por confiança (clampado a `[0, num_spec_tokens]`). `:215-222`: `use_eagle()` (inclui dspark) ⇒ `num_lookahead_tokens = num_spec_tokens`.
- `recipe/overlay/vllm/v1/worker/gpu_model_runner.py`: instancia `DSparkProposer` (`:615-616`); pega `get_dspark_target_hidden_states` em vez de `get_mtp_target_hidden_states` (`:5178-5186`); passa `req_ids` (`:5250-5270`); coleta probs/confiança/draft lengths do drafter (`:5271-5291`); publica `draft_token_lengths` na saída (`:4726-4755`); diagnóstico de posição 0 (`:3566-3651`); timing de iteração (`:441-466`, `:4095-4140`).
- `recipe/overlay/vllm/model_executor/models/registry.py:600`: `"DeepSeekV4DSparkModel": ("vllm.models.deepseek_v4", "DeepSeekV4DSpark")`; `models/deepseek_v4/__init__.py`: exporta `DeepSeekV4DSpark` (NVIDIA) e cai para `amd.model` no ROCm.

#### C12. Overlay `nvidia/model.py` — captura das features para o DSpark e knobs b12x (Stage-C)
- **Arquivo:** `recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py`.
- **Captura de features:** `_dspark_hidden_buffer[max_num_batched_tokens, len(target_ids) × 4096]` (`:1782-1802`); no loop de camadas, ao passar por uma camada-alvo, faz `hc_post` (colapsa os 4 fluxos mHC num tensor `(T, 4, H)`), **quebra** a cadeia fundida post/pre (`residual=None`, a próxima camada faz `hc_pre` isolado) e grava `hidden_states.mean(dim=1)` na fatia daquela camada (`:1898-1917`). Alternativa `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE`: a camada seguinte grava a média durante seu `mhc_fused_post_pre` (fused: `residual.mean(dim=1)`; exact: `dspark_hc_post_mean`) sem quebrar a fusão (`:1260-1308, 1804-1835`).
- **Outros:** `get_dspark_target_hidden_states()` (`:2171-2177`); `load_weights` do alvo pula `mtp.` (`:2179-2184`); mapper HF→vLLM (`:2073-2108`: `layers.`→`model.layers.`, `head.weight`→`lm_head.weight`, `.attn.compressor.`→`.attn.mla_attn.compressor.`, `.ffn.gate.bias`→`e_score_correction_bias`); `DeepseekV4MegaMoEExperts` (DeepGEMM `fp8_fp4_mega_moe`, só SM100 + EP, buffer simétrico por grupo EP, `:329-641`); 3 streams CUDA auxiliares para sobrepor GEMMs de entrada (compressor kv_score, `indexer.weights_proj`, `indexer.compressor`) (`:1700-1709`); b12x mHC (`:1135-1255`); b12x WO projection (`:1001-1022`).

### 3.D — Scheduler, IPC e runtime

#### D1. Issue #27 — teto de prefills parciais em voo (sempre ON)
- **Arquivo:** `patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py`; alvo `vllm/v1/core/sched/scheduler.py`.
- **Problema:** o v1 define `max_num_partial_prefills` mas o laço de admissão nunca lê; com chunked prefill + async + várias requisições longas, prefills no início de `self.running` consomem todo o orçamento e decodes atrás recebem `num_new_tokens == 0` e são pulados (`continue`, não preempção) → **inanição de decode sem preempção**, pior quanto maior o prompt.
- **Mecanismo (r3):** no topo do laço de admissão de `waiting`, conta prefills parciais a partir de `self.running` em paridade exata com o conjunto `_inflight_prefills` do stock: requisições admitidas em passos anteriores contam se `num_computed_tokens < num_prompt_tokens`; admitidas neste passo contam se `num_computed_tokens + num_scheduled_tokens < num_tokens` (um prompt inteiro num chunk não é parcial). Se `≥ cap`, `break`. Cap lido uma vez no construtor (`DSPARK_MAX_INFLIGHT_PREFILLS`, 1–3). Tripwire limitado a 16 warnings se o conjunto subcontar. Recusa aplicações r2/pre-r2.
- **Efeito:** junto com `--long-prefill-token-threshold 1024`, é o que faz "seis prompts frios enormes" enfileirarem em vez de matar o decode.

#### D2. Issue #43 — piso de serviço de decode em passos mistos + diagnóstico (sempre ON)
- **Arquivo:** `patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py`; alvo `scheduler.py`; simulador `tests/sim/test_issue43_scheduler_sim.py`, teste de aplicação `tests/test_issue43_patchapply.py`.
- **Problema:** #27 curou a inanição por passo (pior ITL 2790 ms → 67 ms), mas havia grande dispersão de taxa de decode por requisição (min/max 0,107–0,238).
- **Mecanismo:** (A) ao escalonar um chunk de prefill do `running`, reserva `num_sampled_tokens_per_step` para cada requisição decode-ativa ainda não visitada atrás dela e limita o chunk a `token_budget − reserva` (ou 0 → pula o chunk); (B) registra pulos de decode `(req_id, posição, computed)`; (C) registra tokens por requisição (prefill/decode); (D) log por passo com `DSPARK_ISSUE43_SCHED_DIAG=1`. É no-op com os defaults (1024 ≪ 8192), mas protege contra mis-tuning.
- **Conceitos:** fairness; token budget; decode vs prefill em continuous batching.

#### D3. Adaptive prefill chunk (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-adaptive-prefill-chunk.py` (+ `tests/test_adaptive_prefill_chunk.py`).
- **Mecanismo:** se nenhum membro de `self.running` é decode-ativo (`not r.is_prefill_chunk`), o teto vira `max_num_batched_tokens` (8192), em ambos os laços (running e waiting).
- **Motivação:** um c=1 frio pagava custo fixo por chunk (86 all-reduces, setup de indexer/compressor, ~1,5k launches) 8 vezes por 8K (`fable5-1 §3.7`: estimativa −10–25 % TTFT ≤ 32K).
- **Efeito medido:** rejeitado por memória do host (MemAvailable 2,36/2,28 GB; workspaces maiores vivem em memória unificada fora do orçamento `gpu_memory_utilization`).

#### D4. Issue #26/#36 — prefix cache híbrido MLA + SWA (v2, sempre ON)
- **Arquivo:** `patches/hotfix-dsv4-issue26-hybrid-swa-min.py`; alvo `vllm/v1/core/kv_cache_coordinator.py` (`find_longest_cache_hit`).
- **Contexto:** DSV4-Flash + DSpark tem **quatro grupos de KV** (1× MLA + 3× SlidingWindowMLA); o hit comum é o **mínimo** entre grupos.
- **v1 (#26):** ignorava o mínimo para grupos SWA → hits quentes voltaram, mas quando a SWA não tinha a cauda naquele comprimento, o engine pulava prefill e anexava KV SWA **nulo** → saída "DSML/CJK salad" em prefixos de ~21K (#36).
- **v2 (atual):** volta a deixar a SWA encolher o hit (é o código stock, anotado); os hits quentes passam a vir de `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` (caudas SWA esparsas retidas). "A 21k prefix may cache ~20k instead of 21k; that is correct." Reverte uma injeção v1 se encontrada.
- **Conceitos:** gerenciador de KV híbrido; janelas deslizantes só guardam os últimos W tokens; checkpoints de retenção.

#### D5. Empty encoder output (sempre ON)
- **Arquivo:** `patches/hotfix-vllm-empty-encoder-output.py`; alvos `v1/outputs.py`, `v1/core/sched/scheduler.py`.
- **Problema:** a imagem inventava token 0 quando um produtor encoder-only não amostrava nada e não tinha o ramo de conclusão encoder-only.
- **Mecanismo:** backport de vLLM `7ca49fbe…`: `sampled_token_ids = [[] for _ in req_ids]`; flag `is_mm_encoder_only` e ramo `FINISHED_STOPPED` quando o prompt foi consumido. Pré-valida os dois arquivos antes de escrever qualquer um.

#### D6. Issue #79 — spin-wait do IPC (sempre ON)
- **Arquivo:** `patches/hotfix-gb10-spin-wait.sh`; alvo `vllm/distributed/device_communicators/shm_broadcast.py`.
- **Problema:** `SpinCondition` gira em P-cores por `busy_loop_s` (1 s) antes de dormir; em decode o intervalo entre mensagens é sempre < 1 s ⇒ 3–4 núcleos girando no clock máximo (calor/energia no SoC compartilhado com o GPU).
- **Mecanismo:** `busy_loop_s: float = 1 → 0.002` (2 ms de graça e depois dorme no socket de notificação). Só TP ≥ 2 (TP=1 usa o executor uni).

#### D7. Issue #117 — recuperação limitada do ring buffer SHM (default ON)
- **Arquivo:** `patches/hotfix-vllm-issue117-shm-ring-buffer.py` (+ fixtures `scripts/fixtures/issue117/shm_broadcast-{752a3a504,10c75477}.py`).
- **Problema:** um reader local de `MessageQueue` estacionado em `SpinCondition.wait()` após perder uma notificação PUB/SUB; o dado já está no SHM, mas um poll indefinido no socket nunca relê o flag.
- **Mecanismo (backport vLLM #45224, merge `10c75477…`):** `SHM_READER_RECHECK_INTERVAL_MS = 5000` limita `timeout_ms()` (inclusive leituras sem deadline); `acquire_read()` libera o slot, avança o índice e registra a leitura em `finally` mesmo se o consumidor levantar exceção. Aceita 4 identidades de arquivo (stock/patched × com/sem #79). Não limpa objetos `/dev/shm/psm_*` órfãos.
- **Conceitos:** ring buffer em memória compartilhada; flags de escrita/leitura com memory fence; notificação ZMQ PUB/SUB; "lost wakeup".

#### D8. #44993 — avanço da gramática através do fim do raciocínio (grupo v0.27, ON)
- **Arquivo:** `patches/hotfix-dsv4-grammar-advance.sh`; alvos `v1/structured_output/__init__.py`, `v1/core/sched/scheduler.py`.
- **Problema (issue #24):** `response_format json_schema` + thinking → prefixo duplicado `{"city":"Paris{"city":"Paris"}`: `should_advance` retornava False para JSON/regex/choice quando o raciocínio terminava no meio do passo; com spec decode, o rascunho logo após `</think>` nunca entrava no FSM.
- **Mecanismo:** `should_advance(request, new_token_ids=...)` usa os tokens recém-anexados como janela exata (o cálculo por placeholders erra sob async + spec com rejeições, #43388); registra `reasoning_end_token_index` para **todos** os tipos de restrição; o scheduler passa `new_token_ids`.
- **Conceitos:** structured output com FSM; placeholders assíncronos; fronteira raciocínio/resposta.

### 3.E — Serving, tokenização, tool calling e API

#### E1. Instalação do encoder + correção do mapeamento de esforço (compose, sempre)
- **Onde:** `docker-compose.dspark.yml:430-444`.
- **Mecanismo:** procura `encoding/encoding_dsv4.py` no snapshot (revisão pinada → qualquer snapshot → 0731) e copia para `vllm/tokenizers/deepseek_v4_encoding.py`; edita `vllm/tokenizers/deepseek_v4.py` para que `reasoning_effort` `high` fique `high` e qualquer outro valor (inclusive `medium`) vire `low` (antes caía em `high`). O checkpoint **não tem Jinja**: o encoder Python é o chat template.

#### E2. Issue #21 — argumentos de tool call em dict
- **Arquivo:** `patches/hotfix-encoding-dsv4-issue21.py`; alvo `vllm/tokenizers/deepseek_v4_encoding.py`.
- **Problema:** `encode_arguments_to_dsml` sempre fazia `json.loads(arguments)`; replay OpenAI com `arguments` já `dict` gerava exceção e o `except` embrulhava como `{"arguments": <dict>}` — o modelo imitava o histórico corrompido.
- **Mecanismo:** se `dict`, usa direto; senão `json.loads`, e só embrulha strings não-JSON (`TypeError/ValueError`).

#### E3. Issue #52 (+ #120) — turno final do assistant (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-assistant-final-continuation.py`.
- **Problema:** `render_message()` só anexa o header de geração (`<｜Assistant｜>` + `<think>`) quando a última mensagem é `user`/`developer`; se termina em `assistant` (retry de harness), o prompt termina num EOS "morto" → 1–2 tokens `stop`, markup alucinado; autossustentável. Com um `latest_reminder` depois do assistant, idem.
- **Mecanismo:** amplia a condição para "último = assistant" ou "último = `latest_reminder` precedido de assistant"; mantém o turno fechado e **acrescenta** um header novo (reabrir com `wo_eos` foi medido pior: 1 token vazio num turno completo). Self-check: importa o encoder patchado, renderiza os casos e **restaura os bytes originais** se falhar.
- **Evidência:** A/B causal via `/v1/completions`: turno deixado aberto → 183 tokens coerentes; fechado com EOS → 400 tokens de `<|DSML|tool_calls>` cru como texto. Tabela do docstring: parcial 32/208/260 e completo 16/1/140 tokens (stock / wo_eos / EOS+header).

#### E4. Issue #144 — alinhamento da diretiva de esforço para prefix cache (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-issue144-effort-align.py` (+ fixtures `scripts/fixtures/issue144-effort-align/*`).
- **Problema:** o encoder insere a diretiva de esforço logo após BOS e antes do system (BOS+diretiva = 93 tokens em `max/xhigh`, 80 em `high`, 0 em `low/off/medium`); como o prefix cache encadeia hashes de blocos de 256 tokens, esforços diferentes divergem no bloco 0 e compartilham **zero** blocos (medido: cross-bucket 0 %, intra-bucket 96–98 %).
- **Mecanismo:** renderiza `BOS + system-region + "\n\n" + diretiva + resto`; `low`, chat-mode, continuações e conversas sem system inicial ficam byte-idênticas. Self-check e rollback.
- **Efeito medido:** prompt agentic de 4646 tokens: stock 0 blocos compartilhados; alinhado 18/18 (prefixo comum 4630 tokens; a fusão BPE na junção custa 1 token). Reivindicação do reporter: TTFT 2,3–2,7× em tráfego repetido high/max (a validar). Só o `chat_template_kwargs.reasoning_effort` chega ao encoder — o campo OpenAI top-level `reasoning_effort` é ignorado.

#### E5. Suppress stops in reasoning (ON)
- **Arquivo:** `patches/hotfix-dsv4-suppress-stops-in-reasoning.py`; alvo `vllm/v1/engine/detokenizer.py`. Port do "Patch 5" de tonyd2wild/Capicua25x.
- **Problema:** `stop` do cliente é casado contra todo o stream; com "think-in-prompt" o CoT repete stops de harness (`Question:`) e a requisição termina no meio do raciocínio com `content: null`.
- **Mecanismo:** arma o guard só se o último token do prompt é o marcador de início de raciocínio; mantém stops dormentes até ver `</think>`; em spec decode, se `</think>` chega no mesmo chunk de k+1 tokens, avança `stop_check_offset` para depois do marcador. EOS/max_tokens inalterados.

#### E6. Issue #55 — truncamento de tool call por `max_tokens` (ON)
- **Arquivo:** `patches/hotfix-dsv4-issue55-tool-truncation.py`; alvo `entrypoints/openai/chat_completion/serving.py`.
- **Problema:** truncado por `length`, o vLLM reportava `finish_reason="tool_calls"` e podia entregar `arguments` não-JSON → a próxima rodada inteira dava HTTP 400.
- **Mecanismo:** streaming: se `finish_reason == length`, reporta `length` e remove tool calls não parseáveis do delta final; não-streaming: idem na escolha final. Listas continuam listas (serializers chamam `len`).

#### E7. Issue #191 — contrato fail-closed para `tool_choice` named/required (default OFF)
- **Arquivo:** `patches/hotfix-vllm-issue191-toolcall-failclosed.py` (sobre a identidade pós-#55 `08ddb5f3…` → `873ac9c6…`).
- **Problema:** gate de 145 casos `scripts/verify-issue136-xgrammar-live.py` deu 142/145 duas vezes a c=4: 200 com zero `tool_calls` ou argumentos violando o schema `strict`; rótulos diferentes a cada execução, replay 18/18 limpo.
- **Diagnóstico medido (2026-09-03):** a tag estrutural DSV4 é uma sequência estrita (`\n\n<｜DSML｜tool_calls>\n … </｜DSML｜tool_calls>`) e o XGrammar 0.2.3 a aplica; os `Failed to advance FSM` vinham do ramo tolerado (rascunhos após `</think>` checados contra a gramática nova). A falha residual real é **comprimento de raciocínio**: com `thinking=true, reasoning_effort=low`, parte das requisições raciocina 300–500 tokens e `max_tokens=512` corta antes/dentro da chamada DSML.
- **Mecanismo:** `_issue191_tool_contract_violation` (cardinalidade, nome, JSON, tipo, schema via `jsonschema` 4.26.0 com fallback, truncamento, sem escolhas); em `failclosed`, regenera com id de engine novo (`<id>-issue191r<n>`) até N vezes e depois HTTP 500; a **última** tentativa troca `<think>` final por `</think>` (= render com `thinking=false`), passa `reasoning_ended=True` e parseia sem raciocínio. Streaming fora de escopo; beam search nunca repetido.
- **Companheiro:** `DSPARK_ASYNC_SCHEDULING=0` para o A/B do gatilho no engine.

#### E8. Issues #136 + #210 — terminação do XGrammar e validação pós-raciocínio (default OFF)
- **Arquivo:** `patches/hotfix-vllm-issue136-xgrammar-termination.py`; alvos `v1/structured_output/backend_xgrammar.py` (stock `231f6b9d…` 12.699 B → `6c7e23c0…` 12.983 B) e `v1/structured_output/__init__.py` (duas identidades stock: pós-#44993 e prístina).
- **Problema:** com DSpark + `structural_tag` + async + TP=2, um lote aceito pode conter o token terminal seguido de tokens especulativos; o matcher já terminado recebia tokens (`The matcher has terminated ... but is trying to accept new token`) e a requisição podia parar até o timeout de 1800 s do `sample_tokens`.
- **Mecanismo:** backport de vLLM #52805 (`accept_tokens` para no token terminal, conta-o, ignora o resto e trata aceitações posteriores como no-op; `validate_tokens` para na terminação e devolve `[]` depois; `reset` limpa o flag) + #53046 (na janela especulativa pós-`</think>`, `validate_tokens` antes de `accept_tokens`). Uma flag, uma transação de dois arquivos com rollback do primeiro se o segundo falhar; regiões disjuntas das do #44993 (provado nas duas ordens).
- **Aceitação live:** 145 casos + zero warnings de terminação/FSM/SHM/timeout nos dois ranks.

#### E9. DSML recovery (default OFF)
- **Arquivo:** `patches/hotfix-vllm-dsml-recovery.py` (6 arquivos do parser, pinados por identidade).
- **Problema:** o modelo às vezes emite `<invoke name="...">` completo sem o wrapper `tool_calls` (ou com variante malformada `toolcalls`, vllm#51914) → a chamada vaza como texto/raciocínio.
- **Mecanismo (port de vllm#52645 adaptado à API pre-`token_count`):** transições "provisórias" para `INVOKE_PREFIX` a partir de CONTENT/REASONING; bufferiza eventos e texto; valida o nome contra as tools declaradas na requisição (nada com `tool_choice="none"`); só confirma em `INVOKE_END` (absorvendo um fechamento externo opcional); qualquer outra saída (truncamento, nome rejeitado, `finish()`) devolve o texto cru ao estado original; wrappers V3.2 `function_calls` passam verbatim; nome > 256 chars ou newline aborta.
- **Conceitos:** DSML (markup de tool call do DeepSeek); parser em streaming por máquina de estados.

#### E10. Responses API — store limitado (só com `VLLM_ENABLE_RESPONSES_API_STORE=1`)
- **Arquivo:** `patches/hotfix-dsv4-responses-store.py`; alvo `entrypoints/openai/responses/serving.py` (stock `fe3a48ab…` → `1b003313…`).
- **Problema:** o store stock nunca despeja (vazamento).
- **Mecanismo:** um único LRU para resposta + mensagens renderizadas + eventos de background; recuperação e `previous_response_id` renovam recência; entradas `queued/in_progress`, "pinned" por continuação e produtores rastreados não são despejadas (podem exceder o teto temporariamente); leitores capturam o estado de eventos para que um despejo posterior não trunque o replay. Estado só em memória (perdido em qualquer restart).

#### E11. Issue #138 — replay de histórico sem `type` (default OFF) e E12. Codex `agent_message` (default OFF)
- **Arquivos:** `patches/hotfix-vllm-issue138-responses-history.py`, `patches/hotfix-vllm-codex-agent-message.py`; alvo `entrypoints/openai/responses/protocol.py` (`ResponsesRequest.input_item_parsing`, método inteiro pinado por SHA-256).
- **#138:** aceita exatamente `{"role":"assistant","content":[{"type":"output_text","text":"..."}]}` sem `type` item-level, acrescentando só `"type":"message"` (singleton obrigatório porque a conversão downstream lê `content[0]`).
- **Codex:** converte só o `agent_message` com as chaves exatas evidenciadas para `message(role=assistant)`, descartando `id/author/recipient/metadata`. Aplicado depois do #138; ambos reconhecem o pós-imagem combinado.

#### E13. API keys: multi-chave e redação no log
- **Arquivos:** `docker-compose.dspark.yml:419` (parser de `DSPARK_API_KEYS`), `patches/hotfix-vllm-redact-api-key-log.sh` (alvo `entrypoints/serve/utils/api_utils.py`).
- **Problema:** `log_non_default_args` imprime `'api_key': ['sk-...']` em claro (o irmão `jsonify_non_default_args` tem `exclude`, o de log não).
- **Mecanismo:** `_DSPARK_REDACT_LOG_ARGS = {"api_key"}` e `_dspark_redact_non_default` que substitui por `['<redacted:N value(s)>']` (preserva a **contagem**, que é como se verifica que um único `--api-key` carregou N chaves), sem mutar a lista viva. `--status` faz checagem comportamental (executa o helper). Obrigatório sempre que houver chave. Não fecha `/proc/<pid>/cmdline` nem `docker inspect`.
- **Nota de nomenclatura:** `patches/keys-concurrency.patch` **não** é sobre API keys — "Keys" é o autor (drowzeys) do patch de concorrência do DSpark (C5).

#### E14. Issue #31 — `thinking_token_budget` na GPU (default OFF)
- **Arquivo:** `patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py`; alvos `v1/worker/gpu/sample/thinking_budget_gpu.py` (novo), `v1/engine/input_processor.py`, `config/vllm.py`, `v1/worker/gpu/sample/sampler.py`, `v1/worker/gpu/model_runner.py`.
- **Problema:** a implementação retirada copiava mapeamentos e fatias de tokens GPU→CPU a cada passo → regressão severa de decode em contexto longo; o V2 recusava o campo (HTTP 400).
- **Mecanismo:** estado por requisição em tensores UVA/staged (sem D2H no hot path); `_force_thinking_end_kernel` (Triton) só toca a linha de vocabulário **no passo exato** do limite (tudo −inf, `</think>` = 1e9), olhando para trás `max_lookback = num_speculative_tokens` linhas para não forçar se `</think>` já apareceu no caminho especulativo; `_observe_accepted_end_kernel` desativa o estado quando `</think>` aceito (inclusive tokens MTP). Exige delimitadores de token único (DSV4 tem). Sem default server-side.

### 3.F — Multimodal (Vision-Exp)

#### F1. `hotfix-dsv4-vision-exp.py` + `patches/vision_exp/{apply,vision,image_processor,processor}.py` (sempre ON)
- **Problema:** `DeepseekV4ForCausalLM` da imagem 0.1.1 é só texto; Vision-Exp traz ViT de 32 camadas + Aligner e o token `<｜deepseek_image｜>`.
- **Mecanismo (em 3 arquivos da imagem):**
  1. `nvidia/model.py`: anexa um hook fail-closed que importa `/opt/dspark-patches/vision_exp/apply.py` e faz monkey-patch: constrói `ViT` + `Aligner` + parâmetros `image_start/end/newline/pad`; mapeia pesos `vision.*`, `aligner.*`, `image_*`, `bias_vl`; torna a classe `SupportsMultiModal` (`requires_raw_input_tokens = True`, porque a hash MoE das camadas 0–2 precisa dos ids); registra o processor multimodal.
  2. `nvidia/dspark.py` (Anemll): remapeia `ffn.gate.bias_vl` → `e_score_correction_bias_vl` no loader do drafter (o stock só tratava nomes terminando em `.ffn.gate.bias`).
  3. Encoder: relaxa a rejeição do placeholder já substituído (para `image_url` sobreviver ao parser do vLLM) e reimpõe a regra oficial: **imagens só em mensagens `user`** (`system/assistant` → 400); texto de `tool/function` não é escaneado (issue #167); `<image>…</image>` citado em prosa não é imagem (#181).
- **Modelo de visão (`vision.py`):** patches de 14×14 → `Linear(3·14² → vision_dim)`; blocos pre-norm RMSNorm + atenção **bidirecional** com **RoPE 2D** (freqs para linha e coluna) + MLP SwiGLU; `Aligner`: pad para múltiplo de 3, `unfold` 3×3 (downsample ×3 por lado ⇒ 9 patches por token), `Linear → GELU → Linear` para 4096.
- **Pré-processamento (`image_processor.py`):** redimensiona com orçamento de ≤ 384 tokens (`solve_resize_ratio`, `safe_resize`), mínimo de 147.456 pixels, razão máx. 8, `ImageOps.pad` cinza (127), normalização `(x−0,5)/0,5` bf16; layout "N" do bloco: `[pad de compressão] + IMAGE_START + (linhas de IMAGE + IMAGE_NEW_LINE, intercaladas em pares de linhas) + pad + IMAGE_END`. **`compress_pad_tokens(start_pos) = 3 − start_pos % 4`**: o bloco de imagem começa numa fronteira de 4 tokens — alinhado à compressão C4 (interpretação: para que os blocos comprimidos não misturem texto e imagem).
- **#172 (hash do encoder cache):** o comprimento do bloco depende de `start_pos % 4` (ex.: grade ViT 40×19 → grade LLM 7×14 → 122 + pad 0–3 = 122–125 tokens); o hash multimodal do vLLM é só dos bytes da imagem, então o processor "salga" o hash com `num_tokens` (`salt_mm_image_hash`) e não usa o cache de processor para imagens.
- **#175 (roteamento de tokens de imagem):** linhas de placeholder de imagem roteiam com `e_score_correction_bias_vl` e **sem** a tabela de hash (senão todas as linhas `129264` iriam para os mesmos experts nas camadas 0–2); a classificação `text/image/mixed` é feita **uma vez por forward** em `embed_input_ids` (reusando a soma de `is_multimodal` já feita para a checagem de #172) e publicada numa "célula" compartilhada pelos gates — em vez de 43 syncs host-device por forward; durante captura de CUDA graph nunca resolve (decode graphs só reproduzem texto). Teste: `tests/test_issue175_routing_kind_once.py`.
- **Limites:** sem vídeo; GIF = primeiro quadro; `LIMIT_MM_PER_PROMPT={"image":8}`.
- **Conceitos:** ViT, RoPE 2D, projetor/aligner, placeholders multimodais, encoder cache, roteamento MoE específico por modalidade.

### 3.G — Testes e garantias (o que o repo prova em CPU)

| Suite | O que garante |
|---|---|
| `tests/test_dspark_stacked_mapping.py` | Mapeamento checkpoint → parâmetros "stacked" do DSpark (e que `markov_w1` não é confundido com `w1`). |
| `tests/test_issue27_inflight_cap.py` | Contagem de prefills parciais em paridade com `_inflight_prefills`; cap; r3. |
| `tests/test_issue43_patchapply.py` + `tests/sim/test_issue43_scheduler_sim.py` | Aplicação no `scheduler.py` real (py_compile, idempotência, ordem com #27) e simulador puro-Python mostrando que nenhum decode-ativo é pulado com prefill agendado nas seis células frias (32K/62K × 2/4/8). |
| `tests/test_adaptive_prefill_chunk.py` | Semântica do teto adaptativo. |
| `tests/test_replicate_markov_head.py` | Aplicação no `qwen3_dspark.py` real da imagem. |
| `tests/test_sp_indexer_prefill.py` + `scripts/test-sp-indexer-gpu.py` | Matemática split/bounds vs força bruta (TP 2/3/4) e equivalência GPU com kernels reais (TP=2/3 emulados). |
| `tests/test_issue133_triton_specialization.py` | Família de especialização Triton sem driver. |
| `tests/test_issue175_routing_kind_once.py` | Uma classificação por forward, sem syncs extras. |
| `scripts/test-hotfix-atomic-transaction.py` | Transação de todo o script nos `.sh` (staging, rollback byte-exato, permissões) e inventário congelado de hunks por SHA-256. |
| `scripts/test-python-hotfix-failclosed.py` | Ordem e `|| exit 1` dos patchers Python no compose. |
| `scripts/test-issue{117,136,138,141,144,191}*.py`, `test-dsml-recovery.py`, `test-dspark-{block-k,swa-prefix}.py`, `test-rope-swa-fix.py`, `test-mxfp4-indexer-cache.py`, `test-c128a-prefill-cache.py`, `test-responses-store-hotfix.py`, `test-redact-api-key-log.py`, `test-dspark-api-keys.py`, `test-issue55-tool-truncation.py`, `test-encoding-dsv4-issue21.py`, `test-assistant-final-continuation.py`, `test-suppress-stops-in-reasoning.py`, `test-issue26-swa-min-v2.py`, `test-issue31-thinking-budget-gpu.py`, `test-empty-encoder-output-hotfix.py`, `test-dsv4-vision-exp-hotfix.py`, `test-runtime-ablation.py` | Para cada hotfix: identidades pinadas, transformação pura, idempotência, recusa de drift, rollback atômico, fiação no compose/launcher. `test-issue144` inclui uma simulação de prefix cache com hash encadeado (blocos de 256). |
| `scripts/test-numeric-knob-validation.py`, `test-env-*`, `test-start-*`, `test-status-logs-probes.py`, `test-stop-name-filter.py`, `test-build-rsync-guard.py`, `test-gb10-install-failclosed.py` | Robustez operacional do launcher (fora do escopo principal). |

> Nota: CI (`.github/workflows/validate.yml` → `scripts/ci-validate.sh`) é só CPU; tok/s precisa do par de Sparks.

---

## 4. Números e benchmarks (com datas)

### 4.1 Capacidade de KV / memória

| Medida | Valor | Fonte / data |
|---|---|---|
| Pool de KV (TP=2, util 0,835, Vision-Exp) | **17,04 GiB / 2.331.430 tokens / 2,22× a 1M** | `README.md:117-121, 141` |
| TP=2 com 0731, util 0,83 | 17,04 GiB, pesos 79,17 GiB/rank | `fable5-1:57` (boot 9 ago) |
| A/B 2026-09-02/03 (head/worker) | 15,96–17,46 / 14,73–15,87 GiB (ruído de boot ±0,6 GiB); head "GPU KV cache size" 2.188.268 tokens, 2,09× | `ab-results-2026-09-0{2,3}.md` |
| TP=3 (16 slots) | ≈ 35,0 / 33,8 / 35,1 GiB por rank; **5,0 M tokens**, 4,79× a 1M; pesos 60,69 GiB/rank | `docs/TP3.md`; `fable5-1:56` |
| Histórico preview Anemll (util 0,85, MTP-3) | ~2,83 M tokens, 2,70× | `RESULTS-2026-08-14.md` |
| Keys C12 Stage-C | ~3,23 M tokens (~3,2×) | idem (2026-07-02) |
| Custo físico por token | C4 3,07 KB + C128 0,09 KB + indexer 0,69 KB; SWA ≈ 3,4 MB/seq; headline ≈ 7,8 KB/token | `item8:26-31` |
| Página de KV comum | 37.376 B = 584 × 64 (C4: 256 tokens/4; SWA: 64 tokens) | `item8:24` |
| Cada +0,01 de util | ≈ +1,2 GiB KV ≈ +165 K tokens (TP=2) | `fable5-1:165-166` |
| Buffer MTP inútil | 256 MiB/rank (448 MiB upstream) | #50312 |
| Markov head replicada | 66 MB × 2 matrizes; custou −0,44 GiB de KV no head | #replicate; A/B B |

### 4.2 Decode e agregados (TP=2 salvo indicação)

| Cenário | Valor | Fonte / data |
|---|---|---|
| c=1, qualquer prompt até 128K | ~62–83 tok/s | `README.md:327` |
| 6 chats curtos | ~160–190 tok/s agregado (~30–37/stream) | `README.md:328` |
| Matriz 0731 (2026-08-14), prompt 256 | c1 82,6 · c2 56,9 · c4 54,3 · c6 35,9 · agg c6 **162** | `RESULTS-2026-08-14.md` |
| idem 2.048 / 8.192 / 32.768 / 131.072 | c1 64,6 / 65,9 / 62,0 / 74,9; agg c6 78 / 29 / 8 / 2; 128K × c6 TTFT mediana 282 s | idem |
| Sweep 0731 publicado (2026-08-01, PR #14) | 256×c6 agg **191,2**; 2K×c1 decode 68,8, prefill 2.563 tok/s; 32K×c1 TTFT 22,96 s; 128K×c1 TTFT 78,75 s, prefill 1.665, decode 65,2 | `DEEPSEEK_V4_FLASH_0731.md`, `RESULTS` |
| 900K de aceitação | 899.994 tokens, TTFT **1.028,85 s**, ~874,8 tok/s de prefill | idem |
| Breakable vs regular CUDA graphs (1M ctx) | c1 74,6 → **95,9** (+28,6 %); c2 agg 134,2 → 151,8 (+13,1 %); c4/c6 agg 263,7/340,5 (regular) | idem |
| Decode-bench 2048 tokens | x1 82,4; x2 98,0 agg; x3 134,6; x4 120,4 (TTFT 5,36 s) | `RESULTS` |
| Base2 (2026-09-02, sp-indexer on, capture 48) | c1 **56,6** (50,8–67,0, 8 trials); c2 42,7; c6 agg **139**; TTFT 4,5 / 18,2 / 78,5 s (8K/32K/128K) | `ab-results-2026-09-03.md` |
| "Não-norma" | 2026-09-02 decode 15–25 % abaixo dos números de 14 ago sem knob que explique | `README.md:338-341` |
| Ruído c=1 | ±10 tok/s dependendo do prompt sorteado (47,7–76,5) | `ab-results-2026-09-0{2,3}.md` |
| TP=3 vs TP=2 (2026-09-02) | 1 stream 38,6 → 41,9; 2 streams 63,9 → 72,0; 4 streams 86,0 → 89,7; 8/12/16 streams (TP3) 142,8/164,3/198,0 agg | `docs/TP3.md` |
| TP=3 probe ao vivo (fable) | c1 64,3; c6 33,7/stream, 156 agg | `fable5-1:66-71` |
| Stage-C 200K/16 (Keys) | estático C16 **315,1** agg (19,7/stream), C8 252,6; escalonado C16 **205,0**; aceitação 0,609–0,635 | `RESULTS` |
| Keys C12 Stage-C | C1 52,8; C6 127,8; C12 230,1; aceitação 0,585–0,615 | `RESULTS` (2026-07-02) |
| 1M/6 microbench (Keys mask) | ~182 tok/s agg (6/6), solo ~67 | `RESULTS` (2026-06-29) |

### 4.3 TTFT / prefill

| Cenário | Valor | Fonte |
|---|---|---|
| TP=2: 8K / 16K / 32K / 64K / 128K / 256K | 4,4 / 8,7 / 18,0 / 36,0 / 74,9 / 164,8 s | `docs/TP3.md` |
| TP=3 idem | 5,0 / 9,4 / 18,6 / 38,2 / 91,5 / 202,2 s | idem |
| SP indexer ON (128K) | 76,0 / 74,4 s vs 79,0 / 81,2 s (≈ −4 a −8 %) | `ab-results-2026-09-02.md` E |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512` | 32K −1,5 %, 128K −1,8 % (dentro do ruído ±2 %) | `ab-results-2026-09-03.md` E |
| Fairness c=4 × 8K (`INFLIGHT=1` vs `2`) | TTFT spread 14,6 s vs **9,1 s**; TTFT por stream 4,7/9,4/14,3/19,3 vs 7,4/9,0/15,7/16,5 s; agg +12 % | `ab-results-2026-09-03.md` A/F |
| Issue #217 (pós-r3) | spread gate26 1,60–1,76× (cap 1) vs 1,50–1,57× (cap 2); TTFT spread 4,1× vs 2,1–2,4× | `README.md:278` |
| Issue #43 | pior ITL 2790 → 67 ms após #27; dispersão de taxa de decode por requisição 0,107–0,238 | docstring #43 |

### 4.4 Aceitação de rascunho (DSpark)

| Cenário | Por posição | Fonte |
|---|---|---|
| Vision-Exp k=6 (bench_quick) | 0,89 / 0,73 / 0,49 / 0,34 / 0,23 / 0,15 | `PATCHES.md:842-843` |
| 0731 k=5 | 0,93 / 0,75 / 0,66 / 0,58 / 0,47 | idem |
| Vision-Exp k=5 vs k=6 (2026-09-03) | 0,88 / 0,74 / 0,53 / 0,36 / 0,24 (igual nos dois) | `PATCHES.md:866` |
| Probe TP=3 (fable) | 0,92 / 0,74 / 0,60 / 0,41 / 0,28 / 0,21; 2004/4140 = **48,4 %**, ≈ **3,9 tokens/passo** | `fable5-1:73-75` |
| Prosa natural (A/B 2026-09-02) | 25 % geral, pos0 0,66–0,71 (vs ~0,93 "bons prompts" em AUDIT) | `ab-results-2026-09-02.md` |
| Palavras numeradas | 47–53,6 %, pos0 0,90–0,96 | idem |
| Prosa, posições 3–5 | 0,13 / 0,07 / 0,03 | `ab-results-2026-09-02.md` (next steps) |

### 4.5 Modelo de custo de um passo (fable5-1 §2)

| Item | Valor |
|---|---|
| Bytes por rank por passo (c=1, k=6 → 7 tokens verificados + 6 queries de draft) | Experts ≈ 12 GB (≤ 42 distintos de 256 por camada × 6,7 MB/rank × 43 camadas); atenção ≈ 2,4 GB; drafter ≈ 0,9 GB; `lm_head` ≈ 1,06 GB; Markov w2 ≈ 0,2 GB; **total ≈ 16,5–17 GB** → 68–74 ms a 230–250 GB/s efetivos |
| Coletivas por passo (TP=2) | 43 (`wo_b`) + 43 (MoE) + 6 (draft) + 2 (embeddings) + 2 all-gather de logits (10,9 MB a c=6) + 12 (laço Markov) ≈ **108**, 30–50 µs cada (sem GDR) ≈ 3–5 ms |
| Conclusão | c=1 é ~80 % streaming de pesos; TP=3 corta bytes/rank em 1/3 mas soma ~13 ms de custo fixo (ring de 3 ranks, heads padded 64→72) |
| Prefill | 86 all-reduces de `[1024×4096]` bf16 = 8 MB por chunk de 1024 (≈ 690 MB/chunk/rank); indexer ≈ 80 TFLOP por chunk a 900K (21 camadas) |

### 4.6 Efeitos de patches (resumo)

| Patch | Efeito |
|---|---|
| #22 | ~1 → ~17 tok/s a 600K+ (≈16×) |
| #49486 | 3,4 % E2E TTFT (upstream) |
| #48957 | ~2× no kernel C128 (upstream) |
| #50298 | 1,88× no kernel (upstream) |
| #50312 | 256 MiB/rank (ver inconsistência B2) |
| DSpark SWA prefix | json60 repetido 12,1 → 86,5 tok/s (upstream) |
| Block-k (k=5) | decode 8K greedy 56,6 vs 55,3; 32K 56,6 vs 53,6 |
| #141 | 64 linhas: 3.145.728 tokens sem falha num par; 65 falhou |
| #144 | 0 → 18/18 blocos compartilhados (4646 tokens) |
| Replicate Markov | c=6 122 vs 148 (pior) → OFF |
| TC decode | c=1 ≈ 46 vs 55 (−15 %) → OFF |
| capture 48 vs 42 | c=6 139 vs 122 (+12 %) |
| Patch 2b | GSM8K 93,5 % vs 95,0 %, 97,5 % concordância, 0 erros |
| #191 | 142/145 → meta 145/145 com hotfix |
| TP=3 R1 (teste) | rel 2,3e-06 vs TP=1 (numa checagem do arquivo `dsv4_tp_pad.py`) — mas o caminho aplicado é R2 |

---

## 5. Conceitos necessários (em ordem de pré-requisito)

Cada item: o que é + como aparece concretamente aqui.

### 5.1 Fundamentos de ML / Deep Learning

1. **Rede neural, pesos e forward pass.** Uma função parametrizada que transforma vetores por camadas lineares e não-lineares. Aqui, "servir o modelo" é executar só o forward (inferência), nunca o backward; todo o repo trata de fazer esse forward caber e ser rápido em 2 GPUs.
2. **Multiplicação de matrizes (GEMM) e o custo de memória vs compute.** A maior parte da inferência é GEMM. Com lote pequeno (decode), o gargalo é **ler os pesos da memória** (bandwidth-bound); com lote grande (prefill), é FLOP (compute-bound). O fable5-1 mostra c=1 ≈ 80 % streaming de pesos; o plugin GB10 e o b12x escolhem kernels diferentes por "M" (número de linhas/tokens).
3. **Representação de números: FP32, BF16, FP16.** BF16 tem o expoente do FP32 com mantissa curta; é o dtype de ativações aqui (RoPE no KV fica em bf16, `main_kv_cache` do drafter é bf16).
4. **Softmax estável e log-sum-exp.** Subtrai-se o máximo antes do `exp`. Os kernels DSpark (`dspark_kernels.py:259-289`) e o split-K de decode (`mid_lse`) usam isso.
5. **Normalização (LayerNorm/RMSNorm).** RMSNorm divide pelo RMS sem subtrair média. Aparece em `q_norm`, `kv_norm`, `attn_norm`, `ffn_norm`, `main_norm` e até no Q "sem peso" do drafter (`dspark.py:464`).
6. **Funções de ativação SiLU/SwiGLU e clamp.** SwiGLU = `silu(gate) * up`. DSV4 usa `SiluAndMulWithClamp(swiglu_limit)` (`model.py:298-301`); o b12x recebe `swiglu_limit`.
7. **Embeddings e LM head / logits.** Token id → vetor; no fim, vetor → logits sobre o vocabulário (129.280). Em TP, ambos são vocab-parallel (`VocabParallelEmbedding`, `ParallelLMHead`), gerando all-reduce/all-gather — alvo da replicação da Markov head.
8. **Amostragem: greedy, temperatura, top-p.** Greedy = argmax. `draft_sample_method` escolhe greedy ou probabilístico para o rascunho; temperatura afeta também tokens de sintaxe DSML (GLM-NEW-REPORT §3.1).

### 5.2 Transformer / LLM

9. **Tokenização (BPE) e tokens especiais.** Texto → ids. Aqui: `<think>`, `</think>` (tokens únicos — requisito do #31), `<｜Assistant｜>`, `<｜deepseek_image｜>` (129264), EOS; a fusão BPE numa junção custa 1 token (#144).
10. **Modelo autoregressivo e os dois regimes: prefill e decode.** Prefill processa o prompt inteiro; decode gera um token por passo. Todo o trabalho de scheduler (#27/#43, chunk cap) é equilibrar esses dois regimes.
11. **Atenção (Q, K, V), máscara causal e escala 1/√d.** `softmax_scale = head_dim**-0.5` (`model.py:1024`). O drafter DSpark usa atenção **não-causal** dentro do bloco.
12. **Multi-head / MQA / GQA.** Várias cabeças de query; MQA compartilha um único K/V entre cabeças. DSV4 é MQA sobre um vetor latente de 512 dims (1 "KV head").
13. **Residual stream e pre-norm.** Cada bloco soma sua saída ao residual. DSV4 troca o residual único por **hyper-connections** (conceito 30).
14. **KV cache.** Guardar K/V de tokens passados para não recomputar. É o principal consumidor de memória em contexto longo; tudo sobre `nvfp4_ds_mla`, SWA, compressão e paginação é sobre ele.
15. **Codificação posicional RoPE.** Rotaciona pares de dimensões por ângulo proporcional à posição. DSV4 aplica RoPE só a 64 das 512 dims (RoPE parcial), `is_neox_style=False` (pares intercalados).
16. **Extensão de contexto: YaRN / interpolação de frequências.** Reescala frequências para usar contexto além do treino. Aqui: fator 16 de 64K para 1M. O RoPE SWA fix limita YaRN às camadas com compressor.
17. **Chat template / encoder de mensagens.** Converte mensagens em tokens. DSV4 não tem Jinja: usa `encoding/encoding_dsv4.py` (instalado como `deepseek_v4_encoding.py`); alvo de #21, #52, #144 e Vision-Exp.
18. **Modelos de raciocínio (thinking) e esforço.** O prompt termina em `<think>`; `reasoning_effort` insere diretivas (`low` vazio, `high` ~80 tokens, `max` ~93 tokens). `max_tokens` conta raciocínio + resposta (README "Thinking and max_tokens").
19. **Tool calling e DSML.** O DeepSeek emite chamadas em markup DSML (`<｜DSML｜tool_calls>`, `<｜DSML｜invoke name=…>`, `<｜DSML｜parameter …>`); o parser `deepseek_v4` transforma em `tool_calls` OpenAI. Hotfixes #21, #55, #191, DSML recovery.
20. **Structured output / constrained decoding (XGrammar, FSM, bitmask).** Uma gramática vira autômato; a cada passo um bitmask proíbe tokens inválidos. Interage mal com spec decode (tokens após terminação, #136) e com a fronteira de raciocínio (#44993, #210).

### 5.3 Arquitetura DeepSeek-V4

21. **MLA (Multi-head Latent Attention).** Em vez de K/V por cabeça, guarda-se um vetor latente comprimido por token; as queries passam por projeções de baixo posto (`wq_a` → `q_norm` → `wq_b`, `q_lora_rank` 1024). No DSV4 o latente tem 512 dims = K = V (448 NoPE + 64 RoPE), o que torna o KV 1 vetor por token por camada.
22. **Projeção de saída agrupada (`o_groups`, `wo_a` BMM + `wo_b`).** 64 cabeças em 8 grupos; cada grupo tem uma projeção de baixo posto (`o_lora_rank` 1024) feita como BMM FP8 `einsum("bhr,hdr->bhd")`, seguida de `wo_b` row-parallel. É o "contrato r = 4096" que dita o padding de TP=3.
23. **RoPE inverso na saída (`fused_inv_rope_fp8_quant`).** Como V = K contém as dims rotacionadas, a saída da atenção carrega rotação; o DSV4 aplica RoPE inverso nessas dims e quantiza em FP8 antes do `wo_a` (`dspark.py:516-539`). (Explicação do motivo é interpretação.)
24. **Attention sink.** Um logit aprendido por cabeça somado ao denominador do softmax, permitindo "não atender a nada". Pad de sink = −inf (TP3); kernels DSpark e FlashInfer recebem `attn_sink`.
25. **Sliding Window Attention (SWA).** Cada token atende só aos últimos W tokens (W = 128). Toda camada DSV4 tem esse ramo; camadas 0–1 são só SWA. O drafter DSpark também mantém janela de 128.
26. **Compressão de KV (C4A / C128A; CSA/HCA).** Blocos de 4 ou 128 tokens são comprimidos (compressor com `wkv`/`wgate`, RMSNorm, RoPE, quant) em uma entrada de KV; camadas alternam C4 e C128. O compressor mantém "estados parciais" dos blocos incompletos (#48957). A atenção de cada token = janela SWA + entradas comprimidas.
27. **Atenção esparsa com Lightning indexer (herdado do DSA do V3.2).** Um pequeno "indexer" (64 cabeças × 128 dims, cache K FP8 de 132 B) pontua todas as entradas C4 e escolhe top-512 para a atenção principal. É O(queries × chaves) e replicado entre ranks — alvo do SP indexer, do atalho #49486 e do cache MXFP4.
28. **Atenção híbrida (SWA + esparsa + comprimida) e o cache duplo.** Os kernels recebem dois caches (SWA e comprimido) e duas listas de índices (`extra_kv_cache`, `extra_indices`) — `sm120.py:380-393`, `combine_topk_swa_indices`.
29. **Mixture of Experts (MoE): experts roteados + shared expert, top-k.** 256 experts, 6 ativos por token + 1 compartilhado. Decode lê ≈ 39–42 experts distintos por camada (fable) — daí o custo de bytes.
30. **Roteamento: hash MoE, `noaux_tc`, `sqrtsoftplus`, `e_score_correction_bias`.** Camadas 0–2 roteiam por tabela `token_id → experts` (sem router aprendido); as demais usam score + viés de balanceamento sem loss auxiliar. Vision-Exp adiciona `bias_vl` para tokens de imagem (#175).
31. **Hyper-connections manifold-constrained (mHC).** O residual vira `hc_mult = 4` fluxos; antes de cada sub-bloco, pesos (`hc_*_fn/base/scale`) misturam os fluxos na entrada (pre) e depois redistribuem a saída (post, matrizes `post` e `comb`), com normalização Sinkhorn (`hc_sinkhorn_iters`); `hc_head` colapsa no fim. `mix_hc = (2 + hc_mult) × hc_mult = 24` parâmetros por token. É por isso que os buffers de hidden têm `hc_dim = 16384` e que a captura DSpark tira `mean(dim=1)`. (O nome "manifold-constrained" vem da literatura do DeepSeek; o código só mostra o Sinkhorn — interpretação.)
32. **MTP (Multi-Token Prediction) e camadas `mtp.N.*`.** Cabeças extras treinadas para prever tokens futuros; no DSV4 o checkpoint guarda os estágios do drafter DSpark sob o namespace histórico `mtp.*`.
33. **Visão: ViT, RoPE 2D, aligner/projetor, placeholders.** Imagem → patches 14×14 → ViT bidirecional → downsample 3×3 → MLP para 4096 → substitui tokens placeholder no prompt. Bloco alinhado a 4 tokens (C4).

### 5.4 Quantização

34. **Quantização: escala, zero-point, granularidade (tensor/canal/bloco).** Mapear floats para poucos bits com uma escala. Aqui tudo é por **bloco** (64 dims no KV, 128×128 nos lineares FP8, 32 nos experts MXFP4, 16 no NVFP4).
35. **FP8 (e4m3) e seu máximo 448.** 4 bits de expoente, 3 de mantissa. Usado nos lineares, no KV NoPE e no cache K do indexer.
36. **Escalas UE8M0 (potência de 2).** Escala = `2^(e − 127)` guardada em 1 byte, sem mantissa; `scale = 2^ceil(log2(amax/448))` (`cache_utils.py:99-126`). Barata de aplicar (shift de expoente).
37. **FP4 e2m1, MXFP4 e NVFP4.** FP4 tem 1 bit de mantissa. MXFP4 (padrão OCP MX): blocos de 32 com escala E8M0; NVFP4: blocos de 16 com escala FP8 e4m3 + escala global FP32. Os experts do DSV4 são MXFP4 (`fp4_e8m0_k32`), e isso impede o caminho NVFP4 de tensor cores do b12x.
38. **W4A16 vs W4A4 (e W8A8).** W4A16: pesos 4-bit dequantizados para bf16 e multiplicados em bf16 (bom para decode, desperdiça tensor cores no prefill); W4A4: pesos e ativações FP4 em tensor cores FP4 (bom para prefill). O plugin GB10 alterna entre os dois por M.
39. **Quantização de KV cache e layouts "DS MLA".** Guardar KV em FP8 com escalas por bloco + RoPE em bf16: `fp8_ds_mla` = 584 B/token no V4 (656 B no V3.2). `nvfp4_ds_mla` no V4 é o **mesmo** layout (só nome/despacho).
40. **Quantização dos scores do indexer (MXFP4 indexer cache).** Quantizar K do indexer só afeta a **seleção** top-k, não os valores de atenção — risco é "drift de seleção" em contexto profundo.
41. **QAT e simulação quant/dequant.** Treino com quantização simulada; `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` reproduz no drafter o `act_quant` FP8 do QAT de referência nas dims NoPE.
42. **Formatos de checkpoint e escalas (`weight_scale` vs `weight_scale_inv`; E8M0 como uint8).** O loader precisa mapear `.scale` para o nome certo e **não converter numericamente** escalas E8M0 (`view(torch.uint8)`), senão 2⁻⁷ vira 0 (`model.py:1991-1999`).

### 5.5 Speculative decoding e DSpark

43. **Speculative decoding (draft → verify).** Um modelo barato propõe k tokens; o alvo verifica todos num único forward; aceita o maior prefixo compatível + 1 token bônus. Ganho = tokens aceitos por passo (≈ 3,9 no probe) a custo de um passo um pouco mais caro.
44. **Rejection sampling e garantia de distribuição.** Aceita o token do rascunho com probabilidade `min(1, p/q)`; se rejeitar, reamostra do resíduo. Preserva exatamente a distribuição do alvo. `draft_sample_method="probabilistic"` usa q completo; greedy trata q como one-hot.
45. **Probabilidade de aceitação = 1 − TV(p, q).** `speculative_acceptance_confidence` (`v1/spec_decode/dspark.py:508-539`). A sobrevivência de um prefixo é o produto das aceitações condicionais (`cumulative_survival`).
46. **Aceitação por posição e comprimento esperado.** Aceitação cai com a posição (0,89 → 0,15); passar de k=5 para 6 quase não acrescenta tokens e encarece o passo (block-k).
47. **Família de drafters: n-gram, EAGLE, MTP, DFlash, DSpark.** EAGLE/MTP usam estados ocultos do alvo; MTP clássico reexecuta uma camada por posição (daí a regra de divisibilidade); DFlash/DSpark desenham **em paralelo** (`parallel_drafting=True` na Anemll).
48. **DSpark: drafter em bloco paralelo.** Estágios `mtp.*` empilhados num backbone não-causal; entrada = bloco `[âncora, ruído×(block−1)]` + janela de features do alvo; prevê todas as posições de uma vez; amostragem sequencial barata com a Markov head. Bloco treinado = 5.
49. **Features de camadas intermediárias do alvo ("aux hidden states").** O drafter lê as saídas de `dspark_target_layer_ids` concatenadas (média sobre os 4 fluxos mHC) → `main_proj` → `main_norm`. `compute_hash` do spec config inclui isso porque muda o grafo do alvo.
50. **Ring buffer de janela deslizante por posição.** `slot = pos % 128`; escreve só os últimos 128 tokens de cada segmento; rejeitados mantêm o valor antigo (`torch.where`). Estado do drafter **fora** do KV paginado do vLLM.
51. **Markov head de baixo posto.** Viés de logits `W2·W1[token anterior]` (posto 256) somado ao logit de cada posição, escolhido esquerda→direita — dá dependência entre posições do bloco sem rodar o backbone de novo. Shardado ⇒ 12 coletivas serializadas por passo.
52. **Confidence head e verificação agendada por confiança.** `sigmoid(Linear([h, markov_embed]))` estima a aceitação condicional por posição; o proposer pode cortar o comprimento verificado por limiar (não-antecipativo) ou por um modelo hardware-aware `E[aceitos] × steps/s(batch_tokens)`.
53. **Token de ruído e passo de aquecimento.** Posições futuras do bloco são preenchidas com `dspark_noise_token_id`; o primeiro `propose` devolve só ruído (ainda não há feature de token gerado).
54. **Estado por requisição sob continuous batching (slots estáveis).** O vLLM condensa linhas do lote; estados auxiliares precisam seguir o `req_id` (Patch 1) e sobreviver a lotes ragged (Patch 2/2b).
55. **Tratamento de rejeição no contexto do drafter.** O alvo roda os rascunhos rejeitados como padding; o drafter precisa ignorar esses estados (máscara GPU ou trim CPU) antes de gravar no ring buffer.
56. **Speculative decoding × prefix caching.** Num hit de cache, o alvo não recomputa o prefixo, então a janela do drafter fica vazia; a correção força recomputar os últimos 128 tokens.
57. **Speculative decoding × structured output.** Um lote aceito pode cruzar a terminação da gramática ou o fim do raciocínio; é preciso parar no token terminal e validar rascunhos contra a gramática antes de avançar (#136/#210/#44993).

### 5.6 Motor vLLM

58. **Arquitetura do vLLM v1: API server, EngineCore, scheduler, workers, model runner.** O scheduler decide tokens por passo; workers (um por GPU/rank) executam o model runner; a comunicação local usa SHM. Existem dois model runners: V1 (`gpu_model_runner.py`, overlay) e V2 (`v1/worker/gpu/...`, Anemll).
59. **Continuous batching.** Requisições entram e saem do lote a cada passo; o lote é "condensado" quando alguém termina.
60. **PagedAttention, blocos e block table.** O KV é alocado em páginas de tamanho fixo; cada requisição tem uma tabela de blocos lógicos→físicos; índices top-k locais precisam ser convertidos em slots físicos (`compute_global_topk_indices_and_lens`).
61. **Gerenciador de KV híbrido (grupos de KV).** Modelos com camadas de tipos diferentes têm grupos distintos (1 MLA + 3 SWA aqui) que precisam de páginas de mesmo tamanho em bytes; o hit de prefix cache é o mínimo entre grupos.
62. **Prefix caching por hash encadeado.** Cada bloco (256 tokens) tem hash que inclui o hash do pai; uma diferença no início invalida todos os blocos seguintes (#144). SWA só retém caudas → `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`.
63. **Chunked prefill e token budget.** Prefill em pedaços dentro de `max_num_batched_tokens`; `long_prefill_token_threshold` limita cada pedaço; lotes mistos são a origem dos lotes ragged.
64. **Admissão e fairness (partial prefills, decode floor).** Quantos prefills parciais em voo admitir (#27) e quanto orçamento reservar para decodes (#43).
65. **Agendamento assíncrono e placeholders.** O scheduler agenda o passo seguinte antes de saber os rascunhos reais; usa `-1` e `num_output_placeholders`; DSpark pode encurtar o rascunho (`draft_token_lengths`).
66. **CUDA graphs: captura, replay, FULL vs PIECEWISE, tamanhos de captura.** Grava a sequência de kernels para eliminar overhead de CPU; exige formas fixas e buffers persistentes; FULL não pode ramificar por metadado de CPU (#48957) e não pode "assar" atalhos dependentes de dados (#52492); tamanhos arredondados a múltiplos de `k+1` (7) ⇒ capture 48.
67. **Breakable CUDA graphs (Anemll).** Modo alternativo que quebra o grafo; aqui é mais lento (−28,6 % c=1), por isso `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.
68. **torch.compile / AOT cache e custom ops.** O modelo é compilado; ops customizadas (`direct_register_custom_op`, com `fake_impl` e `mutates_args`) ficam fora do grafo; mudar o código (ablation) exige invalidar o cache AOT.
69. **Workspace manager.** Buffers de scratch compartilhados (`current_workspace_manager().get_simultaneous`), reservados no dummy run antes de `lock_workspace()`; alocar durante captura é proibido.
70. **Backends de atenção e registry.** vLLM escolhe a implementação por compute capability, dtype, head size; aqui `DSV4_SPARSE_MLA_SM120`, `FLASHMLA_SPARSE`, `DEEPSEEK_SPARSE_SWA`, `B12X_MLA_SPARSE` (V3.2).
71. **Oracle/modular kernel de MoE.** Seleção de backend por formato de peso (`map_mxfp4_backend`), prepare/finalize e experts plugáveis (`FusedMoEExpertsModular`), `workspace_shapes`.
72. **Warmup e autotune de kernels.** Pré-JIT de formas conhecidas e escolha de táticas por benchmark; autotune coerente entre ranks via broadcast (`kernel_warmup.py`).
73. **Sistema de env vars do vLLM e plugins.** `envs.py` registra `VLLM_*` (desconhecidas só geram warning); `vllm.general_plugins` permite registrar quantizações (plugin GB10).
74. **Reasoning parser, tool parser, detokenizer incremental e stop strings.** Separam `reasoning_content`, extraem tool calls, e checam stops sobre o texto incremental (suppress-stops).
75. **API OpenAI-compatível: Chat Completions e Responses API.** `finish_reason`, `tool_choice` (`auto/required/named/none`), `strict`, `previous_response_id`, store de respostas (#138, Codex, store LRU).
76. **Multimodal no vLLM: processor, placeholders, encoder cache, `SupportsMultiModal`.** Expande placeholders em N tokens, calcula embeddings e mescla; o cache é chaveado por hash de conteúdo (#172).

### 5.7 Inferência distribuída, rede e hardware

77. **Tensor parallelism (column/row-parallel, all-reduce).** Column-parallel divide saídas; row-parallel divide entradas e soma por all-reduce. DSV4: `wq_b` column, `wo_b` row, `fused_wqa_wkv` replicado (`disable_tp=True`), indexer replicado, experts divididos no eixo intermediário.
78. **Vocab parallelism e argmax distribuído.** Logits divididos por vocabulário; top-1 global via all-gather de pares (valor, índice) (`_vocab_parallel_argmax_from_local`).
79. **Replicação vs sharding e o preço das coletivas.** Replicar matrizes pequenas (Markov, 66 MB) remove coletivas, mas não necessariamente acelera (medido pior).
80. **KV replicado sob TP em MLA.** Com um único vetor latente, cada rank guarda o KV inteiro: TP não reduz KV nem o custo do indexer — por isso TP=3 aumenta capacidade (16 slots) mas não acelera prefill.
81. **Padding para TP não divisível.** Preencher cabeças/grupos/vocabulário/intermediário com zeros (e sink −inf) para dividir por 3, preservando contratos de forma dos kernels (R2).
82. **Expert parallelism (EP), all2all, EPLB, MegaMoE.** Distribuir experts inteiros entre ranks; MegaMoE do DeepGEMM exige EP e SM100 — não usado aqui (B12xExperts não suporta EP).
83. **Sequence/context parallelism (SP, DCP) e top-k distribuído exato.** Dividir o eixo de sequência entre ranks e juntar resultados; o merge de top-k locais é exato (SP indexer reusa o kernel do DCP).
84. **NCCL, RoCE, GID, HCA, GPUDirect RDMA / DMA-BUF.** Coletivas entre nós via ConnectX-7; sem DMA-BUF no GB10 as coletivas passam por buffer no host; MTU, PFs PCIe Gen5 ×4 (~126 Gb/s cada) e pause frames afetam prefill (fable §3.3).
85. **Multi-node do vLLM: `--nnodes`, `--node-rank`, `--headless`, executor `mp`.** O head roda API+scheduler+rank 0; o worker roda headless; broadcast ZMQ do scheduler a cada passo (RTT ≈ 0,33 ms).
86. **IPC por memória compartilhada (MessageQueue/SHM ring buffer, spin vs sleep, lost wakeup).** Dispatch local em ring buffer SHM com flags e memory fence; spin 1 s custava CPU (#79); notificação perdida podia travar (#117).
87. **Watchdogs e timeouts (NCCL 600 s, `sample_tokens` 1800 s, flight recorder).** JIT no meio do serviço pode fazer um rank travar o outro; o compose persiste caches e habilita dumps do flight recorder.

### 5.8 GPU, kernels e memória unificada

88. **GPU GB10 / Blackwell de consumo (sm_120/sm_121) vs datacenter (sm_100).** Muitos kernels (FlashMLA, MegaMoE, cache FP4 do indexer) nasceram para SM100; aqui tudo precisa de variantes SM12x (FlashInfer SM120, b12x, aliases DeepGEMM).
89. **Memória unificada (LPDDR5X, 128 GB) e `gpu_memory_utilization`.** CPU e GPU disputam a mesma RAM; workspaces fora do orçamento (chunks maiores) derrubam o MemAvailable do host (adaptive chunk, chunk 2048), e processos do host (polkit, GNOME) roubam KV (fable §3.1).
90. **Triton, CuTeDSL, TileLang, JIT e especialização.** DSLs de kernel compilados em runtime; especializações por alinhamento/constexpr criam variantes (#133); caches precisam persistir.
91. **FlashInfer sparse MLA SM120: decode ≤ 64 linhas vs orquestrador de prefill; split-K e LSE.** Decode divide o eixo de chaves em partições de 64 e combina via log-sum-exp (`mid_out`, `mid_lse`); > 64 linhas vai ao caminho de prefill (#141).
92. **DeepGEMM (fp8 GEMM, mqa_logits, MegaMoE) e headers por compute capability.** Kernels JIT da DeepSeek; nomeados por cc exata (sm121) — daí o alias.
93. **b12x.** Biblioteca de kernels CuTeDSL para SM120/121 usada para MoE W4A16 (route packing por expert, tiles, blocks/SM), mHC, MLA comprimida/esparsa e projeção WO.
94. **Aritmética de ponteiros e overflow int32→int64.** `block_idx × stride` passa de 2³¹ com dezenas de milhares de blocos a 1M de contexto (`cache_utils.py`).
95. **Streams CUDA auxiliares e sobreposição de GEMMs.** 3 streams para rodar em paralelo as GEMMs de entrada do compressor/indexer (`model.py:1700-1709`).
96. **Sincronização host-device e por que evitá-la no hot path.** `.item()`, `.tolist()`, `.cpu()` param a GPU; #31 v2, #175 e a máscara GPU de rejeitados existem para eliminar syncs.

### 5.9 Operação, segurança e qualidade

97. **Patches source-exact, fail-closed, atômicos e idempotentes.** Fixar SHA-256 do alvo, montar a pós-imagem em memória, `compile()`, publicar por `os.replace` no mesmo diretório com `fsync`, re-verificar e fazer rollback; recriar containers (não reiniciar) para desfazer.
98. **Abliteration (projeção de direção de recusa).** `y ← y − λ(y·v)v` no residual pós-atenção de camadas 10–42 com λ = 3,5; direção de 18 KiB baixada de repo com termos de uso.
99. **Autenticação por API key e exposição de segredos.** Uma flag `--api-key` com N chaves; redação no log; argv/env ainda visíveis; rotas sem chave (`/invocations`, `/tokenize`).
100. **Metodologia de benchmark.** Decode-only window, `usage.completion_tokens` (MTP emite chunks multi-token), prefixos únicos por requisição (anti prefix cache), medianas de ≥ 8 trials (ruído ±10 tok/s), uma variável por boot, gates de boot (FULL 6/6, sem "Truncating", KV, MemAvailable).
101. **Avaliação de qualidade: RULER-lite, context-garble sweep, tool battery, GSM8K, gate de 145 casos.** Usadas para aprovar mudanças que podem alterar numerics (SP indexer, MXFP4 indexer, RoPE SWA).

---

## 6. Receita passo-a-passo (reconstruir a solução do zero, com o raciocínio)

> Foco nesta parte da receita: patches/overlay/kernels. Passos de rede e launcher aparecem só no nível necessário para entender as decisões.

### Passo 0 — Entender o gargalo antes de mexer em qualquer coisa
- Decode c=1 é limitado por bytes de pesos por passo (~16,5–17 GB/rank em TP=2); spec decode aumenta tokens por passo; concorrência aumenta agregado. Prefill é compute-bound (MoE W4A16 + indexer quadrático). Contexto de 1M é limitado por KV (MLA latente replicado em todo rank).
- Decisão: TP=2 (um nó por rank) + DSpark + KV FP8 comprimido + scheduler que protege decode.

### Passo 1 — Hardware/rede/host
- 2× DGX Spark, ConnectX-7, NCCL sobre RoCE com HCA/GID corretos; MTU 9000 (RoCE 4096), anéis 8192, sem IPs duplicados na mesma sub-rede entre PFs (fable §3.3).
- Liberar memória do host (unified memory!): sem sessão gráfica, sem vazamentos (polkit), earlyoom desativado; cada GB de host liberado vira KV.
- Persistir caches JIT (Triton/TileLang/b12x/DeepGEMM/FlashInfer) em volume para nunca compilar no meio do serviço.

### Passo 2 — Imagem
- Usar a imagem Anemll pinada por digest (vLLM `752a3a504`, FlashInfer 0.6.15, b12x 0.15.3, xgrammar 0.2.3) nos dois nós — os hotfixes são source-exact e recusam qualquer outra versão.
- Alternativa didática: construir o Stage-C (`build-dspark-vllm-runtime.sh`: overlay → Stage A → B → C) para estudar a implementação de referência do DSpark e as variáveis `VLLM_DSPARK_*`.

### Passo 3 — Pesos
- Baixar Vision-Exp na revisão pinada nos dois nós (ou NFS a partir do head); manter `encoding/encoding_dsv4.py` (é o chat template).
- Carregar corretamente: experts MXFP4 (`w13` como `[w1,w3]`), escalas E8M0 como uint8, `mtp.*` só no drafter (alvo pula `mtp.`), `attn_sink` fatiado por rank, `gate.bias → e_score_correction_bias`, `bias_vl` para visão.

### Passo 4 — KV cache
- Escolher `nvfp4_ds_mla` sabendo que no DSV4 é o envelope FP8 de 584 B (448 FP8 NoPE + 128 bf16 RoPE + 8 escalas UE8M0), e aplicar **#22** para usar o kernel FP8 rápido (senão 1 tok/s a 600K).
- `--block-size 256` para que C4 (64 linhas) e SWA (64 tokens) tenham páginas de 37.376 B (requisito do alocador híbrido).
- Garantir aritmética int64 nos kernels de cache (1M de contexto) e bounds-checks na conversão de índices.
- Prefix cache híbrido: deixar a SWA encolher o hit (#26 v2) e usar `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`.
- Opcional com gate de qualidade: MXFP4 indexer cache (+ alias DeepGEMM sm121); design futuro de linha de 292 B (item8 opção A) para +65 % de tokens físicos.

### Passo 5 — Atenção esparsa no SM12x
- Backend SM120 com cache duplo (SWA + comprimido), cabeças padded a 16/32/64/128, scratch split-K emprestado do workspace e reservado no dummy run.
- Backports de eficiência: #50298 (workspace para índices combinados), #48957 (pular C128 vazio, só fora de FULL graph), #49486 com guard #52492 (atalho do indexer ≤ 2048 tokens, nunca em captura), #48407 dormente, #133 (sem especialização por alinhamento).
- Se houver travamentos estocásticos no decode com muitas linhas: #141 (fatiar em 64).
- Para prefill longo: SP indexer (merge top-k exato via all-gather) — medir; ganho real foi ~6 % a 128K.

### Passo 6 — MoE
- `--moe-backend flashinfer_b12x` → `B12xExperts` W4A16 (formato `fp4_e8m0_k32`), liberar pesos fonte após preparar o formato do kernel, scratch via `workspace2`, sem EP.
- Não usar TC-decode (−15 % medido); considerar no futuro MXFP4×MXFP8 em tensor cores ou re-quantização para NVFP4 (com gate de qualidade).

### Passo 7 — DSpark
- Método `dspark`, drafter = estágios `mtp.*` como camadas virtuais após a 42; features das camadas-alvo (média dos fluxos mHC) → `main_proj` → ring buffer de 128 por requisição.
- Garantir corretude com concorrência: slots estáveis por `req_id` (Patch 1), caminho ragged via `query_start_loc` (Patch 2/2b), máscara de rejeitados na GPU, clamp de slot, guard para lotes não-uniformes.
- Configurar k: na lane A a regra de divisibilidade força k=6 com `n_predict=3`; o bloco treinado é 5 — com o hotfix block-k, k=5 dá passo mais barato e mesma aceitação.
- Prefix cache: aplicar o DSpark SWA prefix fix para que prompts repetidos não degenerem.
- CUDA graph do drafter PIECEWISE com buffers persistentes; capture size do alvo `((seqs·(k+1)+7)/8)·8` (48 em 6×6), senão c=6 roda eager (−12 %).
- Probabilístico vs greedy: greedy empatou; mantido probabilístico (rejection sampling completo).
- Markov head replicada: correta, mas medida pior — manter shardada.

### Passo 8 — Scheduler e runtime
- `--long-prefill-token-threshold 1024` + #27 (`DSPARK_MAX_INFLIGHT_PREFILLS=1` ou `2` conscientemente) + #43 (piso de decode).
- `--async-scheduling` (e saber que ele gera placeholders que interagem com gramática/tool calls).
- IPC: #79 (`busy_loop_s` 2 ms) e #117 (recheck 5 s, liberação em `finally`).
- Timeouts: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`; caches JIT persistidos; warmups de forma (overlay `kernel_warmup.py`, `boot-shape-warmup.sh`).
- Não habilitar chunks maiores (adaptive/2048) enquanto a memória do host não for resolvida.

### Passo 9 — Serving (texto, raciocínio, tools, API)
- Instalar o encoder do checkpoint e corrigir o mapeamento de esforço; #21 (dict args), #55 (truncamento → `length`), suppress-stops (stops só após `</think>`).
- Opcionais conforme o cliente: #52/#120 (turno assistant final), #144 (prefix cache entre esforços), #191 (contrato de tool call com fallback thinking-off), #136+#210 (XGrammar), DSML recovery, #31 (budget de raciocínio na GPU), Responses store limitado, #138/Codex.
- Autenticação: `DSPARK_API_KEYS` (uma flag) + redação no log + controle de rede para rotas sem chave.

### Passo 10 — Visão
- Hook que constrói ViT+Aligner, processor que expande placeholders em blocos alinhados a 4 tokens, hash salgado por `num_tokens`, roteamento `bias_vl` sem hash para linhas de imagem com classificação uma vez por forward, imagens só em `user`.

### Passo 11 — Verificação
- CPU: `scripts/ci-validate.sh` e as suítes `tests/`/`scripts/test-*.py`.
- Boot: nos dois ranks, `APPLIED`/`patched` de cada hotfix, "Capturing CUDA graphs (FULL) 6/6", sem "Truncating", KV esperado, MemAvailable ≥ ~3 GB.
- Live: smoke, quality probe (391/Tokyo), `spec-acceptance.py`, `bench-miaai.py` c=1/c=6 (≥ 8 trials), `bench-ttft.py` 8K/32K/128K, RULER-lite, gate de 145 casos para tool calls; `/health` sozinho não prova nada (#141).

### Passo 12 — Tuning (uma variável por boot, medianas, mesmo dia)
- Candidatos já medidos e resultado: capture 48 (ganha), inflight 2 (ganha fairness), SP indexer (ganho pequeno a 128K), Markov replicada (perde), TC decode (perde), adaptive/2048 (memória), greedy (empate), logits 512 MB (empate), single vs dual PF (empate).
- Pendências de engenharia: MoE FP4 em tensor cores, KV FP4 real (292 B/linha), cobertura de warmup, NCCL tuning, profiler trace, caça ao dono da memória do host.

---

## 7. Inconsistências e pontos de atenção encontrados (para não ensinar errado)

1. **`nvfp4_ds_mla` não é KV de 4 bits** no DSV4: é o layout FP8 de 584 B; o "NVFP4" só muda o despacho (e, na lane A, a `alignment` do spec: 584 para nvfp4 vs 576 para fp8, segundo `item8:11`). O layout FP4 de 416 B foi abandonado (Stage B → C).
2. **Backports "not yet applied"** (`docs/vllm-027-new-patches.md:13-14`) na verdade estão aplicados por default (`docker-compose.dspark.yml:448`).
3. **#50312 e `use_eagle()`**: o docstring diz que `use_eagle()` é falso para dspark, mas a fixture pinada mostra que inclui `"dspark"` (`speculative-752a3a504-stock.py:1242`) → o buffer MTP pode continuar alocado. Não verificável sem o log.
4. **R1 vs R2 no TP=3**: `patches/dsv4_tp_pad.py` (R1) contradiz `patches/tp3/apply_tp3_patch.py` e `docs/TP3.md` (R2, o aplicado). O módulo que vai para a imagem é o embutido no applier.
5. **Guard de não-uniformidade vs caminho ragged** no `DSparkProposer` (overlay): o guard (`:794-810`) devolve zeros para lotes com linhas não divisíveis pelo batch, antes do caminho ragged — este só cobre lotes divisíveis porém desiguais (interpretação da ordem do código).
6. **RoPE YaRN em camadas SWA**: presente tanto na lane A (corrigível pelo RoPE SWA fix, default off) quanto no overlay Stage-C (sem correção), que também muta `config.rope_parameters` compartilhado.
7. **Números "de ouro" vs números do dia**: 14 ago (c=1 62–83, c=6 162) vs 2 set (c=1 ≈ 55, c=6 ≈ 139) sem knob que explique; tratar os de agosto como melhor caso (`README.md:338-341`).
8. **`GLM-NEW-REPORT.md`** descreve a lane 0731 antiga (`MTP_NUM_TOKENS=5`, util 0,80) e contém um artefato de texto ("opens `未成年`" onde o contexto indica `<think>`); não usar como fonte de defaults atuais.
9. **Warmup de forma**: `boot-shape-warmup.sh` assumia `+6` (k=5) na chave `next_pow2(scheduled_tokens + 6)`; para k=6 seria `+7` (`fable5-1 §3.10`).
10. **`docs/ENVS.md` "Recommended defaults"** cita util ≈ 0,80, mas o compose default é 0,835 (`GPU_MEMORY_UTILIZATION_TEXT`).
11. **Valores de `dspark_target_layer_ids`, `dspark_noise_token_id`, `swiglu_limit`, `hc_sinkhorn_iters`** e a configuração real do ViT **não estão no repo** (só defaults de código).

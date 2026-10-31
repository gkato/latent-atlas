# vLLM: referência conceitual e de parâmetros para as receitas 2x/3x/4x DGX Spark (PT-BR)

Data de referência: 2026-09-28. Escopo: tudo que as três receitas em `scratchpad/repos/` passam para o vLLM
(argumentos de CLI, JSON de configuração, variáveis `VLLM_*`, NCCL/GLOO) e os conceitos do engine necessários
para entender por que cada valor foi escolhido.

Receitas cobertas (abreviações usadas no documento):

| Sigla | Repositório | Modelo servido | Topologia padrão |
|---|---|---|---|
| **DS** | `DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (MXFP4 experts + FP8 bloco 128) | TP=2, 2 nós, spec DSpark k=6, 1M ctx |
| **GLM** | `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (experts EXL3 4 bpw) | TP=2, 2 nós, spec DFlash2 k=7, 850k ctx (TP3/TP4 opcionais) |
| **QW** | `Qwen3.8-Flash-Next-Dual-DGX-Sparks` | `nvidia/Qwen3.8-Flash-Next-NVFP4` | TP=2 + EP, 2 nós, spec MTP k=3, 262k ctx (TP1 e lane v0.30 opcionais) |

Convenções de fonte:

- `[DS: caminho]`, `[GLM: caminho]`, `[QW: caminho]`: arquivo dentro do repositório da receita (fonte primária para "qual valor a receita usa").
- `[Uxx]`: documentação ou código upstream do vLLM (lista numerada na seção 16).
- **[inferência]**: conclusão minha a partir das fontes, não afirmada literalmente por elas.
- **[incerto]**: fontes conflitantes ou não verificadas; confira no boot log da sua imagem antes de confiar.

Regra de ouro: as três receitas rodam **forks/imagens fixadas**, não o vLLM "latest". Um default ou nome de flag
da documentação atual pode não existir (ou ter outro default) na imagem da receita. Quando houver divergência,
este documento indica o valor da versão fixada e o da v0.30.0.

---

## Sumário

1. [Versões usadas pelas receitas e linha do tempo upstream](#1-versões-usadas-pelas-receitas-e-linha-do-tempo-upstream)
2. [Arquitetura do engine V1](#2-arquitetura-do-engine-v1)
3. [Memória: PagedAttention, pool de KV e dtypes de KV](#3-memória-pagedattention-pool-de-kv-e-dtypes-de-kv)
4. [Hybrid KV cache manager (atenção + mamba/linear + SWA + drafter)](#4-hybrid-kv-cache-manager)
5. [Automatic Prefix Caching (APC)](#5-automatic-prefix-caching-apc)
6. [Scheduler: continuous batching, chunked prefill e token budget](#6-scheduler-continuous-batching-chunked-prefill-e-token-budget)
7. [CUDA graphs e torch.compile](#7-cuda-graphs-e-torchcompile)
8. [Decodificação especulativa](#8-decodificação-especulativa)
9. [Structured outputs, tool calling e reasoning](#9-structured-outputs-tool-calling-e-reasoning)
10. [Distribuído: multi-nó, executores, NCCL/GLOO](#10-distribuído-multi-nó-executores-ncclgloo)
11. [Paralelismo: TP, PP, DP, EP, DCP/PCP, DBO](#11-paralelismo-tp-pp-dp-ep-dcppcp-dbo)
12. [Quantização e backends de kernel (inclui como o EXL3 foi plugado)](#12-quantização-e-backends-de-kernel)
13. [Attention backends](#13-attention-backends)
14. [Registro de modelos, plugins out-of-tree e overlays](#14-registro-de-modelos-plugins-out-of-tree-e-overlays)
15. [Referência de argumentos por receita](#15-referência-de-argumentos-por-receita)
16. [Referência de variáveis de ambiente](#16-referência-de-variáveis-de-ambiente)
17. [Rodar local no Mac: contrato OpenAI do vLLM, vllm-metal (MLX) e mlx_lm.server](#17-rodar-local-no-mac-contrato-openai-vllm-metal-e-mlx_lmserver)
18. [Roteiro (lado vLLM) para montar uma receita de um modelo recém-lançado](#18-roteiro-lado-vllm-para-montar-uma-receita-de-um-modelo-recém-lançado)
19. [Fontes](#19-fontes)

---

## 1. Versões usadas pelas receitas e linha do tempo upstream

### 1.1 O que cada receita realmente executa

| Receita | Imagem | vLLM | Pilha relevante |
|---|---|---|---|
| DS | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a8394849…` | `0.25.2.dev0+g752a3a504.d20260714` (fork Anemll, commit upstream `752a3a504485790a2e8491cacbb35c137339ad34`) com backports de PRs DeepSeek-V4 da v0.27.0 | torch 2.11.0+cu130, CUDA 13.0, FlashInfer 0.6.15, b12x 0.15.3, triton 3.6.0, TileLang 0.1.9, NCCL 2.30.7 (pynccl) / 2.28.9 (torch), xgrammar 0.2.3 [DS: docs/CLAUDE/fable5-1-report.md §1; docs/vllm-027-new-patches.md] |
| GLM | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, construída `FROM vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` | commit `487ecf187d3dfe74d2cf6119a92881dba403c219` | arm64, CUDA 13.0, ExLlamaV3 `c5d9c657` (0.0.43), InstantTensor 0.2.0, NCCL de host 2.30.7 opcional via `LD_PRELOAD` [GLM: README.md "What runs", Dockerfile] |
| QW (padrão) | `vllm/vllm-openai:qwen38-flash-next` (`sha256:d464f3b4…`, build "day-0" NVIDIA) | build de desenvolvimento (sem versão numerada no repo) | FlashInfer 0.6.17 [QW: README.md; docs/CLAUDE/fable5-1-report.md] |
| QW (lane `start-v030.sh`) | `vllm/vllm-openai:v0.30.0` | **v0.30.0 estável** | KV BF16 por padrão, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` [QW: start-v030.sh; CHANGELOG 2026-09-25] |

Consequência prática: das três, só a lane v0.30 da QW corresponde a um release upstream. As demais são
snapshots de `main` com patches. Ao ler a doc upstream, use-a para **conceitos**; para **defaults**, confie no
boot log (`non-default args`, `Available KV cache memory`, `Capturing CUDA graphs`).

### 1.2 Linha do tempo upstream relevante (releases de 2026)

| Release | Data | Mudanças que afetam estas receitas |
|---|---|---|
| v0.28.0 | 2026-08-26 | Default de `max_num_batched_tokens` sobe de 8192 para 16384 (#51726, aplicado quando a GPU tem memória suficiente); captura de CUDA graph em Blackwell até 1024 (#49390); prefix caching ligado por padrão para modelos Mamba (#50991); DFlash2 (#52816); verificação DSpark com agendamento por confiança (#47808); async scheduling auto-ligado para draft models (#48341); orçamento adaptativo de tokens especulativos (#51725); `calculate_kv_scales` removido; bitsandbytes virou plugin externo [U3, U38] |
| v0.29.0 | 2026-09-09 | **Model Runner V2 vira padrão** (MRV1 deprecado, remoção prevista na v0.32); `NONE_HASH` determinístico no prefix cache (não precisa mais fixar `PYTHONHASHSEED`); `prefix_cache_retention_interval` vira argumento de CLI (0 para SWA/SSM; retenção densa restaurada automaticamente para modelos híbridos/EAGLE/MTP); `--max-num-queued-reqs` e `--max-num-queued-tokens`; all-reduce FlashInfer padrão em grupos TP (opt-out `VLLM_ALLREDUCE_USE_FLASHINFER=0`); `--per-request-spec-decode-metrics`; `VLLM_TEST_FORCE_FP8_MARLIN` removido (use `--linear-backend`/`--moe-backend`) [U2] |
| v0.30.0 | 2026-09-22 | Suporte oficial a GLM-5.3-Flash, DeepSeek-V4-Flash-Vision-Exp e DeepSeek-V4.1-Flash; **CUDA graphs "breakable" passam a exigir opt-in** `VLLM_USE_BREAKABLE_CUDAGRAPH=1`; modelos não compilados caem para `FULL_DECODE_ONLY`; `disable_eagle_block_drop` opt-in; camadas SWA usam o block size primário; modo Mamba `all` deprecado; **env `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` removida** (use o argumento); alias `use_fp4_indexer_cache` removido (use `indexer_kv_dtype`); NVFP4 W4A4 preferido a weight-only em SM120/121; `--load-format ipc_cache`, `--engram-config`, `--sparse-indexer-topk-backend`, `--attention-backend B12X_ATTN`; endpoints de scale-out só com `--enable-scale-out`; `linear_backend_per_quant` [U1] |

**[incerto]** A página de releases resumida por ferramenta mostrou "2024" nas datas; a busca e o próprio conteúdo
(modelos de 2026) confirmam 2026. Use as datas acima.

---

## 2. Arquitetura do engine V1

### 2.1 Processos e componentes

O V1 separa a pilha em processos para que CPU (tokenização, detokenização, parsers) não bloqueie a GPU [U14, U19]:

```
                     nó head (NODE_RANK=0)                                   nó worker (NODE_RANK=1, --headless)
┌──────────────────────────────────────────────────────────┐         ┌──────────────────────────────────────┐
│ API server (FastAPI/uvicorn)                             │         │  (sem API server, sem EngineCore)    │
│  - OpenAIServingChat/Completion/Responses                │         │                                      │
│  - chat template ou tokenizer-mode, tokenização          │         │                                      │
│  - output processor: detokenização incremental, stop,    │         │                                      │
│    reasoning parser, tool parser                         │         │                                      │
│  AsyncLLM ──EngineCoreClient── ZMQ ─┐                    │         │                                      │
│                                     ▼                    │         │                                      │
│ EngineCore (processo próprio, busy loop)                 │         │                                      │
│  Scheduler ── KVCacheManager ── KVCacheCoordinator       │         │                                      │
│     └ SingleTypeKVCacheManager por grupo ── BlockPool    │         │                                      │
│  StructuredOutputManager (xgrammar)                      │         │                                      │
│  Executor = MultiprocExecutor (mp)                       │         │                                      │
│     └ MessageQueue (shm local / ZMQ remoto) ─────────────┼── TCP ──┼─► Worker rank 1 (GPU 0 do nó)        │
│  Worker rank 0 (GPU 0 do nó)                             │         │     GPUModelRunner / MRV2            │
│     GPUModelRunner / MRV2                                │◄═ NCCL ═┼═►  (all-reduce TP via RoCE/IB)      │
└──────────────────────────────────────────────────────────┘         └──────────────────────────────────────┘
```

| Componente | Papel | Arquivo (árvore v1) |
|---|---|---|
| API server | HTTP OpenAI-compatível, tokenização, parsers, streaming | `vllm/entrypoints/openai/…` (`python -m vllm.entrypoints.openai.api_server` deprecado na v0.29; use `vllm serve`) [U2] |
| `AsyncLLM` | front-end assíncrono do engine no processo do API server | `vllm/v1/engine/async_llm.py` |
| `EngineCoreClient` | envia `EngineCoreRequest` e recebe `EngineCoreOutputs` por ZMQ | `vllm/v1/engine/core_client.py` |
| `EngineCore` / `EngineCoreProc` | loop: `schedule()` → `execute_model()` → `update_from_output()` | `vllm/v1/engine/core.py` |
| `Scheduler` | decide quantos tokens cada request processa no passo; admite/preempta | `vllm/v1/core/sched/scheduler.py` (cópia do fork DS em `[DS: recipe/overlay/vllm/v1/core/sched/scheduler.py]`) |
| `KVCacheManager` / `KVCacheCoordinator` / `SingleTypeKVCacheManager` / `BlockPool` | alocação de blocos, prefix cache, grupos híbridos | `vllm/v1/core/kv_cache_manager.py`, `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py` (citados linha a linha em `[GLM: docs/DESIGN-apc-per-group-retention.md]`) |
| Executor | `uni` (1 processo), `mp` (multiprocessing), `ray`, `external_launcher` | `vllm/v1/executor/…` [U11] |
| Worker | um processo por GPU; inicializa device, NCCL, carrega o modelo | `vllm/v1/worker/gpu_worker.py` |
| ModelRunner | prepara tensores (block tables, slot mapping, posições), metadados de atenção, CUDA graphs, sampler, proposer de spec | MRV1 `vllm/v1/worker/gpu_model_runner.py` (cópia em `[DS: recipe/overlay/vllm/v1/worker/gpu_model_runner.py]`); MRV2 `vllm/v1/worker/gpu/` [U20] |

Contagem de processos: `A` API servers (default = DP), `DP` EngineCores, `DP×PP×TP` workers, mais um coordenador
de DP se DP>1 [U14]. Nas receitas (DP=1, PP=1, TP=2 em 2 nós): 1 API server + 1 EngineCore no head, 1 worker em
cada nó. A doc de otimização pede pelo menos `2 + N` núcleos físicos porque "o EngineCore roda um busy loop e é
particularmente sensível a falta de CPU" [U21].

### 2.2 Ciclo de vida de um passo (step)

1. **Entrada**: o API server renderiza o prompt (Jinja `--chat-template`, ou encoder Python via `--tokenizer-mode`,
   ex. `deepseek_v4`), aplica defaults de amostragem (`--generation-config`, `--override-generation-config`),
   anexa parâmetros de structured output e envia o request ao EngineCore.
2. **`schedule()`**: o scheduler percorre primeiro os requests `RUNNING` e depois os `WAITING`, atribuindo
   `num_new_tokens` a cada um dentro do `token_budget` (seção 6) e pedindo blocos ao `KVCacheManager`
   (`allocate_slots`). O resultado é um `SchedulerOutput`.
3. **`execute_model()`**: o `MultiprocExecutor` faz broadcast do `SchedulerOutput` para todos os workers
   (`MessageQueue` em shared memory para leitores locais e ZMQ para leitores remotos; é esse caminho que o hotfix
   `[DS: patches/hotfix-vllm-issue117-shm-ring-buffer.py]` corrige). Cada worker monta o batch, despacha para um
   CUDA graph (se o shape foi capturado) ou roda eager, executa o forward com all-reduces NCCL, amostra e, se houver
   spec decode, verifica e gera novos rascunhos.
4. **`update_from_output()`**: o scheduler registra tokens aceitos, libera blocos de requests terminados, avança
   máquinas de estado de gramática e devolve `EngineCoreOutputs` ao API server, que detokeniza, aplica stops e
   parsers e faz streaming.

Com **async scheduling** o scheduler já agenda o passo N+1 enquanto a GPU executa o passo N (com "placeholders" de
saída); isso esconde o custo de CPU do scheduler [U20]. O MRV2 (padrão desde v0.29) foi desenhado para esse modo:
estado persistente na GPU, preparação de entradas por kernels Triton, sampler em Triton e nenhum ponto de
sincronização CPU-GPU no loop principal [U20].

Timeouts ligados a esse ciclo (todos com defaults upstream verificados em `vllm/envs.py` v0.30.0 [U13]):

| Variável | Default | Significado | Receitas |
|---|---|---|---|
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 300 | prazo da RPC `execute_model`/`sample_tokens` no executor mp (TP>1) | DS e GLM: 1800, porque JIT de CuTeDSL/TileLang/Triton no meio do serviço pode passar de 300 s e derrubar o EngineCore [DS: docker-compose.dspark.yml; GLM: start.sh] |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 600 | espera pelos EngineCores no boot | GLM: `READY_TIMEOUT` = 3600 |
| `VLLM_RPC_TIMEOUT` | 10000 ms | operações simples do cliente ZMQ | não alterado |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | 16 | tamanho de chunk da message queue | não alterado |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | 0 | API server continua vivo se o engine morrer | não alterado |

Observação: o watchdog do NCCL/PyTorch (600 s) é independente e continua valendo [GLM: start.sh].

---

## 3. Memória: PagedAttention, pool de KV e dtypes de KV

### 3.1 PagedAttention e block tables

- O KV de cada camada é dividido em **blocos** de `block_size` tokens. Um request não precisa de memória contígua:
  cada request tem, por grupo de KV, uma **block table** (lista de IDs físicos) e cada token recebe um **slot**
  (`block_id * block_size + offset`) via `slot_mapping` no passo em que é escrito. Kernels de atenção paginados
  leem o KV seguindo a block table. (Conceito do PagedAttention, Kwon et al., SOSP 2023.)
- O bloco 0 é reservado como **null block** (placeholder), por isso o GLM conta "643 IDs, 642 utilizáveis"
  [GLM: README.md "What the KV cache boot line means"].
- Todos os grupos de KV compartilham **um único `BlockPool` com IDs globalmente únicos** [GLM: docs/DESIGN-apc-per-group-retention.md §1].
- Default de `block_size`: `None`, resolvido para 16 quando o backend não impõe outro [U6]. Backends e modelos
  forçam valores maiores (MLA esparso 64/256, modelos híbridos 1600-3584; ver seção 4).

### 3.2 Como o tamanho do pool é decidido

1. O worker carrega pesos e roda um **profiling** com o maior batch (`max_num_batched_tokens`) para medir
   ativações; opcionalmente estima a memória dos CUDA graphs (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, padrão
   ligado desde v0.21 [DS: recipe/overlay/vllm/envs.py]).
2. KV disponível ≈ `gpu_memory_utilization × memória total do device − pesos − pico de ativações − reservas`.
   Com `--kv-cache-memory-bytes` o valor é fixado por GPU e **`gpu_memory_utilization` é ignorado para o KV**
   [U6; GLM: README.md tabela TP4 "this flag ignores GPU_MEM_UTIL"].
3. O boot imprime `Available KV cache memory: X GiB`, `GPU KV cache size: N tokens` e
   `Maximum concurrency for L tokens per request: Y x`.

Leitura correta dessas linhas (ponto em que as três receitas insistem):

- `--max-model-len` e `--max-num-seqs` são **tetos, não reservas**. O limite real é `soma(tokens vivos) ≤ pool`
  [DS: README.md "How the KV cache works"].
- Em modelos híbridos, `GPU KV cache size` é `max_concurrency × max_model_len`, **não** o tamanho do pool; a
  capacidade de conversas em cache depende de quantos IDs cada segmento custa em cada grupo. O overlay
  `patch_kv_capacity_log.py` imprime o detalhamento por grupo (ex.: 642 IDs utilizáveis, 38 IDs por segmento de
  3584 tokens, ≈57k tokens de histórico em cache) [GLM: README.md]. O vLLM stock não loga `num_blocks` nem
  blocos por request (issue vllm#54662 citada pela receita).

**GB10 / DGX Spark (memória unificada, UMA):** a "memória do device" são os ~121,7 GiB visíveis do pool LPDDR5X
compartilhado com o SO e o page cache. Por isso:

- `gpu_memory_utilization` fica entre 0,75 e 0,865 nas receitas (upstream: 0,92 na v0.30.0 [U6]). Cada 0,01 ≈
  1,2 GiB de folga para o host [GLM: README.md tabela de knobs].
- Em GPU integrada, a checagem de boot do vLLM compara com `MemAvailable`; o GLM derruba o page cache antes do
  `docker run` e espera `MemAvailable ≥ GPU_MEM_UTIL × total` (`GLM53_HOST_MEM_HYGIENE=1`) [GLM: start.sh].
- A QW libera as páginas do próprio checkpoint com `posix_fadvise(DONTNEED)` antes de lançar [QW: README.md].
- DS recomenda desligar o `earlyoom` [DS: README.md].

### 3.3 Dtypes de KV (`--kv-cache-dtype`)

Valores aceitos na v0.30.0: `auto, bfloat16, float16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, nvfp4_ds_mla,
nvfp4, nvfp4_4over6, fp8_per_token_head, int4_per_token_head, int8_per_token_head, turboquant_k8v4,
turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc` [U6, U4].

| Valor | O que é | Uso nas receitas |
|---|---|---|
| `auto` | dtype do modelo (normalmente BF16) | QW lane v0.30 (o QSA da v0.30 só aceita BF16) e QW TP1; draft DFlash2 do GLM (`"kv_cache_dtype":"auto"` no spec config) |
| `fp8` (= e4m3 no CUDA) | KV em FP8; escalas do checkpoint (`k_scale`/`v_scale`) ou 1,0 se ausentes; calibração recomendada via llm-compressor [U30 quantized_kvcache] | QW padrão (exige `files/patch_qsa_fp8_kv.py` na imagem day-0; 3.652.200 tokens vs ~1,5M em BF16); GLM (`KV_CACHE_DTYPE=fp8`, que para o MLA esparso vira o layout empacotado `fp8_ds_mla`) |
| `fp8_ds_mla` | layout empacotado de MLA esparso (DeepSeek-style): parte NoPE em FP8 + escalas + parte RoPE em BF16 por token por camada | GLM: registro de 656 B (512 NoPE + 16 B escalas + 128 B RoPE) no backend `FLASHINFER_MLA_SPARSE_SM120`; o GLM-5.3 é NoPE MLA (`qk_rope_head_dim=0`), e o overlay preenche o RoPE com zeros para caber no layout [GLM: README.md "Why the overlay exists"] |
| `nvfp4_ds_mla` | variante "NVFP4" do layout acima | DS. A auditoria da receita mostra que, nesta imagem, o registro é **byte-idêntico ao `fp8_ds_mla`** (584 B/linha: 448 B NoPE FP8 + 128 B RoPE BF16 + 8 B escala); ainda não existe KV de 4 bits real no GB10 [DS: docs/CLAUDE/fable5-1-report.md item 8] |
| `nvfp4` e `*_per_token_head`, `turboquant_*` | KV de 4 bits/por cabeça genéricos | não usados |

Detalhes que costumam enganar:

- Em MLA o KV **não é fatiado por TP** (todas as ranks guardam o latente inteiro); com TP=2 o KV de MLA é
  duplicado. É exatamente isso que o Decode Context Parallel resolve (seção 11) [U24]. O TP=3 da DS é mais lento em
  prefill "porque no MLA do DeepSeek cada rank guarda o KV latente inteiro e o indexer" [DS: README.md TP3].
- Mudar o dtype do KV não muda o estado Mamba/linear: esse tem `--mamba-cache-dtype` (conv + ssm) e
  `--mamba-ssm-cache-dtype` (só ssm) [U6].
- `--kv-cache-dtype` do draft pode ser diferente do target: campo `kv_cache_dtype` do `--speculative-config`
  (se `None`, herda) [U12].

---

## 4. Hybrid KV cache manager

### 4.1 Conceito

Modelos que misturam tipos de camada (atenção completa, sliding window, Mamba/atenção linear, MLA, indexers,
camadas do drafter) têm necessidades de KV diferentes. O vLLM agrupa camadas em **KV cache groups**: cada grupo
contém camadas do mesmo tipo, que precisam do mesmo número de blocos por request [U16]. Hierarquia:
`KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager` (um por grupo) [U16].

Restrições centrais [U16]:

- **Página física única**: todos os grupos compartilham o mesmo tamanho de página
  (`block_size × kv_hidden_size × num_layers_do_grupo`). Quando os tamanhos divergem (atenção vs estado Mamba), o
  vLLM **aumenta o `block_size` da atenção** até `block_size × kv_hidden_att ≥ state_size_mamba` e faz padding do
  estado Mamba. É por isso que modelos híbridos têm blocos de atenção enormes.
- **Cache hit híbrido = interseção**: acha o maior hit da atenção completa (varredura esquerda→direita) e depois o
  maior hit da sliding window dentro dele (direita→esquerda), para que o hit SWA também seja hit de atenção
  completa [U16].
- Alinhamento do hit: o coordenador usa o **mínimo múltiplo comum dos block sizes dos grupos** ("scheduler block"),
  a menos que todos os managers aceitem hits finos [GLM: docs/DESIGN-apc-per-group-retention.md §2].
- `--disable-hybrid-kv-cache-manager` força KV uniforme em todas as camadas (existe no `SchedulerConfig` [U7]; nenhuma receita usa).

### 4.2 Modos de cache Mamba (`--mamba-cache-mode`)

`none` (padrão), `align`, `all` [U6]:

- `all`: guarda estado em todo múltiplo de `block_size`. Deprecado na v0.30 (cai para MRV1) [U1].
- `align`: guarda estado nos limites de passos do scheduler, alinhados ao bloco. É o modo que Qwen3.8 e GLM-5.3
  usam (a QW registra que o modelo força `align` e que `"all"` levanta erro) [QW: docs/CLAUDE/fable5-1-report.md §1].
- `--mamba-block-size`: bloco próprio para o cache Mamba, múltiplo de 8 [U6].

### 4.3 Geometria real em cada receita

| Receita | Grupos (boot log / docs) | Block sizes | Observações |
|---|---|---|---|
| DS | 1× `MLAAttentionSpec` + 3× `SlidingWindowMLASpec` | 256 (MLA) e 64/4/8 (SWA) | `--block-size 256` explícito; camadas com compressão C4A/C128A e SWA de 128 tokens; o hotfix do issue #26 deixa a SWA encolher o hit comum e `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` esparsifica checkpoints SWA [DS: docs/CLAUDE/fable5-1-report.md App. A; docker-compose.dspark.yml] |
| GLM | 7 grupos: g0 `UniformTypeKVCacheSpecs` (11 MLA + 11 indexer, `FullAttentionManager`, 3584); g1 `KpoolTailSpec` (scratch do indexer, fora do prefix cache); g2-g5 `MambaSpec` align (camadas KDA, atenção linear), 3584; g6 `SlidingWindowSpec` do drafter DFlash2 (janela 2048, bloco 64, `page_size_padded = mla_page`, grupo "eagle") | 3584 / 4 / 3584 / 64 | O "padded slot-share" faz a camada i do drafter co-possuir o tensor MLA i: zero bytes extras por bloco, mas um ID de bloco (com preço de página MLA inteira) a cada 64 tokens de drafter; é a causa do "flood" de IDs que motivou a retenção por grupo [GLM: docs/DESIGN-apc-per-group-retention.md]. O Dockerfile explica a cadeia de alinhamento: página KDA leva o bloco a 1664; o indexer exige múltiplo de 256 → 1792; o manager MLA acaba em 3584 [GLM: Dockerfile, comentário inicial]. Documentos de datas diferentes citam 11 ou 12 camadas MLA e 3 ou 4 grupos Mamba (**[incerto]** entre versões do overlay; confie no `[glm53-kv-capacity-log]` do seu boot) |
| QW | atenção esparsa QSA (12 camadas) + GDN (Gated DeltaNet, 36 camadas, estado recorrente tipo Mamba) | bloco de atenção ≈1600 tokens, "padded to the GDN page" (1664 com MTP 3 segundo o patch de block drop) | `--mamba-ssm-cache-dtype bfloat16` reduz pela metade a página Mamba e permite ao vLLM escolher bloco de atenção menor; estado FP32 do checkpoint custa ~0,23 GB/seq por passo [QW: .env.sample, start.sh; files/patch_block_drop.py; docs/CLAUDE/fable5-1-report.md] |

Lição geral **[inferência]**: em modelos híbridos o gargalo de capacidade costuma ser **quantidade de IDs de
bloco** (que custam uma página inteira cada) e não bytes de KV por token. Reduzir `max_model_len` quase não
libera nada, porque a demanda de IDs de Mamba/janela é quase independente do comprimento [GLM: README.md tabela
"Context"].

---

## 5. Automatic Prefix Caching (APC)

### 5.1 Mecânica

- Cada bloco **cheio** recebe um hash que encadeia: hash do bloco pai + IDs dos tokens do bloco + "extra keys"
  (hashes de mídia multimodal, ID de LoRA, `cache_salt`) [U15]. Blocos parciais não entram no cache.
- Algoritmos: `sha256` (padrão, via pickle), `sha256_cbor` (reprodutível), `xxhash`, `xxhash_cbor`
  (`--prefix-caching-hash-algo`) [U15, U6].
- Desde a v0.29 o `NONE_HASH` inicial é determinístico: cache distribuído (LMCache etc.) não precisa mais de
  `PYTHONHASHSEED` fixo [U2]. Na imagem DS (pré-v0.29) o caminho LMCache exporta `PYTHONHASHSEED=0`
  [DS: lmcache/patch-compose-lmcache.py].
- Reference counting impede despejar blocos em uso. Blocos liberados vão para o **final da fila livre em ordem
  reversa** (os últimos blocos, que hash-eiam mais tokens, são despejados primeiro); a evicção tira da cabeça
  (LRU) [U15]. Consequência explorada pelo GLM: um request "one-off" empurra seus blocos para trás do histórico do
  dono e causa a evicção dele; o overlay `patch_apc_no_store.py` adiciona o opt-out de escrita
  `vllm_xargs: {"skip_writing_prefix_cache": 1}` (o stock só tem `skip_reading_prefix_cache`, lado leitura)
  [GLM: README.md "Opting a request out of the prefix cache"].
- `cache_salt` isola namespaces (só requests com o mesmo salt compartilham blocos) [U15].
- `--prefix-match-unit`: menor granularidade (em tokens) em que um hit pode terminar; sem valor = o vLLM resolve
  [U4 stable]. Hits finos parciais chegaram na v0.28 (#50507) [U3]. O GLM avisa que `64` só é válido na geometria
  testada e `512` é inválido ali [GLM: README.md tabela].
- `--enable-prompt-tokens-details` faz a resposta trazer `usage.prompt_tokens_details.cached_tokens`, o jeito
  mais simples de ver o APC funcionando [U5; DS e QW usam; GLM no perfil long-coding].

### 5.2 Interações que as receitas precisaram tratar

1. **EAGLE block drop.** Com drafters estilo EAGLE (MTP, DFlash, DSpark incluídos) o vLLM descarta
   conservadoramente o último bloco que deu hit, porque o drafter escreve um token de lookahead no KV do drafter
   que pode não ser comum a todos os requests daquele prefixo [U36]. Efeito: o segundo turno de uma conversa perde
   um bloco inteiro (1664 tokens na QW). `disable_eagle_block_drop: true` no `--speculative-config` mantém o
   bloco; a saída não muda (o target verifica tudo), só a aceitação pode variar (#53388, merge em 2026-09-01,
   opt-in na v0.30) [U36, U1]. QW liga (`MTP_DISABLE_BLOCK_DROP=1`, backport na imagem day-0): TTFT do turno 1
   de 3,25 s para 0,63 s [QW: CHANGELOG 2026-09-25].
2. **Retenção esparsa de checkpoints (SWA/Mamba).** Guardar estado Mamba/SWA em cada bloco custa IDs; a
   "retention interval" guarda um checkpoint a cada N tokens.
   - Imagens fixadas: env `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (DS=4096; GLM opcional
     `GLM53_APC_RETENTION_INTERVAL`, ex. 14336 no perfil long-coding) e, no overlay GLM,
     `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA` (0 = só limites alcançáveis + evicção prioritária de blocos só
     do drafter; positivo = múltiplo de 3584) [DS: .env.dspark.example; GLM: start.sh, examples/tp2-long-coding.env].
   - Upstream: vira `--prefix-cache-retention-interval` na v0.29 (default 0 = só checkpoints semânticos) e a env é
     **removida na v0.30** [U2, U1, U6]. Quem migrar a DS/GLM para v0.30 precisa trocar env por argumento.
   - Issue vllm#53504: em híbrido Mamba/GDN com MTP, a primeira repetição perde o cache porque a máscara de
     retenção padrão descarta o estado no limite ajustado pelo EAGLE; `interval` explícito contorna [U37]. A QW
     mediu `1664` e não viu ganho em cima do block drop [QW: CHANGELOG 2026-09-25].
3. **Hits alinhados e replay.** Com o `KpoolTailManager` o coordenador desliga hits finos ("Disabling
   fine-grained prefix-cache hits…") e todo hit fica alinhado a 3584 tokens no GLM [GLM: docs/DESIGN-apc-per-group-retention.md §2].

---

## 6. Scheduler: continuous batching, chunked prefill e token budget

### 6.1 Modelo mental do scheduler V1

No V1 não há fases separadas de prefill e decode: cada request tem `num_computed_tokens` e `num_tokens`
(prompt + saída + tokens especulativos), e a cada passo o scheduler dá a cada um `num_new_tokens` para
"alcançar" o total. Prefill em pedaços (chunked prefill), prefix cache e spec decode são o mesmo mecanismo visto
de ângulos diferentes. Isso é **continuous batching**: requests entram e saem do batch a cada passo
[fonte conceitual: blog de lançamento do V1; código em `[DS: recipe/overlay/vllm/v1/core/sched/scheduler.py]`].

Algoritmo (linhas 331-600 do scheduler do fork DS):

1. `token_budget = max_num_scheduled_tokens` (= `max_num_batched_tokens` se não definido) [U7].
2. **Primeiro os RUNNING**: para cada request, `num_new_tokens` = o que falta; se
   `0 < long_prefill_token_threshold < num_new_tokens`, corta para o limiar; depois `min(num_new_tokens, token_budget)`
   (linhas 392-394). Tokens especulativos pendentes (`spec_token_ids`) entram no mesmo orçamento (linhas 504-515).
3. `allocate_slots(...)` pede blocos (incluindo `num_lookahead_tokens` = nº de tokens especulativos). Se faltar
   bloco, **preempta** o request de menor prioridade/último da fila (modo RECOMPUTE: ele volta a WAITING e refaz o
   prefill depois) (linhas 444-488) [U21].
4. **Depois os WAITING** (se nada foi preemptado): admite enquanto `len(running) < max_num_seqs` e sobra
   orçamento, consultando o prefix cache (`get_computed_blocks`) para pular tokens já em cache.
5. Budget de encoder multimodal (`encoder_compute_budget`) limita quantos tokens de visão entram no passo.

A política prioriza decode: todos os decodes pendentes entram antes de qualquer prefill [U21].

### 6.2 Parâmetros e trade-offs

- `--max-num-batched-tokens` (MNBT): teto de tokens por passo. Menor = melhor latência entre tokens (prefill
  bloqueia menos o decode); maior = TTFT melhor em prompts longos e mais throughput. A doc recomenda >8192 para
  throughput [U21]. Também dimensiona o profiling de ativações e o cache de encoder multimodal (por isso o GLM exige
  `MM_IMAGE_TOKENS ≤ MNBT`) [GLM: start.sh].
- `--max-num-seqs`: teto de requests ativos. Com spec decode cada seq consome `k+1` tokens de decode por passo.
- `--long-prefill-token-threshold`: fatia máxima de um prefill longo por passo; é a principal defesa contra
  "decode starvation" quando vários prefills longos concorrem [DS: .env.dspark.example; issue #27].
- `--max-num-partial-prefills` / `max_long_partial_prefills`: no fork 0.25.2.dev0 da DS o campo existe no
  `SchedulerConfig` mas **o loop de admissão do V1 nunca o lê**, e a flag é rejeitada pela imagem; a DS
  implementa o limite com o hotfix `hotfix-dsv4-issue27-partial-prefill-concurrency.py` e a env
  `DSPARK_MAX_INFLIGHT_PREFILLS` (1-3, padrão 1) [DS: patches/…issue27…; CHANGELOG]. Na v0.30.0 o campo não
  aparece no `SchedulerConfig` extraído **[incerto]** [U7].
- `--async-scheduling`: `None` = automático. Desde v0.28 é ligado automaticamente com draft models (#48341) [U3];
  a auditoria QW registra que liga sozinho para `mtp` + executor `mp` [QW: docs/CLAUDE/fable5-1-report.md §1].
  `--no-async-scheduling` desliga. DS passa explicitamente (`DSPARK_ASYNC_SCHEDULING=1`), e a flag precisa ser
  igual em todas as ranks.
- Fila e QoS (v0.29+): `--max-num-queued-reqs`, `--max-num-queued-tokens` (HTTP 503 quando excede), `watermark`,
  `stream_interval` [U2, U7].
- `--performance-mode {balanced, interactivity, throughput}` (padrão `balanced`): pré-ajustes de CUDA graphs,
  batching e kernels orientados a latência ou a throughput [fonte secundária U44; o `EngineArgs` da v0.30 tem o
  campo `performance_mode` mas o enum não foi extraído; **[incerto]** quanto aos efeitos exatos]. A auditoria QW
  confirma que a flag existe na imagem day-0. Nenhuma receita usa.

Defaults (atenção): o dataclass `SchedulerConfig` da v0.30.0 declara `max_num_batched_tokens=2048` e
`max_num_seqs=128` "para conveniência de testes" [U7, U4]; o `vllm serve` substitui por defaults dependentes de
hardware/uso, e o de MNBT subiu de 8192 para 16384 na v0.28 (#51726) [U38]. O default efetivo de
`max_num_seqs` no serve em GPU **[incerto]** (historicamente 256 ou 1024 conforme a classe de GPU). As três
receitas definem ambos explicitamente, então isso não as afeta.

### 6.3 Políticas próprias das receitas por cima do scheduler

- DS: limite de prefills em voo (#27), "decode floor"/fairness e diagnóstico por passo (#43,
  `DSPARK_ISSUE43_SCHED_DIAG`), chunk adaptativo opcional (`DSPARK_ENABLE_ADAPTIVE_CHUNK`) [DS: docker-compose.dspark.yml].
- GLM: `patch_scheduler_decode_floor.py` com política `GLM53_MIXED_PREFILL_CHUNK=fair` (v5): decodes primeiro,
  crédito de tempo para prefills que competem com decodes, maior chunk que cabe em `GLM53_FAIR_PREFILL_MAX_STEP_MS`
  (2000 ms), sonda para newcomers; `skip` causa congelamento de minutos para quem chega [GLM: README.md tabela].

---

## 7. CUDA graphs e torch.compile

### 7.1 torch.compile (`CompilationConfig.mode`)

| `mode` | Nome | Significado [U8] |
|---|---|---|
| 0 | `NONE` | sem torch.compile, PyTorch eager |
| 1 | `STOCK_TORCH_COMPILE` | pipeline padrão do torch.compile |
| 2 | `DYNAMO_TRACE_ONCE` | um único trace Dynamo, sem recompilação |
| 3 | `VLLM_COMPILE` | backend Inductor do vLLM: cache, compilação piecewise, especialização de shapes, passes de fusão (padrão no V1) |

No modo 3 o grafo FX é **cortado nas operações de atenção** (`splitting_ops`, ex.
`torch.ops.vllm.unified_attention_with_output`), gerando pedaços pré-atenção, entre-atenções e pós-atenção [U18].
O cache fica em `VLLM_CACHE_ROOT/torch_compile_cache/{hash}/{rank}` e pode ser copiado entre deploys [U18]. Todas
as compilações terminam antes de servir requests [U18].

### 7.2 Modos de CUDA graph (`cudagraph_mode`)

| Modo | Semântica [U17] |
|---|---|
| `NONE` | desligado (debug) |
| `PIECEWISE` | atenção e ops incompatíveis ficam eager; o resto vai em graphs (requer compilação piecewise) |
| `FULL` | graph completo inclusive para batches não uniformes |
| `FULL_DECODE_ONLY` | graph completo só para decode uniforme; prefill/misto roda sem graph |
| `FULL_AND_PIECEWISE` | graph completo para decode uniforme + piecewise para o resto (padrão quando há compilação piecewise) |

- Decode "uniforme" = todos os requests com o mesmo `query_len`: 1 sem spec, ou `1 + num_speculative_tokens` com
  spec [U17]. O dispatcher usa um `BatchDescriptor` (`num_tokens`, `num_reqs`, `uniform`, `has_lora`) como chave [U17].
- Cada backend de atenção declara `AttentionCGSupport` (`ALWAYS`, `UNIFORM_BATCH`, `UNIFORM_SINGLE_TOKEN_DECODE`,
  `NEVER`); em modelos com vários backends vale o **mínimo** e o modo é rebaixado [U17].
- v0.30: modelos não compilados caem para `FULL_DECODE_ONLY`; piecewise indisponível agora dá erro em vez de
  saída corrompida [U1].
- `--enforce-eager`: desliga CUDA graphs (e compilação) [U4].

### 7.3 Tamanhos de captura

- `--cudagraph-capture-sizes`: lista explícita; `None` = inferida do config [U8].
- `--max-cudagraph-capture-size`: teto; padrão limitado a 512 (1024 em Blackwell de datacenter) [U8, U4].
- Batches maiores que o maior tamanho capturado rodam eager; menores são "padded" para o próximo tamanho capturado.
- **Com spec decode**, `adjust_cudagraph_sizes_for_spec_decode` arredonda cada tamanho **para cima** até múltiplo
  de `k+1` e descarta o que passar do teto. A DS descobriu que `--max-cudagraph-capture-size 42` (6 seqs × 7) era
  truncado para 40 pela grade padrão `[1,2,4,8,16,24,32,40]`, sobrando graphs `[7,14,21,28,35]`: o passo de 42 tokens
  com 6 seqs rodava **eager**. Correção: arredondar para múltiplo de 8 → 48, e aí `[7,…,42]` é capturado
  ("FULL: 6/6") [DS: docs/CLAUDE/fable5-1-report.md §3.2]. Por isso o compose calcula
  `(( (MAX_NUM_SEQS*(MTP_NUM_TOKENS+1) + 7) / 8 * 8 ))`.
- GLM monta a lista à mão: MTP k=2 → `1 2 3 4 6 8 12`; DFlash2 k=7 → `1 2 4 8 16 24 32` (múltiplos de 8 até
  `MAX_NUM_SEQS×8`); com adaptive-k (`GLM53_ADAPTIVE_K=ema`) adiciona `n×(k'+1)` para cada k' do conjunto
  [GLM: start.sh `configure_capture_sizes`]. O perfil TP3-throughput usa 25 tamanhos até 64
  [GLM: examples/tp3-throughput.env].
- QW: capturas de 1 a 64 tokens (= `max_num_seqs 8 × (1+3) × 2`) [QW: docs/CLAUDE/fable5-1-report.md §1].
- Captura mais larga custa memória (~0,1 GiB por graph na DS) e, no GB10, a QW cita risco de hard-reset com captura
  mais larga [QW: CHANGELOG].

### 7.4 CUDA graphs "breakable" (`VLLM_USE_BREAKABLE_CUDAGRAPH`)

- Alternativa ao `CUDAGraphWrapper` que troca o corte do grafo FX do torch.compile por **quebras em tempo de
  captura**: a função decorada, dentro de uma captura, encerra o segmento, roda eager no stream de captura, registra
  o callable para replay e abre novo segmento [U39].
- Default da env: `0` no fork DS e na v0.30.0 [DS: recipe/overlay/vllm/envs.py; U13]. A v0.30.0 diz que agora
  exige opt-in explícito, "antes padrão no MRV1" [U1].
- **[incerto]** A DS afirma que omitir a env "auto-habilita o caminho breakable (mais lento) no DS4" na imagem
  Anemll, e a QW afirma que "a v0.30 liga `VLLM_USE_BREAKABLE_CUDAGRAPH` para este modelo" (variação de 10-18% entre
  execuções idênticas e perda de 13-19% com S≥2). Isso conflita com a nota de release; pode haver auto-habilitação
  por modelo/condição (existe ao menos a regra de auto-ligar sob `VLLM_BATCH_INVARIANT`, PR #57586 [U39]). Na
  prática as três receitas fixam **`VLLM_USE_BREAKABLE_CUDAGRAPH=0`** (DS no compose; QW v030 via
  `V030_BREAKABLE_CUDAGRAPH`), o que as torna imunes à mudança de default [DS: .env.dspark.example; QW: start.sh].

### 7.5 Memória de graphs e caches de JIT

- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` (padrão 1): inclui graphs no profiling para dimensionar o KV. DS usa
  `0`; GLM expõe `CG_ESTIMATE` (padrão 1) e tem o overlay `patch_skip_cudagraph_profile.py`.
- Caches persistidos em volume para não recompilar a cada recreate (um rank compilando pode estourar o watchdog NCCL
  do par): `TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `B12X_CUTE_COMPILE_CACHE_DIR`,
  `VLLM_CACHE_ROOT` [DS: docker-compose.dspark.yml; GLM: start.sh]. QW v030 põe o autotune do FlashInfer em `/tmp`
  do container (`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi_autotune`) porque caches diferentes entre nós travaram
  o TP2 no coletivo de autotune [QW: CHANGELOG 2026-09-25].
- `--enable-flashinfer-autotune` / `--no-enable-flashinfer-autotune`: `KernelConfig.enable_flashinfer_autotune`,
  default `None` (automático) [U10]. DS liga; GLM desliga porque o autotune de sparse-MLA e de fused-MoE do FlashInfer
  "mata o rank 0 no SM121" [GLM: Dockerfile].

---

## 8. Decodificação especulativa

### 8.1 Framework

1. **Proposer (drafter)** gera `k = num_speculative_tokens` tokens candidatos por request.
2. O **target** processa os `k+1` tokens (último aceito + k rascunhos) num único forward, que é um decode
   "uniforme" com `query_len = k+1` (daí a importância dos tamanhos de CUDA graph múltiplos de `k+1`).
3. **Rejection sampler** decide o prefixo aceito:
   - `rejection_sample_method: "standard"`: aceita o rascunho x com probabilidade `min(1, p_target(x)/p_draft(x))`;
     na primeira rejeição amostra do resíduo normalizado `max(0, p_target − p_draft)`; se todos passam, amostra um
     token bônus do target. Isso preserva a distribuição do target ("sem perda" até a precisão numérica) [U25].
   - `draft_sample_method`: `greedy` (padrão; o rascunho é argmax e é tratado como one-hot) ou `probabilistic`
     (amostra do draft e usa os logits completos do draft na razão; custa memória) [U12].
   - `synthetic` (aceitação sintética calibrada, para benchmark) e `block` (verificação em bloco, v0.30) [U12].
4. KV de tokens rejeitados é descartado; o scheduler reservou `num_lookahead_tokens` slots para isso.

Garantias e medições: greedy com spec decode deve bater com greedy sem spec [U25]; métricas de aceitação por
posição via `/metrics` e, na v0.29+, por request (`--per-request-spec-decode-metrics`) [U2]. As receitas têm
scripts próprios: `[DS: scripts/spec-acceptance.py]`, `[QW: bench/mtp_accept.py]`, `[GLM: scripts/spec-accept-gate.sh]`.

### 8.2 Métodos (`method`)

Lista da v0.30.0: `ngram`, `ngram_gpu`, `suffix`, `draft_model`, `medusa`, `mlp_speculator`, `custom_class`, `eagle`,
`eagle3`, `extract_hidden_states`, `dflash`, `dspark` e as variantes MTP (`mtp` + tipos específicos como
`deepseek_mtp`, `qwen3_5_mtp`, `glm4_moe_mtp`…) [U12; lista do fork DS em `recipe/overlay/vllm/config/speculative.py`].

| Método | Como propõe | Notas |
|---|---|---|
| `ngram` / `ngram_gpu` | casa n-gramas do próprio prompt (`prompt_lookup_min/max`) | sem modelo; bom para código/edição |
| `suffix` | árvore de sufixos global + do prompt | parâmetros `suffix_decoding_*` [U12] |
| `draft_model` | modelo pequeno autônomo; `draft_tensor_parallel_size` 1 ou igual ao TP | `use_heterogeneous_vocab` permite tokenizers diferentes (TLI) [U12] |
| `eagle` / `eagle3` | cabeça leve condicionada em hidden states do target (EAGLE-3 usa estados auxiliares de várias camadas) + KV próprio | o GLM precisou de hooks para capturar os estados auxiliares no mHC (`taps (6,15,25,34,43)`) [GLM: README.md] |
| `mtp` | camadas MTP nativas do checkpoint (`num_nextn_predict_layers` = `n_predict`), tratadas como proposer estilo EAGLE | Se `k > n_predict`, `k` tem de ser **múltiplo** de `n_predict` (reuso do módulo MTP), e reusar a mesma camada pode baixar a aceitação [U12]. Por isso a DS usa k=6 com `n_predict=3` do Vision-Exp. `"mtp"` é mapeado para o tipo específico do modelo |
| `dflash` | drafter de "block diffusion": gera o bloco inteiro de k tokens num forward, com máscara **não causal** dentro do bloco, condicionado em hidden states do target | precisa de backend que suporte não causal (FLASH_ATTN); speculators usam SWA em todas as camadas do draft [U26]. GLM: "não fixe `TRITON_ATTN`", que nesta imagem é causal dentro do bloco e derruba a aceitação das posições finais [GLM: README.md] |
| `dspark` | DFlash + cabeça de Markov (condiciona cada posição no token anterior do bloco) + cabeça de confiança (prevê aceitação por posição); `dspark_block_size` do checkpoint | `enable_adaptive_verification` (só DSpark) dimensiona o orçamento de verificação pela confiança e exige full CUDA graphs [U25 adaptive_verification, U26]; `dspark_draft_topk` (v0.30) |
| `medusa`, `mlp_speculator`, `custom_class`, `extract_hidden_states` | legados / extensibilidade / coleta de estados | não usados |

### 8.3 Campos do `--speculative-config` usados ou relevantes

| Campo | Default (v0.30.0) [U12] | Significado | Uso |
|---|---|---|---|
| `method` | `None` (auto se `model` dado) | ver 8.2 | DS `dspark`; GLM `dflash` (rollback `mtp`); QW `mtp` |
| `num_speculative_tokens` | obrigatório (ou do config do draft) | k | DS 6 (`MTP_NUM_TOKENS`, ≥ `dspark_block_size`=5 e divisível por `n_predict`=3; k<5 trunca blocos em silêncio na 0.25.2 e é rejeitado em ≥0.26); GLM 7 (TP4: 3; MTP rollback: 2); QW 3 |
| `model` | `None` | repo/caminho do draft | GLM: snapshot local do `incoai/GLM-5.3-Flash-DFlash2` (revisão `dc77ff1c…` fixada) |
| `draft_tensor_parallel_size` | `None` (=TP do target) | só 1 ou igual ao TP do target | GLM `DFLASH_DRAFT_TP`: 2 no TP2 (C4 keep), 1 no TP3, 4 no TP4 |
| `kv_cache_dtype` | `None` (herda) | dtype do KV do draft | GLM `auto` (o `fp8_ds_mla` do target não serve para o draft denso, e SM121 não tem FA3/FA4 para FP8 puro) |
| `attention_backend` | `None` | backend do draft | não fixado (SM121 escolhe FLASH_ATTN para SWA não causal) |
| `moe_backend` | `None` (herda) | MoE do draft | não usado |
| `draft_sample_method` | `greedy` | ver 8.1 | DS `probabilistic` (default do compose; os cards oficiais usam `greedy` com k=7); GLM `probabilistic` |
| `rejection_sample_method` | `standard` | `standard` / `synthetic` / `block` | GLM `standard` explícito |
| `use_local_argmax_reduction` | `False` | argmax local por fatia do vocabulário em vez de all-gather dos logits (O(vocab) → O(2·tp) por token; só greedy, sem árvore) | QW liga quando há vocabulário de draft reduzido (`MTP_DRAFT_VOCAB`) |
| `disable_eagle_block_drop` | `False` | mantém o último bloco do prefix cache (5.2) | QW `true` |
| `index_share_for_mtp_iteration` | `None` (usa o config HF) | iterações MTP reutilizam os índices esparsos do primeiro passo | QW `true` (+6 a +27% em código) [QW: CHANGELOG] |
| `disable_padded_drafter_batch` | `False` | permite batches de draft com comprimentos diferentes | não usado |
| `parallel_drafting` | `False` | todos os rascunhos em paralelo (EAGLE/draft model treinados para isso) | não usado |
| `enable_adaptive_verification` | `False` | ver DSpark | não usado (receita DS de upstream para V4.1 usa `false` com `block` [U42]) |

### 8.4 Interações

- **CUDA graphs**: tamanhos múltiplos de `k+1` (7.3). O drafter também despacha pelo dispatcher de graphs; no MRV2
  os passos de draft pós-prefill são capturados num graph só [U20].
- **KV do drafter**: é um grupo próprio no hybrid manager (GLM g6) ou reusa camadas; DFlash/DSpark usam SWA [U26].
- **Prefix cache**: block drop e retenção (5.2).
- **Structured outputs**: o matcher de gramática precisa aceitar e reverter tokens especulativos. A imagem GLM
  (`487ecf187`) e a DS (`752a3a504`) antecedem duas correções do XGrammar (PR #52805: parar `accept_tokens()`/
  `validate_tokens()` no primeiro token terminal; PR #53046: validar rascunhos gerados antes de um fim de reasoning
  no meio da janela). GLM aplica sempre (`patch_xgrammar_termination.py`); DS é opt-in
  (`DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX`) [GLM: README.md; DS: CHANGELOG].
- **Custo por passo em TP**: o laço sequencial do drafter pode gerar coletivos seriados (12 por passo na cabeça de
  Markov fatiada da DS); replicar as matrizes pequenas (66 MB) remove isso (`DSPARK_ENABLE_REPLICATE_MARKOV`)
  [DS: docs/CLAUDE/fable5-1-report.md item 4].
- **Custo do lm_head do draft**: cada passo MTP termina num GEMM de vocabulário completo (248.320 no Qwen); o
  vocabulário de draft reduzido (47k tokens) mais `use_local_argmax_reduction` devolve a paridade de decode na v0.30
  [QW: CHANGELOG 2026-09-25].

---

## 9. Structured outputs, tool calling e reasoning

### 9.1 Structured outputs

- Backends: `xgrammar` (padrão), `guidance`, `outlines`, `lm-format-enforcer`; `auto` escolhe por request [U27].
- A cada passo o `StructuredOutputManager` gera uma **máscara de bits** de tokens permitidos; os tokens amostrados
  avançam a FSM; com spec decode os rascunhos são validados e revertidos [U27].
- Request: `response_format` (json_schema), `structured_outputs` (json, regex, choice, grammar, structural tag) [U27].
- `--structured-outputs-config` (`backend`, `disable_fallback`, `disable_any_whitespace`, `reasoning_parser`,
  `enable_in_reasoning`) [U27]. Nenhuma receita passa esta flag.
- Com reasoning ativo, a gramática só começa depois do fim do bloco de raciocínio (a menos de `enable_in_reasoning`)
  [U27, U29].

### 9.2 Tool calling

- `--enable-auto-tool-choice` + `--tool-call-parser <nome>` (+ opcional `--tool-parser-plugin`, `--chat-template`) [U28].
- `tool_choice`: `auto` (modelo decide; tool calls extraídas do texto, argumentos podem sair malformados a menos
  que alguma tool tenha `strict: true`), `required`, nomeada, `none`. `required`, nomeada e `none` usam structured
  outputs [U28].
- Parsers das receitas: DS `deepseek_v4` (formato DSML), GLM `glm47`, QW `qwen3_coder` (XML).
- Correções locais: DS #191 (contrato fail-closed para `tool_choice` nomeado/required no Chat não streaming, com
  regeneração limitada), #55 (truncamento), recuperação de DSML corrompido (backport vllm#52645); GLM
  `patch_tool_choice_none.py` [DS: docker-compose.dspark.yml; GLM: docs/tool-choice-none.md].

### 9.3 Reasoning

- `--reasoning-parser <nome>` separa `reasoning` (campo padrão; `reasoning_content` deprecado) de `content`; tool
  calls só são extraídas do `content` [U29].
- Parsers: DS `deepseek_v4` (+ `--reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'`);
  GLM `glm45`; QW `qwen3`.
- `--reasoning-config` também habilita orçamento de pensamento (`thinking_token_budget`): conta a partir de
  `reasoning_start_str` e força `reasoning_end_str` ao atingir o limite [U29]. Na DS, enviar `thinking_token_budget`
  sem o hotfix #31 dá HTTP 400, e o hotfix pode derrubar o decode de quem não usa o campo
  (`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`, padrão 0) [DS: docker-compose.dspark.yml].
- `--default-chat-template-kwargs`: defaults de servidor para o template (o request sobrescreve) [U29]. DS:
  `DEFAULT_THINKING=low` → `{"thinking":true,"reasoning_effort":"low"}`; GLM: `GLM53_DEFAULT_REASONING_EFFORT`
  vazio = não passa a flag (omitido renderiza "Max"; `medium` é rejeitado porque o template mapeia para Max).
- Stops dentro do raciocínio: DS e GLM aplicam patch para que strings `stop` do cliente só valham depois de
  `</think>` (`DSPARK_SUPPRESS_STOPS_IN_REASONING=1`, `GLM53_SUPPRESS_STOPS_IN_REASONING=1`).

### 9.4 Geração, tokenização e template

- `--generation-config`: `auto` (lê `generation_config.json` do modelo) ou `vllm` (ignora e usa defaults do vLLM)
  [U5]. DS usa `vllm`; nenhuma usa `--override-generation-config` (a auditoria QW sugere `{"temperature":0.7}` como
  opção de produto).
- `--tokenizer-mode`: `auto, cohere, deepseek_v32, deepseek_v4, hf, inkling, kimi_k3, mistral, slow` [U4]. DS usa
  `deepseek_v4`, que renderiza o chat com o encoder Python do checkpoint (`encoding/encoding_dsv4.py`, obrigatório
  nos dois nós) em vez de Jinja [DS: README.md].
- `--chat-template`: GLM força `files/chat_template.jinja` (montado em `/opt/glm53/chat_template.jinja`).
- `--trust-remote-code`: só DS.
- `DEFAULT_MAX_NEW_TOKENS=65536` (GLM, overlay): default de `max_tokens` só quando o cliente omite, para que
  requests sem limite não consumam todo o contexto restante e preemptem outras sessões [GLM: start.sh].

---

## 10. Distribuído: multi-nó, executores, NCCL/GLOO

### 10.1 Executores (`--distributed-executor-backend`)

Escolhas: `external_launcher`, `mp`, `ray`, `uni` [U4, U11]. Seleção automática (`ParallelConfig.__post_init__`):
multi-nó CUDA → `mp`; `data_parallel_backend="ray"` ou placement group Ray ativo → `ray`; world size 1 → `uni`;
"usamos multiprocessing por padrão se o world size cabe no nó atual e não estamos num placement group Ray" [U11].
As três receitas fixam `mp` (a QW TP1 também: "só o multiproc_executor dispara o worker de offload do PLE").

### 10.2 Multi-nó com `mp` (o padrão das receitas)

```bash
# head (rank 0): API server + EngineCore + worker local
vllm serve <modelo> --tensor-parallel-size 2 --distributed-executor-backend mp \
  --nnodes 2 --node-rank 0 --master-addr <IP_HEAD> --master-port <PORTA> --host 0.0.0.0 --port 8888 ...
# worker (rank 1): só workers, sem API
vllm serve <modelo> --tensor-parallel-size 2 --distributed-executor-backend mp \
  --nnodes 2 --node-rank 1 --master-addr <IP_HEAD> --master-port <PORTA> --headless ...
```

- `--nnodes` (1), `--node-rank` (0), `--master-addr` (`127.0.0.1`), `--master-port` (29501): só valem com executor
  `mp` [U11, U4]. `--headless` roda sem API server (doc aponta para multi-node DP) [U5]. Portas: DS 25000, GLM
  29521, QW 50000.
- **Todas as ranks precisam dos mesmos engine args e dos mesmos bytes de patch** (DS: "engine args must match
  across the TP pair"; todas as receitas aplicam os overlays nas duas ranks).
- Ordem: worker primeiro, depois head (QW espera 15 s; DS e GLM checam contêineres existentes e saem com códigos
  próprios).
- Alternativa Ray: montar cluster Ray (containers), `--distributed-executor-backend ray`; no V1 há
  `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` (padrão 1 no fork DS) [U22; DS: recipe/overlay/vllm/envs.py]. Nenhuma receita usa.
- Verificar o transporte: `NCCL_DEBUG=TRACE` e procurar `[send] via NET/IB/GDRDMA` (bom) vs `NET/Socket` (ruim) [U22].

### 10.3 Grupos de comunicação

- `torch.distributed` sobe um grupo **Gloo** (CPU, metadados e barreiras) e um **NCCL** (tensores); o vLLM usa
  PyNccl para os coletivos do forward. Cross-node não há custom all-reduce (precisa de P2P no mesmo nó) nem symm-mem
  do torch; a auditoria QW registra "PyNCCL only over RoCE v2" [QW: docs/CLAUDE/fable5-1-report.md §1].
- v0.29 liga o all-reduce do FlashInfer por padrão em grupos TP (`VLLM_ALLREDUCE_USE_FLASHINFER=0` desliga) [U2].
  **[incerto]** se isso se aplica a TP entre nós sem NVLink; a v0.30 cita "FlashInfer PCIe IPC all-reduce for
  NVLink-less boxes" [U1]. As receitas fixadas são anteriores à v0.29 (exceto a lane v0.30 da QW).
- Flight recorder do NCCL (DS): `TORCH_FR_BUFFER_SIZE=2000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`,
  `TORCH_NCCL_ENABLE_MONITORING=1`, `TORCH_FR_DUMP_TEMP_FILE`, `TORCH_NCCL_DEBUG_INFO_PIPE_FILE` (named pipe para
  pedir dump de uma rank travada) [DS: docker-compose.dspark.yml].

### 10.4 Variáveis de rede (valores das receitas)

Defaults de NCCL conforme a doc do NCCL 2.32.3 [U35]:

| Variável | Default NCCL | Significado | DS | GLM | QW |
|---|---|---|---|---|---|
| `VLLM_HOST_IP` | `""` (vLLM) | IP que o vLLM anuncia neste nó; defina diferente em cada nó [U13] | IP RoCE do head / `WORKER_VLLM_HOST_IP` | `HEAD_IP` / `WORKER_IP` | `HEAD_IP` / `WORKER_IP` |
| `NCCL_SOCKET_IFNAME` | nenhum | filtro de interfaces IP para bootstrap/socket; prefixo, `=` exato, `^` exclui | `enp1s0f1np1` (exemplo) | por rank: `enp1s0f1np1` (head) / `enp1s0f0np0` (worker) | `IFACE`=`enp1s0f0np0` (`WORKER_IFACE` pode diferir) |
| `GLOO_SOCKET_IFNAME` | (PyTorch) | interface do backend Gloo | = NCCL_SOCKET_IFNAME | por rank | = IFACE |
| `TP_SOCKET_IFNAME` | (PyTorch TensorPipe) | interface do TensorPipe/RPC do torch **[inferência: vLLM não depende dele no caminho quente]** | = NCCL_SOCKET_IFNAME | não define | = IFACE |
| `NCCL_IB_HCA` | todas | dispositivos RDMA `<hca>[:porta]`; `=` exato, `^` exclui | `rocep1s0f1` | `rocep1s0f1` / `rocep1s0f0` | `=rocep1s0f0` (exato) |
| `NCCL_IB_GID_INDEX` | -1 (dinâmico) | índice GID RoCE | vazio → resolvido por nó a partir do sysfs (`NCCL_IB_GID_AUTO=1`); vazio é *removido* porque NCCL leria vazio como GID 0 (link-local fe80) | 3 (`HEAD_GID`/`WORKER_GID` por rank) | 3 ("se 3 não subir, tente 5") |
| `NCCL_IB_ROCE_VERSION_NUM` | 2 | versão RoCE para GID dinâmico | 2 | 2 | – |
| `NCCL_IB_ADDR_FAMILY` | AF_INET | família de endereço do GID | AF_INET | – | – |
| `NCCL_IB_DISABLE` | 0 | 1 desliga IB/RoCE | 0 | 0 | 0 |
| `NCCL_NET` | (auto) | força `IB` ou `Socket` | IB | IB | – |
| `NCCL_NET_PLUGIN` | (default) | plugin de rede externo | – | `none` | – |
| `NCCL_CROSS_NIC` | 2 (auto) | anéis/árvores entre NICs diferentes | 1 | 0 (TP3/TP4: 1) | – |
| `NCCL_CUMEM_ENABLE` | 1* | alocação via cuMem | 0 | 0 | – |
| `NCCL_NVLS_ENABLE` | 2 | NVLink SHARP (inexistente no Spark) | 0 | 0 | – |
| `NCCL_IGNORE_CPU_AFFINITY` | 0 | ignora afinidade herdada | 1 | 1 | – |
| `NCCL_IB_MERGE_NICS` | (varia) | funde NICs em um dispositivo virtual | opcional (vazio = ausente) | 0 | – |
| `NCCL_NET_GDR_LEVEL`, `NCCL_NET_GDR_READ`, `NCCL_DMABUF_ENABLE` | auto | GPUDirect RDMA / dmabuf | passthrough opcional | – | – |
| `NCCL_IB_SUBNET_AWARE_ROUTING`, `NCCL_IB_SUBNET_PREFIX_LEN` | – | **não constam da página de env do NCCL 2.32.3** **[incerto]** | passthrough opcional | – | – |
| `NCCL_GIN_ENABLE` | – | GPU-initiated networking; a DS mede bootstrap ~10x mais rápido com `0`. **Não consta da página de env do NCCL 2.32.3 [incerto]** | opcional | – | – |
| `NCCL_IB_AUTO_DETECT` | – | **não é variável documentada do NCCL** (provável no-op) **[incerto]** | – | – | 0 |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | por plataforma | canais | – | `NCCL_NCHANNELS` opcional fixa ambos | – |
| `NCCL_DEBUG` | VERSION | nível de log | WARN | WARN | WARN |
| `PYTORCH_CUDA_ALLOC_CONF` | – | alocador do PyTorch | `expandable_segments:True` | `expandable_segments:True` (vazio desliga) | – |

Notas de fabric do GB10 (resumo das receitas):

- Os dois Sparks podem estar "cruzados" (porta f1 no head ↔ f0 no worker), então interface e HCA mudam por nó
  (GLM e QW têm variáveis por rank) [QW: start.sh; GLM: .env.example].
- Cada PF do ConnectX-7 é PCIe Gen5 x4 (~126 Gb/s); o link de 200 Gb/s só atinge taxa cheia usando os dois PFs
  (socket-direct), e a DS encontrou MTU RoCE 1024 [DS: docs/CLAUDE/fable5-1-report.md item 3].
- `USE_HOST_NCCL=1` (GLM) faz `LD_PRELOAD` de um `libnccl.so.2.30.7` do host; padrão é o NCCL da imagem, porque o
  preload duplica o do DeepEP [GLM: README.md tabela].
- Contêineres com `--network host --ipc host`, `--device /dev/infiniband`, `--ulimit memlock=-1`,
  `--cap-add IPC_LOCK` (GLM) ou `SYS_NICE` (QW), `--shm-size 32g` (GLM). A doc do vLLM pede IPC_LOCK e `/dev/shm`
  para GPUDirect RDMA [U22].

---

## 11. Paralelismo: TP, PP, DP, EP, DCP/PCP, DBO

| Dimensão | Flag | Default | Semântica no vLLM | Nas receitas |
|---|---|---|---|---|
| Tensor (TP) | `--tensor-parallel-size` | 1 | fatia pesos por coluna/linha; all-reduce por camada; world size = TP×PP×PCP | 2 (DS/GLM/QW); 3 e 4 opcionais (DS `start-tp3.sh`, GLM `start-tp3.sh`/`start-tp4.sh`); 1 na QW TP1 |
| Pipeline (PP) | `--pipeline-parallel-size` | 1 | divide camadas em estágios (`VLLM_PP_LAYER_PARTITION`) | DS passa `1` explicitamente |
| Dados (DP) | `--data-parallel-size`, `-local`, `--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-backend {mp,ray}` | 1 / mp | réplicas do engine (um EngineCore por réplica, coordenador), úteis sobretudo com MoE+EP [U11] | não usado |
| Expert (EP) | `--enable-expert-parallel` | False | MoE usa EP em vez de TP; `EP_SIZE = TP × DP` (× PCP); atenção continua TP/DP [U23, U11] | QW liga (padrão `ENABLE_EXPERT_PARALLEL=true`); GLM TP3 liga (288 experts / 3 = 96 por rank, porque o trellis EXL3 é empacotado em 2048 e não divide por 3) |
| All2all | `--all2all-backend` | `allgather_reducescatter` | backend de dispatch/combine do MoE com EP: `naive, pplx, deepep_high_throughput, deepep_low_latency, deepep_v2, mori_*, nixl_ep, allgather_reducescatter, flashinfer_all2allv, flashinfer_nvlink_one_sided, flashinfer_nvlink_two_sided` [U11] | QW passa `allgather_reducescatter`, mas a auditoria mostra que **nunca é invocado** com DP=1 e PCP=1: tokens são replicados por TP, cada rank calcula seus 256 experts locais e o resultado passa pelo all-reduce normal; EP só muda quais pesos cada GPU guarda [QW: docs/CLAUDE/fable5-1-report.md §1] |
| EPLB | `--enable-eplb` + JSON (`window_size`, `step_interval`, `num_redundant_experts`, …) | False | rebalanceamento de experts [U23] | não usado |
| Decode CP | `--decode-context-parallel-size` (`-dcp`) | 1 | fatia o KV ao longo de T entre ranks do TP (intercalado, `cp_kv_cache_interleave_size`), **sem aumentar o world size**; limite `tp/H_kv`; remove duplicação de KV em MLA com TP [U24, U11] | não usado (o card do checkpoint GLM original cita TP2/EP2/DCP2 numa imagem SM120 B12X diferente [GLM: README.md]) |
| Prefill CP | `--prefill-context-parallel-size` | 1 | divide o prefill (query parcial / ring attention); aumenta o world size [U24, U11] | não usado |
| DBO | `--enable-dbo` | False | dual batch overlap (sobrepõe comunicação e computação em micro-batches) [U4] | não usado; a extensão MoE cooperativa do GLM recusa `--enable-dbo`/ubatching |
| Encoder MM | `--mm-encoder-tp-mode {weights,data}` | weights | `data` replica o encoder de visão e divide o batch entre ranks | QW `data` (MLP de visão 4304/2 = 2152 não é múltiplo de 16, exigido pelos kernels NVFP4); GLM TP3 `data` |

TP não divisível (TP=3): exige padding de cabeças via `--hf-overrides`/patches. DS: cabeças 64→72
(`DSv4 TP pad: heads 64 -> 72 for TP=3` no boot) [DS: README.md]. GLM TP3:
`--hf-overrides '{"num_attention_heads":66,"num_key_value_heads":66,"linear_num_heads":66,"text_config":{…}}'`
(`TP3_HEAD_OVERRIDE=66`; não use 96, que deixa um rank só com cabeças KDA fictícias) + EP para os experts
[GLM: start-tp3.sh; README.md].

---

## 12. Quantização e backends de kernel

### 12.1 A API de quantização do vLLM (como um método novo entra)

Componentes (`vllm/model_executor/layers/quantization/`) [U31]:

1. **`QuantizationConfig`** (em `base_config.py`): lido do `quantization_config` do `config.json` (ou de arquivos
   extras). Métodos: `get_name()`, `get_supported_act_dtypes()`, `get_min_capability()`,
   `get_config_filenames()`, `from_config(dict)`, `override_quantization_method(hf_quant_cfg, user_quant)` (permite
   reivindicar um checkpoint automaticamente) e o central **`get_quant_method(layer, prefix)`**, que devolve, por
   tipo de camada, o método que vai criar e usar os pesos.
2. **`LinearMethodBase`** (camadas `LinearBase`: QKV, gate/up, down, o_proj, lm_head): `create_weights()` (registra
   `Parameter`s com os shapes do checkpoint e `weight_loader`s que sabem fatiar por TP),
   `process_weights_after_loading()` (repack, conversões, pré-cálculos) e `apply(layer, x, bias)` (o GEMM).
   `UnquantizedLinearMethod` é o ponto de partida.
3. **`FusedMoEMethodBase`** (experts): mesmos três ganchos, mais a config de MoE; no fork GLM o alvo é a classe
   `RoutedExperts` (`vllm.model_executor.layers.fused_moe.routed_experts`), não mais o `FusedMoE` antigo.
4. **Registro**: `@register_quantization_config("nome")` (suporte out-of-tree via PR #11969) torna `--quantization nome`
   válido; métodos embutidos ficam no `Literal` `QuantizationMethods` e em `get_quantization_config()` [U31].

### 12.2 Estudo de caso: como o EXL3 foi plugado (GLM)

Arquivo: `[GLM: overlay/exl3.py]` (3.248 linhas), instalado em
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py` pelo Dockerfile.

- `@register_quantization_config("exl3") class Exl3Config(QuantizationConfig)`:
  - `get_name()` → `"exl3"`; `get_supported_act_dtypes()` → BF16/FP16/FP32; `get_min_capability()` → 80 (Ampere+;
    GB10 é SM121); `get_config_filenames()` → `quantization_config.json`.
  - `from_config` lê `bits` (3-6), `codebook` (só `mcg`), `scope` (`glm53_routed_experts_only`) e o bloco opcional
    `non_routed_exl3.layers` (EXL3 também em linears densos, "dense-exl3"). Recusa `self_attn.kv_b_proj` porque a
    absorção de pesos do MLA lê esse peso diretamente.
  - `override_quantization_method`: se `quant_method == "exl3"` no config HF, reivindica o checkpoint.
  - `get_quant_method`:
    - `RoutedExperts` → `Exl3MoEMethod(FusedMoEMethodBase)`: pesos ficam como trellis int16 + `suh`/`svh` fp16 +
      marcador `mcg`; nunca vira BF16 persistente; um único `exllamav3_ext.exl3_moe` fundido por camada (decode) e
      kernels E2/E3 "fat expert" para prefill; TP=2 fatia gate/up por coluna e down por linha, e o runner de MoE faz
      um all-reduce por camada.
    - `ParallelLMHead` declarado → `Exl3LinearMethod`; senão `None` (nativo).
    - `LinearBase` declarado em `non_routed_exl3` → `Exl3LinearMethod(LinearMethodBase)`; grupos
      `GLM53_DENSE_FP8` → `Glm53DenseFp8Method` (FP8 weight-only via Marlin por canal de saída); demais →
      `UnquantizedLinearMethod`. EXL3-dense e FP8-overlay no mesmo módulo levantam erro.
- Registro no pacote (Dockerfile): patch no `quantization/__init__.py` para inserir `"exl3"` no `Literal`
  `QuantizationMethods` e importar `Exl3Config` preguiçosamente em `get_quantization_config`; e
  `overlay/patch_model_overrides.py` insere `"exl3"` na lista ordenada de overrides do `ModelConfig` (sem isso o
  `override_quantization_method` nunca é consultado) [GLM: Dockerfile; overlay/patch_model_overrides.py].
- Lança com `--quantization exl3` (`QUANTIZATION=exl3`). Não passe `--moe-backend marlin`: o MoE é do método EXL3,
  não de um backend do vLLM [GLM: README.md].
- A extensão CUDA (`exllamav3_ext`, ExLlamaV3 `c5d9c657`) é compilada no Dockerfile para `sm_121a`
  (`TORCH_CUDA_ARCH_LIST=12.1a`), com stubs dos alvos AVX de CPU que não compilam em aarch64.

Por que isso importa para receitas futuras: a mesma receita (Config + LinearMethod + FusedMoEMethod + registro)
serve para qualquer formato fora da árvore; se empacotado como **plugin** (12.4), dispensa editar o `__init__.py`.

### 12.3 Métodos nativos usados pelas receitas

| Método | Checkpoint | Detalhes |
|---|---|---|
| ModelOpt (NVIDIA Model Optimizer) | QW `nvidia/Qwen3.8-Flash-Next-NVFP4`: BF16 denso + experts NVFP4 + PLE FP8 | detectado por `hf_quant_config.json`; `quant_algo` FP8 (per-tensor, per-channel, bloco), NVFP4 (W4A4 e W4A16), MXFP8 [U30 modelopt]. A imagem day-0 precisou de `files/patch_modelopt_mxfp8.py` (shapes MXFP8 que o `mm_mxfp8` do FlashInfer não roda vão para emulação BF16) e `patch_modelopt_fp8_block_moe.py` (ramo FP8_BLOCK_SCALES para os experts do MTP) [QW: start.sh]. MoE no GB10: FlashInfer CUTLASS SM120 (TRTLLM e CuteDSL são só SM10x; `FLASHINFER_B12X` fica fora da autoseleção e exige EP desligado) [QW: docs/CLAUDE/fable5-1-report.md §1] |
| FP8 bloco 128x128 + MXFP4 (E8M0 bloco 32) | DS DeepSeek-V4-Flash-Vision-Exp | experts `fp4_e8m0_k32`; boot confirma `Using 'B12X_MXFP4' Mxfp4 MoE backend`, que roda **W4A16** (dequant → MMA BF16) porque o caminho NVFP4 de tensor core do b12x só aceita modelopt [DS: docs/CLAUDE/fable5-1-report.md item 5] |
| `modelopt_gb10_hybrid` (plugin DS, opcional) | NVFP4 W4A4 ModelOpt | Marlin W4A16 para M < `GB10_HYBRID_NVFP4_M_THRESHOLD` (128) e CUTLASS W4A4 para M maior; mantém duas cópias dos pesos; testado só em empacotamento [DS: vllm_patch_gb10/README.md] |
| FP8 oficial (bloco 128) | QW `start-fp8.sh` (`Qwen/Qwen3.8-Flash-Next-FP8`) | ~500k tokens de KV vs 3,65M no NVFP4 com KV fp8 [QW: README.md] |
| compressed-tensors | não usado pelas receitas | formato do llm-compressor (FP8 W8A8, INT4 W4A16, INT8…) [U30] |

### 12.4 Plugin de quantização (exemplo real, DS)

`[DS: vllm_patch_gb10/pyproject.toml]` declara
`[project.entry-points."vllm.general_plugins"] gb10_hybrid_nvfp4 = "vllm_gb10_hybrid_nvfp4:register"`; o
`register()` (reentrante) importa o módulo cujo decorador registra `modelopt_gb10_hybrid`, subclasse de
`ModelOptMixedPrecisionConfig`. No boot: `pip install -e /opt/vllm-gb10-hybrid-nvfp4 --no-deps`,
`VLLM_PLUGINS=gb10_hybrid_nvfp4`, `--quantization modelopt_gb10_hybrid` [DS: docker-compose.dspark.yml].

### 12.5 Backends de kernel (`--moe-backend`, `--linear-backend`)

- `--moe-backend` (default `auto`; v0.30.0): `auto, triton, batched_triton, deep_gemm, deep_gemm_mega_moe, cutlass,
  flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl, flashinfer_b12x, b12x,
  flashinfer_moe_ep_mega_deep_gemm, flashinfer_moe_ep_mega_cutedsl, marlin, humming, triton_unfused, aiter,
  aiter_triton_mxfp4_bf16, flydsl, hpc, emulation, rdna3` [U10].
- `--linear-backend` (default `auto`): `auto, cutlass, flashinfer_cutlass, flashinfer_cutedsl, flashinfer_trtllm,
  flashinfer_cudnn, flashinfer_b12x, b12x, marlin, humming, triton, deep_gemm, torch, aiter, machete, fbgemm, conch,
  exllama, emulation, xpu, xpu_woq`; `linear_backend_per_quant` (v0.30) sobrescreve por esquema [U10].
- Essas flags substituem envs antigas (`VLLM_TEST_FORCE_FP8_MARLIN`, `VLLM_USE_FLASHINFER_MOE_*`…) [U2; DS:
  recipe/overlay/vllm/envs.py marca várias como "Deprecated: use --moe-backend/--linear-backend"].
- Receitas: DS `--moe-backend flashinfer_b12x` + `VLLM_USE_B12X_MOE=1` (env do fork) + seletores experimentais
  `VLLM_B12X_W4A16_FORCE_*`; GLM não usa (MoE vem do método EXL3); QW `auto`. Upstream, no SM121, o MoE B12X só é
  alcançado explicitamente (issue #54666 pede autoseleção) e houve correção de checkpoints W4A16 rodando como W4A4
  no `flashinfer_b12x` (PR #56535) [busca U44b].
- `--speculative-config` aceita `moe_backend` e `quantization` próprios do draft [U12].

---

## 13. Attention backends

### 13.1 Registro e seleção

O enum `AttentionBackendEnum` (arquivo `vllm/v1/attention/backends/registry.py`, cópia em
`[DS: recipe/overlay/vllm/v1/attention/backends/registry.py]`) lista: `FLASH_ATTN`, `FLASH_ATTN_DIFFKV`, `TRITON_ATTN`,
`FLASHINFER`, `FLASHINFER_MLA`, `FLASHINFER_MLA_SPARSE`, `TOKENSPEED_MLA`, `TRITON_MLA`, `CUTLASS_MLA`, `FLASHMLA`,
`FLASHMLA_SPARSE`, `SPARSE_MLA_SM120`, `B12X_MLA_SPARSE` (fork DS, opt-in), `FLASH_ATTN_MLA`, `FLEX_ATTENTION`,
`TURBOQUANT`, `CPU_ATTN`, variantes ROCm/XPU, `NO_ATTENTION`, `TORCH_SDPA` (ViT) e `CUSTOM` (precisa
`register_backend(...)`). O valor do enum é o caminho da classe e pode ser sobrescrito em runtime.

Seleção: a plataforma percorre uma lista de prioridade e escolhe o primeiro backend compatível com capability,
head size, dtype do KV, MLA/esparso, sliding window, não causalidade e nível de suporte a CUDA graph.

Override:

- `--attention-backend <NOME>` ou `--attention-config '{"backend": ...}'` (`AttentionConfig.backend`; `auto`/`None` =
  automático) [U9].
- v0.30: `backend_per_kind` (backend por grupo de KV), `mla_prefill_backend`, `flash_attn_version` (2/3/4),
  `indexer_kv_dtype` (dtype do K do indexer esparso; substitui o alias `use_fp4_indexer_cache`), `use_non_causal`,
  `sparse_mla_force_mqa`, `hisparse_config` [U9, U1].
- `VLLM_ATTENTION_BACKEND` **não existe mais** no `envs.py` da v0.30.0 [U13]; o comentário do fork DS ainda o cita
  para `B12X_MLA_SPARSE` **[incerto se ainda é lido lá]**.
- Draft: `attention_backend` no `--speculative-config` [U12].

### 13.2 Backends que importam no GB10 (SM121)

| Backend | Onde aparece | Observações |
|---|---|---|
| `FLASH_ATTN` | draft DFlash2 do GLM (SWA não causal) | no SM12x é FA2; FA3/FA4 não existem no SM121 [GLM: README.md] |
| `TRITON_ATTN` | evitar no draft GLM | "máscara causal dentro do bloco" nesta imagem; derrubou a aceitação (~29 tok/s, 0,31) [GLM: README.md] |
| `FLASHINFER_MLA_SPARSE_SM120` | target GLM | único MLA esparso no SM12x para o layout GLM_NSA; só aceita `fp8_ds_mla` empacotado; BF16 não tem kernel esparso [GLM: README.md] |
| `DSV4_SPARSE_MLA_SM120` (backend do próprio modelo) / `B12X_MLA_SPARSE` (opt-in) | DS | o DeepSeek-V4 traz backends próprios: em SM120/121 `_select_v4_sparse_impl()` escolhe `DeepseekV4SM120SparseBackend` (nome `DSV4_SPARSE_MLA_SM120`), que usa o `BatchSparseMLAPagedAttentionWrapper` do FlashInfer (o mesmo do `SPARSE_MLA_SM120` da família V3.2) com cache duplo SWA + indexer comprimido; em Hopper/SM10x é o `DeepseekV4FlashMLASparseBackend` [DS: recipe/overlay/vllm/models/deepseek_v4/nvidia/sm120.py]. `B12X_MLA_SPARSE` fica fora da autoseleção [DS: registry.py]. Indexer "Lightning" com top-k 512; `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` limita o tensor de logits do indexer (DS usa 256; padrão 512) [DS: docker-compose.dspark.yml; U13] |
| Kernels QSA próprios (Triton) | QW | a atenção completa do Qwen3.8 não usa FlashAttention: o owner QSA roda `_qsa_mqa_paged_kernel`, `persistent_topk`, `_qsa_sparse_paged_gqa_splitk_kernel`; `FlashAttentionImpl` é só a classe base; "FA2 no README é nominal" [QW: docs/CLAUDE/fable5-1-report.md §1] |
| Atenção linear (GDN, KDA) | QW (GDN), GLM (KDA) | implementadas como camadas tipo Mamba (estado recorrente, grupo `MambaSpec`); prefill GDN no GB10 cai para Triton FLA `chunk_gated_delta_rule` porque o FlashInfer/CuteDSL está limitado a SM90/SM10x [QW: fable5-1-report §1]; KDA no GLM tem caminho BF16 para M grande (`GLM53_KDA_BF16_LARGE_M`) e FlashKDA no TP3 |

MLA esparso em uma frase: um **indexer** leve pontua o contexto (comprimido) e escolhe top-k posições; a atenção MLA
só lê esses tokens. Isso exige caches auxiliares (K do indexer, "kpool" no GLM) que viram grupos de KV próprios
(seção 4), workspaces grandes (`GLM53_INDEXER_WORKSPACE=rightsize` economiza ~5 GB a 1M) e cuidados de smem (o GLM
evita MNBT 8192 porque estoura a smem do `persistent_topk` no GB10) [GLM: start.sh; README.md].

---

## 14. Registro de modelos, plugins out-of-tree e overlays

### 14.1 Registro de modelos

- `config.json` → `architectures: ["XxxForCausalLM"]` → `ModelRegistry` resolve para `(módulo, classe)` via
  `_VLLM_MODELS`, que junta `_TEXT_GENERATION_MODELS`, `_MULTIMODAL_MODELS`, `_SPECULATIVE_DECODING_MODELS` etc. em
  `vllm/model_executor/models/registry.py` [U32; DS: recipe/overlay/vllm/model_executor/models/registry.py].
- O registro é **preguiçoso** (`_LazyRegisteredModel(module, class)`), para não inicializar CUDA no import
  ("Cannot re-initialize CUDA in forked subprocess") [U32].
- Layout novo "isolado por hardware": entradas podem apontar para módulos fora de `model_executor/models`, ex.
  `"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", …)` com implementação em `vllm/models/deepseek_v4/nvidia/`;
  a QW mexe em `vllm/models/qwen3_8_flash_next/nvidia/` (day-0) e `vllm/models/qwen4_exp/nvidia/` (v0.30)
  [DS: registry.py `_resolve_module_name`; QW: start.sh].
- Drafts também são arquiteturas registradas: `DFlashDraftModel` → `qwen3_dflash`, `DeepSeekV4DSparkModel`,
  `DeepSeekV4MTPModel`, `Glm4MoeMTPModel`… [DS: registry.py].
- Out-of-tree: `ModelRegistry.register_model("Arch", "pacote.modulo:Classe")` dentro de um plugin [U32].
- `--hf-overrides`: dicionário mesclado no config HF. A QW descobriu que `_apply_dict_overrides` só recursa em chaves
  que são configs aninhados; para `qwen4_exp`, `{"rope_parameters":…}` no topo era aplicado no pai e **nunca chegava
  ao `text_config`** (YaRN virava no-op), então tudo vai sob `"text_config"` [QW: start.sh comentário].
  `--rope-scaling` (flag antiga) não é usada por nenhuma receita; YaRN é feito via `--hf-overrides` (QW:
  `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}`
  quando `YARN_ENABLE=true`) + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Na v0.30 aliases YaRN de fornecedores não
  reescalam mais `max_position_embeddings` e `attn_factor`/`extrapolation_factor` são ignorados em favor de
  `mscale`/`attention_factor` [U1].

### 14.2 Sistema de plugins

Grupos de entry point: `vllm.general_plugins` (modelos, quantização, ops), `vllm.platform_plugins` (novas
plataformas: é assim que o vllm-metal entra), `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`,
`vllm.endpoint_plugins` [U33]. Todo processo (API server, EngineCore, workers) carrega os plugins; `VLLM_PLUGINS`
filtra (lista separada por vírgula; vazia = nenhum; ausente = todos) [U33, U13]. A função registrada precisa ser
reentrante [U33]. Parsers de tool/reasoning também podem vir por plugin (`--tool-parser-plugin`) [U28].

### 14.3 As três estratégias de overlay usadas

| Receita | Técnica | Detalhes | Salvaguardas |
|---|---|---|---|
| DS | (a) hotfixes aplicados no boot; (b) arquivos inteiros de overlay (lane Stage-C histórica); (c) plugin | (a) `patches/hotfix-*.py` e `patches/hotfix-*.sh` montados no container e executados no `command:` do compose antes do `exec vllm serve`, cada um seguido de `exit 1` em caso de falha; opt-ins por `DSPARK_ENABLE_*`/`DSPARK_SKIP_*`. (b) `recipe/overlay/vllm/**` cobre `envs.py`, `config/speculative.py`, `scheduler.py`, `gpu_model_runner.py`, backends; (c) `vllm_patch_gb10` | fonte exata (âncoras/hashes das fontes fixadas), idempotente, fail-closed, `--check/--status/--before/--after`; fixtures das fontes originais em `scripts/fixtures/**-752a3a504-*.py`; `scripts/verify-overlay-sources.sh` |
| GLM | (a) Dockerfile `FROM` a imagem upstream com patchers Python em `RUN`; (b) `COPY overlay/exl3.py` para dentro do pacote; (c) cadeia `GLM53_OVERLAY_ORDER` aplicada nas duas ranks no start | ordem importa (ex.: replay híbrido antes da retenção porque compartilham ponto de inserção); patches cobrem scheduler, prefix cache, mamba align, xgrammar, kpool, spinwait, adaptive-k, dense FP8, DFlash2 EXL3, loadclone, cache reset, ablit | fixtures `tests/fixtures/*-487ecf187.py.txt`; testes por patch; stamp de receita força rebuild da imagem quando o overlay muda |
| QW | extração + patch no host + bind-mount | `extract_from_image` copia o arquivo da imagem, o patch roda no host, e `-v host:container:ro` sobrepõe o arquivo em `site-packages` (no worker, cópia em `/tmp/vllm-overlay`); `.diff` em `files/overlay/`; sem rebuild de imagem | patches recusam aplicar quando a imagem já tem a funcionalidade (ex. `patch_block_drop.py`), lane v0.30 pula todos os overlays day-0 exceto o vocabulário de draft |

Regras práticas **[inferência a partir das três]**: aplicar o mesmo byte em todas as ranks; verificar a fonte
original antes de escrever (âncora única ou hash); falhar o boot em vez de servir com patch parcial; registrar no log
que o patch foi aplicado; remover o overlay assim que o upstream incorporar (a QW anota "apague quando a imagem for
vLLM 0.31+" para o backport de KV fp8, vllm#55557).

---

## 15. Referência de argumentos por receita

Formato de cada entrada: **o que faz** · **default** (v0.30.0 salvo nota) · **interações** · **valores nas receitas**.
Variáveis entre parênteses são os knobs do launcher que produzem o valor.

### 15.1 Modelo, carga e identidade

**`<modelo>` posicional / `--revision`**
- Repo HF ou caminho local; `--revision` fixa commit.
- DS: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` + `--revision 86f746b3…` (`DSPARK_REVISION`). GLM: caminho local do
  snapshot (`MODEL_DIR`, revisão `MODEL_REVISION`). QW: `nvidia/Qwen3.8-Flash-Next-NVFP4` (`MODEL_ID`) com
  `HF_HUB_OFFLINE=1`.

**`--served-model-name`**
- Nome(s) expostos na API; vários nomes = aliases [U5].
- DS `deepseek-v4-flash-vision-exp` (aceita lista separada por espaço; o primeiro é usado por probes). GLM
  `GLM-5.3-Flash-EXL3`. QW `qwen3.8-flash-next` (`qwen3.8-flash-next-fp8` no lane FP8).

**`--trust-remote-code`** · False · executa código do repo HF · só DS.

**`--tokenizer-mode`** · `auto` · ver 9.4 · DS `deepseek_v4`.

**`--load-format`**
- `auto, pt, safetensors, instanttensor, ipc_cache, npcache, dummy, tensorizer, runai_streamer, runai_streamer_sharded,
  sharded_state, mistral, modelexpress` [U4]; default `auto`.
- QW `safetensors`; GLM `instanttensor` (direct-I/O; padrão quando a imagem contém "instanttensor"; em 850k/0,85 exige
  `--kv-cache-memory-bytes` para bootar, issue #204) [GLM: start.sh]; DS não passa.

**`--safetensors-load-strategy`** · `None` (`eager, lazy, prefetch, torchao`) [U4] · QW `lazy` (menos pico de memória
no UMA). GLM usa loaders próprios (`GLM53_LOAD_CLONE=1`, `GLM53_LOAD_PREFETCH=0`).

**`--quantization`** · `None` (auto-detecta do checkpoint) · força método (12.x) · GLM `exl3`; DS
`modelopt_gb10_hybrid` só com `ENABLE_VLLM_GB10_PATCH=1`; QW auto (ModelOpt).

**`--hf-overrides`** · `{}` · ver 14.1 · QW: `{"text_config":{"ple_embedding_dtype":…,"rope_parameters":{…}}}` quando
necessário; GLM TP3: padding de cabeças 66.

**`--generation-config` / `--override-generation-config`** · `auto` / `{}` · ver 9.4 · DS `vllm`; ninguém usa override.

**`--chat-template`** · modelo · ver 9.4 · GLM `/opt/glm53/chat_template.jinja`.

**`--language-model-only`** · False · desliga entradas multimodais (não carrega a torre de visão) [U4] · GLM quando
`LANGUAGE_MODEL_ONLY=1` (padrão 0).

### 15.2 Paralelismo e distribuído

**`--tensor-parallel-size`** · 1 · world size, fatiamento, all-reduces; interage com divisibilidade de cabeças/experts
(TP3 exige padding) e com o KV de MLA (duplicado) · DS `2` (`TP_SIZE`; 3 via `start-tp3.sh`); GLM `2` (`TP`; 3/4); QW
`2` (`TENSOR_PARALLEL_SIZE`; 1 no TP1).

**`--pipeline-parallel-size`** · 1 · DS passa `1`.

**`--distributed-executor-backend`** · auto · 10.1 · todas `mp`.

**`--nnodes` / `--node-rank` / `--master-addr` / `--master-port` / `--headless`** · 1 / 0 / 127.0.0.1 / 29501 / False ·
10.2 · DS `2`/`NODE_RANK`/`MASTER_ADDR`/`25000`/`HEADLESS` no worker; GLM `2`/`0|1`/`HEAD_IP`/`29521`/worker
headless; QW `2`/`0|1`/`HEAD_IP`/`50000`/worker headless.

**`--enable-expert-parallel`** · False · 11 · QW sim (padrão); GLM TP3 sim; DS não.

**`--all2all-backend`** · `allgather_reducescatter` · 11 · QW `allgather_reducescatter` (inerte com DP=1).

**`--mm-encoder-tp-mode`** · `weights` · 11 · QW `data`; GLM TP3 `data`.

**`--decode-context-parallel-size`, `--data-parallel-*`, `--enable-dbo`** · 1 / 1 / False · 11 · não usados.

### 15.3 Memória e KV

**`--gpu-memory-utilization`** · 0,92 · fração da memória total usada pelo executor; no GB10 disputa com o host; é
ignorada para KV quando `--kv-cache-memory-bytes` está presente · DS `0,835` (derivado de
`GPU_MEMORY_UTILIZATION_TEXT`; não defina `GPU_MEMORY_UTILIZATION` à mão; 0,87 no perfil de código longo); GLM `0,85`
(0,865 long-coding; 0,80 TP3; 0,75 TP4; 0,87 derrubou um head em prefill de 256k); QW `0,80` (0,835 deixava <1 GiB
livre por nó); QW TP1 derivado de `KV_TARGET_GIB=20`.

**`--kv-cache-memory-bytes`** (QW TP1 escreve `--kv-cache-memory`) · `None` · fixa o KV por GPU e pula a derivação por
utilização · GLM `11811160064` (11 GiB) no `.env.example` via `EXTRA_ARGS` (14 GiB em docs antigas; 34359738368 no
TP3-throughput; os launchers TP3/TP4 removem a reserva herdada do `.env` de TP2). **[incerto]** `--kv-cache-memory`
funciona na imagem QW (a auditoria lista a flag), possivelmente por abreviação de prefixo do argparse ou alias.

**`--kv-cache-dtype`** · `auto` · 3.3 · DS `nvfp4_ds_mla`; GLM `fp8` (→ `fp8_ds_mla`); QW `fp8` (day-0 com patch) /
`auto` (v0.30; `fp8` via backport vllm#55557) / `auto` (TP1).

**`--block-size`** · `None` (16) · 3.1/4.1; em híbridos o vLLM aumenta sozinho · DS `256`; demais automático.

**`--max-model-len`** · derivado do config · teto por request; acima do `max_position_embeddings` exige
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` · DS `1048576` (YaRN x16 de 64K no checkpoint); GLM `850000` (262144 long-coding;
1000000 TP3/TP4); QW `262144` (≈1M com YaRN fator 4); QW FP8 `262144`.

**`--mamba-ssm-cache-dtype` / `--mamba-cache-dtype`** · `auto` · 4.2 · QW `bfloat16` (ssm; `MAMBA_SSM_CACHE_DTYPE`).

**`--mamba-cache-mode` / `--mamba-block-size`** · `none` / `None` · 4.2 · não passados (o modelo força `align`).

**`--enable-prefix-caching`** · True no V1 (Mamba desde v0.28) · 5 · DS e GLM explícitos; QW implícito.

**`--prefix-caching-hash-algo`** · `sha256` · 5.1 · não usado.

**`--prefix-match-unit`** · não definido · 5.1 · GLM `PREFIX_MATCH_UNIT` (vazio por padrão; `64` suportado na
geometria testada).

**`--prefix-cache-retention-interval`** (v0.29+) · 0 · 5.2 · as receitas fixadas usam a env equivalente (seção 16).

**`--enable-prompt-tokens-details`** · False · 5.1 · DS e QW; GLM no perfil long-coding.

**`--kv-transfer-config`** · `None` · conector de KV (LMCache, NIXL, offload) · DS opcional (`lmcache/`,
`DSPARK_ENABLE_LMCACHE=1`): `{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.server_urls":"tcp://…"}}`,
junto com `PYTHONHASHSEED=0` e `unset PYTORCH_CUDA_ALLOC_CONF` (o vLLM rejeita conector de KV com
`expandable_segments:True`) [DS: lmcache/README.md]. GLM confirma que não usa.

**`--swap-space`, `--cpu-offload-gb`** · não presentes no `CacheConfig` v0.30.0 extraído (preempção é RECOMPUTE) [U6] ·
não usados. Offload de embeddings grandes: QW usa um offload próprio de PLE (`PLE_OFFLOAD`, `VLLM_PLE_CPU_OFFLOAD`);
upstream v0.30 tem `--engram-config '{"cpu_offload":true}'` [U1, U42].

### 15.4 Scheduler

**`--max-num-seqs`** · ver 6.2 · DS `6` (4 no perfil de código longo; TP3 `16`); GLM `4` (2 long-coding; 8 TP4); QW
`8`; QW TP1 `4`.

**`--max-num-batched-tokens`** · 16384 no serve em GPU desde v0.28 **[incerto no detalhe por hardware]** · 6.2 · DS
`8192` (16384 no perfil de código longo; aumentar para calar o aviso 8150/8168 foi testado sem ganho); GLM `7168`
(8192 estoura smem do top-k; 1024 long-coding; 2048 TP4; 4096 TP3-throughput); QW `8192`; QW TP1 `2048`.

**`--long-prefill-token-threshold`** · 0 (desligado) · 6.2 · DS `1024` (junto com o hotfix #27 derrubou a dispersão
de fairness de ~46x para ~1,05x); GLM vazio por padrão (2048 no TP3-throughput); QW não usa (auditoria sugere 2048).

**`--max-num-partial-prefills`** · ver 6.2 · rejeitado pela imagem DS; substituído por `DSPARK_MAX_INFLIGHT_PREFILLS`.

**`--enable-chunked-prefill`** · True · 6.1 · DS e QW explícitos; GLM implícito.

**`--async-scheduling` / `--no-async-scheduling`** · auto · 2.2/6.2 · DS explícito (`DSPARK_ASYNC_SCHEDULING=1`; 0
remove nas duas ranks para A/B do issue #191).

### 15.5 Compilação e CUDA graphs

**`--compilation-config`** · `mode` 3 (V1), `cudagraph_mode` resolvido pelo backend · 7 · QW
`'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` (sem Inductor; graphs só no decode uniforme; TP1:
`CUDAGRAPH_MODE`, `NONE` para debug eager). DS e GLM usam o default (a DS mostra graphs FULL e PIECEWISE no boot, ou
seja `FULL_AND_PIECEWISE`).

**`--max-cudagraph-capture-size`** · 512 (1024 Blackwell DC) · 7.3 · DS `48` = `ceil(6×7/8)×8`.

**`--cudagraph-capture-sizes`** · inferido · 7.3 · GLM: `1 2 4 8 16 24 32` (DFlash2 k=7), `1 2 3 4 6 8 12` (MTP k=2);
não é adicionado se o operador já passou em `EXTRA_ARGS`.

**`--enforce-eager`** · False · 7.2 · GLM quando `ENFORCE_EAGER=1` (padrão do launcher 0).

**`--enable-flashinfer-autotune` / `--no-enable-flashinfer-autotune`** · `None` (auto) · 7.5 · DS liga; GLM desliga.

### 15.6 Kernels e atenção

**`--moe-backend`** · `auto` · 12.5 · DS `flashinfer_b12x`; GLM não passar (especialmente `marlin`); QW auto.

**`--linear-backend`** · `auto` · 12.5 · não usado.

**`--attention-backend` / `--attention-config`** · auto · 13.1 · DS opcional `--attention-config
{"use_fp4_indexer_cache":true}` (`DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1`, exige `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`;
na v0.30 o alias foi removido e o equivalente é `indexer_kv_dtype`); GLM: não fixar backend nem para o draft.

### 15.7 Decodificação especulativa

**`--speculative-config`** (JSON) · desligado · 8 · valores exatos:

```jsonc
// DS (MTP_NUM_TOKENS=6, DRAFT_SAMPLE_METHOD=probabilistic)
{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}

// GLM TP2 (SPEC_METHOD=dflash, DFLASH_TOKENS=7, DFLASH_DRAFT_TP=2)
{"method":"dflash","model":"<DFLASH_MODEL_DIR>","num_speculative_tokens":7,"kv_cache_dtype":"auto",
 "draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":2}
// GLM rollback (SPEC_METHOD=mtp, MTP_TOKENS=2)
{"method":"mtp","num_speculative_tokens":2}

// QW (MTP_NUM_SPECULATIVE_TOKENS=3, MTP_DRAFT_VOCAB definido, MTP_DISABLE_BLOCK_DROP=1, MTP_INDEX_SHARE=true)
{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,
 "disable_eagle_block_drop":true,"index_share_for_mtp_iteration":true}
// QW TP1
{"method":"mtp","num_speculative_tokens":3}
```

Interações: tamanhos de CUDA graph (7.3), `max_num_seqs×(k+1)` tokens de decode por passo, KV do draft (4.3),
block drop (5.2), XGrammar (8.4). Envs do fork que mexem no draft: QW `VLLM_MTP_DRAFT_VOCAB`; GLM `GLM53_ADAPTIVE_K`,
`GLM53_DRAFT_KV_COMPACT`; DS `DSPARK_ENABLE_DSPARK_BLOCK_K` (permite k=`dspark_block_size`=5),
`DSPARK_ENABLE_REPLICATE_MARKOV`, envs Stage-C `VLLM_DSPARK_*`.

### 15.8 Frontend, parsers e API

**`--host` / `--port`** · host padrão não verificado nesta pesquisa **[incerto]** / porta 8000 · todas `8888` no
head; GLM e QW usam `--host 0.0.0.0`; DS usa `VLLM_HOST` (127.0.0.1 no compose, 0.0.0.0 no `.env.dspark.example`).

**`--api-key`** / `VLLM_API_KEY` · nenhum · exige Bearer nos prefixos guardados (`/v1`, `/v2`, `/inference`); rotas
fora deles (`/health`, `/metrics`, `/version`, `/ping`, `/tokenize`, `/detokenize`, `/invocations`) ficam sem auth
na imagem DS [DS: docs/ENVS.md] · DS: `VLLM_API_KEY` (uma chave) ou `DSPARK_API_KEYS` (várias chaves num único
`--api-key` nargs; mutuamente exclusivos); GLM: só `VLLM_API_KEY` por env, para a chave não aparecer no argv/log.

**`--enable-auto-tool-choice` / `--tool-call-parser`** · False / nenhum · 9.2 · DS `deepseek_v4`; GLM `glm47`; QW `qwen3_coder`.

**`--reasoning-parser` / `--reasoning-config`** · nenhum · 9.3 · DS `deepseek_v4` + config `<think>`/`</think>`; GLM
`glm45`; QW `qwen3`.

**`--default-chat-template-kwargs`** · nenhum · 9.3 · DS `{"thinking":true,"reasoning_effort":"low"}` (padrão
`DEFAULT_THINKING=low`; `off`/`high`/`max`); GLM `{"reasoning_effort":…}` só se configurado.

### 15.9 Multimodal

**`--limit-mm-per-prompt`** · `{}` · teto de itens por prompt (validação; nada é reservado) · DS `{"image":8}` (a
imagem Anemll exige JSON, por isso `image=8` é convertido); GLM `{"image":48,"video":1}` (clientes sem estado
reenviam todas as imagens a cada turno e passam do teto) [GLM: docs/images-per-request.md].

**`--mm-processor-kwargs`** · — · GLM `{"max_image_tokens":2048}` (checkpoint permite 8000; um cliente que picou um
vídeo em 33 frames como imagens gerou 236k tokens de encode e o OOM matou o worker) [GLM: start.sh].

**`--media-io-kwargs`** · `{}` · GLM `{"video":{"num_frames":N}}` só se `VIDEO_NUM_FRAMES` (vazio = 32 do vLLM).

**`--mm-processor-cache-gb`** · 4 · GLM `1` (no UMA são 4 GiB que o modelo perde).

**`--skip-mm-profiling`** · False · não perfila o dummy multimodal máximo no init (evita OOM, mas nada fica reservado
para a torre) · GLM `1`.

### 15.10 Flags citadas nos repos mas fora do caminho padrão

`--enable-dbo` (recusado pela extensão cooperativa GLM), `--kv-sharing-fast-prefill` (mensagem de erro no model
runner DS), `--enable-layerwise-nvtx-tracing`, `--enable-mfu-metrics` (código do fork DS), `--profiler-config`
(scripts de profiling GLM), `--max-log-len` (testes GLM), `--linear-backend` (envs deprecadas no fork DS),
`--thinking-token-budget` (argumento de benchmark da DS, não do vLLM). Não usados: `--data-parallel-*`,
`--decode-context-parallel-size`, `--rope-scaling`, `--cuda-graph-sizes` (nome antigo de
`--cudagraph-capture-sizes`), `--swap-space`.

---

## 16. Referência de variáveis de ambiente

### 16.1 Variáveis `VLLM_*` do vLLM

Defaults da v0.30.0 (`vllm/envs.py`) salvo nota [U13]; "fork" = existe só nas imagens das receitas.

| Variável | Default | Significado | DS | GLM | QW |
|---|---|---|---|---|---|
| `VLLM_HOST_IP` | `""` | IP deste nó para o distribuído | por nó | por rank | por nó |
| `VLLM_PORT` | None | porta base de comunicação | – | – | – |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | caches (torch.compile, etc.) | `/cache/huggingface/vllm-cache` | `/root/.cache/vllm` | montado em `/root/.cache/vllm` |
| `VLLM_API_KEY` | None | chave da API | opcional | opcional | – |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | 0 | permite `max_model_len` acima do config | 1 | – | 1 (necessário com YaRN) |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 300 | 2.2 | 1800 | 1800 | – |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 600 | 2.2 | – | 3600 | – |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | 0 | 7.4 | 0 (manter; omitir fica mais lento no DS4) | – | 0 na lane v0.30 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 1 | 7.5 | 0 | `CG_ESTIMATE` (1) | – |
| `VLLM_USE_FLASHINFER_SAMPLER` | 1 (se suportado) | sampler top-k/top-p do FlashInfer | 1 | – | – |
| `VLLM_PLUGINS` | None (todos) | filtro de plugins | vazio ou `gb10_hybrid_nvfp4` | – | – |
| `VLLM_NO_USAGE_STATS` | 0 | desliga telemetria | – | 1 (+ `DO_NOT_TRACK=1`; py-cpuinfo quebra no aarch64) | – |
| `VLLM_SERVER_DEV_MODE` | 0 | rotas de dev (`/reset_prefix_cache`…) | – | independente; o GLM expõe só rotas de reset com `GLM53_EXPOSE_CACHE_RESET=1` | – |
| `VLLM_ENABLE_RESPONSES_API_STORE` | 0 | `store` da Responses API (memória nunca liberada no stock) | 0 (1 aplica backport com store limitado) | – | – |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | None | diretório do cache de autotune | – | – | `/tmp/fi_autotune` (v030); outro dir com `VLLM_MOE_DET_FINALIZE` |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 512 | teto do tensor de logits do indexer esparso | 256 | – | – |
| `VLLM_DEBUG_WORKSPACE` | 0 | loga redimensionamentos de workspace | – | diagnóstico via `GLM53_EXTRA_ENV` | – |
| `VLLM_LOGGING_LEVEL` | INFO | nível de log | – | diagnóstico | – |
| `VLLM_WORKER_MULTIPROC_METHOD` | fork (força spawn se CUDA já foi iniciado) | start method dos workers [U19] | – | – | – |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | 1 | EngineCore em processo próprio | – | – | – |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | 1 (v0.29+) | all-reduce FlashInfer em TP | – | – | – |
| `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` | 8192 | contexto sintético no profiling da verificação adaptativa | – | – | – |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | fork (removida na v0.30) | 5.2 | 4096 | `GLM53_APC_RETENTION_INTERVAL` opcional | testado 1664, não adotado |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA` | overlay GLM | retenção do grupo SWA do drafter | – | `GLM53_APC_RETENTION_INTERVAL_SWA` (TP3/TP4: 0) | – |
| `VLLM_ATTENTION_BACKEND` | removida (v0.30) | ver 13.1 | – | – | – |
| `VLLM_USE_B12X_MOE`, `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` (0), `..._MAX_M` (16), `..._TILE_CONFIG` | fork DS | MoE b12x MXFP4 e seletor W4A16 | 1 / 0 / 16 / vazio | – | – |
| `VLLM_DSPARK_*`, `VLLM_DSV4_*`, `VLLM_USE_B12X_WO_PROJECTION`, `VLLM_TRITON_MLA_SPARSE`, `VLLM_SKIP_INIT_MEMORY_CHECK` | só overlay Stage-C | no Anemll 0.1.1 geram "Unknown vLLM environment variable" | só com `docker-compose.stage-c.override.yml` | – | – |
| `VLLM_SUPPRESS_STOPS_IN_REASONING` | alias aceito pela DS | ver 9.3 | mapeado para `DSPARK_SUPPRESS_STOPS_IN_REASONING` | – | – |
| `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS` | overlay GLM TP4 | fatia a chamada final de MLA esparso em ≤64 linhas (mitigação de stall) | – | 0 (TP4) | – |
| `VLLM_PLE_CPU_OFFLOAD`, `VLLM_PLE_PACKED_TABLE_DIR`, `VLLM_PLE_OFFLOAD_STEP_TIMEOUT` | overlay QW | offload da tabela PLE (51 GB) | – | – | `PLE_OFFLOAD`; TP1 sempre |
| `VLLM_QSA_PROFILE` / `VLLM_QSA_PROFILE_JSON` | overlay QW | perfil de lançamento dos kernels QSA | – | – | `QSA_PROFILE` |
| `VLLM_MTP_DRAFT_VOCAB` | overlay QW | vocabulário reduzido do draft MTP | – | – | `/etc/vllm-draft-vocab.txt` |
| `VLLM_QSA_DET_TOPK`, `VLLM_MOE_DET_FINALIZE` (+ `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0`) | overlay QW | determinismo (≈−4% decode) | – | – | opt-in |

### 16.2 Outras variáveis (não `VLLM_*`) que o vLLM ou suas libs leem

| Variável | Onde | Valor |
|---|---|---|
| `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_XET`, `HF_TOKEN` | todas | offline depois do download; DS desliga XET |
| `TORCH_CUDA_ARCH_LIST`, `FLASHINFER_CUDA_ARCH_LIST` | DS, GLM | `12.1a` |
| `CUTE_DSL_ARCH` | DS | `sm_121a` (sem isso o b12x/CuTeDSL pode JITar um MoE W4A16 mais lento no meio do serviço) |
| `FLASHINFER_DISABLE_VERSION_CHECK` | DS, GLM | 1 |
| `TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, `TILELANG_CLEANUP_TEMP_FILES`, `FLASHINFER_WORKSPACE_BASE`, `B12X_CUTE_COMPILE_CACHE_DIR`, `DG_JIT_USE_NVRTC`, `DG_JIT_NVCC_COMPILER` | DS (e GLM para Triton/TileLang) | caches persistentes, JIT do DeepGEMM via nvcc |
| `PYTORCH_CUDA_ALLOC_CONF` | DS, GLM | `expandable_segments:True` |
| `TORCH_NCCL_*`, `TORCH_FR_*` | DS | flight recorder (10.3) |
| `NCCL_*`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` | todas | tabela 10.4 |
| `GB10_HYBRID_NVFP4_M_THRESHOLD`, `B12X_W4A16_TC_DECODE` | DS | plugin híbrido (128); decode empacotado b12x (0) |
| `INSTANTTENSOR_*` | GLM | loader direct-I/O (buffer, io_depth…) |
| `PYTHONHASHSEED` | DS (LMCache) | 0 em imagens anteriores à v0.29 |

---

## 17. Rodar local no Mac: contrato OpenAI, vllm-metal e mlx_lm.server

Contexto: o pedido do usuário inclui dar ao modelo do "copilot" um caminho local no Mac (MLX) com contrato no estilo
OpenAI. Esta seção descreve, do lado vLLM, o que é o contrato e quais opções existem; a integração com o cliente fica
para o documento da receita.

### 17.1 O contrato que as receitas expõem

O `vllm serve` expõe [U34]:

- OpenAI: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/embeddings`, áudio,
  `/v1/chat/completions/batch`.
- Outros formatos: `/v1/messages` e `/v1/messages/count_tokens` (Anthropic), `/v2/embed`, `/v2/rerank` (Cohere).
- Utilitários: `/health`, `/load`, `/version`, `/metrics` (Prometheus), `/tokenize`, `/detokenize`,
  `/tokenizer_info`, `/ping`.
- Auth: `--api-key` ou `VLLM_API_KEY` (Bearer) [U34].
- Extensões do vLLM no corpo (via `extra_body` nos SDKs): `chat_template_kwargs` (ex. `{"thinking":false}`,
  `{"reasoning_effort":"high"}`), `structured_outputs`, `vllm_xargs` (ex. `skip_writing_prefix_cache` no overlay
  GLM), `cache_salt`, `thinking_token_budget` (DS com hotfix).
- Respostas: `reasoning` separado de `content`, `tool_calls` padrão OpenAI, `usage.prompt_tokens_details.cached_tokens`
  com `--enable-prompt-tokens-details`.

Um cliente que só dependa de `base_url`, `api_key`, `model` e dos campos OpenAI padrão (mensagens, tools,
`tool_choice`, `stream`, `max_tokens`, `temperature`) funciona igual contra o cluster Spark e contra um servidor
local no Mac. O que **não** é portátil: `chat_template_kwargs` específicos do modelo, `vllm_xargs` de overlays,
`thinking_token_budget`, nomes de modelo.

### 17.2 vllm-metal (plugin de plataforma, MLX por baixo)

- Plugin mantido pela comunidade vLLM que traz o scheduler V1, o KV paginado, chunked prefill, prefix caching,
  sampling e o frontend OpenAI (com streaming e parsers de tool call) para o Apple Silicon; a execução é MLX/Metal e
  os modelos vêm do `mlx_lm` [U40].
- Instalação: `brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal` e
  `brew install vllm-project/vllm-metal/vllm-metal` (estável), ou o instalador
  `curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash` +
  `source ~/.venv-vllm-metal/bin/activate` (desenvolvimento). Requer macOS 15+ em Apple Silicon [U40].
- Uso: o mesmo CLI, ex. `vllm serve mlx-community/Qwen3.8-27B-4bit --gpu-memory-utilization 0.7` (Mac de 64 GB),
  API em `http://localhost:8000/v1` [U40].
- Recursos declarados: V1 scheduler, KV paginado, chunked prefill, prefix caching, três métodos de spec decode (MTP do
  Gemma 4, draft separado, n-gram), structured outputs, parsers de tool call, LoRA, GGUF, modelos híbridos (famílias
  Qwen3.5/3.6/3.8), pipeline parallel entre Macs, VLMs experimentais [U40]. v0.30.0 estável em 2026-09-23 (fonte
  secundária) **[incerto]**.
- Limites: só modelos no formato MLX (`mlx-community`) ou GGUF; nada que dependa de CUDA (FlashInfer, NCCL, CUDA
  graphs, NVFP4/EXL3/MXFP4 kernels, TP/EP multi-nó) se aplica. Portanto flags como `--moe-backend`,
  `--kv-cache-dtype nvfp4_ds_mla`, `--compilation-config`, `--cudagraph-capture-sizes`, `--nnodes`, `--enable-expert-parallel`
  devem sair da linha de comando no Mac **[inferência]**.
- Variáveis de ambiente específicas do plugin: não consegui verificar a tabela oficial nesta pesquisa; consulte
  `https://docs.vllm.ai/projects/vllm-metal/en/latest/` antes de usar nomes como `VLLM_METAL_*` **[incerto]**.

### 17.3 mlx_lm.server (alternativa mínima)

- `mlx_lm.server --model <repo_mlx>` sobe um servidor "semelhante à API de chat da OpenAI" em `localhost:8080`, com
  `/v1/chat/completions` e `/v1/models`, streaming, parâmetros de amostragem, adapters e spec decode com
  `draft_model` [U41].
- A própria doc diz que não é recomendado para produção (checagens de segurança básicas) e que com KV quantizado
  (`--kv-bits`) processa um request por vez; tool calling não está documentado [U41].
- Para um "copilot" local que dependa de tool calls e reasoning separado, o vllm-metal preserva melhor o mesmo
  contrato das receitas **[inferência]**.

### 17.4 Mapeamento prático de uma receita Spark para o Mac

| Aspecto | Spark (receitas) | Mac (vllm-metal) |
|---|---|---|
| Checkpoint | NVFP4 / EXL3 / MXFP4+FP8 com kernels CUDA | versão MLX 4/8 bits do mesmo modelo (ou irmão menor) no `mlx-community`; tamanho limitado pela RAM unificada |
| Memória | `--gpu-memory-utilization` 0,75-0,865 no UMA de 128 GB | mesmo flag, deixando folga para o macOS (exemplos 0,5-0,7) |
| Contexto/concorrência | `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens` | os mesmos flags do vLLM; valores bem menores |
| Parsers | `--tool-call-parser`, `--reasoning-parser`, `--enable-auto-tool-choice` | mesmos flags (parsers são do frontend vLLM) |
| Spec decode | DSpark/DFlash/MTP com kernels CUDA | n-gram, draft separado, MTP do Gemma 4 |
| Cliente | `base_url=http://head:8888/v1` | `base_url=http://localhost:8000/v1`; mesmo SDK OpenAI |

---

## 18. Roteiro (lado vLLM) para montar uma receita de um modelo recém-lançado

Este roteiro resume o que as três receitas fizeram, na ordem em que as decisões do vLLM aparecem. Serve como checklist
técnico para o tutorial de receita pedido pelo usuário (tuning e modificação ficam nos passos 7-10).

### Passo 0: capacidade antes de tudo

Some pesos + KV mínimo + graphs + ativações contra `n_nós × ~121 GiB × gpu_memory_utilization`. Exemplo com um
lançamento recente: **DeepSeek-V4.1-Flash** (lançado em 2026-09-10; suporte em `main` via vllm#56228, nightly a partir
dessa data; 552B de backbone + 196B de Engram; checkpoint de ~510 GB; a receita upstream fala em piso de ~614 GB e
usa TP4 em MI355X, com `--engram-config '{"cpu_offload":true}'` na variante TP2 para mover ~94 GiB de tabelas para a
memória do host) [U42, busca]. Isso não cabe em 2 Sparks (≈2×121 GiB) nem em 4; exigiria um checkpoint quantizado da
comunidade (NVFP4/EXL3) ou mais nós. No UMA do Spark, "offload para o host" não libera memória física
**[inferência]**.

### Passo 1: suporte e versão

1. Confirme a arquitetura no registro (`architectures` do `config.json` vs `_VLLM_MODELS`), na doc de modelos
   suportados e nas release notes (a v0.30.0 lista GLM-5.3-Flash, DeepSeek-V4-Flash-Vision-Exp e V4.1-Flash) [U1].
2. Escolha a base: release estável (`vllm/vllm-openai:vX.Y.Z`), nightly/commit fixado, ou imagem day-0 do fornecedor.
   Fixe por digest e registre o commit (como DS `752a3a504` e GLM `487ecf187`).
3. Liste o que o GB10 (SM121, aarch64, CUDA 13) não tem: FA3/FA4, vários kernels SM90/SM100 (TRTLLM, CuteDSL, DeepGEMM
   variantes), NVLink. Espere precisar de overlays.

### Passo 2: ler o `config.json` como o vLLM vai ler

- Tipos de camada → grupos de KV (atenção completa, SWA, MLA/MLA esparso + indexer, linear/Mamba), e portanto block
  size efetivo e custo de IDs (seção 4).
- `num_nextn_predict_layers` / drafter disponível → método de spec (MTP nativo, EAGLE3, DFlash, DSpark).
- `quantization_config` / `hf_quant_config.json` → método de quantização e backends de kernel possíveis (seção 12).
- `max_position_embeddings`, `rope_parameters` (YaRN) → `--max-model-len`, `--hf-overrides`,
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.
- Divisibilidade por TP: cabeças de atenção/KV, cabeças lineares, `moe_intermediate_size`, número de experts, MLP da
  visão (NVFP4 exige múltiplo de 16) → TP escolhido, EP, `--mm-encoder-tp-mode data`, padding via `--hf-overrides`.

### Passo 3: esqueleto multi-nó

`--tensor-parallel-size N --distributed-executor-backend mp --nnodes N --node-rank r --master-addr … --master-port …`
(+ `--headless` nos workers), variáveis da tabela 10.4 por nó, contêiner com `--network host --ipc host
--device /dev/infiniband --ulimit memlock=-1`. Valide com `NCCL_DEBUG=TRACE` que o transporte é `NET/IB` [U22].

### Passo 4: memória

Comece com `--gpu-memory-utilization 0.80`, `--max-model-len` no nativo, `--max-num-seqs` 4-8. Leia no boot
`Available KV cache memory`, `GPU KV cache size` e (em híbridos) calcule IDs por segmento. Se o profiling for instável
no UMA, fixe `--kv-cache-memory-bytes`. Monitore `MemAvailable` dos dois nós sob carga (as receitas usam 0,8-0,86
porque valores maiores zeraram a memória livre e derrubaram ranks).

### Passo 5: KV dtype e block size

`fp8` quando o backend do modelo aceita (checar se o kernel da atenção do modelo suporta; QSA do Qwen e MLA esparso
têm restrições próprias), `auto` como baseline de qualidade. Deixe o block size automático, a menos que o backend exija
(DS 256).

### Passo 6: parsers e template

Descubra o parser de tool/reasoning que o upstream registrou para a família (release notes, `--tool-call-parser`
choices) e se o modelo usa `--tokenizer-mode` próprio (DeepSeek) ou template Jinja. Defina
`--default-chat-template-kwargs` para o modo de pensamento padrão. Teste `tool_choice` `auto`, `required` e nomeado,
com e sem streaming.

### Passo 7: spec decode e CUDA graphs (primeiro grande tuning)

1. Ligue o método nativo com k pequeno (MTP k=1-3; DFlash/DSpark k=5-7, que costuma ser o ponto bom [busca U26]).
2. Calcule os tamanhos de captura: múltiplos de `k+1` até `max_num_seqs×(k+1)`; arredonde o teto para cima (lição da
   DS) e confira no log que não há "Truncating" e que "Capturing CUDA graphs (FULL): n/n" cobre o seu `max_num_seqs`.
3. Meça aceitação por posição; se as últimas posições aceitam <0,3, k maior não compensa (DS parou em 6).
4. Em multi-turn com MTP/EAGLE, avalie `disable_eagle_block_drop` (TTFT do segundo turno).

### Passo 8: scheduler (segundo tuning)

`--max-num-batched-tokens` (latência entre tokens vs TTFT), `--long-prefill-token-threshold` (proteção de decode sob
prefills longos), `--max-num-seqs`. Teste com concorrência mista (um decode longo + prompts grandes chegando) e meça
dispersão de TPOT entre requests, não só médias. Registre preempções no log.

### Passo 9: kernels

Olhe no boot qual backend de MoE/linear/atenção foi escolhido (`Using '…' MoE backend`, `Using … attention backend`) e
compare alternativas com `--moe-backend`/`--linear-backend`/`--attention-config`. Persista caches de JIT em volume e
aqueça os shapes reais depois do `/health` (as três receitas têm `boot-shape-warmup`).

### Passo 10: modificação (overlays e métodos próprios)

- Patch de comportamento: script fail-closed e idempotente, com âncora/hash da fonte fixada, aplicado em todas as ranks
  antes do `exec vllm serve` (seção 14.3).
- Novo formato de peso: `QuantizationConfig` + `LinearMethodBase`/`FusedMoEMethodBase` + registro, de preferência como
  plugin `vllm.general_plugins` (seção 12).
- Nova arquitetura: `ModelRegistry.register_model("Arch", "pacote:Classe")` num plugin.
- Edição de pesos em tempo de carga (ex.: "ablit" de `o_proj` no GLM e projeção de direção no DS): feita em
  `process_weights_after_loading`/loader, idêntica nas ranks, sempre opt-in.

### Passo 11: qualificação

Boot limpo dos dois nós com mesmo digest; smoke de chat/tools/visão; NLL/logprobs contra referência (a QW compara NLL
em 15.776 posições); aceitação de spec; teste de longa duração (a QW roda 1 h com 1.349 requests e 0 erros);
registro de todas as flags e versões no README (como as três receitas fazem).

---

## 19. Fontes

Upstream (vLLM e dependências):

- [U1] vLLM v0.30.0 release notes: https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- [U2] vLLM v0.29.0 release notes: https://github.com/vllm-project/vllm/releases/tag/v0.29.0
- [U3] vLLM v0.28.0 release notes: https://github.com/vllm-project/vllm/releases/tag/v0.28.0
- [U4] Engine arguments: https://docs.vllm.ai/en/latest/configuration/engine_args.html e https://docs.vllm.ai/en/stable/configuration/engine_args/
- [U5] `vllm serve` CLI: https://docs.vllm.ai/en/stable/cli/serve/
- [U6] `CacheConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py (e no commit fixado da DS: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/cache.py)
- [U7] `SchedulerConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/scheduler.py
- [U8] `CompilationConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/compilation.py
- [U9] `AttentionConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/attention.py
- [U10] `KernelConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/kernel.py
- [U11] `ParallelConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/parallel.py
- [U12] `SpeculativeConfig` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/speculative.py
- [U13] `envs.py` v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/envs.py
- [U14] Architecture overview: https://docs.vllm.ai/en/latest/design/arch_overview.html
- [U15] Automatic prefix caching: https://docs.vllm.ai/en/latest/design/prefix_caching.html
- [U16] Hybrid KV cache manager: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html
- [U17] CUDA graphs: https://docs.vllm.ai/en/latest/design/cuda_graphs.html
- [U18] torch.compile integration: https://docs.vllm.ai/en/latest/design/torch_compile.html
- [U19] Multiprocessing: https://docs.vllm.ai/en/latest/design/multiprocessing.html
- [U20] Model Runner V2: https://docs.vllm.ai/en/latest/design/model_runner_v2/ e blog https://vllm.ai/blog/2026-03-24-mrv2
- [U21] Optimization and tuning: https://docs.vllm.ai/en/latest/configuration/optimization.html
- [U22] Parallelism and scaling: https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html
- [U23] Expert parallel deployment: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html
- [U24] Context parallel deployment: https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html
- [U25] Speculative decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding/ ; MTP: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/ ; EAGLE: https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/ ; adaptive verification: https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/
- [U26] DFlash (speculators): https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/ ; artigo DFlash: https://arxiv.org/abs/2602.06036 ; artigo DSpark: https://arxiv.org/pdf/2607.05147 ; speculators: https://github.com/vllm-project/speculators
- [U27] Structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- [U28] Tool calling: https://docs.vllm.ai/en/latest/features/tool_calling.html
- [U29] Reasoning outputs: https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
- [U30] Quantização: https://docs.vllm.ai/en/latest/features/quantization/ ; KV quantizado: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ ; ModelOpt: https://docs.vllm.ai/en/latest/features/quantization/modelopt/
- [U31] `base_config` (API de quantização): https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/base_config/ ; out-of-tree: https://github.com/vllm-project/vllm/pull/11969
- [U32] Registro de modelos: https://docs.vllm.ai/en/latest/contributing/model/registration.html
- [U33] Plugin system: https://docs.vllm.ai/en/latest/design/plugin_system.html
- [U34] Online serving (endpoints): https://docs.vllm.ai/en/latest/serving/online_serving/
- [U35] NCCL environment variables (NCCL 2.32.3): https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- [U36] PR #53388 (`disable_eagle_block_drop`): https://github.com/vllm-project/vllm/pull/53388
- [U37] Issue #53504 (retenção vs MTP em híbrido): https://github.com/vllm-project/vllm/issues/53504
- [U38] PR #51726 (default de MNBT 8192→16384): https://github.com/vllm-project/vllm/pull/51726
- [U39] Breakable CUDA graphs: https://docs.vllm.ai/en/latest/api/vllm/compilation/breakable_cudagraph/ ; PR #56312: https://github.com/vllm-project/vllm/pull/56312 ; PR #57586: https://github.com/vllm-project/vllm/pull/57586
- [U40] vllm-metal: blog https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0 ; repo https://github.com/vllm-project/vllm-metal ; docs https://docs.vllm.ai/projects/vllm-metal/en/latest/
- [U41] mlx_lm.server: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
- [U42] Receita upstream DeepSeek-V4.1-Flash: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash ; anúncio: https://www.deepseek.com/en/news/deepseek-v4-1-flash/
- [U43] PRs XGrammar citados pelas receitas: https://github.com/vllm-project/vllm/pull/52805 e https://github.com/vllm-project/vllm/pull/53046
- [U44] `--performance-mode` (fonte secundária): https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning ; [U44b] B12X no SM12x: https://github.com/vllm-project/vllm/issues/54666 e https://github.com/vllm-project/vllm/pull/56535

Fontes locais mais usadas (dentro de `scratchpad/repos/`):

- DS: `docker-compose.dspark.yml`, `.env.dspark.example`, `docker-compose.stage-c.override.yml`, `docs/ENVS.md`,
  `docs/CLAUDE/fable5-1-report.md`, `README.md`, `CHANGELOG.md`, `patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py`,
  `recipe/overlay/vllm/{envs.py,config/speculative.py,v1/core/sched/scheduler.py,v1/attention/backends/registry.py,model_executor/models/registry.py}`,
  `vllm_patch_gb10/`, `lmcache/`.
- GLM: `start.sh` (blocos `write_inner_scripts`, `configure_capture_sizes`, env das ranks), `start-tp3.sh`,
  `.env.example`, `.env.tp3.example`, `.env.tp4.example`, `examples/tp2-long-coding.env`, `examples/tp3-throughput.env`,
  `Dockerfile`, `overlay/exl3.py`, `overlay/patch_model_overrides.py`, `README.md`,
  `docs/DESIGN-apc-per-group-retention.md`.
- QW: `start.sh` (passo 7), `start-v030.sh`, `start-fp8.sh`, `.env.sample`, `tp1/start.sh`, `tp1/.env.sample`,
  `CHANGELOG.md`, `README.md`, `docs/CLAUDE/fable5-1-report.md`, `files/patch_block_drop.py`.

Pendências de verificação (itens marcados **[incerto]**): datas exibidas como 2024 no resumo das releases; efeito
exato de `--performance-mode`; default efetivo de `max_num_seqs` no serve em GPU; existência de
`max_num_partial_prefills` na v0.30.0; auto-habilitação de breakable CUDA graphs por modelo na v0.30 (conflito entre
release notes e CHANGELOG da QW); `NCCL_GIN_ENABLE`, `NCCL_IB_SUBNET_AWARE_ROUTING`, `NCCL_IB_AUTO_DETECT` fora da doc do
NCCL 2.32.3; `--kv-cache-memory` na imagem QW; variáveis específicas do vllm-metal; geometria de grupos GLM (11 vs 12
camadas MLA, 3 vs 4 grupos Mamba) entre versões do overlay.

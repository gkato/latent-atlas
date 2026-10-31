# Inventário DS41 — Lançamento / parametrização do DeepSeek-V4-Flash(-Vision-Exp) + DSpark em 2× DGX Spark

> Repositório analisado (somente leitura): `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
> Lido de um clone local (completo) do repositório acima.
> Escopo deste inventário: README.md, `.env.dspark.example` (inteiro), `docker-compose.dspark.yml` (+ overrides `stage-c` e `nfs`), `start-deepseek-v4-flash-dspark.sh` (inteiro, 2154 linhas), `start-tp3.sh`, `stop-/status-/logs-/smoke-*.sh`, `prepare-dspark-model-cache.sh`, `validate-dspark-config.sh`, `dspark-numeric-knobs.sh`, `build-dspark-vllm-runtime.sh`, Dockerfiles (`recipe/**`, `Dockerfile.gb10-dsv4-dspark`), `docs/*.md` (exceto `PATCHES.md`), `docs/CLAUDE/*.md`, `results/**`, `files/**`, `lmcache/**`, `scripts/ci-validate.sh`, `scripts/boot-shape-warmup.sh`, `scripts/validate_tp3.sh`, scripts de benchmark/auditoria (cabeçalhos e argumentos), `CHANGELOG.md` (inteiro), `AUDIT.md`, `CREDITS.md`, `scripts/EVAL.md`, `pi-models.dspark.example.json`, `vllm_patch_gb10/`. Cabeçalhos de alguns patches foram lidos só para descrever o mecanismo com precisão (o detalhamento completo de cada patch é de outro inventário).
> Convenção: identificadores, flags, variáveis de ambiente e código ficam no original. "arquivo:linha" refere-se ao repo acima. Onde algo é inferência minha (não dito pelo repo), marco **[inferência]**. Onde o repo se contradiz, marco **[inconsistência]**.

---

## 1. Visão geral

### 1.1 O que a receita faz, de ponta a ponta

A receita serve o modelo **`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`** (revisão fixada `86f746b36186f0e567729a5c06a8c918caba82a9`) em **dois NVIDIA DGX Spark** (SoC GB10, GPU Blackwell de consumo `sm_121`, 128 GB de memória unificada por nó) usando **vLLM com paralelismo de tensor TP=2** (uma GPU por nó, dois nós), **decodificação especulativa DSpark**, **teto de contexto de 1M tokens** (`MAX_MODEL_LEN=1048576`) e **KV cache `nvfp4_ds_mla`** (README.md:10-15). Suporte nativo a **imagens** (ViT + Aligner do checkpoint Vision-Exp; OpenAI `image_url`) é instalado por um *hotfix* de inicialização; não há codificador de vídeo (GIF vira um quadro estático) (README.md:11-15).

Fluxo completo, do ponto de vista do operador (tudo executado a partir do nó **head**):

1. **Pré-requisitos de hardware/rede**: 2 Sparks com RoCE/NCCL funcionando sobre o ConnectX-7 (QSFP direto), SSH sem senha head→worker, mesma imagem Docker nos dois nós (README.md:27-32; start:1463-1472).
2. **Configuração**: `cp .env.dspark.example .env.dspark`; definir `WORKER_HOST`, `MASTER_ADDR`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` (e `TP_`/`GLOO_`), `VLLM_HOST_IP`, `WORKER_VLLM_HOST_IP`, `HF_CACHE` (README.md:34-58).
3. **Imagem**: `docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1` nos dois nós (fixada por digest `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8` em `.env.dspark.example:219`). Imagem baseada em vLLM `0.25.2.dev0+g752a3a504.d20260714`, torch 2.11.0+cu130, CUDA 13.0, FlashInfer 0.6.15, b12x 0.15.3, triton 3.6.0, tilelang 0.1.9, NCCL 2.30.7 (pynccl) / 2.28.9 (torch), xgrammar 0.2.3 (docs/CLAUDE/fable5-1-report.md:54-55).
4. **Pesos**: `./prepare-dspark-model-cache.sh --official` baixa ~157 GiB do Hub para `HF_CACHE` no head e (padrão `DSPARK_WORKER_HF_NFS=0`) também no worker; alternativamente o worker monta o cache do head via NFSv4 (README.md:73-86, 199-225).
5. **Validação CPU opcional**: `bash scripts/ci-validate.sh` (sem GPU; não mede tok/s) (README.md:88-92).
6. **Start**: `./start-deepseek-v4-flash-dspark.sh` — valida tudo, resolve GIDs RoCEv2, sincroniza compose/env/patches para o worker, faz *preflight* `--check` dos hotfixes opt-in (worker depois head), sobe **worker primeiro (rank 1, `--headless`)** e depois **head (rank 0, serve HTTP :8888)**, espera `/v1/models`, faz uma requisição *smoke* e roda o *boot-shape warmup* (start:1574-2144).
7. **Dentro do container** (entrypoint em `docker-compose.dspark.yml:410-529`): copia o codificador do checkpoint (`encoding/encoding_dsv4.py`) para dentro do vLLM, aplica a cadeia de hotfixes *fail-closed* (cada um com `|| exit 1`) e faz `exec /usr/local/bin/vllm serve …` com ~40 flags.
8. **Verificação**: `curl /v1/models` (espera `"id": "deepseek-v4-flash-vision-exp"`, `"max_model_len": 1048576`), `./smoke-…sh` (6 requisições concorrentes), `./status-…sh`; no log de boot espera-se `Available KV cache memory: 17.04 GiB`, `GPU KV cache size: 2,331,430 tokens`, `Maximum concurrency for 1,048,576 tokens per request: 2.22x` (README.md:106-121).
9. **Tuning** (opcional): perfis alternativos (big-prompt, 200K/16 Stage-C, TP=3), hotfixes opt-in, A/B com `scripts/ab-boot.sh`.

### 1.2 Linhagem (por que as coisas são como são)

- **Pré-visualização** `DeepSeek-V4-Flash-DSpark` (FP8, DSpark γ=5, cabeça de Markov rank 256), KV `fp8`, `max-model-len 262144`, MTP-5, imagens `vllm-dspark-runtime:clean` (docs/SETUP.md:3-22) → lane histórico.
- **Stage-C** (imagem overlay local `vllm-dspark-runtime:dspark-nvfp4-stage-c`, construída por `build-dspark-vllm-runtime.sh` a partir de `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` + overlay DSpark + Stage A/B/C) — onde nasceram os *kill-switches* `VLLM_DSPARK_*` e o patch de concorrência do Keys; produziu o perfil 200K/16 (315 tok/s agregado) (README.md:567-588; recipe/*).
- **Anemll 0.1.1** (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`) — imagem pré-construída que é o padrão atual; muitos `VLLM_DSPARK_*` **não são registrados** nela (viram *warning* e são ignorados) (docs/ENVS.md:9-26).
- **0731** (`deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30…`, só texto, FP8, `n_predict=1`, MTP-5) — lane padrão até ~2026-08-31 (docs/DEEPSEEK_V4_FLASH_0731.md).
- **Vision-Exp** (`86f746b3…`, `num_nextn_predict_layers=3`, `dspark_block_size=5`) — lane atual desde 2026-08-31, com `MTP_NUM_TOKENS=6` obrigatório (CHANGELOG.md:150).

### 1.3 Fatos de arquitetura do checkpoint usados pela receita

Fonte principal: docs/CLAUDE/fable5-1-report.md:449-471 (Apêndice A) + docs/CLAUDE/item8-fp4-kv-design.md + .env.dspark.example + patches/vision_exp.

| Item | Valor | Onde |
|---|---|---|
| Camadas do decoder | 43 (índices 0–42) | fable5-1:451; start:219-221 (ablação limitada a 0–42) |
| Roteamento MoE nas camadas 0–2 | *hash-routed* (sem GEMM de roteador; roteia por id de token via `tid2eid`) | fable5-1:451, 86; CHANGELOG.md:130 |
| hidden size | 4096 | fable5-1:451 |
| Cabeças de atenção × head_dim | 64 × 512 | fable5-1:451 |
| `q_lora_rank` / `o_lora_rank` | 1024 / 1024 | fable5-1:452 |
| `o_groups` (grupos de projeção de saída) | 8 | fable5-1:452; docs/TP3.md:75 |
| dimensão RoPE | 64 (NoPE = 448) | fable5-1:452; item8:140 |
| Experts | 256 roteados, top-6, + 1 compartilhado | fable5-1:452-453 |
| `moe_intermediate_size` | 2048 | fable5-1:453 |
| Dtype dos experts | fp4 com escalas E8M0 por bloco de 32 (`fp4_e8m0_k32`, i.e. MXFP4) | fable5-1:453, 272-276 |
| Demais lineares | fp8, blocos 128×128 | fable5-1:453 |
| Lightning indexer | `index_n_heads` 64 × `index_head_dim` 128, `index_topk` 512 | fable5-1:454 |
| `sliding_window` | 128 | fable5-1:454; .env:442-444 |
| `compress_ratios` | camadas 0–1 só SWA; 2–42 alternam C4A (21 camadas) / C128A (20 camadas); camadas MTP 43–45 só SWA | fable5-1:455-456 |
| Draft (DSpark) | `num_nextn_predict_layers` 3, `dspark_block_size` 5, `dspark_markov_rank` 256 | fable5-1:456; .env:430-436 |
| Extensão de contexto | YaRN ×16 de 64K para 1M (`rope_scaling {type: yarn, factor: 16}`) | fable5-1:457; CHANGELOG.md:71 |
| Vocabulário | 129,280 (lm_head bf16 129,280×4,096 = 1.06 GB) | fable5-1:89; vision_exp/image_processor.py:300 |
| Vision tower (Vision-Exp) | ViT de 32 camadas + Aligner; patch 14, downsample 3, máx 384 tokens de imagem, `vision_dim` 1024, 16 cabeças, `vision_inter_dim` 2816, RoPE 2D θ=10000; token placeholder `<｜deepseek_image｜>` id 129264 | hotfix-dsv4-vision-exp.py:4-6; vision_exp/image_processor.py:36-39, 290-305; CHANGELOG.md:150 |
| Pesos por rank (TP=2) | 79.17 GiB | fable5-1:57 |
| Tamanho do checkpoint | ~157 GiB (Vision-Exp); ~155 GB (0731) | README.md:81; validate-dspark-config.sh:134 |
| Estrutura "hyper-connections" | o código do draft replica o residual em `hc_mult` fluxos (`x.unsqueeze(1).repeat(1, self.config.hc_mult, 1)`) e usa `HCHeadOp` de `vllm.model_executor.layers.mhc` | recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py:38, 894 |

### 1.4 As três "lanes" (perfis de serviço) documentadas

| Lane | Como ativar | O que entrega | Custo |
|---|---|---|---|
| **Padrão Anemll 1M/6 (TP=2)** | `./start-deepseek-v4-flash-dspark.sh` | 1M de teto, 6 slots, ~62–83 tok/s c=1, ~160–190 agregado c=6 (curto) | prefills grandes serializados |
| **Três Sparks (TP=3)** | `./start-tp3.sh` + `WORKER2_*` | 16 slots, ~3× KV (≈5 M tokens), ≈200 tok/s agregado a 16 streams | prefill 4–13 % mais lento até 64K, ≈22 % em 128K–256K |
| **Stage-C 200K/16 (histórico)** | imagem Stage-C + `docker-compose.stage-c.override.yml` + bloco Stage-C no `.env` | 315 tok/s estático / 205 escalonado (200K/16) | construir imagem própria; contexto 200K |

(README.md:318-349, 520-588; docs/TP3.md)

---

## 2. Inventário de parâmetros

Legenda de colunas: **nome | valor(es) default e alternativos | onde (arquivo:linha) | o que faz | conceito(s) | por que importa / trade-off**.
Abreviações de arquivos: `compose` = `docker-compose.dspark.yml`; `.env` = `.env.dspark.example`; `start` = `start-deepseek-v4-flash-dspark.sh`; `ENVS` = `docs/ENVS.md`.

### 2.1 Flags do `vllm serve` (linha de comando final renderizada pelo entrypoint)

Linha de comando: `compose:490-529`. O binário é `/usr/local/bin/vllm` (Anemll), não o `/opt/env` do Stage-C (README.md:508-509).

| nome | valor(es) | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| modelo posicional | `${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}` | compose:490; start:182 força `DSPARK_MODEL="$DSPARK_MODEL_OFFICIAL"` | Id do repositório HF a servir | HF Hub, snapshot | Mesmo com `ABLITERATED=1` serve-se o checkpoint oficial (a ablação é em runtime). README.md:168 proíbe setar `DSPARK_MODEL` à mão. |
| `--revision` | `$DSPARK_REVISION` (padrão `86f746b3…`); vazio = tip de `main` | compose:487-488, 491 | Fixa o commit do checkpoint | Reprodutibilidade, pinning (issue #19) | Evita que `main` mude pesos/encoder sem aviso; `prepare` grava `refs/main` → commit para servir offline (prepare:273-279). |
| `--served-model-name` | `${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}` (aceita aliases separados por espaço) | compose:492; .env:155 | Nome(s) expostos em `/v1/models` | API OpenAI | Probes/warmup/smoke usam o 1º alias (start:2090). |
| `--host` | `${VLLM_HOST:-127.0.0.1}` no compose; `.env` define `0.0.0.0`; CLI `--host` sobrescreve | compose:493; .env:158; start:280 | Endereço de bind do servidor HTTP | Rede | `0.0.0.0` expõe a API na LAN sem autenticação → launcher emite WARN (start:360-376). Healthcheck mapeia wildcard para 127.0.0.1 (compose:25). |
| `--port` | `${VLLM_PORT:-8888}`; CLI `--port` | compose:494; .env:160; start:281-294 | Porta HTTP | — | Validada 1–65535 com normalização decimal (`10#`). |
| `--trust-remote-code` | sempre | compose:495 | Permite código custom do checkpoint | HF remote code | Necessário para arquitetura DeepSeek V4. |
| `--api-key` (condicional) | vazio (sem auth) / 1 chave (`VLLM_API_KEY`) / N chaves (`DSPARK_API_KEYS` → **um** `--api-key k1 k2 …`) | compose:419, 496 | Autenticação Bearer nas rotas `/v1`, `/v2`, `/inference` | Auth; argparse `nargs` | Repetir a flag sobrescreveria (é `list[str]`), por isso uma flag com N valores (compose:162-167). Rotas fora desses prefixos (`/invocations`, `/generative_scoring`, `/tokenize`, `/metrics`…) continuam sem chave (.env:172-177). |
| `--quantization modelopt_gb10_hybrid` (condicional) | só se `ENABLE_VLLM_GB10_PATCH=1` | compose:466-470, 497 | Plugin experimental Marlin W4A16 (M<limiar) / CUTLASS W4A4 (M≥limiar) | Quantização NVFP4, dispatch por M | Default off; exige checkpoint ModelOpt NVFP4 W4A4 (vllm_patch_gb10/README.md). Não se aplica ao checkpoint DSV4 padrão **[inferência: o DSV4 tem experts MXFP4, não ModelOpt]**. |
| `--tensor-parallel-size` | `${TP_SIZE:-2}`; start força 2 (ou 3 via `start-tp3.sh`) | compose:498; start:604-614 | Número de GPUs no grupo TP | Tensor parallelism | O launcher ignora `TP_SIZE` do `.env` no caminho de 2 nós para que um snippet copiado não suba nnodes=3 com um worker (start:604-605). |
| `--pipeline-parallel-size` | `1` | compose:499 | Sem pipeline parallel | PP | Todo o modelo em cada rank particionado por TP. |
| `--kv-cache-dtype` | `nvfp4_ds_mla` (fixo) | compose:500 | Formato do KV cache MLA | KV quantization; layout DS-MLA | Apesar do nome, é **byte-idêntico a `fp8_ds_mla`**: 584 B/linha = 448 B NoPE fp8 e4m3 + 128 B RoPE bf16 + 8 B (7 escalas UE8M0 + 1 pad) (item8:140; fable5-1:342-346). O "nvfp4" só muda o *dispatch* de kernel (issue #22). README.md:600-601: "Stage C padded NVFP4 path (584-byte …). Not the abandoned 416-byte 'true layout'". |
| `--block-size` | `256` | compose:501 | Tokens por bloco do KV paginado (grupo MLA) | PagedAttention | Grupos SWA usam blocos 64/4/8 com mesmo tamanho de página em bytes (item8:144). Também é o grão do *prefix cache hash* (issue #144: blocos de 256 tokens). |
| `--max-model-len` | `${MAX_MODEL_LEN:-1048576}`; alt. `200000` (perfil Keys/alta concorrência) | compose:502; .env:232 | Teto de tokens por requisição (prompt+saída) | Contexto; YaRN | É **teto, não reserva**: a restrição real é `sum(tokens vivos) ≤ pool de KV` (README.md:150-152, 484-500). Requer `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. |
| `--max-num-seqs` | `${MAX_NUM_SEQS:-6}`; alt. 4 (big-prompt), 16 (Stage-C 200K / TP3), 32 (dados de performance, não recomendação) | compose:503; .env:233, 522-583 | Máximo de sequências ativas simultâneas | Continuous batching | Também define tamanho de captura de CUDA graph e "verify rows" = seqs×(k+1); falhas estocásticas #141 parecem crescer com linhas de verificação (.env:542-575). |
| `--max-num-batched-tokens` | `${MAX_NUM_BATCHED_TOKENS:-8192}`; alt. 16384 (big-prompt), 12288 (com N=32) | compose:504; .env:234, 526-531 | Orçamento de tokens por passo do scheduler | Token budget, chunked prefill | Com k=6 e admissão alta, os slots de draft consomem o orçamento (warning `max_num_scheduled_tokens is set to …`) (.env:529-531). |
| `--long-prefill-token-threshold` | `${LONG_PREFILL_TOKEN_THRESHOLD:-1024}`; `0` = sem teto; `2048` testado (rejeitado) | compose:505; .env:235-245 | Tamanho máximo de cada *chunk* de prefill longo | Chunked prefill; fairness | Com 1024 um chunk fica bem abaixo do orçamento 8192 → decodes sempre recebem orçamento (issue #27). `2048` custou ~1,5 GB de RAM do host no head (MemAvailable 2.55→0.99 GB) porque workspaces maiores vivem em memória unificada fora do orçamento `GPU_MEMORY_UTILIZATION` (ab-results-2026-09-03:84, 99). |
| `--max-cudagraph-capture-size` | `$(( (MAX_NUM_SEQS*(MTP_NUM_TOKENS+1)+7)/8*8 ))` → 48 a 6×6; 112 a 16×7; 40 a 6×(5+1) com block-k | compose:506 | Maior tamanho de batch (em tokens) capturado em CUDA graph | CUDA graphs; `uniform_decode_query_len` | O valor "cru" 42 era truncado para 40 e arredondado a múltiplos de 7 → `[7,14,21,28,35]`, deixando o passo c=6 (42 tokens) **eager**. Arredondar para múltiplo de 8 (48) mantém `[7,…,42]` → c=6 agregado 139 vs 122 (−12 % sem isso) (fable5-1:169-196; ab-results-2026-09-03:89, 107). |
| `--gpu-memory-utilization` | `${GPU_MEMORY_UTILIZATION:-0.835}` = `GPU_MEMORY_UTILIZATION_TEXT` | compose:507; start:161-162; .env:421-423 | Fração da memória que o vLLM pode usar (pesos+ativações+KV) | UMA; profiling de memória | Maior = pool de KV maior. Em memória unificada compete com o SO/host: cada +0,01 ≈ +1,2 GiB KV ≈ +165 K tokens (fable5-1:165-166). Se captura OOM, baixar para ~0.78 (.env:230-231). Não setar `GPU_MEMORY_UTILIZATION` à mão (start sobrescreve). |
| `--enable-prefix-caching` | sempre | compose:508 | Reuso de blocos KV com hash de prefixo idêntico | Prefix caching (hash encadeado) | Grande ganho para agentes com system prompt/tool schema longos; interage com SWA (#26/#36) e com diretiva de esforço (#144). |
| `--enable-prompt-tokens-details` | sempre | compose:509 | Retorna `usage.prompt_tokens_details.cached_tokens` | Observabilidade de cache | Permite verificar hits (ex. results/smoke-text-only.json mostra `cached_tokens: 0`). |
| `--async-scheduling` (condicional) | `DSPARK_ASYNC_SCHEDULING=1` (default) → presente; `0` → removido nos dois ranks | compose:458, 510 | Scheduler prepara o passo N+1 enquanto a GPU executa o N | Async scheduling; overlap CPU/GPU | `0` só para A/B do issue #191 (bitmask de gramática vira de drafts reais; ~4 % de custo single-stream) (.env:375-377). Args devem bater entre ranks. |
| `--attention-config {"use_fp4_indexer_cache":true}` (condicional) | só com `DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1` | compose:459, 511 | Cache K do Lightning indexer em MXFP4 | Quantização do indexer | Logits do indexer leem metade dos bytes por chave; alocação continua 132 B/linha na imagem fixada (.env:461-471). |
| `--enable-chunked-prefill` | sempre | compose:512 | Divide prefills longos em chunks misturados com decode | Chunked prefill | Base do controle de fairness (#27/#43). |
| `--speculative-config` | `{"method":"dspark","num_speculative_tokens":${MTP_NUM_TOKENS:-6},"draft_sample_method":"${DRAFT_SAMPLE_METHOD}"}` | compose:475-479, 513 | Liga DSpark com k tokens especulativos | Speculative decoding; DSpark | k deve ser ≥ `dspark_block_size` (5) e divisível por `num_nextn_predict_layers` (3) → 6, 9… (start:254-276). JSON é montado dentro do container (valores de `.env` são *sourced* e perderiam aspas). |
| `--tokenizer-mode deepseek_v4` | fixo | compose:514 | Usa o wrapper de tokenizer DeepSeek-V4 do vLLM, que chama o encoder Python do checkpoint | Chat template vs encoder custom | O checkpoint **não traz Jinja chat_template**; traz `encoding/encoding_dsv4.py` (docs/DEEPSEEK_V4_FLASH_0731.md:22-24). |
| `--limit-mm-per-prompt` | JSON `{"image":8}` (padrão `image=8` convertido); `image=N` | compose:489, 515; .env:424-428 | Máximo de itens multimodais por requisição | Multimodal | Argparse do Anemll é `json.loads`; um `{"image":8}` no `.env` perde as aspas ao ser *sourced* → por isso a forma `image=N` (.env:425-427). Antes de 2026-09-06, processor limitava a 16 (CHANGELOG.md:82). |
| `--distributed-executor-backend mp` | fixo | compose:516 | Executor multiprocessing (sem Ray) | Multi-node vLLM | Combina com `--nnodes/--node-rank/--master-addr/--headless`. |
| `--moe-backend flashinfer_b12x` | fixo | compose:517 | Backend de MoE B12X (CuTeDSL) do FlashInfer | Fused MoE; W4A16 | Log: `Using 'B12X_MXFP4' Mxfp4 MoE backend`, `W4A16FusedMoeKernel` (fable5-1:275). Como a fonte é `fp4_e8m0_k32`, b12x só roda W4A16 (dequant→bf16 MMA) (fable5-1:271-279). |
| `--tool-call-parser deepseek_v4` | fixo | compose:518 | Parser de chamadas de ferramenta DSML | Tool calling | Converte markup `<｜DSML｜tool_calls>/<｜DSML｜invoke>` em `tool_calls` estruturados (docs/DSML_SYNTAX_TEMP_ASYMMETRY.md:12-14). |
| `--enable-auto-tool-choice` | fixo | compose:519 | Permite `tool_choice:"auto"` | Tool calling | Necessário para agentes. |
| `--reasoning-parser deepseek_v4` | fixo | compose:520 | Separa `<think>…</think>` em `reasoning_content` | Reasoning models | Base para suprimir stops no raciocínio e para thinking budget. |
| `--reasoning-config` | `{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}` | compose:521 | Marcadores de início/fim do raciocínio | Reasoning | Usado pelo budget de thinking e supressão de stops. |
| `--default-chat-template-kwargs` | derivado de `DEFAULT_THINKING`: `off`→`{"thinking":false}`, `low`→`{"thinking":true,"reasoning_effort":"low"}`, `high`→`…"high"`, `max`→`…"max"` | compose:480-486, 522 | Valor padrão de `chat_template_kwargs` quando o cliente não envia | Thinking modes / reasoning effort | Kwargs da requisição sempre vencem. `max` ≈3× latência vs `low` (.env:499-503). |
| `--generation-config vllm` | fixo | compose:523 | Usa defaults de amostragem do vLLM, **não** o `generation_config.json` do checkpoint | Sampling params | Sem `--override-generation-config`: cliente controla temperatura/top_p (.env:519-520; GLM-NEW-REPORT §3.3). |
| `--enable-flashinfer-autotune` | fixo | compose:524 | Autotuning de kernels FlashInfer no boot | Autotune | Cache persistido em `vllm-cache/flashinfer_autotune_cache/<hash>/` (fable5-1:365-366). Warmup sparse-MLA só cobre `tokens=16` (fable5-1:362-364). |
| `--nnodes` | `${NNODES:-2}` (3 em TP3) | compose:525 | Número de nós | Multi-node | — |
| `--node-rank` | `${NODE_RANK}` (0 head, 1 worker, 2 worker2) | compose:526 | Rank do nó | Multi-node | Injetado pelo launcher (compose_base / remote_compose). |
| `--master-addr` / `--master-port` | `${MASTER_ADDR}` / `${MASTER_PORT:-25000}` | compose:527-528; .env:35-36 | Endpoint TCPStore/rendezvous do rank 0 | Rendezvous distribuído | Deve ser IP da malha (RoCE) do head. |
| `--headless` | só se `HEADLESS` não vazio (worker) | compose:529 | Worker não sobe API HTTP | Multi-node | Healthcheck do worker sai 0 se `$HEADLESS` (compose:25). |

### 2.2 Configuração do container (Compose)

| nome | valor | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `image` | `${DSPARK_VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}`; `.env` fixa com digest | compose:3; .env:219 | Imagem de runtime | Container pinning | Digest impede retag silencioso; mesma digest nos dois nós (README.md:508). |
| `pull_policy` | `if_not_present` | compose:4 | Não puxa se existir | — | Start recusa se imagem ausente em qualquer nó (start:1457-1472). |
| `entrypoint` | `[]` | compose:5 | Anula entrypoint da imagem | — | O `command` bash controla toda a cadeia. |
| `network_mode` | `host` | compose:6 | Rede do host | RDMA/NCCL | Necessário para RoCE, TCPStore e ZMQ diretos nas interfaces físicas. |
| `ipc` | `host` | compose:7 | IPC do host | Shared memory | vLLM usa shm entre processos (shm_broadcast). |
| `shm_size` | `64gb` | compose:8 | /dev/shm grande | Shared memory | Buffers de broadcast/IPC. |
| `restart` | `${DSPARK_RESTART_POLICY:-unless-stopped}`; alt. `no` | compose:11; .env:44 | Reinicia rank que caiu | Supervisão | Após reboot o dockerd restaura ranks e `start` sai com **3** ("já rodando"); systemd: `SuccessExitStatus=3` (README.md:102; ENVS:178). Reinícios independentes podem separar o par TP (CHANGELOG.md:134). |
| `stop_grace_period` | `${DSPARK_STOP_GRACE:-10s}` | compose:12; .env:45 | Tempo de graça no stop | — | vLLM ignora SIGTERM durante load; 180s travava `compose stop` (ENVS:179). |
| `healthcheck` | `curl /health` em `VLLM_HOST` (wildcard→127.0.0.1), pula se `HEADLESS`; interval 30s, timeout 10s, retries 3, `start_period 480s` | compose:24-29 | Saúde do head | — | 480s cobre load a frio (~6 min medido) (compose:17-18). Issue #185 corrigiu probe para bind em IP de LAN. |
| `ulimits.memlock` | `-1` | compose:32 | Memória travável ilimitada | RDMA | Registro de memória RDMA (verbs) exige memlock. |
| `ulimits.stack` | `67108864` (64 MiB) | compose:33 | Pilha maior | — | **[inferência]** para threads de compilação/JIT. |
| `gpus` | `all` | compose:35 | Acesso à GPU | — | — |
| `devices` | `/dev/infiniband:/dev/infiniband` | compose:37-38 | Expõe dispositivos verbs | RDMA | NCCL IB/RoCE dentro do container. |
| volume HF | `${HF_CACHE}:/cache/huggingface` | compose:41 | Cache HF + todos os caches JIT | Persistência | Triton/TileLang/b12x/vLLM/FlashInfer/NCCL-FR vivem aqui para sobreviver a recreate. |
| volume GB10 plugin | `${VLLM_GB10_PATCH_DIR:-./vllm_patch_gb10}:/opt/vllm-gb10-hybrid-nvfp4` | compose:45 | Plugin opcional | — | Só usado com `ENABLE_VLLM_GB10_PATCH=1`. |
| volumes de hotfix | cada `patches/*.py` montado `:ro` em `/opt/…` e `patches/` inteiro em `/opt/dspark-patches:ro`, `patches/tp3` em `/opt/dsv4-tp3:ro` | compose:47-114 | Disponibiliza patchers no container | Source patching | Cada caminho pode ser sobrescrito por variável `DSPARK_*_HOTFIX` (ex. `DSPARK_ISSUE27_HOTFIX`). |
| `/tmp` | `${DSPARK_TMP_HOST:-${HOME}/.cache/dspark-tmp}:/tmp` | compose:115 | /tmp do container no host | FIFO do flight recorder | O FIFO `fr_dump_pipe_<rank>.pipe` fica visível no host (root-owned) (ENVS:75). |
| override NFS (worker) | `dspark-hf:/cache/huggingface:ro` + overlays locais `${DSPARK_JIT_CACHE}/{triton-cache,tilelang-cache,vllm-cache,flashinfer,b12x-cute-cache,nccl-fr}` | docker-compose.dspark-nfs.override.yml:1-18 | Worker lê pesos do head via NFS, JIT local gravável | NFS, overlay mounts | Evita 2ª cópia de 157 GiB; volume criado com `nfsvers=4.2,ro,nconnect=8,rsize=wsize=1048576,hard,timeo=600` (files/nfs-share.sh:116-120). |
| override Stage-C | injeta `DSPARK_SLOT_CLAMP`, `VLLM_USE_B12X_WO_PROJECTION`, `VLLM_DSPARK_*`, `VLLM_DSV4_*`, `VLLM_TRITON_MLA_SPARSE`, `VLLM_SKIP_INIT_MEMORY_CHECK` | docker-compose.stage-c.override.yml:14-37 | Kill-switches da imagem Stage-C | — | No-op (warning) no Anemll 0.1.1. `start` **não** mescla esse arquivo sozinho (README.md:571-573). |

### 2.3 Topologia do cluster / launcher

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `WORKER_HOST` | IP/host (ssh target, aceita `user@`) — obrigatório | .env:9; start:597 | Worker (rank 1) | Multi-node | Start exige SSH sem senha (`BatchMode=yes`, `ConnectTimeout=10`) (start:21-22). |
| `WORKER_DIR` / `WORKER_SCRIPT_DIR` | vazio = mesmo caminho do head; `WORKER_SCRIPT_DIR` tem precedência | .env:30-34; start:618 | Onde fica o checkout no worker | — | Start sincroniza compose/env/patches para lá. |
| `MASTER_ADDR` | IP RoCE do head (ex. `10.0.0.1`) — obrigatório | .env:35; start:598 | Rendezvous | TCPStore | — |
| `MASTER_PORT` | `25000` | .env:36 | Porta do rendezvous | — | — |
| `NODE_RANK` / `HEADLESS` | head `0`/vazio; worker `1`/`1` | .env:37-49 | Rank e modo headless | — | Injetados pelo launcher, não por `.env` compartilhado (`env -u NODE_RANK -u HEADLESS`) (start:689, 1196). |
| `VLLM_HOST_IP` / `WORKER_VLLM_HOST_IP` | IP RoCE de cada nó (padrões `MASTER_ADDR` / `WORKER_HOST`) | .env:208-211; start:616-617 | IP que o vLLM/ZMQ distribuído usa por nó | Bind distribuído | Evita o worker bindar no IP da malha do head (.env:208-209). |
| `PROJECT_NAME` | `deepseek-v4-flash` | start:7 | Projeto Compose | — | Container `deepseek-v4-flash-vllm-dspark-1`. |
| `WAIT_ATTEMPTS` / `WAIT_SECONDS` | `100` / `15` | start:8-9 | Espera pela API (≈25 min) | — | Boot frio ~6–8 min; TP3 1º boot ~8 min. |
| `ENV_FILE` / `COMPOSE_FILE` | `.env.dspark` / `docker-compose.dspark.yml` | start:5-6 | Arquivos usados | — | `COMPOSE_FILE` troca para `docker-compose.lmcache.yml` no modo LMCache. |
| `DSPARK_PROPOSER_FILE` | `recipe/vllm/v1/spec_decode/dspark_proposer.py` | start:12, 1451-1454, 1607 | Proposer DSpark sincronizado ao worker | Overlay | Start exige que exista e copia; o compose Anemll **não** o monta (comentário compose:42-43 diz "Optional Stage-C overlay") **[inconsistência menor: arquivo é copiado mas não usado no lane Anemll]**. |
| `--host` / `--port` (CLI) | override pontual | start:24-74 | Sobrescreve `VLLM_HOST`/`VLLM_PORT` | — | Precedência CLI > env. |
| Exit codes do start | 0 ok; 1 falha; 2 config inválida; 3 já rodando (head ou worker stale) | start:1486-1537 | Contrato de supervisão | systemd | Sentinel 42 remoto distingue "rank existe" de falha de ssh (255) (start:1501-1521). |
| Normalização do env | remove BOM e CRLF numa cópia privada `0600`; worker recebe cópia atômica `0600` | start:86-156, 1589-1605 | Robustez de parsing | Env sourcing | Operador mantém o arquivo byte-idêntico (CHANGELOG.md:233). |
| `dspark_validate_numeric_knobs` | `MAX_NUM_SEQS` 1–4096, `MTP_NUM_TOKENS` 0–64, `MAX_NUM_BATCHED_TOKENS` 1–8388608; rejeita >10 dígitos; normaliza `10#` | dspark-numeric-knobs.sh:18-46; start:296-297 | Valida números antes de aritmética | Shell arithmetic | Evita overflow 64-bit/octal que vazaria para o `$(( ))` do compose. |

### 2.4 Malha de rede: NCCL / RoCE / Gloo

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `NCCL_IB_HCA` | ex. `rocep1s0f1`; dual-HCA recomendado `=rocep1s0f0,roceP2p1s0f0` (sintaxe NCCL: `^` exclui, `=` match exato, lista `name[:port[:rail[:plane]]]`) — obrigatório | .env:53, 65-72, 697; compose:261; start:600 | Seleciona adaptadores RDMA para NCCL | HCA, RoCE | A porta QSFP do GB10 aparece como **2 NICs virtuais** (2 controladores PCIe Gen5 x4 ~100G cada); com uma só HCA roda-se a meia porta (.env:691-697). Parser do launcher espelha `parseStringList` do NCCL (32 entradas, nomes truncados em 63 bytes, porta via atoi base 10) (start:812-846). |
| `NCCL_SOCKET_IFNAME` | ex. `enp1s0f1np1` — obrigatório | .env:54; compose:262 | Interface para bootstrap TCP do NCCL | NCCL bootstrap | Também é a interface do export NFS (`NFS_IFACE`) (start:776). |
| `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `enp1s0f1np1` (padrão = `NCCL_SOCKET_IFNAME`) | .env:55-56; compose:263-264 | Interfaces de sockets TP e Gloo (grupos CPU) | Gloo | Em TP3 vão para `TP3_BOOTSTRAP_IFNAME` (LAN). |
| `WORKER_NCCL_IB_HCA`, `WORKER_NCCL_SOCKET_IFNAME`, `WORKER_TP_SOCKET_IFNAME`, `WORKER_GLOO_SOCKET_IFNAME` | padrões = valores do head; ex. `rocep1s0f0`/`enp1s0f0np0` | .env:57-64; start:630-633 | Overrides por nó | QSFP ring | Em ring, a porta que "encara" o outro nó tem nome diferente; o start injeta no compose remoto (rank-aware) (.env:58-60; start:1172-1193). |
| `NCCL_IB_GID_AUTO` | `1` (default); `0` = pinagem manual | .env:73-96; start:657-663, 1095-1170 | Valida em sysfs cada HCA/porta selecionada e **deixa `NCCL_IB_GID_INDEX` indefinido** | RoCEv2 GID | O índice GID RoCEv2 difere por nó/HCA e muda após reboot; um pin global pode travar init (`ibv_modify_qp 61`) em dual-HCA (CHANGELOG.md:109). Falha fechado se algum membro não tem GID RoCE v2 que case com IPv4 próprio. |
| `NCCL_IB_GID_INDEX` / `WORKER_NCCL_IB_GID_INDEX` (/ `WORKER2_…`) | vazio (auto); ex. `3` / `6` se AUTO=0 | .env:93-96, 725-728; compose:265 | Pin manual do índice GID | GID table | Com AUTO=1 são reportados e ignorados; entrypoint faz `unset` de valor vazio porque NCCL leria vazio como índice 0 (link-local fe80) (compose:280-285, 427). |
| `NCCL_IB_GID_MATCH_IP` / `WORKER_NCCL_IB_GID_MATCH_IP` | vazio (usa IPv4 da ifname); ex. `10.0.22.1` | .env:97-99; start:664-667 | IP para casar o sufixo `ffff:aabb:ccdd` do GID | IPv4-mapped GID | Só se o endereço RoCE não for o IPv4 primário da interface. |
| `NCCL_CROSS_NIC` | `1` | .env:100; compose:268 | Permite anéis cruzarem NICs diferentes | NCCL topologia | Útil com múltiplas HCAs. |
| `NCCL_NET` | `IB` | .env:663; compose:259 | Usa plugin de rede IB (verbs, inclui RoCE) | NCCL transport | Força RDMA em vez de sockets. |
| `NCCL_IB_DISABLE` | `0` | .env:664; compose:260 | Não desabilita IB | — | — |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET` | compose:266 | GIDs IPv4 | RoCEv2 | — |
| `NCCL_IB_ROCE_VERSION_NUM` | `2` | compose:267 | RoCE v2 (UDP/IP roteável) | RoCEv2 | — |
| `NCCL_CUMEM_ENABLE` | `0` | .env:665; compose:269 | Desliga alocação via cuMem | NCCL memory | **[inferência]** evita problemas de cuMem/VMM em memória unificada. |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | .env:666; compose:270 | Ignora afinidade de CPU | — | — |
| `NCCL_DEBUG` | `WARN` | .env:667; compose:271 | Nível de log | — | `INFO` seria necessário para ver lista de dispositivos NET/IB (fable5-1:500). |
| `NCCL_NVLS_ENABLE` | `0` | .env:668; compose:272 | Desliga NVLink SHARP | NVLS | Não há NVSwitch no Spark. |
| `NCCL_IB_MERGE_NICS` | unset (NCCL default 1 = *permite* fundir NICs dual-port); `0` desliga | .env:677-683; compose:287 | Fusão de portas compatíveis em um NIC lógico | NIC merging | Em TP3 o start força `0` (cada porta vai a um vizinho diferente) (start:1565-1568). Unset é realmente ausente (entrypoint faz unset de vazio, compose:420) para não mascarar `/etc/nccl.conf`. |
| `NCCL_IB_SUBNET_AWARE_ROUTING` / `NCCL_IB_SUBNET_PREFIX_LEN` | unset; TP3: `1` / `24` | compose:288-289; start:1569-1570 | Emparelha GIDs por sub-rede | Subnet-aware routing | Default /16 trataria 10.0.22 e 10.0.23 como mesma sub-rede. |
| `NCCL_NET_GDR_LEVEL` / `NCCL_NET_GDR_READ` / `NCCL_DMABUF_ENABLE` | unset (passthrough); ex. `SYS`/`1`/`0` | .env:730-739; compose:290-292 | Controles GPUDirect RDMA | GDR, DMA-BUF | Num stack GB10 (driver 580.173.02) `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0`, transporte `via NET/IB/x` sem `/GDRDMA` → coletivos passam por buffer do host (ENVS:85; fable5-1:218-220). |
| `NCCL_GIN_ENABLE` | unset (= habilitado); `0` = init de comunicador via CPU | .env:685-689; compose:293-296 | GPU-initiated networking | NCCL GIN | Medido 2026-09-05 a ~97 % de pressão de memória: comm-init ~2 min → ~13 s, banda inalterada. Só acelera bootstrap. |
| MTU | 9000 nas duas interfaces controladoras (RoCE `active_mtu` 4096) | .env:699-703; fable5-1:210-211, 229 | Quadros jumbo | RoCE MTU | Padrão 1500 limita RoCE MTU a 1024. Par 1: 1311 → 1343 (MTU 9000) → 1410 tok/s (duas HCAs), +7,6 % (.env:705-707). |
| `TP3_BOOTSTRAP_IFNAME` | `enP7s7` (RJ45 LAN 192.168.1.0/24) | start:1548; ENVS:92 | Gloo/NCCL-socket/TP TCP de TP3 na LAN | Bootstrap | Pares CX /24 não alcançam todos os ranks; nunca `lo` (Gloo bindaria 127.0.0.1) (start:1539-1547). |
| `TP3_NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1` | start:1560 | Duas portas CX em TP3 | Ring 3 nós | Uma HCA só dá timeout em QP RTR (spark1 tentando 10.0.22.1→10.0.23.3) (start:1559). |
| `TORCH_FR_BUFFER_SIZE` | `2000` | .env:636; compose:244 | Ring buffer do *flight recorder* do ProcessGroupNCCL | NCCL forensics | Pré-requisito para dump-on-timeout. |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` / `TORCH_NCCL_ENABLE_MONITORING` | `1` / `1` | .env:637-638; compose:245-246 | Dump no timeout do watchdog (600 s) | Watchdog | Deixa evidência por rank quando o par morre (#117). |
| `TORCH_FR_DUMP_TEMP_FILE` | `/cache/huggingface/nccl-fr/comm_lib_trace_rank_` | .env:639; compose:247 | Onde gravar dumps | — | Nome estático por rank e truncado a cada dump → arquivar os dois ranks juntos (.env:629-635). |
| `TORCH_NCCL_DEBUG_INFO_PIPE_FILE` | `/tmp/fr_dump_pipe_` | .env:640; compose:248 | FIFO para disparar dump sob demanda | — | Escrever qualquer coisa em `/tmp/fr_dump_pipe_<rank>.pipe`; assíncrono → esperar "Finished writing Flight Recorder debug info". |

### 2.5 Pesos, Hugging Face, NFS

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `HF_CACHE` | `${HOME}/.cache/huggingface` | .env:103; compose:41 | Cache do Hub no host | HF cache | Montado em `/cache/huggingface`. |
| `WORKER_HF_CACHE` | vazio = `HF_CACHE`; com NFS=1 é o diretório de overlays JIT do worker | .env:104-109; start:619, 1894-1899 | Cache do worker | — | Com NFS=1 e vazio, resolve `$HOME/.cache/huggingface` do worker. |
| `DSPARK_WORKER_HF_NFS` | `0` (default: 2ª cópia no worker); `1` (worker monta head via NFSv4) | .env:110; start:620-626, 775-790, 1891-1941 | Estratégia de distribuição de pesos | NFSv4 | `1` evita 2º download; reutiliza exportador existente (ex. `vllm-fn-nfs` do Qwen) ou sobe `dspark-nfs`. |
| `NFS_SERVER_IP` | unset = IPv4 de `NCCL_SOCKET_IFNAME` (ex. `10.0.22.1`) | .env:111-113; files/nfs-share.sh:25-31 | IP do exportador | — | Não usar o alias 10.0.0.1 do loopback. |
| `NFS_VOLUME` / `NFS_CONTAINER` / `NFS_IMAGE` / `NFS_IFACE` | `dspark-hf` / `dspark-nfs` / `dspark-nfs:local` / `NCCL_SOCKET_IFNAME` | start:624-625, 776; nfs-share.sh:9-12 | Nomes do volume/servidor NFS | — | `./stop … --nfs` remove só `dspark-nfs` (nunca `vllm-fn-nfs`). |
| Opções do export | `ro,sync,no_subtree_check,no_root_squash,insecure,fsid=0`; clientes = IP do worker + /24 da interface | files/nfs-server/entrypoint.sh:7; nfs-share.sh:33-52 | Export somente leitura | NFS | `rpc.nfsd 8`, só NFSv4 (desabilita v2/v3). |
| `WORKER2_NFS_SERVER_IP` | ex. `10.0.23.1` | .env:23; start:1921-1932 | IP do head no link spark1↔spark3 | Ring | Worker2 está em outro /24; start adiciona o /24 ao export (`nfs_grant_subnet`). |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` / `1` (serve); `prepare` força `0` | .env:115-116; prepare:223-224, 244-245 | Não acessa o Hub ao servir | — | Evita download silencioso de 155 GB se a revisão não estiver no cache (validate-dspark-config:108-141). |
| `HF_HUB_DISABLE_XET` | `1` | .env:117; compose:125 | Desliga o backend Xet | HF transfer | **[inferência]** estabilidade de download. |
| `HF_HUB_DOWNLOAD_TIMEOUT` / `HF_HUB_ETAG_TIMEOUT` | `120` / `30` (hf_hub default 10/10) | .env:118-121; prepare:247-248 | Timeouts do Hub | — | 10 s abortava shards de vários GB (CHANGELOG.md:248). |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | shell > `.env` > `~/.cache/huggingface/token` | .env:122-125; prepare:81-127 | Autenticação do Hub (só prepare) | — | Obrigatório para o repo gated da ablação; `.env` é copiado ao worker → preferir shell. |
| `HF_DOWNLOAD_WORKERS` | `1` (prepare); ex. `4` | .env:126-127; prepare:76 | Downloads paralelos de shards | — | 1 é seguro para disco mas lento. |
| `DSPARK_MODEL_OFFICIAL` / `DSPARK_MODEL_ABLITERATED` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` / `drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit` | .env:136-137 | Ids dos repos | — | Não apontar para o dump ablit do 0731 (perde `image_url`) (README.md:181). |
| `DSPARK_REVISION` | `86f746b36186f0e567729a5c06a8c918caba82a9`; vazio = tip | .env:149; start:181-185 | Commit do checkpoint | Pinning | Unset → padrão; `DSPARK_REVISION=` (vazio explícito) → tip. |
| `DSPARK_REVISION_ABLITERATED` | vazio (tip) | .env:150 | Pin do repo ablit | — | — |
| `DSPARK_ENCODING_FILE` | vazio = auto-descoberta de `encoding/encoding_dsv4.py` no snapshot (prefere `DSPARK_REVISION`; fallback 0731 / `/models/...`) | .env:151-154; compose:430-444 | Encoder do checkpoint a instalar no vLLM | Encoder custom | Sem ele: WARN e pula cópia/mapeamento de esforço/#21 (compose:444). Deve existir em head **e** worker (README.md:510-511). |
| `ABLITERATED` | `0` (oficial) / `1` (oficial + projeção de direção de recusa em runtime) | .env:128-135; start:175-252 | Flag de produto para ablação | Abliteration | `1` implica `ABLATE=1`; exige `RESPONSIBLE_USE.md` + `direction_r1.pt` (18 KiB) baixados de repo gated; não baixa checkpoint de 157 GiB. |

### 2.6 Perfil de serviço (shape do lane)

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `MAX_MODEL_LEN` | `1048576`; alt. `200000` | .env:232 | → `--max-model-len` | Contexto | Ver 2.1. |
| `MAX_NUM_SEQS` | `6`; alt. 4, 16, 32, 13 (um operador em produção, "sem prova de segurança") | .env:233, 570 | → `--max-num-seqs` | Batching | "NO TESTED MAX_NUM_SEQS VALUE IS ESTABLISHED AS GENERALLY SAFE" (.env:556). |
| `MAX_NUM_BATCHED_TOKENS` | `8192`; alt. 16384, 12288 | .env:234 | → `--max-num-batched-tokens` | Token budget | Subir não ajudou o warning 8150/8168 (fable5-1:44-45). |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024`; `0`; `2048` (rejeitado por RAM) | .env:245; README.md:277 | → `--long-prefill-token-threshold` | Chunked prefill | Ver 2.1. |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `1` (default); `2` (opt-in qualificado #217); `3` (opt-in limitado); >3 → 3; inválido → stock `max_num_partial_prefills` com 1 warning | .env:247-269; compose:155; ENVS:93 | Máximo de prefills parciais simultâneos (lido pelo hotfix #27 na construção do Scheduler) | Partial prefills, fairness | Anemll 0.1.1 rejeita `--max-num-partial-prefills`. Default oscilou: 2 (08-19) → 1 (08-28, #154) → 2 (09-02) → 1 (09-04, #211) (CHANGELOG). |
| `GPU_MEMORY_UTILIZATION_TEXT` | `0.835`; alt. 0.87 (big-prompt), 0.80 (conservador), 0.78 (se captura OOM), 0.83/0.85 (A/Bs) | .env:423; README.md:279 | → `--gpu-memory-utilization` | UMA | Ver 2.1. ENVS:190 ainda diz "≈0.80" **[inconsistência: doc antiga]**. |
| `LIMIT_MM_PER_PROMPT` | `image=8` (→ `{"image":8}`); `image=32` testado (33 → 400) | .env:428; compose:489 | → `--limit-mm-per-prompt` | Multimodal | Ver 2.1. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` (múltiplo de 256) | .env:279-287; compose:136 | Esparsifica checkpoints de prefix cache de SWA: um "tail" por segmento de 4096 tokens + fronteira de replay | Prefix caching híbrido | Sem isso, grupos SWA (que liberam blocos antigos por design) zeram o hit comum; com v2 do #26 a SWA pode encolher o hit até o último tail retido (warm 8/8 em x8 22.8K/44.7K/88.4K) (CHANGELOG.md:347-355). |
| Perfil "big prompt" | `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=16384`, `GPU_MEMORY_UTILIZATION_TEXT=0.87` | README.md:154-160 | Mais throughput de prefill | — | Mantém 1M. |
| Perfil "admissão alta" (dados) | `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=12288` | .env:522-541 | c=1 47–54, c=8 154, c=16 213, c=32 325 agregado | — | Pool de KV quase invariante a N (~2,1× 1M a N=32 vs ~2,6× a N=4). Estabilidade #141 não garantida. |
| Perfil "conservador" | `MAX_NUM_SEQS=6`, `GPU_MEMORY_UTILIZATION_TEXT=0.80` | .env:576-579 | — | — | — |

### 2.7 Decodificação especulativa (DSpark)

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `MTP_NUM_TOKENS` (k) | `6` (Vision-Exp); histórico 3 → 5 (0731, 2026-07-29) → 6 (Vision-Exp 2026-08-31); com block-k: 5 | .env:429-437; compose:305; start:254-276 | `num_speculative_tokens` | Speculative decoding | Regra: k ≥ `dspark_block_size` (5) (k<5 trunca blocos: silencioso no 0.25.2, rejeitado no vLLM ≥0.26) e `k % num_nextn_predict_layers == 0` quando k>n_predict (Anemll `SpeculativeConfig`, "MTP module reuse"). 0731 tinha n_predict=1 → k=5 bootava. Cards oficiais usam k=7+greedy, o que também falha a divisibilidade (.env:494-496). |
| `DRAFT_SAMPLE_METHOD` | `probabilistic` (default) / `greedy` | .env:497-498; compose:306-312, 475-478; validate-dspark-config:79-89 | Como o draft amostra tokens | Rejection sampling | Qualquer outro valor → exit 2 antes de montar JSON. A/B 2026-09-02: greedy empatou → mantido probabilistic (ab-results-2026-09-03:86). |
| `DSPARK_ENABLE_DSPARK_BLOCK_K` | `0` / `1` | .env:433-437; compose:368-371, 460; start:265-275 | Remove a regra de divisibilidade para `method=dspark` (estágios empilhados numa passada paralela) | DSpark block drafting | Permite k=5 (bloco treinado) em vez de k=6 (uma query de "ruído" a mais); medido +2–10 % single-stream (CHANGELOG.md:100). Captura 6×6→40. |
| `dspark_block_size` (checkpoint) | 5 | fable5-1:456 | Tamanho do bloco de draft treinado | DSpark | — |
| `num_nextn_predict_layers` (checkpoint) | 3 (Vision-Exp); 1 (0731) | .env:430-432 | Número de estágios MTP/draft | MTP | Origem da regra de divisibilidade. |
| `dspark_markov_rank` | 256 | fable5-1:456 | Rank da cabeça de Markov (bigrama low-rank) | DSpark Markov head | `markov_w1` (vocab→256) e `markov_w2` (256→vocab) de 66 MB cada (fable5-1:252). |
| `DSPARK_ENABLE_REPLICATE_MARKOV` | `0` / `1` | .env:403-407; compose:343, 453 | Replica cabeça de Markov (remove 12 coletivos serializados por passo) | TP collectives | A/B: aceitação inalterada, sem ganho c=1, c=6 agregado 122 vs 148 → off (ab-results-2026-09-02:25). |
| `DSPARK_ENABLE_DSPARK_SWA_PREFIX` | `0` / `1` | .env:289-295; compose:376-379, 462 | Limita hits de prefix cache para que o target recompute a última janela do draft (128) | Prefix caching × draft SWA | Sem isso, prompts idênticos repetidos degeneram (upstream json60 12.1 → 86.5 tok/s) (CHANGELOG.md:73). |

### 2.8 Thinking, raciocínio, API OpenAI/Responses, tool calling

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `DEFAULT_THINKING` | `low` (default); `off` / `high` / `max` | .env:499-503; compose:313, 480-486; start:300-307 | Kwargs padrão de template | Reasoning effort | `low` abre `<think>` sem diretiva extra; `max` injeta diretiva do checkpoint ("do not stop reasoning until … no error remains undiscovered") → ~50.000 chars (~12,5k tokens) de raciocínio num prompt moderado (README.md:364-374). |
| mapeamento de `reasoning_effort` | patch inline: `max`/`xhigh`→`max`; `high`→`high`; qualquer outro→`low` (stock mapeava o resto para `high`) | compose:444 | Corrige wrapper pré-0731 do tokenizer (`vllm/tokenizers/deepseek_v4.py`) | Reasoning effort | Necessário para semântica 0731 de `low` (docs/DEEPSEEK_V4_FLASH_0731.md:24). |
| `thinking_token_budget` (campo de requisição) | só aceito com `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1`; `0` desliga raciocínio; `N>0` limita | README.md:391-437 | Limita só a parte de raciocínio e força um `</think>` | Thinking budget | Stock V2 retorna HTTP 400. Sem o campo, nenhum default é injetado. |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` / `1` | .env:379-384; compose:396-400, 444 | Instala budget em kernels Triton na GPU | GPU-resident counters | Default off porque tráfego sem o campo pode ter "decode cliff" (#66). Com o flag, o smoke usa `thinking_token_budget:1` (start:2093-2098). |
| `max_tokens` (requisição) | cliente | README.md:355-374 | Limita think + resposta + markup de ferramenta | Generation length | Com `max`, 256/512/800 costuma dar `content: null`/`finish_reason: length`. |
| `DSPARK_SUPPRESS_STOPS_IN_REASONING` (alias `VLLM_SUPPRESS_STOPS_IN_REASONING`) | `1` / `0` | .env:307-311; compose:405-408 | Stops do cliente dormentes até `</think>` | Stop strings | Sem isso, `stop:["Question:"]` matava a resposta dentro do think (content null) (CHANGELOG.md:327-329). |
| `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX` | `0` / `1` | compose:316, 453 | Não aplica o patch | — | — |
| `VLLM_ENABLE_RESPONSES_API_STORE` | `0` / `1` | .env:297-305; compose:137-139, 450 | Store nativo da Responses API (`store=true` + `previous_response_id`) | Responses API stateful | Store nativo é ilimitado; com `1` aplica backport limitado (LRU). Estado se perde em qualquer restart. |
| `DSPARK_RESPONSES_STORE_MAX_ENTRIES` | `256` (inteiro positivo; verificador aceita 2..16 em modo teste) | .env:305; compose:140; start:423-427 | Cap de bundles terminais (response/message/event) | LRU | Não é limite em bytes; trabalho ativo pode exceder temporariamente (ENVS:67). |
| `DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT` | `0` / `1` | .env:325-337; compose:322-324, 451 | Aceita item de histórico assistant sem `type` com 1 parte `output_text` | Responses input validation | Replays de histórico completo de alguns clientes davam 400. |
| `DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT` | `0` / `1` | .env:339-346; compose:325-326, 452 | Converte `agent_message` (Codex) com uma parte `input_text` em mensagem assistant | Responses compat | Descarta `id/author/recipient`. |
| `DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX` | `0` / `1` | .env:313-323; compose:317-321, 454 | Adiciona header de geração `<｜Assistant｜><think>` quando `messages` termina em assistant (ou `latest_reminder` após assistant) | Chat rendering | Stock renderiza EOS e o modelo gera de um estado morto (turnos vazios / DSML alucinado) (#52/#120). |
| `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN` | `0` / `1` | .env:476-493; compose:392-395, 455 | Move a diretiva de esforço do início (após BOS) para o fim da região system | Prefix cache × prompt layout | Diretiva tem 93 tokens (`max`/`xhigh`), 80 (`high`/DEFAULT), 0 (`low/off/medium`) → hashes de bloco divergem no bloco 0 e buckets não compartilham cache; com patch 0 → 18/18 blocos compartilhados num prompt de 4646 tokens (CHANGELOG.md:77). |
| `DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED` | `0` / `1` | .env:361-374; compose:355-360, 457 | Verifica pós-geração contratos de `tool_choice` named/required (non-streaming) | Tool-call contract | Violações → regenera até RETRIES e depois HTTP 500 (em vez de 200 errado). |
| `DSPARK_ISSUE191_TOOLCALL_RETRIES` | `2` (0–5) | .env:368; ENVS:106 | Regenerações | — | — |
| `DSPARK_ISSUE191_TOOLCALL_MODE` | `failclosed` / `log` | .env:369 | Modo | — | `log` só registra (medição do gate de 145 casos). |
| `DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK` | `1` / `0` | .env:370-374 | Última tentativa com thinking off (`<think>`→`</think>`, `reasoning_ended=True`) | Grammar-constrained tool call | Corrige falha residual: raciocínio 300–500 tokens estourando `max_tokens=512` (CHANGELOG.md:99). |
| `DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX` | `0` / `1` | .env:347-360; compose:352-354, 456 | Backports vLLM #52805 (terminação) + #53046 (validação de draft pós-raciocínio) no XGrammar | Structured output + spec decode | Tokens especulativos após término dessincronizavam o matcher; exige vLLM `0.25.2.dev0+g752a3a504.d20260714` e xgrammar `0.2.3`. |
| `DSPARK_ENABLE_DSML_RECOVERY` | `0` / `1` | .env:448-460; compose:380-383, 463 | Recupera `<invoke>` DSML com wrapper `tool_calls` ausente/corrompido (ex. `toolcalls`) | Tool parsing | Valida contra ferramentas declaradas; commit só em `</invoke>`; DSML correto fica byte-idêntico. Gate: paridade vs baseline 42,3 % C1. |
| `SERVED_MODEL_NAME` | ver 2.1 | .env:155 | — | — | — |
| `pi` (`pi-models.dspark.example.json`) | `thinkingLevelMap` low/high/max; `input: ["text","image"]`; `contextWindow 1048576`; `maxTokens 32768`; `supportsThinkingTokenBudget: false`; `chatTemplateKwargs` com `thinking.enabled` e `thinking.effort` (`omitWhenOff`) | pi-models.dspark.example.json:1-52 | Config de cliente | Client contract | Capability `true` com flag de boot `0` → todo request falha com "thinking_token_budget is not yet supported by the V2 model runner". pi mapeia níveis minimal 1024 / low 2048 / medium 8192 / high 16384 (README.md:475-480). Não usar `reasoning_effort` top-level genérico (GLM-NEW-REPORT §3.4). |
| Regras de imagem | `image_url` (JPEG/PNG/GIF/WebP) só em mensagens `user`; `image`/`image_url`/token cru em `system/assistant/tool/function` → 400 | README.md:227-243 | Restrição oficial de Chat Completions | Multimodal roles | Um 400 de tool com imagem fica no histórico e envenena a sessão (#178). |

### 2.9 Autenticação

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `VLLM_API_KEY` | vazio / uma chave | .env:161-202; compose:159-161 | Chave única nativa do vLLM | API auth | Mutuamente exclusiva com `DSPARK_API_KEYS` (entrypoint e scripts saem 2). |
| `DSPARK_API_KEYS` | vazio / `"k1 k2 …"` (aspas obrigatórias; espaço/tab; rejeita CR/LF/VT/FF, `\`, token iniciando com `-`) | .env:179-186; compose:162-172, 419; start:126-143, 321-358 | Várias chaves revogáveis | API auth | Deve estar só no `.env` (valor ambiente divergente → exit 2). Keys ficam visíveis via `ps`/`docker inspect`; rotação = stop/start. |
| redação de log | `patches/hotfix-vllm-redact-api-key-log.sh` obrigatório se houver chave | .env:193-199; compose:449; start:401-406 | Troca valores por `<redacted:N value(s)>` na linha `non-default args` | Secrets hygiene | Sem bypass por `DSPARK_SKIP_HOTFIX`. |
| WARN de bind aberto e permissões | aviso se `0.0.0.0` sem chave; aviso se `.env` com segredos é legível por grupo/outros | start:360-399 | Segurança operacional | — | Nunca falha o start (CHANGELOG.md:4). |

### 2.10 Runtime/kernels (variáveis de ambiente injetadas pelo compose)

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | .env:588; compose:183 | Sampler top-k/top-p do FlashInfer | Sampling kernels | — |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` (manter) | .env:589; compose:184 | Opt-out dos "breakable CUDA graphs" que o Anemll liga automaticamente para DS4 quando a variável está ausente | CUDA graphs | Regular vs breakable: C1 95.9 vs 74.55 tok/s (+28,6 %), C2 151.8 vs 134.2 (+13,1 %) (docs/DEEPSEEK_V4_FLASH_0731.md:81-94). |
| `VLLM_USE_B12X_MOE` | `1` | .env:590; compose:191 | Liga caminho B12X MoE | Fused MoE | — |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` / `_FORCE_BLOCKS_MAX_M` / `_FORCE_TILE_CONFIG` | `0` / `16` / vazio | .env:591-593; compose:192-194 | Seletores experimentais do kernel W4A16 | Kernel occupancy/tiling | Deixar em default salvo perfil indicar problema (fable5-1:301-302). |
| `B12X_W4A16_TC_DECODE` | `0` / `1` | .env:594-595; compose:195-196 | Decode empacotado small-M (M≤8) com soma top-k no epílogo FC2 | Small-M GEMM | A/B: c=1 46 vs 55 tok/s (regressão ≈−15 %) → 0 (ab-results-2026-09-02:26). |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | .env:597; compose:176 | Permite `max_model_len` > config | Contexto | Necessário para 1M. |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | `256` (stock 512); `512` testado | .env:598; compose:177 | Workspace de logits do indexer esparso | Memory/workspace | 256 = mais lançamentos por chunk longo; 512 empatou (−1,5 % 32K, −1,8 % 128K dentro do ruído) (ab-results-2026-09-03:87). |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `0` | .env:599; compose:178 | Não estima memória de CUDA graphs no profiler | Memory profiling | **[inferência]** evita subdimensionar KV por estimativa conservadora. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` (stock 300) | .env:607-609; compose:185-189 | Prazo do RPC `sample_tokens` | RPC timeout | JIT CuTeDSL/TileLang no meio do serviço passava de 300 s e matava o EngineCore (#65/#87). Não estende o watchdog NCCL de 600 s. |
| `CUTE_DSL_ARCH` | `sm_121a` | .env:601-602; compose:200-204 | Alvo de compilação CuTeDSL/b12x | Arquitetura GPU | Sem isso b12x poderia JIT-compilar um caminho W4A16 mais lento. |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | .env:603-604; compose:198-199 | Arquiteturas-alvo de build/JIT | sm_121a | — |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | .env:605; compose:205 | Ignora checagem de versão | — | Cubin/JIT-cache 0.6.13 vs python 0.6.15 (fable5-1:55). |
| `FLASHINFER_WORKSPACE_BASE` | `/cache/huggingface/flashinfer` | compose:206 | Workspace/JIT do FlashInfer persistente | JIT cache | — |
| `TILELANG_CLEANUP_TEMP_FILES` / `TILELANG_CACHE_DIR` | `1` / `/cache/huggingface/tilelang-cache` | .env:606, 610; compose:207-210 | Cache TileLang persistente | JIT cache | #65: `~/.tilelang/cache` morria no recreate. |
| `TRITON_CACHE_DIR` | `/cache/huggingface/triton-cache` | .env:611-615; compose:211-217 | Cache Triton persistente | JIT cache | #117: rank compilando trava o par além do watchdog de 600 s. |
| `B12X_CUTE_COMPILE_CACHE_DIR` | `/cache/huggingface/b12x-cute-cache` | .env:641-644; compose:249-254 | Cache de compilação CuTeDSL do MoE B12X | JIT cache | Evita re-JIT de `W4A16FusedMoeKernel` a cada restart. |
| `DG_JIT_USE_NVRTC` / `DG_JIT_NVCC_COMPILER` | `0` / `/usr/local/cuda/bin/nvcc` | .env:661-662; compose:255-256 | DeepGEMM JIT via nvcc | DeepGEMM | — |
| `VLLM_CACHE_ROOT` | `/cache/huggingface/vllm-cache` | compose:173 | Raiz de caches do vLLM (inclui `deep_gemm`) | — | Cubins DeepGEMM persistidos; `sm121_*` só compila com alias (#item8 §5). |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | .env:740; compose:257 | Alocador CUDA com segmentos expansíveis | Fragmentação | LMCache precisa desligar (conectores KV rejeitam isso) (lmcache/README.md:76-77). |
| `VLLM_PLUGINS` / `GB10_HYBRID_NVFP4_M_THRESHOLD` / `ENABLE_VLLM_GB10_PATCH` | vazio / `128` / `0` | compose:179-181 | Plugin híbrido NVFP4 | Quantization plugin | Experimental, default off. |
| PATH/CUDA_HOME/LD_LIBRARY_PATH | `/usr/local/cuda/bin`, `/usr/local/cuda`, `/usr/local/cuda/lib64` | compose:414-418 | Toolchain CUDA para JITs | — | — |

### 2.11 Chaves de hotfix (liga/desliga) — resumo

| nome | default | efeito | onde |
|---|---|---|---|
| `DSPARK_SKIP_ISSUE22_HOTFIX` | `0` | `1` pula fix de dispatch nvfp4→fp8 (não fazer) | compose:315, 445 |
| `DSPARK_SKIP_HOTFIX` | `0` | `1` pula os 6 backports v0.27 (#50312, #49486+#52492, #48407, #48957, #50298, #44993 grammar); #22 continua | compose:314, 448 |
| `DSPARK_SKIP_SPIN_WAIT_HOTFIX` | `0` | `1` mantém `busy_loop_s=1s` | compose:401, 446 |
| `DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX` | `0` | `1` rollback do backport #45224 (SHM) | compose:402-404, 447 |
| `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX` | `0` | `1` não aplica supressão de stops | compose:316, 453 |
| `DSPARK_ENABLE_*` (ISSUE31_GPU, ASSISTANT_FINAL, ISSUE138, CODEX, ISSUE141, SP_INDEXER, DEEPGEMM_SM121_ALIAS, ADAPTIVE_CHUNK, REPLICATE_MARKOV, ISSUE136, ISSUE191, DSPARK_BLOCK_K, ROPE_SWA_FIX, DSPARK_SWA_PREFIX, DSML_RECOVERY, MXFP4_INDEXER_CACHE, C128A_PREFILL_CACHE, ISSUE144_EFFORT_ALIGN) | `0` | exatamente `1` aplica, fail-closed; qualquer outro valor = stock | compose:396-400 etc. |
| `DSPARK_ISSUE43_SCHED_DIAG` | `0` | `1` = uma linha compacta por passo do scheduler | .env:271-277; compose:141-144 |
| `DSPARK_SP_INDEXER_MIN_KEYS` | `8192` (chaves comprimidas = 32K tokens a razão 4); `0` desliga mesmo aplicado | .env:413; compose:345; ENVS:115 |
| `DSPARK_ENABLE_ROPE_SWA_FIX` | `0` | `1`: camadas sparse-SWA (compress_ratio≤1: camadas 0–1 + 3 camadas do drafter) usam RoPE puro (factor 1.0, θ=10000) em vez de YaRN (port vllm#54815) | .env:438-447 |
| `DSPARK_ENABLE_MXFP4_INDEXER_CACHE` | `0` | exige `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1` (start aborta) | start:579-582 |
| `DSPARK_ENABLE_C128A_PREFILL_CACHE` | `0` | reusa conversão de índices C128A entre camadas no mesmo forward | .env:472-475 |
| `DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK` | `0` | divide chamadas de decode sparse-MLA >64 linhas em fatias de ≤64 | .env:386-394 |
| `DSPARK_ENABLE_ADAPTIVE_CHUNK` | `0` | chunk=1024 só com decodes ativos, senão o orçamento todo | .env:408-412 |

Regra operacional comum: trocar qualquer flag de patch exige **recriar** os dois containers (`./stop` + `./start`), porque `docker compose restart` preserva bytes já patcheados na camada gravável (.env:302-303, 332-336, 356-359, 392-393).

### 2.12 Ablação (direção de recusa em runtime)

| nome | valor(es) | onde | o que faz | conceito | por que importa |
|---|---|---|---|---|---|
| `ABLATE` | interno; `1` só via `ABLITERATED=1` | start:188-211; compose:129, 429 | Liga hook; entrypoint exporta `DSV4_ABLATE_FILE=/cache/huggingface/dspark-ablation/direction_r1.pt` | — | `ABLATE=1` sozinho → exit 2. |
| `DSV4_ABLATE_LAMBDA` | `3.5` (finito, ≥0) | .env:141; start:223-235 | Força λ da projeção | Directional ablation | λ ≥ 4 arrisca degeneração de CoT longo (ENVS:120). Entra no carimbo de compatibilidade do cache AOT. |
| `DSV4_ABLATE_LAYERS` | `10-42` (faixa ordenada dentro de 0–42) | .env:142; start:213-222 | Camadas do decoder onde aplicar | — | Exclui o draft DSpark. |
| `DSPARK_ABLATE_SOURCE_FILE` | `$HF_CACHE/dspark-ablation/direction_r1.pt` | .env:143-144; start:207 | Arquivo-fonte da direção | — | SHA-256 esperado `6e4d8a8f3aa9e21795faab2c5b14d29b019acdf2ddbfbd8238430458a5837fe0` (start:192); copiado a cada nó com verificação. |
| `DSPARK_ABLATE_DIRECTION_REPO` / `_FILE` / `_SHA256` | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-Anchored-Tensors` / `ablit/refusal_direction_r1.pt` / sha acima | prepare:331-335 | Origem do download gated | — | Fallback local `files/direction_r1.pt` (pickle torch com tensor float `broad`, 4096-dim unitário). |

### 2.13 TP=3 (três Sparks)

| nome | valor(es) | onde | o que faz | por que importa |
|---|---|---|---|---|
| `WORKER2_HOST` | ex. `10.0.0.3` (obrigatório em TP3) | .env:15; start:646 | 3º rank | SSH sem senha e imagem já puxada (~19 GB). |
| `WORKER2_VLLM_HOST_IP`, `WORKER2_DIR`, `WORKER2_HF_CACHE` | padrões = `WORKER2_HOST` / `WORKER_*` | .env:16-18 | IP/diretórios do 3º nó | — |
| `WORKER2_NCCL_IB_HCA`, `WORKER2_NCCL_SOCKET_IFNAME`, `WORKER2_TP_SOCKET_IFNAME`, `WORKER2_GLOO_SOCKET_IFNAME` | ex. `rocep1s0f1` / `enp1s0f1np1` | .env:19-22; docs/TP3.md:30-34 | Porta do spark3 voltada ao head | Não copiar `WORKER_NCCL_*`. |
| `WORKER2_NCCL_IB_GID_INDEX` / `WORKER2_NCCL_IB_GID_MATCH_IP` | vazio | start:642-644 | Pins manuais | — |
| `TP3_MAX_NUM_SEQS` / `--max-num-seqs` (start-tp3) | `16` | .env:24; start-tp3.sh:45-53; start:166-173 | Slots só no lane TP3 | CLI vence; captura 112. |
| `DSPARK_TP3` / `TP_SIZE` / `NNODES` | exportados `1`/`3`/`3` por `start-tp3.sh` | start-tp3.sh:66-69 | Seleciona caminho 3 nós | 2 nós sempre força TP=2/NNODES=2. |
| `TP3_PATCH_DIR` | `./patches/tp3` | compose:114 | `apply_tp3_patch.py` montado em `/opt/dsv4-tp3` | Só roda se `TP_SIZE=3` (compose:471-474). |

### 2.14 Stage-C apenas (no-op no Anemll 0.1.1)

| nome | default no override | intenção (ENVS:141-154) |
|---|---|---|
| `VLLM_USE_B12X_WO_PROJECTION` | `1` | Caminho B12X da projeção WO |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | `0.0` | Limiar de confiança do draft |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | `off` (overlay default `auto`) | Scheduler de confiança (escolhe comprimento de draft por confiança/curva de hardware) |
| `VLLM_DSPARK_LOCAL_ARGMAX` | `1` | Argmax local (vocab-parallel) no draft |
| `VLLM_DSPARK_REPLICATE_MARKOV_W1` | `1` | Replica `markov_w1` |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | `0` | Argmax Markov fundido |
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | `1` | Máscara de contexto rejeitado na GPU (caminho ragged do Keys) — exigido para 200K/16 correto |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | `0` | Quant/dequant de referência do KV (checagem numérica) |
| `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP` | `1` | Early stop do scheduler hardware-aware |
| `DSPARK_SLOT_CLAMP` | `1` | Clamp de slot do draft-KV em contexto longo |
| `B12X_W4A16_TC_DECODE` | `0` | (também no default) |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | `0` | Experimento de MLA comprimida |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE(_EXACT)` | `0` | Adiar captura de graph do target |
| `VLLM_TRITON_MLA_SPARSE` / `VLLM_SKIP_INIT_MEMORY_CHECK` | `1` / `1` | Não registrados em nenhum lane nesta forma — evitar no Anemll (ENVS:171-172) |

Fonte: docker-compose.stage-c.override.yml:14-37; .env:742-764. Não usar `VLLM_USE_B12X_FP8_GEMM=1` na imagem Stage-C (assert do DeepGEMM) (RESULTS-2026-08-14.md:157).

### 2.15 Warmup de boot e diagnósticos do launcher

| nome | valor(es) | onde | o que faz | por que importa |
|---|---|---|---|---|
| `DSPARK_BOOT_SHAPE_WARMUP` | `1` / `0` | .env:645-655; start:2110-2140 | Roda `scripts/boot-shape-warmup.sh` após smoke | Materializa buckets de shapes Triton antes do tráfego real (#117). Falha = WARN, nunca falha o boot. |
| `DSPARK_WARMUP_REQ_TIMEOUT` | `240` s | .env:656-660; boot-shape-warmup.sh:79 | `curl --max-time` por requisição | 1º boot paga compilações reais; o `.env` diz pior caso 35 × timeout a N=6, mas o script atual agenda 47 requisições a N=6 (ver 3.3). |
| `DSPARK_WARMUP_BEARER` / `DSPARK_WARMUP_MAX_CONCURRENCY` / `DSPARK_WARMUP_TRITON_CACHE_DIR` | passados pelo launcher (1ª chave; `MAX_NUM_SEQS`; caminho host do cache Triton) | start:2116-2136 | Autenticação/escopo/pós-condição do warmup | Clusters com chave não dão 401 no sweep. |
| `AB_MEM_GATE_KB` | `3000000` | scripts/ab-boot.sh:9, 34-41 | Gate de MemAvailable no A/B | Falha fechado se não conseguir ler. |

### 2.16 LMCache (experimental, fora do Compose)

| nome | valor(es) | onde | o que faz |
|---|---|---|---|
| `DSPARK_ENABLE_LMCACHE` | exatamente `1` para ativar | lmcache/patch-compose-lmcache.py:93-123 | Adiciona `--kv-transfer-config {"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.server_urls":"tcp://…:6667,…"}}`, `export PYTHONHASHSEED=0`, `unset PYTORCH_CUDA_ALLOC_CONF` |
| `LMCACHE_DISK_DIR` / `LMCACHE_L1_GB` / `LMCACHE_PORT` / `LMCACHE_OOM_SCORE_ADJ` / `LMCACHE_FORCE_REPLACE` | `$HOME/lmcache-disk` / `12` / `6667` / `0` / `0` | lmcache/run-lmcache-server.sh:10-17, 78 | L2 em disco, L1 em CPU, porta ZMQ, viés do OOM killer, override de guarda |
| servidor | `lmcache server --host <fabric-ip> --port 6667 --chunk-size 256 --l1-size-gb 12 --l1-use-lazy --eviction-policy LRU --l2-adapter {"type":"fs",…}`; `--restart no` | run-lmcache-server.sh:87-96 | Um por nó; morto deve ficar visível |

### 2.17 Parâmetros de scripts auxiliares

| script | parâmetros | onde |
|---|---|---|
| `prepare-dspark-model-cache.sh` | `--official` (grava `ABLITERATED=0`), `--abliterated` (grava 1), `--yes` (usa `.env`), interativo com TTY; `PREPARE_WORKER`, `THIS_NODE_HF_CACHE`, `IMAGE_PYTHON=/usr/bin/python3` | prepare:7-56, 65-79 |
| `stop-deepseek-v4-flash-dspark.sh` | `--nfs` (também para `dspark-nfs` e remove volume `dspark-hf`) | stop:10-26 |
| `start-tp3.sh` | `--max-num-seqs N`, `--host`, `--port` | start-tp3.sh:12-59 |
| `smoke-…sh` | `CONCURRENCY=6`, `CHAT_URL`; requisições `temperature:0` "Reply with OK and the number i." | smoke:6-100 |
| `status-…sh` / `logs-…sh` | `TAIL=160`, `API_URL`; saem 1 se algum probe falha | status:1-142; logs:1-56 |
| `validate-dspark-config.sh` | renderiza e filtra o comando do compose; avisa se revisão pinada não está no cache | validate-dspark-config.sh:96-167 |
| `build-dspark-vllm-runtime.sh` | `DSPARK_BASE_IMAGE` (`vllm-dspark-runtime:mia-raf-pr1`), `DSPARK_VLLM_IMAGE` (`vllm-dspark-runtime:dspark-nvfp4-stage-c`), `WORKER_BUILD=1`, `--tag-selftest` | build:7-108 |
| `scripts/benchmark-0731.py` | `--prompt-lengths 256,2048,8192,32768,131072`, `--concurrency 1,2,4,6`, `--thinking-token-budget`, `--max-tokens 4096`, `--output` | benchmark-0731.py:96-102 |
| `scripts/bench-miaai.py` | `--prompt 256`, `--concurrency 1`, `--repeat 5`; thinking=false, min=max=128 tokens, ignore_eos, temp 0.6/top_p 0.95 | bench-miaai.py:1-40, 89-93 |
| `scripts/spec-acceptance.py` | `--trials 5 --prompt 256` | spec-acceptance.py:57-61 |
| `scripts/ruler-lite.py` | `--lengths 8192,32768,131072,262144`, `--tasks sniah,mkniah,vartrack,cwe`, `--request-timeout` (900 s cobre só ~787K tokens de prefill) | ruler-lite.py:263-276; CHANGELOG.md:268 |
| `scripts/context-garble-sweep.py` | `--lengths 2048,8192,32768,131072`, `--runs 2` | context-garble-sweep.py:91-96 |
| `scripts/run-audit.sh` | `--base-url`, `--model deepseek-v4-flash-vision-exp`, `--lengths`, `--tool-lengths`, `--garble-lengths`, `--request-timeout` | run-audit.sh:1-45 |
| `scripts/validate_tp3.sh` | `host:port`; 5 grupos de checagem (identidade, recall factual, raciocínio, agulha ~1.5k tokens, degeneração) | validate_tp3.sh:1-100 |
| `scripts/verify-responses-api-live.py` | `--base-url`, `--model`, `--output`, `--store-capacity 2..16`, `--skip-multiturn`, `--skip-disconnect` | README.md:624-669 |
| `scripts/verify-issue138-responses-history-live.py` | `--expect-legacy rejected|accepted` | README.md:640-658 |

---

## 3. Inventário de modificações

### 3.0 Como as modificações são aplicadas (mecanismo comum)

- **Onde**: todas as modificações do lane padrão são aplicadas **dentro do container, no entrypoint, antes de `exec vllm serve`** (`compose:410-489`). Os patchers são montados read-only de `patches/` (compose:47-114). Nada é aplicado com o servidor rodando (antes, o start aplicava `.sh` e reiniciava o compose no meio do load, o que derrubava o TCPStore do head sob o worker — issue #38, CHANGELOG.md:331).
- **Estilo**: *source patching* textual sobre os arquivos Python instalados do vLLM (`/usr/local/lib/python3.12/dist-packages/vllm/...`), com âncoras exatas, marcadores de idempotência (`# [issueXX-hotfix]`), muitas vezes *pin* por SHA-256 do arquivo inteiro (stock → patched), compilação antes de publicar, substituição atômica no mesmo diretório, verificação pós-escrita e rollback byte-exato em falha. Cada passo é encadeado com `|| exit 1` (fail-closed, issue #107; CHANGELOG.md:213).
- **Consequência operacional**: os bytes patcheados ficam na **camada gravável do container**; `docker compose restart` não os desfaz → mudar flag exige `./stop` + `./start` (recriação).
- **Preflight**: para os opt-in mais sensíveis o launcher roda `docker compose run --rm --no-deps --entrypoint python3 vllm-dspark /opt/<patch>.py --check` no **worker e depois no head** antes de subir qualquer rank (start:1942-2050).
- **Sincronização**: o launcher copia cada patcher ao worker (`dscp`) (start:1606-1817); em TP3 sincroniza `patches/` inteiro via tar (start:1860-1889).

### 3.1 Cadeia do entrypoint, na ordem de execução

| # | nome | arquivo(s) | padrão | problema que resolve | o que muda (mecanismo) | conceito(s) | efeito medido |
|---|---|---|---|---|---|---|---|
| 1 | Normalização de NCCL vazio | compose:420-427 | sempre | Compose sempre define a chave; variável definida-vazia mascara `/etc/nccl.conf` (carregado com `overwrite=0`) e `NCCL_IB_GID_INDEX=""` vira índice 0 | `unset` das variáveis NCCL passthrough vazias | Semântica de env | Evita pin acidental de GID 0 (fe80) |
| 2 | Diretório do flight recorder | compose:428 | sempre | Dumps perdidos se o dir não existe | `mkdir -p` não fatal | Forensics NCCL | — |
| 3 | Gate de ablação | compose:429 | sempre | — | `ABLATE=1` → exporta `DSV4_ABLATE_FILE`; outro valor ≠0/1 → exit 2 | — | — |
| 4 | Instalação do encoder do checkpoint | compose:430-444 | sempre (WARN se não achar) | Imagem anterior ao checkpoint: formato de mensagens/tools/raciocínio do DSV4 vive em `encoding/encoding_dsv4.py`, não em Jinja | Copia para `vllm/tokenizers/deepseek_v4_encoding.py` (descobre por `DSPARK_ENCODING_FILE` → snapshot da revisão → qualquer snapshot → 0731 → `/models/...`) | Chat encoding | Obrigatório para semântica de `reasoning_content`, esforço e tool args (docs/DEEPSEEK_V4_FLASH_0731.md:24) |
| 5 | Mapeamento de `reasoning_effort` | compose:444 (python inline sobre `vllm/tokenizers/deepseek_v4.py`) | sempre (com encoder) | Wrapper pré-0731 mapeava `low` (e qualquer outro) para `high` | `else: "high"` → `elif == "high": "high" else: "low"`; `assert` do novo texto | Reasoning effort | `low` passa a ser `low` |
| 6 | Issue #21 — args dict no DSML | patches/hotfix-encoding-dsv4-issue21.py | sempre | `encode_arguments_to_dsml` sempre fazia `json.loads(arguments)`; quando já era dict, a exceção embrulhava como `{"arguments": <dict>}` e envenenava histórico multi-turn de ferramentas | Aceita dict e string JSON | Tool history encoding | Bug upstream no `encoding_dsv4.py` do HF (CHANGELOG.md:404) |
| 7 | Issue #31/#66 — `thinking_token_budget` na GPU | patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py | **opt-in** `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1` | Runner V2 do Anemll rejeita o campo (HTTP 400); versão anterior (withdrawn) escaneava tokens na CPU a cada passo → penhasco de ~4,7× no decode | Dois kernels Triton pequenos residentes na GPU forçam exatamente um `</think>` na fronteira e observam tokens MTP aceitos; sem D2H por passo | GPU-resident counters; spec decode | 65K: sem budget 37.2 tok/s vs com budget 59.3; curto 67.2 tok/s; c=1–6 per-stream ≥ baseline (−0,0 % a +19,4 %) (CHANGELOG.md:299-305). CI proíbe `.cpu(`, `.tolist(` etc. no patch (ci-validate:229-243) |
| 8 | Issue #55 — truncamento de tool call | patches/hotfix-dsv4-issue55-tool-truncation.py | sempre | `max_tokens` no meio de uma tool call → stock devolvia `finish_reason:"tool_calls"` com `arguments` JSON inválido → próximo turno HTTP 400 | Em `FinishReason.LENGTH`, reporta `"length"` e descarta tool_calls cujos `arguments` não parseiam; também corrige serialização de `FinishReason` IntEnum (`'2'`→`'abort'`) | finish_reason; streaming append-only | Todas as células truncadas (300/800/1500/600+high) passam a `length`; natural stop mantém `tool_calls` (CHANGELOG.md:291-297) |
| 9 | Issue #22 — dispatch nvfp4_ds_mla | patches/hotfix-nvfp4-ds-mla-issue22.sh (+ referência `fix-nvfp4-ds-mla-long-context.patch`) | sempre (skip `DSPARK_SKIP_ISSUE22_HOTFIX=1`) | `flashmla_sparse.py` usava `use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"` → `nvfp4_ds_mla` caía no kernel lento `_forward_bf16_kv` | `in ("fp8_ds_mla","nvfp4_ds_mla")` → kernel rápido `_forward_fp8_kv` (bytes idênticos) | KV layout vs kernel dispatch | ~16× no decode a 600K+ (1.0 → 17.3 tok/s) (CHANGELOG.md:416-428) |
| 10 | Issue #79 — spin-wait do SHM | patches/hotfix-gb10-spin-wait.sh | sempre (skip `DSPARK_SKIP_SPIN_WAIT_HOTFIX=1`) | `SpinCondition.busy_loop_s=1` em `shm_broadcast.py`: gaps de IPC de decode < 1 s → 3–4 P-cores Grace girando no máximo, roubando orçamento de energia do SoC | `busy_loop_s: 1 → 0.002` (2 ms de graça, depois dorme via socket de notificação) | IPC busy-wait; power budget SoC | Só política de espera; saída do modelo inalterada |
| 11 | Issue #117 — backport vLLM #45224 (SHM ring) | patches/hotfix-vllm-issue117-shm-ring-buffer.py | sempre (skip `DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX=1`), `--check` preflight e `--status` | Notificações de dispatch perdidas no meio do serviço; slots de leitura não liberados quando o consumidor lança exceção | Espera de leitor ocioso limitada a 5000 ms (recheck) e liberação do slot em `finally` (merge upstream `10c75477…`) | Shared-memory ring buffer | Recupera o caminho observado de missed-notify; não limpa `/dev/shm/psm_*` órfãos (CHANGELOG.md:112) |
| 12 | Trem v0.27 (6 scripts) | patches/hotfix-dsv4-mtp-buffer-50312.sh, -skip-topk-49486.sh, -dense-prefill-indexer-48407.sh, -skip-empty-c128-48957.sh, -flashmla-workspace-50298.sh, -grammar-advance.sh | sempre (skip `DSPARK_SKIP_HOTFIX=1`) | Backports de performance/corretude do vLLM 0.27 para a base 0.25.2 | ver 3.2 | — | ver 3.2 |
| 13 | Redação de chave no log | patches/hotfix-vllm-redact-api-key-log.sh (+ `--status`) | se houver chave (obrigatório, sem bypass) | `log_non_default_args()` imprime `--api-key` em claro | Troca por `'api_key': ['<redacted:N value(s)>']` nos dois entrypoints | Secrets hygiene | Fecha só o canal de log (argv/env continuam visíveis) |
| 14 | Store limitado da Responses API | patches/hotfix-dsv4-responses-store.py | opt-in `VLLM_ENABLE_RESPONSES_API_STORE=1` | Stores nativos (response/message/event) ilimitados | Backport "exact-source" com cap LRU de bundles terminais (`DSPARK_RESPONSES_STORE_MAX_ENTRIES`), protege continuações e produtores ativos, remove mensagens órfãs em erro | Responses API state; LRU | — (issue #62) |
| 15 | Issue #138 — histórico Responses | patches/hotfix-vllm-issue138-responses-history.py | opt-in | Replays de histórico com item assistant sem `type` → 400 | Aceita só o formato exato (1 parte `output_text` string), insere `type=message`, reusa coerção stock | Pydantic validation | Verificador modo-estrito |
| 16 | Codex `agent_message` | patches/hotfix-vllm-codex-agent-message.py | opt-in | Item privado do Codex rejeitado | Converte o shape exato em mensagem assistant, descarta metadados | Compat | — |
| 17 | Vision-Exp nativo | patches/hotfix-dsv4-vision-exp.py + patches/vision_exp/{apply,image_processor,processor,vision}.py | **sempre** | `DeepseekV4ForCausalLM` do Anemll 0.1.1 é só texto | (1) hook de import no fim de `nvidia/model.py` constrói ViT (32 camadas) + Aligner, mapeia pesos `vision.*`/`aligner.*`/`image_*`/`bias_vl`, registra processador multimodal; (2) remapeia `ffn.gate.bias_vl` → `e_score_correction_bias_vl` no draft DSpark; linhas de imagem roteiam com `bias_vl` e sem tabela hash (issue #175); (3) relaxa rejeição do encoder a placeholders já substituídos, restaura regra "imagens só em `user`"; hash de imagem salgado com `num_tokens` (issue #172); `get_supported_mm_limits` → `None` | Multimodal LLM; MoE routing bias | Suporte a `image_url`; máx 384 tokens de imagem; sem vídeo (CHANGELOG.md:130-150) |
| 18 | Issue #141 — chunk de decode sparse-MLA | patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py | opt-in | Travamentos estocásticos de TP=2 dentro do decode sparse-MLA (`shm_broadcast acquire_read → RuntimeError("cancelled")`, ~706–899 s) | Chamadas >64 linhas divididas em views ordenadas de ≤64 (limite fixo do decode standalone do FlashInfer); tensores acoplados fatiados juntos | Workaround de kernel | chunk-64 sobrevive 3.145.728 tokens vs chunk-65 morre na rodada 1 (um par) — **mitigação, não causa raiz** (.env:560-574) |
| 19 | SP indexer (item 6) | patches/hotfix-dsv4-sp-indexer-prefill.py | opt-in | Lightning indexer replicado: cada rank pontua todas as 64 cabeças contra todas as chaves comprimidas (O(queries×keys), duplicado) | Rank r pontua só a fatia `[lo_r,hi_r)` alinhada a página, top-k local, troca `(score, id)` e faz merge top-k estável exato com o kernel DCP (`stable_topk_from_gathered_candidates_cutedsl`); só para chunks ≥ `DSPARK_SP_INDEXER_MIN_KEYS` | Sequence parallelism; top-k distribuído exato | 128K TTFT −4 a −6 % (76.0/74.4 s vs 79.0/81.2 s); 8K/32K inalterados; RULER-lite 8/8 (ab-results-2026-09-02:28, 53) — muito abaixo da estimativa 1,5–1,8× |
| 20 | DeepGEMM sm121 alias | patches/hotfix-deepgemm-sm121-mqa-header-alias.sh | opt-in | DeepGEMM vendorizado gera nomes `sm121_*` no GB10 mas só traz headers `sm120_*`; em *cache miss* falha "Failed to open .../sm121_fp8_mqa_logits.cuh" | Escreve 4 headers alias (`#include "sm120_X.cuh"` + `#define sm121_X sm120_X`) | JIT de kernels por compute capability | Produção sobrevivia de cubins persistidos (item8:208-223) |
| 21 | Issue #109 — saída vazia do encoder | patches/hotfix-vllm-empty-encoder-output.py | sempre | Passos só de encoder (EC producer) emitiam token 0 fantasma e entravam em loop | Uma linha vazia distinta por requisição; scheduler para requisição só-encoder quando consumiu o prompt (upstream `7ca49fb`) | Multimodal scheduling | — |
| 22 | Issue #27 — cap de prefills parciais | patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py | **sempre** | vLLM 0.25.2 define `max_num_partial_prefills` mas o loop de admissão v1 nunca lê → vários prefills admitidos consomem o orçamento e decodes recebem `num_new_tokens==0` (pulados, não preemptados) | Quebra a admissão quando prefills em voo (contados exatamente de `self.running`, reparo r3 #211) atingem `DSPARK_MAX_INFLIGHT_PREFILLS` | Scheduler admission; starvation | x8 8K/16K/32K pior decode ~15 tok/s (era 2.07/0.47/0.36); com chunk 1024, spread x8 de ~46× para ~1,05× (CHANGELOG.md:361-367; .env:242-244) |
| 23 | Adaptive chunk | patches/hotfix-dsv4-adaptive-prefill-chunk.py | opt-in | Com c=1 frio, o cap 1024 ainda obriga 8 chunks para 8K (custo fixo por chunk) | `threshold = 1024 se há decode ativo, senão max_num_batched_tokens` | Scheduler | Rejeitado: MemAvailable do head 2.36/2.28 GB vs 3.8–5.2 GB (workspace maior em memória unificada) (ab-results-2026-09-02:27, 51) |
| 24 | Replicate Markov head | patches/hotfix-dsv4-replicate-markov-head.py | opt-in | Cabeça de Markov TP-sharded (`VocabParallelEmbedding` + `ParallelLMHead`) → loop de draft de 6 passos faz 12 coletivos serializados por passo | `nn.Embedding` + `ReplicatedLinear` (66 MB cada) | TP collectives | Correto, sem ganho; c=6 122 vs 148 → off; −0,44 GiB KV no head |
| 25 | Issue #43 — piso de decode + diagnóstico | patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py | **sempre** | Mesmo com #27, spread de taxa de decode por requisição 0.107–0.238 | Ao iterar RUNNING, se o item é chunk de prefill, reserva 1 passo de decode para cada decode-ativo atrás e limita o chunk; registra skips zero-token e tokens por requisição (`DSPARK_ISSUE43_SCHED_DIAG`) | Fairness por passo | No-op nos defaults atuais (1024/8192/cap 1), mas impede mis-tuning ressuscitar #27; pior ITL 2790 → 67 ms creditado ao #27 |
| 26 | Issue #26/#36 — coordenador híbrido v2 | patches/hotfix-dsv4-issue26-hybrid-swa-min.py | **sempre** | 4 grupos KV (1×MLA + 3×SlidingWindowMLA); `find_longest_cache_hit` usa o mínimo entre grupos → SWA zerava o hit; v1 (ignorar SWA) causou KV SWA nulo e "salada" DSML/CJK em prefixos de 21K | v2: SWA volta a poder encolher o hit; hits longos vêm do `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` | Hybrid KV cache coordinator | x8 warm 8/8 (0.9986/0.9973/0.9996); 32K warm ~9 s vs ~421 s; CI proíbe v1 (ci-validate:259-278) |
| 27 | Issue #133 — especialização Triton | patches/hotfix-dsv4-issue133-triton-specialization.py | **sempre** | `_compute_global_topk_indices_and_lens_kernel`: ponteiros fatiados em offsets dependentes de tráfego → 3 estados de alinhamento × 2 block sizes = 6 entradas de cache e JIT no meio do serviço | `do_not_specialize_on_alignment` para os dois ponteiros escalares; strides/top-k/block como constexpr | Triton JIT cache keys | 2 variantes determinísticas |
| 28 | Runtime ablation | patches/hotfix-dsv4-runtime-ablation.py | sempre roda; só age com `ABLATE=1` (senão verifica stock e mantém carimbo AOT) | — | Injeta em `DeepseekV4DecoderLayer` buffer `_refusal_dir` e, nas camadas `DSV4_ABLATE_LAYERS`, aplica pós-atenção `y ← y − λ·(y·v)·v` em fp32 com `v` normalizado | Directional ablation | Direção capturada no 0731 FP8; transferência para Vision-Exp é **experimental** (README.md:178); abliteração drena aceitação (0.585 vs 0.718, scripts/EVAL.md:50) |
| 29 | Supressão de stops no raciocínio | patches/hotfix-dsv4-suppress-stops-in-reasoning.py | sempre (skip / guarda via `DSPARK_SUPPRESS_STOPS_IN_REASONING`) | Stops do cliente disparavam dentro de `<think>` → `content:null` | No detokenizer do Anemll, se o último token do prompt é o marcador de início do raciocínio, stops só avaliam após `</think>` (chunks especulativos com `</think>` avaliam depois do marcador) | Stop matching | `17 + 25 = 42` com 99 tokens em vez de null (CHANGELOG.md:327-329) |
| 30 | Assistant-final continuation (#52/#120) | patches/hotfix-dsv4-assistant-final-continuation.py | opt-in | `messages` terminando em assistant (ou `latest_reminder` após assistant) renderiza EOS sem header de geração | Anexa `<｜Assistant｜><think>` | Chat rendering | A/B causal: 183 tokens coerentes vs 400 tokens de markup `<|DSML|tool_calls>`; ON preserva 98 token ids stock (CHANGELOG.md:276) |
| 31 | Issue #144 — alinhamento de diretiva de esforço | patches/hotfix-dsv4-issue144-effort-align.py | opt-in | Diretiva após BOS quebra hashes encadeados de prefix cache entre níveis de esforço | Move diretiva para o fim da região system | Prefix-cache hash chain | 0 → 18/18 blocos compartilhados; junção BPE custa 1 token (CHANGELOG.md:77) |
| 32 | Issue #136/#210 — XGrammar | patches/hotfix-vllm-issue136-xgrammar-termination.py | opt-in | Tokens especulativos após o término passavam ao matcher já terminado → estado de gramática dessincronizado, timeout de `sample_tokens` | Hunks de vLLM #52805 + #53046 em transação de 2 arquivos (`backend_xgrammar.py`, `structured_output/__init__.py`) | Structured output FSM + spec decode | — |
| 33 | Issue #191 — tool_choice fail-closed | patches/hotfix-vllm-issue191-toolcall-failclosed.py | opt-in | `tool_choice` named/required podia devolver 200 errado (bitmask de structural tag falha aberto sob spec+async, família vLLM #49694/#54437) | Checagem pós-geração (1 call com `parallel_tool_calls=false`, nome, args JSON-objeto, schema `strict`) + regeneração limitada + fallback thinking-off | API contract | Camada de contrato, não conserta a gramática |
| 34 | `--async-scheduling` configurável | compose:458 | `DSPARK_ASYNC_SCHEDULING=1` | A/B #191 | Arg condicional | Async scheduling | ~4 % single-stream sem async |
| 35 | MXFP4 indexer args | compose:459 | opt-in | — | `--attention-config {"use_fp4_indexer_cache":true}` | — | — |
| 36 | DSpark block-k | patches/hotfix-vllm-dspark-block-k.py | opt-in | Regra `num_speculative_tokens % n_predict` ("MTP module reuse") força k=6 no Vision-Exp | Adiciona `self.method != "dspark"` à regra em `config/speculative.py` (stock `3f1abd1c…` → `7fffe035…`) | DSpark stacked stages | +2–10 % single-stream, aceitação por posição inalterada (CHANGELOG.md:100) |
| 37 | RoPE SWA fix (vllm#54815) | patches/hotfix-vllm-rope-swa-fix.py | opt-in | Builder promove `rope_scaling yarn factor 16` a `deepseek_yarn` em todas as camadas; na referência DSV4, YaRN só nas camadas com compressor | Camadas sparse-SWA (compress_ratio≤1) recebem identidade (factor 1.0, θ=10000); compressoras byte-idênticas | RoPE/YaRN por tipo de camada | Gate de qualidade 128K+ pendente |
| 38 | DSpark SWA prefix | patches/hotfix-vllm-dspark-swa-prefix.py | opt-in | Em hit de prefix cache o target pula o prefixo e o cache de janela deslizante (128) do draft nunca é populado → prompts idênticos repetidos degeneram | `max_cache_hit_length ≤ num_tokens − 1 − sliding_window` em `kv_cache_manager.py`, janela passada pelo scheduler | Prefix cache × draft state | upstream 12.1 → 86.5 tok/s em json60 repetido |
| 39 | DSML recovery (vllm#52645) | patches/hotfix-vllm-dsml-recovery.py | opt-in | Wrapper `tool_calls` ausente/corrompido → invoke vaza como conteúdo | Buffer provisório de `<invoke name=…>`, valida contra tools declaradas, commit em `</invoke>`, rollback verbatim; 6 arquivos pinados | Robust tool parsing | 34 testes CPU; gate de paridade pendente |
| 40 | MXFP4 indexer cache (item8 §3) | patches/hotfix-vllm-mxfp4-indexer-cache.py | opt-in (exige alias sm121) | vLLM tem cache K MXFP4 do indexer mas gateia para Blackwell datacenter | Relaxa gate para `is_device_capability_family(120)` em `v1/attention/backends/mla/indexer.py` (stock `02505c6c…` → `bfb0376d…`) | Quantização de K do indexer | Metade dos bytes lidos por chave; alocação ainda 132 B/linha (CHANGELOG.md:69) |
| 41 | C128A prefill cache | patches/hotfix-vllm-c128a-prefill-cache.py | opt-in | Conversão local→global de índices C128A refeita por camada | Primeira camada C128A consumidora converte, as demais do mesmo forward reusam (`c128a_prefill_global_topk` no metadata) | Metadata caching | "No end-to-end speedup is implied" (CHANGELOG.md:61) |
| 42 | GB10 hybrid NVFP4 plugin | vllm_patch_gb10/ (pip install -e, `VLLM_PLUGINS=gb10_hybrid_nvfp4`) | opt-in `ENABLE_VLLM_GB10_PATCH=1` | Experimento de dispatch W4A16/W4A4 por M | Config `modelopt_gb10_hybrid` troca métodos lineares NVFP4 | Quantization plugin | "Tested here only for syntax packaging" (README do plugin) |
| 43 | TP=3 pad | patches/tp3/apply_tp3_patch.py (+ `patches/dsv4_tp_pad.py`) | só se `TP_SIZE=3` | 8 grupos de saída e 64 cabeças não dividem por 3; stock recusa (`64 % 3 != 0`) ou silenciosamente mantém `8//3==2` grupos | Pad de grupos 8→9 mantendo 8 cabeças/grupo (heads 64→72), zero/−inf nas faixas de pad; no-op em TP≤2 | TP divisibility | Log `DSv4 TP pad: heads 64 -> 72 for TP=3` em todo rank; `validate_tp3.sh` 7/7 |
| 44 | Montagem do speculative-config | compose:475-479 | sempre | Valor cru do `.env` perderia aspas/poderia injetar JSON | Só `probabilistic|greedy` passam; JSON montado no container | Input validation | — |
| 45 | `DEFAULT_THINKING` → kwargs | compose:480-486 | sempre | — | case off/low/high/max | — | — |
| 46 | `LIMIT_MM_PER_PROMPT` → JSON | compose:489 | sempre | Argparse do Anemll é `json.loads`; `image=8` puro dava ArgumentTypeError | Converte `image=N` em `{"image":N}` | — | — |

### 3.2 Trem de backports vLLM 0.27 (aplicado quando `DSPARK_SKIP_HOTFIX` ≠ 1)

Fontes: docs/vllm-027-new-patches.md; CHANGELOG.md:371-383, 210-211; start:1402-1405. Todos idempotentes, com `--status`, `--before`/`--after`, transacionais (valida todos os alvos/âncoras antes de escrever, rollback de todos os publicados em falha — CHANGELOG.md:227).

| script | PR upstream | o que faz | estado/efeito |
|---|---|---|---|
| `hotfix-dsv4-mtp-buffer-50312.sh` | #50312 (+2 None-guards que o upstream não tem) | Aloca `_mtp_hidden_buffer` só se um especulador precisa (`use_eagle()`/`uses_draft_model()`); pula `copy_` quando `None`; guarda os dois sites de `get_mtp_target_hidden_states()` em `v1/worker/gpu/model_runner.py` | Economiza ~256 MiB/rank aqui (448 MiB upstream) — ROI de memória, sem ganho de TTFT |
| `hotfix-dsv4-skip-topk-49486.sh` | #49486 + guarda de captura #52492 | Kernel Triton `_fill_short_context_topk_indices`; se `max_seq_len // compress_ratio <= topk_tokens` (≤2048 tokens = 512 × 4), pula wq_b→RoPE→quant→logits→top-k e escreve todos os candidatos; **não** durante `torch.cuda.is_current_stream_capturing()` | ~3,4 % TTFT E2E upstream; guarda evita que graph capturado com atalho selecione 0..topk-1 sem pontuação em prefixos longos |
| `hotfix-dsv4-dense-prefill-indexer-48407.sh` | #48407 (Stage A) | Maquinaria para pular o indexer em prefills densos curtos | **Dormente de propósito**: fork não tem rota dense-MHA; `dense_mha_metadata_layer_name=""` → gate nunca dispara (ativar descartaria seleção válida) |
| `hotfix-dsv4-skip-empty-c128-48957.sh` | #48957 | Pula o lançamento do compressor C128 quando nenhuma requisição cruza fronteira de 128 tokens no passo | ~2× no kernel; desabilitado sob `CUDAGraphMode.FULL` |
| `hotfix-dsv4-flashmla-workspace-50298.sh` | #50298 | Reuso de workspace para índices combinados top-k+SWA no prefill FlashMLA (`out=` em `combine_topk_swa_indices`, `get_simultaneous`) | 1,88× no kernel combinado upstream |
| `hotfix-dsv4-grammar-advance.sh` | #44993 | `should_advance` usa `new_token_ids` do passo como janela delta e registra `reasoning_end_token_index` para todos os tipos de constraint | Corrige `{"city":"Paris{"city":"Paris"}` com `response_format json_schema` + thinking + spec decode |
| ~~`hotfix-dsv4-adaptive-topk-50004.sh`~~ | #50004 | Largura adaptativa de top-k C128A | **Removido 2026-08-22**: upstream reverteu (#51318): builder escrevia linhas empacotadas no stride do batch vivo enquanto graphs FULL mantinham o stride de captura → linhas ≥1 liam slots velhos (corrupção intermitente de tokens sob MTP + prefix cache + CUDA graphs) |

Não backportados (docs/vllm-027-new-patches.md:63-72): #49236 (scratch pool; precisa rebuild C++; e upstream reverteu em #52836), #46789 (sequence parallelism), #48993 (compact MXFP4 indexer KV), #48047 (remoção de padding de q-heads; exige FlashInfer ≥0.6.14).

### 3.3 Modificações do lado do launcher/host (não mudam o vLLM, mudam o comportamento do sistema)

| nome | arquivo | problema | mecanismo | efeito |
|---|---|---|---|---|
| Resolver de GID RoCEv2 | start:792-1170 | Índice GID difere por nó/HCA e muda após reboot; seletor `NCCL_IB_HCA` com sintaxe de lista fazia o launcher procurar `/sys/class/infiniband/=devA,devB` | Espelha `parseStringList` e o universo de candidatos de `ncclIbInit` (portas ACTIVE, link Ethernet/IB, `MAX_IB_DEVS=32`), valida cada membro contra IPv4 próprio (GID `…ffff:aabb:ccdd`), falha fechado; não pina | 72 checagens em `scripts/test-nccl-ib-hca-gid-resolve.sh` |
| Interfaces de bootstrap TP3 | start:1539-1572 | Pares CX /24 não servem para Gloo/TCPStore entre 3 nós | Move Gloo/NCCL-socket/TP para LAN `enP7s7`, `NCCL_IB_HCA` com as duas portas, `MERGE_NICS=0`, subnet-aware /24 | TP3 funcional |
| Staging da direção de ablação | start:699-759 | — | Cópia com verificação SHA-256 e `mv` atômico em head/worker(/worker2) | — |
| Sincronização atômica do env | start:1589-1605, 1839-1858 | Transferência falha podia expor/truncar env com segredos | Arquivo temp `0600` + `mv -f` | — |
| Wrappers ssh/scp | start:16-22 | ~100 ssh sem timeout → "start lento" ambíguo | `BatchMode=yes`, `ConnectTimeout=10` | — |
| Contrato exit 3 | start:1486-1537 | Após reboot o `unless-stopped` restaura ranks; start saía 1 → retry storm em systemd | Head existente → 3; worker stale (sentinel 42) → 3; ssh falho → 1 | — |
| Export NFS | files/nfs-share.sh, files/nfs-server/* | 2ª cópia de 157 GiB no worker | Container Alpine privilegiado com nfsd kernel (host network), volume Docker NFS 4.2 no worker, overlays JIT locais | — |
| Boot-shape warmup | scripts/boot-shape-warmup.sh | Shapes nunca vistos no smoke JIT-compilam sob tráfego; rank compilando trava o par > 600 s (watchdog) | Escada de completions com contagem exata de tokens s={1,6,20,45,100,200} verificada via `/tokenize` → buckets `next_pow2(s+6)` = {8,…,256} de `_prepare_dflash_inputs_kernel`; bursts C=1/2/4/6 (bounded e serve-default); prefill médio (2600) e longo (9500, cruza 8192); arm thinking-off; arms de sampling (top_k, top_p, ambos; temp 0.8) + pós-condição no cache Triton (`_topk_topp_kernel` TTIR) | Contagem pelo código atual: 6 (escada) + 8 base + 4 (C≥2) + 9 (C≥3, sampler C=3) + 8 (C≥4) + 12 (C≥6) = **47 a N=6**, 35 a N=4, 18 a N=2 (boot-shape-warmup.sh:279-292), igual ao CHANGELOG.md:184; o comentário do `.env:651-658` e ENVS:79 ainda dizem "35 a N=6 (23 a N=4)" **[inconsistência: comentário anterior aos arms de sampler]** |
| Arredondamento da captura | compose:506 | c=6 eager | múltiplo de 8 | +12 % c=6 |
| Validação numérica | dspark-numeric-knobs.sh | overflow/octal | ver 2.3 | — |

### 3.4 Imagens: Anemll vs Stage-C (o que foi mudado no runtime em si)

- **Anemll `dspark-vllm-gx10:0.1.1`** (padrão): pré-construída externamente (github.com/Anemll/dspark-vllm-gx10); a lógica DSpark/concorrência do Keys pode estar embutida sem expor os kill-switches Stage-C (ENVS:23-26). Nenhum arquivo de `recipe/overlay/` é carregado nela (CHANGELOG.md:191).
- **Stage-C** (`build-dspark-vllm-runtime.sh`):
  1. `recipe/Dockerfile.dspark-runtime-overlay`: `FROM ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` + `COPY` de 17 arquivos do overlay (envs.py, config/speculative.py, fused_moe/b12x_moe.py, warmup/kernel_warmup.py, models/registry.py, v1/outputs.py, v1/core/sched/scheduler.py, models/deepseek_v4/{__init__,nvidia/model,common/ops/cache_utils,nvidia/sm120,nvidia/dspark,nvidia/dspark_kernels}.py, attention/backends/{registry,mla/b12x_mla_sparse}.py, v1/spec_decode/{dspark,dspark_proposer}.py, v1/worker/gpu_model_runner.py) + `py_compile`; teste `map_dspark_stacked_param_name('model.layers.0.ffn.shared_experts.w1.weight') == (…gate_up_proj.weight, 0)` (build:67-68).
  2. **Stage A** (`recipe/nvfp4/Dockerfile.stage-a`): "encanamento" do dtype `nvfp4_ds_mla` — adiciona a `CacheDType`, mapeia para `torch.uint8`, `KVQuantMode.NVFP4`, página `storage_block_size * 416`.
  3. **Stage B** (`Dockerfile.stage-b`): *probe* com layout de 416 B/token herdado do caminho sparse-MLA NVFP4 do GLM-5.2; converte `nvfp4`→`nvfp4_ds_mla`, relaxa assert "fp8 only".
  4. **Stage C** (`Dockerfile.stage-c`): volta ao **envelope de 584 B** para que o agrupamento híbrido MLA/SWA (mesmo tamanho de página) funcione — "padded NVFP4". É isso que "Stage C padded NVFP4 path" significa (README.md:600-601).
- **`recipe/official-main/Dockerfile.python-patch`**: variante sobre `vllm-dspark-runtime:official-main-dspark-00eb7ce` que copia arquivos python (cache.py, vllm.py, kv_cache_interface.py, attention.py, sparse_mla.py, flashmla_sparse.py, oracle/mxfp4.py, experts/b12x_mxfp4_moe.py) e verifica `nvfp4_ds_mla` → (3,256,584) e backend MoE `B12xExperts`.
- **`Dockerfile.gb10-dsv4-dspark`**: variante experimental sobre `ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest` (vLLM 0.26) que usa um layout **352 B/token** (256 B E2M1 NoPE + 32 B escalas E4M3 bloco-16 + 64 B RoPE FP8, casando com `concat_and_cache_nvfp4_ds_mla.cu kStride=352`). **Não é o caminho padrão** e contradiz o envelope 584 B do Anemll **[inconsistência: dois layouts "nvfp4_ds_mla" diferentes no mesmo repo, em imagens diferentes]**.
- **`patches/keys-concurrency.patch`** (Keys, commit `7e4d94bb…`): slots de KV principal do DSpark estáveis por requisição (não por posição de linha do batch) e `query_start_loc` ragged para batches mistos prefill/decode — pré-requisito de `max_num_seqs>1` correto no lane Stage-C; com `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` (CREDITS.md:36-46; docs/SETUP.md:64-78).

### 3.5 Coisas removidas/retiradas (importante para não re-introduzir)

| item | por quê | onde |
|---|---|---|
| Hook CPU de `thinking_token_budget` (#31/#34) com defaults `DEFAULT_THINKING_TOKEN_BUDGET=32768`, `DEFAULT_MAX_TOKENS=131072` | Scan O(n) Python × linhas MTP a cada passo → penhasco de ~4,7× (#39) | CHANGELOG.md:321; CI proíbe (ci-validate:229-257) |
| #26 v1 (ignorar SWA no mínimo) | KV SWA nulo → salada DSML/CJK em prefixos compartilhados | CHANGELOG.md:333 |
| #50004 adaptive top-k | Corrupção sob FULL graphs | CHANGELOG.md:210 |
| Sidecar Qwen3-VL / MCP `ds4f-vision` / `ENABLE_VL_SIDECAR` / `GPU_MEMORY_UTILIZATION_VISION` | Substituído por visão nativa | CHANGELOG.md:155 |
| Lane MoonViT | Removido | CHANGELOG.md:397 |
| Layout 416 B "true" | Abandonado em favor de 584 B padded | README.md:600-601 |

---

## 4. Números e benchmarks

> Aviso do próprio repo: os números de 14 Ago são "o melhor visto, não a norma"; em 2026-09-02 o mesmo lane mediu 15–25 % menos decode sem nenhuma mudança de `.env` que explique (README.md:337-341; AUDIT.md:26-30). Ruído de c=1 neste lane ≈ ±10 tok/s dependendo do prompt sorteado (ab-results-2026-09-02:43).

### 4.1 Capacidade de KV cache (linhas de boot)

| data / lane | util | Available KV | tokens no pool | concorrência a 1M | fonte |
|---|---|---|---|---|---|
| Vision-Exp TP=2 (README atual) | 0.835 | 17.04 GiB | 2,331,430 | 2.22× | README.md:117-121, 488 |
| 0731 texto TP=2 | 0.835 | 18.08 GiB | 2,493,464 | 2.38× | RESULTS-2026-08-14.md:206-212 |
| 0731 a 0.85 (lane anterior) | 0.85 | — | ~2.8 M | — | GLM-NEW-REPORT.md:125-126 |
| Vision-Exp 2026-09-02 A/B (baseline) | 0.83 | head 16.55 / worker 14.73 GiB | 2,188,268 (head) | 2.09× | ab-results-2026-09-02:22, 33 |
| Vision-Exp 2026-09-02 boot final | 0.83 | 17.46 / 15.68 GiB | — | — | ab-results-2026-09-02:62 |
| Variação boot-a-boot | 0.83 | head 15.96–17.12 GiB (±0.6 GiB) | — | — | ab-results-2026-09-03:91 |
| TP=3 (16 slots) | 0.83 | 34.97 / 33.79 / 35.05 GiB | ≈5.0 M (5,021,871) | 4.79× | ab-results-2026-09-03:129; fable5-1:56 |
| Preview Anemll (DSpark preview, MTP-3) | 0.85 | — | ~2.83 M | 2.70× | RESULTS:133-136 |
| Stage-C Keys C12 (2026-07-02) | — | — | ~3.23 M | ~3.2× | RESULTS:146-149 |
| 1M/6 microbench (2026-06-29) | — | — | ~1.90 M | ~1.81× | RESULTS:116-121 |
| 1M NVFP4 probes antigos | — | — | ~2.04 M | ~1.95× | RESULTS:200 |
| Era sidecar de visão | 0.80 | main 13.37 GiB + VL 1.54 GiB | 1.37 M + 84 k | — | CHANGELOG.md:408 |

Contas úteis (derivadas):
- 17.04 GiB / 2,331,430 tokens ≈ **7.8 KB/token** de "contabilidade do alocador" (4 grupos); o custo físico em contexto longo é ≈ 3.07 KB (21 camadas C4 × 584/4) + 0.09 KB (20 camadas C128 × 584/128) + 0.69 KB (indexer 21 × 132/4) por token, mais ≈3.4 MB/seq de janelas SWA (46 camadas × 128 linhas × 584 B), **replicados nos dois ranks** (item8:146-150).
- Página comum do alocador híbrido: 64 linhas × 584 B = **37,376 B** (item8:144).
- `2,331,430 / 1,048,576 = 2.22` → "Maximum concurrency 2.22×" significa só "~2 requisições de 1M completas cabem ao mesmo tempo" (README.md:499-500).
- Exemplo do README: 6×50k = 300k fácil; 6×200k = 1.2M cabe; 6×500k = 3.0M perto/acima; 6×1M = 6.0M impossível (README.md:492-497).

### 4.2 Memória

| item | valor | fonte |
|---|---|---|
| Pesos por rank TP=2 | 79.17 GiB | fable5-1:57 |
| Pesos por rank TP=3 | 60.69 GiB | fable5-1:56 |
| MemTotal do host | 121.7 GiB | fable5-1:58 |
| MemAvailable típico após boot (util 0.83) | 3.8–5.2 GB (baseline); cai para 2.3 GB com adaptive-chunk, ~1.0 GB com chunk 2048 | ab-results-2026-09-02:27, 51; -09-03:84 |
| Swap durante boot | head 2.0 → 5.8 GB; worker 0.6 → 2.8 GB | ab-results-2026-09-02:35 |
| `NVRM … NV_ERR_NO_MEMORY` | 12× no boot nos dois nós (autotune/capture) | fable5-1:59, 136-139 |
| Buffer MTP economizado (#50312) | ~256 MiB/rank | vllm-027-new-patches.md:11 |
| Replicar Markov | −0.44 GiB KV no head | ab-results-2026-09-02:25 |
| Cada +0.01 de util | ≈ +1.2 GiB KV ≈ +165 K tokens (TP=2) | fable5-1:165-166 |
| Cabeça de Markov | 2 × 66 MB (129,280 × 256 × 2 B) | fable5-1:252 |
| Imagem Docker | ≈19 GB | docs/TP3.md:16 |

### 4.3 Velocidade esperada (lane padrão Anemll 1M/6) — README

| carga | esperado | fonte |
|---|---|---|
| 1 chat, qualquer prompt até 128K | ~62–83 decode tok/s após 1º token | README.md:327 |
| 6 chats curtos (1M permitido) | ~160–190 tok/s agregado (~30–37 por stream) | README.md:328 |
| 6 prompts frios 32K–128K | prefills em chunks, 1 em voo; ~8 tok/s de piso de decode enquanto prefills rodam; 128K×6 TTFT = minutos | README.md:329 |
| 2026-08-14 | 256×c=6 = 162 agregado; 128K×c=1 = 75 tok/s, TTFT 80 s | README.md:334-335 |
| 2026-09-02 (sp-indexer on, capture 48, inflight 2) | 256×c=1 = 56 tok/s (mediana de 8, 51–67); 256×c=6 = 139 agregado; 128K×c=1 TTFT 78.5 s | README.md:337-339 |

### 4.4 Varredura publicada 0731 (2026-08-01, PR #14) — `benchmark-0731.py`, completions naturais ~512 tokens, prefixos únicos, MTP-5 probabilístico, TP=2

| Prompt | C | TTFT (s) | Prefill tok/s | Decode tok/s | Agregado tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.63 | 447 | 75.4 | 69.1 |
| 256 | 2 | 0.81 | 357 | 58.3 | 104.9 |
| 256 | 4 | 1.26 | 222 | 46.8 | 164.5 |
| 256 | 6 | 1.42 | 197 | 36.9 | **191.2** |
| 2,048 | 1 | 0.81 | 2,563 | 68.8 | 62.0 |
| 2,048 | 2 | 1.11 | 1,911 | 57.0 | 97.6 |
| 2,048 | 4 | 1.38 | 1,505 | 44.0 | 154.7 |
| 2,048 | 6 | 6.06 | 342 | 34.7 | 143.7 |
| 8,192 | 1 | 4.80 | 1,713 | 73.9 | 43.7 |
| 8,192 | 2 | 7.51 | 1,176 | 49.8 | 56.2 |
| 8,192 | 4 | 14.50 | 578 | 37.4 | 72.3 |
| 8,192 | 6 | 18.38 | 454 | 23.6 | 73.1 |
| 32,768 | 1 | 22.96 | 1,428 | 64.0 | 16.6 |
| 32,768 | 2 | 26.82 | 1,287 | 41.5 | 24.8 |
| 32,768 | 4 | 44.85 | 756 | 17.4 | 26.7 |
| 32,768 | 6 | 60.75 | 550 | 10.8 | 27.9 |
| 131,072 | 1 | 78.75 | 1,665 | 65.2 | 5.9 |
| 131,072 | 2 | 111.17 | 1,306 | 30.9 | 6.6 |

128K×c=4 não terminou na janela de 180 s. **Aceitação 900K**: 899,994 tokens de prompt, TTFT 1,028.85 s, ~874.8 prefill tok/s, sentinela correta (docs/DEEPSEEK_V4_FLASH_0731.md:50-77; results/deepseek-v4-flash-0731-2x-dgx-spark.json).

### 4.5 Matriz ao vivo 0731 (2026-08-14) — `bench-miaai.py`: prefixos frios únicos, `thinking=false`, `max_tokens=min_tokens=128`, `ignore_eos`; célula = mediana decode por stream

| Prompt | c=1 | c=2 | c=4 | c=6 | c=6 agregado | TTFT mediano c=1 / c=6 (s) |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 82.6 | 56.9 | 54.3 | 35.9 | 162 | 0.27 / 1.02 |
| 2,048 | 64.6 | 50.1 | 29.6 | 20.8 | 78 | 1.20 / 4.41 |
| 8,192 | 65.9 | 44.2 | 16.0 | 10.2 | 29 | 4.62 / 16.62 |
| 32,768 | 62.0 | 39.7 | 8.7 | 8.2 | 8 | 18.47 / 65.78 |
| 131,072 | 74.9 | 36.0 | 8.4 | 8.6 | 2 | 80.43 / 281.83 |

(RESULTS-2026-08-14.md:27-47; valores de TTFT extraídos de results/bench-screenshot-matrix-20260814.json.) Observação: c=4/6 × 32K+ ficam no piso de ~8 tok/s (um token de decode por passo de prefill serializado).

### 4.6 Capturas de imagem

- `docs/concurrency_bench.png` (Anemll/vLLM 0.25, NVFP4 DS-MLA, TP=2, `max_num_seqs=6`, mediana de 5 trials, `max_tokens=256`, temp 0): agregado/decode médio/TTFT — C1 57.2/76.6/1.13 s; C2 83.6/57.8/1.54 s; C3 83.3/37.9/2.19 s; C4 122.6/41.1/1.92 s; C5 124.4/33.6/2.36 s; C6 158.7/35.6/2.19 s ("C3 ≈ C2 e C5 ≈ C4: passos de TTFT/prefill compensam streams extras").
- `docs/benchmarks.png` (0731, 2048 tokens de completion): x1 TTFT 168 ms, 82.4 agg/stream; x2 295 ms, 98.0 / 53.0; x3 338 ms, 134.6 / 45.8; x4 5.36 s, 120.4 / 33.6 (também RESULTS:107-112).

### 4.7 CUDA graphs regulares vs "breakable" (0731, sonda de completion natural a 1M, temp 0.2, top-p 0.95, MTP-5, seqs 6, batched 8192)

| modo | breakable | regular | Δ |
|---|---:|---:|---:|
| C1 decode mediana quente | 74.55 | 95.9 | +28.6 % |
| C2 agregado mediana | 134.2 | 151.8 | +13.1 % |
| C4 / C6 agregado | — | 263.7 / 340.5 | — |

Prefill 14K dentro do ruído (C1 1,770–1,781 → 1,857 tok/s) (docs/DEEPSEEK_V4_FLASH_0731.md:81-94).

### 4.8 A/Bs de 2026-09-02 (Vision-Exp, TP=2, `scripts/ab-measure.sh`, 8 trials c=1 agrupados)

Sequência 1 (docs/CLAUDE/ab-results-2026-09-02.md:20-29):

| boot | variável | c=1 (mediana, faixa) | c=2 | c=6 agregado | TTFT 8K/32K/128K | aceitação natural temp0 geral/pos0 | numbered-word | veredito |
|---|---|---|---|---|---|---|---|---|
| A baseline (dual-PF) | — | 52.4 (49.8–56.9); pooled ≈55 | 43.4 | 129.7/148.0/147.3 | 4.49–4.53 / 17.8–18.3 / 79.0 s | 25.2 % / 0.68 | 47.0 % (pos0 0.90) | referência |
| A′ single-PF | uma PF por nó | 65.7 (51.5–76.3); extra 54.8 → ≈55 | 43.1 | 137.7/140.0/148.0 | 4.70 / 18.6 / 81.2 s | 27.2 % / 0.71 | 51.1 % | empate → dual-PF mantido |
| B markov | replicate Markov | ≈54 | 40.5 | 142.4/120.5/122.3 | 4.87 / 17.8–18.4 | 26.0 % / 0.69 | 49.5 % | OFF |
| C tcdecode | `B12X_W4A16_TC_DECODE=1` | ≈46 (7/8 abaixo do mínimo) | 41.5 | 132.5/134.8/149.3 | 4.65–4.84 / 18.0 | 25.8 % / 0.69 | 52.0 % | OFF (−15 %) |
| D adaptive-chunk | — | não medido | — | — | — | — | — | OFF (MemAvailable 2.36/2.28 GB) |
| E sp-indexer | MIN_KEYS 8192 | ≈55 | 41.9 | 131.1/143.4/147.6 | 4.45–4.48 / 17.8–18.0 / **76.0 (quente 74.4)** | 25.4 % / 0.68 | 51.2 % | **ON** |
| F mtp3 | `MTP_NUM_TOKENS=3` | — | — | — | — | — | — | não bootável (regra k≥5, k%3==0) |

Sequência 2 (docs/CLAUDE/ab-results-2026-09-03.md:81-89):

| boot | knob | c=1 | c=2 | c=6 agregado | TTFT | fairness c=4×8K (spread TTFT / decode / agg) | veredito |
|---|---|---|---|---|---|---|---|
| A base2 | referência | 56.6 (50.8–67.0) | 42.7 | 139 | 4.47–4.62 / 18.18–18.29 / 78.5 s | 14.6 s / 29.9 / 22.6 — TTFT por stream 4.7/9.4/14.3/19.3 s | ref |
| B chunk2048 | `LONG_PREFILL_TOKEN_THRESHOLD=2048` | — | — | — | — | — | OFF (MemAvailable 0.99 GB, swap +1 GB) |
| D greedy | `DRAFT_SAMPLE_METHOD=greedy` | 56.5 | 44.5 | 140 | — | — | empate → OFF |
| E idxlogits512 | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512` | 54.2 | 42.4 | 142 | 32K −1.5 %, 128K 77.1 s | — | empate → OFF |
| F inflight2 | `DSPARK_MAX_INFLIGHT_PREFILLS=2` | 53.9 | 41.6 | 139 | 32K 17.70–18.26 | **9.1 s** / 24.7 / 25.4 (+12 %) — por stream 7.4/9.0/15.7/16.5 s | ON (depois revertido a 1 por #211) |
| G capture42 | captura sem arredondar (42→40, FULL 5/5) | 55.7 | 42.4 | **122** | inalterado | — | captura 48 fica |

### 4.9 Fairness e admissão de prefills (issue #27 / #154 / #211 / #217)

| data | configuração | resultado | fonte |
|---|---|---|---|
| 2026-08-12 | #27 + chunk 1024 | x8 8K/16K/32K pior decode ~15 tok/s (era 2.07/0.47/0.36), 0 preempções, MTP 96–99 % | CHANGELOG.md:367 |
| 2026-08-19 | inflight 2 (1ª vez) | 32K×c4 decode por stream 8.2 → 24.6 tok/s; 256×c6 ~175 inalterado; soak 12 min 32K×c2 21/21 | CHANGELOG.md:260 |
| 2026-08-28 (#154) | 2 vs 1, 4 lanes × 8K | spreads 3.72×/5.14×/4.94× a 2 vs 2.04×/1.68×/1.69× a 1; Gate26 1.71× a 1 | CHANGELOG.md:161 |
| 2026-09-04 (#211 r3) | cap 1 | gate26 1.60–1.76×, 0 preempções | CHANGELOG.md:89 |
| 2026-09-05 (#217) | cap 2 pós-r3, chunk 1024 | 4×8K gate26 1.50–1.57× (vs 1.53× a 1), TTFT spread 2.1–2.4× vs 4.1×; 4×32K spread 3.0–3.3× vs 2.1×; pré-r3 A/B: 1ºs tokens 7.4/9.0/15.7/16.5 s vs 4.7/9.4/14.3/19.3 s | README.md:278, 329; ENVS:93 |
| .env | chunk 1024 + #27 | spread x8 ~46× → ~1.05× | .env:242-244 |

### 4.10 TP=3 vs TP=2 (2026-09-02, prompts de prosa, 512 tokens de saída; TP=3 a 16 slots, TP=2 a 6)

| carga | TP=2 | TP=3 |
|---|---|---|
| decode 1 stream | 38.6 tok/s | 41.9 (+9 %) |
| decode 2 streams | 63.9 agg | 72.0 (+13 %) |
| decode 4 streams | 86.0 agg | 89.7 (+4 %) |
| 8 / 12 / 16 streams | n/a | 142.8 / 164.3 / 198.0 agg |
| TTFT 8k/16k/32k/64k | 4.4 / 8.7 / 18.0 / 36.0 s | 5.0 / 9.4 / 18.6 / 38.2 s (+4–13 %) |
| TTFT 128k/256k | 74.9 / 164.8 s | 91.5 / 202.2 s (+22–23 %) |
| KV por rank | ≈16.8 / 15 GiB | ≈35 / 34 / 35 GiB (5.0 M tokens, 4.8× a 1M) |

(docs/TP3.md:116-135.) 1º boot TP3: engine init 193 s, ~8 min até healthy; FULL graphs 15/15; c=1 62 tok/s (50.7 frio, 62.1, 86.3); c=16 95 → 177 → 210 aquecendo (ab-results-2026-09-03:129). Sonda viva TP=3 (fable5-1:66-76): c=1 64.3 tok/s; c=6 33.7/stream, 156 agregado; aceitação 2004/4140 = 48.4 %, ≈3.9 tokens/passo, por posição 0.92/0.74/0.60/0.41/0.28/0.21 — "TP=3 não aumentou o decode".

### 4.11 Admissão alta (dados de performance, não recomendação) — 2× GB10 TP=2 @ `70a7cc4`, prompts curtos, thinking=low, 500 tokens

`MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=12288`: c=1 47–54 | c=8 154 | c=16 213 | c=32 325 agregado (vs baseline N=4 onde c=8 enfileirava, ~108). Pool quase invariante: ~2.1× 1M a N=32 vs ~2.6× a N=4 (.env:522-541). Estabilidade #141: N=32/c=32 (224 linhas de verificação a k=6) morreu no burst 4 num par e no burst 2 noutro; tudo acima de ~91–96 linhas morreu em algum lugar; nada acima de 64 linhas é estabelecido limpo; default 48 linhas (.env:542-569).

### 4.12 Stage-C / históricos

| lane | resultado | fonte |
|---|---|---|
| 200K/16 Keys (Stage-C + ragged), estático | C1 57.6 (acc 0.635); C8 252.6 agg / 31.6 stream (0.635); **C16 315.1** agg / 19.7 (0.609) | RESULTS:175-183 |
| 200K/16 escalonado | C16 205.0 agg | RESULTS:183 |
| Keys C12 Stage-C (2026-07-02), seqs 6 | C1 52.8 (0.585); C6 127.8 (0.615); C12 230.1 (0.602) | RESULTS:146-155 |
| 1M/6 microbench (2026-06-29, Keys mask) | 6/6, ~182 agg (~30/stream); solo ~67 | RESULTS:116-121 |
| Preview Anemll (MTP-3, util 0.85), decode-only 512 tokens, temp 0 | C1 66.6; C4 123.8 agg / 32.8; C6 153.7 / 26.8 | RESULTS:133-144 |
| Checkpoint agente limpo 2026-06-30 (MTP-5, thinking off) | C6 104 agg | RESULTS:159-162 |
| Diagnóstico 60 tok/s | single stream, 262144, seqs 1, KV fp8: 63.97 tok/s | RESULTS:187-190 |
| Sondas 1M NVFP4 single stream | p256/g64 54.5 (acc 0.667); p256/g256 65.4 (0.718); p512/g256 54.4 (0.550) | RESULTS:192-198 |
| Réplicas A/B históricas (4 nós, preview) | Rafael (seqs 1) ~52–54; TonyD2Wild ~52 | docs/SETUP.md:36-54 |

### 4.13 Aceitação da especulação

| medição | valor | fonte |
|---|---|---|
| AUDIT "esperado" (08-16) | ~68–75 % geral; pos0 ~0.93 → pos4 ~0.33 | AUDIT.md:22-23; scripts/EVAL.md |
| Prompt patológico (palavra repetida) | pos0 0.72 → 0.11 (vs 0.93 → 0.33 em prompts bons), parece regressão falsa de ~45 tok/s | scripts/EVAL.md:25-28 |
| Natural-prose 2026-09-02 | 24.7–27.2 % / pos0 0.66–0.71 (gap "live vs audit"; maior alavanca de decode restante) | ab-results-2026-09-02/-03 |
| Numbered-word bench | 45–53.6 % (pos0 0.90–0.96) | ab-results |
| Engine 10-s windows (TP=3) | 0.92/0.74/0.60/0.41/0.28/0.21 | fable5-1:74 |
| `results/dspark-metrics.txt` (modelo `deepseek-v4-flash-0731-vision`, ~2026-08-09/10) | 630 drafts em 105 passos (k=6); 149 aceitos (23.7 %); por posição 84/63/1/1/0/0 de 105 | results/dspark-metrics.txt |
| Abliteração | drena aceitação (0.585 vs 0.718) | scripts/EVAL.md:50 |
| Não compensa k>6 | aceitação 0.28/0.21 nas posições 5–6 | fable5-1:44-46 |
| pos3–5 em prosa | 0.13/0.07/0.03 → mtp3 "provavelmente não vale" | ab-results-2026-09-02:68 |

### 4.14 Modelo de custo do passo de decode (fable5-1 §2, TP=2, c=1, k=6)

| componente | bytes/rank/passo |
|---|---|
| Experts roteados (≤42 distintos de 256 por camada × 6.7 MB/rank × 43 camadas) | ≈12 GB |
| Pesos de atenção fp8 (alguns replicados) | ≈2.4 GB |
| Draft (3 camadas DSV4, 6 queries) | ≈0.9 GB |
| `lm_head` bf16 (target + base do draft) | ≈1.06 GB |
| Markov `w2` 6× | ≈0.2 GB |
| **Total** | **≈16.5–17 GB** → a 230–250 GB/s efetivos = **68–74 ms** |

A ~4.15 tokens aceitos/passo → 56–61 tok/s de limite por streaming; "c=1 decode no TP=2 é ~80 % streaming de pesos". Coletivos por passo ≈108 (43 all-reduce `wo_b` + 43 MoE + 6 draft + 2 embedding + 2 all-gather de logits + 12 do loop de Markov) a 30–50 µs cada ≈ 3–5 ms/passo (fable5-1:102-115). Prefill: 86 all-reduces de `[1024×4096]` bf16 por chunk ≈ 690 MB/chunk/rank (fable5-1:222-226). Indexer a 900K: ≈80 TFLOP por chunk de 1024, duplicado nas duas GPUs (fable5-1:306-313).

### 4.15 Rede

| medição | valor | fonte |
|---|---|---|
| PF do CX7 no GB10 | PCIe Gen5 ×4 ≈ 15.7 GB/s ≈ 126 Gb/s (porta de 200 Gb/s só com as duas PFs) | fable5-1:200-205 |
| RTT head→worker | 0.33 ms (via 2ª PF 0.70 ms) | fable5-1:62, 209 |
| Par 1 (2026-08-21) | 1311 tok/s (MTU 1500, 1 HCA) → 1343 (MTU 9000) → 1410 (2 HCAs) = +7.6 % | .env:705-707 |
| Par 2 | 1422 tok/s dual-HCA (sem baseline) | .env:707-708 |
| Decode com dual-HCA | sem mudança material (code 76.0 tok/s a 73.3 % aceitação; prosa 37.3 a 25.7 %; 14–15 passos/s) | .env:709-713 |
| Dual-HCA sem merge (2026-08-06) | +3 % e custou buffer de KV | .env:714-716 |
| nccl-tests busbw | +64 % (não transferível ao E2E) | .env:716-717 |
| `NCCL_GIN_ENABLE=0` | comm-init ~2 min → ~13 s | .env:686-689 |

### 4.16 Correções pontuais com números

| item | número | fonte |
|---|---|---|
| #22 dispatch nvfp4 | 600K+: 1.0 → 17.3 tok/s (~16×) | CHANGELOG.md:416 |
| #26 warm prefix | x8 ~22.8K/~44.7K/~88.4K warm 8/8 (0.9986/0.9973/0.9996); 32K warm ~9 s vs ~421 s; x1 262K 5/5 | CHANGELOG.md:355 |
| #31 GPU budget | 65K: 37.2 → 59.3 tok/s; curto 67.2; implementação anterior ~4.5 tok/s | CHANGELOG.md:303 |
| Block-k (k=5) | +2–10 % single-stream | CHANGELOG.md:100 |
| DSpark SWA prefix (upstream) | json60 repetido 12.1 → 86.5 tok/s | CHANGELOG.md:73 |
| #144 | 0 → 18/18 blocos compartilhados (prompt 4646 tokens); intra-bucket 96–98 % | CHANGELOG.md:77 |
| `max` thinking | ~50,000 chars (~12.5k tokens); ~3× latência vs `low` | README.md:368-369; .env:500-502 |
| LMCache (experimental) | ~107K tokens: re-prefill ~65 s vs reload ~1.8–1.9 s (n=2) | lmcache/README.md:3-6 |
| Cold autotune | 1ª requisição pós-restart 6–10 s TTFT | scripts/EVAL.md:35-37 |
| SSE vs usage | contar chunks subestima ~4.5× | scripts/EVAL.md:39-43 |
| Tokenizer | filler inglês ~1.56 tok/palavra (340,787 palavras → 533,110 tokens) | context-garble-sweep.py:28-33 |
| Boot frio | ~6 min até carregar (start_period 480 s) | compose:17-18 |
| Tool-eval | baseline de aceitação de tool-call 42.3 % C1 (gate do DSML recovery) | .env:458-459 |

### 4.17 Valores de auditoria esperados (AUDIT.md, 2026-08-16)

C1 decode 73–76 tok/s (Keys ablit 73.1, oficial 75.5); C6 agregado 168–180; aceitação ~68–75 %; RULER-lite 8/8 a 8k/32k; tool battery 7/7, deep-context 8/8; issue55 sempre `finish=length`; garble CLEAN até ~900k (cold prefill) (AUDIT.md:20-30).

---

## 5. Conceitos necessários (ordem de pré-requisito)

### Bloco A — Fundamentos de ML / DL

1. **Tensores e multiplicação de matrizes (GEMM).** Um modelo de linguagem é, em essência, uma sequência de multiplicações de matrizes (pesos × ativações) mais algumas funções não lineares. Aqui, quase todo o custo é GEMM: projeções de atenção (fp8 blocos 128×128), experts MoE (FP4) e `lm_head` (bf16 129,280×4,096). Saber que "M" é o número de linhas de ativação (≈ tokens no passo) explica por que decode (M pequeno, ex. 7 a c=1) e prefill (M=1024 por chunk) se comportam diferente.
2. **Rede neural, pesos, forward/backward, treino vs inferência.** Treinar ajusta pesos por gradiente; inferir só executa o forward com pesos congelados. Esta receita é 100 % inferência: nenhum peso é treinado, apenas carregado, quantizado (ou não) e executado; até a "abliteração" é uma projeção em runtime, não um fine-tune.
3. **Logits, softmax e distribuição sobre o vocabulário.** A última camada produz um vetor de logits (129,280 posições aqui) que o softmax transforma em probabilidades do próximo token. Todo o mecanismo de decodificação especulativa (aceitar/rejeitar tokens do draft) e de amostragem (temperatura, top-k/top-p) opera sobre essas distribuições.
4. **Amostragem: greedy, temperatura, top-k, top-p.** Greedy pega o argmax; temperatura achata/afina a distribuição; top-k/top-p cortam a cauda. Aparece em `--generation-config vllm` (cliente controla), nos arms de warmup com `temperature:0.8` para compilar `_topk_topp_kernel`, e no diagnóstico de DSML: a temperatura 1.0 também é aplicada aos tokens de sintaxe das tool calls (docs/DSML_SYNTAX_TEMP_ASYMMETRY.md).
5. **Tokenização (BPE) e tokens especiais.** Texto vira ids inteiros; tokens especiais marcam papéis, raciocínio (`<think>`/`</think>`), imagens (`<｜deepseek_image｜>` id 129264) e DSML. A receita verifica comprimentos via `POST /tokenize` porque estimativas por palavra erram (~1.56 tok/palavra neste tokenizer).
6. **Modelo autoregressivo e next-token prediction.** O LLM gera um token por vez, condicionado em todos os anteriores. Isso cria as duas fases de inferência (prefill do prompt, decode token a token) e a necessidade de KV cache e de especulação.

### Bloco B — Transformer / LLM padrão

7. **Transformer decoder-only (bloco = atenção + FFN, residual, RMSNorm).** Cada uma das 43 camadas aplica atenção e depois uma FFN (aqui MoE), com conexões residuais e normalização RMSNorm. A ablação é aplicada logo após a atenção em cada camada 10–42 (`hotfix-dsv4-runtime-ablation.py`).
8. **Self-attention (Q, K, V; softmax(QKᵀ/√d)V; máscara causal).** Cada token produz query/key/value; a saída é uma média ponderada dos values pelos scores QK. É a operação cujo custo cresce com o contexto e cujo estado (K,V) precisa ser guardado — daí todo o trabalho em KV cache, atenção esparsa e compressão.
9. **Multi-head attention e head_dim.** A atenção é dividida em cabeças paralelas; aqui 64 cabeças × head_dim 512. Isso importa para TP (cabeças são particionadas entre GPUs) e para o pad de TP=3 (64 não divide por 3).
10. **RoPE (Rotary Position Embedding).** Codifica posição girando pares de dimensões de Q/K por ângulos proporcionais à posição. No DSV4 só 64 das 512 dimensões são rotacionadas ("RoPE dim 64", o resto é "NoPE" 448) — por isso a linha de KV tem 448 B NoPE + 128 B RoPE bf16.
11. **Extensão de contexto com YaRN.** YaRN reescala as frequências da RoPE para que um modelo treinado em ~64K funcione em contextos maiores; aqui `rope_scaling {type: yarn, factor: 16}` (64K → 1M). O patch opcional `DSPARK_ENABLE_ROPE_SWA_FIX` corrige que o vLLM aplicava YaRN também às camadas de janela deslizante, onde a referência DSV4 usa RoPE puro.
12. **Prefill vs decode; TTFT, ITL/TPOT, throughput por stream vs agregado.** Prefill processa o prompt inteiro de uma vez (compute-bound, mede-se TTFT e prefill tok/s); decode gera um token por passo (memory-bound, mede-se tok/s por stream). Agregado = soma de todos os streams. Toda a tabela de benchmarks é organizada por essas métricas.
13. **KV cache.** Guardar K e V de tokens anteriores evita recomputá-los a cada passo de decode; o tamanho do cache (bytes/token × tokens) limita contexto e concorrência. Aqui o pool é ~2.33 M tokens em 17.04 GiB, compartilhado por todas as requisições.
14. **Memory-bound vs compute-bound (roofline) e streaming de pesos.** Em decode com batch pequeno, o tempo é dominado por ler pesos da memória, não por FLOPs. O relatório fable5-1 estima ≈16.5–17 GB lidos por rank por passo a 230–250 GB/s ≈ 68–74 ms → "c=1 é ~80 % streaming de pesos"; por isso concorrência (mais tokens por leitura de pesos) e especulação (mais tokens aceitos por passo) são as alavancas.

### Bloco C — Arquitetura específica do DeepSeek-V4-Flash

15. **Mixture of Experts (MoE): roteador top-k, experts roteados e compartilhado.** Em vez de uma FFN densa, cada token é enviado para poucos "experts" escolhidos por um roteador; aqui 256 experts roteados, top-6, mais 1 compartilhado sempre ativo, `moe_intermediate_size` 2048. Isso reduz FLOPs por token, mas os bytes lidos no decode dependem de quantos experts distintos o batch toca (≤42 por camada a c=1 com k=6).
16. **Balanceamento sem perda auxiliar (`e_score_correction_bias`) e camadas hash-routed.** O DSV4 corrige scores do roteador com um bias aprendido; nas camadas 0–2 o roteamento é por hash do id do token (`tid2eid`), sem GEMM de roteador. No Vision-Exp há um bias separado para tokens de imagem (`bias_vl`); o hotfix de visão faz linhas de imagem usarem `bias_vl` e pular a tabela hash (issue #175), senão todo token de imagem colapsava num único expert.
17. **MLA (Multi-head Latent Attention).** Em vez de guardar K/V completos por cabeça, projeta-se para um vetor latente comprimido compartilhado (low-rank; `q_lora_rank`/`o_lora_rank` 1024) e uma parte RoPE separada. É por isso que o KV por token do DSV4 é uma única "linha" de 584 B (não 64 cabeças × …), e também por isso, em TP, **todo rank guarda o KV latente inteiro** (replicado) — TP=3 não reduz o custo de atenção.
18. **Projeção de saída agrupada (`o_groups`=8, `wo_a`/`wo_b`).** A saída da atenção é projetada por grupo (BMM `einsum("bhr,hdr->bhd")` com r = cabeças/grupo × head_dim = 8×512 = 4096) e depois por `wo_b` (RowParallel → all-reduce). Esse contrato "r=4096" é o que obriga o pad de TP=3 a adicionar grupos (8→9), não cabeças dentro de um grupo.
19. **Atenção esparsa com Lightning indexer (top-k 512).** Um "indexador" leve (64 cabeças de índice × 128 dims, fp8) pontua todas as posições comprimidas e só as top-512 entram na atenção principal. O indexer é **replicado** em TP (cada GPU pontua tudo), é O(queries×keys) e domina o prefill a ≥128K; o hotfix SP indexer divide as chaves entre ranks e faz merge exato do top-k.
20. **Atenção comprimida C4A / C128A (camadas com compressor; CSA/HCA no repo).** Nas camadas 2–42, o KV é comprimido ao longo da sequência: 21 camadas guardam uma linha a cada 4 tokens (C4A) e 20 camadas uma a cada 128 (C128A). Isso é o que torna 1M de contexto viável (≈3.07 KB/token nas C4 + 0.09 KB nas C128), e é onde vivem vários backports (#48957 pula compressor C128 quando nenhum request cruza fronteira de 128; C128A prefill cache).
21. **Sliding Window Attention (SWA, janela 128).** Algumas camadas (0–1 e as 3 do drafter) só atendem aos últimos 128 tokens. Janelas deslizantes liberam blocos antigos por design — o que quebrava o prefix cache híbrido (#26) e o estado do draft em hits de prefixo (DSpark SWA prefix fix).
22. **Hyper-connections (`hc_mult`, `mhc`).** O código do draft expande o fluxo residual em `hc_mult` cópias e usa operadores `HCHeadOp` (`vllm.model_executor.layers.mhc`). Não é parametrizado pela receita, mas aparece no código de DSpark e nos knobs Stage-C (`VLLM_USE_B12X_MHC`) — entender que o residual não é um vetor único ajuda a ler o código.
23. **MTP (Multi-Token Prediction) e camadas `nextn`.** O checkpoint traz camadas extras (43–45) treinadas para prever tokens futuros; `num_nextn_predict_layers=3` no Vision-Exp (1 no 0731). Elas são o "draft" embutido que a DSpark usa, e o número 3 é a origem da regra de divisibilidade de k.
24. **Modelo multimodal: ViT + Aligner.** Imagens são cortadas em patches 14×14, codificadas por um Vision Transformer de 32 camadas (dim 1024, 16 cabeças, RoPE 2D) e um Aligner que junta blocos 3×3 (`F.unfold`) e projeta para 4096 dims via MLP GELU. O resultado substitui tokens placeholder no prompt (até 384 tokens por imagem, layout "N" com newlines e pad de alinhamento a 4).

### Bloco D — Quantização

25. **Formatos de ponto flutuante: FP32, BF16, FP8 (E4M3), FP4 (E2M1).** Menos bits = menos memória e banda, com menos precisão. Aqui: lm_head/RoPE em bf16, lineares e KV em fp8 e4m3, experts em fp4 e2m1. Cada formato tem faixa dinâmica pequena e precisa de fatores de escala.
26. **Escala por bloco (block scaling): MXFP4 (E2M1 + escala E8M0 por 32), NVFP4 (E2M1 + escala E4M3 por 16), UE8M0.** Em vez de uma escala por tensor, cada bloco de 16/32 valores tem sua própria escala (potência de 2 em E8M0/UE8M0, ou E4M3 no NVFP4). Os experts do DSV4 são `fp4_e8m0_k32` (MXFP4); o KV usa escalas UE8M0 por 64 dims.
27. **Quantização weight-only (W4A16) vs W4A4/W4A8 e dequant no kernel.** W4A16 guarda pesos em 4 bits mas faz a MMA em bf16 após desquantizar; W4A4/W4A8 usa tensor cores de baixa precisão diretamente. O backend b12x só tem caminho NVFP4 tensor-core para fonte `modelopt_nvfp4`, então roda W4A16 para este checkpoint — ok para decode (memory-bound), ineficiente para prefill (fable5-1 §3.5).
28. **Quantização de KV cache e layouts DS-MLA.** `fp8_ds_mla` = 448 B NoPE fp8 + 128 B RoPE bf16 + 8 B escalas/pad = 584 B por linha. `nvfp4_ds_mla` nesta imagem usa **os mesmos 584 B** (Stage C "padded"); o "nvfp4" só troca o dispatch de kernel (daí o bug #22). Houve tentativas de 416 B (probe GLM-5.2), 352 B (imagem vLLM 0.26 alternativa) e um desenho de 292 B (item8).
29. **Restrição de página uniforme do alocador híbrido.** O vLLM exige que todos os grupos de KV tenham o mesmo tamanho de página em bytes; hoje 64 linhas × 584 B = 37,376 B. Qualquer layout FP4 real precisa dividir essa página com contagem de linhas potência de 2 (292 × 128 ou 146 × 256) — por isso uma linha de 366 B não serve (item8 §2).
30. **Cache K do indexer em MXFP4.** O vLLM já tem K do Lightning indexer em FP4 com kernels DeepGEMM `fp8_fp4_*_mqa_logits`, mas gateado para Blackwell datacenter; o hotfix relaxa para sm_12x e liga `use_fp4_indexer_cache`. Lê metade dos bytes por chave pontuada (a alocação continua 132 B/linha).
31. **ModelOpt NVFP4 e o plugin híbrido GB10.** ModelOpt é o toolkit NVIDIA que produz checkpoints NVFP4 com metadados W4A4 e escalas de ativação. O plugin `modelopt_gb10_hybrid` despacha Marlin W4A16 para M<128 e CUTLASS W4A4 para M≥128; experimental e não aplicável ao checkpoint DSV4 padrão.

### Bloco E — Decodificação especulativa e DSpark

32. **Speculative decoding (draft → verify).** Um modelo barato propõe k tokens; o modelo alvo verifica todos em um único forward (k+1 posições) e aceita o maior prefixo compatível. Como decode é memory-bound, verificar 7 tokens custa quase o mesmo que 1, então cada token aceito é quase "grátis".
33. **Taxa de aceitação, tokens/passo e curva por posição.** A aceitação cai com a posição (ex. 0.92/0.74/0.60/0.41/0.28/0.21); tokens por passo ≈ 1 + soma das aceitações (≈3.9 medido). Métricas em `/metrics` (`vllm:spec_decode_num_*`), lidas por `spec-acceptance.py`; prompts patológicos derrubam a curva e simulam regressões.
34. **Rejection sampling: `draft_sample_method` probabilistic vs greedy.** No modo probabilístico o draft amostra e o verificador aceita com probabilidade min(1, p_alvo/p_draft), preservando exatamente a distribuição do alvo; greedy usa argmax. A receita usa `probabilistic`; os cards oficiais usam greedy com k=7; o A/B deu empate.
35. **DSpark (drafting em bloco paralelo + cabeça de Markov).** O draft recebe o token âncora + `block_size-1` tokens de "ruído" (`dspark_noise_token_id`) e estados ocultos de camadas-alvo do modelo principal, e produz um bloco inteiro em uma passada paralela (com atenção de janela deslizante própria). Depois, uma cabeça de Markov de rank 256 (bigrama low-rank: `markov_w1` embute o token anterior, `markov_w2` projeta para bias de vocabulário) refina sequencialmente cada posição; há também uma cabeça de confiança (recipe/overlay/.../nvidia/dspark.py:870-990).
36. **k (`num_speculative_tokens`) vs `dspark_block_size` vs `num_nextn_predict_layers`.** O bloco treinado é 5; k<5 trunca blocos; a regra "MTP module reuse" exige k divisível por 3 no Vision-Exp → k=6 (uma query de ruído extra). O hotfix block-k remove a regra para DSpark (estágios empilhados, nunca reusados por passo) e permite k=5 (+2–10 %).
37. **Estado persistente do draft por requisição (slot mapping) e batches ragged.** O draft tem seu próprio KV de janela; com vários requests ele precisa seguir a identidade da requisição, não a posição da linha no batch, e lidar com `query_start_loc` irregular quando prefill e decode se misturam. Esse foi o patch de concorrência do Keys (Stage-C) e é por que `max_num_seqs>1` correto não é trivial.
38. **Linhas de verificação (verify rows) = `max_num_seqs × (k+1)`.** É o M do passo de decode especulativo (42 a 6×7), que define tamanhos de CUDA graph e, empiricamente, a probabilidade dos stalls estocásticos #141 (nada acima de 64 linhas estabelecido como limpo; default 48 com padding).

### Bloco F — Motor vLLM

39. **Arquitetura do vLLM (API server, EngineCore, Scheduler, Workers, model runner V1/V2).** O servidor OpenAI recebe requisições, o EngineCore agenda passos, workers executam o modelo em cada GPU. A receita fala com o runner V2 (que rejeita `thinking_token_budget`), com RPC `sample_tokens` (timeout `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`) e com o broadcast de SHM entre processos (`shm_broadcast`).
40. **PagedAttention e blocos (`--block-size 256`).** O KV é alocado em blocos de tamanho fixo endereçados por tabelas de blocos, como memória virtual; elimina fragmentação e permite compartilhamento de prefixos. Aqui o grupo MLA usa blocos de 256 tokens e os grupos SWA 64/4/8 com a mesma página em bytes.
41. **Dimensionamento do pool de KV (`--gpu-memory-utilization`, profiling).** No boot o vLLM carrega pesos, faz *profiling* de ativações e captura de graphs e dá o resto (até `util × memória`) ao KV. Em memória unificada, isso compete com o SO: workspaces de prefill maiores caem fora desse orçamento e aparecem como queda de MemAvailable.
42. **`max_model_len` e `max_num_seqs` como tetos, não reservas.** O vLLM não reserva 1M × 6; ele aloca blocos sob demanda e enfileira/preempta quando o pool acaba. Por isso 1M de teto convive com 6 slots — desde que a soma dos tokens vivos caiba (~2.33 M).
43. **Continuous batching e orçamento de tokens (`--max-num-batched-tokens`).** A cada passo o scheduler junta decodes e pedaços de prefill até o orçamento (8192). Com k=6, cada decode consome k+1 posições do orçamento.
44. **Chunked prefill, `--long-prefill-token-threshold` e prefills parciais.** Prompts longos são fatiados em chunks para não bloquear decodes; o limiar 1024 limita cada chunk; `max_num_partial_prefills` (não aplicado pelo vLLM 0.25.2 — daí o #27) limita quantos prompts podem estar "no meio do prefill" ao mesmo tempo (`DSPARK_MAX_INFLIGHT_PREFILLS`).
45. **Fairness/starvation e preempção no scheduler.** Sem os limites acima, requisições decodificando podiam receber 0 tokens por passo (puladas, sem preempção) enquanto prefills longos comiam o orçamento. #27 (cap de admissão) + #43 (piso de serviço por lane) resolvem; mede-se com "spread" de TTFT/decode entre streams (gate26).
46. **Async scheduling.** O scheduler prepara o passo seguinte enquanto a GPU roda o atual, escondendo overhead de CPU/ZMQ (RTT 0.33 ms para o worker). Interage com gramáticas e spec decode (placeholders de tokens ainda não verificados) — por isso existe o knob de A/B do #191.
47. **Prefix caching com hash encadeado de blocos.** Cada bloco de 256 tokens tem um hash que depende de todos os anteriores; prompts com o mesmo início compartilham blocos. Uma diretiva de esforço logo após BOS muda o bloco 0 e zera o compartilhamento entre níveis (#144); janelas SWA e o estado do draft complicam hits (#26/#36, SWA prefix fix, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`).
48. **Coordenador de KV híbrido (múltiplos grupos de atenção).** Com 1 grupo MLA + 3 grupos SlidingWindowMLA, o hit comum de prefixo é o mínimo entre grupos; grupos SWA só guardam "caudas". O hotfix #26 v2 e a retenção esparsa (um tail a cada 4096 tokens) conciliam isso sem anexar KV nulo.
49. **CUDA graphs (captura, FULL vs PIECEWISE, "breakable", tamanhos de captura).** Gravar a sequência de kernels de um passo e reexecutá-la elimina overhead de lançamento Python. O vLLM captura um conjunto de tamanhos de batch; com spec decode arredonda para múltiplos de `uniform_decode_query_len` (k+1=7). Um tamanho fora da lista roda *eager* (c=6 a 42 tokens → −12 %); "breakable graphs" do Anemll são 13–28 % mais lentos.
50. **Compilação JIT de kernels (Triton, TileLang, CuTeDSL, DeepGEMM, FlashInfer) e caches persistentes.** Muitos kernels são compilados na primeira vez que um shape aparece; a chave de cache inclui parâmetros constexpr (ex. `BLOCK_SIZE = min(256, next_pow2(tokens+6))`) e especializações de alinhamento. Compilar no meio do serviço trava um rank TP e o watchdog NCCL (600 s) mata o par → a receita persiste todos os caches no volume HF e roda o boot-shape warmup.
51. **Autotuning de kernels (FlashInfer autotune).** No boot, testa variantes de kernel por shape e escolhe a mais rápida; o resultado é cacheado por hash de config. Cobertura incompleta (só `tokens=16` no warmup sparse-MLA) gera fallbacks `tactic=-1`.
52. **Tokenizer mode / encoder Python em vez de Jinja chat template.** O checkpoint DSV4 define o formato de conversa em `encoding/encoding_dsv4.py` (papéis, `<think>`, DSML, diretivas de esforço). A receita copia esse arquivo para dentro do vLLM e o corrige (#21, mapeamento de esforço, visão, assistant-final, #144).
53. **Modelos de raciocínio: `<think>`, `reasoning_content`, modos de thinking e `reasoning_effort`.** O modelo escreve raciocínio entre `<think>…</think>` e depois a resposta; o reasoning parser separa isso. `reasoning_effort` `high`/`max` injeta diretivas (80/93 tokens) que aumentam muito o raciocínio; `DEFAULT_THINKING` define o padrão do servidor e `chat_template_kwargs` da requisição vence.
54. **`max_tokens` vs `thinking_token_budget`.** `max_tokens` conta raciocínio + resposta + markup; com `max` o raciocínio pode consumir tudo e devolver `content:null`/`length`. `thinking_token_budget` (opt-in, GPU) limita só o raciocínio e força um `</think>`, deixando o resto para a resposta.
55. **Tool calling em DSML e parsers de ferramentas.** O DSV4 emite chamadas como markup `<｜DSML｜tool_calls>` / `<｜DSML｜invoke name=…>` / `<｜DSML｜parameter …>`; o `--tool-call-parser deepseek_v4` converte em `tool_calls` estruturados. Se um único token estrutural sai errado (temperatura alta) a chamada vira texto e benchmarks contam como "sem tool call".
56. **`tool_choice` (auto/none/named/required), `parallel_tool_calls` e `strict`.** Controlam se e qual ferramenta deve ser chamada e se argumentos devem obedecer o schema. O hotfix #191 transforma violações desses contratos em regeneração + HTTP 500, em vez de um 200 errado.
57. **Saída estruturada (grammar-constrained decoding) com XGrammar.** Um autômato (FSM) derivado de JSON schema/regex mascara (bitmask) tokens inválidos a cada passo. Com spec decode, tokens de draft precisam avançar o FSM corretamente e tokens após o término não podem entrar — bugs corrigidos por #44993 (grammar advance) e #52805/#53046 (#136).
58. **`finish_reason` (stop / length / tool_calls) e streaming SSE append-only.** O motivo de término diz ao cliente se a resposta está completa. O #55 garante `length` (não `tool_calls`) quando `max_tokens` corta uma chamada; em streaming fragmentos já enviados não podem ser retirados.
59. **Stop strings.** Strings do cliente que encerram a geração; o vLLM as testava contra todo o stream, inclusive o raciocínio. O hotfix de supressão só as avalia após `</think>`.
60. **API OpenAI-compatível: Chat Completions, Completions, Responses API, `/tokenize`, `/metrics`.** A receita expõe tudo em `:8888`; a Responses API tem modo stateful (`store` + `previous_response_id`) que exige o backport de store limitado; `/tokenize` é usado por warmup/benchmarks; `/metrics` para aceitação.
61. **Multimodal na API (`image_url`, `--limit-mm-per-prompt`, papéis).** Imagens como data URLs em partes `image_url` de mensagens `user`; limite padrão 8 por requisição. Imagem estruturada em `system/assistant/tool` → 400 que fica no histórico.
62. **Autenticação da API (`--api-key`).** vLLM aplica Bearer só às rotas `/v1`, `/v2`, `/inference`; outras rotas continuam abertas → é preciso controle de rede. A receita suporta uma chave (`VLLM_API_KEY`) ou várias (`DSPARK_API_KEYS`) e redige as chaves do log.

### Bloco G — Inferência distribuída

63. **Tensor parallelism (Column/RowParallelLinear, all-reduce, all-gather, VocabParallelEmbedding).** Cada matriz grande é dividida entre GPUs; saídas parciais são somadas (all-reduce) ou concatenadas (all-gather). Em TP=2 há ≈108 coletivos por passo de decode, latency-bound (30–50 µs cada na RoCE sem GDR).
64. **O que é replicado vs particionado em TP.** Algumas partes não são divididas: o KV latente MLA, o Lightning indexer (`ReplicatedLinear`), `fused_wqa_wkv`. Isso explica por que TP=3 aumenta capacidade (pesos por rank menores, KV maior) mas não acelera o prefill de contexto longo.
65. **Divisibilidade e padding em TP (TP=3: 64→72 cabeças, grupos 8→9).** TP exige que cabeças/grupos dividam pelo número de ranks; o patch TP=3 adiciona um grupo de pad zerado. Um shard errado gera texto fluente e errado sem erro — por isso `validate_tp3.sh` testa aritmética, recall e agulha.
66. **Pipeline parallel, expert parallel, DCP e sequence parallel.** PP=1 aqui; EP não é usado; DCP (decode context parallel) dividiria o KV por tokens entre ranks mas não está ligado para DSV4 nesta imagem; sequence parallelism aparece só no SP indexer opt-in. Conhecer as alternativas explica as escolhas e os trade-offs documentados (item8 opção D).
67. **vLLM multi-nó (`--nnodes`, `--node-rank`, `--master-addr/port`, `--headless`, executor `mp`, TCPStore).** Cada nó roda um container; o rank 0 serve HTTP e hospeda o rendezvous; o worker roda headless. O launcher sobe worker primeiro e passa variáveis rank-aware pelo SSH.
68. **Comunicação intra-nó por shared memory (shm_broadcast, SpinCondition).** O scheduler envia trabalho aos workers por um ring buffer em SHM com espera ativa curta. O hotfix #79 reduz o spin de 1 s para 2 ms (economia de CPU/energia do SoC) e o #117 limita a espera de leitor e libera slots em exceção.
69. **Watchdog NCCL, timeouts e flight recorder.** O ProcessGroupNCCL mata o processo se um coletivo não completa em 600 s; o flight recorder grava os últimos coletivos por rank para diagnosticar travamentos. A receita pina `TORCH_FR_*`/`TORCH_NCCL_*` e expõe um FIFO para dumps sob demanda.

### Bloco H — Rede e hardware

70. **RDMA, InfiniBand vs RoCEv2, HCA, verbs.** RDMA permite uma NIC ler/escrever memória remota sem CPU; RoCEv2 é RDMA sobre Ethernet/UDP/IP. NCCL usa `NCCL_NET=IB` (verbs) sobre as portas RoCE do ConnectX-7 (`/dev/infiniband` no container, `memlock` ilimitado).
71. **Tabela GID e índice GID RoCEv2.** Cada porta tem uma tabela de GIDs (endereços); para RoCEv2/IPv4 o GID termina em `ffff:aabb:ccdd`. O índice certo varia por nó e muda com reboot/link; a receita valida em `/sys/class/infiniband/*/ports/*/gids` e deixa o NCCL escolher por HCA.
72. **Variáveis de ambiente do NCCL.** `NCCL_IB_HCA` (seleção com gramática própria), `NCCL_SOCKET_IFNAME` (bootstrap), `GLOO_SOCKET_IFNAME`, `NCCL_CROSS_NIC`, `NCCL_IB_MERGE_NICS`, subnet-aware routing, GDR/DMA-BUF, NVLS, CUMEM, GIN. Saber o que cada uma controla é necessário para montar a malha 2 ou 3 nós.
73. **ConnectX-7 "socket-direct": 2 PFs PCIe Gen5 ×4 por porta, MTU jumbo.** No GB10, cada porta QSFP de 200 Gb/s é servida por duas funções PCIe de ~126 Gb/s; usar só uma limita a banda. MTU Ethernet 9000 permite RoCE MTU 4096; mediu-se +7.6 % end-to-end com MTU 9000 + duas HCAs.
74. **Topologia em anel QSFP para 3 nós.** Cada par de nós é ligado por um cabo com seu próprio /24; nenhum /24 alcança todos → bootstrap TCP vai pela LAN comum, `NCCL_IB_HCA` lista as duas portas, merge desligado, roteamento por sub-rede /24.
75. **GB10 / sm_121(a) / Blackwell de consumo vs datacenter.** O GB10 tem compute capability 12.1; muitos kernels/gates só miram sm_100 (datacenter) ou sm_120. Daí `TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a`, o alias de headers DeepGEMM sm121→sm120 e a relaxação do gate FP4 do indexer.
76. **Memória unificada (UMA) de 128 GB.** CPU e GPU compartilham a mesma LPDDR5X; o que o SO, o page cache, processos de desktop ou o servidor NFS usam sai do mesmo bolo. Explica: util 0.835 (não 0.95), recomendação de desligar earlyoom e GUI, `NV_ERR_NO_MEMORY` no boot, e por que chunks de prefill maiores derrubam MemAvailable.
77. **Largura de banda de memória como limite de decode.** Com ~230–250 GB/s efetivos, o tempo mínimo de um passo é bytes lidos / banda; por isso o ganho vem de ler menos bytes por token aceito (quantização, especulação, concorrência), não de mais FLOPs.

### Bloco I — Operação / DevOps

78. **Docker Compose para serviço GPU multi-nó.** `network_mode: host`, `ipc: host`, `shm_size`, `ulimits`, `devices`, `gpus`, volumes, healthcheck, `restart: unless-stopped` e o contrato de exit code 3. Cada detalhe resolve um problema real documentado (issues #38, #72, #185).
79. **Arquivos `.env` "sourced" por bash e armadilhas.** Valores são interpretados pelo shell: JSON perde aspas (`{"image":8}` → `{image:8}`), valores com espaço sem aspas executam comandos, BOM/CRLF quebram parsing. Por isso a receita usa `image=N`, monta JSON no container e normaliza o arquivo.
80. **Patching de código-fonte fail-closed e idempotente em containers.** Âncoras exatas, marcadores, pins SHA-256, escrita atômica, verificação e rollback; aplicar antes de `exec vllm`; recriar (não reiniciar) para mudar. É a técnica central de toda a receita — cada correção é um patch sobre vLLM 0.25.2 fixado.
81. **Cache do Hugging Face Hub (snapshots, blobs, refs, revisão, offline).** `snapshot_download` grava em `hub/models--org--name/snapshots/<commit>`; fixar revisão e escrever `refs/main` permite servir offline (`HF_HUB_OFFLINE=1`). Symlinks para blobs exigiram correção no #144 (`realpath`).
82. **NFSv4 e volumes Docker NFS.** O head exporta o cache HF read-only; o worker monta como volume Docker (`nfsvers=4.2, nconnect=8, rsize/wsize 1 MiB, hard`), com overlays locais graváveis para caches JIT. Evita 157 GiB duplicados; carga de pesos passa pelo mesmo link RoCE.
83. **Warmup de shapes no boot.** Disparar requisições sintéticas com contagens exatas de tokens/concorrência para compilar todas as variantes de kernel antes do tráfego real. Complementa caches persistentes e é explicitamente não fatal.

### Bloco J — Medição e qualidade

84. **Metodologia de benchmark de LLM.** Prefixos frios únicos (anti-cache), contar tokens por `usage.completion_tokens` (SSE subestima ~4.5× com MTP), separar TTFT de decode, aquecer ≥2 requisições, medianas de vários trials, uma variável por boot, comparar com baseline do mesmo dia. Toda a pasta `results/` e `docs/CLAUDE/` segue isso.
85. **Avaliação de contexto longo: NIAH, RULER (retrieval multi-chave, variable tracking, common-words), garble sweep.** Testes sintéticos com comprimentos verificados por `/tokenize` que medem se o modelo realmente usa o contexto, não só se "responde". `ruler-lite.py` 8/8 e garble CLEAN até ~900K são gates para aceitar mudanças.
86. **Avaliação de tool calling (tool battery, deep-context tools, tool-eval-bench).** Chamadas simples/complexas/paralelas/multi-turn e truncamento, em contexto curto e profundo. Benchmarks que leem só `message.tool_calls` são sensíveis a erros de sintaxe DSML.
87. **Detecção de loop vs cauda longa de raciocínio.** `loop_detector.py` mede novidade de 8-gramas de palavras por janela: loop colapsa e não se recupera; cauda longa continua dizendo coisas novas. Distingue quando aumentar `max_tokens` ajuda e quando não.
88. **Fairness spread e gates de concorrência (gate26).** Razão entre o pior e o melhor stream (TTFT/decode) numa rajada; usada para decidir `DSPARK_MAX_INFLIGHT_PREFILLS`. Mostra que throughput agregado sozinho esconde starvation.

### Bloco K — Alinhamento

89. **Abliteração / ablação direcional de recusa.** Encontra-se uma direção no espaço residual associada a recusas e remove-se sua componente: `y ← y − λ·(y·v)·v`. Aqui é aplicada em runtime (λ=3.5, camadas 10–42) com uma direção unitária de 4096 dims baixada de repo gated, em vez de baixar um checkpoint de 157 GiB alterado; drena a aceitação do draft e é experimental no Vision-Exp.

### Bloco L — Offload de KV (experimental)

90. **Offload/persistência de KV (LMCache).** Servidores por nó guardam chunks de KV em RAM (L1) e disco (L2), permitindo reusar prefixos entre reinícios (107K tokens: ~1.9 s vs ~65 s). Exige `PYTHONHASHSEED=0` (chaves por `hash()`), desliga `expandable_segments`, e um servidor morto trava o engine silenciosamente — por isso está fora do caminho padrão.

---

## 6. Receita passo-a-passo (recriar do zero, com o porquê de cada decisão)

### Passo 0 — Entender o orçamento antes de tocar em qualquer coisa

- **Memória**: cada Spark tem ~121.7 GiB utilizáveis compartilhados entre CPU e GPU. Os pesos do DSV4-Flash-Vision-Exp ocupam ~79 GiB por rank em TP=2 (o checkpoint total tem ~157 GiB; não cabe inteiro com KV em um nó). Com util 0.835 sobram ~17 GiB de KV por rank (≈2.33 M tokens) e ~3–5 GB de RAM livre para o SO. **Decisão**: TP=2 é o mínimo; o que sobrar vira KV; o host precisa estar "enxuto" (sem GUI, sem earlyoom, polkit/wireplumber controlados) porque qualquer RAM do host sai do pool (fable5-1 §3.1).
- **Banda**: decode é limitado por streaming de pesos (~17 GB/passo/rank). **Decisão**: maximizar tokens por leitura de pesos → especulação (DSpark) + concorrência (6 slots) + CUDA graphs regulares.
- **Contexto**: 1M é possível porque o DSV4 comprime o KV (C4A/C128A + MLA de 584 B/linha). **Decisão**: `--max-model-len 1048576` como *teto*, e controlar concorrência pelo pool, não pela reserva.

### Passo 1 — Hardware e rede

1. Dois DGX Spark ligados por cabo QSFP direto nas portas ConnectX-7; cada lado com IPs em /24 próprio na interface (ex. `10.0.22.1`/`10.0.22.2` em `enp1s0f1np1`/`enp1s0f0np0` — o nome da porta "que encara" o outro nó pode diferir; .env:57-64).
2. Opcional mas medido: **MTU 9000** nas interfaces das duas funções PCIe de cada porta (RoCE `active_mtu 4096`), rings `ethtool -G rx 8192 tx 8192`, e segunda PF em sub-rede distinta (evitar ARP-flux) (fable5-1:228-239; .env:699-703). Por quê: a porta de 200 Gb/s é servida por duas PFs PCIe Gen5 ×4; uma só limita a ~126 Gb/s; MTU 1500 limita RoCE a 1024.
3. Verificar RDMA: `ibstat`, `show_gids`, `ibdev2netdev` para mapear `roce*` ↔ `enp*` (.env:697; start:1132). Cada HCA selecionada precisa de um GID "RoCE v2" cujo sufixo case com um IPv4 da própria netdev — o launcher checa isso e falha fechado.
4. SSH sem senha head→worker (`ssh -o BatchMode=yes worker true`).
5. Host: desligar earlyoom (README.md:126-127), preferir `multi-user.target` (sem Xorg/gnome-shell segurando contexto de GPU), checar swap e MemAvailable (fable5-1 §3.1).
6. Alias de loopback `10.0.0.1/32` roteado pelo CX7 é usado como `MASTER_ADDR`/`VLLM_HOST_IP` no cluster de referência (fable5-1:62; start:1540-1542) — **[inferência: é uma convenção do cluster do autor, não obrigatória]**.

### Passo 2 — Imagem de runtime

1. `docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1` **nos dois nós**, e fixar por digest no `.env` (`@sha256:a8394849…`). Por quê: todos os hotfixes são *source-exact* contra vLLM `0.25.2.dev0+g752a3a504.d20260714`; outra build quebra âncoras (fail-closed) ou pior.
2. Checar registro de env vars da imagem (snippet em docs/ENVS.md:34-44): `VLLM_DSPARK_*` Stage-C não existem no Anemll e só geram warning.
3. Alternativa histórica: `./build-dspark-vllm-runtime.sh` → overlay + Stage A/B/C → `vllm-dspark-runtime:dspark-nvfp4-stage-c` (em ambos os nós), + `docker-compose.stage-c.override.yml` e bloco Stage-C do `.env`.

### Passo 3 — `.env.dspark`

1. `cp .env.dspark.example .env.dspark` e preencher (README.md:34-58): `WORKER_HOST`, `MASTER_ADDR`, `VLLM_HOST_IP`, `WORKER_VLLM_HOST_IP`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `HF_CACHE`, `DSPARK_VLLM_IMAGE`, e `WORKER_NCCL_*` se a porta do worker tiver outro nome.
2. **NCCL_IB_HCA**: se possível, as duas PFs no formato exato `=rocep1s0f1,roceP2p1s0f1` (head) / `=rocep1s0f0,roceP2p1s0f0` (worker), mantendo `NCCL_IB_GID_AUTO=1` e sem `NCCL_IB_GID_INDEX` (.env:691-728).
3. Deixar os knobs de serviço no default: `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=8192`, `LONG_PREFILL_TOKEN_THRESHOLD=1024`, `DSPARK_MAX_INFLIGHT_PREFILLS=1`, `GPU_MEMORY_UTILIZATION_TEXT=0.835`, `MTP_NUM_TOKENS=6`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `DEFAULT_THINKING=low`, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`.
4. Segurança: se expor em `0.0.0.0`, definir `VLLM_API_KEY` **ou** `DSPARK_API_KEYS` (entre aspas) e `chmod 600 .env.dspark`; restringir a porta na rede porque rotas fora de `/v1` ficam abertas.
5. Não definir `GPU_MEMORY_UTILIZATION`, `DSPARK_MODEL`, `TP_SIZE`, `NNODES`, `ABLATE` à mão (o launcher controla).

### Passo 4 — Pesos

1. `./prepare-dspark-model-cache.sh --official` (ou `--yes`): baixa `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b3…` dentro de um container da própria imagem (`snapshot_download`), com HF online forçado, timeouts 120/30 s, grava `refs/main`, verifica que todos os shards de `model.safetensors.index.json` existem (prepare:233-329).
2. Se `DSPARK_WORKER_HF_NFS=0` (default): o prepare se copia ao worker e roda lá (`PREPARE_WORKER=0`). Se `=1`: nada no worker; o start exporta o cache por NFS.
3. Depois manter `HF_HUB_OFFLINE=1`. Se a revisão pinada não estiver no cache, `validate-dspark-config.sh` avisa que o start faria um download de ~155 GB.
4. Opcional (abliteração): aceitar termos em `drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit`, `HF_TOKEN` no shell, `./prepare-dspark-model-cache.sh --abliterated` → baixa `RESPONSIBLE_USE.md` e a direção de 18 KiB (sha256 `6e4d8a8f…`), grava `ABLITERATED=1`.

### Passo 5 — Validação offline

- `bash scripts/ci-validate.sh`: sintaxe de todos os scripts, `py_compile` dos patches, ~60 testes unitários CPU (hotfixes contra fixtures das fontes fixadas), e guardas de regressão (ex. patch CPU de budget ausente, #26 v1 não aplicado, cap #27 default 1 em env e compose, arredondamento da captura, healthcheck ciente de `VLLM_HOST`, TP forçado a 2 no caminho padrão) (scripts/ci-validate.sh).
- `./validate-dspark-config.sh`: imprime o comando renderizado do vLLM (filtrado) para conferir flags.

### Passo 6 — Subir (o que o launcher faz e por quê)

Sequência de `start-deepseek-v4-flash-dspark.sh`:
1. Parse de `--host/--port`; normaliza o `.env` (BOM/CRLF) numa cópia `0600`; protege contra `DSPARK_API_KEYS` divergente no ambiente (start:38-156).
2. Deriva `GPU_MEMORY_UTILIZATION` de `GPU_MEMORY_UTILIZATION_TEXT`; resolve modelo/revisão; valida ablação (faixa de camadas, λ finito, SHA-256) (start:158-252).
3. **Valida k**: sem block-k exige `MTP_NUM_TOKENS ≥ 5` e `% 3 == 0` (start:254-276).
4. Valida porta, knobs numéricos, `DEFAULT_THINKING`; monta headers de auth; avisos de bind aberto/permissões; checa presença dos patchers habilitados (start:278-595).
5. Força `TP_SIZE=2/NNODES=2` (ou 3 com `DSPARK_TP3=1`), resolve IPs/overrides por nó, NFS (start:597-790).
6. Pré-requisitos: `docker`, `ssh`, `scp`, `curl`, `sha256sum`; imagem nos dois nós; SSH; container já existente → exit 3; porta ocupada → exit 1 (start:1435-1537).
7. Staging da direção (se ablação), **validação de GIDs RoCEv2** nos dois nós, ajustes de bootstrap TP3, imprime o perfil resolvido (start:1574-1580).
8. Sincroniza compose, `.env`, proposer, todos os patchers e `patches/vision_exp/` e `patches/tp3/` para o worker (start:1582-1889).
9. NFS (se `=1`): garante exportador, diretórios JIT, recria o volume `dspark-hf` no worker, confirma que o worker enxerga `hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp` (start:1891-1941).
10. `docker compose config --quiet` em head e worker; `--check` dos hotfixes opt-in habilitados (+ #117 sempre) no worker e depois no head (start:1942-2050). Por quê: um patch que não bate na fonte deve impedir o boot **antes** de qualquer rank subir, para não deixar o par meio-patcheado.
11. **Sobe o worker** (`NODE_RANK=1 HEADLESS=1`, variáveis NCCL do worker, `VLLM_HOST_IP` do worker) e depois **o head** (`NODE_RANK=0`) (start:2052-2061). Por quê worker primeiro: o head hospeda o TCPStore; o worker espera por ele — subir e reiniciar o head no meio do load derrubava o handshake (#38).
12. Espera `/v1/models` (até 100×15 s), imprime logs a cada tentativa; faz uma requisição smoke (com `thinking_token_budget:1` se #31 ligado, para compilar os kernels do budget); roda `boot-shape-warmup.sh` (não fatal) (start:2077-2144).

Dentro do container (cada rank), o entrypoint: exporta CUDA no PATH → parse de chaves → unset de NCCL vazio → dir do flight recorder → ablação → instala encoder + mapeamento de esforço + #21 (+ #31) + #55 → #22 → #79 → #117 → trem v0.27 → redação (se chave) → store Responses → #138 → Codex → Vision-Exp → #141 → SP indexer → alias DeepGEMM → #109 → #27 → adaptive chunk → Markov → #43 → #26 → #133 → ablação → supressão de stops → assistant-final → #144 → #136 → #191 → args async/mxfp4 → block-k → rope-swa → swa-prefix → DSML recovery → MXFP4 indexer → C128A → plugin GB10 → pad TP3 → speculative-config → kwargs de thinking → revisão → limit-mm → `exec vllm serve …` (compose:414-529).

### Passo 7 — Verificação

1. `curl -fsS http://127.0.0.1:8888/v1/models` → id `deepseek-v4-flash-vision-exp`, `max_model_len 1048576`.
2. Log do head: `Available KV cache memory: ~17 GiB`, `GPU KV cache size: ~2.3M tokens`, `Maximum concurrency … ~2.2x`; **sem** "Truncating max_cudagraph_capture_size"; `Capturing CUDA graphs (FULL) 6/6`; linhas `[dspark-…] patched` só dos opt-ins ligados; sem `EngineDead`/`NCCL WARN` (fable5-1:431-432; ab-results-2026-09-02:33).
3. `./smoke-deepseek-v4-flash-dspark.sh` (6 requisições concorrentes) e `./status-…sh` (exit 1 se algo falhar).
4. Sondas de qualidade: 17×23 = 391, capital do Japão = Tokyo (ab-measure.sh). Para TP=3: `scripts/validate_tp3.sh` e log `DSv4 TP pad: heads 64 -> 72` em cada rank.
5. Se garbling/loops/XML vazando: primeiro testar direto em `:8888` com `temperature: 0`; checar mesma digest nos dois nós, snapshot completo (incl. `encoding/encoding_dsv4.py`) nos dois nós, e só então culpar o harness (README.md:504-516).
6. Auditoria completa: `bash scripts/run-audit.sh --model deepseek-v4-flash-vision-exp` (throughput, aceitação, RULER-lite, tool battery, garble) (AUDIT.md).
7. Responses API: `python3 scripts/verify-responses-api-live.py …` (com store habilitado se usar `previous_response_id`).

### Passo 8 — Tuning (uma variável por boot, com baseline do mesmo dia)

| objetivo | knob | evidência |
|---|---|---|
| Não perder 13–28 % de decode | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (nunca ausente) | 4.7 |
| c=6 em CUDA graph | captura arredondada a múltiplo de 8 (já default) | 4.8 G |
| Prompts grandes de código | `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=16384`, `GPU_MEMORY_UTILIZATION_TEXT=0.87` | README.md:154-160 |
| TTFT longo (≥128K) | `DSPARK_ENABLE_SP_INDEXER=1` (−4 a −6 %) | 4.8 E |
| Rajadas de prompts médios mais justas | `DSPARK_MAX_INFLIGHT_PREFILLS=2` (qualificado #217) | 4.9 |
| Decode single-stream | `DSPARK_ENABLE_DSPARK_BLOCK_K=1` + `MTP_NUM_TOKENS=5` (+2–10 %) | 4.16 |
| Mais KV | subir util só depois de liberar RAM do host (cada +0.01 ≈ +165K tokens) | 4.2 |
| Bootstrap NCCL rápido | `NCCL_GIN_ENABLE=0` | 4.15 |
| Largura de banda de rede | dual-HCA exata + MTU 9000 | 4.15 |
| Agentes com esforço variável | `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN=1` (gate de qualidade pendente) | 4.16 |
| Budget de raciocínio | `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1` + cliente enviando `thinking_token_budget` | 4.16 |
| **Não fazer** | `LONG_PREFILL_TOKEN_THRESHOLD>1024`, adaptive chunk (RAM), `B12X_W4A16_TC_DECODE=1` (−15 %), Markov replicado (−c=6), k<5, k>6 sem motivo, `DSPARK_SKIP_ISSUE22_HOTFIX=1` | 4.8 |
| Mais slots/capacidade | TP=3 (`./start-tp3.sh`, 16 slots, ~5 M tokens; prefill mais lento) | 4.10 |
| Throughput agregado máximo em contexto curto | Stage-C + 200K/16 + `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` (histórico) | 4.12 |

Regras de A/B do repo: `scripts/ab-boot.sh LABEL KEY VAL [--long]` (edita `.env`, stop/start, gates de boot nos dois ranks, gate de memória ≥3 GB, mede com `ab-measure.sh` + 5 trials extras de c=1); ruído c=1 ≈ ±10 tok/s → ≥8 trials; TTFT ruído ±2 %; rejeitar knob que custe ≥0.5 GB de MemAvailable ou aumente swap (ab-results-2026-09-03:77).

---

## 7. Inconsistências e pontos de atenção encontrados

1. **Contagem do boot-shape warmup**: comentário do `.env:651-658` e ENVS:79 dizem 35 requisições a N=6 (23 a N=4); o script atual agenda 47/35/18 (N=6/4/2) após a adição dos arms de sampler (boot-shape-warmup.sh:279-292; CHANGELOG.md:184).
2. **Chave do bucket do warmup**: o script documenta `next_pow2(scheduled_tokens + 6)` ("+6 = 1 + num_speculative_tokens" a k=5), mas o lane usa k=6; o relatório fable5-1:369-371 aponta que deveria ser `+7`. O comentário em ENVS:77 também diz +6 **[possível lacuna de cobertura dos buckets]**.
3. **Default de `DSPARK_MAX_INFLIGHT_PREFILLS`** mudou quatro vezes (2 → 1 → 2 → 1). O CHANGELOG de 2026-09-02 ainda diz "default is now 2", superado pela entrada de 2026-09-04 (CHANGELOG.md:89, 120).
4. **ENVS.md "GPU_MEMORY_UTILIZATION≈0.80"** (ENVS:190) vs default 0.835 no `.env`/compose.
5. **`print_resolved_profile`** usa fallbacks antigos (`MAX_MODEL_LEN:-1000000`, modelo `deepseek-ai/DeepSeek-V4-Flash-DSpark`, served `deepseek-v4-flash-dspark`) (start:1320-1322) — só afetam a impressão quando a variável está vazia. `status`/`prepare` usam fallback de imagem Stage-C (`vllm-dspark-runtime:dspark-nvfp4-stage-c`) quando `DSPARK_VLLM_IMAGE` não está definido (status:80; prepare:77).
6. **`validate-dspark-config.sh`** imprime captura `MAX_NUM_SEQS*(MTP+1)` sem o arredondamento para múltiplo de 8 e ecoa `DSPARK_SLOT_CLAMP` e `WO projection` (knobs Stage-C) (validate-dspark-config.sh:153-157).
7. **Dois layouts diferentes chamados `nvfp4_ds_mla`**: 584 B (Anemll/Stage C, o que roda) vs 352 B (`Dockerfile.gb10-dsv4-dspark`, imagem vLLM 0.26 alternativa) vs 416 B (Stage B, abandonado).
8. **Narrativa do pad TP=3**: `patches/tp3/apply_tp3_patch.py` (o que é aplicado) e docs/TP3.md dizem "grupos 8→9, 8 cabeças/grupo" e chamam a alternativa de "estruturalmente errada"; `patches/dsv4_tp_pad.py` (também sincronizado) descreve o oposto (R1 = replicar grupos e pad de cabeças por grupo 8→9, "R2 rejeitado"). O log esperado `heads 64 -> 72` é compatível com ambos.
9. **`dspark_proposer.py`** é exigido e copiado ao worker pelo start, mas o compose Anemll não o monta (comentário diz ser overlay opcional Stage-C).
10. **docs/vllm-027-new-patches.md** diz que #48957 e #50298 estão "script verified, not yet applied"; o compose atual os aplica no trem padrão (compose:448).
11. **`VLLM_HOST`**: fallback do compose é `127.0.0.1`, mas o `.env.dspark.example` define `0.0.0.0` (API aberta na LAN por padrão, com WARN).
12. **`results/dspark-metrics.txt`** mostra aceitação quase nula nas posições 2–5 (84/63/1/1/0/0 de 105) num modelo chamado `deepseek-v4-flash-0731-vision` — sem contexto no repo sobre a carga usada; não comparar diretamente com os ~0.9/0.7/0.6… das outras medições.
13. **Scripts de benchmark** (`bench-miaai.py`, `spec-acceptance.py`, `ruler-lite.py`, `tool-battery.py`, `boot-shape-warmup.sh`) ainda têm `deepseek-v4-flash-0731` como modelo default; é preciso passar `--model deepseek-v4-flash-vision-exp` (o `run-audit.sh` já usa o nome certo).
14. **Números "best seen"**: AUDIT/README 73–83 tok/s c=1 vs 52–57 medidos em 2026-09-02 no mesmo lane, sem explicação por knob; tratar os primeiros como teto.


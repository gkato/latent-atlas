# Inventário — GLM-5.3-Flash EXL3 em 2–4× DGX Spark (escopo: lançamento, topologia, parametrização)

> Repositório lido (somente leitura): `repos/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, HEAD `94ae731` ("Merge pull request #292 from MiaAI-Lab/feat/dense-h3-preset").
> Arquivos lidos integralmente: `README.md` (2100 linhas), `.env.example`, `.env.tp3.example`, `.env.tp4.example`, `examples/*`, `start.sh` (3038 linhas), `start-tp3.sh` (partes de configuração, validação, preflight, scripts internos, launch — o restante é espelho do `start.sh`), `start-tp4.sh` (configuração, NCCL, args), `Dockerfile`, `Dockerfile.e3-layer`, `scripts/*`, `files/*`, `CHANGELOG.md`, `docs/benchmarks/tp3-combined/*`, `docs/ladder-final-2026-09-01.json`, `docs/_cold_prefill_raw.json`, e, como apoio, `docs/adopt-fp8-all-kda-bf16-2026-09-26.md`, `docs/cold-load-uma.md`, `docs/uvm-livelock-gb10.md`, `docs/images-per-request.md`, `docs/DESIGN-indexer-workspace.md`, `docs/kda-bf16-large-m.md`, `docs/cold-prefill.md`, `docs/improve-prefill.md`, `docs/tp3-throughput-results.md`, `docs/diditfix.md`, `docs/astra-fix.md` (parcial), `docs/sm121-perf-paths.md`, `docs/tool-choice-none.md`, `docs/apc-retention-qualification.md`, `docs/cooperative-moe.md` e as docstrings de todos os `overlay/*.py`.
> Convenção: "**(repo)**" = afirmação do repositório; "**(interpretação)**" = explicação técnica minha; "**(incerto)**" = o repositório não deixa claro.
> Números entre parênteses com arquivo:linha referem-se ao arquivo no HEAD acima.

---

## 1. Visão geral

### 1.1 O que a receita faz, ponta a ponta

A receita serve o modelo **zai-org/GLM-5.3-Flash** através de uma API compatível com OpenAI (vLLM, `/v1/chat/completions`, porta `8888`) distribuída em **2× NVIDIA DGX Spark (GB10, SM121, 128 GB de memória unificada/UMA)** com **tensor parallelism TP=2** sobre um cabo ConnectX-7 (CX7, RoCEv2). Há irmãos opcionais **TP=3** (`start-tp3.sh`, testado neste kit em 2026-09-14) e **TP=4** (`start-tp4.sh`, experimental, não testado pelo mantenedor).

Os pesos servidos são **Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw** — espelho byte-idêntico de `brandonmusic/GLM-5.3-Flash-tr3-4bpw` snapshot `5ab363a8…` — uma quantização **EXL3/TR3 uniforme K4 (4 bits por peso) somente dos experts roteados** do MoE, ~164 GiB em 120 shards (README.md:10-21). Todo o resto (atenção, experts compartilhados, MLP densos, embeddings, `lm_head`) fica "nativo" (BF16) no checkpoint (README.md:446), embora hoje o default de TP2 re-quantize em tempo de carga as projeções densas para **FP8 weight-only (Marlin)** (`GLM53_DENSE_FP8=all`, `.env.example:158`).

A pilha de software:
1. **Imagem base**: `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (vLLM fork `0.1.dev20051+g487ecf187`, CUDA 13.0, arm64) (Dockerfile:88; README.md:437,616-617).
2. **Dockerfile** que aplica "patches de build": NoPE sparse-MLA no backend `FLASHINFER_MLA_SPARSE_SM120`, registro do método de quantização `exl3`, compilação do `exllamav3_ext` (ExLlamaV3 pin `c5d9c657`, 0.0.43) para `sm_121a`, kernels adicionais E2/E3/thin-decode, instalação do InstantTensor 0.2.0 e de dezenas de overlays Python (Dockerfile:1-580).
3. **Imagem publicada** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (default em `.env.example:96`). O launcher reconstrói localmente se o "recipe stamp" divergir (start.sh:1530-1564,1640-1739).
4. **Overlays de runtime**: a cada `docker run`, ambos os ranks executam, em ordem fixa, 25 scripts de patch (lista `GLM53_OVERLAY_ORDER`, start.sh:2053-2082) montados read-only em `/opt/glm53/`.
5. **Launcher** `start.sh`: preflight → imagem → download → build opcional do par dense-h3 → validação do pack → download do drafter → sync/NFS dos pesos → scripts internos → `docker run` worker (rank 1, `--headless`) e head (rank 0 + API) → espera `/health` → warmup de shapes + canário de corretude → banner READY (start.sh:2908-2942).

Especulação: **DFlash2 k=7** (`incoai/GLM-5.3-Flash-DFlash2`, revisão `dc77ff1c…`, ~2.3 GiB BF16, licença CC BY-NC-ND 4.0), atenção de draft `FLASH_ATTN`, KV do draft `auto`/BF16, draft TP=2 (README.md:35-38,449). Rollback: `SPEC_METHOD=mtp` (MTP nativo k=2).

KV cache: `--kv-cache-dtype fp8` → layout empacotado **`fp8_ds_mla`** (656 B/token/camada) nas camadas MLA; prefix caching ligado (`--enable-prefix-caching`); contexto default **850.000 tokens**; pool de KV fixado em **11 GiB** (`--kv-cache-memory-bytes 11811160064`) com páginas compactas de draft (README.md:443-444; `.env.example:183-191`).

### 1.2 O modelo (como o repositório o descreve)

(repo, compilado de `docs/improve-prefill.md` §2, `ablit/LAYER_MAP.json`, `overlay/tp3/README.md`, Dockerfile:1-83, README.md:566-587,668-680)

| Atributo | Valor | Fonte |
|---|---|---|
| Arquitetura HF | `glm5_next` (classes `Glm5NextModel`, `Glm5NextMTP`) | improve-prefill.md §2; README.md:499 |
| Camadas | 45 camadas + 1 bloco MTP (camada 45) | LAYER_MAP.json (`num_hidden_layers: 45`, 46 entradas) |
| Mistura de atenção | 34 camadas **KDA** (atenção linear) + 11 camadas **NoPE sparse MLA** ("deepseek_sparse_attention", GLM_NSA) nas camadas 3,7,11,…,43; MTP (45) também é DSA | LAYER_MAP.json; improve-prefill.md |
| MLP | camadas 0–2 densas (`intermediate_size` 12288); MoE a partir da camada 3 | LAYER_MAP.json; overlay/tp3/README; README.md:1481 |
| MoE | 288 experts roteados, **top-8**, 1 expert compartilhado, `moe_intermediate_size` 2048 | improve-prefill.md; README.md:1475 |
| Dimensões | `hidden_size` 4096; 64 cabeças de atenção / 64 KV; vocab 154.880 | LAYER_MAP.json; overlay/tp3/README |
| MLA | `kv_lora_rank=512`, `qk_rope_head_dim=0` (NoPE) → d_qk=512 | README.md:567-568; Dockerfile:8 |
| Indexer esparso | `index_topk=2048`, `index_kpool=4` (pools de 4 tokens comprimidos) | Dockerfile:21-55 |
| Conexões residuais | "mHC" (`hc_post`, `hc_contract`) | README.md:580-581; patch_glm_eagle3.py |
| Visão | torre de visão (imagem + vídeo), template com `<|image|>`/`<|video|>` | README.md:450; files/chat_template.jinja |
| Tamanho | "320B MoE" (comentário do launcher, start.sh:2755); ~19,5 B parâmetros ativos/token (estimativa em improve-prefill.md) | |
| `o_proj` | KDA `[4096, 8192]`, DSA `[4096, 16384]`, BF16 | LAYER_MAP.json |

**(interpretação)** KDA é, pelo nome e pelas pistas no código (`A_log`, `f_b_proj`, `g_b_proj`, `linear_attn_config`, "KDA state kernels", `FlashKDA`), a "Kimi Delta Attention": atenção linear com regra delta com gate, cujo estado recorrente de tamanho fixo é gerenciado pelo vLLM como um `MambaSpec` (estado por requisição, "align mode"). mHC é, pelo nome, uma variante de *hyper-connections* (fluxo residual expandido em múltiplas "correntes" misturadas por matrizes aprendidas); o repositório só mostra que é preciso "materializar `hc_post` e depois contrair (`hc_contract`) para 4096" antes de capturar estados ocultos auxiliares.

### 1.3 O que roda (tabela do README.md:430-451, resumida)

| Camada | Runtime |
|---|---|
| API | vLLM OpenAI no head, `:8888`, aberta por default; `VLLM_API_KEY` opcional (Bearer) |
| Executor | `mp` (multiprocessing, não Ray), `--nnodes 2`, `--tensor-parallel-size 2` |
| Head | `HEAD_IP=10.0.0.1`, container `glm53-exl3-head` |
| Worker | `WORKER_USER@WORKER_IP` (kit: `zurih@10.0.0.2`), `--headless`, container `glm53-exl3-worker` |
| Fabric | CX7 QSFP `enp1s0f1np1`/`rocep1s0f1` (head) ↔ `enp1s0f0np0`/`rocep1s0f0` (worker); NCCL da imagem (`USE_HOST_NCCL=0`) |
| Atenção | `FLASHINFER_MLA_SPARSE_SM120` (NoPE MLA com padding para 576) |
| KV | `fp8` → `fp8_ds_mla` (alvo); draft `auto`/bf16 |
| Experts | trellis + suh + svh + mcg; codebook MCG; um lançamento fundido `exllamav3_ext.exl3_moe` por camada |
| Tools/reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45` |
| Graphs | CUDA graphs ligados (`ENFORCE_EAGER=0`) |
| Visão | ligada (`LANGUAGE_MODEL_ONLY=0`) |
| Ablit | desligado (`ABLIT=0`) |

---

## 2. Inventário de parâmetros

Formato: **nome** | valores (default → alternativos) | onde | o que faz | conceito(s) | por que importa / trade-off.

Precedência de configuração (repo, start.sh:76-113; README.md:1354-1359): o launcher tira um snapshot de **todas as variáveis exportadas pelo chamador** antes de `source .env`, e reexporta-as depois — então `KEY=valor ./start.sh` vence o `.env` **inclusive um vazio explícito** (`KEY= ./start.sh`). Variáveis "de identidade" (`HF_HOME MODEL MODEL_REVISION IMAGE PORT TP NNODES`) emitem aviso quando o ambiente sobrescreve o `.env` (start.sh:99-109). Exceção importante: **`ABLIT` é forçado a `0` depois do `.env`** (start.sh:93) — `ABLIT=1` no `.env` não tem efeito no TP2; só `ABLIT=1 ./start.sh` liga (CHANGELOG 1.6.0). Muitos knobs distinguem "não definido" (usa default) de "definido vazio" (erro ou "stock") via `${VAR-default}` vs `${VAR:-default}`.

### 2.1 Nós, SSH e rede física

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `HEAD_IP` | `10.0.0.1` | .env.example:6; start.sh:159 | IP do rank 0; `--master-addr`; `VLLM_HOST_IP` do head; preflight exige que esteja atribuído localmente (start.sh:1345) | rendezvous de processo distribuído | Todos os ranks se conectam aqui (porta `MASTER_PORT`). No kit, `10.0.0.x` são aliases de loopback/gerência, não o caminho de dados. |
| `WORKER_IP` | `10.0.0.2` | .env.example:7; start.sh:160 | IP do rank 1; alvo SSH; `VLLM_HOST_IP` do worker | multi-nó | — |
| `WORKER_USER` / `WORKER_HOME` / `WORKER_SSH` | unset → `$USER`; `$HOME` ou `/home/$WORKER_USER`; `${WORKER_USER}@${WORKER_IP}` | .env.example:10-12; start.sh:162-168 | Conta e cache HF no worker | SSH sem senha (BatchMode) | Kits com contas diferentes (kit: `zurih`). |
| `HEAD_CX7_IF` / `WORKER_CX7_IF` | `enp1s0f1np1` / `enp1s0f0np0` | .env.example:20-21; start.sh:170-171 | Viram `NCCL_SOCKET_IFNAME` e `GLOO_SOCKET_IFNAME` por rank | interface de bootstrap NCCL/gloo | "NCCL cannot use the 10.0.0.x loopback aliases — without these pins ncclCommInitRank hangs" (.env.example:14-15). Exportar `NCCL_SOCKET_IFNAME` genérico **não** sobrescreve (README.md:1707-1710). |
| `HEAD_CX7_IB` / `WORKER_CX7_IB` | `rocep1s0f1` / `rocep1s0f0`; lista separada por vírgula para dual-rail | .env.example:22-24; start.sh:172-173 | `NCCL_IB_HCA` por rank | RDMA/HCA, RoCEv2 | Define qual porta física carrega o all-reduce. |
| `NCCL_IB_GID_INDEX` | `3` (template); alternativos por kit | .env.example:33,355; start.sh:178 | Índice da tabela GID RoCE usado pelo NCCL | GID RoCEv2 (`::ffff:<ip>`) | "An all-zero entry kills that rank ~60 s in (ibv_modify_qp errno 61)" (.env.example:25-27). Preflight valida em **cada HCA listada** (start.sh:1356-1391) e imprime as tabelas 0–7 com tipo (RoCE v1/v2). |
| `HEAD_GID` / `WORKER_GID` | default = `NCCL_IB_GID_INDEX` | .env.example:34-35; start.sh:182-183 | GID por rank | GID é por NIC | "this kit needs head=4, worker=3" (start.sh:180-181). |
| `TP` / `NNODES` | `2` / `2` | .env.example:100-101; start.sh:198-199 | `--tensor-parallel-size`, `--nnodes` | tensor parallelism | "do not change for this recipe" (README.md:1752); TP=3 é `start-tp3.sh`, não `TP=3` no `.env` (README.md:2003). |
| `MASTER_PORT` | `29521` | .env.example:102; start.sh:201 | `--master-port` | rendezvous torch.distributed | Checado livre no preflight (start.sh:1405). |
| `PORT` | `8888` | .env.example:99; start.sh:200 | Porta HTTP da API | — | Checada livre (start.sh:1404); mova se ocupada. |
| `NCCL_NCHANNELS` | vazio → NCCL decide; inteiro positivo | .env.example:361-363; start.sh:177,2463-2470 | Fixa `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS` em ambos os ranks | canais NCCL (paralelismo de anéis) | Pinning ("Entrpi 2× Spark: 8"). Em TP3, se vazio, ainda fixa MIN=MAX=8 (start-tp3.sh:265-302) para limitar memória pinned do host. |
| `NCCL_CROSS_NIC` | `0` (TP2); `1` (TP3/TP4) | start.sh:189; .env.tp3.example:35; .env.tp4.example:29 | Permite ao NCCL parear NICs de índices diferentes entre nós | topologia de anéis | Anéis de 3+ nós com duas portas por nó precisam de cross-NIC. |
| `USE_HOST_NCCL` / `NCCL_HOST_DIR` / `WORKER_NCCL_HOST_DIR` / `NCCL_SO_NAME` | `0` / `~/nccl-2.30.7` / `libnccl.so.2.30.7` | .env.example:351-354; start.sh:190-196,2502-2516 | Se `1`, `LD_PRELOAD` de um NCCL do host | ABI/duplicidade de bibliotecas | "LD_PRELOAD of a second SO makes DeepEP assert duplicate NCCL" — manter `0`. |
| `NCCL_DEBUG` | `WARN` | .env.example:360; start.sh:174 | Verbosidade NCCL | — | `INFO` para diagnosticar. |
| `CONTAINER_HEAD` / `CONTAINER_WORKER` | `glm53-exl3-head` / `glm53-exl3-worker` | start.sh:510-511 | Nomes dos containers | — | TP3/TP4 usam `glm53-exl3-tp3-*`/`-tp4-*` para não reutilizar a pilha TP2. |

#### 2.1.1 Somente TP=3 (`start-tp3.sh`, `.env.tp3.example`)

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `WORKER2_IP`, `WORKER2_USER`, `WORKER2_HOME`, `WORKER2_SSH`, `WORKER2_CX7_IF/IB`, `WORKER2_GID` | `10.0.0.3`; herdam do rank 1 | .env.tp3.example:9-47; start-tp3.sh:232-274 | Rank 2 | Terceiro nó. |
| `SOCKET_IFNAME` (+ `HEAD_/WORKER_/WORKER2_SOCKET_IFNAME`) | vazio → usa os pins CX7; kit: LAN de gerência `enP7s7` | start-tp3.sh:243-253; README.md:1501 | Interface de controle (gloo + bootstrap NCCL) | "A 3-node ring cannot [use the CX7 IF]: every pair has its own /24… gloo takes exactly ONE interface name"; dados continuam em RoCE via `NCCL_IB_HCA`. |
| `HEAD_HOST_IP` / `WORKER_HOST_IP` / `WORKER2_HOST_IP` | vazio → IP 10.0.0.x do rank | start-tp3.sh:254-258 | Endereço anunciado ao process group | Deve estar na interface de socket. |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | start-tp3.sh:281-287 | Pareia NICs por sub-rede, não por índice | "a 3-cycle with two ports per node has no consistent labelling" → `ibv_modify_qp … 110 Connection timed out` sem isso. |
| `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` | `1` / `1` | start-tp3.sh:293-294 | Força transporte IB | "keeps NCCL on IB instead of guessing a local transport" (copiado da receita DS4.1). |
| `NCCL_BUFFSIZE` / `NCCL_LL128_BUFFSIZE` / `NCCL_PROTO` | `1048576` / `262144` / `^LL128` | start-tp3.sh:295-297 | Tamanho de buffers e protocolos | "keep pinned host memory sane on a GB10 (512 connections x 9 MiB otherwise)". |
| `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS` | `8` / = MAX | start-tp3.sh:298-302 | Canais fixos | Idem. |
| Cabeamento | "directed ring": Port0 de cada nó → Port1 do próximo | .env.tp3.example:23-31; README.md:1509-1514 | Topologia física | "a mirrored ring leaves one pair that can never connect"; reordenar `NCCL_IB_HCA` não ajuda. Uma /24 por cabo, MTU 9000, `ipv4.never-default yes`. |

#### 2.1.2 Somente TP=4 (`start-tp4.sh`, `.env.tp4.example`)

| Nome | Valores | Onde | O que faz |
|---|---|---|---|
| `WORKER3_*` | `10.0.0.4` etc. | .env.tp4.example:12-43; start-tp4.sh:190-225 | Rank 3 |
| `NCCL_CROSS_NIC` | `1` | start-tp4.sh:231 | — |
| Kit medido | 4 Sparks (2× 200G + 2× 100G) num switch CRS812 | README.md:1565-1567 | — |

### 2.2 Ambiente fixo dos containers (`nccl_common`, start.sh:2417-2462)

| Variável | Valor | O que faz / por quê |
|---|---|---|
| `NCCL_IB_DISABLE` | `0` | Usa verbs IB/RoCE. |
| `NCCL_IB_ROCE_VERSION_NUM` | `2` | RoCEv2 (UDP/IP roteável). |
| `NCCL_NET` | `IB` | Força o transporte de rede IB. |
| `NCCL_NET_PLUGIN` | `none` | Sem plugin externo de rede. |
| `NCCL_NVLS_ENABLE` | `0` | Sem NVLink SHARP (não existe no GB10). |
| `NCCL_CUMEM_ENABLE` | `0` | Desliga alocação cuMem do NCCL (interpretação: evita alocações VMM problemáticas em UMA). |
| `NCCL_IB_MERGE_NICS` | `0` | Não funde portas numa NIC lógica. |
| `NCCL_CROSS_NIC` | `$NCCL_CROSS_NIC` | ver 2.1. |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | Ignora afinidade de CPU. |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` / `1` | Container nunca baixa nada; permite montar o cache HF read-only via NFS. |
| `HF_HOME` / `VLLM_CACHE_ROOT` | `/root/.cache/huggingface` / `/root/.cache/vllm` | Caminhos no container. |
| `TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR` | `/root/.triton/cache` / `/root/.tilelang/cache` | Caches JIT persistidos no host (ver 2.17). |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` (stock 300) | start.sh:479-481: "mid-serve Triton/TileLang JIT on TP=2 can exceed that without being a true hang. NCCL watchdog is still 600s". |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | Compila/JIT para `sm_121a` (GB10). |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | — |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` quando não definido; vazio explícito desliga | start.sh:2453-2455; README.md:1770 | Alocador de cache do PyTorch com segmentos expansíveis (menos fragmentação). "the hidden-state KV connector (training windows) refuses expandable_segments". |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `$READY_TIMEOUT` (3600) | Timeout de prontidão do engine. |
| `VLLM_NO_USAGE_STATS` / `DO_NOT_TRACK` | `1` | "py-cpuinfo JSON-parses empty output on Grace/aarch64". |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `$CG_ESTIMATE` | ver 2.8. |
| `GLM53_*` de política | (vários) | Encaminhados idênticos a todos os ranks: "a rank that captures different CUDA graphs hangs NCCL after PIECEWISE" (README.md:1486-1487). |
| Por rank | `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `VLLM_HOST_IP` | start.sh:2636-2640,2686-2690 |
| TP3 extra | `PYTHONFAULTHANDLER=1`, `HAREM_KDA_FLASHKDA`, NCCL de anel | start-tp3.sh:2055-2103 |
| TP4 extra | `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS` | start-tp4.sh:1671-1710 |

### 2.3 Flags do `docker run` (start.sh:2595-2642, 2645-2747)

| Flag | Valor | Por quê |
|---|---|---|
| `--gpus all` | — | GPU GB10. |
| `--network host` | — | NCCL/gloo/RoCE e API sem NAT. |
| `--ipc=host` + `--shm-size 32g` | — | Memória compartilhada para broadcast entre processos do vLLM (shm_broadcast). |
| `--device /dev/infiniband` + `--cap-add IPC_LOCK` + `--ulimit memlock=-1` | — | RDMA precisa travar (pin) memória e acessar os devices verbs. |
| `--ulimit stack=67108864` | 64 MiB | Stack grande. |
| `--stop-timeout 60` | — | Mas o stop faz `docker kill` primeiro (start.sh:2949-2964). |
| Montagens | HF cache (ou volume NFS `:ro`), `CACHE_ROOT→/root/.cache/vllm`, Triton, TileLang, `~/.nv/ComputeCache`, script interno em `/start.sh`, template, cada overlay em `/opt/glm53/*.py:ro`, `ablit/` | Os overlays são **bind-mounted e aplicados a cada start** — alterar um patch não exige rebuild da imagem. |
| `--entrypoint bash IMAGE /start.sh` | — | O script interno monta `ARGS`, aplica overlays e faz `exec vllm serve`. |
| `VLLM_API_KEY` | só no head (`-e VLLM_API_KEY` sem valor na linha de comando) | start.sh:2591-2592,2645,2745 — a chave não vai para argv nem para o worker. |

### 2.4 Imagem e build

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`; alt: `:exl3` (sem InstantTensor), `glm53-flash-sm121:local` | .env.example:92-96; start.sh:155 | Imagem de serving | `git pull` não troca a imagem rodando; kits antigos precisam puxar a tag nova (README.md:1267-1342). Não usar `glm53-flash-sm121:v8` (NVFP4/Ray antigo) (README.md:1563). |
| `BUILD` | `0`; `1` força rebuild | start.sh:56,1663-1675 | `docker build` desta Dockerfile | — |
| `SKIP_BUILD` | `0`; `1` mantém GHCR mesmo com stamp divergente | start.sh:1663-1670 | — | Sem ele, divergência de stamp → rebuild automático. |
| `SKIP_PULL` / `PULL` | `0` / `0` | start.sh:1656-1657 | Pula/força `docker pull` | — |
| `SKIP_SHIP` | `0` | start.sh:1700-1728 | Não copia a imagem ao worker | Normal: worker faz `docker pull`; se falhar, `docker save --platform linux/arm64 | ssh docker load` (issue #8). |
| `IMAGE_PLATFORM` | vazio → inspect → `linux/arm64` | start.sh:1520-1528 | Plataforma do `docker save` | Índices OCI multi-arch quebram `docker load` sem `--platform`. |
| `SKIP_OVERLAY_VERIFY` | `0` | start.sh:1729-1737 | Roda `test_exl3_overlay.py` com `EXL3_SELFCHECK_GPU=1` na GPU antes de lançar | Self-check real de cubins `sm_121a`/GEMM fundido. |
| `GHCR_TOKEN` / `GHCR_USER` | unset / `MiaAI-Lab` | .env.example:97-98; start.sh:157,1477-1487 | Login opcional | Só por rate-limit anônimo. |
| `GLM53_RECIPE_STAMP` (label `glm53.recipe.stamp`) | SHA-256 de `Dockerfile` + `overlay/` + `files/` + `tests/` + `ablit/` (exclui transplant, nfs-server, nfs-share.sh) | start.sh:1532-1555; Dockerfile:577-580 | Detecta que o código mudou desde a imagem | Pull de GHCR só é adotado se o stamp publicado também bater; senão restaura a tag local (start.sh:1597-1638). |
| Image key | hash de `RootFS.Layers` | start.sh:1489-1518 | Identidade da imagem entre head/worker | `.Id` difere entre overlay2 e containerd; `docker save|load` perde RepoDigests. |
| `ARG BASE` | `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` | Dockerfile:88 | Base | Pin por digest. |
| `EXLLAMAV3_COMMIT` | `c5d9c657966ffeeaa9353f0cc899f18629da4a13` (0.0.43) | Dockerfile:380; exl3.py | Fonte do `exllamav3_ext` | Expõe `exl3_moe` / `exl3_moe_max_concurrency` (README.md:453-455). |
| `MAX_JOBS` | `8` (build principal); `2` (Dockerfile.e3-layer) | Dockerfile:383; Dockerfile.e3-layer:22 | Paralelismo de compilação | "keep the compile light on a UMA host". |
| `CUDA_HOME`/`CPATH` | `/usr/local/cuda`; headers em `nvidia/cu13/include` | Dockerfile:384-431 | Toolchain | Imagem guarda headers CUDA sob `nvidia/cu13`. |
| `instanttensor==0.2.0` | `--no-deps` | Dockerfile:561-564 | Loader direct-I/O | "a normal pip install pulls nvidia-nccl-cu13==2.29.7 over this image's NCCL 2.30.7". |
| `EXL3_SELFCHECK_GPU` | `0` no build; `1` no launch | Dockerfile:547; start.sh:1732 | Testes do overlay sem/com GPU | — |

### 2.5 Pesos, download, sincronização e NFS

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `MODEL` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | .env.example:42; start.sh:116 | Repo do alvo | Espelho durável. |
| `MODEL_FALLBACK` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | .env.example:43; start.sh:118 | Usado se o espelho 404 ou tiver < 120 shards | — |
| `MODEL_CACHE_NAME` / `MODEL_FALLBACK_CACHE_NAME` | `models--<org>--<repo>` | .env.example:44-46; start.sh:119-120 | Pasta no cache HF | Evita baixar 164 GiB de novo se já tem o upstream. |
| `MODEL_REVISION` | `25a44fdbf16862a46b7cc9921142c6c81350af2f` | .env.example:48; start.sh:122 | Commit no espelho | Pin de pesos. |
| `EXPECTED_SHARDS` | `120` | start.sh:543-546 | Contagem de shards para considerar completo | Preset pinado impõe o seu próprio valor. |
| `HF_HOME` / `HF_TOKEN` / `HF_BIN` | `~/.cache/huggingface` / opcional / opcional | .env.example:49-55; start.sh:513,1769-1785 | Cache e CLI | Fallback para `python3 -m huggingface_hub.commands.huggingface_cli`. |
| `HF_DOWNLOAD_EXCLUDE` | `runtime-results/**,src/**,runtime/src/**,scripts/**,docs/**,results/**,.materialization/**,runtime/scripts/**` | start.sh:1809 | Padrões excluídos do download | Não traz lixo do repo HF. |
| `SKIP_DOWNLOAD` / `REFRESH_WEIGHTS` | `0` / `0` | start.sh:1799-1802 | Pula verificação / força re-fetch | — |
| `SKIP_SYNC` / `FORCE_SYNC` | `0` / `0` | .env.example:90-91; start.sh:1976,2024 | Pula rsync / ignora marcador | Marcador `.glm53-exl3-synced` por revisão; o worker é verificado shard a shard por tamanho desreferenciado (start.sh:1911-2006; CHANGELOG #256). |
| `NFS_SHARE` | `0` (template); kit `1` | .env.example:67; start.sh:555 | `1` = worker monta o cache HF do head por NFSv4 read-only | Evita segunda cópia de 164 GiB; em TP3 evita ~330 GiB de duplicata (.env.tp3.example:132-145). |
| `NFS_VOLUME` / `NFS_CONTAINER` / `NFS_IMAGE` | `glm53-weights` / `glm53-nfs` / `glm53-nfs:local` | files/nfs-share.sh:18-20 | Volume docker/container exportador | Reutiliza exportador vivo (`vllm-fn-nfs`, `glm53fp8-nfs`, `dsv41-nfs`) e **faz união** da lista de clientes. |
| `NFS_CLIENTS` | `10.0.0.2,10.0.0.3,10.0.22.0/24,10.0.23.0/24,10.0.33.0/24` | files/nfs-share.sh:55-57 | ACL de export | Incluir sub-rede de fabric. |
| `NFS_SERVER_IP_<rank>` / `WORKER_FABRIC_IP` | autodetect por `ip route get`; kit TP3 `10.0.22.1`/`10.0.23.1` | files/nfs-share.sh:68-87,149-155 | Endereço CX7 do head que o rank monta | `10.0.0.*`/`127.*` são recusados: montar pela rede de gerência "turns a 164 GiB load into an overnight job". |
| `NFS_OPTS` | `ro,sync,no_subtree_check,no_root_squash,insecure,fsid=0` | files/nfs-share.sh:22 | Export | — |
| `NFS_OPTS_CLIENT` | `nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600` | files/nfs-share.sh:23 | Mount | "nconnect=8 is what makes a cold load use the cable" (.env.tp3.example:163). |
| `NFS_DEVICE` | `:/` | files/nfs-share.sh:24 | Raiz do export = `$HF_CACHE_DIR` | Worker vê `hub/<repo>/snapshots/<rev>` no mesmo caminho. |
| Servidor NFS | Alpine + `nfs-utils`, privilegiado, host network; `rpc.nfsd 8`; `rpc.mountd --no-nfs-version 2 --no-nfs-version 3` | files/nfs-server/* | Kernel nfsd em container | — |
| `GLM53_MODEL_PRESET` | vazio; `abliterated`; `dense-h3` | start.sh:130-154 | Presets pinados de checkpoint | ver 2.12 e 2.16. |
| Preset `abliterated` | `bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit@14858211ed81d7fa773f8a0db02f38f36d230252`, 120 shards (175.642.157.752 bytes) | start.sh:133-144; start-abliterated.sh | Serve checkpoint pré-editado; força `ABLIT=0`; fallback = o mesmo repo; fail-closed | Aplicar o edit de runtime de novo "would produce a different model" (README.md:529-533). |
| Preset `dense-h3` (pins) | TR3 config SHA-256 `4f5341e0…`, index `2f64d21c…`; `turboderp/GLM-5.3-Flash-exl3` branch `4.05bpw` rev `2a30229e…`; overlay SHA-256 `1a1b0793…`; draft fonte `incoai/…DFlash2@dc77ff1c…`; draft construído `local/GLM-5.3-Flash-DFlash2-EXL3-6bpw@27d19286…`; ref `glm53-dense-h3` | start.sh:1126-1139 | Constrói no head o par alvo-denso-EXL3 + draft 6 bpw | Builds byte-reproduzíveis; build divergente é recusado. |

### 2.6 Carregamento de pesos (loader) e memória do host

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `LOAD_FORMAT` | `instanttensor` quando `IMAGE` contém "instanttensor"; vazio = vLLM auto | .env.example:364-370; start.sh:278-283 | `--load-format` | loader direct-I/O (O_DIRECT) vs mmap/safetensors | Medido (#204): 164 GiB carregam em ~65–70 s a frio (<10 s com page cache) vs ~290–300 s no auto; **custo**: ~4,4–5,6 GiB a menos de KV (12,4–12,5 vs 17–18 GiB a 850k/0,85) — por isso o pool fixo em `EXTRA_ARGS` (README.md:1231-1265). |
| `GLM53_LOAD_CLONE` | `1`; exatamente `0`/`1` | .env.example:373; start.sh:286 | Stage de tensores mmap auto/lazy para storage de CPU independente | cópia fora do mmap | `0` não desliga a segurança de páginas não-4KiB (README.md:1190-1229). |
| `GLM53_LOAD_PREFETCH` | `0`; `0..16` | .env.example:377; start.sh:287 | Read-ahead de shards locais (inclui o atual) | prefetch/page cache | "The window is not a bound on host RAM or OS page-cache residency"; `POSIX_FADV_DONTNEED` após consumir cada shard. NFS/Lustre mantêm prefetch stock. |
| `GLM53_HOST_MEM_HYGIENE` | `1` | .env.example:399; start.sh:491,2299-2350 | Antes do `docker run`: `sync; echo 1 > /proc/sys/vm/drop_caches` em ambos os nós (via `sudo -n`) e espera `MemAvailable ≥ GPU_MEM_UTIL × MemTotal + 1,5 GiB` (até 45 tentativas de 2 s) | UMA: page cache = memória "usada" pela GPU | Após rsync de 164 GiB, `MemFree`≈2 GiB e o InstantTensor abortava (`buffer_size … exceeds device memory budget`) ou reduzia `io_depth` para dezenas (docs/cold-load-uma.md). |
| `GLM53_COLD_LOAD_UMA` / `GLM53_COLD_LOAD_STAGE_MMAP` | unset (on); `0` = kill switch | start.sh:2492-2500; cold-load-uma.md | Orçamento do InstantTensor por `MemAvailable`; clone de mmap em kernels de página 64 KiB | `cuMemcpyHtoDAsync` "wedges on file-backed 64 KiB mappings" | Encaminhados só se definidos (vazio exportado ligaria o kill switch). |
| `INSTANTTENSOR_MAX_FREE_MEM_USAGE`, `_BUFFER_SIZE`, `_CHUNK_SIZE`, `_CONCURRENCY`, `_IO_DEPTH`, `_BACKEND`, (`_CACHE_BUFFER` TP3) | unset; exemplo TP3: `BUFFER_SIZE=536870912`, `CACHE_BUFFER=0` | start.sh:2492-2500; start-tp3.sh:2153-2161; examples/tp3-throughput.env:20-21 | Knobs do InstantTensor | chunk 8 MiB × io_depth 256 foi o melhor do sweep (cold-load-uma.md) | Um `MAX_FREE_MEM_USAGE` sozinho que não cubra o buffer de 4 GiB é elevado com aviso (#273). |
| `GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB` | `2097152` (2 GiB) | start.sh:1288 | Folga exigida além de `GPU_MEM_UTIL×total` no preflight | — | `PREFLIGHT FAIL` antes de parar containers. |
| Nota InstantTensor KV | aviso se `LOAD_FORMAT=instanttensor`, `MAX_MODEL_LEN≥850000`, `GPU_MEM_UTIL≤0,85` e sem `--kv-cache-memory-bytes` | start.sh:1313-1332 | Diagnóstico | — | Só avisa; não muda nada. |

### 2.7 Argumentos do `vllm serve` (script interno, start.sh:2093-2170 head; 2173-2249 worker)

Fixos em ambos os ranks:

| Flag | Valor | Conceito | Por que |
|---|---|---|---|
| `--served-model-name` | `GLM-5.3-Flash-EXL3` (`SERVED_MODEL_NAME`) | id OpenAI `model` | — |
| `--host 0.0.0.0 --port $PORT` | — | — | — |
| `--tensor-parallel-size $TP --nnodes $NNODES --node-rank {0,1} --master-addr $HEAD_IP --master-port $MASTER_PORT` | — | TP multi-nó do vLLM | Cada nó roda 1 processo de GPU. |
| `--distributed-executor-backend mp` | — | executor multiprocessing (não Ray) | Simplicidade; um processo por nó. |
| `--headless` (só worker) | — | worker sem API | Rank 1 só executa o modelo. |
| `--tool-call-parser glm47 --enable-auto-tool-choice` | — | parsing de tool calls no formato XML do GLM (`<tool_call>name<arg_key>…`) | Necessário para agentes/ferramentas. |
| `--reasoning-parser glm45` | — | separa `<think>…</think>` em `reasoning` | Também mantém vivo o hook de `tool_choice:none` (docs/tool-choice-none.md). |
| `--enable-prefix-caching` | — | APC (Automatic Prefix Caching) | Conversas stateless reenviam o histórico. |
| `--no-enable-flashinfer-autotune` | — | autotune do FlashInfer | "FlashInfer's sparse-MLA autotune and fused_moe gemm1/gemm2 autotune kill rank 0 on SM121" (Dockerfile:81-83); full autotune "OOMs this UMA" (improve-prefill.md). |

Condicionais:

| Flag | Fonte | Default efetivo TP2 | Notas |
|---|---|---|---|
| `--enforce-eager` | `ENFORCE_EAGER=1` | não (graphs ligados) | Script interno usa `${ENFORCE_EAGER:-1}`, mas o launcher sempre passa `0`. |
| `--quantization exl3` | `QUANTIZATION` | `exl3` | "never marlin" (README.md:1753). |
| `--max-model-len` | `MAX_MODEL_LEN` | `850000` | ver 2.8. |
| `--gpu-memory-utilization` | `GPU_MEM_UTIL` | `0.85` | ver 2.8. |
| `--max-num-seqs` | `MAX_NUM_SEQS` | `4` | ver 2.9. |
| `--max-num-batched-tokens` | `MAX_NUM_BATCHED_TOKENS` | `7168` | ver 2.9. |
| `--long-prefill-token-threshold` | `LONG_PREFILL_TOKEN_THRESHOLD` | omitido (vazio) | TP3 script default `3584`, mas o `.env` compartilhado define vazio explicitamente (.env.example:208), o que **(interpretação)** suprime o default `${…-3584}` do TP3. |
| `--kv-cache-dtype` | `KV_CACHE_DTYPE` | `fp8` | → `fp8_ds_mla`. |
| `--load-format` | `LOAD_FORMAT` | `instanttensor` | — |
| `--prefix-match-unit` | `PREFIX_MATCH_UNIT` | omitido | Grão de hash de prefixo. |
| `--default-chat-template-kwargs '{"reasoning_effort":"…"}'` | `GLM53_DEFAULT_REASONING_EFFORT` | omitido | `low`/`high`/`max`. |
| `--speculative-config` | `SPEC_METHOD` | DFlash2 JSON (abaixo) | MTP: `{"method":"mtp","num_speculative_tokens":2}`; `none`: nada. |
| `--chat-template /opt/glm53/chat_template.jinja` | `CHAT_TEMPLATE` | sim | "checkpoint jinja is language-only" (README.md:1646-1648). |
| `--language-model-only` | `LANGUAGE_MODEL_ONLY=1` | não | Desliga torre de visão. |
| `--limit-mm-per-prompt` | `LIMIT_MM` | `{"image":48,"video":1}` | — |
| `--mm-processor-kwargs '{"max_image_tokens":N}'` | `MM_IMAGE_TOKENS` | `2048` | — |
| `--media-io-kwargs '{"video":{"num_frames":N}}'` | `VIDEO_NUM_FRAMES` | omitido (vLLM 32) | — |
| `--mm-processor-cache-gb` | `MM_PROCESSOR_CACHE_GB` | `1` | — |
| `--skip-mm-profiling` | `SKIP_MM_PROFILING=1` | sim | — |
| `EXTRA_ARGS` (split por espaço) | `.env` | `--kv-cache-memory-bytes 11811160064` + `--cudagraph-capture-sizes …` (auto) | ver 2.8/2.10. |
| TP3: `--hf-overrides` | `TP3_HEAD_OVERRIDE=66` | `{"num_attention_heads":66,"num_key_value_heads":66,"linear_num_heads":66,"text_config":{…mesmo…}}` (start-tp3.sh:1629-1634) | 64 não divide por 3. |
| TP3: `--enable-expert-parallel` | `ENABLE_EXPERT_PARALLEL=1` | sim | 288/3 = 96 experts por rank. |
| TP3: `--mm-encoder-tp-mode data` | `MM_ENCODER_TP_MODE` | `data` | Torre de visão data-parallel. |
| Outros vistos em perfis | `--enable-prompt-tokens-details` (examples/tp2-long-coding.env:47), `--profiler-config {...torch...}` (scripts/boot_candidate.sh:9), `--cudagraph-capture-sizes` explícitos | — | — |

JSON do `--speculative-config` DFlash2 (start.sh:2128-2134):
```json
{"method":"dflash","model":"$DFLASH_MODEL_DIR","num_speculative_tokens":7,
 "kv_cache_dtype":"auto","draft_sample_method":"probabilistic",
 "rejection_sample_method":"standard","draft_tensor_parallel_size":2}
```

### 2.8 Memória, KV cache e contexto

| Nome | Valores | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `MAX_MODEL_LEN` | `850000` (TP2 desde 2026-09-07; era 1M); TP3 `1000000`; TP4 script `1000000` (mas herda 850000 do `.env` se não sobrescrito); long-coding `262144`; validado 1..1.000.000 | .env.example:196; start.sh:219-222,742; .env.tp3.example:173 | Contexto máximo por requisição | janela de contexto; RoPE-less (NoPE) MLA | "At MNBT 7168 one 1M request needs 14.52 GiB KV (10.98 GiB at 500k; ~7.4 GiB fixed + 7.1 GiB per 1M)" (README.md:1768). O E3 "keeps a 560 MiB fat-row scratch that vLLM's profile run charges to the KV budget, so at 1M / util 0.87 the pool no longer fits one 1M request" (README.md:289-293). **Não** reduzir para 256k para "liberar slots": "logged tokens ≈ concurrency × that cap; hybrid block-id overhead then shrinks the pool". |
| `GPU_MEM_UTIL` | `0.85` (TP2; era 0.87); TP3 `.env.tp3` `0.80` (script default 0.87); TP4 `.env.tp4` `0.75` (script 0.87); long-coding `0.865` | .env.example:209; start.sh:223-225; .env.tp3.example:172; .env.tp4.example:115 | `--gpu-memory-utilization`: fração da memória "da GPU" (em UMA = RAM do sistema) que o vLLM pode usar | UMA / orçamento de memória | "each 0.01 is 1.2 GiB of host headroom" (README.md:1769); "a 256k prefill at 0.87 with zero MemAvailable crashed a head on 2026-09-06". 0.87 exige ≥105,9 GiB livres após o init de ~9 GiB do vLLM (README.md:1718-1721). |
| `--kv-cache-memory-bytes` (via `EXTRA_ARGS`) | TP2 atual `11811160064` (11 GiB); antes `15032385536` (14 GiB); `16106127360` (15 GiB) medido; TP3 kit `42949672960` (40 GiB; README) / 35 GiB (adoption doc e CHANGELOG 1.4.0 — divergência); tp3-throughput `34359738368` (32 GiB); long-coding: removido | .env.example:180-191; README.md:1499 | Tamanho explícito do pool de KV, **ignora `GPU_MEM_UTIL` para o KV** | pool de blocos KV; profiling de memória | Com InstantTensor o pool perfilado a 850k/0,85 (~12,4 GiB) fica abaixo dos 13,56 GiB que uma requisição de 850k exige → boot recusa. 11 GiB com páginas compactas de draft = 1.233.779 tokens (1,45× de 850k); sem compact 11 GiB ≈ 690k e o boot recusa 850k → use 14 GiB. 15 GiB deixou só 0,8–2,2 GiB livres sob carga. `start-tp3.sh`/`start-tp4.sh` **removem** esse token do `.env` compartilhado (start-tp3.sh:132-156) — fixe um valor por topologia em `.env.tp3`/`.env.tp4`. |
| `KV_CACHE_DTYPE` | `fp8` (obrigatório); não `nvfp4`, não bf16 | .env.example:210; start.sh:274 | Dtype do KV | FP8 KV; layout `fp8_ds_mla` | "The SM12x sparse-MLA kernel only accepts packed fp8_ds_mla. bf16 KV has no sparse kernel on this arch" (README.md:670-673). "NVFP4 KV is not available here. FlashInfer's SM12x NVFP4 kernels are dense MHA" (README.md:751-752). |
| Layout `fp8_ds_mla` | 656 B/token/camada = 512 NoPE (fp8) + 16 B escalas + 128 B RoPE (64 bf16) | README.md:569-571; Dockerfile:12-29 | Registro KV empacotado | MLA latente; quantização FP8 com escalas | O overlay faz zero-padding do latente de 512 para 576 (geometria GLM_NSA): "A zero RoPE block adds nothing to the QK dot"; custa ~24% a mais de KV DSA (656 vs ~528 B). |
| KV do draft | `auto` (BF16), 2048 B/token/camada no boot de TP2 | README.md:680,693 | Dtype KV do drafter | — | "dense DFlash2 cannot use the target's fp8_ds_mla layout and SM121 has no FA3/FA4 for plain FP8" (README.md:585-587). |
| `CG_ESTIMATE` | `1` (upstream); `0` | .env.example:356-359; start.sh:184-188 | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | reserva de memória para CUDA graphs | "estimate is 2.43 GiB while the captured graphs actually consume -0.19 GiB, so ~2.6 GiB of KV is reserved and never used. 0 keeps CUDA graphs ON and drops only the deduction". Com `0`, `patch_skip_cudagraph_profile.py` pula o dry-capture (KV profile 18→7 s). |
| `GLM53_INDEXER_WORKSPACE` | `rightsize` (TP2 desde 2026-09-07); `stock`; TP3/TP4 scripts default `stock` (mas herdam `rightsize` do `.env`) | .env.example:329-333; start.sh:459-465; start-tp3.sh:500 | Tamanho do workspace de gather do indexer esparso no prefill | workspace pré-alocado, descontado do KV | `stock` = `max_model_len × 40` entradas × 132 B = **5036,40 MB travados a 1M** (medido); `rightsize` = `min(MAX_NUM_SEQS, MNBT) × cdiv(MAX_MODEL_LEN + k, index_kpool)` = 126 MB a seqs=4 / 504 MB a 16 → "~+26–28% KV" (README.md:1788; DESIGN-indexer-workspace.md). |
| `GLM53_DRAFT_KV_COMPACT` | `1` quando `SPEC_METHOD=dflash` e não definido; `0` caso contrário; vazio explícito = erro | .env.example:334-342; start.sh:466-475 | Deriva bloco de draft = maior múltiplo de 64 que divide o bloco MLA e cabe na página física | páginas KV com padding / "slot-share" | TP2: bloco MLA 3584 × 656 B = página 2.351.104 B; draft 2048 B/token → bloco 896 (896×2048 ≤ página). TP3: MLA 2560 → draft 640 (1536 B/token). Reduz IDs de bloco reservados por requisição de máxima duração 180→121 (−32,78%); não reduz VRAM. Troca lookup APC do drafter para terminar exatamente na fronteira. |
| Bloco/página | `block_size=64` (kernel MLA); manager MLA 3584; `mamba_block_size=16`; scheduler block = LCM = 3584; KpoolTail bloco 4 | README.md:691,1880-1885 | Grão de alocação | PagedAttention/BlockPool; hybrid KV | Hits de APC são alinhados a 3584 tokens (7168/10752/14336 = 2/3/4 páginas). |
| `GLM53_KV_CAPACITY_LOG` | `1`; `0` | .env.example:310-315; start.sh:417 | Log por grupo KV + resumo de capacidade | "GPU KV cache size" = `max_concurrency × max_model_len`, não o tamanho do cache de prefixo | Ex.: 643 IDs de bloco, 38 IDs por segmento de 3584 → ~57K tokens de conversa em cache (README.md:1858-1908). |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | start.sh:2455 | ver 2.2 | — | — |
| `SKIP_MM_PROFILING` | `1` | .env.example:213-214 | Pula perfil dummy de imagem+vídeo de tamanho máximo | profiling de memória MM | "a max-size image+video dummy profile OOMs this UMA"; consequência: **nada é reservado para a torre de visão** (README.md:726-729). |

### 2.9 Scheduler, batching e política de prefill

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `MAX_NUM_SEQS` | `4` (TP2/TP3); TP4 `8`; long-coding `2`; tp3-throughput `8`; ladder `16` | .env.example:197; .env.tp4.example:116 | Requisições em voo simultâneas | continuous batching | "four in-flight generations, not four parked sessions" (README.md:1034-1035). TP4: 4 → 84 tok/s agregado a 8 streams e 27 s de TTFT pior; 8 → 135 tok/s e 1,0 s. |
| `MAX_NUM_BATCHED_TOKENS` (MNBT) | `7168` (default); `2048` (PR77 A/B, TP4, geometria seqs=16); `1024` (long-coding); `4096` (tp3-throughput); **nunca 8192** | .env.example:198-202; start.sh:227-229 | Orçamento de tokens por passo (chunked prefill) | chunked prefill; encoder cache size | "8192 chunk × long history oversubscribes GB10 persistent_topk smem (300k crash)"; MNBT também é o tamanho do cache do encoder (por isso `MM_IMAGE_TOKENS ≤ MNBT`). |
| `LONG_PREFILL_TOKEN_THRESHOLD` | vazio (omite flag); positivo ≤ MNBT; TP3 script `3584`; tp3-throughput `2048` | .env.example:203-208; start.sh:244; start-tp3.sh:339-341 | Limita quantos tokens um prefill longo consome por passo | fairness do chunked prefill | A/B/A a MNBT 7168 com 3584: "warm-session freeze 131/108 s stock -> 15 s; contended prefill +6%; solo prefill +3..+11%". |
| `GLM53_MIXED_PREFILL_CHUNK` | `fair` (default TP2/3/4 desde 2026-09-15); `skip`/`-1`; `N>0`; `0`/`off`/`no` | .env.example:290-298; start.sh:393-400 | Política de misturar prefill de uma requisição nova nos passos em que outra está decodificando | decode/prefill interference; step budget | `skip` "starves prefills until decode ends"; `0/off` "collapses the incumbent 10–36× on TP=2"; `fair` v5: decodes primeiro, cobra só prefill que contende, ajusta custo `dt = a + b·n`, escolhe o maior degrau (128..2048) que cabe em `MAX_STEP_MS`. |
| `GLM53_FAIR_PREFILL_CHUNK` | `256` (≤ MNBT) | .env.example:302 | Chunk de sondagem até existirem amostras de tempo | — | "every prefill-bearing step costs ~0.3 s fixed on this kit". |
| `GLM53_FAIR_PREFILL_SHARE` | `0.30` (era 0.20); 0..1 | .env.example:303 | Fração do tempo de engine creditada ao prefill contendido | fair share | 0,30 vs 0,20: novato de 30k espera ~145 s em vez de ~220 s; incumbente 1,3–1,4× mais lento (vs 1,16×). |
| `GLM53_FAIR_PREFILL_MAX_INTERVAL_MS` | `2000` (1..600000) | .env.example:304 | Idade para um prefill já servido tomar emprestado um chunk | aging | — |
| `GLM53_FAIR_PREFILL_MAX_STEP_MS` | `2000` (era 1000) | .env.example:305-308 | Limite de duração estimada do passo misto | latency budget | "2000 ms lets the ladder reach 2048 when a 256-token mixed step costs ~1 s (1000 ms deadlocks that fitter)". |
| `GLM53_FAIR_PREFILL_MAX_CHUNKS` | `1` (1..16) | .env.example:309 | Prefills distintos por turno | — | Aumentar não multiplica o chunk de uma requisição. |
| `GLM53_SPINWAIT_MS` | `stock` (1 s); `1..1000`; sweep escolheu `16` | .env.example:343-345; start.sh:476-478 | Janela de busy-loop do leitor `SpinCondition` (`shm_broadcast.py`) | IPC por memória compartilhada; polling vs sleep | 16 ms: +0,95% decode mediano e −85,3% CPU ativa do EngineCore; 2 ms perdeu 1,68%; 32/64 ms perderam ~1,8%. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` | .env.example:346-347; start.sh:481 | — | — | ver 2.2. |
| `READY_TIMEOUT` | `3600` s | start.sh:390 | Timeout de `/health` e `VLLM_ENGINE_READY_TIMEOUT_S` | — | Carga + warmup de "320B MoE" é lenta. |
| `DEFAULT_MAX_NEW_TOKENS` | `65536` (1..1.000.000); vazio = stock | .env.example:260-273; start.sh:230-242 | Default de `max_tokens` só quando o cliente omite | geração/decode hygiene | Sem isso, omissão pode consumir todo o contexto restante e crescer o KV até preemptar outras sessões (#43). "admission is chunk-based" — não é gate de admissão. `max_tokens: null` explícito mantém 16 nativo. Só TP2. |

### 2.10 Speculative decoding (DFlash2 / MTP / adaptive-k)

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `SPEC_METHOD` | `dflash` (default); `mtp`; `none` | .env.example:130-133; start.sh:204-205 | Seleciona o especulador | speculative decoding (draft+verify) | Rollback: `SPEC_METHOD=mtp ./start.sh restart` (MTP k=2 ~24,6 tok/s de baseline). |
| `DFLASH_MODEL` | `incoai/GLM-5.3-Flash-DFlash2` | .env.example:134 | Repo do drafter (~2,3 GiB BF16) | drafter de "block diffusion" (DFlash) | Licença CC BY-NC-ND 4.0 (pesquisa/avaliação). |
| `DFLASH_REVISION` | `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`; vazio = segue `refs/main` | .env.example:135-137; start.sh:208-211 | Pin do drafter | reprodutibilidade | "A mutable Hub main has already changed weights". |
| `DFLASH_TOKENS` | `7` (bloco treinado 8 = 1 âncora + 7 drafts); TP4 `3` | .env.example:138; .env.tp4.example:117 | `num_speculative_tokens` | k do especulador | TP4: k=3 aceita 53% vs 30% a k=7; prose 31–33 vs 26–28 tok/s; k=2 e 5 piores; sem especulação 22,6. |
| `DFLASH_DRAFT_TP` | `2` (TP2); `1` (TP3: rank 0 apenas); `4` (TP4); vazio = herda TP | .env.example:139-141; .env.tp3.example:82-84 | `draft_tensor_parallel_size` | TP do drafter | C4 keep 2026-08-30: 8k 938 / 16k 972 / 100k 997 tok/s prefill; decode 65,1 / 27,1. TP3: drafter tem 32 cabeças / 8 KV que não dividem por 3 → `pad-tp3-config.py` 32/8 → 36/9 e draft TP=1 ("the divisibility tax"). |
| `kv_cache_dtype` (spec) | `auto` | start.sh:2130 | Draft em BF16 | — | ver 2.8. |
| `draft_sample_method` | `probabilistic` | start.sh:2130 | Amostragem do draft | rejection sampling | — |
| `rejection_sample_method` | `standard` | start.sh:2130 | Verificação padrão | rejection sampling lossless | — |
| Atenção do draft | não fixar; SM121 escolhe `FLASH_ATTN` para SWA não causal | README.md:36-38,1760 | — | máscara causal vs bidirecional no bloco | Fixar `TRITON_ATTN` derrubou structured para ~29 tok/s / 0,31 aceite (máscara causal dentro do bloco). |
| `MTP_TOKENS` | `2` | .env.example:142 | `num_speculative_tokens` do MTP | Multi-Token Prediction | Capture sizes MTP `1 2 3 4 6 8 12`. |
| `GLM53_ADAPTIVE_K` | `off` (default); `ema`/`on`/`1` | .env.example:144-151; start.sh:418-419 | Verifica só um prefixo (2, 4 ou 7) dos 7 drafts, por passo | EMA de aceitação; FULL CUDA graphs uniformes | "Lossless at temperature 0"; prose +13–21%. Mínimo sobre as requisições (batch-uniform) porque `FLASHINFER_MLA_SPARSE_SM120` e KDA só suportam FULL graphs uniformes. |
| `GLM53_ADAPTIVE_K_SET` | `2,4,7` (candidatos); experimento `2,3,4,5,7` | .env.example:151 | Comprimentos candidatos; graphs para cada k+1 | — | Precisa de capture sizes múltiplos de 3, 5, 8 (auto no TP2). |
| `GLM53_ADAPTIVE_K_ALPHA` / `_MARGIN` / `_MIN_STEPS` / `_SATURATE` / `_HIST` / `_FILE` | `0.25` / `1.0` / `4` / `max` / `200` / `/root/.cache/vllm/glm53_adaptive_k.json` | .env.example:192-194; patch_adaptive_k.py:12-27 | α da EMA; n = maior valor do set ≤ ceil(ema+margin); passos antes de cortar; observação quando tudo aceito; histograma; override em runtime | — | Retune sem reboot via `~/.cache/vllm-glm53-flash/glm53_adaptive_k.json` (`{"mode":"off"}` volta a k=7). |
| `--cudagraph-capture-sizes` automático | DFlash sem adaptive: `1 2 4 8 16 24 32`; com adaptive: stock ∪ {n×(k+1)} para n=1..MAX_NUM_SEQS; MTP: `1 2 3 4 6 8 12` | start.sh:315-346 | Tamanhos de batch (tokens) capturados como CUDA graph | CUDA graphs | Explícito no `EXTRA_ARGS` sempre vence; TP3/TP4 **não** calculam automaticamente (start-tp3.sh:433-444) — tp3-throughput.env lista 25 tamanhos. Com MAX_NUM_SEQS=4, k=7 → decode = 8 tokens/seq → 8,16,24,32. |

### 2.11 Kernels EXL3 / MoE

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `QUANTIZATION` | `exl3` | .env.example:103 | Método registrado pelo overlay | EXL3/QTIP trellis | Não passar `--moe-backend marlin` (isso é para NVFP4). |
| `EXL3_FUSED_MOE` | `1`; `0` = loop `LinearEXL3` por expert único | .env.example:107-108; start.sh:347-348 | Um `exllamav3_ext.exl3_moe` por camada | kernel fundido, graph-safe | Decode em CUDA graph. |
| `EXL3_TEMP_ROWS_FUSED` | `32` com E3; `256` com E2; TP4 `128`; tp3-throughput `64` | start.sh:357-364; start-tp4.sh:327 | Linhas temporárias por expert no kernel fundido; experts com mais linhas são "gordos" (fat) | fat vs thin experts | "Keep ≥ MAX_NUM_SEQS × (DFLASH_TOKENS+1) so decode stays one graph-safe launch" (4×8=32). 1024 foi mais lento (P2b). |
| `EXL3_MOE_ROW_TILE` | `0` | .env.example:109-110; start.sh:349-351 | Tiles de 128 linhas para fat experts | — | "Tile (P2a) and TEMP_ROWS=1024 (P2b) both lost at MNBT=1024". |
| `EXL3_FAT_SORTED` / `EXL3_FAT_BATCHED` | `0` / `0` (implícitos por tiers superiores) | .env.example:113-116 | Tiers de roteamento ordenado / E1 batched | — | — |
| `EXL3_FAT_KERNEL` | `1` (E2, PR77) | .env.example:117-120 | Kernels diretos/scatter para fat experts | — | +20–21% prefill a frio (PR77). |
| `EXL3_FAT_GROUPED` | `1` (E3, default desde 2026-09-07); `0` = E2 (1M cabe de novo) | .env.example:121-127; start.sh:352-356 | 3 lançamentos por camada (gather, gate/up+SwiGLU, down+scatter) a partir de tabelas no device, sem sync de host | grouped GEMM; host-sync-free | +37–45% prefill frio; custa scratch de 560 MiB descontado do KV. Fail-closed sem os kernels na imagem. **TP4 não encaminha** (exl3.py trata ausente como off). |
| `EXL3_FAT_GROUPED_TOPK` | `8` | exl3.py:1113 | top-k usado no E3 | — | — |
| `EXL3_FAT_EXPERT_LOG` | (default no código) | exl3.py:214 | Log de estatísticas de fat experts | — | — |
| `GLM53_EXL3_MOE_FAST` | `0`; `1` (long-coding) | start.sh:426-430; README.md:1805 | Kernels K4/N256 "thin-decode" SM121 (1 estágio de registrador, 8 de smem) | spills de registrador, pipelines de smem | +7,71/+8,88/+14,59% decode (3 células); numérica formalmente INCONCLUSIVA. Só TP2; exige imagem construída com o patch. |
| `EXL3_OVERLAY_HOST` | `overlay/exl3.py`; ou overlay cooperativo gerado | start.sh:271 | Arquivo `exl3.py` montado | — | Seleciona o MoE cooperativo de decode (extensions/cooperative_moe). TP3 não herda do `.env` (ABI errada). |
| `GLM53_COOP_GEOMETRY` | vazio (adapter = 1); `0` both-narrow, `1` both-wide, `2` A-wide/B-narrow | .env.example:175-178; start.sh:447-450 | Geometria de tiles do MoE cooperativo | tiling de kernels | Geometria 1: structured +7,3%, ×2 +9,3%. Idêntico em todos os ranks antes da captura de graphs. |
| `HAREM_KDA_FLASHKDA` | `0`; `1` (TP3) | start-tp3.sh:367-368 | FlashKDA para prefill KDA (ABI de 14 args, estado FP32) | kernels de atenção linear | Só TP3. |
| `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS` | `0`; `64` | .env.tp4.example:133-142; start-tp4.sh:381-384 | Fatia a chamada final de sparse-MLA em ≤64 linhas de query | mitigação de stall/race | Só TP4; não corrige a race, limita a chamada onde parou. |

### 2.12 Precisão das projeções densas (não roteadas)

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `GLM53_DENSE_FP8` | `.env.example`: `all`; unset no start.sh: `off`; TP3: `dense,kda`; tp3-throughput: `dense,kda,shared,mla`; long-coding: `dense,kda`; grupos: `shared,dense,kda,mla` | .env.example:152-158; start.sh:431-433 | Quantiza em FP8 **weight-only, por canal de saída** na carga e roda via Marlin | W8A16; Marlin | `dense,kda`: ~−11 ms/passo (+10% em tudo; prose +12–19%; +37% em prose difícil com adaptive-k). `all` vs `dense,kda` (2026-09-26): ms/ciclo −4,7/−4,9%, tok/s +4–5%, TTFT inalterado, NOT-WORSE em kl_panel. KL proxy vs stock 0,002–0,013 nats/posição. TP3: Marlin não aceita KDA `f_b_proj`/`g_b_proj` (pitch não alinhado a 8 com 22 cabeças) → ficam BF16. |
| `GLM53_KDA_BF16_LARGE_M` | `.env.example`: `1`; start.sh unset: `0`; TP3: `0` (limpo antes do `.env.tp3`) | .env.example:159-162; start.sh:434-440 | Retém cópia BF16 do peso FP8 do KDA `in_proj`; M>512 usa cuBLAS BF16, M≤512 Marlin | dispatch por M (linhas); prefill vs decode | Prefill TTFT −12%; agentic −4..−10%; +3,3 GiB/rank (12576×4096×2 B × 34 camadas). Requer `kda` em `GLM53_DENSE_FP8`. |
| `GLM53_DENSE_EXL3` | `0`; `1` exige TP=2, `GLM53_DENSE_FP8=off`, `ABLIT=0` | start.sh:441-443,772-804 | Projeções densas em EXL3 (pack com `non_routed_exl3`) | EXL3 para camadas densas | Pack com `non_routed_exl3` + flag 0 é recusado. |
| `GLM53_DENSE_EXL3_PREFILL_BF16` | `off`; `all`; lista de `kda_in,kda_o,mla_qkv_a,mla_q_b,mla_o,shared_gate_up,shared_down,dense_gate_up,dense_down` | start.sh:444-446,707-734 | Retém BF16 reconstruído para M>144 | — | Perfil H3: `kda_in,shared_down,mla_qkv_a,shared_gate_up,kda_o,mla_q_b`. |
| Perfil de pack `glm53_profile` | `name=dense-exl3-h3-dflash2-6bpw`, `draft.model`, `draft.revision` (40-hex), `draft.config_sha256` | README.md:49-66,195-217; tools/pack_profile.py | Selecione automaticamente `GLM53_DENSE_EXL3=1`, `GLM53_DENSE_FP8=off`, `…PREFILL_BF16=…`, `GLM53_KDA_BF16_LARGE_M=0`, `ABLIT=0`, `SPEC_METHOD=dflash`, `DFLASH_DRAFT_TP=2` | metadata-driven config | Draft 5 camadas, 4096 de largura, 6 bpw EXL3 em q/o/MLP/conv/fc, K/V BF16 (`bf16_shards=[1,2]`). Não A/B-medido vs defaults. |

### 2.13 Prefix caching (APC)

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `--enable-prefix-caching` | sempre | start.sh:2112 | Hash de blocos cheios do prompt e reuso | APC, stateless API | Follow-up de ~8k reutiliza 7168/8004 ≈ 90%. |
| `PREFIX_MATCH_UNIT` | vazio (vLLM resolve); `64` suportado; `512` inválido | .env.example:378-381; README.md:1772-1774 | `--prefix-match-unit` (grão do hash) | granularidade de hash | "512 is illegal on this hybrid stack (KDA align block is 64)" (start-tp3.sh:394). |
| `GLM53_APC_RETENTION_INTERVAL` | unset (grade densa); múltiplo de 3584 ≤ 1.000.000; long-coding `14336` | start.sh:598-629,2476-2479 | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` global (Mamba usa; atenção full permanece densa) | retenção esparsa de checkpoints de estado | Guarda um estado Mamba por intervalo → mais histórico no mesmo pool, mas edições/branches recomputam até o checkpoint anterior. |
| `GLM53_APC_RETENTION_INTERVAL_SWA` | vazio (herda); `0` (TP3/TP4 templates e long-coding); múltiplo de 3584; exige `SPEC_METHOD=dflash` | .env.example:383-387; .env.tp3.example:118-120 | Retenção do grupo SWA do drafter; `0` = só fronteiras alcançáveis + prioridade baixa de despejo para blocos só-draft | per-group retention; LRU | Evita que 33 dos 38 IDs de cada segmento (drafter) empurrem MLA/Mamba para despejo. 0/0: 4 históricos de 210K revisitados em ~2,5 s; mas edição a 90% de 128K: 112,49 s vs 14,70 s. |
| `GLM53_APC_NO_STORE` | `1` (honra o flag do cliente); `0` ignora | .env.example:316-319; start.sh:453-458 | Permite `vllm_xargs {"skip_writing_prefix_cache": 1}` por requisição | write-side opt-out; LRU/LIFO do free queue | Lote/avaliação não "reordena" o LRU contra a conversa dona. Aceita `1`, `"1"`, `true`; rejeita outros com HTTP 400. |
| `cache_salt` (requisição) | string | README.md:1817-1827 | Namespace de cache (ainda armazena) | — | — |
| `GLM53_EXPOSE_CACHE_RESET` | `0`; `1` | start.sh:495-502; README.md:1131-1151 | Monta só `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache` | rotas dev | Rotas na raiz ficam fora do guard Bearer (`GUARDED_PREFIX=/v1,/v2,/inference,/cohere`). `VLLM_SERVER_DEV_MODE=1` monta tudo. Só TP2. |

### 2.14 Multimodal

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `LANGUAGE_MODEL_ONLY` | `0` | .env.example:211-212 | `1` → `--language-model-only` | Sem torre de visão libera memória. |
| `LIMIT_MM` | `{"image":48,"video":1}` (histórico: 4 → 100 → 48); TP4 script default `{"image":100,"video":1}` | .env.example:215-222; start.sh:291-294; start-tp4.sh:300-301 | `--limit-mm-per-prompt` (teto de validação, nada reservado) | Pior caso 48×2048 = 98k tokens; acima → HTTP 500 "At most N image(s)…" em vez de engine morto. Clientes stateless reenviam todas as imagens: sessão "envenena" após o teto (docs/images-per-request.md). |
| `MM_IMAGE_TOKENS` | `2048`; vazio = 8000 do checkpoint | .env.example:223-227; start.sh:310 | `max_image_tokens` | Frame 1080p: 2691 tokens sem cap, 2040 a 2048, 1008 a 1024. 8000 excede MNBT 7168 (= encoder cache). |
| `VIDEO_NUM_FRAMES` | vazio (vLLM 32) | .env.example:228-229 | `--media-io-kwargs` | — |
| `MM_PROCESSOR_CACHE_GB` | `1` (vLLM 4); `0` desliga | .env.example:230-232 | Cache de mídia processada em RAM | Em UMA são 4 GiB a menos para o modelo. |
| `MM_ENCODER_TP_MODE` | TP3 `data`; vazio = `weights` | .env.tp3.example:128-130 | Modo TP do encoder de visão | Divisibilidade por 3. |

### 2.15 API, chat template, reasoning e defaults de cliente

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `SERVED_MODEL_NAME` | `GLM-5.3-Flash-EXL3` | .env.example:128 | id do modelo | — |
| `VLLM_API_KEY` | vazio = sem auth | .env.example:253-258; start.sh:504-508 | Bearer em `/v1` (lido nativamente pelo vLLM, fora do argv) | `/health`, `/metrics`, `/tokenize` ficam sem guarda. |
| `CHAT_TEMPLATE_HOST` / `CHAT_TEMPLATE` | `files/chat_template.jinja` / `/opt/glm53/chat_template.jinja` | start.sh:245-246 | Template multimodal | Validado com Jinja2 (loopcontrols) antes do restart (start.sh:951-969). |
| `GLM53_VALIDATE_PYTHON` | unset → tenta `python3`, `python3.12`, `python3.11`, `/usr/bin/python3` | start.sh:847-862 | Intérprete com jinja2 para validar | Definido = único candidato (fail-closed). |
| `GLM53_DEFAULT_REASONING_EFFORT` | vazio; `low`/`high`/`max` (`medium` rejeitado) | .env.example:320-328; start.sh:452 | Default de esforço via `--default-chat-template-kwargs` | Template mapeia qualquer valor fora de `low`/`high` para `max` (chat_template.jinja:7). |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1`; `0` | .env.example:288-289; start.sh:391-392 | Strings `stop` do cliente dormentes até `</think>` | Evita terminar no meio do raciocínio com `content` vazio (DSpark #42). |
| `generation_config.json` do Hub | `temperature=1.0`, `top_p=0.95` | README.md:1646-1661 | Defaults de amostragem quando o cliente omite | "Sending top_p=1.0 explicitly overrides that and is worse". |
| Cliente: `chat_template_kwargs.enable_thinking` (ou `thinking`) | default ligado; `false` fecha `<think></think>` | chat_template.jinja:2-6,264-266 | Liga/desliga raciocínio | "Prefer low-effort thinking to thinking off for long structured output": thinking off corrompeu 11/12 respostas com tabelas numéricas (#257); `enable_thinking:true,reasoning_effort:"low"` → 0/6 corrompidas, 791/792 números corretos. |
| Cliente: `reasoning_effort` | `low`/`high`; ausente → `Max` | chat_template.jinja:7-8 | Linha `<|system|>Reasoning Effort: …` (só com thinking) | Não variar por requisição numa conversa: cai no char 39 do prompt → miss total de prefix cache. |
| Cliente: `max_tokens` | ≥ 32768 com thinking | README.md:1659 | — | Max-effort passa de 8k tokens de saída. |
| Cliente: `chat_template_kwargs.clear_thinking` | `true` para agentes multi-turno | chat_template.jinja:9,154-158 | Substitui raciocínio de turnos antigos por `<think></think>` | Corta contexto. |
| Resposta: `reasoning` (não `reasoning_content`) | — | README.md:1663-1672 | Campo de saída | Clientes lendo `reasoning_content` "perdem" o raciocínio. |
| `tool_choice:"none"` | com tools declarados | patch_tool_choice_none.py | Adiciona `<tool_call>` a `bad_words` | Mantém tools no prompt (prefixo compartilhado) sem gerar tool call. |
| `extra_body` | — | README.md:1644-1645 | Não enviar aninhado via HTTP cru | É opção do SDK Python. |
| Endpoint `/tokenize` | na raiz; campo `prompt` ou `messages` | README.md:1153-1160 | — | `/v1/tokenize` é 404. |

Perfil de cliente de exemplo (`examples/pi/*.json`, agente "pi"): provider `openai-completions`, `baseUrl http://127.0.0.1:8888/v1`, `contextWindow 500000`, `maxTokens 8192`, `chatTemplateKwargs {"enable_thinking": false}`, `supportsReasoningEffort false`, `maxTokensField max_tokens`; `settings.json`: `defaultThinkingLevel off`, `httpIdleTimeoutMs 600000`, compaction `reserveTokens 65536`, `keepRecentTokens 24000`; `subagents.json`: `maxConcurrent 2`, `maxConcurrentForeground 1`, `defaultMaxTurns 20`, `maxSubagentDepth 1`. **(interpretação)** Consistente com `MAX_NUM_SEQS` pequeno (2 subagentes ativos).

### 2.16 Abliteration (remoção da direção de recusa)

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `ABLIT` | `0`; `1` só via export do chamador (TP2) ou `.env.tp3` | .env.example:234-240; start.sh:93,379 | Edita `self_attn.o_proj` na carga (nada é regravado no disco) | Desligado por default. |
| `ABLIT_METHOD` | `auto` (transplant se `ablit/transplant/` populado, senão `proj`); `transplant`; `proj` | .env.example:241-242 | Transplant = copiar bytes das `o_proj` L15–45 de um checkpoint doador; proj = ortogonalização | Transplant reproduz 32/32 bypass; proj com os vetores publicados é estatisticamente aleatório (max |rᵀW| 0,084 igual a vetores aleatórios). |
| `ABLIT_LAYERS` | `15-45` (45 = MTP); 0–14 ficam "safety anchors" | .env.example:245-246 | Faixa editada | — |
| `ABLIT_ALPHA` | `3.0` (proj); `1.0` = projeção simples | .env.example:247-249 | `W' = (I − α r rᵀ/‖r‖²) W` | α>1 "over-projects — stronger, less coherent". |
| `ABLIT_DIRECTION` | `dealign`; `bf_oproj`; caminho `.pt` | .env.example:243-244 | Vetor de direção | — |
| `ABLIT_INCLUDE_MTP` | `1` | .env.example:250-251 | Edita também o `o_proj` do bloco MTP | "a stock MTP draft head keeps proposing refusals". |
| Transplant | `python3 ablit/fetch_transplant.py` (~2,7 GiB, range requests) | README.md:469-479 | Busca só os tensores `o_proj` do doador | `o_proj` é BF16 nativo e `RowParallelLinear` shard só a entrada → edit por rank sem coletivos. |

### 2.17 Boot, warmup, caches e diagnóstico

| Nome | Valores | Onde | O que faz | Por que importa |
|---|---|---|---|---|
| `GLM53_BOOT_SHAPE_WARMUP` | `1`; `0` pula | .env.example:280-282; start.sh:492-493 | Após `/health`, dispara requisições para compilar shapes do DFlash2 (BLOCK 16/32/64/128/256), prefill (3584/7168/14336/65536), sampler (top-k/top-p/ambos), concorrência 1–4 | JIT Triton no meio de coletivo NCCL em TP=2 → watchdog. |
| `GLM53_WARMUP_REQ_TIMEOUT` | `240` s | start.sh:494 | Timeout por requisição | — |
| `GLM53_WARMUP_CANARY` | `1`; `0` | .env.example:283-287 | Falha o start (exit 3) se "Reply with OK." a temp 0 não disser OK ou se DFlash aceitar 0 de ≥64 drafts | Detecta boot degenerado (#249: "!!!!" e aceitação ~0). |
| `GLM53_WARMUP_CANARY_MIN_DRAFTS` | `64` | boot-shape-warmup.sh:51,71 | Mínimo de drafts para julgar | — |
| `GLM53_WARMUP_MAX_CONCURRENCY` / `_DFLASH_K` / `_TRITON_CACHE_DIR` / `_BEARER` / `WARMUP_CURL` | `MAX_NUM_SEQS` / `DFLASH_TOKENS` / host Triton / `VLLM_API_KEY` / seam de teste | start.sh:2835-2843 | — | Fórmula: `BLOCK_SIZE = min(256, next_pow2(scheduled_tokens + 1 + k))`; "BLOCK 8 is unreachable". |
| `CACHE_ROOT` | `~/.cache/vllm-glm53-flash` | start.sh:518 | Cache vLLM do host | — |
| `TRITON_HOST_CACHE` / `TILELANG_HOST_CACHE` / `NV_HOST_CACHE` | `$CACHE_ROOT/{triton,tilelang,nv}` | start.sh:520-533 | Persistência de JIT (incl. `~/.nv/ComputeCache`) | Sem isso, DFlash2 re-JITava um cubin de 31 MiB a cada boot (30 s → 0 s). |
| `GLM53_EXTRA_ENV` | vazio; `NAME=VALUE …` | start.sh:406-407,2541-2582 | Env extra de diagnóstico em ambos os ranks (ex.: `VLLM_DEBUG_WORKSPACE=1`, `VLLM_LOGGING_LEVEL=DEBUG`) | Rejeita nomes do launcher (`NCCL_*`, `HF_*`, `GLM53_*`, `EXL3_*`, `FLASHINFER_*`, `PATH`, …); valores logados nunca. |
| `TAIL` | `0`; `1` segue logs após READY | start.sh:2899-2904 | — | — |
| Lock | `logs/cluster.lock` (flock); stop espera 30 s | start.sh:535-540,977-1005 | Serializa start/restart/stop | Evita dois launchers se destruírem. |
| `VLLM_DEBUG_WORKSPACE` / `VLLM_SERVER_DEV_MODE` | — | README.md:1146,1785 | Diagnóstico / rotas dev | — |

### 2.18 Perfis (conjuntos de valores coerentes)

| Knob | start.sh (unset) | `.env.example` (TP2 atual) | `.env.tp3.example` | `.env.tp4.example` | `examples/tp2-long-coding.env` | `examples/tp3-throughput.env` |
|---|---|---|---|---|---|---|
| `MAX_MODEL_LEN` | 850000 | 850000 | 1000000 | (herda 850000) | 262144 | 1000000 |
| `MAX_NUM_SEQS` | 4 | 4 | (4) | 8 | 2 | 8 |
| `MAX_NUM_BATCHED_TOKENS` | 7168 | 7168 | (7168) | 2048 | 1024 | 4096 |
| `LONG_PREFILL_TOKEN_THRESHOLD` | vazio | vazio | (script 3584; `.env` vazio) | — (não suportado) | — | 2048 |
| `GPU_MEM_UTIL` | 0.85 | 0.85 | 0.80 | 0.75 | 0.865 | (0.80) |
| KV fixo | — | 11 GiB | (kit 40 ou 35 GiB) | — | removido (auto ≈12,7–12,9 GiB) | 32 GiB |
| `DFLASH_TOKENS` | 7 | 7 | (7) | 3 | 7 | 7 |
| `DFLASH_DRAFT_TP` | 2 | 2 | 1 | 4 | (2) | 1 |
| `GLM53_ADAPTIVE_K` | off | (comentado) | (off) | — | off | ema (2,4,7) |
| `GLM53_DENSE_FP8` | off | all | dense,kda | — | dense,kda | dense,kda,shared,mla |
| `GLM53_KDA_BF16_LARGE_M` | 0 | 1 | 0 (comentado) | — | 1 | — |
| `GLM53_EXL3_MOE_FAST` | 0 | — | (forçado unset) | — | 1 | — |
| `GLM53_DRAFT_KV_COMPACT` | 1 (dflash) | (1) | (1) | (1) | 1 | (1) |
| `GLM53_SPINWAIT_MS` | stock | stock | (stock) | (stock) | 16 | — |
| `GLM53_APC_RETENTION_INTERVAL(_SWA)` | —/— | —/vazio | —/0 | —/0 | 14336/0 | (não mudado) |
| `GLM53_MIXED_PREFILL_CHUNK` | fair | fair | fair | fair | fair | fair |
| `EXL3_TEMP_ROWS_FUSED` | 32 (E3) | (32) | (32) | 128 | (32) | 64 |
| `HAREM_KDA_FLASHKDA` | — | — | 0 | — | — | 1 |
| Outros | — | `LOAD_FORMAT=instanttensor` | `TP3_HEAD_OVERRIDE=66`, `ENABLE_EXPERT_PARALLEL=1`, `MM_ENCODER_TP_MODE=data` | `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=0` | `--enable-prompt-tokens-details` | 25 capture sizes; `INSTANTTENSOR_BUFFER_SIZE=536870912` |

Diferenças importantes entre launchers (verificadas por grep):
- `start-tp4.sh` **não** encaminha `EXL3_FAT_GROUPED`, `GLM53_ADAPTIVE_K*`, `GLM53_DENSE_FP8`, `MM_IMAGE_TOKENS`, `LONG_PREFILL_TOKEN_THRESHOLD`, `DEFAULT_MAX_NEW_TOKENS`, e usa `EXL3_TEMP_ROWS_FUSED=128`. Ou seja, TP4 roda o caminho E2 sem adaptive-k nem FP8 denso (interpretação: os knobs foram adicionados depois da medição TP4 de 2026-09-02).
- `start-tp3.sh` e `start-tp4.sh` não aplicam `patch_tool_choice_none`, `patch_cold_load_uma`, `patch_skip_cudagraph_profile`, `patch_default_max_new_tokens`, `patch_apc_no_store`, `patch_kv_capacity_log`, `patch_cache_reset` em runtime, nem a higiene de memória, `GLM53_EXTRA_ENV` ou o cache NV — porém vários desses estão **assados na imagem** pelo Dockerfile (Dockerfile:516-575), onde rodam com os defaults internos quando a variável não chega.

---

## 3. Inventário de modificações

### 3.1 Patches de build no Dockerfile (vLLM → NoPE sparse MLA em SM12x)

| Nome | Arquivo(s) alterado(s) | Problema | Mecanismo | Conceito | Efeito |
|---|---|---|---|---|---|
| NoPE → geometria GLM_NSA (pad 512→576) | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Imagem stock morre no 1º forward: `pe_dim must be 64 for fp8_ds_mla`; a entrada SM120 exige `kv_lora_rank=512`, `qk_rope_head_dim=64`, query 576 | `rope_pad=64` quando `qk_rope_head_dim==0`; `q = F.pad(q,(0,64))`; `kernel_qk_rope_head_dim=64`; `do_kv_cache_update` grava `k_pe` zerado | MLA, RoPE vs NoPE, produto escalar QK | Exato (bloco RoPE zero não muda QK); +24% KV DSA (Dockerfile:1-30,108-148). |
| `supports_dense_mha_prefill=False` | idem | Impl herda `forward_mha` que lança `NotImplementedError` | Prefill vai pelo mesmo kernel top-k MQA | sparse attention no prefill | (Dockerfile:31-36,108-113). |
| Top-k como capacidade + comprimentos válidos | idem | `index_kpool=4` alarga o buffer do indexer para 2176 → "expects sparse block_tables shape (256,1,2048), got (256,1,2176)" | `triton_convert_req_index_to_global_index(..., return_valid_counts=True)`; `sparse_mla_top_k = capacidade`; linhas vazias → slot 0 comprimento 1, saída zerada | page table esparsa | (Dockerfile:38-47,150-210). |
| `buffer_width = topk_tokens` | `models/glm5next/nvidia/model.py`, `mtp.py` | Kernel SM120 só instancia top-k 2048 | Remove `+ (kpool-1)` | template params de kernel | (Dockerfile:48-56,214-221). |
| Expandir `select_k-1` pools | `model_executor/layers/sparse_attn_indexer_kpool.py` | Indexer emite até 2051 candidatos (512 pools×4 + cauda ≤3) | Expande 511 pools (2044) + 3 de cauda = 2047 | indexer DSA com pools comprimidos | Perde 4 de 2048 candidatos (0,2%), os de menor rank; mantém a cauda recente (Dockerfile:73-80,255-276). |
| Page size do DeepGEMM | `platforms/cuda.py` | SM120 aceita `block_kv` só 64; vLLM fixa (32,64) → bloco 1664 e fallback de 32 → assert | Em capability 12 filtra para 64 → alinhamento `index_kpool×64=256` → bloco 1792 | paged MQA, alinhamento de blocos | (Dockerfile:57-65,239-253). |
| Kernel block sizes `[64]` | `flashinfer_mla_sparse.py` | `[64,256]` escolheria 256 para 1792 | Fixa `[64]` | PAGE_BLOCK_SIZE=64 dos kernels | (Dockerfile:67-71,223-237). |
| Pular autotune FlashInfer | `model_executor/warmup/kernel_warmup.py` | Autotune "wedges rank 0 on GB10" | Remove warmup sparse-MLA e retorna antes do autotuner | autotuning | (Dockerfile:81-83,278-300). |
| PDL só Hopper/SM10x | `platforms/cuda.py` | "PDL lowering races KDA state kernels on SM12x" | `return major in (9, 10)` | Programmatic Dependent Launch | (Dockerfile:302-317). |
| Verificação | — | — | Asserts por `inspect.getsource` | fail-closed | Dockerfile:320-367. |
| Registrar `exl3` | `model_executor/layers/quantization/__init__.py`; `config/model.py` (`patch_model_overrides.py`) | vLLM não conhece EXL3 | Adiciona ao `Literal` e import lazy de `Exl3Config`; "exl3" na lista de overrides | registro de quant method | Dockerfile:387-416,489. |
| Build `exllamav3_ext` | fonte turboderp `c5d9c657` | AVX2/AVX512 não compilam em aarch64 | `patch_exl3_ext_aarch64.py` (stubs AVX); `patch_exl3_fat_kernel.py` (E2 `exl3_fat_gemm`/`_scatter` + E3 `exl3_fat_moe_*`); `patch_exl3_decode_pipeline.py` (thin-decode, `glm53_fast_moe_version()==1`); `TORCH_CUDA_ARCH_LIST=12.1a` | extensão CUDA/pybind | Assert dos símbolos (Dockerfile:421-436). |
| `overlay/exl3.py` → vLLM | `…/quantization/exl3.py` | — | `Exl3Config`, carga empacotada, shard TP (gate/up por coluna, down por linha, all-reduce por camada), `exl3_moe` fundido, E2/E3, dense-FP8, KDA BF16, dense-EXL3 | quantização por método | Dockerfile:438-441; README.md:572-577. |
| Overlays assados + testes | `patch_dflash2.py`, `patch_dflash2_exl3.py`, `patch_glm_eagle3.py`, `patch_glm5_drafter_group.py`, `patch_suppress_stops…`, `patch_scheduler_decode_floor.py`, `patch_mamba_align_chunking.py`, `patch_hybrid_prefix_hit.py`, `patch_apc_per_group_retention.py`, `patch_apc_no_store.py`, `patch_mamba_align_state_free.py`, `patch_kv_capacity_log.py`, `patch_xgrammar_termination.py`, `patch_kpool_*`, `patch_indexer_workspace.py`, `patch_spinwait.py --preflight`, `patch_cache_reset.py`, `patch_tool_choice_none.py`, `patch_loadclone.py`, `patch_ablit.py`, `patch_cold_load_uma.py`, `patch_skip_cudagraph_profile.py` | — | Rodados no build na mesma ordem do runtime; testes CPU rodam antes (ex.: `test_apc_no_store.py` com `GLM53_REQUIRE_VLLM=1`) | fail-closed/idempotente | "keeps a fresh build identical to a patched-at-boot image" (Dockerfile:491-575). |
| Agrupar kpool em 1 camada | Dockerfile:468-470,532-534 | 128 camadas → overlay2 não instanciava | Agrupa COPY/RUN | limite de profundidade de layers | CHANGELOG Fixed. |
| InstantTensor | `pip install --no-deps instanttensor==0.2.0` | — | — | loader direct-I/O | Dockerfile:561-564. |
| Stamp | `LABEL glm53.recipe.stamp` | detectar drift | último layer (barato) | — | Dockerfile:577-580. |
| `Dockerfile.e3-layer` | candidato E3 sobre baseline | experimental | `build_exl3_fat_moe_ext.py` → módulo aditivo `exl3_fat_moe_ext` sem `--use_fast_math` | — | Só experimento. |

### 3.2 Overlays de runtime (ordem `GLM53_OVERLAY_ORDER`, start.sh:2053-2082)

Aplicados em cada container de TP2, como `python3 /opt/glm53/<patch>` se o arquivo existir (start.sh:2086-2091). Todos: idempotentes, com âncoras exatas, fail-closed em drift.

| # | Patch | Problema | Mecanismo | Conceito | Efeito medido |
|---|---|---|---|---|---|
| 1 | `patch_cold_load_uma.py` | InstantTensor aborta/encolhe `io_depth` com page cache cheio em UMA; `cuMemcpyHtoDAsync` trava em mmap de 64 KiB | Orçamento por `MemAvailable` (buffer 4 GiB + maior shard + 2 GiB); clone do mmap para memória anônima quando página ≠4096 | UMA, page cache, O_DIRECT, mmap | Cache cheio: stock nunca chega a `/health`; patch: 164 GiB em 36 s @ 4,82 GB/s, `/health` 230 s. |
| 2 | `patch_skip_cudagraph_profile.py` | Dry-capture de memória de graphs roda e é descartado com `CG_ESTIMATE=0` | Condiciona à flag | profiling de memória | KV profile 18 → 7 s. |
| 3 | `patch_glm_video_placeholders.py` | Timestamps de vídeo (glm46v) desalinhados de `grid_t` (100 tokens para 300 slots); `persistent_topk` do GB10 estoura smem | Alinha timestamps; `if False and … select_k in (512,1024,2048)` desliga persistent top-k | processamento de vídeo; smem | `.pth` não importa mais vLLM a cada interpretador: 100 → 6 s até 1ª linha de log. |
| 4 | `patch_suppress_stops_in_reasoning.py` | `stop` casa dentro do CoT | Guarda no detokenizer até `</think>`; ajusta `stop_check_offset` com spec-decode | detokenização incremental | — |
| 5 | `patch_scheduler_decode_floor.py` (v5) | Newcomers esperam decode inteiro (`skip`) ou matam o incumbente (`0`) | Política `fair`: crédito por share, ajuste `a+b·n`, maior degrau que cabe, sondagem do newcomer, decodes primeiro | scheduling decode/prefill | TP2: newcomer 2k em 11,6 s, 30k em 163,6 s durante o ensaio; incumbente mantém ~80–90%. |
| 6 | `patch_mamba_align_chunking.py` | Checkpoints de estado Mamba alinhados ao bloco errado; progresso sub-bloco travado | Usa LCM do scheduler (3584); respeita fronteiras privadas Mamba; back-off EAGLE só para grupo não-SWA participante | checkpointing de estado recorrente | — |
| 7 | `patch_glm5_drafter_group.py` | DFlash2 não cabe na página MLA de 656 B; herdar o bloco do manager MLA (1152/3584) fazia 5 camadas de draft custarem dezenas de MiB por bloco | Grupo `SlidingWindowSpec` extra; "padded slot-share" (`block_size=64`, `page_size_padded=mla_page`, camada i do draft co-possui tensor MLA i); opcional bloco compacto derivado; guarda de backend | hybrid KV cache; páginas com padding | Pool 1.754.237 tokens a 1M; 36k: 44,6% → ~16% de ocupação. Compact: 180 → 121 IDs por requisição máxima. |
| 8 | `patch_hybrid_prefix_hit.py` | `dflash` é `use_eagle()`: coordenador marca todo grupo como EAGLE e MLA perde a última página de 3584; SWA do draft realinha o min | Só `SlidingWindowSpec` exato é EAGLE; draft não encolhe o hit MLA+Mamba; lookup de fronteira com compact (`[glm53-dflash-boundary-lookup-v1]`); piso de 4 tokens frescos para kpool; migra forma legada | APC híbrido; EAGLE lookup | Prefixo de 28.672 com cauda de 64: repeat TTFT −90,02% (0,302 vs 2,6–3,4 s). |
| 9 | `patch_apc_per_group_retention.py` | Drafter consome 33 de 38 IDs por segmento e seus blocos vão para a cauda LRU (inversão de prioridade) | `retention_interval` por grupo via `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA`; `0` = fronteiras + prioridade baixa | LRU eviction; retenção esparsa | ver 2.13. |
| 10 | `patch_apc_no_store.py` | Não há opt-out do lado escrita | `SamplingParams.skip_writing_prefix_cache`; guarda nos 2 sites `_insert_block_hash`; blocos vão para a frente da fila livre | free queue LRU/LIFO | — |
| 11 | `patch_mamba_align_state_free.py` | Slot único `last_state_block_idx` sobrescrito sob scheduling assíncrono → blocos de estado vazavam até o fim da requisição | Lista de índices superados; reserva `1 + max_concurrent_batches + num_speculative_blocks` (10, era 9) | async scheduling; state blocks | — |
| 12 | `patch_kv_capacity_log.py` | Linha "GPU KV cache size" enganosa | Log por grupo + resumo | — | Log-only. |
| 13 | `patch_tool_choice_none.py` | Modelo gera `<tool_call>` mesmo com `tool_choice:none` | `Glm47MoeParser.adjust_request` adiciona `<tool_call>` a `bad_words` | logit masking | CPU-verificado; sem receita ao vivo. |
| 14 | `patch_xgrammar_termination.py` | XGrammar continua avançando após terminação em batches especulativos | Backport exato de vLLM #52805 (`12f64b39`) e #53046 (`c6e19b3`) | structured output + spec decode | Corrige "Failed to advance FSM". |
| 15 | `patch_kpool_tail_slotmap.py` | Slot-map do `KpoolTailSpec` (1 bloco circular) indexava além da linha → corrupção de indexer em gerações ~2k | `block_indices = tl.minimum(block_indices, block_table_stride - 1)` | ring buffer; slot mapping | — |
| 16 | `patch_kpool_tail_seed_stride.py` | Seed de prefill usa stride denso de 2048 B; página real 118.272 B a bloco 3584 → escreve em blocos 0–9 de outra requisição | Backport vLLM #57477: `TAIL_BLOCK_ELEMS`/strides reais | strides de tensor `as_strided` | Issue #264. |
| 17 | `patch_spinwait.py` | Busy-loop de 1 s no leitor shm | `busy_loop_s` configurável | polling | +0,95% decode, −85,3% CPU a 16 ms. |
| 18 | `patch_adaptive_k.py` | k=7 fixo desperdiça verificação em prose | EMA por requisição, mínimo no batch; graphs FULL extras | adaptive speculation | prose +13–21%. |
| 19 | `patch_dense_fp8.py` | Projeções densas BF16 lentas no decode | Reinstala `exl3.py`; mantém quant config em KDA/MLA quando `exl3`; recusa pack `non_routed_exl3` com flag off | W8A16 FP8 Marlin | ~−11 ms/passo. |
| 20 | `patch_dflash2_exl3.py` | Draft EXL3 (6 bpw) | Instala `qwen3_dflash2.py`; offset de prefixos; shape-guard do KV fundido | quantização do drafter | Inerte para drafts BF16. |
| 21 | `patch_loadclone.py` (v2) | Carga mmap lenta; read-ahead | Clone + prefetch limitado + `POSIX_FADV_DONTNEED` | page cache | Não qualificado em GPU. |
| 22 | `patch_default_max_new_tokens.py` | ver 2.9 | Só fallback de omissão | Pydantic before-validator | — |
| 23 | `patch_indexer_workspace.py` | ver 2.8 | `rightsize` + cross-check `index_kpool == compress_ratio` no builder | workspace do indexer | ~+26–28% KV. |
| 24 | `patch_cache_reset.py` | Reset de prefix cache só com dev mode | `elif GLM53_EXPOSE_CACHE_RESET=1` monta só o router de cache | — | — |
| 25 | `patch_ablit.py` (+ `ablit_runtime.py`) | — | Hook no fim de `Glm5NextModel.load_weights` / `Glm5NextMTP.load_weights` | — | No-op se `ABLIT≠1`. |

Assados só no build (não re-rodados em runtime): `patch_model_overrides.py`, `patch_dflash2.py` (registro DFlash2, `decoder_layer_cls`, checkpoint `is_causal:false` → atenção bidirecional no bloco, speculator V2), `patch_glm_eagle3.py` (`EagleModelMixin`/`SupportsEagle3` no `Glm5NextModel`; captura aux em mHC `hc_post`→`hc_contract`, taps `(6, 15, 25, 34, 43)`).

Modelos do drafter (overlay): `qwen3_dflash2.py` ("grouped dynamic conv + candidate selector", porta do vLLM main; top-k via `torch.topk` porque o `LogitsProcessor` da imagem não tem `get_top_k_tokens`); `dflash2_speculator.py` (walk do seletor de candidatos com Gumbel-max vendorizado).

### 3.3 Overlays exclusivos de TP=3 (`overlay/tp3/`, FlyCockpit MIT; `start-tp3.sh:1712-1735,2257-2264`)

| Nome | Problema | Mecanismo |
|---|---|---|
| `patch_tp3_glm.py` | 64 cabeças, vocab 154880, shared I 2048, A_log 64 não dividem por 3 | Pad de cabeças 64→66 (22 locais); vocab `padding_size=lcm(64,3)=192` → 154944; shared I → 2112 ou `disable_tp`; A_log 64→66 na carga. **Não** pad para 96 ("one rank becomes all dummy KDA heads and logits collapse"); **não** pad `moe_intermediate_size` (trellis EXL3 empacotado 2048). |
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` (bind-mount) | Decode SM120 só instancia cabeças locais {8,16,32,64,128} | Pad Q 22→32 em chamadas de decode e fatia de volta. |
| `vllm/model_executor/parameter.py`, `model_loader/weight_utils.py`, `layers/vocab_parallel_embedding.py` (bind-mount) | Loader EP e padding | Substituem os da imagem (read-only); `validate_loadclone_artifacts` checa o `weight_utils` vendorizado (start-tp3.sh:768-784). |
| `patch_exl3_ep_shard.py` | Sob EP, loader EXL3 ainda fatiava cada expert por TP=3 → "dim 0 size 2048 is not divisible by tp=3" | Usa `moe_parallel_config.tp_size` (=1 sob EP puro). Aplicado **depois** de `patch_dense_fp8.py` (que reinstala `exl3.py`). |
| `patch_exl3_expert_map.py` | `expert_map` é property read-only sob EP → AttributeError | Cache em atributo privado. |
| `pad-tp3-config.py` | `qwen3_dflash.py` assert `total_num_heads % tp_size` com world TP 3 mesmo a draft TP 1 | Cópia TP3-only do config do draft: GQA 32/8 → 36/9 (mantém razão 4:1; "FlashInfer requires qo % kv == 0"); `model.safetensors` hardlink do blob resolvido (start-tp3.sh:938-968). |
| `patch_flashkda_tp3.py` (`HAREM_KDA_FLASHKDA=1`) | Prefill KDA via Triton | FlashKDA (vLLM PR #55737), ABI 14 argumentos, estado FP32. |
| Launcher TP3 | `ABLIT=0` forçado; `EXL3_OVERLAY_HOST` limpo; `GLM53_KDA_BF16_LARGE_M=0`, `GLM53_EXL3_MOE_FAST`/`GLM53_KDA_FP8_FAT` unset; strip do KV cap do `.env` | start-tp3.sh:132-173 |

### 3.4 Exclusivo de TP=4

| Nome | Problema | Mecanismo | Efeito |
|---|---|---|---|
| `patch_sparse_mla_slice.py` (`VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64`) | Stall de todos os ranks (#128/#159) numa chamada H16 de sparse-attention | Fatia `flashinfer_trtllm_batch_decode_with_kv_cache_mla` em ≤64 linhas; preimage e resultado pinados por SHA-256; geometria exigida BF16 `[T,16,576]`, páginas `[64,656]`, top-k 2048, ≥33.685.504 B de workspace | Qualificado noutro kit 4×GB10 (153 min de soak; max abs diff 0,00185609 vs FP32). Não corrige a race. |

### 3.5 Mecanismos do launcher (comportamento, não patches de código do vLLM)

| Mecanismo | Onde | O que resolve |
|---|---|---|
| Captura de overrides do chamador antes do `.env` | start.sh:76-113 | Exports (inclusive vazios) vencem o arquivo. |
| Validação numérica pré-stop | start.sh:576-827 | Typos falham antes de derrubar o par saudável (limites: MAX_MODEL_LEN ≤1e6, MAX_NUM_SEQS ≤4096, MNBT ≤8.388.608, etc.). |
| Guard de artefatos de overlay | start.sh:830-974 | Cada overlay precisa ter string de identidade, última linha exata e fazer parse Python (detecta cópia truncada/errada). |
| Recipe stamp + pull que preserva imagem local | start.sh:1530-1739 | Evita servir imagem GHCR sem os artefatos locais. |
| Ship da imagem por SSH | start.sh:1582-1595 | Worker não precisa de GHCR. |
| Verificação de snapshot por tamanhos | start.sh:1911-2006 | Marker de sync é "claim, not proof". |
| Preflight de GID em cada HCA | start.sh:1356-1391 | errno 61 aos 60 s. |
| Preflight de memória | start.sh:1273-1312,1407-1416 | `MemAvailable ≥ util×total + 2 GiB`. |
| Higiene de memória do host | start.sh:2293-2350 | ver 2.6. |
| Capture sizes automáticos | start.sh:319-346 | Adaptive-k precisa de graphs 3/5/8×n. |
| Canário pós-READY | start.sh:2826-2859; scripts/boot-shape-warmup.sh | Não anunciar READY com engine degenerado. |
| Fail-fast do worker | start.sh:2796-2809 | 3 checagens seguidas (~30 s) de container morto. |
| Stop "kill-first" paralelo | start.sh:2949-2964 | Sem estado a descarregar; ~−15 s. |
| TP3: strip do `--kv-cache-memory-bytes` do `.env` | start-tp3.sh:132-156 | 14 GiB é número de TP2. |
| TP3: draft GQA-padded | start-tp3.sh:946-968,2487-2492 | ver 3.3. |

### 3.6 Extensões e ferramentas opcionais

| Nome | Arquivos | O que é | Efeito |
|---|---|---|---|
| MoE cooperativo de decode (TP2) | `extensions/cooperative_moe/*` | Especialização do kernel de 2 estágios do turboderp (`58d4d732`) para H=4096, I local 1024, top-k 8, K4 MCG, 1–32 linhas; ativado por `EXL3_OVERLAY_HOST` gerado por `prepare_profile.py` | Structured ×1 72,03 → 77,29 tok/s (+7,3%); ×2 113,91 → 124,46 (+9,3%); kernel 32 linhas 1,183 → 0,997 ms. `.so` digest `aa3fe5e9…`. |
| MoE cooperativo TP3 (ABI2) | `extensions/cooperative_moe/tp3/*` | I local 2048 / EP96, 32/64 linhas, manifesto verificado | Parte do perfil combinado TP3 (+17,9%/+20,8% C1). |
| Par dense-h3 | `tools/dense_overlay.py`, `tools/dflash2_exl3_quant.sh`, `tools/dflash2_exl3_convert.py`, `tools/dflash2_exl3_package.py`, `tools/stage_dense_h3.py`, `tools/pack_profile.py` | Range-read de ~5,3 GB de tensores densos EXL3 do turboderp (66 s com `--jobs 16` vs 1261 s serial); quantização do draft a 6 bpw com MiaAI-Lab/exllamav3 `63b32f0` (~25 min no GB10; Hessiano sintético, sem calibração) | Não A/B-medido. |
| Kernel lab | `kernel_lab/exl3/*` | Oráculo K1–K8 e sweep de táticas SM121 | Só desenvolvimento. |

### 3.7 Scripts (`scripts/*`)

| Script | Função |
|---|---|
| `boot-shape-warmup.sh` | Warmup + canário (ver 2.17). Escada DFlash `LADDER_S=(1 24 56 120 248)` → BLOCK 16/32/64/128/256; `PREFILL_S=(3584 7168 14336 65536)` (verificados por `/tokenize` para contagem exata); arms de sampler `top_k=40,top_p=1.0` / `top_k=0,top_p=0.9` / ambos com temp 0,8; pós-condição: o cache Triton de `_topk_topp_kernel` deve ter as 3 combinações constexpr. Exit 0/1/3. |
| `spec-accept-gate.sh` | Detecta vllm#53030 (colisão de `BatchDescriptor` em graphs PIECEWISE fixa aceitação em 1,00 na posição 0). PASS se pos0 < 0,999 com ≥100 drafts. |
| `spark_doctor.sh` | Preflight read-only: ping/SSH, estado das CX7 (`ibv_devinfo PORT_ACTIVE`), GPU, banda 0,70–0,95 de `GPU_MEM_UTIL`, Docker/imagem, `/health` e `/v1/models`. |
| `boot_candidate.sh` / `boot_live_best.sh` | Receitas do experimento noturno 2026-09-07 (131k, adaptive-k set 2,3,4,5,7, profiler torch; "live best": 850k, 2,4,7, `dense,kda`, KV 15 GiB, capture 1 2 3 4 5 6 8 9 10 12 15 16 20 24 32). Caminhos absolutos do mantenedor. |
| `logprob_capture.py` | Captura prompt logprobs top-20 via `/v1/completions` (`echo`, `prompt_logprobs=20`) e compara KL/argmax entre dois servidores. |
| `longgen_probe.py` / `multiturn_repro.py` | Detectam lixo não-ASCII e repetição em gerações longas/multi-turno com defaults servidos. |
| `tool-choice-none-preflight.py` | Gates A/B/C (identidade, digests da cadeia `bad_words`, tokenizer: `<tool_call>` é um único token). |

---

## 4. Números e benchmarks

### 4.1 Decode (TP2)

| Data | Configuração | Resultado |
|---|---|---|
| 2026-08-28 | sparkDash, DFlash2 k=7, Structured (1→200) e Code (`clamp_00…49`), temp 0, thinking off, 400 tok, 1M ctx, pool 1.754.237 | ×1 TTFT 719 ms, 62,9 tok/s; ×2 6,62 s, 51,7 stream / 103,3 agg; ×4 6,30 s, 37,1 / 146,5 |
| 2026-08-30 | Lab `bench_decode.py`, mediana 5×400, C4, draft TP=2 | Structured **65,1** tok/s (aceite 0,959; 6,71 tokens/passo); Prose hash-map **27,1** (0,341; 2,39). TP=1 lab: 61,7/26,9. Contexto longo/misto (~60–100k): 24–27. MTP k=2 ~24,6 |
| 2026-08-30 | Aceitação por posição (lab) | Structured 0,98/0,98/0,94/0,94/0,91/0,83/0,83; Prose 0,75/0,58/0,41/0,28/0,16/0,09/0,06 |
| — | `TRITON_ATTN` fixado | structured ~29 tok/s / 0,31 aceite |
| 2026-09-08 | adaptive-k + `dense,kda` (sem coop) | Prose ×1 32,1; ×2 22,1 stream / 41,2 agg; TTFT 268/399 ms |
| 2026-09-08 | Adaptive-k vs k=7 (8 runs/prompt, 131k e 850k) | Silk Road +21%, sky/sunset +13%, hash-map +10%, code +5–15%, counting inalterado |
| 2026-09-16/17 | Coop MoE geometria 1 (A/B/A, 850k, 14 GiB / 883.552 tokens) | Structured ×1 72,03 → 77,29 (+7,3%); ×2 agg 113,91 → 124,46 (+9,3%) |
| 2026-09-17 | sparkDash prose, adaptive-k ema, `dense,kda`, coop, 850k, KV 14 GiB | ×1 TTFT 333 ms, 36,1 stream / 37,1 agg; ×2 365 ms, 25,0/51,1; ×3 405 ms, 22,3/65,8; ×4 401 ms, 19,4/75,3 |
| 2026-09-21 | Compact draft pages OFF/ON/OFF (custom) | Structured 78,589/78,380/76,390; Code 53,493/52,048/49,361; Prose 33,188/31,131/31,200; holdout prose 32,684/32,286/32,495 (−0,93%) |
| 2026-09-26 | `GLM53_DENSE_FP8=all` vs `dense,kda` | ms/ciclo −4,67% [−5,11, −4,06] / −4,91%; tok/s +5,2/+4,3% |
| (head `3d6ffbd`) | `GLM53_EXL3_MOE_FAST=1` TheGrill A/B/A2 | structured-1 72,74→78,36 (+7,71%); code-1 67,83→73,86 (+8,88%); json-1 51,32→58,81 (+14,59%); 2 células retidas |
| — | Spinwait 16 ms | +0,95% decode mediano; −85,3% CPU EngineCore |

### 4.2 Prefill a frio (TP2)

| Data | Config | Resultados (tok/s = prompt/TTFT) |
|---|---|---|
| 2026-08-29 | MNBT 1024, 1M, pool 1.691.099 | 8k 772 (10,36 s); 12k 896; 16k 893; 100k 947 (105,6 s); 256k 936 (273,4 s); 300k 928 (323,2 s); APC 8k follow-up 7168/8004 hits, 1,30 s |
| 2026-08-30 | MNBT 2048, draft TP=2 (C4 keep) | 8k 938 (8,53 s); 16k 972; 100k 997; 256k 973; 300k 941 |
| 2026-09-01 | PR77 E2 vs legado, MNBT 2048 | ~8K 941,04 → 1132,32 (+20,33%); ~100K 1023,20 → 1241,71 (+21,36%); ~300K 995,05 → 1201,02 (+20,70%); outro kit +21,0%/+20,4% |
| 2026-09-06 | E3 vs E2, 500k, util 0,84, rightsize, cap 32, MNBT 7168 | 16k 1578 (+37%); 32k 1624 (+40%); 64k 1637 (+40%); 128k 1629 (+38–42%); 256k 1576 (+45%); camada MoE isolada 7168 tokens 77–91 ms → 31 ms |
| 2026-09-07 | E3 publicado, 900k, util 0,86, rightsize, MNBT 7168, seqs 4 | 8k 1492,1; 16k 1553,7; 32k 1428,2; 64k 1587,0; 128k 1561,7; 256k 1516,8 (172,84 s) |
| (PR #233) | KDA BF16 large-M | 16k 11,41 → 10,12 s (−11,4%); 100k 70,86 → 62,12 s (−12,3%); misto C3+100k 76,21 → 65,43 s (−14,1%); in_proj M=7168: Marlin 27,21 ms, W8A8 12,57, BF16 7,36 |
| 2026-09-26 | KDA BF16 a 11 GiB | W-PRE 16k −12,06%, 64k −12,61%; W-AGT −3,7…−9,8% |
| roofline | improve-prefill.md | 947 tok/s ≈ 34 TFLOPS combinados ≈ 14% do pico BF16 denso do GB10 (~125 TFLOPS); MoE limitado por banda (~75,6 GB lidos por chunk por GPU a ~230 GB/s) |

### 4.3 KV cache e capacidade

| Data | Config | Pool |
|---|---|---|
| 2026-08-29 | 1M, util 0,87, DFlash2+visão, MNBT 2048 | **1.754.237** tokens, 1,75× a 1M (1,95× a 900k), 690 blocos, 18,67 GiB; boot `padded slot-share block=64 mla_page=2351104 (was block=16); draft_bytes/token=2048` |
| pré-E2 | 1M | 1.670.157 / 1,67× / 638 blocos |
| 2026-08-31 | 1M | 643 IDs (642 úteis), 414 IDs/req, 38 IDs/segmento → ~57.344 tokens de conversa em cache |
| 2026-09-01 | MNBT 7168, seqs 4, E2, rightsize | 1.243.902 / 1,24× a 1M |
| E3 900k | util 0,85 / 0,86 / 0,87 | 989.010; 1.023.626–1.084.615; 16,2 GiB / 1.051.648 |
| #204 | InstantTensor 850k/0,85 | 12,4–12,5 GiB disponível vs 17–18 GiB loader off; 1 req de 850k precisa 13,56 GiB |
| #204 | 14 GiB fixo | 883.552 tok (1,04×), 3/3 boots, ~5 GiB livres ociosos |
| #204 | util 0,88 sem pool fixo | 1.017.763 quando boota (1/4 boots); 0,88 × 121,69 = 107,09 GiB |
| 14 GiB + compact | 850k | 876.958 tok (1,03×) sem compact / **1.572.073 (1,85×)** com compact (2026-09-26, dois kits) |
| 11 GiB + compact | 850k (atual) | **1.233.779** (1,45×); head ≥6,2 GiB livres sob carga; 0 linhas NVRM |
| sem compact, 850k stock | — | 13,56 GiB exigidos vs 10,77 disponíveis (boot recusa) |
| TP3 2026-09-15 | 40 GiB cap, 1M | **3.230.656** tokens, 3,23×, 2213 blocos, block 64, fp8 |
| TP3 combinado | 32 GiB/rank | 3.005.093 tokens lógicos |
| long-coding 262k | auto | 12,7–12,9 GiB; "4.15x–4.21x" |
| Ocupação ao vivo (08-29) | temp 0 | 36k compact-64 sem slot-share 44,6%; com slot-share ~16%; 3×36k 21% (54/96/137 s); 3×256k 29,5% (305/608/916 s); 300k 26% (TTFT 356 s) |

### 4.4 Prefix caching

| Data | Rodada | Hits / Compute | TTFT |
|---|---|---|---|
| 2026-08-30 | ~8k cold / follow-up | 0/7995 → 7168/836 | 8,53 s → 1,30 s |
| | ~12k | 0 → 10752/1263 | 12,96 → 1,94 s |
| | ~16k | 0 → 14336/1679 | 16,45 → 2,18 s |
| | 4× ~7,5k concorrentes | 7168 cada | 1,86–2,50 s |
| 2026-09-21 | 100.701 tokens repeat | 100.352 cached (1 página perdida sem compact em outra rodada: 96.768) | 0,70–1,05 s |
| 2026-09-21 | stock ON 814.571 tokens | cold 633,711 s; repeat 2,748 s (813.568 hits); zero preempções |
| APC retention 0/0 (128K) | edit 90% | 111.104 → 0 reuso | 14,70 → 112,49 s |
| | branch 90% | 111.104 → 0 | 3,35 → 99,89 s |
| TP3 combinado | 131.059 / 524.283 tokens | 130.560 / 522.240 cached | 1,124 / 3,409 s |

### 4.5 Concorrência

Ladder 2026-08-31 (1M, MNBT 2048, seqs 16, `GLM53_MIXED_PREFILL_CHUNK=512`, temp 0) — README.md:2069-2076: code agg ×1 41,9 / ×2 52,3 / ×4 77,9 / ×8 93,8 / ×12 116,2 / ×16 127,2; code por stream 44,3/28,8/21,0/13,9/11,9/10,2; chat agg 18,1/26,3/36,1/55,9/66,4/76,7; TTFT mediana 0,4–0,6 s (×1) a 1,3–1,6 s (×16).

Ladder 2026-09-01 (`docs/ladder-final-2026-09-01.json`, 42 células, reps 2, 400 tok; stack não-main com #83/#84/#80): ctx 0 code agg ×1 37,5/34,3; ×16 134,9/137,2 (1 preempção cada); ctx 50k: ×1–×4 hit 0,999–1,0 e TTFT ≤1,4 s; ×8 colapsa (hit 0,25–0,875, TTFT p95 99–454 s); ×16 hit 0,189, TTFT p95 ~900–963 s.

Fair v5 (docs/diditfix.md, 2026-09-15, share 0,20, step 1000 ms): TP2 rep2k newcomer TTFT 11,6 s (durante o ensaio); rep30k 163,6 s (durante); incumbente 20,6–20,9 vs 24,7 solo. TP3: 8,7 s / 110,0 s; incumbente 24,7–25,8 vs 29,7; gap máximo 0,84–0,95 s.

### 4.6 TP3 e TP4

| Data | Topologia | Resultado |
|---|---|---|
| 2026-09-14 | TP3 lab (temp 0, 400 tok, mediana 3) | structured 87,8; code 54,9; prose 39,6; TTFT 0,25 s (TP2 mesmos prompts: 73,4/45,0/32,9/0,33 s); jspark3 3× ~95 structured |
| 2026-09-17 | TP3 sparkDash prose 512 tok | ×1 255 ms 40,1/40,1; ×2 411 ms 28,7/56,6; ×3 323 ms 25,5/75,5; ×4 351 ms 22,8/88,4 |
| histórico | TP3 combinado (coop ABI2 + FlashKDA + FP8 amplo + 8 slots) | C1 coding 41,46 → 48,88 (+17,90%); prose 32,11 → 38,80 (+20,83%); C8 agg 76,41 → 123,06; C8 TTFT mediana 10,63 → 0,52 s; C8 por stream 21,58 → 17,80 (−17,53%); prefill 7,6K 1549,9 → 1411,6 (−8,92%); bench_decode 101,655 (aceite 95,88%) / 44,151 (49,62%); carga de pesos 324–371 s → ~55,5 s; retrieval 999.485 tokens TTFT 718,747 s |
| 2026-09-02 | TP4 (4 Sparks, 1M, draft TP4) | decode 1,45× vs 2 nós; fases mistas 20–35% mais curtas; prefill frio só +29% (1162 vs 901 tok/s a 282k: "fabric-bound"); 8 streams 135 tok/s; k=3 aceita 53% vs 30% |
| 2026-09-03 | TP4 soak 150 min | Hang 3/3 (31–78 min) com prefill chunked de 96k junto a 6–7 streams especulativas; causa interna não identificada |

### 4.7 Qualidade

KLD(teacher‖model), 5 runs frios, 25 janelas seladas, 51.175 posições (malaiwah; mede os **pesos**, não o overlay): TR3 K6 (6bpw) 0,013723 / 254 GB; FP8 oficial (cross-stack) 0,020615 / 328 GB; **EXL3 4bpw servido 0,024555 / 176 GB**; FP8 oficial (stack brandonmusic v44) 0,024629; NVFP4 (v44) 0,060535 / ~180 GB. "4bpw matches official FP8 (~1.00× KLD) at 54% of the bytes" (README.md:410-428).

Outras: FP8 denso: KL proxy 0,002–0,013 nats/posição, argmax 94–100%; `all`: NOT-WORSE (ΔNLL −0,004…+0,0005 nats/token), KL de texto 25–30% acima de pares stock; KDA BF16: rel RMSE vs Marlin 0,002526 (W8A8 0,0266); thin-decode: INCONCLUSIVE (controles stock falham o gate absoluto KL ≤ 0,01); coop MoE: 18/18 casos dentro de 0,3% do pico de referência; tools 45–47/51 em todos os braços (incl. stock).

### 4.8 Boot e carga

| Métrica | Valor |
|---|---|
| Stream de pesos | 164 GiB em 36 s @ 4,88 GB/s (teto O_DIRECT 4,9 GB/s do NVMe); `Model loading took 82.06 GiB and 43.1 seconds` |
| `./start.sh restart` | 259 → 122 s; `/health` 230 → 99 s (2026-09-19) |
| Decomposição dos 99 s | 26 s init Python/engine/NCCL; 39 s stream de pesos; 7 s KV profile; 11 s captura de graphs + warmup; ~10 s API |
| NCCL allgather | 12,6 GB/s/rank |
| InstantTensor vs auto (#204) | ~65–70 s vs ~290–300 s |
| `Model loading took` | 79,65 GiB/rank (82,94 com KDA BF16) |

### 4.9 Multimodal

1080p frame: 2691 tokens (sem cap), 2040 (2048), 1008 (1024). Incidente 2026-09-14: vídeo de 11,9 MB → ~33 imagens full-res a ~7,2k tokens = 236.544 tokens de encode, 44 MB livres, OOM-kill do `VLLM::Worker_TP`. Com cap: 33 frames = 67.320 tokens. 2026-09-05 (util 0,875, MNBT 2048, 1M): image 4 → 16: KV 12,93 → 12,89 GiB; 16 screenshots 1080×1920 em 63 s, 43.136 prompt tokens.

---

## 5. Conceitos necessários (ordem de pré-requisito)

1. **Tensor, matriz, multiplicação de matrizes (GEMM/GEMV)** — Toda camada de uma rede neural é, em essência, `y = xW`. GEMM (matriz×matriz) domina o prefill (muitos tokens), GEMV (vetor×matriz) domina o decode (1 token por sequência). Aqui aparece em cada knob de kernel: `exl3_moe`, Marlin, cuBLAS BF16, `M` (número de linhas) decide o caminho (`GLM53_KDA_BF16_LARGE_M` troca em M>512).
2. **Representação numérica (FP32, BF16, FP16, FP8 e4m3, INT4)** — Formatos com menos bits economizam memória e banda mas introduzem erro de arredondamento. O checkpoint é BF16 nas partes densas, FP8 no KV (`fp8_ds_mla`) e nas projeções densas re-quantizadas (`GLM53_DENSE_FP8`), e ~4 bits nos experts (EXL3).
3. **Rede neural, treino vs inferência** — Treino ajusta pesos por gradiente; inferência só calcula a saída. Tudo neste repositório é inferência: nenhum peso é treinado; mesmo o "abliteration" é edição algébrica de pesos na carga.
4. **Tokenização e vocabulário** — Texto vira IDs inteiros (vocab 154.880 aqui); `/tokenize` expõe isso. O warmup usa `/tokenize` para construir prompts com contagem exata de tokens (boot-shape-warmup.sh:356-398).
5. **Modelo de linguagem autoregressivo e logits** — O LLM prevê a distribuição do próximo token (logits → softmax); gerar é repetir isso. `lm_head` (154.880 × 4096) projeta o estado oculto em logits.
6. **Amostragem (temperature, top-k, top-p, greedy)** — Transforma logits em um token. O Hub define `temperature=1.0`, `top_p=0.95`; benchmarks usam temp 0 (greedy); o warmup compila as três variantes do kernel `_topk_topp_kernel`.
7. **Transformer: embeddings, blocos, residual, normalização** — Pilha de blocos com atenção + MLP ligados por conexões residuais e RMSNorm. GLM-5.3-Flash tem 45 blocos + 1 MTP; a "mHC" substitui a residual simples por múltiplas correntes (hyper-connections) — relevante porque os estados capturados pelo drafter precisam ser "contraídos" (`hc_contract`) para 4096.
8. **Atenção softmax (Q, K, V, cabeças)** — Cada token consulta os anteriores via `softmax(QKᵀ/√d)V`, com várias cabeças. 64 cabeças aqui; a divisibilidade desse número pelo TP é o motivo do padding 64→66 em TP3.
9. **Máscara causal vs bidirecional** — Em LMs, um token só vê o passado. O drafter DFlash2 usa atenção **bidirecional dentro do bloco** (`is_causal:false`); fixar `TRITON_ATTN` (máscara causal no bloco) colapsou a aceitação das posições tardias.
10. **Positional encoding: RoPE vs NoPE** — RoPE gira Q/K por posição; NoPE não usa codificação posicional explícita. As camadas MLA do GLM-5.3 são NoPE (`qk_rope_head_dim=0`), o que conflita com o kernel SM120 que espera 64 dims de RoPE — resolvido com padding zero (exato).
11. **KV cache** — No decode, K e V dos tokens anteriores ficam guardados para não recomputar. Seu tamanho por token × contexto × camadas é o maior consumidor de memória dinâmica; toda a seção 2.8 gira em torno dele.
12. **Prefill vs decode** — Prefill processa o prompt inteiro (limitado por compute, GEMMs grandes); decode gera um token por passo (limitado por banda de memória). Knobs diferentes otimizam cada fase: E3/KDA-BF16 no prefill, FP8 denso/thin-decode/coop/adaptive-k no decode.
13. **TTFT, ITL, tok/s por stream vs agregado** — Métricas de serving: tempo até o 1º token, intervalo entre tokens, throughput individual vs total. Os benchmarks do repo reportam TTFT, "stream tok/s" e "aggregate tok/s"; "prefill tok/s = prompt tokens / TTFT".
14. **Grouped-Query Attention (GQA)** — Várias cabeças de Q compartilham uma cabeça KV. O drafter tem 32 Q / 8 KV (razão 4); em TP3 é padded para 36/9 mantendo a razão porque o FlashInfer exige `qo % kv == 0`.
15. **Multi-head Latent Attention (MLA)** — Comprime K/V num vetor latente de baixa dimensão (`kv_lora_rank=512`) que é o que se guarda no cache. Reduz drasticamente o KV; aqui 656 B/token/camada em FP8 empacotado.
16. **Atenção esparsa com indexer (DSA/NSA) e top-k** — Um "indexer" barato pontua os tokens passados e a atenção cheia só olha os top-k (2048). O GLM comprime o indexer em pools de 4 tokens (`index_kpool=4`) e seleciona 512 pools + cauda recente; o overlay corta para 511 pools + 3 de cauda para caber no kernel de 2048.
17. **Atenção linear / regra delta (KDA)** — Substitui softmax por um estado recorrente de tamanho fixo atualizado a cada token (custo O(1) por token no decode). 34 das 45 camadas são KDA; o vLLM trata o estado como `MambaSpec` com "align mode", checkpoints a cada 3584 tokens e patches de liberação de estado.
18. **Modelo híbrido** — Mistura de camadas de atenção linear (estado) e atenção esparsa (KV). Isso cria múltiplos "grupos" de KV cache com regras diferentes (MLA, KpoolTail, 4× Mamba, SWA do draft), razão de boa parte dos patches de APC.
19. **Mixture of Experts (MoE), roteamento top-k, expert compartilhado** — Cada token passa por só 8 dos 288 experts escolhidos por um roteador, mais 1 expert compartilhado. Dá muitos parâmetros totais (~320B) com poucos ativos (~19,5B); cria o problema de "fat experts" (experts que recebem muitas linhas num chunk de prefill).
20. **SwiGLU / MLP gate-up-down** — O FFN de cada expert: `down(silu(gate(x)) * up(x))`. O kernel E3 funde "gate/up + SwiGLU" num lançamento; `SWIGLU_LIMIT_DEFAULT=10.0` aparece no exl3.py.
21. **Multi-Token Prediction (MTP)** — Cabeça extra treinada para prever tokens adiante; serve como drafter nativo. É o rollback (`SPEC_METHOD=mtp`, k=2) e é afetado por `ABLIT_INCLUDE_MTP`.
22. **Modelos multimodais (torre de visão, tokens de imagem/vídeo)** — Um encoder transforma imagens/frames em tokens que entram no LM. Cada imagem custa milhares de tokens (2040 a 2048 de cap); em UMA sem reserva (`--skip-mm-profiling`) isso pode matar o worker.
23. **Chat template e tokens especiais** — Jinja que monta `[gMASK]<sop><|system|>…<|user|>…<|assistant|><think>` e o formato de tool calls XML. Qualquer mudança no início do prompt (ex.: `reasoning_effort`) invalida o prefix cache.
24. **Raciocínio (thinking) e reasoning parser** — O modelo escreve `<think>…</think>` antes da resposta; o parser `glm45` separa em `reasoning`. Stop strings, esforço (`low/high/max`) e `clear_thinking` interagem com isso.
25. **Tool calling e parser glm47** — Tools declaradas no prompt; o modelo emite `<tool_call>` que o parser transforma em `tool_calls`. `tool_choice:"none"` é imposto mascarando o token opener com `bad_words`.
26. **Structured output / gramáticas (XGrammar, FSM)** — Restringe a geração a um JSON schema via máquina de estados. Com spec-decode, o matcher precisa parar na terminação e validar drafts — backports #52805/#53046.
27. **Quantização: conceito geral** — Representar pesos com menos bits mais escalas. Distingue weight-only (W8A16: pesos FP8, ativações BF16) de W8A8 (também ativações) — o repo rejeitou W8A8 no KDA por mudar probabilidades.
28. **Quantização por canal e escalas** — Cada canal de saída tem sua escala. É como o `GLM53_DENSE_FP8` quantiza projeções densas na carga; o KDA BF16 reconstrói `e4m3 × escala BF16` exatamente como o Marlin usa.
29. **Transformadas de incoerência (Hadamard)** — Rotacionar pesos com Hadamard aleatório espalha outliers e torna a quantização mais uniforme. No EXL3, `suh`/`svh` são **(interpretação)** os vetores de sinal das rotações de entrada/saída; `hadamard_inner.cuh` aparece nos headers vendorizados.
30. **Quantização trellis/QTIP e codebooks (EXL3, MCG, mul1)** — EXL3 (ExLlamaV3, baseado em QTIP) codifica pesos como caminhos num trellis decodificados por um gerador (codebook "mcg" com multiplicador `0xCBAC1FED` ou "mul1"). Aqui só os experts roteados estão em EXL3 K4 (4 bpw); o kernel dequantiza on-the-fly dentro do GEMM.
31. **bpw e trade-off tamanho×qualidade (KLD)** — Bits por peso determinam tamanho; KL divergence contra um "teacher" mede fidelidade. 4 bpw ≈ FP8 oficial em KLD com 54% dos bytes; 6 bpw é melhor (0,0137).
32. **FP8 KV cache** — Guardar K/V em FP8 com escalas por grupo reduz memória à metade vs BF16; aqui é obrigatório (único kernel sparse-MLA em SM12x) e empacotado com a RoPE em BF16.
33. **Marlin** — Kernel de GEMM misto (pesos de baixa precisão × ativações FP16/BF16) muito eficiente para M pequeno. Usado pelo caminho FP8 denso; lento para M grande (27 ms vs 7 ms BF16 a M=7168), motivo do dispatch KDA.
34. **GPU, SMs, capability (sm_121a), cubins, JIT (Triton, TileLang, ptxas)** — Kernels são compilados por arquitetura; GB10 é SM121 (`TORCH_CUDA_ARCH_LIST=12.1a`). Kernels JIT (Triton) compilam no primeiro uso — por isso caches persistidos e warmup antes de servir.
35. **Memória compartilhada (smem), registradores, spills, occupancy** — Recursos por SM limitam kernels. `persistent_topk` estoura smem no GB10 (MNBT 8192 proibido); o thin-decode reduz spills (39 LDL → 0).
36. **Autotuning** — Testar variantes de kernel na inicialização. Desligado (`--no-enable-flashinfer-autotune`) porque trava o rank 0 e dá OOM na UMA.
37. **CUDA graphs (FULL vs PIECEWISE, capture sizes, padding)** — Grava uma sequência de kernels e a reexecuta sem overhead de CPU; só para shapes capturadas. Decode DFlash k=7 usa 8 tokens/seq → capturas 8/16/24/32; adaptive-k exige capturas 3/5/8×n; ranks com graphs diferentes travam o NCCL.
38. **Programmatic Dependent Launch (PDL)** — Sobreposição de kernels dependentes em Hopper/Blackwell. Desligado em SM12x porque corrompia os kernels de estado KDA.
39. **Memória unificada (UMA) do GB10** — CPU e GPU compartilham os mesmos 128 GB; `cudaMemGetInfo` livre = `MemFree`, page cache conta como usado. Explica `GPU_MEM_UTIL` baixo, higiene de page cache, OOM-kill do worker, `MM_PROCESSOR_CACHE_GB=1`.
40. **Page cache, mmap, O_DIRECT, páginas de 64 KiB** — Como o SO lê arquivos. InstantTensor usa O_DIRECT (bypassa cache); o loader auto usa mmap; em kernels de 64 KiB o H2D a partir de mmap trava → clone para memória anônima.
41. **Swap e livelock de UVM** — Se páginas do vLLM vão para swap, o driver pode entrar em spin (GPU 96% a 12–20 W, `/health` 200). Prevenção: `vm.swappiness=0`, reciclar swap antes do start (docs/uvm-livelock-gb10.md).
42. **Paralelismo de tensor (TP): colunas vs linhas, all-reduce** — Divide cada matriz entre GPUs; gate/up por coluna e down por linha, com um all-reduce por camada. Exige divisibilidade (64 cabeças, 2048 intermediate) — base de todo o TP3.
43. **Paralelismo de experts (EP)** — Cada rank possui experts inteiros (96 de 288 em TP3) em vez de fatias. Evita fatiar o trellis EXL3 de 2048; exige patches no loader (`patch_exl3_ep_shard`, `patch_exl3_expert_map`).
44. **Data parallel para o encoder de visão** — Cada rank processa imagens inteiras em vez de fatiar pesos (`--mm-encoder-tp-mode data`).
45. **Coletivos NCCL, anéis, canais, protocolos (LL, LL128, Simple)** — NCCL implementa all-reduce sobre a rede; canais = anéis paralelos; `NCCL_PROTO=^LL128` exclui LL128; buffers limitados para caber na memória pinned.
46. **RDMA, RoCEv2, HCA, GID, MTU** — Rede de baixa latência sobre Ethernet (ConnectX-7). O GID é o endereço RoCE (tipo v2 com `::ffff:<ip>`); índice errado → errno 61; MTU 9000 recomendado.
47. **Interfaces de bootstrap (gloo/NCCL sockets) vs caminho de dados** — O handshake usa TCP numa interface (`NCCL_SOCKET_IFNAME`); os dados vão por verbs (`NCCL_IB_HCA`). Em TP3 o controle vai pela LAN de gerência.
48. **Topologia de anel dirigido e cross-NIC / subnet-aware routing** — Com 3+ nós e 2 portas cada, o cabeamento precisa formar um anel dirigido e o NCCL precisa parear NICs por sub-rede.
49. **Rendezvous distribuído (master addr/port, ranks, headless)** — Rank 0 hospeda o rendezvous e a API; ranks > 0 rodam `--headless`. `--distributed-executor-backend mp` sem Ray.
50. **Arquitetura do vLLM V1 (API server, EngineCore, workers, shm broadcast)** — Separação em processos; EngineCore agenda, workers executam, comunicação por memória compartilhada (`SpinCondition` — `GLM53_SPINWAIT_MS`).
51. **PagedAttention, blocos KV e BlockPool** — KV em blocos de tamanho fixo com IDs globais; o bloco 0 é o "null block". Hybrid layout compartilha um pool com IDs únicos entre grupos.
52. **KV cache groups e specs (MLAAttentionSpec, MambaSpec, SlidingWindowSpec, KpoolTailSpec)** — Cada tipo de camada tem regras de alocação e cache próprias; o "GPU KV cache size" do log soma blocos por grupo, não é tamanho de cache de prefixo.
53. **Sliding window attention (SWA)** — Atenção só sobre os últimos W tokens (2048 no drafter). Mantém KV do draft limitado; retenção por grupo usa isso.
54. **Profiling de memória do vLLM e `gpu_memory_utilization` / `kv_cache_memory_bytes`** — vLLM roda um passo dummy, mede pico e dá o resto ao KV; ou usa um valor fixo. Workspaces (indexer 5 GiB, scratch E3 560 MiB) e estimativa de graphs (2,43 GiB) saem desse orçamento.
55. **Continuous batching e `max_num_seqs`** — Requisições entram/saem do batch a cada passo. `max_num_seqs` limita sequências em voo; acima disso, fila.
56. **Chunked prefill, `max_num_batched_tokens`, long prefill threshold** — Prompts longos são processados em pedaços por passo, dividindo o orçamento com decodes. MNBT 7168 é o default; limite de long-prefill evita monopolizar.
57. **Interferência decode/prefill e políticas de scheduling (skip, cap, fair)** — Misturar prefill num passo de decode aumenta a latência do incumbente; não misturar faz o novato esperar. O `fair` v5 é um escalonador de fair-share com orçamento de duração do passo.
58. **Preempção** — Quando falta KV, o vLLM expulsa uma requisição e recomputa depois. Métrica acompanhada nos ladders e qualificações.
59. **Scheduling assíncrono** — O scheduler prepara o próximo passo enquanto o atual roda; cria o problema de estado Mamba "em voo" que motivou `patch_mamba_align_state_free`.
60. **Automatic Prefix Caching (APC), hash de blocos, alinhamento** — Blocos cheios do prompt recebem hash encadeado; um prompt que começa igual reusa. Em modelo híbrido o hit é o mínimo entre grupos e alinhado ao LCM (3584).
61. **Checkpoint de estado recorrente para APC** — Para atenção linear não há KV por token: guarda-se o estado em fronteiras ("align mode"). Por isso hits vêm em múltiplos de 3584 e retenção esparsa troca capacidade por recomputação.
62. **Política de despejo LRU e prioridade por grupo** — Blocos livres com hash vão para a cauda (protegidos), sem hash para a frente. Retenção SWA=0 e no-store manipulam essa ordem.
63. **Speculative decoding (draft + verify, aceitação, rejection sampling lossless)** — Um modelo barato propõe k tokens e o alvo verifica todos num único passo; com rejection sampling a distribuição final é a do alvo. Ganho depende da taxa de aceitação (0,96 structured vs 0,34 prose).
64. **EAGLE/EAGLE3 (estados ocultos auxiliares)** — Drafters que usam estados ocultos intermediários do alvo. O GLM ganhou a interface `SupportsEagle3` com taps nas camadas (6,15,25,34,43); a regra de lookup EAGLE exige um bloco após a fronteira.
65. **DFlash/DFlash2 (drafter de difusão em blocos)** — Propõe um bloco inteiro (8 posições) em paralelo com atenção bidirecional, condicionado no alvo; DFlash2 adiciona conv dinâmica agrupada e seletor de candidatos. KV do contexto é projeção por posição do estado oculto do alvo (daí o lookup de fronteira).
66. **Adaptive speculation length (EMA)** — Verificar menos drafts quando a aceitação recente é baixa economiza compute; precisa ser uniforme no batch para caber num CUDA graph FULL.
67. **Draft TP** — O drafter pode rodar em TP próprio (2 no TP2, 1 no TP3). Draft TP=1 evita tráfego na rede por passo de draft, mas concentra trabalho no rank 0.
68. **Loaders de pesos (safetensors, mmap, InstantTensor, prefetch)** — Formas de ler 164 GiB; o InstantTensor é rápido mas consome memória que o KV perderia. Trade-off central do pool fixo de KV.
69. **Distribuição de pesos multi-nó (rsync vs NFSv4)** — Copiar vs montar read-only via rede de dados. NFS com `nconnect=8` sobre CX7 evita duplicar 164 GiB por nó.
70. **Contêineres, bind mounts, host network, IPC** — Toda a pilha roda em Docker com rede do host, IPC do host e devices InfiniBand; overlays são bind-mounted read-only.
71. **Reprodutibilidade (pins de digest/commit, recipe stamp, byte-identical)** — Imagem, vLLM, ExLlamaV3, pesos e drafter são pinados; builds divergentes são recusados.
72. **Idempotência e fail-closed em patches de código-fonte** — Patches por âncora exata que falham se o upstream mudou; preflight antes de parar containers.
73. **OpenAI-compatible API (chat completions, streaming SSE, usage, Bearer)** — Contrato consumido pelos clientes; stateless (cliente reenvia o histórico), o que torna APC e limites por requisição (imagens) centrais.
74. **Logit masking / bad_words** — Proibir tokens zerando seus logits; é o mecanismo do `tool_choice:none` e também atua nas posições de draft.
75. **Abliteration (direção de recusa)** — Remover do espaço de saída de `o_proj` a direção associada a recusas, `W' = (I − α r rᵀ) W`, ou transplantar pesos já editados. Aplicado por rank sem coletivos porque `RowParallelLinear` fatia só a entrada.
76. **Métricas de qualidade: KLD, NLL, argmax agreement, testes A/B/A com bootstrap** — O repo avalia mudanças numéricas com painéis de KL e controles stock-vs-stock; "NOT-WORSE" vs "EQUIV" vs "INCONCLUSIVE".
77. **Metodologia de benchmark (frio vs quente, salt único, A/B/A, drift)** — Prompts com salt para evitar APC, reset de cache, repetição intercalada, controle de drift. Explica por que números antigos e novos não são diretamente comparáveis.
78. **Observabilidade (`/metrics` Prometheus, counters de spec-decode, logs de boot)** — `vllm:spec_decode_num_draft_tokens_total`, `accepted_tokens_per_pos_total`, `prefix_cache_hits_total`; usados pelo canário e pelo gate de aceitação.
79. **Roofline (compute-bound vs memory-bound, intensidade aritmética)** — Explica por que o MoE no prefill a 1024 tokens é limitado por banda (todos os 288 experts lidos) e por que chunks maiores/kernels agrupados ajudam.
80. **Grouped GEMM e sincronização host-device** — Executar muitos GEMMs pequenos num lançamento a partir de tabelas no device; evita `.tolist()` (sync) por camada — essência do E3.

---

## 6. Receita passo-a-passo (reconstruir do zero, com o porquê)

### Passo 0 — Entender o alvo e o orçamento
- Modelo híbrido MoE (~320B total, ~19,5B ativos), 45 camadas + MTP, visão. Em 4 bpw nos experts + BF16 no resto: ~164 GiB de checkpoint; ~79,65 GiB por rank carregados em TP2 (README.md:1241-1242).
- Cada GB10 tem ~121,7 GiB utilizáveis como "GPU" (0,88 × 121,69 = 107,09). Orçamento TP2 típico (interpretação, a partir dos números do repo): util 0,85 → ~103 GiB ao vLLM; pesos 79,65 (+3,3 com KDA BF16 = ~83); KV 11 GiB; resto em ativações, graphs, workspaces (indexer rightsize ~126 MB, scratch E3 560 MiB); ~17–18 GiB ficam ao SO/page cache/encode de visão. Por isso o repo repete "raise only with `MemAvailable` in front of you".
- Por token de contexto (interpretação): MLA 11 camadas × 656 B ≈ 7,2 KB + indexer K (132 B por pool de 4 tokens ≈ 33 B/token/camada × 11 ≈ 0,36 KB) ≈ 7,6 KB/token → ~7,06 GiB por 1M tokens, batendo com "~7.1 GiB per 1M" do README; mais ~7,4 GiB "fixos" (Mamba/KDA em align, janela do draft, etc.).

### Passo 1 — Hardware e rede
1. Dois GB10 com CX7 cabeados diretamente (QSFP), uma /24 por cabo, MTU 9000 (TP3: anel dirigido Port0→Port1).
2. Descobrir nomes: `ls /sys/class/infiniband`, `ip -br a`; preencher `HEAD_CX7_IF/IB`, `WORKER_CX7_IF/IB` (nomes variam por kit).
3. Descobrir o GID RoCEv2 de cada nó: `cat /sys/class/infiniband/<dev>/ports/1/gids/{0..7}` e `gid_attrs/types`; escolher o índice com `::ffff:<ip-da-fabric>` tipo "RoCE v2" (por nó se diferente: `HEAD_GID`/`WORKER_GID`).
4. SSH sem senha head → worker; Docker sem sudo nos dois; `sudo -n` opcional (higiene de page cache).
5. Sistema: `vm.swappiness=0`, manter swap e reciclá-la antes de starts (docs/uvm-livelock-gb10.md). Rodar `scripts/spark_doctor.sh`.
Por quê: NCCL não funciona sobre os aliases 10.0.0.x; GID errado mata o rank aos 60 s; UMA + swap pode travar o driver.

### Passo 2 — Imagem
1. Partir de `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (vLLM 487ecf187, CUDA 13, arm64).
2. Aplicar os patches NoPE/sparse-MLA (seção 3.1): padding 512→576, prefill sem dense MHA, top-k como capacidade, `buffer_width`, `select_k-1` pools, page size 64 do DeepGEMM, kernel block `[64]`, pular autotune, PDL só SM9x/10x.
3. Registrar `exl3` e compilar `exllamav3_ext` pinado (`c5d9c657`) para `sm_121a` com stubs aarch64, kernels E2/E3 e thin-decode.
4. Copiar `overlay/exl3.py`, drafter DFlash2 e todos os overlays; rodar seus testes e aplicá-los no build.
5. Instalar `instanttensor==0.2.0 --no-deps`; aplicar cold-load UMA e skip-cudagraph-profile.
6. Carimbar `glm53.recipe.stamp`. (Ou simplesmente puxar `:exl3-instanttensor`; `SKIP_BUILD=1`.)
Por quê: a imagem stock não serve este checkpoint em SM12x (pe_dim, geometria do kernel, autotune, PDL).

### Passo 3 — Pesos
1. `./download.sh` (head only): `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw@25a44fdb…` (120 shards) e `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c…`.
2. Worker: `NFS_SHARE=1` (export NFSv4 do cache HF pela CX7, `nconnect=8`) ou rsync (padrão do template).
3. Opcional: `ablit/fetch_transplant.py` (abliteration) ou preset `abliterated`; opcional `GLM53_MODEL_PRESET=dense-h3`.
Por quê: pins evitam que mudanças no Hub mudem o modelo servido; NFS evita duplicar 164 GiB.

### Passo 4 — Overlays de runtime (ordem importa)
Montar e executar, em cada rank, a sequência de `GLM53_OVERLAY_ORDER` (seção 3.2). Pontos de ordem: hybrid-prefix antes de per-group-retention (mesmo ponto de inserção no coordinator); no-store depois; kv-capacity-log depois do drafter-group (mesmo arquivo); mamba-chunking depois do decode-floor v5; em TP3, EP-shard/expert-map **depois** de dense-fp8 (que reinstala `exl3.py`).

### Passo 5 — Argumentos de lançamento (TP2)
```
vllm serve <snapshot> --served-model-name GLM-5.3-Flash-EXL3 --host 0.0.0.0 --port 8888 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank {0|1} --master-addr 10.0.0.1 --master-port 29521 \
  --distributed-executor-backend mp [--headless no rank 1] \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
  --enable-prefix-caching --no-enable-flashinfer-autotune \
  --quantization exl3 --max-model-len 850000 --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 --max-num-batched-tokens 7168 --kv-cache-dtype fp8 --load-format instanttensor \
  --speculative-config '{"method":"dflash","model":"<draft>","num_speculative_tokens":7,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":2}' \
  --chat-template /opt/glm53/chat_template.jinja \
  --limit-mm-per-prompt '{"image":48,"video":1}' --mm-processor-kwargs '{"max_image_tokens":2048}' \
  --mm-processor-cache-gb 1 --skip-mm-profiling \
  --kv-cache-memory-bytes 11811160064 --cudagraph-capture-sizes 1 2 4 8 16 24 32
```
Env de política (idêntico nos ranks): `GLM53_DENSE_FP8=all`, `GLM53_KDA_BF16_LARGE_M=1`, `EXL3_FUSED_MOE=1`, `EXL3_FAT_KERNEL=1`, `EXL3_FAT_GROUPED=1`, `EXL3_TEMP_ROWS_FUSED=32`, `GLM53_INDEXER_WORKSPACE=rightsize`, `GLM53_DRAFT_KV_COMPACT=1`, `GLM53_MIXED_PREFILL_CHUNK=fair` (+ knobs fair), `GLM53_SUPPRESS_STOPS_IN_REASONING=1`, `DEFAULT_MAX_NEW_TOKENS=65536`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, NCCL da seção 2.2.
Raciocínio de cada decisão:
- **fp8 KV**: único kernel sparse-MLA em SM12x.
- **850k + 11 GiB fixo + compact**: InstantTensor come ~5 GiB do KV; 11 GiB com compact dá 1,45× de 850k e paga os +3,3 GiB do KDA BF16; 1M não cabe com E3 a util ≤0,87.
- **util 0,85**: cada 0,01 é 1,2 GiB de folga do host; prefill longo a 0,87 derrubou o head.
- **MNBT 7168**: melhor para seqs=4 com E2/E3; nunca 8192 (smem).
- **TEMP_ROWS 32 = 4 seqs × 8 tokens**: decode inteiro num lançamento graph-safe.
- **DFlash2 k=7 draft TP2**: maior aceitação em structured/code; `FLASH_ATTN` bidirecional.
- **capture sizes 8/16/24/32**: batches de decode com 1–4 seqs × 8 tokens.
- **fair**: newcomers não esperam decode inteiro nem destroem o incumbente.
- **MM caps**: nada reservado para visão → teto de validação precisa ser codificável.

### Passo 6 — Subir e verificar
1. Preflight de memória (`MemAvailable ≥ util×total + 2 GiB`), higiene de page cache, `docker run` worker e depois head.
2. Esperar `/health` (até 3600 s). Conferir logs: `padded slot-share block=…`, `[glm53-dflash-boundary-lookup-v1]`, `dense fp8 groups: all`, 34 linhas `kda bf16-large-m retained`, `[glm53-kv-capacity-log] …`, `GPU KV cache size: …`, `patched scheduler.py (# [glm53-decode-floor:v5])`.
3. Warmup de shapes + canário (OK a temp 0; aceitação > 0 de ≥64 drafts).
4. `scripts/spec-accept-gate.sh` após >100 drafts (pos0 < 1,00 com decaimento monotônico).
5. Testes funcionais: `curl /v1/chat/completions` com e sem thinking; `/tokenize`; tools; imagem.

### Passo 7 — Medir
- Decode: `tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400` e prose.
- Prefill frio: `docs/_run_cold_prefill.py` (salt único, `max_tokens=8`, streaming com usage).
- APC: `tests/bench_prefix_cache.py --runs 3` (com `GLM53_EXPOSE_CACHE_RESET=1` em kit isolado).
- Concorrência: `tests/bench_concurrency.py --levels 1,2,4,8,12,16 --modes code,data,chat --ctx 0,50000,100000 --reps 3`.
- Sempre A/B/A com mesmo contexto, util e KV fixos; observar `MemAvailable` mínimo (guardas usados: ≥2 GiB e ≤256 MiB de swap-out novo).

### Passo 8 — Tuning por workload
- Prose/decode: `GLM53_ADAPTIVE_K=ema` (set 2,4,7; capture sizes automáticos), MoE cooperativo (geometria 1), `GLM53_EXL3_MOE_FAST=1` (imagem própria), `GLM53_SPINWAIT_MS=16`.
- Sessões longas de código: perfil `examples/tp2-long-coding.env` (262k, 2 seqs, MNBT 1024, util 0,865, retenção 14336/0 — não qualificada).
- Muitos usuários curtos: aumentar `MAX_NUM_SEQS` (TP4: 8), talvez MNBT 2048; lembrar que capture sizes e `TEMP_ROWS_FUSED` precisam acompanhar.
- Cache de conversas: `GLM53_APC_RETENTION_INTERVAL_SWA=0`; lotes/evals com `vllm_xargs.skip_writing_prefix_cache=1`.
- Mais nós: TP3 (padding 66, EP, draft TP1 com GQA 36/9, anel dirigido, `SOCKET_IFNAME` de gerência, KV fixo por topologia) ou TP4 (util 0,75, 8 seqs, k=3, MNBT 2048, slice 64 opcional).

---

## 7. Discrepâncias e pontos incertos encontrados

- **Número de camadas MLA**: README.md:677 diz "Target MLA (12 layers)"; o log de capacidade (README.md:1880) mostra `MLAAttentionSpec layers=11`. LAYER_MAP.json explica: 11 DSA no corpo + 1 no MTP (camada 45).
- **"Mamba (33 layers, 3 groups)"** (README.md:679) vs log com 4 grupos Mamba × 9 camadas e 34 camadas KDA no LAYER_MAP — incerto como o vLLM agrupa (provavelmente padding de grupos).
- **KV fixo do TP3**: README.md:1499 diz 40 GiB (`42949672960`, pool 3.230.656 tokens); docs/adopt-fp8…md diz que o `.env.tp3` do cluster medido fixa 35 GiB; CHANGELOG 1.4.0 fala em "35 GiB KV cap after a head OOM".
- **Comentários desatualizados**: `.env.example:111` diz "keep 128" para `EXL3_TEMP_ROWS_FUSED` (default atual 32/256); `.env.tp4.example:146` e README (vários) citam o cap do `.env` como `15032385536`, mas o template atual usa `11811160064`; README.md:994 ("Default stays 0") é histórico — o default atual de `GLM53_DRAFT_KV_COMPACT` é 1 sob dflash; README.md:1774 duplica a linha `PREFIX_MATCH_UNIT`.
- **`ABLIT` no `.env` (TP2)** é ignorado pelo `start.sh` (forçado a 0 na linha 93); README.md:451 ("Set `ABLIT=1` to enable") deve ser lido como export do chamador.
- **`LONG_PREFILL_TOKEN_THRESHOLD` no TP3**: script default 3584 só vale se a variável estiver *indefinida*; como o `.env.example` a define vazia, (interpretação) o TP3 sai sem a flag a menos que `.env.tp3` a defina.
- **TP4 defasado**: sem E3, adaptive-k, FP8 denso, caps de imagem por token, default max tokens; `LIMIT_MM` default do script `{"image":100}` (mas herda 48 do `.env`).
- **"TR3"**: o repositório não define a sigla (rótulo do publicador do checkpoint).
- **Significado exato de `suh`/`svh`/`mcg`**: o repo descreve "trellis (int16) + suh/svh (fp16) + mcg (int32 marker)"; a leitura como vetores de sinal de Hadamard e codebook MCG é minha interpretação baseada no ExLlamaV3/QTIP.

---

## Addendum 2026-10-30

Este inventário descreve o HEAD `94ae731` (2026-09-28) e continua válido como retrato dele. O que mudou depois está
verificado em [`mia-updates-2026-10-30.md`](mia-updates-2026-10-30.md); para o escopo deste arquivo (lançamento,
topologia, parâmetros), lido no HEAD `a8cc40d` (2026-10-13):

- **Recomendação upstream:** o README agora se intitula "Use the TensorFold recipe" e diz que todos devem usar a
  receita TensorFold (`cafea0f`, 2026-10-04).
- **Pesos default** (`98cf18f`): `MODEL` e `MODEL_FALLBACK` = `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`,
  `MODEL_REVISION=76c0b5173166d2795dd48860f45d8224817f894c`, `EXPECTED_SHARDS=83` (~164 GiB). O espelho TR3 e o
  fallback `brandonmusic/...` saíram do default; leia as menções a TR3/120 shards acima como o snapshot.
- **TP4 deixou de estar tão defasado:** `start-tp4.sh` encaminha `EXL3_FAT_GROUPED` com default `1` e
  `EXL3_TEMP_ROWS_FUSED` 32 com E3 / 256 com E2 (`e5dc885`; 4 Sparks, 1M ctx: prefill frio +35/+38/+41% em
  8K/32K/100K, decode inalterado), aceita adaptive-k só como opt-in explícito do `.env.tp4` (`1e8f928`) e respeita
  `GLM53_DEFAULT_REASONING_EFFORT` (`4709bc5`). FP8 denso, `MM_IMAGE_TOKENS`, `LONG_PREFILL_TOKEN_THRESHOLD` e
  `DEFAULT_MAX_NEW_TOKENS` continuam sem encaminhamento no TP4.
- **`--enable-prompt-tokens-details`** passou a ir em todos os launchers (`ecd1af6`, PR #206), não só em
  `examples/tp2-long-coding.env`.
- **Overlays:** 26 em vez de 25 (`patch_indexer_warmup_range.py`, #203, como #24; `patch_cache_reset.py` vira #25 e
  `patch_ablit.py` #26); o decode-floor passou de v5 para v7 (`b94fdfd`, #283; detalhes no addendum de `glm-mods.md`).

# Inventário — Qwen3.8-Flash-Next (NVFP4) em 2× DGX Spark com vLLM TP2+EP+MTP3

> Repositório analisado: `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` (clone local, somente leitura).
> Escopo deste arquivo: **lançamento, parâmetros, variáveis de ambiente, args do vLLM, perfis/lanes, orçamento de KV, benchmarks, gotchas**.
> Arquivos lidos integralmente: `README.md` (1073 linhas), `.env.sample`, `start.sh` (1226 linhas), `start-tp1.sh`, `start-fp8.sh`, `start-v030.sh`, `stop.sh`, `download.sh`, `check-weights.sh`, `verify-weights.py`, `CHANGELOG.md`, `docs/CLAUDE/fable5-1-report.md`, `docs/HANDOFF-single-spark.md`, `docs/bench/rerun-20260916.md`, `docs/cache-reporting.md`, `tp1/*`, `bench/*`, `tests/*` (os `.py`; `test_check_weights.sh`/`test_verify_weights.py` por amostragem). Também li os helpers que `start.sh` chama (`files/resolve_snapshot.py`, `files/detect_ple_dtype.py`, `files/patch_checkpoint_config.py`, `files/evict_page_cache.py`, `files/nfs-share.sh`, `files/nfs-server/*`, `files/memwatch.sh`) e os cabeçalhos/mecanismo de todos os `files/patch_*.py`, `files/qsa_gb10/*`, `files/overlay/*.diff`, `files/fp8dense/*` (o detalhamento linha-a-linha dos patches de kernel fica para o inventário de patches, mas o mecanismo está resumido aqui).
> Datas: o repositório registra medições entre 2026-09-02 e 2026-09-27. Hoje: 2026-09-28.
> Convenção: "README:L" = linha do README; "start.sh:L" = linha do start.sh; etc.

---

## 0. Notas de fidelidade (leia antes) — discrepâncias encontradas entre documentação e código

Estas discrepâncias são reais (verificadas lendo os arquivos) e importam para quem vai recriar a receita:

1. **Tabela `.env Reference` do README está parcialmente desatualizada.** README:216–217 diz `MAX_MODEL_LEN` sample = `1000000` e `YARN_ENABLE` sample = `true`. O `.env.sample` real (linhas 45–46) traz `MAX_MODEL_LEN=262144` e `YARN_ENABLE=false`. O CHANGELOG (linha 406–407) confirma a mudança "1000000 → 262144 e true → false" por causa do bug do YaRN (ver §3). **Vale o `.env.sample`.**
2. **Bloco "Default runtime" do README (README:435–450) é um retrato antigo**: mostra `--kv-cache-dtype auto` (o default atual é `fp8`), mostra `--hf-overrides '{"rope_parameters":...}'` no nível raiz (exatamente a forma bugada que o CHANGELOG diz ter sido corrigida para `{"text_config":{...}}`), e não mostra `--enable-prompt-tokens-details`, `--mm-encoder-tp-mode`, `--mamba-ssm-cache-dtype`. O comando real é montado em `start.sh:875–952` (reproduzido em §2.12).
3. **`PLE_QUANT_OVERRIDE=fp8`**: o README (L460, L557) e o relatório fable (L423) dizem que `start.sh` injeta essa env var e que o shim do `ple_layer.py` a lê. **O `start.sh` atual não define `PLE_QUANT_OVERRIDE` em lugar nenhum** (grep sem resultados fora de docs). O mecanismo atual é: `files/detect_ple_dtype.py` deduz `ple_embedding_dtype` do checkpoint e `start.sh` injeta via `--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}'`; o `ple_layer.py` patchado despacha por `text_config.ple_embedding_dtype` (`files/patch_ple_layer.py:185–236`).
4. **`EXTRA_DOCKER_ARGS` é descartado silenciosamente no launcher de 2 nós.** `start.sh:959–998` monta um array `DOCKER_ARGS` (incluindo `EXTRA_DOCKER_ARGS`), mas os heredocs que efetivamente rodam `docker run` no worker (`start.sh:1084–1116`) e no head (`start.sh:1146–1179`) têm a lista de flags escrita à mão e **não referenciam `DOCKER_ARGS` nem `EXTRA_DOCKER_ARGS`**. O relatório fable (§3.5) recomenda passar env NCCL via `EXTRA_DOCKER_ARGS` — isso não teria efeito no `start.sh` atual. (No `start-tp1.sh`/`tp1/start.sh`, `$EXTRA_DOCKER_ARGS` é usado no heredoc — L353.) Este é o mesmo padrão de bug que o fable report encontrou para `VLLM_ARGS` (§4.1), já corrigido para `VLLM_ARGS` mas não para docker args.
5. **`FP8_DENSE=true` e `QSA_PROFILE≠stock` provavelmente falham num clone limpo**: `start.sh:579` roda `python3 files/overlay/apply_patches.py` se `files/overlay/modelopt.py` não existir, mas `apply_patches.py` **não está no repo** — `.gitignore:50` ignora `files/overlay/*.py` (o que também engoliu o gerador). Só os `.diff` estão versionados. Idem `files/qsa_gb10/apply_patch.py` lê `files/qsa_gb10/qsa.py.orig`, que `start.sh` não extrai da imagem (e `*.orig` está no `.gitignore:43`). Para usar essas lanes é preciso extrair/aplicar manualmente (ex.: `docker cp` do arquivo original + `patch -p0 < *.diff`). Isso é inferência a partir do código; o repo não documenta.
6. **Defaults do `start.sh` ≠ `.env.sample` para vários knobs** (o `start.sh` usa fallback quando a chave está ausente do `.env`): `MAMBA_SSM_CACHE_DTYPE` (sample `bfloat16`, fallback vazio = float32), `MTP_DRAFT_VOCAB` (sample 47k, fallback vazio = vocabulário completo), `MTP_DISABLE_BLOCK_DROP` (sample `1`, fallback `0`), `MTP_INDEX_SHARE` (sample `true`, fallback `false`). Se você escrever um `.env` do zero sem essas chaves, perde quatro otimizações.
7. **Números de KV citados com configurações diferentes.** O "3,652,200 tokens" (README) foi medido com GMU 0.835, fp8 KV, e (aparentemente) SSM em float32. Com `MAMBA_SSM_CACHE_DTYPE=bfloat16` os runs de 2026-09-10/16 registram 4,238,238–4,378,829 tokens no mesmo GMU. O README não explicita o dtype do SSM na medição de 3.65M; a diferença é consistente com o estado GDN menor liberando espaço (inferência minha).
8. **Tamanho do checkpoint nvidia aparece como 124 GiB, ~126 GiB, 133 GB, ~133 GiB, 135 GB** em lugares diferentes (README:26, 361, 656, 259; `.env.sample:24`; README:1006). São unidades misturadas (GB vs GiB) e revisões diferentes. Planeje ~126–135 GB de disco por nó.
9. **Tamanho do bloco de atenção**: README:380–381/483 diz 1600 tokens; `files/patch_block_drop.py:7` diz "1,664 tokens at MTP 3" (texto vindo do kit single-Spark com SSM bf16); CHANGELOG:103 testou `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1664`. O valor exato depende de KV dtype × SSM dtype × TP (ver conceito "hybrid KV cache allocator" em §5).
10. **Handoff single-Spark diz "FP8 KV cache is NOT supported"** (HANDOFF:71, tp1/README:21) — isso é anterior ao patch `patch_qsa_fp8_kv.py`; o CHANGELOG:454–458 corrige: não é limitação do checkpoint, é do kernel stock.

---

## 1. Visão geral (o que esta parte da receita faz, ponta a ponta)

A receita serve o modelo **Qwen3.8-Flash-Next** (checkpoint default `nvidia/Qwen3.8-Flash-Next-NVFP4`) em **dois DGX Spark** (cada um: 1 GPU GB10, `sm_121`, 48 SMs, 121.69 GiB de memória **unificada** LPDDR5x ~273 GB/s teóricos, 20 núcleos ARM) ligados por **ConnectX-7 RoCE v2 200 GbE**. O motor é o **vLLM** (imagem day-0 `vllm/vllm-openai:qwen38-flash-next`, vLLM `0.1.dev20073+g8e685d198`, V1 engine, V2 model runner, FlashInfer 0.6.17, torch 2.13+cu130, Triton 3.7.1, NCCL 2.30.7 — fable §0) ou, opcionalmente, o **vLLM 0.30.0** stock.

Arquitetura do modelo (extraída de README, fable report e dos patches — o repo não publica o `config.json` completo):
- 48 camadas de decoder, **híbridas**: `full_attention_interval: 4` → **12 camadas de atenção "full"** (que na verdade usam **QSA**, uma atenção **esparsa** com indexador e top-k de 2048 tokens) + **36 camadas Gated-DeltaNet (GDN)**, atenção linear com estado recorrente constante por requisição.
- Atenção: `num_key_value_heads: 2`, `head_dim: 256`, 24 cabeças de query (GQA; `bench_qsa_kernels.py:44–45` usa `24 // tp` e `max(1, 2 // tp)`), saída com gate (`output_gate_ptr`).
- **MoE** em todas as 48 camadas: 512 experts roteados, top-10, + shared expert (fable §1). Intermediate do expert 640 (fable §3.6: "intermediate 640 → 320 per GPU" se TP-sharded).
- **HyperConnections** (2 por camada + um mixer final), projeções low-rank rank 320 (fable §3.1).
- **Tabela PLE/N-gram** (~51 GB em FP8, "51B-param", na camada de índice 1): embedding de n-gramas por hash com dezenas de cabeças de tabelas de tamanho primo (`Qwen3_8FlashNextNGramEmbedding`, `files/ple_layer_patched.py:409–480`).
- **1 camada MTP** (Multi-Token Prediction) usada como drafter de speculative decoding, com o **próprio `lm_head` BF16** (vocabulário 248,320 × hidden 2560; `tie_word_embeddings` = false).
- Torre de **visão** (imagem + vídeo).

Fluxo do `start.sh` (default: `./start.sh` ou `./start.sh --no-download`):
1. **Download** (opcional) no head: `download.sh` → `huggingface_hub`/`hf download` para `~/.cache/huggingface/hub`.
2. **Resolve cache** e valida que o snapshot está completo (`files/resolve_snapshot.py`: todos os shards do `model.safetensors.index.json` existem); normaliza `refs/main` (bug de newline do hf_hub 1.28).
3. **Detecta dtype da PLE** (`files/detect_ple_dtype.py`) para injetar via `--hf-overrides`.
4. **Distribui pesos ao worker**: `rsync` (default) ou **NFS** (`NFS_SHARE=true`/`--nfs`).
5. **Verificação** de presença (e opcionalmente hash via `check-weights.sh --verify`).
6. **Preflight**: GPUs ociosas nos 2 nós (`REQUIRE_IDLE_GPU`); **despejo do page cache do próprio checkpoint** via `posix_fadvise(DONTNEED)` (`EVICT_PAGE_CACHE`).
7. **Overlays** (arquivos Python extraídos da imagem, patchados e bind-montados `:ro`, sem rebuild): FP8-dense, QSA profile, draft vocab reduzido, **FP8 KV** para QSA, alias MTP `mtp.layers.48`, backport block-drop, determinismo, **PLE**, **MXFP8 fallback** + **FP8_BLOCK_SCALES MoE**.
8. **Imagem Docker** garantida nos 2 nós.
9. **Renderiza `VLLM_ARGS`** (um único array compartilhado) e lança **worker (rank 1, `--headless`) primeiro**, espera 15 s, depois **head (rank 0)** que serve a API OpenAI em `0.0.0.0:$PORT` (8888).
10. Segue logs e faz polling de `/health` até 200. Cold start medido ≈ **10m55s**.

Lanes alternativas: `start-fp8.sh` (checkpoint oficial FP8, contexto nativo 262K, ~500k tokens de KV), `start-v030.sh` (vLLM 0.30.0 stock, BF16 KV, só 1 overlay), `FP8_DENSE=true` (checkpoint híbrido NVFP4 experts + FP8 denso, experimental e não medido em GPU), `start-tp1.sh`/`tp1/` (um único Spark, TP=1, checkpoint menor, PLE com offload mmap).

---

## 2. Inventário de parâmetros

### 2.1 Precedência de configuração (fundamental)

| Regra | Onde | O que acontece | Por que importa |
|---|---|---|---|
| `.env` **vence** o ambiente | `start.sh:51–52` (`source .env` depois de ler env) | `KV_CACHE_DTYPE=fp8 ./start.sh` é ignorado se `.env` define a chave | README Gotcha L988–992: custou um ciclo inteiro de launch; confira a linha `KV dtype:` no resumo do launch |
| Exceção: `ABLIT` e `HF_TOKEN` | `start.sh:43–59`, `download.sh:47–57`, `check-weights.sh:78–86` | capturados em `_CLI_ABLIT`/`_CLI_HF_TOKEN` antes do `source .env` e reaplicados | `ABLIT=1 ./start.sh` funciona mesmo com `ABLIT=0` no `.env` |
| `OVERRIDE_*` (lidos do ambiente depois do `.env`) | `start.sh:109–123` | `OVERRIDE_MODEL_ID`, `OVERRIDE_SERVED_MODEL_NAME`, `OVERRIDE_MAX_MODEL_LEN`, `OVERRIDE_YARN_ENABLE`, `OVERRIDE_IMAGE`, `OVERRIDE_KV_CACHE_DTYPE`, `OVERRIDE_GPU_MEMORY_UTILIZATION` | é como `start-fp8.sh` e `start-v030.sh` reaproveitam o mesmo `start.sh` |
| Ordem de seleção de checkpoint | `start.sh:129–145` | `FP8_DENSE=true` > `OVERRIDE_MODEL_ID` > `ABLIT=1` > `MODEL_ID` | evita servir o ablit com checkpoint incompatível |
| `tp1/start.sh`: env > `tp1/.env` > default | `tp1/start.sh:64` e diff vs `start-tp1.sh` | no `start-tp1.sh` da raiz, os valores do `.env` de 2 nós são **deliberadamente sobrescritos** pelos defaults TP1 | `start-tp1.sh` usa o `.env` de 2 nós só para `IMAGE`, etc. |

### 2.2 Rede do cluster e fabric (RDMA)

| Nome | Default (sample) / alternativos | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `HEAD_IP` | `10.0.0.1` (live igual) | `.env.sample:8`; usado em `--master-addr`, `VLLM_HOST_IP` do head | IP inter-nó do head (rank 0) | rendezvous de processos distribuídos | Obrigatório (`start.sh:64`). No kit, `10.0.0.x` é um alias; o NFS **não** deve usar `10.0.0.1` (README:172) |
| `WORKER_IP` | `10.0.0.2` | `.env.sample:9`; `VLLM_HOST_IP` do worker, ssh/scp/rsync | IP do worker (rank 1) | idem | Obrigatório |
| `WORKER_USER` | vazio (live `zurih`) | `.env.sample:10`; `start.sh:217–223` | usuário SSH do worker | SSH sem senha | pré-requisito (README:24) |
| `IFACE` | `enp1s0f0np0` (live `enp1s0f1np1`) | `.env.sample:15`; `GLOO_/NCCL_/TP_SOCKET_IFNAME` do head | interface Ethernet do link ConnectX no head | bootstrap TCP do NCCL/Gloo; seleção de NIC | Deve ser a interface do cabo entre os Sparks; `ip -o -4 addr` |
| `WORKER_IFACE` | não setado → `IFACE` (live `enp1s0f0np0`) | `start.sh:82`; heredoc do worker L1091–1093 | interface do lado do worker | cabeamento cruzado (f1↔f0) | README Gotcha L982: nós cross-wired, head `f1`, worker `f0` |
| `IB_HCA` | `=rocep1s0f0` (live `=rocep1s0f1`) | `.env.sample:20`; `NCCL_IB_HCA` | HCA RDMA que o NCCL usa; `=` inicial = **match exato, um device** | RDMA/RoCE, HCA, NCCL IB transport | README Gotcha L978: **nunca** apontar para porta cabeada a outro cluster → NCCL trava no init sem erro útil |
| `WORKER_IB_HCA` | não setado → `IB_HCA` (live `=rocep1s0f0`) | `start.sh:83`; heredoc worker L1095 | HCA do worker | idem | cross-wired |
| `IB_GID_INDEX` | `3` (alternativa `5`) | `.env.sample:21`; `NCCL_IB_GID_INDEX` | índice da GID RoCE (v2/IPv4) | GID table do RoCE | README Gotcha L980: se 3 não sobe, tente 5 |
| `MASTER_PORT` | `50000` | `.env.sample:177`; `--master-port` | porta de coordenação distribuída | torch.distributed rendezvous | obrigatório |
| `NCCL_IB_DISABLE=0` | fixo | `start.sh:968`, heredocs | habilita transporte IB/RoCE | NCCL net transports | sem isso NCCL cairia para socket TCP |
| `NCCL_IB_AUTO_DETECT=0` | fixo | heredocs L1097, L1159 | desliga autodetecção de HCA | NCCL | força o HCA exato |
| `NCCL_DEBUG=WARN` | fixo | heredocs | verbosidade | debug | fable §4.4: adicionar `NCCL_DEBUG_SUBSYS=INIT,NET` após mudanças de fabric (não adotado) |
| `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` | `=$IFACE` / `=$WORKER_IFACE` | heredocs | NIC para sockets de controle | Gloo (CPU collectives), bootstrap NCCL | se cair na NIC errada: hang ou tráfego lento |
| `VLLM_HOST_IP` | `$HEAD_IP` / `$WORKER_IP` | heredocs L1101, L1163 | IP que cada nó anuncia ao vLLM | plano de controle ZMQ do vLLM | fable §3.5: o broadcast scheduler→worker a cada step já passa pelo CX7 (~0.15–0.4 ms RTT) |

Knobs de fabric **sugeridos e não adotados** (fable §3.5): `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_NET=IB`, `NCCL_NET_PLUGIN=none`, `NCCL_PROTO=LL|LL128`, `NCCL_ALGO=Ring`, `NCCL_IB_QPS_PER_CONNECTION=2..4`, `NCCL_IB_SPLIT_DATA_ON_QPS=0`, `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_IGNORE_CPU_AFFINITY=1`, **segundo link CX7** (`NCCL_IB_HCA="=rocep1s0f1,roceP2p1s0f1"` + `NCCL_CROSS_NIC=1`, "dobra a banda cross-node, ~2× comunicação de prefill, sem mudança de latência de decode"), GPUDirect RDMA (`NCCL_DMABUF_ENABLE=1`, `NCCL_NET_GDR_LEVEL=SYS`; `nvidia_peermem` não está carregado). Estimativa: +3–8% decode. Motivo: ~106 coletivas cross-node pequenas (5–20 KB) por step de decode, 20–30 µs cada → 2–3 ms/step (6–8%), latência-bound e **não diminui com batch**.

### 2.3 Modelo / checkpoint

| Nome | Default / alternativos | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `MODEL_ID` | `nvidia/Qwen3.8-Flash-Next-NVFP4` (133G/11 shards: BF16 denso + NVFP4 experts + FP8 PLE). Alternativas comentadas: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (99G/37 shards: MXFP8 atenção + NVFP4 PLE), `RadixArk/Qwen3.8-Flash-Next-NVFP4` (126G/206 shards: BF16 atenção + FP8 PLE) | `.env.sample:24–26` | repositório HF a servir | checkpoint quantizado misto | Cada checkpoint grava metadados de quantização de jeito diferente → patches diferentes necessários (ver §3). HANDOFF §2: RadixArk é ~13–16% mais lento em decode (mais bytes), prefill idêntico |
| `ABLIT` | `0` / `1` | `.env.sample:27`; `start.sh:54–60,137–145,205–209` | `1` serve o checkpoint gated `drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47` | abliteration (remoção de direção de recusa) | Mesmo layout do nvidia NVFP4 com "house residual-writer projection" em 12 tensores `self_attn.o_proj` das camadas QSA 3,7,11,…,47. GDN, experts, MTP, PLE, visão e chat template intactos. Só válido nesta receita (não com MXFP8 single-Spark, FP8 oficial ou FP8_DENSE). Ignorado se `FP8_DENSE=true` ou `OVERRIDE_MODEL_ID` |
| `HF_TOKEN` | não setado | `.env.sample:32`; `download.sh:52–54` | autentica repo gated | HF gated repos | `ABLIT=1 ./download.sh` falha imediatamente sem token; 403 = termos não aceitos |
| `PLE_EMBEDDING_DTYPE` | vazio (auto) | `.env.sample:41` (comentado); `start.sh:356–362` | força o dtype da tabela PLE; vazio = detectar com `detect_ple_dtype.py` | metadados de quantização | nvidia declara PLE FP8 só em `quantization_config.config_groups`; sem o override a tabela de 51 GB não carrega |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` (FP8: `qwen3.8-flash-next-fp8`) | `.env.sample:42`; `--served-model-name` | nome em `/v1/models` | API OpenAI | clientes usam este nome em `"model"` |
| `IMAGE` | `vllm/vllm-openai:qwen38-flash-next` (day-0, `sha256:d464f3b4…`, ~20.6 GB, "NVIDIA dev build") / `vllm/vllm-openai:v0.30.0` | `.env.sample:98`; `start.sh:781–799` | imagem Docker | container de inferência | day-0 = código do modelo em `vllm/models/qwen3_8_flash_next/nvidia/`; v0.30 = `vllm/models/qwen4_exp/nvidia/` (classes `Qwen4Exp*`) |
| `HF_HOME` / `WORKER_HF_HOME` / `WORKER_MODEL_PATH` | `~/.cache/huggingface` | `.env.sample:107–108`; `start.sh:237,368–376`; `check-weights.sh:170–182` | raiz do cache HF de cada nó | layout HF hub | caminho do worker espelha o do head, reescrevendo o prefixo `$HOME` |
| `HF_API_BASE` | `https://huggingface.co` | `verify-weights.py:55`; `check-weights.sh:287` | endpoint do manifest (mirror) | — | nós sem rota para HF |

### 2.4 Parâmetros de serving (scheduler / memória / contexto)

| Nome | Default / alternativos | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `MAX_MODEL_LEN` → `--max-model-len` | `262144` (nativo). Alternativas: `1000000` (com YaRN), `512000`, `65536` (TP1 default) | `.env.sample:45`; `start.sh:76–79` (deve ser inteiro positivo), L882 | contexto máximo por requisição | contexto, RoPE, KV cache | Define tamanho de estruturas por step mesmo para prompts curtos: fable §3.4 — o scorer QSA aloca buffer fp32 `[rows, page_table_width×page_size]` por camada; a 1M e compress ratio 4 são 250K colunas → 4 MB por verify pass por camada, `persistent_topk` lançado sobre 250K colunas; block tables de 625 entradas. Recomenda 262144 para agentic (+2–5% esperado) |
| `YARN_ENABLE` | `false` / `true` | `.env.sample:46`; `start.sh:166–170` (força `false` se `MAX_MODEL_LEN ≤ 262144`, imprime NOTE) | liga RoPE scaling YaRN | YaRN, extrapolação de contexto | "YaRN exists to extend context beyond native… costs accuracy" (README:836). Antes de set/2026 era **no-op silencioso** (ver §3 bug) → 1M **não validado** |
| `YARN_FACTOR` | `4.0` | `.env.sample:52`; `start.sh:936–940` | fator de escala | 262144 × 4 ≈ 1M | emitido como `{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}` dentro de `text_config.rope_parameters` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | `.env.sample:176`; heredocs L1102/L1164 | permite `max_model_len` > `max_position_embeddings` | guard do vLLM | loga "must be used with extreme caution". Fable §4.6: fica setado mesmo com YaRN desligado (inofensivo) |
| `GPU_MEMORY_UTILIZATION` → `--gpu-memory-utilization` | `0.80` (desde 2026-09-26; antes `0.835`; histórico `0.85`) | `.env.sample:53–56`; `start.sh:879` | fração dos 121.69 GiB que o vLLM pode orçar (pesos+ativação+grafos+KV) | GMU, profiling run, memória unificada | CHANGELOG 2026-09-26: a 0.835 o driver segurava 114.5 GiB/nó e um soak de 6 workers deixou `MemAvailable` em 0.3–0.9 GiB, zona em que GB10 **dá hard-reset**. A 0.80 (lane v0.30) sobraram 4.5 GiB (head) / 7.8 GiB (worker) em 1 h. KV encolhe ~4 GiB/nó; **decode não depende do GMU**. Regra: +0.01 GMU ≈ +0.83 GiB ≈ +95k tokens fp8 |
| `MAX_NUM_SEQS` → `--max-num-seqs` | `8` (TP1: `4`) | `.env.sample:57`; `start.sh:880` | máximo de sequências concorrentes no batch | continuous batching, scheduler | Com 13.93× de KV a 262K, o **scheduler** (8), não o cache, é o limite. Também define captura de CUDA graphs (1…64 tokens = 8 × (1+3) × 2, fable §1) |
| `MAX_NUM_BATCHED_TOKENS` → `--max-num-batched-tokens` | `8192` (TP1: `2048`) | `.env.sample:58`; `start.sh:881` | tokens por step de engine (chunk de prefill) | chunked prefill, token budget | fable §5: "razoável; chunks maiores não reduzem bytes de all-reduce". Também explica o bug #62: prompt que cabe num único chunk de 8192 erra o cache no 1º repeat |
| `PORT` | `8888` | `.env.sample:59`; `--port` só no head | porta da API (`--network host`) | servidor OpenAI-compatible | TP1: cuidado com `comfy-h3.service` que escuta 8888 (HANDOFF §0) |
| `--enable-chunked-prefill` | sempre | `start.sh:887` | quebra prefill longo em chunks intercalados com decode | chunked prefill | necessário a 262K–1M |
| `--enable-prompt-tokens-details` | sempre | `start.sh:876` (CHANGELOG 2026-09-25, #63) | expõe `usage.prompt_tokens_details.cached_tokens` | observabilidade do prefix cache | `docs/cache-reporting.md`: só metadado; não liga/retuna prefix caching; em streaming use `"stream_options":{"include_usage":true}`. Teste: `tests/test_prompt_token_details.py` |
| `--load-format safetensors` + `--safetensors-load-strategy lazy` | sempre | `start.sh:885–886` | carrega shards sob demanda | safetensors, mmap | Carga de 458 s (target 392 + drafter 67) medida com lazy (README:487); com NFS o worker streama ~126 GiB pelo CX7 |
| `--distributed-executor-backend mp` | sempre | `start.sh:891` | executor multiprocess (não Ray) | executores do vLLM | também é o único que spawna o worker de PLE offload (HANDOFF §4); auto-habilita async scheduling com `mtp` (fable §1) |
| `--reasoning-parser qwen3` | sempre | `start.sh:888` | separa `<think>…</think>` em campo de reasoning | reasoning models | este build streama em `delta["reasoning"]`, **não** `reasoning_content` (HANDOFF §3, `bench/longctx.py:9–11`) |
| `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` | sempre | `start.sh:889–890` | tool calling OpenAI | function calling, parsers | regressão "tool calling" passa com MTP (CHANGELOG:490) |
| `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'` | fixo (TP1: `CUDAGRAPH_MODE` configurável, `NONE` p/ debug) | `start.sh:917`; `start-tp1.sh:113,314` | sem torch.compile/inductor (mode 0); CUDA graphs completos só para batches de decode | CUDA graphs, torch.compile | Captura 1–64, 0.39 GiB (README:484). Fable §3.6 sugere testar `mode 3` + `FULL_AND_PIECEWISE` (0–8%, +2–5 min de cold start) e `--performance-mode interactivity` (captura todo tamanho 1…32, 0–2%) — não adotados |
| `--kv-cache-memory <bytes>` | não emitido (via `EXTRA_VLLM_ARGS`) | README:358, 393–401, 411 | fixa o pool de KV em bytes, ignorando o GMU para o KV | orçamento de KV | valores do log: `33651431445` (31.34 GiB "strict fit"), `41360080896` (38.52 GiB "fully utilize"), `44208844800` (41.17 GiB, run MTP-off), `45502283776` (42.38 GiB → ≈3.0M tokens p/ 1M×3) |
| `EXTRA_VLLM_ARGS` | vazio | `.env.sample:174`; `start.sh:950–952` | anexado **por último, verbatim** (quote JSON com aspas simples como na shell) | escape hatch | É re-parseado pela shell no script renderizado; `.last_head_launch.sh` (0600) guarda o comando para inspeção (pode conter `--api-key`) |
| `EXTRA_DOCKER_ARGS` | vazio | `.env.sample:175`; `start.sh:995–998` | deveria anexar flags ao `docker run` | — | **Bug**: não chega aos heredocs (ver §0.4) |

### 2.5 KV cache e estado SSM

| Nome | Default / alternativos | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` → `--kv-cache-dtype` | `fp8` (default; fallback do start.sh também `fp8`) / `auto` (=bf16). Lane v0.30: `auto` default, `fp8` opt-in via `OVERRIDE_KV_CACHE_DTYPE=fp8` | `.env.sample:60–69`; `start.sh:87,665–680,883` | dtype do KV principal das 12 camadas QSA (+MTP) | KV cache quantizado FP8 E4M3, escala por tensor | Kernels QSA stock só aceitam `["auto","bfloat16"]` e levantam `Qwen3.8-Flash-Next QSA requires a BF16 main KV cache` → `start.sh` aplica `files/patch_qsa_fp8_kv.py` automaticamente quando o valor começa com `fp8`. Ganho: 3,652,200 vs 2,131,159 tokens (1.70×/GiB). Custo (rerun 09-16): −5.3% throughput médio (−1.4% prose, −9.2% code), `tok/step` cai em 8/8 células (indexador esparso escolhe blocos diferentes), `ms/step` +11% a S=8. Não passe a flag à mão: use o `.env` para o patch ser aplicado |
| `MAMBA_SSM_CACHE_DTYPE` → `--mamba-ssm-cache-dtype` | `bfloat16` (sample) / vazio = float32 do checkpoint (fallback do start.sh) | `.env.sample:70–82`; `start.sh:93,884` | dtype do estado recorrente das 36 camadas GDN | estado SSM/linear attention; hybrid allocator | O kernel fused GDN aceita bf16 (`FUSED_GDN_STATE_DTYPES` em `qwen_gdn_linear_attn.py`). Metade dos ~0.23 GB/sequência movidos por step; metade da página mamba → vLLM escolhe bloco de atenção menor (single-Spark: 3,200 → 1,664 tokens). Single-Spark mediu +8.5% agregado a 8 streams (151.6→164.5), +6.8% a 1, step 141.4→130.4 ms, needles 15/15 a 32k. **Não medido isoladamente no kit dual**. Fable §3.7 (anterior) achava <1% e "risco de precisão do estado" |
| Bloco de atenção | 1600 tokens (reportado) | README:380–381, 483 | tamanho da página de KV | PagedAttention + hybrid allocator | vLLM eleva o bloco de atenção para que a página de atenção ≥ página Mamba/GDN. Consequência: prefix cache só acerta em fronteiras de 1600 tokens (modo mamba `align`, fable §4.6) |
| Prefix caching | ligado (default do vLLM) | README:482 | reuso de blocos de KV entre requisições | prefix caching, hashing de blocos | Prefix cache de prompt de 600k: 336 s frio → 2.87–8.82 s quente (HANDOFF §3). Modo mamba forçado a `align` pelo modelo (`"all"` levanta erro) |

### 2.6 Paralelismo

| Nome | Default | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `TENSOR_PARALLEL_SIZE` → `--tensor-parallel-size` | `2` (1 GPU/nó × 2 nós) | `.env.sample:85`; `start.sh:878` | TP=2 atravessando os nós | tensor parallelism, all-reduce | cada camada vira colunas/linhas partidas; ~106 coletivas por step de decode (fable §7.1): 36 GDN `out_proj` AR + 12 atenção `o_proj` AR + 48 MoE AR (20 KB cada) + 1 all-gather de logits (4 × 124160 × fp32 ≈ 2 MB) + 9 dos drafts |
| `--nnodes 2`, `--node-rank 0/1`, `--master-addr $HEAD_IP`, `--master-port $MASTER_PORT` | fixos | `start.sh:893–895,1114,1176` | multi-nó nativo do vLLM (sem Ray) | rendezvous multi-nó | worker precisa subir antes (espera 15 s) |
| `--headless` | só worker | `start.sh:1115` | worker não serve API | arquitetura head/worker | só o rank 0 tem API server |
| `ENABLE_EXPERT_PARALLEL` → `--enable-expert-parallel` | `true` ("required for NVFP4") | `.env.sample:86`; `start.sh:897–900` | cada GPU guarda 256 dos 512 experts | expert parallelism | Fable §1: `--all2all-backend allgather_reducescatter` **nunca é invocado** (`use_all2all_kernels` é falso com DP=1, PCP=1): tokens são replicados por TP, cada rank computa seus experts locais e o resultado vai pelo all-reduce normal de TP. **EP aqui só muda quais pesos cada GPU guarda, não a coletiva** |
| `--all2all-backend allgather_reducescatter` | junto com EP | `start.sh:899` | backend de dispatch/combine de MoE | all-to-all MoE | inerte com DP=1 (ver acima). DeepEP low-latency não ajuda (fable §3.7) |
| `MM_ENCODER_TP_MODE` → `--mm-encoder-tp-mode` | `data` / `weights` | `.env.sample:88–92`; `start.sh:97,892` | `data` replica o encoder de visão em cada GPU; `weights` o shardaria | TP de encoder multimodal | Intermediate do MLP de visão = 4304; 4304/2 = 2152 não é divisível por 16 exigido pelos kernels NVFP4 → `weights` **crasha na carga** |

### 2.7 Speculative decoding (MTP)

| Nome | Default / alternativos | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `MTP_NUM_SPECULATIVE_TOKENS` → `--speculative-config '{"method":"mtp","num_speculative_tokens":N,...}'` | `3` / `0` desliga (não emite a flag), `1` é o "ponto de comparação seguro", `2`/`4` sugeridos para teste | `.env.sample:95`; `start.sh:86,903–915` | K tokens de draft por step, verificados pelo modelo alvo | MTP, speculative decoding | Batch-1 greedy: 24.5 → **52.1 tok/s (2.13×)**, aceitação 72.8% (89/74.5/60% por posição). K>1 loga "running multiple times of forward on same MTP layer, which may result in lower acceptance rate"; QSA reconstrói metadados por draft step. MTP 4 não testado por risco de memória (head com ~1.9 GiB `MemAvailable`, captura de grafo maior = risco de hard-reset). Fable §3.2: se aceitação da posição 3 < ~35%, K=2 é mais rápido |
| `use_local_argmax_reduction: true` | adicionado quando `MTP_DRAFT_VOCAB` setado | `start.sh:907–911` | drafter usa `get_top_tokens` (argmax local por rank + all-gather `(valor,índice)`) | vocab-parallel argmax | único caminho que chega ao head reduzido; corta o all-gather de draft de O(vocab) para O(2·tp_size) por token |
| `MTP_DRAFT_VOCAB` | `files/draft_vocab_en_code_47k.txt` (47,149 ids; default desde 2026-09-10) / vazio = 248,320 | `.env.sample:130`; `start.sh:151–154,623–657` | fatia o `lm_head` do drafter para um subconjunto de ids frequentes | FR-Spec, reduced-vocab drafting | lm_head do drafter = 248320×2560×2 B = 1.18 GiB (0.59 GiB/GPU), lido 3× por step de MTP=3 (3 das 4 leituras de lm_head). Log: "47149 of 248320 tokens (19.0%), 45734 on this rank of 2; draft lm_head shard 0.59 -> 0.22 GiB". +9.6% médio (09-11), +8.4% (09-16). **Não muda a saída** (draft errado é rejeitado). Tuned em código inglês: chinês pode draftar pior. Paths relativos são resolvidos contra a raiz do repo (bind-mount precisa de absoluto). Incompatível com `FP8_DENSE` (ambos sobrepõem `nvidia/mtp.py`) |
| `MTP_DISABLE_BLOCK_DROP` → `"disable_eagle_block_drop":true` | `1` (sample) / `0` (fallback) | `.env.sample:135`; `start.sh:157,740–751,905` | não descarta o último bloco casado do prefix cache | prefix caching + EAGLE | Backport vllm#53388. Corrige #62 (2º turno com zero hit): TTFT turno 1 de sessão 9.5k **3.25 → 0.63 s**; turno 2 1.12 → 0.59 s; após tool output 4k 2.55 → 1.98 s. Decode/prefill/NLL inalterados |
| `MTP_INDEX_SHARE` → `"index_share_for_mtp_iteration":true` | `true` (sample) / `false` (fallback) | `.env.sample:139`; `start.sh:158,906` | reusa o top-k do indexador QSA do 1º draft step nos seguintes | sparse attention + MTP | Só drafter. Code +6.1/+4.6/+21.3/+26.6% (S=1/2/4/8), prose ruído. HumanEval 156/164. Índices ficam 1 posição "velhos" (fable §3.2) |
| `VLLM_QSA_DET_TOPK=1` + `VLLM_MOE_DET_FINALIZE=1` | não setados (opt-in) | `.env.sample:143–144`; `start.sh:159–160,752–776` | top-k do QSA ordenado + finalize não-fundido da MoE FlashInfer (+ `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0` + cache de autotune separado `/root/.cache/vllm/flashinfer_autotune_cache_unfused`) | determinismo numérico | Spread de logprob de prompt median 0.25 / max 6.07 nats → **0/0**; saídas greedy 5/5 distintas → 1/5. Custo ~−4% decode prose e prefill. Não portado para v0.30 |

### 2.8 PLE (tabela N-gram) e offload

| Nome | Default | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `PLE_OFFLOAD` → `VLLM_PLE_CPU_OFFLOAD=1` | `false` (2 nós) / `true` obrigatório (TP1) | `.env.sample:101`; `start.sh:980–982,1077–1080`; `start-tp1.sh:106,131,340` | move a tabela PLE para um processo CPU | offload, memória unificada | Gotcha README:971: precisa ~51 GB de RAM livre; o log mostrou `Available RAM: 44.92 GiB` → OOM/thrash. Em memória unificada "CPU RAM" é o mesmo pool: não libera nada (fable §3.7), só reduz pressão de UVM na carga (HANDOFF §4) |
| `SKIP_PLE_PATCH` | `false` (`true` em start-fp8/start-v030) | `start.sh:125,813–833` | pula o patch do `ple_layer.py` | — | FP8 oficial é FP8 fine-grained no checkpoint todo; v0.30 carrega nativamente |
| `VLLM_PLE_PACKED_TABLE_DIR`, `VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300` | TP1 | `start-tp1.sh:341–342` | tabela PLE empacotada mmap e timeout do handshake | mmap / page cache | TP1 apenas |

### 2.9 Experimentos de banda de decode

| Nome | Default / alternativos | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `FP8_DENSE` | `false` / `true` | `.env.sample:152`; `start.sh:129–134,576–585` | serve o checkpoint híbrido `FP8_DENSE_MODEL_ID` (NVFP4 experts + projeções densas FP8 por canal) e monta 4 overlays | quantização FP8 per-channel/per-token dinâmica | ≈9.2 → ≈5.7 GiB por step, ~1.5× batch-1 **analítico, não medido em GPU**. Construído a partir de `RadixArk/...`. Desliga download (`DO_DOWNLOAD_DEFAULT=false`). Ver §0.5 (gerador ausente) |
| `FP8_DENSE_MODEL_ID` | `MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense` | `.env.sample:153`; `start.sh:130` | id local (nunca no Hub) | — | construído por `files/fp8dense/build.sh` |
| `QSA_PROFILE` → `VLLM_QSA_PROFILE` / `VLLM_QSA_PROFILE_JSON` | `stock` / `gb10` / caminho JSON | `.env.sample:158`; `start.sh:156,586–597` | tabela de launch (block_n, splits, warps; tiles MQA) dos kernels Triton QSA | tuning de kernels, split-K, occupancy | Tabelas stock "tuned on GB300 (160 SMs)"; GB10 tem 48 SMs → decode (16, 64 splits, 4 warps) lança ~5× mais CTAs que cabem e paga merge de 64 parciais. `gb10` = (32,16,4) etc. **Não benchmarkado**. Incompatível com FP8 KV e com `VLLM_QSA_DET_TOPK` (todos sobrepõem `ops/qsa.py`) |

### 2.10 Operação, segurança e distribuição de pesos

| Nome | Default | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `REQUIRE_IDLE_GPU` | `true` | `.env.sample:160`; `start.sh:470–483` | aborta se `nvidia-smi --query-compute-apps` mostra processo em qualquer nó | co-tenancy de GPU | fable §4.3: senão o sintoma é CUDA OOM 10 min depois; o worker estava rodando DeepSeek-V4 dspark |
| `EVICT_PAGE_CACHE` | `true` | `.env.sample:171`; `start.sh:502–540` | `posix_fadvise(POSIX_FADV_DONTNEED)` nos `.safetensors/.bin/.pt/.gguf` do checkpoint, nos 2 nós (via ssh ou container descartável no modo NFS) | page cache em UMA | issue #35: páginas limpas do checkpoint (download/rsync/NFS) competem com o modelo; medido 4.15 → 0.00 GiB (local) e 3.00 → 0.00 GiB (NFS 4.2, via `mincore`). Sem root, best-effort |
| `NFS_SHARE` | `false` (live do autor `true`) / `--nfs` / `--no-nfs` | `.env.sample:186`; `start.sh:104,387–437` | head exporta o cache HF via NFSv4 (container privilegiado `vllm-fn-nfs`); worker monta volume Docker `vllm-fn-hf` read-only | NFS vs rsync | rsync: ~126 GiB no worker, nós desacoplados. NFS: 0 disco no worker, mas **head precisa ficar de pé**, cada cold start streama ~126 GiB; ótimo para trocar checkpoints. Fable §4.6: rsync corta o cold start (458 s eram streaming NFS) |
| `NFS_SERVER_IP` | auto do `IFACE` (live `10.0.22.1`) | `.env.sample:189`; `files/nfs-share.sh:13–19` | endereço do export | — | não usar alias `10.0.0.1` |
| `NFS_IMAGE`/`NFS_CONTAINER`/`NFS_VOLUME` | `vllm-fn-nfs:local` / `vllm-fn-nfs` / `vllm-fn-hf` | `files/nfs-share.sh:8–10` | — | — | opções de montagem: `nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600`; export `ro,sync,no_subtree_check,no_root_squash,insecure,fsid=0`; `rpc.nfsd 8` |
| `DO_DOWNLOAD`/`DO_SYNC`/`DO_LAUNCH` | via flags | `start.sh:175–203` | controle de etapas | — | ver §2.11 |

### 2.11 Flags de linha de comando dos scripts

| Script / flag | Efeito | Onde |
|---|---|---|
| `start.sh` (sem flag) | download no head → sync → patch → launch | `start.sh:14–22` |
| `--no-download` | pula download (pesos já no head) | `start.sh:181` |
| `--no-launch` | só download + distribuição | `start.sh:182` |
| `--launch` | pula download **e** sync; aplica patches e lança | `start.sh:183` |
| `--nfs` / `--no-nfs` | força NFS / força rsync | `start.sh:184–185` |
| `start-fp8.sh [flags]` | exporta `OVERRIDE_MODEL_ID=Qwen/Qwen3.8-Flash-Next-FP8`, `OVERRIDE_SERVED_MODEL_NAME=qwen3.8-flash-next-fp8`, `OVERRIDE_MAX_MODEL_LEN=262144`, `OVERRIDE_YARN_ENABLE=false`, `SKIP_PLE_PATCH=true` e `exec start.sh` | `start-fp8.sh:26–32` |
| `start-v030.sh [flags]` | exporta `V030=true`, `OVERRIDE_IMAGE=vllm/vllm-openai:v0.30.0`, `OVERRIDE_KV_CACHE_DTYPE=auto`, `OVERRIDE_GPU_MEMORY_UTILIZATION=0.80`, `SKIP_PLE_PATCH=true` | `start-v030.sh:9–15` |
| `V030_BREAKABLE_CUDAGRAPH` | valor de `VLLM_USE_BREAKABLE_CUDAGRAPH` na lane v0.30 (default `0`; `1` restaura upstream) | `start.sh:621` |
| `stop.sh [--nfs\|--all]` | `docker rm -f vllm-fn` no worker e depois no head; `--nfs` derruba o share (timeout 15 s; rpcbind em D-state pode ignorar SIGKILL) e remove o volume | `stop.sh:22–60` |
| `download.sh [--fp8 \| org/repo]` | baixa para o head (`snapshot_download(max_workers=4)` para ablit; senão `uvx hf download`/`huggingface-cli`/`hf`), resumível, valida completude | `download.sh:9–16,146–229` |
| `check-weights.sh [--verify\|--dry-run\|-n\|--manifest FILE]` | presença+tamanho nos 2 nós; `--verify` SHA-256 (LFS) / git blob SHA-1 (arquivos git) contra o manifest da API HF (`/api/models/{repo}/tree/{rev}?recursive=true`); `--dry-run` sem hash; `--manifest` offline | `check-weights.sh:7–29` |
| `verify-weights.py --repo --path --revision --manifest --save-manifest --fetch-only --dry-run --workers 8 --quiet` | exit 0 ok / 1 problema de pesos / 2 falha da ferramenta; lista EXTRA sem falhar | `verify-weights.py:17–40,417–544` |
| `start-tp1.sh [--no-launch]` | ver §2.16 | `start-tp1.sh:47–52` |

### 2.12 Linha de comando vLLM efetivamente renderizada (lane day-0, defaults do `.env.sample`)

Reconstruída de `start.sh:875–952` + heredoc `start.sh:1146–1179`. Com `.env.sample` intacto (fp8 KV, SSM bf16, draft vocab 47k, block-drop on, index-share on, YaRN off):

```bash
docker run -d --name vllm-fn \
  --gpus all --network host --ipc host \
  --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e GLOO_SOCKET_IFNAME=$IFACE -e NCCL_SOCKET_IFNAME=$IFACE -e TP_SOCKET_IFNAME=$IFACE \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP=$HEAD_IP -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -v files/ple_layer_patched.py:.../vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v files/modelopt_patched.py:.../vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v <overlays: mtp_patched.py, draft vocab, qsa_ops_patched.py, qsa_nvidia_patched.py,
      config_patched.json, hf_quant_config_patched.json, block_drop/{config/speculative.py,
      v1/core/kv_cache_utils.py, v1/core/sched/scheduler.py}> \
  -e VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt \
  -v $HF_CACHE_DIR:/root/.cache/huggingface -v $HOME/.cache/vllm:/root/.cache/vllm \
  vllm/vllm-openai:qwen38-flash-next \
  nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --enable-prompt-tokens-details \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 \
  --load-format safetensors --safetensors-load-strategy lazy \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --distributed-executor-backend mp \
  --mm-encoder-tp-mode data \
  --nnodes 2 --master-addr $HEAD_IP --master-port 50000 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"disable_eagle_block_drop":true,"index_share_for_mtp_iteration":true}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' \
  --node-rank 0 --host 0.0.0.0 --port 8888
```

Worker: mesmas flags, `VLLM_HOST_IP=$WORKER_IP`, `$WORKER_IFACE`/`$WORKER_IB_HCA`, overlays copiados para `/tmp/vllm-overlay/<basename>` (daí a checagem de colisão de basename em `add_overlay`, `start.sh:551–567`), PLE/modelopt em `/tmp/ple_layer_patched.py` e `/tmp/modelopt_patched.py`, cache HF local ou `-v vllm-fn-hf:/root/.cache/huggingface:ro` (NFS), e `--node-rank 1 --headless` no lugar de host/port. O script do worker vai por **stdin** (`ssh "bash -s" < script`), não por scp para path fixo (sem arquivo para vazar; `mktemp` 0600).

Com `YARN_ENABLE=true` e `MAX_MODEL_LEN=1000000`, o `--hf-overrides` vira:
`{"text_config":{"ple_embedding_dtype":"float8_e4m3fn","rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}` (`start.sh:927–946`).

Container: roda como root → mounts precisam mirar `/root/...` ou lookups offline do HF falham (README:468). `--restart` não setado (morre no crash; fable §4.4: restart automático é arriscado em TP de 2 nós porque um rank reiniciado sozinho trava no NCCL init). `~/.cache/vllm` (autotune FlashInfer, modelinfos) é o único estado por nó que persiste entre launches (README:998).

### 2.13 Variáveis de runtime injetadas por overlay

| Env | Quando | Onde | Efeito |
|---|---|---|---|
| `VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt` | `MTP_DRAFT_VOCAB` setado | `start.sh:633,651` | lido por `_attach_draft_vocab` no `mtp.py` patchado |
| `VLLM_QSA_PROFILE=<nome>` / `VLLM_QSA_PROFILE_JSON=/etc/vllm-qsa-profile.json` | `QSA_PROFILE≠stock` | `start.sh:591–596` | seleciona tabela de launch QSA |
| `VLLM_QSA_DET_TOPK=1` | determinismo | `start.sh:771` | ordena os índices top-k (−1 vai para a cauda) |
| `VLLM_MOE_DET_FINALIZE=1`, `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0`, `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/flashinfer_autotune_cache_unfused` | determinismo | `start.sh:772–775` | MoE CUTLASS com finalize não fundido + cache de autotune separado |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi_autotune` | lane v0.30 | `start.sh:620` | autotune em `/tmp` do container: um cache persistido que diferia entre os nós **deadlockou o TP2 na coletiva de autotune** (CHANGELOG 2026-09-25) |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=${V030_BREAKABLE_CUDAGRAPH:-0}` | lane v0.30 | `start.sh:621` | v0.30 liga por default para Qwen4Exp; no GB10 causava variância de 10–18% e −13–19% a S≥2 |

Knobs do vLLM citados no fable §7.3 como presentes no build (não usados): `--moe-backend {auto,flashinfer_cutlass,flashinfer_b12x,cutlass,marlin,emulation}`, `--linear-backend`, `--performance-mode {balanced,interactivity,throughput}`, `--async-scheduling/--no-async-scheduling`, `--mamba-cache-mode align`, `--long-prefill-token-threshold` (sugerido 2048), `--max-num-partial-prefills` (sugerido 2), `--override-generation-config` (ex.: `{"temperature":0.7}`), `--additional-config '{"gdn_prefill_backend":…}'`, env `VLLM_GDN_DECODE_KERNEL={cuda,triton}`, `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`, `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_TORCH_PROFILER_DIR`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (dspark usa 1800), `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_NCCL_ENABLE_MONITORING=1`, `VLLM_USE_B12X_MOE=1` (usado pela imagem dspark vizinha).

### 2.14 API: chat-template kwargs, sampling, multimodal

| Item | Valores | Onde | O que faz / gotcha |
|---|---|---|---|
| `chat_template_kwargs.reasoning_effort` | `low` / `medium` / `xhigh` (default `xhigh`) | README:628–637 | `high` e `max` **não** aceitos → HTTP 400 `Unexpected reasoning effort max…` antes do engine |
| `chat_template_kwargs.enable_thinking` | `true` default / `false` | README:639–643 | `false` pré-preenche `<think></think>` vazio; o check de `reasoning_effort` fica **dentro** do ramo thinking → com `false`, valor inválido é ignorado silenciosamente |
| `chat_template_kwargs.thinking_budget` | — | README:617 | **não honrado** por este build |
| `max_tokens` | — | README:614–616 | thinking consome `max_tokens` primeiro; `content` vazio + `finish_reason:"length"` → aumente (2–4k para QA de imagem) |
| `generation_config` do checkpoint | `temperature 1.0, top_p 0.95, top_k 20` | README:481; fable §3.2 | benchmarks são greedy; a T=1.0 rejection sampling aceita menos drafts → throughput real de chat menor |
| Sampling | FlashInfer top-k/top-p | README:481 | — |
| Multimodal | texto, imagem, vídeo; auto-detectado; URL ou base64 data-URL (sem `--allowed-local-media-path`); limite default 1 imagem + 1 vídeo por prompt; orçamento do encoder 16,384 tokens (auto-resize / sparse-sampling de vídeo) | README:600–621 | drafter MTP loga "Draft model … does not support external multimodal embeddings" → aceitação menor em prompts com mídia. Verificado: OCR de texto de 48 px num frame 3840×2160 (≈8.2k vision tokens) |
| `stream_options.include_usage` | `true` | `docs/cache-reporting.md` | obter `cached_tokens` no último chunk |
| Endpoints | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`, `/metrics`, `/health`, `/version`, `/docs` | README:490–491 | métricas: `vllm:spec_decode_num_accepted_tokens_per_pos_total`, `vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total`, `vllm:generation_tokens_total`, `vllm:inter_token_latency_seconds_{sum,count}`, `kv_cache_usage`, `preemption` |

### 2.15 Lane vLLM 0.30 (`start-v030.sh`) vs lane day-0

| Aspecto | Day-0 (`qwen38-flash-next`) | vLLM 0.30.0 stock |
|---|---|---|
| Código do modelo | `vllm/models/qwen3_8_flash_next/nvidia/` | `vllm/models/qwen4_exp/nvidia/` (`Qwen4Exp*`), carrega nvidia nativamente |
| Overlays | PLE, MXFP8, FP8-block MoE, alias MTP, FP8 KV, block-drop, draft vocab, (determinismo) | **só** draft vocab (`patch_mtp_draft_vocab_v030.py`) + FP8 KV opcional (`patch_qsa_fp8_kv_v030.py`, backport de vllm#55557; apagar ao ir para 0.31+) |
| `disable_eagle_block_drop`, `index_share_for_mtp_iteration` | via backport | nativos |
| KV default | fp8 | **BF16** (`OVERRIDE_KV_CACHE_DTYPE=auto`); fp8 opt-in: 2.64M tokens, ~5% mais lento |
| GMU | `.env` | `0.80` |
| Breakable CUDA graphs | n/a | forçado `0` |
| Não suportado | — | `FP8_DENSE`, `QSA_PROFILE`, determinismo (erro explícito, `start.sh:606–619`) |
| Resultado | baseline | decode igual ou acima (prose 60.0 vs 56.3 a S=1; code 441 vs 422 a S=8), TTFT multi-turno 2× melhor (0.28–0.40 s vs 0.63 s), #62 corrigido nativamente (vllm#54713, vllm#53945) |
| Imagem no worker | — | worker sem internet: `docker save vllm/vllm-openai:v0.30.0 \| ssh <worker> docker load` |

### 2.16 Lane single-Spark TP=1 (`start-tp1.sh` e `tp1/`)

| Nome | Default (`start-tp1.sh`) | `tp1/.env.sample` | Onde | O que faz / por que |
|---|---|---|---|---|
| `TP1_MODEL_ID` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (98.6 GiB) | — | L83 | RadixArk (125.9 GiB) não cabe |
| `MAX_MODEL_LEN` | `65536` | `262144` | L88 | >262144 recusado (sem YaRN em TP1) |
| `GPU_MEMORY_UTILIZATION` | vazio → derivado | — | L89,169–194 | `budget = (checkpoint − PLE) + OVERHEAD + MTP + max(kv_need, KV_TARGET)`; `gmu = budget/MemTotal` |
| `MAX_NUM_SEQS` | `4` | — | L90 | — |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | — | L91 | ativação menor (1.92 GiB medido a 2048) |
| `MTP_NUM_SPECULATIVE_TOKENS` | `0` | `3` | L92 | MTP custa 1.49 GiB (`MTP_GIB`) |
| `KV_CACHE_DTYPE` | `auto` | — | L93 | `KV_MULT=0.5` se fp8 |
| `KV_CACHE_MEMORY` | vazio | — | L94,310 | pin em bytes |
| `OVERHEAD_GIB` | `5.6` | — | L96 | medido 3.37 non-torch + 1.92 ativação + 0.12 grafos = 5.41 |
| `KV_TARGET_GIB` | `8.0` | `20` | L98 | mais KV = mais UVM; 20 GiB ≈ 686k tokens, MemAvailable ~13 GiB |
| `HOST_SLACK_GIB` | `10.0` | `5` | L101 | cap do cgroup = budget + slack |
| `OS_RESERVE_GIB` | `16.0` | — | L103 | cap nunca a menos de 16 GiB do pool |
| `MEMWATCH_MIN_GIB` | `6` | — | L105 | watchdog `files/memwatch.sh` mata o container se `MemAvailable` < floor |
| `PLE_OFFLOAD` | `true` (obrigatório) | — | L106,131 | "98.6 GiB through UVM hung the host" |
| `PLE_GIB` | `26.82` | — | L107 | tabela NVFP4 PLE fora da GPU (mmap) |
| `KV_BYTES_PER_TOKEN` | `29482` (bf16, medido) | — | L166 | usado no cálculo de orçamento |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | — | L113 | `NONE` para debug eager |
| `TP1_CONTAINER_NAME` | `vllm-fn-tp1` | — | L108 | — |
| docker | `--cap-add SYS_PTRACE` (py-spy), `--memory ${CONTAINER_MEM_GIB}g --memory-swap ${CONTAINER_MEM_GIB}g`, `-e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_PACKED_TABLE_DIR=... -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300`, overlays de `ple_offload_layer.py`, `v1/ple_offload/{connector,worker,protocol}.py` | — | L333–358 | ver §3 (patch_ple_offload) |
| Guardas | recusa se ComfyUI roda; recusa porta 8888 se `comfy-h3.service` ativo; recusa se KV esperado < necessário; recusa se cap > `MemTotal − OS_RESERVE`; recusa se `MemAvailable < cap + 4` | — | L149–220 | host já deu hard-crash 3× |
| `tp1/start.sh` | diferenças: respeita `tp1/.env`, adiciona `--enable-prompt-tokens-details`, `.last_launch.sh` | — | diff | "self-contained recipe" |

### 2.17 Parâmetros dos scripts de benchmark (`bench/`)

| Script | Parâmetros e defaults | Mede | Notas |
|---|---|---|---|
| `bench/decodebench.py` | `--decode 600`, `--contexts 1000,600000`, `--temps 0.0,0.8`; tarefas `prose`, `code` (red-black tree), `entropy` (licenças aleatórias), `copy` (reproduzir entradas do log) | decode tok/s = (ctok−1)/janela após TTFT, `ignore_eos`, `min_tokens=max_tokens` | decode depende de aceitação MTP, não do contexto (1k→600k custa 2–8%); `copy` é o melhor caso (~70 tok/s); `entropy` a T=0 degenera em repetição → ler só a T=0.8 |
| `bench/longctx.py` | `--target 600000`, `--max-tokens 1024`; 3 needles a 5%/50%/95% | prefill tok/s = prompt_tokens/TTFT; needle PASS/FAIL | salt UUID no início mata reuso de prefix cache; calibra com `/tokenize` (~25 tokens/linha) |
| `bench/mtp_accept.py` | `--save`, `--since`, `--url` | delta de contadores spec-decode | cumulativo mistura todos os clientes; sempre medir delta |
| `bench/reasoning_check.py` | 10 tarefas determinísticas + needles `--needles 8000,64000`, `--max-tokens 2048`, T=0, `--compare` (exit≠0 em regressão) | score X/12 | feito porque o modo de falha do FP8 KV é **resposta errada**, não lentidão |
| `bench/sweep.py` | `--tag`, `--streams 1 2 4`, `--prompt prose`, `--repeats 3`, `--max-tokens 600`, `--out logs/sweep.jsonl`; via sparkDash (`:5555`) | ms/step = Δ(ITL_sum)/Δ(count); tok/step = 1 + Δaccepted/Δdrafts; per-position; `MemAvailable`/`MemFree` mínimos; `NV_ERR_NO_MEMORY` do `journalctl -k` | ordem alternada por repetição; retry em 409/429 do sparkDash |
| `bench/gen_draft_corpus.py` | `--per-prompt 2`, `--max-tokens 1200`, `--temperature 0.8`, `top_p 0.95`, `--concurrency 4` | corpus `.jsonl` da **saída do próprio modelo** | distribuição que o drafter precisa prever |
| `files/build_draft_vocab.py` | `corpus...[:N]`, `--size 32768`, `--report-only`, `--balance-shards` (=TP), `--fill-to-size`, `--model` (`DRAFT_VOCAB_MODEL`) | cobertura a 8k/16k/32k/65k/131k | sempre mantém special/added tokens; "tune on coverage, not size" |
| `files/qsa_gb10/bench_qsa_kernels.py` | `--tp 2`, `--rows 1 4 8 32`, `--ctx 4096 32768 131072`, `--out qsa_profile.json` | µs por kernel | sweep block_n∈{16,32,64} × splits∈{1..64} × warps∈{2,4}; MQA tiles∈{1..16} × warps∈{1,2,4}; page 1600, top-k 2048+3, 12 q-heads×1 kv-head×256 (TP2) |
| `nccl-tests all_reduce_perf -b 4K -e 64M -f 2 -g 1` | sugerido | latência 8–32 KB, `GDRDMA`/`dmabuf` | fable §6 |
| `vllm bench serve --dataset-name random --random-input-len 512 --random-output-len 512 --num-prompts 8 --max-concurrency 1 --temperature 0 --seed 0 --percentile-metrics ttft,tpot,itl` | sugerido | decode batch-1 | fable §6 |


---

## 3. Inventário de modificações (patches, overlays, hotfixes, scripts que mudam comportamento)

Mecanismo comum a quase todos: **extrair o arquivo `.py` original da imagem** (`docker create` + `docker cp`, helper `extract_from_image`, `start.sh:568–575`), aplicar edições por **âncora de texto exata** (cada âncora precisa ocorrer exatamente 1 vez; senão o patch aborta sem escrever nada — padrão testado em `tests/*`), validar com `ast.parse`, e **bind-montar `:ro`** o arquivo resultante sobre o caminho do pacote vLLM dentro do container, nos dois nós. Nenhum rebuild de imagem; o cache HF nunca é modificado (configs patchadas também são montadas por cima do snapshot). `add_overlay` recusa dois overlays no mesmo caminho e colisões de basename (worker usa `/tmp/vllm-overlay` plano).

| # | Nome | Arquivo(s) | Problema que resolve | O que muda tecnicamente | Conceitos | Efeito medido |
|---|---|---|---|---|---|---|
| 1 | **Fix do YaRN no-op** | `start.sh:919–946` | `--hf-overrides '{"rope_parameters":...}'` no nível raiz: `ModelConfig._apply_dict_overrides` só recursa em chaves que são configs aninhadas; para `qwen4_exp` o pai expõe `rope_parameters` como dict simples → `setattr` no pai, **nunca chegava a `text_config`**. Verificado: `text_config.rope_parameters` ainda `rope_type: "default"`. Todo run "1M" antigo servia 1M posições com RoPE **não escalado** | Um único payload JSON aninhado sob `text_config` com `ple_embedding_dtype` e `rope_parameters` | RoPE, YaRN, config aninhada multimodal (`text_config`) | 1M agora **não validado**; `YARN_ENABLE` default virou `false`, `MAX_MODEL_LEN` 262144 |
| 2 | **Guard YaRN ≤ nativo** | `start.sh:164–170` | YaRN abaixo de 262144 só custa precisão | força `YARN_ENABLE=false` e imprime NOTE | YaRN | — |
| 3 | **Detecção do dtype da PLE** | `files/detect_ple_dtype.py`; `start.sh:356–362` | nvidia declara PLE FP8 só em `quantization_config.config_groups`; o `ple_layer.py` patchado despacha por `text_config.ple_embedding_dtype` → sem isso a tabela de 51 GB falha | procura o grupo cujo `targets` contém `ple_embedding.ngram_embedding`; `num_bits 8 → float8_e4m3fn`, `4 → nvfp4`; vazio se já declarado (local-inference-lab `nvfp4`, RadixArk `float8_e4m3fn`) | metadados ModelOpt MIXED_PRECISION | tabela carrega |
| 4 | **PLE patch (mixed NVFP4+FP8)** | `files/patch_ple_layer.py` → `files/ple_layer_patched.py` → `.../qwen3_8_flash_next/nvidia/ple_layer.py` | Imagem stock só escolhe o caminho FP8-PLE se o **checkpoint inteiro** for `Fp8Config`; checkpoints com corpo NVFP4 (PLE NVFP4 no-family ou PLE FP8 excluída) montam embedding BF16 (~102 GB) e crasham | Porta o dispatch do vLLM PR #53899 (`qwen4_exp`): nova `Qwen3_8FlashNextPLENVFp4EmbeddingMethod` (linhas empacotadas uint8 `dim/2` + escalas FP8 por bloco de 16 + escala global FP32, LUT de dequant) e `_get_ple_embedding_quant_method` que decide por `ple_embedding_dtype` antes de olhar `Fp8Config`/`ModelOptNvFp4Config`; `Qwen3_8FlashNextPLEFp8EmbeddingMethod` trata "FP8 shards + um `weight_scale` global". Também 3 fixes de offload (HANDOFF §5): `get_offload_output_dtype` restaurado (uint8 p/ NVFP4, `float8_e4m3fn` p/ FP8), `_offload_quant_method` atribuído, `row_width = weight.shape[-1]` (linhas NVFP4 = `head_dim/2 + head_dim/16` bytes) | quantização de embedding, NVFP4 packing, bind-mount | carga OK; offload deixa de gerar "fluent gibberish" |
| 5 | **MXFP8 kernel fallback** | `files/patch_modelopt_mxfp8.py` → `files/modelopt_patched.py` → `.../quantization/modelopt.py` | `FlashInferCutlassMxfp8LinearKernel` (`mm_mxfp8`) exige `N ≥ 128, N%32==0, K ≥ 128, K%32==0` (medido em sm_121; independe de M). `linear_attn.in_proj_a/b` `[48, 2560]` (×72) → `AssertionError ... N >= 128, got N=48` fatal na profiling run (~7 min, pesos já residentes); `visual.blocks.*.mlp.linear_fc1` `[4304,1152]` (×27) → `ValueError: Problem size is not supported` | `ModelOptMxFp8LinearMethod.create_weights` checa o `(N,K)` **pós-split TP** e troca por `EmulationMxfp8LinearKernel` (dequantiza MXFP8→BF16 uma vez na carga; ≈17 MB extra nos 72 `in_proj_a/b`); por instância de camada; todo `visual.*` vai por prefixo para emulação (config multimodal verificada; emular só visão evita OOM do dequant global). Loga 1 WARNING por shape | MXFP8 (microscaling), restrições de shape de GEMM, emulação | engine sobe; relevante sobretudo p/ checkpoints com MXFP8 (local-inference-lab); aplicado sempre na lane day-0 |
| 6 | **FP8_BLOCK_SCALES routed experts** | `files/patch_modelopt_fp8_block_moe.py` (in-place sobre `modelopt_patched.py`) | Experts roteados da MTP no nvidia são FP8 com escala por bloco 128×128, mas `ModelOptMixedPrecisionConfig.get_quant_method` só constrói `RoutedExperts` para `FP8`/`NVFP4`/`W4A16_NVFP4`/`MXFP8`; senão `None` → MoE **silenciosamente não-quantizada** que morre ~7 min depois com `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`. Gap também no upstream (`d4d703ca` e `main`) | adiciona ramo que constrói `Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[g,g])` com `g` lido do `group_size` do checkpoint e roteia para `Fp8MoEMethod` do vLLM (que já nomeia as escalas `weight_scale_inv`). **Recusa adivinhar** o bloco: bloco errado aplica escalas desalinhadas silenciosamente | FP8 block quantization, MoE quant dispatch | MTP funciona: 52.1 tok/s, aceitação 72.8% com curva decrescente 89/74.5/60% ("near-random acceptance" indicaria bloco errado) |
| 7 | **Alias de índice MTP + normalização FP8_PB_WO** | `files/patch_checkpoint_config.py` → `files/config_patched.json`, `files/hf_quant_config_patched.json` montados sobre o snapshot | vLLM cria a camada MTP no índice absoluto `mtp.layers.48` (= `num_hidden_layers + i`) e casa metadados de quantização por string exata; nvidia grava só `mtp.layers.0.*`. `config.json` e o legado `hf_quant_config.json` podem discordar (rev `fc694b54`: `FP8_PB_WO` vs `FP8_BLOCK_SCALES`); issue #38 mostra que o dispatch da MoE consome `config.json`. Keys (ablit) também rotula `FP8_PB_WO` | adiciona aliases `.0.`→`.48.` em `quantized_layers` e `config_groups.targets`; reescreve `FP8_PB_WO` (group_size 128) → `FP8_BLOCK_SCALES` nas entradas `mtp.*.mlp.experts`; modo `--mtp-moe-algo` para preflight (exit 3 se algo não suportado; `start.sh:725–737` falha em segundos em vez de 7 min) | layer naming, quant metadata, preflight | MTP carrega |
| 8 | **FP8 KV para QSA (day-0)** | `files/patch_qsa_fp8_kv.py` → `files/qsa_ops_patched.py` (`nvidia/ops/qsa.py`) e `files/qsa_nvidia_patched.py` (`nvidia/qsa.py`) | Backend QSA declara `supported_kv_cache_dtypes = ["auto","bfloat16"]` e levanta `requires a BF16 main KV cache` | kernels Triton `_qsa_mqa_paged_kernel` (scoring do indexador) e sparse GQA split-K ganham `k_scale_ptr`/`v_scale` e `KV_QUANT_MODE: tl.constexpr`; cache uint8 é reinterpretado como `float8_e4m3fn` sem cópia (`_qsa_as_fp8`, recusa ler uint8 como BF16); **escala por tensor aplicada depois do dot** de tensor core (`scores *= k_scale`) — algebricamente equivalente antes do arredondamento, sem tiles FP32 de dequant, mesma largura de tile que BF16; lista de dtypes suportados ganha fp8; remove as guardas duplicadas; passa `layer._k_scale`. Inerte com BF16 (branch eliminado em compile-time). Vendorizado do single-Spark (crédito à abordagem de `lancelind/qwen3.8-Flash-DGX`) | KV quantizado, escala per-tensor, Triton constexpr, tensor cores | 3,652,200 vs 2,131,159 tokens (1.70×/GiB; não 2× porque indexador/compressor QSA e estado GDN ficam BF16); 12/12 reasoning+needle; −5.3% throughput médio (rerun 09-16). Aviso upstream: um benchmark de raciocínio longo caiu 6/6 → 2/6 na implementação de referência |
| 9 | **FP8 KV para QSA (v0.30)** | `files/patch_qsa_fp8_kv_v030.py` (backport de vllm#55557, diff upstream embutido) → `files/v030_fp8kv/*` montados em `qwen4_exp/nvidia/{qsa.py,ops/qsa.py}` | v0.30 QSA só aceita BF16 | aplica o diff upstream (`IS_FP8: tl.constexpr`, `softmax_scale = head_dim**-0.5 * k_scale`, `output_scale`, `key_cache.view(torch.float8_e4m3fn)`, tabela de tuning sm_120) | idem | 2.64M tokens, NLL 1.328/1.334, needles 3/3 a 200k, ~5% mais lento que BF16 |
| 10 | **Reduced-vocab MTP drafting (TP-aware)** | `files/patch_mtp_draft_vocab.py` → `files/mtp_patched.py` (`nvidia/mtp.py`); `files/draft_vocab_en_code_47k.txt`; v0.30: `files/patch_mtp_draft_vocab_v030.py` (reusa os blocos) | lm_head próprio do drafter (1.18 GiB, 0.59/GPU) lido a cada draft step | `_attach_draft_vocab` (após `load_weights`) lê `VLLM_MTP_DRAFT_VOCAB`, fica só com ids no range `[org_vocab_start_index, org_vocab_end_index)` do rank, `index_select` das linhas → buffer `_draft_lm_head_weight` + mapa `_draft_id_to_target_id`; novo `get_top_tokens`: matmul no head reduzido, argmax local, all-gather de pares `(valor, id_global)` e argmax entre ranks (rank sem ids manda `-inf`). Escala de logit e tanh soft-cap são monotônicas → podem ser puladas. `compute_logits` segue full-vocab | FR-Spec, vocab-parallel LM head, speculative decoding greedy draft | +9.6% médio (09-11), +8.4% (09-16), todo em ms/step (code 61.2→54.0, prose 59.9→53.3 ms), tokens/step inalterados. 47k ids: 45,734 no rank 0 e 1,415 no rank 1 (verifiquei no arquivo: ids < 124160 vs ≥ 124160) → rank 0 lê 0.22 GiB, rank 1 ~0.01, mas o step espera o rank mais lento. Teste: `files/test_draft_vocab.py` |
| 11 | **Block-drop backport (vllm#53388)** | `files/patch_block_drop.py` → `files/block_drop/{config/speculative.py, v1/core/kv_cache_utils.py, v1/core/sched/scheduler.py}` (os 3 conectores de KV também são patchados mas **não montados**) | Com proposers EAGLE-like (inclui MTP), o prefix cache descarta o último bloco casado e o recomputa → 2º turno de sessão errava o cache (#62) | adiciona `disable_eagle_block_drop` ao `SpeculativeConfig` e `use_eagle_block_drop()`; scheduler/kv_cache_utils usam o novo predicado. Pulado se a imagem já tiver a opção. Não inclui o fix de sliding window (QSA não suporta) | prefix caching, speculative config | turno 1 TTFT 3.25 → 0.63 s (mediana de 3), turno 2 1.12 → 0.59 s; decode/prefill/NLL iguais |
| 12 | **Determinismo** | `files/patch_determinism.py` → `qsa_ops_patched.py` (in-place) + `files/determinism/flashinfer_cutlass_moe.py` | ordem não determinística dos índices top-k e finalize fundido da MoE → saídas greedy diferentes entre runs idênticos em TP2+EP | após `topk_op(...)`: `sort` dos índices com `-1`→`INT32_MAX` (sentinelas vão para a cauda) se `VLLM_QSA_DET_TOPK=1`; `use_fused_finalize = VLLM_MOE_DET_FINALIZE != "1"` | reproducibilidade, redução não-associativa em ponto flutuante | spread 0.25/6.07 nats → 0/0; 5/5 → 1/5 saídas distintas; −4% |
| 13 | **QSA launch profile (GB10)** | `files/qsa_gb10/apply_patch.py`, `qsa.diff`, `bench_qsa_kernels.py` → overlay `nvidia/ops/qsa.py` | tabelas de tile "tuned on GB300 (160 SMs)"; GB10 tem 48 SMs | `_load_qsa_profile()` lê `VLLM_QSA_PROFILE_JSON` ou `VLLM_QSA_PROFILE=gb10`; buckets por `base_programs` (`small`, `lt32`, `le256`, `le512`, `large`) → `(block_n, target_splits, num_warps)`; MQA `tiles_small/tiles_large/num_warps`. `stock` = byte-idêntico | occupancy, split-K, waves de CTAs | **não medido** (GPU ocupada); `gb10` "≈1.3 waves em vez de ≈5, um merge sobre 16 parciais em vez de 64" |
| 14 | **FP8-dense hybrid checkpoint + overlay** | `files/fp8dense/{build.sh, make_fp8_dense_checkpoint.py, verify_fp8_dense_checkpoint.py, compute_quant_stats.py, test_quant_config_resolution.py}`; `files/overlay/{modelopt,model,hyperconnection,mtp}.diff` | Densas BF16 (GDN in/out, atenção q/k/v/o, HyperConnections, shared experts, lm_head ≈ 9.1 GiB bf16) são ~6.9 dos ≈9.2 GiB lidos por step; o FP8 oficial não ajuda (mantém densas BF16, experts FP8 = **mais** bytes) | Converter streaming (chunks de linha de 16 MiB, ~1.2 GB cgroup, CPU-only, ~2 min): 591 lineares → FP8 E4M3 com escala **por canal de saída** (`[out]` fp32) + ativação dinâmica por token (sem calibração; funciona nas projeções rank-320 que não são múltiplas de 128; e `ParallelLMHead` sharda escalas por canal corretamente, enquanto escalas de bloco seriam narradas por linhas de vocab). Experts/PLE hard-linked; 4 shards BF16 reescritos (11.8 GB). Config ModelOpt `MIXED_PRECISION` (`quantized_layers` por tensor). Overlays: `modelopt.py` despacha `FP8_PER_CHANNEL_PER_TOKEN` → `ModelOptFp8PcPtLinearMethod` e `FP8_PB_WO` → `ModelOptFp8PbWoLinearMethod`, prefixos extras (`model.`→`model.language_model.`, `lm_head`); `hyperconnection.py` aceita `quant_config` (antes `quant_config=None` hard-coded); `model.py`/`mtp.py` passam `quant_config` a HC e lm_head. Mantidos BF16: embeddings, router gates, shared-expert gate, indexador QSA, GDN `in_proj_a/b`, conv1d, norms, projeções PLE, MTP, visão | per-channel vs block quantization, ativação dinâmica, E4M3 max 448 | Rel. RMSE min/mediana/max 0.0240/0.0262/0.0268 (pior: gates `block_inject_weight`), cos ≥ 0.9996; 30/30 resoluções de quant-method corretas; esperado `Selected CutlassFP8ScaledMMLinearKernel for ModelOptFp8PcPtLinearMethod` e +≈3.5 GiB/GPU de KV. **GPU: não medido.** Se aceitação MTP cair, reconstruir com lm_head BF16 |
| 15 | **PLE CPU offload para GB10 (TP1)** | `files/patch_ple_offload.py` → `files/ple_offload/{protocol,connector,worker,ple_offload_layer}.py`; `files/build_ple_packed_table.py` | GB10 reporta `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0`: `cuStreamWaitValue32` retorna sucesso mas o próximo launch bloqueia o host e dentro de CUDA graph capturado nada espera → worker trava após captura. Além disso a tabela copiada para RAM anônima ocupa memória não-evictável | handshake **host-side** com número de sequência em flag de shared memory (GPU worker bloqueia até o CPU worker sincronizar a cópia H2D); stream-mem-ops viram no-ops; tabela NVFP4 PLE (26.8 GiB) empacotada em um arquivo flat `[total_rows, 90] uint8` (codes + escalas) e **mmapped** (page cache evictável). Requer `--distributed-executor-backend mp` | CUDA stream memory ops, CUDA graphs, mmap, UMA | TP1 funciona: non-evictable ~77–78 GiB + KV em vez de ~104 GiB + KV |
| 16 | **Normalização de `refs/main`** | `start.sh:300–320` (head), `start.sh:421–436` (worker) | hf_hub 1.28 lê `refs/main` com `f.read()` sem `.strip()`: `echo $SHA > refs/main` deixa newline → `snapshots/<sha>\n` não existe → vLLM morre no arg-parse com repo-not-found (issue #36) | reescreve só se os bytes diferem (compara contagem de bytes vs string sem espaços; `$(cat)` esconderia o newline) | layout do HF cache, modo offline | — |
| 17 | **Completude do snapshot** | `files/resolve_snapshot.py` | download interrompido parecia pronto; rsync copiaria o buraco | exige todos os shards do `weight_map`; prefere `refs/main` completo, senão o completo mais novo; exit 0/1/2 | integridade de checkpoint | — |
| 18 | **Verificação por hash** | `verify-weights.py`, `check-weights.sh --verify` | shard corrompido mantém tamanho parecido; engine morre a ~33% da carga com `safe_open` "incomplete metadata, file not fully covered" (issue #30) | manifest da API HF: SHA-256 de `lfs.oid` para LFS; git blob SHA-1 (`"blob <size>\0"+conteúdo`) para arquivos git; 8 threads; manifest salvo com repo/revisão (recusa manifest de outro repo) | hashing, Git LFS | ~1 min para 135 GB a 2.4 GiB/s (NVMe local) |
| 19 | **Despejo do page cache** | `files/evict_page_cache.py`; `start.sh:502–540` | issue #35 (ver §2.10) | `posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)` em cada arquivo pesado, read-only, só páginas limpas | page cache, UMA | 4.15 → 0.00 GiB; NFS 3.00 → 0.00 GiB |
| 20 | **NFS share** | `files/nfs-share.sh`, `files/nfs-server/{Dockerfile,entrypoint.sh}` | evitar cópia de ~126 GiB no worker | Alpine + `nfs-utils`, container `--privileged --network host --restart unless-stopped`, export read-only de `$HF_CACHE_DIR`; reusa servidor NFSv4 vivo (se for estrangeiro, monta pelo caminho absoluto `device=:$HF_CACHE_DIR`, pois o pseudoroot v4 dele é o `/` do host — bug corrigido em 2026-09-10) | NFSv4.2, nconnect | cold start streama ~126 GiB |
| 21 | **Launcher: array `VLLM_ARGS` único** | `start.sh:875–956` (fix do fable §4.1) | antes os heredocs tinham flags hard-coded: `EXTRA_VLLM_ARGS` descartado, `ENABLE_EXPERT_PARALLEL=false` ignorado, `MTP_NUM_SPECULATIVE_TOKENS=0` emitia `num_speculative_tokens:0` (vLLM rejeita `gt=0`) | um array renderizado em string compartilhada por head e worker; `EXTRA_VLLM_ARGS` por último; `.last_head_launch.sh` (0600) | — | — |
| 22 | **Preflight de GPU** | `start.sh:470–483` | co-tenant → OOM 10 min depois | `nvidia-smi --query-compute-apps` nos 2 nós | — | — |
| 23 | **Watchdog de memória** | `files/memwatch.sh` (TP1) | em UMA o pool esgotado **trava o kernel** em vez de levantar OOM | poll 1 s de `MemAvailable`; `docker kill` abaixo do floor; timeline a cada 5 s com `memory.current` do cgroup | UMA, cgroup | segunda linha de defesa; HANDOFF §6: poller userspace não pega colapso de GiB/s sozinho |
| 24 | **Sysctls de reserva de memória** (medido, **não recomendado** no dual) | HANDOFF §6; CHANGELOG 2026-09-25 | proteger host no TP1 | `vm.min_free_kbytes=4194304`, `vm.watermark_scale_factor=300`, `vm.swappiness=30`, `vm.vfs_cache_pressure=200` (runtime, resetam no reboot) | kernel VM tunables | TP1: 6 runs sem crash. Dual: custou 7–14% decode (code S=8 417→360), `MemAvailable` lia 0 (falso alarme) e ainda logou 1 `NV_ERR_NO_MEMORY` em 25 min |

Outros itens "medidos e não adotados" (CHANGELOG 2026-09-25): `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1664` (workaround vllm#53504) — sem mudança sobre o block-drop; o 1º repeat idêntico de prompt curto ainda erra (7.9k: 2.70 s depois 0.58 s) na lane day-0.

---

## 4. Números e benchmarks

### 4.1 Hardware e pool de memória (por nó)

| Item | Valor | Fonte |
|---|---|---|
| Memória total visível ao CUDA | 121.69 GiB (LPDDR5x unificada) | README:349; HANDOFF §1 |
| Banda teórica | ~273 GB/s (realista 235–245 GB/s) | fable §0, §2 |
| Livre no startup | 108.79 GiB | README:350 |
| Orçada a GMU 0.835 | 101.61 GiB | README:351 |
| Pesos (`model_runner.py:407`) | 64.46 GiB (≈ experts 34 GB + shard PLE ~25 GB + BF16 denso ~10 GB) | README:352, 362–363 |
| Pesos + non-torch | 68.52 GiB | README:353 |
| Pico de ativação | 1.07 GiB | README:354 |
| CUDA graphs | 0.54 GiB (outra medição: 0.39 GiB, sizes 1–64) | README:355, 484 |
| **KV cache** | **32.02 GiB** | README:356 |
| Driver a GMU 0.835 | 114.5 GiB/nó; `MemAvailable` 0.3–0.9 GiB sob soak | CHANGELOG 2026-09-26 |
| Nós parados | 116–117 GiB livres | idem |
| Lane v0.30 a GMU 0.80, soak 1 h | head 4.5 GiB, worker 7.8 GiB `MemAvailable` mínimo | idem |
| OS floor | ~6 GiB em boot limpo; swap 16 GiB | HANDOFF §1 |

### 4.2 KV cache (TP2, nvidia NVFP4, MTP3, 262144)

| Config | Pool | Tokens | Conc. 1M | Conc. 262K | Fonte/data |
|---|---|---|---|---|---|
| **fp8 KV, GMU 0.835 (medido)** | **32.02 GiB** | **3,652,200** | 3.65× | **13.93×** | README:311–314, 385 |
| bf16 KV, GMU 0.835 (medido) | 31.70 GiB | 2,131,159 | 2.13× | 8.13× | README:386 |
| fp8, `--kv-cache-memory=41360080896` | 38.52 GiB | ≈4.39M | ≈4.4× | ≈16.8× | README:387 (estimado) |
| bf16, GMU 0.85 (container/checkpoint antigo) | 36.35 GiB | 2,572,755 | 2.57× | 9.8× | README:388 |
| MTP off, bf16 | — | 2,663,445 | — | 10.16× | README:708 |
| fp8, SSM bf16 (A/B 2026-09-10) | — | 4,241,167 / 4,238,238 | — | — | CHANGELOG:189–190 |
| bf16, SSM bf16 (rerun 09-16 `orig`) | — | 2,426,193 | — | 9.26× | rerun:17 |
| fp8, SSM bf16 (rerun 09-16) | — | 4,378,829 | — | 16.70× | rerun:18–19 |
| Lane v0.30, BF16 KV, GMU 0.80 | — | 1,340,476 (perf 09-27) / 1,467,072 (5.6×) / "1.52M (5.8×)" | — | — | README:857; CHANGELOG:53–54; README:85 (três números diferentes em lugares diferentes) |
| Lane v0.30, fp8 KV (backport) | — | 2.64M | — | — | README:86; CHANGELOG:26–28 |
| FP8 oficial (`start-fp8.sh`) | — | "around 500k" | — | — | README:100, 1065 |
| Antigo (HANDOFF, MAX_MODEL_LEN 1M, GMU 0.835, bf16) | 44.24 GiB → 30.37 GiB | 3,246,130 (local-inference-lab) → 2,212,074 (RadixArk) | — | — | HANDOFF §3 |

Aritmética de bytes/token (README:369–381; patch_qsa_fp8_kv.py:14–17):
- Atenção "pura": `13 camadas (12 QSA + 1 MTP) × 2 (K,V) × 2 kv_heads × 256 × 2 B = 26,624 B/token` bf16 (13,312 B/GPU a TP2); fp8 = 13,312 B/token (6,656/GPU). Sem MTP: `12 × 2 × 2 × 256 × 2 = 24,576 B/token` (≈84% dos 29,294 B/token single-Spark).
- Medido: fp8 **9,418 B/token/GPU** (+41% sobre o puro), bf16 **15,192 B/token/GPU** (+14%). O overhead relativo cresce porque só a parte de atenção caiu pela metade; estado GDN e estado do índice QSA ficam BF16 → 1.70× em vez de 2×.
- HANDOFF (bf16, antes do fp8): **29,482 B/token (28.8 KiB)** — usado em `start-tp1.sh:166`.
- Regras práticas: 1 GiB de pool ≈ 71–74K tokens bf16 ≈ +0.07× a 1M (±5% por arredondamento de página do allocator híbrido); 0.835→0.85 = +1.24 GiB → +91,331 tokens; +0.01 GMU ≈ +0.83 GiB ≈ +95k tokens fp8 (`.env.sample:56`).
- Headroom reportado pelo vLLM a 0.835: "Replace gpu_memory_utilization config with `--kv-cache-memory=33651431445` (31.34 GiB) to fit into requested memory, or `--kv-cache-memory=41360080896` (38.52 GiB) to fully utilize gpu memory. Current kv cache memory in use is 32.02 GiB." → ~6.5 GiB (~+740k tokens fp8) não usados.
- Sobreinscrição: `MAX_MODEL_LEN=1000000 × MAX_NUM_SEQS=8` pede 8M tokens; o README diz que o pool "holds 2.48M" (número da era bf16) → só ~2 requisições de 1M residentes; a 3ª+ é preemptada e re-prefilled (vLLM não dá OOM, faz preempção). Tabela de perfis sugeridos: Deep-context sequencial (`1000000`, seqs 2); 1M + concorrência (`--kv-cache-memory=45502283776`, seqs 3 ≈ 3.0M tokens); Throughput/agentic (`262144`, YaRN off, seqs 8, "8 of 9.5 possible" — número bf16); Middle ground (`512000`, seqs 4).

### 4.3 Carga e cold start

| Item | Valor | Fonte |
|---|---|---|
| Cold start | ≈10m55s (21:47:41 → 21:58:36 UTC) | README:486–488 |
| NCCL setup | ~40 s | idem |
| Carga de pesos | 458 s (target 392 s + drafter MTP 67 s), lazy safetensors (em parte streaming NFS) | idem; fable §4.6 |
| Engine init | 92.4 s (profile + KV alloc + warmup); outra medição 122 s | README:487, 665 |
| Graph capture | ~7 s | README:488 |
| Primeira requisição | lenta (FlashInfer autotune) | README:488 |
| Checkpoint "Load time" | ~6:15 pesos | README:665 |
| MXFP8 crash (sem patch) | ~7 min dentro do launch (na profiling run) | README:576–578 |
| TP1 | ~10–12 min até /health (~7 min drenando 26 GiB de PLE à toa no placeholder) | HANDOFF §0 |

### 4.4 Composição do checkpoint e modelo de banda por step (fable §2, RadixArk, parse dos headers)

| Categoria | Tamanho | dtype |
|---|---|---|
| Experts roteados (48 × 512) | 63.28 GiB | NVFP4 packed 56.25 + escalas FP8 7.03 |
| Tabela PLE n-gram | 47.75 GiB | FP8 (vocab-sharded, ~24 GiB/GPU) |
| Experts da MTP | 4.69 GiB | **bf16** no RadixArk (FP8 block no nvidia) |
| Projeções GDN (36) | 3.89 GiB | bf16 |
| HyperConnection (97 módulos) | 1.19 GiB | bf16, replicado |
| lm_head | 1.18 GiB | bf16 |
| embed_tokens | 1.18 GiB | bf16 (só gather) |
| Atenção q/k/v/o (12) | 1.11 GiB | bf16 |
| Shared expert (48) | 0.44 GiB | bf16 |
| Router gate (48) | 0.12 GiB | bf16, replicado |
| Indexer QSA proj | 0.04 GiB | bf16, replicado |
| Visão | 0.84 GiB | bf16 |

Leitura por GPU por step (batch 1, MTP3 = verify de 4 tokens + 3 drafts de 1 token): lm_head 4× = **2.36 GiB (26%)**; experts roteados ~2.1 (23%; ~32 experts únicos/camada × 48 × 2.77 MB, dividido por EP); GDN 1.95 (21%); HC 1.19 (13%, replicado); atenção+indexer 0.60 (7%); MTP denso+experts ×3 ~0.5 (5%); router+shared+PLE proj 0.40 (4%); estado GDN 0.1 (1%; 36 × 1.5 MiB fp32 R+W) → **≈9.2 GiB (9.9 GB)**. A 273 GB/s = 36 ms/step; a 235–245 GB/s = 40–42 ms/step (24–25 steps/s); × ~2.2 tokens aceitos/step = 53–56 tok/s = medido 55.8 → **batch-1 já está no roofline de banda**. Alavancas: menos bytes/step (FP8 denso, draft vocab), mais tokens aceitos/step, mais sequências por step, reduzir o piso de ~106 coletivas (2–3 ms/step).

Prefill (fable §5): chunk de 8192 toca todos os experts (48×512×2.77 MB = 68 GB, 34 GB/GPU → ~140 ms de streaming) + 96 all-reduces de `8192 × 2560 × bf16 = 42 MB` → ~150–180 ms de comunicação por chunk num link de 200 GbE.

### 4.5 Speculative decoding (MTP)

| | MTP off | MTP=3 | Fonte |
|---|---|---|---|
| Decode batch-1 greedy | 24.5 tok/s | **52.1 tok/s (2.13×)** | README:704 |
| Aceitação | — | 72.8% (823/1131) | README:705 |
| Por posição | — | 89% / 74.5% / 60% | README:706 |
| Pesos residentes | 62.72 GiB | 64.3 GiB | README:707 |
| Aceitação por conteúdo (HANDOFF) | — | 62.8–92.6% (comprimento aceito 2.88–3.78 de 4) | HANDOFF §3 |
| Prompt `code` do sparkDash | — | p1/p2/p3 = 1.00/0.99/0.99 (sintético: 50 helpers `clamp_NN` idênticos) | rerun caveat 1 |
| Prompt `prose` | — | 0.81/0.64/0.51 (representativo) | idem |
| TP1 (single-Spark) | 22.0 | prose 24.9 / code 35.1 | HANDOFF §0 |

### 4.6 Reduced-vocab drafting

2026-09-11 (nvidia NVFP4, NFS, TP2+EP, MTP3, fp8 KV, SSM bf16, 600 tokens, 3 repetições alternadas), agregado tok/s:

| | baseline | 47k vocab | Δ |
|---|---|---|---|
| code S=1 | 62.0 | 71.5 | +15.3% |
| code S=2 | 113.8 | 124.9 | +9.7% |
| code S=4 | 193.4 | 211.1 | +9.1% |
| code S=8 | 312.4 | 325.0 | +4.0% |
| prose S=1 | 49.2 | 56.8 | +15.5% |
| prose S=2 | 84.1 | 91.6 | +8.9% |
| prose S=4 | 134.8 | 146.0 | +8.3% |
| prose S=8 | 213.4 | 225.3 | +5.6% |

Média +9.6%; step code 61.2→54.0 ms, prose 59.9→53.3 ms; tokens/step code 3.80→3.85, prose ~2.9. Single-Spark mediu +13.1%. `MemAvailable` mínimo 2.38 vs 2.33 GiB; zero `NV_ERR_NO_MEMORY`.

2026-09-05 (bf16 KV, 400 tokens, 1k ctx, greedy):

| Draft vocab | Head/rank | prose/code/entropy/copy tok/s | Aceitação |
|---|---|---|---|
| full 248,320 | 0.59 GiB | 40.0 / 43.2 / 40.1 / 59.1 | 56.5% |
| 65,536 lowest-id fill | 0.31 GiB | 40.3 / 44.1 / 42.3 / 65.0 | 53.2% |
| 65,536 `--balance-shards 2` | 0.16 GiB | 42.5 / 44.5 / 45.4 / 56.5 | 47.8% |

Corpus daquele vocab: 85 documentos / 49,141 ocorrências → só 8,305 ids distintos; cobertura satura em 16,384; o resto era padding (proxy de ordem de merge BPE). Lowest-id fill pôs 65,392 de 65,536 no rank 0.

Rerun 2026-09-16 (3 braços, 1 launch cada, 24/24 linhas):

| Célula | orig (bf16, full) | + fp8 KV | + 47k | orig→tuned |
|---|---|---|---|---|
| code S=1 | 65.9 | 63.1 | 71.3 | +8.2% |
| code S=2 | 121.7 | 115.8 | 124.3 | +2.2% |
| code S=4 | 211.1 | 189.1 | 206.9 | −1.9% |
| code S=8 | 369.1 | 305.2 | 328.6 | −11.0% |
| prose S=1 | 51.9 | 50.5 | 56.5 | +8.8% |
| prose S=2 | 85.3 | 87.4 | 90.8 | +6.5% |
| prose S=4 | 134.9 | 133.0 | 141.3 | +4.7% |
| prose S=8 | 220.4 | 211.7 | 227.7 | +3.3% |

fp8 KV: −5.3% (prose −1.4%, code −9.2%); vocab: +8.4% (prose +7.4%, code +9.3%); cumulativo +2.6%. ms/step S=1 prose 59.3→51.9, code 60.7→54.0; fp8 ms/step flat a S=1 (60.5→60.7) e +11% a S=8 (84.9→94.6). Stdev intra-sessão 2.0–5.0%; células individuais não reproduzem, a média sim.

### 4.7 Performance — lane day-0 (GMU 0.835, fp8 KV + vocab balanceado 65,536 — não é o default atual)

Decode prose: ×1 54.4 (TTFT 160 ms); ×2 86.5 agg / 45.1 por stream (426 ms); ×4 128.1 / 34.2 (471 ms); ×6 169.7 / 29.7 (490 ms); ×8 **207.0** / 26.7 (432 ms). Agregado escala 3.8× de ×1 a ×8; por stream cai para 49%.

Prefill: 8k 2874.9 tok/s (TTFT 2.86 s); 16k 2962.1 (5.54 s); 32k **2962.2** (11.08 s); 64k 2893.4 (22.66 s); 128k 2727.1 (48.08 s). Plano ~2.96k tok/s de 16k a 64k, −8% em 128k (QSA evita que atenção domine).

Contra run anterior (bf16 KV, vocab completo): decode ×1 52.1→54.4 (+4.4%), ×2 83.3→86.5 (+3.8%), ×4 136.5→128.1 (−6.2%), ×6 170.7→169.7, ×8 210.1→207.0; TTFT ×1 236→160 ms (−32%); prefill 32k 2933.3→2962.2; 128k 2697.1→2727.1; KV 2,131,159→3,652,200 (+71%).

Superseded (GMU 0.85, MTP off, pesos antigos): code 55.8 tok/s (TTFT 0.26 s), reasoning 56.0, C# 44.4, prose 36.4.

### 4.8 Performance — lane vLLM 0.30 (2026-09-27; GMU 0.80, BF16 KV 1,340,476 tokens, seqs 8, 47k vocab, breakable CUDA graphs off, 512 tokens/stream)

| Concorrência | structured agg/stream | prose | code |
|---|---|---|---|
| ×1 | 76.0 / 76.0 | 59.2 / 59.2 | 66.6 / 66.6 |
| ×2 | 126.6 / 63.9 | 75.1 / 37.7 | 140.8 / 70.4 |
| ×4 | 165.9 / 42.4 | 118.5 / 29.9 | 245.6 / 61.4 |
| ×8 | 258.5 / 32.5 | 216.4 / 28.8 | **313.6** / 39.2 |
| ×16 | 326.3 / 41.5 | 194.9 / 25.5 | 327.5 / 42.7 |

×16 excede `MAX_NUM_SEQS=8` → TTFT médio 6.5–9.7 s (vs 0.1–0.4 s). ±20% entre runs únicos em concorrência média (structured deu 210.9 a ×4 e 308.1 a ×8 mais cedo).

Prefill v0.30: 1k 2791.3 tok/s (0.38 s); 4k 3313.5 (1.25 s); 16k **3489.8** (4.71 s); 32k 2982.3 (11.00 s); 64k 2045.7 (32.05 s); 128k 2897.6 (45.25 s) — 128k > 64k reproduziu em 2 runs, sem explicação.

CHANGELOG 2026-09-25 (mesmo par, mesmo harness):

| | day-0 + #66 | v0.30 stock | v0.30 lane |
|---|---|---|---|
| Decode S=1 prose/code | 56.1 / 75.8 | 52.5 / 61.5 | 57.6 / 71.5 |
| Decode S=4 | 146.5 / 244.1 | 135.7 / 197.0 | 141.0 / 247.3 |
| Decode S=8 | 225.8 / 421.9 | 194.4 / 316.5 | 219.8 / 420.7 |
| TTFT turno 1 (sessão 9.5k) | 0.63 s | 0.28 s | — |
| 1º repeat idêntico 7.9k (#62) | 2.70 s (miss) | 0.25 s (hit) | — |
| NLL (15,776 posições) | 1.331 | 1.333 | 1.333 |

v0.30 stock é 5–15% mais lento por draftar sobre 248k (58.3 ms/step a S=1). Com breakable CUDA graphs ligados: prose S=4 118/149/138/109 entre runs idênticos; desligado ~2% de variação, prose 60.0/95.5/145.2/234.8 e code 76.5/140.2/248.0/441.2 a S=1/2/4/8. Soak 1 h, 6 workers: 1,349 requisições, 1,348 corretas, 0 erros de servidor, 0 preempções.

### 4.9 Contexto longo e qualidade

| Item | Valor | Fonte |
|---|---|---|
| Reasoning + needle (fp8 vs bf16 KV) | 12/12 vs 12/12, needles a 13k e 104k | README:324; CHANGELOG:305, 312 |
| Decode prose/code/entropy/copy fp8 vs bf16 | 40.0/47.7/47.0/56.4 vs 42.5/44.5/45.4/56.5 | CHANGELOG:306 |
| NLL com index share | 1.328 vs 1.334 | CHANGELOG:93 |
| HumanEval (arm index share) | 156/164 | CHANGELOG:93 |
| Regressão com MTP | chat, streaming, tool calling, visão, long context 43,281 tokens (needle ok) | CHANGELOG:490–491 |
| Prefill longo (HANDOFF, TP2, YaRN 4.0 — que era no-op!) | 32k 2,808/2,754; 64k 2,654/2,550; 128k 2,432/2,432; 256k 2,171/2,152; 400k 1,955/1,964; **600k 1,711/1,751** tok/s (old/new checkpoint); TTFT 600k 350.6→342.7 s; 6/6 needles incl. 95% de profundidade em 600k | HANDOFF §3 |
| Decode por checkpoint (1k, T=0) | prose 42.9→37.0, code 51.5→43.4, copy 70.0→61.3 (local-inference-lab → RadixArk, −12 a −16%) | HANDOFF §3 |
| Concorrência (RadixArk) | 1→46.1, 4→110.3, 8→191.6 tok/s | HANDOFF §3 |
| Prefix cache 600k | 336 s frio → 2.87–8.82 s quente (~40–110×) | HANDOFF §3 |
| RadixArk qualificação | GSM8K 97.27% (1283/1319), AIME26 pass@1 98.75% (n=8) | fable §7.4; HANDOFF §2 |
| TP1 needle | 90% de profundidade de 128,947 e 201,447 tokens: ok; 72.5 s / 116.2 s (~1,750 tok/s) | HANDOFF §0 |
| TP1 KV | 685,825 tokens (GMU 0.813, budget 98.9 GiB, `KV_TARGET_GIB=20 HOST_SLACK_GIB=5`); 418,013 "at the default 12 GiB target" (o script atual usa default 8.0 — discrepância); antes 105,371 tokens / 3.05 GiB a GMU 0.872 | HANDOFF §0, §4 |
| Determinismo | spread 0.25/6.07 → 0/0 nats; greedy 5/5 → 1/5 distintas; −4% | CHANGELOG:94–98 |
| Block-drop | TTFT turno 1 3.25→0.63 s; turno 2 1.12→0.59; pós-tool 4k 2.55→1.98 | CHANGELOG:83–89 |
| Index share | code +6.1/+4.6/+21.3/+26.6%; prose +0.3/+5.3/−0.1/−1.8% | CHANGELOG:90–93 |
| Mamba SSM bf16 (single-Spark) | +8.5% a 8 streams (151.6→164.5), +6.8% a 1, step 141.4→130.4 ms, needles 15/15 a 32k | `.env.sample:76–80` |
| FP8-dense quant stats | rel RMSE 0.0240/0.0262/0.0268, cos ≥ 0.9996, 591 tensores | fable §8.1 |
| FP8-dense esperado | 9.2→5.7 GiB/step, ~85–90 tok/s (+40–60% batch 1) — **analítico** | fable §3.1; README:527–530 |

---

## 5. Conceitos necessários (ordem de pré-requisito, do básico ao específico)

### Bloco A — Fundamentos de ML / DL

1. **Rede neural e parâmetros (pesos).** Uma rede é uma composição de funções parametrizadas; os parâmetros (pesos) são números aprendidos no treino. Aqui "o modelo" são ~126–133 GB de pesos em arquivos `.safetensors`, e quase todo o custo de inferência vem de **ler** esses pesos da memória.
2. **Multiplicação de matrizes (GEMM/GEMV) como operação central.** Uma camada linear é `y = x·Wᵀ`; em decode com batch 1, `x` é um vetor (GEMV, "skinny GEMM" com M=1..4). O fable report mostra que as GEMMs densas BF16 vão para cuBLAS e que um kernel "skinny" CuteDSL existe mas só para SM103; a eficiência dessas GEMMs pequenas é o que define tok/s.
3. **Tipos numéricos (FP32, BF16, FP16, FP8, FP4).** Cada formato tem bits de expoente/mantissa diferentes: BF16 = 16 bits com faixa do FP32; FP8 E4M3 = 8 bits (máx ±448); FP4 E2M1 = 4 bits. A receita mistura todos: pesos densos BF16, experts NVFP4, PLE FP8, KV FP8, estado GDN FP32/BF16.
4. **Treino vs inferência; forward pass.** Na inferência só há forward (sem gradientes), então memória = pesos + ativações transitórias + caches. É por isso que `gpu_memory_utilization` reparte o orçamento em pesos, "peak activation" (1.07 GiB), CUDA graphs e KV.
5. **Roofline: limitado por memória vs por compute.** Se um kernel faz poucas operações por byte lido (baixa intensidade aritmética), a banda de memória é o gargalo. Decode batch-1 aqui é **memory-bound** (≈9.2 GiB lidos por step a ~240 GB/s ≈ 40 ms), enquanto prefill longo é mais compute-bound — daí "decode não depende de contexto, prefill sim".

### Bloco B — LLMs e Transformers

6. **Tokenização (BPE byte-level) e vocabulário.** Texto vira ids inteiros; este modelo tem vocabulário de **248,320** tokens. O vocab reduzido de draft (47,149 ids) e a observação "byte-level BPE is built in merge order" (ids baixos ≈ merges frequentes) dependem de entender BPE.
7. **Embeddings e `lm_head` (logits).** A entrada é uma tabela `vocab × hidden` (248,320 × 2560) e a saída é outra matriz que projeta o hidden em logits sobre o vocabulário; com `tie_word_embeddings=false` elas são diferentes. O `lm_head` BF16 de 1.18 GiB é lido em cada passo — e o drafter MTP tem o seu próprio, lido 3× por step.
8. **Softmax, amostragem e `generation_config`.** Logits → probabilidades; greedy (T=0) escolhe o argmax, T/top-p/top-k amostram. O checkpoint pede `temperature 1.0, top_p 0.95, top_k 20`; benchmarks são greedy, e a T=1 a aceitação especulativa cai.
9. **Bloco Transformer (residual + norm + atenção + MLP).** Cada camada aplica atenção e MLP com conexões residuais e RMSNorm. Aqui o residual é substituído por **HyperConnections** e a MLP por **MoE**, e 3 de cada 4 atenções são lineares (GDN).
10. **Self-attention Q/K/V e multi-head.** Cada token produz query, key e value; a atenção pondera os values pela similaridade Q·K. `head_dim: 256`, 24 cabeças de query.
11. **GQA (Grouped-Query Attention).** Várias cabeças de query compartilham poucas cabeças de K/V (aqui 2 KV heads para 24 Q heads), reduzindo o KV cache 12×. A fórmula de bytes/token do README usa `2 kv_heads × 256`; em TP2 cada GPU fica com 1 KV head.
12. **Gated attention output.** A saída da atenção é multiplicada por um gate aprendido (`output_gate_ptr` no kernel QSA). Aparece na assinatura dos kernels patchados para FP8.
13. **Posição: RoPE e `max_position_embeddings`.** RoPE rotaciona Q e K por ângulos que dependem da posição; o modelo foi treinado até 262,144 posições. Além disso, as rotações de alta frequência "nunca viram" posições maiores.
14. **YaRN (extensão de contexto por rope scaling).** Reescala as frequências do RoPE (por faixa) com um `factor` (4.0 → ~1M) e `original_max_position_embeddings=262144`. Aqui ele só começou a funcionar depois do fix do `--hf-overrides` aninhado; o repo força-o desligado ≤262144 e declara 1M não validado.
15. **Autoregressão: prefill vs decode, TTFT, ITL/TPOT, tok/s.** Prefill processa o prompt inteiro de uma vez (mede-se TTFT e tok/s de prefill); decode gera 1 token (ou K+1 com especulação) por step (mede-se ms/step e tok/s por stream/agregado). Todas as tabelas do repo separam essas duas fases.
16. **KV cache.** Guardar K e V de tokens passados evita recomputar; custa `camadas × 2 × kv_heads × head_dim × bytes` por token. É o que limita contexto × concorrência; aqui apenas 12 (+1 MTP) camadas têm KV.
17. **Reasoning models e `<think>`.** O modelo "pensa" em um bloco antes da resposta; `--reasoning-parser qwen3` separa esse texto. O pensamento consome `max_tokens`; `reasoning_effort` (`low|medium|xhigh`) e `enable_thinking` são lidos pelo template.
18. **Chat template (Jinja) e `chat_template_kwargs`.** O servidor renderiza as mensagens com o template do checkpoint; kwargs extras entram no template. O repo documenta que valores inválidos de `reasoning_effort` dão 400 só no ramo thinking.
19. **Tool calling.** O modelo emite chamadas de função num formato próprio; `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` convertem para o formato OpenAI.

### Bloco C — Arquitetura específica do Qwen3.8-Flash-Next

20. **Atenção linear / modelos de espaço de estado (SSM).** Substitui a atenção quadrática por uma recorrência com **estado de tamanho fixo por sequência**, custo O(n) e memória O(1) no comprimento. As 36 camadas GDN não têm KV por token — só um estado por requisição (~1.5 MiB/camada/GPU em fp32).
21. **Gated DeltaNet (GDN).** Atenção linear em que o estado é atualizado por uma regra "delta" (corrige a memória associativa) com gates de decaimento (`in_proj_a/b` produzem os escalares por cabeça; `conv1d` curto antes). Kernels: prefill Triton FLA `chunk_gated_delta_rule`, decode CUDA `fused_gdn_decode_post_conv_mtp` (≤8 spec tokens). O dtype do estado é o `--mamba-ssm-cache-dtype`.
22. **Arquiteturas híbridas (`full_attention_interval: 4`).** Intercalam camadas lineares (baratas, memória fixa) com poucas camadas de atenção plena (boa recuperação de longo alcance). Resultado: KV cache minúsculo por token e prefill quase linear até 600k.
23. **Atenção esparsa com indexador (QSA).** Nas 12 camadas "full", um indexador leve pontua chaves **comprimidas** (compress ratio 4, 128 dims) e seleciona top-k (≈2048 tokens) por query; a atenção GQA roda só sobre os selecionados (kernel paged split-K + merge). O repo não expande a sigla QSA. Consequência: quantizar K pode mudar **quais** blocos são escolhidos (risco de qualidade do FP8 KV) e o custo de `MAX_MODEL_LEN` no buffer de scoring.
24. **Split-K, occupancy e SMs.** Dividir a redução de um kernel em partições paralelas (splits) aumenta paralelismo mas exige um kernel de merge; o número ideal depende de quantos SMs a GPU tem (GB300 160 vs GB10 48). É a base do `QSA_PROFILE=gb10`.
25. **Mixture of Experts (MoE).** Cada token passa por um roteador que escolhe top-k experts (10 de 512) + um shared expert sempre ativo; só os escolhidos são computados. Em decode batch-1 lê-se ~32 experts únicos por camada; a quantização NVFP4 dos experts é o que faz o modelo caber.
26. **HyperConnections.** Generalizam a conexão residual com múltiplos fluxos (`hc_count`) e misturas aprendidas low-rank (rank 320) antes/depois de cada bloco (`attn_hyper_connection`, `mlp_hyper_connection`, `hyper_connection_mixer`). São BF16 e **replicadas** nos 2 GPUs (1.19 GiB lidos por step), alvo do FP8-dense.
27. **Tabela N-gram/PLE (embedding de n-gramas por hash).** Um embedding gigante (~51 GB FP8) indexado por hashes de n-gramas dos tokens anteriores, com várias cabeças de tamanhos primos (`ngram_heads_vocab_sizes`), dividida em 512 partes, aplicada na camada 1 com short-conv dilatada. É memória paramétrica barata de consultar (gather) mas enorme; o patch PLE existe porque o loader não sabia montar a versão FP8/NVFP4 dela.
28. **Multi-Token Prediction (MTP).** Uma camada extra treinada para prever tokens futuros a partir do hidden do modelo; aqui 1 camada (`mtp.layers.0`, instanciada como `mtp.layers.48`) usada recursivamente K=3 vezes como drafter. Tem MoE própria (FP8 block no nvidia) e `lm_head` próprio.
29. **Encoder de visão e tokens multimodais.** Imagens/vídeos viram tokens de visão por um ViT; o orçamento é 16,384 tokens por entrada. `--mm-encoder-tp-mode data` replica o encoder porque a MLP 4304 não divide por 16 após TP2.
30. **Abliteration.** Técnica que remove a "direção de recusa" do residual stream editando matrizes que escrevem nele (aqui `o_proj` das camadas QSA 3,7,…,47). O checkpoint `ABLIT=1` é gated e move as salvaguardas para o operador (o servidor escuta em `0.0.0.0`).

### Bloco D — Quantização

31. **Quantização: escala, granularidade e erro.** Representar um tensor com menos bits via `x ≈ q × s`; a granularidade da escala (por tensor, por canal, por token, por bloco) troca precisão por overhead. Todo o repo gira em torno de qual granularidade cada tensor usa.
32. **Quantização de pesos vs ativações; estática vs dinâmica.** Pesos têm escala fixa gravada; ativações podem ter escala calibrada (estática) ou calculada em runtime por token (dinâmica). FP8-dense usa pesos por canal + ativações dinâmicas por token (sem dataset de calibração).
33. **FP8 E4M3.** 1 sinal, 4 expoente, 3 mantissa, máx 448 (`FP8_MAX` em `make_fp8_dense_checkpoint.py`); erro relativo de arredondamento ~2.6% por tensor medido. Usado na PLE, nas experts da MTP (bloco 128×128), no KV cache e no FP8-dense.
34. **FP8 com escala por bloco (FP8_BLOCK_SCALES / FP8_PB_WO).** Uma escala (`weight_scale_inv`) por bloco 2D de 128×128 do peso. O vLLM já tinha o `Fp8MoEMethod`, mas o dispatch ModelOpt não o chamava; o bloco precisa ser lido do `group_size`.
35. **NVFP4.** Formato 4-bit da NVIDIA: valores E2M1 em blocos de 16 elementos, cada bloco com escala FP8 E4M3, mais uma escala global FP32 (`weight_scale_2`); W4A4 quando ativações também são FP4. Os experts roteados são NVFP4 → ~63 GiB em vez de ~250 GiB BF16.
36. **MXFP8 (microscaling OCP).** FP8 com expoente compartilhado por bloco de 32 elementos. Os kernels FlashInfer `mm_mxfp8` exigem N,K ≥128 e múltiplos de 32 → shapes que violam isso caem para emulação BF16.
37. **Checkpoints de precisão mista (ModelOpt MIXED_PRECISION).** Cada camada declara seu `quant_algo` em `quantized_layers`/`config_groups` (em `config.json` e no legado `hf_quant_config.json`), com `exclude_modules`/`ignored_layers`. Casamento por **string exata** do nome de camada é a origem do bug `mtp.layers.0` vs `.48`.
38. **Kernels de emulação.** Quando não há kernel nativo, dequantiza-se para BF16 (uma vez na carga) e roda-se GEMM BF16: correto, mas perde a economia de bytes. Usado para `in_proj_a/b` e toda a visão.
39. **KV cache quantizado (FP8, escala por tensor).** Guarda K/V em FP8 com `k_scale`/`v_scale`; o truque do patch é aplicar a escala escalar **depois** do produto no tensor core. Ganha 1.70× de capacidade; custa ~5% de throughput e pode perturbar a seleção esparsa.
40. **Métricas de qualidade de quantização.** NLL/perplexidade, KL divergence entre logprobs, spread de logprob, rel-RMSE e cosseno de pesos dequantizados, benchmarks (GSM8K, AIME, HumanEval) e needle-in-a-haystack. O HANDOFF sugere KLD como discriminador mais sensível que GSM8K (que a 97.27% com n=1319 só resolve gaps ≥1.3pp).

### Bloco E — Estratégias de inferência

41. **Speculative decoding (draft + verify).** Um drafter barato propõe K tokens; o modelo alvo verifica todos em um forward; aceitam-se os prefixos corretos (rejection sampling preserva a distribuição). Ganho = tokens aceitos por step / custo do step: 2.13× aqui.
42. **Taxa de aceitação por posição e tokens/step.** A aceitação cai com a posição (89/74.5/60%); `tok/step = 1 + aceitos/drafts`. Uma curva decrescente é a "prova de saúde" do MTP; aceitação quase aleatória indicaria pesos errados (ex.: bloco de escala errado).
43. **Proposers EAGLE-like e block drop.** Drafters que reusam o hidden do alvo (EAGLE, MTP) fazem o vLLM descartar o último bloco do prefix cache por segurança; `disable_eagle_block_drop` desliga isso. Resolveu o 2º turno sem cache hit.
44. **FR-Spec / drafting com vocabulário reduzido.** O drafter só precisa do argmax; restringir o `lm_head` a ids frequentes corta bytes sem mudar a saída (draft fora do conjunto = rejeitado). Em TP, o subconjunto deve ser balanceado por faixa de ids, pois o step espera o rank mais lento.
45. **Index sharing entre draft steps.** Reusar os índices top-k do QSA do 1º draft nos seguintes economiza kernels de scoring/top-k (dados ficam 1 posição defasados, só no drafter).
46. **Continuous batching e scheduler.** O vLLM insere/remove requisições do batch a cada step; `max_num_seqs` limita sequências e `max_num_batched_tokens` limita tokens por step. Aqui o limite prático de concorrência é o scheduler (8), não o cache.
47. **Chunked prefill.** Prefill longo é fatiado em chunks (8192 tokens) que dividem o step com decodes; evita travar streams ativos. O fable sugere `--long-prefill-token-threshold 2048` para estabilizar ITL sob concorrência.
48. **Amortização por batch.** Pesos lidos por step são quase constantes no tamanho do batch, então mais streams "pegam carona" (agregado 3.8× de ×1 a ×8, por stream −51%). É a razão do throughput agregado crescer até o step encher.
49. **Preempção e sobreinscrição.** Se a soma de contextos ativos excede o pool, o vLLM preempta e recomputa (não dá OOM). `1M × 8` excede o pool; o README tabela perfis de `MAX_MODEL_LEN × MAX_NUM_SEQS`.
50. **Determinismo em inferência paralela.** Somas em ponto flutuante não são associativas; ordem de redução/top-k varia entre runs e ranks. Tornar determinístico (ordenar top-k, finalize não-fundido) custa ~4%.

### Bloco F — Motor vLLM

51. **Arquitetura do vLLM V1 (API server, EngineCore, workers).** O processo de API recebe requisições, o EngineCore agenda e os workers GPU executam; comunicação por ZMQ/shared memory. Mensagens como "No available shared memory broadcast block found" (HANDOFF) vêm desse plano.
52. **PagedAttention e blocos de KV.** O KV é alocado em páginas (blocos) de tamanho fixo, com block tables por requisição; permite compartilhamento e fragmentação baixa. Bloco = 1600 tokens aqui.
53. **Hybrid KV cache allocator.** Em modelos híbridos o vLLM usa um pool comum para páginas de atenção e de estado Mamba/GDN, e eleva o bloco de atenção para que a página de atenção ≥ a página de estado. Por isso o dtype do SSM muda o tamanho do bloco e a capacidade em tokens.
54. **Prefix caching (hash de blocos).** Blocos completos com o mesmo prefixo são reutilizados entre requisições; só acerta em fronteiras de bloco (modo `align`). Medido 336 s → ~3–9 s em 600k.
55. **`gpu_memory_utilization`, profiling run e `--kv-cache-memory`.** O vLLM roda um forward de profiling, mede pico de ativação e dá ao KV o que sobra do orçamento `GMU × memória total`; `--kv-cache-memory` fixa o KV em bytes. Em memória unificada isso deve deixar folga ao SO.
56. **`max_model_len` e `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.** Contexto máximo aceito; acima de `max_position_embeddings` exige a flag de ambiente. Influencia buffers por step mesmo em prompts curtos.
57. **CUDA graphs (FULL_DECODE_ONLY, PIECEWISE, capture sizes, breakable).** Gravar a sequência de kernels de um step e reexecutá-la elimina overhead de launch da CPU; grafos só para tamanhos de batch capturados (1–64 tokens). Breakable graphs do v0.30 geraram variância no GB10.
58. **torch.compile / `compilation-config mode`.** Mode 0 = sem inductor; mode 3 compila e funde ops (+2–5 min de start). A receita usa mode 0 por estabilidade.
59. **Async scheduling.** Agenda o próximo step enquanto o atual roda na GPU; auto-ligado com MTP + executor `mp`.
60. **Carga de pesos (safetensors, shards, index, lazy).** Checkpoint dividido em shards com `model.safetensors.index.json` (`weight_map`); carga lazy lê só o necessário por rank. Shard corrompido só aparece a ~33% da carga.
61. **`--hf-overrides` e configs aninhadas.** Permite sobrescrever campos do `config.json` sem editar o cache; em modelos multimodais os campos de texto vivem em `text_config`. O bug do YaRN foi exatamente não aninhar.
62. **Backends de kernel (FlashInfer CUTLASS MoE, Triton, cuBLAS, autotune).** O vLLM escolhe implementações por arquitetura: MoE NVFP4 → FlashInfer CUTLASS SM120; QSA → Triton; GEMM densa → cuBLAS. FlashInfer faz autotune na 1ª requisição e guarda em `~/.cache/vllm` (cache divergente entre nós deadlockou TP2 no v0.30).
63. **Model code por arquitetura e overlays por bind-mount.** Cada família tem seu módulo (`qwen3_8_flash_next/nvidia/*.py`, `qwen4_exp/nvidia/*.py`); patchar = substituir um `.py` do pacote via `docker -v host:container:ro`. Técnica usada para todos os ~12 overlays.
64. **Parsers de reasoning/tool e endpoints OpenAI.** O servidor é OpenAI-compatible (`/v1/chat/completions`, streaming SSE, `stream_options.include_usage`, `usage.prompt_tokens_details.cached_tokens`). Tudo o que o bench mede passa por esses endpoints.
65. **Métricas Prometheus (`/metrics`).** Contadores cumulativos (`spec_decode_*`, `inter_token_latency_seconds`, `kv_cache_usage`, `preemption`); sempre medir **deltas** em torno do workload.

### Bloco G — Inferência distribuída

66. **Tensor parallelism (Megatron).** Cada matriz é partida entre GPUs por colunas (sem comunicação) ou por linhas (requer all-reduce da saída). Com TP=2 há um all-reduce por bloco (GDN out_proj, o_proj, MoE) — ~106 coletivas pequenas por step.
67. **Vocab-parallel embedding / LM head.** O vocabulário é dividido por **faixas de ids** entre ranks (rank 0: 0–124,159; rank 1: 124,160–248,319); logits precisam de all-gather ou de argmax local + troca de (valor, índice). Base do patch de draft vocab TP-aware.
68. **Expert parallelism e all-to-all.** Distribuir experts entre GPUs; normalmente exige dispatch/combine all-to-all de tokens. Com DP=1 (este caso) o vLLM não usa all2all: tokens são replicados e cada rank computa seus experts, somando via all-reduce de TP.
69. **Coletivas NCCL (all-reduce, all-gather) e latência de mensagens pequenas.** Em decode as mensagens são de 5–20 KB, dominadas por latência (20–30 µs cada via RoCE), não banda. Protocolos LL/LL128 e anéis afetam isso.
70. **Multi-nó no vLLM (`--nnodes`, `--node-rank`, `--master-addr/port`, `--headless`, executor `mp`).** Um processo por nó, rendezvous no head, worker sem API. Ordem de subida: worker primeiro.
71. **Comunicadores: PyNCCL, custom all-reduce, symmetric memory.** No GB10 multi-nó só PyNCCL funciona (custom all-reduce exige P2P no mesmo nó; symm-mem do NCCL exige world ≥4; MNNVL não existe em sm_121).
72. **Gloo e sockets de controle.** Coletivas de CPU e bootstrap usam TCP na interface definida por `*_SOCKET_IFNAME`.

### Bloco H — Rede e hardware

73. **RDMA, RoCE v2 e InfiniBand verbs.** Acesso direto à memória remota pela NIC, sem CPU; RoCE v2 roda sobre Ethernet/UDP. NCCL usa verbs via `NCCL_IB_*`.
74. **ConnectX-7, HCA, portas e GID index.** Cada porta física aparece como um HCA (`rocep1s0f0`); a GID table lista endereços (índice 3 = RoCEv2/IPv4 típico). Match exato com `=` e cabeamento cruzado entre nós são pegadinhas.
75. **GPUDirect RDMA / dmabuf.** A NIC lê/escreve direto na memória da GPU; no GB10 com memória unificada não está claro se o NCCL passa por buffers de host (fable sugere verificar `GDRDMA`/`dmabuf` no log).
76. **GB10 / DGX Spark (sm_121, 48 SMs).** GPU Blackwell de consumo/workstation integrada ao CPU ARM; muitos kernels/caminhos do vLLM são gated por arquitetura (SM90/SM100/SM103) e caem para Triton/emulação em sm_121. Explica metade dos patches.
77. **Memória unificada (UMA) e LPDDR5x.** CPU e GPU compartilham o mesmo pool físico: page cache, RAM de processos e alocações CUDA competem; esgotar o pool **trava o host** em vez de OOM; `nvidia-smi` mostra `[N/A]` para memória. Motivo do GMU 0.80, do despejo de page cache e do watchdog.
78. **Page cache, `posix_fadvise` e `drop_caches`.** Arquivos lidos ficam em cache de páginas limpas; em UMA isso "rouba" memória do CUDA. `POSIX_FADV_DONTNEED` libera só as páginas de um arquivo sem root; `echo 3 > /proc/sys/vm/drop_caches` limpa tudo mas precisa de root.
79. **UVM, cgroups e tunables de VM.** Alocações da GPU passam pelo UVM; o HANDOFF se contradiz sobre o cgroup: §1 afirma que alocações UVM foram cobradas no memcg do container (OOM-kill do EngineCore a 14.7 GiB com `docker --memory 16g`), e a correção posterior em §0 diz que **alocações de parâmetros da GPU NÃO são cobradas** (72 GiB de pesos apareceram como "used" no host com o container em 10 GiB) — o cap limita o lado host e o GMU limita o lado GPU. `vm.min_free_kbytes`, `watermark_scale_factor`, `swappiness` mudam quando o kernel recupera memória — ajudam no TP1, custam 7–14% no dual.
80. **CUDA stream memory ops.** `cuStreamWaitValue32/WriteValue32` sincronizam GPU e CPU por flags; o GB10 não suporta (`CAN_USE_STREAM_MEM_OPS=0`), o que deadlockava o offload da PLE após a captura de CUDA graphs.

### Bloco I — Operação e reprodutibilidade

81. **Docker para inferência GPU.** `--gpus all`, `--ipc host` (shared memory entre processos), `--network host` (NCCL/NFS/API sem NAT), `--ulimit memlock=-1` (registrar memória RDMA), `--device /dev/infiniband`, `--cap-add SYS_NICE`. Container como root → mounts em `/root`.
82. **Cache do Hugging Face Hub (blobs/refs/snapshots) e modo offline.** `models--org--name/{blobs,refs/main,snapshots/<sha>}`; `HF_HUB_OFFLINE=1` exige cache completo e `refs/main` sem newline (bug do hf_hub 1.28).
83. **Integridade (SHA-256 LFS vs git blob SHA-1, manifest da API).** Tamanho não basta; hash de conteúdo contra o manifest de `api/models/{repo}/tree/{rev}` detecta shard corrompido antes de 10 min de carga.
84. **Distribuição de pesos: rsync vs NFS.** Cópia local (disco, desacoplado) vs share de rede (sem disco, head sempre de pé, streaming a cada cold start).
85. **Metodologia de benchmark.** A/B com um launch por braço, repetições em ordem alternada, deltas de contadores, prompts sintéticos vs representativos, sweep de concorrência, amostragem de `MemAvailable` e `NV_ERR_NO_MEMORY`. O repo mostra que células individuais não reproduzem; médias sim.
86. **Riscos de segurança do serving.** API em `0.0.0.0` sem auth; checkpoint abliterado sem recusas; scripts de launch que podem conter `--api-key` (por isso 0600 e envio por stdin).

---

## 6. Receita passo-a-passo (recriar do zero, com o porquê de cada decisão)

### Passo 0 — Entender o orçamento antes de tocar em qualquer coisa
- Some o que precisa caber **por nó** (TP2 divide quase tudo por 2, mas HyperConnections, router e indexador são replicados): pesos ~64.5 GiB + non-torch ~4 GiB + ativação ~1 GiB + grafos ~0.5 GiB + KV. Com 121.69 GiB por nó e necessidade de deixar vários GiB ao SO (UMA!), o KV fica em ~28–32 GiB. **Raciocínio:** num Spark sozinho o checkpoint nvidia (133 GB) não cabe; o menor (99 GB) só cabe com PLE fora da GPU e ainda assim quase sem margem (TP1). Por isso 2 nós com TP=2.
- Decida o perfil: agentic/throughput (262K, 8 seqs) ou deep-context (1M, 2–3 seqs, YaRN — não validado).

### Passo 1 — Hardware e rede
1. Dois DGX Spark cabeados por ConnectX-7 (porta a porta). Anote em **cada** nó: nome da interface (`ip -o -4 addr`), HCA (`ibstat | grep "Port "`), IPs do link (ex.: `10.0.22.1/24` ↔ `10.0.22.2/24`). Os nós podem estar **cross-wired** (head `f1`, worker `f0`) → `WORKER_IFACE`/`WORKER_IB_HCA`.
2. SSH sem senha head→worker; Docker nos dois; ~126–135 GB livres por nó (ou só no head, se NFS).
3. Valide RDMA/NCCL isolado antes do vLLM (fable §6: `nccl-tests all_reduce_perf` com `NCCL_IB_HCA==<hca>`, `NCCL_IB_GID_INDEX=3`, `NCCL_DEBUG=INFO`). Se não subir, teste GID 5. **Nunca** liste um HCA ligado a outro cluster. **Por quê:** falhas de fabric aparecem como hang silencioso no init do vLLM, 1 min depois de começar.
4. Garanta que não há co-tenant de GPU (outro vLLM, ComfyUI etc.) em nenhum nó.

### Passo 2 — Imagem
- Escolha a lane: **day-0** `vllm/vllm-openai:qwen38-flash-next` (precisa dos patches; é o que tem FP8 KV nos kernels e determinismo) ou **vLLM 0.30.0** (`qwen4_exp` nativo; só o patch de draft vocab; BF16 KV default; TTFT multi-turno melhor). Puxe nos dois nós (ou `docker save | ssh worker docker load` se o worker não tem internet). **Por quê:** o código do modelo vive dentro da imagem; tudo o que a receita faz é substituir arquivos dela.

### Passo 3 — Pesos
1. Escolha o checkpoint: `nvidia/…NVFP4` (default; BF16 denso, NVFP4 experts, PLE FP8, MTP experts FP8-block), `RadixArk/…` (atenção BF16, PLE FP8, MTP experts BF16; mais lento em decode, base do FP8-dense), `local-inference-lab/…` (MXFP8 atenção, PLE NVFP4; o único que cabe em TP1), `Qwen/…-FP8` oficial (`start-fp8.sh`; ~500k tokens de KV; experts FP8 = mais bytes/step) ou o ablit gated (`ABLIT=1` + `HF_TOKEN` + aceitar termos).
2. `./download.sh` no head. Verifique completude (`resolve_snapshot.py` roda automaticamente) e **hash** (`./check-weights.sh --verify`; offline: `verify-weights.py --save-manifest ... --fetch-only` onde houver rota e `--manifest` nos nós).
3. Distribua: rsync (default, cold start mais rápido, nós independentes) ou `--nfs` (sem cópia; head deve ficar de pé; bom para trocar checkpoints). Se escrever `refs/main` à mão, use `printf`, não `echo`.

### Passo 4 — Descobrir e corrigir os metadados do checkpoint (por que existem os patches de config)
1. Leia `config.json` → `text_config` e `quantization_config`. Pergunte: a PLE declara `ple_embedding_dtype`? Se não, deduza do `config_groups` (`num_bits` 8 → `float8_e4m3fn`, 4 → `nvfp4`) e injete via `--hf-overrides '{"text_config":{...}}'` (**aninhado**).
2. As entradas `quantized_layers` da MTP usam índice relativo (`mtp.layers.0`)? O vLLM instancia a MTP em `num_hidden_layers + i` (48): adicione aliases em `config.json` **e** `hf_quant_config.json`, montados por cima do snapshot.
3. Qual o `quant_algo` dos experts da MTP? Se for `FP8_BLOCK_SCALES`/`FP8_PB_WO` (group 128), o dispatch ModelOpt precisa de um ramo para `Fp8MoEMethod` com `weight_block_size=[128,128]`; se for algo não suportado, rode sem MTP. **Por quê:** o sintoma de todos esses erros é um crash 7 min dentro da carga — preflight em segundos economiza horas.

### Passo 5 — Patches de kernel/loader (lane day-0)
Na ordem do `start.sh`, cada um extraído da imagem, patchado por âncora e montado `:ro` nos dois nós:
1. `ple_layer.py` (dispatch PLE FP8/NVFP4 misto).
2. `modelopt.py` = fallback MXFP8 → emulação para shapes fora do envelope (`N,K ≥128`, múltiplos de 32) e todo `visual.*` + ramo FP8_BLOCK_SCALES para MoE.
3. `ops/qsa.py` + `qsa.py` = FP8 KV (se `KV_CACHE_DTYPE=fp8*`); conflita com `QSA_PROFILE` e (sem fp8) com determinismo — escolha um.
4. `mtp.py` = draft vocab reduzido (conflita com `FP8_DENSE`).
5. `config/speculative.py`, `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py` = block-drop.
6. (opcional) `flashinfer_cutlass_moe.py` = determinismo.
**Por quê bind-mount e não rebuild:** a imagem day-0 tem ~20.6 GB e é um dev build; overlays são pequenos, versionáveis, regeneráveis a cada launch e aplicáveis nos dois nós com `scp`.

### Passo 6 — Memória unificada: preparar o host
- Antes de lançar: despeje o page cache do checkpoint (`evict_page_cache.py`, sem root) nos dois nós; se algo mais encheu o cache, `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- Use `GPU_MEMORY_UTILIZATION=0.80` (não 0.835): a 0.835 o `MemAvailable` chegou a 0.3–0.9 GiB, zona de hard-reset do GB10. Não aplique sysctls de reserva no dual (custaram 7–14% e não evitaram `NV_ERR_NO_MEMORY`).
- Não use `PLE_OFFLOAD` no dual (não há ~51 GB de RAM "extra" — é o mesmo pool).

### Passo 7 — Montar os argumentos do vLLM (e o porquê de cada um)
- `--tensor-parallel-size 2 --nnodes 2 --node-rank {0,1} --master-addr $HEAD_IP --master-port 50000 --distributed-executor-backend mp` (+ `--headless` no worker): um processo por nó, rendezvous no head.
- `--enable-expert-parallel --all2all-backend allgather_reducescatter`: cada GPU guarda 256 experts (metade dos bytes de expert por GPU); com DP=1 a coletiva continua sendo o all-reduce de TP.
- `--max-model-len 262144` (nativo) — ou 1M com YaRN `factor 4.0`, `original_max_position_embeddings 262144`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, benchmarkando antes de confiar.
- `--gpu-memory-utilization 0.80`, `--max-num-seqs 8`, `--max-num-batched-tokens 8192`, `--enable-chunked-prefill`.
- `--kv-cache-dtype fp8` (1.70× tokens, −5% throughput, validar qualidade) ou `auto`; `--mamba-ssm-cache-dtype bfloat16`.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"disable_eagle_block_drop":true,"index_share_for_mtp_iteration":true}'` + `VLLM_MTP_DRAFT_VOCAB` montado.
- `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`: grafos só no decode (onde o overhead de launch importa), sem inductor (estabilidade/tempo de start).
- `--mm-encoder-tp-mode data` (encoder de visão replicado).
- `--load-format safetensors --safetensors-load-strategy lazy`.
- `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prompt-tokens-details`.
- `--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"[, "rope_parameters":{...}]}}'`.
- Docker: `--gpus all --network host --ipc host --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband`, env NCCL (`NCCL_IB_DISABLE=0`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_IB_AUTO_DETECT=0`, `*_SOCKET_IFNAME`), `VLLM_HOST_IP`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=/root/.cache/huggingface`, mounts do cache HF e de `~/.cache/vllm`.
- Se quiser env extra no container, **edite os heredocs** (ou corrija o bug) — `EXTRA_DOCKER_ARGS` não chega ao `docker run` no `start.sh` atual.

### Passo 8 — Lançar e verificar
1. Worker primeiro (rank 1), 15 s, depois head. Siga `docker logs -f vllm-fn` até `/health` = 200 (~11 min).
2. Confira no log: `Available KV cache memory` e `GPU KV cache size` / `Maximum concurrency` (README:50), a linha do draft vocab (`47149 of 248320 tokens … 0.59 -> 0.22 GiB`), os WARNINGs de fallback MXFP8 por shape, `EAGLE trailing prefix-cache block dropping is disabled`, e as sugestões `--kv-cache-memory=<bytes>`.
3. Smoke tests: chat simples; `chat_template_kwargs.reasoning_effort` (`low|medium|xhigh`); tool call; imagem via URL; `max_tokens` 2–4k para reasoning.
4. Qualidade: `bench/reasoning_check.py --out bf16.json` em `auto` e `--compare` em `fp8`; `bench/longctx.py` para needles; se possível KLD/NLL entre configs.
5. Aceitação: `bench/mtp_accept.py --save` antes / `--since` depois; curva por posição deve decair (ex.: 0.81/0.64/0.51 em prose).
6. Performance: `bench/decodebench.py` (conteúdos diferentes, T=0 e 0.8), `bench/sweep.py` (S=1/2/4/8, 3 repetições alternadas, sempre medir `MemAvailable`).

### Passo 9 — Tuning (em ordem de alavanca, segundo o próprio repo)
1. **Menos bytes por step**: draft vocab reduzido (feito, +8–10%); FP8-dense (analítico +40–60% batch-1, não medido); `MTP_NUM_SPECULATIVE_TOKENS=2` se a posição 3 aceitar <~35%.
2. **Mais tokens aceitos**: vocab do drafter construído sobre a saída do seu tráfego (`gen_draft_corpus.py` → `build_draft_vocab.py --balance-shards 2`), index share.
3. **Mais sequências por step**: concorrência (até o limite de `MAX_NUM_SEQS` e do KV); perfil de `MAX_MODEL_LEN`.
4. **Piso de comunicação**: env NCCL/segundo link CX7 (não adotados; medir com nccl-tests).
5. **Kernels**: QSA profile para 48 SMs (benchmarkar com `bench_qsa_kernels.py`), skinny GEMM em sm_121, MoE B12X, GDN prefill FlashInfer SM120, torch.compile mode 3 — todos experimentos listados, nenhum adotado.
6. **Multi-turno/TTFT**: block-drop (feito); lane v0.30 (TTFT 0.28–0.40 s).
7. Re-meça depois de mudar `MAX_MODEL_LEN`, `MTP_NUM_SPECULATIVE_TOKENS`, `KV_CACHE_DTYPE`, `MTP_DRAFT_VOCAB`, `FP8_DENSE` ou `QSA_PROFILE` (README:947).

### Passo 10 — Operação contínua
- `./stop.sh` (e `--nfs` para derrubar o share) antes de trocar de checkpoint/lane (todas usam o nome `vllm-fn` e a mesma porta).
- Não pare `vllm-fn-nfs` durante carga/serving.
- Sem restart automático (rank sozinho trava no NCCL); preferir supervisor que faça `stop && start --launch` em falha de `/health` (fable §4.4).
- Monitore `MemAvailable` e `journalctl -k | grep NV_ERR_NO_MEMORY`; `/metrics` para `kv_cache_usage` e `preemption`.
- Após atualizar a imagem: apague os `.orig` extraídos (`files/*.orig`, `files/ple_layer_patched.py`, `files/modelopt_patched.py.orig`) para forçar re-extração — âncoras que não casam abortam o patch em vez de gerar arquivo quebrado.

---

## Apêndice — Diretórios fora do escopo de launch
- `html/` (100 páginas HTML, `BRIEF.md`, `concepts.md`, `render100.py`, `verify100.py`, `shots/`): um conjunto de amostras/sondas de geração de páginas HTML ("rendering probes and a 100-shot visual sample set", CHANGELOG:400–401) — é uma vitrine/teste de capacidade do modelo, não parametriza o serving.
- `.github/` (templates de issue/PR, FUNDING): não afeta a receita.

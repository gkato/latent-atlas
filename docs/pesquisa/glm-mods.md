# Inventário GLM-5.3-Flash EXL3 (2x DGX Spark) — overlays, kernels, extensões, ferramentas, abliteração e docs

Repositório analisado (somente leitura): `repos/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (MiaAI-Lab).
Escopo deste arquivo: `overlay/**` (incluindo `exl3.py`, `exl3_fat_moe.cu`, `exl3_fat_gemm.cu`, todos os `patch_*.py`, `ablit_runtime.py`, `qwen3_dflash2.py`, `dflash2_speculator.py`, `overlay/tp3/**`), `extensions/cooperative_moe/**` (runtime, kernels nativos, headers vendorizados do `exllamav3_ext`), `kernel_lab/**`, `tools/**`, `ablit/**`, `docs/*.md` + `docs/qualification/*`, `tests/*` (usados para entender o que cada patch garante). Também li o `README.md`, o `Dockerfile`, o `Dockerfile.e3-layer`, o `.env.example` e trechos do `start.sh` porque sem eles os patches não fazem sentido (o `Dockerfile` contém patches inline essenciais para o backend de atenção).

Convenções: identificadores, flags, variáveis de ambiente e código ficam no original. Referências `arquivo:linha` são aproximadas ao que foi lido (linhas do checkout local). Quando algo não está explícito no repositório e eu completo com conhecimento externo, marco como **[interpretação]**. Todos os números vêm do repositório (README/docs/logs citados), com a data que o próprio repo informa.

---

## 0. Mapa rápido do que existe (para orientar a leitura)

| Camada | Arquivos principais | Papel |
|---|---|---|
| Imagem base + patches de backend | `Dockerfile` (linhas 1–368) | Faz o GLM-5.3-Flash NoPE MLA rodar no backend `FLASHINFER_MLA_SPARSE_SM120` (padding 512→576, top-k 2048, page 64, desliga autotune/PDL no SM121) |
| Quantização EXL3 no vLLM | `overlay/exl3.py`, `overlay/patch_model_overrides.py`, trecho do `Dockerfile` 387–419 | Registra `exl3` como método de quantização do vLLM, carrega experts como trellis empacotado, TP-shard, MoE fundido, tiers de "fat experts" (legacy→E3), FP8 denso, retenção BF16, EXL3 denso |
| Kernels CUDA próprios | `overlay/exl3_fat_gemm.cu/.cuh` (E2), `overlay/exl3_fat_moe.cu/.cuh` (E3), `overlay/patch_exl3_fat_kernel.py`, `overlay/build_exl3_fat_moe_ext.py`, `overlay/patch_exl3_decode_pipeline.py` (thin-decode FAST), `overlay/patch_exl3_ext_aarch64.py` | Compilados dentro do `exllamav3_ext` (pin `c5d9c657`, 0.0.43) para `sm_121a` |
| MoE cooperativo (decode) | `extensions/cooperative_moe/**` | Kernel de 2 estágios do Turboderp especializado para H=4096, I_local=1024, top-k 8, K4 MCG, 1–32 linhas (TP2) e variante TP3/EP (I=2048, 96 experts, 32/64 linhas) |
| Speculative decoding DFlash2 | `overlay/qwen3_dflash2.py`, `overlay/dflash2_speculator.py`, `overlay/patch_dflash2.py`, `overlay/patch_dflash2_exl3.py`, `overlay/patch_glm_eagle3.py`, `overlay/patch_adaptive_k.py` | Drafter DFlash2 (conv agrupada dinâmica + seletor de candidatos), captura de hidden auxiliares EAGLE3 no mHC, verificação adaptativa (adaptive-k) |
| KV cache híbrido e prefix caching | `overlay/patch_glm5_drafter_group.py`, `overlay/patch_hybrid_prefix_hit.py`, `overlay/patch_apc_per_group_retention.py`, `overlay/patch_apc_no_store.py`, `overlay/patch_mamba_align_state_free.py`, `overlay/patch_mamba_align_chunking.py`, `overlay/patch_kv_capacity_log.py`, `overlay/patch_kpool_tail_slotmap.py`, `overlay/patch_kpool_tail_seed_stride.py`, `overlay/patch_indexer_workspace.py` | Grupo do drafter dividindo páginas do MLA, correções de APC com o drafter EAGLE-like, retenção por grupo, no-store por requisição, correções de estado Mamba/KDA, correções do indexer esparso |
| Scheduler | `overlay/patch_scheduler_decode_floor.py` (fair v5) | Política de mistura prefill/decode por tempo de serviço |
| Carregamento/memória | `overlay/patch_cold_load_uma.py`, `overlay/patch_loadclone.py`, `overlay/patch_skip_cudagraph_profile.py` | InstantTensor em UMA com page cache cheio, staging de mmap 64 KiB, read-ahead |
| API/serving | `overlay/patch_suppress_stops_in_reasoning.py`, `overlay/patch_tool_choice_none.py`, `overlay/patch_xgrammar_termination.py`, `overlay/patch_cache_reset.py`, `overlay/patch_default_max_new_tokens.py`, `overlay/patch_spinwait.py`, `overlay/patch_glm_video_placeholders.py` | Correções de comportamento do servidor OpenAI |
| Abliteração | `overlay/ablit_runtime.py`, `overlay/patch_ablit.py`, `ablit/*` | Edição de `o_proj` em tempo de carga (transplante ou projeção de direção de recusa) |
| TP=3 / TP=4 | `overlay/tp3/**`, `overlay/patch_flashkda_tp3.py`, `overlay/patch_sparse_mla_slice.py` | Padding de cabeças 64→66, vocab, shared-expert, A_log, EP; FlashKDA; slicing do sparse-MLA no TP4 |
| Ferramentas | `tools/dense_overlay.py`, `tools/pack_profile.py`, `tools/dflash2_exl3_*.py/.sh`, `tools/stage_dense_h3.py` | Montar pack "dense EXL3 H3" e drafter EXL3 6 bpw |
| Laboratório | `kernel_lab/**` | Oráculo NumPy do formato EXL3/MCG K1–K8 e sweep de táticas SM121 (não entra na imagem) |

---

## 1. Visão geral

### 1.1 O modelo alvo: GLM-5.3-Flash (arquitetura, como o repositório a descreve)

Fontes: `docs/improve-prefill.md` §2 ("Roofline"), `ablit/LAYER_MAP.json`, `README.md` ("Why the overlay exists", "3x Spark"), `overlay/tp3/*`, `extensions/cooperative_moe/*`.

- **45 camadas de decoder** (`num_hidden_layers=45`, `LAYER_MAP.json`), mais um bloco **MTP** no índice 45 (checkpoint MTP block, usado quando `SPEC_METHOD=mtp`).
- **Atenção híbrida**:
  - **34 camadas `linear_attention` = KDA** (ex.: camadas 0,1,2,4,5,6,8,…). No vLLM elas são modeladas como camadas de estado ("Mamba-like"): o estado recorrente fica em `MambaSpec` (`mamba_cache_mode="align"`), em 4 grupos Mamba (`docs/DESIGN-apc-per-group-retention.md` §1). KDA tem parâmetros `A_log`, `dt_bias`, gate limitado (`lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`), `beta`, normalização L2 de q/k no kernel (`overlay/patch_flashkda_tp3.py`), convolução curta (`conv_state`), estado recorrente FP32 `[heads, 128, 128]` por sequência. O repositório não expande a sigla; **[interpretação]** KDA = *Kimi Delta Attention* (atenção linear com regra delta com gate por canal, da família Gated DeltaNet — o backend usa `GDNAttentionMetadata` e kernels `chunk_kda_*`/`chunk_gla`, e o FlashKDA é descrito como "the fused CUDA kernel Kimi-K3 already uses").
  - **11 camadas `deepseek_sparse_attention`** (camadas 3,7,11,…,43 e o MTP 45): **MLA NoPE** (`qk_rope_head_dim=0`, `kv_lora_rank=512`) com **atenção esparsa estilo DSA**: um *indexer* ("kpool") pontua blocos comprimidos de 4 tokens (`index_kpool=4`), escolhe `select_k=512` pools e expande para ~2048 tokens candidatos (`index_topk=2048`) + uma "cauda" de até 3 tokens ainda não comprimidos. O FlashInfer chama essa geometria de `GLM_NSA`.
- **MoE**: camadas 0–2 têm **MLP denso** (`intermediate_size=12288`); a partir da camada 3, **288 experts roteados, top-8, +1 shared expert**, `moe_intermediate_size=2048`, `hidden=4096`, SwiGLU com limite (`swiglu_limit`, default do overlay 10.0).
- **mHC** (hyper-connections com 4 fluxos residuais): o forward da camada devolve `(hidden_states, residual, post, comb)`; para extrair um hidden de 4096 é preciso `hc_post` + `hc_contract` (`overlay/patch_glm_eagle3.py:70-83`). **[interpretação]** mHC = *manifold-constrained Hyper-Connections*, residual multi-fluxo.
- **Vocab 154.880**, **64 cabeças de atenção** (e 64 cabeças lineares KDA), visão (image+video, torre `visual`), template de chat GLM com raciocínio (`<think>`).
- Parâmetros ativos ≈ **19,5 B/token**, ≈ **36 GFLOP/token** (tabela de `docs/improve-prefill.md`): experts roteados 8,46 B (47% dos FLOPs), KDA 5,8 B (33%), MLA esparso+indexer 1,66 B (9%), shared 1,06 B (6%), MLP denso 0,45 B, lm_head 1,27 GFLOP.

### 1.2 O checkpoint: EXL3/TR3 4 bpw somente nos experts roteados

- `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (espelho byte-idêntico de `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, snapshot `5ab363a8…`, ~164 GiB, 120 shards).
- `quantization_config`: `quant_method=exl3`, `codebook=mcg`, `bits=4`, `scope=glm53_routed_experts_only` (`overlay/exl3.py:4-7`, `from_config` em `exl3.py:1824-1839`). Existe também um ledger de 37 MiB (`tensor_storage`) que o overlay descarta do objeto de config (`exl3.py:1831-1832`).
- Cada matriz de expert tem 4 tensores: `trellis` (int16, `[K/16, N/16, 16*bits]` = 64 palavras por tile 16×16 em K4), `suh` (fp16, vetor de sinal/escala na entrada), `svh` (fp16, na saída), `mcg` (int32 marcador `0xCBAC1FED` = `-877912083`). Todo o resto (atenção, KDA, shared, MLP denso, embeddings, lm_head) está em BF16 nativo.
- Qualidade (KLD, painel independente, pesos e não o overlay): 4 bpw = **0,024555 nats**, 176 GB, ≈ FP8 oficial no mesmo stack (0,024629) com 54% dos bytes (`README.md` "Quality (KLD)").

### 1.3 O problema de engenharia

A imagem `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (vLLM `0.1.dev20051+g487ecf187`) não serve esse checkpoint no GB10 (SM121): (a) morre no primeiro forward com `pe_dim must be 64 for fp8_ds_mla` porque o modelo é NoPE; (b) não conhece o método `exl3`; (c) não tem DFlash2, nem os hooks EAGLE3 do GLM; (d) o KV cache híbrido desperdiça blocos com o drafter; (e) o scheduler trava prefills concorrentes; (f) o carregamento em UMA falha com page cache cheio; etc. O repositório resolve isso em três momentos:

1. **Build da imagem** (`Dockerfile`): patches inline no backend sparse-MLA SM120, registro do `exl3`, compilação do `exllamav3_ext` com os kernels E2/E3/FAST para `sm_121a`, instalação de todos os overlays e testes em CPU.
2. **Start do container** (`start.sh`, `GLM53_OVERLAY_ORDER`, `start.sh:2053-2082`): reaplica 25 overlays de forma idempotente e *fail-closed* nos dois ranks (mesma ordem, mesmo ambiente — ranks que capturam CUDA graphs diferentes travam o NCCL).
3. **Runtime**: env vars (`.env`) controlam os caminhos opcionais (FP8 denso, adaptive-k, E3, cooperativo, retenção de APC, fair scheduler…).

### 1.4 Fluxo ponta a ponta de uma requisição (com os pontos onde cada patch atua)

```
[carga] InstantTensor (patch_cold_load_uma: orçamento vs MemAvailable) / auto-safetensors (patch_loadclone + clone 64KiB)
   -> Exl3Config.get_quant_method (exl3.py:1853):
        RoutedExperts -> Exl3MoEMethod (trellis empacotado, TP shard col/row)
        LinearBase    -> Exl3LinearMethod (pack dense EXL3) | Glm53DenseFp8Method (GLM53_DENSE_FP8) | BF16
   -> process_weights_after_loading: LinearEXL3 por expert, tabelas de ponteiros, temps do exl3_moe,
      resolução do tier "fat" (grouped/kernel/...), FP8 per-channel + Marlin, cópia BF16 large-M
   -> fim de load_weights: ablit_runtime.maybe_apply (o_proj L15-45)
[KV] get_kv_cache_groups (patch_glm5_drafter_group): grupos MLA(+indexer) | KpoolTail | 4x Mamba(KDA) | drafter SWA (slot-share)
[scheduler] fair v5 (patch_scheduler_decode_floor) + alinhamento Mamba (patch_mamba_align_chunking) + adaptive-k
[APC] coordinator (patch_hybrid_prefix_hit, per-group retention, no-store) -> hits alinhados a 3584 tokens
[forward prefill] KDA (Triton chunk_kda ou FlashKDA) | MLA esparso (indexer kpool -> top-2048 -> FlashInfer GLM_NSA 576)
                  MoE: exl3_moe (experts "thin") + E3 grouped (experts "fat")
[forward decode]  MoE: exl3_moe (ou FAST K4/N256, ou cooperativo 2 estágios) ; dense FP8 Marlin
[spec decode] DFlash2: drafter 5 camadas SWA (KV bf16, contexto projetado dos hidden EAGLE3 do alvo) -> 7 tokens
              -> seletor de candidatos (walk Gumbel) -> verificação pelo alvo (rejection sampling) com k adaptativo
[saída] detokenizer (stop dormente até </think>), bad_words p/ tool_choice none, xgrammar termination
```

---

## 2. Inventário de parâmetros

Formato: **nome** | valor(es) default e alternativos | onde (arquivo:linha) | o que faz | conceito(s) | por que importa / trade-off.

### 2.1 Quantização EXL3 e MoE roteado (`overlay/exl3.py`)

| Nome | Default / alternativas | Onde | O que faz | Conceitos | Por que importa / trade-off |
|---|---|---|---|---|---|
| `--quantization exl3` / `QUANTIZATION` | `exl3` (`.env.example:103`); nunca `marlin` | `start.sh` monta `--quantization`; `Exl3Config.override_quantization_method` (`exl3.py:1841-1851`) | Seleciona o método de quantização do vLLM registrado pelo overlay | Registry de quantização do vLLM, QuantizationConfig | Sem o registro, o vLLM expandiria/rejeitaria o checkpoint; `--moe-backend marlin` é o workaround NVFP4 e é errado aqui |
| `bits` (config do checkpoint) | 4 (aceita 3,4,5,6 — `exl3.py:1726`) | `Exl3Config.__init__` | Bits por peso do trellis (K) → `k_words = bits*16` palavras int16 por tile 16×16 | Quantização por treliça (QTIP/EXL3), bpw | K4 = 64 palavras/tile; os kernels E2/E3/coop só suportam K4 MCG |
| `codebook` | só `mcg` (`exl3.py:1722`) | idem | Codebook procedural MCG (multiplicador `0xCBAC1FED`) | Codebook procedural ("3INST"), MCG | O overlay recusa outros codebooks para experts; `mul1` (`0x83DCD12D`) só é aceito nas camadas densas EXL3 |
| `scope` | `glm53_routed_experts_only` | idem | Documenta que só os experts roteados estão quantizados | Quantização mista | Todo o resto fica em BF16 nativo (ou FP8 via overlay) |
| `EXL3_FUSED_MOE` | `1` (`.env.example:108`); `0` = loop Python com `LinearEXL3` | `exl3.py:761-762` | Usa o kernel fundido `exllamav3_ext.exl3_moe` (1 lançamento por camada) | Kernel MoE fundido | `0` só para depuração; `GLM53_EXL3_MOE_FAST=1` exige `1` |
| `EXL3_TEMP_ROWS_FUSED` | código: 128 (`exl3.py:66`, `177-181`); launcher: **32 com E3**, **256 com E2** (`start.sh:361-363`) | `temp_rows_fused()` | Número de linhas (tokens) por expert que cabem nos buffers temporários do `exl3_moe`; experts com mais linhas são "fat" | Scratch do kernel MoE, experts "thin" vs "fat" | Regra do README: manter ≥ `MAX_NUM_SEQS × (DFLASH_TOKENS+1)` = 4×8 = 32 para o decode continuar num único lançamento graph-safe. Medido (P2b, `improve-prefill.md`): subir para 1024 deixou prefill 13–24% mais lento |
| `EXL3_MOE_ROW_TILE` | `0`; `1` = fatias de 128 linhas na GPU | `exl3.py:168-174`, `_exl3_moe_row_tiles` 1405-1457 | Lança `exl3_moe` uma vez por fatia de 128 linhas do buffer ordenado | Row tiling | Medido (P2a): −61% a −18% de tok/s de prefill; mantido desligado |
| `EXL3_FAT_SORTED` | `0` | `exl3.py:183-185` | Tier "sorted": roda experts fat a partir de fatias contíguas do sort (1 `counts.tolist()` em vez de N syncs) | Host sync, ordenação por expert | Tier intermediário; implícito pelos tiers superiores |
| `EXL3_FAT_BATCHED` | `0` | `exl3.py:187-189` | Tier E1: buffers persistentes + cópia dos counts D2H em stream lateral sobrepondo o lançamento dos experts thin | Streams CUDA, pinned memory | Requer SUH compartilhado entre gate/up |
| `EXL3_FAT_KERNEL` | `1` (`.env.example:120`) | `exl3.py:191-193` | Tier E2 (PR77): GEMM direto sobre o trellis (`exl3_fat_gemm`) + GEMM com scatter (`exl3_fat_gemm_scatter`) para os experts fat | GEMM direto em pesos quantizados (sem reconstruir BF16) | +20–21% em prefill frio (PR77, 2026-09-01) |
| `EXL3_FAT_GROUPED` | `1` (default desde 2026-09-07; código default `0`, `exl3.py:196-206`) | idem | Tier E3: 3 lançamentos por camada (gather, gate/up+SwiGLU, down+scatter) dirigidos por tabelas de segmentos no device, sem host sync, capturável em CUDA graph | Grouped GEMM, tabelas de segmento, prefix-sum no device | +37–45% vs E2 em prefill frio; custa 560 MiB de scratch cobrado do orçamento de KV (1M não cabe mais a util ≤ 0,87) |
| `EXL3_FAT_GROUPED_TOPK` | `8` | `exl3.py:1113` | Top-k usado para dimensionar o scratch E3 (`MNBT × topk` linhas) | Dimensionamento de scratch | Evita realocação em prefill estável; crescer dentro de captura de graph é erro (`exl3.py:1119-1123`) |
| `EXL3_FAT_SCRATCH_ROWS` | vazio → usa `MAX_NUM_BATCHED_TOKENS` | `exl3.py:907-913`, `1106-1112` | Capacidade de linhas do scratch E1/E2/E3 | Pré-alocação | Scratch E2 arredonda para potência de 2 (`exl3.py:914-915`) |
| `EXL3_FAT_EXPERT_LOG` | default `1` se não-E3, `0` com E3 (`exl3.py:209-214`) | idem | Histograma de roteamento por passo (`exl3 fat-expert P0`) | Instrumentação | Custa 1 host sync por camada MoE no tier E3 |
| `MAX_NUM_BATCHED_TOKENS` (lido pelo exl3.py) | `7168` (`.env.example:202`) | `exl3.py:910`, `1109` | Também dimensiona scratch dos tiers fat | Chunked prefill | Ver 2.6 |
| `GLM53_EXL3_MOE_FAST` | `0`; `1` = kernels thin-decode K4/N256 | `exl3.py:1266-1276`; nativo `patch_exl3_decode_pipeline.py:80-97` | Seleciona `glm53_exl3_moe_fast_kernel<4,256,shared_input>` (1 estágio de fragmentos, 8 estágios de SMEM; stock 3/3) só em SM121 | Pipeline de registradores vs SMEM, spills | +7,71/+8,88/+14,59% decode (TheGrill A/B/A2); fail-closed se a imagem não tem o patch; só TP2 |
| Constantes: `EXLLAMAV3_COMMIT` | `c5d9c657…` (0.0.43) | `exl3.py:55-56`, `Dockerfile:380` | Pin do ExLlamaV3 compilado | Reprodutibilidade | Expõe `exl3_moe` e `exl3_moe_max_concurrency` |
| `MCG_MULTIPLIER` / `MCG_MARKER_SIGNED_INT32` | `0xCBAC1FED` / `-877912083` | `exl3.py:57-58` | Validação do marcador do codebook em cada expert (`exl3.py:3145-3153`) | ABI do formato | "packed ABI mismatch" se o checkpoint não for MCG |
| `MUL1_MARKER_SIGNED_INT32` | `-2082680531` (`0x83DCD12D`) | `exl3.py:59-61` | Codebook `mul1` das camadas densas EXL3 (turboderp) | Codebook mul1 | Exatamente um marcador por shard (`exl3.py:2754-2771`) |
| `SWIGLU_LIMIT_DEFAULT` | `10.0` | `exl3.py:63`, `3245` | Clamp do SwiGLU: `silu(min(g,limit)) * clamp(u,-limit,limit)` | SwiGLU com limite (estilo gpt-oss) | O coop TP3 só aceita limite 10 (0,5 falhou o screen) |
| `EXL3_FAT_MOE_MIN_CAPABILITY` | `(9,0)` | `exl3.py:249-250` | E3 usa atomics vetoriais de 16 B (`red.v4`, sm_90+) | Atomics vetoriais | Imagem compila `sm_121a` |
| `_FAT_BUCKET_EDGES` | `(16,32,64,128,256,512,1024,2048)` | `exl3.py:75` | Buckets do histograma de linhas máximas por expert | Diagnóstico | — |
| `EXL3_FAT_DIAG_SCHEMA` | `2` | `exl3.py:91-129` | Contrato do diagnóstico `exl3 e2 diag …` (tier configurado/efetivo, símbolos, capacidade, contadores) | Observabilidade | Chaves fixas; mudança exige bump do schema |

### 2.2 Projeções densas: FP8 (Marlin), retenção BF16 large-M, EXL3 denso

| Nome | Default / alternativas | Onde | O que faz | Conceitos | Por que importa / trade-off |
|---|---|---|---|---|---|
| `GLM53_DENSE_FP8` | `.env.example:158` = `all`; se ausente = `off`; grupos `shared,dense,kda,mla` (TP3 template: `dense,kda`) | `exl3.py:1895-1982`; `patch_dense_fp8.py:84` | Quantiza em tempo de carga (per-output-channel, e4m3, escala `amax/448`) as projeções densas do alvo e as executa com Marlin W8A16 | FP8 e4m3, quantização weight-only, per-channel scale, Marlin | ~11 ms a menos por passo; `all` vs `dense,kda`: −4,67%/−4,91% ms/ciclo, +4–5% tok/s (2026-09-26); muda numérica (KL de texto 25–30% acima dos pares stock, "NOT-WORSE") |
| Grupos (sufixos) | `shared`: `mlp.shared_experts.{gate_up_proj,down_proj}`; `dense`: `mlp.{gate_up_proj,down_proj}` das camadas 0–2; `kda`: `self_attn.{in_proj_qkvbfg_a,f_b_proj,g_b_proj,o_proj}`; `mla`: `self_attn.{fused_qkv_a_proj,q_b_proj,o_proj}` | `exl3.py:1912-1917` | Define quais módulos viram FP8 | Projeções MLA (q_a/kv_a fundidos, q_b), KDA (in_proj q/k/v/b/f_a/g_a) | `kv_b_proj` fica BF16 (a absorção de pesos do MLA lê o peso diretamente); drafter, MTP e visão nunca são reivindicados (`exl3.py:1950-1967`) |
| TP=3 exceção | — | `exl3.py:1985-2001`, `2112-2118` | KDA `f_b_proj`/`g_b_proj` ficam BF16 em TP3 (pitch de 8.726 não é múltiplo de 8; offset não alinhado a 16 B) | Requisitos de alinhamento do Marlin | Log `[glm53-dense-fp8] TP=3 keeps KDA f_b_proj/g_b_proj in BF16` |
| `GLM53_KDA_BF16_LARGE_M` | `.env.example:162` = `1`; código default `0` | `exl3.py:2032-2037`, `2146-2248`, `2250-2281` | Guarda uma cópia BF16 do **mesmo peso FP8 lógico** (e4m3 × escala BF16 armazenada) do `in_proj_qkvbfg_a` do KDA; se `M > 512` usa `F.linear` BF16 (cuBLAS), senão Marlin | Crossover de GEMM por tamanho de M, dispatch sem host sync | TTFT frio −11,4% (16k), −12,3% (100k), mixed −14,1%; custa +3,29 GiB/rank (98,25 MiB × 34 camadas); só TP2/TP3 em SM121 |
| `KDA_BF16_LARGE_M_MIN_M` | `512` (não configurável) | `exl3.py:2015` | Fronteira do dispatch | M = linhas agendadas | Decode nunca passou de M=220; banda 221–511 vazia nos passos medidos; prefill ≥768 |
| `KDA_BF16_LARGE_M_SHAPES_BY_TP` | TP2 `(12576,4096)`, TP3 `(8726,4096)` | `exl3.py:2019-2023` | Formas TP-locais qualificadas | Sharding TP da projeção fundida | Fail-closed em qualquer outra forma |
| `KDA_BF16_LARGE_M_CHUNK_ROWS` | `512` | `exl3.py:2026` | Construção da cópia em blocos (≈8 MiB fp32 de scratch) | Pico de memória na carga | Não muda o resultado (produto e4m3×bf16 cabe em 12 bits significativos, exato em fp32) |
| `KDA_LARGE_M_STATS_PATH` | vazio | `exl3.py:2084` | Dump JSON dos contadores bf16/marlin a cada 512 chamadas | Observabilidade | Não conta replays de CUDA graph |
| `GLM53_DENSE_EXL3` | `0`; `1` = serve pack com `non_routed_exl3` | `patch_dense_fp8.py:85-94`, `exl3.py:1728-1754` | Habilita `Exl3LinearMethod` para linears densos declarados pelo pack | EXL3 denso, pack overlay | Pack com `non_routed_exl3` + flag 0 é recusado (os BF16 substituídos não existem no índice) |
| `quantization_config.non_routed_exl3.layers` | `{prefixo: {"bits": K, "bf16_shards":[...]}}` | `exl3.py:1733-1754` | Declara quais módulos densos são EXL3 e quais shards de saída ficam BF16 (cauda contígua) | Mixed precision por shard | `kv_b_proj` é proibido (absorção MLA) |
| `GLM53_DENSE_EXL3_PREFILL_BF16` | `off`; `all`; lista de `shared_gate_up,shared_down,dense_gate_up,dense_down,kda_in,kda_o,mla_qkv_a,mla_q_b,mla_o`; perfil H3 = `kda_in,shared_down,mla_qkv_a,shared_gate_up,kda_o,mla_q_b` | `exl3.py:2294-2380`, `2876-2947` | Mantém cópia BF16 reconstruída (`get_weight_tensor()`, Hadamards já aplicados) para M > 144 | Reconstrução vs GEMM direto no trellis | O gargalo do prefill denso EXL3 são as passadas de Hadamard/cast em volta do reconstruct, não o reconstruct em si |
| `DENSE_EXL3_PREFILL_BF16_MIN_M` | `144` | `exl3.py:2310` | = `AUTO_RECONSTRUCT_THRESHOLD` do `LinearEXL3` | Crossover da biblioteca | Separado do 512 (Marlin) de propósito |
| `GLM53_MODEL_PRESET` | vazio; `dense-h3` | README "Build the H3 pair"; `tools/*` | Constrói/seleciona o par alvo "dense EXL3 H3" + drafter EXL3 6 bpw | Perfil de pack | Opt-in, TP2 only, não A/B medido contra o default |
| `glm53_profile` (config.json do pack) | `name=dense-exl3-h3-dflash2-6bpw`, `draft.model/revision/config_sha256` | README tabela "Pack-selected dense EXL3"; `tools/pack_profile.py` | Liga o alvo ao drafter pareado por hash | Proveniência | Valida inventário e cabeçalhos antes de parar containers |

### 2.3 MoE cooperativo (decode) — `extensions/cooperative_moe/`

| Nome | Default / alternativas | Onde | O que faz | Conceitos | Por que importa |
|---|---|---|---|---|---|
| `EXL3_OVERLAY_HOST` | vazio = `overlay/exl3.py`; caminho de um overlay gerado | README; `prepare_profile.py` | Seleciona um `exl3.py` que termina com `install(..., enabled=True)` do adaptador cooperativo | Overlay gerado com pin de hash | É a única forma real de ativar; `GLM53_COOPERATIVE_MOE=1` sozinho não faz nada |
| `GLM53_COOPERATIVE_MOE` | `0` | `runtime.py:369` | Habilita `install()` em integrações diretas | — | — |
| `GLM53_COOP_GEOMETRY` | `1` (A-wide/B-wide); `0` both-narrow; `2` A-wide/B-narrow (auto Blackwell) | `runtime.py:46-59`, `107-123` | Geometria dos tiles dos kernels A (gate/up) e B (down) | Tiling, tamanho de grid | Geo 2 lança `rows*topk*(H/32)` = 8192 blocos de down com 8 linhas (vs 2048 em geo 1); perda de decode neste kit |
| Constantes TP2 | `HIDDEN=4096`, `INTERMEDIATE_LOCAL=1024`, `TOPK=8`, `ROWS_MAX=32`, `SLOTS_MAX=256`, `EXPERTS_MAX=288`, `BITS=4`, `CTR_LEN=3587`, `PARAMS_SIZE=344` | `runtime.py:39-63`, `native/cooperative_moe.cu:11-21` | Especialização em tempo de compilação | Especialização de kernel | Chave de ordenação `(expert<<8)|slot` limita a 256 slots |
| SHA256 do `.so` | `aa3fe5e9…` | `runtime.py:27`, `prepare_profile.py:15` | Recusa binário não validado | Supply-chain/qualificação | Rebuild com outro toolchain muda o hash → requalificar |
| Variáveis de sobreposição proibidas | `VLLM_UBATCH*`, `VLLM_DUAL_BATCH*`, `VLLM_ENABLE_DBO`, `--enable-dbo`… | `runtime.py:64-80`, `126-144` | Rejeita configurações que sobrepõem lançamentos | Scratch compartilhado por device | Contadores do kernel A/B são reutilizados entre camadas |
| TP3 (ABI 2) | local I=2048, até 96 experts locais, 32 e 64 linhas (256/512 slots), chave de 9 bits, limite SwiGLU 10 | `extensions/cooperative_moe/tp3/README.md`, `native/dispatch.cu` | Variante EP | Expert parallelism | Política por número de linhas em `dispatch_policy.json` (geo 2 em 2,3,4,5,10,18,24; geo 1 em 1,6–9,12,15,16,20,25,30,32,35,40,48,56; stock no resto; 64 linhas = stock) |
| `GLM53_COOP_QUALIFICATION`, `GLM53_COOP_MAINTENANCE_TEST`, `GLM53_COOP_BUNDLE`, `GLM53_COOP_EP_RANK`, `GLM53_COOP_TEST_HELPERS` | vazios | `tp3/runtime.py:270-307, 459-465`, `tp3/README.md` | Modos de manutenção/qualificação | Qualificação numérica | Só em janela reservada, nunca junto de produção |

### 2.4 Speculative decoding (DFlash2, MTP) e adaptive-k

| Nome | Default / alternativas | Onde | O que faz | Conceitos | Por que importa |
|---|---|---|---|---|---|
| `SPEC_METHOD` | `dflash` (`.env.example:133`); `mtp`; `none` | `start.sh` (monta `--speculative-config`) | Escolhe o método | Speculative decoding | Rollback `SPEC_METHOD=mtp` (k=2) |
| `DFLASH_MODEL` / `DFLASH_REVISION` | `incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c…` | `.env.example:134-137` | Drafter BF16 (~2,3 GiB, CC BY-NC-ND 4.0) | Drafter treinado | — |
| `DFLASH_TOKENS` | `7` (bloco treinado 8); TP4 `3` | `.env.example:138` | `num_speculative_tokens` | Tamanho do bloco especulativo | TP4: 3 tokens = 53% aceito vs 30% com 7 |
| `DFLASH_DRAFT_TP` | `2` (TP2); `1` (TP3, divisibilidade) | `.env.example:141` | `draft_tensor_parallel_size` | TP do drafter | TP=1 no TP3 = "imposto de divisibilidade" |
| `--speculative-config` montado | `{"method":"dflash","model":…,"num_speculative_tokens":7,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":2}` | `start.sh` (bloco `SPEC_METHOD=dflash`) | Config do vLLM | Rejection sampling padrão, drafts probabilísticos | KV do drafter `auto`/BF16 |
| KV do drafter | `auto`/bf16 forçado quando o alvo usa `fp8_ds_mla`/`fp8*`/`nvfp4` | `patch_dflash2.py:98-155` | O drafter denso não pode usar o layout MLA do alvo; SM121 não tem FA3/FA4 para FP8 puro | Layout de KV por grupo | 2048 B/token/camada em TP2 |
| Backend de atenção do drafter | não fixar; SM121 escolhe `FLASH_ATTN` (não-causal) | README; `patch_dflash2.py:36-47` (`is_causal: false`) | Atenção bidirecional dentro do bloco | Atenção não-causal em bloco (diffusion-style) | `TRITON_ATTN` é causal dentro do bloco → aceitação 0,31 / ~29 tok/s |
| `MTP_TOKENS` | `2` | `.env.example:142` | k do MTP nativo | Multi-token prediction | Baseline ~24,6 tok/s prosa |
| `GLM53_ADAPTIVE_K` | `off`; `ema`/`on`/`1` | `patch_adaptive_k.py:63`, `302` | Verifica só um prefixo (2/4/7) dos 7 drafts, escolhido por EMA de aceitação, uniforme no batch | Verificação adaptativa, uniform-batch FULL CUDA graph | Prosa +13–21% (2026-09-08); sem perda a temperatura 0 |
| `GLM53_ADAPTIVE_K_SET` | `2,4,7` | `:68`, `305` | Comprimentos candidatos; graphs capturados para k+1 (`[3,5,8]`) | CUDA graph por query-len | Captura precisa incluir múltiplos de 3, 5 e 8 |
| `GLM53_ADAPTIVE_K_ALPHA` | `0.25` | `:65` | Peso da EMA | Média móvel exponencial | — |
| `GLM53_ADAPTIVE_K_MARGIN` | `1.0` | `:66` | `n = maior valor do set ≤ ceil(ema + margin)` | Margem otimista | — |
| `GLM53_ADAPTIVE_K_MIN_STEPS` | `4` | `:67` | Passos observados no comprimento cheio antes de cortar | Aquecimento | — |
| `GLM53_ADAPTIVE_K_SATURATE` | `max`; `n` | `:70`, `126-138` | Se todos aceitos, alimenta a EMA com o máximo (pode subir) ou com n (só desce) | Anti-ratchet | — |
| `GLM53_ADAPTIVE_K_HIST` | `200` | `:71` | Log do histograma a cada N passos | — | — |
| `GLM53_ADAPTIVE_K_FILE` | `/root/.cache/vllm/glm53_adaptive_k.json` | `:81`, `91-124` | Override em runtime (relido a cada 50 passos se mtime mudar) | Retuning sem reboot | `set` é recortado ao set do boot (graphs só existem para ele) |
| `--cudagraph-capture-sizes` (auto) | stock `1 2 4 8 16 24 32`; MTP `1 2 3 4 6 8 12`; com adaptive-k o launcher gera múltiplos de k+1 até `MAX_NUM_SEQS` | `start.sh:320-342` | Tamanhos de captura | CUDA graphs | Um valor explícito em `EXTRA_ARGS` sempre vence |

### 2.5 KV cache, grupos, APC, retenção, indexer

| Nome | Default / alternativas | Onde | O que faz | Conceitos | Por que importa |
|---|---|---|---|---|---|
| `--kv-cache-dtype` / `KV_CACHE_DTYPE` | `fp8` → packed `fp8_ds_mla` | `.env.example:210`; `Dockerfile` | Layout empacotado de 656 B/token/camada MLA: 512 fp8 NoPE + 16 B de escalas + 128 B de RoPE (aqui zeros) | KV FP8, MLA latent cache | O único kernel sparse-MLA do SM12x só aceita `fp8_ds_mla`; BF16 e NVFP4 não têm kernel esparso |
| `--kv-cache-memory-bytes` (em `EXTRA_ARGS`) | TP2: `11811160064` (11 GiB) desde 2026-09-26; antes 14 GiB (`15032385536`); TP3: 40 GiB (`42949672960`) no `.env.tp3` do kit do README, 35 GiB no cluster medido em 2026-09-26, 32 GiB (`34359738368`) em `examples/tp3-throughput.env` | `.env.example:191` | Reserva explícita do pool (ignora `GPU_MEM_UTIL` para o KV) | Orçamento de KV em UMA | Com InstantTensor o profiling deixa ~12,4 GiB < 13,56 GiB necessários para 1 req de 850k |
| `MAX_MODEL_LEN` | `850000` (TP2), `1000000` (TP3) | `.env.example:196` | Contexto máximo | — | Não reduzir para "liberar" KV: a demanda de ids do híbrido é quase independente do comprimento |
| `GPU_MEM_UTIL` | `0.85` (TP2), `0.80` (TP3), `0.75` (TP4) | `.env.example:209` | Fração da UMA para o vLLM | UMA | Cada 0,01 = 1,2 GiB de folga do host |
| `--enable-prefix-caching` | ligado | `start.sh` | APC | Automatic prefix caching | Hits alinhados a 3584 tokens (manager block MLA/Mamba) |
| `PREFIX_MATCH_UNIT` | vazio (vLLM resolve); `64` suportado; `512` inválido | README tabela `.env` | Grão do hash de prefixo | Hash de blocos | Menor grão não cria checkpoints Mamba ausentes |
| `GLM53_DRAFT_KV_COMPACT` | `1` quando `SPEC_METHOD=dflash` (unset); `0` caso contrário; vazio explícito é rejeitado | `patch_glm5_drafter_group.py:684-711`, `714-738`; `patch_hybrid_prefix_hit.py:369-381` | Bloco do drafter = maior múltiplo de 64 que divide o bloco MLA e cabe na página (896 com bloco 3584) + lookup no limite exato | Paginação do KV, slot-sharing | Demanda de ids do drafter 65 → 6 por request (janela 2048 + 2048 in-flight); repeat TTFT −90% no caso de cauda de 64 tokens |
| `GLM53_APC_RETENTION_INTERVAL` → `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | vazio (grade densa) | `start.sh:2477` | Intervalo global de retenção de blocos hashados | Retenção esparsa | Mais capacidade, grão de hit mais grosso |
| `GLM53_APC_RETENTION_INTERVAL_SWA` → `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA` | vazio = herda o global; `0` = só limites alcançáveis + prioridade baixa de eviction do drafter; N múltiplo de 3584 ≤ 1.000.000 | `start.sh:2483`; `patch_apc_per_group_retention.py:124` | Retenção por grupo para o grupo SWA do drafter | Retenção por grupo | `0/0`: 4 históricos de 210k reutilizados em ~2,5 s, mas edição a 90% de 128k: 14,70 s → 112,49 s |
| `GLM53_APC_NO_STORE` | `1`; `0` ignora o flag do cliente | `patch_apc_no_store.py:131-155` | Kill switch do no-store por request | Escrita no prefix cache | Só `0`/`1`; outros valores falham no boot |
| `skip_writing_prefix_cache` (por request) | via `vllm_xargs` `1`/`"1"`/`true` | `patch_apc_no_store.py:111-210` | Request lê o cache mas não insere hashes; blocos voltam para a frente da fila livre (LIFO) | LRU/LIFO do BlockPool | Batch/eval não desloca a conversa residente |
| `GLM53_KV_CAPACITY_LOG` | `1`; `0` | `patch_kv_capacity_log.py:118-133` | Log por grupo + resumo de ids por segmento | Contabilidade de blocos | Só log |
| `GLM53_INDEXER_WORKSPACE` | `.env.example:333` = `rightsize`; unset = `stock` | `patch_indexer_workspace.py:120-138` | Workspace de gather do indexer: `min(max_num_seqs, MNBT) × cdiv(max_model_len + k, compress_ratio)` entradas de 132 B | Workspace fixo subtraído do KV | 5036,40 MiB (stock, 1M) → 125,9 MiB (4 seqs) ≈ +26–28% KV |
| `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS` | `0`; `64` (TP4) | `patch_sparse_mla_slice.py:60-80` | Fatia a chamada final do sparse-MLA em ≤64 linhas de query | Mitigação de stall | Só TP4; não corrige a race |
| `LIMIT_MM`, `MM_IMAGE_TOKENS`, `MM_PROCESSOR_CACHE_GB`, `SKIP_MM_PROFILING` | `{"image":48,"video":1}`, `2048`, `1`, `1` | `.env.example:212-232` | Limites multimodais (nada reservado para a torre de visão) | Encoder cache | Vídeo de 11,9 MB virou 236.544 tokens e matou o worker (2026-09-14) |

### 2.6 Scheduler (mixed prefill) — `overlay/patch_scheduler_decode_floor.py`

| Nome | Default | Onde | O que faz | Conceitos | Trade-off |
|---|---|---|---|---|---|
| `GLM53_MIXED_PREFILL_CHUNK` | código `skip`; launchers/.env `fair` (TP2/TP3/TP4 desde 2026-09-15); `N>0` cap; `0`/`off` | `patch_scheduler_decode_floor.py:1-47`, `297-313` | Política para prefill enquanto há decode | Chunked prefill, mixed batches | `skip` = inanição (221 s / 20 min); `0` = incumbente cai 10–36× |
| `GLM53_FAIR_PREFILL_CHUNK` | `256` | `:317` | Chunk de sonda até existirem amostras de tempo | — | Depois o v5 escolhe o maior degrau do ladder que cabe |
| `GLM53_FAIR_PREFILL_SHARE` | `0.30` (código e .env) | `:323` | Fração do tempo do engine que vira crédito para prefill | Service-time fairness | 0,30 vs 0,20: newcomer 30k espera ~145 s vs ~220 s; incumbente 1,3–1,4× vs ~1,16× mais lento |
| `GLM53_FAIR_PREFILL_MAX_INTERVAL_MS` | `2000` | `:328` | Idade após a qual um prefill já servido pode pegar 1 chunk emprestado (após quitar a dívida) | Aging | Não encurtar primeiro |
| `GLM53_FAIR_PREFILL_MAX_STEP_MS` | `2000` | `:339` | Limite estimado de duração de um passo misto | Orçamento de latência | 1000 ms travava o fitter; 2000 permite 2048 |
| `GLM53_FAIR_PREFILL_MAX_CHUNKS` | `1` (máx 16) | `:334` | Prefills distintos por turno | — | Não multiplica o chunk de um request |
| `LADDER` | `(128,256,512,768,1024,1536,2048)` | `:258` | Degraus de chunk | — | — |
| `COLD_TOK_S` / `FIT_WINDOW` | `1300.0` / `24` | `:259-260` | Estimativa inicial e janela do ajuste `dt = a + b·n` | Regressão linear de custo | Custo fixo ~0,3 s por passo com prefill neste kit |
| `LONG_PREFILL_TOKEN_THRESHOLD` | vazio (TP2); TP3 profile 2048 | `.env.example:208` | Cap global por request (inclusive solo) | — | Não é o conserto de fairness |
| `MAX_NUM_SEQS` | `4` (TP2), `8` (TP4) | `.env.example:197` | Gerações em voo | — | — |
| `MAX_NUM_BATCHED_TOKENS` (MNBT) | `7168` (default com E3); histórico 2048 (vencedor do ladder P1 pré-E2); 8192 proibido (smem do indexer) | `.env.example:202` | Budget de tokens por passo | Chunked prefill | Chunks maiores amortizam custo fixo mas pioram interleave |

### 2.7 Carga de pesos e memória

| Nome | Default | Onde | O que faz | Conceitos | Trade-off |
|---|---|---|---|---|---|
| `LOAD_FORMAT` | `instanttensor` (imagem `:exl3-instanttensor`); vazio = vLLM auto | `.env.example:370` | Loader direct-I/O | O_DIRECT, ring buffer pinned | ~65–70 s frio vs ~290–300 s auto; deixa ~4,4–5,6 GiB a menos de KV |
| `GLM53_COLD_LOAD_UMA` | `1` (kill switch `0`) | `patch_cold_load_uma.py:156` | Dimensiona o buffer do InstantTensor contra `MemAvailable` (não `MemFree`) | UMA, page cache como memória "usada" | Boota com page cache cheio (164 GiB em 36 s @ 4,82 GB/s) |
| `GLM53_COLD_LOAD_STAGE_MMAP` | `1` | `patch_loadclone.py:67`, cold_load | `clone()` de tensores de mmap file-backed quando página ≠ 4 KiB | Kernel 64 KiB, `cuMemcpyHtoDAsync` travando | Necessário no kernel `6.17.13-rocket64k` |
| `INSTANTTENSOR_MAX_FREE_MEM_USAGE` / `INSTANTTENSOR_BUFFER_SIZE` | unset | `patch_cold_load_uma.py:168-231` | Overrides; um `MAX_FREE_MEM_USAGE` isolado insuficiente é elevado (#273), nunca rebaixado | — | Par explícito é usado como dado |
| `GLM53_HOST_MEM_HYGIENE` | `1` | `start.sh`, README | Drop de page cache no host antes do `docker run` e espera `MemAvailable` | — | Container não consegue `drop_caches` |
| `GLM53_LOAD_CLONE` | `1` | `patch_loadclone.py:54` | Staging de tensores auto/lazy mmap em CPU antes de entregar | — | Não desliga a segurança 64 KiB |
| `GLM53_LOAD_PREFETCH` | `0` (0..16) | `patch_loadclone.py:55` | Read-ahead de shards locais + `POSIX_FADV_DONTNEED` em shards consumidos | Page cache advice | Não é limite de RAM |
| `CG_ESTIMATE` → `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `.env.example:359` = `1` | `patch_skip_cudagraph_profile.py:6-59` | Com `0`, pula a dry-capture de memória de graphs (11 s de boot) | Profiling de memória | — |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | README tabela | Alocador | Fragmentação | — |

### 2.8 Patches de API/serving

| Nome | Default | Onde | O que faz | Conceito | Por que |
|---|---|---|---|---|---|
| `GLM53_SUPPRESS_STOPS_IN_REASONING` / `VLLM_SUPPRESS_STOPS_IN_REASONING` | `1` | `patch_suppress_stops_in_reasoning.py:80-81` | `stop` do cliente dormente até `</think>` | Detokenizer, reasoning | CoT repete stops ("Question:") e encerrava com `content` vazio |
| `tool_choice:"none"` (sem env) | sempre aplicado | `patch_tool_choice_none.py` | Adiciona `TOOL_CALL_START` a `bad_words` | Mascaramento de logits | Mantém o bloco de ferramentas no prompt (prefixo compartilhado) |
| `GLM53_EXPOSE_CACHE_RESET` | `0` | `patch_cache_reset.py:54` | Monta só o router de reset de cache | Dev routes do vLLM | Rotas root fora do bearer guard |
| `DEFAULT_MAX_NEW_TOKENS` | `65536` (1..1000000); vazio = stock | `patch_default_max_new_tokens.py:42-119` | Default só para requests que omitem `max_tokens` | Pydantic before-validator | Explícito `null` mantém 16 |
| `GLM53_SPINWAIT_MS` | `stock` (1000 ms); 1..1000 | `patch_spinwait.py:23-54` | Janela de busy-loop do leitor `SpinCondition` | IPC por shared memory | 16 ms: +0,95% decode, −85,3% CPU do EngineCore |
| xgrammar | sempre | `patch_xgrammar_termination.py` | Backports vLLM #52805 e #53046 | FSM de gramática + spec decode | Evita `Failed to advance FSM` |
| `GLM53_DEFAULT_REASONING_EFFORT` | vazio (template → Max); `low/high/max` | README | `--default-chat-template-kwargs` | Chat template | `medium` rejeitado |

### 2.9 TP=3/TP=4 e FlashKDA

| Nome | Default | Onde | O que faz | Conceito | Por que |
|---|---|---|---|---|---|
| `TP3_HEAD_OVERRIDE` | `66` | `start-tp3.sh:421`, `1629-1631` | `--hf-overrides` com 66 cabeças (3×22) | Divisibilidade TP | 64 não divide por 3; 96 (local 32) faz um rank só com cabeças KDA dummy → logits colapsam |
| `ENABLE_EXPERT_PARALLEL` | `1` | `start-tp3.sh:424` | `--enable-expert-parallel` (96 de 288 experts por rank) | Expert parallelism | Trellis empacotado 2048 não divide por 3 |
| `MM_ENCODER_TP_MODE` | `data` | `start-tp3.sh:426` | Torre de visão em data-parallel | — | Divisibilidade |
| `HAREM_KDA_FLASHKDA` | vazio/`0` = Triton; `1` = FlashKDA | `patch_flashkda_tp3.py:63-100` | Kernel de prefill KDA fundido | Chunked linear attention | Exige SM90/10x/12x, bf16, head_dim 128, gate limitado |
| `SOCKET_IFNAME`, `NCCL_CROSS_NIC`, `NCCL_IB_SUBNET_AWARE_ROUTING` | TP3: `enP7s7`, `1`, `1` | README "3x Spark" | Bootstrap em LAN de gerência + ring dirigido RoCE | NCCL multi-NIC | NCCL pareia índice de NIC por canal |

### 2.10 Abliteração (`overlay/ablit_runtime.py`)

| Nome | Default | Onde | O que faz | Conceito | Por que |
|---|---|---|---|---|---|
| `ABLIT` | `0` | `ablit_runtime.py:410` | Liga a edição de `o_proj` na carga | Abliteração | No-op quando off |
| `ABLIT_METHOD` | `auto` (transplant se `ablit/transplant/` existir, senão `proj`); `transplant`; `proj` | `:416-453` | Método | Transplante de pesos vs ortogonalização | Direções publicadas são estatisticamente aleatórias contra o `o_proj` stock (max \|rᵀW\| 0,084 = aleatório) |
| `ABLIT_LAYERS` | `15-45` | `:414`, `116-139` | Faixa inclusiva; 45 = MTP | Safety anchors (0–14 stock) | — |
| `ABLIT_ALPHA` | `3.0` | `:458-462` | `W' = (I − α r rᵀ/‖r‖²) W` | Projeção ortogonal | α>1 sobre-projeta |
| `ABLIT_DIRECTION` | `dealign`; `bf_oproj`; caminho `.pt` | `:95-98`, `456` | Direção de recusa (vetor 4096 fp32) | Direção de recusa | — |
| `ABLIT_INCLUDE_MTP` | `1` | `:415` | Edita também o `o_proj` do bloco MTP | MTP | Um MTP stock continuava propondo recusas |
| `ABLIT_DIR` | `/opt/glm53/ablit` | `:92`, `413` | Diretório de artefatos | — | — |
| `ABLIT_DONOR` (fetch) | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | `ablit/fetch_transplant.py:34` | Doador dos tensores | HTTP range requests | ~2,7 GiB em vez de 164 GiB |

### 2.11 Ferramentas (`tools/`)

| Nome | Default | Onde | O que faz |
|---|---|---|---|
| `BITS` | `6.0` | `tools/dflash2_exl3_quant.sh` | bpw do drafter EXL3 |
| `DRAFT_QUANT_KERNEL_PROJ` | `1` | idem / `dflash2_exl3_convert.py` | Quantiza `kernel_projection` da conv dinâmica |
| `DRAFT_QUANT_FC` | `1` | idem | Quantiza `fc` (projeção de entrada dos hidden do alvo) |
| `PIN` do conversor | `MiaAI-Lab/exllamav3@63b32f0` (v1.4.2) | idem | Fork do conversor |
| `IMG`, `DRAFT_SNAP`, `BUILD`, `OUT`, `TORCH_CUDA_ARCH_LIST=12.1` | ver script | idem | Ambiente de conversão |
| `dense_overlay.py --branch 4.05bpw --prefix-rewrite … --lm-head` | — | `tools/dense_overlay.py` | Range-read dos linears densos de `turboderp/GLM-5.3-Flash-exl3` (commit `2a30229e`, ~5,3 GB) e escreve 1 safetensors + index + config com `non_routed_exl3` |

### 2.12 Constantes e geometrias "escondidas" que qualquer um que recrie a receita precisa saber

| Constante | Valor | Onde | Significado |
|---|---|---|---|
| Página MLA (bloco do manager) | 3584 tokens × 656 B = **2.351.104 B** por camada | README ("padded slot-share … mla_page=2351104"), `docs/DESIGN-apc-per-group-retention.md` §1 | Bloco "mamba-aligned": o vLLM híbrido escolhe um block_size da atenção cujo page cabe o estado KDA; o kernel MLA usa página de 64 |
| Bloco de kernel sparse-MLA | 64 | `Dockerfile` (backend `[64]`) | Todos os kernels GLM_NSA/DSV3_2 são instanciados com `PAGE_BLOCK_SIZE=64` |
| KpoolTail | block_size 4 (= `index_kpool`), 1 bloco por request, circular | `DESIGN-kv-capacity-log.md` §2 | Scratch da cauda de tokens ainda não comprimidos do indexer |
| Página do indexer (tail stride) | 118.272 B (`stride(0)=59136` bf16) no bloco 3584; 38.016 B no bloco 1152 (upstream) | `patch_kpool_tail_seed_stride.py:1-40` | O tail é `as_strided` da página do indexer |
| Drafter SWA | window 2048, block 64 (ou 896 compacto), 5 camadas, 2048 B/token/camada em TP2 | README, `patch_glm5_drafter_group.py` | Grupo 6 do KV |
| Custo em ids de um segmento de 3584 tokens (retenção densa) | 38 = 1 MLA + 4 Mamba + 33 drafter | `patch_kv_capacity_log.py`, `DESIGN-apc-per-group-retention.md` §2.5 | 87% dos ids vão para o drafter (que não contribui para o hit) |
| Blocos por request de 1M | 414 = 280 MLA + 1 tail + 4×9 Mamba + 97 drafter | `DESIGN-kv-capacity-log.md` §2 | Denominador do "GPU KV cache size" |
| Workspace do indexer | 132 B por entrada (128 fp8 + 4 de escala); stock `max_model_len × 40` | `patch_indexer_workspace.py` docstring | Subtraído do KV no profiling |
| top-k do indexer | `select_k=512` pools × 4 = 2048; buffer arredondado a múltiplo de 128 (2048 → 2176 com cauda) | `Dockerfile` (comentários 38–85) | Por que o overlay corta 1 pool |
| Taps EAGLE3 do alvo | `(6, 15, 25, 34, 43)` | README "Why the overlay exists" | Camadas cujo hidden (contraído do mHC) alimenta o drafter |
| DFlash2 GQA | 32 Q / 8 KV cabeças (TP3: 36/9) | `overlay/tp3/pad-tp3-config.py:69-84` | FlashInfer exige qo % kv == 0 |
| Codebooks EXL3 | cb0 = 3INST (`×89226354 + 64248484`), cb1 = MCG (`×0xCBAC1FED`), cb2 = mul1 (`×0x83DCD12D`, soma de bytes via `dp4a`) | `extensions/cooperative_moe/tp3/vendor/exllamav3_ext/quant/codebook.cuh` | Decodificação procedural de 16 bits → fp16 |
| Hadamard | blocos de 128, escala `1/sqrt(128) = 0.088388347648` | `exl3_fat_moe.cu:31`, `hadamard_inner.cuh` | Rotação de incoerência |

---

## 3. Inventário de modificações

Para cada item: **nome** | arquivo(s) | problema que resolve | o que muda tecnicamente | conceito(s) | efeito medido.

### 3.A Patches inline do `Dockerfile` (backend de atenção e registro EXL3)

**M1 — NoPE MLA "embrulhado" na geometria GLM_NSA 576 do FlashInfer SM120**
- Arquivos: `Dockerfile:1-86` (explicação), `Dockerfile:91-318` (patch em `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`), verificação `Dockerfile:320-368`.
- Problema: GLM-5.3-Flash é **NoPE MLA** (`qk_rope_head_dim=0`, `d_qk=512`). No SM12x o único backend sparse-MLA é `FLASHINFER_MLA_SPARSE_SM120`, que exige o registro empacotado `fp8_ds_mla` de 656 B (512 NoPE + 16 B de escalas + 128 B de RoPE) e query de 576. Resultado: `concat_and_cache_mla` → "pe_dim must be 64 for fp8_ds_mla" e `flashinfer/mla/_core.py:583` rejeita. O FlashInfer só instancia DSV3_2/GLM_NSA (d_qk=576, 656 B, topk ∈ {128,512,1024,2048}) e DSV4 (512 = 448 fp8 + 64 bf16, topk ≤1024) — nenhuma serve d_qk=512 com topk 2048.
- Mecanismo: `self.rope_pad = 64` quando `qk_rope_head_dim == 0` (exige `kv_lora_rank == 512`); `q = F.pad(q, (0, 64))` antes do kernel; `qk_rope_head_dim=self.kernel_qk_rope_head_dim` (64); `do_kv_cache_update` grava `k_pe` = zeros de 64. Como o bloco RoPE é zero, o produto QK não muda, o softmax não muda e o valor vem dos 512 NoPE (`d_v=512`) → resultado exato. Declara `supports_dense_mha_prefill = False` (o `forward_mha` herdado levanta `NotImplementedError`), roteando o prefill pelo mesmo kernel MQA top-k (a semântica esparsa do próprio modelo).
- Top-k como capacidade: o impl passava `index_topk` como largura da tabela e como top-k; com `index_kpool=4` o buffer do indexer vira 2176. O patch pede ao conversor `triton_convert_req_index_to_global_index(..., return_valid_counts=True)`, passa `seq_lens=topk_lengths`, `max_seq_len=sparse_mla_top_k=sparse_topk_capacity` (largura do buffer), aponta linhas vazias para o slot 0 com comprimento 1 e zera a saída delas (`out.masked_fill_(empty_rows…)`).
- Custo: 656 B/token em vez de ~528 (~24% a mais de KV DSA em 11 camadas); nenhum kernel novo.
- Conceitos: MLA (latent KV, absorção de pesos), RoPE vs NoPE, sparse attention com indexer, FP8 KV empacotado com escalas por tile, padding exato.

**M2 — Top-k exatamente 2048: descartar o pool de menor rank, preservar a cauda**
- Arquivos: `Dockerfile` (patch em `models/glm5next/nvidia/model.py`, `mtp.py` e `model_executor/layers/sparse_attn_indexer_kpool.py`).
- Problema: a largura de top-k é parâmetro de template nos kernels SM120 (`sparse_mla_sm120_prefill.cu:220` rejeita topk ≠ 2048 para GLM_NSA). O indexer emite até 2051 candidatos: 512 pools × 4 + cauda de até `kpool−1 = 3` tokens sem pool comprimido.
- Mecanismo: `buffer_width = topk_tokens` (remove `+ (kpool-1)`), e `expand_pools_and_append_tail(pool_ids[:, : select_k - 1], …)` nos caminhos de prefill e decode: 511 pools (2044 tokens) + 3 de cauda = 2047 ≤ 2048. Abre mão de 4/2048 candidatos (0,2%), os de menor rank, e mantém a cauda recente (maior peso de atenção num modelo causal e impossível de recuperar pelos pools). `select_k` continua 512, mantendo o fast-path fundido de top-k (`sparse_attn_indexer_kpool.py:810`).
- Conceitos: DSA/NSA (compressão em pools, lightning indexer, top-k), trade-off de recall.

**M3 — Alinhamento do bloco do indexer e página de kernel 64**
- Problema: DeepGEMM paged-MQA aceita `block_kv` ∈ {32,64,128} no sm_10x mas só 64 no sm_120 (cache não-FP4); `vllm/utils/deep_gemm.py` hardcoda `(32, 64)`. O bloco de armazenamento do indexer é `block_size // index_kpool`; com alinhamento `index_kpool × 32 = 128` o bloco híbrido caía em 1664 → 416 entradas (não múltiplo de 64) → página 32 → assert do DeepGEMM.
- Mecanismo: `platforms/cuda.py` usa só páginas 64 quando `capability.major == 12` (`index_kpool × 64 = 256` → bloco 1792, 448 entradas, divisível por 64). E `FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes()` passa de `[64, 256]` para `[64]`, para o seletor não escolher 256 num manager de 1792.
- Conceitos: paged attention, block size de manager vs kernel, restrições de arquitetura.

**M4 — Desligar autotune do FlashInfer e PDL no SM121**
- Mecanismo: remove `flashinfer_sparse_mla_decode_autotune_warmup(worker)` do `kernel_warmup.py` e faz o autotuner retornar cedo ("Skipping FlashInfer autotune on SM121"); `--no-enable-flashinfer-autotune` no launcher; PDL (Programmatic Dependent Launch) restrito a `major in (9, 10)` porque "PDL lowering races KDA state kernels on SM12x".
- Problema: autotune do sparse-MLA e do fused_moe gemm1/gemm2 mata o rank 0 no SM121; autotune completo dá OOM na UMA.
- Conceitos: autotuning de kernels, PDL/launch overlap, estabilidade.

**M5 — Registro do método `exl3` no vLLM**
- Arquivos: `Dockerfile:387-419` (`QuantizationMethods` Literal + import lazy de `Exl3Config` em `get_quantization_config`), `overlay/patch_model_overrides.py` (insere `"exl3"` no topo da lista ordenada de overrides do `ModelConfig`, para `override_quantization_method` ganhar de `auto_gptq` etc.), `overlay/exl3.py:1705` (`@register_quantization_config("exl3")`).
- Conceitos: plugin de quantização do vLLM, resolução de método a partir de `quantization_config` do HF.

**M6 — Compilar o `exllamav3_ext` em aarch64/GB10**
- Arquivo: `overlay/patch_exl3_ext_aarch64.py`.
- Problema: o ExLlamaV3 traz alvos de CPU AVX2/AVX512 (allreduce de CPU) que não compilam em ARM.
- Mecanismo: reescreve `avx2_target.cpp`, `avx512_target.cpp`, `parallel/all_reduce_cpu_avx*.cpp` com stubs (`is_avx2_supported() = false`, `perform_cpu_reduce…` = `std::abort()`).
- Build: `Dockerfile:421-437` baixa `turboderp-org/exllamav3@c5d9c657`, aplica M6, M10/M11 (fat kernels) e M12 (FAST), compila com `TORCH_CUDA_ARCH_LIST=12.1a`, headers CUDA em `nvidia/cu13/include`, e verifica os símbolos `exl3_moe`, `exl3_fat_gemm`, `exl3_fat_gemm_scatter`, `exl3_fat_moe_{gather,gateup,down}`, `glm53_fast_moe_version()==1`.

### 3.B Quantização EXL3 no vLLM (`overlay/exl3.py`)

**M7 — `Exl3Config` + `Exl3MoEMethod`: experts permanecem empacotados (trellis) na GPU**
- Arquivo: `overlay/exl3.py:1705-1892` (config), `2973-3248` (método MoE).
- Problema: registrar o nome não basta; se os experts fossem expandidos para BF16 não caberiam (288 experts × 45 camadas). É preciso carregar `trellis/suh/svh/mcg`, fatiar por TP e executar a GEMM Trellis/MCG.
- Mecanismo:
  - `get_quant_method`: `RoutedExperts` → `Exl3MoEMethod`; `ParallelLMHead` → EXL3 só se declarado; `LinearBase` → `Exl3LinearMethod` (pack denso) | `Glm53DenseFp8Method` (grupo FP8) | `UnquantizedLinearMethod`.
  - `create_weights` (`exl3.py:2985-3070`): `w13_trellis [E, 2, K/16, I/16, 16·bits]` int16 (gate=0, up=1), `w13_suh [E,2,H]`, `w13_svh [E,2,I_local]`, `w13_mcg`, `w2_trellis [E, I/16, H/16, 16·bits]`, `w2_suh [E, I_local]`, `w2_svh [E, H]`. Nomes compatíveis com o `expert_params_mapping` padrão (`experts.w13_*`). Proíbe alocar `w13_weight` denso.
  - `_load_exl3` (`exl3.py:3072-3120`): mapeia expert global→local (`_map_global_expert_id_to_local_expert_id`, retorna sem carregar se não é local), TP shard: gate/up **column-parallel** (`trellis` dim 1, `svh` dim 0; `suh` replicado), down **row-parallel** (`trellis` dim 0, `suh` dim 0; `svh` replicado) (`shard_exl3_col/row`, `exl3.py:661-676`).
  - `process_weights_after_loading` (`exl3.py:3122-3233`): valida o marcador MCG em todos os experts; detecta `_exl3_shared_w13_suh` (gate e up com o mesmo `suh` → uma só Hadamard de entrada); cria um `LinearEXL3` por (expert, gate/up/down) sem reconstrução BF16; resolve o tier "fat" (M8); constrói as tabelas de ponteiros int64 (`gate_trellis`, `gate_suh`, …, `down_svh`) e os 4 buffers temporários `(concurrency, rows, H/I)` fp16 do `exl3_moe` (`build_exl3_fused_state`, `exl3.py:1279-1352`; `concurrency = exl3_moe_max_concurrency(device)`).
  - `apply` → `apply_exl3_experts` (`exl3.py:1653-1688`): `expert_map` fixado no device (cópia CPU→GPU é proibida dentro de captura de graph, `pin_exl3_expert_map`), `map_topk_to_local` (ids inválidos/não locais → sentinela `n_local`), e a saída do MoE runner faz all-reduce uma vez por camada.
- Conceitos: quantização weight-only, TP column/row parallel, expert parallel mapping, CUDA graphs, pointer tables.
- Efeito: é o que permite servir 164 GiB de pesos em 2×128 GB ("Model loading took 79.65 GiB" por rank).

**M8 — MoE fundido e a hierarquia de tiers para "fat experts" (legacy → sorted → E1 → E2 → E3)**
- Arquivo: `overlay/exl3.py:1460-1650` (`apply_exl3_fused_moe`), `362-423` (`resolve_exl3_fat_tier`), `287-333` (`grouped_fat_eligibility`).
- Problema (medido, `docs/improve-prefill.md` §8 P0): num chunk de prefill de 1024 tokens, top-8 → 8192 slots sobre 288 experts; o expert mais quente aparece em quase todo token (max_rows=1024, avg_max≈920); 91–98,6% das camadas MoE tinham experts acima do cap de 128 linhas do `exl3_moe`, que caíam num loop Python com `LinearEXL3` (reconstrução BF16 por expert) e `.nonzero().tolist()` = host sync por camada (1,45 s de CPU em `aten::nonzero` num capture de 3 passos).
- Mecanismo comum: ordena os pares (token, slot) por expert local (`local.argsort()`), `expert_count` via `scatter_add_` (não `bincount`, que pode fazer staging no host e quebrar graphs), `num_active=-1` (grid de concorrência máxima, sem `.item()`). Se `tokens ≤ cap` → **um único lançamento** `exl3_moe` (caminho do decode, graph-safe). Senão (prefill):
  - **legacy**: `exl3_moe` roda os thin (o kernel pula experts com count > cap) e cada fat vai para `apply_exl3_python_loop` (sync + reconstrução).
  - **sorted** (`EXL3_FAT_SORTED`): um `counts.tolist()` e fatias contíguas do sort (`apply_exl3_sorted_fat`).
  - **E1 batched** (`EXL3_FAT_BATCHED`): cópia de `counts` para pinned host num stream lateral enquanto os thin rodam (`_stage_counts_to_host`), scratch persistente (`_fat_scratch`), `had_r_128` de entrada uma vez (gate.suh compartilhado), trellis de gate e up concatenados, `reconstruct` + `hgemm` + `had_r_128` de saída.
  - **E2 kernel** (`EXL3_FAT_KERNEL`): em vez de reconstruir, `exl3_fat_gemm` (GEMM direto sobre o trellis com Hadamard+svh no epílogo) e `exl3_fat_gemm_scatter` (down + peso de roteamento + scatter-add em `out`) — M9.
  - **E3 grouped** (`EXL3_FAT_GROUPED`): thin no `exl3_moe` + todos os fat em 3 lançamentos (M10), sem host sync, capturável.
  - A ordem de resolução: checkpoint sem SUH compartilhado limita a `sorted`; E3 pedido sem símbolos → erro na carga (fail-closed); checkpoint/dispositivo inelegível para E3 (não K4, não MCG, mul1, `hidden % 256`, `intermediate % 128`, capability < 9.0) → cai para E2 com o motivo registrado.
- Diagnóstico: linha `exl3 e2 diag schema=2 configured_tier=… effective_tier=… grouped_calls=…` e histograma `exl3 fat-expert P0`.
- Efeito: E2 +20,33% (8k), +21,36% (100k), +20,70% (300k) (PR77, 45 observações); E3 vs E2 +37% a +45% (16k–256k, 2026-09-07); camada MoE isolada de 7168 tokens 77–91 ms (E2) → 31 ms (E3).
- Conceitos: routing top-k, desbalanceamento de carga entre experts, host-device sync, grouped GEMM, CUDA graphs.

**M9 — Kernel E2 `exl3_fat_gemm` / `exl3_fat_gemm_scatter`**
- Arquivos: `overlay/exl3_fat_gemm.cu` (295 linhas), `.cuh`; instalado por `overlay/patch_exl3_fat_kernel.py` (copia `.cu/.cuh` para `quant/` e adiciona `m.def(...)` no `bindings.cpp`).
- Mecanismo: CTA de 256 threads (8 warps), tile M=128 × N=128, K avançando de 16 em 16; A (ativação fp16 já rotacionada) em SMEM com swizzle XOR; B = 8 tiles 16×16 K4 (64 palavras int16 cada) copiados para SMEM e **dequantizados em registradores** por `dq_dispatch<4,1>` (MCG); `ptx_mma_m16n8k16` (tensor core, fp16→fp32). Epílogo: Hadamard de 128 por linha (`fat_had_ff_128`: butterfly de 4 + `shuffle_had_f2x32` entre lanes) × `svh`. Na versão scatter, multiplica pelo peso de roteamento e faz `out[token] += …` (sem atomics: cada token chega a um expert uma vez e os lançamentos são serializados no stream). Restrito a `K == 4 && mcg && !mul1`.
- Conceitos: dequantização "on the fly" no tensor core, MMA m16n8k16, swizzle de SMEM, Hadamard rápido via shuffles.

**M10 — Kernels E3 agrupados `exl3_fat_moe_{gather,gateup,down}`**
- Arquivos: `overlay/exl3_fat_moe.cu` (656 linhas), `.cuh`; tabelas em `exl3.py:1144-1263` (`build_grouped_fat_tables`, `apply_exl3_grouped_fat`); build aditivo `overlay/build_exl3_fat_moe_ext.py` + `Dockerfile.e3-layer` (módulo `exl3_fat_moe_ext` compilado **sem** `--use_fast_math`).
- Tabelas no device (sem sync): `fat_rows = counts if counts>cap else 0`; prefix-sums exclusivos → `row_off`, `tile_off`; `num_segs`/`num_rows` são tensores [1] int32 lidos pelos kernels; `seg_expert/seg_row0/seg_rows` descrevem fatias de `tile_rows` (64) linhas de um expert; `row_token`/`row_weight`/`row_expert` mapeiam cada linha "fat" para o token/peso/expert original (via `searchsorted`). Kernels lançados com grid dimensionado pela capacidade e *grid-stride* sobre o número real de segmentos (máx 512 CTAs em y).
- `fm_gather_kernel`: `h13[row] = had128(x[token[row]] ⊙ gate_suh[expert[row]])` — a escala de entrada é multiplicação fp16×fp16 arredondada **antes** da Hadamard fp32 (reproduz a fronteira de arredondamento do E2).
- `fm_gateup_kernel`: 2 streams de B (gate e up) sobre as mesmas 128 colunas intermediárias; mainloop com `cp.async` e 4 estágios (TILE_K=32), B dequantizado em registradores por warp e reutilizado em todas as linhas do tile (MB=4 → 64 linhas); epílogo fundido: Hadamard + `svh` de gate e up, clamp (`g ≤ limit`, `|u| ≤ limit`), `sigmoid` em **precisão plena** (`exp` em double e `__fdiv_rn`, imune a `--use_fast_math`) × g × u, arredonda para fp16, × `down_suh` em fp16, Hadamard → `h2` (entrada já rotacionada da down).
- `fm_down_kernel`: GEMM de down com 2 metades de 128 colunas; epílogo Hadamard × `svh` × peso de roteamento e **`atomicAdd` vetorial de 16 B (`float4`, sm_90+)** em `out[token]`.
- `EXL3_FAT_MOE_MIN_CAPABILITY=(9,0)`; `hidden % 256`, `intermediate % 128`.
- Custo: scratch fp16 `h13 [cap, H]` + `h2 [cap, I]` com `cap = MNBT × 8` → 560 MiB cobrados do KV.
- Efeito: "numerically indistinguishable from E2 vs the LinearEXL3 reference"; +37–45% prefill; decode inalterado (E3 nunca roda em passos de decode).
- Conceitos: grouped GEMM com tabelas de segmento (padrão "MoE permute/unpermute"), fusão de epílogo, cp.async multi-buffer, atomics vetoriais, controle de arredondamento.

**M11 — Thin-decode FAST K4/N256 (`GLM53_EXL3_MOE_FAST=1`)**
- Arquivo: `overlay/patch_exl3_decode_pipeline.py` (derivado de `coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`); alias de ponteiros em `exl3.py:1319-1329`.
- Problema: o kernel stock `exl3_moe_kernel<4,256>` tem spill real (88 B de frame, 37 STL / 39 LDL) no SM121.
- Mecanismo: copia `exl3_moe_kernel.cuh` para `glm53_exl3_moe_fast_kernel.cuh` com template `bool shared_input`; quando `shared_input`, pula a Hadamard redundante da entrada do up e a GEMM de up lê o buffer já rotacionado do gate; instancia `<4,256,true/false>` com `MOE_FRAG_STAGES 1`/`MOE_SH_STAGES 8` (stock 3/3); dispatch host só se `K==4 && N_off==1 && GLM53_EXL3_MOE_FAST=1 && SM 12.1`; `shared_input` = ponteiros de `gate_suh` e `up_suh` são a **mesma alocação** (o overlay só faz o alias depois de `torch.equal` em todos os experts). Símbolo `glm53_fast_moe_version()==1`.
- Efeito: SASS 127 regs / 16 B frame / 0 LDL vs 128 / 88 B / 39 LDL; latência de camada +8% a +17%; paridade 119/119 (RMSE relativo ~1e-8); serving A/B/A2 +7,71% (structured-1), +8,88% (code-1), +14,59% (json-1); estudo numérico selado "INCONCLUSIVE" (0 flips, gate absoluto falha igualmente no stock).
- Conceitos: pressão de registradores, spills, profundidade de pipeline, reutilização de transformação.

**M12 — FP8 weight-only (Marlin) nas projeções densas (`GLM53_DENSE_FP8`)**
- Arquivos: `overlay/exl3.py:1895-2281` (`Glm53DenseFp8Method`), `overlay/patch_dense_fp8.py`.
- Problema: projeções BF16 densas (KDA in/out, MLA q/o, shared expert, MLP denso) custam banda de memória em decode.
- Mecanismo: `patch_dense_fp8.py` instala o `exl3.py` e faz os construtores do KDA (`kda.py`) e do MLA (`model.py`) **manterem o `quant_config` quando ele é `exl3`** (o stock zerava para BF16). Em `process_weights_after_loading`: `scales = amax(|W|, dim=1)/448` por canal de saída, `fp8 = clamp(W/scales, ±448).to(float8_e4m3fn)`, `prepare_fp8_layer_for_marlin(size_k_first=False)`; `apply` → `apply_fp8_marlin_linear` (W8A16: ativação continua BF16). Nunca toca drafter/MTP/visão, nem `kv_b_proj`; em TP3 mantém `f_b_proj/g_b_proj` BF16.
- Efeito: ~11 ms a menos por passo; +10% counting, prosa +12–19% sozinho, +37% prosa difícil com adaptive-k (2026-09-08, `dense,kda`); `all` vs `dense,kda`: −4,67% ms/ciclo [−5,11, −4,06] e −4,91% (2026-09-26); KL proxy 0,002–0,013 nats/posição, argmax 94–100% (`dense,kda`).
- Conceitos: FP8 e4m3 (max 448), escalas per-channel, Marlin (kernel W4/W8A16 mixed-input), memory-bound decode.

**M13 — Cópia BF16 para M grande no `in_proj` do KDA (`GLM53_KDA_BF16_LARGE_M`)**
- Arquivo: `overlay/exl3.py:2004-2248`; doc `docs/kda-bf16-large-m.md`.
- Problema: Marlin W8A16 é rápido em M pequeno e lento em prefill (M=7168: Marlin ~27,21 ms, W8A8 ~12,57 ms, BF16 ~7,36 ms). A tentativa W8A8 (#182) quantizava a ativação e produzia mudanças de probabilidade reprodutíveis (KL 7,99× o nulo, 23 flips).
- Mecanismo: materializa o **mesmo peso lógico** que o Marlin usa (e4m3 × escala BF16 armazenada; exato em fp32, 1 arredondamento para BF16) em blocos de 512 linhas; teste de cuBLAS com M=2 na carga; `apply`: se `m = x.numel()//k > 512` → `F.linear(x, w_bf16)`, senão Marlin. Boundary fixo guardado na layer para captura e replay concordarem.
- Efeito: RMSE relativo vs Marlin 0,002526 (vs 0,0266 do W8A8); TTFT −11,4%/−12,3%/−14,1%; +3,29 GiB/rank (79,65→82,94 GiB); em 2026-09-26 adotado com KV de 11 GiB.
- Conceitos: crossover de GEMM por M, quantização de ativação vs só de peso, dispatch por metadado de tensor.

**M14 — EXL3 denso (`Exl3LinearMethod`, pack `non_routed_exl3`, perfil H3)**
- Arquivos: `overlay/exl3.py:2284-2970`; porte de `Alexbob0/glm53-flash-dense-exl3-tp2` (MIT) ← `vcruz305/vllm-exl3`.
- Mecanismo: linears densos declarados pelo pack carregam `trellis/suh/svh/mcg|mul1` por shard de saída (QKV/merged column), com shards BF16 como cauda contígua (`bf16_shards`, ex. K/V do drafter). Loader lida com shard ids string (`q/k/v`), layers replicadas (`disable_tp` → geometria TP 1), `lm_head` só se vocab já for múltiplo de 64 (154.880 é). Forward via **custom op** `torch.ops.vllm.dense_exl3_forward` (opaco ao dynamo/inductor, capturável). Warmup do autotuner coop (`_dense_exl3_warmup_autotune`, `exl3.py:2429-2482`): o autotune do ExLlamaV3 sincroniza o stream na primeira vez que vê um hash de lançamento — dentro de captura isso trava (py-spy 20260924T151053) — então aquece cada forma × bucket de linhas {1,2,4,8,16} antes da captura. Retenção BF16 para M > 144 por tipo de módulo. Mutuamente exclusivo com FP8 no mesmo módulo; ABLIT recusa `o_proj` EXL3.
- `offset_draft_layer_prefixes` (`exl3.py:1774-1797`): o drafter tem camadas numeradas após o alvo (45..49) no runtime, mas o pack declara `model.layers.0…`.
- Efeito: não A/B medido contra o default (README).
- Conceitos: mixed precision por shard, custom ops e compilação, autotuning e CUDA graph capture.

### 3.C MoE cooperativo (decode) — `extensions/cooperative_moe/`

**M15 — Kernel cooperativo de 2 estágios especializado GLM TP2**
- Arquivos: `native/cooperative_moe_kernel.cuh` (base Turboderp `exllamav3@58d4d732`, headers `02aef45c`), `native/cooperative_moe.cu` (ABI C `glm53_coop_abi/info/launch`), `native/exl3_moe_coop.cuh`, `runtime.py` (adaptador), `prepare_profile.py` (gera overlay = `exl3.py` + rodapé `install(...)`), `build.sh`.
- Problema: no decode (8 linhas por sequência com DFlash k=7, até 32), o `exl3_moe` stock já está perto do teto de cópia; faltava agrupar os slots que escolheram o mesmo expert para ler os pesos uma vez por "run".
- Mecanismo: (rot) para bsz>1, um warp por (slot, chunk, projeção) aplica `x·suh` + Hadamard em scratch; (A gate/up) um bloco por (run de expert, grupo de 32/128 colunas, projeção): os até 8 slots do mesmo expert viram as linhas de uma MMA m16, o último a chegar num chunk de 128 colunas (contador de conclusão) aplica rotação de saída × svh, ativação, rotação de entrada da down × down suh; (B down) mesmo esquema, e o último de (token, chunk) soma os slots em ordem fixa e escreve a linha. Geometria 1 (A-wide/B-wide) por padrão. Contadores: A zera os de B e vice-versa → chamadas ordenadas, sem sobreposição (ubatching/DBO rejeitados). Hash do `.so` pinado (`aa3fe5e9…`), ABI 1, SM121, ocupância ≥1, scratch alocado após a carga e antes da captura; nunca re-tenta stock após lançamento parcial. Wrapper substitui `apply_exl3_fused_moe` só quando `1 ≤ rows ≤ 32`, `ids (rows, 8)`, `H=4096` etc.; prefill/E3 ficam stock. Bug achado na revisão ("init dentro de assert" some com `python -O`) corrigido.
- Efeito (2026-09-16, A/B/A, mediana de 5): structured ×1 72,03 → 77,18 tok/s (+7%), ms/passo 108,42 → 99,18; prosa ×1 34,09 → 36,18; ×2 agregado 113,91 → 124,38 (structured) e 48,49 → 51,46 (prosa); microbench 32 linhas 1,183 → 0,997 ms (1,18×); GPU gate 18/18 no screen de 0,3% do pico (não bit-exato).
- Conceitos: GEMV de M pequeno, reutilização de pesos entre slots, contadores de conclusão ("last block done"), kernels cooperativos, especialização em compile-time.

**M16 — Variante TP3/EP (ABI 2)**
- Arquivos: `extensions/cooperative_moe/tp3/**` (runtime, `native/dispatch.cu` que escolhe TU de 32 ou 64 linhas, `dispatch_policy.json`, `select_policy.py`, `manifest.py`, vendor de 9 headers com hashes em `PROVENANCE.json`).
- Mecanismo: I local 2048 (EP, sem shard intra-expert), ≤96 experts locais, 256/512 slots, chave de ordenação de 9 bits; política por linhas escolhida pela menor razão de pior caso sobre 3 faixas de EP × 2 padrões de roteamento, exigindo ≤0,97 (≥3% de ganho); linha 64 suportada mas stock (1,0568).
- Efeito (histórico, bundle combinado, não isolável): C1 +17,90% (código), +20,83% (ensaio); C8 agregado +61%; C8 por stream −17,53%; prefill 7,6K −8,92%.

### 3.D Speculative decoding DFlash2

**M17 — Modelo drafter DFlash2 (`overlay/qwen3_dflash2.py`)**
- Porte de `qwen3_dflash2.py` do vLLM main para esta imagem (o `LogitsProcessor` daqui não tem `get_top_k_tokens`, então usa logits + `torch.topk`).
- Arquitetura (do código): camadas `DFlash2Qwen3DecoderLayer` (Qwen3 com atenção SWA não-causal no bloco) envoltas por **convoluções dinâmicas agrupadas** (`DFlashGroupedConv`, `qwen3_dflash2.py:32-114`): `kernel_projection` (Linear `H → 2·taps·groups`) gera coeficientes por token; `base_kernel [2, taps, H]` + delta; a conv é causal ao longo das posições **dentro do bloco** (posição = `idx mod block_size`, `block_size = 1 + num_speculative_tokens`), aplicada antes (`prepare`) e depois (`finish`) da atenção e da MLP. `input_embedding_scale`, `output_multiplier`, `final_logit_softcapping` do `dflash_config`.
- **Seletor de candidatos** (`CandidateSelector`, `:204-251`): para cada posição do bloco, top-k candidatos do lm_head (`selector_top_k`); pontuação de arestas `unary_logits + einsum(predecessor[prev] ⊙ hidden_proj, successor[cand])` com codebooks de rank baixo `[vocab, rank]` (`selector_rank`) — um modelo bigrama condicionado ao hidden que escolhe uma **sequência coerente** entre os top-k de posições adjacentes. `hidden_projection` fica BF16 ("selector precision directly steers acceptance").
- Conceitos: block-parallel drafting (estilo difusão em bloco), conv dinâmica, reranking de candidatos com modelo de transição.

**M18 — Speculator DFlash2 (`overlay/dflash2_speculator.py`)**
- `_selector_walk_kernel` (Triton): para cada request percorre as 7 posições; em cada uma lê as pontuações condicionadas ao índice escolhido na posição anterior e escolhe via **Gumbel-max** (`gumbel_noised_argmax` vendorizado; argmax puro a temperatura 0; semente por request e posição; opção fp64). `_cache_draft_logits_kernel` grava as pontuações realizadas nos `draft_logits` (−inf fora dos candidatos) para o rejection sampling probabilístico.
- Conceitos: Gumbel-max trick, amostragem de caminho (greedy walk / Viterbi aproximado), rejection sampling.

**M19 — `patch_dflash2.py`: registro, causalidade, KV do drafter, dispatch**
- Mecanismo: instala os dois arquivos; `_dflash_layer_causal` honra `is_causal` do checkpoint (IncoAI DFlash2: `false` → atenção bidirecional no bloco); `decoder_layer_cls`/`model_cls` injetáveis; registry `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")`; KV do drafter vira `auto` quando o alvo usa `fp8_ds_mla`/`fp8`/`fp8_e4m3`/`fp8_e5m2`/`nvfp4`; `speculative_config.method == "dflash"` com arquitetura `DFlash2DraftModel` → `DFlash2Speculator`.
- Conceito: KV por grupo com dtypes diferentes, seleção de backend não-causal.

**M20 — `patch_dflash2_exl3.py`: drafter EXL3**
- Mecanismo: (1) antes de construir as camadas, `quant_config.offset_draft_layer_prefixes(start_layer_id)`; (2) ao montar o buffer de KV de contexto fundido (`_build_context_kv_buffers`), se `qkv_proj.weight` só tem as linhas K/V (q em EXL3, k/v BF16 via `bf16_shards=[1,2]`), usa o tensor inteiro; se é QKV BF16 completo, fatia `[q_size:]` como antes.
- Conceito: context KV do DFlash = projeção por posição do hidden do alvo (RMSNorm → GEMM KV fundida → K-norm → RoPE), precisa de peso plano.

**M21 — `patch_glm_eagle3.py`: hidden auxiliares no mHC**
- Mecanismo: `Glm5NextModel` ganha `EagleModelMixin` e `aux_hidden_state_layers`; no loop de camadas, após a camada `idx` com `idx+1 ∈ aux_hidden_state_layers`, materializa `hc_post(hidden, residual, post, comb)` e contrai os 4 fluxos (`hc_contract(..., layer.n)`) para `[tokens, 4096]` (ou `mean(dim=1)` se 3D), com `sp_all_gather` sob sequence parallel; `forward` devolve `(hidden, aux_hidden_states)`; classes ganham `SupportsEagle3`. Taps `(6, 15, 25, 34, 43)`.
- Conceito: EAGLE-3 (features de várias profundidades do alvo alimentam o drafter), hyper-connections.

**M22 — Adaptive-k (`overlay/patch_adaptive_k.py`)**
- Problema: com prosa a aceitação decai rápido (por posição 0,75/0,58/0,41/0,28/0,16/0,09/0,06); verificar 7 drafts custa ~8 linhas por request mesmo quando só ~2 serão aceitas; e os backends `FLASHINFER_MLA_SPARSE_SM120` e KDA só suportam FULL CUDA graph com **batch uniforme**.
- Mecanismo: `_Glm53AdaptiveK` (CPU): `observe(req, num_draft, num_accepted)` atualiza EMA (α=0,25; se tudo aceito e `saturate=max`, observa o máximo para poder subir); `choose` = maior valor do set ≤ `ceil(ema + margin)` após `min_steps`; `batch_k` no `schedule()` (usado pelo AsyncScheduler para dimensionar placeholders) = **mínimo** entre requests de decode (qualquer request com structured output ou ainda sem observações fixa o batch em k); `apply` recorta `spec_token_ids` no caminho síncrono. `cudagraph_utils.py` ganha `decode_query_lens = sorted(stock ∪ {k+1} ∪ {decode_query_len})` → `[3,5,8]`. Override por arquivo JSON a quente.
- Efeito (2026-09-08, 8 runs/prompt, bootstrap 95%): ensaio Silk Road +21%, sky/sunset +13%, hash-map +10%, código +5–15%, contagem inalterada; sem perda a temperatura 0.
- Conceitos: speculative decoding (aceitação por posição), CUDA graphs por forma, EMA, uniformidade de batch.

### 3.E KV cache híbrido, grupos e prefix caching

**M23 — Grupo do drafter no layout GLM-5-Next (`overlay/patch_glm5_drafter_group.py`)**
- Problema: o fast path `_get_kv_cache_groups_glm5_next` não aceitava camadas `SlidingWindowSpec` extras (o drafter tem 5). Se o drafter herdasse o bloco do manager MLA (1152, depois 3584), cada camada do drafter custaria dezenas de MiB por bloco e a concorrência logada ficava ~1×. Compact-64 sem slot-share queimava ids únicos por camada.
- Mecanismo (14 edits ancorados em `v1/core/kv_cache_utils.py` + guarda em `v1/worker/utils.py`):
  - separa camadas SWA de tipo exato (KpoolTailSpec também é subclasse de SlidingWindowSpec) e as anexa como **último** grupo (ids de grupo estáveis);
  - **exact-fit**: se `mla_page % draft_bytes_per_token == 0`, o bloco múltiplo de 64 que preenche exatamente a página MLA → camada i do drafter "co-possui" o tensor MLA i em ids disjuntos (como o Mamba), view contígua;
  - senão **padded slot-share**: `block_size=64`, `page_size_padded=mla_page` (656 vs 4096 B não encaixa) → view com stride de página; boot log `padded slot-share block=64 mla_page=2351104 (was block=16); draft_bytes/token=2048`;
  - custo por bloco inalterado (o drafter slot-shared não adiciona bytes; standalone adiciona `n_layers × draft_page`); `_max_memory_usage_bytes_from_groups` cobra a demanda de ids limitada pela janela;
  - `GLM53_DRAFT_KV_COMPACT=1`: `_glm53_draft_block_size` = maior múltiplo de 64 que divide o bloco MLA e cabe na página (896 no bloco 3584 com 2048 B/token; 640 no TP3 com página 1.679.360 e bloco 2560); preflight: todas as camadas SWA precisam ser do drafter DFlash (método + contagem de camadas), em qualquer caminho de agrupamento;
  - guarda do worker: página padded não pode ser dividida em kernel blocks menores (o stride físico se aplica uma vez por manager block → OOB) — backend precisa suportar o bloco inteiro (FLASH_ATTN declara múltiplos de 16).
- Efeito: pool de 1,75M tokens a 1M de contexto (2026-08-29); ocupação de um ~36k: 44,6% (compact-64 sem slot-share) → ~16% (slot-share); compact: ids de admissão 65 → 6 por request; ids de reserva −32,78% (180 → 121), repeat TTFT de prefixo 28.672 com cauda 64: 2,623 s → 0,302 s (−90%); "não é redução medida de VRAM".
- Conceitos: KV cache paginado, grupos de KV cache (hybrid allocator), block pool com ids globais, page size vs block size, sliding window attention.

**M24 — Hits híbridos de APC com o drafter EAGLE-like (`overlay/patch_hybrid_prefix_hit.py`)**
- Problema (issue #7, follow-ups pareciam 0% APC): (1) `dflash` é `use_eagle()`; o GLM não anota `is_eagle_group` (só o DeepseekV4 faz) → o coordinator marca **todos** os grupos como EAGLE e o MLA descarta sua última página de 3584; (2) o grupo SWA do drafter participa do `min` híbrido e, após o pop EAGLE, realinha por mais uma página de 3584.
- Mecanismo (estágios versionados, migração de instalações antigas, verificação por AST/bytecode de que cada estágio existe exatamente uma vez):
  - `eagle_group_ids` = só grupos `SlidingWindowSpec` exatos (MTP sem SWA mantém o fallback de todos);
  - o drafter **não reduz** `curr_hit_length`; se sua janela não cobre o hit MLA/Mamba, seus blocos ficam vazios e uma janela nova é alocada; Mamba continua no `min` (pular um miss de Mamba seria buraco de correção);
  - `[glm53-dflash-swa-replay-v1/v2]`: o drafter precisa de uma janela fresca completa (2048 tokens) após troca de APC; reutiliza o limite atual só se todo grupo DFlash devolveu janela completa, não-nula e verificada; senão faz o **replay clamp** (recua o hit por páginas de alinhamento até o sufixo fresco reconstruir a janela) e descarta os blocos do drafter;
  - `[glm53-dflash-eagle-verify-v3]`: um lookup EAGLE que falhou não conta como verificado numa passada seguinte;
  - `[glm53-dflash-boundary-lookup-v1]` (com `GLM53_DRAFT_KV_COMPACT=1`): o DFlash não precisa do bloco "lookahead" do EAGLE (o KV de contexto na posição p só depende do hidden em p), então o grupo é consultado terminando exatamente no limite (`drop_eagle_block=False`) e o manager fica não-EAGLE (1 id a menos por janela retida);
  - `[glm53-participating-fine-hits-v1]`: só grupos que participam do prefix caching podem vetar hits de grão fino (o KpoolTail, scratch, não veta mais); `[glm53-kpool-replay-floor-v1]`: o candidato de hit deixa pelo menos 4 tokens frescos para reconstruir o anel do indexer (`block_size` do KpoolTail); `[glm53-partial-replay-fallback-v1]`: uma cauda fina de Mamba sem janela do drafter tenta primeiro o checkpoint compartilhado anterior.
- Efeito: follow-up de ~8k reutiliza 7168/8004 (≈90%) e TTFT 1,30 s vs 8,53 s frio; 4× follow-ups concorrentes de ~7,5k: 7168 hits cada, 1,86–2,50 s.
- Conceitos: prefix caching por hash de blocos, coordinator híbrido (min entre grupos), EAGLE lookahead block, janelas deslizantes, correção de estado recorrente.

**M25 — Retenção de APC por grupo (`overlay/patch_apc_per_group_retention.py`)**
- Problema: o grupo do drafter hasheia 33 ids a cada limite de 3584 tokens (33 de 38 ids de um segmento); esses blocos vão para o **fim** da fila LRU (mais protegido) enquanto os blocos MLA/Mamba que realmente dão o hit são liberados depois → inversão de prioridade (87% dos 642 ids gastos num grupo cujo hit é descartado pelo `min`). O único knob (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`) era global e engrossava o grid do Mamba/MLA (hits de subagente caíram para 45%).
- Mecanismo: `retention_interval` por grupo; o grupo EAGLE-exempt do drafter recebe `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA` (validado sempre: inteiro, ≥0, múltiplo do scheduler block, ≤1.000.000; setá-lo sem grupo elegível é erro de boot). `0` = só limites alcançáveis e blocos só-drafter com prioridade de eviction mais baixa; Mamba esparso preserva o checkpoint anterior necessário ao replay.
- Efeito: `0/0` em 128K: edição a 90% 14,70 s → 112,49 s e branch a 90% 3,35 s → 99,89 s (reuso 111.104 → 0); append inalterado 2,80 s ambos; 4 históricos de 210K retidos (~2,5 s). Default: unset.
- Conceitos: políticas de eviction, retenção esparsa, custo em ids por segmento.

**M26 — No-store por request (`overlay/patch_apc_no_store.py`)**
- Problema: vLLM tem opt-out só de leitura (`skip_reading_prefix_cache`); `cache_salt` isola mas ainda grava; `BlockPool.free_blocks` põe blocos **hashados** no fim da fila (LRU) e não-hashados na frente (LIFO); um job de batch "reordena" o LRU a seu favor e expulsa a conversa ociosa do dono.
- Mecanismo: `SamplingParams.skip_writing_prefix_cache` (também via `vllm_xargs`), validação estrita 0/1 no `__post_init__` (HTTP 400 em valor inválido), resolvedor que nunca levanta no `Request`, guardas nos dois sítios `_insert_block_hash` do `block_pool.py` (`cache_full_blocks`, `cache_partial_block`) — **não** no `allocate_slots`, porque `num_cached_block` é o sentinela de request em execução para SWA/drafter. Transacional (temp + `os.replace`), idempotente.
- Conceitos: LRU vs LIFO, hash de blocos, invariantes do KV manager.

**M27 — Liberação de estados Mamba "align" superados (`overlay/patch_mamba_align_state_free.py`)**
- Problema: em `mamba_cache_mode="align"` o manager lembra o bloco do estado anterior num único slot `last_state_block_idx`; sob scheduling assíncrono (prefixo processado atrasa 1 batch) e chunks de 7168 atravessando blocos de 3584, o slot é sobrescrito antes de o bloco ser liberável → o bloco fica preso até o fim do request.
- Mecanismo: lista por request `stale_state_block_idxs` (ascendente), liberada sob o mesmo predicado; reserva por request passa de `2 + num_speculative_blocks` para `1 + max_concurrent_batches + num_speculative_blocks` (1 síncrono, 2 assíncrono).
- Conceitos: estado recorrente (SSM/linear attention) em blocos, async scheduling, vazamento de blocos.

**M28 — Chunking alinhado ao Mamba (`overlay/patch_mamba_align_chunking.py`, v2)**
- Problema: o alinhamento usava `cache_config.block_size` (a menor página, do drafter); o back-off EAGLE valia sempre; grants menores que um bloco arredondavam para zero (inanição com caps pequenos do mixed prefill).
- Mecanismo: checkpoints compartilhados usam o **LCM do scheduler** (`self.block_size`); páginas Mamba privadas menores forçam parada no próximo limite delas; back-off EAGLE só se um grupo participante não-SWA é EAGLE; só grants ≥ um bloco entram no arredondamento para página cheia. Rejeita decode-floor < v5.
- Conceitos: alinhamento de checkpoints de estado, LCM de block sizes, progresso garantido.

**M29 — Log honesto de capacidade de KV (`overlay/patch_kv_capacity_log.py`)**
- Problema: `GPU KV cache size: 1,553,140 tokens` é `max_concurrency × max_model_len`, não um tamanho de cache (dois revisores leram como 1,5M de prefix cache; eram 643 ids, ~57K tokens de conversa cacheada).
- Mecanismo: após a linha stock (inalterada), uma linha por grupo (tipo, block_size, page, blocos por request, participa ou não) e um resumo: ids usáveis (`num_blocks − 1`, o id 0 é o null block), ids por segmento alinhado (1+4+33 = 38), capacidade = `642 // 38 × 3584 = 57.344`; retém o número quando um spec não é modelado.
- Conceitos: contabilidade de blocos do vLLM, alinhamento = LCM dos block sizes.

**M30 — Clamp do slot mapping do KpoolTail (`overlay/patch_kpool_tail_slotmap.py`)**
- Problema: `KpoolTailSpec` é scratch circular de 1 bloco, mas o kernel Triton genérico de slot mapping faz `block_table[req, pos // block_size]` sem limitar ao tamanho da linha → para `pos ≥ block_size` lê lixo e os kernels de seed/update do kpool escrevem através de ids errados (~2k tokens de geração: crash ou corrupção silenciosa do indexer de outra camada).
- Mecanismo: `block_indices = tl.minimum(block_indices, block_table_stride - 1)` em `vllm/v1/worker/block_table.py` (identidade para os outros grupos).

**M31 — Stride do seed da cauda do kpool (`overlay/patch_kpool_tail_seed_stride.py`, backport vLLM #57477)**
- Problema: `_kpool_tail_seed_kernel` endereçava o tail como array denso `base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM` (stride 2048 B), mas o tail é `as_strided` da página do indexer (118.272 B no bloco 3584) → cada prefill deixava o próprio tail sem seed e escrevia K/gate bf16 no bloco de indexer de outro request (tipicamente o primeiro prompt longo com prefix cache após o boot).
- Mecanismo: `base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM`, score em `base + KPOOL_HEAD`, passando `tail.stride(0)`/`tail.stride(1)`; detecta "already upstream" e rejeita meia-correção.

**M32 — Workspace do indexer do tamanho certo (`overlay/patch_indexer_workspace.py`)**
- Problema: `get_max_prefill_buffer_size` = `max_model_len × 40` entradas × 132 B = **5036,40 MiB** a 1M (medido com `VLLM_DEBUG_WORKSPACE=1`), reservado no profiling (sai do KV) e travado; o workspace é dimensionado em tokens mas indexado em pools (o builder já divide por `compress_ratio = index_kpool = 4`; o DeepseekV4 divide, o glm5next não).
- Mecanismo (`rightsize`): `entries = min(max_num_seqs, MNBT) × cdiv(max_model_len + num_speculative_tokens, compress_ratio)`, limitado ao stock; o builder levanta erro se `hf_text_config.index_kpool` ≠ `kv_cache_spec.compress_ratio` (nas duas direções). Prova por exaustão que a lista de chunks é idêntica à stock para qualquer batch legal.
- Efeito: 125,9 MiB (4 seqs) / 503,5 MiB (16 seqs) → +4,79 / +4,43 GiB de KV (≈ +26–28% estimado).

### 3.F Scheduler

**M33 — Política de mixed prefill "fair" v5 (`overlay/patch_scheduler_decode_floor.py`)**
- Problema (issue #6 e o relato do "freeze" de 20 min): um lane de decode precisa de ~8 tokens (1 + k=7); o budget restante ia para um chunk de prefill de outro request, que tira o passo do caminho de FULL graph uniforme e custa segundos (sparse-MLA + indexer + MoE); `skip` deixava newcomers esperando o decode inteiro (221 s, sem limite de idade); `0` admitia em ~1 s mas derrubava o incumbente 10–36× (28,8 → 0,8 tok/s).
- Mecanismo v5 (`# [glm53-decode-floor:v5]`, migra v1–v4): decodes alocados primeiro pelo scheduler base (reserva real de tokens/input/draft); só prefill que **contende** com um decoder gera dívida (prefill solo só gera amostra de custo); crédito acumula a `SHARE` do tempo de engine contabilizado (proxy de tempo ocupado no host, contado uma vez para spans assíncronos); ajuste linear `dt = a + b·n` sobre amostras solo+mistas (janela 24, escalado pelo p75 de real/ajuste, clamp [1; 1,5]); alvo = maior degrau do ladder (128..2048) cuja estimativa cabe em `MAX_STEP_MS`, **poupando crédito** para esse degrau em vez de gastar em chunks pequenos; newcomer nunca servido ganha uma sonda imediata; após `MAX_INTERVAL_MS` um prefill já servido pode pegar 1 chunk emprestado só depois de quitar a dívida; um prefill assíncrono em voo bloqueia o próximo turno misto; seleção por último serviço + round-robin; alinhamento híbrido preservado (`_glm53_align_prefill_limit`).
- Motivo do v5: o v4 escalava uma única amostra linearmente (precificava 1024 tokens a ~2,7 s, real ~0,95 s) e ficava preso em 128 tokens (~70 tok/s efetivos).
- Efeito (TP2, 2026-09-15): newcomer 2k 10 s dentro de um ensaio com thinking → primeiro token em 11,6 s durante o ensaio (skip: espera ~163 s); newcomer ~30k → 163,6 s, todos os 29.365 tokens em passos mistos; incumbente 20,6–20,9 tok/s vs 24,7 solo (1,18–1,20×); gaps p50/p95/max 0,09/0,19/1,39–1,44 s. TP3: 8,7 s e 110,0 s.
- Conceitos: chunked prefill, fairness por tempo de serviço (token bucket / crédito), estimativa de custo online, head-of-line blocking.

### 3.G Carga de pesos e memória UMA

**M34 — Cold load em UMA com páginas de 64 KiB (`overlay/patch_cold_load_uma.py`)**
- Problema: no GB10, `torch.cuda.mem_get_info()[0]` = `MemFree` do host, e o page cache conta como usado; após o rsync de 164 GiB, `MemFree` ~2 GiB com `MemAvailable` ~117 GiB → InstantTensor aborta (`buffer_size (1268776960 B) exceeds device memory budget (1255964672 B)`) ou encolhe `io_depth` 512 → dezenas. Separadamente, `cuMemcpyHtoDAsync` trava com fonte mmap file-backed num kernel de 64 KiB.
- Mecanismo: mede `MemAvailable`, detecta UMA (`|cuda_free − MemFree| < 8 GiB`), passa `max_free_mem_usage` explícito (pode passar de 1 relativo ao cuda free) e `buffer_size` pinado de 4 GiB (janela = buffer + maior shard + 2 GiB); `clone()` de cada tensor do mmap quando `SC_PAGE_SIZE != 4096`.
- Efeito: com page cache cheio (118 GiB): stock falha; patch carrega 164 GiB em 36 s a 4,82 GB/s, `/health` em 230 s; com cache limpo: sem ganho (35–36 s ambos).
- Conceitos: memória unificada, page cache reclamável, O_DIRECT/io_uring, pinned memory, tamanho de página do kernel.

**M35 — Staging auto/safetensors e read-ahead (`overlay/patch_loadclone.py`, v2; vendorizado em `overlay/tp3/.../weight_utils.py:829+`)**
- Mecanismo: `GLM53_LOAD_CLONE=1` clona tensores auto/lazy mmap para CPU antes de entregar; `GLM53_LOAD_PREFETCH=0..16` lê shards locais adiante em threads com buffer de 1 MiB e dá `POSIX_FADV_DONTNEED` nos shards consumidos; NFS/Lustre mantêm prefetch stock; compõe com o PR #230 em qualquer ordem.

**M36 — Pular a dry-capture de memória de CUDA graphs (`overlay/patch_skip_cudagraph_profile.py`)**
- Problema: `profile_cudagraph_memory()` sempre roda e só é aplicado com `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`; com `CG_ESTIMATE=0` são 11 s de boot jogados fora.
- Mecanismo: condiciona o profile ao flag.

### 3.H API / serving

**M37 — Placeholders de vídeo e `persistent_topk` (`overlay/patch_glm_video_placeholders.py`)**
- Problema: o `glm4_1v` monta um bloco de imagem por timestamp; um clipe de 1 s pode gerar 3 timestamps com `video_grid_thw T=1` → 100 tokens do encoder para 300 slots. E o `persistent_topk` de decode excede a SMEM do GB10 em sequências longas.
- Mecanismo: hook de import (via `.pth`) que substitui `_construct_video_placeholder` usando timestamps glm46v alinhados a `grid_t` (`_align_timestamps`); desliga `persistent_topk` (`if False and …`) para usar `top_k_per_row_decode`.

**M38 — Stops dormentes no raciocínio (`overlay/patch_suppress_stops_in_reasoning.py`)**
- Problema: o v1 casa `stop` contra todo o stream; com o prompt terminando em `<think>`, o CoT repete stops do harness e o request termina com `content` vazio (DSpark #42).
- Mecanismo: guarda no detokenizer ativa só se o último token do prompt é o marcador de início do raciocínio; em spec decode, se `</think>` chega no mesmo chunk k+1, `stop_check_offset` pula além do marcador.

**M39 — `tool_choice:"none"` na decodificação (`overlay/patch_tool_choice_none.py`)**
- Mecanismo: `Glm47MoeParser.adjust_request` adiciona `TOOL_CALL_START` a `request.bad_words`; o sampler mascara o token em toda posição, inclusive nas posições de draft na verificação (um opener draftado é rejeitado). Mantém o bloco de ferramentas no prompt (preserva o prefixo compartilhado; `--exclude-tools-when-tool-choice-none` foi rejeitado por isso).

**M40 — Backports XGrammar (`overlay/patch_xgrammar_termination.py`)**
- vLLM #52805 (`12f64b39`): para `accept_tokens()`/`validate_tokens()` no primeiro token terminal, ignora avanços após término, limpa a flag no reset. vLLM #53046 (`c6e19b3`): valida drafts produzidos antes de um fim de raciocínio no meio da janela antes de avançar a gramática nova (evita `Failed to advance FSM`).

**M41 — Reset de cache opt-in (`overlay/patch_cache_reset.py`)**: `elif GLM53_EXPOSE_CACHE_RESET=1` no `build_app` monta só `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache` (sem `/sleep`, `/rlhf`, `/rpc`…). Rotas root não passam pelo bearer guard.

**M42 — Default de `max_tokens` omitido (`overlay/patch_default_max_new_tokens.py`)**: muda só o fallback de `get_max_tokens()` quando o request omite; preserva a omissão no before-validator do Pydantic para distinguir de `16`/`null` explícitos.

**M43 — Janela de spin do `SpinCondition` (`overlay/patch_spinwait.py`)**: substitui o literal de 1 s da janela de busy-loop do leitor por `GLM53_SPINWAIT_MS` (1..1000). Sweep TP2 congelado em MNBT 2048: 16 ms = +0,95% decode e −85,3% CPU ativa do EngineCore; 2 ms −1,68%; 32/64 ms ~−1,8%.

### 3.I TP=3 / TP=4

**M44 — Overlays de forma TP3 (`overlay/tp3/**`, FlyCockpit MIT)**
- `patch_tp3_glm.py`: cabeças 64→66 (local 22) em `num_attention_heads`, `num_key_value_heads`, `linear_attn_config.num_heads`, `linear_num_heads`; `A_log` 1-D de 64 → pad para 66; pad genérico de tensores de checkpoint para a forma do parâmetro; vocab `padding_size = lcm(64, tp) = 192` → 154.944; shared expert I 2048→2112 (TP-shardado; replicar via `disable_tp` triplicaria o MLP compartilhado); `moe_intermediate_size` dos experts roteados **não** é tocado.
- `pad-tp3-config.py`: pad idempotente do `config.json` (o `SpeculativeConfig` do MTP lê o arquivo) e do drafter GQA 32/8 → 36/9 (mantendo a razão 4 exigida pelo FlashInfer `qo % kv == 0`).
- `patch_exl3_ep_shard.py`: sob EP o rank já possui experts inteiros; usa `moe_parallel_config.tp_size` (1 em EP puro) em vez do world size (evita `2048 not divisible by 3`).
- `patch_exl3_expert_map.py`: `expert_map` é property read-only sob EP → cacheia em `_exl3_pinned_expert_map`.
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` (vendorizado): o decode SM120 só instancia cabeças locais {8,16,32,64,128}; pad Q 22→32 **em todo tamanho de chamada** (o prefill também rejeitava 22: "Unsupported sparse-MLA prefill configuration: model=GLM_NSA num_heads=22") e fatia de volta (softmax é por cabeça, as extras têm Q zero).
- `vllm/model_executor/parameter.py`: `_pad_then_narrow` quando o pad de TP empurra o fim do shard além da dimensão armazenada; `vocab_parallel_embedding.py`, `weight_utils.py` (loader v2).
- Não pad para 96 (um rank vira todo cabeças KDA dummy → logits colapsam).

**M45 — FlashKDA no prefill (`overlay/patch_flashkda_tp3.py`, adaptado de NNNtrance, integração original vLLM PR #55737)**
- Mecanismo: resolver `HAREM_KDA_FLASHKDA` / `additional_config["kda_prefill_backend"]` (`auto/triton/flashkda`); exige CUDA SM90/10x/12x (cubin família `12.0f` roda no sm_121), bf16, `head_dim=128`, gate limitado; valida o schema de 14 argumentos de `torch.ops._flashkda_C.fwd` (`q,k,v,g,beta,scale,out,workspace,A_log,dt_bias,lower_bound,initial_state,final_state,cu_seqlens`); estado recorrente FP32; buffers pré-alocados no workspace manager (graph-safe) com guardas de capacidade; o kernel recebe gate e beta **crus** e aplica L2-norm de q/k e o gate limitado internamente; escreve direto no buffer de saída da camada no passo não-spec. Só muda o prefill em chunks, não o decode recorrente. Validação GPU do branch upstream pendente.

**M46 — Slicing do sparse-MLA no TP4 (`overlay/patch_sparse_mla_slice.py`, de punkjazz-labs)**
- Mecanismo: com `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64`, a chamada final `flashinfer_trtllm_batch_decode_with_kv_cache_mla` roda em fatias de ≤64 linhas de query; pré-imagem e resultado pinados por SHA-256; falha fechada fora da geometria qualificada (Q BF16 `[T,16,576]`, páginas `[64,656]`, top-k 2048, workspace ≥ 33.685.504 B). Mitiga o stall de 4 ranks (#128/#159/#223), não corrige a race. Qualificado: soak de 153 min, erro máx 0,00185609 vs referência FP32.

### 3.J Abliteração

**M47 — ABLIT em tempo de carga (`overlay/ablit_runtime.py`, `overlay/patch_ablit.py`, `ablit/*`)**
- `patch_ablit.py`: copia o runtime para `models/glm5next/nvidia/glm53_ablit.py` e injeta `_glm53_ablit_maybe_apply(self)` no fim de `Glm5NextModel.load_weights` (camadas 0–44) e de `Glm5NextMTP.load_weights` (bloco 45), antes da captura de graphs.
- `transplant` (padrão via `auto`): substitui `o_proj` L15–45 pelos tensores do doador (`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`), baixados por range-request (`fetch_transplant.py`: lê o cabeçalho safetensors, pega só os bytes de `…layers.{L}.self_attn.o_proj.weight`, ~2,7 GiB, SHA-256 no `MANIFEST.json`). Como o `o_proj` é BF16 nativo tanto no NVFP4 quanto no EXL3 (byte-idêntico), o transplante reproduz exatamente a edição publicada ("dealign-oproj-transplant", layers 15–45 editadas, 0–14 âncoras de segurança).
- `proj`: `W' = W − α · r (rᵀW)/‖r‖²` (`apply_to_o_proj`, `ablit_runtime.py:175-198`); `o_proj` é `RowParallelLinear` (shard só na entrada; as 4096 linhas de saída estão em todos os ranks) → a edição por linhas é idêntica em cada rank, sem coletivas. Em TP3 o doador recebe pad no fim da dimensão de entrada para casar com as cabeças 66.
- `LAYER_MAP.json`: 45 camadas + MTP, papéis (`safety-anchor` 0–14, `edit` 15–44, `mtp-edit` 45), formas `o_proj` 4096×8192 (linear) e 4096×16384 (DSA). Direções: `refusal_direction_glm53_dealign_late.pt` (fonte `dealign-oproj-svd-L15-35-39-43-45`), `refusal_direction_glm53_bf_oproj.pt` (`blackfrost-oproj-svd`, `alpha_ref 3.0`, `n_layers 43`, `skip_early_ref 2`), vetores fp32 de 4096.
- Efeito: fidelidade por camada `rel_l2` médio ≈ 0,126 (L44 0,737 vs 0,74 publicado); o gate do doador: 32/32 bypass no stack NVFP4; direções de projeção chegam só a 9/32. KLD não medido com ablit.
- Conceitos: direção de recusa (representation engineering), ortogonalização de pesos, transplante de tensores.

### 3.K Ferramentas, laboratório e verificações de boot

**M48 — `tools/`**
- `dense_overlay.py`: cria pack overlay que faz symlink de todos os arquivos do pack TR3 e adiciona **um** safetensors com os linears não-expert EXL3 (range-read de `turboderp/GLM-5.3-Flash-exl3` branch `4.05bpw`, commit `2a30229e`, ~5,3 GB), índice reescrito e `config.json` com `non_routed_exl3` (opção `--lm-head`, `--draft-layers 45` para o MTP, `--prefix-rewrite model.language_model.:language_model.model.`).
- `dflash2_exl3_convert.py` / `dflash2_exl3_quant.sh` / `dflash2_exl3_package.py`: quantiza o drafter DFlash2 para 6 bpw com o fork `MiaAI-Lab/exllamav3@63b32f0` **sem calibração** (DFlash2 tem `caps["uncalibrated_quantize"]=True`: quantiza contra um Hessiano sintético); `k_proj/v_proj` ficam BF16 (sentinela 16, o KV de contexto fundido precisa de tensor plano); `kernel_projection` e `fc` entram no orçamento (desligáveis); `candidate_selector.hidden_projection` fica BF16; o pacote copia byte-idênticos os tensores BF16 do snapshot original e escreve `quantization_config` com prefixos relativos `model.layers.N`.
- `stage_dense_h3.py` / `pack_profile.py`: revisões = SHA-1 das identidades dos arquivos (reprodutível), `refs/glm53-dense-h3` (não mexe em `refs/main`), validação de hashes/cabeçalhos antes de qualquer restart.

**M49 — `kernel_lab/exl3/` (desenvolvimento, fora da imagem)**
- `reference.py`: oráculo NumPy — `pack_trellis`/`unpack_trellis` (16 spans por tile, MSB primeiro, swap de palavras int16), `decode_mcg` (multiplica por `0xCBAC1FED`, `lop3.b32` com máscaras `0x8FFF8FFF`/`0x3B603B60`/LUT `0x6A`, soma das duas metades fp16), permutação tensor-core, Hadamard Sylvester normalizada de 128, `reconstruct_weight` (Hadamard à esquerda nos blocos de entrada × `suh`, Hadamard à direita × `svh`) e `matmul_streaming_reference` (rotaciona a ativação, decodifica um tile 16×16 por vez e descarta). `abi.py`: contrato `trellis [K/16, N/16, 16·bits]`, `suh [K]`, `svh [N]`, `mcg` escalar. `benchmark.py`: sweep de táticas SM121 (4 formas de tile, quotas de SM) com recibos `atlas.kernel-benchmark/v1`. Evidência: K1–K8 streaming vs materializado, erro máx 1,08e-6; empacotamento K3 byte-idêntico ao packer CUDA; decode MCG casa com o CUDA nos 65.536 estados. Nenhum speedup declarado (sweep SM121 não rodou). Nota: SM121 é Blackwell de consumo → `mma.sync` (não `tcgen05`).

**M50 — Warmup de formas pós-`/health` + canário (`scripts/boot-shape-warmup.sh`)**
- Queima os `BLOCK_SIZE = min(256, next_pow2(scheduled + 8))` do DFlash2 (s ∈ {1,24,56,120,248} → blocos 16…256), prefills {3584, 7168, 14336, 65536} (kernel `BuildPrefillChunkMetadataKernel`), variantes de sampler (top_k só / top_p só; o `generation_config` injeta `top_p=0.95`), e kpool; com Triton/TileLang cache persistentes, cada forma compila uma vez por imagem. Canário: resposta "OK" a temperatura 0 e DFlash aceitando >0 de ≥64 drafts (assinatura do #249: tudo "!!!!" com aceitação ~0) → falha o start.

---

## 4. Números e benchmarks (todos do repositório; datas como informadas)

### 4.1 Hardware/topologia de referência
- 2× NVIDIA GB10 (DGX Spark), 128 GB UMA LPDDR5x por nó (`MemTotal` ~121,69 GiB utilizável: "0.88 × 121.69 = 107.09 GiB"), `sm_121a`, CUDA 13.0, arm64, kernel `6.17.13-rocket64k` (páginas de 64 KiB).
- Pico BF16 denso GB10 ~125 TFLOPS; banda LPDDR5x efetiva ~230 GB/s (estimativas em `docs/improve-prefill.md` §2).
- Interconexão: ConnectX-7 QSFP RoCEv2, porta de 200 Gb/s (ethtool); NCCL da imagem (2.30.7). NVMe O_DIRECT ~4,9 GB/s por leitor.
- Um all-reduce por camada no TP2: 45 × (1024×4096×2 B) ≈ 378 MB por passo de 1024 tokens ≈ 15–20 ms de barramento (estimativa).

### 4.2 Decode
| Data | Config | Resultado |
|---|---|---|
| 2026-08-28 | sparkDash, DFlash2 k=7, structured/code, temp 0, thinking off, 400 tok, 1M ctx, pool 1.754.237 | ×1: TTFT 719 ms, **62,9 tok/s**; ×2: 51,7 stream / 103,3 agg; ×4: 37,1 / 146,5 |
| 2026-08-30 | lab `tests/bench_decode.py`, C4, draft TP2 | Structured **65,1** (aceitação 0,959, 6,71 tokens/passo); Prose (hash-map) **27,1** (0,341, 2,39/passo); TP=1 lab 61,7/26,9; contexto longo 60–100k 24–27; MTP k=2 ~24,6 |
| — | aceitação por posição (lab) | Structured 0,98/0,98/0,94/0,94/0,91/0,83/0,83; Prose 0,75/0,58/0,41/0,28/0,16/0,09/0,06 |
| — | `TRITON_ATTN` fixado no drafter | ~29 tok/s, aceitação 0,31 (pos0 ok, demais colapsam) |
| 2026-09-08 | adaptive-k `ema` + dense FP8 `dense,kda`, sem coop | Prosa ×1 **32,1**, ×2 22,1 stream (41,2 agg), TTFT 268/399 ms |
| 2026-09-08 | ganhos medidos | adaptive-k: Silk Road +21%, sky/sunset +13%, hash-map +10%, código +5–15%; FP8 dense,kda: ~11 ms/passo, contagem +10%, prosa +12–19%, prosa difícil +37% com adaptive-k |
| 2026-09-16 | cooperativo geo1 (C1) vs stock, A/B/A, mediana de 5 | Structured ×1 72,03 → 77,18; ×2 60,18 → 65,90 (agg 113,91 → 124,38); Prosa ×1 34,09 → 36,18; ×2 25,00 → 26,66 (48,49 → 51,46 agg); Coding ×1 46,86 → 51,45 |
| 2026-09-16 | cooperativo (doc `cooperative-moe.md`) | Structured 78,0 (geo2) / **80,0** (geo1); Prosa 34,8 / **37,4** |
| 2026-09-17 | sparkDash prosa, adaptive-k + dense,kda + coop, 850k, KV 14 GiB | ×1 **36,1** (agg 37,1), TTFT 333 ms; ×2 25,0 (51,1); ×3 22,3 (65,8); ×4 19,4 (75,3) |
| 2026-09-26 | `GLM53_DENSE_FP8=all` vs `dense,kda` | ms/ciclo −4,67% [−5,11; −4,06] / −4,91% [−5,42; −4,44]; tok/s +5,2/+4,3% |
| main `3d6ffbd` | `GLM53_EXL3_MOE_FAST=1` (TheGrill A/B/A2) | structured-1 72,74 → 78,36 (+7,71%); code-1 67,83 → 73,86 (+8,88%); json-1 51,32 → 58,81 (+14,59%); structured-2 e prose-1 retidos pelo range-overlap |
| 2026-09-21 | compact draft pages ON vs OFF (custom) | structured +1,15%, code +1,21%, prosa −3,30% (holdout −0,93%) — "no universal decode speedup" |
| TP3, 2026-09-17 | sparkDash prosa 3× | ×1 **40,1**, TTFT 255 ms; ×2 28,7 (56,6); ×3 25,5 (75,5); ×4 22,8 (88,4); lab 09-14: structured 87,8, code 54,9, prose 39,6 (TP2 mesmos prompts: 73,4/45,0/32,9) |
| TP3 bundle combinado (histórico) | coop ABI2 + FlashKDA + FP8 amplo + 8 slots | Counting 101,655 tok/s, hash maps 44,151; aceitação 95,88%/49,62% |
| TP4, 2026-09-02 | 4 Sparks | decode 1,45× vs TP2; `DFLASH_TOKENS=3`: 53% aceito vs 30% com 7, prosa 31–33 vs 26–28 |

### 4.3 Prefill frio (prompt tokens / TTFT)
| Data | Config | Resultado |
|---|---|---|
| 2026-08-29 | MNBT 1024, pré-E2 | 8k 772; 12k 896; 16k 893; 100k 947; 256k 936; 300k 928 tok/s; ~1,08 s por chunk de 1024 |
| 2026-08-29 P0 (profiler, 100k) | atribuição do passo de 1,086 s | `moe_forward_shared` 623 ms (63%): `exl3_moe` 134 ms, `reconstruct_kernel` 64 ms, `index_add_` 52 ms; `aten::mm` 102 ms; NCCL 71 ms; sparse MLA 43 ms; KDA/GDN ~63 ms |
| 2026-08-29 P1 | ladder de MNBT | 2048 vence (8k 895; 16k 953; 100k 975); 3584 e 4096 revertidos; 8192 proibido (smem do indexer) |
| 2026-09-01 (PR77, E2) | MNBT 2048, A/B/A | 8k 941 → 1132 (+20,33%); 100k 1023 → 1242 (+21,36%); 300k 995 → 1201 (+20,70%) |
| 2026-09-06/07 (E3 vs E2) | 500k, util 0,84, MNBT 7168 | 16k 1578 (+37%); 32k 1624 (+40%); 64k 1637 (+40%); 128k 1629 (+38–42%); 256k 1576 (+45%) |
| 2026-09-07 (E3 shipped) | 900k, util 0,86, rightsize, cap 32, MNBT 7168, C4 | 8k 1492,1; 16k 1553,7; 32k 1428,2; 64k 1587,0; 128k 1561,7; 256k 1516,8 |
| KDA BF16 large-M | flag-only | 16k 11,41 → 10,12 s (−11,4%); 100k 70,86 → 62,12 s (−12,3%); mixed C3+100k 76,21 → 65,43 s (−14,1%) |
| TP4 | 282k | 1162 vs 901 tok/s (+29%, "fabric-bound") |
| Camada MoE isolada | 7168 tokens | E2 77–91 ms → E3 31 ms |

### 4.4 Prefix caching / APC
- ~8k follow-up: 7168 hits de 8004 (89,6%), TTFT 1,30 s vs 8,53 s frio; 12k: 10.752 hits, 1,94 s; 16k: 14.336, 2,18 s; 4× ~7,5k concorrentes: 7168 cada, 1,86–2,50 s.
- Retenção `0/0` (2026-09-05, 128K): append 125.440 hits (2,80 s) igual; edit 90%: 111.104 → 0 hits, 14,70 → 112,49 s; branch 90%: 3,35 → 99,89 s.
- Compact pages (2026-09-21): ids por request de tamanho máximo 180 → 121 (−32,78%); prefixo de 28.672 com cauda 64: 2,623 → 0,302 s; stock + compact ON: 814.571 tokens frios em 633,7 s e repeat em 2,748 s (813.568 hits).
- Ladder de concorrência 2026-08-31 (MNBT 2048, 16 seqs): code agg 41,9 (×1) → 127,2 (×16); até 4 lanes de 50K ficaram totalmente quentes (hit 0,999).

### 4.5 KV cache / memória
- `fp8_ds_mla`: 656 B/token/camada MLA × 11 camadas; drafter 2048 B/token/camada (TP2).
- Pools: 1.754.237 tokens / 690 blocos / 18,67 GiB (2026-08-29, 1M, util 0,87, MNBT 2048); 1.243.902 / 1,24× (2026-09-01, MNBT 7168, rightsize, E2); 1.572.073 / 1,85× a 850k (14 GiB, compact, 2026-09-26); **1.233.779 / 1,45× a 850k (11 GiB, FP8=all + KDA BF16, adoção 2026-09-26)**; TP3: 3.230.656 / 3,23× a 1M (40 GiB, 2213 blocos).
- 1 request de 1M a MNBT 7168 precisa de 14,52 GiB (~7,4 GiB fixo + 7,1 GiB por 1M); 500k: 10,98 GiB; 850k: 13,56 GiB.
- Capacidade real de conversa cacheada (idle, densa): 642 ids / 38 por segmento = 16 segmentos = 57.344 tokens (boot de 643 ids); 819 ids → 75.264.
- Indexer workspace: 5036,40 MiB (stock, 1M) vs 125,9 MiB (rightsize, 4 seqs).
- Pesos: "Model loading took 79.65 GiB" por rank (82,94 com KDA BF16); TP3 ~55,75 GiB por nó.
- InstantTensor: KV disponível ~12,4–12,5 GiB vs ~17–18 GiB sem o loader; 164 GiB em ~36 s @ ~4,8–4,9 GB/s.
- Multimodal: frame 1080p 2691 tokens sem cap, 2040 a 2048, 1008 a 1024; 33 frames = 67.320 tokens.

### 4.6 Qualidade
- KLD (teacher‖model, 51.175 posições, malaiwah): TR3 K6 0,013723 (254 GB); FP8 oficial cross-stack 0,020615 (328 GB); **EXL3 4 bpw 0,024555 (176 GB)**; FP8 oficial stack brandonmusic 0,024629; NVFP4 0,060535 (~180 GB).
- FP8 dense (`dense,kda`, 2026-09-08): KL proxy 0,002–0,013 nats/posição, argmax 94–100%.
- Estudo de 3 boots (2026-09-26): FP8=all e KDA BF16 "NOT-WORSE" em todos os itens; ΔNLL −0,004…+0,0005 nats/token.
- KDA BF16: RMSE relativo vs Marlin 0,002526 (W8A8 antigo 0,0266); M=65: KL 1,10× nulo, 0 flips (W8A8: 7,99×, 23 flips).
- Thin FAST: estudo selado "INCONCLUSIVE" (0 flips consensuais, KL médio dentro do nulo stock 0,0128).
- Cooperativo: não bit-exato (6.473.169 elementos brutos acima do limiar estrito), 18/18 no screen de 0,3% do pico.
- Thinking off em tabelas longas: 11/12 respostas corrompidas; `reasoning_effort: low` com thinking on: 0/6, 791/792 números corretos (#257).

### 4.7 Scheduler (fair v5, 2026-09-15)
- TP2 rep2k: B TTFT 11,6 s durante A (4 passos mistos, chunks 100/256/768); rep30k: 163,6 s, 39 passos mistos, 29.365 tokens; A 20,9/20,6 tok/s vs 24,7 solo; gaps p95 0,19 s, máx 1,39–1,44 s.
- TP3: rep2k 8,7 s; rep30k 110,0 s (~268 tok/s efetivos em mistura vs ~179 no TP2); gaps máx 0,84–0,95 s.
- v2 (histórico): cold ~30k B TTFT 36,3 s (não durante A); custo por passo com prefill ≈ 0,3 s fixo + ~0,68 ms/token.

### 4.8 Boot/carga
- Cold boot TP2 com InstantTensor: `/health` ~230 s; carga de pesos ~36 s; TP3 bundle: 324–372 s → ~55,5 s de carga de pesos.
- `patch_skip_cudagraph_profile`: −11 s de boot com `CG_ESTIMATE=0`.
- Draft H3 6 bpw: ~25 min no GB10 (inclui compilar a extensão); overlay denso ~2 min de download.

---

## 5. Conceitos necessários (em ordem de pré-requisito)

### Nível 0 — Fundamentos de ML/DL
1. **Rede neural como composição de camadas lineares + não-linearidades.** Uma camada linear é `y = x Wᵀ`; empilhar com ativações dá expressividade. Aqui praticamente todo custo de inferência são multiplicações matriz×vetor/matriz (`hgemm`, Marlin, `exl3_moe`), então entender GEMM é entender o desempenho.
2. **Representação numérica (FP32, FP16, BF16, FP8 e4m3/e5m2, int).** Formatos trocam faixa dinâmica por precisão e bytes. O repo usa BF16 para pesos densos, FP16 dentro dos kernels EXL3, FP32 em acumuladores e no estado KDA, e FP8 e4m3 (máx 448) no KV e nas projeções densas.
3. **Treino vs inferência.** Na inferência os pesos são fixos, então dá para quantizá-los, reempacotá-los e até editá-los na carga (FP8 per-channel, cópias BF16, abliteração) — tudo feito no `process_weights_after_loading`.
4. **Softmax, logits, amostragem (temperatura, top-k, top-p).** O LM produz logits sobre o vocabulário (154.880); a amostragem transforma em token. Aparece no Gumbel-max do seletor DFlash2, no `bad_words` (mascara logits) e no warmup do sampler (top_k/top_p mudam os kernels compilados).
5. **Divergência KL / NLL como métrica de fidelidade.** KL(teacher‖modelo) mede quanto a distribuição muda com quantização; o repo usa KLD 0,0246 como referência e "NOT-WORSE" em ΔNLL para aceitar FP8/BF16.

### Nível 1 — Transformer e LLMs
6. **Tokenização e embeddings.** Texto → ids → vetores de 4096. O vocab 154.880 é múltiplo de 64 (importa para o `lm_head` EXL3 e para o padding de vocab no TP3 = 154.944).
7. **Bloco Transformer (pré-norm, atenção, MLP, residual).** Cada uma das 45 camadas tem mistura de tokens (atenção/KDA) + MLP (densa ou MoE). Os patches mexem em cada peça separadamente.
8. **Atenção com Q/K/V e cabeças múltiplas; GQA.** `softmax(QKᵀ/√d)V` por cabeça; GQA compartilha K/V entre cabeças de Q. O drafter DFlash2 tem 32 Q / 8 KV; o FlashInfer exige `qo % kv == 0` (por isso 36/9 no TP3).
9. **Máscara causal vs bidirecional.** Decoders usam causal; o drafter DFlash2 é **não-causal dentro do bloco** (`is_causal: false`), e fixar um backend causal (`TRITON_ATTN`) destrói a aceitação.
10. **Posição: RoPE e NoPE.** RoPE injeta posição rotacionando Q/K; as camadas MLA do GLM-5.3-Flash são **NoPE** (`qk_rope_head_dim=0`). O kernel espera um bloco RoPE de 64 → o overlay usa zeros (exato, pois contribui 0 para QKᵀ).
11. **MLP com SwiGLU (gate/up/down) e limite de ativação.** `down(silu(gate(x)) ⊙ up(x))`; aqui com clamp `swiglu_limit=10`. Os kernels E2/E3/coop fundem gate+up e o SwiGLU no epílogo.
12. **Normalização (RMSNorm) e residual.** Presente no drafter (context KV: RMSNorm → GEMM KV → K-norm → RoPE) e nos blocos. Hyper-connections (mHC) generalizam o residual para vários fluxos.
13. **Hyper-connections / mHC.** Em vez de um único residual, o modelo mantém 4 fluxos e os mistura com matrizes aprendidas; para ler um hidden "normal" de 4096 é preciso `hc_post` + `hc_contract` — essencial para o hook EAGLE3 (**[interpretação]** "manifold-constrained Hyper-Connections").
14. **Geração autoregressiva: prefill vs decode.** Prefill processa o prompt inteiro (compute-bound, M grande); decode gera 1 token por passo (memory-bound, M pequeno). Quase todas as decisões do repo (E3 só no prefill, coop/FAST só no decode, KDA BF16 só em M>512, FP8 Marlin em M pequeno) vêm desse contraste.
15. **KV cache.** Guarda K/V (ou o latente MLA) de tokens passados para não recomputar. Aqui ele é híbrido: latente MLA FP8, pools do indexer, estado recorrente KDA, KV do drafter.
16. **Multi-Token Prediction (MTP).** Cabeça extra treinada para prever tokens à frente; o GLM traz um bloco MTP (camada 45) usado como drafter alternativo (`SPEC_METHOD=mtp`, k=2) e editado pelo ABLIT.
17. **Modelos de raciocínio e chat template.** Tokens `<think>…</think>`, `enable_thinking`, `reasoning_effort` (a palavra de esforço fica no caractere 39 do prompt: mudar por request quebra o prefix cache). Patches de stop dormente dependem disso.
18. **Multimodalidade (vision tower, placeholders de vídeo).** Imagens viram tokens (≈2048 com cap); vídeos viram frames com timestamps; o patch alinha placeholders a `grid_t`.

### Nível 2 — Arquiteturas do GLM-5.3-Flash
19. **Mixture of Experts (MoE) com roteamento top-k.** Um roteador escolhe 8 de 288 experts por token; só esses são computados. Carga desbalanceada (experts "quentes") é o motivo dos tiers fat E1–E3.
20. **Shared expert.** Um expert sempre ativo somado aos roteados; fica BF16 (ou FP8 com `shared`), e no TP3 recebe pad 2048→2112.
21. **Camadas densas iniciais.** Camadas 0–2 têm MLP denso (12288); grupo `dense` do FP8.
22. **Multi-head Latent Attention (MLA).** Comprime K/V num latente de 512 (`kv_lora_rank`) e reconstrói por cabeça via `kv_b_proj`; na inferência a "absorção" multiplica esse peso direto nas queries — por isso `kv_b_proj` nunca é quantizado/FP8/EXL3.
23. **Atenção esparsa com indexer (estilo DeepSeek Sparse Attention / NSA).** Um indexer barato pontua o histórico e seleciona top-k tokens (2048) para a atenção cheia; aqui o indexer trabalha em "pools" de 4 tokens (`index_kpool=4`) e há uma cauda de tokens ainda não comprimidos (KpoolTail). Os patches M2/M30/M31/M32 corrigem esse subsistema.
24. **Atenção linear / regra delta (KDA).** Substitui softmax por um estado recorrente `S ∈ R^{d×d}` atualizado por token com gate (decay) e taxa `beta` (delta rule), custo O(1) por token no decode e "chunked scan" no prefill. No vLLM o estado vive em `MambaSpec` (4 grupos); **[interpretação]** KDA = Kimi Delta Attention, família Gated DeltaNet.
25. **Modelo híbrido (34 KDA + 11 MLA esparsas).** A maior parte das camadas é linear (barata em contexto longo), algumas são atenção esparsa global. Isso cria um KV cache com grupos heterogêneos e requisitos de alinhamento de checkpoints de estado.

### Nível 3 — Quantização
26. **Quantização weight-only vs weight+activation.** Só pesos (W4A16/W8A16) mantêm ativações em BF16/FP16; quantizar ativações (W8A8) acelera mas adiciona erro — o repo mediu que o W8A8 do KDA mudava probabilidades e o descartou.
27. **Granularidade de escala (per-tensor, per-channel, per-group/tile).** O FP8 denso usa uma escala por canal de saída (`amax/448`); o KV `fp8_ds_mla` usa escalas por tile (16 B por token); EXL3 usa `suh/svh`.
28. **Bits por peso (bpw) e tamanho do modelo.** 4 bpw nos experts → ~164 GiB; 6 bpw K6 → 254 GB com KLD 0,0137; o compromisso bpw × KLD × bytes decide o checkpoint.
29. **Incoerência por rotação (Hadamard).** Multiplicar W por matrizes ortogonais aleatorizadas (Hadamard com sinais) espalha outliers e torna os pesos "gaussianos", ideal para quantização vetorial. Em EXL3: `W ≈ diag(suh)·H·Ŵ·H·diag(svh)` em blocos de 128; na execução, a ativação é rotacionada (`had(x⊙suh)`), multiplicada pelo núcleo quantizado e a saída é rotacionada e escalada (`had(y)⊙svh`).
30. **Quantização por treliça (trellis/TCQ, QTIP) e EXL3.** Cada peso consome K bits de um bitstream; o "estado" de 16 bits de cada peso é a janela deslizante do bitstream (bitshift trellis), decodificada por um **codebook procedural** (MCG: multiplica por `0xCBAC1FED`, operação `lop3` e soma de 2 fp16 → valor ~gaussiano) sem tabela em memória. Tiles de 16×16 = 256 pesos = 16·K palavras int16. Entender isso explica `trellis/suh/svh/mcg`, `k_words = 64` e os kernels que dequantizam em registradores.
31. **Codebooks mcg vs mul1 vs 3INST.** Três famílias de decodificação procedural; experts usam MCG (cb=1), os linears densos da turboderp usam mul1 (cb=2). Kernels especializados (E2/E3/coop) só suportam K4 MCG.
32. **FP8 e4m3 e kernel Marlin.** e4m3 tem 3 bits de mantissa e máximo 448; Marlin é um kernel de GEMM mixed-input (pesos em baixa precisão, ativações BF16/FP16) muito eficiente em M pequeno; perde para cuBLAS BF16 em M grande (crossover medido em 512).
33. **NVFP4 (contraste).** Formato FP4 com escalas por bloco da NVIDIA; mencionado como pior em KLD (0,0605) e sem kernel sparse-MLA no SM12x — motivo de não usá-lo aqui.
34. **Quantização do KV cache.** FP8 no KV reduz memória por token (656 B vs ~1 KiB em BF16 com o pad); o drafter fica em BF16 porque não há backend FP8 adequado para ele no SM121.
35. **Quantização sem calibração (Hessiano sintético).** O drafter DFlash2 é quantizado para 6 bpw sem dados de calibração (`uncalibrated_quantize`), mantendo K/V e o seletor em BF16 por sensibilidade.
36. **Métricas de qualidade de quantização.** KLD por posição, flips de argmax, RMSE relativo por camada, "NOT-WORSE" vs controle stock; o repo exige controles stock-vs-stock para calibrar os limiares.

### Nível 4 — GPU e kernels
37. **Hierarquia de execução CUDA (grid, bloco/CTA, warp, lane).** Kernels do repo usam 256 threads (8 warps), um warp por bloco de 16 colunas; shuffles entre lanes fazem Hadamard.
38. **Tensor cores e `mma.sync m16n8k16`.** Unidades de MMA em fragmentos; SM121 (Blackwell de consumo) usa `mma.sync`, não `tcgen05` do Blackwell de datacenter — por isso reutilizar o caminho do ExLlamaV3.
39. **Memória compartilhada, swizzle e `ldmatrix`.** O swizzle XOR evita conflitos de banco; `cp.async` com múltiplos estágios esconde latência de memória (mainloop do E3).
40. **Registradores, occupancy e spills.** O kernel FAST ganha 8–17% por eliminar spills (LDL 39 → 0) trocando estágios de fragmentos por estágios de SMEM.
41. **Atomics (incluindo vetoriais de 16 B).** O E3 acumula a saída do down com `atomicAdd(float4)` (sm_90+), permitindo vários experts escreverem no mesmo token.
42. **Fusão de kernels e epílogos.** Juntar SwiGLU, Hadamards e escalas no epílogo evita idas e voltas à memória (E2/E3/coop).
43. **Host-device sync e por que ele mata desempenho/graphs.** `.item()`, `.tolist()`, `nonzero()` param a CPU esperando a GPU; o E3 substitui isso por tabelas no device.
44. **CUDA Graphs (captura, replay, FULL vs PIECEWISE, capture sizes).** Graphs eliminam overhead de lançamento no decode, mas exigem formas fixas, sem alocação e sem sync; daí `capture sizes`, batch uniforme no adaptive-k, pinagem do `expert_map`, warmup do autotuner antes da captura, e a regra de que todos os ranks capturem os mesmos graphs.
45. **Autotuning de kernels.** Escolhe tiles na primeira execução; no SM121 o autotune do FlashInfer derruba o rank 0 e o do ExLlamaV3 sincroniza dentro da captura — ambos contornados.
46. **Grouped GEMM e tabelas de segmento.** Várias GEMMs de tamanhos diferentes num único lançamento, descritas por offsets/prefix-sums; base do E3 e do MoE cooperativo.
47. **Roofline, intensidade aritmética, memory-bound vs compute-bound.** A 1024 tokens por chunk os experts ficam memory-bound (~112 FLOP/B < ridge 458); explica por que chunks maiores e dequant direto ajudam.
48. **Triton e interpretador Triton.** Vários kernels vLLM (slot mapping, kpool seed, walk do seletor) são Triton; os testes rodam o kernel de seed em CPU com `TRITON_INTERPRET=1`.
49. **Compilação para arquitetura (`sm_121a`, `TORCH_CUDA_ARCH_LIST=12.1a`, fast-math).** Cubins nativos; o E3 é compilado sem `--use_fast_math` para reproduzir o arredondamento do `torch.sigmoid`.
50. **PDL (Programmatic Dependent Launch).** Sobreposição entre kernels dependentes; desativado no SM12x por race com kernels de estado KDA.

### Nível 5 — Motor de inferência (vLLM)
51. **Arquitetura do vLLM V1 (API server, EngineCore, workers, executor `mp`).** O servidor OpenAI recebe requests, o EngineCore agenda e os workers executam; comunicação por shared memory (`SpinCondition`, `shm_broadcast`). Patches atuam em cada nível.
52. **Registry de modelos e de métodos de quantização; `QuantizationConfig`/`LinearMethodBase`/`FusedMoEMethodBase`.** Pontos de extensão que o `exl3.py` implementa (`create_weights`, `process_weights_after_loading`, `apply`).
53. **Weight loading (safetensors, mmap, InstantTensor, weight_loader por parâmetro).** Cada parâmetro tem um `weight_loader` que fatia por TP; o EXL3 define os seus; InstantTensor usa O_DIRECT com ring buffer pinado.
54. **PagedAttention e block tables / slot mapping.** O KV é dividido em blocos; cada request tem uma tabela de blocos e cada token um slot. Bugs de slot mapping (KpoolTail) corrompem outros requests.
55. **Hybrid KV cache manager (grupos de KV, `KVCacheSpec`: FullAttention/MLAAttention/SlidingWindow/Mamba/KpoolTail, `UniformTypeKVCacheSpecs`).** Grupos com block sizes diferentes compartilham um `BlockPool` com ids globais; `page_size_padded` e slot-sharing deixam camadas diferentes ocuparem o mesmo tensor em ids disjuntos.
56. **Manager block size vs kernel block size; alinhamento/LCM.** O manager usa 3584 (MLA/Mamba), o kernel 64; o scheduler alinha checkpoints ao LCM; dividir uma página padded em kernel blocks dá OOB.
57. **Automatic Prefix Caching (APC): hashing de blocos, `find_longest_cache_hit`, coordinator híbrido, LRU/free queue, null block.** Hits só valem se todos os grupos (exceto os isentos) tiverem o prefixo; o drafter e o KpoolTail precisam de regras especiais; blocos hashados vão para o fim da fila livre.
58. **Retenção esparsa de blocos (`retention_interval`, `reachable_block_mask`).** Guardar só alguns checkpoints do prefixo economiza ids mas perde hits de edição/branch tardios.
59. **Mamba "align" mode e checkpoints de estado recorrente.** O estado KDA só pode ser reutilizado em posições alinhadas a blocos; o scheduler tem que parar chunks nesses limites e liberar estados antigos com segurança sob scheduling assíncrono.
60. **Scheduler: continuous batching, chunked prefill, `max_num_batched_tokens`, `max_num_seqs`, `long_prefill_token_threshold`, async scheduling.** Define quanto de prefill e decode entra em cada passo; é onde o "fair" v5 atua.
61. **Mixed batches e uniform-batch FULL graphs.** Misturar prefill num passo de decode tira o passo do caminho de graph uniforme e o torna segundos mais lento neste backend — raiz do problema de fairness.
62. **Speculative decoding: drafter, verificação, rejection sampling, aceitação por posição, tokens por passo.** O drafter propõe k tokens, o alvo verifica todos em um passo; o ganho depende da aceitação. Base para DFlash2, MTP, adaptive-k e o canário do boot.
63. **EAGLE / EAGLE-3.** Drafters que usam hidden states do alvo (de várias camadas no EAGLE-3) como entrada; o vLLM trata `dflash` como `use_eagle()`, o que afeta o APC (bloco lookahead).
64. **DFlash / DFlash2 (drafter de difusão em bloco).** Gera um bloco de 8 posições em paralelo com atenção bidirecional no bloco, KV de contexto projetado dos hidden do alvo e janela deslizante de 2048; o DFlash2 adiciona conv dinâmica agrupada e um seletor de candidatos com Gumbel walk.
65. **Gumbel-max trick.** `argmax(logits/T + Gumbel)` amostra da softmax; a temperatura 0 vira argmax; usado no walk do seletor.
66. **Structured output (XGrammar, FSM).** Gramáticas mascaram tokens inválidos; com spec decode é preciso validar drafts e parar na terminação (backports M40).
67. **Tool calling e reasoning parsers (`glm47`, `glm45`).** Parsers extraem tool calls e raciocínio do texto; `adjust_request` do parser é o hook para mascarar `<tool_call>`.
68. **API OpenAI-compatível (chat completions, stateless, `vllm_xargs`, `/tokenize`, métricas).** O cliente reenvia o histórico inteiro; limites (imagens) são por request; campos extras passam por `vllm_xargs`.
69. **Workspace manager e buffers pré-alocados.** Buffers como o workspace do indexer e o do FlashKDA são reservados no profiling (saem do KV) e reutilizados — dimensioná-los certo devolve GiB ao KV.
70. **Profiling de memória e `gpu_memory_utilization` / `--kv-cache-memory-bytes`.** O vLLM mede o pico com um dummy run e dá o resto ao KV; em UMA isso compete com o sistema, por isso a reserva explícita.
71. **Custom ops (`direct_register_custom_op`) e torch.compile.** Tornam chamadas externas opacas ao compilador e capturáveis (dense EXL3).

### Nível 6 — Inferência distribuída
72. **Tensor parallelism (column/row parallel, all-reduce).** Gate/up são column-parallel (cada rank tem metade das colunas), down é row-parallel (metade das linhas) e um all-reduce por camada soma os resultados; o `o_proj` row-parallel permite editar linhas sem coletivas.
73. **Requisitos de divisibilidade no TP.** Cabeças, vocab, intermediários precisam dividir pelo número de ranks; TP3 força pads (64→66, vocab 192, shared 2112) e EP para os experts.
74. **Expert parallelism.** Cada rank guarda experts inteiros (96 de 288 no TP3) em vez de fatias; o loader não pode fatiar de novo.
75. **Data parallel do encoder de visão (`--mm-encoder-tp-mode data`).** A torre de visão roda replicada/por dados quando não divide.
76. **Multi-node com `--nnodes`, `--headless`, executor `mp`.** Rank 0 no head com a API; ranks headless nos workers.
77. **NCCL sobre RoCE (HCAs, GID index, socket ifname, cross-NIC, ring dirigido).** Configuração por nó das interfaces CX7; GID errado mata o rank (`ibv_modify_qp errno 61`); no TP3 o ring tem que ser dirigido porque o NCCL pareia índices de NIC.
78. **Identidade de configuração entre ranks.** Todos os ranks precisam dos mesmos patches, flags e capture sizes; divergência trava o NCCL após o PIECEWISE.
79. **Stalls distribuídos e mitigação (TP4).** Um rank preso num kernel trava todos; o slicing do sparse-MLA limita a chamada onde o progresso parava.
80. **Compartilhamento de pesos (NFS sobre ConnectX).** Workers montam o cache HF do head read-only para não duplicar 164 GiB.

### Nível 7 — Sistema/memória (UMA GB10)
81. **Memória unificada (UMA) CPU+GPU.** GPU e CPU dividem os 128 GB; `mem_get_info` reporta `MemFree` do host; page cache conta como usado; cada GiB de KV é um GiB a menos para o SO.
82. **Page cache, `MemFree` vs `MemAvailable`, `drop_caches`, `posix_fadvise`.** Páginas limpas são reclamáveis; o loader dimensiona contra `MemAvailable` e dá `DONTNEED` nos shards lidos.
83. **Tamanho de página do kernel (4 KiB vs 64 KiB) e mmap file-backed.** No kernel 64 KiB, copiar H2D direto de mmap de arquivo trava o driver → clonar para memória anônima.
84. **Swap, swappiness e livelock UVM.** Paging de memória do vLLM pode travar o driver (96% util a 12–20 W, `/health` continua 200); prevenção com `vm.swappiness=0` e ciclar swap antes do start.
85. **Pinned memory e streams laterais.** Cópias D2H assíncronas (E1) e ring buffers do InstantTensor.

### Nível 8 — Alinhamento/edição de pesos
86. **Direção de recusa e representation engineering.** Comportamentos como recusar podem corresponder a uma direção `r` no espaço residual; removê-la dos pesos que escrevem no residual altera o comportamento.
87. **Ortogonalização de pesos (abliteração por projeção).** `W' = (I − α r rᵀ/‖r‖²) W` remove (α=1) ou inverte (α>1) a componente em r da saída do `o_proj`; aplicado por linhas, idêntico em todos os ranks.
88. **Transplante de pesos.** Copiar tensores já editados de um checkpoint doador com a mesma arquitetura (o `o_proj` BF16 é byte-idêntico entre o NVFP4 e o EXL3), reproduzindo uma edição publicada exatamente.
89. **Âncoras de segurança (camadas iniciais intocadas) e MTP.** Editar só 15–45 e também o MTP (um MTP stock continuava propondo recusas via spec decode).

### Nível 9 — Metodologia de medição
90. **A/B/A com controle de deriva e bootstrap CI.** Toda mudança é medida com braços intercalados e reinícios frescos; o repo rejeita alegações quando o A2-vs-A (ruído) é grande.
91. **Prefill tok/s = prompt_tokens/TTFT com salt único.** Evita que o APC "trapaceie".
92. **Métricas de decode: tok/s por stream vs agregado, ms por ciclo de draft, aceitação.** Separar aceitação de velocidade de kernel.
93. **Canários de sanidade pós-boot.** "Reply with OK." e aceitação de drafts > 0 detectam motores degenerados que ainda respondem `/health`.
94. **Patches fail-closed, idempotentes e ancorados.** Cada overlay verifica âncoras exatas, reaplica sem efeito e recusa drift — condição para manter dezenas de patches sobre um vLLM pinado.

---

## 6. Receita passo a passo (como recriar do zero e o porquê de cada decisão)

### Passo 1 — Hardware e rede
1. Dois GB10 com o mesmo SO; cabo QSFP CX7 entre eles; IPs de fabric por nó; passwordless SSH head → worker.
2. Descobrir nomes de interface/HCA por nó (`HEAD_CX7_IF/IB`, `WORKER_CX7_IF/IB`) e o GID RoCEv2 correto de cada NIC (`NCCL_IB_GID_INDEX` ou `HEAD_GID/WORKER_GID`). *Por quê:* NCCL não usa os aliases 10.0.0.x; GID zerado derruba o rank em ~60 s.
3. Sistema: `vm.swappiness=0`, manter swap, ciclar swap antes de iniciar, liberar page cache antes do `docker run` (`GLM53_HOST_MEM_HYGIENE=1`). *Por quê:* UMA — page cache e swap afetam diretamente a "memória de GPU" e podem causar livelock UVM.
4. (TP3) Ring dirigido Port0 → Port1 do próximo nó; bootstrap na LAN de gerência (`SOCKET_IFNAME`), dados via `NCCL_IB_HCA`, `NCCL_CROSS_NIC=1`.

### Passo 2 — Imagem
1. Base `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (vLLM `487ecf187`). *Por quê:* todos os patches são ancorados byte a byte nesse pin.
2. Patches de backend (M1–M4): NoPE → GLM_NSA 576 com RoPE zero; `supports_dense_mha_prefill=False`; top-k como capacidade com contagens válidas; descartar 1 pool (511×4+3 ≤ 2048); kernel block `[64]`; alinhamento do indexer `index_kpool×64` no SM12x; desligar autotune do FlashInfer e PDL no SM12x.
3. Registrar `exl3` (Literal + import lazy + overrides do `ModelConfig`) e copiar `overlay/exl3.py` para `model_executor/layers/quantization/exl3.py`.
4. Compilar `exllamav3@c5d9c657` com `TORCH_CUDA_ARCH_LIST=12.1a`, stubs AVX (M6), fat kernels E2/E3 (`patch_exl3_fat_kernel.py`) e o pipeline FAST (`patch_exl3_decode_pipeline.py`); verificar símbolos. *Por quê:* cubins nativos `sm_121a`; kernels aditivos (os stock ficam intactos).
5. Instalar DFlash2 (`patch_dflash2.py`, `patch_dflash2_exl3.py`), hooks EAGLE3 (`patch_glm_eagle3.py`), grupo do drafter (`patch_glm5_drafter_group.py`) e o resto dos overlays na ordem do `Dockerfile`/`GLM53_OVERLAY_ORDER`; rodar os testes de CPU embutidos (`test_exl3_overlay.py` etc.).
6. `pip install --no-deps instanttensor==0.2.0` (sem deps: não trocar o NCCL 2.30.7), `patch_cold_load_uma.py`, `patch_skip_cudagraph_profile.py`.
7. Carimbo `glm53.recipe.stamp` com hash do overlay/Dockerfile para rebuild automático quando mudar.

### Passo 3 — Pesos
1. Baixar `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (120 shards, ~164 GiB) e o drafter `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c` no head.
2. Compartilhar com o worker por NFSv4 sobre o ConnectX (`NFS_SHARE=1`, nunca pelo link de gerência) ou rsync.
3. Opcional: ABLIT (`python3 ablit/fetch_transplant.py`, `ABLIT=1`) ou checkpoint pré-editado (`start-abliterated.sh`); opcional: par dense EXL3 H3 (`GLM53_MODEL_PRESET=dense-h3`: `tools/dense_overlay.py` + `tools/dflash2_exl3_quant.sh`).
*Por quê:* só os experts roteados são EXL3; todo o resto BF16 permite FP8/ABLIT na carga.

### Passo 4 — Aplicação dos overlays no start (ambos os ranks, mesma ordem)
Ordem (`start.sh:2053-2082`): cold_load_uma → skip_cudagraph_profile → video_placeholders → suppress_stops → scheduler_decode_floor (v5) → mamba_align_chunking → glm5_drafter_group → hybrid_prefix_hit → apc_per_group_retention → apc_no_store → mamba_align_state_free → kv_capacity_log → tool_choice_none → xgrammar_termination → kpool_tail_slotmap → kpool_tail_seed_stride → spinwait → adaptive_k → dense_fp8 → dflash2_exl3 → loadclone → default_max_new_tokens → indexer_workspace → cache_reset → ablit. *Por quê:* vários patches compartilham arquivos/âncoras (coordinator, `kv_cache_utils.py`, `scheduler.py`); a ordem garante composição e o resultado byte-idêntico a uma imagem construída do zero.

### Passo 5 — Argumentos de lançamento (TP2)
- Distribuído: `--tensor-parallel-size 2 --nnodes 2 --node-rank {0,1} --master-addr HEAD_IP --distributed-executor-backend mp` (worker `--headless`), NCCL da imagem.
- Modelo: `--quantization exl3`, `--kv-cache-dtype fp8` (→ `fp8_ds_mla`), `--max-model-len 850000`, `--gpu-memory-utilization 0.85`, `--kv-cache-memory-bytes 11811160064`, `--max-num-seqs 4`, `--max-num-batched-tokens 7168`, `--enable-prefix-caching`, `--no-enable-flashinfer-autotune`, `--load-format instanttensor`, `--served-model-name GLM-5.3-Flash-EXL3`.
- Spec: `--speculative-config '{"method":"dflash","model":…,"num_speculative_tokens":7,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":2}'`; não fixar backend de atenção do drafter.
- Parsers/template: `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --chat-template /opt/glm53/chat_template.jinja`.
- Multimodal: `--limit-mm-per-prompt {"image":48,"video":1}`, `--mm-processor-kwargs {"max_image_tokens":2048}`, `--mm-processor-cache-gb 1`, `--skip-mm-profiling`.
- Env: `EXL3_FUSED_MOE=1 EXL3_FAT_KERNEL=1 EXL3_FAT_GROUPED=1 EXL3_TEMP_ROWS_FUSED=32`, `GLM53_DENSE_FP8=all`, `GLM53_KDA_BF16_LARGE_M=1`, `GLM53_INDEXER_WORKSPACE=rightsize`, `GLM53_DRAFT_KV_COMPACT=1`, `GLM53_MIXED_PREFILL_CHUNK=fair` (+ knobs fair), `GLM53_APC_NO_STORE=1`, `GLM53_KV_CAPACITY_LOG=1`, `DEFAULT_MAX_NEW_TOKENS=65536`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; opcionais `GLM53_ADAPTIVE_K=ema` (com capture sizes múltiplos de 3/5/8), `GLM53_EXL3_MOE_FAST=1`, `GLM53_SPINWAIT_MS=16`, overlay cooperativo via `EXL3_OVERLAY_HOST`.
*Por quê de cada um:* `fp8_ds_mla` é o único KV com kernel esparso; 11 GiB de KV foi dimensionado para compact pages e paga a cópia BF16 do KDA; MNBT 7168 com E3 dá o melhor prefill; `temp_rows=32` mantém o decode num lançamento; `fair` evita a inanição sem colapsar o incumbente.

### Passo 6 — Verificação
1. Logs por rank: `exl3 e2 diag … effective_tier=grouped`, `EXL3 MCG trellis engaged … fused_moe=exl3_moe`, `dense fp8 groups: all`, 34 linhas `kda bf16-large-m retained`, `padded slot-share block=…`, `hybrid APC groups … eagle_group_ids=[6]`, `[glm53-dflash-boundary-lookup-v1] boundary_group_ids=[6]`, `[glm53-decode-floor] fair v5 …`, `[glm53-kv-capacity-log] …`, `GPU KV cache size: …` (≥ 1× o `max_model_len`), `[glm53-adaptive-k] uniform decode graph query lens: [3, 5, 8]` (se ligado), `ABLIT_METHOD=auto -> transplant` (se ABLIT).
2. Warmup + canário (`scripts/boot-shape-warmup.sh`): OK a temperatura 0 e aceitação de drafts > 0.
3. `scripts/spec-accept-gate.sh`: curva de aceitação decaindo (pos0 em exatamente 1,00 é o bug vllm#53030).
4. Benchmarks: `tests/bench_decode.py --phase structured --structured` e `--phase prose`; prefill com salt único (`docs/_run_cold_prefill.py`); `tests/bench_prefix_cache.py`; `tests/bench_concurrency.py`.
5. Monitorar `MemAvailable` do head sob carga (≥ ~5 GiB) e a detecção de livelock via `/metrics` (requests rodando + contadores congelados).

### Passo 7 — Tuning (uma variável por vez, A/B/A)
- Contexto × KV × util: subir `MAX_MODEL_LEN` exige subir o pool; 15 GiB deixou só 0,8–2,2 GiB livres; não baixar o contexto para "liberar" slots.
- Prefill: MNBT (2048 foi melhor pré-E2 e com 16 seqs; 7168 com E3/4 seqs); E3 on; KDA BF16; FlashKDA (TP3, experimental).
- Decode: adaptive-k para prosa; FP8 `all`; FAST; cooperativo geo1; spinwait 16.
- Concorrência: `fair` share 0,20–0,30 conforme o quanto o incumbente pode desacelerar; `MAX_STEP_MS` 2000.
- APC: retenção global/SWA só se o workload for de históricos independentes sem edições tardias; no-store para lanes de batch.
- TP3: `--hf-overrides` 66 cabeças, EP, drafter TP=1 com pad 36/9, KV 35–40 GiB, `GLM53_DENSE_FP8=dense,kda`; TP4: `DFLASH_TOKENS=3`, `MAX_NUM_SEQS=8`, util 0,75, mixed `off`, slicing do sparse-MLA.

---

## 7. Incertezas e observações honestas

- A sigla **KDA** não é expandida no repositório; a leitura "Kimi Delta Attention / Gated DeltaNet" vem dos nomes de kernels (`chunk_kda_*`, `GDNAttentionMetadata`), dos parâmetros (`A_log`, `dt_bias`, `beta`, gate limitado) e da menção ao Kimi-K3 no patch FlashKDA. Idem para **mHC** (o repo usa só `hc_post`/`hc_contract`/4 fluxos).
- O oráculo `kernel_lab/exl3/reference.py` modela os "estados" do trellis com `unpack_trellis` + `decode_mcg`; os kernels CUDA (`exl3_dq.cuh`) leem uma janela de 16 bits do bitstream por peso (bitshift trellis). Não validei se o oráculo implementa a janela deslizante ou uma simplificação — o doc afirma equivalência com o CUDA nos 65.536 estados e no empacotamento K3.
- Valores de `selector_rank`, `selector_top_k`, `conv_kernel_size`, `conv_group_size` do DFlash2 vêm do `config.json` do drafter no Hub e não estão no repo.
- Muitas medições são de execuções únicas ou em configurações históricas (o próprio repo marca: "historical", "not a matched A/B", "descriptive"); a tabela 4 mantém as datas para isso.
- O `.env.example` diverge em alguns defaults do código (ex.: `GLM53_MIXED_PREFILL_CHUNK` código `skip` vs launchers `fair`; `EXL3_FAT_GROUPED` código `0` vs `.env` `1`; `GLM53_INDEXER_WORKSPACE` unset = `stock` vs `.env` `rightsize`); o que vale em produção é o que o launcher exporta.

---

## Apêndice A — Contas que explicam os números (derivações verificadas contra valores do repo)

**A.1 Forma do `in_proj_qkvbfg_a` do KDA (TP-local).** O repo dá `[12576 × 4096]` em TP2 e `[8726 × 4096]` em TP3 (`exl3.py:2019-2022`). Decomposição consistente (minha derivação, **[interpretação]** dos sufixos q/k/v/b/f/g): TP2 → 32 cabeças locais × 128 × 3 (q,k,v) = 12.288 + `b` (beta, 1 por cabeça) 32 + `f_a` 128 + `g_a` 128 (projeções de baixo rank dos gates, replicadas) = **12.576**. TP3 → 22 × 128 × 3 = 8.448 + 22 + 128 + 128 = **8.726**. Isso também explica por que no TP3 `f_a`/`g_a` são "views de 128 colunas de uma projeção fundida de 8.726" com pitch não múltiplo de 8 (o Marlin recusa) e por que `f_b_proj`/`g_b_proj` (128 → cabeças×128) ficam BF16 no TP3.

**A.2 Custo da cópia BF16 do KDA.** 12.576 × 4.096 × 2 B = 103.022.592 B = **98,25 MiB/camada/rank** × 34 camadas KDA = **3,26 GiB/rank** (medido +3,29 GiB). TP3: 8.726 × 4.096 × 2 = **68,17 MiB** → ~2,26 GiB.

**A.3 Tile EXL3 K4.** 16×16 = 256 pesos × 4 bits = 1.024 bits = **64 palavras int16** (`k_words = bits*16`). Trellis de um gate/up de expert em TP2: `[4096/16, 1024/16, 64]` = `[256, 64, 64]` int16 = 2 MiB; down `[64, 256, 64]` = 2 MiB. bpw efetivo inclui `suh`/`svh`: `8·(trellis_bytes + 2(K+N) + 4)/(K·N)` (`kernel_lab/exl3/abi.py:79-84`).

**A.4 Scratch do E3.** `cap = MNBT × topk = 7168 × 8 = 57.344` linhas; `h13 = 57.344 × 4.096 × 2 B = 448 MiB`; `h2 = 57.344 × 1.024 × 2 B = 112 MiB` → **560 MiB**, exatamente o valor que o README diz ser cobrado do orçamento de KV.

**A.5 Página MLA e página do drafter.** `3584 × 656 B = 2.351.104 B` (= `mla_page` do log). Drafter: 8 KV heads × 128 × 2 (K,V) × 2 B = 4.096 B/token; em TP2 → **2.048 B/token/camada**. Compacto: maior múltiplo de 64 que divide 3584 e cabe em 2.351.104/2.048 = 1.148 tokens → **896** (1024 não divide 3584 = 2⁹·7).

**A.6 Demanda de ids do drafter por request.** Exemplo do README (janela 2048 + 2048 tokens in-flight): bloco 64 → `cdiv(2047 + 2048, 64) + 1 = 65`; bloco 896 → `cdiv(4095, 896) + 1 = 6`. Na conta de `DESIGN-kv-capacity-log.md` §2 (in-flight = `max_concurrent_batches` 2 × 2048 = 4096, request de 1M): `cdiv(2047 + 4096, 64) + 1 = 97` blocos. Um hit EAGLE precisa de `cdiv(2047, 64) + 1 = 33` blocos contíguos → 33 ids por segmento de 3584.

**A.7 "GPU KV cache size".** Blocos por request de 1M: MLA `cdiv(1e6, 3584) = 280`; KpoolTail 1; Mamba `2 + 7` = 9 × 4 grupos = 36; drafter 97 → **414**. `643 / 414 = 1,553` → "1.553.140 tokens, 1,55×" — mas só 642 ids usáveis / 38 por segmento = 16 segmentos = **57.344 tokens** de conversa cacheada.

**A.8 Workspace do indexer.** Stock: `1e6 × 40 × 132 B = 5.280.000.000 B` + 1 MiB de radix top-k = **5036,40 MiB**. Rightsize com 4 seqs e k=7: `4 × cdiv(1.000.007, 4) = 1.000.008` entradas × 132 B = **125,9 MiB**.

**A.9 Modelo de custo do fair v5.** Medido: cada passo com prefill ≈ 0,3 s fixo + ~0,68 ms/token. Com `MAX_STEP_MS=1000`: 0,3 + 0,00068 × 1024 ≈ 0,996 s → degrau **1024**; com 2000 ms: 0,3 + 0,00068 × 2048 ≈ 1,69 s → **2048** (bate com "1000 ms already selected 1024; 2000 ms allows 2048"). O v4 estimava 1024 a partir de amostras de 128 tokens linearmente (0,34 s × 8 ≈ 2,7 s), ~3× acima do real.

**A.10 Top-k do indexer.** `select_k = 512` pools × `index_kpool = 4` = 2048 + cauda ≤ 3 = 2051 > 2048 (largura de template do kernel) → overlay usa 511 pools: 2044 + 3 = **2047**.

**A.11 FP8 per-channel.** `s_n = max_k |W_nk| / 448`; `Ŵ_nk = round_e4m3(W_nk / s_n)`; o Marlin calcula `y = x · (Ŵ ⊙ s)ᵀ` com x em BF16. A cópia BF16 large-M é exatamente `Ŵ ⊙ s_stored` (s arredondado em BF16), então os dois caminhos implementam o mesmo peso lógico.

**A.12 Reconstrução EXL3 (forma de execução).** Para uma camada `y = x W` com `W ∈ R^{K×N}`: `x' = had128(x ⊙ suh)` (Hadamard por blocos de 128 com escala 1/√128), `y' = x' · Ŵ` (Ŵ decodificado tile a tile do trellis pelo codebook MCG, em registradores), `y = had128(y') ⊙ svh`. No MoE, gate e up compartilham `suh` (uma única rotação da entrada), e a entrada da down já sai rotacionada e escalada (`⊙ down_suh`) do epílogo do gate/up.

**A.13 Ablit por projeção.** `rw = rᵀ W` (vetor na dimensão de entrada local do rank), `W ← W − α · r ⊗ rw / (rᵀr)`; como `o_proj` é row-parallel, `W` local tem todas as 4096 linhas e uma fatia das colunas — o termo `r ⊗ rw` é calculado com a fatia local e é consistente entre ranks.

---

## Apêndice B — Mapa "conceito → onde aparece" (para a plataforma didática)

| Conceito | Onde aparece no GLM |
|---|---|
| Quantização por treliça (EXL3/QTIP), codebook MCG, Hadamard | `overlay/exl3.py` (M7), `exl3_fat_*.cu` (M9/M10), `vendor/.../quant/*.cuh`, `kernel_lab/exl3/reference.py` |
| MoE top-k, experts quentes, grouped GEMM | M8, M10, cooperativo M15/M16 |
| Prefill vs decode (compute vs memory bound) | `docs/improve-prefill.md`, M8 (E3 só no prefill), M11/M15 (só decode), M13 (M>512) |
| FP8/Marlin, crossover por M | M12, M13 |
| MLA + NoPE + absorção | M1, M12 (`kv_b_proj` intocado) |
| Atenção esparsa com indexer (DSA/NSA), top-k | M1, M2, M3, M30, M31, M32 |
| Atenção linear KDA, estado recorrente, FlashKDA | M13, M27, M28, M45 |
| KV cache paginado, grupos híbridos, slot-sharing | M23, M29 |
| Prefix caching (APC), LRU, retenção | M24, M25, M26 |
| Speculative decoding, EAGLE3, DFlash2, Gumbel | M17–M22 |
| CUDA graphs e uniform batch | M8 (graph-safe), M14 (warmup do autotune), M22 (query lens), M33 (mixed steps) |
| Scheduler / chunked prefill / fairness | M33, M28 |
| Tensor/expert parallel, divisibilidade | M7 (col/row shard), M44 (TP3 pads + EP), M47 (row-parallel o_proj) |
| UMA, page cache, mmap, InstantTensor | M34, M35 |
| Abliteração/direção de recusa | M47 |
| Metodologia de benchmark e qualificação | `docs/*` (A/B/A, KLD, NOT-WORSE, canário M50) |

---

## Addendum 2026-10-30

Este inventário descreve o checkout de 2026-09-28 (HEAD `94ae731`). O que mudou depois está verificado em
[`mia-updates-2026-10-30.md`](mia-updates-2026-10-30.md); para o escopo deste arquivo (overlays, kernels,
checkpoints), lido no HEAD `a8cc40d` (2026-10-13):

- **`patch_scheduler_decode_floor.py` v5 → v7** (`b94fdfd`, #283, merge 2026-10-01): junta #246 (uma requisição
  recusada por KV não trava mais um prefill executável), #221 (prioridade explícita sob
  `--scheduling-policy priority`; FCFS inalterado) e #180 (migração fail-closed: pristine, v1, v2, v5 e v5+priority
  viram os mesmos bytes de uma instalação v7; v3/v4 e marcadores desconhecidos são recusados). Mantém o ladder
  `(128 … 2048)` e o ajuste `a + b·n` do v5. Qualificado só em TP2 (baseline → candidato → baseline, 2026-09-30):
  PASS em progresso de admissão e canário de contenção; decode/concorrência inconclusivos (baselines divergentes);
  limite conhecido de ~85 s sob `priority`. `patch_mamba_align_chunking.py` passa a aceitar v5 e v7.
- **Novo overlay `patch_indexer_warmup_range.py`** (#203, merge `b356b2c`): `WarmupIntRange(0, 2)` → `(0, 3)` em
  `BuildPrefillChunkMetadataKernel.get_warmup_keys`, para aquecer o bucket de inteiro `''` do Triton e evitar o JIT
  no meio do serviço em prefills de mais de ~75k tokens; também aplicado no build (Dockerfile).
- **Checkpoint:** o default virou `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@76c0b517` (83 shards, `98cf18f`). Os
  números de KLD citados acima (0,024555 etc.) são medições do **TR3**. O preset `dense-h3` renomeou
  `DENSE_H3_TR3_*` para `DENSE_H3_BASE_*`, mas os SHA-256 fixados continuam os do TR3 enquanto o default mudou:
  inconsistência upstream, não a copie como se fosse o pack novo.

# Inventário — Quantização e formatos numéricos nas 3 receitas (GLM-5.3-Flash EXL3 · Qwen3.8-Flash-Next NVFP4 · DeepSeek-V4-Flash DSpark)

> Escopo: TODO formato numérico / esquema de quantização que aparece (ou é explicitamente descartado) nos três repositórios, explicado desde os bits até o kernel que o consome, com exemplos numéricos passo a passo (Apêndice A) para virarem animações.
>
> Repositórios (somente leitura) — caminhos relativos a `scratchpad/repos/`:
> - **GLM** = `GLM-5.3-Flash-EXL3-2x-DGX-Sparks`
> - **Qwen** = `Qwen3.8-Flash-Next-Dual-DGX-Sparks`
> - **DS** = `DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
>
> Convenções deste documento:
> - Citações `arquivo:linha` referem-se ao repo indicado. Onde escrevo **(interpretação)** é dedução minha a partir do código/aritmética, não uma afirmação textual do repo. Onde escrevo **(externo)** a informação vem de fonte web (Apêndice B).
> - "bpw" = bits por peso (bits per weight). "M" = número de linhas (tokens) de uma GEMM `A[M,K] @ W[K,N]`.
> - GB10 = o SoC do DGX Spark: GPU Blackwell "consumer" com compute capability **12.1 (`sm_121`/`sm_121a`)**, ~121.69 GiB visíveis ao CUDA de memória unificada LPDDR5x (~273 GB/s teóricos segundo `Qwen/docs/CLAUDE/fable5-1-report.md:21`).

---

## 0. Mapa rápido — qual formato aparece onde

| Formato / esquema | GLM-5.3-Flash (EXL3) | Qwen3.8-Flash-Next (NVFP4) | DeepSeek-V4-Flash (DSpark) |
|---|---|---|---|
| **FP32** | escalas do `fp8_ds_mla` (4×FP32 por token), acumuladores, saída do MoE EXL3 (`out` fp32) | escala global NVFP4 (`weight_scale_2`), escalas por canal do FP8-dense (`weight_scale` F32), estado GDN do checkpoint (float32) | acumuladores; escalas FP8 block (`weight_scale_inv`, formato UE8M0 no checkpoint) |
| **BF16** | pesos densos/atenção/shared/embeddings/lm_head do checkpoint TR3 (nativos), draft DFlash2, KV do draft, cópia retida KDA large-M, shards BF16 do dense-EXL3 | pesos densos do checkpoint nvidia (GDN, atenção, HyperConnection, shared expert, visão, lm_head), KV `auto`, estado SSM (`MAMBA_SSM_CACHE_DTYPE=bfloat16`) | ativações; parte RoPE do KV (64×BF16); lm_head bf16 |
| **FP16** | ativações dentro dos kernels EXL3 (entrada `half`), `suh`/`svh`, temps do fused MoE | V desquantizado p/ FP16 no PV-dot (backport v030) | — |
| **FP8 E4M3** | dense FP8 **per-output-channel W8A16 Marlin** (`GLM53_DENSE_FP8`), NoPE do KV `fp8_ds_mla` | KV **per-tensor** (QSA patch), FP8-dense **per-channel W8A8 dinâmico por token**, tabela PLE **per-tensor**, experts MTP **block 128×128**, escalas de bloco NVFP4 | NoPE do KV `fp8_ds_mla`/`nvfp4_ds_mla` com escala **UE8M0 por 64**, lineares densos **block 128×128**, K-cache do indexer (132 B/linha), saída da atenção antes de `wo_a` |
| **FP8 E5M2** | — | — | só aparece como string da lista `CacheDType` do vLLM (`fp8_e5m2`) no patch (`DS/patches/official-main-b12x-nvfp4-python.patch:6`); não usado |
| **NVFP4** (E2M1 + E4M3/16 + FP32 global) | **não servido** (KLD 0.0605 vs 0.0246; SM12x NVFP4 kernels são MHA densos) — `GLM/README.md:423,751`; doador do ablit é um checkpoint NVFP4 | **experts roteados W4A4** (FlashInfer CUTLASS SM120), PLE NVFP4 (checkpoint local-inference-lab) | só o **nome** `nvfp4_ds_mla` (bytes idênticos ao FP8!), plugin opcional `modelopt_gb10_hybrid` (Marlin W4A16 / CUTLASS W4A4) |
| **MXFP4** (E2M1 + E8M0/32) | — | — | **experts roteados nativos do checkpoint** (`fp4_e8m0_k32`) rodados como **W4A16** pelo b12x; K-cache do indexer em MXFP4 (opt-in) |
| **MXFP8** (E4M3 + E8M0/32) | — | lineares MXFP8 de alguns checkpoints (local-inference-lab: "MXFP8 attention"); patch de fallback p/ BF16 | citado como caminho futuro (`kind::mxf8f6f4`) |
| **EXL3 (QTIP trellis)** | **experts roteados K4 MCG ("uniform-K4", 4 bpw)**; opcional dense EXL3 (turboderp 4.05bpw, mul1) + draft 6 bpw | — | — (o repo GLM cita um irmão "DeepSeek-V4.1-Flash-EXL3-2.9bpw", fora deste escopo) |
| **INT (uint8/int16/int32 como contêiner)** | trellis `int16`, marcador `mcg`/`mul1` `int32` | NVFP4 empacotado em `uint8` (2 valores/byte) | MXFP4 empacotado `uint8`, escalas E8M0 como `uint8`, KV como `uint8` |

---

## 1. Visão geral

### 1.1 Por que quantização domina estas receitas
Os três modelos são MoE gigantes (centenas de experts). Numa GB10 a memória é unificada (CPU+GPU compartilham ~121.69 GiB) e a banda é baixa (~273 GB/s teóricos, 230–250 GB/s efetivos segundo os relatórios). Consequências que os repos medem explicitamente:

1. **Decode é limitado por banda**: cada passo de decode precisa *ler* todos os pesos ativos. Qwen: "≈ 9.2 GiB per step … Batch-1 decode on this box is therefore at the memory-bandwidth roofline already" (`Qwen/docs/CLAUDE/fable5-1-report.md:74-92`). DS: "c=1 decode on TP=2 is ~80 % weight streaming" (`DS/docs/CLAUDE/fable5-1-report.md:84-96`). Logo, **menos bits por peso = mais tokens/s** quase linearmente.
2. **Capacidade**: pesos + KV + ativações têm que caber na UMA. Menos bytes de peso liberam KV; menos bytes de KV aumentam tokens de contexto/concorrência.
3. **Prefill é limitado por compute**: aí o que importa é se o formato tem tensor core nativo (FP8/FP4) ou se o kernel precisa desquantizar para BF16/FP16 antes da MMA (DS: "W4A16 (dequant→bf16 MMA) … ~125 dense bf16 TFLOPS class on GB10 instead of FP4×FP8 tensor cores (~4× the dense rate)", `DS/docs/CLAUDE/fable5-1-report.md:37,278`).

### 1.2 O que cada receita escolheu (fim a fim)

**GLM-5.3-Flash (EXL3)** — checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (espelho de `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, "uniform-K4 EXL3/TR3 routed-experts, 4 bpw, ~164 GiB, 120 shards", `GLM/README.md:12-15`):
- **Experts roteados** em **EXL3 K=4, codebook MCG** (quantização por trellis estilo QTIP + transformada de Hadamard). Ficam empacotados em VRAM; nunca viram BF16 (`GLM/overlay/exl3.py:1-14`). Consumidos por `exllamav3_ext.exl3_moe` (1 launch por camada no decode) e pelos kernels "fat expert" E2/E3 do repo no prefill.
- **Denso/atenção/shared/embeddings/lm_head** nativos BF16 no checkpoint (`GLM/README.md:446`), mas o `.env.example` liga `GLM53_DENSE_FP8=all`: na carga, essas projeções são **re-quantizadas para FP8 E4M3 por canal de saída** e executadas por **Marlin W8A16** (`GLM/overlay/exl3.py:1895-1911, 2098-2144`). Para prefill (M>512) a projeção KDA `in_proj` usa uma **cópia BF16 da mesma matriz FP8 lógica** via cuBLAS (`GLM53_KDA_BF16_LARGE_M=1`).
- **KV** do alvo: `--kv-cache-dtype fp8` → layout empacotado **`fp8_ds_mla` de 656 B/token/camada** (512 FP8 NoPE + 16 B de escalas FP32 + 128 B RoPE BF16 preenchidos com zeros porque o GLM é NoPE MLA) (`GLM/Dockerfile:1-29`, `GLM/README.md:670-681`). KV do draft DFlash2: `auto`/BF16.
- **Opcional** (preset `dense-h3`): lineares densos em **EXL3 (turboderp 4.05bpw, codebook mul1)** + draft DFlash2 quantizado a **6 bpw EXL3**, com retenção BF16 para M>144.
- **Qualidade**: painel KLD independente: 4bpw = 0.024555 nats, igual ao FP8 oficial na mesma stack (0.024629) com 54% dos bytes; NVFP4 = 0.060535 (`GLM/README.md:410-428`).

**Qwen3.8-Flash-Next (NVFP4)** — checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` ("133G / 11 shards: BF16 dense + NVFP4 experts + FP8 PLE", `Qwen/.env.sample:24`):
- **Experts roteados em NVFP4** (E2M1 + escala E4M3 a cada 16 + escala global FP32), executados como **W4A4** (ativação também quantizada para NVFP4) no **FlashInfer CUTLASS SM120 grouped GEMM** (`Qwen/README.md:479`, `Qwen/docs/CLAUDE/fable5-1-report.md:36`). EP obrigatório.
- **Tabela PLE (n-gram, ~51 GB)** em **FP8 E4M3 com uma escala global** (per-tensor) — requer patch de dispatch (`Qwen/README.md:550-562`).
- **Experts do MTP** em **FP8 block-scaled 128×128** — requer patch (`Qwen/files/patch_modelopt_fp8_block_moe.py`).
- **KV** FP8 E4M3 **per-tensor** nas 12 camadas QSA (+1 MTP), via patch que ensina os kernels Triton QSA a ler FP8 (`Qwen/files/patch_qsa_fp8_kv.py`): 3,652,200 tokens vs 2,131,159 em BF16 (1.70× por GiB).
- **Opcional**: checkpoint híbrido **NVFP4 experts + FP8 dense (per-channel, ativação dinâmica por token, W8A8 CUTLASS)** (`Qwen/files/fp8dense/`), fallback **MXFP8 → BF16 emulation** para shapes que o `mm_mxfp8` recusa, estado GDN em BF16.

**DeepSeek-V4-Flash (Vision-Exp) DSpark** — checkpoint oficial `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`; config: "256 routed experts top-6 + 1 shared, `moe_intermediate_size` 2048, `expert_dtype fp4` (E8M0 block-32 scales), other linears fp8 block 128×128" (`DS/docs/CLAUDE/fable5-1-report.md:450-454`):
- **Experts roteados em MXFP4 nativo** (`fp4_e8m0_k32`), executados pelo backend **b12x como W4A16** (desquantiza para BF16 e faz MMA BF16), porque o caminho NVFP4 tensor-core do b12x só aceita fonte `modelopt_nvfp4` (`DS/docs/CLAUDE/fable5-1-report.md:269-301`, `DS/recipe/overlay/vllm/model_executor/layers/fused_moe/b12x_moe.py:427-497`).
- **Demais lineares em FP8 block 128×128** com escalas **UE8M0** (`weight_scale_inv`, `scale_fmt` do config) (`DS/recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py:1025, 2073-2090`).
- **KV**: `--kv-cache-dtype nvfp4_ds_mla` — que, apesar do nome, é **byte-idêntico** ao `fp8_ds_mla` do DSV4: **584 B/token = 448 B FP8 E4M3 NoPE (7 blocos de 64 com escala UE8M0) + 128 B RoPE BF16 + 8 B de escalas (7 + 1 pad)**. O hotfix do issue #22 roteia `nvfp4_ds_mla` para o kernel FP8 rápido (`DS/docs/PATCHES.md:169-204`, `DS/docs/CLAUDE/item8-fp4-kv-design.md:7-12`).
- **Indexer (Lightning)**: K-cache FP8 132 B/linha; opcional **MXFP4** (`DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1`).
- O modelo foi **treinado com simulação de quantização FP8 do KV** (QAT) nas dimensões NoPE (`DS/recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark_kernels.py:640-645`).

---

## 2. Inventário de parâmetros

### 2.1 GLM-5.3-Flash EXL3

| Nome | Valor default (alternativos) | Onde | O que faz | Conceitos | Por que importa / trade-off |
|---|---|---|---|---|---|
| `MODEL` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (fallback `brandonmusic/GLM-5.3-Flash-tr3-4bpw`) | `GLM/.env.example:42-43`; `GLM/README.md:1732-1733` | Seleciona o checkpoint EXL3 uniform-K4 | EXL3, bpw, checkpoint safetensors | 4 bpw ≈ 164 GiB; KLD igual ao FP8 oficial com 54% dos bytes (`README.md:425`) |
| `MODEL_REVISION` | `25a44fdbf16862a46b7cc9921142c6c81350af2f` | `GLM/.env.example:48` | Pino imutável do snapshot | reprodutibilidade | a qualidade medida (KLD) vale para esses bytes |
| `QUANTIZATION` | `exl3` ("never `marlin`") | `GLM/.env.example:103`; `GLM/start.sh:288, 2116`; `GLM/README.md:1753` | Passa `--quantization exl3` → `Exl3Config` do overlay | plugin de quantização do vLLM | `--moe-backend marlin` é o workaround do NVFP4 e seria o backend errado (`GLM/Dockerfile:3-5`) |
| `quantization_config` (no checkpoint) `quant_method=exl3, codebook=mcg, scope=glm53_routed_experts_only, bits=4` | — | `GLM/overlay/exl3.py:4-6, 1825-1839` | Define o ABI: por matriz de expert, `trellis` (int16) + `suh`/`svh` (fp16) + `mcg` (int32 marcador) | ABI EXL3 | Overlay só implementa `codebook=mcg` (`exl3.py:1722-1725`) e `bits ∈ {3,4,5,6}` (`exl3.py:1726-1727`) |
| `quantization_config.json` / `tensor_storage` | ledger de 37 MiB ignorado | `GLM/overlay/exl3.py:1821-1832` | Arquivo de config lido pelo método; o ledger por tensor fica fora do objeto | metadados | evita carregar 37 MiB de JSON no config |
| `quantization_config.non_routed_exl3.layers` | ausente (normal) / `{prefix: {bits: K, bf16_shards: [...]}}` | `GLM/overlay/exl3.py:1728-1754`; `GLM/tools/dense_overlay.py:354-377` | Declara lineares densos em EXL3 (K 2–6). `kv_b_proj` é recusado ("MLA weight absorption reads that weight directly") | dense EXL3, absorção de pesos no MLA | Mutuamente exclusivo com `GLM53_DENSE_FP8` por módulo (`exl3.py:1873-1884`) |
| `glm53_profile.name = dense-exl3-h3-dflash2-6bpw` | ausente (normal) | `GLM/README.md:49-65`; `GLM/tools/pack_profile.py:17-20` | Perfil de pack que seleciona dense EXL3 + draft 6 bpw | pareamento target/draft | Validado por hash do config do draft; TP2 apenas |
| `GLM53_MODEL_PRESET` | vazio / `dense-h3` | `GLM/.env.example:169`; `GLM/README.md:67-111` | Constrói o par H3 a partir de TR3 + tensores densos turboderp 4.05bpw + draft IncoAI quantizado a 6 bpw | EXL3 dense, quantização de draft | Opt-in; "not yet A/B measured against defaults" |
| `GLM53_DENSE_FP8` | unset→`off` no launcher; `.env.example` = `all` (alternativas `off`, lista de `shared,dense,kda,mla`) | `GLM/.env.example:152-158`; `GLM/start.sh:433`; `GLM/overlay/exl3.py:1895-1930` | Re-quantiza projeções densas BF16 para **FP8 E4M3 por canal de saída** na carga e roda em **Marlin W8A16** | FP8 E4M3, per-channel, weight-only, Marlin | −11 ms/step; KL proxy 0.002–0.013 nats/pos, argmax 94–100% (`README.md:344`); `all` vs `dense,kda`: decode ms/cycle −4.67/−4.91% (`docs/adopt-fp8-all-kda-bf16-2026-09-26.md:26-27`) |
| grupos de `GLM53_DENSE_FP8` | `shared`: `mlp.shared_experts.{gate_up,down}_proj`; `dense`: `mlp.{gate_up,down}_proj`; `kda`: `self_attn.{in_proj_qkvbfg_a,f_b_proj,g_b_proj,o_proj}`; `mla`: `self_attn.{fused_qkv_a_proj,q_b_proj,o_proj}` (kv_b_proj fica BF16) | `GLM/overlay/exl3.py:1912-1917` | Define quais módulos viram FP8 | seleção por módulo | Só camadas do alvo (índice < `len(layer_types)`); draft/MTP/visual nunca (`exl3.py:1950-1967`) |
| TP=3 exceção Marlin | `f_b_proj`, `g_b_proj` ficam BF16 em TP3 | `GLM/overlay/exl3.py:1985-2001`; `GLM/README.md:1487-1490` | "row pitch is not 8-aligned at 22 local heads" | requisitos de alinhamento de kernel | Marlin exige pitch múltiplo de 8 / offset 16-B alinhado |
| `GLM53_KDA_BF16_LARGE_M` | unset→`0`; `.env.example` = `1` | `GLM/.env.example:159-162`; `GLM/overlay/exl3.py:2004-2037` | Para a `in_proj` KDA: M≤512 → Marlin FP8; M>512 → `F.linear` BF16 com cópia retida da matriz FP8 lógica | crossover de kernel por M; dequant-to-BF16 | +3.29 GiB/rank; prefill TTFT −11.4% a −14.1% (`docs/kda-bf16-large-m.md:98-100,107-111`) |
| `KDA_BF16_LARGE_M_MIN_M` | `512` (fixo, não configurável) | `GLM/overlay/exl3.py:2015` | Fronteira de dispatch | CUDA graph baked branch | "decode never exceeds M=220, prefill chunks land at M>=768" |
| `KDA_BF16_LARGE_M_SHAPES_BY_TP` | TP2 `(12576, 4096)`, TP3 `(8726, 4096)` | `GLM/overlay/exl3.py:2019-2022` | Shapes qualificados (fail-closed fora deles) | TP sharding | cópia 98.25 MiB/camada/rank em TP2 |
| `KDA_BF16_LARGE_M_CHUNK_ROWS` | `512` | `GLM/overlay/exl3.py:2026` | Desquantização em blocos de 512 linhas (~8 MiB FP32) | memória de pico | evita 196 MiB de temporário FP32 |
| `GLM53_DENSE_EXL3` | `0` / `1` | `GLM/start.sh:443, 772-801`; `GLM/overlay/patch_dense_fp8.py:84-94` | Habilita servir pack com `non_routed_exl3` | dense EXL3 | Requer `GLM53_DENSE_FP8=off`, `ABLIT=0`, TP2 |
| `GLM53_DENSE_EXL3_PREFILL_BF16` | `off` / `all` / lista de `kda_in,kda_o,mla_qkv_a,mla_q_b,mla_o,shared_gate_up,shared_down,dense_gate_up,dense_down`; perfil H3 usa `kda_in,shared_down,mla_qkv_a,shared_gate_up,kda_o,mla_q_b` | `GLM/overlay/exl3.py:2294-2352`; `GLM/tools/pack_profile.py:18` | Retém BF16 reconstruído (Hadamards já aplicados) de módulos dense-EXL3; M>144 usa F.linear BF16 | reconstrução vs decodificação em-kernel | 2 B/peso/rank a mais |
| `DENSE_EXL3_PREFILL_BF16_MIN_M` | `144` | `GLM/overlay/exl3.py:2306-2310` | = `AUTO_RECONSTRUCT_THRESHOLD` do LinearEXL3 | crossover EXL3 bitcoder vs reconstruct | acima de 144 linhas o próprio exllamav3 reconstrói o peso |
| `GLM53_EXL3_MOE_FAST` | `0` / `1` | `GLM/start.sh:430`; `GLM/overlay/exl3.py:1266-1294`; `GLM/docs/sm121-perf-paths.md:13-40` | Kernels thin-decode K4/N256 especializados p/ SM121 (1 estágio de registrador, 8 de smem) | pipeline de dequant, pressão de registradores | +7.71% / +8.88% / +14.59% decode (3 células resolvidas); estudo numérico "INCONCLUSIVE" |
| `EXL3_FUSED_MOE` | `1` / `0` (loop LinearEXL3) | `GLM/.env.example:108`; `GLM/overlay/exl3.py:761-762` | Um launch `exl3_moe` por camada | kernel fusionado | decode graph-safe |
| `EXL3_TEMP_ROWS_FUSED` | launcher: `32` com E3, `256` com E2 (código default 128) | `GLM/.env.example:111-112`; `GLM/overlay/exl3.py:66, 177-181`; `GLM/README.md:1765` | Linhas/expert no buffer temporário do kernel fused; acima disso o expert é "fat" | tiling por expert | "Keep ≥ MAX_NUM_SEQS × (DFLASH_TOKENS+1)" |
| `EXL3_FAT_KERNEL` / `EXL3_FAT_GROUPED` / `EXL3_FAT_BATCHED` / `EXL3_FAT_SORTED` / `EXL3_MOE_ROW_TILE` | `1` / `1` / `0` / `0` / `0` | `GLM/.env.example:108-127`; `GLM/overlay/exl3.py:168-226` | Hierarquia de caminhos p/ experts com muitas linhas no prefill (grouped>kernel>batched>sorted>legacy) | consumidores do formato EXL3 no prefill | E3 +37–45% prefill frio; 560 MiB de scratch cobrado do KV |
| `EXL3_FAT_GROUPED_TOPK` | `8` | `GLM/overlay/exl3.py:1113` | top-k para dimensionar scratch E3 | MoE top-k | — |
| `KV_CACHE_DTYPE` | `fp8` ("not `nvfp4`, not bf16") | `GLM/.env.example:210`; `GLM/start.sh:274, 2122`; `GLM/README.md:1771` | Alvo usa `fp8_ds_mla` empacotado | KV quantizado MLA | "bf16 KV has no sparse kernel on this arch" (`README.md:670-673`) |
| draft `kv_cache_dtype` | `"auto"` (BF16) | `GLM/start.sh:2130`; `GLM/README.md:1759` | KV do draft DFlash2 | — | "dense DFlash2 cannot use the target's fp8_ds_mla layout and SM121 has no FA3/FA4 for plain FP8" (`README.md:584-586`) |
| `EXTRA_ARGS="--kv-cache-memory-bytes 11811160064"` | 11 GiB (antes 14 GiB = 15032385536) | `GLM/.env.example:191` | Reserva explícita do pool KV | orçamento de memória | 11 GiB → 1,233,779 tokens (1.45× a 850k), paga os +3.3 GiB/rank da cópia KDA BF16 |
| `DFLASH_MODEL`/`DFLASH_REVISION` | `incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c…` (BF16, ~2.3 GiB) | `GLM/.env.example:132-137` | Draft BF16 | speculative decoding | draft em EXL3 6 bpw só no perfil H3 |
| `BITS` (script de quantização do draft) | `6.0` | `GLM/tools/dflash2_exl3_quant.sh:24-26, 45` | Taxa alvo do conversor EXL3 | bpw | — |
| `DRAFT_QUANT_KERNEL_PROJ` / `DRAFT_QUANT_FC` | `1` / `1` | `GLM/tools/dflash2_exl3_quant.sh:26-27` | Inclui `kernel_projection` e `fc` no orçamento de quantização | seleção de módulos | K/V do draft ficam BF16 sempre |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | `GLM/Dockerfile:381-382` | Compila cubins nativos `sm_121a` | arquitetura alvo | exllamav3_ext compilado para SM121, não SM120 |
| `EXLLAMAV3_COMMIT` | `c5d9c657966ffeeaa9353f0cc899f18629da4a13` (0.0.43) | `GLM/Dockerfile:380`; `GLM/overlay/exl3.py:55-56` | Pino da biblioteca de kernels EXL3 | reprodutibilidade de kernel | upstream candidato `0c49587a` tem GEMV small-M estilo QTIP e GEMV int8-ativação (não adotado) |
| `MCG_MULTIPLIER` / marcadores | `0xCBAC1FED` (int32 `-877912083`); mul1 `0x83DCD12D` (`-2082680531`) | `GLM/overlay/exl3.py:57-61` | Identifica o codebook procedural | codebook computado | carga falha se marcador não bater (`exl3.py:3145-3153`) |
| `SWIGLU_LIMIT_DEFAULT` | `10.0` | `GLM/overlay/exl3.py:63` | clamp da ativação SwiGLU dentro do kernel | estabilidade numérica FP16 | evita overflow em FP16 |
| `GLM53_DRAFT_KV_COMPACT` | `1` com dflash | `GLM/README.md:756-768` | Geometria de páginas do draft | alocador (não quantização) | "changes neither weight nor KV precision" |
| `ENABLE_EXPERT_PARALLEL` (TP3) | on | `GLM/README.md:1475` | Em TP3 cada rank tem 96 de 288 experts | EP vs TP | "do not pad (EXL3 trellis is packed 2048-wide)" |

### 2.2 Qwen3.8-Flash-Next NVFP4

| Nome | Valor default (alternativos) | Onde | O que faz | Conceitos | Por que importa / trade-off |
|---|---|---|---|---|---|
| `MODEL_ID` | `nvidia/Qwen3.8-Flash-Next-NVFP4` (133G/11 shards: BF16 dense + NVFP4 experts + FP8 PLE); alternativas `local-inference-lab/…` (99G/37 shards: MXFP8 attention + NVFP4 PLE), `RadixArk/…` (126G/206 shards: BF16 attention + FP8 PLE) | `Qwen/.env.sample:24-26` | Escolhe o mix de formatos | ModelOpt MIXED_PRECISION | cada checkpoint registra metadados diferentes → patches diferentes |
| `KV_CACHE_DTYPE` | `fp8` (alternativa `auto` = BF16) | `Qwen/.env.sample:60-69`; `Qwen/start.sh:87, 665-674, 883` | KV FP8 E4M3 per-tensor; aplica `patch_qsa_fp8_kv.py` automaticamente | KV quantizado; escala per-tensor | 3,652,200 vs 2,131,159 tokens (1.70×); −5.3% throughput médio (`CHANGELOG.md:129-133`) |
| `OVERRIDE_KV_CACHE_DTYPE` | vazio / `fp8` (lane v0.30) | `Qwen/start.sh:122`; `Qwen/README.md:85-88` | Força dtype no lane v0.30 (backport vllm#55557) | — | 2.64M tokens, NLL 1.328/1.334, ~5% decode mais lento |
| `MAMBA_SSM_CACHE_DTYPE` | `bfloat16` (vazio = float32 do checkpoint) | `Qwen/.env.sample:70-84`; `Qwen/start.sh:93, 884` | Dtype do estado recorrente GDN | precisão do estado recorrente | single-Spark: +8.5% decode a 8 streams; não medido no dual |
| `ENABLE_EXPERT_PARALLEL` | `true` ("Required for NVFP4") | `Qwen/.env.sample:86` | Experts particionados por rank | EP | shapes NVFP4 exigem K múltiplo de 16 |
| `MM_ENCODER_TP_MODE` | `data` | `Qwen/.env.sample:89-92` | Replica o encoder de visão | restrição de divisibilidade | "NVFP4 kernels require input features divisible by 16… 4304/2=2152" (2152/16 não é inteiro) |
| `PLE_EMBEDDING_DTYPE` | vazio (detectado) / `float8_e4m3fn` / `nvfp4` | `Qwen/.env.sample:36-41`; `Qwen/files/detect_ple_dtype.py`; `Qwen/start.sh:919-945` | Injeta `text_config.ple_embedding_dtype` via `--hf-overrides` | metadados de quantização | sem isso a tabela PLE FP8 de 51 GB falha ao carregar |
| `PLE_QUANT_OVERRIDE=fp8` | citado como env injetado | `Qwen/README.md:460, 557` | Shim antigo do resolver PLE | — | **(observação)** não encontrei referência no `start.sh` atual; o mecanismo vigente é o `ple_embedding_dtype` via hf-overrides |
| `PLE_OFFLOAD` | `false` | `Qwen/.env.sample:101` | Offload da tabela PLE para RAM | mmap de linhas empacotadas | precisa ~51 GB de RAM livre |
| `FP8_DENSE` / `FP8_DENSE_MODEL_ID` | `false` / `MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense` | `Qwen/.env.sample:147-153`; `Qwen/start.sh:129-132, 576` | Serve checkpoint híbrido NVFP4 + FP8 per-channel | W8A8 dinâmico | analítico 9.2→5.7 GiB/step (~1.5× batch-1); **não medido em GPU** |
| `SRC_REPO` / `DST_REPO` (build fp8dense) | `RadixArk/Qwen3.8-Flash-Next-NVFP4` / `MiaAI-Lab/…-FP8dense` | `Qwen/files/fp8dense/build.sh:7-9` | Origem/destino da conversão | — | escrito para o layout RadixArk |
| `quant_algo` em `quantized_layers` | `NVFP4` (group_size 16), `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES` (group_size 128), `FP8_PB_WO` (alias mal rotulado), `MXFP8`, `W4A16_NVFP4`, `FP8` | `Qwen/files/fp8dense/make_fp8_dense_checkpoint.py:264,271`; `Qwen/files/patch_checkpoint_config.py:93` | Algoritmo por prefixo de camada | ModelOpt mixed precision | um nome errado → camada construída **não quantizada** e crash tardio |
| `MXFP8_BLOCK_SIZE` / limites `mm_mxfp8` | bloco 32; `N ≥ 128, N%32==0, K ≥ 128, K%32==0` | `Qwen/files/patch_modelopt_mxfp8.py:5-11, 70-77` | Critério de fallback p/ `EmulationMxfp8LinearKernel` | restrições de kernel | ~17 MB extra BF16 |
| `weight_block_size` (MTP experts) | `[group_size, group_size]` = `[128,128]` | `Qwen/files/patch_modelopt_fp8_block_moe.py:51-62` | Constrói `Fp8Config(activation_scheme="dynamic")` | FP8 block-scaled | "refusing to guess a block shape" |
| `MTP_DRAFT_VOCAB` | `files/draft_vocab_en_code_47k.txt` | `Qwen/.env.sample` (seção draft vocab) | Reduz o lm_head BF16 do draft | bytes/step | +9.6% médio; não é quantização mas é o mesmo "raciocínio de bytes" |
| `QSA_PROFILE` | `stock` (`gb10`, JSON) | `Qwen/.env.sample:158` | Tabela de tiles QSA | — | conflita com o patch FP8 KV (ambos sobrepõem `ops/qsa.py`, `start.sh:665-668`) |
| `GPU_MEMORY_UTILIZATION` | `0.80` (medições a 0.835) | `Qwen/.env.sample:53-56` | Orçamento | UMA | +0.01 GMU ≈ +0.83 GiB ≈ +95k tokens FP8 |

### 2.3 DeepSeek-V4-Flash DSpark

| Nome | Valor default (alternativos) | Onde | O que faz | Conceitos | Por que importa / trade-off |
|---|---|---|---|---|---|
| `--kv-cache-dtype` | `nvfp4_ds_mla` | `DS/docker-compose.dspark.yml:500`; `DS/README.md:593` | Layout sparse-MLA de 584 B/token (idêntico ao `fp8_ds_mla`) | KV MLA quantizado; UE8M0 | "there is no accuracy risk from 'FP4 KV' and no memory win either" (`DS/docs/CLAUDE/fable5-1-report.md:345-346`) |
| `--block-size` | `256` | `DS/docker-compose.dspark.yml:501` | Tamanho de bloco de KV | páginas / alocador | página comum 64 linhas × 584 B = 37,376 B em todos os grupos (`item8-fp4-kv-design.md:15`) |
| `DSPARK_SKIP_ISSUE22_HOTFIX` | `0` ("Don't, on this recipe") | `DS/README.md:291`; `DS/docker-compose.dspark.yml:315` | Pula o hotfix de dispatch nvfp4→fp8 | dispatch de kernel | sem ele: ~1 tok/s a 600K vs ~17 tok/s |
| `--moe-backend` | `flashinfer_b12x` | `DS/docker-compose.dspark.yml:517` | Mapeia MXFP4 → `Mxfp4MoeBackend.B12X_MXFP4` | backends MoE | `DS/patches/official-main-b12x-nvfp4-python.patch` (hunk `map_mxfp4_backend`) |
| `VLLM_USE_B12X_MOE` | `1` | `DS/.env.dspark.example:590`; `DS/docker-compose.dspark.yml:191` | Liga o MoE b12x | CuTeDSL | — |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` / `_MAX_M` / `_TILE_CONFIG` | `0` / `16` / vazio | `DS/.env.dspark.example:591-593`; `DS/recipe/overlay/.../b12x_moe.py:174-311` | Override do seletor de tiles W4A16 | tiling/occupancy | "leave them unless a profile shows occupancy trouble" |
| `B12X_W4A16_TC_DECODE` | `0` (comentado) | `DS/.env.dspark.example:595`; `DS/docker-compose.dspark.yml:196` | Kernel "small-M packed decode" (M≤8) | small-M decode | 0–10% c=1 esperado; A/B pendente |
| `B12X_CUTE_COMPILE_CACHE_DIR` | `/cache/huggingface/b12x-cute-cache` | `DS/.env.dspark.example:644` | Persiste JIT do CuTeDSL | JIT | evita re-JIT do `W4A16FusedMoeKernel` |
| `CUTE_DSL_ARCH` / `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | `sm_121a` / `12.1a` / `12.1a` | `DS/.env.dspark.example:602-604` | Alvo de compilação | arch | evita fallback JIT mais lento |
| `DSPARK_ENABLE_MXFP4_INDEXER_CACHE` | `0` / `1` | `DS/.env.dspark.example:461-471`; `DS/docker-compose.dspark.yml:459` | Aplica hotfix do gate + `--attention-config {"use_fp4_indexer_cache":true}` | MXFP4 (E2M1 + UE8M0/32) | lê metade dos bytes por chave; risco = drift de seleção top-k |
| `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS` | `0` / `1` (obrigatório com o MXFP4 indexer) | `DS/.env.dspark.example:414-419` | Cabeçalhos alias `sm121_* → sm120_*` | JIT DeepGEMM | sem ele, cache miss → erro de compilação |
| `ENABLE_VLLM_GB10_PATCH` / `GB10_HYBRID_NVFP4_M_THRESHOLD` / `VLLM_PLUGINS` | `0` / `128` / `gb10_hybrid_nvfp4` | `DS/docker-compose.dspark.yml:179-181, 468-469`; `DS/vllm_patch_gb10/` | `--quantization modelopt_gb10_hybrid`: Marlin W4A16 se M<128, CUTLASS W4A4 se M≥128 | crossover W4A16/W4A4 | experimental; exige checkpoint ModelOpt NVFP4 W4A4; guarda 2 cópias dos pesos |
| `VLLM_USE_B12X_WO_PROJECTION` | `0` (Stage-C only) | `DS/.env.dspark.example:749`; `DS/recipe/overlay/vllm/envs.py:1029-1033` | Projeção WO fusionada b12x preservando escalas UE8M0 | FP8 block UE8M0 | não registrada no Anemll 0.1.1 |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | `0` (Stage-C only) | `DS/.env.dspark.example:756`; `DS/recipe/overlay/.../dspark.py:213-215, 312-319` | Simula quant/dequant FP8 (grupo 64, pow-2) no KV NoPE do draft DSpark | QAT parity | paridade com o forward de referência |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | `0` (Stage-C only) | `DS/.env.dspark.example:760`; `DS/recipe/overlay/.../sm120.py:76-77` | Decode MLA comprimido b12x | — | experimento |
| `quantization_config.scale_fmt` (checkpoint) | UE8M0 (lido) | `DS/recipe/overlay/.../model.py:1025` | Formato das escalas FP8 block | UE8M0 | escalas potência de 2 |
| `expert_dtype` (config) | `fp4` (alternativa FP8 experts) | `DS/recipe/overlay/.../model.py:2073-2125` | Escolhe mapeador de nomes de escala (`weight_scale` vs `weight_scale_inv`) | MXFP4 vs FP8 MoE | — |
| `GPU_MEMORY_UTILIZATION_TEXT` | `0.835` | `DS/.env.dspark.example:423` | Orçamento | UMA | 17.04 GiB KV / 2,331,430 tokens |

---

## 3. Inventário de modificações (patches, overlays, kernels, scripts que tocam formatos numéricos)

Formato de cada item: **nome** · arquivo(s) · problema · mecanismo técnico · conceitos · efeito medido.

### 3.1 GLM-5.3-Flash EXL3

**G1 — Registro do método `exl3` no vLLM**
- Arquivos: `GLM/Dockerfile:385-415` (edita `vllm/model_executor/layers/quantization/__init__.py`), `GLM/overlay/patch_model_overrides.py` ("`exl3` in ModelConfig overrides", `GLM/README.md:1933`), `GLM/overlay/exl3.py:1705-1892`.
- Problema: o vLLM stock não conhece `quant_method=exl3`; sem registro o checkpoint cai em BF16 ou falha.
- Mecanismo: adiciona `"exl3"` ao `Literal` `QuantizationMethods` e ao dicionário `method_to_config` (import preguiçoso de `Exl3Config`); `Exl3Config.override_quantization_method` devolve `"exl3"` quando o `hf_quant_cfg.quant_method == "exl3"` (`exl3.py:1841-1851`). `get_quant_method(layer, prefix)` devolve `Exl3MoEMethod` para `RoutedExperts`, `Exl3LinearMethod` para lineares declarados em `non_routed_exl3`, `Glm53DenseFp8Method` para grupos FP8 e `UnquantizedLinearMethod` para o resto (`exl3.py:1853-1892`). `get_min_capability()=80` ("LinearEXL3 uses CUDA >= Ampere", `exl3.py:1815-1818`); ativações aceitas BF16/FP16/FP32 (`exl3.py:1812-1813`).
- Conceitos: arquitetura de quantização plugável do vLLM (QuantizationConfig → QuantizeMethodBase: `create_weights`, weight loader, `process_weights_after_loading`, `apply`).
- Efeito: "Registering the name `"exl3"` is not enough. Experts must stay trellis + suh + svh + mcg and run Trellis/MCG" (`GLM/README.md:574-577`).

**G2 — `Exl3MoEMethod`: experts roteados empacotados (K4 MCG)**
- Arquivo: `GLM/overlay/exl3.py:2973-3248`.
- Problema: manter os experts no formato comprimido (sem expandir para BF16) e shardeá-los em TP2.
- Mecanismo:
  - `create_weights` aloca, por camada, `w13_trellis[E, 2, in_tiles, out_tiles, 16·K]` int16, `w13_suh[E,2,hidden]` fp16, `w13_svh[E,2,inter_local]` fp16, `w13_mcg[E,2,1]` int32, e idem `w2_*` (`exl3.py:3000-3058`). Exige `hidden % 16 == 0` e `intermediate % 16 == 0` ("EXL3 trellis tiles are 16-wide", `exl3.py:2995-2999`). **Cada tile 16×16 = 256 pesos ocupa exatamente 16·K palavras int16 = 256·K bits → K bits/peso**.
  - Sharding TP: gate/up são *column-parallel* — corta `trellis` na dim 1 e `svh` na dim 0; down é *row-parallel* — corta `trellis` na dim 0 e `suh` na dim 0 (`exl3.py:661-676`). `suh` de gate/up e `svh` de down ficam replicados.
  - `process_weights_after_loading` valida marcador MCG `-877912083` em todos os experts (`exl3.py:3145-3153`), detecta se `suh` de gate e up são iguais (`_exl3_shared_w13_suh`, `exl3.py:3156-3158`) — permite reusar uma única Hadamard de entrada —, constrói um `LinearEXL3` por matriz sem reconstrução ("no BF16 expert reconstruct at load", `exl3.py:3210-3214`) e tabelas de ponteiros GPU para o kernel fusionado (`exl3.py:1279-1353`).
  - `apply` → `apply_exl3_fused_moe`: ordena tokens por expert, conta com `scatter_add_` (não `bincount`, que faria host sync e quebraria CUDA graphs, `exl3.py:1492-1496`), converte ativação para FP16 (`x2d.contiguous().half()`, `exl3.py:1498`), chama `exllamav3_ext.exl3_moe(…, k, k, k, ptrs…, limit)` com saída FP32 (`exl3.py:1355-1402`).
- Conceitos: trellis-coded quantization, Hadamard, column/row parallel, CUDA graphs, weight-only (ativação FP16, compute FP16 tensor core).
- Efeito: decode estruturado 62.9 tok/s ×1 (`GLM/README.md:299-303`).

**G3 — Codebook procedural MCG/mul1/3INST (kernel)**
- Arquivo: `GLM/extensions/cooperative_moe/tp3/vendor/exllamav3_ext/quant/codebook.cuh:26-123` (cópia vendorizada do exllamav3).
- Mecanismo (decodificação de um estado de 16 bits `x` para um FP16):
  - `cb==1` (**MCG**, usado pelo GLM): `x *= 0xCBAC1FED` (mod 2^32) → `lop3.b32 x, x, 0x8fff8fff, 0x3b603b60, 0x6a` (a tabela-verdade 0x6A = `(a & b) ^ c`, i.e. `(x & 0x8fff8fff) ^ 0x3b603b60`) → reinterpreta as duas metades de 16 bits como FP16 e **soma** (`codebook.cuh:67-75`). A máscara mantém o bit de sinal e os 12 bits baixos de cada metade, zera os 3 bits altos do expoente e o XOR os força para `011`, restringindo cada metade a |v| ∈ [0.125, 2): a soma de duas variáveis aproximadamente simétricas dá uma distribuição em sino (**interpretação** da aritmética; o QTIP descreve o 3INST como "sum of two mirrored exponential distributions approximates Gaussian").
  - `cb==0` (**3INST** do QTIP): LCG `x = x·89226354 + 64248484`, mesma máscara/XOR/soma (`codebook.cuh:59-66`).
  - `cb==2` (**mul1**, usado pelos tensores densos do turboderp): `x *= 0x83DCD12D`; soma dos 4 bytes via `__dp4a(x, 0x01010101, 0x6400)` (0x6400 é o FP16 1024.0 — somar um inteiro pequeno aos bits de mantissa produz 1024+soma); depois `h·(1/147.7) − 10.39` (`codebook.cuh:26-42, 76-89`). Soma de 4 bytes uniformes: média 510, desvio ≈147.8 → normaliza para ~N(0,1). Comentário: "dp4a … native on Blackwell where vabsdiff4 is emulated" (`codebook.cuh:80-81`).
- Conceitos: codebook "computado" (lookup-free), LCG/MCG, gaussianidade via soma (TCL), truques de bits de FP16.
- Efeito medido por mim (Apêndice A.9): sobre os 65,536 estados, MCG tem média ≈ 0, desvio ≈ 1.244, faixa ±3.949; mul1 média ≈ −0.002, desvio ≈ 1.000, faixa [−3.45, 3.35]. Valores dourados (estados 0..3 → 1.84375, 0.134521484375, −0.75927734375, 0.83984375) batem com `GLM/tests/test_exl3_kernel_lab.py:61-84`.

**G4 — Extração do estado do trellis ("bitshift trellis", L=16)**
- Arquivo: `GLM/extensions/cooperative_moe/tp3/vendor/exllamav3_ext/quant/exl3_dq.cuh:15-293`.
- Mecanismo: para o peso de índice `t` dentro do tile de 256, o estado é a janela de 16 bits que **termina** no bit `(t+1)·K` do fluxo do tile: `b0 = t·bits + bits − 16 + 256·bits`, lida com `funnelshift` entre duas palavras de 32 bits, índices `mod (bits·256/32)` → **circular dentro do tile** (tail-biting) (`exl3_dq.cuh:15-31`). Especializações por K: `dq8_aligned_4bits` para K=4 (8 pesos por duas palavras), `dq8_aligned_2bits`, `dq8_aligned_1bit`, `dq8<3,…>`, `dq4` para K=5/6/8, `dq2x2` para K=7 (`exl3_dq.cuh:254-293`).
- Conceitos: trellis bitshift (cada passo desloca K bits novos numa janela de L=16), paralelismo (cada peso depende só de uma janela contígua), tail-biting, layout tensor-core (a ordem `t` segue a permutação do fragmento MMA, `GLM/kernel_lab/exl3/reference.py:115-136`).

**G5 — Hadamard de 128 + escalas `suh`/`svh` (kernel)**
- Arquivos: `.../quant/hadamard_inner.cuh:93-484`, `.../quant/exl3_gemv_kernel.cuh:156-175, 374-401`, `GLM/overlay/exl3_fat_moe.cu:44-60, 92-123`.
- Mecanismo: Hadamard de 128 elementos por warp: 4 elementos por lane com butterfly local + 5 estágios `__shfl_xor_sync` (1,2,4,8,16) com troca de sinal, escala `r_scale = 1/√128 = 0.088388347648` (`hadamard_inner.cuh:17-44, 117-133`; `exl3_gemv_kernel.cuh:170`). Pré-escala por `suh` antes da Hadamard da entrada; pós-escala por `svh` depois da Hadamard da saída. Versão fusionada para MoE: `o ← in_had(silu(out_had(g)·svh_g) · out_had(u)·svh_u · suh_d)` com clamp `act_limit` (`hadamard_inner.cuh:281-413`).
- Matemática (do oráculo `GLM/kernel_lab/exl3/reference.py:189-231`): `y = x·W` com `W = diag(suh)·H·Ŵ·H·diag(svh)` (blocos de 128), onde `Ŵ` é a matriz decodificada do trellis (valores ~N(0,1)). Execução "streaming": `xh = H(x ⊙ suh)`; `y' = xh·Ŵ` tile a tile; `y = H(y') ⊙ svh`.
- Conceitos: incoherence processing (RHT), sinais aleatórios ±1, escalas por canal de entrada/saída como único lugar onde vive a magnitude (**interpretação**: o ABI não tem escala por tile; `suh`/`svh` são os únicos tensores float).

**G6 — Caminho GEMV/GEMM EXL3 (consumo do formato)**
- Arquivos: `.../quant/exl3_gemv_kernel.cuh:1-402`, `.../quant/exl3_kernel_map.cuh:56-73`.
- Mecanismo: pesos decodificados em registradores e alimentados a `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` (acumulação FP16 "folded to fp32 on a fixed cadence") (`exl3_gemv_kernel.cuh:13-14, 34-49`). Formas de tile: `(M,K,N) = 16×16×128`, `16×32×128`, `16×32×256`, `16×16×512` (`exl3_kernel_map.cuh:56-63`). Para M>144 o `LinearEXL3` reconstrói o peso FP16 e faz GEMM densa (`AUTO_RECONSTRUCT_THRESHOLD`, citado em `GLM/overlay/exl3.py:2306-2309, 2436-2441`).
- Conceitos: weight-only (W4A16 "à la EXL3"), decodificação em registrador vs reconstrução, crossover por M, tensor cores FP16 (não FP4!). Doc do lab: "NVIDIA's current CuTe implementation directs SM120/121 kernels to warp-level `mma.sync`, not datacenter-Blackwell `tcgen05`" (`GLM/docs/exl3-sm121-kernel-lab.md:103-107`).

**G7 — `Glm53DenseFp8Method`: FP8 E4M3 por canal + Marlin (W8A16)**
- Arquivos: `GLM/overlay/exl3.py:1895-2281`; `GLM/overlay/patch_dense_fp8.py:1-109`.
- Problema: projeções densas BF16 são ~metade dos bytes lidos no decode; reduzir para 1 byte/peso sem calibração.
- Mecanismo: carrega BF16 normalmente; em `process_weights_after_loading`: `scales = |W|.amax(dim=1).clamp(min=1e-12)/448` (**per output channel**), `fp8 = (W/scales).clamp(±448).to(float8_e4m3fn)`, escala guardada em BF16 (`layer.orig_dtype`), `prepare_fp8_layer_for_marlin(size_k_first=False)`, e `apply` → `apply_fp8_marlin_linear` (`exl3.py:2123-2144, 2267-2281`). `patch_dense_fp8.py` troca dois trechos dos construtores KDA/MLA para que eles **mantenham** `quant_config` quando o método é `exl3` (stock passava `quant_config=None` porque "MLA projections are BF16 in checkpoint") (`patch_dense_fp8.py:26-42`).
- Conceitos: quantização simétrica por canal, E4M3 max 448, weight-only (ativação BF16 — Marlin desquantiza no registrador), sem calibração.
- Efeito: "~11 ms less per step on everything (+10 % on counting, prose +12–19 % alone, +37 % on hard prose stacked with adaptive-k). PROVISIONAL: … KL proxy vs stock 0.002–0.013 nats/position, argmax agreement 94–100 %" (`GLM/README.md:344`). Adoção de `all` em 2026-09-26: decode ms/cycle −4.67% [−5.11, −4.06] / −4.91% [−5.42, −4.44]; tok/s +5.2 / +4.3%; "FP8=all's text KL is 25–30 % above every stock pair"; ΔNLL −0.004…+0.0005 nats/token (`GLM/docs/adopt-fp8-all-kda-bf16-2026-09-26.md:24-43`).

**G8 — KDA BF16 large-M (retenção da matriz FP8 lógica em BF16)**
- Arquivos: `GLM/overlay/exl3.py:2004-2248`; `GLM/docs/kda-bf16-large-m.md`.
- Problema: Marlin W8A16 é ótimo em M pequeno mas lento em prefill (M≥768).
- Mecanismo: na carga, materializa `W_bf16 = fp8 × scale_armazenada` (a mesma escala BF16 que o Marlin multiplica), em blocos de 512 linhas; "The product of an e4m3 value (<=4 significant bits) and a bf16 scale (<=8 significant bits) needs at most 12 significant bits, so the fp32 intermediate is exact and the single cast to `out_dtype` rounds once" (`exl3.py:2046-2052`). Dispatch: M>512 → `F.linear(x_bf16, W_bf16)` (cuBLAS); senão Marlin (`exl3.py:2250-2281`). Probe cuBLAS na carga com M=2 (`exl3.py:2223-2238`).
- Por que não W8A8: "activation QDQ alone explains the ~0.0266 layer error vs stock Marlin, while the `_scaled_mm` backend contributes only ~0.002-0.003" (`docs/kda-bf16-large-m.md:45-54`).
- Efeito: rel RMSE vs Marlin: W8A8 antigo ~0.0266; fp8×fp32 scale 0.002938; **fp8×scale bf16 armazenada 0.002526**; Marlin vs peso lógico FP32-exato 0.00005 (`docs/kda-bf16-large-m.md:58-66`). Kernel: M=7168 Marlin ~27.21 ms, W8A8 ~12.57 ms, BF16 ~7.36 ms. Serving: 16k TTFT 11.41→10.12 s (−11.4%), 100k 70.86→62.12 s (−12.3%), misto −14.1%. Memória: +3.29 GiB/rank medidos (`docs/kda-bf16-large-m.md:91-111`).

**G9 — Dense EXL3 (`[dense-exl3]`, portado de Alexbob0/vcruz305)**
- Arquivos: `GLM/overlay/exl3.py:2284-2970`; `GLM/tools/dense_overlay.py`; `GLM/tools/stage_dense_h3.py`; `GLM/tools/pack_profile.py`.
- Problema: servir também os lineares não-roteados em EXL3 (menos bytes), mantendo shards BF16 onde o serving precisa do tensor cru.
- Mecanismo:
  - `dense_overlay.py` **não quantiza nada**: faz *range reads* HTTP dos tensores `trellis/suh/svh/mul1|mcg` do quant turboderp (`turboderp/GLM-5.3-Flash-exl3@4.05bpw`) apenas para os lineares densos (~5.3 GB), escreve um safetensors de overlay, reescreve o índice e injeta `quantization_config.non_routed_exl3` (`dense_overlay.py:19-34, 48-76, 354-377`). Mapeia fusões: `q_proj/k_proj/v_proj` do Hub vêm fundidos em `qkv_proj` e são fatiados por bloco (`dense_overlay.py:71-76, 314-333`); `in_proj_qkvbfg_a` mantém shards 3,4,5 (`b_proj, f_a_proj, g_a_proj`) em BF16 (`dense_overlay.py:69-70`). Opção `--lm-head` traz o lm_head **K6 mul1** (`dense_overlay.py:282-299`).
  - `Exl3LinearMethod.create_weights`: trellis `[in/16, Σout/16, 16·K]`, `suh[n_shards, in]`, `svh[Σout]`, marcadores `mcg`/`mul1` por shard, e um `weight` BF16 para os shards declarados `bf16_shards` (devem ser a cauda contígua) (`exl3.py:2510-2626`). Recusa TP=3 ("dense EXL3 does not support TP=3 trellis/head padding", `exl3.py:2539-2540`). Exige vocabulário 64-alinhado para lm_head EXL3 (`exl3.py:2562-2577`).
  - Forward via custom op `torch.ops.vllm.dense_exl3_forward` (opaco ao dynamo, graph-capturável): shards EXL3 produzem FP16 ("the fast bitcoder path is fp16-only") e a cauda BF16 roda um GEMM no dtype da ativação (`exl3.py:2284-2426`).
  - `_dense_exl3_warmup_autotune`: aquece cada GEMM por forma × bucket de linhas {1,2,4,8,16} **antes** da captura de CUDA graph, porque o autotuner cooperativo sincroniza o stream e "inside graph capture that deadlocks" (`exl3.py:2429-2482`).
  - Retenção BF16 prefill (`GLM53_DENSE_EXL3_PREFILL_BF16`): copia `get_weight_tensor()` (Hadamards já aplicados) → M>144 roda um `F.linear` BF16; medido que o gap vem "the activation passes around the EXL3 reconstruct path (per-shard had_r_128 in/out, casts, cat), not the weight reconstruct" (`exl3.py:2294-2310`).
- Efeito: sem A/B publicado contra o default ("has not yet been A/B measured", `GLM/README.md:110-111`).

**G10 — Quantização do draft DFlash2 para EXL3 6 bpw**
- Arquivos: `GLM/tools/dflash2_exl3_quant.sh`, `GLM/tools/dflash2_exl3_convert.py`, `GLM/tools/dflash2_exl3_package.py`.
- Mecanismo: conversor stock do exllamav3 (fork MiaAI-Lab @ `63b32f0`, v1.4.2) com dois ajustes em processo: (1) `k_proj`/`v_proj` recebem o sentinela **K=16 = não quantizado** porque o serving constrói o peso fundido de context-KV a partir das linhas K/V do qkv, que "must remain a plain tensor" (`dflash2_exl3_convert.py:6-10, 53-71`); (2) `kernel_projection` do DynConv ganha `qmap` para entrar no orçamento. **Sem calibração**: "DFlash2 models carry caps["uncalibrated_quantize"] = True, so the converter skips calibration forwards entirely (state=None) and quantizes each linear against the synthetic Hessian (init_H_data)" (`dflash2_exl3_convert.py:23-26`). Empacotamento: remove `su`/`sv` (tensores de treino), mantém bytes BF16 originais para tudo não quantizado, escreve `quantization_config` com `scope=dflash2_draft`, `bits=6`, `head_bits=16`, `bf16_shards=[1,2]` para qkv (`dflash2_exl3_package.py:9-21, 99-148`). Reprodutível byte a byte (`GLM/README.md:96-100`).
- Conceitos: sentinela K16, Hessiano sintético vs calibrado, head bits.
- Efeito: ~25 min de build no head (`GLM/README.md:85-87`).

**G11 — Validador de perfil de pack (`tools/pack_profile.py`)**
- Mecanismo: sem baixar pesos, lê cabeçalhos safetensors e exige para cada módulo EXL3: trellis `I16` com última dim `16·bits`, `suh` shape `[in]`, `svh` shape `[out]`, exatamente um de `mcg`/`mul1`, bits ∈ {2..6} (`pack_profile.py:54-63`). Draft deve ter 5 camadas, 4096 de largura, `bits=6`, trellis `shape[-1]==96` (=16·6) e K/V BF16 (`pack_profile.py:118-150`).

**G12 — Kernels "fat expert" E2/E3 (consumidores do formato no prefill)**
- Arquivos: `GLM/overlay/exl3_fat_gemm.cu/.cuh` (E2), `GLM/overlay/exl3_fat_moe.cu/.cuh` (E3), `GLM/overlay/patch_exl3_fat_kernel.py`, `GLM/overlay/build_exl3_fat_moe_ext.py`.
- Mecanismo E3: "B tiles (16x16 K4/MCG trellis tiles, 64 int16 words each) are staged through shared memory and dequantized in registers by the warp that owns that 16-column block, once per 16 K, and reused across all M blocks of the tile" (`exl3_fat_moe.cu:12-20`); chama `dq_dispatch<4,1>` (K=4, cb=1=MCG) e `ptx_mma_m16n8k16` (`exl3_fat_moe.cu:237-250`); gather + Hadamard de entrada com `suh` do expert; epílogo gate/up aplica Hadamard + `svh` + SwiGLU + `suh` do down (`exl3_fat_moe.cu:92-123, 322-376`). Restrição `K % 32 == 0`.
- Efeito: layer MoE de 7168 tokens 77–91 ms (E2) → 31 ms (E3); prefill frio +37–45% (`GLM/README.md:262-281`).

**G13 — Thin-decode K4/N256 (`patch_exl3_decode_pipeline.py`)**
- Mecanismo: adiciona kernels `glm53_exl3_moe_fast_kernel<4,256>` com 1 estágio de registradores (vs 3) e 8 de smem (vs 3); reuso da transformada de entrada quando os ponteiros `suh` de gate/up são idênticos (`GLM/docs/sm121-perf-paths.md:15-31`).
- Efeito: SASS "fast 127 regs / 16 B frame / 6 STL / 0 LDL vs stock 128 regs / 88 B frame / 37 STL / 39 LDL"; latência de camada +8% a +17%; paridade 119/119, RMSE relativo ~1e-8 (`docs/sm121-perf-paths.md:44-56`).

**G14 — Cooperative MoE (decode) e `kernel_lab/exl3`**
- `GLM/extensions/cooperative_moe/`: especialização do kernel two-stage do turboderp para H=4096, I local 1024, top-k 8, **"Uniform K4 MCG (codebook cb=1), no mul1"**, 1–32 linhas (`extensions/cooperative_moe/README.md:24-33`). Medido: 37.4 prose / 80.0 structured (vs 32.1 / 62.9).
- `GLM/kernel_lab/exl3/`: contrato ABI versionado (`abi.py:17-24`: `TILE_SIZE=16`, `MCG_MULTIPLIER`, bits 1..8), `effective_bpw = 8·(trellis_bytes + 2·(in+out) + 4)/(in·out)` (`abi.py:74-84`), oráculo NumPy (`reference.py`). **Observação**: o oráculo `decode_tiles` aplica `decode_mcg` sobre o símbolo K-bit desempacotado (`reference.py:139-145` + `unpack_trellis` com máscara K-bit em `reference.py:73-82`), não sobre a janela de 16 bits do kernel CUDA; ele valida bit-order de empacotamento e paridade materializado-vs-streaming, e o decodificador MCG isolado foi comparado com o CUDA "for all 65,536 states" (`docs/exl3-sm121-kernel-lab.md:135-139`). Para a semântica real do trellis use `exl3_dq.cuh`.

**G15 — Layout de KV `fp8_ds_mla` para NoPE MLA (656 B)**
- Arquivo: `GLM/Dockerfile:1-60, 110-366`.
- Problema: GLM-5.3-Flash é NoPE MLA (`qk_rope_head_dim=0`, `kv_lora_rank=512`); no SM12x o único backend sparse-MLA é `FLASHINFER_MLA_SPARSE_SM120`, cujo registro empacotado é "512 NoPE + 16 B scales + 128 B RoPE (656 B)"; o kernel recusava ("pe_dim must be 64 for fp8_ds_mla").
- Mecanismo: zero-pad do latente 512→576 (RoPE pad de 64 zeros, `self.rope_pad = 64`; `q` também é pad): "A zero RoPE block adds nothing to the QK dot, softmax is unchanged, and the value is taken from the 512 NoPE region (d_v=512), so the result is exact. It costs 656 B/token instead of ~528 (~24% more DSA KV over 11 DSA layers)" (`GLM/Dockerfile:24-29`). (O README diz "Target MLA (12 layers)", `README.md:677`; o Dockerfile diz "11 DSA layers" — números do repo, não reconciliados.)
- Conceitos: layout FlashMLA/DSV3.2 FP8: 512 bytes FP8 E4M3 + 4 escalas FP32 (uma por 128 valores) + 64×BF16 de RoPE (`DS/recipe/overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py:157-159`: "V32 fp8_ds_mla packed: 656 B/token (512 NoPE + 16 inline FP32 scales + 128 BF16 RoPE)").
- Efeito: pool de KV 1,754,237 tokens (18.67 GiB) na boot de 2026-08-29; 1,572,073 com 14 GiB; 1,233,779 com 11 GiB (`README.md:444, 686-693`; `docs/adopt…:53`).

**G16 — Outras restrições numéricas no GLM**
- `GLM/README.md:751-752`: "NVFP4 KV is not available here. FlashInfer's SM12x NVFP4 kernels are dense MHA, not sparse MLA. Do not confuse that with NVFP4 weights".
- `GLM/README.md:1999-2001` (não fazer): `--moe-backend marlin`, pesos NVFP4, `--kv-cache-dtype nvfp4` ou bf16.
- TP3: routed `moe_intermediate_size` 2048 não é paddado porque "EXL3 trellis is packed 2048-wide" → usa EP (96 de 288 experts por rank) (`README.md:1475`, `overlay/tp3/README.md:13`).
- `tests/bench_numerical_calibrated.py`: painel "calibrated numerical panel for quantization/kernel A/B": prompt_logprobs teacher-forced; classes CONF/TIE/UNSTABLE com margem 0.25 nats; KL padronizado pelo KL máximo stock-vs-stock por posição; ≥4 capturas stock (`tests/bench_numerical_calibrated.py:1-71`).

### 3.2 Qwen3.8-Flash-Next NVFP4

**Q1 — `patch_qsa_fp8_kv.py` (KV FP8 E4M3 no QSA)**
- Arquivo: `Qwen/files/patch_qsa_fp8_kv.py` (vendorizado de MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark; abordagem de lancelind).
- Problema: kernels Triton QSA só aceitavam `["auto","bfloat16"]` e lançavam "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache".
- Mecanismo: (1) reinterpreta a alocação `uint8` como `float8_e4m3fn` sem cópia (`_qsa_as_fp8`, recusa ler bytes quantizados como BF16, `patch_qsa_fp8_kv.py:84-101`); (2) no kernel de seleção (MQA/indexer): `keys = keys.to(query.dtype)` e **aplica o escalar `k_scale` depois do dot** (`scores *= tl.load(k_scale_ptr)`, `:132-139`); (3) no sparse GQA: upcast K e V, `scores *= k_scale` após `tl.dot`, e `normalized_output *= v_scale` no fim (`:166-182`); (4) `KV_QUANT_MODE` é `tl.constexpr` → o caminho BF16 compila idêntico; (5) neutraliza a recusa do `FlashAttentionImpl` pai (que nunca executa para essas camadas) passando `"auto"` ao construtor pai (`:314-356`); (6) aceita `fp8`/`fp8_e4m3`.
- Matemática: `softmax(q·(s_k·K̂)ᵀ)·(s_v·V̂) = softmax(s_k·(q·K̂ᵀ))·V̂·s_v` — o escalar sai do produto: "algebraically equivalent before rounding, avoids materialising FP32 dequantisation tiles, and lets FP8 use the same tile width as BF16 on GB10" (`:9-12`).
- Conceitos: FP8 per-tensor, fatoração de escalas escalares, sparse attention sensível à quantização de K.
- Risco declarado: "on the reference implementation a long-reasoning benchmark regressed from 6/6 to 2/6 with FP8 KV. This is sparse attention -- quantising keys perturbs which blocks the indexer selects" (`:19-23`). Aqui: 12/12 no `reasoning_check` (`README.md:324`).
- Efeito: 3,652,200 vs 2,131,159 tokens; 114,060 vs 67,229 tokens/GiB (1.70×, não 2× porque indexer/compressor e estado GDN seguem BF16/FP32) (`README.md:316-328`). Re-medição: −5.3% throughput médio (−1.4% prose, −9.2% code), 1.80× KV (`CHANGELOG.md:129-133`).

**Q2 — `patch_qsa_fp8_kv_v030.py` (backport vllm#55557)**
- Mecanismo: `IS_FP8: tl.constexpr`; a escala de K é pré-multiplicada no `softmax_scale` do host (`score_scale = softmax_scale * log2(e)`, `:76`); "e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK measured slower here and less accurate)"; V desquantizado para **FP16** no PV-dot: "P <= 1 (online softmax) so fp16 has the range, its wider mantissa is more accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster" (`:84-110`); escala de V dobrada no normalizador (`inv_normalizer = output_scale / max(normalizer,…)`, `:117`).
- Efeito: 2.64M tokens, NLL 1.328/1.334, needles 3/3 a 200k, decode ~5% mais lento (`CHANGELOG.md:26-28`).

**Q3 — Checkpoint híbrido FP8-dense (`files/fp8dense/`)**
- Arquivos: `make_fp8_dense_checkpoint.py`, `build.sh`, `verify_fp8_dense_checkpoint.py`, `compute_quant_stats.py`, `test_quant_config_resolution.py`; overlays `files/overlay/{modelopt,hyperconnection,model,mtp}.diff`.
- Problema: "the bf16 dense projections (GDN, attention, HyperConnection, shared expert, lm_head) are ~6.9 of the ~9.2 GiB streamed per step" (`make…:5-8`); o FP8 oficial da Qwen **não** quantiza o denso (`fable5-1-report.md:114-119`).
- Mecanismo: conversão CPU em streaming (blocos de 16 MiB de linhas): `quantize_rows`: `amax = |x|.amax(dim=1)`, `scale = (amax/448).clamp_min(1e-12)`, `q = (x/scale).clamp(±448).to(e4m3)` (`make…:142-148`); grava `weight` `F8_E4M3` + `weight_scale` `F32 [out]`; hard-link de shards de experts/PLE; marca 591 lineares como `FP8_PER_CHANNEL_PER_TOKEN` e experts como `NVFP4` group 16; `quant_method="modelopt"`, `quant_algo="MIXED_PRECISION"` em `config.json` e `hf_quant_config.json` (`make…:261-301`). Mantém BF16: embeddings, router, shared-expert gate, indexer QSA, GDN `in_proj_a/b`, conv1d, normas, PLE, MTP, visão (`make…:58-78`). Overlay do loader: `ModelOptFp8PcPtLinearMethod` + `ModelOptFp8PbWoLinearMethod`, prefixos alternativos, e HyperConnection/lm_head passam a receber `quant_config` em vez de `None` (`overlay/*.diff`).
- Porque per-channel e não block-128: "(a) it needs no calibration data, (b) HyperConnection's rank-320 projections and the TP-sharded shared expert (320 rows) are not 128-divisible, and (c) vLLM's `ParallelLMHead` loader shards block scales incorrectly … while per-channel scales shard like the weight" (`fable5-1-report.md:121-124`).
- Kernel selecionado: `CutlassFP8ScaledMMLinearKernel` (W8A8, ativação dinâmica por token); "Marlin is skipped on cc ≥ 89, FlashInfer's needs per-tensor scales, B12X needs static per-tensor"; cubins `cutlass_scaled_mm_sm120_fp8` não-`a` executam no cc 12.1 (`fable5-1-report.md:448-451`).
- Efeito: rel RMSE min/mediana/max 0.0240/0.0262/0.0268; cos ≥ 0.9996; 30/30 resoluções corretas; verificação 0 problemas; ganho **analítico** +40–60% batch-1, não medido (`fable5-1-report.md:108-112, 441-445`; `README.md:516-530`). Verificador marca "HIGH ERROR" se rel_err > 0.06 (`verify_fp8_dense_checkpoint.py:86-87`).

**Q4 — `patch_modelopt_mxfp8.py` (fallback MXFP8 → BF16)**
- Problema: `FlashInferCutlassMxfp8LinearKernel`/`mm_mxfp8` só aceita `N ≥ 128, N % 32 == 0, K ≥ 128, K % 32 == 0` (medido no sm_121); `linear_attn.in_proj_a/b` `[48, 2560]` (N<128) e `visual.blocks.*.mlp.linear_fc1` `[4304, 1152]` (4304 % 32 = 16) falham (`patch_modelopt_mxfp8.py:5-19`).
- Mecanismo: em `create_weights` checa o `(N,K)` **pós-split TP** e troca o kernel por `EmulationMxfp8LinearKernel`, que "dequantizes MXFP8 -> BF16 once at load time"; todo `visual.*` vai para emulação por prefixo ("keeps the load-time BF16 dequant from OOMing") (`:21-30, 82-103, 139-147`).
- Efeito: ≈17 MB extras BF16 (`README.md:580-584`).
- **Observação**: o docstring cita `RadixArk/Qwen3.8-Flash-Next-NVFP4` como tendo esses shapes MXFP8, enquanto `.env.sample:26` descreve RadixArk como "BF16 attention"; `local-inference-lab` é o descrito como "MXFP8 attention" (`.env.sample:25`). A tabela de backends do README diz que no checkpoint default "MXFP8 linear | FlashInferCutlassMxfp8LinearKernel, except …" (`README.md:480`). Os textos não são totalmente consistentes sobre qual checkpoint carrega quais lineares MXFP8.

**Q5 — `patch_modelopt_fp8_block_moe.py` (experts MTP FP8 128×128)**
- Problema: `ModelOptMixedPrecisionConfig.get_quant_method` só constrói `RoutedExperts` para `FP8/NVFP4/W4A16_NVFP4/MXFP8`; `FP8_BLOCK_SCALES` → `None` → MoE **silenciosamente não quantizado** → erro `no parameter 'w2_weight_scale_inv'` ~7 min depois (`:1-19`).
- Mecanismo: ramo novo `if quant_algo in ("FP8_BLOCK_SCALES", "FP8_PB_WO")` → `Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[g,g]))`, com `g` lido do `group_size` do checkpoint — "a wrong block shape does not fail loudly, it silently applies misaligned scales" (`:30-62, 78-100`).
- Efeito: MTP=3 funciona: 24.5 → 52.1 tok/s, aceitação 72.8% (89/74.5/60% por posição) — "a wrong block shape would show up as near-random acceptance" (`README.md:700-712`).

**Q6 — `patch_checkpoint_config.py` (aliases MTP + `FP8_PB_WO`)**
- Problema: vLLM constrói o MTP em `mtp.layers.48` e casa `quantized_layers` por string exata; o checkpoint nvidia só tem `mtp.layers.0`. O config.json (rev `fc694b54`) diz `FP8_PB_WO` e o `hf_quant_config.json` diz `FP8_BLOCK_SCALES` (`:1-33`).
- Mecanismo: adiciona aliases de índice absoluto em `quantized_layers` e `config_groups.targets` de **ambos** os arquivos (bind-mount; o cache HF não é modificado); reescreve MTP experts `FP8_PB_WO → FP8_BLOCK_SCALES` (`:89-105`).

**Q7 — PLE: `patch_ple_layer.py` + `detect_ple_dtype.py`**
- Problema: a tabela PLE de 51 GB é FP8 (shards + uma `weight_scale` global) mas está *excluída* do config NVFP4; o resolver stock só usa FP8-PLE se o checkpoint inteiro for `Fp8Config`, então constrói BF16 (~102 GB) e quebra (`README.md:550-555`).
- Mecanismo: porta o dispatch do vLLM PR #53899: `ple_embedding_dtype` `"float8_e4m3fn"`/`"fp8"` → `Qwen3_8FlashNextPLEFp8EmbeddingMethod` (linhas FP8 + escala global); `"nvfp4"` → `Qwen3_8FlashNextPLENVFp4EmbeddingMethod` (linhas `uint8 [rows, dim/2]` + escalas `float8_e4m3fn [rows, dim/16]` + `weight_scale_2` FP32 escalar; a busca retorna `cat(codes, scales.view(uint8))` e a dequant usa LUT `_FP4_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6)`, sinal no bit 3, nibble baixo primeiro) (`patch_ple_layer.py:1-167`). `detect_ple_dtype.py` recupera o dtype do `config_groups` com `targets` contendo `ple_embedding.ngram_embedding`: `num_bits 8 → float8_e4m3fn`, `4 → nvfp4` (`detect_ple_dtype.py:28-49`).
- Conceitos: quantização de tabelas de embedding (lookup de linhas, não GEMM), escala per-tensor vs NVFP4.

**Q8 — Tabela PLE empacotada p/ offload (`build_ple_packed_table.py`)**
- Mecanismo: escreve `[total_rows, 90]` uint8 = códigos 4-bit + escalas FP8 lado a lado, memory-mapped; tabela de 26.8 GiB em page cache (`build_ple_packed_table.py:1-11, 53-79`). (**Interpretação**: 90 = d/2 + d/16 ⇒ d = 160.)

**Q9 — `MAMBA_SSM_CACHE_DTYPE=bfloat16`** — estado recorrente GDN em BF16 em vez de FP32 do checkpoint (ver 2.2). Portado do single-Spark.

**Q10 — Restrição de divisibilidade NVFP4 (visão)** — `MM_ENCODER_TP_MODE=data` porque K por rank precisa ser múltiplo de 16 (bloco NVFP4) (`.env.sample:89-92`).

### 3.3 DeepSeek-V4-Flash DSpark

**D1 — Issue #22: `nvfp4_ds_mla` → kernel FP8**
- Arquivos: `DS/patches/hotfix-nvfp4-ds-mla-issue22.sh`, `DS/patches/fix-nvfp4-ds-mla-long-context.patch`, `DS/docs/PATCHES.md:169-204`.
- Problema: com `nvfp4_ds_mla`, decode caía a ~1 tok/s a 600K+ vs ~17 tok/s com `fp8_ds_mla` (curto contexto ~66 tok/s intacto).
- Mecanismo: `use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")` em `flashmla_sparse.py` — "The 584-byte KV layout is identical for both dtypes on DSV4; only the kernel dispatch differs".
- Efeito: ~16× mais rápido em contexto longo.

**D2 — Plumbing `nvfp4_ds_mla` + backend b12x MXFP4 (`official-main-b12x-nvfp4-python.patch`)**
- Mecanismo: adiciona `"nvfp4_ds_mla"` ao `CacheDType`; troca o erro "nvfp4 KV cache is not supported with MLA" por conversão automática `nvfp4 → nvfp4_ds_mla`; `_resolve_dsv4_kv_cache_dtype` devolve `("nvfp4_ds_mla", torch.uint8)` exigindo o layout padded; `get_kv_cache_shape` → `(num_blocks, block_size, 584)`; `alignment=584`; `get_kv_quant_mode` mapeia para `KVQuantMode.NVFP4`; `SlidingWindowMLASpec.real_page_size_bytes` = 584/token para ambos. MoE: `Mxfp4MoeBackend.B12X_MXFP4`, `"flashinfer_b12x"` → B12X, `round_up(hidden/intermediate, 128)`, pesos MXFP4 passados sem conversão ("B12X prepares its W4A16 packed representation from the original native MXFP4 tensors") e novo `experts/b12x_mxfp4_moe.py` (785 linhas); envs `VLLM_USE_B12X_MOE`, `VLLM_B12X_W4A16_*` (`patch`: hunks em `config/cache.py`, `config/vllm.py`, `envs.py`, `fused_moe/oracle/mxfp4.py`, `models/deepseek_v4/{attention,sparse_mla}.py`, `flashmla_sparse.py`, `kv_cache_interface.py`).

**D3 — Histórico de layouts "NVFP4 KV": 416 B → 584 B (Stage A/B/C) e 352 B**
- Arquivos: `DS/recipe/nvfp4/Dockerfile.stage-a` (416 B em `kv_cache_interface`), `Dockerfile.stage-b` ("Probe layout from the GLM-5.2 NVFP4 sparse-MLA path", `head_bytes = 416`), `Dockerfile.stage-c` ("keep DeepSeek V4's proven 584-byte cache envelope so hybrid MLA/SWA grouping can proceed"); `DS/Dockerfile.gb10-dsv4-dspark:129-151` (lane alternativo vLLM 0.26: **352 B/token = 256 B E2M1 NoPE + 32 B escalas E4M3 bloco-16 + 64 B RoPE FP8**, `kStride=352`).
- Aritmética (**interpretação** para 416): 512 dims × 4 bits = 256 B + 512/16 = 32 escalas E4M3 (32 B) + 64 RoPE × BF16 (128 B) = 416 B. 352 = 256 + 32 + 64 (RoPE em FP8). README: "This is the Stage C padded NVFP4 path (584-byte sparse-MLA envelope via `nvfp4_ds_mla`). It is not the abandoned 416-byte 'true layout' experiment" (`DS/README.md:600-601`).
- Relação com o GLM: o README do GLM menciona que a imagem SM120 B12X do brandonmusic usa "calibrated NVFP4 MLA KV" (`GLM/README.md:33-34`) — o layout NVFP4 MLA de 416 B é o provável "GLM-5.2 NVFP4 sparse-MLA path" citado pelo Stage B (**interpretação**).

**D4 — Design de um KV FP4 "de verdade" (`docs/CLAUDE/item8-fp4-kv-design.md`)**
- Restrição: "every KV group must have the same page size in bytes. Today all groups land on 64 rows × 584 B = 37,376 B"; opções que dividem 37,376: 584×64, 292×128, 146×256 (`:15, 25-26`).
- Opções: **A** 292 B = 448 NoPE E2M1 (224 B) + 4 UE8M0 por 128 dims (4 B) + 64 RoPE FP8 (64 B) (recomendada); **B** 280/292 B com RoPE em FP4 + escalas MXFP4 por 32 (maior risco a 1M); **C** 366→368 B com RoPE BF16 (não cabe na página); **D** DCP-2 (sem perda numérica, não suportado no DSV4) (`:28-37`). Capacidade esperada A: "+65 % tokens at the physical level; +40–50 % in the allocator headline" (`:39-41`). Fable report alternativo: "A real NVFP4 row (448 dims → 224 B + 28 B scales + 128 B RoPE + 8 B ≈ 390 B) would give +50 % KV tokens" (`DS/docs/CLAUDE/fable5-1-report.md:346-348`).
- Custo físico por token hoje: 21 camadas C4 × 584/4 = 3.07 KB, 20 C128 × 584/128 = 0.09 KB, indexer 21 × 132/4 = 0.69 KB (replicado); janelas SWA ≈ 3.4 MB/seq (`item8:17-21`).

**D5 — `quantize_and_insert_k_cache` / `dequantize_and_gather_k_cache` (UE8M0 FP8 por 64)**
- Arquivo: `DS/recipe/overlay/vllm/models/deepseek_v4/common/ops/cache_utils.py:7-388`.
- Mecanismo: por token, K `[512]` BF16 → 448 dims NoPE em 7 blocos de 64: `amax = max(|x|, 1e-4)`, `scale = 2^ceil(log2(amax/448))` (**arredonda a escala para cima em potência de 2**), `x_fp8 = clamp(x/scale, ±448).to(float8e4nv)`, escala armazenada como byte `exponent + 127` (clamp 0..255); 8º byte de escala = pad 0; 64 dims RoPE copiadas em BF16 (`cache_utils.py:87-139`). Layout do bloco de 64 tokens: `[0, 64·576)` dados (448 FP8 + 128 B BF16 por token), `[64·576, 64·576+64·8)` escalas (8 B por token), padding (`:46-51, 152-158`). Dequant: `scale = 2^(stored−127)`, `x = fp8·scale` → BF16 (`:273-300`). Aritmética de ponteiros em int64 porque `block_idx·block_stride` estoura 2^31 com ≥57K blocos (`:69-71, 244-246`).
- Conceitos: UE8M0 (potência de 2 ⇒ multiplicar = somar ao expoente), escala "ceil" (sem overflow, perde até 1 bit de faixa), separação NoPE/RoPE.

**D6 — MXFP4 no K-cache do indexer (`hotfix-vllm-mxfp4-indexer-cache.py`)**
- Problema: o vLLM do Anemll 0.1.1 tem K-cache MXFP4 do Lightning indexer completo, mas o builder asserta "requires Blackwell datacenter GPUs (sm_10x)".
- Mecanismo: amplia o assert para `is_device_capability_family(120)`; verificação por SHA-256 do arquivo stock (`02505c6c…`) → patched (`bfb0376d…`), replace atômico, fail-closed (`:46-96`). Habilitado com `--attention-config {"use_fp4_indexer_cache":true}`.
- Efeito esperado: "keeps the FP8-size 132 B/row indexer K allocation and uses the first half for FP4 … halves indexer K read bytes per scored key"; "MXFP4 (e2m1 values, per-32 UE8M0 scales) quantizes indexer scores — top-k selection, never attention values" (`DS/docs/PATCHES.md:1248-1260`). (**Interpretação** da aritmética: 128 dims FP8 + 4 B escala FP32 = 132 B; em MXFP4: 64 B + 4 escalas E8M0 = 68 B.)

**D7 — Alias de cabeçalhos DeepGEMM sm121**
- Arquivo: `DS/patches/hotfix-deepgemm-sm121-mqa-header-alias.sh`.
- Problema: DeepGEMM gera `sm121_fp8_mqa_logits<…>` no GB10 mas só traz `sm120_*`; cache JIT quente mascarava o bug.
- Mecanismo: cria `sm121_{fp8,fp4}_{,paged_}mqa_logits.cuh` com `#include <deep_gemm/impls/sm120_X.cuh>` + `#define sm121_X sm120_X` ("SM120 and SM121 share the Blackwell consumer ISA") (`:1-55`).

**D8 — Backend MoE b12x para MXFP4 nativo (W4A16)**
- Arquivo: `DS/recipe/overlay/vllm/model_executor/layers/fused_moe/b12x_moe.py`.
- Mecanismo: `B12xExperts` asserta `weight_quant_dtype == "mxfp4"` (`:427-437`), `_source_format() = "fp4_e8m0_k32"`, layout `w31` (`:446-452`), só `sm_12x` com b12x instalado (`:484-486`), esquema `(kMxfp4Static, None)` = pesos MXFP4, ativação não quantizada (`:494-497`), `expects_unquantized_inputs=True`; `prepare_b12x_fp4_moe_weights(... w1_global_scale=unit, a1_gscale=unit, prepare_w4a16=True, reuse_input_storage=True)` e libera escalas/pesos fonte após preparar (`:548-609`); `quant_mode="w4a16"` em todas as chamadas (`:682, 741, 774`). O seletor de tiles W4A16 aceita override (`scale_format` default `"e4m3_k16"` no kernel b12x, `:216, 236`).
- Por que W4A16: "the NVFP4 kernels currently support only source_format='modelopt_nvfp4' → any non-modelopt FP4 source must use W4A16" (`DS/docs/CLAUDE/fable5-1-report.md:274-276`). Alternativas propostas: MXFP4×MXFP8 via `mma … kind::mxf8f6f4` ("the checkpoint's format is the hardware-native one") ou re-quantizar para ModelOpt NVFP4 — "prefill +30–60 %" estimado (`:295-300`).

**D9 — Carga de escalas E8M0 e FP8 block (model.py)**
- `DS/recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py`:
  - Experts MegaMoE: `w13_weight [E, 2I, H/2] uint8` (2 valores E2M1/byte), `w13_weight_scale [E, 2I, H/32] uint8` (E8M0 por 32), idem w2 (`:355-400`). `_ue8m0_uint8_to_float(sf) = (sf.int32 << 23).view(float32)` (coloca o byte no campo de expoente do FP32 ⇒ `2^(e−127)`) (`:452-453`). MegaMoE exige SM100 (não roda no GB10) (`:463-464`).
  - "E8M0 scales are stored as float8_e8m0fnu in checkpoints but the MoE param is uint8. copy_() would do a numeric conversion (e.g. 2^-7 → 0), destroying the raw exponent bytes" → `loaded_weight.view(torch.uint8)` (`:1990-1997`).
  - Mapeador: experts FP4 usam `w{1,2,3}_weight_scale`; lineares FP8 e shared experts usam `weight_scale_inv` (block scales) (`:2073-2090`).
  - `scale_fmt = config.quantization_config["scale_fmt"]` (`:1025`); `VLLM_USE_B12X_WO_PROJECTION` marca `wo_a.weight_scale_inv.format_ue8m0 = True` para preservar escalas UE8M0 (`:1001-1022`).
- `DS/recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py:218, 516-539`: saída da atenção do draft é quantizada para FP8 (`fused_inv_rope_fp8_quant`) e multiplicada por `wo_a` via `deepseek_v4_fp8_einsum` com recipe `(1,128,128)` em cc ≤ 9 e `(1,1,128)` em cc ≥ 10 (**interpretação**: granularidades de escala (m,n,k) de ativação/peso no estilo DeepGEMM).

**D10 — Paridade QAT do KV (referência)**
- `DS/recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark_kernels.py:604-674`: `dspark_quant_dequant_nope` — "The released DeepSeek V4 Flash path applies `act_quant(..., inplace=True)` after KV norm and RoPE, but only to the non-RoPE dimensions. This helper mirrors that QAT simulation while keeping the RoPE slice in bf16". Grupo 64, `scale = 2^ceil(log2(amax/448))`, `amax ≥ 1e-4`.
- Conceito: **o modelo foi treinado vendo o KV já quantizado** (QAT/QDQ); por isso servir o KV em FP8 UE8M0/64 é "nativo" e o `fp8_ds_mla` não é uma degradação pós-treino.

**D11 — Plugin experimental `vllm_gb10_hybrid_nvfp4`**
- Arquivos: `DS/vllm_patch_gb10/vllm_gb10_hybrid_nvfp4/{__init__,config,kernel}.py`.
- Mecanismo: registra `modelopt_gb10_hybrid` (subclasse de `ModelOptMixedPrecisionConfig`) que troca o kernel NVFP4 linear por `Gb10HybridNvFp4LinearKernel`: `M < GB10_HYBRID_NVFP4_M_THRESHOLD` (128) → `apply_fp4_marlin_linear` (W4A16); senão `scaled_fp4_quant(x, input_global_scale_inv, is_sf_swizzled_layout=True)` + `cutlass_scaled_fp4_mm(x_fp4, w, x_blockscale, w_scale, alpha)` (W4A4) (`kernel.py:51-97`). Guarda um clone Marlin + a cópia CUTLASS ("Uses extra memory", `README.md:35-36`). "Tested here only for syntax packaging, not on a live GB10" (`README.md:38`).
- Nota: não se aplica ao checkpoint DSV4 (MXFP4 nativo, não ModelOpt NVFP4 W4A4) — é uma ferramenta genérica para checkpoints ModelOpt.

**D12 — FP8 attention/indexer e kernels**
- Pesos de atenção FP8 ("`fused_wqa_wkv` and indexer `wq_b`/`weights_proj` replicated", ≈2.4 GB/step), lm_head BF16 129,280×4,096 = 1.06 GB (`DS/docs/CLAUDE/fable5-1-report.md:84-90`).

---

## 4. Números e benchmarks (relacionados a formato numérico)

### 4.1 Bytes por parâmetro / por token (calculados a partir dos formatos)

| Formato | Bits por valor armazenado | Overhead de escala | Total efetivo | Fonte / cálculo |
|---|---|---|---|---|
| FP32 | 32 | — | 32 bpw | padrão |
| BF16 / FP16 | 16 | — | 16 bpw | padrão |
| FP8 per-tensor | 8 | 32 bits / tensor inteiro | ≈ 8.000 | — |
| FP8 per-channel (GLM dense, Qwen fp8dense) | 8 | 1 escala (BF16 no GLM, FP32 no Qwen) por linha de K elementos | 8 + 16/K (GLM) ou 8 + 32/K (Qwen); K=4096 ⇒ ≈ 8.004 / 8.008 | `GLM/overlay/exl3.py:2129-2134`; `Qwen/files/fp8dense/make…:161` |
| FP8 block 128×128 (DS lineares, Qwen MTP experts) | 8 | 1 escala por 16,384 pesos | ≈ 8.002 | `DS/.../fable5-1-report.md:454`; `Qwen/files/patch_modelopt_fp8_block_moe.py` |
| MXFP8 (bloco 32, E8M0) | 8 | 8 bits / 32 | 8.25 | OCP MX (externo) |
| NVFP4 (bloco 16, E4M3, + FP32 global) | 4 | 8 bits / 16 (+32 bits/tensor) | 4.5 | NVIDIA (externo); Qwen RadixArk: experts 56.25 GiB packed + 7.03 GiB escalas ⇒ 7.03/56.25 = 0.125 = 1 byte de escala a cada 8 bytes (16 valores) (`Qwen/docs/CLAUDE/fable5-1-report.md:61`) |
| MXFP4 (bloco 32, E8M0) | 4 | 8 bits / 32 | 4.25 | OCP MX; DS expert: w13 8.4 MB + w2 4.2 MB + escalas E8M0 0.8 MB = 13.4 MB (`DS/.../fable5-1-report.md:86`) |
| EXL3 K=4 (GLM experts) | 4 (trellis) | `suh`+`svh` FP16 + marcador int32 por matriz | ≈ 4.01–4.02 | `effective_bpw = 8·(trellis + 2·(in+out) + 4)/(in·out)` (`GLM/kernel_lab/exl3/abi.py:74-84`); ex.: [4096×2048] → 4.0117; [4096×1024] (shard TP2) → 4.0196 (cálculo meu) |
| EXL3 K=6 (draft H3, lm_head turboderp) | 6 | idem | ≈ 6.0x | `GLM/tools/dflash2_exl3_package.py:168-196` escreve `bpw_effective` por tensor |

| KV cache (por token, por camada salvo indicado) | Bytes | Composição | Fonte |
|---|---|---|---|
| GLM `fp8_ds_mla` (alvo) | **656** | 512 × FP8 E4M3 (NoPE) + 4 × FP32 (escala por 128) + 64 × BF16 (RoPE = zeros) | `GLM/Dockerfile:12-13, 24-29`; `GLM/README.md:677` |
| GLM alternativa sem pad (hipotética) | ~528 | 512 + 16 | `GLM/Dockerfile:26-27` |
| GLM draft DFlash2 (BF16) | 2048 B/token (5 camadas SWA, "this boot") | BF16 | `GLM/README.md:680, 693` |
| DSV4 `fp8_ds_mla` = `nvfp4_ds_mla` | **584** | 448 × FP8 E4M3 NoPE (7 blocos de 64) + 64 × BF16 RoPE (128 B) + 8 B (7 escalas UE8M0 + 1 pad) | `DS/recipe/overlay/.../cache_utils.py:44-51`; `DS/docs/CLAUDE/item8-fp4-kv-design.md:11` |
| DSV4 BF16 hipotético | 1024 | 512 × 2 B | cálculo |
| DSV4 "NVFP4 probe" (Stage A/B, abandonado) | 416 | (interpretação) 256 + 32 + 128 | `DS/recipe/nvfp4/Dockerfile.stage-b` |
| DSV4 layout alternativo lane vLLM 0.26 | 352 | 256 B E2M1 + 32 B E4M3/16 + 64 B RoPE FP8 | `DS/Dockerfile.gb10-dsv4-dspark:129-151` |
| DSV4 opção A proposta | 292 | 224 B E2M1 + 4 B UE8M0/128 + 64 B RoPE FP8 | `DS/docs/CLAUDE/item8-fp4-kv-design.md:30` |
| DSV4 indexer K cache | 132 por token comprimido por camada C4 | (interpretação) 128 FP8 + 4 B FP32 | `DS/docs/CLAUDE/fable5-1-report.md` §3.6; `DS/docs/PATCHES.md:1248` |
| Qwen QSA BF16 | 26,624 B/token (todas as 13 camadas K/V) | 13 × 2 × 2 heads × 256 × 2 B | `Qwen/README.md:371-373` |
| Qwen QSA FP8 | 13,312 B/token | 13 × 2 × 2 × 256 × 1 B | `Qwen/README.md:373` |
| Qwen medido | 9,418 B/token/GPU (FP8) vs 15,192 (BF16) | inclui estado GDN/QSA BF16 | `Qwen/README.md:376-381` |

### 4.2 Qualidade

| Medida | Valor | Data / condição | Fonte |
|---|---|---|---|
| KLD(teacher‖model) GLM — TR3 K6 (6bpw) | 0.013723 nats, 254 GB | malaiwah, 5 runs, 25 janelas, 51,175 posições | `GLM/README.md:413-421` |
| … FP8 oficial (cross-stack) | 0.020615, 328 GB | idem | idem |
| … **EXL3 4bpw (servido)** | **0.024555**, 176 GB | idem | idem |
| … FP8 oficial (stack brandonmusic v44) | 0.024629, 328 GB | idem | idem |
| … NVFP4 (stack brandonmusic v44) | 0.060535, ~180 GB | idem | idem |
| GLM dense FP8 (dense,kda) | KL proxy 0.002–0.013 nats/pos; argmax 94–100% | 2026-09-08 | `GLM/README.md:344` |
| GLM FP8=all + KDA BF16 | NOT-WORSE em todos os itens; ΔNLL −0.004…+0.0005 nats/token; KL texto 25–30% acima dos pares stock | 2026-09-26, 3 boots stock × 2 capturas | `GLM/docs/adopt-fp8-all-kda-bf16-2026-09-26.md:35-44` |
| GLM KDA BF16 vs Marlin (camada) | rel RMSE 0.002526 (vs W8A8 ~0.0266) | — | `GLM/docs/kda-bf16-large-m.md:58-66` |
| GLM A/B/C W8A8 (controle positivo) | M=65: 7.99× KL stock-null, 23 flips; BF16: 1.10×, 0 flips | 15 textos, 33,005 posições | `GLM/docs/kda-bf16-large-m.md:72-81` |
| GLM thin-decode FAST | T1 flips 0; stock null KL 0.0128–0.0131; veredito INCONCLUSIVE (gate absoluto falha no próprio stock) | estudo pr182q-20260917 | `GLM/docs/sm121-perf-paths.md:114-146` |
| Qwen FP8 dense | rel RMSE 0.0240 / 0.0262 / 0.0268 (min/med/max, 591 tensores); cos ≥ 0.9996 | 2026-09-02 | `Qwen/docs/CLAUDE/fable5-1-report.md:442` |
| Qwen FP8 KV | reasoning+needle 12/12 (igual BF16); upstream alertava 6/6 → 2/6 em raciocínio longo | 2026-09-05 | `Qwen/CHANGELOG.md:299-313` |
| Qwen FP8 KV (lane v0.30) | NLL 1.328 / 1.334; needles 3/3 a 200k | 2026-09-25 | `Qwen/CHANGELOG.md:26-28` |
| Qwen checkpoint (model card via relatório) | GSM8K 97.27% (1283/1319), AIME26 pass@1 98.75% "inside the bf16 reference band" | — | `Qwen/docs/CLAUDE/fable5-1-report.md:421-423` |
| Qwen MTP com FP8 block correto | aceitação 72.8% (89 / 74.5 / 60% por posição) | TP2+EP, MTP=3 | `Qwen/README.md:700-712` |
| NVFP4 vs FP8 (externo) | "1% or less accuracy degradation versus FP8 on DeepSeek-R1-0528" | NVIDIA blog | Apêndice B |

### 4.3 Throughput / capacidade atribuíveis a escolhas de formato

| Mudança | Efeito | Fonte |
|---|---|---|
| GLM `GLM53_DENSE_FP8=dense,kda` (FP8 Marlin) | −11 ms/step; prose +12–19%; +37% com adaptive-k | `GLM/README.md:344` |
| GLM `all` vs `dense,kda` | decode ms/cycle −4.67% / −4.91%; tok/s +5.2 / +4.3% | `GLM/docs/adopt…:26-27` |
| GLM KDA BF16 large-M | TTFT 16k −11.4%, 100k −12.3%, misto −14.1%; +3.29 GiB/rank | `GLM/docs/kda-bf16-large-m.md:91-111` |
| GLM KV 14 → 11 GiB (para pagar KDA BF16) | 1,572,073 → 1,233,779 tokens (−21.5%), 1.45× a 850k | `GLM/docs/adopt…:53-57` |
| GLM E3 grouped (kernel EXL3 prefill) | +37–45% prefill frio; 1,580–1,640 tok/s | `GLM/README.md:262-281, 1812` |
| GLM EXL3_MOE_FAST | +7.71/+8.88/+14.59% decode (células resolvidas) | `GLM/docs/sm121-perf-paths.md:78-86` |
| GLM TRITON_ATTN no draft (não formato, mas mesma tabela) | estruturado cai para ~29 tok/s | `GLM/README.md:369-370` |
| Qwen FP8 KV | +71% tokens (2,131,159 → 3,652,200); −5.3% throughput médio | `Qwen/README.md:316-328`; `Qwen/CHANGELOG.md:129-133` |
| Qwen FP8 KV (re-medição) | 2,426,193 → 4,378,829 tokens (9.26× → 16.70× a 262k) | `Qwen/docs/bench/rerun-20260916.md:15-19` |
| Qwen FP8 oficial (todos experts FP8) | ~500k tokens de KV (vs 3.65M) | `Qwen/README.md:99-101` |
| Qwen FP8-dense (analítico) | 9.2 → 5.7 GiB/step; ~85–90 tok/s batch-1 | `Qwen/docs/CLAUDE/fable5-1-report.md:110-112` |
| Qwen SSM BF16 (single-Spark) | +8.5% agregado a 8 streams; step 141.4 → 130.4 ms | `Qwen/.env.sample:70-84` |
| DS issue #22 (dispatch FP8) | ~1 tok/s → ~17 tok/s a 600K | `DS/docs/PATCHES.md:172-186` |
| DS KV | 17.04 GiB / 2,331,430 tokens / 2.22× a 1M (TP2, util 0.835); TP3 ≈ 35 GiB/rank, 5,021,871 tokens | `DS/README.md:141`; `DS/docs/CLAUDE/fable5-1-report.md:56-57` |
| DS W4A16 vs FP4 tensor core | ~125 TFLOPS BF16-class vs ~4× (FP4×FP8); prefill +30–60% estimado com MXFP4×MXFP8 | `DS/docs/CLAUDE/fable5-1-report.md:37, 278-300` |
| DS KV FP4 real (proposto) | +50% (390 B) / +65% físico (292 B) | `DS/docs/CLAUDE/fable5-1-report.md:346-348`; `item8:39-41` |
| DS MXFP4 indexer (opt-in) | metade dos bytes lidos por chave; −9% KV bytes só com alocação de meia-linha | `DS/docs/PATCHES.md:1248-1256` |

### 4.4 Números de hardware relevantes (do repo)
- GB10: `sm_121`, 121.69 GiB visíveis ao CUDA (`Qwen/README.md:349`), "~273 GB/s theoretical" LPDDR5x (`Qwen/docs/CLAUDE/fable5-1-report.md:21`), 230–250 GB/s efetivos (`DS/docs/CLAUDE/fable5-1-report.md:91`).
- "GB10 FP8 tensor rate is 2× BF16" (`GLM/docs/improve-prefill.md:163`).
- QSA: tabela conservadora de "48-SM" para GB10 vs GB300 com 160 SMs (`Qwen/README.md:534-537`).
- SM12x usa `mma.sync` (warp-level) e não `tcgen05` (`GLM/docs/exl3-sm121-kernel-lab.md:103-107`).

---

## 5. Conceitos necessários (em ordem de pré-requisito)

> Cada conceito: o que é (2–4 frases) + como aparece concretamente aqui. Os diagramas de bits estão em texto para virarem animações CSS/Canvas. Exemplos numéricos completos no Apêndice A.

### Nível 0 — Bits e números

**C1. Bit, byte, palavra e contêiner.** Um bit é 0/1; 8 bits = 1 byte; GPUs leem memória em palavras (32 bits) e transações de 32/64/128 bytes. Um *formato numérico* é a regra que dá significado a um padrão de bits; o *contêiner* é o tipo usado para guardá-lo (ex.: dois valores FP4 dentro de um `uint8`). Aqui: trellis EXL3 guardado como `int16` (`GLM/overlay/exl3.py:3008-3013`), MXFP4 como `uint8` com `hidden/2` colunas (`DS/.../model.py:356-363`), escalas E8M0 como `uint8`, KV DS como `uint8` "empacotado" de 584 B por token.

**C2. Inteiros sem sinal e com sinal (complemento de dois).** `uintN` representa 0..2^N−1; `intN` em complemento de dois representa −2^(N−1)..2^(N−1)−1. Operações inteiras (multiplicação mod 2^32, shifts, máscaras) são baratíssimas na GPU. Aqui: o codebook MCG é literalmente `x *= 0xCBAC1FED` mod 2^32 seguido de máscara/XOR (`codebook.cuh:67-75`); o marcador `mcg` é o int32 `-877912083` = `0xCBAC1FED` reinterpretado com sinal.

**C3. Ponto flutuante (anatomia).** Um float tem **S**inal, **E**xpoente (com *bias*) e **M**antissa: valor normal = (−1)^S · 2^(E−bias) · (1 + M/2^m); se E=0, subnormal = (−1)^S · 2^(1−bias) · (M/2^m). Mais bits de expoente = mais **faixa** (range); mais bits de mantissa = mais **precisão** (passos relativos menores, ulp = 2^−m no topo de cada binade; erro de arredondamento ≤ 2^−(m+1)). Todos os formatos abaixo são variações disso.

**C4. FP32 / FP16 / BF16.**
```
FP32  [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM]  1/8/23  bias 127  max ≈3.4e38
FP16  [S|EEEEE|MMMMMMMMMM]                  1/5/10  bias 15   max 65504, min normal 6.1e-5
BF16  [S|EEEEEEEE|MMMMMMM]                  1/8/7   bias 127  mesma faixa do FP32, só 8 bits de precisão
```
BF16 = "FP32 truncado" (mesmos 8 bits de expoente), por isso não estoura; FP16 tem mais precisão mas estoura em 65504. Aqui: o modelo roda em BF16 (ativações, pesos densos), mas os kernels EXL3 trabalham em **FP16** (convertem `x.half()`, `GLM/overlay/exl3.py:1498`) e o GLM precisa de `SWIGLU_LIMIT_DEFAULT = 10.0` para não estourar FP16 (`exl3.py:63`); o backport v030 do Qwen escolhe FP16 para V porque "P ≤ 1 … fp16 has the range, its wider mantissa is more accurate" (`patch_qsa_fp8_kv_v030.py:101-105`). Exemplo: 70000 → BF16 70144, FP16 = overflow (∞) (Apêndice A.1).

**C5. FP8 E4M3 e E5M2.**
```
E4M3 (fn) [S|EEEE|MMM]   bias 7   max 448 = 1.75·2^8   min normal 2^-6   min subnormal 2^-9
          sem ±inf; NaN = S.1111.111   → 253 valores finitos (127 magnitudes incl. 0)
E5M2      [S|EEEEE|MM]   bias 15  max 57344           tem ±inf e NaN (estilo IEEE)
```
E4M3 é o padrão para pesos/ativações/KV em inferência (mais precisão); E5M2 (mais faixa) é típico de gradientes. A constante **448** aparece em todo lugar: `scales = amax/448` (`GLM/overlay/exl3.py:2129`; `Qwen/files/fp8dense/make…:30`; `DS/.../cache_utils.py:174`). Aritmética útil: "e4m3 → bf16/fp16 é exato" (todo E4M3 cabe em BF16) — usado no Qwen (`patch_qsa_fp8_kv_v030.py:84-88`) e no raciocínio de exatidão do GLM KDA ("≤4 significant bits × ≤8 … at most 12 significant bits", `exl3.py:2046-2052`). E5M2 só aparece como opção `fp8_e5m2` do `CacheDType` do vLLM, não é usado.

**C6. FP4 E2M1.**
```
E2M1 [S|EE|M]  bias 1   magnitudes: E=00→{0, 0.5(sub)}  E=01→{1, 1.5}  E=10→{2, 3}  E=11→{4, 6}
               sem inf/NaN; 16 códigos (8 magnitudes × sinal); código = (S<<3)|índice
```
São só 15 valores distintos (±0 conta como 0). Por si só inútil — precisa de uma **escala por bloco** (ver C12–C14). Aqui: LUT `_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)` com sinal no bit 3 e **nibble baixo = primeiro elemento** (`Qwen/files/patch_ple_layer.py:131-155`); é o "e2m1" do NVFP4, do MXFP4 dos experts DSV4 e do indexer MXFP4.

**C7. Formatos só-expoente: E8M0 / UE8M0 e UE4M3.**
```
E8M0 [EEEEEEEE]  valor = 2^(E-127)   0xFF = NaN   sem sinal, sem mantissa, sem zero
```
Serve apenas como **escala potência de 2**: multiplicar por ela = somar ao expoente (barato e sem erro de arredondamento). "UE8M0" = unsigned E8M0 (mesma coisa, ênfase em "sem sinal"). "UE4M3" = E4M3 usado como escala positiva (NVFP4). Aqui: DS armazena escalas do KV como `exponent + 127` em `uint8` (`cache_utils.py:122-126`); converte com `(sf.int32 << 23).view(float32)` — o byte vira o campo de expoente de um FP32 (`DS/.../model.py:452-453`); e alerta que `copy_()` de `float8_e8m0fnu` para `uint8` faz conversão numérica ("2^-7 → 0") e destrói os bytes (`model.py:1990-1997`).

### Nível 1 — Álgebra linear, redes e LLMs (o mínimo para entender *onde* os formatos entram)

**C8. Tensor, matriz, GEMM e o papel de pesos vs ativações.** Uma camada linear computa `Y[M,N] = X[M,K] · W[K,N]`; W são os **pesos** (fixos, lidos da memória a cada passo), X são as **ativações** (mudam a cada token). "Canal de saída" = coluna N de W (ou linha, na convenção `[out,in]` do PyTorch); "canal de entrada" = K. Aqui: o EXL3 usa a convenção `A[M,K] @ W[K,N]` (`GLM/docs/exl3-sm121-kernel-lab.md:50-51`); `nn.Linear` do vLLM guarda `weight[out,in]` e os scripts FP8 fazem `amax(dim=1)` = por linha = **por canal de saída**.

**C9. Transformer / LLM e onde estão os bytes.** Um LLM decoder é uma pilha de blocos (atenção + MLP/MoE) com normalizações e conexões residuais; gera um token por vez (autoregressivo). Em modelos MoE, cada token ativa só top-k experts (GLM top-8 de 288, DS top-6 de 256, Qwen top-10 de 512), mas **todos** os experts ficam em memória. Aqui: experts dominam o tamanho (Qwen: 63.28 GiB de experts NVFP4; DS: ~12 GB lidos por passo só de experts), então os experts recebem a quantização mais agressiva (4 bits) e o resto fica em 8/16 bits.

**C10. Prefill vs decode e o *roofline*.** Prefill processa o prompt inteiro (M grande, milhares de linhas) → limitado por **compute** (FLOPs). Decode processa 1 token por sequência (M = nº de sequências × tokens especulativos, 1–64) → limitado por **banda** (bytes lidos/tempo). Intensidade aritmética = FLOPs/byte; abaixo do "ridge point" o kernel é memory-bound. Aqui: Qwen "Batch-1 decode … at the memory-bandwidth roofline already" (`fable5-1-report.md:89-92`); o GLM troca de kernel em M=144 (EXL3) e M=512 (KDA FP8) justamente nesse crossover; o plugin híbrido do DS troca Marlin/CUTLASS em M=128.

**C11. KV cache.** Na atenção, cada token passado deixa K e V por camada; guardá-los evita recomputar. Tamanho = camadas × (K+V) × heads_kv × head_dim × bytes. Variantes reduzem isso: GQA (menos heads KV), MLA (um latente comprimido compartilhado, ex. 512 dims), sparse attention (lê só top-k blocos), camadas lineares/recorrentes (estado constante). Aqui: GLM e DS usam MLA sparse (latente 512 + RoPE 64); Qwen usa GQA (2 KV heads × 256) só em 12 de 48 camadas + GDN. Quantizar o KV (FP8) dobra aproximadamente a capacidade de tokens.

### Nível 2 — Fundamentos de quantização

**C12. Quantização uniforme (inteira): escala e zero-point.** Mapear real → inteiro: `q = clamp(round(x/s) + z, qmin, qmax)`; `x̂ = (q − z)·s`. Simétrica: z=0, `s = amax/qmax`. Assimétrica: `s = (max−min)/(2^b−1)`, `z = round(−min/s)`. Erro vem de **arredondamento** (≤ s/2) e **clipping**. Aqui: nenhuma receita usa INT4/INT8 de pesos diretamente (EXL3 não é inteiro uniforme), mas o raciocínio escala/zero-point é a base de tudo; exemplo no Apêndice A.2.

**C13. Quantização em ponto flutuante com escala.** Com FP8/FP4 o "grid" não é uniforme (mais denso perto de zero). Escala `s = amax / max_representável` (448 para E4M3, 6 para E2M1) leva o maior valor ao topo do grid; `x̂ = FP(x/s)·s`. Vantagem sobre inteiro: erro relativo ~constante em várias ordens de grandeza (bom para distribuições de cauda longa). Aqui: todos os FP8 do repo (GLM dense, Qwen fp8dense, KV DS) seguem exatamente `x/(amax/448)`.

**C14. Granularidade da escala.** Quem compartilha uma escala: **per-tensor** (1 escala para a matriz inteira), **per-channel** (1 por linha/coluna), **per-token** (1 por linha da ativação, calculada em tempo de execução), **per-group/block 1D** (1 a cada g elementos contíguos: 16, 32, 64, 128), **block 2D** (1 por tile 128×128). Mais fino = menos erro (outliers só "estragam" o próprio bloco) e mais bytes de escala.
- per-tensor: KV FP8 do Qwen (`layer._k_scale`/`_v_scale`), tabela PLE FP8 (uma `weight_scale` global), escala global do NVFP4.
- per-channel: GLM dense FP8 (escala BF16 por linha), Qwen fp8dense (FP32 por linha).
- per-token dinâmico: ativações do Qwen fp8dense (`FP8_PER_CHANNEL_PER_TOKEN`).
- per-group 16: NVFP4; 32: MXFP4/MXFP8; 64: KV DSV4 (UE8M0); 128: KV DSV3.2/GLM `fp8_ds_mla` (4 FP32 para 512 dims).
- block 2D 128×128: lineares DSV4, experts MTP do Qwen (`FP8_BLOCK_SCALES`, `weight_block_size=[128,128]`).

**C15. Outliers e por que LLMs são difíceis de quantizar.** Pesos e (principalmente) ativações de LLMs têm poucos canais com magnitude 10–100× maior; com escala per-tensor eles forçam `s` grande e todos os valores pequenos viram 0. Soluções: granularidade fina (C14), rotações (C22 — Hadamard espalha o outlier por todos os elementos) e manter camadas sensíveis em alta precisão. Aqui: o GLM mantém `kv_b_proj`, embeddings, normas em BF16; o Qwen mantém router, gates, indexer, `in_proj_a/b`, conv1d, MTP e visão em BF16 (`make_fp8_dense_checkpoint.py:58-78`); o DS mantém a parte RoPE do KV em BF16.

**C16. Escala potência-de-2 vs escala fracionária.** Uma escala E8M0 só pode valer 2^k: o bloco usa, no pior caso, apenas metade da faixa (erro maior); uma escala E4M3 pode valer ~1.125·2^k etc. NVIDIA: MSE de codificação de escala "0.08 average" (E4M3) vs "0.72" (E8M0) (externo). Regras de arredondamento: DS **arredonda a escala para cima** (`ceil(log2(amax/448))`) — nunca satura, mas desperdiça até 1 bit (no exemplo A.6, amax=3.7 usa só 236.8 de 448); OCP MX usa `floor(log2(amax)) − emax_elem` (pode saturar e clampar).

**C17. Microscaling (OCP MX): MXFP4, MXFP6, MXFP8, MXINT8.** Padrão da Open Compute Project: blocos de **32** elementos compartilham uma escala **E8M0**; elementos FP8 (E4M3/E5M2), FP6 (E3M2/E2M3), FP4 (E2M1) ou INT8. Custo: +0.25 bit/valor. Aqui: experts DSV4 são MXFP4 (`fp4_e8m0_k32`), indexer MXFP4 opt-in, lineares MXFP8 no Qwen (patch de fallback), e o caminho futuro `kind::mxf8f6f4`.

**C18. NVFP4.** Formato NVIDIA: valores E2M1, blocos de **16** com escala **E4M3** (fracionária) e **uma escala FP32 por tensor** para trazer as escalas de bloco para a faixa do E4M3: `x ≈ q · s_bloco(E4M3) · s_global(FP32)`; 4.5 bits/valor. Convenção ModelOpt/vLLM: `weight_scale` (E4M3 por 16), `weight_scale_2` (FP32 global) e, para W4A4, `input_scale` global da ativação (estática, calibrada) + escalas de bloco da ativação calculadas em tempo real (`scaled_fp4_quant`). Aqui: experts do Qwen (W4A4 CUTLASS), PLE NVFP4 (`patch_ple_layer.py:21-72`), plugin híbrido do DS (`kernel.py:81-95`), e o nome enganoso `nvfp4_ds_mla`.

**C19. Weight-only (WxA16) vs weight+activation (WxAx).** Weight-only: pesos quantizados, ativação em BF16/FP16; o kernel desquantiza os pesos em registrador e faz MMA em 16 bits — ganha banda (decode), não FLOPs. W8A8/W4A4: também quantiza a ativação e usa tensor cores de baixa precisão — ganha FLOPs (prefill) mas adiciona erro de quantização da ativação. Aqui: GLM dense FP8 = **W8A16 Marlin**; EXL3 = **W4A16** (mma f16); DS b12x = **W4A16**; Qwen experts = **W4A4**; Qwen fp8dense = **W8A8**; GLM mostrou que o erro dominante do W8A8 era o QDQ da ativação (0.0266 vs 0.0025) e escolheu BF16 para M grande.

**C20. Quantização estática vs dinâmica; calibração.** Estática: escalas decididas offline (precisa de dados de calibração para ativações). Dinâmica: escalas calculadas em tempo de execução por token/bloco (sem calibração, custo de um `amax`). Calibração para PTQ de pesos (GPTQ/EXL3) usa ativações reais para estimar a importância (Hessiano). Aqui: Qwen fp8dense "no calibration set needed" (ativação dinâmica); GLM dense FP8 sem calibração (weight-only); EXL3 conversão usa `-cr 250` linhas × `-cc 2048` colunas de calibração por default (externo, `doc/convert.md`), exceto o draft DFlash2 ("uncalibrated … synthetic Hessian", `dflash2_exl3_convert.py:23-26`); KV FP8 do Qwen usa escalas per-tensor sem calibração no repo (**interpretação**: vLLM usa 1.0 quando o checkpoint não fornece escalas de KV).

**C21. PTQ vs QAT.** Post-Training Quantization quantiza um modelo já treinado (RTN, GPTQ, EXL3/QTIP, ModelOpt). Quantization-Aware Training insere quant/dequant (QDQ) durante o treino para o modelo aprender a tolerar o erro. Aqui: tudo é PTQ, **exceto** o KV do DSV4: "The released DeepSeek V4 Flash path applies `act_quant(..., inplace=True)` after KV norm and RoPE, but only to the non-RoPE dimensions … QAT simulation" (`DS/.../dspark_kernels.py:640-645`) — por isso o `fp8_ds_mla` do DS é o formato "nativo" do modelo.

### Nível 3 — Algoritmos de PTQ de pesos (para entender EXL3)

**C22. RTN, erro de reconstrução e a *proxy loss* com Hessiano.** Round-to-nearest (RTN) quantiza cada peso independentemente. Melhor: minimizar o erro na **saída** da camada, `E‖(Ŵ−W)x‖² = tr((Ŵ−W) H (Ŵ−W)ᵀ)` com `H = E[x xᵀ]` (Hessiano proxy estimado com dados de calibração). GPTQ/LDLQ quantizam colunas em sequência e propagam o erro para as colunas ainda não quantizadas (feedback via decomposição LDL de H). Aqui: EXL3 herda isso do QTIP/QuIP# ("On-the-fly Hessian computation", README exllamav3, externo).

**C23. Incoherence processing / transformada de Hadamard aleatória (RHT).** Multiplicar W à esquerda e à direita por matrizes ortogonais `H·S` (Hadamard × sinais aleatórios ±1) não muda a função (`xW = (xHS)(SHᵀ W …)`) mas "espalha" outliers: a matriz resultante tem entradas aproximadamente i.i.d. gaussianas (QTIP: "W̃'s entries are approximately independently Gaussian distributed", externo). Hadamard de tamanho 2^n se aplica em O(n·2^n) com butterflies. Aqui: blocos de **128** (`had_*_r_128_inner`, escala 1/√128), sinais/escalas em `suh` (entrada) e `svh` (saída) (`hadamard_inner.cuh`, `reference.py:189-231`). Exemplo 4-D no Apêndice A.7: `[8, 0.5, −0.3, 0.2] → [4.2, 3.5, 4.3, 4.0]` (mesma norma, sem outlier).

**C24. Quantização vetorial (VQ) e codebooks.** Em vez de quantizar escalares, quantiza-se vetores de d pesos escolhendo a entrada mais próxima de um codebook de 2^(k·d) vetores; ótimo para fontes gaussianas mas o codebook cresce exponencialmente com d (QuIP# para em d=8). Aqui: contexto histórico do EXL3 (QuIP# → QTIP).

**C25. Trellis-coded quantization (TCQ) e o *bitshift trellis* do QTIP.** TCQ usa um decodificador com **estado**: com L bits de estado e K bits por peso, cada novo peso desloca K bits novos para dentro do estado de L bits (`estado_t = ((estado_{t−1} << K) | bits_novos) mod 2^L`), e o valor do peso é `codebook(estado_t)`. Pesos vizinhos compartilham L−K bits ⇒ não são livres; o encoder escolhe a sequência ótima de bits com **Viterbi** (programação dinâmica sobre 2^L estados). Isso dá a qualidade de uma VQ de dimensão enorme (QTIP: 256-dim) com custo de decodificação trivial (um shift). *Tail-biting*: o estado inicial reutiliza os bits do fim da sequência (o fluxo é circular), eliminando overhead de inicialização. Aqui: L=16, V=1, K=4 (GLM), tile de 16×16 = 256 pesos, janela circular (`exl3_dq.cuh:15-31`). Exemplo no Apêndice A.8.

**C26. Codebooks computados (lookup-free): 1MAD, 3INST, MCG, mul1.** Em vez de uma tabela de 2^16 valores, o valor é *calculado* a partir do estado com 2–4 instruções inteiras, produzindo uma distribuição ~gaussiana (casada com os pesos pós-Hadamard). QTIP propõe 1MAD (LCG + soma de bytes) e 3INST (LCG + máscara/XOR em FP16 + soma de duas metades). EXL3 usa as variantes **MCG** (multiplicativo, sem termo aditivo: `x·0xCBAC1FED`, depois o mesmo truque do 3INST) e **mul1** (`x·0x83DCD12D` + soma de bytes via `dp4a` + afim). Aqui: GLM experts = MCG (cb=1); tensores densos turboderp = mul1 (cb=2); marcadores int32 no checkpoint.

**C27. Formato EXL3 (ABI) e bits por peso.** Por matriz: `trellis int16 [K/16, N/16, 16·bits]` (tiles 16×16 em ordem de fragmento de tensor core), `suh fp16 [K]`, `svh fp16 [N]`, marcador `mcg` ou `mul1` int32 (`GLM/docs/exl3-sm121-kernel-lab.md:43-48`). K (bits) é inteiro por tensor (1–8); taxas fracionárias do modelo (ex.: "4.05bpw") são **alocações** de K diferentes entre tensores ("Fractional model-wide rates remain allocations across integer-rate tensors", `:52-53`). "Uniform-K4" = todos os experts com K=4. `head_bits` (default 6 no conversor, externo) = bits do lm_head; K=16 = sentinela "não quantizar". Dequant (C23+C25+C26): estado de 16 bits → codebook → valor ~N(0,1) → MMA FP16 → Hadamard de saída → × `svh`.

**C28. Decodificação em registrador vs reconstrução (materialização).** Kernels "fused" desquantizam cada tile em registrador e já fazem a MMA (lê 4 bits/peso, ótimo para decode). Para M grande, compensa reconstruir a matriz inteira em FP16/BF16 uma vez e rodar um GEMM denso de alta eficiência (lê 16 bits/peso mas amortiza sobre muitas linhas). Aqui: LinearEXL3 `AUTO_RECONSTRUCT_THRESHOLD = 144`; retenções BF16 do GLM (KDA M>512, dense-EXL3 M>144); `EmulationMxfp8LinearKernel` do Qwen desquantiza na carga; b12x W4A16 desquantiza para BF16 dentro do kernel.

### Nível 4 — Hardware e kernels

**C29. Tensor cores e instruções MMA.** Unidades de matriz executam `D = A·B + C` em blocos (ex.: `m16n8k16` FP16). Blackwell adiciona MMA **block-scaled**: a instrução recebe também as escalas por bloco e faz a desquantização em hardware. Datacenter Blackwell (sm_100) usa `tcgen05.mma` (assíncrono, operandos em SMEM/TMEM); consumer Blackwell (sm_120/sm_121, **GB10**) usa `mma.sync` warp-level com operandos em registradores. NVFP4 no SM12x: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (4 escalas por K=64 porque bloco=16); MXFP8/MXFP6/MXFP4 mistos via `kind::mxf8f6f4` com escalas UE8M0 (externo, Colfax; `DS/.../fable5-1-report.md:296-297`). Aqui: EXL3 usa só `mma.sync …f16` (não FP4); Qwen experts usam FP4 tensor cores (CUTLASS SM120); DS b12x usa BF16 MMA (W4A16).

**C30. sm_120 vs sm_121 vs sm_100 e compilação (`12.1a`).** Compute capability define ISA/recursos; o sufixo `a` habilita instruções específicas da arquitetura (não portáveis). GB10 = 12.1; compartilha a ISA consumer Blackwell do sm_120 (RTX 50xx/RTX Pro). Aqui: `TORCH_CUDA_ARCH_LIST=12.1a` (GLM `Dockerfile:381`), `CUTE_DSL_ARCH=sm_121a` (DS); DeepGEMM gera nomes `sm121_*` mas só traz `sm120_*` (alias `#define`); Qwen executa cubins `sm_120` não-`a` no 12.1; muitos gates de software dizem "datacenter only" (sm_10x) mesmo quando o kernel SM120 existe (MXFP4 indexer gate).

**C31. Bibliotecas/kernels consumidores de formatos.**
- **exllamav3_ext** (turboderp): GEMM/GEMV/MoE EXL3 (trellis+Hadamard+codebook), autotuner cooperativo — GLM.
- **Marlin**: GEMM mixed-precision weight-only (desquantiza em registrador), variantes FP8 W8A16 e FP4 W4A16 — GLM dense FP8; plugin híbrido DS.
- **CUTLASS** (NVIDIA): GEMMs FP8 scaled-mm (`CutlassFP8ScaledMMLinearKernel`), NVFP4 W4A4 (`cutlass_scaled_fp4_mm`), grouped MoE (FlashInfer CUTLASS SM120) — Qwen.
- **FlashInfer**: sparse MLA SM120 (lê `fp8_ds_mla`), `mm_mxfp8`, MoE CUTLASS, JIT de fontes — GLM/DS/Qwen.
- **b12x** (CuTeDSL para SM12x): MoE W4A16 a partir de MXFP4, MLA sparse, WO fusionado — DS.
- **DeepGEMM**: GEMMs FP8/FP4 e *mqa logits* do indexer (FP8/FP4, paged) — DS.
- **Triton**: kernels QSA (Qwen, patch FP8 KV), insert/gather do KV DS (`cache_utils.py`).
- **cuBLAS**: GEMMs BF16 densos (retenções BF16 do GLM; denso BF16 do Qwen).

**C32. Restrições de forma/alinhamento dos kernels.** Kernels quantizados exigem dimensões múltiplas do bloco/tile: EXL3 16 (tiles) e 128 (Hadamard/N direto); NVFP4 K%16; `mm_mxfp8` N,K ≥128 e %32; Marlin pitch %8 e alinhamento 16 B; MXFP4 b12x arredonda hidden/intermediate a 128; FlashMLA topk %128. Quando o TP divide uma dimensão e quebra o múltiplo, é preciso replicar (Qwen visão `data`), manter BF16 (GLM TP3 `f_b_proj/g_b_proj`), usar EP (GLM TP3 experts) ou emular (MXFP8 fallback).

**C33. Sharding de tensores quantizados em TP/EP.** Column-parallel divide saídas (corta `svh` e colunas do trellis; `suh` replica), row-parallel divide entradas (corta `suh` e linhas do trellis; `svh` replica), seguido de all-reduce (`GLM/overlay/exl3.py:661-676`). Escalas por bloco precisam ser cortadas junto com o bloco — por isso o Qwen evitou block-128 no `ParallelLMHead` ("shards block scales incorrectly"). EP (expert parallel) evita cortar a matriz do expert: cada rank tem experts inteiros.

**C34. CUDA graphs e quantização.** Um CUDA graph grava uma sequência de kernels e a re-executa sem overhead de CPU; não pode haver alocação, sincronização de host nem autotune durante a captura, e o ramo de código tomado na captura fica "congelado" por tamanho de batch. Aqui: fronteiras de dispatch fixas guardadas na camada (`glm53_bf16_lm_min_m`), `scatter_add_` em vez de `bincount`, warmup do autotuner EXL3 antes da captura (deadlock observado), b12x recusa preparar pesos durante captura.

### Nível 5 — Formatos de KV e caches específicos

**C35. Layout `fp8_ds_mla` (DeepSeek-V3.2/GLM, 656 B).** Registro por token: 512 dims NoPE em FP8 E4M3 + 4 escalas FP32 (uma por 128) + 64 dims RoPE em BF16. O kernel sparse MLA do SM120 só aceita isso (sem BF16). GLM (NoPE puro) preenche a RoPE com zeros — exato mas +24% de bytes.

**C36. Layout DSV4 (584 B) e o "`nvfp4_ds_mla`".** Latente 512 = 448 NoPE + 64 RoPE; NoPE em 7 blocos de 64 com escala UE8M0 (potência de 2); RoPE BF16; escalas agrupadas no fim do bloco de páginas. `nvfp4_ds_mla` usa exatamente os mesmos bytes (Stage C) — nome herdado de uma tentativa de layout FP4 (416 B) abandonada; o bug #22 era só de dispatch.

**C37. "FP4 KV de verdade" e a restrição de página uniforme.** O alocador híbrido do vLLM exige mesma página em bytes para todos os grupos de KV (MLA comprimido, SWA, draft); uma linha FP4 só é barata se dividir 37,376 B (584×64) em potência de 2 (292×128). Por isso a opção A (292 B) é recomendada; a C (368 B) quebraria a regra.

**C38. Cache do indexer (Lightning indexer) e MXFP4 de "scores".** DSA/sparse attention usa um indexer leve (64 heads × 128 dims no DSV4) que pontua blocos e escolhe top-k; seu K-cache é separado (132 B/linha FP8) e replicado nos ranks. Quantizá-lo para MXFP4 afeta *qual* bloco é escolhido, não o valor da atenção — risco = "selection drift at depth". O mesmo fenômeno explica o alerta do Qwen: FP8 no K muda o que o indexer QSA seleciona.

**C39. KV FP8 per-tensor em atenção densa/GQA e fatoração da escala.** Com uma escala por tensor, `q·(s·K̂)ᵀ = s·(q·K̂ᵀ)`: aplica-se `s_k` ao score e `s_v` à saída, sem desquantizar tiles (Qwen). Custos observados: −5% de throughput (casts/escala) e ~1.7× de capacidade.

**C40. Estado recorrente (GDN/Mamba) e seu dtype.** Camadas lineares (Gated DeltaNet no Qwen, KDA no GLM, Mamba) guardam um estado fixo por sequência; seu dtype (FP32 do checkpoint vs BF16) afeta bytes por passo e tamanho de página do alocador. Qwen: `MAMBA_SSM_CACHE_DTYPE=bfloat16`; GLM: `mamba_cache_dtype=auto` (`README.md:679`).

**C41. Tabelas de embedding quantizadas (PLE/n-gram).** Uma embedding é um lookup de linhas, não um GEMM; quantizá-la exige um método que desquantize só as linhas buscadas (`F.embedding` sobre códigos e escalas). Qwen: PLE de 51 GB em FP8 per-tensor (nvidia/RadixArk) ou NVFP4 empacotado (local-inference-lab: linhas `dim/2` + `dim/16` escalas).

### Nível 6 — Checkpoints, metadados e o vLLM

**C42. Safetensors, índice e shards.** Arquivo = cabeçalho JSON (nome → dtype, shape, offsets) + bytes; `model.safetensors.index.json` mapeia tensor → shard. Permite ler só partes ("range reads", `dense_overlay.py`), hard-link de shards (fp8dense), validação sem baixar pesos (`pack_profile.py`). Dtypes de safetensors vistos: `BF16, F16, F32, F8_E4M3, U8, I16, I32, I64, BOOL`.

**C43. Metadados de quantização no checkpoint.** `config.json → quantization_config` (lido pelo vLLM) e o sidecar legado `hf_quant_config.json` (ModelOpt). Campos vistos: `quant_method` (`exl3`, `modelopt`), `quant_algo` (`NVFP4`, `MIXED_PRECISION`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES`, `FP8_PB_WO`, `MXFP8`, `W4A16_NVFP4`), `group_size`, `quantized_layers` (por prefixo), `exclude_modules`/`ignore`, `config_groups` com `targets` e `weights.num_bits` (estilo compressed-tensors; o repo lê isso para detectar o dtype da PLE, mas não usa o método `compressed-tensors`), `scale_fmt` (DS), `codebook`/`scope`/`bits`/`non_routed_exl3` (EXL3). Nomes de escala: `weight_scale` (multiplica), `weight_scale_inv` (DeepSeek/FP8 block; nome histórico "inv"), `weight_scale_2` (global NVFP4), `input_scale`. Armadilhas reais: nome do MTP por índice relativo vs absoluto; `FP8_PB_WO` vs `FP8_BLOCK_SCALES`; dtype PLE ausente.

**C44. Arquitetura de quantização do vLLM.** `--quantization X` escolhe um `QuantizationConfig` (registrado com `@register_quantization_config`); `override_quantization_method` permite auto-detecção; `get_quant_method(layer, prefix)` devolve um método por camada (`LinearMethodBase`, `FusedMoEMethodBase`, métodos de embedding) com `create_weights` (aloca parâmetros no formato empacotado e registra `weight_loader`), `process_weights_after_loading` (repack/conversão para o kernel, ex. `prepare_fp8_layer_for_marlin`) e `apply` (chama o kernel). `--kv-cache-dtype` e `--moe-backend` escolhem layouts/kernels de KV e MoE. Todos os patches desta seção operam nesses pontos de extensão.

**C45. Medindo a qualidade de uma quantização.** Perplexidade/NLL (média de −log p do token correto), **KL divergence** teacher-forced (distância entre a distribuição do modelo quantizado e a de referência, por posição — mais sensível que perplexidade), concordância de argmax, e testes de tarefa (GSM8K, AIME, needle-in-haystack, RULER-lite, tool battery). O GLM mostra que limiares absolutos falham porque o próprio stock varia entre execuções ("near-tie tokens flip between identical runs") e usa painéis calibrados contra a variação stock-vs-stock (`tests/bench_numerical_calibrated.py:12-17`).

**C46. O trade-off tríplice: bytes × velocidade × qualidade (e memória unificada).** Cada escolha move três eixos: FP8 KV (+capacidade, −5% velocidade, risco no indexer), FP8 dense (+velocidade, +KL pequeno), cópias BF16 retidas (+prefill, −memória), 4bpw vs NVFP4 (mesmos bytes, KLD 2.5× menor com EXL3). Na GB10 a memória do modelo compete com o SO e o page cache (UMA), então "liberar GiB" também é estabilidade (GLM paga +3.3 GiB/rank de BF16 com −3 GiB de KV).

---

## 6. Receita passo a passo — decidir e implantar os formatos numéricos de um MoE gigante em 2× GB10

> Sequência lógica (hardware → imagem → pesos → patches → argumentos → verificação → ajuste) focada no eixo "formato numérico". Cada passo diz *o que* e *por quê*, com o exemplo das três receitas.

**Passo 1 — Fazer o orçamento de bytes antes de escolher formatos.**
- Some, por rank: pesos (com TP/EP), KV desejado (tokens × bytes/token), estado recorrente, ativações/CUDA graphs, scratch de kernels, e a margem de host (UMA!). Na GB10 são ~121.69 GiB visíveis; os repos mantêm ≥ 4–6 GiB de `MemAvailable` sob carga (Qwen baixou GMU para 0.80 porque 0.835 deixava 0.3–0.9 GiB, `Qwen/CHANGELOG.md:9-15`).
- Calcule bytes lidos por passo de decode (`Qwen/docs/CLAUDE/fable5-1-report.md:74-88`, `DS/docs/CLAUDE/fable5-1-report.md:84-91`): tok/s ≈ (banda efetiva / bytes por passo) × tokens aceitos por passo. Isso diz onde quantizar primeiro (o maior consumidor de bytes).

**Passo 2 — Inventariar o checkpoint (sem baixar tudo).**
- Ler `config.json`/`quantization_config`/`hf_quant_config.json` e os cabeçalhos safetensors (dtype/shape de cada tensor): é assim que o Qwen descobriu que o FP8 oficial não quantiza o denso (`fable5-1-report.md:114-119`) e que o checkpoint nvidia declara a PLE só em `config_groups` (`detect_ple_dtype.py`), e que o DS tem `expert_dtype fp4` + FP8 block 128×128 (`DS/.../fable5-1-report.md:450-454`).
- Anotar formatos por categoria: experts, denso, atenção, embeddings/lm_head, MTP/draft, visão, tabelas especiais (PLE), escalas (nome e convenção: `weight_scale` vs `weight_scale_inv` vs `weight_scale_2`).

**Passo 3 — Escolher o formato dos experts (o grosso dos bytes).**
- Opções vistas: EXL3 K4 (GLM; melhor KLD por byte: 0.0246 @ 176 GB vs NVFP4 0.0605 @ ~180 GB), NVFP4 W4A4 (Qwen; tensor core FP4 nativo, bom prefill), MXFP4 nativo do fabricante (DS; mantido porque é o formato de origem, rodado W4A16).
- Critérios: (a) qualidade medida (KLD/tarefas), (b) existe kernel **para sm_121** que consome o formato em decode (M pequeno) e em prefill (M grande)? (c) restrições de forma (16/32/128) após TP/EP, (d) memória.
- Por quê o GLM não usa NVFP4: "KLD 0.0605 vs 0.0246, and SM12x NVFP4 kernels are dense-MHA only (no fp8_ds_mla)" (`GLM/docs/improve-prefill.md:202-203`).

**Passo 4 — Decidir o denso: BF16, FP8 weight-only, FP8 W8A8 ou EXL3.**
- Se o decode é limitado por banda e o denso é grande (Qwen: 6.9 de 9.2 GiB/passo), quantizar o denso rende mais que mexer nos experts.
- Weight-only FP8 per-channel (GLM, Marlin W8A16): sem calibração, erro pequeno, ganho em decode; para prefill (M grande) Marlin fica lento → reter BF16 da matriz lógica e trocar por cuBLAS acima de um M medido (512).
- W8A8 dinâmico per-token (Qwen fp8dense, CUTLASS): também acelera prefill (FP8 tensor = 2× BF16), mas adiciona erro de ativação (o GLM mediu que é o termo dominante).
- Per-channel em vez de block-128 quando dimensões não são múltiplas de 128 ou o loader não sabe shardear escalas de bloco (Qwen, `fable5-1-report.md:121-124`).
- Mantenha em alta precisão: `kv_b_proj` (lido direto pela absorção MLA), roteadores/gates, normas, conv1d, indexers, embeddings, MTP/draft se a aceitação cair.

**Passo 5 — Escolher o KV cache.**
- Veja quais layouts o **kernel de atenção do seu backend no sm_121** aceita: GLM/DS sparse MLA SM120 → só `fp8_ds_mla` (656 B GLM/V3.2; 584 B DSV4); NVFP4 KV só em MHA denso (GLM `README.md:751`); QSA do Qwen só BF16 até o patch FP8.
- Se o modelo foi treinado com QDQ do KV (DS), use o layout nativo. Se não (Qwen), FP8 per-tensor é um *trade* (capacidade vs velocidade/seleção do indexer) — valide com tarefas de contexto longo.
- Verifique a regra de página uniforme do alocador híbrido antes de inventar layouts (DS item8).
- Draft/speculative: KV do draft pode precisar de outro dtype (GLM: `auto`/BF16 porque "no MLA FP8 backend on SM121" para o draft denso).

**Passo 6 — Montar a imagem com kernels do formato para `sm_121a`.**
- Compilar para `12.1a` (`TORCH_CUDA_ARCH_LIST`, `FLASHINFER_CUDA_ARCH_LIST`, `CUTE_DSL_ARCH=sm_121a`).
- GLM: vLLM pinado + `exllamav3` @ `c5d9c657` compilado com patches (stub AVX aarch64, kernels fat E2/E3, thin-decode), registro de `exl3` no `QuantizationMethods`, overlay `exl3.py` copiado depois do build CUDA (`GLM/Dockerfile:369-449`).
- DS: imagem Anemll `dspark-vllm-gx10:0.1.1` (b12x 0.15.3, FlashInfer 0.6.15) + hotfixes; persistir caches JIT (Triton/TileLang/CuTe/DeepGEMM) para não recompilar formatos no meio do serviço.
- Qwen: imagem day-0 `vllm/vllm-openai:qwen38-flash-next` + bind-mounts de arquivos patchados (sem rebuild).

**Passo 7 — Aplicar patches de metadados/dispatch (fail-closed).**
- Registrar o método (GLM `exl3`), consertar nomes/aliases (Qwen MTP `mtp.layers.48`, `FP8_PB_WO→FP8_BLOCK_SCALES`), adicionar ramos faltantes de dispatch (Qwen FP8 block MoE, PLE FP8/NVFP4), rotas de kernel (DS issue #22), gates conservadores (DS MXFP4 indexer), fallbacks por forma (Qwen MXFP8 emulation).
- Padrão dos repos: patch por âncora exata, idempotente, verificação de SHA do arquivo alvo, e "refuse to guess" (ex.: block shape lido do `group_size`).

**Passo 8 — Gerar/converter checkpoints derivados quando necessário.**
- FP8-dense (Qwen): conversão CPU em streaming, hard-link do resto, `quantized_layers` por prefixo, verificação de erro por tensor (rel RMSE ~2.6%).
- Dense EXL3 (GLM): range-read de tensores já quantizados (turboderp 4.05bpw), overlay + índice reescrito.
- Draft EXL3 6bpw (GLM): conversor exllamav3 sem calibração (Hessiano sintético), K/V BF16, empacotamento com bytes BF16 originais.
- Sempre fixar revisões e validar inventário/cabeçalhos antes de derrubar o serviço (`GLM/tools/pack_profile.py`).

**Passo 9 — Argumentos do vLLM.**
- GLM: `--quantization exl3 --kv-cache-dtype fp8` (+ `--kv-cache-memory-bytes 11811160064`), draft `kv_cache_dtype: auto`, envs `GLM53_DENSE_FP8=all`, `GLM53_KDA_BF16_LARGE_M=1`, `EXL3_FUSED_MOE=1`, `EXL3_FAT_GROUPED=1`.
- Qwen: `--kv-cache-dtype fp8` (com patch), `--mamba-ssm-cache-dtype bfloat16`, `--enable-expert-parallel`, `--hf-overrides '{"text_config":{"ple_embedding_dtype":…, "rope_parameters":…}}'`, `MM_ENCODER_TP_MODE=data`.
- DS: `--kv-cache-dtype nvfp4_ds_mla --block-size 256 --moe-backend flashinfer_b12x`, `VLLM_USE_B12X_MOE=1`, `CUTE_DSL_ARCH=sm_121a`; opcionais `--attention-config '{"use_fp4_indexer_cache":true}'` (+ alias DeepGEMM), `--quantization modelopt_gb10_hybrid` (só para checkpoints ModelOpt NVFP4).

**Passo 10 — Verificar que o formato certo está realmente rodando.**
- Logs: GLM "EXL3 MCG trellis engaged for routed experts: bits=4 … fused_moe=exl3_moe … (no BF16 expert reconstruct at load)" (`exl3.py:3210-3220`), `dense fp8 groups: all`, `kda bf16-large-m retained` ×34 por rank; Qwen `Selected CutlassFP8ScaledMMLinearKernel`, warning de fallback MXFP8 por forma, "Routed experts … use FP8_BLOCK_SCALES; building them with Fp8MoEMethod"; DS `Using 'B12X_MXFP4' Mxfp4 MoE backend`, `W4A16FusedMoeKernel`, status do hotfix #22 "APPLIED".
- Números de KV no boot (`Available KV cache memory`, `GPU KV cache size`, `Maximum concurrency`) e comparar com o esperado de bytes/token.
- Sinais indiretos de erro de formato: aceitação especulativa próxima de aleatória (escala de bloco errada, Qwen), garbling em contexto longo, NaN.

**Passo 11 — Medir qualidade com o instrumento certo.**
- KL teacher-forced contra referência (GLM painel malaiwah; painel calibrado com ≥4 capturas stock), NLL, tarefas (GSM8K/AIME, needles, RULER-lite, tool battery), e aceitação do draft por posição.
- Controle negativo (stock vs stock) para saber o ruído; controle positivo (ex.: W8A8 que sabidamente desloca probabilidades) para saber se o instrumento detecta algo.

**Passo 12 — Ajustar com A/B/A.**
- Fronteiras de dispatch por M (144, 512, 128) a partir de microbenchmarks e da distribuição real de M; tamanhos de scratch/temp; pool de KV vs cópias retidas; FP8 KV vs BF16 conforme a carga (capacidade vs −5%).
- Sempre A/B/A com mesma imagem e boot fresco; reportar CIs; separar "medido" de "analítico" (os repos fazem isso explicitamente).

---

## Apêndice A — Exemplos numéricos passo a passo (para animação)

> Todos calculados por mim com Python puro (arredondamento para o mais próximo; empates para mantissa par). Scripts em `scratchpad/qcalc/`.

### A.1 Anatomia de bits: o mesmo número em vários formatos
- π = 3.14159265…
  - FP32 `0x40490FDB` = `0 10000000 10010010000111111011011` → 3.1415927
  - BF16 `0x4049` = `0 10000000 1001001` → **3.140625** (expoente 128−127 = 1; mantissa 73/128 → 1.5703125 × 2)
  - FP16 `0x4248` = `0 10000 1001001000` → **3.140625** (coincidência: mesma aproximação aqui)
  - E4M3 → **3.25** = `0.1000.101` (2^(8−7) × (1 + 5/8)); vizinhos 3.0 e 3.25; 3.1416 está mais perto de 3.25
  - E5M2 → **3.0** = `0.10000.10` (2^1 × 1.5)
  - E2M1 → **3** (código 0x5)
- 0.1: BF16 0.10009765625 (`0x3DCD`) vs FP16 0.0999755859375 (`0x2E66`) → FP16 é ~8× mais preciso.
- 70000: BF16 70144 (`0x4789`), FP16 = **overflow → ∞** → por que BF16 é o default de treino/inferência e por que o kernel FP16 do EXL3 precisa de clamp.
- 1e-5: FP16 `0x00A8` é **subnormal** (1.00136e-5); BF16 1.00136e-5 normal.
- Erro relativo máximo de arredondamento (unit roundoff 2^−(m+1)): FP32 6e-8, FP16 4.9e-4, BF16 3.9e-3, E4M3 6.25%, E5M2 12.5%, E2M1 25% (sem escala!).

### A.2 INT4 assimétrico (base conceitual)
x = [−0.8, −0.1, 0.3, 1.2]; min=−0.8, max=1.2 → s = 2.0/15 = 0.1333; z = round(0.8/0.1333) = 6.
q = round(x/s)+z = [0, 5 (−0.75→−1+6), 8 (2.25→2+6), 15]; x̂ = (q−z)·s = [−0.8, −0.133, 0.267, 1.2]. Erro máx = s/2 = 0.067.

### A.3 FP8 E4M3 por canal (GLM `Glm53DenseFp8Method` / Qwen `quantize_rows`)
Linha w = [0.0213, −0.0071, 0.0450, −0.0132, 0.0009, −0.0388, 0.0167, 0.0302]
1. amax = 0.0450 → s = 0.0450/448 = 1.00446e-4
2. w/s = [212.05, −70.68, 448.0, −131.41, 8.96, −386.28, 166.26, 300.66]
3. arredonda na grade E4M3 (passos: 8 em [64,128), 16 em [128,256), 32 em [256,448]): [208, −72, 448, −128, 9, −384, 160, 288]
4. bits: `0.1110.101`, `1.1101.001`, `0.1111.110`, `1.1110.000`, `0.1010.001`, `1.1111.100`, `0.1110.010`, `0.1111.001`
5. dequant ×s: [0.020893, −0.007232, 0.045, −0.012857, 0.000904, −0.038571, 0.016071, 0.028929]
6. rel RMSE = 2.09% (o Qwen mediu 2.4–2.7% em 591 tensores reais)
Animação sugerida: régua não uniforme do E4M3 com espaçamento dobrando a cada potência de 2.

### A.4 NVFP4 (bloco de 16, E4M3 + FP32 global)
x (16 valores) = [0.12, −0.95, 0.33, 2.80, −0.04, 0.61, −1.72, 0.08, 0.47, −0.29, 1.10, −0.66, 0.02, 0.91, −2.05, 0.38]; suponha amax do tensor inteiro = 4.0.
1. escala global g = amax_tensor/(6·448) = 4.0/2688 = 0.00148810
2. amax do bloco = 2.80 → escala ideal em unidades de g = 2.80/6/g = 313.6 → E4M3 mais próximo **320** (`0.1111.010`) → escala efetiva s = 320·g = 0.47619
3. x/s = [0.252, −1.995, 0.693, 5.88, −0.084, 1.281, −3.612, 0.168, 0.987, −0.609, 2.31, −1.386, 0.042, 1.911, −4.305, 0.798]
4. grade E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}: q = [0.5, −2, 0.5, 6, 0, 1.5, −4, 0, 1, −0.5, 2, −1.5, 0, 2, −4, 1]
5. códigos 4-bit (S<<3 | índice): [1, C, 1, 7, 0, 3, E, 0, 2, 9, 4, B, 0, 4, E, 2] (hex)
6. empacota 2 por byte, **nibble baixo = elemento par** (convenção do `patch_ple_layer.py:141-143`): bytes [0xC1, 0x71, 0x30, 0x0E, 0x92, 0xB4, 0x40, 0x2E] → 8 bytes + 1 byte de escala = 9 bytes para 16 valores = 4.5 bits/valor
7. dequant q·s: [0.238, −0.952, 0.238, 2.857, 0, 0.714, −1.905, 0, 0.476, −0.238, 0.952, −0.714, 0, 0.952, −1.905, 0.476]; rel RMSE ≈ 8.5%
Note: 0.12, −0.04, 0.08, 0.02 viram 0 ou 0.5·s: é por isso que blocos pequenos (16) importam.

### A.5 MXFP4 no mesmo vetor (escala E8M0, regra OCP)
1. e = floor(log2(2.80)) − 2 = 1 − 2 = −1 (2 = expoente máximo do E2M1, pois 6 = 1.5·2^2) → byte E8M0 = −1+127 = **126** → escala 0.5
2. x/0.5 → grade E2M1: q = [0, −2, 0.5, 6(5.6→6), 0, 1, −3, 0, 1, −0.5, 2, −1.5, 0, 2, −4, 1]
3. dequant: [0, −1.0, 0.25, 3.0, 0, 0.5, −1.5, 0, 0.5, −0.25, 1.0, −0.75, 0, 1.0, −2.0, 0.5]; rel RMSE ≈ 9.6% (pior que NVFP4: escala só potência de 2 e bloco maior, aqui ainda de 16 para comparar; no MXFP4 real o bloco é 32)
4. como o DS converte o byte: `(126 << 23).view(float32)` = bits `0 01111110 000…` = 0.5 (`DS/.../model.py:452-453`)

### A.6 KV do DSV4: FP8 com escala UE8M0 por 64 (arredonda para cima)
Bloco (4 valores ilustrativos de um grupo de 64): x = [3.7, −1.2, 0.05, 0.9]
1. amax = 3.7 → amax/448 = 0.00826 → log2 = −6.92 → **ceil = −6** → escala 2^−6 = 0.015625; byte armazenado −6+127 = **121**
2. x/s = [236.8, −76.8, 3.2, 57.6] (note: o maior valor usa só 236.8 de 448 — ~0.9 bit de faixa desperdiçado)
3. E4M3: [240 (`0.1110.111`), −80 (`1.1101.010`), 3.25 (`0.1000.101`), 56 (`0.1100.110`)]
4. dequant: [3.75, −1.25, 0.0508, 0.875]
5. Comparação com escala fracionária amax/448 (estilo per-channel): x/s = [448, −145.3, 6.05, 108.97] → [448, −144, 6.0, 112] → [3.7, −1.189, 0.0496, 0.925] (menos erro no maior valor, mas a escala não é potência de 2)
6. Layout em memória do bloco de 64 tokens: `[64×576 B de dados (448 FP8 + 128 B RoPE BF16 por token)] [64×8 B de escalas] [pad]` (`cache_utils.py:46-51`)

### A.7 Hadamard: espalhando um outlier (base do EXL3)
H4 normalizado = ½·[[1,1,1,1],[1,−1,1,−1],[1,1,−1,−1],[1,−1,−1,1]]
v = [8.0, 0.5, −0.3, 0.2] (um outlier) → H4·v = [4.2, 3.5, 4.3, 4.0]; norma 8.0237 preservada. Antes: uma escala per-tensor desperdiçaria resolução nos três valores pequenos; depois: todos têm magnitude parecida.
No EXL3 o bloco é 128 (5 estágios de butterfly por shuffle entre lanes + 2 locais; fator 1/√128 = 0.0883883), aplicado nas duas pontas: entrada `H(x ⊙ suh)`, saída `H(y') ⊙ svh`.
Animação sugerida: butterfly de 4→8→…→128 com pares somando/subtraindo.

### A.8 Bitshift trellis (L=16, K=4) — como um fluxo de bits vira pesos
Fluxo de símbolos de 4 bits de um tile (brinquedo com 8 símbolos; real = 256): s = [A, 3, 7, F, 1, C, 5, 9]
Estado do peso t = janela dos últimos 16 bits = (s[t−3], s[t−2], s[t−1], s[t]), **circular** (tail-biting):

| t | janela | estado | valor MCG |
|---|---|---|---|
| 0 | C 5 9 A | 0xC59A | −1.908 |
| 1 | 5 9 A 3 | 0x59A3 | −1.939 |
| 2 | 9 A 3 7 | 0x9A37 | +0.803 |
| 3 | A 3 7 F | 0xA37F | −1.164 |
| 4 | 3 7 F 1 | 0x37F1 | −1.333 |
| 5 | 7 F 1 C | 0x7F1C | +0.763 |
| 6 | F 1 C 5 | 0xF1C5 | −2.033 |
| 7 | 1 C 5 9 | 0x1C59 | +0.854 |

- Cada peso custa só 4 bits novos, mas seu valor depende de 16 bits → o codificador (Viterbi) escolhe o fluxo inteiro para minimizar o erro ponderado pelo Hessiano; pesos vizinhos não são independentes.
- Com K=2 a janela cobre 8 símbolos de 2 bits (14 bits compartilhados entre vizinhos): ex. t=0 janela [0,1,1,2,3,0,2,2] → 0x16CA → −1.266.
- No kernel: `b0 = t·K + K − 16 + 256·K` (bit inicial) e índices `mod (K·256/32)` palavras (`exl3_dq.cuh:18-26`).

### A.9 Codebook MCG e mul1, instrução por instrução
MCG (cb=1), estado 0x0001:
1. `x = 1 × 0xCBAC1FED = 0xCBAC1FED`
2. `y = (x & 0x8FFF8FFF) ^ 0x3B603B60 = 0xB0CC348D`
3. metades FP16: `0x348D` = +0.284424, `0xB0CC` = −0.149902
4. soma = **0.134521484375** (valor dourado do teste)
Estado 0x0000 → `y = 0x3B603B60` → 0.921875 + 0.921875 = **1.84375**; 0x0002 → `0xBC3834BA` → 0.29541 + (−1.05469) = **−0.75928**; 0x0003 → **0.83984**.
Estatística sobre os 65,536 estados: média ≈ −3.7e-5, desvio 1.244, faixa [−3.949, +3.949].
mul1 (cb=2), estado 0x0001:
1. `x = 0x83DCD12D` (bytes 0x2D, 0xD1, 0xDC, 0x83 = 45, 209, 220, 131)
2. `dp4a(x, 0x01010101, 0x6400)` = 45+209+220+131 + 0x6400 = 605 + 25600 = 26205 = `0x665D` → como FP16 = 1024 + 605 = **1629.0**
3. `1629 × 0.0067673 − 10.3828 = 0.641`
Estatística: média ≈ −0.002, desvio 1.000, faixa [−3.45, 3.35] (soma de 4 bytes ≈ gaussiana por TCL; 147.7 ≈ √(4·(256²−1)/12)).

### A.10 Dequant EXL3 completa de um peso (conceitual, para animar)
Matriz equivalente: `W = diag(suh) · H₁₂₈ · Ŵ · H₁₂₈ · diag(svh)` (Hadamards bloco-diagonais de 128, `reference.py:189-197`), onde `Ŵ[k,n] = MCG(janela16(t))`. Em linguagem de fluxo: (1) ler 4 bits novos do trellis → (2) montar a janela de 16 bits → (3) MCG → valor ~N(0,1) em FP16 no fragmento do tensor core → (4) `mma.sync m16n8k16` com a ativação já rotacionada `H(x⊙suh)` → (5) acumular FP32 → (6) Hadamard de saída e × `svh`. Nenhuma matriz BF16 existe em memória para M ≤ 144 (G6).

### A.11 Bits por peso efetivos do EXL3 K=4
Matriz [in=4096, out=2048]: trellis = (4096/16)·(2048/16)·64 palavras × 2 B = 4,194,304 B; suh = 8,192 B; svh = 4,096 B; mcg = 4 B → 8·4,206,596/(4096·2048) = **4.0117 bpw**.
Shard TP2 de gate/up [4096 → 1024]: trellis 2,097,152 B + suh 8,192 (replicado) + svh 2,048 + 4 → **4.0196 bpw**.

### A.12 Fatoração da escala per-tensor no QK (Qwen FP8 KV)
q = [1, 2]; K = [[0.30, −0.12], [0.05, 0.44]]; s_k = amax/448 = 0.44/448 = 9.82e-4
K̂ = round_E4M3(K/s_k) = [[320, −120], [52, 448]]
Scores: exato q·Kᵀ = [0.06, 0.93]; fatorado s_k·(q·K̂ᵀ) = [0.0786, 0.9311].
- A igualdade `q·(s·K̂)ᵀ = s·(q·K̂ᵀ)` é exata (o patch aplica `scores *= k_scale` depois do `tl.dot`); o erro vem só de K̂.
- A primeira linha mostra o perigo da **cancelação** (0.30 − 0.24): erro relativo grande num score pequeno — é assim que FP8 no K pode mudar *quais* blocos o indexer seleciona.

### A.13 Por que a cópia BF16 do GLM KDA é "exata antes de um arredondamento"
Um E4M3 tem ≤ 4 bits significativos (1 implícito + 3), uma escala BF16 ≤ 8; o produto tem ≤ 12 bits significativos → cabe exatamente em FP32 (24 bits) → um único arredondamento ao converter para BF16 (8 bits). Ex.: 0.6 → E4M3 0.625 (`0.0110.010`, 1.25·2^−1); × escala BF16 qualquer = produto exato em FP32; só a conversão final para BF16 arredonda. Resíduo medido vs Marlin: rel RMSE 0.0025 (`GLM/docs/kda-bf16-large-m.md:58-66`).

### A.14 Bytes → tokens/s (roofline de decode)
Qwen: 9.2 GiB/passo ÷ 240 GB/s ≈ 41 ms → ~24 passos/s × ~2.2 tokens aceitos/passo ≈ 53 tok/s (medido 55.8) (`fable5-1-report.md:89-92`). FP8-dense: 5.7 GiB/passo → ~25 ms → ~85–90 tok/s (analítico).
DS: ≈16.5–17 GB/passo ÷ 230–250 GB/s = 68–74 ms; ×4.15 tokens aceitos → 56–61 tok/s (medido 62–83) (`DS/.../fable5-1-report.md:84-96`).

### A.15 Contas de KV
- GLM: 656 B = 512 (FP8) + 16 (4×FP32) + 128 (64×BF16 zeros); sem pad seriam ~528 B (−19.5%).
- DSV4: página comum = 64 × 584 = **37,376 B**; divisores com nº de linhas potência de 2: 292 × 128, 146 × 256 (`item8:25-26`).
- Qwen: 13 × 2 × 2 × 256 × 1 = 13,312 B/token FP8 (6,656 por GPU em TP2) vs 26,624 BF16; medido 9,418 vs 15,192 B/token/GPU incluindo estados BF16 → ganho 1.61–1.70× em vez de 2×.

---

## Apêndice B — Fontes externas consultadas (web)

- QTIP: Tseng, Sun, Hou, De Sa, "QTIP: Quantization with Trellises and Incoherence Processing", arXiv:2406.11235 (NeurIPS 2024 spotlight) — https://arxiv.org/abs/2406.11235 (bitshift trellis com L bits de estado, K bits/peso, V dims; códigos 1MAD/3INST/HYB; RHT; BlockLDLQ; tail-biting via duas passadas de Viterbi; L ∈ {12,14,16}, V ∈ {1,2}, k ∈ {2,3,4}; 3INST: LCG a=89226354, b=64248484; 2-bit Llama-2-70B Wikitext2 3.70 vs QuIP# 3.91 vs AQLM 3.83).
- ExLlamaV3 (turboderp-org) — https://github.com/turboderp-org/exllamav3 ("a streamlined variant of QTIP"; Hessiano on-the-fly e kernel Viterbi fundido) e `doc/convert.md` (`-b` bits médios com alocação por tensor; `-hb` head bits default 6; `-mb` MTP bits default 4, 16 = não quantizado; `-cr` 250 / `-cc` 2048 de calibração; `-cb mul1` (default) | `mcg` | `3inst`).
- NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ (E2M1; bloco 16 com escala E4M3; escala FP32 por tensor; 4.5 bits/valor; vs MXFP4 bloco 32 E8M0; MSE de escala 0.08 vs 0.72; ≤1% de degradação vs FP8 no DeepSeek-R1-0528; 3.5× menos memória que FP16, 1.8× que FP8).
- OCP Microscaling Formats (MX) v1.0 — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf (MXFP8 E5M2/E4M3, MXFP6 E3M2/E2M3, MXFP4 E2M1, MXINT8; bloco 32; escala E8M0 bias 127, 0xFF = NaN).
- Colfax Research, "NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)" — https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/ (SM12x usa `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`; operandos em registradores; vs `tcgen05.mma` do SM10x com SMEM/TMEM).
- Fontes adicionais listadas pelo próprio repo GLM (`docs/exl3-sm121-kernel-lab.md:167-182`): FLUTE (arXiv:2407.10960), exemplo CUTLASS 79a (NVFP4×BF16 GeForce), MoE W4A16 SM12x do FlashInfer.

---

## Apêndice C — Incertezas, inconsistências e pontos a confirmar

1. **Significado de "TR3"** no nome do checkpoint GLM: o repo não define a sigla (só "uniform-K4 EXL3/TR3 routed-experts").
2. **Nº de camadas MLA/DSA do GLM**: README diz "Target MLA (12 layers)" (`README.md:677`), Dockerfile diz "11 DSA layers" (`Dockerfile:27`).
3. **Checkpoint com MXFP8 no Qwen**: docstring do patch MXFP8 atribui os shapes problemáticos ao RadixArk; `.env.sample` descreve RadixArk como "BF16 attention" e local-inference-lab como "MXFP8 attention"; a tabela de backends do README (checkpoint default nvidia) lista um kernel MXFP8 com exceções. Não consegui confirmar (sem acesso aos cabeçalhos) quais lineares são MXFP8 em cada checkpoint.
4. **`PLE_QUANT_OVERRIDE=fp8`** aparece no README do Qwen mas não no `start.sh` atual (o mecanismo vigente é `ple_embedding_dtype` via `--hf-overrides`).
5. **Escalas do KV FP8 do Qwen**: o patch usa `layer._k_scale`/`_v_scale` (per-tensor) mas o repo não calibra escalas de KV; presumo o default do vLLM (1.0) — verificar em runtime.
6. **Layout 416 B** do Stage A/B do DS: decomposição 256+32+128 é minha inferência aritmética; o texto só diz "Probe layout from the GLM-5.2 NVFP4 sparse-MLA path".
7. **Composição do indexer 132 B** (128 FP8 + 4 B escala) e **68 B** em MXFP4: inferência aritmética; o repo afirma só 132 B/linha e "uses the first half for FP4".
8. **Oráculo do `kernel_lab`**: `decode_tiles` aplica MCG ao símbolo K-bit (não à janela de 16 bits do kernel); tratar como oráculo de ABI/bit-order, não como reconstrução fiel de pesos (ver G14).
9. **Codebook do draft 6bpw**: o empacotador escreve `"codebook": "mcg"` no bloco de config (exigido pelo `Exl3Config`), mas aceita tensores `mul1` ou `mcg` por módulo (`dflash2_exl3_package.py:63, 141`); o codebook efetivo depende do default do fork `63b32f0` (o README upstream diz default `mul1`).
10. **Recipe `(1,1,128)` vs `(1,128,128)`** do `deepseek_v4_fp8_einsum`: interpretação como granularidade de escala DeepGEMM não confirmada no repo.
11. **"~125 dense bf16 TFLOPS class on GB10"** e "FP4×FP8 ~4× the dense rate": números do relatório do DS (não medidos em microbench no repo).
12. **KL do dense FP8 do GLM**: marcado como "PROVISIONAL … no full KLD panel yet" (`README.md:344`); a adoção de `all` se baseia em NOT-WORSE de NLL/KL com 3 boots.

---

## Apêndice D — Fichas técnicas por formato (layout · dequant · memória · precisão · hardware · uso)

### D.1 FP32
- **Layout**: `[S:1][E:8, bias 127][M:23]`.
- **Dequant**: nativo.
- **Memória**: 4 B/valor.
- **Precisão**: ~7 dígitos decimais; faixa 1e-38..3e38.
- **Hardware**: FMA de CUDA cores; em tensor cores só como acumulador (e TF32).
- **Uso**: acumuladores de todos os GEMMs; escalas do `fp8_ds_mla` do GLM (4 por token); `weight_scale_2`/`input_scale` globais NVFP4 (Qwen); escalas por canal do FP8-dense (Qwen, `F32 [out]`); saída do `exl3_moe` (`out` fp32, `GLM/overlay/exl3.py:1497`); estado GDN do checkpoint Qwen (float32 por default); intermediário exato na reconstrução KDA do GLM.

### D.2 BF16
- **Layout**: `[S:1][E:8, bias 127][M:7]` — metade superior de um FP32.
- **Dequant**: nativo; conversão FP32→BF16 = arredondar os 16 bits baixos (RNE).
- **Memória**: 2 B/valor.
- **Precisão**: unit roundoff 2^−8 ≈ 0.39%; mesma faixa do FP32 (não estoura).
- **Hardware**: tensor cores BF16 (MMA `m16n8k16` bf16), cuBLAS.
- **Uso**: tipo de ativação de todo o serving; pesos densos nativos do GLM (antes do FP8 opcional) e do Qwen nvidia; lm_head do DS (1.06 GB) e do draft MTP do Qwen (1.18 GiB); RoPE no KV do DS/GLM; KV do draft GLM; cópias retidas "large-M" do GLM; `MAMBA_SSM_CACHE_DTYPE=bfloat16` no Qwen; shards `bf16_shards` do dense-EXL3; K/V do draft EXL3 (obrigatório BF16).

### D.3 FP16
- **Layout**: `[S:1][E:5, bias 15][M:10]`; máx 65504; subnormais abaixo de 6.1e-5.
- **Memória**: 2 B/valor.
- **Precisão**: 2^−11 ≈ 0.049% (8× melhor que BF16) mas faixa pequena.
- **Hardware**: tensor cores FP16 (`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` no EXL3 GEMV).
- **Uso**: domínio de cálculo do EXL3 (ativação convertida para `half`, pesos decodificados em FP16, `suh`/`svh` FP16, temps do fused MoE `float16`, saída `out_dtype=float16` nos shards dense-EXL3); V do PV-dot no backport QSA v030 (Qwen). Risco: overflow → clamps (`SWIGLU_LIMIT_DEFAULT=10.0`, `act_limit`).

### D.4 FP8 E4M3 (`float8_e4m3fn`)
- **Layout**: `[S:1][E:4, bias 7][M:3]`; sem inf; NaN = `S.1111.111`; máx 448; min normal 2^−6; min subnormal 2^−9.
- **Dequant**: `x = fp8 · escala` (escala per-tensor, per-channel, per-block conforme o esquema); upcast FP8→BF16/FP16 é exato.
- **Memória**: 1 B/valor + escalas.
- **Precisão**: erro relativo ≤ 6.25% por valor (3 bits de mantissa); rel RMSE típico de pesos ~2–2.7% (Qwen 591 tensores).
- **Hardware**: tensor cores FP8 no Blackwell (no SM12x via `mma.sync` warp-level; as variantes block-scaled MX usam `kind::mxf8f6f4` — detalhe de PTX a confirmar na doc da NVIDIA); "GB10 FP8 tensor rate is 2× BF16" (`GLM/docs/improve-prefill.md:163`). Kernels: CUTLASS scaled-mm (W8A8), Marlin FP8 (W8A16, desquantiza), FlashInfer sparse MLA (KV), Triton QSA (KV, upcast antes do dot).
- **Uso**: ver D.11–D.16.

### D.5 FP8 E5M2 (`float8_e5m2`)
- **Layout**: `[S:1][E:5, bias 15][M:2]`; máx 57344; tem inf/NaN.
- **Precisão**: 12.5%/valor; mais faixa.
- **Uso**: **nenhum** nas receitas; só a string `fp8_e5m2` no `CacheDType` do vLLM (`DS/patches/official-main-b12x-nvfp4-python.patch:6`). Ensinar como contraste (faixa vs precisão).

### D.6 FP4 E2M1
- **Layout**: `[S:1][E:2, bias 1][M:1]`; magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}; empacotado 2 por byte (nibble baixo primeiro na PLE do Qwen).
- **Dequant**: LUT de 8 magnitudes × sinal, depois × escala(s).
- **Memória**: 0.5 B/valor + escalas.
- **Precisão**: sozinho, 25% de erro relativo e só 15 valores → exige escala por bloco pequena.
- **Hardware**: tensor cores FP4 block-scaled (`kind::mxf4nvf4`, m16n8k64) em Blackwell; SM12x via `mma.sync` (não `tcgen05`).
- **Uso**: valores do NVFP4 (Qwen experts/PLE; plugin DS), do MXFP4 (DS experts; indexer opt-in), dos layouts FP4 KV propostos (DS 292/352/390/416 B).

### D.7 E8M0 / UE8M0 (escala potência de 2)
- **Layout**: `[E:8]`, valor 2^(E−127); 0xFF = NaN; sem sinal/mantissa.
- **Dequant**: `(byte << 23).view(float32)` (DS) ou `2^(byte−127)` (`cache_utils.py:292-294`).
- **Memória**: 1 B por bloco (32 elementos no MX; 64 no KV DS; 7+1 bytes por token no DSV4).
- **Precisão**: escala grosseira (fator 2); no pior caso o bloco usa metade da faixa; regra de arredondamento importa (DS = ceil → nunca satura).
- **Hardware**: tipo de escala `ue8m0` nas MMAs block-scaled (MX).
- **Uso**: MXFP4 dos experts DSV4 (`fp4_e8m0_k32`), escalas do KV DSV4, `scale_fmt` UE8M0 das lineares FP8 block do DSV4 (preservadas por `format_ue8m0` no WO b12x), MXFP8 (Qwen, bloco 32), MXFP4 indexer. Armadilha: `float8_e8m0fnu` → `uint8` via `copy_()` destrói os bytes (`DS/.../model.py:1990-1997`).

### D.8 NVFP4
- **Layout**: por bloco de 16 valores: 16 × E2M1 (8 B) + 1 × E4M3 (escala de bloco, positiva = "ue4m3"); por tensor: 1 × FP32 global. Ativação W4A4: escala global estática (`input_scale`) + escalas de bloco dinâmicas.
- **Dequant**: `x = e2m1 · s_bloco(E4M3) · s_global(FP32)` (`Qwen/files/patch_ple_layer.py:136-155`).
- **Memória**: 4.5 bits/valor (Qwen RadixArk: 56.25 GiB packed + 7.03 GiB escalas nos experts).
- **Precisão**: melhor que MXFP4 (bloco menor + escala fracionária); NVIDIA: ≤1% vs FP8 em DeepSeek-R1 (externo); no GLM, KLD 0.0605 (vs 0.0246 do EXL3 4bpw) na stack brandonmusic.
- **Hardware**: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…e2m1.e2m1.f32.ue4m3` no SM12x (externo); CUTLASS `cutlass_scaled_fp4_mm`, `scaled_fp4_quant` (layout de escala "swizzled"), FlashInfer CUTLASS SM120 grouped MoE, Marlin FP4 (W4A16).
- **Restrições**: K múltiplo de 16 por rank (Qwen visão ⇒ `MM_ENCODER_TP_MODE=data`); EP obrigatório no Qwen.
- **Uso**: Qwen experts roteados (W4A4) e PLE NVFP4 (checkpoint local-inference-lab); DS plugin `modelopt_gb10_hybrid` (W4A16 se M<128, W4A4 se M≥128); GLM descarta.

### D.9 MXFP4
- **Layout**: por bloco de 32: 32 × E2M1 (16 B) + 1 × E8M0 (1 B).
- **Dequant**: `x = e2m1 · 2^(E−127)`.
- **Memória**: 4.25 bits/valor. DS: w13 8.4 MB + w2 4.2 MB + escalas 0.8 MB por expert (full).
- **Precisão**: pior que NVFP4 (A.5), mas é o formato **nativo do checkpoint DSV4** (sem re-quantização).
- **Hardware**: `kind::mxf4nvf4` com `ue8m0` / `kind::mxf8f6f4` (E2M1 × E4M3 com E8M0) no SM12x — "the checkpoint's format is the hardware-native one" (`DS/.../fable5-1-report.md:296-297`); **mas** o b12x atual só usa FP4 tensor core com fonte `modelopt_nvfp4`, então roda **W4A16** (desquantiza para BF16).
- **Uso**: DS experts (`B12X_MXFP4`), MegaMoE (DeepGEMM `fp8_fp4_mega_moe`, só SM100, não usado), DS indexer K-cache opt-in.

### D.10 MXFP8
- **Layout**: bloco de 32: 32 × E4M3 (ou E5M2) + 1 × E8M0.
- **Memória**: 8.25 bits/valor.
- **Hardware/kernel**: FlashInfer `mm_mxfp8` (`FlashInferCutlassMxfp8LinearKernel`) com limites N,K ≥128 e %32 no sm_121; `EmulationMxfp8LinearKernel` desquantiza para BF16 na carga.
- **Uso**: lineares MXFP8 de checkpoints Qwen (ver Apêndice C.3); fallback por forma e para todo `visual.*`.

### D.11 FP8 per-channel weight-only (GLM `GLM53_DENSE_FP8`)
- **Layout**: `weight[N,K]` E4M3 + `weight_scale[N]` BF16 (dtype original), repackado para Marlin.
- **Dequant**: no kernel Marlin: `w = fp8 · scale[n]` em registrador → MMA BF16 (W8A16).
- **Memória**: ≈ 8 bpw (metade do BF16).
- **Precisão**: KL proxy 0.002–0.013 nats/pos; argmax 94–100%.
- **Hardware**: Marlin (não usa FP8 tensor core — desquantiza).
- **Uso/por quê**: corta ~11 ms/step de decode sem calibração; M grande troca para cópia BF16 (D.19).

### D.12 FP8 per-channel + ativação per-token dinâmica (Qwen `FP8_PER_CHANNEL_PER_TOKEN`)
- **Layout**: `weight` `F8_E4M3 [N,K]` + `weight_scale` `F32 [N]`; ativação quantizada em tempo real por linha.
- **Dequant**: GEMM FP8×FP8 com epílogo `y = acc · s_w[n] · s_x[m]`.
- **Precisão**: rel RMSE dos pesos 2.4–2.7%; erro extra da ativação (não medido na GPU).
- **Hardware**: `CutlassFP8ScaledMMLinearKernel` (cubins `cutlass_scaled_mm_sm120_fp8`).
- **Uso/por quê**: hybrid checkpoint para cortar bytes de denso no decode (9.2→5.7 GiB/step analítico) e FLOPs no prefill; per-channel porque dims não são múltiplas de 128 e o `ParallelLMHead` não shardeia escalas de bloco.

### D.13 FP8 block-scaled 128×128 (DeepSeek-style; Qwen `FP8_BLOCK_SCALES`)
- **Layout**: `weight` E4M3 + `weight_scale_inv` com 1 escala por tile 128×128 (FP32 ou UE8M0 conforme `scale_fmt`); ativações com escalas por grupo (1×128) dinâmicas (`activation_scheme="dynamic"`).
- **Dequant**: GEMM FP8 block-scaled (DeepGEMM/CUTLASS/FlashInfer `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`).
- **Memória**: ≈ 8.002 bpw.
- **Uso**: todas as lineares não-expert do DSV4 (inclui atenção `fused_wqa_wkv`, `wq_b`, `wo_a`/`wo_b`, indexer); experts do MTP no checkpoint nvidia Qwen (via `Fp8MoEMethod`, bloco lido do `group_size`=128); saída da atenção do draft DSpark quantizada para FP8 antes do `wo_a` (einsum FP8).

### D.14 FP8 per-tensor (Qwen KV, PLE)
- **Layout**: tensor E4M3 + 1 escalar FP32/BF16.
- **Dequant**: fatorável: aplica o escalar fora do produto (KV) ou após o lookup (PLE).
- **Uso**: KV QSA do Qwen (12+1 camadas; `_k_scale`, `_v_scale`); tabela PLE de 51 GB ("FP8 shards + one global weight_scale").
- **Trade-off**: KV: 1.70× tokens/GiB, −5.3% throughput, risco de seleção do indexer.

### D.15 `fp8_ds_mla` DeepSeek-V3.2/GLM (656 B/token)
- **Layout**: `[512 × E4M3 NoPE][4 × FP32 escala (1/128)][64 × BF16 RoPE]`; GLM: RoPE = zeros (latente 512 zero-paddado a 576).
- **Dequant**: dentro do kernel sparse MLA SM120 (FlashInfer `FLASHINFER_MLA_SPARSE_SM120`, modelo GLM_NSA).
- **Por quê**: único layout que o kernel sparse MLA do SM12x aceita (bf16 não tem kernel sparse; NVFP4 KV só MHA denso).

### D.16 KV DSV4 `fp8_ds_mla` ≡ `nvfp4_ds_mla` (584 B/token)
- **Layout**: por bloco de 64 tokens: dados `[448 × E4M3 NoPE | 64 × BF16 RoPE]` por token (576 B), depois 8 B de escala por token (7 × UE8M0 para 7 grupos de 64 + 1 pad), padding (`cache_utils.py:44-51`).
- **Quant**: `scale = 2^ceil(log2(max(amax,1e-4)/448))`, byte = expoente+127.
- **Por quê**: é o formato do QAT do modelo; a string `nvfp4_ds_mla` só muda o dispatch (issue #22).

### D.17 Layouts FP4 de KV MLA (históricos/propostos)
- 416 B (Stage A/B probe, "GLM-5.2 NVFP4 sparse-MLA path"; interpretação 256+32+128), 352 B (lane vLLM 0.26: 256 B E2M1 + 32 B E4M3/16 + 64 B RoPE FP8), 292 B (opção A: 224 + 4 UE8M0/128 + 64 FP8 RoPE), ~390 B (fable report: 224 + 28 + 128 + 8). Ensinar como exercício de "desenhar um formato": bits de valor × granularidade de escala × tipo de escala × tratamento da RoPE × restrição de página.

### D.18 MXFP4 no indexer (DS opt-in)
- **Layout**: K do indexer em E2M1 com UE8M0 por 32 dims, gravado na primeira metade da linha de 132 B.
- **Hardware**: DeepGEMM `fp8_fp4_(paged_)mqa_logits` SM120 (alias sm121).
- **Trade-off**: metade dos bytes lidos por chave pontuada; risco = mudança do top-k selecionado.

### D.19 Cópias BF16 retidas (GLM) — "desquantizar uma vez e guardar"
- KDA `in_proj`: `W_bf16 = fp8 × scale` (exato antes de 1 arredondamento), +98.25 MiB/camada/rank; dispatch M>512.
- Dense-EXL3: `get_weight_tensor()` (Hadamards aplicados) em BF16; dispatch M>144.
- Lição: é a versão "em memória" da reconstrução do C28 — troca memória por FLOPs eficientes no prefill.

### D.20 EXL3 (QTIP trellis; "TR3" no GLM)
- **Layout**: por matriz `[K_in, N_out]`: `trellis int16 [K/16, N/16, 16·bits]` (cada tile 16×16 = `bits × 256` bits, ordem de fragmento de tensor core, palavras de 16 bits trocadas para leitura little-endian), `suh fp16 [K]`, `svh fp16 [N]`, marcador int32 `mcg` (0xCBAC1FED) ou `mul1` (0x83DCD12D). Sem escala por bloco.
- **Dequant**: janela circular de 16 bits → codebook computado (MCG/mul1) → FP16 ~N(0,1) → MMA FP16 com `H(x⊙suh)` → Hadamard de saída → × `svh`. M>144: reconstrução FP16 + GEMM densa.
- **Memória**: K + ~0.01–0.02 bpw (suh/svh); taxas fracionárias por alocação de K entre tensores.
- **Precisão**: GLM 4bpw KLD 0.0246 ≈ FP8 oficial (0.0246 na mesma stack) com 54% dos bytes; K6 0.0137.
- **Hardware**: FP16 tensor cores via `mma.sync` (funciona em Ampere+ — `get_min_capability 80`); nenhuma instrução FP4; o custo é ALU inteira (multiplicação, lop3, dp4a, shifts) + shuffles da Hadamard.
- **Conversão**: exllamav3 (`-b`, `-hb`, `-cr/-cc`, `-cb`), Hessiano de calibração on-the-fly, Viterbi fundido; sem calibração para o draft DFlash2 (Hessiano sintético).
- **Uso**: GLM experts roteados (K4 MCG, uniform), dense opcional (turboderp 4.05bpw, mul1, lm_head K6), draft H3 6bpw.

### D.21 Metadados: ModelOpt, `hf_quant_config.json`, compressed-tensors, EXL3
- ModelOpt: `quant_method=modelopt`, `quant_algo` (global ou `MIXED_PRECISION` + `quantized_layers` por prefixo), `group_size`, `exclude_modules`/`ignore`, `producer`; sidecar legado `hf_quant_config.json` (vLLM o lê no caminho do draft; pode discordar do `config.json`).
- `config_groups` com `targets` e `weights.num_bits`: esquema estilo compressed-tensors presente nos configs Qwen (usado só para detectar o dtype da PLE); o método `compressed-tensors` só aparece em arquivos vLLM vendorizados do GLM TP3 (`overlay/tp3/vllm/model_executor/model_loader/weight_utils.py:254-260`).
- EXL3: `quant_method=exl3`, `bits`, `codebook`, `scope`, `head_bits`, `non_routed_exl3.layers`, `tensor_storage` (ledger por tensor), `glm53_profile` (extensão do recipe).

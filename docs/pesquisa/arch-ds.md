# DS41 — ARQUITETURA do DeepSeek-V4-Flash (0731 / Vision-Exp) + DSpark

> Inventário de pesquisa (não é o produto final). Foco: **arquitetura do modelo** e tudo o que, na receita
> `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`, depende dela (atenção híbrida CSA/HCA, indexer,
> MLA/MQA de KV compartilhado, MoE, mHC, quantização, KV cache, DSpark, visão).
> Idioma: PT-BR; identificadores, flags, variáveis e código ficam no original.

---

## 0. Fontes, convenções e "qual código roda de fato"

### 0.1 Fontes usadas (todas lidas nesta pesquisa)

| Fonte | Local | O que fornece |
|---|---|---|
| Repo DS41 (commit `97e8733`) | `scratchpad/repos/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` | receita, patches, overlay, docs, resultados |
| `config.json` Vision-Exp @ `86f746b36186f0e567729a5c06a8c918caba82a9` (revisão fixada pela receita; **idêntico** ao `main` em 2026-09-28, cujo HEAD é `6821d6ad…`) | `scratchpad/web/visionexp-config.json` (baixado de `huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/<rev>/config.json`) | todos os hiperparâmetros |
| `config.json` DeepSeek-V4-Flash-0731 e DeepSeek-V4-Flash (preview) | `scratchpad/web/flash0731-config.json`, `flash-config.json` | diferenças entre variantes |
| Implementação de referência oficial (MIT) `inference/model.py`, `kernel.py`, `vision.py`, `image_processor.py`, `config.json`, `README.md` do Vision-Exp @ `86f746b3` | `scratchpad/web/vx-inference_*.py` | **a matemática exata** (compressor, indexer, atenção, mHC, Gate, DSpark, ViT) |
| Model cards HF (V4-Flash, 0731, DSpark, Vision-Exp) | `scratchpad/web/readme-*.md` | parâmetros totais/ativos, receitas oficiais de vLLM/SGLang |
| Tech report V4 — arXiv **2606.19348** (v1, 2026-04-26) | web | CSA/HCA, mHC, Muon, FP4, KV 7–10 % do V3.2 |
| Paper DSpark — arXiv **2607.05147** (2026-07-06) | web | backbone paralelo + Markov head + confidence head + agendamento |
| `configs` V2/V3/V3.2-Exp | `scratchpad/web/v2-config.json`, `v3-config.json`, `v32exp-config.json` | linhagem MLA → DSA |
| `Anemll/dspark-vllm-gx10` (README) | web | imagem `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM 0.25.1/0.25.2.dev0 |

Notação de citação: `arquivo:linha` é relativo à raiz do repo DS41; `HF:inference/model.py:L` é o arquivo de
referência oficial baixado (revisão `86f746b3`); `cfg:L` é a linha do `config.json` Vision-Exp baixado.

### 0.2 IMPORTANTE — três "pistas" de código

1. **Pista padrão (Anemll 0.1.1)** — `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM `0.25.2.dev0+g752a3a504.d20260714`,
   torch 2.11.0+cu130, FlashInfer 0.6.15, b12x 0.15.3 (`docs/CLAUDE/fable5-1-report.md:54-55`). O código do modelo
   (`vllm/models/deepseek_v4/...`) está **dentro da imagem**, não no repo; a receita só o altera por hotfixes de
   startup em `patches/`. O drafter DSpark aqui é o "V2 speculator" `v1/worker/gpu/spec_decode/dspark/speculator.py`
   com `DSparkDraftModel` e Markov head em `qwen3_dspark.py` (`docs/PATCHES.md:846`, `patches/hotfix-dsv4-replicate-markov-head.py:1-20`,
   `scripts/fixtures/dspark-block-k/speculative-752a3a504-stock.py:899-911`).
2. **Pista Stage-C (histórica)** — overlay em `recipe/overlay/vllm/**` (inclui `models/deepseek_v4/nvidia/model.py`,
   `dspark.py`, `dspark_kernels.py`, `sm120.py`, `common/ops/cache_utils.py`, `v1/spec_decode/dspark*.py`,
   `config/speculative.py`). É a mesma arquitetura (mesmas classes `DeepseekV4Attention`, `DeepseekV4MoE`,
   `DeepseekV4DecoderLayer`, `DeepSeekV4DSpark…`), mas com knobs `VLLM_DSPARK_*` que **não existem** na Anemll
   (`docs/ENVS.md:1-25`, `README.md:305-314`). Uso o overlay como **"código-fonte legível" da arquitetura em vLLM**,
   sempre indicando quando algo é só Stage-C.
3. **Referência oficial** (`HF:inference/*`) — PyTorch/TileLang "de livro", é o que define a matemática.

Quando as três divergem, eu digo.

---

## 1. Visão geral

### 1.1 O que é o modelo (e as variantes)

DeepSeek-V4 é uma família de LLMs **Mixture-of-Experts** com **contexto de 1M tokens**. O tech report lista
V4-Pro (1.6T total / 49B ativos) e **V4-Flash (284B total / 13B ativos)**, pré-treinados em >32T tokens
(`scratchpad/web/readme-DeepSeek-V4-Flash.md`, tabela "Model Downloads"). As quatro inovações declaradas:

1. **Atenção híbrida CSA + HCA** (Compressed Sparse Attention + Heavily Compressed Attention) — no ajuste de 1M
   tokens o V4-Pro usa 27 % dos FLOPs por token e 10 % do KV cache do V3.2; o V4-Flash, segundo o report,
   **10 % dos FLOPs e 7 % do KV cache** (arXiv 2606.19348, seção de eficiência).
2. **mHC — Manifold-Constrained Hyper-Connections**: resíduo com 4 "fluxos" misturados por matriz duplamente
   estocástica (Sinkhorn-Knopp, 20 iterações).
3. **Otimizador Muon** (Newton-Schulz híbrido, 10 iterações: 8 com (3.4445, −4.7750, 2.0315) e 2 com (2, −1.5, 0.5));
   AdamW só para embeddings, heads, parte estática do mHC e pesos de RMSNorm; **sem QK-Clip**, porque aplica
   RMSNorm direto em queries e entradas KV (report).
4. **Precisão mista FP4+FP8**: especialistas MoE em FP4 (MXFP4), quase todo o resto em FP8; a atenção do
   lightning indexer calculada em FP4 (report; card: "FP4 + FP8 Mixed").

Variantes relevantes para esta receita:

| Checkpoint | Revisão usada | Diferença arquitetural | `num_nextn_predict_layers` | Visão |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` (preview) | — | MTP-1 "clássico" (1 camada), `compress_ratios` com 44 entradas (43 + 1 MTP) | 1 | não |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | — | "não é um novo modelo; é o mesmo checkpoint com um módulo de speculative decoding anexado" (card) | 1 (mas 3 estágios em `mtp.*`) | não |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (`docs/DEEPSEEK_V4_FLASH_0731.md:8`) | release oficial, "mesma estrutura do DSpark" (card); agentic muito melhor | 1 | não |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (**lane atual**) | `86f746b3…` (`.env.dspark.example:149`, `README.md:136`) | + ViT 32 camadas + Aligner + `bias_vl` por camada MoE; `rms_norm_eps` 1e-20 | **3** | sim (imagem; sem vídeo) |

`scripts`/docs que confirmam a troca: `CHANGELOG.md:150` (feat/vision-exp), `README.md:10-16`.

### 1.2 Mapa de alto nível do forward (texto)

```
input_ids ─► embed (129280×4096, vocab-parallel) ─► repeat ×4  ──► H: [T, 4, 4096]   (4 fluxos mHC)
  (Vision-Exp: placeholders <｜deepseek_image｜> substituídos por embeddings do ViT+Aligner)
for layer 0..42:
    x, post, comb = mHC_pre(H)            # 4 fluxos → 1 vetor de 4096 (pesos dinâmicos por token)
    x = RMSNorm(x); x = Attention_V4(x)   # SWA / CSA(C4A) / HCA(C128A) conforme compress_ratios[layer]
    H = mHC_post(x, H, post, comb)        # 1 → 4 fluxos (Sinkhorn)
    x, post, comb = mHC_pre(H); x = RMSNorm(x); x = MoE(x, input_ids)   # hash routing nas camadas 0-2
    H = mHC_post(x, H, post, comb)
    if layer in {40,41,42}: guarda mean(H, streams) → features p/ DSpark
x = hc_head(H)  (4 → 1 com pesos sigmoid) ; x = RMSNorm(x) ; logits = lm_head(x)  (129280, não amarrado)
DSpark (3 estágios, "mtp.0..2"): features[40,41,42] → main_proj → KV de contexto (janela 128)
    + bloco [âncora, ruído×(k-1)] → 3 blocos SWA+MoE não-causais → lm_head + Markov head (seq.) → rascunho
```

Fontes: `HF:inference/model.py:927-1021`, `recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py:1860-1947`.

### 1.3 Tabela camada-a-camada (43 camadas + 3 estágios DSpark)

`compress_ratios` (cfg:66) = `[0, 0, 4, 128, 4, 128, …, 4, 128, 4, 0, 0, 0]` — 46 entradas no 0731/Vision-Exp
(43 do tronco + 3 dos estágios DSpark); 44 no preview (43 + 1 MTP). `num_hash_layers = 3` (cfg:29).

| Camada(s) | `compress_ratio` | Tipo de atenção (nome no vLLM) | Indexer? | RoPE (referência) | Roteamento MoE | Observação |
|---|---|---|---|---|---|---|
| 0, 1 | 0 → vLLM usa `max(1, ·)` = 1 | **SWA pura** (janela 128) | não | θ=10000, **sem YaRN** | **hash** (`tid2eid`) | `model.py:957-960` |
| 2 | 4 | **CSA / C4A** (esparsa top-512 sobre KV/4 + janela 128) | **sim** | θ=160000 + YaRN ×16 | **hash** | última camada hash |
| 3, 5, …, 41 (20 camadas ímpares) | 128 | **HCA / C128A** (densa sobre KV/128 + janela 128) | não | θ=160000 + YaRN | aprendido (score + bias) | |
| 4, 6, …, 42 (20 camadas pares) | 4 | **CSA / C4A** | sim | θ=160000 + YaRN | aprendido | total C4A = 21 (camadas 2..42 pares) |
| 40, 41, 42 | 4 / 128 / 4 | idem | 40,42 sim | | aprendido | **camadas-alvo do DSpark** (`dspark_target_layer_ids`, cfg:69) |
| 43, 44, 45 (estágios DSpark `mtp.0..2`) | 0 | SWA (janela 128) sobre *features do alvo* + bloco de rascunho não-causal | não | θ=10000 sem YaRN (referência) | aprendido (MoE completo, 256 exp.) | `HF:inference/model.py:800-924` |

Contagens: 2 SWA + 21 C4A + 20 C128A = 43 (confirmado em `docs/CLAUDE/fable5-1-report.md:451-456`).

### 1.4 Anatomia de UM bloco do tronco

```
H (T×4×4096) ──► hc_pre(hc_attn_fn, hc_attn_scale, hc_attn_base) ─► x (T×4096) ─► attn_norm ─► Attention ─► hc_post ─► H'
H'           ──► hc_pre(hc_ffn_fn,  hc_ffn_scale,  hc_ffn_base ) ─► x ─► ffn_norm ─► MoE ─► hc_post ─► H''
```

- Referência: `HF:inference/model.py:702-757` (`Block.forward`).
- vLLM: `recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py:1488-1628` (`_forward_cuda`) funde o `hc_post` de uma
  subcamada com o `hc_pre` da próxima (`mhc_fused_post_pre`) e funde `attn_norm`/`ffn_norm` dentro do mHC
  (comentário em `model.py:1597`). Caminho "nativo" não-fundido para ROCm/XPU em `model.py:1630-1656`.

### 1.5 A atenção do V4 em detalhe

#### 1.5.1 Formas e parâmetros (por camada, TP=1)

| Tensor | Forma (in → out) | Parâmetros | Precisão no checkpoint | TP no vLLM |
|---|---|---|---|---|
| `wq_a` | 4096 → 1024 (`q_lora_rank`) | 4.19 M | FP8 bloco 128 | **replicado** (fundido com `wkv` em `fused_wqa_wkv`, `disable_tp=True`, `model.py:972-979`) |
| `q_norm` | RMSNorm(1024) | 1 k | | replicado |
| `wq_b` | 1024 → 64×512 = 32768 | 33.55 M | FP8 | `ColumnParallelLinear` (32 heads/rank em TP=2) `model.py:981-988` |
| `wkv` | 4096 → 512 (`head_dim`) | 2.10 M | FP8 | replicado (dentro de `fused_wqa_wkv`) |
| `kv_norm` | RMSNorm(512) | 512 | | |
| `attn_sink` | 64 (um logit por head) | 64 | fp32 | fatiado por head; padding = −inf (`model.py:964-970, 2027-2034`) |
| `wo_a` | por grupo: 8 heads×512 = 4096 → 1024 (`o_lora_rank`); 8 grupos | 33.55 M (checkpoint `[8192, 4096]`) | FP8 (referência usa BF16 no einsum) | Column-parallel **por grupo** (`is_bmm`, `bmm_batch_size = n_local_groups` = 4 em TP=2) `model.py:991-1000` |
| `wo_b` | 8×1024 = 8192 → 4096 | 33.55 M | FP8 | `RowParallelLinear` + all-reduce `model.py:1008-1015` |
| Compressor (C4) | `wkv`,`wgate`: 4096 → 2×512; `ape` 4×1024; `norm` 512 | 8.39 M | BF16 no checkpoint (fp32 na ref.) | |
| Compressor (C128) | `wkv`,`wgate`: 4096 → 512; `ape` 128×512 | 4.26 M | BF16 | |
| Indexer (só C4) | `wq_b` 1024 → 64×128; `weights_proj` 4096 → 64; compressor próprio 4096 → 2×128 (×2) | 10.75 M | FP8/BF16 | **replicado** em vLLM (`ReplicatedLinear`, "no tensor parallel, just replicated", `fable5-1-report.md:306-309`) |

Soma "núcleo" da atenção ≈ 107.0 M params/camada (+19.1 M em C4A, +4.3 M em C128A). Cálculo meu a partir das formas.

#### 1.5.2 Caminho da query (Q)

```
c_Q = RMSNorm_q( x · W_qa )                  # 4096 → 1024  (latente de query, "low-rank")
q   = c_Q · W_qb  → [64 heads × 512]
q_h = q_h / RMS(q_h)                          # RMSNorm SEM peso, por head (HF:model.py:539; vLLM dspark.py:464)
RoPE nas ÚLTIMAS 64 dims de cada head (qk_rope_head_dim = 64), estilo "interleaved" (is_neox_style=False)
```
`HF:inference/model.py:537-540`. O mesmo `c_Q` (chamado `qr`) alimenta o **indexer** (`HF:model.py:555`).
O report atribui a ausência de QK-Clip exatamente a essa normalização de Q e KV.

#### 1.5.3 Caminho do KV — "MQA de KV compartilhado"

```
kv = RMSNorm_kv( x · W_kv )                   # 4096 → 512 ; UM único vetor por token (num_key_value_heads = 1)
RoPE nas últimas 64 dims (posição absoluta do token)
FP8-simulado (QAT) nas 448 dims "nope", blocos de 64, escala potência de 2 ; as 64 dims RoPE ficam BF16
```
`HF:inference/model.py:543-547` ("FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision").

**O mesmo vetor `kv` é usado como K e como V** (`sparse_attn(q, kv, …)` recebe um só tensor, `HF:model.py:571,576`;
kernel `HF:inference/kernel.py:276-352` faz `q·kvᵀ` e depois `softmax·kv`). Em vLLM isso aparece como
`kv_lora_rank=self.head_dim` e `v_head_dim=self.head_dim` no wrapper MLA (`model.py:1088-1099`): é a forma "absorvida"
do MLA levada ao extremo — não há `W_UK`/`W_UV` para absorver; o latente **é** a chave e o valor.

#### 1.5.4 RoPE inverso na saída

Como V também carrega as 64 dims rotacionadas pela posição *j* de cada chave, a saída `o_i = Σ_j a_ij · kv_j` teria
posições absolutas misturadas. A V4 aplica **RoPE com posição −i** nas últimas 64 dims de `o_i`
(`HF:inference/model.py:577`, `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` = conjugado), de modo que cada termo
passa a carregar a rotação relativa (j − i). O report descreve isso literalmente ("apply RoPE with position −i on the
last 64 dimensions of each o_{t,i}"). No vLLM do drafter DSpark: `fused_inv_rope_fp8_quant` (`dspark.py:516-525`),
que ao mesmo tempo quantiza a saída em FP8 para o einsum do `wo_a`.

#### 1.5.5 Attention sink

Cada head tem um logit aprendido `z'_h` que entra **só no denominador** do softmax:
`a_ij = exp(s_ij − m) / ( Σ_j exp(s_ij − m) + exp(z'_h − m) )` (`HF:inference/kernel.py:344-347`;
Triton do drafter `dspark_kernels.py:259-262`; referência torch `dspark_kernels.py:704-712`).
Efeito: a head pode "não olhar para nada" (massa vai para o sink) sem forçar atenção espúria no 1º token. No vLLM,
heads de padding recebem `attn_sink = -inf` para não somarem `exp(0)=1` ao denominador (`model.py:964-970`;
alerta explícito em `patches/dsv4_tp_pad.py:50-55`).

#### 1.5.6 Três "ramos" de KV e como se combinam

Para cada token-consulta *t* a lista de índices de KV visíveis é a **concatenação** (`HF:model.py:548-558`):

1. **Janela deslizante (SWA)**: os últimos `sliding_window = 128` tokens brutos (cfg:57), com cache em anel
   `kv_cache[:, pos % 128]` (`HF:model.py:563-573`; `get_window_topk_idxs` `HF:model.py:269-280`).
2. **Entradas comprimidas** (só se `compress_ratio > 0`):
   - **C4A / CSA**: o **lightning indexer** escolhe as **top-512** (`index_topk`, cfg:18) entre as `(t+1)//4`
     entradas comprimidas já "fechadas" (`HF:model.py:420-473`).
   - **C128A / HCA**: **todas** as `(t+1)//128` entradas comprimidas (`get_compress_topk_idxs`, `HF:model.py:308-316`)
     — atenção densa, mas sobre um KV 128× menor.

Tudo entra num único kernel "sparse attention por índices" com softmax online (FlashAttention-style) + sink
(`HF:inference/kernel.py:276-368`). Em vLLM/SM121: FlashInfer `BatchSparseMLAPagedAttentionWrapper` com cache SWA e
`extra_kv_cache` (comprimido) separados (`recipe/overlay/vllm/models/deepseek_v4/nvidia/sm120.py:1-17, 380-393`).

Número de entradas lidas por token-consulta a 1M de contexto (cálculo meu): SWA = 128; C4A = 128 + 512 = **640**;
C128A = 128 + 1,048,576/128 = **8,320**. Contra 1,048,576 numa atenção densa.

#### 1.5.7 O compressor (pooling aprendido com gate)

`Compressor` (`HF:inference/model.py:319-417`), equações do report (seção CSA):

- Projeções por token: `C^a = H·W^a_KV`, `C^b = H·W^b_KV`, `Z^a = H·W^a_Z`, `Z^b = H·W^b_Z` (no código: `wkv` e `wgate`
  com saída `coff·head_dim`, `coff = 2` quando ratio = 4 → metade "a"/"b").
- **C4 (com sobreposição, `overlap = compress_ratio == 4`)**: a entrada comprimida *i* é um softmax **por canal**
  sobre 2m = 8 posições — os 4 tokens do bloco atual (metade "b" das dims, via `overlap_transform`) e os 4 do bloco
  anterior (metade "a"):
  `[S^a; S^b] = Softmax_linha([Z^a_{bloco i} + B^a ; Z^b_{bloco i−1} + B^b])`,
  `C^Comp_i = Σ S^a ⊙ C^a + Σ S^b ⊙ C^b` (report). `B` = `ape` ("absolute position embedding" dentro do bloco,
  `ape: [ratio, coff·head_dim]`, `HF:model.py:334`).
- **C128 (sem sobreposição)**: softmax por canal sobre os 128 tokens do bloco: `C^Comp_i = Σ_{j∈bloco} S_j ⊙ C_j`.
- Depois: `RMSNorm` (`norm`), RoPE nas 64 últimas dims com a **posição do primeiro token do bloco**
  (`freqs_cis[:cutoff:ratio]`, `HF:model.py:403-407`), FP8-simulação das dims nope em blocos de 64 (`HF:model.py:412`).
- **Decode incremental**: estados `kv_state`/`score_state` acumulam os tokens do bloco corrente; só quando
  `(pos+1) % ratio == 0` emite 1 entrada nova (`HF:model.py:383-399`). Consequência: um token só "vê" blocos
  completos; o bloco em formação é coberto pela janela SWA de 128.
- Custo em vLLM: 3 *aux streams* CUDA executam em paralelo as GEMMs do `kv_score` do compressor, `indexer.weights_proj`
  e `kv_score` do compressor do indexer, enquanto `fused_wqa_wkv` roda no stream padrão (`model.py:1700-1709`).

#### 1.5.8 O lightning indexer (só nas camadas C4A)

`Indexer` (`HF:inference/model.py:420-473`); fórmula do report:
`I_{t,s} = Σ_{h=1}^{64} w^I_{t,h} · ReLU( q^I_{t,h} · K^IComp_s )`.

- `q^I = c_Q · W^I_UQ` → 64 heads (`index_n_heads`, cfg:17) × 128 dims (`index_head_dim`, cfg:16); RoPE nas últimas 64
  dims; **rotação de Hadamard** (`rotate_activation`, `HF:model.py:262-266`) e **FP4-simulação** blocos de 32
  (`fp4_act_quant(q, 32, True)`, `HF:model.py:454-456`).
- `K^IComp`: compressor **próprio** do indexer (ratio 4, com sobreposição, `head_dim=128`, `rotate=True` → Hadamard +
  FP4-simulação, `HF:model.py:408-410, 438`). Cache próprio `[max_seq_len/4, 128]`.
- Pesos por head `w^I = weights_proj(x) · (128^-0.5 · 64^-0.5)` (`HF:model.py:458`).
- Score com **ReLU** (não softmax), soma ponderada entre heads, máscara causal por bloco, `topk(min(512, (t+1)//4))`
  (`HF:model.py:460-472`).
- Em vLLM: `DeepseekV4Indexer` recebe um `topk_indices_buffer` compartilhado por todas as camadas C4A,
  `[max_num_batched_tokens, 512] int32` (`model.py:1712-1718`); kernels de logits do DeepGEMM
  (`fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`; `docs/PATCHES.md:1153-1169`).
- Cache K do indexer: **132 B/entrada comprimida** (128 B FP8 + 4 B de escala — composição inferida por analogia com
  o V3.2; o valor 132 é citado em `fable5-1-report.md:309` e `docs/CLAUDE/item8-fp4-kv-design.md:18`).
- Linhagem: é o "DSA" do **V3.2** (`index_n_heads 64`, `index_head_dim 128`, `index_topk 2048` em
  `scratchpad/web/v32exp-config.json`), mas lá ele escolhia 2048 *tokens* brutos; na V4 escolhe 512 *entradas
  comprimidas 4:1* (≈ 2048 tokens de cobertura) e roda em FP4.
- Atalho "contexto curto" (#49486 + guarda #52492): se `max_seq_len // 4 <= 512` (≤ 2048 tokens) todos os candidatos
  seriam escolhidos → pula scoring e escreve índices 0..n−1 (`CHANGELOG.md:210, 375`).

#### 1.5.9 Projeção de saída agrupada ("grouped low-rank O")

`o` (64 heads × 512 = 32768 dims) é dividido em `o_groups = 8` grupos de 8 heads (4096 dims); cada grupo passa por
`wo_a[g]: 4096 → 1024`; concatena 8×1024 = 8192 → `wo_b: 8192 → 4096` (`HF:inference/model.py:579-585`,
`einsum("bsgd,grd->bsgr")`). Uma projeção direta 32768→4096 teria 134 M params; a agrupada tem 67 M (metade) e
reduz FLOPs. **Contrato de forma**: `r = heads_por_grupo × head_dim = 8 × 512 = 4096` é fixo no checkpoint — é por
isso que TP=3 exige *padding de grupos* (8→9) e não de heads dentro do grupo (`docs/TP3.md:73-85`).

#### 1.5.10 Escalas e detalhes numéricos

- `softmax_scale = head_dim^-0.5 = 512^-0.5` (`model.py:1024`; `HF:model.py:505`).
- `rms_norm_eps`: **1e-06** no 0731 e no preview; **1e-20** no Vision-Exp (cfg:46; `inference/config.json`
  `norm_eps: 1e-20`). Afeta todas as RMSNorm e o `rsqrt` do mHC. Impacto prático não medido no repo (observação minha).
- MegaMoE / FlashMLA SM100 não existem no GB10: o overlay seleciona `DeepseekV4SM120SparseImpl` (`sm120.py:172-209`),
  que exige `h_q ∈ {16, 32, 64, 128}` → 32 heads locais em TP=2 cabem sem padding; 24 heads locais em TP=3 viram 32.

### 1.6 Linhagem: MHA → MLA (V2/V3) → DSA (V3.2) → V4

| | V2 | V3 | V3.2-Exp | **V4-Flash** |
|---|---|---|---|---|
| `hidden_size` / camadas | 5120 / 60 | 7168 / 61 | 7168 / 61 | **4096 / 43** |
| heads (Q) / `num_key_value_heads` | 128 / 128 (cfg nominal; na prática MLA) | 128 / 128 | 128 / 128 | **64 / 1** |
| `q_lora_rank` | 1536 | 1536 | 1536 | **1024** |
| `kv_lora_rank` | 512 | 512 | 512 | — (head_dim 512 = latente) |
| `qk_nope/rope/v_head_dim` | 128/64/128 | 128/64/128 | 128/64/128 | **448 nope + 64 rope = 512; V = K (512)** |
| Atenção | densa | densa | DSA: indexer 64×128, top-**2048** tokens | **SWA 128 + CSA (C4, top-512) / HCA (C128, densa)** |
| MoE routed/shared/top-k | 160/2/6 | 256/1/8 | 256/1/8 | **256/1/6** |
| `moe_intermediate_size` | 1536 | 2048 | 2048 | 2048 |
| Primeiras camadas | 1 densa | 3 densas | 3 densas | **3 com hash routing** (MoE) |
| `scoring_func` | softmax | sigmoid | sigmoid | **sqrtsoftplus** |
| `topk_method` | group_limited_greedy (8 grupos, top-3) | noaux_tc (8 grupos, top-4) | noaux_tc | **noaux_tc, SEM grupos** (não há `n_group` no config) |
| `routed_scaling_factor` | 16.0 | 2.5 | 2.5 | **1.5** |
| MTP | — | 1 | 1 | 1 (preview) → **DSpark 3 estágios** |
| Resíduo | padrão | padrão | padrão | **mHC 4 fluxos** |
| Contexto | 160K (YaRN ×40 de 4K) | 160K | 160K | **1M (YaRN ×16 de 64K)** |
| Pesos | BF16 | FP8 bloco 128 | FP8 + ue8m0 | **FP8 ue8m0 + experts MXFP4** |

Fonte: `scratchpad/web/v2-config.json`, `v3-config.json`, `v32exp-config.json`, `visionexp-config.json`.

**MLA do V2/V3 (para entender a V4)**: em vez de cachear K e V por head (128 heads × 2 × 128 dims), o MLA cacheia um
latente `c_KV = x·W_DKV` de 512 dims + uma chave RoPE "desacoplada" de 64 dims compartilhada por todas as heads
(total 576). No treino, K e V de cada head são reconstruídos com `W_UK`, `W_UV`. **Absorção de pesos**: na inferência,
`q_h·k_h = (q_h W_UKᵀ)·c_KV` → dá para pré-multiplicar `W_UK` na query e atender *direto* sobre o latente (vira uma
MQA com chave de 576 dims e valor de 512); `W_UV` é absorvido em `W_O`. O RoPE precisa ficar "desacoplado"
(64 dims à parte) porque uma rotação dependente de posição entre `q` e `W_UK` impediria a absorção.
**A V4 pula a reconstrução**: o latente de 512 dims (448 sem posição + 64 com RoPE embutido) *já é* K e V; o
problema das posições em V é resolvido pelo RoPE inverso na saída (1.5.4). Resultado: KV de 1 "head" de 512 dims
por camada, como no MLA absorvido, mas com projeções menores e sem up-projection.

### 1.7 MoE (DeepSeekMoE na versão V4)

Por camada: **256 especialistas roteados + 1 compartilhado**, top-6, cada especialista é um SwiGLU
4096 → 2048 → 4096 (`w1` gate, `w3` up, `w2` down) = 25.17 M params (cfg:22-27; `HF:model.py:642-699`).

**Roteamento aprendido** (camadas 3..42; `HF:inference/model.py:611-639`; vLLM `model.py:731-800, 859-898`):
```
s      = sqrt( softplus( x · W_gateᵀ ) )          # "sqrtsoftplus", fp32   (scoring_func, cfg:56)
idx    = topk( s + b , 6 )                        # b = e_score_correction_bias (aux-loss-free), só para ESCOLHER
w      = s[idx] / Σ s[idx]                        # norm_topk_prob = true (cfg:25) — pesos vêm de s SEM o bias
w     *= 1.5                                      # routed_scaling_factor (cfg:55)
y      = Σ_k w_k · Expert_idx_k(x)  +  SharedExpert(x)
```
- **Aux-loss-free** (`topk_method = "noaux_tc"`, cfg:60): o viés `b` por especialista é ajustado fora do gradiente
  durante o treino para equilibrar carga; ele muda *quem* é escolhido, não o peso. O report diz que a V4 usa essa
  estratégia "aumentada por uma leve loss de balanceamento por sequência".
- **Sem group-limited routing**: V2/V3 restringiam a escolha a `topk_group` de `n_group` grupos (para limitar
  comunicação entre nós em EP); o config e o `Gate` da V4 não têm grupos.
- **`swiglu_limit = 10.0`** (cfg:58): `up = clamp(up, −10, 10)`, `gate = clamp(gate, max=10)` antes de
  `silu(gate)·up` (`HF:model.py:655-658`; vLLM `SiluAndMulWithClamp`, `model.py:298-301`) — estabilidade numérica
  (evita ativações explosivas, importante com pesos FP4).
- **Hash routing nas camadas 0–2** (`num_hash_layers = 3`, cfg:29): `idx = tid2eid[token_id]` — uma tabela fixa
  `[129280, 6]` (int32) decide os 6 especialistas **pelo id do token**; os *pesos* ainda vêm do gate
  (`HF:model.py:622-635`; vLLM `model.py:761-775`). O report: "substituímos as FFN densas dos primeiros blocos por MoE
  com Hash routing". Consequência de engenharia: essas camadas **precisam de `input_ids`** mesmo quando o runner
  multimodal passa `inputs_embeds` (`requires_raw_input_tokens = True`, `patches/vision_exp/apply.py:559-563`).
- **`bias_vl`** (só Vision-Exp): cada camada tem um segundo viés de roteamento para tokens de imagem; tokens de imagem
  **não** usam `tid2eid` nas camadas hash e sim `topk(s + bias_vl)` (`HF:model.py:601-634`). Implementado no vLLM pela
  receita (issue #175, `patches/vision_exp/apply.py:143-300`).
- `expert_dtype = "fp4"` (cfg:9): pesos dos especialistas roteados em FP4 E2M1 com escala E8M0 por 32 elementos
  (`HF:model.py:146-152`); especialista compartilhado em FP8.
- Backends: MegaMoE (DeepGEMM `fp8_fp4_mega_moe`) exige **SM100 + expert parallel** (`model.py:455-469, 721-729`) —
  impossível no GB10; a receita usa `--moe-backend flashinfer_b12x` + `VLLM_USE_B12X_MOE=1`, que no checkpoint
  MXFP4 (`fp4_e8m0_k32`) cai no caminho **W4A16** (dequant para BF16 + MMA BF16), porque o caminho NVFP4 tensor-core do
  b12x aceita só `modelopt_nvfp4` (`fable5-1-report.md:269-289`).
- TP no vLLM: sem `--enable-expert-parallel`, o `FusedMoE` fatia cada especialista pela dimensão intermediária entre os
  ranks (cada rank tem "metade" de cada um dos 256; `fable5-1-report.md:86`: 13.4 MB por especialista, "halved by TP")
  e o compartilhado é dobrado no mesmo kernel (`model.py:841-857`). A referência oficial, ao contrário, distribui
  especialistas inteiros por rank (EP-like, `HF:model.py:671-697`).

### 1.8 mHC — Manifold-Constrained Hyper-Connections

Estado residual: **4 fluxos** (`hc_mult = 4`, cfg:11) de 4096 → `H ∈ R^{T×4×4096}`. O embedding é simplesmente
repetido 4× na entrada (`model.py:1874`; `HF:model.py:1001`).

Em cada subcamada (atenção e FFN) — `HF:inference/model.py:730-743`, `kernel.py:371-438`:
```
u      = flatten(H) ∈ R^{16384}
mixes  = (u · W_fnᵀ) · rsqrt(mean(u²) + eps)                 # W_fn: [24, 16384] (mix_hc = (2+4)·4 = 24)
pre_j  = sigmoid(mixes[j]·α0 + base[j]) + hc_eps            # 4 pesos: colapsa 4 fluxos → entrada da subcamada
post_j = 2 · sigmoid(mixes[4+j]·α1 + base[4+j])             # 4 pesos: espalha a saída nos 4 fluxos (hc_post_alpha = 2.0)
comb   = SinkhornKnopp( softmax_linha(mixes[8:24]·α2 + base[8:24]) + eps , 20 iters)   # 4×4 duplamente estocástica
x_in   = Σ_j pre_j · H_j
H'_i   = post_i · F(x_in) + Σ_j comb[j,i] · H_j
```
- `α = hc_*_scale` (3 escalares), `base = hc_*_base` (24) — parâmetros fp32 por camada (`model.py:1181-1222`).
- **Por que "manifold-constrained"**: sem restrição, a matriz de mistura residual pode amplificar sinais ao longo de
  dezenas de camadas; projetando-a no **politopo de Birkhoff** (matrizes duplamente estocásticas, linhas e colunas
  somam 1) a norma espectral fica ≤ 1 → propagação estável (report). Sinkhorn-Knopp = alternar normalização de linhas e
  colunas (`hc_sinkhorn_iters = 20`, cfg:12).
- `hc_head` final (`HF:model.py:759-766`): 4 → 1 com pesos `sigmoid(…) + eps` (sem Sinkhorn), antes do `norm` final e
  do `lm_head` (`model.py:1936-1944`).
- Custo: `W_fn` é [24 × 16384] fp32 por subcamada (≈ 0.39 M params) — minúsculo em params, mas é um passo sequencial a
  mais por subcamada; por isso existem kernels fundidos (TileLang `MHCFusedPostPreOp`; b12x opcional
  `VLLM_USE_B12X_MHC`, só para `hidden_size = 4096` e `hc_mult` do b12x, `model.py:1223-1255`).
- Buffer de estado para o MTP clássico: `_mtp_hidden_buffer [max_num_batched_tokens, 4·4096]` bf16 = 256 MiB/rank —
  inútil para DSpark e liberado pelo hotfix #50312 (`patches/hotfix-dsv4-mtp-buffer-50312.sh:8-19`).

### 1.9 RoPE, RoPE parcial e YaRN

- RoPE só nas **últimas 64 dims** de Q/K (e das entradas comprimidas e do indexer); pares consecutivos
  (`view_as_complex`, `HF:model.py:247-259`) → `is_neox_style=False` no vLLM (`model.py:1044-1049`).
- `rope_theta = 10000` (cfg:54) para camadas SWA; `compress_rope_theta = 160000` (cfg:65) para C4A/C128A
  (`HF:model.py:516-522`; vLLM `model.py:1031-1033`).
- **YaRN** (cfg:47-53): `factor 16`, `original_max_position_embeddings 65536` → 65536 × 16 = 1,048,576
  (`max_position_embeddings`, cfg:20); `beta_fast 32`, `beta_slow 1`. Fórmula (`HF:model.py:214-244`): para cada par de
  dims com frequência `f`, calcula-se quantas rotações ele completa em 64K; dims que completam > 32 rotações (alta
  frequência, "local") ficam intactas; dims com < 1 rotação (baixa frequência, "global") são divididas por 16; entre
  elas há uma rampa linear. **Sem "mscale"** (fator de temperatura do YaRN): a referência não aplica e o vLLM zera
  (`rope_parameters["mscale"] = 0`, `model.py:1040-1041`).
- **Onde o YaRN vale**: na referência, *só* nas camadas comprimidas; camadas SWA (0, 1 e os 3 estágios DSpark) usam
  RoPE puro θ=10000 ("disable YaRN and use base rope_theta in pure sliding-window attention", `HF:model.py:518-520`).
  O vLLM da Anemll promove `deepseek_yarn` para **todas** as camadas; a receita tem o fix opt-in
  `DSPARK_ENABLE_ROPE_SWA_FIX` (port de vllm#54815, default OFF; `docs/PATCHES.md:870-903`).

### 1.10 Quantização embutida no modelo (pesos, ativações, KV)

| Objeto | Formato | Granularidade de escala | Fonte |
|---|---|---|---|
| Linears densas (atenção, shared expert, gate?) | **FP8 E4M3** | bloco **128×128**, escala `ue8m0` (potência de 2) | cfg:36-45 |
| Ativações antes de GEMM FP8/FP4 | FP8 dinâmico | 1 escala por bloco de 128 ao longo de K (`act_quant(x, 128, …)`) | `HF:model.py:125-135` |
| Especialistas roteados | **FP4 E2M1** (`float4_e2m1fn_x2`, 2 valores/byte) | escala **E8M0 por 32** (≈ MXFP4; "`fp4_e8m0_k32`") | cfg:9; `HF:model.py:146-152`; `fable5-1-report.md:272` |
| KV "nope" (448 dims) | FP8 E4M3 **simulado no treino (QAT)** | escala potência de 2 por **64** | `HF:model.py:546-547` |
| KV RoPE (64 dims) | BF16 | — | idem; report "BF16 para dims RoPE, FP8 para o resto" |
| Query/K do indexer | FP4 simulado (QAT), após **Hadamard** | 32 | `HF:model.py:408-410, 454-456` |
| lm_head, embeddings, RMSNorm, mHC, compressor `wkv/wgate` | BF16 (fp32 para mHC/ape) | — | comentários `HF:model.py:203, 335, 778` |

Por que **QAT** importa para servir: o modelo *treinou* vendo KV em FP8 (blocos de 64) e indexer em FP4 → guardar o KV
cache em FP8 e rodar o indexer em FP4 não é uma aproximação extra; é o regime do treino. Por que **Hadamard**: a rotação
ortogonal espalha "outliers" entre dimensões antes de quantizar em 4 bits (menos erro com a mesma escala por bloco).
Por que **UE8M0** (escala = 2^e): multiplicar/dividir por potência de 2 é exato e barato (só expoente); os kernels do
cache guardam `e + 127` num byte (`common/ops/cache_utils.py:104-126, 290-300`).

### 1.11 Layout físico do KV cache e por que 1M cabe

#### 1.11.1 A "linha" de 584 bytes

Cada entrada de KV (bruta na SWA ou comprimida C4/C128) é o vetor de 512 dims:
`448 B (FP8 e4m3, 7 blocos de 64) + 128 B (64 × BF16 RoPE) + 8 B (7 escalas UE8M0 + 1 pad) = 584 B`
(`recipe/overlay/vllm/models/deepseek_v4/common/ops/cache_utils.py:43-51, 142-194`;
`docs/CLAUDE/item8-fp4-kv-design.md:11`).

- **`nvfp4_ds_mla` = `fp8_ds_mla` byte a byte** na V4 (mesmos 584 B). O nome "nvfp4" só muda o *dispatch* de kernel
  (vLLM `kv_cache_interface.py` citado em `fable5-1-report.md:342-346`; `alignment = 584 if nvfp4_ds_mla else 576`).
  O issue #22 existia porque `nvfp4_ds_mla` caía no caminho BF16 lento (~1 tok/s a 600K) em vez do FP8 (~17 tok/s)
  (`docs/PATCHES.md:169-205`). **Não há KV de 4 bits hoje**; um layout real de 292 B foi desenhado (opção A:
  448 e2m1 = 224 B + 4 escalas por 128 + RoPE FP8 64 B) mas não implementado (`item8-fp4-kv-design.md:23-41`).
- Diferente do V3.2 (`fp8_ds_mla` = 656 B/token: 512 FP8 + 16 B de 4 escalas fp32 + 128 B RoPE), citado no overlay
  `recipe/overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py:4-8` ("656 B/token, head_size = 576").
- Página comum do alocador híbrido: 64 linhas × 584 B = **37,376 B**. Todos os grupos de KV precisam do mesmo tamanho
  de página: MLA C4 = 256 tokens/4 = 64 linhas (por isso `--block-size 256`); SWA = 64 tokens; grupos SWA do drafter com
  blocos 4/8 e mesma página. Boot mostra `1× MLAAttentionSpec (block 256) + 3× SlidingWindowMLASpec (64/4/8)`
  (`item8-fp4-kv-design.md:15`; `fable5-1-report.md:470-471`). *Como exatamente as camadas C128A se encaixam nesses
  grupos não está documentado no repo.*

#### 1.11.2 Bytes por token (cálculo, TP=2 replica tudo nos dois ranks)

| Componente | Fórmula | Bytes/token |
|---|---|---|
| 21 camadas C4A (KV comprimido 1/4) | 21 × 584 / 4 | 3,066 |
| 20 camadas C128A (1/128) | 20 × 584 / 128 | 91 |
| Cache K do indexer (21 camadas, 1/4) | 21 × 132 / 4 | 693 |
| **Total que cresce com o contexto** | | **≈ 3,850 B ≈ 3.76 KiB** |
| Janelas SWA (43 + 3 camadas × 128 linhas × 584 B) | constante por sequência | ≈ 3.44 MB/seq |

Mesmos números em `docs/CLAUDE/item8-fp4-kv-design.md:17-21`. Para **1,048,576 tokens**: 3,850 × 1,048,576 ≈
**4.04 GB (3.76 GiB)** por sequência. O boot reporta `Available KV cache memory: 17.04 GiB`,
`GPU KV cache size: 2,331,430 tokens`, `Maximum concurrency for 1,048,576 tokens per request: 2.22x`
(`README.md:117-121`) — a contabilidade do alocador dá ≈ 7.8 KB/token (2× o custo físico, porque os 4 grupos têm
páginas de mesmo tamanho), então a concorrência "real" a 1M é maior que 2.22× (observação do próprio doc:
"allocator accounting across the four groups, not the physical row cost", `item8:20-21`).

Comparações (cálculo meu, mesmas 43 camadas):
- Se a V4 **não** comprimisse (MLA/MQA 584 B por token por camada): 43 × 584 = 25.1 KB/token → 1M ≈ 26.3 GB > pool.
- V3.2 (61 camadas × 656 B + indexer 61 × 132 B) ≈ 48 KB/token → a V4-Flash usa ≈ 8 % disso, coerente com o
  "7 % do KV cache do V3.2" do report.
- GQA-8 BF16 típico (8 heads KV × 128 × K+V × 2 B = 4 KB/camada) × 43 ≈ 176 KB/token → 1M ≈ 185 GB. O report cita
  "≈ 2 %" do baseline GQA8 BF16.

**Por que o KV não diminui com TP**: há 1 só "head" de KV (MQA). Não dá para fatiar por heads; cada rank guarda o KV
latente inteiro e o indexer inteiro (`README.md:559-563`, `docs/TP3.md:131-135`). TP só reduz o KV *por rank* de forma
indireta (menos pesos por rank → mais memória livre para KV: TP=3 dá ≈ 35 GiB/rank, 5.0 M tokens, 4.8× a 1M,
`docs/TP3.md:129`). DCP (context parallel de decode) dividiria tokens entre ranks mas não está ligado para os backends
DSV4 nesta imagem (`fable5-1-report.md:347-351`).

#### 1.11.3 Custo de leitura por token-consulta a 1M (cálculo meu)

- Atenção: C4A 640 × 584 B × 21 ≈ 7.9 MB; C128A 8,320 × 584 B × 20 ≈ 97 MB; SWA ≈ 0.15 MB.
- **Indexer**: varre *todas* as 262,144 chaves comprimidas × 132 B × 21 camadas ≈ **0.73 GB**. A 128K: ≈ 91 MB.
  → em contextos muito longos o indexer domina os bytes de atenção (e, no prefill, os FLOPs: ≈ 80 TFLOP por chunk de
  1024 tokens a 900K, duplicado nos 2 GPUs, `fable5-1-report.md:304-322`). É exatamente o que o SP-indexer e o
  cache MXFP4 do indexer atacam.

### 1.12 De MTP a DSpark (o drafter embutido)

#### 1.12.1 MTP (V3 / V4-preview)

Multi-Token Prediction: módulos extras (`num_nextn_predict_layers`) que, a partir do estado oculto do token *t* e do
embedding do token *t+1*, preveem *t+2*; densificam o sinal de treino e servem de rascunho para speculative decoding.
No vLLM, `model_type deepseek_v4` sem campos DSpark vira `deepseek_mtp` / `DeepSeekV4MTPModel` com
`n_predict = num_nextn_predict_layers` (`recipe/overlay/vllm/config/speculative.py:336-341`). O paper DSpark usa
"MTP-1" como baseline de produção.

#### 1.12.2 DSpark — arquitetura (checkpoint 0731 / Vision-Exp)

Config: `dspark_block_size = 5`, `dspark_noise_token_id = 128799`, `dspark_target_layer_ids = [40, 41, 42]`,
`dspark_markov_rank = 256` (cfg:67-70). Os estágios ficam no namespace histórico `mtp.0`, `mtp.1`, `mtp.2`
(3 estágios; `infer_dspark_num_draft_layers`, `recipe/overlay/vllm/v1/spec_decode/dspark.py:450-479`). No 0731 o
config diz `num_nextn_predict_layers = 1` apesar dos 3 estágios; no Vision-Exp diz 3.

Componentes (referência `HF:inference/model.py:793-924`; overlay `recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py`):

1. **Features do alvo**: após as camadas 40, 41, 42 do tronco guarda-se `mean(H, dim=streams)` (média dos 4 fluxos mHC)
   → concat 3 × 4096 = 12288 (`HF:model.py:1005-1010`; vLLM `model.py:1898-1917`; buffer `_dspark_hidden_buffer`
   `model.py:1782-1802`; variante "deferred/exact" com kernel `dspark_hc_post_mean`, `dspark_kernels.py:301-490`).
2. **`main_proj` + `main_norm`** (só no estágio 0): Linear 12288 → 4096 + RMSNorm → `H_ctx`
   (`dspark.py:634-643, 708-711`). É a "injeção de contexto" do DFlash: K,V das camadas do rascunho sobre o contexto são
   `W_K·H_ctx`, `W_V·H_ctx` (paper DSpark).
3. **Cache de contexto do rascunho**: cada estágio projeta `H_ctx` com seu próprio `wkv → kv_norm → RoPE → FP8-sim` e
   guarda numa **janela de 128** (anel por posição `pos % 128`) — `DSparkAttention` com `compress_ratio == 0`
   (`HF:model.py:800-842`). No overlay: buffer interno `main_kv_cache [max_num_seqs, 128, 512]` fora do KV do vLLM
   (`dspark.py:281-292, 396-453`); na Anemll há grupos SWA de KV do drafter e `precompute_and_store_context_kv`
   (`patches/hotfix-vllm-dspark-swa-prefix.py:7`).
4. **Bloco de rascunho**: entrada = `[âncora, ruído, ruído, ruído, ruído]` onde âncora = último token aceito e ruído =
   `dspark_noise_token_id` (`HF:model.py:901-908`; `dspark.py:888-895`); posições `âncora_pos + 0..block-1`
   (`dspark.py:897-906`, comentário "anchor + [0, gamma)").
5. **Atenção não-causal dentro do bloco**: cada query do bloco vê as `min(128, pos+1)` entradas de contexto + **todas** as
   `block_size` entradas do bloco (`get_dspark_topk_idxs`, `HF:model.py:793-797`; kernel Triton `dspark_kernels.py:121-298`
   com `KV_TOKENS = WINDOW_SIZE + BLOCK_SIZE`). Tudo em **um único forward paralelo** para as k posições.
6. **3 blocos "V4 completos"**: cada estágio tem mHC, atenção SWA (mesma geometria: 64 heads × 512, `wo_a/wo_b`
   agrupados, sink) e **MoE completo de 256 especialistas** (`DeepseekV4MoE`, `dspark.py:614-789`). Na entrada da
   atenção o overlay tira a média dos fluxos (`attn_in.mean(dim=1)`, `dspark.py:759-760`).
7. **Cabeça**: `hc_head → norm → lm_head` (lm_head e embedding **compartilhados** com o alvo, `HF:model.py:951-954`)
   → logits base por posição.
8. **Markov head (módulo sequencial leve)**: `bias(prev) = W2 · W1[prev_token]` com `W1: V×256`, `W2: 256×V`
   (rank 256). Amostragem **da esquerda para a direita**: token_{i+1} = sample(logits_base[i] + bias(token_i))
   (`HF:model.py:845-854, 910-924`; `dspark.py:543-588, 977-985`). Por que: posições paralelas "marginalizam" os
   tokens anteriores → combinações de sufixo incoerentes ("multi-modal collision", paper); o viés de transição
   condiciona cada posição no token realmente escolhido antes dela, a custo de uma GEMV de rank 256.
9. **Confidence head**: `c_k = sigmoid( w·[h_k ; W1[x_{k−1}]] )` (Linear 4096+256 → 1, fp32; `HF:model.py:857-865`;
   `dspark.py:591-611`). Supervisionado com `c*_k = 1 − ½‖p_draft − p_target‖₁` (probabilidade analítica de aceitação);
   sobrevivência do prefixo `a_{j} = Π_{i≤j} c_i` (paper; `cumulative_survival`, `v1/spec_decode/dspark.py:542-550`).

Parâmetros do drafter (cálculo meu): 3 × ~6.58 B (cada estágio ≈ uma camada SWA do tronco) + `main_proj` 50 M +
Markov 2 × 33 M ≈ **19.8 B** (fora dos 284B do card); ativos por token de rascunho ≈ 3 × 0.28 B.

Treino (paper): `L = 0.1·L_ce + 0.9·L_tv + 1.0·L_conf`, todas ponderadas por `exp(−(k−1)/γ)`; calibração posterior
por temperatura por posição (STS). O paper também reporta, para V4-Flash, DSpark-5 60–85 % mais rápido por usuário que
MTP-1 a mesmo throughput (e DSpark > DFlash em 16.3–18.4 % de comprimento aceito em Qwen3). *Obs.: o resumo que obtive
do paper fala em "5 transformer layers with MoE" no deployment, o que não bate com os 3 estágios do checkpoint; confio
no checkpoint (3 estágios `mtp.0..2`, `inference/config.json` `n_mtp_layers: 3`).*

#### 1.12.3 Verificação (rejection sampling) e agendamento por confiança

- O alvo verifica as k posições do rascunho **num único forward** (k+1 tokens por sequência). Com
  `draft_sample_method = "probabilistic"`, aceita x com prob. `min(1, p(x)/q(x))` e, na rejeição, reamostra de
  `norm(max(0, p − q))` — preserva exatamente a distribuição do alvo; `"greedy"` trata q como one-hot
  (`recipe/overlay/vllm/config/speculative.py:262-268`). Aceitação esperada por posição = `Σ_x min(p,q) = 1 − TV(p,q)`
  (`v1/spec_decode/dspark.py:508-539`).
- Agendamento (Stage-C overlay; default desligado): `VLLM_DSPARK_CONFIDENCE_SCHEDULER` ∈ {off, threshold, hardware,
  auto}; *threshold* admite o prefixo mais longo com sobrevivência ≥ limiar; *hardware* admite tokens em ordem
  decrescente de sobrevivência maximizando `E[aceitos] × steps_per_second(batch_tokens)` com curva perfilada
  `VLLM_DSPARK_SPS_CURVE` (`dspark_proposer.py:175-275`; `v1/spec_decode/dspark.py:553-685`). Receita oficial do
  Vision-Exp usa `"enable_adaptive_verification": true` e `num_speculative_tokens: 3` (card, linha 125 do README
  baixado) — **a receita da Mia não usa** (a flag não aparece no repo).
- Mapeamento no vLLM: Stage-C transforma `deepseek_v4` + `dspark_block_size` em `deepseek_v4_dspark` com
  `n_predict = dspark_block_size` (`config/speculative.py:321-335`); a **Anemll 0.1.1** usa
  `n_predict = num_nextn_predict_layers` e aplica a regra de MTP "k > n_predict ⇒ k % n_predict == 0"
  (fixture `scripts/fixtures/dspark-block-k/speculative-752a3a504-stock.py:336-341, 922-934`). Resultado:
  0731 (n_predict=1) roda k=5; **Vision-Exp (n_predict=3) exige k ∈ {6, 9, …}** → `MTP_NUM_TOKENS=6`
  (`start-deepseek-v4-flash-dspark.sh:254-275`, `README.md:142, 281`). Como o drafter prevê a posição k com
  "âncora + k−1 queries de ruído", k=6 vai uma posição além do bloco treinado (5), com aceitação baixa nessa posição
  (`docs/PATCHES.md:835-867`). Unlock opt-in `DSPARK_ENABLE_DSPARK_BLOCK_K=1` + `MTP_NUM_TOKENS=5`.
- CUDA graphs: capture size = `MAX_NUM_SEQS × (k+1)` arredondado para múltiplo de 8 → 48 a 6×6
  (`docker-compose.dspark.yml:506`); o vLLM arredonda tamanhos para múltiplos de `k+1 = 7`, e 42 truncava para 40,
  deixando o passo de 6 sequências sem graph (`fable5-1-report.md:169-196`).

### 1.13 Caminho de visão (Vision-Exp)

Config (cfg:71-80): `vision_n_layers 32`, `vision_dim 1024`, `vision_n_heads 16`, `vision_inter_dim 2816`,
`vision_patch_size 14`, `vision_rope_theta 10000`, `vision_downsample_ratio 3`, `vision_max_n_token 384`,
`vision_min_pixels 147456` (= 384²), `vision_max_wh_ratio 8`.

1. **Pré-processamento** (`patches/vision_exp/image_processor.py:398-448`, cópia do oficial): RGB; limita razão W/H a 8;
   se < 147,456 px, amplia; escolhe resolução múltipla de 14 que caiba em ≤ 384 tokens LLM (`safe_resize`,
   `solve_resize_ratio`); faz *pad* cinza (127) com `ImageOps.pad`; normaliza `(x−0.5)/0.5`; recorta patches 14×14×3.
2. **ViT** (`patches/vision_exp/vision.py:101-119`): `PatchEmbed` Linear 588 → 1024; 32 blocos pré-norm (RMSNorm) com
   atenção **bidirecional** (SDPA, 16 heads × 64) e **RoPE 2D** (metade das freqs para linha, metade para coluna,
   `rope_dim = 1024/16/2 = 32`, `vision.py:17-24, 106`), MLP SwiGLU 1024 → 2816 → 1024; RMSNorm final.
   ≈ 412 M params (cálculo meu).
3. **Aligner / projetor** (`vision.py:122-136`): agrupa 3×3 patches vizinhos (`F.unfold`, `downsample_ratio 3`)
   → 9 × 1024 = 9216 → Linear → 4096 → GELU → Linear 4096. **1 token LLM = região 42×42 px**. ≈ 54.5 M params.
4. **Layout "N" dos tokens** (`build_image_block`, `image_processor.py:459-488`): cada linha da grade recebe um token
   `IMAGE_NEW_LINE`; linhas são pareadas e percorridas coluna a coluna (`view(rows//2, 2, row_len).transpose(1,2)`),
   então **4 tokens consecutivos = bloco espacial 2×2**; `compress_pad = 3 − start_pos % 4` tokens de pad antes do
   `IMAGE_START` alinham o corpo da imagem a múltiplo de 4 (`image_processor.py:45-47`). *Interpretação minha:* isso
   faz cada entrada comprimida C4 da atenção (4 tokens) cobrir exatamente um quadrado 2×2 da imagem. O tamanho do bloco
   depende de `start_pos % 4` (ex.: grade ViT 40×19 → 122–125 tokens, `image_processor.py:50-63`) — origem do bug #172.
5. **Tokens especiais aprendidos**: `image_start`, `image_end`, `image_newline`, `image_pad` (vetores de 4096)
   (`apply.py:67-81, 88-125`).
6. **No prompt**: 1 placeholder `<｜deepseek_image｜>` (id **129264**, dentro do vocabulário de 129,280;
   `image_processor.py:36-39`) expandido para `num_tokens` placeholders; `embed_input_ids` troca-os pelos embeddings
   (`apply.py:363-414`).
7. **Roteamento MoE de imagem**: `bias_vl` + sem hash (1.7).
8. **Divergência que encontrei**: a referência oficial dá aos tokens de uma imagem **visibilidade bidirecional dentro do
   span `[IMAGE_START, IMAGE_END]`** e alarga a janela SWA em até `vision_max_n_token` (`get_image_visible`,
   `get_window_topk_idxs_visible`, `HF:inference/model.py:283-305, 548-551, 994-999`) e exige o span pré-preenchido num
   único chunk (`HF:model.py:999`). **Não encontrei isso no hotfix da receita** (grep por "visible"/"bidirectional" em
   `patches/` só acha o ViT) — no vLLM os tokens de imagem seguem atenção causal com janela 128. Não há medição no repo
   do impacto; fica como ponto em aberto.
9. Sem vídeo: GIF = 1º frame (`README.md:14-15, 227`). Imagens só em mensagens `user` (`patches/hotfix-dsv4-vision-exp.py:17-24`).

### 1.14 Parâmetros totais vs ativos (conferência)

Cálculo meu a partir das formas do config:

| Bloco | Params |
|---|---|
| Embedding 129,280 × 4096 | 0.53 B |
| lm_head (não amarrado) | 0.53 B |
| Atenção por camada (núcleo) | 107.0 M; + 19.1 M (C4A) / + 4.3 M (C128A) |
| MoE por camada (256 × 25.17 M + shared 25.17 M + gate 1.05 M) | 6.469 B |
| mHC por camada | 0.79 M |
| **43 camadas** (2 SWA + 21 C4A + 20 C128A) | ≈ 283.3 B |
| **Total tronco** | **≈ 284.3 B** (card: 284B) |
| Ativos por token: por camada ≈ 107 M + compressor/indexer + 7 × 25.17 M (6 roteados + shared) ≈ 285–303 M; × 43 + lm_head | **≈ 13.2 B** (card: 13B) |
| Drafter DSpark (3 estágios) | ≈ 19.8 B (não incluso nos 284B) |
| Visão (ViT 412 M + Aligner 54.5 M) | ≈ 0.47 B |

Memória de pesos: especialistas FP4+E8M0 = 4.25 bits/param → 277 B × 0.531 B ≈ 147 GB; resto FP8/BF16 + drafter
≈ 20 GB → checkpoint ≈ **155–157 GiB** ("157 GiB checkpoint", `README.md:81, 178`; Anemll: "155.43 GiB per node").
Não cabe num GB10 de ~121 GiB utilizáveis → **mínimo 2 nós** (Anemll README). Em TP=2: **79.17 GiB de pesos/rank**;
TP=3: 60.69 GiB/rank (`fable5-1-report.md:56-57`).

### 1.15 Paralelismo tensor (TP) — o que é fatiado e o que é replicado (TP=2)

| Componente | Estratégia | Coletiva |
|---|---|---|
| `embed_tokens` | vocab-parallel | all-reduce |
| `fused_wqa_wkv` (Q-latente + KV) | **replicado** | — |
| `wq_b`, `wo_a` (por grupo), `attn_sink` | fatiado por heads/grupos (32 heads, 4 grupos por rank) | — |
| `wo_b` | row-parallel | **all-reduce** (43/passo) |
| KV cache (SWA + comprimido) | **replicado** (1 head KV) | — |
| Indexer (`wq_b`, `weights_proj`, cache K) | **replicado** | — |
| MoE (roteados + shared) | TP dentro de cada especialista | **all-reduce** (43/passo) |
| `lm_head` | vocab-parallel | **all-gather** dos logits (alvo + rascunho) |
| Markov head do DSpark (Anemll) | `VocabParallelEmbedding` + `ParallelLMHead` | 12 coletivas **seriais** por passo (6 posições × 2) |

Total ≈ 108 coletivas por passo de decode, 30–50 µs cada em RoCE sem GPUDirect ≈ 3–5 ms/passo
(`fable5-1-report.md:102-115`). **TP=3** exige padding: 8 grupos não dividem por 3 → grupos 8→9 (heads 64→72, 3 grupos
e 24 heads por rank, 8 heads/grupo preservado), grupo extra com `wo_a/wo_b` zerados e `attn_sink = -inf`
(`docs/TP3.md:73-85`; injetado por `patches/tp3/apply_tp3_patch.py:22-50`). *Atenção:* o arquivo
`patches/dsv4_tp_pad.py:15-38` contém uma análise antiga contraditória (replicar grupos e fatiar heads dentro deles);
o módulo efetivamente injetado pelo `apply_tp3_patch.py` descreve o contrário e diz que essa abordagem é rejeitada
pelo kernel (`einsum.hpp:164`). O log de boot esperado: `DSv4 TP pad: heads 64 -> 72 for TP=3` (`README.md:555`).

### 1.16 Um passo de decode com DSpark, de ponta a ponta (TP=2, k=6, 1 sequência)

Síntese minha, ancorada em `fable5-1-report.md:82-116`, `dspark_proposer.py:759-1042` e `model.py:1860-1947`:

1. **Entrada do alvo**: 7 tokens (o último aceito + 6 do rascunho anterior) × 1 sequência → embedding vocab-parallel
   (all-reduce) → repetido nos 4 fluxos mHC.
2. **43 camadas**: em cada uma, mHC-pre → atenção (Q-latente e KV replicados; `wq_b`/`wo_a` fatiados; KV novo quantizado e
   gravado no anel SWA; compressor acumula estado e, a cada 4/128 posições, emite entrada comprimida; nas C4A o indexer
   varre o cache K comprimido e escolhe 512 entradas) → `wo_b` all-reduce → mHC-post → mHC-pre → MoE (gate fp32 →
   top-6 → kernels W4A16 do b12x lendo ≤ 42 especialistas distintos; all-reduce) → mHC-post. Nas camadas 40/41/42 copia
   a média dos fluxos para o buffer DSpark.
3. **Cabeça do alvo**: `hc_head` → RMSNorm → `lm_head` vocab-parallel → **all-gather** dos logits (7 × 129,280).
4. **Rejection sampler**: compara as 6 propostas com as distribuições do alvo; aceita um prefixo (τ ≈ 3.8–4.4 tokens
   incluindo o bônus) e amostra o token seguinte.
5. **Proposer DSpark**: descarta as features das posições rejeitadas (máscara GPU ou trim), atualiza a janela de
   contexto do rascunho com as features aceitas (`prefill_main`), monta o bloco `[âncora, ruído×5]` e roda os 3
   estágios num único forward (grafo PIECEWISE) → logits base (lm_head compartilhado, all-gather) → loop sequencial de 6
   passos do Markov head (na Anemll: 2 coletivas por passo) → 6 novos tokens de rascunho.
6. Próximo passo volta ao item 1. Custo dominante: ≈ 16.5–17 GB de pesos lidos por rank.

### 1.17 Um chunk de prefill longo (o que escala com o contexto)

Para um chunk de 1024 tokens numa conversa já com N tokens: o **indexer** calcula 1024 × (N/4) scores × 64 heads × 128
dims em cada uma das 21 camadas C4A e faz top-512 por linha — O(N) por token, O(N²) no prompt inteiro, e **duplicado nos
dois ranks** (≈ 80 TFLOP por chunk a 900K, `fable5-1-report.md:304-322`); as C128A atendem densamente a N/128 entradas;
o MoE processa 1024 × 6 pares token-especialista por camada em W4A16. Por isso o prefill cai de ≈ 2,563 tok/s (2K) para
≈ 875 tok/s (900K) enquanto o decode c=1 fica quase plano (≈ 62–83 tok/s de 256 a 128K).

---

## 2. Inventário de parâmetros

### 2.1 `config.json` (HF) — todos os campos

Valores: **Vision-Exp** (V) / **0731** (0) / **preview Flash** (P) quando diferem. Linha = `cfg:L` no arquivo Vision-Exp
baixado (`scratchpad/web/visionexp-config.json`, idêntico à revisão fixada `86f746b3`).

| nome | valor(es) | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `architectures` | `["DeepseekV4ForCausalLM"]` | cfg:2-4 | classe carregada pelo vLLM (`recipe/overlay/vllm/models/deepseek_v4/__init__.py`) | registro de modelos | V e 0 usam a mesma classe; a visão é monkey-patch (`patches/vision_exp/apply.py`) |
| `model_type` | `deepseek_v4` | cfg:21 | seleciona config/código V4; no spec-decode vira `deepseek_v4_dspark` (Stage-C) ou `DSparkDraftModel` (Anemll) | registry | se não houvesse `dspark_block_size`, viraria MTP (`config/speculative.py:321-341`) |
| `attention_bias` | false | cfg:5 | sem bias nas projeções | linear | — |
| `attention_dropout` | 0.0 | cfg:6 | irrelevante em inferência | regularização | — |
| `bos_token_id` / `eos_token_id` | 0 / 1 | cfg:7-8 | tokens especiais | tokenização | `<｜begin▁of▁sentence｜>` / `<｜end▁of▁sentence｜>` (API HF) |
| `vocab_size` | 129280 | cfg:64 | tamanho do vocabulário (mesmo do V3) | embeddings, lm_head | lm_head BF16 = 1.06 GB; logits all-gather de 129,280 × tokens por passo (10.9 MB a c=6, `fable:110`) |
| `hidden_size` | 4096 | cfg:15 | largura do fluxo residual (por fluxo mHC) | d_model | b12x mHC só suporta 4096 (`model.py:1235-1239`) |
| `num_hidden_layers` | 43 | cfg:28 | camadas do tronco | profundidade | camadas 43-45 são DSpark "virtuais" |
| `num_attention_heads` | 64 | cfg:26 | heads de query | MHA/MQA | TP precisa dividir 64 e `o_groups`; TP=3 → pad 72 |
| `num_key_value_heads` | 1 | cfg:30 | **MQA**: um único vetor KV por token/camada | MQA/MLA | KV não fatiável por TP → replicado em todos os ranks |
| `head_dim` | 512 | cfg:13 | dimensão de cada head Q e do vetor KV (= K = V) | MLA absorvido | linha de cache 584 B; `kv_lora_rank = head_dim` no wrapper (`model.py:1097`) |
| `qk_rope_head_dim` | 64 | cfg:35 | quantas dims (as últimas) recebem RoPE | RoPE parcial | 448 dims "nope" em FP8 + 64 RoPE em BF16 |
| `q_lora_rank` | 1024 | cfg:34 | dimensão do latente de query `c_Q` (`wq_a`) | low-rank Q | reutilizado pelo indexer |
| `o_groups` | 8 | cfg:32 | nº de grupos da projeção de saída | grouped O-proj | TP deve dividir 8 (TP=3 → pad 9) |
| `o_lora_rank` | 1024 | cfg:33 | dimensão intermediária por grupo (`wo_a`) | low-rank O | contrato `r = 8 heads × 512 = 4096` |
| `sliding_window` | 128 | cfg:57 | janela SWA (todas as camadas + drafter) | SWA | cobre o bloco comprimido ainda "aberto"; custo SWA constante 3.4 MB/seq |
| `compress_ratios` | 46 entradas: `[0,0,4,128,4,…,128,4,0,0,0]` (P: 44 entradas, termina `4,0`) | cfg:66 | razão de compressão por camada (0 = só SWA, 4 = CSA/C4A, 128 = HCA/C128A) | atenção híbrida | 21 C4A (indexer), 20 C128A, 2 SWA + 3 SWA do drafter |
| `compress_rope_theta` | 160000 | cfg:65 | base RoPE das camadas comprimidas | RoPE | posições comprimidas "andam" mais devagar; θ maior = comprimentos de onda maiores |
| `rope_theta` | 10000 | cfg:54 | base RoPE das camadas SWA (e drafter) | RoPE | — |
| `rope_scaling.type` | yarn | cfg:52 | extensão de contexto | YaRN | só nas camadas comprimidas na referência (ver fix RoPE SWA) |
| `rope_scaling.factor` | 16 | cfg:50 | 64K × 16 = 1,048,576 | YaRN | define o teto de 1M |
| `rope_scaling.original_max_position_embeddings` | 65536 | cfg:51 | contexto de pré-treino "curto" | YaRN | — |
| `rope_scaling.beta_fast` / `beta_slow` | 32 / 1 | cfg:48-49 | limites da rampa de interpolação | YaRN | dims de alta freq. intactas; baixa freq. ÷16 |
| `max_position_embeddings` | 1048576 | cfg:20 | contexto máximo | — | casa com `MAX_MODEL_LEN=1048576` |
| `index_n_heads` | 64 | cfg:17 | heads do lightning indexer | DSA | indexer replicado por rank em vLLM |
| `index_head_dim` | 128 | cfg:16 | dim por head do indexer | DSA | cache K 132 B/entrada |
| `index_topk` | 512 | cfg:18 | nº de entradas comprimidas selecionadas por token (C4A) | sparse attention | `topk_indices_buffer [8192, 512]` int32; V3.2 usava 2048 tokens |
| `n_routed_experts` | 256 | cfg:23 | especialistas roteados por camada | MoE | 13.4 MB cada em MXFP4 |
| `n_shared_experts` | 1 | cfg:24 | especialista sempre ativo | DeepSeekMoE | dobrado no `FusedMoE` |
| `num_experts_per_tok` | 6 | cfg:27 | top-k do roteador | MoE | 7 especialistas ativos/token; c=1 decode k=6 lê ≤ 42 distintos/camada |
| `moe_intermediate_size` | 2048 | cfg:22 | largura intermediária do SwiGLU de cada especialista | FFN | — |
| `hidden_act` | silu | cfg:14 | ativação do SwiGLU | SwiGLU | vLLM só aceita silu (`model.py:294-297`) |
| `swiglu_limit` | 10.0 | cfg:58 | clamp de gate/up | estabilidade | `SiluAndMulWithClamp` |
| `scoring_func` | sqrtsoftplus | cfg:56 | afinidade token→especialista = √softplus(logit) | roteamento | substitui sigmoid do V3 |
| `topk_method` | noaux_tc | cfg:60 | seleção com viés aux-loss-free | balanceamento | cria `e_score_correction_bias` (`model.py:776-780`) |
| `norm_topk_prob` | true | cfg:25 | normaliza pesos dos top-6 para somar 1 | roteamento | — |
| `routed_scaling_factor` | 1.5 | cfg:55 | multiplica os pesos roteados | roteamento | V3 usava 2.5 |
| `num_hash_layers` | 3 | cfg:29 | camadas 0-2 roteiam por `tid2eid[token_id]` | hash routing | exige `input_ids` no forward (inclusive multimodal) |
| `expert_dtype` | fp4 | cfg:9 | especialistas roteados em FP4 (e2m1 + E8M0/32) | MXFP4 | define mapeamento de nomes de escala (`model.py:2073-2108`) e backend MoE |
| `quantization_config.quant_method` | fp8 | cfg:39 | pesos densos FP8 | FP8 | — |
| `quantization_config.fmt` | e4m3 | cfg:38 | formato FP8 | E4M3 | máx 448 |
| `quantization_config.weight_block_size` | [128,128] | cfg:41-44 | uma escala por bloco 128×128 | block quant | — |
| `quantization_config.scale_fmt` | ue8m0 | cfg:40 | escala = potência de 2 (expoente 8 bits) | UE8M0 | preservado por `VLLM_USE_B12X_WO_PROJECTION` (Stage-C, `model.py:1001-1007`) |
| `quantization_config.activation_scheme` | dynamic | cfg:37 | ativações quantizadas em runtime | quant dinâmica | — |
| `rms_norm_eps` | **V: 1e-20**; 0/P: 1e-06 | cfg:46 | epsilon das RMSNorm e do mHC | RMSNorm | mudança do Vision-Exp; impacto não medido no repo |
| `hc_mult` | 4 | cfg:11 | nº de fluxos residuais mHC | hyper-connections | `H` tem 16,384 floats/token; buffer MTP 256 MiB |
| `hc_sinkhorn_iters` | 20 | cfg:12 | iterações Sinkhorn-Knopp | matriz duplamente estocástica | kernel sequencial por token |
| `hc_eps` | 1e-06 | cfg:10 | epsilon do mHC (pre/Sinkhorn) | — | — |
| `tie_word_embeddings` | false | cfg:59 | lm_head separado do embedding | — | +0.53 B params |
| `torch_dtype` | bfloat16 | cfg:61 | dtype de ativações | BF16 | `--dtype` implícito |
| `transformers_version` | V: 5.0.0; 0/P: 4.57.1 | cfg:62 | metadado | — | — |
| `initializer_range` | 0.02 | cfg:19 | só treino | — | — |
| `use_cache` | true | cfg:63 | — | — | — |
| `num_nextn_predict_layers` | **V: 3**; 0/P: 1 | cfg:31 | nº de estágios MTP declarados | MTP / DSpark | na Anemll vira `n_predict` → k divisível por 3 (V) |
| `dspark_block_size` | 5 (V e 0; ausente em P) | cfg:67 | tamanho do bloco treinado do rascunho (γ) | DSpark | k < 5 trunca; k=5 é o shape treinado |
| `dspark_noise_token_id` | 128799 | cfg:68 | token "máscara/ruído" nas posições 1..γ−1 do bloco | parallel drafting | também usado para padding de CUDA graph (`dspark_proposer.py:496-503`) |
| `dspark_target_layer_ids` | [40, 41, 42] | cfg:69 | camadas do alvo cujas features (média dos 4 fluxos) alimentam o rascunho | DFlash-style context injection | `main_proj` 12288 → 4096 |
| `dspark_markov_rank` | 256 | cfg:70 | rank do Markov head (W1: V×256, W2: 256×V) | bias de transição low-rank | 66 MB por matriz (BF16) → replicável |
| `vision_n_layers` | 32 (só V) | cfg:71 | blocos do ViT | ViT | > 0 ativa `_install_vision_tower` e `bias_vl` |
| `vision_dim` | 1024 | cfg:72 | largura do ViT | ViT | — |
| `vision_n_heads` | 16 | cfg:73 | heads do ViT (64 dims) | MHA bidirecional | — |
| `vision_inter_dim` | 2816 | cfg:74 | MLP SwiGLU do ViT | — | — |
| `vision_patch_size` | 14 | cfg:75 | patch 14×14 px | patchify | — |
| `vision_rope_theta` | 10000.0 | cfg:76 | base do RoPE 2D do ViT | RoPE 2D | — |
| `vision_downsample_ratio` | 3 | cfg:77 | Aligner agrupa 3×3 patches → 1 token LLM (42×42 px) | pixel-unshuffle / projetor | 9× menos tokens |
| `vision_max_n_token` | 384 | cfg:78 | teto de tokens LLM por imagem (inclui start/end/newlines/pad) | orçamento visual | `get_max_image_tokens` (`processor.py:99-100`) |
| `vision_min_pixels` | 147456 | cfg:79 | área mínima (384²); imagens menores são ampliadas | resize | — |
| `vision_max_wh_ratio` | 8 | cfg:80 | razão W/H máxima | resize | panorâmicas são recortadas/redimensionadas |

Campos com **nome diferente** no `inference/config.json` da referência (`scratchpad/web/vx-inference_config.json`):
`dim`=hidden_size, `moe_inter_dim`, `n_layers`, `n_hash_layers`, **`n_mtp_layers: 3`**, `n_activated_experts`,
`score_func`, `route_scale: 1.5`, `norm_eps: 1e-20`, `window_size: 128`, `original_seq_len: 65536`, `rope_factor: 16`,
`dtype: "fp8"`, `scale_fmt: "ue8m0"`, `expert_dtype: "fp4"`.

### 2.2 Constantes "hard-coded" que materializam a arquitetura (vLLM overlay / kernels)

| nome | valor | onde | o que é | por que importa |
|---|---|---|---|---|
| `TOKEN_FP8_DIM` / `TOKEN_BF16_DIM` / `TOKEN_SCALE_DIM` | 448 / 64 / 8 | `common/ops/cache_utils.py:170-175` | layout da linha KV | 448 B + 128 B + 8 B = 584 B |
| `QUANT_BLOCK_SIZE` | 64 | `cache_utils.py:173` | bloco de escala FP8 do KV | casa com a QAT (`act_quant(kv, 64)`) |
| `FP8_MAX` | 448.0 | `cache_utils.py:174` | máximo do E4M3 | clamp na quantização |
| codificação UE8M0 | `exponent + 127` em uint8; escala = `2^ceil(log2(amax/448))`, `amax ≥ 1e-4` | `cache_utils.py:99-126` | escala potência de 2 arredondada para cima | não satura; dequant exato |
| `n_quant_blocks` | 8 (7 reais + 1 pad) | `cache_utils.py:41, 129` | 7 escalas + 1 byte de pad | fecha os 8 B |
| int64 em `block_idx * block_stride` | — | `cache_utils.py:69-71, 244-246` | evita overflow com ≥ 57K blocos | 1M de contexto |
| `_SPARSE_PREFILL_TOPK_ALIGNMENT` | 128 | `cache_utils.py:502-507` | top-k combinado (512 + 128 SWA = 640) alinhado a 128 | contrato FlashMLA prefill |
| `_DECODE_MAX_TOKENS` | 64 | `nvidia/sm120.py:44` | até 64 linhas o wrapper usa kernel de decode | fronteira do bug #141 |
| `_DECODE_SPLIT_TILE` | 64 | `sm120.py:45` | split-K do decode (`cdiv(topk, 64)` fatias) | scratch `mid_out/mid_lse` |
| `_C128A_TOPK_ALIGNMENT` | 128 | `sm120.py:46, 61-68` | capacidade C128A = `cdiv(cdiv(max_model_len,128),128)*128` = 8192 a 1M | tamanho dos índices C128A |
| `get_padded_num_q_heads` | 16/32/64/128 | `sm120.py:196-209` | kernel exige h_q nessas classes | 32 locais em TP=2 |
| `hc_post_alpha` | 2.0 | `model.py:1178` | `post = 2·sigmoid(·)` | igual à referência |
| `mix_hc` | (2 + hc_mult)·hc_mult = 24 | `model.py:1179` | nº de coeficientes mHC por token | — |
| `topk_indices_buffer` | `[max_num_batched_tokens, index_topk]` int32 = [8192, 512] | `model.py:1712-1718` | índices do indexer compartilhados entre camadas C4A | 16 MiB |
| `_mtp_hidden_buffer` | `[max_num_batched_tokens, 4·4096]` bf16 = 256 MiB | `model.py:1766-1780` | estado para MTP clássico | inútil no DSpark (#50312 remove) |
| `_dspark_hidden_buffer` | `[max_num_batched_tokens, 3·4096]` | `model.py:1782-1802` | features-alvo do DSpark | — |
| `main_kv_cache` (drafter, Stage-C) | `[max_num_seqs, 128, 512]` bf16 | `dspark.py:281-292` | janela de contexto do rascunho por slot | Patch 1 (slot estável) |
| `sparse_scores` (drafter) | `[max_num_seqs, block, heads_locais, 128+block]` fp32 | `dspark.py:293-304` | scratch do kernel Triton | — |
| `_einsum_recipe` | (1,128,128) se SM ≤ 9, senão (1,1,128) | `dspark.py:218` | granularidade do einsum FP8 do `wo_a` | Blackwell usa escalas por linha |
| `_DSPARK_SCORE_K_BLOCK` / `_DSPARK_MARKOV_V_BLOCK` / `_R_BLOCK` / `_HC_POST_H_BLOCK` | 8 / 256 / 32 / 64 | `dspark_kernels.py:11-15` | tiles Triton | — |
| `B12X_MHC_MAX_TOKENS` | default 16 | `model.py:234-241` | até 16 tokens usa mHC b12x, acima TileLang | só Stage-C (`VLLM_USE_B12X_MHC`) |
| `IMAGE_TOKEN_ID` | 129264 | `patches/vision_exp/image_processor.py:39` | id do `<｜deepseek_image｜>` | cai no `tid2eid` se não for tratado (#175) |
| `COMPRESS_PAD_TO` | 4 | `image_processor.py:35` | alinhamento do bloco de imagem | = razão C4 |

### 2.3 Flags do `vllm serve` que dependem da arquitetura (`docker-compose.dspark.yml:494-529`)

| flag | valor | por que é determinada pela arquitetura |
|---|---|---|
| `--tensor-parallel-size` | `${TP_SIZE:-2}` (l.498) | pesos 155 GiB > 1 nó; TP deve dividir 64 heads e 8 grupos (TP=3 só com pad) |
| `--kv-cache-dtype` | `nvfp4_ds_mla` (l.500) | formato "DS-MLA" de 584 B da V4 (efetivamente FP8, ver 1.11) |
| `--block-size` | 256 (l.501) | 256 tokens / 4 = 64 linhas comprimidas × 584 B = página comum de 37,376 B |
| `--max-model-len` | 1048576 (l.502) | teto YaRN ×16 de 64K |
| `--max-cudagraph-capture-size` | `((seqs·(k+1)+7)/8)·8` = 48 (l.506) | verificação DSpark lê k+1 tokens/seq; arredondamento a múltiplo de k+1 exigia 48 |
| `--speculative-config` | `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}` (l.479, 513) | drafter embutido no checkpoint (`mtp.*`); k=6 por `n_predict=3` |
| `--moe-backend` | `flashinfer_b12x` (l.517) | especialistas MXFP4 em SM121 (MegaMoE é SM100+EP) |
| `--attention-config {"use_fp4_indexer_cache":true}` | só se `DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1` (l.459) | indexer foi treinado em FP4 (QAT) |
| `--limit-mm-per-prompt` | `{"image":8}` (l.489) | Vision-Exp; ≤ 384 tokens por imagem |
| `--tokenizer-mode deepseek_v4`, `--reasoning-parser deepseek_v4`, `--tool-call-parser deepseek_v4` | l.514-520 | checkpoint sem template Jinja; encoder Python `encoding_dsv4.py` |
| `--enable-prefix-caching` | l.508 | compatível com SWA via retenção (#26) e drafter (fix SWA prefix) |
| `--enable-chunked-prefill`, `--long-prefill-token-threshold 1024` | l.505, 512 | prefill longo (1M) em chunks; indexer quadrático no prefill |

### 2.4 Variáveis de ambiente ligadas à arquitetura

**Pista padrão (Anemll) — `.env.dspark` / compose**

| nome | default | onde | o que faz | conceito |
|---|---|---|---|---|
| `MTP_NUM_TOKENS` | 6 | `.env.dspark.example:429`; validação `start-deepseek-v4-flash-dspark.sh:254-275` | k do DSpark | ≥ 5 e múltiplo de 3 (Vision-Exp) |
| `DRAFT_SAMPLE_METHOD` | probabilistic | `docker-compose.dspark.yml:312, 475-479` | amostragem do rascunho | rejection sampling com q completo vs one-hot; A/B 2026-09-02: empate (`fable5-1-report.md:544`) |
| `DSPARK_ENABLE_DSPARK_BLOCK_K` | 0 | `.env.dspark.example:437` | libera k=5 com n_predict=3 | bloco treinado γ=5 |
| `DSPARK_ENABLE_ROPE_SWA_FIX` | 0 | `.env.dspark.example:447` | RoPE puro nas camadas SWA/drafter | YaRN só em CSA/HCA |
| `DSPARK_ENABLE_DSPARK_SWA_PREFIX` | 0 | `.env.dspark.example:295` | recalcula a janela do rascunho em hit de prefix cache | janela de contexto do drafter |
| `DSPARK_ENABLE_SP_INDEXER` / `DSPARK_SP_INDEXER_MIN_KEYS` | 0 / 8192 | `.env.dspark.example:402, 413` | indexer sequence-parallel no prefill (≥ 8192 chaves = 32K tokens) | SP + merge top-k exato |
| `DSPARK_ENABLE_MXFP4_INDEXER_CACHE` | 0 | `.env.dspark.example:471` | cache K do indexer em MXFP4 | FP4 QAT |
| `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS` | 0 | `.env.dspark.example:419` | headers `sm121_*` → `sm120_*` | JIT DeepGEMM |
| `DSPARK_ENABLE_C128A_PREFILL_CACHE` | 0 | `.env.dspark.example:475` | reutiliza conversão de índices C128A entre camadas | HCA |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 256 (stock 512) | `docker-compose.dspark.yml:177`; `.env.dspark.example:598` | workspace de logits do indexer | memória vs nº de lançamentos (A/B 512: empate, `fable5-1-report.md:545`) |
| `VLLM_USE_B12X_MOE` | 1 | `docker-compose.dspark.yml:191` | MoE MXFP4 via b12x | W4A16 |
| `B12X_W4A16_TC_DECODE` | 0 | `docker-compose.dspark.yml:196` (proposto em `fable:291`) | decode small-M empacotado | A/B: pior (46 vs 55 tok/s) |
| `CUTE_DSL_ARCH` / `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | `sm_121a` / `12.1a` / `12.1a` | `docker-compose.dspark.yml:198-204` | alvo dos JITs | GB10 = SM 12.1 |
| `LIMIT_MM_PER_PROMPT` | `image=8` → `{"image":8}` | compose l.489 | nº de imagens | visão |
| `TP_SIZE` / `TP3_MAX_NUM_SEQS` | 2 / 16 | `docs/TP3.md` | TP=3 aciona `apply_tp3_patch.py` | padding de grupos |
| `ABLATE` / `DSV4_ABLATE_LAMBDA` / `DSV4_ABLATE_LAYERS` | 0 / 3.5 / `10-42` | `patches/hotfix-dsv4-runtime-ablation.py:27-30` | projeta fora uma direção de 4096 dims na saída da atenção: `x ← x − λ(x·v)v` | abliteration (direção capturada no 0731, transferência "experimental", `README.md:178`) |

**Pista Stage-C (overlay) — registradas em `recipe/overlay/vllm/envs.py`; *no-op* na Anemll**

| nome | default | onde é lido | efeito |
|---|---|---|---|
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | 0.0 | `dspark_proposer.py:176-190` | limiar de sobrevivência para cortar o prefixo verificado |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | auto (→ off se limiar 0) | `dspark_proposer.py:192-204` | off / threshold / hardware |
| `VLLM_DSPARK_SPS_CURVE` | "" | `dspark_proposer.py:226-257` | curva `batch_tokens:steps_per_second` p/ agendador hardware |
| `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP` | 1 | `dspark_proposer.py:259-262` | para na 1ª queda de throughput |
| `VLLM_DSPARK_FORCE_DRAFT_LENGTH` | "" | `dspark_proposer.py:206-223` | força comprimento (profiling) |
| `VLLM_DSPARK_EXPORT_DRAFT_PROBS` | 0 | `dspark_proposer.py:284-287, 733-757` | exporta softmax do rascunho (só greedy) |
| `VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS` / `VLLM_DSPARK_POSITION0_DIAGNOSTICS` | 0 | `dspark_proposer.py:289-297` | diagnósticos (copiam p/ CPU) |
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | 0 | `dspark_proposer.py:300-302, 834-851` | mascara sufixo rejeitado na GPU (caminho "ragged" do Patch 2) |
| `VLLM_DSPARK_STAGE_TIMING(_LOG_EVERY)` | 0 / 20 | `dspark_proposer.py:305-319` | tempos por estágio (sincroniza) |
| `VLLM_DSPARK_LOCAL_ARGMAX` / `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | 0 | `dspark.py:800-803, 927-963` | argmax vocab-parallel local / fundido com Markov |
| `VLLM_DSPARK_REPLICATE_MARKOV_W1` | 0 | `dspark.py:547-565` | Markov W1 replicado (remove all-reduce) |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | 0 | `dspark.py:213-215, 312-319` | aplica a mesma FP8-simulação da referência no KV do drafter |
| `DSPARK_SLOT_CLAMP` | 1 | `dspark.py:66-107` | clamp de `slot_index` fora do intervalo (degrada para rascunho rejeitado em vez de crash) |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE(_EXACT)` | 0 | `model.py:99-104, 1804-1835` | captura das features-alvo fundida no `hc_post_pre` da camada seguinte |
| `VLLM_DSPARK_TARGET_TIMING(_LOG_EVERY)` | 0 / 20 | `model.py:105-176` | tempos por estágio do alvo |
| `VLLM_USE_B12X_MHC` / `B12X_MHC_MAX_TOKENS` / `B12X_TRACE_MHC(_LIMIT)` | 0 / 16 / 0 / 240 | `model.py:80-87, 224-241` | mHC via b12x (SM120, hidden 4096) |
| `VLLM_USE_B12X_WO_PROJECTION` | 0 | `model.py:1001-1022, 2066-2070` | kernel b12x fundido para `wo_a/wo_b` com escalas UE8M0 |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | 0 | `sm120.py:76-77, 342-369` | decode MLA comprimido via b12x `sm120_unified` |
| `VLLM_DSV4_TP_PAD` | 1 | `patches/dsv4_tp_pad.py:74-75`; `apply_tp3_patch.py:61` | liga o padding em TP não divisível |
| `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` | 1024 | `recipe/overlay/vllm/envs.py:281` | acima disso não usa overlap multi-stream das GEMMs de entrada | chunks mistos > 1024 perdem o overlap (`fable:329-331`) |


---

## 3. Inventário de modificações (as que tocam a arquitetura)

> O inventário completo de patches operacionais está em `inventory/ds-mods.md`. Aqui listo o que muda (ou expõe)
> **a matemática do modelo, o layout do KV, o roteamento, o drafter ou a geometria de paralelismo**.

### 3.A Visão (Vision-Exp em cima de uma imagem só-texto)

#### A1. Instalação nativa da torre de visão — `patches/hotfix-dsv4-vision-exp.py` + `patches/vision_exp/*`
- **Problema**: o `DeepseekV4ForCausalLM` da Anemll 0.1.1 é só texto; Vision-Exp traz ViT 32 camadas + Aligner +
  tokens `<｜deepseek_image｜>` (`hotfix-dsv4-vision-exp.py:1-10`).
- **Mecanismo**: (1) injeta um hook no fim de `nvidia/model.py` que chama `apply_vision_exp(...)`
  (`vision_exp/apply.py:426-585`): envolve `DeepseekV4Model.__init__` para construir `vision`/`aligner` e os 4
  parâmetros `image_start/end/newline/pad` (`apply.py:67-81`); filtra pesos `vision.*`/`aligner.*`/`image_*` para fora do
  mapeamento "empilhado" (`w1`→`gate_up_proj`), porque o Aligner usa `w1` como Linear inteiro
  (`image_processor.py:173-187`); estende o `WeightsMapper` (`apply.py:26-64`); registra processador multimodal
  próprio (`vision_exp/processor.py:89-299`); adiciona `SupportsMultiModal` ao MRO (`apply.py:551-558`), com
  `get_language_model()` devolvendo o próprio wrapper para o DSpark/Eagle3 continuar achando `.model`
  (`apply.py:567-583`). (2) remapeia `mtp.*.ffn.gate.bias_vl` → `e_score_correction_bias_vl` no loader do drafter
  (`hotfix-dsv4-vision-exp.py:53-72`). (3) relaxa o encoder para aceitar placeholders já substituídos e restringe imagens
  a mensagens `user` (`hotfix-dsv4-vision-exp.py:17-24`).
- **Conceitos**: ViT, RoPE 2D, projetor/aligner, placeholder tokens, `inputs_embeds`, encoder cache, monkey-patching.
- **Efeito**: imagem nativa via `image_url`; max 384 tokens por imagem; `LIMIT_MM_PER_PROMPT` 8 (`CHANGELOG.md:150`).
- **Ponto em aberto**: a máscara bidirecional dentro do span de imagem da referência não é replicada (1.13 item 8).

#### A2. Issue #175 — tokens de imagem roteados com `bias_vl`, sem hash
- **Problema**: `e_score_correction_bias_vl` era carregado mas `fused_topk_bias` sempre usava o viés de texto +
  `tid2eid`; como o id 129264 está no vocabulário, **nas camadas hash 0–2 todos os tokens de imagem iam para os mesmos
  6 especialistas** `tid2eid[129264]` (`CHANGELOG.md:130`).
- **Mecanismo**: `fused_topk_bias_split_vl` classifica o lote em text/image/mixed e roteia linhas de imagem com
  `bias_vl` e `hash_indices_table=None` (`apply.py:143-224`); o "tipo" é calculado **uma vez por forward** em
  `embed_input_ids` a partir de `is_multimodal.sum()` (sem sync extra) e publicado num `kind_cell` compartilhado pelas
  43 gates (`image_processor.py:94-170`, `apply.py:332-414`); durante captura de CUDA graph usa o caminho de texto
  (grafos de decode só veem texto; `apply.py:262-274`).
- **Conceito**: roteamento condicionado à modalidade; hash routing; custo de `.item()` (sync host) em grafos.
- **Efeito**: corrige colapso de especialistas para imagens (sem número de qualidade no repo).

#### A3. Issue #172 — hash do encoder cache "salgado" com `num_tokens`
- **Problema**: o bloco de imagem tem 122 + `compress_pad` (0–3) tokens; `compress_pad = 3 − start_pos % 4` depende da
  posição; o cache de encoder do vLLM usava só os bytes da imagem → a mesma foto em outro offset reaproveitava o layout
  errado (125 placeholders vs 124 embeddings) e matava o EngineCore (`CHANGELOG.md:134`).
- **Mecanismo**: `salt_mm_image_hash(digest, num_tokens)` (`image_processor.py:66-68`, `processor.py:57-75, 140-156`) +
  verificação de contagem em `embed_input_ids` (`apply.py:392-403`).
- **Conceito**: alinhamento do bloco de imagem à compressão C4 (1.13 item 4); encoder cache content-addressed.

### 3.B Atenção, KV cache e indexer

#### B1. Issue #22 — `nvfp4_ds_mla` roteado para o kernel FP8
- `flashmla_sparse.py`: `use_fp8_cache = kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")` (`docs/PATCHES.md:169-205`;
  `patches/hotfix-nvfp4-ds-mla-issue22.sh`, `patches/fix-nvfp4-ds-mla-long-context.patch`).
- **Por que é correto**: o layout de 584 B é idêntico (1.11.1). **Efeito medido**: decode a 600K+ de ~1 tok/s → ~17 tok/s.

#### B2. RoPE SWA fix (opt-in `DSPARK_ENABLE_ROPE_SWA_FIX`) — `patches/hotfix-vllm-rope-swa-fix.py`
- **Problema**: `build_deepseek_v4_rope` promove YaRN ×16 para *todas* as camadas, inclusive SWA (0, 1 e 3 do drafter);
  a referência (`HF:inference/model.py:516-522`) e transformers#45892 usam RoPE puro nelas (`docs/PATCHES.md:870-885`).
- **Mecanismo**: port byte-exato do vllm#54815: YaRN só se `compress_ratio > 1`; demais camadas `deepseek_yarn` com
  `factor=1.0` (identidade); cópia de dict por camada (não muta `config.rope_parameters` compartilhado).
- **Conceito**: YaRN, RoPE por tipo de camada. **Efeito**: não medido; gate recomendado de 128K+ antes de ligar.

#### B3. Issue #26/#36 — prefix cache com grupos SWA (v2) — `patches/hotfix-dsv4-issue26-hybrid-swa-min.py`
- **Contexto arquitetural**: 4 grupos de KV (1× MLA + 3× SlidingWindowMLA); o hit de prefix cache é o **mínimo** entre
  grupos; SWA só guarda a cauda de 128 tokens (`issue26…py:4-19`).
- **Mecanismo v2**: SWA pode encolher o hit comum; hits longos vêm de "caudas SWA esparsas" retidas a cada
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` tokens (`docker-compose.dspark.yml:136`). v1 (anexar SWA nulo) produzia
  "DSML/CJK salad" (#36).
- **Conceito**: KV híbrido (janela + global), prefix caching, correção vs taxa de hit.

#### B4. Item 6 — Lightning indexer **sequence-parallel** no prefill (opt-in `DSPARK_ENABLE_SP_INDEXER`)
- **Problema**: indexer replicado → cada rank pontua *todas* as chaves comprimidas; O(queries × keys) domina o prefill
  longo (900K ≈ 875 tok/s vs ≈ 2,500 a 2K) (`docs/PATCHES.md:1142-1152`).
- **Mecanismo** (`patches/hotfix-dsv4-sp-indexer-prefill.py` em `sparse_attn_indexer.py`): para chunks com
  ≥ `DSPARK_SP_INDEXER_MIN_KEYS` (8192) chaves, o rank *r* pega uma fatia contígua alinhada a página das chaves, roda
  `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` local, faz all-gather de `(score, id)` (tp × 512 × 8 B por query) e
  funde com o `stable_topk_from_gathered_candidates_cutedsl` do DCP (`PATCHES.md:1153-1178`). **Exato**: um token do
  top-k global está no top-k local do rank dono.
- **Efeito medido**: 128K TTFT 76.0/74.4 s vs 79.0/81.2 s (≈ −6 %, bem abaixo dos 1.5–1.8× estimados), ruler-lite 8/8
  (`fable5-1-report.md:536`).

#### B5. Cache K do indexer em MXFP4 (opt-in `DSPARK_ENABLE_MXFP4_INDEXER_CACHE`)
- **Mecanismo**: `patches/hotfix-vllm-mxfp4-indexer-cache.py` alarga o assert "Blackwell datacenter only" de
  `v1/attention/backends/mla/indexer.py` para `is_device_capability_family(120)`; compose passa
  `--attention-config {"use_fp4_indexer_cache":true}` (`docs/PATCHES.md:1218-1260`). Exige o alias DeepGEMM (B6).
- **Conceito**: o indexer foi treinado em FP4 (QAT); MXFP4 = e2m1 + escala UE8M0 por 32. **Efeito esperado**: metade dos
  bytes de K lidos por chave pontuada (a alocação ainda é de 132 B/linha); −9 % de KV físico só com follow-up.
  Receitas oficiais em GB300/B300 usam esse flag (cards 0731/DSpark).

#### B6. Alias de headers DeepGEMM SM121 (opt-in) — `patches/hotfix-deepgemm-sm121-mqa-header-alias.sh`
- DeepGEMM gera `sm121_fp8_mqa_logits` mas só distribui `sm120_*`; produção funciona com cubins persistidos; um cache
  frio falha (`item8-fp4-kv-design.md:79-94`). Escreve 4 headers alias. **Conceito**: JIT de kernels por arquitetura (SM).

#### B7. C128A prefill metadata cache (opt-in) — `patches/hotfix-vllm-c128a-prefill-cache.py`
- Nas camadas HCA os índices "todas as entradas comprimidas" são iguais para todas as 20 camadas no mesmo forward;
  guarda a conversão local→slot físico no metadata e reutiliza (`docs/PATCHES.md:1262-1289`). C4A continua por camada
  (índices do indexer diferem). Sem speedup E2E declarado.

#### B8. Atalhos do indexer / compressor (backports do upstream)
- **#49486 + guarda #52492**: se `max_seq_len // 4 <= 512` (≤ 2048 tokens), pula o scoring (todos seriam escolhidos) —
  mas **não durante captura de CUDA graph** (um grafo capturado com o atalho seria replay com prefixos longos →
  candidatos 0..511 sem score) (`CHANGELOG.md:210, 375`).
- **#50004 removido**: empacotamento de largura top-k adaptativa do C128A causava corrupção silenciosa com MTP +
  prefix cache + CUDA graphs (stride de captura vs stride vivo) (`CHANGELOG.md:210`).
- **#48957** pula lançamentos vazios do compressor C128; **#48407** (skip do indexer em prefill denso) está dormente;
  **#50298** reusa workspace de índices combinados top-k+SWA (ver `ds-mods.md`).

#### B9. Issue #141 — decode sparse-MLA fatiado em ≤ 64 linhas (opt-in)
- Morte estocástica no fallback "paged" do FlashInfer SM120 para > 64 linhas; o patch fatia `_forward_decode` em views de
  ≤ 64 linhas (fronteira `_DECODE_MAX_TOKENS = 64`) (`docs/PATCHES.md:493-595`). Workaround, não causa-raiz.

#### B10. Overlay Stage-C `nvidia/sm120.py` e `common/ops/cache_utils.py`
- `DeepseekV4SM120SparseImpl` (`sm120.py:186-519`): um único `wrapper.run` por chunk, cache SWA como `kv_cache` e cache
  comprimido como `extra_kv_cache`; C4A converte `topk_indices_buffer` local → slots globais por camada
  (`compute_global_topk_indices_and_lens`), C128A usa índices pré-computados no metadata (`sm120.py:309-324, 423-452`);
  scratch de split-K emprestado do workspace compartilhado para não alocar por camada em contextos C128A grandes
  (`sm120.py:94-106`).
- `cache_utils.py`: kernels Triton de quantizar+inserir (UE8M0) e dequantizar+coletar o K cache (`cache_utils.py:23-388`);
  mapeamento de índices com guarda contra blocos stale (`cache_utils.py:391-499`); combinação top-k + janela SWA com
  alinhamento 128 (`cache_utils.py:502-628`).

### 3.C Paralelismo

#### C1. TP=3 — padding de grupos/heads — `patches/tp3/apply_tp3_patch.py` (só `TP_SIZE=3`)
- **Problema**: 64 heads e 8 grupos não dividem por 3; o vLLM recusa ou (pior) usa `8 // 3 = 2` grupos locais e descarta o
  resto em silêncio (`docs/TP3.md:73-78`).
- **Mecanismo**: grupos 8 → 9, heads 64 → 72 (8 por grupo mantido → `r = 4096` preservado para o einsum FP8);
  grupo 9 com `wo_a/wo_b` zero e `attn_sink = -inf`; no-op em TP=1/2 (`apply_tp3_patch.py:22-50`).
- **Efeito**: KV ≈ 35 GiB/rank (5.0 M tokens), decode +4–13 %/stream, 16 slots ≈ 198 tok/s agregados; prefill +4–13 %
  até 64K e +22–23 % a 128K–256K porque atenção/indexer são replicados (`docs/TP3.md:116-135`).

### 3.D DSpark (drafter)

#### D1. Implementação do drafter no vLLM (overlay Stage-C) — `nvidia/dspark.py`, `nvidia/dspark_kernels.py`, `v1/spec_decode/dspark*.py`
- Segue a referência oficial (docstring `dspark.py:3-10`): cache de contexto interno (não usa o KV do vLLM),
  `initialize_attn_backend` define `block_size = 1` e ignora o KV config (`dspark_proposer.py:369-376`); CUDA graph
  PIECEWISE do loop de rascunho (`dspark_proposer.py:415-431`); primeiro passo retorna rascunho de `noise_token_id`
  como "warmup" (`dspark.py`/`v1/spec_decode/dspark.py:38-67`, `dspark_proposer.py:968-970`); kernels Triton: atenção
  esparsa com sink sobre `[janela ; bloco]` (`dspark_kernels.py:121-298`), argmax fundido `base + Markov`
  (`dspark_kernels.py:18-85, 511-601`), quant-dequant FP8 das dims nope (`dspark_kernels.py:88-118, 635-674`),
  média de `hc_post` sem materializar 4 fluxos (`dspark_kernels.py:301-490`).
- Carregamento: nomes `mtp.{i}.*` viram `model.layers.{43+i}.*` (`dspark.py:1117-1126`; `dspark_start_layer_idx =
  max(num_hidden_layers, max(target_ids)+1)`, `dspark.py:804-807`); o alvo pula `mtp.` (`model.py:2179-2181`).

#### D2. "Keys concurrency patch" — Patch 1 / 2 / 2b (`patches/keys-concurrency.patch`, `docs/PATCHES.md:1-138`)
- **Patch 1 (slot estável)**: o `main_kv_cache [max_num_seqs, 128, 512]` era indexado por *linha do batch*; o vLLM v1
  "condensa" o batch quando requests terminam → request lia o anel de outra → aceitação → 0. Agora `_req_id_to_slot`
  mapeia request → slot fixo, e o slot vai para o grafo via `_draft_slot_index_buffer` (`dspark_proposer.py:79-91,
  529-549`; `dspark.py:396-453, 495-505`).
- **Patch 2 / 2b (caminho "ragged")**: chunked prefill mistura prefill e decode → nº de linhas por request desigual;
  o reshape retangular quebrava (HTTP 500 "41 rows for batch_size=2"). Agora usa `query_start_loc` e
  `_store_main_kv_ragged` com endereçamento por posição `pos % 128` (`dspark.py:333-394`;
  `dspark_proposer.py:834-907`). Validação GSM8K N=8: 0 erros, 93.5 % vs 95.0 % sequencial, 97.5 % de concordância.
- **Conceitos**: continuous batching, estado por request em speculative decoding, chunked prefill.

#### D3. Block-k unlock (opt-in `DSPARK_ENABLE_DSPARK_BLOCK_K`) — `patches/hotfix-vllm-dspark-block-k.py`
- Adiciona `self.method != "dspark"` à regra de divisibilidade MTP (`docs/PATCHES.md:833-867`); com `MTP_NUM_TOKENS=5`.
- **Medido (2026-09-03, Vision-Exp)**: greedy 8K 56.6 vs 55.3 tok/s; 32K 56.6 vs 53.6; temp 0.6 49.9–56.9 vs 48.6–53.1;
  c=4 agregado igual; aceitação por posição igual (0.88/0.74/0.53/0.36/0.24) → ganho = passo mais barato.

#### D4. DSpark draft SWA prefix fix (opt-in) — `patches/hotfix-vllm-dspark-swa-prefix.py`
- **Problema**: em hit de prefix cache o alvo não recomputa o prefixo → o drafter não recebe as features-alvo desse
  prefixo → janela de 128 do rascunho vazia → rascunho degenerado aceito (ex.: `["json` + stop).
- **Mecanismo**: limita `max_cache_hit_length` a `num_tokens − 1 − dspark_window_size` (recalcula 128 tokens)
  (`docs/PATCHES.md:905-941`). Upstream: repetido json60 12.1 → 86.5 tok/s.

#### D5. Markov head replicado — `patches/hotfix-dsv4-replicate-markov-head.py`
- `VocabParallelEmbedding` + `ParallelLMHead` → cópia inteira (2 × 66 MB) para remover 12 coletivas seriais/passo
  (`hotfix-dsv4-replicate-markov-head.py:1-13`; `fable5-1-report.md:241-267`). **A/B 2026-09-02**: aceitação igual,
  sem ganho em c=1 e c=6 agregado 122 vs 148 → deixado OFF (`fable5-1-report.md:535`).

#### D6. #50312 — liberar `_mtp_hidden_buffer` (256 MiB/rank) — `patches/hotfix-dsv4-mtp-buffer-50312.sh`
- DSpark usa features médias das camadas 40–42, não o buffer pré-hc_head do MTP; o backport condiciona a alocação e
  adiciona guardas `None` que faltam upstream (`…50312.sh:8-25`).

#### D7. Captura diferida das features-alvo (Stage-C) — `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE(_EXACT)`
- Em vez de quebrar a fusão `hc_post_pre` depois das camadas 40/41/42 para materializar `H`, a camada seguinte grava
  `mean(hc_post(...))` (modo exato via `dspark_hc_post_mean`) ou a média do resíduo (modo "fused")
  (`model.py:1260-1308, 1804-1835`).

### 3.E Outros que tocam o forward

- **Runtime ablation** (`patches/hotfix-dsv4-runtime-ablation.py`): em camadas 10–42 aplica `x ← x − λ(x·v)v` (fp32,
  λ = 3.5) na saída da atenção antes do `hc_post` (`…ablation.py:103-107`); a direção (18 KiB) foi capturada no 0731
  FP8 DSpark.
- **b12x mHC / WO projection** (Stage-C): kernels alternativos para mHC (≤ 16 tokens) e `wo_a/wo_b` fundidos
  preservando escalas UE8M0 (`model.py:224-257, 1001-1022`).

---

## 4. Números e benchmarks

### 4.1 Memória (boot)

| Lane | Data | Pesos/rank | KV pool | Tokens KV | Concorrência a 1M | Fonte |
|---|---|---|---|---|---|---|
| Vision-Exp TP=2, util 0.835 | README atual | — | **17.04 GiB** | **2,331,430** | **2.22×** | `README.md:117-121, 141` |
| 0731/Vision TP=2 (Aug-9 log) | 2026-08-09 | **79.17 GiB** | 17.04 GiB (util 0.83) | 2,331,430 | 2.22× | `fable5-1-report.md:57` |
| TP=3, 16 slots | 2026-09-02 | **60.69 GiB** | 35.01 / 33.82 GiB (mín. do par) | 5,021,871 | 4.79× | `fable5-1-report.md:56` |
| TP=2 baseline A/B | 2026-09-02 | — | 16.55 / 14.73 GiB | — | — | `fable5-1-report.md:528` |
| TP=2 final A/B seq. 2 | 2026-09-02 | — | 16.76 / 15.47 GiB | — | — | `fable5-1-report.md:548` |
| Preview DSpark, util 0.85, MTP-3 | histórico | — | — | ~2.83 M | ~2.70× | `results/RESULTS-2026-08-14.md` |
| 1M NVFP4 single-stream probes | histórico | — | — | ~2.04 M | ~1.95× | idem |
| Stage-C Keys C12 | 2026-07-02 | — | — | ~3.23 M | ~3.2× | idem |

Cada +0.01 de `GPU_MEMORY_UTILIZATION_TEXT` ≈ +1.2 GiB de KV ≈ +165 K tokens em TP=2 (`fable5-1-report.md:165-166`).
Custo físico ≈ 3.85 KB/token + 3.4 MB/seq (1.11.2); contabilidade do alocador ≈ 7.8 KB/token.

### 4.2 Throughput e latência (2× DGX Spark, TP=2)

**0731, 2026-08-01, completions naturais ~512 tokens** (`docs/DEEPSEEK_V4_FLASH_0731.md:54-73`):

| Prompt | c | TTFT (s) | Prefill tok/s | Decode tok/s | Agregado |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.63 | 447 | 75.4 | 69.1 |
| 256 | 6 | 1.42 | 197 | 36.9 | **191.2** |
| 2,048 | 1 | 0.81 | **2,563** | 68.8 | 62.0 |
| 8,192 | 1 | 4.80 | 1,713 | 73.9 | 43.7 |
| 32,768 | 1 | 22.96 | 1,428 | 64.0 | 16.6 |
| 131,072 | 1 | 78.75 | 1,665 | 65.2 | 5.9 |
| 900K (aceitação) | 1 | **1,028.85** | **≈ 874.8** | — | — |

**Matriz Anemll 0731, 2026-08-14** (decode por stream, `results/RESULTS-2026-08-14.md`): c=1 flat 62–83 tok/s de 256 a
128K (256: 82.6; 131,072: 74.9); c=6 × 256 = 162 agregado; c=4/6 × ≥ 32K ≈ 8 tok/s (piso com prefills serializados);
128K × c=6 TTFT mediano 282 s.

**CUDA graphs regulares vs "breakable"** (0731, 1M): c=1 74.6 → 95.9 (+28.6 %); c=2 agregado 134.2 → 151.8 (+13.1 %);
c=4/c=6 agregado 263.7 / 340.5 (`docs/DEEPSEEK_V4_FLASH_0731.md:81-94`).

**Vision-Exp, 2026-09-02** (mesma lane, sp-indexer on, capture 48, inflight 2): c=1 × 256 = **56 tok/s** (8 trials, 51–67);
c=6 × 256 = **139** agregado; 128K × c=1 TTFT **78.5 s** — 15–25 % abaixo de 14/ago sem knob que explique
(`README.md:337-341`; `AUDIT.md:34-37`). TP=2 com k=6: c=1 52.4 (49.8–56.9), c=2 43.4, c=6 148; TTFT 4.5 / 18 / 79 s
(8K/32K/128K) (`fable5-1-report.md:529`). Capture 42 vs 48: c=6 122 vs 139 (`fable5-1-report.md:547`).

**TP=3 vs TP=2** (2026-09-02, prosa, 512 tokens): 1 stream 38.6 vs 41.9; 2 streams 63.9 vs 72.0; 16 streams 198.0 (TP=3);
TTFT 128K 74.9 vs 91.5 s; 256K 164.8 vs 202.2 s (`docs/TP3.md:121-129`).

**Stage-C / Keys 200K-16** (histórico): c=16 **315.1** tok/s estático (19.7/stream, aceitação 0.609); escalonado 205.0
(`results/RESULTS-2026-08-14.md`).

**Anemll README** (Keys DSpark abliterated, 350K ctx, 512 tokens): c=1 40.7 → 48.5 tok/s (+19.1 %), c=4 91.4 → 103.5;
prefill 1K 2,033 tok/s, 32K 2,176 tok/s.

### 4.3 Modelo de custo do decode (TP=2, c=1, k=6)

`fable5-1-report.md:82-100`: por passo e por rank lê-se ≈ 12 GB de especialistas (≤ 42 distintos por camada × 6.7 MB),
≈ 2.4 GB de atenção FP8, ≈ 0.9 GB do drafter, ≈ 1.06 GB do lm_head (logits alvo + base do rascunho), ≈ 0.2 GB do Markov
`w2` → **≈ 16.5–17 GB/passo** a 230–250 GB/s efetivos ≈ **68–74 ms** → com ~4.15 tokens aceitos/passo ≈ 56–61 tok/s:
**decode c=1 ≈ 80 % streaming de pesos**; ~10–15 ms de custo fixo (≈ 108 coletivas/passo, 3–5 ms). Em TP=3 os bytes/rank
caem para ≈ 11.5 GB (≈ 48 ms) mas o passo medido é ≈ 61 ms (custo fixo ≥ 13 ms) → TP=3 não acelera c=1.

### 4.4 Aceitação do DSpark

| Contexto | k | Aceitação por posição (incondicional) | Média / τ | Fonte |
|---|---|---|---|---|
| 0731 `bench_quick` | 5 | 0.93 / 0.75 / 0.66 / 0.58 / 0.47 | τ ≈ 4.39 (cálculo meu: 1 + Σ) | `docs/PATCHES.md:840-844` |
| Vision-Exp `bench_quick` | 6 | 0.89 / 0.73 / 0.49 / 0.34 / 0.23 / 0.15 | τ ≈ 3.83 | idem |
| Vision-Exp k=5 vs k=6 (2026-09-03) | 5 | 0.88 / 0.74 / 0.53 / 0.36 / 0.24 | igual em k=5 e 6 | `docs/PATCHES.md:862-866` |
| TP=3 probe (2026-09-02) | 6 | 0.92 / 0.74 / 0.60 / 0.41 / 0.28 / 0.21 | 2004/4140 = **48.4 %**, ≈ **3.9 tokens/passo** | `fable5-1-report.md:73-75` |
| Audit "prompts bons" (2026-08-16) | 5 | pos0 ~0.93 → pos4 ~0.33 | **~68–75 %** | `AUDIT.md:29` |
| A/B 2026-09-02, prosa natural | 6 | pos0 0.66–0.71 | 24.7–26 % | `fable5-1-report.md:530, 535, 541` |
| A/B 2026-09-02, "numbered words" | 6 | — | 47–47.3 % | idem |
| Stage-C Keys | — | — | 0.585–0.635 | `results/RESULTS-2026-08-14.md` |
| Amostra `/metrics` (lane "0731-vision") | 6 | 84, 63, 1, 1, 0, 0 aceitos em 105 rascunhos | 149/630 = 23.7 % | `results/dspark-metrics.txt` |

Observação do audit: prompts patológicos (palavras repetidas) derrubam o drafter e parecem regressão de ~40 %
(`AUDIT.md:45-46`). Paper DSpark (produção V4-Flash vs MTP-1): 60–85 % mais rápido por usuário a mesmo throughput.

### 4.5 Qualidade / contexto

- RULER-lite 8/8 a 8K/32K; tool battery 7/7; deep-context 8/8; garble **CLEAN** até ~900K (`AUDIT.md:28-32`).
- ruler-lite 8/8 a 32K + 131K com sp-indexer (`fable5-1-report.md:536`).
- GSM8K N=8 (200 Q) com Patch 2b: 93.5 % vs 95.0 % sequencial (`docs/PATCHES.md:134-139`).
- Card Vision-Exp vs 0731 (métrica oficial DeepSeek): Terminal Bench 2.1 83.9 vs 82.7; DeepSWE 59.3 vs 54.4;
  Toolathlon-Verified 75.9 vs 70.3; ApexBench 36.5 vs 26.2†; Chartography 64.3 (`readme-DeepSeek-V4-Flash-Vision-Exp.md`).
- Report V4: V4-Flash usa 10 % dos FLOPs por token e 7 % do KV do V3.2 a 1M; KV vs GQA8 BF16 ≈ 2 %.

---

## 5. Conceitos necessários (em ordem de pré-requisito)

1. **Vetores, matrizes e produto matricial** — tudo no modelo é `y = x·Wᵀ`. Aqui: `wq_a` é uma matriz 4096×1024; o
   "custo" de cada camada é medido em parâmetros (entradas das matrizes) e bytes lidos da memória por token.
2. **Rede neural e camada linear + não-linearidade** — composição de lineares com ativações (SiLU, GELU). Aparece no
   SwiGLU dos especialistas e do ViT e no GELU do Aligner.
3. **Treino: loss, gradiente, otimizador** — parâmetros ajustados para minimizar uma loss. Aqui: V4 treinou com Muon
   (ortogonaliza atualizações via Newton-Schulz) e o DSpark com CE + TV + BCE de confiança; importa para entender QAT e
   por que certos formatos numéricos "são nativos" do modelo.
4. **Tokenização e vocabulário** — texto → ids. Vocab de 129,280 (mesmo do V3); tokens especiais como
   `<｜deepseek_image｜>` (129264) e o "ruído" do DSpark (128799).
5. **Embedding e LM head** — tabela id → vetor 4096 e projeção 4096 → 129,280 logits; aqui não são amarrados
   (`tie_word_embeddings=false`) e o DSpark reutiliza os do alvo.
6. **Softmax, temperatura, amostragem** — logits → distribuição; `temperature`, `top_p`; a referência usa Gumbel-max
   (`HF:model.py:1024-1031`). Base da verificação especulativa.
7. **Modelo autoregressivo, prefill vs decode** — prefill processa o prompt em paralelo (compute-bound); decode gera 1
   token por passo (memory-bound). Explica por que c=1 decode é streaming de pesos e o prefill longo é dominado pelo indexer.
8. **Transformer e resíduo pré-norm** — bloco = norm → atenção → soma residual → norm → FFN → soma. Na V4 o "resíduo" é
   substituído por mHC com 4 fluxos.
9. **RMSNorm** — normaliza pela raiz da média dos quadrados (sem subtrair média) com peso aprendido; aqui também
   aplicada *sem peso* em cada head de Q e com `eps = 1e-20` no Vision-Exp.
10. **Atenção scaled dot-product e máscara causal** — `softmax(QKᵀ/√d)·V`; cada token só vê o passado. Todas as
    variantes da V4 (SWA, CSA, HCA, drafter não-causal, ViT bidirecional) são escolhas de "quais K/V cada Q vê".
11. **Multi-head attention** — várias heads em paralelo; aqui 64 heads de 512 dims para Q.
12. **KV cache** — guardar K/V do passado para não recomputar; o tamanho por token × contexto × camadas define se 1M cabe.
13. **MQA e GQA** — compartilhar K/V entre heads (1 grupo = MQA). V4 tem `num_key_value_heads = 1`: o KV não pode ser
    fatiado por TP e é replicado nos dois nós.
14. **MLA (V2/V3)** — cachear um latente de baixa dimensão (512) + chave RoPE desacoplada (64) em vez de K/V por head.
    Base conceitual do KV de 512 dims da V4.
15. **Absorção de pesos no MLA** — pré-multiplicar `W_UK` na query e `W_UV` no `W_O` para atender direto no latente.
    A V4 é o caso-limite: o latente *é* K e V (`kv_lora_rank = head_dim` no vLLM).
16. **RoPE** — rotação de pares de dims por ângulo proporcional à posição; produto escalar passa a depender só da
    distância. Aqui aplicado só às últimas 64 dims (RoPE parcial), estilo interleaved.
17. **Extensão de contexto com YaRN** — interpolar as frequências baixas (÷16) e manter as altas; 64K → 1M. Só nas
    camadas comprimidas (referência); `mscale` desligado.
18. **θ de RoPE diferente por camada** — `compress_rope_theta = 160000` nas camadas comprimidas vs 10000 nas SWA;
    comprimentos de onda maiores para posições de blocos.
19. **RoPE inverso na saída** — como V = K carrega rotação absoluta, rotaciona-se a saída por −i para tornar a
    contribuição relativa.
20. **Attention sink** — logit aprendido por head somado ao denominador; permite "não atender"; padding de heads precisa
    de `-inf`.
21. **Sliding Window Attention (SWA)** — só os últimos W tokens (W=128); cache em anel; custo constante por sequência.
22. **Compressão de KV por pooling aprendido** — softmax por canal sobre blocos de m tokens com gate e viés posicional
    (`ape`); sobreposição (2m) no C4 para suavizar fronteiras. Base de CSA/HCA.
23. **Atenção esparsa por top-k (DSA)** — um "indexer" barato pontua todas as posições e a atenção cara só olha as top-k.
    V3.2 escolhia 2048 tokens; V4 escolhe 512 entradas comprimidas 4:1.
24. **Lightning indexer** — 64 heads × 128 dims, score `Σ_h w_h ReLU(q_h·k)`, em FP4 com Hadamard; replicado por rank no
    vLLM; O(n) por token no decode e O(n²) no prefill.
25. **CSA vs HCA e o intercalamento** — C4A (esparsa sobre KV/4) e C128A (densa sobre KV/128) alternadas + SWA em todas;
    trocam precisão local/global por memória e FLOPs.
26. **Projeção de saída agrupada / low-rank** — 8 grupos, cada um 4096 → 1024, depois 8192 → 4096; metade dos params;
    impõe o contrato `r = 4096` que dita o padding de TP=3.
27. **FFN SwiGLU e clamp** — `w2(silu(w1 x) ⊙ w3 x)`; `swiglu_limit = 10` limita ativações.
28. **Mixture-of-Experts** — muitos FFNs, cada token usa poucos (top-6 de 256) + 1 compartilhado; 284B totais, 13B ativos.
29. **Função de score do roteador** — softmax (V2) → sigmoid (V3) → √softplus (V4); com normalização dos top-k e
    `routed_scaling_factor = 1.5`.
30. **Balanceamento de carga sem loss auxiliar (aux-loss-free)** — viés por especialista que só afeta a *seleção*;
    `topk_method = noaux_tc` → `e_score_correction_bias`.
31. **Group-limited routing (V2/V3) e sua ausência na V4** — restrição de grupos para limitar tráfego EP; V4 não usa.
32. **Hash routing** — especialistas fixos por id de token nas 3 primeiras camadas; exige `input_ids` e trata tokens de
    imagem à parte (#175).
33. **Roteamento condicionado à modalidade (`bias_vl`)** — viés separado para tokens de imagem no Vision-Exp.
34. **Hyper-connections e mHC** — resíduo com n fluxos e matrizes de mistura dinâmicas (pre/post/comb por token);
    comb projetado em matrizes duplamente estocásticas (Birkhoff) via Sinkhorn-Knopp (20 iters) para estabilidade.
35. **Sinkhorn-Knopp** — alternar normalização de linhas e colunas até convergir para duplamente estocástica.
36. **Ponto flutuante: BF16, FP8 E4M3, FP4 E2M1** — bits de expoente/mantissa; E4M3 máx 448, E2M1 máx 6; define
    erro e bytes por parâmetro.
37. **Quantização por blocos e escalas** — uma escala por bloco (128×128 nos pesos FP8, 64 no KV, 32 no FP4); escala
    UE8M0 = potência de 2 guardada como expoente.
38. **Quantização dinâmica de ativações** — escala calculada em runtime por bloco de 128 antes da GEMM FP8/FP4.
39. **MXFP4 vs NVFP4** — MXFP4: e2m1 + E8M0 por 32 (padrão OCP); NVFP4: e2m1 + E4M3 por 16 + escala global. O
    checkpoint é MXFP4-like (`fp4_e8m0_k32`), por isso o b12x cai em W4A16 no GB10.
40. **W4A16 vs W4A8/FP4 tensor cores** — pesos 4 bits dequantizados para BF16 na hora vs MMA nativo FP4×FP8; prefill
    MoE fica compute-ineficiente (`fable:269-300`).
41. **QAT (quantization-aware training) e quantização simulada** — treinar com quant→dequant no forward; o modelo
    "espera" KV FP8/64 e indexer FP4/32.
42. **Rotação de Hadamard** — transformação ortogonal que espalha outliers antes de quantizar em 4 bits (indexer).
43. **Layout de KV cache em bytes (584 B)** — 448 FP8 + 128 BF16 + 8 escalas; `nvfp4_ds_mla` ≡ `fp8_ds_mla` na V4.
44. **Paged attention, blocos e page size** — KV em páginas de tamanho fixo endereçadas por block table; `--block-size 256`
    para 64 linhas comprimidas por página.
45. **Alocador de KV híbrido e grupos de KV** — vários tipos de camada (MLA global + SWA) exigem mesmo tamanho de
    página; 4 grupos na V4+DSpark; contabilidade (7.8 KB/token) ≠ custo físico (3.85 KB).
46. **Prefix caching e interação com SWA** — reaproveitar KV de prefixos; SWA só guarda a cauda → retenção esparsa a cada
    4096 tokens (#26) e recomputo de 128 tokens para o drafter.
47. **Chunked prefill e long-prefill threshold** — prefill quebrado em chunks (1024) para intercalar com decode.
48. **Continuous batching e condensação do batch** — requests entram/saem a cada passo; linhas do batch mudam → estado
    por request precisa de slot estável (Patch 1).
49. **Speculative decoding** — um rascunho barato propõe k tokens, o alvo verifica todos num forward; τ = tokens aceitos
    por passo.
50. **Rejection sampling na verificação** — aceita com `min(1, p/q)`, reamostra do resíduo; distribuição exata do alvo;
    aceitação esperada = 1 − TV(p, q).
51. **MTP (multi-token prediction)** — cabeças extras treinadas para prever t+2…; drafter natural; V4-preview tinha MTP-1.
52. **Drafters "paralelos" (EAGLE/DFlash) e injeção de features do alvo** — o rascunho lê estados ocultos de camadas do alvo
    e prevê o bloco inteiro num forward não-causal com tokens de máscara.
53. **DSpark** — backbone paralelo (3 estágios V4 SWA+MoE sobre features 40–42) + Markov head low-rank sequencial +
    confidence head; bloco treinado γ = 5.
54. **Bias de transição low-rank (Markov head)** — `W2·W1[prev]` condiciona cada posição no token amostrado antes; resolve
    "multi-modal collision".
55. **Confiança, sobrevivência e verificação agendada** — `c_k` estimado; sobrevivência = produto; escolher quantas posições
    verificar por request segundo limiar ou curva de hardware.
56. **Custo de verificação e CUDA graphs** — cada passo verifica seqs × (k+1) tokens; tamanhos de captura precisam cobrir
    esse shape (48 a 6×6).
57. **CUDA graphs, torch.compile, custom ops** — capturar a sequência de kernels para eliminar overhead de CPU; operações
    que fazem `.item()` não podem ser capturadas (motivo de desvios em #175 e #52492).
58. **Tensor parallelism (Megatron)** — column-parallel (fatia saída), row-parallel (fatia entrada + all-reduce),
    vocab-parallel (embedding/lm_head). Aqui: o que é replicado (Q-latente, KV, indexer) vs fatiado.
59. **TP em MoE vs Expert Parallel** — fatiar cada especialista entre ranks vs distribuir especialistas inteiros; MegaMoE
    exige EP e SM100.
60. **Coletivas (all-reduce, all-gather) e latência de rede** — ≈ 108 por passo em RoCE sem GPUDirect; custo fixo que
    domina quando bytes/rank caem (TP=3).
61. **Padding de heads/grupos para TP não divisível** — adicionar grupos nulos (zeros + sink `-inf`) preservando contratos
    de forma.
62. **Sequence/context parallelism** — dividir a dimensão de sequência entre ranks (SP indexer, DCP); merge exato de top-k.
63. **Memória unificada (GB10) e decode limitado por banda** — CPU e GPU dividem ~121 GiB de LPDDR5X; tok/s ≈ banda / bytes
    por token; memória do host compete com o KV pool.
64. **Arquitetura de GPU / SM e JIT de kernels** — GB10 = SM 12.1 (`sm_121a`); kernels de datacenter (SM100) não servem;
    FlashInfer SM120, DeepGEMM, b12x, CuTeDSL, TileLang, Triton compilam por SM e precisam de cache persistido.
65. **Vision Transformer (ViT)** — imagem em patches 14×14 → tokens → atenção bidirecional; RoPE 2D para posição linha/coluna.
66. **Projetor / Aligner e downsampling espacial** — agrupar 3×3 patches e projetar para o espaço do LLM (4096).
67. **Layout de tokens de imagem e placeholders** — start/newline/pad/end, ordem "N" em pares de linhas, alinhamento a 4
    (compressão C4), expansão de 1 placeholder em N tokens.
68. **Encoder cache multimodal** — cache de embeddings por hash do conteúdo; precisa incluir o layout (#172).
69. **Modos de raciocínio e encoding sem template Jinja** — `encoding_dsv4.py`, `<think>`, efforts low/high/max; o prompt
    `max` injeta instruções de sistema; interage com `max_tokens`.
70. **Abliteration (projeção de direção)** — remover a componente de uma direção de refusal do fluxo residual; aqui na
    saída da atenção das camadas 10–42.

---

## 6. Receita passo-a-passo (do ponto de vista da arquitetura)

1. **Leia o `config.json` e classifique o modelo.** `model_type deepseek_v4`, 43 camadas, `num_key_value_heads = 1`,
   `head_dim 512`, `compress_ratios`, `index_*`, `hc_mult`, `expert_dtype fp4`, `dspark_*`, `vision_*`. Daí saem: (a)
   o backend de atenção precisa suportar *sparse MLA com dois caches* (SWA + comprimido) — no SM121 é o FlashInfer
   SM120 (`sm120.py`); (b) o MoE precisa de kernel MXFP4 (b12x W4A16 no GB10); (c) o speculative é `method: dspark`
   usando os pesos `mtp.*` do próprio checkpoint.
2. **Faça a conta de pesos → decide o nº de nós.** ≈ 155–157 GiB (1.14) > ~121 GiB de um GB10 → 2 nós, TP=2 (divide 64
   heads e 8 grupos). TP=3 só com padding (3.C1). Sem EP porque MegaMoE exige SM100 e EP.
3. **Faça a conta de KV → decide contexto e slots.** Pool ≈ util × memória − pesos − ativações/grafos ≈ 17 GiB em
   TP=2. Custo ≈ 3.85 KB/token + 3.4 MB/seq; alocador reporta 2.33 M tokens → `MAX_MODEL_LEN = 1048576` é um *teto*
   por request, e `MAX_NUM_SEQS = 6` um teto de slots (`README.md:150-152, 484-500`). Lembre que o KV é **replicado** nos
   ranks (1 head KV), então TP não multiplica capacidade exceto via pesos menores por rank.
4. **Escolha o formato do KV cache.** `--kv-cache-dtype nvfp4_ds_mla` (= 584 B FP8/BF16, casado com a QAT) e
   `--block-size 256` (página de 64 linhas comprimidas). Aplique o fix #22 para garantir o kernel FP8 em contexto longo.
5. **Garanta RoPE/YaRN corretos.** YaRN ×16 com `original_max_position_embeddings 65536`; θ = 160000 nas camadas
   comprimidas; θ = 10000 sem YaRN nas SWA (fix opt-in). `mscale` desligado.
6. **Trate o roteamento especial.** Hash nas camadas 0–2 (precisa de `input_ids` crus); `bias_vl` para imagens; o vLLM
   deve carregar `ffn.gate.bias` → `e_score_correction_bias` e ignorar biases não registrados nas camadas hash.
7. **Configure o DSpark de acordo com o checkpoint.** `dspark_block_size = 5`; regra de divisibilidade da Anemll com
   `num_nextn_predict_layers` (1 no 0731 → k=5; 3 no Vision-Exp → k=6) ou o unlock block-k com k=5; `draft_sample_method`
   probabilistic (empatou com greedy no A/B); capture size `seqs × (k+1)` arredondado a 8. Para `max_num_seqs > 1` o
   drafter precisa de estado por request (Patch 1/2/2b — presentes na Anemll segundo o README/créditos).
8. **Visão (se Vision-Exp).** Instale ViT + Aligner + processador (limite 384 tokens/imagem; imagens só em `user`),
   `bias_vl`, hash salgado com `num_tokens`. Saiba que a máscara bidirecional de imagem da referência não está replicada.
9. **Paralelismo e comunicação.** Espere ≈ 108 coletivas por passo; mantenha grafos regulares
   (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, +28.6 % em c=1); considere replicar o Markov head (medido: sem ganho aqui).
10. **Prefill longo.** O indexer replicado é o gargalo ≥ 128K: opções SP-indexer (−6 % TTFT 128K medido), cache MXFP4
    do indexer, workspace de logits (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`), chunk 1024 para não matar o decode.
11. **Verifique.** Boot: `Available KV cache memory`, `GPU KV cache size`, `Maximum concurrency`, `Capturing CUDA graphs
    (FULL): 6/6`, ausência de "Truncating"; `/metrics` `spec_decode_num_accepted_tokens_per_pos_total`; RULER-lite,
    garble sweep até 900K, tool battery (`AUDIT.md`).
12. **Tune com a física em mente.** c=1 decode ≈ banda / (≈ 16.5 GB por passo) × τ; aumentar aceitação (drafter) ou
    concorrência são as alavancas; TP=3 compra capacidade, não latência.

---

## 7. Pontos em aberto / incertezas registradas

- **Máscara de imagem**: a referência dá visibilidade bidirecional dentro do span da imagem e janela alargada; não achei
  equivalente no hotfix do vLLM (1.13 item 8). Impacto na qualidade visual não medido no repo.
- **`rms_norm_eps = 1e-20`** no Vision-Exp (vs 1e-6 no 0731): mudança real do checkpoint; efeito não discutido no repo.
- **Mapeamento das camadas C128A nos 4 grupos de KV** (1 MLA + 3 SWA 64/4/8): não documentado; o que está documentado é
  a página comum de 37,376 B.
- **Composição dos 132 B do cache K do indexer** (128 FP8 + 4 B de escala) é inferência minha por analogia com o V3.2.
- **Paper DSpark** (via resumo) menciona "5 transformer layers with MoE" em produção; o checkpoint tem 3 estágios
  (`n_mtp_layers: 3`). Pode ser a versão V4-Pro ou erro do resumo — não verificado no PDF.
- **`patches/dsv4_tp_pad.py` vs `patches/tp3/apply_tp3_patch.py`** descrevem estratégias de padding opostas; o efetivo
  (injetado) é grupos 8→9 com 8 heads/grupo (`docs/TP3.md`).
- **Código do modelo na Anemll 0.1.1** não está no repo; o overlay Stage-C é a melhor aproximação legível. Diferenças
  conhecidas: Markov head em `qwen3_dspark.py`, V2 speculator, `n_predict = num_nextn_predict_layers`.

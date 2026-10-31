# Inventário de Arquitetura — GLM-5.3-Flash (alvo) + DFlash2 (drafter) na receita `GLM-5.3-Flash-EXL3-2x-DGX-Sparks`

> Escopo deste arquivo: a **arquitetura do modelo** GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`), do seu
> speculator **DFlash2** (`incoai/GLM-5.3-Flash-DFlash2`) e do checkpoint quantizado **EXL3/TR3 4 bpw**
> usado pela receita, mais todas as parametrizações/modificações da receita que existem *por causa* da
> arquitetura (NoPE-MLA, indexador kpool, KDA, mHC, MoE, MTP, DFlash2). Outras partes da receita
> (rede, Docker, scheduler, APC em detalhe, kernels EXL3 em detalhe) aparecem só quando tocam a arquitetura.
>
> Convenções: `[medido-repo]` = número do repositório; `[config]` = valor lido do `config.json`/cabeçalho
> safetensors; `[código]` = lido no código-fonte; `[cálculo próprio]` = conta feita por mim a partir de
> valores exatos; `[inferência]` = dedução minha, não afirmada pela fonte; `[secundário]` = fonte web
> não-primária (pode conter erros).

---

## 0. Fontes, versões e confiabilidade

| Fonte | Versão / data | O que foi usado |
|---|---|---|
| Repositório local `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` | commit `94ae731f` (2026-09-28, merge PR #292 `feat/dense-h3-preset`) | `Dockerfile`, `start.sh`, `start-tp3.sh`, `.env.example`, `overlay/*.py`, `docs/*.md`, `ablit/LAYER_MAP.json`, `tests/fixtures/*.py.txt`, `tools/pack_profile.py`, README (155 KB), CHANGELOG |
| `zai-org/GLM-5.3-Flash` (HF) | sha `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, criado 2026-08-25, última modificação 2026-09-07 | `config.json` (baixado cru), `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json`, `model.safetensors.index.json` (76.108 tensores, `total_size` 328.326.771.576 B) e **cabeçalhos safetensors** dos shards 1, 2, 31, 32, 47, 62 lidos por HTTP range (formas exatas de cada tensor) |
| vLLM upstream no commit fixado pela imagem | `487ecf187d3dfe74d2cf6119a92881dba403c219` (a imagem base `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` reporta `0.1.dev20051+g487ecf187`) | `vllm/models/glm5next/nvidia/{model,attention,kda,mtp,multimodal}.py`, `ops/kpool_compress.py`, `vllm/transformers_utils/configs/glm5_next.py`, `model_executor/layers/{mhc,sparse_attn_indexer_kpool,activation}.py`, `kernels/mhc/torch.py`, `layers/mamba/mamba_utils.py`, `v1/spec_decode/dflash.py`, `models/qwen3_dflash.py` |
| `incoai/GLM-5.3-Flash-DFlash2` (HF) | `main`=`bf582e4e…` (2026-08-31); a receita fixa `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` (2026-08-28). O `config.json` nas duas revisões é **idêntico** (verificado por diff) | `config.json`, model card, cabeçalho de `model.safetensors` (81 tensores, 1.171.080.448 parâmetros BF16) |
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` e espelho `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | espelho byte-idêntico do snapshot `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`; revisão do espelho fixada em `start.sh:122` = `25a44fdbf16862a46b7cc9921142c6c81350af2f` | `config.json`→`quantization_config`, `quantization/recipe.json`, `exl3-mcg-storage-abi.json`, `materialization-receipt.json`, `runtime-results-v44.json`, índice (150.226 tensores, 175.622.979.576 B) e cabeçalho do shard 57 |
| Papers/blogs | DFlash (arXiv 2602.06036, ICML 2026); blog DFlash 2 (inco.ai, ago/2026); Kimi Linear (arXiv 2510.26692); mHC (arXiv 2512.24880, DeepSeek); GLM-5 tech report (arXiv 2602.15763) | conceitos e números dos métodos |
| Secundárias | docs Transformers `glm5_next`, notas de S. Raschka, issue SemiAnalysis/InferenceX #2794, `recipes.vllm.ai` | confirmação cruzada. **Atenção**: a doc Transformers resumida chamou KDA de "Kernel-based Decay Attention" e DSA de "Distributed Sparse Attention" e lista `index_kpool` default 16 — são erros/defaults de classe; o checkpoint usa `index_kpool: 4`. KDA = **Kimi Delta Attention**; DSA = **DeepSeek Sparse Attention** |

**Pontos em que as fontes divergem (e o que prevalece aqui):**

1. Tabela "KV cache" do README (`README.md` ~linha 679) diz "Mamba (33 layers, 3 groups)"; o `config.json` diz **34** camadas KDA (`linear_attn_config.kda_layers` tem 34 índices) e o log de capacidade do próprio README (`README.md` ~1880) mostra **4** grupos `MambaSpec` (grupos 2–5). Prevalecem o config (34) e o log (4 grupos). A linha "33/3" parece desatualizada.
2. Comentário no vLLM (`attention.py`, `Indexer.__init__`: `self.n_head = config.index_n_heads  # 64` e "this checkpoint uses index_n_heads=16") conflita com o checkpoint, que tem `index_n_heads: 32` → prevalece 32 (sem padding de cabeças no indexador, pois DeepGEMM aceita 32/64).
3. `docs/improve-prefill.md` estima "Active params ≈ 19.5 B/token" e "KDA layers (34 × ~170M)". A contagem exata a partir das formas dos tensores dá **137.732.288 parâmetros por camada KDA** e ~**17,4 B** ativos (texto, incluindo embedding+lm_head) / ~**17,9 B** com a torre de visão — consistente com o "18B active" do model card. Ver §4.1.
4. O model card fala em "300,000 tokens (evaluation context)" para alguns benchmarks, mas `max_position_embeddings`=`model_max_length`=**1.048.576**. A receita serve 850k por padrão.

---

## 1. Visão geral

### 1.1 O que é o GLM-5.3-Flash (o modelo "alvo")

GLM-5.3-Flash é um LLM **multimodal nativo** (texto + imagem + vídeo) da Z.ai, do tipo **decoder-only
Transformer** com **Mixture-of-Experts (MoE)**, **320B parâmetros totais / 18B ativos** (model card). A
contagem exata do checkpoint (`safetensors.total` do HF) é **321.323.031.390** parâmetros, e ela bate
**exatamente** com a soma das formas que eu reconstruí (texto 320.759.404.382 + visão 563.627.008) `[cálculo próprio]`.

As quatro ideias arquiteturais que o distinguem (e que **forçam** quase todas as modificações da receita):

1. **Atenção híbrida 3:1** — das 45 camadas do decoder, **34 são KDA** (Kimi Delta Attention: atenção
   *linear* com estado recorrente de tamanho fixo) e **11 são MLA esparsa** (Multi-head Latent Attention
   com DeepSeek Sparse Attention: cada query só atende a **2048** tokens escolhidos por um "indexador").
   Padrão: `KDA, KDA, KDA, DSA` repetido 11 vezes (camadas DSA = 3, 7, 11, …, 43), mais a camada 44 (KDA).
   É exatamente a receita do **Kimi Linear** (3 KDA : 1 MLA, MLA sem posição).
2. **NoPE** — as camadas MLA **não têm RoPE** (`qk_rope_head_dim: 0`, `mla_use_nope: true`), e como o
   indexador herda `rope_dim = qk_rope_head_dim = 0` (`attention.py`, `Indexer.__init__`), **nenhuma
   atenção do texto usa embedding posicional explícito**. A informação de ordem vem das camadas KDA
   (recorrência causal + convolução curta causal) e de um pequeno *APE* (embedding posicional absoluto de
   4 posições) dentro do "pool" de chaves do indexador. `config.json` nem tem `rope_theta`.
3. **mHC (Manifold-Constrained Hyper-Connections)** — o "fio residual" é alargado para **4 fluxos**
   (`hc_mult: 4`): o estado residual por token é `[4, 4096]` em vez de `[4096]`. Antes/depois de cada
   sub-bloco (atenção e FFN) matrizes aprendidas por token misturam os 4 fluxos; a matriz de mistura
   residual 4×4 é projetada (Sinkhorn, 20 iterações) no politopo de Birkhoff (duplamente estocástica).
4. **MoE fino** — 288 experts roteados (SwiGLU, intermediário 2048) + 1 expert compartilhado, **top-8**,
   roteamento **sigmoid** com viés de balanceamento sem perda auxiliar (`noaux_tc`), `routed_scaling_factor 2.5`.
   As 3 primeiras camadas são MLP denso (intermediário 12288).

Mais: **1 camada MTP** (Multi-Token Prediction, "layer 45"), vocabulário **154.880**, embeddings **não
amarrados** ao `lm_head`, contexto **1.048.576**, torre de visão ViT de 24 blocos.

### 1.2 Fluxo de um token pelo modelo (ponta a ponta)

```
token id ──► embed_tokens [154880×4096] (BF16) ──► x ∈ R^4096
          ──► hc_expand: replica em 4 fluxos → X ∈ R^{4×4096}         (mhc.py: hc_expand)
para cada camada ℓ = 0..44:
   ├─ mHC-pre(attn): a partir de X (flatten 16384) calcula
   │     pre_mix (4 pesos, sigmoid),  post_mix (4, 2·sigmoid),  comb (4×4, Sinkhorn)
   │     entrada da sub-camada = Σ_i pre_mix_i·X_i  → RMSNorm (input_layernorm)
   ├─ Atenção:  KDA (34 camadas)  ou  MLA-NoPE + indexador DSA/kpool (11 camadas)
   ├─ mHC-post(attn) fundido com mHC-pre(ffn):  X ← comb^T·X + post_mix ⊗ y_attn
   │     → Σ pre_mix·X → RMSNorm (post_attention_layernorm)
   ├─ FFN:  MLP denso SwiGLU 12288 (ℓ=0,1,2)  ou  MoE (ℓ=3..44): shared + Σ_top8 g_e·E_e
   └─ mHC-post(ffn) é adiado e fundido com o mHC-pre da próxima camada
na última camada (44): hc_post materializado → hc_contract (média dos 4 fluxos) → R^4096
──► RMSNorm final (model.norm) ──► lm_head [154880×4096] (BF16, não amarrado) ──► logits
```

(Fontes: `vllm/models/glm5next/nvidia/model.py` `Glm5NextDecoderLayer.forward` e `Glm5NextModel.forward`;
`vllm/model_executor/layers/mhc.py`.)

### 1.3 Mapa camada-a-camada (46 camadas: 45 do decoder + 1 MTP)

Fontes: `config.json` (`layer_types`, `mlp_layer_types`), `ablit/LAYER_MAP.json`, `incoai/…/config.json`
(`dflash_config.target_layer_ids`), `overlay/patch_glm_eagle3.py`.

| # | Atenção (`layer_types`) | Mixer | MLP (`mlp_layer_types`) | mHC | Tap DFlash2 (`target_layer_ids`) | Papel ABLIT (`ablit/LAYER_MAP.json`) |
|---:|---|---|---|---|---|---|
| 0 | `linear_attention` | KDA (linear) | dense SwiGLU 12288 | sim |  | safety-anchor |
| 1 | `linear_attention` | KDA (linear) | dense SwiGLU 12288 | sim |  | safety-anchor |
| 2 | `linear_attention` | KDA (linear) | dense SwiGLU 12288 | sim |  | safety-anchor |
| 3 | `deepseek_sparse_attention` | MLA NoPE + DSA (indexador kpool) | MoE 288×2048 top-8 + 1 shared | sim |  | safety-anchor |
| 4 | `linear_attention` | KDA (linear) | MoE | sim |  | safety-anchor |
| 5 | `linear_attention` | KDA (linear) | MoE | sim | **saída capturada** (hook `idx+1`=6) | safety-anchor |
| 6 | `linear_attention` | KDA (linear) | MoE | sim |  | safety-anchor |
| 7 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | safety-anchor |
| 8–10 | `linear_attention` | KDA | MoE | sim |  | safety-anchor |
| 11 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | safety-anchor |
| 12–13 | `linear_attention` | KDA | MoE | sim |  | safety-anchor |
| 14 | `linear_attention` | KDA | MoE | sim | **saída capturada** (hook 15) | safety-anchor |
| 15 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 16–18 | `linear_attention` | KDA | MoE | sim |  | edit |
| 19 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 20–22 | `linear_attention` | KDA | MoE | sim |  | edit |
| 23 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 24 | `linear_attention` | KDA | MoE | sim | **saída capturada** (hook 25) | edit |
| 25–26 | `linear_attention` | KDA | MoE | sim |  | edit |
| 27 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 28–30 | `linear_attention` | KDA | MoE | sim |  | edit |
| 31 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 32 | `linear_attention` | KDA | MoE | sim |  | edit |
| 33 | `linear_attention` | KDA | MoE | sim | **saída capturada** (hook 34) | edit |
| 34 | `linear_attention` | KDA | MoE | sim |  | edit |
| 35 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 36–38 | `linear_attention` | KDA | MoE | sim |  | edit |
| 39 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 40–41 | `linear_attention` | KDA | MoE | sim |  | edit |
| 42 | `linear_attention` | KDA | MoE | sim | **saída capturada** (hook 43) | edit |
| 43 | `deepseek_sparse_attention` | MLA NoPE + DSA | MoE | sim |  | edit |
| 44 | `linear_attention` | KDA | MoE | sim |  | edit |
| 45 | camada **MTP** (`num_nextn_predict_layers: 1`) | MLA NoPE + DSA | MoE | **não** (MTP usa residual simples) |  | mtp-edit |

Contagens: 34 KDA, 11 DSA no decoder (+1 DSA na MTP = 12 MLA "no checkpoint"); 3 MLP densas; 42 MoE no
decoder (+1 na MTP = 43 MoE, por isso o índice tem 12.384 = 43 × 288 experts × 3 matrizes).
Observação: as 5 camadas "tapeadas" pelo DFlash2 (5, 14, 24, 33, 42) são **todas KDA**.

### 1.4 O que é o DFlash2 (o "drafter")

DFlash2 é um **modelo rascunho (draft) de 1,17 B parâmetros** treinado especificamente para o GLM-5.3-Flash,
usado em **decodificação especulativa sem perda**. Ele não é um LM independente: não tem `embed_tokens`
nem `lm_head` próprios (usa os do alvo) e **lê estados ocultos do alvo** (saídas das camadas 5, 14, 24, 33,
42). Em vez de gerar rascunhos token a token (como EAGLE/MTP), ele é um **drafter de "difusão em bloco"
(block diffusion)**: recebe um bloco de 8 posições = o último token verificado ("âncora"/bonus) + 7
tokens `[MASK]` (id **154856**) e **prevê as 7 posições em um único forward**, com atenção
**bidirecional dentro do bloco** (`is_causal: false`). O DFlash2 acrescenta ao DFlash original:
(a) **top-16 candidatos por posição** + um **seletor** bilinear de baixo posto (rank 256) que escolhe um
caminho coerente através dos candidatos; (b) **convoluções dinâmicas de 2 taps** antes e depois da atenção
e do FFN de cada camada, para combater o "decaimento de sufixo" (posições finais do bloco piores).
O alvo então **verifica as 8 posições num único forward** e aceita o maior prefixo compatível
(amostragem por rejeição → distribuição exata do alvo). Na receita: **k = 7** (`DFLASH_TOKENS=7`).

### 1.5 Como a receita serve tudo isso em 2× DGX Spark (GB10, sm_121)

* **Pesos**: checkpoint **EXL3/TR3 "uniform-K4"**: *só os experts roteados* (311,65 B de parâmetros, 97 %
  do modelo) são quantizados a **4 bits/peso** em formato trellis (QTIP-like) com codebook "MCG" e
  rotações de Hadamard (`suh`/`svh`); **todo o resto fica nativo BF16** (`non_routed_dtype_policy:
  official_source_native`). Total **175,62 GB = 163,6 GiB** em 120 shards (vs 328,3 GB do FP8 oficial e
  ~642,6 GB em BF16). A receita, por padrão, ainda re-quantiza no *load* as projeções densas não roteadas
  para **FP8 weight-only (Marlin)** (`GLM53_DENSE_FP8=all`).
* **Paralelismo**: **TP=2** (uma GPU por Spark, CX7/RoCE entre eles). Experts: `gate/up` fatiados por coluna,
  `down` por linha → um all-reduce por camada MoE. Drafter também TP=2 (`DFLASH_DRAFT_TP=2`).
* **Atenção**: backend `FLASHINFER_MLA_SPARSE_SM120`, que **só** aceita o registro KV empacotado
  `fp8_ds_mla` de 656 B/token/camada (512 FP8 + 16 B escalas + 128 B de RoPE). Como o modelo é NoPE, o
  `Dockerfile` **zera-preenche** 64 dimensões de RoPE ("geometria GLM_NSA 576") — resultado exato.
* **KV/estado**: 11 camadas MLA (fp8_ds_mla) + 11 caches do indexador (1 chave FP8 comprimida a cada 4
  tokens) + 34 estados KDA de tamanho **constante** + janela deslizante de 2048 tokens do drafter.
  Isso dá ≈ **7.579 B/token/rank** de custo que cresce com o contexto `[cálculo próprio]`, contra
  ~2,95 MB/token se fosse MHA BF16 clássica em 45 camadas.
* **Contexto**: 850.000 tokens por padrão (1M cabe com `EXL3_FAT_GROUPED=0`).

---

## 2. Inventário de parâmetros

Notação de "onde": `HFcfg:N` = linha N do `config.json` cru de `zai-org/GLM-5.3-Flash` (sha `eb9eb208`);
`vcfg` = `vllm/transformers_utils/configs/glm5_next.py` no commit vLLM `487ecf187`; `vmodel` =
`vllm/models/glm5next/nvidia/model.py`; `vattn` = `…/attention.py`; `vkda` = `…/kda.py` (idêntico ao
fixture `tests/fixtures/kda-487ecf187.py.txt` do repo); `DFcfg:N` = linha do `config.json` do DFlash2.

### 2.1 `text_config` — parâmetros de forma (dimensões globais)

| nome | valor (default da classe vLLM, se diferente) | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `hidden_size` | **4096** | HFcfg:20 | Largura do fluxo residual (d_model) de cada um dos 4 fluxos mHC; entrada/saída de toda sub-camada | embedding, residual stream | Dita o tamanho de todas as matrizes: q/k/v KDA `[8192,4096]`, experts `[2048,4096]`. `tools/pack_profile.py:79` recusa um pack com `hidden_size != 4096` ou `num_hidden_layers != 45`. A extensão cooperativa MoE é especializada em H=4096 (`docs/cooperative-moe.md:24`) |
| `num_hidden_layers` | **45** | HFcfg:240 | Nº de camadas do decoder (a camada MTP é a 45, fora dessa conta) | profundidade | `get_spec_layer_idx_from_weight_name` (vmodel:1079) usa `num_hidden_layers` como índice da 1ª camada MTP; o DFlash2 numera suas camadas a partir de 45 (`start_layer_id`), por isso o grupo FP8 denso ignora `layers.N` com N ≥ 45 (`overlay/exl3.py` ~1950) |
| `vocab_size` | **154880** | HFcfg:259 | Linhas de `embed_tokens` e `lm_head` | vocabulário, embedding | O tokenizer tem 154.820 tokens BPE + 36 especiais = 154.856 usados; 154.880 é múltiplo de 128 (padding). O id de máscara do DFlash2 (154856) cai nessa faixa de padding. Em TP=3 vira 154.944 (`padding_size=lcm(64,tp)=192`, `overlay/tp3/README.md`) |
| `tie_word_embeddings` | **false** | HFcfg:254 e :287 | `lm_head` tem pesos próprios (não reusa `embed_tokens`) | tied vs untied embeddings | +634,4 M parâmetros (1,27 GB BF16). O `lm_head` é compartilhado com MTP (`shared_head.head`) e com o DFlash2 (que não tem `lm_head`) |
| `max_position_embeddings` | **1048576** | HFcfg:181 | Comprimento máximo de sequência declarado | janela de contexto | Como não há RoPE nas camadas de texto, esse número não dimensiona tabelas cos/sin do alvo; limita `max_model_len`. Receita usa 850.000 (`start.sh:222`) |
| `head_dim` | **0** | HFcfg:18 | Campo genérico não usado pelos mixers (MLA usa `qk_*`/`v_head_dim`; KDA usa `linear_attn_config.head_dim`) | — | `vcfg` só recalcula se `None`; com 0 fica 0. Inócuo `[inferência]` |
| `num_attention_heads` | **64** | HFcfg:238 | Cabeças de query da MLA (e base do padding TP3) | multi-head attention | 64/TP2 = 32 cabeças por rank. **Não divide por 3** → TP3 força 66 (`start-tp3.sh:418`, `TP3_HEAD_OVERRIDE=66`) |
| `num_key_value_heads` | **64** | HFcfg:241 | Nº "declarado" de cabeças KV | MHA/GQA | Na MLA o que se guarda é um latente único (`kv_lora_rank`), então esse número não dimensiona o KV cache; importa para validações e para o padding TP3 |
| `hidden_act` | `silu` | HFcfg:19 | Ativação do SwiGLU | SiLU/SwiGLU | `vmodel` recusa qualquer outra |
| `rms_norm_eps` | **1e-5** | HFcfg:249 | ε de todas as RMSNorm do texto | RMSNorm | (A visão é forçada a 1e-6 pelo vLLM — ver §2.4) |
| `dtype` | `bfloat16` | HFcfg:8 | dtype de treino/nativo | BF16 | Base para `kda_state_dtype` (conv state em BF16; estado recorrente FP32) |
| `attention_bias` / `attention_dropout` | false / 0.0 | HFcfg:6–7 | Sem bias nas projeções de atenção; sem dropout | — | — |
| `initializer_range` | 0.02 | HFcfg:76 | Desvio da inicialização (treino) | init | Sem efeito em inferência |
| `use_cache` | true | HFcfg:257 | Usa cache de KV/estado em HF | KV cache | — |
| `model_type` | `glm5_next_text` (topo: `glm5_next`) | HFcfg:231, :288 | Seleciona a classe de config/modelo | registry | `architectures: ["Glm5NextForConditionalGeneration"]` (HFcfg:2) → vLLM `Glm5NextForConditionalGeneration` (wrapper multimodal, `IsHybrid`, `HasInnerState`) que instancia `Glm5NextForCausalLM` |

### 2.2 `text_config` — MLA (as 11 camadas `deepseek_sparse_attention` + a MTP)

| nome | valor | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `q_lora_rank` | **1536** | HFcfg:245 | Query passa por um gargalo: `q_a_proj [1536×4096]` → `q_a_layernorm` → `q_b_proj [16384×1536]` | MLA, fatoração de baixo posto | Reduz parâmetros/FLOPs da query; o **mesmo latente de query (1536)** alimenta o indexador (`wq_b [4096×1536]`) |
| `kv_lora_rank` | **512** | HFcfg:78 | Latente comprimido de K e V: `kv_a_proj_with_mqa [512×4096]` → `kv_a_layernorm` → **isso é o que vai para o KV cache** | MLA, latent KV, MQA | KV por token por camada = 512 valores (vs 64×(256+256)=32.768 numa MHA equivalente). O kernel SM120 exige `kv_lora_rank=512` (Dockerfile:14, :121) |
| `qk_nope_head_dim` | **256** | HFcfg:247 | Dimensão de Q/K por cabeça sem RoPE | NoPE | `kv_b_proj [32768×512]` = 64 × (256 K + 256 V) reconstrói K/V de cada cabeça a partir do latente (ou é "absorvido" na query em decode) |
| `qk_rope_head_dim` | **0** | HFcfg:248 | Parte RoPE da cabeça = zero | NoPE | Quebra o kernel stock: `fp8_ds_mla` exige `pe_dim=64` (Dockerfile:12). Solução: pad de 64 zeros (§3.1). vLLM também preenche 0 linhas extras em `kv_a_proj_with_mqa` se `mla_nope and qk_rope_head_dim>0` (vmodel:761–816) — aqui o pad é 0 |
| `qk_head_dim` | **256** | HFcfg:246 | = nope + rope | — | Escala do softmax `scaling = qk_head_dim**-0.5 = 1/16` (vattn:449) |
| `v_head_dim` | **256** | HFcfg:258 | Dimensão de V por cabeça | MLA | `o_proj [4096×16384]` = 64×256 → 4096 (o maior tensor denso de atenção: 67,1 M parâmetros) |
| `mla_use_nope` | **true** | HFcfg:183 | Liga o modo NoPE (`skip_rope`) | NoPE | `vcfg:99` mapeia para `mla_nope`; sem esse alias o vLLM ficaria no default. Com NoPE, `rotary_emb=None` (vattn:529–530) |
| `layer_types` | 34× `linear_attention`, 11× `deepseek_sparse_attention` | HFcfg:79 | Tipo de mixer por camada | hybrid | `is_kda_layer()` (vcfg:228). `layers_block_type` colapsa DSA em `"attention"` para a contabilidade híbrida do vLLM (vcfg:236) |
| `num_nextn_predict_layers` | **1** | HFcfg:242 | Nº de camadas MTP | MTP | Camada 45 = DSA+MoE; usada por `SPEC_METHOD=mtp` (`MTP_TOKENS=2`, `start.sh:203`), aplicada recursivamente |

### 2.3 `text_config` — Indexador DSA / kpool (dentro de cada camada MLA)

| nome | valor | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `index_topk` | **2048** | HFcfg:26 | Nº de tokens do passado que cada query MLA atende | DSA, top-k sparse attention | Custo de atenção por query O(2048) em vez de O(L). Para contexto ≤ 2048 a atenção esparsa = densa (`docs/improve-prefill.md:143`). O kernel SM120 só é instanciado para topk ∈ {128,512,1024,2048} e o prefill exige exatamente 2048 (Dockerfile:49–52) |
| `index_n_heads` | **32** | HFcfg:25 | Cabeças do indexador ("lightning indexer") | MQA scoring | `wq_b` projeta o latente de query 1536 → 32×128. DeepGEMM MQA-logits exige 32 ou 64 cabeças (senão pad com zeros, vattn:392–400) |
| `index_head_dim` | **128** | HFcfg:21 | Dimensão de q/k do indexador | — | Fixa o Hadamard-128 e o bloco de quantização 128 (`kpool_compress.py`: `INDEX_HEAD_DIM = 128`). Entrada de cache = 128 B FP8 + 4 B escala = **132 B** |
| `index_kpool` | **4** | HFcfg:22 | Tamanho do "pool": 4 chaves consecutivas do indexador viram **1** entrada armazenada (`compress_ratio=4`) | compressão de chaves, pooling | Cache do indexador ÷4 (33 B/token/camada). Top-k roda em granularidade de pool: `select_k = index_topk // index_kpool = 512` pools → 2048 tokens. `block_size` do cache precisa ser múltiplo de `index_kpool*32` (vattn:139–151) e, no SM12x, de `index_kpool*64=256` (Dockerfile:57–65) |
| `index_kpool_compress` | **true** | HFcfg:24 | Pool = soma ponderada por softmax (e não "pegar 1 a cada 4") | pooling aprendido | Pesos: `index_kpool_compress_gate [128×4096]` (score por canal por token) + `index_kpool_compress_ape [4×128]` (viés posicional por slot) |
| `index_kpool_always_select_tail` | **true** | HFcfg:23 | Sempre inclui os até 3 tokens mais recentes que ainda não formam um pool completo | recency | Largura do buffer vira 2048+3=2051 → arredondada a 2176 (múltiplo de 128) no stock; a receita corta para 2048 (511 pools + cauda) — §3.1 |
| `index_share_for_mtp_iteration` | **true** | HFcfg:27 | Na MTP recursiva, o passo 0 calcula o top-k e os passos seguintes reutilizam | IndexCache-like reuse | `Glm5NextMultiTokenPredictor.set_skip_topk` (mtp.py) |
| `indexer_rope_interleave` | **true** | HFcfg:28 | Estilo de RoPE do indexador (interleave vs NeoX) | RoPE | **Moot**: `Indexer.rope_dim = config.qk_rope_head_dim = 0`, então o ramo RoPE é pulado (vattn:233, :342–364) `[código]` |
| `indexer_types` | 45× `"full"` | HFcfg:29 | Por camada: `full` = roda o próprio indexador; `shared` = reusaria o top-k da anterior | IndexCache | Aqui toda camada DSA roda seu indexador. `use_index_cache` não está no config → `_skip_topk=False` (vattn:559–571) |
| (derivado) `index_dsa_use_layernorm` | default **true** | vcfg:67 | `k_norm` do indexador é **LayerNorm com bias** (eps 1e-6) | LayerNorm | Tensores `indexer.k_norm.weight/bias [128]` |

### 2.4 `text_config` — KDA (`linear_attn_config`, HFcfg:126–180)

| nome | valor | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `linear_attn_config.num_heads` | **64** | HFcfg:127 | Cabeças KDA | linear attention multi-head | 32 por rank em TP2; estado recorrente `[32,128,128]` por rank. vLLM lê via `linear_num_heads` (vcfg:105–112) |
| `linear_attn_config.head_dim` | **128** | HFcfg:129 | d_k = d_v por cabeça | estado matricial | Estado por cabeça 128×128; q/k/v `[8192×4096]` (64×128) |
| `linear_attn_config.short_conv_kernel_size` | **4** | HFcfg:130 | Convolução causal depthwise de 4 taps + SiLU em q, k e v | ShortConv (Mamba/GDN) | Dá "memória local" e ordem de curto alcance. Estado de conv por rank = canais × (kernel−1 + `num_spec`) (`mamba_utils.kda_state_shape`) — com DFlash k=7 são 10 colunas |
| `linear_attn_config.gate_lower_bound` | **−5.0** | HFcfg:128 | Limite inferior do log-decay por canal ("safe gate") | gated delta rule | `g = −5 · sigmoid(exp(A_log)·(f + dt_bias))` ∈ (−5, 0) → fator de esquecimento α = e^g ∈ (0,0067; 1) (vkda:301–315). Impede esquecimento "infinito" e estabiliza numericamente |
| `linear_attn_config.kda_layers` | `[0,1,2,4,5,6,8,…,42,44]` (34) | HFcfg:131 | Quais camadas são KDA | hybrid layout | Redundante com `layer_types` |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23,27,31,35,39,43]` | HFcfg:167 | Quais camadas são MLA/DSA | hybrid 3:1 | Razão 3:1 = Kimi Linear |

### 2.5 `text_config` — MoE e MLP

| nome | valor | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `n_routed_experts` | **288** | HFcfg:235 | Experts roteados por camada MoE | MoE | 288 × 25.165.824 = 7,25 B parâmetros por camada. EP em TP3: 288/3 = 96 por rank |
| `num_experts_per_tok` | **8** | HFcfg:239 | top-k do roteador | sparse activation | `vcfg:92` alerta: sem o alias, o default da classe (7) rotearia 7 experts em vez dos 8 treinados. Ativos por token: 8×25,17M = 201,3 M por camada |
| `n_shared_experts` | **1** | HFcfg:236 | Expert sempre ativo | shared expert (DeepSeekMoE) | `intermediate = moe_intermediate_size × n_shared = 2048`. No checkpoint EXL3 fica **BF16** (não quantizado) |
| `moe_intermediate_size` | **2048** | HFcfg:232 | Largura interna de cada expert SwiGLU | fine-grained experts | TP2: 1024 local por rank (`docs/cooperative-moe.md:26`). **Não divide por 3** e o trellis EXL3 é empacotado em 2048 → TP3 usa expert parallel em vez de padding (`overlay/tp3/patch_tp3_glm.py:8`) |
| `intermediate_size` | **12288** | HFcfg:77 | Largura do MLP denso (camadas 0–2) | dense FFN | 151 M parâmetros por camada densa |
| `first_k_dense_replace` | **3** | HFcfg:14 | As primeiras 3 camadas usam MLP denso | DeepSeek-style dense warm-up | `mlp_layer_types` explicita isso (3× dense + 42× sparse, HFcfg:184) |
| `mlp_layer_types` | 3× `dense`, 42× `sparse` | HFcfg:184 | Tipo de FFN por camada | — | A MTP (índice 45, fora da lista) herda o último tipo (`sparse`) (vmodel:335–343) |
| `scoring_func` | **sigmoid** | HFcfg:252 | Afinidade token→expert = sigmoid(logit) (não softmax) | sigmoid routing (DeepSeek-V3) | Scores independentes por expert; combinados com `norm_topk_prob` |
| `topk_method` | **noaux_tc** | HFcfg:256 | Seleção com viés de correção sem perda auxiliar | aux-loss-free load balancing | Cria `gate.e_score_correction_bias [288]` FP32 (vmodel:187–192): viés entra **só na seleção** do top-8, não no peso |
| `norm_topk_prob` | **true** | HFcfg:237 | Renormaliza os 8 pesos para somar 1 | — | Passado como `renormalize` ao `FusedMoEFactory` |
| `routed_scaling_factor` | **2.5** | HFcfg:250 | Multiplica os pesos dos experts roteados | output scaling | Compensa a renormalização (pesos somam 1 → ×2,5). `apply_routed_scale_to_output=False` |
| `n_group` / `topk_group` | 1 / 1 | HFcfg:234, :255 | Roteamento por grupos (limite de nós) desativado | group-limited routing | Todos os 288 experts competem livremente |
| `moe_router_dtype` | **float32** | HFcfg:233 | Logits do roteador em FP32 | precisão do router | `_get_moe_router_dtype` (vmodel:180); evita empates/flips de roteamento |
| `router_aux_loss_coef` | 0.001 | HFcfg:251 | Coeficiente de perda auxiliar (treino) | load balancing loss | Sem efeito em inferência |
| `output_router_logits` | false | HFcfg:243 | Não devolve logits do roteador | — | — |
| `swiglu_limit` | **10.0** | HFcfg:253 | Clamp do SwiGLU: `gate=clamp(x_g, max=10)`, `up=clamp(x_u, −10, 10)`, `out=silu(gate)·up` | activation clamping | Usado em MLP denso, shared e experts (`SiluAndMulWithClamp`, vLLM `activation.py:201`). O overlay EXL3 replica: `SWIGLU_LIMIT_DEFAULT = 10.0` (`overlay/exl3.py:63`) |

### 2.6 `text_config` — mHC

| nome | valor | onde | o que faz | conceito(s) | por que importa / trade-off |
|---|---|---|---|---|---|
| `mhc` | **true** | HFcfg:182 | Liga Hyper-Connections com restrição de variedade | mHC | Residual com 4 fluxos em todas as 45 camadas (não na MTP) |
| `hc_mult` | **4** | HFcfg:16 | Nº de fluxos residuais (n) | hyper-connections | `mix_hc = (2+n)·n = 24` coeficientes por token por sub-bloco (vmodel:382–400): 4 `pre` + 4 `post` + 16 `comb`. `hc_attn_fn`/`hc_ffn_fn` `[24, 16384]` (16384 = 4×4096) |
| `hc_sinkhorn_iters` | **20** | HFcfg:17 | Iterações Sinkhorn-Knopp para tornar `comb` (4×4) duplamente estocástica | Birkhoff polytope | Garante mistura "conservativa" (norma preservada, produto de matrizes estáveis); `mhc_sinkhorn_iterations` (vcfg:94) |
| `hc_eps` | **1e-6** | HFcfg:15 | ε somado em `pre_mix` e nas normalizações Sinkhorn | estabilidade numérica | — |
| (default vLLM) `mhc_post_mult_value` | **2.0** | vcfg:81 | `post_mix = 2·sigmoid(·)` ∈ (0,2) | — | Não está no checkpoint; é default da classe |
| (default vLLM) `mhc_tau`, `hres_vwnstyle`, `mhc_no_norm_weight` | 0.05, true, false | vcfg:77–80 | Parâmetros da família mHC | — | Não usados no caminho de código lido (`hc_pre`/`hc_post` não os consultam) `[código]` |

Tensores por camada (checkpoint): `hc_attn_fn`/`hc_ffn_fn` BF16 `[24,16384]` (carregados em FP32 no vLLM),
`hc_attn_base`/`hc_ffn_base` F32 `[24]`, `hc_attn_scale`/`hc_ffn_scale` F32 `[3]` (uma escala para
cada um de pre/post/comb). Custo: 786.486 parâmetros por camada, 35,4 M no total `[cálculo próprio]`.

### 2.7 Tokens especiais, geração e processador

| nome | valor | onde | o que faz / por que importa |
|---|---|---|---|
| `eos_token_id` | `[154820, 154827, 154829]` | HFcfg:9; `generation_config.json` | = `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` (tokenizer.json). Parar ao ver `<\|user\|>`/`<\|observation\|>` é o que encerra o turno do assistente/uma chamada de ferramenta |
| `pad_token_id` | 154820 | HFcfg:244 | = `<\|endoftext\|>`; `padding_side: left` (`tokenizer_config.json`) |
| Tokenizer | BPE byte-level, 154.820 tokens + 321.649 merges; pré-tokenização regex estilo GPT-4 (dígitos em grupos de até 3) | `tokenizer.json` | `model_max_length: 1048576`. Especiais 154820–154837 (`[MASK]`, `[gMASK]`, `<sop>`, `<\|system\|>`, `<\|assistant\|>`, imagem/vídeo/áudio) e não-especiais 154838–154855 (`<think>`=154841, `</think>`=154842, `<tool_call>`=154843, `<arg_key>`…, `/nothink`=154851, `<\|image\|>`=154854, `<\|video\|>`=154855) |
| `image_token_id` / `video_token_id` | 154854 / 154855 | HFcfg:281–282 | placeholders expandidos pelo processador para N tokens de visão |
| `image_start/end`, `video_start/end` | 154830/154831, 154832/154833 | HFcfg:283–286 | delimitadores |
| `generation_config` | `temperature 1.0`, `top_p 0.95` | `generation_config.json` | defaults de amostragem do modelo |
| `reasoning_effort` | `low`/`high`/`max` (default `max`) | model card; `GLM53_DEFAULT_REASONING_EFFORT` (`start.sh`, `--default-chat-template-kwargs`) | orçamento de "pensamento"; `clear_thinking` default false |
| `processor_config.image_processor` | `patch_size 14`, `temporal_patch_size 2`, `merge_size 2`, `min_image_tokens 16`, `max_image_tokens 8000`, mean/std CLIP | `processor_config.json` | A receita sobrescreve `max_image_tokens` para **2048** (`MM_IMAGE_TOKENS`, `start.sh:310`) porque 8000 > `MAX_NUM_BATCHED_TOKENS=7168` (tamanho do encoder cache) e porque vídeos viravam ~33 imagens × 7,2k tokens (README "KV cache") |
| `video_processor` | `fps 2`, `max_image_tokens 240000` | idem | — |

### 2.8 `vision_config` (torre ViT) e wrapper multimodal

| nome | valor | onde | o que faz | observação |
|---|---|---|---|---|
| `depth` | 24 | HFcfg:264 | blocos ViT | pré-norm, atenção + MLP SwiGLU com bias |
| `hidden_size` / `num_heads` | 1024 / 16 | HFcfg:266, :272 | largura e cabeças (head 64) | q/k RMSNorm (eps 1e-5 fixo, `multimodal.py`) |
| `intermediate_size` | 4096 | HFcfg:270 | MLP da visão | — |
| `patch_size` / `temporal_patch_size` | 14 / 2 | HFcfg:274, :279 | patch 14×14×2 frames (`patch_embed.proj [1024,3,2,14,14]`) | imagens estáticas são duplicadas no tempo `[inferência, padrão Qwen2-VL/GLM-4V]` |
| `spatial_merge_size` | 2 | HFcfg:277 | 2×2 patches → 1 token (`downsample [4096,1024,2,2]`) | 1 token de visão ≈ 28×28 px × 2 frames |
| `out_hidden_size` | 4096 | HFcfg:273 | = hidden do texto | — |
| `projection_intermediate_size` | 10240 | HFcfg:275 | merger SwiGLU 4096→10240→4096 (com clamp) | — |
| `image_size` | 448 | HFcfg:267 | resolução nominal | processador faz resize dinâmico |
| `rms_norm_eps` | 1e-5 no checkpoint, **forçado a 1e-6** | HFcfg:276; vcfg:289–293 | "Serving with 1e-5 drifts the RMSNorm and produces repetitive/degraded image descriptions" | modificação do vLLM |
| RoPE da visão | RoPE 2D parcial (`partial_rotary_factor 0.5`) | `multimodal.py:411–415` | **única RoPE do modelo-alvo** | sem embeddings posicionais absolutos (variante "OCR") |
| `swiglu_limit` | 10.0 | HFcfg:278 | clamp no merger | — |

A torre tem 563.627.008 parâmetros (1,13 GB BF16), é mantida em BF16 mesmo no checkpoint FP8
(`quant_config=None` para `visual`, vmodel:1044–1055) e na receita roda TP (TP2) ou **data-parallel** em
TP3 (`--mm-encoder-tp-mode data`).

### 2.9 `quantization_config` do checkpoint oficial (FP8) — referência

| nome | valor | onde | o que faz |
|---|---|---|---|
| `quant_method` / `fmt` | `fp8` / `e4m3` | HFcfg:1804, :292 | FP8 E4M3 |
| `weight_block_size` | `[128,128]` | HFcfg:1805 | uma escala FP32 (`weight_scale_inv`) por bloco 128×128 (ex.: expert `gate_proj [2048,4096]` → escala `[16,32]`) |
| `activation_scheme` | `dynamic` | HFcfg:291 | ativações quantizadas em tempo de execução |
| `modules_to_not_convert` | 1.509 entradas | HFcfg:293 | Ficam BF16: embeddings, `lm_head`, todo o KDA (q/k/v/o, gates, convs, `A_log`, `dt_bias`), todo o indexador, `kv_b_proj`, normas, `hc_*`, roteadores, visão, `eh_proj`/`enorm`/`hnorm` da MTP. Ficam FP8: experts, shared experts, MLP denso, `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` da MLA |

Mesmo nesse checkpoint o vLLM **desquantiza para BF16** as projeções MLA no load (`_try_load_fp8_attn_proj`,
vmodel:1149–1224) e mantém KDA sem quant config (vkda:168–174). Contagem HF: F8_E4M3 = 314.396.639.232,
BF16 = 6.926.096.640, F32 = 295.518 (total 321.323.031.390; as escalas `weight_scale_inv` não entram
nessa contagem). `total_size` do índice = 328.326.771.576 B ≈ **305,8 GiB** (o recipe vLLM cita "~306 GiB").

### 2.10 Formas exatas por tipo de camada (cabeçalhos safetensors do checkpoint oficial)

**KDA (ex.: camada 0 e 4)** — todos BF16 exceto `A_log`/`dt_bias` (F32):

| tensor | forma | significado |
|---|---|---|
| `self_attn.q_proj/k_proj/v_proj.weight` | `[8192, 4096]` cada | 64 cabeças × 128 |
| `self_attn.q_conv1d/k_conv1d/v_conv1d.weight` | `[8192, 1, 4]` cada | conv depthwise causal, kernel 4 |
| `self_attn.b_proj.weight` | `[64, 4096]` | β por cabeça (força de escrita da regra delta) |
| `self_attn.f_a_proj.weight` / `f_b_proj.weight` | `[128, 4096]` / `[8192, 128]` | gate de esquecimento **por canal** em baixo posto (rank 128) |
| `self_attn.A_log` | F32 `[64]` | taxa de decaimento base por cabeça (`exp(A_log)`) |
| `self_attn.dt_bias` | F32 `[8192]` | viés do gate por canal |
| `self_attn.g_a_proj.weight` / `g_b_proj.weight` | `[128, 4096]` / `[8192, 128]` | gate de **saída** (sigmoid) em baixo posto |
| `self_attn.o_norm.weight` | `[128]` | RMSNorm por cabeça antes do gate de saída |
| `self_attn.o_proj.weight` | `[4096, 8192]` | projeção de saída |
| `fused_qkvbfg_a_proj` | (nome em `modules_to_not_convert`) | o vLLM funde q,k,v,b,f_a,g_a num único GEMM `in_proj_qkvbfg_a` (vkda:201–219) — TP2 local `[12576×4096]` = 3·4096 + 32 + 128 + 128 |

Parâmetros por camada KDA: **137.732.288** (q/k/v/o = 134,2 M; gates+convs ≈ 3,5 M) `[cálculo próprio]`.

**MLA+indexador (ex.: camada 3 e 45)**:

| tensor | dtype no FP8 oficial | forma | significado |
|---|---|---|---|
| `self_attn.q_a_proj.weight` | F8 | `[1536, 4096]` | down-proj da query |
| `self_attn.q_a_layernorm.weight` | BF16 | `[1536]` | |
| `self_attn.q_b_proj.weight` | F8 | `[16384, 1536]` | 64 × 256 (NoPE) |
| `self_attn.kv_a_proj_with_mqa.weight` | F8 | `[512, 4096]` | só 512 linhas: **sem as 64 linhas de RoPE** |
| `self_attn.kv_a_layernorm.weight` | BF16 | `[512]` | |
| `self_attn.kv_b_proj.weight` | BF16 | `[32768, 512]` | 64 × (256 K_nope + 256 V) |
| `self_attn.o_proj.weight` | F8 | `[4096, 16384]` | |
| `self_attn.indexer.wq_b.weight` | BF16 | `[4096, 1536]` | 32 cabeças × 128, do latente de query |
| `self_attn.indexer.wk.weight` | BF16 | `[128, 4096]` | **uma** chave de 128 por token (MQA) |
| `self_attn.indexer.k_norm.weight/bias` | BF16 | `[128]` | LayerNorm |
| `self_attn.indexer.weights_proj.weight` | BF16 | `[32, 4096]` | peso por cabeça do indexador (calculado em FP32 no vLLM) |
| `self_attn.indexer.index_kpool_compress_gate` | BF16 | `[128, 4096]` | score por canal para o softmax do pool |
| `self_attn.indexer.index_kpool_compress_ape` | BF16 | `[4, 128]` | viés posicional por slot do pool |

Parâmetros: MLA 117.442.560 + indexador 7.471.872 = **124.914.432** por camada `[cálculo próprio]`.

**MoE (camada ≥3)**: `mlp.gate.weight [288,4096]` BF16, `mlp.gate.e_score_correction_bias [288]` F32,
`mlp.shared_experts.{gate,up}_proj [2048,4096]`, `down_proj [4096,2048]`, e 288 × o mesmo trio em
`mlp.experts.E.*`. **MLP densa (0–2)**: `gate/up [12288,4096]`, `down [4096,12288]`.
**MTP (45)**: `eh_proj [4096,8192]`, `enorm`, `hnorm`, `shared_head.norm` `[4096]` + MLA/indexador + MoE.

### 2.11 DFlash2 — `config.json` (`incoai/GLM-5.3-Flash-DFlash2`, idêntico em `dc77ff1c` e `bf582e4e`)

| nome | valor | onde | o que faz | conceito(s) | por que importa |
|---|---|---|---|---|---|
| `architectures` / `model_type` | `DFlash2DraftModel` / `qwen3` | DFcfg:2, :44 | Backbone estilo Qwen3 (QK-norm) | draft model | Registrado pela receita: `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")` (`overlay/patch_dflash2.py:90–96`) |
| `dflash_config.block_size` | **8** | DFcfg:9 | Bloco de query = 1 âncora + 7 máscaras | block diffusion | Casado com `DFLASH_TOKENS=7`; a conv usa `block_size = 1 + num_speculative_tokens` (`overlay/qwen3_dflash2.py:143`) |
| `dflash_config.mask_token_id` | **154856** | DFcfg:12 | id alimentado nas 7 posições a prever | masked diffusion | Fica fora dos 154.856 tokens reais (0–154855), na faixa de padding da tabela de embedding do alvo `[inferência sobre o significado]` |
| `dflash_config.target_layer_ids` | **[5, 14, 24, 33, 42]** | DFcfg:15 | Camadas do alvo cujas saídas alimentam o draft | feature extraction (EAGLE-3 style) | Todas KDA; o hook do alvo captura em `idx+1` → log `(6, 15, 25, 34, 43)` (README "Why the overlay exists"). `fc [4096×20480]` = 5×4096 → 4096 |
| `dflash_config.selector_top_k` | **16** | DFcfg:14 | candidatos por posição | beam/lattice | Top-16 dos logits de cada posição do bloco |
| `dflash_config.selector_rank` | **256** | DFcfg:13 | posto do seletor bilinear | low-rank bilinear | Codebooks predecessor/sucessor `[154880×256]` cada + `hidden_projection [256×4096]` |
| `dflash_config.conv_kernel_size` | **2** | DFcfg:11 | taps da convolução dinâmica | dynamic depthwise conv | "two-tap" (posição atual + anterior no bloco) |
| `dflash_config.conv_group_size` | **16** | DFcfg:10 | canais que compartilham uma correção dinâmica | grouped conv | 4096/16 = 256 grupos → `kernel_projection [1024×4096]` = 2 lados × 2 taps × 256 |
| `num_hidden_layers` | **5** | DFcfg:46 | camadas do draft | — | `tools/pack_profile.py:120` exige 5 camadas, `hidden_size` 4096 |
| `hidden_size` / `intermediate_size` | 4096 / 12288 | DFcfg:31, :33 | igual ao alvo | — | MLP SwiGLU denso por camada (151 M) |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 32 / 8 / 128 | DFcfg:45, :47, :29 | GQA 4:1 | GQA | Não divide por 3 → TP3 usa `DFLASH_DRAFT_TP=1` e `pad-tp3-config.py` 32/8→36/9 |
| `layer_types` / `sliding_window` / `use_sliding_window` / `max_window_layers` | 5× `sliding_attention` / **2048** / true / 5 | DFcfg:35–43, :55, :59 | janela deslizante de 2048 tokens de contexto | SWA | KV do draft limitado à janela → grupo `SlidingWindowSpec` separado (§3) |
| `is_causal` | **false** | DFcfg:34 | Atenção bidirecional dentro do bloco | non-causal | A receita faz o vLLM honrar esse campo (`patch_dflash2.py:36–47`) |
| `rope_parameters` | `rope_theta 10000`, `default` | DFcfg:51–54 | RoPE do draft | RoPE | O draft **tem** RoPE (o alvo não) — aplicada ao K de contexto na posição do token (`precompute_and_store_context_kv`) |
| `num_target_layers` | 45 | DFcfg:48 | camadas do alvo | — | — |
| `vocab_size` | 154880 | DFcfg:60 | igual ao alvo | — | usa `lm_head` do alvo |
| `tie_word_embeddings` / `use_cache` | false / false | DFcfg:56, :58 | — | — | — |
| `eos_token_id` / `pad_token_id` | igual ao alvo | DFcfg:24, :49 | — | — | — |
| `max_position_embeddings` | 1048576 | DFcfg:42 | — | — | — |
| `rms_norm_eps` | 1e-5 | DFcfg:50 | — | — | — |
| (implícito) `input_embedding_scale`, `output_multiplier`, `final_logit_softcapping` | ausentes → 1.0, 1.0, 0 | `overlay/qwen3_dflash2.py:299–328` | escalas opcionais | — | — |

**Tensores do DFlash2** (cabeçalho de `model.safetensors`, 81 tensores, todos BF16, 1.171.080.448 params ≈ 2,34 GB):

| grupo | forma | total |
|---|---|---:|
| `fc.weight` | `[4096, 20480]` | 83.886.080 |
| `hidden_norm.weight` / `norm.weight` | `[4096]` | 8.192 |
| por camada ×5: `q_proj [4096,4096]`, `k_proj/v_proj [1024,4096]`, `o_proj [4096,4096]`, `q_norm/k_norm [128]` | | 5 × 41,9 M |
| por camada ×5: `mlp.{gate,up}_proj [12288,4096]`, `down_proj [4096,12288]` | | 5 × 151,0 M |
| por camada ×5: `attention_conv` e `mlp_conv`: `base_kernel [2,2,4096]` + `kernel_projection [1024,4096]` | | 5 × 8,42 M (41,9 M ao todo; o blog cita "16.5M added parameters (3%)" para outra escala de modelo) |
| `candidate_selector.predecessor_codebook` / `successor_codebook` | `[154880, 256]` cada | 79.298.560 |
| `candidate_selector.hidden_projection.weight` | `[256, 4096]` | 1.048.576 |

Não há `embed_tokens` nem `lm_head` no draft (compartilhados com o alvo).

### 2.12 Checkpoint EXL3/TR3 4 bpw (`brandonmusic/…tr3-4bpw` = `Mia-AiLab/…EXL3-TR3-4bpw`)

| nome | valor | onde | o que faz / por que importa |
|---|---|---|---|
| `quant_method` | `exl3` | `config.json`→`quantization_config` | registrado na imagem pelo Dockerfile (`QuantizationMethods` + `method_to_config["exl3"]`, Dockerfile:387–416) |
| `bits` | **4** | idem | 4 bits/peso (trellis `I16 [in/16, out/16, 64]`: 256 pesos × 4 bits = 64 int16 por tile 16×16) |
| `codebook` | `mcg` | idem | codebook "multiply-congruential" (`MCG_MULTIPLIER = 0xCBAC1FED`, `overlay/exl3.py:57`; tensor `mcg I32 [1]` por matriz = marcador −877912083) |
| `scope` | `glm53_routed_experts_only` | idem | só os 43×288×3 = 37.152 matrizes de experts |
| `head_bits` | 16 | idem | `lm_head` não quantizado |
| `non_routed_dtype_policy` | `official_source_native` | idem | não-roteados copiados do BF16 oficial (`zai-org/GLM-5.3-Flash-BF16` rev `a6c167b6…`) |
| `version` | 0.0.43 | idem | ExLlamaV3 `c5d9c657` (mesmo pin do Dockerfile:380) |
| `serving_reader_qualified` | false | idem / `exl3-mcg-storage-abi.json` | "ExLlamaV3 v0.0.43 has no audited GLM-5.3 TP model load/inference receipt" — a receita traz seu próprio leitor (`overlay/exl3.py`) |
| `quantization/recipe.json` | `profile k4`, `tensor_policy uniform-routed-experts`, `routed_expert_bits 4`, `target_tensor_parallel 2`, `global_allocator_invoked false` | Mia mirror | "uniform-K4": mesmo bitrate para todos os experts (sem alocação por sensibilidade) |
| Tensores por matriz | `trellis I16`, `suh F16 [in]`, `svh F16 [out]`, `mcg I32 [1]` | shard 57, ex.: `layers.3.mlp.experts.0.gate_proj.trellis [256,128,64]`, `down_proj.trellis [128,256,64]` | `suh`/`svh` = vetores de sinal/escala das rotações de Hadamard de entrada/saída (incoerência) |
| Contagem HF | I16 77.913.391.104; BF16 9.669.171.456; F16 228.261.888; I32 37.152; F32 295.518 | API HF | I16 = 311,65 B pesos × 4/16 **exatamente**; F16 = 12.384 × 18.432 (suh+svh) **exatamente** `[cálculo próprio]` |
| Tamanho | 175.622.979.576 B (≈163,6 GiB), 120 shards, 150.226 tensores | índice / `materialization-receipt.json` | cabe em 2×128 GB UMA com folga para KV |

### 2.13 Knobs da receita que existem por causa da arquitetura

(Somente os que tocam arquitetura/speculator/quantização de pesos; linha em `start.sh` salvo indicação.)

| nome | default (alternativas) | onde | o que faz | conceito | por que importa / trade-off |
|---|---|---|---|---|---|
| `MODEL` / `MODEL_REVISION` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` / `25a44fdb…` | 116, 122 | checkpoint EXL3 alvo | pinning | espelho durável do snapshot `5ab363a8` |
| `QUANTIZATION` | `exl3` (`none`) | 288 | `--quantization exl3` | EXL3 | **não** usar `--moe-backend marlin` (isso é para NVFP4, Dockerfile:3–5) |
| `SPEC_METHOD` | `dflash` (`mtp`, `none`) | 205 | método especulativo | speculative decoding | rollback `mtp` usa a camada 45 |
| `DFLASH_MODEL` / `DFLASH_REVISION` | `incoai/GLM-5.3-Flash-DFlash2` / `dc77ff1c…` | 206, 211 | draft fixado | — | "A mutable Hub main has already changed weights" (start.sh:207–210) |
| `DFLASH_TOKENS` | **7** | 212 | `num_speculative_tokens` | block size 8 | bloco treinado = 8 |
| `DFLASH_DRAFT_TP` | **2** (1, vazio=herda) | 218 | `draft_tensor_parallel_size` | TP do draft | 2 = shard do draft de ~2,3 GiB (C4, 2026-08-30: prefill 8k 938 / 16k 972 / 100k 997 tok/s; decode structured 65.1 / prose 27.1). TP3 → 1 (32/8 cabeças não dividem por 3) |
| `--speculative-config` | `{"method":"dflash","model":…,"num_speculative_tokens":7,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":2}` | 2129–2136 | configura o proposer | rejection sampling | `kv_cache_dtype auto` = KV do draft em BF16 (não há backend MLA-FP8 para o draft denso, nem FA3/FA4 FP8 no SM121) |
| `MTP_TOKENS` | 2 | 203 | `{"method":"mtp","num_speculative_tokens":2}` | MTP | capturas de grafo `1 2 3 4 6 8 12` (múltiplos de 3) |
| attention backend do draft | não fixar (`FLASH_ATTN` escolhido) | 216–217; README topo | SWA não-causal | — | `TRITON_ATTN` nessa imagem é causal dentro do bloco → accept cai (structured ~29 tok/s / 0.31) |
| `KV_CACHE_DTYPE` | `fp8` → `fp8_ds_mla` | 274 | KV do alvo | KV quant | único formato aceito pelo kernel sparse-MLA SM12x; "bf16 KV has no sparse kernel on this arch"; NVFP4 KV não disponível (kernels SM12x NVFP4 são MHA densos) |
| `MAX_MODEL_LEN` | 850000 | 222 | contexto | — | E3 reserva 560 MiB que saem do KV; 1M volta com `EXL3_FAT_GROUPED=0` |
| `MAX_NUM_BATCHED_TOKENS` | 7168 | 229 | chunk de prefill | chunked prefill | "8192 chunk × long history oversubscribes GB10 persistent_topk smem" (indexador) |
| `LANGUAGE_MODEL_ONLY` | 0 | 289 | `--language-model-only` desliga a torre de visão | multimodal | libera memória |
| `MM_IMAGE_TOKENS` / `LIMIT_MM` / `MM_PROCESSOR_CACHE_GB` / `SKIP_MM_PROFILING` | 2048 / `{"image":48,"video":1}` / 1 / 1 | 310, .env | limites da visão | — | ver §2.7 |
| `GLM53_DENSE_FP8` | `.env.example:158` = **`all`**; `start.sh:433` = `off` se não definido | — | FP8 weight-only (Marlin) nas projeções densas BF16, por grupo: `shared`, `dense`, `kda` (`in_proj_qkvbfg_a,f_b_proj,g_b_proj,o_proj`), `mla` (`fused_qkv_a_proj,q_b_proj,o_proj`; **`kv_b_proj` fica BF16** porque é lido direto nas matmuls absorvidas) | W8A16 per-output-channel | ~−11 ms/step; muda numérica (KL proxy 0,002–0,013 nats/pos). Só camadas do alvo (índice < 45) |
| `GLM53_KDA_BF16_LARGE_M` | `.env.example:162` = 1; `start.sh:440` = 0 | — | para M > 512 linhas o `in_proj` KDA usa cópia BF16 da mesma matriz FP8 lógica (cuBLAS) | dispatch por tamanho de M | prefill −12 % TTFT; +3,3 GiB/rank |
| `GLM53_DENSE_EXL3` / `GLM53_MODEL_PRESET=dense-h3` / `GLM53_DENSE_EXL3_PREFILL_BF16` | 0 / vazio / off | 433–455; README "Pack-selected dense EXL3" | quantiza os linears não-roteados em EXL3 (turboderp 4.05bpw) + draft DFlash2 em EXL3 6 bpw | dense EXL3 | opt-in, não A/B medido; seis grupos H3 `kda_in,shared_down,mla_qkv_a,shared_gate_up,kda_o,mla_q_b` retêm BF16 para M > 144 |
| `GLM53_DRAFT_KV_COMPACT` | 1 com `dflash`, 0 caso contrário | 472–474 | páginas de KV do draft maiores (maior múltiplo de 64 que divide o bloco MLA e cabe na página) | paged KV | reduz ids de bloco do draft de 65 para 6 por request (CPU allocator) |
| `GLM53_ADAPTIVE_K` / `_SET` | off / `2,4,7` | 419 | verifica só um prefixo 2/4/7 dos 7 rascunhos, escolhido por EMA | adaptive speculation | lossless; prose +10–21 % (2026-09-08) |
| `GLM53_INDEXER_WORKSPACE` | `rightsize` (`stock`) | 465 | dimensiona o workspace de gather do indexador em pools, não em tokens | compress_ratio | stock = 5036,40 MB travados a 1M; rightsize ≈ 126 MB (MAX_NUM_SEQS=4) → +26–28 % KV |
| `TP3_HEAD_OVERRIDE` / `ENABLE_EXPERT_PARALLEL` / `MM_ENCODER_TP_MODE` | 66 / 1 / `data` | `start-tp3.sh:418–425` | `--hf-overrides {"num_attention_heads":66,"num_key_value_heads":66,"linear_num_heads":66}`; EP; visão DP | divisibilidade TP | "Do not pad heads to 96 (one rank becomes all dummy KDA heads and logits collapse)" |
| `ABLIT`, `ABLIT_METHOD`, `ABLIT_LAYERS`, `ABLIT_ALPHA`, `ABLIT_INCLUDE_MTP`, `ABLIT_DIRECTION` | 0, auto, `15-45`, 3.0, 1, dealign | `.env.example:240–251` | edição de `o_proj` (KDA `[4096,8192]`, DSA `[4096,16384]`) por transplante ou ortogonalização da "direção de recusa" | abliteration | camadas 0–14 = "safety-anchor" (stock); 45 = MTP |
| capture sizes CUDA graph | DFlash2 `1 2 4 8 16 24 32`; MTP `1 2 3 4 6 8 12`; adaptive-k adiciona múltiplos de 3,5,8 | `start.sh:320–345` (`configure_capture_sizes`); README | grafos por tamanho de lote de decode | CUDA graphs | cada step de decode = seqs × (k+1) tokens |

---

## 3. Inventário de modificações (foco: o que muda por causa da arquitetura)

Cada item: **nome | arquivo(s) | problema | mecanismo técnico | conceito(s) | efeito medido**.
Itens marcados "(upstream vLLM)" não são da Mia, mas são adaptações arquiteturais que já vêm no vLLM
`487ecf187` e que você precisa conhecer para reproduzir a receita.

### 3.1 Overlay "NoPE sparse-MLA no SM121" (bloco Python do `Dockerfile`, linhas 91–318; auto-verificação 320–367)

**Problema geral (Dockerfile:1–84, README "Why the overlay exists")**: a imagem stock carrega o
checkpoint e morre no primeiro forward com `pe_dim must be 64 for fp8_ds_mla`. No SM12x o único backend
de MLA esparsa é `FLASHINFER_MLA_SPARSE_SM120`, que exige o registro empacotado `fp8_ds_mla` =
**512 B NoPE (FP8) + 16 B escalas + 128 B RoPE (64×BF16) = 656 B**, `kv_lora_rank=512`,
`qk_rope_head_dim=64` e dimensão de query **576**. FlashInfer só instancia três tipos: DSV3_2 e GLM_NSA
(d_qk=576, 656 B, topk ∈ {128,512,1024,2048}) e DSV4 (d_qke=512, 584 B, topk ≤ 1024). GLM-5.3-Flash
precisa de d_qk=512 com topk 2048 — nenhum kernel existe.

| # | modificação | arquivo alvo na imagem | mecanismo | conceito | efeito |
|---|---|---|---|---|---|
| a | **Pad RoPE de zeros (geometria GLM_NSA 576)** | `v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Se `qk_rope_head_dim == 0` (e `kv_lora_rank == 512`, senão `NotImplementedError`): `self.rope_pad = 64`; `kernel_qk_rope_head_dim = 0 + 64`; query `q = F.pad(q, (0, 64))`; em `do_kv_cache_update` o `k_pe` passado é `new_zeros(…, 64)` (Dockerfile:115–210) | NoPE, produto interno, zero-padding | **Exato**: um bloco RoPE nulo soma 0 ao produto QK, o softmax não muda e V vem da região NoPE de 512 (d_v=512). Custo: 656 B/token em vez de ~528 (≈ +24 % de KV nas 11 camadas DSA). Sem trabalho de kernel. `MLAAttentionSpec.real_page_size_bytes` já aloca `block_size × 656` |
| b | `supports_dense_mha_prefill = False` | idem | O impl herdava `MLAAttentionImpl.forward_mha` (que lança `NotImplementedError`); declarar False roteia **o prefill também** pelo kernel MQA top-k (Dockerfile:31–36, :108–113) | prefill denso vs esparso | o prefill usa a semântica esparsa do próprio modelo; limpa metadados de prefill denso |
| c | **Top-k como capacidade + contagens válidas** | idem | `triton_convert_req_index_to_global_index(..., return_valid_counts=True)`; `sparse_mla_top_k = topk_indices_physical.shape[1]` (largura do buffer), `seq_lens=topk_lengths`, `max_seq_len=capacidade`; linhas com 0 candidatos apontam para slot 0 com comprimento 1 e têm a saída zerada (Dockerfile:150–194) | sparse attention, top-k variável | corrige `expects sparse block_tables shape (256, 1, 2048), got (256, 1, 2176)` |
| d | **Largura do buffer top-k = 2048** (não 2051→2176) | `models/glm5next/nvidia/model.py` e `mtp.py` | substitui `buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)` por `buffer_width = topk_tokens` (Dockerfile:214–221) | kpool tail | o kernel SM120 tem top-k como **parâmetro de template** (prefill rejeita topk ≠ 2048; `_DECODE_DSV3_2_DISPATCH` só {128,512,1024,2048}) |
| e | **Expandir 511 pools + cauda** | `model_executor/layers/sparse_attn_indexer_kpool.py` | `expand_pools_and_append_tail(pool_ids[:, : select_k - 1], …)` nos caminhos de prefill e decode (Dockerfile:255–276) | top-k de pools | 511 pools × 4 = 2044 + até 3 tokens de cauda = 2047 ≤ 2048. Abre mão de 4 de 2048 candidatos (0,2 %), **os últimos em ordem de rank**, preservando a cauda recente (maior peso de atenção num modelo causal e impossível de recuperar pelos pools). `select_k` fica 512 para manter o caminho top-k rápido (gate `select_k in {512,1024,2048}`) |
| f | **Bloco de kernel MLA fixo em 64** | `v1/attention/backends/mla/flashinfer_mla_sparse.py` | `get_supported_kernel_block_sizes()` de `[64, 256]` → `[64]` (Dockerfile:223–237) | paged KV, kernel page | todos os kernels GLM_NSA/DSV3_2 são instanciados só com `PAGE_BLOCK_SIZE=64`; com bloco de gerenciador 1792 o seletor pegaria 256 |
| g | **Alinhamento do indexador ao page 64 do DeepGEMM no sm_120** | `platforms/cuda.py` | `index_kpool * min(PAGED_MQA_PAGE_SIZES)` passa a usar só páginas de 64 quando `capability.major == 12` (Dockerfile:239–253) | DeepGEMM paged MQA | DeepGEMM aceita `block_kv` {32,64,128} no sm_10x mas **só 64** no sm_120. O alinhamento vai de `4×32=128` para `4×64=256`: no exemplo do próprio Dockerfile o bloco híbrido passa de 1664 (armazenamento 416, não múltiplo de 64 → assert) para 1792 (448, múltiplo de 64) (Dockerfile:57–65). Na configuração servida (TP2, DFlash k=7) o bloco resultante é **3584** = 14×256 (§4.3) |
| h | **Pular autotune FlashInfer** | `model_executor/warmup/kernel_warmup.py` | remove `flashinfer_sparse_mla_decode_autotune_warmup` e retorna antes do `AutoTuner` (Dockerfile:278–300); flag de serve `--no-enable-flashinfer-autotune` | autotuning | "FlashInfer's sparse-MLA autotune and fused_moe gemm1/gemm2 autotune kill rank 0 on SM121" |
| i | **PDL só em Hopper/SM10x** | `platforms/cuda.py` | `return major >= 9` → `return major in (9, 10)` (Dockerfile:302–317) | Programmatic Dependent Launch | "PDL lowering races KDA state kernels on SM12x (GB10)" |

### 3.2 Adaptações do modelo já presentes no vLLM `487ecf187` (upstream vLLM)

| modificação | arquivo | problema | mecanismo | conceito |
|---|---|---|---|---|
| Aliases de config | `transformers_utils/configs/glm5_next.py:85–112` | checkpoint usa nomes do SGLang (`num_experts_per_tok`, `hc_mult`, `hc_sinkhorn_iters`, `mla_use_nope`, `linear_attn_config`) | lê os nomes do checkpoint com `.get` mantendo ambos | "Without this, our default (num_experts_per_token=7) was used, routing 7 experts/token instead of the trained 8" |
| Visão eps 1e-6 forçado | `glm5_next.py:289–293` | checkpoint diz 1e-5 mas a torre foi treinada com 1e-6 | `self.rms_norm_eps = 1e-6` | descrições repetitivas/degradadas com 1e-5 |
| Fusões de projeção | `model.py:728–745` | muitos GEMMs pequenos | `gate_proj+up_proj→gate_up_proj`; `q_a_proj+kv_a_proj_with_mqa→fused_qkv_a_proj`; `wk+weights_proj→wk_weights_proj`; KDA `q,k,v,b,f_a,g_a→in_proj_qkvbfg_a` (6→1 lançamentos) | kernel fusion |
| Shards replicados f_a/g_a | `kda.py:58–118` | `f_a`/`g_a` (128) são pequenos e usados por todas as cabeças | `_Glm5NextMergedColumnParallelLinear` multiplica o output_size dessas partes por `tp_size` e força `tp_rank=0` no load | TP com shards replicados |
| Pad NoPE de `kv_a_proj_with_mqa` | `model.py:761–816`, `:1208–1216` | o modelo espera `kv_lora_rank + qk_rope_head_dim` linhas | concatena zeros (aqui 0 linhas, pois rope=0) | NoPE |
| Desquantização FP8→BF16 das projeções MLA | `model.py:1149–1224` | checkpoint FP8 oficial guarda q_a/kv_a/q_b/o em FP8 bloco-128, mas o modelo as mantém BF16 | `_dequant_fp8_block` com padding para blocos parciais | FP8 block scaling |
| Indexador: pesos por cabeça em FP32 | `attention.py:321–336` | "bf16 head-gates introduce ~1e-2 logit error that flips near-tie pool rankings on hard long-context tasks (single-needle retrieval is unaffected, but HLE quality drops)" | `weights = hidden.float() @ W_wp.T` em FP32 | precisão numérica do ranking |
| Indexador: Hadamard-128 na query + FP8 ue8m0 | `attention.py:366–385`; `kpool_compress.py:78–157` | as chaves comprimidas são armazenadas já rotacionadas (Hk); sem girar a query, "the needle's pool drops out of the top-k" | `fwht128_quant_fp8` (butterflies FP32, 1/√128 exato, escala potência de 2) | rotação de Hadamard, FP8 |
| Compressão kpool | `kpool_compress.py:202–327` | armazenar 1 chave a cada 4 tokens | por canal: `softmax_slot(gate_score + ape)` pondera as 4 chaves brutas → soma → Hadamard → FP8 absmax por vetor | pooling aprendido |
| Tail cache (`KpoolTailSpec`) | `attention.py:155–203` | o pool incompleto (≤ 3 tokens) precisa sobreviver entre passos | buffer circular de 1 bloco por request (K bruto BF16 + score), sem prefix caching | circular buffer |
| KDA fora do quant config | `kda.py:168–174` | "GLM5-Next keeps the KDA projections BF16 even in fp8 checkpoints" | `vllm_config.quant_config = None` durante a construção | — |
| KDA "safe gate" | `kda.py:301–315` | gate padrão ilimitado `−exp(A)·softplus(g+bias)` | `y = lower_bound · sigmoid(exp(A)·(g+g_bias))`, `lower_bound=−5` | gated delta rule |
| KDA caminho spec-verify | `kda.py:444–563` | durante a verificação especulativa o estado recorrente precisa poder "voltar" | tokens spec carregam `num_spec+1` colunas de estado e `num_accepted_tokens` escolhe o estado final; conv state com largura `kernel−1+num_spec` | rollback de estado |
| `@eager_break_during_capture` no KDA | `kda.py:373` | o corpo do prefill KDA tem ramificações de host | ponto de quebra "breakable CUDA graph" | CUDA graphs |
| mHC fusões | `model.py:441–511`; `layers/mhc.py` | 2 kernels por fronteira | `hc_post` da camada ℓ fundido com `hc_pre` da ℓ+1 (`MHCFusedPostPreOp`); última camada materializa e contrai | kernel fusion |
| MTP sem mHC | `model.py:419–439`; `mtp.py` | — | camada MTP usa residual simples e devolve o par não somado para um `fused_add_rms_norm` | MTP |

### 3.3 Registro EXL3 + experts empacotados (`Dockerfile:369–441`; `overlay/exl3.py`)

* **Problema**: registrar o nome `"exl3"` não basta; experts precisam ficar **trellis + suh + svh + mcg** e
  rodar os kernels Trellis/MCG (README "Why the overlay exists"). Nunca expandir para BF16 persistente.
* **Mecanismo**: adiciona `"exl3"` a `QuantizationMethods` e `method_to_config["exl3"] = Exl3Config`
  (Dockerfile:387–416); compila ExLlamaV3 `c5d9c657` para `TORCH_CUDA_ARCH_LIST=12.1a` com stubs de CPU
  aarch64 (`overlay/patch_exl3_ext_aarch64.py`), kernels "fat" próprios (`exl3_fat_gemm.cu`,
  `exl3_fat_moe.cu`) e pipeline de decode; copia `overlay/exl3.py` como método de quantização do vLLM.
  TP=2: `gate/up` column-parallel, `down` row-parallel, **um all-reduce por camada MoE** (`overlay/exl3.py:1–14`).
  Não-roteados → `UnquantizedLinearMethod` (BF16) salvo `GLM53_DENSE_FP8`/`GLM53_DENSE_EXL3`.
* **Conceitos**: quantização trellis/QTIP, codebook, incoerência por Hadamard, MoE grouped GEMM.
* **Efeito**: pesos cabem (175,6 GB); decode "one fused `exllamav3_ext.exl3_moe` launch per layer";
  prefill E3 +37–45 % vs E2 (README "Cold prefill").

### 3.4 `overlay/patch_glm_eagle3.py` — estados auxiliares do alvo através do mHC

* **Problema**: o DFlash2 usa a interface EAGLE-3 de "aux hidden states" (saídas de camadas do alvo), que o
  `Glm5NextModel` não implementava; e no meio da pilha mHC o estado é `[tokens, 4, 4096]` com o `hc_post`
  **adiado** (não materializado).
* **Mecanismo**: `Glm5NextModel` herda `EagleModelMixin`; `Glm5NextForCausalLM`/`…ForConditionalGeneration`
  ganham `SupportsEagle3`; no laço de camadas, se `idx+1 ∈ aux_hidden_state_layers`, materializa
  `layer.hc_post(hidden, residual, post, comb)` e faz `hc_contract` (média dos 4 fluxos → `[tokens,4096]`),
  devolvendo `(hidden_states, aux_hidden_states)` (linhas 48–92).
* **Conceito**: EAGLE-3 feature taps; mHC.
* **Efeito**: taps logados como `(6, 15, 25, 34, 43)` = saídas das camadas 5,14,24,33,42.

### 3.5 `overlay/patch_dflash2.py` + `overlay/qwen3_dflash2.py` + `overlay/dflash2_speculator.py` — instalar o DFlash2

| parte | mecanismo | conceito | por quê |
|---|---|---|---|
| Registro | `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")` e o seletor `DFlash2Speculator` quando a arquitetura do draft é DFlash2 (`patch_dflash2.py:90–178`) | model registry | a imagem só conhecia DFlash 1 |
| Honrar `is_causal` | `_dflash_layer_causal` passa a devolver `bool(config.is_causal)` se existir, antes da regra "SWA = causal" (`patch_dflash2.py:36–47`) | máscara bidirecional | o checkpoint IncoAI diz `is_causal: false`; sem isso as camadas SWA seriam causais e a aceitação das posições finais colapsa |
| KV do draft em `auto` | se o KV do alvo é `fp8_ds_mla`/`fp8*`/`nvfp4` e o speculative config não define dtype, `draft_kv = "auto"` (`patch_dflash2.py:98–155`) | KV dtype | "Dense DFlash2 attention cannot use the target's MLA-only fp8_ds_mla layout, and SM121 has no FA3/FA4 for plain FP8 KV" |
| `DFlashGroupedConv` | `base_kernel [2,taps,H]` + `kernel_projection H→2·taps·(H/16)`; `prepare()` convolui antes da sub-camada com os coeficientes do lado 0 e devolve os do lado 1; `finish()` convolui depois. Coeficiente por canal = base + delta dinâmico por grupo de 16 canais; taps só dentro do bloco (`position % block_size >= tap`) (`qwen3_dflash2.py:32–114`) | convolução dinâmica depthwise causal intra-bloco | propaga informação da posição anterior dentro do bloco → combate o "suffix decay" |
| `DFlash2Qwen3DecoderLayer.forward` | `norm → attn_conv.prepare → self_attn → attn_conv.finish → norm → mlp_conv.prepare → mlp → mlp_conv.finish` (`:158–178`) | pre-norm residual | — |
| `CandidateSelector` | `hidden = W_h·h_l` (rank 256); `score[b,l,p,c] = unary_logit[b,l,c] + Σ_r pred[cand_{l−1,p}][r]·hidden[l][r]·succ[cand_{l,c}][r]` (a âncora é o predecessor da posição 0) (`:181–251`) | modelo de transição bilinear de baixo posto (tipo CRF/HMM) | escolhe combinações de tokens coerentes entre posições vizinhas |
| `compute_candidates` | logits do `lm_head` do alvo → `torch.topk(·, 16)` (`:330–337`) | top-k | "The image's LogitsProcessor has no `get_top_k_tokens`" |
| `_selector_walk_kernel` | passo a passo pelas 7 posições: carrega os 16 scores condicionados ao índice escolhido antes (`previous`), faz `argmax` (T=0) ou **Gumbel-max** (T>0), grava token (`dflash2_speculator.py:50–107`) | caminho guloso num treliça; Gumbel-max trick | amostragem do draft compatível com rejection sampling |
| `_cache_draft_logits_kernel` | guarda os scores realizados dos 16 candidatos como "logits do draft" (resto = −inf) (`:110–140`) | distribuição q do draft | a amostragem por rejeição precisa de q(x) para ser sem perda |

### 3.6 `overlay/patch_glm5_drafter_group.py` — grupo KV do drafter no layout híbrido

* **Problema**: o gerenciador híbrido do GLM-5-Next só aceitava grupos MLA + Mamba(KDA) + KpoolTail; as 5
  camadas SWA do DFlash2 desqualificavam o caminho rápido; herdar o bloco MLA (1152 e depois 3584 tokens)
  tornava cada camada do draft "dezenas de MiB por bloco" e fixava a concorrência logada perto de 1×.
* **Mecanismo**: particiona as specs `SlidingWindowSpec` (tipo exato — `KpoolTailSpec` é subclasse) como um
  grupo extra, **anexado por último**. Se a página do draft cabe exatamente na página MLA (`mla_page %
  draft_bytes_per_token == 0`, múltiplo de 64, compatível com o bloco MLA) → **exact-fit**; senão →
  **padded slot-share**: `block_size=64`, `page_size_padded=mla_page`, e a camada *i* do draft
  "co-possui" o tensor MLA *i* em ids de bloco disjuntos (como os grupos Mamba). Com
  `GLM53_DRAFT_KV_COMPACT=1`: maior múltiplo de 64 que divide o bloco MLA e cabe na página física (ex.:
  bloco MLA 3584 a 656 B/token e draft 2048 B/token → 896 tokens). Também protege `worker/utils.py` contra
  dividir páginas "padded" em blocos de kernel menores.
* **Conceito**: PagedAttention, KV cache híbrido com página uniforme, sliding window.
* **Efeito**: `padded slot-share block=64 mla_page=2351104 (was block=16); draft_bytes/token=2048`; ocupação
  ~36k tokens: 44,6 % (compact-64 sem slot-share) → ~16 % (slot-share); admissão do draft 65 → 6 ids por
  request com COMPACT (README "Experimental compact DFlash2 cache pages").

### 3.7 `overlay/patch_hybrid_prefix_hit.py` — prefix caching num modelo híbrido com drafter

* **Problema (issue #7)**: `dflash` conta como `use_eagle()`, e como o GLM não anota `is_eagle_group`, o
  coordenador marcava **todos** os grupos como EAGLE → MLA descartava seu último bloco alinhado (~3584
  tokens); e o grupo SWA do draft, após o "pop" EAGLE, re-alinhava para baixo 3584 tokens, zerando hits.
* **Mecanismo**: só grupos `SlidingWindowSpec` exatos são EAGLE e não encolhem `curr_hit_length`; Mamba
  continua no `min` (pular um checkpoint Mamba faltante seria erro de correção); com COMPACT, o lookup do
  draft termina exatamente na fronteira, porque o KV de contexto do DFlash na posição p é uma projeção por
  posição do estado oculto do alvo em p (ao contrário do EAGLE, cujo KV em p embute o token p+1).
* **Conceito**: prefix caching (APC), estado recorrente não-reconstruível, KV por posição.

### 3.8 Kpool tail: `overlay/patch_kpool_tail_slotmap.py` e `overlay/patch_kpool_tail_seed_stride.py`

| patch | problema | mecanismo | efeito |
|---|---|---|---|
| slotmap clamp | `KpoolTailSpec` tem **1 bloco** por request, mas o kernel genérico faz `block_table[req, pos // block_size]` sem limitar o índice → para `pos ≥ block_size` lê lixo; os kernels seed/update escrevem por esse id e corrompem o indexador de outra camada (~2k tokens de geração) | `block_indices = tl.minimum(block_indices, block_table_stride - 1)` em `vllm/v1/worker/block_table.py` → slot = `block_table[req,0]·block_size + pos % block_size` | identidade para os outros grupos; corrige crashes/corrupção silenciosa |
| seed stride (backport vLLM #57477) | o seed do tail no prefill usava stride denso de 2048 B, mas o tail é uma view `as_strided` da página do indexador: **118.272 B** (`stride(0)=59136` elementos BF16) no bloco 3584 da receita (38.016 B no bloco 1152 upstream) → escrevia em blocos de outros requests | kernel passa a receber `TAIL_BLOCK_ELEMS`/`KPOOL_HEAD` = `tail.stride(0)`/`tail.stride(1)` (fixture `kpool_tail_seed_kernel-db1bfdd.py.txt:41–45`) | "dense tail writes from all 552 block ids land in indexer blocks 0–9" antes da correção |

Nota de cálculo `[cálculo próprio]`: 118.272 B = (3584/4) entradas × 132 B = 896 × (128 FP8 + 4 B escala) —
a página do indexador por camada por bloco de 3584 tokens.

### 3.9 Estado KDA ("Mamba") — `overlay/patch_mamba_align_state_free.py` e `overlay/patch_mamba_align_chunking.py`

* **Problema**: em `mamba_cache_mode="align"` o gerenciador mantém um bloco de estado corrente por request e
  lembrava o bloco anterior num único slot `last_state_block_idx`; com scheduling assíncrono e chunks de
  7168 contra blocos de 3584, o slot era sobrescrito antes de o bloco poder ser liberado (vazamento até o fim
  do request). A reserva era `2 + num_speculative_blocks` páginas.
* **Mecanismo**: lista `stale_state_block_idxs` por request, liberando cada estado superado quando o prefixo
  processado passa dele; reserva máxima residente = `1 + max_concurrent_batches + num_speculative_blocks`.
  O patch de chunking usa o LCM do scheduler para checkpoints compartilhados e respeita fronteiras de
  páginas Mamba menores.
* **Conceito**: estado recorrente (SSM/linear attention) como "KV" de tamanho constante; checkpoints de
  estado para prefix caching; async scheduling.

### 3.10 `overlay/patch_indexer_workspace.py` (`GLM53_INDEXER_WORKSPACE=rightsize`)

* **Problema**: `get_max_prefill_buffer_size` = `max_model_len × 40` entradas de 132 B → a 1M:
  **5.036,40 MB** reservados no profiling (saem do KV) e travados; mas o splitter é alimentado com
  comprimentos **comprimidos** (÷ `compress_ratio = index_kpool = 4`) — dimensionado em tokens, indexado em
  pools. O DeepSeek-V4 upstream já divide por `compress_ratio`; o GLM5Next não.
* **Mecanismo**: `entries = min(max_num_seqs, MNBT) × cdiv(max_model_len + num_speculative_tokens, compress_ratio)`.
* **Efeito**: 126 MB a `MAX_NUM_SEQS=4` (504 MB a 16) → **~+26–28 % KV** `[medido-repo]`.

### 3.11 Pesos densos: `overlay/patch_dense_fp8.py` + `overlay/exl3.py` (FP8 Marlin, KDA BF16 large-M)

* **FP8 weight-only**: após carregar BF16, `process_weights_after_loading` quantiza **por canal de saída**
  para FP8 E4M3 e reempacota para o kernel **Marlin** (W8A16: pesos FP8, ativações BF16). Grupos e sufixos
  em `overlay/exl3.py:1895–1917`. Só camadas do alvo (`layer_idx < len(layer_types)`); `.mtp`, `visual`,
  `draft` excluídos. TP3: `f_b_proj`/`g_b_proj` ficam BF16 (pitch de linha não alinhado a 8 com 22 cabeças).
  **Por que funciona na arquitetura**: no decode, cada passo lê todos os pesos densos (KDA q/k/v/o são
  4,68 B de parâmetros); FP8 halving = menos bytes → "~11 ms less per step".
* **KDA BF16 large-M** (`GLM53_KDA_BF16_LARGE_M=1`, `docs/kda-bf16-large-m.md`): o `in_proj` KDA local
  `[12576×4096]` roda decode com M=1..220 e prefill com M≥768; para M > 512 usa uma cópia BF16 da **mesma**
  matriz lógica FP8 (e4m3 × escala BF16 armazenada) via cuBLAS; ≤ 512 fica no Marlin. Recusa o caminho
  W8A8 (quantizar ativações) por causar flips de probabilidade (KL 7,99× o nulo em M=65).
* **Efeito**: `all` vs `dense,kda`: decode ms/ciclo −4,67 % / −4,91 %, tok/s +4–5 %; KDA BF16: W-PRE
  16k −12,06 %, 64k −12,61 %; +3,3 GiB/rank (79,65 → 82,94 GiB de load) (`docs/adopt-fp8-all-kda-bf16-2026-09-26.md`).

### 3.12 Dense EXL3 + draft EXL3 6 bpw (`GLM53_MODEL_PRESET=dense-h3`, `overlay/patch_dflash2_exl3.py`, `tools/dense_overlay.py`, `tools/dflash2_exl3_*.py/.sh`, `tools/pack_profile.py`)

* **O quê**: alvo TR3 + tensores densos EXL3 de `turboderp/GLM-5.3-Flash-exl3` 4.05bpw (commit `2a30229e`,
  ~5,3 GB por range-read) + DFlash2 IncoAI `dc77ff1c` quantizado para **6 bpw** com o fork
  `MiaAI-Lab/exllamav3@63b32f0` (~25 min no GB10). Metadado `glm53_profile.name = dense-exl3-h3-dflash2-6bpw`.
* **Arquitetura relevante**: no draft, q/o/MLP/conv-kernel/fc em EXL3 6-bit; **K/V BF16** (`bf16_shards=[1,2]`)
  porque o KV de contexto é pré-computado por um GEMM fundido de K/V (`patch_dflash2_exl3.py` "shape-guard the
  fused context-KV weight build"); seletor, normas e `base_kernel` BF16 ("selector precision directly steers
  acceptance", `qwen3_dflash2.py:229–231`). No alvo, `kv_b_proj` e embeddings nativos; seis grupos H3 retêm
  BF16 para M > 144.
* **Status**: opt-in, TP2 only, "not yet A/B measured".

### 3.13 `overlay/patch_adaptive_k.py` (`GLM53_ADAPTIVE_K=ema`)

* **Problema**: com 7 rascunhos e aceitação baixa (prosa: 0,341), verificar sempre 8 posições desperdiça
  compute do alvo.
* **Mecanismo**: o drafter continua propondo 7; o scheduler verifica só um **prefixo** (2, 4 ou 7) escolhido
  por EMA (α 0,25, margem 1,0, MIN_STEPS 4) do nº de rascunhos aceitos, **uniforme no batch** (o
  `FLASHINFER_MLA_SPARSE_SM120` e o backend KDA só suportam grafos FULL de batch uniforme) → captura um
  grafo por comprimento (consultas 3, 5, 8).
* **Efeito**: lossless a T=0; Silk Road +21 %, sky/sunset +13 %, hash-map +10 %, code +5–15 %, counting
  inalterado (2026-09-08, 8 runs/prompt, 131k e 850k).

### 3.14 TP=3 (`overlay/tp3/*`, `start-tp3.sh`)

| forma | stock | correção | por quê |
|---|---|---|---|
| cabeças atenção/KV/KDA | 64 | **66** (22/rank) via `--hf-overrides` + `patch_tp3_glm.py` | 64 não divide por 3; FlashInfer SM120 decode só instancia cabeças locais {8,16,32,64,128} → pad kernel-side 22→32 e fatiar de volta. **96 não**: um rank fica só com cabeças KDA falsas e os logits colapsam |
| `A_log` (1-D, 64) | 64 | 66 no load | acompanha as cabeças KDA |
| vocab | 154.880 | `padding_size = lcm(64,3)=192` → 154.944 | divisibilidade |
| shared-expert intermediate | 2048 | 2112 ou `disable_tp` | divisibilidade |
| routed `moe_intermediate_size` | 2048 | **não padar**; `--enable-expert-parallel` (96 experts/rank) | trellis EXL3 empacotado 2048-wide |
| visão | — | `--mm-encoder-tp-mode data` | divisibilidade |
| DFlash2 32/8 cabeças | — | `DFLASH_DRAFT_TP=1` + `pad-tp3-config.py` GQA 32/8→36/9 | "the divisibility tax" |
| KDA in_proj local | — | `[8726×4096]` (3·22·128 + 22 + 128 + 128) | FP8 large-M path lista essa forma |

### 3.15 Outros patches com efeito arquitetural

| patch | o que faz | conceito |
|---|---|---|
| `overlay/patch_flashkda_tp3.py` | opt-in: prefill KDA chunked via FlashKDA (vLLM PR #55737), mantendo estado recorrente FP32 e gate limitado; decode recorrente inalterado | kernel de linear attention |
| `overlay/patch_sparse_mla_slice.py` (TP4) | fatia a chamada final de decode sparse-MLA em ≤ 64 linhas de query (`VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64`), geometria `[T,16,576]`, páginas `[64,656]`, top-k 2048 | mitigação de stall |
| `overlay/patch_glm_video_placeholders.py` | alinha placeholders de vídeo ao `grid_t` do encoder (1 s de vídeo podia virar 300 slots para 100 tokens) e **desliga `persistent_topk` no GB10** (`select_k in (512,1024,2048)` → `False`), usando `top_k_per_row_decode` | indexador top-k; multimodal |
| `overlay/patch_ablit.py` + `overlay/ablit_runtime.py` + `ablit/*` | edita `o_proj` (saída que escreve no fluxo residual) das camadas 15–45: transplante byte-a-byte do doador `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` (~2,7 GiB por range-read) ou ortogonalização contra a direção `r` (hidden 4096), α=3.0 | abliteração / direção de recusa |
| `overlay/patch_kv_capacity_log.py` | loga cada grupo KV (spec, camadas, block_size, page, blocos/request) | contabilidade híbrida |
| `extensions/cooperative_moe/` | kernel cooperativo de decode especializado: H=4096, I local 1024 (TP2), top-k 8, K4 MCG, 1–32 linhas | MoE decode kernel |

---

## 4. Números e benchmarks

### 4.1 Parâmetros: total vs ativos (contagem exata a partir das formas) `[cálculo próprio]`

| componente | por unidade | nº | total | ativo por token |
|---|---:|---:|---:|---:|
| `embed_tokens` | 634.388.480 | 1 | 0,634 B | (lookup de 1 linha; convenções variam) |
| `lm_head` | 634.388.480 | 1 | 0,634 B | 0,634 B |
| Atenção KDA | 137.732.288 | 34 | 4,683 B | 4,683 B |
| Atenção MLA + indexador | 124.914.432 | 11 | 1,374 B | 1,374 B |
| MLP denso (12288) | 150.994.944 | 3 | 0,453 B | 0,453 B |
| Expert roteado (SwiGLU 2048) | 25.165.824 | 42×288 | 304,406 B | 42×8 = 8,456 B |
| Expert compartilhado | 25.165.824 | 42 | 1,057 B | 1,057 B |
| Roteador (+bias) | 1.179.936 | 42 | 0,050 B | 0,050 B |
| mHC (`hc_*`) | 786.486 | 45 | 0,035 B | 0,035 B |
| Normas (decoder + final) | 8.192 | 45 | ~0,0004 B | — |
| **Camada MTP** (MLA+idx+shared+router+`eh_proj`+normas + 288 experts) | 185 M + 7,248 B | 1 | 7,433 B | (só quando usada como speculator) |
| Torre de visão | 563.627.008 | 1 | 0,564 B | (só em tokens de imagem) |
| **Total** | | | **321.323.031.390** (bate exato com o HF) | **~16,1 B** (camadas) / **17,38 B** (+emb+head) / **17,94 B** (+visão) ≈ "18B active" |

Fração roteada: 311,65 B / 321,32 B = **97,0 %** dos parâmetros estão nos experts (incl. MTP) — por isso
quantizar *só* os experts já resolve a memória.

### 4.2 Memória dos pesos por formato

| formato | tamanho | fonte |
|---|---:|---|
| BF16 total | ~642,6 GB | `[cálculo próprio]` 321,3 B × 2 |
| FP8 oficial (bloco 128×128, não-roteados sensíveis em BF16) | 328.326.771.576 B ≈ **305,8 GiB** | índice HF; recipe vLLM "~306 GiB" |
| TR3 K6 6 bpw (`malaiwah/…TR3-6bpw`) | 254 GB | README "Quality (KLD)" |
| NVFP4 (brandonmusic stack) | ~180 GB | README "Quality (KLD)" |
| **EXL3/TR3 4 bpw (este)** | **175.622.979.576 B ≈ 163,6 GiB** (README "~164 GiB", tabela KLD "176 GB") | índice do espelho |
|  ↳ experts roteados (4 bpw) | 155,83 GB | `[cálculo próprio]` 311,65 B × 0,5 B |
|  ↳ `suh`+`svh` (FP16) | 0,457 GB | `[cálculo próprio]` |
|  ↳ não-roteados BF16 (inclui visão 1,13 GB, embed 1,27, head 1,27, KDA 9,37, MLA 2,75, shared 2,11, dense 0,91, MTP 0,37) | 19,34 GB | `[cálculo próprio]` |
| Por rank em TP2 (load vLLM) | `Model loading took 79.65 GiB` (→ 82,94 GiB com KDA BF16 large-M) | README ~1241; `docs/kda-bf16-large-m.md:108` |
| Por nó em TP3 | ~55,75 GiB | README tabela TP3 |
| DFlash2 BF16 | 2,34 GB ("~2.3 GiB", start.sh:213) | cabeçalho safetensors |
| Um expert a 4 bpw | 12,0 MiB (6,0 MiB por rank em TP2) | `[cálculo próprio]`; `docs/improve-prefill.md` usa "6.3 MB (TP half)" |

### 4.3 KV cache e estado: aritmética por token e por request

**Registro `fp8_ds_mla` (por token, por camada MLA, por rank — o latente MLA é replicado em cada rank TP):**

| parte | bytes |
|---|---:|
| latente NoPE 512 × FP8 | 512 |
| escalas (4 × FP32, uma por 128 dims) | 16 |
| RoPE 64 × BF16 (**zeros** neste modelo) | 128 |
| **total** | **656** (necessário de fato: 528) |

**Indexador**: 1 entrada por pool de 4 tokens × (128 B FP8 + 4 B escala) = 132 B / 4 = **33 B/token/camada**;
página por bloco de 3584 tokens = 896 × 132 = **118.272 B** (confere com `overlay/patch_kpool_tail_seed_stride.py:71`).

**Custo que cresce com o contexto (alvo, por rank)**: 11 × (656 + 33) = **7.579 B/token**
→ 1M tokens = **7,06 GiB**, 850k = 6,0 GiB `[cálculo próprio]`. O README diz "~7.4 GiB fixed + 7.1 GiB per
1M" e "one 1M request needs **14.52 GiB** KV (10.98 GiB at 500k)" (`start.sh:219–221`, README `.env`) — o
termo "7.1 GiB per 1M" bate com a conta.

**Estado KDA (constante no comprimento)**, por camada, por request, por rank TP2, com DFlash k=7
(`mamba_utils.kda_state_shape`, `kda_state_dtype` → conv em BF16, recorrente FP32):

| parte | forma por rank | bytes |
|---|---|---:|
| estado recorrente | `[32 cabeças, 128, 128]` FP32 | 2.097.152 |
| estado de conv | `[3·8192/2 = 12288 canais, 4−1+7 = 10]` BF16 | 245.760 |
| **total** | | **2.342.912** (×34 camadas = 76 MiB/request/rank) |

**Por que o bloco MLA é 3584 tokens** `[cálculo próprio, confere com os logs]`: o alocador híbrido usa uma
página única para todos os grupos; o bloco de atenção precisa ter página ≥ estado KDA:
2.342.912 / 656 = 3571,5 tokens → próximo múltiplo de 256 (= `index_kpool × 64`, §3.1g) = **3584** →
`mla_page = 3584 × 656 = 2.351.104 B` — exatamente o `mla_page=2351104` do log de boot. Em TP3 (22 cabeças
locais): (1.441.792 + 168.960)/656 = 2455 → **2560**, `mla_page = 1.679.360` — exatamente o
`MLA block_size=2560`, `mla_page=1679360` do boot TP3 de 2026-09-23 no README.

**Custo por id de bloco do pool** (`docs/DESIGN-apc-per-group-retention.md:142`): `per_block = 11·mla_page + 11·idx_page`
= 27.163.136 B ≈ 25,9 MiB (= 7.579 B/token × 3584). Um request de 1M consome 280 ids MLA (+1 kpool tail +
4 grupos Mamba × 9 + drafter ~97) = 414 ids no boot de 643 ids (1,55×). Um segmento de 3584 tokens em cache
custa 38 ids (1 MLA + 4 Mamba + 33 drafter) → ~57k tokens de "conversa cacheável" com nada rodando.

**Drafter DFlash2**: 2048 B/token/camada em TP2 (4 cabeças KV locais × 128 × {K,V} × 2 B; 4096 em TP1);
janela 2048 → ~20 MiB para as 5 camadas numa janela cheia `[cálculo próprio]`. Log: `draft_bytes/token=2048`.

**Comparações didáticas** `[cálculo próprio]`:

| hipótese | B/token (todas as camadas) | 1M tokens |
|---|---:|---:|
| MHA BF16 clássica, 45 camadas × 64 cabeças × (256 K + 256 V) | 2.949.120 | ~2,95 TB |
| MLA fp8_ds_mla em todas as 45 camadas (sem KDA) | 29.520 | 27,5 GiB |
| **GLM-5.3-Flash como servido (11 MLA + 11 indexador) + KDA constante** | **7.579** | **7,06 GiB** (+ ~76 MiB de estado KDA/request) |

Alegação da Z.ai citada em fonte secundária (InferenceX #2794): "3.01x less attention compute" e "4.44x
smaller KV cache" que o GLM-5.3 (744B). Kimi Linear (paper): até 75 % menos KV e 6× decode a 1M vs MLA cheia.

### 4.4 Pools de KV medidos (datas)

| data | configuração | pool | fonte |
|---|---|---|---|
| 2026-08-29 | 1M, util 0.87, MNBT 2048, padded slot-share | **1.754.237** tokens (1,75×), 690 blocos, 18,67 GiB | README "KV cache" |
| 2026-09-01 | 1M, MNBT 7168, `MAX_NUM_SEQS=4`, E2 + rightsize | **1.243.902** (1,24×) | README |
| 2026-09-07+ | E3, 900k, util 0.85/0.86 | 989.010 / 1.023.626–1.084.615 | README "What runs" |
| 2026-09-08 | 850k, KV cap 14 GiB (`--kv-cache-memory-bytes 15032385536`), sem compact | 876.958 (1,03×); ~5 GiB livres (15 GiB: 1,11× mas só 0,8–2,2 GiB livres; 13 GiB ≈ 820k não cabe) | README "Faster prose decode" / `.env.example` |
| 2026-09-26 | 850k, 14 GiB, compact draft pages, util 0.85 | **1.572.073** (1,85×) | README "What runs" |
| 2026-09-26 (default atual) | 850k, **11 GiB** (`--kv-cache-memory-bytes 11811160064`), compact, FP8 all + KDA BF16 | **1.233.779** (1,45×) | `docs/adopt-fp8-all-kda-bf16-2026-09-26.md` |
| 2026-09-15 | TP3, 1M, cap 40 GiB | 3.230.656 (3,23×), 2213 blocos | README TP3 |

"GPU KV cache size" = `max_concurrency × max_model_len`, **não** o tamanho do prefix cache (README "What the KV cache boot line means").

### 4.5 Decode (2× GB10, TP2)

| data | carga | ×1 stream | ×2 | ×4 | notas |
|---|---|---:|---:|---:|---|
| 2026-08-28 | sparkDash Structured/Code, DFlash2 k=7, T=0, thinking off, 400 tok, 1M | **62,9** tok/s (TTFT 719 ms) | 51,7 (agg 103,3) | 37,1 (agg 146,5) | README "Decode" |
| 2026-09-17 | sparkDash Prose, adaptive-k ema, FP8 `dense,kda`, coop MoE, 850k, KV 14 GiB | **36,1** (agg 37,1; TTFT 333 ms) | 25,0 (agg 51,1) | 19,4 (agg 75,3) | ×3: 22,3 (65,8) |
| 2026-09-08 | adaptive-k + dense FP8 (sem coop) | 32,1 | 22,1 (agg 41,2) | — | TTFT 268/399 ms |
| 2026-08-30 | lab `tests/bench_decode.py`, C4, `DFLASH_DRAFT_TP=2`, mediana 5×400 | Structured **65,1** (accept 0,959, 6,71/step); Prose hash-map **27,1** (0,341, 2,39/step) | | | TP=1 prévio: 61,7 / 26,9; contexto 60–100k: 24–27; **MTP k=2 ~24,6** |
| — | `attention_backend=TRITON_ATTN` pinado | structured ~29 tok/s / accept 0,31 | | | posição 0 ok, posições seguintes colapsam (máscara causal no bloco) |

**Aceitação por posição (lab, DFlash2 k=7)**: Structured 0,98 / 0,98 / 0,94 / 0,94 / 0,91 / 0,83 / 0,83;
Prose 0,75 / 0,58 / 0,41 / 0,28 / 0,16 / 0,09 / 0,06. O decaimento em prosa é o fenômeno que motiva o
adaptive-k (§3.13) e, no próprio DFlash2, as convoluções de 2 taps (blog: recall@16 cai de 99,5 % na
posição 1 para 87,8 % na posição 6 sem elas).

Ganhos de decode atribuídos a mudanças arquiteturais `[medido-repo]`: FP8 denso (`dense,kda`) "~11 ms less
per step", +10 % counting, prosa +12–19 %, **+37 %** em prosa difícil somado ao adaptive-k (KL proxy 0,002–0,013
nats/posição, argmax 94–100 %); `all` vs `dense,kda` −4,7/−4,9 % ms/ciclo (2026-09-26).

Estimativa didática `[cálculo próprio]` de por que decode é limitado por banda de memória (usando os
"~230 GB/s effective LPDDR5x" de `docs/improve-prefill.md`): um passo DFlash2 verifica 8 tokens × top-8 =
64 slots de expert por camada → ~57 experts distintos esperados (de 288: 288·(1−(1−1/288)^64)) × 6 MiB/rank ×
42 camadas ≈ 15 GB lidos por rank por passo só de experts (~65 ms). Os pesos densos BF16 por rank
(KDA ~4,7 GB, MLA+indexador ~1,3, shared ~1,1, `lm_head` ~0,6, MLP denso ~0,45 ≈ 8 GB) custam ~35 ms;
converter KDA + denso para FP8 corta ~2,6 GB ≈ 11 ms — a mesma ordem dos "~11 ms less per step" medidos.
Com structured a 62,9 tok/s e ~7,7 tokens/passo (6,71 aceitos + 1 bônus) o passo real é ~120 ms, coerente
com essa soma + atenção + drafter + all-reduce. Sem especulação cada token pagaria sozinho ~8 experts × 42 camadas
+ todos os densos — é por isso que especular em MoE compensa, mas menos que em modelo denso (mais experts
distintos por passo).

### 4.6 Prefill frio (2× GB10)

| data | config | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-09-07 (E3, 900k, util 0.86) | MNBT 7168, rightsize, `MAX_NUM_SEQS=4`, DFlash2 draft TP2 | 1.492,1 | 1.553,7 | 1.428,2 | 1.587,0 | 1.561,7 | 1.516,8 tok/s |
| 2026-09-06 A/B (E3, 500k, util 0.84) | idem | 999 (1º após boot) | 1.578 | 1.624 | 1.637 | 1.629 | 1.576 |
| E2 antes | | 1.112 warm | 1.155 | ~1.160 | ~1.170 | 1.148–1.183 (100k) | 1.087 |
| 2026-09-01 PR77 (E2 vs legacy, MNBT 2048) | 5 amostras únicas | 941→1132 (+20,33 %) | | | | 100k: 1023→1242 (+21,36 %) | 300k: 995→1201 (+20,70 %) |
| 2026-08-29 (baseline) | MNBT 1024 | 772 | 893 | | | 100k 947 | 256k 936; 300k 928 |

Camada MoE de 7168 tokens isolada: 77–91 ms (E2) → **31 ms** (E3). Roofline do repo (`docs/improve-prefill.md`):
por chunk de 1024 tokens, top-8 × 1024 = 8192 slots / 288 experts → ~28 linhas por expert (todos ativam),
~75,6 GB lidos por GPU por chunk; a MNBT 7168 são ~199 linhas/expert `[cálculo próprio]`. FLOPs/token
estimados pelo repo: routed 16,9 GFLOP (47 %), KDA 11,6 (33 %; o repo usa ~170 M/camada, eu conto 137,7 M),
MLA+indexador 3,3 (9 %), shared 2,1, denso 0,9, lm_head 1,27; ≈ 36 GFLOP/token.

### 4.7 Qualidade (KLD vs teacher BF16)

| modelo | KLD média (nats) | tamanho | fonte |
|---|---:|---:|---|
| TR3 K6 (6 bpw) | 0,013723 | 254 GB | painel de malaiwah (25 janelas seladas, 51.175 posições), README |
| FP8 oficial (cross-stack) | 0,020615 | 328 GB | idem |
| **EXL3 4 bpw (este)** | **0,024555** | 176 GB | idem; model-index do HF: 0,024554564249958208 |
| FP8 oficial (stack brandonmusic v44) | 0,024629 | 328 GB | idem |
| NVFP4 (stack brandonmusic v44) | 0,060535 | ~180 GB | idem |
| 5 runs, TP2, KV FP8 MLA, eager, sem MTP (v44) | média 0,024629; top-1 agreement 0,93796; primeiros 64 tokens 0,1919 vs restante 0,0192 | — | `runtime-results-v44.json` |
| KV NVFP4 MLA (v75) | 0,054757 (falhou gate de qualidade LAVD-low 3/10) | — | card brandonmusic |

"On the same stack, 4bpw matches official FP8 (~1.00× KLD) at 54% of the bytes."

### 4.8 TP3 (3× GB10)

Prose sparkDash 2026-09-17, 512 tok: ×1 **40,1** (TTFT 255 ms), ×2 28,7 (agg 56,6), ×3 25,5 (75,5),
×4 22,8 (88,4). Lab 2026-09-14 (count/hashmap/LRU): structured 87,8, code 54,9, prose 39,6 vs TP2 73,4 /
45,0 / 32,9. jspark3 3× ~95 structured; `DFLASH_DRAFT_TP=1` é "the divisibility tax".

### 4.9 Números externos dos métodos (para contexto)

| fonte | número |
|---|---|
| DFlash (arXiv 2602.06036) | 5 camadas de draft (8 para Qwen3-Coder), bloco 16 (10 em LLaMA-3.1); Qwen3-8B greedy: GSM8K τ=6,54 (5,15×), MATH-500 τ=7,87 (6,08×), HumanEval τ=6,50 (5,14×); média 4,9×; "2.4× faster than EAGLE-3"; SGLang batch 8: 5,1× com τ=8,01; peso da perda por posição `w_k = exp(−(k−1)/γ)` |
| Blog DFlash 2 (inco.ai) | top-16 candidatos; seletor +0,34 tokens a T=0 com +2,0 M params e +0,6 % latência; conv 2-tap +16,5 M params (3 %), +0,7 % latência; "over 20% more output from every verification pass"; Qwen3.8-27B média de aceitação 4,80 (MTP 4,28; DSpark 3,62); 2,7–3,4× throughput |
| Card brandonmusic (SM120, 2× RTX PRO 6000) | DFlash2-7 aceitação GSM8K 5,43–5,74 média; "the partially ported Triton mask scored 1.017 weighted" → corrigir para "full bidirectional visibility inside the draft block with a backward-only historical window" → 5,068/5,739; C1 decode 145–151 tok/s; prefill 6,2k tok/s |
| Kimi Linear (arXiv 2510.26692) | 48B total / 3B ativos; razão 3:1 KDA:MLA; MLA NoPE; até 75 % menos KV e 6× decode a 1M |
| mHC (arXiv 2512.24880) | Sinkhorn converge "within 20 iterations"; ganho máximo de sinal ~3000 (HC) reduzido em ~3 ordens de grandeza |
| Model card GLM-5.3-Flash | 320B/18B; 30T tokens multimodais; Terminal-Bench 2.1 84,3; DeepSWE 63,4; HLE w/ tools 55,3 (2026-08-26) |

---

## 5. Conceitos necessários (em ordem de pré-requisito)

Cada conceito: **o que é** (2–4 frases) → **como aparece aqui**.

### A. Fundamentos de ML e números

1. **Tensor, matriz e produto matricial (GEMM).** Uma rede neural é uma composição de multiplicações de
   matrizes de pesos W por vetores/matrizes de ativações X, mais funções não lineares. Uma "camada linear"
   `y = W·x` com W `[out, in]` custa `2·out·in` FLOPs por token. → Todas as peças do GLM-5.3 são GEMMs com
   formas exatas (§2.10): ex. `q_proj [8192,4096]`, expert `gate_proj [2048,4096]`.

2. **Parâmetros vs ativações vs estado.** Parâmetros (pesos) são fixos após o treino; ativações são
   intermediários de um forward; "estado" é o que precisa ser guardado entre tokens na geração (KV cache ou
   estado recorrente). → Memória do serve = pesos (163,6 GiB) + KV/estado (pool de 11 GiB) + ativações/
   grafos; em UMA tudo sai do mesmo pool de 128 GB por nó.

3. **Treino vs inferência.** No treino ajustam-se pesos por gradiente numa perda (ex.: entropia cruzada do
   próximo token); na inferência só se roda o forward. → Campos como `router_aux_loss_coef`,
   `initializer_range` e a perda posicional do DFlash `w_k = exp(−(k−1)/γ)` só importam no treino; a receita
   só faz inferência, mas precisa respeitar escolhas de treino (ex.: 8 experts, eps 1e-6 da visão).

4. **Logits, softmax, amostragem e temperatura.** O modelo produz um logit por token do vocabulário; o
   softmax transforma em probabilidades; temperatura T divide os logits antes do softmax (T=0 ≡ argmax).
   → 154.880 logits por posição; `generation_config`: T=1.0, top_p=0.95; benchmarks do repo em T=0.

5. **Divergência KL (KLD).** Mede quanto uma distribuição Q difere de uma referência P: `Σ P·log(P/Q)`, em
   nats. É a métrica padrão para "quanto a quantização mudou o modelo". → EXL3 4 bpw: 0,0246 nats vs teacher
   BF16 (≈ FP8 oficial); NVFP4 0,0605; o FP8 denso da receita é avaliado por "KL proxy" 0,002–0,013.

6. **Formatos numéricos: FP32, BF16, FP16, FP8 (E4M3), inteiros.** BF16 = 8 bits de expoente e 7 de
   mantissa (faixa do FP32, pouca precisão); FP16 = mais precisão, menos faixa; FP8 E4M3 máx ±448;
   quantização inteira guarda códigos + escalas. → pesos não roteados BF16; `suh/svh` FP16; roteador e
   estado KDA em FP32; KV em FP8; `A_log`/`dt_bias`/`e_score_correction_bias` FP32.

### B. Deep learning (blocos básicos)

7. **Funções de ativação: SiLU e SwiGLU (com clamp).** SiLU(x)=x·σ(x). SwiGLU = `silu(W_g x) ⊙ (W_u x)` seguido
   de `W_d`: um "portão" multiplicativo que dá mais expressividade que ReLU. Clamp limita valores extremos.
   → Todas as MLPs/experts são SwiGLU; `swiglu_limit 10.0` → `gate ≤ 10`, `up ∈ [−10, 10]` (SiluAndMulWithClamp).

8. **Normalização: RMSNorm e LayerNorm.** RMSNorm divide o vetor pela raiz da média dos quadrados e multiplica
   por um peso aprendido (sem subtrair a média); LayerNorm subtrai média e tem bias. Estabilizam a escala
   das ativações. → RMSNorm eps 1e-5 em todo o texto; `kv_a_layernorm`, `q_a_layernorm`; o indexador usa
   LayerNorm com bias (`k_norm`); o KDA usa `o_norm` gated (RMSNorm × sigmoid).

9. **Conexão residual e "residual stream".** `x ← x + f(norm(x))`: cada sub-camada *soma* sua contribuição num
   fluxo que atravessa a rede; isso preserva gradientes e cria um "barramento" de informação. → É o que o
   mHC generaliza (4 fluxos) e o que a abliteração edita (o `o_proj` escreve no fluxo).

10. **Embedding e LM head (tied vs untied).** Embedding: tabela `[vocab, d]` que transforma id em vetor. LM
    head: matriz `[vocab, d]` que transforma o último estado em logits; "tied" = mesma matriz. → Não amarrados
    (`tie_word_embeddings: false`), 634 M parâmetros cada; MTP e DFlash2 reusam ambos do alvo.

11. **Convolução 1D causal depthwise.** Cada canal é filtrado independentemente por uma janela pequena de
    tokens passados (kernel k), sem olhar o futuro. Captura padrões locais baratos. → KDA: kernel 4 + SiLU em
    q, k, v (`q/k/v_conv1d [8192,1,4]`); DFlash2: conv **dinâmica** de 2 taps (coeficientes dependem do
    estado oculto) dentro do bloco.

12. **Fatoração de baixo posto (low-rank).** Uma matriz `[m,n]` pode ser aproximada por `[m,r]·[r,n]` com
    r ≪ m,n, reduzindo parâmetros e computação. → MLA (q via 1536, KV via 512), gates KDA (rank 128),
    seletor DFlash2 (rank 256).

### C. Transformer / LLM

13. **Tokenização BPE byte-level e tokens especiais.** O texto é quebrado em bytes e depois mesclado em
    subpalavras frequentes (merges); tokens especiais marcam papéis e modalidades. → 154.820 tokens BPE +
    321.649 merges + 36 especiais (`<|user|>`, `<think>`, `<tool_call>`, `<|image|>`…); EOS inclui
    `<|user|>` e `<|observation|>` (fim de turno/ferramenta).

14. **Modelo autoregressivo (next-token prediction).** O LLM prevê `P(x_t | x_<t)` e gera um token por vez,
    realimentando a saída. → Cada token gerado exige um forward completo das 45 camadas — o gargalo que a
    decodificação especulativa ataca.

15. **Self-attention (Q, K, V) e escala.** Para cada token, uma query q compara-se (produto interno) com as
    keys k de todos os tokens permitidos; softmax(qkᵀ/√d) pondera os values v. Complexidade O(L²) no prefill
    e O(L) por token no decode. → MLA com `scaling = 256^-0.5`; KDA substitui isso por uma recorrência.

16. **Multi-head attention (MHA), MQA e GQA.** Várias cabeças independentes; MQA = todas as cabeças de query
    compartilham 1 K/V; GQA = grupos compartilham. Reduzem KV cache. → Draft DFlash2: GQA 32 q / 8 KV;
    indexador DSA: MQA (1 chave de 128 para 32 cabeças); MLA em decode opera como MQA sobre o latente.

17. **Máscara causal vs bidirecional.** Causal: token t só vê ≤ t (necessário para gerar). Bidirecional: vê
    tudo no escopo. → Alvo: causal. DFlash2: **bidirecional dentro do bloco** de 8 (`is_causal: false`) e
    janela de 2048 para trás no contexto; usar um backend com máscara causal (TRITON_ATTN nessa imagem)
    derruba a aceitação para 0,31.

18. **Codificação posicional: RoPE, APE e NoPE.** RoPE gira pares de dimensões de q e k por um ângulo que
    depende da posição (posição relativa entra no produto qk); APE soma um vetor por posição; NoPE não dá
    posição nenhuma — a ordem precisa vir de outro lugar (máscara causal, recorrência, convolução). →
    GLM-5.3: MLA e indexador **NoPE**; posição vem das camadas KDA e do APE de 4 slots do pool; RoPE 2D só
    na visão; o draft usa RoPE θ=10000.

19. **KV cache.** Guardar K e V dos tokens passados para não recomputá-los a cada token gerado; cresce
    linearmente com o contexto e é o maior consumidor de memória em contexto longo. → Aqui só 11 das 45
    camadas têm KV que cresce (656 + 33 B/token/camada); as 34 KDA têm estado fixo.

20. **Prefill vs decode; compute-bound vs memory-bound.** Prefill processa o prompt inteiro em paralelo
    (muitas linhas por GEMM → limitado por FLOPs); decode gera 1 token por request por passo (poucas linhas →
    limitado por leitura de pesos da memória). → Receita otimiza os dois separadamente: E2/E3 "fat experts"
    no prefill; FP8 denso, grafos CUDA, kernel cooperativo e especulação no decode.

21. **Contexto longo e custo quadrático.** Atenção cheia custa O(L²) de compute e O(L) de KV por camada;
    a 1M tokens isso é proibitivo. → Três respostas no GLM-5.3: atenção linear (KDA), atenção esparsa top-2048
    (DSA) e compressão de KV (MLA latente + FP8).

### D. Arquitetura do modelo

22. **MLA — Multi-head Latent Attention.** K e V de todas as cabeças são gerados de um único vetor latente
    comprimido `c_kv` (aqui 512 dims) por token; só `c_kv` vai para o cache. Em decode a projeção de K pode
    ser "absorvida" na query (`q·W_UKᵀ`) e a de V na saída, de modo que a atenção roda direto no espaço
    latente (MQA com d_qk=512, d_v=512). → `kv_a_proj_with_mqa [512,4096]`, `kv_b_proj [32768,512]`; o
    kernel vê 576 = 512 + 64 (pad); `kv_b_proj` fica BF16 mesmo com FP8 denso "because MLA reads its weight
    directly for the absorbed matmuls".

23. **MLA NoPE.** Variante sem a parte RoPE desacoplada (em DeepSeek-V3 são 64 dims de RoPE por cabeça).
    Simplifica o cache (só o latente) mas depende de outra fonte de posição. → `qk_rope_head_dim: 0`; é o
    motivo do overlay de pad 576 (§3.1a) e de o indexador também não ter RoPE.

24. **Atenção esparsa e DSA (DeepSeek Sparse Attention).** Em vez de atender a todos os L tokens, cada query
    atende a um subconjunto top-k escolhido por um **indexador** barato ("lightning indexer"): no desenho
    DSA, `I(t,s) = Σ_h w_{t,h}·ReLU(q_{t,h}·k_s)` com poucas cabeças em FP8, e só os top-k maiores entram na
    atenção MLA. Custo do indexador ainda é O(L) por query, mas com vetores pequenos em FP8. → `index_topk
    2048`, `index_n_heads 32`, `index_head_dim 128`; kernels DeepGEMM `fp8_mqa_logits`/`fp8_fp4_paged_mqa_logits`.

25. **Pooling de chaves do indexador (kpool) e cauda.** O GLM-5.3 comprime as chaves do indexador: a cada 4
    tokens, as 4 chaves são combinadas por uma média ponderada **por canal** (softmax sobre os 4 slots de
    `gate_score + APE`); o top-k escolhe **pools** (512), que são expandidos para 2048 tokens; os ≤ 3 tokens
    mais recentes que ainda não formam pool são sempre incluídos (cauda). → `index_kpool 4`,
    `index_kpool_compress_gate [128,4096]`, `…_ape [4,128]`, `KpoolTailSpec`; patches §3.1d/e, §3.8.

26. **Rotação de Hadamard para quantização.** Multiplicar um vetor por uma matriz de Hadamard normalizada
    (±1/√n, ortogonal) espalha "outliers" por todas as dimensões, deixando a distribuição mais uniforme — o
    erro de quantização fica menor e o produto interno é preservado se q e k forem ambos girados. → Indexador:
    FWHT-128 em q e nas chaves comprimidas antes do FP8 (escala ue8m0 = potência de 2); EXL3 também usa
    Hadamard (`suh`/`svh`).

27. **Atenção linear (kernel trick) e estado recorrente.** Removendo o softmax, `o_t = Σ_s (q_t·k_s) v_s =
    q_t·(Σ_s k_s v_sᵀ)`: basta manter a matriz `S_t = Σ k vᵀ` (d_k×d_v) e atualizá-la a cada token — custo
    O(1) por token e memória constante. Problema: memória "enche" e interfere. → Estado KDA `[128×128]` por
    cabeça, 64 cabeças, FP32.

28. **Regra delta (DeltaNet).** Em vez de só somar `k vᵀ`, primeiro "apaga" o que o estado já associava a k:
    `S_t = S_{t−1} − β_t k_t (k_tᵀ S_{t−1}) + β_t k_t v_tᵀ = (I − β k kᵀ) S + β k vᵀ` — uma atualização de
    mínimos quadrados online com taxa β. → `b_proj [64,4096]` → β por cabeça (sigmoid); q e k L2-normalizados
    (`use_qk_l2norm_in_kernel=True`).

29. **Gated DeltaNet e KDA (Kimi Delta Attention).** Gated DeltaNet multiplica o estado por um fator de
    esquecimento α ∈ (0,1] (escalar por cabeça) antes da regra delta. KDA torna α **por canal** (diagonal):
    `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`, `o_t = S_tᵀ q_t`, com uma saída
    normalizada e "gated". → GLM: `α = exp(g)`, `g = −5·sigmoid(exp(A_log)·(f_b(f_a(x)) + dt_bias))` (safe
    gate, ∈(−5,0)); gate de saída `sigmoid(g_b(g_a(x)))` sobre `RMSNorm(o)`; conv curta antes.

30. **Algoritmo chunkwise (prefill) vs recorrente (decode) para atenção linear.** No prefill divide-se a
    sequência em chunks: dentro do chunk calcula-se em paralelo (forma matricial, WY/UT transform) e entre
    chunks passa-se o estado; no decode aplica-se a recorrência token a token. → `chunk_kda_with_fused_gate`
    (prefill) e `fused_recurrent_kda` (decode e spec-verify); FlashKDA opcional (§3.15); cópia BF16 para
    M > 512 (§3.11).

31. **Arquitetura híbrida (linear + atenção completa/esparsa).** Camadas lineares são baratas mas têm
    memória limitada (recall exato fraco); intercalar algumas camadas de atenção "exata" recupera recall de
    longo alcance. → 3 KDA : 1 MLA-DSA, como Kimi Linear; é o que faz o vLLM tratar o modelo como
    `IsHybrid` com grupos de KV "Mamba" + atenção.

32. **Mixture-of-Experts (MoE).** Troca a MLP única por N "experts" MLP e um roteador que escolhe k por token;
    parâmetros totais grandes, compute por token pequeno. "Fine-grained" = muitos experts pequenos; "shared
    expert" = um expert sempre ativo que captura conhecimento comum. → 288 experts × 2048, top-8, + 1 shared;
    97 % dos parâmetros; 8,46 B ativos roteados por token.

33. **Roteamento sigmoid, viés sem perda auxiliar e escala roteada.** Scores `s_e = σ(u·e_e)` independentes;
    para balancear carga sem perda auxiliar, soma-se um viés `b_e` aprendido/ajustado **só na escolha** do
    top-k; os pesos dos escolhidos são renormalizados (soma 1) e multiplicados por um fator (2,5). →
    `scoring_func sigmoid`, `topk_method noaux_tc`, `e_score_correction_bias [288]` FP32,
    `norm_topk_prob`, `routed_scaling_factor 2.5`, roteador em FP32, sem limitação por grupo.

34. **Primeiras camadas densas.** Muitas arquiteturas MoE mantêm as primeiras camadas com MLP densa para
    estabilidade e representação inicial rica. → `first_k_dense_replace: 3` (MLP 12288 nas camadas 0–2).

35. **Hyper-Connections e mHC.** Hyper-Connections alargam o residual para n fluxos e aprendem, por token,
    como ler deles (`H_pre`), escrever neles (`H_post`) e misturá-los (`H_res`). Sem restrição isso
    desestabiliza sinais profundos (ganhos de ~3000); mHC projeta `H_res` no **politopo de Birkhoff**
    (matrizes duplamente estocásticas) via Sinkhorn-Knopp, preservando a "identidade" média do residual. →
    `hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_*_fn [24,16384]`; `pre = σ(·)+eps`, `post = 2·σ(·)`,
    `comb = Sinkhorn(softmax(·))`; entrada da camada 0 por `hc_expand` (réplica), saída final por `hc_contract`
    (média). Para o DFlash2 os taps precisam materializar o `hc_post` adiado (§3.4).

36. **MTP — Multi-Token Prediction.** Cabeça extra treinada para prever o token t+2 a partir do estado
    oculto em t e do embedding do token t+1: `eh_proj([enorm(emb(x_{t+1})); hnorm(h_t)])` → uma camada de
    decoder → norma → `lm_head`. Serve como speculator "embutido". → camada 45 (DSA + MoE, sem mHC);
    `SPEC_METHOD=mtp`, `MTP_TOKENS=2` aplicando a mesma camada 2 vezes; `index_share_for_mtp_iteration`.

37. **Parâmetros totais vs ativos.** Total = tudo que precisa estar na memória; ativos = o que é usado por
    token (depende do top-k). → 321,3 B totais / ~17,4–17,9 B ativos (§4.1). Memória é ditada pelo total;
    latência de decode pelos bytes ativos lidos (mais experts distintos quando se verificam vários tokens).

38. **LLM multimodal (ViT + merger).** Imagens viram patches (14×14, e 2 frames no tempo), um ViT os
    processa, patches vizinhos 2×2 são fundidos e projetados para a dimensão do texto; os "tokens de
    imagem" substituem placeholders no prompt. → Torre de 24 blocos (1024), RoPE 2D parcial, merger SwiGLU
    →4096; `max_image_tokens` 8000 → a receita limita a 2048; vídeo a 2 fps.

### E. Quantização

39. **Quantização: bits, escala, grupo/bloco, erro.** Representar pesos com poucos bits: `w ≈ s·q` com
    código q (inteiro ou float curto) e escala s compartilhada por um grupo; menor grupo = mais precisão e
    mais overhead. Pode ser só de pesos (W4A16, W8A16) ou de pesos e ativações (W8A8). → FP8 oficial usa
    blocos 128×128; o FP8 da receita é por canal de saída (W8A16 Marlin); EXL3 é 4 bits por peso com
    codebook.

40. **FP8 weight-only via Marlin vs W8A8.** Marlin desquantiza pesos FP8/INT4 dentro do kernel e multiplica
    com ativações BF16 — rápido em decode (poucas linhas) sem mexer em ativações. W8A8 quantiza também as
    ativações (mais rápido em compute) mas introduz erro dependente do dado. → `GLM53_DENSE_FP8`; o W8A8
    foi rejeitado (flips de probabilidade) em favor da cópia BF16 para M grande.

41. **Quantização trellis/codebook (QTIP → EXL3) e incoerência.** EXL3 (ExLlamaV3, inspirado no QTIP)
    codifica blocos de pesos como caminhos numa treliça ("trellis coded quantization") decodificados por um
    gerador pseudo-aleatório/codebook (aqui "MCG", multiplicador `0xCBAC1FED`); antes, a matriz é
    "incoerentizada" por rotações de Hadamard com sinais aleatórios nas entradas/saídas (`suh`, `svh`), o que
    deixa os pesos quase gaussianos e fáceis de quantizar. Tiles 16×16, K bits/peso. → Experts `trellis I16
    [in/16, out/16, 16·K]` com K=4; "uniform-K4" = mesmo K em todos os experts; não-roteados nativos.

42. **Calibração e métricas de fidelidade.** Quantizações boas usam dados de calibração (Hessianas/proxy de
    importância) e são validadas por KLD, top-1 agreement e tarefas. → ShapleyMCG (pipeline auditável de
    calibração→codificação); painel de 25 janelas seladas; gate "mean KLD < 0.06".

43. **Quantização do KV cache e layouts empacotados.** O KV pode ser guardado em FP8 (ou NVFP4) com escalas
    por grupo; kernels esperam layouts exatos. → `fp8_ds_mla` 656 B (512 FP8 + 4 escalas FP32 + 64 BF16
    RoPE); indexador 132 B por entrada; NVFP4 KV não existe para sparse-MLA no SM12x.

### F. Motor de inferência (vLLM)

44. **PagedAttention: blocos, páginas, block table, slot mapping.** O KV é alocado em blocos de tamanho fixo
    (páginas) e uma tabela por request mapeia posições lógicas → blocos físicos; evita fragmentação. →
    blocos de 3584 tokens no gerenciador MLA, kernel page 64; `slot = block_table[req, pos//B]·B + pos%B`;
    bug do kpool tail sem clamp (§3.8).

45. **KV cache híbrido (vLLM V1): grupos, página uniforme, "Mamba" align.** Modelos com vários tipos de
    camada têm vários "KV cache groups" (atenção cheia, janela deslizante, estado Mamba/linear) num único
    pool com ids de bloco globais; as páginas são igualadas (padding) para compartilhar tensores. → 7 grupos:
    MLA(+indexador), KpoolTail, 4 × Mamba (34 KDA, round-robin ≤ 11 por grupo), SWA do drafter; bloco 3584
    escolhido para a página MLA conter um estado KDA (§4.3).

46. **Prefix caching (APC) em modelos híbridos.** Reusar KV de prefixos repetidos via hash de blocos; em
    modelos recorrentes só se pode reusar onde existe um checkpoint do estado exatamente naquela posição. →
    hits alinhados a 3584 tokens; `patch_hybrid_prefix_hit.py`; o tail do kpool não é cacheável.

47. **Continuous batching, chunked prefill e MNBT.** O scheduler mistura prefills e decodes num mesmo passo
    limitado por `max_num_batched_tokens`; prompts longos são fatiados. → MNBT 7168 (8192 estoura smem do
    `persistent_topk` do indexador), `MAX_NUM_SEQS=4`, política `GLM53_MIXED_PREFILL_CHUNK=fair`.

48. **CUDA graphs e tamanhos de captura.** Grava a sequência de kernels de um passo de decode para
    reexecutar sem overhead de CPU; só funciona para formas pré-capturadas. → DFlash2 `1 2 4 8 16 24 32`
    (múltiplos de 8 = k+1); MTP `1 2 3 4 6 8 12`; adaptive-k adiciona 3 e 5; KDA tem ponto de quebra
    "eager".

49. **Backends de atenção.** Implementações de kernel específicas por hardware/forma (FlashAttention,
    FlashInfer, Triton, FlashMLA…). → Alvo: `FLASHINFER_MLA_SPARSE_SM120` (única MLA esparsa no SM12x);
    draft: `FLASH_ATTN` (suporta não-causal + SWA no SM121).

### G. Decodificação especulativa

50. **Speculative decoding sem perda.** Um modelo barato propõe k tokens; o alvo verifica os k+1 em um
    forward; aceita-se o maior prefixo compatível e ganha-se um token "bônus" do alvo. Com amostragem por
    rejeição (aceita com prob `min(1, p/q)`, senão reamostra de `max(0,p−q)`), a distribuição de saída é
    **exatamente** a do alvo. → `rejection_sample_method standard`, `draft_sample_method probabilistic`;
    métrica: aceitação média e tokens/passo (6,71 structured, 2,39 prose).

51. **Estados ocultos do alvo como features (EAGLE-3).** Drafters modernos não leem só tokens: consomem
    estados ocultos de várias camadas do alvo (baixa, média, alta) concatenados. → DFlash2 lê as camadas
    5, 14, 24, 33, 42 via interface EAGLE-3 (`SupportsEagle3`), `fc [4096,20480]`.

52. **DFlash — drafter de difusão em bloco.** O draft recebe `[âncora, MASK×(B−1)]` e prevê todas as
    posições num único forward (denoising paralelo, atenção bidirecional no bloco), condicionado ao alvo por
    **KV injection**: os estados do alvo são projetados para K/V de **toda** camada do draft (as queries vêm
    só do bloco). Rascunhar custa 1 forward em vez de k. → `block_size 8`, `mask_token_id 154856`,
    `precompute_and_store_context_kv` (RMSNorm → GEMM K/V fundido de 5 camadas → K-norm → RoPE → cache),
    janela SWA 2048.

53. **DFlash2 — candidatos top-16, seletor e conv de 2 taps.** Para cada posição guarda 16 candidatos; um
    seletor bilinear de baixo posto pontua transições (candidato anterior → candidato atual, sob um "gate"
    de contexto) e uma caminhada gulosa/Gumbel escolhe um caminho coerente; convoluções dinâmicas de 2 taps
    antes/depois de atenção e FFN reduzem o "suffix decay" (atenção dedica pouca capacidade às dependências
    dentro do bloco: 30 % na camada 1 → 8 % na 5, segundo o blog). → `selector_top_k 16`, `selector_rank 256`,
    `conv_kernel_size 2`, `conv_group_size 16`; `_selector_walk_kernel`.

54. **Truque Gumbel-max.** `argmax(logit + Gumbel)` amostra exatamente de `softmax(logit)`; permite amostrar
    dentro de um kernel com um argmax. → usado na caminhada do seletor quando T > 0 (`gumbel_noised_argmax`).

55. **Comprimento de verificação adaptativo.** Verificar menos posições quando a aceitação esperada é baixa
    economiza compute do alvo sem mudar a distribuição. → `GLM53_ADAPTIVE_K=ema`, conjunto {2,4,7},
    uniforme no batch por causa dos grafos FULL.

56. **Rollback de estado recorrente na verificação.** Com camadas recorrentes, verificar k tokens avança o
    estado k passos; se só j forem aceitos, é preciso o estado no passo j. → KDA guarda `num_spec+1` estados
    por request e usa `num_accepted_tokens`; conv state com `kernel−1+num_spec` colunas; `num_speculative_blocks`
    no modo align.

### H. Inferência distribuída

57. **Tensor parallelism (column/row, all-reduce).** Divide cada matriz entre GPUs: column-parallel divide
    saídas (sem comunicação), row-parallel divide entradas e exige all-reduce da soma. Cabeças de atenção são
    divididas entre ranks. → TP2: 32 cabeças MLA/KDA por rank; experts `gate/up` coluna, `down` linha, 1
    all-reduce por camada MoE sobre CX7 RoCE (~15–25 ms/passo de prefill estimado no repo).

58. **Divisibilidade e padding de cabeças; expert parallelism.** TP exige que dimensões divididas sejam
    múltiplas do nº de ranks; quando não são, padar com cabeças "falsas" (pesos zero) ou distribuir experts
    inteiros (EP). → TP3: 64→66 cabeças, vocab 154.944, EP 96 experts/rank, draft em 1 rank; "não padar para
    96". MLA replica o latente por rank (por isso existe DCP em outras stacks).

59. **TP do drafter.** O draft pode ter TP próprio; TP maior divide compute mas exige comunicação a cada
    passo de rascunho. → `DFLASH_DRAFT_TP=2` (medido melhor em C4) vs 1.

60. **Memória unificada (UMA) e orçamento.** No GB10 CPU e GPU compartilham 128 GB LPDDR5x; tudo que o host
    usa (page cache, processos, cache multimodal) compete com pesos e KV. → `--kv-cache-memory-bytes`
    explícito (11 GiB), `GPU_MEM_UTIL 0.85`, `MM_PROCESSOR_CACHE_GB 1`, `SKIP_MM_PROFILING`.

### I. Medição

61. **TTFT, tok/s por stream vs agregado, prefill tok/s.** TTFT = tempo até o 1º token (≈ prefill); stream
    tok/s = velocidade de um request; agregado = soma com concorrência. → tabelas §4.5–4.6.

62. **Roofline / intensidade aritmética.** Um kernel é limitado por memória se FLOPs/byte < pico_FLOPs/banda.
    → MoE em chunks de 1024 tokens: ~112 FLOP/B < 458 FLOP/B de "ridge" BF16 do GB10 → memory-bound; decode
    ainda mais.

63. **Aceitação por posição.** Probabilidade de a posição i do rascunho ser aceita (cumulativa); decai com
    i. → Structured 0,98…0,83 vs prose 0,75…0,06.

---

## 6. Receita passo-a-passo (do ponto de vista da arquitetura)

A lógica da Mia pode ser reconstruída como uma cadeia de deduções a partir do `config.json`. Cada passo
abaixo diz **o que fazer**, **qual propriedade da arquitetura obriga a isso** e **onde está no repo**.

### Passo 1 — Ler o `config.json` e extrair as restrições

Checklist (valores deste modelo):

| pergunta | resposta aqui | consequência |
|---|---|---|
| Quantos parâmetros e onde estão? | 321,3 B, 97 % nos experts roteados | quantizar **só experts** resolve memória; manter o resto nativo preserva qualidade |
| Que tipos de mixer existem? | KDA (34) + MLA-DSA (11) | o motor precisa de: kernel de atenção linear (chunk + recorrente, com rollback especulativo), kernel sparse-MLA, gerenciador de KV híbrido |
| A MLA tem RoPE? | não (`qk_rope_head_dim 0`) | kernels que exigem `pe_dim=64` precisam de pad de zeros (exato) |
| O indexador comprime? | sim, `index_kpool 4`, top-k 2048, cauda | largura de buffer 2051 ≠ template 2048 → cortar 1 pool; blocos múltiplos de `4×64` no SM12x |
| Há multi-stream residual? | mHC n=4 | qualquer "tap" de estados ocultos (EAGLE-3/DFlash) precisa materializar `hc_post` e contrair |
| Divisibilidade para TP | 64 cabeças, 288 experts, 2048 intermediário, vocab 154.880 | TP2/TP4 ok; TP3 exige padding de cabeças + EP |
| MTP embutido? | 1 camada (45) | speculator de fallback sem checkpoint extra |
| Contexto | 1.048.576 | KV cresce só em 11 camadas (≈7,6 KB/token/rank) → 850k–1M cabem |

### Passo 2 — Orçar memória e escolher o formato dos pesos

* 2 nós × 128 GB UMA ≈ 256 GB brutos, menos SO/page cache/CUDA → ~200–215 GB úteis para pesos + KV
  `[inferência a partir dos pools medidos]`.
* BF16 (642,6 GB) e FP8 oficial (328,3 GB) **não cabem**. NVFP4 (~180 GB) cabe, mas KLD 0,0605. EXL3/TR3
  K4 (175,6 GB) cabe com KLD 0,0246 (≈ FP8 oficial) → escolha da receita.
* Por que "uniform-K4 só nos experts": experts são 97 % dos bytes e, por serem esparsamente ativados, cada
  um vê menos "tráfego" de gradiente/uso — a perda de 4 bits dilui-se; atenção KDA/MLA, indexador,
  embeddings, roteador e mHC são pequenos e sensíveis (o próprio FP8 oficial os deixa em BF16).
* Resultado por rank (TP2): ~80 GiB de pesos → sobra para KV (pool 11–14 GiB), grafos CUDA, workspace do
  indexador, estado KDA e o drafter (1,17 GB/rank com draft TP2).

### Passo 3 — Montar a imagem (vLLM com o modelo + overlay SM121)

1. Partir de uma build do vLLM que já contém `vllm/models/glm5next/` (aqui
   `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…`, vLLM `487ecf187`, CUDA 13, arm64).
2. Aplicar o overlay NoPE-sparse-MLA (§3.1): pad de 64 zeros no RoPE, prefill esparso, top-k como
   capacidade, buffer 2048 com 511 pools + cauda, kernel page 64, alinhamento do indexador a 256, sem
   autotune FlashInfer, PDL só em SM9x/10x. *Raciocínio*: é a forma mínima de fazer um modelo NoPE usar o
   único kernel sparse-MLA do SM12x sem escrever kernel novo, e é matematicamente exato.
3. Registrar `exl3` e compilar ExLlamaV3 `c5d9c657` para `sm_121a` (`TORCH_CUDA_ARCH_LIST=12.1a`) com os
   kernels "fat" (E2/E3) e o overlay `overlay/exl3.py` (experts permanecem empacotados; TP col/row).
4. Instalar DFlash2 (`patch_dflash2.py`, `qwen3_dflash2.py`, `dflash2_speculator.py`) e a interface EAGLE-3
   do alvo através do mHC (`patch_glm_eagle3.py`).
5. Instalar os patches do alocador/APC híbrido: grupo do drafter (`patch_glm5_drafter_group.py`), prefix hit
   híbrido, retenção por grupo, no-store, correções de estado Mamba/KDA, clamp e stride do kpool tail,
   workspace do indexador, log de capacidade. *Raciocínio*: cada tipo de camada gera um grupo de KV com
   regras próprias; sem essas correções o drafter inflaciona ids, zera hits ou corrompe o indexador.
6. (Opcional) loader InstantTensor + correções de cold-load em UMA (`patch_cold_load_uma.py`).

Build: `docker build -t glm53-flash-sm121:local .` (Dockerfile:85–86) — ou a imagem publicada
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`.

### Passo 4 — Baixar e fixar os pesos

* Alvo: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ `25a44fdb…` (espelho byte-idêntico de
  `brandonmusic/…@5ab363a8`), fallback `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (`start.sh:116–122`).
* Draft: `incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c…` (o `main` mudou pesos; o config é igual).
  Licença CC BY-NC-ND 4.0.
* Validar: `hidden_size 4096`, `num_hidden_layers 45` no alvo; draft `DFlash2DraftModel`, 5 camadas,
  4096 (`tools/pack_profile.py:79, :120`). Ambos os ranks precisam dos mesmos shards (rsync ou `NFS_SHARE=1`).

### Passo 5 — Montar os argumentos de lançamento (e por que cada um)

```text
vllm serve <MODEL_DIR>
  --tensor-parallel-size 2 --nnodes 2 --node-rank {0|1} --distributed-executor-backend mp
  --quantization exl3                      # experts trellis/MCG 4 bpw; resto nativo
  --kv-cache-dtype fp8                     # → fp8_ds_mla 656 B (único formato do sparse-MLA SM12x)
  --max-model-len 850000                   # KV cresce só em 11 camadas; E3 reserva 560 MiB
  --max-num-batched-tokens 7168            # 8192 estoura smem do top-k do indexador
  --max-num-seqs 4
  --enable-prefix-caching                  # hits alinhados a 3584 tokens (bloco híbrido)
  --no-enable-flashinfer-autotune          # autotune derruba o rank 0 no SM121
  --speculative-config '{"method":"dflash","model":"<DFLASH_DIR>","num_speculative_tokens":7,
                         "kv_cache_dtype":"auto","draft_sample_method":"probabilistic",
                         "rejection_sample_method":"standard","draft_tensor_parallel_size":2}'
  --cudagraph-capture-sizes 1 2 4 8 16 24 32   # múltiplos de 8 = âncora + 7
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --chat-template /opt/glm53/chat_template.jinja
  --limit-mm-per-prompt '{"image":48,"video":1}' --mm-processor-kwargs '{"max_image_tokens":2048}'
  --mm-processor-cache-gb 1 --skip-mm-profiling
  --kv-cache-memory-bytes 11811160064      # pool explícito (11 GiB) com páginas compactas do draft
  --load-format instanttensor
```

(Montado em `start.sh:2094–2170` para o head e `:2173–2248` para o worker; `--headless` no worker.)
Variáveis que ativam patches em runtime: `GLM53_DENSE_FP8=all`, `GLM53_KDA_BF16_LARGE_M=1`,
`GLM53_DRAFT_KV_COMPACT=1`, `GLM53_INDEXER_WORKSPACE=rightsize`, `EXL3_FAT_GROUPED=1`,
`EXL3_TEMP_ROWS_FUSED=32` (≥ `MAX_NUM_SEQS × (DFLASH_TOKENS+1)` para o decode caber num único lançamento).

**Não fazer** (derivado da arquitetura): `--moe-backend marlin` (é para NVFP4); fixar `TRITON_ATTN` no draft
(máscara causal no bloco); KV bf16/NVFP4 (sem kernel sparse no SM12x); padar cabeças para 96 em TP3; padar
`moe_intermediate_size` (trellis empacotado em 2048); reduzir o contexto para 256k "para liberar KV" (o custo
fixo híbrido — Mamba + janela do draft — não depende do comprimento).

### Passo 6 — Verificar (o que olhar e por quê)

1. **Boot**: linha `padded slot-share block=64 mla_page=2351104 … draft_bytes/token=2048` (ou exact-fit/compact);
   linhas `[glm53-kv-capacity-log] group N: …` (MLA 11 camadas bloco 3584; KpoolTail; 4× Mamba; SWA 2048);
   taps `(6, 15, 25, 34, 43)`; `dense fp8 groups: all`; `kda bf16-large-m retained` (34 por rank);
   `[glm53-kpool-tail-seed-stride]`; `glm53 NoPE sparse-MLA overlay verify OK` no build.
2. **Aceitação**: `tests/bench_decode.py --phase structured --structured` deve dar ~0,96 de aceitação
   (6,7/step) e ~63–65 tok/s; se a aceitação das posições 2..7 cair para ~0, a máscara do draft ficou causal.
3. **Qualidade**: painel KLD vs teacher BF16 (alvo ~0,025 nats); `tests/bench_numerical_calibrated.py`
   (com cuidado: o próprio screen marcou controles stock); needle em contexto longo (mostra se o indexador/
   kpool está selecionando os pools certos — o Hadamard na query e os pesos FP32 existem por isso).
4. **Memória**: `MemAvailable` no head sob carga ≥ ~5–6 GiB; pool suficiente para 1 request de
   `MAX_MODEL_LEN` (o boot recusa se não couber).

### Passo 7 — Ajustar

| alavanca | efeito arquitetural | medido |
|---|---|---|
| `GLM53_DENSE_FP8` (`dense,kda` → `all`) | menos bytes de pesos densos por passo de decode | −11 ms/step; `all` −4,7 % ms/ciclo |
| `GLM53_KDA_BF16_LARGE_M=1` | GEMM BF16 cuBLAS para o `in_proj` KDA em prefill (M > 512) | −12 % TTFT; +3,3 GiB/rank |
| `GLM53_ADAPTIVE_K=ema` | verifica 2/4/7 conforme aceitação | prosa +10–21 % |
| `DFLASH_DRAFT_TP` 1 vs 2 | compute do draft vs comunicação por passo | 2 venceu em C4 (2026-08-30) |
| `MAX_NUM_BATCHED_TOKENS` | mais linhas por expert no prefill (menos memory-bound) vs smem do indexador | 7168 mantido; 8192 inseguro |
| `--kv-cache-memory-bytes` | tokens no pool vs folga de host (UMA) | 11 GiB = 1,23 M tokens (compact) |
| `EXL3_FAT_GROUPED` | E3 (3 lançamentos por camada MoE no prefill) vs E2 | +37–45 % prefill; −560 MiB de KV |
| `GLM53_MODEL_PRESET=dense-h3` | EXL3 também nos densos + draft 6 bpw | opt-in, sem A/B |

### Passo 8 — Escalar para TP3/TP4 (o que muda por causa da arquitetura)

* **TP3**: `--hf-overrides` 64→66 cabeças (KDA e MLA), `A_log` 64→66, vocab pad 192, shared 2048→2112,
  `--enable-expert-parallel` (96 experts/rank), visão DP, draft TP1 + GQA 36/9; bloco MLA vira 2560 (estado
  KDA menor por rank); FP8 denso sem `f_b_proj/g_b_proj`. 40,1 tok/s prose ×1.
* **TP4**: cabeças 16/rank; mitigação de stall no decode sparse-MLA (`patch_sparse_mla_slice.py`); não testado
  pela Mia.

### Passo 9 — Como generalizar o raciocínio para outro modelo

1. Some parâmetros por tipo de tensor → decida o que quantizar (o grosso é quase sempre experts).
2. Para cada tipo de camada, pergunte: qual kernel existe no meu hardware? que layout de KV ele exige? posso
   transformar meu modelo exatamente nesse layout (pad de zeros, fusão, reshape)?
3. Para cada tipo de estado (KV cheio, KV comprimido, janela, estado recorrente, scratch), calcule bytes por
   token e por request e como o alocador vai paginar (o bloco precisa conter o maior estado).
4. Para speculators: de onde vêm as features (quais camadas, como materializar através de resíduos
   não-padrão), que máscara o draft precisa, que dtype de KV e backend suportam isso.
5. Para paralelismo: liste todas as dimensões divididas (cabeças, intermediários, vocab, experts) e cheque
   divisibilidade pelo nº de ranks; use EP quando padar quebra o formato empacotado.

---

## Apêndice A — Fluxo de tensores por bloco (equações + formas), para animar

Formas em TP1 (modelo inteiro); entre colchetes, **por rank em TP2**. `x ∈ R^4096` é a entrada normalizada
de uma sub-camada para um token.

### A.1 Bloco KDA (34 camadas) — `vkda:320–371` e `_forward`

1. Projeção fundida: `[q | k | v | β̃ | f_a | g_a] = W_in · x`, `W_in ∈ R^{24896×4096}`
   (3·8192 + 64 + 128 + 128) [`12576×4096` = 3·4096 + 32 + 128 + 128; f_a/g_a replicados].
2. Conv curta causal por canal (kernel 4) + SiLU em q, k, v (uma única `causal_conv1d` sobre o concat).
3. Por cabeça h (64 [32]) com d=128: `q_h, k_h ← L2norm`, `β_h = σ(β̃_h)`.
4. Gate de esquecimento por canal: `g = −5 · σ( exp(A_log_h) · (f_b(f_a) + dt_bias) )` ∈ (−5, 0)^{64×128};
   `α = e^g` ∈ (0,0067; 1).
5. Atualização do estado (matriz 128×128 por cabeça, FP32):
   `S_h ← (I − β_h k_h k_hᵀ) · Diag(α_h) · S_h + β_h k_h v_hᵀ`.
6. Leitura: `o_h = S_hᵀ q_h ∈ R^128`.
7. Saída gated: `o_h ← RMSNorm(o_h) ⊙ σ(g_b(g_a))_h` (`FusedRMSNormGated`, activation sigmoid).
8. `y = W_o · concat_h(o_h)`, `W_o ∈ R^{4096×8192}` [row-parallel: `4096×4096` + all-reduce].

Prefill: passos 5–6 feitos em chunks (forma matricial dentro do chunk). Decode especulativo: o kernel guarda
os estados após cada um dos k+1 tokens e `num_accepted_tokens` escolhe qual vira o estado oficial.

### A.2 Bloco MLA NoPE + DSA (11 camadas + MTP) — `vattn:206–625` + backend SM120

1. `[c_q | c_kv] = W_{qkv_a} · x` → `c_q ∈ R^1536`, `c_kv ∈ R^512` (replicado em cada rank);
   `c_q ← RMSNorm(c_q)`, `c_kv ← RMSNorm(c_kv)`.
2. **Cache**: grava `c_kv` em FP8 (512 B) + 4 escalas FP32 (16 B) + 64 zeros BF16 (128 B) = 656 B.
3. Query por cabeça: `q = W_qb · c_q ∈ R^{64×256}` [32×256].
4. **Indexador**:
   * `q_I = W_{wq_b} · c_q ∈ R^{32×128}` (replicado); `k_I = LayerNorm(W_wk · x) ∈ R^128` (1 por token);
     `w = W_wp · x ∈ R^32` em FP32; `gate = W_gate · x ∈ R^128`.
   * Pool a cada 4 tokens (slots j=0..3), por canal c:
     `k_pool[c] = Σ_j softmax_j(gate_j[c] + ape_j[c]) · k_I,j[c]` → Hadamard-128 → FP8 (absmax, escala 2^n).
   * `q_I ← Hadamard(q_I)` → FP8; score do pool p: `I_p = Σ_h w_h · ReLU(q_I,h · k_pool,p)` (semântica do
     lightning indexer DSA como implementada nos kernels DeepGEMM `fp8_mqa_logits`/`fp8_fp4_paged_mqa_logits`;
     a escala `head_dim^-0.5 · n_head^-0.5` e a escala FP8 de q são embutidas em w, `vattn:383–385`).
   * top-512 pools (**511** na receita) → expandir ×4 → 2044 tokens + cauda (≤ 3 tokens mais recentes).
5. Absorção (decode/sparse MQA): `q̃_h = W_UK,hᵀ q_h ∈ R^512` (W_UK = parte K de `kv_b_proj`, 256×512 por
   cabeça); pad para 576 com zeros.
6. Atenção só sobre os tokens selecionados S: `a_h,s = softmax_s(q̃_h · c_kv,s / 16)`;
   `ō_h = Σ_s a_h,s c_kv,s ∈ R^512` (d_v = 512 no espaço latente).
7. `o_h = W_UV,h · ō_h ∈ R^256`; `y = W_o · concat_h(o_h)`, `W_o ∈ R^{4096×16384}` [row-parallel].

(A escala 1/16 = `qk_head_dim^-0.5` com qk_head_dim=256; o produto absorvido `q̃·c_kv` é idêntico a `q·k`.)

### A.3 MoE (42 camadas + MTP) — `vmodel:151–275`

1. `ℓ = W_gate · x ∈ R^288` (FP32); `s = σ(ℓ)`.
2. `E = top8(s + b)` (b = `e_score_correction_bias`, só para escolher).
3. `g_e = s_e / Σ_{e'∈E} s_e'`; `g_e ← 2,5 · g_e`.
4. `y = Shared(x) + Σ_{e∈E} g_e · W_d,e · ( silu(clamp(W_g,e x, ≤10)) ⊙ clamp(W_u,e x, ±10) )`.
   Formas por expert: `W_g, W_u ∈ R^{2048×4096}` [`1024×4096`], `W_d ∈ R^{4096×2048}` [`4096×1024`];
   no checkpoint EXL3: trellis `[256,128,64]`/`[128,256,64]` int16 por matriz.

### A.4 mHC (45 camadas) — `kernels/mhc/torch.py` (referência) e `vmodel:441–511`

Estado residual `X ∈ R^{4×4096}`; `x̂ = flatten(X) ∈ R^16384`.
1. `m = (W_fn · x̂) / RMS(x̂)` com `W_fn ∈ R^{24×16384}` (`hc_attn_fn` ou `hc_ffn_fn`).
2. `pre = σ(α₀·m[0:4] + b[0:4]) + ε` (4 pesos de leitura).
3. `post = 2 · σ(α₁·m[4:8] + b[4:8])` (4 pesos de escrita).
4. `comb = Sinkhorn₂₀( softmax_linha(α₂·m[8:24].view(4,4) + b[8:24]) )` (4×4 duplamente estocástica;
   normalização de colunas e linhas alternadas, ε=1e-6). α = `hc_*_scale [3]`, b = `hc_*_base [24]`.
5. Entrada da sub-camada: `u = Σ_i pre_i · X_i` → RMSNorm → sub-camada F → `y`.
6. Atualização: `X'_j = Σ_i comb_{ij} · X_i + post_j · y` (einsum `"ij,ih->jh"`).
7. Camada 0: `X = [x, x, x, x]` (`hc_expand`); após a camada 44: saída `= média_i X_i` (`hc_contract`).

### A.5 MTP (camada 45) — `mtp.py`

`z = eh_proj([enorm(emb(x_{t+1})) ; hnorm(h_t)])` (`[4096×8192]`) → decoder MLA-DSA + MoE (residual simples)
→ `shared_head.norm` → `lm_head` → previsão de x_{t+2}. Com `num_speculative_tokens=2` a mesma camada roda 2
vezes, reciclando o estado pós-norma; o top-k do indexador do passo 0 é reutilizado no passo 1.

### A.6 DFlash2 (draft) + verificação — `overlay/qwen3_dflash2.py`, `overlay/dflash2_speculator.py`, `qwen3_dflash.py`

**Contexto (para cada posição verificada p)**: `h_p = RMSNorm( W_fc · [h^{(5)}_p ; h^{(14)}_p ; h^{(24)}_p ;
h^{(33)}_p ; h^{(42)}_p] )`, `W_fc ∈ R^{4096×20480}`; para cada camada ℓ do draft:
`K_ℓ,p = RoPE_p(k_norm(W_k,ℓ h_p))`, `V_ℓ,p = W_v,ℓ h_p` (um GEMM fundido para as 5 camadas) → cache SWA
(janela 2048). Esse é o "KV injection".

**Bloco de query**: tokens `[a, M, M, M, M, M, M, M]` (a = último token aceito/bônus; M = 154856) nas
posições p₀…p₀+7 → embedding do alvo. Em cada uma das 5 camadas:
`h ← conv_pre(RMSNorm(h))` → atenção (Q do bloco; K/V = janela de contexto + o próprio bloco;
**bidirecional dentro do bloco**) → `conv_post` → residual → `conv_pre` → MLP SwiGLU 12288 → `conv_post`.
Conv dinâmica de 2 taps: `out_t = (b₀ + δ₀(h_t))·h_t + (b₁ + δ₁(h_t))·h_{t−1}` (só dentro do bloco:
`position % block_size ≥ tap`; δ por grupo de 16 canais, vindo de `kernel_projection`). `prepare()` calcula
os coeficientes dos **dois** lados a partir do estado *antes* da sub-camada: usa o lado 0 na conv de entrada
e guarda o lado 1 para a conv de saída (`finish()`), que é aplicada à saída da atenção/MLP.

**Candidatos e seletor**: `logits_i = lm_head(norm(h_i))` para as 7 posições → top-16 `(c_{i,1..16}, u_{i,·})`;
`score_i(p→c) = u_{i,c} + ⟨pred[c_{i−1,p}] ⊙ W_h h_i , succ[c_{i,c}]⟩` (rank 256; c_{0,·} = âncora a);
caminhada: `idx_i = argmax_c score_i(idx_{i−1}→c)` (ou Gumbel-max a T>0) → 7 tokens de rascunho; os 16
scores realizados viram a distribuição q do draft.

**Verificação**: o alvo roda `[a, d₁…d₇]` (8 tokens) num forward; amostragem por rejeição posição a
posição; aceita o prefixo mais longo e emite 1 token bônus do alvo → 1 a 8 tokens por passo.

---

## Apêndice B — Mapa "nome de tensor → conceito" (para rótulos de figuras)

| prefixo do tensor | conceito |
|---|---|
| `model.language_model.embed_tokens` | embedding de tokens |
| `model.language_model.layers.N.hc_{attn,ffn}_{fn,base,scale}` | mHC (pre/post/comb) |
| `…input_layernorm`, `…post_attention_layernorm`, `model.language_model.norm` | RMSNorm pré-sub-camada / final |
| `…self_attn.{q,k,v}_proj`, `…{q,k,v}_conv1d` | KDA: projeções e conv curta |
| `…self_attn.b_proj` | KDA: β (força da regra delta) |
| `…self_attn.{f_a,f_b}_proj`, `A_log`, `dt_bias` | KDA: gate de esquecimento por canal |
| `…self_attn.{g_a,g_b}_proj`, `o_norm` | KDA: gate de saída + norma |
| `…self_attn.q_a_proj`, `q_a_layernorm`, `q_b_proj` | MLA: query de baixo posto |
| `…self_attn.kv_a_proj_with_mqa`, `kv_a_layernorm` | MLA: latente KV (o que vai para o cache) |
| `…self_attn.kv_b_proj` | MLA: up-projection K/V (absorvida em decode) |
| `…self_attn.o_proj` | saída de atenção (escreve no residual; alvo da abliteração) |
| `…self_attn.indexer.{wq_b,wk,k_norm,weights_proj}` | DSA: lightning indexer |
| `…self_attn.indexer.index_kpool_compress_{gate,ape}` | kpool: pooling aprendido de chaves |
| `…mlp.{gate,up,down}_proj` (camadas 0–2) | MLP denso SwiGLU |
| `…mlp.gate.weight`, `…mlp.gate.e_score_correction_bias` | roteador MoE (sigmoid + viés aux-free) |
| `…mlp.shared_experts.*` | expert compartilhado |
| `…mlp.experts.E.{gate,up,down}_proj.{trellis,suh,svh,mcg}` | expert roteado EXL3 4 bpw |
| `layers.45.{eh_proj,enorm,hnorm,shared_head.norm}` | MTP |
| `lm_head` | cabeça de saída (untied) |
| `model.visual.*` | torre de visão (ViT + merger) |
| DFlash2 `fc`, `hidden_norm` | fusão das 5 features do alvo |
| DFlash2 `layers.i.{attention_conv,mlp_conv}.{base_kernel,kernel_projection}` | conv dinâmica 2-tap |
| DFlash2 `candidate_selector.{predecessor_codebook,successor_codebook,hidden_projection}` | seletor bilinear |

---

## Apêndice C — Links das fontes

* Repo: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (local, commit `94ae731f`)
* Modelo: https://huggingface.co/zai-org/GLM-5.3-Flash (config: `/raw/main/config.json`)
* Draft: https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2 ; blog https://inco.ai/blog/dflash2/
* Quant: https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw ; https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw
* vLLM `487ecf187`: https://github.com/vllm-project/vllm/tree/487ecf187d3dfe74d2cf6119a92881dba403c219/vllm/models/glm5next
* DFlash: https://arxiv.org/abs/2602.06036 ; GLM-5: https://arxiv.org/abs/2602.15763 ; Kimi Linear: https://arxiv.org/abs/2510.26692 ; mHC: https://arxiv.org/abs/2512.24880
* Secundárias: https://huggingface.co/docs/transformers/main/model_doc/glm5_next ; https://sebastianraschka.com/blog/2026/glm-5-3-flash-architecture-notes.html ; https://github.com/SemiAnalysisAI/InferenceX/issues/2794 ; https://recipes.vllm.ai/zai-org/GLM-5.3-Flash

# Arquitetura do Qwen3.8-Flash-Next (base Qwen + `nvidia/Qwen3.8-Flash-Next-NVFP4`) — inventário exaustivo

> Escopo deste arquivo: **a arquitetura do modelo** (o que existe dentro dos pesos e por quê), a
> matemática de cada bloco, a contabilidade de parâmetros/memória, e **todas** as modificações da
> receita `Qwen3.8-Flash-Next-Dual-DGX-Sparks` que tocam arquitetura, checkpoint, quantização ou
> kernels do modelo. Parâmetros puramente operacionais (rede, NCCL, docker, NFS) aparecem apenas
> quando interagem com a arquitetura; o inventário operacional completo é assunto de outro arquivo.
>
> Convenções: `README.md:366` = arquivo do repo, linha 366. "(inferência)" = conclusão minha, derivada
> de números do repo/checkpoint mas não afirmada por eles. "(a verificar)" = ponto em aberto.
> Todos os valores de configuração foram lidos do `config.json` real baixado do Hugging Face em
> 2026-09-28; todas as formas de tensores foram lidas dos cabeçalhos safetensors reais (range requests).

---

## 0. Fontes consultadas

### 0.1 No repositório (somente leitura)
Raiz: `.../scratchpad/repos/Qwen3.8-Flash-Next-Dual-DGX-Sparks` (commit `2c86a1d`)

| Arquivo | O que revela sobre a arquitetura |
|---|---|
| `README.md` (1073 linhas) | contagem de camadas KV (`:366-381`), memória por GPU (`:345-364`), backends por componente (`:470-484`), PLE (`:550-562`), MXFP8 (`:564-598`), multimodal (`:600-621`), MTP (`:654-715`), vocabulário de rascunho (`:717-819`), YaRN (`:821-849`) |
| `docs/CLAUDE/fable5-1-report.md` | o que **executa** de fato no GB10 por componente (`:27-44`), composição do checkpoint por categoria (`:57-72`), modelo de bytes por passo de decode (`:74-99`), formas por GPU em TP=2 (`:188-194`), coletivas por passo (`:385-394`) |
| `docs/HANDOFF-single-spark.md` | comparação dos 3 checkpoints NVFP4 (`:113-142`), custo de KV derivado (`:151-153`), layout do PLE NVFP4 (80 B códigos + 10 B escalas por cabeça, `:32-39`) |
| `files/ple_layer_patched.py` (1669 linhas) | **fonte vLLM do módulo PLE/N-gram** (hash, primos, gate, conv dilatada, estados de cache, métodos FP8/NVFP4) |
| `files/patch_checkpoint_config.py` | como vLLM numera a camada MTP (`mtp.layers.48`) e os algoritmos de quantização das experts do MTP |
| `files/detect_ple_dtype.py` | onde cada checkpoint declara o dtype da tabela PLE |
| `files/patch_qsa_fp8_kv.py`, `files/patch_qsa_fp8_kv_v030.py` | estrutura dos kernels QSA (indexador MQA + GQA esparso split-K) e seu cache |
| `files/qsa_gb10/{qsa.diff,bench_qsa_kernels.py}` | formas de decode do QSA em TP2 (12 q-heads × 1 kv-head × 256, página 1600, largura top-k 2048+3, cache comprimido 1×128) |
| `files/overlay/*.diff` | estrutura do GatedResidual/HyperConnection no vLLM (projeção fundida down+inject, `disable_tp=True`), lm_head do MTP |
| `files/patch_modelopt_fp8_block_moe.py`, `files/patch_modelopt_mxfp8.py` | formatos de quantização por componente e restrições de forma dos kernels |
| `files/patch_mtp_draft_vocab.py`, `files/build_draft_vocab.py` | cabeça de saída do MTP, vocabulário 248.320 |
| `files/patch_ple_offload.py`, `files/build_ple_packed_table.py`, `files/ple_offload/*` | offload da tabela N-gram para host |
| `files/patch_block_drop.py`, `files/patch_determinism.py` | interação do cache híbrido/QSA com MTP e determinismo do top-k |
| `.env.sample`, `start.sh`, `CHANGELOG.md`, `docs/bench/rerun-20260916.md` | knobs e medições |
| `html/concepts.md`, `html/BRIEF.md` | **não** tratam do modelo: são uma lista de 100 conceitos de páginas HTML de teste ("100-html") usadas para avaliar geração de código do modelo. Irrelevantes para a arquitetura. |

### 0.2 Na web (baixado em 2026-09-28)
- `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4` — `config.json` (30.820 B), `hf_quant_config.json`, `generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer_config.json`, `model.safetensors.index.json` (299.545 tensores, `total_size` 132.639.846.394 B) e **os cabeçalhos dos 11 shards** (formas/dtypes/bytes de cada tensor). Card (release 08/31/2026).
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next` — card oficial (Model Overview) e `config.json` (idêntico ao da nvidia salvo `norm_topk_prob`, `number_of_conv_states`, `seed`, `dtype`, `transformers_version`).
- `https://github.com/QwenLM/Qwen3.8-Flash-Next` — README e **`tech_report.pdf`** ("On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability", Qwen Team, 2026-08-26, 28 páginas).
- `huggingface/transformers` `src/transformers/models/qwen4_exp/{modular,modeling,configuration}_qwen4_exp.py` — **implementação de referência** (a mais precisa para a matemática).
- Configs de linhagem: `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B`; configs alternativos `local-inference-lab/…NVFP4`, `RadixArk/…NVFP4`, `Qwen/…-FP8`.
- Engram (DeepSeek): "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs", arXiv 2601.07372 (citado no tech report como Cheng et al., 2026).

---

## 1. Visão geral

### 1.1 O que é o modelo, em uma frase por pilar
Qwen3.8-Flash-Next (`model_type: qwen4_exp`, arquitetura `Qwen4ExpForConditionalGeneration`) é
a **prévia aberta da arquitetura do Qwen4**: um LLM causal multimodal (texto+imagem+vídeo) do tipo
**Mixture-of-Experts esparso**, com **125B parâmetros no modelo principal, ~6B ativos por token,
+51B de tabela de embeddings N-gram e ~4B de MTP** (card oficial e card nvidia). Ele muda quatro
coisas em relação à família Qwen3.5 (tech report §1, README do GitHub):

1. **Atenção** — híbrido camada-a-camada: 3 camadas **Gated DeltaNet (GDN)** (atenção linear
   recorrente, estado de tamanho fixo) para cada 1 camada de **Qwen Sparse Attention (QSA)**
   (atenção softmax global, mas esparsa: um indexador leve comprimido escolhe ≤ 2048 tokens por
   query em micro-blocos de 4).
2. **Resíduo** — **Gated Residual (GR)**: o fluxo residual é alargado para **4 ramos** (4×2560 =
   10240 números por token); cada subcamada **lê** os ramos com um gate elementwise dependente dos
   dados (e esse gate substitui a pré-normalização) e **escreve** sua saída em todos os ramos com um
   escalar por ramo.
3. **Embedding** — **N-gram Embedding** (no código chamado **PLE**): na camada 2, bigramas e
   trigramas do contexto local são convertidos por hash em índices de uma tabela gigante
   (320 milhões de linhas × 160 dims = 51,2B parâmetros), e o vetor recuperado é injetado nos 4
   ramos residuais com um gate de contexto e uma convolução curta dilatada. Capacidade quase sem
   FLOPs, com endereçamento determinístico (prefetch/offload possível).
4. **Otimização** — Muon + AdamW com lei de escala refeita (não afeta inferência diretamente, mas
   explica a estabilidade e a ausência de truques como qk-clip).

Mais: **MoE de 512 experts, top-10 roteados + 1 compartilhado**, experts pequenos (intermediário
640), **MTP** de 1 camada treinado multi-passo (usado como rascunho de decodificação especulativa),
**encoder de visão** ViT de 27 blocos, **mRoPE** parcial (64 de 256 dims) com θ = 10⁷, contexto
nativo 262.144 extensível a 1.000.000 com YaRN.

### 1.2 Linhagem (por que a arquitetura é assim)
| Geração | Atenção | MoE | Resíduo | Extra |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B** (2025) | 3×GDN + 1×Gated Attention (48 camadas, hidden 2048; GDN 16 QK / 32 V heads ×128; atenção 16 Q / 2 KV × 256, `partial_rotary_factor` 0.25, θ 10⁷) | 512 experts, top-10 + 1 shared, intermediário 512 | pré-norm clássico | MTP; vocab 151.936 |
| **Qwen3.5** (ex.: 397B-A17B: 60 camadas, hidden 4096, GDN 64 V heads, atenção 32 Q/2 KV × 256, 512 experts top-10, interm. 1024; 35B-A3B: 40 camadas, 256 experts top-8) | mesmo híbrido GDN + Gated Attention (`attn_output_gate: true`), mRoPE `[11,11,10]` | idem | pré-norm | multimodal nativo (ViT 27×1152), vocab **248.320**, MTP 1 camada |
| **Qwen3.8-Flash-Next / Qwen4-Exp** (2026-08) | 3×GDN + 1×**QSA** (Gated Attention + indexador esparso) | 512 experts top-10 + 1 shared, interm. **640** | **Gated Residual 4 ramos** | **N-gram embedding 51B** na camada 2; MTP com QSA; gate sigmoid no GDN |

Evidência de herança no código: o `modular_qwen4_exp.py` importa `Qwen3_5GatedDeltaNet`,
`Qwen3_5Attention`, `Qwen3_5RMSNorm`, `Qwen3_5TextRotaryEmbedding`, `Qwen3_5MoeVisionModel` e
`Qwen3NextSparseMoeBlock/TopKRouter/Experts/RMSNormGated` (modular `:35-67`). O README do GitHub diz
explicitamente que o Qwen3.8-Flash-Next "plays the same role that Qwen3-Next played for Qwen3.5".

### 1.3 Pilha de camadas (layer-by-layer)
`layer_types` no config lista `full_attention` a cada 4 camadas (o transformers renomeia para
`indexed_attention` porque são QSA — modular `:188-192`). Índices **0-based** (os usados em nomes de
tensores `layers.N`):

| Índice (0-based) | Tipo de mixer de tokens | FFN | Extra |
|---|---|---|---|
| 0 | GDN | MoE | — |
| 1 | GDN | MoE | **N-gram embedding (PLE)** — `ple_layer_ids: [2]` é 1-based (modular `:91-92`, `:796`); tensor `model.language_model.layers.1.ple.*` |
| 2 | GDN | MoE | — |
| 3 | **QSA** | MoE | 1ª camada com KV cache |
| 4–6 | GDN | MoE | |
| 7 | QSA | MoE | |
| … padrão `(i+1) % 4 == 0` → QSA … | | | |
| 43 | QSA | MoE | |
| 44–46 | GDN | MoE | |
| 47 | QSA | MoE | última camada |
| `hyper_connection_mixer` | leitura final dos 4 ramos (GR sem escrita) | — | substitui o RMSNorm final |
| `lm_head` | 2560 → 248.320 (não amarrado ao embedding) | | |
| **MTP** (`mtp.layers.0`, instanciado no vLLM como `mtp.layers.48`) | QSA | MoE 512 experts (FP8 bloco 128×128) | `fc_embedding`, `fc_hidden`, `pre_fc_norm_*`, GR próprios, mixer próprio |

Totais: **36 GDN + 12 QSA = 48 camadas**, **+1 camada MTP (QSA)**. Camadas com KV cache por token:
12 + 1 = **13** (`README.md:366-369`). As QSA estão nos índices 3, 7, 11, 15, 19, 23, 27, 31, 35,
39, 43, 47 — exatamente as camadas cujo `self_attn.o_proj` o checkpoint `ABLIT=1` altera
(`README.md:250-252`).

Layout oficial (card): `12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`.

### 1.4 Fluxo de um token (forward completo, visão de pássaro)
```
input_ids ─► embed_tokens (248320×2560, BF16) ─► x0 ∈ R^2560
            (imagens/vídeo: ViT → merger → vetores 2560 substituem os tokens <|image_pad|>/<|video_pad|>)
x0 ─► repeat ×4  ─► R ∈ R^{4×2560}  (fluxo residual alargado, "hyper hidden" 10240)
para cada camada ℓ = 0..47:
   se ℓ == 1: R ← R + PLE(R, input_ids)            # N-gram embedding, escreve nos 4 ramos
   (x, R, s) ← GR_attn.read(R)                      # norm por ramo + gate elementwise → x ∈ R^2560
   y ← GDN(x)   ou   QSA(x)                         # mixer de tokens
   R ← R + s ⊗ y                                    # escrita: 1 escalar por ramo
   (x, R, s) ← GR_mlp.read(R)
   y ← MoE(x) = Σ_top10 w_e·Expert_e(x) + σ(g·x)·Shared(x)
   R ← R + s ⊗ y
h ← hyper_connection_mixer.read(R)  ∈ R^2560       # leitura final = norma final
logits ← lm_head(h) ∈ R^248320
MTP (só em decode especulativo): usa R/h do alvo + embedding do próximo token → propõe k tokens
```
(Fonte: `Qwen4ExpTextDecoderLayer.forward` modular `:801-838`; `Qwen4ExpTextModel.forward` `:1012-1027`.)

### 1.5 O que a receita faz com essa arquitetura, ponta a ponta
Serve `nvidia/Qwen3.8-Flash-Next-NVFP4` em 2 DGX Spark com vLLM, **TP=2 + EP** (experts divididos
por GPU, 256 por GPU), **MTP=3**, **KV FP8** (patch nos kernels QSA), **estado SSM BF16**, contexto
262.144 nativo (YaRN opcional para 1M), encoder de visão replicado (`--mm-encoder-tp-mode data`).
Para isso ela precisa: (a) carregar a tabela PLE em FP8 que o vLLM não reconhece
(`detect_ple_dtype.py` + `--hf-overrides` + `ple_layer_patched.py`), (b) fazer o vLLM achar e
construir as experts FP8-em-blocos da camada MTP (`patch_checkpoint_config.py` +
`patch_modelopt_fp8_block_moe.py`), (c) ensinar os kernels QSA a ler KV FP8
(`patch_qsa_fp8_kv.py`), (d) reduzir o custo de banda do MTP (vocabulário de rascunho), e
(e) contornar restrições de forma de kernels (MXFP8 `in_proj_a/b` [48,2560], visão fc1 4304).

---

## 2. Inventário de parâmetros

### 2.1 `text_config` do `config.json` — todas as chaves
Valores idênticos em `Qwen/Qwen3.8-Flash-Next` e `nvidia/…-NVFP4` salvo onde indicado.

| Chave | Valor | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|
| `model_type` | `qwen4_exp_text` | classe de config | registro de arquitetura HF/vLLM | vLLM do dia-0 usa módulo `qwen3_8_flash_next/nvidia/*`; vLLM 0.30 usa `qwen4_exp/nvidia/*` (`start.sh:627`) |
| `hidden_size` | **2560** | largura d do modelo (de cada ramo residual) | dimensão do embedding/residual | pequena para 125B: a capacidade está nas experts (512) e no N-gram; é o "K" de quase todos os GEMMs de decode |
| `num_hidden_layers` | **48** | nº de blocos | profundidade | 36 GDN + 12 QSA; MTP é instanciado como camada 48 no vLLM |
| `layer_types` | 48 entradas, `full_attention` em (i+1)%4==0 | tipo do mixer por camada | híbrido linear/softmax | define quais camadas têm KV por token (12) e quais têm estado fixo (36) |
| `full_attention_interval` | **4** | 1 atenção global a cada 4 | proporção híbrida | tech report Tab.1: híbrido GDN > Transformer puro em 8/9 benchmarks; é a razão de o KV ser barato (`README.md:366`) |
| `vocab_size` | **248.320** (padded) | tamanho do vocabulário | tokenizer BPE byte-level (Qwen2Tokenizer) | embed e lm_head = 248320×2560×2 B = 1,18 GiB cada em BF16; o lm_head do rascunho MTP vira gargalo de banda (`README.md:725-729`) |
| `tie_word_embeddings` | `false` | lm_head ≠ embed | pesos amarrados vs separados | +0,64B params; o vLLM cria um `ParallelLMHead` no drafter (`files/overlay/mtp.diff`) |
| `hidden_act` | `silu` | ativação das MLP/experts (SwiGLU) e da conv do GDN | SiLU/Swish, GLU | SwiGLU: `down(silu(gate(x)) ⊙ up(x))` |
| `rms_norm_eps` | `1e-06` | ε das RMSNorm | estabilidade numérica | — |
| `initializer_range` | 0.02 | std de init | inicialização | só treino |
| `attention_bias` | `false` | sem bias em q/k/v/o | — | menos parâmetros; FP8/NVFP4 mais simples |
| `attention_dropout` | 0.0 | — | dropout | inferência |
| `num_attention_heads` | **24** | heads de query do QSA | multi-head attention | com `head_dim` 256 → projeção de atenção de 6144 dims (> hidden 2560) |
| `num_key_value_heads` | **2** | heads de K/V do QSA | **GQA** (12 queries por KV) | KV/token/camada = 2×2×256 = 1024 elementos; em TP=2 cada GPU fica com 1 KV head (`bench_qsa_kernels.py:41-42`) |
| `head_dim` | **256** | dim de cada head do QSA | — | head grande (Qwen3-Next também usa 256) |
| `partial_rotary_factor` | **0.25** | fração da head com RoPE | RoPE parcial | só 64 das 256 dims giram; 192 são "NoPE" (conteúdo puro). Card: "Rotary Position Embedding Dimension: 64" |
| `rope_parameters.rope_type` | `default` | tipo de RoPE | RoPE/YaRN | trocado para `yarn` para 1M (seção 2.7) |
| `rope_parameters.rope_theta` | **10.000.000** | base das frequências | θ alto = comprimentos de onda longos | a menor frequência tem λ ≈ 38M tokens (cálculo §5) |
| `rope_parameters.mrope_section` | `[11, 11, 10]` | nº de pares de frequência para T, H, W | **mRoPE** (posição 3D) | 11+11+10 = 32 pares = 64 dims rotacionadas |
| `rope_parameters.mrope_interleaved` | `true` | T/H/W intercalados nas frequências | mRoPE intercalado (Qwen3-VL) | `recomposition_frequencies` (modeling `:139-148`): pega índices 1,4,7… de H e 2,5,8… de W |
| `rope_parameters.partial_rotary_factor` | 0.25 | (repetido dentro do dict) | — | **atenção**: um override que substitua o dict inteiro precisa repetir isso (§3.12) |
| `max_position_embeddings` | **262.144** | contexto nativo | janela treinada | QSA foi treinado em CPT a 256K; >262.144 exige YaRN + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| `linear_num_key_heads` | **16** | heads de Q/K do GDN | atenção linear multi-head | Q e K: 16×128 = 2048 dims cada |
| `linear_num_value_heads` | **48** | heads de V (e de estado) do GDN | "GVA": 3 heads de V por head QK | cada V head tem seu próprio estado 128×128; em TP=2, 24 por GPU |
| `linear_key_head_dim` | 128 | d_k do GDN | — | exigência dos kernels CUDA de decode GDN (`head_k_dim=128`, `README.md:477`) |
| `linear_value_head_dim` | 128 | d_v do GDN | — | estado por head = 128×128 |
| `linear_conv_kernel_dim` | **4** | kernel da conv causal curta do GDN | ShortConv depthwise | estado de conv = 3 colunas por canal (+ tokens especulativos) |
| `output_gate_type` | `sigmoid` | ativação do gate de saída do GDN | gate limitado | tech report eq.(11): sigmoid > SiLU (GDN original usa SiLU) |
| `mamba_ssm_dtype` | `float32` | dtype do estado recorrente GDN | precisão do estado | a receita sobrescreve com `--mamba-ssm-cache-dtype bfloat16` (§2.7) |
| `num_experts` | **512** | experts roteadas por camada MoE | MoE fino | 48×512 = 24.576 experts no modelo principal |
| `num_experts_per_tok` | **10** | top-k do roteador | esparsidade 10/512 ≈ 1,95% | define os ~2,36B de experts ativos por token |
| `moe_intermediate_size` | **640** | largura interna de cada expert (SwiGLU) | expert "fino" | cada expert = 3×640×2560 = 4,915M params; 640/2 = 320 em TP puro — não divisível por 128 (importa para FP8 em blocos, §3.3) |
| `shared_expert_intermediate_size` | **640** | largura do expert compartilhado | shared expert | sempre ativo, com gate sigmoide escalar |
| `norm_topk_prob` | `true` (só nvidia; o transformers já usa `True` por padrão, modular `:171`) | renormaliza os pesos top-10 para somar 1 | roteamento softmax-topk | — |
| `router_aux_loss_coef` | 0.001 | peso da loss de balanceamento | load balancing | só treino |
| `output_router_logits` | `false` | — | — | — |
| `hc_count` | **4** | nº de ramos do fluxo residual | Hyper-Connections / GR | estado residual = 4×2560 = 10240 por token; `hc_count > 1` é obrigatório (modular `:204`) |
| `hc_lowrank` | **320** | posto do gargalo do gate de leitura | low-rank gating (r = d/8) | `input_mix_weight_down` 10240→320, `up` 320→10240; 320 não é múltiplo de 128 (impede FP8 em blocos, `fable5-1-report.md:122-124`) |
| `indexer_n_heads` | **4** | heads de query do indexador QSA | indexador MQA leve | tech report Fig.5b: 4 heads bastam |
| `indexer_kv_heads` | **1** | chave compartilhada do indexador | MQA | obrigatório = 1 (modular `:229-230`) |
| `indexer_head_dim` | **128** | dim do indexador | — | RoPE parcial em 64 das 128 dims (tech report §2.1.2); a validação exige `rotary_dim ≤ indexer_head_dim` (modular `:233-239`) |
| `indexer_compress_ratio` | **4** | r: tokens por micro-bloco | compressão por average pooling | custo do indexador O(n²/r) |
| `indexer_budget` | **2048** | K: tokens selecionados por query | top-k esparso | = 512 blocos; deve ser divisível por r (modular `:231`); + cauda de até r−1=3 tokens → largura 2051 (`bench_qsa_kernels.py:6,47`) |
| `ple_layer_ids` | `[2]` | camadas (1-based) com N-gram embedding | Per-Layer Embedding / Engram | deve ser camada `linear_attention` (modular `:256-262`); na camada 2, o prefetch da tabela sobrepõe a computação da camada 1 (tech report §2.3.1) |
| `ple_embed_dim` | **2560** | dim concatenada das cabeças N-gram | — | = 16 cabeças × 160 |
| `ngram_size` | **3** | maior n-grama | bigrama + trigrama | também é a **dilatação** da conv do PLE |
| `heads_per_ngram` | **8** | hashes independentes por ordem n | multi-head hashing | (3−1)×8 = **16** cabeças; head_dim = 2560/16 = **160** |
| `ngram_vocab_size_base` | **20.000.000** | limite inferior dos tamanhos primos | tabelas de hash com tamanho primo | cada cabeça usa um primo distinto ≥ 20M (lista em §4.1) |
| `make_ngram_vocab_size_divisible_by` | 128 | pad do total de linhas | alinhamento | 320.001.446 → 320.001.536 linhas |
| `split_ngram_parts` | **128** | nº de shards da tabela no checkpoint | sharding do checkpoint | 128 × 2.500.012 linhas; o default do código é 512 (modular `:163`), o checkpoint usa 128 |
| `seed` | 1234 (só nvidia; default do código também 1234) | semente dos multiplicadores do hash | hash determinístico | multiplicadores também ficam salvos como buffer `layer_multipliers` [3] int64 |
| `ple_conv_kernel_size` | **4** | kernel da conv curta dilatada do PLE | conv causal dilatada | campo receptivo = (4−1)×3+1 = 10 tokens |
| `number_of_conv_states` | 3 (só nvidia; derivado em `__post_init__`, modular `:180`) | nº de "conv states" por camada no cache | cache híbrido | [0] conv do GDN, [1] conv do PLE, [2] contexto de n-grama (últimos 2 ids) |
| `mtp_num_hidden_layers` | **1** | camadas MTP | Multi-Token Prediction | card: "MTP: 1 layer, trained with multi-steps" — o vLLM reusa a mesma camada k vezes |
| `mtp.hybrid` | `true` | MTP com camadas tipadas | — | — |
| `mtp.layer_types` | `["full_attention"]` | o MTP usa atenção global (QSA) | — | por isso o MTP tem KV cache próprio (13ª camada KV) |
| `mtp.num_hidden_layers` | 1 | — | — | — |
| `mtp.rope_theta` | 10.000.000 | θ do MTP | — | — |
| `mtp.mtp_use_hidden_state_from_layer` | `null` | usa o estado final | — | — |
| `mtp_use_dedicated_embeddings` | `false` | o MTP reusa `embed_tokens` | pesos compartilhados | por isso o checkpoint não tem embedding próprio do MTP |
| `bos_token_id` / `eos_token_id` | 248044 (`<|endoftext|>`) | — | — | o PLE usa **248044** como separador de documento/padding dos n-gramas |
| `pad_token_id` | `null` | — | — | — |
| `dtype` | `bfloat16` | dtype de referência | BF16 | partes não quantizadas ficam BF16 |
| `use_cache` | `true` | — | — | — |

### 2.2 Chaves de topo do `config.json`
| Chave | Valor | Significado |
|---|---|---|
| `architectures` | `["Qwen4ExpForConditionalGeneration"]` | modelo multimodal (visão + LM) |
| `model_type` | `qwen4_exp` | — |
| `image_token_id` / `video_token_id` | 248056 (`<|image_pad|>`) / 248057 (`<|video_pad|>`) | placeholders substituídos pelos embeddings visuais (`masked_scatter`, modular `:1091-1107`) |
| `vision_start_token_id` / `vision_end_token_id` | 248053 / 248054 | delimitadores |
| `language_model_only` | `false` | carrega o ViT |
| `tie_word_embeddings` | `false` | — |
| `transformers_version` | base `5.8.0.dev0`; nvidia `5.16.0.dev0` | card nvidia exige Transformers ≥ 5.16 para Qwen4-Exp |

### 2.3 `vision_config`
| Chave | Valor | Significado / conceito |
|---|---|---|
| `model_type` | `qwen4_exp_vision` (nvidia) / `qwen4_exp` (base) | — |
| `depth` | **27** | blocos ViT |
| `hidden_size` | **1152** | largura do ViT |
| `num_heads` | **16** | head_dim = 72 |
| `intermediate_size` | **4304** | MLP do ViT (GELU-tanh, com bias). 4304 % 32 = 16 e 4304/2 = 2152 → origem de dois problemas de kernel (§3.4, `.env.sample:88-92`) |
| `hidden_act` | `gelu_pytorch_tanh` | — |
| `patch_size` | 16 | patch 16×16 px |
| `temporal_patch_size` | 2 | 2 quadros por patch (Conv3d 2×16×16), imagens são duplicadas no tempo |
| `spatial_merge_size` | 2 | o merger junta 2×2 patches → 1 token de 4×1152 = 4608 dims |
| `out_hidden_size` | **2560** | = hidden do LM (merger: LayerNorm → fc1 4608→4608 → GELU → fc2 4608→2560) |
| `num_position_embeddings` | 2304 | posição absoluta aprendida 48×48, interpolada bilinearmente para cada grade + RoPE axial 2D |
| `in_channels` | 3 | RGB |
| `deepstack_visual_indexes` | `[]` | sem DeepStack (sem injeção de features intermediárias) |
| (inferência) | — | 1152/4304/27/16 é a geometria do SigLIP-2 So400m/16, a mesma do encoder Qwen3.5-397B (`out_hidden_size` 4096 lá). Qwen3.8 herda o encoder da família Qwen3.5/Qwen3-VL. |

Pré-processamento (`preprocessor_config.json`): `shortest_edge` 65.536 px, `longest_edge`
16.777.216 px, média/desvio 0.5, `Qwen2VLImageProcessorFast`. Como cada token visual final cobre
32×32 = 1024 px, `longest_edge` = 16.384 tokens — o "vision encoder budget 16,384 tokens" de
`README.md:620-621`; um quadro 3840×2160 → 120×67,5 ≈ 8,1k tokens (`README.md:613-614` mede ≈8,2k).
Vídeo (`video_preprocessor_config.json`): `longest_edge` 25.165.824, `shortest_edge` 4096; o card
sugere 469.762.048 (224k tokens de vídeo) para vídeos de hora.

### 2.4 Quantização declarada — `quantization_config` (config.json) e `hf_quant_config.json`
| Campo | Valor | Significado |
|---|---|---|
| `quant_method` | `modelopt` | NVIDIA Model Optimizer 0.46.0.dev281 |
| `quant_algo` | **`MIXED_PRECISION`** | cada módulo declara seu próprio algoritmo em `quantized_layers` |
| `config_groups.group_0` | pesos e ativações float 4-bit, `group_size` 16, `dynamic: false` → **NVFP4 W4A4** | alvo: `model.language_model.layers.{0..47}.mlp.experts` |
| `config_groups.group_1` | pesos float8 `group_size` 128 estático + ativações float8 **dinâmicas** `group_size` 128 | alvo: `mtp.layers.0.mlp.experts` (FP8 em blocos 128×128) |
| `config_groups.group_2` | pesos float 8-bit (sem group_size = per-tensor) | alvo: `model.language_model.layers.1.ple.ple_embedding.ngram_embedding` |
| `quantized_layers` (50 entradas) | 48× `{"quant_algo":"NVFP4","group_size":16}`, 1× `{"quant_algo":"FP8"}` (PLE), 1× MTP experts | config.json (rev `fc694b54`, 2026-09-05) diz `FP8_PB_WO`; `hf_quant_config.json` diz `FP8_BLOCK_SCALES` — mesmo formato (`patch_checkpoint_config.py:22-27,87-93`) |
| `ignore` / `exclude_modules` (292 padrões) | `lm_head`, `embed_tokens`, `hyper_connection_mixer*`, por camada: `attn_hyper_connection*`, `mlp_hyper_connection*`, `mlp.gate`, `mlp.shared_expert*`, `mlp.shared_expert_gate`, `linear_attn*` (36), `self_attn*` (12), e `model.visual*` | tudo isso fica **BF16** |
| `kv_cache_quant_algo` (hf_quant) | `None` | sem escalas de KV calibradas → KV FP8 usa escala 1.0 por padrão (`_qsa_scale_ptr`, `patch_qsa_fp8_kv.py:69-81`) (inferência sobre o valor efetivo) |

Card nvidia: "routed MoE expert linear layers in the main language model use W4A4 NVFP4 with
MSE-calibrated weight scales… MTP routed experts use 128x128 block-scaled FP8, and the PLE n-gram
embedding uses per-tensor FP8… ~2.7x smaller than the BF16 source (about 63% reduction)". MTP e PLE
são byte-a-byte iguais ao `Qwen/Qwen3.8-Flash-Next-FP8`. Calibração: cnn_dailymail +
Nemotron-Post-Training-Dataset-v2.

### 2.5 Outros arquivos do checkpoint
| Item | Valor | Onde aparece |
|---|---|---|
| `generation_config.json` | `do_sample: true`, `temperature 1.0`, `top_p 0.95`, `top_k 20`, `eos_token_id [248046, 248044]`, `pad 248044`, `bos 248044` | vLLM adota como padrão de amostragem (`README.md:481`) |
| Tokens especiais | 248044 `<|endoftext|>`, 248045 `<|im_start|>`, 248046 `<|im_end|>` (eos do tokenizer), 248053/54 vision start/end, 248055 `<|vision_pad|>`, 248056 `<|image_pad|>`, 248057 `<|video_pad|>`, 248058/59 `<tool_call>`/`</tool_call>` | `tokenizer_config.json` |
| `model_max_length` | 262.144 | — |
| Chat template kwargs | `reasoning_effort` ∈ {`low`,`medium`,`xhigh`} (default `xhigh`; `high`/`max` → HTTP 400), `enable_thinking` (default true; false pré-preenche `<think></think>`), `preserve_thinking` (default true, card oficial) | `README.md:623-652`; card oficial |
| Amostragem recomendada (card oficial) | thinking: T=1.0, top_p 0.95, top_k 20, min_p 0, presence 0; instruct: T=0.7, top_p 0.8, top_k 20, presence 1.5 | card base |

### 2.6 Inventário de tensores do checkpoint nvidia (lido dos cabeçalhos safetensors)
299.545 tensores, 11 arquivos (`model-00001..00010-of-00010.safetensors` + `model-fp8-mtp-ple.safetensors` de 53,7 GB), 123,53 GiB = 132,64 GB.

| Grupo (por camada salvo indicado) | Forma | dtype | Nº | Bytes totais |
|---|---|---|---|---|
| `embed_tokens.weight` | [248320, 2560] | BF16 | 1 | 1,184 GiB |
| `lm_head.weight` | [248320, 2560] | BF16 | 1 | 1,184 GiB |
| **GDN** `linear_attn.in_proj_qkv.weight` | [10240, 2560] (Q 2048 + K 2048 + V 6144) | BF16 | 36 | 1,758 GiB |
| GDN `in_proj_z.weight` (gate de saída) | [6144, 2560] | BF16 | 36 | 1,055 GiB |
| GDN `in_proj_a.weight` (decay α) / `in_proj_b.weight` (β) | [48, 2560] cada | BF16 | 36+36 | 0,016 GiB |
| GDN `conv1d.weight` | [10240, 1, 4] | BF16 | 36 | 0,003 GiB |
| GDN `A_log`, `dt_bias` | [48] | BF16 | 36+36 | ~0 |
| GDN `norm.weight` (RMSNormGated, compartilhado entre heads) | [128] | BF16 | 36 | ~0 |
| GDN `out_proj.weight` | [2560, 6144] | BF16 | 36 | 1,055 GiB |
| **QSA** `self_attn.q_proj.weight` (query + gate) | [12288, 2560] = 24 heads × 256 × 2 | BF16 | 12 | 0,703 GiB |
| QSA `k_proj`, `v_proj` | [512, 2560] = 2 × 256 | BF16 | 12+12 | 0,059 GiB |
| QSA `o_proj.weight` | [2560, 6144] | BF16 | 12 | 0,352 GiB |
| QSA `q_norm`, `k_norm` | [256] | BF16 | 12+12 | ~0 |
| QSA `indexer.index_qk_proj.weight` | [640, 2560] = (4+1)×128 | BF16 | 12 | 0,037 GiB |
| QSA `indexer.q_layernorm`, `k_layernorm` | [128] | BF16 | 12+12 | ~0 |
| **GR** `{attn,mlp}_hyper_connection.hc_norm.weight` | [10240] | BF16 | 96 | ~0 |
| GR `input_mix_weight_down.weight` | [320, 10240] | BF16 | 96 | 0,586 GiB |
| GR `input_mix_weight_up.weight` | [10240, 320] | BF16 | 96 | 0,586 GiB |
| GR `block_inject_weight.weight` (escrita) | [4, 10240] | BF16 | 96 | 0,007 GiB |
| `hyper_connection_mixer.{hc_norm,input_mix_weight_down,up}` (final, sem inject) | [10240], [320,10240], [10240,320] | BF16 | 1 cada | 0,012 GiB |
| **MoE** `mlp.gate.weight` (roteador) | [512, 2560] | BF16 | 48 | 0,117 GiB |
| MoE `shared_expert.{gate,up}_proj` / `down_proj` | [640,2560] / [2560,640] | BF16 | 48×3 | 0,439 GiB |
| MoE `shared_expert_gate.weight` | [1, 2560] | BF16 | 48 | ~0 |
| **Experts roteadas** `experts.E.{gate,up}_proj.weight` | [640, 1280] U8 (2 valores FP4 por byte → 640×2560) | U8 | 24.576 × 2 | 37,5 GiB |
| experts `down_proj.weight` | [2560, 320] U8 (→ 2560×640) | U8 | 24.576 | 18,75 GiB |
| experts `*.weight_scale` (escala por bloco de 16) | [640,160] / [2560,40] | F8_E4M3 | 73.728 | 7,03 GiB |
| experts `*.weight_scale_2` (escala global do tensor) | escalar | F32 | 73.728 | ~0 |
| experts `*.input_scale` (escala global estática de ativação) | escalar | F32 | 73.728 | ~0 |
| **PLE** `ple.ple_embedding.ngram_embedding.shard_S.weight` | [2.500.012, 160] | F8_E4M3 | 128 | **47,68 GiB** |
| PLE `ngram_embedding.weight_scale` (escala única da tabela toda) | [1] | BF16 | 1 | — |
| PLE `layer_multipliers` | [3] | I64 | 1 | multiplicadores do hash |
| PLE `ngram_heads_vocab_sizes`, `ngram_heads_offsets` | [16] | I64 | 1+1 | primos e offsets |
| PLE `key_proj.weight` | [10240, 2560] | BF16 | 1 | 0,049 GiB |
| PLE `value_proj.weight` | [2560, 2560] | BF16 | 1 | 0,012 GiB |
| PLE `norm_key`, `norm_query`, `norm_conv` | [10240] | BF16 | 3 | ~0 |
| PLE `conv1d.weight` | [10240, 1, 4] | BF16 | 1 | ~0 |
| **Visão** 27 blocos `attn.qkv` [3456,1152]+bias, `attn.proj` [1152,1152]+bias, `mlp.linear_fc1` [4304,1152]+bias, `linear_fc2` [1152,4304]+bias, `norm1/2` (LayerNorm com bias) | — | BF16 | — | 0,836 GiB total |
| Visão `patch_embed.proj` | [1152, 3, 2, 16, 16] + bias | BF16 | 1 | — |
| Visão `pos_embed` | [2304, 1152] | BF16 | 1 | — |
| Visão `merger.norm` [1152], `linear_fc1` [4608,4608], `linear_fc2` [2560,4608] | — | BF16 | — | — |
| **MTP** `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight` | [2560, 2560] cada | BF16 | 2 | 0,024 GiB |
| MTP `pre_fc_norm_embedding` [2560], `pre_fc_norm_hidden` [10240] | — | BF16 | 2 | — |
| MTP `layers.0.self_attn.*` (mesmas formas do QSA, incl. indexador) | — | BF16 | — | ~0,09 GiB |
| MTP `layers.0.{attn,mlp}_hyper_connection.*`, `mtp.hyper_connection_mixer.*` | formas do GR | BF16 | — | — |
| MTP `layers.0.mlp.gate`, `shared_expert*` | formas do MoE | BF16 | — | — |
| MTP `layers.0.mlp.experts.E.{gate,up}_proj.weight` | [640, 2560] | **F8_E4M3** | 512×2 | 1,56 GiB |
| MTP experts `down_proj.weight` | [2560, 640] | F8_E4M3 | 512 | 0,78 GiB |
| MTP experts `*.weight_scale_inv` (1 escala por bloco 128×128) | [5,20] / [20,5] | BF16 | 1536 | ~0 |

Observações: (1) o checkpoint **não** tem `mtp.lm_head` nem `mtp.embed_tokens`
(`mtp_use_dedicated_embeddings: false`); (2) o checkpoint não tem `token_lookup` nem `hashstats_*`
— o loader vLLM os ignoraria se existissem (`ple_layer_patched.py:759-761`); (3) não existe
`input_layernorm`/`post_attention_layernorm` nem `model.norm`: o GR faz esse papel.

### 2.7 Knobs da receita que dependem da arquitetura
| Nome | Default(s) / alternativas | Onde | O que faz | Conceito | Por que importa (repo + explicação técnica) |
|---|---|---|---|---|---|
| `MODEL_ID` | `nvidia/Qwen3.8-Flash-Next-NVFP4` (133G/11 shards: BF16 denso + experts NVFP4 + PLE FP8); alt. `local-inference-lab/…` (99G/37 shards: MXFP8 atenção/GDN + **PLE NVFP4**); `RadixArk/…` (126G/206 shards: BF16 atenção + PLE FP8) | `.env.sample:24-26` | escolhe o checkpoint | quantização mista | os três diferem **só em precisão** (`HANDOFF:113-131`): trocar MXFP8→BF16 e PLE NVFP4→FP8 custa 27 GB e ~13-16% de decode (`HANDOFF:174-188`), mas preserva as partes mais sensíveis |
| `PLE_EMBEDDING_DTYPE` | vazio (auto-detecta) → nvidia: `float8_e4m3fn`; local-inference-lab já declara `nvfp4`; RadixArk já declara `float8_e4m3fn` | `.env.sample:36-41`, `start.sh:340-362`, `files/detect_ple_dtype.py` | injeta `text_config.ple_embedding_dtype` via `--hf-overrides` | dispatch de método de quantização | sem isso a tabela FP8 de 51 GB não carrega (`README.md:674-677`) |
| `--hf-overrides '{"text_config":{…}}'` | payload único aninhado em `text_config` | `start.sh:919-946` | sobrescreve config do HF | override de config | aninhar em `text_config` corrigiu o YaRN silencioso (§3.12) |
| `YARN_ENABLE` | `.env.sample`: `false` (README `:217` diz `true` — desatualizado) | `.env.sample:46`, `start.sh:163-170` | liga `rope_type: yarn` | extensão de contexto | força `false` se `MAX_MODEL_LEN ≤ 262144` ("no benefit and costs accuracy", `README.md:836-838`) |
| `YARN_FACTOR` | `4.0` | `.env.sample:52` | s = 4 → 262.144×4 ≈ 1,05M | YaRN | card oficial: usar fator menor (2.0) se o uso típico é 512K |
| `MAX_MODEL_LEN` | `.env.sample`: 262144 (README `:216` diz 1000000 — desatualizado) | `.env.sample:45` | contexto máximo servido | — | a QSA dimensiona buffers de logits do indexador por `MAX_MODEL_LEN` (`fable5-1-report.md:154-166`): servir 1M quando se usa 32k custa alguns % |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | `.env.sample:176` | permite > `max_position_embeddings` | — | vLLM avisa "extreme caution" (`README.md:831-834`) |
| `KV_CACHE_DTYPE` | `fp8` (default); `auto` = bf16 | `.env.sample:60-69`, `start.sh:87` | dtype do KV **principal** das 13 camadas QSA | KV FP8 E4M3 | 1,70× tokens por GiB (não 2×: indexador/compressor ficam BF16); risco: a quantização das chaves muda **quais blocos o indexador escolhe** (`.env.sample:64-68`, `patch_qsa_fp8_kv.py:19-23`) |
| `MAMBA_SSM_CACHE_DTYPE` | `bfloat16` no `.env.sample`; vazio = `float32` do checkpoint | `.env.sample:70-82`, `start.sh:88-93,884` | dtype do estado recorrente GDN | precisão do estado SSM | metade dos bytes lidos/escritos por passo (~0,23 GB/seq em TP1) e página mamba menor → bloco de atenção menor (3.200→1.664 tokens); medido no single-Spark +8,5% decode @8 streams, agulhas 15/15 (§4.4) |
| `MM_ENCODER_TP_MODE` | `data` (replica o ViT); `weights` quebra no load | `.env.sample:88-92`, `start.sh:97,892` | paralelismo do encoder de visão | TP vs replicação | `intermediate_size` 4304 / 2 = 2152, não múltiplo de 16 exigido pelos kernels (comentário do repo); ViT = 0,84 GiB, replicar é barato |
| `TENSOR_PARALLEL_SIZE` | 2 | `.env.sample:85` | fatias densas por GPU | TP | formas por GPU em §2.8 |
| `ENABLE_EXPERT_PARALLEL` | `true` ("Required for NVFP4") | `.env.sample:86`, `start.sh:896-900` | cada GPU guarda 256 das 512 experts inteiras | EP | (inferência + card nvidia) as experts FP8 do MTP têm blocos 128×128; em TP puro o intermediário 640 viraria 320 por GPU (não múltiplo de 128). O card da nvidia pede EP "so the MTP expert width remains compatible with its 128x128 FP8 blocks" |
| `--all2all-backend allgather_reducescatter` | junto com EP | `start.sh:899` | backend de despacho MoE | all-to-all | **nunca é invocado** com DP=1 (`fable5-1-report.md:37`): tokens são replicados por TP, cada GPU computa suas 256 experts, soma via all-reduce do TP |
| `MTP_NUM_SPECULATIVE_TOKENS` | 3 (0 = desliga) | `.env.sample:95` | nº de rascunhos por passo | spec decoding | 1 camada MTP aplicada 3× (vLLM avisa sobre aceitação menor, `README.md:993-995`) |
| `MTP_INDEX_SHARE` → `index_share_for_mtp_iteration` | `.env.sample`: `true`; `start.sh` default `false` | `.env.sample:136-139`, `start.sh:158,906` | reusa o top-k do QSA nos passos 2-3 do rascunho | reuso de índices (GLM-5) | tech report Tab.4: aceitação 4,06→4,07 com reuso; aqui código +6% (S=1) a +27% (S=8) |
| `MTP_DRAFT_VOCAB` + `use_local_argmax_reduction` | `files/draft_vocab_en_code_47k.txt` (47.149 ids) | `.env.sample:110-130`, `start.sh:903-914` | argmax do rascunho só em subconjunto do vocab | FR-Spec | a cabeça 248.320×2560 é lida 3×/passo; ver §3.7 |
| `MTP_DISABLE_BLOCK_DROP` | 1 | `.env.sample:132-135` | mantém o último bloco de prefix cache | cache híbrido + EAGLE | bloco = 1.664 tokens em MTP=3 (`patch_block_drop.py:7-8`) |
| `VLLM_QSA_DET_TOPK` / `VLLM_MOE_DET_FINALIZE` | desligados | `.env.sample:140-144` | ordena o top-k do QSA / finalize MoE não-fundido | determinismo | −4% decode/prefill; saída greedy bit-idêntica |
| `QSA_PROFILE` / `VLLM_QSA_PROFILE(_JSON)` | `stock` | `.env.sample:154-158` | tabela de tiles dos kernels QSA | tuning Triton | tabelas originais afinadas no GB300 (160 SMs); GB10 tem 48 SMs |
| `PLE_OFFLOAD` → `VLLM_PLE_CPU_OFFLOAD=1` | `false` | `.env.sample:101`, `start.sh:980-981` | tabela N-gram em processo CPU separado | offload de embeddings | em memória unificada **não economiza** memória física (`fable5-1-report.md:221-222`); precisa ~51 GB de RAM livre (`README.md:971-974`) |
| `VLLM_PLE_PACKED_TABLE_DIR` | — | `patch_ple_offload.py:20-25` | tabela empacotada mmap | page cache evictável | TP1: 104 → 77 GiB não-evictável |
| `FP8_DENSE` | `false` | `.env.sample:147-153` | projeções densas BF16 → FP8 por canal | FP8 per-channel/per-token | 9,2 → 5,7 GiB lidos por passo (analítico, não medido) |
| `ABLIT` | 0 | `.env.sample:27-34` | checkpoint com `o_proj` das 12 QSA modificado | abliteração (projeção de direção) | só 12 tensores mudam; resto idêntico (`README.md:245-300`) |
| `--kv-cache-memory`, `GPU_MEMORY_UTILIZATION` | GMU 0.80 (0.835 antes de 2026-09-26) | `.env.sample:53-56` | orçamento de memória | — | pesos (68,5 GiB/GPU com não-torch), não KV, são o limite (`README.md:327-328`) |
| `--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | fixo | `start.sh:917` | CUDA graphs só em decode, sem inductor | — | o PLE registra seu conv como custom op com fake impl para ser capturável (`ple_layer_patched.py:1630-1661`) |
| `--mamba-cache-mode` | forçado `align` pelo modelo (`"all"` levanta erro) | `fable5-1-report.md:44` | como o estado GDN participa do prefix caching | cache híbrido | prefix cache só acerta em fronteiras de bloco (1600/1664 tokens) |

### 2.8 Formas por GPU em TP=2 (o que o kernel realmente vê)
De `fable5-1-report.md:188-192` (conferido contra o checkpoint):

| GEMM (por GPU) | Forma [N, K] | Derivação |
|---|---|---|
| GDN qkvz (fundido) | (8192, 2560) | (10240 + 6144)/2 |
| GDN out | (2560, 3072) | 6144/2 na entrada (RowParallel) |
| GDN ba (b e a fundidos) | (48, 2560) | (48+48)/2 — por isso o erro "N=48 < 128" do MXFP8 |
| QSA qkv+gate | (6656, 2560) | (12288 + 512 + 512)/2 |
| QSA out | (2560, 3072) | 6144/2 |
| indexador | (640, 2560) | **replicado** |
| shared-expert gate/up | (640, 2560) | 1280/2 |
| lm_head | (124160, 2560) | 248320/2 (vocab-parallel) |
| HC down+inject | (336, 10240) | 320 + 4 = 324, com padding para 336; **replicado** (`disable_tp=True`) |
| HC up | (10240, 320) | replicado |
| Experts | 256 experts inteiras por GPU | EP |
| PLE | ~160M linhas × 160 por GPU (VocabParallelEmbedding) | 47,7/2 ≈ 23,8 GiB |

---

## 3. Inventário de modificações (tudo que muda o comportamento do modelo/kernels/checkpoint)

### 3.1 Patch de carregamento do PLE (N-gram) em FP8/NVFP4
- **Arquivos**: `files/patch_ple_layer.py` (gerador, 41.776 B) → `files/ple_layer_patched.py`
  (bind-mount `:ro` sobre `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`, `start.sh:800-832`);
  `files/detect_ple_dtype.py`; `--hf-overrides` (`start.sh:919-946`).
- **Problema**: o checkpoint guarda a tabela como 128 shards FP8 + **uma** `weight_scale` global,
  mas declara `*.ple.*` fora do NVFP4. O resolvedor do vLLM do dia-0 só habilitava PLE FP8 quando o
  checkpoint **inteiro** era `Fp8Config`; caso contrário criava uma embedding BF16 de ~102 GB e
  quebrava (`README.md:552-555`). Além disso, o nvidia não declara `text_config.ple_embedding_dtype`
  (só em `config_groups`), então nem o dispatch novo acionava (`detect_ple_dtype.py:4-19`).
- **Mecanismo**: porta o dispatch do vLLM PR #53899 (`qwen4_exp`): `_get_ple_embedding_quant_method`
  (`ple_layer_patched.py:355-406`) primeiro olha `ple_embedding_dtype` (`nvfp4`/`fp4`/`w4a16_nvfp4` →
  `Qwen3_8FlashNextPLENVFp4EmbeddingMethod`; `float8_e4m3fn`/`fp8` → `Qwen3_8FlashNextPLEFp8EmbeddingMethod`),
  só depois os casos `Fp8Config`/`ModelOptNvFp4Config`. O método FP8 cria `weight` FP8 +
  `weight_scale` per-tensor BF16 e faz lookup puro (`F.embedding`); a deseq. é
  `emb.to(bf16) * weight_scale` (`:1057-1064`). O método NVFP4 mantém linhas empacotadas
  `cat(códigos[head_dim/2], escalas_fp8[head_dim/16])` = 80+10 = 90 B por cabeça e deseq. com LUT
  E2M1 (`:193-283`, `:304-338`). `detect_ple_dtype.py` lê o `num_bits` do grupo que mira
  `…ple_embedding.ngram_embedding` (8 → `float8_e4m3fn`, 4 → `nvfp4`) e o `start.sh` injeta em
  `text_config`.
- **Conceitos**: embedding quantizada, escala per-tensor, VocabParallelEmbedding, bind-mount de
  código Python sem rebuild de imagem.
- **Efeito**: sem o patch, a tabela de 51 GB não carrega; com ele, ~24 GiB por GPU.
- **Discrepância**: `README.md:460,557` e `fable5-1-report.md:423` falam de um shim
  `PLE_QUANT_OVERRIDE=fp8`; a string não existe em `start.sh` nem nos `.py` — o mecanismo atual é
  `ple_embedding_dtype` via `--hf-overrides`.

### 3.2 Alias de índice da camada MTP + normalização `FP8_PB_WO`
- **Arquivos**: `files/patch_checkpoint_config.py`; `start.sh:689-743` (passo 4g); outputs
  `files/config_patched.json`, `files/hf_quant_config_patched.json` montados sobre o snapshot.
- **Problema**: o vLLM instancia a camada MTP no índice **absoluto** que continua a pilha
  (`mtp.layers.48` porque `num_hidden_layers` = 48) e casa prefixos com `quantized_layers` por
  string exata. nvidia registra só `mtp.layers.0.*` → o MoE do MTP é construído **não quantizado**
  e o load morre em ~7 min com `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`
  (`patch_checkpoint_config.py:5-20`).
- **Mecanismo**: `add_aliases` cria `mtp.layers.{48+i}.*` para cada `mtp.layers.i.*` em
  `quantized_layers` e em `config_groups.targets` (`:43-81`); `normalize_mtp_fp8_pb_wo` reescreve
  `FP8_PB_WO` com `group_size 128` → `FP8_BLOCK_SCALES` (`:96-127`); `--mtp-moe-algo` faz preflight
  e sai com código 3 se o algoritmo não é suportado (`:130-144,191-196`). Patcheia os dois arquivos
  porque podem discordar (config.json `FP8_PB_WO` vs sidecar `FP8_BLOCK_SCALES`); a evidência
  (issue #38) é que o dispatch lê `config.json`.
- **Conceito**: MTP como camada extra "empilhada" no fim; metadados de quantização por módulo.
- **Efeito**: MTP=3 funciona (52,1 tok/s vs 24,5 sem MTP, §4.5).

### 3.3 Experts FP8 em blocos 128×128 no MTP
- **Arquivo**: `files/patch_modelopt_fp8_block_moe.py` (edita `files/modelopt_patched.py` in-place,
  empilha sobre o patch MXFP8; `start.sh:855-858`).
- **Problema**: `ModelOptMixedPrecisionConfig.get_quant_method` só constrói `RoutedExperts` para
  `FP8`/`NVFP4`/`W4A16_NVFP4`/`MXFP8`; qualquer outra coisa retorna `None` → MoE silenciosamente
  não quantizado. **Upstream também não tem** (nem em `d4d703ca`, recomendado pelo card, nem no
  `main`) — `README.md:694-698`.
- **Mecanismo**: adiciona o ramo `if quant_algo in ("FP8_BLOCK_SCALES", "FP8_PB_WO")` que devolve o
  `Fp8MoEMethod` do próprio vLLM com `Fp8Config(is_checkpoint_fp8_serialized=True,
  activation_scheme="dynamic", weight_block_size=[g, g])`, com `g` lido do `group_size` do
  checkpoint (recusa adivinhar: bloco errado aplicaria escalas desalinhadas sem erro) — `:30-62,78-101`.
  O `Fp8MoEMethod` nomeia as escalas `weight_scale_inv`, exatamente o que o checkpoint tem
  ([5,20] para 640×2560).
- **Conceitos**: FP8 E4M3 com escala por bloco 2D 128×128, ativação FP8 dinâmica por grupo de 128.
- **Efeito**: validação indireta — curva de aceitação decrescente 89% / 74,5% / 60% ("a wrong block
  shape would show up as near-random acceptance, not as a crash", `README.md:711-712`).

### 3.4 Fallback de kernel MXFP8 → emulação BF16
- **Arquivo**: `files/patch_modelopt_mxfp8.py` → `files/modelopt_patched.py` (monta sobre
  `quantization/modelopt.py`, `start.sh:834-862`; pulado no vLLM 0.30).
- **Problema**: `FlashInferCutlassMxfp8LinearKernel` (`mm_mxfp8`) no sm_121 só aceita
  N ≥ 128, N % 32 == 0, K ≥ 128, K % 32 == 0 (medido). Falham: `linear_attn.in_proj_a/b` [48,2560]
  (N < 128 → fatal no 1º forward do profiling, ~7 min após o load) e `visual.blocks.*.mlp.linear_fc1`
  [4304,1152] (4304 % 32 = 16) — `README.md:564-591`.
- **Mecanismo**: `ModelOptMxFp8LinearMethod.create_weights` inspeciona o (N,K) **pós-split de TP** e
  troca o kernel por `EmulationMxfp8LinearKernel` (deseq. MXFP8→BF16 uma vez no load, ~17 MB extra
  nos 72 `in_proj_a/b`); todo `visual.*` vai para emulação por prefixo.
- **Relevância para o nvidia**: nesse checkpoint `linear_attn*` e `visual*` já são **BF16**
  (lista `ignore`), então o patch é inerte; ele importa para `local-inference-lab/…`, cujo
  `quantized_layers` tem 467 entradas MXFP8 (group 32: GDN, QSA, shared experts, visão) + 29
  `W4A16_NVFP4`. O docstring do patch atribui as formas ao RadixArk, mas o `HANDOFF:121` diz que
  RadixArk tem atenção BF16 — inconsistência do repo (a verificar qual checkpoint originou o erro).
- **Conceito**: MX (microscaling OCP): blocos de 32 elementos com expoente compartilhado; restrições
  de tile de GEMM.

### 3.5 KV cache FP8 nos kernels QSA
- **Arquivos**: `files/patch_qsa_fp8_kv.py` (dia-0; vendorizado do single-Spark, abordagem de
  `lancelind/qwen3.8-Flash-DGX`) e `files/patch_qsa_fp8_kv_v030.py` (backport de vllm#55557 para
  v0.30). Saídas montadas sobre `nvidia/ops/qsa.py` e `nvidia/qsa.py` (`start.sh:662-679`).
- **Problema**: o backend QSA declara `supported_kv_cache_dtypes = ["auto","bfloat16"]` e levanta
  `Qwen3.8-Flash-Next QSA requires a BF16 main KV cache`.
- **Mecanismo**: adiciona `KV_QUANT_MODE: tl.constexpr` e ponteiros de escala a dois kernels Triton:
  o **scorer MQA** do indexador (`_qsa_mqa_paged_kernel`: `keys.to(query.dtype)`, `tl.dot`, depois
  `scores *= k_scale`) e o **GQA esparso split-K** (`scores *= k_scale` após `tl.dot(q,k)`,
  `normalized_output *= v_scale` no fim). As escalas por tensor são aplicadas **depois** dos dots
  tensor-core (algebricamente equivalente antes do arredondamento, evita tiles FP32). Reinterpreta a
  alocação `uint8` como `float8_e4m3fn` sem cópia; recusa ler bytes quantizados como BF16. Neutraliza
  a rejeição herdada de `FlashAttentionImpl.__init__` (o forward FA2 nunca roda nessas camadas).
  Com `constexpr`, o caminho BF16 compila igual ao original.
- **Conceitos**: KV quantizado, escala per-tensor, fusão de escala no epílogo, esparsidade dependente
  das chaves.
- **Efeito medido**: 2.131.159 → **3.652.200** tokens (1,70×) com o mesmo pool ~32 GiB; 12/12 no
  `reasoning_check`; decode −5,3% médio no re-run de 2026-09-16 (−1,4% prosa, −9,2% código; tok/step
  cai em 8/8 células). Upstream reportou long-reasoning 6/6 → 2/6 com KV FP8 — não reproduzido aqui.
  vLLM 0.30 + backport: 2,64M tokens, NLL 1,328/1,334, ~5% mais lento.

### 3.6 Perfis de lançamento dos kernels QSA (GB10)
- **Arquivos**: `files/qsa_gb10/{qsa.py,qsa.diff,apply_patch.py,bench_qsa_kernels.py}`.
- **Problema**: tabelas de tiles afinadas no GB300 (160 SMs); no GB10 (48 SMs) o perfil de decode
  `(block_n 16, 64 splits, 4 warps)` lança ~5× mais CTAs que a máquina roda ao mesmo tempo e paga
  um kernel de merge sobre 64 parciais.
- **Mecanismo**: `VLLM_QSA_PROFILE=stock|gb10` ou `VLLM_QSA_PROFILE_JSON`; perfil gb10: sparse
  small `(32,16,4)`, lt32 `(32,8,4)`, le256 `(64,4,2)`, le512 `(64,2,2)`, large `(64,1,2)`; MQA tiles
  2/8, 2 warps. O bench varre block_n ∈ {16,32,64} × splits 1..64 × warps {2,4} e tiles MQA.
- **Efeito**: **não medido** ("a starting point, not a result", `README.md:548`).
- **Conceitos**: split-K / flash-decoding, ocupação de SMs, Triton `num_warps`.

### 3.7 Rascunho MTP com vocabulário reduzido (FR-Spec), TP-aware
- **Arquivos**: `files/patch_mtp_draft_vocab.py` (dia-0), `files/patch_mtp_draft_vocab_v030.py`,
  `files/build_draft_vocab.py`, `files/draft_vocab_en_code_47k.txt`, `bench/gen_draft_corpus.py`,
  `files/test_draft_vocab.py`.
- **Problema (arquitetural)**: vocab 248.320 e `tie_word_embeddings: false` → a cabeça do drafter
  (248320×2560×2 B = 1,18 GiB, 0,59 GiB/GPU em TP2) é lida **3× por passo** com MTP=3, ~26% dos
  ~9,2 GiB movidos por passo (`fable5-1-report.md:79`).
- **Mecanismo**: cada rank fatia **seu** shard do lm_head para os ids do subconjunto que caem no seu
  intervalo, faz argmax local e reduz os vencedores entre ranks com all-gather (valor, índice) —
  O(batch·2·tp) em vez de O(batch·vocab). Só `get_top_tokens` (acionado por
  `use_local_argmax_reduction`) usa a cabeça reduzida; `compute_logits` mantém o vocab completo.
  Escala de logits e soft-cap tanh são monotônicos → não alteram o argmax (`:30-32`).
- **Por que não muda a saída**: rascunho greedy; o alvo verifica todo token.
- **Efeito**: +9,6% médio (2026-09-11), +8,4% (re-run 2026-09-16); ganho todo em ms/passo
  (código S=1 61,2→54,0 ms) com tok/passo inalterado. Os 47k ids caem ~97% no range do rank 0, por
  isso o rank crítico lê 0,22 GiB (o passo espera o rank mais lento).

### 3.8 Backport de "block drop" (vllm#53388)
- **Arquivo**: `files/patch_block_drop.py` (overlay de `config/speculative.py`,
  `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py`).
- **Relação com a arquitetura**: com drafters estilo EAGLE/MTP, o prefix cache descarta o último
  bloco casado; como o cache híbrido força blocos de **1.664 tokens** (MTP=3, KV fp8, SSM bf16),
  isso é muito prefill extra por turno. A QSA não suporta sliding window, por isso o 7º arquivo do
  PR não é portado (`:22-26`).
- **Efeito**: TTFT turno 1 (sessão 9,5k) 3,25 → 0,63 s.

### 3.9 Determinismo (QSA top-k ordenado, MoE finalize)
- **Arquivo**: `files/patch_determinism.py`.
- **Mecanismo**: após `persistent_topk`, ordena os índices de bloco (−1 → INT32_MAX → ordena →
  volta), removendo a dependência da ordem de empates; MoE usa finalize não-fundido com cache de
  autotune próprio.
- **Efeito**: spread de logprob 0,25/6,07 nats → 0/0; greedy 5/5 distintos → 1/5; −4% decode/prefill.
- **Conceito**: não-associatividade do ponto flutuante; seleção top-k com empates.

### 3.10 Offload da tabela N-gram (PLE) para CPU
- **Arquivos**: `files/patch_ple_offload.py`, `files/ple_offload/*` (connector/worker/protocol/
  ple_offload_layer), `files/build_ple_packed_table.py`.
- **Conceito arquitetural**: o N-gram embedding tem **endereçamento determinístico** (os índices
  dependem só dos ids de entrada), então dá para calcular os índices e buscar as linhas em outro
  processo/memória, sobrepondo com a camada 1 (tech report §2.3; `CpuGpuSemaphore` com
  `cuStreamWaitValue32`, `ple_offload_layer.py:1-18`).
- **Problemas no GB10**: `CAN_USE_STREAM_MEM_OPS = 0` → o wait retorna sucesso mas bloqueia o
  host, e dentro de CUDA graph não espera → deadlock; linhas NVFP4 carregavam só códigos (80 B) em
  vez de códigos+escalas (90 B); placeholder GPU não sabia que era NVFP4 → "gibberish fluente"
  (`HANDOFF:23-39`).
- **Mecanismo**: handshake host-side com número de sequência em memória compartilhada; tabela
  empacotada `[total_rows, 90] uint8` mmap (page cache evictável).
- **Efeito**: TP1 viável (685.825 tokens KV); no 2-Spark **não recomendado** (RAM insuficiente, e em
  memória unificada não libera nada).

### 3.11 FP8-dense (checkpoint híbrido experimental)
- **Arquivos**: `files/fp8dense/*` (conversor streaming, verificação, stats), `files/overlay/*.py`
  (+ `.diff`).
- **Motivação arquitetural**: tudo que não é expert roteada é BF16 e grande parte é **replicada**
  por GPU (HC 1,19 GiB replicada, lida inteira a cada passo). O FP8 oficial da Qwen **não** ajuda:
  mantém GDN/atenção/HC/lm_head em BF16 e só faz experts FP8 (mais bytes que NVFP4) —
  `fable5-1-report.md:114-119`.
- **Mecanismo**: 591 lineares densas → FP8 E4M3 com escala **por canal de saída** e ativação
  dinâmica **por token** (sem calibração; funciona com as projeções rank-320 do HC e com o shared
  expert de 320 linhas por GPU, que não são múltiplos de 128). Ficam BF16: embeddings, roteadores,
  shared_expert_gate, indexador, `in_proj_a/b`, conv1d, normas, projeções do PLE, camada MTP, visão.
  Overlays: `hyperconnection.py`/`model.py`/`mtp.py` passam a aceitar `quant_config` (antes
  `quant_config=None` fixo), `modelopt.py` despacha `FP8_PER_CHANNEL_PER_TOKEN`/`FP8_PB_WO`.
- **Efeito**: RMSE relativo 2,40–2,68% (cosseno ≥ 0,9996); ganho **analítico** ~1,5× batch-1 —
  **não medido em GPU**.

### 3.12 Correção do YaRN (override aninhado)
- **Arquivo**: `start.sh:919-946`; `CHANGELOG.md:443-451`.
- **Problema**: `--hf-overrides '{"rope_parameters":…}'` no topo era `setattr` no config pai
  (`ModelConfig._apply_dict_overrides` só recursa em configs aninhados) e nunca chegava a
  `text_config`; verificado no container: `text_config.rope_parameters.rope_type` continuava
  `default`. Todo "1M" anterior rodou com RoPE **não escalada**.
- **Mecanismo**: payload único `{"text_config":{"ple_embedding_dtype":…, "rope_parameters":{
  "rope_type":"yarn","factor":F,"original_max_position_embeddings":262144}}}`.
- **Ponto de atenção (a verificar)**: pela mesma lógica de `_apply_dict_overrides` descrita no
  repo, `rope_parameters` é um dict simples dentro de `text_config` e seria **substituído inteiro**
  — o override do `start.sh` não repete `rope_theta: 10000000`, `partial_rotary_factor: 0.25`,
  `mrope_section: [11,11,10]`, `mrope_interleaved: true`. O card oficial passa o dict **completo**
  (`{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],
  "rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,
  "original_max_position_embeddings":262144}}}`). Se o vLLM não preencher os campos faltantes a
  partir de outras chaves, o θ/fração rotacional poderiam cair em defaults errados. O repo já
  marca 1M como "unvalidated"; ao recriar a receita, use o dict completo do card.

### 3.13 Checkpoint abliterado (`ABLIT=1`)
- `drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47`: "house residual-writer
  projection" em **12 tensores `self_attn.o_proj`** das camadas QSA 3…47; GDN `out_proj`, experts,
  MTP, PLE, visão e chat template intactos. Conceito: abliteração = remover do que uma camada
  **escreve no resíduo** a componente ao longo de uma direção (de recusa) — aqui aplicada só aos
  escritores das camadas de atenção global. Seu `config.json` rotula as experts MTP como
  `FP8_PB_WO` (tratado em §3.2). `README.md:245-300`.

### 3.14 Outras mudanças de runtime com efeito arquitetural
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` na lane v0.30 (vLLM 0.30 liga para Qwen4Exp; variância 10-18%
  no GB10) — `CHANGELOG.md:20-25`.
- `--mm-encoder-tp-mode data` (§2.7).
- `--mamba-ssm-cache-dtype bfloat16` (§2.7, §4.4).

---

## 4. Números e benchmarks

### 4.1 Contagem de parâmetros (calculada dos cabeçalhos do checkpoint nvidia)
| Categoria | Parâmetros | Bytes no checkpoint |
|---|---|---|
| Experts roteadas (48 × 512 × 3 × 640 × 2560) | **120,80 B** | 56,25 GiB (FP4) + 7,03 GiB (escalas) |
| GDN (36 camadas × ~57,96M) | 2,087 B | 3,886 GiB |
| Gated Residual (97 módulos × ~6,6M) | 0,641 B | 1,193 GiB |
| embed_tokens | 0,636 B | 1,184 GiB |
| lm_head | 0,636 B | 1,184 GiB |
| QSA atenção (12 × ~49,8M) | 0,598 B | 1,113 GiB |
| Shared experts (+gate) | 0,236 B | 0,440 GiB |
| Roteadores | 0,063 B | 0,117 GiB |
| PLE projeções/normas/conv | 0,033 B | 0,061 GiB |
| QSA indexador | 0,020 B | 0,037 GiB |
| **Modelo principal (sem visão, sem PLE, sem MTP)** | **125,74 B** → "125B" | |
| Tabela N-gram (320.001.536 × 160) | **51,20 B** → "51B" | 47,68 GiB (FP8) |
| MTP (experts 2,517 B + resto 0,091 B) | 2,61 B | 2,51 GiB |
| Visão | 0,449 B | 0,836 GiB |
| **Total armazenado** | 180,0 B | 123,53 GiB (132,64 GB) |

- **"6B ativos"** (reconstrução, inferência): todo o denso exceto embed/lm_head (3,68 B) + 10
  experts × 48 camadas × 4,915M (2,36 B) = **6,03 B**. Com embed + lm_head: 7,3 B.
- **"4B MTP"** (inferência): 2,61 B dos tensores `mtp.*` + embed e lm_head que o MTP usa
  (1,27 B) ≈ 3,88 B ≈ 4B.
- Tamanhos primos das 16 cabeças N-gram (calculados com `_find_nth_prime_after(19.999.999, h+1)`):
  20.000.003, 20.000.023, 20.000.033, 20.000.047, 20.000.059, 20.000.063, 20.000.069, 20.000.077,
  20.000.081, 20.000.093, 20.000.107, 20.000.147, 20.000.153, 20.000.159, 20.000.161, 20.000.171;
  soma 320.001.446 → pad 128 → **320.001.536** = 128 shards × 2.500.012 (confere exatamente com o
  checkpoint).
- Multiplicadores do hash para `ple_layer_index` 0, `seed` 1234 (recalculados):
  `[23703573157769, 20109073645365, 8052911324071]` (o checkpoint também traz o buffer).

### 4.2 Memória por GPU (TP2 + EP), medida vs estimada
Medido (`README.md:345-356`, container a GMU 0.835, `gpu_worker.py:919`):

| Linha | GiB |
|---|---|
| Memória total visível ao CUDA | 121,69 |
| Livre no início | 108,79 |
| Orçado (GMU 0.835) | 101,61 |
| Pesos (`model_runner.py:407`) | 64,46 |
| Pesos + não-torch | 68,52 |
| Pico de ativação | 1,07 |
| CUDA graphs | 0,54 |
| **KV cache** | **32,02** |

Outras medições: 62,72 GiB/nó sem MTP e 64,3 GiB com MTP (`README.md:662,707`).
Estimativa minha a partir dos bytes do checkpoint (TP2 divide densos shardáveis, EP divide experts,
HC/indexador/roteador/visão replicados): experts 31,64 + PLE 23,84 + GDN 1,94 + QSA 0,56 +
indexador 0,04 + HC 1,19 + embed 0,59 + lm_head 0,59 + shared 0,22 + roteador 0,12 + PLE proj 0,06
+ visão 0,84 = **61,6 GiB**; + MTP (experts 1,17 + resto ~0,17) ≈ **63,0 GiB** — coerente com os
62,7–64,5 medidos. README resume: "experts 34 GB + PLE ~25 GB + bf16 dense ~10 GB" (`:362-364`).

### 4.3 KV e estado por token/requisição
- KV principal (13 camadas: 12 QSA + 1 MTP): `13 × 2 (K,V) × 2 heads × 256 × 2 B = 26.624 B/token`
  bf16 (13.312 B/GPU em TP2); fp8: 13.312 B/token (6.656 B/GPU) — `README.md:371-374`.
- Medido: fp8 → 32,02 GiB / 3.652.200 tokens = **9.418 B/token/GPU** (41% acima do puro); bf16 →
  15.192 B/token/GPU (14% acima). O excesso vem do indexador/compressor BF16 e do estado
  GDN/PLE + arredondamento de página (`README.md:376-381`). No single-Spark: 29.294 B/token, KV
  principal = 84% (`patch_qsa_fp8_kv.py:14-17`); HANDOFF deriva 29.482 B/token (`:151-153`).
- Indexador (estimativa): chave comprimida 1×128 BF16 por bloco de 4 tokens = 64 B/token/camada;
  o pré-indexador também escreve chaves brutas (`fable5-1-report.md:32`); os kernels de bench usam
  cache comprimido `[pages, 1600/4, 1, 128]` (`bench_qsa_kernels.py:52-54`).
- **Estado GDN** por requisição: por camada e por V-head, 128×128. TP2: 24 heads × 128×128 × 4 B =
  **1,5 MiB/camada** fp32 (0,75 MiB bf16); ×36 = 54 MiB fp32 / 27 MiB bf16 por requisição por GPU
  (`fable5-1-report.md:86` "36 × 1.5 MiB fp32"). TP1: 108 MiB fp32 → "~0,23 GB por sequência por
  passo" lidos+escritos (`.env.sample:70-73`).
- Estado de conv do GDN: 10240 canais × (3 + nº especulativos) — em TP2 com MTP3: 5120 × 6 × 2 B =
  60 KiB/camada (cálculo meu).
- Estado do PLE (replicado, `is_kv_cache_tp_replicated = True`): conv 10240 × ((4−1)×3 + 3 spec) × 2 B
  ≈ 240 KiB + 2 ids de contexto por requisição (cálculo meu, `ple_layer_patched.py:1071-1088,1581`).
- **Tamanho de bloco da atenção** (vLLM iguala a página de atenção à página mamba):
  página GDN TP2 = recorrente + conv. Cálculo (meu), batendo com os números do repo:

  | KV | SSM | bytes/token/GPU (KV principal) | página GDN | bloco |
  |---|---|---|---|---|
  | bf16 | fp32 | 1024 | 1.634.304 B | **1600** (README `:380-381,483`) |
  | fp8 | fp32 | 512 | 1.634.304 B | 3200 |
  | fp8 | bf16 | 512 | 847.872 B | **1664** (`patch_block_drop.py:7-8`) |
  | TP1 fp8 | fp32 → bf16 | 1024 | 3.268.608 → 1.695.744 | **3200 → 1664** (`.env.sample:74`) |

  Consequência: o prefix cache só acerta em múltiplos do bloco (`fable5-1-report.md:288-289`).

### 4.4 Capacidade de KV medida
| Config | Pool | Tokens | Concorrência @262k | Fonte |
|---|---|---|---|---|
| fp8 KV, SSM fp32, GMU 0.835 | 32,02 GiB | 3.652.200 | 13,93× | `README.md:311-322` |
| bf16 KV, GMU 0.835 | 31,70 GiB | 2.131.159 | 8,13× | idem |
| bf16 KV, MTP off | — | 2.663.445 | 10,16× | `README.md:708` |
| bf16 KV, SSM bf16 | — | 2.426.193 | 9,26× | `docs/bench/rerun-20260916.md` |
| fp8 KV, SSM bf16 | — | 4.378.829 | 16,70× | idem |
| fp8 KV, SSM bf16 (2026-09-11) | — | 4.241.167 / 4.238.238 | — | `CHANGELOG.md:189-190` |
| vLLM 0.30, bf16, GMU 0.80 | — | 1.467.072 / 1.340.476 / "1.52M" | 5,6× | `CHANGELOG.md:53-54`, `README.md:857,85` |
| Official FP8 (`start-fp8.sh`) | — | ~500k | — | `README.md:99-101` |
| TP1 single-Spark | — | 685.825 | — | `HANDOFF:11` |

Regra de bolso: 1 GiB de pool ≈ 71–74k tokens bf16 (`README.md:426-428`); +0,01 GMU ≈ +0,83 GiB ≈
+95k tokens fp8 (`.env.sample:56`).

### 4.5 Decode e MTP
- Modelo de bytes por passo (batch 1, MTP3, por GPU; `fable5-1-report.md:74-99`, checkpoint
  RadixArk): lm_head 1+3 leituras 2,36 GiB (26%); experts ~2,1 (23%); GDN 1,95 (21%); HC replicado
  1,19 (13%); atenção+indexador 0,60 (7%); MTP ~0,5 (5%); roteador+shared+PLE 0,40 (4%); estado GDN
  0,1 (1%) → **≈9,2 GiB**; a 235–245 GB/s ≈ 40–42 ms/passo; com ~2,2 tokens aceitos/passo ≈
  53–56 tok/s = o medido. "Batch-1 decode on this box is therefore at the memory-bandwidth roofline".
- ~106 coletivas NCCL por passo (36 GDN + 12 atenção + 48 MoE + gather de logits + 9 do rascunho),
  20 KB cada, 2–3 ms/passo (`:385-394`).
- MTP (2026-09-05, TP2+EP, 262k, GMU 0.835, greedy): MTP off 24,5 tok/s → MTP=3 **52,1 tok/s
  (2,13×)**; aceitação 72,8% (823/1131); por posição **89% / 74,5% / 60%** (`README.md:700-709`).
- HANDOFF: aceitação oscila 62,8–92,6% (comprimento aceito 2,88–3,78 de 4) conforme o conteúdo;
  cópia de contexto ~70 tok/s vs prosa ~40 (`:190-195`).
- Tech report Tab.4 (4 passos especulativos): comprimento aceito médio 4,06 (full attn) / 4,07 (QSA
  com reuso de índices) — MT-Bench 3,47, GSM8K 4,20, MATH 4,30, HumanEval 4,26, MBPP 4,13.
- Vocab de rascunho 47k: +9,6% (2026-09-11) / +8,4% (2026-09-16); tabelas completas em `README.md:737-748`.
- vLLM 0.30 lane (2026-09-27, BF16 KV): ×1 prosa 59,2, código 66,6, estruturado 76,0; ×8 código
  313,6 tok/s agregado (`README.md:855-872`). Lane 2026-09-25: código S=8 441,2 tok/s.
- Dia-0 (fp8 KV + vocab 65k balanceado): ×1 54,4 tok/s, ×8 207,0 agregado (`README.md:896-906`).

### 4.6 Prefill / contexto longo
- Prefill ~2,96k tok/s de 16k a 64k, −8% em 128k (2.727 tok/s): "QSA keeps attention from dominating
  at long context, so TTFT stays close to linear" (`README.md:912-923`).
- vLLM 0.30: 16k 3.489,8 tok/s; 128k 2.897,6 (mais rápido que 64k, não explicado) (`README.md:874-883`).
- HANDOFF (checkpoint antigo, 1M com RoPE **não escalada** na época): 600k a 1.751 tok/s, agulha a
  95% de profundidade correta; "19× context costs only 1.64× in per-token rate" (`:155-172`).
- Tech report (kernel, 1M tokens): QSA **7,6×** mais rápido que GQA denso (FlashInfer) no prefill e
  **4,9×** no decode (BS 4, next_n 4); indexador com r=4 3,8× (prefill) e 4,4× (decode) mais rápido
  que r=1 (Fig. 6).
- Tech report RULER: ≤128K 99,89 vs 99,84; 512K–1M **93,00 vs 90,08** (QSA vs full); MRCR 8-agulhas
  512K 40,53 vs 30,66, 1M 26,44 vs 20,71 (Tab. 3).

### 4.7 Qualidade por quantização e ablações
- Card nvidia (T=1.0, top_p 0.95, 131.072 tokens, `reasoning_effort=xhigh`), FP8 vs NVFP4: GPQA
  92,0/91,5; HLE 34,7/35,4; τ²-Telecom 90,8/90,1; MMMU Pro 77,1/78,3; SciCode 16,3/18,8; AA-LCR
  71,9/74,1; IFBench 80,5/81,0; Omniscience 28,1/27,6; Terminal-Bench 2.1 83,3/82,9.
- RadixArk: GSM8K 97,27% (1283/1319), AIME26 pass@1 98,75% (`fable5-1-report.md:419-423`).
- FP8-dense: RMSE relativo 0,0240/0,0262/0,0268 (min/mediana/max) em 591 tensores.
- Tech report, arquitetura (25B-A3B, 28 camadas, média de 9 benchmarks): Full attention 49,87; SWA
  hybrid 51,15; **GDN hybrid 53,81** (Tab. 1). QSA vs full: média 76,8 vs 75,9 (Tab. 2); gap de loss
  ~10⁻⁴ (Fig. 4).
- GR (25B-A3B, 560B tokens): loss pré-norm 1,617 → mHC static 1,596 → mHC dynamic 1,594 → **GR
  1,590**; média 50,91 → 52,49 → 54,47 → **54,66** (Tab. 5).
- N-gram, posição (Tab. 7): sem N-gram loss 1,585 / média 45,44; na 2ª camada 1,541 / **47,94**
  (melhor média); múltiplas camadas não ajudam. Vocab (Tab. 9): 20× 1,553; 50× 1,541; 100× 1,534;
  200× 1,526 — loss cai monotonicamente, benchmarks saturam; chinês melhora consistentemente.
  Tabela atual: 320M linhas ≈ 1.289× o vocab base de 248k em linhas, mas com dim 160 (não 2560).
- Estabilidade (Tab./Fig. 10-12): a 4× o LR ótimo, AdamW+Qwen3.5 tem 183 spikes/10k passos; Muon+GR
  zero spikes; GatedNorm isolado reduz spikes 32,0 → 3,2 /10k.

---

## 5. Conceitos necessários (em ordem de pré-requisito)

### Bloco A — Fundamentos de ML/DL
1. **Tensores, matrizes e produto matricial (GEMM)** — Um modelo de linguagem é quase só
   multiplicações de matriz `y = W·x`. Aqui praticamente todo GEMM de decode tem K = 2560
   (`hidden_size`) e M = nº de tokens no passo (1–64); é por isso que decode é limitado por banda
   (ler W custa mais que multiplicar).
2. **Rede neural, parâmetros, função de perda, gradiente** — Parâmetros são números aprendidos
   minimizando uma perda (entropia cruzada do próximo token). "125B parâmetros" = 125 bilhões desses
   números; cada um ocupa 2 B em BF16, 1 B em FP8, ~0,56 B em NVFP4.
3. **Otimizadores (AdamW, Muon)** — Regras de atualização dos pesos. O tech report usa Muon
   (ortogonaliza o momento com 8 iterações Newton–Schulz) em matrizes "lineares" e AdamW em
   embeddings/roteador/projeções low-rank do GR; isso explica estabilidade de treino, não muda a
   inferência.
4. **Ativações (SiLU, GELU, sigmoid, softplus, ReLU)** — Não-linearidades. SiLU nas experts e na
   conv do GDN; sigmoid em todos os *gates* (GDN, atenção, GR, PLE, shared expert); softplus no
   decay do GDN; ReLU no scorer do indexador QSA; GELU-tanh no ViT.
5. **Normalização (LayerNorm, RMSNorm, zero-centered RMSNorm, RMSNorm em grupos)** — Reescalam o
   vetor para magnitude ~1. Aqui toda RMSNorm é "zero-centered": peso inicializado em 0 e aplicado
   como `(1+w)` (modeling `:151-171`); a do GR/PLE é **em grupos** de 2560 (normaliza cada ramo
   separadamente). O ViT usa LayerNorm com bias.
6. **Conexão residual e pré-normalização** — `x ← x + F(Norm(x))` permite redes profundas. O Qwen4-Exp
   **remove** a pré-norm explícita: o gate de leitura do GR já normaliza (tech report §2.2).
7. **Gating (portas multiplicativas)** — Multiplicar um sinal por `σ(·)` ∈ (0,1) decide "quanto deixar
   passar" e limita magnitudes. É o tema recorrente da arquitetura; o tech report atribui a
   estabilidade (zero spikes) aos gates.
8. **Convolução 1D causal, depthwise, dilatada** — Cada canal filtra só o passado recente com um
   kernel pequeno. GDN: kernel 4 sobre 10240 canais (Q,K,V). PLE: kernel 4 com **dilatação 3**
   (olha t, t−3, t−6, t−9) sobre 10240 canais, inicializada em zero.

### Bloco B — Transformer e LLM
9. **Tokenização (BPE byte-level) e vocabulário** — O texto vira ids de um vocabulário de 248.320
   (Qwen2Tokenizer). Os ids alimentam o `embed_tokens` e também o **hash do N-gram**.
10. **Embedding e lm_head (desamarrados)** — Tabela id→vetor 2560 na entrada e projeção 2560→248.320
    na saída; `tie_word_embeddings: false` → duas matrizes de 1,18 GiB. O lm_head é o tensor mais lido
    no decode com MTP.
11. **Modelo autoregressivo, prefill vs decode** — Prefill processa o prompt inteiro em paralelo
    (compute-bound); decode gera 1 token por passo (memory-bound). Todos os truques da receita
    miram bytes por passo de decode.
12. **Self-attention (Q, K, V, softmax)** — Cada token compara sua query com as keys de todos os
    anteriores e mistura os values. Custo O(n²) no prefill e KV cache O(n) — o problema que GDN e
    QSA atacam.
13. **Multi-head, GQA e MQA** — Várias cabeças em paralelo; em GQA várias queries compartilham K/V
    (QSA: 24 Q / 2 KV → 12:1); em MQA todas compartilham uma (indexador: 4 Q / 1 K).
14. **KV cache** — Guardar K e V dos tokens passados para não recomputar. Só as 13 camadas QSA têm
    KV por token; 26.624 B/token em bf16. As 36 GDN têm estado fixo por requisição.
15. **Codificação posicional RoPE, RoPE parcial, NoPE** — Gira pares de dimensões de q/k por ângulos
    proporcionais à posição; o produto q·k passa a depender da distância relativa. Aqui só 64 das 256
    dims giram (`partial_rotary_factor 0.25`), o resto é conteúdo puro; o tech report mostra que
    NoPE total causa geração sem fim após o pós-treino.
16. **mRoPE (Multimodal RoPE) intercalado** — A posição tem 3 componentes (tempo, altura, largura);
    as 32 frequências são repartidas 11/11/10 e intercaladas. Para texto as três são iguais e vira
    RoPE comum; para imagem/vídeo codifica a grade 2D/3D.
17. **Extensão de contexto: interpolação de posição, NTK, YaRN** — Para ir além do treinado, YaRN
    divide por `s` só as frequências cujos comprimentos de onda excedem o contexto original, mantém
    as altas e aplica uma rampa entre elas, e corrige a temperatura da atenção (`mscale = 0,1·ln s + 1`
    = 1,139 para s=4). Aqui: fator 4, 262.144 → ~1M; "YaRN estático" pode piorar textos curtos
    (card), por isso o repo desliga abaixo de 262k. Com θ = 10⁷ e 64 dims, os pares 0–14
    (λ < 8.192) ficam intactos, 15–21 na rampa e 22–31 (λ > 262k, até ~38M) são interpolados
    (cálculo com β_fast = 32, β_slow = 1, defaults usuais).
18. **Atenção com gate de saída (Gated Attention)** — A saída da atenção é multiplicada elementwise
    por `σ(W_g x)` antes do `o_proj`. No Qwen4-Exp o gate sai da metade extra do `q_proj`
    ([12288] = 24×256×2), por head e por dimensão (modeling `:866-897`). Remove "attention sinks" e
    limita magnitude.
19. **QK-Norm** — RMSNorm por head em q e k antes do RoPE (`q_norm`, `k_norm` [256]). Estabiliza os
    logits da atenção (dispensa qk-clip).

### Bloco C — Atenção linear e SSMs (as 36 camadas GDN)
20. **Atenção linear como memória de pesos rápidos** — Troca softmax por um estado matricial
    `S ∈ R^{d_k×d_v}` atualizado a cada token; leitura `y = Sᵀq`. Custo O(n) e estado de tamanho fixo
    (128×128 por V-head), independente do contexto.
21. **Delta rule** — Em vez de só somar `k vᵀ`, primeiro lê o que o estado já associa a `k`
    (`Sᵀk`), calcula o erro `e = v − Sᵀk` e escreve só a correção `β k eᵀ`. Evita acúmulo ilimitado
    para chaves repetidas.
22. **Gated DeltaNet (GDN)** — Delta rule + decay dependente dos dados:
    `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`, `y_t = S_tᵀ q_t`, com
    `α_t = exp(−exp(A_log)·softplus(W_a x + dt_bias))` (via `in_proj_a`, `A_log`, `dt_bias`) e
    `β_t = σ(W_b x)` (`in_proj_b`). q,k = L2Norm(SiLU(ShortConv(W x))), v = SiLU(ShortConv(W_v x)), q
    escalado por 1/√128; saída `W_o[σ(W_z x) ⊙ RMSNorm(y)]` (tech report eqs. 1–11; modeling
    `:464-624`). 16 heads QK servem 48 heads V (cada QK é repetido 3×).
23. **Forma chunked (WY/UT transform) vs recorrente** — Prefill processa blocos de 64 tokens com uma
    transformação triangular (matmuls em paralelo); decode usa a recorrência token a token. No GB10:
    prefill Triton/FLA `chunk_gated_delta_rule`, decode CUDA `fused_gdn_decode_post_conv_mtp`
    (fp32 state, ≤ 8 tokens especulativos) — `fable5-1-report.md:33-35`.
24. **Estado "mamba" no vLLM** — O vLLM trata GDN (e o conv do PLE) como camadas tipo Mamba: um
    slot de estado por requisição, e páginas do mesmo tamanho que as de atenção. Daí o bloco de
    atenção de 1600/1664/3200 tokens e o `--mamba-ssm-cache-dtype`.
25. **Híbrido linear + global** — GDN comprime, a atenção global (1 em 4) faz recuperação exata de
    tokens; o tech report mostra que o GR entrega saídas antigas justamente às camadas de atenção.

### Bloco D — Atenção esparsa (as 12 camadas QSA)
26. **Atenção esparsa com indexador (DSA)** — Um módulo barato estima a importância dos tokens
    anteriores e a atenção cara roda só nos top-K. DSA (DeepSeek) faz isso por token, com indexador
    O(n²).
27. **QSA (Qwen Sparse Attention)** — Indexador **comprimido**: MQA com 4 heads de query (RMSNorm +
    RoPE parcial 64/128 na posição do token) e 1 chave compartilhada; as chaves são agrupadas em
    blocos de r = 4 tokens, média (AvgPool), RMSNorm e **depois** RoPE na posição inicial do bloco
    (não mistura fases rotacionais). Score `I_ib = Σ_h ReLU(⟨q_i^h, k̄_b⟩)` (÷√128 na referência),
    só para blocos completos (`p_b + r − 1 ≤ i`); top-512 blocos → 2048 tokens + os até 3 tokens do
    bloco incompleto final → atenção GQA normal só sobre esses (modular `:368-474`, tech report
    eqs. 12–16, 19). Custo do indexador O(n²/r).
28. **Treino do indexador por destilação** — Estágio 1: congela o backbone e treina o indexador
    (1.000 passos, LR 1e-3, 8×256K) com KL contra a distribuição de atenção densa max-pooled nos
    blocos; estágio 2: treino esparso conjunto (8.000 passos, LR 2,5e-5, 96×256K ≈ 200B tokens).
    Explica por que a esparsidade quase não perde qualidade.
29. **Por que KV FP8 é arriscado em atenção esparsa** — Quantizar chaves altera os scores do
    indexador e portanto **quais blocos** são escolhidos, não só a saída; o repo mede 12/12 mas com
    tok/step menor em 8/8 células.
30. **Split-K / flash-decoding e tiles Triton** — Decode divide o comprimento das keys em partes
    processadas em paralelo e depois mescladas; o nº de partes deve casar com o nº de SMs (48 no
    GB10 vs 160 no GB300) — `QSA_PROFILE`.

### Bloco E — MoE
31. **Mixture-of-Experts** — A FFN vira N "experts" e um roteador escolhe k por token; parâmetros
    totais ≫ ativos. Aqui 512 experts, top-10 (1,95% ativos) → 120,8B params, ~2,36B ativos.
32. **Roteador softmax + top-k + renormalização** — `probs = softmax(W_r x)` em fp32, top-10, divide
    pela soma (`norm_topk_prob`); saída `Σ w_e E_e(x)` (modeling `:959-999`).
33. **Shared expert com gate** — Um expert sempre ativo (intermediário 640) multiplicado por
    `σ(w_s·x)` escalar; captura conhecimento comum.
34. **SwiGLU** — `down(SiLU(gate(x)) ⊙ up(x))`; nas experts, gate e up ficam fundidos
    (`gate_up_proj` 1280×2560 no transformers; no checkpoint separados por expert).
35. **Load-balancing loss** — `router_aux_loss_coef` 0,001 no treino; evita experts ociosos.
36. **Expert Parallelism vs Tensor Parallelism em MoE** — EP distribui experts inteiras (256 por GPU)
    e mantém a largura 640; TP corta cada expert (640→320). A escolha interage com restrições de
    bloco de quantização (FP8 128×128 do MTP) e de kernel.

### Bloco F — Resíduo alargado (Gated Residual)
37. **Hyper-Connections / mHC / AltUp** — Em vez de um vetor residual, n ramos paralelos com
    operadores de leitura (H_mix), escrita (H_combine) e mistura (H_res) aprendidos; mHC restringe
    H_res a matrizes duplamente estocásticas.
38. **Gated Residual (GR)** — Variante do Qwen: 4 ramos, **leitura elementwise** com gate low-rank e
    **escrita por escalar por ramo**, **sem H_res** (tech report eqs. 29–34):
    `R̂_i = RMSNorm(R_i; γ_i)`; `G = σ(W_up SiLU(W_down vec(R̂)/4))` ∈ R^{4×2560} (posto 320);
    `x = (1/4) Σ_i G_i ⊙ R̂_i`; `s = 2σ(W_inj vec(R̂)/4)` ∈ R^4; `R_i ← R_i + s_i·y`
    (modeling `:1002-1037`, decoder `:1272-1309`). Um GR antes do mixer de tokens, outro antes do
    MoE, e um "mixer" final (só leitura) no lugar da norma final.
39. **GatedNorm** — `RMSNorm(u) ⊙ σ(W2 SiLU(W1 RMSNorm(u)))`; é exatamente a leitura do GR; o tech
    report mostra que ela remove outliers de ativação e spikes de loss.
40. **Custo de inferência do resíduo largo** — Ler/escrever 4×2560 por token por subcamada é tráfego
    de memória; no vLLM o GR é BF16 **replicado** nas duas GPUs (1,19 GiB lidos por passo, 13%).
    O paper menciona armazenar o estado residual em FP8 e kernels fundidos de leitura/escrita.

### Bloco G — N-gram embedding ("PLE")
41. **Embedding como eixo de escala de capacidade** — Tabelas de lookup adicionam parâmetros sem
    FLOPs (só leitura de linhas). Gemma 3n introduziu "Per-Layer Embeddings" (PLE) offloadáveis; o
    Qwen herdou o **nome** (`ple_*`, docstring "Per-Layer Embedding" no config; docstring do vLLM diz
    "position-learning enhancement"), mas o **mecanismo** é o de n-gramas com hash (Engram).
42. **N-gramas com hash multi-cabeça (Engram)** — Para cada token, o bigrama (t−1,t) e o trigrama
    (t−2,t−1,t) viram índices: `h = (t·m0) XOR (t₋₁·m1) [XOR (t₋₂·m2)]`, `idx = (h mod p_{n,k}) +
    offset_{n,k}` para 8 cabeças por ordem (16 no total), cada uma com seu primo ≥ 20M; 16 linhas de
    160 dims concatenadas → e_t ∈ R^2560 (modular `:608-708`; `ple_layer_patched.py:409-665`). Hash
    com colisões é aceitável porque múltiplas cabeças independentes desambiguam.
43. **Segmentação por EOS** — n-gramas não atravessam `<|endoftext|>` (248044); posições sem contexto
    são preenchidas com o id EOS; o contexto dos 2 últimos ids fica no cache (conv state 2).
44. **Gate de contexto por ramo + conv curta dilatada** — `k_i = RMSNorm_i(W_K e)` (4 chaves, W_K
    2560→10240), `v = W_V e` (compartilhado), `q_i = RMSNorm_i(R_i)`; `z_i = ⟨k_i,q_i⟩/√2560`;
    `g_i = σ(sign(z_i)·√|z_i|)`; `u_i = g_i·v`; saída `u + SiLU(Conv1d_dil3(RMSNorm(u)))`, somada aos
    4 ramos **antes** da leitura do GR da camada 2 (modular `:711-783,811-814`). O gate "decide se
    confia na memória" dado o contexto; a conv (init zero) adiciona contexto lexical local.
45. **Por que na camada 2 e por que offloadável** — Endereçamento determinístico permite calcular os
    índices e prefetchar do host durante a camada 1 (tech report §2.3.1). No GB10 (memória
    unificada) o offload não economiza memória física — só pressão de carga.

### Bloco H — MTP e decodificação especulativa
46. **Decodificação especulativa** — Um rascunho barato propõe k tokens; o alvo verifica todos em
    um passo (paralelo) e aceita o prefixo correto; greedy → saída idêntica, amostragem → rejection
    sampling preserva a distribuição. Ganha porque verificar 4 tokens custa quase o mesmo que 1 em
    decode limitado por banda.
47. **MTP (Multi-Token Prediction) como rascunho** — Uma camada extra treinada para prever o token
    t+2 a partir do estado do alvo em t e do embedding do token t+1: `pre_fc_norm_embedding`,
    `pre_fc_norm_hidden` (10240 = 4 ramos), `fc_embedding`/`fc_hidden` 2560×2560, uma camada QSA+MoE
    com GR próprios, mixer, lm_head (compartilhado/copiado). A combinação exata de `fc_hidden` com os
    4 ramos não está no transformers (que ignora `mtp.*`, modeling `:1321`) — (a verificar no
    `nvidia/mtp.py` da imagem). Aplicada recursivamente 3×.
48. **Taxa de aceitação e comprimento aceito** — Medidas por posição (89/74,5/60%); a queda por
    posição é esperada e serve de teste de sanidade de quantização.
49. **Reuso de índices QSA entre passos do rascunho** — `index_share_for_mtp_iteration`: passos 2–3
    reaproveitam o top-k do passo 1 (índices "um token atrasados"); o paper segue o GLM-5.
50. **Vocabulário reduzido de rascunho (FR-Spec)** — Argmax só em ids frequentes; o erro vira
    rejeição; corta bytes da cabeça.

### Bloco I — Multimodal
51. **Vision Transformer, patches e merger** — Imagem/vídeo em patches 16×16×2 (Conv3d) → 27 blocos
    ViT (1152, 16 heads, RoPE axial 2D + pos-embed 48×48 interpolado) → merger 2×2 (4608→2560) →
    tokens injetados no lugar de `<|image_pad|>`. 1 token visual ≈ 32×32 px.
52. **mRoPE para posições visuais** — `get_rope_index` atribui (t,h,w) aos tokens visuais e
    posições 1D ao texto.

### Bloco J — Quantização
53. **Formatos de ponto flutuante (FP32, BF16, FP8 E4M3, FP4 E2M1)** — Bits de sinal/expoente/
    mantissa definem faixa e precisão: BF16 1/8/7; E4M3 1/4/3 (máx 448); E2M1 1/2/1 com valores
    {0, 0,5, 1, 1,5, 2, 3, 4, 6} (`_FP4_VALUES`, `ple_layer_patched.py:304`).
54. **Escalas: por tensor, por canal, por bloco, dupla escala** — Número baixo-bit × escala. PLE:
    uma escala para a tabela toda (per-tensor). MTP experts: uma por bloco 128×128. FP8-dense: uma
    por canal de saída + ativação por token. NVFP4: uma escala FP8 por bloco de 16 **e** uma FP32
    global (`weight_scale_2`).
55. **NVFP4 (W4A4)** — Pesos e ativações em FP4 E2M1, blocos de 16, escala de bloco E4M3, escala
    global FP32; ~4,5 bits/peso efetivos. `w = código × weight_scale[bloco] × weight_scale_2`;
    ativação usa `input_scale` estático calibrado. Aplicado só às experts roteadas (96% dos params).
    Kernel no GB10: FlashInfer CUTLASS SM120 grouped GEMM.
56. **MXFP8 (microscaling OCP)** — FP8 com blocos de 32 e expoente compartilhado; usado pelo
    checkpoint local-inference-lab em atenção/GDN/visão; restrições de forma do `mm_mxfp8`.
57. **FP8 em blocos 128×128 (estilo DeepSeek-V3)** — Escala por tile 2D do peso, ativação dinâmica
    em grupos de 128; `weight_scale_inv`. Exige larguras múltiplas de 128 após o particionamento.
58. **Quantização mista (ModelOpt MIXED_PRECISION)** — Cada módulo declara seu algoritmo
    (`quantized_layers`) e o resto fica BF16 (`ignore`); o loader precisa casar prefixos exatos
    (daí o alias MTP 0→48).
59. **Calibração (MSE) e PTQ** — Escalas escolhidas minimizando erro em dados de calibração
    (cnn_dailymail + Nemotron v2); sem fine-tuning.
60. **Quantização de KV cache** — FP8 para K/V principal com escalas por tensor; indexador BF16.
61. **Quantização do estado SSM** — fp32 → bf16 no estado recorrente GDN: metade dos bytes, risco de
    precisão em recorrências longas (o repo mediu agulhas 15/15 a 32k no single-Spark).

### Bloco K — Execução distribuída e engine (o que a arquitetura exige do vLLM)
62. **Tensor Parallelism (colunas/linhas, all-reduce)** — Cada GPU guarda metade das colunas de uma
    projeção e metade das linhas da próxima; soma parcial via all-reduce (36 GDN + 12 atenção + 48
    MoE por passo).
63. **Vocab-parallel embedding/lm_head** — Cada rank guarda um intervalo de ids (124.160 linhas);
    logits reunidos por all-gather — base do FR-Spec TP-aware e do PLE sharded (160M linhas por GPU).
64. **Módulos replicados** — HC, indexador, roteador, PLE projeções e visão (modo data) ficam
    inteiros em cada GPU: sem comunicação, mas sem economia de memória/banda.
65. **Memory-bound roofline** — tok/s ≈ banda / bytes por passo × tokens aceitos por passo; com
    ~9,2 GiB/passo e ~240 GB/s o teto é ~25 passos/s.
66. **Cache híbrido e páginas uniformes** — vLLM aloca um pool de blocos iguais para camadas de
    atenção (por token) e estado (por requisição); o bloco de atenção cresce até igualar a página
    GDN; prefix caching em modo `align`.
67. **CUDA graphs e custom ops** — Captura de sequências de kernels de decode (tamanhos 1–64 =
    8 seqs × 4 tokens × 2); o conv do PLE é registrado como custom op com fake impl para caber no
    grafo.

---

## 6. Receita passo a passo (visão arquitetural: do config ao servidor)

1. **Leia o `config.json` e desenhe o modelo.** Identifique: camadas (48 = 36 GDN + 12 QSA), onde
   está o PLE (`ple_layer_ids` 1-based → `layers.1`), MTP (1 camada, QSA), hidden 2560 × 4 ramos,
   MoE 512/top-10/640, vocab 248.320 desamarrado, visão 27×1152. *Por quê*: toda decisão de memória
   e kernel depende de quais camadas têm KV por token (13) e quais têm estado fixo (36 + PLE).
2. **Faça a contabilidade de parâmetros e bytes por componente** (tabela §4.1). *Por quê*: 96% dos
   parâmetros estão nas experts; a tabela N-gram (47,7 GiB FP8) é o segundo maior bloco; o denso
   BF16 é pequeno em parâmetros mas é lido inteiro a cada passo.
3. **Escolha o checkpoint pela precisão por componente.** nvidia: experts NVFP4, PLE FP8
   per-tensor, MTP experts FP8 128×128, resto BF16 (133 GB). local-inference-lab: MXFP8 no denso e
   PLE NVFP4 (99 GB, mais rápido, sem avaliação publicada). RadixArk: BF16 denso, PLE FP8, MTP
   BF16. *Por quê*: menos bits no denso = menos bytes por passo = mais tok/s; mais bits nas partes
   sensíveis (atenção, PLE) = mais fidelidade.
4. **Escolha o paralelismo.** 2 GPUs de 121,69 GiB → TP=2 para os densos; **EP** para experts
   (256 inteiras por GPU; mantém 640 de largura, compatível com os blocos FP8 128×128 do MTP); PLE
   vocab-parallel; HC/indexador/roteador replicados; ViT replicado (`--mm-encoder-tp-mode data`,
   porque 4304 não divide bem). Estime ~63 GiB de pesos por GPU.
5. **Garanta que cada formato de quantização tem um método de carga.** (a) PLE FP8: declare
   `text_config.ple_embedding_dtype` (`detect_ple_dtype.py` + `--hf-overrides`) e use o dispatch do
   PR #53899 (`ple_layer_patched.py`). (b) MTP: crie o alias `mtp.layers.48.*` nos dois arquivos de
   quantização e adicione o ramo `FP8_BLOCK_SCALES`/`FP8_PB_WO` → `Fp8MoEMethod` com bloco lido do
   `group_size`. (c) Se usar checkpoint com MXFP8, aplique o fallback de emulação para formas fora do
   envelope do `mm_mxfp8`. *Por quê*: nesses casos o vLLM falha silenciosamente (MoE não quantizado,
   embedding BF16 gigante) e só morre minutos depois.
6. **Decida o KV cache.** KV principal FP8 (patch nos kernels QSA: escalas após os dots,
   `constexpr`) dá 1,70× tokens; valide raciocínio longo porque as chaves quantizadas mudam a
   seleção do indexador. Indexador/compressor continuam BF16.
7. **Decida o dtype do estado GDN.** `--mamba-ssm-cache-dtype bfloat16` reduz bytes por passo e o
   tamanho do bloco de atenção (1664 em vez de 3200 com KV fp8), melhorando granularidade do prefix
   cache; valide com agulhas.
8. **Decida o contexto.** Nativo 262.144 sem YaRN para throughput/agente; 1M só com YaRN fator 4
   (ou 2 para 512K), `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, override **aninhado em `text_config` e
   com o dict `rope_parameters` completo** (θ 10⁷, 0,25, mrope `[11,11,10]` intercalado). Lembre que
   `MAX_MODEL_LEN` também dimensiona buffers do indexador.
9. **Configure o MTP.** `{"method":"mtp","num_speculative_tokens":3}` (1 camada reusada 3×), +
   `index_share_for_mtp_iteration: true` (reuso do top-k QSA), + vocabulário de rascunho reduzido com
   `use_local_argmax_reduction: true` e balanceamento entre ranks, + `disable_eagle_block_drop`.
   Monitore `spec_decode_num_accepted_tokens_per_pos` — a curva deve decair suavemente.
10. **Kernels e grafos.** `cudagraph_mode FULL_DECODE_ONLY`, `mode 0`; perfis QSA opcionais para
    48 SMs; determinismo opcional (top-k ordenado).
11. **Verifique no log.** `Available KV cache memory`, `GPU KV cache size … tokens`, block size da
    atenção, `PLE embedding … uses the runtime FP8 method`, `Routed experts mtp.layers.48… building
    them with Fp8MoEMethod`, backends (QSA Triton, GDN FLA/CUDA, MoE FlashInfer CUTLASS), e que
    `text_config.rope_parameters` tem o que você pediu.
12. **Valide qualidade e depois afine.** Aceitação por posição, reasoning/needle suite, NLL fixa;
    então itere bytes por passo (FP8-dense, vocab de rascunho), concorrência (`MAX_NUM_SEQS`) e
    memória (GMU/`--kv-cache-memory`), sempre medindo um braço por lançamento.

---

## 7. Pontos em aberto e discrepâncias encontradas
1. `README.md:216-219` (tabela `.env`) diz `MAX_MODEL_LEN` 1000000 e `YARN_ENABLE` true no sample;
   o `.env.sample:45-46` real tem 262144 e false (mudança de 2026-09, `CHANGELOG.md:405-407`).
2. `PLE_QUANT_OVERRIDE=fp8` (README/report) não existe no código atual.
3. O docstring do vLLM chama o módulo de "position-learning enhancement"; o transformers de
   "Per-Layer Embedding"; o card/paper de "N-gram Embedding". É o mesmo módulo.
4. `fable5-1-report.md:63` ("MTP layer routed experts 4.69 GiB bf16") refere-se ao RadixArk; no
   nvidia são FP8 (2,34 GiB).
5. `README.md:430-451` ("Default runtime") mostra `--kv-cache-dtype auto` e o override YaRN antigo
   no topo — captura histórica.
6. Override YaRN do `start.sh` substitui o dict `rope_parameters` sem repetir θ/fração/mrope
   (§3.12) — a verificar no container.
7. Estrutura exata da entrada do MTP (como `fc_hidden` 2560×2560 se combina com o estado de 4 ramos
   normalizado por `pre_fc_norm_hidden` [10240]) não é visível no repo nem no transformers.
8. O lm_head do drafter: o repo afirma que é um `ParallelLMHead` próprio (1,18 GiB); os pesos
   residentes medidos (+1,58 GiB com MTP) são compatíveis com experts MTP (1,17) + denso, sugerindo
   que a cópia pode ser compartilhada por referência — os bytes **lidos** por passo são o que conta.
9. Docstring do `patch_modelopt_mxfp8.py` atribui as formas MXFP8 ao RadixArk, mas o HANDOFF diz
   que o RadixArk tem atenção BF16 (o MXFP8 é do local-inference-lab).

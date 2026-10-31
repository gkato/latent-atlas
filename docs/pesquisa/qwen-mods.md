# Inventário — Qwen3.8-Flash-Next (receita dual DGX Spark): `files/**`, `html/concepts*`, `html/BRIEF.md`

> Repositório: `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` (clonado em
> `.../scratchpad/repos/Qwen3.8-Flash-Next-Dual-DGX-Sparks`). Todos os caminhos abaixo são relativos à raiz
> do repositório. Referências `arquivo:linha` apontam para o estado clonado em 2026-09-28.
>
> **Escopo deste documento:** tudo que está sob `files/` (todos os `patch_*.py`, `ple_layer_patched.py`,
> `ple_offload/**` incluindo `orig/`, `fp8dense/**`, `qsa_gb10/**`, `overlay/**`, `draft_vocab_en_code_47k.txt`,
> scripts auxiliares), mais `html/BRIEF.md`, `html/concepts.md` e `html/concepts/**`. Para dar contexto correto
> aos patches também li (sem ser escopo primário) `README.md`, `CHANGELOG.md`, `.env.sample`, trechos de `start.sh`,
> `docs/CLAUDE/fable5-1-report.md`, `docs/HANDOFF-single-spark.md`, `docs/bench/rerun-20260916.md` e `tests/*`.
>
> **Fontes externas usadas (para verificar arquitetura, pois o modelo é posterior ao meu treino):**
> - `config.json` oficial de `Qwen/Qwen3.8-Flash-Next` (baixado de `huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`).
> - Blog LMSYS "Qwen3.8-Flash-Next: Day-0 Support in SGLang" (2026-08-26) — descreve QSA, Gated Residual/HyperConnection, PLE.
> - Model card `nvidia/Qwen3.8-Flash-Next-NVFP4` (mistura de precisões do checkpoint).
>
> Quando algo é **inferência minha** (não dito explicitamente no repo), está marcado com *(inferido)*.
> Quando há contradição entre fontes, está marcado com **⚠**.

---

## 1. Visão geral

### 1.1 O que esta parte da receita faz, de ponta a ponta

A pasta `files/` é o "laboratório de cirurgia" da receita. A imagem Docker usada (`vllm/vllm-openai:qwen38-flash-next`,
um build "day-0" da NVIDIA, vLLM `v0.1.dev20073+g8e685d198`) **nunca é reconstruída**. Em vez disso, o padrão é sempre:

1. **Extrair** o arquivo-fonte original de dentro da imagem (`docker create` + `docker cp`, função `extract_from_image`
   em `start.sh:568-575`), gravando-o como `files/<nome>.orig` (arquivos `.orig` e saídas são *gitignored*, `.gitignore`).
2. **Aplicar um patch por âncoras**: cada `patch_*.py` lê o `.orig`, procura trechos exatos ("âncoras") e exige que cada
   âncora apareça **exatamente uma vez** (`count != 1 → erro`); substitui; valida com `ast.parse` (quase todos) e grava a saída.
   Se a imagem mudar e uma âncora não bater, o patch **falha alto** em vez de gerar código errado.
3. **Bind-mount read-only** (`-v host:container:ro`) do arquivo patcheado por cima do arquivo do pacote vLLM dentro do
   container, nos **dois nós** (o worker recebe cópias via `scp` para `/tmp/...` ou `/tmp/vllm-overlay/`).
   A função `add_overlay` (`start.sh:551-566`) recusa dois overlays no mesmo caminho e colisões de nome-base.

Ordem em que `start.sh` aplica o que está em `files/` (numeração de passos do próprio script):

| Passo `start.sh` | O que roda | Arquivo(s) em `files/` | Condição |
|---|---|---|---|
| resolve snapshot | escolhe o snapshot HF completo | `resolve_snapshot.py` | sempre |
| detecção PLE | descobre `ple_embedding_dtype` ausente | `detect_ple_dtype.py` | sempre (`start.sh:356-362`) |
| 4b-2 | libera page cache do checkpoint | `evict_page_cache.py` | `EVICT_PAGE_CACHE=true` (default) |
| 4c | overlay FP8-dense | `overlay/*` (+ `fp8dense/` constrói o checkpoint) | `FP8_DENSE=true` |
| 4d | perfil de lançamento QSA | `qsa_gb10/apply_patch.py` → `qsa_gb10/qsa.py` | `QSA_PROFILE != stock` |
| 4e | vocabulário reduzido do drafter MTP | `patch_mtp_draft_vocab.py` (ou `_v030`) + `draft_vocab_en_code_47k.txt` | `MTP_DRAFT_VOCAB` não vazio (default: sim) |
| 4f | KV cache FP8 nos kernels QSA | `patch_qsa_fp8_kv.py` (ou `_v030`) | `KV_CACHE_DTYPE=fp8*` (default: sim) |
| 4g | alias de índice da camada MTP + preflight do algoritmo dos experts MTP | `patch_checkpoint_config.py` | sempre que há snapshot |
| 4h | backport vllm#53388 (não descartar o último bloco do prefix cache) | `patch_block_drop.py` | `MTP_DISABLE_BLOCK_DROP=1` (default no `.env.sample`) |
| 4i | decodificação greedy bit-idêntica | `patch_determinism.py` | `VLLM_QSA_DET_TOPK=1` e/ou `VLLM_MOE_DET_FINALIZE=1` (opt-in) |
| 6 | PLE: dispatch de quantização + correções de offload | `patch_ple_layer.py` → `ple_layer_patched.py` | sempre (exceto `SKIP_PLE_PATCH=true`) |
| 6b | fallback MXFP8 → BF16 + ramo FP8_BLOCK_SCALES para experts MTP | `patch_modelopt_mxfp8.py` + `patch_modelopt_fp8_block_moe.py` | sempre (exceto lane v0.30) |
| (só TP1) | offload do PLE para CPU com handshake no host + tabela mmap | `patch_ple_offload.py`, `ple_offload/**`, `build_ple_packed_table.py`, `memwatch.sh` | `start-tp1.sh` / `tp1/start.sh` |
| infra | NFS para compartilhar pesos | `nfs-share.sh`, `nfs-server/*` | `NFS_SHARE=true` |

A pasta `html/` **não contém conceitos de ML**: é um conjunto de páginas-demo geradas (ver §3.17).

### 1.2 O modelo: Qwen3.8-Flash-Next (arquitetura interna "Qwen4Exp")

Valores do `config.json` oficial (`text_config`, baixado do HF) cruzados com o que o código do repo usa:

| Hiperparâmetro | Valor | Onde aparece no repo |
|---|---|---|
| `model_type` | `qwen4_exp_text` (classe `Qwen4ExpForConditionalGeneration`) | vLLM 0.30 usa `vllm/models/qwen4_exp/...` (`files/patch_qsa_fp8_kv_v030.py:16-19`); a imagem day-0 usa `vllm/models/qwen3_8_flash_next/...` |
| `num_hidden_layers` | 48 | alias MTP `mtp.layers.48` (`files/patch_checkpoint_config.py:5-6`) |
| `hidden_size` | 2560 | lm_head 248320×2560 (`files/patch_mtp_draft_vocab.py:10`) |
| `layer_types` / `full_attention_interval` | 3× `linear_attention` + 1× `full_attention`, repetido 12× (`interval=4`) | README "12 KV-bearing layers" |
| Atenção cheia (QSA): `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 | `bench_qsa_kernels.py:39-40` (`24 // tp`, `max(1, 2 // tp)`) |
| `output_gate_type` | `sigmoid` (atenção com gate de saída) | projeção `qkv+gate (6656, 2560)` em TP2 (fable §3.6) = 3072 q + 256 k + 256 v + 3072 gate *(decomposição inferida)* |
| `partial_rotary_factor` | 0.25 (64 das 256 dims recebem RoPE) | — |
| `rope_parameters` | `rope_theta=1e7`, `mrope_interleaved=true`, `mrope_section=[11,11,10]`, `rope_type=default` | YaRN sobrescreve `rope_type` via `--hf-overrides` aninhado em `text_config` (`start.sh:924-946`) |
| Indexador QSA | `indexer_n_heads=4`, `indexer_head_dim=128`, `indexer_kv_heads=1`, `indexer_compress_ratio=4`, `indexer_budget=2048` | `bench_qsa_kernels.py:38,55-57` (`topk=2048`, `compress=4`, cache comprimido 1 head×128, `cq` 4×128) |
| Linear attention (Gated DeltaNet): `linear_num_key_heads` / `linear_num_value_heads` / `linear_key_head_dim` / `linear_value_head_dim` / `linear_conv_kernel_dim` | 16 / 48 / 128 / 128 / 4 | `in_proj_a/b` = `[48, 2560]` (`files/patch_modelopt_mxfp8.py:15`) |
| `mamba_ssm_dtype` | `float32` (estado recorrente GDN) | `MAMBA_SSM_CACHE_DTYPE=bfloat16` no `.env.sample:70` sobrescreve |
| MoE: `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` / `shared_expert_intermediate_size` | 512 / 10 / 640 / 640 | fable §1 ("512 experts, top-10"); `*.mlp.shared_expert_gate` em `fp8dense/make_fp8_dense_checkpoint.py:67` |
| Gated Residual / HyperConnection: `hc_count` / `hc_lowrank` | 4 / 320 | `overlay/hyperconnection.diff:14-15`; `key_proj` → `hidden*hc_count` = 10240 (`ple_layer_patched.py:933,962-968`) |
| PLE / N-gram: `ngram_size` / `heads_per_ngram` / `ngram_vocab_size_base` / `split_ngram_parts` / `make_ngram_vocab_size_divisible_by` / `ple_embed_dim` / `ple_conv_kernel_size` / `ple_layer_ids` | 3 / 8 / 20.000.000 / 128 / 128 / 2560 / 4 / `[2]` | `ple_layer_patched.py:423-477,927-936` |
| `vocab_size` / `tie_word_embeddings` | 248.320 / `false` | `patch_mtp_draft_vocab.py:8-11` |
| MTP | `mtp_num_hidden_layers=1`, `mtp.layer_types=["full_attention"]`, `mtp.hybrid=true`, `mtp_use_dedicated_embeddings=false` | camada construída como `mtp.layers.48` |
| `max_position_embeddings` | 262.144 | YaRN ×4 → ~1M |
| tokens especiais | BOS=EOS=248044; vision start/end 248053/248054; image 248056; video 248057 | presentes no fim de `draft_vocab_en_code_47k.txt` |

Estrutura de cada bloco (síntese a partir do código + blog LMSYS):

```
R (4 ramos × 2560)  ──HC Mix (gates element-wise, low-rank 320)──►  h (2560)
      │                                                               │
      │                             [camada 1 apenas] PLE: N-gram lookup → gate por ramo → short-conv → Δ injetado em R
      │                                                               ▼
      │                        token-mixer:  GDN (36 camadas)  OU  QSA com gate sigmoid (12 camadas)
      │                                                               │
      ◄──────────── HC Combine (4 coeficientes de injeção) ◄──────────┘
      │  (repete com um segundo GatedResidual em volta do MoE: 512 experts top-10 + shared expert com gate)
```

- **GDN (Gated DeltaNet)**: 36 camadas de "atenção linear" com estado recorrente de tamanho fixo por requisição.
- **QSA (Qwen Sparse Attention)**: 12 camadas de atenção "cheia" mas **esparsa**: um indexador leve escolhe ≤2048 posições
  (512 micro-blocos de 4 tokens) + até 3 tokens do bloco corrente = até 2051 posições; a atenção final usa o K/V original.
- **Gated Residual (GR) / HyperConnection (HC)**: o residual é alargado para 4 ramos (`hc_count=4`); cada sub-bloco lê com
  "Mix" e escreve com "Combine". No código vLLM: `attn_hyper_connection`, `mlp_hyper_connection` (classe `GatedResidual`,
  `overlay/model.diff:6-12`) e um `hyper_connection_mixer` final (`use_combine=False`). 48×2+1 = 97 módulos, 1,19 GiB bf16
  replicados (fable §2).
- **PLE (N-gram Embedding)**: uma única camada (índice 1 em base zero, segundo bloco) com uma tabela de 51,2 B parâmetros
  endereçada por hash de 2-gramas e 3-gramas. ⚠ Nome: o docstring do arquivo da imagem diz
  *"position-learning enhancement layers"* (`ple_layer_patched.py:3`); o blog LMSYS chama de *"Per-Layer Embeddings (PLE)"*;
  o README do repo chama de *"N-gram/PLE embedding table"*. O **mecanismo** é o mesmo em qualquer nome.
- **MTP**: 1 camada extra (≈4 B parâmetros segundo o model card NVIDIA) usada como *drafter* de decodificação especulativa,
  reaplicada 3× (`num_speculative_tokens=3`), com **seu próprio** lm_head de 248.320 linhas.
- **Visão**: ViT de 27 blocos, hidden 1152, MLP intermediate 4304 (origem do problema MXFP8 `[4304,1152]`).

### 1.3 O checkpoint e por que ele exige patches

`nvidia/Qwen3.8-Flash-Next-NVFP4` (133 GB / 11 shards, `.env.sample:24`) é um checkpoint ModelOpt `MIXED_PRECISION`:
experts roteados em **NVFP4 (W4A4)**, experts da MTP em **FP8 bloco 128×128**, tabela PLE em **FP8 por tensor** (escala global única),
todo o resto (atenção, GDN, HC, shared expert, lm_head, visão) em **BF16**. Alternativas citadas:
`local-inference-lab/...NVFP4` (99 GB, atenção MXFP8 + PLE NVFP4) e `RadixArk/...NVFP4` (126 GB, atenção BF16 + PLE FP8)
(`.env.sample:25-26`, `docs/HANDOFF-single-spark.md` §2). Cada formato quebra uma suposição diferente do vLLM da imagem —
daí a maioria dos patches deste inventário.

---

## 2. Inventário de parâmetros

Formato: **nome** | valor default e alternativos | onde (arquivo:linha) | o que faz | conceito(s) | por que importa / trade-off.

### 2.1 Parâmetros do PLE (N-gram embedding)

| Nome | Valores | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `ple_embedding_dtype` (em `text_config`) | `"nvfp4"`/`"fp4"`/`"w4a16_nvfp4"` → NVFP4; `"float8_e4m3fn"`/`"fp8"` → FP8; ausente → cai nos ramos antigos | `ple_layer_patched.py:286-301,355-406`; injetado por `start.sh:924-946` | Escolhe o `QuantizeMethodBase` da tabela PLE | Quantização de embeddings; dispatch de quant method no vLLM | Sem isso, num checkpoint MIXED_PRECISION a tabela vira BF16 (≈102 GB) e o load morre (README "The PLE Patch") |
| `PLE_EMBEDDING_DTYPE` (env / `.env`) | vazio (auto-detecção) | `.env.sample:41`, `start.sh:356-362` | Força o valor acima, pulando a detecção | — | Só para override manual |
| saída de `detect_ple_dtype.py` | `float8_e4m3fn` (nvidia), vazio (local-inference-lab e RadixArk já declaram) | `files/detect_ple_dtype.py:28-49` | Lê `quantization_config.config_groups`; grupo que tem alvo `ple_embedding.ngram_embedding` com `num_bits` 8 → FP8, 4 → NVFP4 | Metadados de quantização (compressed-tensors/ModelOpt config groups) | O checkpoint nvidia só declara a PLE nos config groups |
| `ngram_size` | 3 (config) | `ple_layer_patched.py:423,426-427` | Ordem máxima de n-grama (2-gramas e 3-gramas) | N-gramas, hashing | Também é a **dilatação** da short-conv (`:935`) |
| `heads_per_ngram` | 8 | `:424-425` | Cabeças de hash por ordem → `ngram_heads = (3-1)×8 = 16` | Multi-head hashing | 16 linhas buscadas por token |
| `ple_embed_dim` | 2560 | `:927,430-435` | Dim concatenada; `head_dim = 2560/16 = 160` | Embedding | Cada linha tem 160 valores |
| `ngram_vocab_size_base` | 20.000.000 | `:456-465` | Cada cabeça h usa uma tabela de tamanho = o (h+1)-ésimo primo após `base-1` (primos distintos) | Hashing modular com primos | Total ≈ 16 × 20M ≈ 320M linhas × 160 = **51,2 B parâmetros** *(conta minha, bate com README/LMSYS)* |
| `make_ngram_vocab_size_divisible_by` | 128 | `:476-477` | Arredonda o nº total de linhas para múltiplo | Padding para sharding | Divisível entre ranks de TP |
| `split_ngram_parts` | 128 (config; default de código 512) | `:438-440,818-840` | Quantos shards de linhas o checkpoint grava (`ngram_embedding.shard_{i}.weight`) | Sharding do checkpoint | O loader recompõe e recorta para o range do rank TP |
| `seed` | 1234 (default de código, ausente no config) | `:444-445` | Semente dos multiplicadores de hash: `base_seed = seed + 10007 × ple_dense_layer_id` | SplitMix64 | Determina o hashing; tem de ser idêntico ao treino |
| `_SPLITMIX_GAMMA`, `_M1`, `_M2`, `_PLE_LAYER_PRIME` | `0x9E3779B97F4A7C15`, `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`, 10007 | `:62-73` | Mistura de bits (SplitMix64) para gerar multiplicadores ímpares `2·(splitmix(v) mod half_bound)+1` | Funções de hash, PRNG | `half_bound` limita o multiplicador para `token×mult` não estourar int64 (`:442-443`) |
| `ple_layer_ids` | `[2]` | config.json | Em qual bloco o PLE fica | — | ⚠ O código vLLM expõe o módulo em `layers.1.ple` (`fp8dense/test_quant_config_resolution.py:62`); o LMSYS explica: "configured layer ID 2, corresponding to zero-based index 1" |
| `ple_conv_kernel_size` | 4 | `:934` | Kernel da conv 1D depthwise causal | Convolução curta (short conv) | `conv_state_len = (4-1)×3 = 9` → estado `[10240, 9]` por requisição (bate com LMSYS) |
| `hc_count` | 4 | `:932-933` | O PLE gera **uma key por ramo** (`key_proj: 2560 → 10240`) e **um value compartilhado** (`value_proj: 2560 → 2560`) | Gating por ramo de HC | O gate é um escalar por ramo |
| `rms_norm_eps` | 1e-6 | `:976-984` | Norm agrupada (grupos de 2560) em key, query e conv | RMSNorm com peso `(1+w)` (`:147`) | — |
| `VLLM_PLE_CPU_OFFLOAD` (← `PLE_OFFLOAD`) | `false` (default) / `true` → `-e VLLM_PLE_CPU_OFFLOAD=1` | `.env.sample:101`, `start.sh:94,980-982,1080`; lido em `ple_offload/orig/ple_offload_layer.py:201,211` | Constrói a tabela PLE num **processo CPU separado**; os workers GPU recebem só um placeholder | Offload CPU, IPC | README: precisa de ~51 GB de RAM livre e o box tem ~42-45 GiB → **manter false** no dual-Spark |
| `VLLM_PLE_PACKED_TABLE_DIR` | vazio / diretório com `<layer>.ngram_embedding.packed_u8` + `.json` | `patch_ple_offload.py:280-295` (worker) | Faz o worker CPU **mapear em memória** (mmap) uma tabela pré-empacotada | mmap, page cache | Em memória unificada: ~104 GiB → ~77 GiB de footprint não-evictável (`patch_ple_offload.py:21-25`) |
| `VLLM_PLE_OFFLOAD_STEP_TIMEOUT` | 300 (s) | `patch_ple_offload.py:140-142` | Timeout do handshake por passo | Watchdog | Evita travar para sempre silenciosamente |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | (definido em `vllm.envs`, valor não visível no repo) | `ple_offload/orig/worker.py:214` | Timeout de inicialização do processo de offload | — | — |
| `PLE_QUANT_OVERRIDE=fp8` | citado no README | README.md:460,557 | ⚠ **Obsoleto**: nenhum arquivo atual lê essa variável; o dispatch atual é por `ple_embedding_dtype` | — | Não confiar no README neste ponto |
| `SKIP_PLE_PATCH` | `false` | `start.sh:125,813` | Pula o passo 6 (checkpoint FP8 nativo) | — | — |

### 2.2 Parâmetros de KV cache FP8 e dos kernels QSA

| Nome | Valores | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` → `--kv-cache-dtype` | `fp8` (default) / `auto` (=bf16) / `fp8_e4m3` | `.env.sample:60`, `start.sh:87,663-681` | Dtype do KV principal das 12 camadas QSA (+1 MTP) | Quantização de KV cache | fp8: 3.652.200 tokens vs 2.131.159 bf16 (×1,70) no mesmo GMU; custo medido −5,3% throughput médio (rerun 2026-09-16) |
| `supported_kv_cache_dtypes` | stock `["auto","bfloat16"]` → patch `["auto","bfloat16","fp8","fp8_e4m3"]` | `patch_qsa_fp8_kv.py:262-275` | Backend QSA anuncia suporte a fp8 | Registro de backends de atenção no vLLM | Sem isso: `Qwen3.8-Flash-Next QSA requires a BF16 main KV cache` |
| `KV_QUANT_MODE` (`tl.constexpr`) | 0 (bf16) / 1 (fp8) | `patch_qsa_fp8_kv.py:127-130,159-165,280-282` | Liga os ramos de cast/escala no kernel Triton | Especialização em tempo de compilação | Com 0, o código BF16 compilado é idêntico ao original ("inert unless fp8") |
| `k_scale_ptr`, `v_scale_ptr` / `layer._k_scale`, `layer._v_scale` | escalas por tensor (0-d fp32); default 1.0 se `None` | `patch_qsa_fp8_kv.py:66-81,309-311` | `scores *= k_scale` após `dot(q,k)`; `output *= v_scale` no final | Quantização per-tensor, álgebra da escala | Aplicar depois do dot é algebricamente igual e evita tiles FP32 dequantizados. *(inferido: sem escalas calibradas no checkpoint, o vLLM usa 1.0)* |
| `_qsa_as_fp8` | reinterpreta `uint8` → `float8_e4m3fn` sem cópia | `patch_qsa_fp8_kv.py:84-101` | O vLLM aloca KV quantizado como `uint8`; o kernel precisa ver e4m3 | `tensor.view(dtype)` | Recusa ler bytes quantizados como BF16 (falha alto em vez de lixo plausível) |
| `OVERRIDE_KV_CACHE_DTYPE` (lane v0.30) | vazio (BF16 default na lane) / `fp8` | README "vLLM 0.30 lane", `start.sh:609-619` | Ativa o backport vllm#55557 | — | 2,64M tokens, NLL 1,328/1,334, ~5% mais lento que BF16 |
| `IS_FP8` (`tl.constexpr`, v0.30) | bool | `patch_qsa_fp8_kv_v030.py:64,84-105` | Upcast K→dtype de Q; V→**fp16** no PV dot | — | "fp8 QK measured slower here and less accurate"; fp16 tem mantissa maior e P≤1 cabe no range |
| `softmax_scale`, `output_scale` (v0.30) | `head_dim^-0.5 × k_scale`, `v_scale` (floats do host) | `patch_qsa_fp8_kv_v030.py:224-233` | Pré-multiplica as escalas no host | Fusão de escalas | Kernel não precisa de buffers de escala no device |
| `_select_sm120_config` (v0.30) | tabela por `base_programs` × bf16/fp8 | `patch_qsa_fp8_kv_v030.py:129-165` | Tabela afinada para RTX PRO 6000 (sm_120) | Autotuning de kernel | ⚠ `_is_sm120()` compara `== (12, 0)`; o GB10 é sm_121 `(12,1)`, logo **usa a tabela GB300 padrão** *(inferido do código)* |
| `QSA_PROFILE` → `VLLM_QSA_PROFILE` / `VLLM_QSA_PROFILE_JSON` | `stock` (default) / `gb10` / caminho de JSON | `.env.sample:158`, `start.sh:156,586-596`, `qsa_gb10/apply_patch.py:49-57` | Substitui a tabela de lançamento dos kernels QSA | Tiling/launch config, ocupação de SMs | Tabela stock foi afinada no GB300 (160 SMs); GB10 tem 48 SMs. `gb10` **não foi benchmarkado** (fable §8.2) |
| Tabela stock do sparse GQA `(block_n, target_splits, num_warps)` | small `(16,64,4)`, lt32 `(16,32,4)`, le256 `(64,8,2)`, le512 `(64,4,2)`, large `(64,1,2)` | `qsa_gb10/apply_patch.py:94-104` | Buckets por `base_programs = num_rows × num_kv_heads` | Split-K / flash-decoding | Decode stock lança ~5× mais CTAs do que o GB10 roda de uma vez e depois paga merge de 64 parciais (`:24-28`) |
| Tabela `GB10_PROFILES["sparse"]` | small `(32,16,4)`, lt32 `(32,8,4)`, le256 `(64,4,2)`, le512 `(64,2,2)`, large `(64,1,2)` | `qsa_gb10/apply_patch.py:37-46` | Menos partições split-K para 48 SMs | Waves de CTAs | "≈1.3 waves instead of ≈5, one merge over 16 partials instead of 64" (fable §8.2) |
| `GB10_PROFILES["mqa"]` | `tiles_small=2` (stock 1), `tiles_large=8` (stock 8), `num_warps=2` | `qsa_gb10/apply_patch.py:45,65-75` | Tiles por programa do *scorer* do indexador | — | — |
| `bench_qsa_kernels.py` args | `--tp 2`, `--rows 1 4 8 32`, `--ctx 4096 32768 131072`, `--out qsa_profile.json` | `qsa_gb10/bench_qsa_kernels.py:78-84` | Varre `block_n∈{16,32,64}×splits∈{1..64}×warps∈{2,4}` e MQA `tiles∈{1,2,4,8,16}×warps∈{1,2,4}` | Micro-benchmark | Formas reais: 12 q-heads × 1 kv-head × 256, página 1600, top-k 2048+3 |
| `VLLM_QSA_DET_TOPK` | vazio / `1` | `.env.sample:143`, `start.sh:159,752-775`, `patch_determinism.py:25-39` | Ordena as linhas do top-k do indexador | Não-associatividade de float | Bit-identidade greedy; custo ~−4% decode/prefill |

### 2.3 Parâmetros de MTP / decodificação especulativa

| Nome | Valores | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| `MTP_NUM_SPECULATIVE_TOKENS` → `num_speculative_tokens` | 3 (default) / 0 desliga | `.env.sample:95`, `start.sh:903-915` | Rascunhos por passo | Speculative decoding | 24,5 → 52,1 tok/s (2,13×); aceitação 89/74,5/60% por posição |
| `MTP_DRAFT_VOCAB` → `VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt` | `files/draft_vocab_en_code_47k.txt` (default desde 2026-09-10) / vazio = vocab cheio | `.env.sample:130`, `start.sh:151,636-657`, `patch_mtp_draft_vocab.py:56` | Corta o lm_head do drafter para um subconjunto de ids | FR-Spec | +9,6% médio (2026-09-11), +8,4% (rerun 09-16) |
| `use_local_argmax_reduction` | `true` quando há draft vocab | `start.sh:911` | Faz o speculator chamar `get_top_tokens` (argmax local por rank + all-gather de pares) | Redução TP de argmax | Único caminho que chega ao head reduzido; corta o all-gather de O(vocab) para O(2·tp) por token |
| `MTP_INDEX_SHARE` → `index_share_for_mtp_iteration` | `true` no `.env.sample` (default de `start.sh` = `false`) | `.env.sample:139`, `start.sh:158,906` | Reusa o top-k do indexador QSA nos passos de rascunho | IndexShare MTP | code +6,1/+4,6/+21,3/+26,6% (S=1/2/4/8), prose ~ruído |
| `MTP_DISABLE_BLOCK_DROP` → `disable_eagle_block_drop` | `1` no `.env.sample` (default de `start.sh` = `0`) | `.env.sample:135`, `start.sh:157,905,740-751` | Não descarta o último bloco casado do prefix cache | Prefix caching + EAGLE | TTFT turno 1: 3,25 → 0,63 s (sessão 9,5k) |
| `build_draft_vocab.py --size` | 32768 default (README sugere 65536) | `files/build_draft_vocab.py:65` | Tamanho do vocabulário de rascunho | Cobertura de frequência | Ajustar por **cobertura**, não por tamanho |
| `--balance-shards` | 1 default (usar = TP size) | `:67-73,116-149` | Distribui o *padding* igualmente entre faixas de id | Balanceamento vocab-parallel | O passo espera o rank mais lento |
| `--fill-to-size` | off | `:74-79` | Completa com os ids mais baixos não vistos | Proxy "ordem de merge do BPE" | "a proxy, not a measurement" |
| `--report-only`, `--model` / `DRAFT_VOCAB_MODEL` | `nvidia/Qwen3.8-Flash-Next-NVFP4` | `:62-66` | Só relatório / tokenizer | — | — |
| sufixo `corpus:N` | repetir N vezes | `:34-41` | Peso de um corpus pequeno in-distribution | — | — |
| `draft_vocab_en_code_47k.txt` | 47.149 ids ordenados, únicos, min 0, max 248076 | arquivo | Lista (um id por linha) | — | 45.734 ids < 124.160 (rank 0) e 1.415 no rank 1 (≈97% no rank 0) *(contagem minha)*; termina com 248044–248076 (tokens especiais) |

### 2.4 Parâmetros de quantização / checkpoint

| Nome | Valores | Onde | O que faz | Conceito | Por que importa / trade-off |
|---|---|---|---|---|---|
| Restrição `mm_mxfp8` (FlashInfer) | `N ≥ 128`, `N % 32 == 0`, `K ≥ 128`, `K % 32 == 0` (medido no sm_121) | `patch_modelopt_mxfp8.py:5-11,70-77` | Decide se usa `FlashInferCutlassMxfp8LinearKernel` ou `EmulationMxfp8LinearKernel` | MXFP8 microscaling | `in_proj_a/b [48,2560]` falha (N<128); `visual...linear_fc1 [4304,1152]` falha (4304%32=16) |
| `MXFP8_BLOCK_SIZE` | 32 | `patch_modelopt_mxfp8.py:77` | Tamanho de bloco MX | OCP MX | — |
| prefixo `visual.` | sempre emulação | `:66-67,139-148` | Roteia toda a torre de visão para BF16 | — | Dequant global de tudo daria OOM |
| `SUPPORTED_MOE_ALGOS` | `FP8, NVFP4, W4A16_NVFP4, MXFP8, FP8_BLOCK_SCALES, FP8_PB_WO` | `patch_checkpoint_config.py:93` | Preflight: `--mtp-moe-algo` sai com código 3 se o algoritmo não é construível | Fail-fast | Falha em segundos em vez de ~7 min de load |
| `group_size` (FP8_BLOCK_SCALES) | 128 → `weight_block_size=[128,128]` | `patch_modelopt_fp8_block_moe.py:51-62` | Bloco das escalas lido do checkpoint (recusa adivinhar) | Block-wise FP8 | Bloco errado aplica escalas desalinhadas **silenciosamente** |
| `FP8_PB_WO` com `group_size==128` | tratado como `FP8_BLOCK_SCALES` | `patch_checkpoint_config.py:96-127`, `patch_modelopt_fp8_block_moe.py:83-99` | Normaliza rótulo novo/errado do ModelOpt | — | nvidia rev `fc694b54` e o checkpoint Keys usam o rótulo novo |
| `FP8_DENSE` / `FP8_DENSE_MODEL_ID` | `false` / `MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense` | `.env.sample:152-153`, `start.sh:129-133,576-585` | Serve o checkpoint híbrido + monta overlay | Mixed precision | Esperado ≈9,2 → 5,7 GiB/passo (≈1,5× batch-1) — **não medido** |
| `SRC_REPO` / `DST_REPO` / `IMAGE` / `HF_HOME` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` / `MiaAI-Lab/...-FP8dense` / `vllm/vllm-openai:qwen38-flash-next` / `~/.cache/huggingface` | `fp8dense/build.sh:6-9` | Origem/destino do conversor | — | ⚠ SRC default é RadixArk, **não** o nvidia servido por default |
| `QUANT_PATTERNS` | 19 padrões (GDN qkv/z/out, atenção q/k/v/o, HC down/inject/up dos 2 GR + mixer, shared expert gate/up/down, `lm_head.weight`) | `make_fp8_dense_checkpoint.py:36-56` | Quais tensores viram FP8 | Seleção de camadas | lm_head é 26% do passo por causa das 3 leituras extras do drafter |
| `EXCLUDE_MODULES` | embeddings, `mtp.*`, `visual.*`, `*.ple.*`, router `*.mlp.gate`, `*.mlp.shared_expert_gate`, indexer, q/k norm, `in_proj_a/b/ba`, `conv1d`, `linear_attn.norm`, `*.hc_norm`, `hyper_connection_mixer.block_inject_weight` | `:59-78` | Ficam BF16 | Sensibilidade à precisão | Roteamento, seleção do indexador, gates da recorrência e MTP são sensíveis |
| `FP8_MAX` | 448.0 | `:30` | Máximo do E4M3 | Formato FP8 E4M3 | escala = amax/448 |
| `ROW_CHUNK_BYTES` | 16 MiB | `:31,184` | Chunk de linhas no streaming | Streaming | Roda com cgroup de 1,2 GB ao lado de um servidor vivo |
| `--dry-run`, `--resume` | flags | `:208-209,253-260` | Planejar / manter shards já escritos | — | `build.sh` passa `--resume` |
| `quant_algo` gravado | `FP8_PER_CHANNEL_PER_TOKEN` (591 tensores), experts `NVFP4` `group_size 16` | `:264,270-274` | `quantized_layers` do config ModelOpt `MIXED_PRECISION`, `producer modelopt 0.46.0` | Config ModelOpt | — |
| limiar de erro no verify | `rel > 0.06` → HIGH ERROR | `verify_fp8_dense_checkpoint.py:86` | Sanidade | RMSE relativo | Medido: 0,0240 / 0,0262 / 0,0268 (min/mediana/max) |
| `MM_ENCODER_TP_MODE` (relacionado) | `data` (default) / `weights` crasha | `.env.sample:92`, `start.sh:97` | Replica o encoder de visão por GPU | TP vs DP do encoder | 4304/2 = 2152 não é múltiplo de 16 (kernels NVFP4) |
| `MAMBA_SSM_CACHE_DTYPE` (relacionado) | `bfloat16` no `.env.sample`; vazio = float32 do checkpoint | `.env.sample:70-82`, `start.sh:93,884` | Dtype do estado GDN | Estado recorrente | Metade dos bytes/passo e página mamba menor; **não medido no dual** |

### 2.5 Determinismo, prefix cache, infraestrutura

| Nome | Valores | Onde | O que faz | Conceito | Por que importa |
|---|---|---|---|---|---|
| `VLLM_MOE_DET_FINALIZE` | vazio / `1` | `.env.sample:144`, `patch_determinism.py:41-52` | `use_fused_finalize = (env != "1")` na FlashInfer CUTLASS MoE | Ordem de redução determinística | Junto com top-k ordenado: spread de logprob 0,25/6,07 nats → 0/0 |
| `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0` + `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/flashinfer_autotune_cache_unfused` | injetados quando `VLLM_MOE_DET_FINALIZE=1` | `start.sh:772-775` | Cache de autotune separado para a variante não-fundida | Autotune | Cache persistido diferente entre nós já travou TP2 (CHANGELOG 09-25 v0.30) |
| `EVICT_PAGE_CACHE` | `true` | `.env.sample:171`, `start.sh:502` | `posix_fadvise(DONTNEED)` nos `.safetensors/.bin/.pt/.gguf` do checkpoint | Page cache em memória unificada | 4,15 → 0,00 GiB residente por shard, sem root |
| `resolve_snapshot.py` exit codes | 0 completo, 1 incompleto, 2 inexistente | `files/resolve_snapshot.py:4-7` | Snapshot completo = todos os shards do `model.safetensors.index.json` existem | HF cache layout | Download interrompido não é tratado como pronto |
| `memwatch.sh <container> <min_gib>` | min default 6 GiB | `files/memwatch.sh:9-22` | Mata o container se `MemAvailable` < piso | Watchdog | Em GB10, esgotar a memória unificada trava o kernel (sem OOM) |
| NFS: `NFS_IMAGE`, `NFS_CONTAINER`, `NFS_VOLUME`, `NFS_SERVER_IP` | `vllm-fn-nfs:local`, `vllm-fn-nfs`, `vllm-fn-hf`, auto de `IFACE` | `files/nfs-share.sh:8-19` | Exporta o HF cache do head | NFSv4 | Não usar o alias loopback `10.0.0.1` |
| opções do volume NFS | `nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600` | `files/nfs-share.sh:105-109` | Montagem no worker | Throughput NFS | `nconnect=8` paraleliza conexões TCP |
| `NFS_OPTS` (servidor) | `ro,sync,no_subtree_check,no_root_squash,insecure,fsid=0` | `files/nfs-server/entrypoint.sh:7` | Export read-only como pseudoroot | — | `fsid=0` faz o export ser a raiz v4 |
| `V030_BREAKABLE_CUDAGRAPH` (lane v0.30, relacionado) | 0 (default da lane) | `start.sh:621` | `VLLM_USE_BREAKABLE_CUDAGRAPH` | CUDA graphs | Com 1: variação 10-18% entre runs idênticos |

---
## 3. Inventário de modificações

Cada item: **nome** | arquivo(s) | problema que resolve | o que muda tecnicamente | conceito(s) | efeito medido.

### 3.1 PLE — dispatch de quantização misto (port do vLLM PR #53899) e correções do caminho de offload

- **Arquivos:** gerador `files/patch_ple_layer.py` (884 linhas), saída versionada `files/ple_layer_patched.py` (1668 linhas),
  montada em `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro`
  (`start.sh:802-832`). Cópia idêntica em `tp1/files/patch_ple_layer.py` (verificado com `cmp`).
- **Problema:** a imagem day-0 só escolhe o método FP8 para a tabela PLE quando **o checkpoint inteiro** é `Fp8Config`
  (`patch_ple_layer.py:11-12`, versão removida em `:549-573`). Os checkpoints NVFP4 são `MIXED_PRECISION`/`ModelOptNvFp4Config`
  com a PLE **excluída** do config NVFP4 → o vLLM cria uma embedding BF16 de ≈102 GB e cai (README "The PLE Patch").
  Além disso, a imagem não sabia ler uma PLE **NVFP4** (checkpoint local-inference-lab).
- **O que muda:**
  1. **Novo método `Qwen3_8FlashNextPLENVFp4EmbeddingMethod`** (`patch_ple_layer.py:25-112` → `ple_layer_patched.py:196-283`):
     registra `weight` `uint8 [rows, dim/2]` (2 códigos FP4 por byte), `weight_scale` `float8_e4m3fn [rows, dim/16]`
     (escala por bloco de 16) e `weight_scale_2` `float32` escalar (escala global). A busca (`embedding`) devolve, por linha,
     `cat(códigos, escalas.view(uint8))` = 80 + 10 = **90 bytes por cabeça** (head_dim=160). A dequantização
     (`_dequant_nvfp4_codes`, `:307-325`) separa nibbles (`& 0xF`, `>> 4`), usa a LUT E2M1 `(0, 0.5, 1, 1.5, 2, 3, 4, 6)` para
     os 3 bits de magnitude, o bit 3 como sinal, e multiplica `fp4 × escala_bloco × escala_global`. Ou seja, a tabela
     **fica empacotada na memória** e só as 16 linhas buscadas por token são dequantizadas.
  2. **Novo `_get_ple_embedding_quant_method(quant_config, prefix, ple_embedding_dtype)`** (`ple_layer_patched.py:355-406`),
     ordem de decisão: (a) `ple_embedding_dtype` NVFP4 → NVFP4; (b) `ple_embedding_dtype` FP8 → FP8; (c) `Fp8Config`
     serializado e não ignorado → FP8; (d) `ModelOptNvFp4Config` e a PLE não excluída → NVFP4; excluída mas dtype FP8 → FP8;
     (e) senão `None` (BF16). O construtor passa `getattr(config, "ple_embedding_dtype", None)` (`:484-488`).
  3. **`load_weights` reescrito** (`patch_ple_layer.py:239-449`): ignora `hashstats_*` e `token_lookup`; copia buffers
     persistentes (`layer_multipliers`, `ngram_heads_offsets`, `ngram_heads_vocab_sizes`) validando shape; para NVFP4
     coleta `shard_{i}.weight` (códigos) e `shard_{i}.weight_scale` (escalas) por índice de shard e recorta cada shard para o
     intervalo de vocabulário do rank TP (`copy_ple_embedding_shard_(..., checkpoint_start, tp_start, tp_end)`); guarda a escala
     global `weight_scale_2`. Para FP8/BF16 mantém a cópia por shard com validação de shape `(rows, embedding_dim)`.
  4. **`_dequantize_embeddings`** (`ple_layer_patched.py:1011-1064`): ramos NVFP4 (on-device e offload) e FP8
     (`emb.to(bf16) * weight_scale`, escala global única por tensor).
  5. **Três correções do caminho de offload NVFP4** (achadas em `docs/HANDOFF-single-spark.md` §5):
     - restaura `get_offload_output_dtype` (`patch_ple_layer.py:712-728`), que tinha sido apagado por aparecer só como âncora:
       o buffer IPC precisa ser `uint8` (NVFP4) ou `float8_e4m3fn` (FP8), senão `index_select(... out=buffer)` falha com
       "self and result must have the same scalar type";
     - atribui `self._offload_quant_method` (`:729-748`), que era lido mas nunca escrito → o ramo NVFP4 de dequant era código morto;
     - generaliza o *fast path* para a largura de linha empacotada (`:750-807`): antes, reshape para `head_dim` fazia o
       `torch.index_select(out=...)` **redimensionar** o tensor de saída silenciosamente (aloca novo tensor em vez de escrever no
       buffer compartilhado) → o lado GPU esperava para sempre. Agora prefere uma tabela `_packed_table` mmap
       (de `build_ple_packed_table.py`), senão monta `[códigos | escalas]` por linha.
  6. **Correções do placeholder GPU no GB10** (`:809-878`): sob offload, o construtor da subclasse é pulado
     (`PleOffloadLayer.__init_subclass__`, `ple_offload/orig/ple_offload_layer.py:189-206`), então o placeholder não sabia que
     era NVFP4; o patch injeta `_offload_quant_method` a partir do config (`ple_layer_patched.py:953-961`), aceita `load_weights`
     chamado várias vezes (a escala global pode chegar depois) e recorta o buffer IPC de 2560 para os 16×90 = 1440 bytes válidos
     (`:1035-1037`).
- **Conceito subjacente — o que é o PLE / N-gram embedding (mecanismo lido em `ple_layer_patched.py`):**
  - Para cada token `x_t`, `forward_impl` (`:540-665`) monta, por requisição, a sequência `ngram_context` (2 tokens anteriores,
    guardados entre passos/chunks) + tokens novos; gera versões deslocadas por 1 e 2 posições, respeitando fronteiras de
    documento (tokens antes de um EOS não se combinam: `_shift_precompute`/`_shift_apply`, `:505-538`).
  - Para cada ordem n ∈ {2, 3}: `mixed = XOR_{i<n} (token_{t-i} × m_i)` (multiplicadores ímpares de 64 bits por camada, `:442-454`);
    para cada uma das 8 cabeças daquela ordem: `id = (mixed mod p_h) + offset_h`, com `p_h` um primo distinto ≈ 20M (`:612-626`).
    Resultado: 16 ids por token (8 de 2-grama, 8 de 3-grama) numa única `VocabParallelEmbedding` de ≈320M linhas × 160
    (sharded por faixa de linhas entre os ranks TP → ≈24 GiB FP8 por GPU, fable §1).
  - `forward` (`:1597-1635`): `E_t = concat(16 × 160) ∈ R^2560` → `key = key_proj(E)` ∈ R^{4×2560} (uma por ramo de HC),
    `value = value_proj(E)` ∈ R^2560 (compartilhado); `query` = o estado residual de 4 ramos `R_t` (hidden 10240 visto como 4×2560);
    `g = sigmoid(sign(s)·sqrt(|s|))`, com `s = <RMSNorm(key), RMSNorm(query)> / sqrt(2560)` por ramo (`:1625-1626`);
    `U = g ⊙ V`; `Δ = U + SiLU(DWConv_dilatada(RMSNorm(U)))`; `Δ` é injetado no estado HC (LMSYS: "R̃_t = R_t + Δ_t").
  - A conv curta é **depthwise causal** com kernel 4 e **dilatação 3** (= `ngram_size`) sobre 10240 canais (`:985-996`);
    seu estado por requisição tem `(4-1)×3 = 9` posições (+ `num_speculative_tokens` para rollback de MTP, `:1581-1589`).
    O layer se registra como `MambaBase` com backend `SHORT_CONV` (`:1066-1088`), então o vLLM gerencia esse estado como
    estado "mamba" (tamanho constante por requisição, replicado em TP: `is_kv_cache_tp_replicated=True`).
  - Três caminhos de conv: decode (`:1102-1168`), prefill em lote (`:1170-1300`) e especulativo (`:1302-1436`, faz *rollback*
    da janela por `num_accepted_tokens - 1`, igual ao `causal_conv1d_update`); linhas de padding de CUDA graph usam
    `NULL_BLOCK_ID` e são mascaradas. A conv roda como custom op opaco `torch.ops.vllm.qwen3_8_flash_next_ple_short_conv`
    com *fake impl* (`:1638-1661`), para ser compatível com `torch.compile`/captura de CUDA graph.
  - *(Associação minha, não citada no repo)*: é o mesmo desenho da "memória condicional" por hash de N-gramas (família
    "Engram"): hashing multi-cabeça com tabelas de tamanho primo, gate sensível ao contexto e conv causal dilatada.
- **Efeito medido:** sem o patch, o checkpoint não carrega. Com o offload corrigido, o TP1 passou a servir (HANDOFF §0:
  "Three independent bugs had to be fixed"; sintoma anterior do bug 3: "fluent gibberish, no error").

### 3.2 PLE — offload para CPU com handshake no host (GB10 sem *stream memory ops*)

- **Arquivos:** `files/patch_ple_offload.py` (gerador), originais `files/ple_offload/orig/{protocol,ple_offload_layer,connector,worker}.py`
  (extraídos da imagem, versionados), saídas `files/ple_offload/{protocol,ple_offload_layer,connector,worker}.py`
  (diffs: 7, 21, 91 e 101 linhas de `diff`). Montados **apenas** por `start-tp1.sh:266-275,347-350` / `tp1/start.sh`.
- **Problema:** o design stock sincroniza processo CPU e stream da GPU com `cuStreamWaitValue32` / `cuStreamWriteValue32`
  (semáforo num tensor int32 na GPU compartilhado por CUDA IPC, `orig/ple_offload_layer.py:59-173`). O GB10 reporta
  `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0`: o wait retorna sucesso, mas o **próximo** kernel bloqueia a thread do host,
  e dentro de um CUDA graph capturado nada espera → o worker GPU gira para sempre após a captura (`patch_ple_offload.py:4-12`).
- **O que muda:**
  - `protocol.py`: `PleOffloadRegistration.done_flag` (int64 em memória compartilhada) e `PleOffloadRequest.seq` (`:53-75`).
  - `ple_offload_layer.py`: `reset`/`signal`/`wait_reset` e o custom op `ple_offload_wait` viram *no-ops* (`:80-116`).
  - `connector.py`: remove a thread de requisições; o próprio thread do modelo envia via socket ZMQ PUSH e então **bloqueia**
    em `_wait_done(seq)`: 2000 giros ativos, depois `sleep(50 µs)` até o flag ≥ seq, com timeout
    `VLLM_PLE_OFFLOAD_STEP_TIMEOUT` (300 s) (`:121-223`). Dummy runs só zeram buffers; `release_outputs` vira no-op.
  - `worker.py`: após copiar o resultado para o buffer GPU de cada rank TP (cópia H2D assíncrona num `copy_stream`), faz
    `copy_stream.synchronize()` de todos e escreve `flag[0] = seq` (`:389-414`). E **tabela mmap**: se
    `VLLM_PLE_PACKED_TABLE_DIR` contém `<layer>.ngram_embedding.packed_u8` (+ `.json`), os tensores de shard da PLE nem são
    lidos; `_attach_packed_table` valida largura/linhas/tamanho, cria `np.memmap(mode="r")` → `torch.from_numpy` (zero-copy,
    file-backed, **evictável**) e libera as alocações anônimas (`:274-373`).
- **Arquitetura do offload stock (para entender o patch):** um processo `PleOffloadWorker` separado constrói o modelo inteiro
  no device `meta` (sem memória), materializa **só** as subárvores `PleOffloadLayer` em CPU, filtra o iterador do checkpoint
  por prefixo, cria um mundo Gloo isolado TP1 (`orig/worker.py:117-160,349-471`); calcula a PLE **uma vez por rank DP** e faz
  *fan-out* para o buffer GPU de cada rank TP via CUDA IPC (`:613-672`); entradas (`input_ids`, `query_start_loc`,
  `ngram_context`) em buffers CPU compartilhados e *pinned* (`cuMemHostRegister`, `orig/connector.py:161-181`);
  com speculative decoding, clampa ids negativos (placeholders) a ≥0 (`orig/worker.py:329-331,633-637`). Exige
  `--distributed-executor-backend mp` (HANDOFF §4).
- **Conceitos:** CUDA stream memory operations; sincronização host-side com número de sequência; CUDA IPC; memória pinned;
  ZMQ PUSH/PULL + msgpack; construção em `meta` device; mmap vs RSS anônimo em memória unificada.
- **Efeito medido:** TP1 passou a servir; ~104 GiB → ~77 GiB de footprint não-evictável; tabela NVFP4 de 26,8 GiB no page
  cache (HANDOFF §0). "The PLE layer sits at layer 1, so the lost CPU/GPU overlap is negligible" (`patch_ple_offload.py:19`).
  Em memória unificada o offload **não reduz memória física** (HANDOFF §4, fable §3.7) — reduz pressão de UVM no load e,
  com mmap, torna as páginas evictáveis.
- ⚠ **Observação:** no `start.sh` dual, `PLE_OFFLOAD=true` apenas injeta `VLLM_PLE_CPU_OFFLOAD=1` (`start.sh:980-982,1080`)
  e **não** monta os arquivos de `files/ple_offload/` *(verificado por grep: nenhuma referência a `ple_offload` em `start.sh`)*;
  como o README manda manter `false` no dual (RAM insuficiente), isso não afeta o default.

### 3.3 `build_ple_packed_table.py` — tabela PLE NVFP4 empacotada para mmap

- **Arquivo:** `files/build_ple_packed_table.py` (80 linhas; cópia idêntica em `tp1/files/`).
- **Problema:** o checkpoint NVFP4 grava a PLE em 128 shards, cada um separado em códigos `uint8 [rows, 80]` e escalas
  `F8_E4M3 [rows, 10]`; para mmap é preciso uma matriz única com o layout que a busca devolve.
- **O que faz:** lê os headers safetensors, faz `np.memmap` de cada shard, concatena `[códigos | escalas]` por chunks de 2^19
  linhas e grava `[total_rows, 90]` em `<vname>.packed_u8` + `.json` com `rows_per_shard`, `num_shards`, `row_width`, etc.
  (`:45-80`). Renomeia o prefixo HF `model.language_model.` → `language_model.model.` (nome vLLM).
- **Conceitos:** formato safetensors (header JSON de 8 bytes de comprimento + offsets), memmap, layout "array of structs".
- **Efeito:** ~40 s, <1 GiB de RAM (HANDOFF §0, `tp1/start.sh:281`); 26,8 GiB (≈320M × 90 B).

### 3.4 `detect_ple_dtype.py` + `--hf-overrides` aninhado

- **Arquivo:** `files/detect_ple_dtype.py` (60 linhas); consumido em `start.sh:356-362` e injetado em `start.sh:924-946`.
- **Problema:** o PLE patcheado despacha por `text_config.ple_embedding_dtype`, mas o `nvidia/...` só declara a PLE FP8 em
  `quantization_config.config_groups` (local-inference-lab declara `"nvfp4"`, RadixArk declara `"float8_e4m3fn"`,
  `detect_ple_dtype.py:7-10`). Sem a chave, a PLE fica sem quant method e as linhas FP8 não carregam.
- **O que faz:** procura o grupo cujo `targets` contém `ple_embedding.ngram_embedding` e mapeia `weights.num_bits` 8 →
  `float8_e4m3fn`, 4 → `nvfp4`; `start.sh` gera **um único** `--hf-overrides '{"text_config": {...}}'` que contém também o
  `rope_parameters` do YaRN. O aninhamento sob `text_config` é essencial: um override no nível superior era `setattr` no config
  pai e nunca chegava ao `text_config` (YaRN era no-op silencioso; `start.sh:918-923`, CHANGELOG "YaRN was a silent no-op").
- **Conceitos:** configs aninhados HF (`text_config`/`vision_config`), `ModelConfig._apply_dict_overrides` do vLLM.

### 3.5 KV cache FP8 nos kernels QSA (imagem day-0) — `patch_qsa_fp8_kv.py`

- **Arquivos:** `files/patch_qsa_fp8_kv.py` (401 linhas, AGPL, vendorizado sem mudança do repo single-Spark; abordagem creditada
  a `lancelind/qwen3.8-Flash-DGX`); entradas `qsa_ops_patched.py.orig` (`nvidia/ops/qsa.py`) e `qsa_nvidia_patched.py.orig`
  (`nvidia/qsa.py`); saídas `qsa_ops_patched.py` e `qsa_nvidia_patched.py`, montadas em `start.sh:663-681`.
- **Problema:** os kernels Triton da QSA só aceitam BF16 (`supported_kv_cache_dtypes = ["auto","bfloat16"]`, asserts de dtype,
  e a checagem herdada do `FlashAttentionImpl`, que recusa KV quantizado porque o FlashAttention não o lê).
- **O que muda (ops/qsa.py):**
  - helpers `_qsa_scale_ptr` (escala 0-d fp32, 1.0 em cache por device quando ausente) e `_qsa_as_fp8` (view uint8→e4m3,
    recusa ler bytes quantizados como BF16) (`:64-103`);
  - kernel do *scorer* MQA (seleção de blocos): novos args `k_scale_ptr` e `KV_QUANT_MODE`; `keys.to(query.dtype)` antes do
    `tl.dot`; `scores *= k_scale` (`:114-139,183-211`);
  - kernel de atenção esparsa GQA split-K: args `k_scale_ptr`, `v_scale_ptr`, `KV_QUANT_MODE`; `keys/values.to(query.dtype)`
    (upcast e4m3→bf16 é exato); `scores *= k_scale` depois do dot; `normalized_output *= v_scale` no fim (`:140-182`);
  - assert de dtype: Q BF16; K/V BF16 (modo 0) ou e4m3 (modo 1) (`:232-240`).
- **O que muda (nvidia/qsa.py):** anuncia fp8/fp8_e4m3 (`:262-275`); calcula `kv_quant_mode` pelo dtype do cache e só exige Q
  BF16 (`:277-289`); passa `layer._k_scale`/`_v_scale` ao kernel (`:291-313`); **neutraliza a recusa do pai**: no `__init__`,
  entrega `"auto"` ao `FlashAttentionImpl.__init__` e restaura o dtype real depois — seguro porque o único ponto de entrada da
  camada é `forward_qsa`, que usa os kernels Triton (`:314-355`); relaxa as duas outras guardas de dtype (`:356-397`).
- **Por que escalar depois do dot:** `dot(q, s·k) = s·dot(q, k)` e `Σ p·(s_v·v) = s_v·Σ p·v` — algebricamente iguais antes
  do arredondamento; evita materializar tiles FP32 dequantizados e mantém a mesma largura de tile do BF16 no GB10 (`:9-12`).
- **Conceitos:** quantização de KV cache per-tensor; FP8 E4M3; `tl.constexpr` (eliminação de ramos em compile-time);
  hierarquia de backends de atenção do vLLM.
- **Efeito medido (2026-09-05, GMU 0.835):** pool 31,70 → 32,02 GiB; **2.131.159 → 3.652.200 tokens (×1,70)**;
  concorrência a 262k 8,13× → 13,93×; 67.229 → 114.060 tokens/GiB; `reasoning_check` 12/12 em ambos; decode
  prose/code/entropy/copy 42,5/44,5/45,4/56,5 → 40,0/47,7/47,0/56,4. **Rerun 2026-09-16:** −5,3% throughput médio
  (−1,4% prose, −9,2% code, crescendo com concorrência), `tok/step` cai em 8/8 células; ms/step plano em S=1 (60,5→60,7) e +11% em
  S=8 (84,9→94,6). ×1,70 (não ×2) porque o indexador/compressor da QSA e os estados GDN/PLE continuam BF16/fp32.
- **Risco declarado:** na implementação de referência, um benchmark de raciocínio longo caiu 6/6 → 2/6 com FP8 KV; o repo
  atribui à perturbação de *quais blocos o indexador seleciona* (`patch_qsa_fp8_kv.py:19-23`). ⚠ *(inferido)*: como o cache do
  indexador continua BF16, a perturbação provavelmente entra **indiretamente** (a saída de atenção FP8 muda os hidden states que
  alimentam os indexadores das camadas seguintes); o patch também dá ao scorer MQA a *capacidade* de ler FP8, mas a chamada que o
  `qsa_nvidia_patched.py` altera é a da atenção esparsa (não consigo ver o `.orig` para confirmar a do scorer).

### 3.6 Backport vllm#55557 para vLLM 0.30 — `patch_qsa_fp8_kv_v030.py`

- **Arquivo:** `files/patch_qsa_fp8_kv_v030.py` (581 linhas): embute o diff upstream (`:22-491`), um mini-aplicador de hunks
  unificados (`parse_diff`/`find`/`apply`, `:494-532`), detecção de "já aplicado" (`:535-542`) e gravação atômica
  (`.tmp` + `os.replace`). Alvos: `models/qwen4_exp/nvidia/qsa.py` e `.../ops/qsa.py`. Teste: `tests/test_qsa_fp8_kv_v030.py`
  (aplica, idempotente, falha sem escrever se o contexto mudou, passa direto se já aplicado).
- **Diferenças técnicas em relação ao 3.5:** escalas como **floats do host** (`layer._k_scale_float`/`_v_scale_float`):
  `softmax_scale = head_dim^-0.5 × k_scale` e `output_scale = v_scale` (dobrado no normalizador inverso por linha,
  `inv_normalizer = output_scale / max(normalizer, 1e-20)`, mantendo o LSE do split-K não escalado para o merge);
  K upcast para o dtype de Q, **V upcast para fp16** no PV dot; tabela de lançamento própria para sm_120, separada por bf16/fp8;
  warmup do Triton ciente de fp8 (`is_fp8 = kv_cache.dtype == torch.uint8`); `supports_combination` sempre aceita.
- **Efeito medido (CHANGELOG 2026-09-25):** 2,64M tokens KV, NLL 1,328/1,334, needles 3/3 a 200k, decode ~5% mais lento que BF16;
  opt-in (`OVERRIDE_KV_CACHE_DTYPE=fp8`). "Delete this file ... once the image moves to vLLM 0.31" (`:7-8`).

### 3.7 Perfis de lançamento QSA para o GB10 — `qsa_gb10/`

- **Arquivos:** `files/qsa_gb10/apply_patch.py` (gera `qsa.py` a partir de `qsa.py.orig`), `qsa.diff` (o diff resultante,
  documentação), `bench_qsa_kernels.py` (varredura).
- **Problema:** tabelas de tiles da QSA afinadas no GB300 (160 SMs); o GB10 tem 48. O perfil de decode stock
  `(block_n=16, 64 partições split-K, 4 warps)` gera ~5× mais CTAs do que cabem numa "onda" e depois um merge de 64 parciais.
- **O que muda:** injeta `GB10_PROFILES` e `_load_qsa_profile()` (lê `VLLM_QSA_PROFILE_JSON` ou `VLLM_QSA_PROFILE=gb10`),
  rotula cada bucket (`small/lt32/le256/le512/large`) e, se houver perfil, sobrescreve `(block_n, target_splits, partial_warps)`;
  para o scorer MQA, sobrescreve `tiles_per_program` e `num_warps` (`apply_patch.py:16-127`). `stock` = byte a byte upstream.
- **Benchmark:** `make_decode_case` cria Q `[rows, 24/tp, 256]`, K/V paginados página 1600, seleção aleatória ordenada de
  2048+3 posições, cache comprimido do indexador `[pages, 1600/4, 1, 128]` e `cq [rows, 4, 128]`; mede µs por chamada e escolhe o
  config que mais vence por bucket (`bench_qsa_kernels.py:38-134`).
- **Conceitos:** split-K / flash-decoding, ocupação e "ondas" de CTAs, merge por log-sum-exp, autotuning.
- **Efeito:** **não medido** ("reasoned guess... unbenchmarked: the GPU was busy", fable §8.2); default `stock`.
  ⚠ Conflito: `QSA_PROFILE≠stock` e `KV_CACHE_DTYPE=fp8` sobrescrevem o mesmo `ops/qsa.py` → `start.sh` recusa ("pick one",
  `start.sh:667-669`). Como fp8 é default, usar `gb10` exige `KV_CACHE_DTYPE=auto`.

### 3.8 Vocabulário reduzido do drafter MTP (FR-Spec, TP-aware) — `patch_mtp_draft_vocab.py` (+ `_v030`)

- **Arquivos:** `files/patch_mtp_draft_vocab.py` (227 linhas; entrada `mtp_patched.py.orig` = `nvidia/mtp.py` da imagem;
  saída `mtp_patched.py`), `files/patch_mtp_draft_vocab_v030.py` (reusa os mesmos blocos de código para
  `qwen4_exp/nvidia/mtp.py` do vLLM 0.30), `files/build_draft_vocab.py`, `files/test_draft_vocab.py`,
  `files/draft_vocab_en_code_47k.txt`, `tests/test_draft_vocab_v030.py`.
- **Problema:** vocabulário de 248.320 e o drafter MTP tem **seu próprio** `ParallelLMHead` BF16 (`tie_word_embeddings=false`):
  248320 × 2560 × 2 B = 1,18 GiB, 0,59 GiB por GPU em TP=2, lido **uma vez por passo de rascunho**; com MTP=3 são 3 das 4
  leituras de lm_head do passo (~2,4 de ~9,2 GiB por GPU) (`patch_mtp_draft_vocab.py:8-12`).
- **O que muda:**
  - `_attach_draft_vocab(model)` (`:49-113`), chamado após `load_weights`: lê `VLLM_MTP_DRAFT_VOCAB` (um id por linha), filtra
    ids válidos, pega **só os ids no intervalo `[org_vocab_start_index, org_vocab_end_index)` deste rank**, copia essas linhas do
    shard do lm_head para `_draft_lm_head_weight` e guarda o mapa `_draft_id_to_target_id`. Recusa engatar se escala de logit ≤ 0.
    Loga "`47149 of 248320 tokens (19.0%), 45734 on this rank of 2; draft lm_head shard 0.59 -> 0.22 GiB per draft step`".
  - `get_top_tokens(hidden)` (`:118-155`): `logits = hidden @ W_draftᵀ`, argmax local; se o rank não tem ids, "lance perdedor"
    `-inf`; com TP>1 faz all-gather do par `(valor, id_global)` e escolhe o rank vencedor — O(batch·2·tp) em vez de O(batch·vocab).
    Pula escala e soft-cap porque são monotônicos e iguais em todos os ranks (argmax invariante).
  - `compute_logits` continua com o vocabulário cheio (todo o resto do sistema não muda).
- **Por que é seguro:** o drafter é greedy (argmax); um token fora do subconjunto só vira um rascunho **rejeitado** na verificação
  do modelo alvo — custo em aceitação, nunca em correção (`:14-19`).
- **`build_draft_vocab.py`:** conta frequências de ids sobre um corpus (idealmente a **saída do próprio modelo**, gerada por
  `bench/gen_draft_corpus.py`), mantém sempre tokens especiais/adicionados, ranqueia, opcionalmente completa com ids baixos não
  vistos (`--fill-to-size`, proxy de ordem de merge BPE) balanceando por faixas de TP (`--balance-shards`), e imprime cobertura em
  8k/16k/32k/64k/128k (`:58-172`).
- **`test_draft_vocab.py`:** roda as funções patcheadas reais com um lm_head vocab-parallel falso (VOCAB 100, TP 2, 23 ids
  desbalanceados), com o coletivo *stubado*, e compara com o argmax do vocabulário cheio restrito ao conjunto; testa o rank vazio.
- **O arquivo `draft_vocab_en_code_47k.txt`** *(descrito, não copiado)*: 47.149 ids inteiros, ordenados, únicos, de 0 a 248.076;
  densos nos ids baixos (8.735 ids entre 0-9.999, 7.086 entre 10.000-19.999, 5.940 entre 20.000-29.999, ... e apenas dezenas a poucas centenas por faixa de 10k acima de 100.000) — reflexo da
  ordem de merge do BPE; os últimos 33 ids (248044-248076) são tokens especiais (BOS/EOS 248044, visão 248053/248054, imagem
  248056, vídeo 248057, etc.). Vendorizado sem alterações do PR #39 do repo single-Spark, afinado em 30 MiB de código+docs
  (100% de cobertura do corpus, 99,58% held-out) → pode rascunhar pior para outros idiomas (especialmente chinês).
- **Conceitos:** decodificação especulativa, MTP, FR-Spec, lm_head vocab-parallel (sharding por faixa de id), gargalo do rank mais
  lento, redução distribuída de argmax, cobertura de frequência, rejection sampling sem perda.
- **Efeitos medidos:** ver §4.3 (2026-09-05 histórico 65k; 2026-09-11 A/B +9,6%; 2026-09-16 rerun +8,4%; lane v0.30 volta à
  paridade com day-0). Ganho é **todo em tempo de passo** (code S=1 61,2 → 54,0 ms; prose 59,9 → 53,3 ms) com tokens/passo inalterados.
  ⚠ Conflito: o overlay do FP8-dense também monta `nvidia/mtp.py` → `start.sh` recusa os dois juntos (`start.sh:642-643`).

### 3.9 Alias do índice absoluto da camada MTP — `patch_checkpoint_config.py`

- **Arquivo:** `files/patch_checkpoint_config.py` (201 linhas); saídas `files/config_patched.json` e
  `files/hf_quant_config_patched.json` (gitignored), montadas **por cima dos arquivos do snapshot dentro do container**
  (`start.sh:692-736`) — o cache HF nunca é modificado.
- **Problema:** o vLLM constrói a camada MTP no **índice absoluto** que continua a pilha (`mtp.layers.48` com 48 camadas) e
  `ModelOptMixedPrecisionConfig._resolve_quant_algo` casa prefixos com `quantized_layers` por string exata. O `nvidia/...` só
  registra `mtp.layers.0.*` → o MoE da MTP é construído **não quantizado** e o load morre com
  `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'` (`:14-20`).
- **O que muda:** `add_aliases` adiciona `mtp.layers.{48+i}.*` para cada `mtp.layers.{i}.*` em `quantized_layers` e espelha em
  `config_groups[*].targets` (`:43-81`); `normalize_mtp_fp8_pb_wo` reescreve `FP8_PB_WO` (com `group_size 128`) → `FP8_BLOCK_SCALES`
  nos experts MTP (`:104-127`); patcheia `config.json` **e** o legado `hf_quant_config.json` (o caminho do draft model lê o legado;
  a evidência do issue #38 mostra que o dispatch do MoE consome `config.json`). Modo `--mtp-moe-algo <snapshot>` imprime o
  algoritmo e sai com 3 se não for suportado (`:130-144,191-196`) — o preflight falha em segundos.
- **Conceitos:** metadados de quantização ModelOpt (`quantized_layers`, `config_groups`, `exclude_modules`), nomenclatura de
  camadas vLLM vs HF, indexação absoluta da MTP, bind-mount de arquivos de config.

### 3.10 Fallback MXFP8 → emulação BF16 — `patch_modelopt_mxfp8.py`

- **Arquivos:** `files/patch_modelopt_mxfp8.py` (158 linhas; cópia idêntica em `tp1/files/`), entrada `modelopt_patched.py.orig`
  (`vllm/model_executor/layers/quantization/modelopt.py`), saída `modelopt_patched.py` montada nos dois nós (`start.sh:836-867`).
- **Problema:** `FlashInferCutlassMxfp8LinearKernel`/`mm_mxfp8` só aceita `N≥128, N%32==0, K≥128, K%32==0` (medido no sm_121,
  independente do nº de tokens: N=128/160/.../288 ok; N=136/144/.../240 `ValueError`; K=2152/4304 `AssertionError`, `:5-11`).
  `linear_attn.in_proj_a/b [48,2560]` (×72) → "mm_mxfp8 requires N >= 128, got N=48" — fatal no primeiro forward do *profile run*
  (~7 min após o launch); `visual.blocks.*.mlp.linear_fc1 [4304,1152]` (×27) → "Problem size is not supported".
- **O que muda:** insere `ModelOptMxFp8EmulationLinearMethod` e `_mxfp8_native_kernel_supports(n,k)` (`:39-78`); dentro de
  `ModelOptMxFp8LinearMethod.create_weights`, com as dimensões **pós-split de TP** que o kernel verá, troca o kernel por
  `EmulationMxfp8LinearKernel` se o nativo não aceita (por camada, loga uma vez por forma) (`:82-103,129-137`); no dispatch
  `quant_algo == "MXFP8"`, prefixos `visual.` vão sempre para emulação (`:139-148`). Emulação = dequant MXFP8 → BF16 **uma vez no
  load**; essas camadas rodam como linears BF16 (~17 MB extras para os 72 `in_proj_a/b`).
- **Conceitos:** MXFP8 (OCP microscaling: blocos de 32 com escala E8M0 compartilhada), restrições de forma de kernels GEMM,
  dequantização em load-time, dispatch por camada.
- ⚠ **Inconsistência de fonte:** o docstring diz que as formas violadoras são de `RadixArk/...NVFP4`, mas o HANDOFF §2 diz que a
  atenção MXFP8 é do `local-inference-lab/...` e que RadixArk é BF16; o model card NVIDIA diz que o `nvidia/...` é BF16 fora dos
  experts. *(inferido)*: no checkpoint default (nvidia) o patch provavelmente é inerte; ele é necessário para os checkpoints com
  camadas MXFP8.

### 3.11 Ramo `FP8_BLOCK_SCALES` para experts roteados — `patch_modelopt_fp8_block_moe.py`

- **Arquivo:** `files/patch_modelopt_fp8_block_moe.py` (147 linhas); opera **in place** em `modelopt_patched.py` (empilha sobre 3.10;
  idempotente; "upgrade" de versões anteriores do próprio patch para aceitar o alias `FP8_PB_WO`).
- **Problema:** os experts roteados da MTP do nvidia são FP8 com escalas por bloco 128×128, mas
  `ModelOptMixedPrecisionConfig.get_quant_method` só constrói `RoutedExperts` para FP8 / NVFP4 / W4A16_NVFP4 / MXFP8 e devolve
  `None` para o resto → MoE silenciosamente não quantizado → morte no load (~7 min). Lacuna também no upstream (commit `d4d703ca` e `main`).
- **O que muda:** helper `_fp8_block_scales_config(prefix)` que acha a entrada de `quantized_layers` (por candidatos de prefixo e
  prefixo-pai), exige `group_size` inteiro > 0 e devolve `Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
  weight_block_size=[g, g])` (`:30-64`); no dispatch MoE, `quant_algo in ("FP8_BLOCK_SCALES","FP8_PB_WO")` →
  `Fp8MoEMethod` (que já nomeia as escalas `weight_scale_inv`, exatamente como o checkpoint grava) (`:78-101`).
- **Conceitos:** FP8 block-wise (estilo DeepSeek-V3: escala por bloco 128×128 dos pesos, ativações dinâmicas), `weight_scale_inv`.
- **Efeito medido:** habilita MTP: 24,5 → 52,1 tok/s, aceitação 72,8% (823/1131), por posição 89/74,5/60%. "A wrong block shape
  would show up as near-random acceptance, not as a crash" (README).

### 3.12 Backport vllm#53388 (`disable_eagle_block_drop`) — `patch_block_drop.py`

- **Arquivo:** `files/patch_block_drop.py` (155 linhas, AGPL); originais extraídos para `files/block_drop/orig/<path>`, saídas
  `files/block_drop/<path>`; teste `tests/test_block_drop_overlay.py`.
- **Problema:** com drafters estilo EAGLE (MTP incluído), o prefix cache **descarta o último bloco casado** de uma requisição e o
  recomputa (o drafter precisa dos hidden states do alvo para essas posições). Num chat multi-turno isso é um bloco inteiro
  (1.664 tokens com MTP 3, `:6-8`) de prefill extra por turno, e com o bug #62 o segundo turno perdia o cache inteiro.
  A imagem não conhece a chave e o `SpeculativeConfig` rejeita chaves desconhecidas.
- **O que muda:** `SpeculativeConfig.disable_eagle_block_drop: bool = False` + método `use_eagle_block_drop()`
  (= `use_eagle() and not disable_eagle_block_drop`); `kv_cache_utils`, `scheduler` (`last_cache_position -= block_size` só se
  `use_eagle_block_drop`) e três conectores de KV (mooncake, offloading, simple CPU offload) passam a usar a nova checagem
  (`:55-111`). Patcheia tudo antes de gravar (sem conjunto parcial); se a imagem já tem a opção, remove saídas antigas e não monta
  nada (`:137-144`). O `start.sh` monta **só os 3 arquivos core** (`start.sh:748-750`): os conectores não se aplicam sem KV
  connector e colidiriam por nome-base em `/tmp/vllm-overlay`. O 7º arquivo do PR upstream (sliding window) foi omitido porque a
  QSA não suporta sliding window.
- **Conceitos:** prefix caching por hash de blocos, alinhamento de blocos no cache híbrido (`mamba cache mode = align`), drafters
  EAGLE/MTP e dependência de hidden states do alvo.
- **Efeito medido (2026-09-25, medianas de 3 sessões):** TTFT turno 1 **3,25 → 0,63 s**; turno 2 1,12 → 0,59 s; turno após
  4k de tool output 2,55 → 1,98 s; decode, prefill e NLL inalterados. "The drafter still runs. The change can move acceptance only"
  (`:13-14`). Nativo no vLLM 0.30.

### 3.13 Decodificação greedy reprodutível — `patch_determinism.py`

- **Arquivo:** `files/patch_determinism.py` (82 linhas, AGPL); teste `tests/test_determinism_patch.py`.
- **Problema:** em TP2+EP, requisições idênticas davam logprobs e saídas greedy diferentes (spread mediano 0,25 nats, máx 6,07;
  5/5 saídas distintas) — ordem de acumulação em ponto flutuante variável.
- **O que muda:**
  - em `qsa_ops_patched.py` (a saída do patch FP8, ou gerada a partir do original quando o KV é bf16 — nesse caso o patch FP8 é
    aplicado mas fica inerte, `start.sh:754-765`): `_SORTED_TOPK = os.getenv("VLLM_QSA_DET_TOPK") == "1"`; após o `topk_op`, se
    ativo, troca `-1` por `INT32_MAX`, ordena cada linha e restaura `-1` no fim (`:25-39`). A **mesma** seleção passa a ser visitada
    sempre na mesma ordem → mesma ordem de soma no split-K.
  - em `fused_moe/experts/flashinfer_cutlass_moe.py`: `use_fused_finalize = os.getenv("VLLM_MOE_DET_FINALIZE","0") != "1"`
    (`:41-52`); o `start.sh` também injeta `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0` e um diretório de autotune separado.
- **Conceitos:** não-associatividade da soma em float, determinismo de kernels, finalize fundido do MoE (redução top-k das saídas
  dos experts), autotune cache.
- **Efeito medido (2026-09-25):** spread → **0/0**; saídas greedy 5/5 distintas → 1/5 (também a frio e depois de outro prompt);
  custo ~−4% decode prose e ~−4% prefill → opt-in. Não portado para a lane v0.30.

### 3.14 Checkpoint híbrido NVFP4 (experts) + FP8 (densos) — `fp8dense/`

- **Arquivos:** `files/fp8dense/make_fp8_dense_checkpoint.py` (321 linhas), `build.sh`, `verify_fp8_dense_checkpoint.py`,
  `compute_quant_stats.py`, `test_quant_config_resolution.py`.
- **Problema:** no batch 1 o passo de decode é limitado por **banda de memória**; as projeções densas BF16 (GDN, atenção, HC,
  shared expert, lm_head) são ~6,9 dos ~9,2 GiB lidos por passo por GPU. O `Qwen/...-FP8` oficial **não** ajuda: mantém exatamente
  esses densos em BF16 e só põe os experts em FP8 (mais bytes que NVFP4) (fable §3.1).
- **O que faz o conversor:**
  1. Hard-link (do *blob*, não do symlink do snapshot) de todos os shards não-BF16 (experts NVFP4, PLE FP8) e arquivos auxiliares;
     fallback para symlink se hard-link falhar (`:219-246`).
  2. Reescreve só os shards `model-bf16-*` em streaming (chunks de 16 MiB de linhas): tensores que casam `QUANT_PATTERNS` viram
     `F8_E4M3` + `weight_scale` `F32 [out]`; o resto é copiado verbatim em pedaços de 64 MiB (`:151-201`). Um escritor safetensors
     próprio grava o header primeiro (tamanhos conhecidos) e escreve com `seek` (`:97-139`).
  3. Quantização **por canal de saída, simétrica**: `s = amax_linha / 448`, `q = clamp(x/s, ±448).to(e4m3)` (`:142-148`);
     ativações serão quantizadas **dinamicamente por token** em runtime → sem dados de calibração, sem restrição de bloco (as
     projeções rank-320 do HC e o shared expert TP-sharded com 320 linhas não são divisíveis por 128; além disso o loader de
     `ParallelLMHead` do vLLM fatia escalas de bloco errado, enquanto escalas por canal fatiam como o peso — fable §3.1).
  4. Escreve `config.json` com `quantization_config` ModelOpt `MIXED_PRECISION` (`quantized_layers` por tensor: 591 FP8
     `FP8_PER_CHANNEL_PER_TOKEN`, experts `NVFP4` `group_size 16`, `ignore = EXCLUDE_MODULES`) e o `hf_quant_config.json` legado;
     novo `model.safetensors.index.json`; estatísticas `fp8_dense_quant_stats.json` (`:276-316`).
- **`build.sh`:** resolve o snapshot de origem com `resolve_snapshot.py` (mensagens claras de download parcial), cria o repo
  destino `models--MiaAI-Lab--Qwen3.8-Flash-Next-NVFP4-FP8dense` com `refs/main = fp8dense-<rev8>` (escrito com `echo -n`, sem
  newline — o README avisa que `huggingface_hub ≥1.x` falha offline com newline em `refs/main`), roda o conversor e o verificador
  dentro da imagem com `--memory=1200m --cpus=4 CUDA_VISIBLE_DEVICES=` (CPU-only) e faz `chown` (`build.sh:18-46`).
- **Verificação:** todos os tensores de origem existem no destino; cada FP8 tem escala `F32 [rows]`; shards intocados são o
  **mesmo inode**; amostras de dequant com erro relativo ≤ 0,06 e cosseno (`verify...py:36-95`). `compute_quant_stats.py` recalcula o
  RMSE relativo de todos os 591 tensores. `test_quant_config_resolution.py` confere, com o `modelopt.py` do overlay montado, o
  método resolvido para 22 prefixos do caminho `ForConditionalGeneration` e 8 do caminho `CausalLM/MTP` (ex.: `in_proj_qkvz` →
  FP8, `in_proj_ba` → EXCLUDED, `mlp.experts` → NVFP4, `layers.1.ple.key_proj` → EXCLUDED, `lm_head` → FP8, `mtp.*` → EXCLUDED).
- **Efeito:** build ~2 min, ~1 GB de RAM; 202 shards hard-linkados, 4 shards BF16 reescritos (11,8 GB), índice total 123 GiB;
  RMSE relativo min/mediana/max **0,0240 / 0,0262 / 0,0268** (pior: os pequenos gates `block_inject_weight`), cosseno ≥ 0,9996;
  resolução 30/30 correta (fable §8.1). **Esperado analiticamente:** ≈9,2 → ≈5,7 GiB por passo → ~1,5× tok/s batch-1
  (fable estima 85-90 tok/s) — **nunca medido em GPU**; GSM8K/AIME pendentes.
- ⚠ **Problemas de integração observados** *(inferidos da leitura, não testados)*: (1) `start.sh:579` roda
  `files/overlay/apply_patches.py` para gerar os `.py` do overlay, mas esse arquivo **não está no repositório** — o padrão
  `files/overlay/*.py` do `.gitignore` também o ignora; só os `.diff` foram versionados. (2) Com `FP8_DENSE=true`, o overlay monta
  `overlay/modelopt.py` e o passo 6b monta `modelopt_patched.py` **no mesmo caminho** (`start.sh:580` e `:866`), e o
  `add_overlay` não vê o mount do 6b; provável "duplicate mount point" no Docker. (3) Exige esvaziar `MTP_DRAFT_VOCAB` (conflito em
  `mtp.py`). (4) SRC default é RadixArk, não o nvidia.

### 3.15 Overlay do loader para o FP8-dense — `overlay/*.diff`

- **Arquivos:** `files/overlay/hyperconnection.diff`, `model.diff`, `modelopt.diff`, `mtp.diff` (datados 2026-09-02).
- **O que muda:**
  - `hyperconnection.py`: `GatedResidual.__init__` ganha `quant_config=None` e o repassa aos três linears
    (`input_mix_weight_down_block_inject` — o down low-rank empacotado com os gates de injeção e padding, `disable_tp=True` —,
    `input_mix_weight_down`, `input_mix_weight_up`) em vez do `quant_config=None` fixo. Para checkpoints não quantizados vira
    `UnquantizedLinearMethod` (no-op).
  - `model.py`: passa `quant_config` aos dois `GatedResidual` de cada camada, ao `hyper_connection_mixer` final e ao `ParallelLMHead`.
  - `mtp.py`: idem para o mixer e o lm_head do drafter.
  - `modelopt.py`: no `from_config` do `ModelOptMixedPrecisionConfig`, cria configs irmãs `fp8_pcpt_config`
    (`FP8_PER_CHANNEL_PER_TOKEN`) e `fp8_pbwo_config` (`FP8_PB_WO`); candidatos de prefixo extras (`model.` → `model.language_model.`
    para a entrada texto-só; `lm_head`); no dispatch de `LinearBase/ParallelLMHead`, `FP8_PER_CHANNEL_PER_TOKEN` →
    `ModelOptFp8PcPtLinearMethod`, `FP8_PB_WO` → `ModelOptFp8PbWoLinearMethod`.
- **Kernel esperado:** `ModelOptFp8PcPtLinearMethod` → `init_fp8_linear_kernel(ativação dinâmica por token, peso estático por
  canal)` → `CutlassFP8ScaledMMLinearKernel` (a imagem traz cubins `sm_120` não-`a` que o GB10 cc 12.1 executa) (fable §8.1).
- **Conceitos:** quantização weight+activation (W8A8) por canal/por token, dispatch por prefixo, camadas que hard-codam "sem quantização".

### 3.16 Infraestrutura auxiliar

| Nome | Arquivo | Problema | Mecanismo | Conceito | Efeito |
|---|---|---|---|---|---|
| Evicção do page cache do checkpoint | `files/evict_page_cache.py` (83 linhas) | No GB10 o page cache divide o pool unificado com o modelo; checkpoint recém-baixado/rsyncado deixa páginas residentes → CUDA OOM no load (issue #35) | `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` em `.safetensors/.bin/.pt/.gguf`, só páginas limpas, sem root, nunca falha o launch; no modo NFS roda num container descartável com o mesmo volume | Page cache, memória unificada | 4,15 → 0,00 GiB residente por shard; NFS 4.2: 3,00 → 0,00 GiB |
| Resolução de snapshot | `files/resolve_snapshot.py` (67 linhas) | Download interrompido parecia "pronto" | Completo = todos os shards do index existem; prefere `refs/main` completo, senão o completo mais novo | Layout do cache HF (`refs/`, `snapshots/`, `blobs/`) | Exit 0/1/2 |
| Watchdog de memória | `files/memwatch.sh` (30 linhas) | Esgotar memória unificada trava o kernel (sem OOM, sem logs) | Poll 1 s de `MemAvailable`; `docker kill` abaixo do piso (default 6 GiB); log a cada 5 s | Segunda linha de defesa atrás do cgroup | Usado no TP1 |
| Compartilhamento NFS | `files/nfs-share.sh` + `files/nfs-server/{Dockerfile,entrypoint.sh}` | Evitar cópia de ~126 GiB no worker | Container Alpine privilegiado com nfsd do kernel exportando o HF cache (`/export` ro, `fsid=0`); volume Docker NFS 4.2 no worker (`nconnect=8`, 1 MiB rsize/wsize, `hard`); detecta servidor NFSv4 "estrangeiro" e monta pelo caminho absoluto | NFSv4 pseudoroot, RoCE/ConnectX | Cold start transmite ~126 GiB pelo CX7 (lazy safetensors) |

### 3.17 `html/` — páginas-demo (não são conceitos de ML)

- `html/BRIEF.md` é um **contrato de autoria** para um agente gerar **uma** página web autocontida por conceito: dois entregáveis
  (`100-html/NNN-slug.html` + `.txt` com os rótulos `TITLE/PROMPT/DESIGN/TECHNIQUES/INTERACTION`), sem dependências externas, só
  fontes do sistema, `prefers-reduced-motion`, 700-950 linhas, escrita em chunks com sentinelas `<!--CHn-->` ("anti-stall rule"),
  e três *gates*: `verify100.py` (estático), `render100.py` (Chromium headless, `console=0`) e overflow responsivo ≤2 px em
  390/768/1440 via `probe.html`/`probe-deep.html` (este também conta SVG/paths/canvas para provar que a arte existe).
- `html/concepts.md` (80 KB) é a lista-mestra de 100 conceitos **de design visual** (ex.: "Ferroflux: A Magnetic Fluid Cabinet",
  "Halide: A Cyanotype Archive", "Tessellate", "Sonar"...), cada um com Subject/Direction/Palette/Type/Layout/Motion/Interaction,
  e um "diversity ledger" de famílias estéticas; `html/concepts/*.md` são os mesmos briefs (~900 bytes) um por arquivo (87 arquivos).
- `html/100-html/` contém ~15 páginas geradas (+ `.txt`), `html/shots/` screenshots. O CHANGELOG descreve como "rendering probes and a
  100-shot visual sample set". *(inferido)*: é uma carga de trabalho agêntica de exemplo (geração de front-end) servida por este
  deployment; o repo não diz explicitamente qual modelo gerou as páginas. **Nada ali explica arquitetura, quantização ou inferência.**

---
## 4. Números e benchmarks

Todos medidos no kit de 2× DGX Spark (GB10, sm_121, 121,69 GiB unificados, LPDDR5x ~273 GB/s teóricos, 48 SMs), TP2+EP, salvo
indicação. "S" = streams concorrentes. Fontes entre parênteses.

### 4.1 Memória, capacidade de KV e tamanhos

| Item | Valor | Data / fonte |
|---|---|---|
| Tabela PLE (51,2 B parâmetros) em BF16 / FP8 / NVFP4 empacotado | ≈95,4 GiB (≈102 GB) / 47,7 GiB (51,2 GB) / 26,8 GiB (90 B/linha) | README, fable §2, HANDOFF §2, LMSYS |
| PLE FP8 por GPU em TP2 (vocab-sharded) | ~24 GiB | fable §1 |
| Estado PLE por requisição | ids recentes (2 tokens) + short-conv `[10240, 9]` | LMSYS; `ple_layer_patched.py:934-936` |
| lm_head do drafter | 1,18 GiB (0,59 GiB/GPU TP2); com 47k ids: rank 0 0,22 GiB, rank 1 ~0,01 GiB | `patch_mtp_draft_vocab.py:8-12`, CHANGELOG 2026-09-10 |
| KV por token (atenção pura, 13 camadas) | bf16 26.624 B (13.312/GPU); fp8 13.312 B (6.656/GPU) | README "KV cache budget" |
| KV medido por token/GPU | fp8 9.418 B (41% acima do puro); bf16 15.192 B (14% acima) | README |
| Bloco de atenção | 1.600 tokens (casado com a página GDN); 1.664 tokens com MTP 3 citado no block-drop | README; `patch_block_drop.py:7` |
| Pool KV, GMU 0.835, fp8 vs bf16 | 32,02 GiB / **3.652.200 tok** (13,93× a 262k) vs 31,70 GiB / 2.131.159 (8,13×) | 2026-09-05 |
| Rerun (orig bf16 / fp8) | 2.426.193 (9,26×) / 4.378.829 (16,70×), ×1,80 | 2026-09-16 |
| Lane v0.30, GMU 0.80 | BF16 1.467.072 tokens (5,6×); fp8 backport 2,64M | 2026-09-25 |
| Pesos residentes nvidia NVFP4 | 62,72 GiB/nó (66,14 com non-torch); com MTP 64,3 GiB | README |
| Bytes por passo de decode por GPU (batch 1, MTP3) | ≈9,2 GiB: lm_head ×4 2,36 (26%), experts ~2,1 (23%), GDN 1,95 (21%), HC 1,19 (13%), atenção+indexer 0,60, MTP ~0,5, router+shared+PLE proj 0,40, estado GDN 0,1 | fable §2 (analítico) |
| Roofline | 36 ms/passo teórico; 40-42 ms real → 24-25 passos/s → 53-56 tok/s com ~2,2 tok aceitos/passo | fable §2 |
| Coletivos por passo | ≈106 (36 GDN + 12 atenção + 48 MoE + 1 gather de logits + 9 dos rascunhos), 5-20 KB cada, 2-3 ms/passo | fable §7.1 |
| MXFP8 → BF16 emulação | ≈17 MB extras (72 × `[48,2560]`) | `patch_modelopt_mxfp8.py:25-26` |
| PLE offload TP1 | ~104 → ~77 GiB não-evictável; build da tabela ~40 s; handoff: KV 685.825 tokens a GMU 0.813 | HANDOFF §0 |

### 4.2 MTP e decode

| Medição | Valor | Data / fonte |
|---|---|---|
| MTP off → MTP=3 (batch 1, greedy) | 24,5 → **52,1 tok/s (2,13×)**; aceitação 72,8% (823/1131); por posição 89% / 74,5% / 60% | CHANGELOG [Unreleased] |
| MTP acceptance por conteúdo (checkpoints anteriores) | 62,8-92,6% (comprimento aceito 2,88-3,78 de 4) | HANDOFF §3 |
| Rerun: aceitação prose vs code sintético | prose p1/p2/p3 = 0,81/0,64/0,51; code 1,00/0,99/0,99 (prompt de 50 helpers idênticos) | rerun 2026-09-16 |
| IndexShare MTP | code +6,1 / +4,6 / +21,3 / +26,6% (S=1/2/4/8); prose +0,3 / +5,3 / −0,1 / −1,8%; NLL 1,328 vs 1,334; HumanEval 156/164 | 2026-09-25 |
| Block drop desligado | TTFT turno 1 3,25 → 0,63 s; turno 2 1,12 → 0,59 s; pós-tool 2,55 → 1,98 s | 2026-09-25 |
| Determinismo | spread 0,25/6,07 nats → 0/0; saídas 5/5 → 1/5; −4% decode e prefill | 2026-09-25 |

### 4.3 Vocabulário reduzido do drafter

| Experimento | Resultado | Data |
|---|---|---|
| 65.536 ids, preenchimento pelos ids mais baixos (65.392 no rank 0) | head/rank 0,31 GiB; prose/code/entropy/copy 40,3/44,1/42,3/65,0 tok/s; aceitação 53,2% | 2026-09-05 (bf16 KV, 400 tokens, 1k ctx, greedy) |
| 65.536 ids `--balance-shards 2` (32.768/32.768) | 0,16 GiB; 42,5/44,5/45,4/56,5; aceitação **47,8%** (vs 56,5% vocab cheio: 40,0/43,2/40,1/59,1) | 2026-09-05 |
| Corpus daquele vocab | 85 documentos, 49.141 ocorrências, só 8.305 ids distintos (cobertura satura em 16.384) | 2026-09-05 |
| 47k (default) vs cheio, sweep sparkDash 600 tokens | code 62,0→71,5 (+15,3%), 113,8→124,9, 193,4→211,1, 312,4→325,0 (+4,0%); prose 49,2→56,8 (+15,5%), 84,1→91,6, 134,8→146,0, 213,4→225,3; **+9,6% médio**; tok/step code 3,80→3,85, prose ~2,9 | 2026-09-11 |
| Rerun independente | +8,4% (prose +7,4%, code +9,3%); prose S=1 59,3→51,9 ms/passo, code 60,7→54,0 | 2026-09-16 |
| Single-Spark (referência) | +13,1% médio | PR #39 do repo single-Spark |
| Lane v0.30 stock (vocab cheio) vs com port | 58,3 ms/passo S=1 (5-15% mais lento); o port volta à paridade day-0 (±4%) | 2026-09-25 |

### 4.4 FP8 KV (qualidade e throughput)

| Medição | Valor | Data |
|---|---|---|
| `reasoning_check.py` | 12/12 bf16 e fp8 (needles a 13k e 104k) | 2026-09-05 |
| Referência upstream | raciocínio longo 6/6 → 2/6 com FP8 KV | citado em `patch_qsa_fp8_kv.py:19-21` |
| Throughput | −5,3% médio (−1,4% prose, −9,2% code); ms/step S=8 84,9 → 94,6 | 2026-09-16 |
| Lane v0.30 fp8 | NLL 1,328/1,334, needles 3/3 a 200k, ~5% mais lento | 2026-09-25 |

### 4.5 Throughput geral (contexto para os patches)

| Medição | Valor | Data |
|---|---|---|
| Day-0 lane (fp8 KV + 65k balanceado) decode prose | 54,4 / 86,5 / 128,1 / 169,7 / 207,0 tok/s agregados (S=1/2/4/6/8); TTFT 160 ms em S=1 | README |
| Prefill day-0 | 2.874,9 (8k) … pico 2.962,2 (32k) … 2.727,1 tok/s (128k) | README |
| Lane v0.30 (BF16 KV, 47k vocab) | code S=8 313,6; prefill pico 3.489,8 tok/s (16k) | 2026-09-27 |
| v0.30 lane com breakable cudagraph off | prose 60,0/95,5/145,2/234,8, code 76,5/140,2/248,0/441,2 (S=1/2/4/8) | 2026-09-25 |
| Qualidade do checkpoint RadixArk (referência) | GSM8K 97,27% (1283/1319), AIME26 pass@1 98,75% | fable §7.4 |
| FP8-dense (analítico) | 9,2 → 5,7 GiB/passo, ~1,5× batch-1; RMSE 0,0240/0,0262/0,0268 | 2026-09-02 |
| Cold start | ≈10m55s (NCCL ~40 s, load 458 s = alvo 392 + drafter 67, engine init 92,4 s, graphs ~7 s) | README |

---

## 5. Conceitos necessários (em ordem de pré-requisito)

### Nível 0 — Fundamentos de ML / DL

1. **Tensor, dtype e bytes por elemento.** Um tensor é um array n-dimensional; seu custo em memória é `nº de elementos × bytes do
   dtype` (fp32 = 4, bf16 = 2, fp8 = 1, fp4 = 0,5). Todo o raciocínio desta receita é contar bytes: "0,59 GiB por passo de rascunho",
   "90 bytes por linha da PLE", "13.312 B/token de KV".
2. **Camada linear / GEMM.** `y = xWᵀ`: multiplicação de matrizes, a operação dominante de uma LLM. Nas formas `[N, K]` citadas
   (ex.: `in_proj_a [48, 2560]`), N = saídas, K = entradas; kernels de GEMM têm restrições de forma (múltiplos de 32/128), origem do
   patch MXFP8.
3. **Embedding (tabela de consulta).** Uma matriz onde a linha i é o vetor do id i; "forward" é só indexar (`F.embedding`,
   `index_select`). A PLE é uma embedding gigante (320M linhas) indexada por hash de n-gramas em vez do id do token.
4. **Funções de ativação e gates (SiLU, sigmoid).** Sigmoid ∈ (0,1) serve de "porta" multiplicativa; SiLU = x·sigmoid(x). O PLE usa
   `sigmoid(sign·sqrt|s|)` como gate por ramo e SiLU após a conv; a QSA usa gate sigmoid na saída.
5. **Normalização (RMSNorm).** Divide o vetor pela raiz da média dos quadrados e multiplica por um ganho. Aqui aparece como
   `Qwen3_8FlashNextPLEGroupedNorm` (norm por grupos de 2560, ganho `1+w`), QK-norm na atenção e norms do GDN.
6. **Convolução 1D causal, depthwise e dilatada.** Cada canal é convoluído separadamente (depthwise), só com o passado (causal), com
   "buracos" de passo d (dilatação). A short-conv do PLE tem kernel 4 e dilatação 3 → enxerga 9 posições passadas; o GDN também tem
   conv curta kernel 4.
7. **Hashing e aritmética modular.** Mapear uma chave grande (um trio de tokens) para um índice num espaço menor. O PLE usa
   multiplicação por constantes ímpares + XOR + módulo por um primo; várias cabeças com primos diferentes reduzem colisões
   simultâneas.

### Nível 1 — Transformers e LLMs

8. **Modelo de linguagem autoregressivo.** Prediz o próximo token dado o prefixo; gera um token por passo. Tudo em "decode" é um
   passo por token (ou por grupo de tokens com especulação).
9. **Tokenização BPE e vocabulário.** O texto vira ids de um vocabulário (248.320 aqui). Em BPE byte-level, ids baixos tendem a ser
   merges mais frequentes — base da heurística `--fill-to-size` do draft vocab.
10. **Stream residual e blocos do Transformer.** Cada bloco lê o residual, aplica um "token mixer" (atenção) e um MLP, e soma de
    volta. O Qwen3.8-Flash-Next alarga esse residual para 4 ramos (Gated Residual).
11. **Self-attention, Q/K/V, softmax, escala 1/√d.** Scores = QKᵀ/√d, softmax, soma ponderada de V. Nos kernels QSA,
    `softmax_scale = head_dim^-0.5`, convertido para base 2 (`× 1,4427`) para usar `exp2`.
12. **GQA (grouped-query attention).** Várias cabeças de Q compartilham poucas cabeças de K/V (24 Q : 2 KV aqui) → KV cache menor.
    Em TP2 cada GPU tem 12 Q e 1 KV (`bench_qsa_kernels.py:39-40`).
13. **Atenção com gate de saída (gated attention).** A saída de cada cabeça é multiplicada por um sigmoid de uma projeção extra
    (`output_gate_type=sigmoid`), o que explica a projeção `qkv+gate` maior.
14. **RoPE, RoPE parcial e mRoPE.** Posições codificadas por rotação de pares de dimensões. Aqui só 25% das dims giram
    (`partial_rotary_factor`), `rope_theta=1e7`, com seções multimodais (t,h,w) = (11,11,10). YaRN reescala as frequências para
    estender o contexto (262k → ~1M) — por isso o `--hf-overrides` aninhado.
15. **KV cache.** Guardar K e V de tokens passados para não recomputar; cresce linearmente com o contexto. Só as 12 camadas QSA
    (+1 MTP) têm KV que cresce; o resto tem estado fixo.
16. **Prefill vs decode.** Prefill processa o prompt inteiro (compute-bound); decode gera token a token (memory-bound, lê todos os
    pesos por passo). Quase todos os patches deste repo miram o decode (bytes por passo).
17. **Lm_head, logits e argmax.** Projeção final hidden → vocabulário; greedy = argmax. `tie_word_embeddings=false` significa que o
    lm_head é uma matriz separada da embedding de entrada — e o drafter MTP tem mais uma.

### Nível 2 — Arquitetura específica do Qwen3.8-Flash-Next (Qwen4Exp)

18. **Arquitetura híbrida linear + atenção.** 3 de cada 4 camadas são "lineares" (estado de tamanho fixo), 1 em 4 é atenção
    (recuperação precisa). Consequência: KV barato (só 13 camadas) e prefill quase linear em contexto.
19. **Atenção linear e Gated DeltaNet (GDN).** Substitui softmax por um estado matricial S atualizado a cada token pela *regra delta*
    com decaimento: aproximadamente `S_t = α_t·S_{t-1}·(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, saída `o_t = S_t q_t`. α (decay) e β
    (taxa de escrita) vêm de `in_proj_a`/`in_proj_b` (+ `A_log`, `dt_bias`); por isso esses tensores pequenos `[48, 2560]` são
    sensíveis e ficam BF16. Estado por requisição: 48 cabeças × 128 × 128 (fp32 = 3 MiB/camada, 1,5 MiB/GPU em TP2).
    *(Equação é a formulação padrão do Gated DeltaNet; não está escrita no repo.)*
20. **Qwen Sparse Attention (QSA).** Em cada camada de atenção, um **indexador** leve (4 cabeças Q de 128 dims, 1 cabeça K
    compartilhada) resume cada micro-bloco de 4 tokens numa chave comprimida (média FP32, normalizada, com RoPE); o score é
    `s = (1/√128) Σ_h ReLU(⟨q_h, k̄_b⟩)`; mantém os 512 melhores blocos → 2048 posições + 0-3 do bloco incompleto = ≤2051; então
    faz GQA esparsa **sobre o K/V original** dessas posições. Pipeline de kernels na imagem: pre-indexer fundido (norm+RoPE+escrita de
    chaves) → `_qsa_mqa_paged_kernel` (scores) → `persistent_topk` → `_expand_qsa_indices_kernel` → `_qsa_sparse_paged_gqa_splitk_kernel`
    + `_qsa_merge_splitk_kernel` (fable §1). O FlashAttention é só classe-base nominal.
21. **Mixture of Experts (MoE) com shared expert.** Um roteador escolhe 10 de 512 experts (MLPs pequenos, intermediate 640) por
    token; um expert compartilhado (com gate sigmoid) sempre roda. Parâmetros totais 125B, ativos ~6B.
22. **Gated Residual / Hyper-Connections.** O residual vira 4 ramos de 2560; "Mix" gera gates element-wise por projeção low-rank
    (320) e reduz os 4 ramos a um hidden para o sub-bloco; "Combine" calcula 4 coeficientes de injeção e escreve de volta. No vLLM
    esses linears são **replicados** (não sharded) e tinham `quant_config=None` fixo — alvo do overlay FP8-dense.
23. **N-gram embedding / PLE.** Memória de padrões locais: 16 ids por token (8 cabeças de 2-grama + 8 de 3-grama), 160 valores cada
    → E ∈ R^2560 → keys por ramo (4×2560) e value compartilhado → gate sensível ao contexto (compara com o residual) → short-conv
    dilatada → injeta Δ nos 4 ramos. Só uma camada (índice 1). Requer, por requisição, os 2 últimos tokens (`ngram_context`) e o
    estado da conv. O drafter MTP não usa PLE (LMSYS).
24. **Multi-Token Prediction (MTP).** Um módulo extra treinado para prever tokens além do próximo, reaproveitado como drafter
    (estilo EAGLE: recebe hidden states do alvo). Aqui 1 camada (atenção cheia + MoE com experts FP8 bloco) reaplicada 3×; aceitação
    decai por posição (89→74,5→60%).
25. **Modelo multimodal (ViT + merger).** Torre de visão de 27 blocos cuja MLP tem intermediate 4304 — forma que quebra kernels
    MXFP8 e NVFP4 em TP2 → `MM_ENCODER_TP_MODE=data` e emulação BF16 do `visual.*`.

### Nível 3 — Quantização

26. **Quantização linear simétrica.** `q = round(x/s)`, `x ≈ q·s`; a escala `s` pode ser por tensor, por canal, por bloco ou por
    token. Erro ~ passo de quantização; medido aqui como RMSE relativo e cosseno.
27. **Granularidade das escalas.** Por tensor (1 escala: PLE FP8, KV FP8), por canal de saída (FP8-dense), por token dinâmica
    (ativações do FP8-dense), por bloco 1D de 16/32 (NVFP4/MXFP8), por bloco 2D 128×128 (experts da MTP). Mais fino = mais preciso e
    mais metadados; formas precisam ser divisíveis pelo bloco (por isso o FP8-dense escolheu por-canal).
28. **FP8 E4M3.** 1 bit de sinal, 4 de expoente, 3 de mantissa, máximo 448. Upcast e4m3→bf16 é exato (usado nos kernels QSA FP8).
29. **NVFP4.** Valores FP4 E2M1 (magnitudes {0, 0,5, 1, 1,5, 2, 3, 4, 6} + sinal) em blocos de 16 com escala FP8 E4M3 por bloco e
    uma escala FP32 global (dois níveis). Armazenado como 2 códigos por byte. Usado nos experts roteados (W4A4) e, no checkpoint
    local-inference-lab, na PLE (90 B/linha).
30. **MXFP8 (OCP Microscaling).** Elementos FP8 em blocos de 32 que compartilham uma escala potência-de-2 (E8M0). Kernels nativos
    exigem formas alinhadas; fallback = dequantizar para BF16 no load.
31. **FP8 block-wise 128×128 (`FP8_BLOCK_SCALES` / `weight_scale_inv`).** Uma escala por tile 128×128 de pesos, ativações
    quantizadas dinamicamente. Escala de bloco errada não quebra: aplica escalas desalinhadas silenciosamente.
32. **W4A4 vs W4A16 vs W8A8.** Qual precisão têm pesos (W) e ativações (A) no GEMM; W4A4 exige quantizar a ativação em runtime.
    O FP8-dense é W8A8 por canal/por token.
33. **Dequantização na busca vs no GEMM.** Para embeddings, dequantiza-se **só as linhas buscadas** (PLE); para linears, o kernel
    consome o formato quantizado ou (emulação) tudo é dequantizado no load.
34. **Quantização de KV cache.** Armazenar K/V em FP8 halve os bytes de KV; em atenção esparsa a quantização também pode mudar
    *quais* posições são escolhidas → trade de capacidade, não ganho grátis.
35. **Formatos de checkpoint e configs de quantização.** safetensors (header JSON + offsets), `model.safetensors.index.json`
    (`weight_map`), `quantization_config` ModelOpt (`quant_algo`, `MIXED_PRECISION`, `quantized_layers`, `config_groups`,
    `exclude_modules`), sidecar legado `hf_quant_config.json`. O loader do vLLM decide o método por **prefixo exato** de camada.

### Nível 4 — Inferência, vLLM e kernels

36. **Decode limitado por banda de memória (roofline).** tempo/passo ≈ bytes lidos / banda. Com ~9,2 GiB/passo a ~240 GB/s → ~40 ms;
    por isso cortar bytes (draft vocab, FP8-dense, FP8 KV em contexto longo) é a alavanca principal.
37. **Decodificação especulativa e verificação.** Um drafter propõe K tokens, o alvo verifica todos num único forward; tokens aceitos
    saem "de graça". Com rejection sampling a distribuição de saída é a do alvo, qualquer que seja o drafter → cortar o vocab do drafter
    só pode custar aceitação.
38. **FR-Spec (frequency-ranked speculative sampling).** Restringir o lm_head do drafter aos ids mais frequentes; o gargalo é a
    leitura do lm_head, não a qualidade do rascunho.
39. **IndexShare MTP.** Reutilizar a seleção top-k do indexador QSA calculada no draft-extend durante todos os passos de rascunho da
    iteração (a query andou ≤ N posições em L) → corta N-1 execuções do indexador.
40. **Paged KV cache e blocos.** O vLLM aloca KV em páginas fixas (aqui 1600 tokens por bloco de atenção); tabelas de blocos mapeiam
    posições lógicas → físicas; kernels QSA leem K/V paginados (`block_table`).
41. **Gerenciamento de cache híbrido (atenção + "mamba").** Camadas com estado constante (GDN, PLE short-conv) usam páginas de
    estado; o vLLM iguala tamanhos de página → bloco de atenção 1600. PLE/GDN se registram como `MambaBase`.
42. **Prefix caching.** Reuso de blocos de KV entre requisições com o mesmo prefixo (hash de blocos); no cache híbrido só casa em
    fronteiras de bloco (`align`). Drafters EAGLE descartam o último bloco casado — o backport #53388 desliga isso.
43. **Chunked prefill e `max_num_batched_tokens`.** O prompt é processado em pedaços (8192); o PLE precisa de `ngram_context`
    entre pedaços; buffers do PLE são dimensionados por `max_num_batched_tokens` e `max_num_seqs`.
44. **CUDA graphs (FULL_DECODE_ONLY) e padding.** Capturar a sequência de kernels do decode e reexecutá-la sem overhead de CPU;
    lotes são arredondados para tamanhos capturados, e linhas de padding usam `NULL_BLOCK_ID` (tratadas no PLE). Ops opacos precisam
    de *fake impl*.
45. **Custom ops e `torch.compile`.** `direct_register_custom_op` com `mutates_args` e fake impl permite que operações com efeito
    colateral (short-conv do PLE, wait do offload) passem pelo tracing.
46. **Triton: `tl.constexpr`, tiles, warps, split-K.** Parâmetros constexpr geram variantes compiladas (ramos mortos eliminados);
    `BLOCK_N` = largura do tile de posições, `num_warps` = paralelismo interno, `splits` = partições da redução ao longo das posições.
47. **Online softmax e merge por LSE (flash-decoding).** Cada partição mantém máximo, normalizador e acumulador; no fim, os parciais
    são combinados pelo log-sum-exp. Por isso escalas de V podem ser dobradas no normalizador sem quebrar o merge.
48. **Ocupação da GPU e "ondas".** Lançar muito mais CTAs do que SMs (48 no GB10) gera ondas extras e merges caros; tabelas afinadas
    para 160 SMs não servem para 48.
49. **Determinismo numérico.** Soma em ponto flutuante não é associativa; ordens de redução diferentes (top-k em ordem variável,
    finalize fundido do MoE) mudam bits e, em greedy, podem mudar a saída.
50. **Autotuning (FlashInfer).** Primeiro request escolhe configs de kernel e grava cache; caches diferentes entre nós podem
    dessincronizar coletivos (visto no v0.30).

### Nível 5 — Paralelismo e inferência distribuída

51. **Tensor Parallelism (TP).** Matrizes divididas entre GPUs; após certas camadas, all-reduce. Aqui TP=2 = 1 GPU por Spark,
    ≈106 coletivos pequenos por passo pela rede.
52. **Embedding / lm_head vocab-parallel.** Cada rank guarda uma faixa contígua de ids (`org_vocab_start_index..end`). Consequência:
    um draft vocab concentrado em ids baixos fica ~97% no rank 0, e o passo espera o rank mais lento.
53. **Redução distribuída de argmax.** Em vez de juntar logits completos (O(vocab)), cada rank manda (valor máximo, id) e escolhe-se
    o maior (O(2·tp)) — `use_local_argmax_reduction`.
54. **Expert Parallelism (EP).** Cada GPU guarda metade dos experts (256). Com DP=1 os tokens estão replicados em TP e o all2all nem é
    usado; EP muda só quais pesos cada GPU tem.
55. **Camadas replicadas vs sharded.** HC, roteador e indexador são replicados (cada GPU lê tudo) — por isso pesam no byte-model.
56. **Multi-nó com NCCL sobre RoCE.** `--nnodes 2`, `--master-addr`, HCAs e GID index; PyNCCL all-reduce apenas (sem NVLink).
57. **Multiprocess executor (`mp`) e processos auxiliares.** O executor `mp` é o que cria o processo de offload do PLE.

### Nível 6 — Sistema: memória unificada, IPC e operação

58. **Memória unificada (GB10).** CPU e GPU compartilham 121,69 GiB LPDDR5x; page cache, RAM do processo e alocações da GPU disputam o
    mesmo pool; esgotá-lo trava o kernel. Explica `evict_page_cache.py`, `memwatch.sh`, GMU 0.80 e a tabela mmap.
59. **Page cache, mmap e memória evictável.** Páginas de arquivo podem ser descartadas pelo kernel; memória anônima (RSS) não. Mapear a
    tabela PLE em arquivo torna 26,8 GiB evictáveis. `posix_fadvise(DONTNEED)` descarta páginas limpas sem root.
60. **CUDA stream memory ops.** `cuStreamWaitValue32`/`WriteValue32` fazem a GPU esperar/escrever num flag na memória, capturáveis em
    graph; ausentes no GB10 (`CAN_USE_STREAM_MEM_OPS=0`) → handshake no host com número de sequência.
61. **CUDA IPC e memória pinned.** Compartilhar buffers GPU entre processos; `cuMemHostRegister` para cópias H2D assíncronas.
62. **Bind-mount overlays e patches por âncora.** Substituir arquivos do pacote dentro do container sem rebuild; âncoras únicas +
    `ast.parse` tornam o patch seguro contra mudanças da imagem.
63. **Layout do cache Hugging Face.** `models--org--name/{refs/main, snapshots/<rev>, blobs/}`; snapshots são symlinks para blobs
    (por isso o hard-link do FP8-dense aponta para o blob); `refs/main` sem newline.
64. **NFSv4 para compartilhar pesos.** Export read-only do cache do head, montagem no worker; trade: sem cópia local, mas o head
    precisa ficar de pé e cada cold start transmite ~126 GiB.
65. **Abliteration (contexto do `ABLIT`).** Checkpoint derivado com uma projeção "house" em 12 `o_proj` da QSA para remover recusas;
    relevante aqui só porque seu `config.json` rotula os experts MTP como `FP8_PB_WO` (normalizado pelo `patch_checkpoint_config.py`).

---
## 6. Receita passo-a-passo (recriar do zero, com o raciocínio de cada decisão)

Foco nas partes cobertas por `files/`; o que é de `start.sh`/rede está resumido só o necessário para a sequência fazer sentido.

### Passo 1 — Entender o hardware e fazer o orçamento de bytes

- 2 nós GB10, 121,69 GiB **unificados** cada, ~273 GB/s, 48 SMs, sm_121, ligados por ConnectX-7 (RoCE). Decisão: TP=2 (1 GPU/nó)
  + EP para os experts.
- Faça a conta do checkpoint por componente (fable §2): experts NVFP4 ~63 GiB, PLE FP8 ~48 GiB, densos BF16 ~10 GiB, MTP, visão.
  Com TP2 + tabela PLE vocab-sharded cabem ~64,5 GiB/GPU de pesos, sobrando ~32 GiB para KV (GMU 0.835) ou ~28 GiB (GMU 0.80).
- Razão: em memória unificada **nada é "grátis"**: offload para CPU não libera memória física; page cache compete com o modelo.

### Passo 2 — Imagem e método de patch

- Use a imagem day-0 (`vllm/vllm-openai:qwen38-flash-next`) sem rebuild. Para cada arquivo a mudar: `docker cp` do original →
  `.orig` → script de patch por âncoras (uma ocorrência, `ast.parse`) → bind-mount `:ro` nos dois nós.
- Razão: reprodutível, auditável (o diff é o script), e falha alto se a imagem mudar.

### Passo 3 — Pesos: escolher o checkpoint e garantir integridade

- Escolha `nvidia/Qwen3.8-Flash-Next-NVFP4` (BF16 densos, NVFP4 experts, PLE FP8 por tensor, experts MTP FP8 128×128).
- `resolve_snapshot.py`: só considere pronto um snapshot cujos shards do `weight_map` existem todos.
- Distribua para o worker (rsync) ou compartilhe por NFS (`nfs-share.sh`); antes de lançar, `evict_page_cache.py` nos dois nós.
- Razão: shard truncado passa em checagem de tamanho e só explode a 33% do load; page cache residente causa CUDA OOM no GB10.

### Passo 4 — Consertar a leitura do checkpoint (sem isso nada carrega)

1. **PLE:** aplique `patch_ple_layer.py` (dispatch por `ple_embedding_dtype`, métodos FP8/NVFP4 que mantêm a tabela quantizada e
   dequantizam só as 16 linhas buscadas). Rode `detect_ple_dtype.py` e injete `{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}`
   via `--hf-overrides` (aninhado!). Razão: senão a tabela vira BF16 (~102 GB) e o load morre.
2. **MTP:** `patch_checkpoint_config.py` adiciona `mtp.layers.48.*` em `config.json` e `hf_quant_config.json` (e normaliza
   `FP8_PB_WO`→`FP8_BLOCK_SCALES`); monte os JSONs por cima do snapshot. Rode `--mtp-moe-algo` como preflight.
3. **Experts MTP FP8 bloco:** `patch_modelopt_fp8_block_moe.py` para construir `Fp8MoEMethod` com `weight_block_size` do checkpoint.
4. **MXFP8 (se o checkpoint tiver camadas MXFP8):** `patch_modelopt_mxfp8.py` (fallback BF16 por forma + `visual.*`).
- Verificação: log sem "unexpected/missing keys"; `Available KV cache memory` razoável; aceitação MTP com curva decrescente por
  posição (curva plana/aleatória = escalas desalinhadas).

### Passo 5 — Parâmetros de serving base

- `--tensor-parallel-size 2 --nnodes 2 --enable-expert-parallel --distributed-executor-backend mp`, `--max-num-seqs 8`,
  `--max-num-batched-tokens 8192`, `--max-model-len 262144` (YaRN só se precisar de >262k, e sempre aninhado em `text_config`),
  `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`, `--mm-encoder-tp-mode data`,
  `--gpu-memory-utilization 0.80` (0.835 deixou <1 GiB `MemAvailable` sob carga), `--mamba-ssm-cache-dtype bfloat16` (opcional).
- Razão: TP2 porque cada nó tem 1 GPU; EP obrigatório para NVFP4; `data` para a visão porque 4304/2 não é múltiplo de 16.

### Passo 6 — Decodificação especulativa MTP

- `--speculative-config '{"method":"mtp","num_speculative_tokens":3, ...}'`. Meça aceitação por posição em `/metrics`
  (`spec_decode_num_accepted_tokens_per_pos_total`) sempre como **delta** (`bench/mtp_accept.py`).
- Razão: 24,5 → 52,1 tok/s. K=3 porque a 3ª posição ainda aceita ~60%; se cair abaixo de ~35%, K=2 é melhor (fable §3.2).

### Passo 7 — Cortar bytes do drafter (maior alavanca isolada do passo)

1. Gere corpus da **saída do próprio modelo** (`bench/gen_draft_corpus.py`) no seu tipo de tráfego.
2. `build_draft_vocab.py corpus.jsonl --size N [--fill-to-size --balance-shards 2]` — escolha N pela **cobertura** impressa.
3. Aplique `patch_mtp_draft_vocab.py` e ligue `use_local_argmax_reduction: true` no speculative config; monte o arquivo em
   `/etc/vllm-draft-vocab.txt` com `VLLM_MTP_DRAFT_VOCAB`.
4. Rode `test_draft_vocab.py` na imagem após qualquer mudança no patch.
- Razão: o drafter lê um lm_head de 0,59 GiB/GPU por passo de rascunho (3 de 4 leituras de lm_head). Ganho medido +8,4 a +9,6% com
  aceitação inalterada. Cuidado com o desbalanceamento entre ranks (o passo espera o mais lento) e com vocabulários muito
  "acolchoados" (65k com só 8,3k ids reais derrubou a aceitação para 47,8%).

### Passo 8 — Capacidade de KV: FP8

- `--kv-cache-dtype fp8` + `patch_qsa_fp8_kv.py` (day-0) ou o backport `patch_qsa_fp8_kv_v030.py` (vLLM 0.30).
- Razão: ×1,70-1,80 tokens de cache (necessário para 1M de contexto ou muitas streams longas). Trade: −5% throughput médio e risco
  de qualidade em raciocínio longo (a atenção esparsa pode selecionar blocos diferentes). Valide com `bench/reasoning_check.py`
  (tarefas determinísticas + needles) no seu workload. Se o objetivo é throughput em contexto curto, BF16 pode ser melhor.

### Passo 9 — Otimizações do especulativo e do cache

- `index_share_for_mtp_iteration: true` (IndexShare): corta o indexador nos passos de rascunho; code até +26,6% em S=8.
- `disable_eagle_block_drop: true` + `patch_block_drop.py` (se a imagem não tiver a chave): multi-turno volta a acertar o cache
  (TTFT 3,25 → 0,63 s). Monte só os 3 arquivos core.
- Razão: ambas mexem só no drafter/cache; a saída verificada não muda.

### Passo 10 — Tuning opcional

- **Perfis QSA para 48 SMs:** rode `bench_qsa_kernels.py` na imagem com o overlay `qsa_gb10` e use o JSON (`QSA_PROFILE=<json>`).
  Só com KV bf16 (ambos sobrescrevem `ops/qsa.py`). O perfil `gb10` embutido é palpite não medido.
- **Determinismo:** `VLLM_QSA_DET_TOPK=1` + `VLLM_MOE_DET_FINALIZE=1` quando precisar de A/B bit-idênticos (−4%).
- **FP8-dense (experimental, não medido):** `fp8dense/build.sh` → verificar (`verify_fp8_dense_checkpoint.py`,
  `test_quant_config_resolution.py`) → `FP8_DENSE=true` com overlay; esperado ~1,5× no batch 1. Resolva antes: `apply_patches.py`
  ausente, conflito de mount do `modelopt.py`, conflito com o draft vocab (ambos em `mtp.py`). Confirme no log
  `Selected CutlassFP8ScaledMMLinearKernel` e o aumento de ~3,5 GiB/GPU de KV.
- **Single-Spark (TP1):** exige offload do PLE (`patch_ple_offload.py` + `ple_offload/**` + `build_ple_packed_table.py` com
  `VLLM_PLE_PACKED_TABLE_DIR`), `--distributed-executor-backend mp`, `memwatch.sh` e checkpoint menor (local-inference-lab).

### Passo 11 — Verificação e medição

- Logs: `Available KV cache memory`, `GPU KV cache size`, linhas do draft vocab, `MXFP8 layer ... falling back`, `Routed experts ...
  use FP8_BLOCK_SCALES`, `PLE embedding ... uses the runtime FP8 method`.
- `/metrics`: aceitação por posição, `kv_cache_usage`, preempções; `bench/sweep.py` (S=1/2/4/8, 3 repetições alternadas, uma
  launch por braço). Trate o prompt "code" do sparkDash como sintético (aceitação ~1,0).
- Regra do repo: re-medir após mudar `MAX_MODEL_LEN`, `MTP_NUM_SPECULATIVE_TOKENS`, `KV_CACHE_DTYPE`, `MTP_DRAFT_VOCAB`,
  `FP8_DENSE` ou `QSA_PROFILE`.

---

## 7. Observações, lacunas e inconsistências encontradas

1. ⚠ `PLE_QUANT_OVERRIDE=fp8` aparece no README (linhas 460 e 557) e no fable §7.4, mas **nenhum código atual** lê essa variável;
   o mecanismo vigente é `text_config.ple_embedding_dtype` (patch + `detect_ple_dtype.py`).
2. ⚠ `files/overlay/apply_patches.py` é chamado por `start.sh:579`, mas não está versionado (o padrão `files/overlay/*.py` do
   `.gitignore` o ignora). Só os `.diff` existem.
3. ⚠ Com `FP8_DENSE=true`, dois mounts no mesmo destino `modelopt.py` (`start.sh:580` e `:866`) *(inferido: provável erro do Docker)*;
   e o `overlay/modelopt.py` seria gerado do original, sem os ramos MXFP8/FP8_BLOCK_SCALES.
4. ⚠ `PLE_OFFLOAD=true` no `start.sh` dual não monta os patches de offload do GB10 (só o TP1 monta).
5. ⚠ `patch_modelopt_mxfp8.py` atribui as formas MXFP8 ao RadixArk; o HANDOFF diz que RadixArk é BF16 na atenção e que MXFP8 é do
   local-inference-lab. No checkpoint default (nvidia, densos BF16) o patch é provavelmente inerte *(inferido)*.
6. ⚠ `docs/HANDOFF-single-spark.md` ainda diz "FP8 KV cache is NOT supported" — superado pelo `patch_qsa_fp8_kv.py`.
7. ⚠ A tabela `_select_sm120_config` do backport v0.30 não se aplica ao GB10 (sm_121 ≠ sm_120) *(inferido do código)*.
8. `ple_layer_ids=[2]` no config vs módulo em `layers.1` no vLLM: é indexação base-1 vs base-0 (LMSYS confirma).
9. `patch_block_drop.py` gera 6 arquivos, `start.sh` monta 3 — intencional (sem KV connector).
10. O perfil `QSA_PROFILE=gb10` e o FP8-dense nunca foram medidos em GPU (GPU ocupada no dia; fable §8).
11. O nome "PLE" tem duas expansões nas fontes ("position-learning enhancement" no docstring da imagem; "Per-Layer Embeddings" no
    LMSYS). Não confundir com o PLE do Gemma 3n (embeddings por camada), que é um mecanismo diferente *(associação minha)*.
12. `html/**` não tem conteúdo técnico sobre o modelo; é material de demonstração/teste de geração de páginas.

---

## Addendum 2026-10-30

Este inventário descreve o clone de 2026-09-28. O que mudou depois está verificado em
[`mia-updates-2026-10-30.md`](mia-updates-2026-10-30.md); para o escopo de `files/**`, a única mudança real é o
commit `288369d` (2026-09-29) e a documentação que o mediu (até `cd839d0`):

- **`files/draft_vocab_{es,zh,ja,de,pt,fr,ru}_en_code_65k.txt`** (65.536 ids cada): o `draft_vocab_en_code_47k.txt`
  inteiro como piso, os 400 ids de byte-fallback fixados e 18.387 ids por frequência em 668 MiB da Wikipédia do
  idioma. Opt-in via `MTP_DRAFT_VOCAB=files/draft_vocab_<lang>_en_code_65k.txt`; o default segue o 47k. Aceitação
  por rascunho 47k → 65k (TP=2, um boot por braço, temperatura 0, thinking off; alemão em ABBA): ru 0,39 → 1,55,
  ja 0,45 → 1,34, zh 0,45 → 1,14, de 0,81 → 1,40, es 1,35 → 1,88, fr 0,92 → 1,34, pt 0,96 → 1,39; controle em
  inglês 1,98–2,15. O shard de rascunho do rank mais lento cresce de ~0,22 para 0,23–0,29 GiB por passo.
- **`files/build_draft_vocab_extend.py`**: porte do método do recipe single-Spark (piso inteiro, byte-fallback
  sempre fixado, ids só por frequência em texto real), com relatório da distribuição de ids por shard de TP
  (`--tp-report`).

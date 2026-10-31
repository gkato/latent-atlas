# Como criar uma receita do zero para MiMo-V2.6-Flash-MOPD em 2× DGX Spark (vLLM)

> Material didático no padrão das receitas do Mia's AI Lab (GLM-5.3-Flash EXL3, Qwen3.8-Flash-Next NVFP4,
> DeepSeek-V4-Flash DSpark). Pesquisa feita em **2026-09-28**. Todas as afirmações sobre o modelo, o vLLM e
> as receitas da comunidade têm link na seção [Fontes](#fontes).
>
> **Como ler os números.** Seguindo a regra das receitas do laboratório ("every number is a measurement, not
> an estimate"), cada número vem marcado:
> - **[medido: X]** foi lido de um log ou benchmark publicado por X, no hardware indicado;
> - **[estimativa]** é conta feita aqui a partir do `config.json` e dos cabeçalhos safetensors. Serve para
>   planejar e deve ser substituída pelo valor medido no seu kit;
> - **[a medir]** ninguém publicou ainda: é trabalho seu.

---

## Sumário

- [Parte I — Qual modelo e por quê](#parte-i--qual-modelo-e-por-quê)
  - [1. Candidatos de agosto e setembro de 2026](#1-candidatos-de-agosto-e-setembro-de-2026)
  - [2. A escolha: MiMo-V2.6-Flash-MOPD](#2-a-escolha-mimo-v26-flash-mopd)
  - [3. Ficha técnica](#3-ficha-técnica)
  - [4. `config.json` campo a campo](#4-configjson-campo-a-campo)
  - [5. Tokenizer e chat template](#5-tokenizer-e-chat-template)
  - [6. Variantes quantizadas publicadas](#6-variantes-quantizadas-publicadas)
  - [7. Status no vLLM (versões, PRs, issues)](#7-status-no-vllm-versões-prs-issues)
  - [8. Opções de decodificação especulativa](#8-opções-de-decodificação-especulativa)
  - [9. Licença](#9-licença)
- [Parte II — O passo a passo](#parte-ii--o-passo-a-passo)
  - [Passo 0 — Triagem](#passo-0--triagem)
  - [Passo 1 — Matemática de memória](#passo-1--matemática-de-memória)
  - [Passo 2 — Quantização e checkpoint](#passo-2--quantização-e-checkpoint)
  - [Passo 3 — Topologia (TP/EP/PP)](#passo-3--topologia-tpeppp)
  - [Passo 4 — Suporte no motor](#passo-4--suporte-no-motor)
  - [Passo 5 — Argumentos de lançamento](#passo-5--argumentos-de-lançamento)
  - [Passo 6 — Decodificação especulativa](#passo-6--decodificação-especulativa)
  - [Passo 7 — Validação](#passo-7--validação)
  - [Passo 8 — Benchmark](#passo-8--benchmark)
  - [Passo 9 — Tuning iterativo](#passo-9--tuning-iterativo)
  - [Passo 10 — Empacotamento](#passo-10--empacotamento)
- [Apêndice A — Glossário de conceitos](#apêndice-a--glossário-de-conceitos)
- [Apêndice B — Desafio: Naive-N0.5-Flash, um modelo que o vLLM ainda não conhece](#apêndice-b--desafio-naive-n05-flash-um-modelo-que-o-vllm-ainda-não-conhece)
- [Apêndice C — Rodar o mesmo modelo no Mac com MLX e contrato OpenAI](#apêndice-c--rodar-o-mesmo-modelo-no-mac-com-mlx-e-contrato-openai)
- [Fontes](#fontes)

---

# Parte I — Qual modelo e por quê

## 1. Candidatos de agosto e setembro de 2026

Critérios pedidos: pesos abertos; lançado entre agosto e setembro de 2026; de preferência MoE entre 100B e
600B de parâmetros totais, para exigir quantização e dois nós; pontos extras para arquitetura nova,
MTP/speculator próprio ou suporte incompleto no vLLM.

O hardware alvo é 2× DGX Spark: GB10, sm_121, 128 GB LPDDR5X unificada por nó (~273 GB/s), dos quais o
CUDA enxerga **121,69 GiB** [medido: README Qwen3.8 do laboratório], com ConnectX-7 RoCE de 200 Gb/s entre
os nós. **Na prática, cabem ~160–170 GiB de pesos no par.**

| Modelo | Lançamento | Total / ativos | Novidade arquitetural | Pesos oficiais | Cabe em 2× Spark? | vLLM (2026-09-28) | Licença | Veredito |
|---|---|---|---|---|---|---|---|---|
| **XiaomiMiMo/MiMo-V2.6-Flash-MOPD** (e o irmão `-RL`) | MOPD 2026-09-27; RL 2026-09-21 | 309B / 15B | SWA-128 híbrido 39:9, K/V com dimensões diferentes (192/128, "DiffKV"), attention sinks, experts **MXFP4 nativos**, 2 speculators embutidos (MTP-3 e DFlash-5), omni (texto/imagem/vídeo/áudio), 1M de contexto | 172,9 GB de pesos (165,6 GiB no repo) | **Sim, no limite**: ~82,7 GiB por rank [medido: MiaAI-Lab SGLang] | Day-0 só em imagem dedicada ou nightly; **não entrou no v0.30.0**; mais de 10 PRs abertos | MIT | **Escolhido** |
| NaiveAI/Naive-N0.5-Flash | 2026-09-27 | 309B / 15,5B | SWA + DeepSeek Sparse Attention, **sem nenhuma camada de atenção global**; construído sobre a base do MiMo-V2.5 | BF16 617,8 GB; FP8 oficial (~310 GB) e um `FP8-Draft` | Só em 4 bits (há NVFP4 e EXL3 da comunidade com 1 dia de vida) | **Sem suporte** (`naive_n05_flash`, remote code) | MIT | Desafio do [Apêndice B](#apêndice-b--desafio-naive-n05-flash-um-modelo-que-o-vllm-ainda-não-conhece) |
| IQuestLab/IQuest-Q1 | 2026-09-28 | 320B / 15B | 88 camadas, 256 experts top-8, 512K de contexto, pasta `mtp/` | BF16 650 GB | Só quantizando você mesmo | Sem suporte (`iquest_q1`) | Própria ("other") | Cru demais: zero quants e zero relatos |
| deepseek-ai/DeepSeek-V4.1-Flash | 2026-09-10 | 552B de backbone + 196B de Engram; 8B ativos no prefill / 16B no decode | Causal Encoder-Decoder, CSA2, KV em FP4 (890 B/token), DSpark | 510 GB | Não (precisa de 3–4 Sparks ou ≤2 bits) | Suportado no v0.30.0 (#56214 e outros) | MIT | O laboratório já cobre a família DSV4 |
| zai-org/GLM-5.3 | 2026-08-25 | 753B / 40B | `GlmMoeDsaForCausalLM` | 756 GB em FP8 | Não | — | "other" | Grande demais; o GLM-5.3-Flash já tem receita |
| TypeSafeAI/Step-5-Preview-BF16 | 2026-09-20 | 604B | Multimodal com robótica (`step3p5v`) | 1,21 TB em BF16 | Não | — | "other" | Grande demais |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | 2026-09-21 | 1,02T / 42B | Mesma família do Flash | 573,5 GB | Não (há kits com 8 Sparks) | Mesma situação do Flash | MIT | Grande demais para 2 nós |

O Qwen3.8-Flash-Next, o GLM-5.3-Flash e o DeepSeek-V4-Flash ficaram fora da lista porque já têm receita no
laboratório.

## 2. A escolha: MiMo-V2.6-Flash-MOPD

**Por que é um bom caso de ensino.**

1. **Cabe apertado.** São 161 GiB de pesos para 2 × 121,7 GiB. Sobram ~15–20 GiB de KV por rank, e isso obriga
   a fazer a conta do [Passo 1](#passo-1--matemática-de-memória) direito. Um Spark sozinho não roda esse
   modelo em formato nativo.
2. **Quebra várias suposições do motor ao mesmo tempo.** K e V têm dimensões diferentes (192 contra 128), então o
   modelo precisa de backend DiffKV, e no GB10 isso significa Triton. O número de KV heads muda com o tipo de
   camada (8 nas SWA, 4 nas globais). Há sinks só nas camadas SWA. O QKV fundido em FP8 vem **pré-fatiado para
   TP=4** e você vai rodar TP=2. Os experts vêm em MXFP4 com escala E8M0. E há dois speculators diferentes no
   mesmo repositório.
3. **O suporte é "day-0" mas incompleto, e com bugs silenciosos.** Três exemplos no vLLM, todos com PR aberto:
   `--kv-cache-dtype fp8` é ignorado sem aviso; o QKV fica zerado se você usar `fastsafetensors`, `eager` ou
   InstantTensor; e o verify do speculative decoding cai num kernel lento em contexto longo. É o cenário exato
   em que as receitas do laboratório usam *overlays* (Qwen: patches de PLE e MXFP8; GLM: `overlay/`).
4. **Por que MOPD e não RL.** O `config.json` é **idêntico byte a byte** e o drafter é o mesmo (verificado neste
   estudo). O MOPD foi feito para mitigar a *repetição de tool calls* ("tool-call storms"), que é o defeito que
   as receitas comunitárias do RL mais relatam em uso com agentes.
5. **Tem gabarito público para cada passo.** Existem kits para 2× Spark:
   [MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks)
   (**SGLang**, checkpoint RL),
   [Smilehyu/MiMo-V2.6-Flash-MOPD-2x-DGX-Sparks](https://github.com/Smilehyu/MiMo-V2.6-Flash-MOPD-2x-DGX-Sparks)
   (SGLang, MOPD),
   [tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe)
   (vLLM, RL) e
   [Plaaasma/mimo26-flash-dual-dgx-spark](https://github.com/Plaaasma/mimo26-flash-dual-dgx-spark)
   (vLLM customizado). Para SM120, o mais próximo do sm_121, há o
   [diffbot W4A8 2× RTX PRO 6000](https://huggingface.co/diffbot/MiMo-V2.6-Flash-MOPD-FP8KV-W4A8-2x-RTX-PRO-6000).

**O que esta receita traz de novo:** uma **lane vLLM, no padrão do laboratório** (`.env`/`start.sh`/overlays
fail-closed/benchmarks medidos), para o checkpoint **MOPD**. A receita MiMo-V2.6 que o laboratório já tem usa
SGLang e o checkpoint RL. O tutorial manda você construir tudo do zero e usar os kits acima só para conferir a
resposta, nunca para copiar.

## 3. Ficha técnica

Fonte: [card do MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD),
[card do RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), `config.json` e cabeçalhos safetensors
(lidos neste estudo com range requests).

| Item | Valor |
|---|---|
| Repositório / revisão fixada | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `2479e2d0029eca9a34cc7e7f55a121925f81908e` (lastModified 2026-09-27). RL: `5711b268169967567844e1e560e8a3966da959b1` |
| Parâmetros | 309B no total / 15B ativos (card); o HF conta 310.756.322.688 |
| Backbone | 48 camadas: 9 globais (GA) nos índices `[0, 5, 11, 17, 23, 29, 35, 41, 47]` e 39 SWA com janela 128. A camada 0 é GA com FFN densa; as outras 47 são MoE |
| Atenção | Q=64 heads nas duas; KV = 8 (SWA) / 4 (GA); `head_dim` QK=192, V=128; RoPE parcial (fator 0,334 ≈ 64 de 192 dims); theta 10M (GA) e 10k (SWA); `attention_value_scale` 0,707 aplicado em V |
| MoE | 256 experts roteados, top-8, **sem expert compartilhado**, `moe_intermediate_size` 2048, gate sigmoid com `noaux_tc` (bias `e_score_correction_bias` em F32), `norm_topk_prob`, router em BF16 |
| Precisão armazenada | Experts em **MXFP4** (nibbles E2M1 em U8 + escala E8M0 a cada 32 valores): 149,8 GiB no total. `qkv_proj` e a MLP densa em **FP8 E4M3** com blocos 128×128 (`weight_scale_inv` F32). `o_proj`, embeddings, `lm_head`, gates e encoders em **BF16** (`ignored_layers` = os 48 `o_proj` + o do MTP) |
| Layout no disco | Índice com `{"save_format":"mxfp4","total_size":172923364096,"tp_size":4}`. O MOPD tem 128 shards de experts (129 arquivos de pesos); o RL tem 64 |
| Contexto | 1.048.576 (`max_position_embeddings`) |
| Multimodal | ViT de 681M (28 blocos: 24 SWA + 4 globais, com sinks); AudioTokenizer de 308M (pasta `audio_tokenizer/`, 1,87 GB) + patch encoder de áudio de 127M; vídeo com timestamps |
| Speculators | (a) **MTP de 3 camadas** (`model.mtp.layers.0-2`, `model_mtp.safetensors`, 1,19 GB). (b) **DFlash de 5 camadas SWA** (`dflash/`, 2,94 GB BF16, `block_size` 8, até 7 tokens de rascunho por passo, atenção bidirecional dentro do bloco) |
| Amostragem recomendada | `temperature=1.0`, `top_p=0.95` |
| Benchmarks do card (Flash) | DeepSWE v1.1 67,9; Terminal Bench 2.1 87,6; Toolathlon-Verified 73,6; OSWorld-Verified 80,8 |

**Novidades arquiteturais que viram trabalho de engenharia** (é isto que o Passo 0 deve encontrar):

| Novidade | Por que importa no motor |
|---|---|
| K/V com dimensões diferentes (192/128) | Os kernels FA2/FlashInfer assumem `head_size_v == head_size`. O vLLM tem `FLASH_ATTN_DIFFKV` (precisa de FA3/FA4, que não existe no sm_121) e `TRITON_ATTN_DIFFKV`. **No GB10, toda a atenção do alvo roda em Triton DiffKV** |
| Sinks só nas SWA | O backend precisa aceitar `sinks=` por camada |
| SWA com janela 128 e mais KV heads que a GA | O custo de KV por token é dominado pelas 9 camadas globais, mas cada requisição paga um custo fixo da janela. O alocador híbrido do vLLM agrupa camadas e pode desperdiçar memória com "enchimento" ([Passo 1](#passo-1--matemática-de-memória)) |
| `sliding_window: 128` no nível do modelo | Se esse valor vazar para as camadas globais pelo `cache_config`, elas passam a enxergar só 128 tokens (#58128/#58176). **Bug silencioso de qualidade** |
| QKV fundido e pré-fatiado para TP=4 | Com TP=2, cada rank junta 2 chunks. Nas camadas SWA (8 KV heads em 4 chunks) a suposição "1 chunk por KV head" quebra (#57508). O caminho de requantização muda 18% dos pesos de Q, ~51% dos de K e 73–81% dos de V nas camadas globais [medido: diffbot, TP2] |
| Experts MXFP4 | É preciso um kernel MoE MXFP4 que funcione no sm_121 (Marlin W4A16 ou W4A8). Motores antigos expandem para FP8 e derrubam o nó [medido: MiaAI-Lab, SGLang] |
| Dois speculators | Tamanhos de KV e de pesos diferentes; o DFlash usa hidden states de 5 camadas do alvo (`target_layer_ids [0,11,23,35,47]`) |

## 4. `config.json` campo a campo

`config.json` do MOPD, idêntico ao do RL: 66 chaves no nível de topo. "vLLM lê?" indica se
`vllm/model_executor/models/mimo_v2.py` (main de 2026-09-28) referencia o campo; os campos de MTP e
multimodal são lidos em `mimo_v2_mtp.py` e `mimo_v2_omni.py`.

### 4.1 Nível de topo

| Campo | Valor | Significado e consequência para a receita |
|---|---|---|
| `architectures` | `["MiMoV2ForCausalLM"]` | Classe procurada no registry. No vLLM, `MiMoV2ForCausalLM` → `mimo_v2.py`. Com `vision_config` presente, o checkpoint resolve para `MiMoV2OmniForCausalLM` (`mimo_v2_omni.py`) [medido: tonyd2wild] |
| `model_type` | `mimo_v2` | Chave do `AutoConfig` (remote code `configuration_mimo_v2.MiMoV2Config`) |
| `auto_map` | `configuration_mimo_v2.*`, `modeling_mimo_v2.*` | Exige `--trust-remote-code`. O HF usa `modeling_mimo_v2.py` (85 KB), que é a referência numérica |
| `transformers_version` | `5.3.0` | Versão do transformers usada para salvar; use ≥ essa na referência HF |
| `dtype` | `bfloat16` | Dtype das ativações e dos tensores não quantizados |
| `hidden_size` | 4096 | Largura do residual. Em TP=2, cada all-reduce de decode move 4096 × 2 B = 8 KB por token |
| `num_hidden_layers` | 48 | Camadas do backbone. Truncar com `--hf-overrides '{"num_hidden_layers":N}'` é a técnica de paridade do [Passo 4](#passo-4--suporte-no-motor) (#58184) |
| `hybrid_layer_pattern` | lista de 48; `0` = global, `1` = SWA | 9 zeros, nos índices 0,5,11,17,23,29,35,41,47. No vLLM, `is_compressed_softmax_layer()` retorna `pattern[i] == 1` |
| `hybrid_block_size` | `null` | Usado só no HF/config remoto; o vLLM ignora |
| `moe_layer_freq` | lista de 48: `[0, 1, 1, …]` | A camada 0 é densa; as outras 47 são MoE |
| `intermediate_size` | 16384 | FFN densa, só na camada 0 (e nas MLPs do MTP) |
| `moe_intermediate_size` | 2048 | Largura de cada expert. Com TP=2 fica 1024 por rank, múltiplo de 32 (bloco MXFP4) |
| `n_routed_experts` | 256 | Com EP=2, 128 por rank |
| `num_experts_per_tok` | 8 | Top-8, que define os bytes lidos por token no decode ([Passo 1](#passo-1--matemática-de-memória)) |
| `n_shared_experts` | `null` | Sem expert compartilhado. Todo o custo do MoE é roteado, o que deixa o balanceamento com EP mais sensível |
| `n_group`, `topk_group` | 1, 1 | Roteamento em grupo desligado (grupo único) |
| `topk_method` | `noaux_tc` | Seleção top-k com bias de correção sem perda auxiliar (estilo DeepSeek-V3); usa `e_score_correction_bias` |
| `scoring_func` | `sigmoid` | Os scores do gate passam por sigmoid, não softmax |
| `norm_topk_prob` | `true` | Os pesos dos 8 experts escolhidos são renormalizados para somar 1 |
| `routed_scaling_factor` | `null` | 1,0 (sem escala extra na saída do MoE) |
| `moe_router_dtype` | `bfloat16` | Dtype do gate. O vLLM usa `float32` quando o campo falta. Por isso os comandos antigos do MiMo-V2.5 passavam `--hf-overrides '{"moe_router_dtype":"bfloat16"}'`; **no V2.6 isso já vem no config** |
| `num_attention_heads` | 64 | Q heads das camadas globais. TP=2 → 32 por rank |
| `num_key_value_heads` | 4 | KV heads das globais. TP=2 → 2 por rank; TP=4 → 1; TP=8 exige replicação. **Também é o `ckpt_tp` do QKV pré-fatiado** (4 chunks) |
| `head_dim` | 192 | Dimensão de Q/K nas globais |
| `v_head_dim` | 128 | Dimensão de V nas globais. Diferente de `head_dim`, daí o backend DiffKV |
| `swa_num_attention_heads` | 64 | Q heads nas SWA |
| `swa_num_key_value_heads` | 8 | KV heads nas SWA: o dobro das globais. TP=2 → 4 por rank |
| `swa_head_dim` / `swa_v_head_dim` | 192 / 128 | Mesmas dimensões DiffKV nas SWA |
| `sliding_window` / `sliding_window_size` | 128 / 128 | Janela das camadas SWA. O vLLM usa `sliding_window_size` por camada, mas o `sliding_window` global contamina o `cache_config` compartilhado (ver #58128) |
| `attention_chunk_size` | 128 | Não é lido pelo vLLM nem pelo `modeling_mimo_v2.py`; parece ser herança de configuração. Documente que ele é ignorado |
| `attention_projection_layout` | `fused_qkv` | O checkpoint guarda `qkv_proj` fundido, não `q_proj`/`k_proj`/`v_proj` separados. Só o HF lê esse campo; o vLLM sabe disso pelo nome dos tensores |
| `attention_bias` | `false` | Projeções sem bias |
| `attention_dropout` | 0.0 | Irrelevante para inferência |
| `attention_value_scale` | 0.707 | V é multiplicado por 0,707 **antes** da atenção (no vLLM: `v = v * self.v_scale`). O drafter DFlash tem o seu próprio (0,612, no `dflash/config.json`) |
| `add_swa_attention_sink_bias` | `true` | As camadas SWA têm `attention_sink_bias` (64 valores BF16): um logit extra por head no softmax |
| `add_full_attention_sink_bias` | `false` | As globais não têm sink |
| `partial_rotary_factor` | 0.334 | RoPE em ~1/3 das dims de Q/K (64 de 192). O resto é "NoPE" |
| `rope_theta` | 1.0e7 | Base de RoPE das globais (contexto longo) |
| `swa_rope_theta` | 10000 | Base de RoPE das SWA (janela curta) |
| `rope_parameters` | `{"rope_type":"default","rope_theta":1e7,"partial_rotary_factor":0.334}` | Forma nova (transformers 5) dos mesmos parâmetros; sem YaRN. **Não é preciso `--hf-overrides` de rope para chegar a 1M** |
| `max_position_embeddings` | 1048576 | Contexto nativo de 1M. Acima de 262144 alguns builds pedem `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (o laboratório usa isso no Qwen) |
| `layernorm_epsilon` | 1e-6 | Epsilon do RMSNorm |
| `hidden_act` | `silu` | SwiGLU. O vLLM rejeita outra ativação no MoE |
| `initializer_range` | 0.02 | Só para treino |
| `use_cache` | `true` | Só para HF |
| `tie_word_embeddings` | `false` | `lm_head` separado (1,25 GB BF16), que entra na leitura de todo passo de decode |
| `vocab_size` | 152576 | Vocabulário com padding (o último token real do tokenizer é 151674; o DFlash usa `mask_token_id` 151675) |
| `bos_token_id` | `null` | Sem BOS, o que é normal no ChatML |
| `eos_token_id` | 151645 (`<\|im_end\|>`) | O `generation_config.json` amplia para `[151643, 151645, 151672]` |
| `pad_token_id` | 151643 (`<\|endoftext\|>`) | — |
| `num_nextn_predict_layers` | 3 | Número de camadas MTP (`model.mtp.layers.0-2`). Limita `num_speculative_tokens` do método `mtp` a ≤3 |
| `quantization_config` | ver 4.2 | — |
| `image_token_id` / `video_token_id` / `audio_token_id` | 151655 / 151656 / 151669 | Placeholders que o processor expande |
| `vision_start_token_id` / `vision_end_token_id` | 151652 / 151653 | Delimitadores de imagem e vídeo |
| `audio_start_token_id` / `audio_end_token_id` | 151673 / 151674 | Delimitadores de áudio |
| `vision_model_type` | `mimovl` | Seleciona o ViT MiMo |
| `vision_config` / `audio_config` / `processor_config` | ver 4.3 | — |

### 4.2 `quantization_config`

| Campo | Valor | Significado |
|---|---|---|
| `quant_method` | `fp8` | O vLLM trata o checkpoint como FP8 com blocos |
| `fmt` | `e4m3` | Formato FP8 (4 bits de expoente, 3 de mantissa) |
| `activation_scheme` | `dynamic` | Escala de ativação calculada em tempo de execução, por token/bloco (sem calibração) |
| `weight_block_size` | `[128, 128]` | Um `weight_scale_inv` F32 por bloco de 128×128. `qkv_proj` GA: 13568×4096 → 108×32 escalas (4 chunks × 27 blocos de linha); SWA: 14848×4096 → 116×32 (4 × 29) |
| `store_dtype` | `mxfp4` | **Os experts estão armazenados em MXFP4**, embora o método declarado seja fp8. Ex.: `experts.0.gate_proj.weight` U8 `[2048, 2048]` (2 nibbles por byte → K=4096) + `weight_scale` U8 `[2048, 128]` (E8M0, 4096/32) |
| `mxfp4_block_size` | 32 | Tamanho do bloco MX: 4,25 bits por peso |
| `ignored_layers` | 49 nomes | Os 48 `model.layers.N.self_attn.o_proj` + `model.decoder.self_attn.o_proj` (MTP) ficam em **BF16**. São 3,2 GB de BF16 lidos a cada passo; é o alvo da otimização opcional `VLLM_MIMO_OPROJ_FP8` (#59083) |

### 4.3 Configs aninhados (multimodal)

| Bloco | Campos e valores | Leitura |
|---|---|---|
| `vision_config` | `depth` 28, `hidden_size` 1280, `intermediate_size` 4608, `num_heads` 32, `num_key_value_heads` 8, `num_query_groups` 4, `patch_size` 16, `spatial_patch_size` 16, `temporal_patch_size` 2, `spatial_merge_size` 2, `in_chans` 3, `hidden_act` silu, `out_hidden_size` 4096, `fullatt_block_indexes` [0,9,18,27], `vit_window_attn_types` (lista de 28: −1 global, 0/1 tipos de janela), `window_size` 128, `visual_token_window_size` 64, `use_sink` true, `tokens_per_second` 2 | ViT com janela, sinks e 4 blocos globais. `out_hidden_size` 4096 = projeção para o backbone. Os sinks do ViT tiveram bug de semântica (#58235, aberto) |
| `audio_config` | `input_local_layers` 6, `input_local_dim` 1024, `input_local_attn_heads` 16, `input_local_head_dim` 64, `input_local_intermediate_size` 4096, `input_local_hidden_dropout` 0, `input_full_attention` true, `audio_channels` 20, `group_size` 4, `audio_segment_size` 6000, `projection_layers` 2, `out_hidden_size` 4096, `add_post_norm` true, `partial_rotary_factor` 1.0, `rope_theta` 640000, `speech_vocab_size` "1280", `speech_zeroemb_idx` "1024" | Patch encoder de áudio (6 camadas) sobre 20 codebooks RVQ do AudioTokenizer; 4 frames por patch (25 Hz → 6,25 Hz) |
| `processor_config` | Áudio: `audio_sampling_rate` 24000, `audio_n_mels` 128, `audio_nfft` 960, `audio_hop_length` 240, `audio_window_size` 960, `audio_fmin` 0, `audio_fmax` null, `audio_kernel_size` 3, `audio_stride_size` 2, `audio_avg_pooler` 2, `audio_group_size` 4, `audio_segment_size` 6000, `audio_input_id_per_second` 25.0, `audio_zeroemb_idx` [1024]×20, `audio_channels` 20. Imagem: `image_min_pixels` 8192, `image_max_pixels` 8388608, `patch_size` 16, `merge_size` 2, `temporal_patch_size` 2, `temporal_compression_ratio` 1, `rope_type` rope. Vídeo: `fps` 1.0, `max_frames` 3600, `min_frames`/`num_frames` null, `video_min_pixels` 8192, `video_max_pixels` 8388608, `video_total_max_pixels` 268435456, `video_tokens_per_second` 2, `use_video_timestamps` true, `use_per_grid_t_timestamps` false, `video_audio_interleave_length` 0.0, `video_process_num_threads` 16. Tokens: os mesmos IDs do nível de topo + `pad_token_id` | Limites de pixels e frames que o processor aplica. Atenção: o `preprocessor_config.json` separado declara `max_pixels` 12845056 e `processor_class` `Qwen2_5_VLProcessor`, **valor diferente** dos 8388608 acima. Fixe o limite explicitamente com `--mm-processor-kwargs` |

### 4.4 `dflash/config.json` (o drafter)

`DFlashDraftModel`, `model_type` qwen3, 5 camadas `sliding_attention` (janela 1024), `hidden_size` 4096,
`intermediate_size` 16384, 64 Q / 8 KV heads, `head_dim` 128 (K e V), `partial_rotary_factor` 0.5,
`rope_theta` 10000, `is_causal` **false** (bidirecional dentro do bloco), `block_size` 8.
`dflash_config`: `target_layer_ids` [0,11,23,35,47], `mask_token_id` 151675, `num_anchors` 4096,
`loss_decay_gamma` 7.0, `attention_value_scale` 0.612, `attention_sink_bias` true.
O kit tonyd2wild relatou uma vírgula sobrando (JSON inválido) na versão do release; nas revisões atuais do RL
e do MOPD o arquivo é **JSON válido** (verificado neste estudo). Mesmo assim, valide sempre no Passo 0.

## 5. Tokenizer e chat template

- `tokenizer_class` `Qwen2Tokenizer` (BPE byte-level, `vocab.json` + `merges.txt` + `tokenizer.json`),
  `add_bos_token` false, `eos_token` `<|im_end|>`, `pad_token` `<|endoftext|>`, `model_max_length` 1048576.
- Tokens especiais: `<|im_start|>` 151644, `<|im_end|>` 151645, `<think>` 151667 / `</think>` 151668
  (**não** marcados como special, então aparecem no texto se o reasoning parser não os separar),
  `<tool_call>` 151657 / `</tool_call>` 151658, `<tool_response>` 151665/151666, placeholders de
  visão/vídeo/áudio (151652–151656, 151669–151674).
- **Chat template** (`chat_template.jinja`, 3,9 KB), no formato ChatML:
  - `tools` viram uma mensagem `system` com `You are provided with the following tools:\n\n<tools>…</tools>`.
    Não existe prompt de sistema padrão.
  - Tool calls no histórico são renderizados **compactos, sem quebras de linha**:
    `<tool_call><function=NOME><parameter=ARG>VALOR</parameter></function></tool_call>`. É parecido com o
    Qwen3-Coder, mas sem os `\n` obrigatórios. Por isso o parser estrito precisou de uma variante própria
    (#58019).
  - Mensagens de assistente no histórico sempre recebem `<think>{reasoning_content}</think>`. Cliente que não
    devolve o `reasoning_content` gera `<think></think>` vazio no histórico, o que é inofensivo mas muda o
    prefixo (e portanto o prefix cache).
  - Geração: `<|im_start|>assistant\n`. Se `enable_thinking` for **false**, o template adiciona
    `<think></think>`; se não, o modelo começa pensando (**thinking ligado por padrão**).
- Parsers no vLLM: `--reasoning-parser mimo`, `--tool-call-parser mimo`, `--enable-auto-tool-choice`.
- `generation_config.json`: `temperature` 1.0, `top_p` 0.95, `do_sample` false, `max_new_tokens` 2048.
  O card oficial usa `--generation-config vllm`, que ignora esses defaults. Duas receitas independentes
  (tonyd2wild e diffbot) viram tool calls repetidos centenas de vezes com os defaults quase gulosos do vLLM
  e passaram a usar `--generation-config auto` [medido: tonyd2wild, diffbot].

## 6. Variantes quantizadas publicadas

Tamanhos lidos pela API do HF em 2026-09-28. "KLD" indica se há medição de divergência publicada contra o
checkpoint oficial.

| Formato | Repositório | Tamanho | Motor | Observação |
|---|---|---|---|---|
| **Oficial: MXFP4 experts + FP8 dense + BF16 o_proj** | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` (e `-RL`) | 172,9 GB de pesos | vLLM (imagem `mimo-v26` ou nightly), SGLang ≥ nightly 2026-09-21 (#40448) | É **a referência**; não existe checkpoint BF16 público |
| NVFP4 | `ProCreations/…-NVFP4` (395 GB, junto com GGUF e relatórios), `tiyuvta/…`, `primitive-ai/…`, `r0b0tlab/…` (~187 GB) | ~187 GB | vLLM (ModelOpt/compressed-tensors, conforme o repo) | **Maior** que o oficial (NVFP4 = 4,5 bits + dense). Requantizar a partir de MXFP4 acumula erro. A vantagem possível é W4A4 NVFP4 no SM120/121 (padrão no v0.30, #55170) [a medir: KLD] |
| MXFP4 (outras re-embalagens) | `Yanun/MiMo-V2.6-Flash-RL-MXFP4` | — | — | Sem ganho sobre o oficial |
| EXL3 | `benthecarman/MiMo-V2.6-Flash-RL-exl3` (2,27 bpw, 85,3 GiB), `…-MOPD-exl3`, `vcruz305/…-dflash-EXL3-4.0bpw`, `MikeRoz/…-MOPD-4.03bpw-h8-exl3` | 85–170 GB | ExLlamaV3 ≥ 1.5.2 / TabbyAPI (compilar no aarch64). Não é vLLM | 2,27 bpw **cabe num Spark só**. Deltas publicados em relação ao FP8: GPQA-D −6,2%, MMLU-Pro −2,6%, PPL 5,40 na referência. A receita GLM do laboratório mostra que EXL3 em vLLM é possível, mas exige overlay próprio |
| GGUF | `ggml-org/MiMo-V2.6-Flash-MOPD-GGUF` (MXFP4 167,4 GB; Q2_K 126,2 GB; `mtp-*` e `dflash-*` separados), `AesSedai`, `TrevorJS`, `ddh0`, `kernelpool` etc. | 126–310 GB | llama.cpp | Útil como segunda referência e para `llama-perplexity --kl-divergence` |
| MLX | `mlx-community/MiMo-V2.6-Flash-RL-mxfp4-q8` (156 GB), `Vontra/…-MLX-4bit-MTP` (~154 GiB), `nativ-community/…-4bit/8bit` | 150–175 GB | mlx-lm (branch do PR #1219) | Ver [Apêndice C](#apêndice-c--rodar-o-mesmo-modelo-no-mac-com-mlx-e-contrato-openai) |
| Poda de experts (REAP) | `patrickbdevaney/MiMo-V2.6-Flash-REAP50` (94 GB), `tacodevs/…-MLX-REAP25/37/50…` | ~95 GB | vários | Remove experts: é outra classe de compressão, com risco de qualidade maior [a medir] |
| AWQ / FP8 "puro" | — | — | — | **Não encontrado** para o V2.6-Flash |

## 7. Status no vLLM (versões, PRs, issues)

**Versões.**
- **v0.30.0** (estável, 2026-09-22): **não contém** o suporte MXFP4 do MiMo-V2 (#57784, merge em 2026-09-20)
  nem o fix de QKV (#57508, merge em 2026-09-19). Verificado pela API do GitHub: os commits `9b2f34cad4` e
  `211e252d0b` estão "diverged" em relação à tag v0.30.0. O recipe oficial diz que "stable ≤ 0.29.0 cannot
  load mxfp4-stored weights" e manda usar nightly posterior a 2026-09-20 ou a imagem dedicada.
- **Imagens dedicadas** (Docker Hub, 2026-09-21): `vllm/vllm-openai:mimo-v26`, `mimo-v26-aarch64-cu130`
  (arm64, 9,7 GB), `mimo-v26-aarch64-cu129`. Elas contêm vLLM `0.29.1rc1.dev449+geb8798058`
  [medido: Plaaasma, diffbot]. **No GB10, use `-cu130`**: a `cu129` compila parte dos kernels de MoE e `_C` só
  como `sm_120a`, que o sm_121 não executa ("no kernel image is available") [medido: Plaaasma].
- **Nightly** aarch64: `vllm/vllm-openai:nightly-aarch64` e `nightly-dev-arm64-cu130-<sha>` (2026-09-27/28).
  É a "lane upstream", no mesmo espírito da lane v0.30 do Qwen.

**PRs e issues relevantes** (estado em 2026-09-28):

| # | Título resumido | Estado | Efeito na receita |
|---|---|---|---|
| [#57784](https://github.com/vllm-project/vllm/pull/57784) | MXFP4 MoE + router BF16 + DFlash para o MiMo V2 | **merged** 09-20 (não está no v0.30.0) | Pré-requisito; ajuste de value scale do drafter |
| [#57508](https://github.com/vllm-project/vllm/pull/57508) | Sharding do QKV FP8 fundido (`ckpt_tp = num_key_value_heads`) | **merged** 09-19 | Sem ele, TP≠4 quebra nas camadas SWA |
| [#59053](https://github.com/vllm-project/vllm/pull/59053) | QKV exato (sem requantizar) atrás de `VLLM_MIMO_EXACT_QKV` | aberto | Remove o erro de requantização em TP2 |
| [#58142](https://github.com/vllm-project/vllm/pull/58142) | Estado de pareamento do QKV entre chamadas de `load_weights` | aberto | Com `fastsafetensors`/`eager`/InstantTensor o QKV fica **zerado** e a saída vira lixo |
| [#58184](https://github.com/vllm-project/vllm/pull/58184) | Ignorar QKV de camadas inexistentes (boot truncado) | aberto | Viabiliza o teste de paridade com N camadas |
| [#58128](https://github.com/vllm-project/vllm/pull/58128) | KV FP8 no `TRITON_ATTN_DIFFKV` e no MiMo-V2.6-Flash | aberto | Sem ele, **`--kv-cache-dtype fp8` é ignorado sem aviso**. Validado em 2× DGX Spark: KV 1.365.757 → 2.731.530 tokens no mesmo pool de 17 GiB |
| [#58176](https://github.com/vllm-project/vllm/pull/58176) | Encaminhar `cache_config` sem vazar `sliding_window` para as globais | fechado sem merge (dividido de #58128) | Mostra a correção mínima que o overlay precisa |
| [#58175](https://github.com/vllm-project/vllm/pull/58175) / [#58207](https://github.com/vllm-project/vllm/pull/58207) / RFC [#58638](https://github.com/vllm-project/vllm/issues/58638) | Agrupamento do KV híbrido com drafters | #58175 fechado; #58207 aberto | Grupo 5→3 dá +10,57% de capacidade (1.606.781 → 1.776.588 tokens, MiMo BF16 TP2) |
| [#59054](https://github.com/vllm-project/vllm/issues/59054) / [#59085](https://github.com/vllm-project/vllm/pull/59085) / [#58141](https://github.com/vllm-project/vllm/pull/58141) / [#59052](https://github.com/vllm-project/vllm/pull/59052) | Verify do spec decode (q_len>1) forçado no kernel 2D: 7–20× mais lento por chamada em contexto longo | abertos | Decode com DFlash em contexto longo |
| [#58482](https://github.com/vllm-project/vllm/pull/58482) | Split-K adaptativo à ocupação no SM120 (inclui DiffKV) | aberto | A regra é específica de SM120; confirme se vale no sm_121 [a medir] |
| [#58798](https://github.com/vllm-project/vllm/pull/58798) | RoPE parcial + value scale + escrita DiffKV fundidos | aberto | Performance |
| [#59083](https://github.com/vllm-project/vllm/pull/59083) | `o_proj` em FP8 online atrás de `VLLM_MIMO_OPROJ_FP8` | aberto | −0,8 GiB/rank e menos bytes por passo |
| [#58019](https://github.com/vllm-project/vllm/pull/58019) | Tool calling estrito do MiMo-V2.6 | aberto | `tool_choice=required`/nomeado usa o wire format errado sem ele |
| [#57804](https://github.com/vllm-project/vllm/pull/57804) | Carregar o sink bias pelo `weight_loader` | aberto | Relevante para RL e atualização de pesos |
| [#58235](https://github.com/vllm-project/vllm/pull/58235) / [#58938](https://github.com/vllm-project/vllm/pull/58938) | Sinks do ViT e `packed_modules_mapping` do omni; `embedding_fields` para EPD | abertos | Qualidade multimodal |
| [#46669](https://github.com/vllm-project/vllm/issues/46669) | Saída ruim com spec decode + async scheduling (MiMo V2.5-Pro) | aberto | Rode com `--no-async-scheduling` |
| [DeepGEMM#417](https://github.com/deepseek-ai/DeepGEMM/issues/417) | DeepGEMM corrompe GEMMs FP8 no SM12x | — | `VLLM_USE_DEEP_GEMM=0` |

## 8. Opções de decodificação especulativa

| Opção | Onde está | Configuração vLLM | O que se sabe |
|---|---|---|---|
| **DFlash** (5 camadas SWA, bloco 8) | `dflash/` no próprio repo | `{"method":"dflash","model":"<snapshot>/dflash","num_speculative_tokens":7,"draft_tensor_parallel_size":2}` | Em 2× Spark, vLLM, k=7: aceita 6,8 de 7 em texto estruturado e contagem, ~5,2 em código, ~1,2 em prosa [medido: tonyd2wild]. No RTX PRO 6000 a diffbot escolheu k=3 (aceitação média 2,93 com temperatura 1,0) |
| **MTP** (3 camadas) | `model_mtp.safetensors` | `{"method":"mtp","num_speculative_tokens":1..3}` | No SGLang (EAGLE multi-layer) foi melhor em prosa: 35,0 contra 25,8 tok/s do DFlash; e pior em código: 46,0 contra 69,8 [medido: MiaAI-Lab]. No vLLM o `mtp` foi testado com k=1 em #58142 [a medir: GB10] |
| ngram | — | `{"method":"ngram",…}` | Barato para edição de código em agentes, mas tem bug com o parser de tool calls do Qwen (#56077) [a medir] |
| Nenhum | — | — | Linha de base obrigatória no benchmark |

## 9. Licença

**MIT** para o MOPD e o RL, incluindo o drafter e os encoders que estão no mesmo repositório. vLLM é
Apache-2.0. Patches de terceiros que você vendorizar têm licença própria: leia o `LICENSE` e o
`THIRD_PARTY_NOTICES` de cada kit antes de copiar. A receita Qwen do laboratório, por exemplo, vendoriza
código AGPL-3.0 e declara isso.

---

# Parte II — O passo a passo

Cada passo tem **Objetivo → Conceito → Como fazer → Aplicado ao MiMo → Critério de saída**. Não avance sem
cumprir o critério de saída: é isso que separa uma receita reproduzível de um "funcionou na minha máquina".

## Passo 0 — Triagem

**Objetivo.** Antes de baixar 170 GB, descobrir o que o modelo tem de diferente e onde o motor vai quebrar.

**Conceito.** Um checkpoint se descreve em três camadas: (1) o **card**, com a intenção e os comandos oficiais
(pensados para H200/B200, nunca para GB10); (2) o **`config.json`**, com a geometria que determina memória,
paralelismo e kernels; (3) os **cabeçalhos safetensors**, com os dtypes e shapes reais, que às vezes contradizem
o config. Aqui, por exemplo, `quant_method: fp8` convive com experts MXFP4. O cabeçalho de um safetensors são os
primeiros `8 + N` bytes do arquivo, então dá para lê-lo com *range requests* sem baixar o shard.

**Como fazer.**

```bash
# 0.1 Metadados, revisão e árvore do repositório (sem baixar pesos)
curl -s https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Flash-MOPD | jq '{sha,lastModified,tags}'
curl -s "https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/tree/main?recursive=true" \
  | jq '[.[]|.size] | add / 1e9'            # ~177.8 GB no repo

# 0.2 Config, template, generation config, drafter (arquivos pequenos)
for f in config.json generation_config.json chat_template.jinja tokenizer_config.json dflash/config.json; do
  curl -sL "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/raw/main/$f" -o "triage/$(basename $f)"
done
python3 -c "import json; json.load(open('triage/config.json')); print('dflash ok')"   # JSON válido?

# 0.3 Cabeçalhos safetensors (dtypes/shapes reais) — script abaixo
python3 triage.py XiaomiMiMo/MiMo-V2.6-Flash-MOPD 2479e2d0029eca9a34cc7e7f55a121925f81908e
```

`triage.py` (testado neste estudo contra o MOPD):

```python
#!/usr/bin/env python3
"""Triagem de um checkpoint no Hub sem baixar os pesos: config, índice e cabeçalhos por range request."""
import collections, json, re, struct, sys, urllib.request
REPO = sys.argv[1]; REV = sys.argv[2] if len(sys.argv) > 2 else "main"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}"

def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def header(fname):
    n = struct.unpack("<Q", get(f"{BASE}/{fname}", "0-7"))[0]
    return json.loads(get(f"{BASE}/{fname}", f"8-{8 + n - 1}"))

cfg = json.loads(get(f"{BASE}/config.json"))
idx = json.loads(get(f"{BASE}/model.safetensors.index.json")); wm = idx["weight_map"]
print("arch:", cfg["architectures"], "| índice:", idx.get("metadata"), "| tensores:", len(wm))
pat = cfg.get("hybrid_layer_pattern") or []
print("globais:", [i for i, x in enumerate(pat) if x == 0], "| SWA:", pat.count(1))
q = cfg.get("quantization_config", {})
print("quant:", {k: v for k, v in q.items() if k != "ignored_layers"},
      collections.Counter(re.sub(r"\.\d+\.", ".N.", x) for x in q.get("ignored_layers", [])))
wanted = ["model.layers.0.self_attn.qkv_proj.weight", "model.layers.1.self_attn.qkv_proj.weight",
          "model.layers.1.mlp.experts.0.gate_proj.weight", "model.mtp.layers.0.self_attn.qkv_proj.weight"]
for f in dict.fromkeys(wm[t] for t in wanted if t in wm):
    for k, v in header(f).items():
        m = re.match(r"model\.(layers\.[01]|mtp\.layers\.0)\.", k)
        if k != "__metadata__" and m and (".experts." not in k or ".experts.0." in k):
            print(f"  {k:66s} {v['dtype']:8s} {v['shape']}")
```

**Aplicado ao MiMo** (saída real, resumida):

```
arch: ['MiMoV2ForCausalLM'] | índice: {'save_format': 'mxfp4', 'total_size': 172923364096, 'tp_size': 4}
globais: [0, 5, 11, 17, 23, 29, 35, 41, 47] | SWA: 39
quant: {'quant_method': 'fp8', 'store_dtype': 'mxfp4', 'mxfp4_block_size': 32, 'weight_block_size': [128,128], ...}
       Counter({'model.layers.N.self_attn.o_proj': 48, 'model.decoder.self_attn.o_proj': 1})
  model.layers.0.self_attn.qkv_proj.weight        F8_E4M3  [13568, 4096]   # GA: 4×(3072 Q + 192 K + 128 V)
  model.layers.0.self_attn.qkv_proj.weight_scale_inv  F32  [108, 32]       # 4 chunks × 27 blocos
  model.layers.1.self_attn.qkv_proj.weight        F8_E4M3  [14848, 4096]   # SWA: 4×(3072 + 384 + 256)
  model.layers.1.self_attn.qkv_proj.weight_scale_inv  F32  [116, 32]       # 4 × 29
  model.layers.1.self_attn.attention_sink_bias    BF16     [64]
  model.layers.1.mlp.experts.0.gate_proj.weight   U8       [2048, 2048]    # MXFP4: 2 nibbles/byte
  model.layers.1.mlp.experts.0.gate_proj.weight_scale U8   [2048, 128]     # E8M0 por 32
  model.layers.0.self_attn.o_proj.weight          BF16     [4096, 8192]    # ignored_layers → BF16
  model.mtp.layers.0.self_attn.qkv_proj.weight    F8_E4M3  [14848, 4096]   # MTP é do tipo SWA (tem sink)
```

**Checklist de triagem** (preencha para qualquer modelo novo):

| Pergunta | MiMo-V2.6-Flash |
|---|---|
| Arquitetura registrada no motor? | Sim no main e na imagem `mimo-v26`; **não** no v0.30.0 |
| Remote code? | Sim (`--trust-remote-code`) |
| Tipos de atenção e dims | GA + SWA-128, DiffKV 192/128, sinks só nas SWA, RoPE parcial |
| KV heads por tipo e divisibilidade | 4 (GA) e 8 (SWA): TP ∈ {1, 2, 4} sem replicação |
| Formato dos pesos | MXFP4 (experts) + FP8 bloco 128 + BF16 |
| Pré-sharding no disco? | **Sim, `tp_size: 4`** no QKV fundido |
| Speculators embutidos | MTP-3 e DFlash-5 |
| Multimodal | Imagem, vídeo, áudio (encoders: ~1,7 GiB BF16 replicados) |
| Esquisitices | `attention_chunk_size` ignorado; `max_pixels` divergente entre dois arquivos; `sliding_window` global |

**Critério de saída.** Uma tabela como a de cima, mais uma lista de riscos no motor ("DiffKV sem FP8",
"QKV pré-fatiado", "MXFP4 no sm_121"), cada risco ligado a um PR ou issue.

## Passo 1 — Matemática de memória

**Objetivo.** Saber, antes de lançar, quanto sobra para KV e, com isso, quanto contexto e quantas requisições
simultâneas cabem.

**Conceito.** No GB10 a memória é **unificada**: o sistema operacional, o page cache, o Docker e o vLLM
disputam os mesmos 128 GB. O `--gpu-memory-utilization` (GMU) é uma fração dos 121,69 GiB que o CUDA enxerga.
O vLLM divide esse orçamento assim:
`orçamento = GMU × total = pesos + (ativações de pico + non-torch + CUDA graphs) + pool de KV`.
O pool de KV é o que sobra, e ele é compartilhado por todas as requisições. Em modelos híbridos, as camadas SWA
custam um valor fixo por requisição (a janela), e só as camadas globais crescem com o contexto.

**1.1 Pesos por formato** [estimativa a partir dos shapes; confere com o medido]:

| Componente | Parâmetros | Formato | Total | Por rank (TP=2) |
|---|---|---|---|---|
| Experts (47 × 256 × 3 × 2048 × 4096) | 302,8B | MXFP4 (4,25 bits) | 160,9 GB (149,8 GiB) | 74,9 GiB |
| `qkv_proj` (39 SWA + 9 GA) | 2,87B | FP8 | 2,87 GB | 1,34 GiB |
| `o_proj` | 1,61B | BF16 | 3,22 GB | 1,50 GiB |
| MLP densa (camada 0) | 0,2B | FP8 | 0,20 GB | 0,09 GiB |
| Router (gate) | 0,05B | BF16 | 0,10 GB | 0,09 GiB (replicado) |
| Embedding + `lm_head` | 1,25B | BF16 | 2,50 GB | 1,16 GiB |
| **Backbone** | | | **169,8 GB (158,1 GiB)** | **79,1 GiB** |
| Encoders de visão e áudio | ~0,8B | BF16 | ~1,7 GB | ~1,6 GiB (replicado, `--mm-encoder-tp-mode data`) |
| DFlash | ~1,47B | BF16 | 2,94 GB | ~1,4 GiB (TP=2) |
| MTP (se usado no lugar do DFlash) | — | FP8/BF16 | 1,19 GB | ~0,6 GiB |
| **Total por rank** | | | | **≈ 82,0 GiB** [estimativa] contra **82,7 GiB** [medido: MiaAI-Lab, SGLang TP2] e 82,92 GB [medido: Smilehyu] |

Comparação de formatos para o backbone inteiro: BF16 ≈ 620 GB, FP8 ≈ 310 GB, NVFP4 ≈ 187 GB (repositórios da
comunidade), **MXFP4 oficial ≈ 173 GB**, EXL3 2,27 bpw ≈ 92 GB. **Só os formatos de ~4 bits ou menos cabem
em 2 Sparks.**

**1.2 KV por token** (por rank, TP=2):

| Camadas | Fórmula | BF16 | FP8 |
|---|---|---|---|
| 9 globais | 9 × 2 KV heads × (192 + 128) × bytes | **11.520 B/token** | **5.760 B/token** |
| 39 SWA | 39 × 4 × 320 × bytes, **só para a janela** | 99.840 B/token × ~192–256 tokens (janela + k + arredondamento de bloco) ≈ 19–26 MB por requisição | ≈ 10–13 MB por requisição |
| DFlash (5 SWA, janela 1024) | 5 × 4 × (128 + 128) × 2 | ≈ 11 MB por requisição (o drafter fica em BF16 no GB10) | — |

Duas consequências:
- **O custo que cresce com o contexto é pequeno:** 5,76 KB/token com FP8. Para comparação: 13,3 KB no Qwen3.8
  (TP2, BF16) e 656 B por camada no GLM MLA.
- **O alocador híbrido desperdiça espaço.** Com grupos de 5 camadas (o padrão atual), as 9 globais ocupam 10
  "slots": +11%. O PR #58175 mediu +10,57% de capacidade ao passar para grupos de 3.

**1.3 Orçamento com GMU** (`scripts/mem_budget.py`, testado):

```python
#!/usr/bin/env python3
"""Orçamento de KV por rank. Uso:
python3 mem_budget.py config.json --tp 2 --kv fp8 --gmu 0.85 --weights-gib 82.7 --overhead-gib 4"""
import argparse, json, math
ap = argparse.ArgumentParser(); ap.add_argument("config")
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--kv", choices=["bf16", "fp8"], default="fp8")
ap.add_argument("--gmu", type=float, default=0.85); ap.add_argument("--visible-gib", type=float, default=121.69)
ap.add_argument("--weights-gib", type=float, default=82.7); ap.add_argument("--overhead-gib", type=float, default=4.0)
ap.add_argument("--group-size", type=int, default=5); ap.add_argument("--block", type=int, default=64)
ap.add_argument("--spec-k", type=int, default=7)
ap.add_argument("--ctx", type=int, nargs="*", default=[32768, 131072, 262144, 1048576])
a = ap.parse_args(); c = json.load(open(a.config)); b = 2 if a.kv == "bf16" else 1
pat = c["hybrid_layer_pattern"]; n_ga, n_swa = pat.count(0), pat.count(1)
ga_tok = n_ga * (c["num_key_value_heads"] // a.tp) * (c["head_dim"] + c["v_head_dim"]) * b
swa_tok = n_swa * (c["swa_num_key_value_heads"] // a.tp) * (c["swa_head_dim"] + c["swa_v_head_dim"]) * b
swa_req = swa_tok * (math.ceil((c["sliding_window"] + a.spec_k) / a.block) + 1) * a.block
pad = math.ceil(n_ga / a.group_size) * a.group_size / n_ga
kv_gib = a.visible_gib * a.gmu - a.weights_gib - a.overhead_gib
print(f"KV global {ga_tok} B/tok/rank (x{pad:.2f} enchimento) | SWA fixo {swa_req/1e6:.1f} MB/req | pool {kv_gib:.2f} GiB")
cap = kv_gib * 2**30 / (ga_tok * pad); print(f"capacidade ~{cap/1e6:.2f} M tokens")
for L in a.ctx:
    print(f"  ctx {L:>8}: {kv_gib*2**30/(L*ga_tok*pad + swa_req):6.2f} requisições cheias")
```

Resultados [estimativa] com pesos de 82,7 GiB e 4 GiB de overhead:

| GMU | Orçamento | Pool de KV | FP8: tokens | FP8: requisições de 256K | BF16: tokens |
|---|---|---|---|---|---|
| 0,80 | 97,35 GiB | 10,65 GiB | ~1,79 M | ~6,8 | ~0,89 M |
| **0,85** | 103,44 GiB | **16,74 GiB** | **~2,81 M** | **~10,6** | ~1,40 M |
| 0,88 | 107,09 GiB | 20,4 GiB | ~3,42 M | ~13 | ~1,71 M |
| 0,90 | 109,52 GiB | 22,8 GiB | ~3,8 M | ~14,6 | ~1,9 M |

Confira com as medições:
- #58128 (2× Spark, pool de 17 GiB): **BF16 1.365.757 / FP8 2.731.530** tokens. Bate com ~1,40M e ~2,81M.
- tonyd2wild (GMU 0,90, `max-model-len` 300K, omni + DFlash-7): **13,89 GiB / 1.835.052 tokens**. Bem abaixo da
  estimativa porque o *profiling* multimodal infla o pico de ativações, e porque o vLLM informa "tokens" de
  modelos híbridos como `blocos / blocos-por-requisição × max_model_len`, um número que muda com
  `max_model_len` [medido: tonyd2wild].

**Três regras de ouro do GB10:**
1. **O log manda.** Leia `Available KV cache memory` e `GPU KV cache size` e recalibre o `--overhead-gib`.
2. **Deixe folga no host.** O processo da API, o processor multimodal (o cache padrão é de 4 GiB de RAM:
   use `--mm-processor-cache-gb 1`), o page cache e o SO usam a mesma memória. Mantenha `MemAvailable` ≥ 4–6 GiB
   no pico. As receitas do laboratório desligam o `earlyoom` e liberam o page cache do checkpoint com
   `posix_fadvise(DONTNEED)` antes do boot.
3. **O vLLM recusa subir** se, no início, a memória livre for menor que `GMU × total`. Com GMU 0,90 são
   necessários 109,5 GiB livres. Por isso a comunidade usa 0,80–0,85 e fixa o pool com
   `--kv-cache-memory <bytes>` [medido: tonyd2wild, Plaaasma].

**1.4 Quanto contexto e quanta concorrência** [estimativa, FP8, GMU 0,85]: 1M de contexto cabe ~2,7× ao mesmo
tempo; 8 agentes de 128K cabem com folga de ~2,6×. Nessas condições quem limita é o `max_num_seqs`, não o
cache. Com BF16, 1M cabe só 1,3×.

**Critério de saída.** Uma tabela "previsto × medido" para pesos, pool e tokens. O desvio aceitável é de ±10%;
acima disso, descubra o motivo antes de seguir.

## Passo 2 — Quantização e checkpoint

**Objetivo.** Escolher o arquivo de pesos com a melhor relação entre qualidade, memória e velocidade para
**este** hardware, e ter um jeito de medir a perda.

**Conceito.** Num MoE com decode limitado por banda, o que importa são os **bytes lidos por token** (os experts
ativos mais tudo o que é denso) e o **erro em relação à referência**, medido como KLD (divergência de
Kullback-Leibler) entre as distribuições do próximo token. Perplexidade sozinha esconde erros que se cancelam.
Quantizar de novo algo que já está em 4 bits (MXFP4 → NVFP4, por exemplo) soma um segundo erro de
arredondamento.

**2.1 Matriz de decisão para o MiMo:**

| Opção | Por rank | Qualidade | Kernels no sm_121 | Decisão |
|---|---|---|---|---|
| **Oficial (MXFP4 + FP8 + BF16)** | 82,7 GiB | É a referência (KLD 0 por definição) | Marlin MXFP4 (W4A16) ou W4A8 com `VLLM_MARLIN_INPUT_DTYPE=fp8` (+15% de prefill no SM120 [medido: diffbot]) | **Padrão da receita** |
| Oficial + `o_proj` FP8 online (#59083) | −0,8 GiB | Erro por tensor [a medir] | FP8 por tensor | Lane opcional, como o `FP8_DENSE` do Qwen |
| NVFP4 da comunidade | ~87 GiB | Requantização dupla [a medir KLD] | W4A4 NVFP4 (padrão no v0.30 para SM120/121) | Experimento de prefill; não é padrão |
| EXL3 2,27–4 bpw | 43–80 GiB | 2,27 bpw: GPQA −6,2% | ExLlamaV3; em vLLM exige overlay (ver a receita GLM) | Lane de 1 Spark ou de contexto enorme |
| GGUF MXFP4 / Q2_K | 80 / 60 GiB | Q2_K [a medir] | llama.cpp | Segunda referência de KLD |

**Por que o oficial:** ele é o menor formato de 4 bits disponível (4,25 bits contra 4,5 do NVFP4), é
exatamente o que o fabricante avaliou e dispensa calibração. O custo é depender de kernels MXFP4 no sm_121: o
Marlin funciona segundo os dois kits vLLM, e o DeepGEMM precisa ficar desligado.

**2.2 Baixar e verificar:**

```bash
export MODEL_ID=XiaomiMiMo/MiMo-V2.6-Flash-MOPD
export MODEL_REV=2479e2d0029eca9a34cc7e7f55a121925f81908e
export MODEL_DIR=/models/MiMo-V2.6-Flash-MOPD
hf download "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR" \
   --exclude "*.pdf" "assets/*"           # ~175 GB; o PDF e as imagens não são necessários
python3 verify-weights.py "$MODEL_DIR"    # tamanho de cada arquivo = API do Hub; todos os shards do índice existem
```

O `verify-weights.py` segue o padrão de `Qwen3.8…/verify-weights.py` e do "integrity gate" do Smilehyu: compare
o tamanho de cada arquivo com a árvore da API, confira que todos os arquivos citados em
`model.safetensors.index.json` existem e calcule o sha256 dos arquivos pequenos (configs e template).

**2.3 Quantizar você mesmo e medir** (exemplo: `o_proj` em FP8, ou NVFP4 dos experts).

- Ferramentas: `llm-compressor` e `nvidia-modelopt` para gerar FP8/NVFP4; `exllamav3`
  (`convert.py` + `eval/model_diff.py`) para EXL3; `llama-quantize` + `llama-perplexity --kl-divergence-base`
  para GGUF.
- **A referência precisa estar no mesmo motor.** Não existe BF16 público e o HF não roda 173 GB num Spark.
  Então sirva o oficial no vLLM (é a referência) e compare o candidato contra ele pelas *prompt logprobs*.

Esqueleto de KLD top-k via API (vLLM aceita `prompt_logprobs` como parâmetro extra do `/v1/completions`):

```python
# kld_topk.py — KLD aproximada (suporte top-k) entre dois servidores vLLM com o mesmo tokenizer
import json, math, sys, requests
REF, CAND, MODEL_REF, MODEL_CAND, K = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 20
docs = [json.loads(l)["text"] for l in open("kld_corpus.jsonl")]   # ex.: 64 janelas de 2048 tokens (código + prosa)

def plp(url, model, text):
    r = requests.post(f"{url}/v1/completions", json={"model": model, "prompt": text, "max_tokens": 1,
                      "temperature": 0, "prompt_logprobs": K}, timeout=600).json()
    return r["choices"][0]["prompt_logprobs"][1:]            # posição 0 não tem distribuição

tot, n, top1 = 0.0, 0, 0
for d in docs:
    for p, q in zip(plp(REF, MODEL_REF, d), plp(CAND, MODEL_CAND, d)):
        P = {t: v["logprob"] for t, v in p.items()}; Q = {t: v["logprob"] for t, v in q.items()}
        zP = math.log(sum(math.exp(v) for v in P.values()))   # renormaliza no suporte top-k da referência
        kl = sum(math.exp(lp - zP) * ((lp - zP) - Q.get(t, min(Q.values()) - 2.0)) for t, lp in P.items())
        tot += kl; n += 1; top1 += max(P, key=P.get) == max(Q, key=Q.get)
print(f"KLD média ≈ {tot/n:.4f} nats | concordância top-1 {top1/n:.3%} | {n} posições")
```

Limites do método: a KLD top-k subestima a KLD real e a chave de token depende da versão (ID ou string). Use
exatamente o mesmo corpus nos dois lados. Como escala, o painel da receita GLM mediu EXL3 4 bpw a 0,0246 nats,
praticamente igual ao FP8 oficial daquele stack, e NVFP4 a 0,061. Uma heurística razoável: **< 0,03 está ótimo,
0,03–0,06 aceitável, > 0,1 aparece em uso**. Confirme com evals de tarefa (GSM8K, HumanEval+, needle).

**Critério de saída.** Checkpoint baixado, verificado e com revisão fixada. Se você escolheu um checkpoint que
não é o oficial, uma tabela de KLD e evals contra o oficial.

## Passo 3 — Topologia (TP/EP/PP)

**Objetivo.** Decidir como dividir o modelo entre os 2 GB10.

**Conceito.**
- **TP** (tensor parallel) corta cada matriz. Exige 2 all-reduces por camada (atenção e MoE), e cada um é
  sensível à latência do link (8 KB por token no decode). Pede divisibilidade: heads % TP, KV heads % TP
  (ou replicação), `moe_intermediate/TP` múltiplo do bloco de quantização.
- **EP** (expert parallel) põe experts inteiros em cada rank e troca tokens por all-to-all ou
  allgather/reduce-scatter. Sem expert compartilhado, a carga depende do roteamento; com poucos tokens (decode
  batch 1), um rank pode receber 6 dos 8 experts.
- **PP** (pipeline) corta as camadas em fatias (24/24). Passa só ativações entre estágios, mas com batch 1 um
  nó fica ocioso enquanto o outro trabalha. Serve para throughput com muitos streams, não para latência.

**Aplicado ao MiMo:**

| Checagem | TP=2 | TP=4 (4 Sparks) |
|---|---|---|
| Q heads 64 | 32/rank ✔ | 16 ✔ |
| KV GA 4 / SWA 8 | 2 / 4 ✔ | 1 / 2 ✔ |
| Experts 256 (EP) | 128/rank ✔ | 64 ✔ |
| `moe_intermediate` 2048 (TP) | 1024 = 32 × 32 ✔ | 512 ✔ |
| QKV pré-fatiado em 4 | **cada rank junta 2 chunks** (requantiza ou usa o caminho exato de #59053) | **encaixe exato** (1 chunk por rank) |
| Vocabulário 152.576 | ✔ | ✔ |
| DFlash (64 Q / 8 KV) | `draft_tensor_parallel_size` 2 ✔ | 4 ✔ |

**Decisão:** **TP=2** (uma GPU por nó, `--nnodes 2`), com **EP como variável de A/B no Passo 9**. Evidências:
tonyd2wild roda TP2 sem EP; a diffbot usa `EP=0` como padrão; o Smilehyu (SGLang) roda TP2/EP2. PP fica fora
porque dobra a latência de decode com batch pequeno. O suporte a spec decode sob PP é novo (#50514 no v0.30) e
adiciona risco.

Custo de comunicação [estimativa]: são 96 all-reduces por passo de decode (48 camadas × 2), cada um de ~15–30 µs
em RoCE. Isso dá 1,5–3 ms num passo de ~25 ms (~10%). Se o seu `NCCL_DEBUG=INFO` mostrar `NET/Socket` em vez de
`NET/IB`, o RoCE não está sendo usado e esse custo explode. Corrija `NCCL_IB_HCA` e `NCCL_IB_GID_INDEX` como
na receita Qwen.

**Critério de saída.** Uma tabela de divisibilidade sem "✘", uma topologia escolhida e as alternativas
anotadas para o Passo 9.

## Passo 4 — Suporte no motor

**Objetivo.** Fazer o vLLM carregar o modelo **corretamente**, não só carregar. Corrigir o que falta com
overlays auditáveis e provar a correção numericamente.

**4.1 O modelo está registrado?**

```bash
IMAGE=vllm/vllm-openai:mimo-v26-aarch64-cu130
docker run --rm --entrypoint python3 $IMAGE -c "
import vllm, vllm.model_executor.models.registry as r
print(vllm.__version__)
print({k: v for k, v in r._VLLM_MODELS.items() if 'MiMo' in k or 'DFlash' in k})"
# Esperado: MiMoV2ForCausalLM -> ('mimo_v2', ...), MiMoV2OmniForCausalLM, MiMoV2MTPModel, DFlashDraftModel
```

O nome interno do dicionário (`_VLLM_MODELS`) pode mudar entre versões. Se falhar, faça `grep -n MiMo` em
`vllm/model_executor/models/registry.py` dentro da imagem.

**4.2 O que falta: a lista de lacunas no GB10.** Monte uma tabela sintoma → causa → correção. Para o MiMo, com a
imagem `mimo-v26-aarch64-cu130`:

| # | Sintoma | Causa | Correção | Como provar |
|---|---|---|---|---|
| G1 | `--kv-cache-dtype fp8` "funciona", mas o pool não dobra | Decoder e MTP criam `Attention` sem `cache_config`, e o `TRITON_ATTN_DIFFKV` rejeita KV quantizado | Overlay de `mimo_v2.py` (encaminhar uma **cópia** do `cache_config` com `sliding_window=None`) + backend DiffKV com FP8 (porte do #58128) | `GPU KV cache size` ~2× o valor em BF16; needle de 128K passa |
| G2 | Lixo na saída com `--load-format fastsafetensors`/InstantTensor ou `--safetensors-load-strategy eager` | O par peso+escala do QKV é descartado quando atravessa chamadas de `load_weights` | Porte do #58142, ou ficar em `safetensors` com estratégia `lazy` | Soma e NaN de `qkv_proj.weight` pós-load; teste "capital da França" |
| G3 | Erro pequeno e sistemático de K/V em TP2 | Requantização no `_shard_fp8_qkv_proj` | Porte do #59053 (`VLLM_MIMO_EXACT_QKV=1`) | Teste CPU que compara os shards TP2 com o checkpoint (a diffbot tem `test_mimo_exact_qkv_audit.py`) |
| G4 | DFlash não sobe com a classe omni | `MiMoV2OmniForCausalLM` sem o marcador `SupportsEagle3` | Adicionar o marcador; os hooks já delegam [medido: tonyd2wild, patch 02] | Boot com `method: dflash` e aceitação > 0 no `/metrics` |
| G5 | GEMM FP8 com resultado errado | DeepGEMM no SM12x | `VLLM_USE_DEEP_GEMM=0` | GSM8K canário |
| G6 | `cutlass_gemm_caller … Invalid status` / `Error Internal` | GEMM FP8 com blocos CUTLASS c3x em certos shapes no sm_121 | `--linear-backend triton` (obrigatório em TP4 segundo o tonyd2wild; o Plaaasma precisou em TP2) | Boot limpo |
| G7 | Decode lento em contexto longo com DFlash | O verify (q_len>1) cai no kernel 2D (#59054) | Porte de split-KV no verify (patch `triton_unified_attention_diffkv.py` da diffbot; #59085/#58141) | tok/s em 32K e 128K antes/depois |
| G8 | Saída corrompida sob concorrência | Spec decode + async scheduling (#46669) | `--no-async-scheduling` | 24 gerações longas concorrentes sem caracteres espúrios [medido: tonyd2wild] |

**4.3 Como escrever um overlay** (o padrão do laboratório: bind-mount de um `.py` gerado a partir do arquivo da
**própria imagem**, sem rebuild):

```bash
VPY=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p files/orig files/overlay
docker run --rm --entrypoint cat $IMAGE $VPY/model_executor/models/mimo_v2.py > files/orig/mimo_v2.py
sha256sum files/orig/mimo_v2.py | tee files/orig/mimo_v2.py.sha256
python3 files/patch_mimo_cache_config.py files/orig/mimo_v2.py files/overlay/mimo_v2.py
# na execução: -v $PWD/files/overlay/mimo_v2.py:$VPY/model_executor/models/mimo_v2.py:ro  (nos DOIS nós)
```

`files/patch_mimo_cache_config.py`: **fail-closed** (aborta se a âncora não bate), **idempotente** e capaz de
detectar quando a correção já está upstream:

```python
#!/usr/bin/env python3
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
MARK = "# [mimo-recipe] cache_config per layer"
if MARK in s:
    open(dst, "w").write(s); print("[overlay] já aplicado"); sys.exit(0)
if re.search(r"cache_config\s*=\s*layer_cache_config", s):
    open(dst, "w").write(s); print("[overlay] already upstream"); sys.exit(0)
# Âncora 1: início do __init__ do MiMoV2FlashDecoderLayer
a1 = "        quant_config = vllm_config.quant_config\n        layer_id = extract_layer_index(prefix)\n"
if s.count(a1) != 1:
    sys.exit("[overlay] âncora 1 não encontrada exatamente 1x — a imagem mudou; revise o patch")
s = s.replace(a1, a1 + f"        {MARK}\n        import copy\n"
    "        layer_cache_config = copy.copy(vllm_config.cache_config)\n"
    "        if layer_cache_config is not None:\n"
    "            layer_cache_config.sliding_window = None  # globais continuam globais\n")
# Âncora 2: as duas construções de MiMoV2Attention recebem cache_config
n = s.count("                quant_config=quant_config,\n                partial_rotary_factor=")
if n != 2:
    sys.exit(f"[overlay] esperava 2 construções de MiMoV2Attention, achei {n}")
s = s.replace("                quant_config=quant_config,\n                partial_rotary_factor=",
              "                quant_config=quant_config,\n                cache_config=layer_cache_config,\n"
              "                partial_rotary_factor=")
open(dst, "w").write(s); print("[overlay] cache_config encaminhado (2 camadas-tipo)")
```

Este patch **sozinho** transforma o G1 de "silencioso" em "erro de startup" no GB10, porque o backend DiffKV
Triton ainda rejeita FP8. Esse é o comportamento certo: falhar alto. O segundo overlay porta o #58128 para
`v1/attention/backends/triton_attn_diffkv.py` e `v1/attention/ops/triton_unified_attention_diffkv.py`.
Baixe o diff (`curl -L https://github.com/vllm-project/vllm/pull/58128.diff`), rode `patch --dry-run` contra os
arquivos extraídos da imagem e, se não aplicar limpo (o PR é contra o main, e a imagem é o dev449), refaça à mão
com âncoras, como acima. **Nunca** aplique contra o `main` e monte na imagem: arquivos de versões diferentes
misturados quebram imports de forma sutil.

Regras que as três receitas do laboratório seguem e que você deve copiar:
- o overlay **loga um marcador** no boot (`[glm53-kpool-tail-seed-stride]`, `[overlay] …`) para você confirmar
  com `docker logs | grep`;
- sai com `already upstream` quando a imagem nova já tem a correção. Atualizar a imagem não pode quebrar o
  patch;
- o mesmo overlay é montado **nos dois ranks**;
- existe um teste de CPU para cada patch (`tests/test_<patch>.py`) que roda sem GPU (ver `scripts/ci-validate.sh`
  na receita DSV4).

**4.4 Teste numérico contra a referência HF.** Rodar o modelo inteiro no HF é impossível no Spark. A técnica é
comparar um **modelo truncado** com as mesmas N primeiras camadas nos dois lados:

1. **Escolha N = 2**: a camada 0 é GA com MLP densa e a camada 1 é SWA com MoE. As duas cobrem todos os tipos
   de bloco. Em BF16 isso ocupa ~13 GB no HF, porque são 256 experts de uma camada desquantizados.
2. **vLLM:** `--hf-overrides '{"num_hidden_layers": 2}' --enforce-eager`, sem speculator, TP=1 num nó e depois
   TP=2 entre os nós. O TP=2 é onde vivem G2 e G3. Precisa do guard do #58184, ou de um overlay equivalente,
   para ignorar o QKV das camadas 2–47. O texto gerado é lixo, e isso é esperado. O que se compara são as
   **logprobs**.
3. **HF:** `AutoModelForCausalLM.from_config(cfg_truncado, trust_remote_code=True, torch_dtype=bfloat16)`,
   `attn_implementation="eager"` (o modelo HF cai para eager quando há sinks), com um state dict
   **desquantizado** das camadas 0–1:
   - FP8 em blocos: `W = fp8.float() * scale_inv.repeat_interleave(128,0).repeat_interleave(128,1)[:rows,:cols]`,
     respeitando o pré-sharding do QKV: 4 chunks com escalas próprias, `ceil(linhas_do_chunk/128)` blocos de
     linha por chunk (27 na GA, 29 na SWA);
   - MXFP4: tabela E2M1 `[0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]`, escala `2**(e8m0-127)` a cada 32
     valores. A convenção nibble baixo = elemento par (OCP/gpt-oss) **precisa ser confirmada**: desquantize um
     expert e compare o GEMM com o resultado do kernel do vLLM para a mesma entrada.
4. **Métrica:** para 32 prompts de 512 tokens, KLD top-20 por posição e concordância top-1. Como referência
   [heurística, ajuste ao seu caso]: KLD média < 1e-3 e top-1 > 99% em TP1; em TP2 **o mesmo valor que em TP1**.
   Se o TP2 divergir do TP1, o problema está no sharding (G2/G3), não no kernel.
5. **Testes de kernel** já existentes, dentro da imagem, com o overlay montado:
   `pytest tests/kernels/attention/test_triton_unified_attention_diffkv.py` (é o teste que o #58128 estende para
   FP8; no GB10 os casos BF16 que exigem FA3/FA4 aparecem como *skipped*).

**Critério de saída.** Os overlays passam nos testes de CPU; paridade truncada TP1 = TP2 ≈ HF dentro do
limiar; o boot completo loga todos os marcadores; o G1 é provado pelo pool 2×; o G2 pela soma não nula do QKV.

## Passo 5 — Argumentos de lançamento

**Objetivo.** Um comando de head e um de worker em que **cada flag tem motivo** e o valor vem do `.env`.

**Worker (rank 1) primeiro, depois o head (rank 0).** É o mesmo padrão de `start.sh` do Qwen: o script é
renderizado num arquivo e enviado ao worker por `ssh bash -s`.

```bash
# comum aos dois nós (valores do .env)
COMMON_DOCKER=(--gpus all --network host --ipc host --shm-size 16g
  --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_AUTO_DETECT=0
  -e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE -e TP_SOCKET_IFNAME=$IFACE -e NCCL_DEBUG=WARN
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MARLIN_INPUT_DTYPE=${MARLIN_INPUT_DTYPE:-fp8}
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -v $MODEL_DIR:$MODEL_DIR:ro -v $HOME/.cache/vllm:/root/.cache/vllm
  -v $PWD/files/overlay/mimo_v2.py:$VPY/model_executor/models/mimo_v2.py:ro
  -v $PWD/files/overlay/triton_attn_diffkv.py:$VPY/v1/attention/backends/triton_attn_diffkv.py:ro
  -v $PWD/files/overlay/triton_unified_attention_diffkv.py:$VPY/v1/attention/ops/triton_unified_attention_diffkv.py:ro)

VLLM_ARGS=(serve $MODEL_DIR --served-model-name mimo-v2.6-flash-mopd --trust-remote-code
  --tensor-parallel-size 2 --distributed-executor-backend mp
  --nnodes 2 --master-addr $HEAD_IP --master-port 50000
  --max-model-len ${MAX_MODEL_LEN:-262144} --max-num-seqs ${MAX_NUM_SEQS:-8}
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-4096}
  --gpu-memory-utilization ${GMU:-0.85}            # ou --kv-cache-memory <bytes> quando conhecer a folga
  --kv-cache-dtype ${KV_CACHE_DTYPE:-fp8}          # só com os overlays G1; sem eles fica BF16 em silêncio
  --moe-backend marlin ${LINEAR_BACKEND:+--linear-backend $LINEAR_BACKEND}
  --load-format safetensors --safetensors-load-strategy lazy
  --enable-chunked-prefill --enable-prefix-caching --no-async-scheduling
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --cudagraph-capture-sizes 8 16 24 32 40 48 56 64
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$MODEL_DIR/dflash\",\"num_speculative_tokens\":${SPEC_K:-7},\"draft_tensor_parallel_size\":2}"
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice
  --generation-config auto --default-chat-template-kwargs '{"enable_thinking": false}'
  --mm-encoder-tp-mode data --limit-mm-per-prompt '{"image":4,"video":1,"audio":1}'
  --mm-processor-kwargs '{"max_pixels":2073600}' --mm-processor-cache-gb 1
  --enable-prompt-tokens-details)

# worker
docker run -d --name vllm-mimo "${COMMON_DOCKER[@]}" -e VLLM_HOST_IP=$WORKER_IP \
  --entrypoint vllm $IMAGE "${VLLM_ARGS[@]}" --node-rank 1 --headless
# head
docker run -d --name vllm-mimo "${COMMON_DOCKER[@]}" -e VLLM_HOST_IP=$HEAD_IP \
  --entrypoint vllm $IMAGE "${VLLM_ARGS[@]}" --node-rank 0 --host 0.0.0.0 --port 8888
```

Os nomes de flag seguem o vLLM das imagens e kits citados. **Rode `vllm serve --help=all` na imagem fixada** e
deixe o `start.sh` falhar se alguma flag não existir. Um exemplo de como isso aparece: o log sugere
`--kv-cache-memory=<bytes>`, enquanto a receita GLM usa `--kv-cache-memory-bytes`.

| Flag / env | Valor | Por quê (conceito) |
|---|---|---|
| `--trust-remote-code` | — | Config e processor são remote code (`auto_map`) |
| `--tensor-parallel-size 2 --nnodes 2 --node-rank N --master-addr/--master-port`, `--headless` no worker | — | TP=2 com uma GPU por nó ([Passo 3](#passo-3--topologia-tpeppp)); o worker não abre API |
| `--distributed-executor-backend mp` | — | Multiprocessing em vez de Ray: menos memória e é o que as três receitas usam |
| `--max-model-len` | 262144 no começo | Suba para 1M só depois do needle de 128K e do teste de OOM. Nos modelos híbridos, o número de "tokens" informado depende desse valor |
| `--max-num-seqs` | 8 | Admissão de requisições. Precisa combinar com as capture sizes |
| `--max-num-batched-tokens` (MNBT) | 4096 | Tamanho do chunk de prefill e teto do batch por passo: mais prefill por passo contra mais ativação e ITL pior para quem está em decode ([Passo 9](#passo-9--tuning-iterativo)) |
| `--gpu-memory-utilization` / `--kv-cache-memory` | 0,85 / bytes | [Passo 1](#passo-1--matemática-de-memória). Fixar o pool evita depender do profiling no UMA |
| `--kv-cache-dtype fp8` | fp8 | KV global de 11,5 para 5,76 KB/token. A escala unitária é exata para este checkpoint, que não traz escalas calibradas [medido: tonyd2wild] |
| `--moe-backend marlin` + `VLLM_MARLIN_INPUT_DTYPE=fp8` | — | Kernel MXFP4 que funciona no SM12x; FP8 nas ativações (W4A8) deu +15% de prefill no SM120 [medido: diffbot] ([a medir] no sm_121) |
| `VLLM_USE_DEEP_GEMM=0` | — | G5 |
| `--linear-backend triton` | se houver G6 | G6 |
| `--load-format safetensors --safetensors-load-strategy lazy` | — | Evita o G2. Carregadores mais rápidos (InstantTensor: 161 GB em ~40 s [medido: Plaaasma]) exigem o porte do #58142 |
| `--enable-chunked-prefill`, `--enable-prefix-caching` | — | Prefill em pedaços do tamanho do MNBT; reuso de prefixo em agentes (multi-turn) |
| `--no-async-scheduling` | — | G8 |
| `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}` | — | Graphs só no decode. O builder DiffKV com patch exige esse modo [medido: Plaaasma]; é o que o recipe oficial usa com DFlash |
| `--cudagraph-capture-sizes 8…64` | múltiplos de k+1 | Com k=7, cada sequência em decode tem 8 tokens (1 + 7 do verify). 8 sequências = 64. Tamanhos fora da lista caem em eager |
| `--speculative-config` DFlash | k=7, draft TP=2 | [Passo 6](#passo-6--decodificação-especulativa) |
| `--reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice` | — | Separa `<think>` e converte o XML compacto em `tool_calls` |
| `--generation-config auto` | — | Usa temperatura 1,0 e top_p 0,95 do checkpoint quando o cliente não manda nada. Evita as "tool-call storms" com decodificação gulosa |
| `--default-chat-template-kwargs {"enable_thinking": false}` | política | O thinking vem ligado no template. Desligar no servidor evita reasoning vazando em clientes ingênuos; quem quiser liga por requisição. A escolha é sua: documente-a |
| `--mm-encoder-tp-mode data` | — | Encoders replicados (1,7 GiB) em vez de fatiados: menos comunicação por imagem |
| `--limit-mm-per-prompt`, `--mm-processor-kwargs max_pixels`, `--mm-processor-cache-gb 1` | — | Limitam o pior caso multimodal. Na receita GLM, um vídeo decomposto em 33 imagens matou o engine por falta de memória |
| `--enable-prompt-tokens-details` | — | A resposta informa `cached_tokens`, o que ajuda a depurar o prefix cache |
| `--language-model-only` | lane só texto | Sem torres multimodais: ~1,6 GiB/rank a mais de KV e sem profiling multimodal |
| `--hf-overrides '{"moe_router_dtype":"bfloat16"}'` | **não precisa** | Já está no config do V2.6; os comandos antigos do V2.5 precisavam |
| `--kv-offloading-size` | **não use** | Um "tier de KV em CPU" no UMA é a mesma memória; só tem ganho em máquinas com RAM separada (diffbot/x86) |

**Critério de saída.** O boot sobe nos dois nós; os logs mostram `Using TRITON_ATTN_DIFFKV for attention.`, os
marcadores dos overlays, o `GPU KV cache size` esperado e as capture sizes; e o `.last_head_launch.sh`
renderizado fica salvo para auditoria, como faz a receita Qwen.

## Passo 6 — Decodificação especulativa

**Objetivo.** Escolher o speculator e o k que maximizam os tok/s **na sua carga**, medindo a aceitação.

**Conceito.** O drafter propõe k tokens e o alvo verifica todos numa passada. O ganho é
`tokens aceitos por passo / custo relativo do passo`. O custo do passo sobe com k (verify maior, mais KV do
drafter, capture sizes maiores). A aceitação depende do **tipo de texto**: altíssima em JSON, contagem e código,
baixa em prosa. Temperatura 1,0 reduz a aceitação em relação a 0.

**Opções no MiMo:** DFlash (k de 1 a 7; o bloco de 8 é o do treino) e MTP (k de 1 a 3). O ngram é experimental.

**Como medir a aceitação** (contadores Prometheus do vLLM V1):

```bash
m() { curl -s http://$HEAD_IP:8888/metrics | grep -E "^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total" ; }
m > before.txt; python3 bench/run_category.py --category code --n 20; m > after.txt
# tokens aceitos por passo = Δaccepted / Δdrafts ; taxa por token = Δaccepted / Δdraft_tokens
curl -s http://$HEAD_IP:8888/metrics | grep accepted_tokens_per_pos    # curva por posição (decai com a posição)
```

**Protocolo:**
1. Linha de base **sem spec**, por categoria (código, JSON, prosa, raciocínio, resumo) com C=1.
2. DFlash k ∈ {3, 5, 7}, e MTP k ∈ {1, 2, 3}, com o mesmo prompt set, em temperatura 0 **e** 1,0.
3. Repita em 32K e 128K de contexto: o custo do verify em contexto longo depende do G7.
4. Escolha pela **média ponderada da sua carga**. Um laboratório que atende agentes de código pesa código e
   JSON.

**Gabarito** [medido: tonyd2wild, vLLM TP2, DFlash k=7, FP8 KV, temperatura 0, C=1]: tokens aceitos por passo
(de 7): estruturado 6,85, código 5,23, matemática 5,00, JSON 4,00, raciocínio 2,62, prosa 1,23; tok/s por stream:
código 70,5, prosa 25,9, média 53,3. No TP4 (4 Sparks), k=4 foi **pior** que k=7 em todas as categorias em que o
drafter é bom e não trouxe ganho em prosa. No SGLang, o MTP ganhou em prosa (35,0 contra 25,8) e perdeu em
código [medido: MiaAI-Lab].

**Critério de saída.** Uma tabela aceitação × k × categoria × contexto; o speculator e o k escolhidos com
justificativa; o `SPEC_METHOD`/`SPEC_K` no `.env`, com `mtp` e `none` como rollback.

## Passo 7 — Validação

**Objetivo.** Provar que o servidor responde **certo**, não só rápido. Todo teste vira script em `tests/` e roda
depois de cada mudança.

| Teste | Como | Passa se |
|---|---|---|
| Vida | `curl -fsS $API/health`; `curl -s $API/v1/models \| jq '.data[0].max_model_len'` | 200; `max_model_len` = `.env` |
| Sanidade (G2) | "What is the capital of France? One word." com temperatura 0 | "Paris"; nada de lixo ou texto repetido |
| Reasoning parser | Requisição com `chat_template_kwargs: {enable_thinking: true}` | `reasoning_content` (ou `reasoning`, conforme a versão) preenchido; `content` sem `<think>` |
| Thinking off | Requisição sem kwargs (servidor com `enable_thinking: false`) | `content` com a resposta; reasoning vazio |
| Tool calling | 1 tool `get_weather`, `tool_choice: auto`, streaming e não streaming | `finish_reason: tool_calls`, argumentos em JSON válido; sem tool → `stop` |
| Tool calling estrito | `tool_choice: required` e nomeado | Documente o resultado; pode falhar até o #58019 |
| Multi-turn com tool result | Devolver `role: tool` e pedir continuação | Resposta usa o resultado; sem repetir a mesma chamada |
| Repetição de tool calls | Replay de uma sessão de agente capturada (ver tonyd2wild) | ≤ N chamadas por resposta; compare MOPD × RL |
| Chat template | `POST /tokenize` com `messages` + `add_generation_prompt` e comparação com `tokenizer.apply_chat_template` do HF | Tokens idênticos |
| Needle (camadas globais globais) | 16K, 32K e 128K; profundidades 0,1/0,5/0,9 | 100%. **Se falhar só acima de ~128 tokens, é vazamento de `sliding_window`** (#58128) |
| Qualidade canário | GSM8K com 200 questões, 5-shot, gulosa (`lm_eval --model local-chat-completions … --tasks gsm8k`) | Igual à **sua** linha de base, dentro de ±3 questões. Referências publicadas: diffbot 196/200 (98%); #58128 279–281/300 (~93%, com limite de 256 tokens de saída); #58142 ~0,97 (25-shot). O número absoluto depende do protocolo |
| KV FP8 efetivo | Compare `GPU KV cache size` com e sem `fp8` | ~2× |
| Multimodal | Imagem sintética gerada localmente (quadrado vermelho, texto "MIMO 42"), um clipe de vídeo, um áudio com uma senha falada | Descrição correta; áudio transcrito por `input_audio` e `audio_url` |
| Sob carga | 8 requisições longas concorrentes com spec | Nenhum caractere espúrio; `engine errors: 0` |
| OOM no pior caso | `max_num_seqs` × prompts no `max_model_len` | Sem crash; `MemAvailable` ≥ 4 GiB |

Exemplo de tool call:

```bash
curl -s $API/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"mimo-v2.6-flash-mopd","temperature":0,
 "messages":[{"role":"user","content":"Qual o tempo em Lisboa agora?"}],
 "tools":[{"type":"function","function":{"name":"get_weather","description":"Clima atual",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' \
 | jq '.choices[0] | {finish_reason, tool_calls: .message.tool_calls}'
```

**Critério de saída.** `tests/smoke.sh` verde e resultados anotados no README ("Verified end to end: …"), com
data e commit.

## Passo 8 — Benchmark

**Objetivo.** Números reproduzíveis de TTFT, TPOT e tok/s para a escada de concorrência e para o prefill, com
método publicado.

**Conceitos.** **TTFT** é o tempo até o primeiro token (fila + prefill). **TPOT/ITL** é o tempo entre tokens no
decode. **tok/s por stream** mede a experiência de quem usa; **tok/s agregado** mede a capacidade. O **prefill**
é limitado por compute e o **decode** por banda de memória. O teto de decode sem spec [estimativa] é:
`bytes por passo por rank ≈ 2,51 GB (8 experts ativos × 47 camadas, metade por rank) + 1,44 (qkv FP8) + 1,61
(o_proj BF16) + 0,63 (lm_head) + 0,2 (router, MLP densa) ≈ 6,4 GB`, o que dá **≤ 43 tok/s a 273 GB/s**. Com
`o_proj` em FP8 o total cai para ~5,6 GB e o teto sobe para ~49 tok/s. Na prática espere 50–65% disso e depois multiplique pela aceitação do spec.

**Regras de método:**
- Prompts **reais por categoria** para decode com spec. O dataset `random` do `vllm bench` gera tokens
  aleatórios, derruba a aceitação e **subestima** o DFlash.
- Prefill **frio**: um prefixo único por requisição (salt aleatório no começo) para não bater no prefix cache.
  Confirme com `cached_tokens = 0`.
- Aquecimento descartado (o primeiro request compila e faz autotune), mediana de ≥ 3–5 execuções, temperatura e
  thinking registrados.
- Escada de concorrência C ∈ {1, 2, 4, 8} (e 16 se o `max_num_seqs` permitir); escada de prefill de 2K, 8K, 32K,
  64K, 128K e 256K.
- Salve o `.env`, o digest da imagem, a revisão do modelo e os logs de boot junto do JSON do resultado
  (`results/AAAA-MM-DD-<tag>/`).

```bash
# Escada de concorrência (dataset customizado por categoria → JSONL de prompts)
for C in 1 2 4 8; do
  vllm bench serve --backend openai-chat --base-url http://$HEAD_IP:8888 --endpoint /v1/chat/completions \
    --model mimo-v2.6-flash-mopd --dataset-name custom --dataset-path bench/prompts_code.jsonl \
    --num-prompts $((C*8)) --max-concurrency $C --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir results/$(date +%F)-c$C
done
# Prefill frio: 1 requisição por tamanho, prefixo único, max_tokens=1 → prefill tok/s = prompt_tokens / TTFT
python3 bench/cold_prefill.py --sizes 2048 8192 32768 65536 131072 262144 --max-tokens 1
```

Confira os nomes de dataset e flags com `vllm bench serve --help` na imagem fixada. Como alternativa, a
ferramenta [sparkDash](https://github.com/MiaAI-Lab/sparkDash) do laboratório já traz as categorias
estruturado/prosa/código usadas nas receitas GLM e Qwen.

**Gabarito para 2× Spark** (para saber se você está no lugar certo):

| Métrica | vLLM TP2, DFlash k=7, FP8 KV [medido: tonyd2wild] | SGLang TP2 [medido: MiaAI-Lab / Smilehyu] |
|---|---|---|
| C1 por stream (média de categorias) | 53,3 tok/s; TTFT 0,37 s | EAGLE MTP: 31,8–46,8 agregado em 1 stream (MOPD) |
| C1 código / prosa | 70,5 / 25,9 | DFlash 69,8 / 25,8; EAGLE 46,0 / 35,0 |
| C6 agregado | 155,8 tok/s | C8: 98–226 por categoria (MOPD, EAGLE) |
| Prefill frio 2K / 32K / 128K / 248K | 1.947 / 1.425 / 937 / 656 tok/s | pico de 2.558 em 8K (RL); 2.311 em 16K (MOPD) |
| Pool de KV | 1,87M tokens @300K (GMU 0,90) | 2,55M (MOPD, 0,93) / 2,88M (RL, 0,963) |

**Critério de saída.** Tabelas de decode por categoria e C, de prefill frio e de aceitação, com método,
data, commit e digest. Nada de "≈" sem a marcação [estimativa].

## Passo 9 — Tuning iterativo

**Objetivo.** Melhorar um alvo declarado sem quebrar o resto, com uma regra de decisão escrita **antes** do
experimento.

**Regra de decisão** (declare-a no `docs/` antes de medir):
1. **Uma variável por vez**, no desenho A/B/A (baseline → mudança → baseline no mesmo boot, ou em boots
   pareados), com ≥ 3 execuções por braço, usando a mediana.
2. **Aceite** se o alvo melhora ≥ 3% nos dois lados do A/B/A **e** nenhuma métrica secundária piora mais de 2%
   **e** o canário de qualidade continua dentro do ruído (GSM8K-200 ± 3 questões; needle 100%; suíte de tools
   100%) **e** não há OOM no pior caso.
3. **Rejeite e registre** o resultado negativo no CHANGELOG, como o tonyd2wild fez com k=4 e MNBT 8192.
4. Mudanças que afetam memória (dtype do KV, GMU, MNBT) **refazem** o teste de OOM.

**Botões, hipóteses e o que já se sabe:**

| Botão | Hipótese | Troca esperada | Evidência existente |
|---|---|---|---|
| `KV_CACHE_DTYPE` fp8 × auto | 2× de pool | TTFT +4–15% em 16–128K; decode ≈ igual | #58128 (2× Spark); tonyd2wild: decode igual dentro do ruído |
| MNBT 2048 / 4096 / 8192 | Chunk maior = prefill mais rápido | Mais ativação (menos KV), ITL pior para quem está em decode com prefill concorrente | TP4: 8192 rejeitado (prefill pior em 8K/32K, pool −13%) [tonyd2wild]; #58128 usou 2048; diffbot 4096 |
| `max_num_seqs` 4 / 8 / 16 | Mais agregado | Menos tok/s por stream; capture sizes maiores (mais memória de graph) | C1→C6: 45,6→155,8 agregado, 53→31 por stream [tonyd2wild] |
| Capture sizes | Cobrir todo batch real | Memória de graph (~0,2–0,6 GiB) | Múltiplos de k+1 [Plaaasma] |
| `SPEC_K` e método | Ver [Passo 6](#passo-6--decodificação-especulativa) | — | — |
| W4A8 (`VLLM_MARLIN_INPUT_DTYPE=fp8`) × W4A16 | +prefill | Precisão das ativações | +15% de prefill, GSM8K 98,5% no SM120 [diffbot] |
| `o_proj` FP8 (#59083) | −0,8 GiB/rank, +14% no teto de decode [estimativa] | Erro de ~2,6% RMSE por tensor em casos parecidos (FP8_DENSE do Qwen) | [a medir] |
| EP on/off | Menos bytes por rank sob carga? | Desbalanceamento, all-to-all | [a medir] |
| GMU 0,80→0,88 ou `--kv-cache-memory` | Mais pool | Folga do host | SGLang: ganho marginal acima de 0,95 [MiaAI-Lab] |
| Agrupamento do KV híbrido (#58207 `--min-kv-cache-group-layers`) | +10% de pool | Mais grupos (metadados) | +10,57% [#58175] |
| Split-KV no verify (G7) | Decode em contexto longo | Nenhuma esperada | decode 87–94 → 156–175 tok/s no SM120 [diffbot] |
| Segmentos split-K (#58482) | Ocupação dos SMs em batch pequeno | — | Regra de SM120; o GB10 tem ~48 SMs [a medir] |
| Prefix cache e tamanho de bloco | TTFT em multi-turn | Alinhamento de blocos com SWA | Revisita de um prompt de 8K: TTFT 0,34 s (10,8×) no SGLang [Smilehyu] |

**Critério de saída.** Um `docs/tuning-AAAA-MM-DD.md` por rodada com a tabela de experimentos (aceitos e
rejeitados); o `.env.sample` atualizado; o CHANGELOG com o motivo de cada default.

## Passo 10 — Empacotamento

**Objetivo.** Outra pessoa, com outro par de Sparks, reproduz tudo com `cp .env.sample .env && ./start.sh`.

**Estrutura** (a mesma das três receitas do laboratório):

```
MiMo-V2.6-Flash-MOPD-vLLM-2x-DGX-Sparks/
├── README.md              # Status honesto, Quick start, .env reference (sample × live), KV budget medido,
│                          # Default runtime (docker inspect), Patches e por quê, Medições, Gotchas, Credits
├── CHANGELOG.md           # cada default com data, medição e motivo; resultados negativos também
├── LICENSE                # da receita; NOTICE para código vendorizado (licenças dos kits de origem)
├── .env.sample            # comentado; os valores do README vêm daqui
├── Dockerfile             # opcional: só deps de áudio/vídeo
├── download.sh            # hf download com revisão fixa + verify
├── verify-weights.py
├── start.sh               # doctor | download | sync | overlays | launch | --launch | --no-download | --nfs
├── stop.sh  status.sh  logs.sh  smoke.sh
├── files/
│   ├── orig/*.py + .sha256        # arquivos extraídos da imagem fixada
│   ├── patch_*.py                 # geradores fail-closed e idempotentes
│   ├── overlay/*.py               # saída (bind-mount nos 2 ranks)
│   └── evict_page_cache.py        # posix_fadvise(DONTNEED) do checkpoint antes do boot
├── tests/                 # test_<patch>.py (CPU), smoke, needle, tools, multimodal, paridade truncada
├── bench/                 # prompts por categoria, cold_prefill.py, run_ladder.sh
├── results/AAAA-MM-DD-*/  # JSON + .env + boot log + digest
└── docs/                  # triagem, memória, tuning, decisões
```

**Dockerfile** (só se precisar de áudio e vídeo. Os kits relatam que a imagem não tem `soundfile` e PyAV):

```dockerfile
FROM vllm/vllm-openai:mimo-v26-aarch64-cu130@sha256:<digest-fixado>
RUN pip freeze | grep -iE "^(torch|torchaudio|torchvision|numpy|triton|transformers)==" > /tmp/pins.txt && \
    pip install --no-cache-dir -c /tmp/pins.txt librosa soundfile av && \
    python3 -c "import librosa, soundfile, av; print('ok')"
LABEL org.opencontainers.image.source="https://github.com/<você>/MiMo-V2.6-Flash-MOPD-vLLM-2x-DGX-Sparks"
```

Os overlays ficam **fora** da imagem, em bind-mount. Assim você atualiza um patch sem rebuild e a imagem continua
igual ao upstream. Leve a imagem para o worker pelo link CX7: `docker save $IMAGE | ssh $WORKER docker load`.

**`.env.sample`** (trecho):

```bash
# --- cluster ---
HEAD_IP=10.0.0.1
WORKER_IP=10.0.0.2
WORKER_USER=
IFACE=enp1s0f0np0          # interface CX7 do head
WORKER_IFACE=              # vazio = IFACE (nós cruzados: veja o README do Qwen)
IB_HCA==rocep1s0f0         # "=" no início = match exato de um HCA
WORKER_IB_HCA=
IB_GID_INDEX=3
# --- modelo ---
MODEL_ID=XiaomiMiMo/MiMo-V2.6-Flash-MOPD
MODEL_REV=2479e2d0029eca9a34cc7e7f55a121925f81908e
MODEL_DIR=$HOME/models/MiMo-V2.6-Flash-MOPD
NFS_SHARE=false            # true = o worker monta o diretório do head em read-only (sem 2ª cópia)
IMAGE=vllm/vllm-openai:mimo-v26-aarch64-cu130
# --- serving ---
SERVED_MODEL_NAME=mimo-v2.6-flash-mopd
PORT=8888
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=4096
GMU=0.85
KV_CACHE_MEMORY=           # bytes; se definido, fixa o pool (recomendado depois do 1º boot)
KV_CACHE_DTYPE=fp8         # exige overlays G1; o start.sh recusa fp8 sem eles
SPEC_METHOD=dflash         # dflash | mtp | none
SPEC_K=7
MARLIN_INPUT_DTYPE=fp8     # vazio = W4A16
LINEAR_BACKEND=            # triton se aparecer G6
TEXT_ONLY=0                # 1 = --language-model-only
THINKING_DEFAULT=false
EXACT_QKV=1                # overlay G3
OPROJ_FP8=0                # lane experimental (#59083)
REQUIRE_IDLE_GPU=true
EVICT_PAGE_CACHE=true
EXTRA_VLLM_ARGS=
```

**`start.sh`: o que ele precisa fazer, na ordem** (copie a estrutura do `start.sh` da receita Qwen):
1. `doctor`: ssh ao worker, a imagem nos dois nós (e o **digest igual**), `ibv_devinfo`/RoCE ativo, GPU ociosa
   nos dois nós, `MemAvailable`, espaço em disco, `earlyoom` desligado;
2. `download` + `verify` no head; `sync` (rsync para o worker **ou** NFS read-only);
3. extrair os arquivos da imagem, conferir o sha256 contra `files/orig/*.sha256` e gerar os overlays; **abortar**
   se uma âncora falhar;
4. validar combinações proibidas antes de subir (`KV_CACHE_DTYPE=fp8` sem overlay; `SPEC_K` > 7 com DFlash ou
   > 3 com MTP; capture sizes que não são múltiplos de k+1; flags inexistentes na imagem);
5. liberar o page cache do checkpoint; subir o worker; subir o head; esperar o `/health` com timeout de ~20 min
   (cold start de ~11 min com mmap [medido: tonyd2wild]);
6. imprimir as linhas de KV do log e rodar o `smoke.sh`.

**Documentação.** A seção "Status" do README em formato de tabela, só com medições e com "Not measured"
explícito. Uma seção "Why the overlay exists" para cada patch (sintoma, causa, PR upstream, como remover quando
o upstream absorver). Créditos para cada kit de onde veio uma ideia (diffbot, tonyd2wild, Plaaasma, MiaAI-Lab,
local-inference-lab) e para os autores dos PRs.

**Critério de saída ("pronto").** Um segundo par de Sparks reproduz os números do README com diferença ≤ 5–10%,
seguindo só o README.

---

# Apêndice A — Glossário de conceitos

- **Memória unificada (UMA):** CPU e GPU usam o mesmo pool LPDDR5X. O page cache e os processos do host competem
  com o vLLM. Carregar o checkpoint enche o page cache, e isso pode causar OOM mesmo com a GPU "ociosa".
- **GMU / `--kv-cache-memory`:** fração do total visível que o vLLM orça, ou o tamanho fixo do pool de KV. No
  UMA, fixar o pool é mais previsível.
- **KV cache e alocador híbrido:** as camadas globais guardam K/V de todo o contexto; as SWA só da janela. O
  vLLM agrupa camadas com o mesmo tamanho de página, e o enchimento de grupo desperdiça espaço (#58175/#58207).
- **SWA (sliding window attention):** cada token olha só para os últimos W tokens (W=128 aqui). O custo por
  requisição é fixo.
- **DiffKV:** K e V com dimensões de head diferentes (192/128). Exige kernels próprios.
- **Attention sink:** logit aprendido extra no softmax que dá à atenção um lugar para "descarregar" massa.
  Estabiliza janelas curtas.
- **RoPE parcial:** rotação posicional aplicada a uma fração das dimensões (1/3 aqui). O resto não carrega
  posição.
- **MoE top-k, `noaux_tc`, sigmoid:** o roteador escolhe 8 de 256 experts. O bias de correção equilibra a carga
  sem perda auxiliar.
- **MXFP4 / NVFP4 / FP8 em blocos:** MXFP4 = E2M1 + escala E8M0 a cada 32 (4,25 bits). NVFP4 = E2M1 + escala
  FP8 a cada 16 + escala global (~4,5 bits). FP8 em blocos = E4M3 + escala F32 por bloco 128×128.
- **W4A16 / W4A8 / W4A4:** bits de peso / bits de ativação no GEMM. Menos bits de ativação significa mais
  throughput de compute e mais erro.
- **Pré-sharding:** o checkpoint salvo já fatiado para um TP específico (aqui, `tp_size: 4`). Outro TP exige
  reorganizar e, às vezes, requantizar.
- **TP / EP / PP:** ver o [Passo 3](#passo-3--topologia-tpeppp).
- **Speculative decoding (MTP, DFlash, EAGLE):** um drafter barato propõe k tokens e o alvo verifica numa
  passada. O DFlash rascunha um bloco inteiro com atenção bidirecional usando hidden states do alvo; o MTP usa
  camadas extras treinadas junto com o modelo.
- **Aceitação:** tokens aceitos por passo e taxa por posição. É o que decide o k.
- **Chunked prefill / MNBT:** o prefill é quebrado em pedaços de até MNBT tokens, intercalados com o decode.
- **CUDA graphs (`FULL_DECODE_ONLY`, capture sizes):** passos de decode gravados e reproduzidos sem overhead de
  lançamento de kernels. Só servem para os tamanhos de batch capturados.
- **Async scheduling:** sobrepõe o agendamento do passo N+1 à execução do passo N. Aqui foi desligado por bug
  com spec decode.
- **KLD / perplexidade:** divergência entre as distribuições de próximo token de dois modelos; PPL é a
  exponencial da log-verossimilhança média. Use KLD para comparar quantizações.
- **Overlay:** substituir um arquivo `.py` do pacote instalado por bind-mount, gerado a partir do arquivo da
  própria imagem com patch fail-closed.
- **Paridade truncada:** comparar motor e referência com as mesmas N primeiras camadas para isolar erros de
  kernel e de carregamento.
- **TTFT / TPOT / ITL / tok/s agregado:** ver o [Passo 8](#passo-8--benchmark).
- **A/B/A:** medir baseline, mudança e baseline de novo para separar o efeito da deriva (temperatura, page
  cache, autotune).

# Apêndice B — Desafio: Naive-N0.5-Flash, um modelo que o vLLM ainda não conhece

[NaiveAI/Naive-N0.5-Flash](https://huggingface.co/NaiveAI/Naive-N0.5-Flash) (2026-09-27, MIT): 309B / 15,5B
ativos, **derivado da base MiMo-V2.5**. Tem a mesma espinha dorsal (48 camadas, hidden 4096, 256 experts top-8,
4 KV heads, vocabulário 152.576), mas **troca as camadas globais por DeepSeek Sparse Attention (DSA)** num
layout ~5:1 SWA:DSA, sem nenhuma atenção densa global. Os pesos são BF16 (617,8 GB) e FP8 (~310 GB), com um
`Naive-N0.5-Flash-FP8-Draft`; existem NVFP4 e EXL3 da comunidade com um dia de vida.

Refaça os passos com estas diferenças:
- **Passo 0:** `architectures: NaiveN05FlashForCausalLM`, `model_type: naive_n05_flash`, remote code em
  `modeling_naive_n05_flash.py`. **Não há entrada no registry do vLLM.**
- **Passo 1:** não há KV global crescendo com o contexto no sentido clássico. O DSA guarda KV + chaves do
  indexador e lê só o top-k. Refaça a fórmula por camada DSA (KV heads × dims × bytes + chave do indexador) a
  partir do config.
- **Passo 2:** nem o FP8 cabe em 2 Sparks. Você precisa **quantizar os experts** (NVFP4 com `llm-compressor` ou
  ModelOpt, calibrado) e medir a KLD contra o FP8 oficial rodando em ≥ 3 nós, ou contra a referência HF truncada.
- **Passo 4:** escrever o modelo no vLLM. O caminho mais curto é um **plugin out-of-tree** que registra a
  arquitetura sem fork:

  ```python
  # naive_plugin/__init__.py  — instalado com pip install -e . ; entry point "vllm.general_plugins"
  def register():
      from vllm import ModelRegistry
      ModelRegistry.register_model("NaiveN05FlashForCausalLM",
                                   "naive_plugin.model:NaiveN05FlashForCausalLM")
  ```

  Na implementação: reaproveite `mimo_v2.py` (MoE, SWA com sinks, loader do QKV) e troque as camadas globais
  por um bloco DSA. O DSA do vLLM existe para MLA (DeepSeek-V3.2 e V4); **DSA com GQA é trabalho novo**, e no
  sm_121 os backends de sparse-MLA também não servem diretamente (veja o overlay NoPE→GLM_NSA da receita GLM).
  Valide com paridade truncada (N=6, para incluir pelo menos uma camada DSA) contra `modeling_naive_n05_flash.py`.
- **Passo 6:** confira se o `FP8-Draft` segue um formato que o vLLM já conhece (EAGLE3/DFlash) ou se também
  precisa de registro.

É o exercício "nível 2": o Passo 4 deixa de ser overlay e passa a ser uma implementação completa.

# Apêndice C — Rodar o mesmo modelo no Mac com MLX e contrato OpenAI

Se você quer o mesmo modelo **local no Mac**, atrás do mesmo contrato `/v1/chat/completions`, para que o cliente
só troque a `base_url`:

- **Memória:** as conversões MLX pesam 150–172 GB e carregam ~164–170 GB. É preciso um Mac com **≥ 192 GB**
  unificados; na prática, Mac Studio M3 Ultra de 256 GB ou M5 Ultra (até 512 GB).
  [medido: Vontra, M3 Ultra 256 GB] decode de 59,4 tok/s, prompt processing de 563 tok/s em 2K, pico de
  164,3 GB. O decode por stream é **maior** que no par de Sparks sem spec porque a banda do M3 Ultra (~800 GB/s)
  é ~3× a do GB10.
- **Motor:** o suporte ao MiMo-V2 está no **PR #1219 do mlx-lm** (branch `kernelpool/mlx-lm@add-mimo-v2`),
  ainda sem merge quando esta pesquisa foi feita:

  ```bash
  python3 -m venv ~/.venvs/mlx && source ~/.venvs/mlx/bin/activate
  pip install "git+https://github.com/kernelpool/mlx-lm.git@add-mimo-v2"
  mlx_lm.server --model mlx-community/MiMo-V2.6-Flash-RL-mxfp4-q8 --host 127.0.0.1 --port 8080
  # contrato OpenAI: POST http://127.0.0.1:8080/v1/chat/completions  (mesmo payload do vLLM)
  ```

- **Diferenças de contrato em relação à receita vLLM** (documente no cliente):
  - A conversão `mlx-community` é **só texto**: sem visão e áudio, e sem MTP ou DFlash em execução. A Vontra
    empacota esses componentes, mas eles não são executados.
  - O thinking vem ligado no template; passe `enable_thinking=False` via `chat_template_kwargs` quando o servidor
    aceitar, ou faça isso no cliente.
  - Os tool calls usam o formato Qwen3-Coder, tratado pelo parser `qwen3_coder` do mlx-lm (segundo o card).
    Teste com a mesma suíte do [Passo 7](#passo-7--validação).
  - Ainda não há conversão MLX do **MOPD** publicada pela `mlx-community` (existem as do RL). Converter o MOPD
    com `mlx_lm.convert` do mesmo branch é o caminho, mas não foi testado aqui [a medir].
- **Macs com 128 GB:** só com poda de experts (REAP50 MLX, ~95 GB), com perda de qualidade [a medir]. Para
  esses Macs, um modelo menor costuma ser a escolha mais honesta.

# Fontes

Modelo e variantes
- Card do MOPD: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD
- Card do RL e relatório técnico: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL ·
  https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf
- Raw config: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/raw/main/config.json ·
  dflash: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/raw/main/dflash/config.json
- Coleção: https://huggingface.co/collections/XiaomiMiMo/mimo-v26 · Wikipedia (datas):
  https://en.wikipedia.org/wiki/Xiaomi_MiMo
- Recipe vLLM oficial: https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL ·
  V2.5: https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5
- Quants: https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF ·
  https://huggingface.co/benthecarman/MiMo-V2.6-Flash-RL-exl3 ·
  https://huggingface.co/ProCreations/MiMo-V2.6-Flash-RL-NVFP4 ·
  https://huggingface.co/mlx-community/MiMo-V2.6-Flash-RL-mxfp4-q8 ·
  https://huggingface.co/Vontra/MiMo-V2.6-Flash-RL-MLX-4bit-MTP ·
  https://huggingface.co/patrickbdevaney/MiMo-V2.6-Flash-REAP50

vLLM
- Release v0.30.0: https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- Imagens: https://hub.docker.com/r/vllm/vllm-openai/tags?name=mimo-v26
- PRs: #57784, #57508, #59053, #58142, #58184, #58128, #58176, #58175, #58207, #58141, #59085, #59052, #58482,
  #58798, #59083, #58019, #57804, #58235, #58938 em https://github.com/vllm-project/vllm/pull/<n>
- Issues: #59054, #58638, #46669, #56077 em https://github.com/vllm-project/vllm/issues/<n>
- DeepGEMM#417: https://github.com/deepseek-ai/DeepGEMM/issues/417
- `mimo_v2.py` (main): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mimo_v2.py

Receitas da comunidade (gabarito)
- MiaAI-Lab (SGLang, RL): https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks
- Smilehyu (SGLang, MOPD): https://github.com/Smilehyu/MiMo-V2.6-Flash-MOPD-2x-DGX-Sparks
- tonyd2wild (vLLM, RL): https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe
- Plaaasma (vLLM customizado): https://github.com/Plaaasma/mimo26-flash-dual-dgx-spark
- diffbot (SM120, W4A8 + FP8 KV): https://huggingface.co/diffbot/MiMo-V2.6-Flash-MOPD-FP8KV-W4A8-2x-RTX-PRO-6000
- SGLang #40448 (MXFP4 do MiMo-V2.6), citado pelos kits SGLang

Outros candidatos
- https://huggingface.co/NaiveAI/Naive-N0.5-Flash · https://huggingface.co/IQuestLab/IQuest-Q1 ·
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash · https://huggingface.co/zai-org/GLM-5.3 ·
  https://huggingface.co/TypeSafeAI/Step-5-Preview-BF16
- Panoramas de setembro de 2026: https://www.thundercompute.com/blog/best-open-source-llms ·
  https://dev.to/klukyanov/the-local-llm-weight-classes-september-2026-what-actually-fits-on-your-machine-128d

MLX
- mlx-lm PR #1219: https://github.com/ml-explore/mlx-lm/pull/1219

Receitas do laboratório usadas como padrão (neste workspace): `repos/Qwen3.8-Flash-Next-Dual-DGX-Sparks`,
`repos/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, `repos/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`.

**Incertezas declaradas:** (1) nenhuma medição do checkpoint MOPD em vLLM no GB10 foi publicada; os números de
vLLM são do RL, que tem config e drafter idênticos mas pesos diferentes. (2) Os nomes exatos de algumas flags
(`--linear-backend`, `--kv-cache-memory` × `--kv-cache-memory-bytes`, datasets do `vllm bench`) variam entre
builds: confirme com `--help` na imagem fixada. (3) A ordem dos nibbles do MXFP4 na referência HF precisa ser
verificada empiricamente. (4) Os limiares de KLD e de paridade são heurísticas de ensino, não padrões oficiais.
(5) O estado dos PRs é de 2026-09-28 e muda rápido: rode o Passo 4.2 de novo antes de cada release da receita.

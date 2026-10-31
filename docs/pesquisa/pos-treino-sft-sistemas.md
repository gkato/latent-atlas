# Pós-treino I — fundamentos, SFT, PEFT e sistemas de treino

Nota de pesquisa do **Módulo 2 (Pós-treino)** do Latent Atlas. Data de referência: **2026-10-01**.

**Escopo desta nota.** (1) O que é pós-treino, a passagem base → instruct → raciocínio, o pipeline SFT → preferência → RL, on-policy vs off-policy, a âncora de KL num modelo de referência e uma taxonomia completa dos métodos. (2) A matemática de sistemas que um praticante precisa para treinar: memória por parâmetro, precisão mista, BF16/FP8/NVFP4, ativações e checkpointing, ZeRO 1/2/3 e FSDP, TP/PP/EP/CP no treino, FLOPs e tempo, otimizadores (AdamW, Muon), schedules de LR, packing e comprimento de sequência. (3) PEFT: LoRA, QLoRA, DoRA, LoRA em experts de MoE, merge de adaptadores e serviço multi-LoRA no vLLM. (4) SFT em profundidade: perda, máscara, chat templates, dados, sintéticos e destilação, RFT/STaR, destilação on-policy, **MOPD** (o que significa no `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`), esquecimento catastrófico e avaliação. (5) Viabilidade prática em 2× DGX Spark (GB10, 128 GB cada) e num Mac M4 24 GB com MLX, com os frameworks de 2026 e comandos exatos.

Os métodos de preferência (DPO, IPO, KTO, SimPO, ORPO) e de RL (RLHF-PPO, GRPO, AWR, CAI, self-play, debate) aparecem aqui como taxonomia, fórmula-resumo e ligação com SFT. O aprofundamento deles fica nas notas irmãs do módulo.

**Convenções.** `[fonte Sxx]` = verificado em fonte primária (lista com URLs na §10). `[cálculo próprio]` = conta feita aqui com números públicos (script na §8). `[interpretação]` = inferência minha a partir das fontes. `[não verificado]` = plausível, mas sem confirmação primária nesta pesquisa. GB = 10⁹ B, GiB = 2³⁰ B. Identificadores, flags, código e fórmulas ficam como no original. As notas do Módulo 1 citadas: `arch-qwen.md`, `arch-ds.md`, `arch-glm.md`, `glm-launch.md`, `qwen-launch.md`, `ds-launch.md`, `mlx-copilot.md`, `vllm-ref.md`.

---

## 0. Resumo executivo

1. **Pós-treino** é tudo o que se faz depois do pré-treino de próximo token para transformar um *modelo base* (um completador de texto) num assistente (*instruct*) e, desde 2024–25, num *modelo de raciocínio*. Pipeline típico: SFT → otimização de preferência → RL (RLHF e/ou RL com recompensas verificáveis), com destilação para modelos menores ou para fundir capacidades. Em 2026 o estado da arte industrial inclui **SFT → professores especializados por RL → MOPD** (Xiaomi MiMo) `[fonte S2, S3]`.
2. **SFT** é máxima verossimilhança (entropia cruzada do próximo token) sobre demonstrações, com a perda calculada só nos tokens do assistente. Equivale a minimizar o KL *direto* `KL(p_dados ‖ π_θ)` e é **off-policy**: o modelo aprende com prefixos que não gerou (viés de exposição).
3. **On-policy** = os dados vêm da política atual; **off-policy** = vêm de outra política (dataset, professor, versão antiga). A âncora de KL no modelo de referência (`β·KL(π_θ‖π_ref)`) é o que impede o RL de "fugir" do SFT; a solução ótima do objetivo RLHF é `π* ∝ π_ref·exp(r/β)`, que é de onde sai o DPO.
4. **Memória do treino completo** (full fine-tuning) com AdamW em precisão mista clássica: **16 B/parâmetro** (2 de pesos BF16 + 2 de gradientes + 4 de pesos mestres FP32 + 4 + 4 dos momentos do Adam) `[fonte S35]`. Um 8B pede ≈128,5 GB (≈119,7 GiB) só de estado: **não cabe num Spark**. São 121,69 GiB visíveis, ~100–110 GiB utilizáveis na prática, e não sobra espaço para SO e ativações. Cabe em 2 Sparks com FSDP/ZeRO-3 (≈64,2 GB/nó) `[cálculo próprio]`. BF16 puro + AdamW cai para 8 B/param, mas perde atualizações pequenas: com `w = 0,02` e passo `1e-5`, o peso **não muda** em BF16 `[cálculo próprio]`.
5. **Muon** (momento + ortogonalização Newton–Schulz das atualizações de matrizes 2D) guarda **um** buffer em vez de dois: 12 B/param no regime misto. Qwen3.8 e DeepSeek-V4 foram pré-treinados com Muon (notas do Módulo 1). O Moonlight mostra que trocar de otimizador entre pré-treino e SFT tira a vantagem do Muon `[fonte S33]`.
6. **LoRA**: `W' = W + (α/r)·B·A`, com `B = 0` no início. Com o LoRA em todas as camadas lineares (principalmente MLP/MoE), LR ≈ **10×** o do full FT, ~⅔ dos FLOPs por passo e capacidade suficiente para a maioria dos cenários de pós-treino; em RL, LoRA de **posto 1** já iguala o full FT `[fonte S5]`. Llama-3.1-8B, r=16, all-linear = **41,9 M** parâmetros treináveis (0,52 %) `[cálculo próprio]`.
7. **QLoRA**: base congelada em NF4 + quantização dupla (4,127 bits/param), adaptadores em BF16/FP32, otimizadores paginados `[fonte S9]`. Llama-3.3-70B em QLoRA ≈ 39,5 GB de pesos + 3,3 GB de estado do LoRA r=16 + ativações → **cabe num único Spark**; o playbook NeMo da NVIDIA roda a receita 70B QLoRA num container limitado a 64 GB `[fonte S23; cálculo próprio]`.
8. **MOPD = Multi-Teacher On-Policy Distillation.** O aluno gera as próprias trajetórias; um professor por domínio (treinado por RL a partir do mesmo SFT) dá a cada token a vantagem `log π_prof(y_t) − log π_θ(y_t)` (KL reverso por token), opcionalmente somada a uma vantagem de recompensa de resultado (ORM). O `MiMo-V2.6-Flash-MOPD` é o checkpoint `-RL` passado por **MOPD2** ("Multi-Prefix Multi-Teacher On-Policy Distillation" no relatório técnico do V2.6): professores mixRL + professores SFT, com três fluxos (MOPD padrão, Teacher-Prefix OPD e SFT-Prefix OPD). Entre outras coisas, o MOPD2 corrigiu a repetição de tool calls com um professor treinado em 12 passos (~7.000 exemplos) `[fonte S1, S2, S3, S4, S76]`.
9. **Esquecimento**: o quanto se esquece acompanha o KL entre o modelo ajustado e o base *na distribuição da nova tarefa*; RL on-policy esquece menos que SFT porque tende a soluções de KL mínimo (RL's Razor) `[fonte S46]`. LoRA "aprende menos e esquece menos" `[fonte S47]`. Destilação on-policy recupera comportamento perdido (IF-eval 79 % → 83 % após mid-training) `[fonte S6]`.
10. **Viabilidade (2× Spark)**: full FT até ~3–4B num Spark e ~8B em 2 Sparks (FSDP); LoRA BF16 até ~30–40B por nó e 70B em 2 nós (FSDP); QLoRA 70B num nó; MoE grandes (GLM-5.3-Flash 320B, DeepSeek-V4-Flash 284B, Qwen3.8-Flash-Next 125,7B) **não** são viáveis em full FT nem em LoRA BF16; QLoRA/NVFP4-LoRA só como experimento e dependente de suporte de framework à arquitetura `[cálculo próprio; interpretação]`.
11. **Mac M4 24 GB** (MLX recomenda ~19,07 GB de working set, medido no Módulo 1): QLoRA via `mlx_lm.lora` até ~14B 4-bit com batch 1 e `--grad-checkpoint`; full FT até ~1,7B `[cálculo próprio]`.
12. **Ecossistema 2026**: TRL 1.x (estável desde 2026-03-31; `PPOTrainer` removido na 1.13; `DistillationTrainer` estável; `AsyncDistillationTrainer` experimental com roteamento multi-professor no estilo MOPD), Unsloth (kernels MoE), Axolotl (EP, NVFP4 MoE LoRA), LLaMA-Factory, NeMo AutoModel (receitas `_rtx_spark`), verl e OpenRLHF (RL), mlx-lm (Mac). O **torchtune parou de ser desenvolvido em 2025-07-15** `[fonte S18, S19, S26, S27, S29]`.
13. **Alerta de medição**: os "picos de tokens/s" de fine-tuning publicados pela NVIDIA para o Spark (82.739 tok/s 3B full; 53.657 tok/s 8B LoRA; 5.079 tok/s 70B QLoRA) exigiriam 1,4–1,7 PFLOP/s pela contabilidade 6N/4N, ~12× o pico BF16 denso (~125 TFLOPS) que o Módulo 1 usa para o GB10. A definição da métrica não foi publicada; não use esses números para planejar `[fonte S24; cálculo próprio]`.

### 0.1 Versões do ecossistema em 2026-10-01 (PyPI)

| Pacote | Versão | Publicada | Observação |
|---|---|---|---|
| `trl` | 1.14.1 | 2026-09-29 | v1.0 em 2026-03-31; v1.13 removeu `PPOTrainer`; v1.11 trocou o servidor FastAPI pelo `vllm serve` nativo e trouxe `AsyncDistillationTrainer` (MOPD) `[fonte S18, S19]` |
| `peft` | 0.21.2 | 2026-10-01 | `target_parameters` para experts como `nn.Parameter` `[fonte S12]` |
| `transformers` | 5.18.0 | 2026-09-30 | v5 infere o dtype do config no `from_pretrained` (o TRL não: ver §2.1) |
| `unsloth` | 2026.9.14 | 2026-10-01 | kernels MoE (Triton grouped-GEMM + LoRA) `[fonte S26]` |
| `axolotl` | 0.20.0 | 2026-09-30 | EP (2026/06), NVFP4 MoE LoRA (2026/07) `[fonte S27]` |
| `llamafactory` | 0.9.5 | 2026-05-30 | playbook oficial no Spark `[fonte S23]` |
| `verl` | 0.9.1 | 2026-09-20 | RL (FSDP/FSDP2/Megatron; vLLM/SGLang) `[fonte S30]` |
| `openrlhf` | 0.11.2 | 2026-09-14 | RL com Ray + vLLM `[fonte S31]` |
| `bitsandbytes` | 0.50.2 | 2026-08-27 | NF4/8-bit Adam |
| `nemo-automodel` | 0.6.0 | 2026-08-26 | container `nvcr.io/nvidia/nemo-automodel:26.08` no playbook do Spark `[fonte S23]` |
| `mlx-lm` | 0.32.0 | 2026-10-01 | `mlx_lm.lora` com `lora`/`dora`/`full` e otimizador `muon` `[fonte S20]` |
| `torchtune` | — | — | desenvolvimento ativo encerrado em 2025-07-15; o sucessor de RL (`torchforge`) teve o desenvolvimento pausado e o treino de LLMs do PyTorch foi consolidado no `torchtitan` `[fonte S29]` |

---

## 1. O que é pós-treino

### 1.1 Base → instruct → raciocínio

**Modelo base.** Pré-treinado para prever o próximo token em trilhões de tokens: o MiMo-V2-Flash usou 27T tokens em FP8 misto `[fonte S2]`; o DeepSeek-V4 passou de 32T (`arch-ds.md`). Um modelo base completa texto: dado "Qual a capital da França?", pode continuar com outra pergunta, porque é isso que aparece em listas de exercícios na web. Ele sabe muito, mas não tem *formato* de conversa, não sabe quando parar, não segue instruções de forma confiável e não tem política de segurança.

**Modelo instruct (assistente).** Recebe um formato de conversa (chat template com papéis `system`/`user`/`assistant` e tokens de fim de turno), aprende a seguir instruções, a usar ferramentas e a recusar o que deve recusar. O marco é o InstructGPT (2022): SFT com ~13 mil prompts, modelo de recompensa com ~33 mil e PPO com ~31 mil. O InstructGPT de 1,3B foi preferido pelos avaliadores ao GPT-3 de 175B `[fonte S49]`. A "hipótese do alinhamento superficial" do LIMA diz que quase todo o conhecimento vem do pré-treino e que o pós-treino ensina sobretudo *formato e estilo*: com 1.000 exemplos curados num LLaMA-65B, as respostas do LIMA foram equivalentes ou preferidas às do GPT-4 em 43 % dos casos `[fonte S38]`.

**Modelo de raciocínio.** Gera uma cadeia de pensamento longa (`<think>…</think>`) antes da resposta e é treinado com RL sobre tarefas de resposta verificável (matemática, código, instruções checáveis). Marcos:
- **DeepSeek-R1-Zero**: RL (GRPO) direto sobre o base, sem SFT. **DeepSeek-R1**: SFT de "cold start" com milhares de CoTs longos → RL de raciocínio → *rejection sampling* para gerar ~800 mil exemplos (~600 mil de raciocínio + ~200 mil gerais) → SFT → RL geral. Os modelos `R1-Distill-*` (Qwen/Llama) são **só SFT** nesses 800 mil exemplos `[fonte S50]`.
- **Qwen3** (2025): quatro estágios para os modelos de ponta (cold start de CoT longo → RL de raciocínio com GRPO em 3.995 pares pergunta-verificador → fusão dos modos thinking/não-thinking por SFT → RL geral em 20+ tarefas). Os modelos pequenos recebem destilação "strong-to-weak": off-policy e depois **on-policy** `[fonte S7]`.
- **MiMo-V2-Flash** (309B total / 15B ativos): SFT com "milhões" de amostras → RL especializado por domínio (agêntico e não agêntico) → **MOPD** `[fonte S2]`.

A transição base → instruct → raciocínio não troca a arquitetura: os pesos são os mesmos tensores, ajustados. Por isso tudo o que o Módulo 1 ensinou sobre servir (quantização, KV, especulação) vale para qualquer checkpoint pós-treinado. E o pós-treino deixa marcas que o servidor precisa respeitar: chat template, parser de raciocínio, `generation_config` com a amostragem recomendada (o Qwen3.8 pede `temperature 1.0, top_p 0.95, top_k 20` no modo thinking, `qwen-launch.md`/`arch-qwen.md`) e tokens de fim (`eos_token_id [248046, 248044]` = `<|im_end|>` e `<|endoftext|>`).

### 1.2 O pipeline canônico e suas variantes

| Estágio | Dado | Perda / sinal | On/off-policy | O que muda no modelo | Custo relativo |
|---|---|---|---|---|---|
| Mid-training / pré-treino continuado (opcional) | texto de domínio, contexto longo, código | próximo token (todos os tokens) | off | conhecimento novo, contexto longo | alto (muitos tokens) |
| **SFT** | demonstrações (prompt, resposta) | entropia cruzada só no assistente | off | formato, estilo, uso de ferramentas, "ativa" capacidades latentes | baixo–médio |
| Otimização de preferência offline (DPO, IPO, KTO, SimPO, ORPO) | pares (escolhida, rejeitada) ou rótulo bom/ruim | perda logística/quadrática sobre razões de log-prob vs referência | off (ou "semi-on" se os pares vêm do próprio modelo) | preferências finas, tom, segurança, comprimento | baixo |
| RLHF clássico | prompts + modelo de recompensa treinado com pares humanos | PPO com KL ao SFT | on | alinhamento a preferências | alto (4 modelos na memória: política, referência, RM, crítico) |
| RL com recompensa verificável (RLVR) | prompts com verificador (resposta, testes) | GRPO/RLOO/PPO; recompensa 0/1 | on | raciocínio, agência, exploração | alto (geração domina) |
| Destilação off-policy | saídas de um professor | SFT nas saídas (ou KL nos logits) | off | transfere capacidade do professor | baixo–médio |
| Destilação on-policy / MOPD | prompts; professor dá log-probs nos tokens do aluno | KL reverso por token (+ ORM) | on | fecha a distância aluno-professor com sinal denso | médio (≈ RL sem exploração cega) |
| Rodadas iterativas | dados regerados pelo modelo da rodada anterior | SFT + RS + DPO | semi-on | refino incremental | — |

O Tülu 3 é a referência aberta mais documentada da sequência SFT → DPO → RLVR: SFT com uma mistura de 939.344 prompts, DPO normalizado por comprimento em 354.192 pares e RLVR em tarefas verificáveis `[fonte S42]`. O relatório do Llama 3 descreve várias rodadas de SFT + rejection sampling + DPO `[não verificado nesta pesquisa: número exato de rodadas]`.

### 1.3 On-policy vs off-policy

Seja `μ` a política que **gerou** os dados e `π_θ` a política que está sendo **treinada**.
- **On-policy**: `μ = π_θ` (ou muito próxima). O modelo aprende com prefixos que ele mesmo produz, os mesmos que verá na inferência. Exemplos: PPO, GRPO, RLOO, destilação on-policy, MOPD.
- **Off-policy**: `μ ≠ π_θ`. Dados de humanos, de um professor ou de uma versão antiga. Exemplos: SFT, destilação off-policy, DPO offline, AWR (sobre dados de uma política de comportamento).

**Viés de exposição.** No SFT, o modelo é treinado com *teacher forcing*: em cada posição, o prefixo é o do dataset. Na geração, o prefixo é o dele próprio. Um desvio pequeno o leva a estados que nunca viu e os erros se acumulam. Métodos on-policy treinam exatamente nos estados que o modelo visita. Essa é a motivação do GKD e da destilação on-policy `[fonte S8, S6]`.

**"On-policy" na prática é aproximado.** Em frameworks de RL modernos as trajetórias são geradas por um motor de inferência (vLLM/SGLang) e o gradiente é calculado no motor de treino (FSDP/Megatron). Kernels, precisão (rollout em FP8, por exemplo) e ordem de redução diferentes fazem `μ_θ ≠ π_θ` mesmo com os mesmos pesos, e a geração assíncrona adiciona defasagem (*staleness*). Por isso o MiMo pondera cada token por `w_t = sg[π_θ(y_t)/μ_θ(y_t)]` quando essa razão está em `[ε_low, ε_high]` e **descarta** o token fora do intervalo `[fonte S2]`. O verl anunciou um rollout "zero-mismatch" (`vexact`) com kernels batch-invariant `[fonte S30, só pelo anúncio]`. O MiMo-V2.6 vai além: grava no rollout os experts escolhidos e o conjunto de candidatos do top-p de cada token e os reproduz no treino (§3.5) `[fonte S76]`. É o mesmo tema da paridade numérica e do não determinismo do Módulo 1 (capítulos 4 e 25): a soma em ponto flutuante não é associativa, então dois motores "iguais" não dão os mesmos log-probs.

### 1.4 A âncora: KL para um modelo de referência

**Objetivo RLHF regularizado** (o `π_ref` costuma ser o checkpoint SFT, congelado):

$$\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot|x)}\big[r(x,y)\big]\;-\;\beta\,\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{ref}(\cdot|x)\big)$$

A solução fechada é `π*(y|x) = π_ref(y|x)·exp(r(x,y)/β) / Z(x)`. Invertendo, `r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)`: a recompensa fica implícita na razão de log-probs, e é daí que sai o DPO `[fonte S51]`.

**Para que serve o termo de KL.** (a) impede *reward hacking*: o modelo não pode ir para regiões onde o RM erra; (b) preserva capacidades do SFT (o RL's Razor mostra que o esquecimento acompanha o KL ao base na distribuição da tarefa nova, §4.9); (c) mantém a geração fluente. `β` alto = mudança pequena. Exemplos reais: o verl usa `algorithm.kl_ctrl.kl_coef=0.001` no exemplo PPO de GSM8K `[fonte S30]`; o Tülu 3 usou DPO normalizado por comprimento com `β = 5` `[fonte S42]` (a escala de `β` muda com a normalização, não compare `β` entre variantes). Vários pipelines de RLVR modernos removem o termo de KL e controlam a deriva por clipping e pelo próprio on-policy `[interpretação; ex.: DAPO]`.

**KL por token e estimadores.** Em LLMs, `KL(π_θ‖π_ref)` de uma sequência = soma, nos tokens amostrados de `π_θ`, de estimativas de `log π_θ(y_t) − log π_ref(y_t)`. Com `ρ = π_ref(y_t)/π_θ(y_t)`: `k1 = −log ρ` (não enviesado, alta variância, pode ser negativo), `k2 = ½(log ρ)²`, `k3 = (ρ − 1) − log ρ` (não enviesado e sempre ≥ 0, usado pelo GRPO) `[fonte S67, S57]`.

**KL direto vs reverso.** Esta é a distinção conceitual mais útil do módulo:
- **SFT = KL direto.** Minimizar a entropia cruzada nos dados = minimizar `KL(p_dados ‖ π_θ)` (a menos de uma constante). É *mass-covering*: o modelo é punido se der probabilidade baixa a algo que os dados fazem, então espalha massa por todos os modos, inclusive os que não consegue reproduzir bem.
- **RL com KL e destilação on-policy = KL reverso.** `KL(π_θ ‖ π_alvo)` é *mode-seeking*: o modelo é punido por pôr massa onde o alvo não põe, então se concentra nos modos que consegue representar. É por isso que a destilação on-policy funciona bem quando o aluno é menor que o professor `[fonte S8, S66]`.

### 1.5 Taxonomia dos métodos

| Família | Método | Dado necessário | Sinal | On/off | Precisa de π_ref? | Precisa de RM? | Ideia central |
|---|---|---|---|---|---|---|---|
| Imitação | **SFT** | demonstrações | denso (cada token) | off | não | não | MLE nas respostas; base de tudo |
| Imitação | RFT / STaR / ReST-EM | prompts + verificador | denso no que passa no filtro | semi-on (iterado) | não | não (verificador) | gera, filtra os corretos, faz SFT neles |
| Destilação | off-policy (seq-KD) | saídas do professor | denso | off | não | não | SFT nas saídas (ou logits) de um modelo maior |
| Destilação | on-policy (GKD, MiniLLM, OPD) | prompts + professor | denso (log-prob do professor) | on | professor | não | KL reverso por token nos tokens do aluno |
| Destilação | **MOPD** | prompts roteados por domínio + professores RL | denso + ORM opcional | on | professores | opcional | um professor por domínio, fundidos no aluno |
| RLHF | RM + **PPO** | pares humanos → RM; prompts | escalar por sequência | on | sim (KL) | sim | maximizar RM com KL; crítico estima vantagem |
| Preferência offline | **DPO** | pares (y_w, y_l) | logístico sobre razões | off | sim | não (implícito) | RM implícito `β·log π/π_ref` |
| Preferência offline | **IPO** (*Identity Preference Optimisation*; o nome "Identity Policy Optimization" que às vezes aparece não é o do artigo) | pares | quadrático | off | sim | não | regressão da margem para `1/(2τ)`; não satura quando as preferências são determinísticas |
| Preferência offline | **KTO** | rótulo binário bom/ruim por resposta | assimétrico (aversão à perda) | off | sim | não | utilidade de Kahneman-Tversky relativa a um ponto de referência `z_0` |
| Preferência offline | **SimPO** | pares | logístico com margem | off | **não** | não | recompensa = log-prob média por token; margem `γ` |
| Preferência offline | ORPO | pares | SFT + razão de odds | off | não | não | junta SFT e preferência numa perda só |
| Contrastivo | unlikelihood, SLiC, CPO | pares/negativos | margem/hinge | off | varia | não | empurra para cima o positivo e para baixo o negativo diretamente |
| RL "puro" | **GRPO**, RLOO, REINFORCE++ | prompts + verificador/RM | escalar por sequência, vantagem relativa ao grupo | on | opcional | opcional | sem crítico: baseline = média do grupo |
| RL | **PPO** | idem | GAE com crítico | on | opcional | opcional | razão de importância com clipping |
| RL por regressão | **AWR** | dados de uma política de comportamento | `exp(A/β)` | off | política de comportamento | crítico `V` | regressão supervisionada ponderada pela vantagem |
| RM + RL customizado | RM + A3C/TRPO/… | pares → RM | escalar | on | — | sim | qualquer algoritmo de RL contra um RM explícito |
| Feedback de IA | **Constitutional AI / RLAIF** | princípios escritos | crítica e revisão (SL) + preferências de IA (RL) | off (SL) + on (RL) | sim | PM treinado com rótulos de IA | auto-crítica guiada por constituição |
| Auto-jogo | SPIN, self-rewarding, debate | dados SFT / o próprio modelo | comparativo | iterativo | versão anterior | o próprio modelo (self-rewarding) | o modelo compete com versões de si mesmo ou com outro agente |

**Fórmulas-resumo** (convenção: `σ` = sigmoide; `h_θ(x,y) = log π_θ(y|x) − log π_ref(y|x)`; `y_w` = escolhida, `y_l` = rejeitada; somas sobre tokens implícitas em `log π(y|x)`):

- SFT: `L = −Σ_t m_t · log π_θ(y_t | x, y_<t) / Σ_t m_t`, com `m_t = 1` nos tokens do assistente.
- Modelo de recompensa (Bradley–Terry): `P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`, `L_RM = −log σ(r_w − r_l)` `[fonte S49]`.
- PPO (por token): `L = −E[min(ρ_t·Â_t, clip(ρ_t, 1−ε, 1+ε)·Â_t)]`, `ρ_t = π_θ(y_t)/π_old(y_t)` `[fonte S56]`.
- GRPO: para `G` respostas do mesmo prompt com recompensas `r_i`, `Â_i = (r_i − mean(r))/std(r)` aplicado a todos os tokens da resposta `i`, mais o objetivo clipado do PPO e, opcionalmente, `β·k3` `[fonte S57]`.
- DPO: `L = −log σ(β·[h_θ(x,y_w) − h_θ(x,y_l)])` `[fonte S51]`.
- IPO: `L = (h_θ(x,y_w) − h_θ(x,y_l) − 1/(2τ))²` `[fonte S52]`.
- KTO: `z_0 = KL(π_θ‖π_ref)` estimado no lote (sem gradiente); `v = λ_D·σ(β(h_θ − z_0))` se desejável, `v = λ_U·σ(β(z_0 − h_θ))` se indesejável; `L = E[λ_y − v]` `[fonte S53]`.
- SimPO: `L = −log σ( (β/|y_w|)·log π_θ(y_w|x) − (β/|y_l|)·log π_θ(y_l|x) − γ )` `[fonte S54]`.
- ORPO: `L = L_SFT + λ·(−log σ(log odds_θ(y_w) − log odds_θ(y_l)))`, `odds = p/(1−p)` `[fonte S55]`.
- AWR: `π_{k+1} = argmax_π E_{(s,a)∼D}[ log π(a|s) · exp((R_{s,a} − V(s))/β) ]` `[fonte S58]`. A "referência" aqui é a política que coletou os dados (comportamento), e a vantagem vem de um crítico `V` treinado por regressão.
- Destilação on-policy (KL reverso por token): vantagem `Â_t = sg[log π_T(y_t) − log π_θ(y_t)]`, `y ∼ π_θ` `[fonte S3, S6]`.
- CAI: fase SL = o modelo critica e revisa as próprias respostas segundo princípios e faz SFT nas revisões; fase RL = um modelo compara pares segundo os princípios, treina-se um modelo de preferência com esses rótulos e faz-se RL contra ele (RLAIF) `[fonte S59]`.
- SPIN: perda tipo DPO em que `y_w` = resposta humana do SFT e `y_l` = resposta da iteração anterior do próprio modelo `[fonte S60]`.
- Debate: dois agentes argumentam e um juiz decide; proposto como método de supervisão escalável `[fonte S61]`.

**Leitura para o praticante** `[interpretação]`: SFT é o alicerce; preferência offline é barata e pede pares; RL on-policy é caro (a geração domina o tempo), mas generaliza e esquece menos; destilação on-policy dá o sinal denso do SFT com a distribuição do RL, e MOPD é a versão "multi-domínio" disso.

### 1.6 O que se sabe do treino dos modelos da plataforma

| Modelo | O que as notas do Módulo 1 / fontes dizem do treino | Pós-treino documentado |
|---|---|---|
| Qwen3.8-Flash-Next (125,7B principal, ~6B ativos) | pré-treino com **Muon + AdamW** (Muon com 8 iterações Newton–Schulz nas matrizes "lineares"; AdamW em embeddings, roteador e projeções low-rank do GR); a 4× o LR ótimo, AdamW+Qwen3.5 teve 183 spikes/10k passos e Muon+GR zero (`arch-qwen.md`); indexador QSA treinado por **destilação** da atenção densa (KL contra a distribuição max-pooled) | o card traz amostragem recomendada e kwargs (`reasoning_effort`, `enable_thinking`, `preserve_thinking`); detalhes do pós-treino não estão nas notas |
| DeepSeek-V4-Flash (284B / 13B) | **Muon** híbrido: 10 iterações (8 com `(3,4445, −4,7750, 2,0315)` e 2 com `(2, −1,5, 0,5)`); KV com QAT (`arch-ds.md`, `quant.md`) | não documentado nas notas |
| GLM-5.3-Flash (320B / 18B) | — | não documentado nas notas |
| MiMo-V2-Flash / V2.6 (309B / 15B; caso de estudo externo) | AdamW (β1 0,9, β2 0,95, wd 0,1, clip 1,0), FP8 misto (BF16 na projeção de saída da atenção, embedding e head; FP32 no roteador), 27T tokens `[fonte S2]` | SFT → RL especializado → MOPD; V2.6: MOPD2 `[fonte S1–S4]` |

`[interpretação]` O `MiMo-V2.6-Flash-MOPD` tem o mesmo porte do GLM-5.3-Flash (309B vs 320B total). Já existem quantizações EXL3, NVFP4 e GGUF da comunidade no Hub (`MikeRoz/MiMo-V2.6-Flash-MOPD-4.03bpw-h8-exl3`, `revorm/…-NVFP4`, `ggml-org/…-GGUF`) `[fonte: busca no Hub]`. É um candidato natural para o tutorial "receita do zero" do Módulo 1, mas a arquitetura (SWA 128 + GA, MTP de 5 camadas, sem shared experts) exige verificar suporte no vLLM/SGLang para SM121 antes de prometer.

---

## 2. Matemática de sistemas de treino

### 2.1 Memória por parâmetro no full fine-tuning

O que vive na memória durante o treino: **pesos**, **gradientes**, **estado do otimizador**, eventualmente **pesos mestres** em FP32, **ativações** guardadas para o backward e buffers temporários (logits, comunicação). Os quatro primeiros escalam com o nº de parâmetros `Ψ`; as ativações escalam com tokens × largura × profundidade (§2.4).

| Regime | Pesos | Grad | Mestres | Otimizador | **B/param** | Comentário |
|---|---|---|---|---|---|---|
| A. Precisão mista clássica (ZeRO, Megatron) | 2 (BF16) | 2 (BF16) | 4 (FP32) | 8 (m, v FP32) | **16** | a conta do artigo ZeRO: `2Ψ + 2Ψ + KΨ`, `K = 12` `[fonte S35]` |
| A′. Carga em FP32 + autocast BF16 | 4 | 4 | — | 8 | **16** | é o padrão do `SFTTrainer` se você não passar `dtype`: "If dtype is not specified … it defaults to float32" `[fonte S13]` |
| B. BF16 puro + AdamW | 2 | 2 | — | 4 (m, v BF16) | **8** | o AdamW do PyTorch cria `m`, `v` no dtype do parâmetro; perde atualizações pequenas (§2.2) |
| C. BF16 + AdamW 8-bit (bitsandbytes) | 2 | 2 | — | ~2 | **~6** | estados quantizados em blocos; risco de qualidade a medir |
| D. Muon em precisão mista | 2 | 2 | 4 | 4 (1 momento FP32) | **12** | Muon só tem um buffer `[fonte S32]` |
| E. Muon BF16 puro | 2 | 2 | — | 2 | **6** | MLX: estado `zeros_like(param)` = dtype do parâmetro `[fonte S21]` |
| LoRA: por parâmetro **treinável** | 4 (FP32) | 4 | — | 8 | **16** | o PEFT mantém adaptadores em FP32 por padrão (`autocast_adapter_dtype`) `[interpretação a partir da doc do PEFT]` |
| LoRA: por parâmetro **congelado** | 2 (BF16) ou ~0,52 (NF4+DQ) | 0 | 0 | 0 | **2 / 0,52** | sem grad, sem otimizador |

**Tabela de referência** (só estado: pesos + grads + otimizador; sem ativações) `[cálculo próprio]`:

| Modelo | Params | 16 B/p | 12 B/p | 8 B/p | 6 B/p | pesos BF16 | NF4+DQ | MLX 4-bit (g64) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0,60 B | 9,6 GB | 7,2 | 4,8 | 3,6 | 1,2 | 0,3 | 0,3 |
| Qwen3-1.7B | 1,70 B | 27,2 | 20,4 | 13,6 | 10,2 | 3,4 | 0,9 | 1,0 |
| Llama-3.2-3B | 3,21 B | 51,4 | 38,6 | 25,7 | 19,3 | 6,4 | 1,7 | 1,8 |
| Qwen3-4B | 4,02 B | 64,3 | 48,2 | 32,2 | 24,1 | 8,0 | 2,1 | 2,3 |
| Llama-3.1-8B | 8,03 B | **128,5** | 96,4 | 64,2 | 48,2 | 16,1 | 4,1 | 4,5 |
| Qwen3-8B | 8,19 B | 131,1 | 98,3 | 65,5 | 49,1 | 16,4 | 4,2 | 4,6 |
| Qwen3-14B | 14,8 B | 236,8 | 177,6 | 118,4 | 88,8 | 29,6 | 7,6 | 8,3 |
| Qwen3-30B-A3B (MoE) | 30,5 B | 488,5 | 366,4 | 244,3 | 183,2 | 61,1 | 15,8 | 17,2 |
| Llama-3.3-70B | 70,55 B | 1.128,9 | 846,6 | 564,4 | 423,3 | 141,1 | 36,4 | 39,7 |
| gpt-oss-120b (MoE) | 116,8 B | 1.868,8 | 1.401,6 | 934,4 | 700,8 | 233,6 | 60,3 | 65,7 |
| Qwen3.8-Flash-Next (principal) | 125,74 B | 2.011,8 | 1.508,9 | 1.005,9 | 754,4 | 251,5 | 64,9 | 70,7 |
| DeepSeek-V4-Flash | 284,3 B | 4.548,8 | 3.411,6 | 2.274,4 | 1.705,8 | 568,6 | 146,7 | 159,9 |
| GLM-5.3-Flash | 320 B | 5.120,0 | 3.840,0 | 2.560,0 | 1.920,0 | 640,0 | 165,1 | 180,0 |

As contagens de parâmetros dos modelos densos e do Qwen3-30B-A3B foram refeitas a partir das formas do config (Llama-3.1-8B = 8,030 B, igual ao oficial 8.030.261.248; Qwen3-30B-A3B = 30,53 B total / 3,35 B ativos). As dos três modelos da plataforma vêm das notas do Módulo 1. NF4+DQ = 4,127 bits/param (§3.2); MLX 4-bit com grupo 64 e escala+bias em FP16 = 4,5 bits/param.

**Duas consequências para MoE** `[cálculo próprio]`:
1. No full FT de um MoE, a memória escala com os parâmetros **totais** (todo expert tem grad e estado de otimizador), enquanto o compute por token escala com os **ativos**. Qwen3-30B-A3B: 488 GB de estado para um modelo que computa como um ~3,3B.
2. Num MoE, um expert pode passar um micro-lote inteiro sem receber tokens. O gradiente dele é zero naquele passo, mas o Adam continua decaindo os momentos. O MiMo usa exatamente isso como métrica de saúde do SFT de MoE (§4.1).

### 2.2 Por que os pesos mestres em FP32 existem: a conta do BF16

O BF16 tem 7 bits de mantissa explícitos. Para um peso `w` em `[2^e, 2^(e+1))`, a distância entre números representáveis (ulp) é `2^(e−7)` e o arredondamento ao par mais próximo descarta qualquer atualização menor que meio ulp. Simulação com arredondamento BF16 real `[cálculo próprio]`:

| `w` | ulp | ½ ulp | `w + 1e-6` | `w + 1e-5` | `w + 1e-4` |
|---|---|---|---|---|---|
| 0,002 | 1,53e-5 | 7,63e-6 | **sem mudança** | +1,53e-5 (53 % a mais) | +1,07e-4 |
| 0,02 | 1,22e-4 | 6,10e-5 | **sem mudança** | **sem mudança** | +1,22e-4 |
| 0,2 | 9,77e-4 | 4,88e-4 | **sem mudança** | **sem mudança** | **sem mudança** |

Com AdamW, o passo por coordenada é ≈ `lr × m̂/√v̂`, de módulo da ordem de `lr`. Um SFT típico usa `lr ≈ 1e-5` a `2e-5` (o padrão do TRL é `2e-5`). Num peso de magnitude 0,02, isso dá passos de ~1e-5: **todos arredondados para zero** em BF16 puro. O termo de weight decay (`lr·λ·w = 1e-5 × 0,1 × 0,02 = 2e-8`) some sempre. Só os pesos pequenos se movem, e em degraus quantizados. Por isso existem os pesos mestres FP32 (regime A), a soma compensada de Kahan e o arredondamento estocástico nos otimizadores BF16 (que torna o arredondamento correto *em média*) `[interpretação; bibliotecas como torchao oferecem otimizadores com arredondamento estocástico — não verificado nesta pesquisa]`.

**Figura interativa sugerida**: escolha `w` e `lr` e veja, passo a passo, o peso em FP32 vs BF16 (RNE) vs BF16 com arredondamento estocástico (§6, fig. F2).

### 2.3 Treino em FP8 e NVFP4

- **O que o FP8 economiza.** Treinar "em FP8" quer dizer fazer os GEMMs em FP8 (e às vezes guardar ativações em FP8). O estado do otimizador e os pesos mestres continuam em precisão alta. O DeepSeek-V3 é a referência: GEMMs FP8 com quantização fina (blocos 1×128 para ativações e 128×128 para pesos), acumulação promovida para FP32, pesos mestres e gradientes de pesos em precisão alta, momentos do AdamW em **BF16**, e componentes sensíveis (embedding, head, gating do MoE, normalizações, atenção) fora do FP8 `[fonte S63]`. O MiMo-V2-Flash segue o mesmo desenho, com BF16 na projeção de saída da atenção, no embedding e no head, e **FP32 nos parâmetros do roteador** `[fonte S2]`. Conclusão: FP8 acelera compute e encolhe ativações, mas quase não mexe nos 12–16 B/param do estado.
- **NVFP4 no pré-treino.** O playbook `nvfp4-pretraining` da NVIDIA (Megatron-Bridge, receita `bf16_with_nvfp4_mixed`, últimas 4 camadas em BF16) mediu no **GB300 (DGX Station), não no Spark**, Llama 3.1 8B com GBS 64, MBS 4, seq 4096 e dados sintéticos: BF16 9,05 s/passo, ~1.399 TFLOP/s, 221,6 GB de pico; NVFP4 5,39 s/passo, ~2.347 TFLOP/s, 207,8 GB → **1,68×** mais rápido e só ~6 % menos memória `[fonte S23]`. De novo: precisão baixa compra velocidade, não memória de estado.
- **No GB10 (sm_121).** O GB10 tem tensor cores FP8/FP4, mas o suporte dos caminhos de treino (Transformer Engine, kernels NVFP4 de treino) para a família 12x não foi confirmado nesta pesquisa. O playbook NeMo do Spark menciona "FP8 precision options", e o playbook NVFP4 exige GB300 `[fonte S23; não verificado para sm_121]`. O Módulo 1 já mostrou que vários kernels "Blackwell" do vLLM são só `sm_100` (capítulos 23 e 28); espere o mesmo no treino.

### 2.4 Ativações e gradient checkpointing

Para um bloco Transformer com ativações em 16 bits, Korthikanti et al. contam por camada `s·b·h·(34 + 5·a·s/h)` bytes (`s` = sequência, `b` = micro-lote, `h` = hidden, `a` = cabeças). O termo `5as/h` vem de guardar a matriz de atenção; com FlashAttention ele some e sobra **≈ 34·s·b·h bytes por camada** `[fonte S36]`. A conta é para um GPT com MLP 4h e GeLU; refeita para um Llama (SwiGLU com intermediário 3,5h, GQA 4:1, RMSNorm, sem dropout) dá de novo ~34·s·b·h `[cálculo próprio]`.

- **Checkpointing completo**: guarda só a entrada de cada camada (`2·s·b·h` bytes) e recalcula o resto no backward. Pico ≈ `2·s·b·h·L + 34·s·b·h` (o segundo termo é a camada sendo recalculada). Custo: mais um forward, ou seja, o treino passa de ~6N para ~8N FLOPs por token (§2.7).
- **Seletivo**: recalcula só as partes baratas e grandes. O TRL tem `gradient_checkpointing_kwargs={"selective": True}`, que guarda a saída da atenção para não recalculá-la em contexto longo `[fonte S14]`.
- **Logits**: o tensor `[b, s, V]` costuma ser o maior de todos. Em FP32: `s·V·4` bytes por sequência. Para vocabulários de 128–152K isso é ~1 GB a cada 2K tokens. O TRL usa por padrão `loss_type="chunked_nll"` (descarta posições com `labels == -100` **antes** do `lm_head` e calcula a entropia cruzada em pedaços), com ~30 % menos VRAM de pico no Qwen3-1.7B, até ~50 % com FSDP2 × 4 GPUs. Não é compatível com Liger, PEFT ou VLM `[fonte S14]`.

Tabela `[cálculo próprio]` (b = 1, BF16, FlashAttention):

| Modelo | s | sem ckpt | ckpt completo | logits FP32 |
|---|---|---|---|---|
| Llama-3.1-8B (h 4096, L 32, V 128.256) | 2.048 | 9,13 GB | 0,82 GB | 1,05 GB |
| | 8.192 | 36,5 GB | 3,29 GB | 4,20 GB |
| | 32.768 | 146,0 GB | 13,2 GB | 16,8 GB |
| Qwen3-8B (h 4096, L 36, V 151.936) | 2.048 | 10,3 GB | 0,89 GB | 1,24 GB |
| | 32.768 | 164,3 GB | 14,2 GB | 19,9 GB |
| Llama-3.3-70B (h 8192, L 80) | 2.048 | 45,6 GB | 3,25 GB | 1,05 GB |
| | 8.192 | 182,5 GB | 13,0 GB | 4,20 GB |

**UMA e offload** `[interpretação]`: num GPU discreto, offload de ativações (`activation_offloading=True` no TRL), de otimizador (ZeRO-Offload) ou otimizadores paginados movem dados para a RAM do host e liberam VRAM. No GB10 e no Apple Silicon a RAM do host **é** a memória da GPU (capítulo 27 do Módulo 1). Offload para "CPU" não cria capacidade nova; no máximo muda a fragmentação ou o pinning. Só offload para NVMe (ZeRO-Infinity) aumenta a capacidade. Os playbooks da NVIDIA para o Spark repetem a dica do Módulo 1: se o treino falha "dentro da capacidade", libere o page cache com `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` `[fonte S23]`.

### 2.5 ZeRO 1/2/3 e FSDP

O ZeRO divide o estado entre os `N_d` processos de data parallel, em vez de replicá-lo `[fonte S35]`:

| Estágio | O que é particionado | Memória por processo (K = 12) |
|---|---|---|
| DDP | nada | `(2 + 2 + K)·Ψ` |
| ZeRO-1 (P_os) | otimizador | `(2 + 2)·Ψ + K·Ψ/N_d` |
| ZeRO-2 (P_os+g) | + gradientes | `2Ψ + (2 + K)·Ψ/N_d` |
| ZeRO-3 (P_os+g+p) ≈ FSDP `FULL_SHARD` | + parâmetros | `(2 + 2 + K)·Ψ/N_d` |

- Exemplo do artigo (7,5B, `N_d = 64`): 120 GB → 31,4 → 16,6 → 1,9 GB.
- Exemplo da plataforma (8,03B, `N_d = 2`, um processo por Spark) `[cálculo próprio]`: DDP **128,5 GB** (não cabe); ZeRO-1 80,3; ZeRO-2 72,3; **ZeRO-3 64,2 GB por nó** + ativações → cabe.
- **Comunicação** `[fonte S35]`: DDP, ZeRO-1 e ZeRO-2 movem `2Ψ` elementos por passo (reduce-scatter + all-gather dos gradientes); ZeRO-3 move `3Ψ` (1,5×), porque reúne os parâmetros no forward e de novo no backward.
- **FSDP2** (`fully_shard`, por parâmetro, sobre DTensor) é o equivalente nativo do PyTorch. O playbook multi-nó do Spark usa Accelerate com `distributed_type: FSDP`, `fsdp_sharding_strategy: FULL_SHARD`, `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`, `fsdp_backward_prefetch: BACKWARD_PRE`, `fsdp_state_dict_type: SHARDED_STATE_DICT`, `fsdp_use_orig_params: true`, `mixed_precision: 'bf16'`, `num_machines: 2`, `num_processes: 2` `[fonte S23]`. No TRL, os perfis prontos são `--accelerate_config fsdp1|fsdp2|zero1|zero2|zero3|multi_gpu|single_gpu` `[fonte S15]`.

**Quanto custa a rede entre 2 Sparks** `[cálculo próprio]`. Link de 200 Gb/s = 25 GB/s teóricos, servido por duas funções PCIe Gen5 ×4 de ~126 Gb/s cada (uma só limita a ~15,7 GB/s; `ds-launch.md`); usei ~20 GB/s efetivos como hipótese.
- All-reduce dos gradientes BF16 de um 8B (16,1 GB) em anel com N = 2: cada nó envia ~16,1 GB → **~0,8 s** por passo. ZeRO-3: ~1,2 s.
- Compute de um passo com lote global de 64 × 2.048 = 131.072 tokens: `6 × 8,03e9 × 131.072 = 6,3e15 FLOP` → **~63 s** nos 2 Sparks a 50 TFLOPS efetivos cada.
- A comunicação fica na casa de 1–2 % sem sobreposição (mais, se o ZeRO-3 reunir parâmetros a cada micro-lote de acumulação).

Treinar com data parallel sobre o cabo do Spark é muito mais tolerante que fazer decode com TP sobre o mesmo cabo, onde cada token espera ~100 coletivas (capítulo 39 do Módulo 1). A diferença está na razão compute/comunicação por passo.

### 2.6 Tensor, pipeline, expert e context parallel no treino

- **TP (Megatron)**: o mesmo par coluna→linha do capítulo 38, mas o backward também faz all-reduce. Por camada: 2 all-reduces no forward e 2 no backward, cada um de `s·b·h·2` bytes. Exemplo 8B, TP = 2 entre os Sparks, micro-lote 8 × 2.048: 134 MB por all-reduce × 4 × 32 camadas ≈ 17,2 GB por micro-lote (~0,86 s a 20 GB/s) contra ~7,9 s de compute por nó a 50 TFLOPS → ~11 % sem sobreposição `[cálculo próprio]`. O *sequence parallel* do Megatron divide as ativações de LayerNorm/dropout entre os ranks de TP e dá a redução `÷t` de ativações do Korthikanti `[fonte S36]`.
- **PP**: camadas em estágios. A bolha do 1F1B custa `(p − 1)/m` do tempo ideal (`p` estágios, `m` micro-lotes) `[fonte S37]`. Com 2 estágios e 8 micro-lotes, 12,5 %.
- **EP**: experts distribuídos; dois all-to-all por camada MoE no forward (despacho e retorno) e dois no backward. É a forma natural de treinar MoE em vários nós (Megatron-Core, DeepEP). O Axolotl ganhou EP em 2026/06 (DeepEP ou `all_to_all_single`) `[fonte S27]`.
- **CP / ring attention / Ulysses**: dividem a sequência entre ranks para contexto longo. O TRL 1.13 documentou 1.048.576 tokens por passo no Qwen3-8B num nó 8×H100 (~380 s/passo), com `chunked_nll`, offload do checkpointing e YaRN, e sem packing `[fonte S18]`.
- **No Spark** `[interpretação]`: para SFT de densos até ~8B, FSDP/ZeRO-3 em 2 nós é o caminho mais simples (é o do playbook). TP entre nós só se justifica quando um nó não comporta nem uma camada com as ativações. Para MoE grande (fora de alcance em full FT, §5) o EP seria obrigatório.

### 2.7 FLOPs, tempo de treino e a banda do passo do otimizador

- **Full FT**: ≈ `6N` FLOPs por token treinado (forward `2N`, backward `4N`: gradientes das ativações e dos pesos). Com checkpointing completo, ≈ `8N`. Num MoE, `N` = parâmetros **ativos**.
- **LoRA**: o backward não calcula o gradiente dos pesos congelados, então ≈ `4N` (`2N + 2N`), ou ≈ `6N` com checkpointing. O Thinking Machines calcula "slightly more than ⅔ of the FLOPs" do full FT por passe `[fonte S5]`.
- **Teto no GB10** (pico BF16 denso ~125 TFLOPS, estimativa usada no Módulo 1) `[cálculo próprio]`:

| Caso | FLOPs/token | Teto a 100 % MFU | A 40 % MFU | 10M tokens a 40 % |
|---|---|---|---|---|
| 3B full FT (ckpt, 8N) | 2,57e10 | 4.863 tok/s | 1.945 tok/s | 1,4 h |
| 8B LoRA (ckpt, 6N) | 4,82e10 | 2.594 tok/s | 1.038 tok/s | 2,7 h |
| 8B full FT (ckpt, 8N) | 6,42e10 | 1.946 tok/s | 778 tok/s | 3,6 h |
| 70B QLoRA (ckpt, 6N, sem custo de dequant.) | 4,23e11 | 295 tok/s | 118 tok/s | 23,5 h |

  A MFU real do GB10 em treino não foi medida nesta pesquisa. Os 40 % são hipótese, e o deslizador da figura F5 existe para isso.
- **Os números da NVIDIA não fecham.** O post de 2025-10-24 dá "peak tokens/s" de 82.739,2 (Llama 3.2 3B full, batch 8, seq 2048), 53.657,6 (Llama 3.1 8B LoRA, batch 4) e 5.079,4 (Llama 3.3 70B QLoRA, batch 8) `[fonte S24]`. Pela contabilidade acima, isso exigiria 1,60, 1,72 e 1,43 PFLOP/s, de 11 a 14× o pico BF16 denso e acima até do FP4 denso. A métrica deve contar outra coisa (não está definida no post). Para planejar, use FLOPs e meça.
- **Banda do passo do otimizador** `[cálculo próprio]`: um passo AdamW em precisão mista lê grad, mestre, `m`, `v` e escreve mestre, `m`, `v` e a cópia BF16: ~28 B/param. Para 8B: ~225 GB → **~1 s por passo** a ~230 GB/s efetivos. Com LoRA r=16 (41,9 M params): ~1,2 GB, ~5 ms. Em memória LPDDR5x, passos de otimizador frequentes custam caro no full FT. Acumular gradientes (lote efetivo maior) amortiza esse segundo.

### 2.8 Otimizadores

- **AdamW**: `m ← β1·m + (1−β1)·g`; `v ← β2·v + (1−β2)·g²`; `w ← w − lr·( m̂/(√v̂ + ε) + λ·w )`. Padrões do `SFTConfig`: `adam_beta1 0.9`, `adam_beta2 0.999`, `adam_epsilon 1e-8`, `weight_decay 0.0`, `optim 'adamw_torch_fused'`, `max_grad_norm 1.0`, `learning_rate 2e-5`, `lr_scheduler_type 'linear'` `[fonte S13]`. Pré-treinos de LLM costumam usar `β2 = 0,95` (MiMo: β1 0,9, β2 0,95, wd 0,1, clip 1,0) `[fonte S2]`. No SFT de MoE, o MiMo descobriu que o `ε` do AdamW e a taxa de atualização do bias dos experts governam a estabilidade (§4.1).
- **AdamW 8-bit (bitsandbytes)**: momentos quantizados por blocos; usado nos exemplos do Axolotl (`optimizer: adamw_8bit` / `adamw_bnb_8bit`) `[fonte S27]`.
- **Muon** `[fonte S32, S33, S34, S21]`:
  - Momento de SGD (com Nesterov) seguido de ortogonalização da atualização por Newton–Schulz: `O ≈ U·Vᵀ` de `G = U·S·Vᵀ`. A iteração quíntica `X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X` usa `(a, b, c) = (3,4445, −4,7750, 2,0315)` por 5 passos (o MLX tem exatamente isso, com `momentum 0.95`, `nesterov=True`, `weight_decay 0.01`, `ns_steps 5`).
  - Só para matrizes 2D ocultas. Embeddings, head, normas, biases e escalares ficam com AdamW.
  - Overhead de FLOPs ≈ `T·m/B` (T = 5 passos, m = dimensão do modelo, B = tokens do lote): 0,7 % no speedrun do NanoGPT e ~0,5 % num "Llama 405B" hipotético.
  - Memória: um buffer só.
  - Para escalar (Moonlight): weight decay e atualização reescalada para casar o RMS do AdamW, `W ← W − η·(0,2·O·√max(A,B) + λW)`; ~2× de eficiência de compute vs AdamW no regime compute-ótimo (Moonlight, MoE 3B/16B, 5,7T tokens).
  - **Descasamento pré-treino/SFT**: o melhor foi Muon no pré-treino **e** no SFT; com otimizadores diferentes, o Muon no SFT não mostrou vantagem significativa `[fonte S33]`.
  - Disponível como `torch.optim.Muon` desde o PyTorch 2.9 (2025-10-15), só para parâmetros 2D e, nessa versão, sem suporte distribuído `[fonte S34]`; no MLX, `optim.Muon`, e no `mlx_lm.lora --optimizer muon` `[fonte S20]`.
  - Implicação para a plataforma `[interpretação]`: Qwen3.8 e DeepSeek-V4 foram pré-treinados com Muon. Num full FT deles, o Muon é a escolha coerente com o Moonlight. Em LoRA (matrizes `A`, `B` finas) a evidência é fraca; fique com AdamW.
- **LoRA+**: LR maior para `B` que para `A` (`loraplus_lr_ratio`, ex. 16) `[fonte S12, S72]`.
- **Adafactor**: fatora o segundo momento (memória ~sublinear); disponível no `mlx_lm.lora --optimizer adafactor` `[fonte S20]`.

### 2.9 Schedules de LR e hiperparâmetros de referência

Formas usuais: warmup linear + decaimento linear (Tülu 3), warmup + cosseno (Axolotl, LLaMA-Factory, MiMo SFT), **WSD** (warmup-stable-decay; o pré-treino do MiMo faz warmup até 3,2e-4 em 50B tokens, constante por 12T e cosseno até 1e-4 em 10T `[fonte S2]`) e constante com warmup (comum em LoRA curto).

| Fonte | Modelo | Método | LR | Schedule | Lote efetivo | Seq | Épocas |
|---|---|---|---|---|---|---|---|
| Tülu 3 `[fonte S42]` | 8B | SFT full | 5e-6 (perda **soma**) | linear, warmup 0,03 | 128 | 4.096 | 2 |
| Tülu 3 | 70B | SFT full | 2e-6 | linear, warmup 0,03 | 128 | 4.096 | 2 |
| Tülu 3 | 8B / 70B | DPO normalizado | 5e-7 / 2e-7, β = 5 | linear, warmup 0,1 | 128 | 2.048 | 1 |
| MiMo-V2-Flash `[fonte S2]` | 309B MoE | SFT full | 5e-5 → 5e-6 | cosseno | 128 | — | — |
| MOPD (Qwen3-30B-A3B) `[fonte S3]` | 30B MoE | RL por domínio (GRPO) | 3e-6 | — | 144 prompts × 8 rollouts (math/IF) | 32K–64K | — |
| `SFTConfig` (padrão) `[fonte S13]` | — | SFT | 2e-5 | linear | 8/dispositivo | `max_length 1024` | 3 |
| TRL, dica PEFT `[fonte S13]` | — | LoRA | ≈1e-4 | — | — | — | — |
| Axolotl `lora-1b.yml` `[fonte S27]` | Llama-3.2-1B | LoRA r16 α32 | 2e-4 | cosseno, warmup 0,1 | 2 × 2 | 2.048 | 1 |
| LLaMA-Factory `qwen3_lora_sft.yaml` `[fonte S28]` | Qwen3-4B-Instruct-2507 | LoRA r8 all | 1e-4 | cosseno, warmup 0,1 | 1 × 8 | 2.048 | 3 |
| mlx-lm (padrão) `[fonte S20]` | — | LoRA r8 scale 20 | 1e-5 | constante | 4 | 2.048 | 1.000 iters |
| Thinking Machines `[fonte S5]` | Llama 3 / Qwen3 | LoRA vs full | LoRA ≈ **10×** o LR ótimo do full (≈15× em runs < ~100 passos) | — | — | — | — |
| DistillationConfig (padrão) `[fonte S16]` | — | destilação on-policy | 1e-6 | linear | 8 | `max_completion_length 512` | — |

Observações:
- **Perda soma vs média**: o Tülu 3 trocou a média pela **soma** da perda para evitar o viés da acumulação de gradientes (quando a média é feita por micro-lote, amostras com poucos tokens ganham peso demais), "o que exige ajustar o LR" `[fonte S42]`. O TRL normaliza pelo nº total de tokens do lote acumulado (`average_tokens_across_devices=True`) `[fonte S13]`.
- O `scale` do mlx-lm **não** é o `α` do PEFT: `scale = 20` multiplica `B·A` diretamente, o que equivale a `α/r = 20` (α = 160 com r = 8). O código avisa quando se passa `alpha` `[fonte S20]`.

### 2.10 Lotes, packing e comprimento de sequência

- **Lote efetivo** = micro-lote × passos de acumulação × réplicas DP. A conta que importa é **tokens por passo de otimizador**. O LoRA tolera lotes grandes pior que o full FT; a penalidade não cai com o posto, é propriedade da parametrização `B·A` `[fonte S5]`.
- **Packing** (`packing=True`): o `SFTTrainer` empacota exemplos em linhas de `max_length` com *best-fit decreasing*. Estratégias: `"bfd"` (padrão; descarta o excesso de exemplos longos), `"bfd_split"` (corta os longos em pedaços), `"wrapped"` (concatena tudo e corta em blocos, misturando exemplos; o relatório do Qwen3-Coder-Next diz que isso prejudica) `[fonte S14]`. Com `"bfd"`, o `padding_free` liga sozinho: o lote vira uma sequência só, com `position_ids` reiniciados por exemplo, e o FlashAttention de comprimento variável impede um exemplo de atender ao outro. Sem FlashAttention 2/3 há risco de "contaminação de lote" `[fonte S13, S14]`. `[não verificado]` A disponibilidade de FlashAttention 2/3 para sm_121 (GB10) não foi confirmada nesta pesquisa; sem ela, use SDPA e packing sem padding-free, ou só padding.
- **Truncamento**: `max_length` padrão 1024 no SFT; só `truncation_mode="keep_start"` é suportado, o que corta a **resposta**. Filtre ou divida os exemplos longos em vez de truncar `[fonte S13]`.
- **Comprimento de sequência**: ativações ∝ `s` (com FlashAttention); FLOPs da atenção ∝ `s²`; logits ∝ `s·V`. Aumentar `s` de 2K para 32K multiplica as ativações por 16 (tabela da §2.4) e, para modelos de raciocínio, é inevitável: os exemplos do MOPD usam 32.768 (math/IF) e 65.536 (SWE) `[fonte S3]`.

---

## 3. PEFT: LoRA, QLoRA, DoRA e afins

### 3.1 LoRA: a matemática

Para uma camada linear congelada `W₀ ∈ ℝ^{d_out×d_in}`:

$$h = W_0x + \frac{\alpha}{r}\,B\,A\,x,\qquad A\in\mathbb R^{r\times d_{in}},\; B\in\mathbb R^{d_{out}\times r},\; r\ll\min(d_{in},d_{out})$$

- **Inicialização**: `A` aleatória (no PEFT, Kaiming-uniforme; no MLX, uniforme em `±1/√d_in`) e `B = 0`. No passo 0, `ΔW = 0` e o modelo é exatamente o base `[fonte S5, S20]`.
- **Parâmetros treináveis** por matriz: `r·(d_in + d_out)`. É a fatoração de baixo posto `W ≈ A·B` do capítulo 3, a mesma ideia do `q_lora_rank`/`kv_lora_rank` da MLA (capítulos 12 e 18). Lá o gargalo é arquitetura; aqui é o *update*.
- **Merge**: `W = W₀ + (α/r)·B·A`. Depois de fundido, custo zero na inferência.
- **Por que funciona**: as mudanças de pós-treino têm "dimensão intrínseca" baixa. No GPT-3 175B, o artigo original reporta 10.000× menos parâmetros treináveis e 3× menos memória de GPU que o full FT com Adam; o checkpoint do adaptador (r = 4, só `q`/`v`) cai de 350 GB para ~35 MB `[fonte S11]`.
- **Escala `α/r` e posto**: com o prefator `1/r`, a atualização inicial esperada de `(1/r)·Σ_i Δ_i` não depende de `r`, e por isso o LR ótimo quase não muda com o posto (variação < 2× entre r = 4 e r = 512) `[fonte S5]`. O **rsLoRA** usa `α/√r` para estabilizar postos altos (`use_rslora=True`) `[fonte S12]`. Convenções vistas nas fontes: Thinking Machines α = 32 fixo; Unsloth α = 2r; Axolotl r16/α32 (LoRA) e r32/α16 (QLoRA); playbook NVIDIA r8/α16; mlx-lm `scale 20` (≡ α/r = 20).

**Quais módulos** (o resultado mais prático do *LoRA Without Regret*, `[fonte S5]`):
- LoRA em **todas** as matrizes, principalmente MLP/MoE. Só atenção fica claramente pior, **mesmo com o mesmo nº de parâmetros**: atenção r = 256 perde para MLP r = 128 no Llama-3.1-8B. O mesmo vale para o Qwen3-30B-A3B (MoE).
- O LoRA iguala o full FT em eficiência de amostras quando (a) está em todas as camadas e (b) não fica limitado por capacidade (informação do dataset < capacidade do adaptador). Fora disso, não bate num "piso": aprende mais devagar.
- **RL**: igual ao full FT **mesmo com posto 1**. Argumento informacional: um episódio de policy gradient dá O(1) bits (a vantagem), enquanto o SFT dá O(nº de tokens) bits.
- **LR ótimo ≈ 10× o do full FT** (≈ 15× em runs com menos de ~100 passos).
- **Lote**: o LoRA paga mais que o full FT por lotes muito grandes, e aumentar o posto não resolve.

**Contagem para os modelos de referência** (todas as lineares de atenção e MLP; sem embedding/head) `[cálculo próprio]`:

| Modelo | r = 8 | r = 16 | r = 64 | r = 16 só atenção | r = 16 só MLP |
|---|---|---|---|---|---|
| Llama-3.2-3B | 12,2 M | 24,3 M (0,76 %) | 97,3 M | 9,2 M | 15,1 M |
| Llama-3.1-8B | 21,0 M | **41,9 M (0,52 %)** | 167,8 M | 13,6 M | 28,3 M |
| Qwen3-8B | 21,8 M | 43,6 M (0,53 %) | 174,6 M | 15,3 M | 28,3 M |
| Llama-3.3-70B | 103,5 M | 207,1 M (0,29 %) | 828,4 M | 65,5 M | 141,6 M |

(As contagens do Thinking Machines para o Llama-3.1-8B ficam ~4–15 % acima das minhas pela mesma fórmula: "all, rank=256 0.70B" vs 0,67 B; "mlp, rank=256 0.49B" vs 0,45 B; "attn, rank=256 0.25B" vs 0,22 B. A diferença deve vir de outras matrizes incluídas ou de arredondamento; a ordem de grandeza bate.)

**Tokens novos e embeddings.** Se o chat template introduz tokens especiais que o base não treinou, o LoRA nas lineares não aprende os vetores deles. É preciso treinar `embed_tokens`/`lm_head` (`modules_to_save=["embed_tokens","lm_head"]` no PEFT) ou usar um template que só use tokens existentes. O TRL avisa: com um template Jinja próprio, "você deve garantir que os tokens especiais estão no tokenizer e que a camada de embedding foi redimensionada" `[fonte S13]`.

### 3.2 QLoRA

`[fonte S9]` (Dettmers et al., 2023): "fine-tune a 65B parameter model on a single 48GB GPU" mantendo o desempenho do fine-tuning em 16 bits. Três ingredientes:
1. **NF4 (NormalFloat 4-bit)**: 16 níveis nos quantis de uma N(0,1) normalizada para [−1, 1], assimétricos (8 positivos, 7 negativos e um zero exato), com escala absmax por bloco de 64 pesos. É a mesma família "codebook + escala por bloco" dos formatos do capítulo 20, mas otimizada para pesos gaussianos e sem tensor core: o kernel desquantiza para BF16 antes do GEMM.
2. **Quantização dupla**: as escalas FP32 (32 bits / 64 pesos = 0,5 bit/param) são quantizadas em 8 bits com blocos de 256 → `8/64 + 32/(64·256) = 0,127` bit/param. Total **4,127 bits/param** = 0,516 B/param (−0,373 bit, ~3 GB num 65B) `[fonte S9; cálculo próprio]`.
3. **Otimizadores paginados**: usam memória unificada da NVIDIA para paginar estado entre CPU e GPU nos picos. `[interpretação]` No GB10 isso não cria memória (CPU e GPU são a mesma LPDDR5x), só evita o OOM em picos se houver páginas a liberar.

Configuração real (script `Llama3_70B_qLoRA_finetuning.py` do playbook PyTorch do Spark) `[fonte S23]`:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=args.dtype,      # bfloat16
    bnb_4bit_quant_storage=args.dtype
)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(r=args.lora_rank, target_modules="all-linear",
                         lora_alpha=16, lora_dropout=0, task_type=TaskType.CAUSAL_LM)
# SFTConfig: optim "adamw_torch", packing False, max_length = seq_length (2048 por padrão)
```

**Orçamento Llama-3.3-70B QLoRA num Spark** `[cálculo próprio]`: parte quantizada 35,3 GB + `embed_tokens` e `lm_head` em BF16 (o bnb não os quantiza) 4,2 GB = **39,5 GB**; LoRA r = 16 all-linear (207,1 M) × 16 B = **3,3 GB**; ativações com checkpointing ≈ 3,25 GB por sequência de 2.048 (×8 = 26 GB); logits até ~1 GB por sequência sem a perda em pedaços. Total com lote 8 ≈ 70–78 GB; com lote 1, ≈ 45 GB. O playbook NeMo do Spark roda a receita `llama_3_3_70b_instruct_squad_peft_qlora_rtx_spark.yaml` num container com `--memory=64g` `[fonte S23]`, coerente com essa conta.

**Variantes**: LoftQ inicializa `A`, `B` para compensar o erro de quantização (`init_lora_weights="loftq"`) `[fonte S12, S72]`. Para MoE, o bitsandbytes 4-bit não cobre experts guardados como `nn.Parameter`; o Unsloth desaconselha QLoRA em MoE "because BitsandBytes doesn't support it" `[fonte S26]`. A alternativa em 2026 é LoRA sobre uma base **NVFP4** (Axolotl, §3.5).

### 3.3 DoRA

`[fonte S10]` (Liu et al., ICML 2024, oral). Decompõe o peso em **magnitude** e **direção**:

$$W' = m\cdot\frac{W_0 + BA}{\lVert W_0 + BA\rVert_c},\qquad m \text{ inicializado com } \lVert W_0\rVert_c$$

(`‖·‖_c` = norma por coluna; `m` é um vetor treinável). O LoRA atua só na direção. Fecha parte da distância para o full FT, especialmente com posto baixo, e é fundível (custo zero depois do merge). Custo: mais memória e tempo no treino por causa da norma. Uso: `LoraConfig(use_dora=True)` no PEFT; `--fine-tune-type dora` no mlx-lm, que **não** suporta DoRA nas camadas de experts (`SwitchLinear`: "doesn't support DoRA yet") `[fonte S12, S20]`.

### 3.4 Outras variantes (para o glossário)

| Variante | Ideia | Onde |
|---|---|---|
| rsLoRA | escala `α/√r` | `use_rslora=True` |
| LoRA+ | LR de `B` = λ × LR de `A` (λ ≈ 16) | `create_loraplus_optimizer(..., loraplus_lr_ratio=16)` |
| PiSSA | inicializa `A`, `B` com os componentes singulares principais de `W₀` (o resíduo fica congelado) | `init_lora_weights="pissa"` / `"pissa_niter_16"` |
| OLoRA, EVA, CorDA | inicializações por QR, por ativações (EVA aloca posto por camada) e orientadas à tarefa/preservação | `init_lora_weights="olora" / "eva" / "corda"` |
| LoftQ | inicialização que compensa o erro de quantização | `init_lora_weights="loftq"` |
| LoRA-FA | congela `A` e reduz a memória de ativações | `create_lorafa_optimizer` |
| `rank_pattern` / `alpha_pattern` | posto e α por módulo (regex) | `LoraConfig` |

Todas da documentação do PEFT `[fonte S12]`.

### 3.5 LoRA em MoE

**Opções de onde pôr o adaptador** `[interpretação a partir das fontes]`:
1. Só atenção (+ shared expert). Barato, mas o *LoRA Without Regret* mostra que MLP/MoE é onde o LoRA mais rende `[fonte S5]`.
2. **Um LoRA por expert** (adaptadores 3D `[E, r, d_in]` e `[E, d_out, r]`). É o que o MLX faz no `LoRASwitchLinear` (dois `gather_mm` indexados pelo roteamento) `[fonte S20]` e o que o PEFT faz com `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]` e `rank_pattern` `[fonte S12]`. O Thinking Machines usou, por expert, posto = posto total ÷ nº de experts ativos (8 no Qwen3 MoE), o que mantém a razão "parâmetros LoRA / parâmetros full" igual à das camadas densas `[fonte S5]`.
3. Fatores compartilhados entre experts ("shared-outer": `A` ou `B` comum). O vLLM tem `--enable-moe-shared-loras` para carregar esse layout `[fonte S22]`.
4. **Roteador**: congelado. "it's not a good idea to fine-tune the router layer so we disabled it by default" (Unsloth) `[fonte S26]`.

**Quanto custa um LoRA por expert** `[cálculo próprio]`:

| Modelo | Experts por camada MoE × camadas | r por expert | Params LoRA nos experts | Estado (16 B/param) |
|---|---|---|---|---|
| Qwen3-30B-A3B (h 2048, interm. 768) | 128 × 48 | 1 | 51,9 M | 0,8 GB |
| | | 2 (= 16/8) | 103,8 M | 1,7 GB |
| | | 16 | 830,5 M | 13,3 GB |
| GLM-5.3-Flash (h 4096, interm. 2048) | 288 × 42 | 1 | 0,22 B | 3,6 GB |
| | | 8 | 1,78 B | 28,5 GB |
| | | 16 | 3,57 B | 57,1 GB |

(Atenção do Qwen3-30B-A3B com r = 16: 13,4 M. Cada camada MoE do GLM tem 7,25 B de parâmetros de experts, igual ao `arch-glm.md`.)

**Kernels e frameworks** `[fonte S26, S27]`:
- **Unsloth (2026)**: Triton grouped-GEMM + LoRA para MoE, "12x faster with >35% less VRAM and 6x longer context" no agregado. Por modelo: gpt-oss-20b ~7× mais rápido e 36 % menos VRAM a 16K; Qwen3-30B-A3B LoRA 16-bit ~1,7× e ~35 % de memória (B200); GLM-4.7-Flash 2,6× (A100). Alvos: `q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj`; `load_in_4bit = False` para MoE.
- **Axolotl**: ScatterMoE LoRA (2026/02) e **NVFP4 MoE LoRA** (2026/07). A base fica empacotada em 4 bits e só `A`/`B` treinam em BF16. ScatterMoE = W4A16 (Marlin), "any CUDA GPU sm80+"; SonicMoE = W4A4 nativo só no SM100 ("consumer Blackwell sm_120 not yet supported"). Há merge do adaptador de volta num checkpoint NVFP4 (`nvfp4_merge_aware: true`); no `nvidia/Qwen3-30B-A3B-NVFP4` o merge reproduz a perda treinada ("recovery ~1.0"). `[interpretação]` É o caminho mais próximo de "fine-tunar e voltar à receita NVFP4" do Qwen3.8, mas o GB10 é sm_121: o ScatterMoE/Marlin deveria rodar; o SonicMoE, não. Não testado aqui.

**Estabilidade do SFT de MoE (MiMo-V2-Flash)** `[fonte S2]`: a métrica crítica é o **nº de parâmetros com gradiente zero** (`num-zeros`). Se ele sobe, o balanceamento dos experts está piorando; se cai, o modelo está sobreajustando. Ela depende da **taxa de atualização do bias dos experts** (balanceamento sem perda auxiliar, à la DeepSeek) e do **`ε` do AdamW**. Valores usados: cosseno 5e-5 → 5e-6, lote 128, `ε = 1e-8`, taxa do bias 1e-4, coeficiente da perda auxiliar de sequência 1e-6. No TRL, o equivalente acessível é `router_aux_loss_coef` (o `SFTTrainer` registra `aux_loss` em MoE) `[fonte S13]`.

**RL em MoE** `[fonte S76]`:
- **Congele o roteador.** No RL do MiMo-V2.6-Pro (384 experts por camada) com roteador treinável, a carga dos experts colapsou nos primeiros 20 passos (camada 9): coeficiente de variação 0,78 → 2,0, pico 6× → 16× a média, experts "frios" (< 0,1× a média) 0,5 % → 22 %. Restaurar só o roteador do passo 20 aos valores pré-RL recuperou o balanceamento **sem mudar os benchmarks**: a causa era a deriva do roteador, não os experts. Com o roteador congelado, as três métricas ficaram planas (CV ≈ 0,7, pico ≈ 5,5×, ~1 % frios).
- **Roteamento treino ≠ inferência.** Mesmo com parâmetros idênticos, diferenças numéricas entre os motores podem trocar a escolha discreta de experts (o problema `μ ≠ π` da §1.3 aplicado ao roteamento). O MiMo grava os índices de experts usados no rollout e os **reproduz no treino** (*Rollout Routing Replay*, R3). Também grava o conjunto de candidatos do top-k/top-p de cada token e renormaliza os log-probs de treino dentro dele, para casar com o amostrador. O verl anunciou "Megatron backend LoRA and router replay support" (2026-04) `[fonte S30, só pelo anúncio]`.
- **Mesmos pesos nos dois motores.** O rollout do MiMo-V2.6 roda os experts em **MXFP4**. Depois de cada atualização, o treino aplica quantize–dequantize (QDQ) nos experts com as mesmas restrições numéricas dos GEMMs MXFP4 usados no rollout, para os dois motores verem pesos idênticos. É a ponte direta com os capítulos 20–22 e 25 do Módulo 1.

**Ligação com especulação** `[interpretação]`: as receitas do Módulo 1 usam cabeças MTP (GLM, Qwen3.8), DFlash2 ou DSpark treinadas contra o modelo-alvo **original**. Depois de um SFT, a distribuição do alvo muda e a taxa de aceitação do rascunho tende a cair. O MiMo treina as cabeças MTP também no pós-treino: a cabeça única do pré-treino é replicada K vezes e treinada em conjunto `[fonte S2]`. No RL do MiMo-V2.6, o rascunho padrão passou a ser um **DFlash de bloco 6**, treinado primeiro na política SFT e depois ajustado em logs de rollouts do começo do RL, reamostrados para casar com a distribuição de treino. O comprimento aceito médio ficou **31,3 % maior** que o do MTP-3 herdado do SFT, e um DFlash em FP8 deu ~10,3 % a mais de throughput por nó em contexto longo `[fonte S76]`. É o mesmo DFlash da receita GLM do Módulo 1 (capítulo 37), agora com a lição de que o rascunho precisa acompanhar a política. Depois de fine-tunar, meça de novo a aceitação (capítulos 35–37) e, se cair, ajuste também o rascunho.

### 3.6 Fundir (merge) adaptadores

- PEFT: `model = PeftModel.from_pretrained(base, id).merge_and_unload()`; `merge_adapter()`/`unmerge_adapter()` para alternar; `add_weighted_adapter(adapters=[...], weights=[...], combination_type="linear" | "svd" | "cat" | "ties" | "ties_svd" | "dare_ties" | "dare_linear" | "dare_linear_svd" | "magnitude_prune" | "magnitude_prune_svd", density=...)` para combinar adaptadores `[fonte S12]`.
- LLaMA-Factory: `llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml`, com o aviso no próprio arquivo: "DO NOT use quantized model or quantization_bit when merging lora adapters" `[fonte S28]`.
- MLX: `mlx_lm.fuse --model <base> --adapter-path adapters --save-path fused_model` (`--export-gguf` só para Mistral/Mixtral/Llama em fp16) `[fonte S20]`.
- Axolotl: `axolotl merge-lora config.yaml` `[fonte S27]`.

**O problema do QLoRA no merge** `[interpretação]`: o adaptador aprendeu a corrigir `deq(W_NF4)`, não `W_BF16`. Fundir em `W_BF16` dá `W_BF16 + BA ≠ deq(W_NF4) + BA`, e a diferença é o erro de quantização que o adaptador possivelmente compensava. Fundir em `deq(W_NF4)` preserva a função treinada, mas "assa" o erro NF4 no modelo. Nos dois casos, **meça o KLD** contra o modelo treinado (capítulo 25).

**Voltando para as receitas do Módulo 1** `[interpretação]`: o caminho robusto é fundir num BF16, requantizar (NVFP4 via ModelOpt, EXL3, FP8…) e medir o KLD contra o BF16 fundido e a métrica de tarefa contra o base. Um cuidado novo: um delta de LoRA pequeno pode ter magnitude comparável ao ruído de quantização por peso. Confirme que o efeito do fine-tune **sobrevive** à quantização comparando a métrica de tarefa do quantizado com a do BF16 fundido. O merge "consciente de NVFP4" do Axolotl existe exatamente para esse caso.

### 3.7 Servir LoRA (multi-LoRA no vLLM e no MLX)

Flags do vLLM (`LoRAConfig`) `[fonte S22]`:

| Flag | Padrão | Função |
|---|---|---|
| `--enable-lora` | — | liga o suporte a adaptadores |
| `--lora-modules` | — | `nome=caminho` ou JSON `{"name": "...", "path": "...", "base_model_name": "..."}` |
| `--max-loras` | 1 | adaptadores distintos num mesmo lote |
| `--max-lora-rank` | 16 | escolhas: 1, 8, 16, 32, 64, 128, 256, 320, 512; dimensiona a memória, use o maior posto que for servir |
| `--max-cpu-loras` | — | cache no host (≥ `max_loras`) |
| `--lora-dtype` | `auto` | dtype do adaptador |
| `--fully-sharded-loras` | False | "By default, only half of the LoRA computation is sharded with tensor parallelism" |
| `--lora-target-modules` | — | restringe sufixos (ex. `o_proj qkv_proj`) |
| `--enable-moe-shared-loras` | False | layout "shared-outer" para experts |
| `--enable-mixed-moe-lora-format` | False | força o wrapper 2D universal de MoE (adapters 2D "megatron" vs 3D "peft", campo `is_3d_lora_weight`) |
| `--specialize-active-lora` | False | grade do kernel pelo nº de LoRAs ativos |
| `--default-mm-loras` | — | LoRA padrão por modalidade |

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct --enable-lora \
  --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider \
  --max-loras 4 --max-lora-rank 64
# carga dinâmica
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
curl -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "meu-adapter", "lora_path": "/caminho/adapter"}'
# a requisição escolhe o adaptador pelo campo "model": "sql-lora"
```

Ganchos com o Módulo 1 (`vllm-ref.md`): o hash do prefix caching inclui o **ID do LoRA** (blocos de adaptadores diferentes não se misturam); o despacho de CUDA graphs usa um `BatchDescriptor` com o campo `has_lora` (lotes com e sem LoRA caem em grafos diferentes). `[não verificado]` Não está confirmado se o LoRA funciona sobre os caminhos de quantização customizados das receitas (overlay EXL3 do GLM, MoE NVFP4 do Qwen3.8 em SM121). O caminho seguro é fundir e requantizar (§3.6) ou servir LoRA sobre uma base BF16/FP8 para modelos menores.

MLX: `mlx_lm.server --model <base> --adapter-path ./adapters`, ou o campo `"adapters"` por requisição (`mlx-copilot.md` §2.8).

---

## 4. SFT em profundidade

### 4.1 O objetivo e o que se registra

$$\mathcal L_{SFT}(\theta) = -\frac{1}{\sum_{i,t} m_{i,t}}\sum_{i}\sum_{t} m_{i,t}\,\log \pi_\theta\big(y_{i,t}\mid x_i, y_{i,<t}\big)$$

- **Deslocamento de rótulos**: a posição `t` prevê o token `t+1`; posições ignoradas recebem `labels = -100` (o `ignore_index` da entropia cruzada) `[fonte S13]`.
- **Normalização**: a média deve ser sobre **todos** os tokens de perda do lote acumulado e de todos os dispositivos. Fazer a média por micro-lote e depois a média das médias dá peso diferente a cada token e muda o resultado com o tamanho do micro-lote e com DP (o bug da acumulação de gradientes de 2024). O Tülu 3 trocou a média pela soma `[fonte S42]`; o TRL tem `average_tokens_across_devices=True` `[fonte S13]`.
- **Métricas que o `SFTTrainer` registra**: `loss`, `entropy` (entropia média da distribuição prevista nos tokens de perda), `mean_token_accuracy` (top-1 = rótulo), `aux_loss` (MoE), `grad_norm` (antes do clipping), `num_tokens`, `learning_rate` `[fonte S13]`. `[interpretação]` Entropia caindo rápido e acurácia por token perto de 1 no treino com a validação subindo = memorização. Em MoE, acompanhe também o `num-zeros` do MiMo (§3.5).
- **Variantes da perda**:
  - `loss_type="dft"` (*Dynamic Fine-Tuning*, arXiv 2508.05629) multiplica a perda de cada token pela própria probabilidade com stop-gradient, `−sg(p_θ(y_t))·log p_θ(y_t)`. O artigo argumenta que o gradiente do SFT, visto como policy gradient, carrega uma "recompensa implícita" ∝ 1/p que desestabiliza a generalização `[fonte S13, S62]`.
  - `neftune_noise_alpha` (NEFTune: ruído uniforme nos embeddings durante o treino) `[fonte S71]`.
  - `label_smoothing_factor`.

### 4.2 Mascaramento da perda

| Situação | O que entra na perda | Como (TRL) | Como (mlx-lm) |
|---|---|---|---|
| Linguagem pura (`{"text": ...}`) | todos os tokens | padrão | formato `text` |
| Prompt-completion | só a completion | `completion_only_loss` (padrão para esse formato) | `--mask-prompt` (formato `completions`) |
| Conversa | só as mensagens do assistente | `assistant_only_loss=True` (exige `{% generation %}…{% endgeneration %}` no template; o TRL corrige o template de famílias conhecidas como Qwen3) | `--mask-prompt` considera **só a última mensagem** como completion |
| Ferramentas | turnos do assistente com `tool_calls`; resultados de ferramenta (`role: tool`) fora | conversas com `tools` | formato `tools` |
| Agente (destilação) | só os tokens gerados pelo aluno; resultados de ferramentas mascarados | `DistillationTrainer(tools=[...])` | — |

`[fonte S13, S16, S20]`. Pontos que costumam dar errado `[interpretação]`:
- **O token de fim de turno tem que entrar na perda.** Se o `<|im_end|>` (Qwen) ou o equivalente fica mascarado, o modelo não aprende a parar; ele "fala pelo usuário" ou gera até `max_tokens`. O TRL pede `eos_token` alinhado ao template ("for `Qwen/Qwen2.5-1.5B`, one should set `eos_token="<|im_end|>"`") `[fonte S13]`. No Qwen3.8, o `generation_config` lista `eos_token_id [248046, 248044]` (`<|im_end|>`, `<|endoftext|>`).
- **Multi-turno com raciocínio**: vários templates (família Qwen3) **removem** o `<think>` dos turnos anteriores ao renderizar o histórico. Treinar com o histórico completo e servir com o histórico sem `<think>` (ou o contrário) cria descasamento treino/inferência. O kwarg `preserve_thinking` do Qwen3.8 (default true no card) existe para isso (`arch-qwen.md`). Para agentes, o TRL exige templates "prefix-preserving" (anexar uma mensagem não pode mudar como as anteriores são renderizadas) e troca automaticamente por um template de treino corrigido em Qwen3/DeepSeek-V3 `[fonte S16]`.
- **`--mask-prompt` do MLX treina só o último turno**; conversas longas desperdiçam os turnos anteriores do assistente. Divida a conversa em vários exemplos se cada turno importa.

### 4.3 Chat templates no treino

O capítulo 5 do Módulo 1 (tokens, chat templates, `tokenizer_config`) e o capítulo 34 (kwargs como `enable_thinking`, `reasoning_effort`, parsers de raciocínio e de tools) descrevem o template do lado do **servidor**. No treino, o template é parte do **dado**:
1. **O template do treino tem que ser byte a byte o do serviço.** Um espaço ou quebra de linha diferente antes do `<|im_start|>assistant` muda a tokenização e o modelo vê na inferência um prefixo que não viu no treino. Use `tokenizer.apply_chat_template` com o mesmo `tokenizer_config.json` que o vLLM vai usar.
2. **Checkpoints sem Jinja**: o DeepSeek-V4 usa um encoder Python (`encoding_dsv4.py`, `--tokenizer-mode deepseek_v4`) em vez de template Jinja (capítulo 5, adição). Para treinar, gere os tokens com **o mesmo encoder** e passe um dataset pré-tokenizado (`input_ids` + `labels`/`assistant_masks`, que o `SFTTrainer` aceita) `[fonte S13]`.
3. **Base sem template**: o TRL pode clonar o template de outro modelo (`chat_template_path="HuggingFaceTB/SmolLM3-3B"`) `[fonte S13]`; isso exige treinar os embeddings dos tokens novos (§3.1).
4. **Ferramentas e parsers**: o formato dos `tool_calls` no dado tem que bater com o `--tool-call-parser` que vai servir o modelo (`mimo`, `qwen3_coder`, DSML do DeepSeek…). Um SFT que ensina outro formato quebra o parser no vLLM, um problema parecido com o do capítulo 34 (temperatura corrompendo a sintaxe de DSML).

### 4.4 Dados: qualidade, diversidade e tamanho

| Referência | Tamanho | Achado |
|---|---|---|
| LIMA `[fonte S38]` | 1.000 exemplos curados (LLaMA-65B) | alinhamento "superficial": formato com pouco dado; 43 % ≥ GPT-4, 58 % ≥ Bard, 65 % ≥ DaVinci003 |
| s1 `[fonte S39]` | 1.000 perguntas (s1K) com traços de raciocínio destilados do Gemini Thinking | SFT do Qwen2.5-32B-Instruct + *budget forcing* supera o o1-preview em matemática de competição em até 27 % (MATH, AIME24); 26 min em 16 H100 |
| LIMO `[fonte S40]` | 817 exemplos | 57,1 % no AIME e 94,8 % no MATH; o limite é o conhecimento já codificado no base |
| Alpaca `[fonte S73]` | 52K (self-instruct com text-davinci-003) | instrução barata a partir de um modelo mais forte |
| Tülu 3 SFT `[fonte S42]` | 939.344 prompts | mistura por habilidade, descontaminada; o desempenho cresce com o tamanho da mistura (o TruthfulQA cai) |
| OpenThoughts3 `[fonte S41]` | 1,2 M (850K math, 250K código, 100K ciência; 16 respostas por pergunta do QwQ-32B) | OpenThinker3-7B só com SFT: 53 % AIME25, 51 % LCB, 54 % GPQA-D; >1.000 ablações de receita |
| DeepSeek-R1 distill `[fonte S50]` | ~800K | modelos R1-Distill são só SFT |
| MiMo-V2-Flash `[fonte S2]` | "milhões" de amostras | respostas geradas por checkpoints internos especializados por domínio |
| MiMo-V2.6-Distill-Qwen-9B `[fonte S76]` | **77,4 B tokens**, dos quais **27,2 B de perda** (código 23,2/7,3; cyber 11,0/4,8; geral 22,0/5,7; visual 21,2/9,4) | SFT do Qwen3.5-9B em dados gerados pelo MiMo: SWE-bench Pro 32,0 → 44,6; AutomationBench 5,0 % → 30,3 %. Destilação agêntica em escala de pré-treino continuado; só ~35 % dos tokens entram na perda (o resto é contexto, ferramentas e saídas de ambiente) |

Síntese `[interpretação]`:
- **Formato e estilo** pedem poucos exemplos muito bons (LIMA).
- **Raciocínio** pode ser *eliciado* com poucos exemplos se o base já sabe (s1, LIMO), mas o *desempenho de ponta* com SFT puro escala com dados diversos e verificados (OpenThoughts: 1,2 M, várias respostas por pergunta).
- A qualidade vem de filtragem por dificuldade e correção, deduplicação, **descontaminação** contra os benchmarks (o Tülu 3 removeu dados contaminados e mostra quais) e mistura balanceada por habilidade.
- Mais dado do mesmo tipo pode piorar outra coisa (TruthfulQA no Tülu 3). Meça uma suíte larga, não só a tarefa-alvo (§4.10).

### 4.5 Dados sintéticos e destilação de modelos mais fortes

- **Destilação off-policy (sequence-level KD)**: gerar respostas com um professor forte e fazer SFT nelas. É o que fazem o Alpaca (self-instruct com text-davinci-003), o R1-Distill, o OpenThoughts (QwQ-32B) e o próprio MiMo (checkpoints internos) `[fonte S73, S50, S41, S2]`. Variante com logits: minimizar `KL(π_T ‖ π_θ)` sobre a distribuição completa do professor em cada posição; exige o **mesmo vocabulário** (o `DistillationTrainer` do TRL exige "It must share the student's vocabulary") `[fonte S16]`.
- **Escolha do professor.** "Mais forte" não é sempre melhor: no MOPD, trocar o professor de matemática *same-origin* pelo Qwen3-235B-A22B (muito mais forte) **piorou** o aluno; o KL inicial por token subiu de ~0,04 para ~0,19 e a entropia caiu de 0,30 para 0,21 `[fonte S3]`. `[interpretação]` Professores com distribuição próxima à do aluno dão sinal aprendível; professores distantes ensinam a imitar coisas que o aluno não consegue representar (o problema do KL direto da §1.4).
- **Rejection/filtragem**: gerar várias respostas por prompt e manter só as verificadas, curtas ou bem avaliadas por juiz. Os OpenThoughts geram 16 respostas por pergunta `[fonte S41]`.
- **Licenças e termos de uso**: gerar dados com APIs fechadas para treinar modelos concorrentes pode violar os termos do provedor; verifique a licença do professor e dos dados (o MiMo é MIT; os Llama têm licença própria) `[interpretação]`.

### 4.6 Rejection-sampling fine-tuning (RFT), STaR e ReST

Algoritmo genérico ("expert iteration"):
1. Amostrar `k` respostas por prompt do modelo atual.
2. Manter as corretas (verificador, testes, juiz).
3. Fazer SFT nelas, eventualmente misturadas aos dados originais.
4. Repetir com o modelo novo.

- **STaR** (Zelikman et al., 2022): para as perguntas que o modelo erra, faz "racionalização": dá a resposta como dica, pede a justificativa e treina nela `[fonte S44]`.
- **RFT** (Yuan et al., 2023): no GSM8K, o LLaMA-7B vai de 35,9 % (SFT) para **49,3 %** combinando amostras rejeitadas de vários modelos. O ganho cresce com o nº de **caminhos de raciocínio distintos** e é maior para modelos mais fracos `[fonte S43]`.
- **ReST-EM** (Singh et al., 2023): o mesmo laço visto como EM; escala melhor que dados humanos em matemática e código `[fonte S45]`.
- **DeepSeek-R1**: o estágio 3 é RFT em escala (~600K de raciocínio + ~200K gerais) `[fonte S50]`.

`[interpretação]` RFT é um policy gradient com recompensa binária em que os negativos foram jogados fora: aproveita só a metade "positiva" do sinal, é **semi-on-policy** (os dados vêm do modelo, mas de uma versão anterior) e é tão simples quanto um SFT. É a ponte natural entre SFT e RL no módulo.

### 4.7 Destilação on-policy

**GKD** (Agarwal et al., ICLR 2024) `[fonte S8, S16]`: treina o aluno nas **próprias** sequências, com o professor dando a distribuição em cada posição, e permite trocar a divergência. No TRL, a perda é a **JSD generalizada** com interpolação `β`:

$$\mathcal L_\beta = \beta\,\mathrm{KL}(p_T\|p_M) + (1-\beta)\,\mathrm{KL}(p_S\|p_M),\quad p_M = (1-\beta)p_S + \beta p_T$$

`β = 0` → KL direto `KL(p_T‖p_S)`; `β = 1` → KL reverso `KL(p_S‖p_T)` (padrão do `DistillationConfig`: `beta 1.0`, `temperature 1.0`, `max_completion_length 512`, `learning_rate 1e-6`) `[fonte S16]`.

**OPD por amostragem (MiniLLM, Qwen3, Thinking Machines)**: com uma amostra por posição, o gradiente do KL reverso é um policy gradient com **vantagem por token** `Â_t = log π_T(y_t) − log π_θ(y_t)`. Tokens que o professor acha mais prováveis que o aluno são reforçados `[fonte S6, S66]`.

| Método | Amostragem | Sinal | Fonte |
|---|---|---|---|
| SFT | off-policy | denso | `[fonte S6]` |
| RL | on-policy | esparso (um escalar por episódio) | |
| **Destilação on-policy** | **on-policy** | **denso** | |

Números:
- **Qwen3** (Qwen3-8B; Tabela 21 do relatório): AIME'24 RL **67,6** vs destilação on-policy **74,4**; AIME'25 55,5 vs 65,5; **17.920 vs 1.800 GPU-horas** (~1/10) `[fonte S7]`.
- **Thinking Machines** (Tinker): a partir de um SFT de 400K (60 % no AIME'24), RL chega a ~68 % e a destilação on-policy a ~70 %, com redução de custo de **9–30×** contra estender o SFT off-policy (9× quando o dataset de SFT já existe ou é amortizado) `[fonte S6]`.
- **Personalização / aprendizado contínuo** (Thinking Machines, Qwen3-8B): o mid-training em documentos internos (70 % domínio) leva o IF-eval de 85 % para 79 %; a destilação on-policy recupera **83 %** mantendo o conhecimento novo (QA interno 36 % → **41 %**) `[fonte S6]`.

**No TRL (2026)** `[fonte S16, S17, S18]`:
- `DistillationTrainer` (estável desde a v1.10): professor local, geração com vLLM `colocate` ou `server`, perda JSD em pedaços (sem materializar `[s × V]` do professor), suporte a ferramentas e VLMs.
- `trl distillation --model_name_or_path ... --teacher_model_name_or_path ...` (com `--use_peft --lora_r 64 --lora_alpha 16` para LoRA).
- `AsyncDistillationTrainer` (experimental, v1.11): o professor fica num **servidor vLLM via HTTP**, só um top-k esparso de log-probs trafega (`teacher_top_k`), há geração adiantada com `max_staleness` e só FSDP2.

### 4.8 MOPD: o que significa o "MOPD" do `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`

**MOPD = Multi-Teacher On-Policy Distillation** (destilação on-policy com vários professores). Introduzida no relatório do MiMo-V2-Flash (arXiv 2601.02780) e formalizada num artigo próprio (Ma et al., arXiv 2606.30406, 2026-06-29; Peking University + Xiaomi LLM Core) `[fonte S2, S3]`.

**Pipeline de três estágios** `[fonte S3]`:
1. **SFT geral** num corpus amplo: inicialização comum.
2. **RL especializado por domínio** a partir desse SFT: um professor por domínio (matemática, instrução, SWE, ferramentas…), treinados **em paralelo**, cada um com seu algoritmo e hiperparâmetros.
3. **MOPD**: o aluno, reinicializado do SFT do estágio 1, gera rollouts. Cada prompt é **roteado** para o professor do seu domínio, que devolve log-probs por token, e o aluno minimiza o KL reverso por token. A fusão acontece no **espaço da política** (roteamento por prompt), não por média de pesos.

**Fórmulas** (MOPD, eqs. 1–5) `[fonte S3]`:
- `L_rev-KL = E_{x, y∼π_θ}[ (1/|y|) Σ_t Σ_v π_θ(v) log(π_θ(v)/π_φd(v)) ]`, com `π_φd` = professor do domínio `d` do prompt.
- Forma de policy gradient (seguindo o MiniLLM): `Â_MOPD,t = sg[log π_φd(y_t) − log π_θ(y_t)]`, clipada dos dois lados em `[−A_max, +A_max]`, e `L_PG = −E[(1/|y|) Σ_t Â^clip_t · log π_θ(y_t)]`. "This form drops directly into existing PPO/GRPO training frameworks; the advantage computation is the only change."
- Forma **top-k** (menor variância e payload pequeno para o serviço do professor): `Σ_{v ∈ TopK_k(π_φd)} [π_θ(v)·log(π_θ(v)/π_φd(v)) − π_θ(v) + π_φd(v)]`. O termo extra `π_φd − π_θ` corrige o viés do truncamento e faz o mínimo cair em `π_θ = π_φd` nos top-k.

**No MiMo-V2-Flash** (eqs. 5–9) `[fonte S2]`: o MOPD é posto como RL on-policy com a política de amostragem `μ_θ` do motor de inferência:
- `L_MOPD = −E_{y∼μ_θ}[(1/|y|) Σ_t w_t · Â_MOPD,t · log π_θ(y_t)]`;
- `w_t = sg[π_θ(y_t)/μ_θ(y_t)]` se dentro de `[ε_low, ε_high]`, senão 0 (descarta tokens com discrepância treino/inferência grande);
- **por padrão soma-se a vantagem de um ORM** (recompensa de resultado, ex. GRPO): `Â_MOPD,t = sg[log π_domínio(y_t)/π_θ(y_t)] + α·Â_ORM`.

**Infraestrutura** `[fonte S3]`: o prefill do professor é tratado como cálculo de recompensa, um **serviço assíncrono separado** do trainer. Quando uma sequência termina, dispara-se o prefill no professor dela enquanto outras sequências ainda são amostradas, e "the teacher contributed nearly no measurable wall-clock overhead".

**Resultados** `[fonte S3]`:
- Qwen3-30B-A3B, três domínios (math, IF, SWE): score normalizado **0,937** (MOPD) vs 0,882 (Mix-RL), 0,857 (Param-Merge/task arithmetic), 0,824 (Off-Policy Finetune), 0,775 (Cascade RL).
- Eficiência: MOPD chega ao platô do professor com ~25K amostras de IF e ~30K de SWE; o Mix-RL precisa de 150–180K por domínio.
- MiMo-V2-Flash (309B), professores de Math, Code, IF, SWE e Tool Use: aluno → professor → MOPD em AIME25 89,3 → 93,9 → **94,1**; HMMT25 76,9 → 82,6 → **84,4**; LCB 77,5 → 82,6 → **83,2**; IFBench 55,4 → 68,9 → 66,7 (−2,2 vs professor); SWE-Bench Verified 67,8 → 74,2 → 73,4 (−0,8); τ²-Bench 75,9 → 79,6 → **80,3**; τ²-Telecom 92,7 → 95,0 → **95,3**.
- **Same-origin importa**: professor externo mais forte piora (KL inicial ~0,19 vs ~0,04).
- **Multi-rodada**: o aluno do MOPD vira inicialização de novos professores RL, que alimentam outro MOPD, e ainda sobra ganho em math e IF.
- Hiperparâmetros do RL por domínio: GRPO on-policy com *dynamic sampling*, LR 3e-6, lote 144 com 8 rollouts por prompt (math/IF), seq 32.768 (math/IF) e 65.536 (SWE, até 50 turnos).

**O checkpoint `MiMo-V2.6-Flash-MOPD`** `[fonte S1]`:
- "This is the MOPD upgrade of the MiMo-V2.6-Flash-RL checkpoint." Mesma arquitetura: 309B total / 15B ativos, 48 camadas (39 SWA janela 128 + 9 GA), hidden 4096, 256 experts roteados com 8 ativos e sem shared experts, MTP/rascunho de 5 camadas SWA (janela 1024), contexto 1M, ViT de 681M e encoder de áudio, licença MIT.
- **MOPD2** = **Multi-Prefix Multi-Teacher On-Policy Distillation** (relatório técnico do MiMo-V2.6, §5.6, aplicado **depois** do RL misto) `[fonte S76]`. "Fuses several domain-specialized teachers into one model", estendendo-se a domínios em que a verificação no treino é difícil (desenvolvimento de jogos de longo horizonte, pesquisa científica, inteligência incorporada) `[fonte S1]`.
- Duas famílias de professores: **mixRL** (treinados em tarefas verificáveis) e **professores SFT** (treinados em demonstrações sintéticas para domínios abertos, onde é difícil desenhar recompensa).
- Três fluxos contribuem para **uma mesma atualização**: (1) **Standard MOPD**: professores mixRL supervisionam rollouts autônomos completos; (2) **Teacher-Prefix OPD**: os prefixos vêm de rollouts do professor (uma trajetória com `k` turnos dá `k` prefixos); o aluno gera **um** turno novo de cada e o professor o pontua com o mesmo histórico; (3) **SFT-Prefix OPD**: o histórico vem de uma demonstração de SFT e o aluno escreve a continuação.
- Por que prefixos `[fonte S76]`: os professores SFT foram treinados em demonstrações e cobrem pouco os históricos a que se chega depois de **desvios repetidos do aluno** em tarefas longas. Começar cada rollout de um prefixo fixo limita os desvios antes do turno amostrado. A demonstração fornece o *contexto*, não a resposta-alvo ("SFT data provide prefix contexts rather than fixed continuation targets"). `[interpretação]` É uma resposta direta ao viés de exposição da §1.3: o aluno aprende on-policy só no último turno, condicionado a históricos realistas.
- Serviço: `vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD --tensor-parallel-size 4 --trust-remote-code --gpu-memory-utilization 0.95 --max-model-len auto --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --generation-config vllm`; amostragem recomendada `temperature=1.0, top_p=0.95`.

**Estudo de caso: repetição de tool calls** `[fonte S4]`:
- **Sintoma.** O modelo repetia chamadas de ferramenta iguais ou quase iguais, "parecendo ocupado sem progredir", sem falhar explicitamente. Taxas de repetição no `-RL` por harness: OpenCode 1,02 % (Flash) / 0,54 % (Pro); Claude Code 0,27 % / 0,10 %; Codex 0,23 % / 0,07 %.
- **Causa.** Ponto cego na recompensa. A penalidade de "flooding" só disparava acima de **32** tool calls num turno (rollout interrompido, recompensa zero só naquele turno, incluindo os tokens de CoT). O comportamento abaixo desse limiar foi sendo amplificado: a partir de ~passo 15 do MixRL, os exemplos com mais de 32 chamadas começaram a crescer, e a fração de turnos com mais de oito chamadas subiu ao longo do treino.
- **Métrica.** Repetição exata dentro do turno: com `N` chamadas e `U` únicas (mesma ferramenta e argumentos iguais após canonicalizar o JSON), taxa = `(N − U)/N`. É um limite inferior da repetição observável.
- **Alternativa descartada.** Baixar o limiar para 8 chamadas por turno no MixRL suprimiu o flooding, mas só depois de ~20 passos; refazer 20 passos de MixRL custaria cerca de **US$ 2,31 milhões**. Além disso, generalizou mal: em replay, a repetição caiu de 13,45 % para 3,83 %, sem zerar.
- **Correção via MOPD.** Treinou-se um **professor RL especializado de turno único** com recompensa 0 se houvesse repetição e 1 só sem repetição nem uso errado de ferramenta, mais KL para ficar perto da política RL. **12 passos (~7.000 exemplos)** levaram a repetição em replay a zero (treino e held-out). Depois: "we rolled the final MOPD run back by five steps, introduced the repetition-specialized RL teacher, and continued training with teacher-prefix OPD". Custo total ≈ **US$ 90.000, 4 %** da alternativa MixRL, com o restante dos benchmarks estável.

**Por que isso importa para o praticante** `[interpretação]`:
1. O MOPD transforma RL multi-domínio (caro, com interferência entre domínios) em RLs pequenos e independentes + uma destilação densa.
2. Corrigir um defeito comportamental vira "treinar um professor pequeno e destilá-lo", sem refazer o RL inteiro.
3. Com o `AsyncDistillationTrainer` do TRL dá para reproduzir o estágio 3 em escala pequena: `teacher_server_urls` com uma URL por professor e uma coluna `teacher_id` por amostra. A própria doc avisa que o estágio 3 do artigo usa KL reverso (`beta=1.0`), não o `beta=0.0` (direto) padrão desse trainer `[fonte S17]`.
4. Num par de Sparks, um "MOPD de bolso" com aluno de 4–8B em LoRA e dois professores servidos em vLLM é plausível em memória (§5), mas não foi testado.

### 4.9 Esquecimento catastrófico

- **O que é**: ao ajustar para a tarefa nova, o modelo perde capacidades que tinha (instrução, segurança, raciocínio geral, formato).
- **Lei empírica (RL's Razor)**: "the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task". RL on-policy esquece menos que SFT com desempenho igual na tarefa nova porque tende a soluções de KL mínimo; o SFT pode convergir para distribuições longe do base `[fonte S46]`.
- **SFT memoriza, RL generaliza** (Chu et al., 2025): em tarefas com variantes fora da distribuição, o RL generaliza e o SFT tende a memorizar o formato de treino `[fonte S48]`.
- **LoRA aprende menos e esquece menos** (Biderman et al., 2024): em código e matemática (pré-treino continuado e instrução), o LoRA fica atrás do full FT na tarefa nova mas preserva melhor o domínio original `[fonte S47]`.
- **Mid-training destrói o comportamento de assistente** e a destilação on-policy o recupera (85 % → 45 % de IF-eval com 100 % de dados de domínio; 79 % com 70 %; 83 % após destilar) `[fonte S6]`.
- **Mitigações** `[interpretação + fontes acima]`:
  1. Misturar dados gerais ou de "replay" ao SFT de domínio (o Thinking Machines usou 70 % domínio / 30 % chat).
  2. LR menor e menos épocas; parar pelo conjunto de validação geral, não pela perda de treino.
  3. LoRA em vez de full FT quando a tarefa é estreita.
  4. Termo de KL ao base (no RL) ou destilação on-policy do base/professor depois do SFT.
  5. Média de pesos entre base e ajustado (estilo WiSE-FT) ou merge de adaptadores com peso < 1 (`add_weighted_adapter`).
  6. Medir o KL ao base em prompts gerais (capítulo 25) como métrica de "quanto me afastei".

### 4.10 Avaliação de um SFT

| Camada | Métrica | Ferramenta / nota |
|---|---|---|
| Treino | perda de treino e de validação **mascaradas** (só tokens do assistente), `mean_token_accuracy`, `entropy`, `grad_norm`, `aux_loss`/`num-zeros` (MoE) | logs do TRL; validação = split held-out do mesmo domínio |
| Distância ao base | KL(π_θ‖π_base) em prompts gerais; perplexidade num corpus geral | mesma metodologia do capítulo 25 (KLD/paridade), agora para medir **mudança desejada vs indesejada** |
| Tarefa-alvo | acurácia/pass@k/exact match na tarefa | conjunto de teste sem contaminação |
| Capacidades gerais (regressão) | IFEval/IFBench, MMLU-Pro, GPQA, GSM8K/MATH/AIME, LiveCodeBench, BFCL/τ²-Bench (ferramentas), Arena-Hard/AlpacaEval (com controle de comprimento) | lm-evaluation-harness, lighteval, evalscope; mesmos prompts e amostragem do serviço |
| Comportamento/formato | taxa de fechamento de `</think>`, taxa de EOS correto, parse rate do tool call no parser do vLLM, distribuição de comprimentos, **taxa de repetição** (a métrica `(N − U)/N` de chamadas repetidas do MiMo), recusas | scripts próprios contra o servidor real (capítulos 34 e 45) |
| Segurança | taxas de recusa correta/excessiva | suítes de segurança |
| Serviço | aceitação do rascunho MTP/EAGLE/DFlash depois do SFT; KLD após requantizar | capítulos 35–37 e 25 |

`[interpretação]` O protocolo A/B/A do capítulo 45 (mesmo contexto, mesma amostragem, alternando base e ajustado) vale também para qualidade: compare **base vs ajustado vs base** na mesma bateria e reporte intervalos. Com 30 problemas de AIME, uma diferença de 3 pontos é ruído.

---

## 5. Viabilidade prática: 2× DGX Spark, Mac M4 24 GB e frameworks de 2026

### 5.1 O orçamento de cada máquina

| Máquina | Memória | Banda | Compute | Rede | Notas |
|---|---|---|---|---|---|
| DGX Spark (GB10, sm_121, 48 SMs) | 128 GB LPDDR5x unificada; **121,69 GiB** visíveis como "GPU" (`qwen-launch.md`) | ~273 GB/s teóricos, ~230 GB/s efetivos (estimativa do Módulo 1) | ~125 TFLOPS BF16 denso (estimativa do Módulo 1); "1 petaflop of FP4" com esparsidade `[fonte S24]` | ConnectX-7 200 GbE (2 PFs PCIe Gen5 ×4 de ~126 Gb/s) | UMA: page cache conta como memória usada; `drop_caches` antes de treinar |
| 2× Spark | ~243 GiB | — | ~250 TFLOPS BF16 somados | RoCE v2 | treino multi-nó via Docker Swarm + Accelerate/FSDP no playbook `[fonte S23]` |
| Mac M4 24 GB | 24 GB unificada; MLX recomenda **19,07 GB** de working set e buffer único máximo de **14,3 GB** (medido, `mlx-copilot.md`) | 120 GB/s (especificação da Apple) `[não verificado nesta pesquisa]` | — (não medido) | — | `sudo sysctl iogpu.wired_limit_mb=N` sobe o teto (não persiste) |

`[interpretação]` Orçamento prático para o **processo de treino** num Spark: ~100–110 GiB, deixando 10–20 GiB para SO, page cache, dataloader e picos. É coerente com os ~17–18 GiB que a receita GLM deixa para o SO (`glm-launch.md`). O playbook NeMo valida as receitas `_rtx_spark` (8B LoRA, 70B QLoRA, Qwen3-8B LoRA) num container com `--memory=64g` `[fonte S23]`.

### 5.2 Matriz de viabilidade (2× DGX Spark)

Unidades: a tabela usa GB (10⁹); um Spark mostra 121,69 GiB = 130,7 GB, com ~107–118 GB práticos. Legenda: ✅ cabe com folga; 🟡 cabe com ajustes (lote 1, checkpointing, `chunked_nll`, seq ≤ 4K); ❌ não cabe; 🧪 cabe em memória, mas depende de suporte de framework/kernel não verificado para a arquitetura ou para sm_121. Base: tabelas da §2 `[cálculo próprio]`.

| Modelo | Full FT 1 nó (16 B/p) | Full FT 2 nós (FSDP, 16 B/p) | LoRA BF16 1 nó | LoRA BF16 2 nós (FSDP) | QLoRA / base 4-bit 1 nó |
|---|---|---|---|---|---|
| ≤ 3B (Llama-3.2-3B: 51 GB) | ✅ (playbook NVIDIA) | ✅ | ✅ | ✅ | ✅ |
| 4B (64 GB) | 🟡 | ✅ | ✅ | ✅ | ✅ |
| 8B (128,5 GB) | ❌ (🟡 com 8 B/p BF16 puro = 64 GB, mas veja §2.2) | ✅ 64 GB/nó (cálculo; o playbook multi-nó traz full FT de 3B e LoRA de 8B/70B) | ✅ 16 GB + ativ. | ✅ | ✅ |
| 14B (237 GB) | ❌ | ❌ a 16 B/p (118 GB ≈ 110 GiB/nó, sem folga para ativações); 🟡 com Muon (89 GB/nó) ou 8 B/p | ✅ 30 GB | ✅ | ✅ |
| Qwen3-30B-A3B MoE (488 GB de estado) | ❌ | ❌ | 🟡 61 GB de pesos + LoRA + ativ. (o Unsloth mediu 80,9 GB no B200 a 4K de contexto `[fonte S26]`) | ✅ | 🟡/🧪 bnb 4-bit não cobre experts; NVFP4 + ScatterMoE (Axolotl) em tese |
| 70B denso (1.129 GB) | ❌ | ❌ | ❌ (141 GB) | ✅ ~70,6 GB/nó + adaptadores + ativ. (playbook 70B LoRA FSDP) | ✅ ~40 GB + 3,3 + ativ. (playbooks PyTorch e NeMo) |
| gpt-oss-120b MoE (MXFP4) | ❌ | ❌ | ❌ | ❌ (234 GB) | ✅ "around 68GB" com Unsloth QLoRA `[fonte S25]` |
| Qwen3.8-Flash-Next (125,7B + PLE 51,2B + visão/MTP) | ❌ | ❌ | ❌ | ❌ (≥ 250 GB) | 🧪 2 nós: principal 4-bit ~65–71 GB + PLE FP8 ~51 GB > 1 nó; exige suporte a GR/PLE/QSA/GDN no stack de treino |
| DeepSeek-V4-Flash (284B) | ❌ | ❌ | ❌ | ❌ (569 GB) | 🧪 2 nós (~147–160 GB 4-bit); CSA/HCA/mHC sem suporte conhecido em frameworks de fine-tuning |
| GLM-5.3-Flash (320B) | ❌ | ❌ | ❌ | ❌ (640 GB) | 🧪 2 nós (~165–180 GB 4-bit) + LoRA por expert (r = 1: 3,6 GB; r = 8: 28,5 GB) |

Observações `[interpretação]`:
- **Os checkpoints de serviço do Módulo 1 não servem como base de treino.** EXL3 (treliça QTIP) e os formatos empacotados das receitas não têm kernel de backward. Para treinar sobre eles seria preciso partir do BF16/FP8 oficial e quantizar para um formato com caminho de treino (NF4 via bitsandbytes, NVFP4 via ModelOpt + ScatterMoE).
- **Para os três modelos da plataforma**, o pós-treino viável em 2× Spark é: (a) destilar/ensinar um **aluno pequeno** (4–14B) com dados ou log-probs gerados pelo modelo grande servido na receita do Módulo 1. É o padrão do `MiMo-V2.6-Distill-Qwen-9B` e do MOPD, e o professor servido em vLLM é exatamente o que o `AsyncDistillationTrainer` consome via HTTP. (b) LoRA experimental sobre base 4-bit do MoE grande, só com suporte de framework.
- **Tempo**: 10M tokens de SFT custam ~2,7 h num 8B LoRA, ~3,6 h num 8B full FT e ~23,5 h num 70B QLoRA por Spark a 40 % de MFU (§2.7). Com 2 nós em data parallel, divida por ~2 (a rede custa poucos %, §2.5).
- **MoE no Spark é limitado por compute, não por banda** `[cálculo próprio]`: um micro-lote de 16K tokens toca quase todos os experts. No Qwen3-30B-A3B BF16 isso é ler ~61 GB × 3 passes (forward, recomputação, backward) ≈ 183 GB ≈ 0,8 s a 230 GB/s, contra `6 × 3,35e9 × 16.384 ≈ 3,3e14` FLOPs ≈ 6,6 s a 50 TFLOPS. É o contrário do decode do Módulo 1, que é limitado por banda, e por isso treinar é uma boa forma de aproveitar o Spark.

### 5.3 Mac M4 24 GB com MLX

Estimativas `[cálculo próprio]` (pesos MLX 4-bit g64 = 4,5 bits/param; LoRA r = 8 em todas as lineares das últimas 16 camadas, que é o padrão do `mlx_lm.lora` sem `keys`; ativações só das camadas treinadas; logits FP32 como limite superior):

| Caso | Pesos | LoRA (16 B/p) | Ativações | Logits | **Total** | vs 19,07 GB |
|---|---|---|---|---|---|---|
| Qwen3-4B 4-bit, s 2048, b 4, sem ckpt | 2,26 | 0,12 | 11,4 | 4,98 | **18,8 GB** | 🟡 no limite |
| Qwen3-4B 4-bit, s 2048, b 4, `--grad-checkpoint` | 2,26 | 0,12 | 1,38 | 4,98 | **8,7 GB** | ✅ |
| Qwen3-8B 4-bit, s 2048, b 4, sem ckpt | 4,61 | 0,16 | 18,3 | 4,98 | **28,0 GB** | ❌ |
| Qwen3-8B 4-bit, s 2048, b 1, ckpt | 4,61 | 0,16 | 0,55 | 1,24 | **6,6 GB** | ✅ |
| Qwen3-14B 4-bit, s 2048, b 1, ckpt | 8,32 | 0,21 | 0,69 | 1,24 | **10,5 GB** | ✅ |
| Qwen3-14B 4-bit, s 4096, b 1, ckpt | 8,32 | 0,21 | 1,38 | 2,49 | **12,4 GB** | ✅ |
| Qwen3-30B-A3B 4-bit | 17,2 | — | — | — | > 17,2 GB | ❌ na prática |
| Full FT Qwen3-0.6B BF16 (8 B/p: Adam do MLX no dtype do parâmetro), s 2048, b 4, ckpt | 4,8 | — | 0,75 | 4,98 | **10,5 GB** | ✅ |
| Full FT Qwen3-1.7B BF16, s 2048, b 1, ckpt | 13,6 | — | 0,38 | 1,24 | **15,2 GB** | 🟡 |

Notas: o otimizador do MLX cria o estado com `zeros_like(param)`, no dtype do parâmetro `[fonte S21]`. Os adaptadores LoRA do `mlx_lm` nascem em FP32 (`mx.random.uniform`/`mx.zeros`), então valem os 16 B/p. Os logits podem ser BF16 na prática, e então a coluna cai à metade. A velocidade no M4 não foi medida; a única referência publicada é "about 250 tokens-per-second" num **M1 Max 32 GB** com Mistral-7B, `--batch-size 1 --num-layers 4` `[fonte S20]`. O M4 base tem GPU menor que o M1 Max: espere menos e meça.

**Comandos** (`mlx-lm 0.32.0`; treino exige `pip install "mlx-lm[train]"`) `[fonte S20]`:

```bash
# dados: ./data/train.jsonl (obrigatório), valid.jsonl (opcional), test.jsonl (para --test)
# formato chat: {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --train --data ./data \
  --iters 600 --batch-size 2 --num-layers 16 --mask-prompt --grad-checkpoint \
  --max-seq-length 2048 --learning-rate 1e-5 --adapter-path ./adapters
# (modelo quantizado ⇒ QLoRA automaticamente; --fine-tune-type dora | full; --optimizer adam|adamw|muon|sgd|adafactor)
mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path ./adapters --data ./data --test
mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path ./adapters --prompt "..."
mlx_lm.fuse --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path ./adapters --save-path ./fused_model
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path ./adapters --port 8080
```

Config YAML (`-c config.yaml`; as chaves de LoRA **só** existem no arquivo) `[fonte S20]`:

```yaml
model: "mlx-community/Qwen3-4B-Instruct-2507-4bit"
train: true
fine_tune_type: lora          # lora | dora | full
optimizer: adamw              # padrão do código: adam
data: "./data"
num_layers: 16                # -1 = todas
batch_size: 2
iters: 1000
learning_rate: 1e-5
grad_accumulation_steps: 4
max_seq_length: 2048
grad_checkpoint: true
mask_prompt: true
lora_parameters:
  # sem "keys": todas as Linear/SwitchLinear (experts)/Embedding das camadas treinadas
  # o exemplo oficial usa só ["self_attn.q_proj", "self_attn.v_proj"]; o LoRA Without Regret recomenda todas
  rank: 8
  scale: 20.0                 # NÃO é alpha: multiplica B·A diretamente
  dropout: 0.0
lr_schedule:
  name: cosine_decay
  warmup: 100
  warmup_init: 1e-7
  arguments: [1e-5, 1000, 1e-7]
```

Para DPO, GRPO e afins no Mac existe o pacote comunitário `mlx-lm-lora` (Gökdeniz Gülmez, MIT; 3.1.3 em 2026-09-16) `[fonte: PyPI; não avaliado aqui]`.

### 5.4 Frameworks em 2026

| Framework | Versão | Para quê | Pontos fortes | No Spark | No Mac |
|---|---|---|---|---|---|
| **TRL** | 1.14.1 | SFT, DPO, KTO, reward, RLOO, GRPO, destilação (on-policy, assíncrona multi-professor) | referência do ecossistema HF; CLI `trl sft`, `trl dpo`, `trl kto`, `trl reward`, `trl rloo`, `trl grpo`, `trl distillation`; perfis Accelerate; vLLM colocate/server; `chunked_nll` | playbook PyTorch (`nvcr.io/nvidia/pytorch:25.11-py3`) usa TRL + PEFT + bitsandbytes | só com backend MPS (fora do escopo) |
| **Unsloth** | 2026.9.14 | LoRA/QLoRA/full/RL rápidos numa GPU | kernels Triton, MoE 12× (agregado), contextos longos | playbook (pytorch:25.11, `trl==0.26.1`, `pip install --no-deps unsloth unsloth_zoo bitsandbytes`); gpt-oss-120b QLoRA ~68 GB | não |
| **Axolotl** | 0.20.0 | treino por YAML: full, LoRA, QLoRA, QAT, FP8, DPO/KTO/ORPO, GRPO | FSDP2/DeepSpeed, SP, **EP**, ScatterMoE, NVFP4 MoE LoRA com merge | sem playbook oficial; suporte aarch64/sm_121 `[não verificado]` | não |
| **LLaMA-Factory** | 0.9.5 | SFT/DPO/KTO/PPO/reward por YAML + WebUI | muitos templates (`qwen3_nothink`…), export/merge | playbook (venv + PyTorch `cu130`) | não |
| **NeMo AutoModel** | 0.6.0 (container 26.08) | SFT/PEFT HF-nativo, de 1 GPU a multi-nó | receitas `_rtx_spark` validadas | playbook (commit fixado `3b5663a6…`) | não |
| **Megatron-Bridge / NeMo-RL** | — | pré-treino/SFT em escala (TP/PP/EP/CP, FP8/NVFP4) e RL | receita `bf16_with_nvfp4_mixed` | playbook NVFP4 é para GB300, não Spark | não |
| **verl** | 0.9.1 | RL (PPO, GRPO…), SFT | FSDP/FSDP2/Megatron; rollout vLLM/SGLang; LoRA + router replay no Megatron | `[não verificado]` | não |
| **OpenRLHF** | 0.11.2 | RL com Ray + vLLM | PPO, REINFORCE++, GRPO, RLOO; VLM; assíncrono | `[não verificado]` | não |
| **mlx-lm** | 0.32.0 | LoRA/DoRA/full no Apple Silicon | QLoRA automático em modelo quantizado; LoRA em experts (`SwitchLinear`); Muon | — | ✅ |
| **torchtune** | — | — | **descontinuado** (2025-07-15) | — | — |

`[fonte S13–S20, S23, S25–S31]`

#### 5.4.1 Comandos e configs exatos

**PyTorch + TRL no Spark (playbook NVIDIA)** `[fonte S23]`:

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker run --gpus all -it --rm --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${PWD}:/workspace" -w /workspace nvcr.io/nvidia/pytorch:25.11-py3
pip install transformers peft datasets trl bitsandbytes "torchao>=0.16.0"
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets
python Llama3_3B_full_finetuning.py                 # full SFT, Llama 3.2 3B
python Llama3_8B_LoRA_finetuning.py --dataset_size 100 --num_epochs 1 --batch_size 2
python Llama3_70B_qLoRA_finetuning.py               # QLoRA 70B num Spark
# flags comuns: --dtype bfloat16 --batch_size --seq_length 2048 --gradient_accumulation_steps
#               --learning_rate --gradient_checkpointing --lora_rank 8 --dataset_size 512 --output_dir
```

**Dois Sparks (FSDP, playbook)**: Docker Swarm com o GPU anunciado (`node-generic-resources` no `/etc/docker/daemon.json`), `docker stack deploy -c docker-compose.yml finetuning-multinode`, depois em cada nó:

```bash
# configs/config_fsdp_lora.yaml: machine_rank 0|1, main_process_ip <IP do link>, main_process_port <porta>
docker exec -e HF_TOKEN="$HF_TOKEN" -it "$FINETUNING_CONTAINER" bash -c '
  bash /workspace/install-requirements;
  accelerate launch --config_file=/workspace/configs/config_fsdp_lora.yaml /workspace/Llama3_70B_LoRA_finetuning.py'
```

O `docker-compose.yml` do playbook fixa `UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME` e `GLOO_SOCKET_IFNAME` em `enp1s0f1np1`; troque pela sua interface `[fonte S23]`. `[interpretação]` Valem os cuidados de RoCE do Módulo 1 (capítulo 39, `qwen-launch.md`): `NCCL_IB_HCA` certo, `NCCL_IB_GID_INDEX=3`, MTU 9000 e as **duas** PFs da porta para chegar a 200 Gb/s. Teste com `nccl-tests` antes de treinar.

**TRL CLI** (SFT LoRA de 8B num Spark; sintaxe da CLI `[fonte S15]`; campos do `ModelConfig`/`SFTConfig` `[fonte S13]`; valores são escolha minha):

```bash
trl sft \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_name trl-lib/Capybara \
  --dtype bfloat16 \
  --use_peft --lora_r 16 --lora_alpha 32 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --assistant_only_loss True \
  --max_length 4096 \
  --learning_rate 1e-4 --lr_scheduler_type cosine \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --output_dir qwen3-8b-lora-sft
# multi-GPU/multi-nó: --accelerate_config fsdp2 (ou caminho de um YAML do Accelerate) --num_processes N
```

Cuidados `[interpretação]`:
- Sem `--dtype bfloat16`, o `SFTTrainer` carrega em FP32 (§2.1) e dobra a memória dos pesos congelados.
- `assistant_only_loss` exige `{% generation %}` no template (o TRL corrige o do Qwen3).
- Com PEFT, o `chunked_nll` padrão não é compatível ("Not compatible with … PEFT"); confira no log qual perda foi escolhida.
- No `transformers` 5.x a assinatura mostra `warmup_steps: float` (frações < 1 viram razão) e não lista `warmup_ratio`; confira com `trl sft --help`.

**Destilação on-policy com TRL** `[fonte S16, S17]`:

```bash
trl distillation \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name trl-lib/ultrafeedback-prompt \
  --learning_rate 2e-4 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
  --output_dir distilled-model --num_train_epochs 1 \
  --use_peft --lora_r 64 --lora_alpha 16
```

```python
# MOPD "de bolso": professores servidos com vllm serve, roteados por coluna teacher_id
from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer
trainer = AsyncDistillationTrainer(
    model="Qwen/Qwen3-4B",
    args=AsyncDistillationConfig(
        teacher_server_urls=["http://teacher-math:8000", "http://teacher-code:8000"],
        teacher_top_k=16,
        beta=1.0,   # KL reverso, como no estágio 3 do MOPD (o padrão deste trainer é 0.0)
    ),
    train_dataset=dataset,   # coluna "prompt" + coluna "teacher_id"
)
trainer.train()
```

`[interpretação]` Nos Sparks: um nó serve o professor (por exemplo, a própria receita do Módulo 1 com um modelo de mesmo tokenizer) e o outro treina o aluno. Restrições: todo professor tem que **compartilhar o tokenizer do aluno**, e só FSDP2 é suportado `[fonte S17]`. Os modelos da plataforma têm vocabulários próprios (GLM 154.880, Qwen 248.320, DeepSeek 129.280), então o aluno tem que ser da **mesma família de tokenizer** (por exemplo, um Qwen3.5/3.6 pequeno para o Qwen3.8, se o tokenizer for o mesmo; confira o `tokenizer.json`).

**Unsloth** (snippet da doc de MoE) `[fonte S26]`:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # MoE doesn't support bnb 4bit yet
)
model = FastLanguageModel.get_peft_model(
    model, r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_alpha = 32, use_gradient_checkpointing = "unsloth", random_state = 3407,
)
```

Instalação no Spark (playbook): container `nvcr.io/nvidia/pytorch:25.11-py3`, `pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"` e `pip install --no-deps unsloth unsloth_zoo bitsandbytes`; o teste é `python test_unsloth.py` (60 passos) `[fonte S23]`.

**Axolotl** `[fonte S27]`:

```bash
uv pip install --no-build-isolation axolotl[deepspeed]
axolotl fetch examples
axolotl train examples/llama-3/lora-1b.yml
axolotl merge-lora examples/llama-3/lora-1b.yml
```

```yaml
# trecho de examples/llama-3/lora-1b.yml
base_model: NousResearch/Llama-3.2-1B
datasets:
  - path: teknium/GPT4-LLM-Cleaned
    type: alpaca
adapter: lora
sequence_len: 2048
sample_packing: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [gate_proj, down_proj, up_proj, q_proj, v_proj, k_proj, o_proj]
gradient_accumulation_steps: 2
micro_batch_size: 2
num_epochs: 1
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 0.0002
bf16: auto
gradient_checkpointing: true
attn_implementation: flash_attention_2
warmup_ratio: 0.1
# NVFP4 MoE LoRA: use_scattermoe: true (W4A16, sm80+) | use_sonicmoe: true (W4A4, SM100);
#                 nvfp4_merge_aware: true; lora_target_parameters: [...]
```

**LLaMA-Factory** (playbook do Spark) `[fonte S23, S28]`:

```bash
python3 -m venv factoryEnv && source factoryEnv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
pip install -e ".[metrics]"
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

```yaml
# examples/train_lora/qwen3_lora_sft.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

A saída de exemplo do playbook (Qwen3-4B LoRA, 3 épocas): `train_runtime 0:14:32`, `train_samples_per_second 3.749`, `train_loss 0.9993` `[fonte S23]`.

**NeMo AutoModel** (playbook; container 26.08 + commit `3b5663a6fdf0fee03abbe82e431ca3a7ba6dd8e2`) `[fonte S23]`:

```bash
docker run --gpus all --ipc=host --memory=64g --memory-swap=64g --ulimit memlock=-1 \
  -it --ulimit stack=67108864 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$HOME/Automodel:/opt/Automodel" -w /opt/Automodel --entrypoint /usr/bin/bash \
  --rm nvcr.io/nvidia/nemo-automodel:26.08
automodel examples/llm_finetune/llama3_1/llama3_1_8b_squad_peft_rtx_spark.yaml --nproc-per-node 1 --step_scheduler.max_steps 20
automodel examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad_peft_qlora_rtx_spark.yaml --nproc-per-node 1 --step_scheduler.max_steps 20
automodel examples/llm_finetune/qwen/qwen3_8b_squad_rtx_spark.yaml --nproc-per-node 1 --step_scheduler.max_steps 20
```

**verl** (SFT e PPO, doc GSM8K) `[fonte S30]`:

```bash
torchrun -m verl.trainer.sft_trainer \
  data.train_files=$HOME/data/gsm8k/train.parquet data.val_files=$HOME/data/gsm8k/test.parquet \
  data.messages_key=messages data.micro_batch_size_per_gpu=8 \
  model.path=deepseek-ai/deepseek-coder-6.7b-instruct \
  trainer.project_name=gsm8k-sft trainer.total_epochs=4 trainer.logger='["console"]'

python3 -m verl.trainer.main_ppo \
  data.train_files=... data.train_batch_size=1024 data.max_prompt_length=512 data.max_response_length=512 \
  actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  critic.model.path=deepseek-ai/deepseek-llm-7b-chat algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.total_epochs=15
```

### 5.5 Receita mínima ponta a ponta no Spark (fecha o laço com o Módulo 1)

`[interpretação; cada passo usa ferramentas verificadas acima]`
1. **Dados**: conversas no formato `messages` (com `tools` se houver), renderizadas com o **mesmo** chat template do serviço. Deduplicar, descontaminar contra o seu conjunto de avaliação e separar um held-out.
2. **Baseline**: rode a bateria de avaliação (§4.10) no modelo base servido pelo vLLM com a amostragem recomendada.
3. **SFT**: LoRA r 16–64 em todas as lineares, LR ~1e-4 (≈10× o do full), 1–3 épocas, `assistant_only_loss`, packing BFD se houver FlashAttention, checkpointing. Acompanhe perda de validação mascarada, entropia, `grad_norm` e, em MoE, `aux_loss`.
4. **Avaliação A/B/A**: base → ajustado → base, na mesma bateria; inclua o KL ao base em prompts gerais e as métricas de formato (EOS, `</think>`, parse de tool call, repetição).
5. **Servir**: (a) `vllm serve <base> --enable-lora --lora-modules meu=./adapter --max-lora-rank 64`; ou (b) fundir em BF16 → requantizar (NVFP4/FP8/EXL3) → medir o KLD contra o BF16 fundido (capítulo 25) e a métrica de tarefa (o efeito do fine-tune sobreviveu?) → servir com a receita do Módulo 1.
6. **Especulação**: meça a aceitação do rascunho (MTP/EAGLE/DFlash) no modelo ajustado; se caiu, re-treine o rascunho nos logs do novo modelo, como o MiMo fez com o DFlash.

---

## 6. Figuras interativas propostas (com os números)

Todas as figuras usam as fórmulas das §§2–5. Os números padrão abaixo já foram calculados (§8) e podem ser usados como valores iniciais e como teste de regressão do JS.

**F1 — Calculadora de memória do treino** (barra empilhada vs orçamento)
- Entradas: `N` (params totais), `N_ativos`; regime de estado (16 / 12 / 8 / 6 B/p); método (full / LoRA / QLoRA); `r` e módulos do LoRA (atenção, MLP, experts com posto por expert); formato da base congelada (BF16 2 B/p, NF4+DQ 0,516 B/p, MLX 4-bit 0,5625 B/p, FP8 1 B/p); `s`, `b`, checkpointing (sim/não); vocabulário `V` e logits FP32/BF16/em pedaços; ZeRO 0/1/2/3 e `N_d`.
- Saídas: pesos | grads | mestres | otimizador | ativações | logits, contra as linhas "Spark 121,7 GiB", "prático ~105 GiB", "2 Sparks", "Mac 19,07 GB".
- Pré-ajustes (verificar o JS contra estes): Llama-3.1-8B full 16 B/p = 128,5 GB; o mesmo com ZeRO-3 N = 2 = 64,2 GB/nó; Llama-3.3-70B QLoRA r16 = 39,5 + 3,3 + 3,25·b GB (+ logits 1,05·b); Qwen3-8B 4-bit no Mac, b 1, s 2048, ckpt = 6,6 GB; Qwen3-4B 4-bit, b 4, sem ckpt = 18,8 GB.

**F2 — O peso que não se move (BF16)**
- Entradas: `w₀` (padrão 0,02), `lr` (1e-5), nº de passos, sinal da atualização aleatório ou fixo; três trilhas: FP32, BF16 com arredondamento ao par (RNE), BF16 com arredondamento estocástico.
- Mostra: ulp = `2^(⌊log₂|w|⌋ − 7)`; com `w = 0,02`, ulp = 1,22e-4 e meio ulp = 6,1e-5, então passos de 1e-5 nunca mudam o peso BF16; com `w = 0,002` o passo de 1e-5 vira um salto de 1,53e-5. A trilha estocástica acompanha a FP32 em média.

**F3 — ZeRO em barras**
- 8,03B, `N_d = 2`: DDP 128,5 | Z1 80,3 | Z2 72,3 | Z3 64,2 GB. Artigo (7,5B, `N_d = 64`): 120 | 31,4 | 16,6 | 1,9 GB.
- Controle de comunicação por passo: DDP/Z1/Z2 = `2Ψ`, Z3 = `3Ψ` (elementos); com 20 GB/s, all-reduce BF16 de 8B ≈ 0,8 s vs 63 s de compute para 131K tokens em 2 Sparks a 50 TFLOPS cada.

**F4 — Contador de parâmetros LoRA**
- Entradas: `d_in`, `d_out`, `r`, quais módulos; para MoE, `E`, camadas MoE e "posto por expert = r/k_ativos".
- Pré-ajustes: Llama-3.1-8B r16 all = 41.943.040 (0,52 %); só atenção 13,6 M; só MLP 28,3 M; Llama-3.3-70B r16 = 207,1 M; Qwen3-30B-A3B por expert r 2 = 103,8 M; GLM-5.3-Flash por expert r 8 = 1,78 B.

**F5 — Tempo de treino e o teto de FLOPs**
- `t = tokens × k × N_ativos / (MFU × pico × nós)`, com `k ∈ {4 (LoRA), 6 (full ou LoRA+ckpt), 8 (full+ckpt)}`; deslizador de MFU 10–60 %; pico 125 TFLOPS BF16 (editável).
- Sobreposição: os "peak tokens/s" da NVIDIA (82.739 / 53.657 / 5.079) como pontos fora do teto (exigiriam 1,60 / 1,72 / 1,43 PFLOP/s).
- Pré-ajustes a 40 %: 8B LoRA+ckpt 1.038 tok/s (2,7 h por 10M tokens); 8B full+ckpt 778 tok/s (3,6 h); 70B QLoRA+ckpt 118 tok/s (23,5 h).

**F6 — KL direto vs reverso**
- Alvo bimodal (mistura de duas gaussianas) e aluno unimodal ajustado por (a) `KL(p‖q)` (SFT/MLE: cobre os dois modos e põe massa no vale) e (b) `KL(q‖p)` (RL-com-KL / destilação on-policy: escolhe um modo). Deslizador `β` da JSD generalizada do TRL (0 = direto, 1 = reverso).

**F7 — Packing**
- Sequências de comprimentos variados empacotadas por BFD vs "wrapped"; a máscara de atenção bloco-diagonal (varlen/`position_ids` reiniciados) vs a máscara causal única (contaminação); contador de tokens de padding desperdiçados.

**F8 — Máscara de perda numa conversa real**
- Transcrição tokenizada (template estilo Qwen: `<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`) com cor por papel. Alternar "texto inteiro", "só completion", "só assistente", "assistente + EOS", "mlx `--mask-prompt` (só o último turno)". Mostrar o erro clássico: EOS mascarado → o modelo não aprende a parar.

**F9 — Vantagem por token da destilação on-policy**
- Uma resposta do aluno com cada token colorido por `log π_T − log π_S` (verde = professor gosta mais, vermelho = menos), com clipping `[−A_max, A_max]`; alternar "policy gradient (1 amostra)" vs "top-k" (eq. 5 do MOPD).

**F10 — MOPD e MOPD2 (diagrama)**
- Estágios: SFT → RL por domínio (paralelo) → aluno gera → roteador por prompt → prefill assíncrono do professor → vantagem densa (+ `α·Â_ORM`) → atualização.
- MOPD2: os três fluxos (rollout completo; prefixos de trajetória do professor; prefixos de demonstração SFT, `k` turnos → `k` prefixos).
- Painel com a tabela do MiMo-V2-Flash (aluno/professor/MOPD) e o caso da repetição de tool calls (12 passos, ~7.000 exemplos, US$ 90 mil vs US$ 2,31 mi).

**F11 — Esquecimento e recuperação**
- Barras do experimento do Thinking Machines (Qwen3-8B): QA interno / IF-eval = 18/85 (base), 43/45 (mid-train 100 %), 36/79 (mid-train 70 %), 41/83 (+ destilação on-policy). Texto: o RL's Razor (esquecimento ∝ KL na tarefa nova).

**F12 — Matriz de viabilidade**
- Heatmap modelo × método × hardware (tabela da §5.2 e da §5.3), com tooltip da conta.

---

## 7. Proposta de capítulos e ganchos com o Módulo 1

Sugestão de capítulos para a parte "fundamentos + SFT + sistemas" do Módulo 2 (os ids seguem o estilo do `curriculum.json`):

| id sugerido | Título | Conteúdo (seções desta nota) | Pré-requisitos (Módulo 1) | Figuras |
|---|---|---|---|---|
| `pos-treino-mapa` | Do modelo base ao modelo de raciocínio: o que é pós-treino | §1 inteira (pipeline, on/off-policy, KL, taxonomia, MOPD em uma frase) | `aprender-com-dados`, `geracao-amostragem` | F6, F10 (versão simples) |
| `conta-do-treino` | A conta do treino: memória, ativações, ZeRO/FSDP e paralelismos | §2.1–2.7 | `numeros-no-computador`, `paralelismos`, `nccl-rdma`, `memoria-unificada` | F1, F2, F3, F5 |
| `otimizadores-e-lr` | Otimizadores e hiperparâmetros: AdamW, Muon, LR, lotes e packing | §2.8–2.10 | `aprender-com-dados` (Muon), `tensores-algebra` | F2, F7 |
| `lora-qlora-dora` | LoRA, QLoRA e DoRA: ajustar sem reescrever o modelo | §3.1–3.4, 3.6, 3.7 | `tensores-algebra` (baixo posto), `mla`, `quantizacao-fundamentos`, `vllm-arquitetura` | F4, F1 |
| `pos-treino-moe` | Pós-treino de MoE: LoRA em experts, roteador, estabilidade e consistência treino/inferência | §3.5 + MiMo (num-zeros, roteador congelado, R3, QDQ MXFP4, rascunho DFlash re-treinado) | `mixture-of-experts`, `fp8-fp4-microescala`, `dflash-dspark` | F4 |
| `sft-a-fundo` | SFT a fundo: perda, máscara, chat templates e dados | §4.1–4.4 | `tokens-embeddings` (chat templates), `api-tools-estruturado` | F8, F7 |
| `sinteticos-e-destilacao` | Dados sintéticos, RFT e destilação on-policy (do GKD ao MOPD) | §4.5–4.8 | `geracao-amostragem`, `medindo-qualidade` | F9, F10, F6 |
| `esquecimento-avaliacao` | Esquecimento catastrófico e como avaliar um fine-tune | §4.9–4.10 | `medindo-qualidade`, `benchmarking` | F11 |
| `laboratorio-finetune` | Laboratório: fine-tuning no DGX Spark e no Mac | §5 inteira | `memoria-unificada`, `receita-do-zero` | F1, F12 |

**Ganchos explícitos com o Módulo 1** (onde citar o capítulo antigo):
- `numeros-no-computador` (BF16, RNE) → por que existem pesos mestres FP32 (§2.2).
- `tokens-embeddings` e `api-tools-estruturado` → o chat template e os parsers viram parte do **dado** de SFT (§4.3); formato de tool call no dado = parser no serviço.
- `mla` (`q_lora_rank`, `kv_lora_rank`) → a mesma fatoração de baixo posto, agora no *update* (§3.1).
- `quantizacao-fundamentos`, `fp8-fp4-microescala`, `hadamard-trelica-exl3` → NF4 do QLoRA; NVFP4 MoE LoRA; EXL3 não treinável; requantizar depois do merge (§3.2, §3.6).
- `medindo-qualidade` (KLD, paridade) → KL ao base como métrica de esquecimento; KLD após requantizar; `μ ≠ π` no RL (§1.3, §4.9, §4.10).
- `memoria-unificada` → offload não cria memória em UMA; `drop_caches` antes de treinar (§2.4).
- `paralelismos`, `nccl-rdma` → no treino, a razão compute/comunicação é muito maior que no decode; FSDP em 2 Sparks custa poucos % (§2.5, §2.6).
- `especulacao-fundamentos`, `mtp-eagle-vocab`, `dflash-dspark` → o rascunho precisa acompanhar a política: re-treinar MTP/DFlash após SFT/RL (MiMo: +31,3 % de aceitação) (§3.5).
- `abliteracao` → edição de pesos sem treino vs fine-tuning; a direção de recusa que a abliteração remove é criada justamente no pós-treino de segurança `[interpretação]`.
- `receita-do-zero` → o `MiMo-V2.6-Flash-MOPD` (309B/15B, com quantizações EXL3/NVFP4 da comunidade) é candidato a estudo de caso de "receita para modelo recém-lançado" (§1.6).

---

## 8. Script de cálculo (reprodutível)

Todos os números marcados `[cálculo próprio]` saem deste script (Python 3, sem dependências):

```python
import struct, math
GB = 1e9

def dense_params(h, L, nh, nkv, hd, inter, V, tied=False):
    attn = h*nh*hd + 2*h*nkv*hd + nh*hd*h
    per = attn + 3*h*inter + 2*h
    return per*L + V*h*(1 if tied else 2) + h

def lora_dense(h, L, nh, nkv, hd, inter, r):
    q = r*(h+nh*hd); k = v = r*(h+nkv*hd); o = r*(nh*hd+h)
    g = u = r*(h+inter); d = r*(inter+h)
    return L*(q+k+v+o+g+u+d)

def act_bytes(s, b, h, L, ckpt):          # BF16, FlashAttention (Korthikanti: 34·s·b·h por camada)
    return (2*s*b*h*L + 34*s*b*h) if ckpt else 34*s*b*h*L

def zero(psi, Nd, K=12):
    return {"DDP": (4+K)*psi, "Z1": (4+K/Nd)*psi, "Z2": (2+(2+K)/Nd)*psi, "Z3": (4+K)*psi/Nd}

def bf16(x):                               # arredondamento ao par para BF16
    b = struct.unpack('<I', struct.pack('<f', x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', b))[0]

llama8 = dict(h=4096, L=32, nh=32, nkv=8, hd=128, inter=14336)
P8 = dense_params(**llama8, V=128256)                       # 8.030.261.248
print(P8, lora_dense(**llama8, r=16))                       # 41.943.040
print({k: round(v/GB, 1) for k, v in zero(8.03e9, 2).items()})   # 128.5 / 80.3 / 72.3 / 64.2
print(act_bytes(2048, 1, 4096, 32, False)/GB, act_bytes(2048, 1, 4096, 32, True)/GB)  # 9.13 / 0.82
print(4 + 8/64 + 32/(64*256))                               # NF4 + quantização dupla = 4.127 bits
for n, p, k in [("3B full", 3.213e9, 6), ("8B LoRA", 8.03e9, 4), ("70B QLoRA", 70.554e9, 4)]:
    print(n, 125e12/(k*p), "tok/s de teto a 125 TFLOPS")
print(bf16(bf16(0.02) + 1e-5) - bf16(0.02))                 # 0.0 -> a atualização se perde
# MoE: LoRA por expert = camadas_MoE × E × 3 × r × (h + interm.)
print(48*128*3*2*(2048+768)/1e6, 42*288*3*8*(4096+2048)/1e9)  # 103.8 M (Qwen3-30B-A3B r2), 1.78 B (GLM r8)
```

---

## 9. Incertezas e pontos não verificados

1. **Pico de compute do GB10 e MFU em treino.** ~125 TFLOPS BF16 denso é a estimativa usada no Módulo 1 (vinda dos relatórios das receitas), não um microbenchmark. A MFU real de SFT no Spark não foi medida. Os "peak tokens/s" de fine-tuning da NVIDIA não fecham com a contabilidade de FLOPs (§2.7); a métrica não está definida no post.
2. **Kernels de treino no sm_121**: FlashAttention 2/3 (necessária para `padding_free`/packing sem contaminação no TRL), Transformer Engine FP8, kernels NVFP4 de treino e o SonicMoE (W4A4, só SM100) não foram verificados no GB10. O playbook NVFP4 é para GB300. Os playbooks QLoRA rodam bitsandbytes no container `pytorch:25.11`, o que indica suporte a NF4, mas não medi.
3. **Axolotl, verl e OpenRLHF no Spark (aarch64 + CUDA 13)**: sem playbook oficial; não testados.
4. **LoRA no vLLM sobre os caminhos de quantização das receitas** (overlay EXL3 do GLM, MoE NVFP4 do Qwen3.8 em SM121): não verificado. Merge + requantização é o caminho seguro.
5. **Mac M4**: banda de 120 GB/s é especificação da Apple, não medida; o throughput de `mlx_lm.lora` no M4 não foi medido; o comportamento de memória do autodiff do MLX (só ativações das camadas treinadas) é suposição.
6. **Ativações**: `34·s·b·h` por camada é aproximação (Korthikanti para GPT; recontada para Llama). Implementações reais (HF eager vs SDPA, buffers do `torch.compile`) guardam mais. Use a F1 como ordem de grandeza e meça com `torch.cuda.max_memory_allocated`.
7. **Adaptadores em FP32 no PEFT por padrão** (`autocast_adapter_dtype`): lembrado da API, não reconferido nesta pesquisa.
8. **Llama 3**: o nº exato de rodadas SFT+RS+DPO não foi reconferido.
9. **TRL CLI**: a ausência de `warmup_ratio` na assinatura do `SFTConfig` (transformers 5.x) foi inferida da doc gerada; confira com `--help`.
10. **Tokenizers**: a compatibilidade de tokenizer entre o Qwen3.8 (248.320) e alunos Qwen menores precisa ser checada no `tokenizer.json` antes de qualquer destilação por logits.
11. **Muon no SFT dos modelos da plataforma**: recomendação derivada do Moonlight (descasamento pré-treino/SFT), não testada no Qwen3.8 nem no DeepSeek-V4.
12. **Merge de QLoRA** (adaptador treinado contra NF4 fundido em BF16): o descasamento é argumento meu; meça o KLD.
13. **MOPD em pequena escala**: nenhuma reprodução pública encontrada além do exemplo `async_distillation_mopd.py` do TRL; os números do MOPD vêm de Qwen3-30B-A3B e MiMo-V2-Flash (escala industrial).
14. **Quantizações comunitárias do MiMo-V2.6-Flash-MOPD** (EXL3, NVFP4, GGUF): existência vista no Hub; qualidade e compatibilidade com SM121 não avaliadas.

---

## 10. Fontes

Fontes primárias consultadas em 2026-10-01 (as marcadas com * foram baixadas e conferidas trecho a trecho; as demais, pelo resumo, abstract ou documentação):

- **S1*** XiaomiMiMo/MiMo-V2.6-Flash-MOPD (model card): https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD
- **S2*** MiMo-V2-Flash Technical Report, arXiv 2601.02780: https://arxiv.org/abs/2601.02780 (HTML: https://arxiv.org/html/2601.02780)
- **S3*** Ma et al., "MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training", arXiv 2606.30406 (2026-06-29): https://arxiv.org/abs/2606.30406
- **S4*** MiMo, "Diagnosing and Mitigating Tool-Call Repetition in MiMo-V2.6": https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition
- **S5*** Schulman / Thinking Machines, "LoRA Without Regret" (2025-09-29): https://thinkingmachines.ai/blog/lora/
- **S6*** Lu / Thinking Machines, "On-Policy Distillation" (2025-10-27): https://thinkingmachines.ai/blog/on-policy-distillation/
- **S7** Qwen3 Technical Report, arXiv 2505.09388: https://arxiv.org/abs/2505.09388
- **S8** Agarwal et al., GKD / "On-Policy Distillation of Language Models", arXiv 2306.13649 (ICLR 2024): https://arxiv.org/abs/2306.13649
- **S9** Dettmers et al., QLoRA, arXiv 2305.14314: https://arxiv.org/abs/2305.14314
- **S10** Liu et al., DoRA, arXiv 2402.09353 (ICML 2024): https://arxiv.org/abs/2402.09353
- **S11** Hu et al., LoRA, arXiv 2106.09685: https://arxiv.org/abs/2106.09685
- **S12** PEFT — LoRA developer guide: https://huggingface.co/docs/peft/main/en/developer_guides/lora
- **S13*** TRL — SFT Trainer: https://huggingface.co/docs/trl/main/en/sft_trainer
- **S14*** TRL — Reducing memory usage: https://huggingface.co/docs/trl/main/en/reducing_memory_usage
- **S15*** TRL — CLIs: https://huggingface.co/docs/trl/main/en/clis
- **S16*** TRL — Distillation Trainer: https://huggingface.co/docs/trl/main/en/distillation_trainer
- **S17*** TRL — Async Distillation Trainer: https://huggingface.co/docs/trl/main/en/async_distillation_trainer
- **S18** TRL releases (v1.10–v1.14.1): https://github.com/huggingface/trl/releases (v1.11.0: https://github.com/huggingface/trl/releases/tag/v1.11.0; v1.13.0: https://github.com/huggingface/trl/releases/tag/v1.13.0)
- **S19** "TRL v1.0" (2026-03-31): https://huggingface.co/blog/trl-v1
- **S20*** mlx-lm: `mlx_lm/LORA.md`, `mlx_lm/lora.py`, `mlx_lm/tuner/{utils,lora}.py`, `examples/lora_config.yaml`: https://github.com/ml-explore/mlx-lm
- **S21*** MLX optimizers (Adam/AdamW/Muon): https://github.com/ml-explore/mlx/blob/main/python/mlx/optimizers/optimizers.py
- **S22** vLLM — LoRA Adapters: https://docs.vllm.ai/en/latest/features/lora.html ; Engine Args (LoRAConfig): https://docs.vllm.ai/en/latest/configuration/engine_args.html
- **S23*** NVIDIA DGX Spark Playbooks (pytorch-fine-tune, nemo-fine-tune, llama-factory, unsloth, fine-tuning, nvfp4-pretraining): https://github.com/NVIDIA/dgx-spark-playbooks (também em https://build.nvidia.com/spark)
- **S24** NVIDIA Technical Blog, "How NVIDIA DGX Spark's Performance Enables Intensive AI Tasks" (2025-10-24): https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks
- **S25** Unsloth, "Fine-tuning LLMs with NVIDIA DGX Spark and Unsloth": https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth
- **S26** Unsloth, "Fine-tune MoE Models 12x Faster": https://unsloth.ai/docs/basics/faster-moe
- **S27*** Axolotl README e exemplos: https://github.com/axolotl-ai-cloud/axolotl ; NVFP4 LoRA: https://docs.axolotl.ai/docs/nvfp4_lora.html
- **S28*** LLaMA-Factory examples: https://github.com/hiyouga/LLaMA-Factory/tree/main/examples
- **S29** torchtune, "[IMPORTANT] The future of torchtune" (2025-07-15): https://github.com/meta-pytorch/torchtune/issues/2883 ; torchforge: https://github.com/meta-pytorch/torchforge
- **S30** verl docs (GSM8K): https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html ; repo: https://github.com/verl-project/verl
- **S31** OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- **S32** Keller Jordan, "Muon: An optimizer for hidden layers in neural networks": https://kellerjordan.github.io/posts/muon/
- **S33** Liu et al., "Muon is Scalable for LLM Training" (Moonlight), arXiv 2502.16982: https://arxiv.org/abs/2502.16982
- **S34** PyTorch 2.9 release notes (`torch.optim.Muon`): https://github.com/pytorch/pytorch/releases/tag/v2.9.0
- **S35** Rajbhandari et al., ZeRO, arXiv 1910.02054: https://arxiv.org/abs/1910.02054
- **S36** Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", arXiv 2205.05198: https://arxiv.org/abs/2205.05198
- **S37** Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", arXiv 2104.04473: https://arxiv.org/abs/2104.04473
- **S38** Zhou et al., LIMA, arXiv 2305.11206: https://arxiv.org/abs/2305.11206
- **S39** Muennighoff et al., s1, arXiv 2501.19393: https://arxiv.org/abs/2501.19393
- **S40** Ye et al., LIMO, arXiv 2502.03387: https://arxiv.org/abs/2502.03387
- **S41** Guha et al., OpenThoughts, arXiv 2506.04178: https://arxiv.org/abs/2506.04178
- **S42*** Lambert et al., Tülu 3, arXiv 2411.15124: https://arxiv.org/abs/2411.15124
- **S43** Yuan et al., "Scaling Relationship on Learning Mathematical Reasoning with LLMs" (RFT), arXiv 2308.01825: https://arxiv.org/abs/2308.01825
- **S44** Zelikman et al., STaR, arXiv 2203.14465: https://arxiv.org/abs/2203.14465
- **S45** Singh et al., ReST-EM ("Beyond Human Data"), arXiv 2312.06585: https://arxiv.org/abs/2312.06585
- **S46** Shenfeld, Pari, Agrawal, "RL's Razor", arXiv 2509.04259: https://arxiv.org/abs/2509.04259
- **S47** Biderman et al., "LoRA Learns Less and Forgets Less", arXiv 2405.09673: https://arxiv.org/abs/2405.09673
- **S48** Chu et al., "SFT Memorizes, RL Generalizes", arXiv 2501.17161: https://arxiv.org/abs/2501.17161
- **S49** Ouyang et al., InstructGPT, arXiv 2203.02155: https://arxiv.org/abs/2203.02155
- **S50** DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948: https://arxiv.org/abs/2501.12948
- **S51** Rafailov et al., DPO, arXiv 2305.18290: https://arxiv.org/abs/2305.18290
- **S52** Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO), arXiv 2310.12036: https://arxiv.org/abs/2310.12036
- **S53** Ethayarajh et al., KTO, arXiv 2402.01306: https://arxiv.org/abs/2402.01306
- **S54** Meng, Xia, Chen, SimPO, arXiv 2405.14734: https://arxiv.org/abs/2405.14734
- **S55** Hong et al., ORPO, arXiv 2403.07691: https://arxiv.org/abs/2403.07691
- **S56** Schulman et al., PPO, arXiv 1707.06347: https://arxiv.org/abs/1707.06347
- **S57** Shao et al., DeepSeekMath (GRPO), arXiv 2402.03300: https://arxiv.org/abs/2402.03300
- **S58** Peng et al., AWR, arXiv 1910.00177: https://arxiv.org/abs/1910.00177
- **S59** Bai et al., Constitutional AI, arXiv 2212.08073: https://arxiv.org/abs/2212.08073
- **S60** Chen et al., SPIN, arXiv 2401.01335: https://arxiv.org/abs/2401.01335
- **S61** Irving, Christiano, Amodei, "AI safety via debate", arXiv 1805.00899: https://arxiv.org/abs/1805.00899
- **S62** "On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification" (DFT), arXiv 2508.05629: https://arxiv.org/abs/2508.05629
- **S63** DeepSeek-V3 Technical Report (treino FP8), arXiv 2412.19437: https://arxiv.org/abs/2412.19437
- **S64** NVIDIA, "Pretraining Large Language Models with NVFP4", arXiv 2509.25149: https://arxiv.org/abs/2509.25149
- **S65** Ahmadian et al., RLOO ("Back to Basics"), arXiv 2402.14740: https://arxiv.org/abs/2402.14740
- **S66** Gu et al., MiniLLM, arXiv 2306.08543: https://arxiv.org/abs/2306.08543
- **S67** J. Schulman, "Approximating KL Divergence": http://joschu.net/blog/kl-approx.html
- **S68** Zhao et al., SLiC-HF, arXiv 2305.10425: https://arxiv.org/abs/2305.10425 ; Xu et al., CPO, arXiv 2401.08417: https://arxiv.org/abs/2401.08417
- **S69** Mnih et al., A3C, arXiv 1602.01783: https://arxiv.org/abs/1602.01783 ; Schulman et al., TRPO, arXiv 1502.05477: https://arxiv.org/abs/1502.05477
- **S70** Yuan et al., Self-Rewarding Language Models, arXiv 2401.10020: https://arxiv.org/abs/2401.10020
- **S71** Jain et al., NEFTune, arXiv 2310.05914: https://arxiv.org/abs/2310.05914
- **S72** rsLoRA arXiv 2312.03732; LoRA+ arXiv 2402.12354; PiSSA arXiv 2404.02948; LoftQ arXiv 2310.08659 (https://arxiv.org/abs/<id>)
- **S73** Stanford Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca.html ; Self-Instruct arXiv 2212.10560
- **S74** NeMo AutoModel: https://github.com/NVIDIA-NeMo/Automodel
- **S75** PyPI (versões em 2026-10-01): https://pypi.org/project/trl/ , /peft/ , /transformers/ , /unsloth/ , /axolotl/ , /llamafactory/ , /verl/ , /openrlhf/ , /bitsandbytes/ , /nemo-automodel/ , /mlx-lm/ , /mlx-lm-lora/
- **S76*** MiMo-V2.6 Technical Report ("Scaling Reinforcement Learning Towards Self-Improvement"; §5.4 roteador congelado, §5.6 MOPD2, §6.4 consistência treino/inferência, §7.1 MiMo-V2.6-Distill-Qwen-9B): https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf
- Notas do Módulo 1 usadas: `docs/pesquisa/arch-qwen.md` (Muon/GR, tokens, amostragem), `arch-ds.md` (Muon híbrido do V4), `arch-glm.md` (experts do GLM), `glm-launch.md` (orçamento de memória UMA), `qwen-launch.md` (121,69 GiB; RoCE), `ds-launch.md` (PFs do CX7), `mlx-copilot.md` (working set do M4), `vllm-ref.md` (LoRA no prefix caching e no `BatchDescriptor`).

# Módulo 2 · Pós-treino — avaliação de modelos pós-treinados, serviço (LoRA, merge, re-quantização) e receita ponta a ponta (Mac M4 24 GB + 2× DGX Spark)

> Data da pesquisa: **2026-10-01**. Explicações em PT-BR; identificadores, flags, código e fórmulas como estão.
> Escopo: insumo para (a) os capítulos de **avaliação de modelos pós-treinados** e (b) um **laboratório completo de pós-treino** que termina servindo o modelo com o mesmo vLLM/quantização/rede estudados no Módulo 1 (receitas GLM-5.3-Flash EXL3, Qwen3.8-Flash-Next NVFP4, DeepSeek-V4-Flash DSpark em 2× GB10).
>
> **Convenções de marcação.** **[verificado]** = conferido diretamente nesta pesquisa em fonte primária executável/legível (código-fonte em `main` ou na tag citada, JSON da API do PyPI/Hugging Face, `config.json` do checkpoint). **[fonte]** = lido em documentação oficial, model card ou paper (URL no Apêndice F). **[estimativa]** = conta minha, com a fórmula e as premissas ao lado — trate como ordem de grandeza e meça. **[não verificado]** = informação de terceiros (blog, agregador, fórum) ou que não consegui conferir. **[inferência]** = conclusão minha a partir de fatos verificados.
>
> **Nada aqui foi executado em GPU.** Nenhum treino foi rodado nesta pesquisa; todos os tempos e memórias de treino são **[estimativa]** ancorados em medições públicas citadas. Os comandos foram montados a partir das assinaturas/flags conferidas no código e na doc das versões listadas na §0.2.

---

## 0. Resumo executivo

### 0.1 Decisões recomendadas (TL;DR)

1. **Avalie em quatro eixos, sempre contra o modelo de partida e com intervalos de confiança:** (i) ganho no alvo (win rate pareado com juiz LLM, controlado por comprimento, + métricas verificáveis do domínio); (ii) **não-regressão** de capacidades gerais (IFEval, GSM8K/AIME, código, ferramentas); (iii) **segurança** nos dois sentidos (recusa excessiva — XSTest — e cumprimento indevido — StrongREJECT/HarmBench); (iv) **deriva** medida em nats: KL do modelo pós-treinado em relação à referência, em texto genérico e de domínio. Pré-registre limites antes de olhar os números.
2. **Juiz LLM só com mitigação de vieses:** troca de posição (A/B e B/A, conta só concordância), juiz de **família diferente** do professor que gerou os dados (autopreferência), controle de comprimento (regressão tipo AlpacaEval-LC), e de preferência um **painel** de juízes. O Módulo 1 já entrega três famílias servidas nos Sparks (GLM, Qwen, DeepSeek) — use-as como painel.
3. **Benchmarks de 2026:** os clássicos estão saturados (MMLU, GSM8K, HumanEval, MATH-500) e servem só como **teste de regressão**; o que discrimina modelos de fronteira hoje é HLE, GPQA Diamond, SWE-bench Pro (o Verified foi abandonado pela própria OpenAI em fev/2026 — fonte secundária), Terminal-Bench, τ²-bench, BFCL V4, competições recentes via MathArena (AIME 2026, HMMT fev/2026), IFBench, AA-LCR. Para um modelo de 4–27B pós-treinado em casa, o painel útil é: IFEval/IFBench + GSM8K/AIME 2025-26 + HumanEval/LiveCodeBench + BFCL/τ² (subconjunto) + XSTest + win rate de domínio + KL.
4. **Servir um modelo pós-treinado = escolher entre três caminhos:** (A) **adapter LoRA em runtime** no vLLM (`--enable-lora`, multi-LoRA por requisição; ótimo para A/B de adapters); (B) **merge → BF16**; (C) **merge → re-quantizar** (NVFP4/GPTQ/AWQ via llm-compressor ou ModelOpt; EXL3 via exllamav3; MLX via `mlx_lm.dwq/awq/gptq`). Para produção em GB10, (C) é o caminho natural, e **a calibração tem de ser refeita no modelo fine-tunado** (as escalas/Hessianos do base não valem mais), com dados que misturem o domínio do SFT e texto geral.
5. **Duas KLDs diferentes, nunca confunda:** KL(π_ft‖π_base) mede **o que o pós-treino mudou** (desejado no domínio, indesejado fora dele); KL(π_ft,BF16‖π_ft,quant) mede **o estrago da quantização** — a referência é o **merge BF16 do modelo fine-tunado**, não o base original. Calibre ambos contra o ruído "stock vs stock", como a receita GLM faz.
6. **Receita prática:** Mac M4 24 GB → `Qwen/Qwen3-4B-Instruct-2507` (4-bit MLX) com `mlx_lm.lora` (QLoRA SFT) e `mlx_lm_lora.train --train-mode dpo` (DPO; o `mlx-lm` puro não tem DPO, e o `mlx-lm-lora` 3.1.3 **não tem KTO** [verificado no código]). 2× DGX Spark → **`Qwen/Qwen3.8-27B`** (mesma família/arquitetura híbrida GDN+atenção do Qwen3.8-Flash-Next do Módulo 1) com TRL 1.14.1: SFT LoRA BF16 em DDP nos dois nós → DPO (LoRA) → avaliação → merge → NVFP4 (llm-compressor ou ModelOpt, ambos com receita oficial para o Qwen3.8-27B) → vLLM 0.30.0 (EXL3 opcional, para o painel de KLD — o vLLM upstream não carrega EXL3). **GRPO com recompensa verificável** em `Qwen/Qwen3.5-9B` (mesma `model_type: qwen3_5`), usando os dois Sparks em papéis assimétricos: Spark B gera (vLLM server com transferência de pesos NCCL), Spark A treina.
7. **Ordem de grandeza de tempo:** o treino LoRA num GB10 anda a ~**60 TFLOPS efetivos** quando os kernels certos estão instalados (derivado de medições públicas: Qwen3.5-27B LoRA 370 tok/s e 9B 1.095 tok/s, que batem com 6·N FLOPs/token a ~60 TFLOPS) **[estimativa ancorada em medição de terceiros]**. Sem `flash-linear-attention` + `causal-conv1d`, modelos com Gated DeltaNet caem num caminho PyTorch puro várias vezes mais lento (medição de terceiros: 505 → 3.076 tok/s em Qwen3.5-2B só por instalar o caminho rápido). Os números de "pico" do fabricante para fine-tuning no Spark (ex.: 53.657 tok/s em Llama 3.1 8B LoRA) **não fecham** com a conta de FLOPs — não os use para planejar.

### 0.2 Versões verificadas (PyPI/GitHub, consultados em 2026-10-01)

| Componente | Versão | Data | Observação | Status |
|---|---|---|---|---|
| `trl` | **1.14.1** | 2026-09-29 | v1.0 em 2026-03-31 (núcleo estável com semver + camada `trl.experimental`). v1.13.0 removeu o `PPOTrainer`; v1.14.0 removeu `trl.losses` e 6 trainers experimentais (BCO, GRPO-com-replay, PRM, Nash-MD, XPO, GSPO-token); v1.8.0 promoveu `KTOTrainer` a estável; v1.11.0 trocou para o servidor vLLM nativo | [verificado] PyPI + release notes |
| `transformers` | 5.18.0 | 2026-09-30 | v5: `dtype` inferido do config no `from_pretrained` | [verificado] PyPI |
| `peft` | 0.21.2 | 2026-10-01 | — | [verificado] PyPI |
| `accelerate` | 1.15.0 | 2026-09-09 | — | [verificado] PyPI |
| `deepspeed` | 0.19.7 | 2026-09-16 | — | [verificado] PyPI |
| `bitsandbytes` | 0.50.2 | 2026-08-27 | há wheel `manylinux_2_24_aarch64` e `macosx_14_0_arm64` | [verificado] PyPI |
| `liger-kernel` | 0.8.4 | 2026-09-30 | — | [verificado] PyPI |
| `unsloth` | 2026.9.14 | 2026-10-01 | — | [verificado] PyPI |
| `torch` | 2.14.1 | — | NGC `pytorch:26.08-py3` traz 2.14.0a0+4fdf77b940 e CUDA 13.4.1 | [verificado] PyPI; [fonte] release notes NGC |
| `flash-linear-attention` (`fla`) | 0.5.2 | — | caminho rápido do Gated DeltaNet | [verificado] PyPI |
| `causal-conv1d` | 1.7.0 | — | idem (conv curta do GDN) | [verificado] PyPI |
| `vllm` | **0.30.0** | 2026-09-22 | TRL suporta vLLM **0.20.1 a 0.30.0** | [verificado] PyPI; [fonte] doc TRL |
| NGC `nvcr.io/nvidia/vllm` | 26.08 (= vLLM 0.27.1); 26.09 existe | — | o playbook NVFP4 do Spark exige ≥ 26.05 | [fonte] release notes NGC |
| `llmcompressor` | 0.14.0 | 2026-09-22 | exemplos oficiais para Qwen3.8-27B (INT4/NVFP4/MXFP4) | [verificado] PyPI + repo |
| `nvidia-modelopt` | 0.47.0 | 2026-09-23 | o card `nvidia/Qwen3.8-27B-NVFP4` diz "ModelOpt v0.48.0" (provavelmente build do git) | [verificado] PyPI; [fonte] card |
| `exllamav3` | v1.5.3 | 2026-09-27 | (a receita GLM do Módulo 1 usa um fork antigo, 0.0.43 `c5d9c657`) | [verificado] GitHub releases |
| `mlx-lm` | **0.32.0** | **2026-10-01** | lançado hoje; a nota `mlx-copilot.md` testou 0.31.3 | [verificado] PyPI |
| `mlx-lm-lora` | 3.1.3 | 2026-09-16 | requer `mlx>=0.30.6`, `mlx_lm>=0.30.6` | [verificado] PyPI |
| `openrlhf` | 0.11.2 | 2026-09-14 | flags com namespace (`--actor.*`, `--ds.*`, `--algo.*`) | [verificado] PyPI + README |
| `verl` | 0.9.1 | 2026-09-20 | configuração Hydra (`actor_rollout_ref.*`) | [verificado] PyPI + exemplo |
| `lm-eval` | 0.4.13 | 2026-08-31 | (0.5.0.dev1 é pré-release) | [verificado] PyPI |
| `inspect-ai` | 0.3.273 | 2026-09-30 | — | [verificado] PyPI |
| `lighteval` | 0.13.0 | 2025-11-24 | sem release em 2026 | [verificado] PyPI |
| `alpaca-eval` | 0.6.6 | 2024-12-27 | sem release desde 2024 | [verificado] PyPI |
| `tau2-bench` | v1.0.1 | jul/2026 | Python ≥3.12,<3.14; `uv` | [fonte] README |

---

## 1. Avaliação de modelos pós-treinados

### 1.1 Por que avaliar pós-treino é diferente de avaliar quantização

No Módulo 1 (capítulo "Medindo o estrago"), a pergunta era **"o modelo servido ainda é o mesmo?"** — a referência é fixa (o checkpoint oficial) e o ideal é KLD ≈ 0. No pós-treino a pergunta se inverte: **queremos que o modelo mude**, mas só onde pedimos. Daí quatro eixos, que viram a estrutura deste capítulo:

| Eixo | Pergunta | Instrumento principal | Direção desejada |
|---|---|---|---|
| Alvo | O comportamento treinado melhorou? | win rate pareado (juiz LLM ou humano), métricas verificáveis do domínio, loss de validação | ↑ |
| Regressão | Perdi capacidade geral ("alignment tax", esquecimento)? | bateria fixa (IFEval, GSM8K/AIME, código, ferramentas, conhecimento) | ≈ base (Δ dentro do IC) |
| Segurança | Recusa demais? Cumpre o que não devia? | XSTest/OR-Bench (sobre-recusa), StrongREJECT/HarmBench/JailbreakBench (cumprimento indevido) | conforme política |
| Deriva | Quanto a distribuição mudou, e onde? | KL(π_ft‖π_ref) por token em texto genérico e de domínio; comprimento das respostas; aceitação do MTP | ↑ no domínio, ≈0 fora dele |

Princípios que valem para todas as seções:

- **Compare sempre pares de modelos nas mesmas condições**: mesmo template de chat, mesmo `enable_thinking`, mesmos parâmetros de amostragem, mesmo `max_tokens`, mesmo servidor. Diferença de template é a causa nº 1 de "regressões" falsas.
- **Pré-registre** (antes de rodar): conjunto de avaliação, métricas, limites de aceitação, número de amostras. Isso evita escolher o checkpoint "que ficou melhor no benchmark que eu olhei por último" (sobreajuste ao conjunto de avaliação — a mesma armadilha que os laboratórios de fronteira cometem em público; ver §1.8 e §1.14).
- **Intervalos de confiança** em tudo (Apêndice A): com 400 pares, um win rate de 50% tem IC95% de ≈ ±4,9 pontos; diferenças menores que isso são ruído.
- **Ruído de base ("stock vs stock")**: rode o próprio modelo base duas vezes (sementes/lotes diferentes) e meça a métrica entre essas duas execuções. É exatamente o que a receita GLM faz com KLD ("near-tie tokens flip between identical runs" — `tests/bench_numerical_calibrated.py`, ver `quant.md` C45). Um efeito menor que o ruído stock-vs-stock não existe.

### 1.2 Taxonomia rápida

| Família | Exemplos | Mede | Custo | Viés/armadilha principal |
|---|---|---|---|---|
| Verificável por regra | IFEval, IFBench, GSM8K, MATH, AIME, HumanEval, LiveCodeBench, SWE-bench, BFCL | correção objetiva | baixo–médio (código exige sandbox) | contaminação; extração de resposta; templates |
| Juiz LLM com referência/base | AlpacaEval 2 (LC), Arena-Hard-Auto, MT-Bench, win rate de domínio | qualidade "aberta" | médio (API/juiz local) | posição, verbosidade, autopreferência, estilo |
| Humanos pareados | Chatbot Arena/LMArena, avaliação interna | preferência real | alto | população de votantes, estilo, gaming |
| Agentes/ferramentas | τ²-bench, BFCL V4, Terminal-Bench, OSWorld, BrowseComp | tarefas multi-passo | alto | harness, simulador de usuário, variância (use pass^k) |
| Segurança | XSTest, OR-Bench, HarmBench, StrongREJECT, JailbreakBench | recusa e robustez | médio | classificador-juiz; trade-off disposição×capacidade |
| Intrínseca/distribucional | NLL/PPL, KL vs referência, top-1 agreement, aceitação de draft | deriva numérica | baixo | não mede "qualidade" diretamente |

### 1.3 Win rate e julgamento pareado

**Definição.** Para cada prompt *x* do conjunto de avaliação, gere uma resposta do candidato *A* e uma do baseline *B*; um juiz (humano ou LLM) devolve A ≻ B, B ≻ A ou empate. Win rate:

$$\text{WR}(A\,|\,B)=\frac{W+\tfrac12 T}{W+L+T}$$

(empates contam meio ponto — convenção do AlpacaEval/Arena-Hard; reporte também W/L/T separados).

**Protocolo mínimo que recomendo para o laboratório** (cada item tem motivo nas seções seguintes):

1. Duas chamadas por par: (A,B) e (B,A). Se o juiz inverter o veredito quando as posições trocam, conte **empate** (viés de posição, §1.4).
2. Juiz de **família diferente** do professor que gerou os dados de treino e do próprio modelo avaliado (autopreferência, §1.4).
3. Prompt do juiz com **rubrica explícita** e, quando houver, **resposta de referência**; peça raciocínio curto antes do veredito ("multiple evidence calibration", §1.4).
4. Ajuste de comprimento (regressão logística com diferença de comprimento, §1.5) e reporte WR bruto e WR-LC.
5. IC95% por bootstrap sobre prompts (Apêndice A.1) e **teste pareado** (o mesmo prompt para os dois modelos).
6. 10% dos pares julgados também por um humano; calcule a concordância juiz–humano (κ de Cohen). Se κ < ~0,4, o juiz não serve para essa tarefa **[recomendação; o limiar é convenção usual de concordância, não um número de fonte]**.

**Tamanho de amostra [estimativa via aproximação normal].** EP(WR) ≈ √(p(1−p)/n). Em p = 0,5: n = 100 → ±9,8 pp (IC95%); n = 400 → ±4,9 pp; n = 1.000 → ±3,1 pp. Para distinguir 55% de 50% com folga, ~400–800 prompts.

**Bradley–Terry entre vários checkpoints.** Quando você tem base, SFT, DPO, quantizado etc., julgue pares cruzados e ajuste BT (Apêndice A.3): P(i ≻ j) = σ(β_i − β_j). Em escala tipo Elo, R_i = C + (400/ln 10)·β_i. É a mesma família de modelo do LMArena (§1.8), só que com o seu juiz.

### 1.4 LLM-as-judge e seus vieses

**O resultado fundador.** Zheng et al. (NeurIPS 2023, MT-Bench/Chatbot Arena) mostraram que juízes fortes (GPT-4) atingem **"over 80% agreement"** com preferências humanas — o mesmo nível de concordância entre humanos — e catalogaram os vieses: **posição, verbosidade, autoaprimoramento (self-enhancement) e raciocínio limitado** (em matemática/código o juiz erra o que não consegue resolver) [fonte: arXiv 2306.05685].

| Viés | Evidência | Mitigação |
|---|---|---|
| **Posição** | Wang et al. 2023: com ChatGPT como juiz, só reordenando as respostas, "Vicuna-13B could beat ChatGPT on 66 over 80 tested queries" [fonte: arXiv 2305.17926] | julgar nas duas ordens ("Balanced Position Calibration"); pedir evidências antes da nota ("Multiple Evidence Calibration"); casos de alta entropia para humano |
| **Verbosidade/estilo** | juízes e humanos premiam respostas longas e formatadas; o LMArena encontrou que comprimento é o fator de estilo dominante e passou a controlar estilo por padrão (§1.8) | controle de comprimento por regressão (AlpacaEval-LC, §1.5; `--control-features` no Arena-Hard, §1.6); rubrica que penaliza enchimento |
| **Autopreferência** | Panickssery, Bowman & Feng 2024: o avaliador "scores its own outputs higher than others' while human annotators consider them of equal quality", com correlação linear entre capacidade de **se reconhecer** e força do viés (GPT-4, Llama 2) [fonte: arXiv 2404.13076] | juiz de família diferente; painel de famílias disjuntas |
| **Raciocínio limitado** | o juiz erra em itens que ele mesmo não resolve [fonte: arXiv 2306.05685] | juiz com referência (resposta correta no prompt) ou checagem por regra |
| **Painel vs juiz único** | PoLL (Verga et al. 2024): painel de modelos menores de **famílias disjuntas** "exhibits less intra-model bias" e é "over seven times less expensive" que um juiz grande, em 3 configurações e 6 datasets [fonte: arXiv 2404.18796] | 3 juízes de famílias diferentes, voto majoritário |

**Ponte com o Módulo 1 [inferência].** As três receitas servem, nos mesmos dois Sparks, modelos de três famílias distintas — `GLM-5.3-Flash-EXL3`, `qwen3.8-flash-next`, `deepseek-v4-flash-vision-exp` (todas na porta 8888, API OpenAI). Isso é literalmente um painel PoLL caseiro. Regra prática: exclua do painel (a) a família do **professor** que gerou os dados e (b) a família do **aluno**. No laboratório da §5 (professor GLM, aluno Qwen3.8-27B — parente do Qwen3.8-Flash-Next), o único juiz isento entre os três é o **DeepSeek**; se quiser painel, use GLM e Qwen-Flash-Next apenas como juízes secundários e reporte a concordância deles com o DeepSeek separadamente.

### 1.5 AlpacaEval 2 e o win rate controlado por comprimento (LC)

- **Conjunto:** 805 instruções; baseline padrão `gpt4_turbo`; anotador padrão `weighted_alpaca_eval_gpt4_turbo`; custo típico < US$ 10 e < 3 min [fonte: README tatsu-lab/alpaca_eval].
- **LC:** ajusta-se um **modelo linear generalizado** (logístico) que prevê a preferência do anotador a partir de (i) identidade do modelo, (ii) diferença de comprimento entre as respostas e (iii) dificuldade da instrução; o **LC win rate** é a previsão **com a diferença de comprimento zerada**, média sobre as instruções. O paper reporta Spearman com o Chatbot Arena subindo **de 0,94 para 0,98** [fonte: arXiv 2404.04475]. (O README do repositório cita 0,93 para o AE2 bruto; uso o número do paper.)
- Forma do modelo do paper, como a recordo **[fonte: arXiv 2404.04475; transcrição de memória — conferir no paper antes de publicar]**:
  $$q(y{=}m\,|\,x)=\sigma\Big(\theta_m-\theta_b+\phi_{m,b}\tanh\!\big(\tfrac{\ell_m(x)-\ell_b(x)}{\mathrm{std}(\ell_m-\ell_b)}\big)+(\psi_m-\psi_b)\gamma_x\Big),\qquad \mathrm{WR}_{LC}=100\cdot\mathbb E_x\,\sigma(\theta_m-\theta_b+(\psi_m-\psi_b)\gamma_x)$$
- **Ressalvas dos próprios autores:** a preferência por respostas longas diminui mas não some; viés para estilo sobre factualidade; instruções simples demais para o uso de 2026; decisões de alto risco exigem humanos [fonte: README].
- **Para o laboratório:** o AlpacaEval oficial depende de um anotador pago (GPT-4 Turbo). O que eu proponho é **reusar a ideia** com o seu juiz local e o seu conjunto de domínio: regressão logística simplificada (Apêndice B.2) com o termo de comprimento, reportando WR e WR-LC. Para comparabilidade externa, `alpaca_eval evaluate_from_model` aceita configs de modelo apontando para um endpoint OpenAI-compatível (o vLLM) **[fonte: README; configuração exata do endpoint não verificada]**.

### 1.6 Arena-Hard-Auto

- **v0.1:** 500 prompts difíceis destilados do Chatbot Arena pelo pipeline BenchBuilder; o paper reporta **"3x higher separation of model performances compared to MT-Bench"**, **"98.6% correlation with human preference rankings"** e custo de **US$ 20** [fonte: arXiv 2406.11939].
- **v2.0:** 500 consultas novas "fresh, challenging real-world" + 250 de escrita criativa = 750; juiz oficial **Gemini-2.5**, alternativa **GPT-4.1** "for fast, stable judge inference", ou ensemble [fonte: README lmarena/arena-hard-auto].
- **Controle de estilo:** `show_result.py --control-features` (controla densidade de markdown e comprimento) [fonte: README].
- **Pipeline:** `gen_answer.py` → `gen_judgment.py` → `show_result.py`, com `config/api_config.yaml` aceitando servidor OpenAI-compatível (vLLM) [fonte: README].
- Uso no laboratório: a v2.0 com juiz local (um dos modelos do Módulo 1) dá um número "fora do domínio" para pegar regressão de qualidade conversacional; não compare o valor absoluto com o leaderboard (juiz diferente).

### 1.7 MT-Bench

80 perguntas de **dois turnos** em 8 categorias (escrita, roleplay, raciocínio, matemática, código, extração, STEM, humanidades), nota 1–10 por um juiz forte [fonte: arXiv 2306.05685]. Em 2026 é **saturado** para modelos competentes e sensível ao juiz; serve para pegar regressão de **multi-turno** (ex.: o SFT quebrou o segundo turno porque os dados eram todos de um turno só).

### 1.8 Chatbot Arena / LMArena (Elo → Bradley–Terry)

- **Método:** batalhas anônimas entre dois modelos, voto humano; o paper de 2024 relatava "over 240K votes" e boa concordância entre votos coletivos e especialistas [fonte: arXiv 2403.04132].
- **Elo vs BT:** Elo online depende da ordem das batalhas; o leaderboard usa **Bradley–Terry** (máxima verossimilhança sobre todas as batalhas), reportado em escala tipo Elo.
- **Linha do tempo de metodologia** [fonte: arena.ai/blog/leaderboard-changelog e busca]:
  - ago/2024: **style control** — regressão BT com covariáveis de estilo (comprimento, cabeçalhos markdown, listas, negrito) **[fonte secundária: resultados de busca; não li o post original]**;
  - **2025-05-16:** style control vira **padrão** nas arenas de texto e visão (offset ajustado para manter a escala) **[idem]**;
  - **2025-07-23:** ICs deixam o bootstrap e passam a usar CLT para M-estimadores; reponderação para equilibrar modelos com contagens de votos diferentes;
  - 2025-07-14: deduplicação e detecção de vazamento de identidade (~10% e <4% dos votos filtrados);
  - 2025-12-18: metodologia aberta como pacote **Arena-Rank** (`uv pip install arena-rank`, Apache-2.0) — "implements the Bradley-Terry model and an extension for handling contextual features, which we use for the style-controlled leaderboards", com IC em forma fechada [fonte: arena.ai/blog/arena-rank] — dá para usar no laboratório no lugar do Apêndice B.3;
  - **2026-05-12:** votos de sessões "direct" entram no leaderboard; covariáveis `is_direct_battle` e `same_org_indicator` adicionadas ao ajuste BT;
  - 2026-09-30: mudança no "Steerability score" da Agent Arena.
- **Críticas (leia antes de citar ranking):** "The Leaderboard Illusion" (Singh et al. 2025) documenta testes privados de muitas variantes antes do lançamento (≈ **27 variantes da Meta** antes do Llama-4), assimetria de dados (Google ≈ 19,2% e OpenAI ≈ 20,4% de todos os dados do Arena vs ≈ 29,7% para 83 modelos de pesos abertos somados) e ganhos relativos "of up to 112%" na distribuição do Arena com acesso a mais dados [fonte: arXiv 2504.20879].
- **Lição para o laboratório:** o mesmo mecanismo que infla rankings públicos (escolher, entre muitas variantes, a que pontuou melhor no avaliador) acontece quando você testa 20 checkpoints contra o mesmo conjunto de 200 prompts. Separe **conjunto de seleção** e **conjunto de relatório**.

### 1.9 Seguir instruções: IFEval e IFBench

- **IFEval:** "25 types of verifiable instructions" e "around 500 prompts" (o dataset `google/IFEval` tem **541** linhas [verificado: datasets-server]), cada um com uma ou mais instruções verificáveis por código (contagem de palavras, formato JSON, ausência de vírgulas…) [fonte: arXiv 2311.07911]. Métricas: `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_loose_acc` (no lm-eval a task `ifeval` gera até 1.280 tokens, 0-shot) [verificado: `lm_eval/tasks/ifeval/ifeval.yaml`].
- **IFBench:** sucessor com restrições novas, usado em 2026 por model cards (ex.: o card `nvidia/Qwen3.8-27B-NVFP4` reporta IFBench 80,07 BF16 → 78,93 NVFP4) [fonte: card]; o exllamav3 traz `eval/ifbench.py` [verificado: árvore do repo].
- **Por que importa no pós-treino:** SFT de domínio frequentemente **destrói** formato (o modelo aprende o estilo dos dados e ignora "responda em 3 tópicos"). IFEval é o detector barato disso.

### 1.10 Matemática: GSM8K, MATH, AIME

- **GSM8K:** 1.319 problemas de teste (7.473 de treino) [verificado: datasets-server `openai/gsm8k`]; saturado na fronteira (o Módulo 1 cita o Qwen3.8-Flash-Next com 97,27% = 1283/1319 [`quant.md` §4.2]). Útil como regressão. Tasks do lm-eval: `gsm8k` (5-shot, extração por regex), `gsm8k_cot_llama` (8-shot), `gsm8k_cot_zeroshot` [verificado: YAMLs].
- **MATH / MATH-500:** 5.000 de teste no original; o subconjunto MATH-500 (`HuggingFaceH4/MATH-500`) virou padrão e também está saturado.
- **AIME:** 30 problemas por ano (I+II). No lm-eval há `aime24` e `aime25` (0-shot, `max_gen_toks: 32768`) [verificado]; AIME 2026 e HMMT fev/2026 estão em `MathArena/aime_2026` (30 linhas) e `MathArena/hmmt_feb_2026` [verificado: existem no Hub].
- **Como reportar:** pass@1 **médio sobre k amostras** (temperatura recomendada pelo modelo), não greedy único — com 30 itens, uma questão = 3,3 pp. Estimador não-viesado de pass@k (Apêndice A.5).
- **Contaminação em matemática:** GSM1k (Zhang et al. 2024) — problemas novos no estilo GSM8K — mostrou quedas de **"up to 8%"**, famílias inteiras com sinais de sobreajuste e uma relação **Spearman r² = 0,36** entre a probabilidade de o modelo gerar exemplos do GSM8K e o tamanho da queda [fonte: arXiv 2405.00332]. MathArena avalia competições **no dia em que saem** e encontrou "strong signs of contamination in AIME 2024" [fonte: arXiv 2505.23281]. Para 2026, use AIME 2026/HMMT 2026, não AIME 2024.
- **Extração da resposta:** o `accuracy_reward` do TRL usa `math_verify` (parse de LaTeX, `\boxed{}` com prioridade) [verificado: `trl/rewards/accuracy_rewards.py`] — use o mesmo verificador na avaliação e no RL, senão você otimiza uma coisa e mede outra.

### 1.11 Código: HumanEval → LiveCodeBench → SWE-bench → Terminal-Bench

| Benchmark | O que é | Status em 2026 |
|---|---|---|
| HumanEval (164) / HumanEval+ | funções Python com testes; pass@k | saturado; regressão. lm-eval: `humaneval`, `humaneval_instruct`, `humaneval_plus`, `humaneval_64*` (exige `--confirm_run_unsafe_code`) [verificado: árvore de tasks] |
| MBPP | problemas curtos | saturado |
| LiveCodeBench | problemas de competições **com janela de datas** (contaminação controlável escolhendo problemas posteriores ao corte do modelo) | ativo; `livecodebench/code_generation_lite` no Hub [verificado: existe]. O card do Qwen3.8-27B reporta 90,3 [fonte: card] |
| SWE-bench Verified (500) | patches em repositórios Python reais | **abandonado pela OpenAI em fev/2026**: auditoria de 138 tarefas difíceis com **59,4%** de testes com defeito; sinais de que modelos viram as soluções; topo estagnado de 74,9% → 80,9% em seis meses. Recomendou SWE-bench Pro **[não verificado: a página da OpenAI devolveu 403; números de cobertura secundária (Decrypt, blockchain.news, pebblous.ai)]** |
| SWE-bench Pro | sucessor mais difícil (Scale AI) | ativo; card do Qwen3.8-27B: 61,7 [fonte] |
| SWE-Bench Pro Verified | correção do Pro contra **reward hacking por vazamento** de soluções/testes e problemas de qualidade; "some models perform substantially worse than previously evaluated" | arXiv 2609.08149, 8–16/set/2026 [fonte: abstract] |
| Terminal-Bench | tarefas reais em terminal (Stanford/Laude, harness Harbor) | o site já anuncia **Terminal-Bench 4.0**; o índice da Artificial Analysis usa "Terminal-Bench 4.0 — 66 tasks; pass@1 over 3 repeats" [fonte: tbench.ai; artificialanalysis.ai]; o card do Qwen3.8-27B reporta "Terminal Bench 73.0" (versão não indicada) |

### 1.12 Ferramentas e agentes

- **BFCL (Berkeley Function Calling Leaderboard):** versão atual **V4**, atualizada em **2026-04-12**; categorias: não-live (v1), funções enterprise/OSS (v2), multi-turno (v3), **busca web agêntica** (v4), sensibilidade de formato (só para modelos via prompt); "Overall Accuracy is the **unweighted** average of all the sub-categories" [fonte: gorilla.cs.berkeley.edu/leaderboard]. O Qwen3.5-9B reporta BFCL-V4 66,1 [fonte: card].
- **τ²-bench (Sierra):** simulação de atendimento com **simulador de usuário** e ferramentas dos dois lados; domínios `airline`, `retail`, `telecom`, `mock` e, em 2026, `banking_knowledge` (RAG configurável); v1.0.1 (jul/2026); CLI: `tau2 run --domain airline --agent-llm <m> --user-llm <m> --num-trials 1 --num-tasks 5`; modelos locais via LiteLLM (`hosted_vllm/...` ou `base_url` no `.env`) [fonte: README]. Métrica de confiabilidade **pass^k** = probabilidade de acertar a mesma tarefa em **todas** as k tentativas (estimador no Apêndice A.5) — a pergunta certa para agentes ("funciona sempre?"), oposta ao pass@k ("funciona alguma vez?").
- **Outros agentes em uso em 2026** [fonte: cards/AA]: OSWorld-Verified, WebArena-Verified (Qwen3.8-27B: 84,3 e 64,8), BrowseComp, GDPval-AA, AutomationBench-AA, AA-Briefcase.
- **Para modelos pós-treinados em casa:** se o SFT mexeu em tool calling (formato XML `<tool_call><function=…><parameter=…>` do template do Qwen3.5/3.8 [verificado: `chat_template.jinja` do Qwen3.8-27B]), rode um subconjunto do BFCL (non-live + multi-turn) e 20–50 tarefas τ² `retail` com `--num-trials 4` para estimar pass^1 e pass^4. Use o **mesmo parser** que o vLLM vai usar em produção (`--tool-call-parser qwen3_xml` nos cards do RedHatAI para o Qwen3.8-27B; `qwen3_coder` no card do Qwen3.5-9B [fonte]).

### 1.13 Segurança: recusas e robustez a jailbreak

| Benchmark | Conteúdo | Mede | Status |
|---|---|---|---|
| **XSTest** | **250 prompts seguros** (10 tipos) que um modelo calibrado não deveria recusar + **200 inseguros** de contraste [fonte: arXiv 2308.01263] | **sobre-recusa** ("exaggerated safety") | `walledai/XSTest`, `Paul/XSTest` no Hub [verificado: existem] |
| OR-Bench | conjunto grande de prompts "aparentemente tóxicos" | sobre-recusa em escala | [não verificado: tamanhos] |
| **HarmBench** | framework padronizado; "18 red teaming methods and 33 target LLMs and defenses"; categorias funcionais standard/contextual/copyright/multimodal; métrica = **attack success rate** com classificador [fonte: arXiv 2402.04249] | robustez a ataques | — |
| **StrongREJECT** | prompts que exigem informação nociva específica + avaliador que mede **utilidade** da resposta; achado: jailbreaks que furam a recusa **degradam a capacidade** ("willingness–capabilities tradeoff"), então ASR binária superestima [fonte: arXiv 2402.10260] | cumprimento indevido "útil" | `walledai/StrongREJECT` no Hub [verificado: existe] |
| JailbreakBench | comportamentos + artefatos de ataque padronizados | robustez | [não verificado: detalhes] |

**Achados que todo pós-treino deveria conhecer:**
- **Fine-tuning corrói alinhamento:** Qi et al. (2023) furaram as salvaguardas do GPT-3.5 Turbo com **10 exemplos adversariais por < US$ 0,20** via API de fine-tuning, e mostraram que até dados **benignos** degradam a segurança, "though to a lesser extent" [fonte: arXiv 2310.03693]. Consequência: **todo SFT de domínio precisa do eixo de segurança na bateria**, mesmo sem dado "perigoso".
- **DPO pode desalinhar sem querer ("likelihood displacement"):** durante o DPO a probabilidade da resposta *preferida* frequentemente **cai**; quando pares têm embeddings parecidos (medido pelo escore **CHES**), a massa pode migrar para respostas de sentido oposto — no exemplo do paper, treinar para recusar reduziu a taxa de recusa do Llama-3-8B-Instruct **de 74,4% para 33,4%** [fonte: arXiv 2410.08847]. Diagnóstico: `logps/chosen` caindo junto com `rewards/margins` subindo (§1.16).
- **Recusa é (quase) uma direção:** Arditi et al. mostraram em **13 modelos de chat abertos (até 72B)** que a recusa é mediada por um subespaço de dimensão 1; apagar a direção desliga a recusa, somá-la provoca recusa em prompts inofensivos, e a ortogonalização de pesos é um jailbreak white-box com impacto mínimo em outras capacidades [fonte: arXiv 2406.11717]. É a base da **abliteração** do Módulo 1 (cap. 19) — ver §2.7 para a relação com pós-treino, inclusive como **sonda barata de regressão de segurança** (projeção das ativações na direção de recusa antes/depois do fine-tuning).

### 1.14 Contaminação

- **Tipos:** (i) o benchmark está no pré-treino do base; (ii) **você** contamina ao gerar dados sintéticos com um professor que viu o benchmark, ou ao raspar a web; (iii) contaminação "de seleção": escolher checkpoint/hiperparâmetro olhando o conjunto de teste.
- **Detecção:**
  - sobreposição de n-gramas entre treino e avaliação (normalize texto; checar 8–13-gramas é prática comum **[recomendação; o limiar exato varia entre relatórios técnicos e não verifiquei um padrão único]**);
  - **Min-K% Prob** (Shi et al.): média do log-prob dos k% tokens menos prováveis; textos vistos têm menos "palavras-surpresa"; +7,4% sobre métodos anteriores no WikiMIA [fonte: arXiv 2310.16789];
  - conjuntos **com data** (LiveCodeBench, MathArena, competições do ano corrente) e conjuntos "paralelos" frescos (GSM1k) [fonte];
  - strings canário em datasets de avaliação.
- **No laboratório:** rode a descontaminação **do seu conjunto de SFT/DPO contra todos os conjuntos de avaliação** antes de treinar (script no Apêndice B.6). O professor do Módulo 1 pode ter memorizado GSM8K; se você pedir "gere problemas de matemática parecidos com estes", vai gerar quase-cópias.

### 1.15 Regressão de capacidades ("alignment tax", esquecimento)

- **LoRA aprende menos e esquece menos:** em código e matemática, LoRA "substantially underperforms full finetuning", mas preserva melhor o desempenho fora do domínio que weight decay/dropout; o fine-tuning completo aprende perturbações de posto **10–100× maior** que configurações típicas de LoRA [fonte: arXiv 2405.09673].
- **LoRA Without Regret (Thinking Machines, 2025-09-29):** LoRA iguala o full FT quando (1) é aplicado a **todas** as matrizes, "especially MLP and MoE layers" ("attention-only with rank 256 underperforms MLP-only with rank 128"), e (2) não está limitado por capacidade; LR ótimo de LoRA ≈ **10×** o do full FT; em RL com policy gradient, LoRA iguala o full FT "even with ranks as low as 1" (RL carrega ~O(1) bits por episódio); LoRA tolera **menos** batch grande que full FT [fonte: thinkingmachines.ai/blog/lora].
- **Bateria de regressão fixa (proposta para o laboratório):** IFEval (completo), GSM8K (completo, 0-shot CoT), AIME 2025 ou 2026 (k=4 amostras), HumanEval-instruct, MMLU-Pro (amostra estratificada de ~1.000 — via endpoint de *completions*, ver §1.18), BFCL subset, XSTest, + KL em texto genérico. Rode no base e em cada candidato; reporte Δ com IC (bootstrap pareado por item; para métricas binárias por item, teste de McNemar).

### 1.16 KL em relação à referência e diagnóstico de reward hacking

**Por que KL.** RLHF/DPO otimizam recompensa **com uma âncora**: maximize E[r] − β·KL(π‖π_ref). O KL realizado é, portanto, o "orçamento gasto" pelo pós-treino. Gao, Schulman & Hilton mostraram com um RM "ouro" sintético que a recompensa **verdadeira** sobe e depois **cai** à medida que se otimiza o RM proxy, com formas funcionais em d = √KL(π‖π_init): **best-of-n**: R(d) = d(α − β·d); **RL**: R(d) = d(α − β·log d) [fonte: arXiv 2210.10760; fórmulas transcritas de memória do paper — conferir]. Para best-of-n, KL = log n − (n−1)/n.

**Estimadores de KL por amostra** (amostras y ~ π_θ, razão r = π_ref(y)/π_θ(y) por token) [fonte: J. Schulman, "Approximating KL Divergence", 2020; o k3 é o usado no GRPO do DeepSeekMath]:
- k1 = −log r (não-viesado, alta variância, pode ser negativo);
- k2 = ½(log r)² (viesado, baixa variância);
- **k3 = (r − 1) − log r** (não-viesado, sempre ≥ 0). O `mlx-lm-lora` documenta exatamente `expm1(log_ref − log_policy) − (log_ref − log_policy)`, com o log-ratio limitado a 20 e a métrica `kl_clip_ratio` ("KL saturation") reportando quantas vezes o teto é atingido [fonte: README mlx-lm-lora].

**Sinais de alerta por algoritmo** (métricas nos nomes exatos que os trainers logam):

| Algoritmo | Métrica | Normal | Suspeito | Leitura provável |
|---|---|---|---|---|
| SFT | `loss` (treino) vs `eval_loss` | ambos caem | treino ↓, eval ↑ | sobreajuste; reduza épocas/LR |
| SFT | `mean_token_accuracy` | sobe devagar | ~1,0 cedo | memorização/dados duplicados |
| DPO | `rewards/accuracies` | 0,6–0,8 | → 1,0 rápido | pares fáceis ou viés de comprimento explorado |
| DPO | `rewards/margins` | sobe | sobe enquanto `logps/chosen` despenca | **likelihood displacement** (§1.13): ↑β, combine com NLL nas preferidas (`loss_type=["sigmoid","sft"]`, `loss_weights=[1.0, 1.0]` — multi-loss suportado pelo `DPOConfig`), `apo_zero`, ou filtre pares com CHES alto |
| DPO | comprimento médio das respostas | estável | cresce | exploração de verbosidade; `sigmoid_norm`/`ld_alpha`/SimPO |
| KTO | `kl` | positivo, estável | — | estimativa ruim se batch por dispositivo pequeno (doc: ≥ 4) |
| GRPO | `reward` / `rewards/<f>/mean` | sobe | sobe com `completions/mean_length` explodindo | hacking de comprimento/formato |
| GRPO | `frac_reward_zero_std` | < ~0,5 | → 1 | grupos todos certos/errados: sem sinal; dificuldade errada do dataset |
| GRPO | `entropy` | cai devagar | colapsa cedo | colapso de entropia (DAPO usa clip-higher, `epsilon_high`) |
| GRPO | `clip_ratio/region_mean`, `low_mean`, `high_mean` | pequenos | grandes | passos grandes demais; ↓LR |
| GRPO | `completions/clipped_ratio` | baixo | alto | `max_completion_length` curto; use `mask_truncated_completions=True` |
| GRPO | `kl` | só aparece com `beta>0` | crescendo sem parar | política se afastando; ↑beta ou pare |
| GRPO+vLLM | `sampling/sampling_logp_difference/mean`, `sampling/importance_sampling_ratio/*` | ~0 / ~1 | grandes | descompasso treino×inferência (dtype/kernels); a correção IS está ligada por padrão |

[fonte: docs TRL SFT/DPO/KTO/GRPO, métricas logadas]

**Reward hacking verificável (RLVR):** além das métricas, **leia amostras** (`log_completions=True`): respostas que imprimem várias caixas `\boxed{}`, que repetem o enunciado, que deixam a resposta só no `<think>`. Teste o verificador contra respostas adversariais antes de treinar.

### 1.17 Benchmarks que importam em 2026 (estado da arte, verificado onde possível)

| Benchmark | Mede | Situação em 2026 | Fonte/status |
|---|---|---|---|
| **Humanity's Last Exam (HLE)** | perguntas de fronteira, multidisciplinares | no índice AA v4.3.2: 2.158 questões, pass@1 com verificador de igualdade | [fonte: artificialanalysis.ai] |
| **GPQA Diamond** | ciência nível pós-grad | ainda discrimina; Qwen3.8-27B: 89,2 (card) / 88,92 BF16 (card NVIDIA) | [fonte: cards]; agregadores citam ~94% no topo **[não verificado]** |
| **SWE-bench Pro** (+ Pro Verified) | engenharia de software agêntica | substituiu o Verified | §1.11 |
| **Terminal-Bench (4.0)** | agente em terminal | 4.0 anunciado; AA usa 66 tarefas | [fonte] |
| **τ²-bench**, **BFCL V4** | ferramentas | ativos | §1.12 |
| **ARC-AGI-3** | benchmark **interativo** (jogos; aquisição de habilidade e eficiência) | competição ARC Prize 2026 no Kaggle; "100% score means AI agents can beat every game as efficiently as humans" | [fonte: arcprize.org]; placares não verificados |
| **MathArena (AIME 2026, HMMT fev/2026)** | matemática sem contaminação | topo perto de 97% em AIME 2026 segundo agregadores | datasets [verificado]; placares [não verificado] |
| **IFBench**, **AA-LCR**, **SciCode**, **CritPt**, **AA-Omniscience**, **GDPval-AA** | instruções, contexto longo, código científico, física de pesquisa, conhecimento+alucinação, trabalho econômico | compõem o AA Intelligence Index **v4.3.2** (10 avaliações; pesos: Agentes 30%, Código 20%, Geral 30%, Raciocínio científico 20%) | [fonte: artificialanalysis.ai/methodology] |
| Saíram do índice AA | MMLU-Pro, AIME 2025, MATH-500, LiveCodeBench, variantes antigas do Terminal-Bench | "legacy" | [fonte] (motivo não declarado na página) |

Para o **seu** modelo de 4–27B: os benchmarks de fronteira (HLE, ARC-AGI-3, SWE-bench Pro) são caros e pouco sensíveis a um SFT de domínio. Use-os só se o pós-treino mirar essas capacidades; caso contrário, a bateria da §1.15 + avaliação de domínio é o que detecta efeito.

### 1.18 Ferramentas de avaliação e como apontá-las para o vLLM/MLX

- **lm-evaluation-harness 0.4.13** [verificado: código]:
  - `--model local-chat-completions --model_args model=<nome>,base_url=http://HOST:PORT/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False` + `--apply_chat_template` para tarefas **generativas** (IFEval, GSM8K-CoT, AIME, HumanEval-instruct).
  - Tarefas de **log-verossimilhança/múltipla escolha** (MMLU, ARC…) **não** funcionam em chat-completions; use `--model local-completions` (endpoint `/v1/completions`) com `--apply_chat_template` [fonte: `docs/API_guide.md`].
  - Backend `vllm` direto: `--model vllm --model_args pretrained=<path>,tensor_parallel_size=1,lora_local_path=<adapter>,max_lora_rank=32,enable_thinking=False` — o wrapper liga `enable_lora` quando `lora_local_path` é dado e aceita `think_end_token`/`chat_template_args` [verificado: `lm_eval/models/vllm_causallms.py`].
  - Nomes de tasks úteis [verificado: YAMLs]: `ifeval`, `gsm8k`, `gsm8k_cot_zeroshot`, `gsm8k_cot_llama`, `aime24`, `aime25`, `humaneval_instruct`, `humaneval_plus`, `mmlu_pro`, `gpqa`, `minerva_math`, `bbh`.
- **mlx_lm.evaluate** (wrapper do lm-eval para MLX): `--model`, `--tasks`, `--num-shots`, `--apply-chat-template`, `--chat-template-args`, `--fewshot-as-multiturn`, `--limit`, `--confirm-run-unsafe-code`, `--temp/--top-p/--top-k` [verificado: `mlx_lm/evaluate.py`]. **Não tem `--adapter-path`** → avalie o modelo fundido, ou sirva com `mlx_lm.server --adapter-path` e use o lm-eval com `local-chat-completions`.
- **inspect-ai 0.3.273** (UK AISI): bom para avaliações agênticas e de segurança com sandbox [fonte: PyPI]; **lighteval 0.13.0** (sem release em 2026) [verificado].
- **exllamav3** traz `eval/model_diff.py` (KL/PPL/top-K entre dois modelos), `eval/ppl.py`, `eval/humaneval.py`, `eval/ifbench.py`, `eval/mmlu.py`, `eval/longctx.py`, `eval/spec_decode.py` [verificado: árvore do repo].
- **Arena-Hard-Auto**, **tau2-bench**, **BFCL** — cada um com o próprio harness (§1.6, §1.12).

---

## 2. Servindo modelos pós-treinados

### 2.1 Os três caminhos

| Caminho | O que vai para o servidor | Prós | Contras | Quando usar |
|---|---|---|---|---|
| **(A) Adapter em runtime** | base (BF16 ou quantizado) + adapter(s) LoRA carregados pelo vLLM | troca de adapter por requisição (`model=<nome-do-adapter>`); vários adapters no mesmo motor; zero custo de merge; A/B trivial | custo extra por token (kernels Punica/fused-MoE-LoRA), memória reservada por slot; nem toda arquitetura suporta; adapter treinado num base e servido noutro (quantizado) soma erros | experimentos, multi-tenant, comparar checkpoints de DPO lado a lado |
| **(B) Merge → BF16** | pesos W' = W + (α/r)·B·A gravados | idêntico ao treinado (a menos de arredondamento BF16); sem overhead | ocupa a memória de um BF16 inteiro (Qwen3.8-27B: 55,6 GB) e a banda de decode de um BF16 | referência para medir KLD da quantização; servidor de avaliação |
| **(C) Merge → re-quantizar** | checkpoint NVFP4/GPTQ/AWQ/EXL3/MLX-4bit do modelo fundido | velocidade e memória de produção (o que o Módulo 1 inteiro otimiza) | exige **nova calibração**; adiciona erro que precisa ser medido contra (B) | produção em GB10 / Mac |

Regra de bolso **[inferência]**: no GB10 o decode é limitado por banda (~273 GB/s nominais; ~230 GB/s efetivos segundo `glm-mods.md` §4.1); um 27B em BF16 lê ~55,6 GB por passo de decode (≈ 4 passos/s no teto de banda) contra ~22–25 GB nos checkpoints NVFP4 W4A4-MLP/FP8-atenção de referência (`nvidia/…` 21,92 GB; `RedHatAI/…` 24,69 GB) — por isso (C) é o destino natural, e (A)/(B) são ferramentas de desenvolvimento.

### 2.2 LoRA no vLLM 0.30.0 — referência

**Flags do `LoRAConfig`** [fonte: docs.vllm.ai `engine_args`; doc `features/lora.md` na tag v0.30.0, verificado]:

| Flag | Padrão | Valores | Efeito / cuidado |
|---|---|---|---|
| `--enable-lora` | `False` | bool | liga o subsistema (precisa estar ligado também para carga dinâmica) |
| `--lora-modules` | — | `nome=caminho` (legado) ou JSON `{"name":…,"path":…,"base_model_name":…}` | registra adapters na subida; aparecem em `GET /v1/models` com `parent` = base e `root` = caminho do adapter |
| `--max-loras` | **1** | int | quantos adapters **distintos** podem estar ativos no mesmo batch (slots na GPU) |
| `--max-lora-rank` | **16** | 1, 8, 16, 32, 64, 128, 256, 320, 512 | dimensiona os buffers; use o **maior posto real** dos seus adapters, não 256 "por garantia" (desperdiça memória e desempenho, diz a doc) |
| `--max-cpu-loras` | `None` (vira `max_loras` no `__post_init__` [verificado]) | int ≥ `max_loras` | cache LRU de adapters em RAM de CPU. **No GB10 a "RAM de CPU" é a mesma memória unificada** — cada adapter em cache compete com KV e pesos [inferência a partir de `memoria-unificada`] |
| `--lora-dtype` | `auto` | auto = dtype do base | — |
| `--fully-sharded-loras` | `False` | bool | por padrão só metade da computação LoRA é fragmentada no TP; ligado, usa camadas totalmente fragmentadas — "likely faster" com sequência longa, posto alto ou TP grande [verificado: docstring do `LoRAConfig`] |
| `--lora-target-modules` | `None` (= todos os suportados) | sufixos (`o_proj qkv_proj down_proj …`) | restringe onde aplicar LoRA no servidor (desempenho) |
| `--default-mm-loras` | `None` | JSON modalidade→adapter | aplica adapter automaticamente quando há áudio/imagem |
| `--enable-tower-connector-lora` | `False` | bool | LoRA experimental na torre/conector multimodal |
| `--specialize-active-lora` | `False` | bool | monta o grid do kernel pelo nº de adapters ativos e captura CUDA graphs separados para 1, 2, 4… até `max_loras` (só com `cudagraph_specialize_lora`); melhora uso variável de LoRA ao custo de subida e memória [verificado: `vllm/config/lora.py` v0.30.0] |
| `--enable-mixed-moe-lora-format` | `False` | bool | servir no mesmo motor adapters MoE 2D (megatron) e 3D (peft); exige declarar `is_3d_lora_weight` por adapter — **"vLLM does not inspect the checkpoint to verify"**: declaração errada "silently produce garbage outputs" |
| `--enable-moe-shared-loras` | `False` | bool | — |
| `--max-lora-cls-labels` | `None` | int | cabeças de classificação |

**Carga dinâmica** [fonte: doc]: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` + `POST /v1/load_lora_adapter` (`lora_name`, `lora_path`, opcional `base_model_name`, `is_3d_lora_weight`, `load_inplace`) e `POST /v1/unload_lora_adapter`. A doc avisa: "should not be used in production unless it is an isolated, fully trusted environment". Plugins resolvedores: `VLLM_PLUGINS=lora_filesystem_resolver` + `VLLM_LORA_RESOLVER_CACHE_DIR`, ou `lora_hf_hub_resolver` + `VLLM_LORA_RESOLVER_HF_REPO_LIST`.

**Seleção por requisição:** o campo `"model"` da API é o nome do adapter; offline, `LoRARequest(nome, id_único, caminho)` em `llm.generate(..., lora_request=...)` [fonte: doc].

**Interações com o que o Módulo 1 ensinou** [fonte: `vllm-ref.md`]:
- o **hash de bloco do prefix cache inclui o ID do LoRA** — o mesmo prompt em dois adapters não compartilha cache (correto, pois os K/V diferem);
- o despachante de CUDA graphs usa um `BatchDescriptor` com `has_lora` como parte da chave → batches com e sem LoRA usam grafos diferentes (mais capturas, mais memória de grafo).

**MoE + LoRA:** o vLLM ganhou um kernel `fused_moe_lora` que faz os GEMMs shrink/expand do LoRA dentro do `fused_moe` (gate_up e down), com uma dimensão extra no grid para o adapter ativo; há também variante FP8 [fonte: PR vllm#21229 e busca; status por arquitetura não verificado individualmente].

**Suporte por arquitetura (coluna LoRA de `supported_models`)** [fonte: docs.vllm.ai, consultado 2026-10-01; código v0.30.0 verificado onde indicado]:

| Arquitetura | LoRA | Observação |
|---|---|---|
| `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM` | ✅ | — |
| `Qwen3_5ForConditionalGeneration` (Qwen3.5/3.6/**3.8-27B**) | ✅ | [verificado: `qwen3_5.py` v0.30.0 declara `SupportsLoRA` e `packed_modules_mapping` com `qkv_proj←q/k/v_proj`, `gate_up_proj←gate/up_proj`, **`in_proj_qkvz←in_proj_qkv,in_proj_z`** e **`in_proj_ba←in_proj_b,in_proj_a`** (projeções fundidas do Gated DeltaNet)] |
| `GlmMoeDsaForCausalLM` (GLM-5, 5.1, 5.2) | ✅ | o GLM-5.3-Flash do Módulo 1 roda numa classe customizada (`Glm5NextModel`, overlay da receita) — **suporte a LoRA desse caminho não verificado; assuma que não há** [inferência] |
| `DeepseekV4ForCausalLM` | **❌** (coluna vazia) | adapters para o DSV4-Flash só via merge + re-quantização |
| `LlamaForCausalLM`, `Gemma4ForCausalLM` | ✅ | — |

**Custo de memória dos slots [estimativa].** O vLLM pré-aloca, por módulo-alvo, buffers de A (r × in) e B (out × r) para `max_loras` slots no posto `max_lora_rank`. Para o Qwen3.8-27B com LoRA em todas as lineares de texto (q/k/v/o, in_proj_qkv/in_proj_z/out_proj, gate/up/down): **108,8 M parâmetros por slot em r=16, 217,6 M em r=32, 435,2 M em r=64** (conta no Apêndice A.7) → em BF16, 0,22 / 0,44 / 0,87 GB **por slot**. Com `--max-loras 4 --max-lora-rank 32`: ≈ 1,7 GB. Pequeno perto de 55,6 GB de pesos, mas não desprezível no orçamento de KV de um GB10.

### 2.3 LoRA sobre base quantizado

- O vLLM documenta e mantém um exemplo de **LoRA sobre bases quantizados** com `bitsandbytes` (QLoRA), `awq` e `gptq` (`examples/features/lora/lora_with_quantization_offline.py`, `max_lora_rank=64`, `max_loras=4`); o caso bitsandbytes **exige o plugin `vllm-bnb-plugin`** (o bitsandbytes virou plugin externo na v0.28 — `vllm-ref.md`) [verificado: arquivo na tag v0.30.0].
- Mecanismo **[inferência a partir da arquitetura do vLLM]**: a camada LoRA envolve a camada base e soma x·Aᵀ·Bᵀ ao resultado de `quant_method.apply(...)`; por isso, em princípio, qualquer método linear quantizado aceita LoRA. **NVFP4 (ModelOpt/compressed-tensors) + LoRA e EXL3 + LoRA no sm_121 não verifiquei** — teste antes de depender disso.
- **Descasamento:** um adapter treinado sobre o base BF16 e servido sobre o base NVFP4 vê ativações levemente diferentes; o erro do adapter e o da quantização se somam. Um adapter **QLoRA** (treinado sobre NF4) servido sobre NF4 é consistente, mas o NF4 do bitsandbytes não é o formato rápido do GB10. Meça KL(base_BF16+LoRA ‖ base_quant+LoRA) (§2.6) antes de escolher (A) com base quantizado.

### 2.4 Merge

**PEFT (HF).** Carregue o base **em BF16** (nunca o 4-bit do treino QLoRA — `merge_and_unload` sobre pesos 4-bit dequantiza, soma e re-quantiza, perdendo parte do delta), aplique o adapter, funda, salve **com todos os arquivos não-peso** (tokenizer, `chat_template.jinja`, `preprocessor_config.json`/processor para VLMs, `generation_config.json`). Para o Qwen3.8-27B a classe é `Qwen3_5ForConditionalGeneration` — a mesma usada pelos exemplos oficiais do llm-compressor [verificado]. Script completo na §5.9.

**Verificação obrigatória do merge** (barato): gere logits para 8 prompts com (i) base+adapter via PEFT sem merge e (ii) modelo fundido; exija max |Δlogit| pequeno (ordem do arredondamento BF16) e top-1 idêntico em ≥ 99,9% das posições **[recomendação]**.

**DoRA:** `merge_and_unload` também funde (a magnitude é reaplicada); a inferência sem merge é mais cara.

**MLX.** `mlx_lm.fuse --model <base> --adapter-path <adapters> --save-path <saida> [--dequantize] [--export-gguf --gguf-path …] [--upload-repo …]` [verificado: `mlx_lm/fuse.py`]. **Detalhe importante [verificado: `mlx_lm/tuner/lora.py`]**: se o base é quantizado (QLoRA), o `fuse` **dequantiza a camada, soma o delta e re-quantiza com os mesmos parâmetros (RTN, sem calibração)**, a menos que você passe `--dequantize`. O delta LoRA é pequeno perto do passo de quantização de 4 bits, então parte dele **some no arredondamento**. Prefira: fundir sobre o base **BF16** (`mlx-community/<modelo>-bf16`) e re-quantizar depois com um método calibrado (`mlx_lm.dwq`/`awq`/`gptq`, §2.5). O GGUF via `--export-gguf` só funciona para `llama`, `mixtral`, `mistral` [verificado: código].

**OpenRLHF.** `python -m openrlhf.cli.lora_combiner --model_path <base> --lora_path <ckpt> --output_path <saida> [--is_rm] --ds.param_dtype bf16` [fonte: README].

### 2.5 Re-quantizar depois do fine-tuning

**Princípios (valem para todo formato):**

1. **Nada do base quantizado é reaproveitável.** Escalas por grupo/bloco, Hessianos (GPTQ, EXL3/QTIP), escalas de ativação (AWQ, SmoothQuant), escalas globais NVFP4 e escalas estáticas de KV foram estimadas com os pesos/ativações **do base**. Depois do merge, refaça tudo a partir do **modelo fundido em BF16**.
2. **Dados de calibração = distribuição de uso.** Misture (i) amostras do seu SFT/DPO (no **template de chat real**, com `<think>` se o modelo for servido pensando) e (ii) texto geral/chat genérico — só domínio sobreajusta a quantização ao domínio e estraga o resto; só genérico subestima as ativações do domínio **[recomendação; racional: GPTQ/AWQ/EXL3 minimizam erro sob a distribuição de ativações vista na calibração]**.
3. **Mantenha o que o original mantinha em alta precisão**: embeddings, `lm_head` (ou FP8/NVFP4 se a receita oficial fizer), roteadores MoE, `in_proj_a/in_proj_b` do GDN (projeções minúsculas, 48 saídas no 27B), torre de visão, cabeça MTP (se não foi treinada).
4. **Meça contra o merge BF16** (§2.6), não contra o base original.

**Por formato:**

| Formato | Ferramenta (versão) | Comando/receita de partida | Calibração | Observações |
|---|---|---|---|---|
| **W4A16 INT4 (GPTQ+AWQ)** | llm-compressor 0.14.0 | `examples/quantization_w4a16/qwen3_8_gptq_awq_example.py`: `AWQModifier(duo_scaling="both")` + `GPTQModifier(targets="Linear", scheme="W4A16", ignore=[visual, lm_head, embed_tokens, linear_attn.in_proj_a, linear_attn.in_proj_b])` | `dataset="perfectblend"`, `splits="train[:512]"`, `max_seq_length=4096`, `num_calibration_samples=512` | [verificado: arquivo]. Troque o dataset pelo seu mix (exemplo `custom_dataset_example.py` aceita qualquer `Dataset` HF já tokenizado) |
| **NVFP4 W4A4 (MLP) + FP8 dinâmico (atenção/GDN) + KV FP8** | llm-compressor 0.14.0 | `examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py`: grupo `attention` = `self_attn.(q|k|v|o)_proj`, `linear_attn.(in_proj_qkv|in_proj_z|out_proj)`, `lm_head` e **o MLP das 8 últimas camadas (56–63)** em `FP8_DYNAMIC`; grupo `mlp` = demais MLPs em `NVFP4`; `kv_cache_scheme` FP8 por tensor estático (`static_minmax`) | idem (512 × 4096) | [verificado: arquivo]. Note a escolha "últimas camadas mais precisas" — mantenha ao re-quantizar o seu fine-tune |
| **NVFP4 (ModelOpt)** | nvidia-modelopt 0.47.0 (main) | `python examples/hf_ptq/hf_ptq.py --pyt_ckpt_path <merge_bf16> --recipe models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_local_hessian --dataset <seu.jsonl>,cnn_dailymail --calib_size 512,512 --calib_seq 4096 --export_path <saida>` | `--dataset` aceita **`.jsonl` local com coluna `messages`** (aplica o template) e listas separadas por vírgula; `--calib_size` padrão 1024, `--calib_seq` padrão 512 | [verificado: `hf_ptq.py` e `dataset_utils.py` no main; a receita específica do Qwen3.8-27B existe em `modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/`]. O playbook do Spark (tag 0.45.0) usa o caminho antigo `examples/llm_ptq/` e receitas `huggingface/...`; no main é `examples/hf_ptq/` e `model_type/...`/`models/...` — **fixe a versão** |
| Referência NVIDIA | — | card `nvidia/Qwen3.8-27B-NVFP4`: NVFP4 nos MLPs e `lm_head`, FP8 em atenção e linear-attention, KV `fp8_e4m3`; calibração **2.048 amostras do Nemotron-Post-Training-Dataset-v3**, algoritmo **local Hessian** | — | [fonte: card]. Δ vs BF16: GPQA Diamond 88,92→88,01; Terminal-Bench 75,56→74,02; AA-LCR 72,63→73,38; MMMU-Pro 75,14→74,86; SciCode 47,93→48,41; IFBench 80,07→78,93 |
| **EXL3** | exllamav3 v1.5.3 | `python convert.py -i <merge_bf16> -o <saida> -w <workdir> -b 4.0 -hb 6 [-mb 4] [-vb 16] [-hq] [-cd <cal.safetensors>]` | padrão: mistura embutida, `-cr 250` linhas × `-cc 2048` colunas; **`-cd/--cal_data`** aceita traço próprio (safetensors de linhas empacotadas, gerado por `sc_trace.py`) | [verificado: `doc/convert.md`]. `-mb` (MTP, sem calibração, padrão 4; 16 = sem quantizar), `-vb` (visão), `-hq` sobe bits de atenção/shared-expert por +0,05–0,10 bpw em MoE, `-r` retoma, `-d 0,1` multi-GPU. Para PLE (Qwen3.8-Flash-Next) existe `-ngb/-ngf` |
| **FP8 dinâmico** | llm-compressor (`scheme="FP8_DYNAMIC"`) | sem calibração | — | baseline seguro e rápido de produzir; bom para isolar "o problema é a quantização 4-bit ou o fine-tune?" |
| **MLX 4-bit** | mlx-lm 0.32.0 | `mlx_lm.convert --hf-path <fundido_bf16> -q` (RTN); calibrados: `mlx_lm.dwq --model <bf16> --bits 4 --group-size 64 --data-path <hf-dataset> --num-samples 1024 --max-seq-length 1025 --mlx-path <saida>`; `mlx_lm.awq` (`--num-samples 32`, `--n-grid 10`), `mlx_lm.gptq`, `mlx_lm.dynamic_quant --target-bpw 4.5` | DWQ destila do não-quantizado (professor) ajustando escalas/biases; `--data-path` padrão `allenai/tulu-3-sft-mixture` | [verificado: `LEARNED_QUANTS.md` e `quant/dwq.py`]. Dicas oficiais: DWQ funciona melhor para 2–4 bits; `--group-size 32` dobra parâmetros ajustáveis |
| QAT (MLX) | mlx-lm-lora 3.1.3 | `--qat-enable --qat-bits 4 --qat-group-size 64` | projeta os pesos treináveis na grade quantizada a cada passo; suportado em SFT, DPO, ORPO | [fonte: README]. Alternativa a "treinar → quantizar" quando o alvo é 4-bit |

### 2.6 Medindo KLD contra o base (e contra o merge)

**Duas perguntas, duas referências:**

| Pergunta | P (referência) | Q (candidato) | Texto de avaliação | Esperado |
|---|---|---|---|---|
| O que o pós-treino mudou? | base original BF16 | fine-tunado BF16 (merge ou base+adapter) | (a) texto **genérico** (wikitext/C4/chat genérico); (b) texto **do domínio** | (a) pequeno; (b) grande — é o efeito pretendido |
| Quanto a quantização estragou? | **fine-tunado BF16 (merge)** | fine-tunado quantizado | texto genérico + domínio + respostas do próprio modelo (com `<think>` se for o uso) | da ordem do que o mesmo formato causa no base |
| O adapter sobre base quantizado funciona? | base BF16 + LoRA | base quantizado + LoRA | idem | ≈ KLD do base quantizado sozinho |

Use KL(P‖Q) = Σ_v p(v)·(log p(v) − log q(v)) por posição, teacher-forced, média sobre posições (nats). Reporte também **mediana e p90 por token**, PPL de P e Q, e **concordância de top-1** — o `model_diff.py` do exllamav3 já imprime tudo isso, inclusive KLD "by reference confidence" (por faixa de probabilidade do top-1 de referência) [verificado: código].

**Âncoras do Módulo 1 (ordem de grandeza, mesmo protocolo da receita GLM)** [fonte: `quant.md` §4.2]: GLM-5.3-Flash, KLD(teacher‖model) em 51.175 posições (5 execuções, 25 janelas): EXL3 K6 0,013723; FP8 oficial 0,020615; **EXL3 4 bpw (servido) 0,024555**; NVFP4 0,060535 nats. Um NVFP4 do seu fine-tune muito acima do que o NVFP4 oficial causa no base (meça os dois com o mesmo script!) indica calibração ruim.

**Ferramentas:**

1. **Exato, pesos+kernels reais, EXL3 × BF16:** `python eval/model_diff.py -ma <exl3> -mb <merge_bf16> -r 100 -l 2048 [-tkm 5] [-d 0]` (100 linhas × 2.048 tokens por padrão) [verificado: argumentos]. No script, **B é a referência**: a linha "KL divergence (A, B)" é `compute_kl_div(A, B)` = KL(softmax(B)‖softmax(A)), isto é, KL(referência‖candidato), e a estatística "by reference confidence" usa o top-1 de B [verificado: `eval/model_diff.py` e `exllamav3/util/measures.py`] — por isso o BF16 vai em `-mb`. O exllamav3 v1.5.3 lista `Qwen3_5ForConditionalGeneration` (a classe do Qwen3.8-27B), `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next), `Glm5NextForConditionalGeneration` e `DeepseekV4ForCausalLM` entre as arquiteturas suportadas [verificado: README]; carregar o BF16 não quantizado como modelo B é o uso previsto do script **[inferência]**. Memória **[estimativa]**: 55,6 GB (BF16) + ~15–16 GB (27B a 4,0 bpw + cabeça a 6 bpw) ≈ 72 GB → cabe num Spark.
2. **Exato, só pesos, compressed-tensors (NVFP4/W4A16) × BF16:** script HF (Apêndice B.4) que roda P e grava, por posição, os top-64 log-probs + o log-sum-exp; depois roda Q e calcula KL truncado ao suporte top-64 de P + termo de cauda. Carregar o compressed-tensors no transformers **dequantiza** os pesos (isola o erro de pesos, mas **não** usa os kernels W4A4 nem quantiza ativações como o vLLM faria) **[inferência; conferir o modo de carga do compressed-tensors na versão instalada]**.
3. **In situ, kernels de produção, aproximado:** dois servidores vLLM (um por Spark: merge BF16 no A, quantizado no B) e `POST /v1/completions` com `"prompt": <texto>`, `"max_tokens": 1`, `"prompt_logprobs": 20` — o protocolo de completions do vLLM 0.30.0 tem `prompt_logprobs: int | None` [verificado: `entrypoints/openai/completion/protocol.py`]; o limite por padrão é 20 (sobe com `--max-logprobs`; `-1` remove o teto, recurso que o TRL usa para destilação) [fonte: doc TRL vLLM]. Com top-20 dos dois lados você calcula KL truncado, NLL do token real e top-1 agreement (Apêndice B.5). É uma **proxy** — reporte como tal.
4. **Ruído de fundo:** rode P contra P com outra semente/tamanho de lote (ou outro boot, como a receita GLM faz) e trate esse valor como piso.
5. **Sinal grátis de deriva: aceitação do MTP.** O Qwen3.8-27B tem `mtp_num_hidden_layers: 1` [verificado: config]. A cabeça MTP foi treinada para prever o **base**; depois de um fine-tune (que não treinou a MTP), a taxa de aceitação na decodificação especulativa cai na proporção da deriva — o mesmo fenômeno da "aceitação ~0" que o canário da receita GLM usa para detectar modelo quebrado (`glm-mods.md` M50). Compare aceitação por posição antes/depois com `--per-request-spec-decode-metrics` (v0.29+) [fonte: `vllm-ref.md`] **[inferência sobre o efeito; magnitude não medida]**.

### 2.7 Abliteração × pós-treino

O capítulo 19 do Módulo 1 trata a abliteração como **álgebra sobre pesos congelados**. Visto do Módulo 2:

| Aspecto | Abliteração (Módulo 1) | Pós-treino (Módulo 2) |
|---|---|---|
| Mecanismo | edição fechada: W' = (I − α·r̂r̂ᵀ)·W nas matrizes que **escrevem** no resíduo (`o_proj`; DS: em runtime x ← x − λ(x·v)v) | gradiente sobre uma perda (NLL, DPO, recompensa) |
| Dados | ~centenas de prompts para estimar a direção r (diferença de médias de ativações) | milhares de exemplos rotulados/pares/recompensas |
| Posto da mudança | **1** por matriz editada | LoRA: r (8–64); full FT: alto (10–100× o de LoRA típico, §1.15) |
| Custo | minutos, sem backprop | horas–dias |
| Efeito colateral | pouco, se r for bem estimada (Arditi et al.) | esquecimento, deriva de estilo, regressão de segurança |
| No GLM do Módulo 1 | `ABLIT_METHOD=proj` (α=3 por padrão) **ou** `transplant`: troca byte a byte os `o_proj` L15–45 pelos de um checkpoint doador (`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4`), ~2,7 GiB por range-request, 32/32 casos de teste vs 9/32 da projeção | — |

Relações práticas **[inferência, salvo indicação]**:

1. **O transplante do GLM *é* um merge parcial servido na carga.** Ele aplica um delta de posto cheio, restrito a `o_proj`, produzido por pós-treino/edição de terceiros — conceitualmente o caminho (B) da §2.1 limitado a um subconjunto de tensores. Só funciona byte a byte porque o `o_proj` é **BF16 nativo** tanto no checkpoint NVFP4 doador quanto no EXL3 servido [fonte: `glm-mods.md` M47]. Se o seu fine-tune mexer em matrizes que o formato servido quantiza, não há transplante: precisa re-quantizar.
2. **Ordem das operações importa.** Abliterar e depois fazer SFT/DPO pode reintroduzir recusas (os dados de chat do professor contêm recusas) — ou o contrário: o SFT benigno sozinho já corrói recusas (Qi et al.). Abliterar **depois** do pós-treino exige re-estimar r no modelo novo (a direção muda com o fine-tune).
3. **DPO sobre base abliterado:** a referência π_ref do DPO passa a ser o modelo abliterado; o termo KL ancora no comportamento sem recusa. O `mlx-lm-lora` usa justamente checkpoints `...-abliterated` nos exemplos de SFT/DPO [fonte: README].
4. **A direção de recusa como sonda de avaliação:** projete as ativações do resíduo (camadas do meio para o fim) de prompts nocivos e inofensivos sobre r̂ estimada no **base**; uma queda grande da projeção média após o fine-tune antecipa regressão de segurança antes de rodar StrongREJECT/HarmBench — barata e determinística.
5. **Abliteração não substitui avaliação de segurança**: a receita GLM reporta "KLD não medido com ablit" [fonte: `glm-mods.md` M47]. Para um modelo abliterado ou pós-treinado, os eixos da §1 continuam obrigatórios. Uso responsável: remover recusas sem governança é decisão de política, não de engenharia — documente quem pode usar o modelo e para quê.

### 2.8 Ponte com as receitas de fronteira do Módulo 1

Por que o laboratório **não** pós-treina os modelos do Módulo 1 **[estimativa de memória]**:

| Modelo | Parâmetros | BF16 só de pesos | 2× Spark (≈ 2 × 121,7 GiB de MemTotal) |
|---|---|---|---|
| GLM-5.3-Flash | 321,3 B (18 B ativos) | ≈ 643 GB | não cabe nem para LoRA BF16; QLoRA 4-bit ≈ 160+ GB só de pesos + estados → no limite teórico e **sem ferramenta pronta** para a arquitetura customizada (KDA, mHC, DSA) [inferência] |
| DeepSeek-V4-Flash | 284 B (13 B ativos) | ≈ 568 GB | idem; e o vLLM **não** tem LoRA para `DeepseekV4ForCausalLM` |
| Qwen3.8-Flash-Next | 125 B no principal (~6 B ativos) + PLE | > 250 GB | idem |

O caminho realista para esses modelos é: treinar adapters em infraestrutura maior → **merge** → **re-quantizar com as mesmas ferramentas do Módulo 1** (exllamav3 para EXL3, com `-ngb` para PLE; llm-compressor/ModelOpt para NVFP4 — o llm-compressor já publica `RedHatAI/GLM-5.3-Flash-NVFP4` e `RedHatAI/Qwen3.8-Flash-Next-NVFP4` com exemplos [verificado: README do llm-compressor]) → servir com a receita existente. O laboratório da §5 usa o **Qwen3.8-27B**, que é da mesma família do Qwen3.8-Flash-Next (`model_type: qwen3_5`, híbrido 3:1 Gated DeltaNet:atenção com saída gated, `head_dim` 256, `partial_rotary_factor` 0,25 [verificado: config]), para que cada etapa — merge, calibração, NVFP4, KV FP8, MTP, vLLM — seja a mesma, só que em escala que cabe.

---

## 3. A receita: visão geral, dados e formatos

### 3.0 O cenário e as fases

**Objetivo do laboratório:** transformar um modelo aberto num **assistente técnico em PT-BR de engenharia de inferência** (o "tutor" da plataforma): responde perguntas sobre flags do vLLM, formatos numéricos, contas de memória/banda e receitas de serviço, de forma correta, concisa e com números. É um alvo realista (domínio estreito, avaliável) e conversa com o copiloto MLX que a plataforma já tem (`mlx-copilot.md` §2.8).

| Fase | O que sai | Onde roda | Seção |
|---|---|---|---|
| F0 — linha de base congelada | conjuntos de avaliação (domínio + bateria geral), números do base | Spark (vLLM) / Mac (mlx_lm.server) | §3.3, §4.1, §5.3 |
| F1 — dados | `sft.jsonl`, `pref.jsonl`, `kto.jsonl`, `grpo.jsonl` descontaminados | **Sparks servindo as receitas do Módulo 1** como professor e juízes | §3.2 |
| F2 — SFT (LoRA/QLoRA) | adapter SFT | Mac (§4.2) / 2× Spark DDP (§5.4) | — |
| F3 — preferências (DPO; variantes IPO/SimPO/KTO) | adapter DPO | Mac (§4.3) / 2× Spark (§5.6) | — |
| F4 — RL com recompensa verificável (GRPO) | adapter GRPO | Mac, modelo pequeno (§4.4) / Spark A treina + Spark B gera (§5.7) | — |
| F5 — avaliação e portões | relatório com IC, decisão | Spark/Mac | §1, §5.8 |
| F6 — merge | BF16 fundido | Spark (§5.9) / Mac (§4.6) | §2.4 |
| F7 — re-quantização | NVFP4 / EXL3 / W4A16 / MLX-4bit | Spark (§5.10) / Mac (§4.6) | §2.5 |
| F8 — KLD e regressão pós-quant | relatório | Spark (§5.11) | §2.6 |
| F9 — serviço | endpoint OpenAI | vLLM 0.30.0 (§5.12) / mlx_lm.server (§4.6) | §2.2 |

### 3.1 Formatos de dados (exemplos copiáveis)

> Todos JSONL (um objeto por linha). Os exemplos estão em PT-BR porque o alvo é PT-BR.

**(a) SFT conversacional — TRL `SFTTrainer`, `mlx_lm.lora` (formato `chat`), `mlx_lm_lora.train --train-mode sft`** [fonte: docs TRL; `LORA.md`; README mlx-lm-lora]:
```json
{"messages": [{"role": "system", "content": "Você é um engenheiro de inferência. Responda em PT-BR, com números e unidades."}, {"role": "user", "content": "Quanto ocupa o KV de um token no fp8_ds_mla do GLM?"}, {"role": "assistant", "content": "656 bytes por token por camada DSA: 512 B do latente NoPE em FP8, 16 B de escalas e 128 B de RoPE (64 valores BF16)."}]}
```

**(a′) SFT com raciocínio para Qwen3.5/3.6/3.8 — use o campo `reasoning_content`, não `<think>` dentro de `content`.** O template do Qwen3.8-27B renderiza a mensagem do assistente como `<think>\n{reasoning_content}\n</think>\n\n{content}` e **não** procura `<think>` dentro de `content` [verificado: `chat_template.jinja` do `Qwen/Qwen3.8-27B`, linhas 111–117]; colocar o raciocínio inline gera dois blocos de pensamento:
```json
{"messages": [{"role": "user", "content": "Um 27B em NVFP4 cabe em quantos GB?"}, {"role": "assistant", "reasoning_content": "27,8e9 parâmetros × 4,5 bits efetivos (4 bits + escala E4M3 a cada 16) ≈ 15,6 GB, mais tensores mantidos em FP8/BF16…", "content": "Cerca de 16 GB só para os pesos NVFP4; o checkpoint real fica maior porque atenção, linear-attention e lm_head ficam em FP8/BF16."}], "chat_template_kwargs": {"enable_thinking": true}}
```
O `SFTTrainer` aplica `example["chat_template_kwargs"]` por exemplo ao renderizar [verificado: `trl/trainer/sft_trainer.py`, `**example.get("chat_template_kwargs", {})`].

**(b) SFT prompt–completion (perda só na resposta por padrão)** [fonte: doc SFT]:
```json
{"prompt": [{"role": "user", "content": "Qual o padrão de --max-lora-rank no vLLM?"}], "completion": [{"role": "assistant", "content": "16. Aceita 1, 8, 16, 32, 64, 128, 256, 320 ou 512."}]}
```

**(c) Ferramentas (tool calling)** — coluna `tools` com JSON Schemas + mensagens com `tool_calls` e papel `tool` [fonte: docs TRL "Tool Calling with SFT/DPO/KTO"; `LORA.md` formato `tools`]:
```json
{"messages": [{"role": "user", "content": "Quantos tokens de KV cabem em 14 GiB a 656 B/token/camada com 11 camadas?"}, {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{\"expr\": \"14*2**30/(656*11)\"}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "2083202.0"}, {"role": "assistant", "content": "≈ 2,08 milhões de tokens."}], "tools": [{"type": "function", "function": {"name": "calc", "description": "Avalia uma expressão aritmética", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}]}
```

**(d) Pares de preferência — TRL `DPOTrainer`/`RewardTrainer`/`CPOTrainer` (formato conversacional, prompt explícito, recomendado)** [fonte: doc DPO]:
```json
{"prompt": [{"role": "user", "content": "Por que o GLM não usa NVFP4 nos experts?"}], "chosen": [{"role": "assistant", "content": "Porque o KLD medido do NVFP4 (0,0605 nats) foi 2,5× o do EXL3 4 bpw (0,0246) e os kernels NVFP4 do SM12x são só de MHA denso, sem fp8_ds_mla."}], "rejected": [{"role": "assistant", "content": "Porque o GB10 não suporta FP4."}], "chat_template_kwargs": {"enable_thinking": false}}
```
Variantes aceitas: prompt implícito (`chosen`/`rejected` contendo a conversa inteira) e formato padrão em texto puro; coluna opcional `margin` para o `RewardTrainer` [fonte: doc TRL Reward/paper index].

**(e) Pares para `mlx_lm_lora.train --train-mode dpo|cpo|orpo`** [fonte: README mlx-lm-lora]:
```json
{"system": "Responda em PT-BR.", "prompt": "Por que o GLM não usa NVFP4 nos experts?", "chosen": "Porque o KLD medido…", "rejected": "Porque o GB10 não suporta FP4."}
```
ORPO aceita `preference_score` opcional.

**(f) KTO — sinal binário, não pareado — TRL `KTOTrainer`** (se você passar pares, o trainer converte: `chosen`→`label=True`, `rejected`→`label=False`) [fonte: doc KTO]:
```json
{"prompt": [{"role": "user", "content": "Qual o padrão de --max-loras?"}], "completion": [{"role": "assistant", "content": "1."}], "label": true}
{"prompt": [{"role": "user", "content": "Qual o padrão de --max-loras?"}], "completion": [{"role": "assistant", "content": "Ilimitado; o vLLM carrega todos os adapters que couberem."}], "label": false}
```

**(g) GRPO / RLVR — TRL `GRPOTrainer`**: coluna obrigatória `prompt`; demais colunas chegam às funções de recompensa como `**kwargs`. O `trl.rewards.accuracy_reward` espera a coluna **`solution`** e verifica com `math_verify` (prioriza `\boxed{}`) [verificado: `trl/rewards/accuracy_rewards.py`]:
```json
{"prompt": [{"role": "user", "content": "Natalia vendeu presilhas para 48 amigas em abril e metade disso em maio. Quantas no total? Ponha a resposta final em \\boxed{}."}], "solution": "72"}
```

**(h) GRPO em MLX — `mlx_lm_lora.train --train-mode grpo`**: `{"prompt": "...", "answer": "72", "system": "..."}` (opcional `type`); `mlx-community/gsm8k` já tem colunas `question`, `answer`, `prompt`, `reasoning` [verificado: datasets-server]; funções de recompensa recebem `prompts=`, `completions=`, `answer=`, `types=` (atenção: **`answer` no singular**) [fonte: README].

**(i) verl (parquet)** [verificado: `examples/data_preprocess/gsm8k.py`]: `{"data_source": "openai/gsm8k", "prompt": [{"role": "user", "content": "..."}], "ability": "math", "reward_model": {"style": "rule", "ground_truth": "72"}, "extra_info": {...}}`.

**(j) OpenRLHF**: chaves configuráveis — SFT `--data.input_key question --data.output_key response` (ou `--data.apply_chat_template`); RM/DPO `--chosen_key chosen --rejected_key rejected`; RL `--data.input_key context_messages` [fonte: README].

**(k) mlx-lm puro** (`mlx_lm.lora`): `chat`, `tools`, `completions` (`{"prompt":…,"completion":…}`) e `text`; chaves customizáveis via YAML (`prompt_feature`, `completion_feature`, `chat_feature`) [fonte: `LORA.md`]. Com `--mask-prompt`, em `chat` **só a última mensagem** conta como completion [fonte: `LORA.md`] — conversas multi-turno treinam só o último turno.

### 3.2 Gerando e curando dados com as receitas do Módulo 1

Os Sparks, antes de virarem máquinas de treino, servem os modelos do Módulo 1 — use-os como **professor** e **painel de juízes**. Todos expõem API OpenAI na porta **8888** com os nomes `GLM-5.3-Flash-EXL3`, `qwen3.8-flash-next`, `deepseek-v4-flash-vision-exp` [fonte: `glm-launch.md`, `qwen-launch.md`, `ds-launch.md`]. Como cada receita ocupa os dois nós, rode uma de cada vez (professor → desliga → juiz A → desliga → juiz B).

> Licença: verifique se os termos do modelo-professor permitem usar suas saídas para treinar outro modelo antes de publicar qualquer coisa derivada.

**Passo 1 — sementes a partir da própria plataforma.** Os 48 capítulos têm glossários `chapters/NN-slug.glossary.json` no formato `{slug: {"t": título, "d": definição}}` (1.079 termos) e o `docs/APANHADO.md` lista 1.484 parâmetros/modificações das receitas [verificado: arquivos locais]. Cada termo vira 2–4 perguntas.

```python
# f1_seeds.py — gera perguntas-semente a partir dos glossários (PT-BR)
import glob, json, random, re
random.seed(0)
seeds = []
for f in sorted(glob.glob("chapters/*.glossary.json")):
    if ".en." in f:
        continue
    for slug, item in json.load(open(f)).items():
        d = re.sub(r"<[^>]+>|\\\(|\\\)", "", item["d"])
        seeds.append({"chapter": f.split("/")[-1][:2], "term": item["t"], "context": d})
random.shuffle(seeds)
json.dump(seeds, open("data/seeds.json", "w"), ensure_ascii=False, indent=0)
print(len(seeds), "termos")
```

**Passo 2 — perguntas e respostas do professor (n=4 por pergunta, para virar pares).**

```python
# f1_teacher.py — professor = receita GLM do Módulo 1 (vLLM, porta 8888)
import json, concurrent.futures as cf
from openai import OpenAI
cli = OpenAI(base_url="http://SPARK_HEAD:8888/v1", api_key="x")
TEACHER = "GLM-5.3-Flash-EXL3"
SYS = ("Você é um engenheiro de inferência sênior. Responda em PT-BR, de forma correta e concisa, "
       "com números e unidades quando couber. Se não souber, diga que não sabe.")

def ask(seed):
    q = cli.chat.completions.create(model=TEACHER, temperature=0.8, max_tokens=400, messages=[
        {"role": "user", "content": f"Escreva 3 perguntas técnicas, diferentes entre si, que um estudante faria sobre "
                                    f"'{seed['term']}'. Contexto: {seed['context']}\nUma por linha, sem numeração."}])
    out = []
    for question in [l.strip() for l in q.choices[0].message.content.splitlines() if l.strip()][:3]:
        # se o professor pensar por padrão, o vLLM (com --reasoning-parser) devolve o raciocínio em campo separado;
        # garanta max_tokens suficiente ou desligue o pensamento pelo chat_template_kwargs que o template DELE aceitar
        r = cli.chat.completions.create(model=TEACHER, n=4, temperature=0.9, max_tokens=900,
                                        messages=[{"role": "system", "content": SYS},
                                                  {"role": "user", "content": question}])
        out.append({"term": seed["term"], "chapter": seed["chapter"], "question": question,
                    "answers": [c.message.content for c in r.choices]})
    return out

seeds = json.load(open("data/seeds.json"))
with cf.ThreadPoolExecutor(16) as ex, open("data/teacher.jsonl", "w") as w:
    for rows in ex.map(ask, seeds):
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
```

**Passo 3 — julgamento pareado com troca de posição, por um juiz de outra família** (ex.: `qwen3.8-flash-next` ou `deepseek-v4-flash-vision-exp`; ver §1.4 sobre autopreferência e o parentesco Qwen↔Qwen3.8-27B — se o aluno for o Qwen3.8-27B, prefira o DeepSeek como juiz):

```python
# f1_judge.py — gera pref.jsonl (pares) e kto.jsonl (binário) com troca de posição
import json, itertools, re
from openai import OpenAI
cli = OpenAI(base_url="http://SPARK_HEAD:8888/v1", api_key="x")
JUDGE = "deepseek-v4-flash-vision-exp"
RUBRIC = ("Compare as respostas A e B à pergunta. Critérios, em ordem: (1) correção técnica, "
          "(2) responde exatamente o que foi perguntado, (3) números/unidades corretos, (4) concisão. "
          "Ignore o comprimento e a formatação. Explique em até 3 frases e termine com 'VEREDITO: A', "
          "'VEREDITO: B' ou 'VEREDITO: EMPATE'.")

def verdict(q, a, b):
    r = cli.chat.completions.create(model=JUDGE, temperature=0.0, max_tokens=300, messages=[
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"PERGUNTA:\n{q}\n\nRESPOSTA A:\n{a}\n\nRESPOSTA B:\n{b}"}])
    m = re.search(r"VEREDITO:\s*(A|B|EMPATE)", r.choices[0].message.content or "")
    return m.group(1) if m else "EMPATE"

def compare(q, a, b):            # +1 a vence, -1 b vence, 0 empate/inconsistente
    v1, v2 = verdict(q, a, b), verdict(q, b, a)
    if v1 == "A" and v2 == "B": return 1
    if v1 == "B" and v2 == "A": return -1
    return 0

NT = {"enable_thinking": False}   # mesmo modo no treino, na avaliação e no servidor (§5.4)
with open("data/teacher.jsonl") as f, open("data/pref.jsonl", "w") as wp, open("data/kto.jsonl", "w") as wk, \
     open("data/sft.jsonl", "w") as ws:
    for line in f:
        row = json.loads(line); q, ans = row["question"], row["answers"]
        score = [0] * len(ans)
        for i, j in itertools.combinations(range(len(ans)), 2):
            s = compare(q, ans[i], ans[j]); score[i] += s; score[j] -= s
        best, worst = max(range(len(ans)), key=score.__getitem__), min(range(len(ans)), key=score.__getitem__)
        if score[best] >= 1:                         # SFT: a melhor resposta, se venceu ao menos no saldo
            ws.write(json.dumps({"term": row["term"], "messages": [{"role": "user", "content": q},
                                 {"role": "assistant", "content": ans[best]}], "chat_template_kwargs": NT},
                                ensure_ascii=False) + "\n")
        if score[best] - score[worst] >= 2:          # margem mínima: descarta pares ambíguos
            p = [{"role": "user", "content": q}]
            wp.write(json.dumps({"prompt": p, "chosen": [{"role": "assistant", "content": ans[best]}],
                                 "rejected": [{"role": "assistant", "content": ans[worst]}],
                                 "chat_template_kwargs": NT, "term": row["term"]}, ensure_ascii=False) + "\n")
            for k, lab in ((best, True), (worst, False)):
                wk.write(json.dumps({"prompt": p, "completion": [{"role": "assistant", "content": ans[k]}],
                                     "label": lab, "chat_template_kwargs": NT, "term": row["term"]}, ensure_ascii=False) + "\n")
```

O SFT usa a **melhor** resposta de cada pergunta (`sft.jsonl`); o DPO usa (melhor, pior); o KTO usa rótulos binários. A coluna `term` serve para o split por termo do Passo 4; **remova-a depois do split** (ex.: `jq -c 'del(.term)'`) para não depender de como cada trainer trata colunas extras. Custos e tempos dependem da velocidade das receitas do Módulo 1 (ex.: GLM ~36 tok/s em prosa com 1 stream, ~75 tok/s agregados com 4 [fonte: `glm-mods.md` §4.2]) — **[estimativa]**: 3.000 perguntas × 4 respostas × ~400 tokens ≈ 4,8 M tokens de saída; a ~100–150 tok/s agregados com concorrência alta, **9–13 h** de professor; o juiz gera pouco (≈ 6 comparações × 2 ordens × ~150 tokens por pergunta ≈ 5,4 M tokens, mas com prompts longos — prefill domina). Planeje uma noite para o professor e outra para o juiz, ou reduza para 1.500 perguntas.

**Passo 4 — limpeza e descontaminação** (Apêndice B.6): normalize, deduplique por similaridade (MinHash ou n-gramas), remova itens que compartilhem n-gramas longos com **qualquer** conjunto de avaliação (domínio + IFEval + GSM8K + AIME + HumanEval…), descarte respostas com "não sei" quando a pergunta tem resposta no glossário, e separe **por termo** (não por linha) em `train`/`valid`/`test_dominio` — se o mesmo termo aparecer em treino e teste, o teste mede memorização.

**Tamanhos sugeridos [recomendação]:** SFT 3.000–6.000 exemplos; DPO 1.500–3.000 pares; KTO 3.000–6.000 rótulos; teste de domínio **300–500 perguntas** de termos nunca vistos (dá IC de ±4–6 pp no win rate, §1.3).

### 3.3 Avaliação de domínio (F0 — congele antes de treinar)

1. `test_dominio.jsonl` (300–500 perguntas, termos fora do treino) + **gabarito do professor** (melhor resposta, revisada à mão em ~50 itens).
2. Métricas: (i) win rate do candidato vs **base** e vs **SFT** com juiz em ambas as ordens (§1.3), WR e WR-LC (Apêndice B.2; scripts de geração/julgamento no Apêndice B.1); (ii) "acerto numérico": para perguntas com número no gabarito, extraia números da resposta e compare com tolerância relativa de 1% (regra simples e verificável); (iii) comprimento médio (tokens).
3. Bateria geral (§1.15) + segurança (XSTest; subconjunto StrongREJECT) + KL genérico.
4. Grave **todas as amostras** (`--log_samples` no lm-eval; JSONL próprio no juiz) — você vai querer relê-las.

---

## 4. Trilha A — Mac M4 24 GB com MLX

### 4.0 Ambiente e limites

```bash
# Python do sistema é 3.9.6 e o mlx exige ≥3.10 (verificado em mlx-copilot.md); use uv
uv venv --python 3.12 ~/.venvs/mlxft && source ~/.venvs/mlxft/bin/activate
uv pip install "mlx-lm[train]==0.32.0" "mlx-lm-lora==3.1.3" "lm-eval[api]==0.4.13" openai datasets
mlx_lm.lora --help | head -5          # confere a instalação
```

- **Memória utilizável:** neste Mac o MLX reporta **19,07 GB** de working set recomendado (~74% de 24 GB) e buffer máximo de 14,3 GB [verificado em `mlx-copilot.md` Q11]. `sudo sysctl iogpu.wired_limit_mb=N` aumenta o teto até o reboot. Feche navegador/IDE pesados durante o treino.
- **Velocidade de referência:** um matmul BF16 em MLX num M4 (MacBook Air sem ventoinha) mediu **2,78 TFLOPS**, contra 8,52 TFLOPS FP16 teóricos [não verificado: blog dev.to/tinyalg; spec cpu-monkey]. Decode de um 4B 4-bit neste Mac: ~33 tok/s (Qwen3.5-4B, `mlx-copilot.md` Q14) [verificado na pesquisa anterior].
- **Modelo:** `Qwen/Qwen3-4B-Instruct-2507` — 4.022.468.096 parâmetros BF16, 36 camadas, hidden 2560, 32 cabeças Q / 8 KV, `head_dim` 128, FFN 9728, vocabulário 151.936, embeddings amarrados (`tie_word_embeddings: true`), **sem modo de pensamento** (o template não tem `<think>`) [verificado: config e `tokenizer_config.json`]. Versões MLX: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (arquivo único de **2,26 GB**) [verificado], `-8bit` e `-bf16` [verificado: existem]. Alternativas: `mlx-community/Qwen3-1.7B-4bit`/`Qwen3-0.6B-4bit` (para GRPO rápido) e `mlx-community/Qwen3.5-4B-MLX-4bit` (híbrido GDN, pensa por padrão; o `mlx-lm-lora` detecta modelos lineares/híbridos e usa blocos lineares com checkpoint [fonte: README]).

**Dois frameworks, duas escalas de LoRA — não copie LR entre eles sem converter.** No MLX a saída é `y + scale·(x·A·B)` com **`scale` aplicado direto** (padrão **20,0** no `mlx_lm.lora`, **10,0** no `mlx-lm-lora`), `A ~ U(±1/√in)`, `B = 0`, nas últimas `num_layers` camadas e em **todas** as lineares do bloco por padrão [verificado: `mlx_lm/tuner/lora.py`, `tuner/utils.py`, `CONFIG_DEFAULTS` dos dois pacotes]. No PEFT a escala é `lora_alpha/r` (1,0 no padrão `r=8, alpha=8`). Com Adam, o passo efetivo do produto escala aproximadamente com **scale × LR** **[inferência]**: o padrão do `mlx_lm.lora` (LR 1e-5 × scale 20) ≈ PEFT com LR 2e-4 e alpha/r = 1; o padrão do `mlx-lm-lora` (LR 1e-5 × scale 10) ≈ 1e-4 — alto para DPO (o TRL sugere ~1e-5 para adapters em DPO [fonte: doc DPO]). Daí as LRs abaixo.

| Padrão | `mlx_lm.lora` 0.32.0 | `mlx_lm_lora.train` 3.1.3 |
|---|---|---|
| `num_layers` | 16 | **-1 (todas)** |
| `batch_size` | 4 | 4 |
| `iters` | 1000 | None (use `--iters` ou `--epochs`) |
| `learning_rate` | 1e-5 | 1e-5 |
| `max_seq_length` | 2048 | 2048 |
| `lora_parameters` | rank 8, scale **20.0**, dropout 0 | rank 8, scale **10.0**, dropout 0 |
| `grad_checkpoint` | False | False |
| `fuse` ao final | não (use `mlx_lm.fuse`) | **True** — funde e salva em `--adapter-path`, dequantizando, a menos que você tenha usado `--load-in-*` |
| SFT loss | NLL | `nll` (padrão), `chunked_nll`, `dft` |

[verificado: `CONFIG_DEFAULTS` em `mlx_lm/lora.py` e `mlx_lm_lora/train.py`]

### 4.1 Linha de base

```bash
# terminal 1 — servidor com o modelo base (porta 8080, bind 127.0.0.1)
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080

# terminal 2 — bateria geral (generativa) via API; mande o MESMO id de modelo carregado:
# um id diferente faz o mlx_lm.server trocar de modelo e até baixar do Hub (mlx-copilot.md §0.4)
lm_eval --model local-chat-completions \
  --model_args model=mlx-community/Qwen3-4B-Instruct-2507-4bit,base_url=http://127.0.0.1:8080/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
  --apply_chat_template --tasks ifeval,gsm8k_cot_zeroshot --limit 250 \
  --output_path results/mac/base --log_samples

# alternativa sem servidor (wrapper MLX do lm-eval)
mlx_lm.evaluate --model mlx-community/Qwen3-4B-Instruct-2507-4bit --tasks ifeval --apply-chat-template --limit 250 --output-dir results/mac/base_mlx

# domínio: gere respostas do base para test_dominio.jsonl (mesmo endpoint) e guarde para o juiz (§3.3)
```

`--limit 250` mantém a bateria em tempo de Mac; para relatório final, rode completo. Mantenha `num_concurrent=1` nos `--model_args`: o servidor MLX processa as requisições essencialmente em sequência.

### 4.2 SFT com QLoRA (`mlx_lm.lora`)

```bash
mkdir -p data/mlx_sft && cp data/sft_train.jsonl data/mlx_sft/train.jsonl && cp data/sft_valid.jsonl data/mlx_sft/valid.jsonl
cat > cfg/mac_sft.yaml <<'YAML'
model: mlx-community/Qwen3-4B-Instruct-2507-4bit   # quantizado => QLoRA automaticamente
train: true
data: data/mlx_sft                                 # train.jsonl / valid.jsonl, formato "chat"
fine_tune_type: lora
num_layers: 16
batch_size: 4
grad_accumulation_steps: 2                         # lote efetivo 8
iters: 600
learning_rate: 1.0e-5                              # com scale 20 (ver §4.0)
lr_schedule:
  name: cosine_decay
  warmup: 30
  arguments: [1.0e-5, 600, 1.0e-6]
max_seq_length: 1024
mask_prompt: true                                  # perda só na última mensagem (assistente)
grad_checkpoint: true
lora_parameters:
  rank: 16
  scale: 20.0
  dropout: 0.0
steps_per_report: 10
steps_per_eval: 100
val_batches: 25
save_every: 100
adapter_path: adapters/mac_sft
seed: 0
YAML
mlx_lm.lora --config cfg/mac_sft.yaml
```

O bloco `lr_schedule` (`name: cosine_decay`, `warmup`, `warmup_init` opcional, `arguments: [lr_inicial, passos_de_decaimento, lr_final]`) e o bloco `lora_parameters` **só podem ser dados no YAML** no `mlx_lm.lora` — não há flag de CLI para eles [verificado: `examples/lora_config.yaml` no main]. Esse exemplo oficial restringe `keys` a `self_attn.q_proj`/`self_attn.v_proj`; omitir `keys` (como acima) aplica LoRA a **todas** as lineares das últimas `num_layers` camadas, o que o "LoRA Without Regret" recomenda (§1.15).

**Leitura da saída:** o `mlx_lm.lora` imprime a cada `steps_per_report` a loss, `It/sec`, `Tokens/sec` e `Peak mem` — **rode 20 iterações primeiro** e extrapole tempo e memória antes de deixar 600 rodando.

**Orçamento [estimativa]** (conta no Apêndice A.7): pesos 4-bit 2,26 GB; LoRA r=16 nas 16 últimas camadas ≈ 14,7 M parâmetros → ≈ 0,24 GB com gradientes e Adam em fp32; ativações com checkpoint ≈ 0,8 GB de entradas de camada + ≈ 0,5 GB de pico de recomputação; **logits** `4 × 1024 × 151.936 × 4 B ≈ 2,5 GB` (+ gradiente) — o maior termo. Total **≈ 7–10 GB** em `batch 4 × 1024`, dentro dos 19 GB. Se estourar: `batch_size 2` + `grad_accumulation_steps 4`, ou `max_seq_length 768`; no `mlx-lm-lora`, `--sft-loss-type chunked_nll` evita materializar os logits inteiros [fonte: README].

**Tempo [estimativa]:** FLOPs/token ≈ forward completo (2·N ≈ 8,0 GFLOP) + backward de ativações nas 16 camadas treinadas e no `lm_head` (≈ 4,0 GFLOP) + recomputação dessas 16 camadas (≈ 3,2 GFLOP) ≈ **15 GFLOP/token**. A 1–2 TFLOPS efetivos (36–72% do matmul medido) → **65–130 tok/s**. 600 iterações × 4 × ~500 tokens médios ≈ 1,2 M tokens → **2,5–5 h**. Meça e ajuste `iters`.

**Avaliação rápida do adapter (sem fundir):**
```bash
mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path adapters/mac_sft --data data/mlx_sft --test   # perplexidade no test.jsonl
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path adapters/mac_sft --port 8080            # e rode a §4.1 de novo
```

### 4.3 DPO no Mac (`mlx_lm_lora.train`)

O `mlx-lm` puro **não** faz DPO; o `mlx-lm-lora` 3.1.3 tem `--train-mode` ∈ {`sft`, `dpo`, `ftpo`, `cpo`, `orpo`, `grpo`, `klpo`, `online_dpo`, `xpo`, `rlhf_reinforce`, `ppo`} e `--dpo-cpo-loss-type` ∈ {`sigmoid`, `hinge`, `ipo`, `dpop`} — **não há KTO nem SimPO** [verificado: `choices` em `mlx_lm_lora/train.py`]. (O README escreve `rlhf-reinforce`; o código aceita `rlhf_reinforce`.) O modelo de referência é **uma segunda cópia completa** carregada de `--reference-model-path` (ou do próprio `--model`), congelada [verificado: `load_reference_model`].

**Por que fundir o SFT antes do DPO:** a referência do DPO deve ser o modelo **pós-SFT**. Em vez de empilhar adapters, funda o SFT num checkpoint próprio — e em 8 bits, para que o delta do LoRA não suma no arredondamento (§2.4):

```bash
# 1) funde o SFT sobre o base BF16 (adapter treinado em 4-bit aplicado ao BF16: pequeno descasamento, aceitável)
mlx_lm.fuse --model mlx-community/Qwen3-4B-Instruct-2507-bf16 --adapter-path adapters/mac_sft --save-path models/sft_bf16 --dequantize
# 2) quantiza para 8 bits (≈4,3 GB; duas cópias cabem: política + referência)
mlx_lm.convert --hf-path models/sft_bf16 --mlx-path models/sft_8bit -q --q-bits 8 --q-group-size 64

# 3) DPO (QLoRA sobre o 8-bit; referência = o mesmo 8-bit congelado)
mkdir -p data/mlx_dpo && cp data/pref_mlx_train.jsonl data/mlx_dpo/train.jsonl && cp data/pref_mlx_valid.jsonl data/mlx_dpo/valid.jsonl
mlx_lm_lora.train --model models/sft_8bit --train --train-mode dpo \
  --data data/mlx_dpo --beta 0.1 --dpo-cpo-loss-type sigmoid \
  --reference-model-path models/sft_8bit \
  --num-layers 16 --batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-6 --lora-parameters '{"rank": 16, "dropout": 0.0, "scale": 10.0}' \
  --max-seq-length 1024 --grad-checkpoint --epochs 1 \
  --steps-per-eval 50 --save-every 100 --adapter-path adapters/mac_dpo
```

- `pref_mlx_*.jsonl` = formato (e) da §3.1 (texto puro com `prompt`/`chosen`/`rejected`, opcional `system`).
- LR 1e-6 × scale 10 ≈ 1e-5 em escala PEFT (§4.0) **[inferência]**. Variantes: `--dpo-cpo-loss-type ipo` (IPO), `--train-mode orpo` (sem referência, metade da memória), `--train-mode cpo`.
- Ao final o `mlx-lm-lora` **funde por padrão** e salva o modelo fundido no `--adapter-path` (dequantizado, porque o modelo veio quantizado do disco e não de `--load-in-*`) [verificado: bloco `if args.fuse and args.train`]. Para manter só o adapter, ponha `fuse: false` num YAML (`-c`).
- **Memória [estimativa]:** 2 × 4,3 GB + LoRA + ativações de 2 sequências (chosen e rejected) por par com checkpoint + logits (2 × 1024 × 151.936 × 4 B ≈ 1,2 GB) ≈ **11–14 GB**.
- **Tempo [estimativa]:** por par ≈ 2 sequências × (15 GFLOP/token de treino + 8 GFLOP/token de forward da referência) × ~700 tokens ≈ 32 TFLOP → a 1–2 TFLOPS, **16–32 s por par**; 1.000 pares ≈ **4,5–9 h**. Comece com 300–500 pares.
- **KTO no Mac:** não disponível no `mlx-lm-lora` 3.1.3. Opções: rodar o `KTOTrainer` do TRL no backend MPS do PyTorch (funciona em teoria para modelos pequenos, **[não verificado]**), ou converter os rótulos binários em pares (melhor positiva × pior negativa por prompt) e usar DPO.

### 4.4 GRPO pequeno no Mac (recompensa verificável)

```bash
mlx_lm_lora.train --list-reward-functions        # funções embutidas
mlx_lm_lora.train --model mlx-community/Qwen3-1.7B-4bit --train --train-mode grpo \
  --data mlx-community/gsm8k --group-size 4 --max-completion-length 512 --temperature 0.8 \
  --epsilon 1e-4 --grpo-loss-type dr_grpo --beta 0.0 \
  --reward-functions "accuracy_reward,format_reward" --reward-weights "[0.8, 0.2]" \
  --batch-size 2 --iters 200 --learning-rate 1e-6 --num-layers -1 \
  --lora-parameters '{"rank": 8, "dropout": 0.0, "scale": 10.0}' --adapter-path adapters/mac_grpo
```

- Padrões do `mlx-lm-lora` para GRPO: `group_size` 4, `epsilon` **1e-4** (largura do clip — bem menor que o 0,2 clássico), `max_completion_length` 512, `temperature` 0,8, `grpo_loss_type` `grpo` (opções `bnpo`, `dr_grpo`), `importance_sampling_level` `token` (`sequence` = GSPO) [verificado: `CONFIG_DEFAULTS`/`choices`]. Com `--beta 0` a referência não é avaliada e o KL logado é zero [fonte: README].
- Recompensa própria: arquivo com `@register_reward_function()` e assinatura `(prompts, completions, answer, types=None) -> list[float]` + `--reward-functions-file meu.py --reward-functions minha_funcao` [fonte: README]. Exemplo de domínio (contas de memória com resposta exata) no Apêndice B.7.
- **Tempo [estimativa]:** por passo, 2 prompts × 4 amostras × ~300 tokens = 2.400 tokens gerados; o decode do 1,7B 4-bit é mais rápido que os ~33 tok/s do 4B, e o lote de 8 sequências amortiza — digamos 100–200 tok/s agregados → 12–24 s; o treino desses tokens no 1,7B (≈ 6 × 1,7 GFLOP/token, todas as camadas) a 1–2 TFLOPS ≈ 12–25 s → **~0,5–1 min/passo**, 200 passos ≈ **2–3,5 h**. Sinal útil exige que o modelo acerte **às vezes** (veja `frac_reward_zero_std`/estatísticas por grupo): o Qwen3-1.7B em GSM8K está nessa faixa.

### 4.5 Avaliação e KL no Mac

1. Repita a §4.1 com o modelo final (servido com `--adapter-path` ou fundido).
2. Juiz de domínio: os juízes são os modelos do Módulo 1 nos Sparks (§3.2) — o Mac só gera as respostas.
3. **KL(base‖ajustado) em MLX** — script mínimo (compare modelos na mesma precisão, ex. dois 8-bit, para não misturar erro de quantização com deriva):

```python
# kl_mlx.py — KL(P||Q) médio por token, teacher-forced, em textos de avaliação
import json, sys, mlx.core as mx
from mlx_lm import load
P, tok = load(sys.argv[1])          # referência  (ex.: mlx-community/Qwen3-4B-Instruct-2507-8bit)
Q, _ = load(sys.argv[2])            # candidato   (ex.: models/final_8bit)
tot, n, agree = 0.0, 0, 0
for line in open(sys.argv[3]):      # JSONL com {"text": ...} — use o template de chat já aplicado
    ids = mx.array(tok.encode(json.loads(line)["text"])[:1024])[None]
    lp = P(ids).astype(mx.float32); lq = Q(ids).astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
    kl = (mx.exp(lp) * (lp - lq)).sum(axis=-1)
    agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
    tot += kl.sum().item(); n += kl.size
print(f"KL medio {tot/n:.5f} nats/token | top-1 igual {agree/n:.4f} | posicoes {n}")
```
Memória [estimativa]: 2 × 4,3 GB (8-bit) + logits fp32 de 1.024 × 151.936 (~0,6 GB) por modelo. Rode também P contra P quantizado de outra forma (ou P contra P) para o piso de ruído.

### 4.6 Fundir, re-quantizar e servir no Mac

```bash
# funde o adapter final sobre o BF16 do ponto de partida da etapa (aqui: o SFT fundido)
mlx_lm.fuse --model models/sft_bf16 --adapter-path adapters/mac_dpo --save-path models/final_bf16 --dequantize
# quantização calibrada com os SEUS dados (DWQ destila do BF16 para 4 bits)
mlx_lm.dwq --model models/final_bf16 --bits 4 --group-size 64 --data-path data/mlx_sft \
  --num-samples 512 --max-seq-length 1025 --batch-size 2 --grad-checkpoint --mlx-path models/final_4bit_dwq
# alternativa rápida (RTN): mlx_lm.convert --hf-path models/final_bf16 --mlx-path models/final_4bit -q --q-bits 4
python kl_mlx.py models/final_bf16 models/final_4bit_dwq data/kl_texts.jsonl      # erro de quantização vs o merge
mlx_lm.server --model models/final_4bit_dwq --port 8080                              # copiloto do Latent Atlas: preset MLX
```

- `--data-path` do DWQ espera um dataset no formato do mlx-lm (local ou do Hub; padrão `allenai/tulu-3-sft-mixture`) [verificado: `quant/dwq.py`]; misture com dados gerais (§2.5). Que um diretório local com `train.jsonl` funcione como no `mlx_lm.lora` é **[inferência]**.
- Se você deixou o `fuse` padrão do `mlx-lm-lora` ligado na §4.3, o diretório `adapters/mac_dpo` já contém um modelo fundido (dequantizado a partir do 8-bit); use-o como `models/final_bf16` e pule o primeiro comando. O formato do adapter que fica ao lado ser lido pelo `mlx_lm.fuse` é **[não verificado]** — por isso a recomendação de `fuse: false` + `mlx_lm.fuse` explícito.
- **Memória do DWQ [estimativa]:** professor BF16 (8 GB) + aluno 4-bit (2,3 GB) + ativações; o README sugere destilar a partir de um 8-bit para economizar [fonte: `LEARNED_QUANTS.md`].
- No copiloto da plataforma, deixe o campo `model` vazio no preset MLX (o servidor usa o `--model` carregado) [`mlx-copilot.md` §0.4].

### 4.7 Checklist da trilha A

- [ ] Python 3.12 via `uv`; `mlx-lm` 0.32.0 e `mlx-lm-lora` 3.1.3 fixados.
- [ ] Linha de base registrada (IFEval, GSM8K, domínio, XSTest) com o **mesmo** id de modelo no servidor.
- [ ] Dados descontaminados e separados por termo; `valid.jsonl` presente.
- [ ] 20 iterações de teste: `Peak mem` < ~16 GB, `Tokens/sec` medido, tempo extrapolado.
- [ ] LR convertida entre escalas (§4.0) — nada de copiar LR do PEFT.
- [ ] SFT: loss de validação cai; perplexidade de teste melhora; IFEval não cai além do IC.
- [ ] Merge do SFT em BF16 → 8-bit antes do DPO; DPO com `logps/chosen` monitorado (deslocamento de verossimilhança).
- [ ] KL vs base (8-bit × 8-bit) em texto genérico e de domínio + piso de ruído.
- [ ] Fundido final em BF16 → DWQ 4-bit com dados próprios; KL vs BF16 fundido.
- [ ] Servido no `mlx_lm.server`; smoke test no copiloto.

---

## 5. Trilha B — 2× DGX Spark (GB10): Qwen3.8-27B (SFT → DPO → merge → NVFP4 → vLLM) e GRPO em Qwen3.5-9B

### 5.0 Por que esses modelos e esse arranjo

- **`Qwen/Qwen3.8-27B`** [verificado: config/API do Hub]: 27.781.427.952 parâmetros BF16 (**55,56 GB** em safetensors), `Qwen3_5ForConditionalGeneration` (VLM; torre de visão de 27 blocos), texto `qwen3_5_text`: 64 camadas = **48 Gated DeltaNet + 16 atenção completa** (`full_attention_interval: 4`), hidden 5120, 24 cabeças Q / 4 KV com `head_dim` 256 e **saída gated** (`attn_output_gate: true` → `q_proj` sai com 24·256·2 = 12.288), GDN com 16 cabeças K e 48 cabeças V de 128, FFN 17.408, vocabulário 248.320, 262.144 de contexto nativo, **1 camada MTP** (`mtp_num_hidden_layers: 1`), `eos_token` `<|im_end|>`, `pad_token` `<|endoftext|>`; licença Apache-2.0. Pensa por padrão; o template tem `reasoning_effort` (`xhigh` padrão, `medium`, `low`) e `preserve_thinking` [verificado: `chat_template.jinja`].
- É a mesma família do **Qwen3.8-Flash-Next** do Módulo 1 e a ferramenta de quantização já tem receita oficial para ele (llm-compressor e ModelOpt), além de checkpoints de referência para comparar o seu (`nvidia/Qwen3.8-27B-NVFP4` 21,92 GB; `RedHatAI/Qwen3.8-27B-NVFP4` 24,69 GB; `RedHatAI/Qwen3.8-27B-INT4` 19,45 GB; `Qwen/Qwen3.8-27B-FP8` 30,87 GB) [verificado: tamanhos via API].
- **Arranjo para SFT/DPO: DDP simétrico** — cada Spark guarda o modelo BF16 inteiro (55,6 GB cabe em ~121,7 GiB) e processa metade dos dados; só os gradientes do LoRA (217,6 M parâmetros em r=32 → 0,87 GB em fp32) atravessam o RoCE por passo de otimizador. FSDP faria all-gather dos 55,6 GB de pesos a cada forward/backward/recomputação (≈ 3 × 55,6 GB ≈ 167 GB por passo pelo cabo) — desnecessário quando o modelo cabe **[estimativa]**.
- **Arranjo para GRPO: papéis assimétricos** — Spark B roda o `vllm serve` (gerador), Spark A roda o `GRPOTrainer` (aprendiz) e empurra os pesos atualizados pelo NCCL a cada passo. É o modo "server" do TRL; o "colocate" (padrão) dividiria a mesma memória unificada entre treino e vLLM.
- **GRPO no `Qwen/Qwen3.5-9B`** (9.653.104.368 parâmetros, mesma `model_type: qwen3_5`: 32 camadas = 24 GDN + 8 atenção, hidden 4096, FFN 12.288 [verificado: config]) porque RL é dominado por **geração**, e gerar com um 27B BF16 num GB10 é lento demais para um laboratório (§5.7).

### 5.1 Ambiente (os dois nós)

```bash
# ---- em CADA Spark: rede do Módulo 1 (nomes variam por kit: confira `ip -br a` e `ls /sys/class/infiniband`)
# Spark A (rank 0)                         # Spark B (rank 1)
export NCCL_SOCKET_IFNAME=enp1s0f1np1      # enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f1np1      # enp1s0f0np0
export NCCL_IB_HCA=rocep1s0f1              # rocep1s0f0
export NCCL_IB_GID_INDEX=3                 # confira a tabela GID RoCEv2 (::ffff:<ip>) de cada nó
export NCCL_DEBUG=INFO                     # na 1ª vez: confirme "NET/IB" e não "NET/Socket"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'    # UMA: libere page cache antes de carregar 55 GB

# ---- contêiner de TREINO (SFT/DPO/KTO/merge/llm-compressor): NGC PyTorch 26.08 (PyTorch 2.14.0a0, CUDA 13.4.1)
docker run -d --name ft --gpus all --network host --ipc host --shm-size 32g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_SOCKET_IFNAME -e GLOO_SOCKET_IFNAME -e NCCL_IB_HCA -e NCCL_IB_GID_INDEX -e NCCL_DEBUG \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -v /data/work:/work -w /work \
  nvcr.io/nvidia/pytorch:26.08-py3 sleep infinity
docker exec -it ft bash

# dentro do contêiner — NÃO deixe o pip trocar o torch da NGC
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"     # espere (12, 1)
python -c "import torch; print(f'torch=={torch.__version__}')" > /work/constraints.txt
export PIP_CONSTRAINT=/work/constraints.txt
pip install "trl==1.14.1" "peft==0.21.2" "transformers==5.18.0" "accelerate==1.15.0" datasets \
            "bitsandbytes==0.50.2" "liger-kernel==0.8.4" math-verify hf_transfer openai "lm-eval[api]==0.4.13"
# caminho rápido do Gated DeltaNet — sem ele o GDN cai em PyTorch puro (várias vezes mais lento)
pip install "flash-linear-attention==0.5.2"
TORCH_CUDA_ARCH_LIST=12.1 pip install --no-build-isolation "causal-conv1d==1.7.0"
python -c "import fla, causal_conv1d; print('GDN fast path OK')"
```

Notas:
- A combinação exata acima **não foi testada** no sm_121 nesta pesquisa. A referência pública que funcionou no GB10 usou NGC `pytorch:26.03-py3`, transformers **5.5.0** (pino do Unsloth), `trl==0.24.0`, `peft 0.19.1`, `flash-linear-attention 0.5.0`, `causal-conv1d 1.4.0`, Triton 3.6.0 compilado com `TORCH_CUDA_ARCH_LIST=12.1` [não verificado: repo albond/DGX_Spark_Unsloth_Lossless_Speedup, mai/2026]. Se a pilha nova falhar, volte a esses pinos (e use a API 0.x do TRL).
- O impacto do caminho rápido é grande: no mesmo repositório, só instalar `fla`+`causal-conv1d` levou o LoRA do Qwen3.5-2B de **505 → 3.076 tok/s**; o pacote completo de otimizações chegou a 3.461 tok/s (7,67× sobre o Unsloth puro) [não verificado: medição de terceiros]. O transformers 5.x carrega esses kernels via `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")` e `("causal_conv1d_fn", "causal_conv1d")`, com fallback PyTorch [verificado: `modeling_qwen3_5.py`] — por isso **meça** (§5.3) em vez de confiar que "importou".
- Contêiner de **geração/serviço**: `vllm/vllm-openai:v0.30.0` (a mesma imagem da lane v0.30 do Qwen no Módulo 1). Para GRPO, o **lado do treinador também precisa do pacote `vllm`** (o cliente de pesos do TRL importa `vllm.distributed.weight_transfer.*`) [verificado: `trl/generation/vllm_client.py`] — use a imagem do vLLM nos dois nós e instale `trl`/`peft` nela (§5.7).

**Configuração do Accelerate (DDP, 2 máquinas × 1 GPU)** — `cfg/accel_ddp_2nodes.yaml`, idêntico nos dois nós exceto `machine_rank`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 2
num_processes: 2           # total de processos (1 GPU por Spark)
machine_rank: 0            # 1 no Spark B
main_process_ip: 10.10.10.1   # IP da porta CX-7 do Spark A
main_process_port: 29500
rdzv_backend: static
same_network: true
gpu_ids: all
```
(Formato igual ao `config_fsdp_lora.yaml` do playbook oficial `pytorch-fine-tune`, que usa `num_machines: 2`, `num_processes: 2`, `rdzv_backend: static` — lá com `distributed_type: FSDP` [verificado: arquivo do playbook].)

### 5.2 Orçamento de memória por Spark [estimativa]

Conta detalhada no Apêndice A.7. "Usável" ≈ 121,7 GiB de `MemTotal` menos SO/serviços; o Módulo 1 trabalha com `0.88 × 121.69 = 107.09 GiB` como teto prático [fonte: `glm-mods.md` §4.1].

| Trabalho | Pesos | LoRA + grad + Adam (fp32) | Ativações (checkpoint) | Logits | Total estimado |
|---|---|---|---|---|---|
| SFT LoRA BF16, 27B, r=32, seq 4096, micro-lote 1 | 55,6 GB | 3,5 GB | ~2,7 GB (entradas das 64 camadas) + 1–2 GB de pico de recomputação | evitados: `loss_type="chunked_nll"` é o padrão do `SFTConfig` [verificado] | **≈ 65–72 GB** |
| DPO LoRA BF16, 27B, `max_length` 2048 (escolhida+rejeitada = 2 seqs) | 55,6 GB (referência = mesmo modelo com adapter desligado: **0 GB extra**) | 3,5 GB | ~2,7 GB + pico | 2 × 2048 × 248.320 × 4 B ≈ **4,1 GB** (+ gradiente) — o DPO materializa logits, salvo com `use_liger_kernel=True` | **≈ 70–80 GB** |
| QLoRA NF4, 27B (alternativa 1 nó) | ≈ 13,9 GB (4 bits) + ~0,4 GB de constantes + **5,1 GB** de `embed_tokens`+`lm_head` em BF16 (o bnb não quantiza embedding nem `lm_head`) ≈ 19–20 GB | 3,5 GB | igual | igual | **≈ 30–40 GB** — sobra para lote maior; mais lento (dequantização) e o adapter fica "acostumado" ao NF4 |
| GRPO 9B — Spark A (treino, `per_device_train_batch_size` 4, ~900 tokens/seq) | 19,3 GB | ~0,6–1,3 GB (r=16–32) | ~1–2 GB | 4 × 900 × 248.320 × 4 B ≈ 3,6 GB (o TRL ≥1.14 tem kernel Triton fundido de logprob+entropia em `trl.kernels` [fonte: release v1.14.0]) | **≈ 30–40 GB** |
| GRPO 9B — Spark B (vLLM, `--gpu-memory-utilization 0.55`) | 19,3 GB | — | — | — | ≈ 67 GiB reservados → ~45 GiB para KV + estados GDN; 128 seqs × ~1k tokens cabem com folga |

### 5.3 Pré-voo e linha de base

```bash
# 1) NCCL entre os nós (nccl-tests do playbook "nccl" ou all_reduce do torch); procure NET/IB e banda ~ dezenas de GB/s
# 2) velocidade real de treino (detecta fallback do GDN): 10 passos de SFT com seq 2048 e veja tokens/s no log
#    alvo [estimativa]: 27B ≈ 185–370 tok/s por Spark (30–60 TFLOPS efetivos, §5.4); se vier <100, o caminho rápido não está ativo
# 3) linha de base do 27B: sirva o modelo base num Spark (BF16, TP=1) e rode a bateria
docker run -d --name base --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:v0.30.0 \
  Qwen/Qwen3.8-27B --served-model-name q38-base --port 8000 --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.80 --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --language-model-only
```

- `--tool-call-parser qwen3_xml` é o que os cards do RedHatAI usam para o Qwen3.8-27B (o card do Qwen3.5-9B sugere `qwen3_coder`) [fonte: cards]. `--language-model-only` pula a torre de visão [fonte: card Qwen3.5-9B; flag existe no `arg_utils.py` v0.30.0, verificado].
- **Protocolo de avaliação reproduzível do RedHatAI para este modelo** (use igual para base e candidatos) [fonte: card `RedHatAI/Qwen3.8-27B-NVFP4`]: modo pensamento, `temperature=1.0, top_p=0.95, top_k=20`, `max_gen_toks=32000`, **3 sementes** (8 para AIME 2025) e média entre sementes:
  ```bash
  lm_eval --model local-chat-completions --tasks ifeval \
    --model_args "model=q38-base,max_length=69632,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
    --num_fewshot 0 --apply_chat_template --output_path results/base_ifeval_s1234.json --seed 1234 \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=32000,seed=1234"
  # idem com --tasks gsm8k_platinum_cot_llama ; AIME/MATH-500/GPQA via lighteval + litellm (provider hosted_vllm), ex.:
  # lighteval endpoint litellm litellm_config.yaml 'aime25@1@8|0' --output-dir results/ --save-details
  ```
  Referência publicada (B200, BF16, modo pensamento): GSM8K Platinum 96,25%, MATH-500 83,67%, AIME 2025 96,67%, GPQA Diamond 89,56%, IFEval 91,19%, SWE-bench 78,8% [fonte: card]. **No GB10 com `max_gen_toks=32000`, AIME com 8 sementes é caro** [estimativa: 30 × 8 × até 32k tokens ≈ até 7,7 M tokens gerados; a ~90–130 tok/s agregados (faixa medida para NVFP4 com 16 concorrentes no fórum NVIDIA [não verificado]) seriam até ~16–24 h] — para o laboratório, use 2 sementes e `max_gen_toks` 16k, e **compare base e candidato com o mesmo orçamento**.
- **Modo de avaliação do domínio:** como os dados de SFT/DPO serão sem pensamento (§5.4), avalie o domínio com `{"chat_template_kwargs": {"enable_thinking": false}}` nas requisições — e a bateria geral nos **dois** modos (pensando e não pensando) para detectar se o SFT apagou o raciocínio.

### 5.4 SFT LoRA BF16 em DDP (TRL 1.14.1)

**Decisões de template (as que mais quebram fine-tunes de Qwen3.x):**
1. O `SFTTrainer` com `assistant_only_loss=True` **troca automaticamente** o template do Qwen3.8 pelo `qwen3_8_training.jinja`, que sempre emite o bloco de pensamento (prefix-preserving) e marca a saída do assistente com `{% generation %}…{% endgeneration %}` [verificado: `trl/chat_templates/README.md`].
2. Exemplos **sem** `reasoning_content` viram `<think>\n\n</think>\n\n{content}` — o modelo aprende a responder **sem pensar**; é coerente com servir em `enable_thinking=false` (o prompt de geração nesse modo já termina em `<think>\n\n</think>\n\n` [verificado: template]). Se quiser preservar o raciocínio, gere os dados do professor **com** raciocínio e grave-o em `reasoning_content` (§3.1 a′), ou misture uma fração de dados com pensamento.
3. Com `enable_thinking` indefinido/verdadeiro o template **injeta no system prompt** a instrução de esforço (`Reasoning effort is set to xhigh…`); com `false`, não [verificado: template linhas 46–56 e 76–86]. Treine e sirva com o **mesmo** `chat_template_kwargs` — por isso a coluna `chat_template_kwargs` em cada exemplo (§3.1) e `--default-chat-template-kwargs` no servidor (§5.12).
4. `eos_token` do tokenizer é `<|im_end|>` [verificado] — correto para chat; não mude.

```python
# train_sft.py — Qwen3.8-27B, LoRA BF16, DDP em 2 Sparks
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3.8-27B"
# só a parte de linguagem (exclui model.visual.* e o lm_head); in_proj_a/b do GDN ficam de fora (48 saídas)
TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))"

ds = load_dataset("json", data_files={"train": "data/sft_train.jsonl", "validation": "data/sft_valid.jsonl"})

peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")

args = SFTConfig(
    output_dir="ckpt/q38-sft",
    model_init_kwargs={"dtype": torch.bfloat16},   # sem isto o TRL carrega em FLOAT32 (padrão quando o modelo é string)
    max_length=4096,
    packing=False,                                 # ver nota sobre packing + GDN
    assistant_only_loss=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,                # lote efetivo = 1 × 16 × 2 nós = 32 sequências
    num_train_epochs=2,
    learning_rate=1e-4,                            # LoRA ≈ 10× o LR de full FT (LoRA Without Regret); dica do TRL: ~1e-4
    lr_scheduler_type="cosine",
    warmup_steps=0.03,                             # float < 1 = fração dos passos (assinatura `warmup_steps: float`) [inferência]
    gradient_checkpointing=True,                   # já é o padrão do SFTConfig
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    logging_steps=5,
    eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=200, save_total_limit=3,
    ddp_find_unused_parameters=False,
    dataset_num_proc=8,
    report_to="none",
    seed=42,
)
trainer = SFTTrainer(model=MODEL, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], peft_config=peft_config)
trainer.train()
trainer.save_model("adapters/q38-sft")             # salva só o adapter (PEFT)
```

```bash
# Spark A
accelerate launch --config_file cfg/accel_ddp_2nodes.yaml --machine_rank 0 train_sft.py
# Spark B (mesmo código, mesmos dados no mesmo caminho)
accelerate launch --config_file cfg/accel_ddp_2nodes.yaml --machine_rank 1 train_sft.py
```

- Padrões do `SFTConfig` relevantes [verificado: assinatura]: `learning_rate` 2e-5 (full FT), `max_length` 1024 (**aumente**), `gradient_checkpointing` True, `bf16` True se `fp16` não setado, `packing` False, `packing_strategy` `bfd` (com `bfd`, `padding_free` é forçado — exige FlashAttention 2/3 ou equivalente), `loss_type` `chunked_nll` (ou `nll` com Liger), `assistant_only_loss` False, `completion_only_loss` None (= só completion para datasets prompt–completion).
- **Packing com GDN:** o packing `bfd` concatena sequências e depende de reset de estado por fronteira de sequência; para camadas recorrentes (GDN) isso exige suporte a `cu_seqlens` no caminho do `fla` **[não verificado para Qwen3.5/3.8 no transformers 5.18]**. Comece com `packing=False`; teste packing comparando a loss de validação com e sem.
- `target_modules` como **string** é casado por regex completa contra o nome do módulo no PEFT — a regex acima casa `model.language_model.layers.N.…` e exclui a torre `model.visual.*` [verificado: nomes no `model.safetensors.index.json`]. Com `"all-linear"` você treinaria também a visão.
- LoRA total: **217,6 M** parâmetros (r=32) → adapter BF16 de ~435 MB (Apêndice A.7).
- Métricas a acompanhar: `loss`, `eval_loss`, `mean_token_accuracy`, `entropy`, `grad_norm`, `num_tokens` [fonte: doc SFT].

**Tempo [estimativa]:** 4.000 exemplos × ~900 tokens × 2 épocas = 7,2 M tokens; a 6·N FLOPs/token (forward + backward de ativações + recomputação; os gradientes de peso do LoRA são desprezíveis) = 1,62e11 FLOP/token; a 30–60 TFLOPS efetivos por Spark → 185–370 tok/s por nó, **370–740 tok/s** nos dois → **2,7–5,4 h**. A faixa de 60 TFLOPS bate com a medição pública de 370 tok/s para LoRA no Qwen3.5-27B num único GB10 [não verificado: albond]. **Não use** os números de "pico" da NVIDIA para fine-tuning no Spark (Llama 3.2 3B full FT 82.739 tok/s; Llama 3.1 8B LoRA 53.657 tok/s; Llama 3.3 70B QLoRA 5.079 tok/s, seq 2048 [fonte: blog NVIDIA, 2025-10-24]): 53.657 tok/s × 6 × 8e9 ≈ 2,6 PFLOP/s, uma ordem de grandeza acima do que o GB10 entrega em BF16 (~125 TFLOPS de pico denso, estimativa usada no Módulo 1) — a métrica deles não é tokens de treino por segundo no sentido usual **[inferência]**.

**Alternativa QLoRA num Spark só** (libera o outro para gerar dados ou servir): no TRL ≥1.8 o trainer aceita `quantization_config` direto [verificado: assinatura do `SFTTrainer`]:
```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
trainer = SFTTrainer(model=MODEL, args=args, train_dataset=..., peft_config=peft_config, quantization_config=bnb)
```
(mesmos parâmetros do script `Llama3_70B_qLoRA_finetuning.py` do playbook oficial do Spark [verificado]). Lembre: para o merge, carregue o base **BF16**, não o NF4 (§2.4).

### 5.5 Avaliação intermediária do SFT (rápida)

1. `eval_loss` no `sft_valid.jsonl` e curva sem sobreajuste.
2. Sirva base + adapter **sem merge** (caminho A da §2.1) para comparar sem gastar disco:
   ```bash
   docker run -d --name sft --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 \
     -v $HOME/.cache/huggingface:/root/.cache/huggingface -v /data/work:/work vllm/vllm-openai:v0.30.0 \
     Qwen/Qwen3.8-27B --served-model-name q38-base --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.80 \
     --enable-lora --max-loras 2 --max-lora-rank 32 --lora-modules q38-sft=/work/adapters/q38-sft \
     --reasoning-parser qwen3 --language-model-only --default-chat-template-kwargs '{"enable_thinking": false}'
   # requisições com "model": "q38-sft" (adapter) ou "q38-base" (base) no MESMO servidor
   ```
   `--default-chat-template-kwargs` (JSON) existe no frontend do vLLM v0.30.0 [verificado: `entrypoints/launchers/cli_args.py`]. O suporte a LoRA nas projeções do GDN vem do `packed_modules_mapping` (`in_proj_qkvz`) [verificado].
3. 100 perguntas de domínio + IFEval com `--limit 200` nos dois "modelos" do mesmo servidor → decisão de seguir para o DPO.

### 5.6 Preferências: DPO (e as variantes IPO, SimPO, LN-DPO, robusto, KTO)

**Primeiro funda o SFT** (o DPO precisa de π_ref = modelo pós-SFT; com PEFT, a referência é o próprio modelo com o adapter **desligado** — então o adapter do SFT precisa estar **dentro** dos pesos):
```bash
python merge.py --base Qwen/Qwen3.8-27B --adapter adapters/q38-sft --out models/q38-sft-bf16   # script na §5.9
```

```python
# train_dpo.py — DPO LoRA sobre o SFT fundido, DDP em 2 Sparks
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

BASE = "models/q38-sft-bf16"
TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))"
ds = load_dataset("json", data_files={"train": "data/pref_train.jsonl", "validation": "data/pref_valid.jsonl"})

args = DPOConfig(
    output_dir="ckpt/q38-dpo",
    model_init_kwargs={"dtype": torch.bfloat16},
    loss_type=["sigmoid"],          # DPO clássico (Bradley–Terry)
    beta=0.1,                       # padrão; ↑beta = fica mais perto da referência
    max_length=2048,                # padrão 1024; truncamento só "keep_start"
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=5e-6,             # dica do TRL p/ adapters em DPO: ~1e-5; comece abaixo
    lr_scheduler_type="cosine", warmup_steps=0.05,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5, eval_strategy="steps", eval_steps=50, save_steps=100, save_total_limit=3,
    ddp_find_unused_parameters=False, report_to="none", seed=42,
)
trainer = DPOTrainer(model=BASE, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                     peft_config=LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
trainer.train()
trainer.save_model("adapters/q38-dpo")
```

- Padrões/limites do `DPOConfig` [verificado: doc/assinatura]: `beta` 0,1; `loss_type` lista (padrão `["sigmoid"]`), com `loss_weights` para combinar perdas (ex.: MPO = `["sigmoid","bco_pair","sft"]` com `[0.8, 0.2, 1.0]`); `max_length` 1024; `learning_rate` 1e-6; `precompute_ref_log_probs` False; `padding_free` **temporariamente indisponível** desde o refactor (avisa e volta ao padding); `sync_ref_model` (TR-DPO) incompatível com PEFT sem `ref_model` separado; `use_weighting` (WPO); `ld_alpha` (LD-DPO); `f_divergence_type` (`reverse_kl`, `forward_kl`, `js_divergence`, `alpha_divergence`).
- Dataset: formato (d) da §3.1 com `chat_template_kwargs` igual ao do SFT.
- **Monitorar** (§1.16): `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`, `entropy`. Se `logps/chosen` cair continuamente, troque para `loss_type=["sigmoid","sft"]` (NLL nas preferidas ancora a verossimilhança) ou aumente `beta`.

**Variantes — o que trocar** (o resto do script igual):

| Método | Configuração no TRL 1.14 | Observações |
|---|---|---|
| **IPO** | `DPOConfig(loss_type=["ipo"], beta=τ)` | perda quadrática em torno de 1/(2τ); o paper argumenta que o logit do DPO "amplifies near-deterministic preferences and effectively weakens KL regularization" [fonte: TRL paper index / arXiv 2310.12036]. O nome no paper é **IPO = Identity Preference Optimisation** (ΨPO com Ψ = identidade) |
| **LN-DPO** (normalizado por comprimento; Tülu 3) | `loss_type=["sigmoid_norm"]` | divide os log-ratios pelo comprimento de cada resposta — ataca o viés de verbosidade [fonte: TRL paper index] |
| **SimPO** (sem referência) | `from trl.experimental.cpo import CPOConfig, CPOTrainer`; `CPOConfig(loss_type="simpo", cpo_alpha=0.0, beta=2.5, simpo_gamma=1.375, learning_rate=1e-6, num_train_epochs=1)` | configuração do paper (full FT); recompensa = log-prob **média** por token × β, margem γ; **sem modelo de referência** (menos memória e compute) [fonte: TRL paper index, CPO doc]. Com LoRA, LR ≈ 10× [inferência: LoRA Without Regret]. `CPOTrainer` é **experimental** (sem garantia de API) |
| **DPO robusto** (rótulos ruidosos do juiz LLM) | `loss_type=["robust"], label_smoothing=0.1` | `label_smoothing` = probabilidade de o rótulo estar invertido, em [0, 0,5) [fonte: doc DPO] — natural quando os pares vêm de um juiz LLM |
| **APO** | `loss_type=["apo_zero"]` (vencedoras melhores que o modelo) ou `["apo_down"]` | ancorado [fonte: doc DPO] |
| **KTO** (binário) | `from trl import KTOConfig, KTOTrainer`; `KTOConfig(beta=0.1, desirable_weight=1.0, undesirable_weight=…, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=5e-6, max_length=2048)` + dataset (f) | estável desde a v1.8.0 [verificado: release notes]. Exige `per_device_train_batch_size > 1` e `train_sampling_strategy="sequential"` (padrão do `KTOConfig`) porque o termo de KL é estimado pareando cada prompt com uma completion de **outro** exemplo do lote; a doc recomenda lote por passo ≥ 4 e efetivo 16–128, LR entre 5e-7 e 5e-6 (full FT; adapters toleram mais) e pesos tais que (desirable_weight × nº positivos)/(undesirable_weight × nº negativos) ∈ [1, 4/3] [fonte: doc KTO]. Memória: micro-lote 4 × 2048 multiplica ativações e logits por ~4 em relação ao DPO (§5.2) — use `max_length` 1536 se apertar |
| ORPO / CPO | `trl.experimental.orpo` / `trl.experimental.cpo` | sem referência; experimentais |
| PPO (RLHF clássico) | **removido do TRL na v1.13.0** [verificado: release notes] | use OpenRLHF (`--algo.advantage.estimator` padrão = PPO com crítico) ou verl |
| A*-PO (regressão de vantagem ótima; parente do AWR) | `trl.experimental.a2po.A2POTrainer` com `beta1=0.5, beta2=1e-3, num_value_samples=8`, recompensa binária | estima V* offline com amostras da referência e faz regressão quadrática on-policy com 1 geração por prompt; não supera o Pass@K da referência [fonte: TRL paper index] |

**Tempo [estimativa]:** 2.500 pares × 2 sequências × ~900 tokens = 4,5 M tokens-sequência; custo por token ≈ 6·N (política) + 2·N (forward da referência) = 8·N → equivalente a 6,0 M tokens de SFT → **2,3–4,5 h** nos dois Sparks. KTO com 5.000 linhas custa mais por linha (forward extra das completions desemparelhadas para o KL) — **[estimativa grosseira]** 3–7 h.

### 5.7 RL com recompensa verificável (GRPO) — Spark B gera, Spark A treina

**Spark B — gerador vLLM com transferência de pesos** (flags exigidas pelo TRL [fonte: doc TRL vLLM integration; flags conferidas no `arg_utils.py` v0.30.0]):
```bash
docker run -d --name gen --gpus all --network host --ipc host --shm-size 16g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
  -e VLLM_SERVER_DEV_MODE=1 -e NCCL_SOCKET_IFNAME -e GLOO_SOCKET_IFNAME -e NCCL_IB_HCA -e NCCL_IB_GID_INDEX \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:v0.30.0 \
  Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --max-model-len 4096 --gpu-memory-utilization 0.55 --enable-prefix-caching \
  --weight-transfer-config '{"backend": "nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1
```
- `VLLM_SERVER_DEV_MODE=1` expõe os endpoints de atualização de pesos (`/start_weight_update`, `/update_weights`, `/finish_weight_update`) **e outros endpoints de desenvolvimento** — mantenha na rede privada do cabo CX-7 [fonte: doc TRL].
- `--logprobs-mode processed_logprobs` devolve logprobs depois de temperatura/processadores — o que a correção de importance sampling do TRL espera [fonte: doc TRL].
- Não use `--language-model-only` aqui: o treinador sincroniza os pesos do modelo inteiro carregado pelo TRL (inclui a torre de visão) **[inferência; teste se quiser economizar a torre]**.
- O TRL só suporta vLLM de **0.20.1 a 0.30.0** [fonte: doc]. O `trl vllm-serve` está **deprecado** (só monta este mesmo comando) [fonte].

**Spark A — treinador** (mesma imagem do vLLM; `pip install "trl==1.14.1" "peft==0.21.2" math-verify` dentro dela):
```python
# train_grpo.py — GRPO (perda DAPO) com LoRA em Qwen3.5-9B, recompensa = acerto verificado em GSM8K
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward, get_soft_overlong_punishment

def to_rl(ex):
    sol = ex["answer"].split("####")[-1].strip().replace(",", "")
    return {"prompt": [{"role": "user", "content": ex["question"] + "\nResolva e escreva a resposta final em \\boxed{}."}],
            "solution": sol}
train = load_dataset("openai/gsm8k", "main", split="train").map(to_rl, remove_columns=["question", "answer"])

args = GRPOConfig(
    output_dir="ckpt/q35-9b-grpo",
    model_init_kwargs={"dtype": torch.bfloat16},
    use_vllm=True, vllm_mode="server", vllm_server_host="10.10.10.2", vllm_server_port=8000,   # IP CX-7 do Spark B
    chat_template_kwargs={"enable_thinking": False},
    num_generations=8,                      # G (padrão 8)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,         # steps_per_generation = 32 → 4×1×32 = 128 completions = 16 prompts × 8
    max_completion_length=768,              # padrão 512
    temperature=1.0, top_p=0.95, top_k=20,  # exploração do RL; o card sugere 0.7/0.8/20 p/ uso SEM pensamento [recomendação]
    loss_type="dapo",                       # padrão do TRL 1.14
    epsilon=0.2, epsilon_high=0.28,         # clip-higher do DAPO
    mask_truncated_completions=True,        # "overlong filtering" do DAPO
    beta=0.001,                             # padrão 0.0; >0 só para LOGAR o KL (referência = adapter desligado)
    learning_rate=1e-5,                     # padrão 1e-6 (full FT); LoRA ≈ 10× [LoRA Without Regret]
    lr_scheduler_type="constant_with_warmup", warmup_steps=5,
    max_steps=150, logging_steps=1, save_steps=25,
    log_completions=True, num_completions_to_print=2,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none", seed=42,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen3.5-9B",
    reward_funcs=[accuracy_reward, get_soft_overlong_punishment(max_completion_len=768, soft_punish_cache=128)],
    args=args, train_dataset=train,
    peft_config=LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                           target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))"),
)
trainer.train()
trainer.save_model("adapters/q35-9b-grpo")
```
```bash
accelerate launch --num_processes 1 train_grpo.py      # um processo no Spark A; o Spark B só gera
```

- **Padrões do `GRPOConfig` 1.14** [verificado: doc + `grpo_config.py`]: `num_generations` 8, `max_completion_length` 512, `temperature` 1,0, `top_p` 1,0, `top_k` 0, `beta` **0,0** (sem modelo de referência), `epsilon` 0,2, `epsilon_high` = `epsilon`, `num_iterations` 1, `scale_rewards` `"group"`, **`loss_type` `"dapo"`** (opções `grpo`, `dapo`, `dr_grpo`, `sapo`, `cispo`, `vespo`), `importance_sampling_level` `"token"` (`"sequence"` = GSPO), `mask_truncated_completions` False, `learning_rate` **1e-6**, `use_vllm` False, **`vllm_mode` `"colocate"`**, `vllm_gpu_memory_utilization` 0,3, `vllm_server_port` 8000, `vllm_group_port` 51216, `vllm_server_timeout` 240 s, `vllm_importance_sampling_correction` True (`sequence_mask`, teto 3,0), `multi_objective_aggregation` `"sum_then_normalize"`.
- **Funções de recompensa** recebem `prompts`, `completions`, `completion_ids`, `trainer_state`, colunas do dataset via `**kwargs`, e podem devolver `None` por amostra (multitarefa); podem ser `async` [fonte: doc]. Embutidas em `trl.rewards`: `accuracy_reward`, `reasoning_accuracy_reward`, `get_cosine_scaled_reward`, `think_format_reward`, `get_repetition_penalty_reward`, `get_soft_overlong_punishment` [verificado: `trl/rewards/__init__.py`].
- **Com PEFT**, a cada sincronização o TRL faz `merge_adapter()`, transmite os pesos fundidos (BF16 do 9B ≈ 19,3 GB) e `unmerge_adapter()` [verificado: `trl/generation/vllm_generation.py`]. A ~10–20 GB/s efetivos no RoCE de 200 Gb/s, ≈ 1–2 s por passo **[estimativa]**.
- **O que olhar** (§1.16): `reward`, `rewards/accuracy_reward/mean`, `frac_reward_zero_std` (se → 1, o problema é fácil/difícil demais para o modelo), `completions/mean_length`, `completions/clipped_ratio`, `entropy`, `clip_ratio/*`, `kl`, `sampling/sampling_logp_difference/mean`.
- **Tempo [estimativa]:** por passo, 128 completions × ~400 tokens ≈ 51 k tokens gerados; decode do 9B BF16 (19,3 GB por passo de decode a ~230 GB/s ≈ 12 passos/s no teto de banda; lote de 128 → teto ~1.500 tok/s; realista 600–1.200 tok/s) → **45–85 s**; treino de ~128 × 520 tokens ≈ 67 k tokens a 6·N = 5,8e10 FLOP/token e 30–60 TFLOPS → **65–130 s**; sincronização ~2 s. **≈ 2–3,6 min/passo → 150 passos ≈ 5–9 h.** Com `Qwen/Qwen3.5-4B`, ~2× mais rápido.
- **Variante de domínio (RLVR do próprio curso):** problemas gerados por código com resposta exata (bytes de KV por token, GiB de pesos por formato, tokens/s de teto de banda) — gerador e recompensa no Apêndice B.7. Bom para ensinar *reward hacking*: uma recompensa que só confere "o número aparece na resposta" é burlada listando vários números.

**Alternativas de framework para RL em 2 nós** (não testadas no GB10/aarch64):
- **verl 0.9.1** — exemplo oficial `examples/tuning/lora/run_qwen3_8b_fsdp.sh` (GRPO + LoRA, rollout vLLM, treino FSDP) [verificado: arquivo]: `algorithm.adv_estimator=grpo`, `actor_rollout_ref.model.lora_rank=64`, `lora_alpha=32`, `actor.optim.lr=3e-6`, `actor.use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`, `rollout.name=vllm`, `rollout.n=5`, `rollout.gpu_memory_utilization=0.6`, `rollout.layered_summon=True`, `data.max_prompt_length=512`, `data.max_response_length=1024`; dados em parquet no formato (i) da §3.1. Em 2 Sparks: `trainer.nnodes=2 trainer.n_gpus_per_node=1`, rollout TP=1, cluster Ray entre os nós.
- **OpenRLHF 0.11.2** — `python3 -m openrlhf.cli.train_ppo_ray` com `--algo.advantage.estimator group_norm` (GRPO) ou `reinforce_baseline` (recomendado pelo projeto para RLVR), `--rollout.n_samples_per_prompt 8`, `--vllm.num_engines 1 --vllm.tensor_parallel_size 1`, `--ds.lora.rank 16`, `--algo.kl.init_coef 0` (sem referência), `--reward.remote_url http://host:5000/get_reward` (recompensa por HTTP); merge com `openrlhf.cli.lora_combiner` [fonte: README].

### 5.8 Bateria completa e portões de decisão (pré-registrados)

**Logística:** os juízes (receitas do Módulo 1) ocupam os dois Sparks; os candidatos também. Então: (1) sirva os candidatos — base e SFT no servidor da §5.5 (adapter sobre `Qwen/Qwen3.8-27B`) e o DPO **sobre o SFT fundido** (`models/q38-sft-bf16` + `--lora-modules q38-dpo=/work/adapters/q38-dpo`, no outro Spark), porque **um adapter só vale sobre o base em que foi treinado** — ou, mais simples, o merge final `models/q38-final-bf16` — e **gere e grave** todas as respostas (domínio, Arena-Hard v2 opcional, XSTest, StrongREJECT); (2) rode as tarefas verificáveis (lm-eval/lighteval) com os mesmos servidores; (3) derrube tudo, suba a receita-juiz e julgue as respostas gravadas (Apêndice B.1).

| Eixo | Medida | Portão sugerido **[recomendação — ajuste antes de olhar os resultados]** |
|---|---|---|
| Alvo | WR e WR-LC vs base no `test_dominio` (juiz de outra família, duas ordens), IC95% por bootstrap | limite inferior do IC de WR-LC > 50% e WR-LC ≥ 60% |
| Alvo | acerto numérico (tolerância 1%) no domínio | ↑ vs base, IC não cruzando zero |
| Regressão | IFEval (`prompt_level_strict_acc`), GSM8K Platinum, AIME 2025 (2 sementes), em **dois modos** (pensando / não pensando) | Δ ≥ −2 pp (IFEval, GSM8K) e Δ AIME dentro do IC |
| Regressão | BFCL (non-live + multi-turn) ou τ² `retail` 20 tarefas × 4 tentativas (pass^1, pass^4) | Δ dentro do IC |
| Segurança | XSTest: taxa de recusa nos 250 seguros; StrongREJECT (subconjunto): escore médio | recusa nos seguros não sobe > 2 pp; escore StrongREJECT não sobe |
| Deriva | KL(base‖candidato) por token em texto genérico (ex.: 200 janelas de 2.048 tokens de chat genérico) e no domínio; comprimento médio | reporte; investigue se o KL genérico for ≫ o do SFT anterior |
| Velocidade | aceitação MTP por posição (base vs candidato) | reporte (afeta o custo de serviço) |

Use a mesma semente/amostragem para base e candidato; reporte média ± IC entre sementes (o RedHatAI usa 3 sementes; 8 no AIME [fonte: card]).

### 5.9 Merge (com re-adição da MTP)

O transformers **descarta** os tensores `mtp.*` ao carregar o Qwen3.5/3.8 (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` [verificado: `modeling_qwen3_5.py`]); salvar o modelo fundido sem eles quebra o `--speculative-config '{"method":"mtp",…}'`. O checkpoint de referência `RedHatAI/Qwen3.8-27B-NVFP4` resolve exatamente assim: 15 tensores `mtp.*` num shard separado `model_mtp.safetensors` e `re:^mtp.*` no `ignore` da quantização [verificado: `model.safetensors.index.json` e `config.json` do repo]. É a mesma técnica de "adicionar um safetensors e reescrever o índice" do `tools/dense_overlay.py` da receita GLM (`glm-mods.md` M48).

```python
# merge.py — funde um adapter PEFT num Qwen3.5/3.8 (VLM) em BF16 e preserva MTP, processor e template
import argparse, json, os, shutil, torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
src = a.base if os.path.isdir(a.base) else snapshot_download(a.base)

model = Qwen3_5ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")   # BF16, nunca NF4
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, max_shard_size="5GB")
AutoProcessor.from_pretrained(src).save_pretrained(a.out)
for f in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), a.out)

wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
mtp = {}
for k, shard in wm.items():
    if k.startswith("mtp."):
        with safe_open(os.path.join(src, shard), "pt") as f:
            mtp[k] = f.get_tensor(k)
if mtp:   # a MTP continua sendo a do base: espere aceitação menor (§2.6)
    save_file(mtp, os.path.join(a.out, "model_mtp.safetensors"), metadata={"format": "pt"})
    p = os.path.join(a.out, "model.safetensors.index.json"); idx = json.load(open(p))
    idx["weight_map"].update({k: "model_mtp.safetensors" for k in mtp}); json.dump(idx, open(p, "w"), indent=2)
print(f"ok: {len(mtp)} tensores MTP preservados")
```

- Uso: `python merge.py --base models/q38-sft-bf16 --adapter adapters/q38-dpo --out models/q38-final-bf16` (o `--base` aqui é o SFT fundido, porque o DPO foi treinado sobre ele).
- Confira os nomes das chaves salvas contra o índice original (`model.language_model.layers.*`); o transformers 5.x aplica mapeamentos de conversão de pesos no load/save — um diff de nomes indica problema **[recomendação]**.
- Verifique o merge (Apêndice B.8) antes de quantizar.
- **Tempo [estimativa]:** ler e gravar ~111 GB (dois passes de 55,6 GB) — o NVMe do Spark faz ~4,9 GB/s por leitor com O_DIRECT [fonte: `glm-mods.md` §4.1] — ≈ 1–3 min de E/S + carga/merge; **5–15 min**.

### 5.10 Re-quantização do modelo final

**(a) NVFP4 com llm-compressor (vLLM-nativo; recomendado)** — o exemplo oficial do Qwen3.8-27B, trocando o modelo de origem e a calibração:

```python
# quant_nvfp4.py — NVFP4 (MLP) + FP8 dinâmico (atenção/GDN/últimas 8 MLPs) + KV FP8, calibrado com o seu domínio
import json, random
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4, QuantizationScheme
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

SRC, OUT, SEQ = "models/q38-final-bf16", "models/q38-final-nvfp4", 4096
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(SRC)
proc = AutoProcessor.from_pretrained(SRC); tok = proc.tokenizer

random.seed(0)
own = random.sample([json.loads(l) for l in open("data/sft_train.jsonl")], 256)            # domínio, no template real
gen = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]")                  # chat genérico
texts = [tok.apply_chat_template(r["messages"], tokenize=False, **r.get("chat_template_kwargs", {"enable_thinking": False})) for r in own]
texts += [tok.apply_chat_template(m, tokenize=False, enable_thinking=False) for m in gen["messages"]]
ds = Dataset.from_dict({"text": texts}).map(
    lambda s: tok(s["text"], max_length=SEQ, truncation=True, add_special_tokens=False), remove_columns=["text"])

recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(targets=[
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\..*(gate|up|down)_proj$"], **FP8_DYNAMIC),
            "mlp": QuantizationScheme(targets=[r"re:.*mlp\..*(gate|up|down)_proj$"], **NVFP4),
        },
        ignore=["re:visual.*", "re:model.visual.*", "lm_head", "re:^mtp.*"],
        kv_cache_scheme={"num_bits": 8, "type": "float", "symmetric": True, "strategy": "tensor",
                         "dynamic": False, "observer": "static_minmax"},
    ),
]
oneshot(model=model, processor=proc, recipe=recipe, dataset=ds, max_seq_length=SEQ, num_calibration_samples=len(ds))
model.save_pretrained(OUT); proc.save_pretrained(OUT)
# depois: copie chat_template.jinja / *preprocessor_config.json e re-adicione model_mtp.safetensors (mesmo trecho do merge.py)
```

- Diferenças em relação ao exemplo oficial [verificado: `examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py`]: (i) dados próprios + genéricos no lugar de `dataset="perfectblend", splits="train[:512]"`; (ii) `lm_head` e `re:^mtp.*` no `ignore` (o exemplo põe `lm_head` no grupo FP8, mas o checkpoint publicado `RedHatAI/Qwen3.8-27B-NVFP4` mantém `lm_head` em precisão cheia e ignora `mtp` [verificado: `config.json` do repo]); (iii) sem `moe_calibrate_all_experts` (não há MoE no 27B).
- **Tempo [estimativa grosseira]:** 512 × 4.096 ≈ 2,1 M tokens de calibração; um forward completo do 27B nesses tokens ≈ 2 × 27,8e9 × 2,1e6 ≈ 1,2e17 FLOP ≈ **30–60 min** a 30–60 TFLOPS; AWQ faz busca em grade por camada (vários forwards locais) e GPTQ acumula Hessianos e resolve por camada (o kernel Triton novo é "~15x faster" que o caminho eager [fonte: README]) → **várias horas no total**. Teste primeiro com 128 amostras × 2.048 tokens (≈ 1/8 do custo).
- Tamanho esperado: comparável aos 24,69 GB do `RedHatAI/Qwen3.8-27B-NVFP4` [verificado: tamanho].

**(b) NVFP4 com ModelOpt (receita da NVIDIA para este modelo)** — dentro de uma imagem NGC vLLM ≥ 26.05, instalando o Model-Optimizer com `--no-deps` (para não trocar o PyTorch da imagem), como no playbook oficial do Spark [verificado: playbook `nvfp4-quantization`, atualizado 2026-07-22]:
```bash
git clone https://github.com/NVIDIA/Model-Optimizer.git && cd Model-Optimizer
git ls-tree -r --name-only HEAD modelopt_recipes/models/Qwen/Qwen3.8-27B/    # a receita precisa existir na versão escolhida
pip install --no-deps -e . && pip install accelerate datasets omegaconf 'pulp<4.0'
cd examples/hf_ptq        # no main; o playbook (tag 0.45.0) usa examples/llm_ptq
python hf_ptq.py --pyt_ckpt_path /work/models/q38-final-bf16 \
  --recipe models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_local_hessian \
  --dataset /work/data/calib_dominio.jsonl,cnn_dailymail --calib_size 512,512 --calib_seq 4096 \
  --export_path /work/models/q38-final-nvfp4-modelopt
```
- A receita `nvfp4_w4a4_mlp_fp8_attn_local_hessian` importa `nvfp4_w4a4_mlp_fp8_attn_max` e troca o algoritmo para `local_hessian` com `fp8_scale_sweep: true` e calibração camada a camada (`layerwise.enable`, `get_qdq_activations_from_prev_layer`) — descrita como a atribuição NVFP4/FP8 exportada de uma varredura AutoQuantize de 5,5 bits [verificado: YAML no main]. É coerente com o card `nvidia/Qwen3.8-27B-NVFP4` ("Local Hessian", 2.048 amostras) [fonte].
- `--dataset` aceita `.jsonl` local com coluna `messages` (template aplicado) e listas separadas por vírgula, com `--calib_size` por dataset; padrões `--calib_size 1024`, `--calib_seq 512` [verificado: `hf_ptq.py` e `dataset_utils.py`]. O padrão de calibração do playbook é o `nvidia/Nemotron-Post-Training-Dataset-v2` (gated) e o card usou o v3 (gated) [fonte].

**(c) EXL3 (opcional, para comparação/uso fora do vLLM upstream):**
```bash
python convert.py -i /work/models/q38-final-bf16 -o /work/models/q38-final-exl3-4.0 -w /work/exl3_tmp -b 4.0 -hb 6
```
O vLLM upstream **não** carrega EXL3: no Módulo 1 o suporte vem do overlay `exl3.py` da receita GLM (registra `@register_quantization_config("exl3")`, pensado para experts MoE e, opcionalmente, lineares densos via `non_routed_exl3`) [fonte: `vllm-ref.md` §12.2]. Para um 27B denso, sirva o EXL3 com o próprio exllamav3 (ex.: TabbyAPI) **[não verificado no GB10]** e use-o principalmente para o painel de KLD (§5.11), onde o `model_diff.py` é exato.

**(d) Baseline barato:** `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head", "re:visual.*", "re:^mtp.*"])` — sem calibração; serve para separar "o fine-tune piorou" de "a quantização 4-bit piorou".

### 5.11 KLD e regressão pós-quantização

```bash
# (1) Exato, EXL3 × BF16 fundido (um Spark; ~72 GB) — exllamav3 v1.5.3
python eval/model_diff.py -ma /work/models/q38-final-exl3-4.0 -mb /work/models/q38-final-bf16 -r 100 -l 2048   # B = referência
# imprime PPL de A e B, top-K, KL(A,B) e KL(B,A), mediana/p90 por token e KL por faixa de confiança de referência

# (2) Proxy in situ com os kernels de produção (dois Sparks):
#     Spark A: vLLM com models/q38-final-bf16 ; Spark B: vLLM com models/q38-final-nvfp4  (ambos --max-logprobs 20)
python kl_vllm_proxy.py --ref http://SPARK_A:8000/v1 --cand http://SPARK_B:8000/v1 --texts data/kl_texts.jsonl --k 20   # Apêndice B.5

# (3) Mesmo protocolo na referência pública, para calibrar a expectativa:
#     Qwen/Qwen3.8-27B (BF16) × nvidia/Qwen3.8-27B-NVFP4  e  × RedHatAI/Qwen3.8-27B-NVFP4
# (4) Piso: BF16 × BF16 (outro boot / outro max-num-seqs)
```
Critério **[recomendação]**: KL(final_BF16‖final_NVFP4) da mesma ordem que KL(base_BF16‖NVFP4 de referência) medido pelo mesmo script; ΔNLL e top-1 agreement próximos; e a bateria da §5.8 rodada **também no NVFP4** (o RedHatAI publica "recovery" entre 98,27% e 100,88% para o seu NVFP4 do base [fonte: card] — esse é o patamar a buscar).

### 5.12 Servindo o modelo final

```bash
# produção: NVFP4 fundido, TP=1, um servidor por Spark (2 réplicas atrás de um balanceador simples)
docker run -d --name tutor --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 \
  -v /data/work/models:/models vllm/vllm-openai:v0.30.0 \
  /models/q38-final-nvfp4 --served-model-name tutor-q38 --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 65536 --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8_e4m3 --max-num-seqs 16 --enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"tutor-q38",
  "messages":[{"role":"user","content":"Quanto KV ocupa um token no fp8_ds_mla?"}],"max_tokens":300,"temperature":0.7}'
vllm bench serve --model tutor-q38 --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 64 --request-rate inf      # repita A/B/A como no cap. 45 do Módulo 1
```

- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` é o que o card do RedHatAI usa para o Qwen3.8-27B; também há um rascunhador DSpark treinado para o **base** (`RedHatAI/Qwen3.8-27B-speculator.dspark`, `"method":"dspark"`, 7 tokens) [fonte: card]. Ambos foram treinados na distribuição do base → **meça a aceitação** depois do fine-tune; se cair muito, re-treine o rascunhador (assunto do cap. 37 do Módulo 1) ou reduza `num_speculative_tokens`.
- **Expectativa de desempenho [não verificado/estimativa]:** fórum NVIDIA (2026-08-15, vLLM 0.27.1, 16 requisições concorrentes) mediu para o Qwen3.8-27B **NVFP4 87,91–134,41 tok/s** de saída agregada (TPOT 120,60 ms no cenário decode-heavy) contra **FP8 65,58–104,44 tok/s** (TPOT 160,19 ms) — o seu NVFP4 deve ficar perto disso. Em 1 stream sem especulação, o teto de banda é ~230 GB/s ÷ ~22–25 GB ≈ **9–10 tok/s**; com MTP aceitando bem, ~1,5–2× **[estimativa]**.
- **TP=2 entre os Sparks** só se a latência de um único usuário importar mais que vazão: o 27B NVFP4 cabe folgado em um nó, e cada passo de decode em TP=2 paga all-reduces por camada no RoCE (assunto dos caps. 38–39) — meça A/B/A antes de adotar.
- **Alternativa de desenvolvimento** (caminho A): base `models/q38-sft-bf16` + `--enable-lora --max-loras 4 --max-lora-rank 32 --lora-modules dpo-b01=/work/adapters/q38-dpo dpo-b03=/work/adapters/q38-dpo-beta03` para comparar variantes de DPO lado a lado (todas treinadas sobre o mesmo SFT fundido); LoRA sobre base FP8/NVFP4 não foi verificado no sm_121 (§2.3).
- **Copiloto da plataforma:** preset "OpenAI-compatible" com `baseUrl http://SPARK:8000/v1` e `model tutor-q38` (`mlx-copilot.md` §1); o vLLM devolve o raciocínio em campo separado quando há `--reasoning-parser`.

### 5.13 Checklists da trilha B

**Antes de treinar**
- [ ] NCCL entre os nós em `NET/IB` (não socket); `NCCL_IB_GID_INDEX` conferido por nó.
- [ ] `PIP_CONSTRAINT` fixando o torch da imagem; `fla` + `causal-conv1d` importam; teste de 10 passos com tokens/s na faixa esperada (§5.3).
- [ ] `model_init_kwargs={"dtype": torch.bfloat16}` em **todos** os configs do TRL (o padrão, quando o modelo é passado como string, é **float32** [verificado: doc SFT/DPO/KTO]).
- [ ] Regex de `target_modules` casa só `language_model` (imprima `model.print_trainable_parameters()` e confira ~217,6 M em r=32).
- [ ] `chat_template_kwargs` idêntico em dados, avaliação e servidor.
- [ ] Linha de base completa gravada (3 sementes ou 2 no orçamento reduzido).
- [ ] Dados descontaminados; splits por termo; `pref_valid`/`sft_valid` separados.
- [ ] `drop_caches` antes de carregar; disco livre ≥ 4 × 56 GB (base, SFT fundido, final fundido, quantizados).

**Durante**
- [ ] SFT: `eval_loss` cai; `mean_token_accuracy` não satura cedo; `grad_norm` estável.
- [ ] DPO: `rewards/accuracies` < ~0,95; `logps/chosen` sem queda contínua; comprimento médio estável.
- [ ] GRPO: `frac_reward_zero_std` < ~0,5; `completions/clipped_ratio` baixo; `entropy` sem colapso; ler `log_completions`.
- [ ] Checkpoints a cada 100–200 passos; espaço em disco.

**Depois**
- [ ] Portões da 5.8 avaliados com IC; decisão registrada **antes** de quantizar.
- [ ] Merge verificado (logits) e MTP re-adicionada.
- [ ] Quantização calibrada com mistura domínio + genérico; MTP e `lm_head` no `ignore`.
- [ ] KLD do quantizado vs merge BF16 ≈ KLD do NVFP4 de referência vs base; bateria da §5.8 repetida no quantizado.
- [ ] Aceitação de MTP/DSpark medida; `num_speculative_tokens` ajustado.
- [ ] `vllm bench serve` A/B/A contra o NVFP4 de referência; versão, hashes e configs registrados (`sha256sum` dos safetensors, como as receitas do Módulo 1 fazem).
- [ ] Plano de rollback: manter o endpoint anterior até o novo passar nos portões em produção.

### 5.14 Solução de problemas

| Sintoma | Causa provável | Correção |
|---|---|---|
| OOM ao carregar o 27B no treino, com memória "livre" | page cache da memória unificada / modelo em fp32 | `drop_caches`; `model_init_kwargs={"dtype": torch.bfloat16}` |
| tokens/s ≪ esperado no Qwen3.5/3.8 | GDN no fallback PyTorch | instale/compile `fla` e `causal-conv1d` para sm_121 (§5.1); confirme com micro-benchmark |
| `assistant_only_loss` falha ("no assistant masks") | template sem `{% generation %}` | use o template do modelo conhecido (o TRL troca para `qwen3_8_training.jinja`) ou passe `chat_template_path` |
| modelo ajustado responde com dois blocos `<think>` | raciocínio colocado inline em `content` | use `reasoning_content` (§3.1 a′) |
| modelo ajustado parou de pensar no modo pensamento | SFT só com `<think></think>` vazio | inclua dados com raciocínio ou avalie/sirva sem pensamento |
| respostas diferentes entre avaliação e produção | `enable_thinking`/`reasoning_effort` divergentes (o template muda o system prompt) | `--default-chat-template-kwargs` no vLLM igual ao dos dados |
| `ncclCommInitRank` trava | interface errada (aliases 10.0.0.x) | `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` por nó, como no Módulo 1 |
| GRPO: erro ao conectar/sincronizar com o vLLM | versão fora de 0.20.1–0.30.0, falta `VLLM_SERVER_DEV_MODE=1` ou `--weight-transfer-config` | ver §5.7; mesma versão de `vllm` nos dois lados |
| GRPO: `reward` sobe, acerto real não | hacking da recompensa (formato/comprimento) | ler amostras; endurecer o verificador; `mask_truncated_completions` |
| KTO: erro de batch | `per_device_train_batch_size=1` ou amostragem aleatória | ≥ 2 (recomendado ≥ 4), `train_sampling_strategy="sequential"` |
| DPO: `sync_ref_model=True` falha | incompatível com PEFT sem `ref_model` e com `precompute_ref_log_probs` | desligue |
| vLLM sobe o modelo fundido mas o MTP falha | `mtp.*` não salvos | re-adicione (§5.9) |
| vLLM com LoRA: adapter "não aplica" | módulos-alvo não suportados/nome diferente | `--lora-target-modules`; confira nomes do PEFT vs `packed_modules_mapping` |
| NVFP4 do fine-tune com KLD bem maior que o de referência | calibração só de domínio ou sem template; `lm_head`/últimas camadas em 4 bits | mistura de calibração; mantenha as 8 últimas MLPs em FP8 e `lm_head` em BF16 |
| Judge "prefere" sempre o modelo ajustado | autopreferência (juiz parente do professor/aluno) ou viés de posição | juiz de outra família; troca de posição; painel |

---

## Apêndice A — Fórmulas e contas

### A.1 Win rate com intervalo de confiança

- Estimador: p̂ = (W + T/2)/n.
- **Bootstrap por prompt (recomendado):** reamostre os n prompts com reposição B = 2.000 vezes, recalcule p̂, use os percentis 2,5% e 97,5%. Para diferença entre dois candidatos avaliados nos mesmos prompts, reamostre **pares** (mesmo índice para os dois).
- Wilson (sem empates): (p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n), z = 1,96.
- Tamanho de amostra para meia-largura E: n ≈ z²·p(1−p)/E² → p = 0,5, E = 0,05 → n ≈ 384.

### A.2 McNemar para métricas binárias por item (regressão pareada)

Itens certos no base e errados no candidato = b; o contrário = c. χ² = (|b − c| − 1)²/(b + c) com 1 g.l. (ou binomial exato para b + c pequeno). Detecta regressão mesmo quando a acurácia agregada parece igual.

### A.3 Bradley–Terry (e escala tipo Elo)

P(i ≻ j) = σ(β_i − β_j). Ajuste por máxima verossimilhança sobre todas as comparações (empate = meia vitória para cada lado, ou modelo de Davidson), fixando β_base = 0. Escala tipo Elo: R_i = C + (400/ln 10)·β_i. **Controle de estilo** (ideia do LMArena): logit P(i ≻ j) = β_i − β_j + γᵀ(f_i − f_j), com f = (comprimento normalizado, nº de cabeçalhos, listas, negrito…); o ranking usa os β com γ ajustado.

### A.4 Win rate controlado por comprimento (versão simplificada para o laboratório)

Para um candidato m contra o baseline b, por prompt x com veredito y ∈ {0, 1} (empates como 0,5 de peso em cada classe):
logit P(y = 1) = θ + φ·tanh(Δℓ(x)/σ_Δℓ), Δℓ = len_m − len_b (tokens). WR_LC = 100·σ(θ) (contrafactual Δℓ = 0). O paper do AlpacaEval-LC acrescenta um termo de dificuldade por instrução (ψ_m − ψ_b)·γ_x (§1.5).

### A.5 pass@k e pass^k

- pass@k (Chen et al. 2021, HumanEval): com n amostras e c corretas por problema, pass@k = E_problemas[1 − C(n−c, k)/C(n, k)].
- pass^k (Yao et al. 2024, τ-bench): pass^k = E_tarefas[C(c, k)/C(n, k)] — probabilidade de **todas** as k tentativas darem certo. pass^1 = pass@1; pass^k decresce com k e mede confiabilidade.

### A.6 KL: estimadores e superotimização

- KL(π_θ‖π_ref) = E_{y∼π_θ}[log π_θ(y) − log π_ref(y)]. Com r = π_ref/π_θ por token: k1 = −log r; k2 = ½(log r)²; **k3 = (r − 1) − log r** (não-viesado, ≥ 0; usado no GRPO). Em código estável: `expm1(Δ) − Δ`, Δ = log π_ref − log π_θ.
- Objetivo do RLHF: max E[r(x, y)] − β·KL(π‖π_ref). Política ótima: π*(y|x) ∝ π_ref(y|x)·exp(r(x, y)/β) — é daí que o DPO tira a recompensa implícita r̂ = β·log(π/π_ref) (métricas `rewards/*` do TRL).
- Gao et al. 2022 (d = √KL): best-of-n R(d) = d(α − β·d); RL R(d) = d(α − β·log d); KL(best-of-n) = log n − (n−1)/n **[transcrito de memória; conferir no paper]**.
- KL truncado (para comparar modelos com top-k da referência): com S = top-k de P e a cauda T agrupada, KL_S∪T = Σ_{v∈S} p_v(log p_v − log q_v) + p_T·log(p_T/q_T) ≤ KL(P‖Q) (desigualdade de processamento de dados para a partição) — uma **cota inferior** que fica apertada quando p_T é pequeno (top-64 já cobre quase toda a massa em texto comum).

### A.7 Contas de LoRA, memória e tempo usadas no documento

**Parâmetros LoRA** = Σ_módulos r·(d_in + d_out).

Qwen3.8-27B (16 camadas de atenção, 48 GDN, 64 MLPs; dimensões do `config.json`):
- atenção (×16): `q_proj` 5120→12288 (saída gated: 24×256×2), `k_proj` e `v_proj` 5120→1024, `o_proj` 6144→5120;
- GDN (×48): `in_proj_qkv` 5120→10240 (= 2×16×128 + 48×128), `in_proj_z` 5120→6144, `out_proj` 6144→5120 (`in_proj_a/b` 5120→48 fora);
- MLP (×64): `gate_proj`/`up_proj` 5120→17408, `down_proj` 17408→5120.
- Resultado: **r=16 → 108,8 M; r=32 → 217,6 M; r=64 → 435,2 M** (só atenção+GDN: 39,6 / 79,2 / 158,3 M). Estados de treino em fp32 (peso 4 B + gradiente 4 B + Adam m,v 8 B = 16 B/parâmetro): 1,74 / **3,48** / 6,96 GB. Adapter em BF16: 218 / **435** / 870 MB.

Qwen3.5-9B (8 atenção, 24 GDN, 32 MLPs; hidden 4096, Q 16×256×2, KV 4×256, GDN K 16×128 e V 32×128, FFN 12288): **r=16 → 40,1 M; r=32 → 80,2 M**.

Qwen3-4B-Instruct-2507 (36 camadas, hidden 2560, Q 32×128, KV 8×128, FFN 9728): todas as lineares de 16 camadas, r=16 → **14,7 M** (r=8 → 7,3 M); todas as 36 camadas, r=16 → 33,0 M.

**Memória de pesos:** BF16 = 2 B/parâmetro (27,78 B → 55,56 GB; 9,65 B → 19,3 GB; 4,02 B → 8,04 GB). NF4 ≈ 0,5 B/parâmetro + constantes (double quant ≈ 0,127 bit/parâmetro → ~0,44 GB no 27B) — embedding e `lm_head` ficam BF16 no bitsandbytes (2 × 248.320 × 5120 × 2 B = 5,09 GB no 27B).

**Ativações com gradient checkpointing (por sequência de T tokens):** entradas salvas por camada T·h·2 B (27B, T = 4096: 41,9 MB × 64 = 2,7 GB) + pico de recomputação de **uma** camada (MLP: T·17408·2 B × ~3–4 tensores ≈ 0,4–0,6 GB; mais estados do GDN/atenção). **Logits** T·V·4 B em fp32 (27B: 4096 × 248.320 × 4 B = 4,07 GB, e o gradiente do mesmo tamanho) — o `chunked_nll` do SFT do TRL evita materializá-los [verificado].

**FLOPs de treino:** forward ≈ 2·N por token; backward ≈ 4·N no full FT (gradientes de ativação + de peso); com LoRA o gradiente de peso do base some, mas o de ativação (≈ 2·N) continua, e o checkpointing recomputa o forward (≈ 2·N) → **≈ 6·N por token** com LoRA + checkpointing. Âncora empírica: 6 × 27e9 × 370 tok/s ≈ 60 TFLOPS e 6 × 9e9 × 1.095 tok/s ≈ 59 TFLOPS nas medições públicas de LoRA no GB10 [não verificado: albond] → **~60 TFLOPS efetivos com os kernels certos**, ~30 TFLOPS como faixa conservadora.

**Decode (teto de banda):** passos/s ≤ banda ÷ bytes lidos por passo (pesos + KV/estado). GB10: 273 GB/s nominal, ~230 GB/s efetivo (Módulo 1). 9B BF16: 230/19,3 ≈ 12 passos/s; 27B NVFP4 (~22–25 GB): ≈ 9–10 passos/s; com lote B, até B tokens por passo enquanto o compute não satura.

**Mac M4:** matmul BF16 medido 2,78 TFLOPS (não verificado); treino efetivo assumido 1–2 TFLOPS. Qwen3-4B QLoRA nas 16 últimas camadas: forward 2·N ≈ 8,0 GFLOP + backward de ativações nas 16 camadas e no `lm_head` ≈ 2·(16 × 0,101 B + 0,389 B) ≈ 4,0 GFLOP + recomputação ≈ 3,2 GFLOP ≈ **15 GFLOP/token** → 65–130 tok/s.

---

## Apêndice B — Scripts de apoio

### B.1 Avaliação de domínio: gerar e julgar (duas ordens)

```python
# eval_dominio.py — fase 1: gera respostas de N "modelos" (base/adapters) ; fase 2: julga contra o base com troca de posição
import argparse, json, re, concurrent.futures as cf
from openai import OpenAI
ap = argparse.ArgumentParser(); ap.add_argument("fase", choices=["gerar", "julgar"])
ap.add_argument("--url"); ap.add_argument("--modelos", nargs="+"); ap.add_argument("--juiz"); ap.add_argument("--base", default="q38-base")
ap.add_argument("--teste", default="data/test_dominio.jsonl"); ap.add_argument("--saida", default="results/dominio")
a = ap.parse_args(); cli = OpenAI(base_url=a.url, api_key="x")
qs = [json.loads(l) for l in open(a.teste)]

if a.fase == "gerar":
    for m in a.modelos:
        def gen(q):
            r = cli.chat.completions.create(model=m, temperature=0.7, top_p=0.8, max_tokens=1024,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "user", "content": q["question"]}])
            return {"id": q["id"], "resp": r.choices[0].message.content, "toks": r.usage.completion_tokens}
        with cf.ThreadPoolExecutor(16) as ex, open(f"{a.saida}.{m}.jsonl", "w") as w:
            for row in ex.map(gen, qs): w.write(json.dumps(row, ensure_ascii=False) + "\n")
else:
    RUB = ("Compare A e B. Critérios: correção técnica; responde ao que foi perguntado; números/unidades; concisão. "
           "Ignore comprimento e formatação. Até 3 frases e termine com 'VEREDITO: A', 'VEREDITO: B' ou 'VEREDITO: EMPATE'.")
    def v(q, x, y):
        r = cli.chat.completions.create(model=a.juiz, temperature=0, max_tokens=300, messages=[
            {"role": "system", "content": RUB}, {"role": "user", "content": f"PERGUNTA:\n{q}\n\nRESPOSTA A:\n{x}\n\nRESPOSTA B:\n{y}"}])
        m = re.search(r"VEREDITO:\s*(A|B|EMPATE)", r.choices[0].message.content or ""); return m.group(1) if m else "EMPATE"
    base = {r["id"]: r for r in map(json.loads, open(f"{a.saida}.{a.base}.jsonl"))}
    for m in a.modelos:
        cand = {r["id"]: r for r in map(json.loads, open(f"{a.saida}.{m}.jsonl"))}
        def judge(q):
            c, b = cand[q["id"]], base[q["id"]]
            v1, v2 = v(q["question"], c["resp"], b["resp"]), v(q["question"], b["resp"], c["resp"])
            y = 1.0 if (v1, v2) == ("A", "B") else 0.0 if (v1, v2) == ("B", "A") else 0.5
            return {"id": q["id"], "y": y, "len_c": c["toks"], "len_b": b["toks"]}
        with cf.ThreadPoolExecutor(8) as ex, open(f"{a.saida}.juizo.{m}.jsonl", "w") as w:
            for row in ex.map(judge, qs): w.write(json.dumps(row) + "\n")
```
Uso: `python eval_dominio.py gerar --url http://SPARK_A:8000/v1 --modelos q38-base q38-sft` (servidor da §5.5) e `--url http://SPARK_B:8000/v1 --modelos q38-dpo` (DPO sobre o SFT fundido, §5.8) → depois, com a receita-juiz no ar, `python eval_dominio.py julgar --url http://SPARK_HEAD:8888/v1 --juiz deepseek-v4-flash-vision-exp --modelos q38-sft q38-dpo`.

### B.2 Bootstrap, WR e WR-LC

```python
# wr_lc.py — WR com IC por bootstrap e WR controlado por comprimento (regressão logística, Apêndice A.4)
import json, sys, numpy as np
from scipy.optimize import minimize
rows = [json.loads(l) for l in open(sys.argv[1])]
y = np.array([r["y"] for r in rows]); d = np.array([r["len_c"] - r["len_b"] for r in rows], float)
rng = np.random.default_rng(0)
boot = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
print(f"WR = {100*y.mean():.1f}%  IC95% [{100*np.percentile(boot,2.5):.1f}, {100*np.percentile(boot,97.5):.1f}]")
x = np.tanh(d / (d.std() + 1e-9))
def nll(w):                                     # y fracionário (0, 0.5, 1) = entropia cruzada com rótulo suave
    z = w[0] + w[1] * x; p = 1 / (1 + np.exp(-z))
    return -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
w = minimize(nll, np.zeros(2)).x
print(f"WR-LC = {100/(1+np.exp(-w[0])):.1f}%  (coef. de comprimento φ = {w[1]:.3f})")
```

### B.3 Bradley–Terry entre checkpoints

```python
# bt.py — entrada: JSONL {"a": "q38-dpo", "b": "q38-base", "y": 1|0|0.5}; saída: escala tipo Elo (base = 1000)
import json, sys, numpy as np
from scipy.optimize import minimize
rows = [json.loads(l) for l in open(sys.argv[1])]
names = sorted({r["a"] for r in rows} | {r["b"] for r in rows}); ix = {n: i for i, n in enumerate(names)}
A = np.array([ix[r["a"]] for r in rows]); B = np.array([ix[r["b"]] for r in rows]); Y = np.array([r["y"] for r in rows], float)
ref = ix.get("q38-base", 0)
def nll(beta):
    beta = beta.copy(); beta[ref] = 0.0
    z = beta[A] - beta[B]; p = 1 / (1 + np.exp(-z))
    return -np.sum(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12)) + 1e-4 * np.sum(beta ** 2)
beta = minimize(nll, np.zeros(len(names))).x; beta[ref] = 0.0
for n in sorted(names, key=lambda n: -beta[ix[n]]): print(f"{n:20s} {1000 + 400/np.log(10)*beta[ix[n]]:7.1f}")
```

### B.4 KL exato em dois passes (HF/transformers)

```python
# kl_hf_two_pass.py — passo "ref" grava top-k log-probs de P; passo "cand" calcula a cota inferior do KL(P||Q) (Apêndice A.6)
import argparse, json, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
ap = argparse.ArgumentParser(); ap.add_argument("modo", choices=["ref", "cand"]); ap.add_argument("--model"); ap.add_argument("--texts")
ap.add_argument("--cache", default="kl_ref.pt"); ap.add_argument("--k", type=int, default=64); ap.add_argument("--maxlen", type=int, default=2048)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
texts = [json.loads(l)["text"] for l in open(a.texts)]
saved = torch.load(a.cache) if a.modo == "cand" else []
tot = n = agree = 0.0
with torch.no_grad():
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=a.maxlen, add_special_tokens=False).input_ids.cuda()
        logq = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
        if a.modo == "ref":
            v, idx = logq.topk(a.k, dim=-1); saved.append({"v": v.cpu(), "i": idx.cpu()}); continue
        pv, pi = saved[i]["v"].cuda(), saved[i]["i"].cuda()
        q_s = logq.gather(-1, pi); p_s = pv.exp()
        p_t = (1 - p_s.sum(-1)).clamp_min(1e-12); q_t = (1 - q_s.exp().sum(-1)).clamp_min(1e-12)
        kl = (p_s * (pv - q_s)).sum(-1) + p_t * (p_t.log() - q_t.log())
        agree += (logq.argmax(-1) == pi[:, 0]).sum().item(); tot += kl.sum().item(); n += kl.numel()
if a.modo == "ref": torch.save(saved, a.cache)
else: print(f"KL(P||Q) >= {tot/n:.6f} nats/token | top-1 igual {agree/n:.4f} | {int(n)} posições")
```
- Para drift do pós-treino: `ref` com o base BF16, `cand` com o fundido BF16. Para quantização: `ref` com o fundido BF16, `cand` com o quantizado **se** o transformers carregar o formato (compressed-tensors) — dequantiza pesos e não reproduz os kernels W4A4 **[não verificado]**; para kernels reais use o Apêndice B.5.
- Cache: 64 × (4 + 8) B por posição ≈ 768 B; 100 textos × 2.048 posições ≈ 157 MB.

### B.5 Proxy de KL/ΔNLL/top-1 entre dois servidores vLLM

```python
# kl_vllm_proxy.py — usa prompt_logprobs (top-k) e /tokenize; ΔNLL e top-1 são exatos; o "KL no topo comum" é proxy
import argparse, json, math, requests
ap = argparse.ArgumentParser(); ap.add_argument("--ref"); ap.add_argument("--cand"); ap.add_argument("--texts")
ap.add_argument("--k", type=int, default=20); ap.add_argument("--ref-model"); ap.add_argument("--cand-model")
a = ap.parse_args()
def plp(base, model, text):
    r = requests.post(f"{base}/completions", timeout=900, json={"model": model, "prompt": text, "max_tokens": 1,
                      "temperature": 0, "prompt_logprobs": a.k}).json()
    return r["choices"][0]["prompt_logprobs"]                       # [None, {tok_id: {"logprob":…, "rank":…}}, …]
def toks(base, model, text):
    return requests.post(base.replace("/v1", "") + "/tokenize", json={"model": model, "prompt": text}).json()["tokens"]
nll_p = nll_q = kl = agree = n = 0.0
for line in open(a.texts):
    t = json.loads(line)["text"]; ids = toks(a.ref, a.ref_model, t)
    P, Q = plp(a.ref, a.ref_model, t), plp(a.cand, a.cand_model, t)
    for pos in range(1, min(len(P), len(Q), len(ids))):
        p = {int(k): v["logprob"] for k, v in P[pos].items()}; q = {int(k): v["logprob"] for k, v in Q[pos].items()}
        tid = ids[pos]; nll_p -= p[tid]; nll_q -= q[tid]
        agree += max(p, key=p.get) == max(q, key=q.get)
        S = [k for k in p if k in q]
        if S:
            zp = math.log(sum(math.exp(p[k]) for k in S)); zq = math.log(sum(math.exp(q[k]) for k in S))
            kl += sum(math.exp(p[k] - zp) * ((p[k] - zp) - (q[k] - zq)) for k in S)
        n += 1
print(f"NLL ref {nll_p/n:.4f} | NLL cand {nll_q/n:.4f} | ΔNLL {(nll_q-nll_p)/n:+.4f} | top-1 igual {agree/n:.4f} | "
      f"KL(topo comum, renormalizado) {kl/n:.5f} | {int(n)} posições")
```
O `prompt_logprobs` existe no protocolo de completions do vLLM 0.30.0 e o servidor inclui o token real de cada posição no dicionário mesmo fora do top-k **[o primeiro fato verificado no código; o segundo é comportamento documentado do vLLM que não reconferi nesta tag]**; `/tokenize` é endpoint do servidor OpenAI do vLLM **[não reconferido nesta tag]**. Suba os servidores com `--max-logprobs ≥ k`.

### B.6 Descontaminação por n-gramas

```python
# decontam.py — remove do treino itens que compartilham um 13-grama normalizado com qualquer conjunto de avaliação
import json, re, sys, unicodedata
N = 13
def words(s): return re.findall(r"\w+", unicodedata.normalize("NFKC", s.lower()))
def grams(s): w = words(s); return {" ".join(w[i:i + N]) for i in range(max(0, len(w) - N + 1))}
ev = set()
for f in sys.argv[2:]:
    for l in open(f): ev |= grams(l)
keep = drop = 0
with open(sys.argv[1]) as fi, open(sys.argv[1].replace(".jsonl", ".clean.jsonl"), "w") as fo:
    for l in fi:
        if grams(l) & ev: drop += 1; continue
        fo.write(l); keep += 1
print(f"mantidos {keep}, removidos {drop}")
```
O limiar de 13-gramas segue a prática popularizada pelo relatório do GPT-3 **[de memória; conferir]**; para PT-BR e textos curtos (perguntas de uma linha), complemente com similaridade aproximada (MinHash) e checagem de **mesmo termo do glossário** entre treino e teste (§3.2).

### B.7 RLVR de domínio: gerador e recompensas

```python
# rlvr_dominio.py — problemas de contas de serviço com resposta exata (TRL: prompt/solution; mlx-lm-lora: prompt/answer)
import json, random, re
random.seed(0)
BITS = {"BF16": 16, "FP8": 8, "NVFP4": 4 + 8 / 16, "MXFP4": 4 + 8 / 32, "INT4 g128 (escala FP16)": 4 + 16 / 128}
def problema():
    k = random.choice(["pesos", "kv", "decode"])
    if k == "pesos":
        n = random.choice([4, 8, 9, 14, 27, 32, 70]); f = random.choice(list(BITS)); ans = n * 1e9 * BITS[f] / 8 / 1e9
        q = f"Quantos GB ocupam {n}B parâmetros em {f} ({BITS[f]:.4g} bits efetivos por peso)?"
    elif k == "kv":
        L, h, d, b = random.choice([8, 16, 32]), random.choice([2, 4, 8]), random.choice([128, 256]), random.choice([1, 2])
        ans = 2 * L * h * d * b / 1024; q = f"KV por token, em KiB: {L} camadas de atenção, {h} cabeças KV, head_dim {d}, {b} byte(s) por valor (K e V)."
    else:
        gb, bw = random.choice([9.6, 19.3, 24.7, 55.6]), random.choice([230, 273]); ans = bw / gb
        q = f"Teto de decode em tokens/s (1 stream) lendo {gb} GB por token com {bw} GB/s."
    return q + " Responda com 2 casas decimais dentro de um único \\boxed{}.", f"{ans:.2f}"
with open("data/rlvr_dominio.jsonl", "w") as w:
    for _ in range(5000):
        q, s = problema()
        w.write(json.dumps({"prompt": [{"role": "user", "content": q}], "solution": s, "answer": s}, ensure_ascii=False) + "\n")

def _ok(txt, s):
    caixas = re.findall(r"\\boxed\{([^{}]*)\}", txt or "")
    if len(caixas) != 1: return 0.0                   # exatamente UMA caixa: não deixa "chutar" vários números
    try: v = float(caixas[0].replace(",", "."))
    except ValueError: return 0.0
    return 1.0 if abs(v - float(s)) <= 0.01 * max(1.0, abs(float(s))) else 0.0

def recompensa_trl(completions, solution, **kwargs):          # GRPOTrainer(reward_funcs=[recompensa_trl])
    return [_ok(c[-1]["content"] if isinstance(c, list) else c, s) for c, s in zip(completions, solution)]

try:                                                          # mlx-lm-lora: --reward-functions-file rlvr_dominio.py --reward-functions recompensa_mlx
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
    @register_reward_function()
    def recompensa_mlx(prompts, completions, answer, types=None):
        return [_ok(c, s) for c, s in zip(completions, answer)]
except ImportError:
    pass
```
(Para o GRPO do TRL o dataset em conversa é `[{"role":"user",…}]`; para o `mlx-lm-lora`, converta `prompt` para string.)

### B.8 Verificação do merge

```python
# verify_merge.py — compara base+adapter (sem merge) com o fundido, em sequência (dois 27B não cabem juntos num Spark)
import gc, json, sys, torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
base, adapter, merged = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(merged)
prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
           for p in ["Explique o KV cache.", "O que é NVFP4?", "Defina LoRA.", "Quanto é 17*23?"]]
def logits(m):
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            out.append(m(input_ids=ids).logits[0].float().cpu())
    return out
m = PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda"), adapter).eval()
A = logits(m); del m; gc.collect(); torch.cuda.empty_cache()
m = Qwen3_5ForConditionalGeneration.from_pretrained(merged, dtype=torch.bfloat16, device_map="cuda").eval()
B = logits(m)
for a, b in zip(A, B):
    print(f"max|Δlogit| {float((a - b).abs().max()):.4f}  top-1 igual {float((a.argmax(-1) == b.argmax(-1)).float().mean()):.4f}")
```

---

## Apêndice C — Mapa "método do Módulo 2 → onde rodar no laboratório" (verificado nas versões da §0.2)

| Método (lista do usuário) | Ferramenta/flag | Situação |
|---|---|---|
| **SFT** | TRL `SFTTrainer`; `mlx_lm.lora`; `mlx_lm_lora.train --train-mode sft` (perdas `nll`/`chunked_nll`/`dft`); OpenRLHF `openrlhf.cli.train_sft`; verl SFT | estável |
| **RLHF clássico** (RM + PPO) | TRL `RewardTrainer` (estável; `center_rewards_coefficient`, coluna `margin`) + **PPO fora do TRL** (removido na v1.13.0): OpenRLHF `train_ppo_ray` (PPO com crítico é o padrão) ou verl `ppo_trainer`; no Mac, `mlx_lm_lora.train --train-mode ppo` / `rlhf_reinforce` com `--judge <modelo-recompensa>` | estável fora do TRL |
| **DPO** | TRL `DPOTrainer` `loss_type=["sigmoid"]`; MLX `--train-mode dpo`; OpenRLHF `train_dpo` | estável |
| **IPO** | TRL `DPOConfig(loss_type=["ipo"])`; MLX `--dpo-cpo-loss-type ipo`; OpenRLHF (DPO/IPO/cDPO) | estável |
| **KTO** | TRL `KTOTrainer` (estável desde v1.8.0; `loss_type` `kto` ou `apo_zero_unpaired`) | **não** existe no `mlx-lm-lora` 3.1.3 |
| **SimPO** | TRL `trl.experimental.cpo.CPOTrainer(loss_type="simpo", cpo_alpha=0.0)`; aproximação "LN-DPO com referência": `DPOConfig(loss_type=["sigmoid_norm"])` | experimental |
| **PPO** (com outras recompensas) | OpenRLHF/verl; MLX `--train-mode ppo` | — |
| **AWR / regressão de vantagem** | não há "AWR" pronto nas bibliotecas de LLM verificadas; o parente mais próximo é **A\*-PO** (`trl.experimental.a2po.A2POTrainer`) | experimental |
| **Contrastivo** | `DPOConfig(loss_type=["nca_pair"])` (NCA), `["hinge"]` (SLiC/RSO), `["bco_pair"]`, `CPOTrainer` (CPO), `--train-mode cpo` no MLX | estável/experimental |
| **RM + RL customizado** | `RewardTrainer` → servir o RM (HTTP) → OpenRLHF `--reward.remote_url` com `reinforce`/`reinforce_baseline`/`rloo`/`group_norm`; TRL `RLOOTrainer`/`GRPOTrainer` com `reward_funcs` = modelo de recompensa | A3C/TRPO **não** têm trainer pronto nessas bibliotecas [inferência] |
| **Online DPO / RLAIF** | `trl.experimental.online_dpo.OnlineDPOTrainer` (com vLLM); MLX `--train-mode online_dpo --judge` (aceita `human`) | experimental; XPO e Nash-MD **removidos** do TRL na v1.14.0 |
| **Constitutional AI** | pipeline, não trainer: (1) SFT em revisões auto-críticas guiadas por princípios (gere crítica→revisão com o professor/juiz do Módulo 1); (2) RLAIF: pares julgados por um juiz instruído com a constituição → DPO/RM+RL | — |
| **Self-play** | SPPO via `DPOConfig(loss_type=["sppo_hard"], beta=0.001, learning_rate=5e-7)` (configuração do paper) ou receita SPPO do verl; GRPO é "auto-competição" dentro do grupo | — |
| **Debate** | sem trainer pronto nas bibliotecas verificadas [inferência]; implementável como RL multiagente com juiz (agentes do TRL `GRPOTrainer` com ferramentas/ambientes, ou OpenRLHF `--train.agent_func_path`) | pesquisa |
| GRPO e família (DAPO, Dr. GRPO, GSPO, SAPO, CISPO, VESPO) | TRL `GRPOTrainer` (`loss_type`, `importance_sampling_level`); MLX `--grpo-loss-type`; OpenRLHF `group_norm`/`dr_grpo`; verl `adv_estimator=grpo` | estável |
| Destilação | TRL `DistillationTrainer` (estável desde v1.10.0), `GKDTrainer`, `GOLDTrainer` (experimentais) | — |

---

## Apêndice D — Conceitos necessários (ordem sugerida para os capítulos)

1. **Objetivos de pós-treino e a âncora KL** — max E[r] − β·KL; política ótima π* ∝ π_ref·e^{r/β}; recompensa implícita do DPO.
2. **Avaliação pareada** — win rate, empates, bootstrap, tamanho de amostra; teste pareado.
3. **Bradley–Terry e Elo** — de batalhas a escores; controle de estilo como regressão com covariáveis.
4. **LLM-as-judge** — vieses (posição, verbosidade, autopreferência, raciocínio limitado) e mitigação (troca, rubrica, referência, painel PoLL, calibração com humanos).
5. **Controle de comprimento** — por que respostas longas ganham; AlpacaEval-LC; LN-DPO/SimPO como resposta no treino.
6. **Benchmarks verificáveis e seu ciclo de vida** — saturação (MMLU, GSM8K, HumanEval), contaminação (GSM1k, AIME 2024), substituição (SWE-bench Verified → Pro → Pro Verified; competições ao vivo).
7. **pass@k vs pass^k** — capacidade vs confiabilidade em agentes.
8. **Avaliação de segurança nos dois sentidos** — sobre-recusa (XSTest) vs cumprimento indevido (StrongREJECT/HarmBench); trade-off disposição×capacidade; erosão por fine-tuning; deslocamento de verossimilhança no DPO; direção de recusa.
9. **Regressão de capacidades** — alignment tax; "LoRA aprende menos e esquece menos"; "LoRA sem arrependimento" (todas as camadas, 10× LR, RL com posto 1).
10. **KL como orçamento e diagnóstico** — estimadores k1/k2/k3; Gao et al. (superotimização); sinais de reward hacking por algoritmo.
11. **Três caminhos de serviço** — adapter em runtime (multi-LoRA, slots, Punica/fused-MoE-LoRA), merge BF16, merge + re-quantização.
12. **Escalas do LoRA entre frameworks** — `scale` (MLX) vs `alpha/r` (PEFT); LR efetivo ≈ scale × LR.
13. **Re-quantização calibrada** — por que escalas/Hessianos do base não valem; mistura de calibração; camadas mantidas em alta precisão (lm_head, últimas MLPs, in_proj_a/b, MTP).
14. **Duas KLDs** — deriva do pós-treino vs erro de quantização; piso stock-vs-stock; cota inferior com top-k.
15. **Templates de chat como parte do modelo** — `{% generation %}`, `reasoning_content`, `enable_thinking`, `reasoning_effort`, prefix-preservation no GRPO.
16. **DDP vs FSDP vs papéis assimétricos** em 2 nós com memória unificada.
17. **Ligação com speculative decoding** — MTP/DSpark treinados para o base perdem aceitação após o fine-tune.
18. **Abliteração vs pós-treino** — edição de posto 1 sem gradiente vs otimização; transplante como merge parcial; sonda de recusa.

---

## Apêndice E — Incertezas e pontos a confirmar

1. **Nada foi executado em GPU nesta pesquisa.** Tempos e memórias são estimativas; a âncora de ~60 TFLOPS efetivos vem de um repositório de terceiros (albond, mai/2026) com pilha diferente (transformers 5.5.0, TRL 0.24.0, Unsloth).
2. **fla 0.5.2 + causal-conv1d 1.7.0 + transformers 5.18 + TRL 1.14.1 no sm_121** — combinação não testada; o packing com GDN não foi verificado.
3. **SWE-bench Verified abandonado pela OpenAI (fev/2026)** e números (138 tarefas auditadas, 59,4% com testes defeituosos, 74,9% → 80,9%) vêm de cobertura secundária; a página da OpenAI devolveu 403.
4. **Placares de 2026** (GPQA ~94%, HLE, AIME 2026 ~97%, HMMT) vêm de agregadores; só as listas de benchmarks e a composição do índice AA v4.3.2 foram lidas em fonte primária.
5. **Fórmulas do AlpacaEval-LC e de Gao et al.** transcritas de memória — conferir nos papers antes de publicar equações.
6. **LoRA sobre bases NVFP4/EXL3 no vLLM sm_121** — não verificado; o vLLM só documenta exemplo com bnb/AWQ/GPTQ.
7. **`/tokenize` e inclusão do token real em `prompt_logprobs`** no vLLM 0.30.0 — comportamento conhecido, não reconferido no código dessa tag.
8. **Carga de compressed-tensors NVFP4 no transformers para KL exato** (Apêndice B.4) — modo de dequantização não verificado.
9. **`warmup_steps` float < 1 como fração** no TRL/transformers 5.x — inferido da assinatura `warmup_steps: float`.
10. **ModelOpt:** o card NVIDIA cita v0.48.0, mas o PyPI está em 0.47.0; os caminhos de exemplo/receitas mudaram entre a tag 0.45.0 do playbook (`examples/llm_ptq`, `huggingface/...`) e o main (`examples/hf_ptq`, `model_type/...`, `models/...`).
11. **EXL3 para modelos densos no vLLM** — o overlay da receita GLM foi feito para MoE (+ `non_routed_exl3`); servir um 27B denso em EXL3 no vLLM não foi verificado.
12. **Efeito do fine-tune na aceitação MTP/DSpark** — esperado qualitativamente; magnitude não medida.
13. **Números de "pico" da NVIDIA para fine-tuning no Spark** (82.739 / 53.657 / 5.079 tok/s) — inconsistentes com a conta 6·N·tok/s; definição da métrica não publicada.
14. **`mlx-lm` 0.32.0** saiu hoje (2026-10-01); os padrões foram lidos no `main`, que pode diferir levemente do pacote.
15. **IFBench, BFCL V4, τ²** — usados como referência por cards e páginas oficiais; tamanhos/variantes exatos não reconferidos.

---

## Apêndice F — Fontes

**Pós-treino (bibliotecas)**
- TRL docs (main): https://huggingface.co/docs/trl/main/en/index · SFT https://huggingface.co/docs/trl/main/en/sft_trainer · DPO https://huggingface.co/docs/trl/main/en/dpo_trainer · KTO https://huggingface.co/docs/trl/main/en/kto_trainer · GRPO https://huggingface.co/docs/trl/main/en/grpo_trainer · CPO/SimPO https://huggingface.co/docs/trl/main/en/cpo_trainer · vLLM integration https://huggingface.co/docs/trl/main/en/vllm_integration · CLI https://huggingface.co/docs/trl/main/en/clis · Paper index https://huggingface.co/docs/trl/main/en/paper_index
- TRL releases: https://github.com/huggingface/trl/releases · PyPI https://pypi.org/project/trl/ · TRL v1: https://huggingface.co/blog/trl-v1
- Código TRL lido: `trl/trainer/sft_trainer.py`, `trl/trainer/grpo_config.py`, `trl/generation/vllm_generation.py`, `trl/generation/vllm_client.py`, `trl/rewards/*`, `trl/chat_templates/README.md` (https://github.com/huggingface/trl)
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF · verl: https://github.com/volcengine/verl (exemplo `examples/tuning/lora/run_qwen3_8b_fsdp.sh`, `examples/data_preprocess/gsm8k.py`)
- mlx-lm: https://github.com/ml-explore/mlx-lm (`mlx_lm/LORA.md`, `LEARNED_QUANTS.md`, `lora.py`, `fuse.py`, `tuner/lora.py`, `tuner/utils.py`, `quant/dwq.py`, `evaluate.py`, `examples/lora_config.yaml`) · mlx-lm-lora: https://github.com/Goekdeniz-Guelmez/mlx-lm-lora (`README.md`, `mlx_lm_lora/train.py`)
- Unsloth no Spark (playbook): https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/unsloth · PyTorch fine-tune (playbook, 2 Sparks): https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune · NVFP4 (playbook): https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nvfp4-quantization
- Medições de LoRA no GB10 (terceiros): https://github.com/albond/DGX_Spark_Unsloth_Lossless_Speedup · fórum: https://forums.developer.nvidia.com/t/lossless-7-67x-lora-8-35x-full-ft-speedup-for-qwen3-5-on-dgx-spark-gb10-sm-121a/370517
- Blog NVIDIA (desempenho do Spark, 2025-10-24): https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks
- LoRA Without Regret: https://thinkingmachines.ai/blog/lora/ · LoRA Learns Less and Forgets Less: https://arxiv.org/abs/2405.09673

**Serviço e quantização**
- vLLM LoRA (v0.30.0): https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/lora.md · engine args: https://docs.vllm.ai/en/latest/configuration/engine_args/ · modelos suportados: https://docs.vllm.ai/en/latest/models/supported_models/ · código: `vllm/model_executor/models/qwen3_5.py`, `vllm/entrypoints/openai/completion/protocol.py`, `vllm/entrypoints/launchers/cli_args.py`, `vllm/engine/arg_utils.py`, `examples/features/lora/lora_with_quantization_offline.py` · FusedMoE LoRA: https://github.com/vllm-project/vllm/pull/21229
- llm-compressor: https://github.com/vllm-project/llm-compressor (exemplos `quantization_w4a16/qwen3_8_gptq_awq_example.py`, `quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py`, `custom_dataset_example.py`)
- NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer (`examples/hf_ptq/hf_ptq.py`, `modelopt/torch/utils/dataset_utils.py`, `modelopt_recipes/models/Qwen/Qwen3.8-27B/ptq/*.yaml`)
- exllamav3: https://github.com/turboderp-org/exllamav3 (`doc/convert.md`, `doc/optimize.md`, `eval/model_diff.py`)
- Modelos: https://huggingface.co/Qwen/Qwen3.8-27B · https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 · https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4 · https://huggingface.co/Qwen/Qwen3.5-9B · https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 · https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit
- Fórum NVIDIA (Qwen3.8-27B NVFP4 vs FP8 no Spark): https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258
- NGC PyTorch 26.08: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-08.html · NGC vLLM 26.08: https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-08.html

**Avaliação**
- MT-Bench/Chatbot Arena (Zheng et al. 2023): https://arxiv.org/abs/2306.05685 · Chatbot Arena (Chiang et al. 2024): https://arxiv.org/abs/2403.04132 · Changelog do Arena: https://arena.ai/blog/leaderboard-changelog · Arena-Rank: https://arena.ai/blog/arena-rank · Leaderboard Illusion: https://arxiv.org/abs/2504.20879
- LLMs não são avaliadores justos (posição): https://arxiv.org/abs/2305.17926 · Autopreferência: https://arxiv.org/abs/2404.13076 · PoLL: https://arxiv.org/abs/2404.18796
- AlpacaEval: https://github.com/tatsu-lab/alpaca_eval · LC AlpacaEval: https://arxiv.org/abs/2404.04475 · Arena-Hard/BenchBuilder: https://arxiv.org/abs/2406.11939 · https://github.com/lmarena/arena-hard-auto
- IFEval: https://arxiv.org/abs/2311.07911 · GSM1k: https://arxiv.org/abs/2405.00332 · MathArena: https://arxiv.org/abs/2505.23281 · Min-K% Prob: https://arxiv.org/abs/2310.16789
- SWE-Bench Pro Verified: https://arxiv.org/abs/2609.08149 · OpenAI sobre SWE-bench Verified (403 nesta pesquisa): https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ · cobertura: https://decrypt.co/359012/openai-benchmark-measure-ai-coding-supremacy-contaminated
- Terminal-Bench: https://www.tbench.ai/ · BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html · τ²-bench: https://github.com/sierra-research/tau2-bench · ARC-AGI-3: https://arcprize.org/arc-agi/3 · Índice Artificial Analysis: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- XSTest: https://arxiv.org/abs/2308.01263 · HarmBench: https://arxiv.org/abs/2402.04249 · StrongREJECT: https://arxiv.org/abs/2402.10260 · Fine-tuning compromete segurança: https://arxiv.org/abs/2310.03693 · Likelihood displacement: https://arxiv.org/abs/2410.08847 · Direção de recusa: https://arxiv.org/abs/2406.11717
- Superotimização de RM: https://arxiv.org/abs/2210.10760 · DPO: https://arxiv.org/abs/2305.18290 · IPO: https://arxiv.org/abs/2310.12036 · KTO: https://arxiv.org/abs/2402.01306 · SimPO: https://arxiv.org/abs/2405.14734
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness (`docs/API_guide.md`, `lm_eval/models/vllm_causallms.py`, tasks `ifeval`, `gsm8k`, `aime`, `humaneval`)

**Notas internas usadas (Módulo 1)**: `docs/pesquisa/quant.md` (§4.2 KLD; C45), `glm-mods.md` (M47 abliteração, M48 dense overlay, M50 canário, §4.1 hardware), `glm-launch.md` (rede/NCCL/contêiner), `vllm-ref.md` (versões, LoRA no prefix cache, BatchDescriptor, §12.2 EXL3), `qwen-launch.md` (MTP), `mlx-copilot.md` (mlx_lm.server, memória do Mac).

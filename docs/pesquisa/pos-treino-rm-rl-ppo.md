# Pesquisa — Módulo 2 (Pós-treino): dados de preferência, modelos de recompensa, fundamentos de RL e RLHF com PPO

> **Escopo.** Nota de pesquisa para os capítulos do Módulo 2 que cobrem (1) dados de preferência humana, (2) modelagem de recompensa (Bradley–Terry, Plackett–Luce, RM, calibração, viés de comprimento, reward hacking/superotimização, ensembles, ORM vs PRM, RMs generativos/LLM-as-judge, RLAIF), (3) fundamentos de RL para LLMs (MDP em nível de token, teorema do gradiente de política, REINFORCE, baselines, vantagem, função valor, TD, GAE, on/off-policy, importance sampling, A2C/A3C, TRPO, PPO) e (4) RLHF com PPO de ponta a ponta (pipeline InstructGPT, recompensa moldada por KL, os quatro modelos em memória, o motor de inferência dentro do laço, custo, instabilidades e truques, PPO-ptx).
>
> **Data de referência:** 2026-10-01. Os papers foram baixados do arXiv e lidos no texto integral (PDF → texto); documentação de vLLM/SGLang/Hugging Face foi consultada na versão `latest` nesta data. Lista completa no Apêndice B.
>
> **Convenções de marcação:**
> - **[P]** = afirmação textual de paper primário (com seção/tabela/equação quando possível).
> - **[D]** = documentação oficial (vLLM, SGLang, cards de dataset no Hugging Face, README de repositório).
> - **[W]** = fonte web secundária (blog técnico, workshop, notícia) — usar com cautela.
> - **(interpretação)** = dedução minha a partir das fontes; **(cálculo)** = conta feita por mim a partir das fórmulas, reprodutível (o script está descrito no Apêndice A).
> - **[M1: arquivo.md]** = ponte com as notas do Módulo 1 em `docs/pesquisa/`.
> - Nenhum número foi inventado: os exemplos numéricos "de brinquedo" do Apêndice A usam valores escolhidos por mim e estão rotulados como **ilustrativos**.
>
> **Notação usada em toda a nota** (as equações seguem a convenção KaTeX da plataforma, `\( \)` e `\[ \]`, prontas para colar):
>
> | Símbolo | Significado |
> |---|---|
> | \(x\) | prompt (contexto/instrução) |
> | \(y = (y_1,\dots,y_T)\) | resposta (sequência de tokens), \(T\) = comprimento |
> | \(y_w, y_l\) | resposta preferida ("chosen", *winner*) e rejeitada ("rejected", *loser*) |
> | \(r_\phi(x,y)\) ou \(r_\theta(x,y)\) | modelo de recompensa (RM) escalar |
> | \(\pi_\theta\) | política = o LLM sendo treinado (ator) |
> | \(\pi_{\text{ref}}\), \(\pi^{\text{SFT}}\) | política de referência congelada (em geral o modelo SFT) |
> | \(\pi_{\theta_{\text{old}}}\) | política que gerou o lote (antes das atualizações desta iteração) |
> | \(V_\psi(s)\) | função valor (crítico) |
> | \(s_t = (x, y_{<t})\), \(a_t = y_t\) | estado e ação no MDP de tokens |
> | \(\gamma\), \(\lambda\) | fator de desconto e parâmetro do GAE |
> | \(\beta\) | coeficiente da penalidade KL |
> | \(\varepsilon\) | raio de clipping do PPO |
> | \(\sigma(z) = 1/(1+e^{-z})\) | sigmoide logística |

---

## 0. Mapa rápido

### 0.1 Tabela-resumo

| Tema | Ideia central | Equação-chave | Paper primário | Ponte com o Módulo 1 |
|---|---|---|---|---|
| Dados de preferência | Humanos comparam melhor do que pontuam | pares \((x, y_w, y_l)\); rankings de \(K\) respostas | Christiano 2017; Stiennon 2020; Ouyang 2022 | — |
| Bradley–Terry | Preferência = logística da diferença de recompensas | \(P(y_w \succ y_l) = \sigma(r_w - r_l)\) | Bradley & Terry 1952 | sigmoide/softmax (cap. `geracao-amostragem`) |
| Perda do RM | Máxima verossimilhança do BT | \(\mathcal L = -\log\sigma(r_w - r_l)\) | Stiennon 2020; Ouyang 2022 | entropia cruzada (cap. `aprender-com-dados`) |
| Plackett–Luce | BT generalizado para rankings | \(\prod_i e^{r_{\tau(i)}}/\sum_{j\ge i} e^{r_{\tau(j)}}\) | Plackett 1975; Luce 1959 | softmax |
| Superotimização | Proxy otimizado demais piora a verdade | \(R_{\text{RL}}(d) = d(\alpha - \beta\log d)\) | Gao, Schulman, Hilton 2022 | KLD como métrica (cap. `medindo-qualidade`) |
| ORM vs PRM | Recompensa por resultado vs por passo | PRM: \(\prod_k p(\text{passo}_k\ \text{correto})\) | Uesato 2022; Lightman 2023 | raciocínio/`<think>` (cap. `geracao-amostragem`) |
| MDP de tokens | Estado = prefixo, ação = próximo token | \(s_{t+1} = s_t \oplus a_t\) | Ziegler 2019; Ouyang 2022 | decode autoregressivo (cap. `prefill-decode-kv`) |
| Gradiente de política | Subir \(\log\pi\) das ações com vantagem positiva | \(\nabla J = \mathbb E[\sum_t \nabla\log\pi_\theta(a_t \mid s_t)\,\hat A_t]\) | Williams 1992; Sutton et al. 2000 | backprop (cap. `redes-neurais`) |
| GAE | Média exponencial de resíduos TD | \(\hat A_t = \sum_l (\gamma\lambda)^l\delta_{t+l}\) | Schulman et al. 2015b (GAE) | — |
| TRPO | Passo máximo sob restrição de KL | \(\max L\ \text{s.a.}\ \bar D_{KL}\le\delta\) | Schulman et al. 2015a | — |
| PPO | Clipping da razão de probabilidades | \(\min(\rho_t\hat A_t, \text{clip}(\rho_t,1\pm\varepsilon)\hat A_t)\) | Schulman et al. 2017 | — |
| RLHF-PPO | Recompensa do RM menos KL até a referência | \(R = r_\phi - \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}\) | Stiennon 2020; Ouyang 2022 | vLLM no laço de rollout (caps. `vllm-arquitetura`, `scheduler-batching`) |
| Descompasso treino–inferência | Mesmos pesos, logprobs diferentes no vLLM e no FSDP | TIS: \(\min(\pi_{\text{fsdp}}/\pi_{\text{vllm}}, C)\) | Yao et al. 2025; MiniMax-M1 2025 | quantização FP8/INT8, determinismo, MoE (caps. `quantizacao-fundamentos`, `kernels`, `mixture-of-experts`) |

### 0.2 Linha do tempo (o que veio de onde)

| Ano | Marco | Por que importa |
|---|---|---|
| 1952 | Bradley & Terry — comparações pareadas | modelo estatístico de toda a modelagem de recompensa moderna |
| 1959 / 1975 | Luce (axioma da escolha) / Plackett (permutações) | generalização para rankings de \(K\) itens |
| 1992 | Williams — REINFORCE | estimador de gradiente de política por razão de verossimilhança |
| 2000 | Sutton, McAllester, Singh, Mansour — teorema do gradiente de política | base teórica de todo método de política |
| 2002 | Kakade & Langford — lema de diferença de desempenho; Kakade 2001 — gradiente natural | base teórica do TRPO |
| 2015 | Schulman et al. — TRPO (fev) e GAE (jun) | região de confiança e estimador de vantagem |
| 2016 | Mnih et al. — A3C | atores assíncronos paralelos |
| 2017 | Schulman et al. — PPO (jul); Christiano et al. — RL a partir de preferências humanas (jun) | o algoritmo e a ideia do RLHF |
| 2019 | Ziegler et al. — fine-tuning de LMs (GPT-2) com preferências | primeiro RLHF em LM; controlador adaptativo de KL |
| 2020 | Stiennon et al. — "Learning to summarize from human feedback" | receita SFT → RM → PPO com KL, crítico separado |
| 2022 | Ouyang et al. — InstructGPT (mar); Bai et al. — HH-RLHF (abr); Gao et al. — leis de escala da superotimização (out); Uesato et al. — processo vs resultado (nov); Bai et al. — Constitutional AI (dez) | RLHF vira padrão industrial; medição de Goodhart; RLAIF |
| 2023 | Lightman et al. — PRM800K (mai); Rafailov et al. — DPO (mai); Llama 2 (jul); Zheng et al. "Secrets of RLHF" (jul); Lee et al. — RLAIF vs RLHF (set); UltraFeedback (out); Singhal et al. — viés de comprimento (out); Coste et al. — ensembles (out); HelpSteer (nov); Math-Shepherd (dez) | escala, dados abertos, críticas e alternativas |
| 2024 | WARM (jan); DeepSeekMath/GRPO (fev); RLOO "Back to Basics" (fev); RewardBench (mar); "N+ implementation details" (mar); OpenRLHF (mai); HelpSteer2 (jun); GenRM (ago); HybridFlow/verl (set); HelpSteer2-Preference (out); RLHF assíncrono (out) | sistemas de RL com vLLM no laço; abandono parcial do crítico |
| 2025 | DeepSeek-R1 (jan); VC-PPO (mar); VAPO (abr); DeepSeek-GRM (abr); HelpSteer3 (mai); RewardBench 2 (jun); MiniMax-M1 (jun); Skywork-Reward-V2 (jul); "Defeating Nondeterminism" (set); R3 — routing replay p/ MoE (out); TIS (Yao et al., workshop OPT 2025) | PPO com crítico revive para CoT longo; descompasso treino–inferência vira tema central |
| 2026 | vLLM: API de transferência de pesos com backends plugáveis [D]; `vime` (slime + vLLM, jun) [W]; slime declara ser o framework de RL por trás do GLM-5.3 [D] | infraestrutura padronizada de RL com motor de inferência |

### 0.3 Proposta de capítulos (sugestão ao orquestrador)

Ordem pedagógica sugerida para estes tópicos (prereqs entre parênteses). Cada um pede figuras animadas cujos dados estão no Apêndice A.

1. **"Preferências humanas como dado"** — formatos (par, ranking, Likert, binário, por passo), diretrizes, concordância (κ), datasets. (prereq: `aprender-com-dados`). Animações: A.1, A.5.
2. **"Bradley–Terry e o modelo de recompensa"** — derivação BT/Thurstone/Elo, Plackett–Luce, perda \(-\log\sigma\), arquitetura LM+cabeça escalar, treino, calibração. (prereq: cap. 1 do M2; `geracao-amostragem` p/ softmax). Animações: A.1–A.4, A.6–A.8.
3. **"Quando a recompensa mente"** — viés de comprimento, reward hacking, superotimização (Gao), ensembles/WARM, KL como freio. (prereq: cap. 2). Animações: A.9, A.10.
4. **"Recompensas de processo, juízes e RLAIF"** — ORM/PRM, Math-Shepherd, LLM-as-judge e seus vieses, GenRM, RLAIF/CAI. (prereq: cap. 2). Animações: A.24, A.25.
5. **"RL para quem vem de LLMs"** — MDP de tokens, retorno, \(V, Q, A\), teorema do gradiente de política, REINFORCE, baselines, RLOO. (prereq: `redes-neurais`). Animações: A.16, A.22.
6. **"TD, GAE e o crítico"** — Bellman, TD(0), n-passos, λ-retorno, GAE, decaimento do sinal em CoT longo. (prereq: cap. 5). Animações: A.12, A.21.
7. **"De A3C e TRPO a PPO"** — atores assíncronos, região de confiança, gradiente natural, gradiente conjugado, clipping, perda de valor, entropia. (prereq: cap. 6). Animações: A.13, A.17, A.23.
8. **"RLHF com PPO de ponta a ponta"** — InstructGPT, recompensa moldada por KL e política ótima, os quatro modelos, laço passo a passo, truques e instabilidades, PPO-ptx, custo. (prereq: caps. 2, 3, 7). Animações: A.11, A.14, A.15, A.19, A.20.
9. **"O motor de inferência dentro do laço"** — vLLM/SGLang como gerador de rollouts, sincronização de pesos, colocado vs desagregado, RL assíncrono, descompasso treino–inferência (TIS, LM head FP32, kernels batch-invariantes, R3 para MoE). (prereq: cap. 8 + Módulo 1 caps. `vllm-arquitetura`, `kernels`, `quantizacao-fundamentos`, `nccl-rdma`, `mixture-of-experts`). Animação: A.18.

---

## 1. Dados de preferência humana

### 1.1 Por que comparar em vez de pontuar

- **Christiano et al. 2017** [P, §2.2.2 e §3.3]: humanos veem dois clipes de 1–2 s e indicam qual preferem, se são equivalentes, ou se não conseguem comparar. Motivação explícita: *"we found it much easier for humans to provide consistent comparisons than consistent absolute scores"*. Em controle contínuo, prever comparações funcionou "muito melhor" que prever escores. Respondiam em 3–5 s por consulta; experimentos com humanos reais custaram entre 30 min e 5 h de tempo humano; feedback em menos de 1% das interações do agente.
- **Ziegler et al. 2019** [P, nota 1]: "nos primeiros experimentos foi difícil para humanos dar distinções quantitativas finas e consistentes quando pedidos um número absoluto; experimentos sintéticos confirmaram que comparações eram quase tão úteis".
- **Stiennon et al. 2020** [P, nota 6]: "nossa decisão de coletar comparações em vez de escores Likert é apoiada por trabalhos recentes".
- **Contraponto — HelpSteer (Wang et al. 2023)** [P, §3.1]: cada resposta foi avaliada *independentemente* em Likert-5 (0–4), porque comparações têm "crescimento quadrático" do número de anotações com o número de respostas por prompt; por isso o rating absoluto é "mais escalável".
- **Síntese (interpretação):** comparações reduzem a variância entre anotadores (não exigem uma régua absoluta compartilhada) e casam diretamente com o modelo BT; ratings absolutos custam menos por resposta e carregam informação de magnitude, mas sofrem com calibração individual do anotador. Datasets modernos combinam os dois (HelpSteer2 + HelpSteer2-Preference; UltraFeedback dá escores 1–5 que depois são binarizados em pares).

### 1.2 Formatos de anotação (e o modelo estatístico que cada um alimenta)

| Formato | Exemplo real | Informação por anotação | Modelo/perda natural |
|---|---|---|---|
| **Par binário** \((x, y_w, y_l)\) | Stiennon 2020 (TL;DR); HH-RLHF (Bai 2022) | 1 bit de direção | Bradley–Terry, \(-\log\sigma(\Delta)\) |
| **Par com empate / incomparável** | Christiano 2017: empate → \(\mu\) uniforme; incomparável → descartado [P §2.2.2] | direção ou "igual" | BT com rótulo suave (entropia cruzada contra \(\mu\)); extensões Rao–Kupper/Davidson |
| **Par com intensidade** | Llama 2: *significantly better / better / slightly better / negligibly better-unsure* [P §3.2.1]; HelpSteer2-Preference e HelpSteer3: escala −3…+3 [P/D] | direção + magnitude | BT com margem (Llama 2) ou BT escalado (HelpSteer2-Pref) |
| **Ranking de K respostas** | InstructGPT: \(K = 4\) a \(9\) respostas por prompt → \(\binom{K}{2}\) pares [P §3.5] | ordem total | pares decompostos (InstructGPT) ou Plackett–Luce |
| **Melhor de K** | Ziegler 2019: escolher a melhor de 4 [P §2] | índice do vencedor | softmax top-1 (PL de 1 estágio) |
| **Likert absoluto, multi-atributo** | HelpSteer: helpfulness, correctness, coherence, complexity, verbosity em 0–4 [P]; InstructGPT: qualidade geral Likert 1–7 + metadados binários [P Tab. 3] | escore por atributo | regressão (MSE) — "SteerLM Regression" |
| **Escores por aspecto + crítica textual (IA)** | UltraFeedback: GPT-4 dá 1–5 em instruction-following, truthfulness, honesty, helpfulness + justificativa + crítica [P §2.4] | 4 escores + texto | binarização → BT; ou crítico generativo |
| **Binário bom/ruim (sem par)** | (tema do capítulo KTO) | 1 bit absoluto | KTO e afins — fora do escopo desta nota |
| **Rótulo por passo** | PRM800K: cada passo de solução matemática rotulado positivo/negativo/neutro [P §2.4] | 1 rótulo por passo | PRM (classificador por passo) |
| **Justificativa escrita** | HelpSteer2-Preference: justificativas humanas por preferência [P] | texto | "pairwise justifier" / RM generativo |

### 1.3 Como os pares são produzidos

- **Fontes de prompt.** InstructGPT [P §3.2]: prompts da API (filtrados de PII, deduplicados por prefixo comum longo, no máx. 200 prompts por ID de usuário; splits por ID de usuário) + prompts escritos por rotuladores (tipos *plain*, *few-shot*, *user-based*) para "dar a partida". Distribuição de casos de uso na API [P Tab. 1]: Generation 45,6%, Open QA 12,4%, Brainstorming 11,2%, Chat 8,4%, Rewrite 6,6%, … HH-RLHF [P §2]: diálogos abertos com o modelo; no conjunto de *harmlessness* o anotador escolhe a resposta **mais nociva** (red teaming) — o que os autores mesmo notam ser "provavelmente não ideal para ensinar bom comportamento".
- **Fontes de resposta.** Para cobrir a distribuição, amostra-se de modelos/temperaturas diferentes: Llama 2 amostra as duas respostas de "duas variantes de modelo diferentes, variando a temperatura" [P §3.2.1]; UltraFeedback sorteia 4 de um pool de 17 modelos e ainda injeta "princípios" diferentes no prompt para diversificar estilos [P §2.3]; HelpSteer2 gera 2 respostas por prompt (não 4 como HelpSteer) "para reduzir a carga cognitiva" [P §2.1].
- **On-policy vs off-policy (dados).** InstructGPT: "na prática, a maior parte dos nossos dados de comparação vem das políticas supervisionadas, com alguns vindo das políticas PPO" [P §3.1]. Coletar comparações de amostras da política atual mantém o RM "dentro da distribuição". Bai 2022 coletou três tranches: base (modelo de *context distillation*), *rejection sampling* contra PMs iniciais, e **"online" iterado**, com PMs e políticas RLHF atualizados "numa cadência aproximadamente semanal" [P §2.3]. Llama 2 enfatiza o mesmo: "a acumulação de dados iterativos de RM em paralelo com as melhorias do modelo é crucial para manter os RMs dentro da distribuição" [P §1]. Christiano 2017 coleta continuamente, em paralelo com o RL (três processos assíncronos: política, consultas a humanos, ajuste do RM) [P §2.2].
- **Seleção ativa de consultas.** Christiano 2017: amostrar muitos pares e consultar aqueles com **maior variância de predição entre os membros do ensemble** [P §2.2.4] (os próprios autores notam que em algumas tarefas isso piorou). Lightman 2023: *active learning* de soluções "convincentes porém erradas" deu **2,6×** de eficiência de dados ao PRM [P §1].

### 1.4 Diretrizes de anotação e gestão de anotadores

- **InstructGPT** [P §3.4, Apêndice B]: ~40 contratados (Upwork e ScaleAI) selecionados por um teste de triagem (concordância na sinalização de conteúdo sensível, concordância em rankings com pesquisadores, escrita de demonstrações sensíveis — com corte "suave" de 75% de concordância). Critérios: *helpful, truthful, harmless*. **No treino, rotuladores priorizavam helpfulness**; na avaliação final, **truthfulness e harmlessness**. Onboarding, instruções detalhadas por tarefa, sala de chat compartilhada.
- **Stiennon 2020** [P §3.3, Ap. C]: relação "hands-on" com rotuladores — onboarding, chat compartilhado, feedback regular de desempenho, monitoramento contínuo da concordância rotulador–pesquisador.
- **HH-RLHF** [P §2.1]: workers MTurk "master-qualified" dos EUA; ~20 trabalhadores mais prolíficos fizeram ~80% dos dados; avaliados pela sofisticação dos diálogos, **não** por concordância; também Upwork (pagamento por hora favorece qualidade; MTurk produz mais rápido).
- **HelpSteer** [P §3.1]: ~200 anotadores nos EUA, teste de proficiência em inglês, curso introdutório com teste de 35 respostas; guias com critérios por nível da escala e exemplos.
- **HelpSteer2** [P §2.1]: ≥3 anotadores por resposta; se a diferença de helpfulness entre eles for >2, mais dois são chamados (média 3,41 anotadores/amostra); ~50% das anotações foram descartadas no controle de qualidade; anotadores sem expertise em código → prompts de código filtrados por heurística; prompts não-ingleses removidos (FastText); prompts amostrados uniformemente de ~1000 tópicos (BERTopic) e por nível de complexidade.
- **Llama 2** [P §3.2.1]: guias **separados** para helpfulness e safety (porque se contradizem); intensidade da preferência em 4 níveis.

### 1.5 Concordância entre anotadores (e por que ela é o teto do RM)

**Números medidos:**

| Fonte | Medida | Valor |
|---|---|---|
| InstructGPT [P §3.4] | concordância entre rotuladores de treino / entre rotuladores held-out | **72,6 ± 1,5%** / **77,3 ± 1,3%** |
| Stiennon 2020 [P §3.3] | rotulador–pesquisador / pesquisador–pesquisador | **77 ± 2%** / **73 ± 4%** |
| Stiennon 2020 [P §4.3] | inter-rotulador em CNN/DM | **66,9%** (RM 6,7B alcança 66,5%) |
| HH-RLHF [P §2.1] | pesquisadores Anthropic vs crowdworkers | **≈63%** ("poor") |
| HelpSteer2 [P Tab. 1] | κ de Cohen ponderado quadrático (helpfulness) | **0,465** (inicial) → **0,791** (após pós-processamento) |
| HelpSteer2-Preference [P §2] | κ quadrático das preferências brutas −3…+3 | **0,489** ("moderado") |
| MT-bench (Zheng 2023) [P §4.2] | humano–humano vs GPT-4–humano (setup sem empate) | **81%** vs **85%** |
| N+ details (Huang 2024) [P Detalhe 16] | acurácia do RM 1B por lote de anotação do TL;DR | de **0,508** (lote 11, "cara ou coroa") a **0,771** (lote 13) |
| RewardBench [P §1] | teto de acurácia em validação de HH e Summarize | "entre **60 e 70%**, devido à discordância entre anotadores" |

**Métricas:**

- **Concordância percentual** \(p_o\): fração de itens com o mesmo rótulo. Não desconta acaso.
- **κ de Cohen** (dois anotadores):
  \[ \kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = \sum_k p^{(A)}_k\, p^{(B)}_k \]
  - \(p_o\): concordância observada; \(p_e\): concordância esperada por acaso dadas as marginais; \(p^{(A)}_k\): fração de itens que o anotador A rotulou como classe \(k\).
- **κ ponderado (quadrático)** — para escalas ordinais (Likert, −3…+3), penaliza discordâncias grandes mais que pequenas:
  \[ \kappa_w = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \qquad w_{ij} = \frac{(i-j)^2}{(k-1)^2} \]
  - \(O_{ij}\): proporção observada de itens com rótulo \(i\) pelo anotador A e \(j\) pelo B; \(E_{ij}\): produto das marginais; \(k\): número de níveis. HelpSteer2 escolhe esse κ porque "discordâncias entre 0 e 4 devem ser penalizadas muito mais que entre 0 e 1" e porque, com ~1000 anotadores, raramente os mesmos anotadores compartilham amostras (inviabilizando Fleiss κ / α de Krippendorff) [P §2.1].
- **Fleiss κ** (vários anotadores, rótulos nominais) e **α de Krippendorff** (vários anotadores, dados faltantes, qualquer escala) — citados como alternativas por HelpSteer2.
- **Consequência prática:** se humanos concordam ~73% entre si, um RM perfeito para "o humano médio" não pode passar muito disso na validação humana; acurácia de RM de 65–70% em HH/TL;DR **não** indica um RM ruim (interpretação apoiada pelo RewardBench e por Stiennon, cujo RM 6,7B "quase iguala" a concordância inter-rotulador).
- **Ruído explícito no modelo:** Christiano 2017 supõe "10% de chance de o humano responder uniformemente ao acaso" [P §2.2.3] → \(\hat P = 0{,}9\,\sigma(\Delta) + 0{,}05\) (Apêndice A.6), porque a taxa de erro humano "não decai a 0 quando a diferença de recompensa fica extrema".

### 1.6 Datasets de preferência (números verificados)

| Dataset | Origem / anotação | Tamanho | Formato | Licença / notas |
|---|---|---|---|---|
| **Anthropic HH-RLHF** (Bai et al. 2022) | crowdworkers; helpful-base, harmless-base, helpful-online, helpful-rejection-sampled, red-team-attempts | **169.352** linhas (≈161k treino / 8,55k teste) [D card]; base: 44k comparações de helpfulness + 42k de red teaming; RS: 52k + 2k [P §2.3] | `chosen`/`rejected` (diálogos completos) | MIT [D] |
| **OpenAI summarize_from_feedback** (Stiennon 2020) | rotuladores contratados, TL;DR + CNN/DM | Stiennon: "64.832 comparações de resumos no TL;DR" [P §1]; a reprodução N+ treina o RM em **92.858** pares (1 época) [P Detalhe 10]; UltraFeedback lista 92.858 pares [P Tab. 1]; Llama 2 lista 176.625 [P Tab. 6] — ver Apêndice C | par + confiança (1–9) + eixos Likert | — |
| **OpenAI WebGPT comparisons** | rotuladores | 19.578 pares [UltraFeedback Tab. 1] / 13.333 [Llama 2 Tab. 6] | par com escores | divergência de contagem (filtragem diferente) |
| **InstructGPT** (não público) | 40 contratados | RM: 33.207 prompts de treino (6.623 de rotuladores + 26.584 da API), 17.887 de validação; cada prompt com \(K=4..9\) respostas → ordem de magnitude mais pares que prompts [P Tab. 6, 9] | ranking | proprietário |
| **Stanford SHP** | votos do Reddit (preferência derivada de score/tempo) | **≈386k** linhas (349k/18,4k/18,4k), 18 subreddits [D card]; Llama 2 usou 74.882 [P Tab. 6] | par | — |
| **StackExchange** (usado por Llama 2) | votos | 1.038.480 comparações [P Tab. 6] | par | — |
| **OASST1** | voluntários (Open Assistant) | 35.905 conversas, 17.966 pares [UltraFeedback Tab. 1] | ranking + atributos Likert (qualidade, criatividade, humor) | Apache 2.0 [HelpSteer2 §1] |
| **Llama 2 (Meta)** | anotadores contratados; helpfulness e safety separados; 4 níveis de intensidade | 1.418.091 comparações próprias; total com abertos 2.919.326 [P Tab. 6] | par + intensidade | proprietário |
| **UltraFeedback** (Cui et al. 2023) | **IA (GPT-4)**: 1–5 em 4 aspectos + justificativa + crítica | 63.967 instruções × 4 respostas = **255.864** completions; **340.025** pares; >1M feedbacks [P §2, Tab. 1] | escores → pares | versão binarizada `HuggingFaceH4/ultrafeedback_binarized`: **61.135** `train_prefs` / **2.000** `test_prefs`; *chosen* = maior `overall_score`, *rejected* = uma das 3 restantes ao acaso [D card] |
| **HelpSteer** (Wang et al. 2023) | ~200 anotadores; 5 atributos 0–4 | **37.120** amostras (10.459 prompts single-turn) [P §3.1] | Likert multi-atributo | CC-BY-4.0. Correlação de Pearson com helpfulness: correctness 0,8525, coherence 0,6348, complexity 0,2361, verbosity 0,2555 [P §3.2] |
| **HelpSteer2** (Wang et al. 2024) | ≥3 anotadores (média 3,41) | **21.362** amostras = 10.681 prompts × 2 respostas (95/5% treino/val) [P §2.1] | Likert multi-atributo | CC-BY-4.0; RM Nemotron-4 340B → 92,0% no RewardBench (1º em 12/06/2024) [P] |
| **HelpSteer2-Preference** (Wang et al. 2024) | preferência −3…+3 + justificativa | complementa os mesmos prompts [P §2] | par com intensidade + texto | CC-BY-4.0; Llama-3.1-70B com BT escalado → **94,1** no RewardBench (1º em 01/10/2024) [P] |
| **HelpSteer3** (2025) | 3–5 anotadores, liberam as 3 mais concordantes; domínios General, STEM, Code, Multilingual | Preference: **40.476**; Feedback: 40.821; Edit: 14.461; Edit-Quality: 3.274; Principle: 33.905 [D card] | −3…+3 | CC-BY-4.0; papers arXiv 2505.11475, 2503.04378, 2509.21319 [D] |
| **PRM800K** (Lightman et al. 2023) | humanos rotulam cada passo | **800K** rótulos de passo em **75K** soluções para **12K** problemas do MATH [P §2.4] | positivo/negativo/neutro por passo | MIT (repositório OpenAI) |
| **Chatbot Arena / MT-bench** (Zheng et al. 2023) | votos de usuários e especialistas | 3K votos de especialistas + 30K conversas com preferências humanas [P Abstract] | par (A/B/empate) | — |
| **Skywork SynPref-40M** (2025) | pipeline humano + IA de curadoria | **40M** pares (26M curados) → Skywork-Reward-V2 (8 RMs de 0,6B a 8B) [W/arXiv 2507.01352] | par | — |

**Benchmarks de RM (para os capítulos):**
- **RewardBench** (Lambert et al. 2024) [P Tab. 1]: trios prompt–chosen–rejected verificados; núcleo de **2.985** pares = Chat 358 + Chat Hard 456 + Safety 740 + Reasoning 1.431, mais "Prior Sets" (17,2k: Anthropic Helpful 6.192, HHH 221, SHP 1.741, Summarize 9.000).
- **RewardBench 2** (AllenAI, 2025, arXiv 2506.01937) [W/arXiv]: formato **best-of-4** (1 correta, 3 incorretas; empates contam parcialmente), 6 domínios (Factuality, Precise IF, Math, Safety, Focus, Ties); modelos pontuam ~20 pontos a menos que no RewardBench; os autores correlacionam o escore com uso a jusante (best-of-N e PPO).

### 1.7 Vieses conhecidos nos dados

- **Comprimento/verbosidade.** RMs aprendem que "mais longo = melhor" porque os dados contêm essa correlação (Singhal et al. 2023: os RMs "são facilmente influenciados por vieses de comprimento nos dados de preferência" [P Abstract]). Detalhes em §2.7.
- **Posição** (em rotuladores IA): LLMs favorecem a primeira ou a segunda resposta; RLAIF mitiga fazendo duas inferências com ordem trocada e tirando a média [P Lee 2023 §2.1.1]; MT-bench mede a consistência sob troca de posição [P Zheng 2023 Tab. 2].
- **Auto-favorecimento** (LLM-juiz preferindo as próprias respostas) e **verbosidade** — Zheng 2023 [P §3.3].
- **Distribuição de rotuladores**: preferências do "rotulador médio" contratado ≠ usuários reais; InstructGPT discute explicitamente a quem o modelo está sendo alinhado [P §5.2].
- **Mudança de distribuição**: pares coletados do SFT inicial deixam de cobrir as saídas da política após RL → motivação para coleta online (§1.3) e para a penalidade KL (§4.2).

---

## 2. Modelagem de recompensa

### 2.1 Bradley–Terry: derivação

**Modelo de utilidade latente.** Suponha que cada resposta tenha utilidade \(u(x,y) = r(x,y) + \epsilon_y\), com ruído i.i.d. O humano prefere \(y_1\) se \(u_1 > u_2\).

- Se \(\epsilon\) ~ Gumbel(0,1) i.i.d., a diferença de dois Gumbels é **logística**, e
  \[ P(y_1 \succ y_2 \mid x) = P(\epsilon_2 - \epsilon_1 < r_1 - r_2) = \sigma\big(r(x,y_1) - r(x,y_2)\big) = \frac{e^{r_1}}{e^{r_1} + e^{r_2}} \]
  Esta é a forma de **Bradley & Terry (1952)** com "forças" \(p_i = e^{r_i}\): \(P(i \succ j) = p_i/(p_i+p_j)\).
- Se \(\epsilon\) ~ Normal, obtém-se o modelo **probit de Thurstone (1927)**: \(P = \Phi\big((r_1 - r_2)/\sqrt2\,s\big)\). Na prática a logística domina por ter gradiente simples e casar com a softmax dos LMs.
- Christiano 2017 escreve a mesma coisa somando recompensas por passo sobre um clipe e nota que é "a especialização da regra de escolha de Luce–Shepard" e "análoga ao Elo do xadrez" [P §2.2.3].

**Elo como BT em outra base.** Elo: \(P(A \succ B) = 1/(1 + 10^{-(R_A - R_B)/400})\). Igualando a \(\sigma(\Delta r)\): \(\Delta r = \frac{\ln 10}{400}\,\Delta R_{\text{Elo}} \approx 0{,}005756\,\Delta R_{\text{Elo}}\) (cálculo). Ex.: +400 Elo ⇔ \(\Delta r = 2{,}3026\) ⇔ 90,91% de vitória (A.2). Chatbot Arena e Bai 2022 reportam modelos em Elo exatamente por isso.

**Máxima verossimilhança → perda do RM.** Para um dataset \(\mathcal D\) de pares, a log-verossimilhança negativa é

\[ \mathcal L_{\text{RM}}(\phi) = -\,\mathbb E_{(x, y_w, y_l)\sim\mathcal D}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big] \]

- \(r_\phi\): RM com parâmetros \(\phi\); \(y_w\)/\(y_l\): preferida/rejeitada; \(\sigma\): sigmoide.
- É exatamente a **entropia cruzada binária** com rótulo 1 para "\(y_w\) vence", sobre o logit \(\Delta = r_w - r_l\). Stiennon: "a diferença de recompensas representa o log-odds de que uma resposta seja preferida" [P InstructGPT §3.5, resumindo Stiennon].

**Gradiente.** Com \(\Delta = r_w - r_l\):
\[ \frac{\partial \mathcal L}{\partial \Delta} = -\big(1 - \sigma(\Delta)\big) = -\sigma(-\Delta), \qquad \nabla_\phi\mathcal L = -\sigma(-\Delta)\,\big(\nabla_\phi r_w - \nabla_\phi r_l\big) \]
- Interpretação: o peso do exemplo é a **probabilidade de o modelo estar errado**. Pares já bem separados (\(\Delta \gg 0\)) quase não contribuem; pares invertidos contribuem com peso ~1 (A.1).

**Invariâncias e normalização.**
- A perda depende só de diferenças → **invariante a translação** de \(r\). Por isso todos normalizam depois: Stiennon "normalizamos as saídas do RM para que os resumos de referência tenham média 0" [P §3.4]; InstructGPT "como a perda é invariante a deslocamentos, normalizamos o RM com um bias para que as demonstrações dos rotuladores tenham média 0 antes do RL" [P §3.5]; Ziegler normaliza para média 0 e variância 1 sob \(y\sim\rho(\cdot|x)\) [P §2]; N+ implementa isso calculando a média nas referências do SFT e colocando-a como bias da cabeça [P Detalhe 15].
- **Não** é invariante a escala: a escala de \(r\) define a "temperatura" da preferência; ela interage com \(\beta\) da KL no RL (ver Llama 2 `whiten(logit(·))`, §4.1 e §4.8).

**Rótulos suaves e empates.** Se o rótulo é uma distribuição \(\mu\) (empate → \(\mu = (½, ½)\); rotulador IA → \([0{,}6, 0{,}4]\)), usa-se a entropia cruzada completa:
\[ \mathcal L = -\mu(1)\log\hat P(y_1\succ y_2) - \mu(2)\log\hat P(y_2 \succ y_1) \]
(Christiano 2017 [P §2.2.3]; RLAIF treina o RM em rótulos suaves com "entropia cruzada sobre a softmax dos escores" [P Lee 2023 §2.2.1].)

### 2.2 Plackett–Luce: rankings de K itens

**Derivação.** Plackett–Luce modela um ranking \(\tau\) (permutação de \(K\) respostas) como escolhas sequenciais sem reposição, cada uma uma softmax sobre os restantes:

\[ P(\tau \mid x) = \prod_{i=1}^{K} \frac{\exp\big(r(x, y_{\tau(i)})\big)}{\sum_{j=i}^{K} \exp\big(r(x, y_{\tau(j)})\big)} \]

- \(\tau(i)\): índice da resposta colocada na posição \(i\); o denominador soma sobre as ainda não escolhidas.
- \(K=2\) reduz a Bradley–Terry. O 1º fator sozinho é o modelo de "melhor de K" (softmax top-1) — a perda de Ziegler 2019:
  \[ \mathcal L = -\mathbb E\Big[\log\frac{e^{r(x,y_b)}}{\sum_i e^{r(x,y_i)}}\Big] \]
  com \(b\) = índice escolhido pelo humano entre 4 [P §2, eq. 1].
- Equivale a supor ruído Gumbel i.i.d. e ordenar por utilidade (mesma derivação de §2.1).

**O que o InstructGPT fez de fato.** Não usou PL: decompôs cada ranking em **todos os \(\binom{K}{2}\) pares** e aplicou BT [P eq. 1]:
\[ \text{loss}(\theta) = -\frac{1}{\binom{K}{2}}\,\mathbb E_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\Big] \]
- \(K\in[4,9]\) → \(\binom{K}{2}\in\{6,10,15,21,28,36\}\) pares por prompt (cálculo).
- **Truque crucial:** todos os pares de um prompt entram **no mesmo elemento do batch**. Motivo [P §3.5 e nota 5]: pares do mesmo prompt são muito correlacionados; embaralhá-los num dataset "fez o RM sobreajustar em uma única passada", porque cada resposta seria usada em \(K-1\) atualizações. Agrupar exige **uma** forward por resposta (\(K\) forwards, não \(\binom{K}{2}\)) e "atinge acurácia de validação e log loss muito melhores". Batch de 64 prompts → até \(64\times36 = 2.304\) comparações [P Ap. C.2]. Empates foram descartados.
- (interpretação) A decomposição em pares é uma *verossimilhança composta*: ela não é a verossimilhança PL do ranking (os pares não são independentes), mas é consistente sob BT e mais simples. Zhu, Jordan & Jiao 2023 (arXiv 2301.11270) analisam formalmente MLE com comparações pareadas vs \(K\)-wise sob PL. O exemplo A.3 mostra os dois números para o mesmo ranking.

### 2.3 Variantes da perda do RM

| Variante | Fórmula | Quem usa | Para quê |
|---|---|---|---|
| BT padrão | \(-\log\sigma(r_w - r_l)\) | Stiennon, InstructGPT, quase todos | direção |
| BT com margem | \(-\log\sigma\big(r_c - r_r - m(r)\big)\) | Llama 2 [P eq. 2] | exige separação maior em pares "muito melhores" |
| BT escalado | \(-m\,\log\sigma(r_c - r_r)\) | HelpSteer2-Preference [P eq. 3] | pesa pares de alta magnitude ("amostragem repetida") sem impor distância |
| Rótulo suave | entropia cruzada contra \(\mu\) | Christiano; RLAIF | empates / confiança do rotulador IA |
| Melhor de K | softmax top-1 | Ziegler 2019 | coleta "escolha a melhor de 4" |
| Regressão multi-atributo | \(\sum_a (\hat s_a - s_a)^2\) | HelpSteer/HelpSteer2 ("SteerLM Regression") [P] | Likert absoluto; 5 cabeças |
| BT + perda de LM auxiliar | \(\mathcal L_{\text{BT}} + \beta_{\text{rm}}\mathcal L_{\text{LM}}\) | Zheng et al. "Secrets" (\(\beta_{\text{rm}}=1\)) [P §4.2] | regularizar o backbone |

**Margens do Llama 2** [P Tab. 27] (valores de \(m(r)\) por nível de preferência):

| | Significantly better | Better | Slightly better | Negligibly better / Unsure |
|---|---|---|---|---|
| Margin Small | 1 | 2/3 | 1/3 | 0 |
| Margin Large | 3 | 2 | 1 | 0 |

Ablação [P Tab. 28] (acurácia do RM de helpfulness): sem margem 62,5% (média) → Small 63,0% → Large 62,9%; o ganho aparece nos pares "significantly better" (79,1 → 80,4 → 80,7). Os autores notam que margem grande empurra os escores a uma distribuição bimodal e sugerem "investir em calibração da recompensa" [P Ap. A.3.3].

**Resultado HelpSteer2-Preference** [P, tabela principal de resultados]: treinando do zero, Regular BT 91,5 / Margin 91,5 / Scaled 92,7 no RewardBench; inicializando a partir do modelo de regressão de helpfulness: Regular 92,7, Margin 93,0, **Scaled 93,7**, e **Scaled + ExPO = 94,1** (ExPO = extrapolação dos deltas de pesos com fator 1,52 encontrado por busca) — ou seja, regressão e BT carregam informação complementar.

### 2.4 Arquitetura: LM + cabeça escalar

- **Backbone**: um transformer pré-treinado/SFT (o mesmo tipo de modelo do Módulo 1, sem amostragem). InstructGPT: "partindo do modelo SFT com a camada final de *unembedding* removida, treinamos um modelo que recebe prompt e resposta e emite um escalar" [P §3.5]; em geral "a camada de unembedding é substituída por uma projeção para um valor escalar" [P Ap. C]. Llama 2 inicializa os RMs a partir de checkpoints do próprio modelo chat "para que o RM saiba o que o modelo chat sabe" [P §3.2.2]. Stiennon adiciona "uma cabeça linear inicializada aleatoriamente" ao baseline supervisionado [P §3.4].
- **Cabeça**: \(r_\phi(x,y) = w^\top h_L(x,y)_{\text{último}} + b\), com \(h_L \in \mathbb R^{d_{\text{model}}}\) o estado oculto final no **token EOS**. N+ [P Detalhe 11–13]: inicialização \(w\sim\mathcal N(0, 1/\sqrt{d_{\text{model}}+1})\) (corrigindo um typo do paper original); a recompensa é extraída **somente no token EOS** (índice do primeiro PAD − 1); os "logits de recompensa" nas demais posições são inválidos e quase sempre negativos. Karpathy menciona um token especial `<|reward|>` em sistemas mais novos da OpenAI [P Detalhe 12, citando vídeo].
- **Multi-atributo**: HelpSteer2 projeta o estado do token de fim de resposta em **5 escalares** (um por atributo) [P §3]. Skywork, ArmoRM etc. seguem linhas semelhantes (não verificado aqui em detalhe).
- **Tamanho**: InstructGPT usou **um único RM de 6B para políticas de 1,3B, 6B e 175B** — "175B poderia atingir menor perda de validação, mas (1) o treino era mais instável, tornando-o pior como inicialização da função valor do PPO, e (2) RM e crítico de 175B aumentam muito o custo do PPO" [P Ap. C.2]. Stiennon usou RM = política = crítico em tamanho [P §3.4].
- **Escala**: Stiennon treinou 7 RMs de 160M a 13B em 8k–64k comparações: **dobrar os dados → +~1,1%** de acurácia de validação; **dobrar o modelo → +~1,8%** [P §4.3]. Gao 2022: abaixo de ~2.000 comparações, os RMs ficam perto do acaso para todos os tamanhos [P §3.3].
- **Pré-treino de modelo de preferência (PMP)**: Askell et al. 2021 propõem uma etapa intermediária treinando em preferências "baratas" (StackExchange, Reddit, Wikipedia) antes do fine-tuning em dados humanos; melhora a eficiência amostral em modelos grandes; "ranked preference modeling" supera imitação e escala melhor, enquanto "binary discrimination" escala como imitação [P Abstract, §4].

### 2.5 Detalhes de treino que importam

| Detalhe | Evidência |
|---|---|
| **1 época** | InstructGPT: "múltiplas épocas sobreajustam rapidamente, com deterioração óbvia da perda de validação" [P Ap. C.2]; Ziegler: "época única para evitar sobreajuste aos poucos dados humanos" [P §2.2]; Llama 2: "treinar mais leva a sobreajuste" [P §3.2.2]; Stiennon/N+: 1 época. Gao 2022: 4 épocas vs 1 época com 4× mais dados → as 4 épocas "não mudam o escore gold em nada" [P nota 7]. |
| **LR e batch** | InstructGPT 9e-6, cosseno até 10%, batch 64 prompts; insensível a ±50% de LR [P Ap. C.2]. Stiennon 1,5e-5 (1,3B) / 5e-6 (6,7B), batch 64 [P Ap. B]. Llama 2: máx. 5e-6 (70B) / 1e-5 (demais), cosseno até 10%, warmup 3%, batch 512 pares (1024 linhas) [P §3.2.2]. UltraRM (LLaMA2-13B): 1e-5, batch 512 pares, warmup 3% [P Ap. D.1]. N+: 3e-6 para SFT, RM e PPO (sem sweeps) [P Detalhe 10]. HelpSteer2 (regressão): 2 épocas, batch 128, 2e-6 (70B) / 7e-7 (340B) [P §3]. |
| **Sementes** | Stiennon varre 3–10 sementes e escolhe pelo dev set, pois "a ordem dos dados e a inicialização da cabeça afetam os resultados" [P Ap. B]. |
| **Dropout desligado** | N+ desliga dropout em SFT, RM e PPO (essencial no PPO, ver §4.8) [P Detalhe 7]; Ziegler idem [P §2.2]. InstructGPT usa *residual dropout* 0,2 no SFT [P §3.5]. |
| **Padding** | RM treina com right-padding; no PPO as queries são left-padded para geração → ajustar máscaras; diferença média no 6.9B de −0,000544 [P N+ Detalhe 14]. |
| **Mistura de dados** | Llama 2 mistura dados abertos e próprios; RM de safety com 90/10 de safety/helpfulness [P §3.2.2]. |
| **Memória** | cada par custa 2 sequências na forward (chosen + rejected); com ranking agrupado, \(K\) sequências por prompt (InstructGPT). (cálculo) |

### 2.6 Avaliação e calibração do RM

- **Acurácia de pares** em held-out (métrica principal; ver tetos em §1.5).
- **Calibração**: um RM BT prevê \(\hat P = \sigma(\Delta)\). Ele é calibrado se, entre os pares com diferença \(\Delta\), a fração em que o humano concorda for \(\sigma(\Delta)\). Bai 2022 plota acurácia vs \(\Delta\) contra a curva \(1/(1+e^{-\Delta})\): **PMs treinados só em helpfulness são "muito bem calibrados"; os treinados na mistura HH são "ligeiramente subconfiantes"** [P §3.3, Fig. 9]. Os autores destacam que isso importa porque as diferenças de escore do PM viram a recompensa do RL. N+ refaz o gráfico e encontra RMs "ainda sub-calibrados" no TL;DR [P Detalhe 18].
- **ECE (Expected Calibration Error)**: \(\text{ECE} = \sum_b \frac{n_b}{N}\,|\text{acc}_b - \text{conf}_b|\), com bins \(b\) de confiança \(\max(\hat P, 1-\hat P)\). (definição padrão; exemplo em A.8)
- **Recalibração por temperatura**: Gao 2022 reescala os logits dos RMs proxy "para minimizar a entropia cruzada num conjunto de validação com rótulos suaves" [P §2.2] — i.e., ajusta um escalar \(T\) em \(\sigma(\Delta/T)\). Recalibrar não muda BoN (só a ordem importa) e "provavelmente não afeta o RL porque Adam é invariante à escala da perda" [P §2.2].
- **Generalização fora da distribuição**: Stiennon — RM transfere para CNN/DM com 62,4% (1,3B) e 66,5% (6,7B) de concordância vs 66,9% inter-rotulador; RMs preferem edições mínimas melhoradas 79,4%/82,8% (humanos 84,1%) e detectam papéis trocados 92,9%/97,2% [P §4.3].
- **RewardBench / RewardBench 2** como suíte padronizada (§1.6). HelpSteer2 escolhe RewardBench pela diversidade (4 categorias, 23 subcategorias) e por ter >80 modelos no leaderboard na época [P §3].

### 2.7 Viés de comprimento

- **Stiennon 2020** [P §4.3]: o RM 6,7B "prefere edições que encurtam o resumo apenas 62,6% das vezes (vs 76,4% dos humanos)". Gao 2022, nota 10: "políticas otimizadas produzindo respostas muito longas mesmo quando uma curta seria preferida é um problema real que observamos no cenário InstructGPT" [P].
- **Singhal et al. 2023 ("A Long Way to Go")** [P]:
  - PPO aumenta muito o comprimento em todos os cenários (WebGPT, Stack, RLCD).
  - Decompondo o ganho de recompensa: em WebGPT e RLCD, **70–90% da melhoria** vem do comprimento [P §3].
  - **PPO com recompensa = só comprimento** reproduz a maior parte do ganho: 56% vs 58% de win-rate do PPO padrão no WebGPT; 64% vs 63% no RLCD [P Abstract/Tab. 2].
  - A fonte dominante do viés são os **RMs**, "não robustos e facilmente influenciados por vieses de comprimento nos dados".
  - A penalidade KL ajuda até o PPO "só-comprimento" a aprender features boas (saídas patológicas repetitivas teriam KL alto) [P §3].
- **Mitigações** (listadas, sem números verificados aqui): penalidade explícita de comprimento na recompensa; RMs com cabeças desacopladas de comprimento e qualidade (ODIN, Chen et al. 2024, arXiv 2402.07319); avaliação com controle de comprimento (AlpacaEval LC, Dubois et al. 2024, arXiv 2404.04475); dados balanceados por comprimento.

### 2.8 Reward hacking e superotimização

**Goodhart.** "Quando uma medida vira alvo, deixa de ser uma boa medida." O RM é um *proxy* aprendido; otimizá-lo demais explora seus erros.

- **Stiennon 2020** [P §4.3, Fig. 5]: variando \(\beta\), "sob otimização leve os modelos melhoram (segundo rotuladores); sob otimização forte, pioram" — o RM continua dando nota alta a resumos que humanos rejeitam. O mesmo vale para ROUGE.
- **"Secrets of RLHF"** [P §5.1]: "colapso de padrão" no PPO vanilla — a política "tende a enganar o RM com padrões específicos para obter escores anormalmente altos"; a curva de recompensa sobe enquanto humanos/GPT-4 veem pioria.

**Gao, Schulman & Hilton 2022 — leis de escala da superotimização** [P]:
- **Setup sintético**: o RM 6B do InstructGPT é o **"gold"** (faz papel de humano); ele rotula pares de rollouts para treinar **RMs proxy de 3M a 3B**; 100.000 comparações sintéticas (10% held-out); políticas GPT-3 SFT; PPO **sem KL** (exceto §3.6); RMs recentrados para média 0 na política inicial.
- **Distância**: \(d := \sqrt{D_{KL}(\pi \,\|\, \pi_{\text{init}})}\).
- **Formas funcionais do escore gold**:
  \[ R_{\text{bon}}(d) = d\,(\alpha_{\text{bon}} - \beta_{\text{bon}}\,d), \qquad R_{\text{RL}}(d) = d\,(\alpha_{\text{RL}} - \beta_{\text{RL}}\log d) \]
  - \(R(0) := 0\); \(\alpha, \beta\) dependem do tamanho do RM proxy, dos dados etc. A forma RL "provavelmente não vale perto da origem (inclinação infinita)" [P nota 1].
  - **KL do best-of-n** (analítico): \(\text{KL}_{\text{bon}} = \log n - \frac{n-1}{n}\) [P §2, citando Stiennon Ap. G.3]. n = 1.000 → 5,91 nats; n = 60.000 → 10,00 nats (cálculo; o paper diz "≈6" e "≈10").
  - **Máximos** (cálculo a partir das formas): BoN tem pico em \(d^* = \alpha/(2\beta)\), \(R^* = \alpha^2/(4\beta)\); RL tem pico em \(d^* = e^{\alpha/\beta - 1}\), \(R^* = \beta d^*\) (A.9).
- **Achados** [P §1, §3]:
  1. **RL vs BoN**: em função do KL, RL é mais lento tanto para otimizar quanto para superotimizar — RL "consome" muito mais KL (KL cresce ~quadraticamente com os passos sem penalidade; BoN ~\(\log n\)). Logo, **KL não serve para comparar quantidade de otimização entre métodos** [P §4.1]. Mas em função do escore proxy, os dois se parecem.
  2. **Escala suave com o tamanho do RM**: \(\alpha_{\text{bon}}, \beta_{\text{bon}}, \beta_{\text{RL}}\) variam ~logaritmicamente com os parâmetros do RM; \(\alpha_{\text{RL}}\) quase constante.
  3. **Dependência fraca do tamanho da política**: políticas maiores ganham menos com a otimização, mas superotimizam quase igual (gap proxy–gold e KL do pico quase iguais).
  4. **Penalidade KL ≈ early stopping**: no setup deles, a penalidade aumenta o proxy alcançável por KL, mas **não melhora a fronteira gold × KL**; "o efeito da penalidade sobre o gold é parecido com parar cedo" [P §3.6] — com ressalva de sensibilidade a hiperparâmetros. Notam que o clipping do PPO já é uma penalidade implícita em \(D_{KL}(\pi_{\text{old}}\|\pi)\).
  5. RMs de mesma perda de validação parecem igualmente robustos, independentemente da combinação tamanho×dados (evidência "fraca") [P §3.3].
- **Taxonomia de Goodhart** (Manheim & Garrabrant) aplicada [P §4.2]:
  - **Regressional**: proxy = gold + ruído independente → parte da otimização "seleciona ruído"; prediz gold monotônico no proxy (não basta para explicar a queda). Fórmula para \(X\) gold normal e ruído \(Z\): \(\mathbb E[X|\hat X=\hat x] = \mathbb E[X] + (\hat x - \mathbb E X - \mathbb E Z)\frac{\text{Var}X}{\text{Var}X + \text{Var}Z}\). Interpretado como o termo \(\alpha\).
  - **Extremal**: fora da distribuição de treino do RM a relação proxy–gold se desfaz (ex.: comprimento) → principal responsável pela não-monotonicidade e pelo termo \(\beta\).
  - **Causal** e **Adversarial** discutidos qualitativamente.
- **Bai 2022** [P §4.3, Fig. 4]: relação **aproximadamente linear entre o ganho de escore do PM e \(\sqrt{D_{KL}}\)** no início do treino, em todas as execuções; dividindo o dataset 50/50 em PM "de treino" e "de teste", o RL é robusto até ~**150k** amostras de treino; depois, os PMs discordam (o de treino dá recompensa média maior).

### 2.9 Mitigações: ensembles, médias de pesos, conservadorismo

- **Christiano 2017** [P §2.2.3]: ensemble de preditores, cada um treinado em \(|D|\) triplas reamostradas com reposição (bootstrap); \(\hat r\) = média dos membros **normalizados independentemente**; 1/e dos dados como validação; regularização ℓ2 ajustada para manter validação entre 1,1× e 1,5× a perda de treino; dropout em alguns domínios.
- **Coste et al. 2023** [P §3, eqs. 3–5], com \(k\) RMs \(R_1..R_k\):
  \[ R_\mu = \tfrac1k\sum_i R_i, \qquad R_{\text{WCO}} = \min_i R_i, \qquad R_{\text{UWO}} = \tfrac1k\sum_i R_i - \lambda\,\tfrac1k\sum_i\Big(R_i - \tfrac1k\sum_j R_j\Big)^2 \]
  - Média não é conservadora (um membro que superestima basta para ser explorado); WCO não tem hiperparâmetro mas pode custar desempenho; UWO penaliza discordância com \(\lambda\).
  - Resultado: com e sem 25% de ruído de rótulo, otimização conservadora "praticamente elimina a superotimização" e melhora em **até 70%** no BoN; no PPO sempre reduz; combinada com uma KL pequena "previne a superotimização sem custo de desempenho" [P Abstract].
- **Eisenstein et al. 2023 ("Helping or Herding?")** [P Abstract, arXiv 2312.09244]: ensembles que variam a **semente de pré-treino** generalizam melhor do que variar só a semente de fine-tuning; mas **"mesmo ensembles de pré-treino não eliminam o reward hacking"**, porque todos os membros exibem padrões de erro parecidos.
- **WARM (Ramé et al. 2024)** [P §1]: em vez de média de *predições*, média de **pesos** de \(M\) RMs fine-tunados do mesmo pré-treino (com hiperparâmetros/ordens de dados diferentes), explorando *linear mode connectivity*; custa um único RM na inferência; mais robusto a mudança de distribuição e a preferências inconsistentes.
- **Outras alavancas**: penalidade KL (§4.2), parada antecipada com avaliação held-out/humana, RM re-treinado on-policy (Bai, Llama 2), normalização/clipping da recompensa (§4.8), controle de comprimento (§2.7).

### 2.10 Recompensa de resultado (ORM) vs de processo (PRM)

- **Definições (Uesato et al. 2022)** [P §2]: o RM é um LM treinado para prever um token "correto"/"incorreto" **após cada passo**.
  - **ORM**: o rótulo de cada passo é se a **resposta final** da amostra inteira bate com a referência (proposto por Cobbe et al. 2021, arXiv 2110.14168, os "verifiers"). Maximizar o ORM passo a passo ≈ maximizar a probabilidade estimada de **chegar** à resposta certa — i.e., o ORM aprende algo como uma função valor.
  - **PRM**: o rótulo após cada passo diz se **os passos até ali estão corretos** (requer anotação humana).
  - **Resultados no GSM8K** [P Abstract]: supervisão só por resultado dá **erro de resposta final semelhante** com menos rótulos; mas para **raciocínio correto** é preciso supervisão de processo ou RMs que a emulem. Melhor sistema: erro final **16,8% → 12,7%** e **erro de traço** (entre soluções com resposta certa) **14,0% → 3,4%**.
- **Lightman et al. 2023 ("Let's Verify Step by Step")** [P]:
  - PRM800K (§1.6); rotula só até o **primeiro passo incorreto** (para comparar com ORM de forma justa).
  - **Escore da solução = produto** das probabilidades de cada passo estar correto (neutros contados como positivos); min vs produto e neutro±: diferenças "pequenas" [P Ap. F, Tab. 4].
  - **Best-of-1860 no subconjunto MATH de 500 problemas**: PRM **78,2%**, ORM 72,4%, votação por maioria 69,6% [P Fig. 3].
  - Active learning → 2,6× de eficiência de dados.
- **Math-Shepherd (Wang et al. 2023)** [P §3.3]: rótulos de processo **sem humanos**, por rollouts Monte Carlo: um "completer" termina \(N\) soluções a partir do passo \(s_i\); com respostas finais \(\{a_j\}\) e gabarito \(a^*\):
  \[ y^{\text{HE}}_{s_i} = \mathbb 1\big[\exists j: a_j = a^*\big], \qquad y^{\text{SE}}_{s_i} = \frac{1}{N}\sum_{j=1}^{N}\mathbb 1[a_j = a^*] \]
  - HE (*hard estimation*): passo bom se **alguma** continuação acerta; SE (*soft*): fração que acerta (≈ estimativa MC de valor do estado).
  - PPO passo a passo com Math-Shepherd: Mistral-7B **77,9% → 84,1%** no GSM8K e **28,6% → 33,0%** no MATH; com verificação, 89,1% e 43,5% [P Abstract].
- **Ligação conceitual (interpretação):** SE é exatamente uma estimativa Monte Carlo de \(V^\pi(s_i)\) com recompensa binária de acerto; um PRM treinado nesses rótulos é um crítico aprendido offline. Isso conecta §2.10 com o crítico do PPO (§3.5) e explica por que GRPO/RLOO (que estimam baselines por amostragem) e PRMs são "dois lados" do mesmo problema de atribuição de crédito.
- **Recompensas verificáveis (RLVR)**: quando há verificador programático (resposta numérica, testes de código), o "RM" é uma função determinística — tema do capítulo de GRPO/raciocínio (DeepSeekMath arXiv 2402.03300; DeepSeek-R1 arXiv 2501.12948). Aqui só registramos que ORM aprendido → verificador exato é o limite natural.

### 2.11 RMs generativos e LLM-as-judge

- **LLM-as-a-judge (Zheng et al. 2023)** [P]: GPT-4 como juiz em MT-bench e Chatbot Arena; concordância com humanos **85%** (setup sem empates) vs **81%** humano–humano; humanos acharam os julgamentos do GPT-4 "razoáveis" em 75% dos casos em que discordavam [P §4.2]. Modos: comparação pareada, nota única, nota com referência. Vieses: **posição**, **verbosidade** (ataque de "lista repetitiva"), **auto-favorecimento**, raciocínio limitado em matemática [P §3.3]. Mitigações: trocar a ordem e só aceitar vitória consistente; juiz com referência/CoT.
- **GenRM (Zhang et al. 2024, "Generative Verifiers")** [P]: treina o verificador com **next-token prediction**; a recompensa é a probabilidade do token "Yes" após "Is the answer correct (Yes/No)?"; variante **GenRM-CoT** gera uma racionalização antes e faz **votação por maioria** sobre várias racionalizações (computação extra no teste). Best-of-N: algorítmico **5% → 45,3%**, GSM8K **73% → 93,4%**, transferência p/ MATH **28% → 44,6%**, MMLU álgebra abstrata **37,9% → 53,5%** [P Abstract]. Treinar com racionalizações sintéticas basta para achar erros sutis.
- **DeepSeek-GRM (Liu et al. 2025)** [P Abstract]: RM generativo **pointwise** (aceita 1 ou N respostas), treinado com **SPCT** (*Self-Principled Critique Tuning*) via RL online para gerar **princípios** adaptativos e **críticas**; escala na inferência por amostragem paralela + votação, guiada por um **meta-RM**; relatam superar escalar o tamanho do RM no treino.
- **Pointwise vs pairwise (interpretação)**: pairwise casa com BT e é mais robusto a deriva de escala; pointwise é necessário para PPO/GRPO (cada rollout precisa de um escalar) — por isso RMs generativos modernos convertem crítica → escore numérico.
- **Rubricas / feedback binário em linguagem natural**: HelpSteer3 lista o paper **RLBFF** (arXiv 2509.21319) entre seus usos [D] — não lido em detalhe aqui.

### 2.12 RLAIF: preferências rotuladas por IA

- **Constitutional AI (Bai et al. 2022b, arXiv 2212.08073)** [P]: fase supervisionada (auto-crítica e revisão guiadas por uma lista de princípios) + fase de RL em que **um modelo rotula qual de duas respostas é menos nociva** segundo princípios; o PM é treinado na mistura de preferências de IA (harmlessness) e humanas (helpfulness) e usado como recompensa no PPO ("RL-CAI"). Detalhes ficam para o capítulo de CAI do Módulo 2.
- **Lee et al. 2023 ("RLAIF vs RLHF")** [P]:
  - Rotulador: PaLM 2 L instruído (sem RL); política: PaLM 2 XS. O rótulo é a **softmax dos log-probs dos tokens "1" e "2"** após o prompt (preâmbulo, exemplos few-shot opcionais, amostra, "Preferred Response=") → rótulo suave → RM com entropia cruzada ("destilação") [P §2.1, §2.2.1].
  - **Viés de posição** mitigado invertendo a ordem e tirando a média [P §2.1.1]; **CoT** do rotulador melhora o alinhamento com humanos de forma consistente; preâmbulo detalhado e few-shot só ajudam em algumas tarefas.
  - Resultados: humanos preferem RLAIF/RLHF ao SFT **71% / 73%** (sumarização) e **63% / 64%** (diálogo útil) — diferença não significativa; **harmless rate**: RLAIF **88%**, RLHF 76%, SFT 64% [P §1, Fig. 1]. Com rotulador do **mesmo tamanho** da política, 68% vs SFT [P §4.3].
  - **d-RLAIF** (direct): pula o RM — o LLM dá uma nota 1–10 diretamente durante o RL, evitando o RM "velho" (*stale*); iguala ou supera o RLAIF canônico [P §2.2.2].
  - RL usado: **REINFORCE com baseline** (não PPO), "mais simples e ainda eficaz"; política e valor inicializados do SFT [P §3.3].
- **UltraFeedback** é RLAIF em escala de dataset (GPT-4 rotulador); **Skywork-Reward-V2** usa curadoria humano+IA em 40M pares (§1.6).
- **RLAIF e ruído (interpretação)**: rótulos de IA são baratos e consistentes, mas carregam os vieses do juiz (posição, comprimento, auto-favorecimento) — todo o §2.7–§2.9 se aplica, e um RM treinado em rótulos de IA herda e às vezes amplifica esses vieses.

---

## 3. Fundamentos de RL para LLMs

### 3.1 O MDP em nível de token

Um LLM gerando uma resposta é um agente num **processo de decisão de Markov** (MDP) \((\mathcal S, \mathcal A, P, r, \rho_0, \gamma)\):

| Elemento | No LLM | Comentário |
|---|---|---|
| Estado \(s_t\) | \((x, y_1, \dots, y_{t-1})\) — prompt + prefixo gerado | é tudo que o modelo condiciona; na implementação vira o KV cache [M1: cap. `prefill-decode-kv`] |
| Ação \(a_t\) | o próximo token \(y_t \in \mathcal V\) | \(\lvert\mathcal V\rvert\) da ordem de \(10^5\) (ex.: o vocabulário do GLM-5.3 após o pad TP3 é 154.944 [M1: glm-mods.md M44]) |
| Política \(\pi_\theta(a_t \mid s_t)\) | \(\text{softmax}(z_t/T)_{a_t}\), \(z_t\) = logits | a temperatura \(T\) faz parte da política (amostrar com \(T\ne1\) muda \(\pi\)) |
| Transição \(P(s_{t+1} \mid s_t,a_t)\) | **determinística**: \(s_{t+1} = s_t \oplus a_t\) | não há ambiente estocástico (exceto em agentes com ferramentas) |
| Distribuição inicial \(\rho_0\) | distribuição de prompts \(\mathcal D\) | |
| Recompensa \(r_t\) | **esparsa**: 0 em todo \(t<T\); \(r_\phi(x,y)\) no token final (EOS); + termo KL por token no RLHF (§4.2) | InstructGPT: "um ambiente *bandit* que apresenta um prompt aleatório, espera uma resposta, produz uma recompensa determinada pelo RM e termina o episódio" [P §3.5] |
| Horizonte | \(T \le\) `max_tokens` | episódio = uma resposta |
| Desconto \(\gamma\) | quase sempre **1** | Stiennon \(\gamma=1\) [P Ap. B]; InstructGPT "nenhum desconto ao estimar o GAE" [P Ap. C.4]; Bai \(\gamma=1\) [P Ap. B.1] |

**Duas visões equivalentes do mesmo problema:**
- **Visão por token** (PPO clássico, InstructGPT): cada token é uma ação; o crítico estima \(V(s_t)\) em cada posição; a vantagem distribui crédito ao longo da sequência.
- **Visão bandit / por sequência** (Ahmadian et al. 2024, RLOO): a resposta inteira é **uma** ação; como a recompensa só chega no fim, "pode ser mais apropriado modelar a geração inteira como uma única ação" [P §3]; usa-se REINFORCE na sequência com \(\log\pi_\theta(y|x) = \sum_t \log\pi_\theta(y_t|s_t)\).
- (interpretação) Com \(\gamma=1\), recompensa só terminal e sem KL por token, as duas visões têm o **mesmo** gradiente esperado; diferem no estimador de vantagem (crítico por token vs baseline por sequência) e portanto na variância/viés.

### 3.2 Retorno, funções valor e vantagem

\[ G_t = \sum_{k=0}^{T-t} \gamma^k\, r_{t+k}, \qquad V^\pi(s) = \mathbb E_\pi[G_t \mid s_t = s], \qquad Q^\pi(s,a) = \mathbb E_\pi[G_t \mid s_t=s, a_t=a] \]
\[ A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) \]
- \(G_t\): retorno (soma descontada de recompensas futuras); \(V^\pi\): valor esperado de estar em \(s\) seguindo \(\pi\); \(Q^\pi\): idem após tomar \(a\); \(A^\pi\): quanto a ação \(a\) é melhor que a média da política em \(s\). Note que \(\mathbb E_{a\sim\pi}[A^\pi(s,a)] = 0\).
- **Equações de Bellman**: \(V^\pi(s) = \mathbb E_{a\sim\pi}\big[r(s,a) + \gamma\,\mathbb E_{s'}V^\pi(s')\big]\); no MDP de tokens com transição determinística e recompensa terminal: \(Q^\pi(s_t, y_t) = V^\pi(s_t \oplus y_t)\) para \(t<T\).
- **No LLM (interpretação)**: \(V^\pi(x, y_{<t})\) = "qual recompensa final esperar se eu continuar amostrando deste prefixo" — o mesmo objeto que a estimativa *soft* do Math-Shepherd (§2.10) e que o crítico do PPO tenta prever em cada posição; N+ observa que o crítico treinado "parece um RM por token" [P Detalhe 22].

### 3.3 O teorema do gradiente de política

**Objetivo**: \(J(\theta) = \mathbb E_{\tau\sim p_\theta}[R(\tau)]\), com trajetória \(\tau = (s_0,a_0,\dots)\) e \(p_\theta(\tau) = \rho_0(s_0)\prod_t \pi_\theta(a_t|s_t)\,P(s_{t+1}|s_t,a_t)\).

**Truque da log-derivada** (\(\nabla p = p\,\nabla\log p\)):
\[ \nabla_\theta J = \mathbb E_{\tau}\big[R(\tau)\,\nabla_\theta\log p_\theta(\tau)\big] = \mathbb E_{\tau}\Big[R(\tau)\sum_{t}\nabla_\theta\log\pi_\theta(a_t|s_t)\Big] \]
- A dinâmica \(P\) e \(\rho_0\) **somem** porque não dependem de \(\theta\) — não é preciso derivar através do ambiente (nem do RM!).
- **Causalidade**: uma ação em \(t\) não afeta recompensas anteriores → troca-se \(R(\tau)\) pelo *reward-to-go* \(G_t\).
- **Forma geral** (Schulman et al. 2015b, GAE, eq. 1–2) [P §2]: \(\nabla J = \mathbb E\big[\sum_t \Psi_t \nabla\log\pi_\theta(a_t|s_t)\big]\) com \(\Psi_t \in \{R(\tau),\ G_t,\ G_t - b(s_t),\ Q^\pi(s_t,a_t),\ A^\pi(s_t,a_t),\ \delta_t\}\). "A escolha \(\Psi_t = A^\pi\) dá quase a menor variância possível" [P].
- **Teorema (Sutton et al. 2000)** na forma de distribuição de estados: \(\nabla J = \sum_s d^\pi(s)\sum_a \nabla\pi_\theta(a|s)\,Q^\pi(s,a)\) — o gradiente não exige derivar \(d^\pi\) (a distribuição de estados visitados).

**No LLM:**
\[ \nabla_\theta J = \mathbb E_{x\sim\mathcal D,\ y\sim\pi_\theta(\cdot|x)}\Big[\sum_{t=1}^{T}\hat A_t\,\nabla_\theta\log\pi_\theta(y_t \mid x, y_{<t})\Big] \]
- **Leitura pedagógica**: é a **mesma retropropagação do SFT** (gradiente de \(\log\) da probabilidade do token observado), só que cada token é **ponderado pela vantagem** \(\hat A_t\) — positivo: "faça mais disso"; negativo: "faça menos disso". SFT ≡ gradiente de política com \(\hat A_t = 1\) em dados de demonstração (interpretação; amplamente usada como intuição).
- **Gradiente da log-softmax em relação aos logits** (base de A.22): \(\frac{\partial\log\pi(a)}{\partial z_j} = \mathbb 1[j=a] - \pi(j)\). Uma atualização com vantagem \(A\) e passo \(\eta\) faz \(z \leftarrow z + \eta A(\mathbf e_a - \pi)\): sobe o logit do token amostrado e desce os demais proporcionalmente às suas probabilidades.

### 3.4 REINFORCE e baselines

**REINFORCE (Williams 1992)**: estimador Monte Carlo \(\hat g = \frac1N\sum_i R(\tau_i)\sum_t\nabla\log\pi_\theta(a^i_t|s^i_t)\). Não-enviesado, mas de **variância alta** (soma de muitos termos; a escala de \(R\) multiplica tudo).

**Baseline**: qualquer \(b(s_t)\) que não dependa da ação mantém o estimador não-enviesado:
\[ \mathbb E_{a\sim\pi}\big[\nabla_\theta\log\pi_\theta(a|s)\,b(s)\big] = b(s)\sum_a\nabla_\theta\pi_\theta(a|s) = b(s)\,\nabla_\theta\underbrace{\textstyle\sum_a\pi_\theta(a|s)}_{=1} = 0 \]
- Baseline ótimo (escalar, minimiza a variância de cada componente): \(b^* = \mathbb E[g^2 R]/\mathbb E[g^2]\), com \(g = \nabla\log\pi\) (Greensmith, Bartlett & Baxter 2004). Na prática usa-se \(b(s) \approx V^\pi(s)\) — daí \(G_t - V(s_t) \approx A\).
- **RLOO (REINFORCE Leave-One-Out; Kool et al. 2019, usado por Ahmadian et al. 2024)**: amostrar \(k\) respostas por prompt; o baseline de cada uma é a média das outras:
  \[ \hat A_i = R(y_i) - \frac{1}{k-1}\sum_{j\ne i}R(y_j) \]
  Não-enviesado (o baseline não usa a própria amostra) e sem crítico. Equivale a \(\frac{k}{k-1}\big(R_i - \bar R\big)\) (cálculo; A.16). Ahmadian et al.: "muitos componentes do PPO são desnecessários no contexto RLHF" e REINFORCE/RLOO superam PPO, DPO e RAFT; justificativa: o modelo pré-treinado+SFT "está longe de uma parametrização aleatória", concentrando massa em poucos tokens, então a variância não é o problema dominante [P §1, §3]. Em PPO com Llama-7B no HH, \(\lambda = 1\) (gradiente de política sem bootstrap, maior variância e sem viés) foi o melhor entre 1,0 / 0,95 / 0,5 / 0,0, e remover clipping e normalizações do PPO não degradou o desempenho [P Fig. 1].
- **GRPO** (DeepSeekMath, arXiv 2402.03300) também troca o crítico por estatística de grupo — tema de outro capítulo; aqui registramos a motivação textual: "como a função valor usada no PPO é tipicamente outro modelo de tamanho comparável ao da política, ela traz um fardo substancial de memória e computação. […] No contexto de LLMs, geralmente só o último token recebe recompensa do RM, o que pode complicar treinar uma função valor precisa em cada token" [P §4.1].

### 3.5 Ator-crítico: aprendendo \(V\)

- O **crítico** \(V_\psi\) é treinado por regressão: \(\mathcal L_V(\psi) = \hat{\mathbb E}_t\big[(V_\psi(s_t) - \hat R_t)^2\big]\), com alvo \(\hat R_t\) (retorno MC, n-passos ou λ-retorno; §3.7). "Secrets of RLHF" eq. 16 [P].
- **Arquitetura no RLHF**: um LM com cabeça escalar que emite um valor **em cada posição** (diferente do RM, lido só no EOS). Stiennon: "para a função valor do PPO usamos um Transformer com parâmetros **completamente separados** da política. Isso evita que atualizações da função valor destruam parcialmente a política pré-treinada no início do treino (ablação no Ap. G.1). Inicializamos a função valor com os parâmetros do RM" [P §3.4]. InstructGPT: "a função valor é inicializada a partir do RM" [P §3.5].
- **Compartilhar ou não**: A3C compartilha todas as camadas não-de-saída (softmax de política + saída linear de valor) [P Mnih §4]; PPO (Atari/MuJoCo) também permite compartilhamento, combinando as perdas com \(c_1\) (§3.11). Em LLMs, o padrão herdado de Stiennon é **separar** (interpretação: a perda de valor no início, com crítico mal calibrado, gera gradientes grandes que vazariam para o backbone da política).

### 3.6 Diferença temporal (TD)

- **TD(0)**: \(V(s_t) \leftarrow V(s_t) + \alpha\,\delta_t\), com o **resíduo TD**
  \[ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \]
  Se \(V = V^\pi\), \(\mathbb E[\delta_t | s_t, a_t] = A^\pi(s_t, a_t)\): o resíduo TD é um estimador (enviesado se \(V\) é imperfeito) da vantagem [P GAE §3].
- **n-passos**: \(G^{(n)}_t = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n})\); vantagem \(\hat A^{(n)}_t = G^{(n)}_t - V(s_t)\) — é a do A3C (\(n \le t_{\max}\)) [P Mnih §4].
- **Trade-off**: \(n=1\) (TD puro) → baixa variância, viés de \(V\); \(n\to\infty\) (Monte Carlo) → sem viés de bootstrap, variância alta.
- **λ-retorno**: média exponencial dos n-passos: \(G^\lambda_t = (1-\lambda)\sum_{n\ge1}\lambda^{n-1}G^{(n)}_t\) (Sutton & Barto).

### 3.7 GAE(γ, λ)

**Definição** (Schulman, Moritz, Levine, Jordan, Abbeel 2015b) [P §3] — média exponencialmente ponderada dos estimadores de \(k\) passos \(\hat A^{(k)}_t\), que telescopa para:
\[ \hat A^{\text{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \]
- \(\gamma\): desconto (aqui tratado como **redutor de variância**, não como preferência temporal — "introduz viés" [P §2]); \(\lambda\in[0,1]\): controla viés×variância; \(\delta_t\): resíduo TD; \(V\): crítico atual.
- **Recursão** (como se implementa, de trás para frente): \(\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}\), com \(\hat A_{T+1}=0\) e \(V(s_{T+1}) = 0\) no fim do episódio.
- **Casos especiais** [P §3]: \(\lambda=0\) → \(\hat A_t = \delta_t\) (TD de 1 passo; "\(\gamma\)-justo só se \(V = V^{\pi,\gamma}\)", baixa variância); \(\lambda=1\) → \(\hat A_t = \sum_l\gamma^l r_{t+l} - V(s_t)\) (Monte Carlo menos baseline; "\(\gamma\)-justo independentemente da acurácia de \(V\)", alta variância).
- **Alvo do crítico**: \(\hat R_t = \hat A_t + V(s_t)\) — que é exatamente o λ-retorno \(G^\lambda_t\) (com \(V\) fixo durante o cálculo).
- **Valores usados em RLHF**: Stiennon e N+ \(\gamma=1,\ \lambda=0{,}95\) [P]; InstructGPT sem desconto [P]; PPO original \(\gamma=0{,}99,\ \lambda=0{,}95\) [P Tab. 3–5].
- **PPO truncado** (Schulman 2017, eq. 11) [P §5]: \(\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t-1}\delta_{T-1}\) em segmentos de comprimento \(T\).

**O problema do "decaimento do sinal" em CoT longo** (base do VC-PPO):
- Com recompensa só no último token e \(\gamma=1\), a contribuição direta da recompensa final para \(\hat A_t\) via a soma de \(\delta\)'s é ponderada por \(\lambda^{T-t}\). Para \(\lambda=0{,}95\): \(0{,}95^{50}\approx0{,}077\); \(0{,}95^{500}\approx7\times10^{-12}\); \(0{,}95^{8000}\approx6\times10^{-179}\) (cálculo; A.21). Ou seja, em respostas de milhares de tokens os primeiros tokens dependem quase só do crítico (bootstrap).
- **VC-PPO** (Yuan et al. 2025, arXiv 2503.01491) [W/arXiv abstract]: diagnostica "**viés de inicialização do valor** e **decaimento do sinal de recompensa**" como causas do colapso do PPO em CoT longo; propõe **pré-treino do crítico** (alinhá-lo à política inicial antes do RL) e **GAE desacoplado**: \(\lambda_{\text{crítico}} = 1\) (alvo Monte Carlo, propagação sem viés) e \(\lambda_{\text{ator}} = 0{,}95\) (menor variância da vantagem).
- **VAPO** (arXiv 2504.05118) [W/arXiv abstract]: adiciona GAE com \(\lambda\) adaptativo ao comprimento, perda de LM em exemplos positivos e amostragem em grupo; Qwen 32B base → **60,4** no AIME 2024 em **5.000 passos**, superando DeepSeek-R1-Zero-Qwen-32B e DAPO por mais de 10 pontos.

### 3.8 On-policy, off-policy e importance sampling

- **On-policy**: o gradiente usa amostras da política atual \(\pi_\theta\) (REINFORCE, A2C). **Off-policy**: amostras de outra política \(\mu\) (comportamento), corrigidas por **importance sampling** (IS):
  \[ \mathbb E_{a\sim\pi}[f(a)] = \mathbb E_{a\sim\mu}\Big[\tfrac{\pi(a)}{\mu(a)}f(a)\Big] \]
- **PPO é "quase on-policy"**: as amostras vêm de \(\pi_{\theta_{\text{old}}}\) e são reutilizadas por \(K\) épocas; após o 1º passo de gradiente, \(\pi_\theta\ne\pi_{\theta_{\text{old}}}\), e a razão \(\rho_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)\) é exatamente um peso de IS por token. O clipping limita o quão off-policy o lote pode ficar.
- **IS por sequência explode**: \(\prod_t \rho_t\) sobre centenas de tokens tem variância enorme (um único token com \(\rho=5\) já multiplica tudo; A.18). Por isso PPO usa a razão **por token** (aproximação de 1ª ordem que ignora a mudança na distribuição de estados — justificada pelo TRPO; §3.10).
- **IS truncado (TIS)**: \(\min(\rho, C)\) — troca um pequeno viés por variância limitada (Ionides 2008, *J. Comput. Graph. Stat.* 17(2):295–311; reintroduzido para LLMs por Yao et al. 2025, §4.6).
- **Fontes de off-policyness em RLHF moderno** (todas tratadas no §4.6): (i) várias épocas por lote; (ii) **RL assíncrono** (rollouts gerados por pesos de 1+ passos atrás); (iii) **descompasso numérico** entre o motor de inferência (vLLM/SGLang) e o motor de treino (FSDP/Megatron) — "os mesmos pesos" produzem probabilidades diferentes; (iv) rollouts **quantizados** (FP8/INT8); (v) roteamento de **MoE** diferente entre inferência e treino.

### 3.9 A2C / A3C — atores assíncronos

**Mnih et al. 2016 ("Asynchronous Methods for Deep RL")** [P]:
- Vários **atores-aprendizes** em threads de **uma única máquina multi-core** (16 núcleos de CPU, sem GPU), cada um com sua cópia do ambiente, calculando gradientes e aplicando-os **assincronamente**, estilo **Hogwild!** (sem lock), num conjunto de parâmetros compartilhado [P §4].
- Motivação: atores paralelos explorando partes diferentes do ambiente **descorrelacionam** os dados (papel que o replay buffer fazia no DQN), estabilizando o treino on-policy sem replay [P §4].
- **A3C**: política \(\pi(a|s;\theta)\) e valor \(V(s;\theta_v)\) (na prática compartilhando camadas: uma saída softmax e uma linear); atualização a cada \(t_{\max}\) ações ou fim de episódio, com vantagem de n-passos \(\sum_{i=0}^{k-1}\gamma^i r_{t+i} + \gamma^k V(s_{t+k}) - V(s_t)\), \(k\le t_{\max}\) [P §4].
- **Bônus de entropia**: gradiente \(\nabla\log\pi(a_t|s_t)(R_t - V(s_t)) + \beta\nabla H(\pi(s_t))\), "desencorajando convergência prematura a políticas determinísticas subótimas" [P §4]; \(\beta = 0{,}01\) "em todos os experimentos de Atari e TORCS" [P Ap. 8].
- Hiperparâmetros citados: 16 threads de ator-aprendiz numa máquina sem GPU, \(t_{\max}=5\), RMSProp com estatísticas compartilhadas, \(\gamma = 0{,}99\); a política tem uma saída softmax e uma saída linear de valor [P Ap. 8].
- Resultado: superou o estado da arte em Atari "treinando metade do tempo numa única CPU multi-core em vez de GPU" [P Abstract].
- **A2C**: a variante **síncrona** (espera todos os atores e faz uma atualização em lote) — popularizada pelas OpenAI Baselines; usada por Christiano 2017 nos jogos Atari ("advantage actor-critic (A2C; Mnih et al., 2016)") e TRPO nas tarefas de robótica [P Christiano §2.2.1].
- **Por que isso importa para LLMs (interpretação):** a ideia de "atores gerando experiência em paralelo, aprendiz central consumindo" reaparece em **RL assíncrono para LLMs** (§4.6): os "atores" agora são instâncias de vLLM/SGLang gerando rollouts, e o "Hogwild!" vira "pesos um pouco velhos" (staleness), corrigidos por IS/clipping.

### 3.10 TRPO — região de confiança, gradiente natural, gradiente conjugado

**Teoria** (Schulman et al. 2015a; base em Kakade & Langford 2002) [P §2–§3]:
- **Lema da diferença de desempenho**: \(\eta(\tilde\pi) = \eta(\pi) + \mathbb E_{\tau\sim\tilde\pi}\big[\sum_t\gamma^t A_\pi(s_t,a_t)\big]\).
- **Aproximação local** (usa os estados de \(\pi\), não de \(\tilde\pi\)): \(L_\pi(\tilde\pi) = \eta(\pi) + \sum_s\rho_\pi(s)\sum_a\tilde\pi(a|s)A_\pi(s,a)\); coincide com \(\eta\) em 1ª ordem.
- **Limite de melhoria** [P Teorema 1 e eq. 9]: com \(\epsilon = \max_{s,a}|A_\pi(s,a)|\),
  \[ \eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\,D^{\max}_{KL}(\pi,\tilde\pi), \qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2} \]
  Maximizar o lado direito a cada passo garante melhoria monotônica (*minorização–maximização*). Mas \(C\) grande → passos minúsculos.
- **Problema prático** [P eq. 14]: substituir a penalidade por uma **restrição** e o máximo por uma média:
  \[ \max_\theta\ \hat{\mathbb E}_t\Big[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat A_t\Big] \quad\text{s.a.}\quad \hat{\mathbb E}_t\big[D_{KL}(\pi_{\theta_{\text{old}}}(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t))\big]\le\delta \]
  \(\delta = 0{,}01\) em todos os experimentos [P §8].

**Solução aproximada** [P Ap. C]:
1. Aproximação linear do objetivo (\(g = \nabla_\theta L\)) e quadrática da restrição: \(D_{KL}\approx\frac12(\theta-\theta_{\text{old}})^\top F(\theta-\theta_{\text{old}})\), com \(F\) = **matriz de informação de Fisher** = Hessiana do KL em \(\theta_{\text{old}}\).
2. Direção = **gradiente natural** \(s \approx F^{-1}g\) (Kakade 2001; Amari 1998), resolvendo \(Fs = g\) por **gradiente conjugado** (CG) — só precisa de produtos \(v\mapsto Fv\), nunca de \(F\) explícita (impossível com bilhões de parâmetros). O produto Fisher-vetor é calculado analiticamente pela Hessiana do KL (Ap. C.1), sem guardar gradientes por amostra.
3. **Tamanho do passo** máximo que satisfaz a restrição quadrática: \(\delta = \frac12(\beta s)^\top F(\beta s) \Rightarrow \beta = \sqrt{2\delta / s^\top Fs}\) [P Ap. C].
4. **Busca em linha** com encolhimento exponencial de \(\beta\) até o objetivo **não-linear** melhorar e o KL **real** ficar \(\le\delta\); "sem a busca em linha, o algoritmo ocasionalmente dá passos grandes que causam degradação catastrófica" [P Ap. C].
- Diferença do gradiente natural "puro" (NPG): NPG usa um passo \(1/\lambda\) fixo; TRPO **impõe** a restrição a cada atualização, o que "melhora significativamente o desempenho em problemas maiores" [P §7].
- Exemplo 2D completo com CG em 2 iterações em A.17.
- **Por que não se usa TRPO em LLMs (interpretação apoiada pelo PPO paper)**: CG + produtos Fisher-vetor + busca em linha por iteração sobre bilhões de parâmetros, difícil de combinar com minibatches, dropout e arquiteturas com parâmetros compartilhados política/valor; PPO "emula a melhoria monotônica do TRPO" com otimização de 1ª ordem [P PPO §1–§2].

### 3.11 PPO — Proximal Policy Optimization

**Schulman, Wolski, Dhariwal, Radford, Klimov 2017** [P]:

**Razão de probabilidades**: \(\rho_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\), com \(\rho_t(\theta_{\text{old}})=1\).

**Objetivo substituto do TRPO** ("CPI", conservative policy iteration): \(L^{CPI}(\theta) = \hat{\mathbb E}_t[\rho_t(\theta)\hat A_t]\) — sem restrição, levaria a atualizações excessivas [P §3].

**Objetivo clipado** [P eq. 7]:
\[ L^{CLIP}(\theta) = \hat{\mathbb E}_t\Big[\min\big(\rho_t(\theta)\hat A_t,\ \text{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat A_t\big)\Big] \]
- \(\varepsilon\): hiperparâmetro ("digamos 0,2"); o \(\min\) torna o objetivo um **limite inferior pessimista** de \(L^{CPI}\); coincide com \(L^{CPI}\) em 1ª ordem perto de \(\rho=1\).
- **Análise por casos** (gradiente em relação a \(\rho\)):
  - \(\hat A>0\): o objetivo cresce com \(\rho\) até \(1+\varepsilon\) e depois fica **plano** (gradiente 0) — não há incentivo a aumentar ainda mais a probabilidade de uma ação boa num mesmo lote. Para \(\rho<1-\varepsilon\), o termo não-clipado é o menor → gradiente continua (corrige a ação boa que ficou improvável).
  - \(\hat A<0\): o objetivo é plano para \(\rho<1-\varepsilon\) (já reduziu o bastante) e **continua penalizando** para \(\rho>1+\varepsilon\) (a ação ruim ficou mais provável — o \(\min\) escolhe o termo não-clipado).
  - Regra: "ignoramos a mudança na razão só quando ela melhoraria o objetivo, e a incluímos quando o piora" [P §3]. Tabela numérica em A.13.
- Fig. 2 do paper: após uma iteração em Hopper, o KL da política atualizada é ~0,02 — ponto em que \(L^{CLIP}\) é máximo [P].

**Penalidade KL adaptativa** (alternativa/baseline) [P §4, eq. 8]:
\[ L^{KLPEN}(\theta) = \hat{\mathbb E}_t\big[\rho_t(\theta)\hat A_t - \beta\,D_{KL}(\pi_{\theta_{\text{old}}}(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t))\big] \]
Após cada atualização, \(d = \hat{\mathbb E}_t[D_{KL}]\): se \(d < d_{\text{targ}}/1{,}5\), \(\beta\leftarrow\beta/2\); se \(d > 1{,}5\,d_{\text{targ}}\), \(\beta\leftarrow2\beta\). Funcionou pior que o clipping nos benchmarks [P §4, Tab. 1].

**Objetivo combinado (política + valor + entropia)** [P eq. 9]:
\[ L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb E}_t\Big[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2\,S[\pi_\theta](s_t)\Big], \qquad L^{VF}_t = \big(V_\theta(s_t) - V^{\text{targ}}_t\big)^2 \]
- \(c_1\): peso da perda de valor (necessário quando política e valor compartilham parâmetros); \(c_2\): peso do **bônus de entropia** \(S\) (exploração).

**Clipping do valor** (detalhe de implementação das OpenAI Baselines, adotado em RLHF; N+ usa \(\hat\varepsilon = 0{,}2\) [P Tab. 7]):
\[ L^{VF,\text{clip}}_t = \max\Big(\big(V_\psi(s_t) - \hat R_t\big)^2,\ \big(V_{\text{old}}(s_t) + \text{clip}(V_\psi(s_t) - V_{\text{old}}(s_t), -\hat\varepsilon, \hat\varepsilon) - \hat R_t\big)^2\Big) \]
- Pessimista como o \(L^{CLIP}\): se o valor novo saiu da faixa \(\pm\hat\varepsilon\) **na direção que reduziria o erro**, o termo clipado (constante) domina e o gradiente zera (A.23).

**Algoritmo (estilo ator-crítico)** [P Alg. 1]: a cada iteração, \(N\) atores coletam \(T\) passos com \(\pi_{\theta_{\text{old}}}\); calcula-se \(\hat A_1..\hat A_T\); otimiza-se o substituto por **\(K\) épocas** com minibatches de tamanho \(M\le NT\) (Adam); \(\theta_{\text{old}}\leftarrow\theta\).

**Resultados e hiperparâmetros do paper** [P Tab. 1, 3, 5]:

| Ablação (MuJoCo, 7 ambientes, 21 runs; escore normalizado) | Escore |
|---|---|
| sem clipping nem penalidade | −0,39 |
| clipping \(\varepsilon=0{,}1\) / **0,2** / 0,3 | 0,76 / **0,82** / 0,70 |
| KL adaptativo \(d_{\text{targ}}\) = 0,003 / 0,01 / 0,03 | 0,68 / 0,74 / 0,71 |
| KL fixo \(\beta\) = 0,3 / 1 / 3 / 10 | 0,62 / 0,71 / 0,72 / 0,69 |

| Hiperparâmetro | MuJoCo (1M passos) | Atari |
|---|---|---|
| Horizonte \(T\) | 2048 | 128 |
| Adam | 3×10⁻⁴ | 2,5×10⁻⁴ × α (α anelado 1→0) |
| Épocas \(K\) | 10 | 3 |
| Minibatch | 64 | 32 × 8 |
| \(\gamma\) / \(\lambda\) | 0,99 / 0,95 | 0,99 / 0,95 |
| Atores | 1 | 8 |
| \(\varepsilon\) | 0,2 | 0,1 × α |
| \(c_1\) / \(c_2\) | — | 1 / 0,01 |

**Detalhes de código importam**: Engstrom et al. 2020 (arXiv 2005.12729, "Implementation Matters in Deep Policy Gradients") e Huang et al. 2022 ("The 37 Implementation Details of PPO", ICLR Blog Track) mostram que normalização de vantagem, clipping do valor, inicialização ortogonal, LR anelado etc. explicam boa parte do desempenho. O equivalente para RLHF são os "N+ implementation details" (Huang et al. 2024) e "Secrets of RLHF" (§4.8).

---

## 4. RLHF com PPO em detalhe

### 4.1 A linhagem do pipeline (o que cada trabalho acrescentou)

**Christiano et al. 2017 — "Deep RL from Human Preferences"** [P]
- Ambientes: Atari (A2C) e MuJoCo (TRPO); recompensa do ambiente **escondida**; o agente só recebe comparações humanas de segmentos de 1–2 s.
- Três processos **assíncronos**: (1) a política interage e é otimizada por RL contra \(\hat r\); (2) pares de segmentos são enviados a humanos; (3) \(\hat r\) é ajustado por BT aos pares [P §2.2].
- Ensemble + bootstrap, 10% de ruído uniforme, seleção de consultas por discordância (§2.9, §1.3).
- Números: com **700** consultas humanas quase iguala o RL com recompensa real no MuJoCo; com 1.400 sintéticas às vezes supera; Atari com **5.500** consultas; **mortal (backflip) do Hopper com 900 consultas em menos de uma hora**; reduz a complexidade de interação humana "em ~3 ordens de grandeza" [P §3].

**Ziegler et al. 2019 — "Fine-Tuning LMs from Human Preferences"** [P]
- GPT-2 774M; tarefas de continuação estilística (sentimento, descritividade) e sumarização (TL;DR, CNN/DM).
- Coleta "escolha a melhor de 4" → perda softmax (§2.2). RM inicializado do LM \(\rho\) com cabeça linear aleatória; normalizado para média 0 / variância 1; 1 época; LR 1,77×10⁻⁵.
- **Recompensa moldada por KL** (eq. 2): \(R(x,y) = r(x,y) - \beta\log\frac{\pi(y|x)}{\rho(y|x)}\) — "desempenha o papel de bônus de entropia, impede a política de sair da região onde \(r\) é válido e, nas tarefas de estilo, é parte da definição da tarefa (coerência e tópico)".
- **Controlador adaptativo de KL** (log-space proporcional) [P §2.2]:
  \[ e_t = \text{clip}\Big(\frac{\text{KL}(\pi_t,\rho) - \text{KL}_{\text{target}}}{\text{KL}_{\text{target}}}, -0{,}2, 0{,}2\Big), \qquad \beta_{t+1} = \beta_t(1 + K_\beta e_t), \quad K_\beta = 0{,}1 \]
  Motivo: runs com o mesmo \(\beta\) e sementes diferentes terminavam com KLs muito diferentes, difíceis de comparar. Alvos usados: ex. 8 nats (sentimento simulado), 6 e 10 nats [P §3].
- PPO2 (OpenAI Baselines), 2M episódios, \(\gamma=1\), 4 épocas PPO por lote (1 minibatch), batch 1024 (estilo) / 512 (sumarização), dropout desligado [P §2.2].
- Achado qualitativo: os sumarizadores RL viraram **"copiadores espertos"** — copiam frases inteiras do texto, variando o que pular [P §1, §3.2.1] — exemplo precoce de explorar o critério (rotuladores checavam cópia como heurística de qualidade).

**Stiennon et al. 2020 — "Learning to summarize from human feedback"** [P]
- Figura 2 do paper = o pipeline canônico: (1) coletar comparações de amostras de várias políticas; (2) treinar RM; (3) otimizar a política contra o RM com PPO (cada passo de tempo = um token BPE).
- Política inicializada do SFT no TL;DR; recompensa \(R = r_\theta(x,y) - \beta\log[\pi^{RL}_\phi(y|x)/\pi^{SFT}(y|x)]\) com **\(\beta = 0{,}05\)**; **crítico separado** inicializado do RM; RM = política = crítico em tamanho (1,3B e 6,7B).
- PPO: \(\gamma=1\), \(\lambda=0{,}95\), **4 épocas** por lote, LR linear decrescente 1,5×10⁻⁵ (1,3B) / 7×10⁻⁶ (6,7B), batch 512 (1,3B) / 256 (6,7B), **1M episódios** [P Ap. B.1].
- KL final em relação ao SFT [P Tab. 9]: varrendo \(\beta\) no 1,3B com o RM rm3 — \(\beta\) = 0,35 → **1,8 nats**; 0,10 → **3,8**; 0,07 → **9,4**; 0,05 → **19,0**; modelos finais (\(\beta=0{,}05\)): **18,0 nats** (1,3B) e **14,0 nats** (6,7B). Para best-of-N, a mesma tabela dá KL(BoN, sup) = 1,2 (N=8), 3,2 (N=63/64), 3,9 (N=128) — batendo com \(\log N - (N-1)/N\) = 1,20 / 3,17 / 3,86 (cálculo).
- Custo: **~320 GPU-dias** para o RL do 6,7B; a coleta de dados levou "milhares de horas de rotuladores" [P §5].
- Resultado: políticas com feedback humano preferidas a políticas supervisionadas muito maiores e às referências humanas [P §4.1].

**Ouyang et al. 2022 — InstructGPT** [P] — os **três passos** (Fig. 2 do paper):
1. **SFT** em demonstrações escritas por rotuladores: 16 épocas, cosseno, *residual dropout* 0,2; LR 9,65×10⁻⁶ (1,3B/6B, batch 32) e 5,03×10⁻⁶ (175B, batch 8); seleção do checkpoint **pelo escore do RM** na validação — o SFT sobreajusta a perda de validação após 1 época, mas mais épocas melhoram o escore do RM e a preferência humana [P §3.5, Ap. C.1].
2. **RM** de 6B por ranking de \(K=4..9\) (§2.2, §2.4).
3. **PPO** contra o RM, com KL por token até o SFT; função valor inicializada do RM. A política RL parte de um SFT especial: 2 épocas de SFT **com 10% de dados de pré-treino misturados** ("ajuda o PPO"; entre 0%, 10% e 50%, só 10% se destacou) [P Ap. C.3, E.8].

| Conjunto | Treino | Validação | Fonte [P Tab. 6] |
|---|---|---|---|
| SFT | 12.725 prompts (11.295 rotulador + 1.430 cliente) | 1.653 | demonstrações |
| RM | 33.207 prompts (6.623 + 26.584) | 17.887 | rankings \(K=4..9\) |
| PPO | 31.144 prompts (só API) | 16.185 | sem rótulo humano |

Hiperparâmetros do PPO [P Ap. C.4]: \(\beta = 0{,}02\); **256k episódios** (~31k prompts únicos); batch 512, minibatch 64 → 8 minibatches, **1 época interna**; LR constante com warmup de 10 iterações a partir de 1/10 do pico; **EMA dos pesos** com decaimento 0,992; GAE **sem desconto**; clip 0,2; **temperatura de amostragem 1**; RM e crítico de 6B para todas as políticas; LR do crítico 9×10⁻⁶ (1,3B/6B) e 5×10⁻⁶ (175B). Contexto 2k tokens; prompts >1k tokens filtrados; respostas até 1k tokens; pesos e ativações fp16 com cópias mestras fp32; Adam \(\beta_1=0{,}9,\ \beta_2=0{,}95\) [P Ap. C]. Ablações: batch 512 foi o melhor entre 64–1024 (avaliação humana); KL ótimo "por volta de 0,01 e 0,02" (0 e 2 são ruins); treinar além de 256k episódios não ajudou no 1,3B; treinar 512k episódios fez DROP/SQuADv2 caírem abaixo do GPT-3 [P Ap. E.6, E.7, E.11].

Resultado-manchete: saídas do **InstructGPT 1,3B são preferidas às do GPT-3 175B** (100× mais parâmetros) [P Abstract]. Custo: SFT 175B = **4,9 petaflop/s-dias**; PPO-ptx 175B = **60 petaflop/s-dias**; pré-treino do GPT-3 = **3.640** [P §5.1].

**Bai et al. 2022 — HH-RLHF** [P]
- Sem SFT clássico (usa *context distillation*); preferências de helpfulness e harmlessness; RLHF **online iterado** semanal.
- \(\lambda_{KL} = 0{,}001\) ("muito pequeno, provavelmente impacto menor durante a maior parte do treino, já que \(D_{KL}<100\) tipicamente, e talvez totalmente desnecessário"); clip 0,2; \(\gamma=1\); **sem bônus de entropia**; \(K\) = 1 (scan), 2 (robustez), 4 (online) épocas por lote — "\(K\) maior tipicamente leva a resultados mais estáveis" [P §4.1, Ap. B.1].
- Relação linear recompensa ∝ \(\sqrt{KL}\); robustez até ~150k amostras; tensão helpfulness × harmlessness: treinar só em um dos dois dá desempenho "significativamente pior que o acaso" no outro [P §5.1.1].

**Touvron et al. 2023 — Llama 2-Chat** [P §3.2–3.2.3]
- **Dois RMs** (helpfulness e safety), margem de preferência (§2.3), RLHF iterativo (V1…V5) combinando **rejection sampling fine-tuning** (amplitude: \(K\) amostras por prompt, fica-se com a melhor segundo o RM e faz-se SFT nela) e **PPO** (profundidade: cada passo parte da política atualizada) [P §3.2.3].
- Recompensa do PPO: \(R(g|p) = \tilde R_c(g|p) - \beta D_{KL}(\pi_\theta(g|p)\|\pi_0(g|p))\), com combinação por partes: usa \(R_s\) se o prompt é marcado como de segurança **ou** \(R_s < 0{,}15\) (precisão 0,89 / recall 0,55 no teste de segurança), senão \(R_h\); e **\(\tilde R_c = \text{whiten}(\text{logit}(R_c))\)** — "reverter a sigmoide com o logit e branquear para aumentar a estabilidade e balancear com o termo KL".
- AdamW (0,9 / 0,95, eps 10⁻⁵), weight decay 0,1, **clipping de gradiente 1,0**, LR constante 10⁻⁶; batch 512, clip 0,2, minibatch 64, **um passo de gradiente por minibatch**; \(\beta=0{,}01\) (7B/13B) e **0,005** (34B/70B); 200–400 iterações com early stopping em prompts held-out.
- Sistemas: **≈330 s por iteração de PPO no 70B**; FSDP foi eficiente para forward/backward, mas **gerar com FSDP ficou ≈20× mais lento** mesmo com batch grande e KV cache; mitigação: consolidar os pesos em cada nó antes da geração e liberá-los depois [P §3.2.3]. (Esse é o argumento histórico para colocar um motor de inferência dedicado no laço — §4.5.)

### 4.2 A recompensa moldada por KL

**Forma por sequência** (Stiennon, InstructGPT):
\[ R(x,y) = r_\phi(x,y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \]
- \(r_\phi\): escore do RM (normalizado); \(\beta\): coeficiente; \(\pi_{\text{ref}}\): SFT congelado; \(\log\pi(y|x) = \sum_t\log\pi(y_t|s_t)\).
- **Em esperança**: \(\mathbb E_{y\sim\pi_\theta}[R] = \mathbb E[r_\phi] - \beta\,D_{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)\) (porque \(\mathbb E_{y\sim\pi}[\log\pi/\pi_{\text{ref}}]\) é a definição de KL).

**Decomposição por token** (como se implementa no PPO; InstructGPT "penalidade KL por token" [P §3.5]; DeepSeekMath eq. 2; N+ Fig. 9):
\[ r_t = -\beta\big(\log\pi_{\theta_{\text{old}}}(y_t|s_t) - \log\pi_{\text{ref}}(y_t|s_t)\big) + \mathbb 1[t = T]\;r_\phi(x,y) \]
- O escore do RM entra **só no último token** (EOS); todos os tokens recebem a penalidade KL por token. Com \(\gamma=1\), \(\sum_t r_t = R(x,y)\) (A.11). N+ mostra um exemplo real: "4,5000 é o escore do RM no token EOS; os demais números das recompensas são a penalidade KL por token" [P Detalhe 22].

**Por que a KL** (três leituras, todas textuais):
1. **Bônus de entropia**: \(D_{KL}(\pi\|\pi_{\text{ref}}) = -H(\pi) - \mathbb E_\pi[\log\pi_{\text{ref}}]\), então \(-\beta D_{KL} = \beta H(\pi) + \beta\,\mathbb E_\pi[\log\pi_{\text{ref}}]\) — "encoraja a política a explorar e a não colapsar num único modo" [P Stiennon §3.4].
2. **Manter o RM válido**: "garante que a política não aprenda a produzir saídas muito diferentes das que o RM viu no treino" [P Stiennon §3.4] — ataca o Goodhart extremal (§2.8).
3. **Parte da definição da tarefa** (coerência/tópico em tarefas de estilo) [P Ziegler §2].
- Contraponto: Gao 2022 achou a KL ≈ early stopping no seu setup (§2.8); Bai usou KL quase nula.

**A política ótima (fechada)** — derivação (variacional de Gibbs; base do DPO, tema de outro capítulo):
\[ \max_\pi\ \mathbb E_{y\sim\pi}[r(x,y)] - \beta D_{KL}(\pi\|\pi_{\text{ref}}) = \max_\pi\ -\beta\,D_{KL}\Big(\pi\ \Big\|\ \tfrac{1}{Z(x)}\pi_{\text{ref}}\,e^{r/\beta}\Big) + \beta\log Z(x) \]
\[ \Rightarrow\quad \pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\big(r(x,y)/\beta\big), \qquad Z(x) = \sum_y \pi_{\text{ref}}(y|x)e^{r(x,y)/\beta} \]
- Leitura: o RLHF com KL **repondera** o modelo de referência exponencialmente pela recompensa; \(\beta\) é a temperatura. \(\beta\to0\): só a recompensa importa (hacking); \(\beta\to\infty\): fica em \(\pi_{\text{ref}}\). Best-of-n é outra forma (discreta) de mover \(\pi_{\text{ref}}\) em direção a \(\pi^*\), com KL \(=\log n - (n-1)/n\) (§2.8).

**Estimadores de KL a partir de amostras** (Schulman, "Approximating KL Divergence", 2020) [W blog]:
Com amostras \(y_t\sim\pi_\theta\) (o "\(q\)" do blog) e \(r = \pi_{\text{ref}}(y_t|s_t)/\pi_\theta(y_t|s_t)\):
\[ k_1 = -\log r, \qquad k_2 = \tfrac12(\log r)^2, \qquad k_3 = (r - 1) - \log r \]
- \(k_1\): não-enviesado, variância alta, **pode ser negativo** (é o termo usado na recompensa por token acima).
- \(k_2\): enviesado mas de baixa variância; sua esperança é uma f-divergência que coincide com o KL até 2ª ordem.
- \(k_3\): **não-enviesado e sempre ≥ 0** (distância entre \(\log\) e sua tangente — uma divergência de Bregman); o blog mede, para \(q=\mathcal N(0,1)\), \(p=\mathcal N(1,1)\) (KL verdadeiro 0,5): desvio-padrão/verdadeiro = 2 (\(k_1\)), 1,73 (\(k_2\), viés 25%), 1,7 (\(k_3\), sem viés). É o estimador que o GRPO usa **na perda** (DeepSeekMath eq. 4, "garantido positivo") [P].
- Exemplos numéricos por token em A.15.
- **KL na recompensa vs KL na perda (interpretação)**: na recompensa (InstructGPT/PPO), a penalidade passa pela vantagem (é "bootstrapped" pelo crítico, sofre baseline e GAE); na perda (GRPO), é diferenciada diretamente como regularizador. Não são equivalentes em gradiente; em esperança ambas puxam para \(\pi_{\text{ref}}\).

**Valores de \(\beta\) na literatura** (todos [P]):

| Trabalho | \(\beta\) | Observação |
|---|---|---|
| Ziegler 2019 | adaptativo, \(K_\beta=0{,}1\) | alvos de 6–10 nats |
| Stiennon 2020 | 0,05 | KL final 14–19 nats |
| InstructGPT 2022 | 0,02 | ótimo entre 0,01 e 0,02 |
| Bai 2022 | 0,001 | "talvez desnecessário" |
| Llama 2 2023 | 0,01 (7B/13B), 0,005 (34B/70B) | com recompensa `whiten(logit)` |
| N+ 2024 (Pythia) | 0,05 | reprodução do Stiennon |
| HelpSteer2 (DPO) | "KL penalty" 1×10⁻³ | é o \(\beta\) do DPO, não do PPO [P §4.3] |

### 4.3 Os quatro modelos em memória

| Modelo | Papel | Inicialização | Treinável? | Forwards/backwards por iteração | Precisão típica |
|---|---|---|---|---|---|
| **Política (ator)** \(\pi_\theta\) | gera rollouts; recebe o gradiente PPO | SFT | **sim** | geração (no motor de inferência) + forward de log-probs \(\pi_{\theta_{\text{old}}}\) + \(K\) épocas × (forward+backward) | pesos bf16/fp16 + mestre fp32 + Adam fp32 |
| **Referência** \(\pi_{\text{ref}}\) | log-probs para a KL | SFT (cópia congelada) | não | 1 forward por rollout | bf16 (pode ser quantizada/offload) |
| **Modelo de recompensa** \(r_\phi\) | escore no EOS | RM treinado (§2) | não | 1 forward por rollout | bf16 |
| **Crítico (valor)** \(V_\psi\) | \(V(s_t)\) em cada token para GAE | **RM** (Stiennon, InstructGPT, N+) | **sim** | 1 forward (\(V_{\text{old}}\)) + \(K\) épocas × (forward+backward) | como a política |

**Orçamento de memória (cálculo)**:
- Modelo **treinado** com Adam em precisão mista: ≈ **16 bytes/parâmetro** = 2 (pesos fp16/bf16) + 2 (gradientes) + 12 (cópia mestre fp32 4 + momento 4 + variância 4) — contabilidade do ZeRO (Rajbhandari et al. 2019, arXiv 1910.02054). **Sem** ativações, buffers de comunicação e fragmentação.
- Modelo **congelado** em bf16: 2 bytes/parâmetro.
- Total para política e crítico de \(N\) parâmetros e ref + RM de \(N\): \(2\times16N + 2\times2N = 36N\) bytes.

| \(N\) (cada modelo) | 1 treinável | 1 congelado (bf16) | 4 modelos (36N) |
|---|---|---|---|
| 1,5B | 24 GB | 3 GB | **54 GB** (50,3 GiB) |
| 3B | 48 GB | 6 GB | **108 GB** (100,6 GiB) |
| 7B | 112 GB | 14 GB | **252 GB** (234,7 GiB) |
| 8B | 128 GB | 16 GB | **288 GB** (268,2 GiB) |

- **E ainda o motor de inferência**: no modo colocado, o vLLM/SGLang mantém **outra** cópia dos pesos da política (formato de inferência) + **KV cache** para o batch de rollouts — é o que o "3D-HybridEngine" do verl e o sleep mode do vLLM tentam não duplicar (§4.5).
- **Ponte com o Módulo 1 (interpretação + cálculo)**: um DGX Spark expõe ~121,69 GiB de memória unificada ao CUDA [M1: quant.md]; dois somam ~243 GiB. PPO com **full fine-tuning** de política+crítico de 7B (≈235 GiB só de estados) já não cabe com ativações e KV; 3B (≈101 GiB) cabe em um nó só para os estados. Saídas usuais: crítico/RM menores que a política (InstructGPT usou RM/crítico 6B para política 175B), **LoRA** (a referência vira "a mesma rede com o adaptador desligado", sem cópia extra — prática comum em bibliotecas como TRL/PEFT; verificar a versão), offload de Adam para CPU — que **no GB10 não libera memória física**, pois CPU e GPU compartilham a mesma LPDDR5x [M1: cap. `memoria-unificada`] —, ou abandonar o crítico (RLOO/GRPO, §3.4).
- **MoE** (os três modelos do Módulo 1): a memória escala com os parâmetros **totais**, não os ativos; o custo de FLOPs escala com os ativos (interpretação padrão).

### 4.4 Uma iteração de PPO-RLHF, passo a passo

```text
para cada iteração:
  1. prompts  ← amostra B prompts do dataset de RL (InstructGPT B=512; N+ B=512)
  2. rollouts ← motor_inferência.gerar(prompts, π_θold, temperatura T)          # vLLM/SGLang
                 └─ N+: amostra sempre até o máximo e trunca no EOS ("EOS trick")
  3. escore   ← r_φ(x, y) lido no token EOS; se não houver EOS → recompensa −1 (N+ Detalhe 23)
  4. logp_ref ← π_ref.forward(x, y)          # por token
  5. logp_old ← π_θold.forward(x, y)         # recomputado no motor de TREINO (não usar os logprobs do vLLM sem correção; §4.6)
  6. V_old    ← V_ψ.forward(x, y)            # por token
  7. r_t      ← −β (logp_old − logp_ref);  r_T += escore          # (§4.2)
  8. Â_t, R̂_t ← GAE(r, V_old, γ, λ);  R̂_t = Â_t + V_old(s_t)
  9. Â_t      ← whiten(Â)                    # média 0, desvio 1 no lote (N+ Detalhe 25)
 10. para época = 1..K; para minibatch em embaralhar(lote):
        ρ_t   = exp(logπ_θ(y_t|s_t) − logp_old_t)
        L_π   = −mean(min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t))
        L_V   = ½ mean(max((V_ψ − R̂)², (V_old + clip(V_ψ − V_old, ±ε̂) − R̂)²))
        L     = L_π + c1·L_V − c2·H[π_θ]   (+ γ_ptx · L_pretrain se PPO-ptx)
        passo do otimizador (clip de norma global do gradiente)
 11. sincroniza os pesos novos π_θ → motor de inferência; θ_old ← θ
```

- Ordem de forwards e quem roda onde: passos 3–6 podem ser feitos em paralelo por modelos em GPUs diferentes (verl coloca ref e RM no mesmo conjunto de GPUs e executa sequencialmente [P HybridFlow Fig. 3]).
- Contagem de "episódios": InstructGPT 256k episódios = 500 iterações de 512 (cálculo); Stiennon 1M; N+ 1M (≈8,56 épocas do dataset SFT de 116.722 prompts, reamostrado sem reposição) [P N+ Detalhe 21].
- Hiperparâmetros N+ (Pythia, reprodução de Stiennon) [P Tab. 7]: 1M episódios; AdamW (eps 1e-5, lr 3e-6), linear; batch 512; \(\beta=0{,}05\); \(\gamma=1\); \(\lambda=0{,}95\); 1 minibatch; \(K=4\); \(\varepsilon=0{,}2\); \(\hat\varepsilon=0{,}2\); \(c_1 = 0{,}1\); clipping da perda de valor ligado; temperatura de amostragem 0,7.

### 4.5 O motor de inferência dentro do laço

**Por que.** A geração autoregressiva domina o tempo de parede:
- Llama 2: gerar com FSDP ≈ **20× mais lento** (§4.1).
- OpenRLHF: "a fase de inferência frequentemente responde por **mais de 90%** do tempo total de RLHF (ou RLVR)" [P §1].
- HybridFlow: ao discutir o "desequilíbrio entre treino e geração do ator", cita uma fase que "frequentemente representa a maior parte da carga de cada iteração (ex.: 58,9% do tempo total de RLHF com HybridFlow)" [P §2] — o texto extraído do PDF intercala a Fig. 3 no meio da frase; a leitura natural é que se refere à geração (ver Apêndice C).
- Por isso os frameworks usam **vLLM** ou **SGLang** (PagedAttention, batching contínuo, CUDA graphs — todo o Módulo 1) só para os rollouts, e um motor de treino (DeepSpeed ZeRO, FSDP, Megatron) para os gradientes. Frameworks que usam vLLM para rollouts, segundo a documentação do vLLM [D]: Cosmos-RL, ms-swift, NeMo-RL, Open Instruct, OpenRLHF, PipelineRL, Prime-RL, SkyRL, TRL, Unsloth e verl. slime usa **SGLang + Megatron** e declara ser "o framework de RL por trás de GLM-5.3, GLM-5.2, GLM-5.1, GLM-5, GLM-4.7, GLM-4.6 e GLM-4.5" [D README]; **vime** (jun/2026) traz o desenho do slime para o ecossistema vLLM (Megatron + vLLM + Data Buffer; colocado e desagregado; GRPO e PPO com exemplos) [W blog vLLM].

**Dois arranjos:**

| | **Colocado** (*hybrid engine*) | **Desagregado** |
|---|---|---|
| GPUs | as mesmas GPUs alternam treino ↔ geração | pools separados de treino e de rollout |
| Sincronização de pesos | em memória: tensores compartilhados / handles CUDA IPC (SGLang `update_weights_from_tensor`; vLLM backend `ipc`) | rede: broadcast NCCL/IB do treino para o rollout (SGLang `update_weights_from_distributed`; vLLM backend `nccl`) ou checkpoint em disco |
| Memória | precisa liberar o KV/pesos do motor de inferência durante o treino (vLLM *sleep mode*; SGLang `release_memory_occupation`) e o otimizador durante a geração | sem disputa; cada lado dimensionado à parte |
| Ociosidade | nenhuma GPU parada, mas fases serializadas | um lado espera o outro, a menos que se use **RL assíncrono** |
| Exemplos | DeepSpeed-Chat Hybrid Engine; verl 3D-HybridEngine; OpenRLHF modo híbrido | OpenRLHF (Ray, rollout engines vs actor engines); slime modo desagregado; Noukhovitch et al. |

- **verl / HybridFlow** [P]: combina paradigma *single-controller* (orquestração) e *multi-controller* (computação); o **3D-HybridEngine** refaz o particionamento (*resharding*) do ator entre o paralelismo de treino e o de geração "com zero redundância de memória" — guarda pesos de treino e de geração em buffers separados e **offload dos pesos de geração para a CPU durante o treino**; 1,53×–20,57× de throughput vs baselines.
- **OpenRLHF** [P §3]: Ray agenda papéis de "rollout engines" (vLLM) e "actor engines" (DeepSpeed ZeRO + AutoTP + Ring Attention); os pesos fatiados são transferidos ao vLLM; suporta modo híbrido e fluxo **assíncrono**; 1,22×–1,68× de speedup.

**APIs do vLLM para RL** (versão `latest` em 2026-10) [D]:
- **Sleep mode** (`--enable-sleep-mode`): `llm.sleep(level=1)` descarrega pesos para RAM da CPU e descarta o KV cache; `llm.sleep(level=2)` descarta pesos e KV (útil "quando se vai atualizar os pesos, como em RLHF"); `llm.wake_up()` ou `llm.wake_up(tags=["weights"])` / `tags=["kv_cache"]` para acordar em etapas e reduzir o pico de memória durante a sincronização; `llm.collective_rpc("reload_weights")`; `llm.release_kv_cache_memory()`. No servidor: `POST /sleep?level=1`, `POST /wake_up`, `POST /is_sleeping`, `POST /collective_rpc`, `POST /release_kv_cache_memory` — **só com `VLLM_SERVER_DEV_MODE=1`**.
- **Transferência de pesos**: `WeightTransferConfig(backend=...)` com backends `"nccl"`, `"ipc"`, `"sparse_nccl"`, `"sharded_rdt"`, `"nccl_m2n"`; protocolo de 4 fases `init_weight_transfer_engine` → `start_weight_update` → `update_weights` (repetível, em blocos) → `finish_weight_update`; no treinador, `WeightTransferTrainerFactory.trainer_init(init_info=NCCLTrainerInitInfo(...), client=HTTPVLLMWeightSyncClient(url), source=ModuleSource(model))` e `engine.send_weights()` por passo; endpoints HTTP `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/pause`, `/resume`, `/weight_info` (também sob `VLLM_SERVER_DEV_MODE=1`). A página de RLHF cita ainda um "Weight Checker" e uma API de pausa/retomada para atualizar pesos "em pleno voo" em RL assíncrono.

**APIs do SGLang para RL** [D "SGLang for RL Systems"]:
- `update_weights_from_disk` (escala elástica/checkpoint), `update_weights_from_tensor` (colocado; tensores compartilhados), `update_weights_from_distributed` (NCCL/IB; "evita I/O de disco e mantém treino e rollout desacoplados"), com `init_weights_update_group` / `destroy_weights_update_group`; ambos dentro de `begin_weight_update()` … `end_weight_update()` (o pós-processamento de quantização roda uma vez por atualização, não por bloco).
- `release_memory_occupation` / `resume_memory_occupation` (KV e/ou pesos; evita recarregar do disco e recapturar CUDA graphs a cada passo).
- `pause_generation` (modos `abort`, `retract` — devolve requisições à fila —, `in_place` — preserva o KV) e `continue_generation`: o fluxo é pausar → atualizar pesos → continuar.
- `--enable-deterministic-inference`: reduz não-determinismo entre formatos de batch; "on-policy verdadeiro exige kernels de treino correspondentes".

**Custo da sincronização (cálculo, com números do Módulo 1)**: pesos bf16 de 7B = 14 GB. Pela ConnectX-7 do GB10, uma função PCIe ≈ 15,7 GB/s (≈126 Gb/s); a porta de 200 Gb/s só com as duas PFs (≈25 GB/s) [M1: ds-launch.md]. Broadcast de 14 GB ≈ **0,9 s** (1 PF) ou **≈0,56 s** (2 PFs), sem overhead de protocolo. No modo colocado em memória unificada, os pesos já estão na mesma LPDDR5x — a sincronização pode ser só troca de handles/cópia local (interpretação).

**Ponte com a receita GLM (Módulo 1)**: o overlay `patch_cache_reset.py` monta, com `GLM53_EXPOSE_CACHE_RESET=1`, **só** `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache` — "sem `/sleep`, `/rlhf`, `/rpc`…", que `VLLM_SERVER_DEV_MODE=1` montaria [M1: glm-mods.md M41; glm-launch.md]. (interpretação) Uma receita de **serving** fecha deliberadamente as rotas de controle que um sistema de **RL** precisa abrir; transformar a receita em gerador de rollouts exigiria o modo dev (com o cuidado de que essas rotas raiz ficam fora do guard Bearer).

**RL assíncrono** — Noukhovitch et al. 2024 (ICLR 2025) [P]:
- Separar geração (vLLM em GPUs dedicadas) e aprendizado (outras GPUs), estilo Cleanba; gera o lote \(i+1\) enquanto treina no lote \(i\) → treino **off-policy de 1 passo**.
- Pergunta central: "quanta off-policyness podemos tolerar?"; desempenho em geral **degrada** com mais off-policyness; entre as perdas testadas, **Online DPO é a mais robusta** a dados off-policy, e a robustez cresce com o tamanho da política.
- Resultados: mesmo desempenho do síncrono **~25% mais rápido** em 2,8B; chatbot LLaMA 3.1 8B **~40% mais rápido**; Rho 1B no GSM8k **~70% mais rápido** com mesma acurácia.
- slime oferece "fully-async rollout" e sincronização delta de pesos para desagregação [D]; GLM-4.5 descreve rollout exposto diretamente ao ambiente de agente com GPUs de treino e inferência agendadas independentemente [W/arXiv 2508.06471].

### 4.6 O descompasso treino–inferência ("mesmos pesos, outras probabilidades")

- **O problema** (Yao, Liu et al. 2025, workshop OPT/NeurIPS 2025) [W/workshop]: com geração em vLLM e treino em FSDP, \(\pi_{\text{vllm}}(a;\theta)\ne\pi_{\text{fsdp}}(a;\theta)\) apesar dos mesmos parâmetros — kernels, ordem de redução, precisão e batching diferentes. Isso "transforma implicitamente o RL on-policy em off-policy". Piora com **rollouts quantizados (INT8, FP8)**.
- **Correção TIS** (REINFORCE):
  \[ \mathbb E_{a\sim\pi_{\text{vllm}}(\theta)}\Big[\min\Big(\frac{\pi_{\text{fsdp}}(a,\theta)}{\pi_{\text{vllm}}(a,\theta)},\,C\Big)\,R(a)\,\nabla_\theta\log\pi_{\text{fsdp}}(a,\theta)\Big] \]
  e no PPO o fator \(\min(\pi_{\text{fsdp}}(a,\theta_{\text{old}})/\pi_{\text{vllm}}(a,\theta_{\text{old}}), C)\) multiplica o gradiente clipado usual (a razão do PPO continua entre \(\pi_{\text{fsdp}}(\theta)\) e \(\pi_{\text{fsdp}}(\theta_{\text{old}})\)). \(C\) = razão máxima (hiperparâmetro). Alternativas piores na ablação: "PPO-IS" (usar \(\pi_{\text{vllm}}\) no denominador da própria razão clipada) e "Vanilla-IS" (sem truncar) — ambas ~0 de acurácia com rollouts INT8 no Qwen2.5-0.5B/GSM8K. Ganhos relevantes no DAPO-32B (Qwen2.5-32B, AIME 2024).
- **Cabeça de LM em FP32 (MiniMax-M1)** [W/arXiv 2506.13585]: a análise camada a camada apontou **ativações de alta magnitude na LM head** como fonte principal do descompasso; subir a LM head para FP32 elevou a correlação entre as probabilidades de treino e inferência de **~0,9 para ~0,99**; o problema não aparecia em modelos densos menores com atenção softmax.
- **Kernels batch-invariantes (Thinking Machines, set/2025)** [W blog]: a não-reprodutibilidade vem de os kernels (RMSNorm, matmul, atenção) mudarem a ordem de redução com o tamanho do batch; com kernels batch-invariantes, 1.000 execuções do mesmo prompt dão 1.000 saídas idênticas (antes: 80 completions distintas no Qwen3-8B a temperatura 0). Em RLVR (Qwen 2.5-VL 8B, Bigmath), sem correção a KL amostrador–treinador dispara junto com o colapso da recompensa; com IS fica ~0,001; com **"True On-Policy RL"** (numérica bit-idêntica entre amostragem e treino) a KL fica **0**. Custo medido: 55 s (vLLM determinístico não otimizado) vs 26 s (vLLM padrão) no teste deles.
- **MoE: R3 — Rollout Routing Replay** (arXiv 2510.11370) [W/arXiv]: em MoE, a KL treino–inferência é **mais que o dobro** da de modelos densos, e **~94% dos tokens** escolhem experts diferentes em pelo menos uma camada entre as duas fases; R3 **grava as máscaras de roteamento top-K da inferência e as reaplica no treino**, estabilizando o RL e superando GSPO e TIS nos experimentos dos autores. O blog do vime reporta diferença de logprob no MoE caindo de 0,019 para 0,013 com R3, e Qwen3-4B estável em ~0,011 (vs deriva até 0,77 no baseline) [W].
- **Ponte direta com o Módulo 1**:
  - A receita Qwen tem um overlay de **determinismo** (`VLLM_QSA_DET_TOPK=1` ordena os índices top-k do QSA; `VLLM_MOE_DET_FINALIZE=1` + `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0` usa o *finalize* não-fundido do MoE): spread de logprob de prompt **0,25 / 6,07 nats → 0 / 0**; saídas greedy distintas 5/5 → 1/5; custo ~−4% de decode [M1: qwen-launch.md #12]. É exatamente a classe de correção que "on-policy verdadeiro" exige.
  - O vLLM tem `VLLM_BATCH_INVARIANT` [M1: vllm-ref.md].
  - A receita GLM traz `logprob_capture.py`, que compara KL/argmax de logprobs entre dois servidores [M1: glm-launch.md] — a mesma métrica que se monitora entre amostrador e treinador.
  - Os três modelos do Módulo 1 são **MoE** com KV/pesos quantizados (EXL3, NVFP4, MXFP4, KV FP8) [M1: quant.md]: usá-los como geradores de rollout de uma política treinada em BF16 é o pior caso de descompasso (quantização + roteamento) → TIS/R3/LM-head FP32 seriam obrigatórios (interpretação).
  - **Decodificação especulativa** (MTP/EAGLE/DFlash, caps. 35–37 do M1) preserva a distribuição de amostragem em aritmética exata (Leviathan et al. 2023; Chen et al. 2023), mas a numérica do verificador em batch maior muda os logprobs — mais uma fonte de descompasso (interpretação; não verificado em paper de RL).

### 4.7 Custo computacional

**Modelo de FLOPs (cálculo)** — com a regra \(\approx 2N\) FLOPs/token por forward e \(\approx 6N\) por forward+backward (Kaplan et al. 2020, arXiv 2001.08361), ignorando atenção. Exemplo ilustrativo: \(N = 7\)B para os 4 modelos, \(B = 512\) prompts de 256 tokens + 256 tokens de resposta (262.144 tokens por iteração), \(K=1\):

| Fase | FLOPs | Conta |
|---|---|---|
| Geração (prefill + decode) | 3,67 PFLOP | \(2N\times262.144\) |
| Forwards de pontuação (\(\pi_{\text{old}}\), \(\pi_{\text{ref}}\), RM, \(V_{\text{old}}\)) | 14,68 PFLOP | \(4\times2N\times262.144\) |
| Treino política + crítico (por época) | 22,02 PFLOP | \(2\times6N\times262.144\) |
| **Total (K=1)** | **40,37 PFLOP** | |

- Em FLOPs a geração é a **menor** parcela; em **tempo** ela domina, porque o decode é limitado por banda (lê todos os pesos a cada passo) [M1: cap. `prefill-decode-kv`, roofline em quant.md A.14]. Ex. GB10: pesos bf16 de 7B = 14 GB → ≥ 51 ms por passo de decode a 273 GB/s → ≥ 13 s para 256 passos, **se** os 512 rollouts couberem num só batch de KV (cálculo). Na mesma máquina, os 40,37 PFLOP a ~125 TFLOP/s bf16 densos (classe citada para o GB10 [M1: quant.md]) dariam ≥ 323 s a 100% de utilização (cálculo).
- **Números reais**: InstructGPT 175B PPO-ptx = 60 pf/s-dias (vs 3.640 do pré-treino); Stiennon 6,7B ≈ 320 GPU-dias; Llama 2 70B ≈ 330 s/iteração [P].
- **Por que PPO é "caro" (resumo)**: 2 modelos treináveis (memória de otimizador ×2), 4 forwards por rollout, \(K\) épocas, e um gargalo serial de geração; daí a busca por DPO (sem RL), RLOO/GRPO (sem crítico) e RL assíncrono (sem bolhas).

### 4.8 Instabilidades conhecidas e truques

| Truque | O que faz | Fonte |
|---|---|---|
| **Desligar dropout** | com dropout, os log-probs não são reprodutíveis → a KL fica não-confiável e as razões do PPO ≠ 1 já na 1ª época | N+ Detalhe 7 [P]; Ziegler [P] |
| **EOS ≠ PAD** | se PAD = EOS, o EOS é mascarado no SFT e o modelo não aprende a parar → amplifica o viés de comprimento | N+ Detalhe 3 [P] |
| **"EOS trick"** | amostrar comprimento fixo, truncar no EOS; respostas sem EOS recebem **−1** | N+ Detalhe 23 [P] |
| **Normalizar o RM** (bias) | média 0 nas referências/demonstrações | Stiennon, InstructGPT, N+ Detalhe 15 [P] |
| **Branquear/escalar a recompensa** | `whiten(rewards, shift_mean=False)` por minibatch (opcional); Llama 2: `whiten(logit(R_c))` | N+ Detalhe 24; Llama 2 [P] |
| **Normalizar e clipar a recompensa** | \(\tilde r = \text{clip}\big((r - \bar r)/\sigma(r), -\delta, \delta\big)\) com média/variância históricas | Secrets eq. 18 [P] |
| **Branquear a vantagem** | \(\hat A\leftarrow(\hat A - \text{média})/\text{desvio}\) no lote | N+ Detalhe 25; Engstrom 2020 [P] |
| **Controle de KL** | fixo (\(\beta\) da tabela §4.2) ou adaptativo (Ziegler; PPO §4) | [P] |
| **Clipping do valor** | \(\hat\varepsilon = 0{,}2\) | N+ Tab. 7; Secrets PPO-max [P] |
| **Crítico inicializado do RM** | "melhora muito os gradientes iniciais da política e reduz deriva / alignment tax" | Stiennon; InstructGPT; N+ Detalhe 22 [P] |
| **Pré-treinar o crítico** | ajustar \(V\) antes de começar a mexer na política | Secrets PPO-max; VC-PPO [P/W] |
| **Política e crítico separados** | evita que a perda de valor destrua a política no início | Stiennon §3.4 / Ap. G.1 [P] |
| **Clipping global de gradiente** | ex. norma 1,0 | Llama 2; Secrets [P] |
| **LR pequeno e constante/linear** | 10⁻⁶ (Llama 2), 3×10⁻⁶ (N+), 9×10⁻⁶ crítico (InstructGPT) | [P] |
| **Poucas épocas por lote** | InstructGPT 1 época interna; Stiennon/N+ 4; Bai 1/2/4 | [P] |
| **EMA dos pesos** | decaimento 0,992 | InstructGPT [P] |
| **Mistura de pré-treino (ptx)** | §4.9 | InstructGPT; Secrets [P] |
| **Buffer de experiência pequeno** | dados quase on-policy | Secrets PPO-max [P] |
| **Monitorar métricas de "espaço de ação"** | perplexidade, comprimento da resposta, KL até o SFT — "mais informativas da estabilidade do que recompensa e perdas" | Secrets §5 [P] |
| **Sem bônus de entropia** em LLMs | Bai não usa; a KL já age como entropia | Bai [P]; Stiennon §3.4 |
| **Seeds e early stopping** | N+ roda 4 sementes e relata falhas; Llama 2 para cedo em held-out | [P] |

**Sintomas típicos (interpretação a partir das fontes)**: recompensa sobe e qualidade humana cai (superotimização/colapso de padrão — Secrets, Stiennon); comprimento explode (Singhal); KL dispara (β pequeno ou LR alto); a razão média do PPO se afasta de 1 já na 1ª época (dropout ligado, ou descompasso amostrador–treinador §4.6); perda de valor enorme no começo (crítico não inicializado do RM / não pré-treinado).

### 4.9 PPO-ptx (mistura de gradientes de pré-treino)

**Objetivo do InstructGPT** [P eq. 2]:
\[ \text{objective}(\phi) = \mathbb E_{(x,y)\sim D_{\pi^{RL}_\phi}}\Big[r_\theta(x,y) - \beta\log\frac{\pi^{RL}_\phi(y|x)}{\pi^{SFT}(y|x)}\Big] + \gamma\,\mathbb E_{x\sim D_{\text{pretrain}}}\big[\log\pi^{RL}_\phi(x)\big] \]
- \(\gamma\) (coeficiente de pré-treino; **não** confundir com o desconto) = **27,8**; \(\gamma=0\) → modelos "PPO"; "InstructGPT" = PPO-ptx.
- Problema que resolve: o RLHF causava **regressões em datasets públicos** (SQuADv2, DROP, tradução) — o "imposto de alinhamento" (*alignment tax*).
- Implementação [P Ap. C.4]: **8× mais exemplos de pré-treino que episódios de RL**; em cada minibatch calculam-se os gradientes do PPO e do pré-treino **em passos consecutivos, acumulados no mesmo buffer**, com o de pré-treino multiplicado por \(\gamma\).
- Ablações [P Ap. E.6, E.11]: \(\gamma\ge20\) recupera as regressões no 1,3B; 27,8 funcionou de 1,3B a 175B; o escore Likert humano foi insensível ao valor exato. **Aumentar só o \(\beta\) da KL não resolve**: mesmo \(\beta = 2{,}0\) (100× o padrão) não consertou as regressões e derrubou a recompensa de validação — "a distribuição de pré-treino é crítica". Razão de dados 32 deu Likert melhor em testes preliminares, mas multiplicava o tempo; 8 **dobra** o tempo de treino em relação ao PPO sem mistura.
- "Secrets of RLHF" também adiciona a perda de LM de pré-treino ao PPO-max "para reduzir o alignment tax" [P §5.4].

### 4.10 Onde o PPO-RLHF está em 2026 (contexto para o capítulo)

- **Chat/alinhamento geral**: grande parte da comunidade aberta migrou para DPO e variantes (sem RM explícito, sem RL) e para REINFORCE/RLOO/GRPO (sem crítico) — capítulos seguintes do Módulo 2. Mas HelpSteer2-Preference usou o RM com **REINFORCE** para chegar a 85,0 no Arena Hard (1º em 01/10/2024) [P].
- **Raciocínio (RLVR)**: GRPO popularizado pelo DeepSeek-R1; PPO **com crítico** voltou competitivo quando o crítico é bem tratado (VC-PPO, VAPO — §3.7).
- **Infraestrutura**: o gargalo mudou do algoritmo para o **sistema** — geração eficiente, sincronização de pesos, assincronia e **consistência numérica** entre amostrador e treinador (§4.5–§4.6). É aqui que o conhecimento do Módulo 1 (vLLM, quantização, kernels, MoE, NCCL/RDMA) vira pré-requisito do Módulo 2.

---

## Apêndice A — Exemplos numéricos passo a passo (para figuras animadas)

> Todos os números abaixo foram calculados por um script Python (stdlib + NumPy) a partir das fórmulas desta nota; entradas marcadas **ilustrativas** são escolhas minhas, não dados de paper. Arredondamento a 4 casas.

### A.1 Bradley–Terry: probabilidade, perda e gradiente

| \(r_w\) | \(r_l\) | \(\Delta = r_w - r_l\) | \(P(y_w\succ y_l) = \sigma(\Delta)\) | perda \(-\log\sigma(\Delta)\) | \(\partial\mathcal L/\partial\Delta = -\sigma(-\Delta)\) |
|---|---|---|---|---|---|
| 1,2 | 0,4 | 0,80 | 0,6900 | 0,3711 | −0,3100 |
| 0,4 | 1,2 | −0,80 | 0,3100 | 1,1711 | −0,6900 |
| 2,0 | −1,0 | 3,00 | 0,9526 | 0,0486 | −0,0474 |
| 0,0 | 0,0 | 0,00 | 0,5000 | 0,6931 (= ln 2) | −0,5000 |

- **Animação sugerida**: duas barras (\(r_w\), \(r_l\)) e a sigmoide; arrastar \(\Delta\) mostra a perda caindo e o "peso do gradiente" \(\sigma(-\Delta)\) encolhendo. Mostrar que somar +5 às duas barras não muda nada (0,69 → 0,69: invariância a translação).

### A.2 Elo ↔ Bradley–Terry

\(\Delta r = \frac{\ln 10}{400}\Delta\text{Elo} = 0{,}005756\,\Delta\text{Elo}\).

| ΔElo | \(P\) (Elo) | \(\Delta r\) | \(\sigma(\Delta r)\) |
|---|---|---|---|
| 100 | 0,6401 | 0,5756 | 0,6401 |
| 200 | 0,7597 | 1,1513 | 0,7597 |
| 400 | 0,9091 | 2,3026 | 0,9091 |

### A.3 Ranking de K = 4 (InstructGPT) — pares vs Plackett–Luce

Escores **ilustrativos**: \(r_A = 2{,}0,\ r_B = 1{,}0,\ r_C = 0{,}5,\ r_D = -0{,}5\); ranking humano A ≻ B ≻ C ≻ D → \(\binom42 = 6\) pares.

| Par | Δ | σ(Δ) | −log σ(Δ) |
|---|---|---|---|
| A≻B | 1,0 | 0,7311 | 0,3133 |
| A≻C | 1,5 | 0,8176 | 0,2014 |
| A≻D | 2,5 | 0,9241 | 0,0789 |
| B≻C | 0,5 | 0,6225 | 0,4741 |
| B≻D | 1,5 | 0,8176 | 0,2014 |
| C≻D | 1,0 | 0,7311 | 0,3133 |
| **soma / média (÷6)** | | | **1,5823 / 0,2637** |

- Perda InstructGPT deste prompt = 0,2637 (a média sobre os \(\binom K2\) pares, todos no **mesmo elemento do batch**; 4 forwards do RM, não 12).
- **Plackett–Luce** do mesmo ranking: estágios 0,5977 (A entre {A,B,C,D}) × 0,5465 (B entre {B,C,D}) × 0,7311 (C entre {C,D}) × 1 → \(P = 0{,}2388\), NLL = 1,4321. Números diferentes porque são verossimilhanças diferentes (composta vs conjunta), mas ambas são minimizadas empurrando os escores na ordem certa.

### A.4 Plackett–Luce com K = 3 — as 6 permutações somam 1

Escores ilustrativos \(r_a=1,\ r_b=0,\ r_c=-1\):

| Ordem | P |
|---|---|
| a ≻ b ≻ c | 0,4863 |
| b ≻ a ≻ c | 0,2156 |
| a ≻ c ≻ b | 0,1789 |
| c ≻ a ≻ b | 0,0658 |
| b ≻ c ≻ a | 0,0292 |
| c ≻ b ≻ a | 0,0242 |
| **soma** | **1,0000** |

### A.5 Concordância entre anotadores

**κ de Cohen (binário, ilustrativo)** — 100 pares, dois anotadores, rótulo "A melhor"/"B melhor": (A,A) = 40, (A,B) = 10, (B,A) = 15, (B,B) = 35.
- \(p_o = (40+35)/100 = 0{,}75\); marginais 0,50 e 0,55 → \(p_e = 0{,}50\cdot0{,}55 + 0{,}50\cdot0{,}45 = 0{,}50\); **κ = (0,75 − 0,50)/(1 − 0,50) = 0,50**.
- Leitura: 75% de concordância bruta parece alto, mas metade disso é acaso.

**κ ponderado quadrático (ordinal 3 níveis, ilustrativo)** — matriz de contagem (linhas = anotador A, colunas = B):

| | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 20 | 5 | 0 |
| 1 | 4 | 30 | 6 |
| 2 | 1 | 4 | 30 |

- κ não ponderado = **0,6949**; κ ponderado quadrático = **0,8064** (as discordâncias são majoritariamente "vizinhas", que pesam pouco) — o mesmo efeito que levou HelpSteer2 a reportar κ quadrático.

### A.6 Ruído de rotulador (Christiano 2017): \(\hat P = 0{,}9\,\sigma(\Delta) + 0{,}05\)

| Δ | σ(Δ) | \(\hat P\) com 10% de resposta aleatória |
|---|---|---|
| 0 | 0,5000 | 0,5000 |
| 1 | 0,7311 | 0,7080 |
| 3 | 0,9526 | 0,9073 |
| 10 | 0,99995 | 0,9500 (teto) |

- A perda nunca empurra Δ ao infinito para "explicar" um erro humano; o teto de 95% limita o gradiente em pares ruidosos.

### A.7 Margem (Llama 2) e BT escalado (HelpSteer2-Preference), com Δ = 1,0

| Variante | parâmetro | perda | \(\partial\mathcal L/\partial\Delta\) |
|---|---|---|---|
| BT | — | 0,3133 | −0,2689 |
| Margem | m = 1/3 | 0,4144 | −0,3392 |
| Margem | m = 2/3 | 0,5403 | −0,4174 |
| Margem | m = 1 | 0,6931 | −0,5000 |
| Margem | m = 3 | 2,1269 | −0,8808 |
| BT escalado | m = 1 / 2 / 3 | 0,3133 / 0,6265 / 0,9398 | −0,2689·m |

- Margem: o par "significantly better" só fica "satisfeito" quando Δ ≫ m. BT escalado: mesmo formato da curva, só multiplica o peso.

### A.8 Calibração e recalibração por temperatura (ilustrativo)

Três bins de pares por |Δ| médio:

| Δ médio (em módulo) | n | acurácia observada | σ(Δ) prevista | erro |
|---|---|---|---|---|
| 0,25 | 400 | 0,55 | 0,5622 | −0,0122 |
| 1,0 | 350 | 0,70 | 0,7311 | −0,0311 |
| 2,5 | 250 | 0,86 | 0,9241 | −0,0641 |

- **ECE = 0,0318**; o RM é **sobreconfiante** (prevê mais do que acerta), pior nos pares "fáceis".
- Recalibração (estilo Gao 2022): achar \(T\) que minimiza a NLL de \(\sigma(\Delta/T)\) → \(T = 1{,}31\); previsões viram 0,5476 / 0,6821 / 0,8708 — próximas das acurácias observadas. A **ordem** dos escores não muda (BoN idêntico), só a escala.

### A.9 Superotimização (formas de Gao 2022) e KL do best-of-n

**KL do BoN** \(= \log n - (n-1)/n\):

| n | 1 | 2 | 4 | 16 | 64 | 1.000 | 60.000 |
|---|---|---|---|---|---|---|---|
| KL (nats) | 0 | 0,1931 | 0,6363 | 1,8351 | 3,1745 | 5,9088 | 10,0021 |

(n = 1.000 ≈ 6 nats e n = 60.000 ≈ 10 nats, como no paper; a Tab. 9 de Stiennon dá 1,2 / 3,2 / 3,9 para N = 8 / 64 / 128.)

**Curvas gold com coeficientes ilustrativos** (\(\alpha_{\text{bon}}=1,\ \beta_{\text{bon}}=0{,}25\); \(\alpha_{\text{RL}}=1,\ \beta_{\text{RL}}=0{,}5\)):

| \(d=\sqrt{KL}\) | 0,5 | 1 | 2 | 2,718 | 4 | 6 |
|---|---|---|---|---|---|---|
| \(R_{\text{bon}}(d)\) | 0,4375 | 0,75 | **1,00** (pico) | 0,8711 | 0,00 | −3,00 |
| \(R_{\text{RL}}(d)\) | 0,6733 | 1,00 | 1,3069 | **1,3591** (pico) | 1,2274 | 0,6247 |

- Picos: BoN em \(d^*=\alpha/2\beta = 2\) (KL* = 4 nats, R* = 1,0); RL em \(d^* = e^{\alpha/\beta-1} = e\) (KL* = 7,389 nats, R* = \(\beta d^*\) = 1,3591).
- **Animação**: eixo x em \(\sqrt{KL}\); curva proxy (subindo ~linear) vs gold (sobe e cai); um cursor "β da KL" que congela o ponto de parada.

### A.10 Ensemble conservador (Coste et al.), ilustrativo

4 RMs dão \(R_i = (1{,}8;\ 2{,}0;\ 2{,}2;\ 0{,}6)\) para a mesma resposta (um membro discorda muito):
- Média = **1,65**; variância intra-ensemble = 0,3875.
- WCO = min = **0,60**.
- UWO (λ = 0,5) = 1,65 − 0,5·0,3875 = **1,4562**; UWO (λ = 1) = **1,2625**.
- Respostas onde o ensemble discorda (provável exploração de um RM) perdem recompensa.

### A.11–A.12 Recompensa por token com KL, e GAE para λ ∈ {0; 0,95; 1}

Resposta de 5 tokens, \(\beta = 0{,}05\), RM = **2,0** no EOS (ilustrativo):

| t | 1 | 2 | 3 | 4 | 5 (EOS) |
|---|---|---|---|---|---|
| \(\log\pi_{\text{old}}\) | −1,20 | −0,50 | −2,10 | −0,30 | −0,05 |
| \(\log\pi_{\text{ref}}\) | −1,40 | −0,55 | −1,60 | −0,45 | −0,10 |
| KL por token (\(k_1\)) | 0,20 | 0,05 | −0,50 | 0,15 | 0,05 |
| \(r_t = -\beta\,\text{KL}_t\ (+\text{RM})\) | −0,0100 | −0,0025 | +0,0250 | −0,0075 | **1,9975** |
| \(V_{\text{old}}(s_t)\) (crítico, ilustrativo) | 0,90 | 1,10 | 1,00 | 1,50 | 1,80 |
| \(\delta_t = r_t + V(s_{t+1}) - V(s_t)\) | 0,1900 | −0,1025 | 0,5250 | 0,2925 | 0,1975 |
| \(\hat A_t\), λ = 0 | 0,1900 | −0,1025 | 0,5250 | 0,2925 | 0,1975 |
| \(\hat A_t\), λ = 0,95 | 0,9781 | 0,8296 | 0,9811 | 0,4801 | 0,1975 |
| \(\hat A_t\), λ = 1 | 1,1025 | 0,9125 | 1,0150 | 0,4900 | 0,1975 |
| \(\hat R_t = \hat A_t + V\), λ = 0,95 | 1,8781 | 1,9296 | 1,9811 | 1,9801 | 1,9975 |
| \(\hat R_t\), λ = 1 (= retorno MC) | 2,0025 | 2,0125 | 2,0150 | 1,9900 | 1,9975 |

- Soma das recompensas = 2,0025 = RM − β·ΣKL (ΣKL = −0,05 neste exemplo: o estimador \(k_1\) por amostra pode dar negativo).
- Com λ = 0 cada token só "vê" o próximo valor; com λ = 1 todos veem o retorno final (alvo MC, como o \(\lambda_{\text{crítico}}=1\) do VC-PPO).
- **Animação**: varrer λ de 0 a 1 e ver o "crédito" da recompensa final escorrer para trás pelos tokens.

### A.13 PPO clipado, ε = 0,2

| ρ | A = +2: \(\rho A\) | clip·A | \(L=\min\) | ∂L/∂ρ | A = −1: \(\rho A\) | clip·A | \(L=\min\) | ∂L/∂ρ |
|---|---|---|---|---|---|---|---|---|
| 0,7 | 1,40 | 1,60 | 1,40 | 2 | −0,70 | −0,80 | −0,80 | **0** |
| 0,8 | 1,60 | 1,60 | 1,60 | 2 | −0,80 | −0,80 | −0,80 | −1 |
| 0,9 | 1,80 | 1,80 | 1,80 | 2 | −0,90 | −0,90 | −0,90 | −1 |
| 1,0 | 2,00 | 2,00 | 2,00 | 2 | −1,00 | −1,00 | −1,00 | −1 |
| 1,1 | 2,20 | 2,20 | 2,20 | 2 | −1,10 | −1,10 | −1,10 | −1 |
| 1,2 | 2,40 | 2,40 | 2,40 | 2 | −1,20 | −1,20 | −1,20 | −1 |
| 1,3 | 2,60 | 2,40 | 2,40 | **0** | −1,30 | −1,20 | −1,30 | −1 |

- Ação boa (A>0): o incentivo **some** acima de 1,2. Ação ruim (A<0): o incentivo some abaixo de 0,8, mas **continua** acima de 1,2 (corrige quando a ação ruim ficou mais provável). Reproduz a Fig. 1 do paper do PPO.

### A.14 Controlador adaptativo de KL (Ziegler), \(K_\beta = 0{,}1\), alvo 6 nats, \(\beta_0 = 0{,}1\)

| KL medido | \(e_t\) (clip ±0,2) | \(\beta_{t+1}\) |
|---|---|---|
| 12,0 | +0,2000 | 0,102000 |
| 10,0 | +0,2000 | 0,104040 |
| 6,0 | 0,0000 | 0,104040 |
| 3,0 | −0,2000 | 0,101959 |
| 5,5 | −0,0833 | 0,101110 |

- O controlador é **lento** de propósito (no máx. ±2% por passo com \(K_\beta = 0{,}1\)); o do paper do PPO (dobra/divide) é agressivo.

### A.15 Estimadores de KL por token (\(r = \pi_{\text{ref}}/\pi_\theta\))

| \(\log\pi_\theta\) | \(\log\pi_{\text{ref}}\) | \(\log r\) | \(r\) | \(k_1=-\log r\) | \(k_2=\frac12(\log r)^2\) | \(k_3=(r-1)-\log r\) |
|---|---|---|---|---|---|---|
| −1,0 | −1,2 | −0,200 | 0,8187 | 0,2000 | 0,0200 | 0,0187 |
| −2,0 | −1,0 | +1,000 | 2,7183 | **−1,0000** | 0,5000 | 0,7183 |
| −0,1 | −0,1 | 0,000 | 1,0000 | 0,0000 | 0,0000 | 0,0000 |
| −0,5 | −3,0 | −2,500 | 0,0821 | 2,5000 | 3,1250 | 1,5821 |

- \(k_1\) fica negativo quando a política dá menos probabilidade que a referência ao token amostrado; \(k_3\) nunca é negativo.

### A.16 REINFORCE com baseline e RLOO (k = 4 amostras do mesmo prompt)

Recompensas ilustrativas \(R = (1{,}0;\ 0{,}0;\ 0{,}5;\ 0{,}9)\), média 0,6:

| i | \(R_i\) | baseline leave-one-out | \(\hat A_i\) (RLOO) | \(R_i - \bar R\) |
|---|---|---|---|---|
| 1 | 1,0 | 0,4667 | 0,5333 | 0,4 |
| 2 | 0,0 | 0,8000 | −0,8000 | −0,6 |
| 3 | 0,5 | 0,6333 | −0,1333 | −0,1 |
| 4 | 0,9 | 0,5000 | 0,4000 | 0,3 |

- \(\hat A^{\text{RLOO}}_i = \frac{k}{k-1}(R_i - \bar R) = 1{,}333\times(R_i-\bar R)\): mesma direção da média simples, sem o viés de incluir a própria amostra.

### A.17 TRPO em 2D: gradiente natural, gradiente conjugado e passo máximo

\(F = \begin{pmatrix}2 & 0{,}5\\ 0{,}5 & 1\end{pmatrix}\) (Fisher), \(g = (1, 1)\), \(\delta = 0{,}01\):
- CG, iteração 1: \(x = (0{,}5;\ 0{,}5)\); iteração 2: \(x = (0{,}28571;\ 0{,}85714)\) = solução exata \(s = F^{-1}g\) (CG converge em ≤ dim passos).
- \(s^\top F s = 1{,}1429\) → \(\beta = \sqrt{2\delta/s^\top Fs} = 0{,}1323\) → passo \(\beta s = (0{,}0378;\ 0{,}1134)\); KL quadrático = 0,0100 = δ ✓.
- Gradiente "comum" com o mesmo KL: \((0{,}0707;\ 0{,}0707)\) — o natural anda **mais** na direção de baixa curvatura (2ª coordenada) e menos na de alta.
- **Animação**: elipse de KL constante; seta do gradiente vs seta do gradiente natural tocando a elipse; depois a busca em linha encolhendo o passo.

### A.18 Importance sampling e TIS (C = 2) — descompasso vLLM × FSDP

| token | \(\pi_{\text{fsdp}}\) | \(\pi_{\text{vllm}}\) | ρ | min(ρ, 2) |
|---|---|---|---|---|
| 1 | 0,30 | 0,25 | 1,2000 | 1,2000 |
| 2 | 0,50 | 0,52 | 0,9615 | 0,9615 |
| 3 | 0,020 | 0,004 | **5,0000** | **2,0000** |
| 4 | 0,90 | 0,91 | 0,9890 | 0,9890 |

- Produto por sequência = 5,7058 — um token raro com descompasso domina tudo; truncar por token mantém o peso limitado. (Valores ilustrativos; o fenômeno de tokens de baixa probabilidade com ratios extremos é o que motiva TIS e R3.)

### A.19 Memória dos quatro modelos (ver tabela em §4.3)

\(36N\) bytes (2 treináveis × 16 B/param + 2 congelados × 2 B/param): 1,5B → 54 GB; 3B → 108 GB; 7B → 252 GB; 8B → 288 GB — sem ativações, KV do motor de inferência e buffers.

### A.20 FLOPs de uma iteração (ver tabela em §4.7)

7B, 512 × (256 + 256) tokens: geração 3,67 PFLOP; 4 forwards de pontuação 14,68; treino (2 modelos, 1 época) 22,02; total 40,37 PFLOP. Decode no GB10: ≥ 51 ms/passo (14 GB a 273 GB/s) → ≥ 13 s para 256 passos.

### A.21 Decaimento do sinal terminal no GAE: \(\lambda^{T-t}\)

| λ \ distância ao fim | 50 tokens | 500 tokens | 8.000 tokens |
|---|---|---|---|
| 0,95 | 0,0769 | 7,3×10⁻¹² | 6,1×10⁻¹⁷⁹ |
| 0,99 | 0,6050 | 0,0066 | 1,2×10⁻³⁵ |
| 0,999 | 0,9512 | 0,6064 | 3,3×10⁻⁴ |
| 1,0 | 1 | 1 | 1 |

- Por isso VC-PPO usa λ = 1 no crítico e VAPO adapta λ ao comprimento.

### A.22 Gradiente de política numa softmax de 3 tokens

Logits \(z = (2{,}0;\ 1{,}0;\ 0{,}1)\) → \(\pi = (0{,}6590;\ 0{,}2424;\ 0{,}0986)\). Token amostrado: \(a = 2\) (o do meio).
- \(\nabla_z\log\pi(a) = \mathbf e_a - \pi = (-0{,}6590;\ +0{,}7576;\ -0{,}0986)\).
- Passo com \(\eta = 1\), **A = +1,5**: \(z \to (1{,}0115;\ 2{,}1364;\ -0{,}0478)\), \(\pi \to (0{,}2259;\ \mathbf{0{,}6958};\ 0{,}0783)\).
- Passo com **A = −1,5**: \(z \to (2{,}9885;\ -0{,}1364;\ 0{,}2478)\), \(\pi \to (0{,}9021;\ \mathbf{0{,}0396};\ 0{,}0582)\).
- (Passo exagerado de propósito para a animação; é exatamente o que o clipping do PPO impede de acontecer num só lote.)

### A.23 Clipping do valor (\(\hat\varepsilon = 0{,}2\))

| \(V_{\text{old}}\) | \(V_{\text{novo}}\) | alvo \(\hat R\) | \((V_{\text{novo}}-\hat R)^2\) | \(V\) clipado | \((V_{\text{clip}}-\hat R)^2\) | perda = max | gradiente |
|---|---|---|---|---|---|---|---|
| 1,0 | 1,6 | 1,1 | 0,25 | 1,2 | 0,01 | 0,25 | via termo não-clipado (puxa V de volta) |
| 1,0 | 0,7 | 1,1 | 0,16 | 0,8 | 0,09 | 0,16 | via termo não-clipado |
| 1,0 | 1,3 | 2,0 | 0,49 | 1,2 | **0,64** | 0,64 | **zero** (já andou +0,3 > 0,2 na direção certa: para) |

### A.24 Agregação de um PRM

Probabilidades de passo correto (ilustrativas) \((0{,}98;\ 0{,}95;\ 0{,}60;\ 0{,}97)\): **produto = 0,5418** (Lightman), **mínimo = 0,60**. Um único passo duvidoso derruba a solução inteira nos dois; o produto também penaliza soluções longas com muitos passos "quase certos".

### A.25 Rótulos automáticos do Math-Shepherd

A partir do passo \(s_i\), o completer termina N = 4 soluções com respostas (42, 17, 42, 42); gabarito 42 → **HE = 1** (alguma acertou), **SE = 0,75** (3 de 4 — estimativa MC de \(V(s_i)\)).

---

## Apêndice B — Fontes

### B.1 Papers primários (lidos no texto integral)

| Referência | URL |
|---|---|
| Christiano, Leike, Brown, Martic, Legg, Amodei 2017 — *Deep RL from Human Preferences* | https://arxiv.org/abs/1706.03741 |
| Ziegler et al. 2019 — *Fine-Tuning Language Models from Human Preferences* | https://arxiv.org/abs/1909.08593 |
| Stiennon et al. 2020 — *Learning to summarize from human feedback* | https://arxiv.org/abs/2009.01325 |
| Ouyang et al. 2022 — *Training language models to follow instructions with human feedback* (InstructGPT) | https://arxiv.org/abs/2203.02155 |
| Bai et al. 2022a — *Training a Helpful and Harmless Assistant with RLHF* | https://arxiv.org/abs/2204.05862 |
| Bai et al. 2022b — *Constitutional AI* | https://arxiv.org/abs/2212.08073 |
| Askell et al. 2021 — *A General Language Assistant as a Laboratory for Alignment* (PMP) | https://arxiv.org/abs/2112.00861 |
| Gao, Schulman, Hilton 2022 — *Scaling Laws for Reward Model Overoptimization* | https://arxiv.org/abs/2210.10760 |
| Schulman et al. 2015a — *Trust Region Policy Optimization* | https://arxiv.org/abs/1502.05477 |
| Schulman et al. 2015b — *High-Dimensional Continuous Control Using GAE* | https://arxiv.org/abs/1506.02438 |
| Schulman et al. 2017 — *Proximal Policy Optimization Algorithms* | https://arxiv.org/abs/1707.06347 |
| Mnih et al. 2016 — *Asynchronous Methods for Deep RL* (A3C) | https://arxiv.org/abs/1602.01783 |
| Touvron et al. 2023 — *Llama 2: Open Foundation and Fine-Tuned Chat Models* | https://arxiv.org/abs/2307.09288 |
| Zheng R. et al. 2023 — *Secrets of RLHF in LLMs Part I: PPO* | https://arxiv.org/abs/2307.04964 |
| Huang et al. 2024 — *The N+ Implementation Details of RLHF with PPO* | https://arxiv.org/abs/2403.17031 |
| Uesato et al. 2022 — *Solving math word problems with process- and outcome-based feedback* | https://arxiv.org/abs/2211.14275 |
| Lightman et al. 2023 — *Let's Verify Step by Step* | https://arxiv.org/abs/2305.20050 |
| Wang P. et al. 2023 — *Math-Shepherd* | https://arxiv.org/abs/2312.08935 |
| Zheng L. et al. 2023 — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | https://arxiv.org/abs/2306.05685 |
| Lee et al. 2023 — *RLAIF vs. RLHF* | https://arxiv.org/abs/2309.00267 |
| Cui et al. 2023 — *UltraFeedback* | https://arxiv.org/abs/2310.01377 |
| Wang Z. et al. 2023 — *HelpSteer* | https://arxiv.org/abs/2311.09528 |
| Wang Z. et al. 2024 — *HelpSteer2* | https://arxiv.org/abs/2406.08673 |
| Wang Z. et al. 2024 — *HelpSteer2-Preference* | https://arxiv.org/abs/2410.01257 |
| Singhal et al. 2023 — *A Long Way to Go: Investigating Length Correlations in RLHF* | https://arxiv.org/abs/2310.03716 |
| Coste et al. 2023 — *Reward Model Ensembles Help Mitigate Overoptimization* | https://arxiv.org/abs/2310.02743 |
| Eisenstein et al. 2023 — *Helping or Herding?* (abstract) | https://arxiv.org/abs/2312.09244 |
| Ramé et al. 2024 — *WARM* | https://arxiv.org/abs/2401.12187 |
| Lambert et al. 2024 — *RewardBench* | https://arxiv.org/abs/2403.13787 |
| Zhang et al. 2024 — *Generative Verifiers (GenRM)* | https://arxiv.org/abs/2408.15240 |
| Liu et al. 2025 — *Inference-Time Scaling for Generalist Reward Modeling* (DeepSeek-GRM) | https://arxiv.org/abs/2504.02495 |
| Ahmadian et al. 2024 — *Back to Basics: REINFORCE-style optimization for RLHF* (RLOO) | https://arxiv.org/abs/2402.14740 |
| Shao et al. 2024 — *DeepSeekMath* (GRPO) | https://arxiv.org/abs/2402.03300 |
| Hu et al. 2024 — *OpenRLHF* | https://arxiv.org/abs/2405.11143 |
| Sheng et al. 2024 — *HybridFlow* (verl) | https://arxiv.org/abs/2409.19256 |
| Noukhovitch et al. 2024 — *Asynchronous RLHF* | https://arxiv.org/abs/2410.18252 |
| Yao, Liu et al. 2025 — *On the Rollout-Training Mismatch in Modern RL Systems* (OPT 2025) | https://opt-ml.org/papers/2025/paper116.pdf |

### B.2 Papers consultados só pelo abstract / página (verificação parcial)

| Referência | URL |
|---|---|
| VC-PPO — *What's Behind PPO's Collapse in Long-CoT?* | https://arxiv.org/abs/2503.01491 |
| VAPO | https://arxiv.org/abs/2504.05118 |
| RewardBench 2 | https://arxiv.org/abs/2506.01937 |
| Skywork-Reward-V2 | https://arxiv.org/abs/2507.01352 |
| R3 — *Stabilizing MoE RL by Aligning Training and Inference Routers* | https://arxiv.org/abs/2510.11370 |
| MiniMax-M1 | https://arxiv.org/abs/2506.13585 |
| GLM-4.5 | https://arxiv.org/abs/2508.06471 |
| Zhu, Jordan, Jiao 2023 — *Principled RLHF from Pairwise or K-wise Comparisons* | https://arxiv.org/abs/2301.11270 |
| Rafailov et al. 2023 — DPO | https://arxiv.org/abs/2305.18290 |
| ZeRO (contabilidade 16 B/param) | https://arxiv.org/abs/1910.02054 |
| Kaplan et al. 2020 (FLOPs ≈ 6N/token) | https://arxiv.org/abs/2001.08361 |
| Engstrom et al. 2020 — *Implementation Matters in Deep Policy Gradients* | https://arxiv.org/abs/2005.12729 |
| Lambert — *Reinforcement Learning from Human Feedback* (livro) | https://arxiv.org/abs/2504.12501 |
| Clássicos sem URL arXiv: Bradley & Terry 1952 (*Biometrika* 39(3/4):324–345); Luce 1959 (*Individual Choice Behavior*); Plackett 1975 (*Applied Statistics* 24(2)); Thurstone 1927; Elo 1978; Williams 1992 (*Machine Learning* 8:229–256); Sutton et al. 2000 (NeurIPS 1999); Kakade 2001 (NeurIPS); Kakade & Langford 2002 (ICML); Greensmith, Bartlett, Baxter 2004 (JMLR); Ionides 2008 (*JCGS* 17(2)); Kool, van Hoof, Welling 2019 (ICLR workshop, "Buy 4 REINFORCE samples…") | — |

### B.3 Documentação e fontes web

| Fonte | URL |
|---|---|
| vLLM — RLHF | https://docs.vllm.ai/en/latest/training/rlhf.html |
| vLLM — Weight transfer | https://docs.vllm.ai/en/latest/training/weight_transfer/ |
| vLLM — Sleep mode | https://docs.vllm.ai/en/latest/features/sleep_mode/ |
| SGLang — SGLang for RL Systems | https://docs.sglang.io/docs/advanced_features/sglang_for_rl |
| slime (README) | https://github.com/THUDM/slime |
| vLLM blog — *Announcing vime* (2026-06-09) | https://vllm.ai/blog/2026-06-09-announcing-vime |
| Thinking Machines — *Defeating Nondeterminism in LLM Inference* | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ |
| Schulman — *Approximating KL Divergence* (2020) | http://joschu.net/blog/kl-approx.html |
| HF — Anthropic/hh-rlhf | https://huggingface.co/datasets/Anthropic/hh-rlhf |
| HF — HuggingFaceH4/ultrafeedback_binarized | https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized |
| HF — nvidia/HelpSteer3 | https://huggingface.co/datasets/nvidia/HelpSteer3 |
| HF — stanfordnlp/SHP | https://huggingface.co/datasets/stanfordnlp/SHP |
| Huang et al. 2022 — *The 37 Implementation Details of PPO* (ICLR Blog Track) | https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ |

---

## Apêndice C — Incertezas, divergências e pontos a confirmar

1. **Contagens divergentes do mesmo dataset.** OpenAI Summarize: Stiennon fala em 64.832 comparações TL;DR; UltraFeedback e N+ usam 92.858 pares; Llama 2 lista 176.625 (provavelmente inclui validação/CNN-DM ou outras divisões). WebGPT: 19.578 (UltraFeedback) vs 13.333 (Llama 2). Anthropic Helpful: 118.263 (UltraFeedback) vs 122.387 (Llama 2). Diferenças de filtragem/split — no capítulo, citar sempre a fonte junto com o número.
2. **HybridFlow "58,9%"**: a extração do PDF intercala a figura no meio da frase; a leitura natural é "a geração do ator representa a maior parte do tempo", mas convém conferir no PDF renderizado antes de citar como "58,9% é geração".
3. **VC-PPO, VAPO, R3, MiniMax-M1, RewardBench 2, Skywork-V2**: verificados por abstract/página, não no texto integral; números citados são os do abstract (VAPO 60,4 AIME24/5.000 passos; R3 ~94% dos tokens e KL >2×; MiniMax 0,9 → 0,99).
4. **TIS (Yao et al.)**: a versão lida é o paper do workshop OPT 2025 (5 páginas); o valor de \(C\) usado nos experimentos não aparece no texto extraído — não citar um \(C\) específico.
5. **vime**: informação de blog do vLLM (2026-06-09); números de "diferença de logprob" (0,011 vs 0,77; 0,019 → 0,013) dependem de definição de métrica não detalhada no resumo — marcar como [W].
6. **APIs do vLLM**: a página de *weight transfer* (`latest`) descreve backends e endpoints que podem mudar entre versões; as notas do Módulo 1 usam vLLM 0.30/0.31 com overlays — conferir a versão da imagem antes de prometer um endpoint num laboratório.
7. **Sleep mode no GB10 (interpretação minha)**: "descarregar para a RAM da CPU" (nível 1) não libera memória física numa máquina de memória unificada; o nível 2 (descartar) libera. Não encontrei documentação do vLLM específica para GB10 sobre isso — testar antes de afirmar no capítulo.
8. **LoRA com referência = adaptador desligado**: prática comum (TRL/PEFT), mas não verifiquei a API atual nesta data; citar como padrão de engenharia, sem nome de função.
9. **Decodificação especulativa em rollouts**: a preservação exata da distribuição é teorema em aritmética exata; o efeito numérico em RL não foi medido em nenhuma fonte lida.
10. **"SFT ≡ PG com Â = 1"**: é uma intuição pedagógica (gradientes idênticos em forma quando as amostras são demonstrações), não um teorema dos papers citados.
11. **Seções/tabelas citadas**: conferidas no texto extraído; em PDFs de duas colunas a numeração de equações pode ter vindo deslocada (ex.: GAE) — onde não tive certeza, citei só a seção.
12. **GLM-5.3 e slime**: a afirmação "slime é o framework de RL por trás do GLM-5.3" é do README do slime [D]; não há confirmação disso nas notas do Módulo 1 (que tratam só de inferência).

---

## Apêndice D — Conceitos em ordem de pré-requisito

**Nível 0 — Probabilidade e estatística**
1. Variáveis aleatórias, esperança, variância; estimador não-enviesado vs enviesado; variância de estimadores Monte Carlo.
2. Máxima verossimilhança; entropia cruzada (binária e categórica); rótulos suaves.
3. Sigmoide/logística, softmax, log-odds; distribuição de Gumbel e "Gumbel-max".
4. KL divergence (direções \(\text{KL}(p\|q)\) vs \(\text{KL}(q\|p)\)), entropia, f-divergências; estimadores Monte Carlo de KL.
5. Importance sampling; truncamento; variância de produtos de razões.
6. Medidas de concordância: % bruta, κ de Cohen, κ ponderado, Fleiss, Krippendorff.
7. Calibração: diagramas de confiabilidade, ECE, temperature scaling.

**Nível 1 — Modelos de preferência**
8. Bradley–Terry; Thurstone (probit); Elo.
9. Plackett–Luce; Luce choice axiom; verossimilhança composta (pares de um ranking).
10. Empates (Rao–Kupper/Davidson); margens; intensidade de preferência.
11. Regressão em Likert multi-atributo (SteerLM).

**Nível 2 — Modelos de recompensa como redes**
12. LM + cabeça escalar; leitura no EOS; normalização por bias.
13. Treino de 1 época, sobreajuste, agrupamento de pares por prompt.
14. Avaliação: acurácia de pares, tetos por discordância humana, RewardBench.
15. Viés de comprimento; reward hacking; Goodhart (regressional/extremal/causal/adversarial).
16. Leis de escala da superotimização (Gao); KL do best-of-n.
17. Ensembles (média/WCO/UWO), WARM.
18. ORM, PRM, rótulos MC (Math-Shepherd), verificadores; RMs generativos; LLM-as-judge e seus vieses; RLAIF.

**Nível 3 — RL tabular/conceitual**
19. MDP, política, retorno, desconto, episódio; o MDP de tokens; visão bandit.
20. \(V, Q, A\); equações de Bellman.
21. Gradiente de política: truque da log-derivada; teorema de Sutton; causalidade.
22. REINFORCE; baselines; baseline ótimo; RLOO.
23. TD(0), n-passos, λ-retorno; GAE e seus casos-limite.
24. On-policy vs off-policy; importance sampling por token vs por sequência.
25. Entropia como regularizador; exploração.

**Nível 4 — Algoritmos de deep RL**
26. Ator-crítico; parâmetros compartilhados vs separados.
27. A3C (assíncrono, Hogwild!) e A2C (síncrono).
28. Lema de diferença de desempenho; limite de melhoria monotônica; TRPO; Fisher; gradiente natural; gradiente conjugado; busca em linha.
29. PPO: razão, clipping, penalidade KL adaptativa, perda de valor (com clipping), entropia, épocas/minibatches.
30. Detalhes de implementação (normalizações, inicializações, clipping de gradiente).

**Nível 5 — RLHF com PPO**
31. Pipeline SFT → RM → PPO (InstructGPT) e suas variantes (Stiennon, Bai, Llama 2).
32. Recompensa moldada por KL (por sequência e por token); política ótima \(\pi^*\propto\pi_{\text{ref}}e^{r/\beta}\); controlador adaptativo.
33. Os quatro modelos e sua memória; precisão mista e estados do Adam.
34. Rejection sampling fine-tuning vs PPO; PPO-ptx e alignment tax.
35. Truques de estabilidade e métricas de monitoramento.

**Nível 6 — Sistemas de RL para LLMs (ponte com o Módulo 1)**
36. Motor de inferência no laço (vLLM/SGLang); por que a geração domina o tempo.
37. Colocado (hybrid engine, sleep/wake, resharding) vs desagregado (broadcast NCCL, endpoints de weight update).
38. RL assíncrono; staleness; off-policyness tolerável.
39. Descompasso treino–inferência: kernels, precisão (LM head FP32), quantização de rollouts, batch invariance, roteamento MoE (R3), TIS.
40. Custo: FLOPs por fase vs tempo limitado por banda; orçamento de memória num DGX Spark.

---

## Apêndice E — Glossário PT ⇄ EN (termos que aparecem nos capítulos)

| PT (usar no texto) | EN (manter entre parênteses na 1ª ocorrência) |
|---|---|
| modelo de recompensa | reward model (RM) |
| modelo de preferência | preference model (PM) |
| resposta preferida / rejeitada | chosen / rejected (\(y_w\) / \(y_l\)) |
| superotimização | overoptimization |
| hackeamento da recompensa | reward hacking |
| política / política de referência | policy / reference policy |
| crítico, função valor | critic, value function |
| vantagem | advantage |
| resíduo de diferença temporal | TD residual / TD error |
| estimador generalizado de vantagem | generalized advantage estimation (GAE) |
| região de confiança | trust region |
| gradiente natural | natural gradient |
| gradiente conjugado | conjugate gradient (CG) |
| razão de probabilidades | probability ratio |
| amostragem por importância (truncada) | (truncated) importance sampling (TIS) |
| rollout / geração de trajetórias | rollout |
| motor de inferência | inference engine |
| sincronização de pesos | weight sync / weight update |
| colocado / desagregado | colocated / disaggregated |
| RL assíncrono, defasagem | async RL, staleness |
| descompasso treino–inferência | training–inference (rollout–training) mismatch |
| imposto de alinhamento | alignment tax |
| recompensa de resultado / de processo | outcome / process reward model (ORM / PRM) |
| juiz LLM | LLM-as-a-judge |
| feedback de IA | AI feedback (RLAIF) |
| concordância entre anotadores | inter-annotator agreement (IAA) |

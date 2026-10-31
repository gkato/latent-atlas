# Pós-treino (Módulo 2) — A família DPO e as alternativas por regressão e por contraste

Nota de pesquisa para os capítulos de **otimização direta de preferências** (DPO, IPO, KTO, SimPO e variantes) e de **alternativas por regressão/contraste** (AWR e parentes, InfoNCE, SLiC, CPL, unlikelihood).

- **Data da pesquisa:** 2026-10-01.
- **Método:** todos os PDFs citados foram baixados do arXiv e lidos (texto extraído). Cada fórmula abaixo foi conferida contra a equação original, e o número da equação aparece entre parênteses (ex.: "DPO, Eq. 7"). As implementações foram conferidas no código-fonte do TRL (branch `main`, versão `1.15.0.dev0`, baixada nesta data) e no repositório oficial do SimPO.
- **Versões lidas:** DPO arXiv v3 (29/07/2024); IPO/ΨPO v2 (22/11/2023); KTO **v5 (08/09/2026, revisão recente, com a Tabela 1 nova de hiperparâmetros)**; SimPO v3 (01/11/2024, NeurIPS 2024); ORPO v2; CPO v4; β-DPO v2; OAIF v2; AWR v3; AWAC v6; SLiC-HF v1; CPL v3 (ICLR 2024); Likelihood Displacement v4 (ICLR 2025); Overoptimization em DAAs v2 (NeurIPS 2024).
- **Convenções:** identificadores, flags, nomes de parâmetros de biblioteca e fórmulas ficam como no original. Onde eu derivo algo que não está escrito no paper, marco **[derivação minha]**. Onde interpreto, marco **[interpretação]**. Os exemplos numéricos de brinquedo (§12) são **ilustrativos** e foram calculados por script (valores conferidos). Eles não vêm de nenhum paper.
- **Contexto do Atlas:** o Módulo 1 ensinou a *servir* modelos (vLLM, 2× DGX Spark). Este módulo ensina como os pesos servidos foram *moldados* depois do pré-treino. A ponte com o Módulo 1 está em §13.

---

## 0. Notação única (usar a mesma em todos os capítulos)

| Símbolo | Significado |
|---|---|
| $x$ | prompt (contexto) |
| $y$ | resposta completa (sequência de tokens $y_1,\dots,y_{\lvert y\rvert}$) |
| $y_w, y_l$ | resposta preferida (*winner*, "chosen") e preterida (*loser*, "rejected"). Alguns papers escrevem $y^+, y^-$ ou $y_c, y_r$ |
| $\pi_\theta$ | política sendo treinada (o LLM) |
| $\pi_{\text{ref}}$ | política de referência congelada (em geral o modelo SFT) |
| $\log\pi(y\mid x)=\sum_t \log\pi(y_t\mid x,y_{<t})$ | log-prob da sequência (soma sobre tokens) |
| $h_\theta(x,y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ | log-razão política/referência (em "nats") |
| $\hat r_\theta(x,y)=\beta\,h_\theta(x,y)$ | **recompensa implícita** do DPO |
| $\Delta h = h_\theta(x,y_w)-h_\theta(x,y_l)$ | margem de log-razões |
| $\sigma(z)=1/(1+e^{-z})$ | sigmoide (logística) |
| $\beta$ | força da regularização KL (no DPO), "temperatura" da recompensa implícita |
| $\tau$ | o mesmo papel de $\beta$ na notação do paper do IPO. No TRL, `beta` = $\tau$ quando `loss_type="ipo"` |
| $\lvert y\rvert$ | número de tokens da resposta |

Identidades úteis que aparecem o tempo todo:
- $-\log\sigma(z)=\log(1+e^{-z})$ ("softplus" de $-z$). Vale $\log 2\approx 0{,}6931$ em $z=0$.
- $\frac{d}{dz}[-\log\sigma(z)]=-\sigma(-z)$. O gradiente da perda logística é "o quanto o classificador ainda erra".
- $\sigma(-z)=1-\sigma(z)$ e $\sigma'(z)=\sigma(z)(1-\sigma(z))$.

---

## 1. O ponto de partida: Bradley–Terry + RL com penalidade KL

Os métodos da família DPO "dobram" o pipeline clássico de RLHF (SFT → modelo de recompensa → RL) em uma única perda supervisionada. Para isso precisam de três ingredientes do RLHF. Todos estão no DPO (§3 do paper):

### 1.1 Modelo de preferência de Bradley–Terry (BT)

Supõe-se uma recompensa latente $r^*(x,y)$ e (DPO, Eq. 1):

$$
p^*(y_1\succ y_2\mid x)=\frac{\exp r^*(x,y_1)}{\exp r^*(x,y_1)+\exp r^*(x,y_2)}=\sigma\big(r^*(x,y_1)-r^*(x,y_2)\big).
$$

Só a **diferença** de recompensas importa. Logo $r$ e $r+f(x)$ induzem as mesmas preferências (é o Lema 1 do DPO, a "sub-especificação" do Plackett–Luce).

### 1.2 Modelo de recompensa por máxima verossimilhança

Com dados $\mathcal D=\{(x,y_w,y_l)\}$ (DPO, Eq. 2):

$$
\mathcal L_R(r_\phi,\mathcal D)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big].
$$

É uma regressão logística binária. Na prática $r_\phi$ é o modelo SFT com uma cabeça linear escalar no topo. Trabalhos anteriores normalizam as recompensas para média zero por prompt.

### 1.3 Objetivo de RL com KL (DPO, Eq. 3)

$$
\max_{\pi_\theta}\ \mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big]-\beta\,\mathbb D_{\text{KL}}\big[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big].
$$

O KL impede que a política saia da região onde $r_\phi$ é confiável e preserva diversidade (evita *mode collapse*). Como a geração é discreta, isso não é diferenciável e se otimiza com RL. O padrão é PPO com recompensa por amostra $r(x,y)=r_\phi(x,y)-\beta(\log\pi_\theta(y\mid x)-\log\pi_{\text{ref}}(y\mid x))$.

### 1.4 A solução fechada (DPO, Eq. 4, derivação do Apêndice A.1)

Para uma classe de políticas **não paramétrica** (qualquer distribuição), passo a passo:

$$
\begin{aligned}
&\max_\pi\ \mathbb E_x\mathbb E_{y\sim\pi}\Big[r(x,y)-\beta\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]
=\min_\pi\ \mathbb E_x\mathbb E_{y\sim\pi}\Big[\log\tfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}-\tfrac1\beta r(x,y)\Big]\\
&=\min_\pi\ \mathbb E_x\mathbb E_{y\sim\pi}\Big[\log\frac{\pi(y\mid x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac1\beta r(x,y)\big)}-\log Z(x)\Big]
=\min_\pi\ \mathbb E_x\Big[\mathbb D_{\text{KL}}\big(\pi(\cdot\mid x)\,\|\,\pi^*(\cdot\mid x)\big)-\log Z(x)\Big],
\end{aligned}
$$

com

$$
\boxed{\ \pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac1\beta r(x,y)\Big),\qquad Z(x)=\sum_y\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac1\beta r(x,y)\Big).\ }
$$

$Z(x)$ não depende de $\pi$. Pela desigualdade de Gibbs, o KL é mínimo (zero) só quando $\pi=\pi^*$. Intuição para a figura: **$\pi^*$ é a referência "inclinada exponencialmente" pela recompensa** (*exponential tilting*). $\beta$ grande significa inclinação suave (fica perto de $\pi_{\text{ref}}$). $\beta\to0$ significa colapsar no argmax da recompensa. Essa mesma forma aparece em AWR, RWR, AWAC e ReST-EM (§9). É o fio condutor do capítulo de regressão.

**Por que não usar isso direto?** Estimar $Z(x)$ exige somar sobre todas as respostas possíveis, o que é intratável.

### 1.5 A inversão (DPO, Eq. 5)

Tirando o log de 1.4 e isolando $r$:

$$
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x).
$$

Toda recompensa pode ser escrita como "$\beta\times$ log-razão da sua política ótima" **mais um termo que só depende de $x$**. E no BT só diferenças importam (1.1), então esse termo vai cancelar.

---

## 2. DPO — Direct Preference Optimization (Rafailov et al., 2023)

URL: https://arxiv.org/abs/2305.18290 (NeurIPS 2023; arXiv v3 de 29/07/2024)

### 2.1 Derivação da perda

Substitua a Eq. 5 (com $r^*$, $\pi^*$) no BT. O $\beta\log Z(x)$ aparece somado nos dois exponenciais e **cancela** (DPO, Eq. 6, Apêndice A.2):

$$
p^*(y_1\succ y_2\mid x)=\frac{1}{1+\exp\Big(\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\text{ref}}(y_2\mid x)}-\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\text{ref}}(y_1\mid x)}\Big)}.
$$

Agora a preferência está escrita só em termos da política ótima. Basta fazer máxima verossimilhança numa política parametrizada $\pi_\theta$ (DPO, **Eq. 7**):

$$
\boxed{\ \mathcal L_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}})=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]=-\mathbb E\big[\log\sigma(\beta\,\Delta h)\big].\ }
$$

Leitura: **é a mesma regressão logística do modelo de recompensa (Eq. 2), com $r_\phi$ substituído por $\hat r_\theta=\beta\log(\pi_\theta/\pi_{\text{ref}})$**. "Seu modelo de linguagem é secretamente um modelo de recompensa" (título do paper).

Generalização para rankings de $K$ respostas, Plackett–Luce (Apêndice A.3, Eq. 20):

$$
\mathcal L_{\text{DPO}}=-\mathbb E_{\tau,y_1..y_K,x}\Big[\log\prod_{k=1}^{K}\frac{\exp\big(\beta\log\frac{\pi_\theta(y_{\tau(k)}\mid x)}{\pi_{\text{ref}}(y_{\tau(k)}\mid x)}\big)}{\sum_{j=k}^{K}\exp\big(\beta\log\frac{\pi_\theta(y_{\tau(j)}\mid x)}{\pi_{\text{ref}}(y_{\tau(j)}\mid x)}\big)}\Big].
$$

Para $K=2$ volta à Eq. 7. Com $K>2$, cada termo do produto é um **softmax contra todos os piores**. É exatamente a forma InfoNCE (§10.1), a ponte com o capítulo de contraste.

### 2.2 Recompensa implícita e a teoria das classes de equivalência

- **Definição 1 (DPO):** $r\sim r'$ se $r(x,y)-r'(x,y)=f(x)$.
- **Lema 1:** recompensas da mesma classe induzem a mesma distribuição de preferências (Plackett–Luce/BT).
- **Lema 2:** recompensas da mesma classe induzem a **mesma política ótima** no problema com KL. Prova: o $e^{f(x)/\beta}$ sai do numerador e do denominador de 1.4.
- **Teorema 1:** sob hipóteses brandas ($\pi_{\text{ref}}(y\mid x)>0$ para todo par, $\beta>0$), toda classe de equivalência consistente com Plackett–Luce/BT pode ser representada como $r(x,y)=\beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ para algum $\pi$. A projeção é $f(r;\pi_{\text{ref}},\beta)(x,y)=r(x,y)-\beta\log\sum_y\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ (Eq. 8). A Proposição 1 do apêndice mostra que esse representante é **único** em cada classe.
- **Consequência didática:** a reparametrização do DPO não perde generalidade. Ela escolhe, em cada classe, a recompensa cujo $Z(x)=1$ (Eq. 9). É por isso que a política fica analiticamente tratável.

No TRL, as métricas logadas são exatamente essas recompensas implícitas: `rewards/chosen` $=\beta h_\theta(x,y_w)$, `rewards/rejected` $=\beta h_\theta(x,y_l)$, `rewards/margins` (média da diferença) e `rewards/accuracies` (fração de pares com $\hat r_w>\hat r_l$). Também logam `logps/chosen` e `logps/rejected`.

### 2.3 O gradiente e sua leitura mecânica (DPO §4 e Apêndice A.4)

$$
\nabla_\theta\mathcal L_{\text{DPO}}=-\beta\,\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{peso maior quando a recompensa implícita erra}}\Big(\underbrace{\nabla_\theta\log\pi_\theta(y_w\mid x)}_{\text{sobe }y_w}-\underbrace{\nabla_\theta\log\pi_\theta(y_l\mid x)}_{\text{desce }y_l}\Big)\Big].
$$

Pontos para o capítulo:
1. **Peso adaptativo.** $\sigma(-\beta\Delta h)$ vale 0,5 no início ($\pi_\theta=\pi_{\text{ref}}$) e cai para ~0 quando o par já está "bem ordenado". O DPO gasta gradiente nos pares que ainda estão errados. O paper mostra que **sem esse peso** (o baseline "Unlikelihood", que só sobe $\log\pi(y_w)$ e desce $\log\pi(y_l)$) o modelo degenera (DPO, Apêndice, Tabela 3).
2. **Só a margem importa.** A perda depende apenas de $\Delta h$. Ela é indiferente a subir $y_w$ ou descer $y_l$, e não impede que **ambos** caiam, desde que $y_l$ caia mais. Essa é a raiz do *likelihood displacement* (§2.7.2).
3. **Decomposição por token [derivação minha]:** $\nabla\log\pi(y\mid x)=\sum_t\nabla\log\pi(y_t\mid x,y_{<t})$. Se $y_w$ e $y_l$ têm um **prefixo comum**, os termos desse prefixo são idênticos (mesmo contexto, mesmo token) e se cancelam na diferença. O sinal do DPO concentra-se a partir do **primeiro token divergente**. Em pares com pouca distância de edição, quase todo o gradiente cai em poucos tokens. Isso é consistente com a análise do Smaug/DPOP (§6.5), que observa falhas justamente em pares de baixa distância de edição.
4. **Escala.** O fator $\beta$ multiplica o gradiente todo. Um $\beta$ pequeno dá passos pequenos por unidade de $\Delta h$, mas permite ir mais longe da referência antes de saturar.

### 2.4 O papel de $\beta$

- No objetivo 1.3, $\beta$ é o **preço do KL**: maior $\beta$ significa ótimo mais perto de $\pi_{\text{ref}}$.
- Na perda, $\beta$ é a **escala do logit**. Para uma preferência verdadeira $p<1$, o ótimo exige $\beta\Delta h=\operatorname{logit}(p)$, ou seja, $\Delta h^*=\operatorname{logit}(p)/\beta$. Um $\beta$ menor exige um afastamento maior. Para $p=1$ (preferência determinística), $\Delta h^*\to\infty$ **qualquer que seja $\beta$**. É o argumento do IPO (§3.3).
- Valores publicados (todos "como reportados"):
  - DPO original: $\beta=0{,}1$ por padrão e $\beta=0{,}5$ no TL;DR. Batch 64, RMSprop, LR $10^{-6}$ com warmup linear de 150 passos (DPO, Apêndice B).
  - Sweep do DPO no paper do SimPO: $\beta\in\{0{,}01;0{,}05;0{,}1\}$, LR em $\{3,5,6\}\times10^{-7}$ e $10^{-6}$. O melhor DPO por configuração (README do SimPO) foi **$\beta=0{,}01$** com LR $5\times10^{-7}$ (Mistral/Llama-3 Base e Instruct; Llama-3-Instruct com $7\times10^{-7}$).
  - KTO v5 afirma que o LR padrão de DPO na escala 8B é $5\times10^{-7}$, atribuindo ao DPO. O paper do DPO usa $10^{-6}$ nos seus experimentos menores. Ver §14.
  - Tülu 3 (DPO **normalizado por comprimento**): $\beta=5$, LR $5\times10^{-7}$ (8B) e $2\times10^{-7}$ (70B), batch efetivo 128, 1 época, warmup 0,1, comprimento máx. 2.048 (Tülu 3, Tabela 20). O $\beta$ é grande porque a log-razão por token é ~$\lvert y\rvert$ vezes menor que a soma.
  - TRL `DPOConfig`: `beta=0.1`, `learning_rate=1e-6` (padrão), dica de ~`1e-5` com LoRA.

### 2.5 O modelo de referência na prática

- **Escolha canônica:** $\pi_{\text{ref}}=\pi_{\text{SFT}}$ (o mesmo ponto de partida de $\pi_\theta$). Se os dados de preferência vieram de outro modelo e não há $\pi_{\text{SFT}}$, o paper recomenda inicializar $\pi_{\text{ref}}=\arg\max_\pi\mathbb E_{x,y_w\sim\mathcal D}[\log\pi(y_w\mid x)]$ ("Preferred-FT") para reduzir a mudança de distribuição (DPO §4, "DPO outline").
- **Custo:** um passo de DPO faz 4 forwards por par (política e referência em $y_w$ e $y_l$). Só a política tem backward (ORPO §7.3 contabiliza "quatro forward passes").
- **Truques de memória:**
  - *Pré-computar* $\log\pi_{\text{ref}}$ uma vez para todo o dataset: TRL `precompute_ref_log_probs=True`, e o Tülu 3 "cacheia" os logprobs da referência em vez de manter o modelo na GPU (Tülu 3, §"Caching DPO Log Probs").
  - *LoRA:* a referência é a própria base com os adaptadores **desligados**. O TRL faz isso automaticamente quando `ref_model=None` e o modelo é PEFT.
  - Forwards separados para chosen/rejected reduzem o pico de memória (Tülu 3, Fig. 17).
- **Referência móvel (TR-DPO, arXiv 2404.09656):** atualizar $\pi_{\text{ref}}$ durante o treino mitiga sobre-otimização. No TRL: `sync_ref_model=True`, `ref_model_mixup_alpha=0.6`, `ref_model_sync_steps=512`, com mistura de pesos $\pi_{\text{ref}}\leftarrow\alpha\pi_\theta+(1-\alpha)\pi_{\text{ref,prev}}$.
- **Iterativo:** no IRPO (§2.6), a referência de cada iteração é o modelo da iteração anterior.

### 2.6 Offline vs. online/iterativo

- **DPO "puro" é offline:** o dataset é fixo, gerado por outro modelo ($\rho$). OAIF (Guo et al., 2024, arXiv 2402.04792) separa dois desvios: (i) inicial, $\rho\neq\pi_{\theta_0}$; (ii) gradual, $\pi_{\theta_0}\neq\pi_{\theta_t}$. O DPO offline é, portanto, **off-policy**.
- **Online DPO / OAIF (Alg. 1):** a cada passo, amostrar $x$, gerar $y_1,y_2\sim\pi_{\theta_t}(\cdot\mid x)$, pedir a um **LLM anotador** qual é melhor (avaliando as duas ordens e tirando a média, contra viés de posição) e dar um passo da perda DAP (DPO, IPO ou SLiC). O gradiente para no amostrador e no anotador (*stop_gradient*). Resultados reportados: a versão online vence a offline em ~66% das avaliações humanas (média sobre DPO/IPO/SLiC). Online DPO vence SFT, RLHF e RLAIF 58,00% das vezes numa comparação de 4 vias no TL;DR. Pedir ao anotador respostas curtas reduz o comprimento médio de ~120 para ~40 mantendo ganho sobre o SFT.
- **Iterativo (lotes):** gerar → rotular (RM ou juiz) → DPO → repetir.
  - *Iterative DPO / GSHF* (Xiong et al., arXiv 2312.11456).
  - *Self-Rewarding LMs* (Yuan et al., arXiv 2401.10020), em que o próprio modelo julga.
  - *IRPO* (Pang et al., arXiv 2404.19733), para raciocínio: pares (CoT correto, CoT errado) e perda DPO + NLL do vencedor normalizada pelo comprimento, $\mathcal L=\mathcal L_{\text{DPO}}+\alpha\,\frac{-\log M_\theta(c_w,y_w\mid x)}{\lvert c_w\rvert+\lvert y_w\rvert}$, com $M_t$ (iteração anterior) como referência. O termo NLL é "crucial" segundo os autores.
  - *RSO* (arXiv 2309.06657): amostrar pares da política ótima estimada via *rejection sampling*.
  - "Is DPO superior to PPO?" (Xu et al., arXiv 2404.10719): o "DPO-Iter" reduz o desvio de distribuição e melhora segurança, mas fica bem abaixo do PPO em utilidade naquele estudo.
- **Implementação:** TRL `trl.experimental.online_dpo.OnlineDPOTrainer` (`loss_type` ∈ {`sigmoid`, `ipo`}, `reward_model_path` ou juiz, `missing_eos_penalty`, `temperature=0.9`, `max_new_tokens=64` por padrão, `use_vllm` com `vllm_mode` `colocate`/`server`). **Online exige geração rápida.** É aqui que o Módulo 1 (vLLM) volta (§13).

### 2.7 Modos de falha conhecidos

#### 2.7.1 Sobreajuste a preferências determinísticas (o argumento do IPO)
Se num par $p^*(y\succ y')=1$, o BT exige $r(y)-r(y')\to\infty$, e a política ótima zera $\pi(y')$ **qualquer que seja $\beta$**. Com dados finitos, um $\hat p$ empírico igual a 1 é comum (um único rótulo por par!). Detalhes e números em §3.3.

#### 2.7.2 *Likelihood displacement* (deslocamento de verossimilhança)
- **Fenômeno:** durante o DPO, $\log\pi_\theta(y_w\mid x)$ frequentemente **cai** (junto com $\log\pi_\theta(y_l\mid x)$). A massa vai para *outras* respostas, possivelmente de significado oposto.
- **Razin et al. (ICLR 2025, arXiv 2410.08847):**
  - Treinar para preferir *No* a *Never* pode aumentar muito a probabilidade de *Yes*.
  - Num alinhamento para **recusar** prompts inseguros (SORRY-Bench), a taxa de recusa do Llama-3-8B-Instruct caiu de **74,4% para 33,4%**: "desalinhamento não intencional".
  - **Teoria (Teorema 1, caso de um token):** a queda de $\ln\pi(y^+\mid x)$ é maior quanto maior $\langle W_{y^+},W_{y^-}\rangle$ (unembeddings parecidos) mais o alinhamento dos outros tokens com $W_{y^+}-W_{y^-}$.
  - **Caso geral:** o escore **CHES** (*centered hidden embedding similarity*, Def. 2) é $\mathrm{CHES}_x(y^+,y^-)=\big\langle\sum_k h_{x,y^+_{<k}},\sum_{k'}h_{x,y^-_{<k'}}\big\rangle-\big\|\sum_k h_{x,y^+_{<k}}\big\|^2$. Ele identifica os pares que mais causam deslocamento. **Filtrar os pares de CHES alto** (versão normalizada por comprimento) mitigou o problema melhor que adicionar um termo SFT à perda.
- **Smaug/DPOP (Pal et al., arXiv 2402.13228):** mostra teoricamente que o DPO pode reduzir a verossimilhança das preferidas, especialmente em pares de **baixa distância de edição**, e propõe DPO-Positive (§6.5).
- **Exemplo de brinquedo** para animar: §12.10.

#### 2.7.3 Verbosidade / exploração de comprimento
- **Park et al. (arXiv 2403.19159):** nos datasets HH e TL;DR há um pequeno viés de preferência por respostas mais longas. O DPO sem regularização produz respostas **~2× mais longas** em média, fora da distribuição do dataset. No HH com $\beta=0{,}1$, já nos primeiros 10% da época as respostas ficam quase 2× mais longas que as do SFT. O pico de *win rate* (GPT-4 como juiz, que tem viés de verbosidade) ocorre aí, e depois só KL e comprimento continuam subindo. Solução: R-DPO (§6.3).
- **SimPO, Apêndice F:** o gradiente do DPO não é normalizado por comprimento. Sequências longas recebem atualizações maiores e "dominam" o treino.
- **Rafailov et al. 2024 (DAAs):** adicionar correção de comprimento muda a fronteira KL–*win rate*, mas **não** elimina a sobre-otimização (pode até piorar). Modelos de menor capacidade (pequenos ou com KL apertado) "extrapolam" mais em atributos simples como comprimento.

#### 2.7.4 Sobre-otimização (o "reward hacking" das DAAs)
**Rafailov, Chittepu, Park et al. (NeurIPS 2024, arXiv 2406.02900).** *Direct Alignment Algorithms* (DPO, IPO, SLiC) foram testados em TL;DR com Pythia 1B/2,8B/6,9B.
- **Padrão em corcova:** a *win rate* sobe e depois cai à medida que o orçamento de KL cresce.
- **Pico precoce:** com KL mais folgado, o melhor ponto ocorre após apenas **~25% de uma época**.
- **Lei de escala:** o ajuste $R(d)=d(\alpha-\beta\log d)$ com $d=\sqrt{\mathbb D_{\text{KL}}(\pi\|\pi_{\text{ref}})}$, herdado de Gao et al. 2022 para RLHF clássico (arXiv 2210.10760), também descreve as DAAs. Ele reduz pela metade o RMSE em relação a um ajuste quadrático em $\mathbb D_{\text{KL}}$, embora um ajuste quadrático em $d$ dê erro parecido.
- **IPO** parece menos propenso à sobre-otimização, sobretudo a 6,9B.
- A **acurácia da recompensa implícita** praticamente não se correlaciona com a qualidade final (DPO, SLiC). No IPO há uma correlação fraca, explicada pela escala do modelo.
- **Explicação proposta:** o problema é sub-determinado. A matriz de "consultas" (pares do dataset) tem núcleo não trivial em $\lvert\mathcal X\times\mathcal Y\rvert$, logo há infinitos minimizadores que põem massa em respostas **fora da distribuição** (Proposição 1, adaptada do CPL).

#### 2.7.5 Exploração fora da distribuição (DPO vs PPO)
**Xu et al. (arXiv 2404.10719), Teorema 4.1:** a classe de políticas obtidas por "RM + PPO" é **subconjunto próprio** da classe de minimizadores do DPO. O DPO pode achar soluções que o PPO acharia **e outras** que exploram respostas OOD. A performance do DPO é sensível ao desvio entre as saídas do modelo e o dataset. No estudo, o PPO bem configurado (normalização de vantagem, batch grande, EMA da referência) supera o DPO em todos os testes, inclusive código competitivo: o PPO com 34B supera o AlphaCode-41B no CodeContest, com 10@1k indo de 16,4% (AlphaCode-41B) para 22,4%. **[interpretação]** Isso não "invalida" o DPO, mas mostra que a escolha depende de quanto o dataset cobre a distribuição da política.

#### 2.7.6 Desalinhamento entre recompensa implícita e métrica de geração
**SimPO §2.2:** depois de DPO, só **~50%** dos triplos de treino têm $\bar p_\theta(y_w)>\bar p_\theta(y_l)$ na log-prob **média**, que é a métrica que a decodificação efetivamente usa. A recompensa implícita depende de $\pi_{\text{ref}}$, a geração não.

### 2.8 Hiperparâmetros práticos e monitoramento

| Item | Valores publicados | Fonte |
|---|---|---|
| $\beta$ | 0,1 (padrão), 0,5 (TL;DR) / 0,01 (melhor em UltraFeedback 7–9B) / 5 (DPO normalizado por comprimento) | DPO App. B / README SimPO / Tülu 3 Tab. 20 |
| LR | $10^{-6}$ RMSprop (DPO); $3\times10^{-7}$ a $10^{-6}$ AdamW (sweep SimPO); $5\times10^{-7}$ (8B) e $2\times10^{-7}$ (70B) (Tülu 3) | idem |
| Batch | 64 (DPO); 128 (SimPO e Tülu 3) | idem |
| Épocas | 1 é o mais comum. O melhor ponto pode vir antes (~25% da época em regimes de KL folgado) | SimPO App. B; Tülu 3; DAAs |
| Warmup | 150 passos lineares (DPO); 10% com cosseno (SimPO); razão 0,1 linear (Tülu 3) | idem |

O que monitorar (métricas do TRL):
- Margem de recompensas e acurácia.
- `logps/chosen` **caindo** (deslocamento).
- Comprimento médio das gerações (verbosidade).
- KL até a referência.
- *Win rate* em avaliação **por checkpoint**, não só no fim, por causa da corcova de sobre-otimização.

### 2.9 Implementação mínima (pseudocódigo PyTorch, minha versão)

```python
import torch.nn.functional as F

def seq_logps(logits, labels, mask):
    # logits: [B,T,V] já deslocados; labels: [B,T]; mask=1 nos tokens da RESPOSTA
    lp = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1), mask.sum(-1)          # soma e comprimento |y|

def dpo_loss(pw, pl, rw, rl, beta=0.1):               # p*: política, r*: referência (somas)
    delta = (pw - rw) - (pl - rl)                      # Δh
    loss = -F.logsigmoid(beta * delta)
    return loss.mean(), (beta * (pw - rw)).detach(), (beta * (pl - rl)).detach()
```

---

## 3. IPO e o arcabouço ΨPO (Azar et al., 2023)

URL: https://arxiv.org/abs/2310.12036 ("A General Theoretical Paradigm to Understand Learning from Human Preferences", Google DeepMind; AISTATS 2024)

### 3.1 Nota sobre o nome
- O pedido do usuário chama o IPO de "Identity Policy Optimization".
- O paper define a família **ΨPO** = "Ψ-*preference optimisation*" e chama o caso $\Psi=$ identidade de **"Identity-PO (IPO)"**, sem expandir o "PO". Pela construção (ΨPO ⇒ *preference optimisation*), a expansão canônica é **Identity Preference Optimisation**.
- O nome "identity **policy** optimisation" **também circula na literatura**: o paper OAIF (Guo et al., 2024, com coautores do IPO como Bilal Piot) escreve exatamente "identity policy optimisation (IPO, Azar et al., 2023)".
- **Recomendação:** usar "IPO (Identity Preference Optimization)" e mencionar que a variante "Identity Policy Optimization" aparece em trabalhos posteriores. **O método é o mesmo.**

### 3.2 O objetivo ΨPO

Sejam $\Psi:[0,1]\to\mathbb R$ não decrescente, $\mu$ a política que gerou os pares e $\tau>0$ (IPO, **Eq. 6**):

$$
\max_\pi\ \mathbb E_{x\sim\rho,\ y\sim\pi(\cdot\mid x),\ y'\sim\mu(\cdot\mid x)}\big[\Psi\big(p^*(y\succ y'\mid x)\big)\big]-\tau\,\mathbb D_{\text{KL}}(\pi\|\pi_{\text{ref}}).
$$

O objetivo maximiza uma função (possivelmente não linear) da **probabilidade de preferência** contra a política de dados, com KL. Pela mesma derivação de 1.4 (IPO, Eq. 7):

$$
\pi^*(y)\ \propto\ \pi_{\text{ref}}(y)\exp\Big(\tau^{-1}\,\mathbb E_{y'\sim\mu}\big[\Psi(p^*(y\succ y'))\big]\Big).
$$

**Proposição 1 (IPO):** com $\Psi(q)=\log\frac{q}{1-q}$ (o *logit*) e o BT válido, $\mathbb E_{y'}[\Psi(p^*(y\succ y'))]=r(y)-\mathbb E_{y'}[r(y')]$. Ou seja, a recompensa a menos de uma constante. Então **RLHF (Eq. 3) e DPO (Eq. 5 do IPO) têm a mesma política ótima que o ΨPO com Ψ = logit**. RLHF e DPO são *casos particulares* de ΨPO.

### 3.3 Por que o DPO sobreajusta (IPO §4.2)

- **Ψ = logit é ilimitado.** Levar a probabilidade de preferência de 0,98 para 0,99 é tão incentivado quanto de 0,5 para 0,6 (em logit).
- **Preferências determinísticas:** com $p^*(y\succ y')=1$, o BT exige $r(y)-r(y')\to+\infty$ e então $\pi^*(y')/\pi^*(y)=0$, **para qualquer $\tau$**. "A regularização KL fica cada vez mais fraca quanto mais determinísticas são as preferências."
- **Dados finitos agravam:** mesmo com $p^*=0{,}8$, poucas amostras dão $\hat p=1$ com frequência, e o ótimo empírico zera $\pi(y')$ para qualquer $\tau$.
- **Por que o RLHF clássico sofre menos:** o modelo de recompensa é **subajustado** (recompensas infinitas nunca são atingidas e há regularização/parada antecipada no RM). Esse subajuste regulariza a política. O DPO, ao eliminar o RM, perde essa regularização implícita.

### 3.4 Derivação do IPO (IPO §5)

$\Psi=$ identidade dá um objetivo limitado em $[0,1]$ (IPO, Eq. 8):

$$
\max_\pi\ p^*_\rho(\pi\succ\mu)-\tau\,\mathbb D_{\text{KL}}(\pi\|\pi_{\text{ref}}).
$$

**Equações de raiz.** Sejam $g(y)=\mathbb E_{y'\sim\mu}[\Psi(p^*(y\succ y'))]$ e

$$
h_\pi(y,y')=\log\frac{\pi(y)\,\pi_{\text{ref}}(y')}{\pi(y')\,\pi_{\text{ref}}(y)}\quad(=\Delta h\text{ na notação do §0}).
$$

O ótimo satisfaz $h^*(y,y')=\tau^{-1}(g(y)-g(y'))$ (Eq. 11). Para $\Psi=I$, isso vira $h_\pi(y,y')=\tau^{-1}(p^*(y\succ\mu)-p^*(y'\succ\mu))$.

**Perda quadrática (Eq. 13):**

$$
L(\pi)=\mathbb E_{y,y'\sim\mu}\Big[\Big(h_\pi(y,y')-\frac{p^*(y\succ\mu)-p^*(y'\succ\mu)}{\tau}\Big)^2\Big].
$$

- **Teorema 2 (unicidade):** se $\operatorname{Supp}(\mu)=\operatorname{Supp}(\pi_{\text{ref}})$, $L$ tem **um único mínimo local/global** $\pi^*$ no conjunto de políticas com esse suporte. Em logits $s$, $L$ é uma quadrática PSD, estritamente convexa exceto na direção $(1,\dots,1)$, que não muda o softmax.
- **Perda amostrada (Eq. 16) e Proposição 3:** trocar $p^*$ por um rótulo Bernoulli $I(y,y')$ dá a mesma perda a menos de constante, $\mathbb E\big[(h_\pi(y,y')-\tau^{-1}I(y,y'))^2\big]$.
- **Usando a simetria** (cada par contribui com $(y_w,y_l,1)$ e $(y_l,y_w,0)$):

$$
\tfrac12\big[(h-\tau^{-1})^2+h^2\big]=\Big(h-\frac{\tau^{-1}}{2}\Big)^2+\frac{1}{4\tau^2}\quad\textbf{[conferência algébrica minha]}
$$

o que leva à perda final (**IPO, Eq. 17**, Algoritmo 1):

$$
\boxed{\ \mathcal L_{\text{IPO}}=\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\Big(\log\frac{\pi_\theta(y_w\mid x)\,\pi_{\text{ref}}(y_l\mid x)}{\pi_\theta(y_l\mid x)\,\pi_{\text{ref}}(y_w\mid x)}-\frac{1}{2\tau}\Big)^2\Big]=\mathbb E\Big[\Big(\Delta h-\frac{1}{2\tau}\Big)^2\Big].\ }
$$

**Leitura:** o IPO **regride** a margem de log-razões para o alvo **finito** $1/(2\tau)$ em vez de empurrá-la ao infinito. Começa em $\pi=\pi_{\text{ref}}$.

### 3.5 O gradiente: a diferença qualitativa em relação ao DPO

$$
\nabla_\theta\mathcal L_{\text{IPO}}=2\,\mathbb E\Big[\Big(\Delta h-\frac{1}{2\tau}\Big)\big(\nabla\log\pi_\theta(y_w\mid x)-\nabla\log\pi_\theta(y_l\mid x)\big)\Big]\quad\textbf{[derivação minha]}.
$$

- Se $\Delta h<1/(2\tau)$, empurra para cima (como o DPO).
- Se $\Delta h>1/(2\tau)$, **puxa de volta**. O DPO nunca faz isso: o peso dele só tende a zero.
- **Gráfico ideal para animar:** a perda do DPO é uma "rampa" decrescente em $\Delta h$, a do IPO é uma **parábola** com mínimo em $1/(2\tau)$ (§12.3).

### 3.6 Exemplos do paper e simulação

- **Assintótico, 2 ações (IPO §5.3.1):** $p^*(y_1\succ y_2)=1$, $\pi_{\text{ref}}$ e $\mu$ uniformes. O DPO converge para $\pi(y_1)=1$ para qualquer $\tau$. O IPO dá $p^*(y_1\succ\mu)=3/4$, $p^*(y_2\succ\mu)=1/4$ e
  $$\pi^*(y_1)=\frac{e^{0{,}75/\tau}}{e^{0{,}75/\tau}+e^{0{,}25/\tau}}=\sigma\big(0{,}5\,\tau^{-1}\big).$$
  Valores (calculados): $\tau=0{,}1\Rightarrow0{,}9933$; $0{,}25\Rightarrow0{,}8808$; $0{,}5\Rightarrow0{,}7311$; $1\Rightarrow0{,}6225$; $2\Rightarrow0{,}5622$; $5\Rightarrow0{,}5250$. Com $\tau\to\infty$ volta ao uniforme, com $\tau\to0$ vira determinístico. **$\tau$ volta a controlar a distância da referência.**
- **Amostrado, 3 ações (IPO §5.4):** $\mathcal D_1=\{(a,b),(b,c),(a,c)\}$ (ordem total), política softmax com $\theta\in\mathbb R^3$, Adam LR 0,01, 18.000 passos, mini-batch 9. O DPO converge ao determinístico para todo $\tau$, o IPO não. Num segundo cenário, uma ação **sem nenhuma vitória** é zerada pelo DPO para todo $\tau$, e o IPO a mantém perto da referência.
- **Reprodução minha** (mesmo setup, *full batch*, valores em §12.4): com $\tau=1$ após 18.000 passos, DPO dá $\pi=(1{,}0;\,0{,}0;\,0{,}0)$ e IPO dá $\pi=(0{,}4484;\,0{,}3213;\,0{,}2302)$.
- **Solução fechada do IPO em $\mathcal D_1$ [derivação minha]:** minimizar $(d_1-c)^2+(d_2-c)^2+(d_1+d_2-c)^2$ com $c=1/(2\tau)$ dá $d_1=d_2=2c/3=1/(3\tau)$, ou seja, $\theta_a-\theta_b=\theta_b-\theta_c=1/(3\tau)$. Os números simulados batem com isso.

### 3.7 $\tau$ na prática, e uma armadilha de escala

- **TRL:** `beta` = $\tau$ para `loss_type="ipo"`. O código atual **normaliza por token**: usa a log-razão média por token de cada resposta antes de subtrair $1/(2\beta)$. O comentário no código diz que isso não está explícito no paper, mas foi confirmado com os autores ("os resultados do paper correspondem a essa forma normalizada").
- **Consequência:** a escala de $\Delta h$ por token é ~$\lvert y\rvert$ vezes menor. Um $\tau=0{,}1$ (alvo 5 nats/token!) é enorme na versão normalizada, e $\tau\in[0{,}5;\,2]$ dá alvos de 1 a 0,25 nat/token. Veja os números em §12.3.
- **Sweeps publicados:** $\tau\in\{0{,}01;0{,}1;0{,}5;1{,}0\}$ (sweep do SimPO, Tabela 7). Na comparação de DAAs, IPO 6,9B mostrou "pouca ou nenhuma" sobre-otimização.
- **Quando usar:** quando os rótulos são **determinísticos/únicos por par**, há poucos dados ou ruído, e você quer um ancoramento real em $\pi_{\text{ref}}$ sem depender de parada antecipada. **Risco:** sub-ajuste se $1/(2\tau)$ for pequeno demais. Smaug reporta que IPO "não melhora" num dataset de matemática de baixa edição.

---

## 4. KTO — Kahneman–Tversky Optimization (Ethayarajh et al., 2024)

URL: https://arxiv.org/abs/2402.01306 (ICML 2024; **v5 de 08/09/2026**, usada aqui)

### 4.1 Teoria da perspectiva (*prospect theory*) em 5 minutos

- **Ideia central:** humanos não maximizam valor esperado. Eles avaliam resultados **relativos a um ponto de referência** e são **avessos à perda**.
- **Exemplo do paper:** uma aposta que paga \$100 com 80% e \$60 com 20% tem valor esperado \$80, e mesmo assim muita gente aceita \$60 certos para evitá-la.
- **Componentes (Def. 3.1–3.3):** uma **função de valor** $v$ (resultado → valor subjetivo, relativo a $z_0$) e uma **função de ponderação** $\omega$ (distorção de probabilidades, por exemplo superestimar eventos raros). A utilidade é $u(Z)=\sum_z\omega_z\,v(z-z_0)$. O KTO ignora $\omega$, porque humanos não veem a distribuição completa do LLM.
- **Forma de Tversky & Kahneman (1992)**, KTO **Eq. 4**:

$$
v(z;\lambda,\alpha,z_0)=\begin{cases}(z-z_0)^{\alpha} & z\ge z_0\\ -\lambda\,(z_0-z)^{\alpha} & z<z_0\end{cases}\qquad \alpha\approx0{,}88,\ \lambda\approx2{,}25\ \text{(medianas)}.
$$

- $\alpha<1$ dá concavidade nos ganhos (aversão a risco) e convexidade nas perdas. $\lambda>1$ é a **aversão à perda** (a curva é mais íngreme abaixo de $z_0$).
- **Números para a figura:** $v(1)=1$; $v(-1)=-2{,}25$; $v(10)=10^{0{,}88}\approx7{,}586$; $v(-10)\approx-17{,}068$.
- O formato varia entre indivíduos. O que importa é: ponto de referência, concavidade nos ganhos e aversão à perda.

### 4.2 HALOs — *human-aware losses* (Def. 3.4)

- **Ingredientes:** $\pi_\theta$, $\pi_{\text{ref}}$, um normalizador $l:\mathcal Y\to\mathbb R^+$ e a recompensa implícita $r_\theta(x,y)=l(y)\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$.
- **Valor humano:** dados uma distribuição de ponto de referência $Q(Y'\mid x)$ e $v_{x,y}$ não decrescente, o valor de $(x,y)$ é $v_{x,y}\big(r_\theta(x,y)-\mathbb E_Q[r_\theta(x,Y')]\big)$ (Eq. 5).
- **Definição de HALO:** $f$ é uma HALO se existem $a_{x,y}\in\mathbb R\setminus\{0\}$ tais que $f=\mathbb E_{x,y\sim\mathcal D}\big[a_{x,y}\,v_{x,y}(r_\theta(x,y)-\mathbb E_Q[r_\theta(x,Y')])\big]+C_{\mathcal D}$ (Eq. 6).
- **Restrições:** $l$, $a$, $v$ e $C$ não podem depender de $\pi_\theta$ ou $\pi_{\text{ref}}$. Só $Q$ pode.
- **Conexão com o DPO:** com $l(\cdot)=\beta$, no ótimo $r_{\theta^*}(x,y)=r^*(x,y)-\beta\log Z(x)$ (Eq. 7), que está na mesma classe de equivalência que $r^*$ (Lema 1 do DPO).
- **Teorema 3.5:** **DPO e PPO-Clip são HALOs.** CSFT (SFT condicional com token de controle) e SLiC **não são**, porque a CE/NLL varia com $\pi_{\text{ref}}$ ou viola a invariância.
- **Experimento:** as HALOs empatam ou superam as não-HALOs em todas as escalas (Pythia 1,4–12B, Llama 7–30B). A diferença só é significativa (p<0,05) a partir de 13B.
- **Curiosidade útil:** um **PPO offline com recompensas fictícias +1/−1** iguala o DPO em quase todas as escalas (exceto Llama-30B). Isso sugere que "o viés indutivo certo" importa mais que a recompensa aprendida. Esse resultado motivou o KTO.

### 4.3 A perda KTO (Eq. 8)

Três mudanças em relação à forma de Kahneman–Tversky:
1. **Logística no lugar da potência** (a potência $\alpha$ é numericamente instável). Ela continua côncava nos ganhos e convexa nas perdas. Um **$\beta$** controla a saturação: maior $\beta$ dá mais aversão a risco nos ganhos e mais propensão a risco nas perdas. Na prática, o $\beta$ tem efeito parecido com o do DPO.
2. **Aversão à perda separada em $\lambda_D$ e $\lambda_U$** (pesos para desejáveis e indesejáveis).
3. **Ponto de referência = KL da política:** o humano julga $y$ contra *todas* as saídas possíveis, logo $Q=\pi_\theta$ e $z_0=\mathrm{KL}(\pi_\theta(y'\mid x)\,\|\,\pi_{\text{ref}}(y'\mid x))$.

$$
\boxed{\ \mathcal L_{\text{KTO}}(\pi_\theta,\pi_{\text{ref}})=\mathbb E_{x,y\sim\mathcal D}\big[\lambda_y-v(x,y)\big],\quad
v(x,y)=\begin{cases}\lambda_D\,\sigma\big(\beta(r_\theta(x,y)-z_0)\big) & y\sim y_{\text{desejável}}\mid x\\[2pt] \lambda_U\,\sigma\big(\beta(z_0-r_\theta(x,y))\big) & y\sim y_{\text{indesejável}}\mid x\end{cases}\ }
$$

com $r_\theta(x,y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ e $\lambda_y=\lambda_D$ ou $\lambda_U$ conforme o rótulo.
- **Não há backprop em $z_0$.** Ele só controla a saturação.
- A instância de HALO é $l=1$, $Q=\pi_\theta$, $v_{x,y}(z)=\sigma(\beta z)$, $a_{x,y}=-\lambda_y$. O $\lambda_y$ somado só torna a perda não negativa.
- **Intuição do paper:** se o modelo sobe a recompensa de um exemplo desejável "de forma bruta" (subindo tudo), o KL (ponto de referência) também sobe e nada se ganha. O modelo é forçado a aprender o que torna aquela saída boa.
- A tabela do SimPO escreve $z_{\text{ref}}=\mathbb E[\beta\,\mathrm{KL}]$ com $\beta$ fora do sigmoide. É algebricamente a mesma coisa.

### 4.4 Estimativa do ponto de referência $z_0$

- **Problema:** amostrar de $\pi_\theta$ é lento.
- **Solução:** usar **pares desencontrados no microbatch**, $\{(x_1,y_2),(x_2,y_3),\dots,(x_m,y_1)\}$, com $j=(i\bmod m)+1$:

$$
\hat z_0=\max\Big(0,\ \frac1m\sum_{1\le i\le m}\log\frac{\pi_\theta(y_j\mid x_i)}{\pi_{\text{ref}}(y_j\mid x_i)}\Big).
$$

- O estimador é **enviesado** (o *clamp* em 0 adiciona viés para cima e reduz variância) e custa **um forward extra**.
- Usam-se saídas desencontradas porque as próprias $y_i$ foram escolhidas como canonicamente boas ou ruins, com recompensas de magnitude não representativa.
- **Quando dá para pular ($\hat z_0=0$):** se o KTO vier depois de um SFT nos mesmos dados desejáveis, com o SFT como referência, o KL estimado tende rapidamente a ~0. Se não houve SFT, ou se o SFT usou outros dados, é necessário estimar.
- **No TRL** (`KTOTrainer`):
  - KL = média (agregada entre dispositivos) de $\log\pi_\theta-\log\pi_{\text{ref}}$ nos completamentos desencontrados, `.clamp(min=0)`, *detached*.
  - Os completamentos desencontrados são construídos com deslocamento de +1 no lote. Isso exige **`train_sampling_strategy="sequential"`** (é o padrão do `KTOConfig`).
  - Exige batch por dispositivo **> 1**. Com 1, o KL vira a própria recompensa.

### 4.5 Dados binários e desbalanceamento

- **Natureza dos dados:** o KTO só precisa de rótulo **desejável/indesejável por exemplo** (👍/👎). Pares de preferência viram binário assumindo $y_w$ desejável e $y_l$ indesejável. O paper chama isso de suposição ingênua.
- **"one-y-per-x":** usa só uma das respostas de cada par, removendo qualquer pareamento.
- **Ajuste de $\lambda_D,\lambda_U$ (Eq. 9):** com $n_D$ desejáveis e $n_U$ indesejáveis, empiricamente é melhor que

$$
\frac{\lambda_D\,n_D}{\lambda_U\,n_U}\in\Big[1,\ \frac32\Big].
$$

- **Exemplo do paper:** com razão 1:10 (desejável:indesejável), $\lambda_U=1$ e $\lambda_D\in[10,15]$. Isso implica que, após corrigir o desbalanceamento, a sensibilidade a ganhos rende mais que a sensibilidade a perdas na maioria dos benchmarks.
- **Exceção:** em tarefas onde **evitar o pior** importa mais (por exemplo, toxicidade), $\lambda_Dn_D<\lambda_Un_U$ pode ser melhor. O padrão é $\lambda_D=\lambda_U=1$.
- **Resultados reportados:**
  - O KTO lida com desbalanceamento extremo, **igualando o DPO com até 90% menos exemplos desejáveis**.
  - Com um pré-treino bom o suficiente, dá para **pular o SFT** e ir direto ao KTO. Com DPO, o SFT é sempre necessário para o melhor resultado.
  - Em Zephyr-β-SFT/UltraFeedback (Tabela 2): KTO (β=0,1, λ_D=1) teve GSM8k 53,5 contra DPO 40,0 e SFT 39,0. A variante *one-y-per-x* teve 50,0. Ablações que pioram: sem $z_0$, 49,5; valor côncavo $v=\log\sigma$, 42,5; **neutro a risco ($v(\cdot)=\cdot$): 42,0 em GSM8k e BBH desaba para 6,1**.

### 4.6 Teoria: por que o KTO funciona com sinal mais fraco (§4.4)

- **Proposição 4.1:** se a recompensa ajustada pela referência de um exemplo vai a $\pm\infty$ (com $\|\nabla\log\pi\|$ limitado), a contribuição dele ao gradiente tende a 0. Exemplos "fáceis demais" ou "difíceis demais" (possivelmente mal rotulados) são **ignorados**. Isso é robustez a ruído, com risco de **sub-ajuste** a distribuições complexas, mitigável com $\beta$ menor e mais épocas.
  - **Gradiente [derivação minha]:** para um exemplo desejável, $\nabla_\theta(\lambda_D-\lambda_D\sigma(\beta(r_\theta-z_0)))=-\lambda_D\beta\,\sigma(\cdot)(1-\sigma(\cdot))\nabla_\theta r_\theta$. O fator $\sigma(1-\sigma)$ é a "janela" que se fecha nas pontas.
- **Teorema 4.2:** com valor logístico e $z_0$ fixo, existem recompensas na mesma classe de equivalência (mesma política ótima, mesmo BT) com **distribuições de valor diferentes**. Maximizar a verossimilhança de preferências não identifica a utilidade "prospect-theoretic".
- **Teorema 4.3 (preferências contraditórias):** com $y_a\succ y_b$ em proporção $p\in(0{,}5;1)$ e o contrário em $1-p$, se $p^{1/\beta}\pi_{\text{ref}}(y_a\mid x)<(1-p)^{1/\beta}\pi_{\text{ref}}(y_b\mid x)$, a política ótima do DPO dá **mais** probabilidade à resposta minoritária $y_b$. A atualização do KTO com $\lambda_D=\lambda_U$ aumenta a probabilidade relativa da majoritária. O KTO tem melhor pior caso sob intransitividade.
- **Quando usar (§4.5 do paper):**
  - Feedback binário, sobretudo desbalanceado: KTO é a escolha natural.
  - Preferências com pouco ruído e pouca intransitividade: o DPO tende a ser melhor (o KTO pode sub-ajustar).
  - Muito ruído ou intransitividade: o melhor pior caso do KTO vence. Datasets públicos (SHP, OASST, até UltraFeedback sintético) costumam ter ruído desse tipo.

### 4.7 Hiperparâmetros (§4.2, v5)

- **Learning rate (o mais sensível):**
  - O ótimo do KTO costuma ser **2× a 10×** o do DPO, porque a recompensa ajustada é menor em magnitude.
  - Padrão recomendado: **5e-6 com AdamW** (com ou sem SFT prévio, com modelo já instruído e com LoRA). Modelos menores ou maiores pedem LR menor ou maior.
  - **Diagnóstico:** se o LR é agressivo demais, as recompensas implícitas de desejáveis **e** indesejáveis caem. O ideal é só as indesejáveis caírem.
  - Atenção: nos experimentos do paper usou-se o LR do DPO com RMSProp, para comparação justa. A recomendação prática é outra.
- **Batch:** microbatch **≥ 2** (para estimar $z_0$ num passo). Os experimentos usam batch efetivo 32. A recomendação é entre 8 e 128.
- **β (aversão a risco):**
  - 0,1 é o padrão que funciona bem.
  - Use $[0{,}01;0{,}10)$ se o modelo já foi ajustado nos mesmos dados.
  - Use $(0{,}10;0{,}50]$ se a referência já é muito boa (por exemplo, já alinhada, como Llama-3-8B-Instruct).
- **Tabela 1 (UltraFeedback, AdamW, batch efetivo 32, clip de gradiente 10, $\lambda_D=\lambda_U=1$):**

| Modelo | Método | LR | β |
|---|---|---|---|
| Llama-3 8B | SFT+KTO | 5e-6 | 0,05 |
| Llama-3 8B | KTO | 5e-6 | 0,10 |
| Llama-3 8B (LoRA) | SFT+KTO | 5e-6 | 0,05 |
| Llama-3 8B Instruct | KTO | 5e-6 | 0,25 |
| Qwen2.5 3B Instruct | SFT+KTO | 5e-6 | 0,10 |
| Qwen2.5 3B Instruct | KTO | 5e-6 | 0,50 |

- **TRL `KTOConfig`:** `beta=0.1`, `desirable_weight=1.0` ($\lambda_D$), `undesirable_weight=1.0` ($\lambda_U$), `learning_rate=1e-6` (padrão da biblioteca, **diferente** da recomendação do paper), `loss_type` ∈ {`kto`, `apo_zero_unpaired`}.

---

## 5. SimPO — Simple Preference Optimization (Meng, Xia & Chen, 2024)

URL: https://arxiv.org/abs/2405.14734 (NeurIPS 2024). Código: https://github.com/princeton-nlp/SimPO

### 5.1 Motivação
A recompensa implícita do DPO (i) exige $\pi_{\text{ref}}$ no treino, com mais memória e computação, e (ii) **não coincide** com a métrica que guia a geração (≈ log-verossimilhança média, sem referência). Só ~50% dos triplos ficam ordenados pela log-prob média após DPO (§2.7.6).

### 5.2 Recompensa normalizada por comprimento, sem referência (Eq. 3–4)

$$
p_\theta(y\mid x)=\frac{1}{\lvert y\rvert}\log\pi_\theta(y\mid x)=\frac{1}{\lvert y\rvert}\sum_{i=1}^{\lvert y\rvert}\log\pi_\theta(y_i\mid x,y_{<i}),\qquad
r_{\text{SimPO}}(x,y)=\frac{\beta}{\lvert y\rvert}\log\pi_\theta(y\mid x).
$$

- **Por que normalizar:** a soma de log-probs penaliza respostas longas (mais termos negativos). Se $y_w$ é mais longa, otimizar a soma força o modelo a "inflar" probabilidades de sequências longas, o que aumenta o risco de degeneração.
- **Ablação "w/o LN":** sem normalização, o modelo vira tendencioso a respostas longas e piores. A correlação de Spearman log-prob média × comprimento vai de 0,34 (SimPO) a **0,82**, e o AlpacaEval 2 LC no Mistral-Base cai de 21,5 para 11,9 (Tabela 5).

### 5.3 Margem-alvo $\gamma$ e a perda (Eq. 5–6)

BT com "vantagem de mando de campo" (*home advantage*) $\gamma>0$: $p(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l)-\gamma)$. Daí:

$$
\boxed{\ \mathcal L_{\text{SimPO}}(\pi_\theta)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\Big(\frac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w\mid x)-\frac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l\mid x)-\gamma\Big)\Big].\ }
$$

- A qualidade da geração **melhora com $\gamma$ até certo ponto e depois piora**.
- A ablação $\gamma=0$ perde 4,7 pontos de LC no Mistral-Base (16,8 contra 21,5).
- O IPO também tem uma "margem-alvo" ($1/(2\tau)$), mas o objetivo completo dele rendeu menos no estudo.

### 5.4 Gradiente (Apêndice F, Eq. 7)

$$
\nabla_\theta\mathcal L_{\text{SimPO}}=-\beta\,\mathbb E\Big[s_\theta\Big(\frac{1}{\lvert y_w\rvert}\nabla\log\pi_\theta(y_w\mid x)-\frac{1}{\lvert y_l\rvert}\nabla\log\pi_\theta(y_l\mid x)\Big)\Big],\quad
s_\theta=\sigma\Big(\frac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l\mid x)-\frac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w\mid x)+\gamma\Big).
$$

Duas diferenças em relação ao DPO:
1. O peso $s_\theta$ não envolve a referência. Ele é alto quando a política dá mais log-prob média a $y_l$ que a $y_w$.
2. Os gradientes são **normalizados por comprimento**. No DPO, sequências longas recebem gradientes maiores e "dominam".

### 5.5 "Esquecimento" sem KL
- O SimPO não tem KL explícito. Os autores atribuem a estabilidade a (1) LR pequeno, (2) dados de preferência diversos e (3) robustez intrínseca do LLM. Medem KL empírico baixo (Fig. 5a: SimPO com β=2 e β=10 contra DPO com β=0,01 e 0,1).
- **Caveat documentado (Apêndice J, Tabela 16):** continuar treinando o Llama-3-8B-Instruct com LR **1e-6** deu AlpacaEval 2 LC 53,7, mas **GSM 78,5 → 57,4 e MMLU 61,7 → 54,9** (esquecimento catastrófico). Com LR 4e-7 a 5e-7, os ganhos de chat foram menores e GSM/MMLU se preservaram.
- No **Gemma-2-9B-it**, o LR quase não afetou isso. GSM 87,4 → 88,0 e MMLU 72,7 → 72,2 no checkpoint lançado (LR 8e-7), com AlpacaEval 2 LC 72,4 e Arena-Hard 59,1.
- O README do repositório nota que o SimPO v0.2 (Llama-3) "luta" com saídas estruturadas como JSON, e atribui isso ao esquecimento mais o LR alto.

### 5.6 Hiperparâmetros

- **Tabela 8 do paper:**

| Configuração | β | γ | LR |
|---|---|---|---|
| Mistral-Base | 2,0 | 1,6 | 3e-7 |
| Mistral-Instruct | 2,5 | 0,3 | 5e-7 |
| Llama-3-Base | 2,0 | 1,0 | 6e-7 |
| Llama-3-Instruct | 2,5 | 1,4 | 1e-6 |

  Grade de busca: $\beta\in[2{,}0;2{,}5]$, $\gamma\in\{0{,}3;0{,}5;1{,}0;1{,}2;1{,}4;1{,}6\}$, LR $\in\{3,5,6\}\times10^{-7}$ e $10^{-6}$, batch 128, 1 época, cosseno com 10% de warmup, comprimento máx. 2.048.
- **README atual (parâmetro `gamma_beta_ratio` = γ/β):**
  - Mistral-Base β=2,0, γ/β=0,8.
  - Mistral-Instruct β=2,5, γ/β=0,1.
  - Llama3-Base β=2,0, γ/β=0,5.
  - Llama3-Instruct β=2,5, γ/β=0,55.
  - Llama3-Instruct v0.2 β=10, γ/β=0,3.
  - **Gemma β=10, γ/β=0,5, LR 8e-7** (config `gemma-2-9b-it-simpo.yaml`: 1 época, cosseno, warmup 0,1, `max_length` 2048).
- **Recomendações do README:**
  - O LR é o hiperparâmetro mais crítico. Busque em 3e-7, 5e-7, 8e-7 e 1e-6. **Um LR menor (5e-7) é mais adequado a domínios de raciocínio como matemática.**
  - β precisa ser **bem maior que no DPO**: 2,0 a 2,5, às vezes 10.
  - Comece com γ/β = 0,5 e faça grid em [0, 1].
  - Mantenha batch total 128.
- **Eficiência (Fig. 5c, Llama-3-Base 8B, 8×H100):** DPO levou **73 min / 77 GB** de pico, SimPO **60 min / 69 GB**.
- **TRL:** SimPO "de verdade" é `trl.experimental.cpo.CPOTrainer` com `loss_type="simpo"` e `cpo_alpha=0.0`. O `simpo_gamma` padrão é 0,5 e é passado como `gamma/beta` dentro do logit. O `DPOTrainer` tem `loss_type="sigmoid_norm"`, que a documentação associa ao SimPO, mas que **mantém a referência**. Na prática é **DPO normalizado por comprimento** (estilo Tülu 3), não SimPO.

### 5.7 Relações úteis
- **O DPO tem uma "margem implícita por instância"** (SimPO, Apêndice I):

$$
\mathcal L_{\text{DPO}}=-\log\sigma\big(\beta\log\pi_\theta(y_w)-\beta\log\pi_\theta(y_l)-\gamma_{\text{ref}}\big),\qquad \gamma_{\text{ref}}=\beta\log\pi_{\text{ref}}(y_w)-\beta\log\pi_{\text{ref}}(y_l).
$$

  Ou seja, o DPO "é" uma perda sem referência com margem que varia por par. O SimPO troca isso por uma margem **constante** $\gamma$ e pela média por token. Na Tabela 15 do SimPO, aplicar só LN ao DPO ajuda no Mistral-Base mas não no Mistral-Instruct, e aplicar só γ ao DPO não ajuda.
- **DPO normalizado por comprimento (Tülu 3, Eq. 6):** $-\log\sigma\big(\frac{\beta}{\lvert y_c\rvert}\log\frac{\pi_\theta(y_c)}{\pi_{\text{ref}}(y_c)}-\frac{\beta}{\lvert y_r\rvert}\log\frac{\pi_\theta(y_r)}{\pi_{\text{ref}}(y_r)}\big)$. Foi o melhor no Tülu 3 (β=5), comparado com SimPO e DPO.

---

## 6. Outras variantes (resumo com fórmulas)

### 6.1 ORPO — *Odds Ratio Preference Optimization* (Hong, Lee & Thorne, 2024)
URL: https://arxiv.org/abs/2403.07691. SFT e preferência **num único estágio**, **sem referência**.

- **Definições (Eq. 3–5):** $\log P_\theta(y\mid x)=\frac1m\sum_t\log P_\theta(y_t\mid x,y_{<t})$, **média por token**, de modo que $P_\theta$ é a média geométrica das probabilidades dos tokens. $\text{odds}_\theta(y\mid x)=\frac{P_\theta(y\mid x)}{1-P_\theta(y\mid x)}$ e $\text{OR}_\theta=\frac{\text{odds}_\theta(y_w\mid x)}{\text{odds}_\theta(y_l\mid x)}$.
- **Perda (Eq. 6–7):**
$$
\mathcal L_{\text{ORPO}}=\mathbb E_{(x,y_w,y_l)}\big[\mathcal L_{\text{SFT}}+\lambda\,\mathcal L_{\text{OR}}\big],\qquad \mathcal L_{\text{OR}}=-\log\sigma\Big(\log\frac{\text{odds}_\theta(y_w\mid x)}{\text{odds}_\theta(y_l\mid x)}\Big).
$$
- **Gradiente (Eq. 8–10):** $\nabla\mathcal L_{\text{OR}}=\delta(d)\,h(d)$, com $\delta(d)=\big[1+\frac{\text{odds}(y_w)}{\text{odds}(y_l)}\big]^{-1}$ (penalidade que some quando o par já está certo) e $h(d)=\frac{\nabla\log P_\theta(y_w)}{1-P_\theta(y_w)}-\frac{\nabla\log P_\theta(y_l)}{1-P_\theta(y_l)}$. O denominador $1-P$ **amplifica** o gradiente quando $P$ é baixo.
  - **[conferência minha]** $\frac{d}{dz}[-\log\sigma(z)]=-(1+e^{z})^{-1}$ e $\nabla\log\text{odds}=\nabla\log P/(1-P)$. As duas expressões batem.
- **Por que odds e não razão de probabilidades (§7.1):** a log-razão de probabilidades (escalada por β) leva a discriminação **mais extrema** contra a preterida. Num SFT conjunto, um contraste mais suave preserva a adaptação ao domínio ("forte adaptação ao escolhido, penalidade fraca ao rejeitado").
- **Custo:** sem modelo de referência, o ORPO faz **metade dos forwards** por batch em relação a DPO/RLHF.
- **λ:** ablação {0,1; 0,5; 1,0}. λ maior discrimina mais os rejeitados. Com λ=0,1, as log-probs de escolhidos e rejeitados ficam próximas e as dos rejeitados não caem. No MT-Bench, λ=1,0 é pior em extração, matemática e raciocínio e melhor em categorias abertas.
- **Treino reportado:** LR máx. 8e-6 por 10 épocas (OPT, Phi-2, Llama-2), com seleção por menor perda de validação.
- **TRL:** `trl.experimental.orpo`. `beta` = **λ do paper** (no código original, `alpha`), padrão 0,1. A perda é `nll − mean(beta·logsigmoid(log_odds))`, com log-probs médias (`average_log_prob=True`) e `log1mexp` para estabilidade.

### 6.2 CPO — *Contrastive Preference Optimization* (Xu et al., 2024)
URL: https://arxiv.org/abs/2401.08417 (tradução automática, ALMA-R).
- **Aproximação:** com $\pi_{\text{ref}}$ **uniforme** $U$, os termos da referência se cancelam, $\mathcal L(\pi_\theta;U)=-\mathbb E[\log\sigma(\beta\log\pi_\theta(y_w\mid x)-\beta\log\pi_\theta(y_l\mid x))]$ (Eq. 3).
- **Teorema 1:** se $\pi_{\text{ref}}=\pi_w$ (a política "ideal" dos preferidos), a perda DPO $\mathcal L(\pi_\theta;\pi_w)+C$ é limitada superiormente por $\mathcal L(\pi_\theta;U)$.
- **Regularizador BC:** $\mathbb E[\mathrm{KL}(\pi_w\|\pi_\theta)]<\epsilon$. Via Lagrangiano com $\lambda=1$, isso vira um termo NLL nos preferidos:
$$
\mathcal L_{\text{CPO}}=\underbrace{\mathcal L(\pi_\theta,U)}_{\mathcal L_{\text{prefer}}}\ \underbrace{-\ \mathbb E_{(x,y_w)}[\log\pi_\theta(y_w\mid x)]}_{\mathcal L_{\text{NLL}}}.
$$
- **Motivação:** referências "gold" humanas de tradução às vezes são piores que saídas de modelos fortes. O SFT puro copia esse teto.
- **TRL:** `trl.experimental.cpo.CPOTrainer`, `cpo_alpha=1.0` (peso do BC), `beta=0.1`, `loss_type` ∈ {`sigmoid`, `hinge`, `ipo`, `simpo`, `alphapo`}.
- **Advertência:** sem normalização por comprimento, a parte de preferência usa a log-prob **somada**. Respostas preferidas mais longas são penalizadas pela soma (§12.8). No SimPO, o CPO gera respostas ~50% mais longas que o SimPO no Arena-Hard (Tabela 10 do SimPO), o que favorece o CPO nesse benchmark sem penalidade de comprimento.

### 6.3 R-DPO — DPO regularizado por comprimento (Park, Rafailov, Ermon & Finn, 2024)
URL: https://arxiv.org/abs/2403.19159.
- **Objetivo:** RL com penalidade de comprimento, $\max\ \mathbb E[r(x,y)-\alpha\lvert y\rvert]-\beta\,\mathbb D_{\text{KL}}$ (Eq. 6).
- **Ótimo:** $\pi^*\propto\pi_{\text{ref}}\,e^{(r-\alpha\lvert y\rvert)/\beta}$ (Eq. 7), logo $r=\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z+\alpha\lvert y\rvert$ (Eq. 8).
- **Perda (Eq. 9):**
$$
\mathcal L_{\text{R-DPO}}=-\mathbb E\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}+\big(\alpha\lvert y_w\rvert-\alpha\lvert y_l\rvert\big)\Big)\Big].
$$
- **Leitura [interpretação]:** quando a preferida é **mais longa**, o termo $\alpha(\lvert y_w\rvert-\lvert y_l\rvert)>0$ já "explica" parte da preferência. A recompensa implícita (penalizada por comprimento) precisa de menos margem, e há menos pressão para inflar o comprimento. Quando a preferida é mais curta, a pressão aumenta.
- **Valores:**
  - $\alpha=0{,}01$ (HH) e $\alpha=0{,}05$ (TL;DR) na Fig. 2.
  - Sweeps $\alpha\in\{0;0{,}005;0{,}01\}$ (HH) e $\{0;0{,}2;0{,}5\}$ (TL;DR), com $\beta\in\{0{,}05;0{,}1;0{,}5\}$.
  - $\alpha=0{,}1$ na Fig. 5 (HH, β=0,1).
  - $\alpha=0{,}05$ com β=0,1 em UltraFeedback/MT-Bench (Apêndice).
  - Ver §14 sobre a inconsistência.
- **Ganhos:** até **20%** de melhora em *win rate* controlando comprimento.

### 6.4 β-DPO — β dinâmico (Wu et al., NeurIPS 2024)
URL: https://arxiv.org/abs/2407.08639.
- **Discrepância individual:** $M_i=r(y_w^{(i)})-r(y_l^{(i)})$, com recompensa implícita do DPO calculada com $\beta_0$.
- **Por instância (Eq. 5):** $\beta_i=[1+\alpha(M_i-M_0)]\beta_0$. Pode desestabilizar.
- **Por lote (Eq. 6):** $\beta_{\text{batch}}=[1+\alpha(\mathbb E_{i\in\text{batch}}[M_i]-M_0)]\beta_0$, com $M_0\leftarrow mM_0+(1-m)\,\mathbb E_{\text{batch}}[M_i]$ (Eq. 7).
- **Filtro guiado por β (Eq. 8–9):** amostrar $\lvert\text{batch}\rvert\times\rho$ exemplos sem reposição com peso gaussiano $p(M_i)\propto\exp(-(M_i-M_0)^2/2\sigma^2)$. $\sigma$ também é atualizado por média móvel. $\rho=0{,}8$ por padrão.
- **Princípio:** pares "fáceis" (gap alto) pedem β **maior** (cautela contra ruído). Pares próximos e de alta qualidade pedem β **menor** (atualização assertiva).
- **Valores:** $\beta_0=0{,}1$. O ótimo de $\alpha$ fica tipicamente entre 0,6 e 1,0. O $\rho$ ideal variou com o tamanho do modelo (0,3 no Pythia-2.8B, 0,5 no 410M).

### 6.5 DPO-Positive / DPOP (Pal et al., 2024, "Smaug")
URL: https://arxiv.org/abs/2402.13228.
$$
\mathcal L_{\text{DPOP}}=-\mathbb E\Big[\log\sigma\Big(\beta\Big(\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}-\lambda\max\Big(0,\log\frac{\pi_{\text{ref}}(y_w)}{\pi_\theta(y_w)}\Big)\Big)\Big)\Big].
$$
- A penalidade é zero enquanto $\pi_\theta(y_w)\ge\pi_{\text{ref}}(y_w)$ e cresce quando a preferida cai abaixo da referência. É um contra-ataque direto ao deslocamento de verossimilhança.
- **Valores do paper:** β=0,3 e λ=50 (ablação λ ∈ {5, 50, 500}), LR 5e-5, AdamW, 1.000 passos.
- **Resultados:** em MetaMath (baixa distância de edição), DPO e SLiC "falham catastroficamente", IPO não melhora e só o DPOP melhora.

### 6.6 Online DPO, TR-DPO, WPO, RPO, mistura de perdas
- **Online DPO / OAIF:** ver §2.6.
- **TR-DPO:** ver §2.5.
- **WPO (arXiv 2406.11827):** repondera pares off-policy pela probabilidade sob a política atual, para simular dados on-policy. No TRL: `use_weighting=True`.
- **DPO + NLL (RPO/IRPO):** no TRL atual, combina-se com `loss_type=["sigmoid","sft"]` e `loss_weights`. O mesmo mecanismo implementa MPO, por exemplo `["sigmoid","bco_pair","sft"]` com pesos `[0.8, 0.2, 1.0]`.
- **Lista de `loss_type` do `DPOTrainer` (TRL main):** `sigmoid` (padrão), `hinge` (RSO/SLiC, `beta` = recíproco da margem), `ipo`, `exo_pair`, `nca_pair`, `robust` (`label_smoothing` = prob. de inversão do rótulo, em [0; 0,5)), `bco_pair`, `sppo_hard`, `aot`/`aot_unpaired`, `apo_zero`/`apo_down`, `discopop`, `sft`, `sigmoid_norm`.
- **Outras opções:** `f_divergence_type` ∈ {`reverse_kl`, `forward_kl`, `js_divergence`, `alpha_divergence`} e `ld_alpha` (LD-DPO).

---

## 7. Tabela comparativa e guia de decisão

Legenda de custo: "F" = forward por par ou exemplo; só a política tem backward. Os valores são "como publicados".

| Método | Precisa de $\pi_{\text{ref}}$? | Tipo de dado | Custo relativo por passo | Hiperparâmetros-chave (valores publicados) | Estabilidade / riscos | Quando usar |
|---|---|---|---|---|---|---|
| **DPO** | Sim (pode pré-computar ou usar LoRA com adaptador desligado) | Pares $y_w\succ y_l$ | 2F política + 2F referência | β 0,1 (0,01–0,5); LR 5e-7–1e-6; 1 época; batch 64–128 | Estável e simples. Sobreajusta prefs determinísticas, desloca verossimilhança, explora comprimento, corcova de sobre-otimização | Padrão quando há pares de boa qualidade e SFT prévio |
| **IPO** | Sim | Pares | Igual ao DPO | τ (TRL `beta`); grade 0,01–1; alvo $1/(2\tau)$ (cuidado com a normalização por token) | Ancora na referência mesmo com rótulos determinísticos; menos sobre-otimização (DAAs 6,9B). Pode sub-ajustar | Dados pequenos, rótulo único por par, ruído |
| **KTO** | Sim, mais o KL estimado | **Binário** desejável/indesejável; aceita desbalanceamento | 2F política + 2F referência por exemplo (casado + desencontrado) | β 0,1 (0,01–0,5); $\lambda_Dn_D/\lambda_Un_U\in[1;1{,}5]$; LR ~5e-6 (2–10× DPO); microbatch ≥ 2 (8–128) | Robusto a ruído e intransitividade; ignora exemplos extremos. Risco de sub-ajuste | Feedback 👍/👎, logs de produção, classes desbalanceadas |
| **SimPO** | **Não** | Pares | 2F política | β 2–2,5 (até 10); γ/β ≈ 0,5 (0–1); LR 3e-7–1e-6; batch 128 | Sem KL explícito: esquecimento com LR alto (Llama-3-Instruct). Bom contra viés de comprimento | Memória apertada; alinhar treino com a métrica de geração |
| **ORPO** | **Não** | Pares (o NLL usa o escolhido) | 2F política | λ 0,1 (0,1–1,0); LR 8e-6 em modelos ≤ 7B no paper | Faz SFT e preferência juntos; λ alto prejudica tarefas determinísticas | Sem SFT prévio, quando se quer um estágio único |
| **CPO** | **Não** | Pares + NLL | 2F política | β 0,1; $\lambda$ (BC) = 1 | Soma de log-probs é sensível ao comprimento | MT e tarefas em que o "gold" é imperfeito |
| **R-DPO** | Sim | Pares + comprimentos | Igual ao DPO | α 0,005–0,05 (até 0,5 no TL;DR) | Controla verbosidade; não resolve sobre-otimização | Dataset com viés de comprimento |
| **β-DPO** | Sim | Pares | DPO + estatística de lote | β0 0,1; α 0,6–1,0; ρ 0,8 | Mais robusto a qualidade heterogênea | Dados de qualidade misturada |
| **DPOP** | Sim | Pares | Igual ao DPO | β 0,3; λ 50 | Impede $\pi(y_w)$ de cair | Pares de baixa distância de edição (matemática) |
| **Online DPO (OAIF)** | Sim | Pares gerados on-policy e rotulados por juiz | DPO + geração + juiz | Igual ao DPO, mais temperatura e tokens novos | Elimina o desvio off-policy; custo de amostragem | Quando há um juiz ou RM e inferência rápida (vLLM) |
| **SLiC-HF** | Não (regularização por CE; há opção KL) | Pares | 2F política (+CE) | δ = 1,0; λ; LR 1e-5 (T5) | Não é HALO; hinge sem saturação suave | Alternativa histórica; base do `hinge` |
| **AWR / filtered BC (RAFT, ReST)** | Não necessariamente | Recompensa escalar ou verificável | K gerações + RM ou verificador + SFT ponderado | β (temperatura), $\omega_{\max}$, K, limiar τ | Muito estável (é SFT); limitado pela qualidade das amostras | Recompensas verificáveis (matemática, código), destilação de best-of-K |

**Guia rápido [interpretação, a partir das recomendações dos próprios papers]:**
1. Tem **pares** e um SFT? Comece com **DPO**. Se os rótulos forem determinísticos ou poucos, ou houver sobre-otimização, **IPO**. Se a verbosidade incomodar, use **R-DPO** ou **DPO normalizado por comprimento**.
2. Tem **👍/👎** (ou forte desbalanceamento)? Use **KTO**.
3. Memória curta e sem referência? **SimPO** (com LR conservador) ou **ORPO** (se ainda não fez SFT).
4. A verossimilhança da preferida cai e o comportamento piora? Filtre pares por **CHES**, use **DPOP** ou adicione **NLL**.
5. Pode gerar e julgar durante o treino? **Online/iterativo**. É o maior salto estrutural (OAIF ~66% contra offline).

---

## 8. Custo de memória no contexto 2× DGX Spark (estimativa por fórmula, NÃO medida)

- **Contabilidade do ZeRO** (Rajbhandari et al., arXiv 1910.02054): Adam em precisão mista guarda pesos fp16/bf16 (2 B/param) + gradientes (2 B/param) + estados do otimizador (cópia fp32, momento e variância, 12 B/param). Total ≈ **16 bytes/param**, sem ativações.
- **Por método:**
  - **DPO, IPO e KTO com fine-tuning completo:** +2 B/param para a referência em bf16 (só inferência), ou zero se os logprobs forem pré-computados.
  - **SimPO, ORPO e CPO:** sem referência.
  - **Com LoRA:** a base congelada em bf16 (2 B/param) também serve de referência (adaptadores desligados). Gradientes e otimizador só existem para os adaptadores.
- **Exemplo ilustrativo:** para um modelo de 8B parâmetros, o fine-tuning completo exige ~128 GB só de estados, o que já é toda a memória unificada de um Spark (128 GB), antes das ativações. Com LoRA, a base em bf16 ocupa ~16 GB.
- **[interpretação]** Os modelos MoE de fronteira do Módulo 1 estão fora de alcance para pós-treino completo em 2 Sparks. Só o checkpoint GLM-5.3-Flash quantizado (experts EXL3 4 bpw) já ocupa ~164 GiB, segundo `docs/pesquisa/glm-mods.md` §1.2. O laboratório prático do Módulo 2 deve usar modelos pequenos (≤ 8B) com LoRA/QLoRA, ou pré-computar a referência.

---

## 9. AWR e a família de "regressão ponderada" (o caminho por regressão)

### 9.1 RWR — *Reward-Weighted Regression* (Peters & Schaal, 2007)
- **Atualização** (como apresentada no AWR, Eq. 2): $\pi_{k+1}=\arg\max_\pi\ \mathbb E_{s\sim d_{\pi_k},\,a\sim\pi_k}\big[\log\pi(a\mid s)\,\exp(\tfrac1\beta R_{s,a})\big]$, com $R_{s,a}=\sum_t\gamma^tr_t$ (retorno).
- **Leitura:** é um **SFT ponderado** nas próprias amostras, com peso exponencial do retorno e temperatura $\beta$. Vem de um arcabouço EM (há também a derivação dual de busca de política com restrição).

### 9.2 AWR — *Advantage-Weighted Regression* (Peng, Kumar, Zhang & Levine, 2019)
URL: https://arxiv.org/abs/1910.00177

- **Algoritmo 1:** coletar trajetórias com $\pi_k$ para um *replay buffer* $\mathcal D$ (FIFO), depois fazer duas regressões supervisionadas:
  1. Valor: $V_k^{\mathcal D}\leftarrow\arg\min_V\mathbb E_{s,a\sim\mathcal D}\big[\|\mathcal R^{\mathcal D}_{s,a}-V(s)\|^2\big]$.
  2. Política: $\pi_{k+1}\leftarrow\arg\max_\pi\mathbb E_{s,a\sim\mathcal D}\big[\log\pi(a\mid s)\exp\big(\tfrac1\beta(\mathcal R^{\mathcal D}_{s,a}-V^{\mathcal D}_k(s))\big)\big]$.

- **Derivação (§3.1):** maximiza a **melhora esperada** $\eta(\pi)=J(\pi)-J(\mu)=\mathbb E_{s\sim d_\pi}\mathbb E_{a\sim\pi}[A^\mu(s,a)]$ (Eq. 3), aproximada com a distribuição de estados de $\mu$ (Eq. 4) e uma restrição de KL:

$$
\arg\max_\pi\int_s d_\mu(s)\int_a\pi(a\mid s)\big[\mathcal R^\mu_{s,a}-V^\mu(s)\big]\,da\,ds\quad\text{s.a.}\quad\int_s d_\mu(s)\,\mathbb D_{\text{KL}}(\pi(\cdot\mid s)\|\mu(\cdot\mid s))\,ds\le\epsilon.
$$

  - **Lagrangiano (Eq. 7) → ótimo não paramétrico (Eq. 8):**
$$
\boxed{\ \pi^*(a\mid s)=\frac{1}{Z(s)}\,\mu(a\mid s)\exp\Big(\frac1\beta\big(\mathcal R^\mu_{s,a}-V^\mu(s)\big)\Big)\ }
$$
  - **Projeção na família paramétrica (Eq. 9–10):** $\arg\min_\pi\mathbb E_s[\mathbb D_{\text{KL}}(\pi^*\|\pi)]=\arg\max_\pi\mathbb E_{s\sim d_\mu}\mathbb E_{a\sim\mu}\big[\log\pi(a\mid s)\exp(\tfrac1\beta(\mathcal R-V))\big]$. É uma **máxima verossimilhança ponderada** nas amostras de $\mu$.
- **Diferença para o RWR:** a **linha de base** $V(s)$ (vantagem em vez de retorno). Isso vem de maximizar a melhora e não o retorno, e os autores mostram que é crucial empiricamente.
  - **[derivação minha]** Para um mesmo estado, normalizar os pesos torna o $e^{-V(s)/\beta}$ irrelevante: os pesos *normalizados* de AWR e RWR coincidem (§12.12). A linha de base importa **entre estados ou prompts diferentes**, porque iguala a escala dos pesos de prompts "fáceis" e "difíceis" num lote que mistura vários.
- **Detalhes práticos:**
  - Retornos com TD(λ) usando $V_{k-1}$ (λ=0,95).
  - **β fixo** (não adaptado como em REPS/MPO), β=0,05 em todos os experimentos.
  - **Clipping de pesos** $\hat\omega=\min(\omega,\omega_{\max})$ com $\omega_{\max}=20$, contra explosões.
  - Buffer de 50k amostras, ~2.000 amostras por iteração, minibatch 256, 200 passos de valor e 1.000 de política por iteração. O off-policy com *replay* é tratado como mistura de políticas passadas (§3.2).

### 9.3 Relação com o ótimo KL-regularizado dos LLMs
- **A mesma forma:** a Eq. 8 do AWR é a §1.4 com $\mu\leftrightarrow\pi_{\text{ref}}$ e $A\leftrightarrow r$. Para LLMs (bandit contextual: $s=x$, $a=y$), $\pi^*(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$, e subtrair $V(x)$ só muda $Z(x)$.
- O KTO cita explicitamente que essa forma fechada vem de Peng et al. 2019 e de Peters & Schaal 2007. Nos créditos do DPO, um autor "iniciou a discussão de usar métodos de regressão ponderada como alternativa ao PPO" e "escreveu a análise inicial ligando DPO a regressão ponderada e unlikelihood".
- **[interpretação/síntese]** Dois jeitos de explorar o mesmo ótimo "inclinado":
  - **AWR / SFT ponderado:** amostra respostas, pesa por $\exp(A/\beta)$ e faz máxima verossimilhança (projeção $\mathbb D_{\text{KL}}(\pi^*\|\pi_\theta)$, que "cobre a massa"). Precisa de **recompensa escalar** e de amostras.
  - **DPO:** não amostra. Reparametriza a recompensa pela política e ajusta o BT nos **pares**. Precisa só de **preferências**.
- **AWAC** (Nair, Gupta, Dalal & Levine, arXiv 2006.09359, Eq. 13) faz o mesmo com vantagem de um **crítico Q**: $\theta_{k+1}=\arg\max_\theta\mathbb E_{s,a\sim\beta}\big[\log\pi_\theta(a\mid s)\exp(\tfrac1\lambda A^{\pi_k}(s,a))\big]$. A solução fechada é $\pi^*\propto\pi_\beta\exp(A/\lambda)$, **sem** modelar a política comportamental. Usa λ=0,3 (manipulação) e λ=1,0 (MuJoCo).
  - **Nota:** o texto principal do AWAC diz "forward KL" para a projeção $\mathbb D_{\text{KL}}(\pi^*\|\pi_\theta)$ e o apêndice diz "reverse KL" para a mesma projeção. Ver §14.

### 9.4 Como a ponderação estilo AWR aparece no pós-treino de LLMs

O espectro do "SFT ponderado" $\max_\theta\sum_i w_i\log\pi_\theta(y_i\mid x)$ sobre amostras $y_i$ do próprio modelo:

| Esquema | Peso $w_i$ | Exemplo |
|---|---|---|
| SFT puro / "Preferred-FT" | 1 nos escolhidos | baseline do DPO; inicialização de $\pi_{\text{ref}}$ |
| **AWR** | $\exp(A_i/\beta)$ (clipado) | AWR/AWAC; **A-LoL** (arXiv 2305.14718) treina só em dados de vantagem **positiva**, com a sequência inteira como uma ação |
| **RWR / ReST-EM** | $\propto r_i$ (com $f$ identidade e recompensas ≥ 0) ou $\exp(r_i/\beta)$ | ReST-EM: E-step $q\propto p(O{=}1\mid x,y)\,p_{\theta_t}(y\mid x)$; M-step = NLL ponderada (Eq. 3 do ReST-EM) |
| **Filtro por limiar** | $\mathbb 1[R(x,y)>\tau]$ | **ReST** (arXiv 2308.08998): Grow (gerar) + Improve (filtrar com $\tau_1<\tau_2<\dots$ e fazer fine-tuning), com $\tau_1>V^{\pi_\theta}$ e $J(\theta)=\mathbb E[F(x,y;\tau)\mathcal L(x,y;\theta)]$ (Eq. 2) |
| **Binário (verificável)** | $\mathbb 1[\text{correto}]$ | **ReST-EM** (arXiv 2312.06585, MATH e APPS com PaLM 2), STaR (arXiv 2203.14465), RFT (arXiv 2308.01825) |
| **Argmax (best-of-K)** | $\mathbb 1[i=\arg\max_j r_j]$ | **RAFT** (arXiv 2304.06767): gerar K respostas com temperatura λ, pegar a de maior recompensa e fazer fine-tuning. Llama 2 (arXiv 2307.09288) usa *rejection sampling fine-tuning* análogo |

- **Unificação [derivação minha, a partir das fórmulas acima]:** com pesos $\exp(A/\beta)$ normalizados, $\beta\to0$ vira o **argmax** (RAFT/best-of-K), $\beta\to\infty$ vira **SFT uniforme**, e valores intermediários interpolam (números em §12.12). No ReST-EM, escolher $p(O{=}1\mid x,y)\propto\exp(r/\beta)$ dá $q\propto\pi_{\theta_t}\exp(r/\beta)$: o mesmo ótimo da §1.4 com a **política atual como referência**.
- **RAFT, sobre K:** a recompensa média do best-of-K cresce com K, mas o ganho é limitado por algo proporcional a $\sqrt{\log K}$ (recompensa limitada por $B$, desigualdade de concentração). O retorno marginal cai rápido, o que motiva iterar. Os experimentos usam K ∈ {8, 16, 32} com λ=0,85.
- **ReST, detalhes:**
  - O BC (NLL) foi a melhor perda do Improve, contra OAC, BVMPO e GOLD, em tradução automática.
  - Cada nova política é ajustada a partir da anterior com LR menor (subconjuntos cada vez menores).
  - Filtro por limiar pode favorecer resultados de alta variância em ambientes estocásticos (Brandfonbrener et al.). Os autores tratam tradução como determinística.
- **ReST-EM:** a separação "coleta (E) / otimização (M)" é o que facilita escalar para modelos grandes. **O Generate precisa de inferência rápida.** É de novo o vLLM do Módulo 1.

---

## 10. Aprendizado contrastivo e parentes

### 10.1 InfoNCE (van den Oord, Li & Vinyals, 2018; CPC)
URL: https://arxiv.org/abs/1807.03748
- **Perda (Eq. 4):** dado um conjunto $X=\{x_1..x_N\}$ com **1 positivo** de $p(x_{t+k}\mid c_t)$ e $N-1$ negativos de uma proposta $p(x_{t+k})$:
$$
\mathcal L_N=-\mathbb E_X\Big[\log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}\Big].
$$
  É a entropia cruzada categórica de "achar o positivo".
- **Ótimo:** $f_k\propto p(x_{t+k}\mid c_t)/p(x_{t+k})$ (razão de densidades), independentemente de $N$.
- **Limite de informação mútua:** $I(x_{t+k};c_t)\ge\log N-\mathcal L_N$, mais apertado com $N$ maior.
- **Intuição para o capítulo:** "um contra muitos". Pontue alto o certo, baixo os errados, e normalize por softmax.
- **Ponte com o DPO:** com **1 negativo** e escore $s(y)=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$, $-\log\frac{e^{s(y_w)}}{e^{s(y_w)}+e^{s(y_l)}}=-\log\sigma(s(y_w)-s(y_l))$. Isso é **exatamente o DPO**. Com rankings, o Plackett–Luce do DPO (§2.1) é uma soma de InfoNCEs aninhados.

### 10.2 SLiC (Zhao et al., 2022) e SLiC-HF (Zhao et al., 2023)
URLs: https://arxiv.org/abs/2210.00045 e https://arxiv.org/abs/2305.10425
- **SLiC** é um terceiro estágio após o fine-tuning ("calibrar a verossimilhança de sequência"): decodificar $m$ candidatos $\{\hat y\}_m$ do modelo e treinar com $\mathcal L(\theta)=\sum_b\mathcal L^{\text{cal}}(\theta,s;x,\bar y,\{\hat y\}_m)+\lambda\mathcal L^{\text{reg}}(\theta,\theta_{ft};x,\bar y)$.
  - $s$ é a similaridade do candidato com o alvo, um escore estilo BERTScore sobre estados ocultos do próprio decoder.
- **Perdas de calibração (Eq. 1):**
  - *rank:* $\max(0,\beta-\log P_\theta(\hat y^+\mid x)+\log P_\theta(\hat y^-\mid x))$
  - *margin:* $\max(0,\beta(s(\hat y^+)-s(\hat y^-))-\log P_\theta(\hat y^+)+\log P_\theta(\hat y^-))$
  - *list rank:* $\sum_{i<j}\max(0,\beta\lvert i-j\rvert-\log P_\theta(\hat y_i)+\log P_\theta(\hat y_j))$ (a perda do BRIO)
  - *expected reward:* $\sum_i\big[-s(\hat y_i)\cdot\frac{P_\theta(\hat y_i)}{\sum_iP_\theta(\hat y_i)}\big]$
  - **Rank foi a melhor**, seguida de margin, list rank e expected reward.
- **Regularização (Eq. 2):** CE nos tokens do alvo, $\sum_t-\log P_\theta(\bar y_t\mid\bar y_{<t},x)$, ou KL por token $\sum_tP_\theta\log\frac{P_\theta}{P_{\theta_{ft}}}$ no alvo observado.
- **SLiC-HF (Eq. 4):**
$$
\mathcal L(\theta)=\max\big(0,\ \delta-\log P_\theta(y^+\mid x)+\log P_\theta(y^-\mid x)\big)-\lambda\log P_\theta(y_{\text{ref}}\mid x).
$$
  - Hinge de **margem $\delta$** (calibração de *rank*) mais CE no alvo $y_{\text{ref}}$ (o SFT, ou o melhor candidato segundo um ranker ou RM).
  - A CE substitui o KL "sem precisar de uma cópia extra dos pesos SFT". O KL foi testado no SLiC original e teve desempenho similar.
  - **Variantes:** *sample-rank* (candidatos do SFT ranqueados por um RM pontual ou por um **ranker par-a-par** T5-XXL, que foi mais preciso) e *direct* (usa direto os pares humanos, mais simples, mas off-policy).
  - **Valores:** LR 1e-5, margem 1,0, 8 amostras com temperatura 0,7 e top-k 40 (T5, TL;DR).
- **Relação com o DPO:**
  - O SLiC usa log-prob **crua** (sem referência). A "âncora" vem da CE.
  - Trocar por log-razões dá a perda `hinge` do RSO/TRL: $\max(0,1-\beta\Delta h)$, com $\beta$ = recíproco da margem.
  - O hinge **zera o gradiente** além da margem, enquanto a logística do DPO só decai.
  - Pelo KTO, SLiC **não é HALO**.

### 10.3 CPL — *Contrastive Preference Learning* (Hejna et al., ICLR 2024)
URL: https://arxiv.org/abs/2310.13639
- **Premissa:** humanos preferem segmentos segundo o **arrependimento** (*regret*) em relação à política ótima, não segundo o retorno parcial (Knox et al.). Modelo de preferência por vantagem ótima (Eq. 2):
$$
P_{A^*}[\sigma^+\succ\sigma^-]=\frac{\exp\sum_{\sigma^+}\gamma^tA^*(s_t^+,a_t^+)}{\exp\sum_{\sigma^+}\gamma^tA^*(s_t^+,a_t^+)+\exp\sum_{\sigma^-}\gamma^tA^*(s_t^-,a_t^-)}.
$$
- **Truque (RL de máxima entropia, Ziebart):** $\pi^*(a\mid s)=e^{A^*_r(s,a)/\alpha}$, logo $A^*_r(s,a)=\alpha\log\pi^*(a\mid s)$ (Eq. 3). Uma bijeção vantagem ↔ política, que **elimina aprender vantagem ou recompensa e elimina o RL**.
- **Perda (Eq. 5):**
$$
\mathcal L_{\text{CPL}}=\mathbb E_{(\sigma^+,\sigma^-)}\Big[-\log\frac{\exp\sum_{\sigma^+}\gamma^t\alpha\log\pi_\theta(a_t^+\mid s_t^+)}{\exp\sum_{\sigma^+}\gamma^t\alpha\log\pi_\theta(a_t^+\mid s_t^+)+\exp\sum_{\sigma^-}\gamma^t\alpha\log\pi_\theta(a_t^-\mid s_t^-)}\Big].
$$
  - É convexa e totalmente off-policy, funciona em MDPs arbitrários (segmentos sequenciais).
  - **Teorema 1:** com dados ilimitados de um modelo de arrependimento ruidoso-racional, recupera $\pi^*$.
  - **Proposição 1:** sempre aprende uma vantagem **consistente**, isto é, a política ótima de *alguma* recompensa.
- **Regularizador conservador "bias" (Eq. 6):** com dados finitos, há infinitos minimizadores (núcleo da matriz de comparações) que podem pôr massa fora do dataset. A correção pondera os negativos por $\lambda\in(0,1)$:
$$
\mathcal L_{\text{CPL}(\lambda)}=\mathbb E\Big[-\log\frac{\exp\sum_{\sigma^+}\gamma^t\alpha\log\pi_\theta(a^+_t\mid s^+_t)}{\exp\sum_{\sigma^+}\gamma^t\alpha\log\pi_\theta(a^+_t\mid s^+_t)+\exp\lambda\sum_{\sigma^-}\gamma^t\alpha\log\pi_\theta(a^-_t\mid s^-_t)}\Big].
$$
  Subir a log-prob de positivos e negativos **pela mesma quantidade** passa a reduzir a perda, o que favorece ações vistas no dataset (§12.15).
- **Relação com contraste e com o DPO:**
  - A Eq. 5 é uma instância de NCE. Com rankings e Plackett–Luce, o CPL **recupera o InfoNCE** (negativos = segmentos ranqueados abaixo).
  - **O DPO é caso especial do CPL** (Apêndice A.6): bandit contextual (episódio de 1 passo, mesmo estado inicial), em que $A^*=r_E(s,a)-V^*(s)$ e o $V^*$ cancela.
  - **[derivação minha]** No MDP de tokens de um LLM ($s_t=(x,y_{<t})$, $a_t=y_t$, $\gamma=1$), a soma $\sum_t\alpha\log\pi_\theta(y_t\mid x,y_{<t})=\alpha\log\pi_\theta(y\mid x)$. O CPL com segmentos = respostas completas vira **exatamente** $-\log\sigma(\alpha\log\pi_\theta(y_w\mid x)-\alpha\log\pi_\theta(y_l\mid x))$, a perda $\mathcal L(\pi_\theta;U)$ do CPO, isto é, o DPO com **referência uniforme**. Faz sentido: máxima entropia = regularização KL contra a uniforme.

### 10.4 Treinamento por *unlikelihood* (Welleck et al., 2019)
URL: https://arxiv.org/abs/1908.04319
- **Perda de unlikelihood num passo (Eq. 3),** com conjunto de candidatos negativos $\mathcal C^t$:
$$
\mathcal L^t_{\text{UL}}(p_\theta(\cdot\mid x_{<t}),\mathcal C^t)=-\sum_{c\in\mathcal C^t}\log\big(1-p_\theta(c\mid x_{<t})\big).
$$
  Ela diminui quando $p_\theta(c)$ diminui.
- **Objetivo por token (Eq. 4):** $\mathcal L^t_{\text{UL-token}}=-\alpha\sum_{c\in\mathcal C^t}\log(1-p_\theta(c\mid x_{<t}))-\log p_\theta(x_t\mid x_{<t})$.
  - Candidatos usados: os tokens do contexto anterior, $\mathcal C^t_{\text{prev-context}}=\{x_1..x_{t-1}\}\setminus\{x_t\}$ (Eq. 5). Isso combate repetição e tokens frequentes demais.
- **Gradiente (Eq. 6):** em relação aos logits $a$, com um negativo, $\nabla\mathcal L_a=x^*-m\odot p$, onde $m_i=1-\alpha\frac{p_{neg}}{1-p_{neg}}$ para $i\ne i_{neg}$ e $m_{i_{neg}}=1+\alpha$.
- **Nível de sequência (Eq. 7–8):** penaliza tokens que fazem parte de **n-gramas repetidos** em continuações **decodificadas** pelo modelo, o que ataca a discrepância treino × geração.
- **Números motivadores:**
  - Com decodificação gulosa, 43% de n-gramas repetidos contra 0,5% em texto humano.
  - O próximo token previsto já aparecia nos 128 tokens anteriores 62% das vezes contra 49% no texto real.
  - ~40% menos tokens únicos (11,6k contra 18,9k).
- **Relação com preferências:**
  - O baseline "Unlikelihood" do DPO maximiza $\log p(y_w)$ e minimiza $\log p(y_l)$ **sem** o peso adaptativo. O coeficiente $\alpha\in[0,1]$ fica no termo de unlikelihood, com sweep $\alpha\in\{0{,}05;0{,}1;0{,}5;1\}$ no experimento de sentimento. Ele ficou **fora** dos experimentos de resumo e diálogo porque gera respostas "em geral sem sentido", que os autores atribuem à minimização irrestrita de verossimilhança. A Tabela 3 do DPO mostra uma amostra degenerada ("when when when…").
  - **[interpretação]** O termo $-\log(1-p)$ é o "lado negativo" de uma entropia cruzada binária. É o mesmo espírito do lado indesejável do KTO, mas sem referência nem saturação.

### 10.5 Mapa unificador [síntese minha]

Quase tudo neste módulo cabe em "**escore** $s_\theta(x,y)$ + **função de contraste** + **regularizador**":

| Método | Escore $s_\theta(x,y)$ | Contraste | Regularização / âncora |
|---|---|---|---|
| DPO | $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | logística em $s_w-s_l$ | $\pi_{\text{ref}}$ (dentro do escore) |
| IPO | $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | quadrática para alvo $1/(2\tau)$ | $\pi_{\text{ref}}$ + alvo finito |
| KTO | $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | sigmoide contra ponto de referência $z_0$ (não par a par) | $\pi_{\text{ref}}$ + $z_0$ (KL) |
| SimPO | $\frac{\beta}{\lvert y\rvert}\log\pi_\theta$ | logística com margem $\gamma$ | nenhuma explícita (LR pequeno) |
| ORPO | $\log\text{odds}(\bar P_\theta)$ | logística | NLL do escolhido |
| CPO / CPL (bandit) | $\beta\log\pi_\theta$ | logística | NLL (CPO) / viés λ nos negativos (CPL) |
| SLiC-HF | $\log\pi_\theta$ | hinge com margem $\delta$ | CE no alvo |
| InfoNCE | $f(x,c)$ arbitrário | softmax 1 contra N | — |
| Unlikelihood | $p_\theta(c)$ por token | $-\log(1-p)$ | NLL do alvo |
| AWR / RAFT / ReST | — (não contrasta) | peso $\exp(A/\beta)$, $\mathbb 1[\cdot]$ ou argmax | é o próprio SFT (fica no suporte dos dados) |

---

## 11. Sugestões de figuras animadas (com dados da §12)

1. **"Inclinação exponencial" (§1.4):** barras de $\pi_{\text{ref}}$ sobre 5–8 respostas, multiplicadas por $e^{r/\beta}$ e renormalizadas. Slider de β: grande deixa as barras como a referência, pequeno concentra no argmax. Reusar a mesma figura em AWR (§9) com $A$ no lugar de $r$.
2. **Cancelamento do $Z(x)$ (§2.1):** animação da Eq. 5 entrando no BT, com o termo $\beta\log Z(x)$ "evaporando".
3. **Curva da perda DPO vs $\Delta h$** e o peso $\sigma(-\beta\Delta h)$ (§12.2), com slider de β.
4. **DPO (rampa) × IPO (parábola)** com o ponto do par de brinquedo e setas de gradiente que **invertem** no IPO além de $1/(2\tau)$ (§12.3).
5. **Bandit de 3 ações:** barras de $\pi$ ao longo dos passos para DPO e IPO com τ ∈ {0,1; 0,5; 1} (§12.4).
6. **Função de valor de Kahneman–Tversky** (potência, α=0,88, λ=2,25) sobreposta à logística do KTO. Sliders de β e λ_D/λ_U. Marcar o ponto de referência $z_0$ deslizando (§12.5).
7. **SimPO:** duas respostas de comprimentos diferentes, com toggle "soma ↔ média" mostrando o logit de −17 → −0,1 e um slider de γ (§12.6).
8. **Likelihood displacement:** três barras (No, Never, Yes) evoluindo enquanto a margem No/Never cresce. "Yes" engole tudo (§12.10).
9. **Espectro do SFT ponderado:** 4 amostras com recompensas e um slider de β em $\exp(A/\beta)$, de "SFT uniforme" a "best-of-K", com marcadores de ReST (limiar) e ReST-EM (binário) (§12.12).
10. **InfoNCE:** um positivo e N negativos num círculo, softmax animado. Com N=2 vira o sigmoide do DPO (§12.13).
11. **Corcova da sobre-otimização:** curva $R(d)=d(\alpha-\beta\log d)$ com parâmetros **ilustrativos** (o paper não dá valores ajustados reutilizáveis aqui; rotular como forma qualitativa).

---

## 12. Exemplos numéricos de brinquedo (ilustrativos; calculados por script)

### 12.1 O par de brinquedo (usado em todas as perdas)

| | $y_w$ (preferida) | $y_l$ (preterida) |
|---|---|---|
| comprimento $\lvert y\rvert$ | 20 tokens | 10 tokens |
| $\log\pi_{\text{ref}}(y\mid x)$ | −26,0 | −15,0 |
| $\log\pi_\theta(y\mid x)$ (após alguns passos) | −25,0 | −17,0 |
| $h=\log\pi_\theta-\log\pi_{\text{ref}}$ | **+1,0** | **−2,0** |
| log-prob média $\frac1{\lvert y\rvert}\log\pi_\theta$ | −1,25 | −1,70 |

$\Delta h=3{,}0$. A preferida é **mais longa**, o caso típico de viés de verbosidade.

### 12.2 DPO

| β | $\hat r_w$ | $\hat r_l$ | logit $u=\beta\Delta h$ | perda $-\log\sigma(u)$ | peso $\sigma(-u)$ | coef. do gradiente $\beta\sigma(-u)$ |
|---|---|---|---|---|---|---|
| 0,01 | 0,010 | −0,020 | 0,03 | 0,6783 | 0,4925 | 0,00493 |
| 0,05 | 0,050 | −0,100 | 0,15 | 0,6210 | 0,4626 | 0,02313 |
| **0,1** | **0,100** | **−0,200** | **0,30** | **0,5544** | **0,4256** | **0,04256** |
| 0,5 | 0,500 | −1,000 | 1,50 | 0,2014 | 0,1824 | 0,09121 |
| 1,0 | 1,000 | −2,000 | 3,00 | 0,0486 | 0,0474 | 0,04743 |

- No início ($\pi_\theta=\pi_{\text{ref}}$): perda $\log2=0{,}6931$, peso 0,5.
- Curva (β=0,1), pontos $(\Delta h,\ \text{perda})$: (−10; 1,3133), (−5; 0,9741), (0; 0,6931), (3; 0,5544), (5; 0,4741), (10; 0,3133), (20; 0,1269), (50; 0,0067). **A perda nunca chega a 0. Sempre há incentivo para aumentar $\Delta h$.**

### 12.3 IPO

**Forma do paper (log-probs somadas), alvo $1/(2\tau)$:**

| τ | alvo | $\Delta h=0$ | $\Delta h=3$ (nosso par) | $\Delta h=7$ |
|---|---|---|---|---|
| 0,1 | 5,0 | perda 25, $\partial/\partial\Delta h=-10$ (sobe) | perda 4, −4 (sobe) | perda 4, **+4 (puxa de volta)** |
| 0,5 | 1,0 | 1, −2 | 4, **+4** (já passou do alvo) | 36, +12 |
| 1,0 | 0,5 | 0,25, −1 | 6,25, +5 | 42,25, +13 |

**Forma do TRL (média por token):** $\Delta h_{\text{norm}}=1/20-(-2)/10=0{,}25$.

| τ (`beta`) | alvo | perda |
|---|---|---|
| 0,1 | 5,0 | 22,5625 |
| 0,5 | 1,0 | 0,5625 |
| 1,0 | 0,5 | 0,0625 |
| 2,0 | 0,25 | 0,0000 |

**Lição:** o mesmo τ significa coisas muito diferentes com ou sem normalização por comprimento.

### 12.4 DPO × IPO no bandit de 3 ações ($\mathcal D_1$; referência uniforme; Adam LR 0,01; full batch; reprodução minha)

| τ | passo | DPO $\pi(a,b,c)$ | IPO $\pi(a,b,c)$ |
|---|---|---|---|
| 0,1 | 100 | (0,6616; 0,2466; 0,0919) | (0,6493; 0,2525; 0,0982) |
| 0,1 | 1.000 | (0,9998; 0,0002; 0,0000) | (0,9642; 0,0345; 0,0012) |
| 0,1 | 18.000 | (1,0; 0,0; 0,0) | (0,9644; 0,0344; 0,0012) |
| 0,5 | 1.000 | (0,9930; 0,0069; 0,0000) | (0,5627; 0,2889; 0,1483) |
| 0,5 | 18.000 | (1,0; 0,0; 0,0) | (0,5627; 0,2889; 0,1483) |
| 1,0 | 1.000 | (0,9682; 0,0308; 0,0010) | (0,4484; 0,3213; 0,2302) |
| 1,0 | 18.000 | (1,0; 0,0; 0,0) | (0,4484; 0,3213; 0,2302) |

A solução fechada do IPO, $\theta_a-\theta_b=\theta_b-\theta_c=1/(3\tau)$, reproduz exatamente as linhas do IPO. Caso de 2 ações determinístico: ver §3.6.

### 12.5 KTO (β = 0,1; $\lambda_D=\lambda_U=1$; $\hat z_0=0{,}5$ ilustrativo)

| Exemplo | $r_\theta$ | $v$ | perda $\lambda-v$ | coef. do gradiente $\beta v(1-v)$ |
|---|---|---|---|---|
| $y_w$ como **desejável** | +1,0 | $\sigma(0{,}1\cdot(1{,}0-0{,}5))=0{,}51250$ | 0,48750 | 0,02498 |
| $y_l$ como **indesejável** | −2,0 | $\sigma(0{,}1\cdot(0{,}5+2{,}0))=0{,}56218$ | 0,43782 | 0,02461 |

- Média: 0,46266. Se a média desencontrada fosse −0,3, o *clamp* dá $\hat z_0=0$, e as perdas passam a 0,47502 e 0,45017.
- **Escala:** os coeficientes de gradiente (~0,025) são ~0,6× o do DPO no mesmo par (0,0426). Isso ajuda a ver por que o paper recomenda LR maior para o KTO, embora a razão real dependa dos dados.
- Curva $v_D(r)$ (β=0,1, $z_0=0{,}5$): (−40; 0,0171), (−20; 0,1141), (−10; 0,2592), (0; 0,4875), (0,5; 0,5), (10; 0,7211), (20; 0,8754), (40; 0,9811). As caudas achatadas ilustram a Proposição 4.1.
- **Kahneman–Tversky (α=0,88, λ=2,25):** $v(-10)=-17{,}068$; $v(-1)=-2{,}25$; $v(0)=0$; $v(1)=1$; $v(10)=7{,}586$.
- **Desbalanceamento 1:10** ($n_D=100$, $n_U=1000$): $\lambda_U=1$, $\lambda_D\in[10,15]$, o que dá $\lambda_Dn_D/(\lambda_Un_U)\in[1{,}0;1{,}5]$.

### 12.6 SimPO (médias por token −1,25 e −1,70)

| β | γ | γ/β | $r_w$ | $r_l$ | logit $r_w-r_l-\gamma$ | perda | peso $s_\theta$ |
|---|---|---|---|---|---|---|---|
| 2,0 | 1,0 | 0,50 | −2,500 | −3,400 | −0,10 | 0,7444 | 0,5250 |
| 2,5 | 1,4 | 0,56 | −3,125 | −4,250 | −0,275 | 0,8401 | 0,5683 |
| 2,0 | 1,6 | 0,80 | −2,500 | −3,400 | −0,70 | 1,1032 | 0,6682 |
| 2,5 | 0,3 | 0,12 | −3,125 | −4,250 | +0,825 | 0,3634 | 0,3047 |
| 10 | 5,0 | 0,50 | −12,5 | −17,0 | −0,50 | 0,9741 | 0,6225 |

- **Sem normalização por comprimento** (somas; β=2, γ=1): logit $=2\cdot(-25+17)-1=-17$, perda ≈ **17,0**. A soma "acha" que a preferida (mais longa) é muito pior. Para corrigir, o modelo teria de inflar a probabilidade da sequência longa. É o mecanismo da degeneração descrita no paper.

### 12.7 Outras perdas no mesmo par

| Método | Cálculo | Resultado |
|---|---|---|
| DPO normalizado por comprimento (Tülu 3), β=5 | logit $5\cdot0{,}25=1{,}25$ | perda 0,2519 (β=1: 0,5759; β=10: 0,0789) |
| R-DPO, β=0,1, α=0,01 | logit $0{,}3+0{,}01\cdot(20-10)=0{,}4$ | perda 0,5130, peso 0,4013 (α=0,005: 0,5334; α=0,05: logit 0,8, perda 0,3711, peso 0,3100) |
| ORPO, λ=0,1 | $P_w=e^{-1{,}25}=0{,}2865$; $P_l=e^{-1{,}70}=0{,}1827$; odds 0,4016 e 0,2235; log OR = 0,5859; $\mathcal L_{\text{OR}}=0{,}4425$; $\mathcal L_{\text{SFT}}=1{,}25$ | total **1,2943**. A log-razão de probabilidades (médias) seria 0,45: o odds amplia um pouco a diferença quando $P$ é moderado |
| CPO, β=0,1 | $\mathcal L_{\text{prefer}}=-\log\sigma(0{,}1\cdot(-25+17))=-\log\sigma(-0{,}8)$ | 1,1711; NLL 25,0 (soma) ou 1,25 (por token) |
| SLiC-HF, δ=1 | $\max(0,1-(-25)+(-17))$ | 9,0 (log-prob crua penaliza a mais longa) |
| `hinge` RSO/TRL, β=0,1 | $\max(0,1-0{,}1\cdot3)$ | 0,7 |
| DPOP, β=0,3, λ=50 | se $\log\pi_\theta(y_w)$ cai para −27 ($h_w=-1$): DPO logit $0{,}3\cdot(-1+2)=0{,}3$, perda 0,5544; penalidade $\max(0,-26+27)=1$, DPOP logit $0{,}3\cdot(1-50)=-14{,}7$ | perda **14,7**: o DPOP "grita" quando a preferida cai |
| β-DPO (β0=0,1; α=0,6; $M_0=0{,}2$) | $\beta_{\text{batch}}=[1+0{,}6(\bar M-0{,}2)]\cdot0{,}1$ | $\bar M=0$: 0,088; $0{,}2$: 0,100; $0{,}5$: 0,118; $1{,}0$: 0,148 |

### 12.8 Mensagem das tabelas 12.2–12.7
- Com a preferida mais longa, os métodos **sem referência e sem normalização** (CPO, SLiC) veem um par "errado" e empurram com força.
- O DPO "desconta" o comprimento via $\pi_{\text{ref}}$.
- SimPO e LN-DPO o removem por média.
- O R-DPO o desconta explicitamente.

### 12.9 Toy do gradiente do DPO por prefixo comum [derivação minha]
- **Setup:** $y_w$ = "A B C" e $y_l$ = "A B D". Os termos de $\log\pi(A\mid x)$ e $\log\pi(B\mid x,A)$ aparecem iguais nos dois lados e cancelam.
- **Resultado:** o gradiente do DPO é $\propto\nabla\log\pi(C\mid x,AB)-\nabla\log\pi(D\mid x,AB)$. Toda a "aula" está em um único token.

### 12.10 Likelihood displacement em miniatura (ilustrativo, inspirado no Teorema 1 de Razin et al.)
- **Setup:** vocabulário {No, Never, Yes}, unembeddings 2D $W_{\text{No}}=(1;\,0{,}2)$, $W_{\text{Never}}=(1;\,-0{,}2)$ (muito parecidos), $W_{\text{Yes}}=(0;1)$ (alinhado com $W_{\text{No}}-W_{\text{Never}}=(0;\,0{,}4)$). Estado oculto inicial $h=(2;0)$.
- **Dinâmica:** treinar só $h$ com DPO (No ≻ Never) move $h$ na direção $(0;1)$, porque o gradiente de $\log\pi(\text{No})-\log\pi(\text{Never})$ em $h$ é $W_{\text{No}}-W_{\text{Never}}$. Com $h=(2;s)$, os logits são $(2+0{,}2s;\ 2-0{,}2s;\ s)$:

| $s$ | p(No) | p(Never) | p(Yes) | log-razão No/Never |
|---|---|---|---|---|
| 0 | 0,4683 | 0,4683 | 0,0634 | 0,00 |
| 1 | 0,5072 | 0,3400 | 0,1528 | 0,40 |
| 2 | 0,4718 | 0,2120 | 0,3162 | 0,80 |
| 3 | 0,3580 | 0,1078 | 0,5341 | 1,20 |
| 5 | 0,1173 | 0,0159 | 0,8668 | 2,00 |
| 10 | 0,0025 | 0,0000 | 0,9975 | 4,00 |

- **Leitura:** a perda DPO só cai (a margem cresce), mas "No" sobe um pouco e depois **despenca**, e "Yes" (o oposto!) domina.
- **Ressalva:** no teorema do paper, as unembeddings também treinam. Isto é uma caricatura didática.

### 12.11 Unlikelihood
$-\log(1-p)$: p=0,05 → 0,0513; p=0,3 → 0,3567; p=0,9 → 2,3026.

### 12.12 AWR × RWR × filtros (4 amostras para o mesmo prompt)
- **Dados:** recompensas $R=(1{,}0;\ 0{,}0;\ 0{,}5;\ 0{,}2)$. Linha de base $V=0{,}425$ (média), vantagens $A=(0{,}575;\ -0{,}425;\ 0{,}075;\ -0{,}225)$.

| β | pesos $e^{A/\beta}$ | normalizados | normalizados com clip $\omega_{\max}=20$ |
|---|---|---|---|
| 0,05 | (98715,8; 0,0002; 4,48; 0,011) | (1,0; 0; 0; 0) | (0,8166; 0,0; 0,1830; 0,0005) |
| 0,1 | (314,19; 0,0143; 2,117; 0,1054) | (0,9929; 0,0; 0,0067; 0,0003) | (0,8994; 0,0006; 0,0952; 0,0047) |
| 0,5 | (3,158; 0,427; 1,162; 0,638) | (0,5865; 0,0794; 0,2158; 0,1184) | igual |
| 1,0 | (1,777; 0,654; 1,078; 0,799) | (0,4126; 0,1518; 0,2502; 0,1854) | igual |
| 10 | (1,059; 0,958; 1,008; 0,978) | (0,2646; 0,2394; 0,2517; 0,2443) | igual |

- **RWR** (sem linha de base), β=0,5: pesos (7,389; 1,0; 2,718; 1,492), **mesmos normalizados** que o AWR (a linha de base de um mesmo prompt cancela na normalização).
- **RAFT (best-of-4):** (1, 0, 0, 0). **ReST** com τ=0,4: (1, 0, 1, 0). **ReST-EM** binário com "correto" ⇔ $R\ge0{,}5$: (1, 0, 1, 0).

### 12.13 InfoNCE
- **Setup:** escore positivo 2,0, negativos (1,0; 0,5; −0,5).
- N=4: $\mathcal L=0{,}5147$, e o limite de MI é $\log4-\mathcal L=0{,}8716$ nats.
- Com 1 negativo (escore 1,0): $\mathcal L=0{,}3133=-\log\sigma(2{,}0-1{,}0)$, o mesmo número que um DPO com margem 1.

### 12.14 β no DPO: o mesmo par com β 10× maior
De 0,1 para 1,0, o logit vai de 0,3 a 3,0, a perda de 0,554 a 0,049 e o peso de 0,426 a 0,047. Com β grande, o par "já está resolvido" com pouco afastamento da referência. É a leitura "β grande = fica perto de $\pi_{\text{ref}}$".

### 12.15 CPL com regularizador "bias" (α=0,1; somas −25 e −17)
- Vanilla (= CPO $\mathcal L(\pi,U)$): 1,1711. Com λ=0,5: 1,8257.
- Subindo **ambas** as log-probs em +1 nat: a vanilla fica em 1,1711 (invariante) e a com λ cai para 1,7839. **O regularizador recompensa pôr massa nas respostas do dataset.**

---

## 13. Ponte com o Módulo 1 (receitas de inferência)

1. **Geração rápida é o gargalo dos métodos online e iterativos.**
   - OAIF, Online DPO, RAFT, ReST e ReST-EM precisam gerar muitas respostas por prompt.
   - O Tülu 3 descreve um setup RL assíncrono em que "a inferência roda via vLLM enquanto os learners fazem as atualizações".
   - O `OnlineDPOTrainer` do TRL tem `use_vllm` (`colocate` ou `server`).
   - Tudo que o Módulo 1 ensinou (KV cache, batching contínuo, speculative decoding) é diretamente útil aqui.
2. **Comportamento de recusa: treinar vs editar.**
   - A receita GLM do Módulo 1 tem **abliteração** em tempo de carga (`ABLIT=1`). Ela ortogonaliza `o_proj` contra uma "direção de recusa", $W'=(I-\alpha rr^\top/\lVert r\rVert^2)W$ (`docs/pesquisa/glm-mods.md` §2.10/§3.J): edição de pesos **sem treino**.
   - O paper de *likelihood displacement* mostra o lado oposto: um DPO **feito para aumentar** recusas pode reduzi-las sem querer (74,4% → 33,4%).
   - Bom gancho: comportamentos de alinhamento vivem em direções do espaço residual, e otimização de preferências mexe nelas de forma nem sempre intencional.
3. **Servir um modelo pós-treinado não muda a receita.** DPO, KTO etc. produzem pesos com a mesma arquitetura. Se o treino foi com LoRA, o adaptador pode ser mesclado antes da quantização.

---

## 14. Incertezas, divergências entre fontes e cuidados de citação

1. **Nome do IPO:** "Identity-PO", dentro do ΨPO (*preference optimisation*), no paper original, e "identity policy optimisation" no OAIF (mesmo grupo). Use "Identity Preference Optimization" e cite a variante.
2. **Normalização do IPO:** o paper escreve log-probs de sequência. O TRL divide por número de tokens e afirma ter confirmado com os autores que os resultados do paper correspondem a isso. Escalas de τ entre implementações **não são comparáveis**.
3. **LR "padrão" do DPO:** o paper do DPO usa 1e-6 com RMSprop (batch 64, modelos ≤ 6B). O KTO v5 afirma que o padrão do DPO a 8B é 5e-7, citando Rafailov et al. O SimPO achou 5e-7 a 7e-7 ótimos com β=0,01. Trate como faixa, não como constante.
4. **R-DPO α:** o paper usa α=0,01 (HH) e 0,05 (TL;DR) na Fig. 2, sweeps {0; 0,005; 0,01} (HH) e {0; 0,2; 0,5} (TL;DR) e α=0,1 na Fig. 5. Os números dependem da escala de β e do comprimento.
5. **SimPO γ (Mistral-Instruct):** o paper (Tabela 8) dá β=2,5 e γ=0,3 (γ/β=0,12), e o README atual dá γ/β=0,1. O README reflete uma reimplementação ("gamma" virou `gamma_beta_ratio`).
6. **AWAC, direção do KL:** o texto principal chama a projeção $\mathbb D_{\text{KL}}(\pi^*\|\pi_\theta)$ de "forward KL", e o apêndice chama de "reverse KL". Na convenção usual de ML é a projeção-M ("forward" a partir do alvo). Descreva pela fórmula, não pelo rótulo.
7. **TRL `sigmoid_norm`:** a documentação associa ao SimPO, mas o código mantém a referência. É DPO normalizado por comprimento. SimPO de verdade é `CPOTrainer(loss_type="simpo", cpo_alpha=0)`.
8. **APIs mudam:** CPO, ORPO e Online DPO estão em `trl.experimental.*` no TRL `1.15.0.dev0` (01/10/2026). O `DPOTrainer` foi refatorado (`loss_type` virou lista com `loss_weights`, e não há mais `rpo_alpha`: use `"sft"` na lista).
9. **KTO v5** (08/09/2026) traz a Tabela 1 nova (Llama-3/Qwen2.5) e recomendações de LR revistas em relação às primeiras versões. Uma nota de rodapé diz que um rascunho anterior separava $\lambda_D,\lambda_U$ numa função $w$. Citações de versões antigas podem ter fórmulas com outra arrumação.
10. **Escores de benchmarks** (AlpacaEval, Arena-Hard, MT-Bench, GSM8K) são dependentes de juiz e de versão, e os próprios papers alertam sobre viés de comprimento do GPT-4. Use-os só como "o paper reporta".
11. **[interpretação]** As conexões "DPO = InfoNCE com 1 negativo", "CPL no MDP de tokens = CPO/DPO com referência uniforme" e "AWR/ReST-EM/DPO compartilham o ótimo inclinado" são derivações minhas a partir das equações citadas. O DPO⊂CPL e a ligação CPL→InfoNCE são afirmações explícitas do paper do CPL.

---

## 15. Referências (todas verificadas em 2026-10-01)

**Família DPO e variantes**
- Rafailov, Sharma, Mitchell, Ermon, Manning, Finn — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — https://arxiv.org/abs/2305.18290
- Azar, Rowland, Piot, Guo, Calandriello, Valko, Munos — *A General Theoretical Paradigm to Understand Learning from Human Preferences* (ΨPO/IPO) — https://arxiv.org/abs/2310.12036
- Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela — *KTO: Model Alignment as Prospect Theoretic Optimization* — https://arxiv.org/abs/2402.01306
- Meng, Xia, Chen — *SimPO: Simple Preference Optimization with a Reference-Free Reward* — https://arxiv.org/abs/2405.14734 ; código https://github.com/princeton-nlp/SimPO
- Hong, Lee, Thorne — *ORPO: Monolithic Preference Optimization without Reference Model* — https://arxiv.org/abs/2403.07691
- Xu et al. — *Contrastive Preference Optimization* (CPO) — https://arxiv.org/abs/2401.08417
- Park, Rafailov, Ermon, Finn — *Disentangling Length from Quality in Direct Preference Optimization* (R-DPO) — https://arxiv.org/abs/2403.19159
- Wu et al. — *β-DPO: Direct Preference Optimization with Dynamic β* — https://arxiv.org/abs/2407.08639
- Pal et al. — *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive* — https://arxiv.org/abs/2402.13228
- Guo et al. — *Direct Language Model Alignment from Online AI Feedback* (OAIF / Online DPO) — https://arxiv.org/abs/2402.04792
- Xiong et al. — *Iterative Preference Learning from Human Feedback* — https://arxiv.org/abs/2312.11456
- Yuan et al. — *Self-Rewarding Language Models* — https://arxiv.org/abs/2401.10020
- Pang et al. — *Iterative Reasoning Preference Optimization* — https://arxiv.org/abs/2404.19733
- Liu et al. — *Statistical Rejection Sampling Improves Preference Optimization* (RSO) — https://arxiv.org/abs/2309.06657
- Gorbatovski et al. — *Learn Your Reference Model for Real Good Alignment* (TR-DPO) — https://arxiv.org/abs/2404.09656
- Zhou et al. — *WPO: Enhancing RLHF with Weighted Preference Optimization* — https://arxiv.org/abs/2406.11827
- Rafailov, Hejna, Park, Finn — *From r to Q\*: Your Language Model is Secretly a Q-Function* — https://arxiv.org/abs/2404.12358
- Lambert et al. — *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* — https://arxiv.org/abs/2411.15124

**Modos de falha / análises**
- Razin, Malladi, Bhaskar, Chen, Arora, Hanin — *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization* — https://arxiv.org/abs/2410.08847
- Rafailov, Chittepu, Park, Sikchi, Hejna, Knox, Finn, Niekum — *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms* — https://arxiv.org/abs/2406.02900
- Gao, Schulman, Hilton — *Scaling Laws for Reward Model Overoptimization* — https://arxiv.org/abs/2210.10760
- Xu et al. — *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study* — https://arxiv.org/abs/2404.10719

**Regressão ponderada / auto-treino**
- Peng, Kumar, Zhang, Levine — *Advantage-Weighted Regression* — https://arxiv.org/abs/1910.00177
- Nair, Gupta, Dalal, Levine — *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets* — https://arxiv.org/abs/2006.09359
- Peters & Schaal (2007) — *Reinforcement learning by reward-weighted regression for operational space control* (ICML 2007), citado via AWR
- Dong et al. — *RAFT: Reward rAnked FineTuning* — https://arxiv.org/abs/2304.06767
- Gulcehre et al. — *Reinforced Self-Training (ReST) for Language Modeling* — https://arxiv.org/abs/2308.08998
- Singh et al. — *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* (ReST-EM) — https://arxiv.org/abs/2312.06585
- Baheti et al. — *Leftover Lunch: Advantage-based Offline RL for Language Models* (A-LoL) — https://arxiv.org/abs/2305.14718
- Zelikman et al. — *STaR* — https://arxiv.org/abs/2203.14465 ; Yuan et al. — *Scaling Relationship on Learning Mathematical Reasoning* (RFT) — https://arxiv.org/abs/2308.01825 ; Touvron et al. — *Llama 2* — https://arxiv.org/abs/2307.09288

**Contraste**
- van den Oord, Li, Vinyals — *Representation Learning with Contrastive Predictive Coding* (InfoNCE) — https://arxiv.org/abs/1807.03748
- Zhao et al. — *Calibrating Sequence Likelihood Improves Conditional Language Generation* (SLiC) — https://arxiv.org/abs/2210.00045
- Zhao et al. — *SLiC-HF: Sequence Likelihood Calibration with Human Feedback* — https://arxiv.org/abs/2305.10425
- Hejna et al. — *Contrastive Preference Learning: Learning from Human Feedback without RL* — https://arxiv.org/abs/2310.13639
- Welleck et al. — *Neural Text Generation with Unlikelihood Training* — https://arxiv.org/abs/1908.04319
- Yuan et al. — *RRHF* — https://arxiv.org/abs/2304.05302 ; Chen et al. — *Noise Contrastive Alignment* (NCA) — https://arxiv.org/abs/2402.05369

**Implementação / infraestrutura**
- TRL docs (DPO Trainer) — https://huggingface.co/docs/trl/main/en/dpo_trainer ; código: https://github.com/huggingface/trl (arquivos `trl/trainer/dpo_trainer.py`, `trl/trainer/kto_trainer.py`, `trl/trainer/kto_config.py`, `trl/experimental/cpo/*`, `trl/experimental/orpo/*`, `trl/experimental/online_dpo/*`; versão `1.15.0.dev0`)
- Rajbhandari et al. — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — https://arxiv.org/abs/1910.02054
- Tversky & Kahneman (1992) — *Advances in Prospect Theory: Cumulative Representation of Uncertainty*, Journal of Risk and Uncertainty 5 (citado via KTO)

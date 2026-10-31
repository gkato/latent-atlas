# TensorFold — a receita GLM-5.3-Flash EXL3 em 2× DGX Spark sem vLLM (visão geral, arquitetura, inventário)

> Pesquisa para o Latent Atlas, escrita em 2026-10-01.
>
> **Fontes lidas (somente leitura):**
> - `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`, HEAD `92bf731` (merge do PR #20 = **v1.3.2**, 2026-10-01). Lidos integralmente: `README.md`, `CHANGELOG.md`, `CREDITS.md`, `NOTICE`, `start.sh`, `stop.sh`, `scripts/config.sh`, `scripts/nodes.sh`, `scripts/prepare.sh`, `scripts/publish-image.sh`, `scripts/local.sh.example`, `tools/client.py`, `tools/needle.py`, `tools/toolcheck.py`; de `scripts/banner.sh` só o cabeçalho (é arte ASCII do banner). **Não existe `.env.example`** neste repositório: o README descreve `.env` como arquivo do usuário (linhas `KEY=value`, lidas e nunca executadas). Dos 53 `patches/*.patch` li os arquivos tocados e todas as docstrings/comentários adicionados (é lá que o projeto explica o desenho), com leitura de código nos pontos citados.
> - `ashhart/TensorFold`, tag **`v0.6.0`** (release de 30 Sep 2026): `README.md`, `RUNBOOK.md`, `CHANGELOG.md` (0.6.0, 0.5.0, 0.4.0), `docs/api.md`, `docs/recipes/glm-5.3-flash.md`, `docs/recipes/cuda.md`, e código em `src/tensorfold/cuda/{comm,capacity,geometry,sampling}.py`, `src/tensorfold/engine/exact_sampling.py`, `src/tensorfold/families/glm5_next/cuda/{engine,decode,dflash2}.py` (trechos).
> - `jayleaton/glm53-tensorfold-spark` @ `ad63acf` (2026-10-01): `README.md` e `docs/THEORY-2.md` (trechos) — fonte de dois patches adaptados (0046, 0047) e de comparações vLLM×TensorFold **de terceiros**.
> - `local-inference-lab/b12x` @ `a321f9a`: `docs/rocenante.md` (o transporte RoCE one-shot usado no patch 0006).
> - `MiaAI-Lab/sparkDash` @ `b4228a3`: `README.md` (protocolo do benchmark).
> - Receita anterior `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` @ `6278ecb` (README) e o inventário já existente `docs/pesquisa/glm-launch.md` (feito sobre o HEAD `94ae731`).
> - Web (terceiros): bake-off da LimeChain (`github.com/LimeChain/glm-5.3-flash-2xdgx-spark`) e a issue #167 do TensorFold.
>
> **Convenções:** **(repo)** = afirmação do repositório da receita; **(TF)** = afirmação/código do TensorFold upstream; **(código)** = lido no código dos patches; **(interpretação)** = explicação minha; **(incerto)** = não verificado, ambíguo ou divergente; **(terceiros)** = medição de outro projeto, não da Mia. Identificadores, flags, código e fórmulas ficam como no original.

---

## 0. Resumo em 12 linhas

1. A Mia's AI Lab publicou uma segunda receita do **mesmo checkpoint** (`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, experts roteados EXL3 4 bpw, resto BF16, ~176 GB) em **2× DGX Spark**, mas **trocou o motor**: em vez de vLLM, roda **TensorFold v0.6.0** (Ash Hart, Apache-2.0) + **53 patches** próprios.
2. TensorFold é um motor **por família de modelo**: cada família traz seus kernels (Triton + CUDA C++) e sua verificação de rascunhos; não há scheduler genérico de blocos paginados como no vLLM.
3. O contrato central é **exatidão**: a amostragem usa **chave** (semente derivada do prompt, ruído Gumbel indexado por (semente, posição, token)), então **resposta com rascunhos = decodificação serial**, **4 requisições juntas = cada uma sozinha**, **prompt retomado = prompt calculado do zero**.
4. Um **rank por Spark**: rank 0 decide tudo que depende de tempo/clientes e manda **uma mensagem por iteração**; rank 1 aplica. Rendezvous por `TCPStore` na porta 29551; NCCL chamado via `ctypes`; só **all-gather** + soma em ordem de rank (bits iguais nos dois ranks).
5. Pequenos all-gathers (decode) vão por **RDMA one-shot sobre RoCE** (transporte "RoCEnante" do b12x): 11 µs por gather de 16 KiB contra 45 µs do NCCL, **+6% de decode**.
6. Memória: orçamento = `MemAvailable` na partida − reserva (14,5 GiB aqui); o que não cabe é **recusado antes de carregar pesos** (na memória unificada do GB10, faltar memória tende a **congelar a máquina**, não a falhar uma alocação).
7. Um **pool único de caches por token** (latente DSA + chaves do indexador em **FP8 e4m3 com escala potência de 2**, 528 B/token/camada) é dividido em *extents* de 2.048 tokens entre até **4 streams** e os **estados de prompt guardados**: **2.922.496 tokens** na partida medida, janela de **1.048.576** por requisição.
8. Reuso de prompt sem APC: estados guardados numa **grade de 64 tokens**, **prefixo compartilhado** (system prompt) e **replay** (o estado final + a linha de logits da cabeça): prompt idêntico de 64k em **<0,07 s** (34 s a frio); system prompt de 7,9k em **0,13 s** (4,24 s).
9. Rascunhos: **DFlash2** (até 7 por rodada, parada "consciente do ruído" `fnc7:0.3`) + **copy drafts** (prompt lookup, até 15) verificados em **janelas de até 16 linhas** por stream (32 no lote), cada largura com seu **CUDA graph**.
10. Números (sparkDash, clocks travados em 2.200 MHz): prosa **60,4 tok/s** com 1 requisição e **108,8 tok/s** agregados com 4; structured **114,7 / 227,9**; prefill **~1.950 tok/s até 64k**, 1.642 a 262k, 1.015 a 982k (agulha encontrada).
11. Contra a receita vLLM da própria Mia (medições em condições não idênticas): prosa ×1 36,1 → 60,4; ×4 75,3 → 108,8; prefill frio ~1.411–1.511 → ~1.950; contexto 850k → 1.048.576; pool 1,23M → 2,92M tokens.
12. Perdas: só 2 Sparks (sem TP3/TP4), máx. 4 requisições, sem `logprobs`/`n>1`/penalidades, sem API key, defaults com perdas (`DENSE=q4`, `KV=fp8`, `KDA_CHUNKED=1`), problema de fim-de-turno em prompts curtos não-ingleses com `q4` (#18), drafter DFlash2 não-comercial.

---

## 1. Visão geral: o que mudou em relação à receita vLLM, e por quê

### 1.1 O que a nova receita é

**(repo)** "Serve GLM-5.3-Flash from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through an OpenAI-compatible API, with 4 concurrent requests, the model's full 1,048,576-token context and image and video input." Roda TensorFold v0.6.0 nos dois Sparks (um rank em cada), dentro do container `nvcr.io/nvidia/pytorch:26.07-py3`, mais 53 patches: "DFlash2 and copy drafts, 4-bit dense weights, an FP8 KV cache, faster prompt kernels, a one-shot RoCE all-gather between the Sparks, several requests over one shared cache pool, vision, tool calling, `/tokenize` and `/metrics`".

- Checkpoint: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, espelho byte-idêntico de `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (EXL3 nos experts roteados a 4 bits/peso, BF16 no resto, ~176 GB = 164 GiB), revisão pinada `9eaebb7c4e96d983dcd538e18624622ba5b820a8`.
- Drafter: `incoai/GLM-5.3-Flash-DFlash2` (revisão `bf582e4eacc1810f76656d1811693ff6c6737d2a`, ~2,3 GB, CC BY-NC-ND 4.0 = **só uso não-comercial**) ou a cabeça MTP do próprio checkpoint (`DRAFTER=mtp`).
- API: id `GLM-5.3-Flash-EXL3`, porta 8888, um comando no head: `./start.sh` (prepara os dois Sparks e sobe os dois ranks), `./stop.sh`.
- Versões (CHANGELOG, todas datadas 2026-10-01): v1.0 (TensorFold v0.5.0 + 52 patches, imagem `v0.5.0-cefe8bf45d07`) → v1.1 (até 50 imagens/4 vídeos; imagem pinada por digest) → v1.2 (`KV_POOL_GIB` 11 → 12,5; pesos do worker por NFS) → **v1.3** (TensorFold **v0.6.0**; três patches da Mia viraram upstream — o "idle doorbell", `TENSORFOLD_MEMORY_RESERVE_GIB` e `TF_GLM_MTP` — e foram removidos; 53 patches; imagem `v0.6.0-ae8d1c789b47`; `MAX_TOKENS` 4.096 → 32.768) → v1.3.1 (`TF_GLM_MULTI_LONE` → 0, correções de setup) → **v1.3.2** (`TF_GLM_CACHE_ENTRIES` 8 → 32; aviso sobre `DENSE=q4` e prompts não-ingleses).

### 1.2 Antes × depois, com números

> **Atenção às condições.** Os números da receita TensorFold vêm do sparkDash, pela API OpenAI, de outra máquina na rede, com **clocks da GPU travados em 2.200 MHz**, configuração default (4 streams, janela 1.048.576, KV FP8, dense q4, DFlash2 + copy drafts, visão ligada). O sparkDash usa "Lab protocol (temp 0, thinking off)" — **greedy**. Os números vLLM vêm de dois lugares: o README da receita vLLM (2026-09-28: "DFlash2, temperature 0, thinking off, mean of fresh boots", ferramenta de laboratório) e a corrida sparkDash de prosa de 2026-09-17 registrada em `glm-launch.md` (adaptive-k ema, `dense,kda`, MoE cooperativo, 850k, KV 14 GiB). **Não é um A/B controlado**: datas, configurações e (no caso do README vLLM) a ferramenta diferem. Use como ordem de grandeza.

| Dimensão | Receita vLLM (`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`) | Receita TensorFold (v1.3.2) |
|---|---|---|
| Motor | fork vLLM `0.1.dev20051+g487ecf187` (imagem `vllm/vllm-openai:glm53-flash-arm64-cu130`), patches de build no Dockerfile + 25 overlays Python reaplicados a cada `docker run` | TensorFold v0.6.0 (pip de tag git) + 53 diffs unificados aplicados no `site-packages` **durante o build da imagem** (`patch -p0 --forward`) |
| Processos | API server + EngineCore + executor `mp`, `--nnodes 2`, worker `--headless` | dois `tensorfold serve --tp 2 --rank R`; rank 0 serve HTTP; rank 1 só espelha |
| Pesos por rank | `Model loading took 79,65 GiB/rank` | "~80 GiB of weights on each Spark" (README) |
| Projeções densas (não roteadas) | FP8 weight-only via Marlin (`GLM53_DENSE_FP8=all`) | 4-bit afim em grupos de 64, faixa escolhida por MSE (`DENSE=q4`); cabeça em FP8; `kv_b` em BF16 |
| KV (latente MLA) | `fp8_ds_mla`: **656 B/token/camada** (512 NoPE fp8 + 16 B de escalas + 128 B de bloco RoPE zerado, padding para a geometria 576) | linhas FP8 e4m3 + escala fp32 potência de 2: **528 B/token/camada** (512 códigos + 4 B de escala + 12 B de pad) (código: `kv8.row_bytes`) |
| Contexto por requisição | 850.000 (default TP2) | **1.048.576** (nativo do modelo) |
| Concorrência | `MAX_NUM_SEQS=4` | `PARALLEL=4` (1 a 4; teto do código `MAX_SEGS=4`) |
| Pool de KV | 11 GiB fixos (`--kv-cache-memory-bytes`) → **1.233.779 tokens** (1,45× de 850k) | `MemAvailable` − 14,5 GiB, até 12,5 GiB por Spark além da janela → **2.922.496 tokens** na partida medida (~2,1–2,9M conforme a memória livre) |
| Especulação | DFlash2 k=7 (adaptive-k opt-in), `draft_sample_method=probabilistic`, rejection sampling | DFlash2 até 7 com parada `fnc7:0.3` + copy drafts até 15; verificação **exata** contra a amostra com chave |
| Reuso de prompt | APC (hash de blocos), acertos alinhados a páginas de **3.584 tokens** (estado KDA só em checkpoints alinhados) | estados de prompt guardados numa **grade de 64 tokens** + prefixo compartilhado + **replay** |
| Comunicação TP | NCCL | all-gather de parciais fp32 + soma em ordem de rank; os pequenos via RDMA one-shot (RoCE), os grandes via NCCL (2 trilhos, 4 canais) |
| Decode, 1 req (prosa, sparkDash, temp 0) | **36,1 tok/s**, TTFT 333 ms (2026-09-17) | **60,4 tok/s**, TTFT 170 ms |
| Decode, 4 req, agregado (prosa) | **75,3 tok/s** (19,4 por stream), TTFT 401 ms | **108,8 tok/s** (27,9 por stream), TTFT 340 ms |
| Structured, 1 req | 83,0 tok/s (README vLLM, 2026-09-28, lab) | 114,7 tok/s (sparkDash) |
| Prefill a frio | 1.411 / 1.511 / 1.498 tok/s @ 8k / 32k / 96k (2026-09-28) | 1.952 / 1.979 / 1.838 / 1.642 tok/s @ 8,2k / 32,8k / 131k / 262k |
| Prompt longo | 814.571 tokens: 633,7 s a frio (2026-09-21) | 981.841 tokens: 967 s (1.015 tok/s), agulha encontrada |
| Reuso | follow-up de ~8k: 8,53 → 1,30 s; repetição de 814.571 tokens: 2,748 s | prompt idêntico de 64k: 34 s → **<0,07 s**; mesmo system prompt de 7,9k numa conversa nova: 4,24 → **0,13 s** |
| Memória livre sob carga | head ≥ 6,2 GiB livres sob carga (11 GiB + páginas compactas) | mínimo 5,6 GiB (head) / 9,5 GiB (worker) sob um prompt de 1M; ocioso 7,0 / 10,5 GiB |
| Qualidade publicada | KLD 0,024555 do checkpoint 4 bpw (mede os pesos) | GSM8K 98,0% (250), HumanEval 97,6% (164), com KV FP8 + dense q4 |
| Topologias | 2, 3 (`start-tp3.sh`) e 4 (`start-tp4.sh`, experimental) Sparks | **só 2** (o motor CUDA do GLM exige dois ranks) |
| API | superfície vLLM (logprobs, `n>1`, API key Bearer opcional, rotas de reset de cache…) | sem `logprobs`/`top_logprobs` e `n≠1` (HTTP 400); `logit_bias` e penalidades aceitos e **ignorados**; sem `/v1/messages`; **nenhuma autenticação documentada** |
| Abliteração | `ABLIT=1`, checkpoint abliterado | não oferece |
| Boot | `./start.sh restart` ~122 s (InstantTensor, 2026-09-19) | 2–6 min para carregar ~80 GiB por Spark; ~2,2 min com pesos do worker por NFS; a primeira partida de cada imagem compila kernels |

### 1.3 Por que trocar o motor (interpretação fundamentada)

O README da receita TensorFold **não** traz uma seção "por que saímos do vLLM"; o repositório vLLM continua mantido (HEAD de 2026-10-01). O que segue é **(interpretação)**, ancorada nos documentos lidos.

1. **Um modelo híbrido exótico paga caro num motor genérico.** GLM-5.3-Flash mistura 34 camadas KDA (atenção linear com estado recorrente), 11 camadas DSA (MLA NoPE esparsa com indexador de pools de 4 tokens), MoE de 288 experts e hyper-connections de 4 fluxos. No vLLM isso exigiu: padding do latente de 512 para 576 só para casar com o kernel `fp8_ds_mla` (24% a mais de bytes de KV DSA), estado KDA tratado como `MambaSpec` com prefix caching **só em pontos alinhados a 3.584 tokens**, CUDA graphs FULL apenas para lotes uniformes, um scheduler "fair" próprio para domar passos mistos prefill+decode, e dezenas de overlays. O TensorFold escreve **um motor para a família `glm5_next`**: o cache guarda exatamente o latente de 512 (sem RoPE para preencher), o estado KDA é guardado onde for preciso, os graphs são capturados por largura de janela de verificação.
2. **Exatidão como contrato muda o que dá para otimizar.** Com amostragem com chave, verificar um rascunho é só comparar tokens: a resposta com rascunhos é bit-a-bit a serial. Isso habilita, sem risco de mudar respostas, políticas agressivas de rascunho (copy drafts de 15 tokens, parada consciente do ruído), batching de streams, replay de prompts e a escolha de transporte RDMA — e cada otimização vem com um teste objetivo ("o hash da resposta mudou?"). No vLLM, a especulação é "sem perdas" **em distribuição** (rejection sampling), mas o lote, o caminho de kernel e a especulação alteram bits.
3. **Reuso de prompt para modelos com estado recorrente.** O estado KDA não pode ser "cortado" no meio de um bloco; o vLLM só restaura em checkpoints alinhados e arredonda um acerto para baixo até um bloco inteiro. O TensorFold guarda o estado do prompt **no fim do prompt** (ou no último múltiplo de 64) e, num reenvio idêntico, também a **linha de logits** da cabeça: nada é recalculado (64k: 34 s → <0,07 s). **(terceiros)** O jayleaton mediu que, neste modelo, um replay no vLLM recalcula "~8.2k / 5.7k / 11.1k" tokens a 8K/32K/64K.
4. **A rede entre os Sparks é o gargalo de latência do decode.** Num passo de decode de TP=2, cada camada faz all-gathers pequenos (as parciais de uma janela de verificação: linhas × 4.096 fp32). O NCCL gasta a maior parte de cada um em latência fixa (~30–50 µs). Um transporte RDMA de uma única chamada (GPU escreve em memória pinada do host; a CX7 faz RDMA-write no par; o kernel lê no lugar) baixa para 11 µs e dá +6% de decode.
5. **Memória unificada sem surpresas.** O orçamento do TensorFold parte do `MemAvailable` real do host e recusa a configuração **antes** de carregar pesos, informando a maior janela que cabe; o `start.sh` reinicia uma vez com ela. No vLLM a receita teve de fixar `--kv-cache-memory-bytes`, ajustar `gpu_memory_utilization` em centésimos e conviver com crashes por `MemAvailable` zerado.
6. **Ganhos medidos grandes justamente onde importa para agentes:** decode single-stream (+67% em prosa nas medições acima), TTFT sob concorrência, reuso de histórico e o contexto nativo inteiro.

### 1.4 O que se perde (trade-offs explícitos)

- **Escala:** o motor CUDA do GLM exige exatamente dois ranks (TF); nada de TP3/TP4 (a receita vLLM tem ambos).
- **Concorrência máxima 4** (o código segmentado tem `MAX_SEGS = 4`; `PARALLEL` 1–4). Oito prompts longos chegando juntos esperam em fila **(terceiros, LimeChain, sobre outra receita TensorFold:** "8 × 23K: up to 183 s").
- **API menor:** `logprobs`, `top_logprobs` e `n≠1` → HTTP 400 ("this engine returns no token probabilities"); `logit_bias` e penalidades de presença/frequência/repetição aceitos e ignorados; sem `/v1/messages`; sem chave de API (o servidor HTTP e a porta de rendezvous não são autenticados — "keep them on a private link", RUNBOOK do TF).
- **Defaults com perdas** para caber 1M e ganhar velocidade: `DENSE=q4` (lossy), `KV=fp8` (lossy), `KDA_CHUNKED=1` (perto do kernel serial, mas não os mesmos bits). `DENSE=bf16 KV=bf16 KDA_CHUNKED=0` serve o checkpoint como está, com janela menor.
- **Regressão conhecida:** com `q4`, prompts curtos de código em francês perdem o fim de turno e correm até `max_tokens` em 12 de 48 respostas (1 de 48 com `bf16` ou `fp8`) (#18).
- **Fragilidade de manutenção:** 53 patches contra o `site-packages` de **uma** versão do TensorFold; cada release do upstream exige rebase (v1.3 renumerou e removeu três).
- **Licença:** o DFlash2 é CC BY-NC-ND 4.0 (não comercial); `DRAFTER=mtp` evita, mas força `PARALLEL=1`.
- **Ecossistema:** sem métricas Prometheus padrão do vLLM (há `/metrics` próprio), sem LMCache, sem plugins de quantização; o motor só lê os formatos que a família implementa.

### 1.5 Evidência de terceiros (rotulada)

- **(terceiros) jayleaton/glm53-tensorfold-spark**, TensorFold 0.3.4 + 77 patches próprios, checkpoint **abliterado** `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`, mesmo par de Sparks, mesmo cliente para os dois motores; coluna vLLM = kit Reederey87 (fork da receita vLLM da Mia) com os **mesmos pesos**: chat greedy 64 tok 22,8 → 51,6 tok/s (2,26×); code greedy 41,9 → 89,6 (2,14×); structured 72,7 → 112,3 (1,54×); chat sampled (T=1) 27,1 → 48,6. No RigMark (contra o vLLM TP2 k=7 publicado por Alex Ellis, **pesos diferentes**, NVFP4): decode code/prose/structured 72,4/45,7/95,6 contra 44,0/18,9/64,9; **prefill frio ainda atrás** (1.610/1.684/1.667 contra 1.813/1.908/1.922 tok/s a 8K/32K/64K); replay de 64K: TTFT 0,26 s contra 5,77 s.
- **(terceiros) LimeChain bake-off** (2026-09-30, um braço de cada, temp 0): TensorFold (perfil "C8", EXL3 4-bit + DFlash2) contra vLLM + NVFP4: C1 decode 99 contra 37 tok/s; C8 prompts curtos 368 (46/stream) contra 185 (24/stream); acerto de prefixo em follow-ups C8 98,7% contra 76,5%; contexto 1.048.576 contra 262.144; menor `MemAvailable` 12,9/13,2 GiB contra 4,5/6,5 GiB. Limite citado: prompts frescos longos chegando juntos ficam lentos. **(incerto)** O perfil TensorFold usa knobs `GLM53_TF_*` (família do jayleaton, não a receita da Mia), e os números foram lidos de um resumo automático da página (WebFetch), não do repositório clonado.
- **(terceiros) TensorFold issue #167** (2026-09-29/30): dois forks comunitários mais antigos não atingiram o critério pré-registrado de concorrência (C4 ≥ 1,5× e C2 ≥ 1,3× do C1). **(interpretação)** Na receita da Mia, prosa C2/C1 = 79,2/60,4 = 1,31× e C4/C1 = 108,8/60,4 = 1,80× — o lote de verificação, o multi-prefill e o pool compartilhado são exatamente a resposta a esse problema (métodos diferentes; não é a mesma medição).

---

## 2. Arquitetura do TensorFold, explicada conceitualmente

### 2.1 Filosofia: um motor por família, exatidão como contrato

**(TF)** "TensorFold serves language models on Apple Silicon and NVIDIA GPUs through an OpenAI-compatible API. Each model family supplies its own kernels and draft verification." Backends: **MLX** (Mac) e **CUDA** (compute capability ≥ 8.9: Ada, Hopper, Blackwell, incluindo o GB10; NVFP4/FP8 exigem ≥ 9.0). Famílias na v0.6.0: Nemotron 3.5 Lightning, Qwen3.8-27B, Qwen3.8 Flash Next, GLM-5.3-Flash (MLX num Mac de 256 GB; CUDA com dois ranks), Gemma 4, DeepSeek-V4-Flash (MLX), Qwen3.6-35B-A3B, Ternary Bonsai 2.

O contrato "Exact decoding" do TF:
- "A draft is accepted only when it equals the token the same engine would produce serially."
- "Sampling depends on the prompt or explicit seed, absolute position and token ID."
- "Verify kernels keep each row's arithmetic independent of the other rows in the call."
- "Exactness is against the same engine, weights, runtime and settings. It does not imply identical output between MLX and CUDA, different quantizations, or different tensor-parallel rank counts."
- `"draft": false` numa requisição dá a referência serial para comparar.

**(interpretação)** Isso é uma escolha de engenharia radical: em vez de aceitar não-determinismo (ordem de redução que muda com o lote, kernels que escolhem tiles pela forma, rejection sampling probabilístico) e medir "qualidade estatística", o TF exige que **cada linha** de uma janela de verificação ou de um lote tenha os bits que teria sozinha. Daí decorrem quase todas as decisões: kernels "row-independent", split-K fixado pela forma do peso, somas em ordem de rank, amostrador no host em float64, CUDA graphs por largura (o graph não pode mudar a aritmética), estados de prompt guardados em pontos de uma grade (para que prompt retomado = prompt fresco).

O que o upstream v0.6.0 oferece para o GLM no CUDA, **antes** dos patches da Mia: dois ranks; janela default de **2.051 tokens** (onde a atenção DSA ainda é densa) — `--context` maior liga o caminho esparso; "It serves one request at a time"; até 8 prompts guardados (`TF_GLM_CACHE_ENTRIES`) em `TF_GLM_CACHE_GIB` = 3 GiB; DFlash2 opcional (MTP default); EXL3 "experimental". **Os patches da Mia acrescentam**: 4 requisições concorrentes num pool compartilhado, KV FP8, 1M de contexto com 4 streams, multi-prefill, copy drafts, políticas de parada novas, RoCE, divisão das hyper-connections, KDA em forma chunked, pesos densos q4/fp8, L2 prefetch, visão no CUDA, tool calling robusto, `/tokenize`, métricas.

### 2.2 Modelo de processos e ranks: um rank por Spark

- **Comando:** cada Spark roda um container com `tensorfold serve <snapshot> --tp 2 --rank R --master <head-addr> --master-port 29551 …`. O rank 1 (worker) sobe primeiro, por ssh; depois o rank 0 (head), que também abre a API (`--name`, `--host`, `--port`). (repo, `start.sh`)
- **Rendezvous (código, `tensorfold/cuda/comm.py`):** o rank 0 cria um `torch.distributed.TCPStore` em `master:port` (timeout 600 s); gera o `ncclUniqueId` e o publica na store (`tf_nccl_uid`); o rank 1 lê e ambos chamam `ncclCommInitRank`. O NCCL é carregado por **`ctypes`** (`libnccl.so.2`), sem `torch.distributed` process groups; a única coletiva é **`ncclAllGather`** no stream corrente (capturável em CUDA graph). Docstring: "NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal."
- **Comparação de configurações na partida (código):** cada lado empacota suas configurações relevantes (contexto, grade de prompt, copy drafts, split, política multi-stream, `MAX_ROWS`, `WIDE_GRAPHS`, união do núcleo…) numa lista de inteiros, faz all-gather e **recusa subir** se divergirem. Por isso toda variável `TENSORFOLD_*`, `TF_GLM_*`, `TF_ROCE_*` é exportada para **os dois** containers (`start.sh`).
- **Quem decide (código, `multi.py`):** "Rank 0 decides everything that depends on time, clients or placement; rank 1 applies it. Each iteration rank 0 sends ONE message (`GlmEngine._share`): ops ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE [e SLOT], which both ranks apply in order through the same code". A mensagem é um vetor int32 enviado por **dois all-gathers** (o comprimento, depois os valores). O rank 1 deriva os rascunhos sozinho (mesmo prompt, mesmos tokens → mesmas propostas) e só **confere** as profundidades que o rank 0 mandou.
- **Rank ocioso:** "An idle rank 1 waits on the doorbell (`GlmEngine._await_bell`), not in an all-gather." O sino é uma chave `tf_glm_request_<n>` na própria `TCPStore`; assim o rank 1 ocioso não segura GPU nem núcleo de CPU (v0.6.0: "an idle GLM rank no longer spins"; foi patch da Mia promovido a upstream).
- **Falha:** "After an error the two ranks cannot be trusted to agree: `broken` latches and every later call fails until both restart." O `start.sh FOREGROUND=1` derruba um rank se o outro morrer ("a lone rank would wait for its peer forever"). `TF_GLM_MULTI_WATCHDOG_S` (default 300) despeja as pilhas Python de uma iteração travada.
- **Desconexão de cliente:** nos motores de dois ranks, a requisição termina nos dois ranks mesmo se o cliente cair (TF docs).

### 2.3 Tensor parallelism à moda TensorFold

- **(TF, `cuda.md`)** "The default two-rank decode paths gather fp32 partials and add them in rank order. … Rank 0 chooses the window and both ranks execute the same forwards."
- **(interpretação)** Em vez do par Megatron "column-parallel → row-parallel → **all-reduce**", cada rank produz uma **parcial fp32 [R, D]** (R linhas da janela, D = 4.096) após o bloco de atenção e após o bloco MoE/MLP; faz-se um **all-gather** e cada rank soma "rank 0 primeiro" — a mesma soma, na mesma ordem, nos dois ranks, logo bits idênticos. Um all-reduce do NCCL não garante essa ordem.
- **Cabeça (LM head) dividida por vocabulário:** logits `[R, V/world]` por rank. Para amostrar, cada rank manda seus melhores candidatos (top-k + `MARGIN` = 8, ou até 1.024 para o núcleo top-p) num all-gather; a escolha final é feita no host.
- **Cache latente replicado (código, `geometry.mla_geometry`):** o termo do cache é `count × capacity × kv_lora_rank × 2` sem divisão por `world` — cada rank guarda o latente inteiro de cada token (o latente é compartilhado pelas cabeças e pelos ranks, TF docs).
- **Fluxos residuais (hyper-connections) e a "cola":** os dois ranks rodam a cola de hyper-connection (`hc_post`, `hc_pre`) sobre todas as linhas (no decode). No prefill, `SPLIT=1` divide essa cola por linhas (§2.12).

### 2.4 Orçamento de memória: `MemAvailable` menos uma reserva

- **(código, `tensorfold/cuda/capacity.py`)** `reserve_bytes`: "What the startup budget leaves free: max(4 GiB, a tenth of `total`), or `TENSORFOLD_MEMORY_RESERVE_GIB` (>= 2)". `available_bytes`: num GPU **integrado** (GB10, `is_integrated`), o disponível é `MemAvailable` (de `/proc/meminfo`) − reserva — "one pool on a unified GPU: reclaimable page cache is available"; num GPU discreto, o mínimo entre o livre da GPU e o do host.
- **(TF, `api.md`)** No GB10 as alocações do servidor saem da RAM do host "but are not charged to a container's memory limit (`docker run --memory`, cgroup `memory.max`)"; "exhausting a unified GPU's memory can freeze the host, so lower it only with room to spare". A reserva também carrega "CUDA context, NCCL and workspace memory the estimate does not count" (o doc do GLM mediu 3,5–4,8 GiB disso).
- **(repo)** A receita usa **`MEMORY_RESERVE_GIB=14.5`** (default do TF: um décimo da RAM, ~12,2 GiB) e **`KV_POOL_GIB=12.5`** (o `TF_GLM_CACHE_GIB`, default 3). "The server uses about 10 GiB beyond its own estimate at its peak (a 1M-token prompt), so the reserve also sets the lowest free memory": 14,5 deixa ~4,5–5 GiB livres no head no pior momento.
- **Recusa antes de carregar (repo):** "On the Spark's unified memory, running out tends to freeze the machine rather than fail an allocation. A setting that does not fit is refused before any weights load, with the largest window that fits; `start.sh` then restarts once with that window". O `start.sh` lê do log a linha `largest fitting prompt-plus-reply window: N tokens` e repete o boot com `--context N`.
- **Números (repo):** estimativa de partida do rank 0 **88,09 GiB**; `MemAvailable` ocioso 7,0 GiB (rank 0) / 10,5 GiB (rank 1); mínimo sob um prompt de 1M: 5,6 / 9,5 GiB. O `start.sh` avisa se houver menos de **~110 GiB** livres em qualquer Spark (`free -g`, coluna available).
- **Comparação conceitual (interpretação):** no vLLM, `gpu_memory_utilization × memória total` define um teto **fixo** e um profile run mede o resto; em UMA isso ignora o page cache e outros processos. O TF parte do que o SO diz que está disponível **agora** e mantém uma folga explícita.

### 2.5 O pool compartilhado de caches por token

**(código, `pool.py`, `multi.py`)**

- **Arena e planos:** "An `Arena` holds planes of device rows (per DSA layer its latents; per indexed layer its index keys, gates and pooled keys; the MTP layer's)". Cada plano tem `rows // div + pad` linhas: latentes `div 1`; chaves agrupadas (um pool = 4 tokens) `div 4` com 2 linhas de pad ("the token selection reads up to 2 pools past a window").
- **Extents:** um stream (ou um prompt guardado) possui um *extent* `[base, base + size)`, ambos múltiplos de **`ALIGN = 2048`** tokens. O código de stream único (prefill, janelas de decode, snapshots) roda sobre **views** desse extent "exactly as on tensors of its own".
- **Slots:** além do pool por token, cada stream ocupa um **slot** (`forward.Slots`) com o que não cresce com o contexto: estados recorrentes KDA `[count, 2, layers, H, 128, 128]` (duas "paridades"), janelas de convolução, scratch da janela, contexto DFlash2; no host, índice de copy drafts, amostrador e gramática.
- **Contabilidade (host, determinística):** `Pool` faz "lowest-base first fit, growth in place, placement for relocations"; como as mesmas chamadas acontecem na mesma ordem nos dois ranks, os pools ficam iguais sem trocar mapas.
- **Admissão (`fits`):** um stream entra se as linhas livres + extents que só guardam prompts cobrirem `align_up(prompt + MAX_ROWS)` **mais um bloco de folga (2.048) para cada stream decodificando**. Caso contrário espera (`NoRoom`), em ordem. Um prompt maior que o pool inteiro falha.
- **Abrir espaço (`_room`, `_grow`):** despejar prompts guardados (LRU, nunca os protegidos), **compactar** (deslizar extents para baixo com `MOVE`, copiando só as linhas usadas) e crescer no lugar (`GROW`) ou mover.
- **Retomar de um prompt guardado:** se o extent guardado não tem dono, o stream o **assume no lugar** (despejando os prompts guardados que ele sobrescreveria); se outro stream escreve nele, **copia** as linhas para um extent novo.
- **Política do README:** "Any one request can grow to the full window, and the four together share the pool: e.g. one 1M-token conversation next to one more of 1M, or next to three of ~640k. A request the pool cannot place yet waits until others finish (kept prompt states give way first)". `/health` mostra `pool_tokens`, `pool_free_tokens` e os streams decodificando, preenchendo e pausados.

**(interpretação) PagedAttention × extents.** O vLLM pagina o KV em blocos pequenos com uma tabela de blocos por sequência (fragmentação quase zero, mas kernels que fazem gather por tabela e, em modelos híbridos, páginas enormes — 3.584 tokens no GLM — para alinhar estado recorrente). O TF dá a cada sequência um **intervalo contíguo** de linhas e paga a fragmentação com compactação ocasional; ganha kernels de stream único que leem linhas contíguas e uma tabela pequena por segmento no lote (`SegRows`: `pos`, `base`, `seg`, `sparse` por linha; `seg_start`, `seg_rows`, `seg_pbase`, `seg_pools` por segmento).

**Estimativa de bytes por token (interpretação, a partir do código):** linha FP8 do latente = 512 + 16 = 528 B; chaves agrupadas do indexador em FP8 = 128 + 16 = 144 B por pool de 4 tokens = 36 B/token; total ≈ 564 B/token/camada DSA (em bf16: 1.024 + 64 = 1.088 B — "half of bf16's bytes"). Com 11 camadas DSA (e sem a camada MTP, que `TF_GLM_MTP=auto` não carrega ao lado do DFlash2): ≈ 6,2 KB/token por rank; 1.048.576 tokens ≈ 6,1 GiB por rank. As chaves/gates por token do indexador vivem em anéis bf16 por slot (pequenos). **(incerto)** Não consegui reconciliar exatamente 2.922.496 tokens com 88,09 GiB de estimativa + 12,5 GiB de teto; o README só afirma o resultado medido.

### 2.6 O formato do KV FP8 (`kv8`): e4m3 + escala potência de dois

**(código, docstring de `kv8.py`, patch 0038)**
- Cada linha (o latente de 512 de um token numa camada; ou a chave agrupada de 128 de um pool) vira **512 códigos e4m3 + escala fp32 + 12 bytes zero** (linhas alinhadas a 16 B). Um único tensor `uint8 [rows, width + 16]` mantém o tratamento genérico de linhas (slices, clones, snapshots).
- **Escala** = `2^(ceil(log2 amax) − SHIFT)`, `SHIFT = 8`: o maior valor da linha cai em [128, 256], abaixo do máximo 448 do e4m3 → **nada satura**. Expoente polarizado preso em 1..254.
- **Por que potência de dois:** "a floating-point format's relative precision does not depend on the scale, so a per-row power of two loses nothing to a finer or fp32 scale but the values it pushes below e4m3's normal range (2^-6), under 2^-14 of the row's largest". Medido nas linhas reais `kv_a`/`kv_a_layernorm` do GLM (amax ~3–5× o rms): escalas fp32 por token, por bloco de 128 e fixas por camada ficam todas a 0,001 do **erro relativo de 2,65% por linha** desta.
- **Exatidão:** "Lossy (the owner's call), never inexact": os bytes de uma linha são função **só** da linha bf16 (um helper, `store_row`, quantiza prefill e decode); os leitores convertem códigos para bf16 (exato) e multiplicam a escala nos produtos fp32 (exato para potência de dois). Logo `drafted == serial` e `resumed == fresh` continuam valendo.
- Comparação (interpretação): o DeepSeek-V3.2 guarda chaves FP8 do indexador com escalas ue8m0 por token — a mesma ideia (o próprio docstring cita).

### 2.7 Atenção: cache latente NoPE, indexador esparso, pools visíveis

- **(TF, recipe GLM)** As camadas DSA são **MLA NoPE**: a chave da cabeça h é `Wk_h c` e o valor `Wv_h c`, com `c` o latente normalizado de 512. Então `score_h = (Wk_h^T q_h) · c` e `out_h = Wv_h (Σ_j p_j c_j)`. O motor guarda **só `c`** ("1 KB a token and layer" em bf16), **absorve** a query uma vez por linha, atende sobre latentes e **expande** o latente atendido uma vez (`TF_GLM_LATENT=0` volta a K/V por cabeça).
- **Limite denso de 2.051 tokens:** "Past 2,051 tokens a row attends to its top 512 index pools (2,048 tokens) plus its incomplete pool" (512 × 4 = 2.048, + até 3 do pool incompleto). Os pools são escolhidos por **um kernel de radix select** (8 bits por passada) sobre os pools que o chunk enxerga, arredondados para uma potência de dois (os "pool buckets" dos graphs).
- **Pools visíveis (patch 0043, a partir do PR #140 do TF):** uma linha de decode na posição `P + r` só precisa pontuar `min(NPS, max((P + r + 1) // 4, K))` colunas — os pools completos que ela vê; o resto do bucket nem é pontuado nem varrido. "the same tokens".
- **Prefill em blocos de 512 linhas (patch 0024):** o score fp32 dos pools ocupa `SELECT_ROWS = 512` linhas por vez, não o chunk inteiro (menos memória a 1M).
- **Atenção esparsa "one-pass" (`TF_GLM_SPARSE_ONEPASS`)**: a atenção latente esparsa dos chunks de prompt "keeps no partials"; a atenção densa do prompt roda em blocos de `MLA_PROMPT_ATT_ROWS = 512` linhas, então suas parciais fp32 ocupam 512 linhas e não o chunk inteiro (patch 0028, "smaller prompt scratch" — menos memória a 1M).

### 2.8 Estados de prompt guardados, grade de prompt, prefixo compartilhado e replay

**O problema (interpretação):** num modelo com KDA, o "estado do prompt" não é só KV por token: inclui o **estado recorrente** de cada camada linear (matrizes 128×128 por cabeça), as janelas de convolução, o contexto do drafter e os anéis do indexador. Esse estado só existe **num ponto** do prompt; não dá para fatiar.

**A solução TF (código, patches 0008, 0015, 0042; TF upstream):**
- **Kept prompt state / snapshot:** após o prefill, o motor guarda o estado no ponto do prompt (KDA, conv, linhas pendentes do MTP, janela do DFlash2, cauda dos anéis do indexador e, se preciso, linhas salvas). Uma requisição nova cujo prompt **estende** um guardado "token for token" retoma dali.
- **Grade (`TF_GLM_PROMPT_GRID`, valores 16/32/64/128):** "every prompt chunk starts on a multiple of G and a kept prompt state sits at the last multiple of G in the prompt (`grid_point`), so prompt kernels whose rows depend on G-row block alignment resume exactly." Com `KDA_CHUNKED=1` a grade é **64** (o kernel chunked trabalha em sub-chunks em posições absolutas). Se o prompt não termina na grade: "the request then prefills up to B, keeps that state, and resumes from it to n" — então qualquer prompt que estende `prompt[:B]` passa pelo **mesmo estado**, fresco ou retomado.
- **Prefixo compartilhado (`SHARED_PREFIX=1`, patch 0015):** uma requisição com rascunho e só texto também guarda estados (i) onde seu prompt **deixa de compartilhar** com um prompt guardado e (ii) **no fim do bloco de sistema** (primeira ocorrência de `<|user|>`), cada um na grade, se acrescentar ≥ 256 tokens (`SHARED_DEFAULT`), no máximo 2 por requisição (`SHARED_MOST`). Estados salvos levam a janela do DFlash2 junto. Resultado: conversa nova com o mesmo system prompt de 7,9k: **4,24 s → 0,13 s**.
- **Replay (patch 0042):** o estado do **fim** de um prompt guarda também "the head's logits row at ids' end … before any grammar mask". Um prompt **idêntico** retoma o estado e amostra o primeiro token dessa linha: "prefilling nothing". Resultado: 64k idêntico **34 s → <0,07 s**.
- **Quantidade e custo:** `TF_GLM_CACHE_ENTRIES` (TF default 8; receita **32**): "the oldest is dropped past this count, however much of the pool is free"; cada entrada reserva ~**45 MiB** fixos na partida (32 ≈ 1 GiB a mais que 8). "An agent request keeps 1 to 3 (its own plus shared-prefix states)", então 8 deixava 3–4 conversas alternadas se expulsarem (#17).
- **Despejo:** prompts guardados cedem primeiro (LRU) quando um stream vivo precisa de espaço.
- **Não reusam:** prompts com imagens e requisições com gramática não fazem replay; o "lone stream" (`TF_GLM_MULTI_LONE=1`) movia o stream para as primeiras linhas do pool e **despejava** os prompts de outras conversas — medido com 3 conversas alternando (~4,5k tokens, 6 turnos quentes): 0% dos prompts do cache com `1`, **98,1%** com `0` (v1.3.1, #12, #13).

**Contraste com o APC do vLLM (interpretação):** APC = hash encadeado de blocos cheios; reuso em qualquer prefixo **alinhado a bloco**; em híbridos, estado recorrente só em checkpoints (GLM: 3.584 tokens) e o acerto arredonda para baixo; a cauda é recalculada. O TF guarda **poucos estados inteiros** em pontos escolhidos (fim do prompt, fim do system, ponto de divergência) — reuso exato e sem recálculo para os padrões de agentes (reenviar a conversa + anexar), mas sem reuso arbitrário no meio.

### 2.9 Amostragem com chave (keyed sampling): por que rascunho = serial

**(código, `tensorfold/engine/exact_sampling.py`)** — "Key Gumbel draws by seed, position and token id so verification matches serial top-k/top-p/min-p sampling."
- **Semente:** `seed` da requisição, ou `seed_for(tokens)` = SHA-256 dos ids do prompt (8 bytes, 63 bits): "the same conversation samples the same reply" (README: "Without a `seed`, the sampler's key comes from the prompt, so the same request gives the same reply").
- **Uniforme com chave:** `uniform(seed, position, ids)` = três rodadas de mistura **splitmix64** de (semente, posição absoluta, id do token) → double em (0,1).
- **Escolha:** ordenar candidatos (valor desc., id asc.), cortar top-k, dividir por T, cortar top-p (soma cumulativa), cortar min-p (após top-p, "as SGLang does (vLLM applies `min_p` first)"), somar ruído **Gumbel** `-log(-log(u))` e tomar o **argmax** (truque Gumbel-max). Tudo em **float64 no host** (numpy).
- **Consequência:** o token amostrado na posição p é uma **função determinística** dos logits e da chave. Verificar um rascunho = calcular os logits da posição, amostrar com a chave e **comparar**. Aceita-se até o primeiro desacordo e o token amostrado ali (o "bônus"). Logo a saída é idêntica à serial **qualquer** que seja o rascunhador ou a política de parada: "Drafts only propose: every drafted token is checked against the model's own keyed sample, so drafted replies equal TensorFold's serial, one-token-at-a-time decoding" (README).
- **Núcleo distribuído:** com top-k desligado (o default do checkpoint é temperatura 1,0, top_p 0,95), cada GPU lê seus 1.024 melhores candidatos (`NUCLEUS`) e, se o núcleo de uma linha passa disso, o shard inteiro; a massa de cada token é cortada em **ponto fixo de 40 bits** (`MASS = 2^40`) "so shard sums are exact, so ranks agree bit for bit" e uma GPU = duas GPUs. `TENSORFOLD_NUCLEUS_UNION=1` (patch 0034, opt-in) testa a cobertura com os candidatos dos dois ranks juntos (mesmo sorteio, menos leituras de shard inteiro).
- **Bônus para o rascunhador:** como o ruído do alvo é **conhecido antes da verificação**, o DFlash2 pode usá-lo para prever a amostra (§2.10).
- **(interpretação) Contraste:** a especulação do vLLM usa rejection sampling (aceita com min(1, p/q), reamostra do resíduo): preserva a **distribuição**, não a sequência. A amostragem com chave dá reprodutibilidade de sequência (mesmo prompt → mesma resposta), habilita testes de hash e torna otimizações verificáveis, ao custo de exigir aritmética por linha invariante (kernels mais restritos).

**O que NÃO é exato na receita (repo):** "Three defaults are not exact against the checkpoint in bf16, for speed and the 1M window: `DENSE=q4` and `KV=fp8` are lossy …, and `KDA_CHUNKED=1` is close to the serial kernel but not its bits." Mesmo assim, rascunhado = serial e concorrente = sozinho continuam valendo (mesmo motor, mesmas configurações). `COMM=roce`, `SPLIT=1`, `TF_GLM_L2PF` e `TF_GLM_EXL3_LOADS` "move the same bits".

### 2.10 Rodadas de decode, janelas de verificação e CUDA graphs

- **Rodada (round):** proposta de rascunhos → **uma** forward de verificação sobre `[token pendente, rascunhos…]` (a "janela") → amostragem com chave de cada linha → aceite até o primeiro desacordo → commit das linhas aceitas (KV, estado KDA, conv, contexto do drafter).
- **Larguras:** `MAX_ROWS = 16` por stream (o pendente + até 15 rascunhos, para copy drafts longos; patch 0013, inspirado no PR #115 do TF); `TF_GLM_MAX_ROWS` aceita 16–64 em passos de 8. No lote multi-stream, a janela combinada tem no máximo **`MAX_WINDOW = 32`** linhas; se passar, "the longest tails trimmed" (um rascunho por vez da janela mais longa).
- **CUDA graphs por largura:** single-stream: `GRAPH_ROWS = (1..8)` capturadas; com `COPY_CODE=1`, também a largura **16** (`TF_GLM_WIDE_GRAPHS=16`; copy drafts são "acolchoados" repetindo o último rascunho até a largura capturada mais próxima). Multi-stream (`BatchedVerify.capture`): um graph para **cada contagem de linhas 1..32**; posições, extents, slots e paridades vivem em **tabelas no device**, então "one graph a row count covers every mix". Graphs também para o bloco do DFlash2 e as atualizações de contexto. Passado o limite denso, "one per pool bucket". `TF_GLM_MULTI_GRAPHS=0` roda eager com os mesmos bits.
- **Rollback do KDA (código, `kda.py`, `verify.py`):** o estado recorrente tem duas **paridades**; a cadeia da janela escreve na outra paridade; no commit, se o segmento mantém menos linhas que verificou, "its KDA states replayed to them" — recalcula-se o estado até a última linha aceita.
- **Sobreposição host/GPU (patch 0013):** `_Late` entrega os tokens de uma rodada ao cliente **uma rodada depois**, quando a próxima janela já foi enfileirada; `_Clock` mede o tempo de GPU da janela com eventos CUDA sem sincronização extra.
- **Tokens de calibração (patch 0048):** as medições de partida das janelas usam ids **distintos** — "rows of real text route to many experts" (um mesmo token repetido roteia para os mesmos experts e mede uma janela de 16 linhas pela metade do custo real).
- **Custo de uma janela (código, `draft_sim.py`, medido na época do patch 0011, motor anterior às otimizações posteriores):** `VERIFY_MS = (29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0)` ms para 1..8 linhas; bloco DFlash2 `BLOCK_MS = 3.08` ms. **(interpretação)** Verificar 8 linhas custa ~2,3× uma: é a margem que a especulação explora num decode limitado por banda.

### 2.11 Rascunhos: DFlash2, copy drafts e MTP

**DFlash2 (TF + patches):**
- Drafter de blocos (porta da arquitetura de `z-lab/dflash`) treinado pela IncoAI; recebe os **"taps"** — estados ocultos de camadas-alvo (`target_layer_ids`) do próprio alvo — e propõe um bloco de candidatos; um **seletor** escolhe a cadeia: `EDGE = 0.6` (peso das arestas do seletor) e `NOISE = 0.7` (peso do ruído com chave do alvo) (código, `dflash2.py`).
- **Janela deslizante (TF v0.6.0):** o DFlash2 atende só sua janela de **2.048** linhas; o contexto fica num **anel** de 2.176 linhas (janela + bloco + um tile), **21 MiB por rank** "whatever the window" (antes: 10 KiB por rank **por token** da janela); um prompt guardado leva uma cópia da janela (20 MiB/rank). "8% faster at 52k, and 3.8 GiB lighter a rank on two Sparks". O modelo de rascunho ocupa **0,63 GiB/rank** (cópias 4-bit + codebooks do seletor). "Draft steps use the quantized head; verification keeps the original head" (o rascunho usa uma cópia 4-bit da cabeça; a verificação, a cabeça original).
- **Políticas (código, `engine.encode_policy`):** `fN` (N rascunhos fixos), `fcN:P` (até N enquanto o **produto** das confianças da cadeia ≥ P; o antigo default era `fc5:0.3`), `fa:LOW:HIGH` (adaptativo), **`fncN:P[:B]`** (o mesmo produto, mas com **confianças conscientes do ruído**: `softmax((score + g)/B)` no pick, onde `g` são as próprias **extrações Gumbel com chave do alvo**), **`fcostN:noisy[:B]`** (a profundidade k ≤ N que maximiza `E[tokens | k] / ms(k)` pela tabela de custos medida na partida).
- **β = 2,0:** "fitted by log loss … on the 9 teacher-forced dumps of 2026-09-30 (7 sampled prose, 2 greedy code): 2.0 on all of them" (uma grade mais fina deu 1,8 com −0,2% de log loss).
- **Default da receita `DRAFT_POLICY=fnc7:0.3`:** até 7 rascunhos; para quando o produto das chances "under the request's own sampling noise drops below 0.3". Medido: prosa 47,2 → 50,6 tok/s, código 51,7 → 55,5 sobre `fc5:0.3` (dois boots cada). "replies are the same under every policy".
- **Simulador offline (patch 0011):** `TF_GLM_DRAFT_DUMP=<dir>` faz a requisição decodificar **um token por rodada** (resposta = serial) e grava, para cada posição, o que o bloco DFlash2 teria proposto (candidatos, logits, arestas do seletor, as extrações Gumbel com chave), e o que o alvo cometeu. Como a amostragem é com chave por (semente, posição, token), "a policy's rounds are therefore replayed exactly" no CPU (`draft_sim.py`, numpy), com uma tabela de custos → tok/s por política. Inclui um "oracle" (profundidades ótimas em retrospecto) como teto. **(interpretação)** É a exatidão virando ferramenta de pesquisa: dá para tunar políticas sem GPU.

**Copy drafts (prompt lookup, patches 0007, 0032):**
- "when the context's last tokens occurred before, propose what followed them": os últimos `MATCH = 8` tokens (incluindo o pendente) são procurados no prompt e na resposta; os tokens após a ocorrência mais recente viram rascunhos, **verificados antes** dos do DFlash2. Ambos os ranks têm o mesmo prompt e tokens: mesmas propostas sem troca. Puro numpy.
- `COPY_MAX=15`: "lets a long quote go through in one round" — respostas de edição 84 → 119 tok/s.
- `COPY_CODE=1`: graphs de 16 linhas + cópias **da própria resposta** só após casar 16 tokens (`TF_GLM_COPY_REPLY_MATCH=16`), evitando cópias espúrias de código repetitivo: código 55,4 → 56,2, edição 120 → 125.
- `COPY=1`: respostas de citação/edição 80,3 → 84,4 tok/s; prosa e código inalterados.

**MTP:** a cabeça MTP do checkpoint (1,77 GiB por Spark). `TF_GLM_MTP=auto` (receita) a deixa de fora enquanto o DFlash2 rascunha todas as requisições; `1` (default do TF v0.6.0) a carrega com `PARALLEL=1` (permite a política `auto`, que compara drafters em requisições greedy e alterna). `DRAFTER=mtp` usa só MTP (sem a licença do DFlash2), **uma requisição por vez**. No modo multi-stream, políticas MTP são remapeadas para as gêmeas DFlash2 ("drafts never change a reply").

### 2.12 Concorrência: o scheduler do GLM multi-stream

**(código, `multi.py`, `multi_tune.py`, `multi_prefill.py`)**
- **Iteração (uma thread no rank 0):** admissões; depois **ou** um chunk de prompt do stream mais antigo preenchendo (FIFO; no máx. `TF_GLM_FILL_ROWS` = 1.024 linhas quando outros decodificam; o chunk inteiro de 2.048 quando sozinho) **ou** uma rodada de decode — alternando por `TF_GLM_FILL_SHARE` (default 0,5: chunk, rodada, chunk…; 1 = preencher tudo antes de decodificar). "a prompt chunk never shares a forward with decode rows". **(interpretação)** É o oposto do chunked prefill misto do vLLM (que mistura prefill e decode no mesmo passo e, nesse modelo, sai do CUDA graph FULL).
- **Rodada multi-stream:** cada stream decodificando propõe sob sua política (copy drafts primeiro); a janela combinada ≤ 32 linhas; **uma** verificação sobre as linhas de todos (`BatchedVerify`: partes independentes de linha — embeddings, cola HC, projeções, MLP/MoE, cabeça — no lote inteiro; KDA pela cadeia segmentada, cada segmento a partir do estado do seu slot; DSA pela atenção latente/esparsa segmentada sobre os extents de cada stream); **um** all-gather empacotado para os amostradores; cada stream aceita e comete o seu. `SerialVerify` (cada janela como sua própria forward) é a referência (`TF_GLM_MULTI_VERIFY`).
- **Multi-prefill (`MULTI_PREFILL=1`, patch 0049):** prompts que esperam juntos entram num **mesmo chunk** `[s1 rows][s2 rows]…`: o trabalho independente de linha (incluindo os experts roteados, "one expert read for every prompt") roda uma vez; o que toca o estado de cada stream roda peça por peça "through the very calls a chunk of its own makes". Motivo: "A forward's cost is mostly fixed at short lengths (~0.12 s on two Sparks, the weights read once a layer whatever the rows)". Medido (sparkDash): prosa ×4 103,4 → 108,8 tok/s e TTFT **590 → 340 ms**; structured ×3 175,2 → 196,3, ×4 196,3 → 227,9; uma requisição inalterada. Imagens e replays não são agrupados.
- **Profundidade sob concorrência (`TF_GLM_MULTI_DEPTH`):** `policy` (default: cada stream com sua política, como sozinho); `joint` (aloca as linhas da rodada entre streams por tokens esperados por ms, numa curva medida na partida + `TF_GLM_MULTI_OVERHEAD_MS` = 12); `scale:A` (eleva o limiar de parada para `min(0.95, p·S^A)` com S streams).
- **Outros knobs (todos "none of which changes a reply's bits"):** `TF_GLM_MULTI_SAMPLER` (`streams`/`packed`), `TF_GLM_MULTI_ASYNC` (rank 0 envia a mensagem sem esperar a GPU), `TF_GLM_MULTI_PROFILE=N` (log de decomposição de tempo a cada N rodadas), `TF_GLM_MULTI_LONE` (stream sozinho nos graphs de stream único; +0,6–0,9%, mas despeja prompts guardados — receita: 0).
- **Prioridade:** `priority: "background"` serve depois dos outros e cede; ao voltar, **reproduz a partir do prompt** (os tokens já enviados são conferidos, não reenviados) — sua resposta = execução solo (TF docs).
- **Deadlocks corrigidos (patch 0040):** a escolha RoCE × NCCL de um gather passou a depender **só do número de bytes** (igual nos dois ranks); antes dependia do alinhamento do endereço e um rank podia esperar no RoCE enquanto o outro esperava no NCCL "forever".

### 2.13 Prefill: chunks, kernels de prompt, KDA chunked e divisão das hyper-connections

- **Chunks de 2.048 linhas** (`PREFILL_ROWS`) para o GLM; ativações bf16 ("EXL3 prompts never took FP8 activations"); um chunk de prompt pode ser de vários prompts (multi-prefill).
- **Experts EXL3 no prompt (patches 0001, 0004, 0009, 0020):** os blocos n de um item são lançados adjacentes ("NFIRST"), então leem as linhas do item "while L2 still holds them: the same kernel and arithmetic, 2.3-2.5x faster at 4,096 rows"; a ordem de lançamento intercala itens de experts com um só item (lidos da DRAM, poucos mma) e com vários (pesos compartilhados via L2, muitos mma) "so DRAM and the tensor cores stay busy together instead of in turns" (`TF_GLM_EXL3_ORDER`); decodificação da treliça com menos instruções inteiras (PRMT). Mesmos bits.
- **KDA em forma chunked (WY/UT) (patches 0012, 0014, 0039; `KDA_CHUNKED=1`):** em vez da regra delta linha a linha, sub-chunks em posições absolutas, com a parte intra-chunk como matmuls e o estado 128×128 carregado de sub-chunk em sub-chunk. Fórmulas (docstring, sub-chunk com estado de entrada S0, G = soma cumulativa do log-decay):
  - `L[s,j] = β_s Σ_c k_s[c] k_j[c] e^(G_s[c] − G_j[c])` (j < s), `M[s,j] = Σ_c q_s[c] k_j[c] e^(G_s[c] − G_j[c])` (j ≤ s)
  - `T = (I + L)^-1` (triangular inferior unitária, por substituição progressiva em blocos de 16)
  - `W = T (β k e^G)`, `U = T (β v)` (a transformação WY/UT); `V' = U − W S0^T`
  - `O = (q e^G) S0^T + M V'`; `S_end = S0 diag(e^G_last) + V'^T (k e^(G_last − G))`
  - Truque numérico da versão Triton: fatores de linha e coluna em torno de um ponto de referência por bloco de 16 linhas para que nenhum expoente passe de e^75 (< e^88,7 do fp32); matmuls em `tf32x3` (~2e-6 relativo a float64; `tf32` puro ~5e-4, "half a bf16 step").
  - Versão CUDA final (0039): **um bloco de 256 threads por cabeça**, sub-chunks de **32 linhas**, estado em registradores, somas fp32 com FMAs explícitos em ordem fixa, compilado como `kda.cu` (sem contração FMA, `expf` do CUDA, divisão IEEE) — determinístico; os bits de uma linha dependem só do seu sub-chunk e do estado de entrada, então são iguais em qualquer fatiamento de prompt **se** os chunks começam em múltiplos de 32 (daí a grade de 64).
  - GB10: "gives a block 99 KiB of shared memory (101,376 B), and pipelined loads … need up to 154 KiB" → um estágio de pipeline.
  - "Close to the serial kernel, not its bits"; o decode continua com o kernel serial. Medido: prefill de 50k 29,3 → 26,4 s; 149k 91,1 → 84,5 s (um boot cada).
- **Divisão das hyper-connections (`SPLIT=1`, patches 0010, 0033, 0017, 0022):** sem divisão, em cada ponto de troca (após atenção e após MLP/MoE) os dois ranks fazem all-gather das parciais fp32 [R, D] e **ambos** rodam `hc_post` e o próximo `hc_pre` sobre todas as R linhas. Com divisão, o rank r **possui** as linhas `[r·H, r·H + H)` (H = R/2; chunk ímpar ganha uma linha zero): envia ao par só a parcial das linhas do par (send/recv NCCL, `TF_GLM_HC_EXCHANGE=p2p`), soma as duas parciais das suas linhas "rank 0's first" (mesmo kernel), roda a cola nas suas linhas e troca as linhas normalizadas (bf16). "Every glue kernel is row-independent and the partial sum order is unchanged, so every bit is the unsplit path's". `TF_GLM_PREFILL_OVERLAP=2` faz as trocas num segundo stream CUDA em pedaços (`TF_GLM_OVERLAP_PIECES=2`) e roda a "frente" do próximo bloco (projeções de entrada KDA/DSA, expert compartilhado) nas linhas próprias durante as trocas. Chunks < 64 linhas (`TF_GLM_HC_SPLIT_MIN_ROWS`) não dividem. Medido: prefill de 50k ~1.270 → ~1.730 tok/s (com 0009 e 0020); 149k 92,8 → 90,7 s com o overlap 2; "decode rounds pay ~1.5%". **(interpretação)** É uma forma de *sequence parallelism* aplicada só à cola elementwise.

### 2.14 Comunicação: RDMA one-shot sobre RoCE (`COMM=roce`)

**(código, patch 0006; b12x `docs/rocenante.md`)**
- **Por que funciona no GB10:** "The GB10 is an integrated GPU with unified memory: pinned host memory is directly addressable by GPU kernels at full bandwidth, and the NIC can register it with a plain `ibv_reg_mr`." Não há GPUDirect RDMA (dmabuf/`nvidia_peermem`) no stack do DGX Spark; "NCCL therefore stays host-staged there".
- **Protocolo (uma chamada de kernel por coletiva):** (1) cada bloco copia sua parte do shard local para `send[seq & 1]` (região pinada); (2) o último bloco publica `nbytes` e toca a campainha (`ctrl.seq`) após um fence de sistema; uma **thread proxy em C** (`roce_proxy.c`, libibverbs, do b12x "used as is") faz RDMA-write do payload para cada par, dividido em **faixas por HCA**, cada faixa seguida de uma escrita de 4 bytes do `seq` no mesmo QP confiável (a flag não fica visível antes do dado); (3) uma thread por (par, HCA) espera a flag com loads acquire de escopo de sistema (com limite: estourar "envenena" o runtime); (4) cada bloco copia os shards **em ordem de rank** (o próprio da entrada, os dos pares dos slots escritos pela NIC); (5) o último bloco avança uma **época no device**, para que CUDA graphs possam reproduzir a troca. Dois slots bastam (um par nunca está mais de uma operação à frente).
- **Tamanhos:** "A decode round's all-gathers are small (a verify window's partials: rows x 4,096 fp32)" = 16 KiB por linha. `TF_ROCE_MAX_KB` (TF 256; receita **512**) define o maior gather por RoCE: 512 KiB = 32 linhas × 16 KiB = a janela de lote inteira de 4 streams (+1,4% de código a 4 streams, +0,5–1% prosa). O resto vai por NCCL, "with the same output layout (rank order), the same bits".
- **Números:** "11 us a 16 KiB gather against NCCL's 45, decode +6%" (prosa 44,2 → 46,9; código 48,7 → 51,6).
- **Dois trilhos:** "Each Spark's single cabled QSFP port is exposed as two PCIe Gen5 x4 functions" (b12x); o proxy divide cada payload entre as duas funções; o `nodes.sh` passa ao NCCL as duas HCAs quando ambas estão endereçadas na sub-rede do link: "a prompt chunk's all-gather is ~1.8x faster on two rails" (32 MB: 1,9 ms em dois trilhos, 3,5 ms em um).
- **NCCL afinado (nodes.sh):** `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` (os trilhos), `NCCL_IB_GID_INDEX` (a entrada **RoCE v2** que contém o IPv4 do nó — "an all-zero or v1 entry makes NCCL fail about a minute in"), `NCCL_MIN/MAX_NCHANNELS=4`. Medido: "a decode-sized all-gather 32 us (80 with NCCL's defaults)"; "NCCL's other knobs (protocols, buffer sizes, QPs, NCCL_NET=IB and the like) were no better, and some of them kept NCCL on one rail."
- **Partida (patch 0052, #6):** na primeira partida cada rank compila suas extensões CUDA sozinho e eles podem "drift tens of seconds apart", além do que uma espera RoCE permite. Até o motor estar construído, cada gather RoCE eager espera o par numa **barreira NCCL** antes; o kernel RoCE é compilado no setup; `TF_ROCE_WAIT_S` (default 20 s, 1–3.600) = quanto uma espera GPU pelo shard do par pode durar antes de envenenar o runtime; falhas imprimem os contadores do proxy. "Captured gathers never do."
- **Ociosidade:** o proxy gira enquanto há operações (as lacunas entre coletivas num graph são submilissegundo) e, após `ROCE_IDLE_SPINS` sem campainha, pede `nanosleep` curto — "an idle server holds no GPU or CPU core".

### 2.15 Decode: pesos densos 4-bit, L2 prefetch e loads da treliça

- **`DENSE` (patches 0002, 0005):** os pesos BF16 do checkpoint fora dos experts roteados (atenção, experts compartilhados, camadas densas, cabeça: ~9,7 GiB por Spark). `fp8`: e4m3 com escala fp32 por linha e bloco de 128 colunas (absmax/448), quantizado uma vez na carga, ativações bf16, mesma estrutura de kernel (linha independente) — ~+33% de decode sobre bf16. `q4` (default): "affine 4-bit groups of 64, each group's range the one of `grid` clippings (1.0 down to 0.55 of its min / max) with the least squared error, scale and bias as stored (bf16)"; a cabeça fica em FP8 e `kv_b` em BF16 (`TF_GLM_KVB`). Medido q4 sobre fp8 (DFlash2, um boot cada): prosa 38,9 → 44,4 tok/s, código 44,2 → 48,7, prefill ~1.090 → ~1.260 tok/s; GSM8K 98,0% e HumanEval 95,1% nos dois (bf16: 97,2 / 96,3). Um 4-bit **calibrado** para os densos "is being worked on" (v1.3.2).
- **L2 prefetch (`TF_GLM_L2PF=1`, patch 0046, adaptado do patch 0460 do jayleaton):** "A decode window's layer alternates DRAM-bound matmuls … with phases that leave DRAM idle: the two all-gathers a layer (the RoCE wait), the hyper-connection glue, the router and top-k, the KDA recurrence and the latent attention core." Um kernel pequeno num **stream lateral** pede (`cp.async.bulk.prefetch.L2`; ou `prefetch.global.L2::evict_last`; ou `ld.global.cg`) os pesos que o stream principal lerá a seguir. Pontos (`TF_GLM_L2PF_SITES=afo`): `a` no all-gather da atenção (pesos HC/norma da FFN, router, expert compartilhado — que passa a rodar **antes** dos roteados, para o fluxo de ~100+ MB dos roteados não expulsá-lo); `f` no all-gather da FFN (pesos da próxima camada; após a última, norma final e cabeça); `o` após as projeções de entrada da atenção (a projeção de saída). Até 8 MiB por ponto (`TF_GLM_L2PF_MB`); janelas > 64 linhas e prompts pulam. Medido: prosa 1 req 48,36 → 49,46, código 59,54 → 61,08 (dois boots cada).
- **Loads da treliça EXL3 (`TF_GLM_EXL3_LOADS=nc`, patch 0047, adaptado do patch 0580 do jayleaton):** "16-byte `ld.global.nc.L1::no_allocate` loads of the trellis a k step ahead through a staging area in the warp's slice of the reduction buffer" (`nc2`/`nc4` = 2 ou 4 passos; `0` = loads de 32 bits do TF). Sozinho: prosa 48,36 → 48,78, código 59,54 → 60,42. **Os três juntos** (L2PF + loads + `TF_ROCE_MAX_KB=512`): 1 req prosa 49,68, código 61,49 (+2,7% / +3,3%); 4 req prosa 74,8 → 76,6, código 100,0 → 102,7 tok/s. Mesmos bits.
- **Kernels de decode (patches 0016, 0019, 0031):** matmuls de decode, mistura de hyper-connections e pontuação do indexador mais rápidos (índices em registradores no indexador).
- **(terceiros, jayleaton THEORY-2, motor deles)** a rodada de decode é "one serial chain on one CUDA stream" (99,97% dos kernels num stream, 53 ms por rodada a 1 stream); os kernels grandes já leem a 205–217 GB/s contra um teto de leitura de 235 GB/s; o transporte puro de uma troca é 10,8 µs e o resto da espera é **desalinhamento entre ranks** (jitter de kernels idênticos, 9–17% p5–p95). **(interpretação)** Explica por que L2 prefetch rende pouco (~1–3%): só preenche fases ociosas de DRAM.

### 2.16 Visão (imagens e vídeo)

- **(repo)** A torre de visão do próprio GLM roda **só no rank 0**: 1,05 GiB de pesos bf16 + 0,75 GiB de workspace (~1,8 GiB). `VISION=0` devolve esse espaço ao cache. Processadores de imagem e vídeo (resize com pad, escolha de quadros a 2 fps, layout do prompt) "checked bit for bit against Hugging Face transformers 5.17" (patch 0003).
- Limites (patch 0050): até 50 imagens (`TENSORFOLD_GLM_MAX_IMAGES`; 10 MB cada, 64 MB no total) e 4 vídeos (`TENSORFOLD_GLM_MAX_VIDEOS`; 64 MB cada, 96 MB no total, até uma hora); ≤ 2.048 tokens por imagem (1080p = 2.040), 16.384 compartilhados por requisição (acima de 8 imagens, divisão igual: 327 com 50); vídeo a 2 fps, ≤ 128 quadros, ≤ 16.384 tokens por clipe, 32.768 por requisição (3 ou 4 clipes: 10.922 ou 8.192). Corpo até 96 MiB. Só data URLs por default; `VISION_URLS=1` busca URLs `https://` públicas.
- Medido (v1.1): 50 imagens numa requisição respondidas em 12,5 s (16.239 tokens de prompt), menor memória livre no head 6,75 GiB; 4 vídeos 6,90 GiB.
- Com `KV=bf16` e `DENSE` fp8/bf16, a torre custa ~66k tokens de janela no rank 0.

### 2.17 Superfície de API

- Rotas: `/v1/chat/completions`, `/v1/completions`, `/v1/responses` (Responses API, executada como chat completion equivalente; armazena até 1.024 respostas/256 MiB), `/v1/models`, `/tokenize` e `/detokenize` (também sob `/v1/`, **campos do vLLM**: `prompt` ou `messages` → ids com `count` e `max_model_len`), `/health`, `/metrics` (Prometheus: contadores, histogramas de latência e TTFT do TF v0.6.0 + as cifras do `/health` como `tensorfold_health:*`, patch 0045). Sem `/v1/messages`.
- **Contexto:** prompt + `max_tokens` além da janela → HTTP 400 no formato da OpenAI, `"code": "context_length_exceeded"`, `param` = `messages` ou `prompt` (patch 0044) — clientes que compactam a conversa nesse erro funcionam. Sem `max_tokens`, a requisição ganha `MAX_TOKENS` (32.768), cortado ao que sobra da janela, nunca recusado.
- **Ferramentas (patches 0036, 0051, 0053):** chamadas GLM (`arg_key`/`arg_value`) → `tool_calls` OpenAI, argumentos tipados pelo schema (array como array JSON; `null` e `const`). **Cada chamada é transmitida inteira quando terminada**, com deltas vazios a cada 2 s enquanto é escrita (clientes com timeout de ociosidade); uma resposta que termina **dentro** de uma chamada (limite de tokens) termina com `length` e **nunca** envia a chamada cortada. Chamadas escritas dentro do bloco de pensamento contam se a resposta termina nelas. O raciocínio de um passo de ferramenta é **reinserido** na próxima requisição quando o cliente agente o descarta (`TF_GLM_KEEP_REASONING=0` desliga; a assinatura cobre a conversa inteira antes das chamadas). Uma chamada passada cujos argumentos não são um objeto JSON é **omitida** do prompt com seu resultado (e logada), em vez de HTTP 400 em todo turno seguinte (#2).
- **Saídas estruturadas:** `response_format` (`json_object`/`json_schema`) e `guided_json`/`guided_regex`/`guided_choice`/`guided_grammar`/`structured_outputs`, com **xgrammar**, aplicadas **após** o bloco de pensamento. Com rascunhos ligados: "Before each verify forward the engine cuts the drafts the grammar rejects, then masks each remaining row's logits", logo resposta restrita = `"draft": false` = solo (TF docs).
- **Raciocínio:** resposta em `reasoning_content` + `content`. `reasoning_effort` `low`/`high`/`max` (topo ou `chat_template_kwargs`); `medium` = `high`, `xhigh` = `max` (patch 0023); sem ele, o template do GLM usa **Max**; `"none"` ou `enable_thinking: false` responde sem pensar. `THINKING=0` muda o default.
- **Amostragem:** `temperature`, `top_p`, `top_k`, `min_p`, `seed` sobrescrevem os defaults do checkpoint (T 1,0, top_p 0,95); `temperature: 0` = greedy. **Recusados (400):** `logprobs: true`, `top_logprobs`, `n ≠ 1`. **Aceitos e ignorados:** `logit_bias`, penalidades. `priority: "background"`.
- **`"draft": false`** seleciona a referência serial (verificação de exatidão). A resposta traz um bloco `tensorfold` (ex.: `prefill_s`, `decode_s`, `token_sha`).

### 2.18 Imagem, hashing, GHCR, pins de revisão e NFS

- **Build (prepare.sh, Dockerfile inline):** `FROM nvcr.io/nvidia/pytorch:26.07-py3`; `pip install git+https://github.com/ashhart/TensorFold.git@v0.6.0` e os extras `av==18.1.0 xgrammar>=0.2.8,<0.3`; `COPY patches /opt/tf-patches`; no diretório que contém o pacote `tensorfold`, aplica cada `*.patch` em ordem de nome com `patch -p0 --forward` (falha = build falha); importa `tensorfold.cuda.server`, `tensorfold.families.glm5_next.cuda.engine`, `tensorfold.vision.glm`, `av`, `xgrammar` como teste; `LABEL tf.patches=<hash>`; `ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton`.
- **Hash da imagem:** `image_hash()` = SHA-256 de `cat patches/*.patch` + `IMAGE_EXTRAS`, 12 primeiros hex. Tag publicada: `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold:v0.6.0-<hash>`. Enquanto os patches forem os da release, o `prepare.sh` puxa **por digest** (`IMAGE_TAG=v0.6.0-ae8d1c789b47`, `IMAGE_DIGEST=sha256:22789f0c…bb969`): "a tag can be moved, a digest cannot". Patches alterados → procura a tag do novo hash; se não existir (ou `PULL=0`), constrói localmente. A imagem puxada só é aceita se seu label `tf.patches` bater com o hash.
- **Publicar (`publish-image.sh`):** confere o label, re-etiqueta com labels OCI (`org.opencontainers.image.source` = repo, para o pacote herdar visibilidade), faz login com um **`DOCKER_CONFIG` descartável** (nenhuma credencial fica em `~/.docker`), push de `:<tag>` e `:latest`, imprime os valores a fixar.
- **Worker:** tenta puxar a mesma imagem; senão `docker save | ssh docker load` (só camadas faltantes). Identidade comparada **por conteúdo** (`RootFS.Layers` + `Env`/`Entrypoint`/`Cmd`/`WorkingDir`, SHA-256): o `.Id` é o digest do manifesto no containerd e o da config no overlay2, então nunca casaria entre stores mistos (#8).
- **Cache de kernels por imagem:** `~/.cache/tensorfold-glm53/<image hash>` montado em `/cache` — "another image's build of the same extension can no longer be loaded by mistake"; a primeira partida de uma imagem compila (alguns minutos).
- **Pins de revisão:** `MODEL_REVISION` e `DFLASH2_REVISION` (shas de commit do HF) — os dois ranks servem exatamente esses snapshots do cache local (`HF_HUB_OFFLINE=1` nos containers); "a new upstream commit changes nothing until the pin does". Vazio = o `main` do Hub no primeiro download. Download por sha não grava `refs/main`; o script o escreve.
- **Cópia para o worker:** `rsync -a -L --partial` dos blobs da revisão (`-L` porque no backend xet do `huggingface_hub` os blobs são links para o `blobs/` da raiz do cache, #15), depois os links do snapshot e `refs/main`; verificação **arquivo a arquivo** (lista `caminho tamanho` com links seguidos). Retoma se interrompido.
- **NFS (`WORKER_WEIGHTS=nfs`):** o head exporta `HF_CACHE` read-only para o endereço CX7 do worker; o `prepare.sh` cria no worker um **volume docker NFS** (`type=nfs`, `nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600`; sem sudo no worker) e confere que o worker vê cada arquivo de ambos os snapshots como o head. "Loading is as fast as from the worker's own disk (both ranks were live in ~2.2 minutes)"; economiza ~166 GiB de disco no worker.
- **Marcador de preparo:** `prepared_state` = linha com modelo@revisão, drafter@revisão, id da imagem nos dois lados, hash dos patches, worker e modo de pesos; o `start.sh` roda o `prepare.sh` sempre que a linha atual difere da gravada em `STATE_DIR/prepared`.

---

## 3. Inventário de TODOS os parâmetros

**Precedência (repo, `config.sh`):** o primeiro que define vence — (1) o ambiente (`PARALLEL=2 ./start.sh restart`), (2) `scripts/local.sh` (sourced como bash; o ambiente ainda vence o que ele mudou), (3) `./.env` (linhas `KEY=value`, aspas simples/duplas e comentários `#` tratados; **lido, nunca executado**; onde ambos definem, `local.sh` vence), (4) os defaults. Qualquer `TENSORFOLD_*`, `TF_GLM_*`, `TF_ROCE_*` presente no ambiente é passado com `-e` aos **dois** containers (o comando do rank 1 roda por ssh, onde o ambiente local não chega; "None of them is a secret"). Argumentos extras de `./start.sh [restart] …` vão a `tensorfold serve` **depois** dos defaults, nos dois ranks (argparse guarda o último valor), e são validados pelo próprio parser do TensorFold num container descartável **antes** de parar o servidor em execução.

### 3.1 Nós, rede e rendezvous

| Variável | Default | Vai para | O que faz | Por quê / números |
|---|---|---|---|---|
| `WORKER` | vazio (obrigatório) | ssh | alvo ssh do worker (`user@<endereço>`), chave sem senha (`BatchMode=yes`, `ConnectTimeout=10`) | o head roda `./start.sh` e a API; o worker não precisa do repositório. `local.sh.example`: `WORKER=user@192.0.2.2` |
| `FABRIC_PEER` | vazio | `detect_link` | endereço CX7 do worker quando `WORKER` está noutra rede | os ranks falam pela rota até esse endereço. Desde v1.3.1, se a rota sai por uma porta sem RoCE, o script acha sozinho um endereço CX7 do worker alcançável diretamente (#9) |
| `MASTER_PORT` | `29551` | `--master-port` | porta de rendezvous (`TCPStore` do rank 0) | "keep it on the private link" — rendezvous e link não são autenticados (TF RUNBOOK) |
| `NCCL_RAILS` | (2 implícito) | `NCCL_IB_HCA` | `1` = usar uma porta CX7 mesmo com as duas ativas | dois trilhos: all-gather de chunk de prompt (32 MB) 1,9 ms vs 3,5 ms em um (~1,8×) |
| `NCCL_CHANNELS` | `4` | `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS` | canais NCCL | all-gather de decode 32 µs (80 µs com defaults do NCCL) |
| `NCCL_DEBUG` | vazio | `NCCL_DEBUG` | log do NCCL | diagnóstico |
| (derivado) `NCCL_SOCKET_IFNAME` | netdev da rota ao par | env do container | interface de bootstrap | `ip -o -4 route get <peer>` |
| (derivado) `NCCL_IB_HCA` | HCA do link (+ a 2ª função na mesma sub-rede, se RoCE v2 no mesmo índice de GID) | env | trilhos RDMA | só se as duas pontas tiverem o mesmo número de trilhos |
| (derivado) `NCCL_IB_GID_INDEX` | índice da entrada **RoCE v2** com o IPv4 do nó (`::ffff:a.b.c.d`) | env | GID do RoCE | "an all-zero or v1 entry makes NCCL fail about a minute in" |

### 3.2 Modelo, drafter, imagem, caches e disco

| Variável | Default | O que faz | Por quê / números |
|---|---|---|---|
| `MODEL_ID` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | checkpoint servido | EXL3 nos experts roteados (4 bpw), BF16 no resto; ~176 GB (164 GiB) |
| `MODEL_REVISION` | `9eaebb7c4e96d983dcd538e18624622ba5b820a8` (só para o `MODEL_ID` default) | pin do commit HF | reprodutibilidade; vazio = `main` do Hub no 1º download |
| `DFLASH2_ID` | `incoai/GLM-5.3-Flash-DFlash2` | drafter | ~2,3 GB; CC BY-NC-ND 4.0 |
| `DFLASH2_REVISION` | `bf582e4eacc1810f76656d1811693ff6c6737d2a` | pin do drafter | idem |
| `TF_VERSION` | `v0.6.0` | tag do TensorFold instalada | os patches são feitos para ela; mudar exige `prepare.sh --rebuild` |
| `TF_REPO` | `https://github.com/ashhart/TensorFold.git` | origem pip | — |
| `BASE_IMAGE` | `nvcr.io/nvidia/pytorch:26.07-py3` | base do container (CUDA, PyTorch, Triton, compilador de extensões) | TF: "Use NVIDIA's PyTorch container for CUDA"; não há extra `cuda` |
| `IMAGE` | `tensorfold-glm53:${TF_VERSION}` | nome local da imagem | — |
| `IMAGE_EXTRAS` | `av==18.1.0 xgrammar>=0.2.8,<0.3` (fixo no config) | pacotes pip extras | PyAV (vídeo), xgrammar (saídas estruturadas); entram no hash |
| `GHCR_IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` | registro da imagem pronta | — |
| `IMAGE_TAG` / `IMAGE_DIGEST` | `v0.6.0-ae8d1c789b47` / `sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969` | imagem publicada desta release, puxada por digest enquanto o hash dos patches bater | "a tag can be moved, a digest cannot" |
| `CONTAINER_NAME` | `glm53-flash-tf` | mesmo nome nos dois Sparks | `docker logs -f glm53-flash-tf` |
| `HF_CACHE` | `$HF_HOME` ou `~/.cache/huggingface` | cache HF montado em `/root/.cache/huggingface` | — |
| `KERNEL_CACHE` | `~/.cache/tensorfold-glm53` | kernels compilados, uma pasta por hash de imagem, montada em `/cache` | evita carregar build de outra imagem |
| `STATE_DIR` | `~/.local/state/glm53-tensorfold` | locks (`flock`) e marcador de preparo | um `start.sh`/`prepare.sh` por vez |
| `WORKER_WEIGHTS` | `copy` | `copy` (cópia própria no worker, ~166 GiB) ou `nfs` (lê o cache do head por NFS, read-only) | NFS: ambos os ranks no ar em ~2,2 min |
| `NFS_PATH` | `$HF_CACHE` | caminho exportado como o worker monta (`/` para NFSv4 com `fsid=0`) | — |
| `NFS_SERVER` | endereço do head no link | servidor NFS | — |
| `NFS_VOLUME` | `glm53-hf` | volume docker NFS no worker | criado sem sudo |
| `MIN_FREE_GB` | `180` | disco livre exigido sob `HF_CACHE` para o download | checkpoint ~176 GB |
| `IMAGE_FREE_GB` | `35` | disco livre sob a raiz do Docker para build/cópia da imagem (~25 GB) | soma com o anterior se compartilham o filesystem |
| `RSYNC_OPTS` | vazio | opções extras do rsync | — |
| `HF_TOKEN` | — | download (gated/rate limit) | nenhum token entra nos containers |
| `HF_HUB_OFFLINE` | `1` nos containers | `0` deixa o servidor alcançar o Hub | default: serve só do cache local |
| `PREPARE` | `auto` | `1` sempre roda `prepare.sh`, `0` nunca | `auto`: quando o marcador difere |
| `PULL` | `1` | `0` = sempre construir localmente | — |

### 3.3 Serviço (o que vira flag de `tensorfold serve` ou variável do motor)

| Variável da receita | Default | Vai para | O que faz | Por quê / números |
|---|---|---|---|---|
| `SERVED_NAME` | `GLM-5.3-Flash-EXL3` | `--name` (rank 0) | id em `/v1/models` | — |
| `HOST` / `PORT` | `0.0.0.0` / `8888` | `--host` / `--port` (rank 0) | onde a API escuta | TF default é `127.0.0.1:8080` |
| `DRAFTER` | `dflash2` | `--drafter <snapshot>` ou `--drafter none` | `dflash2` (IncoAI, não-comercial; +5–10% de decode) ou `mtp` (cabeça MTP do checkpoint; uma requisição por vez) | definir `mtp` antes do 1º start evita baixar o DFlash2 |
| `TF_GLM_MTP` | `auto` | env | carrega a cabeça MTP ao lado do DFlash2? `auto` = não, enquanto o DFlash2 rascunha tudo; `1` (default do TF v0.6.0) = sim, com `PARALLEL=1`; `0` = nunca | 1,77 GiB por Spark |
| `PARALLEL` | `4` (`1` com `mtp`) | `--parallel` | requisições decodificadas juntas, 1–4 (>1 exige `dflash2`) | prosa 60,4 / 79,2 / 89,5 / 108,8 tok/s a 1/2/3/4; structured 114,7 / 147,6 / 196,3 / 227,9 |
| `CONTEXT` | `1048576` com `KV=fp8`; `196608` com `KV=bf16`+DFlash2 (`163840` com `VISION=1` e `DENSE` fp8/bf16); `524288` com `KV=bf16`+mtp; `0` = o maior que cabe | `--context` | janela prompt + resposta por requisição (≤ 1.048.576) | `0`: "no memory is left to keep other conversations' prompts" |
| `KV` | `fp8` | `TF_GLM_KV` | latente DSA e chaves agrupadas do indexador em `fp8` (e4m3 + escala potência de 2) ou `bf16` (exato) | fp8: metade dos bytes, cabe 1M com 4 streams; lossy (GSM8K 98,0%, HumanEval 97,6%, agulha de 1M achada) |
| `DENSE` | `q4` | `TF_GLM_DENSE` | pesos BF16 não-roteados: `q4` (4-bit grupos de 64, faixa por MSE; cabeça FP8; `kv_b` BF16), `fp8` (e4m3, escala por linha e bloco de 128) ou `bf16` | q4 sobre fp8: prosa 38,9 → 44,4, código 44,2 → 48,7, prefill ~1.090 → ~1.260 tok/s; fp8 ~+33% sobre bf16. **Não-inglês:** q4 perde o fim de turno em prompts curtos de código em francês (12/48 vs 1/48), #18 — use `fp8` (~10% mais lento no decode) |
| `THINKING` | `1` | `--thinking` / `--no-thinking` | pensar antes de responder por default | `0`: responde direto salvo se a requisição pedir |
| `MAX_TOKENS` | `32768` | `--max-tokens` | orçamento de resposta (raciocínio + resposta) sem `max_tokens` | TF default 4.096 podia terminar uma resposta dentro de uma chamada de ferramenta (agentes como Codex não mandam `max_tokens`) |
| `VISION` | `1` | `--vision` | imagens e vídeo | torre no rank 0: ~1,8 GiB |
| `VISION_URLS` | `0` | `--vision-urls` | aceitar URLs `https://` públicas | default: só data URLs |
| `COMM` | `roce` | `TF_GLM_COMM` | all-gathers pequenos por RDMA one-shot; `nccl` = tudo por NCCL | 11 µs vs 45 µs por gather de 16 KiB; decode +6% (prosa 44,2 → 46,9; código 48,7 → 51,6). Mesmos bits |
| `TF_ROCE_MAX_KB` | `512` (TF: 256) | env | maior all-gather (KiB) por RoCE | cobre as janelas de 17–32 linhas do lote: código a 4 streams +1,4%, prosa +0,5–1% |
| `TF_ROCE_WAIT_S` | `20` | env | segundos que uma espera RoCE aguarda o outro Spark (1–3.600) antes de falhar | até o motor estar construído, cada gather encontra o par numa barreira NCCL antes (#6) |
| `COPY` | `1` | `TF_GLM_COPY_DRAFTS` | copy drafts (prompt lookup) | citação/edição 80,3 → 84,4 tok/s; exato |
| `COPY_MAX` | `15` (1–15) | `TF_GLM_COPY_MAX` | copy drafts por rodada | edição 84 → 119 tok/s |
| `COPY_CODE` | `1` | `TF_GLM_WIDE_GRAPHS=16`, `TF_GLM_COPY_REPLY_MATCH=16` | janelas de 16 linhas como CUDA graphs; cópias da própria resposta só após casar 16 tokens | código 55,4 → 56,2, edição 120 → 125 |
| `DRAFT_POLICY` | `fnc7:0.3` | `TF_GLM_DFLASH_POLICY` | até 7 rascunhos DFlash2 enquanto o produto das chances sob o ruído com chave ≥ 0,3 | prosa 47,2 → 50,6, código 51,7 → 55,5 sobre `fc5:0.3`; o `start.sh` exige prefixo `f` (`fc5:0.3`, `fnc7:0.3`, `fcost7:noisy`…) |
| `SPLIT` | `1` | `TF_GLM_HC_SPLIT=1`, `TF_GLM_PREFILL_OVERLAP=2` (ou `0`/`0`) | cola de hyper-connection dos chunks de prompt dividida por linhas entre os Sparks, trocas sobrepostas | 50k: ~1.270 → ~1.730 tok/s (com 0009 e 0020); 149k 92,8 → 90,7 s com o overlap 2; decode paga ~1,5% |
| `KDA_CHUNKED` | `1` | `TF_GLM_KDA_CHUNKED` | KDA dos chunks de prompt em forma WY, um kernel CUDA de sub-chunks de 32 linhas | 50k 29,3 → 26,4 s; 149k 91,1 → 84,5 s; estados na grade de 64; perto do serial, não os mesmos bits |
| `SHARED_PREFIX` | `1` | `TF_GLM_SHARED_PREFIX` | conversas que compartilham system prompt reusam o estado | 7,9k: 4,24 → 0,13 s |
| `MULTI_PREFILL` | `1` | `TF_GLM_MULTI_PREFILL` | prompts que chegam juntos preenchidos numa forward | ×4 prosa 103,4 → 108,8 tok/s, TTFT 590 → 340 ms; structured ×4 196,3 → 227,9. Exato |
| `TF_GLM_MULTI_LONE` | `0` | env | `1`: stream sozinho nos graphs de stream único | +0,6–0,9%, mas despeja prompts guardados (cache 0% vs 98,1%), #12/#13 |
| `TF_GLM_CACHE_ENTRIES` | `32` (TF: 8) | env | máximo de estados de prompt guardados | ~45 MiB cada na partida (32 ≈ +1 GiB vs 8); #17 |
| `TF_GLM_L2PF` | `1` (`0`/`off`, `1`/`bulk`, `lines`, `touch`) | env | L2 prefetch no decode | prosa 48,36 → 49,46, código 59,54 → 61,08 |
| `TF_GLM_EXL3_LOADS` | `nc` (`0`/`ldg`, `1`/`nc`/`nc1`, `nc2`, `nc4`) | env | loads de 16 B não-coerentes da treliça um passo à frente | sozinho: prosa 48,36 → 48,78, código 59,54 → 60,42; com L2PF + 512 KiB: +2,7%/+3,3% |
| `KV_POOL_GIB` | `12.5` | `TF_GLM_CACHE_GIB` (TF: 3) | teto do pool além da janela, por Spark | pool 2.922.496 tokens medido (v1.2: 11 → 12,5 deu 2.684.928 → 2.852.864); TF defaults: 1.411.072 |
| `MEMORY_RESERVE_GIB` | `14.5` | `TENSORFOLD_MEMORY_RESERVE_GIB` (TF: max(4 GiB, RAM/10), ~12,2) | o que o orçamento deixa livre | define a menor memória livre no head (~4,5–5 GiB); aumente se outra carga divide os Sparks |
| `TENSORFOLD_NO_UPDATE_CHECK` | `1` | env | sem checagem de release na partida | serviço offline e reprodutível |
| `TF_GLM_KEEP_REASONING` | (ligado) | env | reinsere o raciocínio de passos de ferramenta descartados pelo cliente | `0` desliga |
| `TENSORFOLD_NUCLEUS_UNION` | desligado | env | teste de cobertura top-p com candidatos dos dois ranks | mesmo sorteio, menos leituras de shard inteiro (patch 0034) |
| `TF_GLM_DRAFT_DUMP` | desligado | env (nos dois ranks) | grava registros teacher-forced do DFlash2 para o simulador | uso de pesquisa (patch 0011) |

### 3.4 Visão

| Variável | Default | Significado |
|---|---|---|
| `TENSORFOLD_GLM_IMAGE_TOKENS` | `2048` | teto de tokens por imagem (1080p = 2.040) |
| `TENSORFOLD_GLM_VIDEO_TOKENS` | `16384` | teto por clipe |
| `TENSORFOLD_GLM_VIDEO_FRAMES` | `128` | quadros por clipe (2 por segundo, espalhados no clipe) |
| `TENSORFOLD_GLM_MAX_IMAGES` | `50` | imagens por requisição (eram 4 em v1.0) |
| `TENSORFOLD_GLM_MAX_VIDEOS` | `4` | vídeos por requisição (eram 2) |
| `TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS` | `16384` | tokens que as imagens de uma requisição dividem |
| `TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS` | `32768` | tokens que os vídeos de uma requisição dividem |

### 3.5 Operação do launcher

| Variável | Default | Significado |
|---|---|---|
| `WAIT_TIMEOUT` | `1800` s | quanto o `start.sh` espera a API responder |
| `STOP_TIMEOUT` | `30` s | `docker stop -t` (não drena requisições; avisa quantas serão cortadas via `/health.requests_running`) |
| `FOREGROUND` | `0` | `1`: fica preso ao log do rank 0 e sai com o código dele (unidade systemd), sem progresso, sem retry de janela e sem smoke test; se um rank termina, para o outro |
| `PREPARE`, `PULL` | `auto`, `1` | ver 3.2 |

### 3.6 Knobs internos do motor que a receita **não** muda (defaults do código dos patches)

| Variável | Default | Significado |
|---|---|---|
| `TF_GLM_FILL_ROWS` | `1024` (múltiplo de 64 e da grade, ≤ chunk) | linhas de um chunk de prompt enquanto outros decodificam |
| `TF_GLM_FILL_SHARE` | `0.5` | fração de iterações para chunks de prompt quando outros decodificam |
| `TF_GLM_MULTI_VERIFY` | `batched` | `serial` = referência |
| `TF_GLM_MULTI_GRAPHS` / `TF_GLM_MULTI_DRAFT_GRAPHS` | `1` / `1` | graphs das janelas em lote / do drafter multi-stream |
| `TF_GLM_MULTI_SAMPLER` | `streams` | `packed`: tudo em dois all-gathers |
| `TF_GLM_MULTI_DEPTH` | `policy` | `joint`, `scale:A` |
| `TF_GLM_MULTI_OVERHEAD_MS` | `12` | resto da rodada na alocação `joint` |
| `TF_GLM_MULTI_ASYNC` | `0` | mensagem do rank 0 sem esperar a GPU |
| `TF_GLM_MULTI_PROFILE` | `0` | perfil a cada N rodadas |
| `TF_GLM_MULTI_WATCHDOG_S` | `300` | despejo de pilhas de iteração travada |
| `TF_GLM_MAX_ROWS` | `16` (16–64, passo 8) | janela de verificação mais larga por stream |
| `TF_GLM_COPY_MATCH` | `8` | tokens que precisam ter ocorrido antes |
| `TF_GLM_COPY_MISS_MAX` | `0` | rascunhos copiados após uma rodada copiada que errou |
| `TF_GLM_PROMPT_GRID` | `0`/`64` (64 com `KDA_CHUNKED=1`; valores 16/32/64/128) | grade dos chunks e estados guardados |
| `TF_GLM_KDA_PREC` / `_BV` / `_STATE_ROWS` / `_KSPLIT` | `tf32x3` / `16` / `32` / `4` | precisão e geometria do KDA chunked (versão Triton) |
| `TF_GLM_HC_SPLIT_MIN_ROWS` | `64` | chunks menores não dividem |
| `TF_GLM_OVERLAP_PIECES` | `2` | pedaços das trocas sobrepostas |
| `TF_GLM_HC_EXCHANGE` | `p2p` | `gather` = mesmo volume pelo all-gather |
| `TF_GLM_L2PF_MB` / `_SITES` / `_ROWS` / `_KB` | `8` / `afo` / `64` / `32` | orçamento, pontos, largura máxima de janela, tamanho do pedaço |
| `TF_GLM_LATENT` | `1` | cache latente (0 = K/V por cabeça) |
| `TF_GLM_DRAFT_RING` | `1` | anel de janela do DFlash2 (0 = buffer da janela inteira) |
| `TF_GLM_EXL3_NFIRST` / `_ORDER` / `_PF` / `_PROMPT` | ligados | ordem e prefetch dos kernels de experts no prompt |
| `TF_GLM_KVB` | (q4 → BF16) | formato do `kv_b` |
| `TF_GLM_SPARSE_ONEPASS` | ligado | atenção esparsa do prompt sem parciais |
| `TENSORFOLD_NO_LIVE` | — | desliga a linha de throughput ao vivo no terminal |
| `TF_NCCL_LIB` | — | caminho do `libnccl` se não for achado |

### 3.7 O que o `start.sh` passa ao Docker e ao TensorFold

```text
docker run -d --name glm53-flash-tf --gpus all --ipc=host --network host --shm-size 16g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 -e <todo TENSORFOLD_*/TF_GLM_*/TF_ROCE_*> \
  -e NCCL_SOCKET_IFNAME=<dev> -e NCCL_IB_HCA=<hca[,hca2]> -e NCCL_IB_GID_INDEX=<gid> \
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 \
  -v $HF_CACHE:/root/.cache/huggingface        # worker: sua cópia, ou o volume NFS :ro
  -v $KERNEL_CACHE/<image hash>:/cache \
  tensorfold-glm53:v0.6.0 \
  tensorfold serve /root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/<rev> \
    --tp 2 --rank {1|0} --master <HEAD_ADDR> --master-port 29551 \
    [rank 0: --name GLM-5.3-Flash-EXL3 --host 0.0.0.0 --port 8888] \
    --drafter <snapshot do DFlash2 | none> --context 1048576 --parallel 4 --max-tokens 32768 \
    --thinking --vision [--vision-urls] [args extras]
```

**(interpretação dos flags do Docker)** `--ipc=host` e `--shm-size 16g` para memória compartilhada; `--network host` para a rede do CX7 e o rendezvous; `--device /dev/infiniband` + `IPC_LOCK` + `memlock=-1` para registrar memória pinada no RDMA (verbs); `stack=67108864` para pilhas grandes de threads.

---

## 4. Números e benchmarks

### 4.1 Condições de medição

- **(repo)** "Two DGX Sparks at the default configuration (4 streams, 1,048,576-token window, FP8 KV cache, 4-bit dense weights, DFlash2 plus copy drafts, vision on), with the GPU clocks capped at 2,200 MHz. Decode and prefill were measured with sparkDash through the OpenAI API, from another machine on the network."
- **sparkDash (MiaAI-Lab):** benchmark de decode multi-concorrência com streaming; tipos Structured / Prose / Code / JSON ("Code is a different Python task per stream"); "Lab protocol (temp 0, thinking off)".
- **(interpretação)** Travar clocks (presumivelmente `nvidia-smi -lgc`) tira a variação térmica/de boost entre corridas e entre ranks — importante num motor TP=2 onde o rank mais lento dita a rodada (o jayleaton mediu 1,3–3,0 ms de espera por desalinhamento entre ranks por rodada). Números com clocks livres podem ser diferentes (nem melhores nem piores por princípio).
- A/Bs de knobs no `config.sh` são "two boots each" ou "one boot each" — descritivos, sem intervalo de confiança.

### 4.2 Decode (sparkDash)

| Requisições | Prosa agregado | Prosa por req | TTFT | Structured agregado | Structured por req | TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60,4 | 60,4 | 170 ms | 114,7 | 114,7 | 149 ms |
| 2 | 79,2 | 40,4 | 269 ms | 147,6 | 77,1 | 295 ms |
| 3 | 89,5 | 30,6 | 330 ms | 196,3 | 67,4 | 317 ms |
| 4 | 108,8 | 27,9 | 340 ms | 227,9 | 61,1 | 415 ms |

Exatidão sob concorrência: "Replies served 4 at a time are identical to the same requests served one at a time (11 of 11 cases staggered, and 11 of 11 sent in a burst)". v1.3: "drafted == serial 6/6, concurrency 22/22".

### 4.3 Prefill (sparkDash)

| Prompt | Prefill | TTFT |
|---:|---:|---:|
| 8.219 | 1.952,2 tok/s | 4,21 s |
| 16.407 | 1.973,5 | 8,31 s |
| 32.790 | 1.978,9 | 16,57 s |
| 65.563 | 1.942,5 | 33,75 s |
| 131.099 | 1.837,9 | 71,33 s |
| 262.170 | 1.641,7 | 159,69 s |
| 981.841 (agulha) | 1.015 | 967 s, agulha encontrada |

### 4.4 Reuso de prompt

| Caso | Primeira vez | Próxima |
|---|---:|---:|
| Prompt idêntico de 64k reenviado | 34 s | < 0,07 s |
| Conversa nova com o mesmo system prompt de 7,9k | 4,24 s | 0,13 s |
| 3 conversas alternando (~4,5k cada, 6 turnos), `MULTI_LONE=1` vs `0` | 0% do cache | 98,1% do cache |

### 4.5 Memória e KV

| Item | Valor |
|---|---|
| Pool compartilhado (partida medida, v1.3) | **2.922.496 tokens** (teto de 12,5 GiB; ~2,1–2,9M conforme memória livre) |
| Histórico do pool | v1.0 ~2,7M; v1.1 2.684.928; v1.2 2.852.864 (KV_POOL 11 → 12,5); defaults do TF (reserva RAM/10, 3 GiB): 1.411.072 |
| Estimativa de partida do rank 0 | 88,09 GiB |
| `MemAvailable` ocioso | 7,0 GiB (rank 0) / 10,5 GiB (rank 1) (v1.2: 7,8 / 10,8) |
| Menor livre sob prompt de 1M | 5,6 / 9,5 GiB (v1.2: 4,7 / 8,8; v1.1: 6,3 / 10,8 com pool 11 GiB) |
| Pico além da estimativa | ~10 GiB (prompt de 1M) |
| Livre exigido na partida | ~110 GiB por Spark |
| Janelas alternativas | `KV=bf16`: 196.608 (163.840 com visão e dense fp8/bf16); `KV=bf16 DRAFTER=mtp`: 524.288 (1 req); `CONTEXT=0`: o maior que cabe |
| Bytes por linha | latente FP8 528 B (bf16 1.024 B); chave agrupada FP8 144 B por pool de 4 tokens (bf16 256 B) |
| Estado de prompt guardado | ~45 MiB cada (fixo) |
| DFlash2 | anel 21 MiB/rank; janela guardada 20 MiB/rank; modelo 0,63 GiB/rank; cabeça MTP 1,77 GiB/Spark |
| Torre de visão | 1,05 GiB + 0,75 GiB (rank 0) |
| Disco por Spark | ~205 GB: ~176 GB checkpoint + ~2,3 GB DFlash2 + ~25 GB imagem |

### 4.6 Qualidade

| Benchmark | Score | Condição |
|---|---:|---|
| GSM8K (250 problemas) | 98,0% | KV FP8 + dense q4, "measured on a build with identical replies" (README) |
| HumanEval (164) | 97,6% | idem |
| GSM8K / HumanEval, q4 vs fp8 (config.sh) | 98,0% / 95,1% nos dois; bf16 97,2% / 96,3% | DFlash2, um boot cada |
| Agulha | 981.841 tokens achada; `tools/needle.py` ~195k tokens (60% de profundidade, greedy, seed 1234) | |
| Prompts curtos de código em francês, `q4` | 12/48 respostas sem fim de turno (vs 1/48 bf16/fp8) | #18 |

### 4.7 Efeito medido de cada knob/patch (do `config.sh` e do README)

| Knob / patch | Antes → depois | Bits |
|---|---|---|
| `DENSE=q4` vs `fp8` (0002, 0005) | prosa 38,9 → 44,4; código 44,2 → 48,7; prefill ~1.090 → ~1.260 tok/s | lossy |
| `DENSE=fp8` vs `bf16` | ~+33% decode | lossy |
| `KV=fp8` (0038) | 1M com 4 streams cabe | lossy |
| `SPLIT=1` (0010 + 0009 + 0020) | 50k ~1.270 → ~1.730 tok/s; decode −~1,5% | iguais |
| `TF_GLM_PREFILL_OVERLAP=2` (0033) | 149k 92,8 → 90,7 s | iguais |
| `KDA_CHUNKED=1` (0012, 0014, 0039) | 50k 29,3 → 26,4 s; 149k 91,1 → 84,5 s | próximos, não iguais |
| Prompt experts NFIRST (0001) | 2,3–2,5× mais rápido a 4.096 linhas (kernel) | iguais |
| Reuso (0008, 0015, 0042) | ver 4.4 | iguais |
| `COPY_MAX=15` (0013) | edição 84 → 119 tok/s | iguais |
| L2PF + EXL3 loads + RoCE 512 KiB (0046, 0047) | 1 req prosa 48,36 → 49,68, código 59,54 → 61,49; 4 req prosa 74,8 → 76,6, código 100,0 → 102,7 | iguais |
| `COMM=roce` (0006, 0052) | 11 µs vs 45 µs por 16 KiB; decode +6% | iguais |
| `TF_ROCE_MAX_KB=512` | código ×4 +1,4%; prosa +0,5–1% | iguais |
| `COPY=1` (0007) | citação/edição 80,3 → 84,4 | iguais |
| `COPY_CODE=1` (0032) | código 55,4 → 56,2; edição 120 → 125 | iguais |
| `DRAFT_POLICY=fnc7:0.3` (0018) vs `fc5:0.3` | prosa 47,2 → 50,6; código 51,7 → 55,5 | iguais |
| `MULTI_PREFILL=1` (0049) | ×4 prosa 103,4 → 108,8, TTFT 590 → 340 ms; structured ×3 175,2 → 196,3; ×4 196,3 → 227,9 | iguais |
| `TF_GLM_MULTI_LONE=1` (0035) | +0,6–0,9% a 1 stream (mas destrói o cache de prompts) | iguais |
| `DRAFTER=dflash2` vs `mtp` | +5–10% de decode | iguais (mesmo motor) |

> Os números "antes → depois" foram medidos em momentos diferentes do desenvolvimento (por isso a base de prosa 1 req varia: 38,9, 44,2, 47,2, 48,36…); eles medem o efeito **incremental** de cada knob, não somam linearmente até os 60,4 do sparkDash (que usa outra carga e clocks travados). **(interpretação)**

### 4.8 Comunicação

| Medição | Valor | Fonte |
|---|---|---|
| All-gather RoCE de 16 KiB | 11 µs (NCCL: 45 µs) | README / config.sh |
| All-gather de decode NCCL com 4 canais | 32 µs (80 com defaults) | nodes.sh |
| All-gather de chunk de prompt (32 MB) | 1,9 ms em 2 trilhos; 3,5 ms em 1 | nodes.sh |
| (terceiros, jayleaton) transporte puro por troca | 10,8 µs; espera por desalinhamento 1,3–1,5 ms/rodada (1 stream), 2,2–3,0 ms (4 streams) | THEORY-2 |

### 4.9 Carga e boot

- Partidas seguintes: **2 a 6 min** para carregar ~80 GiB por Spark; a primeira partida de uma imagem compila kernels CUDA (alguns minutos) em `~/.cache/tensorfold-glm53/<hash>`.
- NFS: ambos os ranks no ar em ~2,2 min.
- O `start.sh` mostra a cada 15 s o tempo decorrido e quanto da estimativa já está na GPU de cada Spark (`nvidia-smi --query-compute-apps` filtrado pelos PIDs do container).

### 4.10 Visão (v1.1)

50 imagens numa requisição: 12,5 s, 16.239 tokens de prompt, menor livre no head 6,75 GiB; 4 vídeos: 6,90 GiB.

### 4.11 Números do TensorFold upstream para o GLM (contexto; outro checkpoint)

**(TF, `glm-5.3-flash.md`)** Dois DGX Sparks, checkpoint **MLX 4-bit**, só MTP, `--context 262144`, prompts frios com um fato escondido: 32.770 tokens → 1.138 tok/s de prefill, TTFT 29 s, decode 50,9 tok/s; 131.074 → 898 / 146 s / 47,2; 261.906 → 849 / 309 s / 31,9 (fato achado em todos). Pico torch 93,7 GiB contra estimativa de 96,1 GiB; `MemAvailable` nunca abaixo de 9,5 GiB. Prompts curtos com DFlash2 + `auto`: 53,8 / 43,4 / 85,0 / 50,0 tok/s (código e chat, sampled e greedy). **(interpretação)** Comparado com a receita da Mia (EXL3 + patches: prefill ~1,9k tok/s, prosa greedy 60,4 tok/s), dá a ordem de grandeza do que os patches e o checkpoint EXL3 acrescentaram — checkpoints, drafters e cargas diferentes, então não é um A/B.

### 4.12 Incertezas e divergências encontradas

1. **HumanEval:** 97,6% (README, KV FP8 + q4) vs 95,1% (comentário do `config.sh`, comparação q4 × fp8). Corridas diferentes; o README não diz qual build/seed.
2. **NCCL por gather pequeno:** 45 µs (README, 16 KiB) vs 32 µs (nodes.sh, "decode-sized", 4 canais). Tamanho/configuração possivelmente diferentes; não reconciliado.
3. **Comentário desatualizado no `start.sh`:** lista "`TF_GLM_MULTI_LONE (1)`" entre os knobs de decode, mas o default no `config.sh` é `0` desde v1.3.1.
4. **Defaults de patch × receita:** as docstrings de `multi_tune.py` (TF_GLM_MULTI_LONE "1 (the default)") e `multi_prefill.py` ("0 (default)") descrevem os defaults **do patch**; a receita define 0 e 1, respectivamente.
5. **"`fp8` keeps it, at ~10% decode speed"** (README sobre #18): leitura correta é "custa ~10% de velocidade de decode" (o `config.sh` mostra q4 ~12–14% mais rápido que fp8).
6. **2.922.496 tokens vs orçamento:** não consegui reconciliar exatamente com 88,09 GiB de estimativa + teto de 12,5 GiB (§2.5).
7. **Pool "a Spark":** a descrição de que o teto é "por Spark" e o código (latente sem divisão por `world`) indicam pool replicado nos dois ranks (interpretação consistente, mas não afirmada literalmente).
8. **Comparação vLLM × TensorFold:** nenhuma das duas receitas publica um A/B controlado entre motores; os números de terceiros (jayleaton, LimeChain) usam outros checkpoints/motores-base.

---

## 5. Conceitos necessários e onde cada um entra no Latent Atlas

### 5.1 Mapa conceito → capítulo(s) existente(s)

| Conceito (TensorFold) | Seção aqui | Capítulo(s) do Atlas (id) |
|---|---|---|
| Motor por família vs motor genérico | 2.1 | `vllm-arquitetura`, `engenharia-de-receita`, `receita-glm` |
| Exatidão: rascunho = serial, concorrente = solo, retomado = fresco | 2.1, 2.9 | `geracao-amostragem`, `especulacao-fundamentos`, `medindo-qualidade`, `numeros-no-computador` |
| Não-associatividade fp32, soma em ordem de rank, kernels "row-independent" | 2.3, 2.9 | `numeros-no-computador`, `kernels`, `paralelismos` |
| Amostragem com chave (SHA-256 do prompt, splitmix64, Gumbel-max, float64 no host, massa em ponto fixo 2^40) | 2.9 | `geracao-amostragem`, `numeros-no-computador` |
| Um rank por nó, rendezvous `TCPStore`, NCCL via ctypes, "rank 0 decide / rank 1 aplica", doorbell | 2.2 | `nccl-rdma`, `vllm-arquitetura`, `paralelismos` |
| TP por all-gather de parciais fp32 (sem all-reduce); cabeça dividida por vocabulário | 2.3 | `paralelismos`, `tokens-embeddings` |
| Orçamento `MemAvailable` − reserva; congelamento em UMA; recusa antes de carregar; refit | 2.4 | `memoria-unificada`, `paged-attention-kv` |
| Pool compartilhado: arena, planos, extents de 2.048, slots, first-fit, compactação, despejo LRU | 2.5 | `paged-attention-kv`, `scheduler-batching` |
| KV FP8 e4m3 com escala potência de 2 por linha (528 B) | 2.6 | `kv-cache-quantizado`, `fp8-fp4-microescala` |
| Cache latente NoPE + absorção; limite denso 2.051; pools visíveis; radix select | 2.7 | `mla`, `atencao-esparsa`, `posicao-rope` |
| Estados de prompt guardados, grade de 64, prefixo compartilhado, replay com linha de logits | 2.8 | `prefix-caching`, `atencao-linear` |
| Rodadas, janelas de verificação até 16/32 linhas, graphs por largura, paridades do KDA | 2.10 | `especulacao-fundamentos`, `cuda-graphs-warmup`, `atencao-linear` |
| DFlash2 com janela deslizante em anel; seletor com ruído do alvo; políticas `fc`/`fnc`/`fcost`; β = 2,0 | 2.11 | `dflash-dspark`, `especulacao-fundamentos` |
| Copy drafts (prompt lookup) | 2.11 | `especulacao-fundamentos`, `mtp-eagle-vocab` |
| Simulador offline de políticas a partir de dumps teacher-forced | 2.11 | `especulacao-fundamentos`, `benchmarking` |
| Scheduler multi-stream (fill vs round, FILL_SHARE), multi-prefill, profundidade conjunta | 2.12 | `scheduler-batching`, `prefill-decode-kv` |
| Kernels EXL3 de prompt (NFIRST, ordem pesado/leve) | 2.13 | `hadamard-trelica-exl3`, `kernels`, `mixture-of-experts` |
| KDA chunked (WY/UT), triangular inversa, tf32x3, smem 99 KiB do GB10 | 2.13 | `atencao-linear`, `gpu-por-dentro`, `tensores-algebra` |
| Divisão das hyper-connections por linhas entre ranks + overlap | 2.13 | `hiperconexoes`, `paralelismos` |
| RDMA one-shot (b12x RoCEnante), dois trilhos, GID RoCE v2, barreira de partida | 2.14 | `nccl-rdma` |
| Pesos densos 4-bit com faixa por MSE; FP8 por bloco de 128 | 2.15 | `quantizacao-fundamentos`, `fp8-fp4-microescala` |
| L2 prefetch em stream lateral; loads `ld.global.nc` | 2.15 | `gpu-por-dentro`, `kernels` |
| Visão no rank 0, limites de imagens/vídeo | 2.16 | `multimodal` |
| API: tool calls inteiras, recuperação de histórico, reinserção de raciocínio, xgrammar com rascunhos, `context_length_exceeded`, `/tokenize`, `/metrics`, campos recusados | 2.17 | `api-tools-estruturado` |
| Imagem por hash de conteúdo, digest pin, GHCR, revisões pinadas, NFS, cache de kernels por imagem | 2.18 | `engenharia-de-receita`, `checkpoints`, `memoria-unificada` |
| Medição com clocks travados, sparkDash, A/B por boots | 4.1 | `benchmarking` |

### 5.2 O que acrescentar em cada capítulo existente (por id)

> Regra (AUTHORING.md): ciência primeiro, receita como exemplo; nunca inventar números; rotular interpretações. Proponho um **novo tag de receita `glm-tf`** (ou `tf`) no campo `recipes` dos capítulos tocados, para distinguir da receita vLLM `glm`.

**`mapa-da-jornada`**
- Acrescentar uma 4ª linha na tabela das receitas: "GLM-5.3-Flash EXL3 + **TensorFold** v0.6.0 + 53 patches" — mesmo checkpoint; motor não-vLLM; TP=2 (um rank por Spark); KV FP8 e4m3 com escala potência de 2 (528 B/token/camada DSA); DFlash2 até 7 (`fnc7:0.3`) + copy drafts até 15; contexto 1.048.576; 4 requisições; pool 2.922.496 tokens; prosa 60,4 tok/s (1 req) e 108,8 (4 req, agregado); structured 114,7 / 227,9; prefill ~1.950 tok/s até 64k.
- Uma frase de enquadramento: "o motor também é uma escolha de receita" — a mesma GPU e os mesmos pesos dão números bem diferentes conforme o motor (comparar 36,1 → 60,4 tok/s em prosa, com a ressalva de condições).

**`aprender-com-dados`, `redes-neurais`, `atencao`, `bloco-transformer`** — nada essencial. Opcional em `bloco-transformer`: citar que um motor dedicado escreve a forward camada a camada à mão (KDA, DSA, MoE, cola HC), sem grafo genérico.

**`tensores-algebra`**
- Exemplo de **associatividade como otimização**: absorção NoPE `score_h = (Wk_h^T q_h)·c` e `out_h = Wv_h(Σ p_j c_j)` — reordenar produtos para nunca materializar K/V por cabeça.
- Inversa de matriz **triangular inferior unitária** por substituição progressiva (`T = (I + L)^-1`), feita em blocos de 16 (`T_bj = −T_bb Σ L_bm T_mj`): exemplo concreto de álgebra linear "de verdade" num kernel de prefill.

**`numeros-no-computador`**
- **Soma em ordem de rank:** por que um all-reduce comum pode dar bits diferentes nos dois ranks e o TF faz all-gather + soma "rank 0 first".
- **Kernels sem contração FMA** (`--fmad=false`, `enable_fp_fusion=False`), `expf` do CUDA e divisão IEEE para reproduzir bits de outro kernel; empate de arredondamento bf16 que move uma chave em 2^-8.
- **TF32 vs `tf32x3`**: ~5e-4 vs ~2e-6 de erro relativo a float64 (TF32 "half a bf16 step").
- **Escala potência de dois** multiplica sem erro (só mexe no expoente); escala SHIFT 8 coloca o máximo da linha em [128, 256] < 448.
- **Ponto fixo de 40 bits** para massas de probabilidade: somas de shards exatas → duas GPUs = uma GPU.
- **Hash splitmix64** como gerador de números aleatórios **indexável** (sem estado): `u = f(seed, posição, token)`.
- **Proteção contra overflow de expoente** no KDA chunked: fatores de referência para manter e^x ≤ e^75 < e^88,7 (limite fp32).

**`tokens-embeddings`**
- Cabeça (lm_head) **dividida por vocabulário** entre ranks (`[R, V/world]`) e a troca de **candidatos** (top-k + margem 8; até 1.024 para o núcleo) em vez de logits inteiros.
- Cópia **4-bit da cabeça só para rascunhar**; a verificação usa a cabeça original (no modo q4 a cabeça fica FP8).
- `/tokenize` e `/detokenize` com campos do vLLM (`count`, `max_model_len`) — contar tokens no servidor.

**`posicao-rope`**
- NoPE na prática: sem RoPE, o cache guarda só os 512 do latente — o TF evita o padding 512→576 que o kernel `fp8_ds_mla` do vLLM exige (528 vs 656 B/token/camada). Exemplo de como a escolha de posição muda bytes de KV.

**`geracao-amostragem`** (prioridade alta)
- Nova seção **"Amostragem com chave"**: semente = SHA-256 dos ids do prompt quando não há `seed` ("a mesma requisição dá a mesma resposta"); uniforme de **(semente, posição absoluta, id do token)** via splitmix64; Gumbel-max `argmax(logits/T + G)`; ordem de cortes top-k → top-p → **min-p depois do top-p** ("as SGLang does (vLLM applies `min_p` first)"); escolha em float64 no host.
- Consequência: **reprodutibilidade de sequência** sem desligar a amostragem; base de testes por hash (`token_sha`).
- Núcleo top-p distribuído: 1.024 candidatos por rank, shard inteiro se o núcleo passa disso, massa em ponto fixo 2^40.
- Defaults do checkpoint GLM (T 1,0, top_p 0,95); `temperature: 0` = greedy; penalidades e `logit_bias` **ignorados** no TF; `logprobs` recusado — trade-off do motor.
- Mapeamento de `reasoning_effort` no TF: `medium`→`high`, `xhigh`→`max`, sem esforço = Max do template, `none` desliga; `MAX_TOKENS` default 32.768 (TF 4.096) para não cortar raciocínio/ferramentas.

**`prefill-decode-kv`**
- **"Rodada"** de decode especulativo (propor → verificar janela → amostrar → aceitar → commit) como unidade de tempo do decode moderno.
- Custo de uma forward curta é quase fixo: "~0.12 s on two Sparks, the weights read once a layer whatever the rows" (motiva o multi-prefill); custo de janelas de verificação 29,2 ms (1 linha) → 66,0 ms (8 linhas) (medição antiga do `draft_sim`).
- Números de TTFT/prefill da receita TF (tabela 4.3) como exemplo real de prefill limitado por compute que cai com o contexto (1.979 → 1.642 → 1.015 tok/s).
- Bytes por token do KV na receita TF (≈ 564 B por camada DSA em FP8, 11 camadas DSA) como exemplo de cálculo (rotular como estimativa a partir do código).

**`mixture-of-experts`**
- Na execução do prompt, experts com **um** item (lidos da DRAM, poucos mma) e com **vários** (pesos reaproveitados via L2) têm perfis diferentes; intercalá-los mantém DRAM e tensor cores ocupados juntos (patch 0020). Bom exemplo de "desbalanceamento de carga importa na inferência".
- Expert compartilhado executado **antes** dos roteados quando há L2 prefetch, para não ser expulso do L2 pelo fluxo de ~100+ MB dos roteados.

**`mla`**
- Implementação do TF: cache **só do latente `c`** (512, "1 KB a token and layer" em bf16), query absorvida uma vez por linha, expansão do latente atendido uma vez; latente **replicado** nos dois ranks (interpretação a partir de `mla_geometry`).
- `kv_b` dividido: a metade de valor é pré-carregada no L2 antes da projeção de saída; com `DENSE=q4`, `kv_b` fica BF16.
- Comparação de layouts: TF FP8 528 B vs vLLM `fp8_ds_mla` 656 B por token/camada.

**`atencao-esparsa`**
- **Limite denso 2.051** = 512 pools × 4 tokens + pool incompleto (até 3): até aí atenção densa; depois, top-512 pools + pool incompleto.
- Seleção por **radix select** (8 bits por passada) num kernel; "buckets" de potência de dois de pools para os CUDA graphs.
- **Pools visíveis** (patch 0043 / PR #140): pontuar só `min(NPS, max((P+r+1)//4, K))` colunas por linha — mesmos tokens, menos trabalho.
- Seleção do prompt em blocos de 512 linhas para limitar a memória de scores fp32 a 1M.
- Chaves agrupadas do indexador também em FP8 (144 B por pool de 4).

**`atencao-linear`** (prioridade alta)
- **Forma chunked WY/UT do KDA** com as fórmulas de L, M, T = (I+L)^-1, W, U, V', O, S_end (docstring do patch 0012) — já há "forma paralela por chunks" no escopo; acrescentar o **detalhe matemático** e a figura de blocos.
- **Sub-chunks em posições absolutas** (64 na versão Triton, 32 no kernel CUDA) e por que isso exige que os chunks de prompt comecem em múltiplos de 32 → **grade de 64** para estados guardados.
- Kernel CUDA: um bloco de 256 threads por cabeça, estado 128×128 em registradores, somas fp32 em ordem fixa: determinismo; "close to the serial kernel, not its bits".
- Números: 50k 29,3 → 26,4 s; 149k 91,1 → 84,5 s.
- **Rollback do estado recorrente** com duas paridades e replay das linhas aceitas (alternativa ao "guardar num_spec+1 estados" do vLLM).
- Consequência para cache: estado de prompt guardado **em pontos** (cada um ~45 MiB fixos com tudo que acompanha).

**`hiperconexoes`**
- Os 4 fluxos residuais e a "cola" `hc_pre`/`hc_post` como custo **latency-bound** (terceiros: 90 fronteiras × 15,6 µs por rodada no motor do jayleaton).
- **Divisão por linhas entre ranks** no prefill (rank r possui metade das linhas; troca parciais fp32 e linhas normalizadas bf16), sobreposta com a frente do próximo bloco: 50k ~1.270 → ~1.730 tok/s, mesmos bits.

**`ngram-mtp`**
- Cabeça MTP como drafter alternativo no TF: 1,77 GiB por Spark; `TF_GLM_MTP=auto/1/0`; `DRAFTER=mtp` evita a licença do DFlash2 mas serve uma requisição por vez; política `auto` que compara drafters por tokens/ms e sonda periodicamente.

**`multimodal`**
- Torre visual do GLM só no rank 0 (1,05 + 0,75 GiB), conferida bit a bit contra transformers 5.17; limites (50 imagens / 4 vídeos; 2.048 tokens por imagem e 16.384 por requisição; vídeo 2 fps, 128 quadros, 16.384/32.768 tokens); corpos de 96 MiB; prompts com imagens não fazem replay nem multi-prefill.

**`quantizacao-fundamentos`**
- Exemplo de **busca de clipping por MSE**: 4-bit afim em grupos de 64, faixa = a de uma grade de cortes (1,0 até 0,55 do min/max) com menor erro quadrático (`DENSE=q4`).
- **Lição de avaliação:** q4 passa GSM8K/HumanEval em inglês, mas perde o fim de turno em prompts curtos em francês (12/48 vs 1/48) — benchmarks monolíngues escondem regressões; calibrar (o projeto trabalha num 4-bit calibrado).
- Mistura de precisões numa receita: experts EXL3 4 bpw + densos q4 + cabeça FP8 + `kv_b` BF16 + KV FP8.

**`fp8-fp4-microescala`**
- **FP8 por linha com escala potência de 2** (KV): por que não perde precisão relativa frente a uma escala fp32 fina (só empurra para subnormais valores < 2^-14 do máximo da linha); 2,65% de erro relativo por linha, igual (±0,001) a escalas por token/bloco/camada; nada satura (máximo em [128, 256] < 448). Comparar com ue8m0 do DeepSeek-V3.2.
- **FP8 de pesos** por linha e bloco de 128 colunas (absmax/448) no `DENSE=fp8`.

**`hadamard-trelica-exl3`**
- O TF lê EXL3 em CUDA por um módulo compartilhado (qualquer codebook — 3inst, mcg, mul1 — e largura 1–8 para outras famílias); para o GLM, só mcg 4-bit nos experts roteados ("experimental").
- Kernels de prompt NFIRST (2,3–2,5× a 4.096 linhas) e ordem pesado/leve; decodificação da treliça com PRMT; loads de 16 B `ld.global.nc.L1::no_allocate` um passo à frente no decode (+~1–1,5% sozinho).

**`kv-cache-quantizado`**
- Formato `kv8` completo (códigos e4m3 + fp32 + 12 B de pad; um helper `store_row` para prefill e decode → exatidão preservada); janela bf16 196.608 vs FP8 1.048.576 com 4 streams; qualidade com KV FP8 (GSM8K 98,0%, HumanEval 97,6%, agulha a 981.841 tokens).
- Comparação com `fp8_ds_mla` (656 B) da receita vLLM.

**`checkpoints`**
- **Pins de revisão** por sha e serviço offline (`HF_HUB_OFFLINE=1`); layout do cache HF (`snapshots/`, `blobs/`, `refs/main`); download por sha não grava `refs/main`; **backend xet**: blobs que são links para o `blobs/` da raiz (#15) → `rsync -L`; verificação **arquivo a arquivo** (caminho + tamanho); `tensorfold info` para validar antes de copiar.

**`medindo-qualidade`**
- **Testes de exatidão** como gate de qualidade: `"draft": false` vs rascunhado (6/6), concorrente vs solo (22/22; 11/11 escalonado + 11/11 em rajada), retomado vs fresco, hashes de referência, imagens/vídeos e tool calls.
- Distinguir "exato contra o mesmo motor" de "exato contra o checkpoint bf16" (`DENSE=bf16 KV=bf16 KDA_CHUNKED=0`).
- Agulha de 981.841 tokens; `tools/needle.py` (~195k, 60% de profundidade); regressão multilíngue (#18).

**`gpu-por-dentro`**
- GB10: **99 KiB de shared memory por bloco** (101.376 B) — kernels Triton do KDA com um único estágio de pipeline; `sm_121`.
- **Fases ociosas de DRAM** num passo de decode (all-gathers, cola, router, recorrência KDA) e **L2 prefetch** num stream lateral (`cp.async.bulk.prefetch.L2`) para escondê-las; 8 MiB por ponto.
- Clocks travados (2.200 MHz) como condição de medição.
- (terceiros) kernels grandes a 205–217 GB/s contra teto de leitura de 235 GB/s.

**`memoria-unificada`** (prioridade alta)
- Orçamento do TF: `MemAvailable` − reserva (`max(4 GiB, RAM/10)` ou `TENSORFOLD_MEMORY_RESERVE_GIB`; 14,5 na receita); page cache recuperável conta como disponível no GPU integrado.
- **"running out tends to freeze the machine rather than fail an allocation"**: recusar antes de carregar, informar a maior janela que cabe, refit automático do `start.sh`.
- `docker run --memory` **não limita** alocações da GPU no GB10.
- A reserva cobre CUDA context, NCCL e graphs (3,5–4,8 GiB medidos no doc do TF); pico ~10 GiB acima da estimativa num prompt de 1M; números de `MemAvailable` ocioso/mínimo (7,0/10,5; 5,6/9,5 GiB); ~110 GiB livres exigidos na partida.
- NFS via **volume docker** (`nfsvers=4.2,ro,nconnect=8,rsize/wsize=1 MiB,hard,timeo=600`), sem sudo no worker.

**`kernels`**
- **Kernels "row-independent"** como restrição de projeto (uma linha tem os mesmos bits em qualquer lote/janela): split-K fixado pela forma do peso, tiles fixos; o que pode depender da contagem de linhas (seletor do cuBLAS, `torch.topk`) roda por stream.
- **Kernels segmentados** para lotes multi-stream (tabelas `SegRows` por linha/segmento no device).
- JIT de extensões CUDA C++ e Triton na primeira partida, cache por hash de imagem.
- L2 prefetch e loads não-coerentes como exemplos de engenharia de memória; RoCE all-gather como kernel que espera flags de sistema.

**`cuda-graphs-warmup`**
- Graphs **por largura de janela** (1..8 + 16 em stream único; 1..32 no lote) e **tabelas no device** para posições/extents/slots → um graph por contagem de linhas cobre qualquer mistura.
- Captura no "home" (slot 0, primeiras linhas do pool) **antes** de admitir streams.
- **Tokens de calibração distintos** (patch 0048): um token repetido roteia para os mesmos experts e mede uma janela de 16 linhas pela metade do custo.
- Época no device para que a coletiva RoCE seja reproduzível em graph; barreira NCCL na partida porque os ranks compilam em tempos diferentes.
- Graph vs eager com os mesmos bits (`TF_GLM_MULTI_GRAPHS=0` como referência); (terceiros, motor do jayleaton) o `cudaGraphLaunch` de um graph de verificação de ~1.650 nós apareceu como ~1,1 ms de GPU ociosa por rodada, mas medido com rastreio por nó do nsys, que infla lançamentos — o custo real fica entre 0,1 e 1,1 ms.

**`vllm-arquitetura`**
- Caixa "**Nem todo motor é o vLLM**": o TF não tem EngineCore separado, block manager paginado, nem registro genérico de modelos — cada família escreve forward, cache e verificação; o scheduler do GLM é uma thread no rank 0 que emite operações (ADMIT/EVICT/MOVE/GROW/FILL/ROUND/FINISH/IDLE/SLOT) aplicadas igualmente pelos dois ranks. Ligar ao cap. de engenharia de receita (patches no `site-packages` vs overlays).

**`paged-attention-kv`**
- **Alternativa ao PagedAttention:** extents contíguos alinhados a 2.048 tokens num pool compartilhado; first-fit pela base mais baixa; crescer no lugar, mover, **compactar**; despejar prompts guardados (LRU); folga de 2.048 por stream decodificando na admissão.
- Orçamento do pool: `MemAvailable` − reserva, teto `KV_POOL_GIB` por Spark além da janela; 2.922.496 tokens; `/health` com `pool_tokens` / `pool_free_tokens`.
- Comparar com a receita vLLM: 11 GiB fixos → 1.233.779 tokens a 850k.

**`scheduler-batching`**
- **Iteração do TF:** admissões, depois um chunk de prompt **ou** uma rodada de decode, alternados por `FILL_SHARE` (0,5); chunk ≤ 1.024 linhas quando outros decodificam; prefill e decode **nunca** na mesma forward (contraste com chunked prefill misto).
- **Multi-prefill:** vários prompts num chunk, trabalho compartilhado uma vez (TTFT ×4 590 → 340 ms).
- Lote de verificação ≤ 32 linhas com corte das caudas mais longas; políticas de profundidade `policy`/`joint`/`scale:A`.
- Prioridade `background` com replay (resposta = solo); espera em ordem quando o pool não cabe (`NoRoom`).

**`prefix-caching`** (prioridade alta)
- Seção **"Reuso sem APC: estados de prompt"**: estado no fim do prompt (ou último múltiplo de 64), prefixo compartilhado (ponto de divergência e fim do bloco de sistema na primeira `<|user|>`; ≥ 256 tokens; ≤ 2 por requisição), **replay** com a linha de logits guardada.
- Números: 64k idêntico 34 s → <0,07 s; system 7,9k 4,24 → 0,13 s; contraste com o APC do vLLM (8,53 → 1,30 s num follow-up de 8k; 2,748 s para reenviar 814.571 tokens; acertos alinhados a 3.584) e o recálculo de milhares de tokens no replay híbrido (terceiros).
- Limites: `TF_GLM_CACHE_ENTRIES` (8 → 32, #17), ~45 MiB por estado, despejo LRU, o caso `MULTI_LONE` que destruía o cache (#12, #13).

**`api-tools-estruturado`**
- Tool calls **enviadas inteiras** (deltas vazios a cada 2 s); resposta cortada dentro de chamada termina em `length` sem enviar a chamada; chamadas no bloco de pensamento contam se a resposta termina nelas; **reinserção do raciocínio** de passos de ferramenta descartado por clientes; histórico com argumentos não-objeto **omitido** em vez de 400 (#2).
- xgrammar com rascunhos: cortar rascunhos rejeitados antes da verificação + mascarar linhas → restrito = serial.
- `context_length_exceeded` com `param`; `/tokenize`; `/metrics` com `tensorfold_health:*`; recusados (`logprobs`, `n≠1`) e ignorados (penalidades, `logit_bias`); sem `/v1/messages`; sem autenticação.

**`especulacao-fundamentos`** (prioridade alta)
- **Verificação exata por comparação** (com amostragem com chave) vs **rejection sampling**: o primeiro preserva a sequência, o segundo a distribuição.
- O rascunhador pode **ver o ruído do alvo** antes da verificação (ruído com chave) → confianças "noise-aware".
- Políticas de parada: produto de confianças (`fc`), versão com ruído (`fnc`), custo-benefício `E[tokens]/ms` (`fcost`) com tabela de custos medida na partida.
- **Copy drafts** (prompt lookup): MATCH 8, até 15 por rodada, 16 tokens para copiar da própria resposta; números de edição/citação.
- **Simulador offline** a partir de dumps teacher-forced (`TF_GLM_DRAFT_DUMP`, `draft_sim`, oracle) — só possível porque a amostragem é com chave.
- Janelas até 16 linhas por stream, 32 no lote; rollback do estado recorrente por paridades + replay.

**`mtp-eagle-vocab`**
- Política `auto` do TF (MTP vs DFlash2 por tokens comprometidos por tempo estimado de rodada, sondando o outro periodicamente); políticas MTP `cN:P`, `a:LOW:HIGH`; cópia 4-bit da cabeça para rascunho.

**`dflash-dspark`**
- DFlash2 no TF: janela deslizante de 2.048 com **anel** de 2.176 linhas (21 MiB/rank em vez de 10 KiB por token da janela; "8% faster at 52k, and 3.8 GiB lighter a rank"); modelo 0,63 GiB/rank; seletor `EDGE=0.6`, `NOISE=0.7`; confiança `softmax((score + g)/β)` com β = 2,0 ajustado em 9 dumps (2026-09-30); `fnc7:0.3` +7% prosa/código sobre `fc5:0.3`; drafter multi-stream (uma sequência de lançamentos para vários streams, cada um com os bits do seu drafter solo); janela do DFlash2 salva com o estado de prompt; licença CC BY-NC-ND.

**`paralelismos`**
- TP do TF: parciais fp32 + **all-gather** + soma em ordem de rank (sem all-reduce) — exatidão entre ranks; cabeça por vocabulário; cache latente replicado.
- Divisão da cola HC por linhas como forma de *sequence parallelism* restrita.
- Motor CUDA do GLM exige **exatamente 2 ranks**.

**`nccl-rdma`** (prioridade alta)
- **RDMA one-shot (b12x RoCEnante)**: memória pinada do host endereçável pela GPU no GB10 (sem GPUDirect RDMA), proxy em C com libibverbs, faixas por HCA seguidas de flag `seq` no mesmo QP, época no device, dois slots; 11 µs vs 45 µs por 16 KiB; +6% de decode; `TF_ROCE_MAX_KB=512` = 32 linhas × 16 KiB.
- **Dois trilhos** (a porta QSFP aparece como duas funções PCIe Gen5 x4): 32 MB em 1,9 ms vs 3,5 ms.
- **GID RoCE v2** com IPv4 mapeado; entrada v1/zerada derruba o NCCL "about a minute in"; `NCCL_MIN/MAX_NCHANNELS=4` (32 µs vs 80 µs).
- Rendezvous por `TCPStore` + `ncclUniqueId`; NCCL via ctypes; **doorbell** do rank ocioso na store.
- **Deadlock** por escolha de transporte dependente de endereço (patch 0040) → regra "eligibility is rank-invariant"; barreira NCCL até o motor estar pronto (#6); `TF_ROCE_WAIT_S`.

**`tp3-tp4`**
- Nota: o motor GLM do TF só roda com 2 ranks; quem precisa de 3–4 Sparks fica com a receita vLLM (trade-off de motor).

**`engenharia-de-receita`** (prioridade alta)
- **Patches como diffs unificados aplicados no build** (`patch -p0 --forward` em ordem de nome, falha = build falha, import de verificação) vs overlays reaplicados a cada start.
- **Hash de conteúdo da imagem** (patches + extras) → tag; **digest pin**; label `tf.patches` conferido no pull; publicação no GHCR com `DOCKER_CONFIG` descartável e labels OCI.
- Identidade de imagem **por conteúdo** entre containerd e overlay2 (#8).
- Precedência ambiente > `local.sh` > `.env` (lido, nunca executado) > defaults; `.env` e `local.sh` são do usuário.
- **Validar argumentos com o parser do próprio motor** num container descartável antes de parar o servidor; marcador de preparo; `flock`; refit da janela; smoke test (thinking off, greedy, 32 tokens, "Reply with OK."); `FOREGROUND=1` para systemd; cache de kernels por hash.
- Upstreaming: três patches da Mia viraram TF v0.6.0 e foram removidos (como aposentar hacks).

**`receita-glm`**
- Acrescentar uma seção "**Variante TensorFold**" (ou criar capítulo próprio, ver 5.3) com o passo a passo da §6 deste documento e a tabela da §7; destacar o que **não** muda (checkpoint, hardware, cabo, disco) e o que muda (motor, KV, reuso, rede, exatidão).

**`receita-qwen`, `receita-deepseek`**
- Nota curta: o TF v0.6.0 também serve Qwen3.8 Flash Next (NVFP4/EXL3, um rank ou dois) e o DeepSeek-V4-Flash só no MLX; mostra que a escolha de motor é por modelo.

**`benchmarking`**
- **Clocks travados** (2.200 MHz) como controle; medir **de outra máquina** pela API (sparkDash); A/B por "two boots each"; gates de exatidão (hashes) junto com velocidade; `/health` (contadores para diferenciar em taxas) e `/metrics`; distinguir greedy (sparkDash temp 0) de sampled; perigo de comparar receitas com condições diferentes (§1.2).
- Tokens de calibração distintos para medir janelas (0048).

**`receita-do-zero`**
- Critério de decisão "**qual motor?**": quando um motor dedicado compensa (arquitetura exótica, 2 nós, foco em single-stream/agentes, necessidade de reprodutibilidade) vs quando o vLLM vence (superfície de API, escala além de 2 nós, várias arquiteturas, ecossistema, prefill frio em alguns casos — terceiros).

**`laboratorio`**
- Opção "motor: vLLM | TensorFold" que troca bytes/token do KV (656 vs 528 B por camada DSA), a regra de orçamento (`gpu_memory_utilization` vs `MemAvailable − reserva`), o teto de concorrência (4 no TF GLM) e o alerta "TF GLM só com 2 nós".

### 5.3 Conceitos sem casa natural (proposta)

- **Proposta de novo capítulo na Parte 6:** `motor-exato-tensorfold` — "Outro jeito de construir um motor: exatidão, estados e um rank por nó". Reúne 2.1–2.5, 2.8–2.12 (o que hoje ficaria espalhado). Pré-requisitos: `vllm-arquitetura`, `paged-attention-kv`, `prefix-caching`, `especulacao-fundamentos`, `nccl-rdma`.
- **Proposta de novo capítulo na Parte 9:** `receita-glm-tensorfold` — reconstrução passo a passo (§6) com a comparação (§7) e os números (§4); ou, alternativamente, uma seção longa dentro de `receita-glm`.
- **Glossário novo:** keyed sampling / amostragem com chave; kept prompt state; prompt grid; replay; shared prefix point; extent; slot; verify window; round; copy draft; noise-aware stop; doorbell; one-shot all-gather; rail; refit; image hash; digest pin.

---

## 6. Receita passo a passo com TensorFold (o que fazer e por quê)

**Passo 0 — Decidir se é o motor certo.** Use TensorFold se: exatamente 2 Sparks; até 4 requisições; quer contexto de 1M, decode single-stream alto e reuso de histórico (agentes); não precisa de `logprobs`/`n>1`/API key no próprio servidor; aceita os defaults com perdas (ou paga com `DENSE=bf16 KV=bf16 KDA_CHUNKED=0` e janela menor) e a licença não-comercial do DFlash2 (ou usa `DRAFTER=mtp`). Senão, receita vLLM.

**Passo 1 — Hardware e rede.**
- Dois DGX Spark (ou GB10 de 128 GB) **sem outra carga grande na GPU**: cada um precisa de ~110 GiB livres na partida (por quê: o orçamento parte do `MemAvailable` e faltar memória em UMA congela a máquina).
- Cabo QSFP entre as portas CX7; IPv4 nas duas pontas na mesma sub-rede privada (ex.: `192.0.2.1/24`, `192.0.2.2/24`), `ping` funcionando, com **GID RoCE v2**. Com as duas portas cabeadas e endereçadas, ambas são usadas (~1,8× no all-gather de prompt).
- ssh por chave do head para o worker: `ssh-keygen -t ed25519`, `ssh-copy-id user@<worker>`, testar `ssh -o BatchMode=yes user@<worker> true`.
- Docker com runtime NVIDIA, usuário no grupo `docker`, `rsync` nos dois.
- Disco por Spark ~205 GB (checkpoint 176 GB + DFlash2 2,3 GB + imagem ~25 GB); `MIN_FREE_GB=180`, `IMAGE_FREE_GB=35`. Com NFS, o worker só precisa da imagem.

**Passo 2 — Clonar e configurar (no head).**
```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git
cd GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh   # WORKER=user@192.0.2.2 (e FABRIC_PEER se o WORKER estiver noutra rede)
```
Por quê: `local.sh` e `.env` são do usuário (fora do git); o `config.sh` define todos os defaults e a precedência.

**Passo 3 — Preparar (o `start.sh` faz sozinho na 1ª vez; ou `scripts/prepare.sh`).**
1. Preflight nos dois Sparks: Docker, GPU (`nvidia-smi -L`), runtime nvidia, `rsync`, ssh por chave, detecção do link (rota → netdev → HCA em sysfs → índice do GID RoCE v2 com o IPv4), disco.
2. Imagem no head: tenta `docker pull ghcr.io/miaai-lab/...@sha256:22789f0c…` (digest pinado, enquanto o hash dos patches for `ae8d1c789b47`); confere o label `tf.patches`; senão constrói (`FROM nvcr.io/nvidia/pytorch:26.07-py3` → `pip install TensorFold@v0.6.0` + `av`, `xgrammar` → aplica os 53 patches → import de verificação). Por quê: reprodutibilidade (digest imutável) e build idempotente.
3. Imagem no worker: pull ou `docker save | ssh docker load`; conferência **por conteúdo** das camadas.
4. Download no head do checkpoint e do DFlash2 nas revisões pinadas (`hf download --revision …`, ou `snapshot_download` dentro do container); grava `refs/main` se faltar; `tensorfold info` valida o checkpoint antes de copiar.
5. Worker: `rsync -a -L --partial` dos blobs + links do snapshot; verificação arquivo a arquivo. Ou, com `WORKER_WEIGHTS=nfs`, export read-only do head (`/etc/exports` + `exportfs -ra`, uma vez, com root no head) e volume docker NFS no worker.
6. Grava o marcador de preparo.

**Passo 4 — Subir.** `./start.sh` (cinco passos visíveis):
1. Setup (se o marcador difere).
2. Checagens: argumentos pelo parser do TensorFold num container sem GPU (typo falha **antes** de parar algo); link; servidor anterior (parado em `restart` ou se só um rank estiver de pé; container parado é removido); porta livre (`ss -ltn`); memória livre (aviso < 110 GiB).
3. Lançamento: rank 1 no worker por ssh, depois rank 0 + API no head (flags e env da §3.7).
4. Carregamento: log do rank 0 filtrado; a cada 15 s, tempo e GiB na GPU de cada Spark vs a estimativa; se o rank 0 morrer dizendo a maior janela que cabe, **reinicia uma vez** com ela (`--context N`) e avisa.
5. Smoke test: uma chat completion pelos dois ranks (thinking off, `temperature: 0`, `max_tokens: 32`, "Reply with OK."), depois `GLM-5.3-Flash-EXL3 is now LIVE! on port 8888`.
Primeira partida de uma imagem compila kernels (alguns minutos); depois 2–6 min de carga.

**Passo 5 — Usar.**
```bash
curl -s http://<head>:8888/v1/models
curl -s http://<head>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000 }'
curl -s http://<head>:8888/health        # busy, streams, pool_tokens, pool_free_tokens
```
- Dê `max_tokens` suficiente (o modelo pensa em Max por default); `content` vazio = pensou até o limite → mais tokens ou `reasoning_effort: "low"`.
- Agentes: cliente com janela 1.048.576 e saída 32.768 (default do servidor); tool calls chegam inteiras.
- Reprodutibilidade: sem `seed`, a mesma requisição dá a mesma resposta; `"draft": false` para a referência serial.

**Passo 6 — Ajustar por carga de trabalho (com o porquê).**
- Prompts não-ingleses curtos → `DENSE=fp8` (custa ~10% de decode; evita #18).
- Exatidão contra o checkpoint → `DENSE=bf16 KV=bf16 KDA_CHUNKED=0` (janela 196.608 com DFlash2).
- Uso comercial → `DRAFTER=mtp` (antes do 1º start, para nunca baixar o DFlash2; `PARALLEL=1`).
- Outra carga dividindo os Sparks → aumente `MEMORY_RESERVE_GIB` (e/ou reduza `CONTEXT`/`PARALLEL`).
- Muitas conversas alternando (agentes) → `TF_GLM_CACHE_ENTRIES` maior (cada estado ~45 MiB na partida); mantenha `TF_GLM_MULTI_LONE=0`.
- Só texto → `VISION=0` (devolve ~1,8 GiB do rank 0 ao cache).
- Disco curto no worker → `WORKER_WEIGHTS=nfs`.
- Aplicar: `./start.sh restart` (ou `VAR=x ./start.sh restart`; extras: `./start.sh restart --max-tokens 16384`).

**Passo 7 — Verificar e medir.**
- `API_URL=http://<head>:8888 tools/needle.py` (agulha a ~195k tokens; argumento `size` para maiores) e `tools/toolcheck.py` (array tipado).
- Exatidão: mesma requisição com e sem `"draft": false`; 4 concorrentes vs sozinhas; reenviar e comparar.
- Velocidade: sparkDash pela API de outra máquina; travar clocks para comparar corridas; registrar imagem (hash/digest), revisões e data.
- Logs: `docker logs -f glm53-flash-tf` (rank 0), `ssh <worker> docker logs -f glm53-flash-tf` (rank 1). Parar: `./stop.sh` (corta requisições em curso; avisa quantas).

**Passo 8 — Manter.** Mudou `patches/`? O próximo `start.sh` reconstrói ou puxa a tag do novo hash; `scripts/publish-image.sh` publica (`v0.6.0-<hash>` e `latest`) e imprime `IMAGE_TAG`/`IMAGE_DIGEST` para fixar. Nova versão do TensorFold exige rebase dos patches e `prepare.sh --rebuild`.

---

## 7. vLLM × TensorFold: decisões de projeto e trade-offs

| Decisão | vLLM (receita GLM da Mia) | TensorFold (receita v1.3.2) | Trade-off |
|---|---|---|---|
| Escopo do motor | genérico (registro de modelos, plugins de quantização, backends de atenção) | um motor por família (`glm5_next`) | TF: kernels sob medida, sem padding, controle total; vLLM: amplitude, comunidade, features |
| Modificação | overlays Python reaplicados a cada `docker run` + patches de build | 53 diffs aplicados no build da imagem, versão fixa | TF: imagem imutável e hasheada; ambos: rebase a cada upstream |
| Processos | API + EngineCore + workers (`mp`), `--headless` | um processo por rank; rank 0 decide e envia uma mensagem por iteração | TF: simples e determinístico; vLLM: isolamento API/engine |
| Garantia de saída | sem perdas **em distribuição** (rejection sampling); bits mudam com lote/especulação | **bits** iguais: rascunhado = serial, concorrente = solo, retomado = fresco | TF: testes por hash e reprodutibilidade; custo: kernels restritos (linha independente) |
| Amostragem | GPU, por passo | candidatos da GPU, escolha com chave em float64 no host | TF: exatidão entre ranks/lotes; custo de host a 4 streams (terceiros: ~2,3 ms/rodada) |
| TP | all-reduce NCCL | all-gather de parciais fp32 + soma em ordem de rank | TF: bits iguais nos ranks; mais bytes no fio (pequenos via RDMA) |
| Comunicação pequena | NCCL (host-staged no GB10) | RDMA one-shot (11 µs/16 KiB) até 512 KiB | TF: +6% decode; custo: código de transporte próprio, modos de falha (envenenamento, barreiras) |
| KV DSA | `fp8_ds_mla` 656 B (latente com RoPE zerado) | FP8 e4m3 + escala 2^k, 528 B | TF: −19,5% de bytes de latente, escala exata; vLLM: kernel FlashInfer pronto |
| Gerência de KV | PagedAttention, páginas híbridas de 3.584 tokens | extents contíguos de 2.048 num pool, compactação | TF: kernels contíguos e simples; custo: compactação/movimento, máx. 4 streams |
| Memória | `gpu_memory_utilization` + `--kv-cache-memory-bytes` fixos | `MemAvailable` − reserva; recusa antes de carregar; refit | TF: seguro em UMA; depende do estado da máquina na partida (pool varia ~2,1–2,9M) |
| Reuso de prompt | APC por hash de blocos, alinhado a 3.584 em híbridos | estados guardados em pontos (grade 64, prefixo compartilhado, replay) | TF: replay instantâneo e exato; vLLM: reuso em qualquer prefixo alinhado, muitos prefixos |
| Especulação | DFlash2 k=7 fixo (+ adaptive-k opt-in) | DFlash2 com parada noise-aware + copy drafts até 15 + MTP opcional | TF: profundidade por rodada, edição/citação rápidas; ambos dependem do DFlash2 não-comercial |
| CUDA graphs | FULL por tamanho de lote uniforme; passos mistos saem do graph | por largura de janela (1..16 / 1..32) com tabelas no device | TF: decode sempre em graph; prefill eager |
| Prefill × decode | chunked prefill misto + scheduler "fair" próprio | prefill e decode nunca na mesma forward; alternância 50/50; multi-prefill | TF: decode estável e TTFT baixo para prompts curtos; prompts frescos longos simultâneos esperam |
| KDA no prefill | kernels chunked do vLLM (FlashKDA só no TP3) | forma WY em kernel CUDA próprio, grade 64 | TF: prefill de 50k 29,3 → 26,4 s (~10%); perto do serial, não os mesmos bits |
| Hyper-connections | cola replicada | cola dividida por linhas entre ranks no prefill | TF: 50k ~1.270 → ~1.730 tok/s (+~36%, junto com 0009 e 0020); decode −~1,5% |
| Pesos densos | FP8 Marlin W8A16 | 4-bit por MSE (q4), FP8 ou BF16 | TF q4: mais rápido; regressão multilíngue (#18) |
| Contexto | 850k default (1M possível) | 1.048.576 com 4 streams | TF usa a janela nativa inteira |
| Escala | 2, 3, 4 Sparks | só 2 | vLLM escala |
| API | completa (logprobs, n, penalidades, API key) | subconjunto (sem logprobs/n/penalidades, sem auth) | TF exige proxy com auth para expor |
| Boot | ~2 min (InstantTensor) | 2–6 min (+ compilação na 1ª partida de cada imagem) | vLLM mais rápido para reiniciar |
| Pesos no worker | rsync ou NFS | rsync verificado ou volume NFS sem sudo | equivalentes |
| Números (com ressalvas §1.2) | prosa 36,1 / 75,3 tok/s (×1/×4); prefill ~1,4–1,5k | prosa 60,4 / 108,8; prefill ~1,95k | TF à frente nesta receita; terceiros mostram vLLM ainda à frente em prefill frio em outro setup |

---

## 8. Fontes (URLs)

- Receita TensorFold: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (HEAD `92bf731`, v1.3.2)
- TensorFold: https://github.com/ashhart/TensorFold (tag `v0.6.0`; `README.md`, `RUNBOOK.md`, `CHANGELOG.md`, `docs/api.md`, `docs/recipes/glm-5.3-flash.md`, `docs/recipes/cuda.md`; PR #115 janelas de copy mais largas; PR #140 pools visíveis)
- Receita vLLM anterior: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (HEAD `6278ecb`); inventário local `docs/pesquisa/glm-launch.md`
- jayleaton/glm53-tensorfold-spark: https://github.com/jayleaton/glm53-tensorfold-spark (patches 0460 L2 prefetch, 0580 expert loads, 0620 tool calls; RigMark e comparação com o kit vLLM)
- b12x / RoCEnante: https://github.com/local-inference-lab/b12x (`docs/rocenante.md`; commit do proxy usado: `8a99d639410e39d5f39cb4037675331beceea1d4`)
- sparkDash: https://github.com/MiaAI-Lab/sparkDash
- Checkpoint: https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw (espelho de https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw); base https://huggingface.co/zai-org/GLM-5.3-Flash
- Drafter: https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0); arquitetura de https://github.com/z-lab/dflash
- flash-linear-attention (algoritmo chunkwise de gated delta / KDA): https://github.com/fla-org/flash-linear-attention
- xgrammar: https://github.com/mlc-ai/xgrammar
- (terceiros) LimeChain bake-off: https://github.com/LimeChain/glm-5.3-flash-2xdgx-spark
- (terceiros) TensorFold issue #167: https://github.com/ashhart/TensorFold/issues/167
- RigMark (usado pelo jayleaton): https://github.com/alexellis/rigmark

---

## Addendum 2026-10-30

Este documento descreve a receita **v1.3.2** (`92bf731`, 53 patches) sobre o TensorFold **v0.6.0**. O que mudou
depois está verificado em [`mia-updates-2026-10-30.md`](mia-updates-2026-10-30.md). Em resumo, a receita está em
`4624c21` (2026-10-25), imagem `v0.6.0-789e8f3bdfa2` com **123 patches**: 3 Sparks (`start-tp3.sh`, 0066–0068), até
8 requisições (0069), fill fatiado e smooth streaming (0061–0062), API Anthropic (0084), camada de spill em NVMe
(0088, 0128), limites de estados guardados (0089, 0097, 0119), sharding de KV (0129), checkpoint próprio da Mia
(`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@078455ff`, Apache-2.0, desde a v1.3.3).

**Links de código fixados na tag.** O TensorFold foi reescrito como binário Zig na v1.0.0 (2026-10-07; a linha
Python parou na 0.6.6), e os caminhos `src/…` citados na seção de fontes não existem mais em `main`. Leia-os na tag
`v0.6.0`, sobre a qual os patches foram escritos:

- `src/tensorfold/cuda/`: [comm.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/cuda/comm.py),
  [capacity.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/cuda/capacity.py),
  [geometry.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/cuda/geometry.py),
  [sampling.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/cuda/sampling.py)
- [`src/tensorfold/engine/exact_sampling.py`](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/engine/exact_sampling.py)
- `src/tensorfold/families/glm5_next/cuda/`: [engine.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/families/glm5_next/cuda/engine.py),
  [decode.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/families/glm5_next/cuda/decode.py),
  [dflash2.py](https://github.com/ashhart/TensorFold/blob/v0.6.0/src/tensorfold/families/glm5_next/cuda/dflash2.py)

(Verificado no clone: os oito arquivos existem em `v0.6.0` e não existem em `v1.1.9`.)

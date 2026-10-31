# Copilot com contrato OpenAI + MLX local no Mac — referência técnica

> Data da pesquisa: **2026-09-28**. Explicações em PT-BR, código em inglês.
> Escopo: widget de chat "copilot" em JS puro (fetch + SSE), rodando a partir de `file://` ou `http://localhost`, que precisa falar com (a) qualquer endpoint compatível com **OpenAI Chat Completions** (base URL + API key + model) e (b) um modelo **local no Mac via MLX**.
>
> **Como isto foi verificado.** Além de ler docs e o código-fonte, instalei `mlx-lm 0.31.3` num venv isolado (scratchpad), subi `mlx_lm.server` no Mac do usuário (Apple M4, 24 GB, macOS 26.3.1) com `mlx-community/Qwen3-0.6B-4bit`, depois troquei (via campo `model`) para `mlx-community/Qwen3.5-4B-MLX-4bit`, `openbmb/MiniCPM5-2B-MLX` e `mlx-community/gemma-4-e4b-it-4bit`, e testei com `curl`, Node 26 e **Google Chrome 154 headless abrindo a página via `file://`**. O que está marcado **[verificado]** foi executado; **[fonte]** foi lido no código/doc; **[não verificado]** é inferência ou informação de terceiros.

---

## 0. TL;DR (decisões recomendadas)

1. **Um único contrato.** O widget só conhece duas rotas: `POST {baseUrl}/chat/completions` (com `stream: true` → SSE) e `GET {baseUrl}/models`. MLX, LM Studio, Ollama, llama.cpp, vLLM, OpenAI e OpenRouter viram apenas *presets* de `baseUrl` + regras pequenas (campo de max tokens, se exige `model`, se precisa de key).
2. **MLX local = `mlx_lm.server`** (pacote `mlx-lm`, da Apple/ml-explore). Porta padrão **8080**, bind **127.0.0.1**, CORS **`Access-Control-Allow-Origin: *` por padrão** [verificado]. Base URL: `http://127.0.0.1:8080/v1`.
3. **Instalação no Mac do usuário:** o `python3` do sistema é **3.9.6** e o `mlx` exige **Python ≥ 3.10** [verificado via PyPI] → use `uv` ou Homebrew:
   ```bash
   uv tool install --python 3.12 mlx-lm        # ou: brew install mlx-lm   (fórmula 0.31.3)
   mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 --max-tokens 2048 --temp 0.4
   ./serve.sh                                   # latent-atlas em http://localhost:8765
   ```
4. **No preset MLX, não envie `model` por padrão** (deixe o campo vazio → o servidor usa o `--model` carregado). Se o widget mandar um id diferente do carregado, o `mlx_lm.server` **troca de modelo e, se não estiver no cache, baixa do Hugging Face** na hora [fonte + verificado].
5. **Raciocínio:** `mlx_lm.server` separa o "thinking" e manda em **`delta.reasoning`** (não `reasoning_content`) [verificado]. Leia os dois nomes e ainda faça fallback para `<think>…</think>` dentro de `content`.
6. **Sirva a plataforma por HTTP** (`python3 -m http.server 8765 --bind 127.0.0.1`, que é o que o `serve.sh` do latent-atlas já faz). Funciona via `file://` também com MLX (testado no Chrome 154), mas Ollama bloqueia `Origin: null` por padrão e LM Studio precisa de CORS ligado.
7. **`max_tokens` padrão do mlx_lm.server é 512** — curto demais para modelos com thinking. No teste, o `Qwen3.5-4B` gastou **1500 de 1500 tokens só pensando** e terminou com `content` vazio (`finish_reason: "length"`); com `chat_template_kwargs: {"enable_thinking": false}` respondeu em PT-BR em **1,9 s** [verificado]. Para um tutor: thinking **desligado por padrão** + `max_tokens` 1500–2048, com um toggle "pensar mais".

---

## 1. O contrato mínimo que o widget precisa

| Item | Valor | Observação |
|---|---|---|
| Chat | `POST {base}/chat/completions` | `{"model"?, "messages":[{role, content:string}], "stream":true, "max_tokens"\|"max_completion_tokens", "temperature"?, "top_p"?, "stop"?, "stream_options"?:{"include_usage":true}}` |
| Stream | `Content-Type: text/event-stream` | eventos `data: {json}\n\n`, fim com `data: [DONE]` |
| Delta | `choices[0].delta.content` | texto da resposta |
| Raciocínio | `choices[0].delta.reasoning` **ou** `reasoning_content` | varia por servidor (ver §3) |
| Usage | chunk final com `usage` e **`choices: []`** | só se `stream_options.include_usage=true` |
| Modelos | `GET {base}/models` → `{"object":"list","data":[{"id":...}]}` | Ollama/LM Studio/MLX/OpenAI respondem nesse formato |
| Auth | `Authorization: Bearer <key>` | **só envie se houver key** (ver §4.3) |

Regras de portabilidade (o que quebra entre provedores):

- **OpenAI (cloud)**: prefira `max_completion_tokens` (o `max_tokens` é legado e é rejeitado por modelos de raciocínio); parâmetros desconhecidos (`top_k`, `chat_template_kwargs`, `min_p`) geram **400**; modelos de raciocínio podem rejeitar `temperature` ≠ padrão. O endpoint aceita CORS inclusive de `Origin: null` [verificado: preflight respondeu `access-control-allow-origin: null` e `access-control-allow-headers: authorization,content-type`; o 401 volta com `access-control-allow-origin: *`, então o navegador consegue ler a mensagem de erro].
- **Ollama**: só `max_tokens` (não há `max_completion_tokens` na struct) [fonte: `openai/openai.go`]; `model` obrigatório.
- **mlx_lm.server**: aceita `max_completion_tokens` **e** `max_tokens` (o primeiro tem prioridade) [fonte]; `model` opcional; ignora campos desconhecidos.
- **llama.cpp**: `max_tokens`/`n_predict`; o `model` é ignorado (usa o carregado).

---

## 2. `mlx_lm.server` — referência detalhada

Versões em 2026-09-28: **PyPI/Homebrew `mlx-lm 0.31.3`** (release de 2026-04-22), `mlx 0.32.2` (PyPI). O `main` do GitHub está à frente (último commit em `mlx_lm/server.py`: `f2b0c47`, 2026-09-11) e tem flags que **não existem** no 0.31.3 (ver tabela). Tudo abaixo foi conferido nos dois, salvo indicação.

### 2.1 Instalação

```bash
# Opção A (recomendada aqui): uv — isola e escolhe um Python compatível
uv tool install --python 3.12 mlx-lm          # expõe mlx_lm.server, mlx_lm.generate, mlx_lm.lora, mlx_lm.fuse...
# Opção B: Homebrew (formula mlx-lm 0.31.3, com bottle)
brew install mlx-lm
# Opção C: venv manual
python3.12 -m venv ~/.venvs/mlx && ~/.venvs/mlx/bin/pip install -U mlx-lm
# Bleeding edge (flags do main, p.ex. --kv-bits, e o remapeamento gemma4_unified):
uv tool install --python 3.12 "git+https://github.com/ml-explore/mlx-lm"
```

- Requisitos: Apple Silicon; `mlx` publica wheels para macOS 14, 15 e 26 (arm64) e exige Python ≥ 3.10 [verificado no PyPI]. O "wiring" de memória (que acelera modelos grandes) exige macOS ≥ 15 [fonte: README].
- Invocação: `mlx_lm.server …` ou `python -m mlx_lm server …`. `python -m mlx_lm.server` imprime aviso de *deprecated* [fonte].
- Pesos vão para o cache do Hugging Face (`~/.cache/huggingface/hub`, ou `$HF_HOME/hub`). Para baixar antes (e não travar a 1ª requisição): `hf download mlx-community/Qwen3-4B-Instruct-2507-4bit`. Dica: se o download ficar parado, `HF_HUB_DISABLE_XET=1` resolveu no meu teste [verificado].

### 2.2 Flags de linha de comando

| Flag | Padrão | O que faz / nota |
|---|---|---|
| `--model` | *(nenhum)* | repo HF ou caminho local. Se omitido, cada requisição **precisa** mandar `model`. |
| `--adapter-path` | — | adapter LoRA/DoRA treinado com `mlx_lm.lora` (ver §2.8) |
| `--host` | `127.0.0.1` | use `0.0.0.0` só se quiser expor na rede (não há autenticação!) |
| `--port` | `8080` | mesma porta padrão do `llama-server` e do `mlx_vlm.server` → cuidado com conflito |
| `--allowed-origins` | `*` | lista separada por vírgula; ver §2.6 |
| `--draft-model`, `--num-draft-tokens` | —, `3` | decodificação especulativa (desliga batching) |
| `--trust-remote-code` | off | para tokenizers com código custom |
| `--log-level` | `INFO` | `DEBUG` loga o corpo das requisições/respostas |
| `--chat-template` | `""` | sobrescreve o template do tokenizer |
| `--use-default-chat-template` | off | usa o template padrão se o modelo não tiver um |
| `--temp` | `0.0` | **default = greedy**. O cliente deve mandar `temperature` |
| `--top-p` / `--top-k` / `--min-p` | `1.0` / `0` / `0.0` | defaults de amostragem |
| `--max-tokens` | `512` | default de geração — **baixo para thinking** |
| `--chat-template-args` | `{}` | JSON repassado ao `apply_chat_template`, ex. `'{"enable_thinking":false}'` |
| `--decode-concurrency` / `--prompt-concurrency` | `32` / `8` | batching contínuo quando o modelo é "batchable" |
| `--prefill-step-size` | `2048` | menor = menos pico de memória no prefill |
| `--prompt-cache-size` / `--prompt-cache-bytes` | `10` / — | cache LRU de KV por prefixo (acelera conversas longas) |
| `--pipeline` | off | inferência distribuída (pipeline em vez de tensor parallel) |
| `--kv-bits`, `--kv-group-size`, `--quantized-kv-start` | —, `64`, `5000` | **só no `main`** (não no 0.31.3): KV cache quantizado; desliga batching |

Não existe flag de API key/autenticação. A própria doc avisa: *"not recommended for production as it only implements basic security checks"*.

### 2.3 Endpoints

| Método | Rota | Nota |
|---|---|---|
| POST | `/v1/chat/completions` | principal |
| POST | `/chat/completions` | alias sem `/v1` (só para chat) |
| POST | `/v1/completions` | text completion (`prompt`) |
| GET | `/v1/models` | lista modelos MLX **presentes no cache HF** (não só o carregado) + o `--model` se for caminho local |
| GET | `/v1/models/<org>/<repo>` | filtra um repo |
| GET | `/health` | `{"status":"ok"}` (200) ou `{"status":"unavailable"}` (503) |
| OPTIONS | qualquer | 204 com headers CORS |
| qualquer outra | — | 404 com corpo **texto** `Not Found` (não há `/v1/responses`, `/v1/embeddings`) |

### 2.4 Campos aceitos no corpo (chat)

Lidos em `APIHandler.do_POST` [fonte, igual no 0.31.3 e no main]:

| Campo | Default | Validação | Observação |
|---|---|---|---|
| `messages` | — (obrigatório) | — | `content` pode ser string ou lista **só** de partes `{"type":"text"}`; imagem → erro `"Only 'text' content type is supported."` [verificado] |
| `model` | `"default_model"` | string | ver quirk Q1 |
| `stream` | `false` | bool | |
| `stream_options.include_usage` | — | — | emite chunk extra de usage |
| `max_completion_tokens` → `max_tokens` | `--max-tokens` (512) | int ≥ 0 | o primeiro tem prioridade |
| `temperature` | `--temp` (0.0) | número ≥ 0 | string (`"0.7"`) derruba a conexão (Q4) |
| `top_p` | `--top-p` (1.0) | 0..1 | `1.5` derruba a conexão [verificado] |
| `top_k`, `min_p` | CLI | int ≥0 / 0..1 | extensões |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty` (+ `*_context_size`) | 0 / 20 | números | |
| `xtc_probability`, `xtc_threshold` | 0.0 / 0.1 | float 0..1 | |
| `logit_bias` | — | dict int→float | |
| `logprobs`, `top_logprobs` | false / -1 | bool / 0..11 | |
| `seed` | — | int | |
| `stop` | `[]` | string ou lista | |
| `tools` | — | — | tool calling (parser por modelo) |
| `chat_template_kwargs` | — | dict | mesclado sobre `--chat-template-args`; ex. `{"enable_thinking": false}` [verificado] |
| `adapters`, `draft_model`, `num_draft_tokens` | — | — | caminhos relativos ao diretório onde o servidor foi iniciado |
| `role_mapping` | — | — | só usado quando o tokenizer não tem chat template |

### 2.5 Formato do stream (captura real) [verificado]

Requisição: `{"messages":[…], "stream":true, "max_tokens":120, "stream_options":{"include_usage":true}}` com `Qwen3-0.6B-4bit` (modelo com thinking). Resposta (campos `id/system_fingerprint/created` encurtados):

```text
HTTP/1.0 200 OK
Content-type: text/event-stream
Cache-Control: no-cache
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *

: keepalive 21/22

: keepalive 22/22

data: {"object": "chat.completion.chunk", "model": "default_model", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "reasoning": "Okay"}}]}

data: {"object": "chat.completion.chunk", "model": "default_model", "choices": [{"index": 0, "finish_reason": null, "delta": {"role": "assistant", "content": "2 + 2 = 4."}}]}

data: {"object": "chat.completion.chunk", "model": "default_model", "choices": [{"index": 0, "finish_reason": "stop", "delta": {"role": "assistant"}}]}

data: {"object": "chat.completion", "model": "default_model", "choices": [], "usage": {"prompt_tokens": 22, "completion_tokens": 103, "total_tokens": 125, "prompt_tokens_details": {"cached_tokens": 21}}}

data: [DONE]
```

O que o parser precisa tolerar (todos observados):

- **Linhas de comentário** `: keepalive N/M` durante o prefill (SSE válido; ignore linhas que começam com `:`).
- `delta.reasoning` (string) durante o thinking; `delta.content` depois (confirmado com Qwen3, Qwen3.5, MiniCPM5 e Gemma 4). As tags `<think>`/`</think>` (e os canais do Gemma 4) são **removidas** pelo servidor quando o tokenizer é reconhecido como "thinking" [fonte: `TextStateMachine` + `tokenizer_utils._infer_thinking`, que reconhece `<think>`, `<longcat_think>`, `<|think:start|>`, `<|channel>thought…<channel|>` (Gemma 4) e o formato "xtml"].
- `"model": "default_model"` quando o cliente não mandou `model` (o servidor ecoa o que recebeu).
- O chunk de usage tem `"object": "chat.completion"` (não `.chunk`) e **`choices: []`**.
- Resposta em **HTTP/1.0** (sem chunked; o fim do corpo é o fechamento da conexão). `fetch` lida bem.

### 2.6 CORS no código e na prática

Trecho de `mlx_lm/server.py` (idêntico no 0.31.3 e no main) [fonte]:

```python
def _set_cors_headers(self):
    allowed_origins = self.response_generator.cli_args.allowed_origins
    origin = self.headers.get("Origin")
    if "*" in allowed_origins:
        self.send_header("Access-Control-Allow-Origin", "*")
    elif origin in allowed_origins:
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
    self.send_header("Access-Control-Allow-Methods", "*")
    self.send_header("Access-Control-Allow-Headers", "*")
```

- Sem `--allowed-origins`, **toda** resposta (inclusive erros 404 e o preflight `OPTIONS` → 204) leva `Access-Control-Allow-Origin: *` [verificado com `curl -H "Origin: null"`].
- Chrome 154 abrindo a página por `file://` (Origin enviado = `null`, confirmado com um servidor de eco) listou modelos, fez streaming e recebeu erros normalmente [verificado].
- Se você restringir (`--allowed-origins http://localhost:8765`), uma página em `file://` **deixa de funcionar** (o `Origin` é a string `null`). Para permitir as duas: `--allowed-origins http://localhost:8765,null`.
- `Access-Control-Allow-Headers: *` — pela spec Fetch o curinga **não** cobre `Authorization`, mas **Chrome e Safari ainda não aplicam essa regra** e o Firefox só com uma pref (dados MDN BCD) [fonte]. No teste, o Chrome 154 aceitou `Authorization: Bearer x` com `*` [verificado]. Mesmo assim: **não mande `Authorization` para servidor local sem key**.

### 2.7 Quirks conhecidos (o que costuma quebrar um widget)

- **Q1 — O campo `model` troca/baixa modelo.** Sem `model`, o servidor usa `"default_model"` → o `--model` da CLI. Com um id diferente do carregado, `ModelProvider.load()` descarrega o atual e carrega o pedido (e baixa do HF se preciso — pode ser GBs, e a requisição fica pendurada). Com um id inexistente (`"gpt-4o-mini"`) → **HTTP 404** `{"error": "<mensagem do huggingface_hub>"}` [verificado]. Recomendação: preset MLX com `model` vazio; quando o usuário escolher na lista de `/v1/models`, avise "vai carregar outro modelo".
- **Q2 — `/v1/models` lista o cache inteiro.** Qualquer repo com `config.json`, `model.safetensors.index.json` e `tokenizer_config.json` aparece — **inclusive downloads ainda incompletos** (vi isso durante o teste) e modelos de outras ferramentas. Não significa "carregado".
- **Q3 — `max_tokens` padrão 512 + thinking.** Modelos com thinking ligado podem gastar os 512 tokens só raciocinando e terminar com `finish_reason: "length"` e `content` vazio. Mande `max_tokens` ≥ 2048 ou desligue o thinking (`chat_template_kwargs: {"enable_thinking": false}` por requisição [verificado], ou `--chat-template-args '{"enable_thinking":false}'` no servidor). O `TokenizerWrapper` **liga o thinking por padrão** quando o modelo suporta [fonte: `kwargs[self._thinking_kwarg] = self.has_thinking`]. Medido: `Qwen3.5-4B-MLX-4bit`, pergunta de uma frase, `max_tokens: 1500` → 6.595 caracteres de `reasoning`, 0 de `content`, `finish_reason: "length"`; mesma pergunta sem thinking → 49 tokens, `finish_reason: "stop"` [verificado].
- **Q4 — Parâmetro inválido derruba a conexão sem resposta.** `validate_model_parameters()` levanta `ValueError` fora de `try` → o servidor fecha o socket (curl: *exit 52, empty reply*), e no navegador isso vira `TypeError: Failed to fetch` — **indistinguível de CORS/servidor fora do ar** [verificado com `temperature:"0.7"`, `top_p:1.5`, `top_k:-1`]. O erro só aparece no terminal do servidor. Mitigação no cliente: converter/clampar números antes de enviar.
- **Q5 — Erros de geração voltam como 404.** Qualquer exceção em `generate()` (modelo não encontrado, conteúdo não-texto etc.) vira `404` + `{"error": "texto"}` — formato **string**, não `{"error":{"message":…}}` da OpenAI [verificado].
- **Q6 — Nome do campo de raciocínio.** `delta.reasoning` (mlx-lm, mlx-vlm, Ollama, LM Studio p/ gpt-oss) vs `delta.reasoning_content` (llama.cpp `--reasoning-format deepseek`, vLLM, DeepSeek API, LM Studio com "separate reasoning_content").
- **Q7 — gpt-oss (formato "harmony") no mlx_lm 0.31.3.** Os canais `<|channel|>analysis<|message|>…<|end|>…<|channel|>final<|message|>…` não são separados → marcação e raciocínio vazam em `content` [fonte: issue ml-explore/mlx-lm#875 e relatos; não reproduzido aqui]. Para gpt-oss prefira LM Studio/Ollama/llama.cpp, ou use o fallback de parsing da §6.
- **Q8 — Só texto.** Partes `image_url` geram erro. Para imagens use `mlx_vlm.server` (§3.1).
- **Q9 — Temperatura padrão 0.0 (greedy).** Sem `temperature` no corpo, as respostas ficam determinísticas e repetitivas; para tutor, 0.3–0.7.
- **Q10 — Checkpoints multimodais funcionam como texto.** Repos convertidos com `mlx-vlm` (Qwen3.5/3.6/3.8, Gemma 4) carregam no `mlx-lm`: os `sanitize()` de `qwen3_5`, `qwen3_5_moe` e `gemma4` descartam `vision_tower`/`model.visual`/`audio_tower` [fonte]; `mlx-community/Qwen3.5-4B-MLX-4bit` e `mlx-community/gemma-4-e4b-it-4bit` (ambos convertidos com mlx-vlm) carregaram e geraram no `mlx-lm 0.31.3` [verificado]. `gemma4_unified` (ex.: Gemma 4 12B QAT) só é remapeado no **main** (`MODEL_REMAPPING["gemma4_unified"] = "gemma4"`), não no 0.31.3 [fonte].
- **Q11 — Memória.** O servidor chama `mx.set_wired_limit(max_recommended_working_set_size)`. No M4 24 GB do usuário o MLX reporta **19,07 GB** de working set recomendado (~74% da RAM) e buffer máximo de 14,3 GB [verificado]. Modelos acima disso ficam lentos; dá para subir o teto com `sudo sysctl iogpu.wired_limit_mb=N` (não persiste após reboot) [fonte: README].
- **Q12 — Cancelamento.** Abortar o `fetch` fecha o socket; o servidor para a geração daquele pedido (`finally: ctx.stop()`) e continua atendendo os próximos [verificado: requisições seguintes funcionaram; o log mostra `BrokenPipeError`, que é esperado].
- **Q13 — Resposta não-stream sem `content`.** No 0.31.3, se o texto final é vazio (ex.: só thinking), `message` vem **sem a chave `content`** (só `role` e `reasoning`) [verificado]; o `main` passou a mandar `"content": null`. Use `msg.content || ''`.
- **Q14 — Velocidade de referência.** M4 (10-core GPU) 24 GB, `Qwen3.5-4B-MLX-4bit`, thinking off: **~33 tokens/s** de decode; a troca de modelo + prefill levou poucos segundos [verificado, medição simples de chunks/s].

### 2.8 Servir um modelo "tunado" (ponte com o tutorial de receita/fine-tuning)

O mesmo servidor serve o resultado de um LoRA/DoRA/QLoRA feito com `mlx_lm.lora`, sem mudar nada no widget:

```bash
# treino (QLoRA se o --model for quantizado); dados em ./data/{train,valid}.jsonl no formato "chat"
mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-2507-4bit --train --data ./data --iters 600 --adapter-path ./adapters

# opção 1: servir base + adapter
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path ./adapters --port 8080

# opção 2: fundir e servir o modelo resultante (pode também --upload-repo ou --export-gguf)
mlx_lm.fuse --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path ./adapters --save-path ./fused_model
mlx_lm.server --model ./fused_model --port 8080
```

Também dá para escolher adapter por requisição com `"adapters": "adapters"` (caminho relativo ao diretório onde o servidor foi iniciado) [fonte]. Formato do `train.jsonl` (chat): `{"messages":[{"role":"system",…},{"role":"user",…},{"role":"assistant",…}]}` [fonte: LORA.md].

---

## 3. Outros servidores locais (e onde entram no widget)

| Servidor | Base URL padrão | Motor no Mac | CORS padrão | Key | Campo de raciocínio | `model` |
|---|---|---|---|---|---|---|
| `mlx_lm.server` | `http://127.0.0.1:8080/v1` | MLX | `*` (inclui `null`) | não existe | `reasoning` | opcional |
| `mlx_vlm.server` | `http://<host>:8080` (`/chat/completions` e `/v1/chat/completions`) | MLX (+visão/áudio) | `*` (FastAPI) | opcional (gestão) | `reasoning` (+ `reasoning_content` no não-stream) | opcional (`--model` pré-carrega) |
| LM Studio | `http://127.0.0.1:1234/v1` | MLX engine ou llama.cpp | **desligado** (ligar "Enable CORS" ou `lms server start --cors`) | opcional (0.4+: "Require Authentication") | `reasoning` (gpt-oss) / `reasoning_content` (opção) | envie o id de `/v1/models` |
| Ollama | `http://127.0.0.1:11434/v1` | llama.cpp (GGUF) e MLX (preview, safetensors) | `localhost/127.0.0.1/0.0.0.0` em qualquer porta + `file://*`, `app://*`… | não | `reasoning` | **obrigatório** |
| llama.cpp `llama-server` | `http://127.0.0.1:8080/v1` | Metal (GGUF) — **não é MLX** | `*` (ecoa `Origin`, com credentials) | `--api-key` | `reasoning_content` | ignorado |

### 3.1 mlx-vlm (`mlx_vlm.server`) — quando precisar de imagem
- `pip install -U mlx-vlm` (PyPI **0.7.4** em 2026-09-28). `mlx_vlm.server --model mlx-community/gemma-4-e4b-it-4bit --port 8081` (mude a porta para não bater com o mlx_lm) [fonte: `server/cli.py`].
- **Atenção: host padrão `0.0.0.0`** (expõe na rede local), porta 8080 [fonte]. Use `--host 127.0.0.1`.
- FastAPI com `CORSMiddleware(allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])` [fonte].
- Rotas: `/chat/completions` e `/v1/chat/completions`, `/models` e `/v1/models`, `/v1/responses`, `/v1/embeddings`, `/health` [fonte]. Aceita `image_url` nas mensagens.

### 3.2 LM Studio (GUI + motor MLX)
- Versão mais recente citada: **0.4.23 (agosto/2026)**; motor MLX baseado em `mlx-lm` (texto) + `mlx-vlm` (visão) [fonte: blog/changelog LM Studio; não verificado localmente].
- Servidor: aba Developer → Start, ou `lms server start [--port 1234] [--cors] [--bind 127.0.0.1]` [fonte: docs CLI]. **CORS vem desligado** → o widget falha até ligar.
- "Just in Time Model Loading": carrega o modelo pedido no `model` na hora; "Only Keep Last JIT Loaded Model" economiza RAM [fonte: Server Settings].
- Autenticação opcional (0.4.0+): se ligada, exige `Authorization: Bearer <token>` [fonte].
- Reasoning: desde 0.3.23, gpt-oss vai para `choices.delta.reasoning`; para outros modelos existe a opção "separate reasoning_content" [fonte: API changelog].

### 3.3 Ollama
- OpenAI compat em `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses` e até `/v1/messages` [fonte: `server/routes.go`].
- CORS: `AllowOrigins` = `OLLAMA_ORIGINS` + `http(s)://localhost[:*]`, `127.0.0.1[:*]`, `0.0.0.0[:*]`, `app://*`, `file://*`, `tauri://*`, `vscode-webview://*`, `vscode-file://*`; `Authorization` está explicitamente em `AllowHeaders` [fonte: `envconfig/config.go`].
  - `http://localhost:8765` → **liberado por padrão**.
  - `file://` → o navegador manda `Origin: null`, que **não casa** com `file://*` → bloqueado (resposta 403 sem header CORS → no JS aparece como `Failed to fetch`) [inferido do código; não testado com Ollama rodando]. Solução: `launchctl setenv OLLAMA_ORIGINS "*"` e reiniciar o app (ou `OLLAMA_ORIGINS="*" ollama serve`).
- Desligar thinking: `"reasoning_effort": "none"` [fonte: `ThinkingFromReasoningEffort`].
- MLX no Ollama: **0.19 (30/03/2026)** trouxe backend MLX em *preview* para Apple Silicon, só para modelos safetensors (ex.: `qwen3.5:35b-a3b-coding-nvfp4`) e com requisito de **mais de 32 GB** de memória unificada; GGUF continua em llama.cpp [fonte: ollama.com/blog/mlx]. O status atual (setembro/2026) pode ter avançado — [não verificado].

### 3.4 llama.cpp (breve)
- `brew install llama.cpp`; `llama-server -hf <repo-GGUF> --port 8080`. Porta 8080 (colide com mlx_lm) [fonte].
- CORS: `--cors-origins` padrão `*` **com** `--cors-credentials` ligado → ecoa o `Origin` da requisição (inclusive `null`); `--cors-headers` padrão `*` [fonte: `common/arg.cpp`, `server-http.cpp`]. Com `--api-key`, o preflight `OPTIONS` é liberado antes da validação.
- `--reasoning-format none|deepseek|…` controla se o thinking vai para `reasoning_content`.

---

## 4. CORS a partir de `file://` e de `http://localhost`

### 4.1 O que acontece
- Página aberta por `file://…/index.html`: `location.origin` é `"file://"`, mas o header enviado é **`Origin: null`** (origem opaca) [verificado no Chrome 154].
- `fetch` com `Content-Type: application/json` **sempre** gera preflight (`OPTIONS`) — o servidor tem que responder ao `OPTIONS` com `Access-Control-Allow-Origin` (`*` ou `null`) e `Allow-Headers` cobrindo `content-type` (e `authorization`, se mandar).
- Se o CORS falhar, o `fetch` rejeita com `TypeError` genérico ("Failed to fetch" no Chrome, "Load failed" no Safari, "NetworkError…" no Firefox). O JS **não consegue** ler status nem corpo.

### 4.2 Matriz

| Servidor | via `file://` (Origin `null`) | via `http://localhost:8765` |
|---|---|---|
| mlx_lm.server (padrão) | OK [verificado] | OK [verificado] |
| mlx_lm.server `--allowed-origins http://localhost:8765` | bloqueado (adicione `,null`) | OK |
| mlx_vlm.server | OK (`*`) [fonte] | OK [fonte] |
| Ollama (padrão) | **bloqueado** → `OLLAMA_ORIGINS="*"` | OK [fonte] |
| LM Studio | só com CORS ligado | só com CORS ligado |
| llama-server (padrão) | OK (ecoa `null`) [fonte] | OK [fonte] |
| OpenAI API | OK [verificado: preflight] | OK |

### 4.3 Recomendações
1. **Sirva a plataforma por HTTP**: `python3 -m http.server 8765 --bind 127.0.0.1` (o `latent-atlas/serve.sh` já faz exatamente isso). Vantagens: origem estável (`http://localhost:8765`) aceita por padrão pelo Ollama; `localStorage` isolado (no Chrome todas as páginas `file://` compartilham o mesmo storage); evita regras especiais de `file://` no Safari/Firefox.
2. **Não envie `Authorization` vazio.** Só adicione o header quando houver key.
3. **Local Network Access (Chrome 142+)**: só pede permissão quando uma página de origem *pública* acessa loopback/rede local. `http://localhost` → `127.0.0.1` é loopback→loopback e **não** é afetado; a spec sugere tratar `file:` como local [fonte: WICG LNA]. No teste via `file://` não houve prompt [verificado no headless]. Se um dia a plataforma for hospedada em `https://…`, o acesso a `http://127.0.0.1:8080` passará a pedir permissão ao usuário (e Safari pode bloquear como conteúdo misto) — [não verificado].
4. **Diagnóstico CORS × servidor fora do ar**: depois de um `TypeError`, faça `fetch(base + '/models', {mode: 'no-cors'})`. Se *resolver* (resposta opaca), o servidor está no ar → o problema é CORS (ou o Q4 do mlx). Se *rejeitar*, é conexão recusada/porta errada [verificado: implementado e testado na §6].

---

## 5. Modelos recomendados (mlx-community / oficiais) por RAM

Todos os ids abaixo **existem** no Hugging Face em 2026-09-28 [verificado via API `huggingface.co/api/models/<id>/tree/main`; tamanho = soma dos `.safetensors`]. Regra prática: pesos ≤ ~60% da RAM para sobrar KV cache, macOS e navegador. Referência medida: M4 24 GB → working set recomendado 19,07 GB.

**Legenda:** *T* = checkpoint só-texto; *MM* = checkpoint multimodal (o `mlx_lm` ignora a visão); *thinking* = raciocínio ligado por padrão (desligável via `chat_template_kwargs`).

**Sugestão para este Mac (M4, 24 GB):** padrão do copilot = `mlx-community/Qwen3.5-9B-MLX-4bit` (5,95 GB, mesma arquitetura do 4B testado) ou `mlx-community/gemma-4-e4b-it-4bit` (testado); para respostas mais rápidas, `mlx-community/Qwen3.5-4B-MLX-4bit` (testado, ~33 tok/s). Teto prático ≈ 15 GB de pesos (`gemma-4-26b-a4b-it-4bit`), com o navegador e o editor abertos.

### 8–16 GB
| Repo | Disco | Tipo | Por quê |
|---|---|---|---|
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 2,26 GB | T, sem thinking | Escolha "à prova de balas": arquitetura `qwen3` antiga e estável no mlx-lm, 262K ctx, responde direto (bom para tutor). |
| `openbmb/MiniCPM5-2B-MLX` | 1,42 GB | T, thinking | Oficial da OpenBMB (set/2026), `model_type: llama`; ótimo para Macs de 8 GB. **[verificado: carrega e responde em PT-BR no mlx_lm 0.31.3; raciocínio sai em `delta.reasoning`]** |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | 3,03 GB | MM, thinking | Geração 3.5 (fev/2026), 201 idiomas, 262K ctx. **[verificado: ~33 tok/s no M4 24 GB; boa resposta em PT-BR com thinking off]** |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | 5,95 GB | MM, thinking | Melhor qualidade que cabe folgado em 16 GB. |
| `mlx-community/gemma-4-e4b-it-4bit` | 5,15 GB | MM, thinking configurável | Gemma 4 E4B (Google, abr/2026), 128K ctx. **[verificado: com thinking, os marcadores `<\|channel>thought…<channel\|>` são removidos e o raciocínio sai em `delta.reasoning` (22 s, 1.884 chars de raciocínio); sem thinking, 1,9 s]** |

### 24–32 GB (o Mac do usuário: M4 24 GB)
| Repo | Disco | Tipo | Nota |
|---|---|---|---|
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 15,34 GB | MM, MoE ~4B ativos | Rápido (MoE) e forte; no 24 GB fica no limite — feche outros apps. O 0.31.3 já tem o bloco MoE em `gemma4_text` [fonte]. |
| `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` | 17,18 GB | T, MoE 3B ativos, sem thinking | Muito rápido; ideal para **32 GB**. |
| `mlx-community/Qwen3.8-27B-4bit` | 16,05 GB | MM, denso, thinking | Qwen mais recente (ago/2026); denso → mais lento por token. 32 GB. |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | 20,40 GB | MM, MoE 3B ativos, thinking | Só em **32 GB+**. |
| `mlx-community/gpt-oss-20b-MXFP4-Q8` | 12,08 GB | T, MoE, harmony | Cabe bem em 24 GB, mas veja Q7 (prefira LM Studio/Ollama para ele). |

### 64 GB+
| Repo | Disco | Nota |
|---|---|---|
| `mlx-community/Qwen3.8-27B-8bit` | 29,50 GB | 8-bit do Qwen3.8 — qualidade máxima do 27B. |
| `mlx-community/Qwen3.6-35B-A3B-8bit` | 37,72 GB | MoE rápido em 8-bit. |
| `mlx-community/gemma-4-31b-it-4bit` | 18,41 GB | Gemma 4 denso 31B, sobra muito KV em 64 GB. |
| `mlx-community/gpt-oss-120b-MXFP4-Q8` | 63,39 GB | Só em **96–128 GB**; mesmo aviso Q7. |

Evite para um tutor: repos "Uncensored/Heretic/abliterated/OBLITERATED" e merges de terceiros que aparecem no topo de downloads do `mlx-community`; quants "OptiQ" exigem o pacote `mlx-optiq` para os recursos extras (MTP/visão) [fonte: model cards].

---

## 6. Implementação JS de referência (`copilot-llm.js`)

Testada: 13 casos em Node 26 (parser SSE com chunks de 3 bytes, CRLF, comentários, `data:` multilinha; splitter de thinking; streaming real; erros de modelo, servidor fora do ar, parâmetro inválido; abort) **e** no Chrome 154 via `file://` e via `http://localhost` contra o `mlx_lm.server` [verificado].

```js
/* copilot-llm.js — provider-agnostic OpenAI-compatible chat client.
 * Plain browser JS, no dependencies. Works from file:// and http://localhost.
 * Exposes window.CopilotLLM = { PRESETS, streamChat, listModels, diagnose, splitThink, normalizeBaseUrl }.
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Presets: defaults per backend. Everything is overridable by the user config.
  // ---------------------------------------------------------------------------
  const PRESETS = {
    openai:   { label: 'OpenAI (cloud)',        baseUrl: 'https://api.openai.com/v1', needsKey: true,  maxTokensField: 'max_completion_tokens', usage: true },
    mlx:      { label: 'MLX (mlx_lm.server)',   baseUrl: 'http://127.0.0.1:8080/v1',  needsKey: false, maxTokensField: 'max_tokens', usage: true, omitModelIfEmpty: true,
                // Thinking OFF by default: thinking models can burn the whole max_tokens budget
                // reasoning (measured with Qwen3.5-4B). Set cfg.extraBody = { chat_template_kwargs: {} } to re-enable.
                extraBody: { chat_template_kwargs: { enable_thinking: false } } },
    lmstudio: { label: 'LM Studio (MLX/GGUF)',  baseUrl: 'http://127.0.0.1:1234/v1',  needsKey: false, maxTokensField: 'max_tokens', usage: true },
    ollama:   { label: 'Ollama',                baseUrl: 'http://127.0.0.1:11434/v1', needsKey: false, maxTokensField: 'max_tokens', usage: true },
    llamacpp: { label: 'llama.cpp (llama-server)', baseUrl: 'http://127.0.0.1:8080/v1', needsKey: false, maxTokensField: 'max_tokens', usage: true, omitModelIfEmpty: true },
    custom:   { label: 'Custom OpenAI-compatible', baseUrl: '', needsKey: true, maxTokensField: 'max_tokens', usage: false },
  };

  function normalizeBaseUrl(url) {
    let u = String(url || '').trim();
    u = u.replace(/\/+$/, '');
    // Users often paste the full endpoint; strip it back to the base.
    u = u.replace(/\/chat\/completions$/, '').replace(/\/models$/, '');
    return u;
  }

  function buildHeaders(cfg) {
    const h = { 'Content-Type': 'application/json' };
    // Only send Authorization when there is a key. Local servers do not need it, and the
    // Fetch spec says "Access-Control-Allow-Headers: *" does NOT cover Authorization
    // (Chrome/Safari do not enforce that yet; Firefox has it behind a pref).
    if (cfg.apiKey && cfg.apiKey.trim()) h['Authorization'] = 'Bearer ' + cfg.apiKey.trim();
    return Object.assign(h, cfg.headers || {});
  }

  function buildBody(cfg, messages, opts) {
    const preset = PRESETS[cfg.provider] || PRESETS.custom;
    const body = { messages, stream: opts.stream !== false };
    const model = (cfg.model || '').trim();
    if (model) body.model = model;
    else if (!preset.omitModelIfEmpty) throw new CopilotError('config', 'Informe o nome do modelo (campo "model").');

    const num = (v) => (v == null || v === '' || !Number.isFinite(Number(v)) ? null : Number(v));
    const maxField = cfg.maxTokensField || preset.maxTokensField || 'max_tokens';
    if (num(opts.maxTokens) != null) body[maxField] = Math.max(1, Math.trunc(num(opts.maxTokens))); // mlx-lm requires int
    if (num(opts.temperature) != null) body.temperature = Math.max(0, num(opts.temperature)); // number, never a string
    if (num(opts.topP) != null) body.top_p = Math.min(1, Math.max(0, num(opts.topP)));    // mlx-lm rejects > 1
    if (opts.stop && opts.stop.length) body.stop = opts.stop;
    if (body.stream && (cfg.includeUsage ?? preset.usage)) body.stream_options = { include_usage: true };
    return Object.assign(body, preset.extraBody || {}, cfg.extraBody || {});
  }

  // ---------------------------------------------------------------------------
  // Errors
  // ---------------------------------------------------------------------------
  class CopilotError extends Error {
    constructor(kind, message, extra) {
      super(message);
      this.name = 'CopilotError';
      this.kind = kind; // network | cors | auth | not_found | bad_request | rate_limit | server | aborted | timeout | config | parse
      Object.assign(this, extra || {});
    }
  }

  async function readErrorBody(res) {
    const text = await res.text().catch(() => '');
    try {
      const j = JSON.parse(text);
      // OpenAI: {error:{message,type,code}} ; mlx-lm: {error:"..."} ; Ollama: {error:{message}} or {error:"..."}
      if (typeof j.error === 'string') return { message: j.error, raw: text };
      if (j.error && j.error.message) return { message: j.error.message, code: j.error.code, raw: text };
      if (j.message) return { message: j.message, raw: text };
    } catch (_) { /* not JSON */ }
    return { message: text.slice(0, 500) || res.statusText, raw: text };
  }

  function httpError(res, info, cfg) {
    const s = res.status, m = info.message || '';
    const where = normalizeBaseUrl(cfg.baseUrl);
    if (s === 401) return new CopilotError('auth', `401 — API key ausente ou inválida para ${where}.`, { status: s, detail: m });
    if (s === 403) return new CopilotError('auth', `403 — acesso negado (chave sem permissão, origem bloqueada ou projeto sem acesso ao modelo). ${m}`, { status: s, detail: m });
    if (s === 404) {
      if (/^not found$/i.test(m.trim())) return new CopilotError('not_found', `404 — endpoint não existe. Confira a Base URL (normalmente termina em /v1): ${where}`, { status: s, detail: m });
      return new CopilotError('not_found', `404 — modelo não encontrado ou falha ao carregar o modelo "${cfg.model || '(padrão do servidor)'}". Detalhe: ${m}`, { status: s, detail: m });
    }
    if (s === 400 || s === 422) return new CopilotError('bad_request', `${s} — requisição rejeitada: ${m}`, { status: s, detail: m });
    if (s === 429) return new CopilotError('rate_limit', `429 — limite de uso/cota atingido. ${m}`, { status: s, detail: m });
    if (s === 503) return new CopilotError('server', `503 — servidor indisponível ou carregando o modelo. Tente de novo em alguns segundos. ${m}`, { status: s, detail: m });
    return new CopilotError('server', `${s} — erro do servidor: ${m}`, { status: s, detail: m });
  }

  // When fetch() rejects with TypeError we cannot tell CORS from "connection refused".
  // Probe with mode:'no-cors': if that succeeds the server is up, so CORS (or a dropped
  // connection caused by an invalid parameter in mlx_lm.server) is the culprit.
  async function diagnose(cfg) {
    const base = normalizeBaseUrl(cfg.baseUrl);
    try {
      await fetch(base + '/models', { mode: 'no-cors', cache: 'no-store' });
      return 'reachable';
    } catch (_) {
      return 'unreachable';
    }
  }

  async function networkError(err, cfg) {
    const base = normalizeBaseUrl(cfg.baseUrl);
    const state = await diagnose(cfg);
    const loc = global.location;
    const origin = !loc ? 'n/a' : loc.protocol === 'file:' ? 'null (página aberta via file://)' : loc.origin;
    if (state === 'unreachable') {
      return new CopilotError('network',
        `Não consegui conectar em ${base}. O servidor está rodando? A porta está certa? ` +
        `(ex.: mlx_lm.server --model <repo> --port 8080). Detalhe: ${err.message}`);
    }
    return new CopilotError('cors',
      `O servidor em ${base} respondeu, mas o navegador bloqueou a resposta (CORS) ou a conexão caiu. ` +
      `Origem desta página: ${origin}. Ollama: defina OLLAMA_ORIGINS="*"; LM Studio: ative "Enable CORS"; ` +
      `mlx_lm.server: já envia "*" por padrão (se usou --allowed-origins, inclua "null" ou a origem). ` +
      `No mlx_lm.server, parâmetros inválidos (ex.: top_p > 1, temperature como texto) também derrubam a conexão — veja o terminal do servidor.`);
  }

  // ---------------------------------------------------------------------------
  // Reasoning splitter: handles <think>…</think>, an orphan </think> (template put
  // <think> in the prompt), Gemma-style <|channel>thought…<channel|>, and leaked
  // gpt-oss "harmony" channels. Pure function over the accumulated raw text, so
  // tags split across chunks are handled for free.
  // ---------------------------------------------------------------------------
  const THINK_PAIRS = [
    ['<think>', '</think>'],
    ['<thinking>', '</thinking>'],
    ['<|channel>thought', '<channel|>'],
    ['<|channel|>analysis<|message|>', '<|end|>'],
  ];
  const HARMONY_NOISE = /<\|start\|>assistant|<\|channel\|>final<\|message\|>|<\|return\|>|<\|end\|>/g;

  function splitThink(raw) {
    let text = raw || '';
    let reasoning = '';
    let thinking = false; // true while inside an unclosed reasoning block

    for (const [open, close] of THINK_PAIRS) {
      const firstOpen = text.indexOf(open);
      const firstClose = text.indexOf(close);
      // Orphan close tag: the opening tag was part of the prompt template.
      if (firstClose !== -1 && (firstOpen === -1 || firstClose < firstOpen) && open === '<think>') {
        reasoning += text.slice(0, firstClose);
        text = text.slice(firstClose + close.length);
      }
      let out = '';
      let i = 0;
      while (i < text.length) {
        const o = text.indexOf(open, i);
        if (o === -1) { out += text.slice(i); break; }
        out += text.slice(i, o);
        const c = text.indexOf(close, o + open.length);
        if (c === -1) { reasoning += text.slice(o + open.length); thinking = true; break; }
        reasoning += text.slice(o + open.length, c);
        i = c + close.length;
      }
      text = out;
    }
    text = text.replace(HARMONY_NOISE, '');
    // Hide a partially-received tag at the very end (e.g. "<thi") to avoid flicker.
    const tail = text.lastIndexOf('<');
    if (tail !== -1 && text.length - tail < 32 && THINK_PAIRS.some(([o]) => o.startsWith(text.slice(tail)))) {
      text = text.slice(0, tail);
    }
    return { content: text.replace(/^\s+/, ''), reasoning: reasoning.trim(), thinking };
  }

  // ---------------------------------------------------------------------------
  // SSE parsing (WHATWG event-stream rules, simplified): handles CRLF, partial lines
  // across chunks, multi-line data:, ":" comments (mlx keepalive), and [DONE].
  // ---------------------------------------------------------------------------
  async function* readSSE(body, signal) {
    const reader = body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let dataLines = [];
    try {
      while (true) {
        if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
        const { value, done } = await reader.read();
        buf += done ? decoder.decode() : decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.search(/\r\n|\r|\n/)) !== -1) {
          const line = buf.slice(0, nl);
          buf = buf.slice(buf[nl] === '\r' && buf[nl + 1] === '\n' ? nl + 2 : nl + 1);
          if (line === '') { // end of event
            if (dataLines.length) { yield dataLines.join('\n'); dataLines = []; }
            continue;
          }
          if (line.startsWith(':')) continue;               // comment / keepalive
          if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
          // "event:", "id:", "retry:" are ignored for chat completions
        }
        if (done) {
          if (buf.startsWith('data:')) dataLines.push(buf.slice(5).replace(/^ /, ''));
          if (dataLines.length) yield dataLines.join('\n');
          return;
        }
      }
    } finally {
      // Cancel (not just release) so an early break after [DONE] closes the connection.
      try { await reader.cancel(); } catch (_) {}
    }
  }

  /**
   * streamChat(cfg, messages, handlers, opts)
   *  cfg:      { provider, baseUrl, apiKey, model, headers?, extraBody?, maxTokensField?, includeUsage? }
   *  handlers: { onDelta({content, reasoning, thinking, raw}), onDone({content, reasoning, usage, finishReason}), onError(err) }
   *  opts:     { signal, maxTokens, temperature, topP, stop, firstByteTimeoutMs }
   * Returns a Promise that resolves with the final result.
   */
  async function streamChat(cfg, messages, handlers = {}, opts = {}) {
    const base = normalizeBaseUrl(cfg.baseUrl);
    if (!/^https?:\/\//.test(base)) throw new CopilotError('config', 'Base URL precisa começar com http:// ou https://');

    const ctrl = new AbortController();
    const onAbort = () => ctrl.abort();
    if (opts.signal) opts.signal.addEventListener('abort', onAbort, { once: true });
    const firstByteMs = opts.firstByteTimeoutMs ?? 120000; // local models may be loading/downloading
    let timedOut = false;
    let timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, firstByteMs);

    let rawContent = '', reasoningField = '', usage = null, finishReason = null;
    const emit = () => {
      const view = splitThink(rawContent);
      const reasoning = [reasoningField.trim(), view.reasoning].filter(Boolean).join('\n');
      handlers.onDelta && handlers.onDelta({ content: view.content, reasoning, thinking: view.thinking || (!!reasoningField && !view.content), raw: rawContent });
      return { content: view.content, reasoning };
    };

    try {
      let res;
      try {
        res = await fetch(base + '/chat/completions', {
          method: 'POST',
          headers: buildHeaders(cfg),
          body: JSON.stringify(buildBody(cfg, messages, opts)),
          signal: ctrl.signal,
          cache: 'no-store',
        });
      } catch (err) {
        if (err.name === 'AbortError') throw timedOut
          ? new CopilotError('timeout', `Sem resposta em ${Math.round(firstByteMs / 1000)}s. O modelo pode estar carregando ou baixando (veja o terminal do servidor).`)
          : new CopilotError('aborted', 'Geração cancelada.');
        if (err instanceof CopilotError) throw err;
        throw await networkError(err, cfg);
      }
      clearTimeout(timer); // headers arrived: the server is alive and generating
      if (!res.ok) throw httpError(res, await readErrorBody(res), cfg);

      const ctype = res.headers.get('content-type') || '';
      if (!ctype.includes('text/event-stream')) {
        // Server ignored stream:true (or stream disabled): parse a normal completion.
        const j = await res.json();
        const msg = (j.choices && j.choices[0] && j.choices[0].message) || {};
        rawContent = msg.content || '';
        reasoningField = msg.reasoning_content || msg.reasoning || '';
        usage = j.usage || null;
        finishReason = j.choices && j.choices[0] && j.choices[0].finish_reason;
        const final = emit();
        const result = { ...final, usage, finishReason };
        handlers.onDone && handlers.onDone(result);
        return result;
      }

      // Throttle UI updates to ~30/s. setTimeout (not requestAnimationFrame) so a
      // hidden/background tab still makes progress.
      let emitPending = false;
      const scheduleEmit = () => {
        if (emitPending) return;
        emitPending = true;
        setTimeout(() => { emitPending = false; emit(); }, 33);
      };

      for await (const data of readSSE(res.body, ctrl.signal)) {
        if (data.trim() === '[DONE]') break;
        let chunk;
        try { chunk = JSON.parse(data); } catch (_) { continue; } // tolerate junk lines
        if (chunk.error) throw new CopilotError('server', 'Erro no meio do stream: ' + (chunk.error.message || chunk.error));
        if (chunk.usage) usage = chunk.usage;              // usage chunk may have choices: []
        const choice = chunk.choices && chunk.choices[0];
        if (!choice) continue;
        const d = choice.delta || choice.message || {};
        if (typeof d.content === 'string') rawContent += d.content;
        // Field name varies: mlx-lm/Ollama/mlx-vlm/LM Studio(gpt-oss) -> reasoning ; DeepSeek/llama.cpp/vLLM -> reasoning_content
        const r = d.reasoning_content ?? d.reasoning;
        if (typeof r === 'string') reasoningField += r;
        if (choice.finish_reason) finishReason = choice.finish_reason;
        scheduleEmit();
      }
      const final = emit();
      const result = { ...final, usage, finishReason };
      handlers.onDone && handlers.onDone(result);
      return result;
    } catch (err) {
      const e = err instanceof CopilotError ? err
        : err.name === 'AbortError' ? new CopilotError(timedOut ? 'timeout' : 'aborted', timedOut ? 'Tempo esgotado aguardando o modelo.' : 'Geração cancelada.')
        : new CopilotError('parse', 'Falha ao ler o stream: ' + err.message);
      if (e.kind === 'aborted' && (rawContent || reasoningField)) { // keep partial answer on cancel
        const partial = emit();
        handlers.onDone && handlers.onDone({ ...partial, usage, finishReason: 'aborted' });
        return { ...partial, usage, finishReason: 'aborted' };
      }
      handlers.onError && handlers.onError(e);
      throw e;
    } finally {
      clearTimeout(timer);
      if (opts.signal) opts.signal.removeEventListener('abort', onAbort);
    }
  }

  async function listModels(cfg) {
    const base = normalizeBaseUrl(cfg.baseUrl);
    let res;
    try {
      res = await fetch(base + '/models', { headers: buildHeaders(cfg), cache: 'no-store' });
    } catch (err) {
      throw await networkError(err, cfg);
    }
    if (!res.ok) throw httpError(res, await readErrorBody(res), cfg);
    const j = await res.json();
    const items = Array.isArray(j.data) ? j.data : Array.isArray(j.models) ? j.models : [];
    return items.map((m) => (typeof m === 'string' ? m : m.id || m.name)).filter(Boolean).sort();
  }

  global.CopilotLLM = { PRESETS, streamChat, listModels, diagnose, splitThink, normalizeBaseUrl, CopilotError, _readSSE: readSSE, _buildBody: buildBody };
})(typeof window !== 'undefined' ? window : globalThis);
```

### 6.1 Uso no widget

```js
const cfg = { provider: 'mlx', baseUrl: CopilotLLM.PRESETS.mlx.baseUrl, apiKey: '', model: '' };
const ac = new AbortController();               // wire to the "Stop" button: ac.abort()

CopilotLLM.streamChat(cfg, [
  { role: 'system', content: systemPrompt() },
  ...history,
  { role: 'user', content: question },
], {
  onDelta: ({ content, reasoning, thinking }) => render({ content, reasoning, thinking }),
  onDone:  ({ usage, finishReason }) => {
    if (finishReason === 'length') showHint('Resposta cortada: aumente max_tokens ou desligue o thinking.');
  },
  onError: (e) => showError(e.message),         // e.kind drives icons/help links
}, { maxTokens: 2048, temperature: 0.4, signal: ac.signal });

// Model picker ("Listar modelos"):
CopilotLLM.listModels(cfg).then(ids => fillDatalist(ids)).catch(e => showError(e.message));
```

Notas de UI:
- **Thinking na UI:** mostre `reasoning` num `<details>` "pensando…" enquanto `thinking === true` e o `content` estiver vazio; ao terminar, colapse.
- **Toggle de thinking:** o preset `mlx` já vem com thinking **desligado** (`chat_template_kwargs: { enable_thinking: false }`). Para ligar ("pensar mais"): `cfg.extraBody = { chat_template_kwargs: { enable_thinking: true } }` (o `Object.assign` é raso: a chave do `cfg` substitui a do preset). vLLM/SGLang usam o mesmo campo; Ollama → `extraBody: { reasoning_effort: 'none' }` para desligar; OpenAI → não mande nada extra (400).
- **API key:** guardar em `localStorage` é aceitável só para uso pessoal (qualquer script da página lê). Ofereça "não lembrar a chave".

---

## 7. Mensagens de erro (o que mostrar ao usuário)

| Sintoma no JS | Causa provável | Mensagem sugerida (PT-BR) | Correção |
|---|---|---|---|
| `TypeError` + probe `no-cors` **falha** | servidor parado / porta errada / host errado | "Não consegui conectar em {base}. O servidor está rodando? A porta está certa?" | `mlx_lm.server --model … --port 8080`; `curl {base}/models` |
| `TypeError` + probe `no-cors` **ok** | CORS bloqueado **ou** mlx derrubou a conexão (Q4) | "O servidor respondeu, mas o navegador bloqueou (CORS) ou a conexão caiu. Origem: {origin}." | Ollama `OLLAMA_ORIGINS="*"`; LM Studio "Enable CORS"; mlx `--allowed-origins …,null`; revisar parâmetros |
| 401 | key ausente/errada | "API key ausente ou inválida para {base}." | configurar key |
| 403 | key sem permissão / origem bloqueada / org sem acesso ao modelo | "Acesso negado…" + detalhe do servidor | |
| 404 corpo `Not Found` | rota errada (faltou `/v1`, ou servidor não tem a rota) | "Endpoint não existe. Confira a Base URL (termina em /v1)." | |
| 404 com JSON | modelo inexistente (OpenAI `model_not_found`; mlx: qualquer falha ao carregar/gerar) | "Modelo não encontrado ou falha ao carregar '{model}'. Detalhe: …" | usar id de `/v1/models`; no mlx, deixar `model` vazio |
| 400/422 | parâmetro não suportado (`max_tokens` vs `max_completion_tokens`, `temperature` em modelo de raciocínio, campo desconhecido) | "Requisição rejeitada: {detalhe}" | trocar campo de max tokens; remover extras (retry automático) |
| 429 | rate limit / sem créditos | "Limite de uso/cota atingido." | |
| 503 | servidor carregando / thread de geração morta (`/health` = unavailable) | "Servidor indisponível ou carregando o modelo." | aguardar; reiniciar |
| timeout sem headers | 1º uso: download/carregamento do modelo | "Sem resposta em {n}s. O modelo pode estar carregando ou baixando." | pré-baixar com `hf download` |
| `finish_reason: "length"` e `content` vazio | thinking consumiu o orçamento (Q3) | "Resposta cortada…" | `max_tokens` maior / `enable_thinking:false` |
| `AbortError` | usuário clicou em parar | "Geração cancelada." (mantenha o parcial) | |

---

## 8. Checklist de verificação rápida (copiar e colar)

```bash
# 1) servidor vivo e modelos
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models

# 2) CORS como o navegador vê a partir de file:// (Origin: null)
curl -s -i -X OPTIONS http://127.0.0.1:8080/v1/chat/completions \
  -H "Origin: null" -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type"
#   -> esperar 204 + Access-Control-Allow-Origin: *

# 3) streaming com usage e thinking desligado
curl -N http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages":[{"role":"user","content":"Explique atenção em 2 frases."}],
  "stream":true, "max_tokens":512, "temperature":0.4,
  "stream_options":{"include_usage":true},
  "chat_template_kwargs":{"enable_thinking":false}}'
```

---

## 9. Observações sobre o `latent-atlas/assets/js/copilot.js` atual

O arquivo existente (671 linhas, lido em 2026-09-28) já segue o contrato certo (presets MLX/OpenAI/OpenRouter/vLLM/LM Studio/Ollama, `reasoning`/`reasoning_content`, retry que troca `max_tokens`↔`max_completion_tokens` e remove `temperature`/`chat_template_kwargs` em 400). Pontos que esta pesquisa sugere ajustar:

1. **Preset MLX manda `model: 'mlx-community/Qwen3-4B-Instruct-2507-4bit'` por padrão.** Se o usuário subiu o `mlx_lm.server` com outro modelo, a 1ª pergunta faz o servidor **trocar e possivelmente baixar ~2,3 GB** (Q1). Deixe `model: ''` no preset MLX (o `buildBody` atual já faz `model: cfg.model || undefined`, então vazio = omitido).
2. **`readStream` descarta o buffer final** quando o stream termina sem `\n` e não junta `data:` multilinha. Inofensivo com mlx (sempre termina em `\n\n`), mas o `readSSE` da §6 cobre os dois casos.
3. **`explainError` não diferencia CORS de conexão recusada.** Acrescentar o probe `no-cors` (§4.3.4) e citar o Q4 do mlx.
4. **`splitThink` só trata `<think>`.** Considerar o fallback para Gemma (`<|channel>thought…<channel|>`) e gpt-oss harmony (§6), úteis com LM Studio/llama.cpp configurados sem separação de reasoning.
5. **Toggle "Pedir resposta sem thinking"** usa `chat_template_kwargs` — correto para mlx/vLLM (verificado com mlx), mas no **Ollama** o equivalente é `reasoning_effort: "none"`. Sugestão: `noThink: true` **por padrão no preset MLX** (DEFAULTS hoje tem `noThink: false`, e com `maxTokens: 1800` um Qwen3.5/MiniCPM5 pode terminar sem resposta — Q3).
6. **Base `http://localhost:8080/v1`** funciona (testei `localhost` → 127.0.0.1 com curl, Node e Chrome); `127.0.0.1` evita qualquer ambiguidade IPv6.

---

## 10. Incertezas e pontos a confirmar

- **Ollama + `file://`**: bloqueio de `Origin: null` inferido do código (`gin-contrib/cors` com a lista padrão); não testei com Ollama instalado.
- **LM Studio**: não instalado aqui; valores de CORS/`reasoning_content` vêm da documentação oficial e de issues. O header exato enviado com "Enable CORS" (`*` vs eco) não foi verificado.
- **gpt-oss no mlx_lm 0.31.3 (Q7)**: baseado em issue/relatos; não baixei o modelo (12 GB).
- **Modelos da §5**: existência, tamanho e `model_type` verificados na API do HF; **carregamento real no mlx_lm 0.31.3 testado com `Qwen3-0.6B-4bit`, `Qwen3.5-4B-MLX-4bit`, `MiniCPM5-2B-MLX` e `gemma-4-e4b-it-4bit`**. Os maiores (Gemma 4 26B/31B, Qwen3.6/3.8, gpt-oss, Qwen3-30B) não foram baixados aqui. Qualidade em PT-BR não foi medida de forma sistemática.
- **Ollama MLX**: o requisito ">32 GB" e a lista de modelos são do anúncio de março/2026; pode ter mudado.
- **Local Network Access**: o tratamento de `file://` é "não normativo" na spec; o Chrome pode mudar.
- **Safari**: não testado (o headless foi Chrome). Safari costuma ser mais restritivo com `file://`; mais um motivo para servir por HTTP.

---

## 11. Fontes

- mlx-lm server (código): https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py (main `f2b0c47`, 2026-09-11) e https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/server.py
- mlx-lm server (doc): https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
- mlx-lm README (instalação, wired limit): https://github.com/ml-explore/mlx-lm/blob/main/README.md
- mlx-lm thinking tokens: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tokenizer_utils.py
- mlx-lm remapeamentos/modelos: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py , https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models
- mlx-lm LoRA/fuse: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- PyPI: https://pypi.org/project/mlx-lm/ (0.31.3), https://pypi.org/project/mlx/ (0.32.2, Python ≥ 3.10), https://pypi.org/project/mlx-vlm/ (0.7.4)
- Homebrew: https://formulae.brew.sh/formula/mlx-lm
- gpt-oss/harmony no mlx-lm: https://github.com/ml-explore/mlx-lm/issues/875
- mlx-vlm server: https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/server (`app.py`, `cli.py`, `openai.py`)
- LM Studio: https://lmstudio.ai/docs/developer/openai-compat , https://lmstudio.ai/docs/cli/serve/server-start , https://lmstudio.ai/docs/developer/core/server/settings , https://lmstudio.ai/docs/developer/core/authentication , https://lmstudio.ai/docs/developer/api-changelog , https://lmstudio.ai/blog/unified-mlx-engine
- Ollama: https://github.com/ollama/ollama/blob/main/envconfig/config.go , https://github.com/ollama/ollama/blob/main/server/routes.go , https://github.com/ollama/ollama/blob/main/openai/openai.go , https://docs.ollama.com/api/openai-compatibility , https://ollama.com/blog/mlx
- llama.cpp: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-http.cpp , https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp
- CORS/Authorization e curinga: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Access-Control-Allow-Headers , https://github.com/mdn/browser-compat-data/blob/main/http/headers/Access-Control-Allow-Headers.json
- Local Network Access: https://developer.chrome.com/blog/local-network-access , https://wicg.github.io/local-network-access/
- Modelos (cards/API HF): https://huggingface.co/mlx-community , https://huggingface.co/Qwen/Qwen3.8-27B , https://huggingface.co/Qwen/Qwen3.5-9B , https://huggingface.co/Qwen/Qwen3.6-35B-A3B , https://huggingface.co/google/gemma-4-E4B-it , https://huggingface.co/openbmb/MiniCPM5-2B , https://huggingface.co/openbmb/MiniCPM5-2B-MLX

---

## 12. Artefatos de teste (scratchpad desta sessão)

Base: o diretório temporário (scratchpad) da sessão de pesquisa; não faz parte do repositório.

- `src/test/copilot-llm.js` — o mesmo código da §6 (arquivo testado).
- `src/test/unit.mjs` — 13 testes (Node 26; os testes "ao vivo" esperam um `mlx_lm.server` em `127.0.0.1:8089`).
- `src/test/index.html` + `cdp_test.mjs` — harness que abre a página em **Chrome headless via `file://`** usando o DevTools Protocol (`node cdp_test.mjs "file://…/index.html"`).
- `mlxenv/` — venv Python 3.12 com `mlx-lm 0.31.3` / `mlx 0.32.2`.
- `hfhome/` — cache HF (~9,6 GB) com `Qwen3-0.6B-4bit`, `Qwen3.5-4B-MLX-4bit`, `MiniCPM5-2B-MLX`, `gemma-4-e4b-it-4bit`. Para reaproveitar sem baixar de novo: `HF_HOME=<base>/hfhome mlx_lm.server --model mlx-community/Qwen3.5-4B-MLX-4bit`. Pode ser apagado quando não for mais útil.
- Todos os servidores de teste (portas 8089, 8765, 8091, 9333) foram encerrados ao final.

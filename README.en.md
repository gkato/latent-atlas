# Latent Atlas — from neuron to cluster

> [🇧🇷 Português](README.md) · 🇺🇸 English

An HTML study platform for AI researchers. In chronological order and with animated, interactive
figures, it teaches **everything** you need to write inference recipes like Mia's AI Lab's
(GLM-5.3-Flash EXL3, Qwen3.8-Flash-Next NVFP4 and DeepSeek-V4-Flash DSpark on 2× DGX Spark with
vLLM): ML/DL fundamentals, Transformers, frontier architectures, quantization, hardware and kernels,
vLLM, speculative decoding and distributed inference. It closes with a tutorial on creating your own
recipe for a newly released model.

**Pillars of AI researcher training** (hardware- and recipe-agnostic — from a 2× DGX Spark lab to
production H200 clusters with NVIDIA AI Enterprise): **1 · Inference engineering** (chapters 00–49) ·
**2 · Post-training** (chapters 50–63) · **3 · Pre-training** and **4 · Benchmarking & testing
strategies** (on the roadmap). The Mia's AI Lab recipes (including the GLM version on **TensorFold**)
appear as **case studies** that run through the trail.

**By the numbers:** 64 chapters in 14 parts and 2 active pillars · ~572k words · 694 dynamic figures · 398 quiz
questions · 1,571 glossary terms · a catalog of 1,734 parameters and modifications from 4 recipes · interface
and content in **English and Portuguese** · a copilot that works with OpenAI, local MLX and other providers.

**Dated case studies.** The recipes are snapshots of a date: vLLM GLM at `94ae731` (2026-09-28), GLM · TensorFold
v1.3.2 (2026-10-01), Qwen and DeepSeek read on 2026-09-28. Upstream keeps moving; every GLM · TensorFold block
carries a **snapshot v1.3.2** tag linking to "what changed since" (ch. 49), and what became wrong was fixed in place.
The 2026-10-30 check is in [`docs/pesquisa/mia-updates-2026-10-30.md`](docs/pesquisa/mia-updates-2026-10-30.md);
among other things it added the switchless 4-Spark ring (ch. 40), the NVMe prompt cache (ch. 33) and the case study
"DeepSeek-V4.1-Flash: one model, five deployments" (ch. 44).

---

## 1. Requirements

| To… | You need |
|---|---|
| **Study** (use the platform) | A modern browser (recent Chrome, Edge, Safari or Firefox). Internet is optional: without it, fonts fall back to system fonts and formulas show as text. |
| **Serve locally** (recommended for the copilot) | Python 3 (ships with macOS and most Linux distributions). |
| **A 100% local copilot on a Mac** | An Apple Silicon Mac (M1–M4) and `mlx-lm` (Python ≥ 3.10). The **MLX** button in the top bar installs `mlx-lm` with `uv` or Homebrew, downloads the model and starts the server for you. |
| **A cloud copilot** | An API key (OpenAI, OpenRouter…) or any OpenAI-compatible server. |
| **Edit and validate** (optional) | Node.js ≥ 18, `npm`, and Google Chrome installed (for the headless QA). |

Nothing has to be installed to **use** the platform: it is plain HTML, CSS and JavaScript with no
mandatory build step (the bundle is already generated in `dist/`).

## 2. Quick start

```bash
cd latent-atlas
./serve.sh
```

The script rebuilds the bundle if a chapter changed (when Node is available), serves the platform at
**http://localhost:8765** and opens the browser. Another port: `./serve.sh 9000` (if the port is
taken, it moves to the next free one and prints the URL).

The server is `tools/atlas_server.py` (plain Python 3 standard library, already on macOS). Besides the
static files, it is the **local helper behind the MLX button**: it downloads models from Hugging Face,
installs `mlx-lm`, and starts/stops `mlx_lm.server` for the copilot. It listens only on `127.0.0.1` and
refuses requests from other websites. **Ctrl+C** stops everything, including the model server.

Alternatives:

```bash
open index.html                               # macOS: open straight from disk (file://)
python3 -m http.server 8765 --bind 127.0.0.1  # any OS; then open http://localhost:8765
```

Opening via `file://` (or through another static server) works for studying. For the copilot, prefer
`serve.sh`: without it the MLX button runs in manual mode (it shows the commands to copy), and some
servers (Ollama, LM Studio) reject calls coming from `file://`.

## 3. Language (English / Portuguese)

- **English is the default** for everyone, whatever the browser language.
- To study in Portuguese, click **PT** at the top right. The choice is saved in the browser (the
  button then reads **EN** to switch back).
- Direct link: `index.html?lang=pt` or `index.html?lang=en`.
- The copilot answers in the selected language.

## 4. Configure the copilot

Open it with the floating button (✦) or **⌘/Ctrl J** and click ⚙. Every provider uses the OpenAI
Chat Completions contract (`POST {base}/chat/completions`). After configuring, click
**Test connection**.

### 4.1 Local MLX on a Mac (no cloud, no key) — the **MLX** button

1. Run `./serve.sh` and open the platform (http://localhost:8765).
2. Click **MLX** in the top bar (between ⏸ and ☀). The modal shows an animated pipeline
   (Hugging Face → disk → unified memory → `mlx_lm.server` → copilot) and four steps:
   - **① Your Mac** — chip, RAM, GPU budget (≈ 2/3 of RAM up to 36 GB, 3/4 above), free disk, the
     Hugging Face cache and whether `mlx-lm` is installed. Missing? **Install mlx-lm** (runs
     `uv tool install --python 3.12 mlx-lm` or `brew install mlx-lm`, with a live log).
   - **② Choose a model** — the list is ranked for your RAM, with the **suggested** model on top (★).
     Each card shows parameters, bits, download size and a memory bar (weights + KV-cache headroom vs.
     GPU budget), grouped as **Fits well**, **Tight fit** and **Too big**. In **Add from Hugging Face**,
     paste an id (`org/model`) or the model URL: the platform asks the Hub, shows the real size, whether
     it is an MLX conversion, whether its architecture runs on your `mlx-lm` and whether it is gated, and
     from then on suggests it. Models you already downloaded appear under “On this Mac”.
   - **③ Download & run** — **Download and start**: a download with progress, speed and time left (cancel
     any time and resume later); then the server starts and a warm-up measures tok/s. Also: switch
     models, stop, show the log, and delete a download to free disk space.
   - **④ Copilot** — when the server is ready, the copilot is configured automatically
     (`http://127.0.0.1:8080/v1`). **Open copilot** or **Ask a test question**.
3. The dot on the MLX button shows the state on every page: gray (off), amber (downloading /
   loading), green (running), red (error).

Models live in the standard Hugging Face cache (`~/.cache/huggingface/hub`), shared with `mlx-lm`,
`transformers` and `huggingface_hub`. For gated models, accept the license on Hugging Face and start with
your token: `HF_TOKEN=hf_… ./serve.sh` (or `hf auth login`, if you have the `hf` CLI). The helper reads
the token at startup and only sends it to Hugging Face itself.

| Mac RAM | MLX button suggestion (mlx-community) | Download |
|---|---|---|
| 8 GB | `Qwen3-4B-Instruct-2507-4bit` | 2.3 GB |
| 16–24 GB | `gemma-4-12B-it-qat-OptiQ-4bit` (QAT, mixed 4/8-bit layers) · alternative `gemma-4-12B-it-4bit` | 8.9 GB · 6.8 GB |
| 32 GB | `gemma-4-26b-a4b-it-4bit` (MoE, ~4B active) | 15.4 GB |
| 48 GB | `Qwen3.6-35B-A3B-4bit` (MoE, ~3B active) | 20.4 GB |
| 64 GB+ | `Qwen3.6-35B-A3B-8bit` (MoE, near-lossless) | 37.7 GB |

Fit rule: weights + headroom (≥ 1 GB or 10%) ≤ 90% of the GPU budget. Small models (≤ 4B) get more
technical details wrong. With “thinking” on, a model can spend the whole token budget reasoning, which is
why **Ask for answers without “thinking”** is on by default for local providers. To test the whole
pipeline in a few minutes, use `Qwen3-0.6B-4bit` (351 MB).

**Without the helper** (page opened via `file://` or another server), the modal runs in manual mode:
model picker, commands to copy and **Check connection**. From a terminal:

```bash
./tools/start-mlx-copilot.sh          # picks the model from your RAM, installs mlx-lm if missing and starts the server
./tools/start-mlx-copilot.sh 24       # force a memory profile (8 | 16 | 24 | 32 | 48 | 64)
MODEL=mlx-community/Qwen3-0.6B-4bit ./tools/start-mlx-copilot.sh   # a specific model
# or by hand:
uv tool install --python 3.12 mlx-lm   # or: brew install mlx-lm
mlx_lm.server --model mlx-community/gemma-4-12B-it-qat-OptiQ-4bit --port 8080 --max-tokens 2048 --temp 0.4
```

In the copilot: provider **Local MLX (Mac)**, Base URL `http://127.0.0.1:8080/v1`, **empty** API key,
**empty** Model (it uses the model loaded by the server).

### 4.2 OpenAI / OpenRouter
Pick the provider, paste the **API key**, click **List models** and choose one. Reasoning models may
require `max_completion_tokens` and reject `temperature`: the copilot detects the error and retries
without the rejected fields.

### 4.3 LM Studio
Load a model (there is an MLX engine on Mac), open **Developer → Start Server** (port 1234) and
enable **CORS** in the server settings. Base URL: `http://localhost:1234/v1`.

### 4.4 Ollama
```bash
OLLAMA_ORIGINS="*" ollama serve
ollama pull qwen3:8b
```
Base URL: `http://localhost:11434/v1`, Model: `qwen3:8b`.

### 4.5 Your own recipe (vLLM)
The recipes studied here expose an OpenAI-compatible server (e.g. port 8888 in the GLM recipe). For
the browser to reach another machine, vLLM must accept CORS:
```bash
vllm serve … --allowed-origins '["*"]'
curl http://HEAD-IP:8888/v1/models
```
Base URL: `http://HEAD-IP:8888/v1` (paste the key if the server uses `--api-key`).

### 4.6 Where the configuration lives
Everything (including the API key) stays **only in this browser’s `localStorage`**. Do not use
production keys on shared machines. “Restore defaults” in the panel erases the configuration. The MLX
button’s choices (model, port, models you added) live under the `la:mlx:*` keys; downloaded models live
in the Hugging Face cache.

## 5. Using the platform

| Feature | How |
|---|---|
| Trail | Sidebar with the 14 parts, grouped into the 2 modules, in study order. ✓ marks completed chapters. |
| Search | **⌘/Ctrl K**: chapters, sections, passages, glossary, parameters. |
| Figures | Controls below each figure · **⤢** full screen · **✦ ask** sends the figure to the copilot · the ⏸ button at the top pauses every animation. |
| Copilot | **⌘/Ctrl J** · context *Section* / *Chapter* / *No context* · select a passage of the text and click “Ask the copilot”. |
| Local copilot (MLX) | **MLX** button at the top: picks, downloads and runs a model on your Mac and points the copilot at it (section 4.1). |
| Catalog | Sidebar → “Catalog of parameters & modifications”: 1,715 items filterable by recipe, kind and category. |
| Glossary | Underlined terms show their definition on hover; click to open the glossary. |
| Progress | A quiz at the end of each chapter · “mark as completed” · everything saved in the browser. |
| Theme and language | ☀/☾ at the top (dark/light) · **PT/EN** switches the language (English is the default). |
| Shortcuts | ⌘K search · ⌘J copilot · Shift+←/→ previous/next chapter · Esc closes panels. |

## 6. Edit, build and validate

```bash
npm install                       # only puppeteer-core (for QA); it drives your installed Google Chrome
node build.js                     # validates and builds dist/chapters.bundle.js (pt) and dist/chapters.en.bundle.js (en)
node build.js --lang en --check   # validate one language only
node tools/qa.js --shots          # headless QA of every chapter in pt (screenshots of each figure)
node tools/qa.js --lang en --mobile atencao mla   # QA of a few chapters, in English, on a phone viewport
node tools/i18n-check.js          # checks the English translation has the same structure and no leaked Portuguese
node tools/shot.js '#/c/atencao' shot.png --lang en --light   # screenshot of one page
node tools/mock-llm.js 18080      # fake OpenAI-compatible server to test the copilot without a model
python3 tools/test_atlas_server.py   # MLX helper tests (local fake hub; downloads and installs nothing)
```

QA screenshots and reports go to `qa-out/` (or to the folder set in `QA_OUT`).

- How to write or extend chapters: [`docs/AUTHORING.md`](docs/AUTHORING.md) (Portuguese)
- How to translate: [`docs/I18N.md`](docs/I18N.md)

## 7. Layout

```
index.html                   application shell (loads the data for the active language)
serve.sh                     serves at http://localhost:8765 (automatic rebuild) through tools/atlas_server.py
assets/css/                  core (tokens, layout), components (content vocabulary), copilot, mlx (MLX button and modal)
assets/js/i18n.js            language and UI dictionary (pt/en)
assets/js/core.js            router, trail, search, glossary, quiz, catalog, progress
assets/js/viz.js             visualization kit (lifecycle-managed canvas, matrices, UI, CSS 3D tensor)
assets/js/num.js             number-format arithmetic (FP8/FP4/MX/NVFP4, RNE, Hadamard, quantization)
assets/js/copilot.js         copilot (OpenAI-compatible + MLX)
assets/js/mlx.js             MLX button + modal (choose, add, download, start and connect the copilot)
chapters/NN-slug.*           chapters in pt (.html, .js, .glossary.json) and in en (.en.html, .en.js, .en.glossary.json)
data/manifest.js             trail parts (pt and en)
data/mlx-models.js           suggested MLX models (verified on Hugging Face) and the RAM fit rule
data/catalog[.en].js         catalog of parameters & modifications
data/glossary[.en].js        curated glossary
dist/chapters[.en].bundle.js bundles generated by build.js
docs/APANHADO.md · docs/APANHADO.en.md   every parameter and modification of the three recipes
docs/pesquisa/               research inventories behind the content (file:line references, measured numbers; Portuguese)
docs/AUTHORING.md · docs/I18N.md · docs/curriculum.json
tools/atlas_server.py        platform server + MLX helper (local API under /api/mlx/*, 127.0.0.1 only)
tools/                       qa.js · i18n-check.js · shot.js · mock-llm.js · start-mlx-copilot.sh · gen-apanhado.py · test_atlas_server.py
```

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Sidebar shows “Chapters that failed to load” | A chapter has a syntax error. Run `node build.js` and read the message. |
| Formulas show as `\( … \)` | No internet (KaTeX comes from a CDN). Connect and reload. |
| An English chapter shows “not translated yet” | The `.en` file for that chapter is missing; the Portuguese content is shown as a fallback. |
| Copilot: “Could not reach…” | Server not running, wrong port/Base URL (remember the `/v1`) or CORS. Use `./serve.sh` and enable CORS on the server. |
| Local MLX answers are empty | The model spent its tokens “thinking”. Turn on “Ask for answers without thinking” or raise the token limit. |
| MLX is very slow or hangs | The model is too big for your RAM. In the MLX button, pick a model from **Fits well** (green memory bar). |
| MLX button says “Manual mode — helper not running” | The page was opened via `file://` or by another server (for example an old `python3 -m http.server`). Stop it (Ctrl+C) and run `./serve.sh`. |
| Slow or interrupted download | Cancel whenever you like: the partial file is kept and **Resume download** continues where it stopped. |
| “Gated” model | Accept the license on the model page on Hugging Face and restart with `HF_TOKEN=hf_… ./serve.sh`. |
| “Unsupported” badge | The installed `mlx-lm` does not know that `model_type`. Click **Update** in step ① or pick another model. |
| Disk full | In step ③, **Delete download** removes the selected model from the Hugging Face cache. |
| `mlx_lm.server: command not found` | Run `./tools/start-mlx-copilot.sh` (installs via uv/brew) or open a new terminal after installing. |
| Port 8765 or 8080 in use | `./serve.sh` moves to the next free port by itself. For the model, the MLX button offers “Use port 8081” (or “Use that server for the copilot” when it is already an OpenAI-compatible server). |
| Progress and settings are gone | `localStorage` was cleared (private browsing, site data cleanup). |
| `node tools/qa.js` fails to start | Run `npm install` and make sure Google Chrome is in `/Applications` (or set `CHROME=/path/to/chrome`). |

## 9. Credits

The recipes used as the lab come from **Mia's AI Lab**:
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) ·
[Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) ·
[DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark).

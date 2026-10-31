# Mia's AI Lab — updates since the Latent Atlas snapshot (checked 2026-10-30)

Snapshots the platform was built on: GLM vLLM recipe `94ae731` (2026-09-28), GLM TensorFold recipe
`92bf731` = v1.3.2 (TensorFold v0.6.0 + 53 patches, 2026-10-01), Qwen dual and DeepSeek DSpark recipes
read on 2026-09-28. Clones used for this check lived in the session scratchpad (`MiaAI-Lab/*`, full
history). Every claim below was read in the repos; commit ids are given where they matter.

**Verdict: yes, the platform needs an update.** Three things changed underneath it:
1. TensorFold became Mia's default engine (the vLLM GLM README is now titled "Use the TensorFold recipe",
   commit `cafea0f`), and the engine itself was rewritten as a native Zig binary (TensorFold v1.0.0,
   2026-10-07; v1.1.9 on 2026-10-29). Many "TensorFold can't…" statements in the Atlas are now false.
2. The GLM TensorFold recipe moved from v1.3.2 (53 patches) to v1.10+ (123 patches): 3 Sparks, 8 requests,
   NVMe spill tier, KV sharding, Anthropic API, Apache-2.0 checkpoint.
3. New recipes scale the same models from 1 to 4 Sparks, including a **switchless 4-Spark ring**, and add
   new quantization formats (QAD FP4, EXL3-EAQ, IQ2_XXS).

Qwen dual: only one real feature (language-extended draft vocabularies). DeepSeek DSpark: credits only.

## 1. Must fix (now factually wrong)

| # | What the Atlas says | What is true now | Evidence | Where |
|---|---|---|---|---|
| 1 | Stay on vLLM for > 2 nodes / many requests; vLLM recipe "still maintained" as the main path | Upstream says everyone should use the TensorFold recipe | GLM vLLM `cafea0f` | ch42, ch48, ch49 step 8 |
| 2 | TensorFold has MLX + CUDA backends (Python) | Native Zig engine since v1.0.0 (Metal/CUDA kernels, no Python/MLX); Python line frozen at 0.6.6 | TensorFold README, tag v1.0.0 `8b003018a` | ch48 |
| 3 | TensorFold GLM needs exactly 2 ranks; no TP3/TP4 | TP=3 via `start-tp3.sh`, patches 0066–0068 (uneven 22/21 head split); TP4 on a branch | TF recipe v1.4 | ch00, 06, 12, 38, 42, 48, 49 |
| 4 | At most 4 requests (`MAX_SEGS = 4`) | `PARALLEL` 1–8 (patch 0069, v1.5): +27–36 % aggregate at 8 vs 4 | TF recipe v1.5 | ch00, 30–32, 34, 42, 45, 48, 49, catalog |
| 5 | Checkpoint is TR3-4bpw (mirror of brandonmusic), 120 shards, ShapleyMcg License | Both lanes default to `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` (vLLM @76c0b517, 83 shards, ~164 GiB; TF @078455ff), Apache-2.0, KL 2–18 % lower than TR3 | vLLM `98cf18f`; TF CHANGELOG v1.3.3 | ch00, 24, 25, 42, 47, 49, catalog |
| 6 | Quality numbers GSM8K 98.0 / HumanEval 97.6; KLD 0.0246 "holds for both" | New pack: GSM8K 98.8, HumanEval 95.7, HE+/MBPP+ 86.5; label 0.0246 as a TR3 measurement | TF README | ch25, 48, 49 |
| 7 | Fill metronome protects the newcomer's TTFT (1,024 rows, 50/50) | Sliced fill (0062) + smooth streaming (0061): protects replies already streaming (gaps 630 → 81 ms p50; new first token 16.1 → 21.1 s); `FILL_ROWS=2048` | TF v1.4, `cefd1b0` | ch32, 48, 49, catalog |
| 8 | Kept states evicted LRU, ~45 MiB each | Superseded first (0063), shared prefixes by recency (0077), forks kept (0120), compaction before eviction (0074), quotas (0089, 0097); ~91 MiB measured (0097) | TF patches | ch31, 33, 48, 49, catalog |
| 9 | Nucleus union is opt-in | On by default since v1.9 (stock sampling ~15 tok/s vs greedy ~58) | TF `3011f22` | ch49, catalog |
| 10 | No `/v1/messages`, no abliteration in TF; no logprobs | Anthropic Messages (0084), `ABLIT=1` (v1.7); native engine has logprobs (1.0.8). Still no API key in the recipe | TF recipe, TF engine | ch09, 25, 34, 42, 45, 48, 49 |
| 11 | Defaults: `TF_ROCE_WAIT_S` 20, `REQUEST_IMAGE_TOKENS` 16384 ("50 images") | 300 (v1.6, #54); 102,400 and vision workspace 1,440 MiB (≈ 1.41 GiB) — both after v1.10, commit `36df86e` (CHANGELOG "Unreleased"); `KV_POOL_GIB` 32 at TP3 (v1.4) | TF CHANGELOG | ch49 step 4, catalog |
| 12 | vLLM TP4 launcher doesn't forward E3/adaptive-k | `EXL3_FAT_GROUPED=1` (cold prefill +35–41 %), adaptive-k opt-in, reasoning effort honoured at TP3/TP4 | vLLM `e5dc885`, `1e8f928`, `4709bc5` | ch40, 42, catalog |

## 2. Should update (stale, but tied to a dated version)

- "53 patches / v1.3.2" appears in ~30 chapter update boxes and in Figure 49.6 → label "as of v1.3.2" once and add a changelog box.
- GLM vLLM scheduler: decode-floor v5 → **v7** (`b94fdfd`, #283: KV-refusal no longer stalls prefill, priority policy, fail-closed migration); 25 → 26 overlays; `--enable-prompt-tokens-details` in all launchers.
- TensorFold performance: `PARALLEL=8` and 3-Spark tables; patch 0085 (3-Spark decode at 590k: 66.2 → 184.4 tok/s); `tools/hermes_benchmark.py` exists now.
- Qwen: `MTP_DRAFT_VOCAB` can point to 7 shipped 65k language files (es, zh, ja, de, pt, fr, ru); acceptance per draft 47k → 65k: ru 0.39 → 1.55, ja 0.45 → 1.34, zh 0.45 → 1.14, de 0.81 → 1.40, es 1.35 → 1.88, fr 0.92 → 1.34, pt 0.96 → 1.39; English flat. Built by `files/build_draft_vocab_extend.py` (47k floor + ids < 400 + 18,387 most frequent ids from Wikipedia). Default is still 47k. → ch36, ch43, catalog.
- TensorFold engine: sampled requests get a fresh random seed by default (1.1.1, `--prompt-seed` restores the prompt key); GLM uses KV pages by default on CUDA (1.1.6). → ch48 keyed sampling, ch31.
- Research notes `glm-*.md`, `tensorfold-*.md`: append addenda, pin `src/…` links to the v0.6.x tags.

## 3. New concepts worth teaching (and how they transfer beyond DGX Spark)

| Concept | Source | Home in the Atlas | Transfers to H200 / clusters? |
|---|---|---|---|
| **Four Sparks, two ports: switch vs switchless ring** (rank placement 0-1-3-2, relayed hop, stock NCCL tree failure, patched `sparkring` NCCL) | DeepSeek V4.1 4x TF, DeepSeek v4.1 SGLang | ch40 new section + figure | Yes: rank-to-topology mapping, NCCL algorithm/topology, α-β cost per hop |
| **NVMe spill tier for prompt states** (2,048-token hashed blocks, CRC, read-ahead; 40k state restores in 147 ms; 94k prompt resumes in 3.0 s vs 232 s prefill) | TF 0088/0128, TensorFold `--spill-gib`, GLM 3x | ch33 | Yes: KV offload tiers (LMCache, Dynamo KV manager) |
| **KV sharding by conversation owner** (6,204 → 3,300 B/token/Spark; 1.87× pool) and context parallelism | TF 0129, GLM 3x, DeepSeek 2x | ch38 (DCP section), ch12, ch31 | Yes: DCP/CP on clusters |
| **Single-flight prefix prefill** (6 agents on a 12k system prompt: 47 s → 15 s) | TF `PREFIX_WAIT` | ch32/33 | Yes |
| **QAD** (quantization-aware distillation, vendor NVFP4) · **EAQ** mixed-width EXL3 · IQ2_XXS · INT4 with half the experts | Qwen single TF, DeepSeek 2x, Ornith, Nemotron Bolt | ch20, ch22 | QAD/NVFP4 yes (Blackwell); EXL3/IQ2 lab only |
| Measured KV vs prompt FP8 error (KL 0.0031 vs 0.0435); FP4 KV checked with McNemar | Qwen 27B TF, GLM 3x | ch23 | Yes |
| Lazy commit of recurrent state in speculation; MTP slower on bandwidth-bound MoE (0.87×) | Qwen 0012, Nemotron `SSM_LAZY`, Ornith | ch35, ch11 | Yes |
| UMA memory accounting (display reservation, per-CPU parked pages missed by `MemAvailable`) | TF 0072, 0122 | ch27 | Partly (UMA only) |
| Batch-invariant kernels, deterministic sampling in a production stack | DeepInfer (DeepSeek's engine), Nemotron Bolt | ch09, ch48 | Yes (RL, testing) |

**Case study candidate:** "DeepSeek-V4.1-Flash: one model, five deployments" — 2× Spark TF EXL3 TP2,
3× Spark SGLang TP3 (heads padded 64 → 96), 4× Spark SGLang TP4 (switch and ring), 4× Spark TF TP4 ring,
and DeepInfer on SM100. It is the best existing material for the platform's hardware-agnostic goal.

## 4. Material for the future Benchmarking & testing pillar

Harness artifacts (identical prompts inflating throughput, shared prefixes cutting TTFT, repeated-token
fillers hitting hot Engram rows, client over a slow link, random-word prompts lowering acceptance);
GPU clock capping and the slowest-GPU straggler (sparkLimiter, `-lgc 0,2200`); McNemar paired tests, ABAB
alternation and stated noise (±3–15 %); decode at depth; DeepInfer's Pareto sweeps, prefill/decode
separation and profiler overhead; correctness gates (token-hash replay, needle grids, mutation-tested
fixes, baseline → candidate → baseline with "inconclusive").

## 5. Upstream inconsistencies (do not copy)

TF README intro says 96 patches (image has 123); TF README says ~45 MiB per kept state (0097 measured ~91);
TF `config.sh` comments still cite 98.0 / 97.6 / 95.1; vLLM dense-h3 preset keeps TR3 SHA pins while the
default weights changed; DeepSeek SGLang 3-Spark decode 51.0 vs 37.9; DeepSeek 4x TF `POOL` tip vs default;
"EAQ" is never expanded.

## 6. Applied to the Atlas (2026-10-30)

All three phases were applied (PT and EN): dated "snapshot v1.3.2" tags added automatically to every GLM · TensorFold
callout and "Update … TensorFold" section (`LA.snapshots` in `data/manifest.js`); "what changed since v1.3.2" section in
ch49 (`#tf-desde-v132`); in-place fixes in ~30 chapters; new material in ch11, 13, 20, 22, 23, 27, 32, 33 (NVMe tier,
new figure), 34, 35, 36, 38, 40 (switch vs ring, new figure); DeepSeek-V4.1-Flash case study in ch44
(`#ds41-cinco-implantacoes`, new figure); catalog 1,715 → 1,734 items (45 updated). Two upstream facts stayed
unresolved and are flagged in the chapters instead of picked: the default sampling path (v0.6.0 code says top_k 20,
issue #91 measured the top_k-off path) and whether the two lanes' checkpoint revisions (@76c0b517 vs @078455ff) are
byte-identical.

/* ==========================================================================
   Chapter 47 — Lab: the recipe builder
   Lab model (LAB): memory, KV, speed and divisibility formulas
   + presets with numbers from the recipes (sources in the comments:
   inventories arch-*.md, *-launch.md, new-model-tutorial.md, vllm-ref.md).
   Everything that comes out of compute() is an ESTIMATE; the meas.* fields are MEASURED.
   ========================================================================== */
(function () {
  'use strict';

  /* ======================================================================
     1. MODELO
     ====================================================================== */
  const LAB = (function () {
    const GiB = 1073741824;
    const VIS = 121.69 * GiB;          // memória visível ao CUDA por GB10 (qwen-launch §4.1)
    // fora do orçamento: o que processo/driver alocam além do GMU, com o nó OCIOSO.
    // Qwen a 0,835: 114,5 − 101,61 = 12,9 GiB, dos quais 0,54 são os CUDA graphs que o boot não descontou
    // (o LAB os soma à parte, em cgRealGiB) → 12,36. SO e page cache consomem a folga que sobra.
    const HOST_EXT = 12.36 * GiB;
    // semáforo da folga OCIOSA estimada (calibrado nas histórias das receitas — interpretação):
    // Qwen 0,835 ≈ 7,2 → 0,3–0,9 GiB em soak (hard reset); Qwen 0,80 ≈ 11,4 → 4,5 GiB mínimos em 1 h de soak.
    const SLACK_RED = 8 * GiB, SLACK_GREEN = 11 * GiB;
    const FREE0 = 108.79 * GiB;        // livre no startup (Qwen, README)
    const BITS = { bf16: 16, fp8: 8 + 32 / 16384, nvfp4: 4.5, mxfp4: 4.25 };
    const EXL3_OVH = 0.0117;           // suh/svh FP16: 0,457 GB / 311,65 B params (cálculo nosso)
    const BLOCK = { bf16: 1, fp8: 128, nvfp4: 16, mxfp4: 32, exl3: 16 };
    const bitsOf = (f, bpw) => (f === 'exl3' ? (bpw || 4) + EXL3_OVH : BITS[f] || 16);
    const tauGeo = (a, k) => (k <= 0 ? 1 : Math.abs(1 - a) < 1e-9 ? k + 1 : (1 - Math.pow(a, k + 1)) / (1 - a));
    const alphaEq = (sum, k) => {
      if (k <= 0) return 0;
      if (sum >= k - 1e-6) return 0.999;
      let lo = 0, hi = 0.9999;
      for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (tauGeo(m, k) - 1 < sum) lo = m; else hi = m; }
      return (lo + hi) / 2;
    };
    const loc = (heads, P) => (heads >= P ? Math.ceil(heads / P) : 1);
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const lcm = (a, b) => (a * b) / gcd(a, b);
    const distinct = (E, n) => E * (1 - Math.pow(1 - 1 / E, n));
    const profSum = (p) => (p.a ? p.a.reduce((s, x) => s + x, 0) : p.sum);

    const PRESETS = {
      /* ---------------------------------------------------------- GLM */
      glm: {
        id: 'glm', rb: 'glm', short: 'GLM-5.3-Flash', name: 'GLM-5.3-Flash (EXL3 4 bpw, TP2)',
        src: 'GLM-5.3-Flash-EXL3-2x-DGX-Sparks recipe (default TP2 profile)',
        arch: '45 layers: 34 KDA (linear) + 11 sparse NoPE MLA (+ MTP) · 288 experts top-8 + 1 shared · hidden 4096 · vocab 154,880',
        ckpt: 'Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold', image: 'ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor', served: 'GLM-5.3-Flash-EXL3', mport: 29521,
        L: 45, hidden: 4096, qHeads: 64, padQ: { 3: 66 }, kvKind: 'mla', kvHeads: 1, linHeads: 64, padLin: { 3: 66 },
        E: 288, kE: 8, moeInter: 2048, expPacked: 2048, nMoE: 42, vocab: 154880, nPredict: 1, maxPos: 1048576,
        expParams: 25165824, Nexp: 311.65e9, Ndense: 9.67e9, denseNativeBytes: 19.34e9,
        extras: [
          { label: 'DFlash2 (drafter, BF16)', bytes: 2.34e9, when: (S) => S.specOn && S.spec === 'dflash' },
          { label: 'BF16 copy of the KDA in_proj (GLM53_KDA_BF16_LARGE_M)', rankBytes: 3.29 * GiB, when: (S) => S.denseFmt === 'fp8' && S.P !== 3 },
        ],
        ckptExtras: 0, replBytes: 0, overheadGiB: 8.0,
        cgEstGiB: 2.43, cgRealGiB: 0, // CG_ESTIMATE=1: estimativa 2,43 GiB; os graphs reais consomem −0,19 GiB (glm-launch §2.8)
        kvOpts: ['fp8'], kvWhy: 'the SM12x sparse-MLA kernel only accepts the fp8_ds_mla layout',
        bTok: () => 11 * (656 + 33),
        stateBytes: (S) => { const hh = S.P === 3 ? 22 : Math.ceil(64 / S.P); const k = S.specOn ? S.k : 0; return 34 * (hh * 128 * 128 * 4 + 3 * hh * 128 * (3 + k) * 2); },
        sSeqExtra: (S) => (S.specOn && S.spec === 'dflash' ? 5 * 2048 * (loc(8, S.P) * 128 * 2 * 2) : 0),
        bRead: () => 11 * 33, readFixed: (S) => Math.min(S.ctx, 2048) * 11 * 656,
        // alocador: por token = físico (README: "7.1 GiB per 1M"); termo fixo por requisição calibrado nos boots
        g: () => 1.0,
        sFix: (S) => {
          if (S.P === 3) return 5.323 * GiB;                                   // kit TP3: 40 GiB → 3,23× a 1M
          if (S.P === 2 && S.specOn && S.spec === 'dflash') return 1.5786 * GiB; // 11 GiB → 1,45× a 850k (compact)
          const k = S.specOn ? S.k : 0; return (1 + 4 * (2 + k)) * 7579 * 3584; // estrutural: kpool + 4 grupos KDA × (2+k) IDs
        },
        sFixSrc: (S) => (S.P === 3 ? 'calibrated on the TP3 kit: 40 GiB → 3.23× at 1M' : S.P === 2 && S.specOn && S.spec === 'dflash' ? 'calibrated: 11 GiB → 1.45× at 850k, compact pages' : 'structural estimate: (1 + 4·(2+k)) IDs × 25.9 MiB'),
        dAct: 3.735e9, dActBits: 16, lmRank2: 0.634e9, dRepl: 0,
        draft: {
          dflash: { kMax: 7, bytes: (S) => (1.17e9 + 0.634e9) * (2 / S.P) },
          mtp: { kMax: 3, bytes: (S) => S.k * 0.87e9 * (2 / S.P) },
          ngram: { kMax: 10, bytes: () => 0 },
        },
        nColl: 96, tColl: 25,
        profiles: [
          { id: 'estr', label: 'structured', k: 7, a: [0.98, 0.98, 0.94, 0.94, 0.91, 0.83, 0.83], src: 'DFlash2 k=7, lab, 2026-08-30 (per-position curve: sum 6.41; the aggregate counter says 6.71)' },
          { id: 'prosa', label: 'prose', k: 7, a: [0.75, 0.58, 0.41, 0.28, 0.16, 0.09, 0.06], src: 'DFlash2 k=7, lab, 2026-08-30 (per-position curve; aggregate 2.39)' },
        ],
        launch: { nodes: 2, ep: false, expFmt: 'exl3', bpw: 4, denseFmt: 'fp8', gmu: 0.85, fix: true, pool: 11, maxLen: 850000, ctx: 131072, seqs: 4, mnbt: 7168, kv: 'fp8', ssm: 'fp32', spec: 'dflash', k: 7, prof: 'estr', bw: 230, cgEst: true },
        meas: {
          cfg: {}, w: 82.94, wNote: 'Model loading took 82.94 GiB (dense FP8 + KDA BF16)',
          pool: 11, poolNote: 'fixed: --kv-cache-memory-bytes 11811160064',
          tokCfg: { fix: true, pool: 14 }, tok: 1572073, tokNote: '14 GiB, 850k, compact pages (2026-09-26); fixed term calibrated on the 11 GiB boot',
          dec: 65.1, decS: 6.41, decNote: 'structured, DFlash2 k=7, C=1 (lab, 2026-08-30); τ from the per-position curve (the bench aggregate says 6.71 accepted)',
        },
        pubSizes: { fp8: 328.3e9, nvfp4: 180e9, exl3: 175.62e9 }, recipeFmt: 'exl3',
      },
      /* --------------------------------------------------------- Qwen */
      qwen: {
        id: 'qwen', rb: 'qwen', short: 'Qwen3.8-Flash-Next', name: 'Qwen3.8-Flash-Next (NVFP4, TP2+EP)',
        src: 'Qwen3.8-Flash-Next-Dual-DGX-Sparks recipe (day-0 lane, .env.sample)',
        arch: '48 layers: 36 Gated DeltaNet + 12 QSA (+1 MTP) · 512 experts top-10 + 1 shared · hidden 2560 · N-gram table 51.2 B',
        ckpt: 'nvidia/Qwen3.8-Flash-Next-NVFP4', image: 'vllm/vllm-openai:qwen38-flash-next', served: 'qwen3.8-flash-next', mport: 50000,
        L: 48, hidden: 2560, qHeads: 24, kvKind: 'gqa', kvHeads: 2, linHeads: 48, linQK: 16,
        E: 512, kE: 10, moeInter: 640, nMoE: 48, vocab: 248320, nPredict: 1, maxPos: 262144, visionMLP: 4304,
        expParams: 4915200, Nexp: 120.80e9, Ndense: 4.95e9, denseNativeBytes: 9.9e9,
        extras: [
          { label: 'N-gram table (PLE) in FP8', bytes: 51.2e9, ckpt: true },
          { label: 'MTP layer', bytes: 2.695e9, ckpt: true, when: (S) => S.specOn && S.spec === 'mtp' },
          { label: 'vision tower (replicated)', bytes: 0.898e9, repl: true, ckpt: true },
        ],
        // overhead = não-torch 4,06 + pico de ativação 1,07 (boot GMU 0,835); os graphs (0,54) NÃO foram descontados
        replBytes: 1.45e9, overheadGiB: 5.13, cgEstGiB: 0.54, cgRealGiB: 0.54,
        kvOpts: ['bf16', 'fp8'],
        mtpExpBlock: 128, // experts da camada MTP em FP8 128×128 (card nvidia: "EP so the MTP expert width remains compatible")
        // KV principal 13 × 2 × h_kv × 256 × β + cache comprimido do indexador QSA 13 × 128 × 2 B / 4 = 832 B (12 QSA + MTP)
        bTok: (S) => 13 * 2 * loc(2, S.P) * 256 * (S.kv === 'fp8' ? 1 : 2) + 13 * 64,
        stateBytes: (S) => { const k = S.specOn && S.spec === 'mtp' ? S.k : 0; return 36 * ((48 / S.P) * 128 * 128 * (S.ssm === 'bf16' ? 2 : 4) + (10240 / S.P) * (3 + k) * 2); },
        sSeqExtra: () => 245760,
        bRead: () => 12 * 64, readFixed: (S) => Math.min(S.ctx, 2048) * 12 * 512,
        // alocador: por token ≈ físico (dois boots KV fp8/bf16 no mesmo L dão g = 0,985); termo fixo depende do dtype do estado
        g: () => 1.0,
        sFix: (S) => (S.ssm === 'bf16' ? 0.0888 : 0.4702) * GiB, // recalibrados com b_tok = 7.488 (mesmos boots)
        sFixSrc: (S) => (S.ssm === 'bf16' ? 'calibrated on the 2026-09-16 rerun: KV fp8, SSM BF16, assumed pool 32.02 GiB' : 'calibrated on the 2026-09-05 boot: 32.02 GiB → 13.93× at 262k'),
        dAct: 1.585e9, dActBits: 16, lmRank2: 0.634e9, dRepl: 1.28e9,
        draft: {
          mtp: { kMax: 4, bytes: (S) => S.k * (0.179e9 + 0.634e9) * (2 / S.P) },
          ngram: { kMax: 10, bytes: () => 0 },
        },
        nColl: 106, tColl: 25,
        profiles: [
          { id: 'geral', label: 'general', k: 3, a: [0.89, 0.745, 0.60], src: 'MTP=3 greedy, README (2026-09-05)' },
          { id: 'prosa', label: 'prose', k: 3, a: [0.81, 0.64, 0.51], src: 'MTP=3, prose prompts (rerun 2026-09-16)' },
        ],
        launch: { nodes: 2, ep: true, expFmt: 'nvfp4', bpw: 4, denseFmt: 'nativo', gmu: 0.80, fix: false, pool: 32, maxLen: 262144, ctx: 65536, seqs: 8, mnbt: 8192, kv: 'fp8', ssm: 'bf16', spec: 'mtp', k: 3, prof: 'geral', bw: 230, cgEst: false },
        meas: {
          cfg: { gmu: 0.835, ssm: 'fp32' }, w: 64.46, wNote: 'boot: weights 64.46 GiB (GMU 0.835)',
          pool: 32.02, poolNote: 'GMU 0.835, KV fp8 (graphs not subtracted)',
          tokCfg: { gmu: 0.835, fix: true, pool: 31.70, kv: 'bf16', ssm: 'fp32' }, tok: 2131159, tokNote: 'KV BF16, SSM FP32, 31.70 GiB, 262k (2026-09-05); fixed term calibrated on the FP8 boot',
          dec: 52.1, decS: 0.89 + 0.745 + 0.60, decNote: 'MTP=3 greedy, C=1 (2026-09-05); the recipe uses ~2.2 accepted/step, without the bonus, and claims to be at the roofline',
        },
        extraTok: { lbl: 'Qwen (vLLM 0.30)', cfg: { gmu: 0.80, kv: 'bf16', ssm: 'bf16' }, tok: 1467072, note: 'vLLM 0.30 lane, GMU 0.80, KV BF16 (5.6×): different engine, pool not published, fixed term not calibrated on that image' },
        pubSizes: { nvfp4: 132.64e9 }, recipeFmt: 'nvfp4',
      },
      /* ----------------------------------------------------- DeepSeek */
      ds: {
        id: 'ds', rb: 'ds', short: 'DeepSeek-V4-Flash', name: 'DeepSeek-V4-Flash (Vision-Exp, TP2)',
        src: 'DeepSeek-v4-Flash-DSpark-2x-DGX-Spark recipe (default Anemll 1M/6 lane)',
        arch: '43 layers: SWA 128 + C4A (21) / C128A (20) compression · 256 experts top-6 + 1 shared · hidden 4096 · 64 heads × 512',
        ckpt: 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp', rev: '86f746b36186f0e567729a5c06a8c918caba82a9', image: 'ghcr.io/anemll/dspark-vllm-gx10:0.1.1', served: 'deepseek-v4-flash-vision-exp', mport: 25000,
        L: 43, hidden: 4096, qHeads: 64, padQ: { 3: 72 }, kvKind: 'mqa', kvHeads: 1, oGroups: 8, padG: { 3: 9 },
        E: 256, kE: 6, moeInter: 2048, nMoE: 43, vocab: 129280, nPredict: 3, dsparkBlock: 5, maxPos: 1048576,
        expParams: 25165824, Nexp: 296.3e9, Ndense: 9.3e9, denseNativeBytes: 10.83e9,
        // overhead = 101,61 − 79,17 − 17,04 (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0: graphs fora do orçamento; tamanho não medido)
        extras: [], replBytes: 0, overheadGiB: 5.40, cgEstGiB: null, cgRealGiB: 0,
        kvOpts: ['fp8'], kvWhy: 'the backend uses the nvfp4_ds_mla layout (584 B/row, byte-identical to fp8_ds_mla)',
        bTok: () => 21 * 584 / 4 + 20 * 584 / 128 + 21 * 132 / 4,
        stateBytes: () => 0,
        sSeqExtra: () => 46 * 128 * 584,
        bRead: () => 21 * 132 / 4 + 20 * 584 / 128, readFixed: () => 21 * 640 * 584 + 46 * 128 * 584,
        // hypothesis: the excess in the log is PER TOKEN (37,376 B pages unified across 4 groups), as the README reads it
        // (6 × 200k fits). All published boots are at 1M: per token vs. per request cannot be separated.
        g: () => 2.04, sFix: null,
        sFixSrc: () => 'per-token hypothesis, g ≈ 2.04: all boots at 1M, not separable',
        dAct: 2.4e9, dActBits: 8.002, lmRank2: 0.53e9, dRepl: 0,
        draft: {
          dspark: { kMin: 5, kMax: 9, bytes: (S) => (1.43e9 + 0.033e9 * S.k) * (2 / S.P) },
          ngram: { kMax: 10, bytes: () => 0 },
        },
        nColl: 108, tColl: 40,
        profiles: [
          { id: 'engine', label: 'TP=3 probe', k: 6, a: [0.92, 0.74, 0.60, 0.41, 0.28, 0.21], src: 'DSpark k=6, 10 s engine windows, TP=3 probe (fable5-1) — not the TP2 config' },
          { id: 'prosa', label: 'natural prose', k: 6, sum: 1.56, src: 'acceptance 24.7–27.2% (TP2 A/B 2026-09-02)' },
        ],
        launch: { nodes: 2, ep: false, expFmt: 'mxfp4', bpw: 4, denseFmt: 'nativo', gmu: 0.835, fix: false, pool: 17, maxLen: 1048576, ctx: 131072, seqs: 6, mnbt: 8192, kv: 'fp8', ssm: 'fp32', spec: 'dspark', k: 6, prof: 'engine', bw: 230, cgEst: false },
        meas: {
          cfg: {}, w: 79.17, wNote: 'weights per rank, TP2 (fable5-1)',
          pool: 17.04, poolNote: 'Available KV cache memory (README)',
          tokCfg: { nodes: 3, fix: true, pool: 33.79 }, tok: 5021871, tokNote: 'TP3, 33.79 GiB (the smallest rank), 1M (4.79×); g calibrated on the TP2 boot (2,331,430)',
          dec: 56, decS: 3.16, decNote: 'c=1, 256 tokens, median of 8 (2026-09-02); τ from the TP=3 probe (different configuration; the report uses ~4.15)',
        },
        pubSizes: { mxfp4: 157 * GiB }, recipeFmt: 'mxfp4',
      },
      /* --------------------------------------------------------- MiMo */
      mimo: {
        id: 'mimo', rb: null, short: 'MiMo-V2.6-Flash', name: 'MiMo-V2.6-Flash-MOPD (MXFP4, tutorial plan)',
        src: '"Recipe from scratch" tutorial (plan; measured numbers from community kits)',
        arch: '48 layers: 9 global (GA) + 39 SWA with window 128 · 256 experts top-8, no shared expert · hidden 4096 · K/V 192/128',
        ckpt: 'XiaomiMiMo/MiMo-V2.6-Flash-MOPD', rev: '2479e2d0029eca9a34cc7e7f55a121925f81908e', image: 'vllm/vllm-openai:mimo-v26-aarch64-cu130', served: 'mimo-v2.6-flash-mopd', mport: 50000,
        L: 48, hidden: 4096, qHeads: 64, kvKind: 'gqa', kvHeads: 4, kvHeadsSWA: 8, dK: 192, dV: 128,
        E: 256, kE: 8, moeInter: 2048, nMoE: 47, vocab: 152576, nPredict: 3, maxPos: 1048576, qkvPreslice: 4,
        expParams: 25165824, Nexp: 302.8e9, Ndense: 5.98e9, denseNativeBytes: 8.89e9,
        extras: [
          { label: 'vision + audio encoders (replicated)', bytes: 1.7e9, repl: true, ckpt: true },
          { label: 'DFlash (5 SWA layers, BF16)', bytes: 2.94e9, when: (S) => S.specOn && S.spec === 'dflash' },
          { label: 'MTP (3 layers)', bytes: 1.19e9, ckpt: true, when: (S) => S.specOn && S.spec === 'mtp' },
        ],
        replBytes: 0.1e9, overheadGiB: 4.0, cgEstGiB: null, cgRealGiB: 0,
        kvOpts: ['bf16', 'fp8'],
        bTok: (S) => 9 * loc(4, S.P) * (192 + 128) * (S.kv === 'fp8' ? 1 : 2),
        stateBytes: () => 0,
        sSeqExtra: (S) => {
          const b = S.kv === 'fp8' ? 1 : 2, k = S.specOn ? S.k : 0;
          const win = (Math.ceil((128 + k) / 64) + 1) * 64;
          return 39 * loc(8, S.P) * 320 * b * win + (S.specOn && S.spec === 'dflash' ? 5 * loc(8, S.P) * 256 * 2 * 1024 : 0);
        },
        bRead: (S) => 9 * loc(4, S.P) * 320 * (S.kv === 'fp8' ? 1 : 2),
        readFixed: (S) => 39 * loc(8, S.P) * 320 * (S.kv === 'fp8' ? 1 : 2) * 128,
        // excesso POR TOKEN calibrado no PR #58128 (FP8, 17 GiB → 2.731.530 a 262k, descontado o fixo de 23,3 MB
        // das janelas SWA + DFlash: g ≈ 1,145; sem o fixo, 17 GiB ÷ 2.731.530 ÷ 5.760 B daria 1,16); o enchimento
        // de grupo do tutorial (as 9 globais ocupam 10 slots em grupos de 5: 10/9 ≈ 1,11) explica a maior parte
        g: () => 1.145, sFix: null,
        sFixSrc: () => 'g ≈ 1.145 calibrated on PR #58128 (FP8, 17 GiB); the tutorial group padding (10/9 ≈ 1.11) explains most of it; fixed = physical SWA windows',
        // dAct × 10,63/8 = 1,44 (qkv FP8) + 1,61 (o_proj BF16) + 0,2 (router, MLP densa): o router já está aqui (dRepl = 0)
        dAct: 2.445e9, dActBits: 10.63, lmRank2: 0.63e9, dRepl: 0,
        draft: {
          dflash: { kMax: 7, bytes: (S) => (1.47e9 + 0.63e9) * (2 / S.P) },
          // MTP de 3 camadas = 1,19 GB (~0,6 GB/rank): k ≤ 3 usa cada camada uma vez → ~0,2 GB/rank por passo + lm_head
          mtp: { kMax: 3, bytes: (S) => S.k * (0.2e9 + 0.63e9) * (2 / S.P) },
          ngram: { kMax: 10, bytes: () => 0 },
        },
        nColl: 96, tColl: 22,
        profiles: [
          { id: 'codigo', label: 'code', k: 7, sum: 5.23, src: 'DFlash k=7, tonyd2wild (vLLM TP2)' },
          { id: 'estr', label: 'structured', k: 7, sum: 6.85, src: 'DFlash k=7, tonyd2wild' },
          { id: 'prosa', label: 'prose', k: 7, sum: 1.23, src: 'DFlash k=7, tonyd2wild' },
        ],
        launch: { nodes: 2, ep: false, expFmt: 'mxfp4', bpw: 4, denseFmt: 'nativo', gmu: 0.85, fix: false, pool: 17, maxLen: 262144, ctx: 65536, seqs: 8, mnbt: 4096, kv: 'fp8', ssm: 'fp32', spec: 'dflash', k: 7, prof: 'codigo', bw: 230, cgEst: false },
        meas: {
          cfg: {}, w: 82.7, wNote: 'SGLang TP2 (MiaAI-Lab) — different engine',
          poolCfg: { gmu: 0.90, maxLen: 300000 }, pool: 13.89, poolNote: 'tonyd2wild: GMU 0.90, omni + DFlash-7, 300K',
          tokCfg: { fix: true, pool: 17, kv: 'bf16' }, tok: 1365757, tokNote: 'PR #58128: 17 GiB pool, KV BF16; g calibrated on FP8 (2,731,530)',
          dec: 70.5, decS: 5.23, decNote: 'code, DFlash k=7, C=1 (tonyd2wild)',
        },
        pubSizes: { mxfp4: 172.92e9, nvfp4: 187e9 }, recipeFmt: 'mxfp4',
      },
    };

    /* ---------------------------------------------- campos livres */
    const FREE_DEF = {
      L: 40, hidden: 4096, qHeads: 32, kvHeads: 4, dK: 128, dV: 128, attn: 'gqa', mlaDim: 576, nFull: 10,
      nLin: 30, linHeads: 32, dLin: 128, nSWA: 0, window: 128, E: 256, kE: 8, moeInter: 1536, nMoE: 40,
      denseB: 8, vocab: 151936, nPredict: 1, draftGB: 1.0,
    };
    function buildFree(F) {
      const expParams = 3 * F.hidden * F.moeInter;
      const emb = F.vocab * F.hidden;
      const dense = Math.max(F.denseB * 1e9, 2 * emb);
      const kvB = (S) => (S.kv === 'fp8' ? 1 : 2);
      const perLayerKV = (S) => (F.attn === 'mla' ? F.mlaDim * kvB(S) : loc(F.kvHeads, S.P) * (F.dK + F.dV) * kvB(S));
      return {
        id: 'free', rb: null, short: 'Free model', name: 'Free fields (illustrative example)', src: 'typed values (the initial example is illustrative, not a real model)',
        arch: `${F.L} layers: ${F.nFull} full attention (${F.attn.toUpperCase()}), ${F.nLin} linear, ${F.nSWA} SWA · ${F.E} experts top-${F.kE} · hidden ${F.hidden}`,
        ckpt: '<org>/<model>', image: 'vllm/vllm-openai:<tag>@sha256:<digest>', served: 'my-model', mport: 29500,
        L: F.L, hidden: F.hidden, qHeads: F.qHeads, kvKind: F.attn === 'mla' ? 'mla' : 'gqa', kvHeads: F.kvHeads, linHeads: F.nLin ? F.linHeads : 0,
        E: F.E, kE: F.kE, moeInter: F.moeInter, nMoE: F.nMoE, vocab: F.vocab, nPredict: F.nPredict, maxPos: 1048576,
        expParams, Nexp: F.nMoE * F.E * expParams, Ndense: dense, denseNativeBytes: dense * 2,
        extras: [], replBytes: 0, overheadGiB: 5.0, cgEstGiB: null, cgRealGiB: 0, kvOpts: ['bf16', 'fp8'],
        bTok: (S) => F.nFull * perLayerKV(S),
        stateBytes: (S) => F.nLin * (F.linHeads / S.P) * F.dLin * F.dLin * (S.ssm === 'bf16' ? 2 : 4),
        sSeqExtra: (S) => F.nSWA * F.window * perLayerKV(S),
        bRead: (S) => F.nFull * perLayerKV(S), readFixed: (S) => F.nSWA * F.window * perLayerKV(S),
        g: () => 1.0,
        // structural estimate of the hybrid allocator: the attention page grows until it holds the state of one linear layer;
        // each group of linear layers takes (2 + k) IDs per request, and each ID costs one page of the whole group.
        sFix: (S) => {
          if (!F.nLin || !F.nFull) return 0;
          const kv = perLayerKV(S), st = (F.linHeads / S.P) * F.dLin * F.dLin * (S.ssm === 'bf16' ? 2 : 4);
          const Bpg = Math.max(64, Math.ceil(st / kv / 64) * 64), pblk = F.nFull * kv * Bpg;
          const k = S.specOn ? S.k : 0;
          return Math.ceil(F.nLin / F.nFull) * (2 + k) * pblk;
        },
        sFixSrc: () => 'structural estimate: ⌈n_lin/n_full⌉·(2+k) IDs × page; check on the first boot',
        dAct: Math.max(0, (dense - 2 * emb) / 2), dActBits: 16, lmRank2: emb, dRepl: 0,
        draft: {
          mtp: { kMax: F.nPredict > 0 ? 4 : 0, bytes: (S) => S.k * F.draftGB * 1e9 * (2 / S.P) },
          dflash: { kMax: 15, bytes: (S) => F.draftGB * 1e9 * (2 / S.P) },
          dspark: { kMax: 15, bytes: (S) => F.draftGB * 1e9 * (2 / S.P) },
          ngram: { kMax: 10, bytes: () => 0 },
        },
        nColl: 2 * F.L + 6, tColl: 25,
        profiles: [{ id: 'ilustr', label: 'illustrative α 0.7', k: 3, sum: 0.7 + 0.49 + 0.343 }],
        launch: { nodes: 2, ep: false, expFmt: 'mxfp4', bpw: 4, denseFmt: 'bf16', gmu: 0.80, fix: false, pool: 16, maxLen: 262144, ctx: 65536, seqs: 8, mnbt: 8192, kv: 'fp8', ssm: 'bf16', spec: 'mtp', k: 3, prof: 'ilustr', bw: 230, cgEst: false },
        meas: null, pubSizes: {}, recipeFmt: 'mxfp4',
      };
    }

    function defaults(M) {
      const S = Object.assign({}, M.launch);
      const pr = M.profiles.find((p) => p.id === S.prof) || M.profiles[0];
      S.alpha = alphaEq(profSum(pr), pr.k);
      S.P = S.nodes; S.specOn = S.spec !== 'none' && !!M.draft[S.spec];
      S.g = null; S.sFix = null; // calculados pelo preset em compute() (dependem de TP, dtype, especulação)
      return S;
    }

    /* ---------------------------------------------- compute() */
    function compute(M, S0) {
      const S = Object.assign({}, S0);
      S.P = S.nodes;
      const dm = S.spec !== 'none' ? M.draft[S.spec] : null;
      S.specOn = !!(dm && (dm.kMax == null || dm.kMax > 0));
      S.kvForced = null;
      if (M.kvOpts.indexOf(S.kv) < 0) { S.kvForced = S.kv; S.kv = M.kvOpts[M.kvOpts.length - 1]; }
      if (S.g == null) S.g = M.g(S);
      if (S.cgEst == null) S.cgEst = !!M.launch.cgEst;
      S.ctx = Math.min(S.ctx, S.maxLen);
      /* pesos */
      const bE = bitsOf(S.expFmt, S.bpw);
      const Wexp = (M.Nexp * bE) / 8;
      const Wdense = S.denseFmt === 'nativo' ? M.denseNativeBytes : (M.Ndense * bitsOf(S.denseFmt)) / 8;
      let Wext = 0, WextRep = 0, Wrk = 0; const ext = [];
      (M.extras || []).forEach((x) => {
        if (x.when && !x.when(S)) return;
        ext.push(x);
        if (x.rankBytes) Wrk += x.rankBytes; else if (x.repl) WextRep += x.bytes; else Wext += x.bytes;
      });
      const Wtot = Wexp + Wdense + Wext + WextRep;
      const Wrep = (M.replBytes || 0) + WextRep;
      const Wrank = (Wtot - Wrep) / S.P + Wrep + Wrk;
      /* nó */
      const budget = S.gmu * VIS;
      const over = M.overheadGiB * GiB;                         // não-torch + pico de ativação
      const gEst = S.cgEst ? (M.cgEstGiB != null ? M.cgEstGiB : (M.cgRealGiB || 0)) * GiB : 0; // estimativa de graphs descontada
      const gReal = Math.max(0, (M.cgRealGiB || 0) * GiB);      // o que os graphs ocupam de fato
      const poolAuto = budget - Wrank - over - gEst;
      const pool = S.fix ? S.pool * GiB : poolAuto;
      const poolPos = Math.max(0, pool);
      const used = Wrank + over + poolPos + gReal;              // memória física do vLLM
      const slack = VIS - used - HOST_EXT;
      /* KV: cada requisição paga g·b_tok·L (cresce) + S_fix (estado + IDs de bloco, fixo) */
      const bTok = M.bTok(S);
      const perTok = bTok * S.g;
      const stateSeq = M.stateBytes(S);
      const sSeq = stateSeq + M.sSeqExtra(S);                   // estado físico
      if (S.sFix == null) S.sFix = M.sFix ? M.sFix(S) : null;
      const sFix = Math.max(sSeq, S.sFix == null ? 0 : S.sFix);
      S.sFix = sFix;
      const sIds = sFix - sSeq;                                 // cobrado pelo alocador além do estado físico
      const perReq = (L) => L * perTok + sFix;
      const cap = poolPos / perTok;                             // teto por token, ignorando os fixos
      const concMax = poolPos / perReq(S.maxLen);
      const concCtx = poolPos / perReq(S.ctx);
      const reported = concMax * S.maxLen;                      // "GPU KV cache size" = C(L_max) × L_max
      const Bm = Math.max(1, Math.min(S.seqs, Math.floor(concCtx) || 1));
      /* velocidade */
      const expFull = (M.expParams * bE) / 8;
      const dBits = S.denseFmt === 'nativo' ? M.dActBits : bitsOf(S.denseFmt);
      const bDense = M.dAct * (2 / S.P) * (dBits / 8) + M.lmRank2 * (2 / S.P) + (M.dRepl || 0);
      const tCom = S.P > 1 ? M.nColl * (M.tColl * 1e-6) * (S.P - 1) : 0;
      function step(B, spec, tauOverride) {
        const k = spec ? S.k : 0;
        const T = B * (k + 1);
        const n = M.kE * T;
        const Ed = distinct(M.E, n);
        const bExp = (M.nMoE * Ed * expFull) / S.P;
        const bDraft = spec ? dm.bytes(S) : 0;
        const bKV = B * (S.ctx * M.bRead(S) + M.readFixed(S));
        const bSt = B * 2 * stateSeq;
        const tot = bExp + bDense + bDraft + bKV + bSt;
        const tMem = tot / (S.bw * 1e9);
        const t = tMem + tCom;
        const tau = spec ? (tauOverride || tauGeo(S.alpha, k)) : 1;
        return { k, T, n, Ed, bExp, bDense, bDraft, bKV, bSt, tot, tMem, tCom, t, tau, tps: tau / t, agg: (B * tau) / t, B };
      }
      const s1 = step(1, false), s1s = S.specOn ? step(1, true) : null;
      const sB = step(Bm, false), sBs = S.specOn ? step(Bm, true) : null;
      return { S, M, bE, Wexp, Wdense, Wext, WextRep, Wtot, Wrep, Wrk, Wrank, ext, budget, over, gEst, gReal, poolAuto, pool, poolPos, used, slack, effGmu: used / VIS, bTok, perTok, stateSeq, sSeq, sFix, sIds, sFixSrc: M.sFixSrc ? M.sFixSrc(S) : '', perReq, cap, concMax, concCtx, reported, Bm, s1, s1s, sB, sBs, step, dm };
    }

    /* ---------------------------------------------- divisibilidade */
    function divChecks(M, S) {
      const P = S.nodes, out = [];
      if (P === 1) { out.push({ dim: 'TP=1', st: 'ok', txt: 'a single node: nothing is sliced' }); return out; }
      const q = M.qHeads;
      if (q % P === 0) out.push({ dim: 'Q heads', st: 'ok', txt: `${q} → ${q / P}/rank` });
      else { const pd = (M.padQ && M.padQ[P]) || Math.ceil(q / P) * P; out.push({ dim: 'Q heads', st: 'pad', txt: `${q} does not divide by ${P} → padding to ${pd} (${pd / P}/rank)`, sol: M.id === 'glm' ? 'GLM TP3: 66 (not 96: one rank would hold only fake heads and the logits collapse)' : M.id === 'ds' ? 'DeepSeek TP3: 72 heads and groups 8→9 (apply_tp3_patch)' : '' }); }
      if (M.kvKind === 'mla' || M.kvKind === 'mqa') out.push({ dim: 'KV', st: 'rep', txt: 'single latent replicated on every rank: the KV does not shrink with TP' });
      else {
        const kvChk = (kv, nm) => {
          if (kv % P === 0) out.push({ dim: nm, st: 'ok', txt: `${kv} → ${kv / P}/rank` });
          else if (kv < P && P % kv === 0) out.push({ dim: nm, st: 'rep', txt: `${kv} < ${P}: each KV head is replicated` });
          else out.push({ dim: nm, st: 'bad', txt: `${kv} KV heads with TP=${P}: GQA requires exact division or exact replication` });
        };
        kvChk(M.kvHeads, M.kvHeadsSWA ? 'KV (global)' : 'KV heads');
        if (M.kvHeadsSWA) kvChk(M.kvHeadsSWA, 'KV (SWA)');
      }
      if (M.linHeads) {
        if (M.linHeads % P === 0) out.push({ dim: 'linear heads', st: 'ok', txt: `${M.linHeads} → ${M.linHeads / P}/rank` });
        else out.push({ dim: 'linear heads', st: 'pad', txt: `${M.linHeads} does not divide by ${P} → padding (${(M.padLin && M.padLin[P]) || Math.ceil(M.linHeads / P) * P})`, sol: M.id === 'glm' ? 'GLM TP3: KDA 64→66 and A_log 64→66 at load time' : '' });
      }
      if (M.linQK && M.linQK % P) out.push({ dim: 'GDN QK heads', st: 'bad', txt: `${M.linQK} does not divide by ${P}` });
      if (M.oGroups) {
        if (M.oGroups % P === 0) out.push({ dim: 'o_groups', st: 'ok', txt: `${M.oGroups} → ${M.oGroups / P}/rank` });
        else out.push({ dim: 'o_groups', st: 'pad', txt: `${M.oGroups} groups do not divide by ${P} → ${(M.padG && M.padG[P]) || Math.ceil(M.oGroups / P) * P}`, sol: 'extra group with zeroed wo_a/wo_b and attn_sink = −∞' });
      }
      const dsNoEP = 'the DeepSeek recipe runs without EP (MegaMoE requires SM100); how the experts are handled on TP3 is not documented: see the TP=3 chapter';
      if (S.ep) {
        if (M.E % P === 0) out.push({ dim: 'experts (EP)', st: 'ok', txt: `${M.E} → ${M.E / P} whole experts/rank` });
        else out.push({ dim: 'experts (EP)', st: 'bad', txt: `${M.E} experts do not divide by ${P}: EP is not viable at this TP`, sol: M.id === 'ds' ? dsNoEP : '' });
      } else {
        const m = M.moeInter / P, blk = BLOCK[S.expFmt] || 1;
        if (M.expPacked && M.moeInter % P) out.push({ dim: 'intermediate', st: 'bad', txt: `${M.moeInter}/${P} is not an integer and the EXL3 trellis is packed in ${M.expPacked}`, sol: 'GLM TP3: expert parallel (96 experts/rank) instead of slicing' });
        else if (m % 1 !== 0) out.push({ dim: 'intermediate', st: 'bad', txt: `${M.moeInter}/${P} is not an integer`, sol: M.id === 'ds' ? dsNoEP : 'expert parallel, if E divides by P' });
        else if (m % blk !== 0) out.push({ dim: 'intermediate', st: 'bad', txt: `${M.moeInter}/${P} = ${m}, not a multiple of the ${S.expFmt.toUpperCase()} block of ${blk}`, sol: 'expert parallel, if E divides by P' });
        else out.push({ dim: 'intermediate', st: 'ok', txt: `${M.moeInter} → ${m}/rank (a multiple of the routed experts' block of ${blk})` });
        if (M.mtpExpBlock && m % 1 === 0 && m % M.mtpExpBlock) out.push({ dim: 'MTP FP8 experts', st: 'bad', txt: `${M.moeInter}/${P} = ${m} is not a multiple of the ${M.mtpExpBlock}×${M.mtpExpBlock} block of the FP8 experts in the MTP layer`, sol: 'Qwen: EP is mandatory because of MTP (whole 640-wide experts on each GPU); the NVFP4 routed experts, with a block of 16, would accept 320' });
      }
      // stock vLLM: pads the vocabulary to a multiple of 64 and requires exact division by P
      const v64 = Math.ceil(M.vocab / 64) * 64;
      if (v64 % P === 0) out.push({ dim: 'vocabulary', st: 'ok', txt: `${M.vocab.toLocaleString('en-US')} → ${v64.toLocaleString('en-US')} (multiple of 64) divides by ${P}` });
      else { const qv = lcm(64, P), vp = Math.ceil(M.vocab / qv) * qv; out.push({ dim: 'vocabulary', st: 'pad', txt: `${v64.toLocaleString('en-US')} (default vLLM padding, multiple of 64) does not divide by ${P}: needs an overlay with padding to lcm(64,${P}) = ${qv} → ${vp.toLocaleString('en-US')}`, sol: M.id === 'glm' ? 'overlay patch_tp3_glm.py from the GLM recipe: padding_size = 192, 154,880 → 154,944' : M.id === 'ds' ? 'DeepSeek: 129,280 does not divide by 3' : '' }); }
      if (M.visionMLP) {
        const v = M.visionMLP / P;
        if (v % 16 === 0) out.push({ dim: 'vision MLP', st: 'ok', txt: `${M.visionMLP}/${P} = ${v} (multiple of 16)` });
        else out.push({ dim: 'vision MLP', st: 'rep', txt: `${M.visionMLP}/${P} = ${Math.round(v * 100) / 100} is not a multiple of 16 (required by the BF16 vision kernels) → replicate the encoder`, sol: '--mm-encoder-tp-mode data' });
      }
      if (M.qkvPreslice) {
        if (P === M.qkvPreslice) out.push({ dim: 'pre-sliced QKV', st: 'ok', txt: `checkpoint sliced for TP=${M.qkvPreslice}: exact fit` });
        else if (M.qkvPreslice % P === 0) out.push({ dim: 'pre-sliced QKV', st: 'pad', txt: `each rank merges ${M.qkvPreslice / P} chunks of the FP8 QKV`, sol: 'requantize (changes 18–81% of the weights) or VLLM_MIMO_EXACT_QKV (#59053)' });
        else out.push({ dim: 'pre-sliced QKV', st: 'bad', txt: `QKV in ${M.qkvPreslice} chunks cannot be split across ${P} ranks` });
      }
      if (M.id === 'glm' && S.specOn && S.spec === 'dflash' && P === 3) out.push({ dim: 'drafter', st: 'pad', txt: 'DFlash2 32 Q / 8 KV does not divide by 3 → 36/9 and draft TP=1', sol: '"the divisibility tax" (pad-tp3-config.py)' });
      return out;
    }

    return { GiB, VIS, HOST_EXT, SLACK_RED, SLACK_GREEN, FREE0, BITS, BLOCK, bitsOf, tauGeo, alphaEq, distinct, loc, lcm, profSum, PRESETS, FREE_DEF, buildFree, defaults, compute, divChecks };
  })();

  /* ======================================================================
     2. CAPÍTULO
     ====================================================================== */
  LA.chapter({
    id: 'laboratorio',
    part: 9,
    order: 47,
    title: 'Lab: the recipe builder',
    navTitle: 'Lab',
    subtitle: 'An interactive bench that brings together the formulas of the trail: memory per node, KV, concurrency, decode ceiling and divisibility, with presets from the recipes and a commented .env + vllm serve skeleton.',
    minutes: 50,
    recipes: ['glm', 'qwen', 'ds'],
    tags: ['lab', 'memory budget', 'KV cache', 'concurrency', 'cost model', 'divisibility', 'vllm serve'],
    goals: [
      'Estimate <b>weights per rank</b> from parameters × effective bits, group by group',
      'Build the <b>budget of a node</b> (GMU, overhead, KV pool, host headroom) and read the boot lines',
      'Compute <b>KV capacity, concurrency</b> and the <b>decode ceiling</b> with and without speculation',
      'Spot <b>divisibility</b> traps and generate a <b>recipe skeleton</b> that is honest about what is an estimate',
    ],
    prereqs: ['receita-do-zero'],
    summary: [
      'Weights: \\(W = \\sum_g N_g\\,b_{\\text{eff},g}/8\\) with effective bits that include the scales (MXFP4 4.25; NVFP4 4.5; block FP8 ≈ 8.002); only the ~4-bit expert formats fit in 2 nodes.',
      'KV pool = GMU × 121.69 GiB − weights − overhead (− the CUDA graph estimate, if the profiler subtracts it), or a fixed value in bytes; ~12.4 GiB stay outside the budget, and the host headroom is what decides whether it crashes.',
      'Each request costs \\(g\\,b_{\\text{tok}}L + S_{\\text{fix}}\\): a term that grows with the context and a fixed term (state + block IDs of the hybrid allocator). Concurrency at \\(L\\) = pool ÷ that cost; the "GPU KV cache size" in the log is \\(C(L_{\\max})\\,L_{\\max}\\). <code>max-model-len</code> and <code>max-num-seqs</code> are caps, not reservations; \\(g\\) and \\(S_{\\text{fix}}\\) are calibrated from the log.',
      'Decode: \\(t_{\\text{step}} \\approx\\) bytes read ÷ bandwidth + collectives; in MoE the distinct experts \\(\\bar E\\) grow with the verified tokens, which is why speculation pays off less on prose.',
      'Divisibility: heads, KV, experts/intermediate, vocabulary and extras must divide by TP; otherwise padding, replication or EP, and a wrong shard usually fails silently.',
      'The lab is a hypothesis: compare it with the log (weights, pool, tokens) and with benchmarks; deviations above ~10% call for an explanation before moving on.',
    ],
    quiz: [
      { q: 'On DeepSeek (17.04 GiB pool, 2,331,430 tokens, <code>max-model-len</code> 1M, <code>max-num-seqs</code> 6), what happens if 6 clients send 1M prompts at the same time?', o: ['Immediate OOM at boot', 'All 6 fit, because the pool was reserved per slot', 'Only ~2 coexist; the others wait or are preempted and redone', 'vLLM cuts the context of all of them to 350k'], a: 2, e: 'Caps do not reserve memory: the constraint is: sum of live tokens ≤ pool. 2,331,430/1,048,576 = 2.22×, so 2 requests of 1M fit; vLLM does not OOM, it preempts and redoes the prefill. With 6 × 200k (1.2M tokens) all 6 fit.' },
      { q: 'How many effective bits per weight does NVFP4 have (E2M1 elements, E4M3 scale every 16, global FP32 scale)?', o: ['4.0', '4.25', '4.5', '8.0'], a: 2, e: '4 bits of the element + 8/16 of the block scale = 4.5 (the global scale vanishes, diluted over the tensor). MXFP4, with an E8M0 scale every 32, comes to 4.25.' },
      { q: 'On GLM with prose (low α), why does speculation with k=7 barely beat decode without speculation?', o: ['Because DFlash2 does not run on TP2', 'Because verifying 8 tokens makes the step read ~57 distinct experts per layer (vs. 8) and few drafts are accepted', 'Because the FP8 KV cache is slower than BF16', 'Because the CX7 bandwidth limits decode'], a: 1, e: 'With top-8 and 288 experts, 64 picks touch on average \\(\\bar E \\approx 57\\) experts. The step becomes ~3× more expensive in bytes, and on prose only ~3.3 tokens come out per step. Hence the adaptive-k in the recipe (verify only 2, 4 or 7).' },
      { q: 'Why does GLM on TP3 pad its 64 heads to 66 and not to 96?', o: ['66 is the only value safetensors accepts', 'With 96 (32 per rank), one rank would hold only fake KDA heads and the logits collapse', '96 does not fit in memory', 'The vocabulary requires 66'], a: 1, e: '66 = 3 × 22 spreads the 2 fake heads without emptying any rank; the kernel still pads 22 → 32 internally and slices back. The recipe measured the collapse with 96.' },
      { q: 'The boot shows far fewer KV tokens than pool ÷ physical bytes per token. What is the correct reading?', o: ['The byte math is wrong and should be ignored', 'It is the cost of the hybrid allocator (pages unified across groups, block IDs per group): calibrate the per-token factor and the fixed per-request cost from the log', 'The KV is in BF16 by mistake, always', 'Just lower max-model-len to recover the tokens'], a: 1, e: 'vLLM requires the same page for all KV groups and charges block IDs for each group per request. On GLM the per-token cost is the physical one and the excess is fixed (~1.6 GiB per request with compact pages; ~7.5 GiB without them). That is why shortening the context frees less than the per-token math suggests, and the "GPU KV cache size" (concurrency × cap) shrinks along with the cap, as the GLM recipe warns.' },
      { q: 'The Qwen recipe lowered the GMU from 0.835 to 0.80. What did it gain and what did it lose?', o: ['It gained faster decode; it lost quality', 'It gained host headroom (MemAvailable left the 0.3–0.9 GiB zone); it lost ~4 GiB of KV per node, without changing decode', 'It gained more KV; it lost stability', 'Nothing: GMU is just a warning'], a: 1, e: 'In unified memory, the GMU budget shares the same RAM with the OS. At 0.835 a soak left MemAvailable at 0.3–0.9 GiB (hard-reset risk); at 0.80, 4.5/7.8 GiB were left. Decode depends on bytes per step, not on the pool size.' },
    ],
    _lab: LAB,

    mount(root, LA) {
      const V = LA.viz, UI = LA.ui, h = LA.h;
      const { GiB, VIS, HOST_EXT, SLACK_RED, SLACK_GREEN, PRESETS } = LAB;
      const slackCol = (C, s) => (s < SLACK_RED ? C.red : s < SLACK_GREEN ? C.d : C.e);
      const MODELS = ['glm', 'qwen', 'ds', 'mimo'];

      /* ---------------------------------------------- estilos locais */
      const css = `
#la-article .la-fig .la-ctl{flex-wrap:wrap;max-width:100%}
#fig-construtor .lab-stage{padding:14px}
.lab-arch{font-size:13px;color:var(--text-2);margin:0 0 10px;line-height:1.5}
.lab-arch b{color:var(--text)}
.lab-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(108px,100%),1fr));gap:8px;margin-bottom:12px}
.lab-strip .la-readout{min-width:0}
.lab-strip .la-readout b{font-size:14px}
.lab-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:12px}
.lab-panel{border:1px solid var(--line);border-radius:12px;background:color-mix(in srgb,var(--surface-2) 45%,transparent);overflow:hidden;min-width:0}
.lab-ph{font-family:var(--font-display);text-transform:uppercase;letter-spacing:.1em;font-size:12px;color:var(--text-3);padding:9px 12px 0}
.lab-panel canvas{display:block;width:100%;height:240px}
.lab-alerts{margin-top:12px;display:grid;gap:6px}
.lab-alert{display:flex;gap:10px;align-items:flex-start;padding:8px 12px;border-radius:10px;border:1px solid var(--line);font-size:13.5px;color:var(--text-2);background:var(--surface);line-height:1.5}
.lab-alert i{flex:0 0 10px;height:10px;border-radius:50%;margin-top:6px;background:var(--c);box-shadow:0 0 8px var(--c)}
.lab-alert span{flex:1;min-width:0}
.lab-alert a{white-space:nowrap;font-size:12px}
.lab-free{display:none;margin-top:12px;padding:12px;border:1px dashed var(--line-2);border-radius:12px}
.lab-free.on{display:block}
.lab-free h5{margin:0 0 8px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-3)}
.lab-free-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(150px,100%),1fr));gap:8px 12px}
.lab-free label{display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em}
.lab-free input,.lab-free select{width:100%;box-sizing:border-box}
.lab-group{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;width:100%;padding:6px 0;border-bottom:1px dashed var(--line)}
.lab-group:last-child{border-bottom:none}
.lab-group>b.gt{font-family:var(--font-display);font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--pc,var(--mint));min-width:96px}
.lab-code-stage pre{margin:0!important;max-height:620px;overflow:auto;font-size:12.5px}
.lab-vt{padding:0 16px 14px;overflow-x:auto}
.lab-vt table{width:100%;border-collapse:collapse;font-size:13px}
.lab-vt th,.lab-vt td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;color:var(--text-2)}
.lab-vt th{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-3)}
.lab-vt table{min-width:640px}
@media (max-width:760px){.lab-group>b.gt{min-width:100%}.lab-alert{flex-wrap:wrap}.lab-alert a{margin-left:20px;flex-basis:100%}#fig-esteira .la-fig-stage{height:470px!important}#fig-bits .la-fig-stage{height:420px!important}}
`;
      const styleEl = h('style', null, css);
      document.head.appendChild(styleEl);
      LA.onCleanup(() => styleEl.remove());

      /* ---------------------------------------------- utilidades */
      const nf = (x, d = 0) => (isFinite(x) ? Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');
      const gi = (b, d = 1) => nf(b / GiB, d) + ' GiB';
      const gb = (b, d = 1) => nf(b / 1e9, d) + ' GB';
      const mb = (b, d = 1) => nf(b / 1e6, d) + ' MB';
      const tk = (n) => (n >= 1e6 ? nf(n / 1e6, 2) + ' M' : n >= 1e4 ? nf(n / 1e3, 0) + ' k' : nf(n, 0));
      const cx = (n) => (n >= 1048576 && n % 1048576 === 0 ? n / 1048576 + 'M' : n % 1024 === 0 ? n / 1024 + 'k' : Math.round(n / 1000) + 'k');
      const CTX = [8192, 16384, 32768, 65536, 131072, 200000, 262144, 300000, 524288, 850000, 1048576];
      const MNBT = [1024, 2048, 4096, 7168, 8192, 12288, 16384];
      const nearIdx = (arr, v) => { let bi = 0; arr.forEach((x, i) => { if (Math.abs(x - v) < Math.abs(arr[bi] - v)) bi = i; }); return bi; };
      const NAVT = { 'quantizacao-fundamentos': 'Quantization fundamentals', 'memoria-unificada': 'Unified memory', 'paged-attention-kv': 'PagedAttention & KV', 'kv-cache-quantizado': 'Quantized KV cache', 'hadamard-trelica-exl3': 'Hadamard, trellises & EXL3', 'medindo-qualidade': 'Measuring quality', 'fp8-fp4-microescala': 'FP8, FP4 & microscaling', 'tp3-tp4': 'TP=3 & TP=4', 'paralelismos': 'Parallelism', 'especulacao-fundamentos': 'Speculation: fundamentals', 'dflash-dspark': 'DFlash2 & DSpark', 'mtp-eagle-vocab': 'MTP, EAGLE & reduced vocab', 'scheduler-batching': 'Scheduler & batching', 'posicao-rope': 'Position & RoPE', 'benchmarking': 'Benchmarking', 'cuda-graphs-warmup': 'CUDA graphs & warmup' };
      const FMTL = { bf16: 'BF16', fp8: 'FP8', nvfp4: 'NVFP4', mxfp4: 'MXFP4', exl3: 'EXL3' };
      const SPECL = { none: 'none', mtp: 'MTP', dflash: 'DFlash', dspark: 'DSpark', ngram: 'n-gram' };
      function hatch(g, x, y, w, hh, color, step, r) {
        if (w <= 0 || hh <= 0) return;
        g.save(); V.rr(g, x, y, w, hh, r == null ? 3 : r); g.clip();
        g.strokeStyle = color; g.lineWidth = 1; g.beginPath();
        for (let i = -hh; i < w + hh; i += step || 6) { g.moveTo(x + i, y + hh); g.lineTo(x + i + hh, y); }
        g.stroke(); g.restore();
      }
      const wrapCache = new Map();
      function wrap(g, s, maxW, size, weight, font) {
        const key = s + '|' + Math.round(maxW) + '|' + size + '|' + (weight || 400) + (font || '');
        if (wrapCache.has(key)) return wrapCache.get(key);
        g.font = V.font(size, font || 'sans', weight || 400);
        const words = String(s).split(' '); const lines = []; let cur = '';
        words.forEach((wd) => { const test = cur ? cur + ' ' + wd : wd; if (g.measureText(test).width > maxW && cur) { lines.push(cur); cur = wd; } else cur = test; });
        if (cur) lines.push(cur);
        if (wrapCache.size > 600) wrapCache.clear();
        wrapCache.set(key, lines);
        return lines;
      }
      const ease = (cur, tgt, dt, k = 7) => cur + (tgt - cur) * Math.min(1, dt * k);
      function para(g, s, x, y, maxW, size, lh, o) {
        const L = wrap(g, s, maxW, size, (o && o.weight) || 400, o && o.font);
        L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({ size }, o || {})));
        return y + L.length * lh;
      }
      const sevCol = (C, s) => (s === 'bad' || s === 'risk' ? C.red : s === 'warn' ? C.d : s === 'ok' ? C.a : C.b);
      function segModels(ctl, value, onChange, label) {
        return UI.seg(ctl, { label: label || 'Model', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }, { v: 'mimo', label: 'MiMo' }], value, onChange });
      }
      const Rdef = {};
      MODELS.forEach((id) => { Rdef[id] = LAB.compute(PRESETS[id], LAB.defaults(PRESETS[id])); });

      /* ==================================================================
         FIGURA 1 — esteira
         ================================================================== */
      (function fig1() {
        const fig = root.querySelector('#fig-esteira');
        const R = Rdef.glm, s1s = R.s1s;
        const steps = [
          { t: 'Read the config.json', d: 'The fields that drive the recipe: <code>num_hidden_layers</code> and the layer types (full attention, linear, SWA), <code>hidden_size</code>, <code>n_routed_experts</code> and top-k, Q/KV heads, <code>vocab_size</code>, <code>quantization_config</code> and the available drafter.' },
          { t: 'Weigh the weights', d: 'Parameters × effective bits, group by group (experts, dense, extras), and the split per rank \\(W_{\\text{rank}} = (W - W_{\\text{rep}})/P + W_{\\text{rep}}\\). The \\(P\\) here is an initial guess: stop 6 confirms whether it divides the model.' },
          { t: 'Budget the node', d: 'GMU × 121.69 GiB, minus weights and overhead, gives the KV pool (or fix it in bytes). What is left outside the budget is the host headroom.' },
          { t: 'Count the KV', d: 'Bytes per token of the growing layers + fixed per-request cost (state and allocator block IDs) → concurrency for a context and the "GPU KV cache size" line in the log.' },
          { t: 'Time the step', d: 'Bytes read per step ÷ bandwidth + collectives, times the tokens per step \\(\\tau\\): the tok/s ceiling with and without speculation.' },
          { t: 'Slice across the nodes', d: 'Heads, experts, vocabulary and intermediate must divide by TP; otherwise, padding, replication or expert parallel. If none of that works out, go back to stop 2 with a different \\(P\\).' },
          { t: 'Write the skeleton', d: '<code>.env</code> + <code>vllm serve</code> with one comment per flag. Then: boot, log, calibration (the arrow back).' },
        ];
        const bub = [
          ['GLM-5.3-Flash: 45 layers (34 KDA + 11 NoPE MLA) + 1 MTP', '288 experts top-8 + 1 shared · hidden 4096', 'vocab 154,880 · DFlash2 drafter (block 8)'],
          [`experts 311.65 B × ${nf(R.bE, 3)} bits + dense 9.67 B (FP8 in memory)`, `total ≈ ${gb(R.Wtot)} → per rank ≈ ${gi(R.Wrank)} (estimate)`, 'measured at boot: 82.94 GiB per rank'],
          [`budget: 0.85 × 121.69 = ${gi(R.budget)}`, `weights ${gi(R.Wrank)} + overhead ${gi(R.over, 0)} + fixed pool of 11 GiB`, `estimated host headroom ≈ ${gi(R.slack)}`],
          [`${nf(R.bTok)} B/token/rank + ${nf(R.sFix / GiB, 2)} GiB fixed per request (${nf(R.sSeq / 1048576, 0)} MiB of state + block IDs)`, `concurrency at 850k ≈ ${nf(R.concMax, 2)}× (measured: 1.45×)`, `"GPU KV cache size" = C × 850k ≈ ${tk(R.reported)} (measured: 1,233,779)`],
          [`step with k=7: ~${gb(s1s.tot)} read → ${nf(s1s.t * 1000, 0)} ms`, `τ = ${nf(s1s.tau, 2)} tokens/step (structured profile)`, `ceiling ≈ ${nf(s1s.tps, 0)} tok/s (measured: 65.1)`],
          ['TP=2: 64 heads → 32/rank ✓ · experts sliced internally', 'MLA KV: the latent stays replicated on both ranks', 'TP=3 would require 64→66 heads and vocabulary 154,944'],
          ['--quantization exl3 · --kv-cache-dtype fp8', '--kv-cache-memory-bytes 11811160064', '--speculative-config dflash k=7 · capture 1 2 4 8 16 24 32'],
        ];
        const fml = [
          'what matters: layer types → KV groups; experts and top-k → bytes per step',
          'formula: W = Σ N_g·b_g/8   ·   W_rank = (W − W_rep)/P + W_rep',
          'formula: KV_pool = GMU × 121.69 GiB − W_rank − overhead (or a fixed value)',
          'formula: C(L) = KV_pool / (g·b_tok·L + S_fix)   ·   log: N = C(L_max)·L_max',
          'formula: t = bytes read / bandwidth + n_col·t_col   ·   tok/s = τ / t',
          'rule: P divides h_q; P divides E (with EP); (m/P) multiple of the block; V_pad multiple of lcm(64,P)',
          'the worker (rank 1, --headless) comes up first; then the head (rank 0) with the API',
        ];
        const st = UI.stepper(fig, steps, { autoplay: 6500 });
        const S = { prev: 0, from: 0, cur: 0 };
        const icons = [
          (g, x, y, c) => { V.rr(g, x - 8, y - 10, 16, 20, 2); g.strokeStyle = c; g.lineWidth = 1.6; g.stroke(); for (let i = 0; i < 3; i++) V.line(g, x - 4, y - 4 + i * 5, x + 4, y - 4 + i * 5, { color: c, lw: 1.2 }); },
          (g, x, y, c) => { V.line(g, x, y - 10, x, y + 9, { color: c, lw: 1.6 }); V.line(g, x - 10, y - 6, x + 10, y - 6, { color: c, lw: 1.6 }); V.line(g, x - 6, y + 9, x + 6, y + 9, { color: c, lw: 1.6 }); g.beginPath(); g.arc(x - 9, y + 1, 4, 0, Math.PI); g.strokeStyle = c; g.stroke(); g.beginPath(); g.arc(x + 9, y + 1, 4, 0, Math.PI); g.stroke(); },
          (g, x, y, c) => { g.strokeStyle = c; g.lineWidth = 1.6; g.strokeRect(x - 7, y - 11, 14, 22); for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) { g.fillStyle = c; g.fillRect(x - 4 + j * 5, y - 8 + i * 6, 3, 3); } },
          (g, x, y, c) => { for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { V.rr(g, x - 10 + j * 7, y - 10 + i * 7, 6, 6, 1.5); g.fillStyle = (i * 3 + j) < 6 ? c : 'transparent'; g.fill(); g.strokeStyle = c; g.lineWidth = 1; g.stroke(); } },
          (g, x, y, c) => { g.beginPath(); g.arc(x, y + 5, 11, Math.PI, 2 * Math.PI); g.strokeStyle = c; g.lineWidth = 2; g.stroke(); V.line(g, x, y + 5, x + 7, y - 3, { color: c, lw: 2 }); },
          (g, x, y, c) => { g.strokeStyle = c; g.lineWidth = 1.6; g.strokeRect(x - 11, y - 8, 22, 16); V.line(g, x, y - 12, x, y + 12, { color: c, lw: 1.4, dash: [3, 2] }); },
          (g, x, y, c) => { V.text(g, '>_', x, y + 1, { size: 15, font: 'mono', weight: 700, color: c, align: 'center', base: 'middle' }); },
        ];
        V.fig(fig, (g, w, hh, t, dt) => {
          const C = V.C();
          const cols = [C.b, C.a, C.d, C.e, C.c, C.f, C.a];
          const N = steps.length;
          const one = w >= 680;
          const pos = [];
          if (one) { const x0 = 56, x1 = w - 56; for (let i = 0; i < N; i++) pos.push([x0 + ((x1 - x0) * i) / (N - 1), 122]); }
          else { const x0 = 40, x1 = w - 40; const xs = [0, 1, 2, 3].map((i) => x0 + ((x1 - x0) * i) / 3); for (let i = 0; i < 4; i++) pos.push([xs[i], 76]); for (let i = 0; i < 3; i++) pos.push([xs[3 - i], 196]); }
          const rad = one ? Math.min(27, (w - 120) / 18) : 22;
          // esteira
          g.lineCap = 'round';
          g.beginPath(); pos.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.strokeStyle = V.alpha(C.muted, 0.25); g.lineWidth = 16; g.stroke();
          g.beginPath(); pos.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.strokeStyle = V.alpha(C.ink, 0.35); g.lineWidth = 2; g.setLineDash([3, 9]); g.lineDashOffset = -t * 30; g.stroke(); g.setLineDash([]);
          g.lineCap = 'butt';
          // seta de volta (medir → recalibrar)
          const pa = pos[N - 1], pb = pos[0];
          g.beginPath();
          if (one) { g.moveTo(pa[0], pa[1] + rad + 24); g.bezierCurveTo(pa[0], pa[1] + 96, pb[0], pb[1] + 96, pb[0], pb[1] + rad + 24); }
          else { g.moveTo(pa[0] - rad - 4, pa[1]); g.bezierCurveTo(10, pa[1], 10, pb[1], pb[0] - rad - 4, pb[1] + 4); }
          g.strokeStyle = V.alpha(C.c, 0.7); g.lineWidth = 1.6; g.setLineDash([5, 5]); g.lineDashOffset = t * 20; g.stroke(); g.setLineDash([]);
          if (one) V.text(g, '↺ measure the boot and recalibrate', (pa[0] + pb[0]) / 2, pa[1] + 90, { size: 12, color: C.c, align: 'center', weight: 600 });
          else V.text(g, '↺ measure and recalibrate', 16, (pa[1] + pb[1]) / 2 + 30, { size: 11, color: C.c, weight: 600 });
          // estações
          const labels = ['config.json', 'Weights', 'Node', 'KV', 'Speed', 'Slice', 'Skeleton'];
          pos.forEach((p, i) => {
            const on = st.i === i, col = cols[i];
            if (on) V.glow(g, p[0], p[1], rad * 2.1, col, 0.8);
            g.beginPath(); g.arc(p[0], p[1], rad + (on ? 3 : 0), 0, Math.PI * 2); g.fillStyle = on ? V.alpha(col, 0.3) : C.panel; g.fill();
            g.strokeStyle = on ? col : V.alpha(col, 0.55); g.lineWidth = on ? 2.4 : 1.4; g.stroke();
            icons[i](g, p[0], p[1], on ? C.ink : col);
            if (!one && i === 3) V.text(g, labels[i], p[0], p[1] - rad - 8, { size: 11.5, align: 'center', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
            else V.text(g, labels[i], p[0], p[1] + rad + 16, { size: 11.5, align: 'center', color: on ? C.ink : C.muted, weight: on ? 600 : 400 });
          });
          // personagem-cartão: caminha até a estação ativa
          if (st.i !== S.cur) { S.from = S.cur; S.cur = st.i; S.t0 = t; }
          const u = V.ease.inOut(V.clamp((t - (S.t0 || 0)) / 1.1, 0, 1));
          const a = pos[S.from], b = pos[S.cur];
          const px = V.lerp(a[0], b[0], u), py = V.lerp(a[1], b[1], u) - rad - 30 + Math.sin(t * 5) * 2;
          const walking = u < 1;
          V.box(g, px - 26, py - 14, 52, 28, { r: 8, fill: V.alpha(C.a, 0.22), stroke: C.a, lw: 1.6 });
          V.text(g, '{ json }', px, py + 1, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          const blink = (t % 3.2) < 0.12;
          g.fillStyle = C.ink; if (!blink) { V.dot(g, px - 8, py - 7, 1.8, C.ink); V.dot(g, px + 8, py - 7, 1.8, C.ink); }
          const lg = walking ? Math.sin(t * 16) * 4 : 0;
          V.line(g, px - 8, py + 14, px - 8 + lg, py + 21, { color: C.a, lw: 2 }); V.line(g, px + 8, py + 14, px + 8 - lg, py + 21, { color: C.a, lw: 2 });
          // balão do resultado
          const bh = one ? 118 : Math.max(118, hh - 262), by = hh - bh - 10, bx = 12, bw = w - 24;
          const col = cols[st.i];
          V.box(g, bx, by, bw, bh, { r: 12, fill: V.alpha(col, 0.08), stroke: V.alpha(col, 0.55), lw: 1.3 });
          let yy = para(g, (st.i + 1) + '. ' + steps[st.i].t.toUpperCase() + ' — GLM-5.3-Flash example (lab estimates)', bx + 14, by + 20, bw - 28, 11.5, 15, { weight: 700, color: col }) + 6;
          const fL = wrap(g, fml[st.i], bw - 32, 11.5, 400);
          const fade = V.clamp(st.elapsed() / 0.5, 0, 1);
          const yLim = by + bh - 8 - fL.length * 14;
          bub[st.i].forEach((ln) => { const L = wrap(g, ln, bw - 44, 12.5, 400, 'mono'); L.forEach((l, li) => { if (yy < yLim) { V.text(g, (li === 0 ? '▸ ' : '  ') + l, bx + 16, yy, { size: 12.5, font: 'mono', color: C.ink, alpha: fade }); yy += 17; } }); });
          fL.forEach((l, i) => V.text(g, l, bx + 16, by + bh - 10 - (fL.length - 1 - i) * 14, { size: 11.5, color: col, alpha: fade }));
        });
      })();

      /* ==================================================================
         FIGURA 2 — bits efetivos + checkpoint por formato
         ================================================================== */
      (function fig2() {
        const fig = root.querySelector('#fig-bits');
        const ctl = UI.controls(fig);
        const xs = LA.num.synthWeights(32, 7, 0.06, 8);
        const S = { fmt: 'nvfp4', m: 'glm' };
        const Q = {};
        Q.bf16 = { deq: xs.map((x) => LA.num.round('bf16', x)), blocks: [], scaleLbl: '', bits: 16 };
        (() => { const r = LA.num.fp8Tensor(xs, 'e4m3'); Q.fp8 = { deq: r.deq, blocks: [{ start: 0, n: 32 }], scaleLbl: 'FP32', bits: 8, note: 'here 1 scale per tensor; in the recipes, 1 per 128×128 block → 8.002 bits' }; })();
        (() => { const r = LA.num.mx(xs, 'e2m1', 32); Q.mxfp4 = { deq: r.deq, codes: r.codes, blocks: r.blocks.map((b) => ({ start: b.start, n: 32, s: b.scale })), scaleLbl: 'E8M0', bits: 4.25 }; })();
        (() => { const r = LA.num.nvfp4(xs, 16); Q.nvfp4 = { deq: r.deq, codes: r.codes, blocks: r.blocks.map((b) => ({ start: b.start, n: 16, s: b.scale })), scaleLbl: 'E4M3', bits: 4.5 }; })();
        Object.keys(Q).forEach((k) => { Q[k].st = LA.num.errStats(xs, Q[k].deq); });
        // cor normalizada pelo p95 de |x| (o outlier satura em vez de apagar o resto); erro relativo a um peso típico
        const absS = xs.map(Math.abs).sort((a, b) => a - b);
        const p95 = absS[Math.floor(0.95 * (absS.length - 1))];
        const errRef = 0.25 * p95;
        // elemento que define a escala de cada bloco (o maior |x| do bloco)
        Object.keys(Q).forEach((k) => Q[k].blocks.forEach((b) => { let bi = b.start; for (let i = b.start; i < b.start + b.n; i++) if (Math.abs(xs[i]) > Math.abs(xs[bi])) bi = i; b.amaxI = bi; }));
        const rB = UI.readout(ctl, 'effective bits', '—'), rS = UI.readout(ctl, 'SQNR', '—');
        UI.seg(ctl, { label: 'Block', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }, { v: 'mxfp4', label: 'MXFP4' }, { v: 'nvfp4', label: 'NVFP4' }], value: S.fmt, onChange: (v) => { S.fmt = v; upd(); } });
        segModels(ctl, S.m, (v) => { S.m = v; H.redraw(); }, 'Checkpoint');
        function upd() { rB.set(nf(Q[S.fmt].bits, S.fmt === 'mxfp4' ? 2 : S.fmt === 'nvfp4' ? 1 : 0) + ' bits'); rS.set(nf(Q[S.fmt].st.sqnr, 1) + ' dB'); H.redraw(); }
        function ckpt(M, fmt) {
          const bits = LAB.bitsOf(fmt, 4);
          const dense = fmt === 'bf16' ? M.Ndense * 2 : M.denseNativeBytes;
          const ex = (M.extras || []).filter((x) => x.ckpt).reduce((s, x) => s + x.bytes, 0);
          return (M.Nexp * bits) / 8 + dense + ex;
        }
        const H = V.fig(fig, (g, w, hh, t, dt, st, HH) => {
          const C = V.C();
          const q = Q[S.fmt];
          if (w < 240) return; // canvas ainda sem largura (layout em andamento)
          const narrowB = w < 520;
          const lw = narrowB ? 50 : 92;
          const x0 = 14 + lw, cw = (w - x0 - 14) / 32, ch = Math.min(26, cw * 1.25);
          const yS = 42, yO = 70, yD = yO + ch + 10, yE = yD + ch + 10;
          if (q.note) para(g, q.note, 14, 16, w - 28, 11.5, 14, { color: C.d });
          else V.text(g, 'a vector of 32 weights (synthetic, with outliers)', 14, 20, { size: 12, color: C.muted, font: 'mono', maxW: w - 28 });
          const rows = [[narrowB ? 'orig.' : 'original', xs, 'div'], [narrowB ? 'deq.' : 'dequantized', q.deq, 'div'], ['error', q.deq.map((d, i) => d - xs[i]), 'err']];
          const ys = [yO, yD, yE];
          const hov = HH.pointer.inside && HH.pointer.x > x0 && HH.pointer.y > yO - 30 && HH.pointer.y < yE + ch ? Math.floor((HH.pointer.x - x0) / cw) : -1;
          const showNum = cw >= 23;
          rows.forEach(([lbl, arr, mode], ri) => {
            V.text(g, lbl, 14, ys[ri] + ch / 2 + 4, { size: 11.5, color: C.muted });
            arr.forEach((v, i) => {
              const x = x0 + i * cw;
              const u = mode === 'err' ? Math.min(1, Math.abs(v) / errRef) : V.clamp(v / p95, -1, 1);
              g.fillStyle = mode === 'err' ? V.cmap(u, 'heat') : V.cmap(u, 'div');
              V.rr(g, x + 0.5, ys[ri], cw - 1.5, ch, Math.max(0.5, Math.min(4, cw / 4))); g.fill();
              if (i === hov) { g.strokeStyle = C.ink; g.lineWidth = 1.5; g.stroke(); }
              if (showNum && mode !== 'err') V.text(g, String(Math.round(v * 1000)), x + cw / 2 - 0.5, ys[ri] + ch / 2 + 4, { size: 11, font: 'mono', align: 'center', color: Math.abs(u) > 0.55 ? (C.dark ? '#08140E' : '#fff') : C.ink });
            });
          });
          // o elemento que define a escala de cada bloco: anel âmbar na linha original
          q.blocks.forEach((b) => { const x = x0 + b.amaxI * cw; g.strokeStyle = C.d; g.lineWidth = 2; V.rr(g, x - 1, yO - 2, cw + 0.5, ch + 4, 5); g.stroke(); });
          if (hov < 0 && narrowB) para(g, 'color: value ÷ p95(|x|) · amber outline: the weight that sets the scale · amber error ≥ 25% of a typical weight', 14, yE + ch + 26, w - 28, 11, 14, { color: C.muted });
          else if (hov < 0) para(g, `color: value ÷ p95(|x|) (lavender = negative, green = positive; the outlier saturates)${showNum ? ' · numbers in thousandths' : ''} · amber outline: the weight that sets the block scale · error: purple = small, amber ≥ 25% of a typical weight`, 14, yE + ch + 26, w - 28, 11, 14, { color: C.muted });
          // blocos e escalas
          q.blocks.forEach((b) => {
            const bx = x0 + b.start * cw, bw = b.n * cw - 1.5;
            V.line(g, bx + 1, yS + 12, bx + bw, yS + 12, { color: C.d, lw: 1.5 });
            V.line(g, bx + 1, yS + 12, bx + 1, yO - 3, { color: C.d, lw: 1.5 }); V.line(g, bx + bw, yS + 12, bx + bw, yO - 3, { color: C.d, lw: 1.5 });
            const sw = Math.max(34, Math.min(70, bw * 0.3));
            V.box(g, bx + bw / 2 - sw / 2, yS - 6, sw, 16, { r: 4, fill: V.alpha(C.d, 0.3), stroke: C.d, lw: 1.2 });
            V.text(g, q.scaleLbl, bx + bw / 2, yS + 2.5, { size: 10.5, font: 'mono', align: 'center', base: 'middle', color: C.ink });
          });
          if (!q.blocks.length) V.text(g, 'BF16: no scale — each weight stores its own exponent (8 bits)', x0, yS + 6, { size: 11.5, color: C.muted });
          // tooltip
          if (hov >= 0 && hov < 32) {
            const x = xs[hov], d = q.deq[hov];
            const code = q.codes ? q.codes[hov].v : null;
            const blk = q.blocks.find((b) => hov >= b.start && hov < b.start + b.n);
            const lines = [`weight #${hov}: ${x.toFixed(5)}`, code != null ? `E2M1 code: ${code}${blk && blk.s ? '  × scale ' + blk.s.toExponential(2) : ''}` : S.fmt === 'fp8' ? 'E4M3 code × FP32 scale' : 'BF16: 8 exponent bits', `dequantized: ${d.toFixed(5)}  (error ${(d - x).toExponential(1)})`];
            const tw = Math.min(290, w - 16), tx = V.clamp(x0 + hov * cw - tw / 2, 8, w - tw - 8), ty = yE + ch + 8;
            V.box(g, tx, ty, tw, 58, { r: 8, fill: C.panel, stroke: C.line || C.muted, lw: 1 });
            lines.forEach((l, i) => V.text(g, l, tx + 10, ty + 17 + i * 16, { size: 11.5, font: 'mono', color: C.ink, maxW: tw - 20 }));
          }
          // ------------- checkpoint inteiro em cada formato
          const M = PRESETS[S.m];
          const top = yE + ch + (narrowB ? 84 : 88);
          V.text(g, narrowB ? `${M.short}: checkpoint by format (estimate)` : `${M.short}: whole checkpoint in each expert format (estimate: parameters × bits)`, 14, top - 14, { size: 12, color: C.muted, font: 'mono', maxW: w - 28 });
          const fmts = ['bf16', 'fp8', 'nvfp4', 'mxfp4', 'exl3'];
          const vals = fmts.map((f) => ckpt(M, f));
          const oneGB = 121.69 * GiB; // bytes de um nó (121,69 GiB)
          const mx = Math.max(...vals, 2 * oneGB) * 1.06;
          const bx0 = 14 + (w < 520 ? 70 : 96), bw = w - bx0 - 16;
          const rowH = Math.min(22, (hh - top - 26) / fmts.length - 6);
          const X = (v) => bx0 + (v / mx) * bw;
          fmts.forEach((f, i) => {
            const y = top + i * (rowH + 6);
            const v = vals[i];
            const fits2 = v <= 2 * oneGB * 0.85;
            V.text(g, (M.recipeFmt === f ? '★ ' : '') + (f === 'exl3' && !narrowB ? 'EXL3 4 bpw' : FMTL[f]), 14, y + rowH / 2 + 4, { size: 11.5, color: M.recipeFmt === f ? C.ink : C.muted, weight: M.recipeFmt === f ? 700 : 400 });
            V.box(g, bx0, y, bw, rowH, { r: 5, fill: V.alpha(C.muted, 0.08) });
            V.box(g, bx0, y, Math.max(3, X(v) - bx0), rowH, { r: 5, fill: V.alpha(fits2 ? C.a : C.red, 0.55) });
            V.text(g, gb(v, 0), Math.min(X(v) + 6, w - 70), y + rowH / 2 + 4, { size: 11.5, font: 'mono', color: C.ink });
            const pub = M.pubSizes[f];
            if (pub) { const px = X(pub), py = y + rowH / 2; g.beginPath(); g.moveTo(px, py - 7); g.lineTo(px + 6, py); g.lineTo(px, py + 7); g.lineTo(px - 6, py); g.closePath(); g.fillStyle = C.d; g.fill(); g.strokeStyle = C.bg; g.lineWidth = 1; g.stroke(); }
          });
          const yb = top + fmts.length * (rowH + 6);
          [[oneGB, narrowB ? '1 node' : '1 node (121.69 GiB)'], [2 * oneGB, '2 nodes']].forEach(([v, l]) => {
            const x = X(v);
            V.line(g, x, top - 4, x, yb, { color: V.alpha(C.ink, 0.6), lw: 1.2, dash: [4, 4] });
            V.text(g, l, x + 4, yb + 13, { size: 11, color: C.muted });
          });
          V.text(g, '◆ published', w - 16, yb + 13, { size: 11, color: C.d, align: 'right' });
          void dt; void t; void st;
        }, { static: true });
        upd();
      })();

      /* ==================================================================
         FIGURA 3 — o prédio de 121,69 GiB
         ================================================================== */
      (function fig3() {
        const fig = root.querySelector('#fig-orcamento');
        const ctl = UI.controls(fig);
        const S = { m: 'qwen' };
        let B = LAB.defaults(PRESETS.qwen);
        let R = LAB.compute(PRESETS.qwen, B);
        const disp = { w: 0, o: 0, p: 0 };
        let lock = false;
        const rP = UI.readout(ctl, 'KV pool', '—'), rF = UI.readout(ctl, 'headroom (est.)', '—');
        segModels(ctl, S.m, (v) => { S.m = v; B = LAB.defaults(PRESETS[v]); lock = true; sG.set(B.gmu); tF.set(B.fix); sP.set(B.pool); tC.set(!!B.cgEst); lock = false; upd(); });
        const sG = UI.slider(ctl, { label: 'GMU', min: 0.7, max: 0.92, step: 0.005, value: B.gmu, fmt: (v) => v.toFixed(3), onInput: (v) => { B.gmu = v; if (!lock) upd(); } });
        const tF = UI.toggle(ctl, { label: 'fix pool (bytes)', value: B.fix, onChange: (v) => { B.fix = v; if (!lock) upd(); } });
        const sP = UI.slider(ctl, { label: 'fixed pool', min: 2, max: 48, step: 0.5, value: B.pool, fmt: (v) => nf(v, 1) + ' GiB', onInput: (v) => { B.pool = v; if (!lock) upd(); } });
        const tC = UI.toggle(ctl, { label: 'subtract graphs from pool (ESTIMATE_CUDAGRAPHS=1)', value: !!B.cgEst, onChange: (v) => { B.cgEst = v; if (!lock) upd(); } });
        function upd() { R = LAB.compute(PRESETS[S.m], B); rP.set(gi(R.poolPos)); rF.set(gi(R.slack)); }
        upd();
        V.fig(fig, (g, w, hh, t, dt) => {
          const C = V.C();
          const M = PRESETS[S.m];
          disp.w = ease(disp.w, R.Wrank, dt); disp.o = ease(disp.o, R.over, dt); disp.p = ease(disp.p, R.poolPos, dt);
          const y0 = 30, y1 = hh - 34, Hh = y1 - y0;
          const Y = (b) => y1 - (b / VIS) * Hh;
          const tx = w < 520 ? 46 : 78, tw = w < 520 ? 82 : Math.min(150, w * 0.2);
          // eixo
          [0, 30, 60, 90, 121.69].forEach((v) => { const y = Y(v * GiB); V.line(g, tx - 6, y, tx, y, { color: C.muted, lw: 1 }); V.text(g, nf(v, v % 1 ? 2 : 0), tx - 9, y + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }); });
          V.text(g, 'GiB', tx - 9, y0 - 12, { size: 11, color: C.muted, align: 'right', font: 'mono' });
          V.box(g, tx, y0, tw, Hh, { r: 8, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          // andares
          const segs = [];
          let acc = 0;
          const push = (v, col, lbl, val, kind) => { const ya = Y(acc), yb = Y(acc + v); segs.push({ ya, yb, col, lbl, val, kind }); acc += v; };
          push(disp.w, C.b, 'weights per rank', gi(R.Wrank) + ' (estimate)', 'w');
          push(disp.o, C.d, 'overhead (non-torch + activations)', gi(R.over), 'o');
          if (R.gReal > 0.05 * GiB) push(R.gReal, C.c, R.S.cgEst ? 'CUDA graphs (inside the budget)' : 'CUDA graphs (outside the budget)', gi(R.gReal, 2) + (R.S.cgEst ? ' · estimate subtracted' : ' · taken from headroom'), 'g');
          push(disp.p, C.a, 'KV pool' + (B.fix ? ' (fixed)' : ''), gi(R.poolPos), 'p');
          const ext = Math.min(HOST_EXT, Math.max(0, VIS - acc));
          push(ext, C.muted, 'process/driver beyond the GMU', gi(HOST_EXT, 1) + ' (Qwen, idle node)', 'x');
          const free = Math.max(0, VIS - acc);
          push(free, slackCol(C, R.slack), 'host headroom (idle)', gi(R.slack) + ' (estimate)', 'f');
          segs.forEach((s) => {
            const hS = s.ya - s.yb; if (hS <= 0.5) return;
            if (s.kind === 'x') { hatch(g, tx + 1, s.yb, tw - 2, hS, V.alpha(C.muted, 0.6), 7, 2); return; }
            if (s.kind === 'g') { V.box(g, tx + 1, s.yb, tw - 2, hS, { r: 2, fill: V.alpha(s.col, 0.35) }); hatch(g, tx + 1, s.yb, tw - 2, hS, V.alpha(s.col, 0.8), 4, 2); return; }
            if (s.kind === 'f') { V.box(g, tx + 1, s.yb, tw - 2, hS, { r: 2, fill: V.alpha(s.col, 0.16) }); return; }
            V.box(g, tx + 1, s.yb, tw - 2, hS, { r: 2, fill: V.alpha(s.col, s.kind === 'p' ? 0.3 : 0.55) });
            if (s.kind === 'p' && hS > 8) { g.beginPath(); for (let x = 0; x <= tw - 2; x += 3) { const yy = s.yb + 3 + Math.sin(x / 9 + t * 2.4) * 2.2; x ? g.lineTo(tx + 1 + x, yy) : g.moveTo(tx + 1 + x, yy); } g.strokeStyle = C.a; g.lineWidth = 2; g.stroke(); }
          });
          if (R.used + HOST_EXT > VIS) { const yT = Y(VIS); V.box(g, tx - 4, yT - 8, tw + 8, 8, { r: 3, fill: V.alpha(C.red, 0.8) }); V.text(g, 'overflow', tx + tw / 2, yT - 12, { size: 11, align: 'center', color: C.red, weight: 700 }); }
          // linha do GMU
          const yG = Y(R.budget);
          V.line(g, tx - 4, yG, tx + tw + 8, yG, { color: C.ink, lw: 1.4, dash: [5, 4] });
          // medido
          if (M.meas && M.meas.w) { const yM = Y(M.meas.w * GiB); V.line(g, tx - 10, yM, tx + 10, yM, { color: C.c, lw: 2.5 }); V.text(g, 'measured ' + nf(M.meas.w, 2), tx + 12, yM - 4, { size: 10.5, color: C.c, font: 'mono' }); }
          // rótulos com anticolisão
          const lx = tx + tw + 22, maxW = w - lx - 40;
          const ls = segs.filter((s) => s.kind !== 'f' || s.ya - s.yb > 0).map((s) => ({ y: (s.ya + s.yb) / 2, s }));
          ls.push({ y: yG, s: { ya: yG, yb: yG, col: C.ink, lbl: `budget: GMU ${nf(B.gmu, 3)} × 121.69`, val: gi(R.budget) } });
          ls.sort((a, b) => b.y - a.y);
          for (let i = 1; i < ls.length; i++) if (ls[i - 1].y - ls[i].y < 30) ls[i].y = ls[i - 1].y - 30;
          const minY = y0 + 14; if (ls.length && ls[ls.length - 1].y < minY) { const d = minY - ls[ls.length - 1].y; ls.forEach((l) => (l.y += d)); }
          ls.forEach((l) => {
            V.line(g, tx + tw + 2, (l.s.ya + l.s.yb) / 2, lx - 4, l.y - 4, { color: V.alpha(l.s.col, 0.7), lw: 1 });
            V.text(g, l.s.lbl, lx, l.y - 5, { size: 11.5, color: C.ink, weight: 600, maxW });
            V.text(g, l.s.val, lx, l.y + 9, { size: 11, color: C.muted, font: 'mono', maxW });
          });
          // semáforo
          const sx = w - 22, sc = [C.red, C.d, C.e];
          const act = R.slack < SLACK_RED ? 0 : R.slack < SLACK_GREEN ? 1 : 2;
          V.box(g, sx - 13, y0, 26, 70, { r: 8, fill: C.panel, stroke: V.alpha(C.muted, 0.4), lw: 1 });
          sc.forEach((c, i) => { g.beginPath(); g.arc(sx, y0 + 13 + i * 22, 8, 0, Math.PI * 2); g.fillStyle = i === act ? c : V.alpha(c, 0.18); g.fill(); if (i === act) V.glow(g, sx, y0 + 13 + i * 22, 16, c, 0.8); });
          if (w >= 520) V.text(g, M.short, w - 42, y0 + 8, { size: 11, color: C.muted, align: 'right' });
          if (w >= 700) {
            const nx = w - 250;
            para(g, `each +0.01 of GMU = +${nf(0.01 * 121.69, 2)} GiB of budget; here it all goes to KV (the Qwen recipe measured only +0.83 GiB).`, nx, y1 - 160, 230, 11.5, 15, { color: C.muted });
            para(g, R.S.cgEst && M.cgEstGiB > (M.cgRealGiB || 0) + 0.3 && !B.fix ? `graphs: ${nf(M.cgEstGiB, 2)} GiB estimated come out of the pool, but the graphs use ~${nf(Math.max(0, M.cgRealGiB), 1)}: KV reserved and never used.` : R.S.cgEst ? 'graphs: the profiler estimate comes out of the pool (with a fixed pool, nothing changes).' : 'graphs outside the budget: the pool loses nothing, but they come out of the host headroom.', nx, y1 - 100, 230, 11.5, 15, { color: C.muted });
            para(g, B.fix ? 'fixed pool: the GMU only limits weights + overhead; the headroom becomes your responsibility.' : 'turn on "fix pool" to see the KV stop following the GMU.', nx, y1 - 40, 230, 11.5, 15, { color: C.muted });
          }
        });
      })();

      /* ==================================================================
         FIGURA 4 — estacionamento de tokens
         ================================================================== */
      (function fig4() {
        const fig = root.querySelector('#fig-estacionamento');
        const ctl = UI.controls(fig);
        const S = { m: 'ds', ctxI: nearIdx(CTX, 200000), n: 6, kv: 'fp8', t0: 0 };
        let R;
        const rC = UI.readout(ctl, 'concurrency', '—'), rT = UI.readout(ctl, 'log: C × cap', '—');
        segModels(ctl, S.m, (v) => { S.m = v; upd(); });
        UI.slider(ctl, { label: 'context/req', min: 0, max: CTX.length - 1, step: 1, value: S.ctxI, fmt: (i) => cx(CTX[i]), onInput: (i) => { S.ctxI = i; upd(); } });
        UI.slider(ctl, { label: 'requests', min: 1, max: 16, step: 1, value: S.n, onInput: (v) => { S.n = v; upd(); } });
        UI.seg(ctl, { label: 'KV', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }], value: S.kv, onChange: (v) => { S.kv = v; upd(); } });
        function upd() {
          const M = PRESETS[S.m];
          const B = LAB.defaults(M); B.kv = S.kv; B.maxLen = Math.max(B.maxLen, CTX[S.ctxI]); B.ctx = CTX[S.ctxI];
          R = LAB.compute(M, B);
          rC.set(nf(R.concCtx, 2) + '×'); rT.set(tk(R.reported));
          S.dirty = true;
        }
        upd();
        V.fig(fig, (g, w, hh, t) => {
          const C = V.C();
          if (S.dirty) { S.t0 = t; S.dirty = false; }
          const M = PRESETS[S.m];
          const L = CTX[S.ctxI];
          const cell = w < 520 ? 11 : 14, gap = 2;
          const cols = Math.floor((w - 28) / (cell + gap));
          const grow = L * R.perTok, fixB = R.sFix, physB = R.sSeq;
          let yh = para(g, `${M.short} · pool ≈ ${gi(R.poolPos)}${R.S.kvForced ? ` (no ${R.S.kvForced.toUpperCase()} KV on the GB10: using FP8)` : ''}`, 14, 18, w - 28, 12, 15, { font: 'mono', color: R.S.kvForced ? C.d : C.ink });
          const mbg = (b) => (b >= 1e8 ? gi(b, 2) : mb(b, 1));
          const hdr = (cbv, nq) => `each square = ${mb(cbv, 0)} · 1 request of ${cx(L)}: ${nf(R.bTok)} B/token × g ${nf(R.S.g, 2)} × ${cx(L)} = ${mbg(grow)} + ${mbg(fixB)} fixed (state ${mbg(physB)} + block IDs ${mbg(R.sIds)}) → ${nq} squares`;
          const hdr3 = `fixed term: ${R.sFixSrc}`;
          // measure the header before deciding how many rows fit (the queue, the legend and the footer go below)
          const nHdr = wrap(g, hdr(R.poolPos / 1e3, 999), w - 28, 11.5, 400, 'mono').length * 14 + wrap(g, hdr3, w - 28, 11, 400).length * 14;
          const rows = Math.max(6, Math.floor((hh - yh - 8 - nHdr - (w < 520 ? 132 : 104)) / (cell + gap)));
          const N = cols * rows, cb = R.poolPos / N;
          const nG = Math.max(1, Math.ceil(grow / cb)), nF = Math.round(fixB / cb), nP = Math.min(nF, Math.round(physB / cb));
          yh = para(g, hdr(cb, nG + nF), 14, yh + 2, w - 28, 11.5, 14, { font: 'mono', color: C.muted });
          yh = para(g, hdr3, 14, yh, w - 28, 11, 14, { color: C.muted });
          const gx = 14 + (w - 28 - cols * (cell + gap)) / 2, gy = yh + 6;
          // alocação
          const need = nG + nF;
          const fit = Math.min(S.n, Math.floor(N / need));
          const prog = V.clamp((t - S.t0) / 1.3, 0, 1);
          const shown = Math.floor(prog * fit * need);
          for (let i = 0; i < N; i++) {
            const r = Math.floor(i / cols), c = i % cols;
            const x = gx + c * (cell + gap), y = gy + r * (cell + gap);
            const req = Math.floor(i / need), off = i % need;
            if (i < shown && req < fit) {
              const pal = [C.a, C.b, C.d, C.c, C.f, C.e], col = pal[req % pal.length];
              if (off < nP) { V.rr(g, x, y, cell, cell, 2.5); g.fillStyle = V.alpha(col, 0.25); g.fill(); hatch(g, x, y, cell, cell, col, 4, 2.5); }
              else if (off < nF) { V.rr(g, x, y, cell, cell, 2.5); g.fillStyle = V.alpha(col, 0.14); g.fill(); g.strokeStyle = V.alpha(col, 0.9); g.lineWidth = 1; g.stroke(); V.dot(g, x + cell / 2, y + cell / 2, cell / 5, V.alpha(col, 0.9)); }
              else { V.rr(g, x, y, cell, cell, 2.5); g.fillStyle = V.alpha(col, 0.75); g.fill(); }
            } else { V.rr(g, x, y, cell, cell, 2.5); g.fillStyle = V.alpha(C.muted, 0.12); g.fill(); }
          }
          // rótulos das requisições
          for (let q = 0; q < fit && prog >= 1; q++) {
            let i0 = q * need; while (i0 % cols > cols - 2 && i0 < (q + 1) * need - 1) i0++;
            const r = Math.floor(i0 / cols), c = i0 % cols;
            if (need >= 3 && cell >= 13) V.text(g, '#' + (q + 1), gx + c * (cell + gap) + 1, gy + r * (cell + gap) + cell - 2, { size: 9.5, color: C.dark ? '#00210F' : '#fff', weight: 700 });
          }
          // fila
          const yq = gy + rows * (cell + gap) + 18;
          const waiting = S.n - fit;
          V.text(g, waiting > 0 ? `queue / preemption: ${waiting} request(s) do not fit right now` : (fit === S.n ? `all ${S.n} requests fit · ${nf(100 * (1 - (fit * need) / N), 0)}% of the pool left` : ''), 14, yq, { size: 12, color: waiting > 0 ? C.red : C.a, weight: 600, maxW: w - 28 });
          for (let q = 0; q < Math.min(waiting, 12); q++) {
            const x = 14 + q * 46 + ((t * 8 + q * 3) % 4) - 2, y = yq + 12;
            if (x + 40 > w - 10) break;
            const col = [C.a, C.b, C.d, C.c, C.f, C.e][(fit + q) % 6];
            V.box(g, x, y, 38, 16, { r: 5, fill: V.alpha(col, 0.45), stroke: col, lw: 1 });
            V.dot(g, x + 9, y + 17, 3, C.ink); V.dot(g, x + 29, y + 17, 3, C.ink);
            V.text(g, '#' + (fit + q + 1), x + 19, y + 11.5, { size: 9.5, align: 'center', color: C.ink, weight: 700 });
          }
          // legenda dos três tipos de quadradinho
          {
            let lx2 = 14, ly2 = hh - (w < 520 ? 62 : 34);
            [['grows with the context', 'full'], ['physical state', 'hatch'], ['block IDs (allocator)', 'dot']].forEach(([nm, kind]) => {
              g.font = V.font(11); const iw = g.measureText(nm).width + 26;
              if (lx2 + iw > w - 10) { lx2 = 14; ly2 += 15; }
              V.rr(g, lx2, ly2 - 9, 11, 11, 2); g.fillStyle = V.alpha(C.a, kind === 'full' ? 0.75 : kind === 'hatch' ? 0.25 : 0.14); g.fill();
              if (kind === 'hatch') hatch(g, lx2, ly2 - 9, 11, 11, C.a, 4, 2);
              if (kind === 'dot') { g.strokeStyle = C.a; g.lineWidth = 1; g.stroke(); V.dot(g, lx2 + 5.5, ly2 - 3.5, 2.2, C.a); }
              V.text(g, nm, lx2 + 15, ly2, { size: 11, color: C.muted }); lx2 += iw;
            });
          }
          const limit = R.concCtx >= S.n ? 'the number of requests (or max-num-seqs)' : 'the KV pool';
          para(g, `concurrency at ${cx(L)} ≈ ${nf(R.concCtx, 2)}× → limited by: ${limit}`, 14, hh - (w < 520 ? 26 : 12), w - 28, 11.5, 14, { font: 'mono', color: C.muted });
        });
      })();

      /* ==================================================================
         FIGURA 5 — anatomia de um passo de decode
         ================================================================== */
      (function fig5() {
        const fig = root.querySelector('#fig-passo');
        const ctl = UI.controls(fig);
        const S = { m: 'ds', B: 1, ctxI: nearIdx(CTX, 8192), bw: 230, alpha: null };
        let R;
        const rT = UI.readout(ctl, 'ceiling w/ spec', '—'), rA = UI.readout(ctl, 'aggregate', '—');
        segModels(ctl, S.m, (v) => { S.m = v; S.alpha = null; syncA(); upd(); });
        UI.slider(ctl, { label: 'batch B', min: 1, max: 16, step: 1, value: S.B, onInput: (v) => { S.B = v; upd(); } });
        UI.slider(ctl, { label: 'context', min: 0, max: CTX.length - 1, step: 1, value: S.ctxI, fmt: (i) => cx(CTX[i]), onInput: (i) => { S.ctxI = i; upd(); } });
        UI.slider(ctl, { label: 'bandwidth', min: 150, max: 273, step: 1, value: S.bw, fmt: (v) => v + ' GB/s', onInput: (v) => { S.bw = v; upd(); } });
        const sA = UI.slider(ctl, { label: 'alpha', min: 0.05, max: 0.99, step: 0.01, value: 0.7, fmt: (v) => v.toFixed(2), onInput: (v) => { S.alpha = v; upd(); } });
        function syncA() { const M = PRESETS[S.m]; const d = LAB.defaults(M); sA.input.value = d.alpha; sA.el.querySelector('.val').textContent = d.alpha.toFixed(2); S.alpha = d.alpha; }
        function upd() {
          const M = PRESETS[S.m];
          const B = LAB.defaults(M); B.ctx = CTX[S.ctxI]; B.maxLen = Math.max(B.maxLen, B.ctx); B.bw = S.bw; if (S.alpha != null) B.alpha = S.alpha;
          R = LAB.compute(M, B);
          const a = R.step(S.B, false), b = R.S.specOn ? R.step(S.B, true) : null;
          S.a = a; S.b = b;
          rT.set(b ? nf(b.tps, 0) + ' tok/s' : '—'); rA.set(nf((b || a).agg, 0) + ' tok/s');
        }
        syncA(); upd();
        V.fig(fig, (g, w, hh, t) => {
          const C = V.C();
          const M = PRESETS[S.m], a = S.a, b = S.b;
          const tMax = Math.max(a.t, b ? b.t : 0) * 1.08;
          const lx = 14, bw = w - 28;
          const cats = [['experts', 'bExp', C.b], ['dense + lm_head', 'bDense', C.a], ['draft', 'bDraft', C.c], ['KV + state', 'bKV', C.d]];
          V.text(g, w < 520 ? `${M.short} · B=${S.B} · ${cx(CTX[S.ctxI])} · ${S.bw} GB/s` : `${M.short} · TP${R.S.P} · batch B=${S.B} · context ${cx(CTX[S.ctxI])} · ${S.bw} GB/s (estimate)`, 14, 16, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
          const bars = [['without speculation', a], [b ? `with ${SPECL[R.S.spec]} k=${R.S.k}` : 'no drafter in this preset', b]];
          bars.forEach(([lbl, s], i) => {
            const yl = 34 + i * 64, y = yl + 14;
            V.text(g, lbl, 14, yl + 8, { size: 11.5, weight: 600, color: C.ink });
            if (!s) return;
            let x = lx;
            cats.forEach(([, key, col]) => {
              const bytes = key === 'bKV' ? s.bKV + s.bSt : s[key];
              const ww = ((bytes / (R.S.bw * 1e9)) / tMax) * bw;
              if (ww > 0.5) { V.box(g, x, y, ww, 26, { r: 3, fill: V.alpha(col, 0.6) }); if (ww > 56) V.text(g, gb(bytes, 1), x + ww / 2, y + 17, { size: 11, font: 'mono', align: 'center', color: C.ink }); }
              x += ww;
            });
            const wc = (s.tCom / tMax) * bw;
            if (wc > 0.5) hatch(g, x, y, wc, 26, C.muted, 5, 3);
            V.text(g, `${nf(s.t * 1000, 1)} ms/step · τ=${nf(s.tau, 2)} → ${nf(s.tps, 0)} tok/s per stream`, lx, y + 40, { size: 11.5, font: 'mono', color: C.ink, maxW: bw });
          });
          // legenda
          let lxx = 14, ly = 170;
          cats.concat([['communication', null, C.muted]]).forEach(([nm, , col]) => {
            g.font = V.font(11); const iw = g.measureText(nm).width + 30;
            if (lxx + iw > w - 10) { lxx = 14; ly += 16; }
            V.box(g, lxx, ly - 9, 10, 10, { r: 2, fill: V.alpha(col, 0.7) });
            V.text(g, nm, lxx + 14, ly, { size: 11, color: C.muted });
            lxx += iw;
          });
          // curva de experts distintos
          const narrow = w < 620;
          const px0 = 54, py0 = ly + 30, pw = narrow ? w - px0 - 20 : w * 0.56 - px0, ph = narrow ? 140 : hh - py0 - 34;
          const nS = b ? b.n : a.n;
          const nMax = Math.max(nS * 1.5, M.E * 1.2);
          V.line(g, px0, py0 + ph, px0 + pw, py0 + ph, { color: C.muted, lw: 1 }); V.line(g, px0, py0, px0, py0 + ph, { color: C.muted, lw: 1 });
          V.text(g, 'Ē (distinct experts per layer)', px0, py0 - 6, { size: 11, color: C.muted });
          V.text(g, 'kₑ·T picks in the step', px0 + pw, py0 + ph + 16, { size: 11, color: C.muted, align: 'right' });
          V.text(g, String(M.E), px0 - 6, py0 + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' });
          V.text(g, '0', px0 - 6, py0 + ph, { size: 11, color: C.muted, align: 'right', font: 'mono' });
          g.beginPath(); for (let i = 0; i <= 80; i++) { const n = (nMax * i) / 80; const x = px0 + (n / nMax) * pw, y = py0 + ph - (LAB.distinct(M.E, n) / M.E) * ph; i ? g.lineTo(x, y) : g.moveTo(x, y); } g.strokeStyle = C.b; g.lineWidth = 2.2; g.stroke();
          const lim = Math.min(nMax, M.E); V.line(g, px0, py0 + ph, px0 + (lim / nMax) * pw, py0 + ph - (lim / M.E) * ph, { color: V.alpha(C.muted, 0.6), lw: 1, dash: [4, 4] });
          [[a, C.a, 'no spec'], [b, C.c, 'with spec']].forEach(([s, col, l], i) => {
            if (!s) return;
            const x = px0 + (s.n / nMax) * pw, y = py0 + ph - (s.Ed / M.E) * ph;
            V.line(g, x, y, x, py0 + ph, { color: V.alpha(col, 0.5), dash: [2, 3] });
            V.dot(g, x, y, 5 + Math.sin(t * 3 + i) * 0.8, col);
            const lbl = `${l}: ${nf(s.n)} → Ē ≈ ${nf(s.Ed, 1)}`;
            if (narrow) {
              // no celular os pontos ficam colados à curva: rótulos no canto inferior direito, com linha-guia
              const lxr = px0 + pw - 4, lyr = py0 + ph - 14 - (1 - i) * 17;
              g.font = V.font(11, 'sans', 600); const tw2 = g.measureText(lbl).width;
              V.line(g, x + 5, y, lxr - tw2 - 6, lyr - 4, { color: V.alpha(col, 0.6), lw: 1, dash: [2, 3] });
              V.text(g, lbl, lxr, lyr, { size: 11, color: col, weight: 600, align: 'right' });
            } else V.text(g, lbl, x + 8, y - 8 - i * 14, { size: 11, color: col, weight: 600, maxW: pw });
          });
          // números à direita/abaixo
          const nx = narrow ? 14 : w * 0.6, ny = narrow ? py0 + ph + 40 : py0 + 16;
          const big = b || a;
          V.text(g, 'CEILING PER STREAM', nx, ny, { size: 11, color: C.muted, weight: 600 });
          V.text(g, nf(big.tps, 0) + ' tok/s', nx, ny + 34, { size: narrow ? 26 : 32, font: 'display', weight: 700, color: b ? C.c : C.a });
          V.text(g, b ? `no spec: ${nf(a.tps, 0)} tok/s · gain ${nf(b.tps / a.tps, 2)}×` : 'no drafter in this preset', nx, ny + 56, { size: 11.5, color: C.ink, maxW: w - nx - 10 });
          V.text(g, `aggregate with B=${S.B}: ${nf(big.agg, 0)} tok/s`, nx, ny + 76, { size: 11.5, color: C.ink, maxW: w - nx - 10 });
          if (!narrow) {
            V.text(g, `step: ${gb(big.tot, 1)} read ÷ ${S.bw} GB/s`, nx, ny + 104, { size: 11, font: 'mono', color: C.muted, maxW: w - nx - 10 });
            V.text(g, `+ ${M.nColl} collectives × ${M.tColl} µs = ${nf(big.tCom * 1000, 1)} ms`, nx, ny + 120, { size: 11, font: 'mono', color: C.muted, maxW: w - nx - 10 });
            V.text(g, 'excludes slow kernels and CPU:', nx, ny + 144, { size: 11, color: C.muted });
            V.text(g, 'the recipes measure 76–98% of it', nx, ny + 160, { size: 11, color: C.muted });
          }
        });
      })();

      /* ==================================================================
         FIGURA 6 — aceitação por posição
         ================================================================== */
      (function fig6() {
        const fig = root.querySelector('#fig-aceite');
        const ctl = UI.controls(fig);
        const curves = [];
        MODELS.forEach((id) => PRESETS[id].profiles.forEach((p) => curves.push({ v: id + ':' + p.id, label: PRESETS[id].short + ' · ' + p.label + ' (k=' + p.k + ')', p, M: PRESETS[id] })));
        const S = { c: curves[0].v, am: 0.6 };
        const rE = UI.readout(ctl, 'equivalent alpha', '—'), rTau = UI.readout(ctl, 'measured tau', '—');
        UI.select(ctl, { label: 'Curve', options: curves.map((c) => ({ v: c.v, label: c.label })), value: S.c, onChange: (v) => { S.c = v; upd(); } });
        UI.slider(ctl, { label: 'manual alpha', min: 0.05, max: 0.99, step: 0.01, value: S.am, fmt: (v) => v.toFixed(2), onInput: (v) => { S.am = v; H.redraw(); } });
        let cur;
        function upd() { cur = curves.find((c) => c.v === S.c); const sum = LAB.profSum(cur.p); cur.sum = sum; cur.aeq = LAB.alphaEq(sum, cur.p.k); rE.set(nf(cur.aeq, 3)); rTau.set(nf(1 + sum, 2)); H.redraw(); }
        const H = V.fig(fig, (g, w, hh) => {
          const C = V.C();
          const k = cur.p.k;
          const narrow = w < 560;
          const px0 = 46, py0 = 26, pw = (narrow ? w - 30 : w * 0.66) - px0, ph = hh - py0 - (narrow ? 104 : 40);
          const X = (i) => px0 + ((i - 0.5) / k) * pw, Y = (v) => py0 + ph - v * ph;
          [0, 0.25, 0.5, 0.75, 1].forEach((v) => { V.line(g, px0, Y(v), px0 + pw, Y(v), { color: C.grid, lw: 1 }); V.text(g, nf(v, 2), px0 - 6, Y(v) + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }); });
          for (let i = 1; i <= k; i++) V.text(g, String(i), X(i), py0 + ph + 16, { size: 11, color: C.muted, align: 'center', font: 'mono' });
          V.text(g, 'draft position', px0 + pw, py0 + ph + 30, { size: 11, color: C.muted, align: 'right' });
          const line = (fn, col, dash, lw) => { g.beginPath(); for (let i = 1; i <= k; i++) { const x = X(i), y = Y(fn(i)); i > 1 ? g.lineTo(x, y) : g.moveTo(x, y); } g.strokeStyle = col; g.lineWidth = lw || 2; if (dash) g.setLineDash(dash); g.stroke(); g.setLineDash([]); };
          line((i) => Math.pow(cur.aeq, i), C.b, [6, 4], 2);
          line((i) => Math.pow(S.am, i), C.d, [2, 4], 2);
          for (let i = 1; i <= k; i++) V.dot(g, X(i), Y(Math.pow(cur.aeq, i)), 3, C.b);
          if (cur.p.a) { line((i) => cur.p.a[i - 1], C.a, null, 2.4); cur.p.a.forEach((v, i) => { V.dot(g, X(i + 1), Y(v), 5.5, C.a); V.text(g, nf(v, 2), X(i + 1), Y(v) - 10, { size: 10.5, align: 'center', color: C.ink, font: 'mono' }); }); }
          else V.text(g, `only the published mean: ${nf(cur.sum, 2)} accepted per step (no per-position curve)`, px0 + 8, py0 + 14, { size: 11.5, color: C.a, weight: 600, maxW: pw - 16 });
          V.text(g, cur.p.src || '', px0 + 8, py0 + ph - 8, { size: 10.5, color: C.muted, maxW: pw - 16 });
          // barras de τ
          const tauM = 1 + cur.sum, tauMan = LAB.tauGeo(S.am, k);
          const items = [['τ measured', tauM, C.a], ['τ geom. (α_eq)', LAB.tauGeo(cur.aeq, k), C.b], ['τ (manual α)', tauMan, C.d]];
          const bx = narrow ? 14 : w * 0.72, by = narrow ? hh - 70 : py0 + 10, bwid = narrow ? w - 28 : w - bx - 14;
          const mx = k + 1;
          items.forEach(([l, v, col], i) => {
            const y = by + i * (narrow ? 22 : 44);
            V.text(g, l, bx, y + (narrow ? 12 : 0), { size: 11.5, color: C.ink, weight: 600 });
            const x0 = narrow ? bx + 110 : bx, yb = narrow ? y + 2 : y + 8, ww = (narrow ? bwid - 150 : bwid - 60) * (v / mx);
            V.box(g, x0, yb, Math.max(2, ww), 14, { r: 4, fill: V.alpha(col, 0.65) });
            V.text(g, nf(v, 2), x0 + ww + 6, yb + 11, { size: 11.5, font: 'mono', color: C.ink });
          });
          if (!narrow) V.text(g, `maximum possible: ${k + 1}`, bx, by + 3 * 44 + 6, { size: 11, color: C.muted });
        }, { static: true });
        upd();
      })();

      /* ==================================================================
         FIGURA 7 — o fatiador
         ================================================================== */
      (function fig7() {
        const fig = root.querySelector('#fig-fatiador');
        const ctl = UI.controls(fig);
        const S = { m: 'glm', P: 3, dim: 'heads' };
        segModels(ctl, S.m, (v) => { S.m = v; });
        UI.seg(ctl, { label: 'TP', options: [1, 2, 3, 4], value: S.P, onChange: (v) => { S.P = v; } });
        UI.seg(ctl, { label: 'Dimension', options: [{ v: 'heads', label: 'Q heads' }, { v: 'kv', label: 'KV' }, { v: 'experts', label: 'experts' }, { v: 'inter', label: 'intermediate' }, { v: 'vocab', label: 'vocab' }], value: S.dim, onChange: (v) => { S.dim = v; } });
        const H = V.fig(fig, (g, w, hh, t) => {
          const C = V.C();
          const M = PRESETS[S.m], P = S.P;
          const B = LAB.defaults(M); B.nodes = P; B.ep = M.launch.ep || (M.id === 'glm' && P === 3);
          const checks = LAB.divChecks(M, B);
          const rc = (r) => C.series[[0, 1, 3, 2][r % 4]];
          let status = null;
          const statusOf = () => {
            if (S.dim === 'heads') return checks.find((c) => c.dim === 'Q heads') || checks[0];
            if (S.dim === 'kv') return checks.find((c) => c.dim.indexOf('KV') >= 0) || checks[0];
            if (S.dim === 'experts') return checks.find((c) => c.dim.indexOf('experts') >= 0 || c.dim === 'intermediate') || checks[0];
            if (S.dim === 'vocab') return checks.find((c) => c.dim === 'vocabulary') || checks[0];
            const blk = LAB.BLOCK[M.launch.expFmt] || 1, m = M.moeInter / P, mb = M.mtpExpBlock;
            if (P === 1) return checks[0];
            const okR = m % 1 === 0 && m % blk === 0 && !(M.expPacked && M.moeInter % P);
            const okM = !mb || (m % 1 === 0 && m % mb === 0);
            const ok = okR && okM;
            const txt = mb && okR && !okM
              ? `${M.moeInter}/${P} = ${nf(m)}: a multiple of the ${FMTL[M.launch.expFmt]} block of ${blk} (routed experts), but not of the ${mb}×${mb} block of the FP8 experts in the MTP layer`
              : `${M.moeInter}/${P} = ${nf(m, m % 1 ? 1 : 0)} ${ok ? 'is' : 'is not'} an integer multiple of the ${FMTL[M.launch.expFmt]} block of ${blk}`;
            return { st: ok ? 'ok' : 'bad', txt,
              sol: M.id === 'qwen' && !ok ? 'EP is mandatory because of the FP8 MTP experts (whole 640-wide experts on each GPU); the NVFP4 routed experts would accept 320' : M.id === 'glm' && !ok ? 'GLM TP3: expert parallel, 96 experts per rank' : M.id === 'ds' && !ok ? 'the recipe does not use EP (MegaMoE requires SM100); how the experts are handled on TP3 is not documented (TP=3 chapter)' : '' };
          };
          const st0 = statusOf() || { st: 'ok', txt: '' };
          const col0 = st0.st === 'bad' ? C.red : st0.st === 'pad' ? C.d : st0.st === 'rep' ? C.b : C.a;
          const mark0 = st0.st === 'bad' ? '✕' : st0.st === 'pad' ? '⚠' : st0.st === 'rep' ? '⇄' : '✓';
          let ytop = para(g, `${M.short} · TP=${P} · ${mark0} ${st0.txt}`, 14, 20, w - 28, 12.5, 16, { weight: 700, color: col0 });
          if (st0.sol) ytop = para(g, 'in the recipe: ' + st0.sol, 14, ytop, w - 28, 11.5, 14, { color: C.ink });
          const top = ytop + 8;
          const areaH = hh - top - 132;
          const drawUnits = (n, nPad, sub) => {
            const tot = n + nPad, W = w - 28, Ha = Math.max(60, areaH - 22);
            let cols = Math.min(tot, Math.max(4, Math.ceil(Math.sqrt((tot * W) / Ha))));
            let rowsN = Math.ceil(tot / cols);
            const cs = Math.max(6, Math.min(W / cols - 2, Ha / rowsN - 2, 30));
            cols = Math.min(tot, Math.floor(W / (cs + 2))); rowsN = Math.ceil(tot / cols);
            const per = tot / P;
            const yL = top + 10 + rowsN * (cs + 2) + 16;
            let lxL = 14;
            for (let r = 0; r < P; r++) {
              const cnt = Math.round(per) - (r === P - 1 ? nPad : 0);
              const lbl = sub ? `rank ${r}: 1/${P} of each` : `rank ${r}: ${cnt}${r === P - 1 && nPad ? ' + ' + nPad + ' fake' : ''}`;
              g.font = V.font(11); const iw = g.measureText(lbl).width + 30;
              V.box(g, lxL, yL - 9, 10, 10, { r: 2, fill: V.alpha(rc(r), 0.8) }); V.text(g, lbl, lxL + 14, yL, { size: 11, color: C.ink }); lxL += iw;
            }
            for (let i = 0; i < tot; i++) {
              const r = Math.floor(i / cols), c = i % cols;
              const x = 14 + c * (cs + 2), y = top + 10 + r * (cs + 2);
              const rank = Math.min(P - 1, Math.floor(i / per));
              if (i >= n) { V.rr(g, x, y, cs, cs, 2.5); g.fillStyle = V.alpha(C.red, 0.15); g.fill(); hatch(g, x, y, cs, cs, C.red, 4, 2.5); continue; }
              if (sub) { for (let s = 0; s < P; s++) { g.fillStyle = V.alpha(rc(s), 0.75); g.fillRect(x + (s * cs) / P, y, cs / P - 0.5, cs); } }
              else { V.rr(g, x, y, cs, cs, 2.5); g.fillStyle = V.alpha(rc(rank), 0.75); g.fill(); }
            }
            return top + 10 + rowsN * (cs + 2);
          };
          let yEnd = top + 20;
          if (S.dim === 'heads') {
            const pad = M.qHeads % P ? ((M.padQ && M.padQ[P]) || Math.ceil(M.qHeads / P) * P) - M.qHeads : 0;
            yEnd = drawUnits(M.qHeads, P === 1 ? 0 : pad, false);
          } else if (S.dim === 'kv') {
            if (M.kvKind === 'mla' || M.kvKind === 'mqa') {
              for (let r = 0; r < P; r++) { const x = 14 + r * Math.min(170, (w - 28) / P), y = top + 20; V.box(g, x, y, Math.min(150, (w - 28) / P - 12), 44, { r: 8, fill: V.alpha(rc(r), 0.3), stroke: rc(r), lw: 1.5 }); V.text(g, `rank ${r}: whole latent`, x + 8, y + 18, { size: 11, color: C.ink, maxW: Math.min(150, (w - 28) / P - 12) - 12 }); V.text(g, M.id === 'ds' ? '584 B/row' : '656 B/token/layer', x + 8, y + 34, { size: 10.5, color: C.muted, font: 'mono' }); }
              yEnd = top + 80;
            } else yEnd = drawUnits(M.kvHeads, 0, M.kvHeads < P);
          } else if (S.dim === 'experts') {
            const ep = B.ep;
            V.text(g, ep ? 'expert parallel: each rank holds whole experts' : 'no EP: each expert is sliced internally (stripes = ranks)', 14, top + 4, { size: 11, color: C.muted, maxW: w - 28 });
            yEnd = drawUnits(M.E, 0, !ep);
          } else if (S.dim === 'inter') {
            const blk = M.mtpExpBlock || LAB.BLOCK[M.launch.expFmt] || 1;
            const bw = w - 28, y = top + 24, m = M.moeInter;
            V.text(g, M.mtpExpBlock ? `moe_intermediate = ${m} · NVFP4 = 16 · MTP FP8 = ${blk} (ticks)` : `moe_intermediate = ${m} · ${FMTL[M.launch.expFmt]} block = ${blk}`, 14, top + 8, { size: 11.5, font: 'mono', color: C.muted, maxW: w - 28 });
            for (let r = 0; r < P; r++) { const xa = 14 + (bw * r) / P, xb = 14 + (bw * (r + 1)) / P; V.box(g, xa + 1, y, xb - xa - 2, 34, { r: 4, fill: V.alpha(rc(r), 0.55) }); V.text(g, nf(m / P, m % P ? 1 : 0), (xa + xb) / 2, y + 22, { size: 12, align: 'center', color: C.ink, font: 'mono', weight: 600 }); }
            const nb = m / blk; if (nb <= 256) for (let i = 1; i < nb; i++) { const x = 14 + (bw * i) / nb; V.line(g, x, y + 34, x, y + 40, { color: C.muted, lw: 1 }); }
            const ok = (m / P) % blk === 0;
            for (let r = 1; r < P; r++) { const x = 14 + (bw * r) / P; V.line(g, x, y - 6, x, y + 44, { color: ok ? C.a : C.red, lw: 2.2, dash: ok ? null : [3, 3] }); }
            yEnd = y + 60;
          } else {
            const qv = LAB.lcm(64, P), vp = Math.ceil(M.vocab / qv) * qv;
            const bw = w - 28, y = top + 24;
            V.text(g, `vocab ${M.vocab.toLocaleString('en-US')} · vLLM: a multiple of 64 that divides by ${P}${qv > 64 && Math.ceil(M.vocab / 64) * 64 % P ? ` → only with an overlay: lcm(64, ${P}) = ${qv}` : ''}`, 14, top + 8, { size: 11.5, font: 'mono', color: C.muted, maxW: bw });
            for (let r = 0; r < P; r++) { const xa = 14 + (bw * r) / P, xb = 14 + (bw * (r + 1)) / P; V.box(g, xa + 1, y, xb - xa - 2, 34, { r: 4, fill: V.alpha(rc(r), 0.55) }); V.text(g, nf(vp / P) + ' rows', (xa + xb) / 2, y + 22, { size: 11.5, align: 'center', color: C.ink, font: 'mono' }); }
            if (vp > M.vocab) { const wp = Math.max(8, (bw * (vp - M.vocab)) / vp); hatch(g, 14 + bw - wp, y, wp, 34, C.red, 3, 3); V.text(g, `+${vp - M.vocab} fake`, 14 + bw, y + 50, { size: 11.5, color: C.red, align: 'right', weight: 600 }); }
            yEnd = y + 60;
          }
          void status;
          // nós e coletivas (ancorados logo abaixo das unidades, sem faixa vazia)
          const ny = Math.min(hh - 104, yEnd + 40), nw = Math.min(120, (w - 28 - (P - 1) * 30) / P);
          const xs = [];
          for (let r = 0; r < P; r++) { const x = 14 + r * (nw + 30) + (w - 28 - P * nw - (P - 1) * 30) / 2; xs.push(x); V.box(g, x, ny, nw, 40, { r: 9, fill: V.alpha(rc(r), 0.2), stroke: rc(r), lw: 1.5 }); V.text(g, `node ${r} · GB10`, x + nw / 2, ny + 17, { size: 11, align: 'center', color: C.ink, weight: 600 }); V.text(g, r === 0 ? 'head (API)' : 'worker', x + nw / 2, ny + 32, { size: 10.5, align: 'center', color: C.muted }); }
          if (P > 1) {
            for (let r = 0; r < P - 1; r++) { const xa = xs[r] + nw, xb = xs[r + 1]; V.line(g, xa, ny + 20, xb, ny + 20, { color: C.a, lw: 2.5 }); const u = (t * 1.6 + r * 0.3) % 1; V.dot(g, V.lerp(xa, xb, u), ny + 20, 3.5, C.ink); }
            if (P > 2) { g.beginPath(); g.moveTo(xs[P - 1] + nw / 2, ny + 40); g.bezierCurveTo(xs[P - 1] + nw / 2, ny + 58, xs[0] + nw / 2, ny + 58, xs[0] + nw / 2, ny + 40); g.strokeStyle = V.alpha(C.a, 0.6); g.lineWidth = 1.6; g.setLineDash([4, 4]); g.stroke(); g.setLineDash([]); }
            const ms = M.nColl * M.tColl * (P - 1) / 1000;
            para(g, `≈ ${M.nColl} collectives/step × ${M.tColl * (P - 1)} µs ≈ ${nf(ms, 1)} ms · ${nf(M.hidden * 2 / 1024, 0)} KB per all-reduce per token`, 14, ny + 76, w - 28, 11, 14, { font: 'mono', color: C.muted });
          } else V.text(g, 'TP=1: no collectives between nodes', 14, ny + 62, { size: 11, color: C.muted });
        });
        void H;
      })();

      /* ==================================================================
         FIGURA 8 — o construtor (estado compartilhado com a figura 9)
         ================================================================== */
      const fig8 = root.querySelector('#fig-construtor');
      const stage8 = fig8.querySelector('.lab-stage');
      const ctl8 = UI.controls(fig8);
      let freeF = Object.assign({}, LAB.FREE_DEF);
      let MODEL = PRESETS.glm;
      let ST = LAB.defaults(MODEL);
      let RR = LAB.compute(MODEL, ST);
      let loading = false;
      const listeners = [];
      const getModel = (id) => (id === 'free' ? LAB.buildFree(freeF) : PRESETS[id]);

      const archP = h('p', { class: 'lab-arch' });
      const strip = h('div', { class: 'lab-strip' });
      stage8.appendChild(archP); stage8.appendChild(strip);
      const RO = {
        w: UI.readout(strip, 'weights/rank', '—'), p: UI.readout(strip, 'KV pool', '—'), t: UI.readout(strip, 'tokens', '—'),
        c: UI.readout(strip, 'concurrency', '—'), d0: UI.readout(strip, 'no spec', '—'), d1: UI.readout(strip, 'with spec', '—'), ag: UI.readout(strip, 'aggregate', '—'),
      };
      const grid = h('div', { class: 'lab-grid' });
      stage8.appendChild(grid);
      const mkPanel = (title) => { const cv = h('canvas'); const p = h('div', { class: 'lab-panel' }, h('div', { class: 'lab-ph' }, title), cv); grid.appendChild(p); return cv; };
      const cvMem = mkPanel('Memory per node'), cvKV = mkPanel('KV and concurrency'), cvSpd = mkPanel('Decode speedometer (ceiling)'), cvTop = mkPanel('Topology');
      const alertsBox = h('div', { class: 'lab-alerts' });
      stage8.appendChild(alertsBox);
      // campos livres
      const freeBox = h('div', { class: 'lab-free' }, h('h5', null, 'Free fields (the architecture of your model — the initial example is illustrative)'));
      const freeGrid = h('div', { class: 'lab-free-grid' });
      freeBox.appendChild(freeGrid);
      stage8.insertBefore(freeBox, grid);
      const FFIELDS = [
        ['L', 'layers'], ['hidden', 'hidden'], ['qHeads', 'Q heads'], ['kvHeads', 'KV heads'], ['dK', 'd_k'], ['dV', 'd_v'],
        ['attn', 'full attention', ['gqa', 'mla']], ['mlaDim', 'MLA latent (d_c+rope)'], ['nFull', 'full attention layers'], ['nLin', 'linear layers'],
        ['linHeads', 'linear heads'], ['dLin', 'linear state d'], ['nSWA', 'SWA layers'], ['window', 'SWA window'],
        ['E', 'experts'], ['kE', 'top-k'], ['moeInter', 'moe_intermediate'], ['nMoE', 'MoE layers'], ['denseB', 'dense params (B)'],
        ['vocab', 'vocabulary'], ['nPredict', 'MTP layers'], ['draftGB', 'draft GB/step'],
      ];
      FFIELDS.forEach(([key, lbl, opts]) => {
        let inp;
        if (opts) { inp = h('select', { class: 'la-select' }, opts.map((o) => h('option', { value: o }, o.toUpperCase()))); inp.value = freeF[key]; }
        else inp = h('input', { class: 'la-input', type: 'number', value: freeF[key], step: key === 'denseB' || key === 'draftGB' ? '0.1' : '1', min: '0' });
        inp.addEventListener('change', () => { freeF[key] = opts ? inp.value : Math.max(0, +inp.value || 0); if (ST.m === 'free') { MODEL = getModel('free'); recompute(); } });
        freeGrid.appendChild(h('label', null, lbl, inp));
      });

      // controles
      const grp = (title) => { const d = h('div', { class: 'lab-group' }, h('b', { class: 'gt' }, title)); ctl8.appendChild(d); return d; };
      const g1 = grp('Model'), g2 = grp('Topology and memory'), g3 = grp('Serving'), g4 = grp('Speculation and hardware');
      const C8 = {};
      C8.model = UI.select(g1, { label: 'preset', options: [{ v: 'glm', label: 'GLM-5.3-Flash (recipe)' }, { v: 'qwen', label: 'Qwen3.8-Flash-Next (recipe)' }, { v: 'ds', label: 'DeepSeek-V4-Flash (recipe)' }, { v: 'mimo', label: 'MiMo-V2.6-Flash (tutorial)' }, { v: 'free', label: 'Free fields' }], value: 'glm', onChange: (v) => loadPreset(v) });
      C8.exp = UI.seg(g1, { label: 'experts', options: ['bf16', 'fp8', 'nvfp4', 'mxfp4', 'exl3'].map((v) => ({ v, label: FMTL[v] })), value: ST.expFmt, onChange: (v) => set('expFmt', v) });
      C8.bpw = UI.slider(g1, { label: 'bpw (EXL3)', min: 2, max: 8, step: 0.25, value: ST.bpw, fmt: (v) => v.toFixed(2), onInput: (v) => set('bpw', v) });
      C8.dense = UI.seg(g1, { label: 'dense', options: [{ v: 'nativo', label: 'native' }, { v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }], value: ST.denseFmt, onChange: (v) => set('denseFmt', v) });
      C8.nodes = UI.seg(g2, { label: 'nodes = TP', options: [1, 2, 3, 4], value: ST.nodes, onChange: (v) => set('nodes', v, true) });
      C8.ep = UI.toggle(g2, { label: 'expert parallel', value: ST.ep, onChange: (v) => set('ep', v) });
      C8.gmu = UI.slider(g2, { label: 'GMU', min: 0.7, max: 0.92, step: 0.005, value: ST.gmu, fmt: (v) => v.toFixed(3), onInput: (v) => set('gmu', v) });
      C8.fix = UI.toggle(g2, { label: 'fix pool', value: ST.fix, onChange: (v) => set('fix', v) });
      C8.pool = UI.slider(g2, { label: 'fixed pool', min: 2, max: 48, step: 0.5, value: ST.pool, fmt: (v) => nf(v, 1) + ' GiB', onInput: (v) => set('pool', v) });
      C8.cg = UI.toggle(g2, { label: 'graphs subtracted from pool', value: !!ST.cgEst, onChange: (v) => set('cgEst', v) });
      C8.g = UI.slider(g2, { label: 'allocator: g per token', min: 1, max: 2.5, step: 0.01, value: RR.S.g, fmt: (v) => v.toFixed(2), onInput: (v) => set('g', v) });
      C8.sfix = UI.slider(g2, { label: 'allocator: fixed/req', min: 0, max: 8, step: 0.01, value: RR.S.sFix / GiB, fmt: (v) => nf(v, 2) + ' GiB', onInput: (v) => set('sFix', v * GiB) });
      C8.maxLen = UI.slider(g3, { label: 'max-model-len', min: 0, max: CTX.length - 1, step: 1, value: nearIdx(CTX, ST.maxLen), fmt: (i) => cx(CTX[i]), onInput: (i) => set('maxLen', CTX[i]) });
      C8.ctx = UI.slider(g3, { label: 'context in use', min: 0, max: CTX.length - 1, step: 1, value: nearIdx(CTX, ST.ctx), fmt: (i) => cx(CTX[i]), onInput: (i) => set('ctx', CTX[i]) });
      C8.seqs = UI.slider(g3, { label: 'max-num-seqs', min: 1, max: 32, step: 1, value: ST.seqs, onInput: (v) => set('seqs', v) });
      C8.mnbt = UI.slider(g3, { label: 'MNBT', min: 0, max: MNBT.length - 1, step: 1, value: nearIdx(MNBT, ST.mnbt), fmt: (i) => String(MNBT[i]), onInput: (i) => set('mnbt', MNBT[i]) });
      C8.kv = UI.seg(g3, { label: 'KV', options: [{ v: 'bf16', label: 'BF16' }, { v: 'fp8', label: 'FP8' }], value: ST.kv, onChange: (v) => set('kv', v, true) });
      C8.ssm = UI.seg(g3, { label: 'linear state', options: [{ v: 'fp32', label: 'FP32' }, { v: 'bf16', label: 'BF16' }], value: ST.ssm, onChange: (v) => set('ssm', v, true) });
      C8.spec = UI.seg(g4, { label: 'method', options: ['none', 'mtp', 'dflash', 'dspark', 'ngram'].map((v) => ({ v, label: SPECL[v] })), value: ST.spec, onChange: (v) => set('spec', v, true) });
      C8.k = UI.slider(g4, { label: 'k', min: 1, max: 10, step: 1, value: ST.k, onInput: (v) => set('k', v, true) });
      C8.alpha = UI.slider(g4, { label: 'alpha', min: 0.05, max: 0.99, step: 0.01, value: ST.alpha, fmt: (v) => v.toFixed(2), onInput: (v) => set('alpha', v) });
      const profBox = h('div', { class: 'la-ctl' }, h('label', null, 'measured profiles'));
      g4.appendChild(profBox);
      C8.bw = UI.slider(g4, { label: 'effective bandwidth', min: 150, max: 273, step: 1, value: ST.bw, fmt: (v) => v + ' GB/s', onInput: (v) => set('bw', v) });

      // kv, estado, TP, método e k mudam o custo do alocador: g e o termo fixo voltam ao valor do preset
      function set(key, v, recalKV) {
        ST[key] = v;
        if (loading) return;
        if (recalKV) { ST.g = null; ST.sFix = null; }
        recompute();
      }
      function buildProfiles() {
        profBox.querySelectorAll('button').forEach((b) => b.remove());
        MODEL.profiles.forEach((p) => {
          UI.button(profBox, p.label, () => { const a = LAB.alphaEq(LAB.profSum(p), p.k); loading = true; C8.alpha.set(Math.round(a * 100) / 100); const kCh = MODEL.draft[ST.spec] && ST.k !== p.k; if (kCh) C8.k.set(p.k); loading = false; ST.alpha = a; if (kCh) { ST.g = null; ST.sFix = null; } recompute(); });
        });
      }
      function loadPreset(id) {
        MODEL = getModel(id);
        const d = LAB.defaults(MODEL);
        loading = true;
        ST = Object.assign({}, d, { m: id });
        C8.exp.set(d.expFmt); C8.bpw.set(d.bpw); C8.dense.set(d.denseFmt); C8.nodes.set(d.nodes); C8.ep.set(d.ep);
        C8.gmu.set(d.gmu); C8.fix.set(d.fix); C8.pool.set(d.pool); C8.maxLen.set(nearIdx(CTX, d.maxLen)); C8.ctx.set(nearIdx(CTX, d.ctx));
        C8.seqs.set(d.seqs); C8.mnbt.set(nearIdx(MNBT, d.mnbt)); C8.kv.set(d.kv); C8.ssm.set(d.ssm); C8.spec.set(d.spec); C8.k.set(d.k);
        C8.alpha.set(Math.round(d.alpha * 100) / 100); C8.bw.set(d.bw); C8.cg.set(!!d.cgEst);
        ST = Object.assign({}, d, { m: id });
        loading = false;
        freeBox.classList.toggle('on', id === 'free');
        buildProfiles();
        recompute();
      }
      function recompute() {
        RR = LAB.compute(MODEL, ST);
        const R = RR;
        if (ST.g == null || ST.sFix == null) {
          // valores do preset (calibrados ou estruturais): sincroniza os sliders sem perder a precisão
          const g0 = R.S.g, s0 = R.S.sFix;
          loading = true; C8.g.set(Math.round(g0 * 100) / 100); C8.sfix.set(Math.round((s0 / GiB) * 100) / 100); loading = false;
          ST.g = g0; ST.sFix = s0;
        }
        archP.innerHTML = `<b>${LA.esc(MODEL.name)}</b> — ${LA.esc(MODEL.arch)}. <span style="color:var(--text-3)">Defaults: ${LA.esc(MODEL.src)}.</span>`;
        RO.w.set(gi(R.Wrank)); RO.p.set(gi(R.poolPos)); RO.t.set(tk(R.reported));
        RO.c.set(nf(R.concMax, 2) + '×');
        RO.c.el.querySelector('span').textContent = 'conc. @' + cx(ST.maxLen);
        RO.d0.set(nf(R.s1.tps, 0) + ' tok/s'); RO.d1.set(R.s1s ? nf(R.s1s.tps, 0) + ' tok/s' : '—');
        RO.ag.set(nf((R.sBs || R.sB).agg, 0) + ' tok/s');
        RO.ag.el.querySelector('span').textContent = 'aggregate B=' + R.Bm;
        renderAlerts();
        listeners.forEach((f) => f());
      }
      function alertsOf(R) {
        const S = R.S, M = R.M, A = [];
        const add = (sev, msg, ch) => A.push({ sev, msg, ch });
        const ctx = S.ctx;
        if (R.Wrank > VIS) add('bad', `Does not fit: ${gi(R.Wrank)} of weights per rank exceed the 121.69 GiB visible on the node. Quantize the experts further or raise the TP.`, 'quantizacao-fundamentos');
        else if (!S.fix && R.poolAuto <= 0) add('bad', `Does not fit in the budget: weights (${gi(R.Wrank)}) + overhead (${gi(R.over)}) exceed GMU × 121.69 = ${gi(R.budget)}.`, 'memoria-unificada');
        if (S.fix && R.used > R.budget + 0.05 * GiB) add('warn', `With the fixed pool vLLM uses ${gi(R.used)} (${nf(R.effGmu * 100, 1)}% of the node), above GMU × total: the GMU no longer governs the KV.`, 'paged-attention-kv');
        if (R.poolPos > 0 && R.concMax < 1) add('bad', `The pool (${gi(R.poolPos)}) cannot hold one request of max-model-len = ${cx(S.maxLen)} (it needs ${gi(R.perReq(S.maxLen))}): the boot refuses. Lower max-model-len, enlarge the pool or quantize the KV.`, 'paged-attention-kv');
        if (R.slack < 0) add('bad', `Overflow: weights, overhead, pool, graphs and the ~${gi(HOST_EXT, 1)} beyond the GMU exceed 121.69 GiB (headroom ${gi(R.slack)}).`, 'memoria-unificada');
        else if (R.slack < SLACK_RED) add('risk', `Estimated idle headroom of ${gi(R.slack)} (< 8 GiB): in the recipes, this is the range where the load drove MemAvailable close to zero (Qwen at 0.835 ≈ 7.2 GiB estimated → 0.3–0.9 GiB under soak, hard-reset zone; DeepSeek at 0.835 ≈ 7.7 → 3.8–5.2 GB after boot, ~1 GB with a 2048 chunk). Run a soak and measure MemAvailable before adopting, or lower the GMU/pool.`, 'memoria-unificada');
        else if (R.slack < SLACK_GREEN) add('warn', `Estimated idle headroom of ${gi(R.slack)} (8–11 GiB): monitor MemAvailable under load (GLM ≈ 8.7 GiB estimated → ≥ 6.2 GiB free under load).`, 'memoria-unificada');
        if (S.cgEst && !S.fix && M.cgEstGiB != null && M.cgEstGiB > (M.cgRealGiB || 0) + 0.3) add('info', `CUDA graph estimate on: the profiler subtracts ${nf(M.cgEstGiB, 2)} GiB from the pool, but the graphs use ~${nf(Math.max(0, M.cgRealGiB || 0), 1)} GiB (GLM: 2.43 estimated, −0.19 measured, ~2.6 GiB of KV reserved and never used). CG_ESTIMATE=0 keeps the graphs and removes only the subtraction.`, 'cuda-graphs-warmup');
        if (!S.cgEst && R.gReal > 0.05 * GiB) add('info', `Graphs outside the budget: the ${gi(R.gReal, 2)} of CUDA graphs do not come out of the pool, they come out of the host headroom.`, 'cuda-graphs-warmup');
        if (S.mnbt !== M.launch.mnbt) add('info', `MNBT ${S.mnbt} (the recipe uses ${M.launch.mnbt}): the lab does not recompute the activation peak or the workspaces, which grow with MNBT and with the prefill chunk (on DeepSeek, a 2048 chunk cost ~1.5 GB of host RAM). Recalibrate the overhead at boot.`, 'scheduler-batching');
        if (S.gmu * VIS > LAB.FREE0) add('bad', `GMU ${S.gmu.toFixed(3)} asks for ${gi(S.gmu * VIS)}; vLLM refuses to start if the free memory at startup is smaller (the Qwen recipe measured 108.79 GiB free).`, 'memoria-unificada');
        else if (S.gmu > 0.865) add('warn', 'GMU above 0.865: only the DeepSeek big-prompt profile uses 0.87, and it requires a lean host; GLM at 0.87 lost the head during a 256k prefill.', 'memoria-unificada');
        if (S.kvForced) add('info', `${M.short}: KV ${S.kvForced.toUpperCase()} unavailable on the GB10 (${M.kvWhy}); the calculation uses FP8.`, 'kv-cache-quantizado');
        if (M.id === 'mimo' && S.kv === 'fp8') add('warn', 'MiMo: without the PR #58128 overlay, --kv-cache-dtype fp8 is silently ignored (the pool stays BF16, half the tokens).', 'kv-cache-quantizado');
        if (M.id === 'qwen' && S.kv === 'fp8') add('info', 'Qwen: FP8 KV requires patch_qsa_fp8_kv.py on the day-0 image (start.sh applies it when KV_CACHE_DTYPE starts with fp8); measured cost: −5.3% of average throughput.', 'kv-cache-quantizado');
        if (S.expFmt === 'exl3' && M.id !== 'glm') add('warn', 'EXL3 in vLLM requires its own quantization method and kernels (overlay); only the GLM recipe ships them.', 'hadamard-trelica-exl3');
        if (M.id === 'glm' && S.expFmt === 'nvfp4') add('info', 'GLM in NVFP4 (~180 GB published) fits, but the measured KLD was 0.0605 vs. 0.0246 for EXL3 4 bpw.', 'medindo-qualidade');
        if ((S.expFmt === 'bf16' || S.expFmt === 'fp8') && R.Wrank > R.budget) add('info', 'Experts at 8 or 16 bits: only ~4-bit formats fit in 2 nodes for these models.', 'fp8-fp4-microescala');
        LAB.divChecks(M, S).forEach((d) => {
          if (d.st === 'bad') add('bad', `Divisibility (${d.dim}): ${d.txt}.${d.sol ? ' In the recipe: ' + d.sol + '.' : ''}`, 'tp3-tp4');
          else if (d.st === 'pad') add('warn', `Divisibility (${d.dim}): ${d.txt}.${d.sol ? ' In the recipe: ' + d.sol + '.' : ''}`, 'tp3-tp4');
        });
        if (S.spec !== 'none' && !S.specOn) add('bad', `${M.short} has no "${SPECL[S.spec]}" drafter in this recipe; speculation was turned off in the calculation.`, 'especulacao-fundamentos');
        if (S.specOn) {
          const d = M.draft[S.spec];
          if (d.kMax && S.k > d.kMax) add('bad', `${SPECL[S.spec]}: k=${S.k} exceeds the drafter maximum (${d.kMax}${S.spec === 'dflash' ? ': trained block of ' + (d.kMax + 1) : ''}).`, 'dflash-dspark');
          if (S.spec === 'dspark' && (S.k < (M.dsparkBlock || 5) || S.k % (M.nPredict || 3))) add('bad', 'DSpark: k must be ≥ dspark_block_size (5) and a multiple of num_nextn_predict_layers (3), that is, 6 or 9.', 'dflash-dspark');
          if (S.spec === 'mtp' && M.nPredict === 1 && S.k > 1) add('info', 'Single-layer MTP reused k times: vLLM warns that acceptance tends to drop.', 'mtp-eagle-vocab');
          if (S.spec === 'mtp' && M.nPredict > 1 && S.k > M.nPredict && S.k % M.nPredict) add('bad', `MTP: with k > n_predict (${M.nPredict}), k must be a multiple of n_predict.`, 'mtp-eagle-vocab');
          if (S.mnbt < S.seqs * (S.k + 1)) add('warn', `MNBT ${S.mnbt} < max-num-seqs × (k+1) = ${S.seqs * (S.k + 1)}: verifying all sequences does not fit in one step.`, 'scheduler-batching');
          const gain = R.s1s.tps / R.s1.tps;
          if (gain < 1.2) add('warn', `With α = ${S.alpha.toFixed(2)} speculation yields only ${nf(gain, 2)}× on the ceiling: the step verifies ${R.s1s.T} tokens and reads Ē ≈ ${nf(R.s1s.Ed, 0)} experts per layer, vs. ${nf(R.s1.Ed, 0)} without speculating.`, 'especulacao-fundamentos');
        }
        if (S.maxLen > M.maxPos) add('warn', `max-model-len above the native context (${cx(M.maxPos)}): requires extension (YaRN) and VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.${M.id === 'qwen' ? ' In the Qwen recipe, YaRN was a silent no-op until Sep 2026: 1M not validated.' : ''}`, 'posicao-rope');
        if (M.id === 'glm' && S.mnbt >= 8192) add('bad', 'GLM: never MNBT 8192 — chunk × long history overflows the persistent_topk smem on the GB10 (crash at 300k).', 'scheduler-batching');
        if (R.poolPos > 0 && R.concMax >= 1) {
          const c = Math.floor(R.concCtx);
          if (c < S.seqs) add('info', `With ${cx(ctx)} tokens per request, only ${c} of the ${S.seqs} slots fit in the pool; the rest will be preempted and redone (vLLM does not OOM).`, 'paged-attention-kv');
          else add('ok', `Concurrency is limited by max-num-seqs (${S.seqs}), not by the pool (${nf(R.concCtx, 1)}× at ${cx(ctx)}).`, 'scheduler-batching');
        }
        if (!A.some((a) => a.sev === 'bad')) add('ok', 'No blockers under the lab rules. Next step: boot, read the log and calibrate (Figure 47.10).', 'benchmarking');
        return A;
      }
      function renderAlerts() {
        const C = V.C();
        alertsBox.innerHTML = '';
        const order = { bad: 0, risk: 0.5, warn: 1, info: 2, ok: 3 };
        alertsOf(RR).sort((a, b) => order[a.sev] - order[b.sev]).forEach((a) => {
          const title = LA.byId[a.ch] ? LA.byId[a.ch].navTitle || LA.byId[a.ch].title : NAVT[a.ch] || a.ch;
          alertsBox.appendChild(h('div', { class: 'lab-alert' }, h('i', { style: { '--c': sevCol(C, a.sev) } }), h('span', null, a.msg), h('a', { href: '#/c/' + a.ch }, '→ ' + title)));
        });
      }

      // painel: memória por nó
      const dm = { w: 0, o: 0, p: 0 };
      V.loop(cvMem, (g, w, hh, t, dt) => {
        const C = V.C(), R = RR, P = R.S.P;
        dm.w = ease(dm.w, Math.min(R.Wrank, VIS * 1.2), dt); dm.o = ease(dm.o, R.over, dt); dm.p = ease(dm.p, R.poolPos, dt);
        const y0 = 16, y1 = hh - 26, Hh = y1 - y0, scale = Math.max(VIS, R.Wrank + R.over) * 1.02;
        const Y = (b) => y1 - (b / scale) * Hh;
        const legW = w < 420 ? 0 : 150;
        const areaW = w - 20 - legW, bw = Math.min(46, (areaW - 10) / P - 10);
        for (let r = 0; r < P; r++) {
          const x = 14 + r * (bw + 12);
          V.box(g, x, Y(VIS), bw, y1 - Y(VIS), { r: 5, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.35), lw: 1 });
          let acc = 0;
          [[dm.w, C.b, 0.6], [dm.o + R.gReal, C.d, 0.6], [dm.p, C.a, 0.35]].forEach(([v, col, al]) => { const ya = Y(acc), yb = Y(acc + v); V.box(g, x + 1, yb, bw - 2, ya - yb, { r: 2, fill: V.alpha(col, al) }); acc += v; });
          const ex = Math.min(HOST_EXT, Math.max(0, VIS - acc)); hatch(g, x + 1, Y(acc + ex), bw - 2, Y(acc) - Y(acc + ex), V.alpha(C.muted, 0.6), 6, 2); acc += ex;
          if (R.Wrank > VIS || R.used + HOST_EXT > VIS) { V.box(g, x - 2, Y(VIS) - 6, bw + 4, 6, { r: 2, fill: C.red }); }
          V.text(g, r === 0 ? 'head' : 'wk' + r, x + bw / 2, hh - 10, { size: 10.5, color: C.muted, align: 'center' });
        }
        const yG = Y(R.budget); V.line(g, 10, yG, 14 + P * (bw + 12), yG, { color: C.ink, lw: 1.2, dash: [4, 3] });
        V.text(g, 'GMU', 14 + P * (bw + 12) + 2, yG + 4, { size: 10.5, color: C.ink, font: 'mono' });
        V.text(g, '121.69', 14 + P * (bw + 12) + 2, Y(VIS) + 4, { size: 10.5, color: C.muted, font: 'mono' });
        const lx = legW ? w - legW : 14 + P * (bw + 12) + 44;
        const items = [[C.b, 'weights', gi(R.Wrank)], [C.d, 'overhead' + (R.gReal > 0.05 * GiB ? ' + graphs' : ''), gi(R.over + R.gReal)], [C.a, 'KV' + (R.S.fix ? ' (fixed)' : ''), gi(R.poolPos)], [C.muted, 'beyond the GMU', gi(HOST_EXT, 1)], [slackCol(C, R.slack), 'headroom (idle)', gi(R.slack)]];
        items.forEach(([c, l, v], i) => { const y = 24 + i * 36; if (lx > w - 60) return; V.box(g, lx, y - 9, 10, 10, { r: 2, fill: V.alpha(c, 0.75) }); V.text(g, l, lx + 15, y, { size: 11.5, color: C.ink, weight: 600 }); V.text(g, v, lx + 15, y + 14, { size: 11, color: C.muted, font: 'mono' }); });
        if (R.Wrank > VIS) V.text(g, 'DOES NOT FIT', 14 + (P * (bw + 12)) / 2, Y(VIS) - 12, { size: 12, color: C.red, weight: 700, align: 'center' });
      });
      // painel: KV
      V.loop(cvKV, (g, w, hh, t) => {
        const C = V.C(), R = RR, S = R.S;
        V.text(g, R.poolPos > 0 ? '≈ ' + tk(R.reported) + ' tokens' : 'no pool', 14, 38, { size: 26, font: 'display', weight: 700, color: R.poolPos > 0 ? C.a : C.red });
        let yk = para(g, `"GPU KV cache size" = C(${cx(S.maxLen)}) × ${cx(S.maxLen)} · estimate`, 14, 56, w - 28, 11, 13, { color: C.muted, font: 'mono' });
        yk = para(g, `Maximum concurrency for ${cx(S.maxLen)} tokens ≈ ${nf(R.concMax, 2)}×`, 14, yk + 6, w - 28, 12, 15, { color: R.concMax < 1 ? C.red : C.ink, font: 'mono', weight: 600 });
        yk = para(g, `per request: ${nf(R.bTok)} B × g ${nf(S.g, 2)} × L + ${R.sFix >= 1e8 ? gi(R.sFix, 2) : mb(R.sFix, 1)} fixed (${R.sFixSrc})`, 14, yk, w - 28, 11, 13, { color: C.muted, font: 'mono' });
        // vagas
        const n = S.seqs, fit = Math.min(n, Math.floor(R.concCtx));
        yk = para(g, `slots (max-num-seqs ${n}) with ${cx(S.ctx)} tokens each:`, 14, yk + 10, w - 28, 11.5, 14, { color: C.ink });
        const perRow = Math.min(n, 16), cs = Math.max(8, Math.min(22, (w - 28) / perRow - 4));
        for (let i = 0; i < n; i++) {
          const r = Math.floor(i / 16), c = i % 16, x = 14 + c * (cs + 4), y = yk - 4 + r * (cs + 4);
          const ok = i < fit;
          V.rr(g, x, y, cs, cs, 4); g.fillStyle = ok ? V.alpha(C.a, 0.3 + 0.2 * Math.sin(t * 2 + i)) : V.alpha(C.red, 0.35); g.fill();
          g.strokeStyle = ok ? C.a : C.red; g.lineWidth = 1; g.stroke();
          if (!ok && cs > 12) V.text(g, '✕', x + cs / 2, y + cs / 2 + 4, { size: 11, color: C.red, align: 'center' });
        }
        const lim = R.concCtx >= n ? 'limit: max-num-seqs (scheduler)' : `limit: KV pool (${nf(R.concCtx, 2)}× at ${cx(S.ctx)})`;
        V.text(g, lim, 14, hh - 16, { size: 12, color: R.concCtx >= n ? C.a : C.d, weight: 600, maxW: w - 28 });
      });
      // painel: velocímetro
      const gz = { a: 0, b: 0 };
      V.loop(cvSpd, (g, w, hh, t, dt) => {
        const C = V.C(), R = RR;
        const va = R.s1.tps, vb = R.s1s ? R.s1s.tps : 0;
        const top = Math.max(va, vb);
        const scale = [50, 100, 150, 200, 300, 400, 600].find((s) => s >= top * 1.15) || 800;
        gz.a = ease(gz.a, va, dt, 4); gz.b = ease(gz.b, vb, dt, 4);
        const cxp = w / 2, cy = hh - 66, Rr = Math.max(40, Math.min(w * 0.33, hh - 122));
        g.lineCap = 'round';
        g.lineWidth = 13; g.strokeStyle = V.alpha(C.muted, 0.16); g.beginPath(); g.arc(cxp, cy, Rr, Math.PI, 2 * Math.PI); g.stroke();
        const grad = g.createLinearGradient(cxp - Rr, 0, cxp + Rr, 0); grad.addColorStop(0, C.b); grad.addColorStop(1, C.c);
        g.strokeStyle = grad; g.beginPath(); g.arc(cxp, cy, Rr, Math.PI, Math.PI + Math.PI * Math.min(1, gz.b / scale || gz.a / scale)); g.stroke();
        g.lineCap = 'butt';
        for (let i = 0; i <= 4; i++) { const an = Math.PI + (Math.PI * i) / 4; V.text(g, nf((scale * i) / 4, (scale * i) % 4 ? 1 : 0), cxp + Math.cos(an) * (Rr + 18), cy + Math.sin(an) * (Rr + 18) + 4, { size: 10.5, color: C.muted, align: 'center', font: 'mono' }); }
        const needle = (v, col, lw, len) => { const an = Math.PI + Math.PI * Math.min(1, v / scale); V.line(g, cxp, cy, cxp + Math.cos(an) * Rr * len, cy + Math.sin(an) * Rr * len, { color: col, lw }); };
        needle(gz.a, C.muted, 3, 0.8); if (R.s1s) needle(gz.b, C.c, 3.5, 0.95);
        V.dot(g, cxp, cy, 6, C.ink);
        V.text(g, nf(R.s1s ? vb : va, 0), cxp, cy + 32, { size: 28, font: 'display', weight: 700, color: R.s1s ? C.c : C.a, align: 'center' });
        V.text(g, R.s1s ? `tok/s per stream with ${SPECL[R.S.spec]} k=${R.S.k} (τ=${nf(R.s1s.tau, 2)})` : 'tok/s per stream (no speculation)', cxp, cy + 50, { size: 11, color: C.muted, align: 'center', maxW: w - 20 });
        V.text(g, `gray: no spec ${nf(va, 0)} · aggregate B=${R.Bm}: ${nf((R.sBs || R.sB).agg, 0)}`, cxp, 16, { size: 11, color: C.muted, align: 'center', maxW: w - 20 });
      });
      // painel: topologia
      V.loop(cvTop, (g, w, hh, t) => {
        const C = V.C(), R = RR, S = R.S, M = R.M, P = S.P;
        const checks = LAB.divChecks(M, S);
        const nw = Math.min(96, (w - 28 - (P - 1) * 22) / P), ny = 18;
        const xs = []; const rc = (r) => C.series[[0, 1, 3, 2][r % 4]];
        const off = (w - 28 - P * nw - (P - 1) * 22) / 2;
        for (let r = 0; r < P; r++) {
          const x = 14 + off + r * (nw + 22); xs.push(x);
          V.box(g, x, ny, nw, 64, { r: 9, fill: V.alpha(rc(r), 0.18), stroke: rc(r), lw: 1.5 });
          V.text(g, r === 0 ? 'head · r0' : 'worker · r' + r, x + nw / 2, ny + 15, { size: 10.5, color: C.ink, align: 'center', weight: 600, maxW: nw - 6 });
          const fr = Math.min(1, R.Wrank / VIS);
          V.box(g, x + 8, ny + 24, nw - 16, 7, { r: 3, fill: V.alpha(C.muted, 0.2) }); V.box(g, x + 8, ny + 24, (nw - 16) * fr, 7, { r: 3, fill: fr >= 1 ? C.red : C.b });
          V.text(g, gi(R.Wrank, 0), x + nw / 2, ny + 44, { size: 10.5, color: C.muted, align: 'center', font: 'mono' });
          V.text(g, S.ep ? `${Math.floor(M.E / P)} exp.` : `Q ${Math.ceil(M.qHeads / P)}/rank`, x + nw / 2, ny + 58, { size: 10.5, color: C.muted, align: 'center', font: 'mono', maxW: nw - 6 });
        }
        if (P > 1) {
          for (let r = 0; r < P - 1; r++) { const xa = xs[r] + nw, xb = xs[r + 1]; V.line(g, xa, ny + 32, xb, ny + 32, { color: C.a, lw: 2.4 }); for (let k = 0; k < 2; k++) { const u = (t * 1.3 + k * 0.5 + r * 0.2) % 1; V.dot(g, V.lerp(xa, xb, u), ny + 32, 3, C.ink); } }
          if (P > 2) { g.beginPath(); g.moveTo(xs[P - 1] + nw / 2, ny + 64); g.bezierCurveTo(xs[P - 1] + nw / 2, ny + 84, xs[0] + nw / 2, ny + 84, xs[0] + nw / 2, ny + 64); g.strokeStyle = V.alpha(C.a, 0.6); g.lineWidth = 1.5; g.setLineDash([4, 4]); g.stroke(); g.setLineDash([]); }
        }
        V.text(g, P > 1 ? `≈ ${M.nColl} collectives/step → ${nf(M.nColl * M.tColl * (P - 1) / 1000, 1)} ms · RoCE CX7` : 'TP=1: no collectives', 14, ny + 102, { size: 11, color: C.muted, font: 'mono', maxW: w - 28 });
        let y = ny + 122; const colW = w < 420 ? w - 28 : (w - 28) / 2;
        checks.slice(0, 8).forEach((c, i) => {
          const col = c.st === 'bad' ? C.red : c.st === 'pad' ? C.d : c.st === 'rep' ? C.b : C.a;
          const mark = c.st === 'bad' ? '✕' : c.st === 'pad' ? '⚠' : c.st === 'rep' ? '⇄' : '✓';
          const x = 14 + (colW < w - 28 ? (i % 2) * colW : 0), yy = y + (colW < w - 28 ? Math.floor(i / 2) : i) * 17;
          if (yy > hh - 6) return;
          V.text(g, `${mark} ${c.dim}`, x, yy, { size: 11, color: col, weight: 600, maxW: colW - 6 });
        });
      });

      loadPreset('glm');

      /* ==================================================================
         FIGURA 9 — esqueleto .env + vllm serve
         ================================================================== */
      (function fig9() {
        const fig = root.querySelector('#fig-esqueleto');
        const ctl = UI.controls(fig);
        const code = fig.querySelector('#lab-code code');
        const S = { rank: 0, cm: true };
        UI.seg(ctl, { label: 'rank', options: [{ v: 0, label: 'head (0)' }, { v: 1, label: 'worker (1)' }], value: 0, onChange: (v) => { S.rank = v; render(); } });
        UI.toggle(ctl, { label: 'comments', value: true, onChange: (v) => { S.cm = v; render(); } });
        const rM = UI.readout(ctl, 'preset', '—');
        function render() {
          const txt = genCode(RR, S.rank, S.cm);
          code.textContent = txt;
          rM.set(RR.M.short);
          if (window.hljs) { try { code.removeAttribute('data-highlighted'); code.className = 'language-bash'; window.hljs.highlightElement(code); } catch (e) { /* sem realce */ } }
        }
        listeners.push(render);
        render();
      })();

      function capList(S) {
        const k = S.specOn ? S.k : 0, set = new Set([1, 2, 4]);
        if (k > 0) for (let n = 1; n <= S.seqs; n++) set.add(n * (k + 1));
        else for (let n = 8; n <= S.seqs; n *= 2) set.add(n);
        if (!k) set.add(S.seqs);
        return Array.from(set).filter((x) => x <= Math.max(4, S.seqs * (k + 1))).sort((a, b) => a - b);
      }
      function genCode(R, rank, cm) {
        const S = R.S, M = R.M, P = S.P, k = S.specOn ? S.k : 0, id = M.id;
        const out = [];
        const row = (code, com) => { if (!cm || !com) { out.push(code); return; } const pad = Math.max(1, 44 - code.length); out.push(code + ' '.repeat(pad) + '# ' + com); };
        const blank = () => out.push('');
        const poolB = Math.round(S.pool * GiB);
        row('# ===== .env — skeleton generated by the Lab (check every value) =====');
        row('# model: ' + M.name);
        row('# defaults: ' + M.src);
        row(`# estimate: weights/rank ≈ ${nf(R.Wrank / GiB, 1)} GiB · pool ≈ ${nf(R.poolPos / GiB, 1)} GiB · ${nf(R.concMax, 2)}× at ${cx(S.maxLen)} (log ≈ ${tk(R.reported)} tokens)`);
        row(`# estimate: each request = ${nf(R.bTok)} B/token × g ${nf(S.g, 2)} × L + ${nf(R.sFix / GiB, 2)} GiB fixed (${R.sFixSrc})`);
        row(`# estimate: idle host headroom ≈ ${nf(R.slack / GiB, 1)} GiB (${R.slack < LAB.SLACK_RED ? 'red: measure under load' : R.slack < LAB.SLACK_GREEN ? 'amber: monitor' : 'green'})`);
        row(`# estimate: decode ceiling C=1 ≈ ${nf(R.s1.tps, 0)} tok/s without spec${R.s1s ? ', ≈ ' + nf(R.s1s.tps, 0) + ' with ' + SPECL[S.spec] + ' k=' + k + ' (α=' + nf(S.alpha, 2) + ')' : ''}`);
        blank();
        row('# --- cluster ---');
        row('HEAD_IP=10.0.0.1', 'CX7 fabric IP of the head (rank 0)');
        row('WORKER_IP=10.0.0.2', 'fabric IP of the worker (rank 1)');
        row('IFACE=enp1s0f0np0', 'CX7 interface (find it with: ip -br a)');
        row('IB_HCA==rocep1s0f0', 'leading "=" = exact HCA match');
        row('IB_GID_INDEX=3', 'RoCEv2 GID (check gids/ and gid_attrs/types)');
        blank();
        row('# --- model ---');
        row('MODEL_ID=' + M.ckpt, id === 'free' ? 'TODO: your model repository' : 'recipe checkpoint');
        if (M.rev) row('MODEL_REV=' + M.rev, 'pinned revision (the Hub main branch moves)');
        row('MODEL_DIR=$HOME/models/' + (M.ckpt.split('/')[1] || 'model'), 'same path on both nodes (rsync or NFS ro)');
        row('IMAGE=' + M.image, id === 'free' ? 'TODO: pin by @sha256 digest' : 'recipe image (pin the digest)');
        if (id === 'glm' && S.specOn && S.spec === 'dflash') { row('DFLASH_MODEL=incoai/GLM-5.3-Flash-DFlash2', 'drafter (CC BY-NC-ND 4.0)'); row('DFLASH_REVISION=dc77ff1c99eeb2df044ee3d4f0094eb033fee410', 'drafter pin'); row('DFLASH_MODEL_DIR=$HOME/models/GLM-5.3-Flash-DFlash2', 'local drafter snapshot (on both nodes)'); }
        blank();
        row('# --- serving ---');
        row('TP=' + P, '1 GPU per node → TP = number of nodes');
        row('MAX_MODEL_LEN=' + S.maxLen, 'cap per request (does not reserve KV)');
        row('MAX_NUM_SEQS=' + S.seqs, 'simultaneous slots in the batch');
        row('MAX_NUM_BATCHED_TOKENS=' + S.mnbt, 'tokens per step / prefill chunk');
        row('GMU=' + S.gmu.toFixed(3), `budget = GMU × 121.69 = ${nf(R.budget / GiB, 1)} GiB`);
        row('KV_CACHE_MEMORY_BYTES=' + (S.fix ? poolB : ''), S.fix ? `fixed pool of ${nf(S.pool, 1)} GiB (ignores the GMU for KV)` : 'empty = pool from the GMU; fix it after the first boot');
        row('KV_CACHE_DTYPE=' + (id === 'ds' ? 'nvfp4_ds_mla' : S.kv === 'fp8' ? 'fp8' : 'auto'), `physical: ${nf(R.bTok)} B/token/rank (+ fixed per request)`);
        if (id === 'qwen' || id === 'free') row('MAMBA_SSM_CACHE_DTYPE=' + (S.ssm === 'bf16' ? 'bfloat16' : ''), S.ssm === 'bf16' ? 'GDN state in BF16: half the page' : 'empty = FP32 from the checkpoint');
        row('SPEC_METHOD=' + (S.specOn ? S.spec : 'none'), S.specOn ? 'rollback: none' : 'mandatory baseline');
        row('SPEC_K=' + (S.specOn ? k : 0), S.specOn ? `τ ≈ ${nf(R.s1s.tau, 2)} tokens/step (estimate)` : '');
        if (id === 'glm') {
          if (P === 3) row('GLM53_DENSE_FP8=' + (S.denseFmt === 'fp8' ? 'dense,kda' : ''), 'TP3: dense,kda (KDA f_b/g_b stay BF16)');
          else row('GLM53_DENSE_FP8=' + (S.denseFmt === 'fp8' ? 'all' : ''), 'weight-only FP8 dense at load time (Marlin)');
          row('GLM53_KDA_BF16_LARGE_M=' + (S.denseFmt === 'fp8' && P !== 3 ? '1' : '0'), P === 3 ? 'TP3: the recipe sets it to 0 (no BF16 copy)' : 'BF16 copy of KDA: +3.3 GiB/rank, prefill −12%');
          row('EXL3_FUSED_MOE=1', 'one fused exl3_moe per layer (graph-safe)');
          row('EXL3_FAT_KERNEL=1', 'E2: direct kernels for "fat" experts (prefill +20–21%)');
          row('EXL3_FAT_GROUPED=1', 'E3: prefill +37–45%; 560 MiB of scratch comes out of KV');
          row('EXL3_TEMP_ROWS_FUSED=' + Math.max(S.seqs * (k + 1), 8), '≥ MAX_NUM_SEQS × (k+1)');
          row('CG_ESTIMATE=' + (S.cgEst ? '1' : '0'), S.cgEst ? 'estimates 2.43 GiB of graphs (real: −0.19)' : 'graphs on, no subtraction from the pool');
          row('GLM53_INDEXER_WORKSPACE=rightsize', 'indexer workspace ∝ seqs, not ∝ context');
          if (S.specOn && S.spec === 'dflash') row('GLM53_DRAFT_KV_COMPACT=1', 'compact draft pages');
          row('GLM53_MIXED_PREFILL_CHUNK=fair', 'fair prefill × decode scheduler (v5)');
        }
        if (id === 'mimo') { row('MARLIN_INPUT_DTYPE=fp8', 'W4A8 in Marlin (empty = W4A16)'); row('EXACT_QKV=1', 'QKV without requantizing (#59053)'); }
        if (id === 'qwen' && S.specOn && S.spec === 'mtp') row('MTP_DRAFT_VOCAB=files/draft_vocab_en_code_47k.txt', 'draft lm_head 0.59 → 0.22 GiB/GPU');
        if (id === 'qwen') row(`# graphs: ${S.cgEst ? 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 (subtracts from the pool)' : 'the 0.835 boot did not subtract them: 0.54 GiB come out of the headroom'}`);
        if (id === 'ds') row(`# graphs: VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${S.cgEst ? '1' : '0 (the recipe: outside the budget)'}`);
        blank();
        row(`# ===== vllm serve (rank ${rank}${rank ? ', worker: comes up FIRST' : ', head: comes up after the worker'}) =====`);
        out.push('VLLM_ARGS=(serve "$MODEL_DIR"');
        const A = (flag, com) => row('  ' + flag, com);
        A(`--served-model-name ${M.served}`, 'OpenAI API "model" id');
        if (M.rev) A('--revision "$MODEL_REV"', 'same commit on both nodes');
        if (id === 'ds' || id === 'mimo' || id === 'free') A('--trust-remote-code', id === 'free' ? 'TODO: only if the config has auto_map' : 'config/processor with checkpoint code');
        A(`--tensor-parallel-size ${P} --nnodes ${P} --node-rank ${rank}`, `TP=${P} spanning ${P} nodes`);
        A(`--master-addr "$HEAD_IP" --master-port ${M.mport}`, 'rendezvous (fabric IP, not the management one)');
        A('--distributed-executor-backend mp', 'multiprocessing, no Ray (all 3 recipes)');
        if (id === 'ds') A('--pipeline-parallel-size 1', 'no PP: the whole model on every rank, sliced by TP');
        if (S.ep) A(id === 'qwen' ? '--enable-expert-parallel --all2all-backend allgather_reducescatter' : '--enable-expert-parallel', `whole experts per rank (${Math.floor(M.E / P)}/rank)`);
        if (id === 'glm') A('--quantization exl3', 'overlay method (never --moe-backend marlin)');
        if (id === 'glm' && P === 3) A('--hf-overrides \'{"num_attention_heads":66,"num_key_value_heads":66,"linear_num_heads":66}\'', 'TP3: 64 → 66 heads (+ text_config; tp3 overlay)');
        A(`--max-model-len "\${MAX_MODEL_LEN:-${S.maxLen}}"`, 'cap; the real constraint is Σ live tokens ≤ pool');
        A(`--max-num-seqs "\${MAX_NUM_SEQS:-${S.seqs}}"`, `estimate: ${nf(R.concCtx, 1)}× fit at ${cx(S.ctx)}`);
        A(`--max-num-batched-tokens "\${MAX_NUM_BATCHED_TOKENS:-${S.mnbt}}"`, S.specOn ? `≥ seqs × (k+1) = ${S.seqs * (k + 1)}` : 'prefill chunk vs ITL');
        if (id === 'ds') A('--long-prefill-token-threshold 1024', 'long prefill chunks yield budget to decode');
        A(`--gpu-memory-utilization "\${GMU:-${S.gmu.toFixed(3)}}"`, `each 0.01 ≈ 1.2 GiB of host headroom`);
        if (S.fix) A(`${id === 'glm' ? '--kv-cache-memory-bytes' : '--kv-cache-memory'} "\${KV_CACHE_MEMORY_BYTES:-${poolB}}"`, 'flag name varies by build: --help=all');
        if (id === 'ds') { A('--kv-cache-dtype nvfp4_ds_mla', '584 B/row (= fp8_ds_mla)'); A('--block-size 256', 'MLA C4: 256 tokens / 4 = 64 rows per page'); }
        else A(`--kv-cache-dtype "\${KV_CACHE_DTYPE:-${S.kv === 'fp8' ? 'fp8' : 'auto'}}"`, id === 'glm' ? 'becomes fp8_ds_mla (656 B/token/layer)' : id === 'qwen' && S.kv === 'fp8' ? 'requires patch_qsa_fp8_kv (start.sh)' : id === 'mimo' && S.kv === 'fp8' ? 'requires overlay #58128 (otherwise ignored)' : '');
        if (id === 'qwen' && S.ssm === 'bf16') A('--mamba-ssm-cache-dtype bfloat16', 'state of the 36 GDN layers in BF16');
        if (id === 'free' && S.ssm === 'bf16' && M.linHeads) A('--mamba-ssm-cache-dtype bfloat16', 'linear state in BF16 (check the kernel)');
        if (id === 'glm') A('--load-format instanttensor', '164 GiB in ~36 s (O_DIRECT)');
        if (id === 'qwen' || id === 'mimo') A('--load-format safetensors --safetensors-load-strategy lazy', 'shards on demand');
        A('--enable-prefix-caching', 'agents resend the whole history');
        if (id !== 'glm') A('--enable-chunked-prefill', 'prefill in slices interleaved with decode');
        if (id !== 'glm') A('--enable-prompt-tokens-details', 'exposes cached_tokens (debugs the prefix cache)');
        if (id === 'glm') A('--no-enable-flashinfer-autotune', 'autotune takes down rank 0 on SM121');
        if (id === 'ds') { A('--enable-flashinfer-autotune', 'cache persisted in vllm-cache/'); A('--async-scheduling', 'CPU prepares step N+1 during step N'); A('--moe-backend flashinfer_b12x', 'W4A16 MXFP4 (B12X)'); }
        if (id === 'mimo') { A('--moe-backend marlin', 'MXFP4 on SM12x (W4A8 with MARLIN_INPUT_DTYPE)'); A('--no-async-scheduling', 'spec + async: bad output (#46669)'); }
        // speculation
        if (S.specOn) {
          let js = '';
          if (S.spec === 'dflash' && id === 'glm') js = `"{\\"method\\":\\"dflash\\",\\"model\\":\\"$DFLASH_MODEL_DIR\\",\\"num_speculative_tokens\\":${k},\\"kv_cache_dtype\\":\\"auto\\",\\"draft_sample_method\\":\\"probabilistic\\",\\"rejection_sample_method\\":\\"standard\\",\\"draft_tensor_parallel_size\\":${P === 3 ? 1 : P}}"`;
          else if (S.spec === 'dflash' && id === 'mimo') js = `"{\\"method\\":\\"dflash\\",\\"model\\":\\"$MODEL_DIR/dflash\\",\\"num_speculative_tokens\\":${k},\\"draft_tensor_parallel_size\\":${P}}"`;
          else if (S.spec === 'mtp' && id === 'qwen') js = `'{"method":"mtp","num_speculative_tokens":${k},"use_local_argmax_reduction":true,"disable_eagle_block_drop":true,"index_share_for_mtp_iteration":true}'`;
          else if (S.spec === 'dspark') js = `'{"method":"dspark","num_speculative_tokens":${k},"draft_sample_method":"probabilistic"}'`;
          else if (S.spec === 'ngram') js = `'{"method":"ngram","num_speculative_tokens":${k}}'`;
          else js = `'{"method":"${S.spec}","num_speculative_tokens":${k}}'`;
          A('--speculative-config ' + js, `k=${k}; τ ≈ ${nf(R.s1s.tau, 2)} (estimate with α=${nf(S.alpha, 2)})`);
          if (id === 'qwen' && cm) row('  # use_local_argmax_reduction only applies with MTP_DRAFT_VOCAB (reduced lm_head; mounted in docker run)');
        }
        // graphs
        if (id === 'qwen') A('--compilation-config \'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}\'', 'graphs only in decode; automatic capture');
        else if (id === 'ds') A(`--max-cudagraph-capture-size ${Math.ceil((S.seqs * (k + 1)) / 8) * 8}`, `seqs × (k+1) = ${S.seqs * (k + 1)} rounded up to a multiple of 8`);
        else {
          if (id === 'mimo') A('--compilation-config \'{"cudagraph_mode":"FULL_DECODE_ONLY"}\'', 'the DiffKV builder requires this mode');
          A('--cudagraph-capture-sizes ' + capList(S).join(' '), k ? `multiples of k+1 = ${k + 1} up to seqs × (k+1)` : 'possible decode batches');
        }
        // parsers and multimodal
        if (id === 'glm') { A('--tool-call-parser glm47 --enable-auto-tool-choice', 'GLM XML tool calls'); A('--reasoning-parser glm45', 'separates <think>'); A('--chat-template /opt/glm53/chat_template.jinja', 'the checkpoint jinja is text-only'); A('--limit-mm-per-prompt \'{"image":48,"video":1}\' --mm-processor-kwargs \'{"max_image_tokens":2048}\'', 'nothing is reserved for vision: cap the worst case'); A('--mm-processor-cache-gb 1 --skip-mm-profiling', 'multimodal profiling OOMs on UMA'); }
        if (id === 'qwen') { A('--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder', 'reasoning in delta["reasoning"]'); A('--mm-encoder-tp-mode data', 'replicated vision (4304/2 does not divide by 16)'); A(`--hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"${S.maxLen > 262144 ? `,"rope_parameters":{"rope_type":"yarn","factor":${Math.ceil(S.maxLen / 262144).toFixed(1)},"original_max_position_embeddings":262144}` : ''}}}'`, S.maxLen > 262144 ? 'PLE FP8 + YaRN nested in text_config' : 'N-gram table in FP8'); }
        if (id === 'ds') { A('--tokenizer-mode deepseek_v4', 'Python encoder from the checkpoint (no jinja)'); A('--limit-mm-per-prompt \'{"image":8}\'', 'Vision-Exp vision'); A('--tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4', 'DSML → tool_calls; separates <think>'); A('--reasoning-config \'{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}\'', 'markers for the thinking budget'); A('--generation-config vllm', 'the client controls sampling'); }
        if (id === 'mimo') { A('--reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice', 'compact XML → tool_calls'); A('--generation-config auto', 'T=1.0 / top_p 0.95 avoids "tool-call storms"'); A('--default-chat-template-kwargs \'{"enable_thinking": false}\'', 'thinking is on by default in the template'); A('--mm-encoder-tp-mode data --mm-processor-cache-gb 1', 'replicated encoders; 1 GiB cache'); A('--limit-mm-per-prompt \'{"image":4,"video":1,"audio":1}\' --mm-processor-kwargs \'{"max_pixels":2073600}\'', 'caps the multimodal worst case'); }
        if (id === 'free') { row('  # TODO --tool-call-parser <family> --reasoning-parser <family>   (vLLM release notes)'); row('  # TODO --moe-backend <kernel>   (check support for sm_121 and for the expert format)'); row('  # TODO --quantization <method>  (if the checkpoint does not declare quant_method)'); }
        if (rank === 0) A('--host 0.0.0.0 --port 8888', 'only the head opens the OpenAI API');
        else A('--headless', 'worker without API: only runs the model');
        out.push(')');
        blank();
        row(`# ===== docker run (rank ${rank}) =====`);
        const envs = [];
        if (id === 'glm' || id === 'qwen' || id === 'ds' || id === 'mimo') envs.push('VLLM_ALLOW_LONG_MAX_MODEL_LEN=1');
        if (id === 'glm') { envs.push('GLM53_DENSE_FP8', 'GLM53_KDA_BF16_LARGE_M', 'EXL3_FUSED_MOE', 'EXL3_FAT_KERNEL', 'EXL3_FAT_GROUPED', 'EXL3_TEMP_ROWS_FUSED', 'GLM53_INDEXER_WORKSPACE', 'GLM53_MIXED_PREFILL_CHUNK', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE', 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'); if (S.specOn && S.spec === 'dflash') envs.push('GLM53_DRAFT_KV_COMPACT'); }
        if (id === 'ds') envs.push('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=' + (S.cgEst ? '1' : '0'), 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800');
        if (id === 'qwen' && S.cgEst) envs.push('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1');
        if (id === 'qwen' && S.specOn && S.spec === 'mtp') envs.push('VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt');
        if (id === 'mimo') envs.push('VLLM_USE_DEEP_GEMM=0', 'VLLM_MARLIN_INPUT_DTYPE=$MARLIN_INPUT_DTYPE', 'VLLM_ENGINE_READY_TIMEOUT_S=3600');
        out.push('docker run -d --name vllm-lab --gpus all --network host --ipc host --shm-size 16g \\');
        out.push('  --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband \\');
        out.push('  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \\');
        out.push('  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" \\');
        out.push(`  -e VLLM_HOST_IP="${rank ? '$WORKER_IP' : '$HEAD_IP'}" -e HF_HUB_OFFLINE=1 \\`);
        envs.forEach((e) => out.push(`  -e ${e.indexOf('=') > 0 ? e : e + '="$' + e + '"'} \\`));
        if (id === 'glm' && S.specOn && S.spec === 'dflash') out.push('  -v "$DFLASH_MODEL_DIR:$DFLASH_MODEL_DIR:ro" \\');
        if (id === 'qwen' && S.specOn && S.spec === 'mtp') out.push('  -v "$PWD/$MTP_DRAFT_VOCAB:/etc/vllm-draft-vocab.txt:ro" \\');
        out.push('  -v "$MODEL_DIR:$MODEL_DIR:ro" -v "$HOME/.cache/vllm:/root/.cache/vllm" \\');
        out.push('  --entrypoint vllm "$IMAGE" "${VLLM_ARGS[@]}"');
        if (cm) { blank(); row('# afterwards: /health, the "GPU KV cache size" and "Maximum concurrency" lines in the log → calibrate g and S_fix'); }
        return out.join('\n');
      }

      /* ==================================================================
         FIGURA 10 — previsto × medido
         ================================================================== */
      (function fig10() {
        const fig = root.querySelector('#fig-validacao');
        const ctl = UI.controls(fig);
        const S = { met: 'w' };
        UI.seg(ctl, { label: 'Metric', options: [{ v: 'w', label: 'weights/rank' }, { v: 'pool', label: 'KV pool' }, { v: 'tok', label: 'log tokens' }, { v: 'dec', label: 'decode C=1' }], value: S.met, onChange: (v) => { S.met = v; table(); H.redraw(); } });
        const tbl = h('div', { class: 'lab-vt' });
        fig.insertBefore(tbl, fig.querySelector('figcaption'));
        const isTok = (r) => r.unit === 'tokens';
        const fmtV = (r, v, d) => (isTok(r) ? tk(v) : nf(v, d));
        function rowsFor(met) {
          const rows = MODELS.map((id) => {
            const M = PRESETS[id], ms = M.meas;
            const base = Object.assign(LAB.defaults(M), ms.cfg || {});
            if (met === 'w') { const R = LAB.compute(M, base); return { id, M, est: R.Wrank / GiB, meas: ms.w, unit: 'GiB', cfg: cfgTxt(R.S, M), note: ms.wNote }; }
            if (met === 'pool') { const c = Object.assign({}, base, ms.poolCfg || {}); const R = LAB.compute(M, c); return { id, M, est: R.poolPos / GiB, meas: ms.pool, unit: 'GiB', cfg: cfgTxt(R.S, M) + (R.S.fix ? ' (fixed pool)' : ''), note: ms.poolNote }; }
            if (met === 'tok') { const c = Object.assign({}, LAB.defaults(M), ms.tokCfg || {}); const R = LAB.compute(M, c); return { id, M, est: R.reported, meas: ms.tok, unit: 'tokens', cfg: `${cfgTxt(R.S, M)}, pool ${nf(R.poolPos / GiB, 2)} GiB, KV ${R.S.kv.toUpperCase()}, ${cx(R.S.maxLen)}`, note: ms.tokNote }; }
            const c = Object.assign({}, base); const R = LAB.compute(M, c); const s = R.step(1, true, 1 + ms.decS);
            return { id, M, est: s.tps, meas: ms.dec, unit: 'tok/s', cfg: `${SPECL[R.S.spec]} k=${R.S.k}, τ=${nf(1 + ms.decS, 2)}, 230 GB/s`, note: ms.decNote, lblE: 'ceiling' };
          });
          if (met === 'tok') MODELS.forEach((id) => { const M = PRESETS[id], x = M.extraTok; if (!x) return; const R = LAB.compute(M, Object.assign(LAB.defaults(M), x.cfg)); rows.push({ id, M, name: x.lbl, est: R.reported, meas: x.tok, unit: 'tokens', cfg: `${cfgTxt(R.S, M)}, estimated pool ${nf(R.poolPos / GiB, 1)} GiB, KV ${R.S.kv.toUpperCase()}`, note: x.note }); });
          return rows;
        }
        function cfgTxt(S) { return `TP${S.P}${S.ep ? '+EP' : ''}, ${FMTL[S.expFmt]}${S.expFmt === 'exl3' ? ' ' + nf(S.bpw, 0) + ' bpw' : ''}, GMU ${nf(S.gmu, 3)}${S.specOn ? ', ' + SPECL[S.spec] : ''}`; }
        function table() {
          const rs = rowsFor(S.met);
          tbl.innerHTML = '<table><thead><tr><th>Recipe</th><th>Configuration</th><th>Estimated</th><th>Measured</th><th>Ratio</th><th>Source</th></tr></thead><tbody>' +
            rs.map((r) => `<tr><td>${LA.esc(r.name || r.M.short)}</td><td>${LA.esc(r.cfg)}</td><td>${fmtV(r, r.est, 1)} ${r.unit}</td><td>${r.meas ? (isTok(r) ? nf(r.meas) : nf(r.meas, 2)) + ' ' + r.unit : '—'}</td><td>${r.meas ? nf(r.est / r.meas, 2) : '—'}</td><td>${LA.esc(r.note || '')}</td></tr>`).join('') + '</tbody></table>';
        }
        const H = V.fig(fig, (g, w, hh) => {
          const C = V.C();
          const rs = rowsFor(S.met);
          const mx = Math.max(...rs.map((r) => Math.max(r.est, r.meas || 0))) * 1.12;
          const lx = w < 520 ? 84 : 150, bw = Math.max(40, w - lx - (w < 520 ? 118 : 160));
          const rh = (hh - 60) / rs.length;
          if (w < 520) V.text(g, 'light = estimated · solid = measured', w - 14, 20, { size: 10.5, color: C.muted, align: 'right' });
          const titles = { w: 'weights per rank (GiB)', pool: 'KV pool (GiB)', tok: 'log tokens ("GPU KV cache size") at points outside the calibration', dec: 'decode C=1: model ceiling vs. measured (tok/s)' };
          V.text(g, titles[S.met], 14, w < 520 ? 34 : 20, { size: 12, color: C.muted, font: 'mono', maxW: w - 28 });
          rs.forEach((r, i) => {
            const y = 36 + i * rh;
            const col = r.M.rb ? C[r.M.rb] : C.d;
            const nar = w < 520;
            const nm = r.name || r.M.short;
            V.text(g, nar ? (r.name ? 'Qwen v0.30' : nm.split('-')[0]) : nm, 14, y + rh / 2 - 2, { size: 12, weight: 600, color: C.ink, maxW: lx - 20 });
            V.text(g, r.name ? 'other engine' : r.M.rb ? 'recipe' : 'tutorial', 14, y + rh / 2 + 13, { size: 10.5, color: C.muted });
            const bh = Math.min(16, rh / 2 - 6);
            const we = (r.est / mx) * bw, wm = ((r.meas || 0) / mx) * bw;
            V.box(g, lx, y + rh / 2 - bh - 2, Math.max(2, we), bh, { r: 4, fill: V.alpha(col, 0.28), stroke: col, lw: 1 });
            V.text(g, `${nar ? '' : (r.lblE || 'estimated') + ' '}${fmtV(r, r.est, 1)}`, lx + we + 6, y + rh / 2 - 5, { size: 11, font: 'mono', color: C.ink });
            if (r.meas) {
              V.box(g, lx, y + rh / 2 + 2, Math.max(2, wm), bh, { r: 4, fill: V.alpha(col, 0.8) });
              V.text(g, `${nar ? '' : (r.lblM || 'measured') + ' '}${fmtV(r, r.meas, 1)}`, lx + wm + 6, y + rh / 2 + bh - 1, { size: 11, font: 'mono', color: C.ink });
              const ratio = r.est / r.meas;
              const ok = Math.abs(ratio - 1) <= 0.1;
              V.text(g, '×' + nf(ratio, 2), w - 14, y + rh / 2 + 4, { size: 12.5, font: 'mono', weight: 700, align: 'right', color: ok ? C.a : C.d });
            }
          });
        }, { static: true });
        table();
      })();
    },
  });
})();

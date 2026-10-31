/* ==========================================================================
   LATENT ATLAS — data/mlx-models.js
   Modelos MLX sugeridos para o copilot local (ids verificados no Hugging Face
   em 2026-10-01) + a regra de encaixe na memória unificada do Mac.
   bytes = tamanho exato do download com os padrões que o mlx-lm baixa
   (LA.mlxPatterns). Sem divisão por idioma: as notas trazem {en, pt}.
   ========================================================================== */
(function () {
  'use strict';
  const LA = (window.LA = window.LA || {});
  const GiB = 1073741824;

  /** arquivos que o mlx-lm baixa (mlx_lm.utils DEFAULT_ALLOW_PATTERNS — fnmatch, "*" também casa "/") */
  LA.mlxPatterns = ['*.json', 'model*.safetensors', '*.py', 'tokenizer.model', '*.tiktoken', 'tiktoken.model', '*.txt', '*.jsonl', '*.jinja'];

  /**
   * Encaixe de um modelo na GPU do Mac.
   * budget ≈ recommendedMaxWorkingSetSize do Metal: 2/3 da RAM até 36 GiB, 3/4 acima.
   * need = pesos + max(1 GiB, 10%) — KV cache de prompts de ~9k tokens + ativações + runtime.
   */
  LA.mlxFit = function (bytes, ramBytes) {
    const ram = +ramBytes || 16 * GiB;
    const budget = ram <= 36 * GiB ? ram * 2 / 3 : ram * 3 / 4;
    const b = +bytes || 0;
    const need = b + Math.max(GiB, 0.1 * b);
    const fit = need <= 0.9 * budget ? 'ok' : b <= budget ? 'tight' : 'no';
    return { fit, budget, need };
  };

  // id, bytes, params, ativos, model_type, flags, rank, nota en, nota pt
  const M = (id, bytes, params, active, type, flags, rank, en, pt) => {
    const f = flags ? flags.split(/\s*,\s*/) : [];
    return {
      id, bytes, params, active: active || null, type,
      moe: f.includes('moe'), think: f.includes('think'), vision: f.includes('vision'),
      harmony: f.includes('harmony'), qat: f.includes('qat'),
      bits: /gpt-oss/.test(id) ? 'MXFP4' : /-8bit$/.test(id) ? '8-bit' : '4-bit',
      rank, note: { en, pt },
    };
  };

  LA.mlxModels = [
    M('mlx-community/Qwen3-0.6B-4bit', 351383618, '0.6B', null, 'qwen3', 'think', 1,
      'Tiny — tests the whole pipeline in minutes; weak answers.',
      'Minúsculo — testa o pipeline inteiro em minutos; respostas fracas.'),
    M('mlx-community/Qwen3-1.7B-4bit', 984013244, '1.7B', null, 'qwen3', 'think', 2,
      'Very light; fine for short definitions.',
      'Bem leve; serve para definições curtas.'),
    M('mlx-community/Qwen3-4B-Instruct-2507-4bit', 2278969697, '4B', null, 'qwen3', '', 3,
      'Best small all-rounder; the pick for 8 GB Macs.',
      'Melhor pequeno para tudo; a escolha para Macs de 8 GB.'),
    M('mlx-community/Llama-3.1-8B-Instruct-4bit', 4534808112, '8B', null, 'llama', '', 3.5,
      'Classic dense 8B; solid English, weaker Portuguese.',
      'Clássico denso de 8B; bom inglês, português mais fraco.'),
    M('mlx-community/Qwen3.5-9B-MLX-4bit', 5977071067, '9B', null, 'qwen3_5', 'think, vision', 4.5,
      'Hybrid thinking model; the copilot turns thinking off by default.',
      'Modelo com thinking híbrido; o copilot desliga o thinking por padrão.'),
    M('mlx-community/gemma-4-12B-it-4bit', 6773371194, '12B', null, 'gemma4_unified', 'vision', 5,
      'Gemma 4 12B in plain 4-bit; strong multilingual answers.',
      'Gemma 4 12B em 4 bits simples; ótimo em várias línguas.'),
    M('mlx-community/gemma-4-12B-it-qat-OptiQ-4bit', 8931635151, '12B', null, 'gemma4_unified', 'qat', 5.5,
      'Quantization-aware training + mixed 4/8-bit layers: closer to BF16 quality.',
      'Treino ciente de quantização + camadas mistas 4/8 bits: mais perto da qualidade BF16.'),
    M('mlx-community/gpt-oss-20b-MXFP4-Q8', 12104215835, '21B', '3.6B', 'gpt_oss', 'moe, think, harmony', 5,
      'MoE with MXFP4 experts — fast, but mlx-lm shows its "harmony" channels as raw text.',
      'MoE com experts em MXFP4 — rápido, mas o mlx-lm mostra os canais "harmony" como texto cru.'),
    M('mlx-community/gemma-4-26b-a4b-it-4bit', 15373586252, '26B', '~4B', 'gemma4', 'moe, vision', 7.6,
      'MoE with ~4B active: big-model quality at small-model speed.',
      'MoE com ~4B ativos: qualidade de modelo grande na velocidade de um pequeno.'),
    M('mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit', 17197084801, '30B', '~3B', 'qwen3_moe', 'moe', 6.5,
      'MoE with ~3B active; very fast, great instruction following.',
      'MoE com ~3B ativos; muito rápido, segue instruções muito bem.'),
    M('mlx-community/Qwen3.8-27B-4bit', 16081488731, '27B', null, 'qwen3_5', 'think, vision', 7.2,
      'Dense 27B: top quality, slower per token.',
      'Denso de 27B: qualidade de ponta, mais lento por token.'),
    M('mlx-community/Qwen3.6-35B-A3B-4bit', 20429166969, '35B', '~3B', 'qwen3_5_moe', 'moe, think, vision', 8,
      'MoE with ~3B active and hybrid attention; close to the 27B dense, much faster.',
      'MoE com ~3B ativos e atenção híbrida; perto do denso de 27B, bem mais rápido.'),
    M('mlx-community/Qwen3.8-27B-8bit', 29531519120, '27B', null, 'qwen3_5', 'think, vision', 7.9,
      '8-bit dense 27B: near-lossless but slow on bandwidth.',
      'Denso de 27B em 8 bits: quase sem perda, mas limitado pela banda.'),
    M('mlx-community/Qwen3.6-35B-A3B-8bit', 37748365700, '35B', '~3B', 'qwen3_5_moe', 'moe, think, vision', 8.6,
      '8-bit MoE: near-lossless and fast — the pick for 64 GB+.',
      'MoE em 8 bits: quase sem perda e rápido — a escolha para 64 GB+.'),
    M('mlx-community/Qwen3-Coder-Next-4bit', 44855963315, '80B', '~3B', 'qwen3_next', 'moe', 6,
      '80B MoE with hybrid linear attention, specialized in code.',
      'MoE de 80B com atenção linear híbrida, especializado em código.'),
    M('mlx-community/gpt-oss-120b-mxfp4-bf16', 65276865000, '117B', '5.1B', 'gpt_oss', 'moe, think, harmony', 8,
      '117B MoE in MXFP4; raw "harmony" channels appear in answers.',
      'MoE de 117B em MXFP4; canais "harmony" aparecem crus nas respostas.'),
  ];

  /** sugestão: o modelo curado com fit 'ok', sem harmony, de maior rank */
  LA.mlxRecommend = function (ramBytes) {
    let best = null;
    LA.mlxModels.forEach((m) => {
      if (m.harmony || LA.mlxFit(m.bytes, ramBytes).fit !== 'ok') return;
      if (!best || m.rank > best.rank) best = m;
    });
    return best;
  };
})();

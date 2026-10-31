#!/usr/bin/env python3
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
MARK = "# [mimo-recipe] cache_config per layer"
if MARK in s:
    open(dst, "w").write(s); print("[overlay] já aplicado"); sys.exit(0)
if re.search(r"cache_config\s*=\s*layer_cache_config", s):
    open(dst, "w").write(s); print("[overlay] already upstream"); sys.exit(0)
# Âncora 1: início do __init__ do MiMoV2FlashDecoderLayer
a1 = "        quant_config = vllm_config.quant_config\n        layer_id = extract_layer_index(prefix)\n"
if s.count(a1) != 1:
    sys.exit("[overlay] âncora 1 não encontrada exatamente 1x — a imagem mudou; revise o patch")
s = s.replace(a1, a1 + f"        {MARK}\n        import copy\n"
    "        layer_cache_config = copy.copy(vllm_config.cache_config)\n"
    "        if layer_cache_config is not None:\n"
    "            layer_cache_config.sliding_window = None  # globais continuam globais\n")
# Âncora 2: as duas construções de MiMoV2Attention recebem cache_config
n = s.count("                quant_config=quant_config,\n                partial_rotary_factor=")
if n != 2:
    sys.exit(f"[overlay] esperava 2 construções de MiMoV2Attention, achei {n}")
s = s.replace("                quant_config=quant_config,\n                partial_rotary_factor=",
              "                quant_config=quant_config,\n                cache_config=layer_cache_config,\n"
              "                partial_rotary_factor=")
open(dst, "w").write(s); print("[overlay] cache_config encaminhado (2 camadas-tipo)")

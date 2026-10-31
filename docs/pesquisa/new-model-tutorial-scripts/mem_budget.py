#!/usr/bin/env python3
"""Orçamento de memória por rank para MiMo-V2 (ou modelos com o mesmo formato de config).
Uso: python3 mem_budget.py config.json --tp 2 --kv fp8 --gmu 0.85 --overhead-gib 4 --weights-gib 82.7
Os números de pesos vêm do cabeçalho safetensors (ou do log do vLLM); este script só faz a aritmética."""
import argparse, json, math

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--kv", choices=["bf16", "fp8"], default="fp8")
ap.add_argument("--gmu", type=float, default=0.85)
ap.add_argument("--visible-gib", type=float, default=121.69)   # GB10: memória visível ao CUDA (medida)
ap.add_argument("--weights-gib", type=float, default=82.7)      # por rank, medido (SGLang/vLLM, TP2)
ap.add_argument("--overhead-gib", type=float, default=4.0)      # ativações de pico + non-torch + CUDA graphs
ap.add_argument("--group-size", type=int, default=5)            # tamanho de grupo do alocador híbrido (main=5; #58175/#58207 -> 3)
ap.add_argument("--block", type=int, default=64)                # tokens por bloco (ordem de grandeza)
ap.add_argument("--spec-k", type=int, default=7)
ap.add_argument("--ctx", type=int, nargs="*", default=[32768, 131072, 262144, 1048576])
a = ap.parse_args()
c = json.load(open(a.config))
b = 2 if a.kv == "bf16" else 1
pat = c["hybrid_layer_pattern"]; n_ga = pat.count(0); n_swa = pat.count(1)
kv_ga = c["num_key_value_heads"] // a.tp; kv_swa = c["swa_num_key_value_heads"] // a.tp
d_ga = c["head_dim"] + c["v_head_dim"]; d_swa = c["swa_head_dim"] + c["swa_v_head_dim"]
ga_tok = n_ga * kv_ga * d_ga * b                 # bytes/token/rank que crescem com o contexto
swa_tok = n_swa * kv_swa * d_swa * b              # bytes/token/rank, mas só para a janela
win = c["sliding_window"]
swa_req = swa_tok * (math.ceil((win + a.spec_k) / a.block) + 1) * a.block
pad = math.ceil(n_ga / a.group_size) * a.group_size / n_ga   # enchimento de grupo das camadas globais
budget = a.visible_gib * a.gmu
kv_gib = budget - a.weights_gib - a.overhead_gib
kv_b = kv_gib * 2**30
print(f"camadas globais={n_ga} SWA={n_swa} | KV heads/rank: global={kv_ga} SWA={kv_swa} | janela={win}")
print(f"KV global: {ga_tok} B/token/rank ({a.kv}); com enchimento gs={a.group_size}: {ga_tok*pad:.0f} B")
print(f"KV SWA: {swa_tok} B/token/rank, custo fixo por requisição ~{swa_req/1e6:.1f} MB (janela+k+bloco)")
print(f"orçamento GMU {a.gmu}: {budget:.2f} GiB - pesos {a.weights_gib} - overhead {a.overhead_gib} = KV {kv_gib:.2f} GiB")
if kv_gib <= 0:
    raise SystemExit("não sobra KV: reduza pesos, overhead ou aumente GMU")
cap = kv_b / (ga_tok * pad)
print(f"capacidade ~{cap/1e6:.2f} M tokens (ideal sem enchimento {kv_b/ga_tok/1e6:.2f} M)")
for L in a.ctx:
    per_req = L * ga_tok * pad + swa_req
    print(f"  ctx {L:>8}: {kv_b/per_req:6.2f} requisições cheias simultâneas")

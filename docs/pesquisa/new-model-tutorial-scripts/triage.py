#!/usr/bin/env python3
"""Triagem de um checkpoint no Hub sem baixar os pesos.
Uso: python3 triage.py XiaomiMiMo/MiMo-V2.6-Flash-MOPD [revision]
Lê config.json, o índice safetensors e o *cabeçalho* (range request) de alguns shards."""
import collections, json, re, struct, sys, urllib.request

REPO = sys.argv[1]
REV = sys.argv[2] if len(sys.argv) > 2 else "main"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}"

def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def header(fname):
    n = struct.unpack("<Q", get(f"{BASE}/{fname}", "0-7"))[0]
    return json.loads(get(f"{BASE}/{fname}", f"8-{8 + n - 1}"))

cfg = json.loads(get(f"{BASE}/config.json"))
idx = json.loads(get(f"{BASE}/model.safetensors.index.json"))
wm = idx["weight_map"]
print("architectures:", cfg.get("architectures"), "| model_type:", cfg.get("model_type"))
print("index metadata:", idx.get("metadata"))
print("tensores:", len(wm), "| arquivos:", len(set(wm.values())))

pat = cfg.get("hybrid_layer_pattern") or []
if pat:
    ga = [i for i, x in enumerate(pat) if x == 0]
    print(f"camadas: {len(pat)} | globais (0): {len(ga)} -> {ga} | SWA (1): {len(pat) - len(ga)}")
q = cfg.get("quantization_config", {})
print("quant:", {k: v for k, v in q.items() if k != "ignored_layers"},
      "| ignored:", collections.Counter(re.sub(r"\.\d+\.", ".N.", x) for x in q.get("ignored_layers", [])))

# Um shard por "tipo" de tensor: camada 0 (densa/global), 1 (SWA+MoE), MTP
wanted = ["model.layers.0.self_attn.qkv_proj.weight", "model.layers.1.self_attn.qkv_proj.weight",
          "model.layers.1.mlp.experts.0.gate_proj.weight", "model.mtp.layers.0.self_attn.qkv_proj.weight"]
seen = {}
for t in wanted:
    f = wm.get(t)
    if f and f not in seen:
        seen[f] = header(f)
dt = collections.Counter()
for f, h in seen.items():
    for k, v in h.items():
        if k == "__metadata__":
            continue
        m = re.match(r"model\.(layers\.[01]|mtp\.layers\.0)\.", k)
        if m and (".experts." not in k or ".experts.0." in k):
            print(f"  {k:70s} {v['dtype']:8s} {v['shape']}")
        dt[v["dtype"]] += 1
print("dtypes (nos shards amostrados):", dict(dt))

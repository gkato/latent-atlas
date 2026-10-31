#!/usr/bin/env bash
# ============================================================================
# Latent Atlas — sobe um modelo MLX local (Apple Silicon) para o copilot.
#   ./tools/start-mlx-copilot.sh           # escolhe o modelo pela RAM do Mac
#   ./tools/start-mlx-copilot.sh 24        # força um perfil (8 | 16 | 24 | 32 | 48 | 64)
#   MODEL=mlx-community/xyz PORT=8080 ./tools/start-mlx-copilot.sh
# Depois, no copilot: provedor "MLX local", Base URL http://127.0.0.1:8080/v1,
# API key vazia, Modelo vazio → "Testar conexão".
# Mais fácil: ./serve.sh e o botão "MLX" na barra do topo (baixa, sobe e liga o copilot sozinho).
# ============================================================================
set -euo pipefail

PORT="${PORT:-8080}"
MAXTOK="${MAXTOK:-2048}"
TEMP="${TEMP:-0.4}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "✕ MLX roda só em Macs com Apple Silicon (arm64). Use outro provedor no copilot." >&2
  exit 1
fi

ram_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
tier="${1:-$ram_gb}"

# ids verificados no Hugging Face em 2026-10-01 — mesma regra de encaixe do botão MLX da plataforma
# (data/mlx-models.js: pesos + folga ≤ 90% do orçamento de GPU ≈ 2/3 da RAM até 36 GB, 3/4 acima)
if [ -z "${MODEL:-}" ]; then
  if   [ "$tier" -ge 64 ]; then MODEL="mlx-community/Qwen3.6-35B-A3B-8bit"            # 37,7 GB · MoE ~3B ativos
  elif [ "$tier" -ge 48 ]; then MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"            # 20,4 GB · MoE ~3B ativos
  elif [ "$tier" -ge 32 ]; then MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"         # 15,4 GB · MoE ~4B ativos
  elif [ "$tier" -ge 16 ]; then MODEL="mlx-community/gemma-4-12B-it-qat-OptiQ-4bit"   #  8,9 GB · QAT, 4/8 bits
  else                          MODEL="mlx-community/Qwen3-4B-Instruct-2507-4bit"     #  2,3 GB
  fi
fi

echo "Mac com ${ram_gb} GB · perfil ${tier} GB · modelo: ${MODEL}"

# mlx-lm precisa de Python >= 3.10 (o python3 do macOS é 3.9)
if ! command -v mlx_lm.server >/dev/null 2>&1; then
  echo "mlx_lm.server não encontrado. Instalando…"
  if command -v uv >/dev/null 2>&1; then
    uv tool install --python 3.12 mlx-lm
    export PATH="$HOME/.local/bin:$PATH"
  elif command -v brew >/dev/null 2>&1; then
    brew install mlx-lm
  else
    cat >&2 <<'MSG'
✕ Preciso de "uv" ou "brew" para instalar o mlx-lm (Python >= 3.10).
  Instale um deles:
    curl -LsSf https://astral.sh/uv/install.sh | sh      # uv
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  # brew
  e rode este script de novo.
MSG
    exit 1
  fi
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "✕ A porta $PORT já está em uso (llama-server/mlx_vlm usam a mesma). Use PORT=8081 ./tools/start-mlx-copilot.sh" >&2
  exit 1
fi

echo "→ http://127.0.0.1:${PORT}/v1   (Ctrl+C para parar; na 1ª vez o download pode levar alguns minutos)"
exec mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$PORT" --max-tokens "$MAXTOK" --temp "$TEMP"

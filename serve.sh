#!/usr/bin/env bash
# Serve o Latent Atlas em http://localhost:8765
# (necessário para o copilot falar com servidores locais que checam CORS/Origin)
# O tools/atlas_server.py também é o helper do botão "MLX" da barra do topo: baixa modelos do Hugging Face,
# instala o mlx-lm e sobe o mlx_lm.server para o copilot. Ctrl+C para tudo (inclusive o servidor MLX).
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
if [ ! -f dist/chapters.bundle.js ] || [ -n "$(find chapters -newer dist/chapters.bundle.js -print -quit 2>/dev/null)" ]; then
  command -v node >/dev/null && node build.js || true
fi
if [ -f tools/atlas_server.py ]; then
  # o helper imprime a URL (se a porta estiver ocupada, usa a próxima livre) e abre o navegador
  exec python3 tools/atlas_server.py --port "${PORT}" --open
fi
# sem o helper: servidor estático simples (o botão MLX fica no modo manual)
echo "Latent Atlas → http://localhost:${PORT}"
( sleep 1; command -v open >/dev/null && open "http://localhost:${PORT}" ) &
exec python3 -m http.server "${PORT}" --bind 127.0.0.1

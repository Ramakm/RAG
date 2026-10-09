#!/usr/bin/env bash
# One-command launcher: creates a virtualenv, installs dependencies and starts BrowseAI.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

if [ ! -d .venv ]; then
  PY=""
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
  done
  [ -z "$PY" ] && { echo "Python 3.10+ is required."; exit 1; }
  echo "→ Creating virtual environment with $PY"
  "$PY" -m venv .venv
fi

if [ ! -f .venv/.deps-installed ] || [ requirements.txt -nt .venv/.deps-installed ]; then
  echo "→ Installing dependencies (first run takes a few minutes)"
  .venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip -q
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt -q
  touch .venv/.deps-installed
fi

if ! curl -s "${OLLAMA_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  echo "⚠  Ollama is not reachable. Start it in another terminal with:  ollama serve"
  echo "   and pull a model, e.g.:                                  ollama pull llama3.2"
  echo "   (BrowseAI still works without it, but answers will be raw passages.)"
fi

echo "→ BrowseAI running at http://localhost:$PORT"
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT"

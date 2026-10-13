#!/usr/bin/env bash
# Set up docket-serve in ./.venv for one backend:
#   ./install.sh gguf   # any Mac, Linux or Windows (WSL) machine, CPU or Apple GPU: serves Docket-3B-GGUF
#   ./install.sh vllm   # Linux + NVIDIA GPU, production serving: Docket-3B
#   ./install.sh hf     # plain Transformers on CPU or GPU, slow, for testing: Docket-3B
# Uses python3 (3.10+), or the interpreter in $PYTHON. Rerun to reinstall.
set -euo pipefail
cd "$(dirname "$0")"
BACKEND=${1:-}
case "$BACKEND" in
  gguf|vllm|hf) ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python 3.10 or newer is required ($PY)."; exit 1; }
if [ "$BACKEND" = vllm ] && [ "$(uname -s)" = Darwin ]; then
  echo "vLLM needs Linux with an NVIDIA GPU. On a Mac use: ./install.sh gguf"; exit 1
fi
"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
if [ "$BACKEND" = gguf ]; then
  # Prebuilt llama.cpp wheels (Metal on Apple Silicon); no compiler or PyTorch needed.
  .venv/bin/pip install -r requirements.txt
else
  .venv/bin/pip install -e ".[$BACKEND]"
fi
echo "$BACKEND" > .venv/docket-backend
echo "Installed the $BACKEND backend. Next: ./serve.sh $([ "$BACKEND" = gguf ] && echo 3b-gguf || echo 3b)"

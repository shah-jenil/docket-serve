#!/usr/bin/env bash
# Set up docket-serve in ./.venv for one backend:
#   ./install.sh gguf   # Mac (Metal), CPU or a small GPU, via llama.cpp: serves Docket-3B-GGUF
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
if [ "$(uname -s)" = Darwin ]; then
  if [ "$BACKEND" = vllm ]; then
    echo "vLLM needs Linux with an NVIDIA GPU. On a Mac use: ./install.sh gguf"; exit 1
  fi
  if [ "$("$PY" -c 'import platform; print(platform.machine())')" != arm64 ]; then
    echo "warning: $PY is not a native arm64 Python; Metal acceleration needs one." >&2
  fi
  if [ "$BACKEND" = gguf ]; then export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_METAL=on}"; fi
fi
"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
if [ "$BACKEND" = hf ]; then
  .venv/bin/pip install .
else
  # --no-cache-dir: never reuse a llama.cpp build compiled without Metal/CUDA.
  .venv/bin/pip install --no-cache-dir ".[$BACKEND]"
fi
echo "$BACKEND" > .venv/docket-backend
echo "Installed the $BACKEND backend. Next: ./serve.sh $([ "$BACKEND" = gguf ] && echo 3b-gguf || echo 3b)"

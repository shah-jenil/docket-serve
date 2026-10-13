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
# Check system packages before downloading anything. Stock Ubuntu/Debian images (including
# most cloud GPU images) lack python3-venv, and vLLM compiles kernels at startup with a C
# compiler and the Python headers.
if [ "$(uname -s)" = Linux ]; then
  V=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  MISSING=()
  "$PY" -c 'import ensurepip' 2>/dev/null || MISSING+=("python$V-venv")
  if [ "$BACKEND" = vllm ]; then
    "$PY" -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
      || MISSING+=("python$V-dev")
    command -v "${CC:-gcc}" >/dev/null || command -v cc >/dev/null || MISSING+=(gcc)
  fi
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Missing system packages: ${MISSING[*]}"
    if command -v apt-get >/dev/null; then
      SUDO=$([ "$(id -u)" = 0 ] || echo "sudo ")
      echo "Install them, then rerun ./install.sh $BACKEND:"
      echo "  ${SUDO}apt-get update && ${SUDO}apt-get install -y ${MISSING[*]}"
    else
      echo "Install the venv module, the Python $V development headers and gcc with your package manager, then rerun ./install.sh $BACKEND."
    fi
    exit 1
  fi
  if [ "$BACKEND" = vllm ] && ! command -v nvidia-smi >/dev/null; then
    echo "Warning: nvidia-smi not found. The vllm backend needs an NVIDIA GPU and driver."
  fi
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
if [ "$BACKEND" = vllm ]; then
  echo "The first start downloads the 7.3 GB model, then vLLM compiles and captures CUDA graphs:"
  echo "allow a few minutes until it prints 'serving on'. See docs/VLLM.md."
fi

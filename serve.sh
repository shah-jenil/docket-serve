#!/usr/bin/env bash
# Start a released Docket model on http://127.0.0.1:8080 (set HOST / PORT to change):
#   ./serve.sh 3b-gguf   # needs ./install.sh gguf
#   ./serve.sh 3b        # needs ./install.sh vllm (or hf)
# The first start downloads the model from Hugging Face into its cache. Extra arguments go to
# docket-serve, e.g. ./serve.sh 3b-gguf --max-model-len 4096 (see docs/API.md).
set -euo pipefail
cd "$(dirname "$0")"
HF_ORG=${DOCKET_HF_ORG:-shahjenil}
[ -f .venv/docket-backend ] || { echo "Run ./install.sh first."; exit 1; }
INSTALLED=$(cat .venv/docket-backend)
case "$HF_ORG" in *"<"*) echo "Set DOCKET_HF_ORG to the Hugging Face namespace that hosts the Docket models."; exit 1 ;; esac
MODEL=${1:-}
if [ $# -gt 0 ]; then shift; fi
case "$MODEL" in
  3b)
    [ "$INSTALLED" != gguf ] || { echo "$MODEL needs ./install.sh vllm (or hf); this install is gguf. Use ./serve.sh 3b-gguf."; exit 1; }
    ARGS=(--backend "$INSTALLED" --model "$HF_ORG/Docket-3B") ;;
  3b-gguf)
    [ "$INSTALLED" = gguf ] || { echo "3b-gguf needs ./install.sh gguf (this install is $INSTALLED)."; exit 1; }
    ARGS=(--backend gguf --model "$HF_ORG/Docket-3B-GGUF") ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
exec .venv/bin/docket-serve "${ARGS[@]}" --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"

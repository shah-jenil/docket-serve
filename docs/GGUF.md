# GGUF serving

The GGUF backend serves Docket's existing System One API using llama.cpp. It reads
candidate logits at the same answer position as the HF backend and applies the
bundled per-type calibration. It does not generate a chat answer to infer a choice.

## Install and serve

From a clone of this repository, on an Apple Silicon Mac with native arm64 Python 3.10+
and Xcode command line tools:

```bash
./install.sh gguf          # builds llama.cpp with Metal on a Mac
./serve.sh 3b-gguf         # downloads shahjenil/Docket-3B-GGUF on first start
python3 scripts/smoke_test.py
```

`install.sh gguf` creates `.venv` and runs
`CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir ".[gguf]"` on a Mac (on Linux,
without the Metal flag). The server itself is
`.venv/bin/docket-serve --backend gguf --model shahjenil/Docket-3B-GGUF`.

`--model` accepts the Hugging Face repo id (the `.gguf`, tokenizer files, chat template and
`docket_calibration.json` are downloaded together into the Hugging Face cache), a local
folder containing exactly one `.gguf`, or a `.gguf` file path. With a local copy, keep the
original tokenizer files, chat template and `docket_calibration.json` beside the GGUF file.
`--tokenizer /path/to/tokenizer` overrides the tokenizer location; calibration is still read
from the GGUF's directory. Model names and temperatures use that calibration sidecar, as with
BF16 serving.

`--n-gpu-layers -1` requests full available GPU offload. Acceleration requires a
binding built for that backend: Metal on macOS, CUDA on NVIDIA machines. A CPU-only
binding cannot gain acceleration just from the command-line flag.

If a CPU binding is already installed, rebuild it for Metal before serving:

```bash
CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install --force-reinstall --no-cache-dir \
  --no-binary=llama-cpp-python "llama-cpp-python==0.3.36"
```

Build flags follow the [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python#installation).

CPU-only operation:

```bash
./serve.sh 3b-gguf --n-gpu-layers 0 --n-threads 4
```

This backend processes one question at a time, with a fresh KV evaluation per
question. Concurrent API requests are serialized around the model. It does not
provide vLLM's continuous batching or shared-state prefix caching.

The default context is 8,192 tokens, reserving one answer token. `--max-model-len`
sets the GGUF context and prompt limit. Context buffers add memory beyond the
weight file, so download size alone is not a total RAM requirement.

Mac performance has not been measured; GPU-host results cannot establish Mac latency.

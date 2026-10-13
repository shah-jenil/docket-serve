# GGUF serving

The GGUF backend serves Docket's existing System One API using llama.cpp. It reads
candidate logits at the same answer position as the HF backend and applies the
bundled per-type calibration. It does not generate a chat answer to infer a choice.

## Install and serve

One install for macOS, Linux and Windows, CPU or Apple GPU. It needs Python 3.10+ and
nothing else: no compiler, Xcode, CUDA or PyTorch. From a clone of this repository:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/docket-serve                  # serves shahjenil/Docket-3B-GGUF on http://127.0.0.1:8080
python3 scripts/smoke_test.py           # in a second terminal
```

On Windows: `py -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`,
`.venv\Scripts\docket-serve`. `./install.sh gguf` followed by `./serve.sh 3b-gguf` does the
same on macOS and Linux.

`requirements.txt` installs llama-cpp-python 0.3.36 (the version the GGUF release was
converted and checked with) from the project's prebuilt wheel index,
`https://abetlen.github.io/llama-cpp-python/whl/cpu`, which has wheels for Apple Silicon
macOS (with Metal), Linux x86_64 and aarch64, and Windows x86_64. On other platforms, such as
Intel Macs, pip falls back to building llama.cpp from source, which needs a C++ compiler and
CMake (on a Mac, `xcode-select --install`). The rest of the install is transformers (for the
tokenizer only), FastAPI and numpy, about 350 MB in total.

The first start downloads the `.gguf` (3.9 GB) together with its tokenizer files, chat
template and `docket_calibration.json` into the Hugging Face cache; later starts load it
from there.

## Model location

Without `--model`, a GGUF install serves `shahjenil/Docket-3B-GGUF`. `--model` accepts that
or another Hugging Face repo id, a local folder containing exactly one `.gguf`, or a `.gguf`
file path; a model whose name contains `gguf` selects the gguf backend automatically, and
`--backend gguf` forces it. With a local copy, keep the original tokenizer files, chat
template and `docket_calibration.json` beside the GGUF file. `--tokenizer /path/to/tokenizer`
overrides the tokenizer location; calibration is still read from the GGUF's directory. Model
names and temperatures use that calibration sidecar, as with BF16 serving.

## Acceleration

`--n-gpu-layers -1` (the default) offloads every layer the binding can accelerate; on a
CPU-only binding it has no effect.

- **Apple Silicon:** the default install uses Metal; nothing to do.
- **NVIDIA (Linux or Windows):** install the CUDA wheel instead, after the default install:

  ```bash
  .venv/bin/pip install --force-reinstall --no-deps "llama-cpp-python==0.3.36" \
    --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
  ```

  It needs an NVIDIA driver for CUDA 12.4 or newer. For production serving on NVIDIA GPUs,
  the vLLM backend with Docket-3B is faster (continuous batching and prefix caching).
- **CPU only:** `docket-serve --n-gpu-layers 0 --n-threads N`, with `N` the number of
  physical cores.
- **Faster CPU on x86 servers (optional):** the prebuilt Linux x86_64 wheel uses AVX2 only.
  On CPUs with AVX-512 (most recent Xeon and EPYC servers), a build compiled for the machine
  is several times faster but holds a second, repacked copy of the weights in memory. It
  needs a C++ compiler and CMake and takes a few minutes:

  ```bash
  .venv/bin/pip install --force-reinstall --no-deps --no-cache-dir \
    --no-binary llama-cpp-python "llama-cpp-python==0.3.36"
  ```

  On a 4-vCPU AVX-512 cloud VM, this cut a 3-question request (355 prompt tokens) from
  about 16 s to 4.3 s, raised server memory from 4.6 GB to 8.0 GB, and returned the same
  answers with probabilities within 0.02 of the prebuilt wheel's. On AVX2-only CPUs
  (most laptops and desktops) the prebuilt wheel already uses the fastest kernels available.

## Limits and memory

This backend processes one question at a time, with a fresh KV evaluation per
question. Concurrent API requests are serialized around the model. It does not
provide vLLM's continuous batching or shared-state prefix caching. On CPU, time per
question grows with prompt length: on a 4-vCPU x86_64 cloud VM with the prebuilt wheel, a
~120-token question took about 5 seconds (about 1.4 seconds with the native build above).

The default context is 8,192 tokens, reserving one answer token. `--max-model-len`
sets the GGUF context and prompt limit. With the prebuilt wheel the server used about
4.6 GB of memory at the default context; `--max-model-len 4096` uses less. Download size
alone is not a total RAM requirement.

Mac performance has not been measured; GPU-host results cannot establish Mac latency.

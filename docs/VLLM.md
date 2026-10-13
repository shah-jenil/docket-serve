# GPU serving (vLLM)

The vLLM backend serves Docket-3B (BF16) on Linux with an NVIDIA GPU, with continuous
batching and prefix caching. It is the production path; for a Mac or CPU see
[GGUF.md](GGUF.md).

## Requirements

- Linux with an NVIDIA GPU (compute capability 8.0+, e.g. RTX 30xx or newer, A10, L4, L40S,
  A100, H100) with **12 GB or more** of memory, and a recent driver (`nvidia-smi` works).
  The weights take 6.9 GB; CUDA graphs and the KV cache use the rest.
- Python 3.10+ with the `venv` module, a C compiler and the Python development headers.
  vLLM compiles small kernels when it starts, so the compiler and headers are needed at
  run time, not just to install. Stock Ubuntu and Debian images, including most cloud GPU
  images, ship without them:

  ```bash
  sudo apt-get update && sudo apt-get install -y python3-venv python3-dev gcc
  ```

  Use the versioned names (`python3.10-venv python3.10-dev` on Ubuntu 22.04) if your
  `python3` is not the system default. `./install.sh vllm` checks for all three and prints
  the exact command if any is missing.
- About 16 GB of free disk: 8 GB for the Python environment, 7 GB for the model.

## Install and serve

```bash
git clone https://github.com/shah-jenil/docket-serve.git && cd docket-serve
./install.sh vllm      # 3-8 minutes, mostly downloads; installs vLLM 0.31.0 into ./.venv
./serve.sh 3b          # http://127.0.0.1:8080
python3 scripts/smoke_test.py      # in a second terminal: prints PASS
```

`./serve.sh 3b` runs `.venv/bin/docket-serve --backend vllm --model shahjenil/Docket-3B`;
any extra arguments are passed through ([options](API.md#server-options)). Set `HOST=0.0.0.0`
to accept connections from other machines, or keep the default and use an SSH tunnel
(`ssh -L 8080:127.0.0.1:8080 <gpu-host>`). Setting `HF_TOKEN` is optional; it only raises
Hugging Face download rate limits.

## What a start looks like

The server prints `serving on http://...` when it is ready. Until then, vLLM logs progress
bars for loading, compiling and capturing CUDA graphs. Measured on an RTX 3060 (12 GB):

| Step | First start | Later starts |
|---|---|---|
| Download the model (7.3 GB, into the Hugging Face cache) | 1-2 minutes | cached |
| torch.compile | 21 s | cached, under 1 s |
| CUDA graph capture (runs twice: once to measure memory, then for serving) | 2 x 100 s | 2 x 100 s |
| **Ready** | **4.5-6.5 minutes** | **about 4 minutes** |

Graph capture time scales with `--cudagraph-max-tokens` (default 2048: whole prefills up to
2,048 tokens run as CUDA graphs, the lowest latency on prompts of that size). On a small GPU,
`./serve.sh 3b --cudagraph-max-tokens 512` was ready in 100 s instead of about 230 s, with the
same 77 ms p50 on the smoke test's 355-token request; prompts longer than 512 tokens then run
without graphs and pay more fixed overhead. Faster GPUs capture graphs faster.

On the RTX 3060, the smoke test (3 questions per request) gave p50 48-78 ms, with a KV cache of
about 23,000 tokens at the default `--gpu-memory-utilization 0.9`.

## Stopping

Ctrl-C, `kill <pid>`, `docker stop` and systemd all shut the server down cleanly, including
vLLM's engine process (`VLLM::EngineCore`), which holds the GPU memory. If a server was
killed with `kill -9`, that engine process can survive and keep the GPU busy; the next start
then fails with `Free memory on device ... is less than desired GPU memory utilization`.
Check with `nvidia-smi` and stop it with `pkill -f 'VLLM::EngineCore'`.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ensurepip is not available` | no venv module | `sudo apt-get install -y python3-venv`, rerun `./install.sh vllm` |
| `Failed to find C compiler` | vLLM compiles kernels at startup | `sudo apt-get install -y gcc` |
| `Python.h: No such file or directory` | no Python headers | `sudo apt-get install -y python3-dev` |
| `Free memory on device ... is less than desired` | another process holds the GPU | `nvidia-smi`; stop a leftover `VLLM::EngineCore`, or lower `--gpu-memory-utilization` |
| `CUDA out of memory` on a 12 GB card | KV cache plus graphs too large | `--max-model-len 4096` or `--gpu-memory-utilization 0.85` |

docket-serve checks for the compiler and headers before loading vLLM and names the packages
to install, so with the current version you see one clear message instead of a traceback.

# Docket

**Typed decisions for workflows and agents: choice, yes/no and score, with calibrated
probabilities, from a model you host yourself.**

Docket is a Jev-style decision model built on IBM Granite. Give it context, instructions,
and valid answers to classify text, route work, score relevance, or choose an agent action.
It scores allowed answers in one forward pass per question, without generating an explanation.

| Model | Base | Enterprise accuracy | Reasoning accuracy | JevBench /231 |
|---|---|---|---|---|
| [Docket-3B](https://huggingface.co/shahjenil/Docket-3B) | IBM Granite 4.2 3B | 90.1% | 78.2% | 177 |

Enterprise: 3,063 questions, equally weighted across ten workflows. Reasoning:
743 questions, equally weighted across five tasks. Results use bf16 and development-fitted
calibration. Docket-3B serves an 8,192-token context; that limit does not establish long-input
accuracy. See the model card and [full benchmarks](https://jenil.net/docket/results/) for conditions.
FP8 quality and serving performance have not been remeasured; existing GPU studies are
reference measurements.

[Docket-3B-GGUF](https://huggingface.co/shahjenil/Docket-3B-GGUF) (Q8_0) runs the 3B on a Mac, a CPU or a small
GPU through llama.cpp. On the 1,326-question core comparison, workflow-weighted accuracy was
85.26% versus 85.28% for BF16, with 98.87% prediction agreement; the GGUF card has the
per-slice results.

## Quick start

Models are on Hugging Face; this repository is the server. Clone it, install one backend, and
start a model. The first start downloads the model into the Hugging Face cache.

```bash
git clone https://github.com/shah-jenil/docket-serve.git docket && cd docket
```

**Mac (Apple Silicon), CPU or a small GPU: Docket-3B GGUF via llama.cpp**

```bash
./install.sh gguf        # needs Python 3.10+; on a Mac, Xcode command line tools
./serve.sh 3b-gguf       # http://127.0.0.1:8080
```

**Linux with an NVIDIA GPU: Docket-3B via vLLM**

```bash
./install.sh vllm
./serve.sh 3b
```

`./install.sh hf` installs plain Transformers instead of vLLM (CPU or GPU, slow; for testing).
Check a running server end to end, including calibration and latency:

```bash
python3 scripts/smoke_test.py
```

| Release | Format | Command |
|---|---|---|
| [Docket-3B](https://huggingface.co/shahjenil/Docket-3B) | BF16 | `./serve.sh 3b` |
| [Docket-3B-GGUF](https://huggingface.co/shahjenil/Docket-3B-GGUF) | GGUF Q8_0 | `./serve.sh 3b-gguf` |

`serve.sh` passes extra options to `docket-serve`, e.g. `./serve.sh 3b-gguf --max-model-len 4096`
(less memory). `HOST` and `PORT` change the address. Without the scripts, the same server runs as
`.venv/bin/docket-serve --model <path-or-repo-id> [--backend vllm|hf|gguf]`
([options](docs/API.md#server-options), [GGUF details](docs/GGUF.md)).
**Docker** (vLLM, GPU): `docker build -t docket-serve . && docker run --gpus all -p 8080:8080 -v /path/to/Docket-3B:/models/docket docket-serve`.

**API.** `docket-serve` speaks the TypeSafe System One wire format, so existing System One
clients work by changing the base URL:

| Endpoint | Purpose |
|---|---|
| `POST /v1/systemone` | System One request/response |
| `POST /api/alpha/decisions` | same body, at OpenRouter's decisions path |
| `GET /v1/models`, `GET /healthz` | discovery, health and counters |

Full reference: [docs/API.md](docs/API.md).

## How Docket was built

Docket-3B is a LoRA fine-tune of IBM Granite 4.2 3B, merged into the weights. It was
selected and reported with grouped held-out splits and a sealed test set. JevBench serves as
an external scoreboard; no JevBench item was used in training or development-time adapter selection.

## Licence

Apache-2.0. The model is a derivative of IBM Granite 4.2 (Apache-2.0). Parts of the server
are adapted from jevify (Apache-2.0). See [NOTICE](NOTICE) for attributions and training
data licences.

Docket is not affiliated with IBM or TypeSafe AI.

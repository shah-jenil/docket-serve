# docket-serve API

The request and response format follows the TypeSafe System One API, so System One clients
work against `docket-serve` by changing the base URL.

## Request

`POST /v1/systemone` (also served at `/api/alpha/decisions`)

```json
{
  "model": "optional, ignored",
  "state": "any text, or a JSON object/array (rendered as indented JSON)",
  "questions": {
    "<question id>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "<one-line description>", "...": "..."}},
    "<question id>": {"type": "noul",   "instructions": "...", "criteria": {"true": "optional: when to answer yes", "false": "optional"}},
    "<question id>": {"type": "score",  "instructions": "...", "criteria": ["level 0 description", "level 1 description", "..."]}
  }
}
```

**Limits:**
- **choice:** 2 to 255 options. Descriptions are optional but improve accuracy.
- **score:** 2 to 10 levels, ordered lowest first.
- **Questions:** at least one per request. All questions about one state are scored
  concurrently and share its cached prefix, so extra questions are cheap.
- **Length:** the prompt (state + question + options) may not exceed `--max-model-len`
  tokens (default 8,192). Longer requests get HTTP 422; nothing is truncated silently.
  Accuracy is validated on prompts up to about 3,000 tokens.

## Response

```json
{
  "model": "docket-3b",
  "answers": {
    "<id>": {"type": "choice", "choice": "<key>", "confidence": 0.0, "probabilities": {"<key>": 0.0}},
    "<id>": {"type": "noul", "noul": 0.0},
    "<id>": {"type": "score", "score": 1.7, "confidence": 0.0, "legend": {"0": "..."}, "probabilities": {"0": 0.0}}
  },
  "usage": {"input_tokens": 512, "output_tokens": 3}
}
```

| Field | Meaning |
|---|---|
| `noul` | P(yes) |
| `choice` | the most probable key |
| `confidence` | for choice: `(p_max - 1/K) / (1 - 1/K)` (0 = uniform, 1 = certain); for score: `1 - std / std_max` |
| `score` | the probability-weighted level index |
| `probabilities` | calibrated with the per-type temperatures in `docket_calibration.json` (start the server with `--no-calibration` for raw probabilities) |
| `usage.input_tokens` | prompt tokens processed |
| `usage.output_tokens` | number of questions answered |

**Errors.** Validation and length errors return HTTP 422 with
`{"detail": [{"loc": ["body"], "msg": "...", "type": "value_error"}]}`.

## Other endpoints

| Endpoint | Returns |
|---|---|
| `GET /v1/models` | the served model name |
| `GET /healthz`, `GET /health` | `{"ok": true, "model": ..., "calibrated": true, "requests": n, "questions": n, "input_tokens": n}` (`calibrated` is false when no `docket_calibration.json` was found or `--no-calibration` is set) |

The response header `x-docket-latency-ms` gives the server-side time per request.

## Server options

```
docket-serve [--model PATH_OR_HF_ID] [--backend vllm|hf|gguf] [--host 127.0.0.1] [--port 8080] [--quantization fp8]
             [--max-model-len 8192] [--max-num-seqs 256] [--gpu-memory-utilization 0.9]
             [--cudagraph-max-tokens 2048] [--name docket-3b] [--no-calibration]
```

`--model` takes a local folder or a Hugging Face repo id; `docket_calibration.json` is read
from the same place. With `--backend gguf` it also accepts a `.gguf` file path, and a repo id such as
`shahjenil/Docket-3B-GGUF` downloads the GGUF together with its tokenizer and calibration.
Without `--model`, the server uses `shahjenil/Docket-3B-GGUF` for the gguf backend and
`shahjenil/Docket-3B` otherwise. Without `--backend`, a model whose id or path contains `gguf`
uses the gguf backend; otherwise vllm when installed, else hf. With neither option, a GGUF
install (`pip install -r requirements.txt`) serves Docket-3B-GGUF.

`--host` defaults to `127.0.0.1` (this machine only); use `--host 0.0.0.0` to accept
connections from other machines. The Docker image sets `HOST=0.0.0.0`.

Environment equivalents: `DOCKET_MODEL`, `DOCKET_NAME`, `DOCKET_BACKEND`, `DOCKET_QUANTIZATION`, `DOCKET_CUDAGRAPH_MAX_TOKENS`, `HOST`, `PORT`;
`DOCKET_HF_ORG` changes the Hugging Face namespace of the default models.

**Latency.** For the lowest latency use an H100-class GPU with `--quantization fp8`, and keep
1-2 requests in flight per replica. Measured p50 at ~1,000 tokens: Docket-3B 17.8 ms. `--cudagraph-max-tokens` (default 2048) runs whole prefills up to that
length as CUDA graphs, which cuts fixed overhead on short prompts. Capturing them takes most
of the startup time on small GPUs (about 200 s at 2048 on an RTX 3060, 100 s total startup at
512); see [VLLM.md](VLLM.md).

**Throughput.** For about 1,000-token decisions, 4-8 requests in flight per GPU reach peak
throughput; more only adds latency. See the model cards for measured numbers.

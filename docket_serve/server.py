"""TypeSafe System One compatible HTTP server.

    docket-serve [--model <path-or-hf-id>] [--backend vllm|hf|gguf] [--port 8080]

With no --model, a GGUF install serves Docket-3B-GGUF and a vLLM/hf install serves Docket-3B,
both downloaded from Hugging Face on first start.

Endpoints
---------
POST /v1/systemone          TypeSafe System One request/response (state + typed questions).
POST /api/alpha/decisions   Same body and response, the path OpenRouter uses for Jev.
GET  /v1/models, /healthz

Answers follow TypeSafe's shapes: noul -> {"noul": P(yes)}; choice -> {"choice", "confidence",
"probabilities"}; score -> {"score": expected level, "confidence", "legend", "probabilities"}.
Every question in a request is scored concurrently; vLLM batches them with other requests'
questions and reuses the shared state prefix from its cache.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .engine import GGUFBackend, HFBackend, VLLMBackend, load_calibration, model_name, probs_from_logprobs
from .wire import SystemOneRequest, SystemOneResponse, choice_confidence, score_confidence, score_expectation


class Service:
    def __init__(self, backend, model_name: str, temperatures: dict[str, float]):
        self.backend = backend
        self.model_name = model_name
        self.temps = temperatures
        self.stats = {"requests": 0, "questions": 0, "input_tokens": 0}

    async def answer(self, req: SystemOneRequest) -> dict:
        qs = {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}
        preps = {k: self.backend.readout.prepare(req.state, q) for k, q in qs.items()}
        logps = await asyncio.gather(*(self.backend.score(p) for p in preps.values()))
        answers, in_tok = {}, 0
        for (qid, q), prep, lp in zip(qs.items(), preps.values(), logps):
            in_tok += len(prep.ids)
            p = probs_from_logprobs(lp, self.temps.get(q["type"], 1.0))
            dist = dict(zip(prep.keys, p))
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(dist.get("1", 0.0), 6)}
            elif q["type"] == "choice":
                ordered = {k: round(dist[k], 6) for k in q["criteria"]}
                answers[qid] = {"type": "choice", "choice": max(dist, key=dist.get), "confidence": round(choice_confidence(ordered), 4),
                                "probabilities": ordered}
            else:
                levels = [dist[str(i)] for i in range(len(q["criteria"]))]
                answers[qid] = {"type": "score", "score": round(score_expectation(levels), 4), "confidence": round(score_confidence(levels), 4),
                                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                "probabilities": {str(i): round(v, 6) for i, v in enumerate(levels)}}
        self.stats["requests"] += 1
        self.stats["questions"] += len(qs)
        self.stats["input_tokens"] += in_tok
        return SystemOneResponse.model_validate({"model": self.model_name, "answers": answers,
                                                 "usage": {"input_tokens": in_tok, "output_tokens": len(qs)}}).model_dump()


def build_app(service: Service) -> FastAPI:
    app = FastAPI(title="docket-serve", version=__version__)

    async def handle(request: Request):
        try:
            req = SystemOneRequest.model_validate(await request.json())
        except Exception as e:  # noqa: BLE001  (TypeSafe returns 422 with a detail list)
            raise HTTPException(status_code=422, detail=[{"loc": ["body"], "msg": str(e), "type": "value_error"}])
        t0 = time.perf_counter()
        try:
            body = await service.answer(req)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=[{"loc": ["body"], "msg": str(e), "type": "value_error"}])
        out = JSONResponse(body)
        out.headers["x-docket-latency-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return out

    app.post("/v1/systemone")(handle)
    app.post("/api/alpha/decisions")(handle)

    @app.get("/healthz")
    @app.get("/health")
    async def health():
        return {"ok": True, "model": service.model_name, "calibrated": bool(service.temps), **service.stats}

    @app.get("/v1/models")
    async def models():
        return {"models": [{"name": service.model_name, "description": "Docket typed-decision model (choice, noul, score)"}]}

    return app


HF_ORG = os.environ.get("DOCKET_HF_ORG", "shahjenil")


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def resolve_model_and_backend(model: str | None, backend: str | None) -> tuple[str, str]:
    """Fill in whichever of model and backend is missing. A model id or path containing "gguf"
    means the gguf backend; otherwise vLLM when installed, else llama.cpp, else transformers."""
    if not backend:
        if model:
            backend = "gguf" if "gguf" in model.lower() else ("vllm" if _installed("vllm") else "hf")
        else:
            backend = "vllm" if _installed("vllm") else ("gguf" if _installed("llama_cpp") else "hf")
    if not model:
        model = f"{HF_ORG}/Docket-3B-GGUF" if backend == "gguf" else f"{HF_ORG}/Docket-3B"
    return model, backend


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DOCKET_MODEL"),
                    help="Hugging Face repo id, local folder or .gguf file (default: the released Docket-3B for this install)")
    ap.add_argument("--name", default=os.environ.get("DOCKET_NAME"), help="model name in responses (default: from docket_calibration.json)")
    ap.add_argument("--backend", choices=["vllm", "hf", "gguf"], default=os.environ.get("DOCKET_BACKEND") or None,
                    help="default: gguf for GGUF models, else vllm if installed, else hf")
    ap.add_argument("--tokenizer", help="GGUF tokenizer directory (default: beside the .gguf file)")
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="GGUF GPU layers: -1 all available, 0 CPU only")
    ap.add_argument("--n-threads", type=int, default=None, help="GGUF CPU thread count")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="token limit per prompt; longer requests get HTTP 422 (the base model supports up to 131072; "
                         "the fine-tune was trained on prompts up to ~3.3K tokens)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--quantization", default=os.environ.get("DOCKET_QUANTIZATION") or None, help="e.g. fp8 (vLLM backend)")
    ap.add_argument("--cudagraph-max-tokens", type=int, default=int(os.environ.get("DOCKET_CUDAGRAPH_MAX_TOKENS", 2048)),
                    help="largest prefill (tokens) run as a captured CUDA graph (vLLM backend)")
    ap.add_argument("--no-calibration", action="store_true")
    import sys
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    a.model, a.backend = resolve_model_and_backend(a.model, a.backend)
    print(f"docket-serve: {a.backend} backend, model {a.model}", file=sys.stderr)
    if a.backend == "vllm":
        backend = VLLMBackend(a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
                              max_num_seqs=a.max_num_seqs, quantization=a.quantization, cudagraph_max_tokens=a.cudagraph_max_tokens)
    else:
        if a.quantization:
            ap.error("--quantization applies only to the vllm backend; GGUF is already quantized")
        backend = (GGUFBackend(a.model, tokenizer=a.tokenizer, max_model_len=a.max_model_len,
                               n_gpu_layers=a.n_gpu_layers, n_threads=a.n_threads)
                   if a.backend == "gguf" else HFBackend(a.model))
    metadata_dir = str(backend.model_path.parent) if a.backend == "gguf" else a.model
    temps = {} if a.no_calibration else load_calibration(metadata_dir)
    if not a.no_calibration and not temps:
        print(f"warning: no docket_calibration.json found for {a.model}; serving raw probabilities", file=sys.stderr)
    if not a.name:
        a.name = model_name(metadata_dir)
    import uvicorn
    print(f"docket-serve: serving on http://{a.host}:{a.port} (check it with: python3 scripts/smoke_test.py)", file=sys.stderr)
    uvicorn.run(build_app(Service(backend, a.name, temps)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

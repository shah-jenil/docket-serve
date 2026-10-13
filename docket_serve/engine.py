"""Decision readout: one forward pass per question, a softmax over the allowed answer tokens.

Three backends share the exact prompt construction:

* ``VLLMBackend`` (production): vLLM's async engine with continuous batching across requests
  and questions, prefix caching (several questions about one state reuse its KV cache), and
  ``allowed_token_ids`` so the returned log-probs are exactly the distribution over the
  question's answer tokens.
* ``HFBackend`` (portable): plain transformers, CPU or GPU, one question at a time. Slow, but
  needs nothing beyond torch and transformers; used for parity checks and small deployments.
* ``GGUFBackend``: llama.cpp on CPU, Metal or CUDA, one question at a time with
  checked tokenizer equivalence and the same candidate-logit scoring contract.

Tokenisation contract (identical to training): the chat-templated prompt is tokenised without
extra special tokens; with Granite's tokenizer the cue's trailing space merges into the answer
token (" A", " yes"), so the prompt is cut at the prefix it shares with prompt+answer and the
answer token is read at that position.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import math
import string
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .render import CUE, render, to_chat


def _cpl(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class Prepared:
    ids: list[int]          # prompt token ids, cut at the answer position
    cand_ids: list[int]     # one token per allowed answer, in presentation order
    keys: list[str]         # answer key per candidate (option key / level index / "1","0" for yes/no)
    qtype: str


class Readout:
    """Builds the model input for one (state, question) pair. Shared by all backends."""

    def __init__(self, tokenizer, max_prompt_tokens: int = 8192):
        self.tok = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        probe = "Allowed answers: A, B\n" + CUE
        base = self._enc(probe)
        ids = []
        for cand in itertools.chain(string.ascii_uppercase, ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))):
            if len(self._enc(probe + cand)) - _cpl(base, self._enc(probe + cand)) == 1:
                ids.append(cand)
            if len(ids) >= 255:
                break
        self.identifiers = ids
        self._tail_cache: dict = {}

    def _enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _cands(self, tail: str, cands: tuple[str, ...]):
        key = (tail, cands)
        if key not in self._tail_cache:
            a = self._enc(tail)
            outs = [self._enc(tail + c) for c in cands]
            cp = min(_cpl(a, o) for o in outs)
            if not all(len(o) == cp + 1 for o in outs):
                raise ValueError("answer identifiers are not single tokens for this tokenizer")
            self._tail_cache[key] = (len(a) - cp, [o[cp] for o in outs])
        return self._tail_cache[key]

    def prepare(self, state: Any, question: dict) -> Prepared:
        r = render(state, question, mode="index", identifiers=self.identifiers)
        text = to_chat(r.prefix, self.tok)
        ids = self._enc(text)
        drop, cids = self._cands(text[-60:], tuple(r.candidates))
        if drop:
            ids = ids[:-drop]
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt is {len(ids)} tokens; the limit is {self.max_prompt_tokens}")
        return Prepared(ids, cids, list(r.keys), question["type"])


def calibration_file(model: str | None) -> Path | None:
    """``docket_calibration.json`` for a local model folder or a Hugging Face repo id."""
    if not model:
        return None
    p = Path(model) / "docket_calibration.json"
    if p.exists() or Path(model).exists():
        return p if p.exists() else None
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # huggingface_hub < 0.23
        from huggingface_hub.utils import EntryNotFoundError
    try:
        return Path(hf_hub_download(model, "docket_calibration.json"))
    except EntryNotFoundError:
        return None


def load_calibration(model_dir: str | None) -> dict[str, float]:
    """Per-type temperatures shipped with the model (``docket_calibration.json``)."""
    p = calibration_file(model_dir)
    return json.loads(p.read_text())["temperatures"] if p else {}


def model_name(model_dir: str) -> str:
    """Name reported in responses: ``model_name`` from docket_calibration.json, else the folder name."""
    p = calibration_file(model_dir)
    if p:
        name = json.loads(p.read_text()).get("model_name")
        if name:
            return name
    return Path(str(model_dir).rstrip("/")).name.lower() or "docket"


def probs_from_logprobs(logps: list[float], temperature: float = 1.0) -> list[float]:
    z = [lp / temperature for lp in logps]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class VLLMBackend:
    def __init__(self, model: str, *, max_model_len: int = 8192, gpu_memory_utilization: float = 0.90,
                 max_num_seqs: int = 256, enable_prefix_caching: bool = True, dtype: str = "bfloat16", quantization: str | None = None,
                 cudagraph_max_tokens: int = 2048):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        args = AsyncEngineArgs(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                               max_num_seqs=max_num_seqs, enable_prefix_caching=enable_prefix_caching, max_logprobs=260,
                               logprobs_mode="processed_logprobs", quantization=quantization,
                               # Capture CUDA graphs for whole prefills up to this many tokens. Decisions are
                               # single-pass prefills of ~0.3-2K tokens, so the default 512 leaves most of them
                               # uncaptured; 2048 cut 250-token latency on an H100 from 13.7 to 8.7 ms.
                               compilation_config={"max_cudagraph_capture_size": cudagraph_max_tokens})
        self.engine = AsyncLLMEngine.from_engine_args(args)
        from transformers import AutoTokenizer
        self.readout = Readout(AutoTokenizer.from_pretrained(model), max_prompt_tokens=max_model_len - 1)

    async def score(self, prep: Prepared) -> list[float]:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=1, temperature=1.0, logprobs=len(prep.cand_ids), allowed_token_ids=list(dict.fromkeys(prep.cand_ids)))
        final = None
        async for out in self.engine.generate({"prompt_token_ids": prep.ids}, params, request_id=uuid.uuid4().hex):
            final = out
        lp = final.outputs[0].logprobs[0] if final and final.outputs[0].logprobs else {}
        return [float(lp[t].logprob) if t in lp else -1e4 for t in prep.cand_ids]


class HFBackend:
    def __init__(self, model: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.readout = Readout(AutoTokenizer.from_pretrained(model))
        self.lock = asyncio.Lock()

    async def score(self, prep: Prepared) -> list[float]:
        async with self.lock:
            with self.torch.no_grad():
                x = self.torch.tensor([prep.ids], device=self.device)
                logits = self.model(input_ids=x, logits_to_keep=1).logits[0, -1].float()
                lp = self.torch.log_softmax(logits[prep.cand_ids], -1)
                return lp.tolist()


GGUF_SIDECARS = ["*.gguf", "*.json", "*.jinja", "tokenizer*"]


def resolve_gguf(model: str) -> Path:
    """Locate the .gguf file for ``model``: a local .gguf file, a local release folder,
    or a Hugging Face repo id such as ``<org>/Docket-3B-GGUF``. A repo is downloaded with
    its tokenizer and calibration sidecars (not other repo files) into the HF cache."""
    path = Path(model)
    if not path.exists():
        if path.suffix.lower() == ".gguf":
            raise ValueError(f"GGUF file not found: {model}")
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model, allow_patterns=GGUF_SIDECARS))
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError("The gguf backend requires a .gguf file, a folder containing one, or a Hugging Face repo id.")
        return path
    found = sorted(path.glob("*.gguf"))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one .gguf file in {path}, found {len(found)}; pass the file path instead.")
    return found[0]


class GGUFBackend:
    """llama.cpp readout using the exact HF prompt, tokenizer and answer position.

    The GGUF release includes the original tokenizer and calibration beside the
    weights. Token IDs are checked against llama.cpp before any inference, so a
    tokenizer conversion mismatch fails instead of returning incorrect scores.
    """

    def __init__(self, model: str, *, tokenizer: str | None = None,
                 max_model_len: int = 8192, n_gpu_layers: int = -1,
                 n_threads: int | None = None):
        import os

        import numpy as np
        from llama_cpp import Llama
        # GGUF installs omit PyTorch on purpose; only the tokenizer is used from transformers.
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        from transformers import AutoTokenizer

        path = resolve_gguf(model)
        self.model_path = path
        tokenizer_path = tokenizer or str(path.parent)
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        options = {"n_threads": n_threads} if n_threads is not None else {}
        self.model = Llama(model_path=str(path), n_ctx=max_model_len,
                           n_gpu_layers=n_gpu_layers, logits_all=False,
                           verbose=False, **options)
        self.np = np
        self.lock = asyncio.Lock()
        llama = self.model

        class CheckedTokenizer:
            def __getattr__(self, name):
                return getattr(hf_tokenizer, name)

            def __call__(self, text, **kwargs):
                result = hf_tokenizer(text, **kwargs)
                native = llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)
                if result["input_ids"] != native:
                    raise ValueError("GGUF and HF tokenizer IDs differ; use the tokenizer "
                                     "bundled with this Docket GGUF release.")
                return result

        self.readout = Readout(CheckedTokenizer(), max_prompt_tokens=max_model_len - 1)

    def _score(self, prep: Prepared) -> list[float]:
        self.model.reset()
        self.model.eval(prep.ids)
        # With logits_all=False, current bindings leave the Python scores cache
        # unfilled. Read the last emitted token's logits from the native context.
        pointer = self.model._ctx.get_logits_ith(-1)
        if not pointer:
            raise ValueError("llama.cpp returned no answer logits")
        vocabulary_logits = self.np.ctypeslib.as_array(pointer, shape=(self.model.n_vocab(),))
        logits = self.np.asarray(vocabulary_logits[prep.cand_ids], dtype=self.np.float64)
        if not self.np.isfinite(logits).all():
            raise ValueError("llama.cpp returned non-finite answer logits")
        shifted = logits - logits.max()
        return (shifted - self.np.log(self.np.exp(shifted).sum())).tolist()

    async def score(self, prep: Prepared) -> list[float]:
        async with self.lock:
            return await asyncio.to_thread(self._score, prep)

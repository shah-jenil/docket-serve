"""Readout invariants for the new backend; real checkpoint parity is measured separately."""
import asyncio
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from docket_serve.engine import GGUF_SIDECARS, GGUFBackend, Prepared, load_calibration, model_name, probs_from_logprobs, resolve_gguf


def test_readout_only_uses_allowed_tokens():
    calls = []
    native = np.array([1000.0, 1.0, -100.0, 3.0], dtype=np.float32)
    model = SimpleNamespace(reset=lambda: calls.append("reset"), eval=lambda ids: calls.append(ids),
                            scores=np.zeros((2, 4), dtype=np.float32), n_vocab=lambda: 4,
                            _ctx=SimpleNamespace(get_logits_ith=lambda i: native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if i == -1 else None))
    backend = GGUFBackend.__new__(GGUFBackend)
    backend.np = np
    backend.model = model
    backend.lock = asyncio.Lock()
    prep = Prepared(ids=[17, 18], cand_ids=[3, 1], keys=["high", "low"], qtype="choice")
    logps = asyncio.run(backend.score(prep))
    probabilities = probs_from_logprobs(logps)
    assert probabilities == pytest.approx([1 / (1 + np.exp(-2)), 1 / (1 + np.exp(2))])
    assert calls == ["reset", [17, 18]]


def test_non_finite_logits_fail():
    native = np.array([np.nan, 1.0], dtype=np.float32)
    backend = GGUFBackend.__new__(GGUFBackend)
    backend.np = np
    backend.model = SimpleNamespace(reset=lambda: None, eval=lambda ids: None,
                                   n_vocab=lambda: 2,
                                   _ctx=SimpleNamespace(get_logits_ith=lambda i: native.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    with pytest.raises(ValueError, match="non-finite"):
        backend._score(Prepared(ids=[1], cand_ids=[0, 1], keys=["1", "0"], qtype="noul"))


def test_calibration_sidecar(tmp_path):
    (tmp_path / "docket_calibration.json").write_text('{"temperatures":{"choice":1.15,"noul":1.3,"score":1.1}}')
    assert load_calibration(str(tmp_path))["noul"] == 1.3
    probabilities = probs_from_logprobs([0.0, -4.0], temperature=1.3)
    assert sum(probabilities) == pytest.approx(1)
    assert probabilities[0] < probs_from_logprobs([0.0, -4.0])[0]


def test_resolve_gguf_local_file_and_folder(tmp_path):
    weights = tmp_path / "Docket-3B-Q8_0.gguf"
    weights.write_bytes(b"GGUF")
    assert resolve_gguf(str(weights)) == weights
    assert resolve_gguf(str(tmp_path)) == weights
    (tmp_path / "other.gguf").write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_gguf(str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        resolve_gguf(str(tmp_path / "missing.gguf"))


def test_resolve_gguf_downloads_repo_with_sidecars(tmp_path, monkeypatch):
    import huggingface_hub
    (tmp_path / "Docket-3B-Q8_0.gguf").write_bytes(b"GGUF")
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, allow_patterns: calls.append((repo, allow_patterns)) or str(tmp_path))
    assert resolve_gguf("org/Docket-3B-GGUF") == tmp_path / "Docket-3B-Q8_0.gguf"
    assert calls == [("org/Docket-3B-GGUF", GGUF_SIDECARS)]


def test_calibration_from_hub_repo_id(tmp_path, monkeypatch):
    import huggingface_hub
    sidecar = tmp_path / "docket_calibration.json"
    sidecar.write_text('{"model_name":"docket-3b","temperatures":{"choice":1.15,"noul":1.3,"score":1.1}}')
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(sidecar))
    assert load_calibration("org/Docket-3B")["choice"] == 1.15
    assert model_name("org/Docket-3B") == "docket-3b"
    local = tmp_path / "local-model"
    local.mkdir()
    assert load_calibration(str(local)) == {}   # an existing local folder never falls back to the Hub


def test_default_model_and_backend(monkeypatch):
    from docket_serve import server
    installed = set()
    monkeypatch.setattr(server, "_installed", lambda module: module in installed)
    installed.add("llama_cpp")   # a requirements.txt install
    assert server.resolve_model_and_backend(None, None) == ("shahjenil/Docket-3B-GGUF", "gguf")
    assert server.resolve_model_and_backend("/models/Docket-3B-Q8_0.gguf", None) == ("/models/Docket-3B-Q8_0.gguf", "gguf")
    assert server.resolve_model_and_backend("org/Docket-3B", None) == ("org/Docket-3B", "hf")
    installed.add("vllm")
    assert server.resolve_model_and_backend(None, None) == ("shahjenil/Docket-3B", "vllm")
    assert server.resolve_model_and_backend("/models/docket", None) == ("/models/docket", "vllm")
    assert server.resolve_model_and_backend(None, "gguf") == ("shahjenil/Docket-3B-GGUF", "gguf")

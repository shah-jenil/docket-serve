"""scripts/smoke_test.py against the real server app, with a fake model backend."""
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

uvicorn = pytest.importorskip("uvicorn")
from docket_serve.engine import Prepared  # noqa: E402
from docket_serve.server import Service, build_app  # noqa: E402

SMOKE = Path(__file__).resolve().parents[1] / "scripts/smoke_test.py"


class FakeReadout:
    def prepare(self, state, q):
        keys = list(q["criteria"]) if q["type"] == "choice" else (
            ["1", "0"] if q["type"] == "noul" else [str(i) for i in range(len(q["criteria"]))])
        return Prepared(ids=[1, 2, 3], cand_ids=list(range(len(keys))), keys=keys, qtype=q["type"])


class FakeBackend:
    readout = FakeReadout()

    async def score(self, prep):
        return [-float(i) for i in range(len(prep.cand_ids))]   # first option most likely


def serve(temps):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(build_app(Service(FakeBackend(), "docket-test", temps)),
                                           host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server, port


def smoke(port):
    return subprocess.run([sys.executable, str(SMOKE), "--url", f"http://127.0.0.1:{port}", "--repeat", "2", "--wait", "10"],
                          capture_output=True, text=True)


def test_smoke_passes_on_a_calibrated_server():
    server, port = serve({"choice": 1.15, "noul": 1.3, "score": 1.1})
    try:
        result = smoke(port)
    finally:
        server.should_exit = True
    assert result.returncode == 0, result.stdout + result.stderr
    assert "choice: billing" in result.stdout and "PASS" in result.stdout


def test_smoke_fails_without_calibration():
    server, port = serve({})
    try:
        result = smoke(port)
    finally:
        server.should_exit = True
    assert result.returncode != 0 and "not calibrated" in result.stderr

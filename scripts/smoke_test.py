#!/usr/bin/env python3
"""Check a running docket-serve end to end: health, calibration, one choice, yes/no and score
question, answer shapes, latency and (if the server runs on this machine) its memory.

    python3 scripts/smoke_test.py [--url http://127.0.0.1:8080] [--repeat 5] [--wait 600]

Uses only the Python standard library. Exits non-zero on any failed check.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

REQUEST = {
    "state": "Customer email: Hi, I was charged twice for order #4471 this month. Please fix this today.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this ticket?",
                  "criteria": {"billing": "payments, refunds, invoices", "shipping": "delivery, tracking",
                               "technical": "bugs, login problems"}},
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply today?",
                   "criteria": {"true": "money lost, service down, or a deadline", "false": "general questions"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["calm", "mildly annoyed", "frustrated", "furious"]},
    },
}


def call(url, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_rss_mb(port):
    """Resident memory of a local docket-serve listening on ``port`` (macOS and Linux), else None."""
    try:
        out = subprocess.run(["ps", "-ax", "-o", "rss=,command="], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    found = []
    for line in out.splitlines():
        rss, _, command = line.strip().partition(" ")
        listens = f"--port {port}" in command or ("--port" not in command and port == 8080)
        if ("bin/docket-serve" in command or "docket_serve" in command) and listens:
            found.append(int(rss) / 1024)
    return max(found, default=None)   # the server, not a shell that launched it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--repeat", type=int, default=5, help="timed requests after one warm-up")
    ap.add_argument("--wait", type=int, default=600, help="seconds to wait for the server (first start downloads the model)")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    failures = []

    deadline = time.time() + a.wait
    while True:
        try:
            health = call(base + "/healthz", timeout=5)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.time() > deadline:
                sys.exit(f"FAIL: no server at {base} after {a.wait} s")
            time.sleep(3)
    print(f"server: model={health.get('model')} calibrated={health.get('calibrated')}")
    if health.get("calibrated") is False:
        failures.append("server is not calibrated: docket_calibration.json was not found")

    latencies = []
    for i in range(a.repeat + 1):
        t0 = time.perf_counter()
        resp = call(base + "/v1/systemone", REQUEST)
        if i:
            latencies.append((time.perf_counter() - t0) * 1000)
    answers = resp.get("answers", {})

    route = answers.get("route", {})
    probs = route.get("probabilities", {})
    if set(probs) != {"billing", "shipping", "technical"} or abs(sum(probs.values()) - 1) > 1e-3:
        failures.append(f"choice probabilities malformed: {probs}")
    elif route.get("choice") != max(probs, key=probs.get):
        failures.append("choice is not the most probable option")
    print(f"choice: {route.get('choice')} (confidence {route.get('confidence')}) {probs}")

    noul = answers.get("urgent", {}).get("noul")
    if not isinstance(noul, (int, float)) or not 0 <= noul <= 1:
        failures.append(f"noul is not a probability: {noul}")
    print(f"noul:   P(yes) = {noul}")

    score = answers.get("frustration", {})
    levels = score.get("probabilities", {})
    if len(levels) != 4 or abs(sum(levels.values()) - 1) > 1e-3 or not 0 <= score.get("score", -1) <= 3:
        failures.append(f"score malformed: {score}")
    print(f"score:  {score.get('score')} (confidence {score.get('confidence')}) {levels}")

    if resp.get("usage", {}).get("output_tokens") != 3:
        failures.append(f"usage.output_tokens should be 3: {resp.get('usage')}")
    if latencies:
        print(f"latency, 3 questions per request: p50 {statistics.median(latencies):.0f} ms, "
              f"max {max(latencies):.0f} ms over {len(latencies)} requests "
              f"({resp['usage']['input_tokens']} input tokens)")
    port = base.rsplit(":", 1)[-1] if base.count(":") > 1 else "8080"
    rss = server_rss_mb(port)
    if rss:
        print(f"server memory (resident): {rss:,.0f} MB")

    if failures:
        sys.exit("FAIL:\n  " + "\n  ".join(failures))
    print("PASS")


if __name__ == "__main__":
    main()

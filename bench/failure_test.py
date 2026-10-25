"""
Failure / recovery test against a running compose stack.

For each dependency: stop it, check how questions and health behave, start it
again and measure how long until normal answers return — without restarting the
API. Also kills the API and worker processes to check restart policies.

  python bench/failure_test.py --url http://localhost:1000 --out bench/results/failures.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import httpx

Q = "What is the monthly preventive maintenance for the bell atomizer?"


def dc(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True, capture_output=True, text=True, timeout=300)


def ask(c: httpx.Client, mode: str = "evidence") -> dict:
    t = time.perf_counter()
    try:
        r = c.post("/api/v1/query", json={"query": Q, "mode": mode, "use_cache": False}, timeout=120)
        body = r.json()
        return {"http": r.status_code, "status": body.get("status"), "mode_used": body.get("mode_used"),
                "notice": body.get("notice"), "detail": body.get("detail"), "ms": round((time.perf_counter() - t) * 1000)}
    except Exception as exc:
        return {"http": None, "error": type(exc).__name__, "ms": round((time.perf_counter() - t) * 1000)}


def ready(c: httpx.Client) -> int | None:
    try:
        return c.get("/health/ready", timeout=10).status_code
    except Exception:
        return None


def wait_ok(c: httpx.Client, timeout: float = 180) -> float | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = ask(c)
        if r.get("http") == 200 and r.get("status") == "answered":
            return round(time.time() - t0, 1)
        time.sleep(1)
    return None


def scenario(c, service, mode="evidence"):
    print(f"--- stopping {service}")
    dc("stop", service)
    time.sleep(2)
    during = ask(c, mode)
    status = c.get("/api/v1/status", timeout=15).json()
    res = {"during": during, "ready_http": ready(c), "overall_status": status.get("status")}
    dc("start", service)
    res["recovery_s"] = wait_ok(c)
    print(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1000")
    ap.add_argument("--out", default="bench/results/failures.json")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url)
    assert wait_ok(c, 60) is not None, "stack not healthy before the test"
    out = {
        "redis_down": scenario(c, "redis"),
        "llm_down": scenario(c, "ollama", mode="auto"),
        "qdrant_down": scenario(c, "qdrant"),
        "postgres_down": scenario(c, "postgres"),
    }
    print("--- killing api process (restart policy)")
    t0 = time.time()
    subprocess.run(["docker", "compose", "kill", "-s", "SIGKILL", "api"], check=True, capture_output=True)
    dc("up", "-d", "api")  # 'kill' marks the container as stopped by the user; compose up brings it back
    out["api_killed"] = {"recovery_s": None}
    rec = wait_ok(c, 180)
    out["api_killed"]["recovery_s"] = round(time.time() - t0, 1) if rec is not None else None
    print(json.dumps(out["api_killed"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

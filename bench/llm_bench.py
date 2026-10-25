"""
CPU LLM benchmark via Ollama's own timing counters.

Compares models and prompt sizes that matter for latency:
  * old design: llama3.1:8b, ~7k-token prompt (12 parent chunks of 2.4k chars + long rules)
  * new design: qwen2.5:3b, ~1.5k-token prompt (≤6k chars of evidence + short rules)

  python bench/llm_bench.py --url http://localhost:11434 --models qwen2.5:3b,llama3.1:8b --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import httpx

EVIDENCE = (
    "Alarm PNT1-128 - Air turbine speed deviation. Severity: Stop. Description: Actual speed deviated from the "
    "setpoint beyond tolerance on the air turbine. Possible causes: worn turbine or motor bearing; low supply air "
    "pressure; contaminated speed feedback sensor. Corrective action procedure: 1. Measure supply air pressure at the "
    "regulator; nominal is 6 bar. 2. Inspect bearing condition and replace if noise or play is detected. 3. Clean the "
    "speed feedback fiber optic sensor and re-check the signal level. Related alarms: SRVO-822, PLC-2551. "
)


def prompt(chars: int) -> str:
    body = (EVIDENCE * (chars // len(EVIDENCE) + 1))[:chars]
    return f"SOURCES:\n[1] (alarm_manual.md | Alarm PNT1-128)\n{body}\n\nQUESTION: Robot R07 stopped with PNT1-128, what should I check first?\nAnswer with citations:"


def run(url: str, model: str, num_ctx: int, chars: int, max_tokens: int) -> dict:
    body = {"model": model, "stream": False, "keep_alive": "10m",
            "messages": [{"role": "system", "content": "Answer only from the sources and cite them as [n]."},
                         {"role": "user", "content": prompt(chars)}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": max_tokens}}
    t = time.perf_counter()
    r = httpx.post(f"{url}/api/chat", json=body, timeout=1800).json()
    wall = time.perf_counter() - t
    if "error" in r:
        return {"error": r["error"]}
    pe_s = r.get("prompt_eval_duration", 0) / 1e9
    ev_s = r.get("eval_duration", 0) / 1e9
    return {
        "wall_s": round(wall, 2),
        "load_s": round(r.get("load_duration", 0) / 1e9, 2),
        "prompt_tokens": r.get("prompt_eval_count"),
        "prompt_tok_s": round(r.get("prompt_eval_count", 0) / pe_s, 1) if pe_s else None,
        "time_to_first_token_s": round(pe_s + r.get("load_duration", 0) / 1e9, 2),
        "gen_tokens": r.get("eval_count"),
        "gen_tok_s": round(r.get("eval_count", 0) / ev_s, 1) if ev_s else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--models", default="qwen2.5:3b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="bench/results/llm.json")
    args = ap.parse_args()
    scenarios = [
        ("3k chars evidence, ctx 4096", 3000, 4096, 200),
        ("6k chars evidence, ctx 4096", 6000, 4096, 200),
        ("old-style 30k chars context, ctx 16384", 30000, 16384, 200),
    ]
    results = {}
    for model in args.models.split(","):
        for label, chars, ctx, max_tok in scenarios:
            print(f"{model} | {label}: warming up…")
            run(args.url, model, ctx, 200, 5)  # load model, excluded from stats
            rows = [run(args.url, model, ctx, chars, max_tok) for _ in range(args.runs)]
            ok = [r for r in rows if "error" not in r]
            agg = {k: round(statistics.median(r[k] for r in ok), 2) for k in ok[0] if isinstance(ok[0][k], (int, float))} if ok else {}
            results[f"{model} | {label}"] = {"median": agg, "runs": rows}
            print(f"  -> {agg}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

"""
End-to-end benchmark against a running stack (docker compose up).

  python bench/e2e_bench.py --url http://localhost:1000 --data bench/data \
      [--upload] [--llm-sample 20] [--concurrency 1,4,8,16] --out bench/results/e2e.json

Measures, through the public HTTP API only:
  * ingestion: upload the corpus, wait until indexed, chunks/s
  * retrieval accuracy: hit@1 / hit@5 / answer@5 / MRR per category (/api/v1/search)
  * evidence answers (/api/v1/query mode=evidence): latency p50/p95/p99,
    correct refusals on out-of-scope questions, false refusals on in-scope ones
  * LLM answers on a sample (mode=answer, streamed): time-to-evidence,
    time-to-first-token, total time, verification confidence
  * concurrency: throughput and latency at N parallel users
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import time
from collections import Counter, defaultdict

import httpx


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return round(s[k], 1)


def summary(values):
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "mean": round(statistics.mean(values), 1) if values else None}


def upload_corpus(client, data_dir):
    files = [f for f in sorted(os.listdir(data_dir)) if f.endswith((".md", ".csv"))]
    t0 = time.time()
    for name in files:
        with open(os.path.join(data_dir, name), "rb") as fh:
            r = client.post("/api/v1/upload", files={"file": (name, fh)}, data={"collection": "bench"}, timeout=300)
            r.raise_for_status()
    print(f"uploaded {len(files)} files in {time.time() - t0:.1f}s; waiting for indexing…")
    while True:
        docs = client.get("/api/v1/documents", params={"collection": "bench", "limit": 100}).json()["documents"]
        states = Counter(d["status"] for d in docs)
        if len(docs) >= len(files) and states.get("indexed", 0) + states.get("failed", 0) == len(docs):
            break
        time.sleep(2)
    elapsed = time.time() - t0
    chunks = sum(d["chunk_count"] for d in docs)
    print(f"indexed {chunks} chunks from {len(docs)} files in {elapsed:.1f}s ({chunks / elapsed:.1f} chunks/s); states={dict(states)}")
    return {"files": len(docs), "chunks": chunks, "seconds": round(elapsed, 1), "chunks_per_s": round(chunks / elapsed, 1),
            "failed": states.get("failed", 0)}


def retrieval_eval(client, queries, rerank=True):
    per = defaultdict(Counter)
    lat = []
    stage = defaultdict(list)
    for q in queries:
        if q["expect"] is None:
            continue
        t = time.perf_counter()
        r = client.post("/api/v1/search", json={"query": q["query"], "top_k": 10, "rerank": rerank}).json()
        lat.append((time.perf_counter() - t) * 1000)
        for k, v in r["timings"].items():
            stage[k].append(v)
        texts = [f"{h.get('section_title') or ''}\n{h['text']}" for h in r["hits"]]
        rank = next((i for i, t in enumerate(texts, 1) if q["expect"] in t), None)
        c = per[q["category"]]
        c["n"] += 1
        if rank == 1:
            c["hit1"] += 1
        if rank and rank <= 5:
            c["hit5"] += 1
            if q.get("answer") is None or q["answer"] in "\n".join(texts[:5]):
                c["ans5"] += 1
        if rank:
            c["mrr"] += 1 / rank
    out, tot = {}, Counter()
    for cat, c in sorted(per.items()):
        out[cat] = {k: round(c[k] / c["n"], 3) for k in ("hit1", "hit5", "ans5", "mrr")}
        tot.update(c)
    out["ALL"] = {k: round(tot[k] / tot["n"], 3) for k in ("hit1", "hit5", "ans5", "mrr")}
    for cat, r in out.items():
        print(f"  {cat:18s} hit@1={r['hit1']:.3f} hit@5={r['hit5']:.3f} answer@5={r['ans5']:.3f} mrr={r['mrr']:.3f}")
    return {"metrics": out, "http_latency_ms": summary(lat), "stage_ms": {k: summary(v) for k, v in stage.items()}}


def evidence_eval(client, queries):
    lat, server = [], []
    refusals = Counter()
    for q in queries:
        t = time.perf_counter()
        r = client.post("/api/v1/query", json={"query": q["query"], "mode": "evidence", "use_cache": False}).json()
        lat.append((time.perf_counter() - t) * 1000)
        server.append(r["timings"]["total_ms"])
        neg = q["expect"] is None
        refused = r["status"] in ("not_found", "partial")  # partial = shown with a "closest match" warning
        refusals[("neg" if neg else "pos", refused)] += 1
        refusals[("partial", neg)] += r["status"] == "partial"
    neg_total = refusals[("neg", True)] + refusals[("neg", False)]
    pos_total = refusals[("pos", True)] + refusals[("pos", False)]
    res = {
        "http_latency_ms": summary(lat), "server_latency_ms": summary(server),
        "correct_refusal_rate": round(refusals[("neg", True)] / neg_total, 3) if neg_total else None,
        "false_refusal_rate": round(refusals[("pos", True)] / pos_total, 3) if pos_total else None,
        "flagged_partial_in_scope": refusals[("partial", False)], "flagged_partial_out_of_scope": refusals[("partial", True)],
    }
    print(f"  evidence: {res}")
    return res


def cache_eval(client, queries):
    lat = []
    for q in queries[:50]:
        client.post("/api/v1/query", json={"query": q["query"], "mode": "evidence"})
        t = time.perf_counter()
        client.post("/api/v1/query", json={"query": q["query"], "mode": "evidence"})
        lat.append((time.perf_counter() - t) * 1000)
    res = summary(lat)
    print(f"  cached repeat latency: {res}")
    return res


def llm_eval(client, queries, n):
    sample = [q for q in queries if q["expect"] is not None][:: max(1, len(queries) // max(n, 1))][:n]
    rows = []
    for q in sample:
        t0 = time.perf_counter()
        t_evidence = t_first = None
        final = {}
        with client.stream("POST", "/api/v1/query", json={"query": q["query"], "mode": "answer", "stream": True,
                                                         "use_cache": False}, timeout=600) as resp:
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    raw, buf = buf.split("\n\n", 1)
                    if not raw.startswith("data:"):
                        continue
                    ev = json.loads(raw[5:])
                    now = (time.perf_counter() - t0) * 1000
                    if ev["type"] == "retrieval" and t_evidence is None:
                        t_evidence = now
                    elif ev["type"] == "token" and t_first is None:
                        t_first = now
                    elif ev["type"] == "done":
                        final = ev
        total = (time.perf_counter() - t0) * 1000
        answer = final.get("answer", "")
        rows.append({
            "category": q["category"], "evidence_ms": t_evidence, "first_token_ms": t_first, "total_ms": total,
            "mode_used": final.get("mode_used"), "status": final.get("status"),
            "confidence": (final.get("verification") or {}).get("confidence"),
            "expect_in_answer": q["expect"] in answer, "answer_in_answer": (q.get("answer") or "")[:40] in answer,
        })
        print(f"  [{q['category']}] evidence={t_evidence:.0f}ms first_token={t_first or 0:.0f}ms total={total:.0f}ms "
              f"conf={rows[-1]['confidence']} expect_in_answer={rows[-1]['expect_in_answer']}")
    res = {
        "n": len(rows),
        "evidence_ms": summary([r["evidence_ms"] for r in rows if r["evidence_ms"]]),
        "first_token_ms": summary([r["first_token_ms"] for r in rows if r["first_token_ms"]]),
        "total_ms": summary([r["total_ms"] for r in rows]),
        "confidence": dict(Counter(r["confidence"] for r in rows)),
        "llm_used": sum(r["mode_used"] == "llm" for r in rows),
        "expect_in_answer_rate": round(sum(r["expect_in_answer"] for r in rows) / max(1, len(rows)), 3),
        "rows": rows,
    }
    return res


def concurrency_eval(base_url, queries, levels, headers):
    out = {}
    pool = [q["query"] for q in queries]
    for n in levels:
        lat, errors = [], 0
        total_requests = max(40, n * 10)

        def one(i):
            with httpx.Client(base_url=base_url, timeout=60, headers=headers) as c:
                t = time.perf_counter()
                r = c.post("/api/v1/query", json={"query": pool[i % len(pool)], "mode": "evidence", "use_cache": False})
                return (time.perf_counter() - t) * 1000, r.status_code

        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            for ms, code in ex.map(one, range(total_requests)):
                lat.append(ms)
                errors += code != 200
        wall = time.perf_counter() - t0
        out[str(n)] = {"requests": total_requests, "errors": errors, "throughput_rps": round(total_requests / wall, 2),
                       "latency_ms": summary(lat)}
        print(f"  concurrency={n}: {out[str(n)]}")
    return out


def docker_stats(project="industrial-rag"):
    """Per-container memory/CPU snapshot of the compose project (best effort)."""
    import subprocess
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception as exc:  # docker CLI not available
        return {"error": str(exc)}
    rows = {}
    for line in out.splitlines():
        d = json.loads(line)
        if d.get("Name", "").startswith(project):
            rows[d["Name"]] = {"mem": d.get("MemUsage"), "cpu": d.get("CPUPerc")}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1000")
    ap.add_argument("--data", default="bench/data")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--llm-sample", type=int, default=0)
    ap.add_argument("--concurrency", default="")
    ap.add_argument("--api-key", default=os.getenv("RAG_API_KEY", ""))
    ap.add_argument("--out", default="bench/results/e2e.json")
    ap.add_argument("--skip", default="", help="comma list: retrieval,evidence,cache")
    args = ap.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    client = httpx.Client(base_url=args.url, timeout=120, headers=headers)
    queries = [json.loads(line) for line in open(os.path.join(args.data, "queries.jsonl"))]
    skip = set(filter(None, args.skip.split(",")))
    results = {"url": args.url, "status": client.get("/api/v1/status").json()}
    if args.upload:
        results["ingestion"] = upload_corpus(client, args.data)
    if "retrieval" not in skip:
        print("retrieval accuracy (with reranker):")
        results["retrieval"] = retrieval_eval(client, queries, rerank=True)
        print("retrieval accuracy (without reranker):")
        results["retrieval_no_rerank"] = retrieval_eval(client, queries, rerank=False)
    if "evidence" not in skip:
        print("evidence answers:")
        results["evidence"] = evidence_eval(client, queries)
    if "cache" not in skip:
        results["cache"] = cache_eval(client, queries)
    if args.llm_sample:
        print("LLM answers:")
        results["llm"] = llm_eval(client, queries, args.llm_sample)
    if args.concurrency:
        print("concurrency:")
        results["concurrency"] = concurrency_eval(args.url, queries, [int(x) for x in args.concurrency.split(",")], headers)
    results["docker_stats"] = docker_stats()
    print("resources:", json.dumps(results["docker_stats"], indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

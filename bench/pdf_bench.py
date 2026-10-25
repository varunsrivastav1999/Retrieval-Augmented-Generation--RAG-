"""
Word-for-word accuracy on the PDF/DOCX maintenance library (bench/pdf_corpus.py).

  python bench/pdf_bench.py --url http://localhost:1000 --data bench/data_pdf --upload [--llm-sample 10]

For every question (mode=evidence, i.e. the exact-answer path, no LLM):
  * exact_verbatim : the expected sentence/step/value appears word for word in the exact answer
  * right_document : the exact answer is quoted from the expected document
  * in_evidence    : the expected text is somewhere in the returned evidence passages
  * full procedure : every step present, in the documented order
  * negatives      : out-of-scope questions answered "not in the documents"
With --llm-sample, AI answers are checked for verbatim content and verification confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict

import httpx


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace(" ", " ")).strip()


def upload(client, data):
    files = sorted(f for f in os.listdir(data) if f.endswith((".pdf", ".docx")))
    t0 = time.time()
    for name in files:
        with open(os.path.join(data, name), "rb") as fh:
            client.post("/api/v1/upload", files={"file": (name, fh)}, data={"collection": "manuals"}, timeout=120).raise_for_status()
    while True:
        docs = client.get("/api/v1/documents", params={"collection": "manuals", "limit": 200}).json()["documents"]
        st = Counter(d["status"] for d in docs)
        if len(docs) >= len(files) and st.get("indexed", 0) + st.get("failed", 0) == len(docs):
            break
        time.sleep(2)
    secs = time.time() - t0
    pages = sum(d["pages"] for d in docs)
    print(f"indexed {len(docs)} files / {pages} pages / {sum(d['chunk_count'] for d in docs)} chunks in {secs:.0f}s; {dict(st)}")
    return {"files": len(docs), "pages": pages, "chunks": sum(d["chunk_count"] for d in docs), "seconds": round(secs, 1),
            "pages_per_minute": round(pages / secs * 60, 1), "failed": st.get("failed", 0)}


def pct(v, p):
    s = sorted(v)
    return round(s[max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))], 1) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1000")
    ap.add_argument("--data", default="bench/data_pdf")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--llm-sample", type=int, default=0)
    ap.add_argument("--out", default="bench/results/pdf_verbatim.json")
    ap.add_argument("--show-failures", type=int, default=8)
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url, timeout=300)
    qs = [json.loads(l) for l in open(os.path.join(args.data, "questions.jsonl"))]
    out = {}
    if args.upload:
        out["ingestion"] = upload(c, args.data)

    per = defaultdict(Counter)
    lat, failures = [], []
    for q in qs:
        t = time.perf_counter()
        r = c.post("/api/v1/query", json={"query": q["query"], "mode": "evidence", "use_cache": False}).json()
        lat.append((time.perf_counter() - t) * 1000)
        cat = q["category"]
        per[cat]["n"] += 1
        if q["verbatim"] is None:
            per[cat]["refused"] += r["status"] == "not_found"
            per[cat]["flagged"] += r["status"] == "partial"
            if r["status"] not in ("not_found", "partial"):
                failures.append({"q": q["query"], "want": "not_found", "got": (r.get("exact") or [{}])[0].get("text", "")[:200]})
            continue
        exact = " ".join(norm(x["text"]) for x in (r.get("exact") or []))
        evidence = " ".join(norm(s["text"]) for s in r.get("sources", []))
        want = norm(q["verbatim"])
        ok = (want in exact or any(norm(a) in exact for a in q.get("alternatives", []))) and all(norm(a) in exact for a in q.get("also", []))
        if q.get("all_steps"):
            pos = [exact.find(norm(f"{i}. {s}")) for i, s in enumerate(q["all_steps"], 1)]
            per[cat]["all_steps_in_order"] += all(p >= 0 for p in pos) and pos == sorted(pos)
        per[cat]["exact_verbatim"] += ok
        per[cat]["in_evidence"] += want in evidence
        docs = {x["document"] for x in (r.get("exact") or [])}
        alt_doc = q["doc"].replace("_maintenance_manual.pdf", "_work_instructions.docx")
        per[cat]["right_document"] += q["doc"] in docs or (bool(q.get("alternatives")) and alt_doc in docs)
        per[cat]["answered"] += r["status"] in ("answered", "partial")
        per[cat]["flagged"] += r["status"] == "partial"
        if not ok and len(failures) < args.show_failures:
            failures.append({"q": q["query"], "want": q["verbatim"], "got": exact[:300], "docs": list(docs)})

    table, tot = {}, Counter()
    for cat, cn in sorted(per.items()):
        n = cn["n"]
        if cat == "negative":
            table[cat] = {"n": n, "refused": round(cn["refused"] / n, 3), "flagged_closest_match": round(cn["flagged"] / n, 3),
                          "answered_without_warning": round(1 - (cn["refused"] + cn["flagged"]) / n, 3)}
            continue
        row = {"n": n, "exact_verbatim": round(cn["exact_verbatim"] / n, 3), "right_document": round(cn["right_document"] / n, 3),
               "in_evidence": round(cn["in_evidence"] / n, 3)}
        if "all_steps_in_order" in cn or cat == "procedure_full":
            row["all_steps_in_order"] = round(cn["all_steps_in_order"] / n, 3)
        table[cat] = row
        tot.update({k: cn[k] for k in ("n", "exact_verbatim", "right_document", "in_evidence", "answered", "flagged")})
    table["ALL (answerable)"] = {"n": tot["n"], "exact_verbatim": round(tot["exact_verbatim"] / tot["n"], 3),
                                 "right_document": round(tot["right_document"] / tot["n"], 3),
                                 "in_evidence": round(tot["in_evidence"] / tot["n"], 3),
                                 "false_refusal": round(1 - tot["answered"] / tot["n"], 3),
                                 "flagged_closest_match": round(tot["flagged"] / tot["n"], 3)}
    for cat, row in table.items():
        print(f"  {cat:22s} {row}")
    out.update({"metrics": table, "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99),
                                                  "mean": round(statistics.mean(lat), 1)}, "failures": failures})
    print("latency", out["latency_ms"])
    for f in failures:
        print("  MISS:", f)

    if args.llm_sample:
        rows = []
        sample = [q for q in qs if q["verbatim"]][:: max(1, len(qs) // args.llm_sample)][: args.llm_sample]
        for q in sample:
            t = time.perf_counter()
            r = c.post("/api/v1/query", json={"query": q["query"], "mode": "answer", "use_cache": False}, timeout=600).json()
            rows.append({"category": q["category"], "seconds": round(time.perf_counter() - t, 1), "mode_used": r.get("mode_used"),
                         "verbatim_in_ai_answer": norm(q["verbatim"]) in norm(r.get("answer", "")),
                         "verbatim_in_exact": norm(q["verbatim"]) in " ".join(norm(x["text"]) for x in r.get("exact") or []),
                         "confidence": (r.get("verification") or {}).get("confidence")})
            print("  LLM:", rows[-1])
        out["llm"] = {"rows": rows, "verbatim_in_ai_answer": round(sum(r["verbatim_in_ai_answer"] for r in rows) / len(rows), 3),
                      "confidence": dict(Counter(r["confidence"] for r in rows))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

"""
Component-level retrieval benchmark: current (baseline) pipeline vs candidates.

Runs CPU-only inside a container (see bench/Dockerfile) so numbers reflect a
low-spec industrial PC. Everything is computed on the SAME chunks so the
comparison isolates retrieval components:

  baseline_fast  : exact catalogue SQL + dense-only (bge-large, no query
                   instruction)  -> what "maintenance safe mode" ran for most
                   maintenance questions
  baseline_full  : dense + HyDE-less Postgres plainto_tsquery BM25, with the
                   original int/str id mismatch in fusion, + bge-reranker-v2-m3
  cand_*         : dense (fastembed) + BM25 sparse (Qdrant/bm25 semantics, IDF)
                   + exact identifier match, RRF fusion, optional cross-encoder

Metrics per category: hit@1, hit@5 (marker string in a top-k chunk),
answer@5 (marker AND expected answer text present in top-5 context), MRR@10,
plus per-query latency of each stage.

Usage (inside bench container):
  python bench/retrieval_bench.py --data bench/data --baseline-app /baseline \
      --pg postgresql://bench:bench@bench-pg:5432/bench --out bench/results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root -> app.rag.identifiers

BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Baseline chunking (imports the ORIGINAL code from the snapshot)
# ---------------------------------------------------------------------------
def baseline_chunks(data_dir: str, baseline_root: str) -> list:
    sys.path.insert(0, baseline_root)
    os.environ.setdefault("DATABASE_URL", "postgresql://x:x@localhost:1/x")
    from app.rag.parsers import parse_file  # noqa: E402  (baseline version)
    from app.rag.ingestion import (  # noqa: E402
        _strip_table_text_from_raw, chunk_table_per_row, semantic_chunking,
    )
    try:
        from app.rag.table_engine import chunk_rich_table
    except Exception:  # pragma: no cover
        chunk_rich_table = None

    chunks = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith((".md", ".csv")):
            continue
        path = os.path.join(data_dir, name)
        res = parse_file(path)
        for page in res.pages:
            pcs = []
            cleaned = _strip_table_text_from_raw(page.text, page.tables) if page.tables else page.text
            if cleaned and cleaned.strip():
                pcs.extend(semantic_chunking(cleaned))
            if chunk_rich_table and getattr(page, "rich_tables", None):
                for rt in page.rich_tables:
                    pcs.extend(chunk_rich_table(rt, name, max_adjacent_rows=1))
            else:
                for ti, tmd in enumerate(page.tables):
                    for part in chunk_table_per_row(tmd, f"{name}_p{page.page_num}_t{ti}"):
                        pcs.append({"text": part["text"], "table_group": part["table_group"]})
            for pc in pcs:
                raw = pc["text"]
                if not raw.strip():
                    continue
                embed_text = f"[{name}, Page {page.page_num}]\n{raw}"
                # NeMo swap: retrieval returns the parent text when present.
                shown = pc.get("parent_text") or embed_text
                chunks.append({"id": len(chunks) + 1, "doc": name, "embed": embed_text, "text": shown})
    sys.path.remove(baseline_root)
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    return chunks


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class STEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.m = SentenceTransformer(name, device="cpu")

    def docs(self, texts):
        return np.asarray(self.m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def query(self, q):  # baseline: raw query, no BGE instruction
        return np.asarray(self.m.encode(q, normalize_embeddings=True), dtype=np.float32)


class FEEmbedder:
    def __init__(self, name, instruction=True):
        from fastembed import TextEmbedding
        self.name = name
        self.instruction = instruction
        self.m = TextEmbedding(name, threads=int(os.getenv("BENCH_THREADS", "4")))

    def docs(self, texts):
        v = np.asarray(list(self.m.embed(texts, batch_size=64)), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def query(self, q):
        text = (BGE_INSTRUCTION + q) if self.instruction else q
        v = np.asarray(next(iter(self.m.embed([text]))), dtype=np.float32)
        return v / np.linalg.norm(v)


class STReranker:
    def __init__(self, name):
        from sentence_transformers import CrossEncoder
        self.name = name
        self.m = CrossEncoder(name, device="cpu", max_length=512)

    def scores(self, q, docs):
        return list(self.m.predict([[q, d] for d in docs], batch_size=16))


class FEReranker:
    def __init__(self, name):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.name = name
        self.m = TextCrossEncoder(name, threads=int(os.getenv("BENCH_THREADS", "4")))

    def scores(self, q, docs):
        return list(self.m.rerank(q, docs, batch_size=32))


class SparseBM25:
    """Qdrant/bm25 sparse vectors scored like Qdrant does with modifier=idf."""

    def __init__(self, texts):
        from fastembed import SparseTextEmbedding
        self.m = SparseTextEmbedding("Qdrant/bm25")
        self.docs = list(self.m.embed(texts, batch_size=256))
        self.postings = defaultdict(list)
        for i, sv in enumerate(self.docs):
            for idx, val in zip(sv.indices.tolist(), sv.values.tolist()):
                self.postings[idx].append((i, val))
        n = len(self.docs)
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in self.postings.items()}

    def search(self, q, k):
        qv = next(iter(self.m.query_embed(q)))
        acc = defaultdict(float)
        for idx in qv.indices.tolist():
            w = self.idf.get(idx)
            if w is None:
                continue
            for doc_i, val in self.postings[idx]:
                acc[doc_i] += w * val
        return sorted(acc.items(), key=lambda kv: -kv[1])[:k]


# ---------------------------------------------------------------------------
# Baseline Postgres FTS (exact SQL from app/rag/retrieval.py)
# ---------------------------------------------------------------------------
class PgFTS:
    def __init__(self, dsn, chunks):
        import psycopg2
        import psycopg2.extras
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn, keepalives=1, keepalives_idle=30)
        self.conn.autocommit = True
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS bench_chunks")
        cur.execute("CREATE TABLE bench_chunks (id int primary key, text_content text, parent_text text, tsvector_content tsvector)")
        psycopg2.extras.execute_values(
            cur, "INSERT INTO bench_chunks (id, text_content, parent_text) VALUES %s",
            [(c["id"], c["embed"], c["text"]) for c in chunks], page_size=1000)
        cur.execute("UPDATE bench_chunks SET tsvector_content = to_tsvector('english', coalesce(text_content,''))")
        cur.execute("CREATE INDEX ON bench_chunks USING gin(tsvector_content)")
        cur.execute("ANALYZE bench_chunks")

    def _cur(self):
        import psycopg2
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            return cur
        except psycopg2.Error:
            self.conn = psycopg2.connect(self.dsn, keepalives=1, keepalives_idle=30)
            self.conn.autocommit = True
            return self.conn.cursor()

    def bm25(self, query, limit):
        clean = re.sub(r"[|\-]{2,}", " ", query).strip()
        cur = self._cur()
        cur.execute(
            """SELECT id, ts_rank(tsvector_content, plainto_tsquery('english', %s)) AS r
               FROM bench_chunks WHERE plainto_tsquery('english', %s) @@ tsvector_content
               ORDER BY r DESC LIMIT %s""", (clean, clean, limit))
        return [row[0] for row in cur.fetchall()]

    def catalogue(self, query):
        pats = re.findall(r"\b[A-Z]{2,5}\d{2,6}[A-Z0-9\-]*\b", query)[:3]  # baseline regex
        cur = self._cur()
        out = []
        for p in pats:
            cur.execute("SELECT id FROM bench_chunks WHERE text_content ILIKE %s LIMIT 20", (f"%{p}%",))
            out += [r[0] for r in cur.fetchall()]
        return out


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def rrf(rank, k=60):
    return 1.0 / (k + rank)


def baseline_fast(q, ctx, top_k=12):
    t0 = time.perf_counter()
    exact = ctx["pg"].catalogue(q)
    qv = ctx["base_emb"].query(q)
    t_embed = time.perf_counter()
    sims = ctx["base_mat"] @ qv
    order = np.argsort(-sims)[: max(top_k * 8, 100)]
    cands = {}
    for rank, i in enumerate(order, 1):
        cands[int(ctx["chunks"][i]["id"])] = 0.5 * rrf(rank)
    for cid in exact:
        cands[cid] = max(cands.get(cid, 0) + 1.5, 2.0)
    ranked = sorted(cands.items(), key=lambda kv: -kv[1])
    out, seen = [], set()
    for cid, _ in ranked:
        t = ctx["by_id"][cid]["text"]
        if t in seen:
            continue
        seen.add(t)
        out.append(cid)
        if len(out) >= top_k:
            break
    t_end = time.perf_counter()
    return out, {"embed_ms": (t_embed - t0) * 1000, "total_ms": (t_end - t0) * 1000}


def baseline_full(q, ctx, top_k=12):
    t0 = time.perf_counter()
    exact = ctx["pg"].catalogue(q)
    qv = ctx["base_emb"].query(q)
    sims = ctx["base_mat"] @ qv
    limit = max(top_k * 8, 100)
    order = np.argsort(-sims)[:limit]
    bm = ctx["pg"].bm25(q, max(top_k * 4, 50))
    cands = {}
    for rank, i in enumerate(order, 1):  # dense keyed by int (Qdrant point id)
        cid = int(ctx["chunks"][i]["id"])
        cands[cid] = cands.get(cid, 0) + 0.5 * rrf(rank)
    for rank, cid in enumerate(bm, 1):   # BM25 keyed by str(row.id): never merges
        cands[str(cid)] = cands.get(str(cid), 0) + 0.4 * rrf(rank)
    for cid in exact:
        cands[cid] = max(cands.get(cid, 0) + 1.5, 2.0)
    ranked = sorted(cands.items(), key=lambda kv: -kv[1])
    out, seen = [], set()
    for cid, _ in ranked:
        t = ctx["by_id"][int(cid)]["text"]
        if t in seen:
            continue
        seen.add(t)
        out.append(int(cid))
        if len(out) >= top_k:
            break
    t_ret = time.perf_counter()
    if ctx.get("base_rr"):
        sc = ctx["base_rr"].scores(q, [ctx["by_id"][c]["text"] for c in out])
        out = [c for _, c in sorted(zip(sc, out), key=lambda x: -x[0])]
    t_end = time.perf_counter()
    return out, {"retrieve_ms": (t_ret - t0) * 1000, "rerank_ms": (t_end - t_ret) * 1000, "total_ms": (t_end - t0) * 1000}


def candidate(q, ctx, emb_key, rr_key=None, top_k=12, rerank_n=24):
    from app.rag.identifiers import extract_identifiers, normalize_identifier
    t0 = time.perf_counter()
    emb, mat = ctx[emb_key]
    qv = emb.query(q)
    t_embed = time.perf_counter()
    sims = mat @ qv
    dense = np.argsort(-sims)[:50]
    sparse = ctx["sparse"].search(q, 50)
    ids = {normalize_identifier(t) for t in extract_identifiers(q)}
    scores = defaultdict(float)
    for rank, i in enumerate(dense, 1):
        scores[int(i)] += rrf(rank)
    for rank, (i, _) in enumerate(sparse, 1):
        scores[int(i)] += rrf(rank)
    if ids:
        hits = [i for i in ctx["id_index"].get_many(ids)]
        hits.sort(key=lambda i: -sims[i])
        for rank, i in enumerate(hits[:30], 1):
            scores[int(i)] += 2 * rrf(rank)  # exact identifier evidence weighs double
    fused = [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    t_fuse = time.perf_counter()
    if rr_key:
        pool = fused[:rerank_n]
        sc = ctx[rr_key].scores(q, [ctx["chunks"][i]["text"] for i in pool])
        fused = [i for _, i in sorted(zip(sc, pool), key=lambda x: -x[0])]
    t_end = time.perf_counter()
    return [int(ctx["chunks"][i]["id"]) for i in fused[:top_k]], {
        "embed_ms": (t_embed - t0) * 1000, "fuse_ms": (t_fuse - t_embed) * 1000,
        "rerank_ms": (t_end - t_fuse) * 1000, "total_ms": (t_end - t0) * 1000}


class IdIndex:
    def __init__(self, chunks):
        from app.rag.identifiers import extract_identifiers, normalize_identifier
        self.ix = defaultdict(set)
        for i, c in enumerate(chunks):
            for t in extract_identifiers(c["embed"], limit=256):
                self.ix[normalize_identifier(t)].add(i)

    def get_many(self, keys):
        out = set()
        for k in keys:
            out |= self.ix.get(k, set())
        return list(out)


# ---------------------------------------------------------------------------
def evaluate(name, fn, queries, ctx):
    per_cat = defaultdict(lambda: Counter())
    lat = defaultdict(list)
    for item in queries:
        ids, timing = fn(item["query"])
        for k, v in timing.items():
            lat[k].append(v)
        cat = item["category"]
        per_cat[cat]["n"] += 1
        if item["expect"] is None:
            continue
        texts = [ctx["by_id"][i]["text"] for i in ids[:10]]
        rank = next((r for r, t in enumerate(texts, 1) if item["expect"] in t), None)
        if rank == 1:
            per_cat[cat]["hit1"] += 1
        if rank and rank <= 5:
            per_cat[cat]["hit5"] += 1
            joined = "\n".join(texts[:5])
            if item.get("answer") is None or item["answer"] in joined:
                per_cat[cat]["ans5"] += 1
        if rank:
            per_cat[cat]["mrr"] += 1.0 / rank
    rows = {}
    tot = Counter()
    for cat, c in sorted(per_cat.items()):
        if cat == "negative":
            continue
        n = c["n"]
        rows[cat] = {k: round(c[k] / n, 3) for k in ("hit1", "hit5", "ans5", "mrr")}
        for k in ("n", "hit1", "hit5", "ans5", "mrr"):
            tot[k] += c[k]
    n = tot["n"]
    rows["ALL"] = {k: round(tot[k] / n, 3) for k in ("hit1", "hit5", "ans5", "mrr")}
    latency = {k: {"p50": round(statistics.median(v), 1), "p95": round(sorted(v)[int(len(v) * 0.95) - 1], 1)} for k, v in lat.items()}
    print(f"\n=== {name} ===")
    for cat, r in rows.items():
        print(f"  {cat:18s} hit@1={r['hit1']:.3f} hit@5={r['hit5']:.3f} answer@5={r['ans5']:.3f} mrr={r['mrr']:.3f}")
    print("  latency:", latency)
    return {"metrics": rows, "latency_ms": latency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/data")
    ap.add_argument("--baseline-app", default="/baseline")
    ap.add_argument("--pg", default=os.getenv("BENCH_PG", "postgresql://bench:bench@bench-pg:5432/bench"))
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--skip-large", action="store_true", help="skip bge-large/v2-m3 baseline models")
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(os.path.join(args.data, "queries.jsonl"))]
    chunks = baseline_chunks(args.data, args.baseline_app)
    print(f"{len(chunks)} baseline chunks, {len(queries)} queries")
    ctx = {"chunks": chunks, "by_id": {c["id"]: c for c in chunks}}
    texts = [c["embed"] for c in chunks]
    results = {"n_chunks": len(chunks), "n_queries": len(queries), "ingest_throughput": {}, "runs": {}}

    ctx["pg"] = PgFTS(args.pg, chunks)

    def timed_docs(emb):
        import hashlib
        key = hashlib.sha1(("\n".join(texts) + emb.name).encode()).hexdigest()[:12]
        cache_path = os.path.join("/models", f"emb_{key}.npy")
        tp_path = cache_path + ".rate"
        if os.path.exists(cache_path) and os.path.exists(tp_path):
            results["ingest_throughput"][emb.name] = float(open(tp_path).read())
            print(f"loaded cached embeddings for {emb.name}")
            return np.load(cache_path)
        t = time.perf_counter()
        m = emb.docs(texts)
        dt = time.perf_counter() - t
        np.save(cache_path, m)
        open(tp_path, "w").write(str(round(len(texts) / dt, 1)))
        results["ingest_throughput"][emb.name] = round(len(texts) / dt, 1)
        print(f"embedded {len(texts)} chunks with {emb.name} in {dt:.1f}s ({len(texts)/dt:.1f}/s)")
        return m

    if not args.skip_large:
        ctx["base_emb"] = STEmbedder("BAAI/bge-large-en-v1.5")
        ctx["base_mat"] = timed_docs(ctx["base_emb"])
        ctx["base_rr"] = STReranker("BAAI/bge-reranker-v2-m3")
        results["runs"]["baseline_fast (bge-large dense only)"] = evaluate("baseline_fast", lambda q: baseline_fast(q, ctx), queries, ctx)
        results["runs"]["baseline_full (+pg fts, +v2-m3 rerank)"] = evaluate("baseline_full", lambda q: baseline_full(q, ctx), queries, ctx)

    ctx["sparse"] = SparseBM25(texts)
    ctx["id_index"] = IdIndex(chunks)
    small = FEEmbedder("BAAI/bge-small-en-v1.5")
    ctx["small"] = (small, timed_docs(small))
    ctx["minilm"] = FEReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    ctx["bgebase_rr"] = FEReranker("BAAI/bge-reranker-base")
    results["runs"]["cand hybrid bge-small"] = evaluate("hybrid small", lambda q: candidate(q, ctx, "small"), queries, ctx)
    results["runs"]["cand hybrid bge-small + MiniLM-L6"] = evaluate("hybrid small + minilm", lambda q: candidate(q, ctx, "small", "minilm"), queries, ctx)
    results["runs"]["cand hybrid bge-small + bge-reranker-base"] = evaluate("hybrid small + bge-rr-base", lambda q: candidate(q, ctx, "small", "bgebase_rr"), queries, ctx)
    if not args.skip_large:
        ctx["large"] = (ctx["base_emb"], ctx["base_mat"])
        ctx["large"][0].query = lambda q, m=ctx["base_emb"].m: np.asarray(m.encode(BGE_INSTRUCTION + q, normalize_embeddings=True), dtype=np.float32)
        results["runs"]["cand hybrid bge-large + MiniLM-L6"] = evaluate("hybrid large + minilm", lambda q: candidate(q, ctx, "large", "minilm"), queries, ctx)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "retrieval_components.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", os.path.join(args.out, "retrieval_components.json"))


if __name__ == "__main__":
    main()

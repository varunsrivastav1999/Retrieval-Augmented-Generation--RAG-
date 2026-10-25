"""Download and verify the ONNX models into the image at build time.

Run during `docker build` (internet needed once). At runtime the containers set
HF_HUB_OFFLINE=1 and load only from this cache.
"""

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/models/fastembed")
    ap.add_argument("--embedding", default=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    ap.add_argument("--sparse", default=os.getenv("RAG_SPARSE_MODEL", "Qdrant/bm25"))
    ap.add_argument("--reranker", default=os.getenv("RAG_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    args = ap.parse_args()

    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    def retry(fn, attempts=5):
        for i in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # network hiccup during build
                if i == attempts:
                    raise
                print(f"download failed ({exc}); retry {i}/{attempts - 1}", flush=True)
                time.sleep(5 * i)

    started = time.time()
    dense = retry(lambda: TextEmbedding(args.embedding, cache_dir=args.cache))
    dim = len(next(iter(dense.embed(["check"]))))
    sparse = retry(lambda: SparseTextEmbedding(args.sparse, cache_dir=args.cache))
    next(iter(sparse.embed(["check"])))
    reranker = retry(lambda: TextCrossEncoder(args.reranker, cache_dir=args.cache))
    list(reranker.rerank("check", ["check"]))
    print(f"models ready in {time.time() - started:.0f}s: {args.embedding} (dim={dim}), {args.sparse}, {args.reranker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

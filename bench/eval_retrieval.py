import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.embedder import Embedder
from retrieval.search import Retriever
from retrieval.sparse import Bm25SparseIndex
from retrieval.vector_store import NumpyVectorStore

PROJECT_ROOT = Path(__file__).parent.parent


def load_queries(limit=500, seed=42):
    by_lang = {}
    with open(PROJECT_ROOT / "data" / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            by_lang.setdefault(q.get("language", "en"), []).append(q)
    rng = random.Random(seed)
    per_lang = limit // len(by_lang)
    sampled = []
    for lang, qs in sorted(by_lang.items()):
        sampled.extend(rng.sample(qs, min(per_lang, len(qs))))
    rng.shuffle(sampled)
    return sampled


def load_gold():
    gold = {}
    with open(PROJECT_ROOT / "data" / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["metadata"].get("is_selected"):
                key = (c["metadata"]["language"], str(c["metadata"]["query_id"]))
                gold.setdefault(key, set()).add(c["document_id"])
    return gold


def evaluate(retriever, queries, gold):
    hits, rr, lats, no_gold = [], [], [], 0
    for q in queries:
        key = (q["language"], str(q["query_id"]))
        targets = gold.get(key)
        if not targets:
            no_gold += 1
            continue
        start = time.perf_counter()
        result = retriever.search(
            q["query"], top_k=10, use_cache=False, language_preference=q["language"]
        )
        lats.append((time.perf_counter() - start) * 1000)
        doc_ids = [c.document_id for c in result.chunks]
        rank = next((i + 1 for i, d in enumerate(doc_ids) if d in targets), None)
        if rank:
            hits.append(1)
            rr.append(1.0 / rank)
        else:
            hits.append(0)
            rr.append(0.0)
    return {
        "n_evaluated": len(hits),
        "no_gold_in_index": no_gold,
        "hit_rate_at_10": round(float(np.mean(hits)) * 100, 2),
        "mrr_at_10": round(float(np.mean(rr)), 4),
        "search_p50_ms": round(float(np.percentile(lats, 50)), 2),
        "search_p95_ms": round(float(np.percentile(lats, 95)), 2),
    }


if __name__ == "__main__":
    config = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    print("Loading model and indexes...")
    embedder = Embedder(config["embedding"]["model"])
    embedder.embed_query("warmup")
    store = NumpyVectorStore()
    store.load(str(PROJECT_ROOT / "index" / "artifacts"))

    sparse = Bm25SparseIndex()
    sparse_ok = sparse.load(str(PROJECT_ROOT / "index" / "artifacts" / "bm25"))
    hybrid_cfg = dict(config["retrieval"].get("hybrid", {}))
    hybrid_cfg["enabled"] = True

    dense = Retriever(embedder, store)
    hybrid = Retriever(
        embedder, store,
        sparse_index=sparse if sparse_ok else None,
        hybrid_config=hybrid_cfg,
    )

    queries = load_queries()
    gold = load_gold()
    print(f"{len(queries)} queries, {len(gold)} gold groups\n")

    for name, r in [("dense", dense), ("hybrid", hybrid)]:
        m = evaluate(r, queries, gold)
        print(f"[{name}] {json.dumps(m)}")

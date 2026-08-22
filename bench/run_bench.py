import json
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import AskRequest
from pipeline.orchestrator import PipelineOrchestrator


def load_test_queries(path: str = None, limit: int = 600, balanced: bool = True, seed: int = 42) -> List[Dict]:
    if path is None:
        path = Path(__file__).parent.parent / "data" / "queries.jsonl"
    all_queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            all_queries.append(json.loads(line))

    if balanced:
        by_lang = {}
        for q in all_queries:
            lang = q.get("language", "en")
            by_lang.setdefault(lang, []).append(q)
        per_lang = limit // max(len(by_lang), 1)
        sampled = []
        rng = random.Random(seed)
        for lang, qs in sorted(by_lang.items()):
            n = min(per_lang, len(qs))
            sampled.extend(rng.sample(qs, n))
        rng.shuffle(sampled)
        return sampled
    else:
        return all_queries[:limit]


def run_benchmark(
    orchestrator: PipelineOrchestrator,
    queries: List[Dict],
    output_dir: str = None,
    soft_lang: bool = False,
    hybrid: bool = False,
) -> Dict:
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(queries):
        query_text = q["query"][:500]
        request = AskRequest(
            query=query_text,
            lang=None if soft_lang else q.get("language"),
            top_k=10,
            allow_generative=False,
        )

        start = time.perf_counter()
        response = orchestrator.process_text(request)
        total_ms = (time.perf_counter() - start) * 1000

        results.append({
            "query": q["query"],
            "language": q.get("language", "en"),
            "status": response.status,
            "method": response.answer.method if response.answer else "none",
            "core_ms": response.latency.total_core_ms,
            "embed_ms": response.latency.embed_ms,
            "retrieve_ms": response.latency.retrieve_ms,
            "guard_input_ms": response.latency.guard_input_ms,
            "guard_relevance_ms": response.latency.guard_relevance_ms,
            "answer_ms": response.latency.answer_ms,
            "guard_grounding_ms": response.latency.guard_grounding_ms,
            "total_ms": total_ms,
            "detected_language": response.detected_language,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(queries)} done...")

    core_lats = [r["core_ms"] for r in results]
    embed_lats = [r["embed_ms"] for r in results]
    retrieve_lats = [r["retrieve_ms"] for r in results]
    answer_lats = [r["answer_ms"] for r in results]
    guard_in_lats = [r["guard_input_ms"] for r in results]
    guard_rel_lats = [r["guard_relevance_ms"] for r in results]
    guard_grd_lats = [r["guard_grounding_ms"] for r in results]
    total_lats = [r["total_ms"] for r in results]

    def pct(arr, p):
        return round(float(np.percentile(arr, p)), 2) if arr else 0

    summary = {
        "mode": ("hybrid" if hybrid else "dense") + ("-soft-lang" if soft_lang else "-hard-lang"),
        "queries_tested": len(results),
        "core_pipeline": {
            "p50_ms": pct(core_lats, 50),
            "p70_ms": pct(core_lats, 70),
            "p90_ms": pct(core_lats, 90),
            "p95_ms": pct(core_lats, 95),
            "p100_ms": round(max(core_lats), 2) if core_lats else 0,
            "mean_ms": round(float(np.mean(core_lats)), 2) if core_lats else 0,
            "std_ms": round(float(np.std(core_lats)), 2) if core_lats else 0,
        },
        "total_e2e": {
            "p50_ms": pct(total_lats, 50),
            "p70_ms": pct(total_lats, 70),
            "p100_ms": round(max(total_lats), 2) if total_lats else 0,
            "mean_ms": round(float(np.mean(total_lats)), 2) if total_lats else 0,
        },
        "component_breakdown": {
            "embed_p50_ms": pct(embed_lats, 50),
            "retrieve_p50_ms": pct(retrieve_lats, 50),
            "answer_p50_ms": pct(answer_lats, 50),
            "guard_input_p50_ms": pct(guard_in_lats, 50),
            "guard_relevance_p50_ms": pct(guard_rel_lats, 50),
            "guard_grounding_p50_ms": pct(guard_grd_lats, 50),
        },
        "within_200ms": f"{sum(1 for x in core_lats if x < 200) / len(core_lats) * 100:.1f}%" if core_lats else "0%",
        "by_language": {},
        "by_status": dict(Counter(r["status"] for r in results)),
        "by_method": dict(Counter(r["method"] for r in results)),
    }

    for lang in ["en", "hi", "gu", "te"]:
        lang_lats = [r["core_ms"] for r in results if r["language"] == lang]
        if lang_lats:
            summary["by_language"][lang] = {
                "count": len(lang_lats),
                "p50_ms": pct(lang_lats, 50),
                "p70_ms": pct(lang_lats, 70),
                "p100_ms": round(max(lang_lats), 2),
                "mean_ms": round(float(np.mean(lang_lats)), 2),
                "within_200ms": f"{sum(1 for x in lang_lats if x < 200) / len(lang_lats) * 100:.1f}%",
            }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"HH GOA RAG BENCHMARK [{summary['mode']}]")
    print("=" * 60)
    print(f"\nQueries: {summary['queries_tested']}")
    print(f"\n--- Core Pipeline Latency ---")
    print(f"  P50:  {summary['core_pipeline']['p50_ms']}ms")
    print(f"  P70:  {summary['core_pipeline']['p70_ms']}ms")
    print(f"  P90:  {summary['core_pipeline']['p90_ms']}ms")
    print(f"  P100: {summary['core_pipeline']['p100_ms']}ms")
    print(f"  Mean: {summary['core_pipeline']['mean_ms']}ms")
    print(f"\n--- Component Breakdown (P50) ---")
    for k, v in summary["component_breakdown"].items():
        print(f"  {k}: {v}ms")
    print(f"\n--- By Language ---")
    for lang, stats in summary["by_language"].items():
        print(f"  {lang}: n={stats['count']} P50={stats['p50_ms']}ms P100={stats['p100_ms']}ms Within200ms={stats['within_200ms']}")
    print(f"\n--- SLA ---")
    print(f"  Within 200ms: {summary['within_200ms']}")
    print(f"  Status: {summary['by_status']}")
    print(f"  Method: {summary['by_method']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    import yaml
    from retrieval.search import Retriever
    from retrieval.embedder import Embedder
    from retrieval.vector_store import NumpyVectorStore
    from guardrails.input_guard import InputGuard
    from guardrails.relevance_guard import RelevanceGuard
    from guardrails.grounding_guard import GroundingGuard
    from generation.extractive import ExtractiveGenerator
    from generation.generative import GenerativeGenerator
    from stt.sarvam import SarvamSTT
    from pipeline.orchestrator import PipelineOrchestrator
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    PROJECT_ROOT = Path(__file__).parent.parent
    CONFIG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    ARTIFACTS = PROJECT_ROOT / "index" / "artifacts"

    print("Loading model...")
    embedder = Embedder(CONFIG["embedding"]["model"])
    embedder.embed_query("warmup")  # exclude one-time model load from measurements
    vector_store = NumpyVectorStore()
    vector_store.load(str(ARTIFACTS))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--soft-lang", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.hybrid:
        from retrieval.sparse import Bm25SparseIndex
        sparse_index = Bm25SparseIndex()
        if not sparse_index.load(str(ARTIFACTS / "bm25")):
            print("ERROR: BM25 index not found at index/artifacts/bm25; run index/build_sparse.py first.")
            sys.exit(1)
        hybrid_cfg = dict(CONFIG["retrieval"].get("hybrid", {}))
        hybrid_cfg["enabled"] = True
        retriever = Retriever(embedder, vector_store, sparse_index=sparse_index, hybrid_config=hybrid_cfg)
        print(f"Hybrid mode: BM25 loaded ({sparse_index.num_docs} docs), candidate_k={hybrid_cfg.get('candidate_k', 30)}, rrf_k={hybrid_cfg.get('rrf_k', 60)}")
    else:
        retriever = Retriever(embedder, vector_store)

    llm_config = CONFIG.get("llm", {})

    orchestrator = PipelineOrchestrator(
        retriever=retriever,
        input_guard=InputGuard(CONFIG["guardrails"]["input"]),
        relevance_guard=RelevanceGuard(CONFIG["guardrails"]["relevance"]),
        grounding_guard=GroundingGuard(CONFIG["guardrails"]["grounding"], embedder=embedder),
        extractive_gen=ExtractiveGenerator(CONFIG["generation"]["extractive"]["min_retrieval_score"]),
        generative_gen=GenerativeGenerator(
            provider=llm_config.get("provider", "openrouter"),
            base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
            model=llm_config.get("model", ""),
            reasoning=llm_config.get("reasoning", False),
        ),
        stt=SarvamSTT(),
        llm_model=llm_config.get("model", ""),
    )

    queries = load_test_queries(limit=args.limit, balanced=not args.sequential)
    lang_counts = Counter(q.get("language", "?") for q in queries)
    modes = []
    if args.hybrid:
        modes.append("hybrid")
    if args.soft_lang:
        modes.append("soft-lang")
    mode_tag = "+".join(modes) if modes else "dense-hard"
    print(f"Running benchmark [{mode_tag}] on {len(queries)} queries: {dict(lang_counts)}")
    run_benchmark(orchestrator, queries, output_dir=args.out, soft_lang=args.soft_lang, hybrid=args.hybrid)

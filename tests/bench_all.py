import pytest
import json
import time
import sys
from pathlib import Path
from typing import List

sys.stdout.reconfigure(encoding='utf-8')

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
from api.schemas import Chunk
from chunking.base import get_chunker, CHUNKERS
from retrieval.embedder import Embedder


# Shared test data
PASSAGES = {
    "en": [
        "India is a country in South Asia. It is the world's most populous country and the seventh-largest by area. The country is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast.",
        "The capital of India is New Delhi, a city with a rich history dating back centuries. New Delhi is home to many government buildings and historical monuments including the Red Fort and India Gate.",
        "Mumbai is the financial capital of India and the largest city by population. It is known for Bollywood, the Hindi film industry, and its bustling street life. The city is located on the coast of Maharashtra.",
        "Bangalore, officially Bengaluru, is the capital of Karnataka and known as the Silicon Valley of India. It is a major center for IT services and technology companies. The city has a pleasant climate year-round.",
        "The Taj Mahal is located in Agra, Uttar Pradesh. It was built by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal. It is one of the Seven Wonders of the World.",
    ],
    "hi": [
        "भारत दक्षिण एशिया में एक देश है। यह दुनिया का सबसे अधिक जनसंख्या वाला देश है और क्षेत्रफल में सातवां सबसे बड़ा है।",
        "भारत की राजधानी नई दिल्ली है, जो सदियों पुराने इतिहास वाला शहर है। नई दिल्ली में कई सरकारी इमारतें और ऐतिहासिक स्मारक हैं।",
        "मुंबई भारत की वित्तीय राजधानी है और जनसंख्या में सबसे बड़ा शहर है। यह बॉलीवुड और हिंदी फिल्म उद्योग के लिए प्रसिद्ध है।",
        "बैंगलोर, आधिकारिक रूप से बेंगलुरु, कर्नाटक की राजधानी है और भारत का सिलिकॉन वैली के रूप में जाना जाता है।",
        "ताजमहल उत्तर प्रदेश के आगरा में स्थित है। इसे मुगल सम्राट शाहजहाँ ने अपनी पत्नी मुमताज महल की याद में बनवाया था।",
    ],
    "gu": [
        "ભારત દક્ષિણ એશિયામાં એક દેશ છે. તે વિશ્વનો સૌથી વધુ વસ્તી ધરાવતો દેશ છે અને ક્ષેત્રફળમાં સાતમો સૌથી મોટો છે.",
        "ભારતની રાજધાની નવી દિલ્હી છે, જે સદીઓ જૂનો ઇતિહાસ ધરાવતું શહેર છે. નવી દિલ્હીમાં ઘણી સરકારી ઇમારતો અને ઐતિહાસિક સ્મારકો છે.",
        "મુંબઈ ભારતનું નાણાકીય કેન્દ્ર છે અને વસ્તીમાં સૌથી મોટું શહેર છે. તે બોલિવૂડ અને હિન્દી ફિલ્મ ઉદ્યોગ માટે પ્રખ્યાત છે.",
        "બેંગલોર, અધિકૃત રીતે બેંગલુરુ, કર્ણાટકની રાજધાની છે અને ભારતની સિલિકોન વેલી તરીકે જાણીતી છે.",
        "તાજમહેલ ઉત્તર પ્રદેશના આગ્રામાં સ્થિત છે. તેને મુઘલ સમ્રાટ શાહજહાંએ પોતાની પત્ની મુમતાજ મહેલની યાદમાં બનાવ્યું હતું.",
    ],
}


def get_all_passages(lang: str) -> List[str]:
    return PASSAGES.get(lang, [])


def get_all_chunks_flat() -> List[Chunk]:
    chunks = []
    for lang, texts in PASSAGES.items():
        for i, text in enumerate(texts):
            chunks.append(Chunk(
                chunk_id=f"{lang}_{i}",
                document_id=f"doc_{lang}_{i}",
                text=text,
                metadata={"language": lang},
            ))
    return chunks


def test_all_chunking_strategies():
    """Benchmark all 6 chunking strategies on multilingual text."""
    all_chunks = get_all_chunks_flat()
    results = {}

    for strategy_name in CHUNKERS:
        chunker = get_chunker(strategy_name)
        strategy_chunks = []
        for chunk in all_chunks:
            strategy_chunks.extend(chunker.chunk(chunk.text, chunk.metadata))

        chunk_lengths = [len(c.text.split()) for c in strategy_chunks]
        results[strategy_name] = {
            "total_chunks": len(strategy_chunks),
            "avg_words": float(np.mean(chunk_lengths)) if chunk_lengths else 0,
            "min_words": int(np.min(chunk_lengths)) if chunk_lengths else 0,
            "max_words": int(np.max(chunk_lengths)) if chunk_lengths else 0,
            "std_words": float(np.std(chunk_lengths)) if chunk_lengths else 0,
        }

    return results


def test_embedding_latency():
    """Benchmark embedding latency across languages."""
    embedder = Embedder()
    embedder.embed_query("warmup")  # pre-warm model
    results = {}
    for lang, texts in PASSAGES.items():
        start = time.perf_counter()
        vecs = embedder.embed_passages(texts, batch_size=5)
        elapsed = (time.perf_counter() - start) * 1000
        results[lang] = {
            "num_passages": len(texts),
            "embed_time_ms": round(elapsed, 2),
            "per_passage_ms": round(elapsed / len(texts), 2),
        }
    return results


def test_search_latency():
    """Benchmark search latency with the full 30K index."""
    embedder = Embedder()
    embedder.embed_query("warmup")  # pre-warm model
    from retrieval.vector_store import NumpyVectorStore
    store = NumpyVectorStore()
    if not store.load(str(root / "index" / "artifacts")):
        return {"error": "Index not found"}

    queries = {
        "en": ["capital of India", "largest city", "technology hub"],
        "hi": ["भारत की राजधानी", "सबसे बड़ा शहर", "तकनीकी केंद्र"],
        "gu": ["ભારતની રાજધાની", "સૌથી મોટું શહેર", "ટેક્નોલોજી હબ"],
    }

    results = {}
    for lang, query_list in queries.items():
        lang_latencies = []
        for q in query_list:
            vec = embedder.embed_query(q)
            start = time.perf_counter()
            hits = store.search(vec, top_k=10)
            elapsed = (time.perf_counter() - start) * 1000
            lang_latencies.append(elapsed)
        results[lang] = {
            "queries": len(query_list),
            "avg_ms": round(float(np.mean(lang_latencies)), 2),
            "min_ms": round(float(np.min(lang_latencies)), 2),
            "max_ms": round(float(np.max(lang_latencies)), 2),
        }
    return results


def test_embed_and_search_p50_p70_p100():
    """Measure embed+search P50/P70/P100 with 50 queries across languages."""
    embedder = Embedder()
    # Pre-warm model to exclude cold start
    embedder.embed_query("warmup")
    from retrieval.vector_store import NumpyVectorStore
    store = NumpyVectorStore()
    if not store.load(str(root / "index" / "artifacts")):
        return {"error": "Index not found"}

    import json
    queries_path = root / "data" / "queries.jsonl"
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    # Sample 50 queries balanced across languages
    sampled = []
    by_lang = {"en": [], "hi": [], "gu": []}
    for q in queries:
        lang = q.get("language", "en")
        if lang in by_lang and len(by_lang[lang]) < 17:
            by_lang[lang].append(q)

    for lang, qs in by_lang.items():
        sampled.extend(qs)

    latencies = []
    for q in sampled:
        start = time.perf_counter()
        vec = embedder.embed_query(q["query"])
        hits = store.search(vec, top_k=10)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append({
            "lang": q.get("language", "en"),
            "embed_search_ms": round(elapsed, 2),
        })

    all_ms = [l["embed_search_ms"] for l in latencies]
    return {
        "n": len(sampled),
        "p50": round(float(np.percentile(all_ms, 50)), 2),
        "p70": round(float(np.percentile(all_ms, 70)), 2),
        "p90": round(float(np.percentile(all_ms, 90)), 2),
        "p95": round(float(np.percentile(all_ms, 95)), 2),
        "p100": round(float(np.max(all_ms)), 2),
        "mean": round(float(np.mean(all_ms)), 2),
        "std": round(float(np.std(all_ms)), 2),
        "within_200ms_pct": round(sum(1 for x in all_ms if x < 200) / len(all_ms) * 100, 1),
        "by_lang": {
            lang: {
                "n": len([l for l in latencies if l["lang"] == lang]),
                "p50": round(float(np.percentile([l["embed_search_ms"] for l in latencies if l["lang"] == lang], 50)), 2),
                "p100": round(float(max([l["embed_search_ms"] for l in latencies if l["lang"] == lang])), 2),
            }
            for lang in ["en", "hi", "gu"]
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK: Chunking Strategies")
    print("=" * 60)
    chunking_results = test_all_chunking_strategies()
    for name, stats in chunking_results.items():
        print(f"\n  {name}:")
        print(f"    Chunks: {stats['total_chunks']}  |  Avg: {stats['avg_words']:.1f} words  |  Min: {stats['min_words']}  |  Max: {stats['max_words']}  |  Std: {stats['std_words']:.1f}")

    print("\n" + "=" * 60)
    print("BENCHMARK: Embedding Latency")
    print("=" * 60)
    embed_results = test_embedding_latency()
    for lang, stats in embed_results.items():
        print(f"  {lang}: {stats['num_passages']} passages in {stats['embed_time_ms']:.1f}ms ({stats['per_passage_ms']:.1f}ms/passage)")

    print("\n" + "=" * 60)
    print("BENCHMARK: Search Latency")
    print("=" * 60)
    search_results = test_search_latency()
    for lang, stats in search_results.items():
        print(f"  {lang}: avg={stats['avg_ms']:.2f}ms  min={stats['min_ms']:.2f}ms  max={stats['max_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("BENCHMARK: P50/P70/P100 (embed + search, 50 queries)")
    print("=" * 60)
    p50_results = test_embed_and_search_p50_p70_p100()
    if "error" not in p50_results:
        print(f"  n={p50_results['n']}")
        print(f"  P50:  {p50_results['p50']:.2f}ms")
        print(f"  P70:  {p50_results['p70']:.2f}ms")
        print(f"  P90:  {p50_results['p90']:.2f}ms")
        print(f"  P95:  {p50_results['p95']:.2f}ms")
        print(f"  P100: {p50_results['p100']:.2f}ms")
        print(f"  Mean: {p50_results['mean']:.2f}ms  Std: {p50_results['std']:.2f}ms")
        print(f"  Within 200ms: {p50_results['within_200ms_pct']}%")
        for lang, stats in p50_results["by_lang"].items():
            print(f"  {lang}: n={stats['n']}  P50={stats['p50']:.2f}ms  P100={stats['p100']:.2f}ms")
    else:
        print(f"  {p50_results['error']}")

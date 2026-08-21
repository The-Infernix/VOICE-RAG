import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from pipeline.orchestrator import PipelineOrchestrator
from retrieval.search import Retriever
from retrieval.embedder import Embedder
from retrieval.vector_store import NumpyVectorStore
from guardrails.input_guard import InputGuard
from guardrails.relevance_guard import RelevanceGuard
from guardrails.grounding_guard import GroundingGuard
from generation.extractive import ExtractiveGenerator
from generation.generative import GenerativeGenerator
from stt.sarvam import SarvamSTT
from api.schemas import AskRequest

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ARTIFACTS_DIR = Path(__file__).parent.parent / "index" / "artifacts"

config = yaml.safe_load(open(CONFIG_PATH, "r"))

print("Loading embedding model...")
embedder = Embedder(config["embedding"]["model"])

print("Loading vector store...")
vector_store = NumpyVectorStore()
if not vector_store.load(str(ARTIFACTS_DIR)):
    print("WARNING: No pre-built index found.")
    vector_store = NumpyVectorStore()

retriever = Retriever(embedder, vector_store)
input_guard = InputGuard(config["guardrails"]["input"])
relevance_guard = RelevanceGuard(config["guardrails"]["relevance"])
grounding_guard = GroundingGuard(config["guardrails"]["grounding"])
extractive_gen = ExtractiveGenerator(config["generation"]["extractive"]["min_retrieval_score"])
llm_cfg = config.get("llm", {})
generative_gen = GenerativeGenerator(
    provider=llm_cfg.get("provider", "openrouter"),
    base_url=llm_cfg.get("base_url", "https://openrouter.ai/api/v1"),
    model=llm_cfg.get("model", ""),
    reasoning=llm_cfg.get("reasoning", False),
)
stt = SarvamSTT()

orch = PipelineOrchestrator(
    retriever=retriever,
    input_guard=input_guard,
    relevance_guard=relevance_guard,
    grounding_guard=grounding_guard,
    extractive_gen=extractive_gen,
    generative_gen=generative_gen,
    stt=stt,
)
print("Orchestrator ready.\n")

queries = [
    ("भारत में कितनी भाषाएँ बोली जाती हैं", "hi"),
    ("गुजरात में कौन सी नदी बहती है", "hi"),
    ("भारत की राजधानी क्या है", "hi"),
]

for q, lang in queries:
    print(f"{'='*60}")
    print(f"Query: {q} (lang={lang})")

    req = AskRequest(query=q, lang=lang, top_k=5, allow_generative=False)

    # First, let's test just retrieval to see scores
    qvec = retriever.embedder.embed_query(q)
    results = retriever.vector_store.search(query_vector=qvec, top_k=5, language_filter=lang)
    print(f"\nRaw retrieval scores:")
    for idx, score in results[:5]:
        chunk = retriever.vector_store.get_chunk(idx)
        print(f"  score={score:.4f} | text={chunk.text[:80]}...")

    # Now full pipeline
    start = time.perf_counter()
    resp = orch.process_text(req)
    ms = (time.perf_counter() - start) * 1000

    print(f"\nStatus: {resp.status}")
    print(f"Detected lang: {resp.detected_language}")

    if resp.answer:
        print(f"Answer: {resp.answer.text[:300]}")
        print(f"Method: {resp.answer.method}")
    else:
        print(f"Answer: NONE (refusal: {resp.refusal_reason})")

    print(f"Latency: {ms:.1f}ms total")
    print(f"Core: {resp.latency.total_core_ms:.1f}ms")
    print(f"  embed: {resp.latency.embed_ms:.1f}ms")
    print(f"  retrieve: {resp.latency.retrieve_ms:.1f}ms")
    print(f"  guard_relevance: {resp.latency.guard_relevance_ms:.1f}ms")
    print(f"  answer: {resp.latency.answer_ms:.1f}ms")
    print()

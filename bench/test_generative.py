import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

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
vector_store.load(str(ARTIFACTS_DIR))

retriever = Retriever(embedder, vector_store)

orch = PipelineOrchestrator(
    retriever=retriever,
    input_guard=InputGuard(config["guardrails"]["input"]),
    relevance_guard=RelevanceGuard(config["guardrails"]["relevance"]),
    grounding_guard=GroundingGuard(config["guardrails"]["grounding"], embedder=embedder),
    extractive_gen=ExtractiveGenerator(config["generation"]["extractive"]["min_retrieval_score"]),
    generative_gen=GenerativeGenerator(
        provider=config.get("llm", {}).get("provider", "openrouter"),
        base_url=config.get("llm", {}).get("base_url", "https://openrouter.ai/api/v1"),
        model=config.get("llm", {}).get("model", ""),
        reasoning=config.get("llm", {}).get("reasoning", False),
    ),
    stt=SarvamSTT(),
)
print("Orchestrator ready.\n")

queries = [
    ("भारत में कितनी भाषाएँ बोली जाती हैं", "hi"),
    ("गुजरात में कौन सी नदी बहती है", "gu"),
    ("भारत की राजधानी क्या है", "hi"),
]

for q, lang in queries:
    print(f"{'='*60}")
    print(f"Query: {q} (lang={lang})")

    req = AskRequest(query=q, lang=lang, top_k=5, allow_generative=True)
    start = time.perf_counter()
    resp = orch.process_text(req)
    ms = (time.perf_counter() - start) * 1000

    print(f"Status: {resp.status}")
    print(f"Detected lang: {resp.detected_language}")

    if resp.answer:
        print(f"Answer: {resp.answer.text[:500]}")
        print(f"Method: {resp.answer.method}")
        print(f"Confidence: {resp.answer.confidence}")
    else:
        print(f"Answer: NONE (refusal: {resp.refusal_reason})")

    print(f"Latency: {ms:.1f}ms total (E2E)")
    print(f"Core: {resp.latency.total_core_ms:.1f}ms")
    print(f"  embed: {resp.latency.embed_ms:.1f}ms")
    print(f"  retrieve: {resp.latency.retrieve_ms:.1f}ms")
    print(f"  answer: {resp.latency.answer_ms:.1f}ms")
    print(f"  grounding: {resp.latency.guard_grounding_ms:.1f}ms")
    print()

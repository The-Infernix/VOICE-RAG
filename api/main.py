import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import AskRequest, AskResponse, HealthResponse
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

import yaml
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ARTIFACTS_DIR = Path(__file__).parent.parent / "index" / "artifacts"

orchestrator: PipelineOrchestrator = None


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    config = load_config()

    print("Loading embedding model...")
    embedder = Embedder(config["embedding"]["model"])
    embedder.embed_query("warmup")
    print("Embedding model warmed up.")

    print("Loading vector store...")
    vector_store = NumpyVectorStore()
    if not vector_store.load(str(ARTIFACTS_DIR)):
        print("WARNING: No pre-built index found. Run index/build_index.py first.")
        vector_store = NumpyVectorStore()

    retriever = Retriever(embedder, vector_store)

    input_guard = InputGuard(config["guardrails"]["input"])
    relevance_guard = RelevanceGuard(config["guardrails"]["relevance"])
    grounding_guard = GroundingGuard(config["guardrails"]["grounding"], embedder=embedder)

    extractive_gen = ExtractiveGenerator(config["generation"]["extractive"]["min_retrieval_score"])

    llm_config = config.get("llm", {})
    generative_gen = GenerativeGenerator(
        provider=llm_config.get("provider", "openrouter"),
        base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
    )

    stt = SarvamSTT(model=config.get("stt", {}).get("sarvam", {}).get("model", "saaras:v3"))

    orchestrator = PipelineOrchestrator(
        retriever=retriever,
        input_guard=input_guard,
        relevance_guard=relevance_guard,
        grounding_guard=grounding_guard,
        extractive_gen=extractive_gen,
        generative_gen=generative_gen,
        stt=stt,
        llm_model=llm_config.get("model", ""),
    )

    print("Orchestrator ready.")
    yield


app = FastAPI(
    title="Voice RAG HH Goa",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/query", response_model=AskResponse)
async def query_v1(request: AskRequest):
    return orchestrator.process_text(request)


@app.post("/api/v1/voice/query", response_model=AskResponse)
async def voice_query_v1(
    audio: UploadFile = File(...),
    top_k: int = Form(default=10),
    allow_generative: bool = Form(default=False),
    debug: bool = Form(default=False),
):
    audio_bytes = await audio.read()
    return await orchestrator.process_voice(audio_bytes, top_k, allow_generative, debug)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    return orchestrator.process_text(request)


@app.post("/ask-voice", response_model=AskResponse)
async def ask_voice(
    audio: UploadFile = File(...),
    top_k: int = Form(default=10),
    allow_generative: bool = Form(default=False),
):
    audio_bytes = await audio.read()
    return await orchestrator.process_voice(audio_bytes, top_k, allow_generative)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        index_size=orchestrator.retriever.vector_store.size(),
        cache_size=orchestrator.retriever.cache.size(),
        providers={
            "sarvam": bool(os.environ.get("SARVAM_API_KEY")),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
        uptime_seconds=orchestrator.uptime(),
    )


from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

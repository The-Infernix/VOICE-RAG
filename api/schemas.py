from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import uuid


class QueryType(str, Enum):
    FACTUAL = "factual"
    DESCRIPTIVE = "descriptive"
    AMBIGUOUS = "ambiguous"
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"


class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    GUJARATI = "gu"
    TELUGU = "te"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    lang: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    use_cache: bool = True
    allow_generative: bool = False
    chunking_strategy: Optional[str] = None
    debug: bool = False


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict = {}
    score: float = 0.0


class RetrievalResult(BaseModel):
    chunks: List[Chunk]
    strategy_used: str
    cache_hit: bool = False
    degradations: List[str] = []


class GuardResult(BaseModel):
    passed: bool
    reason_code: Optional[str] = None
    message: Optional[str] = None
    score: Optional[float] = None


class StageTrace(BaseModel):
    stage: str
    latency_ms: float
    status: str = "success"
    details: Optional[dict] = None


class Citation(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str = ""


class Answer(BaseModel):
    text: str
    method: str
    citations: List[Citation] = []
    confidence: float = 0.0


class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    guard_input_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    guard_relevance_ms: float = 0.0
    answer_ms: float = 0.0
    guard_grounding_ms: float = 0.0
    total_core_ms: float = 0.0
    total_e2e_ms: float = 0.0


class RetrievedPassage(BaseModel):
    rank: int
    score: float
    text: str
    chunk_id: str = ""
    metadata: dict = {}


class DebugInfo(BaseModel):
    retrieved_context: List[RetrievedPassage] = []
    grounding_status: str = ""
    grounding_score: float = 0.0
    grounding_method: str = ""
    grounding_detail: str = ""
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    chunking_strategy: str = ""
    index_size: int = 0
    llm_model: str = ""
    top_similarity: float = 0.0
    relevance_floor: float = 0.0
    budget_ms: float = 200.0
    degradations: List[str] = []


class AskResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: str
    query: str
    detected_language: str = ""
    answer: Optional[Answer] = None
    refusal_reason: Optional[str] = None
    latency: LatencyBreakdown = LatencyBreakdown()
    stages: List[StageTrace] = []
    debug: Optional[DebugInfo] = None


class VoiceRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=50)
    allow_generative: bool = False


class HealthResponse(BaseModel):
    status: str
    index_size: int
    cache_size: int
    providers: dict = {}
    uptime_seconds: float = 0.0

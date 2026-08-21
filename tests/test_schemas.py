import pytest
from api.schemas import (
    AskRequest, AskResponse, Answer, Chunk, RetrievalResult,
    GuardResult, StageTrace, LatencyBreakdown, Citation,
    QueryType, Language, VoiceRequest, HealthResponse,
)


class TestQueryType:
    def test_enum_values(self):
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.DESCRIPTIVE.value == "descriptive"
        assert QueryType.AMBIGUOUS.value == "ambiguous"
        assert QueryType.OFF_TOPIC.value == "off_topic"
        assert QueryType.UNSAFE.value == "unsafe"


class TestLanguage:
    def test_enum_values(self):
        assert Language.ENGLISH.value == "en"
        assert Language.HINDI.value == "hi"
        assert Language.GUJARATI.value == "gu"


class TestAskRequest:
    def test_valid_request(self):
        req = AskRequest(query="What is the capital of India?", top_k=5)
        assert req.query == "What is the capital of India?"
        assert req.top_k == 5
        assert req.lang is None
        assert req.use_cache is True
        assert req.allow_generative is True

    def test_min_query_length(self):
        with pytest.raises(Exception):
            AskRequest(query="", top_k=5)

    def test_max_query_length(self):
        req = AskRequest(query="x" * 600, top_k=5)
        assert req.query

    def test_top_k_bounds(self):
        req = AskRequest(query="test", top_k=1)
        assert req.top_k == 1
        req = AskRequest(query="test", top_k=50)
        assert req.top_k == 50

    def test_top_k_invalid(self):
        with pytest.raises(Exception):
            AskRequest(query="test", top_k=0)
        with pytest.raises(Exception):
            AskRequest(query="test", top_k=51)


class TestChunk:
    def test_defaults(self):
        c = Chunk(chunk_id="c1", document_id="d1", text="hello")
        assert c.metadata == {}
        assert c.score == 0.0

    def test_with_metadata(self):
        c = Chunk(chunk_id="c1", document_id="d1", text="hello", metadata={"lang": "en"}, score=0.9)
        assert c.metadata["lang"] == "en"
        assert c.score == 0.9


class TestGuardResult:
    def test_passed(self):
        r = GuardResult(passed=True)
        assert r.passed is True
        assert r.reason_code is None

    def test_failed(self):
        r = GuardResult(passed=False, reason_code="EMPTY_INPUT", message="Empty")
        assert r.passed is False
        assert r.reason_code == "EMPTY_INPUT"


class TestStageTrace:
    def test_defaults(self):
        s = StageTrace(stage="embed", latency_ms=12.5)
        assert s.status == "success"
        assert s.details is None


class TestLatencyBreakdown:
    def test_defaults(self):
        lb = LatencyBreakdown()
        assert lb.stt_ms == 0.0
        assert lb.total_core_ms == 0.0
        assert lb.total_e2e_ms == 0.0


class TestAnswer:
    def test_extractive(self):
        a = Answer(text="Delhi", method="extractive", confidence=0.9)
        assert a.method == "extractive"
        assert len(a.citations) == 0

    def test_generative(self):
        a = Answer(text="Delhi is the capital", method="generative", confidence=0.8)
        assert a.method == "generative"


class TestRetrievalResult:
    def test_defaults(self):
        r = RetrievalResult(chunks=[], strategy_used="none")
        assert r.cache_hit is False


class TestAskResponse:
    def test_success(self):
        r = AskResponse(status="success", query="test", answer=Answer(text="hi", method="extractive"))
        assert r.status == "success"
        assert r.answer is not None

    def test_refused(self):
        r = AskResponse(status="refused", query="test", refusal_reason="empty")
        assert r.status == "refused"
        assert r.answer is None


class TestHealthResponse:
    def test_fields(self):
        h = HealthResponse(status="ok", index_size=30000, cache_size=0, providers={"groq": True})
        assert h.index_size == 30000
        assert h.providers["groq"] is True

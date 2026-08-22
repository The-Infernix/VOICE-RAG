import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.schemas import Chunk
from generation.generative import GenerativeGenerator
from retrieval.embedder import Embedder
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.search import Retriever
from retrieval.sparse import Bm25SparseIndex, tokenize
from retrieval.vector_store import NumpyVectorStore

TEXTS = [
    "India is a country in South Asia.",
    "The capital of India is New Delhi.",
    "Mumbai is the financial capital of India.",
    "Bangalore is a technology hub in India.",
    "Delhi is the capital city of India.",
]


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def store(embedder):
    vs = NumpyVectorStore()
    vectors = embedder.embed_passages(TEXTS, batch_size=5)
    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="d", text=t, metadata={"language": "en"})
        for i, t in enumerate(TEXTS)
    ]
    vs.add(vectors, chunks)
    return vs


@pytest.fixture(scope="module")
def sparse():
    return Bm25SparseIndex().build(TEXTS)


def make_retriever(embedder, store, sparse_index=None, enabled=False):
    return Retriever(
        embedder,
        store,
        sparse_index=sparse_index,
        hybrid_config={"enabled": enabled},
    )


class TestReciprocalRankFusion:
    def test_exact_math(self):
        fused = reciprocal_rank_fusion([[10, 20], [20, 30]], rrf_k=60)
        assert abs(fused[10] - (1 / 61)) < 1e-9
        assert abs(fused[20] - (1 / 62 + 1 / 61)) < 1e-9
        assert abs(fused[30] - (1 / 62)) < 1e-9

    def test_consensus_beats_single_hit(self):
        # id 1 appears mid-list twice, id 0 tops one list: consensus wins.
        fused = reciprocal_rank_fusion([[0, 1, 2], [9, 1, 3]], rrf_k=60)
        assert fused[1] > fused[0]

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}


class TestTokenize:
    def test_lowercase_ascii(self):
        assert tokenize("The Capital OF India!") == ["the", "capital", "of", "india"]

    def test_devanagari_preserved(self):
        tokens = tokenize("भारत की राजधानी नई दिल्ली है।")
        assert "भारत" in tokens and "राजधानी" in tokens

    def test_empty_and_none(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestBm25SparseIndex:
    def test_build_and_search(self):
        idx = Bm25SparseIndex().build(TEXTS)
        assert idx.num_docs == 5
        hits = idx.search("capital New Delhi", top_k=3)
        assert len(hits) >= 1
        assert all(isinstance(i, int) and s > 0.0 for i, s in hits)
        assert hits[0][0] in (1, 4)  # New Delhi / capital city docs rank first lexically

    def test_no_hits_zero_score(self):
        idx = Bm25SparseIndex().build(["alpha beta gamma"])
        # Unrelated tokens must produce no positive-score hits.
        assert idx.search("zzz qqq xxx", top_k=3) == []

    def test_save_load_round_trip(self, tmp_path):
        idx = Bm25SparseIndex().build(TEXTS)
        out = str(tmp_path / "bm25")
        idx.save(out)

        loaded = Bm25SparseIndex()
        assert loaded.load(out) is True
        assert loaded.num_docs == 5
        original = idx.search("financial capital Mumbai", top_k=2)
        restored = loaded.search("financial capital Mumbai", top_k=2)
        assert [i for i, _ in original] == [i for i, _ in restored]

    def test_load_missing_dir(self, tmp_path):
        assert Bm25SparseIndex().load(str(tmp_path / "missing")) is False

    def test_search_before_build(self):
        assert Bm25SparseIndex().search("anything", top_k=3) == []


class TestScoreCandidates:
    def test_matches_search_scores(self, embedder, store):
        qv = embedder.embed_query("capital of India")
        top = store.search(query_vector=qv, top_k=3, language_filter=None)
        ids = [i for i, _ in top]
        scored = dict(store.score_candidates(ids, qv))
        for i, s in top:
            assert abs(scored[i] - s) < 1e-6


class TestHybridRetriever:
    def test_hybrid_strategy_and_cosine_scores(self, embedder, store, sparse):
        r = make_retriever(embedder, store, sparse, enabled=True)
        result = r.search("capital of India", top_k=3, use_cache=False)
        assert result.strategy_used == "hybrid_rrf"
        assert result.cache_hit is False
        assert result.degradations == []
        assert len(result.chunks) == 3
        for c in result.chunks:
            assert 0.0 <= c.score <= 1.0001
            assert c.chunk_id.startswith("c")

    def test_disabled_flag_stays_dense(self, embedder, store, sparse):
        r = make_retriever(embedder, store, sparse, enabled=False)
        result = r.search("capital of India", top_k=3, use_cache=False)
        assert result.strategy_used != "hybrid_rrf"

    def test_language_filter_bypasses_hybrid(self, embedder, store, sparse):
        r = make_retriever(embedder, store, sparse, enabled=True)
        result = r.search(
            "capital of India", top_k=3, use_cache=False, language_filter="en"
        )
        assert result.strategy_used != "hybrid_rrf"
        assert result.degradations == []

    def test_missing_sparse_index_degrades(self, embedder, store):
        r = make_retriever(embedder, store, sparse_index=None, enabled=True)
        result = r.search("capital of India", top_k=3, use_cache=False)
        assert "sparse_leg_unavailable" in result.degradations
        assert result.strategy_used == "hybrid_rrf"

    def test_expired_deadline_skips_sparse_leg(self, embedder, store, sparse):
        r = make_retriever(embedder, store, sparse, enabled=True)
        past = time.perf_counter() - 1.0
        result = r.search(
            "technology hub Bangalore", top_k=2, use_cache=False, deadline=past
        )
        assert "sparse_leg_skipped_deadline" in result.degradations
        assert result.strategy_used == "hybrid_rrf"
        assert len(result.chunks) == 2

    def test_expired_deadline_skips_language_rerank_dense(self, embedder, store):
        r = make_retriever(embedder, store)
        past = time.perf_counter() - 1.0
        result = r.search(
            "financial capital of India",
            top_k=2,
            use_cache=False,
            language_preference="en",
            deadline=past,
        )
        assert "language_rerank_skipped_deadline" in result.degradations
        assert len(result.chunks) == 2

    def test_live_deadline_no_degradation(self, embedder, store, sparse):
        r = make_retriever(embedder, store, sparse, enabled=True)
        future = time.perf_counter() + 5.0
        result = r.search(
            "country in South Asia", top_k=2, use_cache=False, deadline=future
        )
        assert result.degradations == []


def _fake_llm_response(content: str):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


class TestParseAnswerJson:
    def test_plain_json(self):
        parsed = GenerativeGenerator._parse_answer_json(
            '{"answer": "Goa is a state.", "citations": [1, 2], "confidence": 0.92}'
        )
        assert parsed == ("Goa is a state.", [1, 2], 0.92)

    def test_fenced_json(self):
        parsed = GenerativeGenerator._parse_answer_json(
            '```json\n{"answer": "A", "citations": [1], "confidence": 0.5}\n```'
        )
        assert parsed == ("A", [1], 0.5)

    def test_prose_wrapped_json(self):
        parsed = GenerativeGenerator._parse_answer_json(
            'Sure!\n{"answer": "B", "citations": "bad", "confidence": "oops"}'
        )
        assert parsed is not None
        text, citations, confidence = parsed
        assert text == "B"
        assert citations == []
        assert confidence == 0.85

    def test_garbage_returns_none(self):
        assert GenerativeGenerator._parse_answer_json("no structured data here") is None

    def test_empty_answer_with_expect_json(self):
        assert (
            GenerativeGenerator._parse_answer_json('{"answer": "", "citations": [1]}')
            is None
        )


class TestGenerativeCitationContract:
    def _chunks(self):
        return [
            Chunk(chunk_id=f"c{i}", document_id="d", text=t, metadata={"language": "en"})
            for i, t in enumerate(TEXTS[:2])
        ]

    def test_filters_invented_and_duplicate_citations(self):
        gen = GenerativeGenerator(api_key="test-key", model="m")
        content = '{"answer": "New Delhi is the capital.", "citations": [1, 99, 1], "confidence": 1.7}'
        with patch("generation.generative.httpx.post", return_value=_fake_llm_response(content)):
            ans = gen.generate("capital?", self._chunks())
        assert ans is not None
        assert ans.method == "generative"
        assert ans.text == "New Delhi is the capital."
        assert len(ans.citations) == 1
        assert ans.citations[0].chunk_id == "c0"
        assert ans.confidence == 1.0

    def test_all_citations_invented_returns_none(self):
        gen = GenerativeGenerator(api_key="test-key", model="m")
        content = '{"answer": "Claim.", "citations": [7], "confidence": 0.9}'
        with patch("generation.generative.httpx.post", return_value=_fake_llm_response(content)):
            ans = gen.generate("query?", self._chunks())
        assert ans is None

    def test_empty_chunks_short_circuits(self):
        gen = GenerativeGenerator(api_key="test-key", model="m")
        assert gen.generate("query?", []) is None

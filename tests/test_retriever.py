import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from api.schemas import Chunk, RetrievalResult
from retrieval.search import Retriever
from retrieval.embedder import Embedder
from retrieval.vector_store import NumpyVectorStore


class TestRetriever:
    def setup_method(self):
        self.embedder = Embedder()
        self.vector_store = NumpyVectorStore()
        # Build a small test index
        texts = [
            "India is a country in South Asia.",
            "The capital of India is New Delhi.",
            "Mumbai is the financial capital of India.",
            "Bangalore is a technology hub in India.",
            "Delhi is the capital city of India.",
        ]
        vectors = self.embedder.embed_passages(texts, batch_size=5)
        chunks = [
            Chunk(chunk_id=f"c{i}", document_id="d", text=text, metadata={"language": "en"})
            for i, text in enumerate(texts)
        ]
        self.vector_store.add(vectors, chunks)
        self.retriever = Retriever(self.embedder, self.vector_store)

    def test_search_returns_results(self):
        result = self.retriever.search("capital of India", top_k=3)
        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 3
        assert result.cache_hit is False

    def test_search_sorted_by_score(self):
        result = self.retriever.search("capital of India", top_k=5)
        scores = [c.score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_cache(self):
        result1 = self.retriever.search("capital of India", top_k=3, use_cache=True)
        assert result1.cache_hit is False
        result2 = self.retriever.search("capital of India", top_k=3, use_cache=True)
        assert result2.cache_hit is True
        assert len(result2.chunks) == len(result1.chunks)

    def test_search_without_cache(self):
        self.retriever.search("capital of India", top_k=3, use_cache=True)
        result = self.retriever.search("capital of India", top_k=3, use_cache=False)
        assert result.cache_hit is False

    def test_search_language_filter(self):
        # Add Hindi chunks
        hindi_texts = ["भारत की राजधानी नई दिल्ली है।", "मुंबई भारत का वित्तीय केंद्र है।"]
        hindi_vecs = self.embedder.embed_passages(hindi_texts, batch_size=2)
        hindi_chunks = [
            Chunk(chunk_id=f"h{i}", document_id="d", text=text, metadata={"language": "hi"})
            for i, text in enumerate(hindi_texts)
        ]
        self.vector_store.add(hindi_vecs, hindi_chunks)

        result = self.retriever.search("capital of India", top_k=5, language_filter="hi")
        for chunk in result.chunks:
            assert chunk.metadata.get("language") == "hi"

    def test_search_language_preference_reranks(self):
        hindi_texts = ["भारत की राजधानी नई दिल्ली है।", "मुंबई भारत का वित्तीय केंद्र है।"]
        hindi_vecs = self.embedder.embed_passages(hindi_texts, batch_size=2)
        hindi_chunks = [
            Chunk(chunk_id=f"h{i}", document_id="d", text=text, metadata={"language": "hi"})
            for i, text in enumerate(hindi_texts)
        ]
        self.vector_store.add(hindi_vecs, hindi_chunks)

        result = self.retriever.search(
            "capital of India", top_k=3, use_cache=False, language_preference="en"
        )
        assert len(result.chunks) == 3
        assert result.chunks[0].metadata.get("language") == "en"
        scores = [c.score for c in result.chunks]
        assert scores == sorted(scores, reverse=True) or all(
            result.chunks[i].metadata.get("language") == "en"
            for i in range(min(2, len(result.chunks)))
        )

    def test_cache_returns_copies(self):
        r1 = self.retriever.search("capital of India", top_k=3, use_cache=True)
        r1.chunks[0].score = 0.123
        r2 = self.retriever.search("capital of India", top_k=3, use_cache=True)
        assert r2.cache_hit is True
        assert r2.chunks[0].score != 0.123

    def test_get_embedder(self):
        assert self.retriever.get_embedder() is self.embedder

    def test_relevance_to_query(self):
        result = self.retriever.search("capital of India", top_k=1)
        assert len(result.chunks) == 1
        assert "capital" in result.chunks[0].text.lower() or "new delhi" in result.chunks[0].text.lower()

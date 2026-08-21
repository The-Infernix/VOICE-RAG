import pytest
import numpy as np
import tempfile
import os
from api.schemas import Chunk
from retrieval.vector_store import NumpyVectorStore, SemanticCache


class TestSemanticCache:
    def test_miss_empty(self):
        cache = SemanticCache()
        vec = np.ones(384)
        assert cache.get(vec) is None

    def test_hit_above_threshold(self):
        cache = SemanticCache(threshold=0.9)
        vec = np.ones(384) / np.sqrt(384)
        cache.put(vec, "response1")
        result = cache.get(vec)
        assert result == "response1"

    def test_miss_below_threshold(self):
        cache = SemanticCache(threshold=0.99)
        vec1 = np.array([1, 0, 0], dtype=float)
        vec2 = np.array([0, 1, 0], dtype=float)
        cache.put(vec1, "resp1")
        assert cache.get(vec2) is None

    def test_max_size_eviction(self):
        cache = SemanticCache(max_size=3)
        for i in range(5):
            vec = np.zeros(3)
            vec[i % 3] = 1.0
            cache.put(vec, f"resp{i}")
        assert cache.size() == 3

    def test_size(self):
        cache = SemanticCache()
        assert cache.size() == 0
        cache.put(np.array([1.0]), "a")
        assert cache.size() == 1


class TestNumpyVectorStore:
    def test_add_and_size(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(5, 384)
        chunks = [Chunk(chunk_id=f"c{i}", document_id="d", text=f"text{i}") for i in range(5)]
        store.add(vectors, chunks)
        assert store.size() == 5

    def test_add_incremental(self):
        store = NumpyVectorStore()
        v1 = np.random.randn(3, 384)
        c1 = [Chunk(chunk_id=f"c{i}", document_id="d", text=f"t{i}") for i in range(3)]
        store.add(v1, c1)
        v2 = np.random.randn(2, 384)
        c2 = [Chunk(chunk_id=f"c{i+3}", document_id="d", text=f"t{i+3}") for i in range(2)]
        store.add(v2, c2)
        assert store.size() == 5

    def test_search_returns_top_k(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(20, 384)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        chunks = [Chunk(chunk_id=f"c{i}", document_id="d", text=f"text{i}") for i in range(20)]
        store.add(vectors, chunks)
        query = np.ones((1, 384)) / np.sqrt(384)
        results = store.search(query, top_k=5)
        assert len(results) == 5

    def test_search_empty_store(self):
        store = NumpyVectorStore()
        query = np.ones((1, 384))
        results = store.search(query, top_k=5)
        assert results == []

    def test_search_language_filter(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(10, 384)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        chunks = [
            Chunk(chunk_id=f"c{i}", document_id="d", text=f"text{i}",
                  metadata={"language": "hi" if i < 5 else "en"})
            for i in range(10)
        ]
        store.add(vectors, chunks)
        query = np.ones((1, 384)) / np.sqrt(384)
        results = store.search(query, top_k=10, language_filter="hi")
        assert len(results) == 5
        for idx, score in results:
            assert chunks[idx].metadata["language"] == "hi"

    def test_get_chunk(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(3, 384)
        chunks = [Chunk(chunk_id=f"c{i}", document_id="d", text=f"text{i}") for i in range(3)]
        store.add(vectors, chunks)
        chunk = store.get_chunk(1)
        assert chunk.chunk_id == "c1"
        assert chunk.text == "text1"

    def test_language_index_built(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(6, 384)
        chunks = [
            Chunk(chunk_id=f"c{i}", document_id="d", text=f"t{i}",
                  metadata={"language": "hi" if i < 3 else "en"})
            for i in range(6)
        ]
        store.add(vectors, chunks)
        assert "hi" in store.language_index
        assert "en" in store.language_index
        assert len(store.language_index["hi"]) == 3
        assert len(store.language_index["en"]) == 3

    def test_save_and_load(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(5, 384)
        chunks = [Chunk(chunk_id=f"c{i}", document_id="d", text=f"text{i}") for i in range(5)]
        store.add(vectors, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "vectors.npy"))
            assert os.path.exists(os.path.join(tmpdir, "chunks.json"))

            store2 = NumpyVectorStore()
            loaded = store2.load(tmpdir)
            assert loaded is True
            assert store2.size() == 5
            np.testing.assert_array_almost_equal(store2.vectors, vectors)
            assert store2.chunks[0].text == "text0"

    def test_load_nonexistent(self):
        store = NumpyVectorStore()
        loaded = store.load("/nonexistent/path")
        assert loaded is False

    def test_search_sorted_by_score(self):
        store = NumpyVectorStore()
        vectors = np.random.randn(10, 384)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        chunks = [Chunk(chunk_id=f"c{i}", document_id="d", text=f"text{i}") for i in range(10)]
        store.add(vectors, chunks)
        query = np.ones((1, 384)) / np.sqrt(384)
        results = store.search(query, top_k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

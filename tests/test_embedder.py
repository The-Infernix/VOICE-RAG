import pytest
import numpy as np
from retrieval.embedder import Embedder, EmbeddingCache


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache()
        vec = np.array([1.0, 0.0, 0.0])
        cache.put("hello", vec)
        result = cache.get("hello")
        np.testing.assert_array_equal(result, vec)

    def test_get_missing(self):
        cache = EmbeddingCache()
        assert cache.get("missing") is None

    def test_contains(self):
        cache = EmbeddingCache()
        assert not cache.contains("hello")
        cache.put("hello", np.array([1.0]))
        assert cache.contains("hello")

    def test_size(self):
        cache = EmbeddingCache()
        assert cache.size() == 0
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        assert cache.size() == 2

    def test_overwrite(self):
        cache = EmbeddingCache()
        cache.put("key", np.array([1.0]))
        cache.put("key", np.array([2.0]))
        result = cache.get("key")
        assert result[0] == 2.0
        assert cache.size() == 1


class TestEmbedder:
    def test_lazy_load(self):
        e = Embedder()
        assert e._model is None

    def test_dimension(self):
        e = Embedder()
        assert e.dimension == 384

    def test_model_name(self):
        e = Embedder()
        assert e.model_name == "intfloat/multilingual-e5-small"

    def test_embed_single_text(self):
        e = Embedder()
        vec = e.embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (1, 384)

    def test_embed_multiple_texts(self):
        e = Embedder()
        vecs = e.embed(["hello", "world", "test"])
        assert vecs.shape == (3, 384)

    def test_normalized(self):
        e = Embedder()
        vec = e.embed("test query")
        norms = np.linalg.norm(vec, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_query(self):
        e = Embedder()
        vec = e.embed_query("What is India?")
        assert vec.shape == (1, 384)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_passages(self):
        e = Embedder()
        vecs = e.embed_passages(["Passage one", "Passage two"], batch_size=2)
        assert vecs.shape == (2, 384)

    def test_similar_meaning_higher_score(self):
        e = Embedder()
        v1 = e.embed_query("capital of India").flatten()
        v2 = e.embed_query("India ka rajdhani").flatten()
        v3 = e.embed_query("What is the largest ocean?").flatten()
        score_similar = float(v1 @ v2)
        score_different = float(v1 @ v3)
        assert score_similar > score_different

    def test_multilingual_embeddings(self):
        e = Embedder()
        en = e.embed_query("capital of India").flatten()
        hi = e.embed_query("भारत की राजधानी").flatten()
        gu = e.embed_query("ભારતની રાજધાની").flatten()
        score_en_hi = float(en @ hi)
        score_en_gu = float(en @ gu)
        score_hi_gu = float(hi @ gu)
        assert score_en_hi > 0.5
        assert score_en_gu > 0.5
        assert score_hi_gu > 0.5

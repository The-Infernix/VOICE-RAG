from typing import Optional
from api.schemas import Chunk, RetrievalResult
from retrieval.embedder import Embedder, EmbeddingCache
from retrieval.vector_store import NumpyVectorStore, SemanticCache

LANGUAGE_BONUS = 0.03


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: NumpyVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.cache = SemanticCache(threshold=0.95)
        self.embed_cache = EmbeddingCache()

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        use_cache: bool = True,
        language_preference: Optional[str] = None,
    ) -> RetrievalResult:
        query_vec = self.embedder.embed_query(query)

        if use_cache:
            cached = self.cache.get(query_vec)
            if cached is not None:
                return RetrievalResult(
                    chunks=[c.model_copy(deep=True) for c in cached],
                    strategy_used="cached",
                    cache_hit=True,
                )

        prefer = language_preference and not language_filter
        fetch_k = top_k * 3 if prefer else top_k

        results = self.vector_store.search(
            query_vector=query_vec,
            top_k=fetch_k,
            language_filter=language_filter,
        )

        chunks = []
        for idx, score in results:
            chunk = self.vector_store.get_chunk(idx).model_copy(deep=True)
            chunk.score = score
            chunks.append(chunk)

        if prefer and len(chunks) > top_k:
            def adjusted(c: Chunk) -> float:
                bonus = LANGUAGE_BONUS if c.metadata.get("language") == language_preference else 0.0
                return c.score + bonus

            chunks.sort(key=adjusted, reverse=True)
            chunks = chunks[:top_k]

        retrieval = RetrievalResult(
            chunks=chunks,
            strategy_used=chunks[0].metadata.get("strategy", "unknown") if chunks else "none",
            cache_hit=False,
        )

        if use_cache and chunks:
            self.cache.put(query_vec, [c.model_copy(deep=True) for c in chunks])

        return retrieval

    def get_embedder(self) -> Embedder:
        return self.embedder

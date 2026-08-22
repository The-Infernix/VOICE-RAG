import time
from typing import List, Optional
from api.schemas import Chunk, RetrievalResult
from retrieval.embedder import Embedder, EmbeddingCache
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.vector_store import NumpyVectorStore, SemanticCache

LANGUAGE_BONUS = 0.03


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: NumpyVectorStore,
        sparse_index=None,
        hybrid_config: Optional[dict] = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        cfg = hybrid_config or {}
        self.hybrid_enabled = bool(cfg.get("enabled", False))
        self.candidate_k = int(cfg.get("candidate_k", 30))
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self.cache = SemanticCache(threshold=0.95)
        self.embed_cache = EmbeddingCache()

    def _expired(self, deadline: Optional[float]) -> bool:
        return deadline is not None and time.perf_counter() > deadline

    def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        use_cache: bool = True,
        language_preference: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> RetrievalResult:
        query_vec = self.embedder.embed_query(query)

        if use_cache:
            cached = self.cache.get(query_vec)
            if cached is not None:
                return RetrievalResult(
                    chunks=[c.model_copy(deep=True) for c in cached],
                    strategy_used="cached",
                    cache_hit=True,
                    degradations=[],
                )

        prefer = language_preference and not language_filter

        if self.hybrid_active(language_filter):
            result = self._search_hybrid(
                query, query_vec, top_k, prefer, language_preference, deadline
            )
        else:
            result = self._search_dense(
                query_vec, top_k, prefer, language_preference, language_filter, deadline
            )

        if use_cache and result.chunks:
            self.cache.put(query_vec, [c.model_copy(deep=True) for c in result.chunks])
        return result

    def hybrid_active(self, language_filter: Optional[str]) -> bool:
        return self.hybrid_enabled and language_filter is None

    def _search_dense(
        self,
        query_vec,
        top_k: int,
        prefer: bool,
        language_preference: Optional[str],
        language_filter: Optional[str],
        deadline: Optional[float],
    ) -> RetrievalResult:
        degradations: List[str] = []
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
            if self._expired(deadline):
                degradations.append("language_rerank_skipped_deadline")
                chunks = chunks[:top_k]
            else:
                def adjusted(c: Chunk) -> float:
                    bonus = LANGUAGE_BONUS if c.metadata.get("language") == language_preference else 0.0
                    return c.score + bonus

                chunks.sort(key=adjusted, reverse=True)
                chunks = chunks[:top_k]

        return RetrievalResult(
            chunks=chunks,
            strategy_used=chunks[0].metadata.get("strategy", "unknown") if chunks else "none",
            cache_hit=False,
            degradations=degradations,
        )

    def _search_hybrid(
        self,
        query: str,
        query_vec,
        top_k: int,
        prefer: bool,
        language_preference: Optional[str],
        deadline: Optional[float],
    ) -> RetrievalResult:
        degradations: List[str] = []
        candidate_k = max(top_k, self.candidate_k)

        dense_results = self.vector_store.search(query_vector=query_vec, top_k=candidate_k)
        dense_ids = [idx for idx, _ in dense_results]

        sparse_ids: List[int] = []
        if self.sparse_index is None:
            degradations.append("sparse_leg_unavailable")
        elif self._expired(deadline):
            degradations.append("sparse_leg_skipped_deadline")
        else:
            hits = self.sparse_index.search(query, candidate_k)
            sparse_ids = [idx for idx, _ in hits]
            if not sparse_ids:
                degradations.append("sparse_no_hits")

        rankings = [dense_ids] + ([sparse_ids] if sparse_ids else [])
        fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)
        candidates = list(fused.keys())
        cosines = dict(self.vector_store.score_candidates(candidates, query_vec))

        rerank_skipped = False
        if prefer and self._expired(deadline):
            rerank_skipped = True
            degradations.append("language_rerank_skipped_deadline")

        def final_score(idx: int) -> float:
            s = cosines.get(idx, 0.0)
            if (
                prefer
                and not rerank_skipped
                and self.vector_store.get_chunk(idx).metadata.get("language") == language_preference
            ):
                s += LANGUAGE_BONUS
            return s

        candidates.sort(key=final_score, reverse=True)
        candidates = candidates[:top_k]

        chunks = []
        for idx in candidates:
            chunk = self.vector_store.get_chunk(idx).model_copy(deep=True)
            chunk.score = cosines.get(idx, 0.0)
            chunks.append(chunk)

        return RetrievalResult(
            chunks=chunks,
            strategy_used="hybrid_rrf" if chunks else "none",
            cache_hit=False,
            degradations=degradations,
        )

    def get_embedder(self) -> Embedder:
        return self.embedder

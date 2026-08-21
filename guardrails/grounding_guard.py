from typing import List
from api.schemas import Chunk, GuardResult


class GroundingGuard:
    def __init__(self, config: dict = None, embedder=None):
        config = config or {}
        self.token_overlap_threshold = config.get("token_overlap_threshold", 0.231)
        self.embedding_threshold = config.get("embedding_threshold", 0.794)
        self._embedder = embedder

    def check(self, answer: str, chunks: List[Chunk]) -> GuardResult:
        if not answer or not chunks:
            return GuardResult(
                passed=False,
                reason_code="NO_ANSWER_OR_CONTEXT",
                message="Cannot validate grounding without answer and context",
            )

        answer_tokens = set(answer.lower().split())
        chunk_tokens = set()
        for chunk in chunks:
            chunk_tokens.update(chunk.text.lower().split())

        if not answer_tokens:
            return GuardResult(
                passed=False,
                reason_code="EMPTY_ANSWER",
                message="Answer is empty",
            )

        overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)

        if overlap >= self.token_overlap_threshold:
            return GuardResult(
                passed=True,
                reason_code="GROUNDED_TOKEN_OVERLAP",
                message=f"Token overlap: {overlap:.3f}",
                score=overlap,
            )

        return self._check_embedding_grounding(answer, chunks, overlap)

    def _check_embedding_grounding(self, answer: str, chunks: List[Chunk], token_overlap: float) -> GuardResult:
        try:
            if self._embedder is None:
                from retrieval.embedder import Embedder
                self._embedder = Embedder()
            answer_vec = self._embedder.embed_query(answer)

            best_score = 0.0
            for chunk in chunks:
                chunk_vec = self._embedder.embed_query(chunk.text)
                score = float(answer_vec @ chunk_vec.flatten())
                best_score = max(best_score, score)

            if best_score >= self.embedding_threshold:
                return GuardResult(
                    passed=True,
                    reason_code="GROUNDED_EMBEDDING",
                    message=f"Embedding similarity: {best_score:.3f} (token overlap was {token_overlap:.3f})",
                    score=best_score,
                )

            return GuardResult(
                passed=False,
                reason_code="UNGROUNDED_OUTPUT",
                message=f"Answer not grounded: token_overlap={token_overlap:.3f}, embedding_sim={best_score:.3f}",
                score=best_score,
            )
        except Exception:
            return GuardResult(
                passed=False,
                reason_code="GROUNDING_CHECK_FAILED",
                message=f"Could not verify grounding: token_overlap={token_overlap:.3f}",
                score=token_overlap,
            )

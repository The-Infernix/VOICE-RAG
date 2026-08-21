from typing import List, Optional
from api.schemas import Chunk, Answer, Citation
import re


class ExtractiveGenerator:
    def __init__(self, min_retrieval_score: float = 0.5):
        self.min_retrieval_score = min_retrieval_score

    def generate(self, query: str, chunks: List[Chunk]) -> Optional[Answer]:
        if not chunks:
            return None

        top_chunk = chunks[0]
        if top_chunk.score < self.min_retrieval_score:
            return None

        answer_text = self._extract_answer_span(query, top_chunk.text)
        if not answer_text:
            answer_text = self._extract_best_sentence(query, top_chunk.text)

        if not answer_text:
            return None

        citations = [
            Citation(
                chunk_id=c.chunk_id,
                text=c.text[:200],
                score=c.score,
            )
            for c in chunks[:3]
        ]

        confidence = min(top_chunk.score, 1.0)

        return Answer(
            text=answer_text,
            method="extractive",
            citations=citations,
            confidence=confidence,
        )

    def _extract_answer_span(self, query: str, passage: str) -> Optional[str]:
        query_words = set(query.lower().split())
        sentences = re.split(r'(?<=[.!?])\s+', passage)
        if not sentences:
            return None

        best_sentence = None
        best_score = -1
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            score = overlap / max(len(query_words), 1)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_score >= 0.3 and best_sentence:
            return best_sentence.strip()
        return None

    def _extract_best_sentence(self, query: str, passage: str) -> Optional[str]:
        sentences = re.split(r'(?<=[.!?])\s+', passage)
        if not sentences:
            return None
        return sentences[0].strip()

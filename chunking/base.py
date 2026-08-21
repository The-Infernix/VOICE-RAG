from abc import ABC, abstractmethod
from typing import List
from api.schemas import Chunk
import hashlib


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        pass

    def _make_id(self, text: str, idx: int) -> str:
        h = hashlib.md5(f"{text[:100]}_{idx}".encode()).hexdigest()[:12]
        return f"chunk_{h}"


class PassageNativeChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        return [Chunk(
            chunk_id=self._make_id(text, 0),
            document_id=meta.get("document_id", ""),
            text=text.strip(),
            metadata={**meta, "strategy": "passage_native"},
        )]


class FixedChunker(BaseChunker):
    def __init__(self, chunk_size: int = 128, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        words = text.split()
        chunks = []
        start = 0
        idx = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(Chunk(
                chunk_id=self._make_id(chunk_text, idx),
                document_id=meta.get("document_id", ""),
                text=chunk_text,
                metadata={**meta, "strategy": "fixed", "offset": start},
            ))
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks


class SlidingWindowChunker(BaseChunker):
    def __init__(self, chunk_size: int = 128, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        words = text.split()
        chunks = []
        idx = 0
        for start in range(0, len(words), self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                chunk_id=self._make_id(chunk_text, idx),
                document_id=meta.get("document_id", ""),
                text=chunk_text,
                metadata={**meta, "strategy": "sliding", "offset": start},
            ))
            idx += 1
            if end >= len(words):
                break
        return chunks


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 128, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        pieces = self._split_recursive(text, self.separators)
        chunks = []
        idx = 0
        for piece in pieces:
            words = piece.split()
            for start in range(0, len(words), self.chunk_size - self.overlap):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                if chunk_text.strip():
                    chunks.append(Chunk(
                        chunk_id=self._make_id(chunk_text, idx),
                        document_id=meta.get("document_id", ""),
                        text=chunk_text,
                        metadata={**meta, "strategy": "recursive"},
                    ))
                    idx += 1
        return chunks

    def _split_recursive(self, text: str, separators: list) -> List[str]:
        if not separators or len(text.split()) <= self.chunk_size:
            return [text]
        sep = separators[0]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_recursive(text, separators[1:])
        result = []
        for part in parts:
            if len(part.split()) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split_recursive(part, separators[1:]))
        return result


class SemanticChunker(BaseChunker):
    def __init__(self, threshold: float = 0.5, min_chunk_size: int = 20):
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return [Chunk(
                chunk_id=self._make_id(text, 0),
                document_id=meta.get("document_id", ""),
                text=text.strip(),
                metadata={**meta, "strategy": "semantic"},
            )]
        chunks = []
        current_group = [sentences[0]]
        idx = 0
        for i in range(1, len(sentences)):
            prev_words = set(sentences[i - 1].lower().split())
            curr_words = set(sentences[i].lower().split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
            else:
                overlap = 0.0
            if overlap < self.threshold and len(" ".join(current_group).split()) >= self.min_chunk_size:
                chunk_text = " ".join(current_group)
                chunks.append(Chunk(
                    chunk_id=self._make_id(chunk_text, idx),
                    document_id=meta.get("document_id", ""),
                    text=chunk_text,
                    metadata={**meta, "strategy": "semantic"},
                ))
                idx += 1
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])
        if current_group:
            chunk_text = " ".join(current_group)
            chunks.append(Chunk(
                chunk_id=self._make_id(chunk_text, idx),
                document_id=meta.get("document_id", ""),
                text=chunk_text,
                metadata={**meta, "strategy": "semantic"},
            ))
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class MetadataAwareChunker(BaseChunker):
    def __init__(self, use_language: bool = True, use_query_type: bool = True):
        self.use_language = use_language
        self.use_query_type = use_query_type

    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        meta = metadata or {}
        chunk_meta = {**meta, "strategy": "metadata"}
        if self.use_language and "language" in meta:
            chunk_meta["detected_language"] = meta["language"]
        if self.use_query_type and "query_type" in meta:
            chunk_meta["query_type"] = meta["query_type"]
        words = text.split()
        chunk_size = 128
        overlap = 20
        chunks = []
        idx = 0
        for start in range(0, len(words), chunk_size - overlap):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                chunk_id=self._make_id(chunk_text, idx),
                document_id=meta.get("document_id", ""),
                text=chunk_text,
                metadata={**chunk_meta, "offset": start},
            ))
            idx += 1
        return chunks


CHUNKERS = {
    "passage_native": PassageNativeChunker,
    "fixed": FixedChunker,
    "sliding": SlidingWindowChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "metadata": MetadataAwareChunker,
}


def get_chunker(name: str, **kwargs) -> BaseChunker:
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunker: {name}. Available: {list(CHUNKERS.keys())}")
    return CHUNKERS[name](**kwargs)

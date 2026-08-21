import numpy as np
from typing import List, Tuple, Optional
from api.schemas import Chunk
import json
import os


class NumpyVectorStore:
    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []
        self.language_index: dict = {}  # lang -> list of indices

    def add(self, vectors: np.ndarray, chunks: List[Chunk]):
        if self.vectors is None:
            self.vectors = vectors
            self.chunks = chunks
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.chunks.extend(chunks)
        self._build_language_index()

    def _build_language_index(self):
        self.language_index = {}
        for i, chunk in enumerate(self.chunks):
            lang = chunk.metadata.get("language", "en")
            if lang not in self.language_index:
                self.language_index[lang] = []
            self.language_index[lang].append(i)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        language_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []

        if language_filter and language_filter in self.language_index:
            indices = np.array(self.language_index[language_filter])
            sub_vectors = self.vectors[indices]
            scores = sub_vectors @ query_vector.flatten()
            top_k_actual = min(top_k, len(scores))
            top_local = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
            return [(indices[idx], float(scores[idx])) for idx in top_local]
        else:
            scores = self.vectors @ query_vector.flatten()
            top_k_actual = min(top_k, len(scores))
            top_idx = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return [(int(idx), float(scores[idx])) for idx in top_idx]

    def get_chunk(self, idx: int) -> Chunk:
        return self.chunks[idx]

    def size(self) -> int:
        return len(self.chunks)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        if self.vectors is not None:
            np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        chunks_data = [c.model_dump() for c in self.chunks]
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

    def load(self, directory: str) -> bool:
        vectors_path = os.path.join(directory, "vectors.npy")
        chunks_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(vectors_path) or not os.path.exists(chunks_path):
            return False
        self.vectors = np.load(vectors_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = [Chunk(**c) for c in chunks_data]
        self._build_language_index()
        return True


class SemanticCache:
    def __init__(self, threshold: float = 0.95, max_size: int = 1000):
        self.threshold = threshold
        self.max_size = max_size
        self._embeddings: List[np.ndarray] = []
        self._responses: list = []

    def get(self, query_vector: np.ndarray):
        if not self._embeddings:
            return None
        matrix = np.array(self._embeddings)
        scores = matrix @ query_vector.flatten()
        best_idx = int(np.argmax(scores))
        if scores[best_idx] >= self.threshold:
            return self._responses[best_idx]
        return None

    def put(self, query_vector: np.ndarray, response):
        if len(self._embeddings) >= self.max_size:
            self._embeddings.pop(0)
            self._responses.pop(0)
        self._embeddings.append(query_vector.flatten())
        self._responses.append(response)

    def size(self) -> int:
        return len(self._embeddings)

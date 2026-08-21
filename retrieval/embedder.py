import numpy as np
from typing import List, Union


class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self._model = None
        self.dimension = 384

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        prefixed = [f"query: {t}" if i == 0 else f"passage: {t}" for i, t in enumerate(texts)]
        embeddings = self._model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            embeddings = embeddings / norms
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        self._load()
        prefixed = f"query: {query}"
        embedding = self._model.encode([prefixed], convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        embedding = embedding / norms
        return embedding

    def embed_passages(self, passages: List[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        prefixed = [f"passage: {p}" for p in passages]
        embeddings = self._model.encode(
            prefixed, convert_to_numpy=True, show_progress_bar=True,
            batch_size=batch_size
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        embeddings = embeddings / norms
        return embeddings


class EmbeddingCache:
    def __init__(self):
        self._cache = {}

    def get(self, text: str) -> np.ndarray:
        return self._cache.get(text)

    def put(self, text: str, embedding: np.ndarray):
        self._cache[text] = embedding

    def contains(self, text: str) -> bool:
        return text in self._cache

    def size(self) -> int:
        return len(self._cache)

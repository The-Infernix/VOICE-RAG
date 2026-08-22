import json
import os
import re
import time
from typing import List, Tuple

import bm25s

_TOKEN_RE = re.compile(
    r"[\w\u0900-\u0963\u0966-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]+",
    re.UNICODE,
)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class Bm25SparseIndex:
    """BM25 lexical index over chunk texts; ids align with vector-store positions."""

    def __init__(self):
        self._bm25 = None
        self.num_docs = 0

    def build(self, texts: List[str]) -> "Bm25SparseIndex":
        start = time.perf_counter()
        corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens, show_progress=False)
        self.num_docs = len(texts)
        self._build_seconds = time.perf_counter() - start
        return self

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._bm25.save(directory, corpus=None)
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"num_docs": self.num_docs}, f)

    def load(self, directory: str) -> bool:
        if not os.path.isdir(directory):
            return False
        try:
            self._bm25 = bm25s.BM25.load(directory, load_corpus=False)
            meta_path = os.path.join(directory, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.num_docs = int(json.load(f).get("num_docs", 0))
            return True
        except Exception:
            self._bm25 = None
            return False

    def search(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        k = max(1, min(top_k, self.num_docs))
        ids, scores = self._bm25.retrieve([tokens], k=k, show_progress=False, return_as="tuple")
        out = []
        for idx, score in zip(ids[0], scores[0]):
            s = float(score)
            if s > 0.0:
                out.append((int(idx), s))
        return out

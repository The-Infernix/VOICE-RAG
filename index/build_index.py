import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking.base import get_chunker
from retrieval.embedder import Embedder
from retrieval.vector_store import NumpyVectorStore

DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_corpus(limit: int = None) -> list:
    corpus_path = DATA_DIR / "corpus.jsonl"
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            docs.append(json.loads(line))
    return docs


def build_index(
    strategy: str = "passage_native",
    limit: int = None,
    embed_batch_size: int = 2048,
):
    print(f"Loading corpus...")
    docs = load_corpus(limit)
    print(f"Loaded {len(docs)} documents")

    print(f"Chunking with strategy: {strategy}")
    chunker = get_chunker(strategy)
    all_chunks = []

    for doc in docs:
        chunks = chunker.chunk(
            doc["text"],
            metadata=doc["metadata"],
        )
        for chunk in chunks:
            chunk.document_id = doc["document_id"]
        all_chunks.extend(chunks)

    print(f"Generated {len(all_chunks)} chunks")

    print("Embedding chunks...")
    embedder = Embedder()
    texts = [c.text for c in all_chunks]

    all_embeddings = []
    for i in range(0, len(texts), embed_batch_size):
        batch = texts[i:i + embed_batch_size]
        emb = embedder.embed_passages(batch)
        all_embeddings.append(emb)
        print(f"  Embedded {min(i + embed_batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building vector store...")
    store = NumpyVectorStore()
    store.add(embeddings, all_chunks)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(ARTIFACTS_DIR))

    print(f"Index saved to {ARTIFACTS_DIR}")
    print(f"  Vectors: {embeddings.shape}")
    print(f"  Chunks: {len(all_chunks)}")

    stats = {
        "strategy": strategy,
        "total_chunks": len(all_chunks),
        "total_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "languages": {},
    }
    for lang in ["en", "hi", "gu"]:
        lang_chunks = [c for c in all_chunks if c.metadata.get("language") == lang]
        stats["languages"][lang] = len(lang_chunks)

    with open(ARTIFACTS_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats: {json.dumps(stats, indent=2)}")
    return store


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="passage_native")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_index(strategy=args.strategy, limit=args.limit)

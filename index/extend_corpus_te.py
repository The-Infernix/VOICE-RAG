import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from chunking.base import get_chunker
from retrieval.embedder import Embedder
from retrieval.vector_store import NumpyVectorStore

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
LANG = "te"
MAX_PASSAGES = 10000
MAX_QUERIES = 500


def extract_and_append():
    existing_ids = set()
    corpus_path = DATA_DIR / "corpus.jsonl"
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            existing_ids.add(json.loads(line)["document_id"])

    parquet_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename=f"validation/telval.parquet",
        repo_type="dataset",
    )
    df = pq.read_table(parquet_path).to_pandas()
    print(f"Loaded {len(df)} rows for {LANG}")

    seen_passages = set()
    n_passages = 0
    queries = []
    seen_qids = set()

    with open(corpus_path, "a", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {LANG}"):
            if n_passages >= MAX_PASSAGES and len(queries) >= MAX_QUERIES:
                break

            query_id = int(row.get("query_id", 0))
            query = row.get("query", "")
            if query and query_id not in seen_qids and len(queries) < MAX_QUERIES:
                seen_qids.add(query_id)
                queries.append({
                    "query": query,
                    "language": LANG,
                    "query_type": row.get("query_type", ""),
                    "eng_query": row.get("Eng_Query", ""),
                    "query_id": query_id,
                })

            passages_data = row.get("passages", {})
            if not isinstance(passages_data, dict) or n_passages >= MAX_PASSAGES:
                continue

            translated = passages_data.get("Translated_passages", [])
            english = passages_data.get("English_passages", [])
            is_selected = passages_data.get("is_selected", [])

            for i, passage in enumerate(translated):
                if n_passages >= MAX_PASSAGES:
                    break
                if not passage or not isinstance(passage, str) or not passage.strip():
                    continue
                h = hash(passage.strip())
                if h in seen_passages:
                    continue
                seen_passages.add(h)

                doc_id = f"doc_{LANG}_{query_id}_{i}"
                if doc_id in existing_ids:
                    continue
                existing_ids.add(doc_id)

                selected = is_selected[i] if i < len(is_selected) else 0
                eng_passage = english[i] if i < len(english) and isinstance(english[i], str) else ""
                doc = {
                    "document_id": doc_id,
                    "text": passage.strip(),
                    "metadata": {
                        "language": LANG,
                        "query_id": query_id,
                        "query_type": row.get("query_type", ""),
                        "is_selected": bool(selected),
                        "eng_passage": eng_passage,
                        "eng_query": row.get("Eng_Query", ""),
                        "eng_answer": row.get("Eng_Answer", ""),
                        "translated_query": query,
                        "translated_answer": row.get("Answer", ""),
                        "source": "MSMARCO-XI",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_passages += 1

    queries_path = DATA_DIR / "queries.jsonl"
    with open(queries_path, "a", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Appended {n_passages} passages and {len(queries)} queries")
    return n_passages


def extend_index():
    print("\nLoading new Telugu documents...")
    docs = []
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["metadata"]["language"] == LANG:
                docs.append(d)
    print(f"Found {len(docs)} Telugu documents")

    chunker = get_chunker("passage_native")
    all_chunks = []
    for doc in docs:
        for chunk in chunker.chunk(doc["text"], metadata=doc["metadata"]):
            chunk.document_id = doc["document_id"]
            all_chunks.append(chunk)
    print(f"Generated {len(all_chunks)} chunks")

    print("Embedding...")
    embedder = Embedder()
    embedder.embed_query("warmup")
    embeddings = np.vstack([
        embedder.embed_passages([c.text for c in all_chunks[i:i + 2048]])
        for i in range(0, len(all_chunks), 2048)
    ])
    print(f"Embeddings shape: {embeddings.shape}")

    store = NumpyVectorStore()
    loaded = store.load(str(ARTIFACTS_DIR))
    if not loaded:
        raise SystemExit("No existing index found - cannot extend")
    print(f"Existing index: {store.size()} vectors")

    known_ids = {c.chunk_id for c in store.chunks}
    new_vectors, new_chunks = [], []
    for vec, chunk in zip(embeddings, all_chunks):
        if chunk.chunk_id not in known_ids:
            new_vectors.append(vec)
            new_chunks.append(chunk)
    if not new_chunks:
        print("Nothing new to add")
        return
    store.add(np.vstack(new_vectors), new_chunks)
    store.save(str(ARTIFACTS_DIR))
    print(f"Extended index: {store.size()} vectors")

    stats = json.load(open(ARTIFACTS_DIR / "stats.json"))
    stats["total_chunks"] = store.size()
    stats["total_embeddings"] = int(store.vectors.shape[0])
    stats["languages"][LANG] = sum(
        1 for c in store.chunks if c.metadata.get("language") == LANG
    )
    with open(ARTIFACTS_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats updated: {json.dumps(stats['languages'])}")


if __name__ == "__main__":
    extract_and_append()
    extend_index()

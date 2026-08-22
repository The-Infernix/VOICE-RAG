"""Build the BM25 sparse index from the existing chunk artifacts.

Usage: python index/build_sparse.py [--artifacts index/artifacts]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.sparse import Bm25SparseIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=os.path.join("index", "artifacts"))
    args = parser.parse_args()

    chunks_path = os.path.join(args.artifacts, "chunks.json")
    if not os.path.exists(chunks_path):
        print(f"ERROR: {chunks_path} not found. Run index/build_index.py first.")
        sys.exit(1)

    print("Loading chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c.get("text", "") for c in chunks]
    print(f"{len(texts)} chunks loaded.")

    print("Building BM25 index...")
    index = Bm25SparseIndex()
    index.build(texts)
    print(f"Built in {index._build_seconds:.2f}s over {index.num_docs} docs.")

    out_dir = os.path.join(args.artifacts, "bm25")
    index.save(out_dir)
    size_mb = sum(
        os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
    ) / (1024 * 1024)
    print(f"Saved to {out_dir} ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()

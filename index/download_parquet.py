import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
LANGUAGES = {
    "hi": "hinval",
    "gu": "gujval",
}


def download_parquet():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "corpus.jsonl"
    seen_passages = set()
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for lang, filename in LANGUAGES.items():
            print(f"\nDownloading {filename}.parquet...")
            parquet_path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=f"validation/{filename}.parquet",
                repo_type="dataset",
            )
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            print(f"  Loaded {len(df)} rows for {lang}")

            count = 0
            for _, row in tqdm(df.iterrows(), desc=f"Processing {lang}", total=len(df)):
                query = row.get("query", "")
                answer = row.get("Answer", "")
                query_type = row.get("query_type", "")
                query_id = row.get("query_id", 0)
                eng_query = row.get("Eng_Query", "")
                eng_answer = row.get("Eng_Answer", "")

                passages_data = row.get("passages", {})
                if isinstance(passages_data, dict):
                    translated = passages_data.get("Translated_passages", [])
                    english = passages_data.get("English_passages", [])
                    is_selected = passages_data.get("is_selected", [])
                else:
                    continue

                for i, passage in enumerate(translated):
                    if not passage or not isinstance(passage, str) or not passage.strip():
                        continue

                    passage_hash = hash(passage.strip())
                    if passage_hash in seen_passages:
                        continue
                    seen_passages.add(passage_hash)

                    selected = is_selected[i] if i < len(is_selected) else 0
                    eng_passage = english[i] if i < len(english) and isinstance(english[i], str) else ""

                    doc = {
                        "document_id": f"doc_{lang}_{query_id}_{i}",
                        "text": passage.strip(),
                        "metadata": {
                            "language": lang,
                            "query_id": query_id,
                            "query_type": query_type,
                            "is_selected": bool(selected),
                            "eng_passage": eng_passage,
                            "eng_query": eng_query,
                            "eng_answer": eng_answer,
                            "translated_query": query,
                            "translated_answer": answer,
                            "source": "MSMARCO-XI",
                        },
                    }

                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total += 1

                count += 1

            print(f"  {lang}: {count} queries processed, {total} total passages")

    print(f"\nTotal passages: {total}")
    print(f"Corpus saved to: {output_path}")
    return output_path


def extract_queries(output_path: Path = None, max_per_lang: int = 300):
    if output_path is None:
        output_path = DATA_DIR / "queries.jsonl"

    queries = []
    seen = set()

    for lang, filename in LANGUAGES.items():
        print(f"\nExtracting queries from {filename}...")
        parquet_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=f"validation/{filename}.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        count = 0
        for _, row in df.iterrows():
            if count >= max_per_lang:
                break

            query = row.get("query", "")
            query_id = row.get("query_id", 0)
            if not query or query_id in seen:
                continue
            seen.add(query_id)

            queries.append({
                "query": query,
                "language": lang,
                "query_type": row.get("query_type", ""),
                "eng_query": row.get("Eng_Query", ""),
                "query_id": query_id,
            })
            count += 1

        print(f"  {lang}: {count} queries extracted")

    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nTotal queries: {len(queries)}")
    print(f"Saved to: {output_path}")
    return queries


if __name__ == "__main__":
    download_parquet()
    extract_queries()

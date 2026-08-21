import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tqdm import tqdm

LANG_MAP = {
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def build_corpus(max_per_lang: int = 10000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "corpus.jsonl"
    seen_passages = set()
    total = 0

    print(f"Loading dataset (streaming)...")
    dataset = load_dataset("ai4bharat/MSMARCO-XI", split="validation", streaming=True)

    print(f"Building corpus: hi+gu (translated) + en (original) (max {max_per_lang} each)")
    lang_counts = {"hi": 0, "gu": 0, "en": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Scanning validation set", total=None):
            target_lang = example.get("target_lang", "")

            matched_lang = None
            for lang, code in LANG_MAP.items():
                if target_lang == code:
                    matched_lang = lang
                    break

            if matched_lang is None:
                continue

            query = example.get("query", "")
            answer = example.get("Answer", "")
            query_type = example.get("query_type", "")
            query_id = example.get("query_id", 0)
            eng_query = example.get("Eng_Query", "")
            eng_answer = example.get("Eng_Answer", "")

            passages_data = example.get("passages", {})
            translated_passages = passages_data.get("Translated_passages", [])
            english_passages = passages_data.get("English_passages", [])
            is_selected = passages_data.get("is_selected", [])

            for i, passage in enumerate(translated_passages):
                if not passage or not passage.strip():
                    continue

                passage_hash = hash(passage.strip())
                if passage_hash in seen_passages:
                    continue
                seen_passages.add(passage_hash)

                if lang_counts[matched_lang] >= max_per_lang:
                    continue

                selected = is_selected[i] if i < len(is_selected) else 0
                eng_passage = english_passages[i] if i < len(english_passages) else ""

                doc = {
                    "document_id": f"doc_{matched_lang}_{query_id}_{i}",
                    "text": passage.strip(),
                    "metadata": {
                        "language": matched_lang,
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
                lang_counts[matched_lang] += 1

            if lang_counts["en"] < max_per_lang:
                for i, eng_passage in enumerate(english_passages):
                    if not eng_passage or not eng_passage.strip():
                        continue

                    eng_hash = hash(eng_passage.strip())
                    if eng_hash in seen_passages:
                        continue
                    seen_passages.add(eng_hash)

                    if lang_counts["en"] >= max_per_lang:
                        break

                    selected = is_selected[i] if i < len(is_selected) else 0

                    doc = {
                        "document_id": f"doc_en_{query_id}_{i}",
                        "text": eng_passage.strip(),
                        "metadata": {
                            "language": "en",
                            "query_id": query_id,
                            "query_type": query_type,
                            "is_selected": bool(selected),
                            "eng_passage": eng_passage.strip(),
                            "eng_query": eng_query,
                            "eng_answer": eng_answer,
                            "translated_query": query,
                            "translated_answer": answer,
                            "source": "MSMARCO-XI",
                        },
                    }

                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total += 1
                    lang_counts["en"] += 1

            if all(v >= max_per_lang for v in lang_counts.values()):
                print(f"\nAll languages reached limit of {max_per_lang}. Stopping.")
                break

    print(f"\nCorpus stats:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count} passages")
    print(f"  Total passages: {total}")
    print(f"  Saved to: {output_path}")
    return output_path


def extract_queries(output_path: Path = None, max_per_lang: int = 500):
    if output_path is None:
        output_path = DATA_DIR / "queries.jsonl"

    print(f"Loading dataset for queries (streaming)...")
    dataset = load_dataset("ai4bharat/MSMARCO-XI", split="validation", streaming=True)

    queries = []
    seen = set()
    lang_counts = {"hi": 0, "gu": 0, "en": 0}

    for example in dataset:
        target_lang = example.get("target_lang", "")
        matched_lang = None
        for lang, code in LANG_MAP.items():
            if target_lang == code:
                matched_lang = lang
                break

        if matched_lang is None:
            continue

        query = example.get("query", "")
        query_id = example.get("query_id", 0)
        eng_query = example.get("Eng_Query", "")

        if matched_lang in lang_counts and lang_counts[matched_lang] < max_per_lang:
            if query and query_id not in seen:
                seen.add(query_id)
                queries.append({
                    "query": query,
                    "language": matched_lang,
                    "query_type": example.get("query_type", ""),
                    "eng_query": eng_query,
                    "query_id": query_id,
                })
                lang_counts[matched_lang] += 1

        if lang_counts["en"] < max_per_lang and eng_query and eng_query not in seen:
            seen.add(eng_query)
            queries.append({
                "query": eng_query,
                "language": "en",
                "query_type": example.get("query_type", ""),
                "eng_query": eng_query,
                "query_id": query_id,
            })
            lang_counts["en"] += 1

        if all(v >= max_per_lang for v in lang_counts.values()):
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Extracted {len(queries)} queries to {output_path}")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")
    return queries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-lang", type=int, default=10000)
    parser.add_argument("--queries-only", action="store_true")
    args = parser.parse_args()

    if args.queries_only:
        extract_queries(max_per_lang=args.max_per_lang)
    else:
        build_corpus(max_per_lang=args.max_per_lang)
        extract_queries(max_per_lang=min(args.max_per_lang, 500))

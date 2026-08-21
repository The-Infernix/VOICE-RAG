import json, time, httpx, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

queries = []
with open(r'C:\HH GOA\voice-rag-hhgoa\data\queries.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        queries.append(json.loads(line))

# Sample 100 queries (33 per lang)
sampled = []
by_lang = {'en': [], 'hi': [], 'gu': []}
for q in queries:
    lang = q.get('language', 'en')
    if lang in by_lang and len(by_lang[lang]) < 33:
        by_lang[lang].append(q)
        sampled.append(q)

print(f"Running benchmark on {len(sampled)} queries...")

latencies = []
for i, q in enumerate(sampled):
    r = httpx.post('http://localhost:7860/ask', json={
        'query': q['query'],
        'lang': q.get('language'),
        'top_k': 5,
        'allow_generative': False,
    }, timeout=30.0)
    data = r.json()
    core_ms = data.get('latency', {}).get('total_core_ms', 0)
    latencies.append({
        'core_ms': core_ms,
        'lang': q.get('language', 'en'),
        'status': data.get('status', 'error'),
    })
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(sampled)} done...")

core_ms_all = [l['core_ms'] for l in latencies]
p50 = np.percentile(core_ms_all, 50)
p70 = np.percentile(core_ms_all, 70)
p90 = np.percentile(core_ms_all, 90)
p95 = np.percentile(core_ms_all, 95)
p100 = max(core_ms_all)
within_200 = sum(1 for x in core_ms_all if x < 200) / len(core_ms_all) * 100

print(f"\n=== BENCHMARK RESULTS ({len(sampled)} queries) ===")
print(f"P50:  {p50:.1f}ms")
print(f"P70:  {p70:.1f}ms")
print(f"P90:  {p90:.1f}ms")
print(f"P95:  {p95:.1f}ms")
print(f"P100: {p100:.1f}ms")
print(f"Within 200ms: {within_200:.1f}%")

for lang in ['en', 'hi', 'gu']:
    lang_lats = [l['core_ms'] for l in latencies if l['lang'] == lang]
    if lang_lats:
        print(f"\n  {lang}: n={len(lang_lats)} P50={np.percentile(lang_lats,50):.1f}ms P100={max(lang_lats):.1f}ms")

statuses = {}
for l in latencies:
    statuses[l['status']] = statuses.get(l['status'], 0) + 1
print(f"\nBy status: {statuses}")

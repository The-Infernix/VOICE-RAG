import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8')

# Test English
r = httpx.post('http://localhost:7860/ask', json={'query': 'What is the capital of India', 'top_k': 5, 'allow_generative': False}, timeout=60.0)
data = r.json()
print('=== ENGLISH QUERY ===')
print('Status:', data['status'])
print('Detected lang:', data.get('detected_language',''))
if data.get('answer'):
    print('Method:', data['answer']['method'])
    print('Confidence:', data['answer']['confidence'])
    print('Answer:', data['answer']['text'][:300])
else:
    print('Refusal:', data.get('refusal_reason',''))
lat = data.get('latency',{})
print(f"Core: {lat.get('total_core_ms',0):.1f}ms  embed: {lat.get('embed_ms',0):.1f}ms  retrieve: {lat.get('retrieve_ms',0):.1f}ms")

# Test Hindi
r2 = httpx.post('http://localhost:7860/ask', json={'query': '\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094d\u092f\u093e \u0939\u0948', 'lang': 'hi', 'top_k': 5, 'allow_generative': False}, timeout=60.0)
data2 = r2.json()
print('\n=== HINDI QUERY ===')
print('Status:', data2['status'])
print('Detected lang:', data2.get('detected_language',''))
if data2.get('answer'):
    print('Method:', data2['answer']['method'])
    print('Answer:', data2['answer']['text'][:300])
else:
    print('Refusal:', data2.get('refusal_reason',''))
lat2 = data2.get('latency',{})
print(f"Core: {lat2.get('total_core_ms',0):.1f}ms  embed: {lat2.get('embed_ms',0):.1f}ms  retrieve: {lat2.get('retrieve_ms',0):.1f}ms")

# Test Gujarati
r3 = httpx.post('http://localhost:7860/ask', json={'query': '\u0928\u0930\u0947\u0902\u0926\u094d\u0930 \u092e\u094b\u091c\u0940 \u0939\u0948', 'lang': 'gu', 'top_k': 5, 'allow_generative': False}, timeout=60.0)
data3 = r3.json()
print('\n=== GUJARATI QUERY ===')
print('Status:', data3['status'])
print('Detected lang:', data3.get('detected_language',''))
if data3.get('answer'):
    print('Method:', data3['answer']['method'])
    print('Answer:', data3['answer']['text'][:300])
else:
    print('Refusal:', data3.get('refusal_reason',''))
lat3 = data3.get('latency',{})
print(f"Core: {lat3.get('total_core_ms',0):.1f}ms  embed: {lat3.get('embed_ms',0):.1f}ms  retrieve: {lat3.get('retrieve_ms',0):.1f}ms")

# Health
rh = httpx.get('http://localhost:7860/health', timeout=10.0)
print('\n=== HEALTH ===')
print(json.dumps(rh.json(), indent=2))

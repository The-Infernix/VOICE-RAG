import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8')

audio_path = r'C:\HH GOA\voice-rag-hhgoa\test_audio.wav'

with open(audio_path, 'rb') as f:
    audio_bytes = f.read()

print(f"Audio size: {len(audio_bytes)} bytes")
print("Sending to /ask-voice...")

r = httpx.post(
    'http://localhost:7860/ask-voice',
    files={'audio': ('test.wav', audio_bytes, 'audio/wav')},
    data={'top_k': 5, 'allow_generative': False},
    timeout=60.0,
)

data = r.json()
print(f"\nStatus: {data['status']}")
print(f"Detected lang: {data.get('detected_language','')}")

for stage in data.get('stages', []):
    print(f"  {stage['stage']}: {stage['latency_ms']:.1f}ms ({stage['status']})")
    if stage.get('details') and stage['stage'] == 'stt':
        print(f"    transcript: {stage['details'].get('transcript','(none)')}")

if data.get('answer'):
    print(f"\nAnswer: {data['answer']['text'][:200]}")
    print(f"Method: {data['answer']['method']}")
else:
    print(f"\nRefusal: {data.get('refusal_reason','')}")

lat = data.get('latency', {})
print(f"\nLatency breakdown:")
print(f"  STT:    {lat.get('stt_ms',0):.1f}ms")
print(f"  Embed:  {lat.get('embed_ms',0):.1f}ms")
print(f"  Retr:   {lat.get('retrieve_ms',0):.1f}ms")
print(f"  Total:  {lat.get('total_e2e_ms',0):.1f}ms")

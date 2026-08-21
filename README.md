<div align="center">

# VOICE RAG

### Ask out loud. Get grounded answers — in English, Hindi, Gujarati, or Telugu.

*A voice-first multilingual Retrieval-Augmented Generation system with hard refusal guardrails and a sub-50ms retrieval core.*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0096CE?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-151%20passing-brightgreen?style=for-the-badge&logo=pytest)](tests/)
[![P50](https://img.shields.io/badge/Core%20P50-56%20ms-success?style=for-the-badge)](bench/results/summary.json)
[![SLA](https://img.shields.io/badge/%3C200%20ms%20SLA-99.95%25%20of%202000%20queries-blue?style=for-the-badge)](bench/results/summary.json)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Features](#-features) · [Architecture](#-architecture) · [Performance](#-performance) · [Quickstart](#-quickstart) · [API](#-api-reference) · [Guardrails](#-guardrails)

<img src="docs/ui.png" alt="VOICE RAG interface" width="100%">

</div>

---

## Overview

**VOICE RAG** answers spoken or typed questions strictly from a curated multilingual corpus — and **refuses to answer when the corpus doesn't contain the answer**.

Ask *"Who is the Prime Minister of India?"* and most RAG demos will hallucinate an answer from whatever passage looks closest. VOICE RAG measures retrieval confidence against calibrated per-language floors and says:

> `Refused: Top retrieval score 0.779 below relevance floor 0.840`

Ask something the corpus *does* cover — in English, Hindi, Gujarati, or Telugu — and you get a grounded, cited answer with a full latency breakdown and the exact evidence chain behind it.

| | |
|---|---|
| **Languages** | English, Hindi (हिन्दी), Gujarati (ગુજરાતી), Telugu (తెలుగు) |
| **Input modes** | Speech (hold-to-talk) and text |
| **Corpus** | 40,000 passages · MSMARCO-XI · 10k per language |
| **Embeddings** | `intfloat/multilingual-e5-small` · 384-dim |
| **STT** | Sarvam AI `saaras:v3` |
| **Answering** | Extractive (default, instant) · LLM via OpenRouter (opt-in) |

---

## Features

<div align="center">

<img src="docs/demo.gif" alt="VOICE RAG in action — hold to speak, watch the pipeline run" width="85%">

<sub><b>Hold to speak → transcribe → retrieve → grounded answer, with the full evidence chain.</b></sub>

</div>

- **Voice-native UX** — hold <kbd>Space</kbd> (or the mic button) to speak; audio is captured, encoded to 16 kHz mono WAV in-browser, transcribed by Sarvam, and answered end-to-end.
- **Knows when not to answer** — relevance floors calibrated from measured score distributions refuse off-corpus questions instead of fabricating answers.
- **Language-aware retrieval** — script-based query detection (Latin / Devanagari / Gujarati) with soft same-language reranking, so an English question never surfaces a Gujarati tweet.
- **Grounded by construction** — every answer passes a grounding guard (token overlap → embedding similarity fallback); ungrounded output never reaches the user.
- **Full transparency** — evidence drawer with ranked passages + scores, stage-by-stage latency modal, and a *why-this-answer* chain showing every guard decision.
- **Blazing fast core** — P50 **56 ms**, P90 **69 ms**, 99.95% of queries under 200 ms across 2,000 benchmark queries.
- **Zero-build frontend** — vanilla JS SPA served directly by FastAPI. No node_modules, no bundler.
- **151 passing tests** — unit coverage for chunking, embedding, retrieval, guards, generation, orchestration, schemas, and language detection.
---

## Architecture

```mermaid
flowchart TB

    %% =========================
    %% CLIENT
    %% =========================
    subgraph CLIENT["01 · USER INTERFACE"]
        direction LR
        MIC["🎙 Voice Input"]
        TXT["⌨ Text Input"]
        WAV["16 kHz Mono PCM<br/>Audio Encoder"]
        UI["Evidence · History<br/>Why This Answer · Latency"]
        
        MIC --> WAV
    end

    %% =========================
    %% QUERY UNDERSTANDING
    %% =========================
    subgraph UNDERSTAND["02 · QUERY UNDERSTANDING"]
        STT["Sarvam Saaras v3<br/>Speech-to-Text"]
        IG{"Input Guard<br/>Injection · Unsafe · Length"}
        EMB["Multilingual E5-small<br/>Query Embedding"]
    end

    %% =========================
    %% RETRIEVAL
    %% =========================
    subgraph RETRIEVAL["03 · KNOWLEDGE RETRIEVAL"]
        VS["MSMARCO-XI Knowledge Base<br/>40K Passages · 384-dim Vectors"]
        SEARCH["Semantic Vector Search<br/>NumPy Index"]
        RG{"Relevance Guard<br/>Language-aware Thresholds"}
    end

    %% =========================
    %% GENERATION
    %% =========================
    subgraph GENERATION["04 · GROUNDED GENERATION"]
        GEN["Extractive Answer Generator<br/>LLM Generation · Optional"]
        GG{"Grounding Guard<br/>Lexical + Embedding Verification"}
    end

    %% =========================
    %% OUTPUT
    %% =========================
    subgraph OUTPUT["05 · VERIFIED RESPONSE"]
        ANS["Answer"]
        EVID["Retrieved Evidence<br/>Citations + Relevance"]
        TRACE["Pipeline Trace<br/>Latency + Stage Results"]
    end

    %% =========================
    %% INPUT FLOWS
    %% =========================
    WAV --> STT
    TXT --> IG
    STT --> IG

    %% =========================
    %% QUERY FLOW
    %% =========================
    IG -->|Pass| EMB
    IG -->|Reject| REF1["Safe Refusal"]

    EMB --> SEARCH
    VS --> SEARCH
    SEARCH --> RG

    RG -->|Relevant| GEN
    RG -->|Insufficient Evidence| REF2["Evidence-based Refusal"]

    GEN --> GG

    GG -->|Grounded| ANS
    GG -->|Unsupported| REF3["Grounding Refusal"]

    %% =========================
    %% RESPONSE
    %% =========================
    ANS --> EVID
    ANS --> TRACE
    EVID --> UI
    TRACE --> UI

    REF1 --> UI
    REF2 --> UI
    REF3 --> UI

    %% =========================
    %% STYLES
    %% =========================
    classDef client fill:#111827,stroke:#60a5fa,color:#fff,stroke-width:2px
    classDef process fill:#101827,stroke:#34d399,color:#fff,stroke-width:2px
    classDef guard fill:#18120b,stroke:#f59e0b,color:#fff,stroke-width:2px
    classDef knowledge fill:#111827,stroke:#a78bfa,color:#fff,stroke-width:2px
    classDef output fill:#0f172a,stroke:#22d3ee,color:#fff,stroke-width:2px
    classDef refusal fill:#1f1111,stroke:#ef4444,color:#fff,stroke-width:1.5px

    class MIC,TXT,WAV,UI client
    class STT,EMB,GEN,SEARCH process
    class IG,RG,GG guard
    class VS knowledge
    class ANS,EVID,TRACE output
    class REF1,REF2,REF3 refusal
```

### Stage-by-stage cost (measured medians, n = 2000)

| Stage | P50 latency | What happens |
|---|---:|---|
| Input guard | 0.02 ms | Injection / unsafe content / length checks |
| Embed query | 23.67 ms | E5 `query:` prefix → 384-dim vector |
| Vector search | 30.74 ms | Cosine similarity over 40k passages + language rerank |
| Relevance guard | 0.01 ms | Top score vs calibrated floor |
| Answer generation | 0.09 ms | Best-sentence extraction from top passage |
| Grounding guard | 0.17 ms | Token overlap, embedding fallback if needed |
| **Total core** | **55.75 ms** | Target: < 200 ms |

---

## Performance

Benchmarked on **2,000 corpus queries** (500 per language, balanced sample, seed-fixed) against the prebuilt index, running the full pipeline in-process.

| Metric | All | English | Hindi | Gujarati | Telugu |
|---|---:|---:|---:|---:|---:|
| P50 | 55.8 ms | 54.9 ms | 56.0 ms | 56.7 ms | 55.3 ms |
| P70 | 61.1 ms | — | — | — | — |
| P90 | 69.4 ms | — | — | — | — |
| P100 | 273.2 ms * | 100.8 ms | 273.2 ms * | 178.5 ms | 109.5 ms |
| Within 200 ms | 99.95% | 100% | 99.8% | 100% | 100% |

<sub>\* A single OS-scheduling outlier among 2,000 runs; the next-worst query across all languages was 178 ms.</sub>

**Guardrail activity in-benchmark:** 1,893 answered · 107 refused (weak-match queries correctly declined rather than answered with junk).

Reproduce it yourself:

```bash
python bench/run_bench.py --limit 2000   # full per-query results in bench/results/
```

<sub>The embedding model is warmed before measurement; per-query raw data is committed at `bench/results/results.json`.</sub>

End-to-end voice round-trip: **~1.7 s** (≈1 s Sarvam STT + ≈56 ms pipeline).

---

## Quickstart

### Prerequisites

- Python 3.12+
- A [Sarvam AI](https://www.sarvam.ai/) API key (for speech-to-text)
- *(Optional)* An [OpenRouter](https://openrouter.ai/) API key to enable LLM-generated answers

### Setup

```bash
git clone https://github.com/The-Infernix/VOICE-RAG.git
cd VOICE-RAG

python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

```env
SARVAM_API_KEY=your_sarvam_key        # required for voice
OPENROUTER_API_KEY=your_openrouter_key  # optional, enables generative mode
```

### Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860** — the SPA is served at the root.

> The prebuilt index (`index/artifacts/`, 40k × 384-dim vectors) ships with the repo, so first run needs no indexing step. To rebuild from scratch: `python index/build_corpus.py` then `python index/build_index.py`. To add another language from MSMARCO-XI, `index/extend_corpus_te.py` is a working template.

---

## API Reference

### `POST /api/v1/query` — text question

```bash
curl -X POST http://localhost:7860/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How much does an Xbox 360 cost?", "top_k": 5}'
```

<details>
<summary><b>Example response</b></summary>

```json
{
  "status": "success",
  "query": "How much does an Xbox 360 cost?",
  "detected_language": "en",
  "answer": {
    "text": "On average, the Xbox 360 cost will be anywhere from $80 to as much as $250 depending on the model purchased.",
    "method": "extractive",
    "confidence": 0.933,
    "citations": [
      { "chunk_id": "doc_en_331047_0", "score": 0.933, "text": "..." }
    ]
  },
  "latency": {
    "embed_ms": 12.6,
    "retrieve_ms": 11.3,
    "answer_ms": 0.07,
    "total_core_ms": 24.0
  },
  "stages": [ ... ]
}
```

</details>

**Request fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | — | Required, ≤ 500 chars |
| `lang` | string | auto | Force `en` / `hi` / `gu` (hard filter) |
| `top_k` | int | 10 | Passages retrieved |
| `allow_generative` | bool | `false` | Opt-in LLM answering |
| `use_cache` | bool | `true` | Semantic query cache |
| `debug` | bool | `false` | Include full retrieval context |

### `POST /api/v1/voice/query` — spoken question

```bash
curl -X POST http://localhost:7860/api/v1/voice/query \
  -F "audio=@question.wav" \
  -F "top_k=5"
```

Accepts WAV audio (16 kHz mono recommended). Returns the same envelope as `/api/v1/query`, plus the transcript in `query` and detected language.

### `GET /health`

```json
{ "status": "ok", "index_size": 30000, "providers": { "sarvam": true, "openrouter": true } }
```

Legacy aliases: `POST /ask`, `POST /ask-voice`.

---

## Guardrails

Three independent guards wrap the pipeline. Every decision is surfaced in the response's `stages` trace and the UI's *why-this-answer* chain.

### 1. Input Guard
Rejects empty/oversized queries, common prompt-injection patterns (*"ignore previous instructions"*), unsafe payloads (`<script>`, shell commands), and unsupported languages — before any compute is spent.

### 2. Relevance Guard — the refusal engine

Thresholds were **calibrated empirically**: top-1 cosine scores were measured for known-relevant queries (P25–P50 ≈ 0.84–0.86) and off-corpus junk (≈ 0.79–0.85), then floors were set at the separation point.

| Language | Floor | Low-confidence band |
|---|---:|---|
| English | 0.840 | +0.03 |
| Hindi | 0.850 | +0.03 |
| Gujarati | 0.840 | +0.03 |
| Telugu | 0.840 | +0.03 |

```
Q: Who is the Prime Minister of India?   → REFUSED  (0.779 < 0.840)
Q: Who won the FIFA World Cup 2022?      → REFUSED  (0.796 < 0.840)
Q: How much does an Xbox 360 cost?       → answered ($80–$250, conf 0.933)
```

### 3. Grounding Guard
Answers must be traceable to retrieved text: token-overlap ≥ 0.231 passes instantly; otherwise an embedding-similarity check (≥ 0.794) runs. Ungrounded generations are replaced with the extractive answer or refused outright.

---

## Multilingual Pipeline

1. **Detection** — Unicode script analysis classifies the query (Telugu block → `te`, Gujarati block → `gu`, Devanagari → `hi`, else `en`). Voice queries inherit Sarvam's language code, normalized to base codes.
2. **Retrieval** — explicit `lang` acts as a hard filter; auto-detected language applies a soft +0.03 rerank bonus toward same-language passages over a 3× candidate pool.
3. **Enforcement** — the *query's* language (not the top chunk's) selects the relevance floor, so cross-language junk can't sneak through under a borrowed threshold.

```
Q: కార్పొరేషన్ అంటే ఏమిటి?    → detected te → answered from Telugu corpus
Q: ભારતની રાજધાની શું છે?   → detected gu → Gujarati passages preferred
Q: भारत की राजधानी क्या है?  → detected hi → honest refusal (not in corpus)
```

---

## Frontend

A dependency-free vanilla JS single-page app (served by FastAPI itself):

- **Hold-to-talk** — press-and-hold <kbd>Space</kbd> or the mic button; release to send
- **Live states** — idle → listening → understanding → retrieving → answer / refused / no-evidence / error
- **Canvas visuals** — reactive orb + waveform driven by real analyser data
- **Evidence drawer** — every retrieved passage with rank, score, and metadata
- **Latency modal** — per-stage timing bars straight from the API's stage traces
- **Why-this-answer** — the guard decision chain for the current response
- **History** — recent Q&A persisted in `localStorage`
- **Technical mode** — toggle raw debug payloads for demos and grading

---

## Project Structure

```
VOICE-RAG/
├── api/
│   ├── main.py              # FastAPI app, endpoints, static mount
│   └── schemas.py           # Pydantic request/response models
├── pipeline/
│   └── orchestrator.py      # Stage sequencing, tracing, refusals
├── retrieval/
│   ├── embedder.py          # E5 wrapper (+ embedding cache)
│   ├── vector_store.py      # NumPy index + semantic cache
│   ├── search.py            # Retriever: search, rerank, caching
│   └── lang_detect.py       # Script-based language detection
├── guardrails/
│   ├── input_guard.py       # Injection / safety / validation
│   ├── relevance_guard.py   # Per-language refusal floors
│   └── grounding_guard.py   # Overlap + embedding grounding
├── generation/
│   ├── extractive.py        # Sentence extraction (default)
│   └── generative.py        # OpenRouter LLM (opt-in)
├── stt/
│   └── sarvam.py            # Sarvam saaras:v3 client
├── chunking/
│   └── base.py              # Chunking strategies
├── index/
│   ├── build_corpus.py      # MSMARCO-XI → corpus.jsonl
│   ├── build_index.py       # Embed + persist artifacts
│   ├── extend_corpus_te.py  # Incremental language-extension template
│   └── artifacts/           # Prebuilt: chunks.json, vectors.npy, stats.json
├── frontend/                # Zero-build SPA (HTML/CSS/JS)
├── bench/                   # Latency harness + results
├── tests/                   # 151 pytest cases
└── config.yaml              # Models, thresholds, tuning
```

---

## Testing

```bash
pytest tests/ -q
```

```
151 passed
```

Covers chunking, embedding math, vector-store sorting, cache isolation, all three guards (pass/refuse/edge cases), extractive + generative generation, orchestrator flows including refusal paths, schema validation, and language detection.

---

## Known Limitations

Honesty is a feature. Current boundaries:

- **Corpus-bounded knowledge** — the system only knows what MSMARCO-XI contains. Questions like *"What is the capital of India?"* have zero supporting passages and are correctly refused rather than answered from world knowledge.
- **Borderline similarity zone** — a small band (~0.84–0.86) overlaps between weak-relevant and strong-junk retrievals; per-language floors trade a little recall for a lot of precision.
- **Voice adds ~1 s** — Sarvam STT dominates end-to-end time; the RAG core itself stays under 50 ms.
- **Generative mode is opt-in** — external LLM calls take multiple seconds and are disabled by default to protect the latency budget.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic v2 |
| Embeddings | `intfloat/multilingual-e5-small` (384-dim) |
| Vector store | NumPy cosine index + semantic query cache |
| Speech | Sarvam AI `saaras:v3` |
| LLM (opt-in) | OpenRouter chat completions |
| Frontend | Vanilla JS, Canvas API, Web Audio/MediaRecorder |
| Testing | Pytest (149 cases) |

---

## License

Released under the [MIT License](LICENSE).

<div align="center">

**Built for HH Goa 2026 — Task 2**

*Because a RAG system that knows when NOT to answer is worth more than one that always does.*

</div>

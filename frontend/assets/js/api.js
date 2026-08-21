const TEXT_ENDPOINT = "/api/v1/query";
const VOICE_ENDPOINT = "/api/v1/voice/query";
const HEALTH_ENDPOINT = "/health";

const STAGE_LABELS = {
  stt: "Speech recognition",
  guard_input: "Input guard",
  embed: "Embedding",
  retrieve: "Retrieval",
  guard_relevance: "Relevance check",
  generate: "Generation",
  guard_grounding: "Grounding check",
};

async function parseResponse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    throw new ApiError("SERVER_ERROR", "The system returned an unreadable response.");
  }
  if (!res.ok) {
    const detail =
      body && body.detail ? String(body.detail).slice(0, 200) : `HTTP ${res.status}`;
    throw new ApiError(mapHttpError(res.status), detail);
  }
  return body;
}

function mapHttpError(status) {
  if (status === 0 || status === 502 || status === 503) return "UNREACHABLE";
  if (status === 429) return "RATE_LIMITED";
  if (status >= 500) return "SERVER_ERROR";
  return "REQUEST_FAILED";
}

export class ApiError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

export async function health() {
  const res = await fetch(HEALTH_ENDPOINT, { cache: "no-store" });
  return parseResponse(res);
}

export async function askText(query, { topK = 10, allowGenerative = false } = {}) {
  const res = await fetch(TEXT_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: topK,
      allow_generative: allowGenerative,
      debug: true,
    }),
  });
  return normalize(await parseResponse(res));
}

export async function askVoice(wavBlob, { topK = 10, allowGenerative = false } = {}) {
  const form = new FormData();
  form.append("audio", wavBlob, "audio.wav");
  form.append("top_k", String(topK));
  form.append("allow_generative", String(allowGenerative));
  form.append("debug", "true");
  const res = await fetch(VOICE_ENDPOINT, { method: "POST", body: form });
  return normalize(await parseResponse(res));
}

export function normalize(raw) {
  const stages = (raw.stages || []).map((s) => ({
    key: s.stage,
    label: STAGE_LABELS[s.stage] || s.stage,
    ms: Number(s.latency_ms || 0),
    status: s.status,
    details: s.details || {},
  }));

  const failedStage = stages.find((s) => s.status === "fail");

  let kind = "answer";
  if (raw.status !== "success") {
    if (failedStage && failedStage.key === "guard_input") kind = "refused";
    else if (raw.status === "error") kind = "error";
    else kind = "no_evidence";
  }

  const passages = raw.debug
    ? (raw.debug.retrieved_context || []).map((p) => ({
        rank: p.rank,
        score: p.score,
        text: p.text,
        chunkId: p.chunk_id,
        source: (p.metadata && p.metadata.source) || "MSMARCO-XI",
        language: (p.metadata && p.metadata.language) || "",
      }))
    : [];

  return {
    requestId: raw.request_id || "",
    query: raw.query || "",
    language: raw.detected_language || "",
    transcript: extractTranscript(stages),
    answerText: raw.answer ? raw.answer.text : "",
    method: raw.answer ? raw.answer.method : "",
    confidence: raw.answer ? raw.answer.confidence : null,
    grounded:
      raw.debug && raw.debug.grounding_status
        ? raw.debug.grounding_status === "GROUNDED"
        : null,
    refusalReason: raw.refusal_reason || "",
    kind,
    passages,
    stages,
    latency: {
      stt: raw.latency.stt_ms || 0,
      inputGuard: raw.latency.guard_input_ms || 0,
      embed: raw.latency.embed_ms || 0,
      retrieve: raw.latency.retrieve_ms || 0,
      relevance: raw.latency.guard_relevance_ms || 0,
      generation: raw.latency.answer_ms || 0,
      grounding: raw.latency.guard_grounding_ms || 0,
      core: raw.latency.total_core_ms || 0,
      e2e: raw.latency.total_e2e_ms || raw.latency.total_core_ms || 0,
    },
    tech: {
      chunkingStrategy: raw.debug ? raw.debug.chunking_strategy : "",
      llmModel: raw.debug ? raw.debug.llm_model : "",
      indexSize: raw.debug ? raw.debug.index_size : null,
    },
  };
}

function extractTranscript(stages) {
  const stt = stages.find((s) => s.key === "stt");
  if (stt && stt.details && stt.details.transcript) return stt.details.transcript;
  return "";
}

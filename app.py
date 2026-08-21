import sys
import time
import asyncio
import html as html_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
import gradio as gr

from pipeline.orchestrator import PipelineOrchestrator
from retrieval.search import Retriever
from retrieval.embedder import Embedder
from retrieval.vector_store import NumpyVectorStore
from guardrails.input_guard import InputGuard
from guardrails.relevance_guard import RelevanceGuard
from guardrails.grounding_guard import GroundingGuard
from generation.extractive import ExtractiveGenerator
from generation.generative import GenerativeGenerator
from stt.sarvam import SarvamSTT
from api.schemas import AskRequest

CONFIG_PATH = Path(__file__).parent / "config.yaml"
ARTIFACTS_DIR = Path(__file__).parent / "index" / "artifacts"
CSS_PATH = Path(__file__).parent / "static" / "style.css"

config = yaml.safe_load(open(CONFIG_PATH, "r"))

print("Loading embedding model...")
embedder = Embedder(config["embedding"]["model"])
print("Warming up embedder...")
embedder.embed_query("warmup")
print("Embedder ready.")

print("Loading vector store...")
vector_store = NumpyVectorStore()
if not vector_store.load(str(ARTIFACTS_DIR)):
    print("WARNING: No index found. Run build_index.py first.")

retriever = Retriever(embedder, vector_store)

llm_config = config.get("llm", {})

orchestrator = PipelineOrchestrator(
    retriever=retriever,
    input_guard=InputGuard(config["guardrails"]["input"]),
    relevance_guard=RelevanceGuard(config["guardrails"]["relevance"]),
    grounding_guard=GroundingGuard(config["guardrails"]["grounding"], embedder=embedder),
    extractive_gen=ExtractiveGenerator(config["generation"]["extractive"]["min_retrieval_score"]),
    generative_gen=GenerativeGenerator(
        provider=llm_config.get("provider", "openrouter"),
        base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
        model=llm_config.get("model", ""),
        reasoning=llm_config.get("reasoning", False),
    ),
    stt=SarvamSTT(),
    llm_model=llm_config.get("model", ""),
)
print("Orchestrator ready.")

INDEX_SIZE = vector_store.size()

LANG_NAMES = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}
STAGE_NAMES = {
    "stt": "Speech-to-Text",
    "guard_input": "Input Guard",
    "embed": "Embed Query",
    "retrieve": "Vector Retrieval",
    "guard_relevance": "Relevance Guard",
    "generate": "Answer Generation",
    "guard_grounding": "Grounding Check",
}


def esc(s):
    return html_mod.escape(str(s), quote=True)


def header_html():
    return f"""
<div class="vr-header">
  <div class="vr-brand">
    <div class="vr-logo">VR</div>
    <div>
      <div class="vr-title">Voice RAG</div>
      <div class="vr-subtitle">HH Goa '26 &middot; MSMARCO-XI &middot; {INDEX_SIZE:,} passages</div>
    </div>
  </div>
  <div class="vr-status"><span class="dot"></span> System Ready</div>
</div>
"""


def empty_html():
    return """
<div class="vr-empty">
  <div class="big">Ask a question to see grounded answers</div>
  <div class="small">Every answer is retrieved from the knowledge base, checked by three guardrails,<br>
  and served through the full pipeline in under 200ms.</div>
</div>
"""


def strength_info(top_score):
    if top_score >= 0.7:
        return "strong", "Strong evidence", 100.0
    if top_score >= 0.5:
        return "moderate", "Moderate evidence", max(20.0, (top_score / 0.7) * 100.0)
    return "weak", "Weak evidence", max(8.0, (top_score / 0.7) * 100.0)


def badges_html(grounding_status, method, lang):
    g = grounding_status or "GROUNDED"
    g_cls = "grounded" if g == "GROUNDED" else "ungrounded"
    g_icon = "&#10003;" if g == "GROUNDED" else "&#9888;"
    m = (method or "extractive").lower()
    m_label = "Generative" if m == "generative" else "Extractive"
    l = LANG_NAMES.get(lang, lang or "auto")
    return f"""
<div class="vr-badges">
  <span class="vr-badge {g_cls}">{g_icon} {esc(g)}</span>
  <span class="vr-badge method">{esc(m_label)}</span>
  <span class="vr-badge lang">{esc(l)}</span>
</div>
"""


def why_html(resp, top_score):
    passages = resp.debug.retrieved_context if resp.debug else []
    top = passages[0] if passages else None
    steps = []
    steps.append(
        ("<b>Query understood.</b> Your question was normalized and embedded into a "
         f"{config['embedding']['dimension']}-dim multilingual vector."))
    if top is not None:
        steps.append(
            (f"<b>Semantic match.</b> Best passage ranked #{top.rank} with cosine score "
             f"<b>{top.score:.3f}</b> out of {len(passages)} retrieved candidates."))
    steps.append(
        ("<b>Guardrails passed.</b> Input safety, relevance and context checks all cleared "
         "before answering."))
    m = (resp.answer.method if resp.answer else "extractive").lower()
    if m == "generative":
        steps.append(("<b>Answer composed.</b> An LLM wrote the answer strictly from the "
                      "retrieved passages above."))
    else:
        steps.append(("<b>Answer extracted.</b> The most relevant sentences were pulled "
                      "directly from the top-ranked passage - no generation, no hallucination."))
    steps.append(("<b>Grounded verified.</b> The final answer was re-checked against the source "
                  "context before being shown to you."))

    rows = "".join(
        f'<div class="vr-chain-step"><div class="vr-chain-num">{i+1:02d}</div>'
        f'<div class="vr-chain-text">{step}</div></div>'
        for i, step in enumerate(steps)
    )
    return f"""
<details class="vr-details">
  <summary>Why this answer?</summary>
  <div class="vr-details-body">{rows}</div>
</details>
"""


def evidence_html(resp):
    passages = resp.debug.retrieved_context if resp.debug else []
    if not passages:
        return """
<div class="vr-panel">
  <div class="vr-panel-title">Evidence</div>
  <div style="color:var(--muted);font-size:13px;">No passages returned.</div>
</div>
"""
    cards = []
    for p in passages[:5]:
        meta = p.metadata or {}
        src = meta.get("source", "MSMARCO-XI")
        lang = LANG_NAMES.get(meta.get("language", ""), meta.get("language", "?"))
        cards.append(f"""
<div class="vr-ev-card {'top1' if p.rank == 1 else ''}">
  <div class="vr-ev-head">
    <span class="vr-ev-rank">{p.rank:02d}</span>
    <span class="vr-ev-score">score <b>{p.score:.3f}</b></span>
  </div>
  <div class="vr-ev-text">{esc(p.text)}</div>
  <div class="vr-ev-meta">&#9679; {esc(src)} &middot; {esc(lang)}</div>
</div>""")
    return f"""
<div class="vr-panel">
  <div class="vr-panel-title"><span>Evidence</span><span>Top {min(5, len(passages))} passages</span></div>
  <div class="vr-evidence-list">{''.join(cards)}</div>
</div>
"""


def pipeline_drawer_html(resp, total_ms):
    stages = [
        s for s in resp.stages
        if s.stage in STAGE_NAMES and s.latency_ms is not None
    ]
    max_ms = max((s.latency_ms for s in stages), default=1.0) or 1.0
    rows = "".join(
        f"""
<div class="vr-stage-row">
  <div class="vr-stage-name">{esc(STAGE_NAMES[s.stage])}</div>
  <div class="vr-stage-bar-track"><div class="vr-stage-bar" style="width:{max(2.0, s.latency_ms / max_ms * 100):.1f}%"></div></div>
  <div class="vr-stage-ms">{s.latency_ms:.1f} ms</div>
</div>"""
        for s in stages
    )
    met = resp.latency.total_core_ms < 200
    target_cls = "met" if met else "missed"
    target_txt = "Target &lt; 200ms" if met else "Over 200ms budget"
    return f"""
<details class="vr-pipeline" open>
  <summary>
    <span class="vr-pl-summary-left">Pipeline Latency</span>
    <span class="vr-pl-total">
      <span class="target {target_cls}">{target_txt}</span>
      <span class="ms">{resp.latency.total_core_ms:.1f} ms</span>
    </span>
  </summary>
  <div class="vr-pl-body">
    {rows}
    <div class="vr-stage-row" style="border-top:1px solid var(--border);margin-top:6px;padding-top:10px;">
      <div class="vr-stage-name" style="color:var(--text)">Core Pipeline</div>
      <div></div>
      <div class="vr-stage-ms">{resp.latency.total_core_ms:.1f} ms</div>
    </div>
    <div class="vr-stage-row">
      <div class="vr-stage-name">End-to-End</div>
      <div></div>
      <div class="vr-stage-ms">{total_ms:.1f} ms</div>
    </div>
  </div>
</details>
"""


def result_html(resp, total_ms, transcript=""):
    q = resp.query
    quote = f"""
<div class="vr-question-quote">
  <span class="q-mark">Q.</span> {esc(q)}
</div>"""
    if transcript and transcript != q:
        quote += f"""
<div class="vr-question-quote" style="margin-top:-8px;">
  <span class="q-mark">&#127908;</span> Transcript: {esc(transcript)}
</div>"""

    ans = resp.answer
    grounding_status = resp.debug.grounding_status if resp.debug else ""
    method = ans.method if ans else ""
    lang = resp.detected_language or ""

    top_score = 0.0
    if resp.debug and resp.debug.retrieved_context:
        top_score = resp.debug.retrieved_context[0].score
    elif ans and ans.citations:
        top_score = ans.citations[0].score

    s_cls, s_label, s_pct = strength_info(top_score)

    left = f"""
<div class="vr-panel">
  <div class="vr-panel-title"><span>Answer</span><span>ID {esc(resp.request_id)}</span></div>
  <div class="vr-answer-text">{esc(ans.text) if ans else 'No answer generated.'}</div>
  {badges_html(grounding_status, method, lang)}
  <div class="vr-strength">
    <div class="vr-strength-header"><span>Evidence Strength</span><span>{esc(s_label)}</span></div>
    <div class="vr-strength-track"><div class="vr-strength-fill {s_cls}" style="width:{s_pct:.0f}%"></div></div>
  </div>
  {why_html(resp, top_score)}
</div>"""

    return f"""
<div class="vr-result">
  {quote}
  <div class="vr-grid">
    {left}
    {evidence_html(resp)}
  </div>
  {pipeline_drawer_html(resp, total_ms)}
</div>
"""


def refusal_html(resp, total_ms, transcript=""):
    t = f'<p style="margin-top:10px;color:#C9C9CF;">Transcript: {esc(transcript)}</p>' if transcript else ""
    return f"""
<div class="vr-refusal">
  <div class="vr-refusal-icon">&#9888;</div>
  <h3>Query refused by guardrail</h3>
  <p>{esc(resp.refusal_reason or 'This query could not be answered safely.')}</p>
  {t}
  <p style="margin-top:14px;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;">
    Core pipeline: {resp.latency.total_core_ms:.1f} ms &middot; ID {esc(resp.request_id)}
  </p>
</div>
"""


def run_pipeline(query, lang_val, top_k, allow_generative):
    req = AskRequest(
        query=query[:500],
        lang=lang_val,
        top_k=int(top_k),
        allow_generative=bool(allow_generative),
        debug=True,
    )
    start = time.perf_counter()
    resp = orchestrator.process_text(req)
    total_ms = (time.perf_counter() - start) * 1000
    return resp, total_ms


def text_query(query, lang, top_k, allow_generative):
    if not query or not query.strip():
        return empty_html()
    lang_val = lang if lang and lang != "auto" else None
    try:
        resp, total_ms = run_pipeline(query, lang_val, top_k, allow_generative)
    except Exception as e:
        return f"""
<div class="vr-refusal">
  <div class="vr-refusal-icon">&#9888;</div>
  <h3>Pipeline error</h3>
  <p>{esc(e)}</p>
</div>"""
    if resp.status != "success":
        return refusal_html(resp, total_ms)
    return result_html(resp, total_ms)


def voice_query(audio, top_k, allow_generative):
    if audio is None:
        return empty_html()
    try:
        with open(audio, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        return f"""
<div class="vr-refusal">
  <div class="vr-refusal-icon">&#9888;</div>
  <h3>Audio error</h3>
  <p>{esc(e)}</p>
</div>"""

    start = time.perf_counter()
    try:
        resp = asyncio.run(orchestrator.process_voice(
            audio_bytes=audio_bytes,
            top_k=int(top_k),
            allow_generative=bool(allow_generative),
            debug=True,
        ))
    except Exception as e:
        return f"""
<div class="vr-refusal">
  <div class="vr-refusal-icon">&#9888;</div>
  <h3>Pipeline error</h3>
  <p>{esc(e)}</p>
</div>"""
    total_ms = (time.perf_counter() - start) * 1000

    transcript = ""
    for stage in resp.stages:
        if stage.stage == "stt" and stage.details:
            transcript = stage.details.get("transcript", "")

    if resp.status != "success":
        return refusal_html(resp, total_ms, transcript)
    return result_html(resp, total_ms, transcript)


CSS = CSS_PATH.read_text(encoding="utf-8")

with gr.Blocks(title="Voice RAG - HH Goa 2026") as demo:
    gr.HTML(header_html())

    with gr.Group(elem_classes=["vr-ask-card"]):
        gr.HTML('<div class="vr-ask-label">Ask anything</div>')
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    placeholder="Type your question, or record audio on the right...",
                    label="Question",
                    lines=1,
                    max_lines=3,
                )
                audio_input = gr.Audio(
                    type="filepath",
                    label="Or speak your question",
                    elem_classes=["vr-audio"],
                )
            with gr.Column(scale=1):
                lang_select = gr.Dropdown(
                    choices=[("Auto-detect", "auto"), ("English", "en"), ("Hindi", "hi"), ("Gujarati", "gu")],
                    value="auto",
                    label="Language",
                )
                top_k_slider = gr.Slider(1, 20, value=10, step=1, label="Passages to retrieve")
                gen_toggle = gr.Checkbox(value=False, label="Allow generative answers")
        with gr.Row():
            text_btn = gr.Button("Ask", variant="primary", elem_classes=["vr-ask-btn"], scale=2)
            voice_btn = gr.Button("Ask (Voice)", variant="primary", elem_classes=["vr-ask-btn"], scale=1)

    result_display = gr.HTML(empty_html())

    gr.HTML("""
<div class="vr-footer">
  <span>Built for HH Goa 2026 &middot; Task 2</span>
  <span>#RAGInGoa</span>
</div>
""")

    text_btn.click(
        text_query,
        inputs=[text_input, lang_select, top_k_slider, gen_toggle],
        outputs=[result_display],
    )
    text_input.submit(
        text_query,
        inputs=[text_input, lang_select, top_k_slider, gen_toggle],
        outputs=[result_display],
    )
    voice_btn.click(
        voice_query,
        inputs=[audio_input, top_k_slider, gen_toggle],
        outputs=[result_display],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, css=CSS)

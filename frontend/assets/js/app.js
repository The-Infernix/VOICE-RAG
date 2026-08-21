import { askText, askVoice, health, ApiError } from "./api.js";
import { VoiceRecorder, RecorderError } from "./audio.js";
import { VisualEngine } from "./visuals.js";
import { loadHistory, addHistory, clearHistory, timeAgo } from "./history.js";

const $ = (id) => document.getElementById(id);

const els = {
  glow: $("glow"),
  stage: $("stage"),
  viewTop: $("viewTop"),
  viewBottom: $("viewBottom"),
  viewMain: $("viewMain"),
  orbSlot: $("orbSlot"),
  orbCanvas: $("orbCanvas"),
  orbButton: $("orbButton"),
  sysStatus: $("sysStatus"),
  sysStatusText: $("sysStatusText"),
  kbPassages: $("kbPassages"),
  techToggle: $("techToggle"),
  historyBtn: $("historyBtn"),
  evidenceDrawer: $("evidenceDrawer"),
  historyPanel: $("historyPanel"),
  modalRoot: $("modalRoot"),
  scrim: $("scrim"),
  toasts: $("toasts"),
  srLive: $("srLive"),
};

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const engine = new VisualEngine(els.orbCanvas, { reducedMotion });
const recorder = new VoiceRecorder();

let state = "boot";
let typeToken = 0;
let recording = false;

const LANG_NAMES = { en: "English", hi: "Hindi", gu: "Gujarati" };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function announce(msg) {
  els.srLive.textContent = msg;
}

function setState(next) {
  state = next;
  document.body.classList.toggle("is-listening", next === "listening");
  document.body.classList.toggle("has-answer", ["answer", "refused", "no_evidence", "error"].includes(next));
  const mainVisible = ["answer", "refused", "no_evidence", "error"].includes(next);
  els.orbSlot.classList.toggle("hidden-orb", mainVisible);
  els.viewMain.style.display = mainVisible ? "" : "none";
  if (!mainVisible) {
    els.viewTop.style.display = "";
    els.viewBottom.style.display = "";
  } else {
    els.viewTop.style.display = "none";
    els.viewBottom.style.display = "none";
  }
}

function setOrbMode(mode) {
  engine.setMode(mode);
  if (mode === "idle") engine.start();
}

async function typewriter(element, text, speed = 18) {
  const token = ++typeToken;
  element.innerHTML = "";
  const caret = '<span class="caret"></span>';
  if (reducedMotion) {
    if (token === typeToken) element.innerHTML = esc(text) + caret;
    return;
  }
  for (let i = 0; i <= text.length; i++) {
    if (token !== typeToken) return;
    element.innerHTML = esc(text.slice(0, i)) + caret;
    await delay(speed);
  }
}

function toast(msg, ms = 3400) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  els.toasts.appendChild(el);
  setTimeout(() => {
    el.classList.add("gone");
    setTimeout(() => el.remove(), 320);
  }, ms);
}

/* ---------- views ---------- */

function renderIdle() {
  closeDrawers();
  closeModal();
  setState("idle");
  setOrbMode("idle");
  els.viewTop.innerHTML = `
    <h1 class="headline">Ask.<br><em>We&rsquo;ll find <span class="hl">the evidence.</span></em></h1>
    <p class="tagline">Speak naturally. Every answer traces back to the passages it came from.</p>
  `;
  els.viewBottom.innerHTML = `
    <div class="hint-stack">
      <span class="hint-primary" id="hintPrimary">Hold to speak</span>
      <span class="hint-secondary">or press <kbd>Space</kbd></span>
    </div>
    <form class="typebar" id="typeForm">
      <input id="typeInput" type="text" placeholder="…or type your question" autocomplete="off" enterkeyhint="send" aria-label="Type your question">
      <button type="submit" aria-label="Send question">
        <svg viewBox="0 0 16 16" fill="none"><path d="M2 8h11M9 3.5 13.5 8 9 12.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
    </form>
  `;

  $("typeForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("typeInput").value.trim();
    if (!q) return;
    submitText(q);
  });
  announce("Ready. Hold the orb or press Space to speak, or type a question.");
}

function renderListening() {
  setState("listening");
  setOrbMode("listening");
  els.viewTop.innerHTML = `
    <div class="proc-label"><span class="bars"><i></i><i></i><i></i></span> Listening</div>
  `;
  els.viewBottom.innerHTML = `
    <div class="hint-stack">
      <span class="hint-primary">Release to stop</span>
      <span class="hint-secondary"><kbd>Esc</kbd> to cancel</span>
    </div>
  `;
  announce("Listening");
}

function renderUnderstanding(text) {
  setState("understanding");
  setOrbMode("idle");
  els.viewTop.innerHTML = `<div class="proc-label">Understanding</div>`;
  els.viewBottom.innerHTML = `<p class="proc-transcript" id="procTranscript"></p>`;
  if (text) typewriter($("procTranscript"), text);
  announce("Understanding");
}

function renderRetrieving() {
  setState("retrieving");
  setOrbMode("retrieving");
  els.viewTop.innerHTML = `<div class="proc-label">Searching knowledge</div>`;
  els.viewBottom.innerHTML = "";
  announce("Searching knowledge");
}

function renderFound(n) {
  els.viewTop.innerHTML = `
    <div class="proc-found">Found <b>${n}</b> relevant passage${n === 1 ? "" : "s"}</div>
  `;
  announce(`Found ${n} relevant passages`);
}

/* ---------- result presentation ---------- */

function presentResult(result, { silentHistory = false } = {}) {
  if (!silentHistory && result.kind === "answer") {
    addHistory({
      id: result.requestId + ":" + Date.now(),
      ts: Date.now(),
      query: result.query,
      snippet: result.answerText.slice(0, 160),
      payload: result,
    });
  }
  if (result.kind === "answer") renderAnswer(result);
  else if (result.kind === "refused") renderRefused(result);
  else if (result.kind === "no_evidence") renderNoEvidence(result);
  else renderErrorView(result);
}

function renderAnswer(r) {
  setState("answer");
  setOrbMode("off");
  const groundedPill = r.grounded === false
    ? `<button class="pill ungrounded" data-action="why">&#9888; Ungrounded answer</button>`
    : `<button class="pill grounded" data-action="why">&#10003; Grounded in retrieved evidence</button>`;

  els.viewMain.innerHTML = `
    <div class="ans-wrap">
      <div class="ans-q">
        <span class="micro">Your question</span>
        <p>${esc(r.query)}</p>
      </div>
      <div class="ans-block">
        <span class="micro">Answer</span>
        <h2 class="ans-text">${esc(r.answerText)}</h2>
        <div class="ans-meta">
          ${groundedPill}
          <button class="pill latency-ms" data-action="latency"><b>${Math.round(r.latency.core)}</b> ms <span class="chev">&#9662;</span></button>
          <button class="pill evidence-count" data-action="evidence">Evidence <b>${String(r.passages.length).padStart(2, "0")}</b> <span class="chev">&#9662;</span></button>
          <button class="pill" data-action="why">Why this answer <span class="chev">&rarr;</span></button>
        </div>
      </div>
      <div class="ans-lower">
        <details class="panel">
          <summary>Live pipeline <span class="sum-right">${r.stages.length} stages</span></summary>
          <div class="panel-body">${pipelineTimeline(r)}</div>
        </details>
        <div class="tech-only">
          <details class="panel">
            <summary>Technical detail <span class="sum-right">${esc(r.requestId)}</span></summary>
            <div class="panel-body">${techGrid(r)}</div>
          </details>
        </div>
        <div style="text-align:center;margin-top:10px;">
          <button class="btn-primary" data-action="again">Ask another question</button>
        </div>
      </div>
    </div>
  `;

  wireAnswerActions(r);
  announce(`Answer ready in ${Math.round(r.latency.core)} milliseconds.`);
}

function wireAnswerActions(r) {
  els.viewMain.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      if (action === "evidence") openEvidenceDrawer(r);
      else if (action === "latency") openLatencyModal(r);
      else if (action === "why") openWhyModal(r);
      else if (action === "again") renderIdle();
    });
  });
}

function pipelineTimeline(r) {
  const rows = r.stages.map((s) => {
    const kv = Object.entries(s.details)
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => {
        let val = String(v);
        if (val.length > 90) val = val.slice(0, 90) + "…";
        return `<div><span>${esc(k)}</span> <b>${esc(val)}</b></div>`;
      })
      .join("");
    return `
      <div class="tl-row ${s.status === "fail" ? "fail" : ""}">
        <div class="tl-node"></div>
        <div class="tl-body">
          <div class="tl-name">${esc(s.label)}</div>
          ${kv ? `<details class="tl-details"><summary>Details</summary><div class="tl-kv">${kv}</div></details>` : ""}
        </div>
        <div class="tl-ms">${s.ms.toFixed(1)} ms</div>
      </div>
    `;
  }).join("");
  return `<div class="timeline">${rows}</div>`;
}

function techGrid(r) {
  const lang = LANG_NAMES[r.language] || r.language || "—";
  const cells = [
    ["Request ID", r.requestId],
    ["Chunk strategy", r.tech.chunkingStrategy || "—"],
    ["LLM model", r.tech.llmModel || "extractive only"],
    ["Index size", r.tech.indexSize != null ? r.tech.indexSize.toLocaleString() + " passages" : "—"],
    ["Detected language", lang],
    ["Method", r.method ? `${r.method}${r.confidence != null ? " · " + r.confidence.toFixed(2) : ""}` : "—"],
  ];
  return `<div class="tech-grid">${cells.map(([k, v]) => `
    <div class="tech-cell"><span class="micro">${esc(k)}</span><div>${esc(v)}</div></div>
  `).join("")}</div>`;
}

function renderRefused(r) {
  setState("refused");
  setOrbMode("off");
  els.viewMain.innerHTML = `
    <div class="end-state bad">
      <div class="end-icon">
        <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.6"/><path d="M5.5 5.5l13 13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </div>
      <div class="end-title">Query refused</div>
      <p class="end-msg">${esc(r.refusalReason || "This query can't be answered.")}</p>
      <p class="end-detail">${esc(r.requestId)}</p>
      <button class="btn-primary" id="againBtn">Ask another question</button>
    </div>
  `;
  $("againBtn").addEventListener("click", renderIdle);
  announce("Query refused");
}

function renderNoEvidence(r) {
  setState("no_evidence");
  setOrbMode("off");
  els.viewMain.innerHTML = `
    <div class="end-state ok">
      <div class="end-icon">
        <svg viewBox="0 0 24 24" fill="none"><circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" stroke-width="1.7"/><path d="M15.5 15.5 20 20" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><path d="M8 10.5h5M10.5 8v5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" opacity="0.55"/></svg>
      </div>
      <div class="end-title">No sufficient evidence</div>
      <p class="end-msg">&ldquo;I couldn&rsquo;t find enough relevant information in the provided knowledge base to answer this.&rdquo;</p>
      ${r.refusalReason ? `<p class="end-detail">${esc(r.refusalReason)}</p>` : ""}
      <button class="btn-primary" id="againBtn">Ask another question</button>
    </div>
  `;
  $("againBtn").addEventListener("click", renderIdle);
  announce("No sufficient evidence found");
}

const ERROR_COPY = {
  UNREACHABLE: ["Can't reach the system", "The RAG service isn't responding. Check that the backend is running, then try again."],
  SERVER_ERROR: ["System error", "Something went wrong while processing your question. Please try again."],
  RATE_LIMITED: ["Slow down", "Too many requests are in flight. Give it a moment and try again."],
  REQUEST_FAILED: ["Request failed", "The system rejected the request. Please try again."],
};

function renderErrorView(r) {
  const isStt = !r.transcript && (!r.passages || r.passages.length === 0);
  let title, msg;
  if (isStt) {
    title = "Transcription failed";
    msg = "We couldn't make out any speech in that recording. Try holding the orb a moment longer and speaking clearly.";
  } else {
    [title, msg] = ERROR_COPY.SERVER_ERROR;
  }
  setState("error");
  setOrbMode("off");
  els.viewMain.innerHTML = `
    <div class="end-state bad">
      <div class="end-icon">
        <svg viewBox="0 0 24 24" fill="none"><path d="M12 3 22 20H2L12 3z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M12 10v4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="12" cy="17.2" r="0.4" fill="currentColor"/></svg>
      </div>
      <div class="end-title">${esc(title)}</div>
      <p class="end-msg">${esc(msg)}</p>
      ${r.refusalReason ? `<p class="end-detail">${esc(r.refusalReason)}</p>` : ""}
      <button class="btn-primary" id="againBtn">Try again</button>
    </div>
  `;
  $("againBtn").addEventListener("click", renderIdle);
  announce(title);
}

function renderApiError(err) {
  const copy = ERROR_COPY[err instanceof ApiError ? err.code : "SERVER_ERROR"] || ERROR_COPY.SERVER_ERROR;
  setState("error");
  setOrbMode("off");
  els.viewMain.innerHTML = `
    <div class="end-state bad">
      <div class="end-icon">
        <svg viewBox="0 0 24 24" fill="none"><path d="M12 3 22 20H2L12 3z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M12 10v4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="12" cy="17.2" r="0.4" fill="currentColor"/></svg>
      </div>
      <div class="end-title">${esc(copy[0])}</div>
      <p class="end-msg">${esc(copy[1])}</p>
      <p class="end-detail">${esc(err.message || "")}</p>
      <button class="btn-primary" id="againBtn">Try again</button>
    </div>
  `;
  $("againBtn").addEventListener("click", renderIdle);
  announce(copy[0]);
}

/* ---------- choreography ---------- */

async function submitText(query) {
  closeDrawers();
  renderUnderstanding(query);
  const typing = typewriter($("procTranscript"), query);
  let result;
  try {
    [result] = await Promise.all([askText(query), typing, delay(reducedMotion ? 200 : 1150)]);
  } catch (err) {
    renderApiError(err);
    return;
  }
  await retrievePhase(result);
}

async function submitVoice(wav) {
  renderUnderstanding("");
  let result;
  try {
    [result] = await Promise.all([askVoice(wav), delay(reducedMotion ? 200 : 700)]);
  } catch (err) {
    renderApiError(err);
    return;
  }
  if (result.transcript) {
    renderUnderstanding(result.transcript);
    await Promise.all([typewriter($("procTranscript"), result.transcript, 14), delay(reducedMotion ? 100 : 900)]);
  }
  await retrievePhase(result);
}

async function retrievePhase(result) {
  renderRetrieving();
  await delay(reducedMotion ? 250 : 1050);
  const n = result.passages.length;
  engine.foundCount = Math.max(1, Math.min(n, 5));
  setOrbMode("found");
  renderFound(n);
  await delay(reducedMotion ? 150 : 850);
  presentResult(result);
}

/* ---------- voice recording ---------- */

async function startRecording() {
  if (recording || state !== "idle") return;
  try {
    await recorder.start();
  } catch (err) {
    if (err instanceof RecorderError) {
      if (err.code === "PERMISSION_DENIED") {
        toast("Microphone access denied — allow it in your browser settings");
        return;
      }
      if (err.code === "NO_MIC") {
        toast("No microphone found on this device");
        return;
      }
      toast(err.message);
      return;
    }
    toast("Could not start the microphone");
    return;
  }
  recording = true;
  els.orbButton.setAttribute("aria-pressed", "true");
  engine.setProviders({
    amplitude: () => recorder.amplitude(),
    spectrum: (bins) => recorder.spectrum(bins),
  });
  renderListening();
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  els.orbButton.setAttribute("aria-pressed", "false");
  const wav = await recorder.stop();
  if (!wav) {
    toast("Nothing was recorded — try again");
    renderIdle();
    return;
  }
  submitVoice(wav);
}

function cancelRecording() {
  if (!recording) return;
  recording = false;
  els.orbButton.setAttribute("aria-pressed", "false");
  recorder.cancel();
  renderIdle();
}

/* ---------- drawers & modals ---------- */

function openScrim() {
  els.scrim.hidden = false;
  requestAnimationFrame(() => els.scrim.classList.add("show"));
}

function closeScrim() {
  els.scrim.classList.remove("show");
  setTimeout(() => {
    els.scrim.hidden = true;
  }, 300);
}

let lastFocus = null;

function openDrawer(el, html) {
  lastFocus = document.activeElement;
  el.innerHTML = html;
  el.classList.add("open");
  el.setAttribute("aria-hidden", "false");
  openScrim();
  const closeBtn = el.querySelector(".drawer-close");
  if (closeBtn) closeBtn.focus();
}

function closeDrawers() {
  let wasOpen = false;
  [els.evidenceDrawer, els.historyPanel].forEach((el) => {
    if (el.classList.contains("open")) {
      wasOpen = true;
      el.classList.remove("open");
      el.setAttribute("aria-hidden", "true");
    }
  });
  if (wasOpen) closeScrim();
  if (lastFocus && lastFocus.focus) lastFocus.focus();
  lastFocus = null;
}

function openModal(html) {
  lastFocus = document.activeElement;
  els.modalRoot.innerHTML = `<div class="modal-card" role="dialog" aria-modal="true">${html}</div>`;
  els.modalRoot.classList.add("open");
  els.modalRoot.setAttribute("aria-hidden", "false");
  openScrim();
  const closeBtn = els.modalRoot.querySelector(".drawer-close");
  if (closeBtn) closeBtn.focus();
}

function closeModal() {
  if (!els.modalRoot.classList.contains("open")) return;
  els.modalRoot.classList.remove("open");
  els.modalRoot.setAttribute("aria-hidden", "true");
  els.modalRoot.innerHTML = "";
  if (!anyDrawerOpen()) closeScrim();
  if (lastFocus && lastFocus.focus) lastFocus.focus();
  lastFocus = null;
}

function anyDrawerOpen() {
  return els.evidenceDrawer.classList.contains("open") || els.historyPanel.classList.contains("open");
}

function openEvidenceDrawer(r) {
  const items = r.passages.map((p) => `
    <article class="ev-item">
      <div class="ev-head">
        <span class="ev-rank">${String(p.rank).padStart(2, "0")}</span>
        <div class="ev-score">
          <span class="ev-score-val">similarity ${p.score.toFixed(3)}</span>
          <div class="ev-score-track"><div class="ev-score-fill" style="width:${Math.min(100, p.score * 100).toFixed(1)}%"></div></div>
        </div>
      </div>
      <p class="ev-text">${esc(p.text)}</p>
      <div class="ev-meta">
        <span>${esc(p.source)}</span>
        ${p.language ? `<span>${esc(LANG_NAMES[p.language] || p.language)}</span>` : ""}
        <span class="tech-only">${esc(p.chunkId)}</span>
      </div>
    </article>
  `).join("");

  openDrawer(els.evidenceDrawer, `
    <div class="drawer-head">
      <h2>Evidence <span class="count">${String(r.passages.length).padStart(2, "0")}</span></h2>
      <button class="drawer-close" aria-label="Close evidence">
        <svg viewBox="0 0 14 14" width="13" height="13"><path d="M1 1l12 12M13 1L1 13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button>
    </div>
    <div class="drawer-body">
      ${items || '<div class="hist-empty">No passages were returned for this answer.</div>'}
    </div>
  `);

  els.evidenceDrawer.querySelectorAll(".ev-text").forEach((p) => {
    p.addEventListener("click", () => p.classList.toggle("expanded"));
  });
  els.evidenceDrawer.querySelector(".drawer-close").addEventListener("click", closeDrawers);
}

function openLatencyModal(r) {
  const rows = [];
  const add = (name, ms) => {
    if (ms > 0) rows.push(`
      <div class="lat-row">
        <span class="lat-name"><i></i>${esc(name)}</span>
        <span class="lat-ms">${ms.toFixed(1)} ms</span>
      </div>
    `);
  };
  add("Speech recognition", r.latency.stt);
  add("Input guard", r.latency.inputGuard);
  add("Embedding", r.latency.embed);
  add("Retrieval", r.latency.retrieve);
  add("Relevance check", r.latency.relevance);
  add("Generation", r.latency.generation);
  add("Grounding check", r.latency.grounding);

  const met = r.latency.core < 200;
  openModal(`
    <div class="modal-head">
      <h2>Latency</h2>
      <button class="drawer-close" aria-label="Close latency">
        <svg viewBox="0 0 14 14" width="13" height="13"><path d="M1 1l12 12M13 1L1 13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button>
    </div>
    ${rows.join("")}
    <div class="lat-total"><span>Total pipeline</span><span class="lat-ms">${r.latency.core.toFixed(1)} ms</span></div>
    <div class="lat-target ${met ? "met" : "over"}">
      <span>Target &lt; 200 ms</span>
      <span>${met ? "Met" : "Over budget"}</span>
    </div>
  `);
  els.modalRoot.querySelector(".drawer-close").addEventListener("click", closeModal);
}

function openWhyModal(r) {
  const n = r.passages.length;
  const top = r.passages[0];
  const genStep = r.method === "generative"
    ? "An LLM composed the answer <b>strictly from the retrieved passages</b> — it never saw the question alone."
    : "The answer was <b>extracted directly from the retrieved passages</b> — no free-form generation involved.";
  const groundStep = r.grounded === false
    ? "Post-generation check flagged this answer as <b>not sufficiently supported</b> by the sources."
    : "The final answer was <b>verified against the sources</b> before being shown.";

  openModal(`
    <div class="modal-head">
      <h2>Why this answer</h2>
      <button class="drawer-close" aria-label="Close">
        <svg viewBox="0 0 14 14" width="13" height="13"><path d="M1 1l12 12M13 1L1 13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button>
    </div>
    <div class="chain">
      <div class="chain-step">
        <div class="chain-node"></div>
        <div>
          <div class="chain-title">Question</div>
          <div class="chain-desc">Your question was converted into a semantic vector representation.</div>
        </div>
      </div>
      <div class="chain-step">
        <div class="chain-node"></div>
        <div>
          <div class="chain-title">Retrieved evidence</div>
          <div class="chain-desc"><b>${n} passage${n === 1 ? "" : "s"}</b> matched from MSMARCO-XI${top ? ` — best similarity <b>${top.score.toFixed(3)}</b>` : ""}.</div>
        </div>
      </div>
      <div class="chain-step">
        <div class="chain-node"></div>
        <div>
          <div class="chain-title">Grounded generation</div>
          <div class="chain-desc">${genStep}</div>
        </div>
      </div>
      <div class="chain-step">
        <div class="chain-node"></div>
        <div>
          <div class="chain-title">Answer</div>
          <div class="chain-desc">${groundStep}</div>
        </div>
      </div>
    </div>
  `);
  els.modalRoot.querySelector(".drawer-close").addEventListener("click", closeModal);
}

function openHistory() {
  const items = loadHistory();
  const list = items.length
    ? items.map((h) => `
        <button class="hist-item" data-hist="${esc(h.id)}">
          <div class="hist-q">${esc(h.query)}</div>
          <div class="hist-a">${esc(h.snippet)}</div>
          <div class="hist-time">${timeAgo(h.ts)}</div>
        </button>
      `).join("")
    : '<div class="hist-empty">Questions you ask will appear here.</div>';

  openDrawer(els.historyPanel, `
    <div class="drawer-head">
      <h2>History</h2>
      <button class="drawer-close" aria-label="Close history">
        <svg viewBox="0 0 14 14" width="13" height="13"><path d="M1 1l12 12M13 1L1 13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button>
    </div>
    <div class="drawer-body">
      ${list}
      ${items.length ? '<button class="danger-link" id="clearHist">Clear history</button>' : ""}
    </div>
  `);

  els.historyPanel.querySelector(".drawer-close").addEventListener("click", closeDrawers);
  const clearBtn = els.historyPanel.querySelector("#clearHist");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      clearHistory();
      openHistory();
    });
  }
  els.historyPanel.querySelectorAll("[data-hist]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = loadHistory().find((h) => h.id === btn.dataset.hist);
      if (!item) return;
      closeDrawers();
      presentResult(item.payload, { silentHistory: true });
    });
  });
}

/* ---------- global wiring ---------- */

els.orbButton.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  startRecording();
});
els.orbButton.addEventListener("pointerup", () => stopRecording());
els.orbButton.addEventListener("pointerleave", () => {
  if (recording) stopRecording();
});
els.orbButton.addEventListener("mouseenter", () => engine.setHover(true));
els.orbButton.addEventListener("mouseleave", () => engine.setHover(false));

document.addEventListener("keydown", (e) => {
  const tag = document.activeElement ? document.activeElement.tagName : "";
  const typing = tag === "INPUT" || tag === "TEXTAREA";

  if (e.code === "Escape") {
    if (recording) cancelRecording();
    else if (els.modalRoot.classList.contains("open")) closeModal();
    else if (anyDrawerOpen()) closeDrawers();
    return;
  }

  if (e.code === "Space" && !typing && !e.repeat) {
    e.preventDefault();
    startRecording();
  }
});

document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && recording) {
    e.preventDefault();
    stopRecording();
  }
});

els.scrim.addEventListener("click", () => {
  closeModal();
  closeDrawers();
});

els.historyBtn.addEventListener("click", openHistory);

els.techToggle.addEventListener("click", () => {
  const on = document.body.classList.toggle("tech");
  els.techToggle.setAttribute("aria-pressed", String(on));
  try {
    localStorage.setItem("voicerag.tech", on ? "1" : "0");
  } catch {}
});

if ((() => {
  try {
    return localStorage.getItem("voicerag.tech") === "1";
  } catch {
    return false;
  }
})()) {
  document.body.classList.add("tech");
  els.techToggle.setAttribute("aria-pressed", "true");
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) engine.stop();
  else if (["idle", "listening", "understanding", "retrieving", "found"].includes(state)) engine.start();
});

async function checkHealth() {
  try {
    const h = await health();
    els.sysStatus.classList.remove("offline");
    els.sysStatusText.textContent = "ONLINE";
    if (h.index_size) {
      els.kbPassages.textContent = `${h.index_size.toLocaleString()} passages indexed`;
    }
  } catch {
    els.sysStatus.classList.add("offline");
    els.sysStatusText.textContent = "OFFLINE";
  }
}

engine.setProviders({ amplitude: () => 0, spectrum: () => new Array(56).fill(0) });
engine.start();
renderIdle();
checkHealth();

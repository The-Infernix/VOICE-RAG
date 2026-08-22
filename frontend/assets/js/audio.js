const TARGET_RATE = 16000;
const MIN_RECORD_MS = 350;

export class RecorderError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

export class VoiceRecorder {
  constructor() {
    this.stream = null;
    this.recorder = null;
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.chunks = [];
    this.active = false;
    this.startedAt = 0;
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new RecorderError("UNSUPPORTED", "Audio recording is not supported in this browser.");
    }
    if (!window.MediaRecorder) {
      throw new RecorderError("UNSUPPORTED", "Audio recording is not supported in this browser.");
    }

    const stream = await this.ensureStream();
    this.chunks = [];
    this.startedAt = performance.now();

    try {
      if (!this.audioContext || this.audioContext.state === "closed") {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 64;
        this.analyser.smoothingTimeConstant = 0.55;
        source.connect(this.analyser);
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      }
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume().catch(() => {});
      }
    } catch {
      this.analyser = null;
    }

    const mime = pickMime();
    this.recorder = mime
      ? new MediaRecorder(stream, { mimeType: mime })
      : new MediaRecorder(stream);
    this.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };
    this.recorder.start(120);
    this.active = true;
  }

  async ensureStream() {
    const live = this.stream && this.stream.getAudioTracks().some((t) => t.readyState === "live");
    if (live) return this.stream;
    this.release();

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
        throw new RecorderError("PERMISSION_DENIED", "Microphone access was denied.");
      }
      if (err && err.name === "NotFoundError") {
        throw new RecorderError("NO_MIC", "No microphone was found on this device.");
      }
      throw new RecorderError("MIC_FAILED", "The microphone could not be started.");
    }

    this.stream.getAudioTracks().forEach((t) => {
      t.addEventListener("ended", () => {
        if (this.stream && this.stream.getAudioTracks().includes(t)) this.stream = null;
      });
    });
    return this.stream;
  }

  amplitude() {
    if (!this.analyser || !this.dataArray) return 0;
    this.analyser.getByteFrequencyData(this.dataArray);
    let sum = 0;
    for (let i = 0; i < this.dataArray.length; i++) sum += this.dataArray[i];
    const avg = sum / this.dataArray.length / 255;
    return Math.min(1, avg * 2.4);
  }

  spectrum(bins = 48) {
    const out = new Array(bins).fill(0);
    if (!this.analyser || !this.dataArray) return out;
    this.analyser.getByteFrequencyData(this.dataArray);
    const usable = Math.floor(this.dataArray.length * 0.75);
    const step = usable / bins;
    for (let i = 0; i < bins; i++) {
      const idx = Math.floor(i * step);
      out[i] = this.dataArray[idx] / 255;
    }
    return out;
  }

  async stop() {
    if (!this.active || !this.recorder) return null;
    this.active = false;

    const elapsed = performance.now() - this.startedAt;
    if (elapsed < MIN_RECORD_MS) {
      await new Promise((r) => setTimeout(r, MIN_RECORD_MS - elapsed));
    }

    const stopped = new Promise((resolve) => {
      this.recorder.onstop = resolve;
    });
    if (this.recorder.state !== "inactive") this.recorder.stop();
    await stopped;

    const raw = new Blob(this.chunks, { type: this.recorder.mimeType || "audio/webm" });
    this.chunks = [];
    this.recorder = null;
    if (raw.size === 0) return null;
    return blobToWav(raw);
  }

  cancel() {
    this.active = false;
    if (this.recorder && this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch {}
    }
    this.chunks = [];
    this.recorder = null;
  }

  release() {
    this.active = false;
    if (this.recorder && this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch {}
    }
    this.recorder = null;
    this.chunks = [];
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close().catch(() => {});
    }
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
  }
}

function pickMime() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await decodeCtx.decodeAudioData(arrayBuffer);
  } catch {
    await decodeCtx.close().catch(() => {});
    throw new RecorderError("DECODE_FAILED", "The recorded audio could not be processed.");
  }
  await decodeCtx.close().catch(() => {});

  const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_RATE));
  const offline = new OfflineAudioContext(1, frames, TARGET_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  const samples = rendered.getChannelData(0);
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) sumSquares += samples[i] * samples[i];
  const rms = Math.sqrt(sumSquares / samples.length);
  if (rms < 0.0015) {
    throw new RecorderError("NO_SIGNAL", "The microphone captured silence. Check that the correct mic is selected in Windows sound settings.");
  }
  return encodeWav(samples, TARGET_RATE);
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

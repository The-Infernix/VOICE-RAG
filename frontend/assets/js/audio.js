const TARGET_RATE = 16000;

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
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new RecorderError("UNSUPPORTED", "Audio recording is not supported in this browser.");
    }
    if (!window.MediaRecorder) {
      throw new RecorderError("UNSUPPORTED", "Audio recording is not supported in this browser.");
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
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

    this.stream = stream;
    this.chunks = [];

    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64;
      this.analyser.smoothingTimeConstant = 0.55;
      source.connect(this.analyser);
      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
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

    const stopped = new Promise((resolve) => {
      this.recorder.onstop = resolve;
    });
    if (this.recorder.state !== "inactive") this.recorder.stop();
    await stopped;

    this.teardown();

    const raw = new Blob(this.chunks, { type: this.recorder.mimeType || "audio/webm" });
    this.chunks = [];
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
    this.teardown();
  }

  teardown() {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
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
  return encodeWav(rendered.getChannelData(0), TARGET_RATE);
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

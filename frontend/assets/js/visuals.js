const ACCENT = { r: 99, g: 243, b: 210 };
const BLUE = { r: 122, g: 167, b: 255 };

function accent(alpha) {
  return `rgba(${ACCENT.r}, ${ACCENT.g}, ${ACCENT.b}, ${alpha})`;
}

export class VisualEngine {
  constructor(canvas, { reducedMotion = false } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.mode = "idle";
    this.hover = false;
    this.reducedMotion = reducedMotion;
    this.amplitudeProvider = null;
    this.spectrumProvider = null;
    this.foundCount = 0;
    this.running = false;
    this.raf = 0;
    this.size = 0;
    this.dpr = 1;
    this.particles = [];
    this.foundAt = 0;

    this.resize = this.resize.bind(this);
    this.loop = this.loop.bind(this);

    if ("ResizeObserver" in window) {
      this.observer = new ResizeObserver(this.resize);
      this.observer.observe(canvas.parentElement || canvas);
    } else {
      window.addEventListener("resize", this.resize);
    }
    this.resize();
  }

  setMode(mode) {
    if (mode === this.mode) return;
    const prev = this.mode;
    this.mode = mode;
    if (mode === "retrieving") {
      this.seedParticles();
    }
    if (mode === "found") {
      this.foundAt = performance.now();
      this.assignNodes();
    }
    if (prev === "idle" && mode !== "idle") this.start();
  }

  setHover(v) {
    this.hover = v;
  }

  setProviders({ amplitude, spectrum }) {
    if (amplitude) this.amplitudeProvider = amplitude;
    if (spectrum) this.spectrumProvider = spectrum;
  }

  seedParticles() {
    const n = 96;
    this.particles = [];
    for (let i = 0; i < n; i++) {
      this.particles.push({
        angle: Math.random() * Math.PI * 2,
        dist: 0.52 + Math.random() * 0.26,
        speed: 0.5 + Math.random() * 1.1,
        drift: (Math.random() - 0.5) * 0.9,
        size: 0.8 + Math.random() * 1.1,
        alpha: 0.18 + Math.random() * 0.45,
        blue: Math.random() < 0.12,
        node: -1,
      });
    }
  }

  assignNodes() {
    const n = Math.max(1, this.foundCount);
    const slots = [];
    for (let i = 0; i < n; i++) {
      slots.push((i / n) * Math.PI * 2 - Math.PI / 2);
    }
    this.particles.forEach((p, idx) => {
      p.node = idx % n;
      p.nodeAngle = slots[idx % n];
    });
  }

  resize() {
    const parent = this.canvas.parentElement;
    const rect = parent ? parent.getBoundingClientRect() : this.canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.size = Math.min(rect.width, rect.height);
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.cx = rect.width / 2;
    this.cy = rect.height / 2;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.raf = requestAnimationFrame(this.loop);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  destroy() {
    this.stop();
    if (this.observer) this.observer.disconnect();
    else window.removeEventListener("resize", this.resize);
  }

  loop(now) {
    if (!this.running) return;
    this.draw(now);
    this.raf = requestAnimationFrame(this.loop);
  }

  draw(now) {
    const ctx = this.ctx;
    const S = this.size;
    if (!S) return;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    const t = now * 0.001;
    const amp = this.mode === "listening" && this.amplitudeProvider ? this.amplitudeProvider() : 0;
    const rm = this.reducedMotion;

    const breathe = rm ? 0 : Math.sin(t * 1.15) * 0.028;
    const hoverLift = this.hover && this.mode === "idle" ? 0.05 : 0;
    const listenBoost = this.mode === "listening" ? amp * 0.16 : 0;
    const coreScale = 1 + breathe + hoverLift + listenBoost;

    let glowAlpha = 0.30;
    if (this.mode === "listening") glowAlpha = 0.42 + amp * 0.4;
    if (this.mode === "retrieving") glowAlpha = 0.34;
    if (this.mode === "found") glowAlpha = 0.46;

    const coreR = S * 0.135 * coreScale;

    const glow = ctx.createRadialGradient(this.cx, this.cy, coreR * 0.2, this.cx, this.cy, S * 0.42);
    glow.addColorStop(0, accent(glowAlpha * 0.5));
    glow.addColorStop(0.45, accent(glowAlpha * 0.12));
    glow.addColorStop(1, accent(0));
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, this.canvas.width / this.dpr, this.canvas.height / this.dpr);

    const spin = rm ? 0 : t * 0.12;

    ctx.save();
    ctx.translate(this.cx, this.cy);
    ctx.rotate(spin);
    ctx.strokeStyle = accent(0.34);
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 7]);
    ctx.beginPath();
    ctx.arc(0, 0, S * 0.455, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    ctx.setLineDash([]);
    ctx.strokeStyle = accent(0.10);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, S * 0.385, 0, Math.PI * 2);
    ctx.stroke();

    if (this.mode === "listening" && this.spectrumProvider) {
      const bins = this.spectrumProvider(56);
      const r0 = S * 0.295;
      const maxLen = S * 0.075;
      ctx.lineCap = "round";
      for (let i = 0; i < bins.length; i++) {
        const v = bins[i];
        if (v <= 0.02) continue;
        const ang = (i / bins.length) * Math.PI * 2 - Math.PI / 2;
        const len = maxLen * v;
        const x0 = this.cx + Math.cos(ang) * r0;
        const y0 = this.cy + Math.sin(ang) * r0;
        const x1 = this.cx + Math.cos(ang) * (r0 + len);
        const y1 = this.cy + Math.sin(ang) * (r0 + len);
        ctx.strokeStyle = accent(0.25 + v * 0.65);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }

    if (this.mode === "retrieving") {
      this.drawParticles(ctx, S, now, false, rm);
    } else if (this.mode === "found") {
      this.drawParticles(ctx, S, now, true, rm);
    }

    const coreGrad = ctx.createRadialGradient(
      this.cx - coreR * 0.3, this.cy - coreR * 0.35, coreR * 0.1,
      this.cx, this.cy, coreR
    );
    coreGrad.addColorStop(0, "#EAFFFA");
    coreGrad.addColorStop(0.35, accent(0.95));
    coreGrad.addColorStop(1, "rgba(23, 120, 100, 0.9)");
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, coreR, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = accent(0.5);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, coreR + S * 0.028 * coreScale, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  drawParticles(ctx, S, now, locked, rm) {
    const orbit = S * 0.225;
    const elapsed = now - this.foundAt;

    for (const p of this.particles) {
      let x, y, a;

      if (locked && p.node >= 0) {
        const settle = rm ? 1 : Math.min(1, elapsed / 900);
        const e = 1 - Math.pow(1 - settle, 3);
        const targetDist = S * 0.27;
        const d = p.dist * S + (targetDist - p.dist * S) * e;
        const ang = p.angle + (p.nodeAngle - p.angle) * e;
        x = this.cx + Math.cos(ang) * d;
        y = this.cy + Math.sin(ang) * d;
        a = 0.35 + 0.55 * e;
      } else {
        if (!rm) {
          p.dist += (orbit / S - p.dist) * 0.018 * p.speed;
          p.angle += (0.0016 + p.drift * 0.0012) * p.speed;
        }
        x = this.cx + Math.cos(p.angle) * p.dist * S;
        y = this.cy + Math.sin(p.angle) * p.dist * S;
        a = p.alpha;
      }

      ctx.fillStyle = p.blue
        ? `rgba(${BLUE.r}, ${BLUE.g}, ${BLUE.b}, ${a})`
        : accent(a);
      ctx.beginPath();
      ctx.arc(x, y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }

    if (locked && !rm && elapsed < 1400) {
      const k = elapsed / 1400;
      ctx.strokeStyle = accent(0.35 * (1 - k));
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, S * (0.24 + k * 0.3), 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

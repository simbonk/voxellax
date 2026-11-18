// Front-camera head tracking for off-axis parallax. Produces the viewer's eye
// position in centimetres relative to where they were when tracking started
// (or when "recenter" was pressed), plus an estimated distance.
import { createFaceLandmarker, eyePoints } from './face.js';

const IPD_CM = 6.3;
const CAM_HALF_FOV = (32 * Math.PI) / 180; // horizontal, typical front camera

export class HeadTracker {
  constructor() {
    this.video = null;
    this.stream = null;
    this.face = null;
    this.running = false;
    this.pos = null; // {x, y, z} smoothed, cm, screen-relative
    this.base = null;
    this.lastSeen = 0;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    const v = document.createElement('video');
    v.playsInline = true;
    v.muted = true;
    v.setAttribute('playsinline', '');
    v.srcObject = this.stream;
    await v.play();
    this.video = v;
    this.face = await createFaceLandmarker('VIDEO');
    this.running = true;
    this.loop();
  }

  loop() {
    if (!this.running) return;
    const v = this.video;
    const next = () => {
      if (!this.running) return;
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => this.loop());
      else requestAnimationFrame(() => this.loop());
    };
    if (v.readyState >= 2 && v.videoWidth) {
      try {
        const res = this.face.detectForVideo(v, performance.now());
        if (res.faceLandmarks && res.faceLandmarks.length) this.update(res.faceLandmarks[0]);
      } catch (e) {
        console.warn('head tracking frame failed', e);
      }
    }
    next();
  }

  update(lm) {
    const v = this.video;
    const W = v.videoWidth, H = v.videoHeight;
    const [a, b] = eyePoints(lm);
    const ipdPx = Math.hypot((a.x - b.x) * W, (a.y - b.y) * H);
    if (ipdPx < 2) return;
    const f = W / 2 / Math.tan(CAM_HALF_FOV);
    const z = (IPD_CM * f) / ipdPx;
    const mx = ((a.x + b.x) / 2) * W, my = ((a.y + b.y) / 2) * H;
    // Raw front-camera image: moving to your right moves you to the image's left.
    const raw = { x: (-(mx - W / 2) / f) * z, y: (-(my - H / 2) / f) * z, z };
    if (!this.base) this.base = { x: raw.x, y: raw.y };
    const target = { x: raw.x - this.base.x, y: raw.y - this.base.y, z };
    if (!this.pos) this.pos = target;
    else {
      const k = 0.45;
      this.pos = {
        x: this.pos.x + (target.x - this.pos.x) * k,
        y: this.pos.y + (target.y - this.pos.y) * k,
        z: this.pos.z + (target.z - this.pos.z) * k * 0.5,
      };
    }
    this.lastSeen = performance.now();
  }

  recenter() {
    this.base = null;
    this.pos = null;
  }

  get tracking() {
    return this.running && performance.now() - this.lastSeen < 800;
  }

  stop() {
    this.running = false;
    try { this.face?.close(); } catch {}
    this.stream?.getTracks().forEach((t) => t.stop());
    this.face = null;
    this.stream = null;
    this.video = null;
    this.pos = null;
    this.base = null;
  }
}

// Turns two synced handheld selfie clips into voxel frames.
//
// Per source frame: grab the frame from the <video> element, find the face with
// MediaPipe, run Depth Anything for relative inverse depth, then sample a coarse
// grid of coloured points. Absolute scale comes from the face itself (an average
// inter-pupil distance of 6.3 cm). The two cameras are lined up best-effort by
// fitting a similarity transform between the face landmarks seen in both views.
import { DepthEstimator } from './depth.js';
import { createFaceLandmarker, eyePoints, ALIGN_LANDMARKS } from './face.js';
import { alignSimilarity, applySim, similarityToMat4, identity, median, percentile } from './math.js';
import { DEPTH_SCALE } from './format.js';
import { SYNC_RATE } from './audiosync.js';

export const DETAIL = {
  low: { grid: 120, depth: 252 },
  medium: { grid: 168, depth: 336 },
  high: { grid: 224, depth: 448 },
};

const IPD_CM = 6.3;
const LONG_SIDE_HALF_FOV = (35 * Math.PI) / 180; // typical phone/tablet selfie camera
const DEFAULT_FACE_CM = 40;
const MAX_REL_DEPTH = 4; // background clamp, in multiples of the face distance

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function seekTo(video, t) {
  t = Math.max(0, Math.min(t, (video.duration || t) - 0.001));
  await new Promise((resolve) => {
    let to;
    const done = () => {
      clearTimeout(to);
      video.removeEventListener('seeked', done);
      resolve();
    };
    to = setTimeout(done, 4000);
    video.addEventListener('seeked', done);
    video.currentTime = t;
  });
  if (video.requestVideoFrameCallback) {
    await Promise.race([new Promise((r) => video.requestVideoFrameCallback(() => r())), sleep(80)]);
  }
}

const round14 = (x) => Math.max(14, Math.round(x / 14) * 14);

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function bilinear(map, mw, mh, u, v) {
  const x = Math.min(mw - 1, Math.max(0, u * mw - 0.5));
  const y = Math.min(mh - 1, Math.max(0, v * mh - 0.5));
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(mw - 1, x0 + 1), y1 = Math.min(mh - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  const a = map[y0 * mw + x0], b = map[y0 * mw + x1];
  const c = map[y1 * mw + x0], d = map[y1 * mw + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// Fill gaps (NaN) with the nearest valid value, then median-filter.
function fillAndSmooth(arr, win, fallback) {
  const n = arr.length;
  const out = new Float64Array(n);
  const valid = [];
  for (let i = 0; i < n; i++) if (Number.isFinite(arr[i])) valid.push(i);
  if (!valid.length) return out.fill(fallback);
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(arr[i])) { out[i] = arr[i]; continue; }
    let best = valid[0];
    for (const j of valid) if (Math.abs(j - i) < Math.abs(best - i)) best = j;
    out[i] = arr[best];
  }
  const sm = new Float64Array(n);
  const h = win >> 1;
  for (let i = 0; i < n; i++) {
    const w = [];
    for (let j = Math.max(0, i - h); j <= Math.min(n - 1, i + h); j++) w.push(out[j]);
    sm[i] = median(w);
  }
  return sm;
}

// Drop "flying pixels": points in a depth discontinuity that are neither the
// near surface nor clearly part of the far one.
function edgeFilter(rel, gw, gh) {
  const out = new Float32Array(rel);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x;
      const z = rel[i];
      if (!(z > 0)) continue;
      let zmin = z, zmax = z;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= gh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= gw) continue;
          const v = rel[yy * gw + xx];
          if (!(v > 0)) continue;
          if (v < zmin) zmin = v;
          if (v > zmax) zmax = v;
        }
      }
      if ((zmax - zmin) / z > 0.2 && z > zmin * 1.06) out[i] = 0;
    }
  }
  return out;
}

class ViewProcessor {
  constructor(name, video, t0, opts) {
    this.name = name;
    this.video = video;
    this.t0 = t0;
    const vw = video.videoWidth, vh = video.videoHeight;
    const portrait = vh >= vw;
    const long = portrait ? vh : vw, short = portrait ? vw : vh;
    const { grid, depth } = DETAIL[opts.detail];
    const gl = grid, gs = Math.max(8, Math.round((grid * short) / long));
    const dl = round14(depth), ds = round14((depth * short) / long);
    this.gw = portrait ? gs : gl;
    this.gh = portrait ? gl : gs;
    this.dw = portrait ? ds : dl;
    this.dh = portrait ? dl : ds;
    const tanLong = Math.tan(LONG_SIDE_HALF_FOV);
    this.tanX = portrait ? (tanLong * vw) / vh : tanLong;
    this.tanY = portrait ? tanLong : (tanLong * vh) / vw;
    const fs = 512 / long;
    this.fw = Math.round(vw * fs);
    this.fh = Math.round(vh * fs);
    this.depthCanvas = makeCanvas(this.dw, this.dh);
    this.gridCanvas = makeCanvas(this.gw, this.gh);
    this.faceCanvas = makeCanvas(this.fw, this.fh);
    const F = opts.frameCount;
    this.N = this.gw * this.gh;
    this.rel = new Float32Array(F * this.N);
    this.color = new Uint8Array(F * this.N * 3);
    this.zFaceRaw = new Float64Array(F).fill(NaN);
    this.lmUV = new Array(F).fill(null); // Float32Array [u,v,...] for ALIGN_LANDMARKS
    this.lmRel = new Array(F).fill(null);
    this.faceFrames = 0;
  }

  ctx(c) {
    return c.getContext('2d', { willReadFrequently: true });
  }

  async processFrame(f, t, depth, face) {
    await seekTo(this.video, this.t0 + t);
    const v = this.video;
    // Colour grid.
    const gctx = this.ctx(this.gridCanvas);
    gctx.drawImage(v, 0, 0, this.gw, this.gh);
    const gpx = gctx.getImageData(0, 0, this.gw, this.gh).data;
    const col = this.color.subarray(f * this.N * 3, (f + 1) * this.N * 3);
    for (let i = 0; i < this.N; i++) {
      col[i * 3] = gpx[i * 4];
      col[i * 3 + 1] = gpx[i * 4 + 1];
      col[i * 3 + 2] = gpx[i * 4 + 2];
    }
    // Face.
    let lm = null;
    if (face) {
      const fctx = this.ctx(this.faceCanvas);
      fctx.drawImage(v, 0, 0, this.fw, this.fh);
      try {
        const res = face.detect(this.faceCanvas);
        if (res.faceLandmarks && res.faceLandmarks.length) lm = res.faceLandmarks[0];
      } catch (e) {
        console.warn('face detect failed', e);
      }
    }
    // Depth.
    const dctx = this.ctx(this.depthCanvas);
    dctx.drawImage(v, 0, 0, this.dw, this.dh);
    const dimg = dctx.getImageData(0, 0, this.dw, this.dh);
    const dm = await depth.run(dimg);
    const sample = (u, vv) => bilinear(dm.data, dm.width, dm.height, u, vv);

    let dRef;
    if (lm) {
      this.faceFrames++;
      const ds = [];
      for (const idx of ALIGN_LANDMARKS) ds.push(sample(lm[idx].x, lm[idx].y));
      dRef = median(ds);
      const [e1, e2] = eyePoints(lm);
      const ipdPx = Math.hypot((e1.x - e2.x) * v.videoWidth, (e1.y - e2.y) * v.videoHeight);
      const fy = v.videoHeight / 2 / this.tanY;
      if (ipdPx > 2) this.zFaceRaw[f] = (IPD_CM * fy) / ipdPx;
      const uv = new Float32Array(ALIGN_LANDMARKS.length * 2);
      const rl = new Float32Array(ALIGN_LANDMARKS.length);
      ALIGN_LANDMARKS.forEach((idx, k) => {
        uv[k * 2] = lm[idx].x;
        uv[k * 2 + 1] = lm[idx].y;
        rl[k] = dRef / Math.max(ds[k], dRef * 0.1);
      });
      this.lmUV[f] = uv;
      this.lmRel[f] = rl;
    } else {
      dRef = percentile(dm.data, 0.9);
    }
    if (!(dRef > 0)) dRef = Math.max(1e-6, percentile(dm.data, 0.99));

    const rel = new Float32Array(this.N);
    for (let y = 0; y < this.gh; y++) {
      for (let x = 0; x < this.gw; x++) {
        const d = sample((x + 0.5) / this.gw, (y + 0.5) / this.gh);
        rel[y * this.gw + x] = Math.min(MAX_REL_DEPTH, dRef / Math.max(d, dRef * 0.05));
      }
    }
    this.rel.set(edgeFilter(rel, this.gw, this.gh), f * this.N);
    return { rel, gpx };
  }

  finish(F) {
    this.zFace = fillAndSmooth(this.zFaceRaw, 7, DEFAULT_FACE_CM);
    this.depth = new Uint16Array(F * this.N);
    for (let f = 0; f < F; f++) {
      const zf = this.zFace[f];
      for (let i = 0; i < this.N; i++) {
        const r = this.rel[f * this.N + i];
        this.depth[f * this.N + i] = r > 0 ? Math.min(65535, Math.round(r * zf * DEPTH_SCALE)) : 0;
      }
    }
    this.rel = null;
    // 3D landmarks in this camera's space (GL convention: -z forward).
    this.lm3d = this.lmUV.map((uv, f) => {
      if (!uv) return null;
      const rl = this.lmRel[f];
      const pts = [];
      for (let k = 0; k < rl.length; k++) {
        const z = rl[k] * this.zFace[f];
        pts.push([(uv[k * 2] * 2 - 1) * this.tanX * z, (1 - uv[k * 2 + 1] * 2) * this.tanY * z, -z]);
      }
      return pts;
    });
    const c = [0, 0, 0];
    let n = 0;
    for (const pts of this.lm3d) {
      if (!pts) continue;
      for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; n++; }
    }
    this.center = n ? c.map((x) => x / n) : [0, 0, -DEFAULT_FACE_CM];
    this.xform = new Float32Array(F * 16);
    for (let f = 0; f < F; f++) this.xform.set(identity(), f * 16);
  }

  result() {
    return {
      name: this.name,
      gw: this.gw,
      gh: this.gh,
      tanX: this.tanX,
      tanY: this.tanY,
      center: this.center,
      faceFrames: this.faceFrames,
      depth: this.depth,
      color: this.color,
      xform: this.xform,
    };
  }
}

// Align view B onto view A frame by frame, then smooth over time.
function alignViews(A, B, F) {
  const fits = new Array(F).fill(null);
  const ratios = [];
  for (let f = 0; f < F; f++) {
    const pa = A.lm3d[f], pb = B.lm3d[f];
    if (!pa || !pb) continue;
    let fit = alignSimilarity(pb, pa);
    // One round of outlier rejection (bad depth at some landmarks).
    const res = pb.map((p, i) => {
      const q = applySim(fit.s, fit.q, fit.t, p);
      return Math.hypot(q[0] - pa[i][0], q[1] - pa[i][1], q[2] - pa[i][2]);
    });
    const cut = percentile(res, 0.75);
    const w = res.map((r) => (r <= cut ? 1 : 0));
    fit = alignSimilarity(pb, pa, w);
    let se = 0, k = 0;
    pb.forEach((p, i) => {
      if (!w[i]) return;
      const q = applySim(fit.s, fit.q, fit.t, p);
      se += (q[0] - pa[i][0]) ** 2 + (q[1] - pa[i][1]) ** 2 + (q[2] - pa[i][2]) ** 2;
      k++;
    });
    const rms = Math.sqrt(se / Math.max(1, k));
    const ratio = rms / IPD_CM;
    if (fit.s < 0.6 || fit.s > 1.6 || ratio > 0.45) continue;
    fit.s = Math.min(1.4, Math.max(0.7, fit.s));
    fits[f] = fit;
    ratios.push(ratio);
  }
  const valid = fits.map((x, i) => (x ? i : -1)).filter((i) => i >= 0);
  const info = {
    framesWithBothFaces: A.lm3d.filter((p, f) => p && B.lm3d[f]).length,
    alignedFrames: valid.length,
    frames: F,
    medianResidualCm: ratios.length ? +(median(ratios) * IPD_CM).toFixed(2) : null,
  };
  if (!valid.length) return { ok: false, info };
  const nearest = (i) => valid.reduce((b, j) => (Math.abs(j - i) < Math.abs(b - i) ? j : b), valid[0]);
  const filled = fits.map((x, i) => x || fits[nearest(i)]);
  // Align quaternion signs, then moving-average smoothing.
  const q0 = filled[0].q;
  for (const fit of filled) {
    const dot = fit.q.reduce((s, x, i) => s + x * q0[i], 0);
    if (dot < 0) fit.q = fit.q.map((x) => -x);
  }
  const h = 3;
  for (let f = 0; f < F; f++) {
    let s = 0, n = 0;
    const q = [0, 0, 0, 0], t = [0, 0, 0];
    for (let j = Math.max(0, f - h); j <= Math.min(F - 1, f + h); j++) {
      const x = filled[j];
      s += x.s;
      for (let k = 0; k < 4; k++) q[k] += x.q[k];
      for (let k = 0; k < 3; k++) t[k] += x.t[k];
      n++;
    }
    const qn = Math.hypot(...q) || 1;
    B.xform.set(similarityToMat4(s / n, q.map((x) => x / qn), t.map((x) => x / n)), f * 16);
  }
  const ok = valid.length >= 0.5 * F && median(ratios) < 0.3;
  return { ok, info };
}

/**
 * opts: { sources: [{name, video, t0}], length, fps, detail, audio: {pcm16k, t0} | null,
 *         onProgress(info), previewCanvas, signal }
 */
export async function processClip(opts) {
  const { sources, fps, detail, onProgress = () => {}, signal } = opts;
  const F = Math.max(2, Math.round(opts.length * fps));

  onProgress({ stage: 'Loading face finder…' });
  let face = null;
  try {
    face = await createFaceLandmarker('IMAGE');
  } catch (e) {
    console.warn('Face landmarker unavailable', e);
  }

  const depth = new DepthEstimator();
  await depth.load((p) => {
    if (p.status === 'download') {
      onProgress({
        stage: `Downloading depth model (${(p.loaded / 1e6).toFixed(1)} / ${(p.total / 1e6).toFixed(1)} MB, ${p.device})…`,
        fraction: p.loaded / p.total,
      });
    } else {
      onProgress({ stage: `Loading depth model (${p.device}, ${p.dtype})…` });
    }
  });

  const views = sources.map((s) => new ViewProcessor(s.name, s.video, s.t0, { detail, frameCount: F }));
  const started = performance.now();
  const total = F * views.length;
  let done = 0;
  for (let f = 0; f < F; f++) {
    for (const vp of views) {
      if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
      const { rel, gpx } = await vp.processFrame(f, f / fps, depth, face);
      done++;
      const elapsed = (performance.now() - started) / 1000;
      const eta = (elapsed / done) * (total - done);
      onProgress({
        stage: `Frame ${f + 1} of ${F} (${vp.name}) on ${depth.device}`,
        fraction: done / total,
        eta,
      });
      if (opts.previewCanvas) drawPreview(opts.previewCanvas, vp, rel, gpx);
      // Let the UI breathe.
      await sleep(0);
    }
  }
  try { face?.close(); } catch {}

  onProgress({ stage: 'Building voxels…', fraction: 1 });
  for (const vp of views) vp.finish(F);
  let mergeOk = false;
  let mergeInfo = null;
  if (views.length === 2) {
    const r = alignViews(views[0], views[1], F);
    mergeOk = r.ok;
    mergeInfo = r.info;
  }

  let audio = null;
  if (opts.audio && opts.audio.pcm16k) {
    const a = opts.audio.pcm16k;
    const s0 = Math.max(0, Math.round(opts.audio.t0 * SYNC_RATE));
    const n = Math.min(a.length - s0, Math.round((F / fps) * SYNC_RATE));
    if (n > 0) {
      const pcm = new Int16Array(n);
      for (let i = 0; i < n; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(a[s0 + i] * 32767)));
      audio = { rate: SYNC_RATE, pcm };
    }
  }

  return {
    fps,
    frameCount: F,
    mergeOk,
    mergeInfo,
    views: views.map((v) => v.result()),
    audio,
  };
}

function drawPreview(canvas, vp, rel, gpx) {
  const w = vp.gw, h = vp.gh;
  canvas.width = w * 2;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w * 2, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const o1 = (y * w * 2 + x) * 4, o2 = (y * w * 2 + x + w) * 4;
      img.data[o1] = gpx[i * 4];
      img.data[o1 + 1] = gpx[i * 4 + 1];
      img.data[o1 + 2] = gpx[i * 4 + 2];
      img.data[o1 + 3] = 255;
      const g = rel[i] > 0 ? Math.max(0, Math.min(255, 255 - (rel[i] - 0.5) * 120)) : 0;
      img.data[o2] = g;
      img.data[o2 + 1] = g * 0.8;
      img.data[o2 + 2] = 255 - g * 0.6;
      img.data[o2 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
